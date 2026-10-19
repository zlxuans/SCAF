"""Streaming development benchmark with shared extraction and complete failure logs."""
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import sys
import sysconfig
from time import perf_counter

import numpy as np
import torch
import yaml

from scaf.baselines.adapters import BaselineModels
from scaf.baselines.prepare import verify_assets
from scaf.data.catalog import confined, fingerprint, load_catalog, sequence_by_id
from scaf.data.degradation import Degradation, degrade
from scaf.data.sequence import RGBDSequence
from scaf.runtime import environment, git_info, sha256, write_json
from .backend import PnPConfig, estimate_motion
from .matching import match_metrics
from .trajectory import SegmentedOdometry, error_statistics, write_tum


CONFIG_KEYS = {"version", "catalog", "baselines", "degradations", "protocol", "sequences", "variants",
               "methods", "max_keypoints", "frame_stride", "max_frames", "seed", "degradation_seed",
               "threads", "device", "timing_warmup_pairs", "timing_min_pairs", "performance_benchmark"}


def load_settings(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS or config["version"] != "scaf-stage3-v1":
        raise ValueError("invalid stage-3 configuration schema/version")
    if config["device"] != "cpu" or config["threads"] != 1 or config["performance_benchmark"] is not False:
        raise ValueError("stage-3 acceptance uses one CPU thread and is not a performance benchmark")
    for key in ("max_keypoints", "frame_stride", "max_frames", "seed", "degradation_seed",
                "timing_warmup_pairs", "timing_min_pairs"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"invalid integer: {key}")
    if (min(config["max_keypoints"], config["frame_stride"], config["timing_min_pairs"]) < 1 or
            config["max_frames"] == 1 or max(config["seed"], config["degradation_seed"]) >= 2**31):
        raise ValueError("invalid feature budget, frame count, stride, timing count or seed")
    for key in ("sequences", "variants", "methods"):
        if (not isinstance(config[key], list) or not config[key] or
                len(config[key]) != len(set(config[key])) or
                any(not isinstance(s, str) or not s or not s.replace("_", "").isalnum() for s in config[key])):
            raise ValueError(f"invalid or duplicate {key}")
    if set(config["methods"])-{"superpoint", "featurebooster", "lightglue"}:
        raise ValueError("unknown baseline method")
    return config


def verify_development_data(config, data_root, audit_manifest):
    catalog = load_catalog(config["catalog"], data_root)
    selected = [sequence_by_id(catalog, identifier) for identifier in config["sequences"]]
    if any(spec["split"] == "test" for spec in selected):
        raise ValueError("stage 3 refuses held-out test feature, matching and trajectory evaluation")
    manifest = json.loads(audit_manifest.read_text())
    if manifest.get("status") != "completed" or manifest["catalog_sha256"] != fingerprint(catalog):
        raise ValueError("completed data audit of the same catalog required")
    summary_path = audit_manifest.parent/"summary.json"
    files_path = audit_manifest.parent/"files.jsonl"
    summary = json.loads(summary_path.read_text())
    if (sha256(summary_path) != manifest["summary_sha256"] or
            sha256(files_path) != summary["files_sha256"]):
        raise ValueError("data audit summary/index changed")
    counts = Counter()
    for line in files_path.read_text().splitlines():
        item = json.loads(line)
        if item["sequence"] in config["sequences"]:
            path = confined(data_root, item["path"])
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError(f"dataset file changed: {item['path']}")
            counts[item["sequence"]] += 1
    for spec in selected:
        audited = next(s for s in summary["sequences"] if s["id"] == spec["id"])
        if counts[spec["id"]] != audited["files"] or counts[spec["id"]] == 0:
            raise ValueError("data audit does not cover selected sequence")
    return catalog, {"manifest_sha256": sha256(audit_manifest), "files_sha256": summary["files_sha256"],
                     "catalog_sha256": fingerprint(catalog), "verified_sequence_files": dict(counts)}


def timed(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, (perf_counter()-start)*1000.


def timing_summary(rows, warmup, minimum):
    selected = rows[warmup:]
    names = ("extract_ms", "enhance_ms", "optical_flow_ms", "match_ms", "backend_ms",
             "frontend_ms", "pipeline_compute_ms", "load_ms", "degrade_ms", "input_plus_compute_ms")
    output = {name: {"count": len(selected), "p50": float(np.median([r[name] for r in selected])) if selected else None,
                     "p95": float(np.percentile([r[name] for r in selected], 95)) if selected else None}
              for name in names}
    return {"unit": "ms", "performance_benchmark": False, "warmup_transitions_excluded": min(warmup, len(rows)),
            "timed_unique_transitions": len(selected), "minimum_requested": minimum,
            "sample_count_sufficient": len(selected) >= minimum, "components": output,
            "limitation": "CPU development wall times on a shared host; no real-time claim"}


def write_json_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
    handle.flush()


def evaluate_sequence(spec, variant, level, models, config, protocol, data_root, output):
    output.mkdir(parents=True, exist_ok=False)
    shared = output/"features"
    shared.mkdir()
    sequence = RGBDSequence(spec, data_root)
    indices = list(range(0, len(sequence), config["frame_stride"]))
    if config["max_frames"]:
        indices = indices[:config["max_frames"]]
    if len(indices) < 2:
        raise ValueError("at least two selected frames required")
    settings = protocol["backend"]
    pnp = PnPConfig(reprojection_px=settings["ransac_reprojection_px"],
                    iterations=settings["ransac_iterations"], confidence=settings["ransac_confidence"],
                    min_correspondences=settings["min_correspondences"], min_inliers=settings["min_inliers"],
                    seed=config["seed"])
    trackers = {method: SegmentedOdometry() for method in config["methods"]}
    rows = {method: [] for method in config["methods"]}
    handles, timestamps, references, frame_records = {}, [], [], []
    write_json(output/"selection.json", {"sequence": spec, "variant": variant, "degradation": asdict(level),
               "selected_rgbd_indices": indices, "available_rgbd_frames": len(sequence),
               "all_stride_selected_frames_used": indices[-1] == list(range(0, len(sequence), config["frame_stride"]))[-1],
               "scope": "development_acceptance_not_final_test", "backend_depth": "degraded_input"})
    for method in config["methods"]:
        folder = output/method
        (folder/"matches").mkdir(parents=True)
        if method == "featurebooster":
            (folder/"features").mkdir()
        handles[method] = (folder/"pairs.jsonl").open("x")
    previous_frame, previous_features, previous_enhanced, previous_depth = None, None, {}, None
    try:
        for selected, original_index in enumerate(indices):
            clean, load_ms = timed(sequence.load, original_index,
                                   (protocol["image"]["max_width"], protocol["image"]["max_height"]))
            (rgb, depth), degrade_ms = timed(degrade, clean, level, config["degradation_seed"], variant)
            features, extract_ms = timed(models.extract, rgb)
            np.savez_compressed(shared/f"{selected:06d}.npz", points=features.points, scores=features.scores,
                                descriptors=features.descriptors, image_wh=features.image_wh,
                                timestamp=clean.timestamp, sequence_index=original_index)
            timestamps.append(clean.timestamp)
            references.append(clean.T_world_camera)
            frame_records.append({"selected_frame": selected, "sequence_index": original_index,
                                  "timestamp": clean.timestamp, "identity": clean.identity,
                                  "keypoints": len(features.points), "reference_available": clean.T_world_camera is not None})
            enhanced = {}
            for method in config["methods"]:
                if method == "featurebooster":
                    current, enhance_ms = timed(models.enhance, method, features)
                    np.savez_compressed(output/method/"features"/f"{selected:06d}.npz",
                                        descriptors=current.descriptors, shared_feature=f"../../features/{selected:06d}.npz")
                else:
                    current, enhance_ms = features, 0.
                enhanced[method] = current
                if selected == 0:
                    continue
                (pairs, scores), match_ms = timed(models.match, method, previous_enhanced[method], current,
                                                  protocol["matching"]["ratio"])
                motion, backend_ms = timed(estimate_motion, previous_features.points, features.points, pairs,
                                           previous_depth, previous_frame.K, clean.K, pnp)
                quality, labels = match_metrics(previous_frame, clean, previous_features.points,
                                                features.points, pairs,
                                                protocol["matching"]["geometric_label_reprojection_px"])
                segment = trackers[method].append(selected-1, selected, motion["T_camera1_camera0"])
                metrics = {key: value for key, value in motion.items()
                           if key not in {"T_camera1_camera0", "inlier_match_indices"}}
                frontend_ms = extract_ms+enhance_ms+match_ms
                row = {"frame0": selected-1, "frame1": selected, "timestamp0": previous_frame.timestamp,
                       "timestamp1": clean.timestamp, "interval_s": clean.timestamp-previous_frame.timestamp,
                       "segment": segment, "match_quality": quality, "backend": metrics,
                       "timing": {"extract_ms": extract_ms, "enhance_ms": enhance_ms,
                                  "optical_flow_ms": 0., "match_ms": match_ms, "backend_ms": backend_ms,
                                  "frontend_ms": frontend_ms, "pipeline_compute_ms": frontend_ms+backend_ms,
                                  "load_ms": load_ms, "degrade_ms": degrade_ms,
                                  "input_plus_compute_ms": load_ms+degrade_ms+frontend_ms+backend_ms}}
                write_json_line(handles[method], row)
                rows[method].append(row)
                np.savez_compressed(output/method/"matches"/f"{selected-1:06d}-{selected:06d}.npz",
                                    pairs=pairs, scores=scores, inlier_match_indices=motion["inlier_match_indices"],
                                    T_camera1_camera0=motion["T_camera1_camera0"] if motion["success"] else np.empty((0, 4)),
                                    **labels)
            previous_frame, previous_features, previous_enhanced, previous_depth = clean, features, enhanced, depth
            if selected % 20 == 0 or selected == len(indices)-1:
                print(json.dumps({"sequence": spec["id"], "variant": variant,
                                  "selected_frames_processed": selected+1, "total": len(indices)}), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    write_json(output/"frames.json", frame_records)
    write_tum(output/"groundtruth.txt", [(i, T) for i, T in enumerate(references) if T is not None], timestamps)
    summaries = []
    for method, tracker in trackers.items():
        quality_rows = [r["match_quality"] for r in rows[method]]
        quality = {key: sum(r[key] for r in quality_rows) for key in ("predicted", "evaluable", "correct", "unevaluable")}
        quality["precision"] = quality["correct"]/quality["evaluable"] if quality["evaluable"] else None
        reasons = Counter()
        failures = Counter()
        for row in rows[method]:
            reasons.update(row["match_quality"]["reasons"])
            if not row["backend"]["success"]:
                failures[row["backend"]["reason"]] += 1
        quality["reasons"] = dict(reasons)
        quality["predicted_per_pair"] = error_statistics([r["predicted"] for r in quality_rows])
        quality["correct_per_pair"] = error_statistics([r["correct"] for r in quality_rows])
        for number, segment in enumerate(tracker.segments):
            write_tum(output/method/f"trajectory-segment-{number:03d}.txt", segment, timestamps)
        pose_entries = [(s, i, T) for s, segment in enumerate(tracker.segments) for i, T in segment]
        np.savez_compressed(output/method/"trajectory.npz",
                            segment=np.array([s for s, _, _ in pose_entries], dtype=np.int64),
                            selected_frame=np.array([i for _, i, _ in pose_entries], dtype=np.int64),
                            timestamp=np.array([timestamps[i] for _, i, _ in pose_entries]),
                            T_world_camera=np.array([T for _, _, T in pose_entries]).reshape(-1, 4, 4))
        summary = {"sequence": spec["id"], "split": spec["split"], "variant": variant, "method": method,
                   "match_quality": quality, "trajectory": tracker.summary(timestamps, references),
                   "backend_failures": dict(failures), "pnp_config": asdict(pnp),
                   "timing": timing_summary([r["timing"] for r in rows[method]],
                                            config["timing_warmup_pairs"], config["timing_min_pairs"])}
        write_json(output/method/"summary.json", summary)
        summaries.append(summary)
    return summaries


def write_summary_csv(path, summaries):
    names = ["sequence", "variant", "method", "match_precision", "correct_matches_per_pair",
             "predicted_matches_per_pair", "full_ate_rmse_m", "rpe_translation_rmse_m", "rpe_rotation_rmse_deg",
             "frame_coverage", "tracking_success_rate", "failed_transitions", "segments",
             "longest_segment_frames", "frontend_p50_ms", "frontend_p95_ms", "pipeline_p50_ms"]
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in summaries:
            t, q, timing = row["trajectory"], row["match_quality"], row["timing"]["components"]
            writer.writerow({"sequence": row["sequence"], "variant": row["variant"], "method": row["method"],
                             "match_precision": q["precision"], "correct_matches_per_pair": q["correct_per_pair"]["mean"],
                             "predicted_matches_per_pair": q["predicted_per_pair"]["mean"],
                             "full_ate_rmse_m": t["full_trajectory_ate_m"]["rmse"] if t["full_trajectory_ate_m"] else None,
                             "rpe_translation_rmse_m": t["rpe_translation_m"]["rmse"],
                             "rpe_rotation_rmse_deg": t["rpe_rotation_deg"]["rmse"],
                             "frame_coverage": t["frame_coverage"], "tracking_success_rate": t["tracking_success_rate"],
                             "failed_transitions": t["failed_transitions"], "segments": len(t["segments"]),
                             "longest_segment_frames": t["longest_segment_frames"],
                             "frontend_p50_ms": timing["frontend_ms"]["p50"],
                             "frontend_p95_ms": timing["frontend_ms"]["p95"],
                             "pipeline_p50_ms": timing["pipeline_compute_ms"]["p50"]})


def run(config, data_root, audit_manifest, asset_manifest, output):
    source = Path(__file__).resolve().parents[3]
    if source != Path.cwd().resolve():
        raise ValueError("run from the checkout root with PYTHONPATH=$PWD/src")
    output.mkdir(parents=True, exist_ok=False)
    record = {"kind": "stage3_development_baselines_not_final_test", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "source": git_info(source),
              "source_root": str(source), "invocation": sys.argv, "config": config,
              "config_fingerprint": fingerprint(config), "environment": environment(),
              "data_root": str(data_root), "audit_manifest": str(audit_manifest),
              "asset_manifest": str(asset_manifest), "timing_policy": "shared_extraction_charged_to_each_method",
              "feature_cache": "two_frames_in_memory; NPZ exports excluded from compute timing"}
    cpuinfo = Path("/proc/cpuinfo")
    record["cpu"] = {"model": next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                                   if line.startswith("model name")), "unknown") if cpuinfo.exists() else "unknown",
                     "logical_cores": os.cpu_count(), "load_average_start": list(os.getloadavg()),
                     "threads": config["threads"], "precision": "float32_models_float64_geometry",
                     "cuda_peak_memory": None}
    write_json(output/"manifest.json", record)
    start = perf_counter()
    try:
        if record["source"]["dirty"]:
            raise ValueError("baseline acceptance requires a clean committed checkout")
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        dependencies = sorted(f"{d.metadata['Name']}=={d.version}" for d in
                              distributions(path=[sysconfig.get_paths()["purelib"]]))
        (output/"dependencies.txt").write_text("\n".join(dependencies)+"\n")
        catalog, data_record = verify_development_data(config, data_root, audit_manifest)
        baselines = yaml.safe_load(Path(config["baselines"]).read_text())
        assets = verify_assets(asset_manifest, baselines)
        protocol = yaml.safe_load(Path(config["protocol"]).read_text())
        if protocol["backend"].get("solver") != "EPNP_RANSAC_LM" or protocol["version"] != 2:
            raise ValueError("stage 3 requires protocol v2 fixed EPNP/RANSAC/LM backend")
        degradation_config = yaml.safe_load(Path(config["degradations"]).read_text())
        levels = {name: Degradation(**degradation_config["levels"][name]) for name in config["variants"]}
        record.update(data=data_record, assets=assets, asset_manifest_sha256=sha256(asset_manifest),
                      protocol=protocol, degradations=degradation_config, catalog=catalog)
        write_json(output/"manifest.json", record)
        models = BaselineModels(assets, config["max_keypoints"], config["device"], config["methods"])
        record["parameter_counts"] = models.parameter_counts()
        record["featurebooster_config"] = getattr(models, "booster_config", None)
        record["lightglue_weight_compatibility"] = getattr(models, "lightglue_weight_compatibility", None)
        record["initialization_and_verification_s"] = perf_counter()-start
        write_json(output/"manifest.json", record)
        summaries = []
        for sequence in config["sequences"]:
            spec = sequence_by_id(catalog, sequence)
            for variant, level in levels.items():
                summaries.extend(evaluate_sequence(spec, variant, level, models, config, protocol,
                                                    data_root, output/sequence/variant))
                write_json(output/"summary.json", {"status": "partial", "results": summaries})
        write_summary_csv(output/"summary.csv", summaries)
        write_json(output/"summary.json", {"status": "completed", "results": summaries})
        record.update(status="completed", summary_sha256=sha256(output/"summary.json"),
                      summary_csv_sha256=sha256(output/"summary.csv"), cuda_initialized=torch.cuda.is_initialized())
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_s=perf_counter()-start,
                      load_average_end=list(os.getloadavg()))
        write_json(output/"manifest.json", record)
    return output
