"""CPU stage-4 acceptance: frozen features, tiny training check and validation prefixes."""
import os

# Must precede NumPy/Torch imports; this entry point never schedules GPU work.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from collections import Counter
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
from time import perf_counter
import sys
import sysconfig

import numpy as np
import torch
import yaml

from scaf.baselines.adapters import BaselineModels
from scaf.baselines.matching import Features, mutual_ratio_match
from scaf.baselines.prepare import verify_assets
from scaf.data.catalog import fingerprint, load_catalog, sequence_by_id
from scaf.data.degradation import Degradation, degrade
from scaf.data.sequence import RGBDSequence
from scaf.evaluation.backend import PnPConfig, estimate_motion
from scaf.evaluation.matching import match_metrics
from scaf.evaluation.runner import timed, verify_development_data, write_json_line
from scaf.evaluation.trajectory import SegmentedOdometry, error_statistics, write_tum
from scaf.models.diagnostic import diagnostic_loss, diagnostic_targets
from scaf.models.geometry import GeometryConfig, GeometryEnhancer, build_neighborhood
from scaf.runtime import environment, git_info, sha256, write_json


METHODS = ("superpoint", "featurebooster", "appearance", "geometry_unweighted", "geometry_reliable")
KEYS = {"version", "catalog", "baselines", "degradations", "protocol", "train_sequence",
        "validation_sequences", "variants", "max_keypoints", "frame_stride", "validation_frames",
        "train_pairs", "train_pair_spacing", "train_variant", "train_steps", "learning_rate",
        "temperature", "negative_radius_px", "seed", "degradation_seed", "timing_warmup_frames", "model"}


def load_settings(path):
    c = yaml.safe_load(Path(path).read_text())
    if not isinstance(c, dict) or set(c) != KEYS or c["version"] != "scaf-stage4-v1":
        raise ValueError("invalid stage-4 schema/version")
    for key in ("max_keypoints", "frame_stride", "validation_frames", "train_pairs", "train_pair_spacing",
                "train_steps", "seed", "degradation_seed", "timing_warmup_frames"):
        if type(c[key]) is not int or c[key] < (0 if key in {"seed", "degradation_seed", "timing_warmup_frames"} else 1):
            raise ValueError(f"invalid {key}")
    if c["max_keypoints"] != 512 or c["validation_frames"] < 2 or max(c["seed"], c["degradation_seed"]) >= 2**31:
        raise ValueError("stage-4 check fixes the 512-point budget and needs at least two frames")
    for key in ("learning_rate", "temperature", "negative_radius_px"):
        if type(c[key]) not in (int, float) or not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"invalid {key}")
    if c["negative_radius_px"] <= 3:
        raise ValueError("negative radius must exceed the positive radius")
    for key in ("validation_sequences", "variants"):
        if (not isinstance(c[key], list) or not c[key] or any(not isinstance(x, str) or not x or
                not x.replace("_", "").isalnum() for x in c[key]) or len(c[key]) != len(set(c[key]))):
            raise ValueError(f"invalid {key}")
    if (not isinstance(c["train_sequence"], str) or c["train_sequence"] in c["validation_sequences"] or
            not isinstance(c["train_variant"], str)):
        raise ValueError("invalid train/validation selection")
    cfg = GeometryConfig(**c["model"])
    if cfg.descriptor_dim != 256:
        raise ValueError("frozen SuperPoint descriptor dimension must be 256")
    return c


def validate_roles(config, data_root):
    catalog = load_catalog(config["catalog"], data_root)
    train = sequence_by_id(catalog, config["train_sequence"])
    validation = [sequence_by_id(catalog, s) for s in config["validation_sequences"]]
    if train["split"] != "train" or any(s["split"] != "val" for s in validation):
        raise ValueError("stage 4 only trains on train and evaluates on val; held-out test is refused")
    return catalog


def prepare_frame(sequence, index, variant, level, models, model_config, config, max_wh, folder):
    clean, load_ms = timed(sequence.load, index, max_wh)
    (rgb, depth), degrade_ms = timed(degrade, clean, level, config["degradation_seed"], variant)
    features, extract_ms = timed(models.extract, rgb)
    # Clone NumPy inference outputs into ordinary tensors, safe to save for backward.
    desc = torch.tensor(features.descriptors)[None]
    context, context_ms = timed(build_neighborhood, torch.tensor(features.points)[None],
                                torch.tensor(depth)[None], torch.tensor(clean.K, dtype=torch.float32)[None],
                                config=model_config)
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/f"{index:06d}.npz", points=features.points, descriptors=features.descriptors,
                            scores=features.scores, image_wh=features.image_wh, sequence_index=index,
                            timestamp=clean.timestamp, reliability=context.reliability.numpy()[0],
                            xyz=context.xyz.numpy()[0])
    return {"clean": clean, "depth": depth, "features": features, "desc": desc, "context": context,
            "index": index, "timing": {"load_ms": load_ms, "degrade_ms": degrade_ms, "extract_ms": extract_ms,
                                        "context_total_ms": context_ms, **context.timings_ms}}


def train_check(model, models, catalog, data_root, levels, protocol, config, output):
    sequence = RGBDSequence(sequence_by_id(catalog, config["train_sequence"]), data_root)
    max_wh = (protocol["image"]["max_width"], protocol["image"]["max_height"])
    selected = [(i*config["train_pair_spacing"], i*config["train_pair_spacing"]+config["frame_stride"])
                for i in range(config["train_pairs"])]
    if selected[-1][1] >= len(sequence):
        raise ValueError("training pair selection exceeds sequence")
    data, records = [], []
    for i, j in selected:
        frames = [prepare_frame(sequence, x, config["train_variant"], levels[config["train_variant"]], models,
                                model.config, config, max_wh, output/"train_features") for x in (i, j)]
        if any(f["clean"].T_world_camera is None for f in frames):
            records.append({"indices": [i, j], "usable": False, "reason": "reference_pose_missing"})
            continue
        targets = diagnostic_targets(frames[0]["clean"], frames[1]["clean"],
                                     frames[0]["features"].points, frames[1]["features"].points,
                                     config["negative_radius_px"])
        _, info = diagnostic_loss(frames[0]["desc"][0], frames[1]["desc"][0], targets, config["temperature"])
        records.append({"indices": [i, j], "identities": [f["clean"].identity for f in frames],
                        "timestamps": [f["clean"].timestamp for f in frames], **info})
        if info["anchors"]:
            data.append((frames, targets))
    write_json(output/"training_selection.json", records)
    if not data:
        raise ValueError("no geometrically certified positive pairs with trusted negatives")

    def objective(item):
        frames, targets = item
        a, b = [model(f["desc"], f["context"])[0] for f in frames]
        return diagnostic_loss(a, b, targets, config["temperature"])

    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    def mean_loss():
        with torch.no_grad():
            return float(np.mean([float(objective(item)[0]) for item in data]))
    loss_before = mean_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    model.train()
    with (output/"training.jsonl").open("x") as handle:
        for step in range(config["train_steps"]):
            # Fixed cyclic order; final checkpoint is predetermined, no validation selection.
            optimizer.zero_grad(set_to_none=True)
            loss, stats = objective(data[step % len(data)])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("missing/nonfinite gradients")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
            optimizer.step()
            write_json_line(handle, {"step": step, "pair": step % len(data), "loss": float(loss.detach()),
                                     "gradient_norm_before_clip": grad_norm, **stats})
            if (step+1) % len(data) == 0:
                print(json.dumps({"training_step": step+1, "loss": float(loss.detach())}), flush=True)
    model.eval()
    changes = {name: float((p.detach()-initial[name]).abs().max()) for name, p in model.named_parameters()}
    if not all(torch.isfinite(p).all() for p in model.parameters()) or not max(changes.values()) > 0:
        raise FloatingPointError("model did not make finite parameter updates")
    record = {"purpose": "tiny_overfit_debug_not_generalization_evidence", "initial_mean_loss": loss_before,
              "final_mean_loss": mean_loss(), "usable_pairs": len(data), "steps": config["train_steps"],
              "max_parameter_change": max(changes.values()), "parameter_changes": changes,
              "checkpoint_selection": "last_fixed_step_without_validation_selection",
              "all_gradients_finite": True, "targets": "clean_RGBD_and_reference_poses; model sees perturbed input only"}
    checkpoint = output/"geometry-final.pt"
    torch.save({"model": model.state_dict(), "model_config": asdict(model.config), "seed": config["seed"],
                "steps": config["train_steps"], "kind": "stage4_debug_not_formal_training"}, checkpoint)
    reloaded = GeometryEnhancer(model.config).eval()
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
    with torch.no_grad():
        sample = data[0][0][0]
        torch.testing.assert_close(model(sample["desc"], sample["context"]),
                                   reloaded(sample["desc"], sample["context"]), rtol=0, atol=0)
    record.update(checkpoint_sha256=sha256(checkpoint), checkpoint_reload_exact=True)
    write_json(output/"training_summary.json", record)
    return record


def summarize_times(rows, warmup):
    selected = rows[warmup:]
    return {"warmup_frames_excluded": min(warmup, len(rows)), "samples": len(selected), "unit": "ms",
            "performance_benchmark": False, "components": {
                key: {"p50": float(np.median([r[key] for r in selected])) if selected else None,
                      "p95": float(np.percentile([r[key] for r in selected], 95)) if selected else None}
                for key in rows[0]} if rows else {}}


@torch.no_grad()
def evaluate(model, models, spec, variant, level, data_root, config, protocol, output):
    output.mkdir(parents=True, exist_ok=False)
    sequence = RGBDSequence(spec, data_root)
    indices = list(range(0, len(sequence), config["frame_stride"]))[:config["validation_frames"]]
    if len(indices) < 2:
        raise ValueError("validation requires at least two selected frames")
    settings = protocol["backend"]
    pnp = PnPConfig(reprojection_px=settings["ransac_reprojection_px"], iterations=settings["ransac_iterations"],
                    confidence=settings["ransac_confidence"], min_correspondences=settings["min_correspondences"],
                    min_inliers=settings["min_inliers"], seed=config["seed"])
    trackers = {m: SegmentedOdometry() for m in METHODS}
    pairs_by_method, times_by_method = {m: [] for m in METHODS}, {m: [] for m in METHODS}
    timestamps, references, frame_records = [], [], []
    previous = None
    max_wh = (protocol["image"]["max_width"], protocol["image"]["max_height"])
    write_json(output/"selection.json", {"sequence": spec, "variant": variant, "degradation": asdict(level),
        "indices": indices, "available_frames": len(sequence), "backend_depth": "perturbed_input",
        "scope": "validation_prefix_debug", "interventions": "same_trained_checkpoint_branch_switches_not_retrained_ablations"})
    with (output/"frames.jsonl").open("x") as frame_log, (output/"pairs.jsonl").open("x") as pair_log:
        for ordinal, index in enumerate(indices):
            f = prepare_frame(sequence, index, variant, level, models, model.config, config, max_wh, output/"features")
            clean, features, context = f["clean"], f["features"], f["context"]
            timestamps.append(clean.timestamp)
            references.append(clean.T_world_camera)
            enhanced, timings = {}, {}
            for method in METHODS:
                if method == "superpoint":
                    result, network_ms, info = features, 0., None
                elif method == "featurebooster":
                    result, network_ms = timed(models.enhance, method, features)
                    info = None
                else:
                    (desc, info), network_ms = timed(model, f["desc"], context, geometry=method != "appearance",
                                                     reliability=method != "geometry_unweighted", return_info=True)
                    result = replace(features, descriptors=desc[0].numpy())
                    if method == "appearance":
                        appearance_desc = result.descriptors
                    if method == "geometry_reliable":
                        f["gate"] = info["geometry_gate"][0].numpy()
                    folder = output/method
                    folder.mkdir(exist_ok=True)
                    np.savez_compressed(folder/f"{ordinal:06d}.npz", descriptors=result.descriptors,
                                        sequence_index=index, timestamp=clean.timestamp)
                enhanced[method] = result
                # Appearance-only timing conservatively includes the full shared context construction.
                geometry_ms = f["timing"]["context_total_ms"] if method not in METHODS[:2] else 0.
                timing = {**{k: v for k, v in f["timing"].items() if k in ("load_ms", "degrade_ms", "extract_ms")},
                          "depth_prepare_ms": f["timing"]["depth_prepare_ms"] if geometry_ms else 0.,
                          "neighbor_search_ms": f["timing"]["neighbor_search_ms"] if geometry_ms else 0.,
                          "context_total_ms": geometry_ms, "network_ms": network_ms,
                          "aggregation_network_ms": info["aggregation_network_ms"] if info is not None else 0.,
                          "enhancement_total_ms": geometry_ms+network_ms}
                timing["frontend_extract_enhance_ms"] = timing["extract_ms"]+timing["enhancement_total_ms"]
                times_by_method[method].append(timing)
                timings[method] = timing
                if previous is None:
                    continue
                (pairs, scores), match_ms = timed(mutual_ratio_match, previous["enhanced"][method].descriptors,
                                                  result.descriptors, protocol["matching"]["ratio"])
                motion, backend_ms = timed(estimate_motion, previous["features"].points, features.points, pairs,
                                           previous["depth"], previous["clean"].K, clean.K, pnp)
                quality, labels = match_metrics(previous["clean"], clean, previous["features"].points,
                                                features.points, pairs, protocol["matching"]["geometric_label_reprojection_px"])
                segment = trackers[method].append(ordinal-1, ordinal, motion["T_camera1_camera0"])
                row = {"method": method, "frame0": ordinal-1, "frame1": ordinal, "indices": [previous["index"], index],
                       "timestamp0": previous["clean"].timestamp, "timestamp1": clean.timestamp, "segment": segment,
                       "quality": quality, "backend": {k: v for k, v in motion.items()
                           if k not in {"T_camera1_camera0", "inlier_match_indices"}},
                       "match_ms": match_ms, "backend_ms": backend_ms,
                       "pipeline_ms": timing["frontend_extract_enhance_ms"]+match_ms+backend_ms}
                write_json_line(pair_log, row)
                pairs_by_method[method].append(row)
                folder = output/method/"matches"
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder/f"{ordinal-1:06d}-{ordinal:06d}.npz", pairs=pairs, scores=scores,
                                    inlier_match_indices=motion["inlier_match_indices"],
                                    T_camera1_camera0=motion["T_camera1_camera0"] if motion["success"] else np.empty((0, 4)), **labels)
            q = context.reliability[0].numpy()
            frame_row = {"ordinal": ordinal, "index": index, "identity": clean.identity, "timestamp": clean.timestamp,
                "keypoints": len(q), "positive_reliability_fraction": float(np.mean(q > 0)) if len(q) else 0.,
                "mean_reliability": float(q.mean()) if len(q) else 0., "mean_geometry_gate": float(f["gate"].mean()) if len(q) else 0.,
                "reliable_vs_appearance_l2_mean": float(np.linalg.norm(enhanced["geometry_reliable"].descriptors-appearance_desc, axis=1).mean()) if len(q) else 0.,
                "context_storage": context.storage, "timings": timings}
            write_json_line(frame_log, frame_row)
            frame_records.append(frame_row)
            f["enhanced"] = enhanced
            previous = f
            if (ordinal+1) % 8 == 0 or ordinal == len(indices)-1:
                print(json.dumps({"sequence": spec["id"], "variant": variant, "frames": ordinal+1}), flush=True)
    write_tum(output/"groundtruth.txt", [(i, p) for i, p in enumerate(references) if p is not None], timestamps)
    summaries = []
    for method in METHODS:
        tracker, rows = trackers[method], pairs_by_method[method]
        for i, segment in enumerate(tracker.segments):
            write_tum(output/method/f"trajectory-{i:03d}.txt", segment, timestamps)
        quality = {key: sum(r["quality"][key] for r in rows) for key in ("predicted", "evaluable", "correct", "unevaluable")}
        quality["precision"] = quality["correct"]/quality["evaluable"] if quality["evaluable"] else None
        quality["correct_per_pair"] = error_statistics([r["quality"]["correct"] for r in rows])
        quality["predicted_per_pair"] = error_statistics([r["quality"]["predicted"] for r in rows])
        failures = Counter(r["backend"]["reason"] for r in rows if not r["backend"]["success"])
        summary = {"sequence": spec["id"], "variant": variant, "method": method, "frames": len(indices),
                   "quality": quality, "trajectory": tracker.summary(timestamps, references), "backend_failures": dict(failures),
                   "timing": summarize_times(times_by_method[method], config["timing_warmup_frames"]),
                   "match_ms": error_statistics([r["match_ms"] for r in rows]),
                   "backend_ms": error_statistics([r["backend_ms"] for r in rows]),
                   "pipeline_ms": error_statistics([r["pipeline_ms"] for r in rows]),
                   "mean_reliability": float(np.mean([r["mean_reliability"] for r in frame_records])),
                   "mean_geometry_gate": float(np.mean([r["mean_geometry_gate"] for r in frame_records])),
                   "max_context_tensor_bytes": max(r["context_storage"]["context_tensor_bytes"] for r in frame_records)}
        write_json(output/method/"summary.json", summary)
        summaries.append(summary)
    return summaries


def write_csv(path, results):
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "variant", "method", "precision", "correct_per_pair", "matches_per_pair", "tracking_success",
                         "coverage", "failed", "ate_rmse_m", "rpe_translation_rmse_m", "enhancement_p50_ms", "search_p50_ms"])
        for r in results:
            t, q, c = r["trajectory"], r["quality"], r["timing"]["components"]
            writer.writerow([r["sequence"], r["variant"], r["method"], q["precision"], q["correct_per_pair"]["mean"],
                             q["predicted_per_pair"]["mean"], t["tracking_success_rate"], t["frame_coverage"], t["failed_transitions"],
                             t["full_trajectory_ate_m"]["rmse"] if t["full_trajectory_ate_m"] else None,
                             t["rpe_translation_m"]["rmse"], c["enhancement_total_ms"]["p50"], c["neighbor_search_ms"]["p50"]])


def run(config, data_root, audit_manifest, asset_manifest, output):
    root = Path(__file__).resolve().parents[2]
    if root != Path.cwd().resolve():
        raise ValueError("run from checkout root with PYTHONPATH=$PWD/src")
    output.mkdir(parents=True, exist_ok=False)
    record = {"kind": "stage4_tiny_training_validation_debug", "status": "running", "source": git_info(root),
              "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root), "invocation": sys.argv,
              "config": config, "config_sha256": fingerprint(config), "environment": environment(),
              "device": "cpu", "threads": 1, "data_root": str(data_root), "audit_manifest": str(audit_manifest),
              "asset_manifest": str(asset_manifest), "checkpoint_selection": "last_fixed_step", "test_used": False,
              "interventions": "one_checkpoint; branch switches are not independently trained ablations"}
    start = perf_counter()
    write_json(output/"manifest.json", record)
    try:
        if record["source"]["dirty"]:
            raise ValueError("stage-4 acceptance requires a clean committed checkout")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        dependencies = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions(path=[sysconfig.get_paths()["purelib"]]))
        (output/"dependencies.txt").write_text("\n".join(dependencies)+"\n")
        validate_roles(config, data_root)  # Refuse split misuse before auditing/reading any images.
        sequences = [config["train_sequence"], *config["validation_sequences"]]
        catalog, data_record = verify_development_data({**config, "sequences": sequences}, data_root, audit_manifest)
        assets = verify_assets(asset_manifest, yaml.safe_load(Path(config["baselines"]).read_text()))
        protocol = yaml.safe_load(Path(config["protocol"]).read_text())
        if protocol["version"] != 2 or protocol["backend"].get("solver") != "EPNP_RANSAC_LM":
            raise ValueError("frozen protocol v2 required")
        degradation_config = yaml.safe_load(Path(config["degradations"]).read_text())
        levels = {v: Degradation(**degradation_config["levels"][v]) for v in set(config["variants"]+[config["train_variant"]])}
        models = BaselineModels(assets, config["max_keypoints"], "cpu", ["superpoint", "featurebooster"])
        frozen_state = {k: v.detach().clone() for k, v in models.extractor.state_dict().items()}
        model = GeometryEnhancer(GeometryConfig(**config["model"]))
        record.update(catalog=catalog, data=data_record, assets=assets, asset_manifest_sha256=sha256(asset_manifest),
                      protocol=protocol, degradations=degradation_config, parameter_counts=models.parameter_counts(),
                      enhancer_parameters=sum(p.numel() for p in model.parameters()),
                      enhancer_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                      cpu={"logical_cores": os.cpu_count(), "load_average_start": list(os.getloadavg()),
                           "precision": "float32_module_float64_pose_geometry"})
        write_json(output/"manifest.json", record)
        record["training"] = train_check(model, models, catalog, data_root, levels, protocol, config, output)
        results = []
        for identifier in config["validation_sequences"]:
            for variant in config["variants"]:
                results.extend(evaluate(model, models, sequence_by_id(catalog, identifier), variant, levels[variant],
                                        data_root, config, protocol, output/identifier/variant))
                write_json(output/"summary.json", {"status": "partial", "results": results})
        if (any(p.requires_grad or p.grad is not None for p in models.extractor.parameters()) or
                any(not torch.equal(v, frozen_state[k]) for k, v in models.extractor.state_dict().items())):
            raise AssertionError("frozen SuperPoint parameters changed")
        record.update(superpoint_unchanged=True, cuda_initialized=torch.cuda.is_initialized())
        write_csv(output/"summary.csv", results)
        write_json(output/"summary.json", {"status": "completed", "results": results})
        record.update(status="completed", summary_sha256=sha256(output/"summary.json"), summary_csv_sha256=sha256(output/"summary.csv"))
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_s=perf_counter()-start,
                      load_average_end=list(os.getloadavg()))
        write_json(output/"manifest.json", record)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage4.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(run(load_settings(args.config), args.data_root.resolve(), args.audit_manifest.resolve(),
                  args.asset_manifest.resolve(), args.output.resolve()))
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
