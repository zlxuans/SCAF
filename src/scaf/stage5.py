"""CPU acceptance for temporal weighting and filtering; fixed tiny 2x2 experiment."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
from time import perf_counter
import sys
import sysconfig

import cv2
import numpy as np
import torch
import yaml

from scaf.baselines.adapters import BaselineModels
from scaf.baselines.matching import mutual_ratio_match
from scaf.baselines.prepare import verify_assets
from scaf.data.catalog import fingerprint, load_catalog, sequence_by_id
from scaf.data.degradation import Degradation, degrade
from scaf.data.sequence import RGBDSequence
from scaf.evaluation.backend import PnPConfig, estimate_motion
from scaf.evaluation.matching import match_metrics
from scaf.evaluation.runner import timed, verify_development_data, write_json_line
from scaf.evaluation.trajectory import SegmentedOdometry, error_statistics, write_tum
from scaf.models.diagnostic import diagnostic_targets
from scaf.models.geometry import GeometryConfig, GeometryEnhancer, build_neighborhood
from scaf.models.losses import temporal_descriptor_loss
from scaf.models.temporal import TemporalConfig, track_pair, candidate_reliability, filter_candidates
from scaf.runtime import environment, git_info, sha256, write_json


METHODS = {"superpoint": (None, False), "control_raw": ("control", False),
           "control_filtered": ("control", True), "temporal_raw": ("temporal", False),
           "temporal_filtered": ("temporal", True)}
KEYS = {"version", "catalog", "baselines", "degradations", "protocol", "train_sequence", "validation_sequences",
        "variants", "max_keypoints", "frame_stride", "validation_frames", "train_pairs", "train_pair_spacing",
        "train_variant", "train_steps", "learning_rate", "temperature", "negative_radius_px", "seed",
        "degradation_seed", "timing_warmup_pairs", "consistency_weight", "temporal", "model"}


def load_settings(path):
    c = yaml.safe_load(Path(path).read_text())
    if not isinstance(c, dict) or set(c) != KEYS or c["version"] != "scaf-stage5-v1":
        raise ValueError("invalid stage-5 schema/version")
    for key in ("max_keypoints", "frame_stride", "validation_frames", "train_pairs", "train_pair_spacing",
                "train_steps", "seed", "degradation_seed", "timing_warmup_pairs"):
        if type(c[key]) is not int or c[key] < (0 if key in {"seed", "degradation_seed", "timing_warmup_pairs"} else 1):
            raise ValueError(f"invalid {key}")
    if c["max_keypoints"] != 512 or c["validation_frames"] < 2 or max(c["seed"], c["degradation_seed"]) >= 2**31:
        raise ValueError("stage-5 check needs 512-point budget and at least two frames")
    for key in ("learning_rate", "temperature", "negative_radius_px", "consistency_weight"):
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
    if GeometryConfig(**c["model"]).descriptor_dim != 256:
        raise ValueError("SuperPoint descriptor dimension must be 256")
    TemporalConfig(**c["temporal"])
    return c


def validate_roles(config, data_root):
    catalog = load_catalog(config["catalog"], data_root)
    if (sequence_by_id(catalog, config["train_sequence"])["split"] != "train" or
            any(sequence_by_id(catalog, s)["split"] != "val" for s in config["validation_sequences"])):
        raise ValueError("stage 5 requires train/val roles; held-out test is refused")
    return catalog


def initial_model(checkpoint, cfg):
    manifest = json.loads((checkpoint.parent/"manifest.json").read_text())
    digest = sha256(checkpoint)
    if (manifest.get("kind") != "stage4_tiny_training_validation_debug" or manifest.get("status") != "completed" or
            manifest.get("test_used") is not False or manifest["source"]["dirty"] or
            manifest["training"]["checkpoint_sha256"] != digest):
        raise ValueError("verified completed stage-4 debug checkpoint required")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["model_config"] != asdict(cfg) or state["kind"] != "stage4_debug_not_formal_training":
        raise ValueError("checkpoint model configuration/kind mismatch")
    model = GeometryEnhancer(cfg)
    model.load_state_dict(state["model"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite initial checkpoint")
    return model, {"path": str(checkpoint), "sha256": digest, "manifest_sha256": sha256(checkpoint.parent/"manifest.json"),
                   "source_commit": manifest["source"]["commit"], "kind": state["kind"]}


def prepare(sequence, index, variant, level, extractor, model_cfg, config, max_wh, folder):
    clean, load_ms = timed(sequence.load, index, max_wh)
    (rgb, depth), degrade_ms = timed(degrade, clean, level, config["degradation_seed"], variant)
    features, extract_ms = timed(extractor.extract, rgb)
    gray, gray_ms = timed(cv2.cvtColor, rgb, cv2.COLOR_RGB2GRAY)
    context, context_ms = timed(build_neighborhood, torch.tensor(features.points)[None], torch.tensor(depth)[None],
                                torch.tensor(clean.K, dtype=torch.float32)[None], config=model_cfg)
    desc = torch.tensor(features.descriptors)[None]
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/f"{index:06d}.npz", points=features.points, descriptors=features.descriptors,
                        scores=features.scores, image_wh=features.image_wh, sequence_index=index,
                        timestamp=clean.timestamp, reliability=context.reliability.numpy()[0])
    return {"clean": clean, "depth": depth, "features": features, "gray": gray, "desc": desc, "context": context,
            "index": index, "timing": {"load_ms": load_ms, "degrade_ms": degrade_ms, "extract_ms": extract_ms,
                                        "gray_ms": gray_ms, "context_ms": context_ms}}


def save_flow(path, context):
    np.savez_compressed(path, **{f"{side}_{key}": value for side, tracks in
        (("forward", context.forward), ("reverse", context.reverse)) for key, value in vars(tracks).items()})


def flow_summary(context):
    return {side: {"points": len(t.valid), "valid": int(t.valid.sum()), "reasons": dict(Counter(t.reason.tolist())),
                   "valid_fb_px": error_statistics(t.fb_error_px[t.valid].tolist())}
            for side, t in (("forward", context.forward), ("reverse", context.reverse))}


def train_checks(initial, extractor, catalog, data_root, levels, protocol, cfg, temporal_cfg, output):
    sequence = RGBDSequence(sequence_by_id(catalog, cfg["train_sequence"]), data_root)
    selection = [(i*cfg["train_pair_spacing"], i*cfg["train_pair_spacing"]+cfg["frame_stride"])
                 for i in range(cfg["train_pairs"])]
    if selection[-1][1] >= len(sequence):
        raise ValueError("training selection exceeds sequence")
    max_wh = (protocol["image"]["max_width"], protocol["image"]["max_height"])
    data, records = [], []
    folder = output/"training_pairs"
    folder.mkdir()
    for pair_id, (i, j) in enumerate(selection):
        frames = [prepare(sequence, x, cfg["train_variant"], levels[cfg["train_variant"]], extractor,
                          initial.config, cfg, max_wh, output/"train_features") for x in (i, j)]
        if any(f["clean"].T_world_camera is None for f in frames):
            records.append({"indices": [i, j], "usable": False, "reason": "reference_pose_missing"})
            continue
        a, b = frames
        ids0, ids1, negative = diagnostic_targets(a["clean"], b["clean"], a["features"].points, b["features"].points,
                                                 cfg["negative_radius_px"])
        pairs = torch.column_stack([ids0, ids1])
        flow = track_pair(a["gray"], b["gray"], a["features"].points, b["features"].points, temporal_cfg)
        evidence = candidate_reliability(flow, pairs.numpy())
        weights = torch.tensor(evidence["weights"])
        _, info = temporal_descriptor_loss(a["desc"][0], b["desc"][0], pairs, negative, weights=weights,
                                            temperature=cfg["temperature"], consistency_weight=cfg["consistency_weight"])
        save_flow(folder/f"{pair_id:03d}-flow.npz", flow)
        np.savez_compressed(folder/f"{pair_id:03d}-targets.npz", pairs=pairs.numpy(), negative_mask=negative.numpy(), **evidence)
        records.append({"indices": [i, j], "identities": [f["clean"].identity for f in frames], "usable": bool(info["anchors"]),
                        **info, "flow": flow_summary(flow), "timing": flow.timing_ms})
        if info["anchors"]:
            data.append((frames, pairs, negative, weights))
    write_json(output/"training_selection.json", records)
    if not data:
        raise ValueError("no certified positives with trusted negatives")
    trained, summaries = {}, {}
    for name, temporal in (("control", False), ("temporal", True)):
        model = deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        def objective(item, weighted):
            frames, pairs, negative, weights = item
            a, b = [model(f["desc"], f["context"])[0] for f in frames]
            return temporal_descriptor_loss(a, b, pairs, negative, weights=weights if weighted else None,
                temperature=cfg["temperature"], consistency_weight=cfg["consistency_weight"] if weighted else 0.)
        def mean_loss(weighted):
            with torch.no_grad():
                return float(np.mean([float(objective(x, weighted)[0]) for x in data]))
        before, unweighted_before = mean_loss(temporal), mean_loss(False)
        model.train()
        skipped = 0
        with (output/f"training-{name}.jsonl").open("x") as log:
            for step in range(cfg["train_steps"]):
                optimizer.zero_grad(set_to_none=True)
                loss, info = objective(data[step % len(data)], temporal)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite temporal loss")
                update = info["weighted_anchors"] > 0
                grad_norm = 0.
                if update:
                    loss.backward()
                    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError("missing/nonfinite temporal gradients")
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
                    optimizer.step()
                else:
                    skipped += 1  # Do not let Adam momentum update a zero-weight step.
                write_json_line(log, {"step": step, "pair": step % len(data), "optimizer_updated": update,
                                      "gradient_norm_before_clip": grad_norm, **info})
                if (step+1) % len(data) == 0:
                    print(json.dumps({"training": name, "step": step+1, "loss": float(loss.detach())}), flush=True)
        model.eval()
        changes = {k: float((v-initial.state_dict()[k]).abs().max()) for k, v in model.state_dict().items()}
        if not max(changes.values()) > 0 or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("training did not update finite parameters")
        checkpoint = output/f"{name}-final.pt"
        torch.save({"model": model.state_dict(), "model_config": asdict(model.config), "seed": cfg["seed"],
                    "steps": cfg["train_steps"], "temporal_training": temporal, "kind": "stage5_debug_not_formal_training"}, checkpoint)
        reloaded = GeometryEnhancer(model.config).eval()
        reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
        with torch.no_grad():
            sample = data[0][0][0]
            torch.testing.assert_close(model(sample["desc"], sample["context"]),
                                       reloaded(sample["desc"], sample["context"]), rtol=0, atol=0)
        summaries[name] = {"temporal_training": temporal, "initial_objective": before, "final_objective": mean_loss(temporal),
            "initial_unweighted_discriminative": unweighted_before, "final_unweighted_discriminative": mean_loss(False),
            "scheduled_steps": cfg["train_steps"], "optimizer_steps": cfg["train_steps"]-skipped, "skipped_zero_weight_steps": skipped,
            "parameter_changes": changes, "checkpoint_sha256": sha256(checkpoint), "checkpoint_reload_exact": True,
            "all_gradients_finite": True, "checkpoint_selection": "last_fixed_step_without_validation_selection"}
        trained[name] = model
    write_json(output/"training_summary.json", summaries)
    return trained, summaries


def time_summary(rows, warmup):
    selected = rows[warmup:]
    return {"unit": "ms", "performance_benchmark": False, "warmup_pairs_excluded": min(warmup, len(rows)),
            "samples": len(selected), "components": {k: {"p50": float(np.median([x[k] for x in selected])) if selected else None,
              "p95": float(np.percentile([x[k] for x in selected], 95)) if selected else None} for k in rows[0]} if rows else {}}


@torch.no_grad()
def evaluate(trained, extractor, spec, variant, level, data_root, cfg, protocol, temporal_cfg, output):
    output.mkdir(parents=True, exist_ok=False)
    sequence = RGBDSequence(spec, data_root)
    indices = list(range(0, len(sequence), cfg["frame_stride"]))[:cfg["validation_frames"]]
    if len(indices) < 2:
        raise ValueError("validation needs at least two frames")
    model_cfg = trained["control"].config
    max_wh = (protocol["image"]["max_width"], protocol["image"]["max_height"])
    settings = protocol["backend"]
    pnp = PnPConfig(reprojection_px=settings["ransac_reprojection_px"], iterations=settings["ransac_iterations"],
                    confidence=settings["ransac_confidence"], min_correspondences=settings["min_correspondences"],
                    min_inliers=settings["min_inliers"], seed=cfg["seed"])
    trackers, rows = {m: SegmentedOdometry() for m in METHODS}, {m: [] for m in METHODS}
    stamps, references = [], []
    write_json(output/"selection.json", {"sequence": spec, "variant": variant, "degradation": asdict(level), "indices": indices,
        "available_frames": len(sequence), "backend_depth": "perturbed_input", "scope": "stage5_validation_prefix_debug",
        "training_switch": "equal_initial_checkpoint_fixed_schedule_two_separate_models", "filter_switch": "same_model_candidates"})
    (output/"flow").mkdir()
    previous = None
    with (output/"frames.jsonl").open("x") as frame_log, (output/"pairs.jsonl").open("x") as pair_log:
        for ordinal, index in enumerate(indices):
            f = prepare(sequence, index, variant, level, extractor, model_cfg, cfg, max_wh, output/"features")
            stamps.append(f["clean"].timestamp)
            references.append(f["clean"].T_world_camera)
            enhanced, network_ms = {}, {}
            for name, model in trained.items():
                desc, network_ms[name] = timed(model, f["desc"], f["context"])
                enhanced[name] = replace(f["features"], descriptors=desc[0].numpy())
                folder = output/name
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f"{ordinal:06d}.npz", descriptors=enhanced[name].descriptors,
                                    sequence_index=index, timestamp=f["clean"].timestamp)
            f["enhanced"], f["network_ms"] = enhanced, network_ms
            write_json_line(frame_log, {"ordinal": ordinal, "index": index, "timestamp": f["clean"].timestamp,
                                       "identity": f["clean"].identity, "keypoints": len(f["features"].points),
                                       "timing": f["timing"], "network_ms": network_ms})
            if previous is not None:
                flow, flow_total_ms = timed(track_pair, previous["gray"], f["gray"], previous["features"].points,
                                            f["features"].points, temporal_cfg)
                save_flow(output/"flow"/f"{ordinal-1:06d}-{ordinal:06d}.npz", flow)
                candidates, match_times = {}, {}
                for name in (None, "control", "temporal"):
                    a = previous["features"] if name is None else previous["enhanced"][name]
                    b = f["features"] if name is None else enhanced[name]
                    candidates[name], match_times[name] = timed(mutual_ratio_match, a.descriptors, b.descriptors, protocol["matching"]["ratio"])
                for method, (name, filtered) in METHODS.items():
                    candidate_pairs, candidate_scores = candidates[name]
                    if filtered:
                        evidence, filter_ms = timed(filter_candidates, flow, candidate_pairs, candidate_scores)
                        pairs, scores = evidence["pairs"], evidence["scores"]
                    else:
                        filter_ms = 0.
                        evidence = {"weights": np.ones(len(candidate_pairs), np.float32), "eligible": np.ones(len(candidate_pairs), bool),
                                    "keep": np.ones(len(candidate_pairs), bool), "candidate_scores": candidate_scores.copy(),
                                    "reason": np.full(len(candidate_pairs), "filter_disabled", dtype="U36")}
                        pairs, scores = candidate_pairs, candidate_scores
                    motion, backend_ms = timed(estimate_motion, previous["features"].points, f["features"].points, pairs,
                                               previous["depth"], previous["clean"].K, f["clean"].K, pnp)
                    quality, labels = match_metrics(previous["clean"], f["clean"], previous["features"].points,
                                                    f["features"].points, pairs, protocol["matching"]["geometric_label_reprojection_px"])
                    segment = trackers[method].append(ordinal-1, ordinal, motion["T_camera1_camera0"])
                    enhancement_ms = f["timing"]["context_ms"]+network_ms[name] if name else 0.
                    optical_flow_ms = flow_total_ms+f["timing"]["gray_ms"]+previous["timing"]["gray_ms"] if filtered else 0.
                    timing = {"load_ms": f["timing"]["load_ms"], "degrade_ms": f["timing"]["degrade_ms"],
                              "extract_ms": f["timing"]["extract_ms"], "enhancement_ms": enhancement_ms,
                              "optical_flow_ms": optical_flow_ms, "filter_ms": filter_ms, "match_ms": match_times[name], "backend_ms": backend_ms}
                    timing["frontend_ms"] = sum(timing[k] for k in ("extract_ms", "enhancement_ms", "optical_flow_ms", "filter_ms", "match_ms"))
                    timing["pipeline_ms"] = timing["frontend_ms"]+backend_ms
                    timing["input_plus_compute_ms"] = timing["pipeline_ms"]+timing["load_ms"]+timing["degrade_ms"]
                    row = {"method": method, "training_temporal": name == "temporal", "filter_enabled": filtered,
                           "frame0": ordinal-1, "frame1": ordinal, "indices": [previous["index"], index],
                           "timestamp0": previous["clean"].timestamp, "timestamp1": f["clean"].timestamp, "segment": segment,
                           "candidates": len(candidate_pairs), "retained": len(pairs), "rejected": len(candidate_pairs)-len(pairs),
                           "retained_weight_mean": float(evidence["weights"][evidence["keep"]].mean()) if len(pairs) else None,
                           "filter_reasons": dict(Counter(evidence["reason"].tolist())), "quality": quality,
                           "backend": {k: v for k, v in motion.items() if k not in {"T_camera1_camera0", "inlier_match_indices"}},
                           "flow": flow_summary(flow) if filtered else None, "timing": timing}
                    write_json_line(pair_log, row)
                    rows[method].append(row)
                    folder = output/method/"matches"
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder/f"{ordinal-1:06d}-{ordinal:06d}.npz", candidate_pairs=candidate_pairs,
                        descriptor_scores=candidate_scores, pairs=pairs, scores=scores,
                        **{("filter_reason" if k == "reason" else k): v for k, v in evidence.items() if k not in {"pairs", "scores"}}, **labels,
                        inlier_match_indices=motion["inlier_match_indices"],
                        T_camera1_camera0=motion["T_camera1_camera0"] if motion["success"] else np.empty((0, 4)))
            previous = f
            if (ordinal+1) % 8 == 0 or ordinal == len(indices)-1:
                print(json.dumps({"sequence": spec["id"], "variant": variant, "frames": ordinal+1}), flush=True)
    write_tum(output/"groundtruth.txt", [(i, p) for i, p in enumerate(references) if p is not None], stamps)
    results = []
    for method, tracker in trackers.items():
        method_rows = rows[method]
        for i, segment in enumerate(tracker.segments):
            write_tum(output/method/f"trajectory-{i:03d}.txt", segment, stamps)
        q = {key: sum(r["quality"][key] for r in method_rows) for key in ("predicted", "evaluable", "correct", "unevaluable")}
        q["precision"] = q["correct"]/q["evaluable"] if q["evaluable"] else None
        q["correct_per_pair"] = error_statistics([r["quality"]["correct"] for r in method_rows])
        q["predicted_per_pair"] = error_statistics([r["quality"]["predicted"] for r in method_rows])
        q["reasons"] = dict(sum((Counter(r["quality"]["reasons"]) for r in method_rows), Counter()))
        candidates = sum(r["candidates"] for r in method_rows)
        result = {"sequence": spec["id"], "variant": variant, "method": method, "frames": len(indices), "quality": q,
                  "candidates": candidates, "retained": q["predicted"], "retention": q["predicted"]/candidates if candidates else None,
                  "trajectory": tracker.summary(stamps, references),
                  "backend_failures": dict(Counter(r["backend"]["reason"] for r in method_rows if not r["backend"]["success"])),
                  "timing": time_summary([r["timing"] for r in method_rows], cfg["timing_warmup_pairs"])}
        write_json(output/method/"summary.json", result)
        results.append(result)
    return results


def write_csv(path, results):
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "variant", "method", "precision", "correct_per_pair", "matches_per_pair", "retention", "tracking_success",
                         "coverage", "failed", "segments", "ate_rmse_m", "rpe_translation_rmse_m", "flow_p50_ms", "frontend_p50_ms"])
        for r in results:
            q, t, c = r["quality"], r["trajectory"], r["timing"]["components"]
            writer.writerow([r["sequence"], r["variant"], r["method"], q["precision"], q["correct_per_pair"]["mean"],
                q["predicted_per_pair"]["mean"], r["retention"], t["tracking_success_rate"], t["frame_coverage"], t["failed_transitions"],
                len(t["segments"]), t["full_trajectory_ate_m"]["rmse"] if t["full_trajectory_ate_m"] else None,
                t["rpe_translation_m"]["rmse"], c["optical_flow_ms"]["p50"], c["frontend_ms"]["p50"]])


def run(cfg, data_root, audit_manifest, asset_manifest, geometry_checkpoint, output):
    root = Path(__file__).resolve().parents[2]
    if root != Path.cwd().resolve():
        raise ValueError("run from checkout root with PYTHONPATH=$PWD/src")
    output.mkdir(parents=True, exist_ok=False)
    record = {"kind": "stage5_temporal_tiny_factorial_debug", "status": "running", "source": git_info(root),
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root), "invocation": sys.argv,
        "config": cfg, "config_sha256": fingerprint(cfg), "environment": environment(), "device": "cpu", "threads": 1,
        "data_root": str(data_root), "audit_manifest": str(audit_manifest), "asset_manifest": str(asset_manifest),
        "test_used": False, "experiment": "two_equal_initial_models_with_independent_training_and_each_filter_on_off"}
    start = perf_counter()
    write_json(output/"manifest.json", record)
    try:
        if record["source"]["dirty"]:
            raise ValueError("stage-5 acceptance requires a clean committed checkout")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        cv2.setNumThreads(1)
        torch.use_deterministic_algorithms(True)
        (output/"dependencies.txt").write_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in
            distributions(path=[sysconfig.get_paths()["purelib"]])))+"\n")
        validate_roles(cfg, data_root)
        catalog, data_record = verify_development_data({**cfg, "sequences": [cfg["train_sequence"], *cfg["validation_sequences"]]}, data_root, audit_manifest)
        assets = verify_assets(asset_manifest, yaml.safe_load(Path(cfg["baselines"]).read_text()))
        protocol = yaml.safe_load(Path(cfg["protocol"]).read_text())
        if protocol["version"] != 2 or protocol["backend"].get("solver") != "EPNP_RANSAC_LM":
            raise ValueError("frozen backend protocol v2 required")
        degradation_cfg = yaml.safe_load(Path(cfg["degradations"]).read_text())
        levels = {v: Degradation(**degradation_cfg["levels"][v]) for v in set(cfg["variants"]+[cfg["train_variant"]])}
        extractor = BaselineModels(assets, cfg["max_keypoints"], "cpu", ["superpoint"])
        frozen = {k: v.clone() for k, v in extractor.extractor.state_dict().items()}
        initial, initial_record = initial_model(geometry_checkpoint, GeometryConfig(**cfg["model"]))
        temporal_cfg = TemporalConfig(**cfg["temporal"])
        record.update(catalog=catalog, data=data_record, assets=assets, asset_manifest_sha256=sha256(asset_manifest),
            protocol=protocol, degradations=degradation_cfg, initial_checkpoint=initial_record,
            enhancer_parameters=sum(p.numel() for p in initial.parameters()), temporal_trainable_parameters=0,
            cpu={"logical_cores": os.cpu_count(), "load_average_start": list(os.getloadavg()), "precision": "float32_models_float64_pose_geometry"})
        write_json(output/"manifest.json", record)
        trained, record["training"] = train_checks(initial, extractor, catalog, data_root, levels, protocol, cfg, temporal_cfg, output)
        results = []
        for identifier in cfg["validation_sequences"]:
            for variant in cfg["variants"]:
                results.extend(evaluate(trained, extractor, sequence_by_id(catalog, identifier), variant, levels[variant],
                                        data_root, cfg, protocol, temporal_cfg, output/identifier/variant))
                write_json(output/"summary.json", {"status": "partial", "results": results})
        if (any(p.requires_grad or p.grad is not None for p in extractor.extractor.parameters()) or
                any(not torch.equal(v, frozen[k]) for k, v in extractor.extractor.state_dict().items())):
            raise AssertionError("SuperPoint changed")
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
    parser.add_argument("--config", type=Path, default=Path("configs/stage5.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(run(load_settings(args.config), args.data_root.resolve(), args.audit_manifest.resolve(),
                  args.asset_manifest.resolve(), args.geometry_checkpoint.resolve(), args.output.resolve()))
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
