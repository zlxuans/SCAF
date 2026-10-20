"""Validate persisted temporal experiment outputs without extraction/training/flow."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reliability(flow, pairs, cfg):
    i, j = pairs.T
    e0 = np.linalg.norm(flow["forward_predicted"][i]-flow["reverse_origins"][j], axis=-1)
    e1 = np.linalg.norm(flow["reverse_predicted"][j]-flow["forward_origins"][i], axis=-1)
    eligible = (flow["forward_valid"][i] & flow["reverse_valid"][j] & np.isfinite(e0) & np.isfinite(e1) &
                (e0 <= cfg["candidate_max_px"]) & (e1 <= cfg["candidate_max_px"]))
    weights = np.zeros(len(pairs), np.float32)
    ii, jj = i[eligible], j[eligible]
    weights[eligible] = np.exp(-.25*((flow["forward_fb_error_px"][ii]**2+flow["reverse_fb_error_px"][jj]**2)/cfg["fb_sigma_px"]**2+
                                   (e0[eligible]**2+e1[eligible]**2)/cfg["candidate_sigma_px"]**2))
    return weights, eligible


def check_flow(flow, cfg):
    for side in ("forward", "reverse"):
        valid = flow[f"{side}_valid"]
        assert np.array_equal(valid, flow[f"{side}_reason"] == "ok")
        assert np.isfinite(flow[f"{side}_predicted"]).all() and np.isfinite(flow[f"{side}_returned"]).all()
        errors = np.linalg.norm(flow[f"{side}_origins"][valid]-flow[f"{side}_returned"][valid], axis=-1)
        np.testing.assert_allclose(errors, flow[f"{side}_fb_error_px"][valid], atol=1e-6)
        assert (errors <= cfg["fb_max_px"]+1e-6).all()


def verify(root):
    manifest = read(root/"manifest.json")
    cfg, temporal_cfg = manifest["config"], manifest["config"]["temporal"]
    assert manifest["status"] == "completed" and not manifest["source"]["dirty"]
    assert manifest["superpoint_unchanged"] and not manifest["cuda_initialized"] and not manifest["test_used"]
    assert digest(root/"summary.json") == manifest["summary_sha256"]
    assert digest(root/"summary.csv") == manifest["summary_csv_sha256"]
    roles = {s["id"]: s["split"] for s in manifest["catalog"]["sequences"]}
    assert roles[cfg["train_sequence"]] == "train" and all(roles[s] == "val" for s in cfg["validation_sequences"])
    training = read(root/"training_summary.json")
    assert training["control"]["initial_unweighted_discriminative"] == training["temporal"]["initial_unweighted_discriminative"]
    for name in ("control", "temporal"):
        record = training[name]
        assert digest(root/f"{name}-final.pt") == record["checkpoint_sha256"]
        assert record["checkpoint_reload_exact"] and record["all_gradients_finite"] and max(record["parameter_changes"].values()) > 0
        steps = lines(root/f"training-{name}.jsonl")
        assert [s["step"] for s in steps] == list(range(cfg["train_steps"]))
        assert record["optimizer_steps"] == sum(s["optimizer_updated"] for s in steps)
        assert record["skipped_zero_weight_steps"] == sum(not s["optimizer_updated"] for s in steps)
        for s in steps:
            assert np.isfinite([s[k] for k in ("total", "discriminative", "consistency", "gradient_norm_before_clip")]).all()
            if not s["optimizer_updated"]:
                assert s["weighted_anchors"] == 0 and s["gradient_norm_before_clip"] == 0
    counts = {"train_banks": 0, "flow_files": 0, "shared_features": 0, "enhanced_features": 0,
              "match_files": 0, "filtered_candidate_pairs": 0, "retained_candidate_pairs": 0}
    for path in sorted((root/"training_pairs").glob("*-targets.npz")):
        with np.load(path, allow_pickle=False) as target, np.load(path.with_name(path.name.replace("targets", "flow")), allow_pickle=False) as flow:
            check_flow(flow, temporal_cfg)
            weights, eligible = reliability(flow, target["pairs"], temporal_cfg)
            np.testing.assert_allclose(weights, target["weights"], rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(eligible, target["eligible"])
            negatives = target["negative_mask"]
            assert negatives.dtype == bool and negatives.shape == (len(weights), len(weights)) and not negatives.diagonal().any()
        counts["train_banks"] += 1
    methods = {"superpoint", "control_raw", "control_filtered", "temporal_raw", "temporal_filtered"}
    results = read(root/"summary.json")["results"]
    assert len(results) == len(cfg["validation_sequences"])*len(cfg["variants"])*len(methods)
    assert {(r["sequence"], r["variant"], r["method"]) for r in results} == {
        (s, v, m) for s in cfg["validation_sequences"] for v in cfg["variants"] for m in methods}
    for sequence in cfg["validation_sequences"]:
        for variant in cfg["variants"]:
            folder = root/sequence/variant
            selection = read(folder/"selection.json")
            indices = selection["indices"]
            assert indices == list(range(0, selection["available_frames"], cfg["frame_stride"]))[:cfg["validation_frames"]]
            frames, pair_rows = lines(folder/"frames.jsonl"), lines(folder/"pairs.jsonl")
            assert len(frames) == len(indices) and len(pair_rows) == (len(indices)-1)*len(methods)
            points, stamps, sizes = [], [], []
            for ordinal, index in enumerate(indices):
                with np.load(folder/"features"/f"{index:06d}.npz", allow_pickle=False) as f:
                    d, p = f["descriptors"], f["points"].copy()
                    assert d.shape == (len(p), 256) and len(p) <= 512 and np.isfinite(d).all()
                    np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1, atol=1e-5)
                    points.append(p)
                    stamps.append(float(f["timestamp"]))
                    sizes.append(len(p))
                    assert frames[ordinal]["index"] == index and frames[ordinal]["timestamp"] == stamps[-1]
                    assert int(f["sequence_index"]) == index
                counts["shared_features"] += 1
                for name in ("control", "temporal"):
                    with np.load(folder/name/f"{ordinal:06d}.npz", allow_pickle=False) as f:
                        d = f["descriptors"]
                        assert d.shape == (sizes[-1], 256) and np.isfinite(d).all()
                        np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1, atol=1e-5)
                        assert int(f["sequence_index"]) == index and float(f["timestamp"]) == stamps[-1]
                    counts["enhanced_features"] += 1
            assert (np.diff(stamps) > 0).all()
            for ordinal in range(len(indices)-1):
                with np.load(folder/"flow"/f"{ordinal:06d}-{ordinal+1:06d}.npz", allow_pickle=False) as flow:
                    check_flow(flow, temporal_cfg)
                    np.testing.assert_array_equal(flow["forward_origins"], points[ordinal])
                    np.testing.assert_array_equal(flow["reverse_origins"], points[ordinal+1])
                    for method in methods:
                        path = folder/method/"matches"/f"{ordinal:06d}-{ordinal+1:06d}.npz"
                        row = next(r for r in pair_rows if r["method"] == method and r["frame0"] == ordinal)
                        with np.load(path, allow_pickle=False) as f:
                            p, candidates, keep = f["pairs"], f["candidate_pairs"], f["keep"]
                            for pp in (p, candidates):
                                assert pp.shape == (len(pp), 2) and np.issubdtype(pp.dtype, np.integer)
                                for side in (0, 1):
                                    assert len(np.unique(pp[:, side])) == len(pp)
                                    assert ((pp[:, side] >= 0) & (pp[:, side] < sizes[ordinal+side])).all()
                            np.testing.assert_array_equal(p, candidates[keep])
                            assert row["candidates"] == len(candidates) and row["retained"] == len(p) and row["rejected"] == len(candidates)-len(p)
                            assert row["indices"] == indices[ordinal:ordinal+2]
                            assert row["timestamp0"] == stamps[ordinal] and row["timestamp1"] == stamps[ordinal+1]
                            if method.endswith("_filtered"):
                                weights, eligible = reliability(flow, candidates, temporal_cfg)
                                np.testing.assert_allclose(weights, f["weights"], atol=1e-7, rtol=1e-6)
                                np.testing.assert_array_equal(eligible, f["eligible"])
                                np.testing.assert_array_equal(keep, eligible & (weights >= temporal_cfg["minimum_weight"]))
                                np.testing.assert_allclose(f["scores"], (weights*f["descriptor_scores"])[keep], atol=1e-7)
                                with np.load(folder/method.replace("_filtered", "_raw")/"matches"/path.name, allow_pickle=False) as raw:
                                    np.testing.assert_array_equal(candidates, raw["pairs"])
                                    np.testing.assert_array_equal(f["descriptor_scores"], raw["scores"])
                                counts["filtered_candidate_pairs"] += len(candidates)
                                counts["retained_candidate_pairs"] += len(p)
                                assert row["timing"]["optical_flow_ms"] > 0
                            else:
                                assert keep.all() and row["timing"]["optical_flow_ms"] == 0
                                np.testing.assert_array_equal(f["scores"], f["descriptor_scores"])
                            q = row["quality"]
                            assert (q["predicted"], q["evaluable"], q["correct"], q["unevaluable"]) == (
                                len(p), int(f["evaluable"].sum()), int(f["correct"].sum()), int((~f["evaluable"]).sum()))
                            assert not (f["correct"] & ~f["evaluable"]).any()
                            assert q["precision"] == (q["correct"]/q["evaluable"] if q["evaluable"] else None)
                            assert sum(q["reasons"].values()) == len(p)
                            T = f["T_camera1_camera0"]
                            if row["backend"]["success"]:
                                assert T.shape == (4, 4) and np.isfinite(T).all()
                                np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-6)
                                np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
                                assert abs(np.linalg.det(T[:3, :3])-1) < 1e-6
                                assert len(f["inlier_match_indices"]) == row["backend"]["inliers"]
                            else:
                                assert T.shape == (0, 4)
                        counts["match_files"] += 1
                counts["flow_files"] += 1
            for method in methods:
                rows = [r for r in pair_rows if r["method"] == method]
                result = next(r for r in results if (r["sequence"], r["variant"], r["method"]) == (sequence, variant, method))
                q, t = result["quality"], result["trajectory"]
                assert all(q[k] == sum(r["quality"][k] for r in rows) for k in ("predicted", "evaluable", "correct", "unevaluable"))
                assert result["candidates"] == sum(r["candidates"] for r in rows)
                failed = sum(not r["backend"]["success"] for r in rows)
                assert t["failed_transitions"] == failed and t["tracking_success_rate"] == (len(rows)-failed)/len(rows)
                trajectories = sorted((folder/method).glob("trajectory-*.txt"))
                assert len(trajectories) == len(t["segments"])
                covered = set()
                for path, segment in zip(trajectories, t["segments"]):
                    poses = np.loadtxt(path, ndmin=2)
                    assert len(poses) == segment["frames"] and np.isfinite(poses).all()
                    assert (np.diff(poses[:, 0]) > 0).all()
                    np.testing.assert_allclose(np.linalg.norm(poses[:, 4:], axis=-1), 1, atol=1e-6)
                    covered.update(poses[:, 0])
                assert len(covered) == t["covered_frames"] and t["frame_coverage"] == len(covered)/len(frames)
                assert result["timing"]["samples"] == max(0, len(rows)-cfg["timing_warmup_pairs"])
    return {"status": "passed", "source_commit": manifest["source"]["commit"], "checker_sha256": digest(Path(__file__)),
            "results": len(results), "scheduled_steps_per_model": cfg["train_steps"], **counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.experiment)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result))
