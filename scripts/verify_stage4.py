"""Audit saved stage-4 outputs without rerunning extraction, training or inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(root):
    manifest = read(root/"manifest.json")
    config = manifest["config"]
    assert manifest["status"] == "completed" and not manifest["source"]["dirty"]
    assert not manifest["cuda_initialized"] and not manifest["test_used"] and manifest["superpoint_unchanged"]
    assert digest(root/"summary.json") == manifest["summary_sha256"]
    assert digest(root/"summary.csv") == manifest["summary_csv_sha256"]
    train = read(root/"training_summary.json")
    assert digest(root/"geometry-final.pt") == train["checkpoint_sha256"]
    assert train["checkpoint_reload_exact"] and train["all_gradients_finite"] and train["max_parameter_change"] > 0
    steps = lines(root/"training.jsonl")
    assert [s["step"] for s in steps] == list(range(config["train_steps"]))
    assert all(s["positives"] > 0 and s["negative_edges"] > 0 and s["anchors"] > 0 and
               np.isfinite(s["loss"]) and np.isfinite(s["gradient_norm_before_clip"]) for s in steps)
    roles = {s["id"]: s["split"] for s in manifest["catalog"]["sequences"]}
    assert roles[config["train_sequence"]] == "train"
    assert all(roles[s] == "val" for s in config["validation_sequences"])
    summary = read(root/"summary.json")
    assert summary["status"] == "completed"
    results = summary["results"]
    methods = {"superpoint", "featurebooster", "appearance", "geometry_unweighted", "geometry_reliable"}
    assert {(r["sequence"], r["variant"], r["method"]) for r in results} == {
        (s, v, m) for s in config["validation_sequences"] for v in config["variants"] for m in methods}
    assert len(results) == len(config["validation_sequences"])*len(config["variants"])*len(methods)
    counts = {"base_features": 0, "enhanced_features": 0, "match_files": 0, "zero_depth_fallback_points": 0}
    for sequence in config["validation_sequences"]:
        for variant in config["variants"]:
            directory = root/sequence/variant
            selection = read(directory/"selection.json")
            frames, pairs = lines(directory/"frames.jsonl"), lines(directory/"pairs.jsonl")
            indices = selection["indices"]
            assert indices == list(range(0, selection["available_frames"], config["frame_stride"]))[:config["validation_frames"]]
            assert len(frames) == len(indices) and len(pairs) == (len(indices)-1)*len(methods)
            assert len(list((directory/"features").glob("*.npz"))) == len(indices)
            stamps, keypoints = [], []
            for ordinal, index in enumerate(indices):
                with np.load(directory/"features"/f"{index:06d}.npz", allow_pickle=False) as f:
                    n = len(f["points"])
                    assert 0 <= n <= 512 and f["descriptors"].shape == (n, 256)
                    assert np.isfinite(f["descriptors"]).all() and np.isfinite(f["xyz"]).all()
                    np.testing.assert_allclose(np.linalg.norm(f["descriptors"], axis=-1), 1, atol=1e-5)
                    q = f["reliability"]
                    assert q.shape == (n,) and np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
                    assert int(f["sequence_index"]) == index
                    stamps.append(float(f["timestamp"]))
                    keypoints.append(n)
                    assert frames[ordinal]["index"] == index and frames[ordinal]["timestamp"] == stamps[-1]
                    assert frames[ordinal]["keypoints"] == n
                    counts["base_features"] += 1
                    output = {}
                    for method in ("appearance", "geometry_unweighted", "geometry_reliable"):
                        with np.load(directory/method/f"{ordinal:06d}.npz", allow_pickle=False) as e:
                            d = e["descriptors"].copy()
                            assert d.shape == (n, 256) and np.isfinite(d).all()
                            np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1, atol=1e-5)
                            assert float(e["timestamp"]) == stamps[-1] and int(e["sequence_index"]) == index
                            output[method] = d
                        counts["enhanced_features"] += 1
                    np.testing.assert_array_equal(output["appearance"][q == 0], output["geometry_reliable"][q == 0])
                    counts["zero_depth_fallback_points"] += int((q == 0).sum())
            assert (np.diff(stamps) > 0).all()
            for method in methods:
                rows = [p for p in pairs if p["method"] == method]
                result = next(r for r in results if (r["sequence"], r["variant"], r["method"]) == (sequence, variant, method))
                totals = {k: 0 for k in ("predicted", "evaluable", "correct", "unevaluable")}
                assert len(list((directory/method/"matches").glob("*.npz"))) == len(rows)
                for ordinal, row in enumerate(rows):
                    assert row["frame0"] == ordinal and row["frame1"] == ordinal+1
                    assert row["indices"] == indices[ordinal:ordinal+2]
                    assert row["timestamp0"] == stamps[ordinal] and row["timestamp1"] == stamps[ordinal+1]
                    with np.load(directory/method/"matches"/f"{ordinal:06d}-{ordinal+1:06d}.npz", allow_pickle=False) as f:
                        p = f["pairs"]
                        assert p.shape == (len(p), 2) and np.issubdtype(p.dtype, np.integer)
                        for side in (0, 1):
                            assert len(np.unique(p[:, side])) == len(p)
                            assert ((p[:, side] >= 0) & (p[:, side] < keypoints[ordinal+side])).all()
                        evaluable, correct = f["evaluable"], f["correct"]
                        assert len(evaluable) == len(p) and not (correct & ~evaluable).any()
                        q = row["quality"]
                        assert (q["predicted"], q["evaluable"], q["correct"], q["unevaluable"]) == (
                            len(p), int(evaluable.sum()), int(correct.sum()), int((~evaluable).sum()))
                        assert sum(q["reasons"].values()) == len(p)
                        assert q["precision"] == (q["correct"]/q["evaluable"] if q["evaluable"] else None)
                        T = f["T_camera1_camera0"]
                        if row["backend"]["success"]:
                            assert T.shape == (4, 4) and np.isfinite(T).all()
                            np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-6)
                            np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
                            assert abs(np.linalg.det(T[:3, :3])-1) < 1e-6
                            assert len(f["inlier_match_indices"]) == row["backend"]["inliers"]
                        else:
                            assert T.shape == (0, 4)
                    for k in totals:
                        totals[k] += row["quality"][k]
                    counts["match_files"] += 1
                assert all(totals[k] == result["quality"][k] for k in totals)
                t = result["trajectory"]
                failed = sum(not r["backend"]["success"] for r in rows)
                assert t["failed_transitions"] == failed
                assert t["tracking_success_rate"] == (len(rows)-failed)/len(rows)
                trajectories = sorted((directory/method).glob("trajectory-*.txt"))
                assert len(trajectories) == len(t["segments"])
                covered = set()
                for path, segment in zip(trajectories, t["segments"]):
                    poses = np.loadtxt(path, ndmin=2)
                    assert len(poses) == segment["frames"] and np.isfinite(poses).all()
                    assert (np.diff(poses[:, 0]) > 0).all()
                    np.testing.assert_allclose(np.linalg.norm(poses[:, 4:], axis=-1), 1, atol=1e-6)
                    covered.update(poses[:, 0])
                assert len(covered) == t["covered_frames"]
                assert t["frame_coverage"] == len(covered)/len(frames)
                assert result["timing"]["samples"] == max(0, len(frames)-config["timing_warmup_frames"])
    return {"status": "passed", "source_commit": manifest["source"]["commit"], "results": len(results),
            "checker_sha256": digest(Path(__file__)), "training_steps": len(steps),
            "checkpoint_sha256": train["checkpoint_sha256"], **counts}


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
