"""Analytic ground truth checks for matching, pose direction, alignment and failures."""
from copy import deepcopy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scaf.baselines.matching import Features, mutual_ratio_match, unit_descriptors, validate_pairs
from scaf.evaluation.backend import PnPConfig, estimate_motion
from scaf.evaluation.matching import match_metrics
from scaf.evaluation.runner import load_settings, timing_summary, verify_development_data
from scaf.evaluation.trajectory import SegmentedOdometry, align_se3, matrix_quaternion, relative_error
from scaf.data.geometry import quaternion_matrix
from scaf.runtime import sha256, write_json
from test_data import frame, mini_data


ROOT = Path(__file__).resolve().parents[1]


def pose(translation=(0, 0, 0), rotvec=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rotvec, dtype=float))[0]
    T[:3, 3] = translation
    return T


def test_mutual_two_sided_euclidean_ratio_and_ties():
    # d0/d1 is ~0.86: an incorrectly unsquared threshold on d0^2/d1^2 accepts it.
    a = np.array([[1., 0], [-1., 0]])
    b = np.array([[np.cos(.6), np.sin(.6)], [np.cos(.7), np.sin(.7)]])
    assert len(mutual_ratio_match(a, b)[0]) == 0
    rng = np.random.default_rng(42)
    a = unit_descriptors(rng.normal(size=(20, 32)))
    b = np.concatenate([a[:10]+rng.normal(0, .01, (10, 32)), rng.normal(size=(5, 32))])
    pairs, scores = mutual_ratio_match(a, b)
    np.testing.assert_array_equal(pairs, np.column_stack([np.arange(10), np.arange(10)]))
    assert (scores > .9).all()
    assert not len(mutual_ratio_match(a[:1], b)[0])
    assert not len(mutual_ratio_match(a, b[:1])[0])
    assert not len(mutual_ratio_match(a[:0], b)[0])
    assert not len(mutual_ratio_match(np.ones((2, 3)), np.ones((2, 3)))[0])
    # Reverse ratio is also necessary: both source points are close to target 0.
    a = np.array([[1., 0], [np.cos(.02), np.sin(.02)]])
    b = np.array([[np.cos(.009), np.sin(.009)], [-1, 0.]])
    assert not len(mutual_ratio_match(a, b)[0])


@pytest.mark.parametrize("descriptors", [np.zeros((1, 3)), [[np.nan, 1]], [1, 2]])
def test_invalid_descriptors_rejected(descriptors):
    with pytest.raises(ValueError):
        unit_descriptors(descriptors)


def test_match_contract_and_feature_bounds():
    with pytest.raises(ValueError, match="one-to-one"):
        validate_pairs(np.array([[0, 0], [1, 0]]), 2, 2)
    with pytest.raises(ValueError, match="out of bounds"):
        validate_pairs(np.array([[0, 2]]), 2, 2)
    with pytest.raises(ValueError, match="integer"):
        validate_pairs(np.array([[0., 0.]]), 2, 2)
    with pytest.raises(ValueError, match="outside"):
        Features([[16, 0]], [[1, 0]], [1], (16, 12))


def test_geometric_precision_separates_wrong_and_unevaluable():
    a, b = frame(), frame()
    points0 = np.array([[8., 6], [3, 3], [10, 6], [11, 6]])
    points1 = np.array([[8., 6], [14, 10], [9, 6], [11, 6]])
    b.depth[6, 9] = 1.  # wrong candidate depth, while the projected source remains visible
    a.depth[6, 11] = 0
    summary, labels = match_metrics(a, b, points0, points1, np.column_stack([np.arange(4)]*2))
    assert summary["predicted"] == 4 and summary["evaluable"] == 3
    assert summary["correct"] == 1 and summary["precision"] == pytest.approx(1/3)
    assert summary["reasons"]["incorrect"] == 2
    assert list(labels["correct"]) == [True, False, False, False]
    b.depth[6, 8] = 1.  # projected foreground occlusion: unknown, not an automatic negative
    summary, _ = match_metrics(a, b, points0[:1], points1[:1], np.array([[0, 0]]))
    assert summary["evaluable"] == 0 and summary["precision"] is None
    b.T_world_camera = None
    summary, _ = match_metrics(a, b, points0, points1, np.array([[0, 0]]))
    assert summary["reasons"] == {"reference_pose_missing": 1}


def pnp_fixture():
    rng = np.random.default_rng(17)
    K = np.array([[300., 0, 320], [0, 300, 240], [0, 0, 1]])
    x, y = np.meshgrid(np.arange(150, 500, 50), np.arange(120, 380, 50))
    points = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    depths = rng.uniform(1.5, 4., len(points))
    depth_map = np.zeros((480, 640), dtype=np.float64)
    depth_map[points[:, 1].astype(int), points[:, 0].astype(int)] = depths
    xyz = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(K).T * depths[:, None]
    T = pose((.04, -.015, .02), (.03, -.02, .01))
    transformed = xyz @ T[:3, :3].T+T[:3, 3]
    projected = transformed @ K.T
    target = projected[:, :2]/projected[:, 2, None]
    target[-10:] += rng.uniform(80, 120, (10, 2))
    pairs = np.column_stack([np.arange(len(points))]*2)
    return points, target, pairs, depth_map, K, T


def test_fixed_pnp_recovers_known_transform_with_outliers():
    points, target, pairs, depth, K, expected = pnp_fixture()
    result = estimate_motion(points, target, pairs, depth, K, K)
    assert result["success"], result["reason"]
    np.testing.assert_allclose(result["T_camera1_camera0"], expected, atol=1e-6)
    assert result["inliers"] == len(points)-10
    assert not np.any(result["inlier_match_indices"] >= len(points)-10)
    repeated = estimate_motion(points, target, pairs, depth, K, K)
    np.testing.assert_array_equal(result["T_camera1_camera0"], repeated["T_camera1_camera0"])


def test_pnp_empty_insufficient_depth_and_explicit_failure(monkeypatch):
    points, target, pairs, depth, K, _ = pnp_fixture()
    for p in (pairs[:0], pairs[:5]):
        result = estimate_motion(points, target, p, depth, K, K)
        assert not result["success"] and result["T_camera1_camera0"] is None
    result = estimate_motion(points, target, pairs, np.zeros_like(depth), K, K)
    assert result["reason"] == "insufficient_depth_correspondences"
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda *a, **k: (False, None, None, None))
    assert estimate_motion(points, target, pairs, depth, K, K)["reason"] == "ransac_failed_or_too_few_inliers"
    def fail(*args, **kwargs):
        raise cv2.error("forced failure")
    monkeypatch.setattr(cv2, "solvePnPRansac", fail)
    assert estimate_motion(points, target, pairs, depth, K, K)["reason"] == "opencv_pnp_error"


def test_se3_alignment_and_rpe_are_rigid_invariant_without_scale_fit():
    refs = [pose((0, 0, 0)), pose((1, 0, 0), (0, 0, .2)), pose((1, 1, 0), (.1, 0, .3)),
            pose((0, 1, 1), (.2, .1, 0)), pose((0, 2, 1), (.1, -.1, .1))]
    offset = pose((3, -2, .7), (.3, -.2, .4))
    estimates = [offset @ T for T in refs]
    a = np.array([T[:3, 3] for T in estimates])
    b = np.array([T[:3, 3] for T in refs])
    aligned, rank = align_se3(a, b)
    assert rank == 3
    np.testing.assert_allclose(a @ aligned[:3, :3].T+aligned[:3, 3], b, atol=1e-12)
    assert np.linalg.det(aligned[:3, :3]) == pytest.approx(1.)
    for i in range(1, len(refs)):
        translation, rotation = relative_error(estimates[i-1], estimates[i], refs[i-1], refs[i])
        assert translation < 1e-12 and rotation < 2e-6
    aligned, _ = align_se3(2*b, b)
    assert np.linalg.norm(2*b @ aligned[:3, :3].T+aligned[:3, 3]-b) > .1


def test_rpe_known_translation_and_rotation_errors():
    t, r = relative_error(np.eye(4), pose((.2, 0, 0), (0, 0, np.pi/6)), np.eye(4), np.eye(4))
    assert t == pytest.approx(.2) and r == pytest.approx(30)


def test_segments_direction_failure_coverage_and_no_cross_gap_rpe():
    refs = [pose((0, 0, 0)), pose((1, 0, 0)), pose((1, 1, 0)), pose((2, 1, 0)), pose((2, 2, 0))]
    track = SegmentedOdometry()
    for i in range(4):
        motion = np.linalg.inv(refs[i+1]) @ refs[i]
        track.append(i, i+1, None if i == 2 else motion)
    assert len(track.segments) == 2
    np.testing.assert_allclose(track.segments[0][-1][1], refs[2])
    # The second origin is identity rather than the nonzero ground-truth position.
    np.testing.assert_array_equal(track.segments[1][0][1], np.eye(4))
    summary = track.summary(np.arange(5)*.1, refs)
    assert summary["full_trajectory_ate_m"] is None
    assert summary["failed_transitions"] == 1 and summary["tracking_success_rate"] == .75
    assert summary["frame_coverage"] == 1. and summary["longest_segment_frames"] == 3
    assert summary["rpe_translation_m"]["count"] == 3
    assert summary["rpe_translation_m"]["max"] < 1e-12
    assert summary["segments"][0]["within_segment_ate_m"]["rmse"] < 1e-12
    refs[1] = None
    assert track.summary(np.arange(5)*.1, refs)["rpe_translation_m"]["count"] == 1
    with pytest.raises(ValueError, match="increasing"):
        track.summary([0, .1, .1, .3, .4], refs)


def test_all_failed_is_zero_coverage_without_fake_identity_poses():
    track = SegmentedOdometry()
    track.append(0, 1, None)
    track.append(1, 2, None)
    summary = track.summary([1, 2, 3], [np.eye(4)]*3)
    assert summary["covered_frames"] == 0 and summary["segments"] == []
    assert summary["full_trajectory_ate_m"] is None and summary["rpe_translation_m"]["count"] == 0


def test_full_trajectory_ate_and_quaternion_export():
    refs = [pose((0, 0, 0)), pose((1, 0, 0)), pose((1, 1, 0)), pose((1, 1, 1))]
    track = SegmentedOdometry()
    for i in range(3):
        track.append(i, i+1, np.linalg.inv(refs[i+1]) @ refs[i])
    assert track.summary([0, 1, 2, 3], refs)["full_trajectory_ate_m"]["rmse"] < 1e-12
    for rotvec in ((0, 0, 0), (np.pi, 0, 0), (0, np.pi, 0), (.1, .2, -.3)):
        R = pose(rotvec=rotvec)[:3, :3]
        np.testing.assert_allclose(quaternion_matrix(matrix_quaternion(R)), R, atol=1e-12)


def test_stage3_test_guard_happens_before_audit_or_data_use(tmp_path):
    data, _, _, catalog, config_path = mini_data(tmp_path)
    catalog["sequences"][0]["split"] = "test"
    config_path.write_text(yaml.safe_dump(catalog))
    with pytest.raises(ValueError, match="held-out"):
        verify_development_data({"catalog": str(config_path), "sequences": ["mini"]}, data, tmp_path/"absent")


def test_development_data_rejects_mutation_after_audit(tmp_path):
    from scaf.data.cli import audit
    from scaf.data.catalog import fingerprint
    from types import SimpleNamespace
    data, directory, _, catalog, config_path = mini_data(tmp_path)
    output = tmp_path/"audit"
    output.mkdir()
    audit(catalog, SimpleNamespace(output=output, data_root=data))
    write_json(output/"manifest.json", {"status": "completed", "catalog_sha256": fingerprint(catalog),
                                       "summary_sha256": sha256(output/"summary.json")})
    config = {"catalog": str(config_path), "sequences": ["mini"]}
    assert verify_development_data(config, data, output/"manifest.json")[1]["verified_sequence_files"]["mini"] > 0
    (directory/"groundtruth.txt").write_text("changed")
    with pytest.raises(ValueError, match="dataset file changed"):
        verify_development_data(config, data, output/"manifest.json")


def test_timing_warmup_and_short_sample_limit():
    keys = ("extract_ms", "enhance_ms", "optical_flow_ms", "match_ms", "backend_ms", "frontend_ms",
            "pipeline_compute_ms", "load_ms", "degrade_ms", "input_plus_compute_ms")
    rows = [{key: float(i) for key in keys} for i in range(15)]
    summary = timing_summary(rows, 10, 100)
    assert summary["timed_unique_transitions"] == 5 and not summary["sample_count_sufficient"]
    assert summary["components"]["frontend_ms"]["p50"] == 12.
    assert timing_summary(rows[:2], 10, 100)["components"]["frontend_ms"]["p50"] is None


@pytest.mark.parametrize("key,value", [("device", "cuda"), ("threads", 2), ("max_frames", 1),
                                       ("max_keypoints", 0), ("methods", ["fake"]),
                                       ("variants", ["../escape"]), ("performance_benchmark", True)])
def test_acceptance_config_rejects_invalid_settings(tmp_path, key, value):
    config = yaml.safe_load((ROOT/"configs/stage3.yaml").read_text())
    config[key] = value
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_settings(path)


@pytest.mark.skipif(not os.environ.get("SCAF_BASELINE_ASSET_MANIFEST"), reason="official local assets not configured")
def test_official_models_shared_features_empty_inputs_and_normalization():
    import torch
    from scaf.baselines.adapters import BaselineModels, normalize_booster_keypoints
    from scaf.baselines.prepare import verify_assets
    config = yaml.safe_load((ROOT/"configs/baselines-v1.yaml").read_text())
    lock = verify_assets(Path(os.environ["SCAF_BASELINE_ASSET_MANIFEST"]), config)
    torch.set_num_threads(1)
    models = BaselineModels(lock, max_keypoints=64)
    # Corner-rich synthetic texture, rather than white noise (which a trained
    # detector can legitimately reject almost entirely).
    rgb = np.random.default_rng(21).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    rgb = cv2.GaussianBlur(cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_NEAREST), (5, 5), 1.)
    a = models.extract(rgb)
    assert 6 < len(a.points) <= 64
    original_desc = a.descriptors.copy()
    b = models.enhance("featurebooster", a)
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.scores, b.scores)
    np.testing.assert_array_equal(a.descriptors, original_desc)
    np.testing.assert_allclose(np.linalg.norm(b.descriptors, axis=1), 1, atol=1e-5)
    norm = normalize_booster_keypoints(a)
    np.testing.assert_allclose(norm[:, :2], (a.points-[160, 120])/(320*.7), atol=1e-7)
    np.testing.assert_array_equal(norm[:, 2], a.scores)
    for method in ("superpoint", "featurebooster", "lightglue"):
        f = b if method == "featurebooster" else a
        pairs, _ = models.match(method, f, f)
        assert len(pairs) > 6
        np.testing.assert_array_equal(pairs[:, 0], pairs[:, 1])
        empty = Features(np.empty((0, 2)), np.empty((0, 256)), np.empty(0), (320, 240))
        assert not len(models.match(method, empty, a)[0])
        assert models.enhance(method, empty).descriptors.shape == (0, 256)
    for model in (models.extractor, models.booster, models.lightglue):
        assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert not torch.cuda.is_initialized()
