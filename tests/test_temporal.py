"""Analytic tracking/candidate separation and absolute weighted-gradient checks."""
from dataclasses import replace
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch
import yaml

from scaf.models.temporal import TemporalConfig, Tracks, TemporalContext, track_pair, candidate_reliability, filter_candidates
from scaf.models.losses import temporal_descriptor_loss
from scaf.evaluation.backend import estimate_motion
from scaf.evaluation.trajectory import SegmentedOdometry
from scaf.models.geometry import GeometryConfig, GeometryEnhancer
from scaf.stage5 import load_settings, validate_roles, initial_model
from scaf.runtime import sha256
from test_data import mini_data


ROOT = Path(__file__).resolve().parents[1]


def ideal_context(shift=(2., -1.)):
    p = np.array([[30., 30.], [50., 40.], [70., 60.]], np.float32)
    q = p+np.array(shift, np.float32)
    valid, error, reason = np.ones(3, bool), np.zeros(3, np.float32), np.full(3, "ok", dtype="U36")
    a = Tracks(p, q.copy(), p.copy(), valid.copy(), error.copy(), reason.copy())
    b = Tracks(q, p.copy(), q.copy(), valid.copy(), error.copy(), reason.copy())
    return TemporalContext(a, b, TemporalConfig(), {})


def test_consistent_tracking_does_not_certify_wrong_candidate():
    c = ideal_context()
    wrong = np.array([[0, 1], [1, 2], [2, 0]])
    evidence = candidate_reliability(c, wrong)
    assert c.forward.valid.all() and c.reverse.valid.all()
    assert not evidence["eligible"].any() and not evidence["weights"].any()
    assert set(evidence["reason"]) == {"candidate_inconsistent"}
    right = np.column_stack([np.arange(3)]*2)
    np.testing.assert_array_equal(candidate_reliability(c, right)["weights"], 1.)
    # Reverse candidate proximity must be checked separately, too.
    c.reverse.predicted[0] += 10
    assert candidate_reliability(c, right)["weights"][0] == 0


def test_weight_decreases_and_filter_preserves_order_and_disabled_identity():
    c = ideal_context()
    pairs = np.array([[2, 2], [0, 0], [1, 1]], dtype=np.int64)
    scores = np.array([.7, .8, .9], np.float32)
    c.forward.fb_error_px[1] = 1
    c.forward.predicted[1, 0] += 2
    evidence = candidate_reliability(c, pairs)
    assert 0 < evidence["weights"][2] < evidence["weights"][1]
    c = replace(c, config=replace(c.config, minimum_weight=.9))
    result = filter_candidates(c, pairs, scores)
    np.testing.assert_array_equal(result["pairs"], pairs[:2])
    np.testing.assert_array_equal(result["scores"], scores[:2])
    result = filter_candidates(c, pairs, scores, enabled=False)
    np.testing.assert_array_equal(result["pairs"], pairs)
    np.testing.assert_array_equal(result["scores"], scores)
    empty = filter_candidates(c, pairs[:0], scores[:0])
    assert empty["pairs"].shape == (0, 2) and not len(empty["scores"])
    with pytest.raises(ValueError, match="one-to-one"):
        candidate_reliability(c, np.array([[0, 0], [0, 1]]))


def texture():
    small = np.random.default_rng(9).integers(0, 256, (24, 32), np.uint8)
    return cv2.GaussianBlur(cv2.resize(small, (320, 240), interpolation=cv2.INTER_NEAREST), (5, 5), .8)


def test_real_lk_translation_and_padding_isolation():
    a = texture()
    shift = np.array([3., -2.], np.float32)
    b = cv2.warpAffine(a, np.array([[1., 0, 3.], [0, 1., -2.]]), (320, 240))
    p = cv2.goodFeaturesToTrack(a, 40, .05, 12).reshape(-1, 2)
    p = p[(p[:, 0] > 30) & (p[:, 0] < 280) & (p[:, 1] > 30) & (p[:, 1] < 210)]
    c = track_pair(a, b, p, p+shift)
    assert c.forward.valid.mean() > .8
    np.testing.assert_allclose(c.forward.predicted[c.forward.valid], (p+shift)[c.forward.valid], atol=.1)
    pairs = np.column_stack([np.arange(len(p))]*2)
    assert candidate_reliability(c, pairs)["weights"].mean() > .8
    pp = np.concatenate([p, [[np.nan, np.inf], [1, 1]]]).astype(np.float32)
    mask = np.arange(len(pp)) < len(p)
    padded = track_pair(a, b, pp, pp+shift, mask0=mask, mask1=mask)
    np.testing.assert_array_equal(c.forward.predicted, padded.forward.predicted[:len(p)])
    assert not padded.forward.valid[-2:].any()
    assert list(padded.forward.reason[-2:]) == ["padding", "padding"]
    assert c.timing_ms["optical_flow_ms"] >= 0


def test_blank_empty_boundary_and_opencv_failures(monkeypatch):
    blank = np.zeros((80, 100), np.uint8)
    p = np.array([[30, 30], [1, 1], [np.nan, 2]], np.float32)
    c = track_pair(blank, blank, p, p)
    assert not c.forward.valid.any()
    assert list(c.forward.reason) == ["forward_failed", "source_invalid_or_boundary", "source_invalid_or_boundary"]
    assert not candidate_reliability(c, np.array([[0, 0]]))["weights"].any()
    empty = track_pair(blank, blank, p[:0], p[:0])
    assert empty.forward.predicted.shape == (0, 2)
    def fail(*args, **kwargs):
        raise cv2.error("forced")
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fail)
    assert track_pair(blank, blank, p[:1], p[:1]).forward.reason[0] == "opencv_forward_failure"


@pytest.mark.parametrize("mode", ["forward_status", "forward_bounds", "back_status", "back_bounds", "fb", "nan"])
def test_track_rejects_failure_occlusion_proxies_and_bounds(monkeypatch, mode):
    image = np.zeros((100, 100), np.uint8)
    p = np.array([[30., 30.]], np.float32)
    calls = []
    def fake(a, b, points, *args, **kwargs):
        forward = len(calls) % 2 == 0
        calls.append(1)
        value = p.copy()
        status = np.ones((1, 1), np.uint8)
        if forward and mode == "forward_status" or not forward and mode == "back_status":
            status[:] = 0
        if forward and mode == "forward_bounds" or not forward and mode == "back_bounds":
            value[:] = [1, 1]
        if not forward and mode == "fb":
            value += 5
        if forward and mode == "nan":
            value[:] = np.nan
        return value[:, None], status, np.zeros((1, 1), np.float32)
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake)
    c = track_pair(image, image, p, p)
    assert not c.forward.valid.any()


def loss_example():
    torch.manual_seed(11)
    a = torch.randn(4, 8, requires_grad=True)
    b = torch.randn(4, 8, requires_grad=True)
    pairs = torch.column_stack([torch.arange(4)]*2)
    negative = ~torch.eye(4, dtype=torch.bool)
    return a, b, pairs, negative


def test_absolute_weight_scales_loss_and_gradient_and_detaches_weights():
    a, b, pairs, negative = loss_example()
    w = torch.ones(4, requires_grad=True)
    full, _ = temporal_descriptor_loss(a, b, pairs, negative, weights=w)
    full.backward()
    gradients = (a.grad.clone(), b.grad.clone())
    a.grad = b.grad = None
    low, _ = temporal_descriptor_loss(a, b, pairs, negative, weights=w*.1)
    low.backward()
    torch.testing.assert_close(low, full*.1)
    torch.testing.assert_close(a.grad, gradients[0]*.1)
    torch.testing.assert_close(b.grad, gradients[1]*.1)
    assert w.grad is None
    a.grad = b.grad = None
    zero, stats = temporal_descriptor_loss(a, b, pairs, negative, weights=torch.zeros(4))
    zero.backward()
    assert zero == 0 and stats["weighted_anchors"] == 0
    assert not a.grad.any() and not b.grad.any()


def test_low_reliability_anchor_contributes_less_with_fixed_denominator():
    a, b, pairs, negative = loss_example()
    none, _ = temporal_descriptor_loss(a, b, pairs, negative, weights=torch.zeros(4))
    one = torch.tensor([1., 0, 0, 0])
    high, _ = temporal_descriptor_loss(a, b, pairs, negative, weights=one)
    low, _ = temporal_descriptor_loss(a, b, pairs, negative, weights=one*.01)
    assert high > low > none
    torch.testing.assert_close(low, high*.01)


def test_loss_discriminates_and_unknown_negatives_are_not_invented():
    a = torch.eye(3, requires_grad=True)
    pairs = torch.column_stack([torch.arange(3)]*2)
    negative = ~torch.eye(3, dtype=torch.bool)
    good, _ = temporal_descriptor_loss(a, a, pairs, negative)
    bad, _ = temporal_descriptor_loss(a, a[[1, 2, 0]], pairs, negative)
    collapsed, _ = temporal_descriptor_loss(torch.ones(3, 3), torch.ones(3, 3), pairs, negative)
    assert good < collapsed < bad
    for pp, nn in ((pairs[:0], negative[:0, :0]), (pairs[:1], negative[:1, :1]), (pairs, torch.zeros_like(negative))):
        loss, info = temporal_descriptor_loss(a, a, pp, nn)
        assert loss == 0 and info["anchors"] == 0
        loss.backward()


def test_loss_padding_nan_masks_and_duplicate_rejection():
    a, b, pairs, negative = loss_example()
    expected, _ = temporal_descriptor_loss(a, b, pairs, negative)
    aa = torch.cat([a.detach(), torch.full((2, 8), np.nan)]).requires_grad_()
    bb = torch.cat([b.detach(), torch.full((2, 8), np.nan)]).requires_grad_()
    pp = torch.cat([pairs, torch.tensor([[4, 4], [-1, -1], [0, 0]])])
    nn = torch.zeros(7, 7, dtype=torch.bool)
    nn[:4, :4] = negative
    w = torch.tensor([1., 1, 1, 1, np.nan, np.nan, np.nan])
    result, _ = temporal_descriptor_loss(aa, bb, pp, nn, weights=w,
        mask0=torch.arange(6) < 4, mask1=torch.arange(6) < 4, positive_mask=torch.arange(7) < 5)
    torch.testing.assert_close(result, expected)
    result.backward()
    assert torch.isfinite(aa.grad).all() and not aa.grad[4:].any()
    with pytest.raises(ValueError, match="one-to-one"):
        temporal_descriptor_loss(a, b, torch.tensor([[0, 0], [0, 1]]), torch.zeros(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="also be a negative"):
        temporal_descriptor_loss(a, b, pairs, torch.ones(4, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="bounds"):
        temporal_descriptor_loss(a, b, torch.tensor([[-1, 0]]), torch.zeros(1, 1, dtype=torch.bool))


def test_filter_all_rejections_are_real_backend_failure_not_fallback():
    c = ideal_context()
    c.forward.valid[:] = False
    pairs = np.column_stack([np.arange(3)]*2)
    filtered = filter_candidates(c, pairs, np.ones(3, np.float32))
    assert not len(filtered["pairs"])
    motion = estimate_motion(c.forward.origins, c.reverse.origins, filtered["pairs"], np.ones((100, 100)), np.eye(3), np.eye(3))
    assert not motion["success"]
    tracker = SegmentedOdometry()
    tracker.append(0, 1, motion["T_camera1_camera0"])
    summary = tracker.summary([0., .1], [np.eye(4)]*2)
    assert summary["failed_transitions"] == 1 and summary["frame_coverage"] == 0


@pytest.mark.parametrize("kwargs", [{"window": 20}, {"border_px": 0}, {"minimum_weight": 0}, {"fb_max_px": np.nan}, {"candidate_sigma_px": -1}])
def test_invalid_temporal_configuration(kwargs):
    with pytest.raises(ValueError):
        TemporalConfig(**kwargs)


def test_weighted_loss_trains_real_enhancer_with_finite_gradients():
    from test_geometry_enhancer import example
    cfg, _, _, _, context, desc = example(n=5)
    model = GeometryEnhancer(cfg)
    initial = deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    pairs = torch.column_stack([torch.arange(5)]*2)
    negatives = ~torch.eye(5, dtype=torch.bool)
    a, b = model(desc, context)[0], model(desc+torch.randn_like(desc)*.1, context)[0]
    loss, _ = temporal_descriptor_loss(a, b, pairs, negatives, weights=torch.tensor([0., .1, .3, .8, 1.]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("key,value", [("train_steps", 0), ("timing_warmup_pairs", -1), ("consistency_weight", np.nan),
    ("negative_radius_px", 3), ("max_keypoints", 1024), ("variants", ["../escape"]),
    ("validation_sequences", ["tum_fr1_xyz"]), ("temporal", {"fb_sigma_px": 0})])
def test_invalid_stage5_configuration(tmp_path, key, value):
    cfg = yaml.safe_load((ROOT/"configs/stage5.yaml").read_text())
    cfg[key] = value
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_settings(path)


def test_stage5_refuses_test_role_before_images(tmp_path):
    data, _, _, catalog, path = mini_data(tmp_path)
    catalog["sequences"][0]["split"] = "test"
    path.write_text(yaml.safe_dump(catalog))
    with pytest.raises(ValueError, match="held-out"):
        validate_roles({"catalog": str(path), "train_sequence": "mini", "validation_sequences": ["mini"]}, data)


def test_stage5_checkpoint_requires_matching_completed_record(tmp_path):
    cfg = GeometryConfig(descriptor_dim=8, hidden_dim=12)
    model = GeometryEnhancer(cfg)
    checkpoint = tmp_path/"geometry-final.pt"
    from dataclasses import asdict
    torch.save({"model_config": asdict(cfg), "model": model.state_dict(), "kind": "stage4_debug_not_formal_training"}, checkpoint)
    manifest = {"kind": "stage4_tiny_training_validation_debug", "status": "completed", "test_used": False,
                "source": {"dirty": False, "commit": "test"}, "training": {"checkpoint_sha256": sha256(checkpoint)}}
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    loaded, info = initial_model(checkpoint, cfg)
    assert info["sha256"] == sha256(checkpoint)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, loaded.state_dict()[k])
    with pytest.raises(ValueError, match="configuration"):
        initial_model(checkpoint, GeometryConfig())
    manifest["training"]["checkpoint_sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="verified"):
        initial_model(checkpoint, cfg)


def test_empty_feature_evaluation_keeps_failures_for_all_switches(tmp_path):
    from scaf.baselines.matching import Features
    from scaf.data.degradation import Degradation
    from scaf.stage5 import evaluate
    data, _, spec, _, _ = mini_data(tmp_path)
    class EmptyExtractor:
        def extract(self, rgb):
            return Features(np.empty((0, 2)), np.empty((0, 256)), np.empty(0), (rgb.shape[1], rgb.shape[0]))
    cfg = load_settings(ROOT/"configs/stage5.yaml")
    cfg.update(validation_frames=3, frame_stride=1)
    model = GeometryEnhancer(GeometryConfig(**cfg["model"])).eval()
    trained = {"control": deepcopy(model), "temporal": deepcopy(model)}
    protocol = yaml.safe_load((ROOT/"configs/protocol.yaml").read_text())
    results = evaluate(trained, EmptyExtractor(), spec, "clean", Degradation(), data, cfg, protocol, TemporalConfig(), tmp_path/"eval")
    assert len(results) == 5
    for row in results:
        assert row["trajectory"]["failed_transitions"] == 2 and row["trajectory"]["frame_coverage"] == 0
        assert row["quality"]["precision"] is None and row["quality"]["predicted"] == 0
        assert row["trajectory"]["full_trajectory_ate_m"] is None


@pytest.mark.skipif(not os.environ.get("SCAF_BASELINE_ASSET_MANIFEST"), reason="official assets not configured")
def test_stage5_cli_preserves_split_failure_manifest_and_refuses_reuse(tmp_path):
    data, _, spec, catalog, path = mini_data(tmp_path)
    catalog["sequences"].append(dict(spec, id="val", path="other", scene="other", split="test"))
    path.write_text(yaml.safe_dump(catalog))
    cfg = yaml.safe_load((ROOT/"configs/stage5.yaml").read_text())
    cfg.update(catalog=str(path), train_sequence="mini", validation_sequences=["val"])
    cp = tmp_path/"config.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    output = tmp_path/"run"
    command = [sys.executable, "-m", "scaf.stage5", "--config", str(cp), "--data-root", str(data),
               "--audit-manifest", str(tmp_path/"missing"), "--asset-manifest", os.environ["SCAF_BASELINE_ASSET_MANIFEST"],
               "--geometry-checkpoint", str(tmp_path/"missing.pt"), "--output", str(output)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["status"] == "failed" and "held-out" in manifest["error"]
    before = (output/"manifest.json").read_bytes()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and (output/"manifest.json").read_bytes() == before
