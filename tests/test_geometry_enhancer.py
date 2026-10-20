"""Analytic geometry, padding independence and trainability checks for stage 4."""
from dataclasses import replace
from pathlib import Path
import json
import os
import subprocess
import sys
import shutil

import numpy as np
import pytest
import torch
import yaml

from scaf.models.geometry import GeometryConfig, GeometryEnhancer, build_neighborhood
from scaf.models.diagnostic import diagnostic_loss, diagnostic_targets
from scaf.stage4 import load_settings, validate_roles
from test_data import frame, mini_data


ROOT = Path(__file__).resolve().parents[1]


def example(n=5, batch=1, config=None):
    cfg = config or GeometryConfig(descriptor_dim=8, hidden_dim=12, neighbors=3, radius_m=1.)
    points = torch.stack([torch.arange(n, dtype=torch.float32)+4, torch.full((n,), 6.)], -1)[None].repeat(batch, 1, 1)
    depth = torch.full((batch, 16, max(n+8, 20)), 2.)
    K = torch.tensor([[10., 0, 8], [0, 10, 6], [0, 0, 1]])[None].repeat(batch, 1, 1)
    context = build_neighborhood(points, depth, K, config=cfg)
    desc = torch.randn(batch, n, cfg.descriptor_dim)
    return cfg, points, depth, K, context, desc


def test_backprojection_reliability_and_radius_exclusion():
    cfg, points, depth, K, context, _ = example()
    torch.testing.assert_close(context.xyz[0, 0], torch.tensor([-.8, 0, 2.]))
    torch.testing.assert_close(context.reliability, torch.ones(1, 5))
    assert context.geometry_mask[0, 0].sum() == 3
    depth[0, :, 6:] = 5  # A discontinuity must not bridge the metric neighborhood.
    c = build_neighborhood(points, depth, K, config=cfg)
    assert 0 < c.reliability[0, 1] < 1
    for i in (0, 1):
        assert (c.geometry_indices[0, i][c.geometry_mask[0, i]] < 2).all()
    depth[0, 6, 4] = float("nan")
    c = build_neighborhood(points, depth, K, config=cfg)
    assert c.reliability[0, 0] == 0 and c.xyz[0, 0].count_nonzero() == 0
    assert not c.geometry_mask[0, 0].any()
    assert not (c.geometry_indices[c.geometry_mask] == 0).any()


@pytest.mark.parametrize("bad", [0., -1., .1, 9., float("nan"), float("inf")])
def test_invalid_depth_exactly_falls_back_to_appearance(bad):
    cfg, points, depth, K, _, desc = example()
    c = build_neighborhood(points, torch.full_like(depth, bad), K, config=cfg)
    model = GeometryEnhancer(cfg)
    result, info = model(desc, c, return_info=True)
    expected = model(desc, c, geometry=False)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert not info["geometry_gate"].any()
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(1, 5))


def test_padding_nan_does_not_change_valid_output_or_gradient():
    cfg, points, depth, K, c, desc = example(n=5)
    model = GeometryEnhancer(cfg)
    a = desc.clone().requires_grad_()
    y = model(a, c)
    y[..., 0].sum().backward()
    grad = a.grad.clone()
    padded_points = torch.cat([points, torch.full((1, 4, 2), float("nan"))], 1)
    padded_desc = torch.cat([desc, torch.full((1, 4, cfg.descriptor_dim), float("nan"))], 1).requires_grad_()
    mask = torch.arange(9)[None] < 5
    pc = build_neighborhood(padded_points, depth, K, mask=mask, config=cfg)
    py = model(padded_desc, pc)
    py[..., 0].sum().backward()
    torch.testing.assert_close(y, py[:, :5], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grad, padded_desc.grad[:, :5], atol=1e-6, rtol=1e-6)
    assert not py[:, 5:].any() and not padded_desc.grad[:, 5:].any()
    assert torch.isfinite(padded_desc.grad).all()


@pytest.mark.parametrize("n", [0, 1, 5])
def test_empty_singleton_and_all_padding_have_finite_backward(n):
    cfg, points, depth, K, _, desc = example(n=n, batch=2)
    mask = torch.zeros((2, n), dtype=torch.bool)
    c = build_neighborhood(points*float("nan"), depth, K, mask, cfg)
    model = GeometryEnhancer(cfg)
    x = (desc*float("nan")).requires_grad_()
    result = model(x, c)
    assert result.shape == (2, n, cfg.descriptor_dim)
    assert torch.isfinite(result).all() and not result.any()
    result.sum().backward()
    assert torch.isfinite(x.grad).all() and not x.grad.any()


def test_batch_and_permutation_equivariance_without_distance_ties():
    cfg, points, depth, K, _, desc = example(batch=2)
    points[:, :, 0] = torch.tensor([3., 4.1, 5.6, 8.4, 12.2])
    model = GeometryEnhancer(cfg)
    c = build_neighborhood(points, depth, K, config=cfg)
    y = model(desc, c)
    for b in range(2):
        single = build_neighborhood(points[b:b+1], depth[b:b+1], K[b:b+1], config=cfg)
        torch.testing.assert_close(y[b:b+1], model(desc[b:b+1], single))
    order = torch.tensor([3, 0, 4, 2, 1])
    permuted = build_neighborhood(points[:, order], depth, K, config=cfg)
    torch.testing.assert_close(y[:, order], model(desc[:, order], permuted), atol=1e-6, rtol=1e-6)


def test_absolute_reliability_attenuates_geometry_instead_of_cancelling():
    cfg, _, _, _, c, desc = example()
    model = GeometryEnhancer(cfg)
    # Enlarge just the geometry output for an unambiguous gating check.
    with torch.no_grad():
        model.geometry_head[-1].bias.fill_(.1)
    full, full_info = model(desc, c, return_info=True)
    low = replace(c, reliability=c.reliability*.1)
    reduced, reduced_info = model(desc, low, return_info=True)
    app = model(desc, c, geometry=False)
    torch.testing.assert_close(reduced_info["geometry_gate"], full_info["geometry_gate"]*.01)
    assert torch.norm(reduced-app) < torch.norm(full-app)*.02
    torch.testing.assert_close(model(desc, low, reliability=False), model(desc, c, reliability=False))
    torch.testing.assert_close(model(desc, low, geometry=False), app, rtol=0, atol=0)


def test_forward_backward_updates_both_branches_and_keeps_unit_descriptors():
    cfg, _, _, _, c, desc = example(n=8)
    model = GeometryEnhancer(cfg)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    target = torch.nn.functional.normalize(torch.randn_like(desc), dim=-1)
    for _ in range(3):
        optimizer.zero_grad()
        out = model(desc, c)
        loss = (out-target).square().sum()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert not torch.equal(initial["appearance_head.2.weight"], model.state_dict()["appearance_head.2.weight"])
    assert not torch.equal(initial["geometry_head.2.weight"], model.state_dict()["geometry_head.2.weight"])
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(1, 8))
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_exact_context_storage_and_no_self_neighbors():
    _, _, _, _, c, _ = example()
    tensors = [v for v in vars(c).values() if isinstance(v, torch.Tensor)]
    assert c.storage["context_tensor_bytes"] == sum(t.numel()*t.element_size() for t in tensors)
    assert c.storage["distance_matrix_bytes_each"] == 5*5*4
    for i in range(5):
        assert i not in c.appearance_indices[0, i][c.appearance_mask[0, i]]
    assert c.timings_ms["neighbor_search_ms"] >= 0


def test_diagnostic_negatives_ignore_unknown_and_close_correspondences():
    f = frame()
    points = np.array([[2., 3], [4., 3], [13., 9], [8., 6]])
    f.depth[6, 8] = 0
    targets = diagnostic_targets(f, f, points, points, negative_radius_px=8.)
    i, j, neg = targets
    assert i.tolist() == j.tolist() == [0, 1, 2]
    assert not neg[0, 1] and neg[0, 2] and not neg.diag().any()
    good = torch.eye(4, requires_grad=True)
    good_loss, stats = diagnostic_loss(good, good, targets)
    bad_loss, _ = diagnostic_loss(good, good[[2, 1, 0, 3]], targets)
    assert good_loss < bad_loss and stats["anchors"] == 6
    good_loss.backward()
    assert torch.isfinite(good.grad).all()
    # No positive / no trusted negative: a differentiable zero, never fake negatives.
    empty = (i[:0], j[:0], neg[:0, :0])
    zero, stats = diagnostic_loss(good, good, empty)
    assert zero == 0 and stats["anchors"] == 0
    zero.backward()
    zero, stats = diagnostic_loss(good, good, (i[:1], j[:1], neg[:1, :1]))
    assert zero == 0 and stats["anchors"] == 0


def test_stage4_split_guard_precedes_data_read(tmp_path):
    data, _, _, catalog, path = mini_data(tmp_path)
    config = {"catalog": str(path), "train_sequence": "mini", "validation_sequences": ["mini"]}
    with pytest.raises(ValueError, match="only trains"):
        validate_roles(config, data)
    catalog["sequences"][0]["split"] = "test"
    path.write_text(yaml.safe_dump(catalog))
    with pytest.raises(ValueError, match="held-out"):
        validate_roles(config, data)


@pytest.mark.parametrize("key,value", [("max_keypoints", 0), ("train_steps", 0), ("temperature", float("nan")),
    ("validation_frames", 1), ("validation_sequences", ["tum_fr1_xyz"]), ("variants", ["../test"]),
    ("negative_radius_px", 2), ("model", {"residual_scale": .9})])
def test_invalid_stage4_configuration(tmp_path, key, value):
    c = yaml.safe_load((ROOT/"configs/stage4.yaml").read_text())
    c[key] = value
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        load_settings(path)


def test_invalid_real_inputs_rejected():
    cfg, points, depth, K, c, desc = example()
    with pytest.raises(ValueError, match="intrinsics"):
        build_neighborhood(points, depth, torch.zeros_like(K), config=cfg)
    with pytest.raises(ValueError, match="outside"):
        build_neighborhood(points+100, depth, K, config=cfg)
    with pytest.raises(ValueError, match="nonfinite"):
        build_neighborhood(points*float("nan"), depth, K, config=cfg)
    with pytest.raises(ValueError, match="nonzero"):
        GeometryEnhancer(cfg)(torch.zeros_like(desc), c)


@pytest.mark.skipif(not os.environ.get("SCAF_BASELINE_ASSET_MANIFEST"), reason="official assets not configured")
def test_stage4_real_cli_split_failure_manifest_and_refuses_reuse(tmp_path):
    # Exercise the actual CLI's split guards and preservation of failed output.
    data, _, _, catalog, catalog_path = mini_data(tmp_path)
    validation = dict(catalog["sequences"][0], id="mini_val", scene="other", split="val", path="other")
    catalog["sequences"].append(validation)
    catalog_path.write_text(yaml.safe_dump(catalog))
    c = yaml.safe_load((ROOT/"configs/stage4.yaml").read_text())
    c.update(catalog=str(catalog_path), train_sequence="mini", validation_sequences=["mini_val"],
             train_pairs=1, train_pair_spacing=1, frame_stride=1, train_steps=1)
    cp = tmp_path/"stage4.yaml"
    cp.write_text(yaml.safe_dump(c))
    # Split misuse is refused before the missing audit can be opened.
    c["train_sequence"] = "mini_val"
    cp.write_text(yaml.safe_dump(c))
    output = tmp_path/"run"
    command = [sys.executable, "-m", "scaf.stage4", "--config", str(cp), "--data-root", str(data),
               "--audit-manifest", str(tmp_path/"missing"), "--asset-manifest",
               os.environ["SCAF_BASELINE_ASSET_MANIFEST"], "--output", str(output)]
    # Here the configuration guard runs before output creation.
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and not output.exists()
    c["train_sequence"] = "mini"
    # Now valid configuration, invalid roles (train reused as validation).
    c["validation_sequences"] = ["mini"]
    cp.write_text(yaml.safe_dump(c))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and not output.exists()
    # A catalog role error is recorded in a new output, without touching images.
    c["validation_sequences"] = ["mini_val"]
    catalog["sequences"][1]["split"] = "test"
    catalog_path.write_text(yaml.safe_dump(catalog))
    cp.write_text(yaml.safe_dump(c))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["status"] == "failed" and "held-out" in manifest["error"]
    before = (output/"manifest.json").read_bytes()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and (output/"manifest.json").read_bytes() == before
    # Valid train/val roles + audited, empty-feature RGB: refuse a no-update pass.
    catalog["sequences"][1]["split"] = "val"
    catalog_path.write_text(yaml.safe_dump(catalog))
    shutil.copytree(data/"sequences/mini", data/"other")
    audit = tmp_path/"audit"
    result = subprocess.run([sys.executable, "-m", "scaf.data.cli", "--catalog", str(catalog_path),
                             "--data-root", str(data), "audit", "--output", str(audit)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command[command.index("--audit-manifest")+1] = str(audit/"manifest.json")
    command[command.index("--output")+1] = str(tmp_path/"no_supervision")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    manifest = json.loads((tmp_path/"no_supervision/manifest.json").read_text())
    assert manifest["status"] == "failed" and "no geometrically certified" in manifest["error"]
    assert not (tmp_path/"no_supervision/geometry-final.pt").exists()
