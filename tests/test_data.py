"""Independent analytic geometry and miniature on-disk RGB-D integration checks."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import zipfile

import cv2
import numpy as np
import pytest
import yaml

from scaf.data.catalog import confined, fingerprint, load_catalog
from scaf.data.degradation import Degradation, degrade
from scaf.data.geometry import correspondences, interpolate_pose, sample_depth
from scaf.data.prepare import extract_sequence
from scaf.data.sequence import Frame, RGBDSequence, associate, read_rows
from scaf.runtime import sha256


ROOT = Path(__file__).resolve().parents[1]


def frame(depth=None, pose=None):
    return Frame(np.full((12, 16, 3), 128, np.uint8),
                 np.full((12, 16), 2., np.float32) if depth is None else depth,
                 np.array([[10., 0, 8], [0, 10., 6], [0, 0, 1]]),
                 np.eye(4) if pose is None else pose, 1., 1., "analytic/1")


def mini_data(tmp_path):
    data = tmp_path / "data"
    seqdir = data / "sequences" / "mini"
    (seqdir / "rgb").mkdir(parents=True)
    (seqdir / "depth").mkdir()
    for name in ("1.000", "1.020", "1.040"):
        cv2.imwrite(str(seqdir / "rgb" / f"{name}.png"), np.full((12, 16, 3), (10, 20, 30), np.uint8))
        depth = np.full((12, 16), 10000, np.uint16)
        depth[0, 0] = 0
        cv2.imwrite(str(seqdir / "depth" / f"{name}.png"), depth)
    for kind in ("rgb", "depth"):
        (seqdir / f"{kind}.txt").write_text("\n".join(f"{t} {kind}/{t}.png" for t in ("1.000", "1.020", "1.040")))
    (seqdir / "groundtruth.txt").write_text("1.000 0 0 0 0 0 0 1\n1.020 0.1 0 0 0 0 0 1\n1.040 0.2 0 0 0 0 0 1\n")
    spec = {"id": "mini", "dataset": "tum", "scene": "scene-a", "split": "train",
            "path": "sequences/mini", "illumination": "normal",
            "camera": {"source": "tum_registered_ros_default", "intrinsics": [10, 10, 8, 6]},
            "depth_units_per_metre": 5000., "pose_convention": "T_world_camera_xyzw_metres",
            "groundtruth_source": "analytic fixture", "archives": []}
    catalog = {"version": "fixture-v1", "sequences": [spec]}
    config = tmp_path / "datasets.yaml"
    config.write_text(yaml.safe_dump(catalog))
    return data, seqdir, spec, catalog, config


def test_scene_sequence_and_directory_alias_leakage(tmp_path):
    data, _, spec, catalog, config = mini_data(tmp_path)
    assert load_catalog(config, data) == catalog
    b = deepcopy(spec)
    b.update(id="other", path="sequences/other", split="test")
    catalog["sequences"].append(b)
    config.write_text(yaml.safe_dump(catalog))
    with pytest.raises(ValueError, match="scene leakage"):
        load_catalog(config, data)
    b.update(scene="other", path=spec["path"])
    config.write_text(yaml.safe_dump(catalog))
    with pytest.raises(ValueError, match="overlapping"):
        load_catalog(config, data)
    b["path"] = "sequences/alias"
    (data / b["path"]).symlink_to(data / spec["path"], target_is_directory=True)
    config.write_text(yaml.safe_dump(catalog))
    with pytest.raises(ValueError, match="overlapping"):
        load_catalog(config, data)


@pytest.mark.parametrize("name", ["../escape.png", "/tmp/escape.png", "."])
def test_dataset_path_confinement(tmp_path, name):
    with pytest.raises(ValueError, match="escapes"):
        confined(tmp_path, name)


def test_timestamp_association_no_reuse_and_closed_threshold(tmp_path):
    rgb = [(0., "a"), (.01, "b"), (.04, "c")]
    depth = [(.001, "x"), (.02, "y")]
    pairs = associate(rgb, depth, .02)
    assert pairs == [(0, 0), (1, 1)]
    assert associate([(0., "a")], [(.02, "b")], .02) == [(0, 0)]
    path = tmp_path / "index.txt"
    path.write_text("1 a\n1 b\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        read_rows(path, 2)


def test_pose_slerp_direction_gap_and_antipodal():
    # Halfway between identity and 180 degrees about z gives +90 degrees.
    rows = np.array([[0, 0, 0, 0, 0, 0, 0, 1], [.04, 2, 0, 0, 0, 0, 1, 0.]])
    T = interpolate_pose(rows, .02)
    np.testing.assert_allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1, 0, 0])
    assert interpolate_pose(rows, -.1) is None
    assert interpolate_pose(rows, .1) is None
    assert interpolate_pose(rows, .02, .01) is None
    rows[1, 4:] = [0, 0, 0, -1]
    np.testing.assert_allclose(interpolate_pose(rows, .02)[:3, :3], np.eye(3))


def test_registered_loader_units_rgb_and_pixel_center_resize(tmp_path):
    data, _, spec, _, _ = mini_data(tmp_path)
    seq = RGBDSequence(spec, data)
    a = seq.load(0)
    np.testing.assert_equal(a.rgb[1, 1], [30, 20, 10])
    assert a.depth[1, 1] == 2 and a.depth[0, 0] == 0
    half = seq.load(0, (8, 6))
    np.testing.assert_allclose(half.K, [[5, 0, 3.75], [0, 5, 2.75], [0, 0, 1]])
    assert half.depth.shape == (6, 8) and half.depth[0, 0] == 0
    assert seq.load(0, (1280, 960)).rgb.shape == a.rgb.shape  # never upscale


def test_missing_pose_is_explicit_and_bad_depth_fails(tmp_path):
    data, directory, spec, _, _ = mini_data(tmp_path)
    (directory / "groundtruth.txt").unlink()
    a = RGBDSequence(spec, data).load(0)
    assert a.T_world_camera is None
    with pytest.raises(ValueError, match="reference poses"):
        correspondences(a, a, [[1, 1]], [[1, 1]])
    cv2.imwrite(str(directory / "depth/1.000.png"), np.ones((12, 16), np.uint8))
    with pytest.raises(ValueError, match="uint16"):
        RGBDSequence(spec, data).load(0)


def test_analytic_translation_transform_and_unique_targets():
    a, b = frame(), frame()
    b.T_world_camera = np.eye(4)
    b.T_world_camera[0, 3] = .2  # camera moves right -> projection moves left by 1 pixel
    result = correspondences(a, b, [[8, 6], [8.1, 6], [10, 6]], [[7, 6], [9, 6]], radius=.2)
    np.testing.assert_allclose(result["projected"], [[7, 6], [7.1, 6], [9, 6]])
    np.testing.assert_equal(result["matches0"], [0, -1, 1])
    assert result["reason"][1] == "duplicate_target"
    assert result["T_camera1_camera0"][0, 3] == -.2


def test_visibility_occlusion_disocclusion_invalid_and_empty():
    a, b = frame(), frame()
    b.depth[6, 8] = 1  # foreground occlusion
    b.depth[6, 9] = 3  # incompatible background/disocclusion
    b.depth[6, 10] = 0
    a.depth[6, 11] = 0
    points = [[8, 6], [9, 6], [10, 6], [11, 6], [-1, 6], [float("nan"), 1]]
    result = correspondences(a, b, points, points)
    assert (result["matches0"] == -1).all()
    assert list(result["reason"][:3]) == ["target_depth_inconsistent", "target_depth_inconsistent", "target_bounds_or_depth"]
    assert correspondences(a, b, [], [])["matches0"].shape == (0,)
    b.T_world_camera[2, 3] = 3
    assert correspondences(a, b, [[7, 6]], [[7, 6]])["reason"][0] == "invalid_source_or_behind"


def test_candidate_depth_boundary_and_nonfinite_sampling():
    a, b = frame(), frame()
    b.depth[6, 9] = 1
    result = correspondences(a, b, [[8, 6]], [[9, 6]], radius=2)
    assert result["matches0"][0] == -1  # projected depth alone cannot certify candidate
    values, valid = sample_depth(b.depth, [[-0.1, 0], [15.1, 0], [np.inf, 0], [8, 6]])
    np.testing.assert_equal(valid, [False, False, False, True])


def test_seeded_degradation_keeps_clean_geometry_and_invalid_depth():
    a = frame()
    a.depth[0, 0] = 0
    clean_rgb, clean_depth = a.rgb.copy(), a.depth.copy()
    c = Degradation(gamma=2, gain=.4, rgb_noise_std=.02, depth_dropout=.3, depth_noise_std_m=.01)
    first = degrade(a, c, 13)
    again = degrade(a, c, 13)
    other = degrade(a, c, 29)
    for x, y in zip(first, again):
        np.testing.assert_equal(x, y)
    assert not np.array_equal(first[0], other[0])
    assert first[1][0, 0] == 0 and first[0].mean() < clean_rgb.mean()
    np.testing.assert_equal(a.rgb, clean_rgb)
    np.testing.assert_equal(a.depth, clean_depth)
    assert not degrade(a, Degradation(depth_dropout=1), 13)[1].any()
    assert not np.any(degrade(a, Degradation(), 13)[0] != a.rgb)


@pytest.mark.parametrize("kwargs", [{"gamma": 0}, {"depth_dropout": 1.1}, {"gain": np.nan}, {"depth_noise_std_m": -1}])
def test_degradation_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Degradation(**kwargs)


def test_archive_hash_path_traversal_and_overwrite(tmp_path):
    data = tmp_path / "data"
    (data / "downloads").mkdir(parents=True)
    archive = data / "downloads" / "mini.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escaped", "bad")
    spec = {"path": "sequences/mini", "archives": [{"file": "mini.zip", "sha256": sha256(archive)}]}
    with pytest.raises(ValueError, match="escapes"):
        extract_sequence(spec, data)
    assert not (tmp_path / "escaped").exists()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("mini/file.txt", "correct")
    with pytest.raises(ValueError, match="SHA256"):
        extract_sequence(spec, data)
    spec["archives"][0]["sha256"] = sha256(archive)
    destination = extract_sequence(spec, data)
    assert (destination / "file.txt").read_text() == "correct"
    with pytest.raises(FileExistsError):
        extract_sequence(spec, data)


def test_cpu_cli_audit_fingerprint_and_test_export_guard(tmp_path):
    data, _, spec, catalog, config = mini_data(tmp_path)
    audit = tmp_path / "audit"
    base = [sys.executable, "-m", "scaf.data.cli", "--catalog", str(config), "--data-root", str(data)]
    result = subprocess.run(base+["audit", "--output", str(audit)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((audit / "summary.json").read_text())
    assert summary["catalog_sha256"] == fingerprint(catalog)
    assert summary["sequences"][0]["png_decoded"] == 6
    repeat = subprocess.run(base+["audit", "--output", str(audit)], cwd=ROOT, capture_output=True)
    assert repeat.returncode != 0
    assert json.loads((audit / "manifest.json").read_text())["status"] == "completed"
    spec["split"] = "test"
    config.write_text(yaml.safe_dump(catalog))
    result = subprocess.run(base+["preview", "--sequence", "mini", "--output", str(tmp_path / "qa"),
                                 "--audit-manifest", str(audit / "manifest.json")], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode != 0 and "held-out" in result.stderr


def test_cli_development_export_requires_matching_audit_and_preserves_failures(tmp_path):
    data, directory, spec, catalog, config = mini_data(tmp_path)
    audit = tmp_path / "audit"
    base = [sys.executable, "-m", "scaf.data.cli", "--catalog", str(config), "--data-root", str(data)]
    subprocess.run(base+["audit", "--output", str(audit)], cwd=ROOT, check=True, capture_output=True)
    qa = ["preview", "--sequence", "mini", "--pairs", "1", "--stride", "1", "--audit-manifest", str(audit / "manifest.json")]
    result = subprocess.run(base+qa+["--output", str(tmp_path / "qa")], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "qa" / "pair-000000.npz").exists()
    # Mutation after the audit must fail before emitting labels.
    (directory / "groundtruth.txt").write_text("1.000 0 0 0 0 0 0 1\n")
    result = subprocess.run(base+qa+["--output", str(tmp_path / "changed")], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0 and "dataset file changed" in result.stderr
    assert json.loads((tmp_path / "changed" / "manifest.json").read_text())["status"] == "failed"
