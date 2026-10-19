"""Explicit sequence identities and scene-isolated, versioned splits."""
from pathlib import Path
import hashlib
import json

import numpy as np
import yaml


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def confined(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or path == root or root not in path.parents:
        raise ValueError(f"path escapes dataset root: {name}")
    return path


def load_catalog(path, data_root):
    catalog = yaml.safe_load(Path(path).read_text())
    if set(catalog) != {"version", "sequences"} or not catalog["version"]:
        raise ValueError("catalog needs version and sequences")
    scenes, paths, identifiers = {}, set(), set()
    required = {"id", "dataset", "scene", "split", "path", "illumination", "camera",
                "depth_units_per_metre", "pose_convention", "groundtruth_source", "archives"}
    for seq in catalog["sequences"]:
        if set(seq) != required:
            raise ValueError(f"unexpected/missing sequence keys: {set(seq) ^ required}")
        if seq["id"] in identifiers or not seq["id"] or "/" in seq["id"]:
            raise ValueError("duplicate/invalid sequence ID")
        identifiers.add(seq["id"])
        if seq["split"] not in {"train", "val", "test"}:
            raise ValueError("invalid split")
        if not seq["scene"] or scenes.setdefault(seq["scene"], seq["split"]) != seq["split"]:
            raise ValueError("scene leakage between splits")
        resolved = confined(data_root, seq["path"])
        if any(p == resolved or p in resolved.parents or resolved in p.parents for p in paths):
            raise ValueError("aliased/overlapping sequence directories")
        paths.add(resolved)
        if seq["dataset"] not in {"tum", "eth3d"}:
            raise ValueError("unsupported dataset")
        if seq["pose_convention"] != "T_world_camera_xyzw_metres":
            raise ValueError("unsupported pose convention")
        scale = seq["depth_units_per_metre"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("invalid depth scale")
        for archive in seq["archives"]:
            if set(archive) != {"file", "sha256", "url"} or len(archive["sha256"]) != 64:
                raise ValueError("archive needs filename, SHA256 and source URL")
            confined(data_root, "downloads/" + archive["file"])
    return catalog


def sequence_by_id(catalog, identifier):
    return next(s for s in catalog["sequences"] if s["id"] == identifier)
