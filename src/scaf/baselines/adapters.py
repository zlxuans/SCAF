"""Run official frozen models offline with verified local weights.

Only the three required LightGlue modules are imported; its package __init__ also
imports unrelated extractors. The original source files are never modified.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

from .matching import Features, mutual_ratio_match, validate_pairs


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_modules(lock):
    root = Path(lock["sources"]["lightglue"]["root"])/"lightglue"
    namespace = "_scaf_official_lightglue"
    if namespace in sys.modules and list(sys.modules[namespace].__path__) != [str(root)]:
        raise ValueError("cannot mix official LightGlue source roots in one process")
    package = ModuleType(namespace)
    package.__path__ = [str(root)]
    sys.modules[namespace] = package
    load_module(namespace+".utils", root/"utils.py")
    sp = load_module(namespace+".superpoint", root/"superpoint.py")
    lg = load_module(namespace+".lightglue", root/"lightglue.py")
    fb = load_module("_scaf_official_featurebooster",
                     Path(lock["sources"]["featurebooster"]["root"])/"featurebooster.py")
    return sp.SuperPoint, lg.LightGlue, fb.FeatureBooster


def load_weights(lock, name):
    return torch.load(lock["weights"][name]["path"], map_location="cpu", weights_only=True)


def freeze(model, device):
    return model.eval().requires_grad_(False).to(device)


def normalize_booster_keypoints(features):
    w, h = features.image_wh
    xy = (features.points-np.array([w/2, h/2], dtype=np.float32))/(max(w, h)*.7)
    # Official SuperPoint+Boost-F expects x, y, detector score, with score unscaled.
    return np.column_stack([xy, features.scores]).astype(np.float32)


class BaselineModels:
    def __init__(self, lock, max_keypoints=512, device="cpu", methods=None):
        self.device = torch.device(device)
        self.methods = methods or ["superpoint", "featurebooster", "lightglue"]
        self.config = lock["baseline_config"]
        SP, LG, FB = official_modules(lock)
        state = load_weights(lock, "superpoint")
        # Official SuperPoint's constructor uses torch.hub; route it to verified
        # local data so inference cannot fetch or replace unregistered weights.
        with patch("torch.hub.load_state_dict_from_url", return_value=state):
            self.extractor = freeze(SP(max_num_keypoints=max_keypoints, **self.config["superpoint"]), self.device)
        self.booster, self.lightglue = None, None
        if "featurebooster" in self.methods:
            config = yaml.safe_load((Path(lock["sources"]["featurebooster"]["root"])/"config.yaml").read_text())
            self.booster_config = config["SuperPoint+Boost-F"]
            self.booster = FB(self.booster_config)
            self.booster.load_state_dict(load_weights(lock, "featurebooster"), strict=True)
            self.booster = freeze(self.booster, self.device)
        if "lightglue" in self.methods:
            self.lightglue = LG(features=None, weights=None, **self.config["lightglue"])
            state = load_weights(lock, "lightglue")
            for i in range(self.config["lightglue"]["n_layers"]):
                for old, new in ((f"self_attn.{i}", f"transformers.{i}.self_attn"),
                                 (f"cross_attn.{i}", f"transformers.{i}.cross_attn")):
                    state = {k.replace(old, new): v for k, v in state.items()}
            missing, unexpected = self.lightglue.load_state_dict(state, strict=False)
            # Official released checkpoints retain one unused legacy `r` scalar
            # per assignment head; the pinned implementation no longer reads it.
            legacy = {f"log_assignment.{i}.r" for i in range(self.config["lightglue"]["n_layers"])}
            if set(missing)-{"confidence_thresholds"} or set(unexpected)-legacy:
                raise ValueError(f"LightGlue weight mismatch: missing={missing}, unexpected={unexpected}")
            self.lightglue_weight_compatibility = {"initialized_buffers": missing, "unused_legacy_keys": unexpected}
            self.lightglue = freeze(self.lightglue, self.device)

    def parameter_counts(self):
        return {name: sum(p.numel() for p in model.parameters()) if model is not None else 0
                for name, model in (("extractor", self.extractor), ("booster", self.booster),
                                    ("lightglue", self.lightglue))}

    @torch.inference_mode()
    def extract(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
        tensor = torch.from_numpy(gray)[None, None].to(self.device)
        # Call forward directly: no extractor.extract() auto-resize or coordinate rescaling.
        out = self.extractor({"image": tensor})
        w, h = rgb.shape[1], rgb.shape[0]
        return Features(out["keypoints"][0].cpu().numpy(), out["descriptors"][0].cpu().numpy(),
                        out["keypoint_scores"][0].cpu().numpy(), (w, h))

    @torch.inference_mode()
    def enhance(self, method, features):
        if method != "featurebooster" or not len(features.points):
            return features
        desc = torch.from_numpy(features.descriptors).to(self.device)
        points = torch.from_numpy(normalize_booster_keypoints(features)).to(self.device)
        result = self.booster(desc, points).cpu().numpy()
        return replace(features, descriptors=result)

    @torch.inference_mode()
    def match(self, method, a, b, ratio=.8):
        if method != "lightglue":
            return mutual_ratio_match(a.descriptors, b.descriptors, ratio)
        if not len(a.points) or not len(b.points):
            return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)
        def tensor(values):
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)[None]
        def pack(features):
            return {"keypoints": tensor(features.points), "descriptors": tensor(features.descriptors),
                    "image_size": tensor(features.image_wh)}
        out = self.lightglue({"image0": pack(a), "image1": pack(b)})
        pairs = out["matches"][0].cpu().numpy()
        return validate_pairs(pairs, len(a.points), len(b.points)), out["scores"][0].cpu().numpy()
