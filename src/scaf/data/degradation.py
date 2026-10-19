"""Seeded synthetic sensor perturbations. Clean RGB-D remains available for labels."""
from dataclasses import asdict, dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class Degradation:
    gamma: float = 1.0
    gain: float = 1.0
    rgb_noise_std: float = 0.0  # normalized [0,1] intensity
    depth_dropout: float = 0.0
    depth_noise_std_m: float = 0.0
    depth_noise_relative: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("nonfinite degradation")
        if self.gamma <= 0 or not 0 <= self.gain <= 1 or not 0 <= self.depth_dropout <= 1:
            raise ValueError("invalid gamma/gain/dropout")
        if min(self.rgb_noise_std, self.depth_noise_std_m, self.depth_noise_relative) < 0:
            raise ValueError("negative noise")


def degrade(frame, config, seed, variant="default"):
    key = f"scaf-degradation-v1|{seed}|{variant}|{frame.identity}".encode()
    stable_seed = int.from_bytes(hashlib.sha256(key).digest()[:16], "little")
    # Independent streams: changing RGB settings does not change depth perturbations.
    rgb_seed, depth_seed = np.random.SeedSequence(stable_seed).spawn(2)
    rgb_rng, depth_rng = np.random.default_rng(rgb_seed), np.random.default_rng(depth_seed)
    rgb = config.gain*np.power(frame.rgb.astype(np.float32)/255., config.gamma)
    if config.rgb_noise_std:
        rgb += rgb_rng.normal(0, config.rgb_noise_std, rgb.shape)
    rgb = np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)
    depth = frame.depth.copy()
    valid = np.isfinite(depth) & (depth > 0)
    sigma = config.depth_noise_std_m + config.depth_noise_relative*np.where(valid, depth, 0)
    if np.any(sigma):
        depth[valid] += depth_rng.normal(size=depth.shape)[valid]*sigma[valid]
    drop = depth_rng.random(depth.shape) < config.depth_dropout
    depth[~valid | drop | ~np.isfinite(depth) | (depth < 0.2) | (depth > 8)] = 0
    return rgb, depth
