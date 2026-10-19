"""Deterministic, mutual nearest neighbours with a two-sided Euclidean ratio test."""
from dataclasses import dataclass

import numpy as np


def unit_descriptors(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("descriptors must be a finite N x D matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("zero descriptor cannot be L2 normalized")
    return values / np.maximum(norms, 1e-8)


@dataclass
class Features:
    points: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    image_wh: tuple

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32)
        self.scores = np.asarray(self.scores, dtype=np.float32)
        self.descriptors = unit_descriptors(self.descriptors)
        n = len(self.descriptors)
        if self.points.shape != (n, 2) or self.scores.shape != (n,):
            raise ValueError("inconsistent feature shapes")
        if not np.isfinite(self.points).all() or not np.isfinite(self.scores).all():
            raise ValueError("nonfinite features")
        w, h = self.image_wh
        if w <= 0 or h <= 0 or np.any(self.points < 0) or np.any(self.points > [w-1, h-1]):
            raise ValueError("keypoints outside image")


def mutual_ratio_match(descriptors0, descriptors1, ratio=0.8):
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between zero and one")
    a, b = unit_descriptors(descriptors0), unit_descriptors(descriptors1)
    if a.shape[1] != b.shape[1]:
        raise ValueError("descriptor dimensions differ")
    if min(len(a), len(b)) < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)
    # Do not compare squared distances with an unsquared ratio threshold.
    squared = np.maximum(0., np.sum(a*a, 1)[:, None] + np.sum(b*b, 1)[None] - 2*(a @ b.T))
    distances = np.sqrt(squared)
    forward = np.argsort(distances, axis=1, kind="stable")[:, :2]
    backward = np.argsort(distances, axis=0, kind="stable")[:2, :]
    i = np.arange(len(a))
    j = forward[:, 0]
    d0, d1 = distances[i, j], distances[i, forward[:, 1]]
    r0, r1 = distances[backward[0], np.arange(len(b))], distances[backward[1], np.arange(len(b))]
    keep = (backward[0, j] == i) & (d0 < ratio*d1) & (r0[j] < ratio*r1[j])
    pairs = np.column_stack([i[keep], j[keep]]).astype(np.int64)
    return pairs, (1.-d0[keep]/2.).astype(np.float32)


def validate_pairs(pairs, count0, count1):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("matches must be an integer M x 2 matrix")
    if len(pairs) and (np.any(pairs < 0) or np.any(pairs >= [count0, count1])):
        raise ValueError("match index out of bounds")
    if any(len(np.unique(pairs[:, k])) != len(pairs) for k in (0, 1)):
        raise ValueError("matches must be one-to-one")
    return pairs.astype(np.int64, copy=False)
