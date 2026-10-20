"""Sparse image-only temporal reliability; no static-object or visibility guarantee."""
from dataclasses import dataclass, fields
from time import perf_counter
import math

import cv2
import numpy as np

from scaf.baselines.matching import validate_pairs


@dataclass(frozen=True)
class TemporalConfig:
    window: int = 21
    pyramid_levels: int = 3
    iterations: int = 30
    epsilon: float = .01
    min_eigenvalue: float = .0001
    border_px: int = 10
    fb_max_px: float = 1.5
    candidate_max_px: float = 3.
    fb_sigma_px: float = 1.
    candidate_sigma_px: float = 2.
    minimum_weight: float = .2

    def __post_init__(self):
        for name in ("window", "pyramid_levels", "iterations", "border_px"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in {"pyramid_levels", "border_px"} else 1):
                raise ValueError(f"invalid {name}")
        for field in fields(self):
            if type(getattr(self, field.name)) not in (int, float) or not math.isfinite(getattr(self, field.name)):
                raise ValueError(f"invalid {field.name}")
        if (self.window < 3 or self.window % 2 != 1 or self.border_px < self.window//2 or
                min(self.epsilon, self.min_eigenvalue, self.fb_max_px, self.candidate_max_px,
                    self.fb_sigma_px, self.candidate_sigma_px) <= 0 or not 0 < self.minimum_weight <= 1):
            raise ValueError("invalid optical-flow thresholds/window")


@dataclass
class Tracks:
    origins: np.ndarray
    predicted: np.ndarray
    returned: np.ndarray
    valid: np.ndarray
    fb_error_px: np.ndarray
    reason: np.ndarray


@dataclass
class TemporalContext:
    forward: Tracks
    reverse: Tracks
    config: TemporalConfig
    timing_ms: dict


def _inside(points, shape, border):
    h, w = shape
    return np.isfinite(points).all(-1) & (points >= border).all(-1) & (points <= [w-1-border, h-1-border]).all(-1)


def _track(image0, image1, points, mask, config):
    n = len(points)
    origins = np.asarray(points, dtype=np.float32)
    if origins.shape != (n, 2):
        raise ValueError("points must be N,2")
    if mask is None:
        mask = np.ones(n, dtype=bool)
    mask = np.asarray(mask)
    if mask.shape != (n,) or mask.dtype != bool:
        raise ValueError("invalid temporal padding mask")
    predicted, returned = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32)
    errors = np.full(n, np.inf, dtype=np.float32)
    reason = np.full(n, "padding", dtype="U36")
    reason[mask] = "source_invalid_or_boundary"
    ids = np.flatnonzero(mask & _inside(origins, image0.shape, config.border_px))
    options = dict(winSize=(config.window, config.window), maxLevel=config.pyramid_levels,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, config.iterations, config.epsilon),
                   flags=0, minEigThreshold=config.min_eigenvalue)
    if len(ids):
        reason[ids] = "forward_failed"
        try:
            p, status, _ = cv2.calcOpticalFlowPyrLK(image0, image1, origins[ids, None].copy(), None, **options)
        except cv2.error:
            p, status = None, None
            reason[ids] = "opencv_forward_failure"
        if p is not None and status is not None:
            p, status = p.reshape(-1, 2), status.reshape(-1).astype(bool)
            predicted[ids] = np.where(np.isfinite(p), p, 0.)
            reason[ids[status]] = "forward_boundary"
            eligible = status & _inside(p, image1.shape, config.border_px)
            back_ids = ids[eligible]
            if len(back_ids):
                reason[back_ids] = "backward_failed"
                try:
                    back, back_status, _ = cv2.calcOpticalFlowPyrLK(image1, image0, p[eligible, None].copy(), None, **options)
                except cv2.error:
                    back, back_status = None, None
                    reason[back_ids] = "opencv_backward_failure"
                if back is not None and back_status is not None:
                    back, back_status = back.reshape(-1, 2), back_status.reshape(-1).astype(bool)
                    returned[back_ids] = np.where(np.isfinite(back), back, 0.)
                    reason[back_ids[back_status]] = "backward_boundary"
                    ok = back_status & _inside(back, image0.shape, config.border_px)
                    complete = back_ids[ok]
                    errors[complete] = np.linalg.norm(back[ok]-origins[complete], axis=-1)
                    reason[complete] = "fb_inconsistent"
                    reason[complete[errors[complete] <= config.fb_max_px]] = "ok"
    return Tracks(origins.copy(), predicted, returned, reason == "ok", errors, reason)


def track_pair(image0, image1, points0, points1, config=None, mask0=None, mask1=None):
    """Track all keypoints in both directions, each with its own round-trip test.

    Images are observed/degraded grayscale uint8, equal resolution. No reference
    poses or clean RGB/depth are accepted by this interface. NaN padding and
    invalid unpadded source positions never enter OpenCV. Unknown errors are Inf
    in diagnostic arrays, but never enter finite training/filtering weights.
    """
    cfg = config or TemporalConfig()
    if (image0.ndim != 2 or image0.shape != image1.shape or min(image0.shape) < 1 or
            image0.dtype != np.uint8 or image1.dtype != np.uint8):
        raise ValueError("optical flow needs equal-size grayscale uint8 images")
    start = perf_counter()
    a = _track(np.ascontiguousarray(image0), np.ascontiguousarray(image1), points0, mask0, cfg)
    middle = perf_counter()
    b = _track(np.ascontiguousarray(image1), np.ascontiguousarray(image0), points1, mask1, cfg)
    finish = perf_counter()
    return TemporalContext(a, b, cfg, {"forward_roundtrip_ms": (middle-start)*1000,
                                      "reverse_roundtrip_ms": (finish-middle)*1000,
                                      "optical_flow_ms": (finish-start)*1000})


def candidate_reliability(context, pairs):
    """Check actual candidate endpoints as well as round-trip flow consistency."""
    a, b, cfg = context.forward, context.reverse, context.config
    pairs = validate_pairs(pairs, len(a.origins), len(b.origins))
    i, j = pairs.T
    e0 = np.linalg.norm(a.predicted[i]-b.origins[j], axis=-1)
    e1 = np.linalg.norm(b.predicted[j]-a.origins[i], axis=-1)
    reason = np.full(len(pairs), "flow_unreliable", dtype="U36")
    tracked = a.valid[i] & b.valid[j]
    reason[tracked] = "candidate_inconsistent"
    eligible = tracked & np.isfinite(e0) & np.isfinite(e1) & (e0 <= cfg.candidate_max_px) & (e1 <= cfg.candidate_max_px)
    weight = np.zeros(len(pairs), dtype=np.float32)
    # Evaluate only finite certified tracks: 0 * NaN is never used as sanitization.
    ii, jj = i[eligible], j[eligible]
    exponent = -.25*((a.fb_error_px[ii]**2+b.fb_error_px[jj]**2)/cfg.fb_sigma_px**2+
                     (e0[eligible]**2+e1[eligible]**2)/cfg.candidate_sigma_px**2)
    weight[eligible] = np.exp(exponent)
    reason[eligible] = "ok"
    return {"weights": weight, "eligible": eligible, "reason": reason,
            "candidate_forward_px": np.where(np.isfinite(e0), e0, np.inf).astype(np.float32),
            "candidate_reverse_px": np.where(np.isfinite(e1), e1, np.inf).astype(np.float32),
            "fb_forward_px": a.fb_error_px[i], "fb_reverse_px": b.fb_error_px[j]}


def filter_candidates(context, pairs, descriptor_scores, enabled=True):
    pairs = validate_pairs(pairs, len(context.forward.origins), len(context.reverse.origins))
    scores = np.asarray(descriptor_scores, dtype=np.float32)
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("descriptor scores must be finite M values in [0,1]")
    evidence = candidate_reliability(context, pairs)
    keep = evidence["eligible"] & (evidence["weights"] >= context.config.minimum_weight) if enabled else np.ones(len(pairs), bool)
    if enabled:
        evidence["reason"][evidence["eligible"] & ~keep] = "below_minimum_weight"
    # Preserve candidate order and one-to-one uniqueness; no automatic recovery of rejected pairs.
    adjusted = scores*evidence["weights"] if enabled else scores.copy()
    return {**evidence, "keep": keep, "pairs": pairs[keep], "scores": adjusted[keep], "candidate_scores": adjusted}
