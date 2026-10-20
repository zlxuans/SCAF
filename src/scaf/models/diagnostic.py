"""Small discriminative debugging objective, not the stage-5/6 training pipeline."""
import numpy as np
import torch
from torch.nn import functional as F

from scaf.data.geometry import correspondences


def diagnostic_targets(frame0, frame1, points0, points1, negative_radius_px=12.):
    """Mutual clean-geometry positives; negatives must be far in BOTH projections.

    The bank contains only endpoints of certified positive pairs. A -1 label is
    never turned into a negative. Close or ambiguous cross-pairs are ignored.
    This shares the static-scene assumptions of the stage-2 geometry labels.
    """
    if not np.isfinite(negative_radius_px) or negative_radius_px <= 3:
        raise ValueError("negative radius must exceed the positive radius (3px)")
    a = correspondences(frame0, frame1, points0, points1)
    b = correspondences(frame1, frame0, points1, points0)
    i = np.flatnonzero(a["matches0"] >= 0)
    j = a["matches0"][i]
    mutual = b["matches0"][j] == i
    i, j = i[mutual], j[mutual]
    forward = np.linalg.norm(a["projected"][i, None]-points1[None, j], axis=-1)
    reverse = np.linalg.norm(points0[i, None]-b["projected"][None, j], axis=-1)
    negatives = (forward >= negative_radius_px) & (reverse >= negative_radius_px)
    np.fill_diagonal(negatives, False)
    return torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(negatives)


def diagnostic_loss(desc0, desc1, targets, temperature=.1):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite temperature required")
    i, j, negatives = (x.to(desc0.device) for x in targets)
    logits = desc0[i] @ desc1[j].T/temperature
    if not len(i):
        return (desc0.sum()+desc1.sum())*0, {"positives": 0, "negative_edges": 0, "anchors": 0}
    allowed = negatives | torch.eye(len(i), dtype=torch.bool, device=desc0.device)
    logits = logits.masked_fill(~allowed, -torch.inf)
    labels = torch.arange(len(i), device=desc0.device)
    valid0, valid1 = negatives.any(1), negatives.any(0)
    pieces = []
    if valid0.any():
        pieces.append(F.cross_entropy(logits[valid0], labels[valid0], reduction="sum"))
    if valid1.any():
        pieces.append(F.cross_entropy(logits.T[valid1], labels[valid1], reduction="sum"))
    count = int(valid0.sum()+valid1.sum())
    loss = sum(pieces)/count if count else (desc0.sum()+desc1.sum())*0
    return loss, {"positives": len(i), "negative_edges": int(negatives.sum()), "anchors": count}
