"""Discriminative geometry-supervised loss with optional temporal reliability."""
import math

import torch
from torch.nn import functional as F


def temporal_descriptor_loss(desc0, desc1, pairs, negative_mask, *, weights=None, mask0=None, mask1=None,
                             positive_mask=None, temperature=.1, consistency_weight=.1):
    """One frame pair, ragged endpoints and padded positive bank supported.

    Positive pairs are certified by independent clean geometry. negative_mask is
    an explicit P,P bank (True = geometrically far, not an unknown -1 label).
    Active positives must be unique on both sides. Detached reliability weights
    scale each positive anchor AND its consistency term. Divide by eligible
    anchor counts, never the weight sum, preserving absolute attenuation.
    Consistency excludes isolated positives without any trusted negative.
    """
    if (not math.isfinite(temperature) or temperature <= 0 or
            not math.isfinite(consistency_weight) or consistency_weight < 0):
        raise ValueError("invalid loss temperature/coefficient")
    if (desc0.ndim != 2 or desc1.ndim != 2 or desc0.shape[1] != desc1.shape[1] or
            desc0.device != desc1.device or desc0.dtype != torch.float32 or desc1.dtype != torch.float32):
        raise ValueError("expected colocated float32 N,D descriptors")
    device = desc0.device
    clean = []
    endpoint_masks = []
    for desc, mask in ((desc0, mask0), (desc1, mask1)):
        mask = torch.ones(len(desc), dtype=torch.bool, device=device) if mask is None else mask.to(device)
        if mask.shape != (len(desc),) or mask.dtype != torch.bool:
            raise ValueError("invalid endpoint padding mask")
        if not torch.isfinite(desc[mask]).all() or (desc[mask].norm(dim=-1) < 1e-8).any():
            raise ValueError("active descriptors must be finite and nonzero")
        clean.append(F.normalize(torch.where(mask[:, None], desc, 0.), dim=-1))
        endpoint_masks.append(mask)
    a, b = clean
    zero = (a.sum()+b.sum())*0
    pairs, negative_mask = pairs.to(device), negative_mask.to(device)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError("positive bank must be int64 P,2")
    p = len(pairs)
    if negative_mask.shape != (p, p) or negative_mask.dtype != torch.bool:
        raise ValueError("invalid trusted-negative mask")
    positive_mask = torch.ones(p, dtype=torch.bool, device=device) if positive_mask is None else positive_mask.to(device)
    if positive_mask.shape != (p,) or positive_mask.dtype != torch.bool:
        raise ValueError("invalid positive padding mask")
    selected = torch.nonzero(positive_mask, as_tuple=False).flatten()
    pp = pairs[selected]
    if ((pp < 0).any() or (pp[:, 0] >= len(a)).any() or (pp[:, 1] >= len(b)).any()):
        raise ValueError("positive bank index out of bounds")
    selected = selected[endpoint_masks[0][pp[:, 0]] & endpoint_masks[1][pp[:, 1]]]
    pp = pairs[selected]
    if any(len(torch.unique(pp[:, k])) != len(pp) for k in (0, 1)):
        raise ValueError("active positive pairs must be one-to-one")
    negatives = negative_mask[selected[:, None], selected[None, :]]
    if negatives.diag().any():
        raise ValueError("positive pair cannot also be a negative")
    weights = torch.ones(p, device=device) if weights is None else weights.to(device)
    if weights.shape != (p,):
        raise ValueError("invalid reliability shape")
    w = weights[selected].detach()
    if not torch.isfinite(w).all() or ((w < 0) | (w > 1)).any():
        raise ValueError("active reliability must be finite in [0,1]")
    valid0, valid1 = negatives.any(1), negatives.any(0)
    anchors = int(valid0.sum()+valid1.sum())
    valid_consistency = valid0 | valid1
    stats = {"positives": len(pp), "negative_edges": int(negatives.sum()), "anchors": anchors,
             "positive_weight_sum": float(w.sum()), "weighted_anchors": float((w*valid0+w*valid1).sum())}
    if not anchors:
        return zero, {**stats, "discriminative": 0., "consistency": 0., "total": 0.}
    x, y = a[pp[:, 0]], b[pp[:, 1]]
    logits = x @ y.T/temperature
    allowed = negatives | torch.eye(len(pp), dtype=torch.bool, device=device)
    logits = logits.masked_fill(~allowed, -torch.inf)
    labels = torch.arange(len(pp), device=device)
    discriminative = zero
    if valid0.any():
        discriminative = discriminative+(F.cross_entropy(logits[valid0], labels[valid0], reduction="none")*w[valid0]).sum()/anchors
    if valid1.any():
        discriminative = discriminative+(F.cross_entropy(logits.T[valid1], labels[valid1], reduction="none")*w[valid1]).sum()/anchors
    consistency = ((1-(x*y).sum(-1)).clamp_min(0)*w)[valid_consistency].sum()/valid_consistency.sum()
    total = discriminative+consistency_weight*consistency
    return total, {**stats, "discriminative": float(discriminative.detach()), "consistency": float(consistency.detach()),
                   "total": float(total.detach())}
