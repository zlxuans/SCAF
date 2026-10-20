"""Frame-local RGB-D descriptor refinement with an explicit appearance fallback."""
from dataclasses import dataclass, fields
from time import perf_counter
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GeometryConfig:
    descriptor_dim: int = 256
    hidden_dim: int = 64
    neighbors: int = 16
    radius_m: float = .30
    radius_px: float = 96.
    patch_radius: int = 2
    min_depth_m: float = .2
    max_depth_m: float = 8.
    continuity_abs_m: float = .03
    continuity_rel: float = .02
    residual_scale: float = .15

    def __post_init__(self):
        for name in ("descriptor_dim", "hidden_dim", "neighbors", "patch_radius"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "patch_radius" else 1):
                raise ValueError(f"invalid integer {name}")
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid {field.name}")
        if (min(self.radius_m, self.radius_px, self.min_depth_m, self.continuity_abs_m) <= 0 or
                self.max_depth_m <= self.min_depth_m or self.continuity_rel < 0 or
                not 0 < self.residual_scale < .5):
            raise ValueError("invalid geometry thresholds/residual scale")


@dataclass
class Neighborhood:
    mask: torch.Tensor                 # B,N valid (unpadded) keypoints
    xyz: torch.Tensor                  # B,N,3 meters, invalid depth -> zero
    reliability: torch.Tensor          # B,N in [0,1]
    appearance_indices: torch.Tensor   # B,N,k, masked slots safely index zero
    appearance_mask: torch.Tensor
    appearance_relative: torch.Tensor  # B,N,k,3
    geometry_indices: torch.Tensor
    geometry_mask: torch.Tensor
    geometry_relative: torch.Tensor    # B,N,k,5
    timings_ms: dict
    storage: dict


def gather(values, indices):
    """Batch gather B,N,C at B,N,k; also handles k=0/N=0."""
    batch = torch.arange(len(values), device=values.device)[:, None, None]
    return values[batch, indices]


def _nearest(distance, eligible, k):
    values = distance.masked_fill(~eligible, float("inf"))
    indices = torch.argsort(values, dim=-1, stable=True)[..., :k]
    valid = torch.gather(eligible, -1, indices)
    return indices.masked_fill(~valid, 0), valid


@torch.no_grad()
def build_neighborhood(points, depth, K, mask=None, config=None):
    """No GT/poses or cross-frame information. Geometry is deliberately detached.

    points B,N,2 (pixel centers), registered depth B,H,W in meters, K B,3,3.
    Invalid depths include NaN/Inf/out-of-range; padding may contain arbitrary data.
    Stable index order resolves equal-distance ties. Exact search costs O(B*N^2).
    CPU wall times are provided only for CPU tensors (no hidden CUDA synchronization).
    """
    cfg = config or GeometryConfig()
    start = perf_counter()
    if (points.ndim != 3 or points.shape[-1] != 2 or depth.ndim != 3 or
            K.shape != (points.shape[0], 3, 3) or depth.shape[0] != points.shape[0] or
            points.device != depth.device or points.device != K.device or
            points.dtype != torch.float32 or depth.dtype != points.dtype or K.dtype != points.dtype):
        raise ValueError("expected colocated float32 points B,N,2, depth B,H,W and K B,3,3")
    b, n, _ = points.shape
    h, w = depth.shape[-2:]
    if min(b, h, w) < 1 or not torch.isfinite(K).all() or (torch.linalg.det(K).abs() < 1e-8).any():
        raise ValueError("empty image/batch or invalid intrinsics")
    if mask is None:
        mask = torch.ones((b, n), dtype=torch.bool, device=points.device)
    if mask.shape != (b, n) or mask.dtype != torch.bool or mask.device != points.device:
        raise ValueError("invalid point padding mask")
    if not torch.isfinite(points[mask]).all():
        raise ValueError("nonfinite unpadded point")
    xy = torch.where(mask[..., None], points, 0.)
    if (((xy[..., 0] < 0) | (xy[..., 0] > w-1) | (xy[..., 1] < 0) | (xy[..., 1] > h-1)) & mask).any():
        raise ValueError("unpadded point outside image")
    pixels = torch.floor(xy+.5).long()
    batch = torch.arange(b, device=points.device)[:, None]
    z = depth[batch, pixels[..., 1], pixels[..., 0]]
    good = mask & torch.isfinite(z) & (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m)
    safe_z = torch.where(good, z, 0.)
    rays = torch.cat([xy, torch.ones_like(xy[..., :1])], -1) @ torch.linalg.inv(K).transpose(-1, -2)
    xyz = torch.where(good[..., None], rays*safe_z[..., None], 0.)
    # Invalid samples count against density; out-of-image samples do not.
    continuous, in_image = torch.zeros_like(z), torch.zeros_like(z)
    for dy in range(-cfg.patch_radius, cfg.patch_radius+1):
        for dx in range(-cfg.patch_radius, cfg.patch_radius+1):
            u, v = pixels[..., 0]+dx, pixels[..., 1]+dy
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            sample = depth[batch, v.clamp(0, h-1), u.clamp(0, w-1)]
            valid_sample = torch.isfinite(sample) & (sample >= cfg.min_depth_m) & (sample <= cfg.max_depth_m)
            consistent = (sample-safe_z).abs() <= cfg.continuity_abs_m+cfg.continuity_rel*safe_z
            continuous += (inside & valid_sample & consistent).float()
            in_image += inside.float()
    q = torch.where(good, continuous/in_image.clamp_min(1), 0.)
    prepared = perf_counter()
    k = min(cfg.neighbors, max(n-1, 0))
    distance2 = torch.cdist(xy, xy, compute_mode="donot_use_mm_for_euclid_dist")
    eligible = mask[:, :, None] & mask[:, None, :]
    eligible &= ~torch.eye(n, dtype=torch.bool, device=points.device)[None]
    image_eligible = eligible & (distance2 <= cfg.radius_px)
    ai, am = _nearest(distance2, image_eligible, k)
    delta2 = (gather(xy, ai)-xy[:, :, None])/cfg.radius_px
    ar = torch.cat([delta2, delta2.norm(dim=-1, keepdim=True)], -1).masked_fill(~am[..., None], 0.)
    distance3 = torch.cdist(xyz, xyz, compute_mode="donot_use_mm_for_euclid_dist")
    geo_eligible = image_eligible & good[:, :, None] & good[:, None, :] & (distance3 <= cfg.radius_m)
    gi, gm = _nearest(distance3, geo_eligible, k)
    delta3 = (gather(xyz, gi)-xyz[:, :, None])/cfg.radius_m
    ratios = torch.log(gather(safe_z[..., None], gi).squeeze(-1).clamp_min(1e-8)/
                       safe_z[:, :, None].clamp_min(1e-8)).clamp(-4, 4)
    gr = torch.cat([delta3, delta3.norm(dim=-1, keepdim=True), ratios[..., None]], -1)
    gr = gr.masked_fill(~gm[..., None], 0.)
    searched = perf_counter()
    arrays = (mask, xyz, q, ai, am, ar, gi, gm, gr)
    storage = {"context_tensor_bytes": sum(t.numel()*t.element_size() for t in arrays),
               "distance_matrix_bytes_each": b*n*n*points.element_size(),
               "distance_matrices_materialized": 2,
               "search_complexity": "O(B*N^2) distances; stable sorting O(B*N^2*log N)",
               "message_shape": [b, n, k], "excludes": "allocator, sorting workspace, autograd, MLP activations and model"}
    times = ({"depth_prepare_ms": (prepared-start)*1000, "neighbor_search_ms": (searched-prepared)*1000}
             if points.device.type == "cpu" else {})
    return Neighborhood(mask.clone(), xyz, q, ai, am, ar, gi, gm, gr, times, storage)


def _mean(values, weights):
    return (values*weights[..., None]).sum(-2)/weights.sum(-1, keepdim=True).clamp_min(1e-8)


def _bounded(values):
    return values/values.norm(dim=-1, keepdim=True).clamp_min(1.)


class GeometryEnhancer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or GeometryConfig()
        c, d = self.config.hidden_dim, self.config.descriptor_dim
        self.encoder = nn.Sequential(nn.Linear(d, c), nn.GELU())
        self.appearance_edge = nn.Sequential(nn.Linear(c+3, c), nn.GELU())
        self.geometry_edge = nn.Sequential(nn.Linear(c+5, c), nn.GELU())
        self.appearance_head = nn.Sequential(nn.Linear(2*c, c), nn.GELU(), nn.Linear(c, d))
        self.geometry_head = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, d))
        for head in (self.appearance_head, self.geometry_head):
            nn.init.normal_(head[-1].weight, std=.001)
            nn.init.zeros_(head[-1].bias)

    def forward(self, descriptors, context, *, geometry=True, reliability=True, return_info=False):
        cfg, m = self.config, context.mask
        if (descriptors.shape != (*m.shape, cfg.descriptor_dim) or descriptors.device != m.device or
                descriptors.dtype != torch.float32):
            raise ValueError("invalid descriptor tensor")
        if not torch.isfinite(descriptors[m]).all() or (descriptors[m].norm(dim=-1) < 1e-8).any():
            raise ValueError("valid descriptors must be finite and nonzero")
        # Sanitize BEFORE nonlinear operations; NaN * zero is still NaN.
        d = F.normalize(torch.where(m[..., None], descriptors, 0.), dim=-1)
        h = self.encoder(d).masked_fill(~m[..., None], 0.)
        start = perf_counter()
        ae = torch.cat([gather(h, context.appearance_indices)-h[:, :, None], context.appearance_relative], -1)
        app = self.appearance_head(torch.cat([h, _mean(self.appearance_edge(ae), context.appearance_mask.float())], -1))
        geo = torch.zeros_like(app)
        gate = torch.zeros_like(context.reliability)
        if geometry:
            q = context.reliability if reliability else (context.xyz[..., 2] > 0).float()
            qw = gather(q[..., None], context.geometry_indices).squeeze(-1)*context.geometry_mask
            ge = torch.cat([gather(h, context.geometry_indices)-h[:, :, None], context.geometry_relative], -1)
            geo = self.geometry_head(_mean(self.geometry_edge(ge), qw))
            # Preserve absolute confidence: uniformly tiny weights must not cancel in a normalized mean.
            gate = q*qw.sum(-1)/context.geometry_mask.sum(-1).clamp_min(1)
        residual = cfg.residual_scale*(_bounded(app)+gate[..., None]*_bounded(geo))
        result = F.normalize(d+residual, dim=-1).masked_fill(~m[..., None], 0.)
        info = {"geometry_gate": gate, "aggregation_network_ms":
                (perf_counter()-start)*1000 if descriptors.device.type == "cpu" else None}
        return (result, info) if return_info else result
