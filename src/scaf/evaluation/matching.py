"""Evaluate proposed matches against clean RGB-D/reference geometry, not RANSAC."""
from collections import Counter

import numpy as np

from scaf.baselines.matching import validate_pairs
from scaf.data.geometry import project_points, sample_depth


def match_metrics(frame0, frame1, points0, points1, pairs, radius=3., depth_abs=.05, depth_rel=.02):
    pairs = validate_pairs(pairs, len(points0), len(points1))
    reasons = np.full(len(pairs), "reference_pose_missing", dtype="U40")
    errors = np.full(len(pairs), np.nan)
    evaluable = np.zeros(len(pairs), dtype=bool)
    correct = np.zeros(len(pairs), dtype=bool)
    if radius <= 0 or min(depth_abs, depth_rel) < 0:
        raise ValueError("invalid geometric thresholds")
    if frame0.T_world_camera is not None and frame1.T_world_camera is not None:
        a, b = np.asarray(points0)[pairs[:, 0]], np.asarray(points1)[pairs[:, 1]]
        T = np.linalg.inv(frame1.T_world_camera) @ frame0.T_world_camera
        projected, z, source_ok = project_points(a, frame0.depth, frame0.K, frame1.K, T)
        target_depth, visible = sample_depth(frame1.depth, projected)
        candidate_depth, candidate_ok = sample_depth(frame1.depth, b)
        tolerance = depth_abs+depth_rel*np.abs(z)
        reasons[:] = "source_depth_or_projection_invalid"
        reasons[source_ok] = "projected_target_bounds_or_depth"
        stage = source_ok & visible
        reasons[stage] = "projected_target_depth_inconsistent"
        stage &= np.abs(target_depth-z) <= tolerance
        reasons[stage] = "candidate_depth_invalid"
        evaluable = stage & candidate_ok
        # A visible source with a valid but geometrically wrong candidate is an error,
        # not an ignored match. Otherwise outliers could inflate precision.
        errors[evaluable] = np.linalg.norm(projected[evaluable]-b[evaluable], axis=1)
        correct = evaluable & (errors <= radius) & (np.abs(candidate_depth-z) <= tolerance)
        reasons[evaluable] = "incorrect"
        reasons[correct] = "correct"
    n, c = int(evaluable.sum()), int(correct.sum())
    result = {"predicted": len(pairs), "evaluable": n, "correct": c,
              "precision": c/n if n else None, "unevaluable": len(pairs)-n,
              "reasons": dict(Counter(reasons.tolist()))}
    return result, {"evaluable": evaluable, "correct": correct,
                    "reprojection_error_px": errors, "reason": reasons}
