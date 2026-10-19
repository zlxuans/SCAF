"""Source RGB-D to target 2D PnP. No reference pose is used by this backend."""
from dataclasses import dataclass

import cv2
import numpy as np

from scaf.baselines.matching import validate_pairs
from scaf.data.geometry import sample_depth


@dataclass(frozen=True)
class PnPConfig:
    reprojection_px: float = 3.
    iterations: int = 100
    confidence: float = .999
    min_correspondences: int = 6
    min_inliers: int = 6
    seed: int = 7

    def __post_init__(self):
        if (not np.isfinite(self.reprojection_px) or self.reprojection_px <= 0 or
                not 0 < self.confidence < 1 or self.iterations < 1 or
                self.min_correspondences < 6 or self.min_inliers < 6 or
                not 0 <= self.seed < 2**31):
            raise ValueError("invalid PnP configuration")


def estimate_motion(points0, points1, pairs, depth0, K0, K1, config=PnPConfig()):
    pairs = validate_pairs(pairs, len(points0), len(points1))
    source, target = np.asarray(points0)[pairs[:, 0]], np.asarray(points1)[pairs[:, 1]]
    z, valid = sample_depth(depth0, source)
    valid &= np.isfinite(target).all(axis=1)
    ids = np.flatnonzero(valid)
    result = {"success": False, "reason": "insufficient_depth_correspondences",
              "valid_correspondences": len(ids), "inliers": 0, "ransac_inlier_ratio": None,
              "T_camera1_camera0": None, "inlier_match_indices": np.empty(0, dtype=np.int64)}
    if len(ids) < config.min_correspondences:
        return result
    xyz = np.column_stack([source[valid], np.ones(len(ids))]) @ np.linalg.inv(K0).T
    xyz *= z[valid, None]
    image = np.ascontiguousarray(target[valid], dtype=np.float64)
    cv2.setRNGSeed(config.seed)  # same deterministic RANSAC stream for every method/pair
    try:
        ok, rotation, translation, inliers = cv2.solvePnPRansac(
            np.ascontiguousarray(xyz, dtype=np.float64), image, np.asarray(K1, dtype=np.float64), None,
            iterationsCount=config.iterations, reprojectionError=config.reprojection_px,
            confidence=config.confidence, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < config.min_inliers:
            result["reason"] = "ransac_failed_or_too_few_inliers"
            return result
        selected = inliers.ravel()
        rotation, translation = cv2.solvePnPRefineLM(xyz[selected], image[selected], K1, None,
                                                    rotation, translation)
        R = cv2.Rodrigues(rotation)[0]
        t = translation.ravel()
        transformed = xyz @ R.T+t
        projected = transformed @ K1.T
        positive = transformed[:, 2] > 1e-8
        errors = np.full(len(xyz), np.inf)
        errors[positive] = np.linalg.norm(projected[positive, :2]/projected[positive, 2, None]-image[positive], axis=1)
        # Recheck refinement using only the RANSAC consensus, never recruit new inliers.
        selected = selected[positive[selected] & (errors[selected] <= config.reprojection_px)]
        if not np.isfinite(R).all() or not np.isfinite(t).all() or len(selected) < config.min_inliers:
            result["reason"] = "refinement_failed_cheirality_or_inliers"
            return result
    except cv2.error:
        result["reason"] = "opencv_pnp_error"
        return result
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    result.update(success=True, reason="ok", inliers=len(selected),
                  ransac_inlier_ratio=len(selected)/len(ids), T_camera1_camera0=T,
                  inlier_match_indices=ids[selected])
    return result
