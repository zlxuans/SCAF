"""Pixel-center pinhole geometry; optical coordinates x-right/y-down/z-forward."""
import numpy as np


def quaternion_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("invalid xyzw quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def interpolate_pose(rows, timestamp, max_gap=0.05):
    """Linear world translation + shortest-path SLERP. No extrapolation/gap filling."""
    times = rows[:, 0]
    hi = int(np.searchsorted(times, timestamp))
    if hi < len(rows) and times[hi] == timestamp:
        row = rows[hi]
        translation, q = row[1:4], row[4:8]
    else:
        if hi == 0 or hi == len(rows) or times[hi] - times[hi-1] > max_gap:
            return None
        lo = hi - 1
        alpha = (timestamp-times[lo]) / (times[hi]-times[lo])
        translation = (1-alpha)*rows[lo, 1:4] + alpha*rows[hi, 1:4]
        qa, qb = rows[lo, 4:8].copy(), rows[hi, 4:8].copy()
        qa /= np.linalg.norm(qa)
        qb /= np.linalg.norm(qb)
        dot = np.dot(qa, qb)
        if dot < 0:
            qb, dot = -qb, -dot
        angle = np.arccos(np.clip(dot, -1, 1))
        if dot > 0.9995:
            q = (1-alpha)*qa + alpha*qb
        else:
            q = (np.sin((1-alpha)*angle)*qa + np.sin(alpha*angle)*qb) / np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix(q)
    result[:3, 3] = translation
    return result


def resized_intrinsics(K, old_wh, new_wh):
    sx, sy = np.asarray(new_wh) / np.asarray(old_wh)
    result = np.asarray(K, dtype=np.float64).copy()
    result[0, 0] *= sx
    result[1, 1] *= sy
    result[0, 2] = (result[0, 2]+0.5)*sx-0.5
    result[1, 2] = (result[1, 2]+0.5)*sy-0.5
    return result


def sample_depth(depth, points):
    """Nearest pixel-center sample. Return zeros/false for invalid or out of bounds."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    h, w = depth.shape
    inside = np.isfinite(points).all(1) & (points[:, 0] >= 0) & (points[:, 0] <= w-1)
    inside &= (points[:, 1] >= 0) & (points[:, 1] <= h-1)
    values = np.zeros(len(points), dtype=np.float64)
    pix = np.floor(points[inside]+0.5).astype(int)
    values[inside] = depth[pix[:, 1], pix[:, 0]]
    valid = inside & np.isfinite(values) & (values > 0)
    return values, valid


def project_points(points, depth, K0, K1, T_1_0):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    z, valid = sample_depth(depth, points)
    rays = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(K0).T
    xyz = (rays*z[:, None]) @ T_1_0[:3, :3].T + T_1_0[:3, 3]
    valid &= np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
    projected = np.full((len(points), 2), np.nan)
    image = xyz[valid] @ K1.T
    projected[valid] = image[:, :2] / image[:, 2, None]
    return projected, xyz[:, 2], valid


def correspondences(frame0, frame1, points0, points1, radius=3.0, depth_abs=0.05, depth_rel=0.02):
    """Conservative one-to-one labels. -1 means ignored/unknown, never a proven negative.

    Static-scene geometry can reject some movers but cannot certify static points.
    Candidate visibility is tested at both projected and candidate coordinates.
    """
    if frame0.T_world_camera is None or frame1.T_world_camera is None:
        raise ValueError("geometric supervision requires reliable reference poses")
    if radius <= 0 or depth_abs < 0 or depth_rel < 0:
        raise ValueError("invalid correspondence thresholds")
    p0 = np.asarray(points0, dtype=np.float64).reshape(-1, 2)
    p1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    T = np.linalg.inv(frame1.T_world_camera) @ frame0.T_world_camera
    uv, z, valid = project_points(p0, frame0.depth, frame0.K, frame1.K, T)
    target_z, target_ok = sample_depth(frame1.depth, uv)
    labels = np.full(len(p0), -1, dtype=np.int64)
    reasons = np.full(len(p0), "invalid_source_or_behind", dtype="U32")
    reasons[valid] = "target_bounds_or_depth"
    visible = valid & target_ok
    tolerance = depth_abs + depth_rel*np.abs(z)
    reasons[visible] = "target_depth_inconsistent"
    visible &= np.abs(target_z-z) <= tolerance
    reasons[visible] = "no_candidate"
    candidates_z, candidates_ok = sample_depth(frame1.depth, p1)
    edges = []
    for i in np.flatnonzero(visible):
        distances = np.linalg.norm(p1-uv[i], axis=1)
        eligible = candidates_ok & (distances <= radius) & (np.abs(candidates_z-z[i]) <= tolerance[i])
        for j in np.flatnonzero(eligible):
            edges.append((float(distances[j]), int(i), int(j)))
    used = set()
    for error, i, j in sorted(edges):
        if labels[i] < 0 and j not in used:
            labels[i] = j
            reasons[i] = "positive"
            used.add(j)
        elif labels[i] < 0:
            reasons[i] = "duplicate_target"
    return {"matches0": labels, "reason": reasons, "projected": uv, "projected_depth": z,
            "T_camera1_camera0": T}
