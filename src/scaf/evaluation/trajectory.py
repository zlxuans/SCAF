"""SE(3), never Sim(3), trajectory scores with explicit disconnected segments."""
import numpy as np


def error_statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "rmse": None, "mean": None, "median": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite metric values")
    return {"count": len(values), "rmse": float(np.sqrt(np.mean(values**2))),
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def align_se3(estimated, reference):
    a, b = np.asarray(estimated, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError("SE3 alignment requires at least three associated positions")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite trajectory")
    ac, bc = a-a.mean(0), b-b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(vt.T @ u.T)
    R = vt.T @ fix @ u.T
    t = b.mean(0)-R @ a.mean(0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T, min(int(np.linalg.matrix_rank(ac)), int(np.linalg.matrix_rank(bc)))


def relative_error(estimate0, estimate1, reference0, reference1):
    predicted = np.linalg.inv(estimate0) @ estimate1
    expected = np.linalg.inv(reference0) @ reference1
    residual = np.linalg.inv(expected) @ predicted
    translation = float(np.linalg.norm(residual[:3, 3]))
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(residual[:3, :3])-1)/2, -1, 1))))
    return translation, rotation


class SegmentedOdometry:
    """Restart from identity at the preceding frame after failure, with no GT reset.

    Only frames participating in at least one successful transition get a pose.
    Isolated identity anchors and failed transitions do not count as tracking.
    """
    def __init__(self):
        self.segments = []
        self.current = None
        self.transitions = []

    def append(self, previous_index, current_index, motion):
        if current_index != previous_index+1 or (self.transitions and self.transitions[-1][1] != previous_index):
            raise ValueError("trajectory needs adjacent selected frame indices")
        if motion is None:
            self.current = None
            self.transitions.append((previous_index, current_index, False))
            return None
        if self.current is None:
            self.current = [(previous_index, np.eye(4))]
            self.segments.append(self.current)
        T_world_current = self.current[-1][1] @ np.linalg.inv(motion)
        self.current.append((current_index, T_world_current))
        self.transitions.append((previous_index, current_index, True))
        return len(self.segments)-1

    def summary(self, timestamps, references):
        timestamps = np.asarray(timestamps, dtype=float)
        if (len(timestamps) != len(references) or not np.isfinite(timestamps).all() or
                np.any(np.diff(timestamps) <= 0)):
            raise ValueError("trajectory timestamps must be finite, unique and increasing")
        if self.transitions and len(self.transitions) != len(timestamps)-1:
            raise ValueError("missing selected frame transitions")
        coverage = set()
        segments, rpe_t, rpe_r, intervals = [], [], [], []
        for number, segment in enumerate(self.segments):
            indices = [i for i, _ in segment]
            coverage.update(indices)
            associated = [(i, T) for i, T in segment if references[i] is not None]
            ate, alignment, rank = None, None, None
            if len(associated) >= 3:
                positions = np.array([T[:3, 3] for _, T in associated])
                truth = np.array([references[i][:3, 3] for i, _ in associated])
                alignment, rank = align_se3(positions, truth)
                errors = np.linalg.norm(positions @ alignment[:3, :3].T+alignment[:3, 3]-truth, axis=1)
                ate = error_statistics(errors)
            for (i, a), (j, b) in zip(segment, segment[1:]):
                if references[i] is not None and references[j] is not None:
                    t, r = relative_error(a, b, references[i], references[j])
                    rpe_t.append(t)
                    rpe_r.append(r)
                    intervals.append(timestamps[j]-timestamps[i])
            segments.append({"id": number, "first_selected_frame": indices[0],
                             "last_selected_frame": indices[-1], "frames": len(indices),
                             "duration_s": float(timestamps[indices[-1]]-timestamps[indices[0]]),
                             "associated_reference_frames": len(associated),
                             "within_segment_ate_m": ate, "alignment_position_rank": rank,
                             "alignment_se3": alignment.tolist() if alignment is not None else None})
        successes = sum(t[2] for t in self.transitions)
        complete = bool(self.transitions) and successes == len(self.transitions)
        return {"selected_frames": len(timestamps), "attempted_transitions": len(self.transitions),
                "successful_transitions": successes, "failed_transitions": len(self.transitions)-successes,
                "tracking_success_rate": successes/len(self.transitions) if self.transitions else None,
                "covered_frames": len(coverage), "frame_coverage": len(coverage)/len(timestamps) if len(timestamps) else 0.,
                "longest_segment_frames": max((s["frames"] for s in segments), default=0),
                "longest_segment_duration_s": max((s["duration_s"] for s in segments), default=0.),
                "reference_available_frames": sum(T is not None for T in references),
                "full_trajectory_ate_m": segments[0]["within_segment_ate_m"] if complete and segments else None,
                "segments": segments, "rpe_translation_m": error_statistics(rpe_t),
                "rpe_rotation_deg": error_statistics(rpe_r),
                "rpe_actual_interval_s": error_statistics(intervals)}


def matrix_quaternion(R):
    """Unit xyzw quaternion, using the largest eigenvector (stable at pi)."""
    R = np.asarray(R, dtype=float)
    K = np.array([[R[0, 0]-R[1, 1]-R[2, 2], R[1, 0]+R[0, 1], R[2, 0]+R[0, 2], R[2, 1]-R[1, 2]],
                  [R[1, 0]+R[0, 1], R[1, 1]-R[0, 0]-R[2, 2], R[2, 1]+R[1, 2], R[0, 2]-R[2, 0]],
                  [R[2, 0]+R[0, 2], R[2, 1]+R[1, 2], R[2, 2]-R[0, 0]-R[1, 1], R[1, 0]-R[0, 1]],
                  [R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1], np.trace(R)]])/3.
    _, vectors = np.linalg.eigh(K)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def write_tum(path, entries, timestamps):
    with path.open("x") as handle:
        for i, T in entries:
            values = [timestamps[i], *T[:3, 3], *matrix_quaternion(T[:3, :3])]
            handle.write(" ".join(f"{v:.9f}" for v in values)+"\n")
