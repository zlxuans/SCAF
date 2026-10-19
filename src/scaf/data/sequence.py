"""Strict TUM-format readers, one-to-one RGB-D association, explicit calibration."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .catalog import confined
from .geometry import interpolate_pose, quaternion_matrix, resized_intrinsics

cv2.setNumThreads(1)


def read_rows(path, columns, allow_equal_times=False):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != columns:
            raise ValueError(f"{path}:{number}: expected {columns} columns")
        timestamp = float(fields[0])
        out_of_order = rows and (timestamp < rows[-1][0] or (timestamp == rows[-1][0] and not allow_equal_times))
        if not np.isfinite(timestamp) or out_of_order:
            raise ValueError(f"{path}:{number}: timestamps must be finite and strictly increasing")
        rows.append((timestamp, *fields[1:]))
    if not rows:
        raise ValueError(f"empty index: {path}")
    return rows


def associate(rgb, depth, max_difference=0.02):
    """Globally nearest greedy pairs within tolerance; each timestamp used once."""
    if not np.isfinite(max_difference) or max_difference < 0:
        raise ValueError("invalid association tolerance")
    times = np.asarray([r[0] for r in depth])
    edges = []
    for i, row in enumerate(rgb):
        lo = int(np.searchsorted(times, row[0]-max_difference, side="left"))
        hi = int(np.searchsorted(times, row[0]+max_difference, side="right"))
        for j in range(int(lo), hi):
            edges.append((abs(row[0]-times[j]), i, j))
    used_rgb, used_depth, pairs = set(), set(), []
    for difference, i, j in sorted(edges):
        if i not in used_rgb and j not in used_depth:
            pairs.append((i, j))
            used_rgb.add(i)
            used_depth.add(j)
    return sorted(pairs)


@dataclass
class Frame:
    rgb: np.ndarray  # uint8 RGB, grayscale inputs repeated across channels
    depth: np.ndarray  # metres, zero means invalid; reference geometry is never degraded in place
    K: np.ndarray
    T_world_camera: np.ndarray | None
    timestamp: float
    depth_timestamp: float
    identity: str


class RGBDSequence:
    def __init__(self, spec, data_root, max_difference=0.02, pose_max_gap=0.05):
        self.spec = spec
        self.root = confined(data_root, spec["path"])
        self.rgb = read_rows(self.root / "rgb.txt", 2)
        self.depth = read_rows(self.root / "depth.txt", 2)
        for rows in (self.rgb, self.depth):
            files = [confined(self.root, row[1]) for row in rows]
            if len(set(files)) != len(files):
                raise ValueError("same image reused at multiple timestamps")
            if any(not path.is_file() for path in files):
                raise FileNotFoundError(f"missing image file in {self.root}")
        self.pairs = associate(self.rgb, self.depth, max_difference)
        if not self.pairs:
            raise ValueError("no synchronized RGB-D pairs")
        if spec["dataset"] == "eth3d" and any(self.rgb[i][0] != self.depth[j][0] for i, j in self.pairs):
            raise ValueError("ETH3D RGB/depth should have identical timestamps")
        pose_file = self.root / "groundtruth.txt"
        self.poses = None
        self.ambiguous_pose_times = np.empty(0)
        self.ambiguous_pose_rows = 0
        if pose_file.exists():
            raw_poses = np.asarray(read_rows(pose_file, 8, allow_equal_times=True), dtype=np.float64)
            if not np.isfinite(raw_poses).all():
                raise ValueError("nonfinite reference pose")
            for row in raw_poses:
                quaternion_matrix(row[4:])
            times, counts = np.unique(raw_poses[:, 0], return_counts=True)
            self.ambiguous_pose_times = times[counts > 1]
            self.ambiguous_pose_rows = int(counts[counts > 1].sum())
            poses = raw_poses[~np.isin(raw_poses[:, 0], self.ambiguous_pose_times)]
            self.poses = poses if len(poses) else None
        self.pose_max_gap = pose_max_gap
        camera = spec["camera"]
        if set(camera) == {"source", "intrinsics"} and camera["source"] == "tum_registered_ros_default":
            values = camera["intrinsics"]
        elif camera == {"source": "calibration.txt"}:
            values = np.loadtxt(self.root / "calibration.txt").reshape(-1)
        else:
            raise ValueError("unsupported/ambiguous camera calibration")
        values = np.asarray(values, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all() or min(values[:2]) <= 0:
            raise ValueError("invalid pinhole calibration")
        fx, fy, cx, cy = values
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])

    def __len__(self):
        return len(self.pairs)

    def pose(self, timestamp):
        if self.poses is None:
            return None
        hi = int(np.searchsorted(self.poses[:, 0], timestamp))
        if hi < len(self.poses) and self.poses[hi, 0] == timestamp:
            return interpolate_pose(self.poses, timestamp, self.pose_max_gap)
        if 0 < hi < len(self.poses):
            lower, upper = self.poses[hi-1, 0], self.poses[hi, 0]
            if np.any((self.ambiguous_pose_times >= lower) & (self.ambiguous_pose_times <= upper)):
                return None  # never bridge a timestamp with ambiguous truth
        return interpolate_pose(self.poses, timestamp, self.pose_max_gap)

    def load(self, index, max_wh=(640, 480)):
        i, j = self.pairs[index]
        rt, rp = self.rgb[i]
        dt, dp = self.depth[j]
        bgr = cv2.imread(str(confined(self.root, rp)), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(confined(self.root, dp)), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None or raw.ndim != 2 or raw.dtype != np.uint16:
            raise ValueError("RGB decoding or uint16 depth check failed")
        if bgr.shape[:2] != raw.shape:
            raise ValueError("RGB and depth must be registered at identical dimensions")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = raw.shape
        scale = min(1., max_wh[0]/w, max_wh[1]/h)
        new_wh = (max(1, round(w*scale)), max(1, round(h*scale)))
        if new_wh != (w, h):
            rgb = cv2.resize(rgb, new_wh, interpolation=cv2.INTER_AREA)
            raw = cv2.resize(raw, new_wh, interpolation=cv2.INTER_NEAREST)
        depth = raw.astype(np.float32) / self.spec["depth_units_per_metre"]
        depth[(depth < 0.2) | (depth > 8)] = 0
        return Frame(rgb, depth, resized_intrinsics(self.K, (w, h), new_wh), self.pose(rt),
                     rt, dt, f"{self.spec['id']}/{rt:.9f}")
