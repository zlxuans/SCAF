"""CPU-only data preparation, integrity inventory and development-only QA export."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import cv2
import numpy as np
import yaml

from scaf.runtime import environment, git_info, sha256, write_json
from .catalog import confined, fingerprint, load_catalog, sequence_by_id
from .degradation import Degradation, degrade
from .geometry import correspondences
from .prepare import extract_sequence
from .sequence import RGBDSequence


def run_record(output, catalog, args):
    root = Path.cwd().resolve()
    if Path(__file__).resolve().parents[3] != root:
        raise ValueError("run from the source checkout root with PYTHONPATH=$PWD/src")
    output.mkdir(parents=True, exist_ok=False)
    record = {"kind": "data_preparation_not_algorithm_evaluation", "status": "running",
              "source": git_info(root), "catalog": catalog, "catalog_sha256": fingerprint(catalog),
              "invocation": sys.argv, "created_utc": datetime.now(timezone.utc).isoformat(),
              "environment": environment(), "data_root": str(args.data_root.resolve())}
    write_json(output / "manifest.json", record)
    return record


def audit(catalog, args):
    summaries = []
    with (args.output / "files.jsonl").open("x") as index:
        for spec in catalog["sequences"]:
            seq = RGBDSequence(spec, args.data_root)
            archive_checks = []
            for archive in spec["archives"]:
                path = confined(args.data_root, "downloads/"+archive["file"])
                digest = sha256(path)
                if digest != archive["sha256"]:
                    raise ValueError(f"archive changed: {path.name}")
                archive_checks.append({"file": path.name, "sha256": digest, "bytes": path.stat().st_size})
            files = sorted(path for path in seq.root.rglob("*") if path.is_file())
            total_bytes, decoded = 0, 0
            dimensions = set()
            for path in files:
                size = path.stat().st_size
                total_bytes += size
                entry = {"sequence": spec["id"], "path": str(path.relative_to(args.data_root)),
                         "bytes": size, "sha256": sha256(path)}
                if path.suffix.lower() == ".png":
                    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise ValueError(f"cannot decode {path}")
                    if path.parent.name == "depth" and (image.ndim != 2 or image.dtype != np.uint16):
                        raise ValueError(f"depth must be uint16: {path}")
                    dimensions.add(tuple(image.shape[:2]))
                    decoded += 1
                index.write(json.dumps(entry, allow_nan=False)+"\n")
            if len(dimensions) != 1:
                raise ValueError("image/depth dimensions vary or differ in sequence")
            pose_count = sum(seq.pose(seq.rgb[i][0]) is not None for i, _ in seq.pairs)
            deltas = [abs(seq.rgb[i][0]-seq.depth[j][0]) for i, j in seq.pairs]
            # Held-out test: schema/hash/decode/calibration only, no matching or trajectory scores.
            result = {"id": spec["id"], "scene": spec["scene"], "split": spec["split"],
                      "rgb_frames": len(seq.rgb), "depth_frames": len(seq.depth), "rgbd_pairs": len(seq),
                      "rgb_unmatched": len(seq.rgb)-len(seq), "depth_unmatched": len(seq.depth)-len(seq),
                      "pose_supported_pairs": pose_count, "pose_missing_pairs": len(seq)-pose_count,
                      "max_rgb_depth_delta_s": max(deltas), "image_hw": list(next(iter(dimensions))),
                      "K_native": seq.K.tolist(), "files": len(files), "bytes": total_bytes,
                      "png_decoded": decoded, "archives": archive_checks}
            summaries.append(result)
            print(json.dumps(result), flush=True)
    summary = {"catalog_sha256": fingerprint(catalog), "files_sha256": sha256(args.output / "files.jsonl"),
               "split_scene_check": "passed", "all_png_decode": "passed", "sequences": summaries}
    write_json(args.output / "summary.json", summary)
    return summary


def grid(frame, step=16):
    h, w = frame.depth.shape
    x, y = np.meshgrid(np.arange(step, w-step, step), np.arange(step, h-step, step))
    return np.column_stack([x.ravel(), y.ravel()]).astype(float)


def save_preview(path, frame0, frame1, points0, points1, labels):
    a, b = frame0.rgb.copy(), frame1.rgb.copy()
    canvas = np.concatenate([a, b], axis=1)
    w = a.shape[1]
    valid = np.flatnonzero(labels["matches0"] >= 0)
    for i in valid[::max(1, len(valid)//50)]:
        j = labels["matches0"][i]
        p = tuple(np.rint(points0[i]).astype(int))
        q = tuple(np.rint(points1[j]+[w, 0]).astype(int))
        cv2.line(canvas, p, q, (40, 220, 40), 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 2, (255, 80, 30), -1)
        uv = tuple(np.rint(labels["projected"][i]+[w, 0]).astype(int))
        cv2.circle(canvas, uv, 3, (255, 80, 30), 1)
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise OSError("preview image write failed")


def preview(catalog, args):
    spec = sequence_by_id(catalog, args.sequence)
    if spec["split"] == "test":
        raise ValueError("stage-2 QA refuses held-out test matching/supervision exports")
    audit_manifest = json.loads(args.audit_manifest.read_text())
    if audit_manifest.get("status") != "completed" or audit_manifest["catalog_sha256"] != fingerprint(catalog):
        raise ValueError("completed audit of the same catalog required")
    files_path = args.audit_manifest.parent / "files.jsonl"
    audit_summary = json.loads((args.audit_manifest.parent / "summary.json").read_text())
    if (sha256(args.audit_manifest.parent / "summary.json") != audit_manifest["summary_sha256"]
            or sha256(files_path) != audit_summary["files_sha256"]):
        raise ValueError("audit file index changed")
    # Verify the selected sequence against the frozen file hashes before using it.
    for line in files_path.read_text().splitlines():
        item = json.loads(line)
        if item["sequence"] == args.sequence and sha256(confined(args.data_root, item["path"])) != item["sha256"]:
            raise ValueError(f"dataset file changed: {item['path']}")
    config = yaml.safe_load(args.degradations.read_text())
    if set(config) != {"version", "seed", "levels"}:
        raise ValueError("invalid degradation configuration")
    levels = {name: Degradation(**values) for name, values in config["levels"].items()}
    seq = RGBDSequence(spec, args.data_root)
    if args.stride >= len(seq):
        raise ValueError("stride must be smaller than the synchronized sequence length")
    starts = np.unique(np.linspace(0, len(seq)-args.stride-1, args.pairs, dtype=int))
    results = []
    for start in starts:
        a, b = seq.load(int(start)), seq.load(int(start)+args.stride)
        if a.T_world_camera is None or b.T_world_camera is None:
            results.append({"frame0": int(start), "status": "reference_pose_missing"})
            continue
        p0, p1 = grid(a), grid(b)
        labels = correspondences(a, b, p0, p1)
        prefix = f"pair-{int(start):06d}"
        np.savez_compressed(args.output / (prefix+".npz"), points0=p0, points1=p1, **labels)
        save_preview(args.output / (prefix+".png"), a, b, p0, p1, labels)
        row = {"frame0": int(start), "frame1": int(start)+args.stride,
               "timestamp0": a.timestamp, "timestamp1": b.timestamp, "source_points": len(p0),
               "positive_grid_correspondences": int(np.sum(labels["matches0"] >= 0)),
               "reasons": dict(zip(*[x.tolist() for x in np.unique(labels["reason"], return_counts=True)])),
               "rgbd0_delta_s": abs(a.timestamp-a.depth_timestamp)}
        variants = []
        for name, level in levels.items():
            rgb, depth = degrade(a, level, config["seed"], name)
            np.savez_compressed(args.output / f"{prefix}-{name}.npz", rgb=rgb, depth=depth)
            cv2.putText(rgb, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 100, 30), 1)
            variants.append(rgb)
        montage = np.concatenate(variants, axis=1)
        cv2.imwrite(str(args.output / f"{prefix}-degradations.png"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        results.append(row)
    summary = {"sequence": args.sequence, "kind": "geometry_QA_grid_not_feature_matching_metrics",
               "audit_manifest_sha256": sha256(args.audit_manifest),
               "dataset_files_sha256": audit_summary["files_sha256"], "degradations": config,
               "supervision": {"radius_px": 3., "depth_abs_m": .05, "depth_relative": .02,
                               "grid_step_px": 16, "stride": args.stride}, "pairs": results}
    write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("configs/datasets-v1.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("extract", help="verify hashes and extract fresh sequence folders")
    prepare.add_argument("--sequence", required=True)
    inspect = sub.add_parser("audit", help="hash every file and decode all PNGs, without algorithm evaluation")
    inspect.add_argument("--output", type=Path, required=True)
    qa = sub.add_parser("preview", help="export development-only geometric supervision and seeded degradation")
    qa.add_argument("--sequence", required=True)
    qa.add_argument("--output", type=Path, required=True)
    qa.add_argument("--audit-manifest", type=Path, required=True)
    qa.add_argument("--degradations", type=Path, default=Path("configs/degradations-v1.yaml"))
    qa.add_argument("--pairs", type=int, default=3)
    qa.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    record = None
    try:
        catalog = load_catalog(args.catalog, args.data_root)
        if args.command == "extract":
            print(extract_sequence(sequence_by_id(catalog, args.sequence), args.data_root))
            return 0
        if args.command == "preview" and (args.pairs <= 0 or args.stride <= 0):
            raise ValueError("pairs and stride must be positive")
        record = run_record(args.output, catalog, args)
        result = audit(catalog, args) if args.command == "audit" else preview(catalog, args)
        record.update(status="completed", summary_sha256=sha256(args.output / "summary.json"))
        print(args.output)
    except (Exception, KeyboardInterrupt) as exc:
        if record is not None:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if record is not None:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(args.output / "manifest.json", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
