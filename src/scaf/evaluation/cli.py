"""CPU-only stage-3 baseline evaluation; held-out test sequences are refused."""
import argparse
import os
from pathlib import Path
import sys

# Set before NumPy/Torch imports; leave the project's CUDA environment untouched.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

from .runner import load_settings, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage3.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--asset-manifest", type=Path, default=Path("weights/baselines/manifest.json"))
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, help="override development prefix length; 0 uses all stride-selected frames")
    args = parser.parse_args()
    try:
        config = load_settings(args.config)
        if args.max_frames is not None:
            if args.max_frames < 0 or args.max_frames == 1:
                raise ValueError("max frames must be zero or at least two")
            config["max_frames"] = args.max_frames
        print(run(config, args.data_root.resolve(), args.audit_manifest.resolve(),
                  args.asset_manifest.resolve(), args.output.resolve()))
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
