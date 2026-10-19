"""Command-line inspection and CPU/GPU infrastructure checks."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from .config import load_config
from .runtime import RunRecord, environment, write_json
from .smoke import run_smoke, selected_device


def main():
    parser = argparse.ArgumentParser(prog="scaf")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="read-only inventory; does not initialize CUDA")
    doctor.add_argument("--output", type=Path)
    smoke = sub.add_parser("smoke", help="bounded tensor/runtime check, not a model training run")
    smoke.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    smoke.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    smoke.add_argument("--gpu", type=int)
    smoke.add_argument("--allow-busy-gpu", action="store_true")
    smoke.add_argument("--output-root")
    smoke.add_argument("--steps", type=int)
    smoke.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.command == "doctor":
        import json
        result = environment()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    record = None
    try:
        config = load_config(args.config)
        overrides = {k: v for k, v in {"steps": args.steps, "output_root": args.output_root}.items() if v is not None}
        config = replace(config, **overrides)
        record = RunRecord(config, Path.cwd(), args.device, sys.argv)
        with selected_device(args.device, args.gpu, args.allow_busy_gpu) as gpu:
            record.manifest["selected_gpu"] = gpu
            record.manifest["shared_gpu_override"] = args.allow_busy_gpu
            result = run_smoke(config, record, args.resume)
        write_json(record.path / "result.json", result)
        record.finish("completed")
        print(record.path)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if record is not None:
            record.finish("failed", error=f"{type(exc).__name__}: {exc}")
            print(f"run record: {record.path}", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
