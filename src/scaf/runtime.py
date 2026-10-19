"""Versioned run records; environment inspection does not initialize CUDA."""
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import uuid


def command(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"command failed: {args[0]}")
    return result.stdout.strip()


def git_info(root):
    try:
        commit = command(["git", "rev-parse", "HEAD"], root)
        dirty = bool(command(["git", "status", "--porcelain"], root))
    except (OSError, RuntimeError):
        raise RuntimeError("run from a Git checkout so the source revision can be recorded") from None
    return {"commit": commit, "dirty": dirty}


def gpu_inventory():
    fields = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu"
    try:
        text = command(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        rows = []
        for line in text.splitlines():
            i, uid, name, driver, total, used, util = [v.strip() for v in line.split(",")]
            rows.append(dict(index=int(i), uuid=uid, name=name, driver=driver,
                             total_mib=int(total), used_mib=int(used), utilization_pct=int(util)))
        return {"available": True, "gpus": rows}
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "gpus": [], "error": str(exc)}


def environment():
    packages = {}
    for name in ("scaf-research", "numpy", "PyYAML", "torch", "pytest"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "gpu_inventory": gpu_inventory(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class RunRecord:
    def __init__(self, config, root, device, invocation):
        root = Path(root).resolve()
        source_root = Path(__file__).resolve().parents[2]
        if source_root != root:
            raise RuntimeError("run from the source checkout root; use PYTHONPATH=$PWD/src for detached worktrees")
        self.revision = git_info(root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        output = Path(config.output_root)
        if not output.is_absolute():
            output = root / output
        self.path = output / f"{config.name}_{stamp}_{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.manifest = {"kind": "infrastructure_smoke_not_research_evidence", "status": "running",
                         "created_utc": datetime.now(timezone.utc).isoformat(),
                         "source": self.revision, "config": config.as_dict(),
                         "config_fingerprint": config.fingerprint(), "device": device,
                         "source_root": str(source_root),
                         "invocation": invocation, "environment": environment(),
                         "dataset_manifest": None, "split_manifest": None, "pretrained_weights": None}
        self.save()
        write_json(self.path / "config.json", config.as_dict())
        dependencies = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())
        (self.path / "dependencies.txt").write_text("\n".join(dependencies) + "\n", encoding="utf-8")

    def save(self):
        write_json(self.path / "manifest.json", self.manifest)

    def event(self, **values):
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, allow_nan=False) + "\n")

    def finish(self, status, **values):
        self.manifest.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(), **values)
        self.save()
