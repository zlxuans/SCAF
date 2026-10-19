"""Bounded tensor check with explicit GPU selection and resumable RNG state."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import os
import random
import tempfile

import numpy as np

from .runtime import gpu_inventory, sha256


@contextmanager
def selected_device(device, gpu=None, allow_busy=False):
    if device == "cpu":
        if gpu is not None or allow_busy:
            raise ValueError("GPU flags are valid only with --device cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        yield None
        return
    if gpu is None or gpu < 0:
        raise ValueError("CUDA requires an explicit nonnegative --gpu physical index")
    inventory = gpu_inventory()
    found = [row for row in inventory["gpus"] if row["index"] == gpu]
    if not found:
        raise RuntimeError(f"GPU {gpu} could not be inspected")
    info = found[0]
    lock_path = Path(tempfile.gettempdir()) / f"scaf-{info['uuid']}.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another SCAF diagnostic holds this GPU lock") from None
        if not allow_busy and (info["used_mib"] > 256 or info["utilization_pct"] > 10):
            raise RuntimeError("GPU is busy; defer the check or explicitly use --allow-busy-gpu for this tiny diagnostic")
        if info["total_mib"] - info["used_mib"] < 1024:
            raise RuntimeError("less than 1 GiB free; CUDA diagnostic deferred")
        # UUID avoids ambiguity if CUDA and nvidia-smi enumerate devices differently.
        os.environ["CUDA_VISIBLE_DEVICES"] = info["uuid"]
        yield info


def run_smoke(config, record, resume=None):
    # Import only after the caller has constrained CUDA visibility.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    record.manifest["runtime_environment"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_threads": config.threads,
    }
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    target = "cpu" if record.manifest["device"] == "cpu" else "cuda:0"
    if target != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("installed torch cannot initialize the selected CUDA device")
        torch.cuda.set_per_process_memory_fraction(0.05, 0)
        record.manifest["cuda"] = {"torch_cuda": torch.version.cuda,
                                   "device_name": torch.cuda.get_device_name(0)}
    completed, checksums = 0, []
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["config_fingerprint"] != config.fingerprint():
            raise ValueError("resume configuration differs")
        if checkpoint["commit"] != record.revision["commit"]:
            raise ValueError("resume requires the same source commit")
        if checkpoint["source_dirty"] or record.revision["dirty"]:
            raise ValueError("resume requires clean source checkouts")
        completed, checksums = checkpoint["completed_steps"], checkpoint["checksums"]
        if completed > config.steps:
            raise ValueError("resume step exceeds configured steps")
        torch.set_rng_state(checkpoint["rng_state"])
        record.manifest["resume"] = {"path": str(Path(resume).resolve()), "sha256": sha256(resume),
                                     "completed_steps": completed}
    record.save()
    for step in range(completed, config.steps):
        # Generate on CPU so GPU/CPU diagnostics use exactly the same random input.
        values = torch.rand(32, 32)
        result = (values.to(target) @ values.to(target).T).cpu()
        reference = values.numpy() @ values.numpy().T
        if not np.allclose(result.numpy(), reference, rtol=1e-4, atol=1e-5):
            raise RuntimeError("tensor result differs from CPU reference")
        checksum = float(result.sum())
        checksums.append(checksum)
        record.event(step=step + 1, checksum=checksum)
        state = {"completed_steps": step + 1, "checksums": checksums,
                 "rng_state": torch.get_rng_state(), "config_fingerprint": config.fingerprint(),
                 "commit": record.revision["commit"], "source_dirty": record.revision["dirty"]}
        temporary = record.path / "checkpoint.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(record.path / "checkpoint.pt")
    if target != "cpu":
        torch.cuda.synchronize(0)
        record.manifest["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
    return {"completed_steps": config.steps, "checksums": checksums,
            "note": "runtime validation only; no algorithm trained or benchmark measured"}
