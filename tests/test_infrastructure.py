from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scaf.config import SmokeConfig, load_config
from scaf import smoke


def test_config_rejects_typos_and_invalid_limits(tmp_path):
    file = tmp_path / "invalid.yaml"
    file.write_text("seed: 7\nstepz: 4\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(file)
    for overrides in ({"steps": 0}, {"threads": 100}, {"seed": True}, {"name": "../escape"}):
        with pytest.raises(ValueError):
            SmokeConfig(**overrides)
    assert SmokeConfig().fingerprint() == replace(SmokeConfig(), steps=8).fingerprint()


def test_gpu_requires_explicit_index_and_refuses_shared_card(monkeypatch):
    def inventory():
        return {"gpus": [dict(index=0, uuid="test-scaf", total_mib=24576,
                              used_mib=10000, utilization_pct=60)]}
    monkeypatch.setattr(smoke, "gpu_inventory", inventory)
    with pytest.raises(ValueError, match="explicit"):
        with smoke.selected_device("cuda"):
            pytest.fail("missing GPU selection allowed")
    with pytest.raises(RuntimeError, match="busy"):
        with smoke.selected_device("cuda", gpu=0):
            pytest.fail("busy GPU allowed")


def test_cpu_check_hides_all_gpus_and_rejects_gpu_flags(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with smoke.selected_device("cpu"):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(ValueError):
        with smoke.selected_device("cpu", gpu=0):
            pass


def test_gpu_uuid_mapping_and_exclusive_diagnostic_lock(monkeypatch):
    monkeypatch.setattr(smoke, "gpu_inventory", lambda: {"gpus": [
        dict(index=1, uuid="scaf-test-exclusive", total_mib=24576, used_mib=0, utilization_pct=0)]})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with smoke.selected_device("cuda", gpu=1):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "scaf-test-exclusive"
        with pytest.raises(RuntimeError, match="lock"):
            with smoke.selected_device("cuda", gpu=1):
                pytest.fail("concurrent diagnostics acquired the same GPU")


def test_shared_override_still_refuses_low_free_memory(monkeypatch):
    monkeypatch.setattr(smoke, "gpu_inventory", lambda: {"gpus": [
        dict(index=0, uuid="scaf-test-low-memory", total_mib=24576, used_mib=24000, utilization_pct=90)]})
    with pytest.raises(RuntimeError, match="1 GiB"):
        with smoke.selected_device("cuda", gpu=0, allow_busy=True):
            pytest.fail("insufficient free memory allowed")


@pytest.fixture
def checkout(tmp_path):
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copytree(source / "configs", tmp_path / "configs")
    (tmp_path / ".gitignore").write_text("runs/\n__pycache__/\n")
    for args in (["init"], ["add", "."], ["-c", "user.name=Test", "-c", "user.email=test@example.com",
                                            "commit", "-m", "test fixture"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def invoke(checkout, *args):
    env = dict(os.environ, PYTHONPATH=str(checkout / "src"), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return subprocess.run([sys.executable, "-m", "scaf.cli", "smoke", *args], cwd=checkout,
                          env=env, text=True, capture_output=True)


def run_path(result):
    assert result.returncode == 0, result.stderr
    return Path(result.stdout.strip())


def test_cpu_reproducibility_resume_and_failure_record(checkout):
    first = run_path(invoke(checkout, "--steps", "1"))
    resumed = run_path(invoke(checkout, "--steps", "3", "--resume", str(first / "checkpoint.pt")))
    full = run_path(invoke(checkout, "--steps", "3"))
    result = lambda p: json.loads((p / "result.json").read_text())
    assert result(resumed)["checksums"] == result(full)["checksums"]
    metadata = json.loads((resumed / "manifest.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["source"]["dirty"] is False
    assert metadata["resume"]["sha256"]
    assert (resumed / "dependencies.txt").is_file()
    failed = invoke(checkout, "--device", "cuda")
    assert failed.returncode == 1
    records = [json.loads(p.read_text()) for p in (checkout / "runs").glob("*/manifest.json")]
    assert sum(r["status"] == "failed" for r in records) == 1


def test_resume_rejects_dirty_or_changed_source(checkout):
    first = run_path(invoke(checkout, "--steps", "1"))
    (checkout / "changed.txt").write_text("changed source")
    result = invoke(checkout, "--resume", str(first / "checkpoint.pt"))
    assert result.returncode == 1
    assert "clean source" in result.stderr


def test_snapshot_keeps_running_revision_when_main_changes(checkout, tmp_path_factory):
    script = Path(__file__).resolve().parents[1] / "scripts/snapshot.sh"
    (checkout / "scripts").mkdir()
    shutil.copy(script, checkout / "scripts/snapshot.sh")
    subprocess.run(["git", "add", "."], cwd=checkout, check=True, capture_output=True)
    commit = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "snapshot script"]
    subprocess.run(commit, cwd=checkout, check=True, capture_output=True)
    destination = tmp_path_factory.mktemp("worktree") / "pinned"
    env = dict(os.environ, SCAF_PYTHON=sys.executable)
    subprocess.run(["bash", "scripts/snapshot.sh", str(destination)], cwd=checkout, env=env, check=True, capture_output=True)
    before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination)
    (checkout / "later.txt").write_text("next stage")
    subprocess.run(["git", "add", "."], cwd=checkout, check=True, capture_output=True)
    subprocess.run(commit, cwd=checkout, check=True, capture_output=True)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination) == before
    assert not (destination / "later.txt").exists()
