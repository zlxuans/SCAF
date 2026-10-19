"""Fetch pinned official sources/weights and seal a local baseline asset manifest."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

import yaml

from scaf.runtime import command, sha256, write_json


def source_snapshot(spec, root):
    root = root.resolve()
    if command(["git", "rev-parse", "HEAD"], root) != spec["revision"]:
        raise ValueError(f"official source revision mismatch: {root}")
    if command(["git", "status", "--porcelain", "--untracked-files=no"], root):
        raise ValueError(f"modified official source: {root}")
    return {"root": str(root), "url": spec["url"], "revision": spec["revision"],
            "files": {name: sha256(root/name) for name in spec["files"]}}


def verify_assets(manifest_path, config):
    lock = json.loads(Path(manifest_path).read_text())
    if lock["version"] != "scaf-baseline-assets-v1" or lock["baseline_config"] != config:
        raise ValueError("baseline asset manifest/config mismatch")
    for name, spec in config["sources"].items():
        current = source_snapshot(spec, Path(lock["sources"][name]["root"]))
        if current != lock["sources"][name]:
            raise ValueError(f"official source changed: {name}")
    for name, item in lock["weights"].items():
        path = Path(item["path"])
        if (path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"] or
                item["sha256"] != config["weights"][name]["sha256"]):
            raise ValueError(f"baseline weights changed: {name}")
    return lock


def prepare(config, source_root, weights_root, output):
    if output.exists():
        verify_assets(output, config)
        return  # reuse only an exact, verified lock; never silently rewrite it
    source_root.mkdir(parents=True, exist_ok=True)
    weights_root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, spec in config["sources"].items():
        root = source_root/spec["directory"]
        if not root.exists():
            subprocess.run(["git", "clone", "--no-checkout", spec["url"], str(root)], check=True)
            subprocess.run(["git", "checkout", "--detach", spec["revision"]], cwd=root, check=True)
        sources[name] = source_snapshot(spec, root)
    weights = {}
    for name, spec in config["weights"].items():
        if "source" in spec:
            path = Path(sources[spec["source"]]["root"])/spec["file"]
            url = config["sources"][spec["source"]]["url"].removesuffix(".git")
            url += f"/blob/{sources[spec['source']]['revision']}/{spec['file']}"
        else:
            path, url = weights_root/spec["file"], spec["url"]
            if not path.exists():
                temporary = path.with_suffix(path.suffix+".download")
                try:
                    with urlopen(url, timeout=90) as response, temporary.open("xb") as handle:
                        while block := response.read(1024*1024):
                            handle.write(block)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        if path.stat().st_size < 1024:
            raise ValueError(f"weight file is too small (possibly a Git LFS pointer): {path}")
        weights[name] = {"path": str(path.resolve()), "source_url": url,
                         "sha256": sha256(path), "bytes": path.stat().st_size}
        if weights[name]["sha256"] != spec["sha256"]:
            raise ValueError(f"weight SHA256 differs from frozen baseline registry: {name}")
    write_json(output, {"version": "scaf-baseline-assets-v1", "baseline_config": config,
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "sources": sources, "weights": weights})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baselines-v1.yaml"))
    parser.add_argument("--source-root", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--weights-root", type=Path, default=Path("weights/baselines"))
    parser.add_argument("--output", type=Path, default=Path("weights/baselines/manifest.json"))
    args = parser.parse_args()
    prepare(yaml.safe_load(args.config.read_text()), args.source_root, args.weights_root, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
