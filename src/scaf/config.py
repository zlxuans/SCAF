"""Small, strict configuration for the stage-one runtime check."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import re

import yaml


@dataclass(frozen=True)
class SmokeConfig:
    name: str = "stage1_smoke"
    seed: int = 7
    degradation_seed: int = 13
    steps: int = 3
    threads: int = 1
    output_root: str = "runs"

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.name):
            raise ValueError("name must contain 1–64 letters, digits, underscores or hyphens")
        for key in ("seed", "degradation_seed", "steps", "threads"):
            value = getattr(self, key)
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not 1 <= self.steps <= 100 or not 1 <= self.threads <= 4:
            raise ValueError("smoke steps must be 1–100 and threads 1–4")
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("output_root must be a nonempty path")
        if max(self.seed, self.degradation_seed) >= 2**32:
            raise ValueError("seeds must be smaller than 2**32")

    def as_dict(self):
        return asdict(self)

    def fingerprint(self):
        # Destination and final step may change when resuming a diagnostic run.
        values = self.as_dict()
        for key in ("output_root", "steps"):
            values.pop(key)
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def load_config(path):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("configuration must be a mapping")
    unknown = set(values) - set(SmokeConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return SmokeConfig(**values)
