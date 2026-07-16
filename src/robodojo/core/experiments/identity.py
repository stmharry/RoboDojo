"""Identity generation for one locally composed experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def task_input_hash(task_module: Path, task_config: Path) -> str:
    """Hash selected local task inputs without enforcing an upstream snapshot."""

    digest = hashlib.sha256(b"robodojo-task-input-v1\0")
    for path in (task_module, task_config):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def experiment_hash(*values: Any) -> str:
    digest = hashlib.sha256(b"robodojo-experiment-v4\0")
    for value in values:
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())
        digest.update(b"\0")
    return digest.hexdigest()
