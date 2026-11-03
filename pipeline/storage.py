"""Atomic file helpers: write to a temp file, then rename, so readers never see half a file."""
import json
import os
import re
import shutil
from pathlib import Path

import joblib


def run_dir(root, run_id: str) -> Path:
    """Directory for one pipeline run. Same run id -> same directory (idempotent re-runs)."""
    path = Path(root) / re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic(path: Path, write) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    os.replace(tmp, path)


def atomic_write_csv(df, path) -> None:
    _atomic(path, lambda tmp: df.to_csv(tmp, index=False))


def atomic_write_json(obj, path) -> None:
    _atomic(path, lambda tmp: tmp.write_text(json.dumps(obj, indent=2, default=str)))


def atomic_dump_joblib(obj, path) -> None:
    _atomic(path, lambda tmp: joblib.dump(obj, tmp))


def atomic_copy(src, dst) -> None:
    _atomic(dst, lambda tmp: shutil.copyfile(src, tmp))
