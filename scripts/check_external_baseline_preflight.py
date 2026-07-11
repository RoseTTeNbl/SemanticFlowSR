#!/usr/bin/env python3
"""Read-only local dependency preflight for external baseline reproduction."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def validate_preflight(lock_path: Path) -> tuple[dict[str, object], list[str]]:
    lock = json.loads(lock_path.read_text())
    errors: list[str] = []
    conda_exe = Path(str(lock["conda_exe"]))
    envs: set[str] = set()
    if not conda_exe.is_file():
        errors.append(f"missing conda executable: {conda_exe}")
    else:
        payload = json.loads(subprocess.check_output([str(conda_exe), "env", "list", "--json"], text=True))
        envs = {Path(value).name for value in payload.get("envs", [])}
        for env in lock.get("environments", []):
            if str(env) not in envs:
                errors.append(f"missing conda environment: {env}")
    for raw_path in lock.get("required_paths", []):
        if not Path(str(raw_path)).exists():
            errors.append(f"missing required path: {raw_path}")
    for raw_path, expected in lock.get("repositories", {}).items():
        path = Path(str(raw_path))
        if not (path / ".git").exists():
            errors.append(f"missing external repository: {raw_path}")
            continue
        actual = _git_revision(path)
        if actual != str(expected):
            errors.append(f"repository revision mismatch: {raw_path}: {actual}")
    for raw_path, expected in lock.get("files", {}).items():
        path = Path(str(raw_path))
        if not path.is_file():
            errors.append(f"missing required file: {raw_path}")
        elif _sha256(path) != str(expected):
            errors.append(f"file checksum mismatch: {raw_path}")
    return {
        "lock": str(lock_path),
        "environments_found": sorted(envs),
        "valid": not errors,
    }, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("configs/eval/external_baselines.local.lock.json"))
    args = parser.parse_args()
    summary, errors = validate_preflight(args.lock)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
