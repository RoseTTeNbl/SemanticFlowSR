"""Command builders for SFSR and external-baseline benchmark matrices."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import shlex

import yaml


@dataclass(frozen=True)
class MatrixCommand:
    argv: list[str]
    metadata: dict[str, str]

    def shell(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)


def load_matrix_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def write_command_plan(commands: list[MatrixCommand], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"command": cmd.shell(), **cmd.metadata} for cmd in commands]
    path.write_text(json.dumps(rows, indent=2, sort_keys=True))


def build_sfsr_matrix_commands(
    config: dict[str, Any],
    *,
    ckpt_by_vars: dict[int, str] | None = None,
    suite_groups: list[str] | None = None,
    methods: list[str] | None = None,
    python: str = "python",
) -> list[MatrixCommand]:
    selected_groups = _select_names(config.get("suite_groups", {}), suite_groups)
    selected_methods = _select_methods(config.get("methods", []), methods)
    commands: list[MatrixCommand] = []
    for group_name, suites in selected_groups.items():
        for method in selected_methods:
            method_id = str(method["id"])
            argv = [python, "scripts/run_experiment.py"]
            _append_flag_value(argv, "manifest", config["manifest"])
            _append_multi(argv, "manifest_suite", suites)
            _append_flag_value(argv, "manifest_root", config.get("root", "."))
            _append_flag_value(argv, "out", config.get("out", "results/sfsr_full_benchmark"))
            _append_flag_value(argv, "tag", _tag(method_id, group_name))
            _append_ckpt_map(argv, ckpt_by_vars or config.get("ckpt_by_vars", {}))
            _append_cli_args(argv, config.get("common_args", {}))
            _append_cli_args(argv, method.get("args", {}))
            commands.append(MatrixCommand(
                argv=argv,
                metadata={"method": method_id, "suite_group": group_name},
            ))
    return commands


def build_external_baseline_commands(
    config: dict[str, Any],
    *,
    suite_groups: list[str] | None = None,
    methods: list[str] | None = None,
    conda_exe: str = "conda",
    python: str = "python",
) -> list[MatrixCommand]:
    selected_groups = _select_names(config.get("suite_groups", {}), suite_groups)
    selected_methods = _select_methods(config.get("methods", []), methods)
    commands: list[MatrixCommand] = []
    for group_name, suites in selected_groups.items():
        for method in selected_methods:
            method_id = str(method["id"])
            env = str(method.get("env", ""))
            script = str(method["script"])
            argv = [python, script] if not env or env == "current" else [conda_exe, "run", "-n", env, python, script]
            if bool(method.get("accepts_manifest", True)):
                _append_flag_value(argv, "manifest", config["manifest"])
                _append_multi(argv, "suite", suites)
                _append_flag_value(argv, "root", config.get("root", "."))
                _append_flag_value(argv, "out", config.get("out", "results/external_baselines"))
                _append_flag_value(argv, "tag", _tag(method_id, group_name))
            _append_cli_args(argv, method.get("args", {}))
            commands.append(MatrixCommand(
                argv=argv,
                metadata={
                    "method": method_id,
                    "suite_group": group_name,
                    "env": env or "current",
                    "role": str(method.get("role", "external_comparison")),
                },
            ))
    return commands


def _select_names(items: dict[str, Any], selected: list[str] | None) -> dict[str, Any]:
    if not selected:
        return dict(items)
    missing = [name for name in selected if name not in items]
    if missing:
        raise ValueError(f"unknown suite groups: {missing}")
    return {name: items[name] for name in selected}


def _select_methods(items: list[dict[str, Any]], selected: list[str] | None) -> list[dict[str, Any]]:
    if not selected:
        return [item for item in items if bool(item.get("enabled", True))]
    by_id = {str(item["id"]): item for item in items}
    missing = [name for name in selected if name not in by_id]
    if missing:
        raise ValueError(f"unknown methods: {missing}")
    return [by_id[name] for name in selected]


def _append_ckpt_map(argv: list[str], mapping: dict[int, str] | dict[str, str]) -> None:
    if not mapping:
        return
    argv.append("--ckpt_by_vars")
    for key, value in sorted(mapping.items(), key=lambda kv: int(kv[0])):
        argv.append(f"{int(key)}:{value}")


def _append_cli_args(argv: list[str], args: dict[str, Any]) -> None:
    for key, value in args.items():
        if value is None or value == "full":
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            _append_multi(argv, key, value)
            continue
        if isinstance(value, dict):
            value = json.dumps(value, sort_keys=True)
        _append_flag_value(argv, key, value)


def _append_flag_value(argv: list[str], name: str, value: Any) -> None:
    argv.extend([f"--{name}", str(value)])


def _append_multi(argv: list[str], name: str, values: list[Any] | tuple[Any, ...]) -> None:
    argv.append(f"--{name}")
    argv.extend(str(value) for value in values)


def _tag(method: str, suite_group: str) -> str:
    return f"{_slug(method)}_{_slug(suite_group)}"


def _slug(value: str) -> str:
    return value.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
