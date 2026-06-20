#!/usr/bin/env python
"""Download/materialize benchmark suites into the unified CSV + manifest layout."""
from __future__ import annotations

import argparse
import json
import time
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml

from semflow_sr.data.benchmark_loader import materialize_formula
from semflow_sr.data.benchmark_manifest import (
    BenchmarkSuiteSpec,
    BenchmarkTaskSpec,
    build_benchmark_index,
    write_benchmark_manifest,
)
from semflow_sr.data.benchmark_prepare import (
    PMLBFilter,
    filter_pmlb_metadata,
    materialize_arrays,
    parse_srsd_text_table,
    srsd_problem_names_from_siblings,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMULA_CFG_DIR = REPO_ROOT / "configs" / "data" / "formula_benchmarks"

SRSD_REPOS = {
    "srsd_feynman_easy": "yoshitomo-matsubara/srsd-feynman_easy",
    "srsd_feynman_medium": "yoshitomo-matsubara/srsd-feynman_medium",
    "srsd_feynman_hard": "yoshitomo-matsubara/srsd-feynman_hard",
    "srsd_feynman_easy_dummy": "yoshitomo-matsubara/srsd-feynman_easy_dummy",
    "srsd_feynman_medium_dummy": "yoshitomo-matsubara/srsd-feynman_medium_dummy",
    "srsd_feynman_hard_dummy": "yoshitomo-matsubara/srsd-feynman_hard_dummy",
}

EXTERNAL_REPOS = {
    "srbench": "https://github.com/cavalab/srbench.git",
    "llm_srbench": "https://github.com/deep-symbolic-mathematics/llm-srbench.git",
    "cp3_bench": "https://github.com/CP3-Origins/cp3-bench.git",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["formula_dev", "srsd_main", "srsd_dummy", "pmlb"])
    ap.add_argument("--out-root", default="data/benchmark_suites/materialized")
    ap.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    ap.add_argument("--index", default="data/benchmark_suites/benchmark_index.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--limit-per-suite", type=int, default=None)
    ap.add_argument("--pmlb-root", default="external/pmlb")
    ap.add_argument("--pmlb-limit", type=int, default=50)
    ap.add_argument("--pmlb-max-samples", type=int, default=5000)
    ap.add_argument("--pmlb-max-features", type=int, default=20)
    ap.add_argument("--pmlb-include-feynman", action="store_true")
    ap.add_argument("--pmlb-fetch-missing", action="store_true", help="fetch missing PMLB datasets through the pmlb package")
    ap.add_argument("--download-external", action="store_true")
    ap.add_argument("--external-root", default="external")
    a = ap.parse_args()

    suites: dict[str, list[BenchmarkTaskSpec]] = {}
    out_root = Path(a.out_root)

    if "formula_dev" in a.sources:
        suites.update(_prepare_formula_dev(out_root, seeds=a.seeds, limit=a.limit_per_suite))
    if "srsd_main" in a.sources:
        suites.update(_prepare_srsd(out_root, include_dummy=False, limit=a.limit_per_suite))
    if "srsd_dummy" in a.sources:
        suites.update(_prepare_srsd(out_root, include_dummy=True, only_dummy=True, limit=a.limit_per_suite))
    if "pmlb" in a.sources:
        suites.update(_prepare_pmlb(
            out_root,
            pmlb_root=Path(a.pmlb_root),
            limit=a.pmlb_limit,
            max_samples=a.pmlb_max_samples,
            max_features=a.pmlb_max_features,
            include_feynman=a.pmlb_include_feynman,
            fetch_missing=a.pmlb_fetch_missing,
        ))
    if a.download_external:
        _download_external_repos(Path(a.external_root))

    manifest = BenchmarkSuiteSpec(
        version="1.0",
        suites=suites,
        metadata={
            "sources": a.sources,
            "layout": "CSV splits with target column named 'target'",
        },
    )
    write_benchmark_manifest(manifest, a.manifest)
    _write_index(build_benchmark_index(manifest), Path(a.index))
    print(f"wrote {a.manifest}")
    print(f"wrote {a.index}")
    print(f"tasks: {sum(len(v) for v in suites.values())}")


def _prepare_formula_dev(out_root: Path, *, seeds: list[int], limit: int | None) -> dict[str, list[BenchmarkTaskSpec]]:
    out: dict[str, list[BenchmarkTaskSpec]] = {}
    for suite in ["nguyen", "constant", "livermore", "jin"]:
        entries = yaml.safe_load((FORMULA_CFG_DIR / f"{suite}.yaml").read_text())
        if limit is not None:
            entries = entries[: max(int(limit), 0)]
        specs: list[BenchmarkTaskSpec] = []
        for entry in entries:
            entry = dict(entry)
            entry.setdefault("suite", suite)
            task = materialize_formula(entry, seed=seeds[0])
            spec = materialize_arrays(
                task_id=f"{suite}/{task.name}",
                suite=suite,
                root=out_root,
                name=task.name,
                X_train=task.X_train,
                y_train=task.y_train,
                X_test=task.X_test,
                y_test=task.y_test,
                variable_names=task.variable_names,
                ground_truth=task.expression,
                domain="formula",
                metrics=["r2", "nmse", "symbolic_equivalence", "complexity"],
                split="dev",
                tags=["formula_dev"],
                source="configs/data/formula_benchmarks",
                metadata={"seed": seeds[0]},
            )
            specs.append(spec)
        out[suite] = specs
    return out


def _prepare_srsd(
    out_root: Path,
    *,
    include_dummy: bool,
    only_dummy: bool = False,
    limit: int | None,
) -> dict[str, list[BenchmarkTaskSpec]]:
    out: dict[str, list[BenchmarkTaskSpec]] = {}
    for suite, repo in SRSD_REPOS.items():
        is_dummy = suite.endswith("_dummy")
        if only_dummy and not is_dummy:
            continue
        if not include_dummy and is_dummy:
            continue
        specs = _materialize_srsd_repo(out_root, suite=suite, repo=repo, limit=limit)
        out[suite] = specs
    return out


def _materialize_srsd_repo(out_root: Path, *, suite: str, repo: str, limit: int | None) -> list[BenchmarkTaskSpec]:
    raw_cache = out_root.parent / "raw" / "srsd" / suite
    info = _load_json_cached(
        f"https://huggingface.co/datasets/{repo}/resolve/main/supp_info.json",
        raw_cache / "supp_info.json",
    )
    api = _load_json_cached(
        f"https://huggingface.co/api/datasets/{repo}",
        raw_cache / "dataset_api.json",
    )
    names = srsd_problem_names_from_siblings(api.get("siblings", []))
    if limit is not None:
        names = names[: max(int(limit), 0)]
    specs: list[BenchmarkTaskSpec] = []
    for name in names:
        X_train, y_train = _load_srsd_split(repo, "train", name, raw_cache)
        X_val, y_val = _load_srsd_split(repo, "validation", name, raw_cache)
        X_test, y_test = _load_srsd_split(repo, "test", name, raw_cache)
        meta = info.get(name, {})
        variables = [f"x{i}" for i in range(X_train.shape[1])]
        specs.append(materialize_arrays(
            task_id=f"{suite}/{name}",
            suite=suite,
            root=out_root,
            name=name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            variable_names=variables,
            ground_truth=meta.get("sympy_eq_str"),
            domain="physics",
            has_dummy_vars=suite.endswith("_dummy"),
            metrics=["r2", "nmse", "symbolic_equivalence", "complexity"]
            if not suite.endswith("_dummy") else ["r2", "nmse", "variable_selection", "complexity"],
            split="appendix" if suite.endswith("_dummy") else "main",
            tags=["srsd_feynman", "dummy"] if suite.endswith("_dummy") else ["srsd_feynman"],
            source=f"https://huggingface.co/datasets/{repo}",
            metadata={
                "dataset_class_key": meta.get("dataset_class_key"),
                "symbols": meta.get("symbols", []),
                "symbols_descs": meta.get("symbols_descs", []),
            },
        ))
        print(f"materialized {suite}/{name}")
    return specs


def _prepare_pmlb(
    out_root: Path,
    *,
    pmlb_root: Path,
    limit: int,
    max_samples: int,
    max_features: int,
    include_feynman: bool,
    fetch_missing: bool,
) -> dict[str, list[BenchmarkTaskSpec]]:
    stats = pd.read_csv(pmlb_root / "pmlb" / "all_summary_stats.tsv", sep="\t")
    names = filter_pmlb_metadata(
        stats,
        PMLBFilter(max_samples=max_samples, max_features=max_features, limit=None),
    )
    if not include_feynman:
        names = [n for n in names if not n.startswith("feynman_")]
    specs: list[BenchmarkTaskSpec] = []
    for name in names:
        path = pmlb_root / "datasets" / name / f"{name}.tsv.gz"
        if path.exists():
            df = pd.read_csv(path, sep="\t", compression="gzip")
        elif fetch_missing:
            fetched = _fetch_pmlb_dataset(name, pmlb_root)
            if fetched is None:
                continue
            X, y, cols = fetched
            df = pd.DataFrame(X, columns=cols)
            df["target"] = y
        else:
            continue
        if "target" not in df.columns:
            continue
        spec = _materialize_tabular_df(out_root, "pmlb_regression", name, df)
        specs.append(spec)
        if len(specs) >= int(limit):
            break
    return {"pmlb_regression": specs}


def _fetch_pmlb_dataset(name: str, cache_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    try:
        from pmlb import fetch_data
    except ImportError:
        print("skip PMLB fetch: install pmlb or omit --pmlb-fetch-missing")
        return None
    try:
        X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(cache_root / "cache"))
    except Exception as exc:
        print(f"skip {name}: {exc}")
        return None
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return X, y, [f"x{i}" for i in range(X.shape[1])]


def _materialize_tabular_df(out_root: Path, suite: str, name: str, df: pd.DataFrame) -> BenchmarkTaskSpec:
    cols = [c for c in df.columns if c != "target"]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))
    n_test = max(1, int(round(0.2 * len(df))))
    n_val = max(1, int(round(0.1 * len(df))))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return materialize_arrays(
        task_id=f"{suite}/{name}",
        suite=suite,
        root=out_root,
        name=name,
        X_train=df.iloc[train_idx][cols].to_numpy(float),
        y_train=df.iloc[train_idx]["target"].to_numpy(float),
        X_val=df.iloc[val_idx][cols].to_numpy(float),
        y_val=df.iloc[val_idx]["target"].to_numpy(float),
        X_test=df.iloc[test_idx][cols].to_numpy(float),
        y_test=df.iloc[test_idx]["target"].to_numpy(float),
        variable_names=[f"x{i}" for i in range(len(cols))],
        domain="black_box",
        metrics=["r2", "nmse", "complexity", "runtime"],
        split="main",
        tags=["pmlb", "srbench"],
        source="external/pmlb",
        metadata={"original_columns": cols},
    )


def _download_external_repos(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, url in EXTERNAL_REPOS.items():
        dst = root / name.replace("_", "-")
        if dst.exists():
            print(f"exists {dst}")
            continue
        subprocess.run(["git", "clone", "--depth", "1", url, str(dst)], check=True)


def _load_srsd_split(repo: str, split: str, name: str, raw_cache: Path) -> tuple[np.ndarray, np.ndarray]:
    raw_split = "val" if split == "validation" else split
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{raw_split}/{name}.txt"
    path = raw_cache / raw_split / f"{name}.txt"
    text = _load_text_cached(url, path)
    return parse_srsd_text_table(text)


def _load_json_cached(url: str, path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    data = json.loads(_read_url_text(url))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def _load_text_cached(url: str, path: Path) -> str:
    if path.exists():
        return path.read_text()
    text = _read_url_text(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def _read_url_text(url: str, retries: int = 4, timeout: int = 90) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return urlopen(url, timeout=timeout).read().decode()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt + 1 >= retries:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last}") from last


def _write_index(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
