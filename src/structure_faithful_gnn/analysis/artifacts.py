from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..utils.io import read_json, write_json

MASTER_RESULTS_COLUMNS = [
    "dataset",
    "model",
    "method",
    "seed",
    "target_rho",
    "epsilon",
    "lsp_variant",
    "lsp_k",
    "lsp_sparsity",
    "lsp_m",
    "lsp_l",
    "achieved_edge_reduction",
    "accuracy",
    "macro_f1",
    "train_sec",
    "infer_sec",
    "largest_component_ratio",
    "num_components",
    "clustering_delta",
    "mean_delta_sig",
    "median_delta_sig",
    "mean_delta_rel",
    "median_delta_rel",
    "runtime_sec",
    "pruning_seed",
    "pruned_graph_run_dir",
    "run_dir",
]


def load_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    _require_path(run_dir, kind="run directory")
    metrics_path = _require_path(run_dir / "metrics.json")
    resolved_path = _require_path(run_dir / "resolved_config.json")
    pruning_path = run_dir / "pruning_result.json"
    return {
        "run_dir": run_dir,
        "metrics": read_json(metrics_path),
        "resolved_config": read_json(resolved_path),
        "pruning_result": read_json(pruning_path) if pruning_path.exists() else None,
    }


def canonical_run_row(run_dir: str | Path) -> dict[str, Any]:
    artifacts = load_run_artifacts(run_dir)
    metrics = artifacts["metrics"]
    resolved = artifacts["resolved_config"]
    pruning_result = artifacts["pruning_result"] or {}

    dataset = resolved.get("dataset", {}).get("name")
    model = resolved.get("model", {}).get("name")
    seed = resolved.get("seed")
    method = _infer_method(resolved, pruning_result)
    if dataset is None or model is None or seed is None or method is None:
        raise ValueError(f"Incomplete artifact metadata in {run_dir}")

    baseline = resolved.get("baseline", {})
    target_rho = resolved.get("pruning", {}).get("rho")
    epsilon = baseline.get("epsilon")
    runtime_sec = pruning_result.get("runtime_sec", metrics.get("sparsify_sec"))
    pruning_seed = (
        metrics.get("metadata", {}).get("pruning_seed")
        if isinstance(metrics.get("metadata"), dict)
        else None
    )
    pruned_graph_run_dir = resolved.get("pruned_graph_artifact_dir")
    if pruned_graph_run_dir is None and isinstance(metrics.get("metadata"), dict):
        pruned_graph_run_dir = metrics["metadata"].get("pruned_graph_artifact_dir")

    return {
        "dataset": dataset,
        "model": model,
        "method": method,
        "seed": int(seed),
        "target_rho": _maybe_float(target_rho),
        "epsilon": _maybe_float(epsilon),
        "lsp_variant": baseline.get("variant"),
        "lsp_k": _maybe_float(baseline.get("k")),
        "lsp_sparsity": _maybe_float(baseline.get("sparsity")),
        "lsp_m": _maybe_float(baseline.get("m")),
        "lsp_l": _maybe_float(baseline.get("l", baseline.get("quantization_step"))),
        "achieved_edge_reduction": _required_float(metrics, "edge_reduction", run_dir),
        "accuracy": _required_float(metrics, "accuracy", run_dir),
        "macro_f1": _required_float(metrics, "macro_f1", run_dir),
        "train_sec": _required_float(metrics, "train_sec", run_dir),
        "infer_sec": _required_float(metrics, "infer_sec", run_dir),
        "largest_component_ratio": _required_float(metrics, "largest_component_ratio", run_dir),
        "num_components": _required_int(metrics, "num_components", run_dir),
        "clustering_delta": _required_float(metrics, "clustering_delta", run_dir),
        "mean_delta_sig": _maybe_float(metrics.get("mean_delta_sig")),
        "median_delta_sig": _maybe_float(metrics.get("median_delta_sig")),
        "mean_delta_rel": _maybe_float(metrics.get("mean_delta_rel")),
        "median_delta_rel": _maybe_float(metrics.get("median_delta_rel")),
        "runtime_sec": _maybe_float(runtime_sec),
        "pruning_seed": _maybe_float(pruning_seed),
        "pruned_graph_run_dir": None if pruned_graph_run_dir in {None, ""} else str(Path(pruned_graph_run_dir).resolve()),
        "run_dir": str(Path(run_dir).resolve()),
    }


def discover_method_run_dirs(root: str | Path, *, method: str) -> list[Path]:
    root = Path(root)
    _require_path(root, kind="analysis root")
    discovered: list[Path] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        resolved_path = run_dir / "resolved_config.json"
        if not resolved_path.exists():
            raise FileNotFoundError(f"Missing resolved_config.json next to {metrics_path}")
        resolved = read_json(resolved_path)
        pruning_path = run_dir / "pruning_result.json"
        pruning_result = read_json(pruning_path) if pruning_path.exists() else {}
        if _infer_method(resolved, pruning_result) == method:
            if method != "dense":
                _require_path(pruning_path)
            discovered.append(run_dir)
    if not discovered:
        raise FileNotFoundError(f"No {method} runs found under {root}")
    return discovered


def load_relshift_rows(root: str | Path, *, seed: int | None = None) -> list[dict[str, Any]]:
    return _load_rows(root, method="relshift", seed=seed)


def load_dspar_rows(root: str | Path, *, seed: int | None = None) -> list[dict[str, Any]]:
    return _load_rows(root, method="dspar", seed=seed)


def load_lsp_rows(root: str | Path, *, seed: int | None = None) -> list[dict[str, Any]]:
    return _load_rows(root, method="lsp", seed=seed)


def load_dense_rows(root: str | Path, *, seed: int | None = None) -> list[dict[str, Any]]:
    return _load_rows(root, method="dense", seed=seed)


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer fieldnames for empty rows at {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_json(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_json(path, {"rows": rows})


def _load_rows(root: str | Path, *, method: str, seed: int | None) -> list[dict[str, Any]]:
    rows = [canonical_run_row(run_dir) for run_dir in discover_method_run_dirs(root, method=method)]
    if seed is not None:
        rows = [row for row in rows if int(row["seed"]) == seed]
    if not rows:
        raise FileNotFoundError(f"No {method} rows found under {root} for seed={seed}")
    return sorted(rows, key=_row_sort_key)


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"],
        row["model"],
        row["method"],
        int(row["seed"]),
        float(row["target_rho"]) if row["target_rho"] is not None else -1.0,
        float(row["epsilon"]) if row["epsilon"] is not None else -1.0,
        str(row.get("lsp_variant") or ""),
        float(row["lsp_k"]) if row.get("lsp_k") is not None else -1.0,
        float(row["lsp_sparsity"]) if row.get("lsp_sparsity") is not None else -1.0,
        float(row["lsp_l"]) if row.get("lsp_l") is not None else -1.0,
    )


def _infer_method(resolved: dict[str, Any], pruning_result: dict[str, Any] | None) -> str | None:
    if pruning_result and pruning_result.get("method"):
        return str(pruning_result["method"])
    if resolved.get("pruning", {}).get("method"):
        return str(resolved["pruning"]["method"])
    if resolved.get("baseline", {}).get("method"):
        return str(resolved["baseline"]["method"])
    return "dense"


def _require_path(path: Path, *, kind: str = "artifact") -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind}: {path}")
    return path


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _required_float(payload: dict[str, Any], key: str, run_dir: str | Path) -> float:
    if key not in payload:
        raise KeyError(f"Missing required field '{key}' in {run_dir}")
    return float(payload[key])


def _required_int(payload: dict[str, Any], key: str, run_dir: str | Path) -> int:
    if key not in payload:
        raise KeyError(f"Missing required field '{key}' in {run_dir}")
    return int(payload[key])
