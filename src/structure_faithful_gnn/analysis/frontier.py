from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable


FRONTIER_COLUMNS = [
    "dataset",
    "method",
    "pruning_seed",
    "target_rho",
    "epsilon",
    "lsp_variant",
    "lsp_k",
    "lsp_sparsity",
    "lsp_m",
    "lsp_l",
    "achieved_edge_reduction",
    "before_edge_count",
    "after_edge_count",
    "pruning_runtime_sec",
    "num_components",
    "largest_component_ratio",
    "clustering_delta",
    "isolated_node_count",
    "mean_delta_sig",
    "median_delta_sig",
    "mean_delta_rel",
    "median_delta_rel",
    "saturated",
    "run_dir",
]

MATCHED_TARGET_COLUMNS = [
    "dataset",
    "method",
    "target_edge_reduction",
    "achieved_edge_reduction",
    "abs_gap",
    "pruning_seed",
    "target_rho",
    "epsilon",
    "lsp_variant",
    "lsp_k",
    "lsp_sparsity",
    "lsp_m",
    "lsp_l",
    "saturated",
    "run_dir",
]

COMMON_GRID_COLUMNS = [
    "dataset",
    "target_edge_reduction",
    "status",
    "relshift_achieved_edge_reduction",
    "relshift_abs_gap",
    "relshift_run_dir",
    "relshift_target_rho",
    "dspar_achieved_edge_reduction",
    "dspar_abs_gap",
    "dspar_run_dir",
    "dspar_epsilon",
    "lsp_achieved_edge_reduction",
    "lsp_abs_gap",
    "lsp_run_dir",
    "lsp_variant",
    "lsp_k",
    "lsp_sparsity",
    "lsp_l",
]


def target_matches(
    rows: list[dict[str, Any]],
    *,
    targets_by_dataset: dict[str, list[float]],
    tie_break: Callable[[dict[str, Any]], tuple[Any, ...]] | None = None,
) -> list[dict[str, Any]]:
    matched_rows: list[dict[str, Any]] = []
    for dataset, targets in targets_by_dataset.items():
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        if not dataset_rows:
            raise ValueError(f"No frontier rows found for dataset={dataset}")
        for target in targets:
            matched = min(
                dataset_rows,
                key=lambda row: (
                    abs(float(row["achieved_edge_reduction"]) - float(target)),
                    (() if tie_break is None else tie_break(row)),
                ),
            )
            matched_rows.append(
                {
                    "dataset": dataset,
                    "method": matched["method"],
                    "target_edge_reduction": float(target),
                    "achieved_edge_reduction": float(matched["achieved_edge_reduction"]),
                    "abs_gap": abs(float(matched["achieved_edge_reduction"]) - float(target)),
                    "pruning_seed": int(matched["pruning_seed"]),
                    "target_rho": matched.get("target_rho"),
                    "epsilon": matched.get("epsilon"),
                    "lsp_variant": matched.get("lsp_variant"),
                    "lsp_k": matched.get("lsp_k"),
                    "lsp_sparsity": matched.get("lsp_sparsity"),
                    "lsp_m": matched.get("lsp_m"),
                    "lsp_l": matched.get("lsp_l"),
                    "saturated": bool(matched.get("saturated", False)),
                    "run_dir": matched["run_dir"],
                }
            )
    return sort_rows(matched_rows, keys=["dataset", "target_edge_reduction", "method"])


def build_common_attainable_grid(
    *,
    relshift_rows: list[dict[str, Any]],
    dspar_rows: list[dict[str, Any]],
    lsp_rows: list[dict[str, Any]],
    targets_by_dataset: dict[str, list[float]],
    main_gap: float,
    aux_gap: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, targets in targets_by_dataset.items():
        rel_dataset = [row for row in relshift_rows if str(row["dataset"]) == dataset]
        dspar_dataset = [row for row in dspar_rows if str(row["dataset"]) == dataset]
        lsp_dataset = [row for row in lsp_rows if str(row["dataset"]) == dataset]
        if not rel_dataset or not dspar_dataset or not lsp_dataset:
            raise ValueError(f"Incomplete frontier rows for dataset={dataset}")
        for target in targets:
            rel_row = nearest_by_achieved(rel_dataset, target)
            dspar_row = nearest_by_achieved(dspar_dataset, target)
            lsp_row = nearest_by_achieved(lsp_dataset, target, tie_break=_lsp_family_tie_break)
            rel_gap = abs(float(rel_row["achieved_edge_reduction"]) - float(target))
            dspar_gap = abs(float(dspar_row["achieved_edge_reduction"]) - float(target))
            lsp_gap = abs(float(lsp_row["achieved_edge_reduction"]) - float(target))
            max_gap = max(rel_gap, dspar_gap, lsp_gap)
            if max_gap <= main_gap:
                status = "main_comparable"
            elif max_gap <= aux_gap:
                status = "aux_comparable"
            else:
                status = "unmatched"
            rows.append(
                {
                    "dataset": dataset,
                    "target_edge_reduction": float(target),
                    "status": status,
                    "relshift_achieved_edge_reduction": float(rel_row["achieved_edge_reduction"]),
                    "relshift_abs_gap": rel_gap,
                    "relshift_run_dir": rel_row["run_dir"],
                    "relshift_target_rho": rel_row.get("target_rho"),
                    "dspar_achieved_edge_reduction": float(dspar_row["achieved_edge_reduction"]),
                    "dspar_abs_gap": dspar_gap,
                    "dspar_run_dir": dspar_row["run_dir"],
                    "dspar_epsilon": dspar_row.get("epsilon"),
                    "lsp_achieved_edge_reduction": float(lsp_row["achieved_edge_reduction"]),
                    "lsp_abs_gap": lsp_gap,
                    "lsp_run_dir": lsp_row["run_dir"],
                    "lsp_variant": lsp_row.get("lsp_variant"),
                    "lsp_k": lsp_row.get("lsp_k"),
                    "lsp_sparsity": lsp_row.get("lsp_sparsity"),
                    "lsp_l": lsp_row.get("lsp_l"),
                }
            )
    return sort_rows(rows, keys=["dataset", "target_edge_reduction"])


def nearest_by_achieved(
    rows: list[dict[str, Any]],
    target: float,
    *,
    tie_break: Callable[[dict[str, Any]], tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            abs(float(row["achieved_edge_reduction"]) - float(target)),
            (() if tie_break is None else tie_break(row)),
        ),
    )


def sort_rows(rows: list[dict[str, Any]], *, keys: list[str]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: tuple(_sort_value(row.get(key)) for key in keys),
    )


def unique_target_values(values: Iterable[float]) -> list[float]:
    return sorted({round(float(value), 6) for value in values})


def _sort_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _lsp_family_tie_break(row: dict[str, Any]) -> tuple[Any, ...]:
    variant_rank = 0 if str(row.get("lsp_variant")) == "lsp_p" else 1
    return (
        variant_rank,
        float(row.get("lsp_k") or 0.0),
        float(row.get("lsp_sparsity") or 0.0),
        float(row.get("lsp_l") or -1.0),
    )
