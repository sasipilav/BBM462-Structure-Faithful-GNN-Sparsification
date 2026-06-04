from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the common attainable reduction grid across RelShift, DSpar, and LSP.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset config paths used in the sweeps.")
    parser.add_argument("--relshift-frontier", required=True)
    parser.add_argument("--dspar-frontier", required=True)
    parser.add_argument("--lsp-frontier", required=True)
    parser.add_argument("--main-gap", type=float, default=0.015)
    parser.add_argument("--aux-gap", type=float, default=0.02)
    parser.add_argument(
        "--comparison-target-set",
        action="append",
        default=[],
        help="Optional dataset-specific comparison targets formatted as dataset:r1,r2,...",
    )
    parser.add_argument("--output-root", default=str(ROOT / "results" / "analysis" / "common_grid"))
    args = parser.parse_args()

    output = build_common_attainable_grid_bundle(
        datasets=args.datasets,
        relshift_frontier=args.relshift_frontier,
        dspar_frontier=args.dspar_frontier,
        lsp_frontier=args.lsp_frontier,
        main_gap=args.main_gap,
        aux_gap=args.aux_gap,
        comparison_target_sets=args.comparison_target_set,
        output_root=args.output_root,
    )
    print(json.dumps(output, indent=2))


def build_common_attainable_grid_bundle(
    *,
    datasets: list[str],
    relshift_frontier: str | Path,
    dspar_frontier: str | Path,
    lsp_frontier: str | Path,
    main_gap: float,
    aux_gap: float,
    comparison_target_sets: list[str],
    output_root: str | Path,
) -> dict[str, str]:
    from structure_faithful_gnn.analysis.artifacts import write_rows_csv, write_rows_json
    from structure_faithful_gnn.analysis.frontier import COMMON_GRID_COLUMNS, build_common_attainable_grid
    from structure_faithful_gnn.analysis.target_grids import DEFAULT_COMPARISON_TARGETS
    from structure_faithful_gnn.config import load_dataset_config
    from structure_faithful_gnn.utils.io import ensure_dir

    output_root = ensure_dir(output_root)
    target_map = _merge_dataset_float_sets(DEFAULT_COMPARISON_TARGETS, comparison_target_sets)
    dataset_config_map = {load_dataset_config(path).name: str(Path(path).resolve()) for path in datasets}
    selected_target_map = {dataset.lower(): target_map[dataset.lower()] for dataset in dataset_config_map}
    relshift_rows = pd.read_csv(relshift_frontier).to_dict(orient="records")
    dspar_rows = pd.read_csv(dspar_frontier).to_dict(orient="records")
    lsp_rows = pd.read_csv(lsp_frontier).to_dict(orient="records")

    common_rows = build_common_attainable_grid(
        relshift_rows=relshift_rows,
        dspar_rows=dspar_rows,
        lsp_rows=lsp_rows,
        targets_by_dataset=selected_target_map,
        main_gap=float(main_gap),
        aux_gap=float(aux_gap),
    )
    unmatched_rows = [row for row in common_rows if str(row["status"]) == "unmatched"]
    coverage_rows = _coverage_summary(common_rows)
    training_manifest = _training_manifest_rows(common_rows, dataset_config_map)

    common_csv = output_root / "common_attainable_grid.csv"
    common_json = output_root / "common_attainable_grid.json"
    unmatched_csv = output_root / "unmatched_targets.csv"
    coverage_csv = output_root / "method_coverage_summary.csv"
    manifest_csv = output_root / "training_manifest.csv"
    manifest_json = output_root / "training_manifest.json"
    write_rows_csv(common_csv, common_rows, fieldnames=COMMON_GRID_COLUMNS)
    write_rows_json(common_json, common_rows)
    write_rows_csv(unmatched_csv, unmatched_rows, fieldnames=COMMON_GRID_COLUMNS)
    write_rows_csv(coverage_csv, coverage_rows)
    write_rows_csv(manifest_csv, training_manifest)
    write_rows_json(manifest_json, training_manifest)
    return {
        "common_attainable_grid_csv": str(common_csv),
        "common_attainable_grid_json": str(common_json),
        "unmatched_targets_csv": str(unmatched_csv),
        "method_coverage_summary_csv": str(coverage_csv),
        "training_manifest_csv": str(manifest_csv),
        "training_manifest_json": str(manifest_json),
    }


def _coverage_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    for dataset, dataset_rows in grouped.items():
        summary.append(
            {
                "dataset": dataset,
                "total_targets": len(dataset_rows),
                "main_comparable_targets": sum(1 for row in dataset_rows if str(row["status"]) == "main_comparable"),
                "aux_comparable_targets": sum(1 for row in dataset_rows if str(row["status"]) == "aux_comparable"),
                "unmatched_targets": sum(1 for row in dataset_rows if str(row["status"]) == "unmatched"),
                "relshift_mean_gap": sum(float(row["relshift_abs_gap"]) for row in dataset_rows) / max(len(dataset_rows), 1),
                "dspar_mean_gap": sum(float(row["dspar_abs_gap"]) for row in dataset_rows) / max(len(dataset_rows), 1),
                "lsp_mean_gap": sum(float(row["lsp_abs_gap"]) for row in dataset_rows) / max(len(dataset_rows), 1),
            }
        )
    return sorted(summary, key=lambda row: row["dataset"])


def _training_manifest_rows(
    common_rows: list[dict[str, object]],
    dataset_config_map: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in common_rows:
        if str(row["status"]) != "main_comparable":
            continue
        dataset = str(row["dataset"])
        dataset_config = dataset_config_map[dataset]
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "dataset_config": dataset_config,
                    "status": str(row["status"]),
                    "target_edge_reduction": float(row["target_edge_reduction"]),
                    "method": "relshift",
                    "achieved_edge_reduction": float(row["relshift_achieved_edge_reduction"]),
                    "artifact_run_dir": str(row["relshift_run_dir"]),
                    "target_rho": row.get("relshift_target_rho"),
                    "epsilon": None,
                    "lsp_variant": None,
                    "lsp_k": None,
                    "lsp_sparsity": None,
                    "lsp_l": None,
                },
                {
                    "dataset": dataset,
                    "dataset_config": dataset_config,
                    "status": str(row["status"]),
                    "target_edge_reduction": float(row["target_edge_reduction"]),
                    "method": "dspar",
                    "achieved_edge_reduction": float(row["dspar_achieved_edge_reduction"]),
                    "artifact_run_dir": str(row["dspar_run_dir"]),
                    "target_rho": None,
                    "epsilon": row.get("dspar_epsilon"),
                    "lsp_variant": None,
                    "lsp_k": None,
                    "lsp_sparsity": None,
                    "lsp_l": None,
                },
                {
                    "dataset": dataset,
                    "dataset_config": dataset_config,
                    "status": str(row["status"]),
                    "target_edge_reduction": float(row["target_edge_reduction"]),
                    "method": "lsp",
                    "achieved_edge_reduction": float(row["lsp_achieved_edge_reduction"]),
                    "artifact_run_dir": str(row["lsp_run_dir"]),
                    "target_rho": None,
                    "epsilon": None,
                    "lsp_variant": row.get("lsp_variant"),
                    "lsp_k": row.get("lsp_k"),
                    "lsp_sparsity": row.get("lsp_sparsity"),
                    "lsp_l": row.get("lsp_l"),
                },
            ]
        )
    return sorted(rows, key=lambda item: (item["dataset"], float(item["target_edge_reduction"]), item["method"]))


def _merge_dataset_float_sets(
    defaults: dict[str, list[float]],
    overrides: list[str],
) -> dict[str, list[float]]:
    merged = {key.lower(): [float(value) for value in values] for key, values in defaults.items()}
    for override in overrides:
        if ":" not in override:
            raise ValueError(f"Invalid dataset float-set override: {override}")
        dataset, payload = override.split(":", 1)
        values = [float(item) for item in payload.split(",") if item]
        if not values:
            raise ValueError(f"Invalid override with no values: {override}")
        merged[dataset.lower()] = values
    return merged


if __name__ == "__main__":
    main()
