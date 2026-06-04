from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prune-only DSpar calibration sweeps.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset config paths.")
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[round(0.2 + (0.05 * idx), 2) for idx in range(27)],
        help="Coarse epsilon grid.",
    )
    parser.add_argument(
        "--comparison-target-set",
        action="append",
        default=[],
        help="Optional dataset-specific comparison targets formatted as dataset:r1,r2,...",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Calibration namespace seed. Official DSpar sparsification still uses its hard-coded seed 42.",
    )
    parser.add_argument("--output-root", default=str(ROOT / "results" / "dspar_frontier"))
    args = parser.parse_args()

    output = run_dspar_calibration(
        datasets=args.datasets,
        epsilons=args.epsilons,
        comparison_target_sets=args.comparison_target_set,
        seed=args.seed,
        output_root=args.output_root,
    )
    print(json.dumps(output, indent=2))


def run_dspar_calibration(
    *,
    datasets: list[str],
    epsilons: list[float],
    comparison_target_sets: list[str],
    seed: int,
    output_root: str | Path,
) -> dict[str, str]:
    from structure_faithful_gnn.analysis.artifacts import write_rows_csv, write_rows_json
    from structure_faithful_gnn.analysis.frontier import FRONTIER_COLUMNS, MATCHED_TARGET_COLUMNS, target_matches
    from structure_faithful_gnn.analysis.target_grids import DEFAULT_COMPARISON_TARGETS
    from structure_faithful_gnn.config import load_dataset_config
    from structure_faithful_gnn.data.loaders import load_dataset
    from structure_faithful_gnn.utils.io import ensure_dir

    output_root = ensure_dir(output_root)
    target_map = _merge_dataset_float_sets(DEFAULT_COMPARISON_TARGETS, comparison_target_sets)
    coarse_epsilons = sorted({round(float(value), 6) for value in epsilons if float(value) > 0})
    frontier_rows: list[dict[str, object]] = []
    selected_target_map: dict[str, list[float]] = {}
    for dataset_path in datasets:
        dataset_config = load_dataset_config(dataset_path)
        bundle = load_dataset(dataset_config)
        selected_target_map[bundle.name.lower()] = target_map[bundle.name.lower()]
        dataset_rows_by_epsilon: dict[float, dict[str, object]] = {}
        for epsilon in coarse_epsilons:
            row = _ensure_dspar_row(bundle=bundle, dataset_path=dataset_path, epsilon=epsilon, seed=seed, output_root=output_root)
            dataset_rows_by_epsilon[epsilon] = row

        refined_epsilons = set(dataset_rows_by_epsilon.keys())
        for target in target_map[bundle.name.lower()]:
            nearest = min(dataset_rows_by_epsilon.values(), key=lambda row: abs(float(row["achieved_edge_reduction"]) - float(target)))
            if abs(float(nearest["achieved_edge_reduction"]) - float(target)) <= 0.01:
                continue
            epsilon = float(nearest["epsilon"])
            for delta in (0.01, 0.02, 0.03, 0.04, 0.05):
                for sign in (-1.0, 1.0):
                    candidate = round(epsilon + (sign * delta), 6)
                    if candidate <= 0 or candidate in refined_epsilons:
                        continue
                    row = _ensure_dspar_row(bundle=bundle, dataset_path=dataset_path, epsilon=candidate, seed=seed, output_root=output_root)
                    dataset_rows_by_epsilon[candidate] = row
                    refined_epsilons.add(candidate)

        frontier_rows.extend(dataset_rows_by_epsilon.values())

    frontier_rows = sorted(frontier_rows, key=lambda row: (row["dataset"], float(row["epsilon"]), float(row["achieved_edge_reduction"])))
    matched_rows = target_matches(frontier_rows, targets_by_dataset=selected_target_map, tie_break=lambda row: (float(row["epsilon"]),))

    frontier_csv = output_root / "frontier.csv"
    frontier_json = output_root / "frontier.json"
    matched_csv = output_root / "matched_targets.csv"
    matched_json = output_root / "matched_targets.json"
    write_rows_csv(frontier_csv, frontier_rows, fieldnames=FRONTIER_COLUMNS)
    write_rows_json(frontier_json, frontier_rows)
    write_rows_csv(matched_csv, matched_rows, fieldnames=MATCHED_TARGET_COLUMNS)
    write_rows_json(matched_json, matched_rows)
    return {
        "frontier_csv": str(frontier_csv),
        "frontier_json": str(frontier_json),
        "matched_targets_csv": str(matched_csv),
        "matched_targets_json": str(matched_json),
    }


def _ensure_dspar_row(
    *,
    bundle,
    dataset_path: str,
    epsilon: float,
    seed: int,
    output_root: Path,
) -> dict[str, object]:
    from structure_faithful_gnn.artifacts import load_pruned_graph_artifact, write_pruned_graph_artifact
    from structure_faithful_gnn.baselines.dspar import dspar_sparsify
    from structure_faithful_gnn.calibration import frontier_row, graph_summary
    from structure_faithful_gnn.config import load_dataset_config
    from structure_faithful_gnn.types import PrunedGraphArtifact
    from structure_faithful_gnn.utils.graph import edge_pairs
    from structure_faithful_gnn.utils.io import ensure_dir

    run_dir = ensure_dir(output_root / bundle.name / "dspar" / "pruned" / f"seed_42-eps_{float(epsilon):g}")
    artifact_path = run_dir / "pruned_graph.json"
    if artifact_path.exists():
        _, artifact = load_pruned_graph_artifact(run_dir)
    else:
        start = time.perf_counter()
        pruned_edge_index, edge_weight, metadata = dspar_sparsify(
            bundle.num_nodes,
            bundle.edge_index,
            epsilon=float(epsilon),
        )
        runtime_sec = time.perf_counter() - start
        removed_edge_index = _removed_edges(bundle.edge_index, pruned_edge_index)
        artifact = PrunedGraphArtifact(
            dataset=bundle.name,
            method="dspar",
            pruning_seed=int(metadata["sampling_seed"]),
            pruned_edge_index=pruned_edge_index,
            removed_edge_index=removed_edge_index,
            before_edge_count=bundle.num_edges,
            after_edge_count=int(pruned_edge_index.shape[1]),
            runtime_sec=float(runtime_sec),
            edge_weight=edge_weight,
            metadata=metadata,
        )
        resolved = {
            "dataset": load_dataset_config(dataset_path).__dict__,
            "baseline": {"method": "dspar", "epsilon": float(epsilon)},
            "seed": int(metadata["sampling_seed"]),
            "calibration_seed_namespace": seed,
            "artifact_type": "pruned_graph",
        }
        write_pruned_graph_artifact(run_dir, resolved, artifact)
    stats = graph_summary(bundle, artifact.pruned_edge_index)
    return frontier_row(
        dataset=bundle.name,
        method="dspar",
        pruning_seed=int(artifact.pruning_seed),
        artifact=artifact,
        run_dir=run_dir,
        graph_stats=stats,
        epsilon=float(epsilon),
        saturated=False,
    )


def _removed_edges(original_edge_index: torch.Tensor, pruned_edge_index: torch.Tensor) -> torch.Tensor:
    from structure_faithful_gnn.utils.graph import edge_pairs

    pruned = {tuple(sorted(pair)) for pair in edge_pairs(pruned_edge_index)}
    removed = [tuple(pair) for pair in edge_pairs(original_edge_index) if tuple(sorted(pair)) not in pruned]
    if not removed:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(removed, dtype=torch.long).t().contiguous()


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
