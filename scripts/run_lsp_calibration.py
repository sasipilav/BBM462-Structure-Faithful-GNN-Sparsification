from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

DEFAULT_LSP_VARIANTS = ["lsp_p", "lsp_t"]
DEFAULT_LSP_K_VALUES = [1, 2, 3, 5, 7, 9]
DEFAULT_LSP_SPARSITY_VALUES = [round(0.05 * idx, 2) for idx in range(1, 20)]
DEFAULT_LSP_PROJECTION_STEPS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
REFINED_LSP_K_VALUES = [1, 2, 3, 5, 7, 9, 12]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prune-only LSP calibration sweeps.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset config paths.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_LSP_VARIANTS, choices=DEFAULT_LSP_VARIANTS)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_LSP_K_VALUES)
    parser.add_argument("--sparsity-values", nargs="+", type=float, default=DEFAULT_LSP_SPARSITY_VALUES)
    parser.add_argument("--projection-steps", nargs="+", type=float, default=DEFAULT_LSP_PROJECTION_STEPS)
    parser.add_argument(
        "--comparison-target-set",
        action="append",
        default=[],
        help="Optional dataset-specific comparison targets formatted as dataset:r1,r2,...",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default=str(ROOT / "results" / "lsp_frontier"))
    args = parser.parse_args()

    output = run_lsp_calibration(
        datasets=args.datasets,
        variants=args.variants,
        k_values=args.k_values,
        sparsity_values=args.sparsity_values,
        projection_steps=args.projection_steps,
        comparison_target_sets=args.comparison_target_set,
        seed=args.seed,
        output_root=args.output_root,
    )
    print(json.dumps(output, indent=2))


def run_lsp_calibration(
    *,
    datasets: list[str],
    variants: list[str],
    k_values: list[int],
    sparsity_values: list[float],
    projection_steps: list[float],
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
    frontier_rows: list[dict[str, object]] = []
    selected_target_map: dict[str, list[float]] = {}
    for dataset_path in datasets:
        dataset_config = load_dataset_config(dataset_path)
        bundle = load_dataset(dataset_config)
        selected_target_map[bundle.name.lower()] = target_map[bundle.name.lower()]
        dataset_rows: dict[tuple[object, ...], dict[str, object]] = {}
        for variant in sorted(set(variants)):
            for k in sorted({int(value) for value in k_values}):
                for sparsity in sorted({round(float(value), 6) for value in sparsity_values}):
                    local_steps = [None] if variant == "lsp_t" else sorted({float(value) for value in projection_steps})
                    for projection_step in local_steps:
                        row = _ensure_lsp_row(
                            bundle=bundle,
                            dataset_path=dataset_path,
                            variant=variant,
                            k=int(k),
                            sparsity_ratio=float(sparsity),
                            projection_step=projection_step,
                            seed=seed,
                            output_root=output_root,
                        )
                        dataset_rows[_row_key(row)] = row

        for variant in sorted(set(variants)):
            variant_rows = [row for row in dataset_rows.values() if str(row["lsp_variant"]) == variant]
            for target in target_map[bundle.name.lower()]:
                nearest_rows = sorted(
                    variant_rows,
                    key=lambda row: (
                        abs(float(row["achieved_edge_reduction"]) - float(target)),
                        _lsp_tie_break(row),
                    ),
                )[:3]
                for base_row in nearest_rows:
                    for refined in _refined_lsp_candidates(base_row):
                        candidate_key = (
                            str(refined["variant"]),
                            int(refined["k"]),
                            round(float(refined["sparsity"]), 6),
                            None if refined["l"] is None else round(float(refined["l"]), 6),
                        )
                        if candidate_key in dataset_rows:
                            continue
                        row = _ensure_lsp_row(
                            bundle=bundle,
                            dataset_path=dataset_path,
                            variant=str(refined["variant"]),
                            k=int(refined["k"]),
                            sparsity_ratio=float(refined["sparsity"]),
                            projection_step=refined["l"],
                            seed=seed,
                            output_root=output_root,
                        )
                        dataset_rows[_row_key(row)] = row

        frontier_rows.extend(dataset_rows.values())

    frontier_rows = sorted(
        frontier_rows,
        key=lambda row: (
            row["dataset"],
            str(row["lsp_variant"]),
            float(row["lsp_k"]),
            float(row["lsp_sparsity"]),
            -1.0 if row["lsp_l"] is None else float(row["lsp_l"]),
        ),
    )
    matched_rows = target_matches(frontier_rows, targets_by_dataset=selected_target_map, tie_break=_lsp_tie_break)

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


def _ensure_lsp_row(
    *,
    bundle,
    dataset_path: str,
    variant: str,
    k: int,
    sparsity_ratio: float,
    projection_step: float | None,
    seed: int,
    output_root: Path,
) -> dict[str, object]:
    from structure_faithful_gnn.artifacts import load_pruned_graph_artifact, write_pruned_graph_artifact
    from structure_faithful_gnn.baselines.lsp import lsp_prune
    from structure_faithful_gnn.calibration import frontier_row, graph_summary
    from structure_faithful_gnn.config import load_dataset_config
    from structure_faithful_gnn.types import PrunedGraphArtifact
    from structure_faithful_gnn.utils.io import ensure_dir

    resolved_sparsity = _resolve_lsp_sparsity(bundle, variant=variant, sparsity_ratio=sparsity_ratio)
    run_dir = ensure_dir(output_root / bundle.name / variant / "pruned" / _run_tag(seed, k, sparsity_ratio, projection_step))
    artifact_path = run_dir / "pruned_graph.json"
    if artifact_path.exists():
        _, artifact = load_pruned_graph_artifact(run_dir)
    else:
        result = lsp_prune(
            bundle.x,
            bundle.edge_index,
            variant=variant,
            k=int(k),
            sparsity=resolved_sparsity,
            quantization_step=projection_step,
            seed=seed,
        )
        artifact = PrunedGraphArtifact(
            dataset=bundle.name,
            method="lsp",
            pruning_seed=seed,
            pruned_edge_index=result.pruned_edge_index,
            removed_edge_index=result.removed_edge_index,
            before_edge_count=result.before_edge_count,
            after_edge_count=result.after_edge_count,
            runtime_sec=result.runtime_sec,
            metadata={
                **result.metadata,
                "variant": variant,
                "k": int(k),
                "sparsity": float(sparsity_ratio),
                "resolved_sparsity_dims": int(resolved_sparsity),
                "m": None,
                "l": None if projection_step is None else float(projection_step),
            },
        )
        resolved = {
            "dataset": load_dataset_config(dataset_path).__dict__,
            "baseline": {
                "method": "lsp",
                "variant": variant,
                "k": int(k),
                "sparsity": float(sparsity_ratio),
                "resolved_sparsity_dims": int(resolved_sparsity),
                "m": None,
                "l": None if projection_step is None else float(projection_step),
                "quantization_step": None if projection_step is None else float(projection_step),
            },
            "seed": seed,
            "artifact_type": "pruned_graph",
        }
        write_pruned_graph_artifact(run_dir, resolved, artifact)
    stats = graph_summary(bundle, artifact.pruned_edge_index)
    return frontier_row(
        dataset=bundle.name,
        method="lsp",
        pruning_seed=seed,
        artifact=artifact,
        run_dir=run_dir,
        graph_stats=stats,
        lsp_variant=variant,
        lsp_k=int(k),
        lsp_sparsity=float(sparsity_ratio),
        lsp_m=None,
        lsp_l=None if projection_step is None else float(projection_step),
        saturated=False,
    )


def _resolved_input_dim(bundle, *, variant: str) -> int:
    if variant == "lsp_p":
        return int(bundle.num_features * 2)
    return int(bundle.num_features)


def _resolve_lsp_sparsity(bundle, *, variant: str, sparsity_ratio: float) -> int:
    if sparsity_ratio <= 0:
        raise ValueError("LSP sparsity_ratio must be positive.")
    input_dim = _resolved_input_dim(bundle, variant=variant)
    if sparsity_ratio > 1:
        return max(1, min(input_dim, int(round(sparsity_ratio))))
    return max(1, min(input_dim, int(round(input_dim * sparsity_ratio))))


def _run_tag(seed: int, k: int, sparsity_ratio: float, projection_step: float | None) -> str:
    base = f"seed_{seed}-k_{int(k)}-s_{float(sparsity_ratio):g}"
    if projection_step is not None:
        return f"{base}-l_{float(projection_step):g}"
    return base


def _row_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        str(row["lsp_variant"]),
        int(row["lsp_k"]),
        round(float(row["lsp_sparsity"]), 6),
        None if row["lsp_l"] is None else round(float(row["lsp_l"]), 6),
    )


def _lsp_tie_break(row: dict[str, object]) -> tuple[object, ...]:
    variant_rank = 0 if str(row["lsp_variant"]) == "lsp_p" else 1
    return (
        variant_rank,
        float(row["lsp_k"]),
        float(row["lsp_sparsity"]),
        -1.0 if row["lsp_l"] is None else float(row["lsp_l"]),
    )


def _refined_lsp_candidates(base_row: dict[str, object]) -> list[dict[str, object]]:
    variant = str(base_row["lsp_variant"])
    k = int(base_row["lsp_k"])
    sparsity = float(base_row["lsp_sparsity"])
    projection_step = None if base_row["lsp_l"] is None else float(base_row["lsp_l"])
    k_index = REFINED_LSP_K_VALUES.index(k) if k in REFINED_LSP_K_VALUES else 0
    candidate_ks = {k}
    if k_index > 0:
        candidate_ks.add(REFINED_LSP_K_VALUES[k_index - 1])
    if k_index + 1 < len(REFINED_LSP_K_VALUES):
        candidate_ks.add(REFINED_LSP_K_VALUES[k_index + 1])
    candidate_sparsities = {max(0.02, min(0.98, round(sparsity + delta, 6))) for delta in (-0.04, -0.02, 0.0, 0.02, 0.04)}
    if variant == "lsp_t":
        return [
            {"variant": variant, "k": candidate_k, "sparsity": candidate_sparsity, "l": None}
            for candidate_k in sorted(candidate_ks)
            for candidate_sparsity in sorted(candidate_sparsities)
        ]
    candidate_steps = {round(projection_step * scale, 6) for scale in (0.5, 1.0, 2.0)} if projection_step is not None else {1.0}
    return [
        {"variant": variant, "k": candidate_k, "sparsity": candidate_sparsity, "l": candidate_step}
        for candidate_k in sorted(candidate_ks)
        for candidate_sparsity in sorted(candidate_sparsities)
        for candidate_step in sorted(candidate_steps)
        if candidate_step > 0
    ]


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
