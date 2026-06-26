from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prune-only RelShift calibration sweeps.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset config paths.")
    parser.add_argument("--pruning", required=True, help="Base pruning config path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--comparison-target-set",
        action="append",
        default=[],
        help="Optional dataset-specific comparison targets formatted as dataset:r1,r2,...",
    )
    parser.add_argument(
        "--rho-grid-set",
        action="append",
        default=[],
        help="Optional dataset-specific coarse rho grid formatted as dataset:r1,r2,...",
    )
    parser.add_argument("--output-root", default=str(ROOT / "results" / "relshift_frontier"))
    parser.add_argument("--write-edge-scores", action="store_true", help="Write large edge_scores.csv files for score-analysis runs.")
    args = parser.parse_args()

    output = run_relshift_calibration(
        datasets=args.datasets,
        pruning=args.pruning,
        seed=args.seed,
        comparison_target_sets=args.comparison_target_set,
        rho_grid_sets=args.rho_grid_set,
        output_root=args.output_root,
        write_edge_scores=args.write_edge_scores,
    )
    print(json.dumps(output, indent=2))


def run_relshift_calibration(
    *,
    datasets: list[str],
    pruning: str,
    seed: int,
    comparison_target_sets: list[str],
    rho_grid_sets: list[str],
    output_root: str | Path,
    write_edge_scores: bool = False,
) -> dict[str, str]:
    from structure_faithful_gnn.config import load_pruning_config
    from structure_faithful_gnn.utils.io import ensure_dir

    output_root = ensure_dir(output_root)
    base_pruning_config = load_pruning_config(pruning)
    return _run_relshift_calibration_with_base_config(
        datasets=datasets,
        base_pruning_config=base_pruning_config,
        pruning_config_payload_path=pruning,
        seed=seed,
        comparison_target_sets=comparison_target_sets,
        rho_grid_sets=rho_grid_sets,
        output_root=output_root,
        write_edge_scores=write_edge_scores,
    )


def _run_relshift_calibration_with_base_config(
    *,
    datasets: list[str],
    base_pruning_config,
    pruning_config_payload_path: str,
    seed: int,
    comparison_target_sets: list[str],
    rho_grid_sets: list[str],
    output_root: str | Path,
    write_edge_scores: bool = False,
) -> dict[str, str]:
    from structure_faithful_gnn.analysis.artifacts import write_rows_csv, write_rows_json
    from structure_faithful_gnn.analysis.frontier import FRONTIER_COLUMNS, MATCHED_TARGET_COLUMNS, target_matches
    from structure_faithful_gnn.analysis.target_grids import DEFAULT_COMPARISON_TARGETS, DEFAULT_RELSHIFT_CEILING, DEFAULT_RELSHIFT_RHO_GRID
    from structure_faithful_gnn.config import load_dataset_config
    from structure_faithful_gnn.data.loaders import load_dataset
    from structure_faithful_gnn.gdv.backends import GDVService, fit_standardization
    from structure_faithful_gnn.pruning._incremental_ext import require_incremental_extension
    from structure_faithful_gnn.utils.io import ensure_dir

    output_root = ensure_dir(output_root)
    if base_pruning_config.method.lower() != "relshift":
        raise ValueError("run_relshift_calibration only supports method=relshift.")
    if str((base_pruning_config.options or {}).get("relshift_engine", "orca_local")).strip().lower() == "incremental_sequential_exact":
        require_incremental_extension()

    target_map = _merge_dataset_float_sets(DEFAULT_COMPARISON_TARGETS, comparison_target_sets)
    rho_grid_map = _merge_dataset_float_sets(DEFAULT_RELSHIFT_RHO_GRID, rho_grid_sets)

    frontier_rows: list[dict[str, object]] = []
    selected_target_map: dict[str, list[float]] = {}
    for dataset_path in datasets:
        dataset_config = load_dataset_config(dataset_path)
        bundle = load_dataset(dataset_config)
        dataset_name = bundle.name.lower()
        selected_target_map[dataset_name] = target_map[dataset_name]
        targets = target_map[dataset_name]
        coarse_rhos = rho_grid_map[dataset_name]
        ceiling = DEFAULT_RELSHIFT_CEILING[dataset_name]
        dataset_rows_by_rho: dict[float, dict[str, object]] = {}
        gdv_service = GDVService(
            base_pruning_config.backend,
            cache_root="data/cached_gdv",
            orca_path=base_pruning_config.orca_path,
        )
        original_raw = gdv_service.compute_graph_gdv(bundle.num_nodes, bundle.edge_index, cache_namespace="original_full")
        stats = fit_standardization(original_raw)

        for rho in coarse_rhos:
            row = _ensure_relshift_row(
                bundle=bundle,
                dataset_config_dict=load_dataset_config(dataset_path).__dict__,
                base_pruning_config=base_pruning_config,
                rho=float(rho),
                seed=seed,
                frontier_root=output_root,
                pruning_config_payload_path=pruning_config_payload_path,
                gdv_service=gdv_service,
                original_raw=original_raw,
                stats=stats,
                write_edge_scores=write_edge_scores,
            )
            dataset_rows_by_rho[float(rho)] = row

        refined_rhos = set(dataset_rows_by_rho.keys())
        for target in targets:
            nearest = min(dataset_rows_by_rho.values(), key=lambda row: abs(float(row["achieved_edge_reduction"]) - float(target)))
            if abs(float(nearest["achieved_edge_reduction"]) - float(target)) <= 0.01:
                continue
            if bool(nearest.get("saturated")):
                continue
            nearest_rho = float(nearest["target_rho"])
            candidate_rhos = [nearest_rho - 0.02, nearest_rho - 0.01, nearest_rho + 0.01, nearest_rho + 0.02]
            for rho in candidate_rhos:
                rounded = round(float(rho), 6)
                if rounded <= 0 or rounded > ceiling or rounded in refined_rhos:
                    continue
                row = _ensure_relshift_row(
                    bundle=bundle,
                    dataset_config_dict=load_dataset_config(dataset_path).__dict__,
                    base_pruning_config=base_pruning_config,
                    rho=rounded,
                    seed=seed,
                    frontier_root=output_root,
                    pruning_config_payload_path=pruning_config_payload_path,
                    gdv_service=gdv_service,
                    original_raw=original_raw,
                    stats=stats,
                    write_edge_scores=write_edge_scores,
                )
                dataset_rows_by_rho[rounded] = row
                refined_rhos.add(rounded)

        frontier_rows.extend(dataset_rows_by_rho.values())

    frontier_rows = sorted(frontier_rows, key=lambda row: (row["dataset"], float(row["target_rho"]), float(row["achieved_edge_reduction"])))
    matched_rows = target_matches(frontier_rows, targets_by_dataset=selected_target_map, tie_break=lambda row: (float(row["target_rho"]),))

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


def _ensure_relshift_row(
    *,
    bundle,
    dataset_config_dict: dict[str, object],
    base_pruning_config,
    rho: float,
    seed: int,
    frontier_root: Path,
    pruning_config_payload_path: str,
    gdv_service,
    original_raw,
    stats,
    write_edge_scores: bool = False,
) -> dict[str, object]:
    from structure_faithful_gnn.artifacts import load_pruned_graph_artifact, write_pruned_graph_artifact
    from structure_faithful_gnn.calibration import frontier_row, graph_summary, pruning_result_to_artifact
    from structure_faithful_gnn.metrics.structural import compute_structural_metrics
    from structure_faithful_gnn.pruning.registry import prune_graph
    from structure_faithful_gnn.utils.io import ensure_dir

    pruning_options = dict(base_pruning_config.options or {})
    pruning_options["write_edge_scores"] = bool(write_edge_scores)
    pruning_config = replace(base_pruning_config, rho=float(rho), options=pruning_options)
    run_dir = ensure_dir(frontier_root / bundle.name / "relshift" / "pruned" / f"seed_{seed}-rho_{float(rho):g}")
    artifact_path = run_dir / "pruned_graph.json"
    artifact = None
    if artifact_path.exists():
        resolved, loaded_artifact = load_pruned_graph_artifact(run_dir)
        if _relshift_artifact_is_current(loaded_artifact, base_pruning_config=pruning_config):
            artifact = loaded_artifact
        else:
            print(f"[relshift calibration] rebuilding stale artifact at {run_dir}")
    if artifact is None:
        pruning_result = prune_graph(bundle, pruning_config, seed=seed, gdv_service=gdv_service, artifact_dir=run_dir)
        structural = compute_structural_metrics(
            bundle,
            bundle.clone_with_edges(pruning_result.pruned_edge_index, name=f"{bundle.name}_relshift_calibration"),
            gdv_service,
            original_raw=original_raw,
            stats=stats,
            eps=pruning_config.eps,
        )
        artifact = pruning_result_to_artifact(
            dataset=bundle.name,
            pruning_seed=seed,
            pruning_result=pruning_result,
            metadata_update={
                "artifact_kind": "pruned_graph",
                "structural_metrics": structural,
            },
        )
        resolved = {
            "dataset": dataset_config_dict,
            "pruning": asdict(pruning_config),
            "seed": seed,
            "artifact_type": "pruned_graph",
            "pruning_config_payload_path": pruning_config_payload_path,
        }
        write_pruned_graph_artifact(run_dir, resolved, artifact)
    stats = graph_summary(bundle, artifact.pruned_edge_index)
    saturated = bool(artifact.metadata.get("structural_guard_ceiling_hit", False))
    structural_metrics = artifact.metadata.get("structural_metrics", {})
    return frontier_row(
        dataset=bundle.name,
        method="relshift",
        pruning_seed=seed,
        artifact=artifact,
        run_dir=run_dir,
        graph_stats=stats,
        target_rho=float(rho),
        mean_delta_sig=structural_metrics.get("mean_delta_sig"),
        median_delta_sig=structural_metrics.get("median_delta_sig"),
        mean_delta_rel=structural_metrics.get("mean_delta_rel"),
        median_delta_rel=structural_metrics.get("median_delta_rel"),
        saturated=saturated,
    )


def _relshift_artifact_is_current(artifact, *, base_pruning_config) -> bool:
    metadata = artifact.metadata or {}
    requested_engine = str((base_pruning_config.options or {}).get("relshift_engine", "auto")).strip().lower() or "auto"
    requested_write_edge_scores = bool((base_pruning_config.options or {}).get("write_edge_scores", False))
    requested_runtime_profile = bool((base_pruning_config.options or {}).get("write_runtime_profile", False))
    requested_kernel_variant = str((base_pruning_config.options or {}).get("native_kernel_variant", "mask_count_v4_combinatorial")).strip().lower()
    expected_kernel_variant = requested_kernel_variant
    expected_kernel_versions = {
        "mask_count_v4_combinatorial": "mask_count_combinatorial_best_v4",
    }
    expected_kernel_version = expected_kernel_versions.get(expected_kernel_variant)
    if expected_kernel_version is None:
        return False
    if requested_engine == "incremental_sequential_exact":
        if metadata.get("relshift_engine") != "incremental_sequential_exact":
            return False
        if metadata.get("round_state_update_mode") != "single_edge_exact_incremental":
            return False
        if metadata.get("incremental_backend") != "native_cpp_extension":
            return False
        if metadata.get("cache_invalidation_mode") != "native_or_boolean_state_changed_incident_plus_delta_impacted":
            return False
        if metadata.get("state_update_mode") != "in_place_affected_rows":
            return False
        if metadata.get("cache_partition_mode") != "native_eligibility_valid_score_split" and not requested_write_edge_scores:
            return False
        if not requested_write_edge_scores and metadata.get("native_graph_state_mode") != "persistent_dynamic_csr":
            return False
        if not requested_write_edge_scores and metadata.get("candidate_delta_cache_mode") != "mixed_correction":
            return False
        if not requested_write_edge_scores and not metadata.get("native_edge_id_scoring", False):
            return False
        if not requested_write_edge_scores and not metadata.get("native_cached_best", False):
            return False
        if not requested_write_edge_scores and not metadata.get("native_cache_invalidation", False):
            return False
        if metadata.get("incremental_degree_support_mode") != "mutable_degree_on_demand_support":
            return False
        if metadata.get("native_kernel_variant") != expected_kernel_variant:
            return False
        if metadata.get("native_kernel_version") != expected_kernel_version:
            return False
        expected_selection_mode = "materialized_edge_scores" if requested_write_edge_scores else "native_best_with_array_cache"
        if metadata.get("native_selection_mode") != expected_selection_mode:
            return False
    elif requested_engine == "orca_local":
        if metadata.get("relshift_engine") != "orca_local":
            return False
        if metadata.get("round_state_update_mode") != "union_two_hop_exact_local_recount":
            return False
    else:
        if metadata.get("relshift_engine") not in {"orca_local", "incremental_sequential_exact"}:
            return False
        if metadata.get("round_state_update_mode") not in {"union_two_hop_exact_local_recount", "single_edge_exact_incremental"}:
            return False
    if metadata.get("score_norm") != "l1":
        return False
    if metadata.get("score_node_scope") != "edge_endpoints_only":
        return False
    if requested_write_edge_scores and not metadata.get("write_edge_scores", False):
        return False
    if requested_runtime_profile and not metadata.get("write_runtime_profile", False):
        return False
    structural_metrics = metadata.get("structural_metrics", {})
    required = {"mean_delta_sig", "median_delta_sig", "mean_delta_rel", "median_delta_rel"}
    return isinstance(structural_metrics, dict) and required.issubset(structural_metrics.keys())


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
