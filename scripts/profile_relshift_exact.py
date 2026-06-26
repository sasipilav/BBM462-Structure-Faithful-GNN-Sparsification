from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import torch

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the exact sequential RelShift engine without changing its pruning semantics."
    )
    parser.add_argument("--dataset", required=True, help="Dataset YAML path.")
    parser.add_argument("--pruning", required=True, help="RelShift pruning YAML path.")
    parser.add_argument("--rho", type=float, default=None, help="Optional rho override.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output-root", default=str(ROOT / "results" / "relshift_exact_profile"))
    parser.add_argument("--cache-root", default=str(ROOT / "data" / "cached_gdv_profile"))
    parser.add_argument("--clear-cache", action="store_true", help="Clear the selected GDV cache before profiling.")
    parser.add_argument("--native-omp-threads", type=int, default=0)
    parser.add_argument("--profile-native-kernel", action="store_true")
    parser.add_argument("--profile-update-diagnostics", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative.")

    output = profile_relshift_exact(
        dataset_path=args.dataset,
        pruning_path=args.pruning,
        rho=args.rho,
        seed=args.seed,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        output_root=args.output_root,
        cache_root=args.cache_root,
        clear_cache=args.clear_cache,
        native_omp_threads=args.native_omp_threads,
        profile_native_kernel=args.profile_native_kernel,
        profile_update_diagnostics=args.profile_update_diagnostics,
    )
    print(json.dumps(output, indent=2))


def profile_relshift_exact(
    *,
    dataset_path: str | Path,
    pruning_path: str | Path,
    rho: float | None,
    seed: int,
    repeats: int,
    warmup_runs: int,
    output_root: str | Path,
    cache_root: str | Path,
    clear_cache: bool,
    native_omp_threads: int,
    profile_native_kernel: bool,
    profile_update_diagnostics: bool,
) -> dict[str, object]:
    from structure_faithful_gnn.config import load_dataset_config, load_pruning_config
    from structure_faithful_gnn.data.loaders import load_dataset
    from structure_faithful_gnn.gdv.backends import GDVService
    from structure_faithful_gnn.pruning.registry import prune_graph
    from structure_faithful_gnn.utils.io import ensure_dir, write_json

    dataset_config = load_dataset_config(dataset_path)
    bundle = load_dataset(dataset_config)
    base_config = load_pruning_config(pruning_path)
    if base_config.method.lower() != "relshift":
        raise ValueError("The pruning config must use method=relshift.")

    options = dict(base_config.options or {})
    options.update(
        {
            "relshift_engine": "incremental_sequential_exact",
            "profile_rounds": True,
            "write_runtime_profile": True,
            "profile_memory": True,
            "profile_update_diagnostics": bool(profile_update_diagnostics),
            "profile_native_kernel": bool(profile_native_kernel),
            "native_omp_threads": int(native_omp_threads),
            "write_edge_scores": False,
        }
    )
    pruning_config = replace(
        base_config,
        rho=float(base_config.rho if rho is None else rho),
        options=options,
    )

    output_root = ensure_dir(output_root)
    cache_root = Path(cache_root)
    if clear_cache and cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    gdv_service = GDVService(
        pruning_config.backend,
        cache_root=cache_root,
        orca_path=pruning_config.orca_path,
    )

    # Warm-ups use the same cache and process so extension build/import and first-touch effects
    # do not contaminate measured runs. Their artifacts remain available for inspection.
    for warmup_idx in range(warmup_runs):
        warmup_dir = ensure_dir(output_root / "warmup" / f"run_{warmup_idx + 1}")
        warmup_options = dict(pruning_config.options or {})
        warmup_options["profile_label"] = f"warmup_{warmup_idx + 1}"
        warmup_config = replace(pruning_config, options=warmup_options)
        prune_graph(bundle, warmup_config, seed=seed, gdv_service=gdv_service, artifact_dir=warmup_dir)

    run_rows: list[dict[str, object]] = []
    run_dirs: list[str] = []
    for run_idx in range(repeats):
        run_dir = ensure_dir(output_root / "measured" / f"run_{run_idx + 1}")
        run_options = dict(pruning_config.options or {})
        run_options["profile_label"] = f"measured_{run_idx + 1}"
        run_config = replace(pruning_config, options=run_options)
        result = prune_graph(bundle, run_config, seed=seed, gdv_service=gdv_service, artifact_dir=run_dir)

        torch.save(result.pruned_edge_index, run_dir / "pruned_edges.pt")
        torch.save(result.removed_edge_index, run_dir / "removed_edges.pt")
        write_json(run_dir / "pruning_result.json", result.to_dict())
        write_json(
            run_dir / "resolved_profile_config.json",
            {
                "dataset": asdict(dataset_config),
                "pruning": asdict(run_config),
                "seed": seed,
                "cache_root": str(cache_root),
            },
        )

        summary_path = run_dir / "runtime_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = _summary_row(run_idx + 1, summary)
        run_rows.append(row)
        run_dirs.append(str(run_dir))

    csv_path = output_root / "profile_runs.csv"
    json_path = output_root / "profile_runs.json"
    _write_rows_csv(csv_path, run_rows)
    write_json(json_path, {"runs": run_rows})

    return {
        "profile_runs_csv": str(csv_path),
        "profile_runs_json": str(json_path),
        "run_dirs": run_dirs,
        "cache_root": str(cache_root),
    }


def _summary_row(run_index: int, summary: dict[str, object]) -> dict[str, object]:
    run = dict(summary.get("run", {}))
    gdv = dict(summary.get("gdv", {}))
    wall = dict(summary.get("wall_clock", {}))
    timings = dict(summary.get("diagnostic_timing_totals_sec", {}))
    memory = dict(summary.get("memory", {}))
    cache = dict(summary.get("cache_behavior", {}))
    return {
        "run": run_index,
        "dataset": dict(summary.get("dataset", {})).get("name"),
        "target_rho": run.get("target_rho"),
        "requested_budget": run.get("requested_budget"),
        "achieved_budget": run.get("achieved_budget"),
        "gdv_cache_hit": gdv.get("cache_hit"),
        "algorithm_runtime_sec": wall.get("algorithm_runtime_sec"),
        "initial_gdv_runtime_sec": wall.get("initial_gdv_runtime_sec"),
        "round_wall_runtime_sec": wall.get("round_wall_runtime_sec"),
        "round_runtime_coverage_ratio": wall.get("round_runtime_coverage_ratio"),
        "bridge_runtime_sec": timings.get("bridge_runtime_sec"),
        "eligibility_runtime_sec": timings.get("eligibility_runtime_sec"),
        "native_score_runtime_sec": timings.get("native_score_runtime_sec"),
        "native_scalar_refresh_runtime_sec": timings.get("native_scalar_refresh_runtime_sec"),
        "best_selection_runtime_sec": timings.get("best_selection_runtime_sec"),
        "selected_update_runtime_sec": timings.get("selected_update_runtime_sec"),
        "native_graph_edge_removal_runtime_sec": timings.get("native_graph_edge_removal_runtime_sec"),
        "active_edge_list_rebuild_runtime_sec": timings.get("active_edge_list_rebuild_runtime_sec"),
        "reuse_ratio": cache.get("reuse_ratio"),
        "full_rescore_ratio": cache.get("full_rescore_ratio"),
        "peak_rss_observed_mb": memory.get("peak_rss_observed_mb"),
        "known_numpy_state_total_mib": memory.get("known_numpy_state_total_mib"),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
