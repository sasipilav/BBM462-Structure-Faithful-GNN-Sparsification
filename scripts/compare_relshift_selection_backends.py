from __future__ import annotations

import argparse
import json
import statistics
import time
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
        description=(
            "Compare exact RelShift linear-scan and versioned-heap selection backends "
            "under identical graph, seed, guard, and GDV-cache conditions."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset YAML path.")
    parser.add_argument("--pruning", required=True, help="Base exact RelShift pruning YAML path.")
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output-root", default=str(ROOT / "results" / "selection_backend_comparison"))
    parser.add_argument("--cache-root", default=str(ROOT / "data" / "cached_gdv_selection_compare"))
    parser.add_argument("--native-omp-threads", type=int, default=1)
    parser.add_argument("--heap-rebuild-ratio", type=float, default=4.0)
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative.")
    if args.heap_rebuild_ratio < 1.0:
        raise ValueError("--heap-rebuild-ratio must be at least 1.0.")

    result = compare_selection_backends(
        dataset_path=args.dataset,
        pruning_path=args.pruning,
        rho=args.rho,
        seed=args.seed,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        output_root=args.output_root,
        cache_root=args.cache_root,
        native_omp_threads=args.native_omp_threads,
        heap_rebuild_ratio=args.heap_rebuild_ratio,
    )
    print(json.dumps(result, indent=2))


def compare_selection_backends(
    *,
    dataset_path: str | Path,
    pruning_path: str | Path,
    rho: float | None,
    seed: int,
    repeats: int,
    warmup_runs: int,
    output_root: str | Path,
    cache_root: str | Path,
    native_omp_threads: int,
    heap_rebuild_ratio: float,
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

    common_options = dict(base_config.options or {})
    common_options.update(
        {
            "relshift_engine": "incremental_sequential_exact",
            "use_score_cache": True,
            "write_edge_scores": False,
            "use_native_graph_state": True,
            "profile_rounds": True,
            "write_runtime_profile": True,
            "profile_memory": True,
            "native_omp_threads": int(native_omp_threads),
            "heap_rebuild_ratio": float(heap_rebuild_ratio),
        }
    )
    target_rho = float(base_config.rho if rho is None else rho)
    configs = {
        backend: replace(
            base_config,
            rho=target_rho,
            options={**common_options, "incremental_selection_backend": backend},
        )
        for backend in ("linear_scan", "versioned_heap")
    }

    output_root = ensure_dir(output_root)
    cache_root = ensure_dir(cache_root)
    gdv_service = GDVService(
        base_config.backend,
        cache_root=cache_root,
        orca_path=base_config.orca_path,
    )

    for warmup_idx in range(warmup_runs):
        for backend in ("linear_scan", "versioned_heap"):
            warmup_dir = ensure_dir(output_root / "warmup" / backend / f"run_{warmup_idx + 1}")
            prune_graph(
                bundle,
                configs[backend],
                seed=seed,
                gdv_service=gdv_service,
                artifact_dir=warmup_dir,
            )

    sequences: dict[str, torch.Tensor] = {}
    runs: dict[str, list[dict[str, object]]] = {"linear_scan": [], "versioned_heap": []}
    for run_idx in range(repeats):
        order = (
            ("linear_scan", "versioned_heap")
            if run_idx % 2 == 0
            else ("versioned_heap", "linear_scan")
        )
        for backend in order:
            run_dir = ensure_dir(output_root / "measured" / backend / f"run_{run_idx + 1}")
            start = time.perf_counter()
            result = prune_graph(
                bundle,
                configs[backend],
                seed=seed,
                gdv_service=gdv_service,
                artifact_dir=run_dir,
            )
            wall_runtime_sec = float(time.perf_counter() - start)
            torch.save(result.removed_edge_index, run_dir / "removed_edges.pt")
            torch.save(result.pruned_edge_index, run_dir / "pruned_edges.pt")

            if backend not in sequences:
                sequences[backend] = result.removed_edge_index.clone()
            elif not torch.equal(sequences[backend], result.removed_edge_index):
                raise RuntimeError(f"{backend} produced different edge sequences across repeated runs.")

            summary = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
            runs[backend].append(_run_row(run_idx + 1, wall_runtime_sec, summary))
            write_json(
                run_dir / "resolved_comparison_config.json",
                {
                    "dataset": asdict(dataset_config),
                    "pruning": asdict(configs[backend]),
                    "seed": seed,
                },
            )

    if not torch.equal(sequences["linear_scan"], sequences["versioned_heap"]):
        linear_edges = sequences["linear_scan"].t().tolist()
        heap_edges = sequences["versioned_heap"].t().tolist()
        mismatch = next(
            (
                idx
                for idx, (left, right) in enumerate(zip(linear_edges, heap_edges, strict=False))
                if left != right
            ),
            min(len(linear_edges), len(heap_edges)),
        )
        raise RuntimeError(f"Selection backend edge-sequence mismatch at round {mismatch + 1}.")

    medians = {
        backend: {
            key: statistics.median(float(row[key]) for row in rows)
            for key in _NUMERIC_KEYS
        }
        for backend, rows in runs.items()
    }
    linear_round = medians["linear_scan"]["round_wall_runtime_sec"]
    heap_round = medians["versioned_heap"]["round_wall_runtime_sec"]
    payload = {
        "dataset": bundle.name,
        "num_nodes": bundle.num_nodes,
        "num_edges": bundle.num_edges,
        "target_rho": target_rho,
        "seed": seed,
        "repeats": repeats,
        "sequence_equal": True,
        "removed_edge_count": int(sequences["linear_scan"].shape[1]),
        "runs": runs,
        "medians": medians,
        "round_speedup": linear_round / heap_round if heap_round > 0.0 else None,
        "round_runtime_reduction_percent": (
            100.0 * (1.0 - heap_round / linear_round) if linear_round > 0.0 else None
        ),
    }
    write_json(output_root / "comparison_summary.json", payload)
    return payload


_NUMERIC_KEYS = (
    "wall_runtime_sec",
    "algorithm_runtime_sec",
    "round_wall_runtime_sec",
    "bridge_runtime_sec",
    "eligibility_runtime_sec",
    "best_selection_runtime_sec",
    "heap_update_runtime_sec",
    "heap_pop_runtime_sec",
    "active_edge_id_entries_scanned",
    "dirty_edge_entries_scanned",
    "heap_rebuild_edge_entries_scanned",
    "heap_stale_entries_popped",
    "heap_max_size_observed",
    "heap_rebuild_count_total",
)


def _run_row(run_index: int, wall_runtime_sec: float, summary: dict[str, object]) -> dict[str, object]:
    wall = dict(summary.get("wall_clock", {}))
    timings = dict(summary.get("diagnostic_timing_totals_sec", {}))
    counts = dict(summary.get("count_totals", {}))
    selection = dict(summary.get("selection_backend", {}))
    return {
        "run": run_index,
        "wall_runtime_sec": wall_runtime_sec,
        "algorithm_runtime_sec": float(wall.get("algorithm_runtime_sec", 0.0)),
        "round_wall_runtime_sec": float(wall.get("round_wall_runtime_sec", 0.0)),
        "bridge_runtime_sec": float(timings.get("bridge_runtime_sec", 0.0)),
        "eligibility_runtime_sec": float(timings.get("eligibility_runtime_sec", 0.0)),
        "best_selection_runtime_sec": float(timings.get("best_selection_runtime_sec", 0.0)),
        "heap_update_runtime_sec": float(timings.get("heap_update_runtime_sec", 0.0)),
        "heap_pop_runtime_sec": float(timings.get("heap_pop_runtime_sec", 0.0)),
        "active_edge_id_entries_scanned": int(counts.get("active_edge_id_entries_scanned", 0)),
        "dirty_edge_entries_scanned": int(counts.get("dirty_edge_entries_scanned", 0)),
        "heap_rebuild_edge_entries_scanned": int(
            counts.get("heap_rebuild_edge_entries_scanned", 0)
        ),
        "heap_stale_entries_popped": int(counts.get("heap_stale_entries_popped", 0)),
        "heap_max_size_observed": int(selection.get("heap_max_size_observed", 0)),
        "heap_rebuild_count_total": int(selection.get("heap_rebuild_count_total", 0)),
    }


if __name__ == "__main__":
    main()
