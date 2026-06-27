from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

# Test utilities are deliberately shared so the permanent pytest suite and the
# standalone 240-case validation use identical deterministic graph generators.
from tests.exact_equivalence_utils import bundle, config, graph_family, service  # noqa: E402
from structure_faithful_gnn.pruning.relshift import relshift_prune  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 240-case exact RelShift backend/cache equivalence matrix."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def _edge_rows(tensor: torch.Tensor) -> list[list[int]]:
    return [[int(u), int(v)] for u, v in tensor.t().tolist()]


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "gdv_cache"

    families = ("er", "ba", "sbm", "clustered", "hub")
    node_sizes = (8, 10)
    score_modes = ("absolute", "relative")
    d_mins = (0, 1, 2)
    bridge_options = (False, True)
    cache_options = (False, True)
    rho_values = (0.15, 0.30, 0.50)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    cached_trajectories: dict[tuple[object, ...], list[list[int]]] = {}
    case_id = 0
    start = time.perf_counter()

    for family in families:
        for num_nodes in node_sizes:
            for score_mode in score_modes:
                for d_min in d_mins:
                    for guard_bridges in bridge_options:
                        group_key = (family, num_nodes, score_mode, d_min, guard_bridges)
                        graph_seed = (
                            families.index(family) * 31
                            + num_nodes * 7
                            + score_modes.index(score_mode) * 11
                            + d_min * 5
                            + int(guard_bridges)
                        ) % 17
                        edges = graph_family(family, num_nodes, graph_seed)
                        rho = rho_values[
                            (families.index(family) + num_nodes + d_min + int(guard_bridges))
                            % len(rho_values)
                        ]
                        graph_bundle = bundle(
                            f"matrix-{family}-{num_nodes}-{graph_seed}",
                            num_nodes,
                            edges,
                        )

                        for cache_enabled in cache_options:
                            if args.max_cases and case_id >= args.max_cases:
                                break
                            case_start = time.perf_counter()
                            reference = relshift_prune(
                                graph_bundle,
                                config(
                                    rho=rho,
                                    score_mode=score_mode,
                                    d_min=d_min,
                                    guard_bridges=guard_bridges,
                                    candidate_delta_cache=cache_enabled,
                                    optimized=False,
                                ),
                                service(cache_root / f"case-{case_id}-reference"),
                                seed=0,
                            )
                            optimized = relshift_prune(
                                graph_bundle,
                                config(
                                    rho=rho,
                                    score_mode=score_mode,
                                    d_min=d_min,
                                    guard_bridges=guard_bridges,
                                    candidate_delta_cache=cache_enabled,
                                    optimized=True,
                                    compaction_threshold=0.20,
                                ),
                                service(cache_root / f"case-{case_id}-optimized"),
                                seed=0,
                            )

                            reference_removed = _edge_rows(reference.removed_edge_index)
                            optimized_removed = _edge_rows(optimized.removed_edge_index)
                            removed_equal = reference_removed == optimized_removed
                            final_equal = torch.equal(
                                reference.pruned_edge_index,
                                optimized.pruned_edge_index,
                            )
                            achieved_equal = (
                                int(reference.metadata["achieved_total_budget"])
                                == int(optimized.metadata["achieved_total_budget"])
                            )
                            cache_key = group_key
                            if cache_enabled:
                                cached_trajectories[cache_key] = optimized_removed
                                cache_pair_equal: bool | None = None
                            else:
                                cached_trajectories[("uncached",) + cache_key] = optimized_removed
                                cache_pair_equal = None

                            row = {
                                "case_id": case_id,
                                "graph_family": family,
                                "num_nodes": num_nodes,
                                "graph_seed": graph_seed,
                                "num_edges": len(edges),
                                "rho": rho,
                                "score_mode": score_mode,
                                "d_min": d_min,
                                "guard_bridges": guard_bridges,
                                "candidate_delta_cache": cache_enabled,
                                "requested_budget": int(reference.metadata["requested_total_budget"]),
                                "reference_achieved_budget": int(reference.metadata["achieved_total_budget"]),
                                "optimized_achieved_budget": int(optimized.metadata["achieved_total_budget"]),
                                "removed_equal": removed_equal,
                                "final_equal": final_equal,
                                "achieved_equal": achieved_equal,
                                "reference_runtime_sec": float(reference.runtime_sec),
                                "optimized_runtime_sec": float(optimized.runtime_sec),
                                "case_runtime_sec": time.perf_counter() - case_start,
                            }
                            rows.append(row)
                            if not (removed_equal and final_equal and achieved_equal):
                                failure = dict(row)
                                failure["edges"] = edges
                                failure["reference_removed"] = reference_removed
                                failure["optimized_removed"] = optimized_removed
                                failures.append(failure)
                            case_id += 1
                        if args.max_cases and case_id >= args.max_cases:
                            break
                    if args.max_cases and case_id >= args.max_cases:
                        break
                if args.max_cases and case_id >= args.max_cases:
                    break
            if args.max_cases and case_id >= args.max_cases:
                break
        if args.max_cases and case_id >= args.max_cases:
            break

    cache_pair_failures: list[dict[str, object]] = []
    for key, cached_removed in cached_trajectories.items():
        if key and key[0] == "uncached":
            continue
        uncached = cached_trajectories.get(("uncached",) + key)
        if uncached is not None and cached_removed != uncached:
            cache_pair_failures.append(
                {
                    "group": list(key),
                    "cached_removed": cached_removed,
                    "uncached_removed": uncached,
                }
            )

    csv_path = output_root / "exact_equivalence_matrix.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "case_count": len(rows),
        "expected_full_case_count": 240,
        "backend_failure_count": len(failures),
        "cache_pair_failure_count": len(cache_pair_failures),
        "all_backend_equal": not failures,
        "all_cache_pairs_equal": not cache_pair_failures,
        "elapsed_sec": time.perf_counter() - start,
        "failures": failures,
        "cache_pair_failures": cache_pair_failures,
    }
    (output_root / "exact_equivalence_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if failures or cache_pair_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
