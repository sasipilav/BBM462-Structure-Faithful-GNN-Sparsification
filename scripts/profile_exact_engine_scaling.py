from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from _bootstrap import bootstrap

bootstrap()

from structure_faithful_gnn.config import PruningConfig  # noqa: E402
from structure_faithful_gnn.gdv.exact_sparse import ExactSparseGDVResult, exact_sparse_graph_gdv  # noqa: E402
from structure_faithful_gnn.pruning._incremental_ext import require_incremental_extension  # noqa: E402
from structure_faithful_gnn.pruning.relshift import relshift_prune  # noqa: E402
from structure_faithful_gnn.types import DatasetBundle  # noqa: E402
from structure_faithful_gnn.utils.graph import adjacency_sets  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    selection_backend: str
    native_fusion: bool
    heap_storage_mode: str
    bridge_mode: str
    compaction_threshold: float


VARIANTS = {
    "active_mask_linear": Variant(
        name="active_mask_linear",
        selection_backend="linear_scan",
        native_fusion=False,
        heap_storage_mode="versioned",
        bridge_mode="global_tarjan",
        compaction_threshold=0.0,
    ),
    "step3_versioned": Variant(
        name="step3_versioned",
        selection_backend="versioned_heap",
        native_fusion=False,
        heap_storage_mode="versioned",
        bridge_mode="global_tarjan",
        compaction_threshold=0.0,
    ),
    "optimized_exact": Variant(
        name="optimized_exact",
        selection_backend="versioned_heap",
        native_fusion=True,
        heap_storage_mode="indexed",
        bridge_mode="lazy_exact",
        compaction_threshold=0.20,
    ),
}


class InMemoryGDVService:
    backend_name = "exact_sparse_in_memory"

    def __init__(self, raw: np.ndarray) -> None:
        self.raw = np.asarray(raw, dtype=np.float64)
        self.last_compute_info: dict[str, object] = {}

    def compute_graph_gdv(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        *,
        cache_namespace: str,
    ) -> np.ndarray:
        if self.raw.shape != (num_nodes, 15):
            raise ValueError(
                f"In-memory GDV shape mismatch: expected {(num_nodes, 15)}, got {self.raw.shape}."
            )
        self.last_compute_info = {
            "cache_hit": True,
            "cache_path": None,
            "backend": self.backend_name,
            "num_nodes": int(num_nodes),
            "num_edges": int(edge_index.shape[1]),
            "cache_namespace": cache_namespace,
        }
        return self.raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile exact RelShift engine scaling on deterministic sparse synthetic graphs. "
            "Initial 15-orbit GDVs are counted exactly from connected 2--4 node subsets."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--families", default="fixed,ba,clustered,hub")
    parser.add_argument("--sizes", default="50,100,200,400")
    parser.add_argument("--hub-sizes", default="30,50,75,100")
    parser.add_argument("--variants", default="active_mask_linear,step3_versioned,optimized_exact")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument("--mean-degree", type=float, default=6.0)
    parser.add_argument("--d-min", type=int, default=2)
    parser.add_argument("--guard-bridges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if any(number < 2 for number in result):
        raise ValueError("All graph sizes must be at least two.")
    return result


def _canonical_edges(edges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges if u != v})


def _random_tree_edges(num_nodes: int, rng: random.Random) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for node in range(1, num_nodes):
        parent = rng.randrange(node)
        edges.add((parent, node))
    return edges


def _fill_random_edges(
    edges: set[tuple[int, int]],
    *,
    num_nodes: int,
    target_edges: int,
    rng: random.Random,
    node_low: int = 0,
) -> None:
    maximum = num_nodes * (num_nodes - 1) // 2
    target_edges = min(maximum, max(len(edges), target_edges))
    while len(edges) < target_edges:
        u = rng.randrange(node_low, num_nodes)
        v = rng.randrange(node_low, num_nodes)
        if u == v:
            continue
        edges.add((min(u, v), max(u, v)))


def generate_graph(
    family: str,
    num_nodes: int,
    *,
    mean_degree: float,
    seed: int,
) -> list[tuple[int, int]]:
    rng = random.Random(seed * 1_000_003 + num_nodes * 97 + sum(map(ord, family)))
    target_edges = max(num_nodes - 1, int(round(mean_degree * num_nodes / 2.0)))

    if family == "fixed":
        edges = _random_tree_edges(num_nodes, rng)
        _fill_random_edges(edges, num_nodes=num_nodes, target_edges=target_edges, rng=rng)
        return sorted(edges)

    if family == "ba":
        attachment_count = max(1, min(3, num_nodes - 1))
        initial = min(num_nodes, attachment_count + 1)
        edges: set[tuple[int, int]] = {
            (u, v) for u in range(initial) for v in range(u + 1, initial)
        }
        degree = [0] * num_nodes
        repeated: list[int] = []
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
            repeated.extend((u, v))
        for node in range(initial, num_nodes):
            selected: set[int] = set()
            while len(selected) < min(attachment_count, node):
                selected.add(rng.choice(repeated) if repeated else rng.randrange(node))
            for target in selected:
                edges.add((target, node))
                degree[target] += 1
                degree[node] += 1
                repeated.extend((target, node))
        _fill_random_edges(edges, num_nodes=num_nodes, target_edges=target_edges, rng=rng)
        return sorted(edges)

    if family == "clustered":
        edges: set[tuple[int, int]] = set()
        for node in range(num_nodes):
            next_node = (node + 1) % num_nodes
            edges.add((min(node, next_node), max(node, next_node)))
        for node in range(num_nodes):
            next_two = (node + 2) % num_nodes
            edges.add((min(node, next_two), max(node, next_two)))
        _fill_random_edges(edges, num_nodes=num_nodes, target_edges=target_edges, rng=rng)
        return sorted(edges)

    if family == "hub":
        edges = {(0, node) for node in range(1, num_nodes)}
        _fill_random_edges(
            edges,
            num_nodes=num_nodes,
            target_edges=target_edges,
            rng=rng,
            node_low=1,
        )
        return sorted(edges)

    raise ValueError(f"Unsupported graph family: {family}")


def edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_bundle(name: str, num_nodes: int, edges: list[tuple[int, int]]) -> DatasetBundle:
    return DatasetBundle(
        name=name,
        x=torch.zeros((num_nodes, 1), dtype=torch.float32),
        y=torch.arange(num_nodes, dtype=torch.long) % 2,
        edge_index=edge_index(edges),
        train_idx=torch.tensor([0]),
        val_idx=torch.tensor([min(1, num_nodes - 1)]),
        test_idx=torch.tensor([min(2, num_nodes - 1)]),
    )


def make_config(
    variant: Variant,
    *,
    rho: float,
    d_min: int,
    guard_bridges: bool,
    profile_rounds: bool,
) -> PruningConfig:
    return PruningConfig(
        method="relshift",
        rho=rho,
        score_mode="relative",
        d_min=d_min,
        guard_bridges=guard_bridges,
        backend="exact_sparse_in_memory",
        options={
            "relshift_engine": "incremental_sequential_exact",
            "use_score_cache": True,
            "candidate_delta_cache_enabled": True,
            "write_edge_scores": False,
            "use_native_graph_state": True,
            "native_state_fusion": variant.native_fusion,
            "incremental_selection_backend": variant.selection_backend,
            "heap_storage_mode": variant.heap_storage_mode,
            "heap_rebuild_ratio": 4.0,
            "bridge_maintenance_mode": variant.bridge_mode,
            "adjacency_compaction_threshold": variant.compaction_threshold,
            "profile_rounds": profile_rounds,
            "write_runtime_profile": False,
            "profile_memory": False,
            "profile_native_kernel": False,
            "native_omp_threads": 1,
        },
    )


def graph_metrics(num_nodes: int, edges: list[tuple[int, int]]) -> dict[str, float | int]:
    adjacency = [set() for _ in range(num_nodes)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    degrees = [len(row) for row in adjacency]
    wedge_count = sum(degree * (degree - 1) // 2 for degree in degrees)
    triangle_edge_support_sum = 0
    for u, v in edges:
        triangle_edge_support_sum += len(adjacency[u] & adjacency[v])
    triangle_count = triangle_edge_support_sum // 3

    import heapq

    mutable_degree = degrees.copy()
    heap = [(degree, node) for node, degree in enumerate(mutable_degree)]
    heapq.heapify(heap)
    removed = [False] * num_nodes
    degeneracy = 0
    while heap:
        degree, node = heapq.heappop(heap)
        if removed[node] or degree != mutable_degree[node]:
            continue
        removed[node] = True
        degeneracy = max(degeneracy, degree)
        for neighbor in adjacency[node]:
            if not removed[neighbor]:
                mutable_degree[neighbor] -= 1
                heapq.heappush(heap, (mutable_degree[neighbor], neighbor))

    return {
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "mean_degree": (2.0 * len(edges) / num_nodes) if num_nodes else 0.0,
        "max_degree": max(degrees, default=0),
        "wedge_count": wedge_count,
        "triangle_count": triangle_count,
        "degeneracy": degeneracy,
    }


def _sum_round(metadata: dict[str, Any], key: str) -> float:
    return float(sum(float(row.get(key, 0.0)) for row in metadata.get("round_summaries", [])))


def _native_stat(metadata: dict[str, Any], key: str, default: float = 0.0) -> float:
    state = metadata.get("native_fused_state_statistics") or {}
    return float(state.get(key, default))


def _heap_stat(metadata: dict[str, Any], key: str, default: float = 0.0) -> float:
    state = metadata.get("versioned_heap_statistics") or {}
    return float(state.get(key, default))


def extract_run_metrics(
    *,
    result: Any,
    wall_runtime_sec: float,
    graph_info: dict[str, float | int],
    gdv_result: ExactSparseGDVResult,
    variant: Variant,
    family: str,
    rho: float,
    d_min: int,
    guard_bridges: bool,
) -> dict[str, Any]:
    metadata = result.metadata
    rows = metadata.get("round_summaries", [])
    return {
        "variant": variant.name,
        "family": family,
        **graph_info,
        "rho": rho,
        "d_min": d_min,
        "guard_bridges": guard_bridges,
        "exact_gdv_runtime_sec": gdv_result.runtime_sec,
        "connected_two_node_graphlets": gdv_result.connected_two_node_count,
        "connected_three_node_graphlets": gdv_result.connected_three_node_count,
        "connected_four_node_graphlets": gdv_result.connected_four_node_count,
        "wall_runtime_sec": wall_runtime_sec,
        "algorithm_runtime_sec": float(result.runtime_sec),
        "round_runtime_sec": _sum_round(metadata, "round_runtime_sec"),
        "bridge_runtime_sec": _sum_round(metadata, "bridge_runtime_sec"),
        "eligibility_runtime_sec": _sum_round(metadata, "eligibility_runtime_sec"),
        "native_score_runtime_sec": _sum_round(metadata, "native_score_runtime_sec"),
        "scalar_refresh_runtime_sec": _sum_round(metadata, "native_scalar_refresh_runtime_sec"),
        "mixed_correction_runtime_sec": _sum_round(metadata, "native_mixed_correction_runtime_sec"),
        "selected_update_runtime_sec": _sum_round(metadata, "selected_update_runtime_sec"),
        "heap_update_runtime_sec": _sum_round(metadata, "heap_update_runtime_sec"),
        "heap_pop_runtime_sec": _sum_round(metadata, "heap_pop_runtime_sec"),
        "active_edge_entries_scanned": _sum_round(metadata, "active_edge_id_entries_scanned"),
        "dirty_edge_entries_scanned": _sum_round(metadata, "dirty_edge_entries_scanned"),
        "bridge_adjacency_entries_visited": _sum_round(metadata, "bridge_adjacency_entries_visited"),
        "lazy_bridge_queries": _sum_round(metadata, "lazy_bridge_queries"),
        "lazy_bridge_adjacency_entries_visited": _sum_round(metadata, "lazy_bridge_adjacency_entries_visited"),
        "selected_four_node_pair_count": _native_stat(metadata, "selected_four_node_pairs_total"),
        "selected_affected_node_count": _native_stat(metadata, "node_rows_updated"),
        "achieved_budget": int(metadata["achieved_total_budget"]),
        "requested_budget": int(metadata["requested_total_budget"]),
        "after_edge_count": int(result.after_edge_count),
        "heap_max_estimated_bytes": _heap_stat(metadata, "heap_max_estimated_bytes"),
        "heap_current_estimated_bytes": _heap_stat(metadata, "heap_current_estimated_bytes"),
        "heap_rebuild_count": _heap_stat(metadata, "heap_rebuild_count_total"),
        "heap_stale_pop_count": _heap_stat(metadata, "heap_stale_pop_count_total"),
        "adjacency_compaction_count": _native_stat(metadata, "adjacency_compaction_count"),
        "candidate_delta_cache_bytes": (
            _native_stat(metadata, "candidate_delta_cache_bytes")
            if variant.native_fusion
            else float(result.before_edge_count * 2 * 15 * 8)
        ),
        "native_state_total_bytes": _native_stat(metadata, "native_numeric_state_total_bytes"),
        "score_scalarization_kernel": metadata.get("score_scalarization_kernel"),
        "removed_edges": [list(map(int, edge)) for edge in result.removed_edge_index.t().tolist()],
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


PROFILE_ONLY_KEYS = {
    "round_runtime_sec",
    "bridge_runtime_sec",
    "eligibility_runtime_sec",
    "native_score_runtime_sec",
    "scalar_refresh_runtime_sec",
    "mixed_correction_runtime_sec",
    "selected_update_runtime_sec",
    "heap_update_runtime_sec",
    "heap_pop_runtime_sec",
    "active_edge_entries_scanned",
    "dirty_edge_entries_scanned",
    "bridge_adjacency_entries_visited",
    "lazy_bridge_queries",
    "lazy_bridge_adjacency_entries_visited",
    "selected_four_node_pair_count",
    "selected_affected_node_count",
    "heap_max_estimated_bytes",
    "heap_current_estimated_bytes",
    "heap_rebuild_count",
    "heap_stale_pop_count",
    "adjacency_compaction_count",
    "candidate_delta_cache_bytes",
    "native_state_total_bytes",
}


def aggregate_runs(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = run_rows[0]
    aggregate: dict[str, Any] = {
        key: value
        for key, value in first.items()
        if key not in {"removed_edges"}
    }
    numeric_keys = [
        key
        for key, value in first.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    for key in numeric_keys:
        aggregate[key] = _median([float(row[key]) for row in run_rows])
    aggregate["repeat_count"] = len(run_rows)
    aggregate["removed_edges"] = first["removed_edges"]
    if any(row["removed_edges"] != first["removed_edges"] for row in run_rows[1:]):
        raise AssertionError("Repeated runs produced different exact trajectories.")
    return aggregate


def fit_exponents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "algorithm_runtime_sec",
        "round_runtime_sec",
        "bridge_runtime_sec",
        "scalar_refresh_runtime_sec",
        "selected_update_runtime_sec",
    )
    fits: list[dict[str, Any]] = []
    groups = sorted({(row["variant"], row["family"]) for row in rows})
    for variant, family in groups:
        group = [row for row in rows if row["variant"] == variant and row["family"] == family]
        for metric in metrics:
            valid = [row for row in group if float(row[metric]) > 0.0 and float(row["num_edges"]) > 0.0]
            if len(valid) < 3:
                continue
            x = np.log(np.asarray([float(row["num_edges"]) for row in valid], dtype=np.float64))
            y = np.log(np.asarray([float(row[metric]) for row in valid], dtype=np.float64))
            slope, intercept = np.polyfit(x, y, 1)
            predicted = intercept + slope * x
            residual = float(np.sum((y - predicted) ** 2))
            total = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - residual / total if total > 0.0 else 1.0
            fits.append(
                {
                    "variant": variant,
                    "family": family,
                    "metric": metric,
                    "edge_scaling_exponent": float(slope),
                    "intercept": float(intercept),
                    "r_squared": r_squared,
                    "point_count": len(valid),
                }
            )
    return fits


def write_csv(path: Path, rows: list[dict[str, Any]], *, exclude: set[str] | None = None) -> None:
    exclude = exclude or set()
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [key for key in rows[0].keys() if key not in exclude]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if not (0.0 <= args.rho < 1.0):
        raise ValueError("--rho must be in [0, 1).")

    families = _parse_csv_strings(args.families)
    regular_sizes = _parse_csv_ints(args.sizes)
    hub_sizes = _parse_csv_ints(args.hub_sizes)
    variant_names = _parse_csv_strings(args.variants)
    unknown_variants = [name for name in variant_names if name not in VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    require_incremental_extension()  # Exclude compilation/loading from all timed cases.

    aggregate_rows: list[dict[str, Any]] = []
    gdv_rows: list[dict[str, Any]] = []
    equivalence_failures: list[dict[str, Any]] = []
    case_count = 0
    started = time.perf_counter()

    for family in families:
        sizes = hub_sizes if family == "hub" else regular_sizes
        for num_nodes in sizes:
            if args.max_cases and case_count >= args.max_cases:
                break
            edges = generate_graph(
                family,
                num_nodes,
                mean_degree=args.mean_degree,
                seed=args.seed,
            )
            graph_info = graph_metrics(num_nodes, edges)
            graph_bundle = make_bundle(
                f"scaling-{family}-{num_nodes}-{args.seed}",
                num_nodes,
                edges,
            )
            adjacency = adjacency_sets(num_nodes, graph_bundle.edge_index)
            print(
                json.dumps({
                    "event": "gdv_start",
                    "family": family,
                    "num_nodes": num_nodes,
                    "num_edges": len(edges),
                }),
                flush=True,
            )
            gdv_result = exact_sparse_graph_gdv(adjacency)
            gdv_rows.append(
                {
                    "family": family,
                    **graph_info,
                    "runtime_sec": gdv_result.runtime_sec,
                    "connected_two_node_graphlets": gdv_result.connected_two_node_count,
                    "connected_three_node_graphlets": gdv_result.connected_three_node_count,
                    "connected_four_node_graphlets": gdv_result.connected_four_node_count,
                    "raw_gdv_bytes": int(gdv_result.raw.nbytes),
                }
            )

            write_csv(output_root / "exact_gdv_generation.partial.csv", gdv_rows)
            print(
                json.dumps({
                    "event": "gdv_complete",
                    "family": family,
                    "num_nodes": num_nodes,
                    "runtime_sec": gdv_result.runtime_sec,
                    "connected_four_node_graphlets": gdv_result.connected_four_node_count,
                }),
                flush=True,
            )

            reference_trajectory: list[list[int]] | None = None
            for variant_name in variant_names:
                print(
                    json.dumps({
                        "event": "variant_start",
                        "family": family,
                        "num_nodes": num_nodes,
                        "variant": variant_name,
                    }),
                    flush=True,
                )
                variant = VARIANTS[variant_name]
                run_rows: list[dict[str, Any]] = []
                for _ in range(args.repeats):
                    wall_start = time.perf_counter()
                    result = relshift_prune(
                        graph_bundle,
                        make_config(
                            variant,
                            rho=args.rho,
                            d_min=args.d_min,
                            guard_bridges=args.guard_bridges,
                            profile_rounds=False,
                        ),
                        InMemoryGDVService(gdv_result.raw),
                        seed=0,
                    )
                    wall_runtime = time.perf_counter() - wall_start
                    run_rows.append(
                        extract_run_metrics(
                            result=result,
                            wall_runtime_sec=wall_runtime,
                            graph_info=graph_info,
                            gdv_result=gdv_result,
                            variant=variant,
                            family=family,
                            rho=args.rho,
                            d_min=args.d_min,
                            guard_bridges=args.guard_bridges,
                        )
                    )
                aggregate = aggregate_runs(run_rows)

                profile_wall_start = time.perf_counter()
                profile_result = relshift_prune(
                    graph_bundle,
                    make_config(
                        variant,
                        rho=args.rho,
                        d_min=args.d_min,
                        guard_bridges=args.guard_bridges,
                        profile_rounds=True,
                    ),
                    InMemoryGDVService(gdv_result.raw),
                    seed=0,
                )
                profile_row = extract_run_metrics(
                    result=profile_result,
                    wall_runtime_sec=time.perf_counter() - profile_wall_start,
                    graph_info=graph_info,
                    gdv_result=gdv_result,
                    variant=variant,
                    family=family,
                    rho=args.rho,
                    d_min=args.d_min,
                    guard_bridges=args.guard_bridges,
                )
                if profile_row["removed_edges"] != aggregate["removed_edges"]:
                    raise AssertionError("Instrumented and uninstrumented trajectories differ.")
                for key in PROFILE_ONLY_KEYS:
                    aggregate[key] = profile_row[key]

                trajectory = aggregate["removed_edges"]
                if reference_trajectory is None:
                    reference_trajectory = trajectory
                elif trajectory != reference_trajectory:
                    equivalence_failures.append(
                        {
                            "family": family,
                            "num_nodes": num_nodes,
                            "variant": variant_name,
                            "reference_removed": reference_trajectory,
                            "observed_removed": trajectory,
                        }
                    )
                aggregate_rows.append(aggregate)
                write_csv(
                    output_root / "exact_engine_scaling.partial.csv",
                    aggregate_rows,
                    exclude={"removed_edges"},
                )
                print(
                    json.dumps({
                        "event": "variant_complete",
                        "family": family,
                        "num_nodes": num_nodes,
                        "variant": variant_name,
                        "algorithm_runtime_sec": aggregate["algorithm_runtime_sec"],
                        "achieved_budget": aggregate["achieved_budget"],
                    }),
                    flush=True,
                )
            case_count += 1
        if args.max_cases and case_count >= args.max_cases:
            break

    fit_rows = fit_exponents(aggregate_rows)
    write_csv(output_root / "exact_engine_scaling.csv", aggregate_rows, exclude={"removed_edges"})
    write_csv(output_root / "exact_engine_scaling_fits.csv", fit_rows)
    write_csv(output_root / "exact_gdv_generation.csv", gdv_rows)

    speedups: list[dict[str, Any]] = []
    lookup = {
        (row["family"], int(row["num_nodes"]), row["variant"]): row
        for row in aggregate_rows
    }
    for family in families:
        sizes = hub_sizes if family == "hub" else regular_sizes
        for num_nodes in sizes:
            baseline = lookup.get((family, num_nodes, "step3_versioned"))
            optimized = lookup.get((family, num_nodes, "optimized_exact"))
            if baseline and optimized:
                speedups.append(
                    {
                        "family": family,
                        "num_nodes": num_nodes,
                        "num_edges": baseline["num_edges"],
                        "algorithm_speedup": float(baseline["algorithm_runtime_sec"])
                        / max(float(optimized["algorithm_runtime_sec"]), 1e-15),
                        "profiled_round_speedup": float(baseline["round_runtime_sec"])
                        / max(float(optimized["round_runtime_sec"]), 1e-15),
                        "step3_algorithm_runtime_sec": baseline["algorithm_runtime_sec"],
                        "optimized_algorithm_runtime_sec": optimized["algorithm_runtime_sec"],
                        "step3_profiled_round_runtime_sec": baseline["round_runtime_sec"],
                        "optimized_profiled_round_runtime_sec": optimized["round_runtime_sec"],
                    }
                )
    write_csv(output_root / "exact_engine_speedups.csv", speedups)

    summary = {
        "graph_case_count": case_count,
        "engine_row_count": len(aggregate_rows),
        "repeat_count": args.repeats,
        "families": families,
        "regular_sizes": regular_sizes,
        "hub_sizes": hub_sizes,
        "variants": variant_names,
        "rho": args.rho,
        "d_min": args.d_min,
        "guard_bridges": args.guard_bridges,
        "all_trajectories_equal": not equivalence_failures,
        "equivalence_failures": equivalence_failures,
        "speedups": speedups,
        "elapsed_sec": time.perf_counter() - started,
        "notes": [
            "Initial GDVs are exact counts of all connected induced 2--4 node graphlets.",
            "Native extension compilation/loading is completed before timed cases.",
            "Timing medians use profile_rounds=false; one separate instrumented run supplies work counters and profiled round runtime.",
            "Peak process RSS is not compared in-process because ru_maxrss is monotone across cases.",
        ],
    }
    (output_root / "exact_engine_scaling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if equivalence_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
