from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def _generate_connected_graph(num_nodes: int, num_edges: int, seed: int) -> list[tuple[int, int]]:
    maximum = num_nodes * (num_nodes - 1) // 2
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2.")
    if num_edges < num_nodes - 1 or num_edges > maximum:
        raise ValueError(f"num_edges must be in [{num_nodes - 1}, {maximum}].")
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    for node in range(1, num_nodes):
        parent = rng.randrange(node)
        edges.add((parent, node))
    while len(edges) < num_edges:
        u, v = rng.sample(range(num_nodes), 2)
        edges.add((u, v) if u < v else (v, u))
    return sorted(edges)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _sha256_edges(edge_index: torch.Tensor) -> str:
    array = edge_index.t().contiguous().cpu().numpy().astype(np.int64, copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _trajectory_ids(result, edges: list[tuple[int, int]]) -> list[int]:
    lookup = {edge: edge_id for edge_id, edge in enumerate(edges)}
    return [lookup[tuple(edge)] for edge in result.removed_edge_index.t().tolist()]


def _jaccard(left: list[int], right: list[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact uniform and orbit-weighted RelShift on one deterministic random graph.")
    parser.add_argument("--num-nodes", type=int, default=80)
    parser.add_argument("--num-edges", type=int, default=240)
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "phase2_orbit_weighted_random_graph"))
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be positive.")

    from structure_faithful_gnn.config import PruningConfig
    from structure_faithful_gnn.gdv.backends import GDVService
    from structure_faithful_gnn.gdv.exact_sparse import exact_sparse_graph_gdv
    from structure_faithful_gnn.pruning.relshift import relshift_prune
    from structure_faithful_gnn.types import DatasetBundle
    from structure_faithful_gnn.utils.graph import adjacency_sets

    class ExactSparseBackend:
        name = "exact_sparse"

        def compute(self, num_nodes: int, graph_edge_index: torch.Tensor) -> np.ndarray:
            return exact_sparse_graph_gdv(adjacency_sets(num_nodes, graph_edge_index)).raw

    def make_service(cache_root: Path) -> GDVService:
        service = GDVService.__new__(GDVService)
        service.cache_root = cache_root
        cache_root.mkdir(parents=True, exist_ok=True)
        service.backend = ExactSparseBackend()
        service.last_compute_info = {}
        return service

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = _generate_connected_graph(args.num_nodes, args.num_edges, args.seed)
    graph_edge_index = _edge_index(edges)
    bundle = DatasetBundle(
        name=f"random-n{args.num_nodes}-m{args.num_edges}-s{args.seed}",
        x=torch.zeros((args.num_nodes, 1), dtype=torch.float32),
        y=torch.arange(args.num_nodes) % 3,
        edge_index=graph_edge_index,
        train_idx=torch.tensor([0]),
        val_idx=torch.tensor([1]),
        test_idx=torch.tensor([2]),
    )
    adjacency = adjacency_sets(args.num_nodes, graph_edge_index)
    degrees = [len(neighbors) for neighbors in adjacency]

    cache_root = output_dir / "gdv_cache"
    service = make_service(cache_root)
    initial_started = time.perf_counter()
    initial_raw = service.compute_graph_gdv(
        args.num_nodes, graph_edge_index, cache_namespace="original_full"
    )
    initial_wall = time.perf_counter() - initial_started
    initial_detail = exact_sparse_graph_gdv(adjacency)
    np.testing.assert_array_equal(initial_raw, initial_detail.raw)

    weighted = [0.25, 1.5, 0.75, 2.0, 0.0, 0.5, 3.0, 1.25, 0.8, 1.7, 0.4, 2.3, 0.6, 1.1, 2.8]
    variants: dict[str, dict[str, object]] = {
        "uniform": {},
        "weighted_manual": {"orbit_weights": weighted},
        "leave_dense4_out": {"orbit_leave_out_groups": ["dense4"]},
    }

    def make_config(extra: dict[str, object], *, explainability: bool) -> PruningConfig:
        options: dict[str, object] = {
            "relshift_engine": "incremental_sequential_exact",
            "use_score_cache": True,
            "candidate_delta_cache_enabled": True,
            "write_edge_scores": False,
            "use_native_graph_state": True,
            "native_state_fusion": True,
            "incremental_selection_backend": "versioned_heap",
            "heap_storage_mode": "indexed",
            "heap_rebuild_ratio": 4.0,
            "bridge_maintenance_mode": "lazy_exact",
            "adjacency_compaction_threshold": 0.20,
            "native_omp_threads": 1,
            "profile_rounds": False,
            "write_runtime_profile": False,
            "profile_memory": False,
            "orbit_explainability_enabled": explainability,
            "orbit_checkpoint_rhos": [0.0, float(args.rho)],
            "orbit_checkpoint_store_node_snapshots": explainability,
        }
        options.update(extra)
        return PruningConfig(
            method="relshift",
            rho=float(args.rho),
            score_mode="relative",
            d_min=2,
            guard_bridges=True,
            backend="exact_sparse",
            options=options,
        )

    variant_results: dict[str, dict[str, object]] = {}
    trajectories: dict[str, list[int]] = {}
    for name, extra in variants.items():
        # Exclude one-time native initialization, allocator growth, and code-path
        # warm-up from the clean timing samples. The warm-up trajectory is still
        # checked against every measured repetition.
        warmup = relshift_prune(
            bundle,
            make_config(extra, explainability=False),
            make_service(cache_root),
            seed=args.seed,
        )
        warmup_trajectory = _trajectory_ids(warmup, edges)

        wall_times: list[float] = []
        algorithm_times: list[float] = []
        initial_times: list[float] = []
        trajectories_in_repeats: list[list[int]] = []
        representative = None
        for repeat in range(args.repeats):
            started = time.perf_counter()
            result = relshift_prune(
                bundle,
                make_config(extra, explainability=False),
                make_service(cache_root),
                seed=args.seed,
            )
            wall_times.append(time.perf_counter() - started)
            algorithm_times.append(float(result.runtime_sec))
            initial_times.append(float(result.metadata["initial_gdv_runtime_sec"]))
            trajectory = _trajectory_ids(result, edges)
            trajectories_in_repeats.append(trajectory)
            representative = result
        if any(item != warmup_trajectory for item in trajectories_in_repeats):
            raise RuntimeError(
                f"Variant {name} produced a trajectory inconsistent with its warm-up run."
            )
        if any(item != trajectories_in_repeats[0] for item in trajectories_in_repeats[1:]):
            raise RuntimeError(f"Variant {name} produced a non-deterministic trajectory across repeats.")

        artifact_dir = output_dir / f"validation_{name}"
        validation = relshift_prune(
            bundle,
            make_config(extra, explainability=True),
            make_service(cache_root),
            seed=args.seed,
            artifact_dir=artifact_dir,
        )
        if _trajectory_ids(validation, edges) != trajectories_in_repeats[0]:
            raise RuntimeError(f"Explainability logging changed the {name} pruning trajectory.")
        snapshot_path = Path(
            validation.metadata["orbit_explainability_artifacts"]["checkpoint_snapshots_path"]
        )
        with np.load(snapshot_path) as checkpoint_data:
            final_native_raw = np.asarray(checkpoint_data["raw_snapshots"][-1], dtype=np.int64)
            final_checkpoint_removed = int(checkpoint_data["removed_edge_counts"][-1])
        final_exact = exact_sparse_graph_gdv(
            adjacency_sets(args.num_nodes, validation.pruned_edge_index)
        )
        exact_final_equal = bool(np.array_equal(final_native_raw, final_exact.raw.astype(np.int64)))
        if not exact_final_equal:
            raise RuntimeError(f"Final native GDV mismatch for benchmark variant {name}.")
        if final_checkpoint_removed != validation.removed_edge_index.shape[1]:
            raise RuntimeError(f"Final checkpoint budget mismatch for benchmark variant {name}.")

        trajectory = trajectories_in_repeats[0]
        trajectories[name] = trajectory
        first_edges = [list(edges[edge_id]) for edge_id in trajectory[:10]]
        state_stats = representative.metadata["native_fused_state_statistics"]
        variant_results[name] = {
            "repeats": args.repeats,
            "median_wall_runtime_sec": statistics.median(wall_times),
            "min_wall_runtime_sec": min(wall_times),
            "max_wall_runtime_sec": max(wall_times),
            "median_algorithm_runtime_sec": statistics.median(algorithm_times),
            "median_cached_initial_gdv_runtime_sec": statistics.median(initial_times),
            "removed_edge_count": len(trajectory),
            "trajectory_edge_ids": trajectory,
            "first_removed_edges": first_edges,
            "final_edge_sha256": _sha256_edges(validation.pruned_edge_index),
            "exact_final_gdv_equal": exact_final_equal,
            "orbit_weight_fingerprint": validation.metadata["orbit_weight_spec"]["fingerprint"],
            "orbit_weight_nonzero_count": int(state_stats["orbit_weight_nonzero_count"]),
            "orbit_weight_sum": float(state_stats["orbit_weight_sum"]),
            "candidate_delta_cache_bytes": int(state_stats["candidate_delta_cache_bytes"]),
            "checkpoint_count": int(validation.metadata["orbit_checkpoint_count"]),
            "event_count": int(validation.metadata["orbit_explainability_event_count"]),
        }

    uniform_trajectory = trajectories["uniform"]
    for name, trajectory in trajectories.items():
        variant_results[name]["trajectory_position_match_fraction_vs_uniform"] = (
            sum(left == right for left, right in zip(uniform_trajectory, trajectory))
            / max(len(uniform_trajectory), len(trajectory), 1)
        )
        variant_results[name]["removed_edge_set_jaccard_vs_uniform"] = _jaccard(
            uniform_trajectory, trajectory
        )

    payload = {
        "benchmark_schema_version": "relshift-orbit-weighted-random-benchmark-v1",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "native_omp_threads": 1,
            "warmup_runs_per_variant": 1,
            "measured_repeats_per_variant": args.repeats,
        },
        "graph": {
            "edge_index_sha256": _sha256_edges(graph_edge_index),
            "num_nodes": args.num_nodes,
            "num_edges": args.num_edges,
            "seed": args.seed,
            "rho": args.rho,
            "minimum_degree": min(degrees),
            "maximum_degree": max(degrees),
            "mean_degree": float(np.mean(degrees)),
            "initial_exact_gdv_wall_sec": initial_wall,
            "initial_exact_gdv_kernel_sec": initial_detail.runtime_sec,
            "connected_two_node_graphlets": initial_detail.connected_two_node_count,
            "connected_three_node_graphlets": initial_detail.connected_three_node_count,
            "connected_four_node_graphlets": initial_detail.connected_four_node_count,
        },
        "variants": variant_results,
    }
    output_path = output_dir / "random_graph_benchmark.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **payload}, indent=2))


if __name__ == "__main__":
    main()
