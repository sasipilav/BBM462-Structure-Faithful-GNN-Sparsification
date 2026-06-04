from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .types import DatasetBundle, PrunedGraphArtifact, PruningResult
from .utils.graph import clustering_coefficient, connected_component_summary, isolated_node_count


def graph_summary(bundle: DatasetBundle, edge_index: torch.Tensor) -> dict[str, float | int]:
    num_components, largest_component_ratio = connected_component_summary(bundle.num_nodes, edge_index)
    clustering_delta = clustering_coefficient(bundle.num_nodes, edge_index) - clustering_coefficient(
        bundle.num_nodes,
        bundle.edge_index,
    )
    return {
        "num_components": int(num_components),
        "largest_component_ratio": float(largest_component_ratio),
        "clustering_delta": float(clustering_delta),
        "isolated_node_count": int(isolated_node_count(bundle.num_nodes, edge_index)),
    }


def pruning_result_to_artifact(
    *,
    dataset: str,
    pruning_seed: int,
    pruning_result: PruningResult,
    edge_weight: torch.Tensor | None = None,
    metadata_update: dict[str, Any] | None = None,
) -> PrunedGraphArtifact:
    metadata = dict(pruning_result.metadata)
    if metadata_update:
        metadata.update(metadata_update)
    return PrunedGraphArtifact(
        dataset=dataset,
        method=pruning_result.method,
        pruning_seed=pruning_seed,
        pruned_edge_index=pruning_result.pruned_edge_index,
        removed_edge_index=pruning_result.removed_edge_index,
        before_edge_count=pruning_result.before_edge_count,
        after_edge_count=pruning_result.after_edge_count,
        runtime_sec=pruning_result.runtime_sec,
        edge_weight=edge_weight,
        metadata=metadata,
    )


def frontier_row(
    *,
    dataset: str,
    method: str,
    pruning_seed: int,
    artifact: PrunedGraphArtifact,
    run_dir: str | Path,
    graph_stats: dict[str, float | int],
    target_rho: float | None = None,
    epsilon: float | None = None,
    lsp_variant: str | None = None,
    lsp_k: int | None = None,
    lsp_sparsity: float | int | None = None,
    lsp_m: float | int | None = None,
    lsp_l: float | None = None,
    isolated_node_count: int | None = None,
    mean_delta_sig: float | None = None,
    median_delta_sig: float | None = None,
    mean_delta_rel: float | None = None,
    median_delta_rel: float | None = None,
    saturated: bool = False,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "pruning_seed": int(pruning_seed),
        "target_rho": None if target_rho is None else float(target_rho),
        "epsilon": None if epsilon is None else float(epsilon),
        "lsp_variant": lsp_variant,
        "lsp_k": None if lsp_k is None else int(lsp_k),
        "lsp_sparsity": None if lsp_sparsity is None else float(lsp_sparsity),
        "lsp_m": None if lsp_m is None else float(lsp_m),
        "lsp_l": None if lsp_l is None else float(lsp_l),
        "achieved_edge_reduction": float(artifact.achieved_edge_reduction),
        "before_edge_count": int(artifact.before_edge_count),
        "after_edge_count": int(artifact.after_edge_count),
        "pruning_runtime_sec": float(artifact.runtime_sec),
        "num_components": int(graph_stats["num_components"]),
        "largest_component_ratio": float(graph_stats["largest_component_ratio"]),
        "clustering_delta": float(graph_stats["clustering_delta"]),
        "isolated_node_count": int(graph_stats["isolated_node_count"]) if isolated_node_count is None else int(isolated_node_count),
        "mean_delta_sig": None if mean_delta_sig is None else float(mean_delta_sig),
        "median_delta_sig": None if median_delta_sig is None else float(median_delta_sig),
        "mean_delta_rel": None if mean_delta_rel is None else float(mean_delta_rel),
        "median_delta_rel": None if median_delta_rel is None else float(median_delta_rel),
        "saturated": bool(saturated),
        "run_dir": str(Path(run_dir).resolve()),
    }
