from __future__ import annotations

import numpy as np

from ..gdv.backends import GDVService, NormalizationStats, apply_standardization, fit_standardization
from ..types import DatasetBundle
from ..utils.graph import (
    clustering_coefficient,
    connected_component_summary,
    degree_distribution_distances,
    edge_homophily,
    isolated_node_count,
)


def compute_structural_metrics(
    original: DatasetBundle,
    pruned: DatasetBundle,
    gdv_service: GDVService,
    *,
    original_raw: np.ndarray | None = None,
    stats: NormalizationStats | None = None,
    eps: float = 1e-8,
) -> dict[str, float | int]:
    if original_raw is None:
        original_raw = gdv_service.compute_graph_gdv(original.num_nodes, original.edge_index, cache_namespace="original")
    if stats is None:
        stats = fit_standardization(original_raw)
    pruned_raw = gdv_service.compute_graph_gdv(pruned.num_nodes, pruned.edge_index, cache_namespace="pruned")

    original_std = apply_standardization(original_raw, stats)
    pruned_std = apply_standardization(pruned_raw, stats)

    delta_sig = np.abs(original_std - pruned_std).sum(axis=1)
    denom = np.abs(original_std).sum(axis=1) + eps
    delta_rel = delta_sig / denom

    num_components, largest_component_ratio = connected_component_summary(pruned.num_nodes, pruned.edge_index)
    clustering_delta = clustering_coefficient(pruned.num_nodes, pruned.edge_index) - clustering_coefficient(
        original.num_nodes,
        original.edge_index,
    )

    return {
        "mean_delta_sig": float(np.mean(delta_sig)),
        "median_delta_sig": float(np.median(delta_sig)),
        "mean_delta_rel": float(np.mean(delta_rel)),
        "median_delta_rel": float(np.median(delta_rel)),
        "largest_component_ratio": float(largest_component_ratio),
        "num_components": int(num_components),
        "clustering_delta": float(clustering_delta),
    }


def compute_structural_controls(
    original: DatasetBundle,
    pruned: DatasetBundle,
) -> dict[str, float | int]:
    """Compute finite graph-level controls used by Phase-2 sensitivity models."""

    if original.num_nodes != pruned.num_nodes:
        raise ValueError("Structural-control graphs must have the same node count.")
    num_components, largest_component_ratio = connected_component_summary(
        pruned.num_nodes, pruned.edge_index
    )
    original_clustering = clustering_coefficient(original.num_nodes, original.edge_index)
    pruned_clustering = clustering_coefficient(pruned.num_nodes, pruned.edge_index)
    degree_tv, degree_js = degree_distribution_distances(
        original.num_nodes, original.edge_index, pruned.edge_index
    )
    original_homophily = edge_homophily(original.y, original.edge_index)
    pruned_homophily = edge_homophily(pruned.y, pruned.edge_index)
    original_isolated = isolated_node_count(original.num_nodes, original.edge_index)
    pruned_isolated = isolated_node_count(pruned.num_nodes, pruned.edge_index)
    denominator = float(max(original.num_nodes, 1))
    return {
        "largest_component_ratio": float(largest_component_ratio),
        "num_components": int(num_components),
        "clustering_delta": float(pruned_clustering - original_clustering),
        "degree_distribution_tv": float(degree_tv),
        "degree_distribution_js": float(degree_js),
        "homophily_delta": float(pruned_homophily - original_homophily),
        "original_homophily": float(original_homophily),
        "pruned_homophily": float(pruned_homophily),
        "isolated_node_count": int(pruned_isolated),
        "isolated_node_delta": int(pruned_isolated - original_isolated),
        "isolated_node_ratio_delta": float((pruned_isolated - original_isolated) / denominator),
    }
