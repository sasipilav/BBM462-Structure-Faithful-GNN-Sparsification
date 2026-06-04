from __future__ import annotations

from collections import Counter
from typing import Iterable

import networkx as nx
import numpy as np
import torch


def canonicalize_undirected_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int | None = None,
) -> torch.Tensor:
    edge_index = edge_index.long()
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    low = torch.minimum(edge_index[0], edge_index[1])
    high = torch.maximum(edge_index[0], edge_index[1])
    pairs = torch.stack([low, high], dim=1).cpu().numpy()
    pairs = np.unique(pairs, axis=0)
    if num_nodes is not None and pairs.size:
        if pairs.min() < 0 or pairs.max() >= num_nodes:
            raise ValueError("edge_index contains node ids outside [0, num_nodes)")
    if pairs.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(pairs.T).long()


def bidirectional_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.clone()
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    return torch.cat([edge_index, rev], dim=1)


def edge_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    return [(int(u), int(v)) for u, v in edge_index.t().tolist()]


def adjacency_sets(num_nodes: int, edge_index: torch.Tensor) -> list[set[int]]:
    adjacency = [set() for _ in range(num_nodes)]
    for u, v in edge_pairs(edge_index):
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def degree_array(num_nodes: int, edge_index: torch.Tensor) -> np.ndarray:
    degrees = np.zeros(num_nodes, dtype=np.int64)
    if edge_index.numel() == 0:
        return degrees
    flat = edge_index.flatten().cpu().numpy()
    counts = Counter(flat.tolist())
    for node, count in counts.items():
        degrees[node] = count
    return degrees


def build_networkx_graph(num_nodes: int, edge_index: torch.Tensor) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_pairs(edge_index))
    return graph


def remove_edge_pairs(
    edge_index: torch.Tensor,
    removed: Iterable[tuple[int, int]],
) -> torch.Tensor:
    removed_set = {tuple(sorted((int(u), int(v)))) for u, v in removed}
    kept = [pair for pair in edge_pairs(edge_index) if pair not in removed_set]
    if not kept:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(kept, dtype=torch.long).t().contiguous()


def support_scores(num_nodes: int, edge_index: torch.Tensor) -> dict[tuple[int, int], int]:
    adjacency = adjacency_sets(num_nodes, edge_index)
    scores: dict[tuple[int, int], int] = {}
    for u, v in edge_pairs(edge_index):
        scores[(u, v)] = len(adjacency[u].intersection(adjacency[v]))
    return scores


def bridge_edges(num_nodes: int, edge_index: torch.Tensor) -> set[tuple[int, int]]:
    graph = build_networkx_graph(num_nodes, edge_index)
    return {tuple(sorted(edge)) for edge in nx.bridges(graph)}


def bridge_edges_from_adjacency(adjacency: list[set[int]]) -> set[tuple[int, int]]:
    """Return undirected bridge edges with Tarjan low-link traversal."""

    import sys

    num_nodes = len(adjacency)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), num_nodes + 100))
    discovery = [-1] * num_nodes
    low = [0] * num_nodes
    parent = [-1] * num_nodes
    bridges: set[tuple[int, int]] = set()
    time = 0

    def visit(node: int) -> None:
        nonlocal time
        discovery[node] = time
        low[node] = time
        time += 1
        for neighbor in adjacency[node]:
            if discovery[neighbor] == -1:
                parent[neighbor] = node
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add((node, neighbor) if node < neighbor else (neighbor, node))
            elif neighbor != parent[node]:
                low[node] = min(low[node], discovery[neighbor])

    for node in range(num_nodes):
        if discovery[node] == -1:
            visit(node)
    return bridges


def connected_component_summary(num_nodes: int, edge_index: torch.Tensor) -> tuple[int, float]:
    graph = build_networkx_graph(num_nodes, edge_index)
    components = list(nx.connected_components(graph))
    if not components:
        return 0, 0.0
    largest = max(len(component) for component in components)
    return len(components), largest / max(1, num_nodes)


def isolated_node_count(num_nodes: int, edge_index: torch.Tensor) -> int:
    graph = build_networkx_graph(num_nodes, edge_index)
    return sum(1 for _, degree in graph.degree() if degree == 0)


def clustering_coefficient(num_nodes: int, edge_index: torch.Tensor) -> float:
    graph = build_networkx_graph(num_nodes, edge_index)
    return float(nx.average_clustering(graph))


def induced_subgraph_edges(nodes: list[int], adjacency: list[set[int]]) -> torch.Tensor:
    node_set = set(nodes)
    index_map = {node: idx for idx, node in enumerate(nodes)}
    local_edges: list[tuple[int, int]] = []
    for node in nodes:
        for neighbor in adjacency[node]:
            if neighbor in node_set and node < neighbor:
                local_edges.append((index_map[node], index_map[neighbor]))
    if not local_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(local_edges, dtype=torch.long).t().contiguous()


def k_hop_nodes(adjacency: list[set[int]], seeds: Iterable[int], hops: int) -> list[int]:
    visited = {int(seed) for seed in seeds}
    frontier = set(visited)
    for _ in range(max(hops, 0)):
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(adjacency[node])
        next_frontier -= visited
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier
    return sorted(visited)


def split_budget(total_remove: int, rounds: int) -> list[int]:
    rounds = max(1, rounds)
    base = total_remove // rounds
    remainder = total_remove % rounds
    return [base + (1 if idx < remainder else 0) for idx in range(rounds)]


def stable_hash_edges(num_nodes: int, edge_index: torch.Tensor) -> str:
    import hashlib

    payload = f"{num_nodes}|".encode("utf-8")
    if edge_index.numel():
        payload += edge_index.cpu().numpy().tobytes()
    return hashlib.sha1(payload).hexdigest()
