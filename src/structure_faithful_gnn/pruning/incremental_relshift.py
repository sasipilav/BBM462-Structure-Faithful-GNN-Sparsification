from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..gdv.orbits import ORBIT_DIM, OrbitId
from ..utils.graph import k_hop_nodes


@dataclass
class IncrementalEdgeDelta:
    edge: tuple[int, int]
    affected_nodes: tuple[int, ...]
    raw_delta: np.ndarray
    impacted_edges: tuple[tuple[int, int], ...] = tuple()


def compute_edge_raw_delta(
    current_adjacency: list[set[int]],
    edge: tuple[int, int],
) -> IncrementalEdgeDelta:
    u, v = _canonical_edge(edge)
    affected_nodes = tuple(k_hop_nodes(current_adjacency, (u, v), hops=2))
    affected_index = {node: idx for idx, node in enumerate(affected_nodes)}
    raw_delta = np.zeros((len(affected_nodes), ORBIT_DIM), dtype=np.float64)
    for subset in iter_relevant_subsets(current_adjacency, edge):
        _accumulate_subset_delta(
            nodes=subset,
            deleted_edge=(u, v),
            adjacency=current_adjacency,
            affected_index=affected_index,
            raw_delta=raw_delta,
        )

    return IncrementalEdgeDelta(edge=(u, v), affected_nodes=affected_nodes, raw_delta=raw_delta)


def compute_edge_endpoint_delta(
    current_adjacency: list[set[int]],
    edge: tuple[int, int],
) -> np.ndarray:
    u, v = _canonical_edge(edge)
    endpoint_index = {u: 0, v: 1}
    endpoint_delta = np.zeros((2, ORBIT_DIM), dtype=np.float64)
    for subset in iter_relevant_subsets(current_adjacency, edge):
        _accumulate_subset_delta(
            nodes=subset,
            deleted_edge=(u, v),
            adjacency=current_adjacency,
            affected_index=endpoint_index,
            raw_delta=endpoint_delta,
            tracked_nodes={u, v},
        )
    return endpoint_delta


def iter_relevant_subsets(
    current_adjacency: list[set[int]],
    edge: tuple[int, int],
) -> list[tuple[int, ...]]:
    u, v = _canonical_edge(edge)
    directly_attached = sorted((current_adjacency[u] | current_adjacency[v]) - {u, v})
    subsets: list[tuple[int, ...]] = [(u, v)]
    subsets.extend((u, v, int(node)) for node in directly_attached)
    subsets.extend((u, v, int(a), int(b)) for a, b in _relevant_four_node_pairs(current_adjacency, u=u, v=v, directly_attached=directly_attached))
    return subsets


def _relevant_four_node_pairs(
    current_adjacency: list[set[int]],
    *,
    u: int,
    v: int,
    directly_attached: list[int],
) -> list[tuple[int, int]]:
    pair_candidates: set[tuple[int, int]] = set(combinations(directly_attached, 2))
    for a in directly_attached:
        for b in current_adjacency[a]:
            if b in {u, v, a}:
                continue
            pair_candidates.add((a, b) if a < b else (b, a))
    return sorted(pair_candidates)


def brute_force_graph_gdv(num_nodes: int, adjacency: list[set[int]]) -> np.ndarray:
    raw = np.zeros((num_nodes, ORBIT_DIM), dtype=np.float64)
    all_nodes = list(range(num_nodes))
    for size in (2, 3, 4):
        for subset in combinations(all_nodes, size):
            orbits = classify_connected_orbits(subset, adjacency)
            if orbits is None:
                continue
            for idx, node in enumerate(subset):
                raw[node, orbits[idx]] += 1.0
    return raw


def classify_connected_orbits(
    nodes: tuple[int, ...] | list[int],
    adjacency: list[set[int]],
    *,
    removed_edge: tuple[int, int] | None = None,
) -> tuple[int, ...] | None:
    ordered_nodes = tuple(int(node) for node in nodes)
    size = len(ordered_nodes)
    if size < 2 or size > 4:
        raise ValueError(f"Expected 2-4 nodes, found {size}.")

    removed = _canonical_edge(removed_edge) if removed_edge is not None else None
    degrees = [0] * size
    local_adjacency = [set() for _ in range(size)]
    edge_count = 0
    for left_idx in range(size):
        left = ordered_nodes[left_idx]
        for right_idx in range(left_idx + 1, size):
            right = ordered_nodes[right_idx]
            if removed is not None and _canonical_edge((left, right)) == removed:
                continue
            if right in adjacency[left]:
                edge_count += 1
                degrees[left_idx] += 1
                degrees[right_idx] += 1
                local_adjacency[left_idx].add(right_idx)
                local_adjacency[right_idx].add(left_idx)

    if edge_count == 0 or not _is_connected(local_adjacency):
        return None
    return _orbit_indices_from_signature(size=size, edge_count=edge_count, degrees=degrees)


def _accumulate_subset_delta(
    *,
    nodes: tuple[int, ...],
    deleted_edge: tuple[int, int],
    adjacency: list[set[int]],
    affected_index: dict[int, int],
    raw_delta: np.ndarray,
    tracked_nodes: set[int] | None = None,
) -> None:
    before_orbits = classify_connected_orbits(nodes, adjacency)
    if before_orbits is None:
        return
    after_orbits = classify_connected_orbits(nodes, adjacency, removed_edge=deleted_edge)
    for idx, node in enumerate(nodes):
        if tracked_nodes is not None and node not in tracked_nodes:
            continue
        node_pos = affected_index[node]
        raw_delta[node_pos, before_orbits[idx]] -= 1.0
        if after_orbits is not None:
            raw_delta[node_pos, after_orbits[idx]] += 1.0


def _orbit_indices_from_signature(*, size: int, edge_count: int, degrees: list[int]) -> tuple[int, ...]:
    ordered_degrees = sorted(degrees)
    if size == 2:
        if edge_count != 1:
            raise ValueError(f"Invalid connected 2-node graphlet with {edge_count} edges.")
        return (int(OrbitId.EDGE_ENDPOINT), int(OrbitId.EDGE_ENDPOINT))

    if size == 3:
        if edge_count == 2 and ordered_degrees == [1, 1, 2]:
            return tuple(
                int(OrbitId.PATH3_CENTER) if degree == 2 else int(OrbitId.PATH3_ENDPOINT)
                for degree in degrees
            )
        if edge_count == 3 and ordered_degrees == [2, 2, 2]:
            return (int(OrbitId.TRIANGLE_NODE),) * 3
        raise ValueError(f"Invalid connected 3-node graphlet signature: edges={edge_count}, degrees={ordered_degrees}")

    if size != 4:
        raise ValueError(f"Unsupported graphlet size: {size}")

    if edge_count == 3:
        if ordered_degrees == [1, 1, 1, 3]:
            return tuple(
                int(OrbitId.STAR4_CENTER) if degree == 3 else int(OrbitId.STAR4_LEAF)
                for degree in degrees
            )
        if ordered_degrees == [1, 1, 2, 2]:
            return tuple(
                int(OrbitId.PATH4_INTERNAL) if degree == 2 else int(OrbitId.PATH4_ENDPOINT)
                for degree in degrees
            )
    elif edge_count == 4:
        if ordered_degrees == [2, 2, 2, 2]:
            return (int(OrbitId.CYCLE4_NODE),) * 4
        if ordered_degrees == [1, 2, 2, 3]:
            # ORCA paw/tailed-triangle order: tail leaf=9, triangle degree-2=10, attachment degree-3=11.
            return tuple(
                int(OrbitId.TAILED_TRIANGLE_TAIL)
                if degree == 1
                else int(OrbitId.TAILED_TRIANGLE_ATTACHMENT)
                if degree == 3
                else int(OrbitId.TAILED_TRIANGLE_TRIANGLE_NODE)
                for degree in degrees
            )
    elif edge_count == 5 and ordered_degrees == [2, 2, 3, 3]:
        return tuple(
            int(OrbitId.DIAMOND_DEGREE2) if degree == 2 else int(OrbitId.DIAMOND_DEGREE3)
            for degree in degrees
        )
    elif edge_count == 6 and ordered_degrees == [3, 3, 3, 3]:
        return (int(OrbitId.CLIQUE4_NODE),) * 4

    raise ValueError(f"Invalid connected 4-node graphlet signature: edges={edge_count}, degrees={ordered_degrees}")


def _is_connected(adjacency: list[set[int]]) -> bool:
    visited = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited) == len(adjacency)


def _canonical_edge(edge: tuple[int, int] | None) -> tuple[int, int]:
    if edge is None:
        raise ValueError("edge cannot be None.")
    left, right = int(edge[0]), int(edge[1])
    return (left, right) if left <= right else (right, left)
