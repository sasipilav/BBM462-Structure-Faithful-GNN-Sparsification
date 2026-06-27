from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np

from ..pruning.incremental_relshift import ORBIT_DIM, classify_connected_orbits


@dataclass(frozen=True)
class ExactSparseGDVResult:
    raw: np.ndarray
    connected_two_node_count: int
    connected_three_node_count: int
    connected_four_node_count: int
    runtime_sec: float


def exact_sparse_graph_gdv(adjacency: list[set[int]]) -> ExactSparseGDVResult:
    """Count all connected induced 2--4 node graphlet orbits exactly.

    The routine enumerates only connected subsets:

    * size two: every active undirected edge once;
    * size three: every pair of neighbors around a center, deduplicated;
    * size four: every connected triple expanded by one adjacent node,
      deduplicated.

    Every connected induced three-node set has a node of induced degree at
    least two, so it appears in the neighbor-pair enumeration.  Every connected
    four-node graph has a connected induced three-node subset; expanding that
    subset by the remaining adjacent node therefore generates the four-set.
    Tuple sets remove the benign multiple generation of triangles, cycles and
    denser graphlets.
    """

    started = perf_counter()
    num_nodes = len(adjacency)
    raw = np.zeros((num_nodes, ORBIT_DIM), dtype=np.float64)

    edge_count = 0
    for u, neighbors in enumerate(adjacency):
        for v in neighbors:
            if u < v:
                raw[u, 0] += 1.0
                raw[v, 0] += 1.0
                edge_count += 1

    connected_triples: set[tuple[int, int, int]] = set()
    for center, neighbors in enumerate(adjacency):
        ordered_neighbors = sorted(neighbors)
        for left, right in combinations(ordered_neighbors, 2):
            connected_triples.add(tuple(sorted((center, left, right))))

    for triple in connected_triples:
        orbits = classify_connected_orbits(triple, adjacency)
        if orbits is None:  # Defensive: generation already guarantees connectivity.
            raise RuntimeError(f"Generated disconnected triple: {triple}")
        for local_idx, node in enumerate(triple):
            raw[node, orbits[local_idx]] += 1.0

    connected_quads: set[tuple[int, int, int, int]] = set()
    for triple in connected_triples:
        triple_nodes = set(triple)
        frontier: set[int] = set()
        for node in triple:
            frontier.update(adjacency[node])
        frontier.difference_update(triple_nodes)
        for fourth in frontier:
            connected_quads.add(tuple(sorted((*triple, fourth))))

    for quad in connected_quads:
        orbits = classify_connected_orbits(quad, adjacency)
        if orbits is None:  # Defensive: expansion already guarantees connectivity.
            raise RuntimeError(f"Generated disconnected quadruple: {quad}")
        for local_idx, node in enumerate(quad):
            raw[node, orbits[local_idx]] += 1.0

    return ExactSparseGDVResult(
        raw=raw,
        connected_two_node_count=edge_count,
        connected_three_node_count=len(connected_triples),
        connected_four_node_count=len(connected_quads),
        runtime_sec=perf_counter() - started,
    )
