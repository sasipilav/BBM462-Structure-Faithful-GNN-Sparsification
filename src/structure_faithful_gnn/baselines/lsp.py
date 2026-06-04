from __future__ import annotations

"""Locality Sensitive Pruning baseline aligned to the upstream dotd/GNN_experiments surface.

Method freeze:
- Public parameter surface follows the upstream README and code-path names:
  `minhash_lsh_projection`, `minhash_lsh_thresholding`, `num_minhash_funcs`, `sparsity`,
  and `quantization_step`.
- For node-feature-only datasets, edge attributes are constructed as ordered endpoint concatenation
  `[x_u, x_v]` when evaluating node `u` against a neighbor `v`.
- Upstream `LSP-T` does not expose the paper's `m`-bin parameter as a runnable hyperparameter.
  This port keeps `lsp_m=None` for compatibility and uses the upstream-exposed `sparsity` knob.
- Upstream `MinHashRep` and `MinHashRandomProj` suppress duplicate winners across hash functions via
  `used_indices`. This port preserves that behavior.

Hashing note:
- The upstream thresholded-signature path hashes discrete signatures with Python `hash()`.
- This port preserves that behavior so the hashing semantics match the upstream code path exactly.

Upstream reference:
- Repository: dotd/GNN_experiments
- Branch: main
- Commit SHA: aa9cf7a38d23cee005a3e5b4df9a60fee1f46ea3
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..utils.graph import bidirectional_edge_index, canonicalize_undirected_edge_index, edge_pairs


UPSTREAM_REPO = "dotd/GNN_experiments"
UPSTREAM_REF = "main"
UPSTREAM_COMMIT_SHA = "aa9cf7a38d23cee005a3e5b4df9a60fee1f46ea3"
DEFAULT_THRESHOLD_STD = 1.0
DEFAULT_THRESHOLD_MEAN = 1.0


def lsp_edge_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must have shape [num_nodes, num_features].")
    edge_index = edge_index.long()
    if edge_index.numel() == 0:
        return torch.empty((0, int(x.shape[1] * 2)), dtype=x.dtype)
    src = x[edge_index[0]]
    dst = x[edge_index[1]]
    return torch.cat([src, dst], dim=1)


@dataclass
class LSPResult:
    pruned_edge_index: torch.Tensor
    removed_edge_index: torch.Tensor
    before_edge_count: int
    after_edge_count: int
    runtime_sec: float
    metadata: dict[str, object]


class _MinHashRep:
    def __init__(
        self,
        num_funcs: int,
        random: np.random.RandomState,
        *,
        max_val: int = (2**32) - 1,
        perms: list[tuple[int, int]] | None = None,
        prime: int = 2_147_483_647,
    ) -> None:
        self.num_funcs = int(num_funcs)
        self.max_val = int(max_val)
        self.prime = int(prime)
        upper = min(self.max_val, self.prime)
        if perms is None:
            self.perms = [
                (int(random.randint(0, upper)), int(random.randint(0, upper)))
                for _ in range(self.num_funcs)
            ]
        else:
            self.perms = [(int(a), int(b)) for a, b in perms]

    def hash_values(self, signatures: list[str | int]) -> np.ndarray:
        base_values = np.array([_upstream_token_hash(signature, self.prime) for signature in signatures], dtype=np.int64)
        if base_values.size == 0:
            return np.empty((0, self.num_funcs), dtype=np.int64)
        columns = []
        for a, b in self.perms:
            columns.append(((a * base_values + b) % self.prime).astype(np.int64))
        return np.stack(columns, axis=1)

    def apply(self, signatures: list[str | int], metas: list[tuple[int, int]]) -> list[tuple[int, int]]:
        values = self.hash_values(signatures)
        selected: list[tuple[int, int]] = []
        used_indices = [0] * len(signatures)
        for func_idx in range(values.shape[1]):
            minhashes = [
                (int(values[item_idx, func_idx]), item_idx, metas[item_idx])
                for item_idx in range(values.shape[0])
            ]
            minhashes.sort()
            for _, item_idx, meta in minhashes:
                if used_indices[item_idx]:
                    continue
                used_indices[item_idx] = 1
                selected.append(meta)
                break
        return selected


class _MinHashRandomProj:
    def __init__(
        self,
        num_funcs: int,
        random: np.random.RandomState,
        *,
        sparsity: int,
        input_dim: int,
        quantization_step: float = 1.0,
        planes: np.ndarray | None = None,
        biases: np.ndarray | None = None,
        indices_for_planes: np.ndarray | None = None,
    ) -> None:
        self.num_funcs = int(num_funcs)
        self.sparsity = int(min(sparsity, input_dim))
        self.quantization_step = float(quantization_step)
        if self.quantization_step <= 0:
            raise ValueError("quantization_step must be positive.")
        self.planes = random.randn(self.num_funcs, self.sparsity) if planes is None else np.asarray(planes, dtype=np.float64)
        self.biases = random.randn(self.num_funcs) if biases is None else np.asarray(biases, dtype=np.float64)
        self.indices_for_planes = (
            random.randint(low=0, high=input_dim, size=(self.num_funcs, self.sparsity))
            if indices_for_planes is None
            else np.asarray(indices_for_planes, dtype=np.int64)
        )

    def hash_values(self, reps: np.ndarray) -> np.ndarray:
        if reps.size == 0:
            return np.empty((0, self.num_funcs), dtype=np.int64)
        columns = []
        for plane, bias, indices in zip(self.planes, self.biases, self.indices_for_planes):
            projected = reps[:, indices] @ plane
            columns.append(np.floor((projected + bias) / self.quantization_step).astype(np.int64))
        return np.stack(columns, axis=1)

    def apply(self, reps: np.ndarray, metas: list[tuple[int, int]]) -> list[tuple[int, int]]:
        values = self.hash_values(reps)
        selected: list[tuple[int, int]] = []
        used_indices = [0] * len(reps)
        for func_idx in range(values.shape[1]):
            minhashes = [
                (int(values[item_idx, func_idx]), item_idx, metas[item_idx])
                for item_idx in range(values.shape[0])
            ]
            minhashes.sort()
            for _, item_idx, meta in minhashes:
                if used_indices[item_idx]:
                    continue
                used_indices[item_idx] = 1
                selected.append(meta)
                break
        return selected


class _ThresholdSignatureLSH:
    def __init__(
        self,
        *,
        input_dim: int,
        num_functions: int,
        sparsity: int,
        std_of_threshold: float,
        random: np.random.RandomState,
        mean_of_threshold: float,
        indices: list[np.ndarray] | np.ndarray | None = None,
        thresholds: list[np.ndarray] | np.ndarray | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.num_functions = int(num_functions)
        self.sparsity = int(min(sparsity, input_dim))
        self.std_of_threshold = float(std_of_threshold)
        self.mean_of_threshold = float(mean_of_threshold)
        self.random = random
        if indices is None and thresholds is None:
            self.indices = []
            self.thresholds = []
            for _ in range(self.num_functions):
                self.indices.append(random.permutation(self.input_dim)[: self.sparsity])
                self.thresholds.append(
                    random.normal(0.0, self.std_of_threshold, size=self.sparsity) + self.mean_of_threshold
                )
        else:
            if indices is None or thresholds is None:
                raise ValueError("indices and thresholds must be provided together when overriding LSP-T randomness.")
            self.indices = [np.asarray(arr, dtype=np.int64) for arr in indices]
            self.thresholds = [np.asarray(arr, dtype=np.float64) for arr in thresholds]

    def signature_matrix(self, reps: np.ndarray) -> np.ndarray:
        if reps.size == 0:
            return np.empty((0, self.num_functions * self.sparsity), dtype=np.uint8)
        signatures = np.array([reps[:, self.indices[i]] <= self.thresholds[i] for i in range(self.num_functions)])
        return signatures.transpose((1, 0, 2)).reshape(signatures.shape[1], -1).astype(np.uint8)

    def signature_strings(self, reps: np.ndarray) -> list[str]:
        matrix = self.signature_matrix(reps)
        return ["".join(map(str, row.tolist())) for row in matrix]


def lsp_prune(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    variant: str,
    k: int,
    sparsity: int,
    quantization_step: float | None = None,
    seed: int = 0,
) -> LSPResult:
    start = time.perf_counter()
    if k <= 0:
        raise ValueError("k must be positive.")
    if sparsity <= 0:
        raise ValueError("sparsity must be positive.")
    if x.ndim != 2:
        raise ValueError("x must have shape [num_nodes, num_features].")

    num_nodes = int(x.shape[0])
    original_edge_index = canonicalize_undirected_edge_index(edge_index, num_nodes=num_nodes)
    if original_edge_index.numel() == 0:
        return LSPResult(
            pruned_edge_index=original_edge_index.clone(),
            removed_edge_index=torch.empty((2, 0), dtype=torch.long),
            before_edge_count=0,
            after_edge_count=0,
            runtime_sec=float(time.perf_counter() - start),
            metadata={
                "method": "lsp",
                "variant": variant,
                "k": int(k),
                "sparsity": int(sparsity),
                "lsp_m": None,
                "lsp_l": float(quantization_step) if quantization_step is not None else None,
                "seed": int(seed),
                "upstream_repo": UPSTREAM_REPO,
                "upstream_ref": UPSTREAM_REF,
                "upstream_commit_sha": UPSTREAM_COMMIT_SHA,
                "upstream_pruning_method": _upstream_method_name(variant),
                "selected_directed_edges_before_union": 0,
            },
        )

    adjacency = _adjacency_lists(num_nodes, original_edge_index)
    x_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
    rng = np.random.RandomState(seed)

    if variant == "lsp_p":
        projector = _MinHashRandomProj(
            int(k),
            rng,
            sparsity=int(sparsity),
            input_dim=int(x_np.shape[1] * 2),
            quantization_step=float(1.0 if quantization_step is None else quantization_step),
        )
        selected = _select_projection_edges(adjacency, x_np, projector)
        lsp_l = float(projector.quantization_step)
        lsp_m = None
    elif variant == "lsp_t":
        # Upstream constructs MinHashRep first and LSH second from the same RandomState.
        # Keep that order so a fixed seed reproduces the same pruning choices.
        minhash = _MinHashRep(int(k), rng, prime=2_147_483_647)
        node_lsh = _ThresholdSignatureLSH(
            input_dim=int(x_np.shape[1]),
            num_functions=int(k),
            sparsity=int(sparsity),
            std_of_threshold=DEFAULT_THRESHOLD_STD,
            random=rng,
            mean_of_threshold=DEFAULT_THRESHOLD_MEAN,
        )
        node_signatures = node_lsh.signature_matrix(x_np)
        selected = _select_threshold_edges(adjacency, node_signatures, minhash)
        lsp_l = None
        lsp_m = None
    else:
        raise ValueError(f"Unsupported LSP variant: {variant}")

    pruned_edge_index = _pairs_to_edge_index(selected, num_nodes=num_nodes)
    removed_edge_index = _removed_edges(original_edge_index, pruned_edge_index)
    runtime_sec = float(time.perf_counter() - start)
    return LSPResult(
        pruned_edge_index=pruned_edge_index,
        removed_edge_index=removed_edge_index,
        before_edge_count=int(original_edge_index.shape[1]),
        after_edge_count=int(pruned_edge_index.shape[1]),
        runtime_sec=runtime_sec,
        metadata={
            "method": "lsp",
            "variant": variant,
            "k": int(k),
            "sparsity": int(sparsity),
            "lsp_m": lsp_m,
            "lsp_l": lsp_l,
            "seed": int(seed),
            "upstream_repo": UPSTREAM_REPO,
            "upstream_ref": UPSTREAM_REF,
            "upstream_commit_sha": UPSTREAM_COMMIT_SHA,
            "upstream_pruning_method": _upstream_method_name(variant),
            "selected_directed_edges_before_union": len(selected),
        },
    )


def _select_projection_edges(
    adjacency: list[list[int]],
    x_np: np.ndarray,
    projector: _MinHashRandomProj,
) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for node, neighbors in enumerate(adjacency):
        if not neighbors:
            continue
        reps, metas = _edge_reps_for_node(node, neighbors, x_np)
        selected.extend(projector.apply(reps, metas))
    return selected


def _select_threshold_edges(
    adjacency: list[list[int]],
    node_signatures: np.ndarray,
    minhash: _MinHashRep,
) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for node, neighbors in enumerate(adjacency):
        if not neighbors:
            continue
        source = np.repeat(node_signatures[node][np.newaxis, :], repeats=len(neighbors), axis=0)
        target = np.stack([node_signatures[neighbor] for neighbor in neighbors], axis=0)
        rep_tensor = np.hstack([source, target])
        signatures = ["".join(map(str, row.tolist())) for row in rep_tensor.astype(int)]
        metas = [(node, int(neighbor)) for neighbor in neighbors]
        selected.extend(minhash.apply(signatures, metas))
    return selected


def _edge_reps_for_node(node: int, neighbors: list[int], x_np: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    source = np.repeat(x_np[node][np.newaxis, :], repeats=len(neighbors), axis=0)
    target = np.stack([x_np[neighbor] for neighbor in neighbors], axis=0)
    reps = np.hstack([source, target])
    metas = [(node, int(neighbor)) for neighbor in neighbors]
    return reps, metas


def _adjacency_lists(num_nodes: int, edge_index: torch.Tensor) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    # Upstream helper code traverses a directed edge_index and appends outgoing
    # neighbors in column order. To match that behavior on our canonical
    # undirected graph view, expand once to a bidirectional edge_index and keep
    # the same column ordering.
    for u, v in edge_pairs(bidirectional_edge_index(edge_index)):
        adjacency[u].append(v)
    return adjacency


def _pairs_to_edge_index(edges: list[tuple[int, int]], *, num_nodes: int) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return canonicalize_undirected_edge_index(edge_index, num_nodes=num_nodes)


def _removed_edges(original_edge_index: torch.Tensor, pruned_edge_index: torch.Tensor) -> torch.Tensor:
    pruned = {tuple(sorted(pair)) for pair in edge_pairs(pruned_edge_index)}
    removed = [tuple(pair) for pair in edge_pairs(original_edge_index) if tuple(sorted(pair)) not in pruned]
    if not removed:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(removed, dtype=torch.long).t().contiguous()


def _upstream_token_hash(value: str | int, prime: int) -> int:
    if isinstance(value, int):
        return value
    return hash(value) % prime


def _upstream_method_name(variant: str) -> str:
    if variant == "lsp_p":
        return "minhash_lsh_projection"
    if variant == "lsp_t":
        return "minhash_lsh_thresholding"
    raise ValueError(f"Unsupported LSP variant: {variant}")
