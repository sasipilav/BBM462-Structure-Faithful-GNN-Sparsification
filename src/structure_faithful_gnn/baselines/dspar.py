from __future__ import annotations

"""DSpar baseline aligned to the upstream zirui-ray-liu/DSpar_tmlr sparsification path.

Method freeze:
- Edge probabilities follow the official code exactly:
  `p_e ∝ 1 / deg(u) + 1 / deg(v)`.
- The sample budget follows the official formula exactly:
  `Q = int(0.16 * num_nodes * log(num_nodes) / epsilon^2)`.
- Graph sparsification uses the upstream fixed seed `42` regardless of any outer
  experiment or training seed. The upstream project explicitly hard-codes this.
- If the official C++ sampler is importable, use it. Otherwise fall back to a
  deterministic inverse-CDF sampler with the same fixed seed and probability model.

Important scope note:
- The upstream repo typically sparsifies PyG graphs and then calls
  `to_undirected(...)`. Our pipeline operates on a simple-undirected unique-edge
  view and duplicates edges only at model time. This module therefore returns one
  canonical undirected edge per pair, which is the correct adaptation to the
  pipeline-wide graph convention.
"""

import math

import numpy as np
import torch


UPSTREAM_DSPAR_SAMPLING_SEED = 42


def dspar_sparsify(
    num_nodes: int,
    edge_index: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    edge_index = edge_index.cpu().long()
    num_edges = int(edge_index.shape[1])
    if num_edges == 0:
        return edge_index.clone(), torch.empty((0,), dtype=torch.float32), {
            "epsilon": float(epsilon),
            "q_samples": 0,
            "unique_sampled_edges": 0,
            "sampling_seed": UPSTREAM_DSPAR_SAMPLING_SEED,
            "sampler_backend": "empty_graph",
        }

    q_samples = _upstream_q_samples(num_nodes=num_nodes, epsilon=float(epsilon))
    if q_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32), {
            "epsilon": float(epsilon),
            "q_samples": 0,
            "unique_sampled_edges": 0,
            "sampling_seed": UPSTREAM_DSPAR_SAMPLING_SEED,
            "sampler_backend": "zero_budget",
            "probability": "p_e proportional to 1/deg(u) + 1/deg(v)",
            "reweighted": True,
        }

    probs = _edge_probabilities(num_nodes=num_nodes, edge_index=edge_index)
    sampled, sampler_backend = _sample_edge_indices(probs=probs, q_samples=q_samples)
    unique_indices, counts = torch.unique(sampled, sorted=True, return_counts=True)
    weights = counts.double() / q_samples / probs[unique_indices]
    sampled_edges = edge_index[:, unique_indices].contiguous()
    return sampled_edges.long(), weights.float(), {
        "epsilon": float(epsilon),
        "q_samples": int(q_samples),
        "unique_sampled_edges": int(unique_indices.numel()),
        "sampling_seed": UPSTREAM_DSPAR_SAMPLING_SEED,
        "sampler_backend": sampler_backend,
        "probability": "p_e proportional to 1/deg(u) + 1/deg(v)",
        "reweighted": True,
    }


def _upstream_q_samples(*, num_nodes: int, epsilon: float) -> int:
    if num_nodes <= 1:
        return 0
    return int(0.16 * num_nodes * math.log(num_nodes) / (epsilon**2))


def _edge_probabilities(*, num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    num_edges = int(edge_index.shape[1])
    src, dst = edge_index
    degrees = torch.zeros(num_nodes, dtype=torch.float64)
    degrees.scatter_add_(0, src, torch.ones(num_edges, dtype=torch.float64))
    degrees.scatter_add_(0, dst, torch.ones(num_edges, dtype=torch.float64))
    probs = torch.nan_to_num(1.0 / degrees[src]) + torch.nan_to_num(1.0 / degrees[dst])
    probs = probs.double()
    return probs / probs.sum()


def _sample_edge_indices(*, probs: torch.Tensor, q_samples: int) -> tuple[torch.Tensor, str]:
    p_cumsum = torch.cumsum(probs.double(), dim=0).contiguous()
    try:
        import dspar.cpp_extension.sampler as upstream_sampler
    except ImportError:
        upstream_sampler = None

    if upstream_sampler is not None:
        sampled = upstream_sampler.edge_sample(
            p_cumsum,
            int(q_samples),
            int(UPSTREAM_DSPAR_SAMPLING_SEED),
        )
        return sampled.long().cpu(), "official_cpp_extension"

    uniforms = np.random.RandomState(UPSTREAM_DSPAR_SAMPLING_SEED).uniform(0.0, 1.0, size=q_samples)
    sampled_np = np.searchsorted(p_cumsum.cpu().numpy(), uniforms, side="left")
    return torch.from_numpy(sampled_np).long(), "python_inverse_cdf_fallback"
