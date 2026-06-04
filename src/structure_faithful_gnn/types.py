from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DatasetBundle:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    edge_weight: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = self.x.float()
        self.y = self.y.long().view(-1)
        self.edge_index = self.edge_index.long()
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.float().view(-1)
        self.train_idx = self.train_idx.long().view(-1)
        self.val_idx = self.val_idx.long().view(-1)
        self.test_idx = self.test_idx.long().view(-1)

        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.x.ndim != 2:
            raise ValueError("x must have shape [num_nodes, num_features]")
        if self.y.shape[0] != self.x.shape[0]:
            raise ValueError("x and y must agree on num_nodes")
        if self.edge_weight is not None and self.edge_weight.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_weight must have one value per edge")

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        return int(torch.unique(self.y).numel())

    def model_edge_index(self) -> torch.Tensor:
        from .utils.graph import bidirectional_edge_index

        return bidirectional_edge_index(self.edge_index)

    def model_edge_weight(self) -> torch.Tensor | None:
        if self.edge_weight is None:
            return None
        return torch.cat([self.edge_weight, self.edge_weight], dim=0)

    def clone_with_edges(
        self,
        edge_index: torch.Tensor,
        *,
        edge_weight: torch.Tensor | None = None,
        name: str | None = None,
        metadata_update: dict[str, Any] | None = None,
    ) -> "DatasetBundle":
        metadata = dict(self.metadata)
        if metadata_update:
            metadata.update(metadata_update)
        return DatasetBundle(
            name=name or self.name,
            x=self.x.clone(),
            y=self.y.clone(),
            edge_index=edge_index.clone(),
            edge_weight=edge_weight.clone() if edge_weight is not None else None,
            train_idx=self.train_idx.clone(),
            val_idx=self.val_idx.clone(),
            test_idx=self.test_idx.clone(),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "has_edge_weight": self.edge_weight is not None,
            "metadata": self.metadata,
        }


@dataclass
class PruningResult:
    method: str
    pruned_edge_index: torch.Tensor
    removed_edge_index: torch.Tensor
    before_edge_count: int
    after_edge_count: int
    runtime_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "before_edge_count": self.before_edge_count,
            "after_edge_count": self.after_edge_count,
            "runtime_sec": self.runtime_sec,
            "metadata": self.metadata,
        }


@dataclass
class PrunedGraphArtifact:
    dataset: str
    method: str
    pruning_seed: int
    pruned_edge_index: torch.Tensor
    removed_edge_index: torch.Tensor
    before_edge_count: int
    after_edge_count: int
    runtime_sec: float
    edge_weight: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def achieved_edge_reduction(self) -> float:
        return 1.0 - (self.after_edge_count / max(self.before_edge_count, 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "method": self.method,
            "pruning_seed": self.pruning_seed,
            "before_edge_count": self.before_edge_count,
            "after_edge_count": self.after_edge_count,
            "runtime_sec": self.runtime_sec,
            "achieved_edge_reduction": self.achieved_edge_reduction,
            "has_edge_weight": self.edge_weight is not None,
            "metadata": self.metadata,
        }

    def to_pruning_result(self) -> PruningResult:
        return PruningResult(
            method=self.method,
            pruned_edge_index=self.pruned_edge_index,
            removed_edge_index=self.removed_edge_index,
            before_edge_count=self.before_edge_count,
            after_edge_count=self.after_edge_count,
            runtime_sec=self.runtime_sec,
            metadata=dict(self.metadata),
        )


@dataclass
class RunMetrics:
    accuracy: float
    macro_f1: float
    train_sec: float
    infer_sec: float
    edge_reduction: float
    mean_delta_sig: float | None
    median_delta_sig: float | None
    mean_delta_rel: float | None
    median_delta_rel: float | None
    largest_component_ratio: float
    num_components: int
    clustering_delta: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "train_sec": self.train_sec,
            "infer_sec": self.infer_sec,
            "edge_reduction": self.edge_reduction,
            "mean_delta_sig": self.mean_delta_sig,
            "median_delta_sig": self.median_delta_sig,
            "mean_delta_rel": self.mean_delta_rel,
            "median_delta_rel": self.median_delta_rel,
            "largest_component_ratio": self.largest_component_ratio,
            "num_components": self.num_components,
            "clustering_delta": self.clustering_delta,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload
