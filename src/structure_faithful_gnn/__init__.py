"""Structure-faithful GNN pruning research package."""

from .config import DatasetConfig, ExperimentConfig, ModelConfig, PruningConfig
from .types import DatasetBundle, PrunedGraphArtifact, PruningResult, RunMetrics

__all__ = [
    "DatasetBundle",
    "DatasetConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PruningConfig",
    "PrunedGraphArtifact",
    "PruningResult",
    "RunMetrics",
]
