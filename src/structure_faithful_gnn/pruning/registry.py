from __future__ import annotations

from pathlib import Path

from ..config import PruningConfig
from ..gdv.backends import GDVService
from ..types import DatasetBundle, PruningResult
from .relshift import relshift_prune


def prune_graph(
    bundle: DatasetBundle,
    config: PruningConfig,
    *,
    seed: int,
    gdv_service: GDVService | None = None,
    artifact_dir: str | Path | None = None,
) -> PruningResult:
    method = config.method.lower()
    if method == "relshift":
        if gdv_service is None:
            raise ValueError("relshift pruning requires a GDVService")
        return relshift_prune(bundle, config, gdv_service, seed=seed, artifact_dir=artifact_dir)
    raise ValueError(
        f"Unsupported pruning method: {config.method}. "
        "This repo is currently focused on the proposal method only: relshift."
    )
