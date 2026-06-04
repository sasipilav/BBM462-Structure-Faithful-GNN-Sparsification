from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import torch

from .types import PrunedGraphArtifact, PruningResult, RunMetrics
from .utils.io import ensure_dir, write_json


def write_run_artifacts(
    run_dir: str | Path,
    resolved_config: Mapping[str, Any],
    metrics: RunMetrics,
    *,
    pruning_result: PruningResult | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    run_dir = ensure_dir(run_dir)

    write_json(run_dir / "resolved_config.json", dict(resolved_config))
    write_json(run_dir / "metrics.json", metrics.to_dict())

    summary_row = {
        "dataset": resolved_config["dataset"]["name"],
        "model": resolved_config["model"]["name"],
        "seed": resolved_config["seed"],
        "method": pruning_result.method if pruning_result else "dense",
        **metrics.to_dict(),
    }
    if extra:
        summary_row.update(dict(extra))
    _write_single_row_csv(run_dir / "summary_row.csv", summary_row)

    if pruning_result is not None:
        pruned_edges_path = run_dir / "pruned_edges.pt"
        removed_edges_path = run_dir / "removed_edges.pt"
        torch.save(pruning_result.pruned_edge_index, pruned_edges_path)
        torch.save(pruning_result.removed_edge_index, removed_edges_path)
        pruning_payload = pruning_result.to_dict()
        pruning_payload["pruned_edges_path"] = str(pruned_edges_path)
        pruning_payload["removed_edges_path"] = str(removed_edges_path)
        write_json(run_dir / "pruning_result.json", pruning_payload)

    if extra:
        write_json(run_dir / "extra.json", dict(extra))


def write_pruned_graph_artifact(
    run_dir: str | Path,
    resolved_config: Mapping[str, Any],
    artifact: PrunedGraphArtifact,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    run_dir = ensure_dir(run_dir)
    write_json(run_dir / "resolved_config.json", dict(resolved_config))

    pruned_edges_path = run_dir / "pruned_edges.pt"
    removed_edges_path = run_dir / "removed_edges.pt"
    torch.save(artifact.pruned_edge_index, pruned_edges_path)
    torch.save(artifact.removed_edge_index, removed_edges_path)

    payload = artifact.to_dict()
    payload["pruned_edges_path"] = str(pruned_edges_path)
    payload["removed_edges_path"] = str(removed_edges_path)
    if artifact.edge_weight is not None:
        edge_weight_path = run_dir / "edge_weight.pt"
        torch.save(artifact.edge_weight, edge_weight_path)
        payload["edge_weight_path"] = str(edge_weight_path)
    write_json(run_dir / "pruned_graph.json", payload)

    pruning_payload = artifact.to_pruning_result().to_dict()
    pruning_payload["pruned_edges_path"] = str(pruned_edges_path)
    pruning_payload["removed_edges_path"] = str(removed_edges_path)
    if artifact.edge_weight is not None:
        pruning_payload["edge_weight_path"] = str(run_dir / "edge_weight.pt")
    write_json(run_dir / "pruning_result.json", pruning_payload)

    summary_row = {
        "dataset": artifact.dataset,
        "method": artifact.method,
        "seed": artifact.pruning_seed,
        "achieved_edge_reduction": artifact.achieved_edge_reduction,
        "before_edge_count": artifact.before_edge_count,
        "after_edge_count": artifact.after_edge_count,
        "runtime_sec": artifact.runtime_sec,
    }
    if extra:
        summary_row.update(dict(extra))
    _write_single_row_csv(run_dir / "pruned_graph_row.csv", summary_row)
    if extra:
        write_json(run_dir / "extra.json", dict(extra))


def load_pruned_graph_artifact(run_dir: str | Path) -> tuple[dict[str, Any], PrunedGraphArtifact]:
    run_dir = Path(run_dir)
    resolved = _read_json(run_dir / "resolved_config.json")
    payload = _read_json(run_dir / "pruned_graph.json")
    pruned_edge_index = torch.load(run_dir / "pruned_edges.pt", map_location="cpu", weights_only=True)
    removed_edge_index = torch.load(run_dir / "removed_edges.pt", map_location="cpu", weights_only=True)
    edge_weight_path = run_dir / "edge_weight.pt"
    edge_weight = torch.load(edge_weight_path, map_location="cpu", weights_only=True) if edge_weight_path.exists() else None
    artifact = PrunedGraphArtifact(
        dataset=str(payload["dataset"]),
        method=str(payload["method"]),
        pruning_seed=int(payload["pruning_seed"]),
        pruned_edge_index=pruned_edge_index.long(),
        removed_edge_index=removed_edge_index.long(),
        before_edge_count=int(payload["before_edge_count"]),
        after_edge_count=int(payload["after_edge_count"]),
        runtime_sec=float(payload["runtime_sec"]),
        edge_weight=edge_weight.float() if edge_weight is not None else None,
        metadata=dict(payload.get("metadata", {})),
    )
    return resolved, artifact


def _write_single_row_csv(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
