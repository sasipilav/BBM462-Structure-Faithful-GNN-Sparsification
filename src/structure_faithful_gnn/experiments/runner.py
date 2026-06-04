from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import load_pruned_graph_artifact, write_run_artifacts
from ..config import DatasetConfig, ModelConfig, PruningConfig, resolved_config_dict
from ..data.loaders import load_dataset
from ..gdv.backends import GDVService
from ..metrics.structural import compute_structural_metrics
from ..models.trainer import TrainingResult, train_and_evaluate
from ..pruning.registry import prune_graph
from ..types import PrunedGraphArtifact, PruningResult, RunMetrics
from ..utils.graph import clustering_coefficient, connected_component_summary
from ..utils.io import ensure_dir


@dataclass
class ExperimentOutcome:
    run_dir: Path
    metrics: RunMetrics
    resolved_config: dict[str, Any]
    training: TrainingResult
    pruning_result: PruningResult | None = None


def run_dense_experiment(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    *,
    seed: int,
    output_root: str | Path,
    run_name: str | None = None,
) -> ExperimentOutcome:
    bundle = load_dataset(dataset_config)
    training = train_and_evaluate(bundle, model_config, seed)
    num_components, largest_component_ratio = connected_component_summary(bundle.num_nodes, bundle.edge_index)
    metrics = RunMetrics(
        accuracy=training.accuracy,
        macro_f1=training.macro_f1,
        train_sec=training.train_sec,
        infer_sec=training.infer_sec,
        edge_reduction=0.0,
        mean_delta_sig=0.0,
        median_delta_sig=0.0,
        mean_delta_rel=0.0,
        median_delta_rel=0.0,
        largest_component_ratio=largest_component_ratio,
        num_components=num_components,
        clustering_delta=0.0,
        metadata={"training": training.to_dict()},
    )
    resolved = resolved_config_dict(dataset_config, model_config, seed=seed)
    run_dir = _run_dir(output_root, bundle.name, model_config.name, "dense", seed, run_name)
    write_run_artifacts(
        run_dir,
        resolved,
        metrics,
        extra={"run_name": run_dir.name, "model_backend": training.backend},
    )
    return ExperimentOutcome(run_dir=run_dir, metrics=metrics, resolved_config=resolved, training=training)


def run_pruning_experiment(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    pruning_config: PruningConfig,
    *,
    seed: int,
    output_root: str | Path,
    run_name: str | None = None,
) -> ExperimentOutcome:
    bundle = load_dataset(dataset_config)
    gdv_service = GDVService(
        pruning_config.backend,
        cache_root="data/cached_gdv",
        orca_path=pruning_config.orca_path,
    )
    run_dir = _run_dir(
        output_root,
        bundle.name,
        model_config.name,
        pruning_config.method,
        seed,
        run_name,
        seed_suffix=_rho_seed_suffix(pruning_config),
    )
    pruning_result = prune_graph(bundle, pruning_config, seed=seed, gdv_service=gdv_service, artifact_dir=run_dir)
    pruned_bundle = bundle.clone_with_edges(
        pruning_result.pruned_edge_index,
        name=f"{bundle.name}_{pruning_config.method}",
        metadata_update={"pruning_method": pruning_config.method},
    )
    training = train_and_evaluate(pruned_bundle, model_config, seed)
    structural = compute_structural_metrics(bundle, pruned_bundle, gdv_service, eps=pruning_config.eps)
    metrics = RunMetrics(
        accuracy=training.accuracy,
        macro_f1=training.macro_f1,
        train_sec=training.train_sec,
        infer_sec=training.infer_sec,
        edge_reduction=1.0 - (pruning_result.after_edge_count / max(pruning_result.before_edge_count, 1)),
        mean_delta_sig=structural["mean_delta_sig"],
        median_delta_sig=structural["median_delta_sig"],
        mean_delta_rel=structural["mean_delta_rel"],
        median_delta_rel=structural["median_delta_rel"],
        largest_component_ratio=structural["largest_component_ratio"],
        num_components=structural["num_components"],
        clustering_delta=structural["clustering_delta"],
        metadata={
            "training": training.to_dict(),
            "pruning": pruning_result.metadata,
            "gdv_backend": gdv_service.backend_name,
        },
    )
    resolved = resolved_config_dict(dataset_config, model_config, pruning_config, seed=seed)
    write_run_artifacts(
        run_dir,
        resolved,
        metrics,
        pruning_result=pruning_result,
        extra={
            "run_name": run_dir.name,
            "model_backend": training.backend,
            "pruning_runtime_sec": pruning_result.runtime_sec,
            "gdv_backend": gdv_service.backend_name,
        },
    )
    return ExperimentOutcome(
        run_dir=run_dir,
        metrics=metrics,
        resolved_config=resolved,
        training=training,
        pruning_result=pruning_result,
    )


def run_training_on_pruned_artifact(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    *,
    artifact_dir: str | Path,
    seed: int,
    output_root: str | Path,
    run_name: str | None = None,
) -> ExperimentOutcome:
    bundle = load_dataset(dataset_config)
    artifact_resolved, pruned_artifact = load_pruned_graph_artifact(artifact_dir)
    pruned_bundle = bundle.clone_with_edges(
        pruned_artifact.pruned_edge_index,
        edge_weight=pruned_artifact.edge_weight,
        name=f"{bundle.name}_{pruned_artifact.method}",
        metadata_update={
            "pruning_method": pruned_artifact.method,
            "pruning_seed": pruned_artifact.pruning_seed,
            "pruned_graph_artifact_dir": str(Path(artifact_dir).resolve()),
        },
    )
    training = train_and_evaluate(pruned_bundle, model_config, seed)
    num_components, largest_component_ratio = connected_component_summary(bundle.num_nodes, pruned_artifact.pruned_edge_index)
    clustering_delta = clustering_coefficient(bundle.num_nodes, pruned_artifact.pruned_edge_index) - clustering_coefficient(
        bundle.num_nodes,
        bundle.edge_index,
    )
    metrics = RunMetrics(
        accuracy=training.accuracy,
        macro_f1=training.macro_f1,
        train_sec=training.train_sec,
        infer_sec=training.infer_sec,
        edge_reduction=pruned_artifact.achieved_edge_reduction,
        mean_delta_sig=None,
        median_delta_sig=None,
        mean_delta_rel=None,
        median_delta_rel=None,
        largest_component_ratio=largest_component_ratio,
        num_components=num_components,
        clustering_delta=clustering_delta,
        metadata={
            "training": training.to_dict(),
            "pruning_seed": pruned_artifact.pruning_seed,
            "pruned_graph_artifact_dir": str(Path(artifact_dir).resolve()),
        },
    )
    resolved = resolved_config_dict(dataset_config, model_config, seed=seed)
    if pruned_artifact.method == "relshift":
        resolved["pruning"] = dict(artifact_resolved.get("pruning", {}))
    else:
        resolved["baseline"] = dict(artifact_resolved.get("baseline", {}))
    resolved["pruned_graph_artifact_dir"] = str(Path(artifact_dir).resolve())
    run_dir = _run_dir(
        output_root,
        bundle.name,
        model_config.name,
        _method_dir_name(pruned_artifact, artifact_resolved),
        seed,
        run_name,
        seed_suffix=_artifact_seed_suffix(pruned_artifact, artifact_resolved),
    )
    write_run_artifacts(
        run_dir,
        resolved,
        metrics,
        pruning_result=pruned_artifact.to_pruning_result(),
        extra={
            "run_name": run_dir.name,
            "model_backend": training.backend,
            "pruning_runtime_sec": pruned_artifact.runtime_sec,
            "pruning_seed": pruned_artifact.pruning_seed,
            "pruned_graph_artifact_dir": str(Path(artifact_dir).resolve()),
        },
    )
    return ExperimentOutcome(
        run_dir=run_dir,
        metrics=metrics,
        resolved_config=resolved,
        training=training,
        pruning_result=pruned_artifact.to_pruning_result(),
    )


def _run_dir(
    output_root: str | Path,
    dataset_name: str,
    model_name: str,
    method: str,
    seed: int,
    run_name: str | None,
    seed_suffix: str | None = None,
) -> Path:
    seed_dir = f"seed_{seed}"
    if seed_suffix:
        seed_dir = f"{seed_dir}-{seed_suffix}"
    base = ensure_dir(Path(output_root) / dataset_name / model_name / method / seed_dir)
    if run_name:
        base = ensure_dir(base / run_name)
    return base


def _rho_seed_suffix(pruning_config: PruningConfig) -> str:
    return f"rho_{pruning_config.rho:g}"


def _artifact_seed_suffix(artifact: PrunedGraphArtifact, resolved_config: dict[str, Any]) -> str:
    if artifact.method == "relshift":
        pruning = resolved_config.get("pruning", {})
        return f"rho_{float(pruning['rho']):g}"
    baseline = resolved_config.get("baseline", {})
    if artifact.method == "dspar":
        return f"eps_{float(baseline['epsilon']):g}"
    if artifact.method == "lsp":
        parts = [
            f"k_{int(baseline['k'])}",
            f"s_{_format_sparsity_suffix(baseline['sparsity'])}",
        ]
        l_value = baseline.get("l", baseline.get("quantization_step"))
        if l_value is not None:
            parts.append(f"l_{float(l_value):g}")
        return "-".join(parts)
    return artifact.method


def _method_dir_name(artifact: PrunedGraphArtifact, resolved_config: dict[str, Any]) -> str:
    if artifact.method == "lsp":
        return str(resolved_config.get("baseline", {}).get("variant", "lsp"))
    return artifact.method


def _format_sparsity_suffix(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"
