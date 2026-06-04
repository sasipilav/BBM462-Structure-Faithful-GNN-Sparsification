from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _split_fields(cls: type[T], payload: dict[str, Any]) -> dict[str, Any]:
    known = {field.name for field in fields(cls)}
    result = {key: value for key, value in payload.items() if key in known}
    unknown = {key: value for key, value in payload.items() if key not in known}
    if "options" in known:
        result.setdefault("options", {})
        result["options"].update(unknown)
    return result


@dataclass
class DatasetConfig:
    name: str
    root: str = "data/raw"
    make_undirected: bool = True
    preserve_splits: bool = True
    options: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetConfig":
        return cls(**_split_fields(cls, payload))


@dataclass
class ModelConfig:
    name: str
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 300
    early_stop: int = 50
    options: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(**_split_fields(cls, payload))


@dataclass
class PruningConfig:
    method: str
    rho: float = 0.3
    score_mode: str = "relative"
    d_min: int = 2
    guard_bridges: bool = False
    recompute_rounds: int = 1
    backend: str = "orca"
    orca_path: str | None = None
    eps: float = 1e-8
    options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.recompute_rounds = max(1, int(math.ceil(float(self.recompute_rounds))))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PruningConfig":
        return cls(**_split_fields(cls, payload))


@dataclass
class ExperimentConfig:
    datasets: list[str]
    models: list[str]
    prunings: list[str]
    seeds: list[int]
    output_root: str = "results/default"
    include_dense: bool = True
    tag: str = "default"
    options: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(**_split_fields(cls, payload))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML at {path} must contain a mapping at the top level.")
    return payload


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return DatasetConfig.from_dict(load_yaml(path))


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.from_dict(load_yaml(path))


def load_pruning_config(path: str | Path) -> PruningConfig:
    return PruningConfig.from_dict(load_yaml(path))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(load_yaml(path))


def resolved_config_dict(
    dataset: DatasetConfig,
    model: ModelConfig,
    pruning: PruningConfig | None = None,
    *,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "dataset": asdict(dataset),
        "model": asdict(model),
        "seed": seed,
    }
    if pruning is not None:
        payload["pruning"] = asdict(pruning)
    return payload
