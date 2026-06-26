from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from ..utils.graph import edge_pairs, stable_hash_edges
from ..utils.io import ensure_dir


class GDVBackend(Protocol):
    name: str

    def compute(self, num_nodes: int, edge_index: torch.Tensor) -> np.ndarray:
        ...


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


def fit_standardization(raw: np.ndarray) -> NormalizationStats:
    transformed = np.log1p(np.maximum(raw, 0.0))
    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return NormalizationStats(mean=mean, std=std)


def apply_standardization(raw: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    transformed = np.log1p(np.maximum(raw, 0.0))
    return (transformed - stats.mean) / stats.std


class OrcaBackend:
    name = "orca"

    def __init__(self, orca_path: str | None = None) -> None:
        self.orca_path = orca_path or os.environ.get("ORCA_BINARY") or "orca"

    def compute(self, num_nodes: int, edge_index: torch.Tensor) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "graph.txt"
            output_path = tmpdir_path / "orbits.txt"
            _write_orca_input(input_path, num_nodes, edge_index)
            cmd = [self.orca_path, "node", "4", str(input_path), str(output_path)]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"ORCA binary not found at '{self.orca_path}'. Set pruning.orca_path or ORCA_BINARY."
                ) from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"ORCA execution failed: {exc.stderr}") from exc
            return _read_orca_output(output_path, num_nodes)


class GDVService:
    def __init__(self, backend: str, cache_root: str | Path, *, orca_path: str | None = None) -> None:
        self.cache_root = ensure_dir(cache_root)
        self.last_compute_info: dict[str, object] = {}
        backend_name = backend.lower()
        if backend_name == "orca":
            self.backend: GDVBackend = OrcaBackend(orca_path=orca_path)
        else:
            raise ValueError(f"Unsupported GDV backend: {backend}")

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def compute_graph_gdv(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        *,
        cache_namespace: str,
    ) -> np.ndarray:
        edge_index = edge_index.long().contiguous()
        cache_path = self.cache_root / cache_namespace / f"{self.backend.name}_{stable_hash_edges(num_nodes, edge_index)}.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            result = np.load(cache_path)
            self.last_compute_info = {
                "cache_hit": True,
                "cache_path": str(cache_path),
                "backend": self.backend.name,
                "num_nodes": int(num_nodes),
                "num_edges": int(edge_index.shape[1]),
            }
            return result
        result = self.backend.compute(num_nodes, edge_index)
        np.save(cache_path, result)
        self.last_compute_info = {
            "cache_hit": False,
            "cache_path": str(cache_path),
            "backend": self.backend.name,
            "num_nodes": int(num_nodes),
            "num_edges": int(edge_index.shape[1]),
        }
        return result


def _write_orca_input(path: Path, num_nodes: int, edge_index: torch.Tensor) -> None:
    edges = edge_pairs(edge_index)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{num_nodes} {len(edges)}\n")
        for u, v in edges:
            handle.write(f"{u} {v}\n")


def _read_orca_output(path: Path, num_nodes: int) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append([float(value) for value in stripped.split()])
    if len(rows) != num_nodes:
        raise RuntimeError(f"Expected {num_nodes} ORCA output rows, found {len(rows)}.")
    return np.asarray(rows, dtype=np.float64)
