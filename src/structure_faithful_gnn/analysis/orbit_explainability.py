from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..gdv.orbits import ORBIT_DIM, ORBIT_REGISTRY_VERSION, orbit_registry_payload
from ..utils.io import ensure_dir, write_json


ORBIT_EVENT_SCHEMA_VERSION = "relshift-orbit-edge-events-v1"
ORBIT_TRANSITION_SCHEMA_VERSION = "relshift-orbit-transitions-v1"
ORBIT_EXPLAINABILITY_MANIFEST_VERSION = "relshift-orbit-explainability-manifest-v1"
DESTROYED_ORBIT_COLUMN = ORBIT_DIM
ORBIT_TRANSITION_COLUMNS = ORBIT_DIM + 1


@dataclass(frozen=True, slots=True)
class OrbitEdgeEvent:
    """Exact orbit-level structural event produced by one selected-edge deletion.

    ``transition`` stores node-orbit incidence transitions. Rows are source
    orbits 0..14. Columns 0..14 are destination orbits; column 15 is the
    destroyed/disconnected sink. ``loss`` and ``gain`` exclude diagonal
    self-transitions. ``net`` is gain minus loss. ``absolute_delta`` is the
    node-wise L1 change aggregated independently for each orbit.
    """

    round_index: int
    selected_edge_id: int
    u: int
    v: int
    relshift_score: float
    degree_score: float
    support_score: float
    active_edges_before: int
    active_edges_after: int
    changed_node_count: int
    directly_attached_size: int
    four_node_pair_count: int
    transition: np.ndarray
    loss: np.ndarray
    gain: np.ndarray
    net: np.ndarray
    absolute_delta: np.ndarray
    destroyed_incidence_count: int
    unchanged_incidence_count: int
    transition_incidence_count: int

    @classmethod
    def from_native_result(
        cls,
        *,
        round_index: int,
        selected_edge_id: int,
        u: int,
        v: int,
        relshift_score: float,
        degree_score: float,
        support_score: float,
        active_edges_before: int,
        active_edges_after: int,
        native_result: Mapping[str, Any],
    ) -> "OrbitEdgeEvent":
        transition = np.asarray(native_result["orbit_transition_matrix"], dtype=np.int64)
        loss = np.asarray(native_result["orbit_loss"], dtype=np.int64)
        gain = np.asarray(native_result["orbit_gain"], dtype=np.int64)
        net = np.asarray(native_result["orbit_net"], dtype=np.int64)
        absolute_delta = np.asarray(native_result["orbit_absolute_delta"], dtype=np.int64)
        event = cls(
            round_index=int(round_index),
            selected_edge_id=int(selected_edge_id),
            u=int(u),
            v=int(v),
            relshift_score=float(relshift_score),
            degree_score=float(degree_score),
            support_score=float(support_score),
            active_edges_before=int(active_edges_before),
            active_edges_after=int(active_edges_after),
            changed_node_count=int(native_result["changed_node_count"]),
            directly_attached_size=int(native_result.get("directly_attached_size", 0)),
            four_node_pair_count=int(native_result.get("four_node_pair_count", 0)),
            transition=transition.copy(),
            loss=loss.copy(),
            gain=gain.copy(),
            net=net.copy(),
            absolute_delta=absolute_delta.copy(),
            destroyed_incidence_count=int(native_result["destroyed_incidence_count"]),
            unchanged_incidence_count=int(native_result["unchanged_incidence_count"]),
            transition_incidence_count=int(native_result["transition_incidence_count"]),
        )
        event.validate()
        return event

    def validate(self) -> None:
        if self.round_index <= 0:
            raise ValueError("Orbit edge-event round_index must be positive.")
        if self.selected_edge_id < 0:
            raise ValueError("Orbit edge-event selected_edge_id must be non-negative.")
        if self.u < 0 or self.v < 0 or self.u == self.v:
            raise ValueError("Orbit edge-event endpoints must be distinct non-negative nodes.")
        if self.active_edges_before - self.active_edges_after != 1:
            raise ValueError("A selected-edge event must reduce the active edge count by exactly one.")
        if self.changed_node_count <= 0:
            raise ValueError("changed_node_count must be positive for a selected-edge deletion.")
        for name, value in (
            ("relshift_score", self.relshift_score),
            ("degree_score", self.degree_score),
            ("support_score", self.support_score),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        _require_shape("transition", self.transition, (ORBIT_DIM, ORBIT_TRANSITION_COLUMNS))
        for name, vector in (
            ("loss", self.loss),
            ("gain", self.gain),
            ("net", self.net),
            ("absolute_delta", self.absolute_delta),
        ):
            _require_shape(name, vector, (ORBIT_DIM,))
        if np.any(self.transition < 0):
            raise ValueError("Orbit transition counts must be non-negative.")
        if np.any(self.loss < 0) or np.any(self.gain < 0) or np.any(self.absolute_delta < 0):
            raise ValueError("Orbit loss/gain/absolute-delta vectors must be non-negative.")

        expected_loss, expected_gain, expected_net, destroyed, unchanged = summarize_transition_matrix(
            self.transition
        )
        if not np.array_equal(self.loss, expected_loss):
            raise ValueError("Orbit loss vector is inconsistent with the transition matrix.")
        if not np.array_equal(self.gain, expected_gain):
            raise ValueError("Orbit gain vector is inconsistent with the transition matrix.")
        if not np.array_equal(self.net, expected_net):
            raise ValueError("Orbit net vector is inconsistent with the transition matrix.")
        if self.destroyed_incidence_count != destroyed:
            raise ValueError("Destroyed-incidence count is inconsistent with the transition matrix.")
        if self.unchanged_incidence_count != unchanged:
            raise ValueError("Unchanged-incidence count is inconsistent with the transition matrix.")
        if self.transition_incidence_count != int(self.transition.sum()):
            raise ValueError("Transition-incidence count is inconsistent with the transition matrix.")
        if int(self.net.sum()) != -self.destroyed_incidence_count:
            raise ValueError("Net orbit incidence change must equal negative destroyed incidences.")
        if np.any(self.absolute_delta < np.abs(self.net)):
            raise ValueError("Orbit absolute delta cannot be smaller than absolute aggregate net delta.")

    def csv_row(self) -> dict[str, int | float]:
        row: dict[str, int | float] = {
            "round": self.round_index,
            "selected_edge_id": self.selected_edge_id,
            "u": self.u,
            "v": self.v,
            "relshift_score": self.relshift_score,
            "degree_score": self.degree_score,
            "support_score": self.support_score,
            "active_edges_before": self.active_edges_before,
            "active_edges_after": self.active_edges_after,
            "changed_node_count": self.changed_node_count,
            "directly_attached_size": self.directly_attached_size,
            "four_node_pair_count": self.four_node_pair_count,
            "destroyed_incidence_count": self.destroyed_incidence_count,
            "unchanged_incidence_count": self.unchanged_incidence_count,
            "transition_incidence_count": self.transition_incidence_count,
        }
        for orbit_id in range(ORBIT_DIM):
            row[f"orbit_{orbit_id}_loss"] = int(self.loss[orbit_id])
            row[f"orbit_{orbit_id}_gain"] = int(self.gain[orbit_id])
            row[f"orbit_{orbit_id}_net"] = int(self.net[orbit_id])
            row[f"orbit_{orbit_id}_absolute_delta"] = int(self.absolute_delta[orbit_id])
        return row


def summarize_transition_matrix(
    transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    matrix = np.asarray(transition, dtype=np.int64)
    _require_shape("transition", matrix, (ORBIT_DIM, ORBIT_TRANSITION_COLUMNS))
    if np.any(matrix < 0):
        raise ValueError("Orbit transition counts must be non-negative.")

    loss = np.zeros(ORBIT_DIM, dtype=np.int64)
    gain = np.zeros(ORBIT_DIM, dtype=np.int64)
    unchanged = 0
    for source in range(ORBIT_DIM):
        for destination in range(ORBIT_TRANSITION_COLUMNS):
            count = int(matrix[source, destination])
            if count == 0:
                continue
            if destination == source:
                unchanged += count
                continue
            loss[source] += count
            if destination < ORBIT_DIM:
                gain[destination] += count
    destroyed = int(matrix[:, DESTROYED_ORBIT_COLUMN].sum())
    net = gain - loss
    return loss, gain, net, destroyed, unchanged


def write_orbit_explainability_artifacts(
    artifact_dir: str | Path,
    *,
    events: Iterable[OrbitEdgeEvent],
    dataset: str,
    pruning_seed: int,
    score_mode: str,
    target_rho: float,
) -> dict[str, Any]:
    output_dir = ensure_dir(Path(artifact_dir) / "orbit_explainability")
    materialized = list(events)
    for event in materialized:
        event.validate()

    edge_events_path = output_dir / "orbit_edge_events.csv"
    transitions_path = output_dir / "orbit_transitions.npz"
    registry_path = output_dir / "orbit_registry.json"
    manifest_path = output_dir / "orbit_explainability_manifest.json"

    rows = [event.csv_row() for event in materialized]
    fieldnames = list(rows[0].keys()) if rows else _empty_event_fieldnames()
    with edge_events_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    transitions = np.stack([event.transition for event in materialized], axis=0) if materialized else np.zeros(
        (0, ORBIT_DIM, ORBIT_TRANSITION_COLUMNS), dtype=np.int64
    )
    losses = np.stack([event.loss for event in materialized], axis=0) if materialized else np.zeros(
        (0, ORBIT_DIM), dtype=np.int64
    )
    gains = np.stack([event.gain for event in materialized], axis=0) if materialized else np.zeros(
        (0, ORBIT_DIM), dtype=np.int64
    )
    nets = np.stack([event.net for event in materialized], axis=0) if materialized else np.zeros(
        (0, ORBIT_DIM), dtype=np.int64
    )
    absolute_deltas = np.stack([event.absolute_delta for event in materialized], axis=0) if materialized else np.zeros(
        (0, ORBIT_DIM), dtype=np.int64
    )
    np.savez_compressed(
        transitions_path,
        schema_version=np.asarray(ORBIT_TRANSITION_SCHEMA_VERSION),
        registry_version=np.asarray(ORBIT_REGISTRY_VERSION),
        rounds=np.asarray([event.round_index for event in materialized], dtype=np.int64),
        selected_edge_ids=np.asarray([event.selected_edge_id for event in materialized], dtype=np.int64),
        selected_edges=np.asarray([(event.u, event.v) for event in materialized], dtype=np.int64).reshape((-1, 2)),
        transitions=transitions,
        cumulative_transition=transitions.sum(axis=0, dtype=np.int64),
        loss=losses,
        gain=gains,
        net=nets,
        absolute_delta=absolute_deltas,
    )
    write_json(registry_path, orbit_registry_payload())

    manifest = {
        "schema_version": ORBIT_EXPLAINABILITY_MANIFEST_VERSION,
        "event_schema_version": ORBIT_EVENT_SCHEMA_VERSION,
        "transition_schema_version": ORBIT_TRANSITION_SCHEMA_VERSION,
        "orbit_registry_version": ORBIT_REGISTRY_VERSION,
        "dataset": str(dataset),
        "pruning_seed": int(pruning_seed),
        "score_mode": str(score_mode),
        "target_rho": float(target_rho),
        "event_count": len(materialized),
        "transition_shape": list(transitions.shape),
        "cumulative_transition_shape": [ORBIT_DIM, ORBIT_TRANSITION_COLUMNS],
        "files": {
            "edge_events_csv": _file_record(edge_events_path),
            "transitions_npz": _file_record(transitions_path),
            "orbit_registry_json": _file_record(registry_path),
        },
    }
    write_json(manifest_path, manifest)

    return {
        "enabled": True,
        "schema_version": ORBIT_EXPLAINABILITY_MANIFEST_VERSION,
        "event_schema_version": ORBIT_EVENT_SCHEMA_VERSION,
        "transition_schema_version": ORBIT_TRANSITION_SCHEMA_VERSION,
        "orbit_registry_version": ORBIT_REGISTRY_VERSION,
        "event_count": len(materialized),
        "artifact_dir": str(output_dir),
        "edge_events_path": str(edge_events_path),
        "transitions_path": str(transitions_path),
        "registry_path": str(registry_path),
        "manifest_path": str(manifest_path),
    }


def _require_shape(name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, found {array.shape}.")


def _empty_event_fieldnames() -> list[str]:
    base = [
        "round",
        "selected_edge_id",
        "u",
        "v",
        "relshift_score",
        "degree_score",
        "support_score",
        "active_edges_before",
        "active_edges_after",
        "changed_node_count",
        "directly_attached_size",
        "four_node_pair_count",
        "destroyed_incidence_count",
        "unchanged_incidence_count",
        "transition_incidence_count",
    ]
    for orbit_id in range(ORBIT_DIM):
        base.extend(
            [
                f"orbit_{orbit_id}_loss",
                f"orbit_{orbit_id}_gain",
                f"orbit_{orbit_id}_net",
                f"orbit_{orbit_id}_absolute_delta",
            ]
        )
    return base


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


__all__ = [
    "DESTROYED_ORBIT_COLUMN",
    "ORBIT_EVENT_SCHEMA_VERSION",
    "ORBIT_EXPLAINABILITY_MANIFEST_VERSION",
    "ORBIT_TRANSITION_COLUMNS",
    "ORBIT_TRANSITION_SCHEMA_VERSION",
    "OrbitEdgeEvent",
    "summarize_transition_matrix",
    "write_orbit_explainability_artifacts",
]
