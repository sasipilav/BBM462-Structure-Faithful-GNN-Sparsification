from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..gdv.orbits import ORBIT_DIM, ORBIT_REGISTRY_VERSION, orbit_registry_payload
from ..utils.io import ensure_dir, write_json


ORBIT_EVENT_SCHEMA_VERSION = "relshift-orbit-edge-events-v1"
ORBIT_TRANSITION_SCHEMA_VERSION = "relshift-orbit-transitions-v1"
ORBIT_CHECKPOINT_SCHEMA_VERSION = "relshift-orbit-checkpoints-v1"
ORBIT_CHECKPOINT_SNAPSHOT_SCHEMA_VERSION = "relshift-orbit-checkpoint-snapshots-v1"
ORBIT_EXPLAINABILITY_MANIFEST_VERSION = "relshift-orbit-explainability-manifest-v2"
DEFAULT_ORBIT_CHECKPOINT_RHOS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
DESTROYED_ORBIT_COLUMN = ORBIT_DIM
ORBIT_TRANSITION_COLUMNS = ORBIT_DIM + 1


@dataclass(frozen=True, slots=True)
class OrbitCheckpointTarget:
    """One deduplicated edge-removal target for checkpoint collection."""

    removed_edge_count: int
    requested_rhos: tuple[float, ...]
    is_final_target: bool = False

    def validate(self, *, original_edge_count: int, final_budget: int) -> None:
        if original_edge_count < 0 or final_budget < 0:
            raise ValueError("Edge counts must be non-negative.")
        if not 0 <= self.removed_edge_count <= final_budget <= original_edge_count:
            raise ValueError("Checkpoint target edge counts are inconsistent.")
        if not self.requested_rhos:
            raise ValueError("Checkpoint targets must retain at least one requested rho.")
        if any((not math.isfinite(rho)) or rho < 0.0 or rho > 1.0 for rho in self.requested_rhos):
            raise ValueError("Checkpoint target rho values must be finite and lie in [0, 1].")


def resolve_orbit_checkpoint_targets(
    *,
    original_edge_count: int,
    target_rho: float,
    requested_rhos: Iterable[float] | None = None,
) -> tuple[OrbitCheckpointTarget, ...]:
    """Resolve requested reduction fractions into unique reachable edge counts.

    RelShift's exact budget convention is retained: every target uses
    ``round(original_edge_count * rho)``. Fractions above the run's target rho
    are unreachable and omitted. Initial and configured final states are always
    included. Multiple fractions mapping to the same integer edge count are
    grouped so a snapshot is never duplicated.
    """

    if original_edge_count < 0:
        raise ValueError("original_edge_count must be non-negative.")
    target_rho = float(target_rho)
    if not math.isfinite(target_rho) or not 0.0 <= target_rho <= 1.0:
        raise ValueError("target_rho must be finite and lie in [0, 1].")
    final_budget = int(round(original_edge_count * target_rho))
    source = DEFAULT_ORBIT_CHECKPOINT_RHOS if requested_rhos is None else tuple(requested_rhos)
    normalized: list[float] = []
    for value in source:
        rho = float(value)
        if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
            raise ValueError("orbit checkpoint rho values must be finite and lie in [0, 1].")
        if rho <= target_rho + 1e-15:
            normalized.append(rho)
    normalized.extend((0.0, target_rho))

    grouped: dict[int, set[float]] = {}
    for rho in normalized:
        count = min(final_budget, max(0, int(round(original_edge_count * rho))))
        grouped.setdefault(count, set()).add(rho)

    targets: list[OrbitCheckpointTarget] = []
    for count in sorted(grouped):
        target = OrbitCheckpointTarget(
            removed_edge_count=count,
            requested_rhos=tuple(sorted(grouped[count])),
            is_final_target=count == final_budget,
        )
        target.validate(original_edge_count=original_edge_count, final_budget=final_budget)
        targets.append(target)
    return tuple(targets)


@dataclass(frozen=True, slots=True)
class OrbitCheckpoint:
    """Full-graph orbit distortion summary at one pruning checkpoint."""

    checkpoint_index: int
    round_index: int
    target_removed_edge_count: int
    requested_rhos: tuple[float, ...]
    removed_edge_count: int
    active_edge_count: int
    original_edge_count: int
    actual_rho: float
    event_count: int
    is_initial: bool
    is_final: bool
    raw_initial_total: np.ndarray
    raw_current_total: np.ndarray
    raw_signed_delta: np.ndarray
    raw_absolute_delta: np.ndarray
    raw_relative_absolute_delta: np.ndarray
    raw_mean_absolute_delta: np.ndarray
    raw_max_absolute_delta: np.ndarray
    raw_changed_node_count: np.ndarray
    raw_initial_nonzero_node_count: np.ndarray
    raw_current_nonzero_node_count: np.ndarray
    raw_any_changed_node_count: int
    standardized_signed_delta: np.ndarray
    standardized_absolute_delta: np.ndarray
    standardized_mean_absolute_delta: np.ndarray
    standardized_max_absolute_delta: np.ndarray
    standardized_l2_delta: np.ndarray
    cumulative_event_net: np.ndarray
    cumulative_event_absolute_delta: np.ndarray
    cumulative_transition: np.ndarray
    raw_snapshot: np.ndarray | None = None
    standardized_snapshot: np.ndarray | None = None

    @classmethod
    def from_snapshots(
        cls,
        *,
        checkpoint_index: int,
        round_index: int,
        target: OrbitCheckpointTarget,
        removed_edge_count: int,
        active_edge_count: int,
        original_edge_count: int,
        event_count: int,
        is_final: bool,
        initial_raw: np.ndarray,
        current_raw: np.ndarray,
        initial_standardized: np.ndarray,
        current_standardized: np.ndarray,
        cumulative_event_net: np.ndarray,
        cumulative_event_absolute_delta: np.ndarray,
        cumulative_transition: np.ndarray,
        store_node_snapshots: bool = False,
    ) -> "OrbitCheckpoint":
        target.validate(
            original_edge_count=int(original_edge_count),
            final_budget=int(original_edge_count),
        )
        initial_raw_int = _integral_orbit_matrix("initial_raw", initial_raw)
        current_raw_int = _integral_orbit_matrix("current_raw", current_raw)
        if current_raw_int.shape != initial_raw_int.shape:
            raise ValueError("Initial and current raw GDV snapshots must have equal shape.")
        initial_std = np.asarray(initial_standardized, dtype=np.float64)
        current_std = np.asarray(current_standardized, dtype=np.float64)
        if initial_std.shape != initial_raw_int.shape or current_std.shape != initial_raw_int.shape:
            raise ValueError("Standardized GDV snapshots must match raw GDV shape.")
        if not np.all(np.isfinite(initial_std)) or not np.all(np.isfinite(current_std)):
            raise ValueError("Standardized GDV snapshots must be finite.")

        raw_delta = current_raw_int - initial_raw_int
        std_delta = current_std - initial_std
        raw_initial_total = initial_raw_int.sum(axis=0, dtype=np.int64)
        raw_current_total = current_raw_int.sum(axis=0, dtype=np.int64)
        raw_signed_delta = raw_delta.sum(axis=0, dtype=np.int64)
        raw_absolute_delta = np.abs(raw_delta).sum(axis=0, dtype=np.int64)
        raw_relative_absolute_delta = raw_absolute_delta.astype(np.float64) / np.maximum(
            np.abs(raw_initial_total).astype(np.float64), 1.0
        )
        node_count = int(initial_raw_int.shape[0])
        denominator = float(max(node_count, 1))
        checkpoint = cls(
            checkpoint_index=int(checkpoint_index),
            round_index=int(round_index),
            target_removed_edge_count=int(target.removed_edge_count),
            requested_rhos=tuple(float(rho) for rho in target.requested_rhos),
            removed_edge_count=int(removed_edge_count),
            active_edge_count=int(active_edge_count),
            original_edge_count=int(original_edge_count),
            actual_rho=(float(removed_edge_count) / float(original_edge_count)) if original_edge_count else 0.0,
            event_count=int(event_count),
            is_initial=bool(int(removed_edge_count) == 0),
            is_final=bool(is_final),
            raw_initial_total=raw_initial_total,
            raw_current_total=raw_current_total,
            raw_signed_delta=raw_signed_delta,
            raw_absolute_delta=raw_absolute_delta,
            raw_relative_absolute_delta=raw_relative_absolute_delta,
            raw_mean_absolute_delta=raw_absolute_delta.astype(np.float64) / denominator,
            raw_max_absolute_delta=np.abs(raw_delta).max(axis=0, initial=0).astype(np.int64),
            raw_changed_node_count=np.count_nonzero(raw_delta, axis=0).astype(np.int64),
            raw_initial_nonzero_node_count=np.count_nonzero(initial_raw_int, axis=0).astype(np.int64),
            raw_current_nonzero_node_count=np.count_nonzero(current_raw_int, axis=0).astype(np.int64),
            raw_any_changed_node_count=int(np.count_nonzero(np.any(raw_delta != 0, axis=1))),
            standardized_signed_delta=std_delta.sum(axis=0, dtype=np.float64),
            standardized_absolute_delta=np.abs(std_delta).sum(axis=0, dtype=np.float64),
            standardized_mean_absolute_delta=np.abs(std_delta).sum(axis=0, dtype=np.float64) / denominator,
            standardized_max_absolute_delta=np.abs(std_delta).max(axis=0, initial=0.0),
            standardized_l2_delta=np.sqrt(np.square(std_delta).sum(axis=0, dtype=np.float64)),
            cumulative_event_net=np.asarray(cumulative_event_net, dtype=np.int64).copy(),
            cumulative_event_absolute_delta=np.asarray(
                cumulative_event_absolute_delta, dtype=np.int64
            ).copy(),
            cumulative_transition=np.asarray(cumulative_transition, dtype=np.int64).copy(),
            raw_snapshot=current_raw_int.copy() if store_node_snapshots else None,
            standardized_snapshot=current_std.copy() if store_node_snapshots else None,
        )
        checkpoint.validate()
        return checkpoint

    def validate(self) -> None:
        if self.checkpoint_index < 0 or self.round_index < 0:
            raise ValueError("Checkpoint and round indices must be non-negative.")
        if not 0 <= self.removed_edge_count <= self.original_edge_count:
            raise ValueError("Checkpoint removed-edge count is invalid.")
        if self.active_edge_count != self.original_edge_count - self.removed_edge_count:
            raise ValueError("Checkpoint active-edge count is inconsistent.")
        if self.event_count != self.removed_edge_count or self.round_index != self.event_count:
            raise ValueError("Exact sequential checkpoints require one event per removed edge.")
        if self.target_removed_edge_count != self.removed_edge_count:
            raise ValueError("A requested checkpoint must be captured at its exact target count.")
        if self.is_initial != (self.removed_edge_count == 0):
            raise ValueError("Checkpoint initial-state flag is inconsistent.")
        expected_rho = (
            float(self.removed_edge_count) / float(self.original_edge_count)
            if self.original_edge_count
            else 0.0
        )
        if not math.isclose(self.actual_rho, expected_rho, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Checkpoint actual rho is inconsistent with edge counts.")
        for name, vector in (
            ("raw_initial_total", self.raw_initial_total),
            ("raw_current_total", self.raw_current_total),
            ("raw_signed_delta", self.raw_signed_delta),
            ("raw_absolute_delta", self.raw_absolute_delta),
            ("raw_max_absolute_delta", self.raw_max_absolute_delta),
            ("raw_changed_node_count", self.raw_changed_node_count),
            ("raw_initial_nonzero_node_count", self.raw_initial_nonzero_node_count),
            ("raw_current_nonzero_node_count", self.raw_current_nonzero_node_count),
            ("cumulative_event_net", self.cumulative_event_net),
            ("cumulative_event_absolute_delta", self.cumulative_event_absolute_delta),
        ):
            _require_shape(name, np.asarray(vector), (ORBIT_DIM,))
        for name, vector in (
            ("raw_relative_absolute_delta", self.raw_relative_absolute_delta),
            ("raw_mean_absolute_delta", self.raw_mean_absolute_delta),
            ("standardized_signed_delta", self.standardized_signed_delta),
            ("standardized_absolute_delta", self.standardized_absolute_delta),
            ("standardized_mean_absolute_delta", self.standardized_mean_absolute_delta),
            ("standardized_max_absolute_delta", self.standardized_max_absolute_delta),
            ("standardized_l2_delta", self.standardized_l2_delta),
        ):
            _require_shape(name, np.asarray(vector), (ORBIT_DIM,))
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be finite.")
        _require_shape(
            "cumulative_transition",
            self.cumulative_transition,
            (ORBIT_DIM, ORBIT_TRANSITION_COLUMNS),
        )
        if np.any(self.raw_absolute_delta < np.abs(self.raw_signed_delta)):
            raise ValueError("Checkpoint raw absolute delta violates the triangle inequality.")
        if np.any(self.cumulative_event_absolute_delta < self.raw_absolute_delta):
            raise ValueError("Cumulative event absolute delta cannot be below checkpoint displacement.")
        if not np.array_equal(self.raw_current_total - self.raw_initial_total, self.raw_signed_delta):
            raise ValueError("Checkpoint raw totals do not reconstruct signed delta.")
        if not np.array_equal(self.cumulative_event_net, self.raw_signed_delta):
            raise ValueError("Cumulative event net does not match checkpoint raw signed delta.")
        _, _, transition_net, _, _ = summarize_transition_matrix(self.cumulative_transition)
        if not np.array_equal(transition_net, self.raw_signed_delta):
            raise ValueError("Cumulative transition matrix does not match checkpoint raw signed delta.")
        if self.raw_snapshot is not None:
            if self.raw_snapshot.ndim != 2 or self.raw_snapshot.shape[1] != ORBIT_DIM:
                raise ValueError("Raw checkpoint snapshot must have shape [num_nodes, 15].")
        if self.standardized_snapshot is not None:
            if self.standardized_snapshot.ndim != 2 or self.standardized_snapshot.shape[1] != ORBIT_DIM:
                raise ValueError("Standardized checkpoint snapshot must have shape [num_nodes, 15].")
        if (self.raw_snapshot is None) != (self.standardized_snapshot is None):
            raise ValueError("Raw and standardized node snapshots must be stored together.")

    def csv_row(self) -> dict[str, int | float | str]:
        row: dict[str, int | float | str] = {
            "checkpoint_index": self.checkpoint_index,
            "round": self.round_index,
            "target_removed_edge_count": self.target_removed_edge_count,
            "requested_rhos": json.dumps(list(self.requested_rhos), separators=(",", ":")),
            "removed_edge_count": self.removed_edge_count,
            "active_edge_count": self.active_edge_count,
            "original_edge_count": self.original_edge_count,
            "actual_rho": self.actual_rho,
            "event_count": self.event_count,
            "is_initial": int(self.is_initial),
            "is_final": int(self.is_final),
            "raw_total_l1_distortion": int(self.raw_absolute_delta.sum()),
            "standardized_total_l1_distortion": float(self.standardized_absolute_delta.sum()),
            "raw_any_changed_node_count": self.raw_any_changed_node_count,
        }
        for orbit_id in range(ORBIT_DIM):
            row[f"orbit_{orbit_id}_initial_total"] = int(self.raw_initial_total[orbit_id])
            row[f"orbit_{orbit_id}_current_total"] = int(self.raw_current_total[orbit_id])
            row[f"orbit_{orbit_id}_raw_signed_delta"] = int(self.raw_signed_delta[orbit_id])
            row[f"orbit_{orbit_id}_raw_absolute_delta"] = int(self.raw_absolute_delta[orbit_id])
            row[f"orbit_{orbit_id}_raw_relative_absolute_delta"] = float(
                self.raw_relative_absolute_delta[orbit_id]
            )
            row[f"orbit_{orbit_id}_raw_mean_absolute_delta"] = float(
                self.raw_mean_absolute_delta[orbit_id]
            )
            row[f"orbit_{orbit_id}_raw_max_absolute_delta"] = int(
                self.raw_max_absolute_delta[orbit_id]
            )
            row[f"orbit_{orbit_id}_raw_changed_node_count"] = int(
                self.raw_changed_node_count[orbit_id]
            )
            row[f"orbit_{orbit_id}_initial_nonzero_node_count"] = int(
                self.raw_initial_nonzero_node_count[orbit_id]
            )
            row[f"orbit_{orbit_id}_current_nonzero_node_count"] = int(
                self.raw_current_nonzero_node_count[orbit_id]
            )
            row[f"orbit_{orbit_id}_standardized_signed_delta"] = float(
                self.standardized_signed_delta[orbit_id]
            )
            row[f"orbit_{orbit_id}_standardized_absolute_delta"] = float(
                self.standardized_absolute_delta[orbit_id]
            )
            row[f"orbit_{orbit_id}_standardized_mean_absolute_delta"] = float(
                self.standardized_mean_absolute_delta[orbit_id]
            )
            row[f"orbit_{orbit_id}_standardized_max_absolute_delta"] = float(
                self.standardized_max_absolute_delta[orbit_id]
            )
            row[f"orbit_{orbit_id}_standardized_l2_delta"] = float(
                self.standardized_l2_delta[orbit_id]
            )
            row[f"orbit_{orbit_id}_cumulative_event_net"] = int(
                self.cumulative_event_net[orbit_id]
            )
            row[f"orbit_{orbit_id}_cumulative_event_absolute_delta"] = int(
                self.cumulative_event_absolute_delta[orbit_id]
            )
        return row


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
    checkpoints: Iterable[OrbitCheckpoint] = (),
    dataset: str,
    pruning_seed: int,
    score_mode: str,
    target_rho: float,
) -> dict[str, Any]:
    output_dir = ensure_dir(Path(artifact_dir) / "orbit_explainability")
    materialized = list(events)
    materialized_checkpoints = list(checkpoints)
    for event in materialized:
        event.validate()
    for checkpoint in materialized_checkpoints:
        checkpoint.validate()
    if materialized_checkpoints:
        indices = [checkpoint.checkpoint_index for checkpoint in materialized_checkpoints]
        removed_counts = [checkpoint.removed_edge_count for checkpoint in materialized_checkpoints]
        if indices != list(range(len(materialized_checkpoints))):
            raise ValueError("Orbit checkpoint indices must be contiguous and zero-based.")
        if removed_counts != sorted(set(removed_counts)):
            raise ValueError("Orbit checkpoints must have strictly increasing removed-edge counts.")
        if not materialized_checkpoints[0].is_initial:
            raise ValueError("The first orbit checkpoint must describe the initial graph.")
        if not materialized_checkpoints[-1].is_final:
            raise ValueError("The last orbit checkpoint must describe the final achieved graph.")

    edge_events_path = output_dir / "orbit_edge_events.csv"
    transitions_path = output_dir / "orbit_transitions.npz"
    checkpoints_path = output_dir / "orbit_checkpoints.csv"
    checkpoint_snapshots_path = output_dir / "orbit_checkpoint_snapshots.npz"
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

    checkpoint_rows = [checkpoint.csv_row() for checkpoint in materialized_checkpoints]
    checkpoint_fieldnames = (
        list(checkpoint_rows[0].keys()) if checkpoint_rows else _empty_checkpoint_fieldnames()
    )
    with checkpoints_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=checkpoint_fieldnames)
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    checkpoint_count = len(materialized_checkpoints)
    vector_shape = (0, ORBIT_DIM)
    transition_shape = (0, ORBIT_DIM, ORBIT_TRANSITION_COLUMNS)
    checkpoint_payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(ORBIT_CHECKPOINT_SNAPSHOT_SCHEMA_VERSION),
        "checkpoint_schema_version": np.asarray(ORBIT_CHECKPOINT_SCHEMA_VERSION),
        "registry_version": np.asarray(ORBIT_REGISTRY_VERSION),
        "checkpoint_indices": np.asarray(
            [checkpoint.checkpoint_index for checkpoint in materialized_checkpoints], dtype=np.int64
        ),
        "rounds": np.asarray(
            [checkpoint.round_index for checkpoint in materialized_checkpoints], dtype=np.int64
        ),
        "target_removed_edge_counts": np.asarray(
            [checkpoint.target_removed_edge_count for checkpoint in materialized_checkpoints], dtype=np.int64
        ),
        "removed_edge_counts": np.asarray(
            [checkpoint.removed_edge_count for checkpoint in materialized_checkpoints], dtype=np.int64
        ),
        "active_edge_counts": np.asarray(
            [checkpoint.active_edge_count for checkpoint in materialized_checkpoints], dtype=np.int64
        ),
        "actual_rhos": np.asarray(
            [checkpoint.actual_rho for checkpoint in materialized_checkpoints], dtype=np.float64
        ),
        "event_counts": np.asarray(
            [checkpoint.event_count for checkpoint in materialized_checkpoints], dtype=np.int64
        ),
        "is_initial": np.asarray(
            [checkpoint.is_initial for checkpoint in materialized_checkpoints], dtype=np.uint8
        ),
        "is_final": np.asarray(
            [checkpoint.is_final for checkpoint in materialized_checkpoints], dtype=np.uint8
        ),
        "requested_rhos_json": np.asarray(
            [
                json.dumps(list(checkpoint.requested_rhos), separators=(",", ":"))
                for checkpoint in materialized_checkpoints
            ]
        ),
        "raw_initial_total": _stack_checkpoint_vectors(
            materialized_checkpoints, "raw_initial_total", dtype=np.int64, empty_shape=vector_shape
        ),
        "raw_current_total": _stack_checkpoint_vectors(
            materialized_checkpoints, "raw_current_total", dtype=np.int64, empty_shape=vector_shape
        ),
        "raw_signed_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "raw_signed_delta", dtype=np.int64, empty_shape=vector_shape
        ),
        "raw_absolute_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "raw_absolute_delta", dtype=np.int64, empty_shape=vector_shape
        ),
        "raw_relative_absolute_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "raw_relative_absolute_delta", dtype=np.float64, empty_shape=vector_shape
        ),
        "raw_mean_absolute_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "raw_mean_absolute_delta", dtype=np.float64, empty_shape=vector_shape
        ),
        "raw_max_absolute_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "raw_max_absolute_delta", dtype=np.int64, empty_shape=vector_shape
        ),
        "raw_changed_node_count": _stack_checkpoint_vectors(
            materialized_checkpoints, "raw_changed_node_count", dtype=np.int64, empty_shape=vector_shape
        ),
        "raw_initial_nonzero_node_count": _stack_checkpoint_vectors(
            materialized_checkpoints, "raw_initial_nonzero_node_count", dtype=np.int64, empty_shape=vector_shape
        ),
        "raw_current_nonzero_node_count": _stack_checkpoint_vectors(
            materialized_checkpoints, "raw_current_nonzero_node_count", dtype=np.int64, empty_shape=vector_shape
        ),
        "raw_any_changed_node_count": np.asarray(
            [checkpoint.raw_any_changed_node_count for checkpoint in materialized_checkpoints], dtype=np.int64
        ),
        "standardized_signed_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "standardized_signed_delta", dtype=np.float64, empty_shape=vector_shape
        ),
        "standardized_absolute_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "standardized_absolute_delta", dtype=np.float64, empty_shape=vector_shape
        ),
        "standardized_mean_absolute_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "standardized_mean_absolute_delta", dtype=np.float64, empty_shape=vector_shape
        ),
        "standardized_max_absolute_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "standardized_max_absolute_delta", dtype=np.float64, empty_shape=vector_shape
        ),
        "standardized_l2_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "standardized_l2_delta", dtype=np.float64, empty_shape=vector_shape
        ),
        "cumulative_event_net": _stack_checkpoint_vectors(
            materialized_checkpoints, "cumulative_event_net", dtype=np.int64, empty_shape=vector_shape
        ),
        "cumulative_event_absolute_delta": _stack_checkpoint_vectors(
            materialized_checkpoints, "cumulative_event_absolute_delta", dtype=np.int64, empty_shape=vector_shape
        ),
        "cumulative_transition": (
            np.stack([checkpoint.cumulative_transition for checkpoint in materialized_checkpoints], axis=0)
            if materialized_checkpoints
            else np.zeros(transition_shape, dtype=np.int64)
        ),
    }
    snapshots_stored = bool(
        materialized_checkpoints and materialized_checkpoints[0].raw_snapshot is not None
    )
    if any(
        (checkpoint.raw_snapshot is not None) != snapshots_stored
        for checkpoint in materialized_checkpoints
    ):
        raise ValueError(
            "Node-level checkpoint snapshots must be enabled consistently for all checkpoints."
        )
    if snapshots_stored:
        checkpoint_payload["raw_snapshots"] = np.stack(
            [
                np.asarray(checkpoint.raw_snapshot, dtype=np.int64)
                for checkpoint in materialized_checkpoints
            ],
            axis=0,
        )
        checkpoint_payload["standardized_snapshots"] = np.stack(
            [
                np.asarray(checkpoint.standardized_snapshot, dtype=np.float64)
                for checkpoint in materialized_checkpoints
            ],
            axis=0,
        )
    np.savez_compressed(checkpoint_snapshots_path, **checkpoint_payload)
    write_json(registry_path, orbit_registry_payload())

    manifest = {
        "schema_version": ORBIT_EXPLAINABILITY_MANIFEST_VERSION,
        "event_schema_version": ORBIT_EVENT_SCHEMA_VERSION,
        "transition_schema_version": ORBIT_TRANSITION_SCHEMA_VERSION,
        "checkpoint_schema_version": ORBIT_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_snapshot_schema_version": ORBIT_CHECKPOINT_SNAPSHOT_SCHEMA_VERSION,
        "orbit_registry_version": ORBIT_REGISTRY_VERSION,
        "dataset": str(dataset),
        "pruning_seed": int(pruning_seed),
        "score_mode": str(score_mode),
        "target_rho": float(target_rho),
        "event_count": len(materialized),
        "checkpoint_count": checkpoint_count,
        "checkpoint_removed_edge_counts": [
            checkpoint.removed_edge_count for checkpoint in materialized_checkpoints
        ],
        "checkpoint_actual_rhos": [checkpoint.actual_rho for checkpoint in materialized_checkpoints],
        "checkpoint_node_snapshots_stored": snapshots_stored,
        "transition_shape": list(transitions.shape),
        "cumulative_transition_shape": [ORBIT_DIM, ORBIT_TRANSITION_COLUMNS],
        "checkpoint_vector_shape": [checkpoint_count, ORBIT_DIM],
        "checkpoint_transition_shape": [checkpoint_count, ORBIT_DIM, ORBIT_TRANSITION_COLUMNS],
        "files": {
            "edge_events_csv": _file_record(edge_events_path),
            "transitions_npz": _file_record(transitions_path),
            "checkpoints_csv": _file_record(checkpoints_path),
            "checkpoint_snapshots_npz": _file_record(checkpoint_snapshots_path),
            "orbit_registry_json": _file_record(registry_path),
        },
    }
    write_json(manifest_path, manifest)

    return {
        "enabled": True,
        "schema_version": ORBIT_EXPLAINABILITY_MANIFEST_VERSION,
        "event_schema_version": ORBIT_EVENT_SCHEMA_VERSION,
        "transition_schema_version": ORBIT_TRANSITION_SCHEMA_VERSION,
        "checkpoint_schema_version": ORBIT_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_snapshot_schema_version": ORBIT_CHECKPOINT_SNAPSHOT_SCHEMA_VERSION,
        "orbit_registry_version": ORBIT_REGISTRY_VERSION,
        "event_count": len(materialized),
        "checkpoint_count": checkpoint_count,
        "checkpoint_node_snapshots_stored": snapshots_stored,
        "artifact_dir": str(output_dir),
        "edge_events_path": str(edge_events_path),
        "transitions_path": str(transitions_path),
        "checkpoints_path": str(checkpoints_path),
        "checkpoint_snapshots_path": str(checkpoint_snapshots_path),
        "registry_path": str(registry_path),
        "manifest_path": str(manifest_path),
    }


def _stack_checkpoint_vectors(
    checkpoints: list[OrbitCheckpoint],
    attribute: str,
    *,
    dtype: np.dtype[Any] | type[Any],
    empty_shape: tuple[int, int],
) -> np.ndarray:
    if not checkpoints:
        return np.zeros(empty_shape, dtype=dtype)
    return np.stack(
        [np.asarray(getattr(checkpoint, attribute), dtype=dtype) for checkpoint in checkpoints], axis=0
    )


def _integral_orbit_matrix(name: str, value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != ORBIT_DIM:
        raise ValueError(f"{name} must have shape [num_nodes, {ORBIT_DIM}], found {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    if np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain non-negative raw orbit counts.")
    rounded = np.rint(matrix)
    if not np.allclose(matrix, rounded, rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} must contain integer-valued raw orbit counts.")
    return rounded.astype(np.int64)


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


def _empty_checkpoint_fieldnames() -> list[str]:
    base = [
        "checkpoint_index",
        "round",
        "target_removed_edge_count",
        "requested_rhos",
        "removed_edge_count",
        "active_edge_count",
        "original_edge_count",
        "actual_rho",
        "event_count",
        "is_initial",
        "is_final",
        "raw_total_l1_distortion",
        "standardized_total_l1_distortion",
        "raw_any_changed_node_count",
    ]
    for orbit_id in range(ORBIT_DIM):
        base.extend(
            [
                f"orbit_{orbit_id}_initial_total",
                f"orbit_{orbit_id}_current_total",
                f"orbit_{orbit_id}_raw_signed_delta",
                f"orbit_{orbit_id}_raw_absolute_delta",
                f"orbit_{orbit_id}_raw_relative_absolute_delta",
                f"orbit_{orbit_id}_raw_mean_absolute_delta",
                f"orbit_{orbit_id}_raw_max_absolute_delta",
                f"orbit_{orbit_id}_raw_changed_node_count",
                f"orbit_{orbit_id}_initial_nonzero_node_count",
                f"orbit_{orbit_id}_current_nonzero_node_count",
                f"orbit_{orbit_id}_standardized_signed_delta",
                f"orbit_{orbit_id}_standardized_absolute_delta",
                f"orbit_{orbit_id}_standardized_mean_absolute_delta",
                f"orbit_{orbit_id}_standardized_max_absolute_delta",
                f"orbit_{orbit_id}_standardized_l2_delta",
                f"orbit_{orbit_id}_cumulative_event_net",
                f"orbit_{orbit_id}_cumulative_event_absolute_delta",
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
    "DEFAULT_ORBIT_CHECKPOINT_RHOS",
    "DESTROYED_ORBIT_COLUMN",
    "ORBIT_CHECKPOINT_SCHEMA_VERSION",
    "ORBIT_CHECKPOINT_SNAPSHOT_SCHEMA_VERSION",
    "ORBIT_EVENT_SCHEMA_VERSION",
    "ORBIT_EXPLAINABILITY_MANIFEST_VERSION",
    "ORBIT_TRANSITION_COLUMNS",
    "ORBIT_TRANSITION_SCHEMA_VERSION",
    "OrbitCheckpoint",
    "OrbitCheckpointTarget",
    "OrbitEdgeEvent",
    "resolve_orbit_checkpoint_targets",
    "summarize_transition_matrix",
    "write_orbit_explainability_artifacts",
]
