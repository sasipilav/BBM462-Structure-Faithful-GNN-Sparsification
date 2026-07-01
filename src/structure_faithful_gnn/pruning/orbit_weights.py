from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..gdv.orbits import (
    ANALYSIS_ORBIT_GROUPS,
    ORBIT_BY_ENUM_NAME,
    ORBIT_DIM,
    ORBIT_REGISTRY,
    ORBIT_REGISTRY_VERSION,
)
from ..utils.io import write_json


ORBIT_WEIGHT_SCHEMA_VERSION = "relshift-orbit-weight-spec-v1"
_ORBIT_FEATURE_PATTERN = re.compile(r"^orbit_(\d+)_")
_ALLOWED_NORMALIZATIONS = {"none", "mean_one", "sum_one"}
_ALLOWED_TRANSFORMS = {"positive", "absolute", "softplus"}
_ALLOWED_AGGREGATIONS = {"mean", "median"}


@dataclass(frozen=True, slots=True)
class OrbitWeightSpec:
    weights: tuple[float, ...]
    mode: str = "uniform"
    normalization: str = "none"
    source: str = "uniform"
    source_sha256: str | None = None
    leave_out_orbits: tuple[int, ...] = ()
    leave_out_groups: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validated = _validate_weights(self.weights)
        object.__setattr__(self, "weights", validated)
        if self.normalization not in _ALLOWED_NORMALIZATIONS:
            raise ValueError(
                f"Unknown orbit-weight normalization {self.normalization!r}; "
                f"expected one of {sorted(_ALLOWED_NORMALIZATIONS)}."
            )
        orbit_ids = tuple(sorted({int(value) for value in self.leave_out_orbits}))
        if any(value < 0 or value >= ORBIT_DIM for value in orbit_ids):
            raise ValueError(f"leave_out_orbits must be within 0..{ORBIT_DIM - 1}.")
        groups = tuple(sorted({str(value) for value in self.leave_out_groups}))
        unknown = sorted(set(groups) - set(ANALYSIS_ORBIT_GROUPS))
        if unknown:
            raise ValueError(
                f"Unknown leave-out orbit groups {unknown}; "
                f"expected names from {sorted(ANALYSIS_ORBIT_GROUPS)}."
            )
        object.__setattr__(self, "leave_out_orbits", orbit_ids)
        object.__setattr__(self, "leave_out_groups", groups)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_uniform(self) -> bool:
        return all(value == 1.0 for value in self.weights)

    @property
    def nonzero_orbit_count(self) -> int:
        return sum(value > 0.0 for value in self.weights)

    @property
    def active_orbits(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.weights) if value > 0.0)

    @property
    def fingerprint(self) -> str:
        # Identity is semantic and path-independent: moving the same source
        # table/spec must not create a different pruning method identity.
        payload = {
            "schema_version": ORBIT_WEIGHT_SCHEMA_VERSION,
            "orbit_registry_version": ORBIT_REGISTRY_VERSION,
            "orbit_dim": ORBIT_DIM,
            "mode": self.mode,
            "normalization": self.normalization,
            "source_sha256": self.source_sha256,
            "weights": list(self.weights),
            "leave_out_orbits": list(self.leave_out_orbits),
            "leave_out_groups": list(self.leave_out_groups),
            "metadata": dict(self.metadata),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": ORBIT_WEIGHT_SCHEMA_VERSION,
            "orbit_registry_version": ORBIT_REGISTRY_VERSION,
            "orbit_dim": ORBIT_DIM,
            "mode": self.mode,
            "normalization": self.normalization,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "weights": list(self.weights),
            "nonzero_orbit_count": self.nonzero_orbit_count,
            "active_orbits": list(self.active_orbits),
            "leave_out_orbits": list(self.leave_out_orbits),
            "leave_out_groups": list(self.leave_out_groups),
            "metadata": dict(self.metadata),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


def uniform_orbit_weight_spec() -> OrbitWeightSpec:
    return OrbitWeightSpec(weights=(1.0,) * ORBIT_DIM)


def resolve_orbit_weight_spec(options: Mapping[str, Any] | None) -> OrbitWeightSpec:
    options = dict(options or {})
    inline = options.get("orbit_weights")
    file_value = options.get("orbit_weight_file")
    if inline is not None and file_value is not None:
        raise ValueError("Specify only one of orbit_weights and orbit_weight_file.")

    base_metadata: dict[str, Any] = {}
    source_leave_out_orbits: tuple[int, ...] = ()
    source_leave_out_groups: tuple[str, ...] = ()
    if file_value is not None:
        source_path = Path(str(file_value)).expanduser().resolve()
        payload = _load_weight_payload(source_path)
        weights = _extract_weight_vector(payload)
        source = str(source_path)
        source_sha256 = _sha256_file(source_path)
        mode = str(payload.get("mode", "file"))
        source_leave_out_orbits = _parse_orbit_ids(payload.get("leave_out_orbits", ()))
        source_leave_out_groups = _parse_group_names(payload.get("leave_out_groups", ()))
        base_metadata = {
            "source_spec_fingerprint": payload.get("fingerprint"),
            "source_schema_version": payload.get("schema_version"),
            "source_normalization": payload.get("normalization"),
        }
    elif inline is not None:
        weights = _coerce_inline_weights(inline)
        source = "inline"
        source_sha256 = None
        mode = "manual"
    else:
        weights = np.ones(ORBIT_DIM, dtype=np.float64)
        source = "uniform"
        source_sha256 = None
        mode = "uniform"

    requested_leave_out_orbits = _parse_orbit_ids(options.get("orbit_leave_out_orbits", ()))
    requested_leave_out_groups = _parse_group_names(options.get("orbit_leave_out_groups", ()))
    leave_out_orbits = tuple(sorted(set(source_leave_out_orbits) | set(requested_leave_out_orbits)))
    leave_out_groups = tuple(sorted(set(source_leave_out_groups) | set(requested_leave_out_groups)))
    zeroed = set(leave_out_orbits)
    for group in leave_out_groups:
        zeroed.update(ANALYSIS_ORBIT_GROUPS[group])
    if zeroed:
        weights = np.asarray(weights, dtype=np.float64).copy()
        weights[list(sorted(zeroed))] = 0.0
    if requested_leave_out_orbits or requested_leave_out_groups:
        mode = "leave_out" if mode == "uniform" else f"{mode}_leave_out"

    normalization = str(options.get("orbit_weight_normalization", "none")).strip().lower()
    weights = _normalize_weights(weights, normalization)
    validated = _validate_weights(weights)

    metadata = {
        **base_metadata,
        "orbit_labels": [spec.label for spec in ORBIT_REGISTRY],
    }
    return OrbitWeightSpec(
        weights=validated,
        mode=mode,
        normalization=normalization,
        source=source,
        source_sha256=source_sha256,
        leave_out_orbits=leave_out_orbits,
        leave_out_groups=leave_out_groups,
        metadata=metadata,
    )


def load_orbit_weight_spec(path: str | Path) -> OrbitWeightSpec:
    path = Path(path).expanduser().resolve()
    payload = _load_weight_payload(path)
    weights = _extract_weight_vector(payload)
    return OrbitWeightSpec(
        weights=_validate_weights(weights),
        mode=str(payload.get("mode", "file")),
        normalization=str(payload.get("normalization", "none")),
        source=str(path),
        source_sha256=_sha256_file(path),
        leave_out_orbits=tuple(int(value) for value in payload.get("leave_out_orbits", ())),
        leave_out_groups=tuple(str(value) for value in payload.get("leave_out_groups", ())),
        metadata=dict(payload.get("metadata", {})),
    )


def write_orbit_weight_spec(path: str | Path, spec: OrbitWeightSpec) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, spec.as_dict())
    return path


def derive_orbit_weight_spec_from_table(
    *,
    table_path: str | Path,
    value_column: str,
    filters: Mapping[str, str | int | float] | None = None,
    transform: str = "positive",
    aggregation: str = "median",
    normalization: str = "mean_one",
    minimum_sign_consistency: float | None = None,
    minimum_nonzero_fraction: float | None = None,
    floor: float = 0.0,
) -> OrbitWeightSpec:
    """Build a non-negative orbit-weight vector from a regression summary table.

    The input must contain a ``feature`` column whose orbit features begin with
    ``orbit_<id>_``. All user-provided filters are exact string matches. Multiple
    rows for one orbit are aggregated only after filtering, so the produced spec
    records a reproducible, auditable source slice.
    """

    path = Path(table_path).expanduser().resolve()
    if transform not in _ALLOWED_TRANSFORMS:
        raise ValueError(f"transform must be one of {sorted(_ALLOWED_TRANSFORMS)}.")
    if aggregation not in _ALLOWED_AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {sorted(_ALLOWED_AGGREGATIONS)}.")
    if floor < 0.0 or not math.isfinite(floor):
        raise ValueError("floor must be a finite non-negative number.")

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Orbit-weight source table is empty: {path}")
    if "feature" not in rows[0] or value_column not in rows[0]:
        raise ValueError(
            f"Orbit-weight source must contain feature and {value_column!r} columns."
        )

    selected: list[dict[str, str]] = []
    for row in rows:
        if filters and any(str(row.get(key)) != str(value) for key, value in filters.items()):
            continue
        match = _ORBIT_FEATURE_PATTERN.match(str(row.get("feature", "")))
        if not match:
            continue
        orbit_id = int(match.group(1))
        if not 0 <= orbit_id < ORBIT_DIM:
            continue
        if minimum_sign_consistency is not None:
            observed = float(row.get("sign_consistency", "nan"))
            if not math.isfinite(observed) or observed < minimum_sign_consistency:
                continue
        if minimum_nonzero_fraction is not None:
            observed = float(row.get("nonzero_fraction", "nan"))
            if not math.isfinite(observed) or observed < minimum_nonzero_fraction:
                continue
        selected.append(row)
    if not selected:
        raise ValueError("No orbit rows remain after applying source-table filters.")

    values_by_orbit: dict[int, list[float]] = {index: [] for index in range(ORBIT_DIM)}
    for row in selected:
        orbit_id = int(_ORBIT_FEATURE_PATTERN.match(str(row["feature"])).group(1))  # type: ignore[union-attr]
        value = float(row[value_column])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite source value for orbit {orbit_id}: {value}")
        values_by_orbit[orbit_id].append(value)

    raw = np.zeros(ORBIT_DIM, dtype=np.float64)
    for orbit_id, values in values_by_orbit.items():
        if not values:
            continue
        vector = np.asarray(values, dtype=np.float64)
        aggregate = float(np.mean(vector) if aggregation == "mean" else np.median(vector))
        raw[orbit_id] = _transform_weight_value(aggregate, transform)
    if floor > 0.0:
        raw = np.where(raw > 0.0, np.maximum(raw, floor), 0.0)
    weights = _normalize_weights(raw, normalization)

    metadata = {
        "value_column": value_column,
        "filters": dict(filters or {}),
        "transform": transform,
        "aggregation": aggregation,
        "minimum_sign_consistency": minimum_sign_consistency,
        "minimum_nonzero_fraction": minimum_nonzero_fraction,
        "floor": floor,
        "selected_row_count": len(selected),
    }
    return OrbitWeightSpec(
        weights=_validate_weights(weights),
        mode="learned_from_table",
        normalization=normalization,
        source=str(path),
        source_sha256=_sha256_file(path),
        metadata=metadata,
    )


def _coerce_inline_weights(value: Any) -> np.ndarray:
    if isinstance(value, Mapping):
        result = np.full(ORBIT_DIM, np.nan, dtype=np.float64)
        for key, item in value.items():
            orbit_id = _resolve_orbit_key(key)
            if not math.isnan(float(result[orbit_id])):
                raise ValueError(f"Duplicate inline weight for orbit {orbit_id}.")
            result[orbit_id] = float(item)
        missing = np.flatnonzero(np.isnan(result)).tolist()
        if missing:
            raise ValueError(f"Inline orbit-weight mapping is missing orbit ids {missing}.")
        return result
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("orbit_weights must be a 15-value sequence or complete orbit mapping.")
    return np.asarray(list(value), dtype=np.float64)


def _resolve_orbit_key(key: Any) -> int:
    if isinstance(key, int) or (isinstance(key, str) and key.strip().isdigit()):
        orbit_id = int(key)
    else:
        text = str(key).strip()
        if text in ORBIT_BY_ENUM_NAME:
            orbit_id = ORBIT_BY_ENUM_NAME[text].orbit_id
        else:
            by_label = {spec.label: spec.orbit_id for spec in ORBIT_REGISTRY}
            if text not in by_label:
                raise ValueError(f"Unknown orbit weight key {key!r}.")
            orbit_id = by_label[text]
    if not 0 <= orbit_id < ORBIT_DIM:
        raise ValueError(f"Orbit id {orbit_id} is outside 0..{ORBIT_DIM - 1}.")
    return orbit_id


def _parse_orbit_ids(value: Any) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    result = tuple(sorted({_resolve_orbit_key(item) for item in values}))
    return result


def _parse_group_names(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    result = tuple(sorted({str(item).strip() for item in values}))
    unknown = sorted(set(result) - set(ANALYSIS_ORBIT_GROUPS))
    if unknown:
        raise ValueError(
            f"Unknown orbit groups {unknown}; expected {sorted(ANALYSIS_ORBIT_GROUPS)}."
        )
    return result


def _load_weight_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Orbit weight file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Orbit weight file must contain a JSON object: {path}")
    schema = payload.get("schema_version")
    if schema not in (None, ORBIT_WEIGHT_SCHEMA_VERSION):
        raise ValueError(f"Unsupported orbit-weight schema {schema!r} in {path}.")
    registry = payload.get("orbit_registry_version")
    if registry not in (None, ORBIT_REGISTRY_VERSION):
        raise ValueError(
            f"Orbit registry mismatch in {path}: {registry!r} != {ORBIT_REGISTRY_VERSION!r}."
        )
    dimension = payload.get("orbit_dim")
    if dimension not in (None, ORBIT_DIM):
        raise ValueError(f"Orbit dimension mismatch in {path}: {dimension!r} != {ORBIT_DIM}.")
    return payload


def _extract_weight_vector(payload: Mapping[str, Any]) -> np.ndarray:
    if "weights" not in payload:
        raise ValueError("Orbit weight payload is missing the weights vector.")
    return np.asarray(payload["weights"], dtype=np.float64)


def _normalize_weights(weights: Sequence[float] | np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode not in _ALLOWED_NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {sorted(_ALLOWED_NORMALIZATIONS)}.")
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    _validate_weights(vector)
    if mode == "none":
        return vector.copy()
    total = float(vector.sum())
    if mode == "sum_one":
        return vector / total
    return vector * (ORBIT_DIM / total)


def _validate_weights(weights: Sequence[float] | np.ndarray) -> tuple[float, ...]:
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vector.size != ORBIT_DIM:
        raise ValueError(f"Orbit weights must have exactly {ORBIT_DIM} entries, found {vector.size}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Orbit weights must all be finite.")
    if np.any(vector < 0.0):
        raise ValueError("Orbit weights must be non-negative.")
    if not float(vector.sum()) > 0.0:
        raise ValueError("At least one orbit weight must be positive.")
    return tuple(float(value) for value in vector)


def _transform_weight_value(value: float, transform: str) -> float:
    if transform == "positive":
        return max(value, 0.0)
    if transform == "absolute":
        return abs(value)
    # Numerically stable softplus.
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
