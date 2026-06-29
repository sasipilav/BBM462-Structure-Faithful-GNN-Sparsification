from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..gdv.orbits import ANALYSIS_ORBIT_GROUPS, ORBIT_DIM, ORBIT_REGISTRY_VERSION
from ..utils.io import ensure_dir, read_json, write_json
from .artifacts import canonical_run_row, discover_method_run_dirs
from .orbit_explainability import (
    ORBIT_CHECKPOINT_SCHEMA_VERSION,
    ORBIT_CHECKPOINT_SNAPSHOT_SCHEMA_VERSION,
    ORBIT_EXPLAINABILITY_MANIFEST_VERSION,
    ORBIT_TRANSITION_COLUMNS,
)


ORBIT_SENSITIVITY_DATASET_SCHEMA_VERSION = "relshift-orbit-sensitivity-dataset-v1"
ORBIT_SENSITIVITY_REGRESSION_SCHEMA_VERSION = "relshift-orbit-sensitivity-regression-v1"

FIXED_EFFECT_COLUMNS: tuple[str, ...] = ("dataset", "model")

CONTROL_COLUMNS: tuple[str, ...] = (
    "achieved_edge_reduction",
    "largest_component_loss",
    "num_components",
    "clustering_delta",
    "degree_distribution_tv",
    "degree_distribution_js",
    "homophily_delta",
    "isolated_node_ratio_delta",
)

ORBIT_FEATURE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "raw_relative_absolute": tuple(
        f"orbit_{orbit_id}_raw_relative_absolute_delta" for orbit_id in range(ORBIT_DIM)
    ),
    "standardized_absolute": tuple(
        f"orbit_{orbit_id}_standardized_absolute_delta" for orbit_id in range(ORBIT_DIM)
    ),
    "standardized_mean_absolute": tuple(
        f"orbit_{orbit_id}_standardized_mean_absolute_delta" for orbit_id in range(ORBIT_DIM)
    ),
    "standardized_l2": tuple(
        f"orbit_{orbit_id}_standardized_l2_delta" for orbit_id in range(ORBIT_DIM)
    ),
    "transition_destroyed": tuple(
        f"orbit_{orbit_id}_transition_destroyed" for orbit_id in range(ORBIT_DIM)
    ),
    "transition_out": tuple(
        f"orbit_{orbit_id}_transition_out" for orbit_id in range(ORBIT_DIM)
    ),
}

TARGET_COLUMNS: tuple[str, ...] = (
    "accuracy_delta",
    "macro_f1_delta",
    "accuracy_loss",
    "macro_f1_loss",
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model",
    "method",
    "training_seed",
    "pruning_seed",
    "target_rho",
    "reduction_band",
    "pruning_artifact_id",
    "training_pair_id",
    "training_run_dir",
    "dense_run_dir",
    "pruning_artifact_dir",
    "orbit_manifest_path",
)

_REQUIRED_STRUCTURAL_CONTROLS = (
    "degree_distribution_tv",
    "degree_distribution_js",
    "homophily_delta",
    "isolated_node_ratio_delta",
)


@dataclass(frozen=True, slots=True)
class OrbitSensitivityDatasetResult:
    rows: tuple[dict[str, Any], ...]
    csv_path: Path
    json_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class OuterSplit:
    split_id: str
    held_out_value: str
    test_artifact_ids: tuple[str, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray

    def validate(self, frame: pd.DataFrame) -> None:
        if self.train_indices.size == 0 or self.test_indices.size == 0:
            raise ValueError(f"Outer split {self.split_id} has an empty train or test partition.")
        train_ids = set(frame.iloc[self.train_indices]["pruning_artifact_id"].astype(str))
        test_ids = set(frame.iloc[self.test_indices]["pruning_artifact_id"].astype(str))
        overlap = train_ids.intersection(test_ids)
        if overlap:
            raise ValueError(
                f"Outer split {self.split_id} leaks pruning artifacts across train/test: {sorted(overlap)}"
            )


@dataclass(frozen=True, slots=True)
class RegressionRunResult:
    output_dir: Path
    fold_metrics_path: Path
    coefficients_path: Path
    coefficient_stability_path: Path
    permutation_importance_path: Path
    spearman_path: Path
    summary_path: Path
    manifest_path: Path


def build_orbit_sensitivity_dataset(
    *,
    training_root: str | Path,
    output_dir: str | Path,
    dense_root: str | Path | None = None,
    require_full_controls: bool = True,
) -> OrbitSensitivityDatasetResult:
    """Join RelShift orbit checkpoints with matched dense training outcomes.

    Dense baselines are matched exactly on ``(dataset, model, training_seed)``.
    Orbit features are sourced only from the final exact checkpoint of the
    pruning artifact. Duplicate training pairs and checksum mismatches are
    rejected instead of silently deduplicated.
    """

    training_root = Path(training_root)
    dense_root = training_root if dense_root is None else Path(dense_root)
    output_dir = ensure_dir(output_dir)

    relshift_dirs = discover_method_run_dirs(training_root, method="relshift")
    dense_dirs = discover_method_run_dirs(dense_root, method="dense")
    dense_by_key: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    for run_dir in dense_dirs:
        row = canonical_run_row(run_dir)
        key = (str(row["dataset"]), str(row["model"]), int(row["seed"]))
        if key in dense_by_key:
            raise ValueError(f"Duplicate dense baseline for {key}: {dense_by_key[key][0]} and {run_dir}")
        dense_by_key[key] = (run_dir, row)

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str, int]] = set()
    source_manifests: dict[str, str] = {}
    for run_dir in relshift_dirs:
        run_row = canonical_run_row(run_dir)
        dataset = str(run_row["dataset"])
        model = str(run_row["model"])
        training_seed = int(run_row["seed"])
        dense_key = (dataset, model, training_seed)
        if dense_key not in dense_by_key:
            raise FileNotFoundError(
                f"Missing dense baseline for dataset={dataset}, model={model}, seed={training_seed}."
            )
        dense_dir, dense_row = dense_by_key[dense_key]

        artifacts = _load_training_run(run_dir)
        pruning_artifact_dir = _resolve_pruning_artifact_dir(run_dir, artifacts)
        (
            checkpoint,
            transition,
            orbit_manifest_path,
            orbit_manifest_sha,
            orbit_manifest,
        ) = _load_final_orbit_checkpoint(pruning_artifact_dir)
        source_manifests[str(orbit_manifest_path)] = orbit_manifest_sha
        target_rho = _extract_target_rho(artifacts)
        pruning_seed = _validate_orbit_training_identity(
            artifacts=artifacts,
            orbit_manifest=orbit_manifest,
            dataset=dataset,
            target_rho=target_rho,
            checkpoint=checkpoint,
        )
        controls = _extract_structural_controls(artifacts, require_full=require_full_controls)
        achieved = float(run_row["achieved_edge_reduction"])
        original_edge_count = int(checkpoint["original_edge_count"])
        tolerance = (0.5 / float(max(original_edge_count, 1))) + 1e-12
        if abs(float(checkpoint["actual_rho"]) - achieved) > tolerance:
            raise ValueError(
                f"Final orbit checkpoint rho {checkpoint['actual_rho']} does not match training artifact "
                f"edge reduction {achieved} within one half-edge tolerance for {run_dir}."
            )

        artifact_id = _pruning_artifact_id(
            pruning_artifact_dir,
            dataset=dataset,
            checkpoint=checkpoint,
            cumulative_transition=transition,
        )
        pair_key = (artifact_id, dataset, model, training_seed)
        if pair_key in seen_pairs:
            raise ValueError(f"Duplicate sensitivity training pair: {pair_key}")
        seen_pairs.add(pair_key)
        training_pair_id = _sha256_text("|".join(map(str, pair_key)))[:24]

        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "method": "relshift",
            "training_seed": training_seed,
            "pruning_seed": pruning_seed,
            "target_rho": target_rho,
            "reduction_band": _reduction_band(target_rho, achieved),
            "pruning_artifact_id": artifact_id,
            "training_pair_id": training_pair_id,
            "training_run_dir": str(run_dir.resolve()),
            "dense_run_dir": str(dense_dir.resolve()),
            "pruning_artifact_dir": str(pruning_artifact_dir.resolve()),
            "orbit_manifest_path": str(orbit_manifest_path.resolve()),
            "achieved_edge_reduction": achieved,
            "pruned_accuracy": float(run_row["accuracy"]),
            "dense_accuracy": float(dense_row["accuracy"]),
            "accuracy_delta": float(run_row["accuracy"]) - float(dense_row["accuracy"]),
            "accuracy_loss": float(dense_row["accuracy"]) - float(run_row["accuracy"]),
            "pruned_macro_f1": float(run_row["macro_f1"]),
            "dense_macro_f1": float(dense_row["macro_f1"]),
            "macro_f1_delta": float(run_row["macro_f1"]) - float(dense_row["macro_f1"]),
            "macro_f1_loss": float(dense_row["macro_f1"]) - float(run_row["macro_f1"]),
            "largest_component_ratio": float(run_row["largest_component_ratio"]),
            "largest_component_loss": 1.0 - float(run_row["largest_component_ratio"]),
            "num_components": int(run_row["num_components"]),
            "clustering_delta": float(run_row["clustering_delta"]),
            "degree_distribution_tv": float(controls["degree_distribution_tv"]),
            "degree_distribution_js": float(controls["degree_distribution_js"]),
            "homophily_delta": float(controls["homophily_delta"]),
            "isolated_node_ratio_delta": float(controls["isolated_node_ratio_delta"]),
            "checkpoint_index": int(checkpoint["checkpoint_index"]),
            "checkpoint_round": int(checkpoint["round"]),
            "checkpoint_event_count": int(checkpoint["event_count"]),
            "checkpoint_removed_edge_count": int(checkpoint["removed_edge_count"]),
            "checkpoint_original_edge_count": original_edge_count,
            "raw_total_l1_distortion": int(checkpoint["raw_total_l1_distortion"]),
            "standardized_total_l1_distortion": float(
                checkpoint["standardized_total_l1_distortion"]
            ),
            "raw_any_changed_node_count": int(checkpoint["raw_any_changed_node_count"]),
        }
        _append_orbit_features(row, checkpoint, transition)
        rows.append(row)

    rows.sort(
        key=lambda item: (
            item["dataset"],
            item["model"],
            int(item["training_seed"]),
            float(item["target_rho"]),
            int(item["pruning_seed"]),
            item["pruning_artifact_id"],
        )
    )
    if not rows:
        raise ValueError("No RelShift sensitivity rows were produced.")
    _validate_sensitivity_rows(rows)

    csv_path = output_dir / "orbit_sensitivity_dataset.csv"
    json_path = output_dir / "orbit_sensitivity_dataset.json"
    manifest_path = output_dir / "orbit_sensitivity_dataset_manifest.json"
    _write_rows_csv(csv_path, rows)
    write_json(json_path, {"schema_version": ORBIT_SENSITIVITY_DATASET_SCHEMA_VERSION, "rows": rows})
    manifest = {
        "schema_version": ORBIT_SENSITIVITY_DATASET_SCHEMA_VERSION,
        "orbit_registry_version": ORBIT_REGISTRY_VERSION,
        "row_count": len(rows),
        "pruning_artifact_count": len({row["pruning_artifact_id"] for row in rows}),
        "datasets": sorted({row["dataset"] for row in rows}),
        "models": sorted({row["model"] for row in rows}),
        "training_seeds": sorted({int(row["training_seed"]) for row in rows}),
        "pruning_seeds": sorted({int(row["pruning_seed"]) for row in rows}),
        "control_columns": list(CONTROL_COLUMNS),
        "fixed_effect_columns": list(FIXED_EFFECT_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "orbit_feature_families": {
            name: list(columns) for name, columns in ORBIT_FEATURE_FAMILIES.items()
        },
        "source_orbit_manifests": source_manifests,
        "files": {
            "csv": _file_record(csv_path),
            "json": _file_record(json_path),
        },
    }
    write_json(manifest_path, manifest)
    return OrbitSensitivityDatasetResult(tuple(rows), csv_path, json_path, manifest_path)


def load_orbit_sensitivity_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if payload.get("schema_version") != ORBIT_SENSITIVITY_DATASET_SCHEMA_VERSION:
            raise ValueError(f"Unsupported sensitivity dataset schema in {path}.")
        frame = pd.DataFrame(payload.get("rows", []))
    else:
        frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Sensitivity dataset is empty: {path}")
    _validate_sensitivity_frame(frame)
    return frame


def build_leakage_safe_outer_splits(
    frame: pd.DataFrame,
    *,
    split_mode: str,
    artifact_folds: int = 3,
) -> tuple[OuterSplit, ...]:
    """Construct outer folds with zero pruning-artifact overlap.

    Category-based modes hold out one dataset/model/reduction/pruning seed. When
    the held category contains multiple pruning artifacts, deterministic
    artifact subfolds ensure that the same sparsified graph never appears in
    training, even through another model or training seed.
    """

    _validate_sensitivity_frame(frame)
    mode_to_column = {
        "dataset": "dataset",
        "model": "model",
        "reduction_band": "reduction_band",
        "pruning_seed": "pruning_seed",
        "pruning_artifact": "pruning_artifact_id",
    }
    if split_mode not in mode_to_column:
        raise ValueError(f"Unknown split_mode {split_mode!r}; expected one of {sorted(mode_to_column)}")
    if artifact_folds <= 0:
        raise ValueError("artifact_folds must be positive.")
    column = mode_to_column[split_mode]
    splits: list[OuterSplit] = []
    values = sorted(frame[column].astype(str).unique())
    if len(values) < 2 and split_mode != "pruning_artifact":
        raise ValueError(f"split_mode={split_mode} requires at least two unique held-out values.")

    for held_value in values:
        candidate_mask = frame[column].astype(str).to_numpy() == held_value
        candidate_ids = sorted(frame.loc[candidate_mask, "pruning_artifact_id"].astype(str).unique())
        fold_count = 1 if split_mode == "pruning_artifact" else min(artifact_folds, len(candidate_ids))
        for fold_index in range(fold_count):
            test_ids = tuple(candidate_ids[fold_index::fold_count])
            if not test_ids:
                continue
            test_mask = candidate_mask & frame["pruning_artifact_id"].astype(str).isin(test_ids).to_numpy()
            train_mask = (~candidate_mask) & (~frame["pruning_artifact_id"].astype(str).isin(test_ids).to_numpy())
            split = OuterSplit(
                split_id=f"{split_mode}={held_value};artifact_fold={fold_index}",
                held_out_value=held_value,
                test_artifact_ids=test_ids,
                train_indices=np.flatnonzero(train_mask),
                test_indices=np.flatnonzero(test_mask),
            )
            split.validate(frame)
            splits.append(split)
    if not splits:
        raise ValueError(f"No valid outer splits produced for split_mode={split_mode}.")
    return tuple(splits)


def run_controlled_orbit_regression(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    target: str = "accuracy_loss",
    split_modes: Sequence[str] = ("dataset", "model", "reduction_band", "pruning_seed"),
    orbit_feature_family: str = "standardized_absolute",
    estimators: Sequence[str] = ("ridge", "elastic_net"),
    artifact_folds: int = 3,
    permutation_repeats: int = 32,
    bootstrap_repeats: int = 200,
    random_seed: int = 0,
) -> RegressionRunResult:
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported regression target {target!r}.")
    if orbit_feature_family not in ORBIT_FEATURE_FAMILIES:
        raise ValueError(f"Unsupported orbit feature family {orbit_feature_family!r}.")
    if permutation_repeats < 0 or bootstrap_repeats < 0:
        raise ValueError("Permutation and bootstrap repeats must be non-negative.")

    frame = load_orbit_sensitivity_dataset(dataset_path).reset_index(drop=True)
    orbit_columns = list(ORBIT_FEATURE_FAMILIES[orbit_feature_family])
    control_columns = list(CONTROL_COLUMNS)
    fixed_effect_columns = list(FIXED_EFFECT_COLUMNS)
    _require_numeric_finite(frame, orbit_columns + control_columns + [target])
    _validate_artifact_invariant_features(frame, orbit_columns + control_columns)

    feature_sets = {
        "controls_only": control_columns,
        "orbits_only": orbit_columns,
        "orbits_plus_controls": orbit_columns + control_columns,
    }
    estimator_names = tuple(str(name) for name in estimators)
    for name in estimator_names:
        if name not in {"ridge", "elastic_net"}:
            raise ValueError(f"Unsupported estimator: {name}")

    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    split_manifest: list[dict[str, Any]] = []

    for split_mode in split_modes:
        splits = build_leakage_safe_outer_splits(
            frame, split_mode=split_mode, artifact_folds=artifact_folds
        )
        for split in splits:
            split_manifest.append(
                {
                    "split_mode": split_mode,
                    "split_id": split.split_id,
                    "held_out_value": split.held_out_value,
                    "test_artifact_ids": list(split.test_artifact_ids),
                    "train_row_count": int(split.train_indices.size),
                    "test_row_count": int(split.test_indices.size),
                }
            )
            train_frame = frame.iloc[split.train_indices].reset_index(drop=True)
            test_frame = frame.iloc[split.test_indices].reset_index(drop=True)
            y_train = train_frame[target].to_numpy(dtype=np.float64)
            y_test = test_frame[target].to_numpy(dtype=np.float64)
            inner_groups = train_frame["pruning_artifact_id"].astype(str).to_numpy()

            for feature_set_name, feature_columns in feature_sets.items():
                model_columns = feature_columns + fixed_effect_columns
                x_train = train_frame[model_columns].copy()
                x_test = test_frame[model_columns].copy()
                for estimator_name in estimator_names:
                    fitted, chosen_params, inner_cv_splits = _fit_grouped_regressor(
                        estimator_name=estimator_name,
                        x_train=x_train,
                        y_train=y_train,
                        groups=inner_groups,
                        numeric_columns=feature_columns,
                        categorical_columns=fixed_effect_columns,
                    )
                    predictions = np.asarray(fitted.predict(x_test), dtype=np.float64)
                    metrics = _regression_metrics(y_test, predictions)
                    fold_rows.append(
                        {
                            "target": target,
                            "orbit_feature_family": orbit_feature_family,
                            "split_mode": split_mode,
                            "split_id": split.split_id,
                            "held_out_value": split.held_out_value,
                            "feature_set": feature_set_name,
                            "estimator": estimator_name,
                            "train_rows": int(len(train_frame)),
                            "test_rows": int(len(test_frame)),
                            "train_artifacts": int(train_frame["pruning_artifact_id"].nunique()),
                            "test_artifacts": int(test_frame["pruning_artifact_id"].nunique()),
                            "inner_cv_splits": inner_cv_splits,
                            "selected_params": json.dumps(chosen_params, sort_keys=True),
                            **metrics,
                        }
                    )
                    coefficients = _pipeline_coefficients(fitted)
                    for feature, coefficient in coefficients.items():
                        coefficient_rows.append(
                            {
                                "target": target,
                                "orbit_feature_family": orbit_feature_family,
                                "split_mode": split_mode,
                                "split_id": split.split_id,
                                "held_out_value": split.held_out_value,
                                "feature_set": feature_set_name,
                                "estimator": estimator_name,
                                "feature": feature,
                                "coefficient_transformed_scale": coefficient,
                            }
                        )
                    if permutation_repeats > 0 and test_frame["pruning_artifact_id"].nunique() >= 2:
                        permutation_rows.extend(
                            _grouped_permutation_importance(
                                fitted=fitted,
                                test_frame=test_frame,
                                model_columns=model_columns,
                                permuted_feature_columns=feature_columns,
                                target=target,
                                baseline_mae=metrics["mae"],
                                repeats=permutation_repeats,
                                random_seed=_stable_seed(
                                    random_seed,
                                    split_mode,
                                    split.split_id,
                                    feature_set_name,
                                    estimator_name,
                                ),
                                metadata={
                                    "target": target,
                                    "orbit_feature_family": orbit_feature_family,
                                    "split_mode": split_mode,
                                    "split_id": split.split_id,
                                    "held_out_value": split.held_out_value,
                                    "feature_set": feature_set_name,
                                    "estimator": estimator_name,
                                },
                            )
                        )

    if not fold_rows:
        raise ValueError("Controlled regression produced no fold results.")
    coefficient_stability_rows = _coefficient_stability(coefficient_rows)
    spearman_rows = _grouped_spearman_table(
        frame,
        feature_columns=orbit_columns,
        target=target,
        bootstrap_repeats=bootstrap_repeats,
        random_seed=random_seed,
    )

    output_dir = ensure_dir(output_dir)
    fold_metrics_path = output_dir / "fold_metrics.csv"
    coefficients_path = output_dir / "coefficients.csv"
    coefficient_stability_path = output_dir / "coefficient_stability.csv"
    permutation_importance_path = output_dir / "permutation_importance.csv"
    spearman_path = output_dir / "orbit_spearman.csv"
    summary_path = output_dir / "regression_summary.json"
    manifest_path = output_dir / "regression_manifest.json"

    _write_rows_csv(fold_metrics_path, fold_rows)
    _write_rows_csv(coefficients_path, coefficient_rows)
    _write_rows_csv(coefficient_stability_path, coefficient_stability_rows)
    _write_rows_csv(permutation_importance_path, permutation_rows)
    _write_rows_csv(spearman_path, spearman_rows)
    summary = _regression_summary(fold_rows)
    summary.update(
        {
            "schema_version": ORBIT_SENSITIVITY_REGRESSION_SCHEMA_VERSION,
            "target": target,
            "orbit_feature_family": orbit_feature_family,
            "row_count": int(len(frame)),
            "artifact_count": int(frame["pruning_artifact_id"].nunique()),
            "split_modes": list(split_modes),
            "split_manifest": split_manifest,
        }
    )
    write_json(summary_path, summary)
    manifest = {
        "schema_version": ORBIT_SENSITIVITY_REGRESSION_SCHEMA_VERSION,
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": _sha256_file(Path(dataset_path)),
        "target": target,
        "orbit_feature_family": orbit_feature_family,
        "control_columns": control_columns,
        "fixed_effect_columns": fixed_effect_columns,
        "orbit_columns": orbit_columns,
        "feature_sets": feature_sets,
        "estimators": list(estimator_names),
        "artifact_folds": int(artifact_folds),
        "permutation_repeats": int(permutation_repeats),
        "bootstrap_repeats": int(bootstrap_repeats),
        "random_seed": int(random_seed),
        "files": {
            "fold_metrics": _file_record(fold_metrics_path),
            "coefficients": _file_record(coefficients_path),
            "coefficient_stability": _file_record(coefficient_stability_path),
            "permutation_importance": _file_record(permutation_importance_path),
            "spearman": _file_record(spearman_path),
            "summary": _file_record(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    return RegressionRunResult(
        output_dir=output_dir,
        fold_metrics_path=fold_metrics_path,
        coefficients_path=coefficients_path,
        coefficient_stability_path=coefficient_stability_path,
        permutation_importance_path=permutation_importance_path,
        spearman_path=spearman_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )


def _load_training_run(run_dir: Path) -> dict[str, Any]:
    return {
        "resolved": read_json(run_dir / "resolved_config.json"),
        "metrics": read_json(run_dir / "metrics.json"),
        "pruning": read_json(run_dir / "pruning_result.json"),
    }


def _resolve_pruning_artifact_dir(run_dir: Path, artifacts: Mapping[str, Any]) -> Path:
    metrics_metadata = artifacts["metrics"].get("metadata", {})
    candidates = [
        artifacts["resolved"].get("pruned_graph_artifact_dir"),
        metrics_metadata.get("pruned_graph_artifact_dir") if isinstance(metrics_metadata, dict) else None,
        run_dir,
    ]
    for candidate in candidates:
        if candidate in {None, ""}:
            continue
        path = Path(candidate)
        if (path / "orbit_explainability" / "orbit_explainability_manifest.json").exists():
            return path
    raise FileNotFoundError(f"Could not locate orbit explainability artifact for training run {run_dir}.")


def _load_final_orbit_checkpoint(
    pruning_artifact_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, Path, str, dict[str, Any]]:
    manifest_path = pruning_artifact_dir / "orbit_explainability" / "orbit_explainability_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != ORBIT_EXPLAINABILITY_MANIFEST_VERSION:
        raise ValueError(f"Unsupported orbit manifest schema: {manifest.get('schema_version')}")
    if manifest.get("orbit_registry_version") != ORBIT_REGISTRY_VERSION:
        raise ValueError("Orbit manifest registry version does not match the active canonical registry.")
    if manifest.get("checkpoint_schema_version") != ORBIT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Orbit checkpoint schema version mismatch.")
    if manifest.get("checkpoint_snapshot_schema_version") != ORBIT_CHECKPOINT_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Orbit checkpoint snapshot schema version mismatch.")

    files = manifest.get("files", {})
    for record in files.values():
        _validate_file_record(manifest_path.parent, record)
    checkpoints_path = _record_path(manifest_path.parent, files["checkpoints_csv"])
    snapshots_path = _record_path(manifest_path.parent, files["checkpoint_snapshots_npz"])

    with checkpoints_path.open("r", encoding="utf-8", newline="") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    final_rows = [row for row in checkpoint_rows if int(row["is_final"]) == 1]
    if len(final_rows) != 1:
        raise ValueError(f"Expected exactly one final orbit checkpoint, found {len(final_rows)}.")
    final = {key: _parse_scalar(value) for key, value in final_rows[0].items()}
    checkpoint_index = int(final["checkpoint_index"])

    with np.load(snapshots_path, allow_pickle=False) as payload:
        indices = np.asarray(payload["checkpoint_indices"], dtype=np.int64)
        positions = np.flatnonzero(indices == checkpoint_index)
        if positions.size != 1:
            raise ValueError("Final checkpoint index is absent or duplicated in checkpoint NPZ.")
        position = int(positions[0])
        if int(np.asarray(payload["is_final"], dtype=np.uint8)[position]) != 1:
            raise ValueError("CSV final checkpoint is not marked final in checkpoint NPZ.")
        transition = np.asarray(payload["cumulative_transition"], dtype=np.int64)[position].copy()
        if transition.shape != (ORBIT_DIM, ORBIT_TRANSITION_COLUMNS):
            raise ValueError(f"Invalid final cumulative transition shape: {transition.shape}")
        npz_actual_rho = float(np.asarray(payload["actual_rhos"], dtype=np.float64)[position])
        if not math.isclose(npz_actual_rho, float(final["actual_rho"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Final checkpoint CSV and NPZ actual_rho values disagree.")
    return final, transition, manifest_path, _sha256_file(manifest_path), manifest


def _validate_orbit_training_identity(
    *,
    artifacts: Mapping[str, Any],
    orbit_manifest: Mapping[str, Any],
    dataset: str,
    target_rho: float,
    checkpoint: Mapping[str, Any],
) -> int:
    if str(orbit_manifest.get("dataset")) != dataset:
        raise ValueError(
            f"Orbit artifact dataset {orbit_manifest.get('dataset')!r} does not match training dataset {dataset!r}."
        )
    manifest_rho = float(orbit_manifest.get("target_rho"))
    if not math.isclose(manifest_rho, target_rho, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Orbit target rho {manifest_rho} does not match training target rho {target_rho}."
        )
    manifest_event_count = int(orbit_manifest.get("event_count"))
    if manifest_event_count != int(checkpoint["event_count"]):
        raise ValueError("Orbit manifest event count does not match the final checkpoint.")
    if manifest_event_count != int(checkpoint["removed_edge_count"]):
        raise ValueError("Sequential exact RelShift event count must equal removed-edge count.")
    pruning_seed = int(orbit_manifest.get("pruning_seed"))
    metrics_metadata = artifacts["metrics"].get("metadata", {})
    if isinstance(metrics_metadata, dict) and metrics_metadata.get("pruning_seed") is not None:
        if int(metrics_metadata["pruning_seed"]) != pruning_seed:
            raise ValueError("Training metrics pruning seed does not match orbit artifact pruning seed.")
    return pruning_seed


def _extract_target_rho(artifacts: Mapping[str, Any]) -> float:
    pruning = artifacts["resolved"].get("pruning", {})
    if pruning.get("rho") is None:
        raise ValueError("RelShift training artifact is missing pruning.rho.")
    rho = float(pruning["rho"])
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError(f"Invalid target rho: {rho}")
    return rho


def _extract_structural_controls(
    artifacts: Mapping[str, Any], *, require_full: bool
) -> dict[str, float]:
    metrics = artifacts["metrics"]
    metadata = metrics.get("metadata", {})
    controls = metadata.get("structural_controls", {}) if isinstance(metadata, dict) else {}
    result: dict[str, float] = {}
    for key in _REQUIRED_STRUCTURAL_CONTROLS:
        if key not in controls:
            if require_full:
                raise KeyError(
                    f"Training artifact lacks Phase-2 structural control '{key}'. Rerun training with Step-4 code."
                )
            result[key] = 0.0
        else:
            value = float(controls[key])
            if not math.isfinite(value):
                raise ValueError(f"Structural control {key} must be finite.")
            result[key] = value
    return result


def _append_orbit_features(
    row: dict[str, Any], checkpoint: Mapping[str, Any], transition: np.ndarray
) -> None:
    off_diagonal = transition[:, :ORBIT_DIM].copy()
    diagonal = np.diag(off_diagonal).copy()
    np.fill_diagonal(off_diagonal, 0)
    transition_out = off_diagonal.sum(axis=1, dtype=np.int64) + transition[:, ORBIT_DIM]
    transition_in = off_diagonal.sum(axis=0, dtype=np.int64)
    destroyed = transition[:, ORBIT_DIM]
    for orbit_id in range(ORBIT_DIM):
        for suffix in (
            "initial_total",
            "current_total",
            "raw_signed_delta",
            "raw_absolute_delta",
            "raw_relative_absolute_delta",
            "raw_mean_absolute_delta",
            "raw_max_absolute_delta",
            "raw_changed_node_count",
            "initial_nonzero_node_count",
            "current_nonzero_node_count",
            "standardized_signed_delta",
            "standardized_absolute_delta",
            "standardized_mean_absolute_delta",
            "standardized_max_absolute_delta",
            "standardized_l2_delta",
            "cumulative_event_net",
            "cumulative_event_absolute_delta",
        ):
            key = f"orbit_{orbit_id}_{suffix}"
            row[key] = checkpoint[key]
        row[f"orbit_{orbit_id}_transition_out"] = int(transition_out[orbit_id])
        row[f"orbit_{orbit_id}_transition_in"] = int(transition_in[orbit_id])
        row[f"orbit_{orbit_id}_transition_destroyed"] = int(destroyed[orbit_id])
        row[f"orbit_{orbit_id}_transition_self"] = int(diagonal[orbit_id])

    row["transition_total_incidence"] = int(transition.sum())
    row["transition_total_destroyed"] = int(destroyed.sum())
    row["transition_total_changed"] = int(transition_out.sum())
    row["transition_total_self"] = int(diagonal.sum())
    for group_name, orbit_ids in ANALYSIS_ORBIT_GROUPS.items():
        row[f"group_{group_name}_standardized_absolute_delta"] = float(
            sum(float(row[f"orbit_{orbit_id}_standardized_absolute_delta"]) for orbit_id in orbit_ids)
        )
        row[f"group_{group_name}_raw_absolute_delta"] = int(
            sum(int(row[f"orbit_{orbit_id}_raw_absolute_delta"]) for orbit_id in orbit_ids)
        )
        row[f"group_{group_name}_transition_destroyed"] = int(
            sum(int(row[f"orbit_{orbit_id}_transition_destroyed"]) for orbit_id in orbit_ids)
        )


def _pruning_artifact_id(
    path: Path,
    *,
    dataset: str,
    checkpoint: Mapping[str, Any],
    cumulative_transition: np.ndarray,
) -> str:
    """Return a path-independent identity for one structural pruning artifact."""

    digest = hashlib.sha256()
    digest.update(str(dataset).encode("utf-8"))
    for name in ("pruned_edges.pt", "removed_edges.pt"):
        candidate = path / name
        if not candidate.exists():
            raise FileNotFoundError(f"Missing structural identity tensor: {candidate}")
        tensor = torch.load(candidate, map_location="cpu", weights_only=True)
        array = np.asarray(tensor.long().contiguous().cpu().numpy(), dtype=np.int64)
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    for orbit_id in range(ORBIT_DIM):
        for suffix in (
            "raw_signed_delta",
            "raw_absolute_delta",
            "standardized_absolute_delta",
        ):
            digest.update(
                np.asarray([float(checkpoint[f"orbit_{orbit_id}_{suffix}"])], dtype=np.float64).tobytes()
            )
    digest.update(np.asarray(cumulative_transition, dtype=np.int64).tobytes())
    return digest.hexdigest()[:24]


def _reduction_band(target_rho: float, achieved: float) -> str:
    return f"target_{target_rho:.6f}_achieved_{achieved:.6f}"


def _validate_sensitivity_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    _validate_sensitivity_frame(frame)
    if frame["training_pair_id"].duplicated().any():
        raise ValueError("Sensitivity dataset contains duplicate training_pair_id values.")


def _validate_sensitivity_frame(frame: pd.DataFrame) -> None:
    required = set(IDENTITY_COLUMNS) | set(TARGET_COLUMNS) | set(CONTROL_COLUMNS)
    for columns in ORBIT_FEATURE_FAMILIES.values():
        required.update(columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Sensitivity dataset is missing required columns: {missing}")
    if frame["pruning_artifact_id"].isna().any():
        raise ValueError("pruning_artifact_id cannot be missing.")
    _require_numeric_finite(
        frame,
        list(TARGET_COLUMNS)
        + list(CONTROL_COLUMNS)
        + [column for columns in ORBIT_FEATURE_FAMILIES.values() for column in columns],
    )


def _validate_artifact_invariant_features(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    grouped = frame.groupby("pruning_artifact_id", sort=False)
    for column in columns:
        spread = grouped[column].agg(lambda values: float(np.max(values) - np.min(values)))
        bad = spread[spread > 1e-12]
        if not bad.empty:
            raise ValueError(
                f"Structural feature {column} varies across training rows for the same pruning artifact: "
                f"{bad.index.tolist()}"
            )


def _fit_grouped_regressor(
    *,
    estimator_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[Pipeline, dict[str, Any], int]:
    unique_groups = np.unique(groups)
    if estimator_name == "ridge":
        model = Ridge()
        grid = {"model__alpha": [0.1, 1.0, 10.0]}
        defaults = {"model__alpha": 1.0}
    elif estimator_name == "elastic_net":
        model = ElasticNet(max_iter=20000, random_state=0)
        grid = {
            "model__alpha": [0.001, 0.01, 0.1],
            "model__l1_ratio": [0.2, 0.8],
        }
        defaults = {"model__alpha": 0.01, "model__l1_ratio": 0.5}
    else:
        raise ValueError(estimator_name)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(numeric_columns)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_columns),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    inner_splits = min(3, len(unique_groups))
    if inner_splits >= 2 and len(y_train) >= inner_splits:
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="neg_mean_absolute_error",
            cv=GroupKFold(n_splits=inner_splits),
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train, groups=groups)
        return search.best_estimator_, dict(search.best_params_), inner_splits
    pipeline.set_params(**defaults)
    pipeline.fit(x_train, y_train)
    return pipeline, defaults, 0


def _regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    mae = float(mean_absolute_error(y_true, prediction))
    rmse = float(math.sqrt(mean_squared_error(y_true, prediction)))
    if len(y_true) >= 2 and float(np.var(y_true)) > 0.0:
        r2 = float(r2_score(y_true, prediction))
        spearman = _safe_spearman(y_true, prediction)
    else:
        r2 = None
        spearman = None
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman}


def _pipeline_coefficients(fitted: Pipeline) -> dict[str, float]:
    coefficients = np.asarray(fitted.named_steps["model"].coef_, dtype=np.float64).reshape(-1)
    feature_names = list(fitted.named_steps["preprocess"].get_feature_names_out())
    if coefficients.size != len(feature_names):
        raise ValueError("Regressor coefficient dimension does not match transformed features.")
    return {feature: float(value) for feature, value in zip(feature_names, coefficients, strict=True)}


def _grouped_permutation_importance(
    *,
    fitted: Pipeline,
    test_frame: pd.DataFrame,
    model_columns: Sequence[str],
    permuted_feature_columns: Sequence[str],
    target: str,
    baseline_mae: float,
    repeats: int,
    random_seed: int,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    artifact_ids = sorted(test_frame["pruning_artifact_id"].astype(str).unique())
    representatives = test_frame.groupby("pruning_artifact_id", sort=False)[list(permuted_feature_columns)].first()
    x_base = test_frame[list(model_columns)].copy()
    y = test_frame[target].to_numpy(dtype=np.float64)
    output: list[dict[str, Any]] = []
    for feature in permuted_feature_columns:
        increases: list[float] = []
        for _ in range(repeats):
            permuted_ids = list(rng.permutation(artifact_ids))
            mapping = {
                target_id: float(representatives.loc[source_id, feature])
                for target_id, source_id in zip(artifact_ids, permuted_ids, strict=True)
            }
            x_permuted = x_base.copy()
            x_permuted[feature] = test_frame["pruning_artifact_id"].astype(str).map(mapping).to_numpy()
            prediction = np.asarray(fitted.predict(x_permuted), dtype=np.float64)
            increases.append(float(mean_absolute_error(y, prediction) - baseline_mae))
        output.append(
            {
                **metadata,
                "feature": feature,
                "permutation_repeats": repeats,
                "mae_increase_mean": float(np.mean(increases)),
                "mae_increase_std": float(np.std(increases, ddof=0)),
            }
        )
    return output


def _coefficient_stability(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    grouping = ["target", "orbit_feature_family", "split_mode", "feature_set", "estimator", "feature"]
    for key, group in frame.groupby(grouping, sort=True):
        values = group["coefficient_transformed_scale"].to_numpy(dtype=np.float64)
        nonzero = values[np.abs(values) > 1e-12]
        sign_consistency = 0.0
        if nonzero.size:
            sign_consistency = float(max(np.mean(nonzero > 0.0), np.mean(nonzero < 0.0)))
        output.append(
            {
                **dict(zip(grouping, key, strict=True)),
                "fold_count": int(values.size),
                "coefficient_mean": float(np.mean(values)),
                "coefficient_std": float(np.std(values, ddof=0)),
                "coefficient_median": float(np.median(values)),
                "nonzero_fraction": float(np.mean(np.abs(values) > 1e-12)),
                "sign_consistency": sign_consistency,
            }
        )
    return output


def _grouped_spearman_table(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target: str,
    bootstrap_repeats: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    artifact_ids = sorted(frame["pruning_artifact_id"].astype(str).unique())
    output: list[dict[str, Any]] = []
    for feature in feature_columns:
        correlation = _safe_spearman(
            frame[feature].to_numpy(dtype=np.float64),
            frame[target].to_numpy(dtype=np.float64),
        )
        bootstrap: list[float] = []
        for _ in range(bootstrap_repeats):
            sampled_ids = rng.choice(artifact_ids, size=len(artifact_ids), replace=True)
            sampled_parts = [frame[frame["pruning_artifact_id"].astype(str) == artifact_id] for artifact_id in sampled_ids]
            sampled = pd.concat(sampled_parts, ignore_index=True)
            value = _safe_spearman(
                sampled[feature].to_numpy(dtype=np.float64),
                sampled[target].to_numpy(dtype=np.float64),
            )
            bootstrap.append(value)
        if bootstrap:
            lower, upper = np.quantile(np.asarray(bootstrap), [0.025, 0.975])
            bootstrap_std = float(np.std(bootstrap, ddof=0))
        else:
            lower = upper = correlation
            bootstrap_std = 0.0
        output.append(
            {
                "target": target,
                "feature": feature,
                "spearman": correlation,
                "bootstrap_repeats": bootstrap_repeats,
                "bootstrap_valid": len(bootstrap),
                "bootstrap_std": bootstrap_std,
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
            }
        )
    return output


def _regression_summary(fold_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(fold_rows)
    aggregates: list[dict[str, Any]] = []
    grouping = ["split_mode", "feature_set", "estimator"]
    for key, group in frame.groupby(grouping, sort=True):
        aggregates.append(
            {
                **dict(zip(grouping, key, strict=True)),
                "fold_count": int(len(group)),
                "mae_mean": float(group["mae"].mean()),
                "mae_std": float(group["mae"].std(ddof=0)),
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_std": float(group["rmse"].std(ddof=0)),
            }
        )
    incremental: list[dict[str, Any]] = []
    for (split_mode, estimator), group in frame.groupby(["split_mode", "estimator"], sort=True):
        means = group.groupby("feature_set")["mae"].mean()
        if "controls_only" in means and "orbits_plus_controls" in means:
            incremental.append(
                {
                    "split_mode": split_mode,
                    "estimator": estimator,
                    "controls_only_mae": float(means["controls_only"]),
                    "orbits_plus_controls_mae": float(means["orbits_plus_controls"]),
                    "mae_improvement": float(
                        means["controls_only"] - means["orbits_plus_controls"]
                    ),
                }
            )
    return {"aggregate_metrics": aggregates, "incremental_orbit_value": incremental}


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or right.size != left.size:
        return 0.0
    if float(np.var(left)) == 0.0 or float(np.var(right)) == 0.0:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def _require_numeric_finite(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"Missing required numeric column: {column}")
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Column {column} contains missing or non-finite values.")


def _validate_file_record(base_dir: Path, record: Mapping[str, Any]) -> None:
    path = _record_path(base_dir, record)
    if not path.exists():
        raise FileNotFoundError(path)
    if int(record["size_bytes"]) != int(path.stat().st_size):
        raise ValueError(f"Artifact size mismatch for {path}")
    if str(record["sha256"]) != _sha256_file(path):
        raise ValueError(f"Artifact SHA-256 mismatch for {path}")


def _record_path(base_dir: Path, record: Mapping[str, Any]) -> Path:
    recorded = Path(str(record["path"]))
    local = base_dir / recorded.name
    if local.exists():
        return local
    if recorded.exists():
        return recorded
    return local


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped == "":
        return ""
    if stripped.startswith("[") or stripped.startswith("{"):
        return json.loads(stripped)
    try:
        integer = int(stripped)
        if str(integer) == stripped or stripped in {f"+{integer}", f"-{abs(integer)}"}:
            return integer
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        if list(row.keys()) != fieldnames:
            raise ValueError("All CSV rows must share the same deterministic field order.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join((str(base_seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


__all__ = [
    "CONTROL_COLUMNS",
    "FIXED_EFFECT_COLUMNS",
    "IDENTITY_COLUMNS",
    "ORBIT_FEATURE_FAMILIES",
    "ORBIT_SENSITIVITY_DATASET_SCHEMA_VERSION",
    "ORBIT_SENSITIVITY_REGRESSION_SCHEMA_VERSION",
    "TARGET_COLUMNS",
    "OrbitSensitivityDatasetResult",
    "OuterSplit",
    "RegressionRunResult",
    "build_leakage_safe_outer_splits",
    "build_orbit_sensitivity_dataset",
    "load_orbit_sensitivity_dataset",
    "run_controlled_orbit_regression",
]
