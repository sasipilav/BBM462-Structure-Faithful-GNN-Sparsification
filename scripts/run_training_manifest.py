from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiple models and seeds on a manifest of pruned-graph artifacts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--include-status", nargs="+", default=["main_comparable"])
    parser.add_argument("--include-methods", nargs="+", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "results" / "frontier_training"))
    parser.add_argument("--dense-output-root", default=None)
    args = parser.parse_args()

    output = run_training_manifest(
        manifest=args.manifest,
        models=args.models,
        seeds=args.seeds,
        include_status=args.include_status,
        include_methods=args.include_methods,
        output_root=args.output_root,
        dense_output_root=args.dense_output_root,
    )
    print(json.dumps(output, indent=2))


def run_training_manifest(
    *,
    manifest: str | Path,
    models: list[str],
    seeds: list[int],
    include_status: list[str],
    include_methods: list[str] | None = None,
    output_root: str | Path,
    dense_output_root: str | Path | None,
) -> dict[str, str]:
    from structure_faithful_gnn.analysis.artifacts import write_rows_csv, write_rows_json
    from structure_faithful_gnn.config import load_dataset_config, load_model_config
    from structure_faithful_gnn.experiments.runner import run_dense_experiment, run_training_on_pruned_artifact
    from structure_faithful_gnn.utils.io import ensure_dir

    output_root = ensure_dir(output_root)
    dense_output_root = ensure_dir(dense_output_root or output_root)
    manifest_df = pd.read_csv(manifest)
    manifest_df = manifest_df[manifest_df["status"].isin(include_status)].copy()
    if include_methods:
        include_methods_normalized = {method.lower() for method in include_methods}
        manifest_df = manifest_df[
            manifest_df["method"].astype(str).str.lower().isin(include_methods_normalized)
        ].copy()
    if manifest_df.empty:
        raise ValueError(
            f"No manifest rows match include_status={include_status}"
            f"{'' if not include_methods else f' and include_methods={include_methods}'}"
        )

    _preflight_weighted_graphsage_dependencies(
        manifest_df=manifest_df,
        model_paths=models,
    )

    dense_done: set[tuple[str, str, int]] = set()
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for _, entry in manifest_df.iterrows():
            dataset_path = str(entry["dataset_config"])
            dataset_config = load_dataset_config(dataset_path)
            for model_path in models:
                model_config = load_model_config(model_path)
                dense_key = (dataset_config.name, model_config.name, int(seed))
                if dense_key not in dense_done:
                    run_dense_experiment(
                        dataset_config,
                        model_config,
                        seed=int(seed),
                        output_root=dense_output_root,
                    )
                    dense_done.add(dense_key)
                outcome = run_training_on_pruned_artifact(
                    dataset_config,
                    model_config,
                    artifact_dir=str(entry["artifact_run_dir"]),
                    seed=int(seed),
                    output_root=output_root,
                    run_name=None if "run_name" not in entry or pd.isna(entry["run_name"]) else str(entry["run_name"]),
                )
                rows.append(
                    {
                        "dataset": dataset_config.name,
                        "model": model_config.name,
                        "seed": int(seed),
                        "method": str(entry["method"]),
                        "status": str(entry["status"]),
                        "target_edge_reduction": float(entry["target_edge_reduction"]),
                        "achieved_edge_reduction": float(entry["achieved_edge_reduction"]),
                        "artifact_run_dir": str(entry["artifact_run_dir"]),
                        "run_name": None if "run_name" not in entry or pd.isna(entry["run_name"]) else str(entry["run_name"]),
                        "training_run_dir": str(Path(outcome.run_dir).resolve()),
                    }
                )

    summary_csv = output_root / "training_manifest_summary.csv"
    summary_json = output_root / "training_manifest_summary.json"
    write_rows_csv(summary_csv, rows)
    write_rows_json(summary_json, rows)
    return {
        "training_manifest_summary_csv": str(summary_csv),
        "training_manifest_summary_json": str(summary_json),
    }


def _preflight_weighted_graphsage_dependencies(
    *,
    manifest_df: pd.DataFrame,
    model_paths: list[str],
) -> None:
    from structure_faithful_gnn.config import load_model_config

    graphsage_requested = any(load_model_config(path).name.lower() == "graphsage" for path in model_paths)
    if not graphsage_requested:
        return

    weighted_artifact_rows = []
    for artifact_run_dir in manifest_df["artifact_run_dir"].astype(str).unique():
        if (Path(artifact_run_dir) / "edge_weight.pt").exists():
            weighted_artifact_rows.append(artifact_run_dir)
    if not weighted_artifact_rows:
        return

    try:
        import torch_scatter  # noqa: F401
        import torch_sparse  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "This training manifest includes weighted pruned-graph artifacts and GraphSAGE, "
            "so both torch_scatter and torch_sparse must be installed first. "
            "Rerun the notebook dependency-install cell, then rerun the training-manifest cell."
        ) from exc


if __name__ == "__main__":
    main()
