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
    parser = argparse.ArgumentParser(description="Build numeric runtime comparison tables for RelShift, DSpar, and LSP.")
    parser.add_argument("--master-results", required=True)
    parser.add_argument("--common-grid", required=True)
    parser.add_argument("--relshift-frontier", required=True)
    parser.add_argument("--dspar-frontier", required=True)
    parser.add_argument("--lsp-frontier", required=True)
    parser.add_argument("--output-root", default=str(ROOT / "results" / "analysis" / "runtime_comparison"))
    args = parser.parse_args()

    output = analyze_runtime_comparison(
        master_results=args.master_results,
        common_grid=args.common_grid,
        relshift_frontier=args.relshift_frontier,
        dspar_frontier=args.dspar_frontier,
        lsp_frontier=args.lsp_frontier,
        output_root=args.output_root,
    )
    print(json.dumps(output, indent=2))


def analyze_runtime_comparison(
    *,
    master_results: str | Path,
    common_grid: str | Path,
    relshift_frontier: str | Path,
    dspar_frontier: str | Path,
    lsp_frontier: str | Path,
    output_root: str | Path,
) -> dict[str, str]:
    from structure_faithful_gnn.utils.io import ensure_dir

    output_root = ensure_dir(output_root)
    master_df = pd.read_csv(master_results)
    grid_df = pd.read_csv(common_grid)
    frontier_df = _load_frontiers(relshift_frontier, dspar_frontier, lsp_frontier)

    frontier_summary = _frontier_pruning_runtime_summary(frontier_df)
    matched_pruning = _matched_pruning_runtime_by_target(grid_df, frontier_df)
    matched_pruning_summary = _matched_pruning_runtime_summary(matched_pruning)
    training_by_run = _training_runtime_by_run(master_df)
    training_summary = _training_runtime_summary(training_by_run)
    end_to_end_summary = _end_to_end_runtime_summary(training_by_run)
    pairwise_summary = _pairwise_runtime_summary(training_by_run)

    outputs = {
        "frontier_pruning_runtime_summary_csv": output_root / "frontier_pruning_runtime_summary.csv",
        "matched_pruning_runtime_by_target_csv": output_root / "matched_pruning_runtime_by_target.csv",
        "matched_pruning_runtime_summary_csv": output_root / "matched_pruning_runtime_summary.csv",
        "training_runtime_by_run_csv": output_root / "training_runtime_by_run.csv",
        "training_runtime_summary_csv": output_root / "training_runtime_summary.csv",
        "end_to_end_runtime_summary_csv": output_root / "end_to_end_runtime_summary.csv",
        "runtime_pairwise_summary_csv": output_root / "runtime_pairwise_summary.csv",
    }
    frontier_summary.to_csv(outputs["frontier_pruning_runtime_summary_csv"], index=False)
    matched_pruning.to_csv(outputs["matched_pruning_runtime_by_target_csv"], index=False)
    matched_pruning_summary.to_csv(outputs["matched_pruning_runtime_summary_csv"], index=False)
    training_by_run.to_csv(outputs["training_runtime_by_run_csv"], index=False)
    training_summary.to_csv(outputs["training_runtime_summary_csv"], index=False)
    end_to_end_summary.to_csv(outputs["end_to_end_runtime_summary_csv"], index=False)
    pairwise_summary.to_csv(outputs["runtime_pairwise_summary_csv"], index=False)
    return {key: str(value) for key, value in outputs.items()}


def _load_frontiers(
    relshift_frontier: str | Path,
    dspar_frontier: str | Path,
    lsp_frontier: str | Path,
) -> pd.DataFrame:
    frames = []
    for method, path in [
        ("relshift", relshift_frontier),
        ("dspar", dspar_frontier),
        ("lsp", lsp_frontier),
    ]:
        frame = pd.read_csv(path).copy()
        frame["method"] = method
        frame["normalized_run_dir"] = frame["run_dir"].map(_normalize_path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _frontier_pruning_runtime_summary(frontier_df: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["dataset", "method"]
    rows = []
    for key, group in frontier_df.groupby(group_keys, dropna=False):
        dataset, method = key
        runtimes = group["pruning_runtime_sec"].astype(float)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "num_pruned_artifacts": int(len(group)),
                "sum_pruning_runtime_sec": float(runtimes.sum()),
                "mean_pruning_runtime_sec": float(runtimes.mean()),
                "median_pruning_runtime_sec": float(runtimes.median()),
                "min_pruning_runtime_sec": float(runtimes.min()),
                "max_pruning_runtime_sec": float(runtimes.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def _matched_pruning_runtime_by_target(grid_df: pd.DataFrame, frontier_df: pd.DataFrame) -> pd.DataFrame:
    frontier_by_dir = frontier_df.set_index("normalized_run_dir", drop=False)
    rows: list[dict[str, object]] = []
    for _, grid_row in grid_df.iterrows():
        for method in ["relshift", "dspar", "lsp"]:
            run_dir = _normalize_path(grid_row[f"{method}_run_dir"])
            frontier_row = frontier_by_dir.loc[run_dir]
            if isinstance(frontier_row, pd.DataFrame):
                frontier_row = frontier_row.iloc[0]
            rows.append(
                {
                    "dataset": str(grid_row["dataset"]),
                    "status": str(grid_row["status"]),
                    "target_edge_reduction": float(grid_row["target_edge_reduction"]),
                    "method": method,
                    "achieved_edge_reduction": float(grid_row[f"{method}_achieved_edge_reduction"]),
                    "abs_gap": float(grid_row[f"{method}_abs_gap"]),
                    "pruning_runtime_sec": float(frontier_row["pruning_runtime_sec"]),
                    "lsp_variant": None if method != "lsp" else grid_row.get("lsp_variant"),
                    "run_dir": run_dir,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "target_edge_reduction", "method"]).reset_index(drop=True)


def _matched_pruning_runtime_summary(matched_pruning: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in matched_pruning.groupby(["dataset", "status", "method"], dropna=False):
        dataset, status, method = key
        runtimes = group["pruning_runtime_sec"].astype(float)
        rows.append(
            {
                "dataset": dataset,
                "status": status,
                "method": method,
                "num_targets": int(len(group)),
                "mean_pruning_runtime_sec": float(runtimes.mean()),
                "median_pruning_runtime_sec": float(runtimes.median()),
                "sum_pruning_runtime_sec": float(runtimes.sum()),
                "min_pruning_runtime_sec": float(runtimes.min()),
                "max_pruning_runtime_sec": float(runtimes.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "status", "method"]).reset_index(drop=True)


def _training_runtime_by_run(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()
    df["pruning_runtime_sec"] = df["runtime_sec"].astype(float)
    df["train_sec"] = df["train_sec"].astype(float)
    df["infer_sec"] = df["infer_sec"].astype(float)
    df["training_plus_infer_sec"] = df["train_sec"] + df["infer_sec"]
    df["single_run_total_sec"] = df["pruning_runtime_sec"] + df["training_plus_infer_sec"]
    df["normalized_pruned_graph_run_dir"] = df["pruned_graph_run_dir"].map(_normalize_path)
    reuse_count = df.groupby("normalized_pruned_graph_run_dir", dropna=False)["run_dir"].transform("count").astype(float)
    df["artifact_reuse_count"] = reuse_count
    df["amortized_pruning_runtime_sec"] = df["pruning_runtime_sec"] / reuse_count.clip(lower=1.0)
    df["amortized_total_sec"] = df["amortized_pruning_runtime_sec"] + df["training_plus_infer_sec"]
    columns = [
        "dataset",
        "model",
        "method",
        "seed",
        "achieved_edge_reduction",
        "pruning_runtime_sec",
        "train_sec",
        "infer_sec",
        "training_plus_infer_sec",
        "single_run_total_sec",
        "artifact_reuse_count",
        "amortized_pruning_runtime_sec",
        "amortized_total_sec",
        "pruned_graph_run_dir",
        "run_dir",
    ]
    return df[columns].sort_values(["dataset", "model", "method", "seed", "achieved_edge_reduction"]).reset_index(drop=True)


def _training_runtime_summary(training_by_run: pd.DataFrame) -> pd.DataFrame:
    return _summarize_runtime(
        training_by_run,
        group_keys=["dataset", "model", "method"],
        metrics=["train_sec", "infer_sec", "training_plus_infer_sec"],
    )


def _end_to_end_runtime_summary(training_by_run: pd.DataFrame) -> pd.DataFrame:
    return _summarize_runtime(
        training_by_run,
        group_keys=["dataset", "model", "method"],
        metrics=[
            "pruning_runtime_sec",
            "single_run_total_sec",
            "amortized_pruning_runtime_sec",
            "amortized_total_sec",
        ],
    )


def _pairwise_runtime_summary(training_by_run: pd.DataFrame) -> pd.DataFrame:
    summary = _end_to_end_runtime_summary(training_by_run)
    rows: list[dict[str, object]] = []
    for (dataset, model), group in summary.groupby(["dataset", "model"], dropna=False):
        by_method = {str(row["method"]): row for _, row in group.iterrows()}
        if not {"relshift", "dspar", "lsp"}.issubset(by_method):
            continue
        row: dict[str, object] = {"dataset": dataset, "model": model}
        for metric in [
            "mean_pruning_runtime_sec",
            "mean_single_run_total_sec",
            "mean_amortized_total_sec",
        ]:
            rel = float(by_method["relshift"][metric])
            dspar = float(by_method["dspar"][metric])
            lsp = float(by_method["lsp"][metric])
            row[f"relshift_{metric}"] = rel
            row[f"dspar_{metric}"] = dspar
            row[f"lsp_{metric}"] = lsp
            row[f"relshift_vs_dspar_{metric}_delta"] = rel - dspar
            row[f"relshift_vs_lsp_{metric}_delta"] = rel - lsp
            row[f"relshift_vs_dspar_{metric}_ratio"] = None if dspar == 0.0 else rel / dspar
            row[f"relshift_vs_lsp_{metric}_ratio"] = None if lsp == 0.0 else rel / lsp
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def _summarize_runtime(df: pd.DataFrame, *, group_keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in df.groupby(group_keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {name: value for name, value in zip(group_keys, key)}
        row["num_runs"] = int(len(group))
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"median_{metric}"] = float(values.median())
            row[f"std_{metric}"] = None if len(values) <= 1 else float(values.std(ddof=1))
            row[f"min_{metric}"] = float(values.min())
            row[f"max_{metric}"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)


def _normalize_path(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(Path(str(value)).resolve())


if __name__ == "__main__":
    main()
