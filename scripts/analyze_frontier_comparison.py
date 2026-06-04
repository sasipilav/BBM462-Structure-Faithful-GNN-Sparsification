from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dense frontier comparisons across RelShift, DSpar, and LSP.")
    parser.add_argument("--master-results", required=True)
    parser.add_argument("--common-grid", required=True)
    parser.add_argument("--output-root", default=str(ROOT / "results" / "analysis" / "frontier"))
    args = parser.parse_args()

    output = analyze_frontier_comparison(
        master_results=args.master_results,
        common_grid=args.common_grid,
        output_root=args.output_root,
    )
    print(json.dumps(output, indent=2))


def analyze_frontier_comparison(
    *,
    master_results: str | Path,
    common_grid: str | Path,
    output_root: str | Path,
) -> dict[str, str]:
    from structure_faithful_gnn.utils.io import ensure_dir

    output_root = ensure_dir(output_root)
    master_df = pd.read_csv(master_results)
    grid_df = pd.read_csv(common_grid)

    frontier_summary = _frontier_summary(master_df)
    matched = _matched_three_way(master_df, grid_df)
    seed_level_summary = _seed_level_summary(matched)
    pairwise_summary = _pairwise_summary(matched, seed_level_summary)
    unmatched = grid_df[grid_df["status"] == "unmatched"].copy()
    coverage = _coverage_summary(grid_df)

    frontier_summary_path = output_root / "frontier_summary.csv"
    matched_path = output_root / "matched_three_way_comparison.csv"
    seed_level_summary_path = output_root / "seed_level_summary.csv"
    unmatched_path = output_root / "unmatched_targets.csv"
    coverage_path = output_root / "method_coverage_summary.csv"
    pairwise_summary_path = output_root / "pairwise_summary.csv"
    frontier_summary.to_csv(frontier_summary_path, index=False)
    matched.to_csv(matched_path, index=False)
    seed_level_summary.to_csv(seed_level_summary_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    pairwise_summary.to_csv(pairwise_summary_path, index=False)

    figures = {
        "accuracy_vs_achieved_reduction": output_root / "accuracy_vs_achieved_reduction.png",
        "macro_f1_vs_achieved_reduction": output_root / "macro_f1_vs_achieved_reduction.png",
        "largest_component_ratio_vs_achieved_reduction": output_root / "largest_component_ratio_vs_achieved_reduction.png",
        "num_components_vs_achieved_reduction": output_root / "num_components_vs_achieved_reduction.png",
        "coverage_by_method": output_root / "coverage_by_method.png",
        "match_gap_by_target": output_root / "match_gap_by_target.png",
    }
    _plot_metric(master_df, grid_df, metric="accuracy", ylabel="Accuracy", output_path=figures["accuracy_vs_achieved_reduction"])
    _plot_metric(master_df, grid_df, metric="macro_f1", ylabel="Macro-F1", output_path=figures["macro_f1_vs_achieved_reduction"])
    _plot_metric(
        master_df,
        grid_df,
        metric="largest_component_ratio",
        ylabel="Largest Component Ratio",
        output_path=figures["largest_component_ratio_vs_achieved_reduction"],
    )
    _plot_metric(master_df, grid_df, metric="num_components", ylabel="Number of Components", output_path=figures["num_components_vs_achieved_reduction"])
    _plot_coverage(coverage, output_path=figures["coverage_by_method"])
    _plot_match_gaps(grid_df, output_path=figures["match_gap_by_target"])

    return {
        "frontier_summary_csv": str(frontier_summary_path),
        "matched_three_way_comparison_csv": str(matched_path),
        "seed_level_summary_csv": str(seed_level_summary_path),
        "unmatched_targets_csv": str(unmatched_path),
        "method_coverage_summary_csv": str(coverage_path),
        "pairwise_summary_csv": str(pairwise_summary_path),
        **{name: str(path) for name, path in figures.items()},
    }


def _frontier_summary(master_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, model, method), group in master_df.groupby(["dataset", "model", "method"]):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "num_points": int(len(group)),
                "min_achieved_edge_reduction": float(group["achieved_edge_reduction"].min()),
                "max_achieved_edge_reduction": float(group["achieved_edge_reduction"].max()),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_macro_f1": float(group["macro_f1"].mean()),
                "mean_largest_component_ratio": float(group["largest_component_ratio"].mean()),
                "mean_num_components": float(group["num_components"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "model", "method"]).reset_index(drop=True)


def _matched_three_way(master_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_grid = grid_df[grid_df["status"] == "main_comparable"].copy()
    for _, grid_row in selected_grid.iterrows():
        dataset = str(grid_row["dataset"])
        for model in sorted(master_df[master_df["dataset"] == dataset]["model"].unique()):
            model_df = master_df[(master_df["dataset"] == dataset) & (master_df["model"] == model)]
            for seed in sorted(model_df["seed"].unique()):
                rel = _pick_training_row(model_df, seed=seed, method="relshift", artifact_dir=str(grid_row["relshift_run_dir"]))
                dspar = _pick_training_row(model_df, seed=seed, method="dspar", artifact_dir=str(grid_row["dspar_run_dir"]))
                lsp = _pick_training_row(model_df, seed=seed, method="lsp", artifact_dir=str(grid_row["lsp_run_dir"]))
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": int(seed),
                        "status": str(grid_row["status"]),
                        "target_edge_reduction": float(grid_row["target_edge_reduction"]),
                        "relshift_achieved_edge_reduction": float(rel["achieved_edge_reduction"]),
                        "dspar_achieved_edge_reduction": float(dspar["achieved_edge_reduction"]),
                        "lsp_achieved_edge_reduction": float(lsp["achieved_edge_reduction"]),
                        "relshift_accuracy": float(rel["accuracy"]),
                        "dspar_accuracy": float(dspar["accuracy"]),
                        "lsp_accuracy": float(lsp["accuracy"]),
                        "relshift_macro_f1": float(rel["macro_f1"]),
                        "dspar_macro_f1": float(dspar["macro_f1"]),
                        "lsp_macro_f1": float(lsp["macro_f1"]),
                        "relshift_largest_component_ratio": float(rel["largest_component_ratio"]),
                        "dspar_largest_component_ratio": float(dspar["largest_component_ratio"]),
                        "lsp_largest_component_ratio": float(lsp["largest_component_ratio"]),
                        "relshift_num_components": int(rel["num_components"]),
                        "dspar_num_components": int(dspar["num_components"]),
                        "lsp_num_components": int(lsp["num_components"]),
                        "relshift_vs_dspar_accuracy_delta": float(rel["accuracy"]) - float(dspar["accuracy"]),
                        "relshift_vs_lsp_accuracy_delta": float(rel["accuracy"]) - float(lsp["accuracy"]),
                        "relshift_vs_dspar_macro_f1_delta": float(rel["macro_f1"]) - float(dspar["macro_f1"]),
                        "relshift_vs_lsp_macro_f1_delta": float(rel["macro_f1"]) - float(lsp["macro_f1"]),
                        "relshift_vs_dspar_lcc_delta": float(rel["largest_component_ratio"]) - float(dspar["largest_component_ratio"]),
                        "relshift_vs_lsp_lcc_delta": float(rel["largest_component_ratio"]) - float(lsp["largest_component_ratio"]),
                        "relshift_vs_dspar_components_delta": int(rel["num_components"]) - int(dspar["num_components"]),
                        "relshift_vs_lsp_components_delta": int(rel["num_components"]) - int(lsp["num_components"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "model", "seed", "target_edge_reduction"]).reset_index(drop=True)


def _pick_training_row(df: pd.DataFrame, *, seed: int, method: str, artifact_dir: str) -> pd.Series:
    matches = df[
        (df["seed"] == seed)
        & (df["method"] == method)
        & (df["pruned_graph_run_dir"] == str(Path(artifact_dir).resolve()))
    ]
    if matches.empty:
        raise FileNotFoundError(
            f"Missing training row for dataset/model group, method={method}, seed={seed}, pruned_graph_run_dir={artifact_dir}"
        )
    return matches.iloc[0]


def _seed_level_summary(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, model, seed), group in matched.groupby(["dataset", "model", "seed"]):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": int(seed),
                "num_targets": int(len(group)),
                "mean_relshift_accuracy": float(group["relshift_accuracy"].mean()),
                "mean_dspar_accuracy": float(group["dspar_accuracy"].mean()),
                "mean_lsp_accuracy": float(group["lsp_accuracy"].mean()),
                "mean_relshift_macro_f1": float(group["relshift_macro_f1"].mean()),
                "mean_dspar_macro_f1": float(group["dspar_macro_f1"].mean()),
                "mean_lsp_macro_f1": float(group["lsp_macro_f1"].mean()),
                "mean_relshift_largest_component_ratio": float(group["relshift_largest_component_ratio"].mean()),
                "mean_dspar_largest_component_ratio": float(group["dspar_largest_component_ratio"].mean()),
                "mean_lsp_largest_component_ratio": float(group["lsp_largest_component_ratio"].mean()),
                "mean_relshift_num_components": float(group["relshift_num_components"].mean()),
                "mean_dspar_num_components": float(group["dspar_num_components"].mean()),
                "mean_lsp_num_components": float(group["lsp_num_components"].mean()),
                "mean_relshift_vs_dspar_accuracy_delta": float(group["relshift_vs_dspar_accuracy_delta"].mean()),
                "mean_relshift_vs_lsp_accuracy_delta": float(group["relshift_vs_lsp_accuracy_delta"].mean()),
                "mean_relshift_vs_dspar_macro_f1_delta": float(group["relshift_vs_dspar_macro_f1_delta"].mean()),
                "mean_relshift_vs_lsp_macro_f1_delta": float(group["relshift_vs_lsp_macro_f1_delta"].mean()),
                "mean_relshift_vs_dspar_lcc_delta": float(group["relshift_vs_dspar_lcc_delta"].mean()),
                "mean_relshift_vs_lsp_lcc_delta": float(group["relshift_vs_lsp_lcc_delta"].mean()),
                "mean_relshift_vs_dspar_components_delta": float(group["relshift_vs_dspar_components_delta"].mean()),
                "mean_relshift_vs_lsp_components_delta": float(group["relshift_vs_lsp_components_delta"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "model", "seed"]).reset_index(drop=True)


def _pairwise_summary(matched: pd.DataFrame, seed_level_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, model), group in matched.groupby(["dataset", "model"]):
        seed_group = seed_level_summary[
            (seed_level_summary["dataset"] == dataset)
            & (seed_level_summary["model"] == model)
        ].copy()
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "num_points": int(len(group)),
                "num_targets": int(group["target_edge_reduction"].nunique()),
                "num_seeds": int(group["seed"].nunique()),
                "relshift_vs_dspar_accuracy_wins": int((group["relshift_vs_dspar_accuracy_delta"] > 0).sum()),
                "relshift_vs_lsp_accuracy_wins": int((group["relshift_vs_lsp_accuracy_delta"] > 0).sum()),
                "relshift_vs_dspar_macro_f1_wins": int((group["relshift_vs_dspar_macro_f1_delta"] > 0).sum()),
                "relshift_vs_lsp_macro_f1_wins": int((group["relshift_vs_lsp_macro_f1_delta"] > 0).sum()),
                "relshift_vs_dspar_connectivity_wins": int((group["relshift_vs_dspar_lcc_delta"] > 0).sum()),
                "relshift_vs_lsp_connectivity_wins": int((group["relshift_vs_lsp_lcc_delta"] > 0).sum()),
                "relshift_vs_dspar_component_wins": int((group["relshift_vs_dspar_components_delta"] < 0).sum()),
                "relshift_vs_lsp_component_wins": int((group["relshift_vs_lsp_components_delta"] < 0).sum()),
                "mean_relshift_accuracy": float(seed_group["mean_relshift_accuracy"].mean()),
                "std_relshift_accuracy": _safe_std(seed_group["mean_relshift_accuracy"]),
                "mean_dspar_accuracy": float(seed_group["mean_dspar_accuracy"].mean()),
                "std_dspar_accuracy": _safe_std(seed_group["mean_dspar_accuracy"]),
                "mean_lsp_accuracy": float(seed_group["mean_lsp_accuracy"].mean()),
                "std_lsp_accuracy": _safe_std(seed_group["mean_lsp_accuracy"]),
                "mean_relshift_macro_f1": float(seed_group["mean_relshift_macro_f1"].mean()),
                "std_relshift_macro_f1": _safe_std(seed_group["mean_relshift_macro_f1"]),
                "mean_dspar_macro_f1": float(seed_group["mean_dspar_macro_f1"].mean()),
                "std_dspar_macro_f1": _safe_std(seed_group["mean_dspar_macro_f1"]),
                "mean_lsp_macro_f1": float(seed_group["mean_lsp_macro_f1"].mean()),
                "std_lsp_macro_f1": _safe_std(seed_group["mean_lsp_macro_f1"]),
                "mean_relshift_largest_component_ratio": float(seed_group["mean_relshift_largest_component_ratio"].mean()),
                "std_relshift_largest_component_ratio": _safe_std(seed_group["mean_relshift_largest_component_ratio"]),
                "mean_dspar_largest_component_ratio": float(seed_group["mean_dspar_largest_component_ratio"].mean()),
                "std_dspar_largest_component_ratio": _safe_std(seed_group["mean_dspar_largest_component_ratio"]),
                "mean_lsp_largest_component_ratio": float(seed_group["mean_lsp_largest_component_ratio"].mean()),
                "std_lsp_largest_component_ratio": _safe_std(seed_group["mean_lsp_largest_component_ratio"]),
                "mean_relshift_num_components": float(seed_group["mean_relshift_num_components"].mean()),
                "std_relshift_num_components": _safe_std(seed_group["mean_relshift_num_components"]),
                "mean_dspar_num_components": float(seed_group["mean_dspar_num_components"].mean()),
                "std_dspar_num_components": _safe_std(seed_group["mean_dspar_num_components"]),
                "mean_lsp_num_components": float(seed_group["mean_lsp_num_components"].mean()),
                "std_lsp_num_components": _safe_std(seed_group["mean_lsp_num_components"]),
                "mean_relshift_vs_dspar_accuracy_delta": float(seed_group["mean_relshift_vs_dspar_accuracy_delta"].mean()),
                "std_relshift_vs_dspar_accuracy_delta": _safe_std(seed_group["mean_relshift_vs_dspar_accuracy_delta"]),
                "mean_relshift_vs_lsp_accuracy_delta": float(seed_group["mean_relshift_vs_lsp_accuracy_delta"].mean()),
                "std_relshift_vs_lsp_accuracy_delta": _safe_std(seed_group["mean_relshift_vs_lsp_accuracy_delta"]),
                "mean_relshift_vs_dspar_macro_f1_delta": float(seed_group["mean_relshift_vs_dspar_macro_f1_delta"].mean()),
                "std_relshift_vs_dspar_macro_f1_delta": _safe_std(seed_group["mean_relshift_vs_dspar_macro_f1_delta"]),
                "mean_relshift_vs_lsp_macro_f1_delta": float(seed_group["mean_relshift_vs_lsp_macro_f1_delta"].mean()),
                "std_relshift_vs_lsp_macro_f1_delta": _safe_std(seed_group["mean_relshift_vs_lsp_macro_f1_delta"]),
                "mean_relshift_vs_dspar_lcc_delta": float(seed_group["mean_relshift_vs_dspar_lcc_delta"].mean()),
                "std_relshift_vs_dspar_lcc_delta": _safe_std(seed_group["mean_relshift_vs_dspar_lcc_delta"]),
                "mean_relshift_vs_lsp_lcc_delta": float(seed_group["mean_relshift_vs_lsp_lcc_delta"].mean()),
                "std_relshift_vs_lsp_lcc_delta": _safe_std(seed_group["mean_relshift_vs_lsp_lcc_delta"]),
                "mean_relshift_vs_dspar_components_delta": float(seed_group["mean_relshift_vs_dspar_components_delta"].mean()),
                "std_relshift_vs_dspar_components_delta": _safe_std(seed_group["mean_relshift_vs_dspar_components_delta"]),
                "mean_relshift_vs_lsp_components_delta": float(seed_group["mean_relshift_vs_lsp_components_delta"].mean()),
                "std_relshift_vs_lsp_components_delta": _safe_std(seed_group["mean_relshift_vs_lsp_components_delta"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def _safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1))


def _coverage_summary(grid_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in grid_df.groupby("dataset"):
        rows.append(
            {
                "dataset": dataset,
                "main_comparable": int((group["status"] == "main_comparable").sum()),
                "aux_comparable": int((group["status"] == "aux_comparable").sum()),
                "unmatched": int((group["status"] == "unmatched").sum()),
                "mean_relshift_gap": float(group["relshift_abs_gap"].mean()),
                "mean_dspar_gap": float(group["dspar_abs_gap"].mean()),
                "mean_lsp_gap": float(group["lsp_abs_gap"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _plot_metric(master_df: pd.DataFrame, grid_df: pd.DataFrame, *, metric: str, ylabel: str, output_path: Path) -> None:
    methods = ["relshift", "dspar", "lsp"]
    colors = {"relshift": "#1f77b4", "dspar": "#d62728", "lsp": "#9467bd"}
    plot_df = master_df[master_df["method"].isin(methods)].copy()
    datasets = sorted(plot_df["dataset"].unique())
    models = sorted(plot_df["model"].unique())
    fig, axes = plt.subplots(
        len(datasets),
        len(models),
        figsize=(6.5 * max(len(models), 1), 4.8 * max(len(datasets), 1)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for row_idx, dataset in enumerate(datasets):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            subset = plot_df[(plot_df["dataset"] == dataset) & (plot_df["model"] == model)].copy()
            grouped = (
                subset.groupby(["method", "pruned_graph_run_dir"], dropna=False)
                .agg(
                    achieved_edge_reduction=("achieved_edge_reduction", "first"),
                    metric_value=(metric, "mean"),
                )
                .reset_index()
            )
            for method in methods:
                method_group = grouped[grouped["method"] == method].sort_values("achieved_edge_reduction")
                if method_group.empty:
                    continue
                ax.plot(
                    method_group["achieved_edge_reduction"],
                    method_group["metric_value"],
                    color=colors[method],
                    marker="o",
                    linewidth=1.8,
                    markersize=4.5,
                    label=method,
                )
            ax.set_title(f"{dataset} / {model}")
            ax.set_xlabel("Achieved Edge Reduction")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_coverage(coverage_df: pd.DataFrame, *, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(coverage_df))
    ax.bar(x, coverage_df["main_comparable"], label="main_comparable")
    ax.bar(x, coverage_df["aux_comparable"], bottom=coverage_df["main_comparable"], label="aux_comparable")
    bottom = coverage_df["main_comparable"] + coverage_df["aux_comparable"]
    ax.bar(x, coverage_df["unmatched"], bottom=bottom, label="unmatched")
    ax.set_xticks(list(x), coverage_df["dataset"].tolist())
    ax.set_ylabel("Target count")
    ax.set_title("Coverage by Method Family Alignment")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_match_gaps(grid_df: pd.DataFrame, *, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    gap_columns = [
        ("relshift_abs_gap", "RelShift"),
        ("dspar_abs_gap", "DSpar"),
        ("lsp_abs_gap", "LSP"),
    ]
    for ax, (column, title) in zip(axes, gap_columns):
        for dataset, group in grid_df.groupby("dataset"):
            group = group.sort_values("target_edge_reduction")
            ax.plot(group["target_edge_reduction"], group[column], marker="o", label=dataset)
        ax.set_title(title)
        ax.set_xlabel("Target Edge Reduction")
        ax.set_ylabel("Absolute Gap")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
