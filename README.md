# Structure-Faithful GNN Pruning

Research codebase for testing whether graphlet-informed node-level structural distortion is a meaningful pruning signal for GNN sparsification.

## Scope

- Datasets: `cora`, `citeseer`
- Models: `GCN`, `GraphSAGE`
- Methods: exact sequential incremental `relshift`, `DSpar`, and `LSP`

## Environment

```bash
pip install -r requirements.txt
```

Real dataset runs also need:

- `torch-geometric`
- PyG companion wheels such as `torch_scatter` and `torch_sparse` for weighted DSpar GraphSAGE runs
- an ORCA-compatible node orbit counting binary for exact GDV4 computation
- the native RelShift incremental extension for exact sequential RelShift on Linux/Colab

For Colab or any fresh environment where `GraphSAGE` is trained on weighted `DSpar` artifacts, install the
matching PyG companion wheels after `torch` is already present:

```bash
python - <<'PY'
import torch
torch_version = torch.__version__.split("+")[0]
cuda_tag = f"cu{torch.version.cuda.replace('.', '')}" if torch.version.cuda else "cpu"
print(f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html")
PY
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "https://data.pyg.org/whl/torch-<TORCH_VERSION>+<CUDA_TAG>.html"
```

## ORCA backend

The exact backend expects an ORCA-compatible binary callable like:

```bash
orca node 4 input.txt output.txt
```

Set the binary path through either:

- pruning config field `orca_path`
- environment variable `ORCA_BINARY`

The upstream `thocevar/orca` repo does not ship a `Makefile`. Compile it directly:

```bash
g++ -O2 -std=c++11 -o orca tmp/orca/orca.cpp
export ORCA_BINARY="$PWD/orca"
```

Build the native sequential RelShift extension on Linux/Colab. The production RelShift workflow uses this exact incremental engine.

```bash
python scripts/build_relshift_incremental_extension.py --smoke
```

This extension is intentionally Linux/Colab-only in this phase. Sequential exact RelShift will fail fast if the
extension is unavailable.

## Dense Frontier Workflow

The fair-comparison pipeline is staged around `achieved_edge_reduction`:

1. build prune-only frontiers
2. build the common attainable grid
3. train only on `main_comparable` points
4. rebuild `master_results`
5. run frontier-first analysis
6. build numeric runtime tables

### 1. Prune-only calibration

```bash
python scripts/run_relshift_calibration.py --datasets configs/datasets/cora.yaml configs/datasets/citeseer.yaml --pruning configs/pruning/relshift_incremental_sequential.yaml --seed 0 --output-root results/frontiers/incremental_relshift
python scripts/run_dspar_calibration.py --datasets configs/datasets/cora.yaml configs/datasets/citeseer.yaml --seed 0 --output-root results/frontiers/dspar
python scripts/run_lsp_calibration.py --datasets configs/datasets/cora.yaml configs/datasets/citeseer.yaml --seed 0 --output-root results/frontiers/lsp
```

RelShift calibration does not write large `edge_scores.csv` files by default. Add `--write-edge-scores`
only when detailed per-edge diagnostics are explicitly needed.

### 2. Common attainable grid

```bash
python scripts/build_common_attainable_grid.py --datasets configs/datasets/cora.yaml configs/datasets/citeseer.yaml --relshift-frontier results/frontiers/incremental_relshift/frontier.csv --dspar-frontier results/frontiers/dspar/frontier.csv --lsp-frontier results/frontiers/lsp/frontier.csv --main-gap 0.015 --aux-gap 0.02 --output-root results/analysis/common_grid
```

### 3. Train selected comparable points

```bash
python scripts/run_training_manifest.py --manifest results/analysis/common_grid/training_manifest.csv --models configs/models/gcn.yaml configs/models/graphsage.yaml --seeds 0 1 2 --include-status main_comparable --output-root results/frontier_training --dense-output-root results/frontier_dense
```

### 4. Rebuild canonical results

```bash
python scripts/build_master_results_table.py --relshift-root results/frontier_training --dspar-root results/frontier_training --lsp-root results/frontier_training --output-root results/analysis/frontier_master
```

### 5. Frontier-first analysis

```bash
python scripts/analyze_frontier_comparison.py --master-results results/analysis/frontier_master/master_results.csv --common-grid results/analysis/common_grid/common_attainable_grid.csv --output-root results/analysis/frontier
```

### 6. Numeric runtime tables

```bash
python scripts/analyze_runtime_comparison.py --master-results results/analysis/frontier_master/master_results.csv --common-grid results/analysis/common_grid/common_attainable_grid.csv --relshift-frontier results/frontiers/incremental_relshift/frontier.csv --dspar-frontier results/frontiers/dspar/frontier.csv --lsp-frontier results/frontiers/lsp/frontier.csv --output-root results/analysis/runtime_comparison
```

## Colab Production Notebook

Use `notebooks/01_colab_runner.ipynb` for the full output run. It mounts Drive in the first cell, works under `/content/structure_faithful_gnn/`, rebuilds incremental RelShift, DSpar, and LSP pruning/training outputs, produces comparison artifacts, then zips `results/` and copies it to Drive in the final cell.

## User-facing scripts

Frontier building and training:

- `scripts/run_relshift_calibration.py`
- `scripts/run_dspar_calibration.py`
- `scripts/run_lsp_calibration.py`
- `scripts/build_common_attainable_grid.py`
- `scripts/run_training_manifest.py`

Analysis:

- `scripts/build_master_results_table.py`
- `scripts/analyze_frontier_comparison.py`
- `scripts/analyze_runtime_comparison.py`

## Artifact contract

Training runs write:

- `resolved_config.json`
- `metrics.json`
- `summary_row.csv`
- `pruning_result.json` and `pruned_edges.pt` for pruning/baseline runs
- optional analysis artifacts such as `edge_scores.csv` when explicitly enabled

Prune-only frontier artifacts write:

- `resolved_config.json`
- `pruned_graph.json`
- `pruning_result.json`
- `pruned_edges.pt`
- `removed_edges.pt`
- optional `edge_weight.pt`
- `pruned_graph_row.csv`

Each frontier root writes:

- `frontier.csv`
- `frontier.json`
- `matched_targets.csv`
- `matched_targets.json`

The common attainable grid stage writes:

- `common_attainable_grid.csv`
- `common_attainable_grid.json`
- `training_manifest.csv`
- `training_manifest.json`
- `unmatched_targets.csv`
- `method_coverage_summary.csv`

The frontier analysis stage writes:

- `frontier_summary.csv`
- `matched_three_way_comparison.csv`
- `pairwise_summary.csv`
- `method_coverage_summary.csv`
- `unmatched_targets.csv`
- `accuracy_vs_achieved_reduction.png`
- `macro_f1_vs_achieved_reduction.png`
- `largest_component_ratio_vs_achieved_reduction.png`
- `num_components_vs_achieved_reduction.png`
- `coverage_by_method.png`
- `match_gap_by_target.png`

RelShift pruning artifacts also include:

- per-round guard metadata inside `pruning_result.json`:
  - `requested_round_budget`, `achieved_round_budget`, `eligible_count`, `blocked_by_bridge_count`, `blocked_by_d_min_count`, `selected_count`, `remaining_edges`
- per-round local-update metadata inside `pruning_result.json`:
  - `update_union_size`, `update_union_edge_count_before`, `update_union_edge_count_after`, `local_update_runtime_sec`
- final-run metadata:
  - `round_state_update_mode = "single_edge_exact_incremental"`
  - `native_kernel_variant = "mask_count_v4_combinatorial"` and `native_kernel_version = "mask_count_combinatorial_best_v4"` for pruning/training runs
  - `native_selection_mode = "native_best_with_array_cache"` for default pruning/training runs
  - `cache_invalidation_mode = "native_or_boolean_state_changed_incident_plus_delta_impacted"`
- enriched `edge_scores.csv` columns, emitted only when `write_edge_scores=true`:
  - `mean_abs_delta_sig`, `mean_rel_delta_sig`, `mean_denom`, `min_denom`

## Notes

- Graph preprocessing is simple-undirected and removes self-loops before pruning.
- `RelShift` uses fixed normalization statistics from the original graph.
- The intended research path is exact GDV4 through ORCA, not any proxy signature.
- `2-hop` is only the local update scope for exact delta maintenance. It is not a redefinition of the node signature.
- Production RelShift removes at most one edge per round through `relshift_engine = "incremental_sequential_exact"`.
- The dense fair-comparison workflow uses `achieved_edge_reduction` as the only valid x-axis.
- Main comparison points are those with `abs_gap <= 0.015` across all three methods.
- `0.015 < gap <= 0.02` is auxiliary only and must not be mixed into the main claim.
- The local DSpar baseline keeps the paper/source sampling rule: `p_e proportional to 1/deg(u)+1/deg(v)` and `Q = 0.16 n log(n) / epsilon^2`.
- The local LSP baseline is re-frozen to the upstream `dotd/GNN_experiments` pruning surface at commit `aa9cf7a38d23cee005a3e5b4df9a60fee1f46ea3`. `LSP-T` does not expose the paper-level `m` bin parameter as a runnable hyperparameter here; `lsp_m` stays empty and `sparsity` remains the active control.

## Exact RelShift runtime profiling

Phase 1 profiling is enabled with `configs/pruning/relshift_incremental_profile.yaml` or by setting the following pruning options:

```yaml
profile_rounds: true
write_runtime_profile: true
profile_memory: true
profile_update_diagnostics: false
profile_native_kernel: false
```

Run a controlled profile:

```bash
python scripts/profile_relshift_exact.py \
  --dataset configs/datasets/cora.yaml \
  --pruning configs/pruning/relshift_incremental_profile.yaml \
  --rho 0.05 \
  --warmup-runs 1 \
  --repeats 3 \
  --output-root results/relshift_exact_profile/cora
```

Each measured run writes:

- `runtime_summary.json`: wall-clock decomposition, cache behavior, counts, memory lower bounds, environment information, and profiling coverage.
- `runtime_by_round.csv`: round-level bridge, eligibility, rescoring, selection, state update, edge removal, active-list maintenance, local-structure, and memory metrics.
- `pruning_result.json`, `pruned_edges.pt`, and `removed_edges.pt`: the corresponding exact pruning result.

The aggregate files `profile_runs.csv` and `profile_runs.json` compare repeated measured runs. `profile_native_kernel: true` exposes native sub-kernel timings but disables the normal parallel scoring path, so it should be used as a separate diagnostic run rather than mixed with production runtime measurements.
