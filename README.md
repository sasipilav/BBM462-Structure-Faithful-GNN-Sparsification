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

## Phase 1 step 2: immutable active-mask graph storage

The exact incremental engine now keeps the initial CSR adjacency immutable. Stable edge IDs are mapped to every directed adjacency entry, and an edge deletion only:

- clears `active_edge_mask[edge_id]`,
- decrements the two endpoint degrees,
- decrements the active-edge counter.

Native graphlet, support, two-hop, and bridge traversals skip inactive adjacency entries through the mask. The Python engine also keeps a fixed `all_edge_ids` array and no longer rebuilds the active-edge ID list after every deletion. This preserves the reference sequential pruning semantics while removing CSR element shifts and per-round active-list copies.

The runtime profiler now also records scale-sensitive work that may be small on Cora but grow sharply on larger graphs:

- active edge-ID entries scanned and inactive IDs skipped,
- bridge nodes and adjacency entries visited,
- inactive adjacency entries skipped during bridge traversal,
- immutable/active/inactive directed adjacency counts,
- tombstone ratios,
- candidate-delta and total known edge-state memory projections at 1M, 10M, and 60M edges.

`active_edge_list_rebuild_runtime_sec` is retained for before/after compatibility and is zero on the immutable active-mask path.

## Phase 1 step 3: exact versioned heap and safe lazy deletion

The exact sequential engine can select candidates through either:

```yaml
incremental_selection_backend: linear_scan
```

or:

```yaml
incremental_selection_backend: versioned_heap
heap_rebuild_ratio: 4.0
```

Ready-to-run configurations are provided in:

- `configs/pruning/relshift_incremental_heap.yaml`
- `configs/pruning/relshift_incremental_heap_profile.yaml`

The versioned heap preserves the complete reference comparison key:

```text
(RelShift score, endpoint degree tie-break, triangle-support tie-break, stable edge ID)
```

Only obsolete heap records are deleted lazily. Score recomputation is not deferred until pop time. Whenever an edge score or tie-break value may have changed, that edge is immediately marked dirty, its version is incremented, and its new exact key is computed before the next selection. This prevents a decreased stale score from being hidden below an incorrect old heap key.

Bridge and minimum-degree exclusions are maintained as permanent guard states. This is exact in the deletion-only process because an active bridge cannot become non-bridge after further edge deletions, and node degrees cannot increase. The native Tarjan bridge pass is still executed each round to discover newly created bridges; Step 3 does not solve the global bridge-maintenance cost.

The heap uses deterministic rebuilding to bound stale-record growth. A rebuild is triggered when the heap contains more than 1,024 entries and exceeds `heap_rebuild_ratio * eligible_active_edges`. The default ratio `4.0` is a speed-memory compromise; larger ratios reduce rebuild frequency but increase heap memory.

Run a direct exact comparison:

```bash
python scripts/compare_relshift_selection_backends.py \
  --dataset configs/datasets/cora.yaml \
  --pruning configs/pruning/relshift_incremental_profile.yaml \
  --rho 0.10 \
  --warmup-runs 1 \
  --repeats 3 \
  --heap-rebuild-ratio 4.0 \
  --output-root results/relshift_heap_comparison/cora_rho_010
```

The script alternates the two backends in one process, asserts deterministic within-backend output, and fails if their complete removed-edge sequences differ.

Additional profiler fields include:

- dirty edge entries scanned;
- heap keys pushed;
- heap update and pop times;
- stale, inactive, guarded, and dirty records popped;
- rebuild count and rebuild edge-ID scans;
- current and maximum heap size;
- heap auxiliary-state and rebuild-bound memory projections.

Validated Cora results and the exactness argument are documented in `docs/phase1_step3_versioned_heap.md` and `docs/phase1_step3_validation.json`.

## Phase 1 step 3.5: optimized exact engine

The exact sequential engine now provides an optimized fused configuration:

```yaml
native_state_fusion: true
incremental_selection_backend: versioned_heap
heap_storage_mode: indexed
bridge_maintenance_mode: lazy_exact
adjacency_compaction_threshold: 0.20
```

Ready-to-run configurations:

- `configs/pruning/relshift_incremental_exact_optimized.yaml`
- `configs/pruning/relshift_incremental_exact_optimized_profile.yaml`

The optimized path fuses GDV/candidate state into the native engine, maintains triangle support incrementally, caches endpoint contributions and node denominators, skips zero orbit-delta coordinates, uses reusable epoch workspaces, removes selected-pair sort/unique materialization, verifies bridges lazily and exactly, compacts tombstones deterministically, and uses a one-entry-per-edge indexed heap.

On the validated Cora setup it preserved the complete deletion sequence and reduced round runtime by approximately `2.35x` at `rho=0.10` and `2.62x` at `rho=0.30` relative to the Phase 1 Step 3 active-mask/versioned-heap reference. Exactness arguments, implementation details, limitations, and benchmark methodology are documented in:

- `docs/phase1_step35_exact_engine_optimization.md`
- `docs/phase1_step35_validation.json`


## Phase 1 steps 4 and 5: exact validation and scaling

The optimized exact engine is now covered by an independent validation chain rather than backend-to-backend comparison alone. The permanent suite includes exhaustive five-node orbit-delta checks, a brute-force greedy oracle, a 240-run production configuration matrix, cache-on/cache-off trajectory checks, and controlled synthetic scaling with exact initial GDVs.

A floating-point tie regression discovered by the wide matrix was fixed by routing all exact score paths through `canonical_native_raw_log1p_v1`. Current validation and scaling reports are available in:

- `docs/phase1_step4_exact_equivalence.md`
- `docs/phase1_step4_validation.json`
- `docs/phase1_step5_reprofiling_scaling.md`
- `docs/phase1_step5_validation.json`
- `docs/phase1_exact_engine_completion.md`

Raw matrices are stored under `results/phase1_exact_engine/`.

## Phase 2 steps 1--3: canonical orbits, exact edge events, and checkpoints

Phase 2 begins with a versioned canonical registry for the 15 ORCA-compatible
node orbits used by RelShift. Python graphlet classification and the native C++
canonical tables are exhaustively cross-checked for every connected labeled
2--4 node mask.

The optimized exact engine can also emit one exact orbit event for every
selected edge. Enable it with:

```yaml
orbit_explainability_enabled: true
```

The event path reuses the selected-edge graphlet enumeration and records a
`15 x 16` source-to-destination orbit transition matrix, where column 15 is the
destroyed/disconnected sink. The same opt-in path now captures deterministic
full-graph GDV checkpoints at requested edge-reduction fractions. Every
checkpoint's raw signed displacement must exactly equal both the cumulative
selected-edge event net and the net reconstructed from the cumulative
transition matrix. Full node-by-orbit snapshots remain optional.

Logging is opt-in, requires an artifact directory, and does not use the
all-candidate `write_edge_scores` path.

Ready-to-run configuration and documentation:

- `configs/pruning/relshift_incremental_exact_orbit_explainability.yaml`
- `docs/phase2_step1_canonical_orbit_registry.md`
- `docs/phase2_step2_exact_orbit_event_logging.md`
- `docs/phase2_step3_checkpoint_orbit_distortion.md`
- `docs/phase2_step3_validation.json`
- `docs/phase2_orbit_registry.json`
