from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..config import PruningConfig
from ..gdv.backends import GDVService, apply_standardization, fit_standardization
from .incremental_relshift import IncrementalEdgeDelta
from ..types import DatasetBundle, PruningResult
from ..utils.graph import (
    adjacency_sets,
    bridge_edges,
    bridge_edges_from_adjacency,
    degree_array,
    induced_subgraph_edges,
    k_hop_nodes,
    remove_edge_pairs,
    split_budget,
    support_scores,
)


@dataclass
class EdgeScore:
    edge: tuple[int, int]
    relshift: float
    degree_score: float
    support_score: float
    region_size: int
    update_size: int
    mean_abs_delta_sig: float
    mean_rel_delta_sig: float
    mean_denom: float
    min_denom: float
    update_nodes: tuple[int, ...]
    update_node_set: frozenset[int]
    directly_attached_size: int
    four_node_pair_count: int
    raw_delta: np.ndarray | None


def relshift_prune(
    bundle: DatasetBundle,
    config: PruningConfig,
    gdv_service: GDVService,
    *,
    seed: int,
    artifact_dir: str | Path | None = None,
) -> PruningResult:
    start = time.perf_counter()
    budget = int(round(bundle.num_edges * config.rho))
    current_edges = bundle.edge_index.clone()
    requested_relshift_engine = str((config.options or {}).get("relshift_engine", "auto")).strip().lower() or "auto"
    score_norm = "l1"
    use_score_cache = bool((config.options or {}).get("use_score_cache", True))
    write_edge_scores = bool((config.options or {}).get("write_edge_scores", False))
    verbose = bool((config.options or {}).get("verbose", False))
    profile_rounds = bool((config.options or {}).get("profile_rounds", False))
    profile_update_diagnostics = bool((config.options or {}).get("profile_update_diagnostics", False))
    write_runtime_profile = bool((config.options or {}).get("write_runtime_profile", profile_rounds))
    collect_runtime_profile = bool(profile_rounds or write_runtime_profile)
    profile_memory = bool((config.options or {}).get("profile_memory", write_runtime_profile))
    profile_label = str((config.options or {}).get("profile_label", "")).strip()
    peak_rss_start_mb = _peak_rss_mb() if profile_memory else None
    use_native_graph_state = bool((config.options or {}).get("use_native_graph_state", True))
    profile_native_kernel = bool((config.options or {}).get("profile_native_kernel", False))
    native_omp_threads = int((config.options or {}).get("native_omp_threads", 0) or 0)
    if native_omp_threads < 0:
        raise ValueError("native_omp_threads must be non-negative.")
    native_kernel_variant = str((config.options or {}).get("native_kernel_variant", "mask_count_v4_combinatorial")).strip().lower()
    if native_kernel_variant != "mask_count_v4_combinatorial":
        raise ValueError(f"Unsupported native_kernel_variant: {native_kernel_variant}")
    if requested_relshift_engine == "incremental_sequential_exact":
        round_budgets = [1 for _ in range(max(budget, 0))]
    else:
        round_budgets = split_budget(budget, config.recompute_rounds)
    effective_recompute_rounds = len(round_budgets)
    relshift_engine = _resolve_relshift_engine(requested_relshift_engine, round_budgets=round_budgets)
    is_incremental_sequential = relshift_engine == "incremental_sequential_exact"
    candidate_delta_cache_enabled = bool(is_incremental_sequential and use_score_cache and not write_edge_scores)
    candidate_delta_cache_mode = "mixed_correction" if candidate_delta_cache_enabled else "off"
    original_region_edge_map = (
        None
        if is_incremental_sequential
        else _build_original_region_edge_map(
            original_edges=[tuple(edge) for edge in bundle.edge_index.t().tolist()],
            num_nodes=bundle.num_nodes,
        )
    )
    configuration_setup_runtime_sec = float(time.perf_counter() - start)
    extension_setup_start = time.perf_counter()
    if relshift_engine == "incremental_sequential_exact":
        _require_incremental_extension()
        if native_omp_threads > 0:
            _set_native_openmp_threads(native_omp_threads)
    native_openmp_info = _native_openmp_info() if is_incremental_sequential else {"openmp_enabled": False, "openmp_max_threads": 1}
    extension_setup_runtime_sec = float(time.perf_counter() - extension_setup_start)
    _relshift_log(
        verbose,
        f"[relshift] start dataset={bundle.name} backend={gdv_service.backend_name} "
        f"edges={bundle.num_edges} rho={config.rho} budget={budget} rounds={effective_recompute_rounds}",
    )
    initial_gdv_start = time.perf_counter()
    original_raw = gdv_service.compute_graph_gdv(bundle.num_nodes, bundle.edge_index, cache_namespace="original_full")
    initial_gdv_runtime_sec = float(time.perf_counter() - initial_gdv_start)
    initial_gdv_compute_info = dict(getattr(gdv_service, "last_compute_info", {}) or {})
    standardization_start = time.perf_counter()
    stats = fit_standardization(original_raw)
    current_raw = original_raw.copy()
    current_std = apply_standardization(current_raw, stats)
    standardization_runtime_sec = float(time.perf_counter() - standardization_start)

    state_initialization_start = time.perf_counter()
    removed_total: list[tuple[int, int]] = []
    round_summaries: list[dict[str, object]] = []
    analysis_rows: list[dict[str, object]] = []
    score_cache: dict[tuple[int, int], EdgeScore] = {}
    invalidated_score_edges: set[tuple[int, int]] = set()
    active_edges = [tuple(edge) for edge in bundle.edge_index.t().tolist()]
    edge_by_id = list(active_edges)
    edge_array_by_id = np.asarray(edge_by_id, dtype=np.int64).reshape((-1, 2)) if edge_by_id else np.empty((0, 2), dtype=np.int64)
    edge_to_id = {edge: edge_id for edge_id, edge in enumerate(edge_by_id)}
    active_edge_ids = list(range(len(edge_by_id)))
    incident_edge_ids: list[list[int]] = [[] for _ in range(bundle.num_nodes)]
    for edge_id, (u, v) in enumerate(edge_by_id):
        incident_edge_ids[u].append(edge_id)
        incident_edge_ids[v].append(edge_id)
    score_cache_values = np.full(len(edge_by_id), np.inf, dtype=np.float64)
    degree_score_cache_values = np.full(len(edge_by_id), np.inf, dtype=np.float64)
    support_score_cache_values = np.full(len(edge_by_id), np.inf, dtype=np.float64)
    valid_score_cache = np.zeros(len(edge_by_id), dtype=np.uint8)
    candidate_delta_cache = np.zeros((len(edge_by_id), 2, 15), dtype=np.int64)
    delta_valid_cache = np.zeros(len(edge_by_id), dtype=np.uint8)
    current_adjacency = adjacency_sets(bundle.num_nodes, bundle.edge_index)
    current_degrees = np.asarray([len(neighbors) for neighbors in current_adjacency], dtype=np.int64)
    fast_incremental_default = is_incremental_sequential and not write_edge_scores
    native_graph_state = None
    initial_native_graph_state_runtime_sec = 0.0
    if fast_incremental_default and use_native_graph_state:
        timer = time.perf_counter()
        initial_row_ptr, initial_col_idx = _adjacency_to_csr(current_adjacency)
        native_graph_state = _create_native_graph_state(initial_row_ptr, initial_col_idx, edge_array_by_id)
        initial_native_graph_state_runtime_sec = float(time.perf_counter() - timer)
    if candidate_delta_cache_enabled and native_graph_state is None:
        raise RuntimeError("mixed-correction candidate delta cache requires persistent native graph state.")
    invalidation_marks = np.zeros(len(edge_by_id), dtype=np.uint32)
    invalidation_epoch = 1
    pruning_state_initialization_runtime_sec = float(time.perf_counter() - state_initialization_start)

    for round_idx, round_budget in enumerate(round_budgets, start=1):
        current_edge_count = len(active_edge_ids) if is_incremental_sequential else int(current_edges.shape[1])
        if round_budget <= 0 or current_edge_count == 0:
            continue

        round_profile: dict[str, object] = {
            "active_edge_count_before": int(current_edge_count),
        }
        if collect_runtime_profile:
            round_profile["mean_degree_before"] = float(np.mean(current_degrees)) if current_degrees.size else 0.0
            round_profile["max_degree_before"] = int(np.max(current_degrees)) if current_degrees.size else 0
            round_profile["min_degree_before"] = int(np.min(current_degrees)) if current_degrees.size else 0
            if profile_memory:
                round_profile["peak_rss_mb_before"] = _peak_rss_mb()
        setup_start = time.perf_counter()
        round_start = setup_start
        if is_incremental_sequential:
            round_profile["build_adjacency_runtime_sec"] = 0.0
            round_profile["degree_runtime_sec"] = 0.0
            round_profile["support_runtime_sec"] = 0.0
            degrees = current_degrees
            round_edges = [] if fast_incremental_default else [edge_by_id[edge_id] for edge_id in active_edge_ids]
            round_edge_ids: list[int] | None = active_edge_ids
            support = None
        else:
            timer = time.perf_counter()
            current_adjacency = adjacency_sets(bundle.num_nodes, current_edges)
            round_profile["build_adjacency_runtime_sec"] = float(time.perf_counter() - timer)
            timer = time.perf_counter()
            degrees = degree_array(bundle.num_nodes, current_edges)
            round_profile["degree_runtime_sec"] = float(time.perf_counter() - timer)
            timer = time.perf_counter()
            support = support_scores(bundle.num_nodes, current_edges)
            round_profile["support_runtime_sec"] = float(time.perf_counter() - timer)
            round_edges = [tuple(edge) for edge in current_edges.t().tolist()]
            round_edge_ids = None
        guard_counts = {"eligible": 0, "bridge_guard": 0, "d_min_guard": 0}
        eligible: list[tuple[int, int]] = []
        eligible_edge_ids: list[int] = []
        native_rescored_edge_ids: list[int] = []
        native_reused_edge_ids: list[int] = []
        native_refresh_edge_ids: list[int] = []
        native_reused_count = 0
        native_cached_best_edge_id = -1
        prebuilt_csr: tuple[np.ndarray, np.ndarray] | None = None
        prebuilt_csr_runtime_sec = 0.0
        if fast_incremental_default:
            if round_edge_ids is None:
                raise RuntimeError("Incremental RelShift requires active edge ids.")
            if native_graph_state is None:
                timer = time.perf_counter()
                prebuilt_csr = _adjacency_to_csr(current_adjacency)
                prebuilt_csr_runtime_sec = float(time.perf_counter() - timer)
            guard_result = _eligible_edge_id_partitions_incremental_native(
                row_ptr=None if prebuilt_csr is None else prebuilt_csr[0],
                col_idx=None if prebuilt_csr is None else prebuilt_csr[1],
                active_edge_ids=round_edge_ids,
                edge_array_by_id=edge_array_by_id,
                degrees=degrees,
                d_min=config.d_min,
                guard_bridges=config.guard_bridges,
                valid_score_cache=valid_score_cache,
                delta_valid_cache=delta_valid_cache if candidate_delta_cache_enabled else None,
                use_score_cache=use_score_cache,
                native_graph_state=native_graph_state,
                score_cache_values=score_cache_values,
                degree_score_cache_values=degree_score_cache_values,
                support_score_cache_values=support_score_cache_values,
            )
            eligible_edge_ids = guard_result["eligible_edge_ids"]
            native_rescored_edge_ids = guard_result["rescored_edge_ids"]
            native_reused_edge_ids = guard_result["reused_edge_ids"]
            native_refresh_edge_ids = guard_result.get("refresh_edge_ids", [])
            native_reused_count = int(guard_result.get("reused_count", 0))
            native_cached_best_edge_id = int(guard_result.get("cached_best_edge_id", -1))
            guard_counts["eligible"] = int(guard_result["eligible_count"])
            guard_counts["bridge_guard"] = int(guard_result["blocked_by_bridge_count"])
            guard_counts["d_min_guard"] = int(guard_result["blocked_by_d_min_count"])
            round_profile["bridge_count"] = int(guard_result.get("bridge_count", 0))
            round_profile["bridge_runtime_sec"] = float(guard_result["bridge_runtime_sec"])
            round_profile["tarjan_bridge_runtime_sec"] = round_profile["bridge_runtime_sec"] if config.guard_bridges else 0.0
            round_profile["eligibility_runtime_sec"] = float(guard_result["eligibility_runtime_sec"])
            round_profile["score_csr_runtime_sec"] = prebuilt_csr_runtime_sec
            round_profile["cache_partition_runtime_sec"] = float(guard_result.get("cache_partition_runtime_sec", 0.0))
        else:
            timer = time.perf_counter()
            bridges = (
                bridge_edges_from_adjacency(current_adjacency)
                if config.guard_bridges and is_incremental_sequential
                else bridge_edges(bundle.num_nodes, current_edges)
                if config.guard_bridges
                else set()
            )
            round_profile["bridge_runtime_sec"] = float(time.perf_counter() - timer)
            round_profile["bridge_count"] = int(len(bridges))
            round_profile["tarjan_bridge_runtime_sec"] = round_profile["bridge_runtime_sec"] if config.guard_bridges and is_incremental_sequential else 0.0
            timer = time.perf_counter()
            if is_incremental_sequential:
                if round_edge_ids is None:
                    raise RuntimeError("Incremental RelShift requires active edge ids.")
                for edge_id in round_edge_ids:
                    edge_pair = edge_by_id[edge_id]
                    reason = _guard_reason(edge_pair, degrees, bridges, config.d_min)
                    guard_counts[reason] += 1
                    if reason == "eligible":
                        eligible.append(edge_pair)
                        eligible_edge_ids.append(edge_id)
            else:
                for edge in round_edges:
                    edge_pair = tuple(edge)
                    reason = _guard_reason(edge_pair, degrees, bridges, config.d_min)
                    guard_counts[reason] += 1
                    if reason == "eligible":
                        eligible.append(edge_pair)
            round_profile["eligibility_runtime_sec"] = float(time.perf_counter() - timer)
        round_profile["round_setup_runtime_sec"] = float(time.perf_counter() - setup_start)
        eligible_count = guard_counts["eligible"]
        if eligible_count == 0:
            _relshift_log(verbose, f"[relshift] round={round_idx} no eligible edges")
            no_eligible_runtime_sec = float(time.perf_counter() - round_start)
            round_profile["active_edge_count_after"] = int(current_edge_count)
            round_profile["accounted_runtime_sec"] = float(round_profile["round_setup_runtime_sec"])
            round_profile["unaccounted_runtime_sec"] = max(
                0.0, no_eligible_runtime_sec - float(round_profile["round_setup_runtime_sec"])
            )
            round_profile["runtime_coverage_ratio"] = (
                min(1.0, float(round_profile["round_setup_runtime_sec"]) / no_eligible_runtime_sec)
                if no_eligible_runtime_sec > 0.0
                else 1.0
            )
            if profile_memory:
                round_profile["peak_rss_mb_after"] = _peak_rss_mb()
            round_summaries.append(
                {
                    "round": round_idx,
                    "requested_round_budget": round_budget,
                    "achieved_round_budget": 0,
                    "active_edge_count_before": current_edge_count,
                    "eligible_count": 0,
                    "blocked_by_bridge_count": guard_counts["bridge_guard"],
                    "blocked_by_d_min_count": guard_counts["d_min_guard"],
                    "selected_count": 0,
                    "remaining_edges": current_edge_count,
                    "update_union_size": 0,
                    "update_union_edge_count_before": 0,
                    "update_union_edge_count_after": 0,
                    "local_update_runtime_sec": 0.0,
                    "selected_delta_runtime_sec": 0.0,
                    "apply_state_update_runtime_sec": 0.0,
                    "round_runtime_sec": no_eligible_runtime_sec,
                    "round_runtime_excluding_analysis_sec": no_eligible_runtime_sec,
                    "round_runtime_excluding_setup_sec": 0.0,
                    **(round_profile if collect_runtime_profile else {}),
                }
            )
            continue

        batch_size = min(round_budget, eligible_count)
        round_after_setup_start = time.perf_counter()
        _relshift_log(
            verbose,
            f"[relshift] round={round_idx} current_edges={current_edge_count} "
            f"eligible={eligible_count} target_remove={batch_size}",
        )
        fast_incremental_selection = fast_incremental_default
        exact_scores: list[EdgeScore] = []
        rescored_edges: list[tuple[int, int]] = []
        rescored_edge_ids = native_rescored_edge_ids if fast_incremental_selection else []
        reused_edge_ids = native_reused_edge_ids if fast_incremental_selection else []
        refresh_edge_ids = native_refresh_edge_ids if fast_incremental_selection else []
        reused_score_count = native_reused_count if fast_incremental_selection else 0
        scalar_refreshed_score_count = 0
        native_score_runtime_sec = 0.0
        native_scalar_refresh_runtime_sec = 0.0
        native_cached_best_edge_id = native_cached_best_edge_id if fast_incremental_selection else -1
        native_invalidated_score_count: int | None = None
        native_cache_update_profile: dict[str, float | int] = {}
        incremental_score_profile: dict[str, float] = {}
        timer = time.perf_counter()
        if not fast_incremental_selection:
            for edge in eligible:
                cached_score = score_cache.get(edge) if use_score_cache else None
                if cached_score is not None and edge not in invalidated_score_edges:
                    exact_scores.append(cached_score)
                    reused_score_count += 1
                else:
                    rescored_edges.append(edge)
        if not fast_incremental_selection:
            round_profile["cache_partition_runtime_sec"] = float(time.perf_counter() - timer)

        selected_edge_id: int | None = None
        selected: list[EdgeScore] = []
        round_profile["best_selection_runtime_sec"] = 0.0
        if fast_incremental_selection:
            refreshed_best_edge_id = None
            if refresh_edge_ids:
                (
                    refreshed_best_edge_id,
                    native_scalar_refresh_runtime_sec,
                    scalar_refresh_profile,
                ) = _refresh_incremental_scores_from_delta_cache(
                    refresh_edge_ids=refresh_edge_ids,
                    current_raw=current_raw,
                    current_std=current_std,
                    stats=stats,
                    score_mode=config.score_mode,
                    eps=config.eps,
                    candidate_delta_cache=candidate_delta_cache,
                    delta_valid_cache=delta_valid_cache,
                    score_cache_values=score_cache_values,
                    degree_score_cache_values=degree_score_cache_values,
                    support_score_cache_values=support_score_cache_values,
                    valid_score_cache=valid_score_cache,
                    native_graph_state=native_graph_state,
                )
                scalar_refreshed_score_count = len(refresh_edge_ids)
                round_profile.update(scalar_refresh_profile)
            rescored_best_edge_id = None
            if rescored_edge_ids:
                (
                    rescored_best_edge_id,
                    native_score_runtime_sec,
                    incremental_score_profile,
                ) = _score_incremental_round_best(
                    rescored_edge_ids=rescored_edge_ids,
                    edge_array_by_id=edge_array_by_id,
                    current_adjacency=current_adjacency,
                    current_raw=current_raw,
                    current_std=current_std,
                    stats=stats,
                    score_mode=config.score_mode,
                    eps=config.eps,
                    score_cache_values=score_cache_values,
                    degree_score_cache_values=degree_score_cache_values,
                    support_score_cache_values=support_score_cache_values,
                    valid_score_cache=valid_score_cache,
                    candidate_delta_cache=candidate_delta_cache if candidate_delta_cache_enabled else None,
                    delta_valid_cache=delta_valid_cache if candidate_delta_cache_enabled else None,
                    profile_native_kernel=profile_native_kernel,
                    native_kernel_variant=native_kernel_variant,
                    native_graph_state=native_graph_state,
                    prebuilt_csr=prebuilt_csr,
                    prebuilt_csr_runtime_sec=prebuilt_csr_runtime_sec,
                )
                _relshift_log(
                    verbose,
                    f"[relshift] round={round_idx} rescored={len(rescored_edge_ids)}/{len(rescored_edge_ids)} "
                    f"reused={reused_score_count} native_score_sec={native_score_runtime_sec:.4f}",
                )
            else:
                _relshift_log(verbose, f"[relshift] round={round_idx} rescored=0 reused={reused_score_count} native_score_sec=0.0000")
            best_selection_start = time.perf_counter()
            if native_cached_best_edge_id >= 0:
                cached_best_edge_id = native_cached_best_edge_id
            else:
                cached_best_edge_id = _best_cached_edge_id(
                    reused_edge_ids,
                    score_cache_values=score_cache_values,
                    degree_score_cache_values=degree_score_cache_values,
                    support_score_cache_values=support_score_cache_values,
                )
            selected_edge_id = _merge_best_edge_ids(
                cached_best_edge_id,
                _merge_best_edge_ids(
                    refreshed_best_edge_id,
                    rescored_best_edge_id,
                    score_cache_values=score_cache_values,
                    degree_score_cache_values=degree_score_cache_values,
                    support_score_cache_values=support_score_cache_values,
                ),
                score_cache_values=score_cache_values,
                degree_score_cache_values=degree_score_cache_values,
                support_score_cache_values=support_score_cache_values,
            )
            if selected_edge_id is None:
                raise RuntimeError("Incremental RelShift could not select an edge from non-empty eligible set.")
            selected_edge = edge_by_id[selected_edge_id]
            removed_round = [selected_edge]
            round_profile["best_selection_runtime_sec"] = float(time.perf_counter() - best_selection_start)
        elif relshift_engine == "incremental_sequential_exact":
            if rescored_edges:
                rescored_scores, native_score_runtime_sec, incremental_score_profile = _compute_edge_scores_incremental_round(
                    rescored_edges=rescored_edges,
                    current_adjacency=current_adjacency,
                    current_raw=current_raw,
                    current_std=current_std,
                    degrees=degrees,
                    support=support,
                    stats=stats,
                    score_mode=config.score_mode,
                    eps=config.eps,
                    include_update_nodes=False,
                    include_update_sizes=write_edge_scores,
                )
                for computed_score in rescored_scores:
                    score_cache[computed_score.edge] = computed_score
                    exact_scores.append(computed_score)
                _relshift_log(
                    verbose,
                    f"[relshift] round={round_idx} rescored={len(rescored_edges)}/{len(rescored_edges)} "
                    f"reused={reused_score_count} native_score_sec={native_score_runtime_sec:.4f}",
                )
            else:
                _relshift_log(verbose, f"[relshift] round={round_idx} rescored=0 reused={reused_score_count} native_score_sec=0.0000")
        else:
            progress_step = max(1, len(rescored_edges) // 20) if rescored_edges else 1
            if not rescored_edges:
                _relshift_log(verbose, f"[relshift] round={round_idx} rescored=0 reused={reused_score_count}")
            for candidate_idx, edge in enumerate(rescored_edges, start=1):
                computed_score = _compute_edge_score_orca(
                    edge=edge,
                    num_nodes=bundle.num_nodes,
                    current_edges=current_edges,
                    current_adjacency=current_adjacency,
                    current_raw=current_raw,
                    current_std=current_std,
                    degrees=degrees,
                    support=support,
                    stats=stats,
                    gdv_service=gdv_service,
                    score_mode=config.score_mode,
                    eps=config.eps,
                    cache_namespace=f"proposal_exact_round_{round_idx}_candidates",
                )
                score_cache[edge] = computed_score
                exact_scores.append(computed_score)
                if candidate_idx == 1 or candidate_idx == len(rescored_edges) or candidate_idx % progress_step == 0:
                    _relshift_log(
                        verbose,
                        f"[relshift] round={round_idx} rescored={candidate_idx}/{len(rescored_edges)} "
                        f"reused={reused_score_count} last_edge={edge[0]}-{edge[1]}",
                    )
        round_profile.update(incremental_score_profile)
        timer = time.perf_counter()
        if not fast_incremental_selection:
            selected = _select_edge_scores(exact_scores, batch_size=batch_size, sequential=is_incremental_sequential)
            removed_round = [score.edge for score in selected]
        round_profile["score_sort_runtime_sec"] = float(time.perf_counter() - timer)
        round_profile["score_selection_runtime_sec"] = round_profile["score_sort_runtime_sec"]
        if not fast_incremental_selection:
            round_profile["best_selection_runtime_sec"] = round_profile["score_selection_runtime_sec"]
        removed_total.extend(removed_round)
        update_block_start = time.perf_counter()
        selected_delta_runtime_sec = 0.0
        apply_state_update_runtime_sec = 0.0
        if relshift_engine == "incremental_sequential_exact":
            timer = time.perf_counter()
            selected_edge_delta, native_invalidated_score_count, native_cache_update_profile = _compute_selected_edge_delta_incremental(
                current_adjacency,
                removed_round[0],
                native_graph_state=native_graph_state,
                prebuilt_csr=prebuilt_csr,
                selected_edge_id=selected_edge_id,
                valid_score_cache=valid_score_cache,
                delta_valid_cache=delta_valid_cache,
                candidate_delta_cache=candidate_delta_cache,
                update_candidate_delta_cache=candidate_delta_cache_enabled,
                invalidate_cache=bool(use_score_cache and fast_incremental_selection and selected_edge_id is not None),
            )
            selected_delta_runtime_sec = float(time.perf_counter() - timer)
            timer = time.perf_counter()
            (
                current_raw,
                current_std,
                local_update_summary,
            ) = _apply_single_edge_incremental_update(
                current_adjacency=current_adjacency,
                current_raw=current_raw,
                current_std=current_std,
                edge_delta=selected_edge_delta,
                stats=stats,
                profile_update_diagnostics=profile_update_diagnostics,
            )
            apply_state_update_runtime_sec = float(time.perf_counter() - timer)
        else:
            timer = time.perf_counter()
            (
                current_raw,
                current_std,
                local_update_summary,
            ) = _apply_batch_local_update(
                num_nodes=bundle.num_nodes,
                removed_round=removed_round,
                current_adjacency=current_adjacency,
                current_raw=current_raw,
                current_std=current_std,
                stats=stats,
                gdv_service=gdv_service,
                cache_namespace=f"proposal_exact_round_{round_idx}_batch_update",
            )
            apply_state_update_runtime_sec = float(time.perf_counter() - timer)
        local_update_summary["local_update_runtime_sec"] = float(time.perf_counter() - update_block_start)
        local_update_summary["selected_delta_runtime_sec"] = selected_delta_runtime_sec
        local_update_summary["apply_state_update_runtime_sec"] = apply_state_update_runtime_sec
        if native_cache_update_profile:
            round_profile.update({key: float(value) for key, value in native_cache_update_profile.items()})
        timer = time.perf_counter()
        dirty_signature_nodes = list(selected_edge_delta.affected_nodes) if relshift_engine == "incremental_sequential_exact" else _batch_update_nodes(current_adjacency, removed_round)
        round_profile["dirty_signature_nodes_runtime_sec"] = float(time.perf_counter() - timer)
        timer = time.perf_counter()
        invalidated_score_count = 0
        if use_score_cache:
            if fast_incremental_selection:
                if selected_edge_id is None:
                    raise RuntimeError("Fast incremental invalidation requires selected_edge_id.")
                if native_invalidated_score_count is None:
                    valid_score_cache[selected_edge_id] = 0
                    invalidated_score_count = _invalidate_edge_ids_after_selected(
                        selected_edge_id=selected_edge_id,
                        edge_delta=selected_edge_delta,
                        edge_to_id=edge_to_id,
                        incident_edge_ids=incident_edge_ids,
                        valid_score_cache=valid_score_cache,
                        invalidation_marks=invalidation_marks,
                        invalidation_epoch=invalidation_epoch,
                    )
                    invalidation_epoch = 1 if invalidation_epoch >= np.iinfo(np.uint32).max else invalidation_epoch + 1
                    if invalidation_epoch == 1:
                        invalidation_marks.fill(0)
                else:
                    invalidated_score_count = native_invalidated_score_count
                invalidated_score_edges = set()
            else:
                for edge in removed_round:
                    score_cache.pop(edge, None)
                if is_incremental_sequential:
                    invalidated_score_edges = _invalidated_score_edges_after_selected(
                        score_cache=score_cache,
                        edge_delta=selected_edge_delta,
                    )
                else:
                    if original_region_edge_map is None:
                        raise RuntimeError("Batch cache invalidation requires original_region_edge_map.")
                    invalidated_score_edges = _invalidated_score_edges_after_batch(
                        score_cache=score_cache,
                        removed_round=removed_round,
                        dirty_signature_nodes=dirty_signature_nodes,
                        original_region_edge_map=original_region_edge_map,
                    )
                for edge in invalidated_score_edges:
                    score_cache.pop(edge, None)
                invalidated_score_count = len(invalidated_score_edges)
        else:
            invalidated_score_edges = set()
        round_profile["cache_invalidation_runtime_sec"] = float(time.perf_counter() - timer)
        remove_edge_block_start = time.perf_counter()
        round_profile["python_incremental_state_removal_runtime_sec"] = 0.0
        round_profile["native_graph_edge_removal_runtime_sec"] = 0.0
        round_profile["active_edge_list_rebuild_runtime_sec"] = 0.0
        round_profile["batch_edge_tensor_removal_runtime_sec"] = 0.0
        if is_incremental_sequential:
            timer = time.perf_counter()
            _remove_edges_from_incremental_state(
                current_adjacency=current_adjacency,
                current_degrees=current_degrees,
                removed_round=removed_round,
            )
            round_profile["python_incremental_state_removal_runtime_sec"] = float(time.perf_counter() - timer)
            if native_graph_state is not None:
                timer = time.perf_counter()
                native_graph_state.remove_edge(int(removed_round[0][0]), int(removed_round[0][1]))
                round_profile["native_graph_edge_removal_runtime_sec"] = float(time.perf_counter() - timer)
            timer = time.perf_counter()
            removed_edge_ids = [edge_to_id[edge] for edge in removed_round]
            removed_edge_id_set = set(removed_edge_ids)
            active_edge_ids = [edge_id for edge_id in active_edge_ids if edge_id not in removed_edge_id_set]
            round_profile["active_edge_list_rebuild_runtime_sec"] = float(time.perf_counter() - timer)
        else:
            timer = time.perf_counter()
            current_edges = remove_edge_pairs(current_edges, removed_round)
            round_profile["batch_edge_tensor_removal_runtime_sec"] = float(time.perf_counter() - timer)
        round_profile["remove_edge_runtime_sec"] = float(time.perf_counter() - remove_edge_block_start)
        round_profile["active_edge_count_after"] = int(len(active_edge_ids) if is_incremental_sequential else current_edges.shape[1])
        round_profile["valid_score_cache_count_after"] = int(valid_score_cache.sum()) if is_incremental_sequential else int(len(score_cache))
        round_profile["delta_valid_cache_count_after"] = int(delta_valid_cache.sum()) if is_incremental_sequential else 0
        if profile_memory:
            round_profile["peak_rss_mb_after"] = _peak_rss_mb()

        if fast_incremental_selection:
            avg_directly_attached_size = float(round_profile.get("avg_directly_attached_size", 0.0))
            avg_four_node_pair_count = float(round_profile.get("avg_four_node_pair_count", 0.0))
        else:
            avg_directly_attached_size = 0.0 if not exact_scores else float(np.mean([score.directly_attached_size for score in exact_scores]))
            avg_four_node_pair_count = 0.0 if not exact_scores else float(np.mean([score.four_node_pair_count for score in exact_scores]))
        round_runtime_excluding_analysis_sec = float(time.perf_counter() - round_start)
        round_runtime_excluding_setup_sec = float(time.perf_counter() - round_after_setup_start)
        local_update_runtime_sec = float(local_update_summary["local_update_runtime_sec"])

        timer = time.perf_counter()
        if write_edge_scores:
            for score in exact_scores:
                analysis_rows.append(
                    {
                        "round": round_idx,
                        "u": score.edge[0],
                        "v": score.edge[1],
                        "proposal_score": score.relshift,
                        "degree_score": score.degree_score,
                        "support_score": score.support_score,
                        "region_size": score.region_size,
                        "update_size": score.update_size,
                        "mean_abs_delta_sig": score.mean_abs_delta_sig,
                        "mean_rel_delta_sig": score.mean_rel_delta_sig,
                        "mean_denom": score.mean_denom,
                        "min_denom": score.min_denom,
                        "directly_attached_size": score.directly_attached_size,
                        "four_node_pair_count": score.four_node_pair_count,
                        "selected": int(score.edge in removed_round),
                    }
                )
        round_profile["analysis_row_build_runtime_sec"] = float(time.perf_counter() - timer)
        round_runtime_sec = float(time.perf_counter() - round_start)
        round_profile["round_runtime_excluding_analysis_sec"] = round_runtime_excluding_analysis_sec
        round_profile["round_runtime_excluding_setup_sec"] = round_runtime_excluding_setup_sec
        round_profile["round_runtime_sec"] = round_runtime_sec
        cache_partition_outside_setup_sec = (
            0.0 if fast_incremental_selection else float(round_profile.get("cache_partition_runtime_sec", 0.0))
        )
        accounted_runtime_sec = (
            float(round_profile.get("round_setup_runtime_sec", 0.0))
            + cache_partition_outside_setup_sec
            + native_score_runtime_sec
            + native_scalar_refresh_runtime_sec
            + float(round_profile.get("best_selection_runtime_sec", 0.0))
            + local_update_runtime_sec
            + float(round_profile.get("dirty_signature_nodes_runtime_sec", 0.0))
            + float(round_profile.get("cache_invalidation_runtime_sec", 0.0))
            + float(round_profile.get("remove_edge_runtime_sec", 0.0))
            + float(round_profile.get("analysis_row_build_runtime_sec", 0.0))
        )
        round_profile["accounted_runtime_sec"] = accounted_runtime_sec
        round_profile["unaccounted_runtime_sec"] = max(0.0, round_runtime_sec - accounted_runtime_sec)
        round_profile["runtime_coverage_ratio"] = (
            min(1.0, accounted_runtime_sec / round_runtime_sec) if round_runtime_sec > 0.0 else 1.0
        )
        python_round_overhead_sec = (
            max(0.0, round_runtime_sec - native_score_runtime_sec - local_update_runtime_sec)
            if relshift_engine == "incremental_sequential_exact"
            else 0.0
        )

        round_summaries.append(
            {
                "round": round_idx,
                "requested_round_budget": round_budget,
                "achieved_round_budget": len(removed_round),
                "active_edge_count_before": current_edge_count,
                "eligible_count": guard_counts["eligible"],
                "blocked_by_bridge_count": guard_counts["bridge_guard"],
                "blocked_by_d_min_count": guard_counts["d_min_guard"],
                "candidate_pool_size": eligible_count,
                "selected_count": len(removed_round),
                "rescored_edge_count": len(rescored_edge_ids) if fast_incremental_selection else len(rescored_edges),
                "scalar_refreshed_edge_count": scalar_refreshed_score_count,
                "refresh_candidate_count": len(refresh_edge_ids),
                "reused_score_count": reused_score_count,
                "invalidated_score_count": invalidated_score_count,
                "bridge_count": int(round_profile.get("bridge_count", 0)),
                "native_score_runtime_sec": native_score_runtime_sec,
                "native_scalar_refresh_runtime_sec": native_scalar_refresh_runtime_sec,
                "python_round_overhead_sec": python_round_overhead_sec,
                "selected_update_runtime_sec": local_update_runtime_sec,
                "avg_directly_attached_size": avg_directly_attached_size,
                "avg_four_node_pair_count": avg_four_node_pair_count,
                "remaining_edges": len(active_edge_ids) if is_incremental_sequential else int(current_edges.shape[1]),
                "removed_edges": [[edge[0], edge[1]] for edge in removed_round],
                **local_update_summary,
                **(round_profile if collect_runtime_profile else {}),
            }
        )
        _relshift_log(
            verbose,
            f"[relshift] round={round_idx} removed={len(removed_round)} "
            f"remaining_edges={len(active_edge_ids) if is_incremental_sequential else int(current_edges.shape[1])}",
        )

    runtime = time.perf_counter() - start
    score_table_path = None
    score_table_write_runtime_sec = 0.0
    if artifact_dir is not None and write_edge_scores and analysis_rows:
        score_table_path = Path(artifact_dir) / "edge_scores.csv"
        score_table_write_start = time.perf_counter()
        _write_score_table(score_table_path, analysis_rows)
        score_table_write_runtime_sec = float(time.perf_counter() - score_table_write_start)
    total_runtime_including_score_write_sec = float(time.perf_counter() - start)
    final_active_edges = [edge_by_id[edge_id] for edge_id in active_edge_ids] if is_incremental_sequential else active_edges
    pruned_edge_index = _edges_tensor(final_active_edges) if is_incremental_sequential else current_edges

    peak_rss_end_mb = _peak_rss_mb() if profile_memory else None
    known_state_bytes = _known_profile_state_bytes(
        edge_array_by_id=edge_array_by_id,
        current_raw=current_raw,
        current_std=current_std,
        score_cache_values=score_cache_values,
        degree_score_cache_values=degree_score_cache_values,
        support_score_cache_values=support_score_cache_values,
        valid_score_cache=valid_score_cache,
        candidate_delta_cache=candidate_delta_cache,
        delta_valid_cache=delta_valid_cache,
        invalidation_marks=invalidation_marks,
    )
    runtime_summary = _build_runtime_summary(
        bundle=bundle,
        config=config,
        seed=seed,
        profile_label=profile_label,
        relshift_engine=relshift_engine,
        budget=budget,
        removed_total=removed_total,
        runtime_sec=float(runtime),
        total_runtime_including_score_write_sec=total_runtime_including_score_write_sec,
        configuration_setup_runtime_sec=configuration_setup_runtime_sec,
        extension_setup_runtime_sec=extension_setup_runtime_sec,
        initial_gdv_runtime_sec=initial_gdv_runtime_sec,
        initial_gdv_compute_info=initial_gdv_compute_info,
        standardization_runtime_sec=standardization_runtime_sec,
        initial_native_graph_state_runtime_sec=initial_native_graph_state_runtime_sec,
        pruning_state_initialization_runtime_sec=pruning_state_initialization_runtime_sec,
        score_table_write_runtime_sec=score_table_write_runtime_sec,
        round_summaries=round_summaries,
        peak_rss_start_mb=peak_rss_start_mb,
        peak_rss_end_mb=peak_rss_end_mb,
        known_state_bytes=known_state_bytes,
        native_openmp_info=native_openmp_info,
        profile_native_kernel=profile_native_kernel,
    )
    runtime_summary_path: Path | None = None
    runtime_by_round_path: Path | None = None
    runtime_profile_write_runtime_sec = 0.0
    if artifact_dir is not None and write_runtime_profile:
        runtime_summary_path, runtime_by_round_path, runtime_profile_write_runtime_sec = _write_runtime_profile_artifacts(
            artifact_dir,
            summary=runtime_summary,
            round_summaries=round_summaries,
        )

    return PruningResult(
        method="relshift",
        pruned_edge_index=pruned_edge_index,
        removed_edge_index=_edges_tensor(removed_total),
        before_edge_count=bundle.num_edges,
        after_edge_count=int(pruned_edge_index.shape[1]),
        runtime_sec=float(runtime),
        metadata={
            "budget": budget,
            "score_mode": config.score_mode,
            "guard_bridges": config.guard_bridges,
            "bridge_guard_effective": config.guard_bridges,
            "d_min": config.d_min,
            "proposal_exact": True,
            "recompute_rounds": effective_recompute_rounds,
            "configured_recompute_rounds": config.recompute_rounds,
            "candidate_update_scope": "exact_connected_graphlets_containing_edge" if relshift_engine == "incremental_sequential_exact" else "two_hop_local_orca",
            "relshift_engine": relshift_engine,
            "requested_relshift_engine": requested_relshift_engine,
            "incremental_backend": "native_cpp_extension" if relshift_engine == "incremental_sequential_exact" else "",
            "score_norm": score_norm,
            "score_node_scope": "edge_endpoints_only",
            "round_state_update_mode": "single_edge_exact_incremental" if relshift_engine == "incremental_sequential_exact" else "union_two_hop_exact_local_recount",
            "score_reuse_mode": "cache_with_local_invalidation" if use_score_cache else "full_rescore_per_round",
            "write_edge_scores": write_edge_scores,
            "verbose": verbose,
            "profile_rounds": profile_rounds,
            "profile_update_diagnostics": profile_update_diagnostics,
            "write_runtime_profile": write_runtime_profile,
            "profile_memory": profile_memory,
            "profile_label": profile_label,
            "runtime_summary_path": str(runtime_summary_path) if runtime_summary_path else None,
            "runtime_by_round_path": str(runtime_by_round_path) if runtime_by_round_path else None,
            "runtime_profile_write_runtime_sec": runtime_profile_write_runtime_sec,
            "runtime_profile": runtime_summary if write_runtime_profile else None,
            "use_native_graph_state": use_native_graph_state,
            "profile_native_kernel": profile_native_kernel,
            "native_omp_threads_requested": native_omp_threads,
            "candidate_delta_cache_mode": candidate_delta_cache_mode,
            "delta_cache_valid_count": int(delta_valid_cache.sum()) if is_incremental_sequential else 0,
            "scalar_refresh_edge_count": int(sum(int(row.get("scalar_refreshed_edge_count", 0)) for row in round_summaries)),
            "delta_impacted_full_rescore_count": int(sum(int(row.get("delta_impacted_full_rescore_count", 0)) for row in round_summaries)),
            "mixed_correction_edge_count": int(sum(int(row.get("mixed_correction_edge_count", 0)) for row in round_summaries)),
            "native_scalar_refresh_runtime_sec": float(sum(float(row.get("native_scalar_refresh_runtime_sec", 0.0)) for row in round_summaries)),
            "native_mixed_correction_runtime_sec": float(sum(float(row.get("native_mixed_correction_runtime_sec", 0.0)) for row in round_summaries)),
            "native_kernel_variant": native_kernel_variant if is_incremental_sequential else "",
            "incremental_degree_support_mode": "mutable_degree_on_demand_support" if is_incremental_sequential else "global_recompute",
            "state_update_mode": "in_place_affected_rows" if is_incremental_sequential else "copy_batch_local_rows",
            "update_diagnostics_mode": "edge_counts" if profile_update_diagnostics else "off",
            "cache_partition_mode": "native_eligibility_valid_score_split" if is_incremental_sequential and not write_edge_scores else "python_score_cache_split",
            "cache_invalidation_mode": "native_or_boolean_state_changed_incident_plus_delta_impacted" if is_incremental_sequential else "batch_original_region_and_local_neighbors",
            "native_kernel_version": _native_kernel_version(native_kernel_variant) if is_incremental_sequential else "",
            "native_selection_mode": "native_best_with_array_cache" if is_incremental_sequential and not write_edge_scores else "materialized_edge_scores",
            "native_guard_mode": "native_tarjan_eligibility" if is_incremental_sequential and not write_edge_scores else "python_guard_loop",
            "native_graph_state_mode": "persistent_dynamic_csr" if native_graph_state is not None else "per_round_python_csr",
            "csr_reuse_mode": "persistent_native_graph_state" if native_graph_state is not None else "shared_round_csr" if is_incremental_sequential and not write_edge_scores else "per_operation_csr",
            "configuration_setup_runtime_sec": configuration_setup_runtime_sec,
            "extension_setup_runtime_sec": extension_setup_runtime_sec,
            "initial_gdv_runtime_sec": initial_gdv_runtime_sec,
            "initial_gdv_cache_hit": initial_gdv_compute_info.get("cache_hit"),
            "initial_gdv_cache_path": initial_gdv_compute_info.get("cache_path"),
            "standardization_runtime_sec": standardization_runtime_sec,
            "initial_native_graph_state_runtime_sec": initial_native_graph_state_runtime_sec,
            "pruning_state_initialization_runtime_sec": pruning_state_initialization_runtime_sec,
            "peak_rss_start_mb": peak_rss_start_mb,
            "peak_rss_end_mb": peak_rss_end_mb,
            "known_numpy_state_total_bytes": known_state_bytes["known_numpy_state_total_bytes"],
            "openmp_enabled": bool(native_openmp_info.get("openmp_enabled", False)),
            "openmp_max_threads": int(native_openmp_info.get("openmp_max_threads", 1)),
            "native_edge_id_scoring": bool(native_graph_state is not None),
            "native_cached_best": bool(native_graph_state is not None),
            "native_cache_invalidation": bool(native_graph_state is not None),
            "score_table_write_runtime_sec": score_table_write_runtime_sec,
            "total_runtime_including_score_write_sec": total_runtime_including_score_write_sec,
            "requested_total_budget": budget,
            "achieved_total_budget": len(removed_total),
            "budget_shortfall": max(0, budget - len(removed_total)),
            "achieved_edge_reduction": 1.0 - (float(pruned_edge_index.shape[1]) / max(bundle.num_edges, 1)),
            "structural_guard_ceiling_hit": len(removed_total) < budget,
            "edge_level_guard_diagnostics": False,
            "round_summaries": round_summaries,
            "score_table_path": str(score_table_path) if score_table_path else None,
        },
    )


def _resolve_relshift_engine(requested_engine: str, *, round_budgets: list[int]) -> str:
    if requested_engine not in {"auto", "orca_local", "incremental_sequential_exact"}:
        raise ValueError(f"Unsupported relshift_engine: {requested_engine}")
    is_sequential = bool(round_budgets) and sum(round_budgets) > 0 and max(round_budgets) <= 1
    if requested_engine == "incremental_sequential_exact":
        if any(budget > 1 for budget in round_budgets):
            raise ValueError("relshift incremental_sequential_exact requires sequential pruning with at most one edge removed per round.")
        return "incremental_sequential_exact"
    if is_sequential:
        return "incremental_sequential_exact"
    return "orca_local"


def _relshift_log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _guard_reason(edge: tuple[int, int], degrees: np.ndarray, bridges: set[tuple[int, int]], d_min: int) -> str:
    u, v = edge
    if edge in bridges:
        return "bridge_guard"
    if degrees[u] - 1 < d_min or degrees[v] - 1 < d_min:
        return "d_min_guard"
    return "eligible"


def _select_edge_scores(scores: list[EdgeScore], *, batch_size: int, sequential: bool) -> list[EdgeScore]:
    if batch_size <= 0:
        return []
    if sequential:
        return [min(scores, key=_edge_score_key)]
    return sorted(scores, key=_edge_score_key)[:batch_size]


def _edge_score_key(score: EdgeScore) -> tuple[float, float, float]:
    return (score.relshift, score.degree_score, score.support_score)


def _native_kernel_version(native_kernel_variant: str) -> str:
    if native_kernel_variant == "mask_count_v4_combinatorial":
        return "mask_count_combinatorial_best_v4"
    raise ValueError(f"Unsupported native_kernel_variant: {native_kernel_variant}")


def _edge_support(adjacency: list[set[int]], edge: tuple[int, int]) -> int:
    u, v = edge
    if len(adjacency[u]) <= len(adjacency[v]):
        return sum(1 for node in adjacency[u] if node in adjacency[v])
    return sum(1 for node in adjacency[v] if node in adjacency[u])


def _compute_edge_score_orca(
    *,
    edge: tuple[int, int],
    num_nodes: int,
    current_edges: torch.Tensor,
    current_adjacency: list[set[int]],
    current_raw: np.ndarray,
    current_std: np.ndarray,
    degrees: np.ndarray,
    support: dict[tuple[int, int], int],
    stats,
    gdv_service: GDVService,
    score_mode: str,
    eps: float,
    cache_namespace: str,
) -> EdgeScore:
    u, v = edge
    region = [u, v]
    update_nodes = k_hop_nodes(current_adjacency, edge, hops=2)
    local_index_map = {node: idx for idx, node in enumerate(update_nodes)}
    local_current_edges = induced_subgraph_edges(update_nodes, current_adjacency)
    local_edge = (local_index_map[u], local_index_map[v])
    local_candidate_edges = remove_edge_pairs(local_current_edges, [local_edge])
    local_current_raw = gdv_service.compute_graph_gdv(
        len(update_nodes),
        local_current_edges,
        cache_namespace=f"{cache_namespace}_current_local",
    )
    local_candidate_raw = gdv_service.compute_graph_gdv(
        len(update_nodes),
        local_candidate_edges,
        cache_namespace=f"{cache_namespace}_minus_edge_local",
    )
    local_delta_raw = local_candidate_raw - local_current_raw

    candidate_region_std = current_std[region].copy()
    overlap_positions = [idx for idx, node in enumerate(region) if node in local_index_map]
    if overlap_positions:
        overlap_nodes = [region[idx] for idx in overlap_positions]
        overlap_local = [local_index_map[node] for node in overlap_nodes]
        candidate_overlap_raw = current_raw[overlap_nodes] + local_delta_raw[overlap_local]
        candidate_region_std[overlap_positions] = apply_standardization(candidate_overlap_raw, stats)

    delta = np.abs(current_std[region] - candidate_region_std).sum(axis=1)
    denom = np.abs(current_std[region]).sum(axis=1) + eps
    relshift = float(np.mean(delta if score_mode == "absolute" else delta / denom))
    degree_score = float(degrees[u] + degrees[v])
    support_score = float(support.get(edge, 0))
    mean_abs_delta_sig = float(np.mean(delta))
    mean_rel_delta_sig = float(np.mean(delta / denom))
    mean_denom = float(np.mean(denom))
    min_denom = float(np.min(denom))

    return EdgeScore(
        edge=edge,
        relshift=relshift,
        degree_score=degree_score,
        support_score=support_score,
        region_size=len(region),
        update_size=len(update_nodes),
        mean_abs_delta_sig=mean_abs_delta_sig,
        mean_rel_delta_sig=mean_rel_delta_sig,
        mean_denom=mean_denom,
        min_denom=min_denom,
        update_nodes=tuple(update_nodes),
        update_node_set=frozenset(update_nodes),
        directly_attached_size=len((current_adjacency[u] | current_adjacency[v]) - {u, v}),
        four_node_pair_count=0,
        raw_delta=None,
    )


def _compute_edge_scores_incremental_round(
    *,
    rescored_edges: list[tuple[int, int]],
    current_adjacency: list[set[int]],
    current_raw: np.ndarray,
    current_std: np.ndarray,
    degrees: np.ndarray,
    support: dict[tuple[int, int], int] | None,
    stats,
    score_mode: str,
    eps: float,
    include_update_nodes: bool,
    include_update_sizes: bool = True,
) -> tuple[list[EdgeScore], float, dict[str, float]]:
    extension = _require_incremental_extension()
    profile: dict[str, float] = {}
    timer = time.perf_counter()
    row_ptr, col_idx = _adjacency_to_csr(current_adjacency)
    profile["score_csr_runtime_sec"] = float(time.perf_counter() - timer)
    timer = time.perf_counter()
    candidate_edges = np.asarray(rescored_edges, dtype=np.int64)
    profile["score_candidate_array_runtime_sec"] = float(time.perf_counter() - timer)
    extension_start = time.perf_counter()
    ext_result = extension.score_edges_round(
        row_ptr,
        col_idx,
        candidate_edges,
        current_raw,
        current_std,
        stats.mean,
        stats.std,
        score_mode,
        float(eps),
        bool(include_update_sizes),
    )
    extension_runtime_sec = float(time.perf_counter() - extension_start)
    timer = time.perf_counter()
    scores = np.asarray(ext_result["scores"], dtype=np.float64)
    mean_abs_delta = np.asarray(ext_result["mean_abs_delta_sig"], dtype=np.float64)
    mean_rel_delta = np.asarray(ext_result["mean_rel_delta_sig"], dtype=np.float64)
    mean_denom = np.asarray(ext_result["mean_denom"], dtype=np.float64)
    min_denom = np.asarray(ext_result["min_denom"], dtype=np.float64)
    update_sizes = np.asarray(ext_result["update_sizes"], dtype=np.int64)
    directly_attached_sizes = np.asarray(ext_result["directly_attached_sizes"], dtype=np.int64)
    four_node_pair_counts = np.asarray(ext_result["four_node_pair_counts"], dtype=np.int64)
    profile["score_result_array_runtime_sec"] = float(time.perf_counter() - timer)

    computed_scores: list[EdgeScore] = []
    materialize_start = time.perf_counter()
    update_nodes_runtime_sec = 0.0
    for idx, edge in enumerate(rescored_edges):
        if include_update_nodes:
            timer = time.perf_counter()
            update_nodes = tuple(k_hop_nodes(current_adjacency, edge, hops=2))
            update_nodes_runtime_sec += float(time.perf_counter() - timer)
        else:
            update_nodes = tuple()
        computed_scores.append(
            EdgeScore(
                edge=edge,
                relshift=float(scores[idx]),
                degree_score=float(degrees[edge[0]] + degrees[edge[1]]),
                support_score=float(support.get(edge, 0) if support is not None else _edge_support(current_adjacency, edge)),
                region_size=2,
                update_size=int(update_sizes[idx]),
                mean_abs_delta_sig=float(mean_abs_delta[idx]),
                mean_rel_delta_sig=float(mean_rel_delta[idx]),
                mean_denom=float(mean_denom[idx]),
                min_denom=float(min_denom[idx]),
                update_nodes=update_nodes,
                update_node_set=frozenset(update_nodes),
                directly_attached_size=int(directly_attached_sizes[idx]),
                four_node_pair_count=int(four_node_pair_counts[idx]),
                raw_delta=None,
            )
        )
    profile["score_update_nodes_runtime_sec"] = update_nodes_runtime_sec
    profile["score_object_materialization_runtime_sec"] = float(time.perf_counter() - materialize_start)
    return computed_scores, extension_runtime_sec, profile


def _score_incremental_round_best(
    *,
    rescored_edge_ids: list[int],
    edge_array_by_id: np.ndarray,
    current_adjacency: list[set[int]],
    current_raw: np.ndarray,
    current_std: np.ndarray,
    stats,
    score_mode: str,
    eps: float,
    score_cache_values: np.ndarray,
    degree_score_cache_values: np.ndarray,
    support_score_cache_values: np.ndarray,
    valid_score_cache: np.ndarray,
    candidate_delta_cache: np.ndarray | None = None,
    delta_valid_cache: np.ndarray | None = None,
    profile_native_kernel: bool,
    native_kernel_variant: str,
    native_graph_state=None,
    prebuilt_csr: tuple[np.ndarray, np.ndarray] | None = None,
    prebuilt_csr_runtime_sec: float = 0.0,
) -> tuple[int | None, float, dict[str, float]]:
    extension = _require_incremental_extension()
    profile: dict[str, float] = {}
    if native_graph_state is not None:
        row_ptr = None
        col_idx = None
        profile["score_csr_runtime_sec"] = 0.0
    elif prebuilt_csr is None:
        timer = time.perf_counter()
        row_ptr, col_idx = _adjacency_to_csr(current_adjacency)
        profile["score_csr_runtime_sec"] = float(time.perf_counter() - timer)
    else:
        row_ptr, col_idx = prebuilt_csr
        profile["score_csr_runtime_sec"] = float(prebuilt_csr_runtime_sec)
    timer = time.perf_counter()
    candidate_edge_ids = np.asarray(rescored_edge_ids, dtype=np.int64)
    candidate_edges = None if native_graph_state is not None and hasattr(native_graph_state, "score_edge_ids_round_best") else edge_array_by_id[candidate_edge_ids]
    profile["score_candidate_array_runtime_sec"] = float(time.perf_counter() - timer)
    extension_start = time.perf_counter()
    if native_graph_state is not None and hasattr(native_graph_state, "score_edge_ids_round_best"):
        ext_result = native_graph_state.score_edge_ids_round_best(
            candidate_edge_ids,
            current_raw,
            current_std,
            stats.mean,
            stats.std,
            score_mode,
            float(eps),
            score_cache_values,
            degree_score_cache_values,
            support_score_cache_values,
            valid_score_cache,
            native_kernel_variant,
            bool(profile_native_kernel),
            candidate_delta_cache if candidate_delta_cache is not None else None,
            delta_valid_cache if delta_valid_cache is not None else None,
        )
    elif native_graph_state is not None:
        ext_result = native_graph_state.score_edges_round_best(
            candidate_edges,
            candidate_edge_ids,
            current_raw,
            current_std,
            stats.mean,
            stats.std,
            score_mode,
            float(eps),
            score_cache_values,
            degree_score_cache_values,
            support_score_cache_values,
            valid_score_cache,
            native_kernel_variant,
            bool(profile_native_kernel),
            candidate_delta_cache if candidate_delta_cache is not None else None,
            delta_valid_cache if delta_valid_cache is not None else None,
        )
    else:
        ext_result = extension.score_edges_round_best(
            row_ptr,
            col_idx,
            candidate_edges,
            candidate_edge_ids,
            current_raw,
            current_std,
            stats.mean,
            stats.std,
            score_mode,
            float(eps),
            score_cache_values,
            degree_score_cache_values,
            support_score_cache_values,
            valid_score_cache,
            native_kernel_variant,
            bool(profile_native_kernel),
            candidate_delta_cache if candidate_delta_cache is not None else None,
            delta_valid_cache if delta_valid_cache is not None else None,
        )
    extension_runtime_sec = float(time.perf_counter() - extension_start)
    timer = time.perf_counter()
    best_edge_id = int(ext_result["best_edge_id"])
    profile["score_result_array_runtime_sec"] = float(time.perf_counter() - timer)
    profile["score_update_nodes_runtime_sec"] = 0.0
    profile["score_object_materialization_runtime_sec"] = 0.0
    profile["avg_directly_attached_size"] = float(ext_result.get("avg_directly_attached_size", 0.0))
    profile["avg_four_node_pair_count"] = float(ext_result.get("avg_four_node_pair_count", 0.0))
    profile["native_pair_generation_runtime_sec"] = float(ext_result.get("native_pair_generation_runtime_sec", 0.0))
    profile["native_delta_accumulation_runtime_sec"] = float(ext_result.get("native_delta_accumulation_runtime_sec", 0.0))
    profile["native_score_scalarization_runtime_sec"] = float(ext_result.get("native_score_scalarization_runtime_sec", 0.0))
    return (best_edge_id if best_edge_id >= 0 else None), extension_runtime_sec, profile


def _refresh_incremental_scores_from_delta_cache(
    *,
    refresh_edge_ids: list[int],
    current_raw: np.ndarray,
    current_std: np.ndarray,
    stats,
    score_mode: str,
    eps: float,
    candidate_delta_cache: np.ndarray,
    delta_valid_cache: np.ndarray,
    score_cache_values: np.ndarray,
    degree_score_cache_values: np.ndarray,
    support_score_cache_values: np.ndarray,
    valid_score_cache: np.ndarray,
    native_graph_state=None,
) -> tuple[int | None, float, dict[str, float]]:
    if not refresh_edge_ids:
        return None, 0.0, {"native_scalar_refresh_runtime_sec": 0.0}
    if native_graph_state is None or not hasattr(native_graph_state, "refresh_scores_from_delta_cache"):
        raise RuntimeError("candidate delta cache refresh requires NativeGraphState.refresh_scores_from_delta_cache.")
    timer = time.perf_counter()
    result = native_graph_state.refresh_scores_from_delta_cache(
        np.asarray(refresh_edge_ids, dtype=np.int64),
        current_raw,
        current_std,
        stats.mean,
        stats.std,
        score_mode,
        float(eps),
        candidate_delta_cache,
        delta_valid_cache,
        score_cache_values,
        degree_score_cache_values,
        support_score_cache_values,
        valid_score_cache,
    )
    runtime_sec = float(time.perf_counter() - timer)
    best_edge_id = int(result.get("best_edge_id", -1))
    return (
        best_edge_id if best_edge_id >= 0 else None,
        runtime_sec,
        {
            "native_scalar_refresh_runtime_sec": runtime_sec,
            "scalar_refreshed_edge_count": float(result.get("refreshed_count", len(refresh_edge_ids))),
        },
    )


def _best_cached_edge_id(
    edge_ids: list[int],
    *,
    score_cache_values: np.ndarray,
    degree_score_cache_values: np.ndarray,
    support_score_cache_values: np.ndarray,
) -> int | None:
    best_edge_id: int | None = None
    best_key: tuple[float, float, float, int] | None = None
    for edge_id in edge_ids:
        key = _edge_id_score_key(
            edge_id,
            score_cache_values=score_cache_values,
            degree_score_cache_values=degree_score_cache_values,
            support_score_cache_values=support_score_cache_values,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_edge_id = edge_id
    return best_edge_id


def _merge_best_edge_ids(
    left: int | None,
    right: int | None,
    *,
    score_cache_values: np.ndarray,
    degree_score_cache_values: np.ndarray,
    support_score_cache_values: np.ndarray,
) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    left_key = _edge_id_score_key(
        left,
        score_cache_values=score_cache_values,
        degree_score_cache_values=degree_score_cache_values,
        support_score_cache_values=support_score_cache_values,
    )
    right_key = _edge_id_score_key(
        right,
        score_cache_values=score_cache_values,
        degree_score_cache_values=degree_score_cache_values,
        support_score_cache_values=support_score_cache_values,
    )
    return left if left_key <= right_key else right


def _edge_id_score_key(
    edge_id: int,
    *,
    score_cache_values: np.ndarray,
    degree_score_cache_values: np.ndarray,
    support_score_cache_values: np.ndarray,
) -> tuple[float, float, float, int]:
    return (
        float(score_cache_values[edge_id]),
        float(degree_score_cache_values[edge_id]),
        float(support_score_cache_values[edge_id]),
        int(edge_id),
    )


def _write_score_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _edges_tensor(edges: list[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _require_incremental_extension():
    from ._incremental_ext import require_incremental_extension

    return require_incremental_extension()


def _adjacency_to_csr(current_adjacency: list[set[int]]) -> tuple[np.ndarray, np.ndarray]:
    row_ptr = np.zeros(len(current_adjacency) + 1, dtype=np.int64)
    columns: list[int] = []
    offset = 0
    for idx, neighbors in enumerate(current_adjacency):
        sorted_neighbors = sorted(neighbors)
        columns.extend(sorted_neighbors)
        offset += len(sorted_neighbors)
        row_ptr[idx + 1] = offset
    return row_ptr, np.asarray(columns, dtype=np.int64)


def _create_native_graph_state(row_ptr: np.ndarray, col_idx: np.ndarray, edge_array_by_id: np.ndarray | None = None):
    extension = _require_incremental_extension()
    graph_state_cls = getattr(extension, "NativeGraphState", None)
    if graph_state_cls is None:
        return None
    if edge_array_by_id is None:
        return graph_state_cls(np.asarray(row_ptr, dtype=np.int64), np.asarray(col_idx, dtype=np.int64))
    return graph_state_cls(
        np.asarray(row_ptr, dtype=np.int64),
        np.asarray(col_idx, dtype=np.int64),
        np.asarray(edge_array_by_id, dtype=np.int64),
    )


def _native_openmp_info() -> dict[str, object]:
    extension = _require_incremental_extension()
    if hasattr(extension, "openmp_info"):
        result = extension.openmp_info()
        return {
            "openmp_enabled": bool(result.get("openmp_enabled", False)),
            "openmp_max_threads": int(result.get("openmp_max_threads", 1)),
        }
    return {"openmp_enabled": False, "openmp_max_threads": 1}


def _set_native_openmp_threads(thread_count: int) -> dict[str, object]:
    extension = _require_incremental_extension()
    if not hasattr(extension, "set_openmp_threads"):
        raise RuntimeError("RelShift native extension does not expose set_openmp_threads; rebuild the extension.")
    result = extension.set_openmp_threads(int(thread_count))
    return {
        "openmp_enabled": bool(result.get("openmp_enabled", False)),
        "openmp_max_threads": int(result.get("openmp_max_threads", 1)),
    }


def _eligible_edge_id_partitions_incremental_native(
    *,
    row_ptr: np.ndarray | None,
    col_idx: np.ndarray | None,
    active_edge_ids: list[int],
    edge_array_by_id: np.ndarray,
    degrees: np.ndarray,
    d_min: int,
    guard_bridges: bool,
    valid_score_cache: np.ndarray,
    delta_valid_cache: np.ndarray | None = None,
    use_score_cache: bool,
    native_graph_state=None,
    score_cache_values: np.ndarray | None = None,
    degree_score_cache_values: np.ndarray | None = None,
    support_score_cache_values: np.ndarray | None = None,
) -> dict[str, object]:
    extension = _require_incremental_extension()
    if native_graph_state is not None and hasattr(native_graph_state, "eligible_edge_id_partitions_with_cached_best"):
        result = native_graph_state.eligible_edge_id_partitions_with_cached_best(
            np.asarray(active_edge_ids, dtype=np.int64),
            np.asarray(degrees, dtype=np.int64),
            int(d_min),
            bool(guard_bridges),
            np.asarray(valid_score_cache, dtype=np.uint8),
            bool(use_score_cache),
            score_cache_values if score_cache_values is not None else None,
            degree_score_cache_values if degree_score_cache_values is not None else None,
            support_score_cache_values if support_score_cache_values is not None else None,
            delta_valid_cache if delta_valid_cache is not None else None,
        )
    elif native_graph_state is not None:
        result = native_graph_state.eligible_edge_id_partitions(
            np.asarray(active_edge_ids, dtype=np.int64),
            edge_array_by_id,
            np.asarray(degrees, dtype=np.int64),
            int(d_min),
            bool(guard_bridges),
            np.asarray(valid_score_cache, dtype=np.uint8),
            bool(use_score_cache),
        )
    elif hasattr(extension, "eligible_edge_id_partitions_from_csr"):
        if row_ptr is None or col_idx is None:
            raise RuntimeError("CSR arrays are required when native graph state is unavailable.")
        result = extension.eligible_edge_id_partitions_from_csr(
            row_ptr,
            col_idx,
            np.asarray(active_edge_ids, dtype=np.int64),
            edge_array_by_id,
            np.asarray(degrees, dtype=np.int64),
            int(d_min),
            bool(guard_bridges),
            np.asarray(valid_score_cache, dtype=np.uint8),
            bool(use_score_cache),
        )
    else:
        if row_ptr is None or col_idx is None:
            raise RuntimeError("CSR arrays are required when native graph state is unavailable.")
        result = None

    if result is None:
        result = _eligible_edge_ids_incremental_native(
            row_ptr=row_ptr,
            col_idx=col_idx,
            active_edge_ids=active_edge_ids,
            edge_array_by_id=edge_array_by_id,
            degrees=degrees,
            d_min=d_min,
            guard_bridges=guard_bridges,
        )
        eligible_edge_ids = result["eligible_edge_ids"]
        rescored_edge_ids: list[int] = []
        reused_edge_ids: list[int] = []
        refresh_edge_ids: list[int] = []
        for edge_id in eligible_edge_ids:
            if use_score_cache and bool(valid_score_cache[edge_id]):
                reused_edge_ids.append(edge_id)
            elif use_score_cache and delta_valid_cache is not None and bool(delta_valid_cache[edge_id]):
                refresh_edge_ids.append(edge_id)
            else:
                rescored_edge_ids.append(edge_id)
        return {
            **result,
            "rescored_edge_ids": rescored_edge_ids,
            "reused_edge_ids": reused_edge_ids,
            "refresh_edge_ids": refresh_edge_ids,
            "rescored_count": len(rescored_edge_ids),
            "reused_count": len(reused_edge_ids),
            "refresh_count": len(refresh_edge_ids),
            "cache_partition_runtime_sec": 0.0,
            "cached_best_edge_id": -1,
        }

    eligible_edge_ids = np.asarray(result["eligible_edge_ids"], dtype=np.int64).tolist()
    rescored_edge_ids = np.asarray(result["rescored_edge_ids"], dtype=np.int64).tolist()
    reused_edge_ids = np.asarray(result["reused_edge_ids"], dtype=np.int64).tolist()
    refresh_edge_ids = np.asarray(result.get("refresh_edge_ids", np.empty(0, dtype=np.int64)), dtype=np.int64).tolist()
    return {
        "eligible_edge_ids": [int(edge_id) for edge_id in eligible_edge_ids],
        "rescored_edge_ids": [int(edge_id) for edge_id in rescored_edge_ids],
        "reused_edge_ids": [int(edge_id) for edge_id in reused_edge_ids],
        "refresh_edge_ids": [int(edge_id) for edge_id in refresh_edge_ids],
        "eligible_count": int(result["eligible_count"]),
        "rescored_count": int(result["rescored_count"]),
        "reused_count": int(result["reused_count"]),
        "refresh_count": int(result.get("refresh_count", len(refresh_edge_ids))),
        "blocked_by_bridge_count": int(result["blocked_by_bridge_count"]),
        "blocked_by_d_min_count": int(result["blocked_by_d_min_count"]),
        "bridge_count": int(result.get("bridge_count", 0)),
        "bridge_runtime_sec": float(result["bridge_runtime_sec"]),
        "eligibility_runtime_sec": float(result["eligibility_runtime_sec"]),
        "cache_partition_runtime_sec": float(result.get("cache_partition_runtime_sec", 0.0)),
        "cached_best_edge_id": int(result.get("cached_best_edge_id", -1)),
    }


def _eligible_edge_ids_incremental_native(
    *,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    active_edge_ids: list[int],
    edge_array_by_id: np.ndarray,
    degrees: np.ndarray,
    d_min: int,
    guard_bridges: bool,
) -> dict[str, object]:
    extension = _require_incremental_extension()
    result = extension.eligible_edge_ids_from_csr(
        row_ptr,
        col_idx,
        np.asarray(active_edge_ids, dtype=np.int64),
        edge_array_by_id,
        np.asarray(degrees, dtype=np.int64),
        int(d_min),
        bool(guard_bridges),
    )
    eligible_edge_ids = np.asarray(result["eligible_edge_ids"], dtype=np.int64).tolist()
    return {
        "eligible_edge_ids": [int(edge_id) for edge_id in eligible_edge_ids],
        "eligible_count": int(result["eligible_count"]),
        "blocked_by_bridge_count": int(result["blocked_by_bridge_count"]),
        "blocked_by_d_min_count": int(result["blocked_by_d_min_count"]),
        "bridge_count": int(result.get("bridge_count", 0)),
        "bridge_runtime_sec": float(result["bridge_runtime_sec"]),
        "eligibility_runtime_sec": float(result["eligibility_runtime_sec"]),
    }


def _compute_selected_edge_delta_incremental(
    current_adjacency: list[set[int]],
    edge: tuple[int, int],
    *,
    native_graph_state=None,
    prebuilt_csr: tuple[np.ndarray, np.ndarray] | None = None,
    selected_edge_id: int | None = None,
    valid_score_cache: np.ndarray | None = None,
    delta_valid_cache: np.ndarray | None = None,
    candidate_delta_cache: np.ndarray | None = None,
    update_candidate_delta_cache: bool = False,
    invalidate_cache: bool = False,
) -> tuple[IncrementalEdgeDelta, int | None, dict[str, float | int]]:
    extension = _require_incremental_extension()
    native_invalidated_count: int | None = None
    native_cache_update_profile: dict[str, float | int] = {}
    if (
        native_graph_state is not None
        and invalidate_cache
        and selected_edge_id is not None
        and valid_score_cache is not None
        and update_candidate_delta_cache
        and delta_valid_cache is not None
        and candidate_delta_cache is not None
        and hasattr(native_graph_state, "compute_selected_edge_delta_and_update_candidate_cache")
    ):
        result = native_graph_state.compute_selected_edge_delta_and_update_candidate_cache(
            int(selected_edge_id),
            valid_score_cache,
            delta_valid_cache,
            candidate_delta_cache,
        )
        native_invalidated_count = int(result.get("invalidated_count", 0))
        native_cache_update_profile = {
            "mixed_correction_edge_count": int(result.get("mixed_correction_edge_count", 0)),
            "delta_impacted_full_rescore_count": int(result.get("delta_impacted_full_rescore_count", 0)),
            "native_mixed_correction_runtime_sec": float(result.get("native_mixed_correction_runtime_sec", 0.0)),
        }
    elif (
        native_graph_state is not None
        and invalidate_cache
        and selected_edge_id is not None
        and valid_score_cache is not None
        and hasattr(native_graph_state, "compute_selected_edge_delta_and_invalidate")
    ):
        result = native_graph_state.compute_selected_edge_delta_and_invalidate(int(selected_edge_id), valid_score_cache)
        native_invalidated_count = int(result.get("invalidated_count", 0))
    elif native_graph_state is not None:
        result = native_graph_state.compute_selected_edge_delta(int(edge[0]), int(edge[1]))
    elif prebuilt_csr is None:
        row_ptr, col_idx = _adjacency_to_csr(current_adjacency)
        result = extension.compute_selected_edge_delta(row_ptr, col_idx, int(edge[0]), int(edge[1]))
    else:
        row_ptr, col_idx = prebuilt_csr
        result = extension.compute_selected_edge_delta(row_ptr, col_idx, int(edge[0]), int(edge[1]))
    affected_nodes = tuple(int(node) for node in np.asarray(result["affected_nodes"], dtype=np.int64).tolist())
    raw_delta = np.asarray(result["raw_delta"], dtype=np.float64)
    impacted_edges_array = np.asarray(result.get("impacted_edges", np.empty((0, 2), dtype=np.int64)), dtype=np.int64).reshape((-1, 2))
    impacted_edges = tuple((int(row[0]), int(row[1])) for row in impacted_edges_array.tolist())
    return IncrementalEdgeDelta(
        edge=(int(edge[0]), int(edge[1])),
        affected_nodes=affected_nodes,
        raw_delta=raw_delta,
        impacted_edges=impacted_edges,
    ), native_invalidated_count, native_cache_update_profile


def _apply_batch_local_update(
    *,
    num_nodes: int,
    removed_round: list[tuple[int, int]],
    current_adjacency: list[set[int]],
    current_raw: np.ndarray,
    current_std: np.ndarray,
    stats,
    gdv_service: GDVService,
    cache_namespace: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    update_nodes = _batch_update_nodes(current_adjacency, removed_round)
    if not update_nodes:
        return current_raw, current_std, {
            "update_union_size": 0,
            "update_union_edge_count_before": 0,
            "update_union_edge_count_after": 0,
        }

    local_index_map = {node: idx for idx, node in enumerate(update_nodes)}
    local_current_edges = induced_subgraph_edges(update_nodes, current_adjacency)
    local_removed_edges = [(local_index_map[u], local_index_map[v]) for u, v in removed_round]
    local_candidate_edges = remove_edge_pairs(local_current_edges, local_removed_edges)
    local_current_raw = gdv_service.compute_graph_gdv(
        len(update_nodes),
        local_current_edges,
        cache_namespace=f"{cache_namespace}_current_local",
    )
    local_candidate_raw = gdv_service.compute_graph_gdv(
        len(update_nodes),
        local_candidate_edges,
        cache_namespace=f"{cache_namespace}_minus_batch_local",
    )
    local_delta_raw = local_candidate_raw - local_current_raw

    next_raw = current_raw.copy()
    next_std = current_std.copy()
    next_raw[update_nodes] = current_raw[update_nodes] + local_delta_raw
    next_std[update_nodes] = apply_standardization(next_raw[update_nodes], stats)
    return next_raw, next_std, {
        "update_union_size": len(update_nodes),
        "update_union_edge_count_before": int(local_current_edges.shape[1]),
        "update_union_edge_count_after": int(local_candidate_edges.shape[1]),
    }


def _apply_single_edge_incremental_update(
    *,
    current_adjacency: list[set[int]],
    current_raw: np.ndarray,
    current_std: np.ndarray,
    edge_delta,
    stats,
    profile_update_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    update_nodes = list(edge_delta.affected_nodes)
    if not update_nodes:
        return current_raw, current_std, {
            "update_union_size": 0,
            "update_union_edge_count_before": 0,
            "update_union_edge_count_after": 0,
        }

    update_index = np.asarray(update_nodes, dtype=np.int64)
    current_raw[update_index] = np.maximum(current_raw[update_index] + edge_delta.raw_delta, 0.0)
    current_std[update_index] = apply_standardization(current_raw[update_index], stats)

    edge_count_before = 0
    edge_count_after = 0
    if profile_update_diagnostics:
        local_current_edges = induced_subgraph_edges(update_nodes, current_adjacency)
        local_index_map = {node: idx for idx, node in enumerate(update_nodes)}
        local_candidate_edges = remove_edge_pairs(local_current_edges, [(local_index_map[edge_delta.edge[0]], local_index_map[edge_delta.edge[1]])])
        edge_count_before = int(local_current_edges.shape[1])
        edge_count_after = int(local_candidate_edges.shape[1])

    return current_raw, current_std, {
        "update_union_size": len(update_nodes),
        "update_union_edge_count_before": edge_count_before,
        "update_union_edge_count_after": edge_count_after,
    }


def _batch_update_nodes(current_adjacency: list[set[int]], removed_round: list[tuple[int, int]]) -> list[int]:
    update_node_set: set[int] = set()
    for edge in removed_round:
        update_node_set.update(k_hop_nodes(current_adjacency, edge, hops=2))
    return sorted(update_node_set)


def _build_original_region_edge_map(
    *,
    original_edges: list[tuple[int, int]],
    num_nodes: int,
) -> list[set[tuple[int, int]]]:
    node_to_edges = [set() for _ in range(num_nodes)]
    for edge in original_edges:
        u, v = edge
        node_to_edges[u].add(edge)
        node_to_edges[v].add(edge)
    return node_to_edges


def _invalidated_score_edges_after_batch(
    *,
    score_cache: dict[tuple[int, int], EdgeScore],
    removed_round: list[tuple[int, int]],
    dirty_signature_nodes: list[int],
    original_region_edge_map: list[set[tuple[int, int]]],
) -> set[tuple[int, int]]:
    invalidated: set[tuple[int, int]] = set()
    removed_endpoint_nodes = {node for edge in removed_round for node in edge}
    for node in dirty_signature_nodes:
        invalidated.update(original_region_edge_map[node])

    for edge, cached_score in score_cache.items():
        u, v = edge
        if u in removed_endpoint_nodes or v in removed_endpoint_nodes:
            invalidated.add(edge)
            continue
        for removed_edge in removed_round:
            x, y = removed_edge
            if x in cached_score.update_node_set and y in cached_score.update_node_set:
                invalidated.add(edge)
                break
    return invalidated


def _invalidated_score_edges_after_selected(
    *,
    score_cache: dict[tuple[int, int], EdgeScore],
    edge_delta: IncrementalEdgeDelta,
) -> set[tuple[int, int]]:
    changed_nodes = set(_changed_signature_nodes_from_edge_delta(edge_delta))
    impacted_edges = set(edge_delta.impacted_edges)
    if not impacted_edges:
        affected_nodes = set(edge_delta.affected_nodes)
        return {edge for edge in score_cache if edge[0] in affected_nodes or edge[1] in affected_nodes}
    return {
        edge
        for edge in score_cache
        if edge[0] in changed_nodes or edge[1] in changed_nodes or edge in impacted_edges
    }


def _invalidate_edge_ids_after_selected(
    *,
    selected_edge_id: int,
    edge_delta: IncrementalEdgeDelta,
    edge_to_id: dict[tuple[int, int], int],
    incident_edge_ids: list[list[int]] | list[set[int]],
    valid_score_cache: np.ndarray,
    invalidation_marks: np.ndarray,
    invalidation_epoch: int,
) -> int:
    invalidated_count = 0

    def invalidate(edge_id: int) -> None:
        nonlocal invalidated_count
        if edge_id == selected_edge_id:
            return
        if invalidation_marks[edge_id] == invalidation_epoch:
            return
        invalidation_marks[edge_id] = invalidation_epoch
        if bool(valid_score_cache[edge_id]):
            valid_score_cache[edge_id] = 0
            invalidated_count += 1

    for node in _changed_signature_nodes_from_edge_delta(edge_delta):
        for edge_id in incident_edge_ids[node]:
            invalidate(int(edge_id))
    if edge_delta.impacted_edges:
        for edge in edge_delta.impacted_edges:
            edge_id = edge_to_id.get(edge)
            if edge_id is not None:
                invalidate(edge_id)
    else:
        # Compatibility fallback for test doubles or stale native builds: preserve the old
        # conservative rule rather than risk a false negative.
        for node in edge_delta.affected_nodes:
            for edge_id in incident_edge_ids[node]:
                invalidate(int(edge_id))
    return invalidated_count


def _invalidated_edge_ids_after_selected(
    *,
    selected_edge_id: int,
    edge_delta: IncrementalEdgeDelta,
    edge_to_id: dict[tuple[int, int], int],
    active_incident_edge_ids: list[set[int]] | list[list[int]],
    valid_score_cache: np.ndarray,
) -> set[int]:
    invalidated: set[int] = set()
    for node in _changed_signature_nodes_from_edge_delta(edge_delta):
        invalidated.update(active_incident_edge_ids[node])
    if edge_delta.impacted_edges:
        for edge in edge_delta.impacted_edges:
            edge_id = edge_to_id.get(edge)
            if edge_id is not None:
                invalidated.add(edge_id)
    else:
        # Compatibility fallback for test doubles or stale native builds: preserve the old
        # conservative rule rather than risk a false negative.
        affected_nodes = set(edge_delta.affected_nodes)
        for node in affected_nodes:
            invalidated.update(active_incident_edge_ids[node])
    invalidated.discard(selected_edge_id)
    return {edge_id for edge_id in invalidated if bool(valid_score_cache[edge_id])}


def _changed_signature_nodes_from_edge_delta(edge_delta: IncrementalEdgeDelta) -> list[int]:
    if edge_delta.raw_delta.size == 0:
        return []
    changed_mask = np.any(edge_delta.raw_delta != 0.0, axis=1)
    return [node for node, changed in zip(edge_delta.affected_nodes, changed_mask.tolist(), strict=True) if changed]


def _remove_edges_from_incremental_state(
    *,
    current_adjacency: list[set[int]],
    current_degrees: np.ndarray,
    removed_round: list[tuple[int, int]],
) -> None:
    for u, v in removed_round:
        if v in current_adjacency[u]:
            current_adjacency[u].remove(v)
            current_degrees[u] -= 1
        if u in current_adjacency[v]:
            current_adjacency[v].remove(u)
            current_degrees[v] -= 1


def _peak_rss_mb() -> float | None:
    """Return process peak resident memory in MiB when the platform exposes it."""
    try:
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    # Linux reports KiB; macOS/BSD report bytes.
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _cpu_model_name() -> str:
    cpu_model = platform.processor().strip()
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return cpu_model or "unknown"


def _sum_round_metric(round_summaries: list[dict[str, object]], key: str) -> float:
    total = 0.0
    for row in round_summaries:
        value = row.get(key, 0.0)
        if isinstance(value, (int, float, np.integer, np.floating)):
            total += float(value)
    return total


def _mean_round_metric(round_summaries: list[dict[str, object]], key: str) -> float:
    values = [
        float(row[key])
        for row in round_summaries
        if isinstance(row.get(key), (int, float, np.integer, np.floating))
    ]
    return float(np.mean(values)) if values else 0.0


def _known_profile_state_bytes(
    *,
    edge_array_by_id: np.ndarray,
    current_raw: np.ndarray,
    current_std: np.ndarray,
    score_cache_values: np.ndarray,
    degree_score_cache_values: np.ndarray,
    support_score_cache_values: np.ndarray,
    valid_score_cache: np.ndarray,
    candidate_delta_cache: np.ndarray,
    delta_valid_cache: np.ndarray,
    invalidation_marks: np.ndarray,
) -> dict[str, int]:
    components = {
        "edge_array_by_id_bytes": int(edge_array_by_id.nbytes),
        "current_raw_bytes": int(current_raw.nbytes),
        "current_std_bytes": int(current_std.nbytes),
        "score_cache_bytes": int(score_cache_values.nbytes),
        "degree_score_cache_bytes": int(degree_score_cache_values.nbytes),
        "support_score_cache_bytes": int(support_score_cache_values.nbytes),
        "valid_score_mask_bytes": int(valid_score_cache.nbytes),
        "candidate_delta_cache_bytes": int(candidate_delta_cache.nbytes),
        "delta_valid_mask_bytes": int(delta_valid_cache.nbytes),
        "invalidation_marks_bytes": int(invalidation_marks.nbytes),
    }
    components["known_numpy_state_total_bytes"] = int(sum(components.values()))
    return components


def _build_runtime_summary(
    *,
    bundle: DatasetBundle,
    config: PruningConfig,
    seed: int,
    profile_label: str,
    relshift_engine: str,
    budget: int,
    removed_total: list[tuple[int, int]],
    runtime_sec: float,
    total_runtime_including_score_write_sec: float,
    configuration_setup_runtime_sec: float,
    extension_setup_runtime_sec: float,
    initial_gdv_runtime_sec: float,
    initial_gdv_compute_info: dict[str, object],
    standardization_runtime_sec: float,
    initial_native_graph_state_runtime_sec: float,
    pruning_state_initialization_runtime_sec: float,
    score_table_write_runtime_sec: float,
    round_summaries: list[dict[str, object]],
    peak_rss_start_mb: float | None,
    peak_rss_end_mb: float | None,
    known_state_bytes: dict[str, int],
    native_openmp_info: dict[str, object],
    profile_native_kernel: bool,
) -> dict[str, object]:
    round_wall_sec = _sum_round_metric(round_summaries, "round_runtime_sec")
    round_accounted_sec = _sum_round_metric(round_summaries, "accounted_runtime_sec")
    round_unaccounted_sec = max(0.0, round_wall_sec - round_accounted_sec)
    top_level_accounted_sec = (
        float(configuration_setup_runtime_sec)
        + float(extension_setup_runtime_sec)
        + float(initial_gdv_runtime_sec)
        + float(standardization_runtime_sec)
        + float(pruning_state_initialization_runtime_sec)
        + round_wall_sec
    )
    top_level_unaccounted_sec = max(0.0, float(runtime_sec) - top_level_accounted_sec)

    timing_keys = [
        "round_setup_runtime_sec",
        "bridge_runtime_sec",
        "tarjan_bridge_runtime_sec",
        "eligibility_runtime_sec",
        "cache_partition_runtime_sec",
        "native_score_runtime_sec",
        "native_pair_generation_runtime_sec",
        "native_delta_accumulation_runtime_sec",
        "native_score_scalarization_runtime_sec",
        "native_scalar_refresh_runtime_sec",
        "best_selection_runtime_sec",
        "selected_update_runtime_sec",
        "selected_delta_runtime_sec",
        "apply_state_update_runtime_sec",
        "native_mixed_correction_runtime_sec",
        "dirty_signature_nodes_runtime_sec",
        "cache_invalidation_runtime_sec",
        "remove_edge_runtime_sec",
        "python_incremental_state_removal_runtime_sec",
        "native_graph_edge_removal_runtime_sec",
        "active_edge_list_rebuild_runtime_sec",
        "batch_edge_tensor_removal_runtime_sec",
        "analysis_row_build_runtime_sec",
    ]
    diagnostic_timings = {key: _sum_round_metric(round_summaries, key) for key in timing_keys}

    count_keys = [
        "eligible_count",
        "blocked_by_bridge_count",
        "blocked_by_d_min_count",
        "rescored_edge_count",
        "scalar_refreshed_edge_count",
        "refresh_candidate_count",
        "reused_score_count",
        "invalidated_score_count",
        "mixed_correction_edge_count",
        "delta_impacted_full_rescore_count",
        "update_union_size",
    ]
    totals = {key: int(round(_sum_round_metric(round_summaries, key))) for key in count_keys}
    round_count = len(round_summaries)
    achieved_budget = len(removed_total)
    total_candidate_actions = (
        totals["rescored_edge_count"]
        + totals["scalar_refreshed_edge_count"]
        + totals["reused_score_count"]
    )

    peak_values = [
        float(value)
        for row in round_summaries
        for value in (row.get("peak_rss_mb_before"), row.get("peak_rss_mb_after"))
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    if peak_rss_start_mb is not None:
        peak_values.append(float(peak_rss_start_mb))
    if peak_rss_end_mb is not None:
        peak_values.append(float(peak_rss_end_mb))

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_label": profile_label,
        "dataset": {
            "name": bundle.name,
            "num_nodes": bundle.num_nodes,
            "num_edges": bundle.num_edges,
        },
        "run": {
            "seed": int(seed),
            "target_rho": float(config.rho),
            "requested_budget": int(budget),
            "achieved_budget": int(achieved_budget),
            "achieved_edge_reduction": 1.0 - ((bundle.num_edges - achieved_budget) / max(bundle.num_edges, 1)),
            "round_count": int(round_count),
            "relshift_engine": relshift_engine,
            "score_mode": config.score_mode,
            "guard_bridges": bool(config.guard_bridges),
            "d_min": int(config.d_min),
            "profile_native_kernel": bool(profile_native_kernel),
        },
        "gdv": {
            "backend": str(initial_gdv_compute_info.get("backend", getattr(config, "backend", ""))),
            "cache_hit": initial_gdv_compute_info.get("cache_hit"),
            "cache_path": initial_gdv_compute_info.get("cache_path"),
        },
        "wall_clock": {
            "algorithm_runtime_sec": float(runtime_sec),
            "runtime_including_score_write_sec": float(total_runtime_including_score_write_sec),
            "score_table_write_runtime_sec": float(score_table_write_runtime_sec),
            "configuration_setup_runtime_sec": float(configuration_setup_runtime_sec),
            "extension_setup_runtime_sec": float(extension_setup_runtime_sec),
            "initial_gdv_runtime_sec": float(initial_gdv_runtime_sec),
            "standardization_runtime_sec": float(standardization_runtime_sec),
            "initial_native_graph_state_runtime_sec": float(initial_native_graph_state_runtime_sec),
            "pruning_state_initialization_runtime_sec": float(pruning_state_initialization_runtime_sec),
            "round_wall_runtime_sec": round_wall_sec,
            "round_accounted_runtime_sec": round_accounted_sec,
            "round_unaccounted_runtime_sec": round_unaccounted_sec,
            "round_runtime_coverage_ratio": min(1.0, round_accounted_sec / round_wall_sec) if round_wall_sec > 0 else 1.0,
            "top_level_accounted_runtime_sec": top_level_accounted_sec,
            "top_level_unaccounted_runtime_sec": top_level_unaccounted_sec,
            "top_level_runtime_coverage_ratio": min(1.0, top_level_accounted_sec / runtime_sec) if runtime_sec > 0 else 1.0,
        },
        "diagnostic_timing_totals_sec": diagnostic_timings,
        "count_totals": totals,
        "count_means_per_round": {
            key: _mean_round_metric(round_summaries, key)
            for key in count_keys
        },
        "cache_behavior": {
            "candidate_action_count": int(total_candidate_actions),
            "reuse_ratio": (totals["reused_score_count"] / total_candidate_actions) if total_candidate_actions else 0.0,
            "full_rescore_ratio": (totals["rescored_edge_count"] / total_candidate_actions) if total_candidate_actions else 0.0,
            "scalar_refresh_ratio": (totals["scalar_refreshed_edge_count"] / total_candidate_actions) if total_candidate_actions else 0.0,
        },
        "local_structure": {
            "mean_avg_directly_attached_size": _mean_round_metric(round_summaries, "avg_directly_attached_size"),
            "mean_avg_four_node_pair_count": _mean_round_metric(round_summaries, "avg_four_node_pair_count"),
            "mean_update_union_size": _mean_round_metric(round_summaries, "update_union_size"),
            "max_update_union_size": int(max((int(row.get("update_union_size", 0)) for row in round_summaries), default=0)),
            "mean_bridge_count": _mean_round_metric(round_summaries, "bridge_count"),
        },
        "memory": {
            "peak_rss_start_mb": peak_rss_start_mb,
            "peak_rss_end_mb": peak_rss_end_mb,
            "peak_rss_observed_mb": max(peak_values) if peak_values else None,
            **known_state_bytes,
            "known_numpy_state_total_mib": known_state_bytes.get("known_numpy_state_total_bytes", 0) / (1024.0 * 1024.0),
            "note": "Known-state bytes exclude Python container overhead, native CSR capacity, and allocator fragmentation.",
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model_name(),
            "logical_cpu_count": os.cpu_count(),
            "openmp_enabled": bool(native_openmp_info.get("openmp_enabled", False)),
            "openmp_max_threads": int(native_openmp_info.get("openmp_max_threads", 1)),
        },
    }


def _profile_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_runtime_profile_artifacts(
    artifact_dir: str | Path,
    *,
    summary: dict[str, object],
    round_summaries: list[dict[str, object]],
) -> tuple[Path, Path, float]:
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "runtime_summary.json"
    rounds_path = output_dir / "runtime_by_round.csv"
    write_start = time.perf_counter()

    preferred_fields = [
        "round",
        "active_edge_count_before",
        "active_edge_count_after",
        "eligible_count",
        "blocked_by_bridge_count",
        "blocked_by_d_min_count",
        "bridge_count",
        "rescored_edge_count",
        "scalar_refreshed_edge_count",
        "refresh_candidate_count",
        "reused_score_count",
        "invalidated_score_count",
        "mixed_correction_edge_count",
        "delta_impacted_full_rescore_count",
        "update_union_size",
        "avg_directly_attached_size",
        "avg_four_node_pair_count",
        "round_runtime_sec",
        "round_setup_runtime_sec",
        "bridge_runtime_sec",
        "eligibility_runtime_sec",
        "native_score_runtime_sec",
        "native_scalar_refresh_runtime_sec",
        "best_selection_runtime_sec",
        "selected_update_runtime_sec",
        "selected_delta_runtime_sec",
        "apply_state_update_runtime_sec",
        "cache_invalidation_runtime_sec",
        "remove_edge_runtime_sec",
        "python_incremental_state_removal_runtime_sec",
        "native_graph_edge_removal_runtime_sec",
        "active_edge_list_rebuild_runtime_sec",
        "analysis_row_build_runtime_sec",
        "accounted_runtime_sec",
        "unaccounted_runtime_sec",
        "runtime_coverage_ratio",
        "peak_rss_mb_before",
        "peak_rss_mb_after",
        "removed_edges",
    ]
    all_fields = {key for row in round_summaries for key in row.keys()}
    fieldnames = [field for field in preferred_fields if field in all_fields]
    fieldnames.extend(sorted(all_fields - set(fieldnames)))
    with rounds_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in round_summaries:
            writer.writerow({key: _profile_csv_value(row.get(key)) for key in fieldnames})

    summary["artifacts"] = {
        "runtime_summary_json": str(summary_path),
        "runtime_by_round_csv": str(rounds_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    elapsed = float(time.perf_counter() - write_start)
    return summary_path, rounds_path, elapsed
