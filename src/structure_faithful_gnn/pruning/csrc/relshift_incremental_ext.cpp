#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

constexpr int kOrbitDim = 15;
constexpr int kUvEdgeBit = 1;
constexpr int kUAEdgeBit = 2;
constexpr int kUBEdgeBit = 4;
constexpr int kVAEdgeBit = 8;
constexpr int kVBEdgeBit = 16;
constexpr int kABEdgeBit = 32;

using OrbitRow = std::array<int8_t, 4>;

struct PairCode {
    uint64_t code;
    int edge_mask;
};

struct PairMask {
    int first;
    int second;
    int mask;
};

struct EdgeKey {
    double score;
    double degree_score;
    double support_score;
    int64_t edge_id;
};

const std::array<OrbitRow, 2> kSize2Tables = [] {
    std::array<OrbitRow, 2> table{};
    for (auto& row : table) {
        row = OrbitRow{-1, -1, -1, -1};
    }
    table[1] = OrbitRow{0, 0, -1, -1};
    return table;
}();

const std::array<OrbitRow, 8> kSize3Tables = [] {
    std::array<OrbitRow, 8> table{};
    for (auto& row : table) {
        row = OrbitRow{-1, -1, -1, -1};
    }
    table[3] = OrbitRow{2, 1, 1, -1};
    table[5] = OrbitRow{1, 2, 1, -1};
    table[6] = OrbitRow{1, 1, 2, -1};
    table[7] = OrbitRow{3, 3, 3, -1};
    return table;
}();

const std::array<OrbitRow, 64> kSize4Tables = [] {
    std::array<OrbitRow, 64> table{};
    for (auto& row : table) {
        row = OrbitRow{-1, -1, -1, -1};
    }
    table[7] = OrbitRow{7, 6, 6, 6};
    table[13] = OrbitRow{5, 5, 4, 4};
    table[14] = OrbitRow{5, 4, 5, 4};
    // ORCA paw/tailed-triangle order: tail leaf=9, triangle degree-2=10, attachment degree-3=11.
    table[15] = OrbitRow{11, 10, 10, 9};
    table[19] = OrbitRow{5, 5, 4, 4};
    table[22] = OrbitRow{5, 4, 4, 5};
    table[23] = OrbitRow{11, 10, 9, 10};
    table[25] = OrbitRow{6, 7, 6, 6};
    table[26] = OrbitRow{4, 5, 5, 4};
    table[27] = OrbitRow{10, 11, 10, 9};
    table[28] = OrbitRow{4, 5, 4, 5};
    table[29] = OrbitRow{10, 11, 9, 10};
    table[30] = OrbitRow{8, 8, 8, 8};
    table[31] = OrbitRow{13, 13, 12, 12};
    table[35] = OrbitRow{5, 4, 5, 4};
    table[37] = OrbitRow{5, 4, 4, 5};
    table[39] = OrbitRow{11, 9, 10, 10};
    table[41] = OrbitRow{4, 5, 5, 4};
    table[42] = OrbitRow{6, 6, 7, 6};
    table[43] = OrbitRow{10, 10, 11, 9};
    table[44] = OrbitRow{4, 4, 5, 5};
    table[45] = OrbitRow{8, 8, 8, 8};
    table[46] = OrbitRow{10, 9, 11, 10};
    table[47] = OrbitRow{13, 12, 13, 12};
    table[49] = OrbitRow{4, 5, 4, 5};
    table[50] = OrbitRow{4, 4, 5, 5};
    table[51] = OrbitRow{8, 8, 8, 8};
    table[52] = OrbitRow{6, 6, 6, 7};
    table[53] = OrbitRow{10, 10, 9, 11};
    table[54] = OrbitRow{10, 9, 10, 11};
    table[55] = OrbitRow{13, 12, 12, 13};
    table[57] = OrbitRow{9, 11, 10, 10};
    table[58] = OrbitRow{9, 10, 11, 10};
    table[59] = OrbitRow{12, 13, 13, 12};
    table[60] = OrbitRow{9, 10, 10, 11};
    table[61] = OrbitRow{12, 13, 12, 13};
    table[62] = OrbitRow{12, 12, 13, 13};
    table[63] = OrbitRow{14, 14, 14, 14};
    return table;
}();

inline const OrbitRow& orbit_row_for_mask(const int size, const int mask) {
    if (size == 2) {
        return kSize2Tables.at(static_cast<size_t>(mask));
    }
    if (size == 3) {
        return kSize3Tables.at(static_cast<size_t>(mask));
    }
    if (size == 4) {
        return kSize4Tables.at(static_cast<size_t>(mask));
    }
    throw std::runtime_error("Unsupported graphlet size.");
}

inline bool is_valid_orbit_row(const OrbitRow& row) {
    return row[0] >= 0;
}

inline uint64_t encode_pair(const int a, const int b) {
    const uint32_t low = static_cast<uint32_t>(std::min(a, b));
    const uint32_t high = static_cast<uint32_t>(std::max(a, b));
    return (static_cast<uint64_t>(low) << 32U) | static_cast<uint64_t>(high);
}

inline std::pair<int, int> decode_pair(const uint64_t code) {
    const int low = static_cast<int>(code >> 32U);
    const int high = static_cast<int>(code & 0xffffffffU);
    return {low, high};
}

inline bool adjacency_entry_is_active(
    const int64_t adjacency_index,
    const int64_t* adjacency_edge_ids,
    const uint8_t* active_edge_mask
) {
    if (adjacency_edge_ids == nullptr || active_edge_mask == nullptr) {
        return true;
    }
    const int64_t edge_id = adjacency_edge_ids[adjacency_index];
    return edge_id >= 0 && active_edge_mask[edge_id] != 0;
}

int edge_support(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int u,
    const int v,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    int64_t left = row_ptr[u];
    int64_t left_stop = row_ptr[u + 1];
    int64_t right = row_ptr[v];
    int64_t right_stop = row_ptr[v + 1];
    int count = 0;
    while (left < left_stop && right < right_stop) {
        while (
            left < left_stop &&
            !adjacency_entry_is_active(left, adjacency_edge_ids, active_edge_mask)
        ) {
            ++left;
        }
        while (
            right < right_stop &&
            !adjacency_entry_is_active(right, adjacency_edge_ids, active_edge_mask)
        ) {
            ++right;
        }
        if (left >= left_stop || right >= right_stop) {
            break;
        }
        const int64_t left_node = col_idx[left];
        const int64_t right_node = col_idx[right];
        if (left_node == right_node) {
            ++count;
            ++left;
            ++right;
        } else if (left_node < right_node) {
            ++left;
        } else {
            ++right;
        }
    }
    return count;
}

inline bool edge_key_less(const EdgeKey& left, const EdgeKey& right) {
    if (left.score != right.score) {
        return left.score < right.score;
    }
    if (left.degree_score != right.degree_score) {
        return left.degree_score < right.degree_score;
    }
    if (left.support_score != right.support_score) {
        return left.support_score < right.support_score;
    }
    return left.edge_id < right.edge_id;
}

void collect_attached_nodes(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int u,
    const int v,
    std::vector<int>& marks,
    std::vector<int>& attachment_masks,
    std::vector<int>& directly_attached,
    int& epoch,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    ++epoch;
    directly_attached.clear();
    directly_attached.reserve(
        static_cast<size_t>(row_ptr[u + 1] - row_ptr[u] + row_ptr[v + 1] - row_ptr[v])
    );
    auto add_node = [&](const int node, const int edge_bit) {
        if (node == u || node == v) {
            return;
        }
        if (marks[node] == epoch) {
            attachment_masks[node] |= edge_bit;
            return;
        }
        marks[node] = epoch;
        attachment_masks[node] = edge_bit;
        directly_attached.push_back(node);
    };
    for (int64_t idx = row_ptr[u]; idx < row_ptr[u + 1]; ++idx) {
        if (!adjacency_entry_is_active(idx, adjacency_edge_ids, active_edge_mask)) {
            continue;
        }
        add_node(static_cast<int>(col_idx[idx]), kUAEdgeBit);
    }
    for (int64_t idx = row_ptr[v]; idx < row_ptr[v + 1]; ++idx) {
        if (!adjacency_entry_is_active(idx, adjacency_edge_ids, active_edge_mask)) {
            continue;
        }
        add_node(static_cast<int>(col_idx[idx]), kVAEdgeBit);
    }
}

int collect_two_hop_size(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int u,
    const int v,
    std::vector<int>& marks,
    int& epoch,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    ++epoch;
    int count = 0;
    auto mark_node = [&](const int node) {
        if (marks[node] == epoch) {
            return;
        }
        marks[node] = epoch;
        ++count;
    };
    mark_node(u);
    mark_node(v);

    std::vector<int> frontier;
    frontier.reserve(32);
    for (const int seed : {u, v}) {
        for (int64_t idx = row_ptr[seed]; idx < row_ptr[seed + 1]; ++idx) {
            if (!adjacency_entry_is_active(idx, adjacency_edge_ids, active_edge_mask)) {
                continue;
            }
            const int neighbor = static_cast<int>(col_idx[idx]);
            if (marks[neighbor] != epoch) {
                marks[neighbor] = epoch;
                ++count;
                frontier.push_back(neighbor);
            }
        }
    }
    for (const int node : frontier) {
        for (int64_t idx = row_ptr[node]; idx < row_ptr[node + 1]; ++idx) {
            if (!adjacency_entry_is_active(idx, adjacency_edge_ids, active_edge_mask)) {
                continue;
            }
            const int neighbor = static_cast<int>(col_idx[idx]);
            mark_node(neighbor);
        }
    }
    return count;
}

std::vector<int> collect_two_hop_nodes_sorted(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int u,
    const int v,
    std::vector<int>& marks,
    int& epoch,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    ++epoch;
    std::vector<int> touched;
    touched.reserve(64);
    auto mark_node = [&](const int node) {
        if (marks[node] == epoch) {
            return;
        }
        marks[node] = epoch;
        touched.push_back(node);
    };
    mark_node(u);
    mark_node(v);

    std::vector<int> frontier;
    frontier.reserve(32);
    for (const int seed : {u, v}) {
        for (int64_t idx = row_ptr[seed]; idx < row_ptr[seed + 1]; ++idx) {
            if (!adjacency_entry_is_active(idx, adjacency_edge_ids, active_edge_mask)) {
                continue;
            }
            const int neighbor = static_cast<int>(col_idx[idx]);
            if (marks[neighbor] != epoch) {
                marks[neighbor] = epoch;
                touched.push_back(neighbor);
                frontier.push_back(neighbor);
            }
        }
    }
    for (const int node : frontier) {
        for (int64_t idx = row_ptr[node]; idx < row_ptr[node + 1]; ++idx) {
            if (!adjacency_entry_is_active(idx, adjacency_edge_ids, active_edge_mask)) {
                continue;
            }
            mark_node(static_cast<int>(col_idx[idx]));
        }
    }
    std::sort(touched.begin(), touched.end());
    return touched;
}

inline int mask_for_three_from_attachment(const int attachment_mask) {
    int mask = kUvEdgeBit;
    if ((attachment_mask & kUAEdgeBit) != 0) {
        mask |= kUAEdgeBit;
    }
    if ((attachment_mask & kVAEdgeBit) != 0) {
        mask |= 4;
    }
    return mask;
}

inline int mask_for_four_from_attachments(
    const int first_attachment,
    const int second_attachment,
    const int pair_edge_mask
) {
    int mask = kUvEdgeBit | pair_edge_mask;
    if ((first_attachment & kUAEdgeBit) != 0) {
        mask |= kUAEdgeBit;
    }
    if ((second_attachment & kUAEdgeBit) != 0) {
        mask |= kUBEdgeBit;
    }
    if ((first_attachment & kVAEdgeBit) != 0) {
        mask |= kVAEdgeBit;
    }
    if ((second_attachment & kVAEdgeBit) != 0) {
        mask |= kVBEdgeBit;
    }
    return mask;
}

void collect_relevant_pair_masks(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int u,
    const int v,
    const std::vector<int>& directly_attached,
    const std::vector<int>& marks,
    const std::vector<int>& attachment_masks,
    const int epoch,
    std::vector<PairCode>& encoded_pairs,
    std::vector<PairMask>& pair_masks,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    encoded_pairs.clear();
    pair_masks.clear();
    encoded_pairs.reserve(directly_attached.size() * directly_attached.size());

    auto add_pair = [&](const int left, const int right, const int edge_mask) {
        if (left == right || left == u || left == v || right == u || right == v) {
            return;
        }
        encoded_pairs.push_back(PairCode{encode_pair(left, right), edge_mask});
    };

    for (size_t idx = 0; idx < directly_attached.size(); ++idx) {
        for (size_t jdx = idx + 1; jdx < directly_attached.size(); ++jdx) {
            add_pair(directly_attached[idx], directly_attached[jdx], 0);
        }
    }
    for (const int a : directly_attached) {
        for (int64_t idx = row_ptr[a]; idx < row_ptr[a + 1]; ++idx) {
            if (!adjacency_entry_is_active(idx, adjacency_edge_ids, active_edge_mask)) {
                continue;
            }
            add_pair(a, static_cast<int>(col_idx[idx]), kABEdgeBit);
        }
    }

    std::sort(
        encoded_pairs.begin(),
        encoded_pairs.end(),
        [](const PairCode& left, const PairCode& right) {
            return left.code < right.code;
        }
    );

    pair_masks.reserve(encoded_pairs.size());
    size_t idx = 0;
    while (idx < encoded_pairs.size()) {
        const uint64_t code = encoded_pairs[idx].code;
        int pair_edge_mask = 0;
        while (idx < encoded_pairs.size() && encoded_pairs[idx].code == code) {
            pair_edge_mask |= encoded_pairs[idx].edge_mask;
            ++idx;
        }
        const auto [first, second] = decode_pair(code);
        const int first_attachment = marks[first] == epoch ? attachment_masks[first] : 0;
        const int second_attachment = marks[second] == epoch ? attachment_masks[second] : 0;
        const int mask = mask_for_four_from_attachments(first_attachment, second_attachment, pair_edge_mask);
        pair_masks.push_back(PairMask{first, second, mask});
    }
}

inline int attachment_class_index(const int attachment_mask) {
    const bool attached_to_u = (attachment_mask & kUAEdgeBit) != 0;
    const bool attached_to_v = (attachment_mask & kVAEdgeBit) != 0;
    if (attached_to_u && attached_to_v) {
        return 2;
    }
    if (attached_to_v) {
        return 1;
    }
    if (attached_to_u) {
        return 0;
    }
    return -1;
}

inline int attachment_mask_for_class(const int class_idx) {
    if (class_idx == 0) {
        return kUAEdgeBit;
    }
    if (class_idx == 1) {
        return kVAEdgeBit;
    }
    if (class_idx == 2) {
        return kUAEdgeBit | kVAEdgeBit;
    }
    return 0;
}

void count_relevant_four_node_masks_combinatorial(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int u,
    const int v,
    std::vector<int>& directly_attached,
    const std::vector<int>& marks,
    const std::vector<int>& attachment_masks,
    const int epoch,
    std::array<int64_t, 64>& mask_counts,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    mask_counts.fill(0);

    std::sort(directly_attached.begin(), directly_attached.end());

    auto attachment_for_node = [&](const int node) {
        return marks[node] == epoch ? attachment_masks[node] : 0;
    };

    auto mask_without_ab = [&](const int left, const int right) {
        return mask_for_four_from_attachments(attachment_for_node(left), attachment_for_node(right), 0);
    };

    auto mask_with_ab = [&](const int left, const int right) {
        return mask_for_four_from_attachments(attachment_for_node(left), attachment_for_node(right), kABEdgeBit);
    };

    std::array<int64_t, 3> suffix_class_counts{0, 0, 0};
    for (const int node : directly_attached) {
        const int class_idx = attachment_class_index(attachment_masks[node]);
        if (class_idx >= 0) {
            ++suffix_class_counts[static_cast<size_t>(class_idx)];
        }
    }

    // Count all D-D no-edge masks without materializing every unordered pair.
    // Sorting preserves the old local ordering convention: first=min(a,b), second=max(a,b).
    for (const int left : directly_attached) {
        const int left_attachment = attachment_masks[left];
        const int left_class = attachment_class_index(left_attachment);
        if (left_class >= 0) {
            --suffix_class_counts[static_cast<size_t>(left_class)];
        }
        for (int right_class = 0; right_class < 3; ++right_class) {
            const int64_t count = suffix_class_counts[static_cast<size_t>(right_class)];
            if (count == 0) {
                continue;
            }
            const int right_attachment = attachment_mask_for_class(right_class);
            const int mask = mask_for_four_from_attachments(left_attachment, right_attachment, 0);
            mask_counts[static_cast<size_t>(mask)] += count;
        }
    }

    // Correct the histogram for actual a-b edges, and add D-outside-D pairs
    // that are only relevant when the a-b edge exists.
    for (const int node : directly_attached) {
        for (int64_t idx = row_ptr[node]; idx < row_ptr[node + 1]; ++idx) {
            if (!adjacency_entry_is_active(idx, adjacency_edge_ids, active_edge_mask)) {
                continue;
            }
            const int neighbor = static_cast<int>(col_idx[idx]);
            if (neighbor == u || neighbor == v || neighbor == node) {
                continue;
            }
            const bool neighbor_in_d = marks[neighbor] == epoch;
            if (neighbor_in_d) {
                if (node < neighbor) {
                    const int no_ab = mask_without_ab(node, neighbor);
                    const int with_ab = mask_with_ab(node, neighbor);
                    --mask_counts[static_cast<size_t>(no_ab)];
                    ++mask_counts[static_cast<size_t>(with_ab)];
                }
            } else {
                const int left = std::min(node, neighbor);
                const int right = std::max(node, neighbor);
                ++mask_counts[static_cast<size_t>(mask_with_ab(left, right))];
            }
        }
    }
}

void accumulate_endpoint_delta_for_mask(
    const int size,
    const int before_mask,
    double* endpoint_delta
) {
    const OrbitRow& before = orbit_row_for_mask(size, before_mask);
    if (!is_valid_orbit_row(before)) {
        return;
    }
    const OrbitRow& after = orbit_row_for_mask(size, before_mask & (~1));
    for (int local_idx = 0; local_idx < std::min(size, 2); ++local_idx) {
        const int before_orbit = before[local_idx];
        endpoint_delta[local_idx * kOrbitDim + before_orbit] -= 1.0;
        if (is_valid_orbit_row(after)) {
            endpoint_delta[local_idx * kOrbitDim + after[local_idx]] += 1.0;
        }
    }
}

void accumulate_endpoint_delta_for_mask_count(
    const int size,
    const int before_mask,
    const int64_t count,
    double* endpoint_delta
) {
    if (count == 0) {
        return;
    }
    const OrbitRow& before = orbit_row_for_mask(size, before_mask);
    if (!is_valid_orbit_row(before)) {
        return;
    }
    const OrbitRow& after = orbit_row_for_mask(size, before_mask & (~1));
    const double weight = static_cast<double>(count);
    for (int local_idx = 0; local_idx < std::min(size, 2); ++local_idx) {
        const int before_orbit = before[local_idx];
        endpoint_delta[local_idx * kOrbitDim + before_orbit] -= weight;
        if (is_valid_orbit_row(after)) {
            endpoint_delta[local_idx * kOrbitDim + after[local_idx]] += weight;
        }
    }
}

struct ScoreResult {
    double score;
    double mean_abs_delta;
    double mean_rel_delta;
    double mean_denom;
    double min_denom;
    int update_size;
    int directly_attached_size;
    int four_node_pair_count;
};

void copy_endpoint_delta_to_int_cache(const double* endpoint_delta, int64_t* endpoint_delta_out) {
    if (endpoint_delta_out == nullptr) {
        return;
    }
    for (int idx = 0; idx < 2 * kOrbitDim; ++idx) {
        endpoint_delta_out[idx] = static_cast<int64_t>(std::llround(endpoint_delta[idx]));
    }
}

void add_orbit_vector_contribution(
    const int size,
    const int mask,
    const int local_idx,
    const int64_t sign,
    int64_t* slot_delta
) {
    const OrbitRow& row = orbit_row_for_mask(size, mask);
    if (!is_valid_orbit_row(row)) {
        return;
    }
    const int orbit = row[local_idx];
    if (orbit >= 0) {
        slot_delta[orbit] += sign;
    }
}

template <typename RawView, typename StdView, typename MeanView, typename StdStatsView>
ScoreResult score_from_endpoint_delta_cache(
    const int u,
    const int v,
    const RawView& current_raw,
    const StdView& current_std,
    const MeanView& stats_mean,
    const StdStatsView& stats_std,
    const std::string& score_mode,
    const double eps,
    const int64_t* endpoint_delta
) {
    double absolute_sum = 0.0;
    double relative_sum = 0.0;
    double denom_sum = 0.0;
    double denom_min = 0.0;
    for (int local_idx = 0; local_idx < 2; ++local_idx) {
        const int node = local_idx == 0 ? u : v;
        double endpoint_l1 = 0.0;
        double base_l1 = 0.0;
        for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
            const double candidate_raw =
                std::max(current_raw(node, orbit_idx) + static_cast<double>(endpoint_delta[local_idx * kOrbitDim + orbit_idx]), 0.0);
            const double candidate_std =
                (std::log1p(candidate_raw) - stats_mean(orbit_idx)) / stats_std(orbit_idx);
            endpoint_l1 += std::abs(current_std(node, orbit_idx) - candidate_std);
            base_l1 += std::abs(current_std(node, orbit_idx));
        }
        const double endpoint_denom = base_l1 + eps;
        absolute_sum += endpoint_l1;
        relative_sum += endpoint_l1 / endpoint_denom;
        denom_sum += endpoint_denom;
        if (local_idx == 0 || endpoint_denom < denom_min) {
            denom_min = endpoint_denom;
        }
    }
    const double mean_abs = absolute_sum / 2.0;
    const double mean_rel = relative_sum / 2.0;
    return ScoreResult{
        score_mode == "absolute" ? mean_abs : mean_rel,
        mean_abs,
        mean_rel,
        denom_sum / 2.0,
        denom_min,
        0,
        0,
        0,
    };
}

void accumulate_full_delta_for_mask(
    const int size,
    const int before_mask,
    const std::array<int, 4>& nodes,
    const std::unordered_map<int, int>& affected_index,
    double* raw_delta
) {
    const OrbitRow& before = orbit_row_for_mask(size, before_mask);
    if (!is_valid_orbit_row(before)) {
        return;
    }
    const OrbitRow& after = orbit_row_for_mask(size, before_mask & (~1));
    for (int local_idx = 0; local_idx < size; ++local_idx) {
        const int global_node = nodes[local_idx];
        const auto found = affected_index.find(global_node);
        if (found == affected_index.end()) {
            continue;
        }
        const int row_offset = found->second * kOrbitDim;
        raw_delta[row_offset + before[local_idx]] -= 1.0;
        if (is_valid_orbit_row(after)) {
            raw_delta[row_offset + after[local_idx]] += 1.0;
        }
    }
}

template <typename RawView, typename StdView, typename MeanView, typename StdStatsView>
ScoreResult score_single_edge_mask_count(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int u,
    const int v,
    const RawView& current_raw,
    const StdView& current_std,
    const MeanView& stats_mean,
    const StdStatsView& stats_std,
    const std::string& score_mode,
    const double eps,
    const bool include_update_sizes,
    std::vector<int>& marks,
    std::vector<int>& attachment_masks,
    std::vector<int>& directly_attached,
    std::array<int64_t, 64>& four_node_mask_counts,
    int& epoch,
    double* pair_generation_sec = nullptr,
    double* delta_accumulation_sec = nullptr,
    double* score_scalarization_sec = nullptr,
    int64_t* endpoint_delta_out = nullptr,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    const auto pair_start = pair_generation_sec == nullptr ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
    collect_attached_nodes(
        row_ptr,
        col_idx,
        u,
        v,
        marks,
        attachment_masks,
        directly_attached,
        epoch,
        adjacency_edge_ids,
        active_edge_mask
    );
    count_relevant_four_node_masks_combinatorial(
        row_ptr,
        col_idx,
        u,
        v,
        directly_attached,
        marks,
        attachment_masks,
        epoch,
        four_node_mask_counts,
        adjacency_edge_ids,
        active_edge_mask
    );
    if (pair_generation_sec != nullptr) {
        *pair_generation_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - pair_start).count();
    }

    const auto delta_start = delta_accumulation_sec == nullptr ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
    std::array<double, 2 * kOrbitDim> endpoint_delta{};
    endpoint_delta.fill(0.0);

    accumulate_endpoint_delta_for_mask(2, kUvEdgeBit, endpoint_delta.data());
    for (const int node : directly_attached) {
        const int mask = mask_for_three_from_attachment(attachment_masks[node]);
        accumulate_endpoint_delta_for_mask(3, mask, endpoint_delta.data());
    }
    int64_t four_node_pair_count = 0;
    for (size_t mask = 0; mask < four_node_mask_counts.size(); ++mask) {
        const int64_t count = four_node_mask_counts[mask];
        four_node_pair_count += count;
        accumulate_endpoint_delta_for_mask_count(4, static_cast<int>(mask), count, endpoint_delta.data());
    }
    copy_endpoint_delta_to_int_cache(endpoint_delta.data(), endpoint_delta_out);

    const int update_size = include_update_sizes
        ? collect_two_hop_size(
            row_ptr,
            col_idx,
            u,
            v,
            marks,
            epoch,
            adjacency_edge_ids,
            active_edge_mask
        )
        : 0;
    if (delta_accumulation_sec != nullptr) {
        *delta_accumulation_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - delta_start).count();
    }

    const auto scalar_start = score_scalarization_sec == nullptr ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
    double absolute_sum = 0.0;
    double relative_sum = 0.0;
    double denom_sum = 0.0;
    double denom_min = 0.0;
    for (int local_idx = 0; local_idx < 2; ++local_idx) {
        const int node = local_idx == 0 ? u : v;
        double endpoint_l1 = 0.0;
        double base_l1 = 0.0;
        for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
            const double candidate_raw =
                std::max(current_raw(node, orbit_idx) + endpoint_delta[local_idx * kOrbitDim + orbit_idx], 0.0);
            const double candidate_std =
                (std::log1p(candidate_raw) - stats_mean(orbit_idx)) / stats_std(orbit_idx);
            endpoint_l1 += std::abs(current_std(node, orbit_idx) - candidate_std);
            base_l1 += std::abs(current_std(node, orbit_idx));
        }
        const double endpoint_denom = base_l1 + eps;
        absolute_sum += endpoint_l1;
        relative_sum += endpoint_l1 / endpoint_denom;
        denom_sum += endpoint_denom;
        if (local_idx == 0 || endpoint_denom < denom_min) {
            denom_min = endpoint_denom;
        }
    }

    const double mean_abs = absolute_sum / 2.0;
    const double mean_rel = relative_sum / 2.0;
    if (score_scalarization_sec != nullptr) {
        *score_scalarization_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - scalar_start).count();
    }
    return ScoreResult{
        score_mode == "absolute" ? mean_abs : mean_rel,
        mean_abs,
        mean_rel,
        denom_sum / 2.0,
        denom_min,
        update_size,
        static_cast<int>(directly_attached.size()),
        static_cast<int>(four_node_pair_count),
    };
}

py::dict score_edges_round(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> row_ptr_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> col_idx_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_edges_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> current_raw_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> current_std_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> stats_mean_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> stats_std_array,
    const std::string& score_mode,
    const double eps,
    const bool include_update_sizes
) {
    const auto row_ptr = row_ptr_array.unchecked<1>();
    const auto col_idx = col_idx_array.unchecked<1>();
    const auto candidate_edges = candidate_edges_array.unchecked<2>();
    const auto current_raw = current_raw_array.unchecked<2>();
    const auto current_std = current_std_array.unchecked<2>();
    const auto stats_mean = stats_mean_array.unchecked<1>();
    const auto stats_std = stats_std_array.unchecked<1>();

    if (candidate_edges.shape(1) != 2) {
        throw std::runtime_error("candidate_edges must have shape [num_edges, 2].");
    }
    if (current_raw.shape(1) != kOrbitDim || current_std.shape(1) != kOrbitDim) {
        throw std::runtime_error("current_raw/current_std must have 15 columns.");
    }
    if (stats_mean.shape(0) != kOrbitDim || stats_std.shape(0) != kOrbitDim) {
        throw std::runtime_error("stats mean/std must have length 15.");
    }

    const int num_nodes = static_cast<int>(current_raw.shape(0));
    const ssize_t num_edges = candidate_edges.shape(0);

    py::array_t<double> scores_array(num_edges);
    py::array_t<double> mean_abs_delta_array(num_edges);
    py::array_t<double> mean_rel_delta_array(num_edges);
    py::array_t<double> mean_denom_array(num_edges);
    py::array_t<double> min_denom_array(num_edges);
    py::array_t<int64_t> update_sizes_array(num_edges);
    py::array_t<int64_t> directly_attached_sizes_array(num_edges);
    py::array_t<int64_t> four_node_pair_counts_array(num_edges);

    auto scores = scores_array.mutable_unchecked<1>();
    auto mean_abs_delta = mean_abs_delta_array.mutable_unchecked<1>();
    auto mean_rel_delta = mean_rel_delta_array.mutable_unchecked<1>();
    auto mean_denom = mean_denom_array.mutable_unchecked<1>();
    auto min_denom = min_denom_array.mutable_unchecked<1>();
    auto update_sizes = update_sizes_array.mutable_unchecked<1>();
    auto directly_attached_sizes = directly_attached_sizes_array.mutable_unchecked<1>();
    auto four_node_pair_counts = four_node_pair_counts_array.mutable_unchecked<1>();

    std::vector<int> marks(static_cast<size_t>(num_nodes), 0);
    std::vector<int> attachment_masks(static_cast<size_t>(num_nodes), 0);
    std::vector<int> directly_attached;
    std::array<int64_t, 64> four_node_mask_counts{};
    int epoch = 0;

    for (ssize_t edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
        const int u = static_cast<int>(candidate_edges(edge_idx, 0));
        const int v = static_cast<int>(candidate_edges(edge_idx, 1));
        const ScoreResult scored = score_single_edge_mask_count(
            row_ptr.data(0),
            col_idx.data(0),
            u,
            v,
            current_raw,
            current_std,
            stats_mean,
            stats_std,
            score_mode,
            eps,
            include_update_sizes,
            marks,
            attachment_masks,
            directly_attached,
            four_node_mask_counts,
            epoch
        );
        scores(edge_idx) = scored.score;
        mean_abs_delta(edge_idx) = scored.mean_abs_delta;
        mean_rel_delta(edge_idx) = scored.mean_rel_delta;
        mean_denom(edge_idx) = scored.mean_denom;
        min_denom(edge_idx) = scored.min_denom;
        update_sizes(edge_idx) = static_cast<int64_t>(scored.update_size);
        directly_attached_sizes(edge_idx) = static_cast<int64_t>(scored.directly_attached_size);
        four_node_pair_counts(edge_idx) = static_cast<int64_t>(scored.four_node_pair_count);
    }

    py::dict result;
    result["scores"] = std::move(scores_array);
    result["mean_abs_delta_sig"] = std::move(mean_abs_delta_array);
    result["mean_rel_delta_sig"] = std::move(mean_rel_delta_array);
    result["mean_denom"] = std::move(mean_denom_array);
    result["min_denom"] = std::move(min_denom_array);
    result["update_sizes"] = std::move(update_sizes_array);
    result["directly_attached_sizes"] = std::move(directly_attached_sizes_array);
    result["four_node_pair_counts"] = std::move(four_node_pair_counts_array);
    return result;
}

py::dict compute_selected_edge_delta_impl(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int num_nodes,
    const int u,
    const int v,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    std::vector<int> marks(static_cast<size_t>(num_nodes), 0);
    std::vector<int> attachment_masks(static_cast<size_t>(num_nodes), 0);
    std::vector<int> directly_attached;
    std::vector<PairCode> encoded_pairs;
    std::vector<PairMask> pair_masks;
    int epoch = 0;
    collect_attached_nodes(
        row_ptr,
        col_idx,
        u,
        v,
        marks,
        attachment_masks,
        directly_attached,
        epoch,
        adjacency_edge_ids,
        active_edge_mask
    );
    collect_relevant_pair_masks(
        row_ptr,
        col_idx,
        u,
        v,
        directly_attached,
        marks,
        attachment_masks,
        epoch,
        encoded_pairs,
        pair_masks,
        adjacency_edge_ids,
        active_edge_mask
    );
    auto affected_nodes = collect_two_hop_nodes_sorted(
        row_ptr,
        col_idx,
        u,
        v,
        marks,
        epoch,
        adjacency_edge_ids,
        active_edge_mask
    );

    std::unordered_map<int, int> affected_index;
    affected_index.reserve(affected_nodes.size() * 2U + 1U);
    for (size_t idx = 0; idx < affected_nodes.size(); ++idx) {
        affected_index.emplace(affected_nodes[idx], static_cast<int>(idx));
    }

    py::array_t<int64_t> affected_nodes_array(static_cast<ssize_t>(affected_nodes.size()));
    auto affected_nodes_view = affected_nodes_array.mutable_unchecked<1>();
    for (size_t idx = 0; idx < affected_nodes.size(); ++idx) {
        affected_nodes_view(static_cast<ssize_t>(idx)) = static_cast<int64_t>(affected_nodes[idx]);
    }

    py::array_t<double> raw_delta_array({static_cast<ssize_t>(affected_nodes.size()), static_cast<ssize_t>(kOrbitDim)});
    auto raw_delta_mut = raw_delta_array.mutable_unchecked<2>();
    for (ssize_t row = 0; row < raw_delta_mut.shape(0); ++row) {
        for (ssize_t col = 0; col < raw_delta_mut.shape(1); ++col) {
            raw_delta_mut(row, col) = 0.0;
        }
    }
    double* raw_delta_ptr = static_cast<double*>(raw_delta_array.mutable_data());

    std::vector<uint64_t> impacted_edge_codes;
    impacted_edge_codes.reserve(pair_masks.size() * 5U + directly_attached.size() * 2U);
    const uint64_t selected_code = encode_pair(u, v);
    auto add_impacted_edge = [&](const int left, const int right) {
        const uint64_t code = encode_pair(left, right);
        if (code != selected_code) {
            impacted_edge_codes.push_back(code);
        }
    };

    accumulate_full_delta_for_mask(2, kUvEdgeBit, {u, v, -1, -1}, affected_index, raw_delta_ptr);
    for (const int node : directly_attached) {
        const int mask = mask_for_three_from_attachment(attachment_masks[node]);
        accumulate_full_delta_for_mask(3, mask, {u, v, node, -1}, affected_index, raw_delta_ptr);
        if ((mask & kUAEdgeBit) != 0) {
            add_impacted_edge(u, node);
        }
        if ((mask & 4) != 0) {
            add_impacted_edge(v, node);
        }
    }
    for (const PairMask& pair : pair_masks) {
        accumulate_full_delta_for_mask(4, pair.mask, {u, v, pair.first, pair.second}, affected_index, raw_delta_ptr);
        if ((pair.mask & kUAEdgeBit) != 0) {
            add_impacted_edge(u, pair.first);
        }
        if ((pair.mask & kUBEdgeBit) != 0) {
            add_impacted_edge(u, pair.second);
        }
        if ((pair.mask & kVAEdgeBit) != 0) {
            add_impacted_edge(v, pair.first);
        }
        if ((pair.mask & kVBEdgeBit) != 0) {
            add_impacted_edge(v, pair.second);
        }
        if ((pair.mask & kABEdgeBit) != 0) {
            add_impacted_edge(pair.first, pair.second);
        }
    }
    std::sort(impacted_edge_codes.begin(), impacted_edge_codes.end());
    impacted_edge_codes.erase(std::unique(impacted_edge_codes.begin(), impacted_edge_codes.end()), impacted_edge_codes.end());

    py::array_t<int64_t> impacted_edges_array(
        {static_cast<ssize_t>(impacted_edge_codes.size()), static_cast<ssize_t>(2)}
    );
    auto impacted_edges_view = impacted_edges_array.mutable_unchecked<2>();
    for (size_t idx = 0; idx < impacted_edge_codes.size(); ++idx) {
        const auto [left, right] = decode_pair(impacted_edge_codes[idx]);
        impacted_edges_view(static_cast<ssize_t>(idx), 0) = static_cast<int64_t>(left);
        impacted_edges_view(static_cast<ssize_t>(idx), 1) = static_cast<int64_t>(right);
    }

    py::dict result;
    result["affected_nodes"] = std::move(affected_nodes_array);
    result["raw_delta"] = std::move(raw_delta_array);
    result["impacted_edges"] = std::move(impacted_edges_array);
    result["directly_attached_size"] = static_cast<int64_t>(directly_attached.size());
    result["four_node_pair_count"] = static_cast<int64_t>(pair_masks.size());
    return result;
}

py::dict compute_selected_edge_delta(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> row_ptr_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> col_idx_array,
    const int u,
    const int v
) {
    const auto row_ptr = row_ptr_array.unchecked<1>();
    const auto col_idx = col_idx_array.unchecked<1>();
    return compute_selected_edge_delta_impl(
        row_ptr.data(0),
        col_idx.data(0),
        static_cast<int>(row_ptr.shape(0) - 1),
        u,
        v
    );
}

py::dict eligible_edge_ids_from_csr(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> row_ptr_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> col_idx_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> active_edge_ids_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> edge_array_by_id_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> degrees_array,
    const int d_min,
    const bool guard_bridges
) {
    const auto row_ptr = row_ptr_array.unchecked<1>();
    const auto col_idx = col_idx_array.unchecked<1>();
    const auto active_edge_ids = active_edge_ids_array.unchecked<1>();
    const auto edge_array_by_id = edge_array_by_id_array.unchecked<2>();
    const auto degrees = degrees_array.unchecked<1>();
    if (edge_array_by_id.shape(1) != 2) {
        throw std::runtime_error("edge_array_by_id must have shape [num_edges, 2].");
    }

    using Clock = std::chrono::high_resolution_clock;
    const int num_nodes = static_cast<int>(row_ptr.shape(0) - 1);
    std::unordered_set<uint64_t> bridge_codes;
    const auto bridge_start = Clock::now();
    if (guard_bridges) {
        std::vector<int> discovery(static_cast<size_t>(num_nodes), -1);
        std::vector<int> low(static_cast<size_t>(num_nodes), 0);
        std::vector<int> parent(static_cast<size_t>(num_nodes), -1);
        int visit_time = 0;
        bridge_codes.reserve(static_cast<size_t>(active_edge_ids.shape(0) / 8 + 1));
        std::function<void(int)> visit = [&](const int node) {
            discovery[static_cast<size_t>(node)] = visit_time;
            low[static_cast<size_t>(node)] = visit_time;
            ++visit_time;
            for (int64_t idx = row_ptr(node); idx < row_ptr(node + 1); ++idx) {
                const int neighbor = static_cast<int>(col_idx(idx));
                if (discovery[static_cast<size_t>(neighbor)] == -1) {
                    parent[static_cast<size_t>(neighbor)] = node;
                    visit(neighbor);
                    low[static_cast<size_t>(node)] = std::min(
                        low[static_cast<size_t>(node)],
                        low[static_cast<size_t>(neighbor)]
                    );
                    if (low[static_cast<size_t>(neighbor)] > discovery[static_cast<size_t>(node)]) {
                        bridge_codes.insert(encode_pair(node, neighbor));
                    }
                } else if (neighbor != parent[static_cast<size_t>(node)]) {
                    low[static_cast<size_t>(node)] = std::min(
                        low[static_cast<size_t>(node)],
                        discovery[static_cast<size_t>(neighbor)]
                    );
                }
            }
        };
        for (int node = 0; node < num_nodes; ++node) {
            if (discovery[static_cast<size_t>(node)] == -1) {
                visit(node);
            }
        }
    }
    const double bridge_runtime_sec = std::chrono::duration<double>(Clock::now() - bridge_start).count();

    const auto eligibility_start = Clock::now();
    std::vector<int64_t> eligible_ids;
    eligible_ids.reserve(static_cast<size_t>(active_edge_ids.shape(0)));
    int64_t blocked_by_bridge = 0;
    int64_t blocked_by_d_min = 0;
    for (ssize_t idx = 0; idx < active_edge_ids.shape(0); ++idx) {
        const int64_t edge_id = active_edge_ids(idx);
        const int u = static_cast<int>(edge_array_by_id(edge_id, 0));
        const int v = static_cast<int>(edge_array_by_id(edge_id, 1));
        if (guard_bridges && bridge_codes.find(encode_pair(u, v)) != bridge_codes.end()) {
            ++blocked_by_bridge;
            continue;
        }
        if (degrees(u) - 1 < d_min || degrees(v) - 1 < d_min) {
            ++blocked_by_d_min;
            continue;
        }
        eligible_ids.push_back(edge_id);
    }
    const double eligibility_runtime_sec = std::chrono::duration<double>(Clock::now() - eligibility_start).count();

    py::array_t<int64_t> eligible_ids_array(static_cast<ssize_t>(eligible_ids.size()));
    auto eligible_ids_view = eligible_ids_array.mutable_unchecked<1>();
    for (size_t idx = 0; idx < eligible_ids.size(); ++idx) {
        eligible_ids_view(static_cast<ssize_t>(idx)) = eligible_ids[idx];
    }

    py::dict result;
    result["eligible_edge_ids"] = std::move(eligible_ids_array);
    result["eligible_count"] = static_cast<int64_t>(eligible_ids.size());
    result["blocked_by_bridge_count"] = blocked_by_bridge;
    result["blocked_by_d_min_count"] = blocked_by_d_min;
    result["bridge_count"] = static_cast<int64_t>(bridge_codes.size());
    result["bridge_runtime_sec"] = bridge_runtime_sec;
    result["eligibility_runtime_sec"] = eligibility_runtime_sec;
    return result;
}

py::dict eligible_edge_id_partitions_from_csr(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> row_ptr_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> col_idx_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> active_edge_ids_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> edge_array_by_id_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> degrees_array,
    const int d_min,
    const bool guard_bridges,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array,
    const bool use_score_cache
) {
    const auto row_ptr = row_ptr_array.unchecked<1>();
    const auto col_idx = col_idx_array.unchecked<1>();
    const auto active_edge_ids = active_edge_ids_array.unchecked<1>();
    const auto edge_array_by_id = edge_array_by_id_array.unchecked<2>();
    const auto degrees = degrees_array.unchecked<1>();
    const auto valid_score_cache = valid_score_cache_array.unchecked<1>();
    if (edge_array_by_id.shape(1) != 2) {
        throw std::runtime_error("edge_array_by_id must have shape [num_edges, 2].");
    }
    if (valid_score_cache.shape(0) < edge_array_by_id.shape(0)) {
        throw std::runtime_error("valid_score_cache must have at least one entry per edge id.");
    }

    using Clock = std::chrono::high_resolution_clock;
    const int num_nodes = static_cast<int>(row_ptr.shape(0) - 1);
    std::unordered_set<uint64_t> bridge_codes;
    const auto bridge_start = Clock::now();
    if (guard_bridges) {
        std::vector<int> discovery(static_cast<size_t>(num_nodes), -1);
        std::vector<int> low(static_cast<size_t>(num_nodes), 0);
        std::vector<int> parent(static_cast<size_t>(num_nodes), -1);
        int visit_time = 0;
        bridge_codes.reserve(static_cast<size_t>(active_edge_ids.shape(0) / 8 + 1));
        std::function<void(int)> visit = [&](const int node) {
            discovery[static_cast<size_t>(node)] = visit_time;
            low[static_cast<size_t>(node)] = visit_time;
            ++visit_time;
            for (int64_t idx = row_ptr(node); idx < row_ptr(node + 1); ++idx) {
                const int neighbor = static_cast<int>(col_idx(idx));
                if (discovery[static_cast<size_t>(neighbor)] == -1) {
                    parent[static_cast<size_t>(neighbor)] = node;
                    visit(neighbor);
                    low[static_cast<size_t>(node)] = std::min(
                        low[static_cast<size_t>(node)],
                        low[static_cast<size_t>(neighbor)]
                    );
                    if (low[static_cast<size_t>(neighbor)] > discovery[static_cast<size_t>(node)]) {
                        bridge_codes.insert(encode_pair(node, neighbor));
                    }
                } else if (neighbor != parent[static_cast<size_t>(node)]) {
                    low[static_cast<size_t>(node)] = std::min(
                        low[static_cast<size_t>(node)],
                        discovery[static_cast<size_t>(neighbor)]
                    );
                }
            }
        };
        for (int node = 0; node < num_nodes; ++node) {
            if (discovery[static_cast<size_t>(node)] == -1) {
                visit(node);
            }
        }
    }
    const double bridge_runtime_sec = std::chrono::duration<double>(Clock::now() - bridge_start).count();

    const auto eligibility_start = Clock::now();
    std::vector<int64_t> eligible_ids;
    std::vector<int64_t> rescored_ids;
    std::vector<int64_t> reused_ids;
    eligible_ids.reserve(static_cast<size_t>(active_edge_ids.shape(0)));
    rescored_ids.reserve(static_cast<size_t>(active_edge_ids.shape(0)));
    reused_ids.reserve(static_cast<size_t>(active_edge_ids.shape(0) / 2 + 1));
    int64_t blocked_by_bridge = 0;
    int64_t blocked_by_d_min = 0;
    for (ssize_t idx = 0; idx < active_edge_ids.shape(0); ++idx) {
        const int64_t edge_id = active_edge_ids(idx);
        const int u = static_cast<int>(edge_array_by_id(edge_id, 0));
        const int v = static_cast<int>(edge_array_by_id(edge_id, 1));
        if (guard_bridges && bridge_codes.find(encode_pair(u, v)) != bridge_codes.end()) {
            ++blocked_by_bridge;
            continue;
        }
        if (degrees(u) - 1 < d_min || degrees(v) - 1 < d_min) {
            ++blocked_by_d_min;
            continue;
        }
        eligible_ids.push_back(edge_id);
        if (use_score_cache && valid_score_cache(edge_id) != 0) {
            reused_ids.push_back(edge_id);
        } else {
            rescored_ids.push_back(edge_id);
        }
    }
    const double eligibility_runtime_sec = std::chrono::duration<double>(Clock::now() - eligibility_start).count();

    auto to_array = [](const std::vector<int64_t>& values) {
        py::array_t<int64_t> array(static_cast<ssize_t>(values.size()));
        auto view = array.mutable_unchecked<1>();
        for (size_t idx = 0; idx < values.size(); ++idx) {
            view(static_cast<ssize_t>(idx)) = values[idx];
        }
        return array;
    };

    py::dict result;
    result["eligible_edge_ids"] = to_array(eligible_ids);
    result["rescored_edge_ids"] = to_array(rescored_ids);
    result["reused_edge_ids"] = to_array(reused_ids);
    result["eligible_count"] = static_cast<int64_t>(eligible_ids.size());
    result["rescored_count"] = static_cast<int64_t>(rescored_ids.size());
    result["reused_count"] = static_cast<int64_t>(reused_ids.size());
    result["blocked_by_bridge_count"] = blocked_by_bridge;
    result["blocked_by_d_min_count"] = blocked_by_d_min;
    result["bridge_count"] = static_cast<int64_t>(bridge_codes.size());
    result["bridge_runtime_sec"] = bridge_runtime_sec;
    result["eligibility_runtime_sec"] = eligibility_runtime_sec;
    result["cache_partition_runtime_sec"] = 0.0;
    return result;
}

py::dict score_edges_round_best(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> row_ptr_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> col_idx_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_edges_array,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_edge_ids_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> current_raw_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> current_std_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> stats_mean_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> stats_std_array,
    const std::string& score_mode,
    const double eps,
    py::array_t<double, py::array::c_style | py::array::forcecast> score_cache_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> degree_score_cache_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> support_score_cache_array,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array,
    const std::string& kernel_variant,
    const bool profile_native_kernel,
    py::object candidate_delta_cache_object = py::none(),
    py::object delta_valid_cache_object = py::none()
) {
    const auto row_ptr = row_ptr_array.unchecked<1>();
    const auto col_idx = col_idx_array.unchecked<1>();
    const auto candidate_edges = candidate_edges_array.unchecked<2>();
    const auto candidate_edge_ids = candidate_edge_ids_array.unchecked<1>();
    const auto current_raw = current_raw_array.unchecked<2>();
    const auto current_std = current_std_array.unchecked<2>();
    const auto stats_mean = stats_mean_array.unchecked<1>();
    const auto stats_std = stats_std_array.unchecked<1>();
    auto score_cache = score_cache_array.mutable_unchecked<1>();
    auto degree_score_cache = degree_score_cache_array.mutable_unchecked<1>();
    auto support_score_cache = support_score_cache_array.mutable_unchecked<1>();
    auto valid_score_cache = valid_score_cache_array.mutable_unchecked<1>();
    const bool write_delta_cache = !candidate_delta_cache_object.is_none() && !delta_valid_cache_object.is_none();
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_delta_cache_array;
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> delta_valid_cache_array;
    int64_t* candidate_delta_cache_ptr = nullptr;
    uint8_t* delta_valid_cache_ptr = nullptr;

    if (candidate_edges.shape(1) != 2) {
        throw std::runtime_error("candidate_edges must have shape [num_edges, 2].");
    }
    if (candidate_edge_ids.shape(0) != candidate_edges.shape(0)) {
        throw std::runtime_error("candidate_edge_ids length must match candidate_edges.");
    }
    if (current_raw.shape(1) != kOrbitDim || current_std.shape(1) != kOrbitDim) {
        throw std::runtime_error("current_raw/current_std must have 15 columns.");
    }
    if (kernel_variant != "mask_count_v4_combinatorial") {
        throw std::runtime_error("Unsupported RelShift native kernel variant: " + kernel_variant);
    }
    if (write_delta_cache) {
        candidate_delta_cache_array = candidate_delta_cache_object.cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
        delta_valid_cache_array = delta_valid_cache_object.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        if (
            candidate_delta_cache_array.ndim() != 3 ||
            candidate_delta_cache_array.shape(1) != 2 ||
            candidate_delta_cache_array.shape(2) != kOrbitDim
        ) {
            throw std::runtime_error("candidate_delta_cache must have shape [num_edges, 2, 15].");
        }
        if (delta_valid_cache_array.ndim() != 1 || delta_valid_cache_array.shape(0) < candidate_delta_cache_array.shape(0)) {
            throw std::runtime_error("delta_valid_cache must have one entry per edge id.");
        }
        candidate_delta_cache_ptr = static_cast<int64_t*>(candidate_delta_cache_array.mutable_data());
        delta_valid_cache_ptr = static_cast<uint8_t*>(delta_valid_cache_array.mutable_data());
    }

    const int num_nodes = static_cast<int>(current_raw.shape(0));
    const ssize_t num_edges = candidate_edges.shape(0);
    bool has_best = false;
    EdgeKey best{0.0, 0.0, 0.0, -1};
    double pair_generation_sec = 0.0;
    double delta_accumulation_sec = 0.0;
    double score_scalarization_sec = 0.0;
    int64_t directly_attached_total = 0;
    int64_t four_node_pair_total = 0;

    auto score_candidate = [&](
        const ssize_t edge_idx,
        std::vector<int>& local_marks,
        std::vector<int>& local_attachment_masks,
        std::vector<int>& local_directly_attached,
        std::array<int64_t, 64>& local_four_node_mask_counts,
        int& local_epoch,
        double* local_pair_generation_sec,
        double* local_delta_accumulation_sec,
        double* local_score_scalarization_sec
    ) {
        const int u = static_cast<int>(candidate_edges(edge_idx, 0));
        const int v = static_cast<int>(candidate_edges(edge_idx, 1));
        const int64_t edge_id = candidate_edge_ids(edge_idx);
        if (edge_id < 0 || edge_id >= valid_score_cache.shape(0)) {
            throw std::runtime_error("candidate_edge_ids contains invalid edge id.");
        }
        std::array<int64_t, 2 * kOrbitDim> endpoint_delta_cache{};
        const ScoreResult scored = score_single_edge_mask_count(
            row_ptr.data(0),
            col_idx.data(0),
            u,
            v,
            current_raw,
            current_std,
            stats_mean,
            stats_std,
            score_mode,
            eps,
            false,
            local_marks,
            local_attachment_masks,
            local_directly_attached,
            local_four_node_mask_counts,
            local_epoch,
            local_pair_generation_sec,
            local_delta_accumulation_sec,
            local_score_scalarization_sec,
            endpoint_delta_cache.data()
        );
        const double degree_score =
            static_cast<double>((row_ptr[u + 1] - row_ptr[u]) + (row_ptr[v + 1] - row_ptr[v]));
        const double support_score = static_cast<double>(edge_support(row_ptr.data(0), col_idx.data(0), u, v));

        score_cache(edge_id) = scored.score;
        degree_score_cache(edge_id) = degree_score;
        support_score_cache(edge_id) = support_score;
        valid_score_cache(edge_id) = 1;
        if (write_delta_cache) {
            if (edge_id >= candidate_delta_cache_array.shape(0)) {
                throw std::runtime_error("candidate_delta_cache has fewer rows than candidate edge ids require.");
            }
            const int64_t offset = edge_id * 2 * kOrbitDim;
            for (int idx = 0; idx < 2 * kOrbitDim; ++idx) {
                candidate_delta_cache_ptr[offset + idx] = endpoint_delta_cache[static_cast<size_t>(idx)];
            }
            delta_valid_cache_ptr[edge_id] = 1;
        }
        return std::pair<ScoreResult, EdgeKey>{
            scored,
            EdgeKey{scored.score, degree_score, support_score, edge_id},
        };
    };

    const bool use_parallel = !profile_native_kernel && num_edges >= 64;
    if (!use_parallel) {
        std::vector<int> marks(static_cast<size_t>(num_nodes), 0);
        std::vector<int> attachment_masks(static_cast<size_t>(num_nodes), 0);
        std::vector<int> directly_attached;
        std::array<int64_t, 64> four_node_mask_counts{};
        int epoch = 0;
        for (ssize_t edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
            const auto [scored, candidate_key] = score_candidate(
                edge_idx,
                marks,
                attachment_masks,
                directly_attached,
                four_node_mask_counts,
                epoch,
                profile_native_kernel ? &pair_generation_sec : nullptr,
                profile_native_kernel ? &delta_accumulation_sec : nullptr,
                profile_native_kernel ? &score_scalarization_sec : nullptr
            );
            directly_attached_total += scored.directly_attached_size;
            four_node_pair_total += scored.four_node_pair_count;
            if (!has_best || edge_key_less(candidate_key, best)) {
                best = candidate_key;
                has_best = true;
            }
        }
    } else {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<int> marks(static_cast<size_t>(num_nodes), 0);
            std::vector<int> attachment_masks(static_cast<size_t>(num_nodes), 0);
            std::vector<int> directly_attached;
            std::array<int64_t, 64> four_node_mask_counts{};
            int epoch = 0;
            bool local_has_best = false;
            EdgeKey local_best{0.0, 0.0, 0.0, -1};
            int64_t local_directly_attached_total = 0;
            int64_t local_four_node_pair_total = 0;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 32)
#endif
            for (ssize_t edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
                const auto [scored, candidate_key] = score_candidate(
                    edge_idx,
                    marks,
                    attachment_masks,
                    directly_attached,
                    four_node_mask_counts,
                    epoch,
                    nullptr,
                    nullptr,
                    nullptr
                );
                local_directly_attached_total += scored.directly_attached_size;
                local_four_node_pair_total += scored.four_node_pair_count;
                if (!local_has_best || edge_key_less(candidate_key, local_best)) {
                    local_best = candidate_key;
                    local_has_best = true;
                }
            }
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                directly_attached_total += local_directly_attached_total;
                four_node_pair_total += local_four_node_pair_total;
                if (local_has_best && (!has_best || edge_key_less(local_best, best))) {
                    best = local_best;
                    has_best = true;
                }
            }
        }
    }

    py::dict result;
    result["best_edge_id"] = static_cast<int64_t>(has_best ? best.edge_id : -1);
    result["best_score"] = has_best ? best.score : std::numeric_limits<double>::infinity();
    result["best_degree_score"] = has_best ? best.degree_score : std::numeric_limits<double>::infinity();
    result["best_support_score"] = has_best ? best.support_score : std::numeric_limits<double>::infinity();
    result["rescored_count"] = static_cast<int64_t>(num_edges);
    result["avg_directly_attached_size"] = num_edges == 0 ? 0.0 : static_cast<double>(directly_attached_total) / static_cast<double>(num_edges);
    result["avg_four_node_pair_count"] = num_edges == 0 ? 0.0 : static_cast<double>(four_node_pair_total) / static_cast<double>(num_edges);
    result["native_pair_generation_runtime_sec"] = pair_generation_sec;
    result["native_delta_accumulation_runtime_sec"] = delta_accumulation_sec;
    result["native_score_scalarization_runtime_sec"] = score_scalarization_sec;
    return result;
}

class NativeGraphState {
public:
    NativeGraphState(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> row_ptr_array,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> col_idx_array
    ) {
        initialize_csr(row_ptr_array, col_idx_array);
    }

    NativeGraphState(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> row_ptr_array,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> col_idx_array,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> edge_array_by_id_array
    ) {
        initialize_csr(row_ptr_array, col_idx_array);
        initialize_edge_ids(edge_array_by_id_array);
    }

    void initialize_csr(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> row_ptr_array,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> col_idx_array
    ) {
        const auto row_ptr = row_ptr_array.unchecked<1>();
        const auto col_idx = col_idx_array.unchecked<1>();
        if (row_ptr.shape(0) < 1) {
            throw std::runtime_error("row_ptr must contain at least one element.");
        }
        row_ptr_.clear();
        col_idx_.clear();
        row_ptr_.reserve(static_cast<size_t>(row_ptr.shape(0)));
        for (ssize_t idx = 0; idx < row_ptr.shape(0); ++idx) {
            row_ptr_.push_back(row_ptr(idx));
        }
        col_idx_.reserve(static_cast<size_t>(col_idx.shape(0)));
        for (ssize_t idx = 0; idx < col_idx.shape(0); ++idx) {
            col_idx_.push_back(col_idx(idx));
        }
        if (row_ptr_.front() != 0 || row_ptr_.back() != static_cast<int64_t>(col_idx_.size())) {
            throw std::runtime_error("Invalid CSR graph state: row_ptr boundaries do not match col_idx length.");
        }
        for (size_t idx = 1; idx < row_ptr_.size(); ++idx) {
            if (row_ptr_[idx] < row_ptr_[idx - 1]) {
                throw std::runtime_error("Invalid CSR graph state: row_ptr must be nondecreasing.");
            }
        }
        num_nodes_ = static_cast<int>(row_ptr_.size() - 1);
        current_degrees_.assign(static_cast<size_t>(num_nodes_), 0);
        for (int node = 0; node < num_nodes_; ++node) {
            current_degrees_[static_cast<size_t>(node)] =
                row_ptr_[static_cast<size_t>(node + 1)] - row_ptr_[static_cast<size_t>(node)];
        }
    }

    void initialize_edge_ids(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> edge_array_by_id_array
    ) {
        const auto edge_array_by_id = edge_array_by_id_array.unchecked<2>();
        if (edge_array_by_id.shape(1) != 2) {
            throw std::runtime_error("edge_array_by_id must have shape [num_edges, 2].");
        }
        edge_by_id_.clear();
        edge_code_to_id_.clear();
        incident_edge_ids_.assign(static_cast<size_t>(num_nodes_), {});
        active_edge_mask_.assign(static_cast<size_t>(edge_array_by_id.shape(0)), static_cast<uint8_t>(1));
        edge_by_id_.reserve(static_cast<size_t>(edge_array_by_id.shape(0)));
        edge_code_to_id_.reserve(static_cast<size_t>(edge_array_by_id.shape(0) * 2 + 1));
        for (ssize_t edge_id = 0; edge_id < edge_array_by_id.shape(0); ++edge_id) {
            const int u = static_cast<int>(edge_array_by_id(edge_id, 0));
            const int v = static_cast<int>(edge_array_by_id(edge_id, 1));
            if (u < 0 || v < 0 || u >= num_nodes_ || v >= num_nodes_) {
                throw std::runtime_error("edge_array_by_id contains node id outside graph range.");
            }
            const int left = std::min(u, v);
            const int right = std::max(u, v);
            edge_by_id_.push_back(std::array<int, 2>{left, right});
            edge_code_to_id_[encode_pair(left, right)] = static_cast<int64_t>(edge_id);
            incident_edge_ids_[static_cast<size_t>(left)].push_back(static_cast<int64_t>(edge_id));
            incident_edge_ids_[static_cast<size_t>(right)].push_back(static_cast<int64_t>(edge_id));
        }
        adjacency_edge_ids_.assign(col_idx_.size(), static_cast<int64_t>(-1));
        for (int node = 0; node < num_nodes_; ++node) {
            for (
                int64_t idx = row_ptr_[static_cast<size_t>(node)];
                idx < row_ptr_[static_cast<size_t>(node + 1)];
                ++idx
            ) {
                const int neighbor = static_cast<int>(col_idx_[static_cast<size_t>(idx)]);
                const auto found = edge_code_to_id_.find(encode_pair(node, neighbor));
                if (found == edge_code_to_id_.end()) {
                    throw std::runtime_error(
                        "CSR contains an adjacency entry that is missing from edge_array_by_id."
                    );
                }
                adjacency_edge_ids_[static_cast<size_t>(idx)] = found->second;
            }
        }
        original_edge_count_ = static_cast<int64_t>(edge_by_id_.size());
        active_edge_count_ = original_edge_count_;
        has_edge_ids_ = true;
    }

    py::dict eligible_edge_id_partitions(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> active_edge_ids_array,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> edge_array_by_id_array,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> degrees_array,
        const int d_min,
        const bool guard_bridges,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array,
        const bool use_score_cache
    ) const {
        if (has_edge_ids_) {
            return eligible_edge_id_partitions_with_cached_best(
                active_edge_ids_array,
                degrees_array,
                d_min,
                guard_bridges,
                valid_score_cache_array,
                use_score_cache,
                py::none(),
                py::none(),
                py::none()
            );
        }
        return eligible_edge_id_partitions_from_csr(
            row_ptr_array(),
            col_idx_array(),
            active_edge_ids_array,
            edge_array_by_id_array,
            degrees_array,
            d_min,
            guard_bridges,
            valid_score_cache_array,
            use_score_cache
        );
    }

    py::dict eligible_edge_id_partitions_with_cached_best(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> active_edge_ids_array,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> degrees_array,
        const int d_min,
        const bool guard_bridges,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array,
        const bool use_score_cache,
        py::object score_cache_object,
        py::object degree_score_cache_object,
        py::object support_score_cache_object,
        py::object delta_valid_cache_object = py::none()
    ) const {
        if (!has_edge_ids_) {
            throw std::runtime_error("NativeGraphState cached-best partition requires edge id state.");
        }
        const auto active_edge_ids = active_edge_ids_array.unchecked<1>();
        const auto degrees = degrees_array.unchecked<1>();
        const auto valid_score_cache = valid_score_cache_array.unchecked<1>();
        const bool compute_cached_best =
            use_score_cache &&
            !score_cache_object.is_none() &&
            !degree_score_cache_object.is_none() &&
            !support_score_cache_object.is_none();
        const bool split_delta_refresh = use_score_cache && !delta_valid_cache_object.is_none();

        py::array_t<double, py::array::c_style | py::array::forcecast> score_cache_array;
        py::array_t<double, py::array::c_style | py::array::forcecast> degree_score_cache_array;
        py::array_t<double, py::array::c_style | py::array::forcecast> support_score_cache_array;
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> delta_valid_cache_array;
        const uint8_t* delta_valid_cache_ptr = nullptr;
        ssize_t delta_valid_cache_size = 0;
        if (compute_cached_best) {
            score_cache_array = score_cache_object.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            degree_score_cache_array = degree_score_cache_object.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            support_score_cache_array = support_score_cache_object.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        }
        if (split_delta_refresh) {
            delta_valid_cache_array = delta_valid_cache_object.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
            if (delta_valid_cache_array.ndim() != 1) {
                throw std::runtime_error("delta_valid_cache must be a one-dimensional array.");
            }
            delta_valid_cache_ptr = static_cast<const uint8_t*>(delta_valid_cache_array.data());
            delta_valid_cache_size = delta_valid_cache_array.shape(0);
        }

        using Clock = std::chrono::high_resolution_clock;
        std::unordered_set<uint64_t> bridge_codes;
        int64_t bridge_nodes_visited = 0;
        int64_t bridge_adjacency_entries_visited = 0;
        int64_t bridge_inactive_adjacency_entries_skipped = 0;
        const auto bridge_start = Clock::now();
        if (guard_bridges) {
            std::vector<int> discovery(static_cast<size_t>(num_nodes_), -1);
            std::vector<int> low(static_cast<size_t>(num_nodes_), 0);
            std::vector<int> parent(static_cast<size_t>(num_nodes_), -1);
            int visit_time = 0;
            bridge_codes.reserve(static_cast<size_t>(active_edge_ids.shape(0) / 8 + 1));
            std::function<void(int)> visit = [&](const int node) {
                ++bridge_nodes_visited;
                discovery[static_cast<size_t>(node)] = visit_time;
                low[static_cast<size_t>(node)] = visit_time;
                ++visit_time;
                for (int64_t idx = row_ptr_[static_cast<size_t>(node)]; idx < row_ptr_[static_cast<size_t>(node + 1)]; ++idx) {
                    ++bridge_adjacency_entries_visited;
                    if (!adjacency_entry_is_active(
                            idx,
                            adjacency_edge_ids_.data(),
                            active_edge_mask_.data()
                        )) {
                        ++bridge_inactive_adjacency_entries_skipped;
                        continue;
                    }
                    const int neighbor = static_cast<int>(col_idx_[static_cast<size_t>(idx)]);
                    if (discovery[static_cast<size_t>(neighbor)] == -1) {
                        parent[static_cast<size_t>(neighbor)] = node;
                        visit(neighbor);
                        low[static_cast<size_t>(node)] = std::min(
                            low[static_cast<size_t>(node)],
                            low[static_cast<size_t>(neighbor)]
                        );
                        if (low[static_cast<size_t>(neighbor)] > discovery[static_cast<size_t>(node)]) {
                            bridge_codes.insert(encode_pair(node, neighbor));
                        }
                    } else if (neighbor != parent[static_cast<size_t>(node)]) {
                        low[static_cast<size_t>(node)] = std::min(
                            low[static_cast<size_t>(node)],
                            discovery[static_cast<size_t>(neighbor)]
                        );
                    }
                }
            };
            for (int node = 0; node < num_nodes_; ++node) {
                if (discovery[static_cast<size_t>(node)] == -1) {
                    visit(node);
                }
            }
        }
        const double bridge_runtime_sec = std::chrono::duration<double>(Clock::now() - bridge_start).count();

        const auto eligibility_start = Clock::now();
        std::vector<int64_t> eligible_ids;
        std::vector<int64_t> rescored_ids;
        std::vector<int64_t> reused_ids;
        std::vector<int64_t> refresh_ids;
        eligible_ids.reserve(static_cast<size_t>(active_edge_ids.shape(0)));
        rescored_ids.reserve(static_cast<size_t>(active_edge_ids.shape(0)));
        reused_ids.reserve(static_cast<size_t>(active_edge_ids.shape(0) / 4 + 1));
        refresh_ids.reserve(static_cast<size_t>(active_edge_ids.shape(0) / 4 + 1));
        int64_t blocked_by_bridge = 0;
        int64_t blocked_by_d_min = 0;
        int64_t active_edge_id_entries_scanned = 0;
        int64_t inactive_edge_ids_skipped = 0;
        bool has_cached_best = false;
        EdgeKey cached_best{0.0, 0.0, 0.0, -1};

        auto update_cached_best = [&](const int64_t edge_id) {
            if (!compute_cached_best) {
                return;
            }
            const auto score_cache = score_cache_array.unchecked<1>();
            const auto degree_score_cache = degree_score_cache_array.unchecked<1>();
            const auto support_score_cache = support_score_cache_array.unchecked<1>();
            const EdgeKey key{score_cache(edge_id), degree_score_cache(edge_id), support_score_cache(edge_id), edge_id};
            if (!has_cached_best || edge_key_less(key, cached_best)) {
                cached_best = key;
                has_cached_best = true;
            }
        };

        for (ssize_t idx = 0; idx < active_edge_ids.shape(0); ++idx) {
            ++active_edge_id_entries_scanned;
            const int64_t edge_id = active_edge_ids(idx);
            if (edge_id < 0 || edge_id >= static_cast<int64_t>(edge_by_id_.size())) {
                throw std::runtime_error("active_edge_ids contains invalid edge id.");
            }
            if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                ++inactive_edge_ids_skipped;
                continue;
            }
            const int u = edge_by_id_[static_cast<size_t>(edge_id)][0];
            const int v = edge_by_id_[static_cast<size_t>(edge_id)][1];
            if (guard_bridges && bridge_codes.find(encode_pair(u, v)) != bridge_codes.end()) {
                ++blocked_by_bridge;
                continue;
            }
            if (degrees(u) - 1 < d_min || degrees(v) - 1 < d_min) {
                ++blocked_by_d_min;
                continue;
            }
            eligible_ids.push_back(edge_id);
            if (use_score_cache && valid_score_cache(edge_id) != 0) {
                reused_ids.push_back(edge_id);
                update_cached_best(edge_id);
            } else if (split_delta_refresh && edge_id < delta_valid_cache_size && delta_valid_cache_ptr[edge_id] != 0) {
                refresh_ids.push_back(edge_id);
            } else {
                rescored_ids.push_back(edge_id);
            }
        }
        const double eligibility_runtime_sec = std::chrono::duration<double>(Clock::now() - eligibility_start).count();

        auto to_array = [](const std::vector<int64_t>& values) {
            py::array_t<int64_t> array(static_cast<ssize_t>(values.size()));
            auto view = array.mutable_unchecked<1>();
            for (size_t idx = 0; idx < values.size(); ++idx) {
                view(static_cast<ssize_t>(idx)) = values[idx];
            }
            return array;
        };

        py::dict result;
        result["eligible_edge_ids"] = to_array(eligible_ids);
        result["rescored_edge_ids"] = to_array(rescored_ids);
        result["reused_edge_ids"] = to_array(reused_ids);
        result["refresh_edge_ids"] = to_array(refresh_ids);
        result["eligible_count"] = static_cast<int64_t>(eligible_ids.size());
        result["rescored_count"] = static_cast<int64_t>(rescored_ids.size());
        result["reused_count"] = static_cast<int64_t>(reused_ids.size());
        result["refresh_count"] = static_cast<int64_t>(refresh_ids.size());
        result["blocked_by_bridge_count"] = blocked_by_bridge;
        result["blocked_by_d_min_count"] = blocked_by_d_min;
        result["bridge_count"] = static_cast<int64_t>(bridge_codes.size());
        result["bridge_runtime_sec"] = bridge_runtime_sec;
        result["bridge_nodes_visited"] = bridge_nodes_visited;
        result["bridge_adjacency_entries_visited"] = bridge_adjacency_entries_visited;
        result["bridge_inactive_adjacency_entries_skipped"] = bridge_inactive_adjacency_entries_skipped;
        result["eligibility_runtime_sec"] = eligibility_runtime_sec;
        result["active_edge_id_entries_scanned"] = active_edge_id_entries_scanned;
        result["inactive_edge_ids_skipped"] = inactive_edge_ids_skipped;
        result["cache_partition_runtime_sec"] = 0.0;
        result["cached_best_edge_id"] = static_cast<int64_t>(has_cached_best ? cached_best.edge_id : -1);
        return result;
    }

    py::dict score_edges_round_best_state(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_edges_array,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_edge_ids_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> current_raw_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> current_std_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> stats_mean_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> stats_std_array,
        const std::string& score_mode,
        const double eps,
        py::array_t<double, py::array::c_style | py::array::forcecast> score_cache_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> degree_score_cache_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> support_score_cache_array,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array,
        const std::string& kernel_variant,
        const bool profile_native_kernel,
        py::object candidate_delta_cache_object = py::none(),
        py::object delta_valid_cache_object = py::none()
    ) const {
        (void)candidate_edges_array;
        return score_edge_ids_round_best(
            candidate_edge_ids_array,
            current_raw_array,
            current_std_array,
            stats_mean_array,
            stats_std_array,
            score_mode,
            eps,
            score_cache_array,
            degree_score_cache_array,
            support_score_cache_array,
            valid_score_cache_array,
            kernel_variant,
            profile_native_kernel,
            candidate_delta_cache_object,
            delta_valid_cache_object
        );
    }

    py::dict score_edge_ids_round_best(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_edge_ids_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> current_raw_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> current_std_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> stats_mean_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> stats_std_array,
        const std::string& score_mode,
        const double eps,
        py::array_t<double, py::array::c_style | py::array::forcecast> score_cache_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> degree_score_cache_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> support_score_cache_array,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array,
        const std::string& kernel_variant,
        const bool profile_native_kernel,
        py::object candidate_delta_cache_object = py::none(),
        py::object delta_valid_cache_object = py::none()
    ) const {
        if (!has_edge_ids_) {
            throw std::runtime_error("NativeGraphState score_edge_ids_round_best requires edge id state.");
        }
        const auto candidate_edge_ids = candidate_edge_ids_array.unchecked<1>();
        const auto current_raw = current_raw_array.unchecked<2>();
        const auto current_std = current_std_array.unchecked<2>();
        const auto stats_mean = stats_mean_array.unchecked<1>();
        const auto stats_std = stats_std_array.unchecked<1>();
        auto score_cache = score_cache_array.mutable_unchecked<1>();
        auto degree_score_cache = degree_score_cache_array.mutable_unchecked<1>();
        auto support_score_cache = support_score_cache_array.mutable_unchecked<1>();
        auto valid_score_cache = valid_score_cache_array.mutable_unchecked<1>();
        const bool write_delta_cache = !candidate_delta_cache_object.is_none() && !delta_valid_cache_object.is_none();
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_delta_cache_array;
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> delta_valid_cache_array;
        int64_t* candidate_delta_cache_ptr = nullptr;
        uint8_t* delta_valid_cache_ptr = nullptr;

        if (current_raw.shape(1) != kOrbitDim || current_std.shape(1) != kOrbitDim) {
            throw std::runtime_error("current_raw/current_std must have 15 columns.");
        }
        if (kernel_variant != "mask_count_v4_combinatorial") {
            throw std::runtime_error("Unsupported RelShift native kernel variant: " + kernel_variant);
        }
        if (write_delta_cache) {
            candidate_delta_cache_array = candidate_delta_cache_object.cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
            delta_valid_cache_array = delta_valid_cache_object.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
            if (
                candidate_delta_cache_array.ndim() != 3 ||
                candidate_delta_cache_array.shape(1) != 2 ||
                candidate_delta_cache_array.shape(2) != kOrbitDim
            ) {
                throw std::runtime_error("candidate_delta_cache must have shape [num_edges, 2, 15].");
            }
            if (delta_valid_cache_array.ndim() != 1 || delta_valid_cache_array.shape(0) < candidate_delta_cache_array.shape(0)) {
                throw std::runtime_error("delta_valid_cache must have one entry per edge id.");
            }
            candidate_delta_cache_ptr = static_cast<int64_t*>(candidate_delta_cache_array.mutable_data());
            delta_valid_cache_ptr = static_cast<uint8_t*>(delta_valid_cache_array.mutable_data());
        }

        const ssize_t num_edges = candidate_edge_ids.shape(0);
        bool has_best = false;
        EdgeKey best{0.0, 0.0, 0.0, -1};
        double pair_generation_sec = 0.0;
        double delta_accumulation_sec = 0.0;
        double score_scalarization_sec = 0.0;
        int64_t directly_attached_total = 0;
        int64_t four_node_pair_total = 0;

        auto score_candidate = [&](
            const ssize_t edge_idx,
            std::vector<int>& local_marks,
            std::vector<int>& local_attachment_masks,
            std::vector<int>& local_directly_attached,
            std::array<int64_t, 64>& local_four_node_mask_counts,
            int& local_epoch,
            double* local_pair_generation_sec,
            double* local_delta_accumulation_sec,
            double* local_score_scalarization_sec
        ) {
            const int64_t edge_id = candidate_edge_ids(edge_idx);
            if (edge_id < 0 || edge_id >= static_cast<int64_t>(edge_by_id_.size())) {
                throw std::runtime_error("candidate_edge_ids contains invalid edge id.");
            }
            const int u = edge_by_id_[static_cast<size_t>(edge_id)][0];
            const int v = edge_by_id_[static_cast<size_t>(edge_id)][1];
            std::array<int64_t, 2 * kOrbitDim> endpoint_delta_cache{};
            const ScoreResult scored = score_single_edge_mask_count(
                row_ptr_.data(),
                col_idx_.data(),
                u,
                v,
                current_raw,
                current_std,
                stats_mean,
                stats_std,
                score_mode,
                eps,
                false,
                local_marks,
                local_attachment_masks,
                local_directly_attached,
                local_four_node_mask_counts,
                local_epoch,
                local_pair_generation_sec,
                local_delta_accumulation_sec,
                local_score_scalarization_sec,
                endpoint_delta_cache.data(),
                adjacency_edge_ids_.data(),
                active_edge_mask_.data()
            );
            const double degree_score =
                static_cast<double>(
                    current_degrees_[static_cast<size_t>(u)] +
                    current_degrees_[static_cast<size_t>(v)]
                );
            const double support_score = static_cast<double>(edge_support(
                row_ptr_.data(),
                col_idx_.data(),
                u,
                v,
                adjacency_edge_ids_.data(),
                active_edge_mask_.data()
            ));

            score_cache(edge_id) = scored.score;
            degree_score_cache(edge_id) = degree_score;
            support_score_cache(edge_id) = support_score;
            valid_score_cache(edge_id) = 1;
            if (write_delta_cache) {
                if (edge_id >= candidate_delta_cache_array.shape(0)) {
                    throw std::runtime_error("candidate_delta_cache has fewer rows than candidate edge ids require.");
                }
                const int64_t offset = edge_id * 2 * kOrbitDim;
                for (int idx = 0; idx < 2 * kOrbitDim; ++idx) {
                    candidate_delta_cache_ptr[offset + idx] = endpoint_delta_cache[static_cast<size_t>(idx)];
                }
                delta_valid_cache_ptr[edge_id] = 1;
            }
            return std::pair<ScoreResult, EdgeKey>{
                scored,
                EdgeKey{scored.score, degree_score, support_score, edge_id},
            };
        };

        const bool use_parallel = !profile_native_kernel && num_edges >= 64;
        if (!use_parallel) {
            std::vector<int> marks(static_cast<size_t>(num_nodes_), 0);
            std::vector<int> attachment_masks(static_cast<size_t>(num_nodes_), 0);
            std::vector<int> directly_attached;
            std::array<int64_t, 64> four_node_mask_counts{};
            int epoch = 0;
            for (ssize_t edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
                const auto [scored, candidate_key] = score_candidate(
                    edge_idx,
                    marks,
                    attachment_masks,
                    directly_attached,
                    four_node_mask_counts,
                    epoch,
                    profile_native_kernel ? &pair_generation_sec : nullptr,
                    profile_native_kernel ? &delta_accumulation_sec : nullptr,
                    profile_native_kernel ? &score_scalarization_sec : nullptr
                );
                directly_attached_total += scored.directly_attached_size;
                four_node_pair_total += scored.four_node_pair_count;
                if (!has_best || edge_key_less(candidate_key, best)) {
                    best = candidate_key;
                    has_best = true;
                }
            }
        } else {
            py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                std::vector<int> marks(static_cast<size_t>(num_nodes_), 0);
                std::vector<int> attachment_masks(static_cast<size_t>(num_nodes_), 0);
                std::vector<int> directly_attached;
                std::array<int64_t, 64> four_node_mask_counts{};
                int epoch = 0;
                bool local_has_best = false;
                EdgeKey local_best{0.0, 0.0, 0.0, -1};
                int64_t local_directly_attached_total = 0;
                int64_t local_four_node_pair_total = 0;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 32)
#endif
                for (ssize_t edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
                    const auto [scored, candidate_key] = score_candidate(
                        edge_idx,
                        marks,
                        attachment_masks,
                        directly_attached,
                        four_node_mask_counts,
                        epoch,
                        nullptr,
                        nullptr,
                        nullptr
                    );
                    local_directly_attached_total += scored.directly_attached_size;
                    local_four_node_pair_total += scored.four_node_pair_count;
                    if (!local_has_best || edge_key_less(candidate_key, local_best)) {
                        local_best = candidate_key;
                        local_has_best = true;
                    }
                }
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    directly_attached_total += local_directly_attached_total;
                    four_node_pair_total += local_four_node_pair_total;
                    if (local_has_best && (!has_best || edge_key_less(local_best, best))) {
                        best = local_best;
                        has_best = true;
                    }
                }
            }
        }

        py::dict result;
        result["best_edge_id"] = static_cast<int64_t>(has_best ? best.edge_id : -1);
        result["best_score"] = has_best ? best.score : std::numeric_limits<double>::infinity();
        result["best_degree_score"] = has_best ? best.degree_score : std::numeric_limits<double>::infinity();
        result["best_support_score"] = has_best ? best.support_score : std::numeric_limits<double>::infinity();
        result["rescored_count"] = static_cast<int64_t>(num_edges);
        result["avg_directly_attached_size"] = num_edges == 0 ? 0.0 : static_cast<double>(directly_attached_total) / static_cast<double>(num_edges);
        result["avg_four_node_pair_count"] = num_edges == 0 ? 0.0 : static_cast<double>(four_node_pair_total) / static_cast<double>(num_edges);
        result["native_pair_generation_runtime_sec"] = pair_generation_sec;
        result["native_delta_accumulation_runtime_sec"] = delta_accumulation_sec;
        result["native_score_scalarization_runtime_sec"] = score_scalarization_sec;
        return result;
    }

    py::dict refresh_scores_from_delta_cache(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_edge_ids_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> current_raw_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> current_std_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> stats_mean_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> stats_std_array,
        const std::string& score_mode,
        const double eps,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_delta_cache_array,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> delta_valid_cache_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> score_cache_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> degree_score_cache_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> support_score_cache_array,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array
    ) const {
        if (!has_edge_ids_) {
            throw std::runtime_error("NativeGraphState refresh_scores_from_delta_cache requires edge id state.");
        }
        const auto candidate_edge_ids = candidate_edge_ids_array.unchecked<1>();
        const auto current_raw = current_raw_array.unchecked<2>();
        const auto current_std = current_std_array.unchecked<2>();
        const auto stats_mean = stats_mean_array.unchecked<1>();
        const auto stats_std = stats_std_array.unchecked<1>();
        const auto candidate_delta_cache = candidate_delta_cache_array.unchecked<3>();
        const auto delta_valid_cache = delta_valid_cache_array.unchecked<1>();
        auto score_cache = score_cache_array.mutable_unchecked<1>();
        auto degree_score_cache = degree_score_cache_array.mutable_unchecked<1>();
        auto support_score_cache = support_score_cache_array.mutable_unchecked<1>();
        auto valid_score_cache = valid_score_cache_array.mutable_unchecked<1>();

        if (
            candidate_delta_cache.shape(1) != 2 ||
            candidate_delta_cache.shape(2) != kOrbitDim ||
            current_raw.shape(1) != kOrbitDim ||
            current_std.shape(1) != kOrbitDim
        ) {
            throw std::runtime_error("Invalid candidate-delta refresh array shape.");
        }

        bool has_best = false;
        EdgeKey best{0.0, 0.0, 0.0, -1};
        const ssize_t num_edges = candidate_edge_ids.shape(0);
        for (ssize_t idx = 0; idx < num_edges; ++idx) {
            const int64_t edge_id = candidate_edge_ids(idx);
            if (edge_id < 0 || edge_id >= static_cast<int64_t>(edge_by_id_.size())) {
                throw std::runtime_error("candidate_edge_ids contains invalid edge id.");
            }
            if (edge_id >= candidate_delta_cache.shape(0) || edge_id >= delta_valid_cache.shape(0)) {
                throw std::runtime_error("candidate delta cache is too small for edge id.");
            }
            if (delta_valid_cache(edge_id) == 0) {
                throw std::runtime_error("refresh_scores_from_delta_cache received an edge with invalid candidate delta.");
            }
            if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                continue;
            }
            const int u = edge_by_id_[static_cast<size_t>(edge_id)][0];
            const int v = edge_by_id_[static_cast<size_t>(edge_id)][1];
            const int64_t* endpoint_delta = &candidate_delta_cache(edge_id, 0, 0);
            const ScoreResult scored = score_from_endpoint_delta_cache(
                u,
                v,
                current_raw,
                current_std,
                stats_mean,
                stats_std,
                score_mode,
                eps,
                endpoint_delta
            );
            const double degree_score =
                static_cast<double>(
                    current_degrees_[static_cast<size_t>(u)] +
                    current_degrees_[static_cast<size_t>(v)]
                );
            const double support_score = static_cast<double>(edge_support(
                row_ptr_.data(),
                col_idx_.data(),
                u,
                v,
                adjacency_edge_ids_.data(),
                active_edge_mask_.data()
            ));
            score_cache(edge_id) = scored.score;
            degree_score_cache(edge_id) = degree_score;
            support_score_cache(edge_id) = support_score;
            valid_score_cache(edge_id) = 1;
            const EdgeKey key{scored.score, degree_score, support_score, edge_id};
            if (!has_best || edge_key_less(key, best)) {
                best = key;
                has_best = true;
            }
        }

        py::dict result;
        result["best_edge_id"] = static_cast<int64_t>(has_best ? best.edge_id : -1);
        result["best_score"] = has_best ? best.score : std::numeric_limits<double>::infinity();
        result["best_degree_score"] = has_best ? best.degree_score : std::numeric_limits<double>::infinity();
        result["best_support_score"] = has_best ? best.support_score : std::numeric_limits<double>::infinity();
        result["refreshed_count"] = static_cast<int64_t>(num_edges);
        return result;
    }

    py::dict compute_selected_edge_delta_state(const int u, const int v) const {
        return compute_selected_edge_delta_impl(
            row_ptr_.data(),
            col_idx_.data(),
            num_nodes_,
            u,
            v,
            adjacency_edge_ids_.data(),
            active_edge_mask_.data()
        );
    }

    py::dict compute_selected_edge_delta_and_invalidate(
        const int64_t selected_edge_id,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array
    ) const {
        if (!has_edge_ids_) {
            throw std::runtime_error("NativeGraphState native invalidation requires edge id state.");
        }
        if (selected_edge_id < 0 || selected_edge_id >= static_cast<int64_t>(edge_by_id_.size())) {
            throw std::runtime_error("selected_edge_id is out of range.");
        }
        const int u = edge_by_id_[static_cast<size_t>(selected_edge_id)][0];
        const int v = edge_by_id_[static_cast<size_t>(selected_edge_id)][1];
        py::dict result = compute_selected_edge_delta_impl(
            row_ptr_.data(),
            col_idx_.data(),
            num_nodes_,
            u,
            v,
            adjacency_edge_ids_.data(),
            active_edge_mask_.data()
        );

        auto valid_score_cache = valid_score_cache_array.mutable_unchecked<1>();
        auto affected_nodes_array = result["affected_nodes"].cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
        auto raw_delta_array = result["raw_delta"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto impacted_edges_array = result["impacted_edges"].cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
        const auto affected_nodes = affected_nodes_array.unchecked<1>();
        const auto raw_delta = raw_delta_array.unchecked<2>();
        const auto impacted_edges = impacted_edges_array.unchecked<2>();

        int64_t invalidated_count = 0;
        std::unordered_set<int64_t> touched_edge_ids;
        touched_edge_ids.reserve(static_cast<size_t>(affected_nodes.shape(0) * 4 + impacted_edges.shape(0) + 1));

        auto invalidate = [&](const int64_t edge_id) {
            if (edge_id < 0 || edge_id >= valid_score_cache.shape(0)) {
                return;
            }
            if (touched_edge_ids.find(edge_id) != touched_edge_ids.end()) {
                return;
            }
            touched_edge_ids.insert(edge_id);
            if (edge_id >= static_cast<int64_t>(active_edge_mask_.size()) || active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                return;
            }
            if (valid_score_cache(edge_id) != 0) {
                valid_score_cache(edge_id) = 0;
                ++invalidated_count;
            }
        };

        touched_edge_ids.insert(selected_edge_id);
        if (selected_edge_id >= 0 && selected_edge_id < valid_score_cache.shape(0)) {
            valid_score_cache(selected_edge_id) = 0;
        }
        for (ssize_t node_idx = 0; node_idx < affected_nodes.shape(0); ++node_idx) {
            bool changed = false;
            for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
                if (raw_delta(node_idx, orbit_idx) != 0.0) {
                    changed = true;
                    break;
                }
            }
            if (!changed) {
                continue;
            }
            const int node = static_cast<int>(affected_nodes(node_idx));
            if (node < 0 || node >= num_nodes_) {
                continue;
            }
            for (const int64_t edge_id : incident_edge_ids_[static_cast<size_t>(node)]) {
                invalidate(edge_id);
            }
        }
        for (ssize_t idx = 0; idx < impacted_edges.shape(0); ++idx) {
            const int left = static_cast<int>(impacted_edges(idx, 0));
            const int right = static_cast<int>(impacted_edges(idx, 1));
            const auto found = edge_code_to_id_.find(encode_pair(left, right));
            if (found != edge_code_to_id_.end()) {
                invalidate(found->second);
            }
        }

        result["invalidated_count"] = invalidated_count;
        result["cache_invalidated"] = true;
        return result;
    }

    py::dict compute_selected_edge_delta_and_update_candidate_cache(
        const int64_t selected_edge_id,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> delta_valid_cache_array,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_delta_cache_array
    ) const {
        if (!has_edge_ids_) {
            throw std::runtime_error("NativeGraphState candidate-delta update requires edge id state.");
        }
        if (selected_edge_id < 0 || selected_edge_id >= static_cast<int64_t>(edge_by_id_.size())) {
            throw std::runtime_error("selected_edge_id is out of range.");
        }
        if (
            candidate_delta_cache_array.ndim() != 3 ||
            candidate_delta_cache_array.shape(1) != 2 ||
            candidate_delta_cache_array.shape(2) != kOrbitDim
        ) {
            throw std::runtime_error("candidate_delta_cache must have shape [num_edges, 2, 15].");
        }
        auto valid_score_cache = valid_score_cache_array.mutable_unchecked<1>();
        auto delta_valid_cache = delta_valid_cache_array.mutable_unchecked<1>();
        int64_t* candidate_delta_cache = static_cast<int64_t*>(candidate_delta_cache_array.mutable_data());

        using Clock = std::chrono::high_resolution_clock;
        const auto update_start = Clock::now();
        const int u = edge_by_id_[static_cast<size_t>(selected_edge_id)][0];
        const int v = edge_by_id_[static_cast<size_t>(selected_edge_id)][1];

        std::vector<int> marks(static_cast<size_t>(num_nodes_), 0);
        std::vector<int> attachment_masks(static_cast<size_t>(num_nodes_), 0);
        std::vector<int> directly_attached;
        std::vector<PairCode> encoded_pairs;
        std::vector<PairMask> pair_masks;
        int epoch = 0;
        collect_attached_nodes(
            row_ptr_.data(),
            col_idx_.data(),
            u,
            v,
            marks,
            attachment_masks,
            directly_attached,
            epoch,
            adjacency_edge_ids_.data(),
            active_edge_mask_.data()
        );
        collect_relevant_pair_masks(
            row_ptr_.data(),
            col_idx_.data(),
            u,
            v,
            directly_attached,
            marks,
            attachment_masks,
            epoch,
            encoded_pairs,
            pair_masks,
            adjacency_edge_ids_.data(),
            active_edge_mask_.data()
        );
        auto affected_nodes = collect_two_hop_nodes_sorted(
            row_ptr_.data(),
            col_idx_.data(),
            u,
            v,
            marks,
            epoch,
            adjacency_edge_ids_.data(),
            active_edge_mask_.data()
        );

        std::unordered_map<int, int> affected_index;
        affected_index.reserve(affected_nodes.size() * 2U + 1U);
        for (size_t idx = 0; idx < affected_nodes.size(); ++idx) {
            affected_index.emplace(affected_nodes[idx], static_cast<int>(idx));
        }

        py::array_t<int64_t> affected_nodes_array(static_cast<ssize_t>(affected_nodes.size()));
        auto affected_nodes_view = affected_nodes_array.mutable_unchecked<1>();
        for (size_t idx = 0; idx < affected_nodes.size(); ++idx) {
            affected_nodes_view(static_cast<ssize_t>(idx)) = static_cast<int64_t>(affected_nodes[idx]);
        }

        py::array_t<double> raw_delta_array({static_cast<ssize_t>(affected_nodes.size()), static_cast<ssize_t>(kOrbitDim)});
        auto raw_delta_mut = raw_delta_array.mutable_unchecked<2>();
        for (ssize_t row = 0; row < raw_delta_mut.shape(0); ++row) {
            for (ssize_t col = 0; col < raw_delta_mut.shape(1); ++col) {
                raw_delta_mut(row, col) = 0.0;
            }
        }
        double* raw_delta_ptr = static_cast<double*>(raw_delta_array.mutable_data());

        int64_t invalidated_count = 0;
        std::unordered_set<int64_t> invalidation_touched;
        std::unordered_set<int64_t> corrected_edge_ids;
        std::unordered_set<int64_t> full_rescore_edge_ids;
        std::vector<uint64_t> impacted_edge_codes;
        invalidation_touched.reserve(static_cast<size_t>(affected_nodes.size() * 4U + pair_masks.size() * 5U + 1U));
        corrected_edge_ids.reserve(pair_masks.size() * 2U + directly_attached.size() + 1U);
        full_rescore_edge_ids.reserve(pair_masks.size() * 2U + directly_attached.size() + 1U);
        impacted_edge_codes.reserve(pair_masks.size() * 5U + directly_attached.size() * 2U);

        auto invalidate_score = [&](const int64_t edge_id) {
            if (edge_id < 0 || edge_id >= valid_score_cache.shape(0)) {
                return;
            }
            if (edge_id == selected_edge_id) {
                return;
            }
            if (invalidation_touched.find(edge_id) != invalidation_touched.end()) {
                return;
            }
            invalidation_touched.insert(edge_id);
            if (edge_id >= static_cast<int64_t>(active_edge_mask_.size()) || active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                return;
            }
            if (valid_score_cache(edge_id) != 0) {
                valid_score_cache(edge_id) = 0;
                ++invalidated_count;
            }
        };

        auto endpoint_slot = [&](const int64_t edge_id, const int node) {
            const auto& edge = edge_by_id_[static_cast<size_t>(edge_id)];
            if (edge[0] == node) {
                return 0;
            }
            if (edge[1] == node) {
                return 1;
            }
            return -1;
        };

        auto apply_omega_to_slot = [&](const int size, const int before_mask, const int removed_bit, const int local_idx, int64_t* slot_delta) {
            const int without_selected = before_mask & (~kUvEdgeBit);
            const int without_candidate = before_mask & (~removed_bit);
            const int without_both = before_mask & (~kUvEdgeBit) & (~removed_bit);
            add_orbit_vector_contribution(size, without_both, local_idx, 1, slot_delta);
            add_orbit_vector_contribution(size, without_selected, local_idx, -1, slot_delta);
            add_orbit_vector_contribution(size, without_candidate, local_idx, -1, slot_delta);
            add_orbit_vector_contribution(size, before_mask, local_idx, 1, slot_delta);
        };

        auto process_candidate_edge_in_graphlet = [&](const int size, const int before_mask, const std::array<int, 4>& nodes, const int edge_bit, const int left_idx, const int right_idx) {
            if ((before_mask & edge_bit) == 0 || edge_bit == kUvEdgeBit) {
                return;
            }
            const int left = nodes[static_cast<size_t>(left_idx)];
            const int right = nodes[static_cast<size_t>(right_idx)];
            const uint64_t code = encode_pair(left, right);
            impacted_edge_codes.push_back(code);
            const auto found = edge_code_to_id_.find(code);
            if (found == edge_code_to_id_.end()) {
                return;
            }
            const int64_t edge_id = found->second;
            if (edge_id == selected_edge_id || edge_id < 0 || edge_id >= static_cast<int64_t>(active_edge_mask_.size())) {
                return;
            }
            if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                return;
            }
            invalidate_score(edge_id);
            if (edge_id >= delta_valid_cache.shape(0) || edge_id >= candidate_delta_cache_array.shape(0)) {
                full_rescore_edge_ids.insert(edge_id);
                return;
            }
            if (delta_valid_cache(edge_id) == 0) {
                full_rescore_edge_ids.insert(edge_id);
                return;
            }
            const int left_slot = endpoint_slot(edge_id, left);
            const int right_slot = endpoint_slot(edge_id, right);
            if (left_slot < 0 || right_slot < 0) {
                full_rescore_edge_ids.insert(edge_id);
                delta_valid_cache(edge_id) = 0;
                return;
            }
            int64_t* edge_delta_base = candidate_delta_cache + edge_id * 2 * kOrbitDim;
            apply_omega_to_slot(size, before_mask, edge_bit, left_idx, edge_delta_base + left_slot * kOrbitDim);
            apply_omega_to_slot(size, before_mask, edge_bit, right_idx, edge_delta_base + right_slot * kOrbitDim);
            corrected_edge_ids.insert(edge_id);
        };

        auto process_graphlet_edges = [&](const int size, const int before_mask, const std::array<int, 4>& nodes) {
            if (size == 3) {
                process_candidate_edge_in_graphlet(size, before_mask, nodes, kUAEdgeBit, 0, 2);
                process_candidate_edge_in_graphlet(size, before_mask, nodes, 4, 1, 2);
            } else if (size == 4) {
                process_candidate_edge_in_graphlet(size, before_mask, nodes, kUAEdgeBit, 0, 2);
                process_candidate_edge_in_graphlet(size, before_mask, nodes, kUBEdgeBit, 0, 3);
                process_candidate_edge_in_graphlet(size, before_mask, nodes, kVAEdgeBit, 1, 2);
                process_candidate_edge_in_graphlet(size, before_mask, nodes, kVBEdgeBit, 1, 3);
                process_candidate_edge_in_graphlet(size, before_mask, nodes, kABEdgeBit, 2, 3);
            }
        };

        accumulate_full_delta_for_mask(2, kUvEdgeBit, {u, v, -1, -1}, affected_index, raw_delta_ptr);
        for (const int node : directly_attached) {
            const int mask = mask_for_three_from_attachment(attachment_masks[node]);
            const std::array<int, 4> nodes{u, v, node, -1};
            accumulate_full_delta_for_mask(3, mask, nodes, affected_index, raw_delta_ptr);
            process_graphlet_edges(3, mask, nodes);
        }
        for (const PairMask& pair : pair_masks) {
            const std::array<int, 4> nodes{u, v, pair.first, pair.second};
            accumulate_full_delta_for_mask(4, pair.mask, nodes, affected_index, raw_delta_ptr);
            process_graphlet_edges(4, pair.mask, nodes);
        }

        if (selected_edge_id >= 0 && selected_edge_id < valid_score_cache.shape(0)) {
            valid_score_cache(selected_edge_id) = 0;
        }
        if (selected_edge_id >= 0 && selected_edge_id < delta_valid_cache.shape(0)) {
            delta_valid_cache(selected_edge_id) = 0;
        }

        for (ssize_t node_idx = 0; node_idx < raw_delta_mut.shape(0); ++node_idx) {
            bool changed = false;
            for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
                if (raw_delta_mut(node_idx, orbit_idx) != 0.0) {
                    changed = true;
                    break;
                }
            }
            if (!changed) {
                continue;
            }
            const int node = static_cast<int>(affected_nodes_view(node_idx));
            if (node < 0 || node >= num_nodes_) {
                continue;
            }
            for (const int64_t edge_id : incident_edge_ids_[static_cast<size_t>(node)]) {
                invalidate_score(edge_id);
            }
        }

        std::sort(impacted_edge_codes.begin(), impacted_edge_codes.end());
        impacted_edge_codes.erase(std::unique(impacted_edge_codes.begin(), impacted_edge_codes.end()), impacted_edge_codes.end());
        py::array_t<int64_t> impacted_edges_array(
            {static_cast<ssize_t>(impacted_edge_codes.size()), static_cast<ssize_t>(2)}
        );
        auto impacted_edges_view = impacted_edges_array.mutable_unchecked<2>();
        for (size_t idx = 0; idx < impacted_edge_codes.size(); ++idx) {
            const auto [left, right] = decode_pair(impacted_edge_codes[idx]);
            impacted_edges_view(static_cast<ssize_t>(idx), 0) = static_cast<int64_t>(left);
            impacted_edges_view(static_cast<ssize_t>(idx), 1) = static_cast<int64_t>(right);
        }

        py::dict result;
        result["affected_nodes"] = std::move(affected_nodes_array);
        result["raw_delta"] = std::move(raw_delta_array);
        result["impacted_edges"] = std::move(impacted_edges_array);
        result["directly_attached_size"] = static_cast<int64_t>(directly_attached.size());
        result["four_node_pair_count"] = static_cast<int64_t>(pair_masks.size());
        result["invalidated_count"] = invalidated_count;
        result["cache_invalidated"] = true;
        result["mixed_correction_edge_count"] = static_cast<int64_t>(corrected_edge_ids.size());
        result["delta_impacted_full_rescore_count"] = static_cast<int64_t>(full_rescore_edge_ids.size());
        result["native_mixed_correction_runtime_sec"] = std::chrono::duration<double>(Clock::now() - update_start).count();
        return result;
    }

    void remove_edge(const int u, const int v) {
        if (!has_edge_ids_) {
            throw std::runtime_error("NativeGraphState logical removal requires edge id state.");
        }
        const auto found = edge_code_to_id_.find(encode_pair(u, v));
        if (found == edge_code_to_id_.end()) {
            throw std::runtime_error("NativeGraphState remove_edge called for an unknown undirected edge.");
        }
        const int64_t edge_id = found->second;
        if (edge_id < 0 || edge_id >= static_cast<int64_t>(active_edge_mask_.size())) {
            throw std::runtime_error("NativeGraphState edge id is outside active mask range.");
        }
        if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
            throw std::runtime_error("NativeGraphState remove_edge called for a non-active undirected edge.");
        }
        active_edge_mask_[static_cast<size_t>(edge_id)] = 0;
        const auto& edge = edge_by_id_[static_cast<size_t>(edge_id)];
        if (current_degrees_[static_cast<size_t>(edge[0])] <= 0 || current_degrees_[static_cast<size_t>(edge[1])] <= 0) {
            throw std::runtime_error("NativeGraphState degree underflow during logical edge removal.");
        }
        --current_degrees_[static_cast<size_t>(edge[0])];
        --current_degrees_[static_cast<size_t>(edge[1])];
        --active_edge_count_;
    }

    int64_t directed_edge_count() const {
        return active_edge_count_ * 2;
    }

    int64_t immutable_directed_adjacency_entry_count() const {
        return static_cast<int64_t>(col_idx_.size());
    }

    int64_t active_edge_count() const {
        return active_edge_count_;
    }

    py::dict storage_statistics() const {
        const int64_t immutable_directed_entries = static_cast<int64_t>(col_idx_.size());
        const int64_t active_directed_entries = active_edge_count_ * 2;
        const int64_t inactive_directed_entries = immutable_directed_entries - active_directed_entries;
        py::dict result;
        result["original_edge_count"] = original_edge_count_;
        result["active_edge_count"] = active_edge_count_;
        result["inactive_edge_count"] = original_edge_count_ - active_edge_count_;
        result["immutable_directed_adjacency_entries"] = immutable_directed_entries;
        result["active_directed_adjacency_entries"] = active_directed_entries;
        result["inactive_directed_adjacency_entries"] = inactive_directed_entries;
        result["tombstone_ratio"] = immutable_directed_entries == 0
            ? 0.0
            : static_cast<double>(inactive_directed_entries) / static_cast<double>(immutable_directed_entries);
        return result;
    }

    int num_nodes() const {
        return num_nodes_;
    }

private:
    const int64_t* adjacency_edge_ids_ptr() const {
        return has_edge_ids_ ? adjacency_edge_ids_.data() : nullptr;
    }

    const uint8_t* active_edge_mask_ptr() const {
        return has_edge_ids_ ? active_edge_mask_.data() : nullptr;
    }

    py::array_t<int64_t> row_ptr_array() const {
        return py::array_t<int64_t>(static_cast<ssize_t>(row_ptr_.size()), row_ptr_.data());
    }

    py::array_t<int64_t> col_idx_array() const {
        return py::array_t<int64_t>(static_cast<ssize_t>(col_idx_.size()), col_idx_.data());
    }

    int num_nodes_ = 0;
    std::vector<int64_t> row_ptr_;
    std::vector<int64_t> col_idx_;
    std::vector<int64_t> adjacency_edge_ids_;
    std::vector<int64_t> current_degrees_;
    bool has_edge_ids_ = false;
    int64_t original_edge_count_ = 0;
    int64_t active_edge_count_ = 0;
    std::vector<std::array<int, 2>> edge_by_id_;
    std::unordered_map<uint64_t, int64_t> edge_code_to_id_;
    std::vector<std::vector<int64_t>> incident_edge_ids_;
    std::vector<uint8_t> active_edge_mask_;
};

py::dict canonical_tables() {
    auto as_numpy = [](const auto& table, const ssize_t rows) {
        py::array_t<int8_t> array({rows, static_cast<ssize_t>(4)});
        auto view = array.mutable_unchecked<2>();
        for (ssize_t row = 0; row < rows; ++row) {
            for (ssize_t col = 0; col < 4; ++col) {
                view(row, col) = table[static_cast<size_t>(row)][static_cast<size_t>(col)];
            }
        }
        return array;
    };

    py::dict result;
    result["size2"] = as_numpy(kSize2Tables, 2);
    result["size3"] = as_numpy(kSize3Tables, 8);
    result["size4"] = as_numpy(kSize4Tables, 64);
    return result;
}

py::dict openmp_info() {
    py::dict result;
#ifdef _OPENMP
    result["openmp_enabled"] = true;
    result["openmp_max_threads"] = static_cast<int64_t>(omp_get_max_threads());
#else
    result["openmp_enabled"] = false;
    result["openmp_max_threads"] = static_cast<int64_t>(1);
#endif
    return result;
}

py::dict set_openmp_threads(const int64_t thread_count) {
    if (thread_count <= 0) {
        throw std::runtime_error("set_openmp_threads requires a positive thread count.");
    }
    py::dict result;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(thread_count));
    result["openmp_enabled"] = true;
    result["openmp_dynamic"] = static_cast<int64_t>(omp_get_dynamic());
    result["openmp_max_threads"] = static_cast<int64_t>(omp_get_max_threads());
    result["requested_openmp_threads"] = thread_count;
#else
    result["openmp_enabled"] = false;
    result["openmp_dynamic"] = static_cast<int64_t>(0);
    result["openmp_max_threads"] = static_cast<int64_t>(1);
    result["requested_openmp_threads"] = thread_count;
#endif
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<NativeGraphState>(module, "NativeGraphState")
        .def(py::init<
            py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
            py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        >())
        .def(py::init<
            py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
            py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
            py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        >())
        .def("eligible_edge_id_partitions", &NativeGraphState::eligible_edge_id_partitions, py::arg("active_edge_ids"), py::arg("edge_array_by_id"), py::arg("degrees"), py::arg("d_min"), py::arg("guard_bridges"), py::arg("valid_score_cache"), py::arg("use_score_cache"))
        .def("eligible_edge_id_partitions_with_cached_best", &NativeGraphState::eligible_edge_id_partitions_with_cached_best, py::arg("active_edge_ids"), py::arg("degrees"), py::arg("d_min"), py::arg("guard_bridges"), py::arg("valid_score_cache"), py::arg("use_score_cache"), py::arg("score_cache"), py::arg("degree_score_cache"), py::arg("support_score_cache"), py::arg("delta_valid_cache") = py::none())
        .def("score_edges_round_best", &NativeGraphState::score_edges_round_best_state, py::arg("candidate_edges"), py::arg("candidate_edge_ids"), py::arg("current_raw"), py::arg("current_std"), py::arg("stats_mean"), py::arg("stats_std"), py::arg("score_mode"), py::arg("eps"), py::arg("score_cache"), py::arg("degree_score_cache"), py::arg("support_score_cache"), py::arg("valid_score_cache"), py::arg("kernel_variant") = "mask_count_v4_combinatorial", py::arg("profile_native_kernel") = false, py::arg("candidate_delta_cache") = py::none(), py::arg("delta_valid_cache") = py::none())
        .def("score_edge_ids_round_best", &NativeGraphState::score_edge_ids_round_best, py::arg("candidate_edge_ids"), py::arg("current_raw"), py::arg("current_std"), py::arg("stats_mean"), py::arg("stats_std"), py::arg("score_mode"), py::arg("eps"), py::arg("score_cache"), py::arg("degree_score_cache"), py::arg("support_score_cache"), py::arg("valid_score_cache"), py::arg("kernel_variant") = "mask_count_v4_combinatorial", py::arg("profile_native_kernel") = false, py::arg("candidate_delta_cache") = py::none(), py::arg("delta_valid_cache") = py::none())
        .def("refresh_scores_from_delta_cache", &NativeGraphState::refresh_scores_from_delta_cache, py::arg("candidate_edge_ids"), py::arg("current_raw"), py::arg("current_std"), py::arg("stats_mean"), py::arg("stats_std"), py::arg("score_mode"), py::arg("eps"), py::arg("candidate_delta_cache"), py::arg("delta_valid_cache"), py::arg("score_cache"), py::arg("degree_score_cache"), py::arg("support_score_cache"), py::arg("valid_score_cache"))
        .def("compute_selected_edge_delta", &NativeGraphState::compute_selected_edge_delta_state, py::arg("u"), py::arg("v"))
        .def("compute_selected_edge_delta_and_invalidate", &NativeGraphState::compute_selected_edge_delta_and_invalidate, py::arg("selected_edge_id"), py::arg("valid_score_cache"))
        .def("compute_selected_edge_delta_and_update_candidate_cache", &NativeGraphState::compute_selected_edge_delta_and_update_candidate_cache, py::arg("selected_edge_id"), py::arg("valid_score_cache"), py::arg("delta_valid_cache"), py::arg("candidate_delta_cache"))
        .def("remove_edge", &NativeGraphState::remove_edge, py::arg("u"), py::arg("v"))
        .def("directed_edge_count", &NativeGraphState::directed_edge_count)
        .def("immutable_directed_adjacency_entry_count", &NativeGraphState::immutable_directed_adjacency_entry_count)
        .def("active_edge_count", &NativeGraphState::active_edge_count)
        .def("storage_statistics", &NativeGraphState::storage_statistics)
        .def("num_nodes", &NativeGraphState::num_nodes);

    module.def("score_edges_round", &score_edges_round, py::arg("row_ptr"), py::arg("col_idx"), py::arg("candidate_edges"), py::arg("current_raw"), py::arg("current_std"), py::arg("stats_mean"), py::arg("stats_std"), py::arg("score_mode"), py::arg("eps"), py::arg("include_update_sizes") = true);
    module.def("score_edges_round_best", &score_edges_round_best, py::arg("row_ptr"), py::arg("col_idx"), py::arg("candidate_edges"), py::arg("candidate_edge_ids"), py::arg("current_raw"), py::arg("current_std"), py::arg("stats_mean"), py::arg("stats_std"), py::arg("score_mode"), py::arg("eps"), py::arg("score_cache"), py::arg("degree_score_cache"), py::arg("support_score_cache"), py::arg("valid_score_cache"), py::arg("kernel_variant") = "mask_count_v4_combinatorial", py::arg("profile_native_kernel") = false, py::arg("candidate_delta_cache") = py::none(), py::arg("delta_valid_cache") = py::none());
    module.def("compute_selected_edge_delta", &compute_selected_edge_delta, py::arg("row_ptr"), py::arg("col_idx"), py::arg("u"), py::arg("v"));
    module.def("eligible_edge_ids_from_csr", &eligible_edge_ids_from_csr, py::arg("row_ptr"), py::arg("col_idx"), py::arg("active_edge_ids"), py::arg("edge_array_by_id"), py::arg("degrees"), py::arg("d_min"), py::arg("guard_bridges"));
    module.def("eligible_edge_id_partitions_from_csr", &eligible_edge_id_partitions_from_csr, py::arg("row_ptr"), py::arg("col_idx"), py::arg("active_edge_ids"), py::arg("edge_array_by_id"), py::arg("degrees"), py::arg("d_min"), py::arg("guard_bridges"), py::arg("valid_score_cache"), py::arg("use_score_cache"));
    module.def("canonical_tables", &canonical_tables);
    module.def("openmp_info", &openmp_info);
    module.def("set_openmp_threads", &set_openmp_threads, py::arg("thread_count"));
}
