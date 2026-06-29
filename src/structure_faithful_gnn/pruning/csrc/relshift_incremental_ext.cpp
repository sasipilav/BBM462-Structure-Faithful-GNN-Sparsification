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
#include <queue>
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
constexpr const char* kOrbitRegistryVersion = "orca-node-orbits-2to4-v1";
constexpr int kUvEdgeBit = 1;
constexpr int kUAEdgeBit = 2;
constexpr int kUBEdgeBit = 4;
constexpr int kVAEdgeBit = 8;
constexpr int kVBEdgeBit = 16;
constexpr int kABEdgeBit = 32;

inline double canonical_standardized_coordinate(
    const double raw_value,
    const double mean,
    const double standard_deviation
) {
    const double clamped_raw = std::max(raw_value, 0.0);
    return (std::log1p(clamped_raw) - mean) / standard_deviation;
}

using OrbitRow = std::array<int8_t, 4>;

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

inline bool active_edge_between(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int left,
    const int right,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    const int64_t begin = row_ptr[left];
    const int64_t end = row_ptr[left + 1];
    const int64_t* found = std::lower_bound(
        col_idx + begin,
        col_idx + end,
        static_cast<int64_t>(right)
    );
    if (found == col_idx + end || *found != right) {
        return false;
    }
    const int64_t adjacency_index = static_cast<int64_t>(found - col_idx);
    return adjacency_entry_is_active(
        adjacency_index,
        adjacency_edge_ids,
        active_edge_mask
    );
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

struct VersionedHeapEntry {
    EdgeKey key;
    uint64_t version;
};

struct VersionedHeapGreater {
    bool operator()(const VersionedHeapEntry& left, const VersionedHeapEntry& right) const {
        if (edge_key_less(right.key, left.key)) {
            return true;
        }
        if (edge_key_less(left.key, right.key)) {
            return false;
        }
        return left.version > right.version;
    }
};

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

void collect_two_hop_nodes_sorted_into(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int u,
    const int v,
    std::vector<int>& marks,
    int& epoch,
    std::vector<int>& touched,
    std::vector<int>& frontier,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    ++epoch;
    touched.clear();
    frontier.clear();
    touched.reserve(std::max<size_t>(touched.capacity(), 64));
    frontier.reserve(std::max<size_t>(frontier.capacity(), 32));
    auto mark_node = [&](const int node) {
        if (marks[node] == epoch) {
            return;
        }
        marks[node] = epoch;
        touched.push_back(node);
    };
    mark_node(u);
    mark_node(v);
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

void collect_relevant_pair_masks_direct(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const int u,
    const int v,
    std::vector<int>& directly_attached,
    const std::vector<int>& marks,
    const std::vector<int>& attachment_masks,
    const int epoch,
    std::vector<PairMask>& pair_masks,
    const int64_t* adjacency_edge_ids = nullptr,
    const uint8_t* active_edge_mask = nullptr
) {
    pair_masks.clear();
    std::sort(directly_attached.begin(), directly_attached.end());
    const size_t direct_count = directly_attached.size();
    const size_t direct_pair_count = direct_count < 2
        ? 0
        : direct_count * (direct_count - 1) / 2;
    pair_masks.reserve(direct_pair_count + direct_count * 4);

    // Every connected four-node induced graphlet containing (u,v) either has
    // both extra nodes in D=N(u) union N(v), or has one node a in D and one
    // outside node b joined to a.  Enumerate the first class exactly once,
    // determining the a-b bit directly from immutable CSR plus the active mask.
    for (size_t left_idx = 0; left_idx < direct_count; ++left_idx) {
        const int first = directly_attached[left_idx];
        for (size_t right_idx = left_idx + 1; right_idx < direct_count; ++right_idx) {
            const int second = directly_attached[right_idx];
            const int pair_edge_mask = active_edge_between(
                row_ptr,
                col_idx,
                first,
                second,
                adjacency_edge_ids,
                active_edge_mask
            ) ? kABEdgeBit : 0;
            const int mask = mask_for_four_from_attachments(
                attachment_masks[first],
                attachment_masks[second],
                pair_edge_mask
            );
            pair_masks.push_back(PairMask{first, second, mask});
        }
    }

    // For the second class, only active D-outside edges are relevant.  Since
    // the outside endpoint is not in D it is never used as an outer-loop node,
    // so each unordered pair is emitted exactly once without sort/unique.
    for (const int attached : directly_attached) {
        for (
            int64_t idx = row_ptr[attached];
            idx < row_ptr[attached + 1];
            ++idx
        ) {
            if (!adjacency_entry_is_active(
                idx, adjacency_edge_ids, active_edge_mask
            )) {
                continue;
            }
            const int outside = static_cast<int>(col_idx[idx]);
            if (
                outside == u || outside == v || outside == attached ||
                marks[outside] == epoch
            ) {
                continue;
            }
            const int first = std::min(attached, outside);
            const int second = std::max(attached, outside);
            const int first_attachment = marks[first] == epoch
                ? attachment_masks[first]
                : 0;
            const int second_attachment = marks[second] == epoch
                ? attachment_masks[second]
                : 0;
            const int mask = mask_for_four_from_attachments(
                first_attachment,
                second_attachment,
                kABEdgeBit
            );
            pair_masks.push_back(PairMask{first, second, mask});
        }
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

struct FlatMatrixConstView {
    const std::vector<double>* values;
    int columns;

    double operator()(const int row, const int column) const {
        return (*values)[static_cast<size_t>(row * columns + column)];
    }
};

struct FlatVectorConstView {
    const std::vector<double>* values;

    double operator()(const int index) const {
        return (*values)[static_cast<size_t>(index)];
    }
};

ScoreResult score_from_internal_endpoint_delta(
    const int u,
    const int v,
    const std::vector<double>& current_raw,
    const std::vector<double>& current_std,
    const std::vector<double>& stats_mean,
    const std::vector<double>& stats_std,
    const std::vector<double>& node_denominator,
    const std::string& score_mode,
    const int64_t* endpoint_delta
) {
    // Canonical exact scalarization: recompute both base and counterfactual
    // standardized coordinates from raw counts in the same C++ kernel.  Mixing
    // a NumPy-produced base value with std::log1p counterfactual values can
    // create implementation-only 1e-15 differences and break edge-id ties.
    double absolute_sum = 0.0;
    double relative_sum = 0.0;
    double denom_sum = 0.0;
    double denom_min = 0.0;
    for (int local_idx = 0; local_idx < 2; ++local_idx) {
        const int node = local_idx == 0 ? u : v;
        const int64_t row_offset = static_cast<int64_t>(node) * kOrbitDim;
        double endpoint_l1 = 0.0;
        for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
            const size_t coordinate = static_cast<size_t>(row_offset + orbit_idx);
            const double base_raw = std::max(current_raw[coordinate], 0.0);
            const double base_std = canonical_standardized_coordinate(
                base_raw,
                stats_mean[static_cast<size_t>(orbit_idx)],
                stats_std[static_cast<size_t>(orbit_idx)]
            );
            const double candidate_raw = std::max(
                base_raw + static_cast<double>(endpoint_delta[local_idx * kOrbitDim + orbit_idx]),
                0.0
            );
            const double candidate_std = canonical_standardized_coordinate(
                candidate_raw,
                stats_mean[static_cast<size_t>(orbit_idx)],
                stats_std[static_cast<size_t>(orbit_idx)]
            );
            endpoint_l1 += std::abs(base_std - candidate_std);
        }
        const double endpoint_denom = node_denominator[static_cast<size_t>(node)];
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

double endpoint_score_from_internal_delta_masked(
    const int node,
    const int local_idx,
    const std::vector<double>& current_raw,
    const std::vector<double>& current_std,
    const std::vector<double>& stats_mean,
    const std::vector<double>& stats_std,
    const std::vector<double>& node_denominator,
    const std::string& score_mode,
    const int64_t* endpoint_delta,
    const uint16_t nonzero_mask
) {
    const int64_t row_offset = static_cast<int64_t>(node) * kOrbitDim;
    double endpoint_l1 = 0.0;
    uint16_t mask = nonzero_mask;
    while (mask != 0) {
        const int orbit_idx = __builtin_ctz(static_cast<unsigned int>(mask));
        mask = static_cast<uint16_t>(mask & static_cast<uint16_t>(mask - 1));
        const size_t coordinate = static_cast<size_t>(row_offset + orbit_idx);
        const double base_raw = std::max(current_raw[coordinate], 0.0);
        const double base_std = canonical_standardized_coordinate(
            base_raw,
            stats_mean[static_cast<size_t>(orbit_idx)],
            stats_std[static_cast<size_t>(orbit_idx)]
        );
        const double candidate_raw = std::max(
            base_raw + static_cast<double>(endpoint_delta[local_idx * kOrbitDim + orbit_idx]),
            0.0
        );
        const double candidate_std = canonical_standardized_coordinate(
            candidate_raw,
            stats_mean[static_cast<size_t>(orbit_idx)],
            stats_std[static_cast<size_t>(orbit_idx)]
        );
        endpoint_l1 += std::abs(base_std - candidate_std);
    }
    return score_mode == "absolute"
        ? endpoint_l1
        : endpoint_l1 / node_denominator[static_cast<size_t>(node)];
}

ScoreResult score_from_internal_endpoint_delta_masked(
    const int u,
    const int v,
    const std::vector<double>& current_raw,
    const std::vector<double>& current_std,
    const std::vector<double>& stats_mean,
    const std::vector<double>& stats_std,
    const std::vector<double>& node_denominator,
    const std::string& score_mode,
    const int64_t* endpoint_delta,
    const uint16_t* endpoint_nonzero_masks
) {
    double absolute_sum = 0.0;
    double relative_sum = 0.0;
    double denom_sum = 0.0;
    double denom_min = 0.0;
    for (int local_idx = 0; local_idx < 2; ++local_idx) {
        const int node = local_idx == 0 ? u : v;
        const int64_t row_offset = static_cast<int64_t>(node) * kOrbitDim;
        double endpoint_l1 = 0.0;
        uint16_t mask = endpoint_nonzero_masks[local_idx];
        while (mask != 0) {
            const int orbit_idx = __builtin_ctz(static_cast<unsigned int>(mask));
            mask = static_cast<uint16_t>(mask & static_cast<uint16_t>(mask - 1));
            const size_t coordinate = static_cast<size_t>(row_offset + orbit_idx);
            const double base_raw = std::max(current_raw[coordinate], 0.0);
            const double base_std = canonical_standardized_coordinate(
                base_raw,
                stats_mean[static_cast<size_t>(orbit_idx)],
                stats_std[static_cast<size_t>(orbit_idx)]
            );
            const double candidate_raw = std::max(
                base_raw + static_cast<double>(endpoint_delta[local_idx * kOrbitDim + orbit_idx]),
                0.0
            );
            const double candidate_std = canonical_standardized_coordinate(
                candidate_raw,
                stats_mean[static_cast<size_t>(orbit_idx)],
                stats_std[static_cast<size_t>(orbit_idx)]
            );
            endpoint_l1 += std::abs(base_std - candidate_std);
        }
        const double endpoint_denom = node_denominator[static_cast<size_t>(node)];
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

uint16_t endpoint_delta_nonzero_mask(const int64_t* delta) {
    uint16_t mask = 0;
    for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
        if (delta[orbit_idx] != 0) {
            mask = static_cast<uint16_t>(mask | static_cast<uint16_t>(1U << orbit_idx));
        }
    }
    return mask;
}

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
            const double base_raw = std::max(current_raw(node, orbit_idx), 0.0);
            const double base_std = canonical_standardized_coordinate(
                base_raw, stats_mean(orbit_idx), stats_std(orbit_idx)
            );
            const double candidate_raw =
                std::max(base_raw + static_cast<double>(endpoint_delta[local_idx * kOrbitDim + orbit_idx]), 0.0);
            const double candidate_std = canonical_standardized_coordinate(
                candidate_raw, stats_mean(orbit_idx), stats_std(orbit_idx)
            );
            endpoint_l1 += std::abs(base_std - candidate_std);
            base_l1 += std::abs(base_std);
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

void accumulate_full_delta_for_mask_dense(
    const int size,
    const int before_mask,
    const std::array<int, 4>& nodes,
    const std::vector<uint32_t>& node_epochs,
    const uint32_t node_epoch,
    const std::vector<int>& node_local_indices,
    double* raw_delta
) {
    const OrbitRow& before = orbit_row_for_mask(size, before_mask);
    if (!is_valid_orbit_row(before)) {
        return;
    }
    const OrbitRow& after = orbit_row_for_mask(size, before_mask & (~1));
    for (int local_idx = 0; local_idx < size; ++local_idx) {
        const int global_node = nodes[local_idx];
        if (
            global_node < 0 ||
            global_node >= static_cast<int>(node_epochs.size()) ||
            node_epochs[static_cast<size_t>(global_node)] != node_epoch
        ) {
            continue;
        }
        const int row_offset =
            node_local_indices[static_cast<size_t>(global_node)] * kOrbitDim;
        raw_delta[row_offset + before[local_idx]] -= 1.0;
        if (is_valid_orbit_row(after)) {
            raw_delta[row_offset + after[local_idx]] += 1.0;
        }
    }
}

// All exact candidate scorers use this same raw-to-standardized operation
// order.  The StdView argument remains for API compatibility and diagnostics.
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
            const double base_raw = std::max(current_raw(node, orbit_idx), 0.0);
            const double base_std = canonical_standardized_coordinate(
                base_raw, stats_mean(orbit_idx), stats_std(orbit_idx)
            );
            const double candidate_raw =
                std::max(base_raw + endpoint_delta[local_idx * kOrbitDim + orbit_idx], 0.0);
            const double candidate_std = canonical_standardized_coordinate(
                candidate_raw, stats_mean(orbit_idx), stats_std(orbit_idx)
            );
            endpoint_l1 += std::abs(base_std - candidate_std);
            base_l1 += std::abs(base_std);
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
    collect_relevant_pair_masks_direct(
        row_ptr,
        col_idx,
        u,
        v,
        directly_attached,
        marks,
        attachment_masks,
        epoch,
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
        edge_output_by_id_.clear();
        edge_code_to_id_.clear();
        incident_edge_ids_.assign(static_cast<size_t>(num_nodes_), {});
        active_edge_mask_.assign(static_cast<size_t>(edge_array_by_id.shape(0)), static_cast<uint8_t>(1));
        node_workspace_epoch_.assign(static_cast<size_t>(num_nodes_), 0);
        node_workspace_index_.assign(static_cast<size_t>(num_nodes_), -1);
        graphlet_marks_workspace_.assign(static_cast<size_t>(num_nodes_), 0);
        graphlet_attachment_workspace_.assign(static_cast<size_t>(num_nodes_), 0);
        graphlet_epoch_counter_ = 0;
        bridge_query_visited_epoch_.assign(static_cast<size_t>(num_nodes_), 0);
        bridge_query_frontier_.clear();
        bridge_query_frontier_.reserve(64);
        bridge_query_epoch_counter_ = 0;
        edge_workspace_epoch_.assign(static_cast<size_t>(edge_array_by_id.shape(0)), 0);
        edge_workspace_flags_.assign(static_cast<size_t>(edge_array_by_id.shape(0)), 0);
        node_workspace_epoch_counter_ = 0;
        edge_workspace_epoch_counter_ = 0;
        edge_by_id_.reserve(static_cast<size_t>(edge_array_by_id.shape(0)));
        edge_output_by_id_.reserve(static_cast<size_t>(edge_array_by_id.shape(0)));
        edge_code_to_id_.reserve(static_cast<size_t>(edge_array_by_id.shape(0) * 2 + 1));
        for (ssize_t edge_id = 0; edge_id < edge_array_by_id.shape(0); ++edge_id) {
            const int u = static_cast<int>(edge_array_by_id(edge_id, 0));
            const int v = static_cast<int>(edge_array_by_id(edge_id, 1));
            if (u < 0 || v < 0 || u >= num_nodes_ || v >= num_nodes_) {
                throw std::runtime_error("edge_array_by_id contains node id outside graph range.");
            }
            const int left = std::min(u, v);
            const int right = std::max(u, v);
            edge_output_by_id_.push_back(std::array<int, 2>{u, v});
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
        removed_edge_ids_.clear();
        removed_edge_ids_.reserve(static_cast<size_t>(original_edge_count_));
        heap_versions_.assign(static_cast<size_t>(original_edge_count_), 0);
        heap_dirty_mask_.assign(static_cast<size_t>(original_edge_count_), static_cast<uint8_t>(0));
        guard_reason_.assign(static_cast<size_t>(original_edge_count_), static_cast<uint8_t>(0));
        heap_dirty_edge_ids_.clear();
        selection_heap_ = decltype(selection_heap_)();
        indexed_heap_edges_.clear();
        indexed_heap_positions_.clear();
        indexed_heap_keys_.clear();
        heap_storage_mode_ = "versioned";
        heap_storage_mode_initialized_ = false;
        heap_initialized_ = false;
        heap_guard_configuration_initialized_ = false;
        heap_d_min_ = -1;
        heap_guard_bridges_ = false;
        heap_bridge_maintenance_mode_ = "global_tarjan";
        eligible_active_count_ = 0;
        bridge_blocked_count_ = 0;
        d_min_blocked_count_ = 0;
        heap_push_count_total_ = 0;
        heap_pop_count_total_ = 0;
        heap_stale_pop_count_total_ = 0;
        heap_inactive_pop_count_total_ = 0;
        heap_guard_pop_count_total_ = 0;
        heap_dirty_pop_count_total_ = 0;
        heap_rebuild_count_total_ = 0;
        heap_rebuild_edge_entries_scanned_total_ = 0;
        heap_max_size_observed_ = 0;
        has_edge_ids_ = true;
    }

    void initialize_relshift_state(
        py::array_t<double, py::array::c_style | py::array::forcecast> current_raw_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> stats_mean_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> stats_std_array,
        const std::string& score_mode,
        const double eps,
        const bool enable_candidate_delta_cache = true
    ) {
        if (!has_edge_ids_) {
            throw std::runtime_error("Native RelShift state requires stable edge ids.");
        }
        const auto current_raw = current_raw_array.unchecked<2>();
        const auto stats_mean = stats_mean_array.unchecked<1>();
        const auto stats_std = stats_std_array.unchecked<1>();
        if (current_raw.shape(0) != num_nodes_ || current_raw.shape(1) != kOrbitDim) {
            throw std::runtime_error("current_raw must have shape [num_nodes, 15].");
        }
        if (stats_mean.shape(0) != kOrbitDim || stats_std.shape(0) != kOrbitDim) {
            throw std::runtime_error("standardization statistics must have 15 entries.");
        }
        if (score_mode != "absolute" && score_mode != "relative") {
            throw std::runtime_error("score_mode must be absolute or relative.");
        }
        if (!(eps > 0.0)) {
            throw std::runtime_error("eps must be positive.");
        }

        native_score_mode_ = score_mode;
        native_eps_ = eps;
        native_delta_cache_enabled_ = enable_candidate_delta_cache;
        stats_mean_state_.resize(kOrbitDim);
        stats_std_state_.resize(kOrbitDim);
        for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
            if (!(stats_std(orbit_idx) > 0.0)) {
                throw std::runtime_error("standardization std entries must be positive.");
            }
            stats_mean_state_[static_cast<size_t>(orbit_idx)] = stats_mean(orbit_idx);
            stats_std_state_[static_cast<size_t>(orbit_idx)] = stats_std(orbit_idx);
        }

        current_raw_state_.assign(static_cast<size_t>(num_nodes_ * kOrbitDim), 0.0);
        current_std_state_.assign(static_cast<size_t>(num_nodes_ * kOrbitDim), 0.0);
        node_denominator_state_.assign(static_cast<size_t>(num_nodes_), native_eps_);
        for (int node = 0; node < num_nodes_; ++node) {
            double denominator = native_eps_;
            for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
                const size_t offset = static_cast<size_t>(node * kOrbitDim + orbit_idx);
                const double raw_value = std::max(current_raw(node, orbit_idx), 0.0);
                const double standardized = canonical_standardized_coordinate(
                    raw_value,
                    stats_mean_state_[static_cast<size_t>(orbit_idx)],
                    stats_std_state_[static_cast<size_t>(orbit_idx)]
                );
                current_raw_state_[offset] = raw_value;
                current_std_state_[offset] = standardized;
                denominator += std::abs(standardized);
            }
            node_denominator_state_[static_cast<size_t>(node)] = denominator;
        }

        const size_t edge_count = static_cast<size_t>(original_edge_count_);
        score_cache_state_.assign(edge_count, std::numeric_limits<double>::infinity());
        endpoint_score_cache_state_.assign(edge_count * 2, 0.0);
        endpoint_score_valid_mask_state_.assign(edge_count, static_cast<uint8_t>(0));
        degree_score_cache_state_.assign(edge_count, std::numeric_limits<double>::infinity());
        support_score_cache_state_.assign(edge_count, 0.0);
        support_initialized_state_.assign(edge_count, static_cast<uint8_t>(0));
        valid_score_cache_state_.assign(edge_count, static_cast<uint8_t>(0));
        delta_valid_cache_state_.assign(edge_count, static_cast<uint8_t>(0));
        candidate_delta_nonzero_mask_state_.assign(edge_count * 2, static_cast<uint16_t>(0));
        candidate_delta_cache_state_.assign(
            enable_candidate_delta_cache ? edge_count * 2 * kOrbitDim : 0,
            static_cast<int64_t>(0)
        );
        native_state_round_count_ = 0;
        native_state_support_initializations_ = 0;
        native_state_support_decrements_ = 0;
        native_state_node_rows_updated_ = 0;
        endpoint_score_recomputed_total_ = 0;
        endpoint_score_reused_total_ = 0;
        selected_four_node_pairs_total_ = 0;
        selected_four_node_pairs_peak_ = 0;
        selected_affected_nodes_peak_ = 0;
        adjacency_compaction_count_ = 0;
        adjacency_compaction_entries_copied_total_ = 0;
        adjacency_compaction_runtime_sec_total_ = 0.0;
        relshift_state_initialized_ = true;
    }

    bool relshift_state_initialized() const {
        return relshift_state_initialized_;
    }

    py::dict prepare_versioned_heap_round_fused(
        const int d_min,
        const bool guard_bridges,
        const std::string& bridge_maintenance_mode = "global_tarjan"
    ) {
        require_relshift_state_internal();
        py::object native_state_owner = py::cast(this, py::return_value_policy::reference);
        py::array_t<uint8_t> valid_score_cache_array(
            {static_cast<ssize_t>(valid_score_cache_state_.size())},
            {static_cast<ssize_t>(sizeof(uint8_t))},
            valid_score_cache_state_.data(),
            native_state_owner
        );
        py::object delta_valid_object = py::none();
        py::array_t<uint8_t> delta_valid_array;
        if (native_delta_cache_enabled_) {
            delta_valid_array = py::array_t<uint8_t>(
                {static_cast<ssize_t>(delta_valid_cache_state_.size())},
                {static_cast<ssize_t>(sizeof(uint8_t))},
                delta_valid_cache_state_.data(),
                native_state_owner
            );
            delta_valid_object = delta_valid_array;
        }
        return prepare_versioned_heap_round(
            d_min,
            guard_bridges,
            valid_score_cache_array,
            delta_valid_object,
            bridge_maintenance_mode
        );
    }

    py::dict score_edge_ids_round_best_fused(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_edge_ids_array,
        const std::string& kernel_variant = "mask_count_v4_combinatorial",
        const bool profile_native_kernel = false
    ) {
        require_relshift_state_internal();
        py::object native_state_owner = py::cast(this, py::return_value_policy::reference);
        // All arrays below are non-owning contiguous views over state owned by this
        // NativeGraphState.  Explicit strides are used to prevent pybind11 from
        // selecting the copy-constructing overload.  Their lifetime is bounded by
        // this call and no vector is resized while a view exists.
        py::array_t<double> current_raw_array(
            {static_cast<ssize_t>(num_nodes_), static_cast<ssize_t>(kOrbitDim)},
            {static_cast<ssize_t>(kOrbitDim * sizeof(double)), static_cast<ssize_t>(sizeof(double))},
            current_raw_state_.data(),
            native_state_owner
        );
        py::array_t<double> current_std_array(
            {static_cast<ssize_t>(num_nodes_), static_cast<ssize_t>(kOrbitDim)},
            {static_cast<ssize_t>(kOrbitDim * sizeof(double)), static_cast<ssize_t>(sizeof(double))},
            current_std_state_.data(),
            native_state_owner
        );
        py::array_t<double> stats_mean_array(
            {static_cast<ssize_t>(stats_mean_state_.size())},
            {static_cast<ssize_t>(sizeof(double))},
            stats_mean_state_.data(),
            native_state_owner
        );
        py::array_t<double> stats_std_array(
            {static_cast<ssize_t>(stats_std_state_.size())},
            {static_cast<ssize_t>(sizeof(double))},
            stats_std_state_.data(),
            native_state_owner
        );
        py::array_t<double> score_cache_array(
            {static_cast<ssize_t>(score_cache_state_.size())},
            {static_cast<ssize_t>(sizeof(double))},
            score_cache_state_.data(),
            native_state_owner
        );
        py::array_t<double> degree_score_cache_array(
            {static_cast<ssize_t>(degree_score_cache_state_.size())},
            {static_cast<ssize_t>(sizeof(double))},
            degree_score_cache_state_.data(),
            native_state_owner
        );
        py::array_t<double> support_score_cache_array(
            {static_cast<ssize_t>(support_score_cache_state_.size())},
            {static_cast<ssize_t>(sizeof(double))},
            support_score_cache_state_.data(),
            native_state_owner
        );
        py::array_t<uint8_t> valid_score_cache_array(
            {static_cast<ssize_t>(valid_score_cache_state_.size())},
            {static_cast<ssize_t>(sizeof(uint8_t))},
            valid_score_cache_state_.data(),
            native_state_owner
        );
        py::object candidate_delta_object = py::none();
        py::object delta_valid_object = py::none();
        py::array_t<int64_t> candidate_delta_array;
        py::array_t<uint8_t> delta_valid_array;
        if (native_delta_cache_enabled_) {
            candidate_delta_array = py::array_t<int64_t>(
                {static_cast<ssize_t>(original_edge_count_), static_cast<ssize_t>(2), static_cast<ssize_t>(kOrbitDim)},
                {
                    static_cast<ssize_t>(2 * kOrbitDim * sizeof(int64_t)),
                    static_cast<ssize_t>(kOrbitDim * sizeof(int64_t)),
                    static_cast<ssize_t>(sizeof(int64_t))
                },
                candidate_delta_cache_state_.data(),
                native_state_owner
            );
            delta_valid_array = py::array_t<uint8_t>(
                {static_cast<ssize_t>(delta_valid_cache_state_.size())},
                {static_cast<ssize_t>(sizeof(uint8_t))},
                delta_valid_cache_state_.data(),
                native_state_owner
            );
            candidate_delta_object = candidate_delta_array;
            delta_valid_object = delta_valid_array;
        }
        py::dict result = score_edge_ids_round_best(
            candidate_edge_ids_array,
            current_raw_array,
            current_std_array,
            stats_mean_array,
            stats_std_array,
            native_score_mode_,
            native_eps_,
            score_cache_array,
            degree_score_cache_array,
            support_score_cache_array,
            valid_score_cache_array,
            kernel_variant,
            profile_native_kernel,
            candidate_delta_object,
            delta_valid_object
        );
        const auto candidate_edge_ids = candidate_edge_ids_array.unchecked<1>();
        for (ssize_t idx = 0; idx < candidate_edge_ids.shape(0); ++idx) {
            const int64_t edge_id = candidate_edge_ids(idx);
            if (edge_id >= 0 && edge_id < original_edge_count_) {
                support_initialized_state_[static_cast<size_t>(edge_id)] = 1;
                if (native_delta_cache_enabled_ && delta_valid_cache_state_[static_cast<size_t>(edge_id)] != 0) {
                    const int64_t* delta = candidate_delta_cache_state_.data()
                        + static_cast<size_t>(edge_id) * 2 * kOrbitDim;
                    candidate_delta_nonzero_mask_state_[static_cast<size_t>(edge_id) * 2] =
                        endpoint_delta_nonzero_mask(delta);
                    candidate_delta_nonzero_mask_state_[static_cast<size_t>(edge_id) * 2 + 1] =
                        endpoint_delta_nonzero_mask(delta + kOrbitDim);
                    const auto& edge = edge_by_id_[static_cast<size_t>(edge_id)];
                    const size_t endpoint_offset = static_cast<size_t>(edge_id) * 2;
                    endpoint_score_cache_state_[endpoint_offset] =
                        endpoint_score_from_internal_delta_masked(
                            edge[0], 0, current_raw_state_, current_std_state_,
                            stats_mean_state_, stats_std_state_, node_denominator_state_,
                            native_score_mode_, delta,
                            candidate_delta_nonzero_mask_state_[endpoint_offset]
                        );
                    endpoint_score_cache_state_[endpoint_offset + 1] =
                        endpoint_score_from_internal_delta_masked(
                            edge[1], 1, current_raw_state_, current_std_state_,
                            stats_mean_state_, stats_std_state_, node_denominator_state_,
                            native_score_mode_, delta,
                            candidate_delta_nonzero_mask_state_[endpoint_offset + 1]
                        );
                    endpoint_score_valid_mask_state_[static_cast<size_t>(edge_id)] = 3;
                    endpoint_score_recomputed_total_ += 2;
                    score_cache_state_[static_cast<size_t>(edge_id)] = (
                        endpoint_score_cache_state_[endpoint_offset]
                        + endpoint_score_cache_state_[endpoint_offset + 1]
                    ) / 2.0;
                }
            }
        }
        return result;
    }

    py::dict refresh_scores_from_delta_cache_fused(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_edge_ids_array
    ) {
        require_relshift_state_internal();
        if (!native_delta_cache_enabled_) {
            throw std::runtime_error("Native candidate delta cache is disabled.");
        }
        using Clock = std::chrono::high_resolution_clock;
        const auto start = Clock::now();
        const auto candidate_edge_ids = candidate_edge_ids_array.unchecked<1>();
        bool has_best = false;
        EdgeKey best{0.0, 0.0, 0.0, -1};
        int64_t refreshed_count = 0;
        int64_t nonzero_orbit_coordinates = 0;
        for (ssize_t idx = 0; idx < candidate_edge_ids.shape(0); ++idx) {
            const int64_t edge_id = candidate_edge_ids(idx);
            if (edge_id < 0 || edge_id >= original_edge_count_) {
                throw std::runtime_error("candidate edge id is out of range during native refresh.");
            }
            if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                continue;
            }
            if (delta_valid_cache_state_[static_cast<size_t>(edge_id)] == 0) {
                throw std::runtime_error("Native scalar refresh requires a valid candidate delta.");
            }
            const int64_t* delta = candidate_delta_cache_state_.data()
                + static_cast<size_t>(edge_id) * 2 * kOrbitDim;
            const uint16_t* masks = candidate_delta_nonzero_mask_state_.data()
                + static_cast<size_t>(edge_id) * 2;
            nonzero_orbit_coordinates += __builtin_popcount(static_cast<unsigned int>(masks[0]));
            nonzero_orbit_coordinates += __builtin_popcount(static_cast<unsigned int>(masks[1]));
            const auto& edge = edge_by_id_[static_cast<size_t>(edge_id)];
            const size_t endpoint_offset = static_cast<size_t>(edge_id) * 2;
            uint8_t endpoint_valid_mask = endpoint_score_valid_mask_state_[static_cast<size_t>(edge_id)];
            if ((endpoint_valid_mask & static_cast<uint8_t>(1)) == 0) {
                ++endpoint_score_recomputed_total_;
                endpoint_score_cache_state_[endpoint_offset] =
                    endpoint_score_from_internal_delta_masked(
                        edge[0], 0, current_raw_state_, current_std_state_,
                        stats_mean_state_, stats_std_state_, node_denominator_state_,
                        native_score_mode_, delta, masks[0]
                    );
                endpoint_valid_mask = static_cast<uint8_t>(endpoint_valid_mask | 1);
            } else {
                ++endpoint_score_reused_total_;
            }
            if ((endpoint_valid_mask & static_cast<uint8_t>(2)) == 0) {
                ++endpoint_score_recomputed_total_;
                endpoint_score_cache_state_[endpoint_offset + 1] =
                    endpoint_score_from_internal_delta_masked(
                        edge[1], 1, current_raw_state_, current_std_state_,
                        stats_mean_state_, stats_std_state_, node_denominator_state_,
                        native_score_mode_, delta, masks[1]
                    );
                endpoint_valid_mask = static_cast<uint8_t>(endpoint_valid_mask | 2);
            } else {
                ++endpoint_score_reused_total_;
            }
            endpoint_score_valid_mask_state_[static_cast<size_t>(edge_id)] = endpoint_valid_mask;
            const double combined_score = (
                endpoint_score_cache_state_[endpoint_offset]
                + endpoint_score_cache_state_[endpoint_offset + 1]
            ) / 2.0;
            const double degree_score = static_cast<double>(
                current_degrees_[static_cast<size_t>(edge[0])]
                + current_degrees_[static_cast<size_t>(edge[1])]
            );
            const double support_score = static_cast<double>(support_for_edge_internal(edge_id));
            score_cache_state_[static_cast<size_t>(edge_id)] = combined_score;
            degree_score_cache_state_[static_cast<size_t>(edge_id)] = degree_score;
            support_score_cache_state_[static_cast<size_t>(edge_id)] = support_score;
            valid_score_cache_state_[static_cast<size_t>(edge_id)] = 1;
            const EdgeKey candidate_key{combined_score, degree_score, support_score, edge_id};
            if (!has_best || edge_key_less(candidate_key, best)) {
                best = candidate_key;
                has_best = true;
            }
            ++refreshed_count;
        }
        py::dict result;
        result["best_edge_id"] = static_cast<int64_t>(has_best ? best.edge_id : -1);
        result["best_score"] = has_best ? best.score : std::numeric_limits<double>::infinity();
        result["best_degree_score"] = has_best ? best.degree_score : std::numeric_limits<double>::infinity();
        result["best_support_score"] = has_best ? best.support_score : std::numeric_limits<double>::infinity();
        result["refreshed_count"] = refreshed_count;
        result["nonzero_orbit_coordinates"] = nonzero_orbit_coordinates;
        result["native_scalar_refresh_runtime_sec"] =
            std::chrono::duration<double>(Clock::now() - start).count();
        return result;
    }

    py::dict commit_dirty_heap_keys_fused(const double rebuild_ratio = 4.0) {
        require_relshift_state_internal();
        py::object native_state_owner = py::cast(this, py::return_value_policy::reference);
        py::array_t<double> score_cache_array(
            {static_cast<ssize_t>(score_cache_state_.size())},
            {static_cast<ssize_t>(sizeof(double))},
            score_cache_state_.data(),
            native_state_owner
        );
        py::array_t<double> degree_score_cache_array(
            {static_cast<ssize_t>(degree_score_cache_state_.size())},
            {static_cast<ssize_t>(sizeof(double))},
            degree_score_cache_state_.data(),
            native_state_owner
        );
        py::array_t<double> support_score_cache_array(
            {static_cast<ssize_t>(support_score_cache_state_.size())},
            {static_cast<ssize_t>(sizeof(double))},
            support_score_cache_state_.data(),
            native_state_owner
        );
        py::array_t<uint8_t> valid_score_cache_array(
            {static_cast<ssize_t>(valid_score_cache_state_.size())},
            {static_cast<ssize_t>(sizeof(uint8_t))},
            valid_score_cache_state_.data(),
            native_state_owner
        );
        return commit_dirty_heap_keys(
            score_cache_array,
            degree_score_cache_array,
            support_score_cache_array,
            valid_score_cache_array,
            rebuild_ratio
        );
    }

    py::dict commit_dirty_indexed_heap_keys_fused() {
        require_relshift_state_internal();
        if (!heap_initialized_) {
            throw std::runtime_error("Indexed heap has not been prepared.");
        }
        if (heap_storage_mode_ != "indexed") {
            throw std::runtime_error("Indexed heap commit requested for non-indexed heap mode.");
        }
        using Clock = std::chrono::high_resolution_clock;
        const auto start = Clock::now();
        int64_t updated_count = 0;
        int64_t cleared_blocked_count = 0;
        for (const int64_t edge_id : heap_dirty_edge_ids_) {
            if (edge_id < 0 || edge_id >= original_edge_count_) {
                throw std::runtime_error("Internal dirty edge id is out of range during indexed heap commit.");
            }
            if (heap_dirty_mask_[static_cast<size_t>(edge_id)] == 0) {
                continue;
            }
            if (
                active_edge_mask_[static_cast<size_t>(edge_id)] == 0 ||
                guard_reason_[static_cast<size_t>(edge_id)] != 0
            ) {
                heap_dirty_mask_[static_cast<size_t>(edge_id)] = 0;
                indexed_heap_remove_internal(edge_id);
                ++cleared_blocked_count;
                continue;
            }
            if (valid_score_cache_state_[static_cast<size_t>(edge_id)] == 0) {
                throw std::runtime_error("Dirty eligible edge was not refreshed before indexed heap commit.");
            }
            // The old heap remains valid with respect to indexed_heap_keys_.  Publish
            // this edge's new key only now, then repair the single changed position.
            heap_dirty_mask_[static_cast<size_t>(edge_id)] = 0;
            if (indexed_heap_insert_or_update_internal(edge_id)) {
                ++updated_count;
                ++heap_push_count_total_;
            }
        }
        heap_dirty_edge_ids_.clear();
        const double runtime_sec = std::chrono::duration<double>(Clock::now() - start).count();
        py::dict result;
        result["heap_update_runtime_sec"] = runtime_sec;
        result["heap_keys_pushed"] = updated_count;
        result["heap_dirty_blocked_cleared"] = cleared_blocked_count;
        result["heap_rebuilt"] = false;
        result["heap_rebuild_edge_entries_scanned"] = static_cast<int64_t>(0);
        result["heap_size_after_update"] = static_cast<int64_t>(indexed_heap_edges_.size());
        result["heap_max_size_observed"] = heap_max_size_observed_;
        return result;
    }

    py::dict pop_best_indexed_heap(
        const std::string& bridge_maintenance_mode = "global_tarjan"
    ) {
        if (!heap_initialized_ || heap_storage_mode_ != "indexed") {
            throw std::runtime_error("Indexed heap has not been initialized.");
        }
        if (bridge_maintenance_mode != "global_tarjan" && bridge_maintenance_mode != "lazy_exact") {
            throw std::runtime_error(
                "bridge_maintenance_mode must be global_tarjan or lazy_exact."
            );
        }
        using Clock = std::chrono::high_resolution_clock;
        const auto start = Clock::now();
        int64_t popped_count = 0;
        int64_t inactive_popped = 0;
        int64_t guard_popped = 0;
        int64_t dirty_popped = 0;
        int64_t selected_edge_id = -1;
        EdgeKey selected_key{0.0, 0.0, 0.0, -1};
        int64_t lazy_bridge_queries = 0;
        int64_t lazy_bridge_support_certificates = 0;
        int64_t lazy_bridge_nodes_visited = 0;
        int64_t lazy_bridge_adjacency_entries_visited = 0;
        int64_t lazy_bridge_inactive_entries_skipped = 0;
        int64_t lazy_bridges_rejected = 0;
        double lazy_bridge_runtime_sec = 0.0;

        while (!indexed_heap_edges_.empty()) {
            const int64_t edge_id = indexed_heap_edges_.front();
            if (edge_id < 0 || edge_id >= original_edge_count_) {
                indexed_heap_remove_internal(edge_id);
                ++popped_count;
                ++heap_pop_count_total_;
                continue;
            }
            if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                indexed_heap_remove_internal(edge_id);
                ++popped_count;
                ++inactive_popped;
                ++heap_pop_count_total_;
                ++heap_inactive_pop_count_total_;
                continue;
            }
            if (guard_reason_[static_cast<size_t>(edge_id)] != 0) {
                indexed_heap_remove_internal(edge_id);
                ++popped_count;
                ++guard_popped;
                ++heap_pop_count_total_;
                ++heap_guard_pop_count_total_;
                continue;
            }
            if (heap_dirty_mask_[static_cast<size_t>(edge_id)] != 0) {
                // A dirty root here means commit was skipped; selecting it would be
                // unsafe because the cached key may have moved in either direction.
                ++dirty_popped;
                ++heap_dirty_pop_count_total_;
                throw std::runtime_error("Indexed heap root remained dirty after exact key commit.");
            }
            if (heap_guard_bridges_ && bridge_maintenance_mode == "lazy_exact") {
                const auto bridge_query_start = Clock::now();
                bool support_certified = false;
                const bool is_bridge = exact_bridge_query_internal(
                    edge_id,
                    support_certified,
                    lazy_bridge_nodes_visited,
                    lazy_bridge_adjacency_entries_visited,
                    lazy_bridge_inactive_entries_skipped
                );
                lazy_bridge_runtime_sec += std::chrono::duration<double>(
                    Clock::now() - bridge_query_start
                ).count();
                ++lazy_bridge_queries;
                if (support_certified) {
                    ++lazy_bridge_support_certificates;
                }
                if (is_bridge) {
                    mark_heap_guard_reason(edge_id, static_cast<uint8_t>(2));
                    ++lazy_bridges_rejected;
                    ++popped_count;
                    ++guard_popped;
                    ++heap_pop_count_total_;
                    ++heap_guard_pop_count_total_;
                    continue;
                }
            }
            selected_edge_id = edge_id;
            selected_key = indexed_heap_keys_[static_cast<size_t>(edge_id)];
            indexed_heap_remove_internal(edge_id);
            ++popped_count;
            ++heap_pop_count_total_;
            break;
        }

        const double runtime_sec = std::chrono::duration<double>(Clock::now() - start).count();
        py::dict result;
        result["selected_edge_id"] = selected_edge_id;
        result["best_score"] = selected_edge_id >= 0 ? selected_key.score : std::numeric_limits<double>::infinity();
        result["best_degree_score"] = selected_edge_id >= 0 ? selected_key.degree_score : std::numeric_limits<double>::infinity();
        result["best_support_score"] = selected_edge_id >= 0 ? selected_key.support_score : std::numeric_limits<double>::infinity();
        result["heap_pop_runtime_sec"] = runtime_sec;
        result["heap_entries_popped"] = popped_count;
        result["heap_stale_entries_popped"] = static_cast<int64_t>(0);
        result["heap_inactive_entries_popped"] = inactive_popped;
        result["heap_guard_entries_popped"] = guard_popped;
        result["heap_dirty_entries_popped"] = dirty_popped;
        result["heap_size_after_pop"] = static_cast<int64_t>(indexed_heap_edges_.size());
        result["lazy_bridge_queries"] = lazy_bridge_queries;
        result["lazy_bridge_support_certificates"] = lazy_bridge_support_certificates;
        result["lazy_bridge_nodes_visited"] = lazy_bridge_nodes_visited;
        result["lazy_bridge_adjacency_entries_visited"] = lazy_bridge_adjacency_entries_visited;
        result["lazy_bridge_inactive_entries_skipped"] = lazy_bridge_inactive_entries_skipped;
        result["lazy_bridges_rejected"] = lazy_bridges_rejected;
        result["lazy_bridge_runtime_sec"] = lazy_bridge_runtime_sec;
        return result;
    }

    py::dict select_best_edge_fused(
        const int d_min,
        const bool guard_bridges,
        const std::string& kernel_variant = "mask_count_v4_combinatorial",
        const bool profile_native_kernel = false,
        const double rebuild_ratio = 4.0,
        const std::string& bridge_maintenance_mode = "global_tarjan",
        const std::string& heap_storage_mode = "versioned"
    ) {
        require_relshift_state_internal();
        configure_heap_storage_mode_internal(heap_storage_mode);
        using Clock = std::chrono::high_resolution_clock;
        const auto total_start = Clock::now();

        py::dict prepared = prepare_versioned_heap_round_fused(
            d_min, guard_bridges, bridge_maintenance_mode
        );
        auto rescored_edge_ids = prepared["rescored_edge_ids"].cast<
            py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        >();
        auto refresh_edge_ids = prepared["refresh_edge_ids"].cast<
            py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        >();

        py::dict refresh_result;
        if (refresh_edge_ids.size() > 0) {
            refresh_result = refresh_scores_from_delta_cache_fused(refresh_edge_ids);
        } else {
            refresh_result["best_edge_id"] = static_cast<int64_t>(-1);
            refresh_result["refreshed_count"] = static_cast<int64_t>(0);
            refresh_result["nonzero_orbit_coordinates"] = static_cast<int64_t>(0);
            refresh_result["native_scalar_refresh_runtime_sec"] = 0.0;
        }

        py::dict score_result;
        if (rescored_edge_ids.size() > 0) {
            const auto score_start = Clock::now();
            score_result = score_edge_ids_round_best_fused(
                rescored_edge_ids,
                kernel_variant,
                profile_native_kernel
            );
            score_result["native_score_runtime_sec"] =
                std::chrono::duration<double>(Clock::now() - score_start).count();
        } else {
            score_result["best_edge_id"] = static_cast<int64_t>(-1);
            score_result["rescored_count"] = static_cast<int64_t>(0);
            score_result["avg_directly_attached_size"] = 0.0;
            score_result["avg_four_node_pair_count"] = 0.0;
            score_result["native_pair_generation_runtime_sec"] = 0.0;
            score_result["native_delta_accumulation_runtime_sec"] = 0.0;
            score_result["native_score_scalarization_runtime_sec"] = 0.0;
            score_result["native_score_runtime_sec"] = 0.0;
        }

        py::dict heap_update = heap_storage_mode_ == "indexed"
            ? commit_dirty_indexed_heap_keys_fused()
            : commit_dirty_heap_keys_fused(rebuild_ratio);
        py::dict heap_pop = heap_storage_mode_ == "indexed"
            ? pop_best_indexed_heap(bridge_maintenance_mode)
            : pop_best_versioned_heap(bridge_maintenance_mode);

        py::dict result;
        // Guard and dirty-partition metrics.
        for (const char* key : {
            "eligible_count", "rescored_count", "refresh_count", "reused_count",
            "blocked_by_bridge_count", "blocked_by_d_min_count", "bridge_count",
            "bridge_runtime_sec", "bridge_nodes_visited",
            "bridge_adjacency_entries_visited",
            "bridge_inactive_adjacency_entries_skipped", "eligibility_runtime_sec",
            "active_edge_id_entries_scanned", "inactive_edge_ids_skipped",
            "dirty_edge_entries_scanned", "dirty_inactive_or_guarded_skipped",
            "heap_size_before_update"
        }) {
            result[key] = prepared[key];
        }

        // Scoring metrics.
        result["scalar_refreshed_edge_count"] = refresh_result["refreshed_count"];
        result["scalar_refresh_nonzero_orbit_coordinates"] =
            refresh_result["nonzero_orbit_coordinates"];
        result["native_scalar_refresh_runtime_sec"] =
            refresh_result["native_scalar_refresh_runtime_sec"];
        result["native_score_runtime_sec"] = score_result["native_score_runtime_sec"];
        result["avg_directly_attached_size"] = score_result["avg_directly_attached_size"];
        result["avg_four_node_pair_count"] = score_result["avg_four_node_pair_count"];
        result["native_pair_generation_runtime_sec"] =
            score_result["native_pair_generation_runtime_sec"];
        result["native_delta_accumulation_runtime_sec"] =
            score_result["native_delta_accumulation_runtime_sec"];
        result["native_score_scalarization_runtime_sec"] =
            score_result["native_score_scalarization_runtime_sec"];

        // Heap metrics and exact selected key.
        for (const char* key : {
            "heap_update_runtime_sec", "heap_keys_pushed", "heap_dirty_blocked_cleared",
            "heap_rebuilt", "heap_rebuild_edge_entries_scanned",
            "heap_size_after_update", "heap_max_size_observed"
        }) {
            result[key] = heap_update[key];
        }
        for (const char* key : {
            "selected_edge_id", "best_score", "best_degree_score", "best_support_score",
            "heap_pop_runtime_sec", "heap_entries_popped", "heap_stale_entries_popped",
            "heap_inactive_entries_popped", "heap_guard_entries_popped",
            "heap_dirty_entries_popped", "heap_size_after_pop",
            "lazy_bridge_queries", "lazy_bridge_support_certificates",
            "lazy_bridge_nodes_visited", "lazy_bridge_adjacency_entries_visited",
            "lazy_bridge_inactive_entries_skipped", "lazy_bridges_rejected",
            "lazy_bridge_runtime_sec"
        }) {
            result[key] = heap_pop[key];
        }
        result["bridge_runtime_sec"] =
            prepared["bridge_runtime_sec"].cast<double>()
            + heap_pop["lazy_bridge_runtime_sec"].cast<double>();
        result["bridge_nodes_visited"] =
            prepared["bridge_nodes_visited"].cast<int64_t>()
            + heap_pop["lazy_bridge_nodes_visited"].cast<int64_t>();
        result["bridge_adjacency_entries_visited"] =
            prepared["bridge_adjacency_entries_visited"].cast<int64_t>()
            + heap_pop["lazy_bridge_adjacency_entries_visited"].cast<int64_t>();
        result["bridge_inactive_adjacency_entries_skipped"] =
            prepared["bridge_inactive_adjacency_entries_skipped"].cast<int64_t>()
            + heap_pop["lazy_bridge_inactive_entries_skipped"].cast<int64_t>();
        result["bridge_count"] = bridge_blocked_count_;
        result["blocked_by_bridge_count"] = bridge_blocked_count_;
        result["blocked_by_d_min_count"] = d_min_blocked_count_;
        const int64_t selected_edge_id = result["selected_edge_id"].cast<int64_t>();
        result["eligible_count"] = selected_edge_id >= 0
            ? std::max<int64_t>(eligible_active_count_, 1)
            : static_cast<int64_t>(0);
        if (selected_edge_id >= 0 && selected_edge_id < original_edge_count_) {
            const auto& selected_edge = edge_output_by_id_[static_cast<size_t>(selected_edge_id)];
            result["selected_u"] = selected_edge[0];
            result["selected_v"] = selected_edge[1];
        } else {
            result["selected_u"] = -1;
            result["selected_v"] = -1;
        }
        result["native_selection_transaction_runtime_sec"] =
            std::chrono::duration<double>(Clock::now() - total_start).count();
        return result;
    }

    py::dict apply_selected_edge_and_remove_fused(
        const int64_t selected_edge_id,
        const double adjacency_compaction_threshold = 0.20
    ) {
        require_relshift_state_internal();
        py::object native_state_owner = py::cast(this, py::return_value_policy::reference);
        if (adjacency_compaction_threshold < 0.0 || adjacency_compaction_threshold >= 1.0) {
            throw std::runtime_error("adjacency compaction threshold must be in [0, 1).");
        }
        if (selected_edge_id < 0 || selected_edge_id >= original_edge_count_) {
            throw std::runtime_error("selected edge id is out of range.");
        }
        if (active_edge_mask_[static_cast<size_t>(selected_edge_id)] == 0) {
            throw std::runtime_error("selected edge is already inactive.");
        }
        using Clock = std::chrono::high_resolution_clock;
        const auto transaction_start = Clock::now();
        const auto& selected_edge = edge_by_id_[static_cast<size_t>(selected_edge_id)];

        py::array_t<uint8_t> valid_score_cache_array(
            {static_cast<ssize_t>(valid_score_cache_state_.size())},
            {static_cast<ssize_t>(sizeof(uint8_t))},
            valid_score_cache_state_.data(),
            native_state_owner
        );
        py::dict delta_result;
        py::array_t<uint8_t> delta_valid_array;
        py::array_t<int64_t> candidate_delta_array;
        if (native_delta_cache_enabled_) {
            delta_valid_array = py::array_t<uint8_t>(
                {static_cast<ssize_t>(delta_valid_cache_state_.size())},
                {static_cast<ssize_t>(sizeof(uint8_t))},
                delta_valid_cache_state_.data(),
                native_state_owner
            );
            candidate_delta_array = py::array_t<int64_t>(
                {static_cast<ssize_t>(original_edge_count_), static_cast<ssize_t>(2), static_cast<ssize_t>(kOrbitDim)},
                {
                    static_cast<ssize_t>(2 * kOrbitDim * sizeof(int64_t)),
                    static_cast<ssize_t>(kOrbitDim * sizeof(int64_t)),
                    static_cast<ssize_t>(sizeof(int64_t))
                },
                candidate_delta_cache_state_.data(),
                native_state_owner
            );
            delta_result = compute_selected_edge_delta_and_update_candidate_cache(
                selected_edge_id,
                valid_score_cache_array,
                delta_valid_array,
                candidate_delta_array,
                false
            );
        } else {
            delta_result = compute_selected_edge_delta_and_invalidate(
                selected_edge_id, valid_score_cache_array
            );
            const auto affected_nodes_array = delta_result["affected_nodes"].cast<
                py::array_t<int64_t, py::array::c_style | py::array::forcecast>
            >();
            const auto raw_delta_array = delta_result["raw_delta"].cast<
                py::array_t<double, py::array::c_style | py::array::forcecast>
            >();
            const auto affected_nodes_view = affected_nodes_array.unchecked<1>();
            const auto raw_delta_view = raw_delta_array.unchecked<2>();
            selected_affected_nodes_workspace_.resize(
                static_cast<size_t>(affected_nodes_view.shape(0))
            );
            selected_raw_delta_workspace_.resize(
                static_cast<size_t>(affected_nodes_view.shape(0))
                * static_cast<size_t>(kOrbitDim)
            );
            for (ssize_t row = 0; row < affected_nodes_view.shape(0); ++row) {
                selected_affected_nodes_workspace_[static_cast<size_t>(row)] =
                    static_cast<int>(affected_nodes_view(row));
                for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
                    selected_raw_delta_workspace_[
                        static_cast<size_t>(row) * static_cast<size_t>(kOrbitDim)
                        + static_cast<size_t>(orbit_idx)
                    ] = raw_delta_view(row, orbit_idx);
                }
            }
        }
        const auto& affected_nodes = selected_affected_nodes_workspace_;
        const auto& raw_delta = selected_raw_delta_workspace_;
        if (raw_delta.size() != affected_nodes.size() * static_cast<size_t>(kOrbitDim)) {
            throw std::runtime_error("selected-edge native delta workspace has inconsistent shape.");
        }
        int64_t changed_node_count = 0;
        for (size_t row = 0; row < affected_nodes.size(); ++row) {
            const int node = affected_nodes[row];
            const size_t delta_offset = row * static_cast<size_t>(kOrbitDim);
            bool changed = false;
            for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
                if (raw_delta[delta_offset + static_cast<size_t>(orbit_idx)] != 0.0) {
                    changed = true;
                    break;
                }
            }
            if (!changed) {
                continue;
            }
            ++changed_node_count;
            const int64_t row_offset = static_cast<int64_t>(node) * kOrbitDim;
            double denominator = native_eps_;
            for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
                const size_t offset = static_cast<size_t>(row_offset + orbit_idx);
                const double updated_raw = std::max(
                    current_raw_state_[offset] + raw_delta[delta_offset + static_cast<size_t>(orbit_idx)],
                    0.0
                );
                const double updated_std = canonical_standardized_coordinate(
                    updated_raw,
                    stats_mean_state_[static_cast<size_t>(orbit_idx)],
                    stats_std_state_[static_cast<size_t>(orbit_idx)]
                );
                current_raw_state_[offset] = updated_raw;
                current_std_state_[offset] = updated_std;
                denominator += std::abs(updated_std);
            }
            node_denominator_state_[static_cast<size_t>(node)] = denominator;
        }
        native_state_node_rows_updated_ += changed_node_count;

        // Exact incremental triangle-support maintenance.  Only the two non-selected
        // edges of triangles containing the selected edge lose one support count.
        int64_t support_decrement_count = 0;
        int64_t left = row_ptr_[static_cast<size_t>(selected_edge[0])];
        int64_t right = row_ptr_[static_cast<size_t>(selected_edge[1])];
        const int64_t left_stop = row_ptr_[static_cast<size_t>(selected_edge[0] + 1)];
        const int64_t right_stop = row_ptr_[static_cast<size_t>(selected_edge[1] + 1)];
        while (left < left_stop && right < right_stop) {
            while (left < left_stop && !adjacency_entry_is_active(
                left, adjacency_edge_ids_.data(), active_edge_mask_.data()
            )) {
                ++left;
            }
            while (right < right_stop && !adjacency_entry_is_active(
                right, adjacency_edge_ids_.data(), active_edge_mask_.data()
            )) {
                ++right;
            }
            if (left >= left_stop || right >= right_stop) {
                break;
            }
            const int left_node = static_cast<int>(col_idx_[static_cast<size_t>(left)]);
            const int right_node = static_cast<int>(col_idx_[static_cast<size_t>(right)]);
            if (left_node == right_node) {
                const int common = left_node;
                for (const uint64_t code : {
                    encode_pair(selected_edge[0], common),
                    encode_pair(selected_edge[1], common)
                }) {
                    const auto found = edge_code_to_id_.find(code);
                    if (found == edge_code_to_id_.end()) {
                        throw std::runtime_error("triangle partner edge is missing from edge id map.");
                    }
                    const int64_t partner_edge_id = found->second;
                    const int current_support = support_for_edge_internal(partner_edge_id);
                    if (current_support <= 0) {
                        throw std::runtime_error("triangle support underflow during selected-edge update.");
                    }
                    support_score_cache_state_[static_cast<size_t>(partner_edge_id)] =
                        static_cast<double>(current_support - 1);
                    support_initialized_state_[static_cast<size_t>(partner_edge_id)] = 1;
                    valid_score_cache_state_[static_cast<size_t>(partner_edge_id)] = 0;
                    mark_heap_dirty(partner_edge_id);
                    ++support_decrement_count;
                }
                ++left;
                ++right;
            } else if (left_node < right_node) {
                ++left;
            } else {
                ++right;
            }
        }
        native_state_support_decrements_ += support_decrement_count;
        support_initialized_state_[static_cast<size_t>(selected_edge_id)] = 0;
        valid_score_cache_state_[static_cast<size_t>(selected_edge_id)] = 0;
        if (native_delta_cache_enabled_) {
            delta_valid_cache_state_[static_cast<size_t>(selected_edge_id)] = 0;
        }

        remove_edge(selected_edge[0], selected_edge[1]);
        const py::dict compaction_result = maybe_compact_adjacency_internal(
            adjacency_compaction_threshold
        );
        ++native_state_round_count_;

        delta_result["changed_node_count"] = changed_node_count;
        delta_result["adjacency_compacted"] = compaction_result["adjacency_compacted"];
        delta_result["adjacency_compaction_runtime_sec"] =
            compaction_result["adjacency_compaction_runtime_sec"];
        delta_result["adjacency_entries_before_compaction"] =
            compaction_result["adjacency_entries_before_compaction"];
        delta_result["adjacency_entries_after_compaction"] =
            compaction_result["adjacency_entries_after_compaction"];
        delta_result["support_decrement_count"] = support_decrement_count;
        delta_result["native_state_transaction_runtime_sec"] =
            std::chrono::duration<double>(Clock::now() - transaction_start).count();
        delta_result["state_updated_and_edge_removed"] = true;
        // The evolving state and cache invalidation have already been applied in
        // this object.  Do not transfer O(|B_e|*15) diagnostic arrays back to
        // Python on every exact round; only scalar audit counters cross the
        // language boundary.  Dedicated snapshot methods remain available for
        // equivalence tests.
        delta_result.attr("pop")("affected_nodes", py::none());
        delta_result.attr("pop")("raw_delta", py::none());
        delta_result.attr("pop")("impacted_edges", py::none());
        return delta_result;
    }

    py::array_t<int64_t> candidate_delta_cache_snapshot() const {
        if (!relshift_state_initialized_) {
            throw std::runtime_error("RelShift state is not initialized.");
        }
        py::array_t<int64_t> result(py::array::ShapeContainer{
            static_cast<ssize_t>(original_edge_count_),
            static_cast<ssize_t>(2),
            static_cast<ssize_t>(kOrbitDim)
        });
        auto* output = static_cast<int64_t*>(result.mutable_data());
        if (candidate_delta_cache_state_.empty()) {
            std::fill(output, output + static_cast<size_t>(original_edge_count_) * 2U * kOrbitDim, 0);
        } else {
            std::copy(candidate_delta_cache_state_.begin(), candidate_delta_cache_state_.end(), output);
        }
        return result;
    }

    py::array_t<uint8_t> delta_valid_cache_snapshot() const {
        if (!relshift_state_initialized_) {
            throw std::runtime_error("RelShift state is not initialized.");
        }
        py::array_t<uint8_t> result(static_cast<ssize_t>(delta_valid_cache_state_.size()));
        std::copy(delta_valid_cache_state_.begin(), delta_valid_cache_state_.end(), static_cast<uint8_t*>(result.mutable_data()));
        return result;
    }

    py::array_t<uint8_t> valid_score_cache_snapshot() const {
        if (!relshift_state_initialized_) {
            throw std::runtime_error("RelShift state is not initialized.");
        }
        py::array_t<uint8_t> result(static_cast<ssize_t>(valid_score_cache_state_.size()));
        std::copy(valid_score_cache_state_.begin(), valid_score_cache_state_.end(), static_cast<uint8_t*>(result.mutable_data()));
        return result;
    }

    py::array_t<double> score_cache_snapshot() const {
        if (!relshift_state_initialized_) {
            throw std::runtime_error("RelShift state is not initialized.");
        }
        py::array_t<double> result(static_cast<ssize_t>(score_cache_state_.size()));
        std::copy(score_cache_state_.begin(), score_cache_state_.end(), static_cast<double*>(result.mutable_data()));
        return result;
    }

    py::array_t<double> current_raw_snapshot() const {
        require_relshift_state_internal();
        py::array_t<double> result(
            {static_cast<ssize_t>(num_nodes_), static_cast<ssize_t>(kOrbitDim)}
        );
        std::copy(current_raw_state_.begin(), current_raw_state_.end(), result.mutable_data());
        return result;
    }

    py::array_t<double> current_std_snapshot() const {
        require_relshift_state_internal();
        py::array_t<double> result(
            {static_cast<ssize_t>(num_nodes_), static_cast<ssize_t>(kOrbitDim)}
        );
        std::copy(current_std_state_.begin(), current_std_state_.end(), result.mutable_data());
        return result;
    }

    py::array_t<uint8_t> active_edge_mask_snapshot() const {
        py::array_t<uint8_t> result(static_cast<ssize_t>(active_edge_mask_.size()));
        std::copy(active_edge_mask_.begin(), active_edge_mask_.end(), result.mutable_data());
        return result;
    }

    py::array_t<int64_t> active_edges_snapshot() const {
        py::array_t<int64_t> result({active_edge_count_, static_cast<int64_t>(2)});
        auto view = result.mutable_unchecked<2>();
        ssize_t row = 0;
        for (int64_t edge_id = 0; edge_id < original_edge_count_; ++edge_id) {
            if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                continue;
            }
            const auto& edge = edge_output_by_id_[static_cast<size_t>(edge_id)];
            view(row, 0) = edge[0];
            view(row, 1) = edge[1];
            ++row;
        }
        if (row != active_edge_count_) {
            throw std::runtime_error("active edge snapshot count mismatch.");
        }
        return result;
    }

    py::array_t<int64_t> removed_edges_snapshot() const {
        py::array_t<int64_t> result({static_cast<ssize_t>(removed_edge_ids_.size()), static_cast<ssize_t>(2)});
        auto view = result.mutable_unchecked<2>();
        for (size_t row = 0; row < removed_edge_ids_.size(); ++row) {
            const int64_t edge_id = removed_edge_ids_[row];
            const auto& edge = edge_output_by_id_[static_cast<size_t>(edge_id)];
            view(static_cast<ssize_t>(row), 0) = edge[0];
            view(static_cast<ssize_t>(row), 1) = edge[1];
        }
        return result;
    }

    py::array_t<int64_t> current_degrees_snapshot() const {
        py::array_t<int64_t> result(static_cast<ssize_t>(current_degrees_.size()));
        std::copy(current_degrees_.begin(), current_degrees_.end(), result.mutable_data());
        return result;
    }

    int64_t edge_support_fused(const int64_t edge_id) {
        require_relshift_state_internal();
        if (edge_id < 0 || edge_id >= original_edge_count_) {
            throw std::runtime_error("edge_support_fused edge id is out of range.");
        }
        if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
            return -1;
        }
        return static_cast<int64_t>(support_for_edge_internal(edge_id));
    }

    py::dict relshift_state_statistics() const {
        py::dict result;
        result["initialized"] = relshift_state_initialized_;
        result["native_delta_cache_enabled"] = native_delta_cache_enabled_;
        result["round_count"] = native_state_round_count_;
        result["node_rows_updated"] = native_state_node_rows_updated_;
        result["support_initializations"] = native_state_support_initializations_;
        result["support_decrements"] = native_state_support_decrements_;
        result["endpoint_score_recomputed_total"] = endpoint_score_recomputed_total_;
        result["endpoint_score_reused_total"] = endpoint_score_reused_total_;
        result["selected_four_node_pairs_total"] = selected_four_node_pairs_total_;
        result["selected_four_node_pairs_peak"] = selected_four_node_pairs_peak_;
        result["selected_affected_nodes_peak"] = selected_affected_nodes_peak_;
        result["selected_raw_delta_workspace_capacity_bytes"] = static_cast<int64_t>(
            selected_raw_delta_workspace_.capacity() * sizeof(double)
        );
        result["selected_pair_workspace_capacity_bytes"] = static_cast<int64_t>(
            selected_pair_masks_workspace_.capacity() * sizeof(PairMask)
        );
        result["adjacency_compaction_count"] = adjacency_compaction_count_;
        result["adjacency_compaction_entries_copied_total"] =
            adjacency_compaction_entries_copied_total_;
        result["adjacency_compaction_runtime_sec_total"] =
            adjacency_compaction_runtime_sec_total_;
        result["raw_state_bytes"] = static_cast<int64_t>(current_raw_state_.size() * sizeof(double));
        result["std_state_bytes"] = static_cast<int64_t>(current_std_state_.size() * sizeof(double));
        result["node_denominator_bytes"] = static_cast<int64_t>(node_denominator_state_.size() * sizeof(double));
        result["graphlet_workspace_bytes"] = static_cast<int64_t>(
            graphlet_marks_workspace_.size() * sizeof(int)
            + graphlet_attachment_workspace_.size() * sizeof(int)
            + node_workspace_epoch_.size() * sizeof(uint32_t)
            + node_workspace_index_.size() * sizeof(int)
        );
        result["score_cache_bytes"] = static_cast<int64_t>(score_cache_state_.size() * sizeof(double));
        result["endpoint_score_cache_bytes"] = static_cast<int64_t>(
            endpoint_score_cache_state_.size() * sizeof(double)
        );
        result["endpoint_score_valid_mask_bytes"] = static_cast<int64_t>(
            endpoint_score_valid_mask_state_.size() * sizeof(uint8_t)
        );
        result["degree_cache_bytes"] = static_cast<int64_t>(degree_score_cache_state_.size() * sizeof(double));
        result["support_cache_bytes"] = static_cast<int64_t>(support_score_cache_state_.size() * sizeof(double));
        result["valid_cache_bytes"] = static_cast<int64_t>(valid_score_cache_state_.size() * sizeof(uint8_t));
        result["candidate_delta_cache_bytes"] = static_cast<int64_t>(candidate_delta_cache_state_.size() * sizeof(int64_t));
        result["candidate_delta_nonzero_mask_bytes"] = static_cast<int64_t>(
            candidate_delta_nonzero_mask_state_.size() * sizeof(uint16_t)
        );
        result["delta_valid_cache_bytes"] = static_cast<int64_t>(delta_valid_cache_state_.size() * sizeof(uint8_t));
        const int64_t numeric_state_bytes = static_cast<int64_t>(
            current_raw_state_.size() * sizeof(double)
            + current_std_state_.size() * sizeof(double)
            + node_denominator_state_.size() * sizeof(double)
            + graphlet_marks_workspace_.size() * sizeof(int)
            + graphlet_attachment_workspace_.size() * sizeof(int)
            + node_workspace_epoch_.size() * sizeof(uint32_t)
            + node_workspace_index_.size() * sizeof(int)
            + selected_raw_delta_workspace_.capacity() * sizeof(double)
            + selected_pair_masks_workspace_.capacity() * sizeof(PairMask)
            + score_cache_state_.size() * sizeof(double)
            + endpoint_score_cache_state_.size() * sizeof(double)
            + endpoint_score_valid_mask_state_.size() * sizeof(uint8_t)
            + degree_score_cache_state_.size() * sizeof(double)
            + support_score_cache_state_.size() * sizeof(double)
            + valid_score_cache_state_.size() * sizeof(uint8_t)
            + candidate_delta_cache_state_.size() * sizeof(int64_t)
            + candidate_delta_nonzero_mask_state_.size() * sizeof(uint16_t)
            + delta_valid_cache_state_.size() * sizeof(uint8_t)
        );
        result["native_numeric_state_total_bytes"] = numeric_state_bytes;
        result["valid_score_count"] = static_cast<int64_t>(std::count(
            valid_score_cache_state_.begin(), valid_score_cache_state_.end(), static_cast<uint8_t>(1)
        ));
        result["delta_valid_count"] = static_cast<int64_t>(std::count(
            delta_valid_cache_state_.begin(), delta_valid_cache_state_.end(), static_cast<uint8_t>(1)
        ));
        return result;
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

    py::dict prepare_versioned_heap_round(
        const int d_min,
        const bool guard_bridges,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array,
        py::object delta_valid_cache_object = py::none(),
        const std::string& bridge_maintenance_mode = "global_tarjan"
    ) {
        if (!has_edge_ids_) {
            throw std::runtime_error("Heap selection requires stable edge id state.");
        }
        if (!heap_storage_mode_initialized_) {
            configure_heap_storage_mode_internal("versioned");
        }
        if (valid_score_cache_array.ndim() != 1 || valid_score_cache_array.shape(0) != original_edge_count_) {
            throw std::runtime_error("valid_score_cache must have one entry per original edge.");
        }
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> delta_valid_cache_array;
        const uint8_t* delta_valid_cache_ptr = nullptr;
        ssize_t delta_valid_cache_size = 0;
        if (!delta_valid_cache_object.is_none()) {
            delta_valid_cache_array = delta_valid_cache_object.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
            if (delta_valid_cache_array.ndim() != 1 || delta_valid_cache_array.shape(0) != original_edge_count_) {
                throw std::runtime_error("delta_valid_cache must have one entry per original edge.");
            }
            delta_valid_cache_ptr = static_cast<const uint8_t*>(delta_valid_cache_array.data());
            delta_valid_cache_size = delta_valid_cache_array.shape(0);
        }
        if (bridge_maintenance_mode != "global_tarjan" && bridge_maintenance_mode != "lazy_exact") {
            throw std::runtime_error(
                "bridge_maintenance_mode must be global_tarjan or lazy_exact."
            );
        }
        if (!heap_guard_configuration_initialized_) {
            heap_d_min_ = d_min;
            heap_guard_bridges_ = guard_bridges;
            heap_bridge_maintenance_mode_ = bridge_maintenance_mode;
            heap_guard_configuration_initialized_ = true;
        } else if (
            heap_d_min_ != d_min ||
            heap_guard_bridges_ != guard_bridges ||
            heap_bridge_maintenance_mode_ != bridge_maintenance_mode
        ) {
            throw std::runtime_error("Versioned heap guard configuration cannot change during pruning.");
        }

        using Clock = std::chrono::high_resolution_clock;
        const auto bridge_start = Clock::now();
        std::unordered_set<uint64_t> bridge_codes;
        int64_t bridge_nodes_visited = 0;
        int64_t bridge_adjacency_entries_visited = 0;
        int64_t bridge_inactive_adjacency_entries_skipped = 0;
        if (guard_bridges && bridge_maintenance_mode == "global_tarjan") {
            std::vector<int> discovery(static_cast<size_t>(num_nodes_), -1);
            std::vector<int> low(static_cast<size_t>(num_nodes_), 0);
            std::vector<int> parent(static_cast<size_t>(num_nodes_), -1);
            int visit_time = 0;
            bridge_codes.reserve(static_cast<size_t>(active_edge_count_ / 8 + 1));
            std::function<void(int)> visit = [&](const int node) {
                ++bridge_nodes_visited;
                discovery[static_cast<size_t>(node)] = visit_time;
                low[static_cast<size_t>(node)] = visit_time;
                ++visit_time;
                for (
                    int64_t idx = row_ptr_[static_cast<size_t>(node)];
                    idx < row_ptr_[static_cast<size_t>(node + 1)];
                    ++idx
                ) {
                    ++bridge_adjacency_entries_visited;
                    if (!adjacency_entry_is_active(idx, adjacency_edge_ids_.data(), active_edge_mask_.data())) {
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
        if (!heap_initialized_) {
            eligible_active_count_ = active_edge_count_;
            for (const uint64_t bridge_code : bridge_codes) {
                const auto found = edge_code_to_id_.find(bridge_code);
                if (found != edge_code_to_id_.end()) {
                    mark_heap_guard_reason(found->second, static_cast<uint8_t>(2));
                }
            }
            for (int node = 0; node < num_nodes_; ++node) {
                if (current_degrees_[static_cast<size_t>(node)] <= d_min) {
                    for (const int64_t edge_id : incident_edge_ids_[static_cast<size_t>(node)]) {
                        mark_heap_guard_reason(edge_id, static_cast<uint8_t>(1));
                    }
                }
            }
            heap_initialized_ = true;
            for (int64_t edge_id = 0; edge_id < original_edge_count_; ++edge_id) {
                if (
                    active_edge_mask_[static_cast<size_t>(edge_id)] != 0 &&
                    guard_reason_[static_cast<size_t>(edge_id)] == 0
                ) {
                    mark_heap_dirty(edge_id);
                }
            }
        } else {
            for (const uint64_t bridge_code : bridge_codes) {
                const auto found = edge_code_to_id_.find(bridge_code);
                if (found != edge_code_to_id_.end()) {
                    mark_heap_guard_reason(found->second, static_cast<uint8_t>(2));
                }
            }
        }

        const auto valid_score_cache = valid_score_cache_array.unchecked<1>();
        std::vector<int64_t> rescored_ids;
        std::vector<int64_t> refresh_ids;
        rescored_ids.reserve(heap_dirty_edge_ids_.size());
        refresh_ids.reserve(heap_dirty_edge_ids_.size());
        int64_t dirty_entries_scanned = 0;
        int64_t dirty_inactive_or_guarded_skipped = 0;
        for (const int64_t edge_id : heap_dirty_edge_ids_) {
            ++dirty_entries_scanned;
            if (edge_id < 0 || edge_id >= original_edge_count_) {
                throw std::runtime_error("Internal dirty edge id is out of range.");
            }
            if (heap_dirty_mask_[static_cast<size_t>(edge_id)] == 0) {
                continue;
            }
            if (
                active_edge_mask_[static_cast<size_t>(edge_id)] == 0 ||
                guard_reason_[static_cast<size_t>(edge_id)] != 0
            ) {
                heap_dirty_mask_[static_cast<size_t>(edge_id)] = 0;
                ++dirty_inactive_or_guarded_skipped;
                continue;
            }
            (void)valid_score_cache;
            if (
                delta_valid_cache_ptr != nullptr &&
                edge_id < delta_valid_cache_size &&
                delta_valid_cache_ptr[edge_id] != 0
            ) {
                refresh_ids.push_back(edge_id);
            } else {
                rescored_ids.push_back(edge_id);
            }
        }
        const int64_t dirty_candidate_count = static_cast<int64_t>(rescored_ids.size() + refresh_ids.size());
        const int64_t reused_count = std::max<int64_t>(0, eligible_active_count_ - dirty_candidate_count);
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
        result["rescored_edge_ids"] = to_array(rescored_ids);
        result["refresh_edge_ids"] = to_array(refresh_ids);
        result["eligible_count"] = eligible_active_count_;
        result["rescored_count"] = static_cast<int64_t>(rescored_ids.size());
        result["refresh_count"] = static_cast<int64_t>(refresh_ids.size());
        result["reused_count"] = reused_count;
        result["blocked_by_bridge_count"] = bridge_blocked_count_;
        result["blocked_by_d_min_count"] = d_min_blocked_count_;
        result["bridge_count"] = static_cast<int64_t>(bridge_codes.size());
        result["bridge_runtime_sec"] = bridge_runtime_sec;
        result["bridge_nodes_visited"] = bridge_nodes_visited;
        result["bridge_adjacency_entries_visited"] = bridge_adjacency_entries_visited;
        result["bridge_inactive_adjacency_entries_skipped"] = bridge_inactive_adjacency_entries_skipped;
        result["eligibility_runtime_sec"] = eligibility_runtime_sec;
        result["active_edge_id_entries_scanned"] = static_cast<int64_t>(0);
        result["inactive_edge_ids_skipped"] = static_cast<int64_t>(0);
        result["dirty_edge_entries_scanned"] = dirty_entries_scanned;
        result["dirty_inactive_or_guarded_skipped"] = dirty_inactive_or_guarded_skipped;
        result["heap_size_before_update"] = selection_heap_size_internal();
        return result;
    }

    py::dict commit_dirty_heap_keys(
        py::array_t<double, py::array::c_style | py::array::forcecast> score_cache_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> degree_score_cache_array,
        py::array_t<double, py::array::c_style | py::array::forcecast> support_score_cache_array,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> valid_score_cache_array,
        const double rebuild_ratio = 4.0
    ) {
        if (!heap_initialized_) {
            throw std::runtime_error("Versioned heap has not been prepared.");
        }
        if (rebuild_ratio < 1.0) {
            throw std::runtime_error("heap rebuild ratio must be at least 1.0.");
        }
        const auto score_cache = score_cache_array.unchecked<1>();
        const auto degree_score_cache = degree_score_cache_array.unchecked<1>();
        const auto support_score_cache = support_score_cache_array.unchecked<1>();
        const auto valid_score_cache = valid_score_cache_array.unchecked<1>();
        using Clock = std::chrono::high_resolution_clock;
        const auto start = Clock::now();
        int64_t pushed_count = 0;
        int64_t cleared_blocked_count = 0;
        for (const int64_t edge_id : heap_dirty_edge_ids_) {
            if (edge_id < 0 || edge_id >= original_edge_count_) {
                throw std::runtime_error("Internal dirty edge id is out of range during heap commit.");
            }
            if (heap_dirty_mask_[static_cast<size_t>(edge_id)] == 0) {
                continue;
            }
            if (
                active_edge_mask_[static_cast<size_t>(edge_id)] == 0 ||
                guard_reason_[static_cast<size_t>(edge_id)] != 0
            ) {
                heap_dirty_mask_[static_cast<size_t>(edge_id)] = 0;
                ++cleared_blocked_count;
                continue;
            }
            if (valid_score_cache(edge_id) == 0) {
                throw std::runtime_error("Dirty eligible edge was not refreshed before heap commit.");
            }
            const EdgeKey key{
                score_cache(edge_id),
                degree_score_cache(edge_id),
                support_score_cache(edge_id),
                edge_id,
            };
            selection_heap_.push(VersionedHeapEntry{key, heap_versions_[static_cast<size_t>(edge_id)]});
            ++heap_push_count_total_;
            ++pushed_count;
            heap_dirty_mask_[static_cast<size_t>(edge_id)] = 0;
        }
        heap_dirty_edge_ids_.clear();
        heap_max_size_observed_ = std::max<int64_t>(
            heap_max_size_observed_,
            static_cast<int64_t>(selection_heap_.size())
        );

        bool rebuilt = false;
        const int64_t rebuild_scan_count_before = heap_rebuild_edge_entries_scanned_total_;
        const int64_t rebuild_floor = 1024;
        const double rebuild_limit = rebuild_ratio * static_cast<double>(std::max<int64_t>(eligible_active_count_, 1));
        if (
            static_cast<int64_t>(selection_heap_.size()) > rebuild_floor &&
            static_cast<double>(selection_heap_.size()) > rebuild_limit
        ) {
            rebuild_selection_heap(score_cache, degree_score_cache, support_score_cache, valid_score_cache);
            rebuilt = true;
        }
        const double runtime_sec = std::chrono::duration<double>(Clock::now() - start).count();
        py::dict result;
        result["heap_update_runtime_sec"] = runtime_sec;
        result["heap_keys_pushed"] = pushed_count;
        result["heap_dirty_blocked_cleared"] = cleared_blocked_count;
        result["heap_rebuilt"] = rebuilt;
        result["heap_rebuild_edge_entries_scanned"] =
            heap_rebuild_edge_entries_scanned_total_ - rebuild_scan_count_before;
        result["heap_size_after_update"] = static_cast<int64_t>(selection_heap_.size());
        result["heap_max_size_observed"] = heap_max_size_observed_;
        return result;
    }

    py::dict pop_best_versioned_heap(
        const std::string& bridge_maintenance_mode = "global_tarjan"
    ) {
        if (!heap_initialized_) {
            throw std::runtime_error("Versioned heap has not been initialized.");
        }
        using Clock = std::chrono::high_resolution_clock;
        const auto start = Clock::now();
        int64_t popped_count = 0;
        int64_t stale_popped = 0;
        int64_t inactive_popped = 0;
        int64_t guard_popped = 0;
        int64_t dirty_popped = 0;
        int64_t selected_edge_id = -1;
        EdgeKey selected_key{0.0, 0.0, 0.0, -1};
        int64_t lazy_bridge_queries = 0;
        int64_t lazy_bridge_support_certificates = 0;
        int64_t lazy_bridge_nodes_visited = 0;
        int64_t lazy_bridge_adjacency_entries_visited = 0;
        int64_t lazy_bridge_inactive_entries_skipped = 0;
        int64_t lazy_bridges_rejected = 0;
        double lazy_bridge_runtime_sec = 0.0;
        if (bridge_maintenance_mode != "global_tarjan" && bridge_maintenance_mode != "lazy_exact") {
            throw std::runtime_error(
                "bridge_maintenance_mode must be global_tarjan or lazy_exact."
            );
        }
        while (!selection_heap_.empty()) {
            const VersionedHeapEntry entry = selection_heap_.top();
            selection_heap_.pop();
            ++popped_count;
            ++heap_pop_count_total_;
            const int64_t edge_id = entry.key.edge_id;
            if (edge_id < 0 || edge_id >= original_edge_count_) {
                ++stale_popped;
                ++heap_stale_pop_count_total_;
                continue;
            }
            if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                ++inactive_popped;
                ++heap_inactive_pop_count_total_;
                continue;
            }
            if (guard_reason_[static_cast<size_t>(edge_id)] != 0) {
                ++guard_popped;
                ++heap_guard_pop_count_total_;
                continue;
            }
            if (heap_dirty_mask_[static_cast<size_t>(edge_id)] != 0) {
                ++dirty_popped;
                ++heap_dirty_pop_count_total_;
                continue;
            }
            if (entry.version != heap_versions_[static_cast<size_t>(edge_id)]) {
                ++stale_popped;
                ++heap_stale_pop_count_total_;
                continue;
            }
            if (
                heap_guard_bridges_ &&
                bridge_maintenance_mode == "lazy_exact"
            ) {
                const auto bridge_query_start = Clock::now();
                bool support_certified = false;
                const bool is_bridge = exact_bridge_query_internal(
                    edge_id,
                    support_certified,
                    lazy_bridge_nodes_visited,
                    lazy_bridge_adjacency_entries_visited,
                    lazy_bridge_inactive_entries_skipped
                );
                lazy_bridge_runtime_sec += std::chrono::duration<double>(
                    Clock::now() - bridge_query_start
                ).count();
                ++lazy_bridge_queries;
                if (support_certified) {
                    ++lazy_bridge_support_certificates;
                }
                if (is_bridge) {
                    mark_heap_guard_reason(edge_id, static_cast<uint8_t>(2));
                    ++lazy_bridges_rejected;
                    ++guard_popped;
                    ++heap_guard_pop_count_total_;
                    continue;
                }
            }
            selected_edge_id = edge_id;
            selected_key = entry.key;
            break;
        }
        const double runtime_sec = std::chrono::duration<double>(Clock::now() - start).count();
        py::dict result;
        result["selected_edge_id"] = selected_edge_id;
        result["best_score"] = selected_edge_id >= 0 ? selected_key.score : std::numeric_limits<double>::infinity();
        result["best_degree_score"] = selected_edge_id >= 0 ? selected_key.degree_score : std::numeric_limits<double>::infinity();
        result["best_support_score"] = selected_edge_id >= 0 ? selected_key.support_score : std::numeric_limits<double>::infinity();
        result["heap_pop_runtime_sec"] = runtime_sec;
        result["heap_entries_popped"] = popped_count;
        result["heap_stale_entries_popped"] = stale_popped;
        result["heap_inactive_entries_popped"] = inactive_popped;
        result["heap_guard_entries_popped"] = guard_popped;
        result["heap_dirty_entries_popped"] = dirty_popped;
        result["heap_size_after_pop"] = static_cast<int64_t>(selection_heap_.size());
        result["lazy_bridge_queries"] = lazy_bridge_queries;
        result["lazy_bridge_support_certificates"] = lazy_bridge_support_certificates;
        result["lazy_bridge_nodes_visited"] = lazy_bridge_nodes_visited;
        result["lazy_bridge_adjacency_entries_visited"] =
            lazy_bridge_adjacency_entries_visited;
        result["lazy_bridge_inactive_entries_skipped"] =
            lazy_bridge_inactive_entries_skipped;
        result["lazy_bridges_rejected"] = lazy_bridges_rejected;
        result["lazy_bridge_runtime_sec"] = lazy_bridge_runtime_sec;
        return result;
    }

    py::dict validate_heap_invariants() const {
        py::dict result;
        result["heap_storage_mode"] = heap_storage_mode_;
        result["valid"] = true;
        if (heap_storage_mode_ != "indexed") {
            result["checked_entries"] = static_cast<int64_t>(0);
            result["indexed_position_count"] = static_cast<int64_t>(0);
            return result;
        }
        if (
            indexed_heap_positions_.size() != static_cast<size_t>(original_edge_count_) ||
            indexed_heap_keys_.size() != static_cast<size_t>(original_edge_count_)
        ) {
            throw std::runtime_error("Indexed heap state arrays do not match original edge count.");
        }
        std::vector<uint8_t> seen(static_cast<size_t>(original_edge_count_), static_cast<uint8_t>(0));
        for (size_t position = 0; position < indexed_heap_edges_.size(); ++position) {
            const int64_t edge_id = indexed_heap_edges_[position];
            if (edge_id < 0 || edge_id >= original_edge_count_) {
                throw std::runtime_error("Indexed heap contains edge id outside valid range.");
            }
            if (seen[static_cast<size_t>(edge_id)] != 0) {
                throw std::runtime_error("Indexed heap contains a duplicate edge id.");
            }
            seen[static_cast<size_t>(edge_id)] = 1;
            if (indexed_heap_positions_[static_cast<size_t>(edge_id)] != static_cast<int64_t>(position)) {
                throw std::runtime_error("Indexed heap position map is inconsistent.");
            }
            if (indexed_heap_keys_[static_cast<size_t>(edge_id)].edge_id != edge_id) {
                throw std::runtime_error("Indexed heap key is associated with the wrong edge id.");
            }
            if (position > 0) {
                const size_t parent = (position - 1) / 2;
                if (indexed_heap_edge_less_internal(edge_id, indexed_heap_edges_[parent])) {
                    throw std::runtime_error("Indexed heap parent-child ordering invariant is violated.");
                }
            }
        }
        int64_t position_count = 0;
        for (int64_t edge_id = 0; edge_id < original_edge_count_; ++edge_id) {
            const int64_t position = indexed_heap_positions_[static_cast<size_t>(edge_id)];
            if (position < 0) {
                if (seen[static_cast<size_t>(edge_id)] != 0) {
                    throw std::runtime_error("Indexed heap seen-state disagrees with absent position.");
                }
                continue;
            }
            ++position_count;
            if (
                static_cast<size_t>(position) >= indexed_heap_edges_.size() ||
                indexed_heap_edges_[static_cast<size_t>(position)] != edge_id
            ) {
                throw std::runtime_error("Indexed heap reverse position lookup is inconsistent.");
            }
        }
        if (position_count != static_cast<int64_t>(indexed_heap_edges_.size())) {
            throw std::runtime_error("Indexed heap position count does not match heap size.");
        }
        result["checked_entries"] = static_cast<int64_t>(indexed_heap_edges_.size());
        result["indexed_position_count"] = position_count;
        return result;
    }

    py::dict versioned_heap_statistics() const {
        int64_t dirty_count = 0;
        for (const uint8_t value : heap_dirty_mask_) {
            dirty_count += value != 0 ? 1 : 0;
        }
        py::dict result;
        result["heap_initialized"] = heap_initialized_;
        result["heap_storage_mode"] = heap_storage_mode_;
        result["heap_size"] = selection_heap_size_internal();
        result["heap_max_size_observed"] = heap_max_size_observed_;
        result["heap_dirty_edge_count"] = dirty_count;
        result["eligible_active_count"] = eligible_active_count_;
        result["bridge_blocked_count"] = bridge_blocked_count_;
        result["d_min_blocked_count"] = d_min_blocked_count_;
        result["heap_push_count_total"] = heap_push_count_total_;
        result["heap_pop_count_total"] = heap_pop_count_total_;
        result["heap_stale_pop_count_total"] = heap_stale_pop_count_total_;
        result["heap_inactive_pop_count_total"] = heap_inactive_pop_count_total_;
        result["heap_guard_pop_count_total"] = heap_guard_pop_count_total_;
        result["heap_dirty_pop_count_total"] = heap_dirty_pop_count_total_;
        result["heap_rebuild_count_total"] = heap_rebuild_count_total_;
        result["heap_rebuild_edge_entries_scanned_total"] = heap_rebuild_edge_entries_scanned_total_;
        if (heap_storage_mode_ == "indexed") {
            result["heap_entry_size_bytes"] = static_cast<int64_t>(sizeof(int64_t));
            result["heap_current_estimated_bytes"] = static_cast<int64_t>(
                indexed_heap_edges_.capacity() * sizeof(int64_t) +
                indexed_heap_positions_.capacity() * sizeof(int64_t) +
                indexed_heap_keys_.capacity() * sizeof(EdgeKey)
            );
            result["heap_max_estimated_bytes"] = result["heap_current_estimated_bytes"];
        } else {
            result["heap_entry_size_bytes"] = static_cast<int64_t>(sizeof(VersionedHeapEntry));
            result["heap_current_estimated_bytes"] = static_cast<int64_t>(selection_heap_.size() * sizeof(VersionedHeapEntry));
            result["heap_max_estimated_bytes"] = static_cast<int64_t>(heap_max_size_observed_ * sizeof(VersionedHeapEntry));
        }
        result["heap_auxiliary_state_bytes"] = static_cast<int64_t>(
            heap_versions_.size() * sizeof(uint64_t) +
            heap_dirty_mask_.size() * sizeof(uint8_t) +
            guard_reason_.size() * sizeof(uint8_t) +
            heap_dirty_edge_ids_.capacity() * sizeof(int64_t)
        );
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
    ) {
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
            mark_heap_dirty(edge_id);
            if (valid_score_cache(edge_id) != 0) {
                valid_score_cache(edge_id) = 0;
                ++invalidated_count;
            }
        };

        touched_edge_ids.insert(selected_edge_id);
        if (selected_edge_id >= 0 && selected_edge_id < valid_score_cache.shape(0)) {
            valid_score_cache(selected_edge_id) = 0;
            if (
                relshift_state_initialized_ &&
                selected_edge_id < static_cast<int64_t>(endpoint_score_valid_mask_state_.size())
            ) {
                endpoint_score_valid_mask_state_[static_cast<size_t>(selected_edge_id)] = 0;
            }
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
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_delta_cache_array,
        const bool materialize_impacted_edges = true
    ) {
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

        if (graphlet_epoch_counter_ > std::numeric_limits<int>::max() - 4) {
            std::fill(graphlet_marks_workspace_.begin(), graphlet_marks_workspace_.end(), 0);
            graphlet_epoch_counter_ = 0;
        }
        auto& marks = graphlet_marks_workspace_;
        auto& attachment_masks = graphlet_attachment_workspace_;
        auto& directly_attached = selected_directly_attached_workspace_;
        auto& pair_masks = selected_pair_masks_workspace_;
        int& epoch = graphlet_epoch_counter_;
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
        collect_relevant_pair_masks_direct(
            row_ptr_.data(),
            col_idx_.data(),
            u,
            v,
            directly_attached,
            marks,
            attachment_masks,
            epoch,
            pair_masks,
            adjacency_edge_ids_.data(),
            active_edge_mask_.data()
        );
        collect_two_hop_nodes_sorted_into(
            row_ptr_.data(),
            col_idx_.data(),
            u,
            v,
            marks,
            epoch,
            selected_affected_nodes_workspace_,
            selected_frontier_workspace_,
            adjacency_edge_ids_.data(),
            active_edge_mask_.data()
        );
        auto& affected_nodes = selected_affected_nodes_workspace_;
        selected_four_node_pairs_total_ += static_cast<int64_t>(pair_masks.size());
        selected_four_node_pairs_peak_ = std::max<int64_t>(
            selected_four_node_pairs_peak_,
            static_cast<int64_t>(pair_masks.size())
        );
        selected_affected_nodes_peak_ = std::max<int64_t>(
            selected_affected_nodes_peak_,
            static_cast<int64_t>(affected_nodes.size())
        );

        ++node_workspace_epoch_counter_;
        if (node_workspace_epoch_counter_ == 0) {
            std::fill(node_workspace_epoch_.begin(), node_workspace_epoch_.end(), 0);
            node_workspace_epoch_counter_ = 1;
        }
        const uint32_t node_epoch = node_workspace_epoch_counter_;
        for (size_t idx = 0; idx < affected_nodes.size(); ++idx) {
            const int node = affected_nodes[idx];
            node_workspace_epoch_[static_cast<size_t>(node)] = node_epoch;
            node_workspace_index_[static_cast<size_t>(node)] = static_cast<int>(idx);
        }

        selected_raw_delta_workspace_.assign(
            affected_nodes.size() * static_cast<size_t>(kOrbitDim),
            0.0
        );
        double* raw_delta_ptr = selected_raw_delta_workspace_.data();

        int64_t invalidated_count = 0;
        int64_t corrected_edge_count = 0;
        int64_t full_rescore_edge_count = 0;
        ++edge_workspace_epoch_counter_;
        if (edge_workspace_epoch_counter_ == 0) {
            std::fill(edge_workspace_epoch_.begin(), edge_workspace_epoch_.end(), 0);
            edge_workspace_epoch_counter_ = 1;
        }
        const uint32_t edge_epoch = edge_workspace_epoch_counter_;
        auto& touched_edge_ids = selected_touched_edge_ids_workspace_;
        touched_edge_ids.clear();
        touched_edge_ids.reserve(directly_attached.size() * 4U + pair_masks.size() * 5U + 8U);
        auto& impacted_edge_codes = selected_impacted_edge_codes_workspace_;
        impacted_edge_codes.clear();
        if (materialize_impacted_edges) {
            impacted_edge_codes.reserve(pair_masks.size() * 5U + directly_attached.size() * 2U);
        }

        auto touch_edge_flags = [&](const int64_t edge_id) -> uint8_t& {
            if (edge_workspace_epoch_[static_cast<size_t>(edge_id)] != edge_epoch) {
                edge_workspace_epoch_[static_cast<size_t>(edge_id)] = edge_epoch;
                edge_workspace_flags_[static_cast<size_t>(edge_id)] = 0;
                touched_edge_ids.push_back(edge_id);
            }
            return edge_workspace_flags_[static_cast<size_t>(edge_id)];
        };
        auto mark_full_rescore = [&](const int64_t edge_id) {
            if (relshift_state_initialized_ && edge_id >= 0 && edge_id < original_edge_count_) {
                endpoint_score_valid_mask_state_[static_cast<size_t>(edge_id)] = 0;
            }
            uint8_t& flags = touch_edge_flags(edge_id);
            if ((flags & static_cast<uint8_t>(4)) == 0) {
                flags |= static_cast<uint8_t>(4);
                ++full_rescore_edge_count;
            }
        };
        auto mark_corrected = [&](const int64_t edge_id) {
            uint8_t& flags = touch_edge_flags(edge_id);
            if ((flags & static_cast<uint8_t>(2)) == 0) {
                flags |= static_cast<uint8_t>(2);
                ++corrected_edge_count;
            }
        };

        auto invalidate_score = [&](const int64_t edge_id, const uint8_t endpoint_mask) {
            if (edge_id < 0 || edge_id >= valid_score_cache.shape(0)) {
                return;
            }
            if (edge_id == selected_edge_id) {
                return;
            }
            if (
                relshift_state_initialized_ &&
                edge_id < static_cast<int64_t>(endpoint_score_valid_mask_state_.size())
            ) {
                endpoint_score_valid_mask_state_[static_cast<size_t>(edge_id)] =
                    static_cast<uint8_t>(
                        endpoint_score_valid_mask_state_[static_cast<size_t>(edge_id)]
                        & static_cast<uint8_t>(~endpoint_mask)
                    );
            }
            uint8_t& flags = touch_edge_flags(edge_id);
            if ((flags & static_cast<uint8_t>(1)) != 0) {
                return;
            }
            flags |= static_cast<uint8_t>(1);
            if (edge_id >= static_cast<int64_t>(active_edge_mask_.size()) || active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                return;
            }
            mark_heap_dirty(edge_id);
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
            if (materialize_impacted_edges) {
                impacted_edge_codes.push_back(code);
            }
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
            invalidate_score(edge_id, static_cast<uint8_t>(3));
            if (edge_id >= delta_valid_cache.shape(0) || edge_id >= candidate_delta_cache_array.shape(0)) {
                mark_full_rescore(edge_id);
                return;
            }
            if (delta_valid_cache(edge_id) == 0) {
                mark_full_rescore(edge_id);
                return;
            }
            const int left_slot = endpoint_slot(edge_id, left);
            const int right_slot = endpoint_slot(edge_id, right);
            if (left_slot < 0 || right_slot < 0) {
                mark_full_rescore(edge_id);
                delta_valid_cache(edge_id) = 0;
                return;
            }
            int64_t* edge_delta_base = candidate_delta_cache + edge_id * 2 * kOrbitDim;
            apply_omega_to_slot(size, before_mask, edge_bit, left_idx, edge_delta_base + left_slot * kOrbitDim);
            apply_omega_to_slot(size, before_mask, edge_bit, right_idx, edge_delta_base + right_slot * kOrbitDim);
            mark_corrected(edge_id);
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

        accumulate_full_delta_for_mask_dense(
            2, kUvEdgeBit, {u, v, -1, -1},
            node_workspace_epoch_, node_epoch, node_workspace_index_, raw_delta_ptr
        );
        for (const int node : directly_attached) {
            const int mask = mask_for_three_from_attachment(attachment_masks[node]);
            const std::array<int, 4> nodes{u, v, node, -1};
            accumulate_full_delta_for_mask_dense(
                3, mask, nodes,
                node_workspace_epoch_, node_epoch, node_workspace_index_, raw_delta_ptr
            );
            process_graphlet_edges(3, mask, nodes);
        }
        for (const PairMask& pair : pair_masks) {
            const std::array<int, 4> nodes{u, v, pair.first, pair.second};
            accumulate_full_delta_for_mask_dense(
                4, pair.mask, nodes,
                node_workspace_epoch_, node_epoch, node_workspace_index_, raw_delta_ptr
            );
            process_graphlet_edges(4, pair.mask, nodes);
        }

        if (selected_edge_id >= 0 && selected_edge_id < valid_score_cache.shape(0)) {
            valid_score_cache(selected_edge_id) = 0;
            if (
                relshift_state_initialized_ &&
                selected_edge_id < static_cast<int64_t>(endpoint_score_valid_mask_state_.size())
            ) {
                endpoint_score_valid_mask_state_[static_cast<size_t>(selected_edge_id)] = 0;
            }
        }
        if (selected_edge_id >= 0 && selected_edge_id < delta_valid_cache.shape(0)) {
            delta_valid_cache(selected_edge_id) = 0;
        }

        for (size_t node_idx = 0; node_idx < affected_nodes.size(); ++node_idx) {
            bool changed = false;
            const size_t delta_offset = node_idx * static_cast<size_t>(kOrbitDim);
            for (int orbit_idx = 0; orbit_idx < kOrbitDim; ++orbit_idx) {
                if (selected_raw_delta_workspace_[delta_offset + static_cast<size_t>(orbit_idx)] != 0.0) {
                    changed = true;
                    break;
                }
            }
            if (!changed) {
                continue;
            }
            const int node = affected_nodes[node_idx];
            if (node < 0 || node >= num_nodes_) {
                continue;
            }
            for (const int64_t edge_id : incident_edge_ids_[static_cast<size_t>(node)]) {
                const int slot = endpoint_slot(edge_id, node);
                const uint8_t endpoint_mask = slot < 0
                    ? static_cast<uint8_t>(3)
                    : static_cast<uint8_t>(1U << slot);
                invalidate_score(edge_id, endpoint_mask);
            }
        }

        const bool updating_native_delta_cache =
            native_delta_cache_enabled_ &&
            !candidate_delta_cache_state_.empty() &&
            candidate_delta_cache == candidate_delta_cache_state_.data();
        if (updating_native_delta_cache) {
            candidate_delta_nonzero_mask_state_[static_cast<size_t>(selected_edge_id) * 2] = 0;
            candidate_delta_nonzero_mask_state_[static_cast<size_t>(selected_edge_id) * 2 + 1] = 0;
            for (const int64_t edge_id : touched_edge_ids) {
                if (edge_id < 0 || edge_id >= original_edge_count_) {
                    continue;
                }
                const uint8_t flags = edge_workspace_flags_[static_cast<size_t>(edge_id)];
                const size_t mask_offset = static_cast<size_t>(edge_id) * 2;
                if (delta_valid_cache(edge_id) == 0 || (flags & static_cast<uint8_t>(4)) != 0) {
                    candidate_delta_nonzero_mask_state_[mask_offset] = 0;
                    candidate_delta_nonzero_mask_state_[mask_offset + 1] = 0;
                    continue;
                }
                if ((flags & static_cast<uint8_t>(2)) != 0) {
                    const int64_t* delta = candidate_delta_cache_state_.data()
                        + static_cast<size_t>(edge_id) * 2 * kOrbitDim;
                    candidate_delta_nonzero_mask_state_[mask_offset] =
                        endpoint_delta_nonzero_mask(delta);
                    candidate_delta_nonzero_mask_state_[mask_offset + 1] =
                        endpoint_delta_nonzero_mask(delta + kOrbitDim);
                }
            }
        }

        py::array_t<int64_t> impacted_edges_array;
        if (materialize_impacted_edges) {
            std::sort(impacted_edge_codes.begin(), impacted_edge_codes.end());
            impacted_edge_codes.erase(
                std::unique(impacted_edge_codes.begin(), impacted_edge_codes.end()),
                impacted_edge_codes.end()
            );
            impacted_edges_array = py::array_t<int64_t>(
                {static_cast<ssize_t>(impacted_edge_codes.size()), static_cast<ssize_t>(2)}
            );
            auto impacted_edges_view = impacted_edges_array.mutable_unchecked<2>();
            for (size_t idx = 0; idx < impacted_edge_codes.size(); ++idx) {
                const auto [left, right] = decode_pair(impacted_edge_codes[idx]);
                impacted_edges_view(static_cast<ssize_t>(idx), 0) = static_cast<int64_t>(left);
                impacted_edges_view(static_cast<ssize_t>(idx), 1) = static_cast<int64_t>(right);
            }
        } else {
            impacted_edges_array = py::array_t<int64_t>(py::array::ShapeContainer{0, 2});
        }

        py::array_t<int64_t> affected_nodes_array;
        py::array_t<double> raw_delta_array;
        if (materialize_impacted_edges) {
            affected_nodes_array = py::array_t<int64_t>(
                static_cast<ssize_t>(affected_nodes.size())
            );
            auto affected_nodes_view = affected_nodes_array.mutable_unchecked<1>();
            for (size_t idx = 0; idx < affected_nodes.size(); ++idx) {
                affected_nodes_view(static_cast<ssize_t>(idx)) =
                    static_cast<int64_t>(affected_nodes[idx]);
            }
            raw_delta_array = py::array_t<double>(py::array::ShapeContainer{
                static_cast<ssize_t>(affected_nodes.size()),
                static_cast<ssize_t>(kOrbitDim)
            });
            std::copy(
                selected_raw_delta_workspace_.begin(),
                selected_raw_delta_workspace_.end(),
                static_cast<double*>(raw_delta_array.mutable_data())
            );
        } else {
            affected_nodes_array = py::array_t<int64_t>(0);
            raw_delta_array = py::array_t<double>(py::array::ShapeContainer{0, kOrbitDim});
        }

        py::dict result;
        result["affected_nodes"] = std::move(affected_nodes_array);
        result["raw_delta"] = std::move(raw_delta_array);
        result["impacted_edges"] = std::move(impacted_edges_array);
        result["directly_attached_size"] = static_cast<int64_t>(directly_attached.size());
        result["four_node_pair_count"] = static_cast<int64_t>(pair_masks.size());
        result["invalidated_count"] = invalidated_count;
        result["cache_invalidated"] = true;
        result["mixed_correction_edge_count"] = corrected_edge_count;
        result["delta_impacted_full_rescore_count"] = full_rescore_edge_count;
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
        if (heap_initialized_) {
            const uint8_t reason = guard_reason_[static_cast<size_t>(edge_id)];
            if (reason == 0) {
                --eligible_active_count_;
            } else if (reason == 1) {
                --d_min_blocked_count_;
            } else if (reason == 2) {
                --bridge_blocked_count_;
            }
            ++heap_versions_[static_cast<size_t>(edge_id)];
            heap_dirty_mask_[static_cast<size_t>(edge_id)] = 0;
            if (heap_storage_mode_ == "indexed") {
                indexed_heap_remove_internal(edge_id);
            }
        }
        active_edge_mask_[static_cast<size_t>(edge_id)] = 0;
        removed_edge_ids_.push_back(edge_id);
        const auto& edge = edge_by_id_[static_cast<size_t>(edge_id)];
        if (current_degrees_[static_cast<size_t>(edge[0])] <= 0 || current_degrees_[static_cast<size_t>(edge[1])] <= 0) {
            throw std::runtime_error("NativeGraphState degree underflow during logical edge removal.");
        }
        --current_degrees_[static_cast<size_t>(edge[0])];
        --current_degrees_[static_cast<size_t>(edge[1])];
        --active_edge_count_;
        if (heap_initialized_) {
            for (const int node : {edge[0], edge[1]}) {
                if (current_degrees_[static_cast<size_t>(node)] <= heap_d_min_) {
                    for (const int64_t incident_edge_id : incident_edge_ids_[static_cast<size_t>(node)]) {
                        mark_heap_guard_reason(incident_edge_id, static_cast<uint8_t>(1));
                    }
                }
            }
        }
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
        result["adjacency_compaction_count"] = adjacency_compaction_count_;
        result["adjacency_compaction_entries_copied_total"] =
            adjacency_compaction_entries_copied_total_;
        result["adjacency_compaction_runtime_sec_total"] =
            adjacency_compaction_runtime_sec_total_;
        return result;
    }

    int num_nodes() const {
        return num_nodes_;
    }

private:
    void require_relshift_state_internal() const {
        if (!relshift_state_initialized_) {
            throw std::runtime_error("Native RelShift numerical state has not been initialized.");
        }
    }

    py::dict maybe_compact_adjacency_internal(const double threshold) {
        using Clock = std::chrono::high_resolution_clock;
        const int64_t before_entries = static_cast<int64_t>(col_idx_.size());
        const int64_t active_entries = active_edge_count_ * 2;
        const double tombstone_ratio = before_entries == 0
            ? 0.0
            : static_cast<double>(before_entries - active_entries)
                / static_cast<double>(before_entries);
        py::dict result;
        result["adjacency_compacted"] = false;
        result["adjacency_compaction_runtime_sec"] = 0.0;
        result["adjacency_entries_before_compaction"] = before_entries;
        result["adjacency_entries_after_compaction"] = before_entries;
        if (threshold <= 0.0 || tombstone_ratio < threshold) {
            return result;
        }

        const auto start = Clock::now();
        std::vector<std::vector<std::pair<int, int64_t>>> rows(
            static_cast<size_t>(num_nodes_)
        );
        for (int64_t edge_id = 0; edge_id < original_edge_count_; ++edge_id) {
            if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
                continue;
            }
            const auto& edge = edge_by_id_[static_cast<size_t>(edge_id)];
            rows[static_cast<size_t>(edge[0])].push_back({edge[1], edge_id});
            rows[static_cast<size_t>(edge[1])].push_back({edge[0], edge_id});
        }

        std::vector<int64_t> new_row_ptr(static_cast<size_t>(num_nodes_ + 1), 0);
        for (int node = 0; node < num_nodes_; ++node) {
            auto& row = rows[static_cast<size_t>(node)];
            std::sort(row.begin(), row.end(), [](const auto& left, const auto& right) {
                if (left.first != right.first) {
                    return left.first < right.first;
                }
                return left.second < right.second;
            });
            new_row_ptr[static_cast<size_t>(node + 1)] =
                new_row_ptr[static_cast<size_t>(node)] + static_cast<int64_t>(row.size());
        }
        std::vector<int64_t> new_col_idx(static_cast<size_t>(active_entries), 0);
        std::vector<int64_t> new_adjacency_edge_ids(static_cast<size_t>(active_entries), -1);
        for (int node = 0; node < num_nodes_; ++node) {
            int64_t offset = new_row_ptr[static_cast<size_t>(node)];
            for (const auto& [neighbor, edge_id] : rows[static_cast<size_t>(node)]) {
                new_col_idx[static_cast<size_t>(offset)] = neighbor;
                new_adjacency_edge_ids[static_cast<size_t>(offset)] = edge_id;
                ++offset;
            }
        }
        row_ptr_.swap(new_row_ptr);
        col_idx_.swap(new_col_idx);
        adjacency_edge_ids_.swap(new_adjacency_edge_ids);
        ++adjacency_compaction_count_;
        adjacency_compaction_entries_copied_total_ += active_entries;
        const double runtime = std::chrono::duration<double>(Clock::now() - start).count();
        adjacency_compaction_runtime_sec_total_ += runtime;
        result["adjacency_compacted"] = true;
        result["adjacency_compaction_runtime_sec"] = runtime;
        result["adjacency_entries_after_compaction"] = active_entries;
        return result;
    }

    int support_for_edge_internal(const int64_t edge_id) {
        if (edge_id < 0 || edge_id >= original_edge_count_) {
            throw std::runtime_error("support edge id is out of range.");
        }
        if (support_initialized_state_[static_cast<size_t>(edge_id)] == 0) {
            const auto& edge = edge_by_id_[static_cast<size_t>(edge_id)];
            support_score_cache_state_[static_cast<size_t>(edge_id)] = static_cast<double>(edge_support(
                row_ptr_.data(),
                col_idx_.data(),
                edge[0],
                edge[1],
                adjacency_edge_ids_.data(),
                active_edge_mask_.data()
            ));
            support_initialized_state_[static_cast<size_t>(edge_id)] = 1;
            ++native_state_support_initializations_;
        }
        return static_cast<int>(std::llround(support_score_cache_state_[static_cast<size_t>(edge_id)]));
    }

    bool exact_bridge_query_internal(
        const int64_t edge_id,
        bool& support_certified,
        int64_t& nodes_visited,
        int64_t& adjacency_entries_visited,
        int64_t& inactive_entries_skipped
    ) {
        if (edge_id < 0 || edge_id >= original_edge_count_) {
            throw std::runtime_error("Bridge query edge id is out of range.");
        }
        if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
            throw std::runtime_error("Bridge query requires an active edge.");
        }
        support_certified = false;
        if (support_for_edge_internal(edge_id) > 0) {
            // An edge in a triangle has the other two triangle edges as an
            // explicit alternate path, so it cannot be a bridge.
            support_certified = true;
            return false;
        }
        const auto& edge = edge_by_id_[static_cast<size_t>(edge_id)];
        const int source = edge[0];
        const int target = edge[1];
        if (
            current_degrees_[static_cast<size_t>(source)] <= 1 ||
            current_degrees_[static_cast<size_t>(target)] <= 1
        ) {
            return true;
        }
        ++bridge_query_epoch_counter_;
        if (bridge_query_epoch_counter_ == 0) {
            std::fill(
                bridge_query_visited_epoch_.begin(),
                bridge_query_visited_epoch_.end(),
                static_cast<uint32_t>(0)
            );
            bridge_query_epoch_counter_ = 1;
        }
        const uint32_t epoch = bridge_query_epoch_counter_;
        bridge_query_frontier_.clear();
        bridge_query_frontier_.push_back(source);
        bridge_query_visited_epoch_[static_cast<size_t>(source)] = epoch;
        size_t cursor = 0;
        while (cursor < bridge_query_frontier_.size()) {
            const int node = bridge_query_frontier_[cursor++];
            ++nodes_visited;
            for (
                int64_t idx = row_ptr_[static_cast<size_t>(node)];
                idx < row_ptr_[static_cast<size_t>(node + 1)];
                ++idx
            ) {
                ++adjacency_entries_visited;
                const int64_t candidate_edge_id =
                    adjacency_edge_ids_[static_cast<size_t>(idx)];
                if (
                    candidate_edge_id < 0 ||
                    active_edge_mask_[static_cast<size_t>(candidate_edge_id)] == 0
                ) {
                    ++inactive_entries_skipped;
                    continue;
                }
                if (candidate_edge_id == edge_id) {
                    continue;
                }
                const int neighbor = static_cast<int>(
                    col_idx_[static_cast<size_t>(idx)]
                );
                if (neighbor == target) {
                    return false;
                }
                if (
                    bridge_query_visited_epoch_[static_cast<size_t>(neighbor)] == epoch
                ) {
                    continue;
                }
                bridge_query_visited_epoch_[static_cast<size_t>(neighbor)] = epoch;
                bridge_query_frontier_.push_back(neighbor);
            }
        }
        return true;
    }

    void configure_heap_storage_mode_internal(const std::string& mode) {
        if (mode != "versioned" && mode != "indexed") {
            throw std::runtime_error("heap_storage_mode must be versioned or indexed.");
        }
        if (!heap_storage_mode_initialized_) {
            heap_storage_mode_ = mode;
            heap_storage_mode_initialized_ = true;
            if (mode == "indexed") {
                indexed_heap_edges_.clear();
                indexed_heap_edges_.reserve(static_cast<size_t>(original_edge_count_));
                indexed_heap_positions_.assign(
                    static_cast<size_t>(original_edge_count_), static_cast<int64_t>(-1)
                );
                indexed_heap_keys_.assign(
                    static_cast<size_t>(original_edge_count_),
                    EdgeKey{
                        std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity(),
                        -1
                    }
                );
            }
            return;
        }
        if (heap_storage_mode_ != mode) {
            throw std::runtime_error("Heap storage mode cannot change during pruning.");
        }
    }

    int64_t selection_heap_size_internal() const {
        return heap_storage_mode_ == "indexed"
            ? static_cast<int64_t>(indexed_heap_edges_.size())
            : static_cast<int64_t>(selection_heap_.size());
    }

    EdgeKey current_native_edge_key_internal(const int64_t edge_id) const {
        if (edge_id < 0 || edge_id >= original_edge_count_) {
            throw std::runtime_error("Edge id outside range while constructing indexed heap key.");
        }
        return EdgeKey{
            score_cache_state_[static_cast<size_t>(edge_id)],
            degree_score_cache_state_[static_cast<size_t>(edge_id)],
            support_score_cache_state_[static_cast<size_t>(edge_id)],
            edge_id,
        };
    }

    bool indexed_heap_edge_less_internal(const int64_t left_edge_id, const int64_t right_edge_id) const {
        return edge_key_less(
            indexed_heap_keys_[static_cast<size_t>(left_edge_id)],
            indexed_heap_keys_[static_cast<size_t>(right_edge_id)]
        );
    }

    void indexed_heap_swap_internal(const size_t left, const size_t right) {
        if (left == right) {
            return;
        }
        std::swap(indexed_heap_edges_[left], indexed_heap_edges_[right]);
        indexed_heap_positions_[static_cast<size_t>(indexed_heap_edges_[left])] = static_cast<int64_t>(left);
        indexed_heap_positions_[static_cast<size_t>(indexed_heap_edges_[right])] = static_cast<int64_t>(right);
    }

    size_t indexed_heap_sift_up_internal(size_t position) {
        while (position > 0) {
            const size_t parent = (position - 1) / 2;
            if (!indexed_heap_edge_less_internal(indexed_heap_edges_[position], indexed_heap_edges_[parent])) {
                break;
            }
            indexed_heap_swap_internal(position, parent);
            position = parent;
        }
        return position;
    }

    size_t indexed_heap_sift_down_internal(size_t position) {
        const size_t heap_size = indexed_heap_edges_.size();
        while (true) {
            const size_t left = position * 2 + 1;
            if (left >= heap_size) {
                break;
            }
            const size_t right = left + 1;
            size_t best = left;
            if (
                right < heap_size &&
                indexed_heap_edge_less_internal(indexed_heap_edges_[right], indexed_heap_edges_[left])
            ) {
                best = right;
            }
            if (!indexed_heap_edge_less_internal(indexed_heap_edges_[best], indexed_heap_edges_[position])) {
                break;
            }
            indexed_heap_swap_internal(position, best);
            position = best;
        }
        return position;
    }

    bool indexed_heap_remove_internal(const int64_t edge_id) {
        if (edge_id < 0 || edge_id >= original_edge_count_) {
            return false;
        }
        const int64_t stored_position = indexed_heap_positions_[static_cast<size_t>(edge_id)];
        if (stored_position < 0) {
            return false;
        }
        const size_t position = static_cast<size_t>(stored_position);
        const size_t last = indexed_heap_edges_.size() - 1;
        if (position != last) {
            indexed_heap_swap_internal(position, last);
        }
        indexed_heap_edges_.pop_back();
        indexed_heap_positions_[static_cast<size_t>(edge_id)] = -1;
        if (position < indexed_heap_edges_.size()) {
            const size_t moved = indexed_heap_sift_up_internal(position);
            indexed_heap_sift_down_internal(moved);
        }
        return true;
    }

    bool indexed_heap_insert_or_update_internal(const int64_t edge_id) {
        if (edge_id < 0 || edge_id >= original_edge_count_) {
            throw std::runtime_error("Indexed heap update received edge id outside range.");
        }
        if (
            active_edge_mask_[static_cast<size_t>(edge_id)] == 0 ||
            guard_reason_[static_cast<size_t>(edge_id)] != 0 ||
            heap_dirty_mask_[static_cast<size_t>(edge_id)] != 0 ||
            valid_score_cache_state_[static_cast<size_t>(edge_id)] == 0
        ) {
            indexed_heap_remove_internal(edge_id);
            return false;
        }
        indexed_heap_keys_[static_cast<size_t>(edge_id)] = current_native_edge_key_internal(edge_id);
        int64_t stored_position = indexed_heap_positions_[static_cast<size_t>(edge_id)];
        if (stored_position < 0) {
            const size_t position = indexed_heap_edges_.size();
            indexed_heap_edges_.push_back(edge_id);
            indexed_heap_positions_[static_cast<size_t>(edge_id)] = static_cast<int64_t>(position);
            indexed_heap_sift_up_internal(position);
        } else {
            const size_t position = static_cast<size_t>(stored_position);
            const size_t moved = indexed_heap_sift_up_internal(position);
            indexed_heap_sift_down_internal(moved);
        }
        heap_max_size_observed_ = std::max<int64_t>(
            heap_max_size_observed_,
            static_cast<int64_t>(indexed_heap_edges_.size())
        );
        return true;
    }

    void rebuild_indexed_heap_internal() {
        indexed_heap_edges_.clear();
        std::fill(indexed_heap_positions_.begin(), indexed_heap_positions_.end(), static_cast<int64_t>(-1));
        indexed_heap_edges_.reserve(static_cast<size_t>(std::max<int64_t>(eligible_active_count_, 0)));
        heap_rebuild_edge_entries_scanned_total_ += original_edge_count_;
        for (int64_t edge_id = 0; edge_id < original_edge_count_; ++edge_id) {
            if (
                active_edge_mask_[static_cast<size_t>(edge_id)] == 0 ||
                guard_reason_[static_cast<size_t>(edge_id)] != 0 ||
                heap_dirty_mask_[static_cast<size_t>(edge_id)] != 0 ||
                valid_score_cache_state_[static_cast<size_t>(edge_id)] == 0
            ) {
                continue;
            }
            indexed_heap_keys_[static_cast<size_t>(edge_id)] = current_native_edge_key_internal(edge_id);
            indexed_heap_positions_[static_cast<size_t>(edge_id)] = static_cast<int64_t>(indexed_heap_edges_.size());
            indexed_heap_edges_.push_back(edge_id);
        }
        if (!indexed_heap_edges_.empty()) {
            for (size_t position = indexed_heap_edges_.size() / 2; position > 0; --position) {
                indexed_heap_sift_down_internal(position - 1);
            }
        }
        heap_push_count_total_ += static_cast<int64_t>(indexed_heap_edges_.size());
        ++heap_rebuild_count_total_;
        heap_max_size_observed_ = std::max<int64_t>(
            heap_max_size_observed_,
            static_cast<int64_t>(indexed_heap_edges_.size())
        );
    }

    void mark_heap_dirty(const int64_t edge_id) {
        if (!heap_initialized_) {
            return;
        }
        if (edge_id < 0 || edge_id >= original_edge_count_) {
            return;
        }
        if (
            active_edge_mask_[static_cast<size_t>(edge_id)] == 0 ||
            guard_reason_[static_cast<size_t>(edge_id)] != 0
        ) {
            return;
        }
        if (heap_dirty_mask_[static_cast<size_t>(edge_id)] != 0) {
            return;
        }
        heap_dirty_mask_[static_cast<size_t>(edge_id)] = 1;
        heap_dirty_edge_ids_.push_back(edge_id);
        ++heap_versions_[static_cast<size_t>(edge_id)];
    }

    void mark_heap_guard_reason(const int64_t edge_id, const uint8_t reason) {
        if (edge_id < 0 || edge_id >= original_edge_count_ || reason == 0) {
            return;
        }
        if (active_edge_mask_[static_cast<size_t>(edge_id)] == 0) {
            return;
        }
        uint8_t& current_reason = guard_reason_[static_cast<size_t>(edge_id)];
        if (reason == 1) {
            if (current_reason != 0) {
                return;
            }
            current_reason = 1;
            ++d_min_blocked_count_;
            --eligible_active_count_;
        } else if (reason == 2) {
            if (current_reason == 2) {
                return;
            }
            if (current_reason == 1) {
                --d_min_blocked_count_;
            } else {
                --eligible_active_count_;
            }
            current_reason = 2;
            ++bridge_blocked_count_;
        } else {
            throw std::runtime_error("Unknown heap guard reason.");
        }
        if (heap_storage_mode_ == "indexed") {
            indexed_heap_remove_internal(edge_id);
        }
        ++heap_versions_[static_cast<size_t>(edge_id)];
    }

    template <typename ScoreView, typename DegreeView, typename SupportView, typename ValidView>
    void rebuild_selection_heap(
        const ScoreView& score_cache,
        const DegreeView& degree_score_cache,
        const SupportView& support_score_cache,
        const ValidView& valid_score_cache
    ) {
        decltype(selection_heap_) rebuilt;
        heap_rebuild_edge_entries_scanned_total_ += original_edge_count_;
        for (int64_t edge_id = 0; edge_id < original_edge_count_; ++edge_id) {
            if (
                active_edge_mask_[static_cast<size_t>(edge_id)] == 0 ||
                guard_reason_[static_cast<size_t>(edge_id)] != 0 ||
                heap_dirty_mask_[static_cast<size_t>(edge_id)] != 0 ||
                valid_score_cache(edge_id) == 0
            ) {
                continue;
            }
            const EdgeKey key{
                score_cache(edge_id),
                degree_score_cache(edge_id),
                support_score_cache(edge_id),
                edge_id,
            };
            rebuilt.push(VersionedHeapEntry{key, heap_versions_[static_cast<size_t>(edge_id)]});
            ++heap_push_count_total_;
        }
        selection_heap_.swap(rebuilt);
        ++heap_rebuild_count_total_;
        heap_max_size_observed_ = std::max<int64_t>(
            heap_max_size_observed_,
            static_cast<int64_t>(selection_heap_.size())
        );
    }

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
    std::vector<std::array<int, 2>> edge_output_by_id_;
    std::vector<int64_t> removed_edge_ids_;
    std::unordered_map<uint64_t, int64_t> edge_code_to_id_;
    std::vector<std::vector<int64_t>> incident_edge_ids_;
    std::vector<uint8_t> active_edge_mask_;
    // Reusable dense workspaces replace per-round unordered_map/set allocations.
    std::vector<uint32_t> node_workspace_epoch_;
    std::vector<int> node_workspace_index_;
    std::vector<int> graphlet_marks_workspace_;
    std::vector<int> graphlet_attachment_workspace_;
    int graphlet_epoch_counter_ = 0;
    std::vector<uint32_t> bridge_query_visited_epoch_;
    std::vector<int> bridge_query_frontier_;
    uint32_t bridge_query_epoch_counter_ = 0;
    std::vector<int> selected_directly_attached_workspace_;
    std::vector<PairMask> selected_pair_masks_workspace_;
    std::vector<int> selected_affected_nodes_workspace_;
    std::vector<int> selected_frontier_workspace_;
    std::vector<double> selected_raw_delta_workspace_;
    std::vector<int64_t> selected_touched_edge_ids_workspace_;
    std::vector<uint64_t> selected_impacted_edge_codes_workspace_;
    uint32_t node_workspace_epoch_counter_ = 0;
    std::vector<uint32_t> edge_workspace_epoch_;
    std::vector<uint8_t> edge_workspace_flags_;
    uint32_t edge_workspace_epoch_counter_ = 0;

    // Phase 1 Step 3.5A/B: native-owned RelShift numerical state.
    bool relshift_state_initialized_ = false;
    bool native_delta_cache_enabled_ = false;
    std::string native_score_mode_ = "relative";
    double native_eps_ = 1e-8;
    std::vector<double> current_raw_state_;
    std::vector<double> current_std_state_;
    std::vector<double> stats_mean_state_;
    std::vector<double> stats_std_state_;
    std::vector<double> node_denominator_state_;
    std::vector<double> score_cache_state_;
    std::vector<double> endpoint_score_cache_state_;
    std::vector<uint8_t> endpoint_score_valid_mask_state_;
    std::vector<double> degree_score_cache_state_;
    std::vector<double> support_score_cache_state_;
    std::vector<uint8_t> support_initialized_state_;
    std::vector<uint8_t> valid_score_cache_state_;
    std::vector<int64_t> candidate_delta_cache_state_;
    std::vector<uint16_t> candidate_delta_nonzero_mask_state_;
    std::vector<uint8_t> delta_valid_cache_state_;
    int64_t native_state_round_count_ = 0;
    int64_t native_state_support_initializations_ = 0;
    int64_t native_state_support_decrements_ = 0;
    int64_t native_state_node_rows_updated_ = 0;
    int64_t endpoint_score_recomputed_total_ = 0;
    int64_t endpoint_score_reused_total_ = 0;
    int64_t selected_four_node_pairs_total_ = 0;
    int64_t selected_four_node_pairs_peak_ = 0;
    int64_t selected_affected_nodes_peak_ = 0;
    int64_t adjacency_compaction_count_ = 0;
    int64_t adjacency_compaction_entries_copied_total_ = 0;
    double adjacency_compaction_runtime_sec_total_ = 0.0;

    std::priority_queue<VersionedHeapEntry, std::vector<VersionedHeapEntry>, VersionedHeapGreater> selection_heap_;
    // Exact O(m)-memory alternative to duplicate-entry versioned storage.
    std::vector<int64_t> indexed_heap_edges_;
    std::vector<int64_t> indexed_heap_positions_;
    std::vector<EdgeKey> indexed_heap_keys_;
    std::string heap_storage_mode_ = "versioned";
    bool heap_storage_mode_initialized_ = false;
    std::vector<uint64_t> heap_versions_;
    std::vector<uint8_t> heap_dirty_mask_;
    std::vector<int64_t> heap_dirty_edge_ids_;
    // Guard reason priority matches the reference scan: bridge before d_min.
    // 0 = eligible/unblocked, 1 = d_min blocked, 2 = bridge blocked.
    std::vector<uint8_t> guard_reason_;
    bool heap_initialized_ = false;
    bool heap_guard_configuration_initialized_ = false;
    int heap_d_min_ = -1;
    bool heap_guard_bridges_ = false;
    std::string heap_bridge_maintenance_mode_ = "global_tarjan";
    int64_t eligible_active_count_ = 0;
    int64_t bridge_blocked_count_ = 0;
    int64_t d_min_blocked_count_ = 0;
    int64_t heap_push_count_total_ = 0;
    int64_t heap_pop_count_total_ = 0;
    int64_t heap_stale_pop_count_total_ = 0;
    int64_t heap_inactive_pop_count_total_ = 0;
    int64_t heap_guard_pop_count_total_ = 0;
    int64_t heap_dirty_pop_count_total_ = 0;
    int64_t heap_rebuild_count_total_ = 0;
    int64_t heap_rebuild_edge_entries_scanned_total_ = 0;
    int64_t heap_max_size_observed_ = 0;
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
    result["registry_version"] = kOrbitRegistryVersion;
    result["orbit_dim"] = static_cast<int64_t>(kOrbitDim);
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
        .def("initialize_relshift_state", &NativeGraphState::initialize_relshift_state, py::arg("current_raw"), py::arg("stats_mean"), py::arg("stats_std"), py::arg("score_mode"), py::arg("eps"), py::arg("enable_candidate_delta_cache") = true)
        .def("relshift_state_initialized", &NativeGraphState::relshift_state_initialized)
        .def("prepare_versioned_heap_round_fused", &NativeGraphState::prepare_versioned_heap_round_fused, py::arg("d_min"), py::arg("guard_bridges"), py::arg("bridge_maintenance_mode") = "global_tarjan")
        .def("score_edge_ids_round_best_fused", &NativeGraphState::score_edge_ids_round_best_fused, py::arg("candidate_edge_ids"), py::arg("kernel_variant") = "mask_count_v4_combinatorial", py::arg("profile_native_kernel") = false)
        .def("refresh_scores_from_delta_cache_fused", &NativeGraphState::refresh_scores_from_delta_cache_fused, py::arg("candidate_edge_ids"))
        .def("commit_dirty_heap_keys_fused", &NativeGraphState::commit_dirty_heap_keys_fused, py::arg("rebuild_ratio") = 4.0)
        .def("select_best_edge_fused", &NativeGraphState::select_best_edge_fused, py::arg("d_min"), py::arg("guard_bridges"), py::arg("kernel_variant") = "mask_count_v4_combinatorial", py::arg("profile_native_kernel") = false, py::arg("rebuild_ratio") = 4.0, py::arg("bridge_maintenance_mode") = "global_tarjan", py::arg("heap_storage_mode") = "versioned")
        .def("apply_selected_edge_and_remove_fused", &NativeGraphState::apply_selected_edge_and_remove_fused, py::arg("selected_edge_id"), py::arg("adjacency_compaction_threshold") = 0.20)
        .def("candidate_delta_cache_snapshot", &NativeGraphState::candidate_delta_cache_snapshot)
        .def("delta_valid_cache_snapshot", &NativeGraphState::delta_valid_cache_snapshot)
        .def("score_cache_snapshot", &NativeGraphState::score_cache_snapshot)
        .def("valid_score_cache_snapshot", &NativeGraphState::valid_score_cache_snapshot)
        .def("current_raw_snapshot", &NativeGraphState::current_raw_snapshot)
        .def("current_std_snapshot", &NativeGraphState::current_std_snapshot)
        .def("active_edge_mask_snapshot", &NativeGraphState::active_edge_mask_snapshot)
        .def("active_edges_snapshot", &NativeGraphState::active_edges_snapshot)
        .def("removed_edges_snapshot", &NativeGraphState::removed_edges_snapshot)
        .def("current_degrees_snapshot", &NativeGraphState::current_degrees_snapshot)
        .def("edge_support_fused", &NativeGraphState::edge_support_fused, py::arg("edge_id"))
        .def("relshift_state_statistics", &NativeGraphState::relshift_state_statistics)
        .def("eligible_edge_id_partitions", &NativeGraphState::eligible_edge_id_partitions, py::arg("active_edge_ids"), py::arg("edge_array_by_id"), py::arg("degrees"), py::arg("d_min"), py::arg("guard_bridges"), py::arg("valid_score_cache"), py::arg("use_score_cache"))
        .def("eligible_edge_id_partitions_with_cached_best", &NativeGraphState::eligible_edge_id_partitions_with_cached_best, py::arg("active_edge_ids"), py::arg("degrees"), py::arg("d_min"), py::arg("guard_bridges"), py::arg("valid_score_cache"), py::arg("use_score_cache"), py::arg("score_cache"), py::arg("degree_score_cache"), py::arg("support_score_cache"), py::arg("delta_valid_cache") = py::none())
        .def("prepare_versioned_heap_round", &NativeGraphState::prepare_versioned_heap_round, py::arg("d_min"), py::arg("guard_bridges"), py::arg("valid_score_cache"), py::arg("delta_valid_cache") = py::none(), py::arg("bridge_maintenance_mode") = "global_tarjan")
        .def("commit_dirty_heap_keys", &NativeGraphState::commit_dirty_heap_keys, py::arg("score_cache"), py::arg("degree_score_cache"), py::arg("support_score_cache"), py::arg("valid_score_cache"), py::arg("rebuild_ratio") = 4.0)
        .def("pop_best_versioned_heap", &NativeGraphState::pop_best_versioned_heap, py::arg("bridge_maintenance_mode") = "global_tarjan")
        .def("validate_heap_invariants", &NativeGraphState::validate_heap_invariants)
        .def("versioned_heap_statistics", &NativeGraphState::versioned_heap_statistics)
        .def("score_edges_round_best", &NativeGraphState::score_edges_round_best_state, py::arg("candidate_edges"), py::arg("candidate_edge_ids"), py::arg("current_raw"), py::arg("current_std"), py::arg("stats_mean"), py::arg("stats_std"), py::arg("score_mode"), py::arg("eps"), py::arg("score_cache"), py::arg("degree_score_cache"), py::arg("support_score_cache"), py::arg("valid_score_cache"), py::arg("kernel_variant") = "mask_count_v4_combinatorial", py::arg("profile_native_kernel") = false, py::arg("candidate_delta_cache") = py::none(), py::arg("delta_valid_cache") = py::none())
        .def("score_edge_ids_round_best", &NativeGraphState::score_edge_ids_round_best, py::arg("candidate_edge_ids"), py::arg("current_raw"), py::arg("current_std"), py::arg("stats_mean"), py::arg("stats_std"), py::arg("score_mode"), py::arg("eps"), py::arg("score_cache"), py::arg("degree_score_cache"), py::arg("support_score_cache"), py::arg("valid_score_cache"), py::arg("kernel_variant") = "mask_count_v4_combinatorial", py::arg("profile_native_kernel") = false, py::arg("candidate_delta_cache") = py::none(), py::arg("delta_valid_cache") = py::none())
        .def("refresh_scores_from_delta_cache", &NativeGraphState::refresh_scores_from_delta_cache, py::arg("candidate_edge_ids"), py::arg("current_raw"), py::arg("current_std"), py::arg("stats_mean"), py::arg("stats_std"), py::arg("score_mode"), py::arg("eps"), py::arg("candidate_delta_cache"), py::arg("delta_valid_cache"), py::arg("score_cache"), py::arg("degree_score_cache"), py::arg("support_score_cache"), py::arg("valid_score_cache"))
        .def("compute_selected_edge_delta", &NativeGraphState::compute_selected_edge_delta_state, py::arg("u"), py::arg("v"))
        .def("compute_selected_edge_delta_and_invalidate", &NativeGraphState::compute_selected_edge_delta_and_invalidate, py::arg("selected_edge_id"), py::arg("valid_score_cache"))
        .def("compute_selected_edge_delta_and_update_candidate_cache", &NativeGraphState::compute_selected_edge_delta_and_update_candidate_cache, py::arg("selected_edge_id"), py::arg("valid_score_cache"), py::arg("delta_valid_cache"), py::arg("candidate_delta_cache"), py::arg("materialize_impacted_edges") = true)
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
