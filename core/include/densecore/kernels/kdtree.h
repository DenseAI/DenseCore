/**
 * @file kdtree.h
 * @brief Lightweight header-only KD-tree for 3D kNN queries
 *
 * O(N log N) build, O(k log N) query.
 * Designed for Point Attention in Transformers (PointNet++, AlphaFold).
 */

#ifndef DENSECORE_KERNELS_KDTREE_H
#define DENSECORE_KERNELS_KDTREE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

namespace densecore {
namespace kernels {

class KDTree3D {
public:
    /**
     * @brief Construct a KD-tree from 3D points
     * @param positions Flattened array of points [N, 3] (x, y, z)
     * @param N Number of points
     */
    KDTree3D(const float* positions, int64_t N) : points_(positions) {
        if (N == 0) return;

        // Initialize indices [0, 1, ..., N-1]
        std::vector<int32_t> indices(N);
        for (int64_t i = 0; i < N; ++i) {
            indices[i] = static_cast<int32_t>(i);
        }

        // Reserve memory for nodes to avoid reallocations
        // A balanced tree has N nodes.
        nodes_.reserve(N);

        BuildRecursive(indices, 0, N, 0);
    }

    /**
     * @brief Find k-Nearest Neighbors
     * @param query Query point [3] (x, y, z)
     * @param k Number of neighbors to find
     * @param out_indices Output buffer for indices [k]
     * @param out_dists Output buffer for distances [k]
     */
    void KNN(const float* query, int k, int32_t* out_indices, float* out_dists) const {
        if (nodes_.empty() || k <= 0) return;

        // Max-heap to store k closest neighbors: (distance_sq, index)
        // We want to keep the smallest distances. Priority queue pops the largest.
        // So we maintain size k. If new point is closer than top(), pop top and push new.
        std::priority_queue<std::pair<float, int32_t>> pq;

        KNNRecursive(0, query, k, pq);

        // Extract from PQ to output arrays
        // PQ returns largest distance first, so we fill from back to front
        // to get ascending order (closest first).
        int idx = k - 1;
        while (!pq.empty() && idx >= 0) {
            out_indices[idx] = pq.top().second;
            out_dists[idx] = std::sqrt(pq.top().first);  // Convert dist_sq to dist
            pq.pop();
            idx--;
        }
    }

private:
    struct Node {
        int32_t point_idx;   // Index into original points_ array
        int8_t split_axis;   // 0=x, 1=y, 2=z
        int32_t left = -1;   // Index into nodes_ vector
        int32_t right = -1;  // Index into nodes_ vector
    };

    std::vector<Node> nodes_;
    const float* points_;


    /**
     * @brief Recursive builder
     * @return Index of the created node in nodes_
     */
    int32_t BuildRecursive(std::vector<int32_t>& indices, int64_t start, int64_t end, int depth) {
        if (start >= end) return -1;

        int axis = depth % 3;
        int64_t mid = (start + end) / 2;

        // Partial sort (Median-of-Medians) to find median element
        auto nth = indices.begin() + mid;
        auto start_it = indices.begin() + start;
        auto end_it = indices.begin() + end;

        std::nth_element(start_it, nth, end_it,
                         [this, axis](int32_t a, int32_t b) { return points_[a * 3 + axis] < points_[b * 3 + axis]; });

        int32_t node_idx = static_cast<int32_t>(nodes_.size());
        nodes_.push_back({*nth, static_cast<int8_t>(axis)});

        // Important: nodes_ reference might be invalidated by push_back if reallocated.
        // But we reserved N size, so it shouldn't happen.
        // Still, safe to compute children indices then assign.

        int32_t left_child = BuildRecursive(indices, start, mid, depth + 1);
        int32_t right_child = BuildRecursive(indices, mid + 1, end, depth + 1);

        nodes_[node_idx].left = left_child;
        nodes_[node_idx].right = right_child;

        return node_idx;
    }

    void KNNRecursive(int32_t node_idx, const float* query, int k,
                      std::priority_queue<std::pair<float, int32_t>>& pq) const {
        if (node_idx == -1) return;

        const Node& node = nodes_[node_idx];
        const float* point = points_ + node.point_idx * 3;

        // Calculate distance between query and current node point
        float d_sq = 0.0f;
        for (int i = 0; i < 3; ++i) {
            float diff = query[i] - point[i];
            d_sq += diff * diff;
        }

        // Add to PQ
        if (static_cast<int>(pq.size()) < k) {
            pq.push({d_sq, node.point_idx});
        } else {
            if (d_sq < pq.top().first) {
                pq.pop();
                pq.push({d_sq, node.point_idx});
            }
        }

        // Determine which side to search first
        float diff = query[node.split_axis] - point[node.split_axis];
        int32_t near_child = (diff < 0) ? node.left : node.right;
        int32_t far_child = (diff < 0) ? node.right : node.left;

        KNNRecursive(near_child, query, k, pq);

        // Pruning: Check if we need to search the far side
        // If the hypersphere intersects the splitting plane
        // Plane distance squared = diff * diff
        float plane_dist_sq = diff * diff;

        // If PQ is full, check if plane is closer than the K-th nearest neighbor found so far
        if (static_cast<int>(pq.size()) < k || plane_dist_sq < pq.top().first) {
            KNNRecursive(far_child, query, k, pq);
        }
    }
};

}  // namespace kernels
}  // namespace densecore

#endif  // DENSECORE_KERNELS_KDTREE_H
