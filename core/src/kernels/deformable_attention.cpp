/**
 * @file deformable_attention.cpp
 * @brief CPU fallback for DeformableAttention
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace densecore {
namespace {

class CpuDeformableAttentionOp : public AttentionOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (outputs.empty()) {
            return;
        }
        const auto* p = static_cast<const DeformableAttentionParams*>(params);

        // Point-cloud path:
        // inputs: [query(B,Q,D), key(B,N,D), value(B,N,D), ref(B,Q,3),
        //          offsets(B,Q,H,S*3), weights(B,Q,H,S), key_points(B,N,3)?]
        if (inputs.size() >= 6) {
            const Tensor* key_points = inputs.size() >= 7 ? inputs[6] : nullptr;
            PointCloudDeformableAttention3D(*inputs[0], *inputs[1], *inputs[2], *inputs[3], *inputs[4], *inputs[5],
                                            key_points, p, outputs[0]);
            return;
        }

        if (inputs.size() < 4) {
            return;
        }
        DeformableAttention(*inputs[0], *inputs[1], *inputs[2], *inputs[3], outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = true,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 0};
    }

    bool SupportsLayout(TensorLayout layout) const override {
        return layout == TensorLayout::SEQ || layout == TensorLayout::UNKNOWN;
    }

    void DeformableAttention(const Tensor& query, const Tensor& spatial_features, const Tensor& sampling_offsets,
                             const Tensor& attention_weights, Tensor* output) override {
        if (!query.IsValid() || !spatial_features.IsValid() || !sampling_offsets.IsValid() ||
            !attention_weights.IsValid() || !output) {
            return;
        }

        const int64_t B = query.shape[0];
        const int64_t Q = query.shape[1];
        const int64_t D = query.shape[2];

        const int64_t L = spatial_features.shape[1];
        const int64_t H_feat = spatial_features.shape[2];
        const int64_t W_feat = spatial_features.shape[3];

        const int64_t num_heads = sampling_offsets.shape[2];
        const int64_t num_points = sampling_offsets.shape[3];

        const float* feat_data = spatial_features.DataAs<float>();
        const float* offset_data = sampling_offsets.DataAs<float>();
        const float* weight_data = attention_weights.DataAs<float>();
        float* out_data = output->DataAs<float>();

        std::memset(out_data, 0, B * Q * D * sizeof(float));

        const int64_t D_per_head = D / num_heads;

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t q = 0; q < Q; ++q) {
                for (int64_t h = 0; h < num_heads; ++h) {
                    for (int64_t p = 0; p < num_points; ++p) {
                        const int64_t offset_idx = b * (Q * num_heads * num_points * 2) +
                                                   q * (num_heads * num_points * 2) + h * (num_points * 2) + p * 2;

                        float norm_x = offset_data[offset_idx];
                        float norm_y = offset_data[offset_idx + 1];

                        float x = (norm_x + 1.0f) * 0.5f * (W_feat - 1);
                        float y = (norm_y + 1.0f) * 0.5f * (H_feat - 1);

                        // Clamp coordinates
                        // Note: If using multiple levels, clamping logic might differ
                        x = std::max(0.0f, std::min(x, static_cast<float>(W_feat - 1)));
                        y = std::max(0.0f, std::min(y, static_cast<float>(H_feat - 1)));

                        int x0 = static_cast<int>(std::floor(x));
                        int y0 = static_cast<int>(std::floor(y));
                        int x1 = std::min(x0 + 1, static_cast<int>(W_feat - 1));
                        int y1 = std::min(y0 + 1, static_cast<int>(H_feat - 1));

                        float wx1 = x - x0;
                        float wx0 = 1.0f - wx1;
                        float wy1 = y - y0;
                        float wy0 = 1.0f - wy1;

                        const int64_t weight_idx =
                            b * (Q * num_heads * num_points) + q * (num_heads * num_points) + h * num_points + p;
                        float attn_weight = weight_data[weight_idx];

#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX)
                        // Prefetch next point offsets
                        if (p + 1 < num_points) {
                            const int64_t next_offset_idx = offset_idx + 2;
                            _mm_prefetch(reinterpret_cast<const char*>(&offset_data[next_offset_idx]), _MM_HINT_T0);
                        }
#endif

                        const int64_t level = 0;
                        const int64_t d_start = h * D_per_head;
                        const int64_t d_end = (h + 1) * D_per_head;

                        // Base offsets for 4 neighbors
                        const int64_t base_idx = b * (L * H_feat * W_feat * D) + level * (H_feat * W_feat * D);
                        const int64_t idx00 = base_idx + y0 * (W_feat * D) + x0 * D;
                        const int64_t idx01 = base_idx + y0 * (W_feat * D) + x1 * D;
                        const int64_t idx10 = base_idx + y1 * (W_feat * D) + x0 * D;
                        const int64_t idx11 = base_idx + y1 * (W_feat * D) + x1 * D;
                        const int64_t out_base_idx = b * Q * D + q * D;

                        int64_t d = d_start;

#if defined(__AVX512F__)
                        __m512 v_wx0 = _mm512_set1_ps(wx0);
                        __m512 v_wx1 = _mm512_set1_ps(wx1);
                        __m512 v_wy0 = _mm512_set1_ps(wy0);
                        __m512 v_wy1 = _mm512_set1_ps(wy1);
                        __m512 v_weight = _mm512_set1_ps(attn_weight);

                        for (; d + 16 <= d_end; d += 16) {
                            __m512 v00 = _mm512_loadu_ps(feat_data + idx00 + d);
                            __m512 v01 = _mm512_loadu_ps(feat_data + idx01 + d);
                            __m512 v10 = _mm512_loadu_ps(feat_data + idx10 + d);
                            __m512 v11 = _mm512_loadu_ps(feat_data + idx11 + d);

                            // Bilinear interpolation
                            // row0 = v00 * wx0 + v01 * wx1
                            __m512 row0 = _mm512_fmadd_ps(v00, v_wx0, _mm512_mul_ps(v01, v_wx1));
                            // row1 = v10 * wx0 + v11 * wx1
                            __m512 row1 = _mm512_fmadd_ps(v10, v_wx0, _mm512_mul_ps(v11, v_wx1));
                            // interp = row0 * wy0 + row1 * wy1
                            __m512 interp = _mm512_fmadd_ps(row0, v_wy0, _mm512_mul_ps(row1, v_wy1));

                            // Accumulate to output
                            __m512 out_val = _mm512_loadu_ps(out_data + out_base_idx + d);
                            out_val = _mm512_fmadd_ps(interp, v_weight, out_val);
                            _mm512_storeu_ps(out_data + out_base_idx + d, out_val);
                        }
#elif defined(__AVX2__)
                        __m256 v_wx0 = _mm256_set1_ps(wx0);
                        __m256 v_wx1 = _mm256_set1_ps(wx1);
                        __m256 v_wy0 = _mm256_set1_ps(wy0);
                        __m256 v_wy1 = _mm256_set1_ps(wy1);
                        __m256 v_weight = _mm256_set1_ps(attn_weight);

                        for (; d + 8 <= d_end; d += 8) {
                            __m256 v00 = _mm256_loadu_ps(feat_data + idx00 + d);
                            __m256 v01 = _mm256_loadu_ps(feat_data + idx01 + d);
                            __m256 v10 = _mm256_loadu_ps(feat_data + idx10 + d);
                            __m256 v11 = _mm256_loadu_ps(feat_data + idx11 + d);

                            // Bilinear interpolation
                            __m256 row0 = _mm256_add_ps(_mm256_mul_ps(v00, v_wx0), _mm256_mul_ps(v01, v_wx1));
                            __m256 row1 = _mm256_add_ps(_mm256_mul_ps(v10, v_wx0), _mm256_mul_ps(v11, v_wx1));
                            __m256 interp = _mm256_add_ps(_mm256_mul_ps(row0, v_wy0), _mm256_mul_ps(row1, v_wy1));

                            // Accumulate to output
                            __m256 out_val = _mm256_loadu_ps(out_data + out_base_idx + d);
                            // out += interp * weight
                            // Using FMA if available
#if defined(__FMA__)
                            out_val = _mm256_fmadd_ps(interp, v_weight, out_val);
#else
                            out_val = _mm256_add_ps(out_val, _mm256_mul_ps(interp, v_weight));
#endif
                            _mm256_storeu_ps(out_data + out_base_idx + d, out_val);
                        }
#endif

                        // Scalar remainder
                        for (; d < d_end; ++d) {
                            float v00 = feat_data[idx00 + d];
                            float v01 = feat_data[idx01 + d];
                            float v10 = feat_data[idx10 + d];
                            float v11 = feat_data[idx11 + d];

                            float interp = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                            out_data[out_base_idx + d] += attn_weight * interp;
                        }
                    }
                }
            }
        }
    }

private:
    static int64_t ArgMinSquaredDistance3DStrided(const float* points_xyz, int64_t num_points, int64_t point_stride,
                                                  float x, float y, float z) {
        int64_t best_idx = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (int64_t i = 0; i < num_points; ++i) {
            const float* p = points_xyz + i * point_stride;
            const float dx = p[0] - x;
            const float dy = p[1] - y;
            const float dz = p[2] - z;
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best_dist) {
                best_dist = d2;
                best_idx = i;
            }
        }
        return best_idx;
    }

    class KdTree3D {
    public:
        KdTree3D(const float* points_xyz, int64_t num_points, int64_t point_stride)
            : points_xyz_(points_xyz), num_points_(num_points), point_stride_(point_stride) {
            if (!points_xyz_ || num_points_ <= 0 || point_stride_ < 3) {
                return;
            }

            std::vector<int64_t> indices(static_cast<size_t>(num_points_));
            for (int64_t i = 0; i < num_points_; ++i) {
                indices[static_cast<size_t>(i)] = i;
            }

            nodes_.reserve(static_cast<size_t>(num_points_));
            root_ = BuildRecursive(indices, 0, num_points_, 0);
        }

        bool IsValid() const { return root_ >= 0; }

        int64_t Nearest(float x, float y, float z) const {
            if (root_ < 0) {
                return 0;
            }

            int64_t best_idx = nodes_[static_cast<size_t>(root_)].point_idx;
            float best_dist = SquaredDistance(best_idx, x, y, z);
            NearestRecursive(root_, x, y, z, &best_idx, &best_dist);
            return best_idx;
        }

    private:
        struct Node {
            int64_t point_idx = 0;
            int left = -1;
            int right = -1;
            uint8_t axis = 0;
        };

        float Coord(int64_t point_idx, int axis) const { return points_xyz_[point_idx * point_stride_ + axis]; }

        float SquaredDistance(int64_t point_idx, float x, float y, float z) const {
            const float dx = Coord(point_idx, 0) - x;
            const float dy = Coord(point_idx, 1) - y;
            const float dz = Coord(point_idx, 2) - z;
            return dx * dx + dy * dy + dz * dz;
        }

        int BuildRecursive(std::vector<int64_t>& indices, int64_t begin, int64_t end, int depth) {
            if (begin >= end) {
                return -1;
            }

            const int axis = depth % 3;
            const int64_t mid = begin + (end - begin) / 2;
            auto cmp = [&](int64_t a, int64_t b) { return Coord(a, axis) < Coord(b, axis); };
            std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end, cmp);

            const int node_idx = static_cast<int>(nodes_.size());
            Node node;
            node.point_idx = indices[static_cast<size_t>(mid)];
            node.axis = static_cast<uint8_t>(axis);
            nodes_.push_back(node);

            nodes_[static_cast<size_t>(node_idx)].left = BuildRecursive(indices, begin, mid, depth + 1);
            nodes_[static_cast<size_t>(node_idx)].right = BuildRecursive(indices, mid + 1, end, depth + 1);
            return node_idx;
        }

        void NearestRecursive(int node_idx, float x, float y, float z, int64_t* best_idx, float* best_dist) const {
            if (node_idx < 0) {
                return;
            }

            const Node& node = nodes_[static_cast<size_t>(node_idx)];
            const float d2 = SquaredDistance(node.point_idx, x, y, z);
            if (d2 < *best_dist) {
                *best_dist = d2;
                *best_idx = node.point_idx;
            }

            const int axis = static_cast<int>(node.axis);
            const float split = Coord(node.point_idx, axis);
            const float target = (axis == 0) ? x : ((axis == 1) ? y : z);
            const float delta = target - split;

            const int near_child = (delta < 0.0f) ? node.left : node.right;
            const int far_child = (delta < 0.0f) ? node.right : node.left;

            NearestRecursive(near_child, x, y, z, best_idx, best_dist);
            if (delta * delta < *best_dist) {
                NearestRecursive(far_child, x, y, z, best_idx, best_dist);
            }
        }

        const float* points_xyz_ = nullptr;
        int64_t num_points_ = 0;
        int64_t point_stride_ = 0;
        int root_ = -1;
        std::vector<Node> nodes_;
    };

    static int64_t HashToIndex(float x, float y, float z, int64_t num_points) {
        const float ux = std::clamp((x + 1.0f) * 0.5f, 0.0f, 1.0f);
        const float uy = std::clamp((y + 1.0f) * 0.5f, 0.0f, 1.0f);
        const float uz = std::clamp((z + 1.0f) * 0.5f, 0.0f, 1.0f);
        const float mixed = 0.43f * ux + 0.37f * uy + 0.20f * uz;
        const float pos = mixed * static_cast<float>(std::max<int64_t>(0, num_points - 1));
        return static_cast<int64_t>(std::clamp<int64_t>(static_cast<int64_t>(std::llround(pos)), 0, num_points - 1));
    }

    void PointCloudDeformableAttention3D(const Tensor& query, const Tensor& key, const Tensor& value,
                                         const Tensor& reference_points, const Tensor& sampling_offsets,
                                         const Tensor& attention_weights, const Tensor* key_points,
                                         const DeformableAttentionParams* params, Tensor* output) {
        if (!query.IsValid() || !key.IsValid() || !value.IsValid() || !reference_points.IsValid() ||
            !sampling_offsets.IsValid() || !attention_weights.IsValid() || !output) {
            return;
        }
        if (query.ndim != 3 || key.ndim != 3 || value.ndim != 3 || reference_points.ndim != 3 || query.shape[0] <= 0 ||
            query.shape[1] <= 0 || query.shape[2] <= 0 || value.shape[1] <= 0) {
            return;
        }

        const int64_t B = query.shape[0];
        const int64_t Q = query.shape[1];
        const int64_t D = query.shape[2];
        const int64_t N = value.shape[1];
        const int64_t Dv = value.shape[2];
        const int64_t Dout = std::min(D, Dv);

        const int64_t configured_heads = params ? std::max<int64_t>(1, params->num_heads) : 1;
        int64_t H = configured_heads;
        bool shared_offsets_across_heads = false;
        bool shared_weights_across_heads = false;
        if (sampling_offsets.ndim >= 4) {
            const int64_t offset_heads = sampling_offsets.shape[2];
            shared_offsets_across_heads = (offset_heads == 1 && configured_heads > 1);
            H = shared_offsets_across_heads ? configured_heads : offset_heads;
        } else if (sampling_offsets.ndim == 3) {
            H = configured_heads;
            shared_offsets_across_heads = true;
        }
        if (H <= 0) {
            H = 1;
        }
        if (attention_weights.ndim >= 4) {
            const int64_t weight_heads = attention_weights.shape[2];
            shared_weights_across_heads = (weight_heads == 1 && H > 1);
        } else if (attention_weights.ndim == 3) {
            shared_weights_across_heads = true;
        }

        int64_t offsets_tail = 0;
        if (sampling_offsets.ndim >= 4) {
            offsets_tail = sampling_offsets.shape[3];
        } else if (sampling_offsets.ndim == 3) {
            offsets_tail = sampling_offsets.shape[2];
        }
        if (offsets_tail <= 0) {
            return;
        }
        const int64_t S = std::max<int64_t>(1, offsets_tail / 3);

        const float* key_data = key.DataAs<float>();
        const float* val_data = value.DataAs<float>();
        const float* ref_data = reference_points.DataAs<float>();
        const float* off_data = sampling_offsets.DataAs<float>();
        const float* w_data = attention_weights.DataAs<float>();
        float* out_data = output->DataAs<float>();

        std::memset(out_data, 0, static_cast<size_t>(B * Q * D) * sizeof(float));

        const bool has_key_points = key_points && key_points->IsValid() && key_points->ndim == 3 &&
                                    key_points->shape[0] == B && key_points->shape[1] == N && key_points->shape[2] >= 3;
        const bool key_has_xyz = key.shape[2] >= 3;
        const float* kp_data = has_key_points ? key_points->DataAs<float>() : nullptr;

        const int64_t d_per_head = std::max<int64_t>(1, D / H);
        for (int64_t b = 0; b < B; ++b) {
            const float* nn_points = nullptr;
            int64_t nn_stride = 0;
            if (has_key_points) {
                nn_points = kp_data + static_cast<size_t>(b) * N * 3;
                nn_stride = 3;
            } else if (key_has_xyz) {
                nn_points = key_data + static_cast<size_t>(b) * N * key.shape[2];
                nn_stride = key.shape[2];
            }

            // Build once per point-cloud batch and reuse for all query samples.
            const KdTree3D kd_tree(nn_points, N, nn_stride);
            const bool use_kd_tree = kd_tree.IsValid();

            for (int64_t q = 0; q < Q; ++q) {
                const float* ref = ref_data + (b * Q + q) * 3;

                for (int64_t h = 0; h < H; ++h) {
                    float sum_w = 0.0f;
                    for (int64_t s = 0; s < S; ++s) {
                        size_t w_idx = 0;
                        if (attention_weights.ndim >= 4) {
                            const int64_t weight_head = shared_weights_across_heads ? 0 : h;
                            w_idx = static_cast<size_t>(((b * Q + q) * attention_weights.shape[2] + weight_head) *
                                                        S + s);
                        } else {
                            w_idx = static_cast<size_t>((b * Q + q) * S + s);
                        }
                        sum_w += std::max(0.0f, w_data[w_idx]);
                    }
                    const float inv_sum_w = (sum_w > 1e-6f) ? (1.0f / sum_w) : (1.0f / static_cast<float>(S));

                    const int64_t d_start = h * d_per_head;
                    const int64_t d_end = std::min<int64_t>(Dout, (h == H - 1) ? Dout : (h + 1) * d_per_head);
                    for (int64_t s = 0; s < S; ++s) {
                        size_t off_idx = 0;
                        if (sampling_offsets.ndim >= 4) {
                            const int64_t offset_head = shared_offsets_across_heads ? 0 : h;
                            off_idx = static_cast<size_t>(((b * Q + q) * sampling_offsets.shape[2] + offset_head) *
                                                          (S * 3) + s * 3);
                        } else {
                            off_idx = static_cast<size_t>((b * Q + q) * (S * 3) + s * 3);
                        }
                        const float sx = ref[0] + off_data[off_idx + 0];
                        const float sy = ref[1] + off_data[off_idx + 1];
                        const float sz = ref[2] + off_data[off_idx + 2];

                        int64_t n_idx = 0;
                        if (nn_points && nn_stride >= 3) {
                            if (use_kd_tree) {
                                n_idx = kd_tree.Nearest(sx, sy, sz);
                            } else {
                                n_idx = ArgMinSquaredDistance3DStrided(nn_points, N, nn_stride, sx, sy, sz);
                            }
                        } else {
                            n_idx = HashToIndex(sx, sy, sz, N);
                        }

                        size_t w_idx = 0;
                        if (attention_weights.ndim >= 4) {
                            const int64_t weight_head = shared_weights_across_heads ? 0 : h;
                            w_idx = static_cast<size_t>(((b * Q + q) * attention_weights.shape[2] + weight_head) *
                                                        S + s);
                        } else {
                            w_idx = static_cast<size_t>((b * Q + q) * S + s);
                        }
                        const float w = std::max(0.0f, w_data[w_idx]) * inv_sum_w;
                        const float* v_ptr = val_data + (static_cast<size_t>(b) * N + n_idx) * Dv;
                        float* out_ptr = out_data + (static_cast<size_t>(b) * Q + q) * D;
                        for (int64_t d = d_start; d < d_end; ++d) {
                            out_ptr[d] += w * v_ptr[d];
                        }
                    }
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuDeformableAttentionOp, OpType::DeformableAttention, DeviceType::CPU);

}  // namespace
}  // namespace densecore
