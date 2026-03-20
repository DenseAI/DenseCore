/**
 * @file triangular_mult_update.cpp
 * @brief Triangular Multiplication Update for Protein Structure (AlphaFold2/3)
 *
 * Computes triangular multiplication with optional sigmoid gating.
 * Outgoing: out[i,j,c] = sum_k left[i,k,c] * right[j,k,c]  (L @ R^T per channel)
 * Incoming: out[i,j,c] = sum_k left[k,i,c] * right[k,j,c]  (L^T @ R per channel)
 *
 * Inputs:
 *   [0] pair_repr  [N_res*N_res, C_z]   - Pair representation (flattened)
 *   [1] w_left     [C_z, C_mul]          - Left projection weight
 *   [2] w_right    [C_z, C_mul]          - Right projection weight
 *   [3] w_out      [C_mul, C_z]          - Output projection weight
 *   [4] gate_w     [C_z, C_mul]          - Gate weight (optional, if has_gate)
 *   [5] gate_b     [C_mul]               - Gate bias (optional, if has_gate)
 *
 * Output:
 *   [0] output     [N_res*N_res, C_z]    - Updated pair representation delta
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace densecore {
namespace {

class CpuTriangularMultUpdateOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 4 || outputs.empty()) return;

        const auto* p = static_cast<const TriangularMultUpdateParams*>(params);
        TriangularMultUpdateParams default_params;
        if (!p) p = &default_params;

        const Tensor& pair_repr = *inputs[0];
        const Tensor& w_left = *inputs[1];
        const Tensor& w_right = *inputs[2];
        const Tensor& w_out = *inputs[3];

        const Tensor* gate_w = (p->has_gate && inputs.size() > 4) ? inputs[4] : nullptr;
        const Tensor* gate_b = (p->has_gate && inputs.size() > 5) ? inputs[5] : nullptr;

        TriangularMultUpdate(pair_repr, w_left, w_right, w_out, gate_w, gate_b, *p, outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 8 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};
    }

private:
    static bool CheckedMul(size_t a, size_t b, size_t* out) {
        if (!out) return false;
        if (a == 0 || b == 0) {
            *out = 0;
            return true;
        }
        if (a > std::numeric_limits<size_t>::max() / b) {
            return false;
        }
        *out = a * b;
        return true;
    }

    static bool CheckedMul3(size_t a, size_t b, size_t c, size_t* out) {
        size_t ab = 0;
        if (!CheckedMul(a, b, &ab)) return false;
        return CheckedMul(ab, c, out);
    }

    static bool FitsInt(size_t v) { return v <= static_cast<size_t>(std::numeric_limits<int>::max()); }

    static void Transpose2D(const float* src, float* dst, int rows, int cols) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                dst[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
            }
        }
    }

    static void MatMulWithBTransposed(const float* A, const float* B_transposed, float* C, int M, int N, int K) {
        hwy_kernels::GemmFP32_Hwy(C, A, B_transposed, M, N, K, 0, N);
    }

    void TriangularMultUpdate(const Tensor& pair_repr, const Tensor& w_left, const Tensor& w_right, const Tensor& w_out,
                              const Tensor* gate_w, const Tensor* gate_b, const TriangularMultUpdateParams& params,
                              Tensor* output) {
        if (!pair_repr.IsValid() || !output) return;

        const int64_t n_res_sq_i64 = pair_repr.shape[0];
        const int64_t c_z_i64 = pair_repr.shape[1];
        const int64_t c_mul_i64 = w_left.shape[1];
        if (n_res_sq_i64 <= 0 || c_z_i64 <= 0 || c_mul_i64 <= 0) return;
        if (!w_left.IsValid() || !w_right.IsValid() || !w_out.IsValid()) return;
        if (w_left.shape[0] != c_z_i64 || w_right.shape[0] != c_z_i64 || w_right.shape[1] != c_mul_i64 ||
            w_out.shape[0] != c_mul_i64 || w_out.shape[1] != c_z_i64) {
            return;
        }

        // Keep int-based kernels safe: reject dimensions that cannot be represented.
        if (n_res_sq_i64 > std::numeric_limits<int>::max() || c_z_i64 > std::numeric_limits<int>::max() ||
            c_mul_i64 > std::numeric_limits<int>::max()) {
            return;
        }

        const int64_t n_res_i64 = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(n_res_sq_i64))));
        if (n_res_i64 <= 0 || n_res_i64 > std::numeric_limits<int>::max()) return;
        if (n_res_i64 * n_res_i64 != n_res_sq_i64) return;

        const int n_res_sq = static_cast<int>(n_res_sq_i64);
        const int c_z = static_cast<int>(c_z_i64);
        const int n_res = static_cast<int>(n_res_i64);
        const int c_mul = static_cast<int>(c_mul_i64);

        const float* pair_data = pair_repr.DataAs<float>();
        const float* wl = w_left.DataAs<float>();
        const float* wr = w_right.DataAs<float>();
        const float* wo = w_out.DataAs<float>();
        float* out_data = output->DataAs<float>();

        size_t pair_size = 0;
        if (!CheckedMul(static_cast<size_t>(n_res), static_cast<size_t>(n_res), &pair_size)) return;
        const bool incoming = params.incoming;
        const float* gate_w_data = (gate_w && gate_w->IsValid()) ? gate_w->DataAs<float>() : nullptr;
        const float* gate_b_data = (gate_b && gate_b->IsValid()) ? gate_b->DataAs<float>() : nullptr;
        const size_t c_z_sz = static_cast<size_t>(c_z);
        const size_t c_mul_sz = static_cast<size_t>(c_mul);
        if (gate_w_data && (gate_w->shape[0] != c_z_i64 || gate_w->shape[1] != c_mul_i64)) return;
        if (gate_b_data && gate_b->shape[0] != c_mul_i64) return;

        // Pre-transpose static weights once so all GEMMs can use a single
        // Highway kernel signature: A[M,K] @ B_t[N,K]^T.
        size_t wl_size = 0;
        size_t wo_size = 0;
        if (!CheckedMul(c_mul_sz, c_z_sz, &wl_size)) return;
        if (!CheckedMul(c_z_sz, c_mul_sz, &wo_size)) return;
        std::vector<float> wl_t(wl_size);
        std::vector<float> wr_t(wl_size);
        std::vector<float> wo_t(wo_size);
        Transpose2D(wl, wl_t.data(), c_z, c_mul);
        Transpose2D(wr, wr_t.data(), c_z, c_mul);
        Transpose2D(wo, wo_t.data(), c_mul, c_z);

        std::vector<float> gate_w_t;
        if (gate_w_data) {
            gate_w_t.resize(wl_size);
            Transpose2D(gate_w_data, gate_w_t.data(), c_z, c_mul);
        }

        const bool has_tile = params.tile_row_start >= 0 && params.tile_row_end > params.tile_row_start &&
                              params.tile_col_start >= 0 && params.tile_col_end > params.tile_col_start;

        if (has_tile) {
            const int row_start = std::max(0, params.tile_row_start);
            const int row_end = std::min(n_res, params.tile_row_end);
            const int col_start = std::max(0, params.tile_col_start);
            const int col_end = std::min(n_res, params.tile_col_end);
            if (row_end <= row_start || col_end <= col_start) {
                return;
            }

            const size_t tile_rows = static_cast<size_t>(row_end - row_start);
            const size_t tile_cols = static_cast<size_t>(col_end - col_start);
            size_t tile_tokens = 0;
            if (!CheckedMul(tile_rows, tile_cols, &tile_tokens)) return;
            if (!FitsInt(tile_rows) || !FitsInt(tile_cols) || !FitsInt(tile_tokens)) return;

            const int tile_rows_i = static_cast<int>(tile_rows);
            const int tile_cols_i = static_cast<int>(tile_cols);
            const int tile_tokens_i = static_cast<int>(tile_tokens);

            const bool output_is_tiled =
                output->shape[0] == static_cast<int64_t>(tile_tokens) && output->shape[1] == c_z;
            const bool output_is_full = output->shape[0] == n_res_sq && output->shape[1] == c_z;
            if (!output_is_tiled && !output_is_full) {
                return;
            }

            const int k_tile = std::max(16, std::min(128, n_res));
            size_t out_cmul_size = 0;
            if (!CheckedMul(tile_tokens, c_mul_sz, &out_cmul_size)) return;
            std::vector<float> out_cmul(out_cmul_size, 0.0f);

            std::vector<float> left_in;
            std::vector<float> right_in;
            std::vector<float> left_proj;
            std::vector<float> right_proj;
            std::vector<float> left_gate;
            std::vector<float> right_gate;
            size_t left_in_cap = 0;
            size_t right_in_cap = 0;
            size_t left_proj_cap = 0;
            size_t right_proj_cap = 0;
            if (!CheckedMul3(tile_rows, static_cast<size_t>(k_tile), c_z_sz, &left_in_cap)) return;
            if (!CheckedMul3(tile_cols, static_cast<size_t>(k_tile), c_z_sz, &right_in_cap)) return;
            if (!CheckedMul3(tile_rows, static_cast<size_t>(k_tile), c_mul_sz, &left_proj_cap)) return;
            if (!CheckedMul3(tile_cols, static_cast<size_t>(k_tile), c_mul_sz, &right_proj_cap)) return;
            left_in.reserve(left_in_cap);
            right_in.reserve(right_in_cap);
            left_proj.reserve(left_proj_cap);
            right_proj.reserve(right_proj_cap);

            auto gather_pairs_offset = [&](int outer_start, int outer, int inner, int inner_start, bool outer_is_row,
                                           std::vector<float>* dst) -> bool {
                size_t elems = 0;
                if (!CheckedMul3(static_cast<size_t>(outer), static_cast<size_t>(inner), c_z_sz, &elems)) {
                    return false;
                }
                dst->resize(elems);
                for (int o = 0; o < outer; ++o) {
                    for (int in = 0; in < inner; ++in) {
                        int row = 0;
                        int col = 0;
                        if (outer_is_row) {
                            row = outer_start + o;
                            col = inner_start + in;
                        } else {
                            row = inner_start + in;
                            col = outer_start + o;
                        }
                        const float* src = pair_data + (static_cast<size_t>(row) * n_res + col) * c_z;
                        float* dst_ptr = dst->data() + (static_cast<size_t>(o) * inner + in) * c_z;
                        std::memcpy(dst_ptr, src, static_cast<size_t>(c_z) * sizeof(float));
                    }
                }
                return true;
            };

            for (int k0 = 0; k0 < n_res; k0 += k_tile) {
                const int k_len = std::min(k_tile, n_res - k0);

                if (!incoming) {
                    // Outgoing:
                    // left uses (i, k), i in [row_start,row_end), k in [k0,k0+k_len)
                    // right uses (j, k), j in [col_start,col_end), k in [k0,k0+k_len)
                    if (!gather_pairs_offset(row_start, tile_rows_i, k_len, k0, true, &left_in)) return;
                    if (!gather_pairs_offset(col_start, tile_cols_i, k_len, k0, true, &right_in)) return;
                } else {
                    // Incoming:
                    // left uses (k, i), k in [k0,k0+k_len), i in [row_start,row_end)
                    // right uses (k, j), k in [k0,k0+k_len), j in [col_start,col_end)
                    if (!gather_pairs_offset(row_start, tile_rows_i, k_len, k0, false, &left_in)) return;
                    if (!gather_pairs_offset(col_start, tile_cols_i, k_len, k0, false, &right_in)) return;
                }

                size_t left_proj_size = 0;
                size_t right_proj_size = 0;
                if (!CheckedMul3(tile_rows, static_cast<size_t>(k_len), c_mul_sz, &left_proj_size)) return;
                if (!CheckedMul3(tile_cols, static_cast<size_t>(k_len), c_mul_sz, &right_proj_size)) return;
                left_proj.resize(left_proj_size);
                right_proj.resize(right_proj_size);

                if (tile_rows_i > std::numeric_limits<int>::max() / std::max(1, k_len) ||
                    tile_cols_i > std::numeric_limits<int>::max() / std::max(1, k_len)) {
                    return;
                }
                const int left_rows = tile_rows_i * k_len;
                const int right_rows = tile_cols_i * k_len;
                MatMulWithBTransposed(left_in.data(), wl_t.data(), left_proj.data(), left_rows, c_mul, c_z);
                MatMulWithBTransposed(right_in.data(), wr_t.data(), right_proj.data(), right_rows, c_mul, c_z);

                if (params.has_gate && gate_w_data && gate_b_data) {
                    left_gate.resize(left_proj_size);
                    right_gate.resize(right_proj_size);
                    MatMulWithBTransposed(left_in.data(), gate_w_t.data(), left_gate.data(), left_rows, c_mul, c_z);
                    MatMulWithBTransposed(right_in.data(), gate_w_t.data(), right_gate.data(), right_rows, c_mul, c_z);
                    for (size_t idx = 0; idx < left_gate.size(); ++idx) {
                        const float x = left_gate[idx] + gate_b_data[idx % static_cast<size_t>(c_mul)];
                        const float sig = 1.0f / (1.0f + std::exp(-x));
                        left_proj[idx] *= sig;
                    }
                    for (size_t idx = 0; idx < right_gate.size(); ++idx) {
                        const float x = right_gate[idx] + gate_b_data[idx % static_cast<size_t>(c_mul)];
                        const float sig = 1.0f / (1.0f + std::exp(-x));
                        right_proj[idx] *= sig;
                    }
                }

                for (size_t ii = 0; ii < tile_rows; ++ii) {
                    for (size_t jj = 0; jj < tile_cols; ++jj) {
                        float* out_row = out_cmul.data() + (ii * tile_cols + jj) * c_mul_sz;
                        for (int kk = 0; kk < k_len; ++kk) {
                            const float* left_row =
                                left_proj.data() +
                                (ii * static_cast<size_t>(k_len) + static_cast<size_t>(kk)) * c_mul_sz;
                            const float* right_row =
                                right_proj.data() +
                                (jj * static_cast<size_t>(k_len) + static_cast<size_t>(kk)) * c_mul_sz;
                            for (int c = 0; c < c_mul; ++c) {
                                out_row[c] += left_row[c] * right_row[c];
                            }
                        }
                    }
                }
            }

            if (output_is_tiled) {
                MatMulWithBTransposed(out_cmul.data(), wo_t.data(), out_data, tile_tokens_i, c_z, c_mul);
                return;
            }

            size_t out_tile_size = 0;
            if (!CheckedMul(tile_tokens, c_z_sz, &out_tile_size)) return;
            std::vector<float> out_tile(out_tile_size, 0.0f);
            MatMulWithBTransposed(out_cmul.data(), wo_t.data(), out_tile.data(), tile_tokens_i, c_z, c_mul);
            for (size_t ii = 0; ii < tile_rows; ++ii) {
                for (size_t jj = 0; jj < tile_cols; ++jj) {
                    const int global_row = row_start + static_cast<int>(ii);
                    const int global_col = col_start + static_cast<int>(jj);
                    const float* src = out_tile.data() + (ii * tile_cols + jj) * c_z_sz;
                    float* dst = out_data + (static_cast<size_t>(global_row) * n_res + global_col) * c_z;
                    std::memcpy(dst, src, static_cast<size_t>(c_z) * sizeof(float));
                }
            }
            return;
        }

        // Linear projections: left[N*N, c_mul] = pair[N*N, c_z] @ w_left[c_z, c_mul]
        size_t pair_c_mul = 0;
        if (!CheckedMul(pair_size, c_mul_sz, &pair_c_mul)) return;
        std::vector<float> left(pair_c_mul);
        std::vector<float> right(pair_c_mul);
        MatMulWithBTransposed(pair_data, wl_t.data(), left.data(), n_res_sq, c_mul, c_z);
        MatMulWithBTransposed(pair_data, wr_t.data(), right.data(), n_res_sq, c_mul, c_z);

        // Optional sigmoid gating
        if (params.has_gate && gate_w && gate_b && gate_w->IsValid() && gate_b->IsValid()) {
            std::vector<float> gate(pair_c_mul);
            MatMulWithBTransposed(pair_data, gate_w_t.data(), gate.data(), n_res_sq, c_mul, c_z);
            const float* gb = gate_b_data;
            for (size_t i = 0; i < gate.size(); ++i) {
                const float x = gate[i] + gb[i % c_mul];
                const float sig = 1.0f / (1.0f + std::exp(-x));
                left[i] *= sig;
                right[i] *= sig;
            }
        }

        // Transpose [n_res, n_res, c_mul] -> [c_mul, n_res, n_res] for batched matmul
        std::vector<float> left_batch(pair_c_mul);
        std::vector<float> right_batch(pair_c_mul);
        for (int i = 0; i < n_res; ++i) {
            for (int j = 0; j < n_res; ++j) {
                const size_t src = (static_cast<size_t>(i) * n_res + j) * c_mul;
                for (int c = 0; c < c_mul; ++c) {
                    const size_t r_dst = static_cast<size_t>(c) * n_res * n_res + i * n_res + j;
                    right_batch[r_dst] = right[src + c];
                    if (incoming) {
                        const size_t l_dst = static_cast<size_t>(c) * n_res * n_res + j * n_res + i;
                        left_batch[l_dst] = left[src + c];
                    } else {
                        left_batch[r_dst] = left[src + c];
                    }
                }
            }
        }

        // Per-channel matmul
        std::vector<float> out_batch(pair_c_mul, 0.0f);
        std::vector<float> right_channel_t;
        if (incoming) {
            right_channel_t.resize(pair_size);
        }
        for (int c = 0; c < c_mul; ++c) {
            float* lp = left_batch.data() + static_cast<size_t>(c) * n_res * n_res;
            float* rp = right_batch.data() + static_cast<size_t>(c) * n_res * n_res;
            float* op = out_batch.data() + static_cast<size_t>(c) * n_res * n_res;
            if (incoming) {
                Transpose2D(rp, right_channel_t.data(), n_res, n_res);
                MatMulWithBTransposed(lp, right_channel_t.data(), op, n_res, n_res, n_res);
            } else {
                MatMulWithBTransposed(lp, rp, op, n_res, n_res, n_res);
            }
        }

        // Transpose back [c_mul, n_res, n_res] -> [n_res, n_res, c_mul]
        std::vector<float> transposed(pair_c_mul);
        for (int c = 0; c < c_mul; ++c) {
            for (int i = 0; i < n_res; ++i) {
                for (int j = 0; j < n_res; ++j) {
                    const size_t src = static_cast<size_t>(c) * n_res * n_res + i * n_res + j;
                    const size_t dst = (static_cast<size_t>(i) * n_res + j) * c_mul + c;
                    transposed[dst] = out_batch[src];
                }
            }
        }

        // Output projection: out[N*N, c_z] = transposed[N*N, c_mul] @ w_out[c_mul, c_z]
        MatMulWithBTransposed(transposed.data(), wo_t.data(), out_data, n_res_sq, c_z, c_mul);
    }
};

DENSECORE_REGISTER_OP(CpuTriangularMultUpdateOp, OpType::TriangularMultUpdate, DeviceType::CPU);

}  // namespace
}  // namespace densecore
