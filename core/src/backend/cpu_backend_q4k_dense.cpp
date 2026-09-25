#include "backend/cpu_backend_q4k_dense.h"
#include "backend/cpu_moe_execution.h"

#include "densecore/backend/cpu_backend.h"
#include "ggml-cpu.h"
#include "thread_pool_impl.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace densecore::internal {
namespace {

struct Q4KDenseScratch {
    std::vector<uint8_t> q8_rows;
    std::vector<uint8_t> q8x4_groups;
    std::vector<uint8_t> q8_tail;
};

Q4KDenseScratch& GetQ4KDenseScratch() {
    static thread_local Q4KDenseScratch scratch;
    return scratch;
}

bool RunQ4KRepackedDenseGemv(CpuBackend* backend, const std::shared_ptr<kernels::Q4KRepackedGemvWeight>& packed,
                             const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data, int64_t rows,
                             int64_t output_cols, int numa_node, bool allow_parallel) {
    if (!backend || !packed || !qinput_data || !output_data || rows <= 0 || packed->rows != output_cols ||
        (output_cols % 8) != 0 || packed->cols <= 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int thread_count = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(output_cols / 8);
    const auto compute_tiles = [&](int tile_start, int tile_end) {
        return kernels::RunQ4KRepackedGemvRows(packed, qinput_data, qinput_row_bytes, output_data, rows, output_cols,
                                               tile_start, tile_end);
    };
    if (thread_count > 1 && tile_count >= 2) {
        std::atomic<bool> ok{true};
        pool.ParallelFor(tile_count, [&](int start, int end, int) {
            if (!compute_tiles(start, end)) {
                ok.store(false, std::memory_order_relaxed);
            }
        });
        return ok.load(std::memory_order_relaxed);
    }
    return compute_tiles(0, tile_count);
}

}  // namespace

bool RunQ4KRawBatchedDenseGemm(const CpuExecutionOptions& options, CpuBackend* backend, const void* weight_data, const float* input_data,
                               float* output_data, int64_t rows, int64_t output_cols, int64_t input_cols, int numa_node,
                               bool allow_parallel) {
    constexpr int64_t kMaxRawBatch = 256;
    if (!backend || !weight_data || !input_data || !output_data || rows <= 0 || output_cols <= 0 || input_cols <= 0 ||
        (input_cols % kernels::kQ4KSuperBlock) != 0) {
        return false;
    }

    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, input_cols);
    if (!q8_traits || !q8_traits->from_float || q8_row_bytes == 0) {
        return false;
    }
    auto& scratch = GetQ4KDenseScratch();
    const size_t q8_bytes = static_cast<size_t>(std::min(rows, kMaxRawBatch)) * q8_row_bytes;
    if (scratch.q8_rows.size() < q8_bytes) {
        scratch.q8_rows.resize(q8_bytes);
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int thread_count = allow_parallel ? pool.GetNumThreads() : 1;
    for (int64_t row_offset = 0; row_offset < rows; row_offset += kMaxRawBatch) {
        const int chunk_rows = static_cast<int>(std::min(kMaxRawBatch, rows - row_offset));
        const auto quantize_rows = [&](int row_start, int row_end, int) {
            for (int row = row_start; row < row_end; ++row) {
                q8_traits->from_float(input_data +
                                          static_cast<size_t>(row_offset + row) * static_cast<size_t>(input_cols),
                                      scratch.q8_rows.data() + static_cast<size_t>(row) * q8_row_bytes, input_cols);
            }
        };
        if (thread_count > 1 && chunk_rows >= 8) {
            pool.ParallelFor(chunk_rows, quantize_rows);
        } else {
            quantize_rows(0, chunk_rows, 0);
        }
        if (!RunQ4KRawBatchedQuantizedProjection(options, backend, weight_data, scratch.q8_rows.data(), q8_row_bytes,
                                                 output_data +
                                                     static_cast<size_t>(row_offset) * static_cast<size_t>(output_cols),
                                                 chunk_rows, output_cols, input_cols, numa_node, allow_parallel)) {
            return false;
        }
    }
    return true;
}

bool RunQ4KRepackedDenseGemm(CpuBackend* backend, const std::shared_ptr<kernels::Q4KRepackedGemvWeight>& packed,
                             const float* input_data, float* output_data, int64_t rows, int64_t output_cols,
                             int64_t input_cols, int numa_node, bool allow_parallel) {
    if (!backend || !packed || !input_data || !output_data || rows < 4 || packed->rows != output_cols ||
        (output_cols % 8) != 0 || packed->cols != input_cols || (input_cols % kernels::kQ4KSuperBlock) != 0 ||
        !kernels::Q4KRealPackedGemvKernelAvailable()) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int thread_count = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(output_cols / 8);
    const int blocks_per_row = static_cast<int>(packed->blocks_per_row);
    const size_t q8x4_group_bytes = 4 * ggml_row_size(GGML_TYPE_Q8_K, input_cols);
    const int64_t group_count = rows / 4;

    auto& scratch = GetQ4KDenseScratch();
    const size_t q8x4_bytes = static_cast<size_t>(group_count) * q8x4_group_bytes;
    if (scratch.q8x4_groups.size() < q8x4_bytes) {
        scratch.q8x4_groups.resize(q8x4_bytes);
    }

    const auto quantize_groups = [&](int group_start, int group_end, int) {
        for (int group = group_start; group < group_end; ++group) {
            const int64_t row = static_cast<int64_t>(group) * 4;
            const float* input_group = input_data + static_cast<size_t>(row) * static_cast<size_t>(input_cols);
            uint8_t* q8x4_group = scratch.q8x4_groups.data() + static_cast<size_t>(group) * q8x4_group_bytes;
            ggml_quantize_mat_q8_K_4x8(input_group, q8x4_group, input_cols);
        }
    };
    if (thread_count > 1 && group_count >= 2) {
        pool.ParallelFor(static_cast<int>(group_count), quantize_groups);
    } else {
        quantize_groups(0, static_cast<int>(group_count), 0);
    }

    const int64_t total_tiles = group_count * static_cast<int64_t>(tile_count);
    const auto compute_tiles = [&](int unit_start, int unit_end, int) {
        for (int unit = unit_start; unit < unit_end; ++unit) {
            const int64_t group = static_cast<int64_t>(unit) / tile_count;
            const int tile = unit % tile_count;
            const int64_t row = group * 4;
            const void* packed_weight = packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
            const uint8_t* q8x4_group = scratch.q8x4_groups.data() + static_cast<size_t>(group) * q8x4_group_bytes;
            float* output_tile = output_data + static_cast<size_t>(row) * static_cast<size_t>(output_cols) +
                                 static_cast<size_t>(tile) * 8;
            ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(input_cols), output_tile, static_cast<size_t>(output_cols),
                                    packed_weight, q8x4_group, 4, 8);
        }
    };
    if (thread_count > 1 && total_tiles >= 2) {
        pool.ParallelFor(static_cast<int>(total_tiles), compute_tiles);
    } else {
        compute_tiles(0, static_cast<int>(total_tiles), 0);
    }

    const int64_t tail_row = group_count * 4;
    if (tail_row == rows) {
        return true;
    }

    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, input_cols);
    if (!q8_traits || !q8_traits->from_float || q8_row_bytes == 0) {
        return false;
    }
    const int64_t tail_rows = rows - tail_row;
    const size_t tail_bytes = static_cast<size_t>(tail_rows) * q8_row_bytes;
    if (scratch.q8_tail.size() < tail_bytes) {
        scratch.q8_tail.resize(tail_bytes);
    }
    for (int64_t row = 0; row < tail_rows; ++row) {
        q8_traits->from_float(input_data + static_cast<size_t>(tail_row + row) * static_cast<size_t>(input_cols),
                              scratch.q8_tail.data() + static_cast<size_t>(row) * q8_row_bytes, input_cols);
    }
    return RunQ4KRepackedDenseGemv(backend, packed, scratch.q8_tail.data(), q8_row_bytes,
                                   output_data + static_cast<size_t>(tail_row) * static_cast<size_t>(output_cols),
                                   tail_rows, output_cols, numa_node, allow_parallel);
}

}  // namespace densecore::internal
