using MoEQ4Kx8Block = densecore::kernels::Q4KRepackedGemvBlock;

struct MoEQ5Kx8Block {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[96];
    uint8_t qh[QK_K];
    uint8_t qs[QK_K * 4];
};
static_assert(sizeof(MoEQ5Kx8Block) == 16 * sizeof(uint16_t) + 8 * K_SCALE_SIZE + QK_K * 5,
              "MoE Q5_Kx8 block layout must match ggml block_q5_Kx8");

size_t GetRepackedMoECacheLimitBytes() {
    constexpr size_t kMinBytes = 1024ull * 1024ull * 1024ull;
    constexpr size_t kMaxBytes = 4ull * 1024ull * 1024ull * 1024ull;
#if defined(__linux__)
    struct sysinfo info {};
    if (sysinfo(&info) == 0 && info.mem_unit > 0) {
        const uint64_t unit = static_cast<uint64_t>(info.mem_unit);
        const uint64_t free_bytes =
            (static_cast<uint64_t>(info.freeram) + static_cast<uint64_t>(info.bufferram)) * unit;
        const size_t target = static_cast<size_t>(free_bytes / 8);
        return std::clamp(target, kMinBytes, kMaxBytes);
    }
#endif
    return 2ull * 1024ull * 1024ull * 1024ull;
}

struct RepackedMoEKey {
    const void* weight_ptr = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    uint64_t fingerprint = 0;

    bool operator==(const RepackedMoEKey& other) const {
        return weight_ptr == other.weight_ptr && rows == other.rows && cols == other.cols &&
               fingerprint == other.fingerprint;
    }
};

struct RepackedMoEKeyHash {
    size_t operator()(const RepackedMoEKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight_ptr);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.fingerprint) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

uint64_t FingerprintRepackedMoEWeight(const void* weight_ptr, size_t bytes) {
    if (!weight_ptr || bytes == 0) {
        return 0;
    }
    const auto* data = static_cast<const uint8_t*>(weight_ptr);
    constexpr size_t kWindow = 4096;
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const uint8_t* ptr, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            h ^= static_cast<uint64_t>(ptr[i]);
            h *= 1099511628211ull;
        }
    };
    mix(reinterpret_cast<const uint8_t*>(&bytes), sizeof(bytes));
    if (bytes <= kWindow * 2) {
        mix(data, bytes);
    } else {
        mix(data, kWindow);
        mix(data + bytes - kWindow, kWindow);
    }
    return h;
}

using Q4KRepackedMoEWeight = densecore::kernels::Q4KRepackedGemvWeight;

std::shared_ptr<Q4KRepackedMoEWeight> GetOrCreateQ4KRepackedMoEWeight(const void* weight_ptr, int64_t rows,
                                                                      int64_t cols) {
    if (!weight_ptr || rows <= 0 || cols <= 0 || (rows % 8) != 0 || (cols % QK_K) != 0 ||
        !(CanUseQ4KRepackedMoEGemvFastPath() || CanUseQ4KRepackedMoEPrefillFastPath() ||
          CanUseQ4KRepackedMoEGEGLUFastPath())) {
        return nullptr;
    }
    return densecore::kernels::GetOrCreateQ4KRepackedGemvWeight(weight_ptr, rows, cols);
}

struct Q5KRepackedMoEWeight {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t bytes = 0;
    uint64_t last_use = 0;
    std::vector<MoEQ5Kx8Block> blocks;
};

constexpr size_t kMoEQ6Kx8BlockBytes = sizeof(ggml_fp16_t) * 8 + QK_K / 16 * 8 + 3 * QK_K / 4 * 8;

struct Q6KRepackedMoEWeight {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t bytes = 0;
    uint64_t last_use = 0;
    std::vector<uint8_t> blocks;
};

std::shared_ptr<Q5KRepackedMoEWeight> GetOrCreateQ5KRepackedMoEWeight(const void* weight_ptr, int64_t rows,
                                                                      int64_t cols) {
    if (!weight_ptr || rows <= 0 || cols <= 0 || (rows % 8) != 0 || (cols % QK_K) != 0 ||
        !CanUseQ5KRepackedMoEGemvFastPath()) {
        return nullptr;
    }
    const size_t cache_limit = GetRepackedMoECacheLimitBytes();
    if (cache_limit == 0) {
        return nullptr;
    }

    static std::mutex mutex;
    static std::unordered_map<RepackedMoEKey, std::shared_ptr<Q5KRepackedMoEWeight>, RepackedMoEKeyHash> cache;
    static size_t cache_bytes = 0;
    static std::atomic<uint64_t> use_clock{0};

    const size_t raw_bytes = static_cast<size_t>(rows) * ggml_row_size(GGML_TYPE_Q5_K, cols);
    const RepackedMoEKey key{weight_ptr, rows, cols, FingerprintRepackedMoEWeight(weight_ptr, raw_bytes)};
    const uint64_t now = use_clock.fetch_add(1, std::memory_order_relaxed) + 1;

    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found != cache.end()) {
        found->second->last_use = now;
        return found->second;
    }

    const int64_t blocks_per_row = cols / QK_K;
    const size_t packed_blocks = static_cast<size_t>(rows / 8) * static_cast<size_t>(blocks_per_row);
    auto packed = std::make_shared<Q5KRepackedMoEWeight>();
    packed->rows = rows;
    packed->cols = cols;
    packed->blocks_per_row = blocks_per_row;
    packed->blocks.resize(packed_blocks);
    packed->bytes = packed->blocks.size() * sizeof(MoEQ5Kx8Block);
    packed->last_use = now;

    if (ggml_repack_q5_K_8x8(weight_ptr, raw_bytes, rows, cols, packed->blocks.data(), packed->bytes) != 0) {
        return nullptr;
    }

    cache.emplace(key, packed);
    cache_bytes += packed->bytes;
    while (cache_bytes > cache_limit && cache.size() > 1) {
        auto oldest = cache.end();
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->first == key) {
                continue;
            }
            if (oldest == cache.end() || it->second->last_use < oldest->second->last_use) {
                oldest = it;
            }
        }
        if (oldest == cache.end()) {
            break;
        }
        cache_bytes -= oldest->second->bytes;
        cache.erase(oldest);
    }

    return packed;
}

std::shared_ptr<Q6KRepackedMoEWeight> GetOrCreateQ6KRepackedMoEWeight(const void* weight_ptr, int64_t rows,
                                                                      int64_t cols) {
    if (!weight_ptr || rows <= 0 || cols <= 0 || (rows % 8) != 0 || (cols % QK_K) != 0 ||
        !CanUseQ6KRepackedMoEGemvFastPath()) {
        return nullptr;
    }
    const size_t cache_limit = GetRepackedMoECacheLimitBytes();
    if (cache_limit == 0) {
        return nullptr;
    }

    static std::mutex mutex;
    static std::unordered_map<RepackedMoEKey, std::shared_ptr<Q6KRepackedMoEWeight>, RepackedMoEKeyHash> cache;
    static size_t cache_bytes = 0;
    static std::atomic<uint64_t> use_clock{0};

    const size_t raw_bytes = static_cast<size_t>(rows) * ggml_row_size(GGML_TYPE_Q6_K, cols);
    const RepackedMoEKey key{weight_ptr, rows, cols, FingerprintRepackedMoEWeight(weight_ptr, raw_bytes)};
    const uint64_t now = use_clock.fetch_add(1, std::memory_order_relaxed) + 1;

    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found != cache.end()) {
        found->second->last_use = now;
        return found->second;
    }

    const int64_t blocks_per_row = cols / QK_K;
    const size_t packed_blocks = static_cast<size_t>(rows / 8) * static_cast<size_t>(blocks_per_row);
    auto packed = std::make_shared<Q6KRepackedMoEWeight>();
    packed->rows = rows;
    packed->cols = cols;
    packed->blocks_per_row = blocks_per_row;
    packed->bytes = packed_blocks * kMoEQ6Kx8BlockBytes;
    packed->blocks.resize(packed->bytes);
    packed->last_use = now;

    if (ggml_repack_q6_K_8x8(weight_ptr, raw_bytes, rows, cols, packed->blocks.data(), packed->bytes) != 0) {
        return nullptr;
    }

    cache.emplace(key, packed);
    cache_bytes += packed->bytes;
    while (cache_bytes > cache_limit && cache.size() > 1) {
        auto oldest = cache.end();
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->first == key) {
                continue;
            }
            if (oldest == cache.end() || it->second->last_use < oldest->second->last_use) {
                oldest = it;
            }
        }
        if (oldest == cache.end()) {
            break;
        }
        cache_bytes -= oldest->second->bytes;
        cache.erase(oldest);
    }

    return packed;
}

bool RunQ4KRepackedMoEGemv(CpuBackend* backend, const std::shared_ptr<Q4KRepackedMoEWeight>& packed,
                           const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data, int64_t rows,
                           int64_t cols, int numa_node, bool allow_parallel) {
    if (!backend || !packed || !qinput_data || !output_data || packed->rows != cols || (cols % 8) != 0 ||
        packed->cols <= 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const auto compute_tiles = [&](int tile_start, int tile_end) {
        return densecore::kernels::RunQ4KRepackedGemvRows(packed, qinput_data, qinput_row_bytes, output_data, rows,
                                                          cols, tile_start, tile_end);
    };
    if (n_threads > 1 && tile_count >= 2) {
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

bool RunQ4KRepackedMoEGemmM4(CpuBackend* backend, const std::shared_ptr<Q4KRepackedMoEWeight>& packed,
                             const float* input_data, float* output_data, int64_t rows, int64_t cols,
                             int64_t input_cols, int numa_node, bool allow_parallel) {
    if (!backend || !packed || !input_data || !output_data || rows < 4 || packed->rows != cols || (cols % 8) != 0 ||
        packed->cols <= 0 || input_cols != packed->cols || (input_cols % QK_K) != 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(packed->blocks_per_row);
    const size_t q8_tile_bytes = static_cast<size_t>(input_cols / QK_K) * sizeof(MoEBlockQ8Kx4);

    static thread_local std::vector<uint8_t> q8x4_buf;
    if (q8x4_buf.size() < q8_tile_bytes) {
        q8x4_buf.resize(q8_tile_bytes);
    }

    int64_t row = 0;
    for (; row + 3 < rows; row += 4) {
        const float* input_tile = input_data + static_cast<size_t>(row) * static_cast<size_t>(input_cols);
        ggml_quantize_mat_q8_K_4x8(input_tile, q8x4_buf.data(), input_cols);
        float* out_tile = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            if (tile_start >= tile_end) {
                return;
            }
            const void* vx = packed->blocks.data() + static_cast<size_t>(tile_start) * blocks_per_row;
            ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(packed->cols), out_tile + static_cast<size_t>(tile_start) * 8,
                                    static_cast<size_t>(cols), vx, q8x4_buf.data(), 4, (tile_end - tile_start) * 8);
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    if (row < rows) {
        const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, input_cols);
        static thread_local std::vector<uint8_t> q8_tail_buf;
        const size_t tail_bytes = static_cast<size_t>(rows - row) * q8_row_bytes;
        if (q8_tail_buf.size() < tail_bytes) {
            q8_tail_buf.resize(tail_bytes);
        }
        const auto* iq_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        if (!iq_traits || !iq_traits->from_float) {
            return false;
        }
        for (int64_t m = row; m < rows; ++m) {
            iq_traits->from_float(input_data + static_cast<size_t>(m) * static_cast<size_t>(input_cols),
                                  q8_tail_buf.data() + static_cast<size_t>(m - row) * q8_row_bytes, input_cols);
        }
        return RunQ4KRepackedMoEGemv(backend, packed, q8_tail_buf.data(), q8_row_bytes,
                                     output_data + static_cast<size_t>(row) * static_cast<size_t>(cols), rows - row,
                                     cols, numa_node, allow_parallel);
    }

    return true;
}

bool RunQ4KRepackedMoEFusedSwiGLUM4(CpuBackend* backend, const std::shared_ptr<Q4KRepackedMoEWeight>& gate_packed,
                                    const std::shared_ptr<Q4KRepackedMoEWeight>& up_packed, const float* input_data,
                                    const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data,
                                    int64_t rows, int64_t cols, int64_t input_cols, int numa_node,
                                    bool allow_parallel) {
    if (!backend || !gate_packed || !up_packed || !qinput_data || !output_data || rows <= 0 ||
        gate_packed->rows != cols || up_packed->rows != cols || gate_packed->cols <= 0 ||
        gate_packed->cols != up_packed->cols || gate_packed->blocks_per_row != up_packed->blocks_per_row ||
        (cols % 8) != 0 || input_cols != gate_packed->cols || (input_cols % QK_K) != 0 ||
        (rows >= 4 && !input_data)) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(gate_packed->blocks_per_row);
    const size_t q8_tile_bytes = static_cast<size_t>(input_cols / QK_K) * sizeof(MoEBlockQ8Kx4);

    static thread_local std::vector<uint8_t> q8x4_buf;
    if (q8x4_buf.size() < q8_tile_bytes) {
        q8x4_buf.resize(q8_tile_bytes);
    }

    int64_t row = 0;
    for (; row + 3 < rows; row += 4) {
        const float* input_tile = input_data + static_cast<size_t>(row) * static_cast<size_t>(input_cols);
        ggml_quantize_mat_q8_K_4x8(input_tile, q8x4_buf.data(), input_cols);
        float* out_tile = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);

        const auto compute_tiles = [&](int tile_start, int tile_end) {
            std::array<float, 32> gate_tile{};
            std::array<float, 32> up_tile{};
            for (int tile = tile_start; tile < tile_end; ++tile) {
                const void* gate_vx = gate_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                const void* up_vx = up_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(gate_packed->cols), gate_tile.data(), 8, gate_vx,
                                        q8x4_buf.data(), 4, 8);
                ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(up_packed->cols), up_tile.data(), 8, up_vx, q8x4_buf.data(), 4,
                                        8);
                for (int r = 0; r < 4; ++r) {
                    float* out_row =
                        out_tile + static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(tile) * 8;
                    const float* gate_row = gate_tile.data() + static_cast<size_t>(r) * 8;
                    const float* up_row = up_tile.data() + static_cast<size_t>(r) * 8;
                    for (int c = 0; c < 8; ++c) {
                        const float gate_sum = gate_row[c];
                        out_row[c] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_row[c];
                    }
                }
            }
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    for (; row < rows; ++row) {
        const auto* qi = qinput_data + static_cast<size_t>(row) * qinput_row_bytes;
        float* out_row = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            std::array<float, 8> gate_tile{};
            std::array<float, 8> up_tile{};
            for (int tile = tile_start; tile < tile_end; ++tile) {
                const void* gate_vx = gate_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                const void* up_vx = up_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                ggml_gemv_q4_K_8x8_q8_K(static_cast<int>(gate_packed->cols), gate_tile.data(), 0, gate_vx, qi, 1, 8);
                ggml_gemv_q4_K_8x8_q8_K(static_cast<int>(up_packed->cols), up_tile.data(), 0, up_vx, qi, 1, 8);
                float* out = out_row + static_cast<size_t>(tile) * 8;
                for (int c = 0; c < 8; ++c) {
                    const float gate_sum = gate_tile[c];
                    out[c] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_tile[c];
                }
            }
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    return true;
}

bool RunQ4KRepackedMoEFusedGEGLU(CpuBackend* backend, const std::shared_ptr<Q4KRepackedMoEWeight>& gate_packed,
                                 const std::shared_ptr<Q4KRepackedMoEWeight>& up_packed, const float* input_data,
                                 const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data, int64_t rows,
                                 int64_t cols, int64_t input_cols, int numa_node, bool allow_parallel) {
    if (!backend || !gate_packed || !up_packed || !input_data || !qinput_data || !output_data || rows <= 0 ||
        gate_packed->rows != cols || up_packed->rows != cols || gate_packed->cols <= 0 ||
        gate_packed->cols != up_packed->cols || gate_packed->blocks_per_row != up_packed->blocks_per_row ||
        (cols % 8) != 0 || input_cols != gate_packed->cols || (input_cols % QK_K) != 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(gate_packed->blocks_per_row);
    const size_t q8_tile_bytes = static_cast<size_t>(input_cols / QK_K) * sizeof(MoEBlockQ8Kx4);

    static thread_local std::vector<uint8_t> q8x4_buf;
    if (q8x4_buf.size() < q8_tile_bytes) {
        q8x4_buf.resize(q8_tile_bytes);
    }

    int64_t row = 0;
    for (; row + 3 < rows; row += 4) {
        const float* input_tile = input_data + static_cast<size_t>(row) * static_cast<size_t>(input_cols);
        ggml_quantize_mat_q8_K_4x8(input_tile, q8x4_buf.data(), input_cols);
        float* out_tile = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);

        const auto compute_tiles = [&](int tile_start, int tile_end) {
            std::array<float, 32> gate_tile{};
            std::array<float, 32> up_tile{};
            for (int tile = tile_start; tile < tile_end; ++tile) {
                const void* gate_vx = gate_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                const void* up_vx = up_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(gate_packed->cols), gate_tile.data(), 8, gate_vx,
                                        q8x4_buf.data(), 4, 8);
                ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(up_packed->cols), up_tile.data(), 8, up_vx, q8x4_buf.data(), 4,
                                        8);
                for (int r = 0; r < 4; ++r) {
                    float* out_row =
                        out_tile + static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(tile) * 8;
                    const float* gate_row = gate_tile.data() + static_cast<size_t>(r) * 8;
                    const float* up_row = up_tile.data() + static_cast<size_t>(r) * 8;
                    for (int c = 0; c < 8; ++c) {
                        out_row[c] = GeluTanhApprox(gate_row[c]) * up_row[c];
                    }
                }
            }
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    for (; row < rows; ++row) {
        const auto* qi = qinput_data + static_cast<size_t>(row) * qinput_row_bytes;
        float* out_row = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            std::array<float, 8> gate_tile{};
            std::array<float, 8> up_tile{};
            for (int tile = tile_start; tile < tile_end; ++tile) {
                const void* gate_vx = gate_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                const void* up_vx = up_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                ggml_gemv_q4_K_8x8_q8_K(static_cast<int>(gate_packed->cols), gate_tile.data(), 0, gate_vx, qi, 1, 8);
                ggml_gemv_q4_K_8x8_q8_K(static_cast<int>(up_packed->cols), up_tile.data(), 0, up_vx, qi, 1, 8);
                float* out = out_row + static_cast<size_t>(tile) * 8;
                for (int c = 0; c < 8; ++c) {
                    out[c] = GeluTanhApprox(gate_tile[c]) * up_tile[c];
                }
            }
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    return true;
}

bool RunQ5KRepackedMoEGemv(CpuBackend* backend, const std::shared_ptr<Q5KRepackedMoEWeight>& packed,
                           const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data, int64_t rows,
                           int64_t cols, int numa_node, bool allow_parallel) {
    if (!backend || !packed || !qinput_data || !output_data || packed->rows != cols || (cols % 8) != 0 ||
        packed->cols <= 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(packed->blocks_per_row);

    for (int64_t row = 0; row < rows; ++row) {
        const auto* qi = qinput_data + static_cast<size_t>(row) * qinput_row_bytes;
        float* out = output_data + static_cast<size_t>(row) * cols;
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            if (tile_start >= tile_end) {
                return;
            }
            const void* vx = packed->blocks.data() + static_cast<size_t>(tile_start) * blocks_per_row;
            ggml_gemv_q5_K_8x8_q8_K(static_cast<int>(packed->cols), out + static_cast<size_t>(tile_start) * 8, 0, vx,
                                    qi, 1, (tile_end - tile_start) * 8);
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }
    return true;
}

bool RunQ6KRepackedMoEGemv(CpuBackend* backend, const std::shared_ptr<Q6KRepackedMoEWeight>& packed,
                           const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data, int64_t rows,
                           int64_t cols, int numa_node, bool allow_parallel) {
    if (!backend || !packed || !qinput_data || !output_data || packed->rows != cols || (cols % 8) != 0 ||
        packed->cols <= 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(packed->blocks_per_row);

    for (int64_t row = 0; row < rows; ++row) {
        const auto* qi = qinput_data + static_cast<size_t>(row) * qinput_row_bytes;
        float* out = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            if (tile_start >= tile_end) {
                return;
            }
            const void* vx = packed->blocks.data() + static_cast<size_t>(tile_start) *
                                                       static_cast<size_t>(blocks_per_row) * kMoEQ6Kx8BlockBytes;
            ggml_gemv_q6_K_8x8_q8_K(static_cast<int>(packed->cols),
                                    out + static_cast<size_t>(tile_start) * 8, 0, vx, qi, 1,
                                    (tile_end - tile_start) * 8);
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }
    return true;
}

bool RunQ5KRepackedMoEFusedSwiGLURawProjectionImpl(CpuBackend* backend, const void* gate_weight_ptr,
                                                   const void* up_weight_ptr, const float* input_data,
                                                   const uint8_t* qinput_data, size_t qinput_row_bytes,
                                                   float* output_data, int64_t rows, int64_t cols,
                                                   int64_t input_cols, int numa_node, bool allow_parallel) {
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !qinput_data || !output_data || rows <= 0 || cols <= 0 ||
        input_cols <= 0 || qinput_row_bytes == 0 || (cols % 8) != 0 || (input_cols % QK_K) != 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(input_cols / QK_K);
    const size_t q8_tile_bytes = static_cast<size_t>(input_cols / QK_K) * sizeof(MoEBlockQ8Kx4);
    const auto* gate_blocks = static_cast<const MoEQ5Kx8Block*>(gate_weight_ptr);
    const auto* up_blocks = static_cast<const MoEQ5Kx8Block*>(up_weight_ptr);

    static thread_local std::vector<uint8_t> q8x4_buf;
    if (q8x4_buf.size() < q8_tile_bytes) {
        q8x4_buf.resize(q8_tile_bytes);
    }

    bool used_gemm_m4 = false;
    int64_t row = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    for (; row + 3 < rows; row += 4) {
        if (!input_data) {
            return false;
        }
        used_gemm_m4 = true;
        const float* input_tile = input_data + static_cast<size_t>(row) * static_cast<size_t>(input_cols);
        ggml_quantize_mat_q8_K_4x8(input_tile, q8x4_buf.data(), input_cols);
        float* out_tile = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            std::array<float, 32> gate_tile{};
            std::array<float, 32> up_tile{};
            for (int tile = tile_start; tile < tile_end; ++tile) {
                const void* gate_vx = gate_blocks + static_cast<size_t>(tile) * blocks_per_row;
                const void* up_vx = up_blocks + static_cast<size_t>(tile) * blocks_per_row;
                ggml_gemm_q5_K_8x8_q8_K(static_cast<int>(input_cols), gate_tile.data(), 8, gate_vx,
                                         q8x4_buf.data(), 4, 8);
                ggml_gemm_q5_K_8x8_q8_K(static_cast<int>(input_cols), up_tile.data(), 8, up_vx, q8x4_buf.data(), 4,
                                         8);
                for (int r = 0; r < 4; ++r) {
                    float* out_row =
                        out_tile + static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(tile) * 8;
                    const float* gate_row = gate_tile.data() + static_cast<size_t>(r) * 8;
                    const float* up_row = up_tile.data() + static_cast<size_t>(r) * 8;
                    for (int c = 0; c < 8; ++c) {
                        const float gate_sum = gate_row[c];
                        out_row[c] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_row[c];
                    }
                }
            }
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }
#else
    (void)input_data;
    if (rows >= 4) {
        return false;
    }
#endif

    for (; row < rows; ++row) {
        const auto* qi = qinput_data + static_cast<size_t>(row) * qinput_row_bytes;
        float* out_row = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            std::array<float, 8> gate_tile{};
            std::array<float, 8> up_tile{};
            for (int tile = tile_start; tile < tile_end; ++tile) {
                const void* gate_vx = gate_blocks + static_cast<size_t>(tile) * blocks_per_row;
                const void* up_vx = up_blocks + static_cast<size_t>(tile) * blocks_per_row;
                ggml_gemv_q5_K_8x8_q8_K(static_cast<int>(input_cols), gate_tile.data(), 0, gate_vx, qi, 1, 8);
                ggml_gemv_q5_K_8x8_q8_K(static_cast<int>(input_cols), up_tile.data(), 0, up_vx, qi, 1, 8);
                float* out = out_row + static_cast<size_t>(tile) * 8;
                for (int c = 0; c < 8; ++c) {
                    const float gate_sum = gate_tile[c];
                    out[c] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_tile[c];
                }
            }
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    InferenceWorkContext* census_ctx = GetCurrentWorkContext();
    RecordMoEQ5KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/true, nullptr);
    LogMoEMatmulPath(used_gemm_m4 ? "ggml_q5k_repacked_prefill_gemm_m4_fused_swiglu"
                                  : "ggml_q5k_repacked_fused_swiglu",
                     static_cast<int>(rows), static_cast<int>(input_cols), static_cast<int>(cols), 0,
                     allow_parallel);
    return true;
}

