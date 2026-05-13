struct Q8RepackedGemvWeight {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t bytes = 0;
    std::vector<uint8_t> data;
};

struct Q8RepackedGemvKey {
    const void* weight = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;

    bool operator==(const Q8RepackedGemvKey& other) const {
        return weight == other.weight && rows == other.rows && cols == other.cols;
    }
};

struct Q8RepackedGemvKeyHash {
    size_t operator()(const Q8RepackedGemvKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

constexpr int kQ8RepackedGemvMinOutputRows = 4096;

static bool IsGemma4SharedDenseFfnWeightName(const char* weight_name) {
    if (!weight_name) {
        return false;
    }
    return std::strstr(weight_name, ".ffn_gate.weight") || std::strstr(weight_name, ".ffn_up.weight") ||
           std::strstr(weight_name, ".ffn_down.weight");
}

static bool IsQ8RepackedGemvEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DISABLE_Q8_REPACKED_GEMV");
        if (env && env[0] != '\0' && std::strcmp(env, "0") != 0 &&
            std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0) {
            return false;
        }
#if defined(__aarch64__) || defined(_M_ARM64)
        return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
        const char* enable_env = std::getenv("DENSECORE_ENABLE_Q8_REPACKED_GEMV");
        return enable_env && enable_env[0] != '\0' && std::strcmp(enable_env, "0") != 0 &&
               std::strcmp(enable_env, "false") != 0 && std::strcmp(enable_env, "off") != 0;
#endif
    }();
    return enabled;
}

static std::shared_ptr<Q8RepackedGemvWeight> GetOrCreateQ8RepackedGemvWeight(const void* weight_data, int64_t rows,
                                                                              int64_t cols) {
    if (!weight_data || rows <= 0 || cols <= 0 || (rows % 4) != 0 || (cols % QK8_0) != 0 ||
        !IsQ8RepackedGemvEnabled()) {
        return nullptr;
    }
    static std::mutex mutex;
    static std::unordered_map<Q8RepackedGemvKey, std::shared_ptr<Q8RepackedGemvWeight>, Q8RepackedGemvKeyHash> cache;

    const Q8RepackedGemvKey key{weight_data, rows, cols};
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    const size_t src_bytes = static_cast<size_t>(rows) * ggml_row_size(GGML_TYPE_Q8_0, cols);
    const size_t block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
    const size_t dst_bytes = static_cast<size_t>(rows / 4) * static_cast<size_t>(cols / QK8_0) * block_bytes;
    auto packed = std::make_shared<Q8RepackedGemvWeight>();
    packed->rows = rows;
    packed->cols = cols;
    packed->blocks_per_row = cols / QK8_0;
    packed->bytes = dst_bytes;
    packed->data.resize(dst_bytes);
    if (ggml_repack_q8_0_4x8(weight_data, src_bytes, rows, cols, packed->data.data(), packed->data.size()) != 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex);
    auto [it, inserted] = cache.emplace(key, packed);
    return inserted ? packed : it->second;
}

/**
 * Custom callback for parallel GEMV (decode-phase)
 *
 * REFACTORED: Uses GGML_OP_CUSTOM signature to allow output tensor shape
 * to be independent of input tensor shape. This fixes memory corruption
 * when Qwen3 projections change dimensions (e.g., 1024 -> 2048).
 *
 * Signature: void (*)(struct ggml_tensor *dst, int ith, int nth, void
 * *userdata) Input tensor accessed via dst->src[0]
 */
void cb_gemv_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<GemvUserData*>(userdata);
    if (!ud || !ud->weight_tensor) return;

    // Extract input tensor from dst->src[0] (GGML_OP_CUSTOM convention)
    const struct ggml_tensor* src = dst->src[0];
    if (!src) return;

    // Extract weight tensor from dst->src[1] (reliable graph topology, not
    // userdata)
    const struct ggml_tensor* weight_tensor = dst->src[1];
    if (!weight_tensor || !weight_tensor->data) {
        fprintf(stderr, "CRITICAL: GEMV weight tensor is null or has no data\n");
        return;
    }

    // Get data pointers at runtime (guaranteed valid after GGML allocates)
    const float* x_f32 = reinterpret_cast<const float*>(src->data);
    const void* weight_data = weight_tensor->data;
    float* output = reinterpret_cast<float*>(dst->data);

    if (!x_f32 || !weight_data || !output) return;

    // ==========================================================================
    // DIMENSION VALIDATION (using weight tensor from graph, not stale userdata)
    // ==========================================================================
    const int K = static_cast<int>(weight_tensor->ne[1]);  // Output dimension from weight
    const int N = static_cast<int>(weight_tensor->ne[0]);  // Input dimension from weight

    // Validate output tensor matches expected K
    if (dst->ne[0] != K) {
        fprintf(stderr,
                "CRITICAL: GEMV buffer mismatch! dst->ne[0](%ld) != weight->ne[1](%d). "
                "Output tensor was sized incorrectly.\n",
                (long)dst->ne[0], K);
        return;
    }

    const ggml_type weight_type = weight_tensor->type;
    const char* weight_name = weight_tensor->name[0] ? weight_tensor->name : "(unnamed)";
    const bool debug_gemv_timing = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_GEMV_TIMING");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    const auto gemv_begin =
        (debug_gemv_timing && ith == 0) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto maybe_log_gemv_timing = [&](const char* path) {
        if (!debug_gemv_timing || ith != 0) {
            return;
        }
        static std::atomic<int> log_budget{0};
        const int current = log_budget.fetch_add(1, std::memory_order_relaxed);
        if (current >= 256) {
            return;
        }
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gemv_begin).count();
        std::fprintf(stderr,
                     "[GEMV_TIMING] weight=%s type=%s path=%s K_out=%d N_in=%d nth=%d ith0_ms=%.3f\n",
                     weight_name, ggml_type_name(weight_type), path ? path : "unknown", K, N, nth, ms);
    };

    // ==========================================================================
    // SHARED PRE-QUANTIZATION (token-position synchronized):
    // Thread 0 quantizes once per decode token position, other threads wait for
    // that position marker and reuse the same quantized buffer.
    // ==========================================================================
    const void* quant_input = nullptr;
    bool used_shared_quant_buffer = false;
    if (ud->input_quant_type != GGML_TYPE_F32) {
        const auto* input_type_traits = ggml_get_type_traits_cpu(ud->input_quant_type);
        if (input_type_traits && input_type_traits->from_float) {
            const size_t quant_input_size = ggml_row_size(ud->input_quant_type, N);
            if (quant_input_size > 0 && quant_input_size <= kMaxQuantInputBufferSize) {
                const BatchSpec* batch = GetCurrentBatch();
                const bool has_valid_stamp = ud->slot_id >= 0;
                const uint64_t expected_stamp =
                    ComputeGemvBatchedQuantStamp(batch, 1, ud->slot_id, src->data, weight_tensor->data);

                if (nth <= 1 || !has_valid_stamp) {
                    alignas(64) thread_local std::vector<uint8_t> quant_input_tls;
                    quant_input_tls.resize(quant_input_size);
                    input_type_traits->from_float(x_f32, quant_input_tls.data(), static_cast<int64_t>(N));
                    quant_input = quant_input_tls.data();
                } else {
                    if (!ud->quant_input_shared || !ud->quantized_stamp) {
                        alignas(64) thread_local std::vector<uint8_t> quant_input_tls;
                        quant_input_tls.resize(quant_input_size);
                        input_type_traits->from_float(x_f32, quant_input_tls.data(), static_cast<int64_t>(N));
                        quant_input = quant_input_tls.data();
                    } else if (ith == 0) {
                        input_type_traits->from_float(x_f32, ud->quant_input_shared, static_cast<int64_t>(N));
                        ud->quantized_stamp->store(expected_stamp, std::memory_order_release);
                        quant_input = ud->quant_input_shared;
                        used_shared_quant_buffer = true;
                    } else {
                        int spin_count = 0;
                        while (ud->quantized_stamp->load(std::memory_order_acquire) != expected_stamp) {
                            SpinPause(spin_count++);
                        }
                        quant_input = ud->quant_input_shared;
                        used_shared_quant_buffer = true;
                    }
                }
            }
        }
    }
    if (ith == 0 && quant_input) {
        RecordSharedQuantReuse(used_shared_quant_buffer);
    }

    // Partition output dimension across threads
    const int k_per_thread = (K + nth - 1) / nth;
    const int k_start = ith * k_per_thread;
    const int k_end = std::min(k_start + k_per_thread, K);

    if (k_start >= K) return;

    // ==========================================================================
    // CASE A: FP32 weights - use optimized simd::GemvParallel
    // ==========================================================================
    if (weight_type == GGML_TYPE_F32) {
        if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0) {
            LogHybridSSMQkvDispatch(weight_name, weight_type, 1, K, N, "GEMV_F32_CALLBACK", false, false, false);
        }
        const float* weight = reinterpret_cast<const float*>(weight_data);
        densecore::simd::GemvParallel(output, x_f32, weight, N, K, ith, nth);
        maybe_log_gemv_timing("f32");
        return;
    }

    // ==========================================================================
    // CASE B: Quantized weights with pre-quantized input - use native vec_dot
    // ==========================================================================
    const size_t row_stride = weight_tensor->nb[1];  // Bytes per row
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight_type);
    const void* sample_row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k_start) * row_stride;
    const bool allow_native_q4k_vecdot =
        ShouldUseArmNativeQ4KVecDotValidated(weight_type, type_traits_cpu, weight_name, sample_row_ptr, quant_input,
                                             x_f32, N);

    if (!ud->force_reference_scalar && quant_input && type_traits_cpu && type_traits_cpu->vec_dot &&
        allow_native_q4k_vecdot) {
        if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0) {
            LogHybridSSMQkvDispatch(weight_name, weight_type, 1, K, N, "GEMV_NATIVE_VECDOT_CALLBACK", false, true,
                                    false);
        }
        const int q8_repacked_min_rows =
            IsGemma4SharedDenseFfnWeightName(weight_name)
                ? ParsePositiveEnvInt("DENSECORE_GEMMA4_SHARED_Q8_REPACKED_MIN_ROWS",
                                      kQ8RepackedGemvMinOutputRows)
                : kQ8RepackedGemvMinOutputRows;
        if (!ud->disable_q8_repacked_gemv && weight_type == GGML_TYPE_Q8_0 && ud->input_quant_type == GGML_TYPE_Q8_0 &&
            K >= q8_repacked_min_rows &&
            (N % QK8_0) == 0 && (K % 4) == 0) {
            auto packed = GetOrCreateQ8RepackedGemvWeight(weight_data, K, N);
            if (packed && packed->blocks_per_row > 0) {
                const size_t block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
                int k = k_start;
                for (; k < k_end && (k & 3); ++k) {
                    const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                    type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
                }
                const int packed_end = k_end & ~3;
                if (k < packed_end) {
                    const size_t packed_offset =
                        static_cast<size_t>(k / 4) * static_cast<size_t>(packed->blocks_per_row) * block_bytes;
                    ggml_gemv_q8_0_4x8_q8_0(N, output + k, 0, packed->data.data() + packed_offset, quant_input, 1,
                                            packed_end - k);
                    k = packed_end;
                }
                for (; k < k_end; ++k) {
                    const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                    type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
                }
                maybe_log_gemv_timing("q8_0_repacked_4x8");
                return;
            }
        }
        static const bool q8_rowpair_enabled = []() {
            const char* env = std::getenv("DENSECORE_ENABLE_Q8_ROWPAIR_VEC_DOT");
            return env && env[0] != '\0' && std::strcmp(env, "0") != 0 &&
                   std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0;
        }();
        const bool can_use_rowpair =
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_MATMUL_INT8)
            ((weight_type == GGML_TYPE_Q4_K && ud->input_quant_type == GGML_TYPE_Q8_K) ||
             (q8_rowpair_enabled && weight_type == GGML_TYPE_Q8_0 && ud->input_quant_type == GGML_TYPE_Q8_0)) &&
            (N % ggml_blck_size(weight_type)) == 0;
#else
            false;
#endif
        int k = k_start;
        if (can_use_rowpair && (k & 1)) {
            const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
            type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
            ++k;
        }
        if (can_use_rowpair) {
            for (; k + 1 < k_end; k += 2) {
                const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                type_traits_cpu->vec_dot(N, sums, 2, row_ptr, row_stride, quant_input, 0, 2);
                output[k] = sums[0];
                output[k + 1] = sums[1];
            }
        }
        for (; k < k_end; ++k) {
            const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
            type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
        }
        maybe_log_gemv_timing(can_use_rowpair ? "quant_vecdot_rowpair" : "quant_vecdot");
        return;
    }

    // ==========================================================================
    // CASE C: Fallback - dequantize weights (no pre-quantized input available)
    // ==========================================================================
    thread_local std::vector<float> dequant_buffer_tls;
    const auto* type_traits = ggml_get_type_traits(weight_type);
    if (!type_traits || !type_traits->to_float || N > static_cast<int>(kMaxDequantBufferSize)) {
        for (int k = k_start; k < k_end; k++) output[k] = 0.0f;
        return;
    }
    if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0) {
        LogHybridSSMQkvDispatch(weight_name, weight_type, 1, K, N, "GEMV_DEQUANT_REFERENCE_CALLBACK", false, false,
                                false);
    }
    dequant_buffer_tls.resize(static_cast<size_t>(N));
    float* dequant_buffer = dequant_buffer_tls.data();

    for (int k = k_start; k < k_end; k++) {
        const void* row_ptr = reinterpret_cast<const char*>(weight_data) + k * row_stride;
        type_traits->to_float(row_ptr, dequant_buffer, N);
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += x_f32[i] * dequant_buffer[i];
        }
        output[k] = sum;
    }
    maybe_log_gemv_timing("dequant_reference");
}

struct DensecoreBlockQ8K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(DensecoreBlockQ8K) == sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t),
              "DensecoreBlockQ8K layout mismatch");

// True-batched Q4_K x Q8_K row dot:
// - Reuses Q4_K decode/scales once per weight row.
// - Computes all M column dots in one pass.
static inline bool ComputeQ4KQ8KBatchedRowScalar(const void* weight_row, const uint8_t* quant_input_base,
                                                 size_t quant_row_stride, int M, int N, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums) return false;
    if (M <= 0 || M > kMaxSmallBatchColsHard) return false;
    if (N <= 0 || (N % QK_K) != 0) return false;
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        return false;
    }

    const auto* x_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    const int nb = N / QK_K;
    float lane_acc[kMaxSmallBatchColsHard][8];
    float min_acc[kMaxSmallBatchColsHard];
    std::memset(lane_acc, 0, sizeof(lane_acc));
    std::memset(min_acc, 0, sizeof(min_acc));

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;

    int8_t unpacked_q4[QK_K];
    uint32_t utmp[4];
    int32_t dot_chunks[8];

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = x_blocks[bi];
        const uint8_t* q4 = xb.qs;
        int8_t* uq4 = unpacked_q4;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) uq4[l] = static_cast<int8_t>(q4[l] & 0xF);
            uq4 += 32;
            for (int l = 0; l < 32; ++l) uq4[l] = static_cast<int8_t>(q4[l] >> 4);
            uq4 += 32;
            q4 += 32;
        }

        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
        const float x_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float x_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        for (int m = 0; m < M; ++m) {
            const auto* y_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = y_blocks[bi];

            int32_t sumi = 0;
            for (int j = 0; j < QK_K / 16; ++j) {
                sumi += static_cast<int32_t>(yb.bsums[j]) * static_cast<int32_t>(mins[j / 2]);
            }

            std::memset(dot_chunks, 0, sizeof(dot_chunks));
            const int8_t* q8 = yb.qs;
            const int8_t* uq = unpacked_q4;
            int is = 0;
            for (int j = 0; j < QK_K / 32; ++j) {
                const int32_t scale = static_cast<int32_t>(scales[is++]);
                for (int rep = 0; rep < 4; ++rep) {
                    for (int l = 0; l < 8; ++l) {
                        dot_chunks[l] += scale * (static_cast<int32_t>(q8[l]) * static_cast<int32_t>(uq[l]));
                    }
                    q8 += 8;
                    uq += 8;
                }
            }

            const float yd = yb.d;
            const float d = x_d * yd;
            const float dmin = x_dmin * yd;
            for (int l = 0; l < 8; ++l) {
                lane_acc[m][l] += d * static_cast<float>(dot_chunks[l]);
            }
            min_acc[m] -= dmin * static_cast<float>(sumi);
        }
    }

    for (int m = 0; m < M; ++m) {
        float sum = min_acc[m];
        for (int l = 0; l < 8; ++l) {
            sum += lane_acc[m][l];
        }
        out_sums[m] = sum;
    }

    return true;
}

#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
static inline __m256i M256SetM128i(const __m128i hi, const __m128i lo) {
    return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline float HSumFloat8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i GetScaleShuffleK4(int i) {
    static const uint8_t k_shuffle[256] = {
        0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
        0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
        2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
        4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,  6,  7,  6,  7,
        6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,
        8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
        8,  9,  8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_shuffle) + i);
}

static inline bool ComputeQ4KQ8KBatchedRowAvx2(const void* weight_row, const uint8_t* quant_input_base,
                                               size_t quant_row_stride, int M, int N, float* out_sums) {
    const bool debug_q4k_path = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_PATH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    static std::atomic<bool> logged_invalid_args{false};
    static std::atomic<bool> logged_invalid_m{false};
    static std::atomic<bool> logged_invalid_n{false};
    static std::atomic<bool> logged_stride{false};
    static std::atomic<bool> logged_qkk{false};

    if (!weight_row || !quant_input_base || !out_sums) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_invalid_args.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable invalid_args" << std::endl;
            }
        }
        return false;
    }
    if (M <= 0 || M > kMaxSmallBatchColsHard) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_invalid_m.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable invalid_m M=" << M << std::endl;
            }
        }
        return false;
    }
    if (N <= 0 || (N % QK_K) != 0) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_invalid_n.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable invalid_n N=" << N << std::endl;
            }
        }
        return false;
    }
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_stride.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable stride stride=" << quant_row_stride << " need="
                          << (static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K))
                          << std::endl;
            }
        }
        return false;
    }
    if (QK_K != 256) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_qkk.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable QK_K=" << QK_K << std::endl;
            }
        }
        return false;
    }
    const auto* x_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    const int nb = N / QK_K;
    alignas(64) std::array<float, kMaxSmallBatchColsHard> sums{};

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = x_blocks[bi];

        uint32_t utmp[4];
        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = M256SetM128i(sc128, sc128);

        const float x_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float x_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        // Reuse decoded q4 nibble vectors/scales for all M columns.
        __m256i q4l[QK_K / 64];
        __m256i q4h[QK_K / 64];
        __m256i scale_l[QK_K / 64];
        __m256i scale_h[QK_K / 64];
        const uint8_t* q4 = xb.qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, GetScaleShuffleK4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, GetScaleShuffleK4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }

        for (int m = 0; m < M; ++m) {
            const auto* y_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = y_blocks[bi];

            const float yd = yb.d;
            const float d = x_d * yd;
            const float dmin = -x_dmin * yd;

            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb.bsums));
            const __m128i q8s =
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            __m128i sum32 = _mm_hadd_epi32(prod, prod);
            sum32 = _mm_hadd_epi32(sum32, sum32);
            const int32_t min_dot = _mm_cvtsi128_si32(sum32);

            const int8_t* q8 = yb.qs;
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);

                const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }

            sums[static_cast<size_t>(m)] +=
                d * HSumFloat8(_mm256_cvtepi32_ps(sumi)) + dmin * static_cast<float>(min_dot);
        }
    }

    for (int m = 0; m < M; ++m) {
        out_sums[m] = sums[static_cast<size_t>(m)];
    }
    return true;
}
#endif

static inline bool ComputeQ4KQ8KBatchedRow(const void* weight_row, const uint8_t* quant_input_base,
                                           size_t quant_row_stride, int M, int N, float* out_sums) {
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    static const bool debug_q4k_path = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_PATH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    static std::atomic<bool> logged_avx2{false};
    static std::atomic<bool> logged_scalar{false};
    if (IsQ4KTrueBatchedAvx2Enabled() &&
        ComputeQ4KQ8KBatchedRowAvx2(weight_row, quant_input_base, quant_row_stride, M, N, out_sums)) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_avx2.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2" << std::endl;
            }
        }
        return true;
    }
    if (debug_q4k_path) {
        bool expected = false;
        if (logged_scalar.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
            std::cerr << "[Q4K_BATCHED_PATH] scalar" << std::endl;
        }
    }
#endif
    return ComputeQ4KQ8KBatchedRowScalar(weight_row, quant_input_base, quant_row_stride, M, N, out_sums);
}

/**
 * Custom callback for small-batch GEMV/GEMM hybrid (2 <= M <= 8).
 *
 * Computes output[:, m] = weight @ input[:, m] for all m in [0, M) while
 * reusing each weight row across M tokens before evicting it from cache.
 */
void cb_gemv_batched_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto profile_begin =
        (ith == 0 && IsQwen36ProfilingEnabled()) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto* ud = static_cast<GemvBatchedUserData*>(userdata);
    if (!ud || !ud->weight_tensor) return;
    if (nth <= 0) return;
    if (!dst || !dst->src[0] || !dst->src[1]) return;

    const struct ggml_tensor* src = dst->src[0];
    const struct ggml_tensor* weight_tensor = dst->src[1];
    if (!src->data || !weight_tensor->data || !dst->data) return;
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return;

    const int N = static_cast<int>(src->ne[0]);
    const int M = static_cast<int>(src->ne[1]);
    const int K = static_cast<int>(weight_tensor->ne[1]);
    if (N <= 0 || M <= 0 || K <= 0) return;
    if (M != ud->M || K != ud->K || N != ud->N) return;
    if (static_cast<int>(dst->ne[0]) != K || static_cast<int>(dst->ne[1]) != M) return;

    const int k_per_thread = (K + nth - 1) / nth;
    const int k_start = ith * k_per_thread;
    const int k_end = std::min(K, k_start + k_per_thread);
    if (k_start >= k_end) return;

    const bool input_contig = src->nb[0] == sizeof(float);
    const bool output_contig = dst->nb[0] == sizeof(float);
    const char* input_base = reinterpret_cast<const char*>(src->data);
    char* output_base = reinterpret_cast<char*>(dst->data);
    const size_t input_col_stride = static_cast<size_t>(src->nb[1]);
    const size_t output_col_stride = static_cast<size_t>(dst->nb[1]);
    const size_t weight_row_stride = static_cast<size_t>(weight_tensor->nb[1]);
    const ggml_type weight_type = weight_tensor->type;
    const char* weight_name = weight_tensor->name[0] ? weight_tensor->name : "(unnamed)";
    const auto record_quant_profile = [&](bool used_quantized, bool used_true_batched) {
        if (ith != 0 || profile_begin == std::chrono::steady_clock::time_point()) {
            return;
        }
        InferenceWorkContext* work_ctx = GetCurrentWorkContext();
        AddQwen36ProfileNs(
            work_ctx->qwen36_profile.quant_matmul_ns,
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - profile_begin)
                    .count()));
        if (used_quantized && densecore::simd::IsArmFamily(GetRuntimeSimdLevel())) {
            MarkQwen36ProfileFlag(work_ctx->qwen36_profile.arm_batched_quant_used);
        }
        if (used_true_batched) {
            MarkQwen36ProfileFlag(work_ctx->qwen36_profile.q4k_true_batched_used);
        }
    };

    thread_local std::vector<const float*> x_rows;
    thread_local std::vector<float> gathered_inputs;
    thread_local std::vector<float> sums;
    x_rows.resize(static_cast<size_t>(M));
    sums.resize(static_cast<size_t>(M));

    if (input_contig) {
        for (int m = 0; m < M; ++m) {
            x_rows[static_cast<size_t>(m)] =
                reinterpret_cast<const float*>(input_base + static_cast<size_t>(m) * input_col_stride);
        }
    } else {
        gathered_inputs.resize(static_cast<size_t>(M) * static_cast<size_t>(N));
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * input_col_stride;
            float* dst_col = gathered_inputs.data() + static_cast<size_t>(m) * static_cast<size_t>(N);
            for (int i = 0; i < N; ++i) {
                dst_col[static_cast<size_t>(i)] =
                    *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * src->nb[0]);
            }
            x_rows[static_cast<size_t>(m)] = dst_col;
        }
    }

    auto store_out = [&](int m, int k, float value) {
        char* out_col = output_base + static_cast<size_t>(m) * output_col_stride;
        if (output_contig) {
            reinterpret_cast<float*>(out_col)[k] = value;
        } else {
            *reinterpret_cast<float*>(out_col + static_cast<size_t>(k) * dst->nb[0]) = value;
        }
    };

    const char* weight_base = reinterpret_cast<const char*>(weight_tensor->data);

    if (weight_type == GGML_TYPE_F32) {
        if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0) {
            LogHybridSSMQkvDispatch(weight_name, weight_type, M, K, N, "BATCHED_F32_CALLBACK", false, false, false);
        }
        // Fast path: contiguous layout -> Highway SIMD GEMM with Split-N parallelism.
        // GemmFP32_Hwy computes C[:, n_start:n_end) = A[M,K_gemm] x B[N_gemm,K_gemm]^T
        // Our mapping: A=input[M, N_input], B=weight[K_output, N_input], C=output[M, K_output]
        // GEMM K_gemm = N (input dim), GEMM N_gemm = K (output dim)
        // Split-N over K_output (the output dimension, which IS GEMM's N).
        const bool weight_contig = (weight_row_stride == static_cast<size_t>(N) * sizeof(float));
        if (input_contig && output_contig && weight_contig) {
            const float* A = reinterpret_cast<const float*>(input_base);
            const float* B = reinterpret_cast<const float*>(weight_base);
            float* C = reinterpret_cast<float*>(output_base);
            // k_start/k_end map to n_start/n_end in GEMM Split-N convention
            densecore::hwy_kernels::GemmFP32_Hwy(C, A, B, M, K, N, k_start, k_end);
            record_quant_profile(false, false);
            return;
        }

        // Strided fallback: scalar with weight row reuse
        for (int k = k_start; k < k_end; ++k) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            const float* w_row =
                reinterpret_cast<const float*>(weight_base + static_cast<size_t>(k) * weight_row_stride);
            for (int i = 0; i < N; ++i) {
                const float w = w_row[i];
                for (int m = 0; m < M; ++m) {
                    sums[static_cast<size_t>(m)] += x_rows[static_cast<size_t>(m)][i] * w;
                }
            }
            for (int m = 0; m < M; ++m) {
                store_out(m, k, sums[static_cast<size_t>(m)]);
            }
        }
        record_quant_profile(false, false);
        return;
    }

    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight_type);
    if (type_traits_cpu && type_traits_cpu->vec_dot) {
        const ggml_type vec_dot_type =
            (ud->input_quant_type != GGML_TYPE_F32) ? ud->input_quant_type : type_traits_cpu->vec_dot_type;
        const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
        const size_t quant_row_size = ggml_row_size(vec_dot_type, static_cast<int64_t>(N));
        const size_t quant_row_stride = (ud->quant_row_stride > 0)
                                            ? ud->quant_row_stride
                                            : densecore::AlignUp(quant_row_size, static_cast<size_t>(64));
        const int vec_dot_nrows = std::max<int>(1, static_cast<int>(type_traits_cpu->nrows));
        const bool can_quantize_inputs = input_type_traits && input_type_traits->from_float && quant_row_size > 0 &&
                                         quant_row_stride <= kMaxQuantInputBufferSize;
        const bool can_use_q4k_true_batched = can_quantize_inputs && weight_type == GGML_TYPE_Q4_K &&
                                              vec_dot_type == GGML_TYPE_Q8_K && IsQ4KTrueBatchedKernelEnabled() &&
                                              (N % QK_K == 0);
        const int quant_tile_cols = ResolveQuantBatchedTileCols(
            ParsePositiveEnvInt("DENSECORE_BATCHED_QUANT_TILE_COLS", kMaxSmallBatchColsHard), vec_dot_nrows,
            can_use_q4k_true_batched);

        if (can_quantize_inputs) {
            thread_local std::vector<uint8_t> quant_inputs_tls;
            const BatchSpec* batch = GetCurrentBatch();
            const bool can_sync_on_stamp =
                M <= quant_tile_cols && nth > 1 && ud->slot_id >= 0 && ud->quant_input_shared && ud->quantized_stamp;
            const uint64_t expected_stamp =
                can_sync_on_stamp ? ComputeGemvBatchedQuantStamp(batch, M, ud->slot_id, src->data, weight_tensor->data)
                                  : 0;
            bool logged_quant_reuse = false;

            for (int tile_start = 0; tile_start < M; tile_start += quant_tile_cols) {
                const int tile_m = std::min(quant_tile_cols, M - tile_start);
                const size_t quant_total_size = quant_row_stride * static_cast<size_t>(tile_m);
                const bool use_shared_quant_buffer = can_sync_on_stamp && tile_start == 0;
                const uint8_t* quant_input_base = nullptr;

                if (use_shared_quant_buffer) {
                    if (ith == 0) {
                        for (int m = 0; m < tile_m; ++m) {
                            uint8_t* q_ptr = ud->quant_input_shared + static_cast<size_t>(m) * quant_row_stride;
                            input_type_traits->from_float(x_rows[static_cast<size_t>(tile_start + m)], q_ptr,
                                                          static_cast<int64_t>(N));
                        }
                        ud->quantized_stamp->store(expected_stamp, std::memory_order_release);
                    } else {
                        int spin_count = 0;
                        while (ud->quantized_stamp->load(std::memory_order_acquire) != expected_stamp) {
                            SpinPause(spin_count++);
                        }
                    }
                    quant_input_base = ud->quant_input_shared;
                    if (ith == 0 && !logged_quant_reuse) {
                        RecordSharedQuantReuse(true);
                        logged_quant_reuse = true;
                    }
                } else {
                    quant_inputs_tls.resize(quant_total_size);
                    for (int m = 0; m < tile_m; ++m) {
                        uint8_t* q_ptr = quant_inputs_tls.data() + static_cast<size_t>(m) * quant_row_stride;
                        input_type_traits->from_float(x_rows[static_cast<size_t>(tile_start + m)], q_ptr,
                                                      static_cast<int64_t>(N));
                    }
                    quant_input_base = quant_inputs_tls.data();
                    if (ith == 0 && !logged_quant_reuse) {
                        RecordSharedQuantReuse(false);
                        logged_quant_reuse = true;
                    }
                }

                const bool can_use_quant_nrc_fast = output_contig && vec_dot_nrows >= tile_m;
                const void* sample_row_ptr = weight_base + static_cast<size_t>(k_start) * weight_row_stride;
                const bool allow_native_q4k_vecdot =
                    ShouldUseArmNativeQ4KVecDotValidated(weight_type, type_traits_cpu, weight_name, sample_row_ptr,
                                                         quant_input_base, x_rows[static_cast<size_t>(tile_start)], N);

                if (!ud->force_reference_scalar && can_use_quant_nrc_fast && quant_input_base &&
                    allow_native_q4k_vecdot) {
                    if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0 &&
                        tile_start == 0) {
                        LogHybridSSMQkvDispatch(weight_name, weight_type, M, K, N, "GGML_QUANT_NRC_M_CALLBACK", true,
                                                true, false);
                    }
                    for (int k = k_start; k < k_end; ++k) {
                        char* out_col = output_base + static_cast<size_t>(tile_start) * output_col_stride;
                        float* out_ptr = reinterpret_cast<float*>(out_col + static_cast<size_t>(k) * sizeof(float));
                        const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
                        type_traits_cpu->vec_dot(N, out_ptr, dst->nb[1], row_ptr, 0, quant_input_base, quant_row_stride,
                                                 tile_m);
                    }
                    LogMatmulPathOnce("gemv_batched_quant_nrc");
                    continue;
                }

                if (!ud->force_reference_scalar && can_use_q4k_true_batched && quant_input_base) {
                    alignas(64) std::array<float, kMaxSmallBatchColsHard> row_sums{};
                    static const bool debug_q4k_kernel_check = []() {
                        const char* env = std::getenv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_CHECK");
                        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
                    }();
                    static const float kDebugKernelWarnDiff = []() {
                        const char* env = std::getenv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_TOL");
                        if (!env || env[0] == '\0') return 1e-3f;
                        char* end = nullptr;
                        const float v = std::strtof(env, &end);
                        if (end == env || !std::isfinite(v) || v < 0.0f) return 1e-3f;
                        return v;
                    }();
                    static std::atomic<int> q4k_kernel_warn_count{0};
                    float max_abs_diff = 0.0f;
                    int max_diff_k = -1;
                    int max_diff_m = -1;
                    float max_diff_batched = 0.0f;
                    float max_diff_ref = 0.0f;
                    bool all_rows_ok = true;
                    for (int k = k_start; k < k_end; ++k) {
                        const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
                        if (!ComputeQ4KQ8KBatchedRow(row_ptr, quant_input_base, quant_row_stride, tile_m, N,
                                                     row_sums.data())) {
                            all_rows_ok = false;
                            break;
                        }

                        if (debug_q4k_kernel_check) {
                            for (int m = 0; m < tile_m; ++m) {
                                float ref = 0.0f;
                                const void* q_ptr = quant_input_base + static_cast<size_t>(m) * quant_row_stride;
                                type_traits_cpu->vec_dot(N, &ref, 0, row_ptr, 0, q_ptr, 0, 1);
                                const float diff = std::fabs(row_sums[static_cast<size_t>(m)] - ref);
                                if (diff > max_abs_diff) {
                                    max_abs_diff = diff;
                                    max_diff_k = k;
                                    max_diff_m = tile_start + m;
                                    max_diff_batched = row_sums[static_cast<size_t>(m)];
                                    max_diff_ref = ref;
                                }
                            }
                        }

                        for (int m = 0; m < tile_m; ++m) {
                            store_out(tile_start + m, k, row_sums[static_cast<size_t>(m)]);
                        }
                    }
                    if (debug_q4k_kernel_check && max_abs_diff > kDebugKernelWarnDiff) {
                        const int warn_idx = q4k_kernel_warn_count.fetch_add(1, std::memory_order_relaxed);
                        if (warn_idx < 32) {
                            std::cerr << "[Q4K_BATCHED_CHECK] WARN ith=" << ith << " max_abs_diff=" << max_abs_diff
                                      << " at(k,m)=" << max_diff_k << "," << max_diff_m
                                      << " batched=" << max_diff_batched << " ref=" << max_diff_ref << std::endl;
                        }
                    }
                    if (all_rows_ok) {
                        record_quant_profile(true, true);
                        continue;
                    }
                }

                for (int k = k_start; k < k_end; ++k) {
                    const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
                    for (int m = 0; m < tile_m; ++m) {
                        float sum = 0.0f;
                        const void* q_ptr = quant_input_base + static_cast<size_t>(m) * quant_row_stride;
                        type_traits_cpu->vec_dot(N, &sum, 0, row_ptr, 0, q_ptr, 0, 1);
                        store_out(tile_start + m, k, sum);
                    }
                }
            }
            record_quant_profile(true, false);
            return;
        }
    }

    if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0) {
        LogHybridSSMQkvDispatch(weight_name, weight_type, M, K, N, "BATCHED_DEQUANT_REFERENCE_CALLBACK", false, false,
                                false);
    }

    const auto* type_traits = ggml_get_type_traits(weight_type);
    if (!type_traits || !type_traits->to_float) {
        for (int k = k_start; k < k_end; ++k) {
            for (int m = 0; m < M; ++m) {
                store_out(m, k, 0.0f);
            }
        }
        return;
    }

    thread_local std::vector<float> dequant_row;
    dequant_row.resize(static_cast<size_t>(N));
    for (int k = k_start; k < k_end; ++k) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        const void* row_ptr =
            reinterpret_cast<const char*>(weight_tensor->data) + static_cast<size_t>(k) * weight_row_stride;
        type_traits->to_float(row_ptr, dequant_row.data(), N);
        for (int i = 0; i < N; ++i) {
            const float w = dequant_row[static_cast<size_t>(i)];
            for (int m = 0; m < M; ++m) {
                sums[static_cast<size_t>(m)] += x_rows[static_cast<size_t>(m)][i] * w;
            }
        }
        for (int m = 0; m < M; ++m) {
            store_out(m, k, sums[static_cast<size_t>(m)]);
        }
    }
    record_quant_profile(ggml_is_quantized(weight_type), false);
}

static void ComputeFlashAttentionReference(const float* q, const float* k, const float* v, float* out, int n_head,
                                           int n_head_kv, int seq_q, int seq_kv, int head_dim, float scale, bool causal,
                                           int q_start_offset, int kv_start_offset, int sliding_window,
                                           float logit_softcap = 0.0f);

static void ComputePagedAttentionScalarHeads(const PagedAttentionUserData* ud, const std::vector<int>& block_table,
                                             int context_len, const KVRetentionSpan& retained_span,
                                             const KVRetentionSpan& mask_span, int current_pos, float scale,
                                             const float* q_data, float* out_data, int h_start, int h_end) {
    if (!ud || !ud->cache || !q_data || !out_data) {
        return;
    }

    const int n_head_kv = ud->n_head_kv > 0 ? ud->n_head_kv : ud->cache->n_head_kv;
    const int n_head_total = ud->n_head;
    const int head_dim = ud->head_dim;
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    if (head_dim <= 0 || v_head_dim <= 0 || n_head_kv <= 0 || n_head_total <= 0 || (n_head_total % n_head_kv) != 0) {
        for (int h = h_start; h < h_end; ++h) {
            float* out_head = out_data + static_cast<size_t>(h) * v_head_dim;
            std::fill(out_head, out_head + v_head_dim, 0.0f);
        }
        return;
    }

    const int kv_group_size = n_head_total / n_head_kv;
    const auto k_layout = ud->cache->GetBlockLayout();
    const auto v_layout = ud->cache->GetVBlockLayout();
    if (k_layout.head_stride_bytes == 0 || k_layout.slot_stride_bytes == 0 || v_layout.head_stride_bytes == 0 ||
        v_layout.slot_stride_bytes == 0) {
        for (int h = h_start; h < h_end; ++h) {
            float* out_head = out_data + static_cast<size_t>(h) * v_head_dim;
            std::fill(out_head, out_head + v_head_dim, 0.0f);
        }
        return;
    }

    thread_local std::vector<const uint8_t*> k_blocks;
    thread_local std::vector<const uint8_t*> v_blocks;
    k_blocks.resize(block_table.size(), nullptr);
    v_blocks.resize(block_table.size(), nullptr);
    for (size_t bi = 0; bi < block_table.size(); ++bi) {
        const int block_id = block_table[bi];
        if (block_id < 0 || block_id >= ud->cache->max_blocks) {
            continue;
        }
        k_blocks[bi] = reinterpret_cast<const uint8_t*>(ud->cache->GetKBlockPtr(block_id, ud->read_layer));
        v_blocks[bi] = reinterpret_cast<const uint8_t*>(ud->cache->GetVBlockPtr(block_id, ud->read_layer));
    }

    thread_local std::vector<float> k_head_scratch;
    thread_local std::vector<float> v_head_scratch;
    thread_local std::vector<float> scores;
    k_head_scratch.resize(static_cast<size_t>(head_dim));
    v_head_scratch.resize(static_cast<size_t>(v_head_dim));
    scores.resize(static_cast<size_t>(context_len));

    const auto* quant_traits =
        ggml_is_quantized(k_layout.cache_type) ? ggml_get_type_traits(k_layout.cache_type) : nullptr;

    for (int h = h_start; h < h_end; ++h) {
        const float* q_head = q_data + static_cast<size_t>(h) * head_dim;
        float* out_head = out_data + static_cast<size_t>(h) * v_head_dim;
        std::fill(out_head, out_head + v_head_dim, 0.0f);

        int kv_head = h / kv_group_size;
        if (kv_head < 0) kv_head = 0;
        if (kv_head >= n_head_kv) kv_head = n_head_kv - 1;
        const size_t k_head_offset_bytes = static_cast<size_t>(kv_head) * k_layout.head_stride_bytes;
        const size_t v_head_offset_bytes = static_cast<size_t>(kv_head) * v_layout.head_stride_bytes;

        float max_score = -INFINITY;
        for (int t = 0; t < context_len; ++t) {
            const int token_pos =
                (t < retained_span.history_kept)
                    ? densecore::llm::config::MapRetainedHistoryIndex(retained_span, t)
                    : current_pos;
            const int mask_key_pos =
                (ud->sliding_window >= 0)
                    ? ((t < mask_span.history_kept) ? densecore::llm::config::MapRetainedHistoryIndex(mask_span, t)
                                                    : (current_pos + (t - mask_span.history_kept)))
                    : token_pos;
            if (mask_key_pos > current_pos ||
                (ud->sliding_window >= 0 && mask_key_pos < (current_pos - ud->sliding_window))) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot_idx = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }

            const uint8_t* k_block_base = k_blocks[static_cast<size_t>(logical_block)];
            if (!k_block_base) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }
            const uint8_t* k_ptr =
                k_block_base + static_cast<size_t>(slot_idx) * k_layout.slot_stride_bytes + k_head_offset_bytes;
            const float* k_head = nullptr;
            if (k_layout.cache_type == GGML_TYPE_F32) {
                k_head = reinterpret_cast<const float*>(k_ptr);
            } else if (k_layout.cache_type == GGML_TYPE_F16) {
                densecore::simd::ConvertF16ToF32(k_head_scratch.data(), reinterpret_cast<const ggml_fp16_t*>(k_ptr),
                                                 head_dim);
                k_head = k_head_scratch.data();
            } else if (ggml_is_quantized(k_layout.cache_type) && quant_traits && quant_traits->to_float) {
                quant_traits->to_float(k_ptr, k_head_scratch.data(), head_dim);
                k_head = k_head_scratch.data();
            }
            if (!k_head) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_head[d] * k_head[d];
            }
            float score = dot;
            if (ud->logit_softcap > 0.0f && std::isfinite(score)) {
                score = std::tanh(score / ud->logit_softcap) * ud->logit_softcap;
            }
            score *= scale;
            scores[static_cast<size_t>(t)] = score;
            if (std::isfinite(score) && score > max_score) {
                max_score = score;
            }
        }

        if (!std::isfinite(max_score)) {
            continue;
        }

        float denom = 0.0f;
        for (int t = 0; t < context_len; ++t) {
            const float score = scores[static_cast<size_t>(t)];
            if (!std::isfinite(score)) continue;

            const float weight = std::exp(score - max_score);
            if (!(weight > 0.0f) || !std::isfinite(weight)) continue;

            const int token_pos =
                (t < retained_span.history_kept)
                    ? densecore::llm::config::MapRetainedHistoryIndex(retained_span, t)
                    : current_pos;
            const int mask_key_pos =
                (ud->sliding_window >= 0)
                    ? ((t < mask_span.history_kept) ? densecore::llm::config::MapRetainedHistoryIndex(mask_span, t)
                                                    : (current_pos + (t - mask_span.history_kept)))
                    : token_pos;
            if (mask_key_pos > current_pos ||
                (ud->sliding_window >= 0 && mask_key_pos < (current_pos - ud->sliding_window))) {
                continue;
            }
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot_idx = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) continue;
            const uint8_t* v_block_base = v_blocks[static_cast<size_t>(logical_block)];
            if (!v_block_base) continue;

            const uint8_t* v_ptr =
                v_block_base + static_cast<size_t>(slot_idx) * v_layout.slot_stride_bytes + v_head_offset_bytes;
            const float* v_head = nullptr;
            if (v_layout.cache_type == GGML_TYPE_F32) {
                v_head = reinterpret_cast<const float*>(v_ptr);
            } else if (v_layout.cache_type == GGML_TYPE_F16) {
                densecore::simd::ConvertF16ToF32(v_head_scratch.data(), reinterpret_cast<const ggml_fp16_t*>(v_ptr),
                                                 v_head_dim);
                v_head = v_head_scratch.data();
            } else if (ggml_is_quantized(v_layout.cache_type) && quant_traits && quant_traits->to_float) {
                quant_traits->to_float(v_ptr, v_head_scratch.data(), v_head_dim);
                v_head = v_head_scratch.data();
            }
            if (!v_head) continue;

            for (int d = 0; d < v_head_dim; ++d) {
                out_head[d] += weight * v_head[d];
            }
            denom += weight;
        }

        if (!(denom > 0.0f) || !std::isfinite(denom)) {
            std::fill(out_head, out_head + v_head_dim, 0.0f);
            continue;
        }

        const float inv = 1.0f / denom;
        for (int d = 0; d < v_head_dim; ++d) {
            out_head[d] *= inv;
        }
    }
}

static void LogPagedAttentionEagerReferenceProbe(const PagedAttentionUserData* ud, const std::vector<int>& block_table,
                                                 int context_len, const KVRetentionSpan& retained_span, int current_pos,
                                                 const float* q_token, const float* runtime_out, int token_idx,
                                                 int seq_idx, int h_start, int h_end) {
    if (!ud || !ud->cache || !q_token || !runtime_out || context_len <= 0 || ud->n_head <= 0 || ud->head_dim <= 0) {
        return;
    }

    const int n_head_kv = ud->n_head_kv > 0 ? ud->n_head_kv : ud->cache->n_head_kv;
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    if (n_head_kv <= 0 || ud->n_head % n_head_kv != 0 || v_head_dim <= 0) {
        return;
    }

    thread_local std::vector<float> k_slot;
    thread_local std::vector<float> v_slot;
    thread_local std::vector<float> k_all;
    thread_local std::vector<float> v_all;
    thread_local std::vector<float> ref_out;

    const size_t k_slot_elems = static_cast<size_t>(ud->head_dim) * static_cast<size_t>(n_head_kv);
    const size_t v_slot_elems = static_cast<size_t>(v_head_dim) * static_cast<size_t>(n_head_kv);
    const size_t k_all_elems = static_cast<size_t>(n_head_kv) * static_cast<size_t>(context_len) *
                               static_cast<size_t>(ud->head_dim);
    const size_t v_all_elems = static_cast<size_t>(n_head_kv) * static_cast<size_t>(context_len) *
                               static_cast<size_t>(v_head_dim);
    const size_t out_elems = static_cast<size_t>(ud->n_head) * static_cast<size_t>(v_head_dim);

    k_slot.resize(k_slot_elems);
    v_slot.resize(v_slot_elems);
    k_all.assign(k_all_elems, 0.0f);
    v_all.assign(v_all_elems, 0.0f);
    ref_out.assign(out_elems, 0.0f);

    for (int t = 0; t < context_len; ++t) {
        const int token_pos =
            (t < retained_span.history_kept) ? densecore::llm::config::MapRetainedHistoryIndex(retained_span, t)
                                             : current_pos;
        const int logical_block = token_pos / BLOCK_SIZE;
        const int slot = token_pos % BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            continue;
        }
        const int block_id = block_table[static_cast<size_t>(logical_block)];
        if (block_id < 0 || block_id >= ud->cache->max_blocks) {
            continue;
        }

        ud->cache->ReadKSlot(block_id, ud->read_layer, slot, k_slot.data());
        ud->cache->ReadVSlot(block_id, ud->read_layer, slot, v_slot.data());

        for (int kv_head = 0; kv_head < n_head_kv; ++kv_head) {
            const float* src_k = k_slot.data() + static_cast<size_t>(kv_head) * ud->head_dim;
            float* dst_k = k_all.data() + (static_cast<size_t>(kv_head) * context_len + static_cast<size_t>(t)) *
                                              static_cast<size_t>(ud->head_dim);
            std::memcpy(dst_k, src_k, static_cast<size_t>(ud->head_dim) * sizeof(float));

            const float* src_v = v_slot.data() + static_cast<size_t>(kv_head) * v_head_dim;
            float* dst_v = v_all.data() + (static_cast<size_t>(kv_head) * context_len + static_cast<size_t>(t)) *
                                              static_cast<size_t>(v_head_dim);
            std::memcpy(dst_v, src_v, static_cast<size_t>(v_head_dim) * sizeof(float));
        }
    }

    const float scale =
        ud->attention_scale > 0.0f ? ud->attention_scale : (1.0f / std::sqrt(static_cast<float>(ud->head_dim)));
    ComputeFlashAttentionReference(q_token, k_all.data(), v_all.data(), ref_out.data(), ud->n_head, n_head_kv, 1,
                                   context_len, ud->head_dim, scale, false, 0, 0, -1, ud->logit_softcap);

    const int h_begin = std::max(0, h_start);
    const int h_limit = std::min(ud->n_head, h_end);
    if (h_begin >= h_limit) {
        return;
    }

    float max_abs_diff = 0.0f;
    int max_h = -1;
    int max_d = -1;
    float fast_val = 0.0f;
    float ref_val = 0.0f;
    bool runtime_nonfinite = false;
    bool ref_nonfinite = false;
    int first_bad_idx = -1;
    for (int h = h_begin; h < h_limit; ++h) {
        const float* got_head = runtime_out + static_cast<size_t>(h) * v_head_dim;
        const float* ref_head = ref_out.data() + static_cast<size_t>(h) * v_head_dim;
        for (int d = 0; d < v_head_dim; ++d) {
            const float got = got_head[d];
            const float expect = ref_head[d];
            if (!std::isfinite(got) || !std::isfinite(expect)) {
                if (first_bad_idx < 0) {
                    first_bad_idx = h * v_head_dim + d;
                    runtime_nonfinite = !std::isfinite(got);
                    ref_nonfinite = !std::isfinite(expect);
                    fast_val = got;
                    ref_val = expect;
                }
                continue;
            }
            const float diff = std::fabs(got - expect);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_h = h;
                max_d = d;
                fast_val = got;
                ref_val = expect;
            }
        }
    }

    std::fprintf(stderr,
                 "[PagedAttnEagerRef] layer=%d seq=%d token_idx=%d pos=%d heads=[%d,%d) context=%d first_bad_idx=%d "
                 "runtime_nonfinite=%d ref_nonfinite=%d max_abs_diff=%g head=%d dim=%d fast=%g ref=%g\n",
                 ud->layer, seq_idx, token_idx, current_pos, h_begin, h_limit, context_len, first_bad_idx,
                 runtime_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, static_cast<double>(max_abs_diff), max_h, max_d,
                 static_cast<double>(fast_val), static_cast<double>(ref_val));
}

void cb_paged_attention_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<PagedAttentionUserData*>(userdata);
    if (!ud || !ud->cache || !dst || !dst->data) return;
    if (nth <= 0) return;
    if (!dst->src[0] || !dst->src[1] || !dst->src[2]) return;

    const auto* q_tensor = dst->src[0];
    const auto* k_tensor = dst->src[1];
    const auto* v_tensor = dst->src[2];
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    if (!q_tensor->data || !k_tensor->data || !v_tensor->data) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const int q_tokens = static_cast<int>(q_tensor->ne[2]);
    if (q_tokens <= 0 || ud->head_dim <= 0 || v_head_dim <= 0 || ud->n_head <= 0) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const int n_head_kv = ud->n_head_kv > 0 ? ud->n_head_kv : ud->cache->n_head_kv;
    if (n_head_kv <= 0 || (ud->n_head % n_head_kv) != 0) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    if (q_tensor->type != GGML_TYPE_F32 || k_tensor->type != GGML_TYPE_F32 || v_tensor->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    if (static_cast<int>(q_tensor->ne[0]) != ud->head_dim || static_cast<int>(q_tensor->ne[1]) != ud->n_head ||
        static_cast<int>(k_tensor->ne[0]) != ud->head_dim || static_cast<int>(k_tensor->ne[1]) != n_head_kv ||
        static_cast<int>(v_tensor->ne[0]) != v_head_dim || static_cast<int>(v_tensor->ne[1]) != n_head_kv ||
        static_cast<int>(k_tensor->ne[2]) != q_tokens || static_cast<int>(v_tensor->ne[2]) != q_tokens ||
        static_cast<int>(dst->ne[0]) != v_head_dim || static_cast<int>(dst->ne[1]) != ud->n_head ||
        static_cast<int>(dst->ne[2]) != q_tokens) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const bool layout_ok = q_tensor->nb[0] == sizeof(float) && k_tensor->nb[0] == sizeof(float) &&
                           v_tensor->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
                           q_tensor->nb[1] == static_cast<size_t>(ud->head_dim) * sizeof(float) &&
                           k_tensor->nb[1] == static_cast<size_t>(ud->head_dim) * sizeof(float) &&
                           v_tensor->nb[1] == static_cast<size_t>(v_head_dim) * sizeof(float) &&
                           dst->nb[1] == static_cast<size_t>(v_head_dim) * sizeof(float);
    if (!layout_ok) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch || batch->num_seqs != q_tokens || static_cast<int>(batch->tokens.size()) != q_tokens ||
        static_cast<int>(batch->seq_id.size()) != q_tokens || static_cast<int>(batch->pos.size()) != q_tokens ||
        static_cast<int>(batch->block_tables.size()) != batch->num_seqs ||
        static_cast<int>(batch->n_past.size()) != batch->num_seqs) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    // Token-parallel fast path is only valid when the scheduler intentionally
    // collapsed work to one task per token. Otherwise keep the mixed
    // token/head-tiled path so batched decode can use more than q_tokens tasks.
    const bool token_parallel_mode = (nth == q_tokens && q_tokens > 1);

    // Phase 1: Write current decode K/V to paged cache (parallel over tokens).
    // Each thread handles a disjoint subset of decode tokens, then synchronizes
    // via an epoch barrier before any thread starts attention reads.
    const bool do_profile = (ith == 0) && IsDecodeProfileEnabled();
    const bool do_qwen36_profile = (ith == 0) && IsQwen36ProfilingEnabled();
    std::chrono::steady_clock::time_point kv_begin, kv_end;
    if (do_profile || do_qwen36_profile) kv_begin = std::chrono::steady_clock::now();

    const bool write_current_kv = ud->write_current_kv;

    if (!write_current_kv) {
        // Shared Gemma4 reader layers reuse the source-layer cache and must not
        // publish their local K/V into that source cache.
    } else if (token_parallel_mode) {
        // Each thread writes KV for exactly token_idx == ith. No barrier needed
        // because each thread only reads from its own token's KV cache slot.
        const int token_idx = ith;
        int writes_ok = 0;
        int writes_skipped = 0;
        WriteCurrentBatchKvToCache(batch, k_tensor, ud->cache, ud->write_layer, ud->head_dim, n_head_kv,
                                   /*is_k=*/true, token_idx, token_idx + 1, &writes_ok, &writes_skipped);
        WriteCurrentBatchKvToCache(batch, v_tensor, ud->cache, ud->write_layer, v_head_dim, n_head_kv,
                                   /*is_k=*/false, token_idx, token_idx + 1, &writes_ok, &writes_skipped);
    } else if (q_tokens == 1) {
        // Single-token decode only has one KV write, but the legacy barrier made
        // every GGML worker participate in kv_writers_done and spin until the last
        // idle worker arrived. Let thread 0 perform the write and release the
        // readers as soon as the slot becomes visible.
        uint64_t epoch = 0;
        if (ith == 0) {
            epoch = ud->epoch_started.fetch_add(1, std::memory_order_acq_rel) + 1;
            int writes_ok = 0;
            int writes_skipped = 0;
            WriteCurrentBatchKvToCache(batch, k_tensor, ud->cache, ud->write_layer, ud->head_dim, n_head_kv,
                                       /*is_k=*/true, 0, 1, &writes_ok, &writes_skipped);
            WriteCurrentBatchKvToCache(batch, v_tensor, ud->cache, ud->write_layer, v_head_dim, n_head_kv,
                                       /*is_k=*/false, 0, 1, &writes_ok, &writes_skipped);
            ud->epoch_done.store(epoch, std::memory_order_release);
        } else {
            int spin_count = 0;
            while (true) {
                const uint64_t started = ud->epoch_started.load(std::memory_order_acquire);
                if (started != 0) {
                    epoch = started;
                    break;
                }
                SpinPause(spin_count++);
            }
            spin_count = 0;
            while (ud->epoch_done.load(std::memory_order_acquire) < epoch) {
                SpinPause(spin_count++);
            }
        }
    } else {
        // Standard barrier-based path for n_tokens==1 or head-tiled parallelism
        uint64_t epoch = 0;
        if (ith == 0) {
            ud->kv_writers_done.store(0, std::memory_order_release);
            epoch = ud->epoch_started.fetch_add(1, std::memory_order_acq_rel) + 1;
        } else {
            int spin_count = 0;
            while (true) {
                const uint64_t started = ud->epoch_started.load(std::memory_order_acquire);
                const uint64_t done = ud->epoch_done.load(std::memory_order_acquire);
                if (started > done) {
                    epoch = started;
                    break;
                }
                SpinPause(spin_count++);
            }
        }

        for (int i = ith; i < q_tokens; i += nth) {
            int writes_ok = 0;
            int writes_skipped = 0;
            WriteCurrentBatchKvToCache(batch, k_tensor, ud->cache, ud->write_layer, ud->head_dim, n_head_kv,
                                       /*is_k=*/true, i, i + 1, &writes_ok, &writes_skipped);
            WriteCurrentBatchKvToCache(batch, v_tensor, ud->cache, ud->write_layer, v_head_dim, n_head_kv,
                                       /*is_k=*/false, i, i + 1, &writes_ok, &writes_skipped);
        }

        const int writers_done = ud->kv_writers_done.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (writers_done == nth) {
            ud->epoch_done.store(epoch, std::memory_order_release);
        } else {
            int spin_count = 0;
            while (ud->epoch_done.load(std::memory_order_acquire) < epoch) {
                SpinPause(spin_count++);
            }
        }
    }
    if (do_profile || do_qwen36_profile) kv_end = std::chrono::steady_clock::now();

    // Prefetch next layer's KV blocks to warm L2 cache and TLB before the next
    // layer's callback starts. Only thread 0 issues prefetches to avoid duplicate
    // prefetch storms across threads. With the layer-major arena layout, all blocks
    // for layer L+1 are in a contiguous address band, so even a single cache-line
    // prefetch per block is enough to kick off the hardware stream prefetcher.
    if (ith == 0 && ud->cache && (ud->layer + 1) < ud->cache->n_layer) {
        const int next_layer = ud->layer + 1;
        const BatchSpec* batch_for_pf = GetCurrentBatch();
        if (batch_for_pf) {
            for (int si = 0; si < batch_for_pf->num_seqs; ++si) {
                if (si >= static_cast<int>(batch_for_pf->block_tables.size())) break;
                const auto& bt = batch_for_pf->block_tables[static_cast<size_t>(si)];
                // Prefetch up to first 16 blocks — enough to prime the HW prefetcher
                // for the sequential scan that follows. Remaining blocks are covered
                // by the hardware stream prefetcher once access begins.
                const int max_pf_blocks = std::min(static_cast<int>(bt.size()), 16);
                for (int bi = 0; bi < max_pf_blocks; ++bi) {
                    const int bid = bt[static_cast<size_t>(bi)];
                    if (bid < 0 || bid >= ud->cache->max_blocks) continue;
                    const void* k_ptr = ud->cache->GetKBlockPtr(bid, next_layer);
                    const void* v_ptr = ud->cache->GetVBlockPtr(bid, next_layer);
                    if (k_ptr) densecore::simd::Prefetch(k_ptr);
                    if (v_ptr) densecore::simd::Prefetch(v_ptr);
                }
            }
        }
    }

    // Phase 2: Attention computation
    const int head_tile = ResolvePagedAttentionDecodeHeadTile(ud->n_head, q_tokens, nth);
    const int tiles_per_token = std::max(1, (ud->n_head + head_tile - 1) / head_tile);
    int tile_start, tile_end;
    if (token_parallel_mode) {
        // Each thread handles all heads for its token (token_idx == ith)
        tile_start = ith * tiles_per_token;
        tile_end = tile_start + tiles_per_token;
    } else {
        const int total_tiles = q_tokens * tiles_per_token;
        if (total_tiles <= 0) {
            return;
        }
        tile_start = (total_tiles * ith) / nth;
        tile_end = (total_tiles * (ith + 1)) / nth;
    }
    if (tile_start >= tile_end) {
        return;
    }

    const bool use_hwy = IsPagedAttentionHwyEnabled();
    const float scale =
        ud->attention_scale > 0.0f ? ud->attention_scale : (1.0f / std::sqrt(static_cast<float>(ud->head_dim)));
    const char* q_base = reinterpret_cast<const char*>(q_tensor->data);
    char* out_base = reinterpret_cast<char*>(dst->data);
    const size_t q_token_stride = static_cast<size_t>(q_tensor->nb[2]);
    const size_t out_token_stride = static_cast<size_t>(dst->nb[2]);

    // Precompute cache_type_id once to skip per-tile Tensor wrapper + validation
    int32_t cache_type_id = -1;
    if (use_hwy) {
        if (ud->cache->cache_type == GGML_TYPE_F32)
            cache_type_id = 0;
        else if (ud->cache->cache_type == GGML_TYPE_F16)
            cache_type_id = 1;
        else if (ud->cache->cache_type == GGML_TYPE_Q4_0)
            cache_type_id = 4;
        else if (ud->cache->cache_type == GGML_TYPE_Q8_0)
            cache_type_id = 8;
    }
    const auto k_cache_layout = ud->cache->GetBlockLayout();
    const auto v_cache_layout = ud->cache->GetVBlockLayout();
    const bool hwy_ready = use_hwy && cache_type_id >= 0 && k_cache_layout.head_stride_bytes > 0 &&
                           k_cache_layout.slot_stride_bytes > 0 && v_cache_layout.head_stride_bytes > 0 &&
                           v_cache_layout.slot_stride_bytes > 0;
    thread_local std::vector<const void*> k_block_ptrs;
    thread_local std::vector<const void*> v_block_ptrs;
    const void* const* shared_k_block_ptrs_data = nullptr;
    const void* const* shared_v_block_ptrs_data = nullptr;
    int cached_token_idx = -1;
    int cached_seq_idx = -1;
    const std::vector<int>* cached_block_table = nullptr;
    int cached_ptr_seq_idx = -1;
    const std::vector<int>* cached_ptr_block_table = nullptr;
    int cached_context_len = 0;
    int cached_context_start_pos = 0;
    int cached_pos_i = -1;
    KVRetentionSpan cached_retained_span{};
    KVRetentionSpan cached_attention_mask_span{};
    bool cached_retention_truncated = false;
    bool cached_noncontiguous_retention = false;
    bool cached_token_valid = false;
    const float* cached_q_token = nullptr;
    float* cached_out_token = nullptr;

    auto zero_token_heads = [&](float* out_token, int h_start, int h_end) {
        for (int h = h_start; h < h_end; ++h) {
            float* out_head = out_token + static_cast<size_t>(h) * v_head_dim;
            std::fill(out_head, out_head + v_head_dim, 0.0f);
        }
    };

    std::chrono::steady_clock::time_point attn_begin;
    if (do_profile || do_qwen36_profile) attn_begin = std::chrono::steady_clock::now();

    if (q_tokens == 1 && hwy_ready && ud->shared_k_block_ptrs && ud->shared_v_block_ptrs) {
        if (ith == 0) {
            ud->shared_k_block_ptrs->clear();
            ud->shared_v_block_ptrs->clear();
            const int seq_idx = batch->seq_id[0];
            if (seq_idx >= 0 && seq_idx < batch->num_seqs && seq_idx < static_cast<int>(batch->block_tables.size())) {
                const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
                ud->shared_k_block_ptrs->resize(block_table.size(), nullptr);
                ud->shared_v_block_ptrs->resize(block_table.size(), nullptr);
                for (size_t bi = 0; bi < block_table.size(); ++bi) {
                    const int block_id = block_table[bi];
                    if (block_id < 0 || block_id >= ud->cache->max_blocks) {
                        continue;
                    }
                    (*ud->shared_k_block_ptrs)[bi] = ud->cache->GetKBlockPtr(block_id, ud->read_layer);
                    (*ud->shared_v_block_ptrs)[bi] = ud->cache->GetVBlockPtr(block_id, ud->read_layer);
                }
            }
            ud->shared_block_ptrs_ready.store(1, std::memory_order_release);
        } else {
            int spin_count = 0;
            while (ud->shared_block_ptrs_ready.load(std::memory_order_acquire) == 0) {
                SpinPause(spin_count++);
            }
        }
        shared_k_block_ptrs_data = ud->shared_k_block_ptrs->empty() ? nullptr : ud->shared_k_block_ptrs->data();
        shared_v_block_ptrs_data = ud->shared_v_block_ptrs->empty() ? nullptr : ud->shared_v_block_ptrs->data();
    }

    for (int tile = tile_start; tile < tile_end; ++tile) {
        const int token_idx = tile / tiles_per_token;
        const int head_tile_idx = tile % tiles_per_token;
        const int h_start = head_tile_idx * head_tile;
        const int h_end = std::min(ud->n_head, h_start + head_tile);
        if (h_start >= h_end) {
            continue;
        }

        if (token_idx != cached_token_idx) {
            cached_token_idx = token_idx;
            cached_seq_idx = -1;
            cached_block_table = nullptr;
            cached_context_len = 0;
            cached_context_start_pos = 0;
            cached_pos_i = -1;
            cached_retained_span = {};
            cached_attention_mask_span = {};
            cached_retention_truncated = false;
            cached_noncontiguous_retention = false;
            cached_token_valid = false;
            cached_q_token = nullptr;
            cached_out_token = nullptr;

            if (token_idx >= 0 && token_idx < q_tokens) {
                cached_q_token =
                    reinterpret_cast<const float*>(q_base + static_cast<size_t>(token_idx) * q_token_stride);
                cached_out_token =
                    reinterpret_cast<float*>(out_base + static_cast<size_t>(token_idx) * out_token_stride);
                const int seq_idx = batch->seq_id[static_cast<size_t>(token_idx)];
                if (seq_idx >= 0 && seq_idx < batch->num_seqs) {
                    const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
                    const int pos_i = batch->pos[static_cast<size_t>(token_idx)];
                    const int n_past_i = batch->n_past[static_cast<size_t>(seq_idx)];
                    if (!block_table.empty() && pos_i >= 0 && n_past_i >= 0) {
                        cached_seq_idx = seq_idx;
                        cached_block_table = &block_table;
                        cached_pos_i = pos_i;
                        KVRetentionPolicy retention_policy = GetKVRetentionPolicy();
                        if (ud->force_full_history) {
                            retention_policy.enabled = false;
                            retention_policy.sliding_window = -1;
                            retention_policy.sink_tokens = 0;
                        }
                        cached_retained_span =
                            densecore::llm::config::ComputeKVRetentionSpan(n_past_i, retention_policy);
                        KVRetentionPolicy attention_mask_policy;
                        attention_mask_policy.enabled = ud->sliding_window >= 0;
                        attention_mask_policy.sliding_window = ud->sliding_window >= 0 ? ud->sliding_window : -1;
                        attention_mask_policy.sink_tokens =
                            std::max(0, densecore::env::ParseIntEnv(
                                            "DENSECORE_KV_SINK_TOKENS",
                                            densecore::env::ParseIntEnv("DENSECORE_SINK_TOKENS", 0)));
                        cached_attention_mask_span =
                            densecore::llm::config::ComputeKVRetentionSpan(n_past_i, attention_mask_policy);
                        if (ud->sliding_window >= 0) {
                            // Sliding-window attention must read the same logical
                            // tail tokens it masks as visible. Keeping full
                            // history here made long prompts read early KV slots
                            // while labeling them as tail positions.
                            cached_retained_span = cached_attention_mask_span;
                        }
                        cached_retention_truncated = cached_retained_span.history_kept < n_past_i;
                        cached_noncontiguous_retention =
                            cached_retention_truncated && cached_retained_span.sink_kept > 0;
                        cached_context_start_pos =
                            (cached_retention_truncated && cached_retained_span.sink_kept == 0)
                                ? cached_retained_span.tail_start
                                : 0;
                        const int max_context = static_cast<int>(block_table.size()) * BLOCK_SIZE;
                        cached_context_len = std::max(1, std::min(cached_retained_span.history_kept + 1, max_context));
                        cached_token_valid = cached_context_len > 0;
                    }
                }
            }
        }

        float* out_token = cached_out_token;
        if (!out_token) {
            continue;
        }
        if (!cached_token_valid || !cached_block_table || !cached_q_token) {
            zero_token_heads(out_token, h_start, h_end);
            continue;
        }

        const float* q_token = cached_q_token;
        const auto& block_table = *cached_block_table;

        if (!hwy_ready || cached_noncontiguous_retention) {
            ComputePagedAttentionScalarHeads(ud, block_table, cached_context_len, cached_retained_span,
                                             cached_attention_mask_span, cached_pos_i, scale, q_token, out_token,
                                             h_start, h_end);
            if (ShouldRunPagedAttentionEagerReferenceProbe(ud->layer, token_idx)) {
                LogPagedAttentionEagerReferenceProbe(ud, block_table, cached_context_len, cached_retained_span,
                                                    cached_pos_i, q_token, out_token, token_idx, cached_seq_idx,
                                                    h_start, h_end);
            }
            continue;
        }

        if (cached_ptr_block_table != &block_table || cached_ptr_seq_idx != cached_seq_idx) {
            if (!(q_tokens == 1 && shared_k_block_ptrs_data && shared_v_block_ptrs_data)) {
                k_block_ptrs.resize(block_table.size(), nullptr);
                v_block_ptrs.resize(block_table.size(), nullptr);
                for (size_t bi = 0; bi < block_table.size(); ++bi) {
                    const int block_id = block_table[bi];
                    if (block_id < 0 || block_id >= ud->cache->max_blocks) {
                        continue;
                    }
                    k_block_ptrs[bi] = ud->cache->GetKBlockPtr(block_id, ud->read_layer);
                    v_block_ptrs[bi] = ud->cache->GetVBlockPtr(block_id, ud->read_layer);
                }
            }
            cached_ptr_seq_idx = cached_seq_idx;
            cached_ptr_block_table = &block_table;
        }

        densecore::hwy_kernels::PagedAttention_Hwy(
            q_token, shared_k_block_ptrs_data ? shared_k_block_ptrs_data : k_block_ptrs.data(),
            shared_v_block_ptrs_data ? shared_v_block_ptrs_data : v_block_ptrs.data(), cache_type_id, ud->n_head,
            ud->head_dim, v_head_dim, n_head_kv, static_cast<int32_t>(block_table.size()),
            cached_context_len, cached_context_start_pos, cached_pos_i, ud->sliding_window,
            cached_attention_mask_span.history_kept, cached_attention_mask_span.sink_kept,
            cached_attention_mask_span.tail_start,
            static_cast<int64_t>(k_cache_layout.head_stride_bytes),
            static_cast<int64_t>(k_cache_layout.slot_stride_bytes),
            static_cast<int64_t>(v_cache_layout.head_stride_bytes),
            static_cast<int64_t>(v_cache_layout.slot_stride_bytes), scale, ud->logit_softcap, out_token, h_start,
            h_end, ud->n_head);

        if (ShouldRunPagedAttentionReferenceProbe(ud->layer, token_idx)) {
            thread_local std::vector<float> scalar_ref;
            const size_t token_elems = static_cast<size_t>(ud->n_head) * static_cast<size_t>(v_head_dim);
            scalar_ref.assign(token_elems, 0.0f);
            ComputePagedAttentionScalarHeads(ud, block_table, cached_context_len, cached_retained_span,
                                             cached_attention_mask_span, cached_pos_i, scale, q_token,
                                             scalar_ref.data(), h_start, h_end);

            float max_abs_diff = 0.0f;
            int max_h = -1;
            int max_d = -1;
            float fast_val = 0.0f;
            float ref_val = 0.0f;
            for (int h = h_start; h < h_end; ++h) {
                const float* fast_head = out_token + static_cast<size_t>(h) * v_head_dim;
                const float* ref_head = scalar_ref.data() + static_cast<size_t>(h) * v_head_dim;
                for (int d = 0; d < v_head_dim; ++d) {
                    const float got = fast_head[d];
                    const float expect = ref_head[d];
                    const float diff = std::fabs(got - expect);
                    if (diff > max_abs_diff) {
                        max_abs_diff = diff;
                        max_h = h;
                        max_d = d;
                        fast_val = got;
                        ref_val = expect;
                    }
                }
            }
            if (max_abs_diff > 1e-4f || !std::isfinite(max_abs_diff)) {
                std::fprintf(
                    stderr,
                    "[PagedAttnRef] layer=%d seq=%d token_idx=%d pos=%d heads=[%d,%d) context=%d max_abs_diff=%g "
                    "head=%d dim=%d fast=%g ref=%g\n",
                    ud->layer, cached_seq_idx, token_idx, cached_pos_i, h_start, h_end, cached_context_len,
                    static_cast<double>(max_abs_diff), max_h, max_d, static_cast<double>(fast_val),
                    static_cast<double>(ref_val));
            }
        }
        if (ShouldRunPagedAttentionEagerReferenceProbe(ud->layer, token_idx)) {
            LogPagedAttentionEagerReferenceProbe(ud, block_table, cached_context_len, cached_retained_span, cached_pos_i,
                                                q_token, out_token, token_idx, cached_seq_idx, h_start, h_end);
        }
    }

    // Decode profiling: log KV write + attention compute timing (thread 0, every 100th call)
    if (do_profile || do_qwen36_profile) {
        const auto attn_end = std::chrono::steady_clock::now();
        if (do_qwen36_profile) {
            InferenceWorkContext* work_ctx = GetCurrentWorkContext();
            AddQwen36ProfileNs(
                work_ctx->qwen36_profile.kv_update_ns,
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(kv_end - kv_begin).count()));
            AddQwen36ProfileNs(
                work_ctx->qwen36_profile.paged_attention_ns,
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(attn_end - attn_begin).count()));
            AddQwen36ProfileNs(
                work_ctx->qwen36_profile.attention_ns,
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(attn_end - attn_begin).count()));
            MarkQwen36ProfileFlag(work_ctx->qwen36_profile.attention_path_paged);
        }
        static thread_local int profile_cb_count = 0;
        if (do_profile && ++profile_cb_count % 100 == 0) {
            const long kv_us = std::chrono::duration_cast<std::chrono::microseconds>(kv_end - kv_begin).count();
            const long attn_us = std::chrono::duration_cast<std::chrono::microseconds>(attn_end - attn_begin).count();
            fprintf(stderr, "[DecodeProfile] bs=%d threads=%d layer=%d token_parallel=%d kv_us=%ld attn_us=%ld\n",
                    q_tokens, nth, ud->layer, token_parallel_mode ? 1 : 0, kv_us, attn_us);
        }
    }
}

void cb_glm_dsa_attention_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<PagedAttentionUserData*>(userdata);
    if (!ud || !ud->cache || !dst || !dst->data || nth <= 0) return;
    if (!ud->cache->has_index_cache) return;
    if (!dst->src[0] || !dst->src[1] || !dst->src[2] || !dst->src[3] || !dst->src[4] || !dst->src[5]) return;

    const auto* q_tensor = dst->src[0];
    const auto* k_tensor = dst->src[1];
    const auto* v_tensor = dst->src[2];
    const auto* index_q_tensor = dst->src[3];
    const auto* index_weights_tensor = dst->src[4];
    const auto* index_k_tensor = dst->src[5];
    if (!q_tensor->data || !k_tensor->data || !v_tensor->data || !index_q_tensor->data || !index_weights_tensor->data ||
        !index_k_tensor->data) {
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    const int q_tokens = static_cast<int>(q_tensor->ne[2]);
    const int n_head = ud->n_head;
    const int n_head_kv = ud->n_head_kv > 0 ? ud->n_head_kv : ud->cache->n_head_kv;
    const int head_dim = ud->head_dim;
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    const int index_n_heads = ud->index_n_heads;
    const int index_head_dim = ud->index_head_dim;
    const int index_topk = ud->index_topk;
    if (q_tokens <= 0 || n_head <= 0 || n_head_kv <= 0 || head_dim <= 0 || v_head_dim <= 0 || index_n_heads <= 0 ||
        index_head_dim <= 0 || (n_head % n_head_kv) != 0) {
        return;
    }

    const char* q_base = reinterpret_cast<const char*>(q_tensor->data);
    const char* k_base = reinterpret_cast<const char*>(k_tensor->data);
    const char* v_base = reinterpret_cast<const char*>(v_tensor->data);
    const char* index_q_base = reinterpret_cast<const char*>(index_q_tensor->data);
    const char* index_weights_base = reinterpret_cast<const char*>(index_weights_tensor->data);
    const char* index_k_base = reinterpret_cast<const char*>(index_k_tensor->data);
    char* out_base = reinterpret_cast<char*>(dst->data);

    const size_t q_token_stride = static_cast<size_t>(q_tensor->nb[2]);
    const size_t k_token_stride = static_cast<size_t>(k_tensor->nb[2]);
    const size_t v_token_stride = static_cast<size_t>(v_tensor->nb[2]);
    const size_t index_q_token_stride = static_cast<size_t>(index_q_tensor->nb[2]);
    const size_t index_weights_token_stride = static_cast<size_t>(index_weights_tensor->nb[1]);
    const size_t index_k_token_stride = static_cast<size_t>(index_k_tensor->nb[1]);
    const size_t out_token_stride = static_cast<size_t>(dst->nb[2]);
    const int kv_group_size = n_head / n_head_kv;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float index_scale = 1.0f / std::sqrt(static_cast<float>(index_head_dim));

    uint64_t epoch = 0;
    if (ith == 0) {
        ud->kv_writers_done.store(0, std::memory_order_release);
        epoch = ud->epoch_started.fetch_add(1, std::memory_order_acq_rel) + 1;
    } else {
        int spin_count = 0;
        while (true) {
            const uint64_t started = ud->epoch_started.load(std::memory_order_acquire);
            const uint64_t done = ud->epoch_done.load(std::memory_order_acquire);
            if (started > done) {
                epoch = started;
                break;
            }
            SpinPause(spin_count++);
        }
    }

    for (int i = ith; i < q_tokens; i += nth) {
        if (i >= static_cast<int>(batch->seq_id.size()) || i >= static_cast<int>(batch->pos.size())) continue;
        const int seq_idx = batch->seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= static_cast<int>(batch->block_tables.size())) continue;
        const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
        const int pos_i = batch->pos[static_cast<size_t>(i)];
        if (pos_i < 0) continue;

        const int logical_block = pos_i / BLOCK_SIZE;
        const int slot = pos_i % BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) continue;
        const int block_id = block_table[static_cast<size_t>(logical_block)];
        if (block_id < 0 || block_id >= ud->cache->max_blocks) continue;

        const float* k_src = reinterpret_cast<const float*>(k_base + static_cast<size_t>(i) * k_token_stride);
        const float* v_src = reinterpret_cast<const float*>(v_base + static_cast<size_t>(i) * v_token_stride);
        const float* index_k_src =
            reinterpret_cast<const float*>(index_k_base + static_cast<size_t>(i) * index_k_token_stride);
        ud->cache->WriteKSlot(block_id, ud->layer, slot, k_src);
        ud->cache->WriteVSlot(block_id, ud->layer, slot, v_src);
        ud->cache->WriteIndexSlot(block_id, ud->layer, slot, index_k_src);
    }

    const int writers_done = ud->kv_writers_done.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (writers_done == nth) {
        ud->epoch_done.store(epoch, std::memory_order_release);
    } else {
        int spin_count = 0;
        while (ud->epoch_done.load(std::memory_order_acquire) < epoch) {
            SpinPause(spin_count++);
        }
    }

    thread_local std::vector<float> index_key_scratch;
    thread_local std::vector<float> k_slot_scratch;
    thread_local std::vector<float> v_slot_scratch;
    thread_local std::vector<std::pair<float, int>> index_scores;
    thread_local std::vector<float> attn_scores;
    thread_local std::vector<int> selected_positions;
    thread_local std::unordered_map<int, std::vector<int>> token_selected_positions_cache;
    token_selected_positions_cache.clear();
    index_key_scratch.resize(static_cast<size_t>(index_head_dim));
    k_slot_scratch.resize(static_cast<size_t>(ud->cache->GetElementsPerSlot()));
    v_slot_scratch.resize(static_cast<size_t>(ud->cache->GetVElementsPerSlot()));

    // Flatten token and head loops for 2D parallelization
    const int total_work = q_tokens * n_head;
    for (int work_idx = ith; work_idx < total_work; work_idx += nth) {
        const int token_idx = work_idx / n_head;
        const int h = work_idx % n_head;

        if (token_idx >= static_cast<int>(batch->seq_id.size()) || token_idx >= static_cast<int>(batch->pos.size())) {
            continue;
        }

        const int seq_idx = batch->seq_id[static_cast<size_t>(token_idx)];
        if (seq_idx < 0 || seq_idx >= batch->num_seqs || seq_idx >= static_cast<int>(batch->block_tables.size()) ||
            seq_idx >= static_cast<int>(batch->n_past.size())) {
            continue;
        }

        const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
        const int pos_i = batch->pos[static_cast<size_t>(token_idx)];
        const int n_past_i = batch->n_past[static_cast<size_t>(seq_idx)];
        if (block_table.empty() || pos_i < 0 || n_past_i < 0) {
            continue;
        }

        const KVRetentionSpan retained_span =
            densecore::llm::config::ComputeKVRetentionSpan(n_past_i, GetKVRetentionPolicy());
        const int max_context = static_cast<int>(block_table.size()) * BLOCK_SIZE;
        const int context_len = std::max(1, std::min(retained_span.history_kept + 1, max_context));
        const int effective_topk = index_topk > 0 ? std::min(index_topk, context_len) : context_len;

        // Compute selected positions once per token (with lazy cache)
        if (token_selected_positions_cache.find(token_idx) == token_selected_positions_cache.end()) {
            const float* index_weights_token = reinterpret_cast<const float*>(
                index_weights_base + static_cast<size_t>(token_idx) * index_weights_token_stride);
            const char* index_q_token_base = index_q_base + static_cast<size_t>(token_idx) * index_q_token_stride;
            index_scores.clear();
            index_scores.reserve(static_cast<size_t>(context_len));
            for (int t = 0; t < context_len; ++t) {
                const int token_pos =
                    (t < retained_span.history_kept)
                        ? densecore::llm::config::MapRetainedHistoryIndex(retained_span, t)
                        : pos_i;
                const int logical_block = token_pos / BLOCK_SIZE;
                const int slot = token_pos % BLOCK_SIZE;
                if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) continue;
                const int block_id = block_table[static_cast<size_t>(logical_block)];
                if (block_id < 0 || block_id >= ud->cache->max_blocks) continue;

                ud->cache->ReadIndexSlot(block_id, ud->layer, slot, index_key_scratch.data());
                float score = 0.0f;
                for (int ih = 0; ih < index_n_heads; ++ih) {
                    const float* q_index_head = reinterpret_cast<const float*>(
                        index_q_token_base + static_cast<size_t>(ih) * index_q_tensor->nb[1]);
                    float dot = 0.0f;
                    for (int d = 0; d < index_head_dim; ++d) {
                        dot += q_index_head[d] * index_key_scratch[static_cast<size_t>(d)];
                    }
                    score += index_weights_token[ih] * (dot * index_scale);
                }
                index_scores.emplace_back(score, token_pos);
            }

            if (!index_scores.empty()) {
                if (static_cast<int>(index_scores.size()) > effective_topk) {
                    std::partial_sort(index_scores.begin(), index_scores.begin() + effective_topk, index_scores.end(),
                                      [](const auto& a, const auto& b) { return a.first > b.first; });
                }
                const int selected_count = std::min(effective_topk, static_cast<int>(index_scores.size()));
                std::vector<int>& cached = token_selected_positions_cache[token_idx];
                cached.resize(static_cast<size_t>(selected_count));
                for (int i = 0; i < selected_count; ++i) {
                    cached[static_cast<size_t>(i)] = index_scores[static_cast<size_t>(i)].second;
                }
            }
        }

        const auto& cached_it = token_selected_positions_cache.find(token_idx);
        if (cached_it == token_selected_positions_cache.end() || cached_it->second.empty()) {
            continue;
        }
        const std::vector<int>& selected_positions_ref = cached_it->second;
        const int selected_count = static_cast<int>(selected_positions_ref.size());

        float* out_token = reinterpret_cast<float*>(out_base + static_cast<size_t>(token_idx) * out_token_stride);
        const float* q_token = reinterpret_cast<const float*>(q_base + static_cast<size_t>(token_idx) * q_token_stride);
        const float* q_head = q_token + static_cast<size_t>(h) * head_dim;
        float* out_head = out_token + static_cast<size_t>(h) * v_head_dim;
        std::fill(out_head, out_head + v_head_dim, 0.0f);

        const int kv_head = std::min(n_head_kv - 1, std::max(0, h / kv_group_size));
        float max_score = -std::numeric_limits<float>::infinity();
        attn_scores.resize(static_cast<size_t>(selected_count));
        for (int i = 0; i < selected_count; ++i) {
            const int token_pos = selected_positions_ref[static_cast<size_t>(i)];
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            const int block_id = block_table[static_cast<size_t>(logical_block)];
            ud->cache->ReadKSlot(block_id, ud->layer, slot, k_slot_scratch.data());
            const float* k_head = k_slot_scratch.data() + static_cast<size_t>(kv_head) * head_dim;

            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_head[d] * k_head[d];
            }
            score *= attn_scale;
            attn_scores[static_cast<size_t>(i)] = score;
            if (score > max_score) max_score = score;
        }

        if (!std::isfinite(max_score)) {
            continue;
        }

        float denom = 0.0f;
        for (int i = 0; i < selected_count; ++i) {
            const float weight = std::exp(attn_scores[static_cast<size_t>(i)] - max_score);
            if (!(weight > 0.0f) || !std::isfinite(weight)) continue;
            denom += weight;

            const int token_pos = selected_positions_ref[static_cast<size_t>(i)];
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            const int block_id = block_table[static_cast<size_t>(logical_block)];
            ud->cache->ReadVSlot(block_id, ud->layer, slot, v_slot_scratch.data());
            const float* v_head = v_slot_scratch.data() + static_cast<size_t>(kv_head) * v_head_dim;
            for (int d = 0; d < v_head_dim; ++d) {
                out_head[d] += weight * v_head[d];
            }
        }

        if (!(denom > 0.0f) || !std::isfinite(denom)) {
            std::fill(out_head, out_head + v_head_dim, 0.0f);
            continue;
        }
        const float inv = 1.0f / denom;
        for (int d = 0; d < v_head_dim; ++d) {
            out_head[d] *= inv;
        }
    }
}

inline struct ggml_tensor* ggml_glm_dsa_attention(struct ggml_context* ctx, struct ggml_tensor* q_cur,
                                                  struct ggml_tensor* k_cur, struct ggml_tensor* v_cur,
                                                  struct ggml_tensor* index_q, struct ggml_tensor* index_weights,
                                                  struct ggml_tensor* index_k, PagedAttentionUserData* userdata) {
    const int64_t ne_res[4] = {v_cur->ne[0], q_cur->ne[1], q_cur->ne[2], 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = q_cur;
    result->src[1] = k_cur;
    result->src[2] = v_cur;
    result->src[3] = index_q;
    result->src[4] = index_weights;
    result->src[5] = index_k;

    const BatchSpec* batch = GetCurrentBatch();
    int n_tasks = ResolveInferenceConfig(batch).num_threads;
    if (n_tasks <= 0) {
        n_tasks = std::thread::hardware_concurrency();
        if (n_tasks <= 0) n_tasks = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_tasks = std::min(n_tasks, physical_cores);
    }
    const int n_head = std::max(1, static_cast<int>(q_cur->ne[1]));
    const int q_tokens = std::max(1, static_cast<int>(q_cur->ne[2]));
    n_tasks = std::max(1, std::min(n_tasks, q_tokens * n_head));

    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_glm_dsa_attention_custom, n_tasks, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}

/**
 * Create a custom GGML operation for parallel GEMV
 *
 * REFACTORED: Uses GGML_OP_CUSTOM instead of GGML_OP_MAP_CUSTOM1.
 * GGML_OP_MAP_CUSTOM1 assumes output shape == input shape, which causes
 * buffer overflows when Qwen3 projections change dimensions (e.g., 1024->2048).
 * GGML_OP_CUSTOM allows the output tensor shape to be independent of inputs.
 */
inline struct ggml_tensor* ggml_mul_mat_gemv(struct ggml_context* ctx, struct ggml_tensor* weight,
                                             struct ggml_tensor* input, GemvUserData* userdata) {
    const int K = weight->ne[1];  // Output dimension
    const int N = weight->ne[0];  // Input dimension

    userdata->weight_tensor = weight;
    userdata->N = N;
    userdata->K = K;
    userdata->weight_type = weight->type;
    userdata->input_quant_type = GGML_TYPE_F32;

    const ggml_type wtype = weight->type;
    if (ggml_is_quantized(wtype)) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(wtype);
        if (type_traits_cpu && type_traits_cpu->vec_dot) {
            const ggml_type vec_dot_type = type_traits_cpu->vec_dot_type;
            const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
            if (input_type_traits && input_type_traits->from_float) {
                const size_t quant_input_size = ggml_row_size(vec_dot_type, N);
                if (quant_input_size > 0 && quant_input_size <= kMaxQuantInputBufferSize) {
                    userdata->input_quant_type = vec_dot_type;
                }
            }
        }
    }

    const BatchSpec* batch = GetCurrentBatch();
    int n_threads = ResolveInferenceConfig(batch).num_threads;
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }

    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores <= 0) physical_cores = 4;

    n_threads = std::min(n_threads, physical_cores);

    if (K < 64) {
        n_threads = 1;
    } else if (K < 512) {
        n_threads = std::min(n_threads, 2);
    } else if (K < 1536) {
        n_threads = std::min(n_threads, 4);
    } else if (K < 3072) {
        n_threads = std::min(n_threads, 6);
    }

    // ===========================================================================
    // Create output tensor with correct dimension K (INDEPENDENT of input shape)
    // This is the critical fix: GGML_OP_CUSTOM allows explicit output dimensions
    // ===========================================================================
    const int64_t ne_res[4] = {K, 1, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    // ===========================================================================
    // Configure GGML_OP_CUSTOM (NOT MAP_CUSTOM1 which assumes shape preservation)
    // ===========================================================================
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;   // Input tensor accessible via dst->src[0] in callback
    result->src[1] = weight;  // Weight tensor accessible via dst->src[1] in callback

    // Custom op params (layout must match ggml_custom_op_params)
    // Signature: { ggml_custom_op_t fun, int n_tasks, void *userdata }
    // NOTE: userdata is still passed for pre-quantized input buffer pointer,
    //       but dimensions/weight are read directly from dst->src[1] for
    //       reliability
    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_gemv_custom, n_threads, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));

    return result;
}

inline struct ggml_tensor* ggml_mul_mat_gemv_batched(struct ggml_context* ctx, struct ggml_tensor* weight,
                                                     struct ggml_tensor* input, GemvBatchedUserData* userdata) {
    const int K = static_cast<int>(weight->ne[1]);  // Output dimension
    const int N = static_cast<int>(weight->ne[0]);  // Input dimension
    const int M = static_cast<int>(input->ne[1]);   // Batch columns
    if (K <= 0 || N <= 0 || M <= 0) {
        return ggml_mul_mat(ctx, weight, input);
    }

    userdata->weight_tensor = weight;
    userdata->N = N;
    userdata->K = K;
    userdata->M = M;
    userdata->weight_type = weight->type;
    userdata->input_quant_type = GGML_TYPE_F32;

    if (ggml_is_quantized(weight->type)) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
        if (!type_traits_cpu || !type_traits_cpu->vec_dot) {
            return ggml_mul_mat(ctx, weight, input);
        }

        const ggml_type vec_dot_type = type_traits_cpu->vec_dot_type;
        const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
        if (!input_type_traits || !input_type_traits->from_float) {
            return ggml_mul_mat(ctx, weight, input);
        }

        const size_t quant_row_size = ggml_row_size(vec_dot_type, static_cast<int64_t>(N));
        const size_t quant_row_stride = densecore::AlignUp(quant_row_size, static_cast<size_t>(64));
        if (quant_row_size == 0 || quant_row_stride > kMaxQuantInputBufferSize) {
            return ggml_mul_mat(ctx, weight, input);
        }

        userdata->input_quant_type = vec_dot_type;
        userdata->quant_row_stride = quant_row_stride;
    }

    const BatchSpec* batch = GetCurrentBatch();
    int n_threads = ResolveInferenceConfig(batch).num_threads;
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_threads = std::min(n_threads, physical_cores);
    }
    if (K < 256) {
        n_threads = std::min(n_threads, 2);
    } else if (K < 1024) {
        n_threads = std::min(n_threads, 4);
    }
    n_threads = std::max(1, std::min(n_threads, K));

    const int64_t ne_res[4] = {K, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_gemv_batched_custom, n_threads, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}

struct ggml_tensor* ggml_paged_attention_decode(struct ggml_context* ctx, struct ggml_tensor* q_cur,
                                                struct ggml_tensor* k_cur, struct ggml_tensor* v_cur,
                                                PagedAttentionUserData* userdata) {
    const int64_t ne_res[4] = {v_cur->ne[0], q_cur->ne[1], q_cur->ne[2], 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = q_cur;
    result->src[1] = k_cur;
    result->src[2] = v_cur;

    const BatchSpec* batch = GetCurrentBatch();
    int n_tasks = ResolveInferenceConfig(batch).num_threads;
    if (n_tasks <= 0) {
        n_tasks = std::thread::hardware_concurrency();
        if (n_tasks <= 0) n_tasks = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_tasks = std::min(n_tasks, physical_cores);
    }
    const int n_heads = static_cast<int>(q_cur->ne[1]);
    const int n_tokens = std::max(1, static_cast<int>(q_cur->ne[2]));
    const int head_tile = ResolvePagedAttentionDecodeHeadTile(n_heads, n_tokens, n_tasks);
    const int tiles_per_token = std::max(1, (std::max(1, n_heads) + head_tile - 1) / head_tile);
    const int total_tiles = std::max(1, n_tokens * tiles_per_token);
    // Preserve head-tiled parallelism for batched decode. Collapsing batch=2~4
    // to n_tasks=n_tokens strands CPU threads when each token still has
    // multiple head tiles to process.
    const bool token_parallel = (n_tokens > 1 && tiles_per_token == 1 && n_tokens <= n_tasks);
    n_tasks = std::max(1, std::min(n_tasks, total_tiles));
    // Scalar fallback remains correctness-first, but we keep the same token/head
    // tiling to preserve parallelism on non-Highway hosts.
    if (!token_parallel && !IsPagedAttentionHwyEnabled()) {
        const int scalar_tasks = ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_SCALAR_TASKS", n_tasks);
        n_tasks = std::max(1, std::min(scalar_tasks, total_tiles));
    }
    if (IsDecodeAttentionPathLoggingEnabled() && n_tokens > 1) {
        static std::atomic<uint64_t> paged_decode_task_logs{0};
        const uint64_t log_idx = paged_decode_task_logs.fetch_add(1, std::memory_order_relaxed);
        if (log_idx < 32) {
            std::cerr << "[PagedDecodeTasks] batch=" << n_tokens << " heads=" << n_heads
                      << " tiles_per_token=" << tiles_per_token << " total_tiles=" << total_tiles
                      << " n_tasks=" << n_tasks << " token_parallel=" << (token_parallel ? 1 : 0) << std::endl;
        }
    }
    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_paged_attention_decode, n_tasks, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));

    return result;
}

// ============================================================================
// oneDNN MatMul Custom Op (Prefill-only acceleration)
// ============================================================================

struct MatmulOpData {
    densecore::MatmulBackendKind backend;
    densecore::DType a_type;
    densecore::DType b_type;
    densecore::DType c_type;
    bool convert_a_f32_to_bf16 = false;
};

struct MatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    MatmulOpData data;
};

// GgmlTypeToDType is now defined in densecore/runtime/dtype_utils.h

void cb_matmul_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (ith != 0 || nth <= 0) {
        return;
    }

    const auto* params = reinterpret_cast<const MatmulCustomParams*>(dst->op_params);
    if (!params || !dst || !dst->src[0] || !dst->src[1]) return;
    const MatmulOpData& ud = params->data;

    const struct ggml_tensor* input = dst->src[0];
    const struct ggml_tensor* weight = dst->src[1];
    if (!input->data || !weight->data || !dst->data) return;

    const int64_t M = input->ne[1];
    const int64_t K = input->ne[0];
    const int64_t N = weight->ne[1];

    const int64_t lda = input->nb[1] / ggml_type_size(input->type);
    const int64_t ldb = weight->nb[1] / ggml_type_size(weight->type);
    const int64_t ldc = dst->nb[1] / ggml_type_size(dst->type);

    densecore::MatmulParams matmul_params;
    const void* a_ptr = input->data;
    InferenceWorkContext* work_ctx = GetCurrentWorkContext();
    if (!work_ctx) {
        throw densecore::InvalidArgumentException("cb_matmul_custom called without active InferenceWorkContext");
    }
    std::vector<ggml_bf16_t>& bf16_buffer = work_ctx->bf16_buffer;
    bool used_bf16_buffer = false;
    if (ud.convert_a_f32_to_bf16 && input->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_BF16 && M > 1) {
        const size_t total = static_cast<size_t>(M * K);
        if (bf16_buffer.size() < total) {
            bf16_buffer.resize(total);
        }
        for (int64_t m = 0; m < M; ++m) {
            const float* src =
                reinterpret_cast<const float*>(reinterpret_cast<const char*>(input->data) + m * input->nb[1]);
            ggml_fp32_to_bf16_row(src, bf16_buffer.data() + m * K, K);
        }
        a_ptr = bf16_buffer.data();
        used_bf16_buffer = true;
    }
    matmul_params.a = a_ptr;
    matmul_params.b = weight->data;
    matmul_params.c = dst->data;
    matmul_params.M = M;
    matmul_params.N = N;
    matmul_params.K = K;
    matmul_params.lda = used_bf16_buffer ? K : lda;
    matmul_params.ldb = ldb;
    matmul_params.ldc = ldc;
    matmul_params.trans_b = true;
    matmul_params.a_type = ud.a_type;
    matmul_params.b_type = ud.b_type;
    matmul_params.c_type = ud.c_type;

    densecore::MatmulBackend& backend = (ud.backend == densecore::MatmulBackendKind::OneDNN)
                                            ? densecore::GetOneDnnMatmulBackend()
                                            : densecore::GetDenseCoreMatmulBackend();
    backend.Execute(matmul_params);
}

inline struct ggml_tensor* ggml_mul_mat_onednn(struct ggml_context* ctx, struct ggml_tensor* weight,
                                               struct ggml_tensor* input, const MatmulOpData& data) {
    const int64_t N = weight->ne[1];  // Output dimension
    const int64_t M = input->ne[1];   // Tokens

    const int64_t ne_res[4] = {N, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    MatmulCustomParams params = {cb_matmul_custom, 1, nullptr, data};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));

    return result;
}

struct HalMatmulOpData {
    densecore::DeviceType preferred_device = densecore::DeviceType::CPU;
};

struct HalMatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    HalMatmulOpData data;
};

void cb_matmul_hal_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (ith != 0 || nth <= 0 || !dst || !dst->src[0] || !dst->src[1]) {
        return;
    }

    const auto* params = reinterpret_cast<const HalMatmulCustomParams*>(dst->op_params);
    if (!params) {
        return;
    }
    const struct ggml_tensor* input = dst->src[0];
    const struct ggml_tensor* weight = dst->src[1];
    if (!input->data || !weight->data || !dst->data) {
        return;
    }

    const bool input_contig =
        input->nb[0] == sizeof(float) && input->nb[1] == static_cast<size_t>(input->ne[0]) * sizeof(float);
    const bool weight_contig =
        weight->nb[0] == sizeof(float) && weight->nb[1] == static_cast<size_t>(weight->ne[0]) * sizeof(float);
    const bool output_contig =
        dst->nb[0] == sizeof(float) && dst->nb[1] == static_cast<size_t>(dst->ne[0]) * sizeof(float);

    if (input->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !input_contig ||
        !weight_contig || !output_contig) {
        // Fallback to local backend matmul when HAL path is inapplicable.
        densecore::MatmulParams matmul_params;
        matmul_params.a = input->data;
        matmul_params.b = weight->data;
        matmul_params.c = dst->data;
        matmul_params.M = input->ne[1];
        matmul_params.N = weight->ne[1];
        matmul_params.K = input->ne[0];
        matmul_params.lda = input->nb[1] / ggml_type_size(input->type);
        matmul_params.ldb = weight->nb[1] / ggml_type_size(weight->type);
        matmul_params.ldc = dst->nb[1] / ggml_type_size(dst->type);
        matmul_params.trans_b = true;
        matmul_params.a_type = densecore::DType::F32;
        matmul_params.b_type = densecore::DType::F32;
        matmul_params.c_type = densecore::DType::F32;
        densecore::GetDenseCoreMatmulBackend().Execute(matmul_params);
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    densecore::BackendRegistry& registry = ResolveBackendRegistry(batch);

    densecore::ComputeBackend* backend = registry.Get(params->data.preferred_device);
    if (!backend && params->data.preferred_device != densecore::DeviceType::CPU) {
        backend = registry.Get(densecore::DeviceType::CPU);
    }
    if (!backend) {
        return;
    }

    densecore::Tensor A = densecore::Tensor::Make2D(const_cast<void*>(input->data), input->ne[1], input->ne[0],
                                                    densecore::DType::F32, densecore::DeviceType::CPU);
    densecore::Tensor B = densecore::Tensor::Make2D(const_cast<void*>(weight->data), weight->ne[1], weight->ne[0],
                                                    densecore::DType::F32, densecore::DeviceType::CPU);
    densecore::Tensor C =
        densecore::Tensor::Make2D(dst->data, dst->ne[1], dst->ne[0], densecore::DType::F32, densecore::DeviceType::CPU);
    backend->MatMulTransB(A, B, &C);
}

inline struct ggml_tensor* ggml_mul_mat_hal(struct ggml_context* ctx, struct ggml_tensor* weight,
                                            struct ggml_tensor* input, densecore::DeviceType preferred_device) {
    const int64_t N = weight->ne[1];
    const int64_t M = input->ne[1];

    const int64_t ne_res[4] = {N, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    HalMatmulCustomParams params = {cb_matmul_hal_custom, 1, nullptr, {preferred_device}};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));
    return result;
}

struct HalAttentionOpData {
    float scale = 1.0f;
    int n_head_kv = -1;
    int q_start_offset = 0;
    int kv_start_offset = 0;
    int sliding_window = -1;
    float logit_softcap = 0.0f;
    uint32_t semantic_flags = 0;
    uint8_t causal = 1;
    uint8_t layout = static_cast<uint8_t>(HalAttentionTensorLayout::HeadSeq);
    densecore::DeviceType preferred_device = densecore::DeviceType::CPU;
    int layer = -1;
};

struct HalAttentionCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    HalAttentionOpData data;
};

static bool IsPortableFlashParityCheckEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_CHECK_PORTABLE_FLASH_ATTN");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static float PortableFlashParityTolerance() {
    static const float tol = []() {
        const char* env = std::getenv("DENSECORE_CHECK_PORTABLE_FLASH_ATTN_TOL");
        if (!env || env[0] == '\0') {
            return 1e-3f;
        }
        char* end = nullptr;
        const float parsed = std::strtof(env, &end);
        if (end == env || !std::isfinite(parsed) || parsed <= 0.0f) {
            return 1e-3f;
        }
        return parsed;
    }();
    return tol;
}

static bool ShouldRunPortableFlashParityCheck(int layer) {
    if (!IsPortableFlashParityCheckEnabled()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_CHECK_PORTABLE_FLASH_ATTN_LAYER", -1);
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_CHECK_PORTABLE_FLASH_ATTN_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static void ComputeFlashAttentionReference(const float* q, const float* k, const float* v, float* out, int n_head,
                                           int n_head_kv, int seq_q, int seq_kv, int head_dim, float scale, bool causal,
                                           int q_start_offset, int kv_start_offset, int sliding_window,
                                           float logit_softcap) {
    if (!q || !k || !v || !out || n_head <= 0 || n_head_kv <= 0 || seq_q <= 0 || seq_kv <= 0 || head_dim <= 0) {
        return;
    }

    const int n_rep = n_head / n_head_kv;
    for (int h = 0; h < n_head; ++h) {
        const int kv_head = h / n_rep;
        const float* q_head = q + static_cast<size_t>(h) * seq_q * head_dim;
        const float* k_head = k + static_cast<size_t>(kv_head) * seq_kv * head_dim;
        const float* v_head = v + static_cast<size_t>(kv_head) * seq_kv * head_dim;
        float* out_head = out + static_cast<size_t>(h) * seq_q * head_dim;

        for (int tq = 0; tq < seq_q; ++tq) {
            const float* q_row = q_head + static_cast<size_t>(tq) * head_dim;
            float* out_row = out_head + static_cast<size_t>(tq) * head_dim;
            std::fill(out_row, out_row + head_dim, 0.0f);

            std::vector<float> scores(static_cast<size_t>(seq_kv), -INFINITY);
            float row_max = -INFINITY;
            for (int tk = 0; tk < seq_kv; ++tk) {
                const int query_pos = q_start_offset + tq;
                const int key_pos = kv_start_offset + tk;
                if (causal && key_pos > query_pos) {
                    continue;
                }
                if (sliding_window >= 0 && key_pos < (query_pos - sliding_window)) {
                    continue;
                }
                const float* k_row = k_head + static_cast<size_t>(tk) * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q_row[d] * k_row[d];
                }
                if (logit_softcap > 0.0f) {
                    score = std::tanh(score / logit_softcap) * logit_softcap;
                }
                score *= scale;
                scores[static_cast<size_t>(tk)] = score;
                row_max = std::max(row_max, score);
            }

            if (!std::isfinite(row_max)) {
                continue;
            }

            float denom = 0.0f;
            for (int tk = 0; tk < seq_kv; ++tk) {
                float& score = scores[static_cast<size_t>(tk)];
                if (!std::isfinite(score)) {
                    score = 0.0f;
                    continue;
                }
                score = std::exp(score - row_max);
                denom += score;
            }
            if (!(denom > 0.0f) || !std::isfinite(denom)) {
                continue;
            }

            const float inv_denom = 1.0f / denom;
            for (int tk = 0; tk < seq_kv; ++tk) {
                const float p = scores[static_cast<size_t>(tk)] * inv_denom;
                if (!(p > 0.0f)) {
                    continue;
                }
                const float* v_row = v_head + static_cast<size_t>(tk) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    out_row[d] += p * v_row[d];
                }
            }
        }
    }
}

static densecore::Tensor MakeHalAttentionTensorFromGgml(const struct ggml_tensor* tensor, int n_head, int seq_len,
                                                        int head_dim) {
    densecore::Tensor wrapped = densecore::Tensor::Make4D(const_cast<void*>(tensor->data), 1, n_head, seq_len, head_dim,
                                                          densecore::DType::F32, densecore::DeviceType::CPU);
    wrapped.stride[0] = static_cast<int64_t>(n_head) * seq_len * head_dim;
    wrapped.stride[1] = static_cast<int64_t>(tensor->nb[1] / sizeof(float));
    wrapped.stride[2] = static_cast<int64_t>(tensor->nb[2] / sizeof(float));
    wrapped.stride[3] = static_cast<int64_t>(tensor->nb[0] / sizeof(float));
    return wrapped;
}

void cb_flash_attention_hal_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (nth <= 0 || !dst || !dst->src[0] || !dst->src[1] || !dst->src[2]) {
        return;
    }

    const auto* params = reinterpret_cast<const HalAttentionCustomParams*>(dst->op_params);
    if (!params) {
        return;
    }

    const struct ggml_tensor* q = dst->src[0];
    const struct ggml_tensor* k = dst->src[1];
    const struct ggml_tensor* v = dst->src[2];
    if (!q || !k || !v || !q->data || !k->data || !v->data || !dst->data) {
        return;
    }
    const bool direct_cpu_flash_path = params->data.preferred_device == densecore::DeviceType::CPU;
    if (!direct_cpu_flash_path && ith != 0) {
        return;
    }

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return;
    }
    if (q->ne[0] <= 0 || q->ne[1] <= 0 || q->ne[2] <= 0 || k->ne[0] <= 0 || k->ne[1] <= 0 || k->ne[2] <= 0 ||
        v->ne[0] <= 0 || v->ne[1] <= 0 || v->ne[2] <= 0) {
        return;
    }
    if (q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0] || k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2]) {
        return;
    }

    const HalAttentionTensorLayout layout = static_cast<HalAttentionTensorLayout>(params->data.layout);

    int head_dim = static_cast<int>(q->ne[0]);
    int seq_q = 0;
    int n_head = 0;
    int seq_kv = 0;
    int inferred_n_head_kv = 0;

    densecore::Tensor Q;
    densecore::Tensor K;
    densecore::Tensor V;
    densecore::Tensor O;

    if (layout == HalAttentionTensorLayout::HeadSeq) {
        if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_is_contiguous(v) || !ggml_is_contiguous(dst)) {
            return;
        }

        seq_q = static_cast<int>(q->ne[1]);
        n_head = static_cast<int>(q->ne[2]);
        seq_kv = static_cast<int>(k->ne[1]);
        inferred_n_head_kv = static_cast<int>(k->ne[2]);

        Q = densecore::Tensor::Make4D(const_cast<void*>(q->data), 1, n_head, seq_q, head_dim, densecore::DType::F32,
                                      densecore::DeviceType::CPU);
        K = densecore::Tensor::Make4D(const_cast<void*>(k->data), 1, inferred_n_head_kv, seq_kv, head_dim,
                                      densecore::DType::F32, densecore::DeviceType::CPU);
        V = densecore::Tensor::Make4D(const_cast<void*>(v->data), 1, inferred_n_head_kv, seq_kv, head_dim,
                                      densecore::DType::F32, densecore::DeviceType::CPU);
        O = densecore::Tensor::Make4D(dst->data, 1, n_head, seq_q, head_dim, densecore::DType::F32,
                                      densecore::DeviceType::CPU);
    } else {
        if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(float) || v->nb[0] != sizeof(float) ||
            dst->nb[0] != sizeof(float)) {
            return;
        }

        n_head = static_cast<int>(q->ne[1]);
        seq_q = static_cast<int>(q->ne[2]);
        inferred_n_head_kv = static_cast<int>(k->ne[1]);
        seq_kv = static_cast<int>(k->ne[2]);
        if (seq_q != 1) {
            return;
        }

        Q = MakeHalAttentionTensorFromGgml(q, n_head, seq_q, head_dim);
        K = MakeHalAttentionTensorFromGgml(k, inferred_n_head_kv, seq_kv, head_dim);
        V = MakeHalAttentionTensorFromGgml(v, inferred_n_head_kv, seq_kv, head_dim);
        O = MakeHalAttentionTensorFromGgml(dst, n_head, seq_q, head_dim);
    }

    const int n_head_kv = params->data.n_head_kv > 0 ? params->data.n_head_kv : inferred_n_head_kv;

    if (n_head_kv <= 0 || n_head <= 0 || seq_q <= 0 || seq_kv <= 0 || head_dim <= 0) {
        return;
    }
    if (n_head_kv != inferred_n_head_kv) {
        return;
    }
    if (n_head % n_head_kv != 0) {
        return;
    }

    if (direct_cpu_flash_path) {
        densecore::FlashAttentionConfig config = densecore::AutoTuneFlashConfig(head_dim, seq_kv);
        config.scale = params->data.scale;
        config.logit_softcap = params->data.logit_softcap;
        config.causal = params->data.causal != 0;
        config.sliding_window = params->data.sliding_window;
        config.num_threads = std::max(1, nth);
        config.q_start_offset = std::max(0, params->data.q_start_offset);
        config.kv_start_offset = std::max(0, params->data.kv_start_offset);
        config.semantic_flags = params->data.semantic_flags;

        if (layout == HalAttentionTensorLayout::HeadSeq) {
            if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_is_contiguous(v) || !ggml_is_contiguous(dst)) {
                return;
            }

            const float* q_data = reinterpret_cast<const float*>(q->data);
            const float* k_data = reinterpret_cast<const float*>(k->data);
            const float* v_data = reinterpret_cast<const float*>(v->data);
            float* o_data = reinterpret_cast<float*>(dst->data);

            #if defined(DENSECORE_X86) && !defined(__AVX512F__)
            if (ith == 0) {
                ComputeFlashAttentionReference(q_data, k_data, v_data, o_data, n_head, n_head_kv, seq_q, seq_kv,
                                               head_dim, params->data.scale, params->data.causal != 0,
                                               params->data.q_start_offset, params->data.kv_start_offset,
                                               params->data.sliding_window, params->data.logit_softcap);
            }
            #else
            if (n_head == n_head_kv) {
                densecore::FlashAttentionBatched(q_data, k_data, v_data, o_data, 1, n_head, seq_q, seq_kv, head_dim,
                                                 config, ith, nth);
            } else {
                densecore::FlashAttentionGQA(q_data, k_data, v_data, o_data, 1, n_head, n_head_kv, seq_q, seq_kv,
                                             head_dim, config, ith, nth);
            }
            #endif
        } else {
            if (seq_q != 1 || q->nb[0] != sizeof(float) || k->nb[0] != sizeof(float) || v->nb[0] != sizeof(float) ||
                dst->nb[0] != sizeof(float)) {
                return;
            }

            static thread_local densecore::FlashAttentionScratch tl_scratch;
            tl_scratch.Resize(config.block_m, config.block_n, head_dim);

            const int total_work = n_head;
            const int work_per_thread = std::max(1, (total_work + nth - 1) / nth);
            const int work_start = ith * work_per_thread;
            const int work_end = std::min(total_work, work_start + work_per_thread);
            const int n_rep = n_head / n_head_kv;

            const float* q_data = reinterpret_cast<const float*>(q->data);
            const float* k_data = reinterpret_cast<const float*>(k->data);
            const float* v_data = reinterpret_cast<const float*>(v->data);
            float* o_data = reinterpret_cast<float*>(dst->data);

            const int64_t q_head_stride = static_cast<int64_t>(q->nb[1] / sizeof(float));
            const int64_t kv_head_stride = static_cast<int64_t>(k->nb[1] / sizeof(float));
            const int64_t kv_seq_stride = static_cast<int64_t>(k->nb[2] / sizeof(float));
            const int64_t v_head_stride = static_cast<int64_t>(v->nb[1] / sizeof(float));
            const int64_t v_seq_stride = static_cast<int64_t>(v->nb[2] / sizeof(float));
            const int64_t o_head_stride = static_cast<int64_t>(dst->nb[1] / sizeof(float));

            for (int h = work_start; h < work_end; ++h) {
                const int h_kv = h / n_rep;
                const float* q_ptr = q_data + static_cast<ptrdiff_t>(h) * q_head_stride;
                const float* k_ptr = k_data + static_cast<ptrdiff_t>(h_kv) * kv_head_stride;
                const float* v_ptr = v_data + static_cast<ptrdiff_t>(h_kv) * v_head_stride;
                float* o_ptr = o_data + static_cast<ptrdiff_t>(h) * o_head_stride;

                densecore::FlashAttentionSingleQueryStridedKV(
                    q_ptr, k_ptr, kv_seq_stride, v_ptr, v_seq_stride, o_ptr, seq_kv, head_dim, config, tl_scratch);
            }
        }
        if (layout == HalAttentionTensorLayout::HeadSeq && ShouldRunPortableFlashParityCheck(params->data.layer) &&
            ith == 0) {
            const float* q_data = reinterpret_cast<const float*>(q->data);
            const float* k_data = reinterpret_cast<const float*>(k->data);
            const float* v_data = reinterpret_cast<const float*>(v->data);
            float* o_data = reinterpret_cast<float*>(dst->data);
            std::vector<float> ref(static_cast<size_t>(n_head) * seq_q * head_dim, 0.0f);
            ComputeFlashAttentionReference(q_data, k_data, v_data, ref.data(), n_head, n_head_kv, seq_q, seq_kv,
                                           head_dim, params->data.scale, params->data.causal != 0,
                                           params->data.q_start_offset, params->data.kv_start_offset,
                                           params->data.sliding_window, params->data.logit_softcap);

            float max_abs = 0.0f;
            int max_idx = -1;
            for (size_t i = 0; i < ref.size(); ++i) {
                const float diff = std::fabs(o_data[i] - ref[i]);
                if (diff > max_abs) {
                    max_abs = diff;
                    max_idx = static_cast<int>(i);
                }
            }
            if (max_abs > PortableFlashParityTolerance()) {
                const float got_val = (max_idx >= 0) ? o_data[max_idx] : 0.0f;
                const float ref_val = (max_idx >= 0) ? ref[static_cast<size_t>(max_idx)] : 0.0f;
                fprintf(stderr,
                        "[PortableFlashParity] FAIL layer=%d seq_q=%d seq_kv=%d n_head=%d n_head_kv=%d head_dim=%d "
                        "max_abs=%.6f idx=%d got=%.6f ref=%.6f\n",
                        params->data.layer, seq_q, seq_kv, n_head, n_head_kv, head_dim, max_abs, max_idx, got_val,
                        ref_val);
            }
        }
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    densecore::BackendRegistry& registry = ResolveBackendRegistry(batch);
    densecore::ComputeBackend* backend = registry.Get(params->data.preferred_device);
    if (!backend && params->data.preferred_device != densecore::DeviceType::CPU) {
        backend = registry.Get(densecore::DeviceType::CPU);
    }
    if (!backend) {
        return;
    }

    try {
        backend->FlashAttention(Q, K, V, &O, params->data.scale, params->data.causal != 0, n_head_kv,
                                params->data.sliding_window, params->data.logit_softcap,
                                params->data.semantic_flags);
    } catch (...) {
        densecore::ComputeBackend* cpu = registry.Get(densecore::DeviceType::CPU);
        if (cpu && cpu != backend) {
            cpu->FlashAttention(Q, K, V, &O, params->data.scale, params->data.causal != 0, n_head_kv,
                                params->data.sliding_window, params->data.logit_softcap,
                                params->data.semantic_flags);
        }
    }

    if (layout == HalAttentionTensorLayout::HeadSeq && ShouldRunPortableFlashParityCheck(params->data.layer)) {
        std::vector<float> ref(static_cast<size_t>(n_head) * seq_q * head_dim, 0.0f);
        ComputeFlashAttentionReference(reinterpret_cast<const float*>(q->data), reinterpret_cast<const float*>(k->data),
                                       reinterpret_cast<const float*>(v->data), ref.data(), n_head, n_head_kv, seq_q,
                                       seq_kv, head_dim, params->data.scale, params->data.causal != 0,
                                       params->data.q_start_offset, params->data.kv_start_offset,
                                       params->data.sliding_window,
                                       params->data.logit_softcap);

        const float* got = reinterpret_cast<const float*>(dst->data);
        float max_abs = 0.0f;
        int max_idx = -1;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float diff = std::fabs(got[i] - ref[i]);
            if (diff > max_abs) {
                max_abs = diff;
                max_idx = static_cast<int>(i);
            }
        }
        if (max_abs > PortableFlashParityTolerance()) {
            const float got_val = (max_idx >= 0) ? got[max_idx] : 0.0f;
            const float ref_val = (max_idx >= 0) ? ref[static_cast<size_t>(max_idx)] : 0.0f;
            fprintf(stderr,
                    "[PortableFlashParity] FAIL layer=%d seq_q=%d seq_kv=%d n_head=%d n_head_kv=%d head_dim=%d "
                    "max_abs=%.6f idx=%d got=%.6f ref=%.6f\n",
                    params->data.layer, seq_q, seq_kv, n_head, n_head_kv, head_dim, max_abs, max_idx, got_val, ref_val);
        }
    }
}

struct ggml_tensor* ggml_flash_attention_hal(struct ggml_context* ctx, struct ggml_tensor* Q, struct ggml_tensor* K,
                                             struct ggml_tensor* V, float scale, bool causal, int n_head_kv,
                                             int sliding_window, float logit_softcap, uint32_t semantic_flags,
                                             int layer, densecore::DeviceType preferred_device,
                                             HalAttentionTensorLayout layout = HalAttentionTensorLayout::HeadSeq,
                                             int q_start_offset = 0, int kv_start_offset = 0) {
    const int64_t ne_res[4] = {Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3]};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = Q;
    result->src[1] = K;
    result->src[2] = V;

    const int n_tasks = preferred_device == densecore::DeviceType::CPU ? GGML_N_TASKS_MAX : 1;
    HalAttentionCustomParams params = {cb_flash_attention_hal_custom,
                                       n_tasks,
                                       nullptr,
                                       {scale, n_head_kv, q_start_offset, kv_start_offset, sliding_window, logit_softcap,
                                        semantic_flags,
                                        static_cast<uint8_t>(causal ? 1 : 0), static_cast<uint8_t>(layout),
                                        preferred_device, layer}};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));
    return result;
}

struct Int4MatmulOpData {
    const uint8_t* packed_weights = nullptr;
    const float* scales = nullptr;
    const float* zeros = nullptr;
    int K = 0;
    int N = 0;
    int group_size = 0;
};

struct Int4MatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    Int4MatmulOpData data;
};

[[maybe_unused]] static void ComputeInt4MatmulReference(float* output, const float* input,
                                                        const uint8_t* packed_weights, const float* scales,
                                                        const float* zeros, int M, int K, int N, int group_size) {
    if (!output || !input || !packed_weights || !scales || M <= 0 || K <= 0 || N <= 0 || group_size <= 0 ||
        (K % group_size) != 0) {
        return;
    }

    const int num_groups = K / group_size;
    const int packed_K = (K + 1) / 2;

    for (int m = 0; m < M; ++m) {
        const float* a_row = input + static_cast<size_t>(m) * static_cast<size_t>(K);
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int g = 0; g < num_groups; ++g) {
                const float scale = scales[static_cast<size_t>(n) * static_cast<size_t>(num_groups) + g];
                const float zero = zeros ? zeros[static_cast<size_t>(n) * static_cast<size_t>(num_groups) + g] : 0.0f;
                const uint8_t* w_ptr = packed_weights + static_cast<size_t>(n) * static_cast<size_t>(packed_K) +
                                       static_cast<size_t>(g) * static_cast<size_t>(group_size / 2);
                const float* a_ptr = a_row + static_cast<size_t>(g) * static_cast<size_t>(group_size);

                for (int k = 0; k < group_size; ++k) {
                    int q = (w_ptr[k / 2] >> ((k & 1) ? 4 : 0)) & 0x0F;
                    if (q > 7) q -= 16;
                    acc += scale * (static_cast<float>(q) - zero) * a_ptr[k];
                }
            }
            output[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = acc;
        }
    }
}

static bool ShouldUseArmInt4DirectFastPath(const ggml_tensor* input, const Int4MatmulOpData& ud, int ith) {
#if defined(__aarch64__) || defined(_M_ARM64)
    if (!input || !ud.packed_weights || !ud.scales || !ud.zeros || ud.K <= 0 || ud.N <= 0 || ud.group_size <= 0 ||
        input->type != GGML_TYPE_F32 || input->ne[0] != ud.K || input->ne[1] <= 0) {
        return false;
    }

    const RuntimeToggleMode mode = GetArmInt4DirectFastPathMode();
    if (mode == RuntimeToggleMode::Off) {
        return false;
    }
    if (mode == RuntimeToggleMode::On) {
        return true;
    }

    static std::atomic<int> state{0};  // 0=unknown, 1=enabled, 2=disabled
    int current = state.load(std::memory_order_acquire);
    if (current != 0) {
        return current == 1;
    }

    if (ith != 0) {
        int spin_count = 0;
        while ((current = state.load(std::memory_order_acquire)) == 0) {
            SpinPause(spin_count++);
        }
        return current == 1;
    }

    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    current = state.load(std::memory_order_relaxed);
    if (current == 0) {
        if (!densecore::OpsRegistry::IsInitialized()) {
            densecore::OpsRegistry::Init();
        }
        auto& reg = densecore::OpsRegistry::Instance();

        bool ok = false;
        float max_abs_diff = std::numeric_limits<float>::infinity();

        if (reg.GemmInt4Batched) {
            const int M_check = std::min<int>(static_cast<int>(input->ne[1]), 2);
            const int N_check = std::min(ud.N, 32);
            const int K_check = ud.K;
            const int num_groups = K_check / ud.group_size;
            const int packed_K = (K_check + 1) / 2;

            std::vector<float> sample_input(static_cast<size_t>(M_check) * static_cast<size_t>(K_check));
            for (int m = 0; m < M_check; ++m) {
                const char* src_row =
                    reinterpret_cast<const char*>(input->data) + static_cast<size_t>(m) * input->nb[1];
                if (input->nb[0] == sizeof(float)) {
                    std::memcpy(sample_input.data() + static_cast<size_t>(m) * static_cast<size_t>(K_check), src_row,
                                static_cast<size_t>(K_check) * sizeof(float));
                } else {
                    for (int k = 0; k < K_check; ++k) {
                        sample_input[static_cast<size_t>(m) * static_cast<size_t>(K_check) + static_cast<size_t>(k)] =
                            *reinterpret_cast<const float*>(src_row + static_cast<size_t>(k) * input->nb[0]);
                    }
                }
            }

            std::vector<uint8_t> sample_weights(static_cast<size_t>(N_check) * static_cast<size_t>(packed_K));
            std::vector<float> sample_scales(static_cast<size_t>(N_check) * static_cast<size_t>(num_groups));
            std::vector<float> sample_zeros(static_cast<size_t>(N_check) * static_cast<size_t>(num_groups));
            for (int n = 0; n < N_check; ++n) {
                std::memcpy(sample_weights.data() + static_cast<size_t>(n) * static_cast<size_t>(packed_K),
                            ud.packed_weights + static_cast<size_t>(n) * static_cast<size_t>(packed_K),
                            static_cast<size_t>(packed_K));
                std::memcpy(sample_scales.data() + static_cast<size_t>(n) * static_cast<size_t>(num_groups),
                            ud.scales + static_cast<size_t>(n) * static_cast<size_t>(num_groups),
                            static_cast<size_t>(num_groups) * sizeof(float));
                std::memcpy(sample_zeros.data() + static_cast<size_t>(n) * static_cast<size_t>(num_groups),
                            ud.zeros + static_cast<size_t>(n) * static_cast<size_t>(num_groups),
                            static_cast<size_t>(num_groups) * sizeof(float));
            }

            std::vector<float> fast_output(static_cast<size_t>(M_check) * static_cast<size_t>(N_check), 0.0f);
            std::vector<float> ref_output(static_cast<size_t>(M_check) * static_cast<size_t>(N_check), 0.0f);

            reg.GemmInt4Batched(fast_output.data(), sample_input.data(), sample_weights.data(), sample_scales.data(),
                                sample_zeros.data(), M_check, K_check, N_check, ud.group_size, 0, M_check, 0, N_check,
                                static_cast<size_t>(K_check) * sizeof(float));
            ComputeInt4MatmulReference(ref_output.data(), sample_input.data(), sample_weights.data(),
                                       sample_scales.data(), sample_zeros.data(), M_check, K_check, N_check,
                                       ud.group_size);

            ok = true;
            max_abs_diff = 0.0f;
            for (size_t i = 0; i < fast_output.size(); ++i) {
                const float diff = std::fabs(fast_output[i] - ref_output[i]);
                max_abs_diff = std::max(max_abs_diff, diff);
                const float tol = std::max(1e-4f, 1e-4f * std::fabs(ref_output[i]));
                if (!std::isfinite(fast_output[i]) || diff > tol) {
                    ok = false;
                    break;
                }
            }
        }

        state.store(ok ? 1 : 2, std::memory_order_release);
        LogMatmulValidationOnce("arm_int4_direct_fastpath", ok, max_abs_diff);
        current = ok ? 1 : 2;
    }

    return current == 1;
#else
    (void)input;
    (void)ud;
    (void)ith;
    return true;
#endif
}

void cb_matmul_int4_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (nth <= 0) return;
    if (!dst || !dst->src[0]) return;

    const auto* params = reinterpret_cast<const Int4MatmulCustomParams*>(dst->op_params);
    if (!params) return;
    const Int4MatmulOpData& ud = params->data;
    if (!ud.packed_weights || !ud.scales || !ud.zeros || ud.K <= 0 || ud.N <= 0 || ud.group_size <= 0) return;

    const struct ggml_tensor* input = dst->src[0];
    if (!input || !input->data || !dst->data || input->type != GGML_TYPE_F32) return;

    const int64_t M = input->ne[1];
    const int64_t K = input->ne[0];
    if (K != ud.K || M <= 0) return;

    const bool input_contig =
        input->nb[0] == sizeof(float) && input->nb[1] == static_cast<size_t>(ud.K) * sizeof(float);
    const bool output_contig = dst->nb[0] == sizeof(float) && dst->nb[1] == static_cast<size_t>(ud.N) * sizeof(float);
    const bool single_threading_layer = IsInt4SingleThreadingLayerEnabled();
    const bool allow_direct_fast_path = ShouldUseArmInt4DirectFastPath(input, ud, ith);

    // Legacy escape hatch: keep DenseCore backend threadpool path for
    // platform-specific tuning. This path is intentionally serialized at GGML
    // level to avoid nested parallelism.
    if (!allow_direct_fast_path || !single_threading_layer) {
        if (ith != 0) return;

        static thread_local std::vector<float> legacy_contig_input;
        static thread_local std::vector<float> legacy_contig_output;
        const float* input_ptr = reinterpret_cast<const float*>(input->data);
        float* output_ptr = reinterpret_cast<float*>(dst->data);

        if (!input_contig || !output_contig) {
            legacy_contig_input.resize(static_cast<size_t>(M * ud.K));
            legacy_contig_output.resize(static_cast<size_t>(M * ud.N));

            for (int64_t m = 0; m < M; ++m) {
                const char* src_row = reinterpret_cast<const char*>(input->data) + m * input->nb[1];
                if (input->nb[0] == sizeof(float)) {
                    std::memcpy(legacy_contig_input.data() + m * ud.K, src_row,
                                static_cast<size_t>(ud.K) * sizeof(float));
                } else {
                    for (int k = 0; k < ud.K; ++k) {
                        legacy_contig_input[static_cast<size_t>(m) * ud.K + k] =
                            *reinterpret_cast<const float*>(src_row + static_cast<size_t>(k) * input->nb[0]);
                    }
                }
            }
            input_ptr = legacy_contig_input.data();
            output_ptr = legacy_contig_output.data();
        }

        densecore::Tensor A = densecore::Tensor::Make2D(const_cast<float*>(input_ptr), M, ud.K);
        densecore::Tensor W =
            densecore::Tensor::Make2D(const_cast<uint8_t*>(ud.packed_weights), ud.N, ud.K, densecore::DType::INT8);
        const int64_t groups_per_row = ud.K / ud.group_size;
        densecore::Tensor S = densecore::Tensor::Make2D(const_cast<float*>(ud.scales), ud.N, groups_per_row);
        densecore::Tensor Z = densecore::Tensor::Make2D(const_cast<float*>(ud.zeros), ud.N, groups_per_row);
        densecore::Tensor C = densecore::Tensor::Make2D(output_ptr, M, ud.N);
        densecore::GetCpuBackend().GemmInt4(A, W, S, Z, &C, ud.group_size);

        if (!input_contig || !output_contig) {
            for (int64_t m = 0; m < M; ++m) {
                char* dst_row = reinterpret_cast<char*>(dst->data) + m * dst->nb[1];
                if (dst->nb[0] == sizeof(float)) {
                    std::memcpy(dst_row, legacy_contig_output.data() + m * ud.N,
                                static_cast<size_t>(ud.N) * sizeof(float));
                } else {
                    for (int n = 0; n < ud.N; ++n) {
                        *reinterpret_cast<float*>(dst_row + static_cast<size_t>(n) * dst->nb[0]) =
                            legacy_contig_output[static_cast<size_t>(m) * ud.N + n];
                    }
                }
            }
        }
        return;
    }

    // GGML-thread partitioning over N tiles: each task writes a disjoint output
    // column range [n_start, n_end), so no synchronization is required.
    const int n_per_task = (ud.N + nth - 1) / nth;
    const int n_start = ith * n_per_task;
    const int n_end = std::min(ud.N, n_start + n_per_task);
    if (n_start >= n_end) return;

    // Use batched kernel whenever input elements are contiguous within each row
    // (nb[0] == sizeof(float)). The kernel handles row padding via input_stride_bytes,
    // eliminating the per-row gather/scatter fallback for padded GGML tensors.
    const bool input_elements_contig = input->nb[0] == sizeof(float);
    if (input_elements_contig && output_contig) {
        const float* in_ptr = reinterpret_cast<const float*>(input->data);
        float* out_ptr = reinterpret_cast<float*>(dst->data);
        densecore::Ops::GemmInt4Batched(out_ptr, in_ptr, ud.packed_weights, ud.scales, ud.zeros, static_cast<int>(M),
                                        ud.K, ud.N, ud.group_size, 0, static_cast<int>(M), n_start, n_end,
                                        input->nb[1]);
        return;
    }

    // Strided fallback: gather one input row at a time, compute assigned output
    // tile, scatter back. This path avoids backend threadpool usage entirely.
    static thread_local std::vector<float> gathered_input;
    static thread_local std::vector<float> partial_output;
    const int n_count = n_end - n_start;
    gathered_input.resize(static_cast<size_t>(ud.K));
    partial_output.resize(static_cast<size_t>(n_count));

    for (int64_t m = 0; m < M; ++m) {
        const char* src_row = reinterpret_cast<const char*>(input->data) + m * input->nb[1];
        if (input->nb[0] == sizeof(float)) {
            std::memcpy(gathered_input.data(), src_row, static_cast<size_t>(ud.K) * sizeof(float));
        } else {
            for (int k = 0; k < ud.K; ++k) {
                gathered_input[k] = *reinterpret_cast<const float*>(src_row + static_cast<size_t>(k) * input->nb[0]);
            }
        }

        densecore::hwy_kernels::GemvInt4_Hwy(partial_output.data(), gathered_input.data(), ud.packed_weights, ud.scales,
                                             ud.zeros, ud.K, ud.N, ud.group_size, n_start, n_end);

        char* dst_row = reinterpret_cast<char*>(dst->data) + m * dst->nb[1];
        if (dst->nb[0] == sizeof(float)) {
            float* dst_row_f32 = reinterpret_cast<float*>(dst_row);
            std::memcpy(dst_row_f32 + n_start, partial_output.data(), static_cast<size_t>(n_count) * sizeof(float));
        } else {
            for (int n = n_start; n < n_end; ++n) {
                *reinterpret_cast<float*>(dst_row + static_cast<size_t>(n) * dst->nb[0]) =
                    partial_output[static_cast<size_t>(n - n_start)];
            }
        }
    }
}

inline struct ggml_tensor* ggml_mul_mat_int4(struct ggml_context* ctx, struct ggml_tensor* weight,
                                             struct ggml_tensor* input,
                                             const TransformerModel::Int4WeightBinding& binding) {
    const int64_t M = input->ne[1];
    const int64_t ne_res[4] = {binding.n, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    Int4MatmulCustomParams params = {};
    params.fun = cb_matmul_int4_custom;
    const BatchSpec* batch = GetCurrentBatch();
    if (IsInt4SingleThreadingLayerEnabled()) {
        params.n_tasks = ResolveTaskCount(batch, static_cast<int>(std::max<int64_t>(1, binding.n)));
    } else {
        params.n_tasks = 1;
    }
    params.userdata = nullptr;
    params.data.packed_weights = reinterpret_cast<const uint8_t*>(binding.packed ? binding.packed->data : nullptr);
    params.data.scales = binding.scales ? reinterpret_cast<const float*>(binding.scales->data) : nullptr;
    params.data.zeros = binding.zeros ? reinterpret_cast<const float*>(binding.zeros->data) : nullptr;
    params.data.K = static_cast<int>(binding.k);
    params.data.N = static_cast<int>(binding.n);
    params.data.group_size = binding.group_size;

    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}

struct FP8MatmulOpData {
    const uint8_t* packed_weights = nullptr;
    int K = 0;
    int N = 0;
    TransformerModel::FP8Format format = TransformerModel::FP8Format::E4M3FN;
};

struct FP8MatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    FP8MatmulOpData data;
};

void cb_matmul_fp8_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (nth <= 0) return;
    if (!dst || !dst->src[0]) return;

    const auto* params = reinterpret_cast<const FP8MatmulCustomParams*>(dst->op_params);
    if (!params) return;
    const FP8MatmulOpData& ud = params->data;
    if (!ud.packed_weights || ud.K <= 0 || ud.N <= 0) return;

    const struct ggml_tensor* input = dst->src[0];
    if (!input || !input->data || !dst->data || input->type != GGML_TYPE_F32) return;

    const int64_t M = input->ne[1];
    const int64_t K = input->ne[0];
    if (K != ud.K || M <= 0) return;

    static std::once_flag fp8_lut_once;
    std::call_once(fp8_lut_once, []() { densecore::hwy_kernels::InitFP8LUTs_Hwy(); });

    // =========================================================================
    // FP8 Batched GEMM: Tile-based dequant for weight reuse across M tokens
    // =========================================================================
    // For M>1, dequantizing FP8 weights tile-by-tile and computing F32 GEMM
    // on tiles avoids reloading the full weight matrix for each token.
    //
    // Tile size: TILE_N weight rows × K elements dequantized to F32.
    // The F32 tile stays in L2 cache while all M activation rows access it.
    // =========================================================================
    if (M > 1) {
        const bool input_contig =
            input->nb[0] == sizeof(float) && input->nb[1] == static_cast<size_t>(ud.K) * sizeof(float);
        const bool output_contig =
            dst->nb[0] == sizeof(float) && dst->nb[1] == static_cast<size_t>(ud.N) * sizeof(float);

        // Thread-local dequant buffer: TILE_N × K floats
        constexpr int FP8_TILE_N = 8;  // 8 × 4096 × 4 = 128KB (fits L2)
        const int total_tiles = (ud.N + FP8_TILE_N - 1) / FP8_TILE_N;
        const int tiles_per_task = (total_tiles + nth - 1) / nth;
        const int tile_start = ith * tiles_per_task;
        const int tile_end = std::min(total_tiles, tile_start + tiles_per_task);
        if (tile_start >= tile_end) return;

        static thread_local std::vector<float> fp8_dequant_buf;
        fp8_dequant_buf.resize(static_cast<size_t>(FP8_TILE_N) * ud.K);

        const uint8_t* w_fp8 = ud.packed_weights;

        for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
            const int n = tile_idx * FP8_TILE_N;
            const int tile_n = std::min(FP8_TILE_N, ud.N - n);

            // Dequantize tile_n weight rows: FP8 → F32 (done ONCE per tile)
            for (int i = 0; i < tile_n; i++) {
                const uint8_t* w_row = w_fp8 + static_cast<size_t>(n + i) * ud.K;
                float* f32_row = fp8_dequant_buf.data() + static_cast<size_t>(i) * ud.K;
                if (ud.format == TransformerModel::FP8Format::E5M2) {
                    densecore::hwy_kernels::ConvertFP8E5M2ToFP32_Hwy(w_row, f32_row, static_cast<int64_t>(ud.K));
                } else {
                    densecore::hwy_kernels::ConvertFP8E4M3FNToFP32_Hwy(w_row, f32_row, static_cast<int64_t>(ud.K));
                }
            }

            // Tiled GEMM: C[:, n:n+tile_n] += A × W_tile^T
            // Weight tile in L2, activation rows cycle through L1
            for (int64_t m = 0; m < M; m++) {
                const float* a_row = nullptr;
                if (input_contig) {
                    a_row = reinterpret_cast<const float*>(input->data) + m * ud.K;
                } else {
                    a_row =
                        reinterpret_cast<const float*>(reinterpret_cast<const char*>(input->data) + m * input->nb[1]);
                }
                float* c_row = nullptr;
                if (output_contig) {
                    c_row = reinterpret_cast<float*>(dst->data) + m * ud.N;
                } else {
                    c_row = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + m * dst->nb[1]);
                }
                for (int i = 0; i < tile_n; i++) {
                    const float* w_row_f32 = fp8_dequant_buf.data() + static_cast<size_t>(i) * ud.K;
                    c_row[n + i] = densecore::simd::DotF32(a_row, w_row_f32, static_cast<size_t>(ud.K));
                }
            }
        }
        return;
    }

    // =========================================================================
    // M=1 DECODE: Single-token GEMV (existing optimized path)
    // =========================================================================
    const int64_t rows_per_task = (M + nth - 1) / nth;
    const int64_t m_start = static_cast<int64_t>(ith) * rows_per_task;
    const int64_t m_end = std::min<int64_t>(M, m_start + rows_per_task);
    if (m_start >= m_end) return;

    for (int64_t m = m_start; m < m_end; ++m) {
        const float* x_row =
            reinterpret_cast<const float*>(reinterpret_cast<const char*>(input->data) + m * input->nb[1]);
        float* y_row = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + m * dst->nb[1]);
        if (ud.format == TransformerModel::FP8Format::E5M2) {
            densecore::hwy_kernels::Gemv_FP8_E5M2_Hwy(/*M=*/ud.N, /*N=*/ud.K, /*alpha=*/1.0f, ud.packed_weights, x_row,
                                                      /*beta=*/0.0f, y_row);
        } else {
            densecore::hwy_kernels::Gemv_FP8_E4M3FN_Hwy(/*M=*/ud.N, /*N=*/ud.K, /*alpha=*/1.0f, ud.packed_weights,
                                                        x_row,
                                                        /*beta=*/0.0f, y_row);
        }
    }
}

inline struct ggml_tensor* ggml_mul_mat_fp8(struct ggml_context* ctx, struct ggml_tensor* weight,
                                            struct ggml_tensor* input,
                                            const TransformerModel::FP8WeightBinding& binding) {
    const int64_t M = input->ne[1];
    const int64_t ne_res[4] = {binding.n, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    FP8MatmulCustomParams params = {};
    params.fun = cb_matmul_fp8_custom;
    const BatchSpec* batch = GetCurrentBatch();
    int n_threads = ResolveInferenceConfig(batch).num_threads;
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores <= 0) physical_cores = 4;
    n_threads = std::min(n_threads, physical_cores);
    constexpr int FP8_TILE_N = 8;
    int64_t max_parallel_tasks = M;
    if (M > 1) {
        max_parallel_tasks = std::max<int64_t>(1, (binding.n + FP8_TILE_N - 1) / FP8_TILE_N);
    }
    const int64_t capped_tasks = std::min<int64_t>(static_cast<int64_t>(n_threads), max_parallel_tasks);
    n_threads = static_cast<int>(std::max<int64_t>(1, capped_tasks));
    params.n_tasks = n_threads;
    params.userdata = nullptr;
    params.data.packed_weights = reinterpret_cast<const uint8_t*>(binding.packed ? binding.packed->data : nullptr);
    params.data.K = static_cast<int>(binding.k);
    params.data.N = static_cast<int>(binding.n);
    params.data.format = binding.format;

    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}

/**
 * Smart matrix multiplication dispatcher
 *
 * CRITICAL:
 * - For decode-phase (batch=1), uses optimized GEMV path when dimensions are
 *   compatible.
 * - For small decode micro-batches (2 <= M <= 8), uses a custom batched path
 *   that reuses each weight row across M tokens.
 * - If incompatible (e.g., transposed layout mismatch), falls back to
 *   ggml_mul_mat which handles stride/transpose correctly.
 */
struct ggml_tensor* smart_mul_mat(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input,
                                  TransformerModel* model) {
    bool using_cpu_repack_alias = false;
    if (model && input && input->type == GGML_TYPE_F32) {
        auto it_repack = model->cpu_repack_aliases.find(weight);
        if (it_repack != model->cpu_repack_aliases.end() && it_repack->second) {
            weight = it_repack->second;
            using_cpu_repack_alias = true;
        }
    }
    const int M = static_cast<int>(input->ne[1]);
    const int K_dim = static_cast<int>(weight->ne[0]);
    const int N_dim = static_cast<int>(weight->ne[1]);
    const char* w_name = (weight->name[0] ? weight->name : "(unnamed)");
    const bool is_hybrid_ssm_qkv = model && model->arch_flags.is_hybrid_ssm && IsHybridSSMQkvWeightName(w_name);
    const bool is_qwen35_hybrid_ssm = model && model->variant == ModelVariant::QWEN35 && model->arch_flags.is_hybrid_ssm;
    const bool is_qwen35_hybrid_ssm_gate = is_qwen35_hybrid_ssm && std::strstr(w_name, "attn_gate");
    const bool is_qwen35_hybrid_ssm_out = is_qwen35_hybrid_ssm && std::strstr(w_name, "ssm_out");
    const bool is_qwen36_hybrid_ssm = model && model->variant == ModelVariant::QWEN36 &&
                                      model->arch_flags.is_hybrid_ssm;
    const bool is_qwen36_hybrid_ssm_qkv = is_qwen36_hybrid_ssm && std::strstr(w_name, "attn_qkv");
    const bool is_qwen36_hybrid_ssm_gate = is_qwen36_hybrid_ssm && std::strstr(w_name, "attn_gate");
    const bool is_qwen36_hybrid_ssm_out = model && model->variant == ModelVariant::QWEN36 &&
                                          model->arch_flags.is_hybrid_ssm && std::strstr(w_name, "ssm_out");
    const bool is_qwen36_lm_head = model && model->output == weight && model->variant == ModelVariant::QWEN36 &&
                                   model->arch_flags.is_hybrid_ssm;
    bool qwen36_lm_head_q4k_prefill_fast_path_eligible = false;
    if (is_qwen36_lm_head && M > 1 && weight->type == GGML_TYPE_Q4_K && input->type == GGML_TYPE_F32 &&
        weight->ne[0] == input->ne[0] && !IsBatchedQuantDisabled() && IsQ4KTrueBatchedKernelEnabled()) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
        qwen36_lm_head_q4k_prefill_fast_path_eligible =
            type_traits_cpu && type_traits_cpu->vec_dot && type_traits_cpu->vec_dot_type == GGML_TYPE_Q8_K &&
            (input->ne[0] % QK_K == 0);
    }

    // ========================================================================
    // HYBRID SSM QKV CONSERVATIVE FALLBACK (Highest priority for correctness)
    // ========================================================================
    // CRITICAL: This guard must remain at the absolute top of the function to 
    // prevent any optimized quantized/packed paths from returning silently 
    // wrong results for hybrid SSM projections.
    if (is_hybrid_ssm_qkv && ShouldForcePlainGgmlForHybridSSMQkv()) {
        LogMatmulDispatch(w_name, "PLAIN_GGML", M, N_dim, K_dim, "CONSERVATIVE_FALLBACK");
        LogHybridSSMQkvDispatch(w_name, weight->type, M, N_dim, K_dim, "PLAIN_GGML_CONSERVATIVE_FALLBACK", false, false, false);
        return ggml_mul_mat(ctx, weight, input);
    }

    if (using_cpu_repack_alias) {
        LogMatmulDispatch(w_name, MatmulWeightTypeLabel(weight->type, false, false), M, N_dim, K_dim,
                          "GGML_CPU_REPACK");
        return ggml_mul_mat(ctx, weight, input);
    }

    // Hybrid-SSM producer/output projections are correctness-critical on C4A.
    // Keep quantized large-M prefill for qkv, gate, and ssm_out on native GGML;
    // custom quant lanes for Qwen3.6 hybrid SSM have crashed the Go-server path.
    const bool force_plain_hybrid_ssm_prefill =
        is_qwen35_hybrid_ssm_gate || is_qwen35_hybrid_ssm_out ||
        (is_hybrid_ssm_qkv &&
         (!model || model->variant != ModelVariant::QWEN36)) ||
        is_qwen36_hybrid_ssm_qkv || is_qwen36_hybrid_ssm_gate || is_qwen36_hybrid_ssm_out;
    if (force_plain_hybrid_ssm_prefill &&
        ggml_is_quantized(weight->type) && input->type == GGML_TYPE_F32 && M > 1) {
        LogMatmulDispatch(w_name, MatmulWeightTypeLabel(weight->type, false, false), M, N_dim, K_dim, "GGML_NATIVE",
                          "hybrid_ssm_prefill_correctness");
        return ggml_mul_mat(ctx, weight, input);
    }

    // The wider qwen35 long-prefill graph is still unstable on C4A when the
    // custom quantized projection lane is used outside the SSM blocks as well.
    // Keep qwen35 quantized prefill projections on native GGML until the graph
    // is fully stable end-to-end. Decode (M==1) remains on the faster path.
    if (model && model->variant == ModelVariant::QWEN35 && ggml_is_quantized(weight->type) &&
        input->type == GGML_TYPE_F32 && M > 1) {
        LogMatmulDispatch(w_name, MatmulWeightTypeLabel(weight->type, false, false), M, N_dim, K_dim, "GGML_NATIVE",
                          "qwen35_prefill_quant_correctness");
        return ggml_mul_mat(ctx, weight, input);
    }

    const int gemma4_prefill_native_quant_max_cols =
        ParsePositiveEnvInt("DENSECORE_GEMMA4_PREFILL_NATIVE_QUANT_MAX_COLS", 512);
    if (model && model->arch_flags.is_gemma4 && ggml_is_quantized(weight->type) && input->type == GGML_TYPE_F32 &&
        M > 1 && M <= gemma4_prefill_native_quant_max_cols) {
        LogMatmulDispatch(w_name, MatmulWeightTypeLabel(weight->type, false, false), M, N_dim, K_dim, "GGML_NATIVE",
                          "gemma4_short_prefill_quant_correctness");
        return ggml_mul_mat(ctx, weight, input);
    }

    // Qwen3.6 long-prefill logits are release-critical for correctness. The
    // current ARM batched-quant path can silently corrupt large-M lm_head
    // results, while native GGML matmul remains correct. Keep the hot custom
    // path for inner model projections and decode, but force the lm_head back
    // to GGML once prompt batches get large enough to trigger the bad lane.
    if (is_qwen36_lm_head && ggml_is_quantized(weight->type) && M > 1 &&
        !qwen36_lm_head_q4k_prefill_fast_path_eligible) {
        LogMatmulDispatch(w_name, MatmulWeightTypeLabel(weight->type, false, false), M, N_dim, K_dim, "GGML_NATIVE",
                          "qwen36_lm_head_prefill_correctness");
        return ggml_mul_mat(ctx, weight, input);
    }

    // ========================================================================
    // PATH 1: DenseCore-packed INT4
    // ========================================================================
    if (model) {
        auto it_int4 = model->int4_weight_bindings.find(weight);
        if (it_int4 != model->int4_weight_bindings.end()) {
            const auto& binding = it_int4->second;
            if (binding.k > 0 && binding.n > 0 && binding.group_size > 0 && binding.packed && binding.scales &&
                binding.zeros && !IsPackedInt4CustomDisabled()) {
                LogMatmulDispatch(w_name, "PACKED_INT4", M, N_dim, K_dim,
                                  M == 1 ? "HWY_INT4_GEMV" : "HWY_INT4_BATCHED");
                LogHybridSSMQkvDispatch(w_name, weight->type, M, N_dim, K_dim,
                                        M == 1 ? "HWY_INT4_GEMV" : "HWY_INT4_BATCHED", false, false, true);
                return ggml_mul_mat_int4(ctx, weight, input, binding);
            }
        }

        // Custom FP8 packed path (optional metadata-driven route).
        auto it_fp8 = model->fp8_weight_bindings.find(weight);
        if (it_fp8 != model->fp8_weight_bindings.end()) {
            const auto& binding = it_fp8->second;
            if (binding.k > 0 && binding.n > 0 && binding.packed) {
                LogMatmulDispatch(w_name, "PACKED_FP8", M, N_dim, K_dim, M == 1 ? "FP8_GEMV" : "FP8_TILED_GEMM");
                LogHybridSSMQkvDispatch(w_name, weight->type, M, N_dim, K_dim, M == 1 ? "FP8_GEMV" : "FP8_TILED_GEMM",
                                        false, false, false);
                return ggml_mul_mat_fp8(ctx, weight, input, binding);
            }
        }
    }

    const int input_cols = M;

    // STRICT COMPATIBILITY CHECK (zero-overhead: evaluated at graph build time)
    const bool is_compatible = (weight->ne[0] == input->ne[0]);
    const BatchSpec* current_batch = GetCurrentBatch();
    const densecore::DeviceType preferred_matmul_device = ResolvePreferredMatmulDevice(current_batch);
    const bool hal_matmul_candidate = preferred_matmul_device != densecore::DeviceType::CPU && is_compatible &&
                                      weight->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 &&
                                      weight->nb[0] == static_cast<int64_t>(sizeof(float)) &&
                                      input->nb[0] == static_cast<int64_t>(sizeof(float));
    if (hal_matmul_candidate) {
        LogMatmulDispatch(w_name, "FLOAT_F32", M, N_dim, K_dim, "HAL_MATMUL_ROUTE");
        LogHybridSSMQkvDispatch(w_name, weight->type, M, N_dim, K_dim, "HAL_MATMUL_ROUTE", false, false, false);
        return ggml_mul_mat_hal(ctx, weight, input, preferred_matmul_device);
    }

    const int max_small_batch_cols = ParsePositiveEnvInt("DENSECORE_SMALL_BATCH_GEMV_MAX_COLS", kMaxSmallBatchColsHard);
    const int max_small_batch_quant_cols = ParsePositiveEnvInt("DENSECORE_SMALL_BATCH_GEMV_QUANT_MAX_COLS", 4096);
    const bool is_gemv_candidate =
        (input_cols == 1) && (weight->type == GGML_TYPE_F32 || ggml_is_quantized(weight->type));
    const bool is_small_batch_f32_candidate = (input_cols > 1 && input_cols <= max_small_batch_cols &&
                                               input->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32);
    bool has_quant_vec_dot = false;
    bool has_quant_from_float = false;
    bool quant_input_size_ok = false;
    bool quant_nrc_batch_ready = false;
    bool quant_true_batched_kernel_ready = false;
    if (ggml_is_quantized(weight->type) && input->type == GGML_TYPE_F32) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
        if (type_traits_cpu && type_traits_cpu->vec_dot) {
            has_quant_vec_dot = true;
            const auto* input_type_traits = ggml_get_type_traits_cpu(type_traits_cpu->vec_dot_type);
            if (input_type_traits && input_type_traits->from_float) {
                has_quant_from_float = true;
                const size_t quant_row_size = ggml_row_size(type_traits_cpu->vec_dot_type, input->ne[0]);
                quant_input_size_ok = (quant_row_size > 0 && quant_row_size <= kMaxQuantInputBufferSize);
            }
            const int vec_dot_nrows = std::max<int>(1, static_cast<int>(type_traits_cpu->nrows));
            quant_nrc_batch_ready =
                IsQuantNrcBatchEnabled() && vec_dot_nrows >= std::min(input_cols, kMaxSmallBatchColsHard);
            quant_true_batched_kernel_ready = IsQ4KTrueBatchedKernelEnabled() && (weight->type == GGML_TYPE_Q4_K) &&
                                              (type_traits_cpu->vec_dot_type == GGML_TYPE_Q8_K) &&
                                              (input->ne[0] % QK_K == 0);
        }
    }
    const bool is_small_batch_quant_candidate =
        (input_cols > 1 && input_cols <= max_small_batch_quant_cols && input->type == GGML_TYPE_F32 &&
         ggml_is_quantized(weight->type) && !IsBatchedQuantDisabled() && IsArmBatchedQuantSafeByDefault() &&
         has_quant_vec_dot && has_quant_from_float && quant_input_size_ok);
    const bool is_small_batch_candidate = is_small_batch_f32_candidate || is_small_batch_quant_candidate;

    const char* wtype_label = MatmulWeightTypeLabel(weight->type, false, false);

    if (is_gemv_candidate && IsDebugGemvSelectionEnabled()) {
        static int dbg_gemv_ct = 0;
        if (dbg_gemv_ct < 64) {
            fprintf(stderr, "[GEMV_SEL #%d] w=%s w.ne=[%ld,%ld] in.ne=[%ld,%ld] compat=%d q=%d\n", dbg_gemv_ct, w_name,
                    (long)weight->ne[0], (long)weight->ne[1], (long)input->ne[0], (long)input->ne[1],
                    is_compatible ? 1 : 0, ggml_is_quantized(weight->type) ? 1 : 0);
            dbg_gemv_ct++;
        }
    }

    // ========================================================================
    // PATH 2: M==1 GEMV (decode single-token)
    // ========================================================================
    if (!IsCustomGemvDisabled() && is_gemv_candidate && is_compatible) {
        if (ShouldForcePlainGgmlForHybridSSMQkv() && is_hybrid_ssm_qkv) {
            LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "GGML_NATIVE", "forced_ssm_qkv_plain_ggml");
            LogHybridSSMQkvDispatch(w_name, weight->type, M, N_dim, K_dim, "GGML_NATIVE", false, false, false);
            LogMatmulPathOnce("ggml_mul_mat");
            return ggml_mul_mat(ctx, weight, input);
        }
        if (input->type != GGML_TYPE_F32) {
            fprintf(stderr, "CRITICAL: smart_mul_mat input type is %d! Tensor name: %s\n", input->type, input->name);
        }
        LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim,
                          ggml_is_quantized(weight->type) ? "GEMV_QUANT" : "GEMV_F32");
        GemvUserData* ud = GetGemvUserData();
        ud->force_reference_scalar = false;
        const bool gemma4_q8_repacked_enabled =
            densecore::env::ParseNonZeroEnv("DENSECORE_GEMMA4_ENABLE_Q8_REPACKED_GEMV", false);
        ud->disable_q8_repacked_gemv = model && model->arch_flags.is_gemma4 && !gemma4_q8_repacked_enabled &&
                                       !IsGemma4SharedDenseFfnWeightName(w_name);
        LogHybridSSMQkvDispatch(w_name, weight->type, M, N_dim, K_dim,
                                ggml_is_quantized(weight->type) ? "GEMV_QUANT" : "GEMV_F32", false, false, false);
        return ggml_mul_mat_gemv(ctx, weight, input, ud);
    }

    // ========================================================================
    // PATH 3: Small-batch (2<=M<=8) custom batched path
    //   - GGML_QUANT: shared-quant + nrc=M vec_dot (weight row reuse)
    //   - FP32: batched dot with weight row reuse
    // ========================================================================
    if (!IsCustomGemvDisabled() && is_small_batch_candidate && is_compatible) {
        if (ShouldForcePlainGgmlForHybridSSMQkv() && is_hybrid_ssm_qkv) {
            LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "GGML_NATIVE", "forced_ssm_qkv_plain_ggml");
            LogHybridSSMQkvDispatch(w_name, weight->type, M, N_dim, K_dim, "GGML_NATIVE", false, false, false);
            LogMatmulPathOnce("ggml_mul_mat");
            return ggml_mul_mat(ctx, weight, input);
        }
        if (is_small_batch_quant_candidate) {
            const bool use_true_batched_q4k_path = quant_true_batched_kernel_ready && weight->type == GGML_TYPE_Q4_K;
            LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim,
                              use_true_batched_q4k_path ? "GGML_QUANT_Q4K_TRUE_BATCHED" : "GGML_QUANT_NRC_M");
        } else {
            LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "BATCHED_F32");
        }
        GemvBatchedUserData* ud = GetGemvBatchedUserData();
        ud->force_reference_scalar = false;
        LogHybridSSMQkvDispatch(w_name, weight->type, M, N_dim, K_dim,
                                is_small_batch_quant_candidate
                                    ? (quant_true_batched_kernel_ready && weight->type == GGML_TYPE_Q4_K
                                           ? "GGML_QUANT_Q4K_TRUE_BATCHED"
                                           : "GGML_QUANT_NRC_M")
                                    : "BATCHED_F32",
                                is_small_batch_quant_candidate, false, false);
        return ggml_mul_mat_gemv_batched(ctx, weight, input, ud);
    }

    // Log fallback reasons for quant weights that didn't take the batched path
    if (IsDebugMatmulDispatchEnabled() && input_cols > 1 && ggml_is_quantized(weight->type) && is_compatible &&
        !is_small_batch_quant_candidate) {
        const char* reason = "unknown";
        if (IsCustomGemvDisabled())
            reason = "custom_gemv_disabled";
        else if (input_cols > max_small_batch_quant_cols)
            reason = "M>max_quant_cols";
        else if (IsBatchedQuantDisabled())
            reason = "batched_quant_disabled";
        else if (!IsArmBatchedQuantSafeByDefault())
            reason = "batched_quant_arm_guard";
        else if (!has_quant_vec_dot)
            reason = "no_vec_dot";
        else if (!has_quant_from_float)
            reason = "no_from_float";
        else if (!quant_input_size_ok)
            reason = "quant_input_too_large";
        else if (!quant_nrc_batch_ready && !quant_true_batched_kernel_ready)
            reason = "quant_batched_kernel_unavailable";
        LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "GGML_FALLBACK", reason);
    }

    // ========================================================================
    // PATH 4: OneDNN for large batches (prefill)
    // ========================================================================
    if (is_compatible && input_cols > 1) {
        densecore::MatmulParams params;
        params.M = input->ne[1];
        params.K = input->ne[0];
        params.N = weight->ne[1];
        params.lda = input->nb[1] / ggml_type_size(input->type);
        params.ldb = weight->nb[1] / ggml_type_size(weight->type);
        params.ldc = params.N;
        params.trans_b = true;
        params.a_type = densecore::GgmlTypeToDType(input->type);
        params.b_type = densecore::GgmlTypeToDType(weight->type);
        params.c_type = densecore::DType::F32;

        const bool convert_f32_to_bf16 =
            (input->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_BF16 && input_cols > 1);
        if (convert_f32_to_bf16) {
            params.a_type = densecore::DType::BF16;
        }

        if (densecore::SelectMatmulBackend(params, true) == densecore::MatmulBackendKind::OneDNN) {
            LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "ONEDNN_GEMM");
            LogHybridSSMQkvDispatch(w_name, weight->type, M, N_dim, K_dim, "ONEDNN_GEMM", false, false, false);
            MatmulOpData data;
            data.backend = densecore::MatmulBackendKind::OneDNN;
            data.a_type = params.a_type;
            data.b_type = params.b_type;
            data.c_type = params.c_type;
            data.convert_a_f32_to_bf16 = convert_f32_to_bf16;
            return ggml_mul_mat_onednn(ctx, weight, input, data);
        }
    }

    // Standard GGML fallback (handles transpose/stride correctly)
    LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "GGML_NATIVE");
    LogHybridSSMQkvDispatch(w_name, weight->type, M, N_dim, K_dim, "GGML_NATIVE", false, false, false);
    LogMatmulPathOnce("ggml_mul_mat");
    return ggml_mul_mat(ctx, weight, input);
}

// ============================================================================
// SIMPLIFIED UNIVERSAL ATTENTION (llama.cpp style)
// This version trades the complex paged KV cache for correctness and
// clarity. Once working, KV cache can be re-added following the proven
// llama.cpp pattern.
// ============================================================================

// ============================================================================
// MoE Forward Callbacks
// ============================================================================

// Helper: Convert GGML tensors to DenseCore HAL tensors using DenseCore's
// row-major convention. GGML stores 2D tensors as [cols, rows], while
// DenseCore expects [rows, cols].
inline densecore::Tensor GgmlToRowMajorTensor(const struct ggml_tensor* t) {
    if (!t) {
        return {};
    }

    densecore::Tensor out;
    out.data = t->data;
    out.dtype = densecore::GgmlTypeToDType(t->type);
    out.shape[0] = t->ne[1];
    out.shape[1] = t->ne[0];
    out.shape[2] = t->ne[2];
    out.shape[3] = t->ne[3];
    out.stride[0] = t->ne[0];
    out.stride[1] = 1;
    out.stride[2] = t->nb[2];
    out.stride[3] = t->nb[3];
    out.ndim = 2;
    out.device_type = densecore::DeviceType::CPU;
    return out;
}
