bool ExpertUsesPackedInt4Only(const CpuBackend::ExpertWeights& expert) {
    if (!expert.w1_int4.IsValid() || !expert.w2_int4.IsValid()) {
        return false;
    }
    if (expert.w3.ptr != nullptr && !expert.w3_int4.IsValid()) {
        return false;
    }
    return true;
}

// Check if expert weights are in a ggml quantized format that supports vec_dot.
// Examples: Q4_K, Q4_0, Q6_K. This enables zero-dequantization GEMV.
bool ExpertHasGgmlQuantizedWeights(const CpuBackend::ExpertWeights& expert) {
    if (!expert.w1.ptr || !expert.w2.ptr) return false;
    const auto check_type = [](int type_id) -> bool {
        if (type_id == GGML_TYPE_F32) return false;
        const ggml_type wtype = static_cast<ggml_type>(type_id);
        if (!ggml_is_quantized(wtype)) return false;
        const auto* tc = ggml_get_type_traits_cpu(wtype);
        return tc && tc->vec_dot;
    };
    return check_type(expert.w1_type) && check_type(expert.w2_type) &&
           (expert.w3.ptr == nullptr || check_type(expert.w3_type));
}

// Run ggml vec_dot based GEMV for quantized expert weights.
// This bypasses F32 dequantization on the hot MoE path.
// Complexity: O(M * N * K) where expert matrices are [N, K].
bool TryRunGgmlQuantizedProjection(CpuBackend* backend, const void* weight_ptr, int ggml_type_id, const Tensor& input,
                                   Tensor* output, int64_t N, int64_t K, int numa_node, bool allow_parallel = true,
                                   QuantizedProjectionInputCache* input_cache = nullptr,
                                   bool allow_kquant_rowpair_vec_dot = true, bool prefer_q4k_repacked_prefill = false) {
    if (!backend || !weight_ptr || !output || !input.IsValid() || !output->IsValid()) return false;
    if (input.dtype != DType::F32 || output->dtype != DType::F32) return false;
    if (input.ndim != 2 || output->ndim != 2) return false;

    const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
    if (wtype == GGML_TYPE_F32 || !ggml_is_quantized(wtype)) return false;
    InferenceWorkContext* census_ctx = GetCurrentWorkContext();
    RecordMoEExpertMatmulWeightType(census_ctx, wtype);
    const bool dispatch_census_enabled = []() {
        const char* env = std::getenv("DENSECORE_MATMUL_DISPATCH_CENSUS");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 &&
               std::strcmp(env, "off") != 0;
    }();
    const auto dispatch_begin =
        dispatch_census_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto record_dispatch = [&](const char* path) {
        if (!dispatch_census_enabled) {
            return;
        }
        const uint64_t wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - dispatch_begin)
                .count());
        RecordMatmulDispatchCensus(census_ctx, GetCurrentExecutionPhase(), path, wtype, input.shape[0], N, K, wall_ns);
    };

    const auto* type_traits = ggml_get_type_traits(wtype);
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(wtype);
    if (!type_traits || !type_traits_cpu || !type_traits_cpu->vec_dot) return false;

    const int64_t M = input.shape[0];
    if (M <= 0 || M > kMoEQuantizedProjectionMaxBatch || K != input.shape[1] || N != output->shape[1]) return false;

    // Resolve the input quantization type required by vec_dot
    const ggml_type iq_type = type_traits_cpu->vec_dot_type;
    const auto* iq_traits = ggml_get_type_traits_cpu(iq_type);
    if (!iq_traits || !iq_traits->from_float) return false;

    const size_t w_row_bytes = ggml_row_size(wtype, K);
    const size_t iq_row_bytes = ggml_row_size(iq_type, K);
    const float* in_data = input.DataAs<float>();
    float* out_data = output->DataAs<float>();
    const char* w_data = static_cast<const char*>(weight_ptr);

    const size_t total_qbytes = static_cast<size_t>(M) * iq_row_bytes;
    uint8_t* qinput_data = nullptr;
    if (input_cache) {
        const bool cache_hit = input_cache->source == in_data && input_cache->rows == M && input_cache->cols == K &&
                               input_cache->type == iq_type && input_cache->row_bytes == iq_row_bytes &&
                               input_cache->bytes.size() >= total_qbytes;
        if (!cache_hit) {
            input_cache->source = in_data;
            input_cache->rows = M;
            input_cache->cols = K;
            input_cache->type = iq_type;
            input_cache->row_bytes = iq_row_bytes;
            input_cache->bytes.resize(total_qbytes);
            for (int64_t m = 0; m < M; ++m) {
                iq_traits->from_float(in_data + m * K,
                                      input_cache->bytes.data() + static_cast<size_t>(m) * iq_row_bytes, K);
            }
        }
        qinput_data = input_cache->bytes.data();
    } else {
        // Quantize the active expert batch once, then reuse it for every expert row.
        static thread_local std::vector<uint8_t> qinput_buf;
        if (qinput_buf.size() < total_qbytes) qinput_buf.resize(total_qbytes);
        for (int64_t m = 0; m < M; ++m) {
            iq_traits->from_float(in_data + m * K, qinput_buf.data() + static_cast<size_t>(m) * iq_row_bytes, K);
        }
        qinput_data = qinput_buf.data();
    }
    const bool q4k_repacked_candidate = wtype == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && (N % 8) == 0 &&
                                        (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0;
    bool q4k_repacked_reported = false;
    const auto record_q4k_repacked = [&](bool used, const char* reject_reason) {
        if (q4k_repacked_candidate && !q4k_repacked_reported) {
            RecordMoEQ4KRepackedDecision(census_ctx, true, used, reject_reason);
            q4k_repacked_reported = true;
        }
    };

    const bool use_kquant_rowpair_vec_dot = allow_kquant_rowpair_vec_dot && CanUseKQuantRowPairVecDotFastPath() &&
                                            (wtype == GGML_TYPE_Q4_K || wtype == GGML_TYPE_Q6_K) &&
                                            iq_type == GGML_TYPE_Q8_K && M >= 2 && K % ggml_blck_size(wtype) == 0 &&
                                            N >= 2;
    const bool q4k_prefill_repacked_eligible = CanUseQ4KRepackedMoEPrefillFastPath() && wtype == GGML_TYPE_Q4_K &&
                                               iq_type == GGML_TYPE_Q8_K && M > 1 && (N % 8) == 0 &&
                                               (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0;
    const auto try_q5k_repacked_gemv = [&]() -> bool {
        if (wtype != GGML_TYPE_Q5_K) {
            return false;
        }
        const bool q5k_input_supported = iq_type == GGML_TYPE_Q8_K;
        const bool q5k_shape_supported =
            q5k_input_supported && (N % 8) == 0 && (K % ggml_blck_size(GGML_TYPE_Q5_K)) == 0;
        if (!q5k_input_supported) {
            RecordMoEQ5KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/false,
                                         "unsupported_input_quant_type");
            return false;
        }
        if (!q5k_shape_supported) {
            RecordMoEQ5KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/false, "unsupported_shape");
            return false;
        }
        if (!CanUseQ5KRepackedMoEGemvFastPath()) {
            RecordMoEQ5KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/false, "kernel_unavailable");
            return false;
        }
        auto packed = GetOrCreateQ5KRepackedMoEWeight(weight_ptr, N, K);
        if (packed && RunQ5KRepackedMoEGemv(backend, packed, qinput_data, iq_row_bytes, out_data, M, N, numa_node,
                                            allow_parallel)) {
            RecordMoEQ5KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/true, nullptr);
            LogMoEMatmulPath("ggml_q5k_repacked_gemv", static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0,
                             allow_parallel);
            record_dispatch("moe_expert");
            return true;
        }
        RecordMoEQ5KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/false, "pack_or_run_failed");
        return false;
    };
    const bool enable_q4k_repacked_projection_prefill = true;
    if (enable_q4k_repacked_projection_prefill && q4k_prefill_repacked_eligible && prefer_q4k_repacked_prefill) {
        auto packed = GetOrCreateQ4KRepackedMoEWeight(weight_ptr, N, K);
        if (packed && M >= 4 &&
            RunQ4KRepackedMoEGemmM4(backend, packed, in_data, out_data, M, N, K, numa_node, allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_repacked_prefill_gemm_m4", static_cast<int>(M), static_cast<int>(K),
                             static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(true, nullptr);
            record_dispatch("q4k_repacked_gemv");
            return true;
        }
        if (packed && RunQ4KRepackedMoEGemv(backend, packed, qinput_data, iq_row_bytes, out_data, M, N, numa_node,
                                            allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_repacked_prefill_gemv", static_cast<int>(M), static_cast<int>(K),
                             static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(true, nullptr);
            record_dispatch("q4k_repacked_gemv");
            return true;
        }
    }
    if (CanUseMoEQ4KRawBatchedScalar() && wtype == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && M > 1 &&
        K % ggml_blck_size(GGML_TYPE_Q4_K) == 0) {
        if (RunMoEQ4KRawBatchedProjectionImpl(backend, weight_ptr, qinput_data, iq_row_bytes, out_data, M, N, K,
                                              numa_node, allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_raw_batched", static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0,
                             allow_parallel);
            record_q4k_repacked(false, "raw_batched_used");
            record_dispatch("moe_expert");
            return true;
        }
    }
    if (q4k_prefill_repacked_eligible && !prefer_q4k_repacked_prefill) {
        record_q4k_repacked(false, "prefill_repack_not_preferred");
    }
    if (use_kquant_rowpair_vec_dot) {
        auto& pool = backend->GetThreadPool(numa_node);
        const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
        const int64_t pair_count = N / 2;
        for (int64_t m = 0; m < M; ++m) {
            float* out_row = out_data + static_cast<size_t>(m) * N;
            const void* qi = qinput_data + static_cast<size_t>(m) * iq_row_bytes;
            const auto compute_pair_range = [&](int pair_start, int pair_end) {
                for (int pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t n = static_cast<int64_t>(pair) * 2;
                    const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                    float sums[32] = {};
                    type_traits_cpu->vec_dot(static_cast<int>(K), sums, 2, w_row, w_row_bytes, qi, 0, 2);
                    out_row[n] = sums[0];
                    out_row[n + 1] = sums[1];
                }
            };
            if (n_threads <= 1) {
                compute_pair_range(0, static_cast<int>(pair_count));
            } else {
                pool.ParallelFor(static_cast<int>(pair_count),
                                 [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
            }
            if ((N & 1) != 0) {
                const int64_t n = N - 1;
                const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
            }
        }
        const char* path_name = wtype == GGML_TYPE_Q4_K ? "ggml_q4k_rowpair_m2_vecdot" : "ggml_q6k_rowpair_m2_vecdot";
        LogMoEMatmulPath(path_name, static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
        record_q4k_repacked(false, "rowpair_used");
        record_dispatch("moe_rowblock");
        return true;
    }

    if (try_q5k_repacked_gemv()) {
        return true;
    }

    const bool use_q5k_colpair_vec_dot = allow_kquant_rowpair_vec_dot && CanUseKQuantRowPairVecDotFastPath() &&
                                         wtype == GGML_TYPE_Q5_K && iq_type == GGML_TYPE_Q8_K &&
                                         K % ggml_blck_size(GGML_TYPE_Q5_K) == 0 && N >= 2;
    if (use_q5k_colpair_vec_dot) {
        auto& pool = backend->GetThreadPool(numa_node);
        const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
        const int64_t pair_count = N / 2;
        for (int64_t m = 0; m < M; ++m) {
            float* out_row = out_data + static_cast<size_t>(m) * N;
            const void* qi = qinput_data + static_cast<size_t>(m) * iq_row_bytes;
            const auto compute_pair_range = [&](int pair_start, int pair_end) {
                for (int pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t n = static_cast<int64_t>(pair) * 2;
                    const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                    float sums[32] = {};
                    type_traits_cpu->vec_dot(static_cast<int>(K), sums, 2, w_row, w_row_bytes, qi, 0, 2);
                    out_row[n] = sums[0];
                    out_row[n + 1] = sums[1];
                }
            };
            if (n_threads <= 1) {
                compute_pair_range(0, static_cast<int>(pair_count));
            } else {
                pool.ParallelFor(static_cast<int>(pair_count),
                                 [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
            }
            if ((N & 1) != 0) {
                const int64_t n = N - 1;
                const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
            }
        }
        LogMoEMatmulPath("ggml_q5k_colpair_vecdot", static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0,
                         allow_parallel);
        record_dispatch("moe_rowblock");
        return true;
    }

    if (wtype == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && (N % 8) == 0 &&
        (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0) {
        auto packed = GetOrCreateQ4KRepackedMoEWeight(weight_ptr, N, K);
        if (packed && RunQ4KRepackedMoEGemv(backend, packed, qinput_data, iq_row_bytes, out_data, M, N, numa_node,
                                            allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_repacked_gemv", static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0,
                             allow_parallel);
            record_q4k_repacked(true, nullptr);
            record_dispatch("q4k_repacked_gemv");
            return true;
        }
    }

    const bool use_q4k_rowpair_vec_dot =
        allow_kquant_rowpair_vec_dot && CanUseQ4KRowPairVecDotFastPath() &&
        (wtype == GGML_TYPE_Q4_K || wtype == GGML_TYPE_Q5_K || wtype == GGML_TYPE_Q5_1 || wtype == GGML_TYPE_Q8_0) &&
        (iq_type == GGML_TYPE_Q8_K || iq_type == GGML_TYPE_Q8_1 || iq_type == GGML_TYPE_Q8_0) && M == 1 &&
        K % ggml_blck_size(wtype) == 0 && N >= 4;
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    for (int64_t m = 0; m < M; ++m) {
        float* out_row = out_data + m * N;
        const void* qi = qinput_data + static_cast<size_t>(m) * iq_row_bytes;

        if (use_q4k_rowpair_vec_dot) {
            const int64_t pair_count = N / 2;
            const auto compute_pair_range = [&](int pair_start, int pair_end) {
                for (int pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t n = static_cast<int64_t>(pair) * 2;
                    const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                    float sums[32] = {};
                    type_traits_cpu->vec_dot(static_cast<int>(K), sums, 2, w_row, w_row_bytes, qi, 0, 2);
                    out_row[n] = sums[0];
                    out_row[n + 1] = sums[1];
                }
            };
            if (n_threads <= 1) {
                compute_pair_range(0, static_cast<int>(pair_count));
            } else {
                pool.ParallelFor(static_cast<int>(pair_count),
                                 [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
            }
            if ((N & 1) != 0) {
                const int64_t n = N - 1;
                const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
            }
        } else if (n_threads <= 1) {
            for (int64_t n = 0; n < N; ++n) {
                const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
            }
        } else {
            pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) {
                for (int n = n_start; n < n_end; ++n) {
                    const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                    type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
                }
            });
        }
    }
    if (use_q4k_rowpair_vec_dot) {
        const char* path = wtype == GGML_TYPE_Q5_K   ? "ggml_q5k_rowpair_m1_vecdot"
                           : wtype == GGML_TYPE_Q5_1 ? "ggml_q5_1_rowpair_m1_vecdot"
                           : wtype == GGML_TYPE_Q8_0 ? "ggml_q8_0_rowpair_m1_vecdot"
                                                     : "ggml_q4k_rowpair_m1_vecdot";
        LogMoEMatmulPath(path, static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
    }
    record_q4k_repacked(false, use_q4k_rowpair_vec_dot ? "rowpair_used" : "native_vecdot");
    record_dispatch(use_q4k_rowpair_vec_dot ? "moe_rowblock" : "moe_expert");
    return true;
}

bool TryRunGgmlQuantizedFusedGEGLUProjection(CpuBackend* backend, const void* gate_weight_ptr, int gate_ggml_type_id,
                                             const void* up_weight_ptr, int up_ggml_type_id, const Tensor& input,
                                             Tensor* output, int64_t N, int64_t K, int numa_node,
                                             bool allow_parallel = true,
                                             QuantizedProjectionInputCache* input_cache = nullptr,
                                             bool prefer_q4k_repacked_prefill = false) {
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !output || !input.IsValid() || !output->IsValid()) {
        return false;
    }
    if (input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

    const ggml_type gate_type = static_cast<ggml_type>(gate_ggml_type_id);
    const ggml_type up_type = static_cast<ggml_type>(up_ggml_type_id);
    if (gate_type == GGML_TYPE_F32 || up_type == GGML_TYPE_F32 || !ggml_is_quantized(gate_type) ||
        !ggml_is_quantized(up_type)) {
        return false;
    }

    const auto* gate_traits = ggml_get_type_traits(gate_type);
    const auto* up_traits = ggml_get_type_traits(up_type);
    const auto* gate_traits_cpu = ggml_get_type_traits_cpu(gate_type);
    const auto* up_traits_cpu = ggml_get_type_traits_cpu(up_type);
    if (!gate_traits || !up_traits || !gate_traits_cpu || !up_traits_cpu || !gate_traits_cpu->vec_dot ||
        !up_traits_cpu->vec_dot) {
        return false;
    }
    if (gate_traits_cpu->vec_dot_type != up_traits_cpu->vec_dot_type) {
        return false;
    }

    const int64_t M = input.shape[0];
    if (M <= 0 || M > kMoEQuantizedProjectionMaxBatch || K != input.shape[1] || N != output->shape[1]) {
        return false;
    }

    const ggml_type iq_type = gate_traits_cpu->vec_dot_type;
    const auto* iq_traits = ggml_get_type_traits_cpu(iq_type);
    if (!iq_traits || !iq_traits->from_float) {
        return false;
    }

    const size_t iq_row_bytes = ggml_row_size(iq_type, K);
    const float* in_data = input.DataAs<float>();
    float* out_data = output->DataAs<float>();
    const size_t gate_row_bytes = ggml_row_size(gate_type, K);
    const size_t up_row_bytes = ggml_row_size(up_type, K);
    const char* gate_data = static_cast<const char*>(gate_weight_ptr);
    const char* up_data = static_cast<const char*>(up_weight_ptr);
    InferenceWorkContext* census_ctx = GetCurrentWorkContext();
    RecordMoEExpertMatmulWeightType(census_ctx, gate_type);
    RecordMoEExpertMatmulWeightType(census_ctx, up_type);
    const bool q4k_repacked_candidate = gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K &&
                                        iq_type == GGML_TYPE_Q8_K && (N % 8) == 0 &&
                                        (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0;
    bool q4k_repacked_reported = false;
    const auto record_q4k_repacked = [&](bool used, const char* reject_reason) {
        if (q4k_repacked_candidate && !q4k_repacked_reported) {
            RecordMoEQ4KRepackedDecision(census_ctx, true, used, reject_reason);
            q4k_repacked_reported = true;
        }
    };

    const size_t total_qbytes = static_cast<size_t>(M) * iq_row_bytes;
    uint8_t* qinput_data = nullptr;
    if (input_cache) {
        const bool cache_hit = input_cache->source == in_data && input_cache->rows == M && input_cache->cols == K &&
                               input_cache->type == iq_type && input_cache->row_bytes == iq_row_bytes &&
                               input_cache->bytes.size() >= total_qbytes;
        if (!cache_hit) {
            input_cache->source = in_data;
            input_cache->rows = M;
            input_cache->cols = K;
            input_cache->type = iq_type;
            input_cache->row_bytes = iq_row_bytes;
            input_cache->bytes.resize(total_qbytes);
            for (int64_t m = 0; m < M; ++m) {
                iq_traits->from_float(in_data + m * K,
                                      input_cache->bytes.data() + static_cast<size_t>(m) * iq_row_bytes, K);
            }
        }
        qinput_data = input_cache->bytes.data();
    } else {
        static thread_local std::vector<uint8_t> qinput_buf;
        if (qinput_buf.size() < total_qbytes) qinput_buf.resize(total_qbytes);
        for (int64_t m = 0; m < M; ++m) {
            iq_traits->from_float(in_data + m * K, qinput_buf.data() + static_cast<size_t>(m) * iq_row_bytes, K);
        }
        qinput_data = qinput_buf.data();
    }

    const bool q4k_prefill_repacked_eligible = CanUseQ4KRepackedMoEGEGLUFastPath() && gate_type == GGML_TYPE_Q4_K &&
                                               up_type == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && M > 1 &&
                                               (N % 8) == 0 && (K % QK_K) == 0;
    // The graph-level Gemma4 native prefill path owns Q4_K GEGLU micro-batches.
    // This older backend helper can crash in ggml_gemm_q4_K_8x8_q8_K on small
    // synthetic M>1 GEGLU fixtures, so keep it out of admission here.
    const bool enable_q4k_repacked_geglu_prefill = false;
    if (enable_q4k_repacked_geglu_prefill && q4k_prefill_repacked_eligible && prefer_q4k_repacked_prefill) {
        auto gate_packed = GetOrCreateQ4KRepackedMoEWeight(gate_weight_ptr, N, K);
        auto up_packed = GetOrCreateQ4KRepackedMoEWeight(up_weight_ptr, N, K);
        if (gate_packed && up_packed &&
            RunQ4KRepackedMoEFusedGEGLU(backend, gate_packed, up_packed, in_data, qinput_data, iq_row_bytes, out_data,
                                        M, N, K, numa_node, allow_parallel)) {
            LogMoEMatmulPath(M >= 4 ? "ggml_q4k_repacked_prefill_gemm_m4_fused_geglu"
                                    : "ggml_q4k_repacked_prefill_fused_geglu",
                             static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(true, nullptr);
            return true;
        }
    }
    if (CanUseMoEQ4KRawBatchedScalar() && gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K &&
        iq_type == GGML_TYPE_Q8_K && M > 1 && K % ggml_blck_size(GGML_TYPE_Q4_K) == 0) {
        if (RunMoEQ4KRawBatchedFusedGEGLU(backend, gate_weight_ptr, up_weight_ptr, qinput_data, iq_row_bytes, out_data,
                                          M, N, K, numa_node, allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_raw_batched_fused_geglu", static_cast<int>(M), static_cast<int>(K),
                             static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(false, "raw_batched_used");
            return true;
        }
    }
    if (CanUseQ4KRepackedMoEGEGLUFastPath() && gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K &&
        iq_type == GGML_TYPE_Q8_K && M == 1 && (N % 8) == 0 && (K % QK_K) == 0) {
        auto gate_packed = GetOrCreateQ4KRepackedMoEWeight(gate_weight_ptr, N, K);
        auto up_packed = GetOrCreateQ4KRepackedMoEWeight(up_weight_ptr, N, K);
        if (gate_packed && up_packed &&
            RunQ4KRepackedMoEFusedGEGLU(backend, gate_packed, up_packed, in_data, qinput_data, iq_row_bytes, out_data,
                                        M, N, K, numa_node, allow_parallel)) {
            LogMoEMatmulPath(M > 1 ? (M >= 4 ? "ggml_q4k_repacked_prefill_gemm_m4_fused_geglu"
                                             : "ggml_q4k_repacked_prefill_fused_geglu")
                                   : "ggml_q4k_repacked_fused_geglu",
                             static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(true, nullptr);
            return true;
        }
    }
    const bool use_kquant_rowpair_vec_dot = CanUseQ4KRowPairVecDotFastPath() && gate_type == up_type &&
                                            IsKQuantRowPairGatedProjectionType(gate_type, iq_type) &&
                                            gate_traits_cpu->vec_dot == up_traits_cpu->vec_dot && M == 1 &&
                                            K % ggml_blck_size(gate_type) == 0 && N >= 4;
    if (use_kquant_rowpair_vec_dot) {
        auto& pool = backend->GetThreadPool(numa_node);
        const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
        const void* qi = qinput_data;
        float* out_row = out_data;
        const int64_t pair_count = N / 2;
        const auto compute_pair_range = [&](int pair_start, int pair_end) {
            for (int pair = pair_start; pair < pair_end; ++pair) {
                const int64_t n = static_cast<int64_t>(pair) * 2;
                const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                float gate_sums[32] = {};
                float up_sums[32] = {};
                gate_traits_cpu->vec_dot(static_cast<int>(K), gate_sums, 2, gate_row, gate_row_bytes, qi, 0, 2);
                up_traits_cpu->vec_dot(static_cast<int>(K), up_sums, 2, up_row, up_row_bytes, qi, 0, 2);
                out_row[n] = GeluTanhApprox(gate_sums[0]) * up_sums[0];
                out_row[n + 1] = GeluTanhApprox(gate_sums[1]) * up_sums[1];
            }
        };
        if (n_threads <= 1) {
            compute_pair_range(0, static_cast<int>(pair_count));
        } else {
            pool.ParallelFor(static_cast<int>(pair_count),
                             [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
        }
        if ((N & 1) != 0) {
            const int64_t n = N - 1;
            const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
            const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            gate_traits_cpu->vec_dot(static_cast<int>(K), &gate_sum, 0, gate_row, 0, qi, 0, 1);
            up_traits_cpu->vec_dot(static_cast<int>(K), &up_sum, 0, up_row, 0, qi, 0, 1);
            out_row[n] = GeluTanhApprox(gate_sum) * up_sum;
        }
        const char* path = gate_type == GGML_TYPE_Q5_K   ? "ggml_q5k_rowpair_m1_fused_geglu"
                           : gate_type == GGML_TYPE_Q5_1 ? "ggml_q5_1_rowpair_m1_fused_geglu"
                                                         : "ggml_q4k_rowpair_m1_fused_geglu";
        LogMoEMatmulPath(path, static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
        record_q4k_repacked(false, "rowpair_used");
        return true;
    }
    return false;
}

bool TryRunGgmlQuantizedFusedSwiGLUProjection(CpuBackend* backend, const void* gate_weight_ptr, int gate_ggml_type_id,
                                              const void* up_weight_ptr, int up_ggml_type_id, const Tensor& input,
                                              Tensor* output, int64_t N, int64_t K, int numa_node,
                                              bool allow_parallel = true,
                                              QuantizedProjectionInputCache* input_cache = nullptr,
                                              bool prefer_q4k_repacked_prefill = false) {
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !output || !input.IsValid() || !output->IsValid()) {
        return false;
    }
    if (input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

    const ggml_type gate_type = static_cast<ggml_type>(gate_ggml_type_id);
    const ggml_type up_type = static_cast<ggml_type>(up_ggml_type_id);
    if (gate_type == GGML_TYPE_F32 || up_type == GGML_TYPE_F32 || !ggml_is_quantized(gate_type) ||
        !ggml_is_quantized(up_type)) {
        return false;
    }

    const auto* gate_traits = ggml_get_type_traits(gate_type);
    const auto* up_traits = ggml_get_type_traits(up_type);
    const auto* gate_traits_cpu = ggml_get_type_traits_cpu(gate_type);
    const auto* up_traits_cpu = ggml_get_type_traits_cpu(up_type);
    if (!gate_traits || !up_traits || !gate_traits_cpu || !up_traits_cpu || !gate_traits_cpu->vec_dot ||
        !up_traits_cpu->vec_dot) {
        return false;
    }
    if (gate_traits_cpu->vec_dot_type != up_traits_cpu->vec_dot_type) {
        return false;
    }

    const int64_t M = input.shape[0];
    if (M <= 0 || M > kMoEQuantizedProjectionMaxBatch || K != input.shape[1] || N != output->shape[1]) {
        return false;
    }

    const ggml_type iq_type = gate_traits_cpu->vec_dot_type;
    const auto* iq_traits = ggml_get_type_traits_cpu(iq_type);
    if (!iq_traits || !iq_traits->from_float) {
        return false;
    }

    const size_t gate_row_bytes = ggml_row_size(gate_type, K);
    const size_t up_row_bytes = ggml_row_size(up_type, K);
    const size_t iq_row_bytes = ggml_row_size(iq_type, K);
    const float* in_data = input.DataAs<float>();
    float* out_data = output->DataAs<float>();
    const char* gate_data = static_cast<const char*>(gate_weight_ptr);
    const char* up_data = static_cast<const char*>(up_weight_ptr);
    InferenceWorkContext* census_ctx = GetCurrentWorkContext();
    RecordMoEExpertMatmulWeightType(census_ctx, gate_type);
    RecordMoEExpertMatmulWeightType(census_ctx, up_type);
    const bool q4k_repacked_candidate = gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K &&
                                        iq_type == GGML_TYPE_Q8_K && (N % 8) == 0 &&
                                        (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0;
    bool q4k_repacked_reported = false;
    const auto record_q4k_repacked = [&](bool used, const char* reject_reason) {
        if (q4k_repacked_candidate && !q4k_repacked_reported) {
            RecordMoEQ4KRepackedDecision(census_ctx, true, used, reject_reason);
            q4k_repacked_reported = true;
        }
    };

    const size_t total_qbytes = static_cast<size_t>(M) * iq_row_bytes;
    uint8_t* qinput_data = nullptr;
    if (input_cache) {
        const bool cache_hit = input_cache->source == in_data && input_cache->rows == M && input_cache->cols == K &&
                               input_cache->type == iq_type && input_cache->row_bytes == iq_row_bytes &&
                               input_cache->bytes.size() >= total_qbytes;
        if (!cache_hit) {
            input_cache->source = in_data;
            input_cache->rows = M;
            input_cache->cols = K;
            input_cache->type = iq_type;
            input_cache->row_bytes = iq_row_bytes;
            input_cache->bytes.resize(total_qbytes);
            for (int64_t m = 0; m < M; ++m) {
                iq_traits->from_float(in_data + m * K,
                                      input_cache->bytes.data() + static_cast<size_t>(m) * iq_row_bytes, K);
            }
        }
        qinput_data = input_cache->bytes.data();
    } else {
        static thread_local std::vector<uint8_t> qinput_buf;
        if (qinput_buf.size() < total_qbytes) qinput_buf.resize(total_qbytes);
        for (int64_t m = 0; m < M; ++m) {
            iq_traits->from_float(in_data + m * K, qinput_buf.data() + static_cast<size_t>(m) * iq_row_bytes, K);
        }
        qinput_data = qinput_buf.data();
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const bool use_q4k_rowpair_m2_vec_dot = CanUseKQuantRowPairVecDotFastPath() && gate_type == GGML_TYPE_Q4_K &&
                                            up_type == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K &&
                                            gate_traits_cpu->vec_dot == up_traits_cpu->vec_dot && M >= 2 &&
                                            K % ggml_blck_size(GGML_TYPE_Q4_K) == 0 && N >= 2;
    const bool q4k_prefill_repacked_eligible = CanUseQ4KRepackedMoEPrefillFastPath() && gate_type == GGML_TYPE_Q4_K &&
                                               up_type == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && M > 1 &&
                                               (N % 8) == 0 && (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0;
    if (q4k_prefill_repacked_eligible && prefer_q4k_repacked_prefill) {
        auto gate_packed = GetOrCreateQ4KRepackedMoEWeight(gate_weight_ptr, N, K);
        auto up_packed = GetOrCreateQ4KRepackedMoEWeight(up_weight_ptr, N, K);
        if (gate_packed && up_packed) {
            if (RunQ4KRepackedMoEFusedSwiGLUM4(backend, gate_packed, up_packed, in_data, qinput_data, iq_row_bytes,
                                               out_data, M, N, K, numa_node, allow_parallel)) {
                LogMoEMatmulPath(M >= 4 ? "ggml_q4k_repacked_prefill_gemm_m4_tile_fused_swiglu"
                                        : "ggml_q4k_repacked_prefill_tile_fused_swiglu",
                                 static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
                record_q4k_repacked(true, nullptr);
                return true;
            }
        }
    }
    if (CanUseMoEQ4KRawBatchedScalar() && gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K &&
        iq_type == GGML_TYPE_Q8_K && M > 1 && K % ggml_blck_size(GGML_TYPE_Q4_K) == 0) {
        if (RunMoEQ4KRawBatchedFusedSwiGLUImpl(backend, gate_weight_ptr, up_weight_ptr, qinput_data, iq_row_bytes,
                                               out_data, M, N, K, numa_node, allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_raw_batched_fused_swiglu", static_cast<int>(M), static_cast<int>(K),
                             static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(false, "raw_batched_used");
            return true;
        }
    }
    if (q4k_prefill_repacked_eligible && !prefer_q4k_repacked_prefill) {
        record_q4k_repacked(false, "prefill_repack_not_preferred");
    }
    if (use_q4k_rowpair_m2_vec_dot) {
        const int64_t pair_count = N / 2;
        for (int64_t m = 0; m < M; ++m) {
            float* out_row = out_data + static_cast<size_t>(m) * N;
            const void* qi = qinput_data + static_cast<size_t>(m) * iq_row_bytes;
            const auto compute_pair_range = [&](int pair_start, int pair_end) {
                for (int pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t n = static_cast<int64_t>(pair) * 2;
                    const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                    const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                    float gate_sums[32] = {};
                    float up_sums[32] = {};
                    gate_traits_cpu->vec_dot(static_cast<int>(K), gate_sums, 2, gate_row, gate_row_bytes, qi, 0, 2);
                    up_traits_cpu->vec_dot(static_cast<int>(K), up_sums, 2, up_row, up_row_bytes, qi, 0, 2);
                    out_row[n] = (gate_sums[0] / (1.0f + internal::FastExp(-gate_sums[0]))) * up_sums[0];
                    out_row[n + 1] = (gate_sums[1] / (1.0f + internal::FastExp(-gate_sums[1]))) * up_sums[1];
                }
            };
            if (n_threads <= 1) {
                compute_pair_range(0, static_cast<int>(pair_count));
            } else {
                pool.ParallelFor(static_cast<int>(pair_count),
                                 [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
            }
            if ((N & 1) != 0) {
                const int64_t n = N - 1;
                const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                gate_traits_cpu->vec_dot(static_cast<int>(K), &gate_sum, 0, gate_row, 0, qi, 0, 1);
                up_traits_cpu->vec_dot(static_cast<int>(K), &up_sum, 0, up_row, 0, qi, 0, 1);
                out_row[n] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_sum;
            }
        }
        LogMoEMatmulPath("ggml_q4k_rowpair_m2_fused_swiglu", static_cast<int>(M), static_cast<int>(K),
                         static_cast<int>(N), 0, allow_parallel);
        record_q4k_repacked(false, "rowpair_used");
        return true;
    }
    if (M == 1 && gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K &&
        (N % 8) == 0 && (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0) {
        auto gate_packed = GetOrCreateQ4KRepackedMoEWeight(gate_weight_ptr, N, K);
        auto up_packed = GetOrCreateQ4KRepackedMoEWeight(up_weight_ptr, N, K);
        if (gate_packed && up_packed) {
            if (RunQ4KRepackedMoEFusedSwiGLUM4(backend, gate_packed, up_packed, in_data, qinput_data, iq_row_bytes,
                                               out_data, M, N, K, numa_node, allow_parallel)) {
                LogMoEMatmulPath("ggml_q4k_repacked_tile_fused_swiglu", static_cast<int>(M), static_cast<int>(K),
                                 static_cast<int>(N), 0, allow_parallel);
                record_q4k_repacked(true, nullptr);
                return true;
            }
        }
    }
    const bool use_q4k_rowpair_vec_dot = CanUseQ4KRowPairVecDotFastPath() && gate_type == GGML_TYPE_Q4_K &&
                                         up_type == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K &&
                                         gate_traits_cpu->vec_dot == up_traits_cpu->vec_dot && M == 1 &&
                                         K % ggml_blck_size(GGML_TYPE_Q4_K) == 0 && N >= 4;

    for (int64_t m = 0; m < M; ++m) {
        float* out_row = out_data + m * N;
        const void* qi = qinput_data + static_cast<size_t>(m) * iq_row_bytes;

        if (use_q4k_rowpair_vec_dot) {
            const int64_t pair_count = N / 2;
            const auto compute_pair_range = [&](int pair_start, int pair_end) {
                for (int pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t n = static_cast<int64_t>(pair) * 2;
                    const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                    const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                    float gate_sums[32] = {};
                    float up_sums[32] = {};
                    gate_traits_cpu->vec_dot(static_cast<int>(K), gate_sums, 2, gate_row, gate_row_bytes, qi, 0, 2);
                    up_traits_cpu->vec_dot(static_cast<int>(K), up_sums, 2, up_row, up_row_bytes, qi, 0, 2);
                    out_row[n] = (gate_sums[0] / (1.0f + internal::FastExp(-gate_sums[0]))) * up_sums[0];
                    out_row[n + 1] = (gate_sums[1] / (1.0f + internal::FastExp(-gate_sums[1]))) * up_sums[1];
                }
            };
            if (n_threads <= 1) {
                compute_pair_range(0, static_cast<int>(pair_count));
            } else {
                pool.ParallelFor(static_cast<int>(pair_count),
                                 [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
            }
            if ((N & 1) != 0) {
                const int64_t n = N - 1;
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                gate_traits_cpu->vec_dot(static_cast<int>(K), &gate_sum, 0, gate_row, 0, qi, 0, 1);
                up_traits_cpu->vec_dot(static_cast<int>(K), &up_sum, 0, up_row, 0, qi, 0, 1);
                out_row[n] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_sum;
            }
            continue;
        }

        const auto compute_range = [&](int n_start, int n_end) {
            for (int n = n_start; n < n_end; ++n) {
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                gate_traits_cpu->vec_dot(static_cast<int>(K), &gate_sum, 0, gate_row, 0, qi, 0, 1);
                up_traits_cpu->vec_dot(static_cast<int>(K), &up_sum, 0, up_row, 0, qi, 0, 1);
                out_row[n] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_sum;
            }
        };

        if (n_threads <= 1) {
            compute_range(0, static_cast<int>(N));
        } else {
            pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) { compute_range(n_start, n_end); });
        }
    }
    if (use_q4k_rowpair_vec_dot) {
        LogMoEMatmulPath("ggml_q4k_rowpair_m1_fused_swiglu", static_cast<int>(M), static_cast<int>(K),
                         static_cast<int>(N), 0, allow_parallel);
    }
    record_q4k_repacked(false, use_q4k_rowpair_vec_dot ? "rowpair_used" : "native_vecdot");
    return true;
}
