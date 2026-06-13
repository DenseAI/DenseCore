bool TryRunPackedInt4ProjectionDirect(CpuBackend* backend, const CpuBackend::ExpertPackedInt4Weight& binding,
                                      const Tensor& input, Tensor* output, int numa_node, bool allow_parallel = true) {
    if (!backend || !output || !binding.IsValid() || !input.IsValid() || !output->IsValid() ||
        input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

    // All platforms: Highway INT4 GEMV kernels with runtime ISA dispatch
    // (NEON/SVE2 on ARM, AVX2/AVX-512 on x86)
    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[1]);
    const int N = static_cast<int>(output->shape[1]);
    if (M <= 0 || K != binding.K || N != binding.N) {
        return false;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    if (!IsArmPackedInt4HwyProjectionSupported()) {
        (void)backend;
        (void)numa_node;
        (void)allow_parallel;
        return false;
    }
#endif

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    if (M == 1) {
        if (n_threads <= 1) {
            densecore::hwy_kernels::GemvInt4_Hwy(output_data, input_data, binding.packed_weights, binding.scales,
                                                 binding.zeros, K, N, binding.group_size, 0, N);
        } else {
            pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
                densecore::hwy_kernels::GemvInt4_Hwy(output_data, input_data, binding.packed_weights, binding.scales,
                                                     binding.zeros, K, N, binding.group_size, n_start, n_end);
            });
        }
        return true;
    }

    if (M <= 4) {
        const size_t input_stride_bytes = static_cast<size_t>(K) * sizeof(float);
        const bool should_parallelize = n_threads > 1 && N >= 128;
        if (should_parallelize) {
            pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
                densecore::hwy_kernels::GemmInt4Batched_Hwy(output_data, input_data, binding.packed_weights,
                                                            binding.scales, binding.zeros, M, K, N, binding.group_size,
                                                            0, M, n_start, n_end, input_stride_bytes);
            });
        } else {
            densecore::hwy_kernels::GemmInt4Batched_Hwy(output_data, input_data, binding.packed_weights, binding.scales,
                                                        binding.zeros, M, K, N, binding.group_size, 0, M, 0, N,
                                                        input_stride_bytes);
        }
        return true;
    }

    return false;
}

bool TryRunPackedInt4FusedGateUpProjectionDirect(CpuBackend* backend,
                                                 const CpuBackend::ExpertPackedInt4Weight& gate_binding,
                                                 const CpuBackend::ExpertPackedInt4Weight& up_binding,
                                                 const Tensor& input, Tensor* output, int numa_node,
                                                 bool use_gelu_activation, bool allow_parallel = true) {
    if (!backend || !output || !gate_binding.IsValid() || !up_binding.IsValid() || !input.IsValid() ||
        !output->IsValid() || input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 ||
        output->ndim != 2) {
        return false;
    }

    // All platforms: Highway fused INT4 gate/up kernels with runtime ISA dispatch.
    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[1]);
    const int N = static_cast<int>(output->shape[1]);
    if (M <= 0 || M > 4 || gate_binding.K != K || up_binding.K != K || gate_binding.N != N || up_binding.N != N ||
        gate_binding.group_size != up_binding.group_size) {
        return false;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    if (!IsArmPackedInt4HwyProjectionSupported()) {
        (void)backend;
        (void)numa_node;
        (void)allow_parallel;
        return false;
    }
#endif

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    const size_t input_stride_bytes = static_cast<size_t>(K) * sizeof(float);
    auto compute_range = [&](int n_start, int n_end) {
        if (use_gelu_activation) {
            densecore::hwy_kernels::GemmInt4DualFusedGeluBatched_Hwy(
                output_data, input_data, gate_binding.packed_weights, gate_binding.scales, gate_binding.zeros,
                up_binding.packed_weights, up_binding.scales, up_binding.zeros, M, K, N, gate_binding.group_size, 0, M,
                n_start, n_end, input_stride_bytes);
        } else {
            densecore::hwy_kernels::GemmInt4DualFusedSiluBatched_Hwy(
                output_data, input_data, gate_binding.packed_weights, gate_binding.scales, gate_binding.zeros,
                up_binding.packed_weights, up_binding.scales, up_binding.zeros, M, K, N, gate_binding.group_size, 0, M,
                n_start, n_end, input_stride_bytes);
        }
    };

    if (n_threads <= 1 || N < 128) {
        compute_range(0, N);
    } else {
        pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) { compute_range(n_start, n_end); });
    }
    return true;
}

struct MoEFusedGateUpRequest {
    CpuBackend* backend = nullptr;
    int numa_node = 0;
    const Tensor* input = nullptr;
    const Tensor* dense_gate_weight = nullptr;
    Tensor* hidden = nullptr;
    const CpuBackend::ExpertWeights* expert = nullptr;
    QuantizedProjectionInputCache* input_projection_cache = nullptr;
    InferenceWorkContext* gemma4_quant_prefill_ctx = nullptr;
    int64_t intermediate_dim = 0;
    int64_t hidden_dim = 0;
    bool safe_reference_mode = false;
    bool ggml_quantized_vecdot_safe = false;
    bool gemma4_quant_prefill_batch_safe = false;
    bool force_gemma4_quant_prefill_fast_path = false;
    bool record_gemma4_quant_prefill_batch = false;
    bool prefer_q4k_repacked_prefill = false;
    bool allow_q4k_repacked_decode_fused_swiglu = true;
    bool enable_inner_parallel = false;
};

struct MoEFusedGateUpPlan {
    bool try_packed_int4 = false;
    bool try_ggml_quantized_swiglu = false;
    bool try_ggml_quantized_geglu = false;
    bool requires_unfused_gate_projection = false;
    bool force_quant_prefill_failure = false;
};

MoEFusedGateUpPlan ResolveMoEFusedGateUpPlan(const MoEFusedGateUpRequest& req) {
    MoEFusedGateUpPlan plan;
    if (!req.expert) {
        return plan;
    }
    plan.try_packed_int4 = !req.safe_reference_mode;
    plan.try_ggml_quantized_swiglu = !req.safe_reference_mode && req.ggml_quantized_vecdot_safe &&
                                     !req.expert->use_gelu_activation && req.expert->w1.ptr && req.expert->w3.ptr;
    plan.try_ggml_quantized_geglu = !req.safe_reference_mode && req.ggml_quantized_vecdot_safe &&
                                    req.expert->use_gelu_activation && req.expert->w1.ptr && req.expert->w3.ptr;
    plan.requires_unfused_gate_projection =
        (req.dense_gate_weight && req.dense_gate_weight->IsValid()) || req.expert->w3_int4.IsValid() ||
        req.expert->w3.ptr;
    plan.force_quant_prefill_failure =
        req.force_gemma4_quant_prefill_fast_path && req.expert->w1.ptr && req.expert->w3.ptr;
    return plan;
}

bool EmitMoEFusedGateUpFromPlan(const MoEFusedGateUpRequest& req, const MoEFusedGateUpPlan& plan) {
    if (!req.backend || !req.input || !req.hidden || !req.expert) {
        return false;
    }

    if (plan.try_packed_int4 &&
        TryRunPackedInt4FusedGateUpProjectionDirect(req.backend, req.expert->w1_int4, req.expert->w3_int4, *req.input,
                                                    req.hidden, req.numa_node, req.expert->use_gelu_activation,
                                                    req.enable_inner_parallel)) {
        if (req.expert->use_gelu_activation && req.gemma4_quant_prefill_ctx) {
            RecordGemma4NativeFusedGateUpUsed(req.gemma4_quant_prefill_ctx);
        }
        GetMoEInt4PathHistogram().fused_swiglu_hwy.fetch_add(1, std::memory_order_relaxed);
        LogMoEMatmulPath(req.expert->use_gelu_activation ? "fused_geglu_hwy" : "fused_swiglu_hwy",
                         static_cast<int>(req.input->shape[0]), static_cast<int>(req.input->shape[1]),
                         static_cast<int>(req.hidden->shape[1]), req.expert->w1_int4.group_size,
                         req.enable_inner_parallel);
        return true;
    }

    if (plan.try_ggml_quantized_swiglu &&
        TryRunGgmlQuantizedFusedSwiGLUProjection(
            req.backend, req.expert->w1.ptr, req.expert->w1_type, req.expert->w3.ptr, req.expert->w3_type,
            *req.input, req.hidden, req.intermediate_dim, req.hidden_dim, req.numa_node, req.enable_inner_parallel,
            req.input_projection_cache, req.prefer_q4k_repacked_prefill,
            req.allow_q4k_repacked_decode_fused_swiglu)) {
        LogMoEMatmulPath("ggml_quantized_fused_swiglu", static_cast<int>(req.input->shape[0]),
                         static_cast<int>(req.input->shape[1]), static_cast<int>(req.hidden->shape[1]), 0,
                         req.enable_inner_parallel);
        if (req.record_gemma4_quant_prefill_batch && req.gemma4_quant_prefill_batch_safe) {
            RecordGemma4MoEPrefillQuantBatchDecision(req.gemma4_quant_prefill_ctx, false, true, nullptr, true, false);
        }
        return true;
    }

    if (plan.try_ggml_quantized_geglu &&
        TryRunGgmlQuantizedFusedGEGLUProjection(
            req.backend, req.expert->w1.ptr, req.expert->w1_type, req.expert->w3.ptr, req.expert->w3_type,
            *req.input, req.hidden, req.intermediate_dim, req.hidden_dim, req.numa_node, req.enable_inner_parallel,
            req.input_projection_cache, req.prefer_q4k_repacked_prefill)) {
        LogMoEMatmulPath("ggml_quantized_fused_geglu", static_cast<int>(req.input->shape[0]),
                         static_cast<int>(req.input->shape[1]), static_cast<int>(req.hidden->shape[1]), 0,
                         req.enable_inner_parallel);
        if (req.record_gemma4_quant_prefill_batch && req.gemma4_quant_prefill_batch_safe) {
            RecordGemma4MoEPrefillQuantBatchDecision(req.gemma4_quant_prefill_ctx, false, true, nullptr, true, false);
        }
        return true;
    }

    return false;
}
