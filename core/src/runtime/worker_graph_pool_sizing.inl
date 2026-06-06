// Runtime graph-pool sizing and dry-run reservation helpers.
bool ParseBoolEnvDefault(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }
    return !(std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0 || std::strcmp(raw, "FALSE") == 0 ||
             std::strcmp(raw, "off") == 0 || std::strcmp(raw, "OFF") == 0);
}

bool IsFlexibleGraphPoolSizingEnabled(const TransformerModel* model) {
    if (!model) {
        return false;
    }
    return ParseBoolEnvDefault("DENSECORE_FLEXIBLE_GRAPH_POOL", true);
}

size_t ReadAvailableMemoryBytesForRuntimePools() {
#if defined(__linux__)
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (!file) {
        return 0;
    }
    char line[256] = {};
    unsigned long long kb = 0;
    while (std::fgets(line, sizeof(line), file)) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            std::fclose(file);
            return static_cast<size_t>(kb) * 1024ULL;
        }
    }
    std::fclose(file);
#endif
    return 0;
}

void AccumulateDryRunTensorBytes(const ggml_tensor* tensor, std::unordered_set<const ggml_tensor*>& seen,
                                 size_t& data_bytes) {
    if (!tensor || !seen.insert(tensor).second) {
        return;
    }
    if (tensor->view_src) {
        AccumulateDryRunTensorBytes(tensor->view_src, seen, data_bytes);
    } else if (!tensor->data) {
        data_bytes += ggml_nbytes_pad(tensor);
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        AccumulateDryRunTensorBytes(tensor->src[i], seen, data_bytes);
    }
}

struct FlexibleGraphPoolSizing {
    bool ok = false;
    size_t required_bytes = 0;
    size_t reserved_bytes = 0;
    size_t available_bytes = 0;
    size_t reservation_payload_bytes = 0;
    size_t reservation_slack_bytes = 0;
    size_t dry_context_bytes = 0;
    size_t dry_metadata_bytes = 0;
    size_t graph_tensor_bytes = 0;
    size_t margin_bytes = 0;
    int graph_nodes = 0;
};

struct RuntimeGraphPoolReservation {
    size_t total_bytes = 0;
    size_t payload_bytes = 0;
    size_t slack_bytes = 0;
};

size_t ApplyFlexibleGraphPoolGrowthReserve(size_t required_bytes,
                                           const EngineState::GraphContextEstimate& graph_estimate) {
    if (required_bytes == 0 || graph_estimate.effective_query_len <= 1) {
        return required_bytes;
    }
    constexpr size_t MB = 1024ULL * 1024ULL;
    const size_t measured_margin = std::max(required_bytes / 8, graph_estimate.long_context_safety_pad_bytes);
    const size_t reserve_bytes = required_bytes + std::max<size_t>(measured_margin, 128ULL * MB);
    return AlignUpBytes(reserve_bytes, 512ULL * MB);
}

RuntimeGraphPoolReservation ClampRuntimeGraphPoolReservation(size_t requested_bytes, size_t available_bytes) {
    if (requested_bytes == 0 || available_bytes == 0) {
        return {requested_bytes, requested_bytes, 0};
    }
    constexpr size_t MB = 1024ULL * 1024ULL;
    const size_t max_total_bytes =
        available_bytes > 256ULL * MB ? available_bytes - 256ULL * MB : (available_bytes * 3) / 4;
    const size_t runtime_reserve_bytes = std::clamp<size_t>(available_bytes / 12, 512ULL * MB, 8192ULL * MB);
    const size_t allocator_slack_bytes = std::max<size_t>(available_bytes / 64, 128ULL * MB);
    const size_t reserved_bytes = runtime_reserve_bytes + allocator_slack_bytes;
    const size_t usable_bytes =
        available_bytes > reserved_bytes ? available_bytes - reserved_bytes : available_bytes / 2;
    const size_t payload_bytes = std::min(requested_bytes, std::max<size_t>(usable_bytes, 512ULL * MB));
    const size_t context_object_slack_bytes = std::clamp<size_t>(payload_bytes / 6, 128ULL * MB, 2048ULL * MB);
    // Dry-run sizing can still undercount ggml object-pool pressure when the
    // live request shape is not byte-identical to the calibration shape. Scale
    // that reserve from the measured payload instead of baking in a VM-specific
    // RAM number; larger hosts and larger graphs naturally get more cushion.
    const size_t object_pool_variance_bytes = std::max(payload_bytes / 16, requested_bytes / 32);
    size_t total_bytes =
        AlignUpBytes(payload_bytes + context_object_slack_bytes + object_pool_variance_bytes, 64ULL * MB);
    size_t capped_payload_bytes = payload_bytes;
    size_t capped_slack_bytes = context_object_slack_bytes;
    if (max_total_bytes > 0 && total_bytes > max_total_bytes) {
        total_bytes = std::max(512ULL * MB, (max_total_bytes / (64ULL * MB)) * (64ULL * MB));
        capped_slack_bytes = std::min(context_object_slack_bytes, std::max<size_t>(128ULL * MB, total_bytes / 16));
        capped_payload_bytes = total_bytes > capped_slack_bytes ? total_bytes - capped_slack_bytes : total_bytes;
        capped_payload_bytes = std::min(capped_payload_bytes, requested_bytes);
        capped_slack_bytes = total_bytes > capped_payload_bytes ? total_bytes - capped_payload_bytes : 0;
    }
    return {total_bytes, capped_payload_bytes, capped_slack_bytes};
}

FlexibleGraphPoolSizing MeasureFlexibleGraphPoolSize(TransformerModel* model, PagedKVCache* cache,
                                                     const BatchSpec& batch, bool embedding_mode,
                                                     const EngineState::GraphContextEstimate& fallback_estimate) {
    FlexibleGraphPoolSizing result{};
    if (!IsFlexibleGraphPoolSizingEnabled(model)) {
        return result;
    }

    constexpr size_t MB = 1024ULL * 1024ULL;
    const size_t fallback_mb = std::max<size_t>(512, fallback_estimate.total_bytes / MB);
    const size_t available_bytes = ReadAvailableMemoryBytesForRuntimePools();
    const size_t available_mb = available_bytes / MB;
    const size_t auto_dry_cap_mb =
        available_mb > 0 ? std::max<size_t>(512, (available_mb * 3) / 4) : std::max<size_t>(fallback_mb, 1024);
    const size_t dry_run_headroom_mb = fallback_estimate.effective_query_len > 1
                                           ? std::max<size_t>(512, fallback_mb / 8)
                                           : std::max<size_t>(128, fallback_mb / 16);
    const size_t default_dry_mb =
        std::min<size_t>(std::max<size_t>(fallback_mb + dry_run_headroom_mb, 1024), auto_dry_cap_mb);
    const size_t dry_mb = ParseSizeEnvMb("DENSECORE_GRAPH_DRY_RUN_CTX_MB", default_dry_mb, 512, auto_dry_cap_mb);
    const size_t dry_context_bytes = dry_mb * MB;

    void* dry_buffer = nullptr;
#if defined(_WIN32)
    dry_buffer = _aligned_malloc(dry_context_bytes, 64);
#else
    if (posix_memalign(&dry_buffer, 64, dry_context_bytes) != 0) {
        dry_buffer = nullptr;
    }
#endif
    if (!dry_buffer) {
        return result;
    }

    ggml_context* dry_ctx = nullptr;
    try {
        ggml_init_params params{
            .mem_size = dry_context_bytes,
            .mem_buffer = dry_buffer,
            .no_alloc = true,
        };
        dry_ctx = ggml_init(params);
        if (!dry_ctx) {
            throw densecore::OutOfMemoryException("flexible graph pool dry-run ggml_init failed");
        }
        ggml_cgraph* dry_graph = ggml_new_graph_custom(dry_ctx, 32768, false);
        ggml_tensor* dry_embd = nullptr;
        ggml_tensor* dry_pos = nullptr;
        auto dry_work_ctx = std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)>(
            CreateInferenceWorkContext(), DestroyInferenceWorkContext);
        const InferenceExecutionPhase dry_phase = batch.tokens.size() > static_cast<size_t>(batch.num_seqs)
                                                      ? InferenceExecutionPhase::Prefill
                                                      : InferenceExecutionPhase::Decode;
        ScopedBatchWorkContext dry_scope(dry_work_ctx.get(), &batch, dry_phase,
                                         /*reset_context=*/true, ModelVariant::UNKNOWN,
                                         /*graph_build_no_alloc=*/true);
        ggml_tensor* dry_output =
            BuildTransformerGraph(model, cache, dry_ctx, batch, embedding_mode, dry_graph, &dry_embd, &dry_pos);
        if (!dry_graph || !dry_output || !dry_embd || !dry_pos) {
            throw densecore::GraphBuildException("flexible graph pool dry-run graph build returned incomplete graph");
        }

        std::unordered_set<const ggml_tensor*> seen;
        size_t data_bytes = 0;
        const int n_nodes = ggml_graph_n_nodes(dry_graph);
        for (int i = 0; i < n_nodes; ++i) {
            AccumulateDryRunTensorBytes(ggml_graph_node(dry_graph, i), seen, data_bytes);
        }
        AccumulateDryRunTensorBytes(dry_embd, seen, data_bytes);
        AccumulateDryRunTensorBytes(dry_pos, seen, data_bytes);
        AccumulateDryRunTensorBytes(dry_output, seen, data_bytes);

        const size_t metadata_bytes = ggml_used_mem(dry_ctx);
        const size_t measured_bytes = metadata_bytes + data_bytes;
        const size_t percent_margin = measured_bytes / 8;
        const size_t min_margin = model->arch_flags.is_gemma4 ? 512ULL * MB : 128ULL * MB;
        const size_t margin_bytes = std::max(percent_margin, min_margin);
        const RuntimeGraphPoolReservation required_reservation =
            ClampRuntimeGraphPoolReservation(AlignUpBytes(measured_bytes + margin_bytes, 64ULL * MB), available_bytes);
        result.ok = true;
        result.required_bytes = required_reservation.total_bytes;
        const RuntimeGraphPoolReservation growth_reservation = ClampRuntimeGraphPoolReservation(
            ApplyFlexibleGraphPoolGrowthReserve(result.required_bytes, fallback_estimate), available_bytes);
        result.reserved_bytes = growth_reservation.total_bytes;
        result.available_bytes = available_bytes;
        result.reservation_payload_bytes = growth_reservation.payload_bytes;
        result.reservation_slack_bytes = growth_reservation.slack_bytes;
        result.dry_context_bytes = dry_context_bytes;
        result.dry_metadata_bytes = metadata_bytes;
        result.graph_tensor_bytes = data_bytes;
        result.margin_bytes = margin_bytes;
        result.graph_nodes = n_nodes;
    } catch (const std::exception& e) {
        std::cerr << "[DenseCore] FlexibleGraphPool dry-run skipped: " << e.what() << std::endl;
    }

    if (dry_ctx) {
        ggml_free(dry_ctx);
    }
#if defined(_WIN32)
    _aligned_free(dry_buffer);
#else
    std::free(dry_buffer);
#endif
    return result;
}
