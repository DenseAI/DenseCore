// Worker-local observability: request lifecycle traces, determinism hashes,
// tensor/graph dumps, and hybrid SSM snapshot logs. Included once inside
// worker.cpp's anonymous namespace, after IsDeterminismBoundaryDebugEnabled.
// Uses worker.cpp's request/model, GGML, runtime-env, and standard-library types;
// owns no scheduling, request lifetime, or graph-cache mutation policy.

bool IsRequestLifecycleTraceEnabled() {
    static const bool enabled = densecore::env::ParseDiagnosticEnv("DENSECORE_DEBUG_REQUEST_LIFECYCLE");
    return enabled;
}

void LogRequestLifecyclePhase(const char* phase, const Request* req, const TransformerModel* model, int seq_id = -1,
                              int batch_seqs = 0, int batch_tokens = 0, int active_threads = 0,
                              const char* detail = nullptr) {
    if (!IsRequestLifecycleTraceEnabled()) {
        return;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    std::cerr << "[RequestLifecycle] phase=" << (phase ? phase : "unknown") << " request_id=" << (req ? req->id : -1)
              << " seq_id=" << seq_id << " variant=" << densecore::models::ModelVariantName(descriptor.variant)
              << " is_prefill=" << (req && req->is_prefill ? 1 : 0)
              << " prompt_tokens=" << (req ? req->tokens.size() : 0) << " n_past=" << (req ? req->n_past : 0)
              << " generated=" << (req ? req->generated_count : 0) << " max_tokens=" << (req ? req->max_tokens : 0)
              << " batch_seqs=" << batch_seqs << " batch_tokens=" << batch_tokens
              << " active_threads=" << active_threads;
    if (detail && detail[0] != '\0') {
        std::cerr << " detail=" << detail;
    }
    std::cerr << std::endl;
}

uint64_t Fnv1aInit() {
    return 1469598103934665603ull;
}

void Fnv1aMixU64(uint64_t* hash, uint64_t value) {
    if (!hash) {
        return;
    }
    *hash ^= value;
    *hash *= 1099511628211ull;
}

uint64_t HashIntVectorSummary(const std::vector<int>& values, size_t max_items = 32) {
    uint64_t hash = Fnv1aInit();
    Fnv1aMixU64(&hash, static_cast<uint64_t>(values.size()));
    const size_t limit = std::min(max_items, values.size());
    for (size_t i = 0; i < limit; ++i) {
        Fnv1aMixU64(&hash, static_cast<uint64_t>(static_cast<uint32_t>(values[i])));
    }
    if (values.size() > limit) {
        for (size_t i = values.size() - std::min<size_t>(4, values.size()); i < values.size(); ++i) {
            Fnv1aMixU64(&hash, static_cast<uint64_t>(static_cast<uint32_t>(values[i])));
        }
    }
    return hash;
}

uint64_t HashByteSpanSummary(const void* data, size_t bytes) {
    if (!data || bytes == 0) {
        return 0;
    }
    const auto* ptr = reinterpret_cast<const uint8_t*>(data);
    uint64_t hash = Fnv1aInit();
    Fnv1aMixU64(&hash, static_cast<uint64_t>(bytes));
    const size_t window = std::min<size_t>(bytes, 4096);
    for (size_t i = 0; i < window; ++i) {
        Fnv1aMixU64(&hash, ptr[i]);
    }
    if (bytes > window) {
        const size_t mid = bytes / 2;
        for (size_t i = 0; i < std::min<size_t>(256, bytes - mid); ++i) {
            Fnv1aMixU64(&hash, ptr[mid + i]);
        }
        const size_t tail_start = bytes - std::min<size_t>(256, bytes);
        for (size_t i = tail_start; i < bytes; ++i) {
            Fnv1aMixU64(&hash, ptr[i]);
        }
    }
    return hash;
}

uint64_t HashHybridSSMStateSummary(const std::vector<TransformerModel::SSMSequenceRuntimeState>& states) {
    uint64_t hash = Fnv1aInit();
    Fnv1aMixU64(&hash, static_cast<uint64_t>(states.size()));
    for (const auto& state : states) {
        Fnv1aMixU64(&hash, static_cast<uint64_t>(state.conv_state.size()));
        Fnv1aMixU64(&hash, static_cast<uint64_t>(state.ssm_state.size()));
        if (!state.conv_state.empty()) {
            const size_t mid = state.conv_state.size() / 2;
            uint32_t bits = 0;
            std::memcpy(&bits, &state.conv_state[0], sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
            std::memcpy(&bits, &state.conv_state[mid], sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
            std::memcpy(&bits, &state.conv_state.back(), sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
        }
        if (!state.ssm_state.empty()) {
            const size_t mid = state.ssm_state.size() / 2;
            uint32_t bits = 0;
            std::memcpy(&bits, &state.ssm_state[0], sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
            std::memcpy(&bits, &state.ssm_state[mid], sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
            std::memcpy(&bits, &state.ssm_state.back(), sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
        }
    }
    return hash;
}

void LogPrefillStage(const char* stage, const Request* req, const TransformerModel* model,
                     const InferenceContext* inference_ctx, const void* input_buffer, size_t input_buffer_bytes,
                     const struct ggml_tensor* output, bool is_prefill_batch, int active_threads) {
    if (!IsDeterminismBoundaryDebugEnabled() || !req || !is_prefill_batch) {
        return;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    const uint64_t token_hash = HashIntVectorSummary(req->tokens);
    const uint64_t block_hash = HashIntVectorSummary(req->block_table);
    const uint64_t ssm_hash = HashHybridSSMStateSummary(req->ssm_runtime_states);
    const uint64_t input_hash = HashByteSpanSummary(input_buffer, input_buffer_bytes);
    const uint64_t graph_hash =
        inference_ctx ? HashByteSpanSummary(inference_ctx->compute_buffer, inference_ctx->compute_buffer_size) : 0;
    uint64_t output_hash = 0;
    size_t output_bytes = 0;
    if (output && output->data) {
        output_bytes = ggml_nbytes(output);
        output_hash = HashByteSpanSummary(output->data, output_bytes);
    }
    std::cerr << "[PrefillStage] stage=" << (stage ? stage : "<unknown>") << " req=" << req->id
              << " variant=" << densecore::models::ModelVariantName(descriptor.variant) << " threads=" << active_threads
              << " prompt_tokens=" << req->tokens.size() << " n_past=" << req->n_past << " token_hash=0x" << std::hex
              << token_hash << " block_hash=0x" << block_hash << " ssm_hash=0x" << ssm_hash << " input_hash=0x"
              << input_hash << " graph_hash=0x" << graph_hash << " output_hash=0x" << output_hash << std::dec
              << " input_bytes=" << input_buffer_bytes << " output_bytes=" << output_bytes << std::endl;
}

void LogDeterminismBoundary(const char* stage, const Request* req, const TransformerModel* model,
                            bool prefix_cache_allowed, bool prefix_cache_hit, bool hybrid_restore_attempted,
                            bool hybrid_restore_applied, bool chunked_prefill, bool decode_cache_active,
                            bool decode_cache_reused, bool prefill_cache_active) {
    if (!IsDeterminismBoundaryDebugEnabled() || !req) {
        return;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    const uint64_t token_hash = HashIntVectorSummary(req->tokens);
    const uint64_t block_hash = HashIntVectorSummary(req->block_table);
    const uint64_t ssm_hash = HashHybridSSMStateSummary(req->ssm_runtime_states);
    std::cerr << "[DeterminismBoundary] stage=" << (stage ? stage : "<unknown>") << " req=" << req->id
              << " variant=" << densecore::models::ModelVariantName(descriptor.variant)
              << " prompt_tokens=" << req->tokens.size() << " n_past=" << req->n_past
              << " generated=" << req->generated_count << " is_prefill=" << (req->is_prefill ? 1 : 0)
              << " prefix_cache_allowed=" << (prefix_cache_allowed ? 1 : 0)
              << " prefix_cache_hit=" << (prefix_cache_hit ? 1 : 0)
              << " hybrid_restore_attempted=" << (hybrid_restore_attempted ? 1 : 0)
              << " hybrid_restore_applied=" << (hybrid_restore_applied ? 1 : 0)
              << " chunked_prefill=" << (chunked_prefill ? 1 : 0)
              << " decode_cache_active=" << (decode_cache_active ? 1 : 0)
              << " decode_cache_reused=" << (decode_cache_reused ? 1 : 0)
              << " prefill_cache_active=" << (prefill_cache_active ? 1 : 0) << " token_hash=0x" << std::hex
              << token_hash << " block_hash=0x" << block_hash << " ssm_hash=0x" << ssm_hash << std::dec << std::endl;
}

bool IsValidGgmlType(enum ggml_type type) {
    const int value = static_cast<int>(type);
    return value >= 0 && value < static_cast<int>(GGML_TYPE_COUNT);
}

bool IsValidGgmlOp(enum ggml_op op) {
    const int value = static_cast<int>(op);
    return value >= 0 && value < static_cast<int>(GGML_OP_COUNT);
}

const char* SafeGgmlTypeName(enum ggml_type type) {
    return IsValidGgmlType(type) ? ggml_type_name(type) : "<invalid-type>";
}

const char* SafeGgmlOpName(enum ggml_op op) {
    return IsValidGgmlOp(op) ? ggml_op_name(op) : "<invalid-op>";
}

std::string TensorDebugSummary(const struct ggml_tensor* tensor) {
    if (!tensor) {
        return "<null>";
    }

    std::string summary;
    summary.reserve(256);
    summary += "ptr=" + std::to_string(reinterpret_cast<uintptr_t>(tensor));
    summary += " name=";
    summary += tensor->name[0] ? tensor->name : "<unnamed>";
    summary += " op=";
    summary += SafeGgmlOpName(tensor->op);
    summary += " type=";
    summary += SafeGgmlTypeName(tensor->type);
    summary += " ne=[";
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d != 0) summary += ",";
        summary += std::to_string(static_cast<long long>(tensor->ne[d]));
    }
    summary += "] nb=[";
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d != 0) summary += ",";
        summary += std::to_string(static_cast<long long>(tensor->nb[d]));
    }
    summary += "]";
    return summary;
}

bool IsHybridSSMSnapshotDebugEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_HYBRID_SSM_SNAPSHOT");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsGraphNodeDumpEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_GRAPH_NODES");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

int GraphNodeDumpLimit() {
    const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_GRAPH_NODE_LIMIT");
    if (!env || env[0] == '\0') {
        return 32;
    }
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || parsed <= 0 || parsed > 4096) {
        return 32;
    }
    return static_cast<int>(parsed);
}

void DebugDumpGraphNodes(const struct ggml_cgraph* graph, const char* stage, int max_nodes = 32) {
    if (!IsGraphNodeDumpEnabled() || !graph) {
        return;
    }
    const int n_nodes = ggml_graph_n_nodes(const_cast<struct ggml_cgraph*>(graph));
    const int configured_limit = GraphNodeDumpLimit();
    const int limit = std::min(n_nodes, std::max(1, std::max(max_nodes, configured_limit)));
    std::cerr << "[GraphNodeDump] stage=" << (stage ? stage : "<unknown>") << " nodes=" << n_nodes
              << " showing=" << limit << std::endl;
    for (int i = 0; i < limit; ++i) {
        struct ggml_tensor* node = ggml_graph_node(const_cast<struct ggml_cgraph*>(graph), i);
        std::cerr << "  node[" << i << "] " << TensorDebugSummary(node) << std::endl;
        if (node) {
            std::cerr << "    src0: " << TensorDebugSummary(node->src[0]) << std::endl;
            std::cerr << "    src1: " << TensorDebugSummary(node->src[1]) << std::endl;
            std::cerr << "    src2: " << TensorDebugSummary(node->src[2]) << std::endl;
        }
    }
}

void DebugLogHybridSSMSnapshot(const char* stage, int req_id, int cached_tokens, int block_id,
                               const std::vector<TransformerModel::SSMSequenceRuntimeState>& states) {
    if (!IsHybridSSMSnapshotDebugEnabled()) {
        return;
    }
    const size_t n_layers = states.size();
    float conv0 = 0.0f;
    float ssm0 = 0.0f;
    if (!states.empty()) {
        if (!states[0].conv_state.empty()) conv0 = states[0].conv_state[0];
        if (!states[0].ssm_state.empty()) ssm0 = states[0].ssm_state[0];
    }
    std::cerr << "[HybridSSMSnapshot] stage=" << (stage ? stage : "<unknown>") << " req=" << req_id
              << " cached_tokens=" << cached_tokens << " block_id=" << block_id << " layers=" << n_layers
              << " conv0=" << conv0 << " ssm0=" << ssm0 << std::endl;
}
