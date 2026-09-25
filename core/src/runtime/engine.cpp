#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

#include "densecore/utils/logging.h"

#include "densecore/hal/backend_registry.h"  // Backend abstraction layer

#include "densecore.h"
#include "densecore/backend/apple/apple_silicon.h"
#include "densecore/backend/cpu_backend.h"
#include "densecore/backend/hardware_topology.h"
#include "densecore/exceptions.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/hal/device_interface.h"  // For Device/Allocator
#include "densecore/hal/tensor.h"            // For Tensor struct
#include "densecore/memory/kv_cache.h"
#include "densecore/models/embedding.h"
#include "densecore/models/model_descriptor.h"
#include "densecore/models/model_graph_capabilities.h"
#include "densecore/models/model_loader.h"
#include "densecore/models/model_types.h"
#include "densecore/models/tokenizer.h"
#include "densecore/runtime/ggml_compute_policy.h"
#include "densecore/runtime/inference.h"
#include "densecore/runtime/optimization_bridge.h"  // Runtime SIMD dispatch
#include "densecore/simd/simd_ops.h"
#include "llm/config/runtime_config.h"
#include "models/model_prompt_templates.h"
#include "runtime/engine_internal.h"
#include "runtime/request_builder.h"
#include "runtime/runtime_env.h"
#include "runtime/worker_internal.h"

#ifdef __APPLE__
static void ConfigureHybridScheduler(EngineState* state, TransformerModel* model) {
    if (!densecore::apple::IsAppleHybridEnabled()) {
        return;
    }

#ifdef DENSECORE_TEST_BUILD
    if (model && model->is_mock) {
        return;
    }
#endif

    state->hybrid_enabled = true;
    state->hybrid_scheduler = std::make_unique<densecore::HybridScheduler>();
    state->hybrid_scheduler->SetCpuBackend(static_cast<densecore::CpuBackend*>(
        state->backend_registry->Get(densecore::DeviceType::CPU)));

    if (densecore::MetalBackend::IsAvailable()) {
        state->hybrid_gpu_backend = std::make_unique<densecore::MetalBackend>();
        state->hybrid_scheduler->SetGpuBackend(state->hybrid_gpu_backend.get());
    }

    if (densecore::ANEBackend::IsAvailable()) {
        state->hybrid_ane_backend = std::make_unique<densecore::ANEBackend>();
        state->hybrid_scheduler->SetAneBackend(state->hybrid_ane_backend.get());
    }

    densecore::HybridScheduler::ModelConfig config = {};
    config.n_layers = model->hparams.n_layer;
    config.hidden_dim = model->hparams.n_embd;
    config.n_heads = model->hparams.n_head;
    config.n_kv_heads = model->hparams.n_head_kv;
    config.head_dim = model->hparams.n_embd_head_k > 0 ? model->hparams.n_embd_head_k
                                                       : (model->hparams.n_embd / model->hparams.n_head);
    config.vocab_size = model->hparams.n_vocab;
    config.max_seq_len = model->hparams.n_ctx;
    if (!model->layers.empty()) {
        config.use_qk_norm = model->layers[0].Get(model_keys::kAttnQNorm) != nullptr;
    } else {
        config.use_qk_norm = false;
    }
    config.intermediate_dim = config.hidden_dim * 4;
    if (!model->layers.empty()) {
        auto* w1 = model->layers[0].Get(model_keys::kFfnGate);
        if (w1) {
            config.intermediate_dim = static_cast<int>(w1->ne[1]);
        }
    }

    state->hybrid_scheduler->ProfileModel(config);

    if (state->hybrid_ane_backend) {
        const densecore::llm::config::EngineAneBootstrapConfig ane_bootstrap_config =
            densecore::llm::config::LoadEngineAneBootstrapConfig();
        const bool bootstrap_buckets = ane_bootstrap_config.bootstrap_buckets;
        if (bootstrap_buckets) {
            densecore::ANEBackend::TransformerLayerConfig ane_cfg = {};
            ane_cfg.hidden_dim = std::max(1, config.hidden_dim);
            ane_cfg.intermediate_dim = std::max(1, config.intermediate_dim);
            ane_cfg.n_heads = std::max(1, config.n_heads);
            ane_cfg.n_kv_heads = std::max(1, config.n_kv_heads);
            ane_cfg.head_dim = std::max(1, config.head_dim);
            ane_cfg.max_seq_len = std::max(1, config.max_seq_len);
            ane_cfg.rms_norm_eps = model->hparams.f_norm_rms_eps > 0.0f ? model->hparams.f_norm_rms_eps : 1e-5f;
            ane_cfg.use_qk_norm = config.use_qk_norm;

            const std::string layer_prefix = ane_bootstrap_config.layer_prefix;

            state->hybrid_ane_backend->PrecompileBucketedModels(layer_prefix, ane_cfg, nullptr);

            // Prime the dynamic bucket path once so missing buckets enter
            // background compile queue and runtime fallback policy is active.
            std::vector<float> warm_input(static_cast<size_t>(ane_cfg.hidden_dim), 0.0f);
            std::vector<float> warm_output(static_cast<size_t>(ane_cfg.hidden_dim), 0.0f);
            std::vector<int> warm_pos(1, 0);
            state->hybrid_ane_backend->ExecuteTransformerLayerDynamic(layer_prefix, warm_input.data(),
                                                                      warm_output.data(), warm_pos.data(), 1, ane_cfg);
        }
    }
}
#endif

namespace {
namespace fs = std::filesystem;

std::unique_ptr<ModelEntry> MakeModelEntry(std::string model_id, std::string model_path,
                                           std::unique_ptr<TransformerModel> model,
                                           std::unique_ptr<PagedKVCache> kv_cache) {
    auto entry = std::make_unique<ModelEntry>();
    entry->model_id = std::move(model_id);
    entry->model_path = std::move(model_path);
    entry->model = std::move(model);
    entry->kv_cache = std::move(kv_cache);
    entry->last_used = std::chrono::steady_clock::now();
    entry->is_loaded = true;
    if (entry->model) {
        entry->transformer_execution_plan = densecore::ResolveTransformerGraphExecutionPlan(entry->model.get());
    }
    return entry;
}

int ReadEnvInt(const char* name, int default_value, bool* was_set = nullptr) {
    return densecore::llm::config::ReadPositiveIntEnv(name, default_value, was_set);
}

size_t ReadAvailableMemoryMbForKVAutoSizing() {
    const char* hint = std::getenv("DENSECORE_KV_AVAILABLE_MB_HINT");
    if (hint && hint[0] != '\0') {
        char* end = nullptr;
        const unsigned long long hinted = std::strtoull(hint, &end, 10);
        if (end != hint && hinted > 0) {
            return static_cast<size_t>(hinted);
        }
    }
    constexpr unsigned long long MB = 1024ULL * 1024ULL;
    auto read_ull_file = [](const char* path) -> unsigned long long {
        std::FILE* file = std::fopen(path, "r");
        if (!file) {
            return 0;
        }
        char buffer[128] = {};
        if (!std::fgets(buffer, sizeof(buffer), file)) {
            std::fclose(file);
            return 0;
        }
        std::fclose(file);
        if (std::strncmp(buffer, "max", 3) == 0) {
            return 0;
        }
        char* end = nullptr;
        const unsigned long long value = std::strtoull(buffer, &end, 10);
        if (end == buffer || value == 0 || value > (1ULL << 50)) {
            return 0;
        }
        return value;
    };

#if defined(__linux__)
    size_t cgroup_available_mb = 0;
    const unsigned long long cgroup_v2_limit = read_ull_file("/sys/fs/cgroup/memory.max");
    const unsigned long long cgroup_v2_current = read_ull_file("/sys/fs/cgroup/memory.current");
    if (cgroup_v2_limit > 0) {
        cgroup_available_mb = static_cast<size_t>(((cgroup_v2_current > 0 && cgroup_v2_limit > cgroup_v2_current)
                                                       ? (cgroup_v2_limit - cgroup_v2_current)
                                                       : cgroup_v2_limit) /
                                                  MB);
    }
    const unsigned long long cgroup_v1_limit = read_ull_file("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    const unsigned long long cgroup_v1_current = read_ull_file("/sys/fs/cgroup/memory/memory.usage_in_bytes");
    if (cgroup_available_mb == 0 && cgroup_v1_limit > 0) {
        cgroup_available_mb = static_cast<size_t>(((cgroup_v1_current > 0 && cgroup_v1_limit > cgroup_v1_current)
                                                       ? (cgroup_v1_limit - cgroup_v1_current)
                                                       : cgroup_v1_limit) /
                                                  MB);
    }

    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (!file) {
        return cgroup_available_mb;
    }
    char line[256] = {};
    unsigned long long kb = 0;
    while (std::fgets(line, sizeof(line), file)) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            std::fclose(file);
            const size_t mem_available_mb = static_cast<size_t>(kb / 1024ULL);
            if (cgroup_available_mb > 0) {
                return std::min(mem_available_mb, cgroup_available_mb);
            }
            return mem_available_mb;
        }
    }
    std::fclose(file);
    return cgroup_available_mb;
#endif
    return 0;
}

size_t SaturatingMulSize(size_t a, size_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > std::numeric_limits<size_t>::max() / b) {
        return std::numeric_limits<size_t>::max();
    }
    return a * b;
}

size_t ResolveAutoKVTargetMb(const TransformerModel* model, int max_num_seqs, size_t bytes_per_token, int max_seq_len) {
    const size_t available_mb = ReadAvailableMemoryMbForKVAutoSizing();
    if (available_mb == 0) {
        return 512;
    }

    const size_t seqs = static_cast<size_t>(std::max(1, max_num_seqs));
    constexpr size_t MB = 1024ULL * 1024ULL;
    const size_t available_bytes = SaturatingMulSize(available_mb, MB);

    // KV sizing should follow the current host/container state instead of a
    // model-name table. The model contributes bytes_per_token and context; the
    // host contributes current free memory. Keep generic runtime and allocator
    // slack, then let the cache use the remaining capacity up to model context.
    const size_t runtime_reserve_bytes = std::max<size_t>(available_bytes / 10, 512ULL * MB);
    const size_t allocator_slack_bytes = std::max<size_t>(available_bytes / 32, 256ULL * MB);
    size_t graph_reserve_bytes = std::max<size_t>(available_bytes / 6, 1024ULL * MB);
    if (model) {
        const size_t graph_seq_hint = static_cast<size_t>(std::max(1, std::min(max_seq_len, 2048)));
        const size_t graph_chunk_hint =
            model->arch_flags.is_hybrid_ssm ? graph_seq_hint : std::min<size_t>(graph_seq_hint, 192);
        const auto graph_estimate =
            EngineState::EstimateGraphContextSize(model, graph_seq_hint, seqs, graph_chunk_hint);
        graph_reserve_bytes = std::max(graph_reserve_bytes, graph_estimate.total_bytes);
        graph_reserve_bytes += std::max<size_t>(graph_estimate.total_bytes / 4, 2048ULL * MB);
    }
    const size_t total_reserve_bytes =
        SaturatingMulSize(1, runtime_reserve_bytes + allocator_slack_bytes + graph_reserve_bytes);
    size_t total_kv_budget_bytes =
        available_bytes > total_reserve_bytes ? available_bytes - total_reserve_bytes : available_bytes / 2;

    if (bytes_per_token > 0 && max_seq_len > 0) {
        const size_t requested_per_seq_bytes =
            SaturatingMulSize(bytes_per_token, static_cast<size_t>(std::max(1, max_seq_len)));
        const size_t requested_total_bytes = SaturatingMulSize(requested_per_seq_bytes, seqs);
        if (requested_total_bytes > 0) {
            total_kv_budget_bytes = std::min(total_kv_budget_bytes, requested_total_bytes);
        }
    }

    const size_t per_seq_budget_mb = (total_kv_budget_bytes / seqs) / MB;
    return std::max<size_t>(1, per_seq_budget_mb);
}

int ResolveDefaultMaxNumSeqs(const TransformerModel* model) {
    if (!model) {
        return 4;
    }

    return 4;
}

}  // namespace

densecore::SchedulerConfig BuildRuntimeSchedulerConfig(const KVCacheConfig& kv_config, const TransformerModel* model) {
    densecore::SchedulerConfig config;
    config.max_num_seqs = std::max(1, kv_config.max_num_seqs);
    // The C4A Qwen3.6 qualification found parallel long-prompt prefill slower
    // than the maintained serial policy, so keep one prefill sequence active.
    config.max_prefill_seqs = 1;
#if defined(DENSECORE_TARGET_C4A)
    // Fill an initial group before advancing its decode positions. Existing
    // streams retain the normal decode/prefill fairness policy.
    if (model && model->variant == ModelVariant::QWEN36 && model->arch_flags.is_hybrid_ssm)
        config.fill_slots_before_first_decode = true;
#else
    (void)model;
#endif
    config.max_model_len = std::max(1, kv_config.max_seq_len);
    config.max_num_batched_tokens = std::max(1, config.max_num_batched_tokens);
    config.max_prefill_tokens = std::min(std::max(1, config.max_prefill_tokens), config.max_num_batched_tokens);
    config.max_parallel_prefill_chunk_tokens = 256;
    config.max_parallel_prefill_chunk_tokens =
        std::min(std::max(1, config.max_parallel_prefill_chunk_tokens), config.max_prefill_tokens);
    config.enable_mixed_prefill_decode = true;
    config.max_mixed_prefill_tokens =
        std::min(std::max(1, config.max_mixed_prefill_tokens), std::max(1, config.max_prefill_tokens / 4));
    return config;
}

namespace {

int ResolveKVHeadDim(const TransformerModel* model) {
    if (!model) {
        return 0;
    }
    if (model->hparams.n_embd_head_k > 0) {
        return model->hparams.n_embd_head_k;
    }
    if (model->hparams.n_head <= 0) {
        return 0;
    }
    return model->hparams.n_embd / model->hparams.n_head;
}

int ResolveKVValueHeadDim(const TransformerModel* model) {
    if (!model) {
        return 0;
    }
    if (model->hparams.n_embd_head_v > 0) {
        return model->hparams.n_embd_head_v;
    }
    if (model->hparams.n_embd_head_k > 0) {
        return model->hparams.n_embd_head_k;
    }
    if (model->hparams.n_head <= 0) {
        return 0;
    }
    return model->hparams.n_embd / model->hparams.n_head;
}

int ResolveKVIndexHeadDim(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_glm_dsa || model->glm_index_head_dim <= 0) {
        return 0;
    }
    return model->glm_index_head_dim;
}

bool IsSupportedKVCacheType(ggml_type cache_type) {
    return cache_type == GGML_TYPE_F32 || cache_type == GGML_TYPE_F16 || cache_type == GGML_TYPE_Q8_0 ||
           cache_type == GGML_TYPE_Q4_0;
}

const char* KVCacheTypeDisplayName(ggml_type cache_type) {
    switch (cache_type) {
    case GGML_TYPE_F32: return "FP32";
    case GGML_TYPE_Q8_0: return "INT8 (Q8_0)";
    case GGML_TYPE_Q4_0: return "INT4 (Q4_0)";
    case GGML_TYPE_F16:
    default: return "FP16";
    }
}

}  // namespace

ggml_type ResolveEffectiveKVCacheType(const TransformerModel* model, ggml_type requested_cache_type) {
    if (!IsSupportedKVCacheType(requested_cache_type)) {
        return GGML_TYPE_F16;
    }

    if (model && model->arch_flags.is_gemma4 && requested_cache_type == GGML_TYPE_Q4_0) {
        return ResolveEffectiveKVCacheType(model, GGML_TYPE_Q8_0);
    }

    if (!ggml_is_quantized(requested_cache_type)) {
        return requested_cache_type;
    }

    if (model && densecore::runtime::IsQwenTargetVariant(model->variant)) {
        return GGML_TYPE_F16;
    }

    const int k_head_dim = ResolveKVHeadDim(model);
    const int v_head_dim = ResolveKVValueHeadDim(model);
    const int quant_block_size = ggml_blck_size(requested_cache_type);
    if (k_head_dim <= 0 || v_head_dim <= 0 || quant_block_size <= 0 || (k_head_dim % quant_block_size) != 0 ||
        (v_head_dim % quant_block_size) != 0) {
        return GGML_TYPE_F16;
    }

    return requested_cache_type;
}

ggml_type ResolveDefaultKVCacheType(const TransformerModel* model) {
    if (model && model->arch_flags.is_gemma4 && ResolveEffectiveKVCacheType(model, GGML_TYPE_Q8_0) == GGML_TYPE_Q8_0) {
        return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_F16;
}

size_t ComputeKVCacheBytesPerToken(ggml_type cache_type, int k_head_dim, int v_head_dim, int n_head_kv, int n_layer,
                                   int index_head_dim) {
    if (k_head_dim <= 0 || v_head_dim <= 0 || n_head_kv <= 0 || n_layer <= 0) {
        return 1;
    }

    if (ggml_is_quantized(cache_type) &&
        ((k_head_dim % ggml_blck_size(cache_type)) != 0 || (v_head_dim % ggml_blck_size(cache_type)) != 0)) {
        cache_type = GGML_TYPE_F16;
    }

    auto bytes_per_slot_for_dim = [&](int head_dim) -> size_t {
        if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
            return ggml_row_size(cache_type, static_cast<int64_t>(head_dim)) * static_cast<size_t>(n_head_kv);
        }
        return ggml_row_size(cache_type, static_cast<int64_t>(head_dim) * static_cast<int64_t>(n_head_kv));
    };

    const size_t k_bytes_per_slot = bytes_per_slot_for_dim(k_head_dim);
    const size_t v_bytes_per_slot = bytes_per_slot_for_dim(v_head_dim);
    const size_t index_bytes_per_slot =
        index_head_dim > 0 ? ggml_row_size(GGML_TYPE_F16, static_cast<int64_t>(index_head_dim)) : 0;

    return (k_bytes_per_slot + v_bytes_per_slot + index_bytes_per_slot) * static_cast<size_t>(n_layer);
}

KVCacheConfig ComputeKVCacheConfig(const TransformerModel* model, ggml_type requested_cache_type) {
    KVCacheConfig config;
    bool max_seq_len_env_set = false;

    config.requested_cache_type = requested_cache_type;
    config.effective_cache_type = ResolveEffectiveKVCacheType(model, requested_cache_type);
    config.max_num_seqs = ReadEnvInt("DENSECORE_MAX_NUM_SEQS", ResolveDefaultMaxNumSeqs(model));
    const int model_ctx = model ? std::max<int32_t>(1, model->hparams.n_ctx) : 4096;
    config.max_seq_len = ReadEnvInt("DENSECORE_MAX_SEQ_LEN", model_ctx, &max_seq_len_env_set);

    const int k_head_dim = ResolveKVHeadDim(model);
    const int v_head_dim = ResolveKVValueHeadDim(model);
    const int index_head_dim = ResolveKVIndexHeadDim(model);
    const int n_head_kv = model ? model->hparams.n_head_kv : 0;
    const int n_layer = model ? model->hparams.n_layer : 0;

    if (model && model->arch == ModelArch::BERT) {
        config.bytes_per_token = 4;
    } else {
        config.bytes_per_token = ComputeKVCacheBytesPerToken(config.effective_cache_type, k_head_dim, v_head_dim,
                                                             n_head_kv, n_layer, index_head_dim);
    }
    if (config.bytes_per_token == 0) {
        config.bytes_per_token = 1;
    }

    bool target_mb_env_set = false;
    int target_mb = ReadEnvInt("DENSECORE_KV_TARGET_MB", 0, &target_mb_env_set);
    if (target_mb <= 0) {
        target_mb = static_cast<int>(
            ResolveAutoKVTargetMb(model, config.max_num_seqs, config.bytes_per_token, config.max_seq_len));
    }
    config.target_kv_memory = static_cast<size_t>(target_mb) * 1024ULL * 1024ULL;

    int optimal_seq_len = static_cast<int>(config.target_kv_memory / config.bytes_per_token);
    optimal_seq_len = std::max(256, std::min(optimal_seq_len, config.max_seq_len));

    if (!target_mb_env_set && !max_seq_len_env_set) {
        const size_t target_total_tokens =
            (config.target_kv_memory * static_cast<size_t>(std::max(1, config.max_num_seqs))) / config.bytes_per_token;
        optimal_seq_len =
            std::max(optimal_seq_len,
                     static_cast<int>(target_total_tokens / static_cast<size_t>(std::max(1, config.max_num_seqs))));
        optimal_seq_len = std::min(optimal_seq_len, config.max_seq_len);
    } else if (!max_seq_len_env_set && config.bytes_per_token < 1024) {
        optimal_seq_len = std::min(optimal_seq_len * 2, 8192);
    }

    config.max_seq_len = optimal_seq_len;
    return config;
}

namespace {

void SetError(DenseCoreStatus status, const std::string& message) {
    SetLastError(status, message);
}

void ClearError() {
    ClearLastError();
}

std::string ReadEnvString(const char* name) {
    return densecore::llm::config::ReadStringEnv(name);
}

TransformerModel* LoadModelWithNuma(const char* model_path, int numa_node_id);

std::string TrimWhitespace(std::string value) {
    const size_t first = value.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return {};
    }
    const size_t last = value.find_last_not_of(" \t\n\r");
    return value.substr(first, last - first + 1);
}

bool IsPathWithinBase(const fs::path& candidate, const fs::path& base_dir) {
    const fs::path relative = candidate.lexically_relative(base_dir);
    if (relative.empty()) {
        return false;
    }
    auto it = relative.begin();
    if (it == relative.end()) {
        return true;
    }
    return *it != "..";
}

bool ContainsParentTraversalSegment(const fs::path& input_path) {
    for (const auto& component : input_path) {
        if (component == "..") {
            return true;
        }
    }
    return false;
}

bool ValidateModelPathArg(const std::string& path, const char* api_name, const char* arg_name,
                          std::string* canonical_out) {
    if (path.find('\0') != std::string::npos) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, std::string(api_name) + ": " + arg_name + " contains null byte");
        return false;
    }

    const std::string trimmed = TrimWhitespace(path);
    if (trimmed.empty()) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                 std::string(api_name) + ": " + arg_name + " is empty or whitespace");
        return false;
    }

    const fs::path input_path(trimmed);
    if (ContainsParentTraversalSegment(input_path)) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                 std::string(api_name) + ": " + arg_name + " contains path traversal");
        return false;
    }

#ifdef DENSECORE_TEST_BUILD
    if (trimmed == "mock") {
        if (canonical_out) {
            *canonical_out = trimmed;
        }
        return true;
    }
#endif

    std::error_code ec;
    const fs::path canonical_path = fs::canonical(input_path, ec);
    if (ec) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                 std::string(api_name) + ": " + arg_name + " cannot be resolved to a canonical path");
        return false;
    }

    if (!fs::is_regular_file(canonical_path, ec) || ec) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                 std::string(api_name) + ": " + arg_name + " must point to an existing file");
        return false;
    }

    const std::string model_base_dir = TrimWhitespace(ReadEnvString("DENSECORE_MODEL_BASE_DIR"));
    if (!model_base_dir.empty()) {
        ec.clear();
        const fs::path canonical_base = fs::canonical(fs::path(model_base_dir), ec);
        if (ec || !fs::is_directory(canonical_base)) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                     std::string(api_name) + ": DENSECORE_MODEL_BASE_DIR is invalid");
            return false;
        }
        if (!IsPathWithinBase(canonical_path, canonical_base)) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                     std::string(api_name) + ": " + arg_name + " must be under DENSECORE_MODEL_BASE_DIR");
            return false;
        }
    }

    if (canonical_out) {
        *canonical_out = canonical_path.string();
    }
    return true;
}

std::string ResolveDraftModelPath(const char* reserved) {
    if (reserved && *reserved != '\0') {
        return TrimWhitespace(std::string(reserved));
    }

    std::string env_draft = TrimWhitespace(ReadEnvString("DRAFT_MODEL_PATH"));
    if (!env_draft.empty()) {
        return env_draft;
    }

    return TrimWhitespace(ReadEnvString("DENSECORE_DRAFT_MODEL_PATH"));
}

bool LoadOptionalDraftModel(EngineState* state, const std::string& main_model_path, const std::string& draft_model_path,
                            ggml_type cache_type, const char* api_name) {
    if (!state || draft_model_path.empty()) {
        return true;
    }

    if (draft_model_path == main_model_path) {
        state->draft_model_id = "default";
        state->draft_model_path = draft_model_path;
        LOG_WARN("Draft model path equals main model path; reusing default model");
        return true;
    }

    TransformerModel* draft_model = LoadModelWithNuma(draft_model_path.c_str(), state->numa_node_id);
    if (!draft_model) {
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, std::string(api_name) + ": failed to load draft model");
        return false;
    }

    KVCacheConfig draft_kv_config = ComputeKVCacheConfig(draft_model, cache_type);
    if (draft_kv_config.effective_cache_type != draft_kv_config.requested_cache_type) {
        LOG_WARN("Draft model KV cache request {} is incompatible with head_dim={}, using {} instead",
                 KVCacheTypeDisplayName(draft_kv_config.requested_cache_type), ResolveKVHeadDim(draft_model),
                 KVCacheTypeDisplayName(draft_kv_config.effective_cache_type));
    }
    PagedKVCache* draft_cache = InitPagedKVCache(draft_model, draft_kv_config.max_num_seqs, draft_kv_config.max_seq_len,
                                                 draft_kv_config.effective_cache_type, state->numa_node_id);
    if (!draft_cache) {
        delete draft_model;
        SetError(DENSECORE_STATUS_OUT_OF_MEMORY, std::string(api_name) + ": failed to initialize draft KV cache");
        return false;
    }

    auto draft_entry = MakeModelEntry("draft", draft_model_path, std::unique_ptr<TransformerModel>(draft_model),
                                      std::unique_ptr<PagedKVCache>(draft_cache));

    state->models["draft"] = std::move(draft_entry);
    state->draft_model_id = "draft";
    state->draft_model_path = draft_model_path;
    LOG_INFO("Draft model loaded: {}", draft_model_path);

    return true;
}

std::string ToLower(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

bool ParseBoolEnv(const char* name, bool default_value) {
    return densecore::llm::config::ReadBoolEnv(name, default_value);
}

TransformerModel* LoadModelWithNuma(const char* model_path, int numa_node_id) {
    const std::string mode_raw = ToLower(ReadEnvString("DENSECORE_NUMA_WEIGHTS"));
    const bool use_huge_pages = ParseBoolEnv("DENSECORE_NUMA_HUGEPAGES", false);

    if (mode_raw.empty()) {
        if (numa_node_id >= 0) {
            std::cout << "[DenseCore] NUMA weights: pinned to node " << numa_node_id << std::endl;
            return LoadGGUFModelNuma(model_path, numa_node_id, use_huge_pages);
        }
        return LoadGGUFModel(model_path);
    }

    if (mode_raw == "off" || mode_raw == "0" || mode_raw == "false") {
        return LoadGGUFModel(model_path);
    }

    if (mode_raw == "auto" || mode_raw == "interleaved") {
        std::cout << "[DenseCore] NUMA weights: interleaved" << std::endl;
        return LoadGGUFModelNuma(model_path, -1, use_huge_pages);
    }

    if (mode_raw == "round_robin" || mode_raw == "round-robin" || mode_raw == "rr") {
        std::cout << "[DenseCore] NUMA weights: round-robin" << std::endl;
        return LoadGGUFModelNuma(model_path, -2, use_huge_pages);
    }

    if (mode_raw == "pinned" || mode_raw == "pin") {
        const int target_node = (numa_node_id >= 0) ? numa_node_id : 0;
        std::cout << "[DenseCore] NUMA weights: pinned to node " << target_node << std::endl;
        return LoadGGUFModelNuma(model_path, target_node, use_huge_pages);
    }

    char* end = nullptr;
    long node = std::strtol(mode_raw.c_str(), &end, 10);
    if (end != mode_raw.c_str() && *end == '\0' && node >= 0) {
        std::cout << "[DenseCore] NUMA weights: pinned to node " << node << std::endl;
        return LoadGGUFModelNuma(model_path, static_cast<int>(node), use_huge_pages);
    }

    std::cerr << "[DenseCore] Unknown DENSECORE_NUMA_WEIGHTS mode '" << mode_raw << "', using default loader"
              << std::endl;
    return LoadGGUFModel(model_path);
}
}  // namespace

int DenseCoreGetRuntimeOptimizationState(DenseCoreHandle handle, DenseCoreRuntimeOptimizationState* out) {
    if (!handle || !out) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreGetRuntimeOptimizationState: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    *out = DenseCoreRuntimeOptimizationState{};
    out->struct_size = static_cast<int>(sizeof(DenseCoreRuntimeOptimizationState));

    EngineState* state = static_cast<EngineState*>(handle);
    const ModelEntry* entry = state->GetDefaultModel();
    const TransformerModel* model = (entry && entry->model) ? entry->model.get() : nullptr;
    const auto& config = state->fast_path_config;
    const ModelVariant model_variant = model ? densecore::models::DescribeModel(model).variant : ModelVariant::UNKNOWN;
    const bool qwen35_hybrid =
        model && model->arch_flags.is_hybrid_ssm && model_variant == ModelVariant::QWEN35;
    const bool qwen36_hybrid =
        model && model->arch_flags.is_hybrid_ssm && model_variant == ModelVariant::QWEN36;
    const bool qwen38_hybrid =
        model && model->arch_flags.is_hybrid_ssm && model_variant == ModelVariant::QWEN38;
    const bool graph_reuse_globally_enabled = !config.worker.graph_cache_reuse_disabled;
    const bool prefill_graph_model_eligible =
        model && model->hparams.n_experts == 0 && !model->arch_flags.is_hybrid_ssm;
    const bool prefill_arena_model_eligible = model && model->arch_flags.is_gemma4;
    out->token_id_submit_supported = 1;
    out->prefix_cache_reuse_enabled =
        (!config.worker.prefix_cache_reuse_disabled &&
         (!qwen36_hybrid || config.worker.qwen36_prefix_cache_reuse_enabled) && !qwen35_hybrid && !qwen38_hybrid)
            ? 1
            : 0;
    // Match worker prefix eligibility and recurrent-state restore gating.
    // Attention-only models have no snapshot to restore; LFM2 uses conv state.
    out->hybrid_ssm_snapshot_restore_enabled =
        (out->prefix_cache_reuse_enabled && model &&
         (model->arch_flags.is_hybrid_ssm || model->arch_flags.is_lfm2_shortconv) &&
         !config.worker.hybrid_ssm_snapshot_restore_disabled &&
         (!qwen36_hybrid || config.worker.qwen36_hybrid_ssm_snapshot_restore_enabled))
            ? 1
            : 0;
    out->prefill_graph_cache_enabled =
        (config.prefill_graph_cache.enabled && graph_reuse_globally_enabled && prefill_graph_model_eligible) ? 1 : 0;
    // The maintained Gemma4 safety lane keeps graph topology reuse fail-closed.
    // Arena/context reuse is the safe prefill optimization represented by the
    // same graph-cache disable gate until the dedicated arena pool grows a
    // separate runtime knob.
    out->prefill_arena_reuse_enabled = (graph_reuse_globally_enabled && prefill_arena_model_eligible) ? 1 : 0;
    out->decode_graph_cache_enabled = (IsDecodeGraphCacheEnabled() && graph_reuse_globally_enabled) ? 1 : 0;
    out->decode_graph_cache_max_batch = DecodeGraphCacheMaxBatch();
    out->decode_graph_cache_lru_size = DecodeGraphCacheLruSize();
    out->moe_dequant_cache_mb = []() -> int {
        const char* env = std::getenv("DENSECORE_MOE_DEQUANT_CACHE_MB");
        if (!env || *env == '\0') {
            return 512;
        }
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end == env || *end != '\0' || parsed < 0) {
            return 512;
        }
        return static_cast<int>(std::min<long>(parsed, std::numeric_limits<int>::max()));
    }();

    std::snprintf(out->active_thread_policy_label, sizeof(out->active_thread_policy_label), "%s",
                  "model_aware_thread_policy");
    ClearError();
    return DENSECORE_STATUS_OK;
}

namespace {

struct InitPaths {
    std::string model_path;
    std::string draft_model_path;
};

bool ResolveInitPaths(const char* model_path, const char* reserved, const char* api_name, InitPaths* paths) {
    if (!model_path) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "InitEngineEx: model_path is null");
        return false;
    }

    if (!ValidateModelPathArg(model_path, api_name, "model_path", &paths->model_path)) {
        return false;
    }

    const std::string draft_model_path = ResolveDraftModelPath(reserved);
    if (!draft_model_path.empty() &&
        !ValidateModelPathArg(draft_model_path, api_name, "draft_model_path", &paths->draft_model_path)) {
        return false;
    }
    return true;
}

bool ValidateRuntimeEnvironment(const char* api_name) {
    const std::string removed_alias_error = densecore::llm::config::RemovedEnvironmentAliasError();
    if (removed_alias_error.empty()) {
        return true;
    }
    SetError(DENSECORE_STATUS_INVALID_ARGUMENT, std::string(api_name) + ": " + removed_alias_error);
    return false;
}

int ResolveEngineThreadCount(int threads) {
    if (threads > 0) {
        return threads;
    }
    auto& topo = densecore::HardwareTopology::GetInstance();
    threads = topo.GetPhysicalCoreCount();
    if (threads <= 0) {
        threads = topo.GetLogicalCoreCount();
    }
    if (threads <= 0) {
        threads = densecore::simd::GetNumCores();
    }
    return threads;
}

void InitializeRuntimeBackends(int threads) {
    // =========================================================================
    // SIMD DISPATCH TABLE INITIALIZATION (Must happen before any inference!)
    // =========================================================================
    // OpsRegistry selects optimal kernel implementations based on CPU caps.
    // Initializing here ensures the dispatch table is ready before generate()
    // is called, even if the worker thread hasn't started yet.
    // =========================================================================
    if (!densecore::OpsRegistry::IsInitialized()) {
        densecore::OpsRegistry::Init();
    }

    // =========================================================================
    // BACKEND REGISTRY INITIALIZATION
    // =========================================================================
    // Register the CPU backend (AVX2/AVX-512 kernels wrapped in ComputeBackend
    // interface). Future ASIC backends can be registered similarly.
    // =========================================================================
    // Mutable backend state is initialized on EngineState after model loading.
    (void)threads;
}

std::unique_ptr<EngineState> InitializeEngineResources(const InitPaths& paths, int threads, int numa_node_id,
                                                       int pinning_policy) {
    auto model_owner = std::unique_ptr<TransformerModel>(LoadModelWithNuma(paths.model_path.c_str(), numa_node_id));
    TransformerModel* model = model_owner.get();
    if (!model) {
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "InitEngineEx: failed to load model");
        return nullptr;
    }

    auto state = std::make_unique<EngineState>();
    state->numa_node_id = numa_node_id;
    state->pinning_policy = pinning_policy;
    state->n_threads = threads;
    state->fast_path_config = densecore::llm::config::LoadFastPathRuntimeConfig();
    state->op_registry = std::make_unique<densecore::OpRegistry>();
    state->InitializeExecutionDependencies(threads);

    const ggml_type default_cache_type = ResolveDefaultKVCacheType(model);
    state->requested_kv_cache_type = default_cache_type;
    const KVCacheConfig kv_config = ComputeKVCacheConfig(model, default_cache_type);
    std::cout << "[DenseCore] Auto-configured max_seq_len: " << kv_config.max_seq_len << " (KV capacity: ~"
              << (kv_config.bytes_per_token * kv_config.max_seq_len *
                  static_cast<size_t>(std::max(1, kv_config.max_num_seqs)) / 1024 / 1024)
              << " MB total, " << kv_config.bytes_per_token << " bytes/token, max_num_seqs=" << kv_config.max_num_seqs
              << ")" << std::endl;

    // Log Flash Attention status based on CPU capabilities
    densecore::simd::SimdLevel simd_level = densecore::simd::DetectSimdLevel();
#if defined(__AVX512F__)
    constexpr bool compiled_with_x86_avx512 = true;
#else
    constexpr bool compiled_with_x86_avx512 = false;
#endif
    if (densecore::simd::HasX86Avx512OrBetter(simd_level)) {
        LOG_INFO("Native x86 Flash Attention Enabled ({} detected)", densecore::simd::SimdLevelName(simd_level));
    } else if (densecore::simd::IsArmFamily(simd_level)) {
        LOG_INFO("Native x86 Flash Attention Unavailable on {}; portable CPU flash attention remains eligible",
                 densecore::simd::SimdLevelName(simd_level));
    } else {
        LOG_WARN("Native x86 Flash Attention Disabled (requires AVX-512, detected: {})",
                 densecore::simd::SimdLevelName(simd_level));
    }
    LOG_INFO("Compile-time SIMD features: x86_avx512={} runtime_simd={}", compiled_with_x86_avx512 ? "1" : "0",
             densecore::simd::SimdLevelName(simd_level));

    // Log NUMA configuration
    if (state->numa_node_id >= 0) {
        LOG_INFO("NUMA binding: node {}", state->numa_node_id);
    }

    // Initialize Paged KV Cache with NUMA-aware allocation
    PagedKVCache* cache = InitPagedKVCache(model, kv_config.max_num_seqs, kv_config.max_seq_len,
                                           kv_config.effective_cache_type, state->numa_node_id);
    if (!cache) {
        // model_owner releases the loaded model on this failure path.
        SetError(DENSECORE_STATUS_OUT_OF_MEMORY, "InitEngineEx: failed to initialize KV cache");
        return nullptr;
    }
    auto entry = MakeModelEntry("default", paths.model_path, std::move(model_owner),
                                std::unique_ptr<PagedKVCache>(cache));
    state->models["default"] = std::move(entry);
    state->default_model_id = "default";

    if (!LoadOptionalDraftModel(state.get(), paths.model_path, paths.draft_model_path, kv_config.effective_cache_type,
                                "InitEngineEx")) {
        return nullptr;
    }

#ifdef __APPLE__
    ConfigureHybridScheduler(state.get(), state->models["default"]->model.get());
    // Create platform-specific BackendSelector (uses HybridScheduler on Apple)
    state->backend_selector = densecore::CreateBackendSelector(state->hybrid_scheduler.get());
#else
    // Create default CPU-only BackendSelector on non-Apple platforms
    state->backend_selector = densecore::CreateBackendSelector(nullptr);
#endif

    if (!state->models["default"]->kv_cache || !state->models["default"]->kv_cache->block_manager) {
        LOG_CRITICAL("FATAL: Failed to initialize Scheduler. BlockManager is missing.");
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, "InitEngineEx: failed to initialize scheduler");
        return nullptr;
    }
    state->scheduler = std::make_unique<densecore::Scheduler>(state->models["default"]->kv_cache->block_manager,
                                                              BuildRuntimeSchedulerConfig(kv_config, state->models["default"]->model.get()));
    LOG_INFO("Scheduler initialized.");

    state->InitComputeBuffer();
    return state;
}

bool StartEngineThreads(EngineState* state) {
    if (!state) {
        return false;
    }
    state->status = EngineStatus::RUNNING;
    state->worker_thread = std::thread(EngineLoop, state);
    state->callback_thread = std::thread(CallbackLoop, state);
    LOG_INFO("Started worker thread and callback thread");
    return true;
}

}  // namespace

/**
 * @brief Initialize engine with default NUMA settings (simplified API)
 *
 * This is a convenience wrapper that calls InitEngineEx with:
 * - numa_node_id = -1 (auto-detect)
 * - pinning_policy = 0 (SCATTER)
 */
DENSECORE_API DenseCoreHandle InitEngine(const char* model_path, const char* reserved, int threads) {
    return InitEngineEx(model_path, reserved, threads, -1, 0);
}

/**
 * @brief Initialize engine with NUMA and thread pinning control (extended API)
 */
DENSECORE_API DenseCoreHandle InitEngineEx(const char* model_path, const char* reserved, int threads, int numa_node_id,
                                           int pinning_policy) {
    try {
        ClearError();
        // Initialize logging subsystem (once)
        static std::once_flag log_init_flag;
        std::call_once(log_init_flag, []() { densecore::utils::InitLogging(); });

        if (!ValidateRuntimeEnvironment("InitEngineEx")) {
            return nullptr;
        }

        InitPaths init_paths;
        if (!ResolveInitPaths(model_path, reserved, "InitEngineEx", &init_paths)) {
            return nullptr;
        }


        threads = ResolveEngineThreadCount(threads);
        LOG_INFO("Threading configured: {} threads (GGML thread pool + std::thread, no OpenMP)", threads);

        InitializeRuntimeBackends(threads);
        auto state = InitializeEngineResources(init_paths, threads, numa_node_id, pinning_policy);
        if (!state) {
            return nullptr;
        }

        if (!StartEngineThreads(state.get())) {
            return nullptr;
        }

        ClearError();
        return (DenseCoreHandle)state.release();  // Transfer ownership to caller
    } catch (const densecore::DenseCoreException& e) {
        LOG_ERROR("Exception in InitEngine: {}", e.what());
        SetError(MapErrorCodeToStatus(e.Code()), e.what());
        return nullptr;
    } catch (const std::exception& e) {
        LOG_ERROR("Exception in InitEngine: {}", e.what());
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, "InitEngineEx: exception");
        return nullptr;
    }
}

/**
 * @brief Initialize engine with KV cache type specification
 *
 * Extended initialization with control over KV cache precision.
 * INT8 KV cache uses approximately half the storage of FP16. Output quality
 * must be validated for the target model and workload.
 */
DENSECORE_API DenseCoreHandle InitEngineWithKVType(const char* model_path, const char* reserved, int threads,
                                                   int numa_node_id, int pinning_policy,
                                                   DenseCoreKVType kv_cache_type) {
    try {
        ClearError();
        if (!ValidateRuntimeEnvironment("InitEngineWithKVType")) {
            return nullptr;
        }
        if (!model_path) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "InitEngineWithKVType: model_path is null");
            return nullptr;
        }

        std::string canonical_model_path;
        if (!ValidateModelPathArg(model_path, "InitEngineWithKVType", "model_path", &canonical_model_path)) {
            return nullptr;
        }
        const std::string draft_model_path = ResolveDraftModelPath(reserved);
        std::string canonical_draft_model_path;
        if (!draft_model_path.empty() && !ValidateModelPathArg(draft_model_path, "InitEngineWithKVType",
                                                               "draft_model_path", &canonical_draft_model_path)) {
            return nullptr;
        }
        // Convert enum to ggml_type
        ggml_type cache_type = GGML_TYPE_F16;  // Default
        switch (kv_cache_type) {
        case DENSECORE_KV_FP32: cache_type = GGML_TYPE_F32; break;
        case DENSECORE_KV_INT8:
            cache_type = GGML_TYPE_Q8_0;
            std::cout << "[DenseCore] Using INT8 KV Cache (50% memory reduction)" << std::endl;
            break;
        case DENSECORE_KV_INT4:
            cache_type = GGML_TYPE_Q4_0;
            std::cout << "[DenseCore] Using INT4 KV Cache (75% memory reduction)" << std::endl;
            break;
        case DENSECORE_KV_FP16:
        default: cache_type = GGML_TYPE_F16; break;
        }

        // Reuse InitEngineEx logic but with custom KV cache type
        // Thread configuration
        if (threads <= 0) {
            threads = densecore::simd::GetNumCores();
        }
        std::cout << "[DenseCore] Threading configured: " << threads << " threads" << std::endl;

        // SIMD dispatch initialization
        if (!densecore::OpsRegistry::IsInitialized()) {
            densecore::OpsRegistry::Init();
        }

        // Backend registration
        // Each EngineState initializes its own backend after creating OpRegistry.

        auto model_owner = std::unique_ptr<TransformerModel>(LoadModelWithNuma(canonical_model_path.c_str(), numa_node_id));
        TransformerModel* model = model_owner.get();
        if (!model) {
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "InitEngineWithKVType: failed to load model");
            return nullptr;
        }

        // RAII: Use unique_ptr to guarantee cleanup on exception
        auto state = std::make_unique<EngineState>();
        state->numa_node_id = numa_node_id;
        state->pinning_policy = pinning_policy;
        state->n_threads = threads;
        state->requested_kv_cache_type = cache_type;
        state->fast_path_config = densecore::llm::config::LoadFastPathRuntimeConfig();

        // Initialize OpRegistry (Dependency Injection)
        state->op_registry = std::make_unique<densecore::OpRegistry>();
        state->InitializeExecutionDependencies(threads);

        KVCacheConfig kv_config = ComputeKVCacheConfig(model, cache_type);
        if (kv_config.effective_cache_type != kv_config.requested_cache_type) {
            std::cout << "[DenseCore] Requested " << KVCacheTypeDisplayName(kv_config.requested_cache_type)
                      << " KV cache is not supported for this model/runtime; using "
                      << KVCacheTypeDisplayName(kv_config.effective_cache_type) << " instead" << std::endl;
        }

        std::cout << "[DenseCore] Auto-configured max_seq_len: " << kv_config.max_seq_len << " (KV capacity: ~"
                  << (kv_config.bytes_per_token * kv_config.max_seq_len *
                      static_cast<size_t>(std::max(1, kv_config.max_num_seqs)) / 1024 / 1024)
                  << " MB total, max_num_seqs=" << kv_config.max_num_seqs << ")" << std::endl;

        // Initialize PagedKVCache with specified type
        PagedKVCache* cache = InitPagedKVCache(model, kv_config.max_num_seqs, kv_config.max_seq_len,
                                               kv_config.effective_cache_type, state->numa_node_id);
        if (!cache) {
            // model_owner releases the loaded model on this failure path.
            // state is automatically cleaned up by unique_ptr
            SetError(DENSECORE_STATUS_OUT_OF_MEMORY, "InitEngineWithKVType: failed to initialize KV cache");
            return nullptr;
        }

        // Log cache type
        const char* type_name = KVCacheTypeDisplayName(kv_config.effective_cache_type);
        std::cout << "[DenseCore] KV Cache initialized with type: " << type_name << std::endl;

        // Create model entry
        auto entry = MakeModelEntry("default", canonical_model_path, std::move(model_owner),
                                    std::unique_ptr<PagedKVCache>(cache));
        state->models["default"] = std::move(entry);
        state->default_model_id = "default";
        if (!LoadOptionalDraftModel(state.get(), canonical_model_path, canonical_draft_model_path, cache_type,
                                    "InitEngineWithKVType")) {
            return nullptr;
        }

#ifdef __APPLE__
        ConfigureHybridScheduler(state.get(), state->models["default"]->model.get());
        state->backend_selector = densecore::CreateBackendSelector(state->hybrid_scheduler.get());
#else
        state->backend_selector = densecore::CreateBackendSelector(nullptr);
#endif

        // Initialize Scheduler
        if (state->models["default"]->kv_cache && state->models["default"]->kv_cache->block_manager) {
            state->scheduler = std::make_unique<densecore::Scheduler>(state->models["default"]->kv_cache->block_manager,
                                                                      BuildRuntimeSchedulerConfig(kv_config, state->models["default"]->model.get()));
            std::cout << "[DenseCore] Scheduler initialized." << std::endl;
        } else {
            std::cerr << "[DenseCore] FATAL: Failed to initialize Scheduler." << std::endl;
            SetError(DENSECORE_STATUS_INTERNAL_ERROR, "InitEngineWithKVType: failed to initialize scheduler");
            // state is automatically cleaned up by unique_ptr
            return nullptr;
        }

        // Initialize compute buffer
        state->InitComputeBuffer();

        // Start background threads
        state->status = EngineStatus::RUNNING;
        state->worker_thread = std::thread(EngineLoop, state.get());
        state->callback_thread = std::thread(CallbackLoop, state.get());

        std::cout << "[DenseCore] Started worker thread and callback thread" << std::endl;
        ClearError();
        return (DenseCoreHandle)state.release();  // Transfer ownership to caller

    } catch (const densecore::DenseCoreException& e) {
        std::cerr << "[DenseCore] Exception in InitEngineWithKVType: " << e.what() << std::endl;
        SetError(MapErrorCodeToStatus(e.Code()), e.what());
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "[DenseCore] Exception in InitEngineWithKVType: " << e.what() << std::endl;
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, "InitEngineWithKVType: exception");
        return nullptr;
    }
}

// =============================================================================
// HAL & Plugin API Implementation
// =============================================================================

DENSECORE_API int DenseCoreLoadPlugin(const char* path) {
    if (!path || path[0] == '\0') {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreLoadPlugin: path is null or empty");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::string error_message;
        if (!densecore::BackendRegistry::Instance().LoadPlugin(path, &error_message)) {
            SetError(DENSECORE_STATUS_BACKEND_ERROR,
                     "DenseCoreLoadPlugin: " +
                         (error_message.empty() ? std::string("backend plugin load failed") : error_message));
            return DENSECORE_STATUS_BACKEND_ERROR;
        }
        ClearError();
        return DENSECORE_STATUS_OK;
    } catch (const std::exception& e) {
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, std::string("DenseCoreLoadPlugin: ") + e.what());
        return DENSECORE_STATUS_INTERNAL_ERROR;
    }
}

DENSECORE_API DenseCoreDeviceHandle DenseCoreCreateDevice(const char* backend_name, int device_id) {
    (void)device_id;
    if (!backend_name) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreCreateDevice: backend_name is null");
        return nullptr;
    }
    try {
        std::string name = backend_name;
        densecore::DeviceType type = densecore::DeviceType::CPU;

        // Simple string parsing (case-insensitive preferred but strict here for simplicity)
        if (name == "CPU")
            type = densecore::DeviceType::CPU;
        else if (name == "METAL")
            type = densecore::DeviceType::METAL;
        else if (name == "NPU")
            type = densecore::DeviceType::NPU;
        else if (name == "ASIC")
            type = densecore::DeviceType::ASIC;
        else {
            // Try to find custom device by ID if needed, or fail
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreCreateDevice: unknown backend name");
            return nullptr;
        }

        densecore::ComputeBackend* backend = densecore::BackendRegistry::Instance().Get(type);
        if (!backend) {
            SetError(DENSECORE_STATUS_INTERNAL_ERROR, "DenseCoreCreateDevice: backend not registered");
            return nullptr;
        }

        // Return the backend pointer as the handle (it effectively acts as the device/context provider)
        // In a real multi-device scenario, we might return a specific Device instance.
        // For this API level, ComputeBackend suffices as the entry point.
        ClearError();
        return (DenseCoreDeviceHandle)backend;
    } catch (const std::exception& e) {
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, std::string("DenseCoreCreateDevice: ") + e.what());
        return nullptr;
    }
}

namespace {
// Helper to parse "layers.N.name"
struct TensorPath {
    int layer_idx = -1;
    std::string subname;
    bool is_layer = false;
};

TensorPath ParseTensorName(const std::string& name) {
    TensorPath res;
    if (name.rfind("layers.", 0) == 0) {
        // Starts with "layers."
        size_t first_dot = 6;  // len("layers.")
        size_t second_dot = name.find('.', first_dot);
        if (second_dot != std::string::npos) {
            std::string num_str = name.substr(first_dot, second_dot - first_dot);
            try {
                res.layer_idx = std::stoi(num_str);
                res.subname = name.substr(second_dot + 1);
                res.is_layer = true;
            } catch (...) {
                // Parse error
            }
        }
    } else {
        res.subname = name;  // Top-level
    }
    return res;
}

ggml_tensor** FindTensorPointer(TransformerModel* model, const std::string& name) {
    // 1. Check top-level mappings
    if (name == "model.embed_tokens.weight") return &model->tok_embeddings;
    if (name == "model.norm.weight") return &model->output_norm;
    if (name == "lm_head.weight") return &model->output;

    // 2. Check layers
    TensorPath path = ParseTensorName(name);
    if (!path.is_layer || path.layer_idx < 0 || path.layer_idx >= (int)model->layers.size()) {
        return nullptr;
    }

    TransformerLayer& layer = model->layers[path.layer_idx];
    const std::string& sub = path.subname;

    static const std::unordered_map<std::string, const char*> layer_map = {
        {"self_attn.q_proj.weight", model_keys::kAttnQWeight},
        {"self_attn.k_proj.weight", model_keys::kAttnKWeight},
        {"self_attn.v_proj.weight", model_keys::kAttnVWeight},
        {"self_attn.o_proj.weight", model_keys::kAttnOWeight},
        {"mlp.gate_proj.weight", model_keys::kFfnGate},
        {"mlp.up_proj.weight", model_keys::kFfnUp},
        {"mlp.down_proj.weight", model_keys::kFfnDown},
        {"input_layernorm.weight", model_keys::kAttnNorm},
        {"post_attention_layernorm.weight", model_keys::kFfnNorm},

        // RoPE Norms (Qwen)
        {"self_attn.q_norm.weight", model_keys::kAttnQNorm},
        {"self_attn.k_norm.weight", model_keys::kAttnKNorm},

        // Biases
        {"self_attn.q_proj.bias", model_keys::kAttnQBias},
        {"self_attn.k_proj.bias", model_keys::kAttnKBias},
        {"self_attn.v_proj.bias", model_keys::kAttnVBias},
        {"self_attn.o_proj.bias", model_keys::kAttnOBias},
    };

    auto it = layer_map.find(sub);
    if (it != layer_map.end()) {
        return layer.GetMutable(it->second);
    }

    return nullptr;
}
}  // namespace

DENSECORE_API int DenseCoreSetWeight(DenseCoreHandle engine, const char* name, const DenseCoreTensor* tensor) {
    if (!engine || !name || !tensor) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreSetWeight: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)engine;
    ModelEntry* entry = state->GetDefaultModel();
    if (!entry || !entry->model) {
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "DenseCoreSetWeight: no model loaded");
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    }

    TransformerModel* model = entry->model.get();
    ggml_tensor** target_ptr = FindTensorPointer(model, name);

    if (!target_ptr) {
        // Tensor not found in model structure
        // This is expected for some keys that might be in the safetensors file but unused by our runtime
        return DENSECORE_STATUS_OK;
    }

    ggml_tensor* target = *target_ptr;
    if (!target) {
        // Start initialization if null? Usually model structure is pre-allocated.
        // Assuming target IS instantiated but data might be empty.
        // If pointers are null, we might need to verify if we should allocate them.
        SetError(DENSECORE_STATUS_INTERNAL_ERROR,
                 std::string("DenseCoreSetWeight: target tensor structure is null for ") + name);
        return DENSECORE_STATUS_INTERNAL_ERROR;
    }

    // Basic shape validation
    // GGML is column-major in dimension definitions? No, ne[0] is fastest changing.
    // safetensors/numpy is row-major.
    // DenseCoreTensor shape is [d0, d1, d2, d3] row-major.
    // We should copy data bytes directly.

    size_t size_bytes = tensor->shape[0] * std::max((int64_t)1, tensor->shape[1]) *
                        std::max((int64_t)1, tensor->shape[2]) * std::max((int64_t)1, tensor->shape[3]);

    // Calculate bytes based on dtype
    switch (tensor->dtype) {
    case 0: size_bytes *= 4; break;  // F32
    case 1: size_bytes *= 2; break;  // F16
    case 2: size_bytes *= 2; break;  // BF16
    case 3: size_bytes *= 1; break;  // INT8
    default: break;
    }

    if (ggml_nbytes(target) != size_bytes) {
        std::cerr << "[DenseCore] Warning: DenseCoreSetWeight size mismatch for tensor '" << name
                  << "'. Target bytes: " << ggml_nbytes(target) << ", Source bytes: " << size_bytes << std::endl;
        // Mismatch size - logging warning or error?
        // This might happen due to transposition.
        // Let's assume the user knows what they are doing and simply copy for now,
        // but strictly checking size is safer.
    }

    // Perform copy
    // If target->data is null, we need to allocate it?
    // GGUF loader usually mmap, but here we are manual loading.
    // We assume the model structure was created with "empty" tensors or we need to allocate them.
    if (!target->data) {
        // Allocate host memory (or backend memory)
        // Here we just use malloc/posix_memalign or backend->AllocateUnified
        // Simplified: use malloc
        target->data = malloc(size_bytes);
    }

    std::memcpy(target->data, tensor->data, size_bytes);

    ClearError();
    return DENSECORE_STATUS_OK;
}

// =============================================================================
// Internal Workers
// =============================================================================
/**
 * CallbackLoop - Dedicated thread for callback execution.
 *
 * This function runs on a dedicated thread to execute callbacks without
 * blocking the worker thread. It waits on result_cv for new ResultEvents,
 * pops events from the result_queue, and executes the callbacks.
 *
 * This decoupling prevents the Python GIL from blocking the inference
 * hot path, resolving the streaming deadlock.
 */
void CallbackLoop(EngineState* state) {
    try {
        while (true) {
            ResultEvent event;
            {
                std::unique_lock<std::mutex> lock(state->result_mu);
                state->result_cv.wait(lock, [state]() {
                    return !state->result_queue.empty() || state->result_producers_done.load(std::memory_order_acquire);
                });

                if (state->result_queue.empty() &&
                    state->result_producers_done.load(std::memory_order_acquire)) {
                    break;  // Exit: shutdown complete
                }

                if (state->result_queue.empty()) continue;

                event = std::move(state->result_queue.front());
                state->result_queue.pop_front();

                // ===========================================================
                // BACKPRESSURE RELEASE: Notify producers if queue drains
                // ===========================================================
                if (state->result_queue.size() <= EngineState::RESULT_QUEUE_LOW_WATERMARK) {
                    state->result_drain_cv.notify_all();
                }
            }

            if (state->fast_path_config.engine_debug.verbose_token_trace) {
                std::cerr << "[TRACE] CallbackLoop popped event for request " << event.request_id
                          << " (finished=" << event.finished << ")" << std::endl;
            }

            // Execute callback OUTSIDE the lock (GIL acquisition happens here)
            // This is the ONLY place callbacks should be invoked!
            try {
                if (event.completion_callback) {
                    event.completion_callback(event.generated_tokens, event.finish_reason, event.user_data);
                }
                if (event.callback_ex) {
                    event.callback_ex(event.token_str.data(), static_cast<int>(event.token_str.size()), event.token_id,
                                      event.finished ? 1 : (event.error ? 1 : 0), event.user_data);
                } else if (event.callback) {
                    event.callback(event.token_str.c_str(), event.finished ? 1 : (event.error ? 1 : 0),
                                   event.user_data);
                } else if (event.token_result_callback) {
                    TokenResult result;
                    result.token_id = event.token_id;
                    result.text = event.token_str.c_str();
                    result.is_finished = event.finished ? 1 : (event.error ? 1 : 0);
                    event.token_result_callback(&result, event.user_data);
                } else if (event.emb_callback) {
                    if (event.embedding_status == DENSECORE_STATUS_OK && !event.embedding_data.empty()) {
                        event.emb_callback(event.embedding_data.data(), static_cast<int>(event.embedding_data.size()),
                                           event.user_data);
                    } else {
                        const int terminal_status = event.embedding_status < 0 ? event.embedding_status
                                                                               : DENSECORE_STATUS_INTERNAL_ERROR;
                        event.emb_callback(nullptr, terminal_status, event.user_data);
                    }
                    // Note: embedding_data is RAII-managed by std::vector, no manual free
                }
            } catch (const std::exception& e) {
                std::cerr << "[DenseCore] Exception in callback (request " << event.request_id << "): " << e.what()
                          << std::endl;
            } catch (...) {
                std::cerr << "[DenseCore] Unknown exception in callback (request " << event.request_id << ")"
                          << std::endl;
            }
        }

        std::cerr << "[DenseCore] CallbackLoop exiting" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[DenseCore] CallbackLoop thread exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[DenseCore] CallbackLoop thread unknown exception" << std::endl;
    }
}

int CancelRequest(DenseCoreHandle handle, int request_id) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "CancelRequest: handle is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    // ==========================================================================
    // LOCK-FREE QUEUE LIMITATION: Cannot iterate/remove from pending queue
    // ==========================================================================
    // With lock-free queues, we can only cancel active requests.
    // Pending requests will be checked for cancellation when dequeued.
    // ==========================================================================
    {
        std::lock_guard<std::mutex> lock(state->active_mu);
        for (Request* req : state->active_requests) {
            if (req->id == request_id) {
                req->cancelled.store(true, std::memory_order_relaxed);
                LOG_INFO("Marked active request ", request_id, " for cancellation.");
                ClearError();
                return 0;  // Success
            }
        }
    }

    // Request not found in active list - it might be pending
    // Mark cancellation will happen lazily when request is dequeued
    if (state->IsPendingRequest(request_id)) {
        state->RecordPendingCancellation(request_id);
        LOG_INFO("Request ", request_id, " not active - cancellation queued.");
        ClearError();
        return 0;
    }
    LOG_INFO("Request ", request_id, " not active.");
    SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "CancelRequest: request not active");
    return DENSECORE_STATUS_INVALID_ARGUMENT;
}

DenseCoreMetrics GetMetrics(DenseCoreHandle handle) {
    DenseCoreMetrics m = {};
    if (handle) {
        EngineState* state = (EngineState*)handle;
        m.active_requests = state->metrics.active_requests;
        m.total_tokens_generated = state->metrics.total_tokens_generated;
        auto now = std::chrono::steady_clock::now();
        double elapsed_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - state->metrics.start_time).count();
        if (elapsed_s > 0.0) {
            m.tokens_per_second = static_cast<float>(state->metrics.total_tokens_generated.load() / elapsed_s);
            m.requests_per_second = static_cast<float>(state->metrics.total_requests.load() / elapsed_s);
        }
    }
    return m;
}

DetailedMetrics GetDetailedMetrics(DenseCoreHandle handle) {
    DetailedMetrics m = {};
    if (!handle) return m;

    EngineState* state = (EngineState*)handle;

    // Request metrics
    m.active_requests = state->metrics.active_requests;
    m.total_requests = state->metrics.total_requests;
    m.completed_requests = state->metrics.completed_requests;
    m.failed_requests = state->metrics.failed_requests;

    // Count pending requests (lock-free approximate count)
    m.pending_requests = state->pending_requests.Size();
    {
        std::lock_guard<std::mutex> lock(state->active_mu);
        m.current_batch_size = state->active_requests.size();
    }

    // Token metrics
    m.total_tokens_generated = state->metrics.total_tokens_generated;
    m.total_prompt_tokens = state->metrics.total_prompt_tokens;

    // Calculate TPS (tokens per second) over engine lifetime
    {
        auto now = std::chrono::steady_clock::now();
        double elapsed_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - state->metrics.start_time).count();
        if (elapsed_s > 0.0) {
            m.tokens_per_second = static_cast<float>(m.total_tokens_generated / elapsed_s);
        }
    }

    // Latency metrics from samples
    {
        std::lock_guard<std::mutex> lock(state->metrics.metrics_mu);

        // TTFT metrics
        m.avg_time_to_first_token = state->metrics.CalculateAverage(state->metrics.ttft_samples);
        m.p50_time_to_first_token = state->metrics.CalculatePercentile(state->metrics.ttft_samples, 0.5f);
        m.p90_time_to_first_token = state->metrics.CalculatePercentile(state->metrics.ttft_samples, 0.9f);
        m.p99_time_to_first_token = state->metrics.CalculatePercentile(state->metrics.ttft_samples, 0.99f);

        // ITL metrics
        m.avg_inter_token_latency = state->metrics.CalculateAverage(state->metrics.itl_samples);
        m.p50_inter_token_latency = state->metrics.CalculatePercentile(state->metrics.itl_samples, 0.5f);
        m.p90_inter_token_latency = state->metrics.CalculatePercentile(state->metrics.itl_samples, 0.9f);
        m.p99_inter_token_latency = state->metrics.CalculatePercentile(state->metrics.itl_samples, 0.99f);

        // Queue wait time
        m.avg_queue_wait_time = state->metrics.CalculateAverage(state->metrics.queue_wait_samples);
        m.p99_queue_wait_time = state->metrics.CalculatePercentile(state->metrics.queue_wait_samples, 0.99f);
    }

    // KV Cache metrics
    ModelEntry* entry = state->GetDefaultModel();
    if (entry && entry->kv_cache && entry->kv_cache->block_manager) {
        m.kv_cache_total_blocks = entry->kv_cache->block_manager->num_blocks;
        int free_blocks = entry->kv_cache->block_manager->GetFreeBlockCount();
        m.kv_cache_usage_blocks = m.kv_cache_total_blocks - free_blocks;
        m.kv_cache_usage_percent = (float)m.kv_cache_usage_blocks / m.kv_cache_total_blocks * 100.0f;
    }

    // Batch metrics
    if (m.completed_requests > 0) {
        m.avg_batch_size = (float)m.total_requests / m.completed_requests;
    }

    // Error metrics
    m.oom_errors = state->metrics.oom_errors;
    m.timeout_errors = state->metrics.timeout_errors;

    return m;
}

void BeginEngineShutdown(DenseCoreHandle handle) {
    if (handle) {
        static_cast<EngineState*>(handle)->Shutdown();
    }
}

void FreeEngine(DenseCoreHandle handle) {
    if (handle) {
        EngineState* state = (EngineState*)handle;
        BeginEngineShutdown(handle);
        delete state;
    }
}

// Multi-Model API implementations

int LoadModel(DenseCoreHandle handle, const char* model_id, const char* model_path, int /*threads*/) {
    if (!handle || !model_id || !model_path) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "LoadModel: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)handle;
    std::string canonical_model_path;
    if (!ValidateModelPathArg(model_path, "LoadModel", "model_path", &canonical_model_path)) {
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    // Check if model already exists
    {
        std::lock_guard<std::mutex> lock(state->models_mu);
        if (state->models.find(model_id) != state->models.end()) {
            std::cerr << "[DenseCore] Model " << model_id << " already loaded" << std::endl;
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "LoadModel: model already loaded");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
    }

#ifdef LOG_INFO
    LOG_INFO("Loading model from: ", model_path);
#endif
    TransformerModel* model = LoadModelWithNuma(canonical_model_path.c_str(), state->numa_node_id);
    if (!model) {
#ifdef LOG_ERROR
        LOG_ERROR("Failed to load main model");
#else
        std::cerr << "[DenseCore] Failed to load model from " << model_path << std::endl;
#endif
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "LoadModel: failed to load model");
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    }

    // Initialize KV cache for this model
    KVCacheConfig kv_config = ComputeKVCacheConfig(model, state->requested_kv_cache_type);
    PagedKVCache* cache = InitPagedKVCache(model, kv_config.max_num_seqs, kv_config.max_seq_len,
                                           kv_config.effective_cache_type, state->numa_node_id);
    if (!cache) {
        delete model;
        std::cerr << "[DenseCore] Failed to initialize KV cache for " << model_id << std::endl;
        SetError(DENSECORE_STATUS_OUT_OF_MEMORY, "LoadModel: failed to initialize KV cache");
        return DENSECORE_STATUS_OUT_OF_MEMORY;
    }

    // Create model entry with unique_ptr ownership
    auto entry = MakeModelEntry(model_id, canonical_model_path, std::unique_ptr<TransformerModel>(model),
                                std::unique_ptr<PagedKVCache>(cache));

    // Add to pool
    {
        std::lock_guard<std::mutex> lock(state->models_mu);
        state->models[model_id] = std::move(entry);

        // Set as default if no default exists
        if (state->default_model_id.empty()) {
            state->default_model_id = model_id;
        }
    }

    ClearError();
    return 0;
}

int UnloadModel(DenseCoreHandle handle, const char* model_id) {
    if (!handle || !model_id) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "UnloadModel: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)handle;

    std::lock_guard<std::mutex> lock(state->models_mu);
    auto it = state->models.find(model_id);
    if (it == state->models.end()) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "UnloadModel: model not found");
        return DENSECORE_STATUS_INVALID_ARGUMENT;  // Not found
    }

    // Cannot unload default model if it's the only one
    if (state->default_model_id == model_id && state->models.size() == 1) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "UnloadModel: cannot unload last model");
        return DENSECORE_STATUS_INVALID_ARGUMENT;  // Cannot unload last model
    }

    state->models.erase(it);

    // Update default if needed
    if (state->default_model_id == model_id) {
        if (!state->models.empty()) {
            state->default_model_id = state->models.begin()->first;
        } else {
            state->default_model_id = "";
        }
    }

    ClearError();
    return 0;
}

int ListModels(DenseCoreHandle handle, char* out_models, int buffer_size) {
    if (!handle || !out_models || buffer_size <= 0) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "ListModels: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)handle;
    std::lock_guard<std::mutex> lock(state->models_mu);

    std::stringstream ss;
    for (auto it = state->models.begin(); it != state->models.end(); ++it) {
        if (it != state->models.begin()) ss << ",";
        ss << it->first;
    }

    std::string s = ss.str();
    if (s.length() >= (size_t)buffer_size) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "ListModels: output buffer too small");
        return DENSECORE_STATUS_INVALID_ARGUMENT;  // Buffer too small
    }

    strcpy(out_models, s.c_str());
    ClearError();
    return state->models.size();
}

int SetDefaultModel(DenseCoreHandle handle, const char* model_id) {
    if (!handle || !model_id) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SetDefaultModel: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)handle;
    std::lock_guard<std::mutex> lock(state->models_mu);

    if (state->models.find(model_id) == state->models.end()) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SetDefaultModel: model not found");
        return DENSECORE_STATUS_INVALID_ARGUMENT;  // Not found
    }

    state->default_model_id = model_id;
    ClearError();
    return 0;
}

// =============================================================================
// Universal Graph Execution API
// =============================================================================

namespace {

bool GraphInputByteSize(const DenseCoreTensorInput& input, size_t* byte_size, std::string* error) {
    if (!byte_size || !error) return false;
    if (!input.data) {
        *error = "input data is null";
        return false;
    }
    if (input.ndim <= 0 || input.ndim > DENSECORE_MAX_DIMS) {
        *error = "input ndim must be between 1 and DENSECORE_MAX_DIMS";
        return false;
    }

    size_t elements = 1;
    for (int dim = 0; dim < input.ndim; ++dim) {
        if (input.shape[dim] <= 0) {
            *error = "input shape dimensions must be positive";
            return false;
        }
        const uint64_t extent_u64 = static_cast<uint64_t>(input.shape[dim]);
        if (extent_u64 > std::numeric_limits<size_t>::max()) {
            *error = "input dimension exceeds addressable size";
            return false;
        }
        const size_t extent = static_cast<size_t>(extent_u64);
        if (elements > std::numeric_limits<size_t>::max() / extent) {
            *error = "input element count overflow";
            return false;
        }
        elements *= extent;
    }

    size_t bytes_per_element = 0;
    switch (input.dtype) {
    case DENSECORE_DTYPE_F32:
    case DENSECORE_DTYPE_INT32: bytes_per_element = 4; break;
    case DENSECORE_DTYPE_F16:
    case DENSECORE_DTYPE_BF16: bytes_per_element = 2; break;
    case DENSECORE_DTYPE_INT8: bytes_per_element = 1; break;
    case DENSECORE_DTYPE_INT4: *byte_size = elements / 2 + elements % 2; return true;
    default: *error = "unsupported input dtype"; return false;
    }

    if (elements > std::numeric_limits<size_t>::max() / bytes_per_element) {
        *error = "input byte size overflow";
        return false;
    }
    *byte_size = elements * bytes_per_element;
    return true;
}

}  // namespace

/**
 * @brief Submit a graph execution request with tensor inputs (Non-blocking)
 */
DENSECORE_API int SubmitGraphRequest(DenseCoreHandle handle, const DenseCoreTensorInput* inputs, int num_inputs,
                                     const char* graph_name, GraphResultCallback callback, void* user_data) {
    return RequestApiBoundary("SubmitGraphRequest", [&]() -> int {
        if (!handle) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitGraphRequest: handle is null");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        if (!inputs || num_inputs <= 0) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitGraphRequest: invalid inputs");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        if (!graph_name) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitGraphRequest: graph_name is null");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }

        EngineState* state = (EngineState*)handle;
        Request* req = AcquireAndInitRequest(state);
        RequestGuard guard(state, req);

        req->is_graph_execution = true;
        req->graph_name = graph_name;
        req->graph_callback = callback;
        req->user_data = user_data;

        try {
            req->graph_inputs.reserve(static_cast<size_t>(num_inputs));
            for (int i = 0; i < num_inputs; ++i) {
                size_t byte_size = 0;
                std::string validation_error;
                if (!GraphInputByteSize(inputs[i], &byte_size, &validation_error)) {
                    SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                             "SubmitGraphRequest: input " + std::to_string(i) + ": " + validation_error);
                    return DENSECORE_STATUS_INVALID_ARGUMENT;
                }

                OwnedGraphInput owned;
                owned.name = inputs[i].name ? inputs[i].name : "";
                owned.ndim = inputs[i].ndim;
                owned.dtype = inputs[i].dtype;
                std::copy_n(inputs[i].shape, owned.ndim, owned.shape.begin());
                owned.bytes.resize(byte_size);
                std::memcpy(owned.bytes.data(), inputs[i].data, byte_size);
                req->graph_inputs.push_back(std::move(owned));
            }
        } catch (const std::bad_alloc&) {
            SetError(DENSECORE_STATUS_OUT_OF_MEMORY, "SubmitGraphRequest: failed to copy input data");
            return DENSECORE_STATUS_OUT_OF_MEMORY;
        } catch (const std::exception& e) {
            SetError(DENSECORE_STATUS_INTERNAL_ERROR,
                     std::string("SubmitGraphRequest: input copy failed: ") + e.what());
            return DENSECORE_STATUS_INTERNAL_ERROR;
        }

        // Graph requests get high priority by default
        req->tier = "premium";
        req->priority = 50;

        return PublishPreparedRequest(state, req, guard, "SubmitGraphRequest: engine is draining");
    });
}

namespace {
// Context for synchronous execution
struct GraphSyncContext {
    DenseCoreTensorOutput* user_outputs;
    int num_user_outputs;
    int status = DENSECORE_STATUS_OK;
    std::promise<void> promise;
};

// Callback for synchronous execution
void GraphSyncCallback(const DenseCoreTensorOutput* outputs, int num_outputs, void* user_data) {
    GraphSyncContext* ctx = (GraphSyncContext*)user_data;
    if (!ctx) return;

    try {
        if (num_outputs < 0 || (!outputs && num_outputs > 0)) {
            ctx->status = num_outputs < 0 ? num_outputs : DENSECORE_STATUS_INTERNAL_ERROR;
            ctx->promise.set_value();
            return;
        }
        int count = std::min(ctx->num_user_outputs, num_outputs);
        for (int i = 0; i < count; ++i) {
            const auto& src = outputs[i];
            auto& dst = ctx->user_outputs[i];

            // Copy metadata
            dst.ndim = src.ndim;
            std::memcpy(dst.shape, src.shape, sizeof(dst.shape));
            dst.dtype = src.dtype;

            // Copy data if destination buffer is provided
            if (dst.data && src.data) {
                // Calculate size
                size_t size = 1;
                for (int d = 0; d < src.ndim; ++d) size *= src.shape[d];

                size_t bytes_per_element = 4;  // Default F32
                switch (src.dtype) {
                case DENSECORE_DTYPE_F16:
                case DENSECORE_DTYPE_BF16: bytes_per_element = 2; break;
                case DENSECORE_DTYPE_INT8: bytes_per_element = 1; break;
                case DENSECORE_DTYPE_INT32: bytes_per_element = 4; break;
                case DENSECORE_DTYPE_INT4:
                    bytes_per_element = 0;
                    break;  // Packed, handled differently? Assuming byte-aligned for now
                default: break;
                }
                // INT4 packed: size / 2? Let's assume standard types for now.
                size_t total_bytes = size * bytes_per_element;
                if (src.dtype == DENSECORE_DTYPE_INT4) total_bytes = (size + 1) / 2;

                std::memcpy(dst.data, src.data, total_bytes);
            }
        }
        ctx->promise.set_value();
    } catch (...) {
        try {
            ctx->promise.set_exception(std::current_exception());
        } catch (...) {
            // The request completion contract is exactly once; contain broken-promise callbacks.
        }
    }
}
}  // namespace

/**
 * @brief Execute graph synchronously and get output tensors directly
 */
DENSECORE_API int ExecuteGraphSync(DenseCoreHandle handle, const DenseCoreTensorInput* inputs, int num_inputs,
                                   const char* graph_name, DenseCoreTensorOutput* outputs, int num_outputs) {
    if (!handle || !inputs || num_inputs <= 0 || !graph_name || !outputs || num_outputs <= 0) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "ExecuteGraphSync: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    GraphSyncContext ctx;
    ctx.user_outputs = outputs;
    ctx.num_user_outputs = num_outputs;
    auto future = ctx.promise.get_future();

    int req_id = SubmitGraphRequest(handle, inputs, num_inputs, graph_name, GraphSyncCallback, &ctx);
    if (req_id < 0) {
        return req_id;  // Error from SubmitGraphRequest
    }

    // Wait for completion
    try {
        future.get();
        if (ctx.status < 0) {
            SetError(static_cast<DenseCoreStatus>(ctx.status),
                     "ExecuteGraphSync: graph execution failed with status " + std::to_string(ctx.status));
            return ctx.status;
        }
        ClearError();
        return DENSECORE_STATUS_OK;
    } catch (const std::exception& e) {
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, std::string("ExecuteGraphSync error: ") + e.what());
        return DENSECORE_STATUS_INTERNAL_ERROR;
    }
}
