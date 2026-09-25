#include "densecore/llm/weights/prepared_weights.h"

#include "llm/weights/internal.h"
#include <algorithm>
#include <cstring>
#include <ggml-cpu.h>
#include <utility>
#if defined(__linux__)
#include <malloc.h>
#endif

#include "densecore/models/model_types.h"
#include "llm/config/runtime_config.h"

// Compatibility entry point used by existing graph tests and callers.
bool PrepareQwen36SSMQ8PrefillAMXAliasesForExecution(TransformerModel* model);

namespace densecore::llm::weights {

CpuRepackBufferTypes FindCpuRepackBufferTypes(ggml_backend_t backend) {
    CpuRepackBufferTypes result{};
    ggml_backend_dev_t device =
        backend ? ggml_backend_get_device(backend) : ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    if (!device) {
        return result;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    auto* proc = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
    if (!proc) {
        return result;
    }
    ggml_backend_buffer_type_t* bufts = proc(device);
    if (!bufts) {
        return result;
    }
    for (size_t i = 0; bufts[i] != nullptr; ++i) {
        const char* name = ggml_backend_buft_name(bufts[i]);
        if (name && std::strcmp(name, "AMX") == 0) {
            result.cpu_amx = bufts[i];
        }
        if (name && std::strcmp(name, "CPU_KLEIDIAI") == 0) {
            result.cpu_kleidiai = bufts[i];
        }
        if (name && std::strcmp(name, "CPU_REPACK") == 0) {
            result.cpu_repack = bufts[i];
        }
    }
    return result;
}


PreparedWeights::~PreparedWeights() {
    Reset();
}

PreparedWeights::PreparedWeights(PreparedWeights&& other) noexcept {
    Swap(other);
}

PreparedWeights& PreparedWeights::operator=(PreparedWeights&& other) noexcept {
    if (this != &other) {
        Reset();
        Swap(other);
    }
    return *this;
}

void PreparedWeights::Swap(PreparedWeights& other) noexcept {
    using std::swap;
    swap(host_buffers, other.host_buffers);
    swap(ctx_cpu_repack, other.ctx_cpu_repack);
    swap(ctx_cpu_amx, other.ctx_cpu_amx);
    swap(ctx_cpu_kleidiai, other.ctx_cpu_kleidiai);
    swap(cpu_repack_buffers, other.cpu_repack_buffers);
    swap(cpu_repack_aliases, other.cpu_repack_aliases);
    swap(cpu_repack_alias_sources, other.cpu_repack_alias_sources);
    swap(cpu_repack_alias_layouts, other.cpu_repack_alias_layouts);
    swap(cpu_decode_repack_aliases, other.cpu_decode_repack_aliases);
    swap(q5k_8x8_repacked_tensors, other.q5k_8x8_repacked_tensors);
    swap(ctx_qwen36_ssm_q8_prefill_amx, other.ctx_qwen36_ssm_q8_prefill_amx);
    swap(qwen36_ssm_q8_prefill_amx_buffers, other.qwen36_ssm_q8_prefill_amx_buffers);
    swap(qwen36_ssm_q8_prefill_amx_aliases, other.qwen36_ssm_q8_prefill_amx_aliases);
    swap(cpu_amx_aliases, other.cpu_amx_aliases);
}

void PreparedWeights::ClearPrefillAliases() noexcept {
    const bool had_state = !qwen36_ssm_q8_prefill_amx_aliases.empty() || !qwen36_ssm_q8_prefill_amx_buffers.empty() ||
                           ctx_qwen36_ssm_q8_prefill_amx;
    if (!had_state) return;
    for (const auto& alias : qwen36_ssm_q8_prefill_amx_aliases) {
        cpu_amx_aliases.erase(alias.second);
    }
    qwen36_ssm_q8_prefill_amx_aliases.clear();
    for (auto* buffer : qwen36_ssm_q8_prefill_amx_buffers) {
        if (buffer) ggml_backend_buffer_free(buffer);
    }
    qwen36_ssm_q8_prefill_amx_buffers.clear();
    if (ctx_qwen36_ssm_q8_prefill_amx) ggml_free(ctx_qwen36_ssm_q8_prefill_amx);
    ctx_qwen36_ssm_q8_prefill_amx = nullptr;
#if defined(__linux__)
    // Preserve prompt-to-decode memory release for potentially large Q8 aliases.
    malloc_trim(0);
#endif
}

void PreparedWeights::Reset() noexcept {
    // Full teardown keeps the existing order and does not request malloc_trim.
    for (auto* buffer : cpu_repack_buffers) {
        if (buffer) ggml_backend_buffer_free(buffer);
    }
    cpu_repack_buffers.clear();
    for (auto* buffer : qwen36_ssm_q8_prefill_amx_buffers) {
        if (buffer) ggml_backend_buffer_free(buffer);
    }
    qwen36_ssm_q8_prefill_amx_buffers.clear();
    if (ctx_cpu_kleidiai) ggml_free(ctx_cpu_kleidiai);
    if (ctx_cpu_amx) ggml_free(ctx_cpu_amx);
    if (ctx_qwen36_ssm_q8_prefill_amx) ggml_free(ctx_qwen36_ssm_q8_prefill_amx);
    if (ctx_cpu_repack) ggml_free(ctx_cpu_repack);
    ctx_cpu_kleidiai = nullptr;
    ctx_cpu_amx = nullptr;
    ctx_qwen36_ssm_q8_prefill_amx = nullptr;
    ctx_cpu_repack = nullptr;
    cpu_repack_aliases.clear();
    cpu_repack_alias_sources.clear();
    cpu_repack_alias_layouts.clear();
    cpu_decode_repack_aliases.clear();
    q5k_8x8_repacked_tensors.clear();
    qwen36_ssm_q8_prefill_amx_aliases.clear();
    cpu_amx_aliases.clear();
    // Fused alias wrappers do not own their backing host allocations.
    for (auto& buffer : host_buffers) {
        if (buffer.ptr && buffer.size > 0) {
            densecore::NumaAllocator::Free(buffer.ptr, buffer.size, buffer.type);
        }
    }
    host_buffers.clear();
}

PreparedWeightResult PreparedWeights::PrepareForExecution(TransformerModel& model, WeightExecutionPhase phase,
                                                          int prompt_token_count,
                                                          const densecore::llm::config::FastPathRuntimeConfig& config) {
    PreparedWeightResult result;
    if (phase == WeightExecutionPhase::Decode) {
        ClearPrefillAliases();
        return result;
    }
    const bool hybrid_ssm = (model.variant == ModelVariant::QWEN35 || model.variant == ModelVariant::QWEN36) &&
                            model.arch_flags.is_hybrid_ssm;
    if (phase != WeightExecutionPhase::Prefill || !hybrid_ssm ||
        config.qwen36_ssm_q8_prefill_amx == densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::Off ||
        prompt_token_count < config.qwen36_ssm_q8_prefill_amx_min_tokens) {
        return result;
    }
    result.prefill_aliases_prepared = PrepareQwen36SSMQ8PrefillAMXAliasesForExecution(&model);
    result.prefill_alias_mode = static_cast<int>(config.qwen36_ssm_q8_prefill_amx);
    return result;
}

}  // namespace densecore::llm::weights

namespace {
using densecore::llm::weights::CpuRepackBufferTypes;
using densecore::llm::weights::FindCpuRepackBufferTypes;

bool IsQwenHybridSSMQ8ProjectionTensor(const ggml_tensor* source) {
    if (!source || source->type != GGML_TYPE_Q8_0 || !source->name[0]) {
        return false;
    }
    const char* name = source->name;
    return std::strstr(name, "attn_qkv.weight") || std::strstr(name, "attn_gate.weight") ||
           std::strstr(name, "ssm_out.weight");
}

void ClearQwen36SSMQ8PrefillAMXAliasesImpl(TransformerModel* model) {
    if (!model) {
        return;
    }
    model->prepared_weights.ClearPrefillAliases();
}

bool PrepareQwen36SSMQ8PrefillAMXAliasesForExecutionImpl(TransformerModel* model) {
#if defined(__aarch64__) || defined(_M_ARM64)
    (void)model;
    return false;
#else
    if (!model || (model->variant != ModelVariant::QWEN35 && model->variant != ModelVariant::QWEN36) ||
        !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    if (!model->prepared_weights.qwen36_ssm_q8_prefill_amx_aliases.empty() &&
        !model->prepared_weights.qwen36_ssm_q8_prefill_amx_buffers.empty()) {
        return true;
    }
    ClearQwen36SSMQ8PrefillAMXAliasesImpl(model);

    const CpuRepackBufferTypes repack_bufts =
        FindCpuRepackBufferTypes(model->cpu_backend ? model->cpu_backend : model->backend);
    if (!repack_bufts.cpu_amx) {
        return false;
    }

    const size_t tensor_slots = static_cast<size_t>(std::max<uint32_t>(1, model->hparams.n_layer)) * 3 + 16;
    ggml_init_params params{
        /*.mem_size   =*/tensor_slots * ggml_tensor_overhead() + 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    model->prepared_weights.ctx_qwen36_ssm_q8_prefill_amx = ggml_init(params);
    if (!model->prepared_weights.ctx_qwen36_ssm_q8_prefill_amx) {
        return false;
    }

    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> pending;
    pending.reserve(static_cast<size_t>(model->hparams.n_layer) * 3);
    auto add_alias = [&](ggml_tensor* source) {
        if (!source || !source->data || source->view_src || source->ne[0] <= 0 || source->ne[1] <= 0 ||
            source->ne[2] != 1 || source->ne[3] != 1 || !IsQwenHybridSSMQ8ProjectionTensor(source)) {
            return;
        }
        ggml_tensor* alias = ggml_new_tensor_2d(model->prepared_weights.ctx_qwen36_ssm_q8_prefill_amx, source->type,
                                                source->ne[0], source->ne[1]);
        if (!alias) {
            return;
        }
        const std::string name = std::string(source->name[0] ? source->name : "weight") + ".prefill_amx_scoped";
        ggml_set_name(alias, name.c_str());
        model->prepared_weights.qwen36_ssm_q8_prefill_amx_aliases[source] = alias;
        model->prepared_weights.cpu_amx_aliases[alias] = true;
        pending.emplace_back(source, alias);
    };

    for (auto& layer : model->layers) {
        add_alias(layer.Get(model_keys::kAttnQkvWeight));
        add_alias(layer.Get(model_keys::kAttnGate));
        add_alias(layer.Get(model_keys::kSSMOut));
    }
    if (pending.empty()) {
        ClearQwen36SSMQ8PrefillAMXAliasesImpl(model);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        model->prepared_weights.ctx_qwen36_ssm_q8_prefill_amx, repack_bufts.cpu_amx);
    if (!buffer) {
        ClearQwen36SSMQ8PrefillAMXAliasesImpl(model);
        return false;
    }
    model->prepared_weights.qwen36_ssm_q8_prefill_amx_buffers.push_back(buffer);

    bool ok = true;
    for (const auto& item : pending) {
        if (!item.first || !item.second || !item.second->extra) {
            ok = false;
            break;
        }
        ggml_backend_tensor_set(item.second, item.first->data, 0, ggml_nbytes(item.first));
    }
    if (!ok) {
        ClearQwen36SSMQ8PrefillAMXAliasesImpl(model);
        return false;
    }
    return true;
#endif
}

}  // namespace

void ClearQwen36SSMQ8PrefillAMXAliases(TransformerModel* model) {
    ClearQwen36SSMQ8PrefillAMXAliasesImpl(model);
}

bool PrepareQwen36SSMQ8PrefillAMXAliasesForExecution(TransformerModel* model) {
    return PrepareQwen36SSMQ8PrefillAMXAliasesForExecutionImpl(model);
}
