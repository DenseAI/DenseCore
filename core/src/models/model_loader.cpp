#include "densecore/models/model_loader.h"

#include <ggml-alloc.h>
#include <ggml-cpu.h>
#ifdef __APPLE__
#include <ggml-metal.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>

#include "densecore/backend/hardware_topology.h"
#include "densecore/memory/numa_allocator.h"
#include "densecore/models/model_descriptor.h"
#include "densecore/models/model_graph_bridge.h"  // Universal graph execution bridge
#include "densecore/models/tokenizer.h"
#include "densecore/runtime/inference.h"  // For InitRoPETable
#include "llm/config/runtime_config.h"

#if defined(__linux__)
#include <malloc.h>
#include <sys/mman.h>  // For mmap, MAP_HUGETLB
#endif

#include "densecore/backend/apple/apple_silicon.h"
#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/qwen35_ssm_math.h"
#include "densecore/runtime/dtype_utils.h"
#include "models/gemma4_packed_expert_layout.h"
#include "models/model_inference_policy.h"

namespace {
constexpr const char* kGemma4RouterScaleKey = "gemma4.router.scale";
constexpr const char* kGemma4RouterPerExpertScaleKey = "gemma4.router.per_expert_scale";
constexpr const char* kGemma4PreMoeNormKey = "gemma4.pre_feedforward_layernorm_2.weight";
constexpr const char* kGemma4PostSharedNormKey = "gemma4.post_feedforward_layernorm_1.weight";
constexpr const char* kGemma4PostMoeNormKey = "gemma4.post_feedforward_layernorm_2.weight";
constexpr const char* kGemma4PostFfnNormKey = "gemma4.post_feedforward_layernorm.weight";

bool ResolveGenericPackedProjectionBinding(ggml_context* vctx, const TransformerModel::Int4WeightBinding& root_binding,
                                           ggml_tensor* packed_view, int64_t cols, int64_t rows, int64_t row_start,
                                           int expert_axis, int expert_index,
                                           TransformerModel::Int4WeightBinding* out) {
    if (!out) {
        return false;
    }
    *out = TransformerModel::Int4WeightBinding{};
    if (!vctx || !packed_view || !packed_view->data || !root_binding.packed || !root_binding.scales ||
        !root_binding.zeros || !root_binding.packed->data || !root_binding.scales->data || !root_binding.zeros->data) {
        return false;
    }
    if (root_binding.group_size <= 0 || cols <= 0 || rows <= 0 || row_start < 0 || root_binding.k != cols ||
        row_start + rows > root_binding.n) {
        return false;
    }

    const int64_t groups_per_row = root_binding.k / root_binding.group_size;
    if (groups_per_row <= 0 || root_binding.scales->nb[1] == 0 || root_binding.zeros->nb[1] == 0) {
        return false;
    }

    size_t scales_offset = static_cast<size_t>(row_start) * static_cast<size_t>(root_binding.scales->nb[1]);
    size_t zeros_offset = static_cast<size_t>(row_start) * static_cast<size_t>(root_binding.zeros->nb[1]);
    if (expert_axis >= 0 && expert_index >= 0) {
        if (root_binding.scales->ne[expert_axis] <= expert_index ||
            root_binding.zeros->ne[expert_axis] <= expert_index || root_binding.scales->nb[expert_axis] == 0 ||
            root_binding.zeros->nb[expert_axis] == 0) {
            return false;
        }
        scales_offset += static_cast<size_t>(expert_index) * static_cast<size_t>(root_binding.scales->nb[expert_axis]);
        zeros_offset += static_cast<size_t>(expert_index) * static_cast<size_t>(root_binding.zeros->nb[expert_axis]);
    }

    ggml_tensor* scales_view = ggml_view_2d(vctx, const_cast<ggml_tensor*>(root_binding.scales), groups_per_row, rows,
                                            root_binding.scales->nb[1], scales_offset);
    ggml_tensor* zeros_view = ggml_view_2d(vctx, const_cast<ggml_tensor*>(root_binding.zeros), groups_per_row, rows,
                                           root_binding.zeros->nb[1], zeros_offset);
    if (!scales_view || !zeros_view) {
        return false;
    }

    out->packed = packed_view;
    out->scales = scales_view;
    out->zeros = zeros_view;
    out->group_size = root_binding.group_size;
    out->k = cols;
    out->n = rows;
    return true;
}

bool FindInt4BindingInViewChain(const TransformerModel* model, const ggml_tensor* tensor,
                                const ggml_tensor** root_tensor_out,
                                TransformerModel::Int4WeightBinding* root_binding_out, size_t* total_view_offs_out) {
    if (!model || !tensor || !root_tensor_out || !root_binding_out || !total_view_offs_out) {
        return false;
    }
    const ggml_tensor* current = tensor;
    size_t total_view_offs = 0;
    while (current) {
        const auto it = model->int4_weight_bindings.find(current);
        if (it != model->int4_weight_bindings.end()) {
            *root_tensor_out = current;
            *root_binding_out = it->second;
            *total_view_offs_out = total_view_offs;
            return true;
        }
        total_view_offs += current->view_offs;
        current = current->view_src;
    }
    return false;
}

bool ResolveStackedInt4ViewBinding(ggml_context* vctx, const ggml_tensor* root_tensor,
                                   const TransformerModel::Int4WeightBinding& root_binding, ggml_tensor* packed_view,
                                   size_t total_view_offs, TransformerModel::Int4WeightBinding* out) {
    if (!out) {
        return false;
    }
    *out = TransformerModel::Int4WeightBinding{};
    if (!vctx || !root_tensor || !packed_view || !packed_view->data || !root_binding.packed || !root_binding.scales ||
        !root_binding.zeros || !root_binding.packed->data || !root_binding.scales->data || !root_binding.zeros->data) {
        return false;
    }

    const int64_t expected_k = packed_view->ne[0];
    const int64_t expected_n = packed_view->ne[1];
    if (root_binding.group_size <= 0 || expected_k <= 0 || expected_n <= 0 || root_binding.k != expected_k ||
        root_binding.n < expected_n) {
        return false;
    }

    const size_t row_stride_bytes = static_cast<size_t>(root_tensor->nb[1]);
    if (row_stride_bytes == 0) {
        return false;
    }
    const size_t plane_stride_bytes =
        (root_tensor->ne[2] > 1 && root_tensor->nb[2] > 0) ? static_cast<size_t>(root_tensor->nb[2]) : 0;

    size_t expert_idx = 0;
    size_t row_offs_bytes = total_view_offs;
    if (plane_stride_bytes > 0) {
        expert_idx = total_view_offs / plane_stride_bytes;
        row_offs_bytes = total_view_offs % plane_stride_bytes;
    }
    if ((row_offs_bytes % row_stride_bytes) != 0) {
        return false;
    }

    const size_t row_start = row_offs_bytes / row_stride_bytes;
    if (row_start + static_cast<size_t>(expected_n) > static_cast<size_t>(root_binding.n)) {
        return false;
    }

    const int64_t groups_per_row = root_binding.k / root_binding.group_size;
    if (groups_per_row <= 0 || root_binding.scales->nb[1] == 0 || root_binding.zeros->nb[1] == 0) {
        return false;
    }

    size_t scales_off = row_start * static_cast<size_t>(root_binding.scales->nb[1]);
    size_t zeros_off = row_start * static_cast<size_t>(root_binding.zeros->nb[1]);
    if (expert_idx > 0) {
        if (root_binding.scales->ne[2] <= static_cast<int64_t>(expert_idx) ||
            root_binding.zeros->ne[2] <= static_cast<int64_t>(expert_idx) || root_binding.scales->nb[2] == 0 ||
            root_binding.zeros->nb[2] == 0) {
            return false;
        }
        scales_off += expert_idx * static_cast<size_t>(root_binding.scales->nb[2]);
        zeros_off += expert_idx * static_cast<size_t>(root_binding.zeros->nb[2]);
    }

    ggml_tensor* scales_view = ggml_view_2d(vctx, const_cast<ggml_tensor*>(root_binding.scales), groups_per_row,
                                            expected_n, root_binding.scales->nb[1], scales_off);
    ggml_tensor* zeros_view = ggml_view_2d(vctx, const_cast<ggml_tensor*>(root_binding.zeros), groups_per_row,
                                           expected_n, root_binding.zeros->nb[1], zeros_off);
    if (!scales_view || !zeros_view) {
        return false;
    }

    out->packed = packed_view;
    out->scales = scales_view;
    out->zeros = zeros_view;
    out->group_size = root_binding.group_size;
    out->k = expected_k;
    out->n = expected_n;
    return true;
}

bool EnvFlagEnabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_value;
    }
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (text == "0" || text == "false" || text == "off" || text == "no") {
        return false;
    }
    if (text == "1" || text == "true" || text == "on" || text == "yes" || text == "force") {
        return true;
    }
    return default_value;
}

constexpr size_t kGemma4RequantSafetyHeadroomBytes = size_t{8} * 1024 * 1024 * 1024;
constexpr size_t kGemma4GateUpRequantSafetyHeadroomBytes = size_t{24} * 1024 * 1024 * 1024;

size_t ReadMemAvailableBytes() {
#if defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    uint64_t value_kib = 0;
    std::string unit;
    while (meminfo >> key >> value_kib >> unit) {
        if (key == "MemAvailable:") {
            constexpr uint64_t kBytesPerKiB = 1024;
            if (value_kib > std::numeric_limits<size_t>::max() / kBytesPerKiB) {
                return std::numeric_limits<size_t>::max();
            }
            return static_cast<size_t>(value_kib * kBytesPerKiB);
        }
    }
#endif
    return 0;
}

size_t Requantized3DQ8Bytes(int64_t ne0, int64_t ne1, int64_t ne2) {
    if (ne0 <= 0 || ne1 <= 0 || ne2 <= 0) {
        return 0;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, ne0);
    const uint64_t rows = static_cast<uint64_t>(ne1) * static_cast<uint64_t>(ne2);
    if (row_bytes == 0 || rows == 0 || rows > std::numeric_limits<size_t>::max() / static_cast<uint64_t>(row_bytes)) {
        return 0;
    }
    return row_bytes * static_cast<size_t>(rows);
}

size_t EstimateGemma4Q5DownRequantBytes(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_gemma4) {
        return 0;
    }
    size_t total = 0;
    for (const auto& layer : model->layers) {
        ggml_tensor* gate_up =
            layer.Get("ffn_gate_up_exps.weight") ? layer.Get("ffn_gate_up_exps.weight") : layer.Get("ffn_gate_up_exps");
        ggml_tensor* down =
            layer.Get("ffn_down_exps.weight") ? layer.Get("ffn_down_exps.weight") : layer.Get("ffn_down_exps");
        if (!gate_up || !down || !down->data || down->type != GGML_TYPE_Q5_1 || down->view_src) {
            continue;
        }
        densecore::gemma4::PackedExpertLayout layout{};
        std::string reason;
        if (!densecore::gemma4::InferPackedExpertLayout(gate_up, down, &layout, &reason)) {
            continue;
        }
        const size_t bytes = Requantized3DQ8Bytes(layout.intermediate_dim, layout.hidden_dim, layout.num_experts);
        if (bytes == 0) {
            continue;
        }
        if (total <= std::numeric_limits<size_t>::max() - bytes) {
            total += bytes;
        }
    }
    return total;
}

size_t EstimateGemma4Q4GateUpRequantBytes(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_gemma4) {
        return 0;
    }
    size_t total = 0;
    for (const auto& layer : model->layers) {
        ggml_tensor* gate_up =
            layer.Get("ffn_gate_up_exps.weight") ? layer.Get("ffn_gate_up_exps.weight") : layer.Get("ffn_gate_up_exps");
        ggml_tensor* down =
            layer.Get("ffn_down_exps.weight") ? layer.Get("ffn_down_exps.weight") : layer.Get("ffn_down_exps");
        if (!gate_up || !down || !gate_up->data || gate_up->type != GGML_TYPE_Q4_K || gate_up->view_src) {
            continue;
        }
        densecore::gemma4::PackedExpertLayout layout{};
        std::string reason;
        if (!densecore::gemma4::InferPackedExpertLayout(gate_up, down, &layout, &reason)) {
            continue;
        }
        const size_t bytes = Requantized3DQ8Bytes(layout.hidden_dim, layout.intermediate_dim * 2, layout.num_experts);
        if (bytes == 0) {
            continue;
        }
        if (total <= std::numeric_limits<size_t>::max() - bytes) {
            total += bytes;
        }
    }
    return total;
}

struct CpuRepackBufferTypes {
    ggml_backend_buffer_type_t cpu_amx = nullptr;
    ggml_backend_buffer_type_t cpu_repack = nullptr;
    ggml_backend_buffer_type_t cpu_kleidiai = nullptr;
};

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

bool CanUseCpuKleidiaiRepack(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0;
}

bool IsQwen36SSMQ8ProjectionTensor(const ggml_tensor* source) {
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
    const bool has_scoped_alias_state = !model->qwen36_ssm_q8_prefill_amx_aliases.empty() ||
                                        !model->qwen36_ssm_q8_prefill_amx_buffers.empty() ||
                                        model->ctx_qwen36_ssm_q8_prefill_amx != nullptr;
    if (!has_scoped_alias_state) {
        return;
    }
    for (const auto& kv : model->qwen36_ssm_q8_prefill_amx_aliases) {
        if (kv.second) {
            model->cpu_amx_aliases.erase(kv.second);
        }
    }
    model->qwen36_ssm_q8_prefill_amx_aliases.clear();
    for (auto* buffer : model->qwen36_ssm_q8_prefill_amx_buffers) {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
    model->qwen36_ssm_q8_prefill_amx_buffers.clear();
    if (model->ctx_qwen36_ssm_q8_prefill_amx) {
        ggml_free(model->ctx_qwen36_ssm_q8_prefill_amx);
        model->ctx_qwen36_ssm_q8_prefill_amx = nullptr;
    }
#if defined(__linux__)
    // Scoped prefill AMX aliases can temporarily allocate large Q8 buffers.
    // Return freed pages promptly before decode continues on canonical Q8_0.
    malloc_trim(0);
#endif
}

bool PrepareQwen36SSMQ8PrefillAMXAliasesForExecutionImpl(TransformerModel* model) {
#if defined(__aarch64__) || defined(_M_ARM64)
    (void)model;
    return false;
#else
    if (!model || model->variant != ModelVariant::QWEN36 || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    if (!model->qwen36_ssm_q8_prefill_amx_aliases.empty() && !model->qwen36_ssm_q8_prefill_amx_buffers.empty()) {
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
    model->ctx_qwen36_ssm_q8_prefill_amx = ggml_init(params);
    if (!model->ctx_qwen36_ssm_q8_prefill_amx) {
        return false;
    }

    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> pending;
    pending.reserve(static_cast<size_t>(model->hparams.n_layer) * 3);
    auto add_alias = [&](ggml_tensor* source) {
        if (!source || !source->data || source->view_src || source->ne[0] <= 0 || source->ne[1] <= 0 ||
            source->ne[2] != 1 || source->ne[3] != 1 || !IsQwen36SSMQ8ProjectionTensor(source)) {
            return;
        }
        ggml_tensor* alias =
            ggml_new_tensor_2d(model->ctx_qwen36_ssm_q8_prefill_amx, source->type, source->ne[0], source->ne[1]);
        if (!alias) {
            return;
        }
        const std::string name = std::string(source->name[0] ? source->name : "weight") + ".prefill_amx_scoped";
        ggml_set_name(alias, name.c_str());
        model->qwen36_ssm_q8_prefill_amx_aliases[source] = alias;
        model->cpu_amx_aliases[alias] = true;
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

    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(model->ctx_qwen36_ssm_q8_prefill_amx, repack_bufts.cpu_amx);
    if (!buffer) {
        ClearQwen36SSMQ8PrefillAMXAliasesImpl(model);
        return false;
    }
    model->qwen36_ssm_q8_prefill_amx_buffers.push_back(buffer);

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

void PrepareGenericCpuFastMatmulAliases(TransformerModel* model) {
    if (!model || model->arch_flags.is_gemma4 || model->layers.empty()) {
        return;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    if (model->arch_flags.is_hybrid_ssm) {
        return;
    }
#endif

    const CpuRepackBufferTypes repack_bufts =
        FindCpuRepackBufferTypes(model->cpu_backend ? model->cpu_backend : model->backend);
    if (!repack_bufts.cpu_repack) {
        return;
    }

    const size_t tensor_slots = static_cast<size_t>(std::max<uint32_t>(1, model->hparams.n_layer)) * 16 + 128;
    auto init_alias_context = [&](ggml_context** ctx, const char* name) -> bool {
        if (*ctx) {
            return true;
        }
        struct ggml_init_params params = {
            /*.mem_size   =*/tensor_slots * ggml_tensor_overhead() + 1024 * 1024,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        *ctx = ggml_init(params);
        if (!*ctx) {
            std::cerr << "[DenseCore] Warning: failed to allocate " << name << " alias metadata context" << std::endl;
            return false;
        }
        return true;
    };

    struct PendingAlias {
        ggml_tensor* source = nullptr;
        ggml_tensor* alias = nullptr;
        bool decode_only = false;
    };
    struct PendingFusedAlias {
        TransformerLayer* layer = nullptr;
        ggml_tensor* alias = nullptr;
        std::string key;
        std::vector<uint8_t> bytes;
        ggml_tensor* first = nullptr;
        ggml_tensor* second = nullptr;
        bool expert_pair = false;
    };

    std::vector<PendingAlias> pending_cpu_repack;
    pending_cpu_repack.reserve(model->layers.size() * 8);
    std::vector<PendingFusedAlias> pending_fused_cpu_repack;
    auto is_qwen36_ssm_q8_projection = [&](const ggml_tensor* source) -> bool {
        if (!source || source->type != GGML_TYPE_Q8_0 || !source->name[0]) {
            return false;
        }
        const char* name = source->name;
        return std::strstr(name, "attn_qkv.weight") || std::strstr(name, "attn_gate.weight") ||
               std::strstr(name, "ssm_out.weight");
    };
    auto model_has_qwen36_ssm_q8_projection = [&]() -> bool {
        if (model->variant != ModelVariant::QWEN36 || !model->arch_flags.is_hybrid_ssm) {
            return false;
        }
        for (const auto& layer : model->layers) {
            for (const char* key : {model_keys::kAttnQkvWeight, model_keys::kAttnGate, model_keys::kSSMOut}) {
                if (is_qwen36_ssm_q8_projection(layer.Get(key))) {
                    return true;
                }
            }
        }
        return false;
    };

    std::vector<PendingAlias> pending_cpu_amx;
    pending_cpu_amx.reserve(model->layers.size() * 8);
    std::vector<PendingFusedAlias> pending_fused_cpu_amx;
    const auto loader_config = densecore::llm::config::LoadFastPathRuntimeConfig();
    const bool qwen36_has_ssm_q8_projection = model_has_qwen36_ssm_q8_projection();
    const bool qwen36_ssm_q8_amx_alias_enabled =
        model->variant == ModelVariant::QWEN36 && qwen36_has_ssm_q8_projection && repack_bufts.cpu_amx &&
        loader_config.qwen36_ssm_q8_amx_alias != densecore::env::RuntimeToggleMode::Off;
    const bool qwen36_ssm_q8_prefill_amx_requested =
        model->variant == ModelVariant::QWEN36 && qwen36_has_ssm_q8_projection && repack_bufts.cpu_amx &&
        loader_config.qwen36_ssm_q8_prefill_amx == densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::On;
    // C4 validation showed model-owned AMX aliases for the Qwen3.6 SSM Q8_0
    // projections move long-prefill from ~101 to ~115 tok/s but also push
    // single-token decode from ~25 to ~16 tok/s. Keep this fail-closed until
    // the AMX storage is request/prefill-local or otherwise proven not to
    // perturb decode residency/locality.
    const bool qwen36_ssm_q8_prefill_amx_enabled = false;
    const bool qwen36_expert_cpu_repack_enabled =
        model->variant != ModelVariant::QWEN36 ||
        loader_config.qwen36_expert_cpu_repack == densecore::env::RuntimeToggleMode::On ||
        (loader_config.qwen36_expert_cpu_repack == densecore::env::RuntimeToggleMode::Auto &&
         !qwen36_ssm_q8_amx_alias_enabled);
    if (model->variant == ModelVariant::QWEN36 && model->arch_flags.is_hybrid_ssm) {
        std::cout << "[DenseCore] Qwen3.6 fast-matmul alias policy: ssm_q8_amx="
                  << (qwen36_ssm_q8_amx_alias_enabled ? "enabled" : "disabled")
                  << ", ssm_q8_prefill_loader_alias=" << (qwen36_ssm_q8_prefill_amx_enabled ? "enabled" : "disabled")
                  << ", ssm_q8_prefill_scoped_amx=" << (qwen36_ssm_q8_prefill_amx_requested ? "requested" : "disabled")
                  << ", expert_cpu_repack=" << (qwen36_expert_cpu_repack_enabled ? "enabled" : "disabled")
                  << ", ssm_q8_projection=" << (qwen36_has_ssm_q8_projection ? "present" : "absent") << std::endl;
    }

    auto choose_alias_buffer = [&](const ggml_tensor* source, bool is_2d) -> ggml_backend_buffer_type_t {
        if (!source) {
            return nullptr;
        }
        if (model->variant == ModelVariant::QWEN36 && is_2d) {
            // The Qwen3.6 UD-Q4_K_M GGUF stores SSM qkv/gate/out projections
            // as Q8_0, so the Q4_K prefill probe is not applicable there. The
            // load-time AMX alias remains an explicit experiment only because
            // it is not phase-aware and can push single-token decode onto an
            // unfavorable layout. The maintained Q8 prefill path prepares a
            // scoped alias during prefill execution and clears it before decode.
            if (qwen36_ssm_q8_amx_alias_enabled && is_qwen36_ssm_q8_projection(source)) {
                return repack_bufts.cpu_amx;
            }
            return nullptr;
        }
#if !defined(__aarch64__) && !defined(_M_ARM64)
        if ((model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) && is_2d &&
            repack_bufts.cpu_amx && ggml_is_quantized(source->type)) {
            return repack_bufts.cpu_amx;
        }
#endif
        if (model->arch_flags.is_hybrid_ssm) {
            if (model->variant == ModelVariant::QWEN36 && !is_2d && !qwen36_expert_cpu_repack_enabled) {
                return nullptr;
            }
            if (!repack_bufts.cpu_repack || (!ggml_is_quantized(source->type) && source->type != GGML_TYPE_F16)) {
                return nullptr;
            }
            return repack_bufts.cpu_repack;
        }
        if (repack_bufts.cpu_repack && (ggml_is_quantized(source->type) || source->type == GGML_TYPE_F16)) {
            return repack_bufts.cpu_repack;
        }
        return nullptr;
    };

    auto make_alias = [&](ggml_tensor* source, const char* suffix) {
        if (!source || !source->data || source->view_src || source->ne[0] <= 0 || source->ne[1] <= 0 ||
            source->ne[3] != 1) {
            return;
        }
        const bool is_2d = source->ne[2] == 1;
        const bool is_3d_expert = model->arch_flags.is_hybrid_ssm && source->ne[2] > 1;
        if (!is_2d && !is_3d_expert) {
            return;
        }
        if (model->cpu_repack_aliases.find(source) != model->cpu_repack_aliases.end()) {
            return;
        }
        ggml_backend_buffer_type_t alias_buft = choose_alias_buffer(source, is_2d);
        if (!alias_buft) {
            return;
        }
        const bool use_amx = alias_buft == repack_bufts.cpu_amx;
        ggml_context** ctx = use_amx ? &model->ctx_cpu_amx : &model->ctx_cpu_repack;
        const char* buft_name = use_amx ? "AMX" : "CPU_REPACK";
        if (!init_alias_context(ctx, buft_name)) {
            return;
        }
        ggml_tensor* alias = is_2d
                                 ? ggml_new_tensor_2d(*ctx, source->type, source->ne[0], source->ne[1])
                                 : ggml_new_tensor_3d(*ctx, source->type, source->ne[0], source->ne[1], source->ne[2]);
        if (!alias) {
            return;
        }
        const std::string name = std::string(source->name[0] ? source->name : "weight") + suffix;
        ggml_set_name(alias, name.c_str());
        model->cpu_repack_aliases[source] = alias;
        if (use_amx) {
            model->cpu_amx_aliases[alias] = true;
        }
        PendingAlias pending{source, alias, false};
        if (use_amx) {
            pending_cpu_amx.push_back(pending);
        } else {
            pending_cpu_repack.push_back(pending);
        }
    };

    auto make_decode_alias = [&](ggml_tensor* source, const char* suffix) {
        if (!source || !source->data || source->view_src || source->ne[0] <= 0 || source->ne[1] <= 0 ||
            source->ne[2] != 1 || source->ne[3] != 1 || !repack_bufts.cpu_repack) {
            return;
        }
        if (!ggml_is_quantized(source->type) && source->type != GGML_TYPE_F16) {
            return;
        }
        if (model->cpu_decode_repack_aliases.find(source) != model->cpu_decode_repack_aliases.end()) {
            return;
        }
        if (!init_alias_context(&model->ctx_cpu_repack, "CPU_REPACK")) {
            return;
        }
        ggml_tensor* alias = ggml_new_tensor_2d(model->ctx_cpu_repack, source->type, source->ne[0], source->ne[1]);
        if (!alias) {
            return;
        }
        const std::string name = std::string(source->name[0] ? source->name : "weight") + suffix;
        ggml_set_name(alias, name.c_str());
        model->cpu_decode_repack_aliases[source] = alias;
        pending_cpu_repack.push_back(PendingAlias{source, alias, true});
    };

    auto make_fused_pair_alias = [&](TransformerLayer& layer, uint32_t layer_idx, const char* output_key,
                                     const char* name_fragment, ggml_tensor* first, ggml_tensor* second,
                                     bool force_cpu_repack = false) {
        if (model->variant == ModelVariant::QWEN36) {
            return;
        }
        if (!first || !second || !first->data || !second->data || first->view_src || second->view_src) {
            return;
        }
        if (first->type != second->type || first->ne[0] != second->ne[0] || first->ne[2] != 1 || second->ne[2] != 1 ||
            first->ne[3] != 1 || second->ne[3] != 1) {
            return;
        }
        if (!ggml_is_quantized(first->type) || first->ne[0] <= 0 || first->ne[1] <= 0 || second->ne[1] <= 0) {
            return;
        }
        ggml_backend_buffer_type_t alias_buft =
            force_cpu_repack ? repack_bufts.cpu_repack : choose_alias_buffer(first, /*is_2d=*/true);
        if (!alias_buft) {
            return;
        }
        const size_t row_bytes = ggml_row_size(first->type, first->ne[0]);
        if (row_bytes == 0 || first->nb[1] < row_bytes || second->nb[1] < row_bytes) {
            return;
        }
        const bool use_amx = alias_buft == repack_bufts.cpu_amx;
        ggml_context** ctx = use_amx ? &model->ctx_cpu_amx : &model->ctx_cpu_repack;
        const char* buft_name = use_amx ? "AMX" : "CPU_REPACK";
        if (!init_alias_context(ctx, buft_name)) {
            return;
        }

        const int64_t fused_rows = first->ne[1] + second->ne[1];
        ggml_tensor* alias = ggml_new_tensor_2d(*ctx, first->type, first->ne[0], fused_rows);
        if (!alias) {
            return;
        }
        char name[128];
        std::snprintf(name, sizeof(name), "blk.%u.%s.%s_fused_2d", layer_idx, name_fragment,
                      use_amx ? "amx" : "cpu_repack");
        ggml_set_name(alias, name);
        if (use_amx) {
            model->cpu_amx_aliases[alias] = true;
        }

        PendingFusedAlias fused{};
        fused.layer = &layer;
        fused.alias = alias;
        fused.key = output_key;
        fused.bytes.resize(row_bytes * static_cast<size_t>(fused_rows));

        size_t dst_row = 0;
        auto append_rows = [&](const ggml_tensor* src) {
            const uint8_t* src_bytes = static_cast<const uint8_t*>(src->data);
            for (int64_t r = 0; r < src->ne[1]; ++r) {
                std::memcpy(fused.bytes.data() + dst_row * row_bytes,
                            src_bytes + static_cast<size_t>(r) * static_cast<size_t>(src->nb[1]), row_bytes);
                ++dst_row;
            }
        };
        append_rows(first);
        append_rows(second);
        if (use_amx) {
            pending_fused_cpu_amx.push_back(std::move(fused));
        } else {
            pending_fused_cpu_repack.push_back(std::move(fused));
        }
    };

    auto make_fused_expert_pair_alias = [&](TransformerLayer& layer, uint32_t layer_idx, const char* output_key,
                                            const char* name_fragment, ggml_tensor* first, ggml_tensor* second) {
        if (!first || !second || !first->data || !second->data || first->view_src || second->view_src) {
            return;
        }
        if (first->type != second->type || first->ne[0] != second->ne[0] || first->ne[2] <= 1 ||
            first->ne[2] != second->ne[2] || first->ne[3] != 1 || second->ne[3] != 1) {
            return;
        }
        if (!ggml_is_quantized(first->type) || first->ne[0] <= 0 || first->ne[1] <= 0 || second->ne[1] <= 0) {
            return;
        }
        ggml_backend_buffer_type_t alias_buft = choose_alias_buffer(first, /*is_2d=*/false);
        if (!alias_buft) {
            return;
        }
        if (alias_buft == repack_bufts.cpu_amx) {
            return;
        }
        const size_t row_bytes = ggml_row_size(first->type, first->ne[0]);
        if (row_bytes == 0 || first->nb[1] < row_bytes || second->nb[1] < row_bytes) {
            return;
        }
        const bool use_amx = alias_buft == repack_bufts.cpu_amx;
        ggml_context** ctx = use_amx ? &model->ctx_cpu_amx : &model->ctx_cpu_repack;
        if (!init_alias_context(ctx, use_amx ? "AMX" : "CPU_REPACK")) {
            return;
        }

        const int64_t fused_rows = first->ne[1] + second->ne[1];
        ggml_tensor* alias = ggml_new_tensor_3d(*ctx, first->type, first->ne[0], fused_rows, first->ne[2]);
        if (!alias) {
            return;
        }
        char name[128];
        std::snprintf(name, sizeof(name), "blk.%u.%s.%s_fused_3d", layer_idx, name_fragment,
                      use_amx ? "amx" : "cpu_repack");
        ggml_set_name(alias, name);
        if (use_amx) {
            model->cpu_amx_aliases[alias] = true;
        }

        PendingFusedAlias fused{};
        fused.layer = &layer;
        fused.alias = alias;
        fused.key = output_key;
        fused.first = first;
        fused.second = second;
        fused.expert_pair = true;
        if (use_amx) {
            pending_fused_cpu_amx.push_back(std::move(fused));
        } else {
            pending_fused_cpu_repack.push_back(std::move(fused));
        }
    };

    auto make_qwen35_ssm_qkv_gate_amx_fused_alias = [&](TransformerLayer& layer, uint32_t layer_idx) {
#if !defined(__aarch64__) && !defined(_M_ARM64)
        if (model->variant != ModelVariant::QWEN35 || !model->arch_flags.is_hybrid_ssm || !repack_bufts.cpu_amx) {
            return;
        }
        ggml_tensor* qkv = layer.Get(model_keys::kAttnQkvWeight);
        ggml_tensor* gate = layer.Get(model_keys::kAttnGate);
        if (!qkv || !gate || !qkv->data || !gate->data || qkv->view_src || gate->view_src) {
            return;
        }
        if (qkv->type != gate->type || qkv->ne[0] != gate->ne[0] || qkv->ne[2] != 1 || gate->ne[2] != 1 ||
            qkv->ne[3] != 1 || gate->ne[3] != 1 || !ggml_is_quantized(qkv->type)) {
            return;
        }
        const size_t row_bytes = ggml_row_size(qkv->type, qkv->ne[0]);
        if (row_bytes == 0 || qkv->nb[1] < row_bytes || gate->nb[1] < row_bytes) {
            return;
        }
        if (!init_alias_context(&model->ctx_cpu_amx, "AMX")) {
            return;
        }
        const int64_t fused_rows = qkv->ne[1] + gate->ne[1];
        ggml_tensor* alias = ggml_new_tensor_2d(model->ctx_cpu_amx, qkv->type, qkv->ne[0], fused_rows);
        if (!alias) {
            return;
        }
        char name[128];
        std::snprintf(name, sizeof(name), "blk.%u.ssm_qkv_gate.amx_fused_2d", layer_idx);
        ggml_set_name(alias, name);
        model->cpu_amx_aliases[alias] = true;

        PendingFusedAlias fused{};
        fused.layer = &layer;
        fused.alias = alias;
        fused.key = "attn_qkv_gate.amx_fused_decode";
        fused.bytes.resize(row_bytes * static_cast<size_t>(fused_rows));
        size_t dst_row = 0;
        auto append_rows = [&](const ggml_tensor* src) {
            const uint8_t* src_bytes = static_cast<const uint8_t*>(src->data);
            for (int64_t r = 0; r < src->ne[1]; ++r) {
                std::memcpy(fused.bytes.data() + dst_row * row_bytes,
                            src_bytes + static_cast<size_t>(r) * static_cast<size_t>(src->nb[1]), row_bytes);
                ++dst_row;
            }
        };
        append_rows(qkv);
        append_rows(gate);
        pending_fused_cpu_amx.push_back(std::move(fused));
#else
        (void)layer;
        (void)layer_idx;
#endif
    };

    if (model->tok_embeddings && model->tok_embeddings != model->output) {
        make_alias(model->tok_embeddings, ".fast_matmul_2d");
    }
    if (model->output && model->output != model->tok_embeddings) {
        make_alias(model->output, ".fast_matmul_2d");
    }
#if !defined(__aarch64__) && !defined(_M_ARM64)
    if (model->output && (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
        ggml_is_quantized(model->output->type)) {
        make_decode_alias(model->output, ".decode_fast_matmul");
    }
#endif
    for (auto& layer : model->layers) {
        for (const auto& item : layer.tensors) {
            if (model->arch_flags.is_hybrid_ssm && model->variant == ModelVariant::QWEN35 &&
                (item.first.find("ffn_gate_exps") != std::string::npos ||
                 item.first.find("ffn_up_exps") != std::string::npos)) {
                continue;
            }
            make_alias(item.second, ".fast_matmul");
        }
    }
#if !defined(__aarch64__) && !defined(_M_ARM64)
    const bool enable_qwen_fused_ffn_aliases =
        model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36;
    if (enable_qwen_fused_ffn_aliases) {
        for (uint32_t i = 0; i < model->layers.size(); ++i) {
            auto& layer = model->layers[i];
            make_qwen35_ssm_qkv_gate_amx_fused_alias(layer, i);
            make_fused_pair_alias(layer, i, "ffn_gate_up.cpu_repack_fused", "ffn_gate_up.weight",
                                  layer.Get(model_keys::kFfnGate), layer.Get(model_keys::kFfnUp));
            if (model->variant == ModelVariant::QWEN35) {
                make_fused_pair_alias(layer, i, "ffn_gate_up.cpu_repack_fused_decode", "ffn_gate_up.weight.decode",
                                      layer.Get(model_keys::kFfnGate), layer.Get(model_keys::kFfnUp),
                                      /*force_cpu_repack=*/true);
            }
            if (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) {
                make_fused_expert_pair_alias(
                    layer, i, "ffn_gate_up_exps.cpu_repack_fused", "ffn_gate_up_exps.weight",
                    layer.Get("ffn_gate_exps.weight") ? layer.Get("ffn_gate_exps.weight") : layer.Get("ffn_gate_exps"),
                    layer.Get("ffn_up_exps.weight") ? layer.Get("ffn_up_exps.weight") : layer.Get("ffn_up_exps"));
            }
        }
    }
#endif

    if (pending_cpu_repack.empty() && pending_fused_cpu_repack.empty() && pending_cpu_amx.empty() &&
        pending_fused_cpu_amx.empty()) {
        return;
    }

    auto allocate_and_commit_aliases = [&](std::vector<PendingAlias>& pending, ggml_context* ctx,
                                           std::vector<PendingFusedAlias>& pending_fused,
                                           ggml_backend_buffer_type_t buft, const char* buft_name) -> size_t {
        if ((pending.empty() && pending_fused.empty()) || !ctx || !buft) {
            return 0;
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buffer) {
            std::cerr << "[DenseCore] Warning: failed to allocate " << buft_name << " alias buffer" << std::endl;
            for (const auto& item : pending) {
                if (item.source) {
                    if (item.decode_only) {
                        model->cpu_decode_repack_aliases.erase(item.source);
                    } else {
                        model->cpu_repack_aliases.erase(item.source);
                    }
                }
            }
            return 0;
        }
        model->cpu_repack_buffers.push_back(buffer);

        size_t supported_count = 0;
        size_t fused_supported_count = 0;
        size_t bytes = 0;
        std::map<ggml_type, size_t> type_counts;
        for (const auto& item : pending) {
            if (!item.source || !item.alias || !item.alias->extra) {
                if (item.source) {
                    if (item.decode_only) {
                        model->cpu_decode_repack_aliases.erase(item.source);
                    } else {
                        model->cpu_repack_aliases.erase(item.source);
                    }
                }
                continue;
            }
            ggml_backend_tensor_set(item.alias, item.source->data, 0, ggml_nbytes(item.source));
            ++supported_count;
            bytes += ggml_nbytes(item.alias);
            type_counts[item.alias->type]++;
        }
        for (auto& item : pending_fused) {
            if (!item.layer || !item.alias || !item.alias->extra) {
                continue;
            }
            if (item.expert_pair) {
                if (!item.first || !item.second || !item.first->data || !item.second->data) {
                    continue;
                }
                const size_t row_bytes = ggml_row_size(item.first->type, item.first->ne[0]);
                if (row_bytes == 0) {
                    continue;
                }
                std::vector<uint8_t> fused_bytes(ggml_nbytes(item.alias));
                size_t dst_off = 0;
                auto append_one_expert_rows = [&](const ggml_tensor* src, int64_t expert) {
                    const uint8_t* src_bytes = static_cast<const uint8_t*>(src->data);
                    const size_t expert_base = static_cast<size_t>(expert) * static_cast<size_t>(src->nb[2]);
                    for (int64_t r = 0; r < src->ne[1]; ++r) {
                        const uint8_t* row =
                            src_bytes + expert_base + static_cast<size_t>(r) * static_cast<size_t>(src->nb[1]);
                        std::memcpy(fused_bytes.data() + dst_off, row, row_bytes);
                        dst_off += row_bytes;
                    }
                };
                for (int64_t expert = 0; expert < item.first->ne[2]; ++expert) {
                    append_one_expert_rows(item.first, expert);
                    append_one_expert_rows(item.second, expert);
                }
                ggml_backend_tensor_set(item.alias, fused_bytes.data(), 0, fused_bytes.size());
            } else if (!item.bytes.empty()) {
                ggml_backend_tensor_set(item.alias, item.bytes.data(), 0, item.bytes.size());
            } else {
                continue;
            }
            item.layer->Set(item.key, item.alias);
            ++fused_supported_count;
            bytes += ggml_nbytes(item.alias);
            type_counts[item.alias->type]++;
        }
        if (supported_count == 0 && fused_supported_count == 0) {
            std::cerr << "[DenseCore] Warning: no " << buft_name << " aliases were supported by this ggml build"
                      << std::endl;
            return 0;
        }

        std::ostringstream type_summary;
        bool first = true;
        for (const auto& [type, count] : type_counts) {
            if (!first) {
                type_summary << ",";
            }
            first = false;
            type_summary << ggml_type_name(type) << ":" << count;
        }
        std::cout << "[DenseCore] CPU fast matmul aliases prepared: buft=" << buft_name
                  << ", aliases=" << supported_count << ", fused_ffn=" << fused_supported_count << ", alias_types=["
                  << type_summary.str() << "]" << ", bytes=" << (bytes / 1024 / 1024) << " MiB" << std::endl;
        return supported_count + fused_supported_count;
    };

    allocate_and_commit_aliases(pending_cpu_amx, model->ctx_cpu_amx, pending_fused_cpu_amx, repack_bufts.cpu_amx,
                                "AMX");
    allocate_and_commit_aliases(pending_cpu_repack, model->ctx_cpu_repack, pending_fused_cpu_repack,
                                repack_bufts.cpu_repack, "CPU_REPACK");
}

void PrepareGemma4CpuRepackAliases(TransformerModel* model) {
    if (!model || !model->arch_flags.is_gemma4 || model->layers.empty()) {
        return;
    }

    const CpuRepackBufferTypes repack_bufts =
        FindCpuRepackBufferTypes(model->cpu_backend ? model->cpu_backend : model->backend);
    ggml_backend_buffer_type_t primary_repack_buft = repack_bufts.cpu_repack;
    const char* primary_repack_buft_name = "CPU_REPACK";
    if (!primary_repack_buft && !repack_bufts.cpu_kleidiai) {
        if (EnvFlagEnabled("DENSECORE_DEBUG_GEMMA4_CPU_REPACK", false)) {
            std::cerr << "[DenseCore] Gemma4 CPU_REPACK aliases unavailable: ggml CPU_REPACK buffer type not present"
                      << std::endl;
        }
        return;
    }

    struct PendingAlias {
        ggml_tensor* source = nullptr;
        ggml_tensor* alias = nullptr;
        std::vector<uint8_t> bytes;
        bool use_kleidiai = false;
    };
    struct PendingFusedAlias {
        TransformerLayer* layer = nullptr;
        ggml_tensor* alias = nullptr;
        std::string key;
        std::vector<uint8_t> bytes;
        bool use_kleidiai = false;
    };
    std::vector<PendingAlias> pending;
    std::vector<PendingFusedAlias> pending_fused;

    const size_t tensor_slots = static_cast<size_t>(std::max<uint32_t>(1, model->hparams.n_layer)) * 16 + 96;
    auto init_repack_context = [&](ggml_context** ctx, const char* label) -> bool {
        if (*ctx) {
            return true;
        }
        struct ggml_init_params params = {
            /*.mem_size   =*/tensor_slots * ggml_tensor_overhead() + 1024 * 1024,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        *ctx = ggml_init(params);
        if (!*ctx) {
            std::cerr << "[DenseCore] Warning: failed to allocate Gemma4 " << label << " metadata context" << std::endl;
            return false;
        }
        return true;
    };

    if (!init_repack_context(&model->ctx_cpu_repack, "CPU_REPACK")) {
        return;
    }
    const bool allow_gemma4_kleidiai_aliases = false;
    if (allow_gemma4_kleidiai_aliases && repack_bufts.cpu_kleidiai &&
        !init_repack_context(&model->ctx_cpu_kleidiai, "CPU_KLEIDIAI")) {
        return;
    }

    auto use_kleidiai_for_type = [&](ggml_type type) -> bool {
        // Gemma4 free-form QA currently fails or slows down when only the Q8_0
        // subset is moved to ggml's CPU_KLEIDIAI buffer while Q4_K/Q5_1 stay on
        // CPU_REPACK. Keep the loader quality gate closed until a per-op parity
        // and server-throughput probe promotes this mixed-buffer path.
        if (!allow_gemma4_kleidiai_aliases) {
            return false;
        }
        return repack_bufts.cpu_kleidiai && model->ctx_cpu_kleidiai && CanUseCpuKleidiaiRepack(type);
    };

    auto context_for_type = [&](ggml_type type) -> ggml_context* {
        return use_kleidiai_for_type(type) ? model->ctx_cpu_kleidiai : model->ctx_cpu_repack;
    };

    auto repack_buffer_available_for_type = [&](ggml_type type) -> bool {
        if (use_kleidiai_for_type(type)) {
            return true;
        }
        if (!primary_repack_buft) {
            return false;
        }
        return true;
    };

    if (!primary_repack_buft) {
        bool has_non_kleidiai_type = false;
        for (const auto& layer : model->layers) {
            for (const auto& kv : layer.tensors) {
                const ggml_tensor* tensor = kv.second;
                if (tensor && !CanUseCpuKleidiaiRepack(tensor->type)) {
                    has_non_kleidiai_type = true;
                    break;
                }
            }
            if (has_non_kleidiai_type) {
                break;
            }
        }
        if (has_non_kleidiai_type) {
            std::cerr << "[DenseCore] Warning: Gemma4 CPU_REPACK aliases require CPU_REPACK for non-KleidiAI "
                         "quant types, but CPU_REPACK is unavailable"
                      << std::endl;
            return;
        }
    }

    auto register_alias = [&](ggml_tensor* source, ggml_tensor* alias, const char* suffix,
                              std::vector<uint8_t> bytes = {}) -> bool {
        if (!source || !alias) {
            return false;
        }
        if (model->cpu_repack_aliases.find(source) != model->cpu_repack_aliases.end()) {
            return true;
        }
        const std::string name = std::string(source->name[0] ? source->name : "gemma4_moe_weight") + suffix;
        ggml_set_name(alias, name.c_str());
        if (bytes.empty() && ggml_nbytes(alias) != ggml_nbytes(source)) {
            std::cerr << "[DenseCore] Warning: skipping Gemma4 CPU_REPACK alias for "
                      << (source->name[0] ? source->name : "<unnamed>") << " because packed byte size differs"
                      << std::endl;
            return false;
        }
        if (!bytes.empty() && bytes.size() != ggml_nbytes(alias)) {
            std::cerr << "[DenseCore] Warning: skipping Gemma4 CPU_REPACK alias for "
                      << (source->name[0] ? source->name : "<unnamed>")
                      << " because converted byte size does not match alias" << std::endl;
            return false;
        }
        model->cpu_repack_aliases[source] = alias;
        pending.push_back({source, alias, std::move(bytes), use_kleidiai_for_type(alias->type)});
        return true;
    };

    auto make_alias_2d = [&](ggml_tensor* source, const char* suffix) -> bool {
        if (!source || !source->data || source->view_src || source->ne[0] <= 0 || source->ne[1] <= 0) {
            return false;
        }
        if (!repack_buffer_available_for_type(source->type)) {
            return false;
        }
        ggml_tensor* alias =
            ggml_new_tensor_2d(context_for_type(source->type), source->type, source->ne[0], source->ne[1]);
        return register_alias(source, alias, suffix);
    };

    auto make_alias_3d = [&](ggml_tensor* source, int64_t ne0, int64_t ne1, int64_t ne2, const char* suffix) -> bool {
        if (!source || !source->data || ne0 <= 0 || ne1 <= 0 || ne2 <= 0) {
            return false;
        }
        if (!repack_buffer_available_for_type(source->type)) {
            return false;
        }
        ggml_tensor* alias = ggml_new_tensor_3d(context_for_type(source->type), source->type, ne0, ne1, ne2);
        return register_alias(source, alias, suffix);
    };

    const size_t available_requant_bytes = ReadMemAvailableBytes();
    const size_t q5_down_requant_bytes = EstimateGemma4Q5DownRequantBytes(model);
    const size_t q4_gate_up_requant_bytes = EstimateGemma4Q4GateUpRequantBytes(model);
    const bool auto_requant_q5_1_down_to_q8_0 =
        available_requant_bytes > 0 &&
        available_requant_bytes >= q5_down_requant_bytes + kGemma4RequantSafetyHeadroomBytes;
    const bool auto_requant_q4_gate_up_to_q8_0 = available_requant_bytes > 0 && q4_gate_up_requant_bytes > 0 &&
                                                 available_requant_bytes >= q5_down_requant_bytes +
                                                                                q4_gate_up_requant_bytes +
                                                                                kGemma4GateUpRequantSafetyHeadroomBytes;
    if (q5_down_requant_bytes > 0) {
        std::cout << "[DenseCore] Gemma4 Q5_1 down-expert Q8_0 repack auto "
                  << (auto_requant_q5_1_down_to_q8_0 ? "enabled" : "disabled")
                  << ": required=" << (q5_down_requant_bytes / 1024 / 1024)
                  << " MiB available=" << (available_requant_bytes / 1024 / 1024)
                  << " MiB headroom=" << (kGemma4RequantSafetyHeadroomBytes / 1024 / 1024) << " MiB" << std::endl;
    }
    if (q4_gate_up_requant_bytes > 0) {
        std::cout << "[DenseCore] Gemma4 Q4_K gate/up-expert Q8_0 repack auto "
                  << (auto_requant_q4_gate_up_to_q8_0 ? "enabled" : "disabled")
                  << ": required_gate_up=" << (q4_gate_up_requant_bytes / 1024 / 1024)
                  << " MiB required_down=" << (q5_down_requant_bytes / 1024 / 1024)
                  << " MiB available=" << (available_requant_bytes / 1024 / 1024)
                  << " MiB headroom=" << (kGemma4GateUpRequantSafetyHeadroomBytes / 1024 / 1024) << " MiB" << std::endl;
    }
    auto make_requant_q8_0_alias_3d = [&](ggml_tensor* source, int64_t ne0, int64_t ne1, int64_t ne2,
                                          const char* suffix, bool enabled) -> bool {
        if (!enabled) {
            return false;
        }
        if (!source || !source->data || !ggml_is_quantized(source->type) || source->view_src || ne0 <= 0 || ne1 <= 0 ||
            ne2 <= 0 || source->ne[0] != ne0 || source->ne[1] != ne1 || source->ne[2] != ne2) {
            return false;
        }
        const ggml_type_traits* src_traits = ggml_get_type_traits(source->type);
        const ggml_type_traits_cpu* dst_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if (!src_traits || !src_traits->to_float || !dst_traits || !dst_traits->from_float) {
            return false;
        }
        if (!repack_buffer_available_for_type(GGML_TYPE_Q8_0)) {
            return false;
        }
        ggml_tensor* alias = ggml_new_tensor_3d(context_for_type(GGML_TYPE_Q8_0), GGML_TYPE_Q8_0, ne0, ne1, ne2);
        if (!alias) {
            return false;
        }
        std::vector<uint8_t> converted(ggml_nbytes(alias));
        std::vector<float> row(static_cast<size_t>(ne0), 0.0f);
        const size_t dst_row_size = ggml_row_size(GGML_TYPE_Q8_0, ne0);
        for (int64_t expert = 0; expert < ne2; ++expert) {
            for (int64_t row_idx = 0; row_idx < ne1; ++row_idx) {
                const char* src_row = static_cast<const char*>(source->data) +
                                      static_cast<size_t>(expert) * static_cast<size_t>(source->nb[2]) +
                                      static_cast<size_t>(row_idx) * static_cast<size_t>(source->nb[1]);
                uint8_t* dst_row = converted.data() + (static_cast<size_t>(expert) * static_cast<size_t>(ne1) +
                                                       static_cast<size_t>(row_idx)) *
                                                          dst_row_size;
                src_traits->to_float(src_row, row.data(), ne0);
                dst_traits->from_float(row.data(), dst_row, ne0);
            }
        }
        return register_alias(source, alias, suffix, std::move(converted));
    };

    auto make_fused_pair_alias = [&](TransformerLayer& layer, uint32_t layer_idx, const char* output_key,
                                     const char* name_fragment, ggml_tensor* first, ggml_tensor* second) -> bool {
        if (!first || !second || !first->data || !second->data || first->view_src || second->view_src) {
            return false;
        }
        if (first->type != second->type || first->ne[0] != second->ne[0]) {
            return false;
        }
        if (!ggml_is_quantized(first->type) || first->ne[0] <= 0 || first->ne[1] <= 0 || second->ne[1] <= 0) {
            return false;
        }

        const size_t row_bytes = ggml_row_size(first->type, first->ne[0]);
        if (row_bytes == 0 || first->nb[1] < static_cast<int64_t>(row_bytes) ||
            second->nb[1] < static_cast<int64_t>(row_bytes)) {
            return false;
        }

        const int64_t fused_rows = first->ne[1] + second->ne[1];
        if (!repack_buffer_available_for_type(first->type)) {
            return false;
        }
        ggml_tensor* alias = ggml_new_tensor_2d(context_for_type(first->type), first->type, first->ne[0], fused_rows);
        if (!alias) {
            return false;
        }
        char name[128];
        std::snprintf(name, sizeof(name), "blk.%u.%s.cpu_repack_fused_2d", layer_idx, name_fragment);
        ggml_set_name(alias, name);

        PendingFusedAlias fused{};
        fused.layer = &layer;
        fused.alias = alias;
        fused.key = output_key;
        fused.bytes.resize(row_bytes * static_cast<size_t>(fused_rows));
        fused.use_kleidiai = use_kleidiai_for_type(alias->type);

        size_t dst_row = 0;
        auto append_rows = [&](const ggml_tensor* src) {
            const uint8_t* src_bytes = static_cast<const uint8_t*>(src->data);
            for (int64_t r = 0; r < src->ne[1]; ++r) {
                std::memcpy(fused.bytes.data() + dst_row * row_bytes,
                            src_bytes + static_cast<size_t>(r) * static_cast<size_t>(src->nb[1]), row_bytes);
                ++dst_row;
            }
        };
        append_rows(first);
        append_rows(second);
        pending_fused.push_back(std::move(fused));
        return true;
    };

    auto make_fused_qkv_alias = [&](TransformerLayer& layer, uint32_t layer_idx) -> bool {
        ggml_tensor* q = layer.Get(model_keys::kAttnQWeight);
        ggml_tensor* k = layer.Get(model_keys::kAttnKWeight);
        ggml_tensor* v = layer.Get(model_keys::kAttnVWeight);
        if (!q || !k || !v || !q->data || !k->data || !v->data || q->view_src || k->view_src || v->view_src) {
            return false;
        }
        if (q->type != k->type || q->type != v->type || q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0]) {
            return false;
        }
        if (!ggml_is_quantized(q->type) || q->ne[0] <= 0 || q->ne[1] <= 0 || k->ne[1] <= 0 || v->ne[1] <= 0) {
            return false;
        }

        const size_t row_bytes = ggml_row_size(q->type, q->ne[0]);
        if (row_bytes == 0 || q->nb[1] < static_cast<int64_t>(row_bytes) ||
            k->nb[1] < static_cast<int64_t>(row_bytes) || v->nb[1] < static_cast<int64_t>(row_bytes)) {
            return false;
        }

        const int64_t fused_rows = q->ne[1] + k->ne[1] + v->ne[1];
        if (!repack_buffer_available_for_type(q->type)) {
            return false;
        }
        ggml_tensor* alias = ggml_new_tensor_2d(context_for_type(q->type), q->type, q->ne[0], fused_rows);
        if (!alias) {
            return false;
        }
        char name[128];
        std::snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight.cpu_repack_fused_2d", layer_idx);
        ggml_set_name(alias, name);

        PendingFusedAlias fused{};
        fused.layer = &layer;
        fused.alias = alias;
        fused.key = "attn_qkv.cpu_repack_fused";
        fused.bytes.resize(row_bytes * static_cast<size_t>(fused_rows));
        fused.use_kleidiai = use_kleidiai_for_type(alias->type);

        size_t dst_row = 0;
        auto append_rows = [&](const ggml_tensor* src) {
            const uint8_t* src_bytes = static_cast<const uint8_t*>(src->data);
            for (int64_t r = 0; r < src->ne[1]; ++r) {
                std::memcpy(fused.bytes.data() + dst_row * row_bytes,
                            src_bytes + static_cast<size_t>(r) * static_cast<size_t>(src->nb[1]), row_bytes);
                ++dst_row;
            }
        };
        append_rows(q);
        append_rows(k);
        append_rows(v);
        pending_fused.push_back(std::move(fused));
        return true;
    };

    // Gemma4 ties lm_head to token_embd. Repacking that shared tensor duplicates
    // a very large vocab matrix without a stable server-throughput win on C4A.
    if (model->tok_embeddings && model->tok_embeddings != model->output) {
        make_alias_2d(model->tok_embeddings, ".cpu_repack_2d");
    }
    if (model->output && model->output != model->tok_embeddings) {
        make_alias_2d(model->output, ".cpu_repack_2d");
    }
    for (uint32_t i = 0; i < model->layers.size(); ++i) {
        auto& layer = model->layers[i];
        make_fused_qkv_alias(layer, i);
        make_fused_pair_alias(layer, i, "ffn_gate_up.cpu_repack_fused", "ffn_gate_up.weight",
                              layer.Get(model_keys::kFfnGate), layer.Get(model_keys::kFfnUp));
        make_alias_2d(layer.Get(model_keys::kAttnQWeight), ".cpu_repack_2d");
        make_alias_2d(layer.Get(model_keys::kAttnKWeight), ".cpu_repack_2d");
        make_alias_2d(layer.Get(model_keys::kAttnVWeight), ".cpu_repack_2d");
        make_alias_2d(layer.Get(model_keys::kAttnOWeight), ".cpu_repack_2d");
        make_alias_2d(layer.Get(model_keys::kFfnGate), ".cpu_repack_2d");
        make_alias_2d(layer.Get(model_keys::kFfnUp), ".cpu_repack_2d");
        make_alias_2d(layer.Get(model_keys::kFfnDown), ".cpu_repack_2d");
    }

    for (uint32_t i = 0; i < model->layers.size(); ++i) {
        auto& layer = model->layers[i];
        ggml_tensor* gate_up =
            layer.Get("ffn_gate_up_exps.weight") ? layer.Get("ffn_gate_up_exps.weight") : layer.Get("ffn_gate_up_exps");
        ggml_tensor* down =
            layer.Get("ffn_down_exps.weight") ? layer.Get("ffn_down_exps.weight") : layer.Get("ffn_down_exps");
        if (!gate_up || !down) {
            continue;
        }
        densecore::gemma4::PackedExpertLayout layout{};
        std::string reason;
        if (!densecore::gemma4::InferPackedExpertLayout(gate_up, down, &layout, &reason)) {
            continue;
        }
        if (!make_requant_q8_0_alias_3d(gate_up, layout.hidden_dim, layout.intermediate_dim * 2, layout.num_experts,
                                        ".cpu_repack_gate_up_q8_0_3d", auto_requant_q4_gate_up_to_q8_0)) {
            make_alias_3d(gate_up, layout.hidden_dim, layout.intermediate_dim * 2, layout.num_experts,
                          ".cpu_repack_gate_up_3d");
        }
        if (!make_requant_q8_0_alias_3d(down, layout.intermediate_dim, layout.hidden_dim, layout.num_experts,
                                        ".cpu_repack_down_q8_0_3d", auto_requant_q5_1_down_to_q8_0)) {
            make_alias_3d(down, layout.intermediate_dim, layout.hidden_dim, layout.num_experts, ".cpu_repack_down_3d");
        }
    }

    if (pending.empty() && pending_fused.empty()) {
        return;
    }

    std::map<ggml_type, size_t> alias_type_counts;
    size_t cpu_repack_aliases = 0;
    size_t cpu_kleidiai_aliases = 0;
    for (const auto& item : pending) {
        if (!item.alias) {
            continue;
        }
        alias_type_counts[item.alias->type]++;
        if (item.use_kleidiai) {
            ++cpu_kleidiai_aliases;
        } else {
            ++cpu_repack_aliases;
        }
    }
    for (const auto& item : pending_fused) {
        if (!item.alias) {
            continue;
        }
        alias_type_counts[item.alias->type]++;
        if (item.use_kleidiai) {
            ++cpu_kleidiai_aliases;
        } else {
            ++cpu_repack_aliases;
        }
    }

    if (cpu_repack_aliases > 0 && !repack_bufts.cpu_repack) {
        std::cerr << "[DenseCore] Warning: Gemma4 CPU_REPACK aliases require CPU_REPACK because this model uses "
                     "non-KleidiAI quant types, but CPU_REPACK is unavailable"
                  << std::endl;
        model->cpu_repack_aliases.clear();
        return;
    }

    if (cpu_repack_aliases > 0) {
        ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(model->ctx_cpu_repack, primary_repack_buft);
        if (!buffer) {
            std::cerr << "[DenseCore] Warning: failed to allocate Gemma4 " << primary_repack_buft_name << " buffer"
                      << std::endl;
            model->cpu_repack_aliases.clear();
            return;
        }
        model->cpu_repack_buffers.push_back(buffer);
    }
    if (cpu_kleidiai_aliases > 0) {
        ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(model->ctx_cpu_kleidiai, repack_bufts.cpu_kleidiai);
        if (!buffer) {
            std::cerr << "[DenseCore] Warning: failed to allocate Gemma4 CPU_KLEIDIAI buffer" << std::endl;
            model->cpu_repack_aliases.clear();
            return;
        }
        model->cpu_repack_buffers.push_back(buffer);
    }

    auto commit_plain_cpu_alias = [&](PendingAlias& item) -> bool {
        if (!item.source || !item.alias || item.bytes.empty() || item.alias->type != GGML_TYPE_Q8_0) {
            return false;
        }
        ggml_tensor* alias = ggml_new_tensor(model->ctx_cpu_repack, item.alias->type, GGML_MAX_DIMS, item.alias->ne);
        if (!alias) {
            return false;
        }
        ggml_format_name(alias, "%s.cpu_q8_0", item.source->name[0] ? item.source->name : "gemma4_weight");

        void* data = densecore::NumaAllocator::AllocateAlignedOnNode(item.bytes.size(), 64, -1);
        if (!data) {
            return false;
        }
        ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(data, item.bytes.size());
        if (!buffer) {
            densecore::NumaAllocator::Free(data, item.bytes.size(), densecore::AllocationType::Aligned);
            return false;
        }
        if (ggml_backend_tensor_alloc(buffer, alias, data) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            densecore::NumaAllocator::Free(data, item.bytes.size(), densecore::AllocationType::Aligned);
            return false;
        }
        std::memcpy(alias->data, item.bytes.data(), item.bytes.size());
        model->numa_buffers.push_back({data, item.bytes.size(), densecore::AllocationType::Aligned});
        model->cpu_repack_buffers.push_back(buffer);
        model->cpu_repack_aliases[item.source] = alias;
        item.alias = alias;
        return true;
    };

    auto commit_plain_cpu_fused_alias = [&](PendingFusedAlias& item) -> bool {
        if (!item.alias || !item.layer || item.key.empty() || item.bytes.empty() ||
            item.alias->type != GGML_TYPE_Q8_0 || item.bytes.size() != ggml_nbytes(item.alias)) {
            return false;
        }
        ggml_tensor* alias = ggml_new_tensor(model->ctx_cpu_repack, item.alias->type, GGML_MAX_DIMS, item.alias->ne);
        if (!alias) {
            return false;
        }
        ggml_set_name(alias, item.alias->name[0] ? item.alias->name : "gemma4_fused_q8_0");

        void* data = densecore::NumaAllocator::AllocateAlignedOnNode(item.bytes.size(), 64, -1);
        if (!data) {
            return false;
        }
        ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(data, item.bytes.size());
        if (!buffer) {
            densecore::NumaAllocator::Free(data, item.bytes.size(), densecore::AllocationType::Aligned);
            return false;
        }
        if (ggml_backend_tensor_alloc(buffer, alias, data) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            densecore::NumaAllocator::Free(data, item.bytes.size(), densecore::AllocationType::Aligned);
            return false;
        }
        std::memcpy(alias->data, item.bytes.data(), item.bytes.size());
        model->numa_buffers.push_back({data, item.bytes.size(), densecore::AllocationType::Aligned});
        model->cpu_repack_buffers.push_back(buffer);
        item.layer->Set(item.key, alias);
        item.alias = alias;
        return true;
    };

    std::vector<PendingAlias> supported;
    supported.reserve(pending.size());
    for (auto& item : pending) {
        if (!item.alias || !item.source) {
            continue;
        }
        if (!item.alias->extra && commit_plain_cpu_alias(item)) {
            supported.push_back(item);
            continue;
        }
        if (!item.alias->extra) {
            if (EnvFlagEnabled("DENSECORE_DEBUG_GEMMA4_CPU_REPACK", false)) {
                std::cerr << "[DenseCore] Warning: Gemma4 CPU_REPACK did not support tensor "
                          << (item.source->name[0] ? item.source->name : "<unnamed>")
                          << " type=" << ggml_type_name(item.source->type) << " alias_ne=[" << item.alias->ne[0] << ","
                          << item.alias->ne[1] << "," << item.alias->ne[2] << "," << item.alias->ne[3]
                          << "]; keeping raw tensor for that projection" << std::endl;
            }
            model->cpu_repack_aliases.erase(item.source);
            continue;
        }
        supported.push_back(item);
    }
    size_t fused_supported = 0;
    for (auto& item : pending_fused) {
        if (!item.alias || !item.layer || item.bytes.empty()) {
            continue;
        }
        if (!item.alias->extra && commit_plain_cpu_fused_alias(item)) {
            ++fused_supported;
            continue;
        }
        if (!item.alias->extra) {
            continue;
        }
        ggml_backend_tensor_set(item.alias, item.bytes.data(), 0, item.bytes.size());
        item.layer->Set(item.key, item.alias);
        ++fused_supported;
    }
    if (supported.empty() && fused_supported == 0) {
        std::cerr << "[DenseCore] Warning: no Gemma4 CPU_REPACK aliases were supported by this ggml build" << std::endl;
        model->cpu_repack_aliases.clear();
        return;
    }
    size_t bytes = 0;
    for (const auto& item : supported) {
        const void* data = item.bytes.empty() ? item.source->data : item.bytes.data();
        const size_t nbytes = item.bytes.empty() ? ggml_nbytes(item.source) : item.bytes.size();
        ggml_backend_tensor_set(item.alias, data, 0, nbytes);
        bytes += ggml_nbytes(item.alias);
    }
    for (const auto& item : pending_fused) {
        if (item.alias && (item.alias->extra || item.alias->data)) {
            bytes += ggml_nbytes(item.alias);
        }
    }

    std::ostringstream type_summary;
    bool first_type = true;
    for (const auto& [type, count] : alias_type_counts) {
        if (!first_type) {
            type_summary << ",";
        }
        first_type = false;
        type_summary << ggml_type_name(type) << ":" << count;
    }
    std::cout << "[DenseCore] Gemma4 CPU_REPACK aliases prepared: buft="
              << (cpu_kleidiai_aliases > 0 ? std::string(primary_repack_buft_name) + "+CPU_KLEIDIAI"
                                           : std::string(primary_repack_buft_name))
              << ", kleidiai_available=" << (repack_bufts.cpu_kleidiai ? 1 : 0)
              << ", kleidiai_aliases=" << cpu_kleidiai_aliases << ", cpu_repack_aliases=" << cpu_repack_aliases
              << ", alias_types=[" << type_summary.str() << "]" << ", tensors=" << supported.size()
              << ", fused_aliases=" << fused_supported << ", bytes=" << (bytes / 1024 / 1024) << " MiB" << std::endl;
}
}  // namespace

void ClearQwen36SSMQ8PrefillAMXAliases(TransformerModel* model) {
    ClearQwen36SSMQ8PrefillAMXAliasesImpl(model);
}

bool PrepareQwen36SSMQ8PrefillAMXAliasesForExecution(TransformerModel* model) {
    return PrepareQwen36SSMQ8PrefillAMXAliasesForExecutionImpl(model);
}

bool IsMoELoaderDebugEnabled();
bool IsQwen36MoEBindingDebugEnabled();

// ============================================================================
// Mock Model (Test Build Only)
// ============================================================================

#ifdef DENSECORE_TEST_BUILD
/**
 * Creates a mock model for testing purposes.
 * Only available when DENSECORE_TEST_BUILD is defined.
 */
static TransformerModel* CreateMockModel() {
    std::cout << "[DenseCore] Initializing MOCK model..." << std::endl;
    TransformerModel* model = new TransformerModel();
    model->is_mock = true;
    model->arch = ModelArch::LLAMA;
    model->variant = ModelVariant::LLAMA;
    model->hparams.n_vocab = 32000;
    model->hparams.n_embd = 256;
    model->hparams.n_layer = 2;  // Small for mock
    model->hparams.n_head = 8;
    model->hparams.n_head_kv = 8;
    model->hparams.n_embd_head_k = 32;  // n_embd / n_head = 256 / 8
    model->hparams.n_rot = 32;

    // Mock vocab
    for (int i = 0; i < 32000; i++) {
        std::string s = "t" + std::to_string(i);
        model->vocab_tokens.push_back(s);
        model->token_to_id[s] = i;
    }
    for (int c = 1; c < 128; ++c) {
        const int id = c;
        std::string s(1, static_cast<char>(c));
        model->vocab_tokens[static_cast<size_t>(id)] = s;
        model->token_to_id[s] = id;
    }

    // Initialize backend
    model->backend = ggml_backend_cpu_init();
    if (!model->backend) {
        std::cerr << "[DenseCore] Error: Failed to initialize CPU backend" << std::endl;
        delete model;
        return nullptr;
    }
    model->cpu_backend = model->backend;
    model->cpu_backend = model->backend;

#if defined(__APPLE__) && !defined(DENSECORE_TEST_BUILD)
    if (densecore::apple::IsAppleHybridEnabled()) {
        ggml_backend_t metal_backend = ggml_backend_metal_init();
        if (metal_backend) {
            model->metal_backend = metal_backend;
            std::cout << "[DenseCore] GGML Metal backend initialized" << std::endl;
        } else {
            std::cerr << "[DenseCore] Warning: Failed to initialize GGML Metal backend" << std::endl;
        }
    }

#endif

    // Allocate dummy weights
    struct ggml_init_params params = {
        .mem_size = 1024LL * 1024LL * 1024LL * 2LL,  // 2 GB for dummy weights
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    model->ctx_w = ggml_init(params);

    auto create_tensor = [&](int ne0, int ne1) {
        struct ggml_tensor* t = ggml_new_tensor_2d(model->ctx_w, GGML_TYPE_F32, ne0, ne1);
        ggml_set_f32(t, 0.01f);  // Set to small value
        return t;
    };

    auto create_tensor_1d = [&](int ne0) {
        struct ggml_tensor* t = ggml_new_tensor_1d(model->ctx_w, GGML_TYPE_F32, ne0);
        ggml_set_f32(t, 0.01f);
        return t;
    };

    model->tok_embeddings = create_tensor(model->hparams.n_embd, model->hparams.n_vocab);
    model->output_norm = create_tensor_1d(model->hparams.n_embd);
    model->output = create_tensor(model->hparams.n_embd, model->hparams.n_vocab);

    model->layers.resize(model->hparams.n_layer);
    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        model->layers[i].Set(model_keys::kAttnNorm, create_tensor_1d(model->hparams.n_embd));
        model->layers[i].Set(model_keys::kFfnNorm, create_tensor_1d(model->hparams.n_embd));

        model->layers[i].Set(model_keys::kAttnQWeight, create_tensor(model->hparams.n_embd, model->hparams.n_embd));
        model->layers[i].Set(model_keys::kAttnKWeight, create_tensor(model->hparams.n_embd, model->hparams.n_embd));
        model->layers[i].Set(model_keys::kAttnVWeight, create_tensor(model->hparams.n_embd, model->hparams.n_embd));
        model->layers[i].Set(model_keys::kAttnOWeight, create_tensor(model->hparams.n_embd, model->hparams.n_embd));

        model->layers[i].Set(model_keys::kFfnGate, create_tensor(model->hparams.n_embd, model->hparams.n_embd * 4));
        model->layers[i].Set(model_keys::kFfnDown, create_tensor(model->hparams.n_embd * 4, model->hparams.n_embd));
        model->layers[i].Set(model_keys::kFfnUp, create_tensor(model->hparams.n_embd, model->hparams.n_embd * 4));
    }

    model->decoder_spec = densecore::models::MakeDecoderModelSpec(model);
    return model;
}
#endif  // DENSECORE_TEST_BUILD

// ============================================================================
// LoadGGUFModel
// ============================================================================

TransformerModel* LoadGGUFModel(const char* path) {
    if (!path) {
        std::cerr << "Error: Model path is NULL" << std::endl;
        return nullptr;
    }
    std::cout << "[DenseCore] Loading model from '" << path << "'..." << std::endl;

    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = nullptr,
    };

    struct ggml_context* ctx_w = nullptr;
    params.ctx = &ctx_w;

#ifdef DENSECORE_TEST_BUILD
    if (std::string(path) == "mock") {
        return CreateMockModel();
    }
#else
    if (std::string(path) == "mock") {
        std::cerr << "[DenseCore] Error: Mock model not available in release build" << std::endl;
        return nullptr;
    }
#endif

    struct gguf_context* ctx_gguf = gguf_init_from_file(path, params);
    if (!ctx_gguf) {
        std::cerr << "[DenseCore] Error: Failed to load GGUF file" << std::endl;
        return nullptr;
    }

    TransformerModel* model = new TransformerModel();
    model->ctx_gguf = ctx_gguf;
    model->ctx_w = ctx_w;
    const auto fail_load = [&](const std::string& reason) -> TransformerModel* {
        std::cerr << "[DenseCore] FATAL: " << reason << std::endl;
        delete model;
        return nullptr;
    };

    // Allocate a small separate context for view tensor metadata (expert slices, etc.).
    // gguf_init_from_file leaves no room for additional ggml_tensor structs in ctx_w,
    // so MoE view tensors (n_experts × n_layers × 3) must live in a dedicated pool.
    // 256 experts × 64 layers × 3 views × 512 bytes ≈ 24MB; allocate 32MB to be safe.
    {
        struct ggml_init_params vp = {
            /*.mem_size   =*/32LL * 1024LL * 1024LL,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/false,
        };
        model->ctx_views = ggml_init(vp);
    }

    // 1. Detect architecture from GGUF metadata
    std::string arch = "llama";  // default fallback
    int idx_arch = gguf_find_key(ctx_gguf, "general.architecture");
    if (idx_arch != -1) {
        arch = gguf_get_val_str(ctx_gguf, idx_arch);
    }

    std::cout << "[DenseCore] Detected architecture: " << arch << std::endl;

    auto get_exact_string = [&](const char* key) -> std::string {
        if (!key) {
            return {};
        }
        const int idx = gguf_find_key(ctx_gguf, key);
        if (idx == -1 || gguf_get_kv_type(ctx_gguf, idx) != GGUF_TYPE_STRING) {
            return {};
        }
        const char* value = gguf_get_val_str(ctx_gguf, idx);
        return value ? value : "";
    };

    auto get_exact_string_array = [&](const char* key) -> std::vector<std::string> {
        std::vector<std::string> values;
        if (!key) {
            return values;
        }
        const int idx = gguf_find_key(ctx_gguf, key);
        if (idx == -1 || gguf_get_kv_type(ctx_gguf, idx) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx_gguf, idx) != GGUF_TYPE_STRING) {
            return values;
        }
        const int n = gguf_get_arr_n(ctx_gguf, idx);
        values.reserve(static_cast<size_t>(std::max(0, n)));
        for (int i = 0; i < n; ++i) {
            const char* value = gguf_get_arr_str(ctx_gguf, idx, i);
            values.emplace_back(value ? value : "");
        }
        return values;
    };

    // Detect tokenizer metadata from GGUF.
    //
    // `tokenizer.ggml.model` describes the base tokenizer family (e.g. gpt2),
    // while `tokenizer.ggml.pre` selects the pre-tokenizer variant (e.g. qwen2).
    // For runtime tokenization behavior we prioritize `tokenizer.ggml.pre` when
    // present because prompt segmentation depends on that field.
    std::string tokenizer_model;
    std::string tokenizer_pre;
    int idx_tokenizer = gguf_find_key(ctx_gguf, "tokenizer.ggml.model");
    if (idx_tokenizer != -1) {
        tokenizer_model = gguf_get_val_str(ctx_gguf, idx_tokenizer);
    }
    int idx_tokenizer_pre = gguf_find_key(ctx_gguf, "tokenizer.ggml.pre");
    if (idx_tokenizer_pre != -1) {
        tokenizer_pre = gguf_get_val_str(ctx_gguf, idx_tokenizer_pre);
    }
    std::string tokenizer_type =
        (!tokenizer_pre.empty() && tokenizer_pre != "default") ? tokenizer_pre : tokenizer_model;
    std::string tokenizer_lower = tokenizer_type;
    std::transform(tokenizer_lower.begin(), tokenizer_lower.end(), tokenizer_lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    int idx_chat_template = gguf_find_key(ctx_gguf, "tokenizer.chat_template");
    std::string chat_template;
    if (idx_chat_template != -1) {
        chat_template = gguf_get_val_str(ctx_gguf, idx_chat_template);
    }

    densecore::models::ModelDetectionHints detection_hints;
    detection_hints.has_gemma4_metadata = gguf_find_key(ctx_gguf, "gemma4.attention.head_count_kv") != -1 ||
                                          gguf_find_key(ctx_gguf, "gemma4.attention.key_length_swa") != -1 ||
                                          gguf_find_key(ctx_gguf, "gemma4.attention.value_length_swa") != -1 ||
                                          gguf_find_key(ctx_gguf, "gemma4.attention.shared_kv_layers") != -1 ||
                                          gguf_find_key(ctx_gguf, "gemma4.embedding_length_per_layer_input") != -1 ||
                                          gguf_find_key(ctx_gguf, "gemma4.layer_types") != -1;
    detection_hints.has_gemma4_tensor_signatures = gguf_find_tensor(ctx_gguf, "per_layer_model_proj.weight") != -1 ||
                                                   gguf_find_tensor(ctx_gguf, "per_layer_proj_norm.weight") != -1 ||
                                                   gguf_find_tensor(ctx_gguf, "per_layer_token_embd.weight") != -1 ||
                                                   gguf_find_tensor(ctx_gguf, "blk.0.inp_gate.weight") != -1;
    detection_hints.has_gemma4_tokenizer_hint = tokenizer_lower.find("gemma4") != std::string::npos;

    bool used_hint_upgrade = false;
    auto resolved_arch = densecore::models::ResolveModelDescriptorWithHints(arch, detection_hints, &used_hint_upgrade);

    const auto ascii_lower_copy = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    };
    const auto contains_lower = [&](const std::string& haystack, const char* needle) {
        if (!needle || !*needle) {
            return false;
        }
        return ascii_lower_copy(haystack).find(needle) != std::string::npos;
    };

    const std::string general_name = get_exact_string("general.name");
    const std::string general_basename = get_exact_string("general.basename");
    const std::string general_repo = get_exact_string("general.repo");
    const std::string general_hf_repo = get_exact_string("general.hf_repo_id");
    const std::string hf_repo = get_exact_string("hf.repo_id");
    const std::string hf_model_type = !get_exact_string("hf.model_type").empty() ? get_exact_string("hf.model_type")
                                      : !get_exact_string("general.hf_model_type").empty()
                                          ? get_exact_string("general.hf_model_type")
                                          : get_exact_string("transformers.model_type");
    std::vector<std::string> hf_architectures = get_exact_string_array("hf.architectures");
    if (hf_architectures.empty()) hf_architectures = get_exact_string_array("general.hf_architectures");
    if (hf_architectures.empty()) hf_architectures = get_exact_string_array("transformers.architectures");
    if (hf_architectures.empty()) {
        const std::string single_arch = get_exact_string("hf.architecture");
        if (!single_arch.empty()) {
            hf_architectures.push_back(single_arch);
        }
    }

    const std::string arch_lower = ascii_lower_copy(arch);
    const bool qwen35_architecture = arch_lower == "qwen35" || arch_lower == "qwen35moe";
    const std::string hf_model_type_lower = ascii_lower_copy(hf_model_type);
    const bool qwen35_hf_model_type = hf_model_type_lower == "qwen3_5_moe" || hf_model_type_lower == "qwen3_5_text";
    bool qwen35_hf_architecture = false;
    for (const auto& value : hf_architectures) {
        const std::string lowered = ascii_lower_copy(value);
        if (lowered == "qwen3_5moeforconditionalgeneration" || lowered == "qwen3_5forconditionalgeneration") {
            qwen35_hf_architecture = true;
            break;
        }
    }
    const bool qwen36_named_variant = contains_lower(general_name, "qwen3.6") ||
                                      contains_lower(general_basename, "qwen3.6") ||
                                      contains_lower(general_repo, "qwen3.6") ||
                                      contains_lower(general_hf_repo, "qwen3.6") || contains_lower(hf_repo, "qwen3.6");
    const bool qwen35_family_aux =
        qwen35_architecture || qwen35_hf_model_type || qwen35_hf_architecture || qwen36_named_variant;
    std::string qwen35_resolution_signal;
    if (qwen35_architecture) {
        qwen35_resolution_signal =
            arch_lower == "qwen35" ? "general.architecture=qwen35" : "general.architecture=qwen35moe";
    } else if (qwen35_hf_model_type) {
        qwen35_resolution_signal = "hf.model_type=" + hf_model_type;
    } else if (qwen35_hf_architecture) {
        qwen35_resolution_signal = "hf.architectures contains Qwen3_5*ForConditionalGeneration";
    } else if (contains_lower(general_name, "qwen3.6")) {
        qwen35_resolution_signal = "general.name contains Qwen3.6";
    } else if (contains_lower(general_basename, "qwen3.6")) {
        qwen35_resolution_signal = "general.basename contains Qwen3.6";
    } else if (contains_lower(general_repo, "qwen3.6")) {
        qwen35_resolution_signal = "general.repo contains Qwen3.6";
    } else if (contains_lower(general_hf_repo, "qwen3.6")) {
        qwen35_resolution_signal = "general.hf_repo_id contains Qwen3.6";
    } else if (contains_lower(hf_repo, "qwen3.6")) {
        qwen35_resolution_signal = "hf.repo_id contains Qwen3.6";
    }
    if (!resolved_arch.known && qwen35_family_aux) {
        resolved_arch = densecore::models::ResolveModelDescriptor(qwen36_named_variant ? "qwen36" : "qwen35");
    } else if (resolved_arch.known && resolved_arch.arch == ModelArch::QWEN35 && qwen36_named_variant) {
        resolved_arch = densecore::models::ResolveModelDescriptor("qwen36");
    }

    model->arch = resolved_arch.arch;
    model->variant = resolved_arch.variant;
    model->arch_flags = resolved_arch.arch_flags;
    if (!resolved_arch.known) {
        model->arch = ModelArch::UNKNOWN;
        model->variant = ModelVariant::UNKNOWN;
        std::cerr << "[DenseCore] Warning: Unknown architecture '" << arch << "'. Model may not load correctly."
                  << std::endl;
    } else if (used_hint_upgrade) {
        std::cerr << "[DenseCore] Warning: architecture '" << arch
                  << "' classified as Gemma4 from auxiliary metadata/tensor signatures" << std::endl;
    }
    if (qwen35_family_aux && model->arch == ModelArch::QWEN35 && !qwen35_resolution_signal.empty()) {
        std::cout << "[DenseCore] Resolved " << densecore::models::ModelVariantName(model->variant)
                  << " hybrid-SSM family via " << qwen35_resolution_signal << std::endl;
    }

    model->tokenizer_type = tokenizer_type;
    model->chat_template = std::move(chat_template);
    const int gemma4_head_count_kv_idx = gguf_find_key(ctx_gguf, "gemma4.attention.head_count_kv");
    const gguf_type gemma4_head_count_kv_type =
        gemma4_head_count_kv_idx != -1 ? gguf_get_kv_type(ctx_gguf, gemma4_head_count_kv_idx) : GGUF_TYPE_UINT8;
    const bool has_gemma4_kv_array = gemma4_head_count_kv_type == GGUF_TYPE_ARRAY;

    if (!tokenizer_type.empty()) {
        if (!densecore::models::IsKnownTokenizerModel(tokenizer_lower)) {
            std::cerr << "[DenseCore] Warning: tokenizer model '" << tokenizer_type
                      << "' may not be fully compatible. Consider using external tokenization and input_ids."
                      << std::endl;
        }
    }

    auto find_prefixed_key = [&](const std::string& suffix) -> int {
        std::vector<std::string> prefixes;
        prefixes.reserve(8);
        if (!arch.empty()) {
            prefixes.push_back(arch);
        }
        if (model->arch_flags.is_hybrid_ssm) {
            static const char* kHybridPrefixes[] = {"qwen35moe",    "qwen3_5_moe", "qwen3_5_moe_text",
                                                    "qwen3_5_text", "qwen35",      "qwen36"};
            for (const char* prefix : kHybridPrefixes) {
                if (!prefix) continue;
                if (std::find(prefixes.begin(), prefixes.end(), prefix) == prefixes.end()) {
                    prefixes.emplace_back(prefix);
                }
            }
        }
        if (model->arch_flags.is_lfm2_shortconv) {
            static const char* kLFM2Prefixes[] = {"lfm2moe", "lfm2"};
            for (const char* prefix : kLFM2Prefixes) {
                if (std::find(prefixes.begin(), prefixes.end(), prefix) == prefixes.end()) {
                    prefixes.emplace_back(prefix);
                }
            }
        }
        for (const auto& prefix : prefixes) {
            std::string key = prefix + "." + suffix;
            int idx = gguf_find_key(ctx_gguf, key.c_str());
            if (idx != -1) {
                return idx;
            }
        }
        if (model->arch_flags.is_gemma4 && arch != "gemma4") {
            const std::string key = "gemma4." + suffix;
            const int idx = gguf_find_key(ctx_gguf, key.c_str());
            if (idx != -1) {
                return idx;
            }
        }
        const std::string key = "general." + suffix;
        return gguf_find_key(ctx_gguf, key.c_str());
    };

    // 2. Generic parameter loader using architecture prefix
    auto get_u32 = [&](const std::string& suffix, uint32_t& val) {
        const int idx = find_prefixed_key(suffix);
        if (idx == -1) {
            return;
        }

        const gguf_type type = gguf_get_kv_type(ctx_gguf, idx);
        if (type == GGUF_TYPE_UINT32) {
            val = gguf_get_val_u32(ctx_gguf, idx);
            return;
        }
        if (type == GGUF_TYPE_INT32) {
            val = static_cast<uint32_t>(gguf_get_val_i32(ctx_gguf, idx));
            return;
        }
        if (type == GGUF_TYPE_ARRAY) {
            const int n = gguf_get_arr_n(ctx_gguf, idx);
            const gguf_type arr_type = gguf_get_arr_type(ctx_gguf, idx);
            const void* arr_data = gguf_get_arr_data(ctx_gguf, idx);
            if (!arr_data || n <= 0) {
                return;
            }
            if (arr_type == GGUF_TYPE_UINT32) {
                val = static_cast<const uint32_t*>(arr_data)[0];
                return;
            }
            if (arr_type == GGUF_TYPE_INT32) {
                val = static_cast<uint32_t>(static_cast<const int32_t*>(arr_data)[0]);
                return;
            }
        }
    };

    auto get_f32 = [&](const std::string& suffix, float& val) {
        const int idx = find_prefixed_key(suffix);
        if (idx != -1) {
            val = gguf_get_val_f32(ctx_gguf, idx);
        }
    };

    auto get_i32_arr4 = [&](const std::string& suffix, std::array<int32_t, 4>& vals) {
        const int idx = find_prefixed_key(suffix);
        if (idx == -1) {
            return;
        }
        const int n = gguf_get_arr_n(ctx_gguf, idx);
        const gguf_type arr_type = gguf_get_arr_type(ctx_gguf, idx);
        const void* arr_data = gguf_get_arr_data(ctx_gguf, idx);
        if (!arr_data || n < 4) {
            return;
        }
        if (arr_type == GGUF_TYPE_INT32) {
            const int32_t* p = reinterpret_cast<const int32_t*>(arr_data);
            for (int i = 0; i < 4; ++i) vals[static_cast<size_t>(i)] = p[i];
        } else if (arr_type == GGUF_TYPE_UINT32) {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(arr_data);
            for (int i = 0; i < 4; ++i) vals[static_cast<size_t>(i)] = static_cast<int32_t>(p[i]);
        }
    };

    auto get_bool = [&](const std::string& suffix, bool& val) {
        const int idx = find_prefixed_key(suffix);
        if (idx != -1) {
            val = gguf_get_val_bool(ctx_gguf, idx);
        }
    };

    auto get_str_array = [&](const std::string& suffix, std::vector<std::string>& vals) {
        const int idx = find_prefixed_key(suffix);
        if (idx == -1 || gguf_get_arr_type(ctx_gguf, idx) != GGUF_TYPE_STRING) {
            return;
        }
        const int n = gguf_get_arr_n(ctx_gguf, idx);
        vals.clear();
        vals.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const char* str = gguf_get_arr_str(ctx_gguf, idx, i);
            vals.emplace_back(str ? str : "");
        }
    };

    auto get_u8_array = [&](const std::string& suffix, std::vector<uint8_t>& vals) {
        const int idx = find_prefixed_key(suffix);
        if (idx == -1 || gguf_get_kv_type(ctx_gguf, idx) != GGUF_TYPE_ARRAY) {
            return;
        }
        const gguf_type arr_type = gguf_get_arr_type(ctx_gguf, idx);
        const int n = gguf_get_arr_n(ctx_gguf, idx);
        const void* raw = gguf_get_arr_data(ctx_gguf, idx);
        if (!raw || n <= 0 || (arr_type != GGUF_TYPE_UINT8 && arr_type != GGUF_TYPE_BOOL)) {
            return;
        }
        vals.resize(static_cast<size_t>(n));
        if (arr_type == GGUF_TYPE_UINT8) {
            const auto* src = static_cast<const uint8_t*>(raw);
            std::copy(src, src + n, vals.begin());
        } else {
            const auto* src = static_cast<const int8_t*>(raw);
            for (int i = 0; i < n; ++i) {
                vals[static_cast<size_t>(i)] = src[i] ? 1u : 0u;
            }
        }
    };

    auto has_key = [&](const std::string& suffix) -> bool { return find_prefixed_key(suffix) != -1; };
    auto has_exact_key = [&](const char* key) -> bool { return gguf_find_key(ctx_gguf, key) != -1; };

    // Load hyperparameters using dynamic architecture prefix
    get_u32("vocab_size", model->hparams.n_vocab);
    get_u32("embedding_length", model->hparams.n_embd);
    get_u32("block_count", model->hparams.n_layer);
    get_u32("attention.head_count", model->hparams.n_head);
    get_u32("attention.head_count_kv", model->hparams.n_head_kv);
    get_u32("context_length", model->hparams.n_ctx);

    if (model->hparams.n_head_kv == 0) model->hparams.n_head_kv = model->hparams.n_head;
    model->gemma4_layer_n_head_kv.clear();
    if (has_gemma4_kv_array) {
        const int idx = gemma4_head_count_kv_idx;
        const gguf_type arr_type = gguf_get_arr_type(ctx_gguf, idx);
        if (idx != -1 && (arr_type == GGUF_TYPE_INT32 || arr_type == GGUF_TYPE_UINT32)) {
            model->gemma4_layer_n_head_kv.assign(model->hparams.n_layer, model->hparams.n_head_kv);
            const int n = gguf_get_arr_n(ctx_gguf, idx);
            const void* raw = gguf_get_arr_data(ctx_gguf, idx);
            if (raw && n > 0) {
                const size_t limit = std::min<size_t>(model->gemma4_layer_n_head_kv.size(), static_cast<size_t>(n));
                for (size_t i = 0; i < limit; ++i) {
                    const uint32_t kv = arr_type == GGUF_TYPE_UINT32
                                            ? static_cast<const uint32_t*>(raw)[i]
                                            : static_cast<uint32_t>(static_cast<const int32_t*>(raw)[i]);
                    if (kv > 0) {
                        model->gemma4_layer_n_head_kv[i] = kv;
                    }
                }
            }
        }
    }

    // llama.cpp style: Load head dimensions from GGUF.
    //
    // Gemma4 uses per-layer KV-head metadata and exposes key/value lengths
    // that do not match the current runtime reshape path. Leave those at 0
    // here so the later weight-shape inference can derive a consistent head
    // dimension from the actual attention tensors.
    if (!model->arch_flags.is_gemma4) {
        get_u32("attention.key_length", model->hparams.n_embd_head_k);
        get_u32("attention.value_length", model->hparams.n_embd_head_v);
    }

    // Resolve n_rot (rotary embedding dimension).
    //
    // Priority:
    //   1) explicit rope.dimension_count (or general fallback)
    //   2) explicit attention.rotary_dim (legacy exporters)
    //   3) attention.key_length (common for GQA models, e.g. Qwen3)
    //   4) n_embd / n_head fallback
    //
    // This keeps partial-RoPE support while avoiding incorrect defaults on
    // architectures where key/query head dims differ from n_embd/n_head.
    const bool has_rope_dimension_count = has_key("rope.dimension_count");
    const bool has_attention_rotary_dim = has_key("attention.rotary_dim");
    const uint32_t fallback_rot =
        (model->hparams.n_embd_head_k > 0)
            ? model->hparams.n_embd_head_k
            : ((model->hparams.n_head > 0) ? (model->hparams.n_embd / model->hparams.n_head) : 0u);
    model->hparams.n_rot = fallback_rot;
    if (has_rope_dimension_count) {
        get_u32("rope.dimension_count", model->hparams.n_rot);
    } else if (has_attention_rotary_dim) {
        get_u32("attention.rotary_dim", model->hparams.n_rot);
    }
    if (model->hparams.n_rot == 0) {
        model->hparams.n_rot = fallback_rot;
    }

    // Load norm epsilon. Decoder models generally expose RMSNorm epsilon;
    // encoder-only BERT GGUFs expose standard LayerNorm epsilon.
    model->hparams.f_norm_rms_eps = 1e-5f;
    get_f32("attention.layer_norm_rms_epsilon", model->hparams.f_norm_rms_eps);
    if (model->arch == ModelArch::BERT) {
        get_f32("attention.layer_norm_epsilon", model->hparams.f_norm_rms_eps);
    }

    std::cout << "[DenseCore] Loaded RMS norm epsilon: " << model->hparams.f_norm_rms_eps << std::endl;

    // Load RoPE parameters
    get_f32("attention.scale", model->hparams.f_attention_scale);
    if (model->arch_flags.is_gemma4 && !(model->hparams.f_attention_scale > 0.0f)) {
        // Gemma4 uses self.scaling = 1.0. When GGUF omits attention.scale,
        // keep the fast attention kernels on the same no-pre-scale semantics
        // instead of falling back to the generic 1/sqrt(head_dim) scale.
        model->hparams.f_attention_scale = 1.0f;
    }
    get_f32("rope.freq_base", model->hparams.rope_freq_base);
    get_f32("rope.freq_scale", model->hparams.rope_freq_scale);
    get_i32_arr4("rope.dimension_sections", model->hparams.rope_sections);
    get_bool("rope.mrope_interleaved", model->hparams.rope_mrope_interleaved);
    get_bool("rope.scaling.mrope_interleaved", model->hparams.rope_mrope_interleaved);
    get_bool("mrope_interleaved", model->hparams.rope_mrope_interleaved);
    if (!model->hparams.rope_mrope_interleaved && model->arch_flags.is_hybrid_ssm &&
        (model->hparams.rope_sections[0] > 0 || model->hparams.rope_sections[1] > 0)) {
        model->hparams.rope_mrope_interleaved = true;
        std::cout << "[DenseCore] Defaulting hybrid-SSM MRoPE to interleaved mode" << std::endl;
    }

    if (model->arch_flags.is_gemma4) {
        model->gemma4_full_attention_partial_rotary_factor = 0.25f;
        model->gemma4_rope_freq_base_full = model->hparams.rope_freq_base;
        model->gemma4_rope_freq_base_swa = model->hparams.rope_freq_base;
        model->gemma4_rope_dim_full = static_cast<int>(model->hparams.n_rot);
        model->gemma4_rope_dim_swa = static_cast<int>(model->hparams.n_rot);
        model->gemma4_key_length_full = 0;
        model->gemma4_value_length_full = 0;
        model->gemma4_key_length_swa = 0;
        model->gemma4_value_length_swa = 0;

        get_f32("rope.freq_base", model->gemma4_rope_freq_base_full);
        get_f32("rope.freq_base_swa", model->gemma4_rope_freq_base_swa);

        uint32_t tmp_u32 = static_cast<uint32_t>(model->gemma4_rope_dim_full);
        get_u32("rope.dimension_count", tmp_u32);
        model->gemma4_rope_dim_full = static_cast<int>(tmp_u32);
        tmp_u32 = static_cast<uint32_t>(model->gemma4_rope_dim_swa > 0 ? model->gemma4_rope_dim_swa
                                                                       : model->gemma4_rope_dim_full);
        get_u32("rope.dimension_count_swa", tmp_u32);
        model->gemma4_rope_dim_swa = static_cast<int>(tmp_u32);

        get_u32("attention.key_length", model->gemma4_key_length_full);
        get_u32("attention.value_length", model->gemma4_value_length_full);
        model->gemma4_key_length_swa = model->gemma4_key_length_full;
        model->gemma4_value_length_swa = model->gemma4_value_length_full;
        get_u32("attention.key_length_swa", model->gemma4_key_length_swa);
        get_u32("attention.value_length_swa", model->gemma4_value_length_swa);

        tmp_u32 = 0;
        get_u32("attention.sliding_window", tmp_u32);
        model->gemma4_sliding_window = static_cast<int>(tmp_u32);

        tmp_u32 = 0;
        get_u32("attention.shared_kv_layers", tmp_u32);
        model->gemma4_n_shared_kv_layers = static_cast<int>(tmp_u32);

        tmp_u32 = 0;
        get_u32("embedding_length_per_layer_input", tmp_u32);
        model->gemma4_hidden_size_per_layer_input = static_cast<int>(tmp_u32);

        float gemma4_attention_logit_cap = 50.0f;
        const bool has_attention_logit_cap = has_key("attention_logit_cap");
        if (has_attention_logit_cap) {
            get_f32("attention_logit_cap", gemma4_attention_logit_cap);
        }
        model->gemma4_attention_logit_softcapping =
            (has_attention_logit_cap && gemma4_attention_logit_cap == 0.0f)
                ? 0.0f
                : densecore::models::SanitizeAttentionLogitSoftcapForLoad(model, gemma4_attention_logit_cap);
        get_f32("final_logit_softcapping", model->gemma4_final_logit_softcapping);

        std::vector<uint8_t> sliding_pattern;
        get_u8_array("attention.sliding_window_pattern", sliding_pattern);
        model->gemma4_layer_is_sliding.assign(model->hparams.n_layer, 0);
        bool sliding_pattern_complete = false;
        if (sliding_pattern.size() >= static_cast<size_t>(model->hparams.n_layer)) {
            for (size_t i = 0; i < static_cast<size_t>(model->hparams.n_layer); ++i) {
                model->gemma4_layer_is_sliding[i] = sliding_pattern[i] != 0 ? 1 : 0;
            }
            sliding_pattern_complete = true;
        }
        if (!sliding_pattern_complete) {
            // Try layer_types metadata
            std::vector<std::string> layer_types;
            get_str_array("layer_types", layer_types);
            if (layer_types.size() >= static_cast<size_t>(model->hparams.n_layer)) {
                for (size_t i = 0; i < static_cast<size_t>(model->hparams.n_layer); ++i) {
                    std::string type = layer_types[i];
                    std::transform(type.begin(), type.end(), type.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    model->gemma4_layer_is_sliding[i] = (type == "sliding_attention") ? 1 : 0;
                }
                sliding_pattern_complete = true;
            }
        }
        if (!sliding_pattern_complete) {
            return fail_load("Gemma4 GGUF is missing a complete layer_types/sliding_window_pattern export");
        }

        model->gemma4_layer_kv_source.assign(model->hparams.n_layer, -1);
        const int first_shared_layer =
            std::max(0, static_cast<int>(model->hparams.n_layer) - std::max(0, model->gemma4_n_shared_kv_layers));
        int last_non_shared_sliding = -1;
        int last_non_shared_full = -1;
        for (int i = 0; i < static_cast<int>(model->hparams.n_layer); ++i) {
            const bool is_sliding = i < static_cast<int>(model->gemma4_layer_is_sliding.size()) &&
                                    model->gemma4_layer_is_sliding[static_cast<size_t>(i)] != 0;
            if (i < first_shared_layer) {
                if (is_sliding) {
                    last_non_shared_sliding = i;
                } else {
                    last_non_shared_full = i;
                }
                model->gemma4_layer_kv_source[static_cast<size_t>(i)] = i;
            } else {
                model->gemma4_layer_kv_source[static_cast<size_t>(i)] =
                    is_sliding ? last_non_shared_sliding : last_non_shared_full;
                if (model->gemma4_layer_kv_source[static_cast<size_t>(i)] < 0) {
                    return fail_load("Gemma4 shared-KV metadata is inconsistent for at least one layer");
                }
            }
        }

        std::cout << "[DenseCore] Gemma4 metadata:" << " rope_dim_full=" << model->gemma4_rope_dim_full
                  << " rope_dim_swa=" << model->gemma4_rope_dim_swa << " key_len_full=" << model->gemma4_key_length_full
                  << " key_len_swa=" << model->gemma4_key_length_swa
                  << " value_len_full=" << model->gemma4_value_length_full
                  << " value_len_swa=" << model->gemma4_value_length_swa
                  << " sliding_window=" << model->gemma4_sliding_window
                  << " shared_kv_layers=" << model->gemma4_n_shared_kv_layers
                  << " attn_logit_cap=" << model->gemma4_attention_logit_softcapping
                  << " final_logit_softcap=" << model->gemma4_final_logit_softcapping << std::endl;
        std::cout << "[DenseCore] Gemma4 layer pattern:";
        for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
            const bool is_sliding = i < model->gemma4_layer_is_sliding.size() && model->gemma4_layer_is_sliding[i] != 0;
            const int kv_src = i < model->gemma4_layer_kv_source.size() ? model->gemma4_layer_kv_source[i] : -1;
            std::cout << " L" << i << "=" << (is_sliding ? "S" : "F") << "/src" << kv_src;
        }
        std::cout << std::endl;
    }

    // Load MoE parameters when present.
    if (model->arch_flags.is_glm_moe || has_key("n_routed_experts") || has_key("num_experts_per_tok") ||
        has_key("expert_count") || has_key("expert_used_count") || has_key("enable_moe_block") ||
        has_key("num_experts") || has_key("top_k_experts")) {
        uint32_t tmp = 0;

        tmp = model->hparams.n_experts;
        get_u32("n_routed_experts", tmp);
        if (tmp == 0) get_u32("num_local_experts", tmp);
        if (tmp == 0) get_u32("expert_count", tmp);  // qwen35moe naming
        if (tmp == 0) get_u32("num_experts", tmp);   // gemma4 / transformers naming
        model->hparams.n_experts = tmp;

        tmp = model->hparams.n_experts_used;
        get_u32("num_experts_per_tok", tmp);
        if (tmp == 0) get_u32("n_experts_used", tmp);
        if (tmp == 0) get_u32("expert_used_count", tmp);  // qwen35moe naming
        if (tmp == 0) get_u32("top_k_experts", tmp);      // gemma4 naming
        model->hparams.n_experts_used = tmp;

        tmp = static_cast<uint32_t>(model->moe_n_shared_experts);
        get_u32("n_shared_experts", tmp);
        model->moe_n_shared_experts = static_cast<int>(tmp);

        tmp = static_cast<uint32_t>(model->moe_n_group);
        get_u32("n_group", tmp);
        model->moe_n_group = static_cast<int>(tmp);

        tmp = static_cast<uint32_t>(model->moe_topk_group);
        get_u32("topk_group", tmp);
        model->moe_topk_group = static_cast<int>(tmp);

        tmp = static_cast<uint32_t>(model->moe_first_k_dense_replace);
        get_u32("first_k_dense_replace", tmp);
        model->moe_first_k_dense_replace = static_cast<int>(tmp);

        get_f32("routed_scaling_factor", model->moe_routed_scaling_factor);
        get_bool("norm_topk_prob", model->moe_norm_topk_prob);
    }

    if (model->arch_flags.is_glm_dsa) {
        uint32_t tmp = 0;

        tmp = static_cast<uint32_t>(model->glm_q_lora_rank);
        get_u32("q_lora_rank", tmp);
        model->glm_q_lora_rank = static_cast<int>(tmp);

        tmp = static_cast<uint32_t>(model->glm_kv_lora_rank);
        get_u32("kv_lora_rank", tmp);
        model->glm_kv_lora_rank = static_cast<int>(tmp);

        tmp = static_cast<uint32_t>(model->glm_qk_rope_head_dim);
        get_u32("qk_rope_head_dim", tmp);
        model->glm_qk_rope_head_dim = static_cast<int>(tmp);

        tmp = static_cast<uint32_t>(model->glm_qk_nope_head_dim);
        get_u32("qk_nope_head_dim", tmp);
        model->glm_qk_nope_head_dim = static_cast<int>(tmp);

        tmp = static_cast<uint32_t>(model->glm_v_head_dim);
        get_u32("v_head_dim", tmp);
        model->glm_v_head_dim = static_cast<int>(tmp);

        tmp = static_cast<uint32_t>(model->glm_index_topk);
        get_u32("index_topk", tmp);
        model->glm_index_topk = static_cast<int>(tmp);

        tmp = static_cast<uint32_t>(model->glm_index_head_dim);
        get_u32("index_head_dim", tmp);
        model->glm_index_head_dim = static_cast<int>(tmp);

        tmp = static_cast<uint32_t>(model->glm_index_n_heads);
        get_u32("index_n_heads", tmp);
        model->glm_index_n_heads = static_cast<int>(tmp);

        // Validate required GLM-5 DSA parameters
        if (model->glm_kv_lora_rank <= 0 || model->glm_v_head_dim <= 0 || model->glm_index_head_dim <= 0 ||
            model->glm_index_n_heads <= 0) {
            fprintf(stderr,
                    "[ModelLoader] Warning: GLM-5 DSA model has missing required parameters "
                    "(kv_lora_rank=%d, v_head_dim=%d, index_head_dim=%d, index_n_heads=%d). "
                    "DSA attention will be disabled and inference may be incorrect.\n",
                    model->glm_kv_lora_rank, model->glm_v_head_dim, model->glm_index_head_dim,
                    model->glm_index_n_heads);
        }
    }

    // Load SSM parameters for hybrid models (Qwen3.5, Jamba, etc.)
    if (model->arch_flags.is_hybrid_ssm) {
        uint32_t tmp = 0;
        const bool has_full_attention_interval_key = has_key("full_attention_interval");
        tmp = model->ssm_conv_kernel;
        get_u32("ssm.conv_kernel", tmp);
        model->ssm_conv_kernel = static_cast<int>(tmp);
        tmp = model->ssm_state_size;
        get_u32("ssm.state_size", tmp);
        model->ssm_state_size = static_cast<int>(tmp);
        tmp = model->ssm_group_count;
        get_u32("ssm.group_count", tmp);
        model->ssm_group_count = static_cast<int>(tmp);
        tmp = model->ssm_time_step_rank;
        get_u32("ssm.time_step_rank", tmp);
        model->ssm_time_step_rank = static_cast<int>(tmp);
        tmp = model->ssm_inner_size;
        get_u32("ssm.inner_size", tmp);
        model->ssm_inner_size = static_cast<int>(tmp);
        tmp = 0;
        if (has_full_attention_interval_key) {
            get_u32("full_attention_interval", tmp);
        }
        model->ssm_full_attn_interval = static_cast<int>(tmp);

        const auto fail_hybrid_ssm = [&](const std::string& reason) {
            return fail_load("invalid hybrid SSM configuration: " + reason);
        };
        if (model->ssm_inner_size <= 0 || model->ssm_state_size <= 0 || model->ssm_group_count <= 0 ||
            model->ssm_time_step_rank <= 0 || model->ssm_conv_kernel <= 0) {
            return fail_hybrid_ssm("non-positive SSM hyperparameter");
        }
        if (model->ssm_inner_size % model->ssm_time_step_rank != 0) {
            return fail_hybrid_ssm("ssm_inner_size must be divisible by ssm_time_step_rank");
        }
        if (model->ssm_time_step_rank % model->ssm_group_count != 0) {
            return fail_hybrid_ssm("ssm_time_step_rank must be divisible by ssm_group_count");
        }

        std::vector<std::string> layer_types;
        get_str_array("layer_types", layer_types);
        if (!layer_types.empty() && layer_types.size() != static_cast<size_t>(model->hparams.n_layer)) {
            return fail_hybrid_ssm("partial hybrid layer_types export");
        }
        if (!layer_types.empty()) {
            model->hybrid_layer_is_ssm.assign(model->hparams.n_layer, 0);
            for (size_t i = 0; i < static_cast<size_t>(model->hparams.n_layer); ++i) {
                std::string type = layer_types[i];
                std::transform(type.begin(), type.end(), type.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                const bool is_full_attn = (type == "full_attention" || type == "attention" || type == "transformer" ||
                                           type == "self_attention");
                const bool is_linear_attn =
                    (type == "linear_attention" || type == "linear_attn" || type == "ssm" || type == "gated_deltanet");
                if (!is_full_attn && !is_linear_attn) {
                    return fail_hybrid_ssm("unknown layer_types entry '" + layer_types[i] + "'");
                }
                model->hybrid_layer_is_ssm[i] = is_full_attn ? 0 : 1;
            }
            if (model->ssm_full_attn_interval > 0) {
                for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
                    const bool expected_ssm =
                        (i % model->ssm_full_attn_interval) != static_cast<uint32_t>(model->ssm_full_attn_interval - 1);
                    if ((model->hybrid_layer_is_ssm[i] != 0) != expected_ssm) {
                        return fail_hybrid_ssm("layer_types schedule disagrees with full_attention_interval");
                    }
                }
            }
        } else if (model->arch == ModelArch::QWEN35 && has_full_attention_interval_key &&
                   model->ssm_full_attn_interval == 4) {
            int full_attention_phase = 3;
            if (model->variant == ModelVariant::QWEN36) {
                const char* env = std::getenv("DENSECORE_QWEN36_FULL_ATTN_PHASE");
                if (env && env[0] != '\0') {
                    char* end = nullptr;
                    const long parsed = std::strtol(env, &end, 10);
                    if (end != env && end && *end == '\0') {
                        full_attention_phase = static_cast<int>((parsed % 4 + 4) % 4);
                    }
                }
            }
            model->hybrid_layer_is_ssm.assign(model->hparams.n_layer, 1);
            for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
                if (static_cast<int>(i % 4) == full_attention_phase) {
                    model->hybrid_layer_is_ssm[i] = 0;
                }
            }
            std::cout << "[DenseCore] Reconstructed hybrid layer schedule from full_attention_interval="
                      << model->ssm_full_attn_interval << " because layer_types metadata is missing"
                      << " (full_attention_phase=" << full_attention_phase << ")" << std::endl;
        } else {
            return fail_hybrid_ssm("missing reliable layer_types/full_attention_interval metadata");
        }

        // Initialize SSM runtime states
        const int conv_channels = model->ssm_inner_size + 2 * model->ssm_group_count * model->ssm_state_size;
        const int head_dim = model->ssm_inner_size / model->ssm_time_step_rank;
        if (conv_channels <= 0 || head_dim <= 0) {
            return fail_hybrid_ssm("derived conv_channels/head_dim must be positive");
        }
        int n_ssm_layers = 0;
        int n_full_attn_layers = 0;
        for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
            if (model->IsHybridSSMLayer(static_cast<int>(i))) {
                n_ssm_layers++;
            } else {
                n_full_attn_layers++;
            }
        }
        if (n_ssm_layers == 0 || n_full_attn_layers == 0) {
            return fail_hybrid_ssm("hybrid schedule must contain both SSM and full-attention layers");
        }
        model->ssm_layer_states.resize(n_ssm_layers);
        for (auto& s : model->ssm_layer_states) {
            s.Init(conv_channels, model->ssm_conv_kernel, model->ssm_time_step_rank, head_dim, model->ssm_state_size);
        }

        std::cout << "[DenseCore] SSM params: conv_kernel=" << model->ssm_conv_kernel
                  << " state_size=" << model->ssm_state_size << " groups=" << model->ssm_group_count
                  << " n_heads=" << model->ssm_time_step_rank << " inner=" << model->ssm_inner_size
                  << " attn_interval=" << model->ssm_full_attn_interval << " n_ssm_layers=" << n_ssm_layers
                  << std::endl;
        std::cout << "[DenseCore] Hybrid schedule summary: ssm_layers=" << n_ssm_layers
                  << " attention_layers=" << n_full_attn_layers << std::endl;
        if (model->variant == ModelVariant::QWEN36) {
            struct Qwen36ExpectedProfile {
                const char* name = "dense-27b";
                int hidden_size = 5120;
                int n_layer = 64;
                int full_attention_interval = 4;
                int attn_heads = 24;
                int kv_heads = 4;
                int head_dim = 128;
                int linear_key_heads = 16;
                int linear_value_heads = 48;
                int linear_key_dim = 128;
                int linear_value_dim = 128;
                int conv_kernel = 4;
                int experts = 0;
                int experts_per_token = 0;
                int shared_experts = 0;
                bool shared_experts_must_be_positive = false;
            };
            const bool qwen36_dense_profile =
                model->hparams.n_experts == 0 && model->hparams.n_experts_used == 0 && model->moe_n_shared_experts == 0;
            const Qwen36ExpectedProfile expected = [&]() {
                Qwen36ExpectedProfile profile;
                if (qwen36_dense_profile) {
                    return profile;
                }
                profile.name = "moe-35b-a3b";
                profile.hidden_size = 2048;
                profile.n_layer = 40;
                profile.full_attention_interval = 4;
                profile.attn_heads = 16;
                profile.kv_heads = 2;
                profile.head_dim = 256;
                profile.linear_key_heads = 16;
                profile.linear_value_heads = 32;
                profile.linear_key_dim = 128;
                profile.linear_value_dim = 128;
                profile.conv_kernel = 4;
                profile.experts = 256;
                profile.experts_per_token = 8;
                profile.shared_experts = 1;
                profile.shared_experts_must_be_positive = true;
                return profile;
            }();

            const bool strict_qwen36_profile = []() {
                const char* env = std::getenv("DENSECORE_QWEN36_STRICT_PROFILE");
                return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
            }();
            const bool allow_qwen36_test_profile = []() {
                const char* env = std::getenv("DENSECORE_QWEN36_ALLOW_TEST_PROFILE");
                return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
            }();

            std::vector<std::string> mismatches;
            if (static_cast<int>(model->hparams.n_embd) != expected.hidden_size) {
                mismatches.push_back("hidden_size");
            }
            if (static_cast<int>(model->hparams.n_layer) != expected.n_layer) {
                mismatches.push_back("n_layer");
            }
            if (model->ssm_full_attn_interval != expected.full_attention_interval) {
                mismatches.push_back("full_attention_interval");
            }
            if (static_cast<int>(model->hparams.n_head) != expected.attn_heads) {
                mismatches.push_back("attention.head_count");
            }
            if (static_cast<int>(model->hparams.n_head_kv) != expected.kv_heads) {
                mismatches.push_back("attention.head_count_kv");
            }
            if (static_cast<int>(model->hparams.n_embd_head_k) != expected.head_dim) {
                mismatches.push_back("attention.key_length/head_dim");
            }
            if (model->ssm_group_count != expected.linear_key_heads) {
                mismatches.push_back("linear_num_key_heads");
            }
            if (model->ssm_time_step_rank != expected.linear_value_heads) {
                mismatches.push_back("linear_num_value_heads");
            }
            if ((model->ssm_group_count > 0 ? model->ssm_state_size : 0) != expected.linear_key_dim) {
                mismatches.push_back("linear_key_head_dim");
            }
            if ((model->ssm_time_step_rank > 0 ? model->ssm_inner_size / model->ssm_time_step_rank : 0) !=
                expected.linear_value_dim) {
                mismatches.push_back("linear_value_head_dim");
            }
            if (model->ssm_conv_kernel != expected.conv_kernel) {
                mismatches.push_back("linear_conv_kernel_dim");
            }
            if (static_cast<int>(model->hparams.n_experts) != expected.experts) {
                mismatches.push_back("num_experts");
            }
            if (static_cast<int>(model->hparams.n_experts_used) != expected.experts_per_token) {
                mismatches.push_back("num_experts_per_tok");
            }
            if (expected.shared_experts_must_be_positive) {
                if (model->moe_n_shared_experts <= 0) {
                    mismatches.push_back("shared_expert");
                }
            } else if (model->moe_n_shared_experts != expected.shared_experts) {
                mismatches.push_back("shared_expert");
            }

            if (!mismatches.empty()) {
                std::cerr << "[DenseCore] Warning: Qwen3.6 profile metadata differs from the official " << expected.name
                          << " text profile in: ";
                for (size_t i = 0; i < mismatches.size(); ++i) {
                    if (i != 0) std::cerr << ", ";
                    std::cerr << mismatches[i];
                }
                std::cerr << std::endl;
                if (strict_qwen36_profile && !allow_qwen36_test_profile) {
                    return fail_hybrid_ssm("Qwen3.6 strict profile validation failed");
                }
            }
        }
        if (model->hparams.rope_sections[0] > 0 || model->hparams.rope_sections[1] > 0) {
            std::cout << "[DenseCore] RoPE sections: [" << model->hparams.rope_sections[0] << ", "
                      << model->hparams.rope_sections[1] << ", " << model->hparams.rope_sections[2] << ", "
                      << model->hparams.rope_sections[3] << "]" << std::endl;
            std::cout << "[DenseCore] MRoPE interleaved: " << (model->hparams.rope_mrope_interleaved ? "true" : "false")
                      << std::endl;
        }
    }

    // Load LFM2 / LFM2.5 parameters (double-gated short conv + GQA attention hybrid, MoE FFN).
    if (model->arch_flags.is_lfm2_shortconv) {
        const auto fail_lfm2 = [&](const std::string& reason) {
            return fail_load("invalid LFM2 configuration: " + reason);
        };

        uint32_t tmp = static_cast<uint32_t>(model->lfm2_conv_kernel);
        get_u32("shortconv.l_cache", tmp);
        if (tmp == static_cast<uint32_t>(model->lfm2_conv_kernel)) get_u32("conv_L_cache", tmp);
        model->lfm2_conv_kernel = static_cast<int>(tmp);

        // Leading dense FFN blocks; remaining blocks are MoE. Reuse moe_first_k_dense_replace so the
        // existing per-layer dense/MoE split (see expert loading below) applies unchanged.
        tmp = 0;
        get_u32("leading_dense_block_count", tmp);
        if (tmp == 0) get_u32("num_dense_layers", tmp);
        if (tmp == 0) get_u32("n_dense_layers", tmp);
        model->lfm2_num_dense_layers = static_cast<int>(tmp);
        if (model->moe_first_k_dense_replace <= 0) {
            model->moe_first_k_dense_replace = model->lfm2_num_dense_layers;
        }

        if (model->lfm2_conv_kernel <= 1) {
            return fail_lfm2("short conv kernel (conv_L_cache) must be >= 2");
        }

        // Per-layer mixer schedule: 1 = short-conv layer, 0 = full attention layer.
        std::vector<std::string> layer_types;
        get_str_array("layer_types", layer_types);
        model->lfm2_layer_is_conv.assign(model->hparams.n_layer, 0);
        if (!layer_types.empty()) {
            if (layer_types.size() != static_cast<size_t>(model->hparams.n_layer)) {
                return fail_lfm2("partial layer_types export");
            }
            for (size_t i = 0; i < static_cast<size_t>(model->hparams.n_layer); ++i) {
                std::string type = layer_types[i];
                std::transform(type.begin(), type.end(), type.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                const bool is_full_attn =
                    (type == "full_attention" || type == "attention" || type == "self_attention");
                const bool is_conv = (type == "conv" || type == "shortconv" || type == "short_conv");
                if (!is_full_attn && !is_conv) {
                    return fail_lfm2("unknown layer_types entry '" + layer_types[i] + "'");
                }
                model->lfm2_layer_is_conv[i] = is_conv ? 1 : 0;
            }
        } else {
            // No explicit schedule: fall back to per-layer shortconv tensor presence.
            for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
                const std::string prefix = "blk." + std::to_string(i) + ".";
                const bool has_conv = gguf_find_tensor(ctx_gguf, (prefix + "shortconv.conv.weight").c_str()) != -1 ||
                                      gguf_find_tensor(ctx_gguf, (prefix + "conv.conv.weight").c_str()) != -1;
                model->lfm2_layer_is_conv[i] = has_conv ? 1 : 0;
            }
        }

        int n_conv_layers = 0;
        int n_attn_layers = 0;
        for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
            if (model->lfm2_layer_is_conv[i]) {
                n_conv_layers++;
            } else {
                n_attn_layers++;
            }
        }
        if (n_conv_layers == 0 || n_attn_layers == 0) {
            return fail_lfm2("schedule must contain both short-conv and full-attention layers");
        }

        std::cout << "[DenseCore] LFM2 params: conv_L_cache=" << model->lfm2_conv_kernel
                  << " dense_layers=" << model->lfm2_num_dense_layers << " conv_layers=" << n_conv_layers
                  << " attention_layers=" << n_attn_layers << " experts=" << model->hparams.n_experts
                  << " experts_used=" << model->hparams.n_experts_used
                  << " rope_freq_base=" << model->hparams.rope_freq_base << std::endl;
    }

    // Load BOS/EOS token IDs from tokenizer metadata
    int idx_bos = gguf_find_key(ctx_gguf, "tokenizer.ggml.bos_token_id");
    if (idx_bos != -1) {
        model->bos_token_id = gguf_get_val_u32(ctx_gguf, idx_bos);
    } else {
        model->bos_token_id = -1;
    }

    int idx_eos = gguf_find_key(ctx_gguf, "tokenizer.ggml.eos_token_id");
    if (idx_eos != -1) model->eos_token_id = gguf_get_val_u32(ctx_gguf, idx_eos);
    int idx_unk = gguf_find_key(ctx_gguf, "tokenizer.ggml.unknown_token_id");
    if (idx_unk != -1) model->unk_token_id = gguf_get_val_u32(ctx_gguf, idx_unk);
    int idx_sep = gguf_find_key(ctx_gguf, "tokenizer.ggml.seperator_token_id");
    if (idx_sep == -1) idx_sep = gguf_find_key(ctx_gguf, "tokenizer.ggml.separator_token_id");
    if (idx_sep != -1) model->sep_token_id = gguf_get_val_u32(ctx_gguf, idx_sep);
    int idx_mask = gguf_find_key(ctx_gguf, "tokenizer.ggml.mask_token_id");
    if (idx_mask != -1) model->mask_token_id = gguf_get_val_u32(ctx_gguf, idx_mask);

    // Check if model actually wants BOS added
    int idx_add_bos = gguf_find_key(ctx_gguf, "tokenizer.ggml.add_bos_token");
    bool add_bos = (idx_bos != -1);
    if (idx_add_bos != -1) {
        add_bos = gguf_get_val_bool(ctx_gguf, idx_add_bos);
    }
    if (model->arch_flags.is_gemma4) {
        // HF Gemma4 raw text tokenization does not auto-prepend BOS.
        // Some GGUF exports still carry add_bos_token=true, but the official
        // tokenizer/chat template emits <bos> explicitly when needed.
        static bool warned_gemma4_bos_override = false;
        if (idx_add_bos != -1 && gguf_get_val_bool(ctx_gguf, idx_add_bos) && !warned_gemma4_bos_override) {
            std::cerr << "[DenseCore] Warning: overriding Gemma4 tokenizer.ggml.add_bos_token=true to false "
                         "for Hugging Face tokenizer parity on raw text prompts"
                      << std::endl;
            warned_gemma4_bos_override = true;
        }
        add_bos = false;
    }

    if (model->bos_token_id < 0) {
        add_bos = false;
    }

    // Disable BOS if BOS equals PAD token
    int idx_pad = gguf_find_key(ctx_gguf, "tokenizer.ggml.padding_token_id");
    if (idx_pad != -1) {
        uint32_t pad_id = gguf_get_val_u32(ctx_gguf, idx_pad);
        model->pad_token_id = static_cast<int32_t>(pad_id);
        if (model->bos_token_id == (int)pad_id) {
            std::cout << "[DenseCore] BOS == PAD, disabling BOS (model likely "
                         "doesn't use BOS)"
                      << std::endl;
            add_bos = false;
        }
    }

    model->tokenizer_add_bos = add_bos;

    // 2. Load Vocab
    int token_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
    if (token_idx != -1) {
        int n_tokens = gguf_get_arr_n(ctx_gguf, token_idx);
        model->vocab_tokens.reserve(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            const char* str = gguf_get_arr_str(ctx_gguf, token_idx, i);
            std::string s(str);
            model->vocab_tokens.push_back(s);
            model->token_to_id[s] = i;
        }

        // Update n_vocab to match actual loaded vocab size
        if (n_tokens != (int)model->hparams.n_vocab) {
            std::cout << "[DenseCore] Normalizing n_vocab from metadata " << model->hparams.n_vocab
                      << " to tokenizer token count " << n_tokens << std::endl;
            model->hparams.n_vocab = n_tokens;
        }
    }

    if (model->arch == ModelArch::BERT) {
        auto find_token = [&](std::initializer_list<const char*> literals) -> int32_t {
            for (const char* literal : literals) {
                auto it = model->token_to_id.find(literal);
                if (it != model->token_to_id.end()) {
                    return static_cast<int32_t>(it->second);
                }
            }
            return -1;
        };
        auto id_matches = [&](int32_t id, std::initializer_list<const char*> literals) -> bool {
            if (id < 0 || id >= static_cast<int32_t>(model->vocab_tokens.size())) {
                return false;
            }
            const std::string& token = model->vocab_tokens[static_cast<size_t>(id)];
            for (const char* literal : literals) {
                if (token == literal) {
                    return true;
                }
            }
            return false;
        };

        if (!id_matches(model->unk_token_id, {"[UNK]", "<unk>"})) {
            model->unk_token_id = find_token({"[UNK]", "<unk>"});
        }
        if (!id_matches(model->pad_token_id, {"[PAD]", "<pad>"})) {
            model->pad_token_id = find_token({"[PAD]", "<pad>"});
        }
        if (!id_matches(model->mask_token_id, {"[MASK]", "<mask>"})) {
            model->mask_token_id = find_token({"[MASK]", "<mask>"});
        }

        if (!id_matches(model->bos_token_id, {"[CLS]", "<s>"})) {
            model->bos_token_id = find_token({"[CLS]", "<s>"});
        }
        if (!id_matches(model->sep_token_id, {"[SEP]", "</s>", "<sep>"})) {
            model->sep_token_id = find_token({"[SEP]", "</s>", "<sep>"});
        }
        if (model->sep_token_id >= 0 && !id_matches(model->eos_token_id, {"[SEP]", "</s>", "<eos>"})) {
            model->eos_token_id = model->sep_token_id;
        }
        if (model->bos_token_id >= 0) {
            add_bos = true;
        }
        model->tokenizer_add_bos = add_bos;
    }

    std::cout << "[DenseCore] Model params: n_vocab=" << model->hparams.n_vocab << ", n_embd=" << model->hparams.n_embd
              << ", n_layer=" << model->hparams.n_layer << ", n_head=" << model->hparams.n_head
              << ", n_head_kv=" << model->hparams.n_head_kv << ", n_rot=" << model->hparams.n_rot
              << ", n_ctx=" << model->hparams.n_ctx << std::endl;
    std::cout << "[DenseCore] BOS=" << model->bos_token_id << ", EOS=" << model->eos_token_id
              << ", rope_freq=" << model->hparams.rope_freq_base << ", rope_scale=" << model->hparams.rope_freq_scale
              << std::endl;

    // Optional token scores (SentencePiece / BPE rank hints).
    int scores_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.scores");
    if (scores_idx != -1) {
        const size_t n_scores = gguf_get_arr_n(ctx_gguf, scores_idx);
        model->token_scores.resize(n_scores);
        const gguf_type score_type = gguf_get_arr_type(ctx_gguf, scores_idx);
        const void* score_data = gguf_get_arr_data(ctx_gguf, scores_idx);
        if (score_data) {
            if (score_type == GGUF_TYPE_FLOAT32) {
                const float* src = static_cast<const float*>(score_data);
                std::copy(src, src + n_scores, model->token_scores.begin());
            } else if (score_type == GGUF_TYPE_FLOAT64) {
                const double* src = static_cast<const double*>(score_data);
                for (size_t i = 0; i < n_scores; ++i) {
                    model->token_scores[i] = static_cast<float>(src[i]);
                }
            } else {
                model->token_scores.clear();
            }
        } else {
            model->token_scores.clear();
        }
    }

    // Optional token types: 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte.
    int token_type_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.token_type");
    if (token_type_idx != -1) {
        const size_t n_types = gguf_get_arr_n(ctx_gguf, token_type_idx);
        model->token_types.assign(n_types, 1);
        const gguf_type arr_type = gguf_get_arr_type(ctx_gguf, token_type_idx);
        const void* arr_data = gguf_get_arr_data(ctx_gguf, token_type_idx);
        if (arr_data) {
            switch (arr_type) {
            case GGUF_TYPE_INT8: {
                const int8_t* src = static_cast<const int8_t*>(arr_data);
                for (size_t i = 0; i < n_types; ++i) model->token_types[i] = src[i];
                break;
            }
            case GGUF_TYPE_UINT8: {
                const uint8_t* src = static_cast<const uint8_t*>(arr_data);
                for (size_t i = 0; i < n_types; ++i) model->token_types[i] = static_cast<int32_t>(src[i]);
                break;
            }
            case GGUF_TYPE_INT16: {
                const int16_t* src = static_cast<const int16_t*>(arr_data);
                for (size_t i = 0; i < n_types; ++i) model->token_types[i] = src[i];
                break;
            }
            case GGUF_TYPE_UINT16: {
                const uint16_t* src = static_cast<const uint16_t*>(arr_data);
                for (size_t i = 0; i < n_types; ++i) model->token_types[i] = static_cast<int32_t>(src[i]);
                break;
            }
            case GGUF_TYPE_INT32: {
                const int32_t* src = static_cast<const int32_t*>(arr_data);
                for (size_t i = 0; i < n_types; ++i) model->token_types[i] = src[i];
                break;
            }
            case GGUF_TYPE_UINT32: {
                const uint32_t* src = static_cast<const uint32_t*>(arr_data);
                for (size_t i = 0; i < n_types; ++i) model->token_types[i] = static_cast<int32_t>(src[i]);
                break;
            }
            default: model->token_types.clear(); break;
            }
        } else {
            model->token_types.clear();
        }
    }

    // Collect additional stop IDs for chat/instruct tokenizers (e.g. <|im_end|>, <|eot_id|>).
    {
        std::unordered_set<int32_t> stop_ids;
        auto add_stop_id = [&](int32_t id) {
            if (id >= 0 && id < static_cast<int32_t>(model->hparams.n_vocab)) {
                stop_ids.insert(id);
            }
        };

        add_stop_id(model->eos_token_id);

        auto add_stop_key = [&](const char* key) {
            int idx = gguf_find_key(ctx_gguf, key);
            if (idx == -1) return;
            const gguf_type t = gguf_get_kv_type(ctx_gguf, idx);
            switch (t) {
            case GGUF_TYPE_UINT8: add_stop_id(static_cast<int32_t>(gguf_get_val_u8(ctx_gguf, idx))); break;
            case GGUF_TYPE_INT8: add_stop_id(static_cast<int32_t>(gguf_get_val_i8(ctx_gguf, idx))); break;
            case GGUF_TYPE_UINT16: add_stop_id(static_cast<int32_t>(gguf_get_val_u16(ctx_gguf, idx))); break;
            case GGUF_TYPE_INT16: add_stop_id(static_cast<int32_t>(gguf_get_val_i16(ctx_gguf, idx))); break;
            case GGUF_TYPE_UINT32: add_stop_id(static_cast<int32_t>(gguf_get_val_u32(ctx_gguf, idx))); break;
            case GGUF_TYPE_INT32: add_stop_id(static_cast<int32_t>(gguf_get_val_i32(ctx_gguf, idx))); break;
            case GGUF_TYPE_UINT64: add_stop_id(static_cast<int32_t>(gguf_get_val_u64(ctx_gguf, idx))); break;
            case GGUF_TYPE_INT64: add_stop_id(static_cast<int32_t>(gguf_get_val_i64(ctx_gguf, idx))); break;
            default: break;
            }
        };

        add_stop_key("tokenizer.ggml.eot_token_id");
        add_stop_key("tokenizer.ggml.eom_token_id");
        add_stop_key("tokenizer.ggml.end_of_turn_token_id");
        add_stop_key("tokenizer.ggml.end_of_message_token_id");

        const std::vector<std::string> stop_token_literals = {
            "<|im_end|>",
            "<|eot_id|>",
            "<|eom_id|>",
            "<|endoftext|>",
        };
        for (const auto& tok : stop_token_literals) {
            auto it = model->token_to_id.find(tok);
            if (it != model->token_to_id.end()) {
                add_stop_id(it->second);
            }
        }

        // If token_type is available, harvest known end/control markers.
        // GGUF is supposed to keep token_type aligned with tokens, but some
        // Gemma4 exports in the wild have partial metadata. Use the overlap we
        // have instead of dropping stop-id detection entirely.
        const size_t token_type_count = std::min(model->token_types.size(), model->vocab_tokens.size());
        for (size_t i = 0; i < token_type_count; ++i) {
            if (model->token_types[i] != 3) continue;  // control
            const std::string& tok = model->vocab_tokens[i];
            if (tok.find("im_end") != std::string::npos || tok.find("eot") != std::string::npos ||
                tok.find("eom") != std::string::npos || tok.find("endoftext") != std::string::npos) {
                add_stop_id(static_cast<int32_t>(i));
            }
        }

        model->stop_token_ids.assign(stop_ids.begin(), stop_ids.end());
        std::sort(model->stop_token_ids.begin(), model->stop_token_ids.end());
        if (!model->stop_token_ids.empty()) {
            std::cout << "[DenseCore] Registered " << model->stop_token_ids.size()
                      << " stop token IDs (including EOS/chat terminators)" << std::endl;
        }
    }

    // Load BPE merge ranks when available (GPT-2/Qwen2-style tokenizers).
    int merges_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.merges");
    if (merges_idx != -1) {
        int n_merges = gguf_get_arr_n(ctx_gguf, merges_idx);
        model->bpe_merge_ranks.reserve(static_cast<size_t>(n_merges));
        for (int i = 0; i < n_merges; ++i) {
            const char* merge_cstr = gguf_get_arr_str(ctx_gguf, merges_idx, i);
            if (!merge_cstr) continue;

            std::string merge(merge_cstr);
            const size_t split = merge.find(' ');
            if (split == std::string::npos || split == 0 || split + 1 >= merge.size()) {
                continue;
            }

            std::string key;
            key.reserve(merge.size());
            key.append(merge, 0, split);
            key.push_back('\x1f');  // unit separator
            key.append(merge, split + 1, std::string::npos);
            model->bpe_merge_ranks.emplace(std::move(key), i);
        }

        if (!model->bpe_merge_ranks.empty()) {
            std::cout << "[DenseCore] Loaded " << model->bpe_merge_ranks.size() << " BPE merges" << std::endl;
        }
    }

    if (!model->vocab_tokens.empty()) {
        Tokenizer::BuildStreamTokenPieceCache(model);
    }

    if (model->arch_flags.is_gemma4) {
        if (model->tokenizer_type.empty()) {
            return fail_load("Gemma4 GGUF is missing tokenizer.ggml.model/tokenizer.ggml.pre metadata");
        }
        if (densecore::models::ResolveTokenizerFamily(model) !=
            densecore::models::TokenizerFamily::GEMMA_SENTENCEPIECE) {
            return fail_load("Gemma4 requires GEMMA_SENTENCEPIECE tokenizer metadata");
        }
        if (model->bos_token_id < 0 || model->eos_token_id < 0) {
            return fail_load("Gemma4 GGUF is missing BOS/EOS tokenizer metadata");
        }
        if (model->vocab_tokens.empty()) {
            return fail_load("Gemma4 GGUF is missing tokenizer.ggml.tokens");
        }
        if (model->token_scores.size() != model->vocab_tokens.size()) {
            return fail_load("Gemma4 GGUF is missing complete tokenizer.ggml.scores for sentencepiece parity");
        }
        if (model->token_types.size() != model->vocab_tokens.size()) {
            return fail_load("Gemma4 GGUF is missing complete tokenizer.ggml.token_type metadata");
        }
        if (!has_key("layer_types") && !has_key("attention.sliding_window_pattern")) {
            return fail_load("Gemma4 GGUF must export explicit layer_types or sliding_window_pattern metadata");
        }
        if (!has_key("attention.shared_kv_layers")) {
            return fail_load("Gemma4 GGUF is missing attention.shared_kv_layers metadata");
        }
        if (detection_hints.has_gemma4_tensor_signatures && !has_key("embedding_length_per_layer_input")) {
            return fail_load("Gemma4 GGUF is missing embedding_length_per_layer_input metadata");
        }
        if (model->gemma4_key_length_full == 0 || model->gemma4_key_length_swa == 0 ||
            model->gemma4_value_length_full == 0 || model->gemma4_value_length_swa == 0) {
            return fail_load("Gemma4 GGUF is missing explicit key/value length metadata");
        }
        if (model->gemma4_rope_dim_full <= 0 || model->gemma4_rope_dim_swa <= 0) {
            return fail_load("Gemma4 GGUF is missing explicit RoPE dimension metadata");
        }
    }

    // 3. Initialize backend with error checking
    model->backend = ggml_backend_cpu_init();
    if (!model->backend) {
        std::cerr << "[DenseCore] Error: Failed to initialize CPU backend" << std::endl;
        gguf_free(ctx_gguf);
        if (ctx_w) ggml_free(ctx_w);
        delete model;
        return nullptr;
    }
    model->cpu_backend = model->backend;

#ifdef __APPLE__
    if (densecore::apple::IsAppleHybridEnabled()) {
        ggml_backend_t metal_backend = ggml_backend_metal_init();
        if (metal_backend) {
            model->metal_backend = metal_backend;
            std::cout << "[DenseCore] GGML Metal backend initialized" << std::endl;
        } else {
            std::cerr << "[DenseCore] Warning: Failed to initialize GGML Metal backend" << std::endl;
        }
    }
#endif

    // 4. Map Tensors
    model->layers.resize(model->hparams.n_layer);

    auto get_tensor = [&](const std::string& name) -> struct ggml_tensor* {
        struct ggml_tensor* t = ggml_get_tensor(model->ctx_w, name.c_str());
        return t;
    };

    // Helper with fallback: tries primary key, then fallback without ".weight"
    // suffix
    auto get_tensor_with_fallback = [&](const std::string& primary) -> struct ggml_tensor* {
        struct ggml_tensor* t = ggml_get_tensor(model->ctx_w, primary.c_str());
        if (t) return t;

        // Fallback: strip ".weight" suffix if present and try again
        const std::string suffix = ".weight";
        if (primary.size() > suffix.size() &&
            primary.compare(primary.size() - suffix.size(), suffix.size(), suffix) == 0) {
            std::string fallback = primary.substr(0, primary.size() - suffix.size());
            t = ggml_get_tensor(model->ctx_w, fallback.c_str());
        }
        return t;
    };

    auto get_tensor_any = [&](const std::vector<std::string>& names) -> struct ggml_tensor* {
        for (const auto& name : names) {
            if (name.empty()) continue;
            if (auto* t = get_tensor_with_fallback(name)) {
                return t;
            }
        }
        return nullptr;
    };

    auto get_layer_tensor_any = [&](uint32_t layer_idx,
                                    const std::vector<std::string>& suffixes) -> struct ggml_tensor* {
        const std::string layer_prefix = "blk." + std::to_string(layer_idx) + ".";
        std::vector<std::string> full_names;
        full_names.reserve(suffixes.size());
        for (const auto& suffix : suffixes) {
            full_names.push_back(layer_prefix + suffix);
        }
        return get_tensor_any(full_names);
    };

    auto populate_layer_tensors_from_gguf = [&]() {
        if (!model || !model->ctx_w) return;
        struct ggml_tensor* t = ggml_get_first_tensor(model->ctx_w);
        while (t) {
            const char* name = t->name;
            if (name && name[0] != '\0') {
                std::string tensor_name(name);
                if (tensor_name.rfind("blk.", 0) == 0) {
                    size_t first_dot = tensor_name.find('.', 4);
                    if (first_dot != std::string::npos) {
                        std::string layer_id = tensor_name.substr(4, first_dot - 4);
                        try {
                            int layer_idx = std::stoi(layer_id);
                            if (layer_idx >= 0 && layer_idx < static_cast<int>(model->layers.size())) {
                                std::string key = tensor_name.substr(first_dot + 1);
                                if (!model->layers[layer_idx].Get(key)) {
                                    model->layers[layer_idx].Set(key, t);
                                }
                            }
                        } catch (...) {
                            // Ignore malformed layer index
                        }
                    }
                }
            }
            t = ggml_get_next_tensor(model->ctx_w, t);
        }
    };

    model->tok_embeddings = get_tensor("token_embd.weight");
    model->position_embeddings = get_tensor("position_embd.weight");
    model->token_type_embeddings = get_tensor("token_types.weight");
    model->token_embd_norm = get_tensor("token_embd_norm.weight");
    model->token_embd_norm_bias = get_tensor("token_embd_norm.bias");
    model->output_norm = get_tensor("output_norm.weight");
    if (arch_lower == "nomic-bert") {
        return fail_load("nomic-bert GGUF embeddings are not enabled: this encoder uses fused QKV/RoPE/SwiGLU "
                         "layout and has not passed DenseCore embedding parity QA");
    }
    if (arch_lower == "jina-bert-v2") {
        return fail_load("jina-bert-v2 GGUF embeddings are not enabled: this encoder architecture has not passed "
                         "DenseCore embedding parity QA");
    }
    if (tokenizer_lower == "bert-bpe") {
        return fail_load("bert-bpe GGUF embeddings are not enabled: this tokenizer/model combination has not passed "
                         "DenseCore embedding parity QA");
    }
    if (model->arch_flags.is_gemma4) {
        model->gemma4_per_layer_model_projection = get_tensor("per_layer_model_proj.weight");
        model->gemma4_per_layer_projection_norm = get_tensor("per_layer_proj_norm.weight");
        model->gemma4_per_layer_token_embeddings = get_tensor("per_layer_token_embd.weight");
    }

    // Try multiple possible names for lm_head/output projection
    model->output = get_tensor("output.weight");
    if (!model->output) {
        model->output = get_tensor("lm_head.weight");
    }
    // Tie embeddings fallback
    if (!model->output && model->tok_embeddings) {
        std::cout << "[DenseCore] output.weight not found, using tied embeddings" << std::endl;
        model->output = model->tok_embeddings;
        model->tied_embeddings = true;
    }

    if (!model->output) {
        std::cout << "[DenseCore] CRITICAL ERROR: Could not find output weight tensor!" << std::endl;
        std::cout << "[DenseCore] Available tensors:" << std::endl;
        struct ggml_tensor* t = ggml_get_first_tensor(model->ctx_w);
        while (t) {
            std::cout << "  - " << t->name << " [" << t->ne[0] << ", " << t->ne[1] << "]" << std::endl;
            t = ggml_get_next_tensor(model->ctx_w, t);
        }
    }

    if (model->arch_flags.is_gemma4) {
        const bool has_any_per_layer_input_tensor = model->gemma4_per_layer_model_projection ||
                                                    model->gemma4_per_layer_projection_norm ||
                                                    model->gemma4_per_layer_token_embeddings;
        if (has_any_per_layer_input_tensor) {
            if (!model->gemma4_per_layer_model_projection || !model->gemma4_per_layer_projection_norm ||
                !model->gemma4_per_layer_token_embeddings) {
                return fail_load("Gemma4 GGUF has incomplete per-layer-input tensor export");
            }
            if (model->gemma4_hidden_size_per_layer_input <= 0) {
                return fail_load("Gemma4 GGUF is missing embedding_length_per_layer_input metadata");
            }
        }
    }

    // Debug: Print tensor shapes
    if (model->tok_embeddings) {
        std::cout << "[DenseCore] tok_embeddings shape: [" << model->tok_embeddings->ne[0] << ", "
                  << model->tok_embeddings->ne[1] << "]" << std::endl;
    }
    if (model->output) {
        std::cout << "[DenseCore] output shape: [" << model->output->ne[0] << ", " << model->output->ne[1] << "]"
                  << std::endl;
    }
    if (model->tok_embeddings) {
        const uint32_t embedding_vocab_size = static_cast<uint32_t>(model->tok_embeddings->ne[1]);
        if (embedding_vocab_size > 0 && embedding_vocab_size != model->hparams.n_vocab) {
            return fail_load("GGUF tokenizer/embedding vocab size mismatch");
        }
    }
    if (model->output) {
        const uint32_t output_vocab_size = static_cast<uint32_t>(model->output->ne[1]);
        if (output_vocab_size > 0 && output_vocab_size != model->hparams.n_vocab) {
            return fail_load("GGUF tokenizer/output vocab size mismatch");
        }
    }

    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        std::string layer_prefix = "blk." + std::to_string(i) + ".";

        // Common to all layer types: norms and FFN
        model->layers[i].Set(model_keys::kAttnNorm,
                             get_layer_tensor_any(i, {"attn_norm.weight", "input_layernorm.weight",
                                                      "attention_norm.weight", "self_attn_layernorm.weight"}));
        model->layers[i].Set(model_keys::kFfnNorm,
                             get_layer_tensor_any(i, {"ffn_norm.weight", "pre_feedforward_layernorm.weight",
                                                      "post_attention_layernorm.weight", "post_attention_norm.weight",
                                                      "mlp_layernorm.weight"}));
        model->layers[i].Set(model_keys::kPostAttnNorm, get_layer_tensor_any(i, {"post_attention_norm.weight",
                                                                                 "post_attention_layernorm.weight"}));
        model->layers[i].Set(model_keys::kAttnOutputNorm, get_layer_tensor_any(i, {"attn_output_norm.weight"}));
        model->layers[i].Set(model_keys::kAttnOutputNormBias, get_layer_tensor_any(i, {"attn_output_norm.bias"}));
        // Fallback: Qwen3.5 uses post_attention_norm instead of ffn_norm
        if (!model->layers[i].Get(model_keys::kFfnNorm) && model->layers[i].Get(model_keys::kPostAttnNorm)) {
            model->layers[i].Set(model_keys::kFfnNorm, model->layers[i].Get(model_keys::kPostAttnNorm));
        }
        model->layers[i].Set(
            model_keys::kFfnGate,
            get_layer_tensor_any(i, {"ffn_gate.weight", "ffn_gate_shexp.weight", "shared_expert.gate_proj.weight",
                                     "mlp.gate_proj.weight", "gate_proj.weight"}));
        model->layers[i].Set(
            model_keys::kFfnDown,
            get_layer_tensor_any(i, {"ffn_down.weight", "ffn_down_shexp.weight", "shared_expert.down_proj.weight",
                                     "mlp.down_proj.weight", "down_proj.weight"}));
        model->layers[i].Set(model_keys::kFfnDownBias, get_layer_tensor_any(i, {"ffn_down.bias"}));
        model->layers[i].Set(model_keys::kFfnUp, get_layer_tensor_any(i, {"ffn_up.weight", "ffn_up_shexp.weight",
                                                                          "shared_expert.up_proj.weight",
                                                                          "mlp.up_proj.weight", "up_proj.weight"}));
        model->layers[i].Set(model_keys::kFfnUpBias, get_layer_tensor_any(i, {"ffn_up.bias"}));
        model->layers[i].Set(model_keys::kLayerOutputNorm, get_layer_tensor_any(i, {"layer_output_norm.weight"}));
        model->layers[i].Set(model_keys::kLayerOutputNormBias, get_layer_tensor_any(i, {"layer_output_norm.bias"}));
        model->layers[i].Set(model_keys::kFfnSharedGate,
                             get_layer_tensor_any(i, {"ffn_gate_inp_shexp.weight", "shared_expert_gate.weight"}));
        model->layers[i].Set(kGemma4RouterScaleKey, get_layer_tensor_any(i, {"router.scale", "ffn_gate_inp.scale"}));
        model->layers[i].Set(kGemma4RouterPerExpertScaleKey, get_layer_tensor_any(i, {"router.per_expert_scale"}));
        model->layers[i].Set(kGemma4PreMoeNormKey,
                             get_layer_tensor_any(i, {"pre_ffw_norm_2.weight", "pre_feedforward_layernorm_2.weight"}));
        model->layers[i].Set(
            kGemma4PostSharedNormKey,
            get_layer_tensor_any(i, {"post_ffw_norm_1.weight", "post_feedforward_layernorm_1.weight"}));
        model->layers[i].Set(kGemma4PostMoeNormKey, get_layer_tensor_any(i, {"post_ffw_norm_2.weight",
                                                                             "post_feedforward_layernorm_2.weight"}));
        model->layers[i].Set(kGemma4PostFfnNormKey,
                             get_layer_tensor_any(i, {"post_feedforward_layernorm.weight", "post_ffw_norm.weight"}));
        model->layers[i].Set(model_keys::kGemma4PerLayerInputGate, get_layer_tensor_any(i, {"inp_gate.weight"}));
        model->layers[i].Set(model_keys::kGemma4PerLayerProjection, get_layer_tensor_any(i, {"proj.weight"}));
        model->layers[i].Set(model_keys::kGemma4PostPerLayerInputNorm, get_layer_tensor_any(i, {"post_norm.weight"}));
        model->layers[i].Set(model_keys::kGemma4LayerOutputScale,
                             get_layer_tensor_any(i, {"layer_output_scale.weight"}));
        {
            struct ggml_tensor* rope_freqs = get_layer_tensor_any(i, {"rope_freqs.weight"});
            if (!rope_freqs && model->arch_flags.is_gemma4) {
                rope_freqs = get_tensor("rope_freqs.weight");
            }
            model->layers[i].Set(model_keys::kAttnRopeFreqs, rope_freqs);
        }
        model->layers[i].Set(model_keys::kMoeGate,
                             get_layer_tensor_any(i, {"moe_gate.weight", "router.proj.weight", "mlp.gate.weight",
                                                      "ffn_gate_inp.weight", "ffn_gate_inp"}));

        // Determine mixer type: LFM2 short-conv, hybrid SSM, or full attention.
        const bool is_conv = model->IsLFM2ConvLayer(static_cast<int>(i));
        const bool is_ssm = model->IsHybridSSMLayer(static_cast<int>(i));

        if (is_conv) {
            // LFM2 / LFM2.5 short-conv mixer: in_proj -> (B*x) -> depthwise causal conv -> (C*) -> out_proj
            model->layers[i].Set(model_keys::kShortConvInProj,
                                 get_layer_tensor_any(i, {"shortconv.in_proj.weight", "conv.in_proj.weight"}));
            model->layers[i].Set(model_keys::kShortConvConv,
                                 get_layer_tensor_any(i, {"shortconv.conv.weight", "conv.conv.weight"}));
            model->layers[i].Set(model_keys::kShortConvOutProj,
                                 get_layer_tensor_any(i, {"shortconv.out_proj.weight", "conv.out_proj.weight"}));
            if (i == 0) {
                auto* in_proj = model->layers[i].Get(model_keys::kShortConvInProj);
                auto* conv = model->layers[i].Get(model_keys::kShortConvConv);
                if (in_proj && conv) {
                    std::cout << "[DenseCore] LFM2 short-conv layer 0: in_proj [" << in_proj->ne[0] << "x"
                              << in_proj->ne[1] << "], conv [" << conv->ne[0] << "x" << conv->ne[1] << "]" << std::endl;
                }
            }
        } else if (is_ssm) {
            // SSM/Mamba layer: fused QKV + SSM weights + gate
            struct ggml_tensor* fused_ba = get_layer_tensor_any(i, {"ssm_ba.weight", "linear_attn.in_proj_ba.weight"});
            struct ggml_tensor* ssm_alpha =
                get_layer_tensor_any(i, {"ssm_alpha.weight", "linear_attn.in_proj_a.weight"});
            struct ggml_tensor* ssm_beta = get_layer_tensor_any(i, {"ssm_beta.weight", "linear_attn.in_proj_b.weight"});
            if (!ssm_alpha && fused_ba) ssm_alpha = fused_ba;
            if (!ssm_beta && fused_ba) ssm_beta = fused_ba;

            model->layers[i].Set(model_keys::kAttnQkvWeight,
                                 get_layer_tensor_any(i, {"attn_qkv.weight", "linear_attn.in_proj_qkv.weight"}));
            model->layers[i].Set(model_keys::kAttnGate,
                                 get_layer_tensor_any(i, {"attn_gate.weight", "linear_attn.in_proj_z.weight"}));
            model->layers[i].Set(model_keys::kSSMConv1d,
                                 get_layer_tensor_any(i, {"ssm_conv1d.weight", "linear_attn.conv1d.weight"}));
            model->layers[i].Set(model_keys::kSSMA, get_layer_tensor_any(i, {"ssm_a", "linear_attn.A_log"}));
            model->layers[i].Set(model_keys::kSSMAlpha, ssm_alpha);
            model->layers[i].Set(model_keys::kSSMBeta, ssm_beta);
            model->layers[i].Set(model_keys::kSSMDtBias,
                                 get_layer_tensor_any(i, {"ssm_dt.bias", "linear_attn.dt_bias"}));
            model->layers[i].Set(model_keys::kSSMNorm,
                                 get_layer_tensor_any(i, {"ssm_norm.weight", "linear_attn.norm.weight"}));
            model->layers[i].Set(model_keys::kSSMOut,
                                 get_layer_tensor_any(i, {"ssm_out.weight", "linear_attn.out_proj.weight"}));

            if (i == 0) {
                auto* conv1d = model->layers[i].Get(model_keys::kSSMConv1d);
                auto* ssm_a = model->layers[i].Get(model_keys::kSSMA);
                if (conv1d && ssm_a) {
                    std::cout << "[DenseCore] SSM layer 0: conv1d [" << conv1d->ne[0] << "x" << conv1d->ne[1]
                              << "], ssm_a [" << ssm_a->ne[0] << "]" << std::endl;
                }
            }
        } else {
            // Full attention layer: separate Q/K/V + QK-norms
            model->layers[i].Set(
                model_keys::kAttnQWeight,
                get_layer_tensor_any(i, {"attn_q.weight", "self_attn.q_proj.weight", "q_proj.weight"}));
            model->layers[i].Set(
                model_keys::kAttnKWeight,
                get_layer_tensor_any(i, {"attn_k.weight", "self_attn.k_proj.weight", "k_proj.weight"}));
            model->layers[i].Set(
                model_keys::kAttnVWeight,
                get_layer_tensor_any(i, {"attn_v.weight", "self_attn.v_proj.weight", "v_proj.weight"}));
            model->layers[i].Set(model_keys::kAttnOWeight,
                                 get_layer_tensor_any(i, {"attn_output.weight", "self_attn.o_proj.weight",
                                                          "o_proj.weight", "self_attn.out_proj.weight"}));

            model->layers[i].Set(model_keys::kAttnQBias,
                                 get_layer_tensor_any(i, {"attn_q.bias", "self_attn.q_proj.bias", "q_proj.bias"}));
            model->layers[i].Set(model_keys::kAttnKBias,
                                 get_layer_tensor_any(i, {"attn_k.bias", "self_attn.k_proj.bias", "k_proj.bias"}));
            model->layers[i].Set(model_keys::kAttnVBias,
                                 get_layer_tensor_any(i, {"attn_v.bias", "self_attn.v_proj.bias", "v_proj.bias"}));
            model->layers[i].Set(model_keys::kAttnOBias,
                                 get_layer_tensor_any(i, {"attn_output.bias", "self_attn.o_proj.bias", "o_proj.bias"}));

            // QK-Norm (Qwen3/3.5) - uses fallback for different GGUF naming
            model->layers[i].Set(
                model_keys::kAttnQNorm,
                get_layer_tensor_any(i, {"attn_q_norm.weight", "self_attn.q_norm.weight", "q_norm.weight"}));
            model->layers[i].Set(
                model_keys::kAttnKNorm,
                get_layer_tensor_any(i, {"attn_k_norm.weight", "self_attn.k_norm.weight", "k_norm.weight"}));
            model->layers[i].Set(
                model_keys::kAttnVNorm,
                get_layer_tensor_any(i, {"attn_v_norm.weight", "self_attn.v_norm.weight", "v_norm.weight"}));
        }

        if (i == 0 || (model->arch_flags.is_hybrid_ssm && !is_ssm)) {
            auto* q_norm = model->layers[i].Get(model_keys::kAttnQNorm);
            auto* k_norm = model->layers[i].Get(model_keys::kAttnKNorm);
            if (q_norm && k_norm && i < 2) {
                std::cout << "[DenseCore] Attention layer " << i << ": Q/K Norms enabled" << " q_norm[" << q_norm->ne[0]
                          << "] k_norm[" << k_norm->ne[0] << "]" << std::endl;
            }
        }
    }

    // Populate any missing layer tensors from GGUF metadata.
    populate_layer_tensors_from_gguf();

    auto find_layer_tensor_with_tokens = [&](const TransformerLayer& layer, const std::vector<std::string>& required,
                                             const std::vector<std::string>& forbidden = {}) -> struct ggml_tensor* {
        for (const auto& entry : layer.tensors) {
            const std::string key_lower = ascii_lower_copy(entry.first);
            bool matches = true;
            for (const auto& token : required) {
                if (key_lower.find(token) == std::string::npos) {
                    matches = false;
                    break;
                }
            }
            if (!matches) continue;
            for (const auto& token : forbidden) {
                if (key_lower.find(token) != std::string::npos) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                return entry.second;
            }
        }
        return nullptr;
    };

    std::regex expert_gate_re(R"(experts[._](\d+)[._].*gate_proj\.weight)", std::regex::icase);
    std::regex expert_up_re(R"(experts[._](\d+)[._].*up_proj\.weight)", std::regex::icase);
    std::regex expert_down_re(R"(experts[._](\d+)[._].*down_proj\.weight)", std::regex::icase);

    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        auto& layer = model->layers[i];
        int expert_regex_hits = 0;
        bool used_packed_expert_slices = false;
        bool used_sep_expert_slices = false;

        if (!layer.Get(model_keys::kFfnGate)) {
            layer.Set(model_keys::kFfnGate, find_layer_tensor_with_tokens(layer, {"shared_experts", "gate_proj"}));
            if (!layer.Get(model_keys::kFfnGate)) {
                layer.Set(model_keys::kFfnGate, find_layer_tensor_with_tokens(layer, {"shared_expert", "gate_proj"}));
            }
        }
        if (!layer.Get(model_keys::kFfnUp)) {
            layer.Set(model_keys::kFfnUp, find_layer_tensor_with_tokens(layer, {"shared_experts", "up_proj"}));
            if (!layer.Get(model_keys::kFfnUp)) {
                layer.Set(model_keys::kFfnUp, find_layer_tensor_with_tokens(layer, {"shared_expert", "up_proj"}));
            }
        }
        if (!layer.Get(model_keys::kFfnDown)) {
            layer.Set(model_keys::kFfnDown, find_layer_tensor_with_tokens(layer, {"shared_experts", "down_proj"}));
            if (!layer.Get(model_keys::kFfnDown)) {
                layer.Set(model_keys::kFfnDown, find_layer_tensor_with_tokens(layer, {"shared_expert", "down_proj"}));
            }
        }
        if (!layer.Get(model_keys::kFfnSharedGate)) {
            auto* t = find_layer_tensor_with_tokens(layer, {"shared_expert_gate"});
            if (!t) t = find_layer_tensor_with_tokens(layer, {"ffn_gate_inp_shexp"});
            layer.Set(model_keys::kFfnSharedGate, t);
        }
        if (!layer.Get(model_keys::kMoeGate)) {
            auto* t =
                get_layer_tensor_any(static_cast<int>(i), {"moe_gate.weight", "router.proj.weight", "mlp.gate.weight",
                                                           "ffn_gate_inp.weight", "ffn_gate_inp"});
            if (!t) t = find_layer_tensor_with_tokens(layer, {"mlp", "gate.weight"}, {"shared"});
            if (!t) t = find_layer_tensor_with_tokens(layer, {"ffn_gate_inp"});
            layer.Set(model_keys::kMoeGate, t);
        }
        if (!layer.Get(model_keys::kMoeCorrectionBias)) {
            layer.Set(model_keys::kMoeCorrectionBias,
                      find_layer_tensor_with_tokens(layer, {"e_score_correction_bias"}));
        }
        if (!layer.Get(model_keys::kMoeCorrectionBias)) {
            // LFM2 / LFM2.5 (use_expert_bias): aux-loss-free routing bias exported on the router gate.
            auto* t = get_layer_tensor_any(static_cast<int>(i), {"ffn_gate_inp.bias", "exp_probs_b.bias"});
            if (!t) t = find_layer_tensor_with_tokens(layer, {"expert_bias"});
            layer.Set(model_keys::kMoeCorrectionBias, t);
        }

        const struct ggml_tensor* packed_gate_up =
            get_layer_tensor_any(static_cast<int>(i), {"experts.gate_up_proj.weight", "ffn_gate_up_exps.weight"});
        if (!packed_gate_up) {
            packed_gate_up = find_layer_tensor_with_tokens(layer, {"experts", "gate_up_proj"}, {"shared"});
        }
        const struct ggml_tensor* packed_down =
            get_layer_tensor_any(static_cast<int>(i), {"experts.down_proj.weight", "ffn_down_exps.weight"});
        if (!packed_down) {
            packed_down = find_layer_tensor_with_tokens(layer, {"experts", "down_proj"}, {"shared"});
        }
        const struct ggml_tensor* packed_down_scale = nullptr;

        // Gemma4 E26B/A4B GGUFs use packed MoE tensors named
        // `ffn_gate_up_exps.weight` + `ffn_down_exps.weight` (with optional
        // sidecar `ffn_down_exps.scale`) instead of `experts.*gate_up_proj`.
        // Keep this narrowly scoped to Gemma4 to avoid changing unrelated MoE
        // loaders.
        if (model->arch_flags.is_gemma4) {
            if (!packed_gate_up) {
                packed_gate_up = find_layer_tensor_with_tokens(layer, {"ffn_gate_up_exps", "weight"});
                if (!packed_gate_up) {
                    packed_gate_up = find_layer_tensor_with_tokens(layer, {"ffn_gate_up_exps"}, {"scale"});
                }
            }
            if (!packed_down) {
                packed_down = find_layer_tensor_with_tokens(layer, {"ffn_down_exps", "weight"});
                if (!packed_down) {
                    packed_down = find_layer_tensor_with_tokens(layer, {"ffn_down_exps"}, {"scale"});
                }
            }
            packed_down_scale = get_layer_tensor_any(static_cast<int>(i), {"ffn_down_exps.scale"});
            if (!packed_down_scale) {
                packed_down_scale = find_layer_tensor_with_tokens(layer, {"ffn_down_exps", "scale"});
            }
        }

        if (IsQwen36MoEBindingDebugEnabled() && model->variant == ModelVariant::QWEN36 &&
            model->hparams.n_experts > 0 && i < 2) {
            std::fprintf(
                stderr, "[QWEN36_MOE_PACKED_TENSORS] layer=%u packed_gate_up=%s packed_down=%s packed_down_scale=%s\n",
                i, packed_gate_up ? (packed_gate_up->name[0] ? packed_gate_up->name : "<unnamed>") : "<null>",
                packed_down ? (packed_down->name[0] ? packed_down->name : "<unnamed>") : "<null>",
                packed_down_scale ? (packed_down_scale->name[0] ? packed_down_scale->name : "<unnamed>") : "<null>");
        }

        if (packed_gate_up && packed_down &&
            i >= static_cast<uint32_t>(std::max(0, model->moe_first_k_dense_replace))) {
            densecore::gemma4::PackedExpertLayout packed_layout{};
            std::string packed_layout_reason;
            const bool has_canonical_gemma4_layout =
                model->arch_flags.is_gemma4 && densecore::gemma4::InferPackedExpertLayout(
                                                   packed_gate_up, packed_down, &packed_layout, &packed_layout_reason);
            if (model->arch_flags.is_gemma4 && !has_canonical_gemma4_layout) {
                std::cerr << "[DenseCore] FATAL: unsupported Gemma4 packed MoE layout at layer " << i << ": "
                          << packed_layout_reason << std::endl;
                return nullptr;
            }
            if (model->arch_flags.is_gemma4 && !packed_down_scale) {
                std::cerr << "[DenseCore] FATAL: missing Gemma4 packed MoE down-scale sidecar at layer " << i
                          << std::endl;
                return nullptr;
            }
            int packed_experts = 0;
            if (has_canonical_gemma4_layout) {
                packed_experts = packed_layout.num_experts;
            } else if (packed_gate_up->ne[2] > 1) {
                packed_experts = static_cast<int>(packed_gate_up->ne[2]);
            } else if (packed_gate_up->ne[3] > 1) {
                packed_experts = static_cast<int>(packed_gate_up->ne[3]);
            }
            if (model->hparams.n_experts == 0 && packed_experts > 0) {
                model->hparams.n_experts = static_cast<uint32_t>(packed_experts);
            }
            const int expert_count = std::min<int>(packed_experts, static_cast<int>(model->hparams.n_experts));
            if (model->arch_flags.is_gemma4 && expert_count > 0 && !layer.Get(model_keys::kMoeGate)) {
                std::cerr << "[DenseCore] FATAL: missing Gemma4 MoE router weight for packed layer " << i << std::endl;
                return nullptr;
            }
            if (expert_count > 0 && layer.Get(model_keys::kMoeGate)) {
                layer.is_moe = true;
                used_packed_expert_slices = true;
                struct ggml_context* vctx = model->ctx_views ? model->ctx_views : model->ctx_w;
                for (int expert_idx = 0; expert_idx < expert_count; ++expert_idx) {
                    if (has_canonical_gemma4_layout) {
                        densecore::gemma4::PackedExpertViews expert_views{};
                        std::string packed_view_reason;
                        if (!densecore::gemma4::MakePackedExpertViews(
                                vctx, const_cast<struct ggml_tensor*>(packed_gate_up),
                                const_cast<struct ggml_tensor*>(packed_down),
                                const_cast<struct ggml_tensor*>(packed_down_scale), expert_idx, &expert_views,
                                &packed_view_reason)) {
                            std::cerr << "[DenseCore] FATAL: Gemma4 packed MoE view creation failed at layer " << i
                                      << " expert " << expert_idx << ": " << packed_view_reason << std::endl;
                            return nullptr;
                        }
                        layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kGemma4PackedGateUpExpert,
                                        expert_views.gate_up);
                        layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kFfnGate, expert_views.gate);
                        layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kFfnUp, expert_views.up);
                        layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kFfnDown, expert_views.down);
                        if (expert_views.down_scale) {
                            layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kGemma4PackedDownScale,
                                            expert_views.down_scale);
                        }

                        const auto gate_it = model->int4_weight_bindings.find(packed_gate_up);
                        if (gate_it != model->int4_weight_bindings.end()) {
                            TransformerModel::Int4WeightBinding gate_binding{};
                            std::string gate_binding_reason;
                            if (densecore::gemma4::ResolvePackedProjectionBinding(vctx, gate_it->second,
                                                                                  expert_views.gate_view, &gate_binding,
                                                                                  &gate_binding_reason)) {
                                model->int4_weight_bindings[expert_views.gate] = gate_binding;
                            } else {
                                std::cerr << "[DenseCore] FATAL: Gemma4 gate INT4 binding failed at layer " << i
                                          << " expert " << expert_idx << ": " << gate_binding_reason << std::endl;
                                return nullptr;
                            }
                            TransformerModel::Int4WeightBinding up_binding{};
                            std::string up_binding_reason;
                            if (densecore::gemma4::ResolvePackedProjectionBinding(
                                    vctx, gate_it->second, expert_views.up_view, &up_binding, &up_binding_reason)) {
                                model->int4_weight_bindings[expert_views.up] = up_binding;
                            } else {
                                std::cerr << "[DenseCore] FATAL: Gemma4 up INT4 binding failed at layer " << i
                                          << " expert " << expert_idx << ": " << up_binding_reason << std::endl;
                                return nullptr;
                            }
                        }
                        const auto down_it = model->int4_weight_bindings.find(packed_down);
                        if (down_it != model->int4_weight_bindings.end()) {
                            TransformerModel::Int4WeightBinding down_binding{};
                            std::string down_binding_reason;
                            if (densecore::gemma4::ResolvePackedProjectionBinding(vctx, down_it->second,
                                                                                  expert_views.down_view, &down_binding,
                                                                                  &down_binding_reason)) {
                                model->int4_weight_bindings[expert_views.down] = down_binding;
                            } else {
                                std::cerr << "[DenseCore] FATAL: Gemma4 down INT4 binding failed at layer " << i
                                          << " expert " << expert_idx << ": " << down_binding_reason << std::endl;
                                return nullptr;
                            }
                        }
                        continue;
                    }

                    const int gate_up_expert_axis =
                        packed_gate_up->ne[2] > 1 ? 2 : (packed_gate_up->ne[3] > 1 ? 3 : -1);
                    const int down_expert_axis = packed_down->ne[2] > 1 ? 2 : (packed_down->ne[3] > 1 ? 3 : -1);
                    if (gate_up_expert_axis < 0 || down_expert_axis < 0) {
                        continue;
                    }

                    const size_t gate_up_stride = static_cast<size_t>(packed_gate_up->nb[gate_up_expert_axis]);
                    const size_t down_stride = static_cast<size_t>(packed_down->nb[down_expert_axis]);
                    const size_t gate_up_offset = static_cast<size_t>(expert_idx) * gate_up_stride;
                    const size_t down_offset = static_cast<size_t>(expert_idx) * down_stride;
                    if (ggml_nbytes(packed_gate_up) > 0 &&
                        gate_up_offset + static_cast<size_t>(packed_gate_up->ne[0]) *
                                             static_cast<size_t>(packed_gate_up->nb[1]) >
                            static_cast<size_t>(ggml_nbytes(packed_gate_up))) {
                        continue;
                    }
                    if (ggml_nbytes(packed_down) > 0 && down_offset + static_cast<size_t>(packed_down->ne[0]) *
                                                                          static_cast<size_t>(packed_down->nb[1]) >
                                                            static_cast<size_t>(ggml_nbytes(packed_down))) {
                        continue;
                    }
                    struct ggml_tensor* gate_up_slice =
                        ggml_view_2d(vctx, const_cast<struct ggml_tensor*>(packed_gate_up), packed_gate_up->ne[0],
                                     packed_gate_up->ne[1], packed_gate_up->nb[1], gate_up_offset);
                    if (!gate_up_slice) {
                        continue;
                    }
                    const int64_t gate_up_rows = gate_up_slice->ne[1];
                    if (gate_up_rows < 2 || (gate_up_rows % 2) != 0) {
                        continue;
                    }
                    const int64_t intermediate = gate_up_rows / 2;
                    struct ggml_tensor* gate_w =
                        ggml_view_2d(vctx, gate_up_slice, gate_up_slice->ne[0], intermediate, gate_up_slice->nb[1], 0);
                    struct ggml_tensor* up_w =
                        ggml_view_2d(vctx, gate_up_slice, gate_up_slice->ne[0], intermediate, gate_up_slice->nb[1],
                                     static_cast<size_t>(intermediate) * gate_up_slice->nb[1]);
                    struct ggml_tensor* down_w =
                        ggml_view_2d(vctx, const_cast<struct ggml_tensor*>(packed_down), packed_down->ne[0],
                                     packed_down->ne[1], packed_down->nb[1], down_offset);
                    if (!gate_w || !up_w || !down_w) {
                        continue;
                    }
                    layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kFfnGate, gate_w);
                    layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kFfnUp, up_w);
                    layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kFfnDown, down_w);

                    const auto gate_it = model->int4_weight_bindings.find(packed_gate_up);
                    if (gate_it != model->int4_weight_bindings.end()) {
                        TransformerModel::Int4WeightBinding gate_binding{};
                        if (ResolveGenericPackedProjectionBinding(vctx, gate_it->second, gate_w, gate_w->ne[0],
                                                                  gate_w->ne[1], 0, gate_up_expert_axis, expert_idx,
                                                                  &gate_binding)) {
                            model->int4_weight_bindings[gate_w] = gate_binding;
                        }
                        TransformerModel::Int4WeightBinding up_binding{};
                        if (ResolveGenericPackedProjectionBinding(vctx, gate_it->second, up_w, up_w->ne[0], up_w->ne[1],
                                                                  intermediate, gate_up_expert_axis, expert_idx,
                                                                  &up_binding)) {
                            model->int4_weight_bindings[up_w] = up_binding;
                        }
                    }
                    const auto down_it = model->int4_weight_bindings.find(packed_down);
                    if (down_it != model->int4_weight_bindings.end()) {
                        TransformerModel::Int4WeightBinding down_binding{};
                        if (ResolveGenericPackedProjectionBinding(vctx, down_it->second, down_w, down_w->ne[0],
                                                                  down_w->ne[1], 0, down_expert_axis, expert_idx,
                                                                  &down_binding)) {
                            model->int4_weight_bindings[down_w] = down_binding;
                        }
                    }
                }
            }
        }

        for (const auto& entry : layer.tensors) {
            const std::string& name = entry.first;
            struct ggml_tensor* tensor = entry.second;
            std::smatch match;
            if (std::regex_search(name, match, expert_gate_re) && match.size() >= 2) {
                const size_t expert_idx = static_cast<size_t>(std::stoul(match[1].str()));
                layer.SetExpert(expert_idx, model_keys::kFfnGate, tensor);
                expert_regex_hits++;
                continue;
            }
            if (std::regex_search(name, match, expert_up_re) && match.size() >= 2) {
                const size_t expert_idx = static_cast<size_t>(std::stoul(match[1].str()));
                layer.SetExpert(expert_idx, model_keys::kFfnUp, tensor);
                expert_regex_hits++;
                continue;
            }
            if (std::regex_search(name, match, expert_down_re) && match.size() >= 2) {
                const size_t expert_idx = static_cast<size_t>(std::stoul(match[1].str()));
                layer.SetExpert(expert_idx, model_keys::kFfnDown, tensor);
                expert_regex_hits++;
            }
        }

        // Handle separate stacked expert format: ffn_gate_exps / ffn_up_exps / ffn_down_exps
        // (used by qwen35moe bartowski GGUFs — experts stacked along ne[2] axis)
        if (layer.NumExperts() == 0 && layer.Get(model_keys::kMoeGate)) {
            const struct ggml_tensor* sep_gate =
                get_layer_tensor_any(static_cast<int>(i), {"ffn_gate_exps.weight", "ffn_gate_exps"});
            if (!sep_gate) sep_gate = find_layer_tensor_with_tokens(layer, {"ffn_gate_exps"});
            const struct ggml_tensor* sep_up =
                get_layer_tensor_any(static_cast<int>(i), {"ffn_up_exps.weight", "ffn_up_exps"});
            if (!sep_up) sep_up = find_layer_tensor_with_tokens(layer, {"ffn_up_exps"});
            const struct ggml_tensor* sep_down =
                get_layer_tensor_any(static_cast<int>(i), {"ffn_down_exps.weight", "ffn_down_exps"});
            if (!sep_down) sep_down = find_layer_tensor_with_tokens(layer, {"ffn_down_exps"});
            if (sep_gate && sep_up && sep_down) {
                int n_exp = (sep_gate->ne[2] > 1)   ? static_cast<int>(sep_gate->ne[2])
                            : (sep_gate->ne[3] > 0) ? static_cast<int>(sep_gate->ne[3])
                                                    : 0;
                if (model->hparams.n_experts == 0 && n_exp > 0) {
                    model->hparams.n_experts = static_cast<uint32_t>(n_exp);
                }
                int expert_count = std::min<int>(n_exp, static_cast<int>(model->hparams.n_experts));
                if (expert_count > 0 && i >= static_cast<uint32_t>(std::max(0, model->moe_first_k_dense_replace))) {
                    layer.is_moe = true;
                    used_sep_expert_slices = true;
                    struct ggml_context* vctx2 = model->ctx_views ? model->ctx_views : model->ctx_w;
                    for (int ei = 0; ei < expert_count; ++ei) {
                        const size_t g_off = static_cast<size_t>(ei) * sep_gate->nb[2];
                        const size_t u_off = static_cast<size_t>(ei) * sep_up->nb[2];
                        const size_t d_off = static_cast<size_t>(ei) * sep_down->nb[2];
                        struct ggml_tensor* gw = ggml_view_2d(vctx2, const_cast<struct ggml_tensor*>(sep_gate),
                                                              sep_gate->ne[0], sep_gate->ne[1], sep_gate->nb[1], g_off);
                        struct ggml_tensor* uw = ggml_view_2d(vctx2, const_cast<struct ggml_tensor*>(sep_up),
                                                              sep_up->ne[0], sep_up->ne[1], sep_up->nb[1], u_off);
                        struct ggml_tensor* dw = ggml_view_2d(vctx2, const_cast<struct ggml_tensor*>(sep_down),
                                                              sep_down->ne[0], sep_down->ne[1], sep_down->nb[1], d_off);
                        layer.SetExpert(static_cast<size_t>(ei), model_keys::kFfnGate, gw);
                        layer.SetExpert(static_cast<size_t>(ei), model_keys::kFfnUp, uw);
                        layer.SetExpert(static_cast<size_t>(ei), model_keys::kFfnDown, dw);
                    }
                }
            }
        }

        if (layer.Get(model_keys::kMoeGate) && layer.NumExperts() > 0 &&
            i >= static_cast<uint32_t>(std::max(0, model->moe_first_k_dense_replace))) {
            layer.is_moe = true;
            if (model->hparams.n_experts == 0) {
                model->hparams.n_experts = static_cast<uint32_t>(layer.NumExperts());
            }
        }

        if (IsMoELoaderDebugEnabled()) {
            int expert_name_candidates = 0;
            for (const auto& entry : layer.tensors) {
                const std::string key_lower = ascii_lower_copy(entry.first);
                if (key_lower.find("expert") != std::string::npos ||
                    key_lower.find("ffn_gate_exps") != std::string::npos ||
                    key_lower.find("ffn_up_exps") != std::string::npos ||
                    key_lower.find("ffn_down_exps") != std::string::npos) {
                    expert_name_candidates++;
                }
            }
            int reason_code = 0;
            if (!layer.Get(model_keys::kMoeGate)) {
                reason_code = 1;  // no_moe_gate
            } else if (layer.NumExperts() == 0) {
                reason_code = expert_name_candidates > 0 ? 3 : 2;  // 3: candidates but unmatched, 2: no candidates
            }
            size_t gate_int4_bindings = 0;
            size_t up_int4_bindings = 0;
            size_t down_int4_bindings = 0;
            size_t fused_gate_up_available = 0;
            for (size_t expert_idx = 0; expert_idx < layer.NumExperts(); ++expert_idx) {
                const bool has_gate_int4 =
                    model->int4_weight_bindings.find(layer.GetExpert(expert_idx, model_keys::kFfnGate)) !=
                    model->int4_weight_bindings.end();
                const bool has_up_int4 =
                    model->int4_weight_bindings.find(layer.GetExpert(expert_idx, model_keys::kFfnUp)) !=
                    model->int4_weight_bindings.end();
                const bool has_down_int4 =
                    model->int4_weight_bindings.find(layer.GetExpert(expert_idx, model_keys::kFfnDown)) !=
                    model->int4_weight_bindings.end();
                gate_int4_bindings += has_gate_int4 ? 1u : 0u;
                up_int4_bindings += has_up_int4 ? 1u : 0u;
                down_int4_bindings += has_down_int4 ? 1u : 0u;
                fused_gate_up_available += (has_gate_int4 && has_up_int4) ? 1u : 0u;
            }
            std::fprintf(stderr,
                         "[MOE_LOADER_LAYER] layer=%u is_moe=%d has_moe_gate=%d num_experts=%zu packed_gate_up=%d "
                         "packed_down=%d used_packed=%d used_sep=%d regex_hits=%d expert_name_candidates=%d "
                         "gate_int4=%zu up_int4=%zu down_int4=%zu fused_gate_up=%zu reason_code=%d\n",
                         i, layer.is_moe ? 1 : 0, layer.Get(model_keys::kMoeGate) ? 1 : 0, layer.NumExperts(),
                         packed_gate_up ? 1 : 0, packed_down ? 1 : 0, used_packed_expert_slices ? 1 : 0,
                         used_sep_expert_slices ? 1 : 0, expert_regex_hits, expert_name_candidates, gate_int4_bindings,
                         up_int4_bindings, down_int4_bindings, fused_gate_up_available, reason_code);
        }

        if (model->arch_flags.is_glm_dsa) {
            layer.Set(model_keys::kAttnQAProj, find_layer_tensor_with_tokens(layer, {"q_a_proj"}, {"indexer"}));
            layer.Set(model_keys::kAttnQANorm, find_layer_tensor_with_tokens(layer, {"q_a_layernorm"}, {"indexer"}));
            layer.Set(model_keys::kAttnQBProj, find_layer_tensor_with_tokens(layer, {"q_b_proj"}, {"indexer"}));
            layer.Set(model_keys::kAttnKvAProj, find_layer_tensor_with_tokens(layer, {"kv_a_proj_with_mqa"}));
            layer.Set(model_keys::kAttnKvANorm, find_layer_tensor_with_tokens(layer, {"kv_a_layernorm"}));
            layer.Set(model_keys::kAttnKvBProj, find_layer_tensor_with_tokens(layer, {"kv_b_proj"}));
            layer.Set(model_keys::kIndexerWqB, find_layer_tensor_with_tokens(layer, {"indexer", "wq_b"}));
            layer.Set(model_keys::kIndexerWk, find_layer_tensor_with_tokens(layer, {"indexer", "wk"}));
            layer.Set(model_keys::kIndexerKNorm, find_layer_tensor_with_tokens(layer, {"indexer", "k_norm"}));
            layer.Set(model_keys::kIndexerWeightsProj,
                      find_layer_tensor_with_tokens(layer, {"indexer", "weights_proj"}));
            if (!layer.Get(model_keys::kAttnOWeight)) {
                layer.Set(model_keys::kAttnOWeight, find_layer_tensor_with_tokens(layer, {"o_proj"}));
            }
        }

        if (!layer.Get(model_keys::kAttnQWeight)) {
            layer.Set(model_keys::kAttnQWeight, find_layer_tensor_with_tokens(layer, {"q_proj"}));
        }
        if (!layer.Get(model_keys::kAttnKWeight)) {
            layer.Set(model_keys::kAttnKWeight, find_layer_tensor_with_tokens(layer, {"k_proj"}));
        }
        if (!layer.Get(model_keys::kAttnVWeight)) {
            layer.Set(model_keys::kAttnVWeight, find_layer_tensor_with_tokens(layer, {"v_proj"}));
        }
        if (!layer.Get(model_keys::kAttnOWeight)) {
            layer.Set(model_keys::kAttnOWeight, find_layer_tensor_with_tokens(layer, {"o_proj"}));
        }
        if (!layer.Get(model_keys::kAttnQBias)) {
            layer.Set(model_keys::kAttnQBias, find_layer_tensor_with_tokens(layer, {"q_proj", "bias"}));
        }
        if (!layer.Get(model_keys::kAttnKBias)) {
            layer.Set(model_keys::kAttnKBias, find_layer_tensor_with_tokens(layer, {"k_proj", "bias"}));
        }
        if (!layer.Get(model_keys::kAttnVBias)) {
            layer.Set(model_keys::kAttnVBias, find_layer_tensor_with_tokens(layer, {"v_proj", "bias"}));
        }
        if (!layer.Get(model_keys::kAttnOBias)) {
            layer.Set(model_keys::kAttnOBias, find_layer_tensor_with_tokens(layer, {"o_proj", "bias"}));
        }
        if (!layer.Get(model_keys::kAttnQNorm)) {
            layer.Set(model_keys::kAttnQNorm, find_layer_tensor_with_tokens(layer, {"q_norm"}));
        }
        if (!layer.Get(model_keys::kAttnKNorm)) {
            layer.Set(model_keys::kAttnKNorm, find_layer_tensor_with_tokens(layer, {"k_norm"}, {"indexer"}));
        }
        if (!layer.Get(model_keys::kAttnVNorm)) {
            layer.Set(model_keys::kAttnVNorm, find_layer_tensor_with_tokens(layer, {"v_norm"}));
        }
    }

    // Detect shared experts from tensor presence when moe_n_shared_experts was not in metadata.
    // qwen35moe GGUFs use ffn_gate_shexp.weight tensors but no n_shared_experts GGUF key.
    if (model->moe_n_shared_experts == 0 && model->hparams.n_experts > 0) {
        for (const auto& layer : model->layers) {
            if (layer.is_moe && layer.Get(model_keys::kFfnGate)) {
                model->moe_n_shared_experts = 1;
                break;
            }
        }
    }
    if (model->hparams.n_experts > 0) {
        std::cout << "[DenseCore] MoE summary: experts=" << model->hparams.n_experts
                  << " top_k=" << model->hparams.n_experts_used << " shared_experts=" << model->moe_n_shared_experts
                  << " n_group=" << model->moe_n_group << " topk_group=" << model->moe_topk_group
                  << " norm_topk_prob=" << (model->moe_norm_topk_prob ? 1 : 0)
                  << " routed_scaling_factor=" << model->moe_routed_scaling_factor << std::endl;
    }

    // Auto-compute head dimensions from weight tensor shapes.
    // For hybrid SSM models (Qwen3.5), layer 0 may be SSM (no wk/wv),
    // so search for the first attention layer.
    struct ggml_tensor* wq_ref = nullptr;
    struct ggml_tensor* wk_ref = nullptr;
    struct ggml_tensor* wv_ref = nullptr;
    struct ggml_tensor* k_norm_ref = nullptr;
    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        auto* wq_i = model->layers[i].Get(model_keys::kAttnQWeight);
        auto* wk_i = model->layers[i].Get(model_keys::kAttnKWeight);
        auto* wv_i = model->layers[i].Get(model_keys::kAttnVWeight);
        auto* k_norm_i = model->layers[i].Get(model_keys::kAttnKNorm);
        if (wq_i && !wq_ref) wq_ref = wq_i;
        if (wk_i && !wk_ref) wk_ref = wk_i;
        if (wv_i && !wv_ref) wv_ref = wv_i;
        if (k_norm_i && !k_norm_ref) k_norm_ref = k_norm_i;
        if (wq_ref && wk_ref && wv_ref && k_norm_ref) break;
    }
    if (model->arch_flags.is_glm_dsa) {
        if (model->glm_qk_nope_head_dim > 0 || model->glm_qk_rope_head_dim > 0) {
            model->hparams.n_embd_head_k =
                static_cast<uint32_t>(model->glm_qk_nope_head_dim + model->glm_qk_rope_head_dim);
        }
        if (model->glm_v_head_dim > 0) {
            model->hparams.n_embd_head_v = static_cast<uint32_t>(model->glm_v_head_dim);
        }
        if (model->glm_qk_rope_head_dim > 0) {
            model->hparams.n_rot = static_cast<uint32_t>(model->glm_qk_rope_head_dim);
        }
    }
    if (model->arch_flags.is_gemma4 && model->gemma4_layer_n_head_kv.empty()) {
        if (model->hparams.n_head_kv > 0) {
            model->gemma4_layer_n_head_kv.assign(model->hparams.n_layer, model->hparams.n_head_kv);
        } else {
            return fail_load("Gemma4 GGUF is missing explicit attention.head_count_kv metadata");
        }
    }
    if (!model->gemma4_layer_n_head_kv.empty()) {
        uint32_t max_head_k = 0;
        uint32_t max_head_v = 0;
        for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
            const uint32_t layer_n_head_kv =
                (i < model->gemma4_layer_n_head_kv.size() && model->gemma4_layer_n_head_kv[i] > 0)
                    ? model->gemma4_layer_n_head_kv[i]
                    : model->hparams.n_head_kv;
            if (layer_n_head_kv == 0) continue;
            auto* wk_i = model->layers[i].Get(model_keys::kAttnKWeight);
            auto* wv_i = model->layers[i].Get(model_keys::kAttnVWeight);
            if (wk_i && wk_i->ne[1] > 0 && (wk_i->ne[1] % layer_n_head_kv) == 0) {
                max_head_k = std::max(max_head_k, static_cast<uint32_t>(wk_i->ne[1] / layer_n_head_kv));
            }
            if (wv_i && wv_i->ne[1] > 0 && (wv_i->ne[1] % layer_n_head_kv) == 0) {
                max_head_v = std::max(max_head_v, static_cast<uint32_t>(wv_i->ne[1] / layer_n_head_kv));
            } else if (wk_i && wk_i->ne[1] > 0 && (wk_i->ne[1] % layer_n_head_kv) == 0) {
                // Gemma4 omits V on some layers; use K width as the fallback.
                max_head_v = std::max(max_head_v, static_cast<uint32_t>(wk_i->ne[1] / layer_n_head_kv));
            }
            const bool is_sliding = i < model->gemma4_layer_is_sliding.size() &&
                                    model->gemma4_layer_is_sliding[static_cast<size_t>(i)] != 0;
            const uint32_t meta_key_len = is_sliding ? model->gemma4_key_length_swa : model->gemma4_key_length_full;
            const uint32_t meta_value_len =
                is_sliding ? model->gemma4_value_length_swa : model->gemma4_value_length_full;
            if (meta_key_len > 0) {
                max_head_k = std::max(max_head_k, meta_key_len);
            }
            if (meta_value_len > 0) {
                max_head_v = std::max(max_head_v, meta_value_len);
            }
        }
        if (max_head_k > 0) model->hparams.n_embd_head_k = max_head_k;
        if (max_head_v > 0) model->hparams.n_embd_head_v = max_head_v;
    } else if (model->arch_flags.is_gemma4) {
        const uint32_t gemma4_head_dim_k =
            std::max(std::max(model->gemma4_key_length_full, model->gemma4_key_length_swa),
                     k_norm_ref && k_norm_ref->ne[0] > 0 ? static_cast<uint32_t>(k_norm_ref->ne[0]) : 0u);
        const uint32_t gemma4_head_dim_v =
            std::max(std::max(model->gemma4_value_length_full, model->gemma4_value_length_swa), gemma4_head_dim_k);
        if (gemma4_head_dim_k > 0) {
            model->hparams.n_embd_head_k = gemma4_head_dim_k;
        }
        if (gemma4_head_dim_v > 0) {
            model->hparams.n_embd_head_v = gemma4_head_dim_v;
        }
    } else {
        if (model->hparams.n_embd_head_k == 0 && wk_ref) {
            model->hparams.n_embd_head_k = wk_ref->ne[1] / model->hparams.n_head_kv;
        }
        if (model->hparams.n_embd_head_v == 0 && wv_ref) {
            model->hparams.n_embd_head_v = wv_ref->ne[1] / model->hparams.n_head_kv;
        }
    }
    // Fallback to n_embd/n_head
    if (model->hparams.n_embd_head_k == 0) {
        model->hparams.n_embd_head_k = model->hparams.n_embd / model->hparams.n_head;
    }
    if (model->hparams.n_embd_head_v == 0) {
        model->hparams.n_embd_head_v = model->hparams.n_embd / model->hparams.n_head;
    }
    std::cout << "[DenseCore] Head dimensions: n_embd_head_k=" << model->hparams.n_embd_head_k
              << ", n_embd_head_v=" << model->hparams.n_embd_head_v << ", n_head=" << model->hparams.n_head
              << std::endl;

    // =========================================================================
    // Canonicalize LFM2 / LFM2.5 depthwise short-conv weights to F32.
    //
    // The short-conv decode custom op reads conv weights through a raw float
    // pointer, so dequantize each conv layer's weight into [channel][tap] layout
    // ([channel * kernel + tap]). GGUF stores the weight as [kernel x channels]
    // or [channels x kernel]; both are normalized here.
    // =========================================================================
    if (model->arch_flags.is_lfm2_shortconv) {
        const int kernel = model->lfm2_conv_kernel;
        const int channels = static_cast<int>(model->hparams.n_embd);
        const int n_conv = model->LFM2NumConvLayers();
        model->lfm2_conv_weight_f32.assign(static_cast<size_t>(std::max(0, n_conv)), {});
        auto dequant_to_f32 = [](struct ggml_tensor* t, std::vector<float>& out) -> bool {
            out.clear();
            if (!t) return false;
            const int64_t n = ggml_nelements(t);
            if (n <= 0) return false;
            out.resize(static_cast<size_t>(n));
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(out.data(), t->data, static_cast<size_t>(n) * sizeof(float));
                return true;
            }
            const auto* traits = ggml_get_type_traits(t->type);
            if (traits && traits->to_float) {
                traits->to_float(t->data, out.data(), n);
                return true;
            }
            out.clear();
            return false;
        };
        for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
            if (!model->IsLFM2ConvLayer(static_cast<int>(i))) continue;
            const int ordinal = model->LFM2ConvOrdinal(static_cast<int>(i));
            struct ggml_tensor* conv_w = model->layers[i].Get(model_keys::kShortConvConv);
            std::vector<float> raw;
            if (!conv_w || !dequant_to_f32(conv_w, raw)) {
                return fail_load("LFM2 layer " + std::to_string(i) + " missing/undecodable shortconv.conv.weight");
            }
            const bool kxc = (conv_w->ne[0] == kernel && conv_w->ne[1] == channels);
            const bool cxk = (conv_w->ne[0] == channels && conv_w->ne[1] == kernel);
            if (!kxc && !cxk) {
                return fail_load("LFM2 layer " + std::to_string(i) + " shortconv.conv.weight has unexpected shape");
            }
            std::vector<float>& out = model->lfm2_conv_weight_f32[static_cast<size_t>(ordinal)];
            out.assign(static_cast<size_t>(channels) * static_cast<size_t>(kernel), 0.0f);
            for (int ch = 0; ch < channels; ++ch) {
                for (int k = 0; k < kernel; ++k) {
                    // raw is row-major over (ne[1], ne[0]); index accordingly for each layout.
                    const float v = kxc ? raw[static_cast<size_t>(ch) * kernel + k]
                                        : raw[static_cast<size_t>(k) * channels + ch];
                    out[static_cast<size_t>(ch) * kernel + k] = v;
                }
            }
        }
        std::cout << "[DenseCore] Canonicalized " << n_conv << " LFM2 short-conv weight tensors (channels=" << channels
                  << ", kernel=" << kernel << ")" << std::endl;
    }

    // =========================================================================
    // Dequantize SSM callback tensors to F32.
    //
    // Qwen3.5's SSM decode path reads several tensors through raw float pointers.
    // GGUF may store these tensors as F16/quantized blocks, so keep explicit F32
    // copies to avoid type-dependent corruption in the custom callback path.
    // =========================================================================
    if (model->arch_flags.is_hybrid_ssm) {
        int ssm_ordinal = 0;
        auto dequant_raw = [](struct ggml_tensor* t, std::vector<float>& out) {
            out.clear();
            if (!t) return false;
            const int64_t n = ggml_nelements(t);
            if (n <= 0) return false;
            out.resize(static_cast<size_t>(n));
            const auto* traits = ggml_get_type_traits(t->type);
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(out.data(), t->data, static_cast<size_t>(n) * sizeof(float));
                return true;
            }
            if (traits && traits->to_float) {
                traits->to_float(t->data, out.data(), n);
                return true;
            }
            out.clear();
            return false;
        };
        const int conv_channels = model->ssm_inner_size + 2 * model->ssm_group_count * model->ssm_state_size;
        const int n_heads = model->ssm_time_step_rank;
        const int head_dim = model->ssm_inner_size / std::max(1, n_heads);
        const int conv_expected = conv_channels * model->ssm_conv_kernel;
        const int per_head_expected = n_heads;
        const int norm_head_expected = head_dim;
        const int norm_full_expected = model->ssm_inner_size;
        auto shape_string = [](const struct ggml_tensor* t) {
            if (!t) return std::string("<null>");
            std::string out = "[";
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (d != 0) out += "x";
                out += std::to_string(static_cast<long long>(t->ne[d]));
            }
            out += "]";
            return out;
        };
        auto validate_finite = [&](const char* label, const std::vector<float>& values, uint32_t layer_idx) -> bool {
            for (size_t i = 0; i < values.size(); ++i) {
                if (!std::isfinite(values[i])) {
                    std::cerr << "[DenseCore] FATAL: non-finite " << label << " value at layer " << layer_idx
                              << " elem " << i << ": " << values[i] << std::endl;
                    return false;
                }
            }
            return true;
        };
        auto canonicalize_conv1d = [&](const char* label, struct ggml_tensor* t, std::vector<float>& out,
                                       uint32_t layer_idx) -> bool {
            std::vector<float> raw;
            if (!dequant_raw(t, raw)) {
                std::cerr << "[DenseCore] FATAL: unable to dequantize " << label << " at layer " << layer_idx
                          << std::endl;
                return false;
            }
            if (!t || (t->ne[0] != model->ssm_conv_kernel && t->ne[1] != model->ssm_conv_kernel) ||
                (t->ne[0] != conv_channels && t->ne[1] != conv_channels)) {
                std::cerr << "[DenseCore] FATAL: invalid " << label << " shape at layer " << layer_idx << ": "
                          << shape_string(t) << ", expected [" << model->ssm_conv_kernel << "x" << conv_channels
                          << "] or [" << conv_channels << "x" << model->ssm_conv_kernel << "]" << std::endl;
                return false;
            }
            out.assign(static_cast<size_t>(conv_expected), 0.0f);
            if (t->ne[0] == model->ssm_conv_kernel && t->ne[1] == conv_channels) {
                out = std::move(raw);
                return validate_finite(label, out, layer_idx);
            }
            for (int ch = 0; ch < conv_channels; ++ch) {
                for (int k = 0; k < model->ssm_conv_kernel; ++k) {
                    out[static_cast<size_t>(ch) * model->ssm_conv_kernel + k] =
                        raw[static_cast<size_t>(k) * conv_channels + ch];
                }
            }
            return validate_finite(label, out, layer_idx);
        };
        auto canonicalize_head_by_embd = [&](const char* label, struct ggml_tensor* t, std::vector<float>& out,
                                             uint32_t layer_idx) -> bool {
            std::vector<float> raw;
            if (!dequant_raw(t, raw)) {
                std::cerr << "[DenseCore] FATAL: unable to dequantize " << label << " at layer " << layer_idx
                          << std::endl;
                return false;
            }
            if (!t || ggml_n_dims(t) < 2) {
                std::cerr << "[DenseCore] FATAL: invalid " << label << " rank at layer " << layer_idx << std::endl;
                return false;
            }
            if (Qwen35CanonicalizeHeadByEmbd(raw.data(), t->ne, static_cast<int>(model->hparams.n_embd), n_heads,
                                             &out)) {
                return validate_finite(label, out, layer_idx);
            }
            std::cerr << "[DenseCore] FATAL: invalid " << label << " shape at layer " << layer_idx << ": "
                      << shape_string(t) << ", expected [" << model->hparams.n_embd << "x" << n_heads << "] or ["
                      << n_heads << "x" << model->hparams.n_embd << "]" << std::endl;
            return false;
        };
        auto canonicalize_fused_ba = [&](const char* label, struct ggml_tensor* t, std::vector<float>& beta_out,
                                         std::vector<float>& alpha_out, uint32_t layer_idx) -> bool {
            std::vector<float> raw;
            if (!dequant_raw(t, raw)) {
                std::cerr << "[DenseCore] FATAL: unable to dequantize " << label << " at layer " << layer_idx
                          << std::endl;
                return false;
            }
            if (Qwen35CanonicalizeFusedBA(raw.data(), t ? t->ne : nullptr, static_cast<int>(model->hparams.n_embd),
                                          n_heads, model->ssm_group_count, &beta_out, &alpha_out)) {
                if (model->variant == ModelVariant::QWEN36 && model->ssm_group_count > 0 &&
                    model->ssm_group_count != n_heads) {
                    // llama.cpp's qwen35moe path treats V heads as tiled after
                    // conversion and repeats Q/K with h % num_k_heads. Keep the
                    // alpha/beta rows in that same tiled V-head order.
                    Qwen35ReorderVHeadsGroupedToTiled(&beta_out, static_cast<int>(model->hparams.n_embd),
                                                      model->ssm_group_count, n_heads);
                    Qwen35ReorderVHeadsGroupedToTiled(&alpha_out, static_cast<int>(model->hparams.n_embd),
                                                      model->ssm_group_count, n_heads);
                }
                return validate_finite("ssm_ba/beta", beta_out, layer_idx) &&
                       validate_finite("ssm_ba/alpha", alpha_out, layer_idx);
            }
            std::cerr << "[DenseCore] FATAL: invalid " << label << " shape at layer " << layer_idx << ": "
                      << shape_string(t) << ", expected [" << model->hparams.n_embd << "x" << (2 * n_heads) << "] or ["
                      << (2 * n_heads) << "x" << model->hparams.n_embd << "] with grouped [beta,alpha]" << " layout"
                      << std::endl;
            return false;
        };
        auto canonicalize_per_head = [&](const char* label, struct ggml_tensor* t, std::vector<float>& out,
                                         uint32_t layer_idx) -> bool {
            std::vector<float> raw;
            if (!dequant_raw(t, raw)) {
                std::cerr << "[DenseCore] FATAL: unable to dequantize " << label << " at layer " << layer_idx
                          << std::endl;
                return false;
            }
            if (!Qwen35CanonicalizePerHeadVector(raw.data(), t ? t->ne : nullptr, per_head_expected, &out)) {
                std::cerr << "[DenseCore] FATAL: invalid " << label << " shape at layer " << layer_idx << ": "
                          << shape_string(t) << ", expected [" << per_head_expected << "x1x1x1]" << std::endl;
                return false;
            }
            return validate_finite(label, out, layer_idx);
        };
        auto canonicalize_norm = [&](const char* label, struct ggml_tensor* t, std::vector<float>& out,
                                     Qwen35SSMNormLayout* out_layout, uint32_t layer_idx) -> bool {
            std::vector<float> raw;
            if (!dequant_raw(t, raw)) {
                std::cerr << "[DenseCore] FATAL: unable to dequantize " << label << " at layer " << layer_idx
                          << std::endl;
                return false;
            }
            const Qwen35SSMNormLayout norm_layout =
                Qwen35CanonicalizeNorm(raw.data(), t ? t->ne : nullptr, norm_head_expected, norm_full_expected, &out);
            if (norm_layout == Qwen35SSMNormLayout::INVALID) {
                std::cerr << "[DenseCore] FATAL: invalid " << label << " shape at layer " << layer_idx << ": "
                          << shape_string(t) << ", expected [" << norm_head_expected << "x1x1x1] or ["
                          << norm_full_expected << "x1x1x1]" << std::endl;
                return false;
            }
            if (out_layout) {
                *out_layout = norm_layout;
            }
            return validate_finite(label, out, layer_idx);
        };
        for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
            const bool is_ssm = model->IsHybridSSMLayer(static_cast<int>(i));
            if (!is_ssm) continue;
            if (ssm_ordinal < static_cast<int>(model->ssm_layer_states.size())) {
                auto& state = model->ssm_layer_states[ssm_ordinal];
                auto* conv = model->layers[i].Get(model_keys::kSSMConv1d);
                auto* alpha = model->layers[i].Get(model_keys::kSSMAlpha);
                auto* beta = model->layers[i].Get(model_keys::kSSMBeta);
                auto* dt = model->layers[i].Get(model_keys::kSSMDtBias);
                auto* ssm_a = model->layers[i].Get(model_keys::kSSMA);
                auto* norm = model->layers[i].Get(model_keys::kSSMNorm);
                const bool has_fused_ba = alpha && beta && alpha == beta;

                if (!canonicalize_conv1d("ssm_conv1d", conv, state.conv1d_f32, i) ||
                    !(has_fused_ba ? canonicalize_fused_ba("ssm_ba", alpha, state.beta_f32, state.alpha_f32, i)
                                   : (canonicalize_head_by_embd("ssm_alpha", alpha, state.alpha_f32, i) &&
                                      canonicalize_head_by_embd("ssm_beta", beta, state.beta_f32, i))) ||
                    !canonicalize_per_head("ssm_dt_bias", dt, state.dt_bias_f32, i) ||
                    !canonicalize_per_head("ssm_a/A_log", ssm_a, state.a_log_f32, i) ||
                    !canonicalize_norm("ssm_norm", norm, state.norm_f32, &state.norm_layout, i)) {
                    if (model->backend) ggml_backend_free(model->backend);
                    gguf_free(ctx_gguf);
                    if (ctx_w) ggml_free(ctx_w);
                    delete model;
                    return nullptr;
                }
                if (i == 0) {
                    const char* ssm_debug_values = std::getenv("DENSECORE_DEBUG_QWEN36_SSM_VALUES");
                    std::cout << "[DenseCore] Qwen3.5 SSM tensor types: conv1d=" << ggml_type_name(conv->type)
                              << " alpha=" << ggml_type_name(alpha->type) << " beta=" << ggml_type_name(beta->type)
                              << " dt_bias=" << ggml_type_name(dt->type) << " ssm_a=" << ggml_type_name(ssm_a->type)
                              << " norm=" << ggml_type_name(norm->type) << std::endl;
                    std::cout << "[DenseCore] Qwen3.5 SSM tensor names: alpha="
                              << (alpha && alpha->name[0] ? alpha->name : "<unnamed>")
                              << " beta=" << (beta && beta->name[0] ? beta->name : "<unnamed>")
                              << " fused_ba=" << (has_fused_ba ? "true" : "false") << std::endl;
                    std::cout << "[DenseCore] Qwen3.5 SSM tensor shapes: conv1d=" << shape_string(conv)
                              << " alpha=" << shape_string(alpha) << " beta=" << shape_string(beta)
                              << " dt_bias=" << shape_string(dt) << " A_log=" << shape_string(ssm_a)
                              << " norm=" << shape_string(norm) << std::endl;
                    std::cout << "[DenseCore] Qwen3.5 SSM canonical layouts: conv1d=" << state.conv1d_f32.size()
                              << " alpha=" << state.alpha_f32.size() << " beta=" << state.beta_f32.size()
                              << " dt_bias=" << state.dt_bias_f32.size() << " A_log=" << state.a_log_f32.size()
                              << " norm=" << state.norm_f32.size() << " norm_semantics="
                              << (state.norm_layout == Qwen35SSMNormLayout::SHARED_HEAD_DIM     ? "shared_head_dim"
                                  : state.norm_layout == Qwen35SSMNormLayout::FLATTENED_D_INNER ? "flattened_d_inner"
                                                                                                : "invalid")
                              << std::endl;
                    if (ssm_debug_values && ssm_debug_values[0] != '\0' && std::strcmp(ssm_debug_values, "0") != 0) {
                        auto print_head = [](const char* label, const std::vector<float>& values) {
                            std::cerr << "[QWEN36_SSM_VALUES] " << label << "=";
                            const size_t limit = std::min<size_t>(8, values.size());
                            for (size_t idx = 0; idx < limit; ++idx) {
                                if (idx != 0) std::cerr << ",";
                                std::cerr << values[idx];
                            }
                            std::cerr << std::endl;
                        };
                        print_head("a_log", state.a_log_f32);
                        print_head("dt_bias", state.dt_bias_f32);
                        print_head("alpha", state.alpha_f32);
                        print_head("beta", state.beta_f32);
                    }
                }
            }
            ++ssm_ordinal;
        }
    }

    // =========================================================================
    // QWEN3 ARCHITECTURE VALIDATION
    // =========================================================================
    // Qwen3 requires per-head Q/K normalization. If these tensors are missing,
    // the model will produce incorrect outputs. Fail fast instead of silently
    // corrupting results.
    // =========================================================================
    if (model->arch == ModelArch::QWEN3 || model->arch == ModelArch::QWEN35) {
        bool has_qk_norms = true;
        for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
            // Hybrid SSM models (Qwen3.5): SSM layers have no Q/K norms — skip them.
            if (model->arch_flags.is_hybrid_ssm && model->IsHybridSSMLayer(static_cast<int>(i))) {
                continue;
            }
            if (!model->layers[i].Get(model_keys::kAttnQNorm) || !model->layers[i].Get(model_keys::kAttnKNorm)) {
                has_qk_norms = false;
                break;
            }
        }
        if (!has_qk_norms) {
            std::cerr << "[DenseCore] FATAL: " << arch << " model requires attn_q_norm and "
                      << "attn_k_norm tensors on full-attention layers, but they are missing from GGUF file."
                      << std::endl;
            std::cerr << "[DenseCore] This is a corrupted or incompatible GGUF. "
                      << "Aborting load to prevent incorrect outputs." << std::endl;
            if (model->backend) ggml_backend_free(model->backend);
            gguf_free(ctx_gguf);
            if (ctx_w) ggml_free(ctx_w);
            delete model;
            return nullptr;
        }
        std::cout << "[DenseCore] " << arch << " architecture validated: Q/K norms present on attention layers"
                  << std::endl;
    }

    if (model->arch_flags.is_hybrid_ssm) {
        bool saw_qwen_vision_tensor = false;
        for (const char* probe : {"visual.patch_embed.proj.weight", "vision_tower.patch_embed.proj.weight",
                                  "vision_model.embeddings.patch_embedding.weight"}) {
            if (ggml_get_tensor(model->ctx_w, probe) != nullptr) {
                saw_qwen_vision_tensor = true;
                break;
            }
        }
        if (saw_qwen_vision_tensor) {
            std::cout << "[DenseCore] Qwen3.6 vision tensors detected; running language-only text path; "
                         "multimodal execution is unsupported in this patch."
                      << std::endl;
        }
    }

    // =========================================================================
    // VISION ENCODER LOADING (ViT, CLIP, SigLIP)
    // =========================================================================
    if (model->arch == ModelArch::VIT || model->arch == ModelArch::CLIP_VISION || model->arch == ModelArch::SIGLIP) {
        model->has_vision = true;
        model->vision_encoder = std::make_unique<VisionEncoder>();

        // Load vision-specific hyperparameters
        get_u32("image_size", model->vision_hparams.image_size);
        get_u32("patch_size", model->vision_hparams.patch_size);
        get_u32("embedding_length", model->vision_hparams.n_embd);
        get_u32("attention.head_count", model->vision_hparams.n_head);
        get_u32("block_count", model->vision_hparams.n_layer);
        get_u32("feed_forward_length", model->vision_hparams.n_intermediate);
        get_f32("attention.layer_norm_epsilon", model->vision_hparams.layer_norm_eps);

        // Fallback: infer from tensor shapes if not in metadata
        if (model->vision_hparams.n_embd == 0) {
            struct ggml_tensor* patch_w = get_tensor("v.patch_embd.weight");
            if (!patch_w) patch_w = get_tensor("visual.patch_embed.proj.weight");
            if (!patch_w) patch_w = get_tensor("vision_model.embeddings.patch_embedding.weight");
            if (patch_w) {
                model->vision_hparams.n_embd = patch_w->ne[0];
                model->vision_hparams.patch_size = patch_w->ne[2];
            }
        }

        std::cout << "[DenseCore] Vision params: image_size=" << model->vision_hparams.image_size
                  << ", patch_size=" << model->vision_hparams.patch_size << ", n_embd=" << model->vision_hparams.n_embd
                  << ", n_layer=" << model->vision_hparams.n_layer << ", n_head=" << model->vision_hparams.n_head
                  << std::endl;

        // Load vision encoder tensors (try multiple naming conventions)
        auto get_vision_tensor = [&](const std::vector<std::string>& names) -> struct ggml_tensor* {
            for (const auto& name : names) {
                struct ggml_tensor* t = ggml_get_tensor(model->ctx_w, name.c_str());
                if (t) return t;
            }
            return nullptr;
        };

        model->vision_encoder->patch_embed_w =
            get_vision_tensor({"v.patch_embd.weight", "visual.patch_embed.proj.weight",
                               "vision_model.embeddings.patch_embedding.weight"});
        model->vision_encoder->patch_embed_b = get_vision_tensor(
            {"v.patch_embd.bias", "visual.patch_embed.proj.bias", "vision_model.embeddings.patch_embedding.bias"});
        model->vision_encoder->cls_token =
            get_vision_tensor({"v.class_embd", "visual.class_embedding", "vision_model.embeddings.class_embedding"});
        model->vision_encoder->pos_embed = get_vision_tensor({"v.position_embd.weight", "visual.positional_embedding",
                                                              "vision_model.embeddings.position_embedding.weight"});
        model->vision_encoder->ln_post_w =
            get_vision_tensor({"v.post_ln.weight", "visual.ln_post.weight", "vision_model.post_layernorm.weight"});
        model->vision_encoder->ln_post_b =
            get_vision_tensor({"v.post_ln.bias", "visual.ln_post.bias", "vision_model.post_layernorm.bias"});
        model->vision_encoder->proj = get_vision_tensor({"v.proj", "visual.proj", "visual_projection.weight"});

        // Load vision encoder layers
        model->vision_encoder->layers.resize(model->vision_hparams.n_layer);
        for (uint32_t i = 0; i < model->vision_hparams.n_layer; ++i) {
            auto& layer = model->vision_encoder->layers[i];
            std::string p1 = "v.blk." + std::to_string(i) + ".";
            std::string p2 = "visual.transformer.resblocks." + std::to_string(i) + ".";
            std::string p3 = "vision_model.encoder.layers." + std::to_string(i) + ".";

            layer.ln1_w = get_vision_tensor({p1 + "ln1.weight", p2 + "ln_1.weight", p3 + "layer_norm1.weight"});
            layer.ln1_b = get_vision_tensor({p1 + "ln1.bias", p2 + "ln_1.bias", p3 + "layer_norm1.bias"});
            layer.ln2_w = get_vision_tensor({p1 + "ln2.weight", p2 + "ln_2.weight", p3 + "layer_norm2.weight"});
            layer.ln2_b = get_vision_tensor({p1 + "ln2.bias", p2 + "ln_2.bias", p3 + "layer_norm2.bias"});

            // Attention (try fused QKV first, then separate)
            layer.wqkv = get_vision_tensor({p1 + "attn_qkv.weight", p2 + "attn.in_proj_weight"});
            layer.bqkv = get_vision_tensor({p1 + "attn_qkv.bias", p2 + "attn.in_proj_bias"});
            if (!layer.wqkv) {
                layer.wq = get_vision_tensor({p1 + "attn_q.weight", p3 + "self_attn.q_proj.weight"});
                layer.wk = get_vision_tensor({p1 + "attn_k.weight", p3 + "self_attn.k_proj.weight"});
                layer.wv = get_vision_tensor({p1 + "attn_v.weight", p3 + "self_attn.v_proj.weight"});
            }
            layer.wo = get_vision_tensor(
                {p1 + "attn_out.weight", p2 + "attn.out_proj.weight", p3 + "self_attn.out_proj.weight"});
            layer.bo =
                get_vision_tensor({p1 + "attn_out.bias", p2 + "attn.out_proj.bias", p3 + "self_attn.out_proj.bias"});

            // MLP
            layer.mlp_fc1_w = get_vision_tensor({p1 + "ffn_up.weight", p2 + "mlp.c_fc.weight", p3 + "mlp.fc1.weight"});
            layer.mlp_fc1_b = get_vision_tensor({p1 + "ffn_up.bias", p2 + "mlp.c_fc.bias", p3 + "mlp.fc1.bias"});
            layer.mlp_fc2_w =
                get_vision_tensor({p1 + "ffn_down.weight", p2 + "mlp.c_proj.weight", p3 + "mlp.fc2.weight"});
            layer.mlp_fc2_b = get_vision_tensor({p1 + "ffn_down.bias", p2 + "mlp.c_proj.bias", p3 + "mlp.fc2.bias"});
        }

        // Validate critical tensors
        if (!model->vision_encoder->patch_embed_w) {
            std::cerr << "[DenseCore] WARNING: Vision patch_embed weight not found. "
                      << "Check GGUF tensor naming convention." << std::endl;
        } else {
            std::cout << "[DenseCore] Vision encoder loaded: " << model->vision_encoder->layers.size() << " layers"
                      << std::endl;
        }
    }

    // =========================================================================
    // WHISPER MODEL LOADING (Encoder-Decoder)
    // =========================================================================
    if (model->arch == ModelArch::WHISPER) {
        model->has_whisper = true;
        model->whisper_model = std::make_unique<WhisperModel>();

        // Load whisper-specific hyperparameters
        get_u32("audio.ctx", model->whisper_hparams.n_audio_ctx);
        get_u32("audio.head_count", model->whisper_hparams.n_audio_head);
        get_u32("audio.layer_count", model->whisper_hparams.n_audio_layer);
        get_u32("audio.state", model->whisper_hparams.n_audio_state);
        get_u32("text.ctx", model->whisper_hparams.n_text_ctx);
        get_u32("text.head_count", model->whisper_hparams.n_text_head);
        get_u32("text.layer_count", model->whisper_hparams.n_text_layer);
        get_u32("text.state", model->whisper_hparams.n_text_state);
        get_u32("vocab_size", model->whisper_hparams.n_vocab);
        get_f32("attention.layer_norm_epsilon", model->whisper_hparams.layer_norm_eps);

        std::cout << "[DenseCore] Whisper params: encoder_layers=" << model->whisper_hparams.n_audio_layer
                  << ", decoder_layers=" << model->whisper_hparams.n_text_layer
                  << ", n_mels=" << model->whisper_hparams.n_mels << ", n_vocab=" << model->whisper_hparams.n_vocab
                  << std::endl;

        // Load audio frontend (Conv1D layers)
        model->whisper_model->conv1_w = get_tensor("encoder.conv1.weight");
        model->whisper_model->conv1_b = get_tensor("encoder.conv1.bias");
        model->whisper_model->conv2_w = get_tensor("encoder.conv2.weight");
        model->whisper_model->conv2_b = get_tensor("encoder.conv2.bias");

        // Positional embeddings
        model->whisper_model->encoder_pos = get_tensor("encoder.position_embedding.weight");
        model->whisper_model->decoder_pos = get_tensor("decoder.position_embedding.weight");

        // Token embedding
        model->whisper_model->tok_embed = get_tensor("decoder.token_embedding.weight");

        // Final layer norms
        model->whisper_model->encoder_ln_w = get_tensor("encoder.ln_post.weight");
        model->whisper_model->encoder_ln_b = get_tensor("encoder.ln_post.bias");
        model->whisper_model->decoder_ln_w = get_tensor("decoder.ln.weight");
        model->whisper_model->decoder_ln_b = get_tensor("decoder.ln.bias");

        // Output projection
        model->whisper_model->output = get_tensor("decoder.output.weight");
        if (!model->whisper_model->output) {
            model->whisper_model->output = model->whisper_model->tok_embed;  // Tied weights
        }

        // Load encoder layers
        model->whisper_model->encoder_layers.resize(model->whisper_hparams.n_audio_layer);
        for (uint32_t i = 0; i < model->whisper_hparams.n_audio_layer; ++i) {
            auto& layer = model->whisper_model->encoder_layers[i];
            std::string p = "encoder.blocks." + std::to_string(i) + ".";

            layer.ln1_w = get_tensor(p + "attn_ln.weight");
            layer.ln1_b = get_tensor(p + "attn_ln.bias");
            layer.ln2_w = get_tensor(p + "mlp_ln.weight");
            layer.ln2_b = get_tensor(p + "mlp_ln.bias");

            layer.wq = get_tensor(p + "attn.query.weight");
            layer.wk = get_tensor(p + "attn.key.weight");
            layer.wv = get_tensor(p + "attn.value.weight");
            layer.bq = get_tensor(p + "attn.query.bias");
            layer.bv = get_tensor(p + "attn.value.bias");
            layer.wo = get_tensor(p + "attn.out.weight");
            layer.bo = get_tensor(p + "attn.out.bias");

            layer.mlp_fc1_w = get_tensor(p + "mlp.0.weight");
            layer.mlp_fc1_b = get_tensor(p + "mlp.0.bias");
            layer.mlp_fc2_w = get_tensor(p + "mlp.2.weight");
            layer.mlp_fc2_b = get_tensor(p + "mlp.2.bias");
        }

        // Load decoder layers (with cross-attention)
        model->whisper_model->decoder_layers.resize(model->whisper_hparams.n_text_layer);
        for (uint32_t i = 0; i < model->whisper_hparams.n_text_layer; ++i) {
            auto& layer = model->whisper_model->decoder_layers[i];
            std::string p = "decoder.blocks." + std::to_string(i) + ".";

            layer.ln1_w = get_tensor(p + "attn_ln.weight");
            layer.ln1_b = get_tensor(p + "attn_ln.bias");
            layer.ln2_w = get_tensor(p + "cross_attn_ln.weight");
            layer.ln2_b = get_tensor(p + "cross_attn_ln.bias");
            layer.ln3_w = get_tensor(p + "mlp_ln.weight");
            layer.ln3_b = get_tensor(p + "mlp_ln.bias");

            // Self-attention
            layer.self_wq = get_tensor(p + "attn.query.weight");
            layer.self_wk = get_tensor(p + "attn.key.weight");
            layer.self_wv = get_tensor(p + "attn.value.weight");
            layer.self_bq = get_tensor(p + "attn.query.bias");
            layer.self_bv = get_tensor(p + "attn.value.bias");
            layer.self_wo = get_tensor(p + "attn.out.weight");
            layer.self_bo = get_tensor(p + "attn.out.bias");

            // Cross-attention
            layer.cross_wq = get_tensor(p + "cross_attn.query.weight");
            layer.cross_wk = get_tensor(p + "cross_attn.key.weight");
            layer.cross_wv = get_tensor(p + "cross_attn.value.weight");
            layer.cross_bq = get_tensor(p + "cross_attn.query.bias");
            layer.cross_bv = get_tensor(p + "cross_attn.value.bias");
            layer.cross_wo = get_tensor(p + "cross_attn.out.weight");
            layer.cross_bo = get_tensor(p + "cross_attn.out.bias");

            // MLP
            layer.mlp_fc1_w = get_tensor(p + "mlp.0.weight");
            layer.mlp_fc1_b = get_tensor(p + "mlp.0.bias");
            layer.mlp_fc2_w = get_tensor(p + "mlp.2.weight");
            layer.mlp_fc2_b = get_tensor(p + "mlp.2.bias");
        }

        std::cout << "[DenseCore] Whisper model loaded: " << model->whisper_model->encoder_layers.size()
                  << " encoder layers, " << model->whisper_model->decoder_layers.size() << " decoder layers"
                  << std::endl;
    }

    // Initialize pre-computed RoPE table for LLM architectures only
    // Vision models use absolute/learned positional embeddings
    // Whisper uses sinusoidal positional encodings (pre-computed)
    bool needs_rope = !model->has_vision && !model->has_whisper && model->arch != ModelArch::UNKNOWN &&
                      model->arch != ModelArch::BERT;
    if (needs_rope && model->hparams.n_ctx > 0 && model->hparams.n_rot > 0) {
        std::cout << "[DenseCore] Initializing RoPE table..." << std::endl;
        InitRoPETable(model);
        std::cout << "[DenseCore] RoPE table initialized: " << model->rope_cos_sin.size() << " values ("
                  << model->hparams.n_ctx << " positions × " << model->rope_head_dim << " dims)" << std::endl;
    }

    // ==========================================================================
    // Universal Graph Execution: Register weight-aware graph builders
    // ==========================================================================
    // For vision/whisper models, register builders that can use the loaded weights
    // through the universal graph execution path in worker.cpp
    // ==========================================================================
    if (densecore::ModelGraphBridge::RegisterFromModel(model)) {
        const char* graph_name = densecore::ModelGraphBridge::GetGraphName(model);
        if (graph_name) {
            std::cout << "[DenseCore] Graph builder registered for: " << graph_name << std::endl;
        } else {
            std::cout << "[DenseCore] Universal template graph builders registered." << std::endl;
        }
    }

    // ==========================================================================
    // Custom Quantized Weight Bindings (INT4 / FP8)
    // ==========================================================================
    // Connect split tensors and metadata to runtime bindings used by smart_mul_mat.
    // INT4 format:
    //   - {name} (packed int4 bytes in GGML_TYPE_I8)
    //   - {name}_scales (F32)
    //   - {name}_zeros  (F32)
    //   - KV: densecore.int4.{name}.group_size, .K, .N
    //
    // FP8 format (optional/custom):
    //   - {name} (packed fp8 bytes in GGML_TYPE_I8)
    //   - KV: densecore.fp8.{name}.format, .K, .N
    // ==========================================================================
    auto starts_with = [](const std::string& s, const std::string& prefix) {
        return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
    };
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    auto get_kv_u32 = [&](const std::string& key, uint32_t& out) -> bool {
        int idx = gguf_find_key(ctx_gguf, key.c_str());
        if (idx < 0) return false;
        const gguf_type type = gguf_get_kv_type(ctx_gguf, idx);
        switch (type) {
        case GGUF_TYPE_UINT8: out = gguf_get_val_u8(ctx_gguf, idx); return true;
        case GGUF_TYPE_INT8: out = static_cast<uint32_t>(gguf_get_val_i8(ctx_gguf, idx)); return true;
        case GGUF_TYPE_UINT16: out = gguf_get_val_u16(ctx_gguf, idx); return true;
        case GGUF_TYPE_INT16: out = static_cast<uint32_t>(gguf_get_val_i16(ctx_gguf, idx)); return true;
        case GGUF_TYPE_UINT32: out = gguf_get_val_u32(ctx_gguf, idx); return true;
        case GGUF_TYPE_INT32: out = static_cast<uint32_t>(gguf_get_val_i32(ctx_gguf, idx)); return true;
        case GGUF_TYPE_UINT64: out = static_cast<uint32_t>(gguf_get_val_u64(ctx_gguf, idx)); return true;
        case GGUF_TYPE_INT64: out = static_cast<uint32_t>(gguf_get_val_i64(ctx_gguf, idx)); return true;
        default: return false;
        }
    };
    auto get_kv_i64 = [&](const std::string& key, int64_t& out) -> bool {
        int idx = gguf_find_key(ctx_gguf, key.c_str());
        if (idx < 0) return false;
        const gguf_type type = gguf_get_kv_type(ctx_gguf, idx);
        switch (type) {
        case GGUF_TYPE_UINT8: out = gguf_get_val_u8(ctx_gguf, idx); return true;
        case GGUF_TYPE_INT8: out = gguf_get_val_i8(ctx_gguf, idx); return true;
        case GGUF_TYPE_UINT16: out = gguf_get_val_u16(ctx_gguf, idx); return true;
        case GGUF_TYPE_INT16: out = gguf_get_val_i16(ctx_gguf, idx); return true;
        case GGUF_TYPE_UINT32: out = gguf_get_val_u32(ctx_gguf, idx); return true;
        case GGUF_TYPE_INT32: out = gguf_get_val_i32(ctx_gguf, idx); return true;
        case GGUF_TYPE_UINT64: out = static_cast<int64_t>(gguf_get_val_u64(ctx_gguf, idx)); return true;
        case GGUF_TYPE_INT64: out = gguf_get_val_i64(ctx_gguf, idx); return true;
        default: return false;
        }
    };

    {
        const int n_kv = gguf_get_n_kv(ctx_gguf);
        const std::string int4_prefix = "densecore.int4.";
        const std::string int4_suffix = ".group_size";
        const std::string fp8_prefix = "densecore.fp8.";
        const std::string fp8_suffix = ".format";

        int int4_bound = 0;
        int fp8_bound = 0;

        for (int i = 0; i < n_kv; ++i) {
            const char* key_c = gguf_get_key(ctx_gguf, i);
            if (!key_c) continue;
            const std::string key(key_c);

            if (starts_with(key, int4_prefix) && ends_with(key, int4_suffix)) {
                const std::string tensor_name =
                    key.substr(int4_prefix.size(), key.size() - int4_prefix.size() - int4_suffix.size());
                struct ggml_tensor* packed = get_tensor(tensor_name);
                struct ggml_tensor* scales = get_tensor(tensor_name + "_scales");
                struct ggml_tensor* zeros = get_tensor(tensor_name + "_zeros");
                if (!packed || !scales || !zeros || !packed->data || !scales->data || !zeros->data) {
                    continue;
                }

                uint32_t group_size_u32 = 0;
                if (!get_kv_u32(key, group_size_u32) || group_size_u32 == 0) {
                    continue;
                }
                if ((group_size_u32 & 1u) != 0u) {
                    std::cerr << "[DenseCore] WARN: INT4 binding skipped for " << tensor_name
                              << " (group_size must be even, got " << group_size_u32 << ")" << std::endl;
                    continue;
                }

                int64_t k = 0;
                int64_t n = 0;
                if (!get_kv_i64(int4_prefix + tensor_name + ".K", k)) continue;
                if (!get_kv_i64(int4_prefix + tensor_name + ".N", n)) continue;
                if (k <= 0 || n <= 0) continue;
                if (k % static_cast<int64_t>(group_size_u32) != 0) {
                    std::cerr << "[DenseCore] WARN: INT4 binding skipped for " << tensor_name
                              << " (K must be divisible by group_size, K=" << k << ", group_size=" << group_size_u32
                              << ")" << std::endl;
                    continue;
                }

                const int64_t expected_packed_bytes = n * ((k + 1) / 2);
                if (expected_packed_bytes > static_cast<int64_t>(ggml_nbytes(packed))) {
                    std::cerr << "[DenseCore] WARN: INT4 binding size mismatch for " << tensor_name
                              << " (expected packed bytes=" << expected_packed_bytes
                              << ", actual=" << static_cast<int64_t>(ggml_nbytes(packed)) << ")" << std::endl;
                    continue;
                }

                const int64_t group_size = static_cast<int64_t>(group_size_u32);
                const int64_t num_groups = (group_size > 0) ? (k / group_size) : 0;
                if (num_groups <= 0) continue;
                const int64_t expected_meta = n * num_groups;
                const int64_t scales_elems = static_cast<int64_t>(ggml_nelements(scales));
                const int64_t zeros_elems = static_cast<int64_t>(ggml_nelements(zeros));
                if (expected_meta > scales_elems || expected_meta > zeros_elems) {
                    std::cerr << "[DenseCore] WARN: INT4 binding metadata mismatch for " << tensor_name
                              << " (expected meta elements=" << expected_meta << ", scales=" << scales_elems
                              << ", zeros=" << zeros_elems << ")" << std::endl;
                    continue;
                }

                TransformerModel::Int4WeightBinding binding;
                binding.packed = packed;
                binding.scales = scales;
                binding.zeros = zeros;
                binding.group_size = static_cast<int>(group_size_u32);
                binding.k = k;
                binding.n = n;
                model->int4_weight_bindings[packed] = binding;
                int4_bound++;
                continue;
            }

            if (starts_with(key, fp8_prefix) && ends_with(key, fp8_suffix)) {
                if (gguf_get_kv_type(ctx_gguf, i) != GGUF_TYPE_STRING) {
                    continue;
                }
                const std::string tensor_name =
                    key.substr(fp8_prefix.size(), key.size() - fp8_prefix.size() - fp8_suffix.size());
                struct ggml_tensor* packed = get_tensor(tensor_name);
                if (!packed || !packed->data) continue;

                const std::string format = gguf_get_val_str(ctx_gguf, i);
                TransformerModel::FP8Format fp8_format;
                if (format == "e5m2") {
                    fp8_format = TransformerModel::FP8Format::E5M2;
                } else if (format == "e4m3fn") {
                    fp8_format = TransformerModel::FP8Format::E4M3FN;
                } else {
                    continue;
                }

                int64_t k = 0;
                int64_t n = 0;
                if (!get_kv_i64(fp8_prefix + tensor_name + ".K", k)) continue;
                if (!get_kv_i64(fp8_prefix + tensor_name + ".N", n)) continue;
                if (k <= 0 || n <= 0) continue;

                const int64_t expected_bytes = n * k;
                if (expected_bytes > static_cast<int64_t>(ggml_nbytes(packed))) {
                    std::cerr << "[DenseCore] WARN: FP8 binding size mismatch for " << tensor_name
                              << " (expected bytes=" << expected_bytes
                              << ", actual=" << static_cast<int64_t>(ggml_nbytes(packed)) << ")" << std::endl;
                    continue;
                }

                TransformerModel::FP8WeightBinding binding;
                binding.packed = packed;
                binding.format = fp8_format;
                binding.k = k;
                binding.n = n;
                model->fp8_weight_bindings[packed] = binding;
                fp8_bound++;
            }
        }

        if (int4_bound > 0) {
            std::cout << "[DenseCore] Registered " << int4_bound << " INT4 custom weight bindings" << std::endl;
        }
        if (fp8_bound > 0) {
            std::cout << "[DenseCore] Registered " << fp8_bound << " FP8 custom weight bindings" << std::endl;
        }

        auto bind_expert_view_int4 = [&](ggml_tensor* tensor) {
            if (!tensor || model->int4_weight_bindings.find(tensor) != model->int4_weight_bindings.end()) {
                return;
            }
            const ggml_tensor* root_tensor = nullptr;
            TransformerModel::Int4WeightBinding root_binding{};
            size_t total_view_offs = 0;
            if (!FindInt4BindingInViewChain(model, tensor, &root_tensor, &root_binding, &total_view_offs)) {
                return;
            }
            TransformerModel::Int4WeightBinding resolved{};
            if (ResolveStackedInt4ViewBinding(model->ctx_views ? model->ctx_views : model->ctx_w, root_tensor,
                                              root_binding, tensor, total_view_offs, &resolved)) {
                model->int4_weight_bindings[tensor] = resolved;
            }
        };

        for (size_t layer_idx = 0; layer_idx < model->layers.size(); ++layer_idx) {
            auto& layer = model->layers[layer_idx];
            if (!layer.is_moe || layer.NumExperts() == 0) {
                continue;
            }
            size_t gate_int4_bindings = 0;
            size_t up_int4_bindings = 0;
            size_t down_int4_bindings = 0;
            size_t fused_gate_up_available = 0;
            for (size_t expert_idx = 0; expert_idx < layer.NumExperts(); ++expert_idx) {
                ggml_tensor* gate = layer.GetExpert(expert_idx, model_keys::kFfnGate);
                ggml_tensor* up = layer.GetExpert(expert_idx, model_keys::kFfnUp);
                ggml_tensor* down = layer.GetExpert(expert_idx, model_keys::kFfnDown);
                bind_expert_view_int4(gate);
                bind_expert_view_int4(up);
                bind_expert_view_int4(down);

                const bool has_gate_int4 = model->int4_weight_bindings.find(gate) != model->int4_weight_bindings.end();
                const bool has_up_int4 = model->int4_weight_bindings.find(up) != model->int4_weight_bindings.end();
                const bool has_down_int4 = model->int4_weight_bindings.find(down) != model->int4_weight_bindings.end();
                gate_int4_bindings += has_gate_int4 ? 1u : 0u;
                up_int4_bindings += has_up_int4 ? 1u : 0u;
                down_int4_bindings += has_down_int4 ? 1u : 0u;
                fused_gate_up_available += (has_gate_int4 && has_up_int4) ? 1u : 0u;
            }

            if (IsMoELoaderDebugEnabled() || IsQwen36MoEBindingDebugEnabled()) {
                std::fprintf(stderr,
                             "[QWEN36_MOE_BINDINGS] layer=%zu expert_count=%zu gate_binding_count=%zu "
                             "up_binding_count=%zu down_binding_count=%zu fused_gate_up_availability=%zu\n",
                             layer_idx, layer.NumExperts(), gate_int4_bindings, up_int4_bindings, down_int4_bindings,
                             fused_gate_up_available);
                if (IsQwen36MoEBindingDebugEnabled() && layer.NumExperts() > 0) {
                    const auto dump_tensor = [](const char* tag, const ggml_tensor* tensor) {
                        if (!tensor) {
                            std::fprintf(stderr, "[QWEN36_MOE_BINDINGS] %s=<null>\n", tag);
                            return;
                        }
                        std::fprintf(stderr,
                                     "[QWEN36_MOE_BINDINGS] %s name=%s type=%s ne=[%lld,%lld,%lld,%lld] "
                                     "view_offs=%zu view_src=%s\n",
                                     tag, tensor->name[0] ? tensor->name : "<unnamed>", ggml_type_name(tensor->type),
                                     static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
                                     static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]),
                                     tensor->view_offs,
                                     tensor->view_src
                                         ? (tensor->view_src->name[0] ? tensor->view_src->name : "<unnamed>")
                                         : "<null>");
                    };
                    const auto dump_binding = [model](const char* tag, const ggml_tensor* tensor) {
                        if (!tensor) {
                            return;
                        }
                        const auto it = model->int4_weight_bindings.find(tensor);
                        if (it != model->int4_weight_bindings.end()) {
                            std::fprintf(
                                stderr,
                                "[QWEN36_MOE_BINDINGS] %s binding group_size=%d K=%lld N=%lld "
                                "packed=%s scales=%s zeros=%s\n",
                                tag, it->second.group_size, static_cast<long long>(it->second.k),
                                static_cast<long long>(it->second.n),
                                it->second.packed && it->second.packed->name[0] ? it->second.packed->name : "<unnamed>",
                                it->second.scales && it->second.scales->name[0] ? it->second.scales->name : "<unnamed>",
                                it->second.zeros && it->second.zeros->name[0] ? it->second.zeros->name : "<unnamed>");
                            return;
                        }
                        if (tensor->view_src) {
                            const auto root_it = model->int4_weight_bindings.find(tensor->view_src);
                            if (root_it != model->int4_weight_bindings.end()) {
                                std::fprintf(stderr,
                                             "[QWEN36_MOE_BINDINGS] %s root_binding group_size=%d K=%lld N=%lld "
                                             "root=%s scales=%s zeros=%s\n",
                                             tag, root_it->second.group_size, static_cast<long long>(root_it->second.k),
                                             static_cast<long long>(root_it->second.n),
                                             root_it->second.packed && root_it->second.packed->name[0]
                                                 ? root_it->second.packed->name
                                                 : "<unnamed>",
                                             root_it->second.scales && root_it->second.scales->name[0]
                                                 ? root_it->second.scales->name
                                                 : "<unnamed>",
                                             root_it->second.zeros && root_it->second.zeros->name[0]
                                                 ? root_it->second.zeros->name
                                                 : "<unnamed>");
                            }
                        }
                    };
                    const auto dump_expert = [&](size_t expert_idx, const char* prefix) {
                        auto* gate = layer.GetExpert(expert_idx, model_keys::kFfnGate);
                        auto* up = layer.GetExpert(expert_idx, model_keys::kFfnUp);
                        auto* down = layer.GetExpert(expert_idx, model_keys::kFfnDown);
                        std::string gate_tag = std::string(prefix) + "_gate";
                        std::string up_tag = std::string(prefix) + "_up";
                        std::string down_tag = std::string(prefix) + "_down";
                        dump_tensor(gate_tag.c_str(), gate);
                        dump_tensor(up_tag.c_str(), up);
                        dump_tensor(down_tag.c_str(), down);
                        dump_binding(gate_tag.c_str(), gate);
                        dump_binding(up_tag.c_str(), up);
                        dump_binding(down_tag.c_str(), down);
                    };
                    dump_tensor("shared_gate_tensor", layer.Get(model_keys::kFfnGate));
                    dump_tensor("shared_up_tensor", layer.Get(model_keys::kFfnUp));
                    dump_tensor("shared_down_tensor", layer.Get(model_keys::kFfnDown));
                    dump_expert(0, "expert0");
                    if (layer.NumExperts() > 1) {
                        dump_expert(1, "expert1");
                    }
                    if (layer.NumExperts() > 2) {
                        dump_expert(layer.NumExperts() - 1, "expert_last");
                    }
                }
            }
        }
    }

    PrepareGenericCpuFastMatmulAliases(model);
    PrepareGemma4CpuRepackAliases(model);

    // ==========================================================================
    // NUMA Optimization: Interleaved Allocation for Multi-Socket Systems
    // ==========================================================================
    // For multi-socket servers, spread large weight tensors across all NUMA nodes
    // using numa_alloc_interleaved. This maximizes aggregate memory bandwidth
    // when multiple threads across different sockets access the weights.
    //
    // For single-socket/non-NUMA systems, use madvise(MADV_WILLNEED) to pre-fault
    // pages into memory, reducing page fault latency during inference.
    // ==========================================================================

#if defined(__linux__)
    // Get total weight memory size for reporting
    size_t total_weight_bytes = 0;
    struct ggml_tensor* t = ggml_get_first_tensor(model->ctx_w);
    while (t) {
        total_weight_bytes += ggml_nbytes(t);
        t = ggml_get_next_tensor(model->ctx_w, t);
    }

    const bool prepopulate_weights = [total_weight_bytes]() {
        const char* env = std::getenv("DENSECORE_PREPOPULATE_WEIGHTS");
        if (env && env[0] != '\0') {
            std::string mode(env);
            std::transform(mode.begin(), mode.end(), mode.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (mode == "0" || mode == "false" || mode == "off" || mode == "no") {
                return false;
            }
            if (mode == "1" || mode == "true" || mode == "on" || mode == "yes" || mode == "force") {
                return true;
            }
        }

        const char* threshold_env = std::getenv("DENSECORE_PREPOPULATE_WEIGHTS_THRESHOLD_MB");
        size_t threshold_mb = 8192;
        if (threshold_env && threshold_env[0] != '\0') {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(threshold_env, &end, 10);
            if (end != threshold_env && *end == '\0' && parsed > 0) {
                threshold_mb = static_cast<size_t>(parsed);
            }
        }

        const size_t threshold_bytes = threshold_mb * 1024ULL * 1024ULL;
        return total_weight_bytes <= threshold_bytes;
    }();

    if (densecore::NumaAllocator::IsNumaAvailable()) {
        std::cout << "[DenseCore] NUMA detected - weights will use interleaved "
                  << "allocation on next load via LoadGGUFModelNuma()" << std::endl;
        std::cout << "[DenseCore] For optimal multi-socket performance, use "
                  << "LoadGGUFModelNuma(path, -1, false) for interleaved layout" << std::endl;
    } else if (prepopulate_weights) {
        // Non-NUMA: Pre-populate pages with MADV_WILLNEED
        t = ggml_get_first_tensor(model->ctx_w);
        size_t prefetched_bytes = 0;
        while (t) {
            if (t->data && ggml_nbytes(t) >= 1024 * 1024) {  // Only for tensors >= 1MB
                madvise(t->data, ggml_nbytes(t), MADV_WILLNEED);
                prefetched_bytes += ggml_nbytes(t);
            }
            t = ggml_get_next_tensor(model->ctx_w, t);
        }
        if (prefetched_bytes > 0) {
            std::cout << "[DenseCore] Pre-populated " << (prefetched_bytes / 1024 / 1024)
                      << " MB of weight pages (MADV_WILLNEED)" << std::endl;
        }
    } else {
        std::cout << "[DenseCore] Skipping weight pre-population";
        if (const char* env = std::getenv("DENSECORE_PREPOPULATE_WEIGHTS"); env && env[0] != '\0') {
            std::cout << " (DENSECORE_PREPOPULATE_WEIGHTS=" << env << ")";
        } else {
            std::cout << " (auto-disabled for large model; override with DENSECORE_PREPOPULATE_WEIGHTS=1)";
        }
        std::cout << std::endl;
    }
#endif

    model->decoder_spec = densecore::models::MakeDecoderModelSpec(model);
    std::cout << "[DenseCore] Model loaded successfully" << std::endl;
    return model;
}

// ============================================================================
// SaveModel
// ============================================================================

int SaveModel(const TransformerModel* model, const char* path) {
    if (!model || !path) return -1;

    std::cout << "[DenseCore] Saving model to " << path << "..." << std::endl;
    std::cerr << "[DenseCore] WARNING: SaveModel currently only supports "
                 "metadata updates. Tensor data writing is experimental."
              << std::endl;

    struct gguf_context* ctx = gguf_init_empty();

    // 1. Write metadata
    gguf_set_val_str(ctx, "general.architecture", "llama");
    gguf_set_val_str(ctx, "general.name", "DenseCore-Optimized");

    // Write hyperparameters
    gguf_set_val_u32(ctx, "llama.vocab_size", model->hparams.n_vocab);
    gguf_set_val_u32(ctx, "llama.embedding_length", model->hparams.n_embd);
    gguf_set_val_u32(ctx, "llama.block_count", model->hparams.n_layer);
    gguf_set_val_u32(ctx, "llama.attention.head_count", model->hparams.n_head);
    gguf_set_val_u32(ctx, "llama.attention.head_count_kv", model->hparams.n_head_kv);
    gguf_set_val_u32(ctx, "llama.context_length", model->hparams.n_ctx);
    gguf_set_val_f32(ctx, "llama.attention.layer_norm_rms_epsilon", model->hparams.f_norm_rms_eps);
    gguf_set_val_f32(ctx, "llama.rope.freq_base", model->hparams.rope_freq_base);
    gguf_set_val_f32(ctx, "llama.rope.freq_scale", model->hparams.rope_freq_scale);

    // Write vocab if available
    if (!model->vocab_tokens.empty()) {
        std::vector<const char*> tokens_cstr;
        tokens_cstr.reserve(model->vocab_tokens.size());
        for (const auto& s : model->vocab_tokens) {
            tokens_cstr.push_back(s.c_str());
        }
        gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", tokens_cstr.data(), tokens_cstr.size());
    }

    // 2. Write Tensors (experimental)
    // Note: Full tensor serialization requires properly set up tensors in the
    // context. This is a simplified implementation that may not work for all
    // models.

    // Write file (metadata only is safer, but we try full write)
    bool ok = gguf_write_to_file(ctx, path, false);  // false = include tensors

    gguf_free(ctx);

    if (!ok) {
        std::cerr << "[DenseCore] Error: Failed to write GGUF file" << std::endl;
        return -2;
    }

    std::cout << "[DenseCore] Model saved (metadata-only mode)" << std::endl;
    return 0;
}

// ============================================================================
// NUMA-Aware Model Loading
// ============================================================================

/**
 * Rebind a single tensor's data using NUMA-interleaved allocation.
 *
 * Uses numa_alloc_interleaved to spread memory pages across all NUMA nodes,
 * maximizing aggregate memory bandwidth when multiple threads access the data.
 *
 * @param tensor Tensor to rebind (must have valid data pointer)
 * @param model Model to track allocation for cleanup
 * @param min_size_bytes Minimum tensor size threshold (skip smaller tensors)
 * @return True if tensor was successfully rebound
 */
static bool RebindTensorNumaInterleaved(struct ggml_tensor* tensor, TransformerModel* model,
                                        size_t min_size_bytes = 1024 * 1024) {
    if (!tensor || !tensor->data) {
        return false;
    }

    const size_t tensor_size = ggml_nbytes(tensor);
    if (tensor_size < min_size_bytes) {
        return false;  // Skip small tensors - overhead not worth it
    }

#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
    // Use numa_alloc_interleaved for true interleaved allocation
    void* new_buffer = numa_alloc_interleaved(tensor_size);
    if (!new_buffer) {
        std::cerr << "[NUMA] numa_alloc_interleaved failed for " << (tensor_size / 1024 / 1024) << " MB" << std::endl;
        return false;
    }

    // Copy data to interleaved buffer
    std::memcpy(new_buffer, tensor->data, tensor_size);

    // Replace tensor data pointer
    tensor->data = new_buffer;

    // Track allocation for cleanup (use Numa type for numa_free)
    model->numa_buffers.push_back({new_buffer, tensor_size, densecore::AllocationType::Numa});

    return true;
#else
    // Fallback: use aligned allocation with page pre-population
    void* new_buffer = nullptr;
    if (posix_memalign(&new_buffer, 64, tensor_size) != 0 || !new_buffer) {
        return false;
    }

    std::memcpy(new_buffer, tensor->data, tensor_size);

#if defined(__linux__)
    // Pre-populate pages to reduce page fault latency
    madvise(new_buffer, tensor_size, MADV_WILLNEED);
#endif

    tensor->data = new_buffer;
    model->numa_buffers.push_back({new_buffer, tensor_size, densecore::AllocationType::Aligned});
    return true;
#endif
}

/**
 * Rebind a single tensor's data to a specific NUMA node (round-robin mode).
 *
 * Uses dedicated thread pinned to target NUMA node to enforce first-touch
 * policy for physical memory allocation on that node.
 *
 * @param tensor Tensor to rebind (must have valid data pointer)
 * @param numa_node Target NUMA node for allocation
 * @param model Model to track allocation for cleanup
 * @param min_size_bytes Minimum tensor size threshold (skip smaller tensors)
 * @return True if tensor was successfully rebound
 */
static bool RebindTensorNuma(struct ggml_tensor* tensor, int numa_node, TransformerModel* model,
                             size_t min_size_bytes = 1024 * 1024) {
    if (!tensor || !tensor->data) {
        return false;
    }

    const size_t tensor_size = ggml_nbytes(tensor);
    if (tensor_size < min_size_bytes) {
        return false;  // Skip small tensors - overhead not worth it
    }

    // Allocate NUMA-aware buffer
    auto result = densecore::NumaAllocator::AllocatePreferred(tensor_size, 64, numa_node);

    if (!result.ptr) {
        std::cerr << "[NUMA] Failed to allocate " << (tensor_size / 1024 / 1024) << " MB on node " << numa_node
                  << std::endl;
        return false;
    }

    // =========================================================================
    // CRITICAL: Copy data using thread pinned to target NUMA node
    // =========================================================================
    // This enforces first-touch policy: the physical memory pages are allocated
    // on the NUMA node where the first write occurs. By pinning our copy thread
    // to the target node, we ensure pages are allocated there.
    // =========================================================================
    const void* src_data = tensor->data;
    void* dst_data = result.ptr;

    std::thread copy_thread([dst_data, src_data, tensor_size, numa_node]() {
        // Pin this thread to the target NUMA node
        densecore::HardwareTopology::GetInstance().PinCurrentThreadToNumaNode(numa_node,
                                                                              densecore::PinningPolicy::SCATTER);

        // Perform the copy - this triggers first-touch allocation
        std::memcpy(dst_data, src_data, tensor_size);
    });

    // MUST join to ensure copy completes before we use the tensor
    copy_thread.join();

    // Replace tensor data pointer
    // NOTE: Original mmap data will be freed when ctx_gguf is freed
    tensor->data = result.ptr;

    // Track allocation for cleanup in model destructor
    model->numa_buffers.push_back({result.ptr, result.size, result.allocation_type});

    return true;
}

/**
 * Get the number of available NUMA nodes for round-robin allocation.
 */
static int GetNumaNodeCount() {
#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
    if (numa_available() >= 0) {
        return numa_max_node() + 1;
    }
#endif
    return 1;  // Single node fallback
}

/**
 * Load GGUF model with NUMA-interleaved memory placement.
 *
 * After initial mmap load, identifies large tensors (>1MB) and rebinds
 * their data for optimized memory bandwidth using one of three strategies:
 *
 * 1. INTERLEAVED (numa_node == -1): Uses numa_alloc_interleaved to spread
 *    pages across ALL NUMA nodes for maximum aggregate bandwidth.
 *
 * 2. ROUND-ROBIN (numa_node == -2): Alternates tensors between nodes
 *    (Tensor 0 -> Node 0, Tensor 1 -> Node 1, etc.)
 *
 * 3. PINNED (numa_node >= 0): Pins all tensors to a specific node.
 *
 * @param path Path to GGUF file
 * @param numa_node Allocation strategy:
 *                  -1 = Interleaved (recommended for multi-socket)
 *                  -2 = Round-robin across nodes
 *                  >= 0 = Pin to specific node
 * @param use_huge_pages Whether to request huge pages (TBD)
 * @return Loaded model with NUMA-optimized memory layout
 */
TransformerModel* LoadGGUFModelNuma(const char* path, int numa_node, bool use_huge_pages) {
    // Step 1: Load model normally (mmap-based)
    TransformerModel* model = LoadGGUFModel(path);
    if (!model) {
        return nullptr;
    }

    // Step 2: Check if NUMA rebinding is available
    bool numa_available = densecore::NumaAllocator::IsNumaAvailable();
    int num_nodes = GetNumaNodeCount();

    if (!numa_available && numa_node < 0) {
        std::cout << "[DenseCore] NUMA not available, using standard memory layout" << std::endl;
        return model;
    }

    // =========================================================================
    // HUGE PAGES SUPPORT FOR LARGE MODEL WEIGHTS
    // =========================================================================
    // Huge pages (2MB or 1GB) reduce TLB misses for large memory allocations.
    // This significantly improves performance for models > 1GB in size.
    //
    // Requirements:
    // - Linux: vm.nr_hugepages must be configured (echo 512 > /proc/sys/vm/nr_hugepages)
    // - Fallback: Regular 4KB pages if huge pages unavailable
    // =========================================================================
#if defined(__linux__)
    if (use_huge_pages) {
        // Attempt to reallocate large tensors using huge pages
        struct ggml_tensor* t = ggml_get_first_tensor(model->ctx_w);
        size_t huge_page_threshold = 2ULL * 1024 * 1024;  // 2MB threshold
        size_t hugepage_bytes = 0;
        int hugepage_count = 0;

        while (t) {
            const size_t tensor_size = ggml_nbytes(t);
            if (tensor_size >= huge_page_threshold && t->data) {
                // Align size to 2MB boundary for huge pages
                size_t aligned_size = (tensor_size + (2ULL * 1024 * 1024 - 1)) & ~(2ULL * 1024 * 1024 - 1);

                // Try MAP_HUGETLB allocation
                void* huge_ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

                if (huge_ptr != MAP_FAILED) {
                    // Copy data to huge page allocation
                    std::memcpy(huge_ptr, t->data, tensor_size);

                    // Track for cleanup
                    model->numa_buffers.push_back({huge_ptr, aligned_size, densecore::AllocationType::Mmap});

                    // Update tensor pointer
                    t->data = huge_ptr;

                    hugepage_bytes += aligned_size;
                    hugepage_count++;
                }
                // If MAP_HUGETLB fails, silently continue with original mmap
            }
            t = ggml_get_next_tensor(model->ctx_w, t);
        }

        if (hugepage_count > 0) {
            std::cout << "[DenseCore] Huge pages allocated: " << hugepage_count << " tensors ("
                      << (hugepage_bytes / 1024 / 1024) << " MB)" << std::endl;
        } else {
            std::cout << "[DenseCore] Huge pages requested but not available " << "(check vm.nr_hugepages)"
                      << std::endl;
        }
    }
#else
    (void)use_huge_pages;  // Huge pages only supported on Linux
#endif


    // Determine allocation mode
    enum class NumaMode { INTERLEAVED, ROUND_ROBIN, PINNED };
    NumaMode mode;
    const char* mode_name;

    if (numa_node == -1) {
        mode = NumaMode::INTERLEAVED;
        mode_name = "INTERLEAVED";
        std::cout << "[DenseCore] Using NUMA INTERLEAVED allocation " << "(spreading weights across " << num_nodes
                  << " nodes)" << std::endl;
    } else if (numa_node == -2) {
        mode = NumaMode::ROUND_ROBIN;
        mode_name = "ROUND-ROBIN";
        std::cout << "[DenseCore] Using NUMA ROUND-ROBIN allocation across " << num_nodes << " nodes" << std::endl;
    } else {
        mode = NumaMode::PINNED;
        mode_name = "PINNED";
        std::cout << "[DenseCore] PINNING all weights to NUMA node " << numa_node << std::endl;
    }

    // Step 3: Collect all large tensors to rebind
    struct TensorInfo {
        struct ggml_tensor* tensor;
        std::string name;
    };
    std::vector<TensorInfo> tensors_to_rebind;

    auto collect = [&](struct ggml_tensor* t, const char* name) {
        if (t && t->data && ggml_nbytes(t) >= 1024 * 1024) {
            tensors_to_rebind.push_back({t, name});
        }
    };

    // Collect critical weight tensors
    collect(model->tok_embeddings, "tok_embeddings");
    collect(model->output, "output");
    collect(model->output_norm, "output_norm");
    collect(model->gemma4_per_layer_model_projection, "gemma4.per_layer_model_projection");
    collect(model->gemma4_per_layer_projection_norm, "gemma4.per_layer_projection_norm");
    collect(model->gemma4_per_layer_token_embeddings, "gemma4.per_layer_token_embeddings");

    // Collect layer weights
    for (size_t i = 0; i < model->layers.size(); ++i) {
        auto& layer = model->layers[i];
        std::string prefix = "blk." + std::to_string(i) + ".";

        collect(layer.Get(model_keys::kAttnQWeight), (prefix + "wq").c_str());
        collect(layer.Get(model_keys::kAttnKWeight), (prefix + "wk").c_str());
        collect(layer.Get(model_keys::kAttnVWeight), (prefix + "wv").c_str());
        collect(layer.Get(model_keys::kAttnOWeight), (prefix + "wo").c_str());
        collect(layer.Get(model_keys::kFfnGate), (prefix + "w1").c_str());
        collect(layer.Get(model_keys::kFfnDown), (prefix + "w2").c_str());
        collect(layer.Get(model_keys::kFfnUp), (prefix + "w3").c_str());
        collect(layer.Get(model_keys::kAttnNorm), (prefix + "attn_norm").c_str());
        collect(layer.Get(model_keys::kFfnNorm), (prefix + "ffn_norm").c_str());
        collect(layer.Get(model_keys::kGemma4PerLayerInputGate), (prefix + "inp_gate").c_str());
        collect(layer.Get(model_keys::kGemma4PerLayerProjection), (prefix + "proj").c_str());
        collect(layer.Get(model_keys::kGemma4PostPerLayerInputNorm), (prefix + "post_norm").c_str());
    }

    // Step 4: Rebind tensors based on mode
    size_t rebound_bytes = 0;
    int rebound_count = 0;
    std::vector<int> node_distribution(num_nodes, 0);

    for (size_t i = 0; i < tensors_to_rebind.size(); ++i) {
        auto& info = tensors_to_rebind[i];
        bool success = false;
        int target_node = -1;

        switch (mode) {
        case NumaMode::INTERLEAVED: success = RebindTensorNumaInterleaved(info.tensor, model); break;

        case NumaMode::ROUND_ROBIN:
            target_node = static_cast<int>(i % num_nodes);
            success = RebindTensorNuma(info.tensor, target_node, model);
            if (success) {
                node_distribution[target_node]++;
            }
            break;

        case NumaMode::PINNED:
            target_node = numa_node;
            success = RebindTensorNuma(info.tensor, target_node, model);
            break;
        }

        if (success) {
            rebound_bytes += ggml_nbytes(info.tensor);
            rebound_count++;
        }
    }

    // Step 5: Print summary
    std::cout << "[DenseCore] NUMA allocation complete (" << mode_name << "): " << rebound_count << " tensors ("
              << (rebound_bytes / 1024 / 1024) << " MB)";

    if (mode == NumaMode::ROUND_ROBIN) {
        std::cout << " [Distribution:";
        for (int n = 0; n < num_nodes; ++n) {
            std::cout << " N" << n << "=" << node_distribution[n];
        }
        std::cout << "]";
    } else if (mode == NumaMode::PINNED) {
        std::cout << " to Node " << numa_node;
    }
    std::cout << std::endl;

    // Optional: Verify placement for interleaved mode
    if (mode == NumaMode::INTERLEAVED && model->tok_embeddings && rebound_count > 0) {
        densecore::MemoryDiagnostics::PrintSystemTopologyReport(
            model->tok_embeddings->data, ggml_nbytes(model->tok_embeddings), -1, "tok_embeddings (interleaved sample)");
    }

    return model;
}

// ============================================================================
// SmartLoader Implementation
// ============================================================================

TransformerModel* LoadModelFromExternal(const TransformerHParams& hparams, const std::vector<ExternalTensor>& tensors,
                                        ModelArch arch) {
    std::cout << "[DenseCore] SmartLoader: Loading model from " << tensors.size() << " external tensors..."
              << std::endl;

    auto model = new TransformerModel();
    model->hparams = hparams;
    model->arch = arch;

    // 1. Initialize Backend (CPU as default controller)
    model->backend = ggml_backend_cpu_init();
    if (!model->backend) {
        std::cerr << "[DenseCore] Error: Failed to initialize backend" << std::endl;
        delete model;
        return nullptr;
    }
    model->cpu_backend = model->backend;

    // 2. Initialize Weight Context (Zero-Copy)
    // We only allocate struct overhead, data remains in external memory
    size_t ctx_size = (tensors.size() + 128) * ggml_tensor_overhead();
    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    model->ctx_w = ggml_init(params);
    if (!model->ctx_w) {
        std::cerr << "[DenseCore] Error: Failed to allocate weight context (" << (ctx_size / 1024)
                  << " KB) - out of memory" << std::endl;
        ggml_backend_free(model->backend);
        delete model;
        return nullptr;
    }

    // 3. Register Tensors
    for (const auto& et : tensors) {
        struct ggml_tensor* t = nullptr;
        int ndim = et.shape.size();

        // GGML uses column-major internal logical layout (ne0 is fastest dim)
        // If data is row-major (standard C/Python), we map:
        // Python [A, B] -> GGML ne=[B, A]
        if (ndim == 1) {
            t = ggml_new_tensor_1d(model->ctx_w, et.type, et.shape[0]);
        } else if (ndim == 2) {
            t = ggml_new_tensor_2d(model->ctx_w, et.type, et.shape[1], et.shape[0]);
        } else if (ndim == 3) {
            t = ggml_new_tensor_3d(model->ctx_w, et.type, et.shape[2], et.shape[1], et.shape[0]);
        } else if (ndim == 4) {
            t = ggml_new_tensor_4d(model->ctx_w, et.type, et.shape[3], et.shape[2], et.shape[1], et.shape[0]);
        } else {
            std::cerr << "[DenseCore] Warning: Unsupported tensor rank " << ndim << " for " << et.name << std::endl;
            continue;
        }

        if (t) {
            t->data = et.data;  // ZERO-COPY MAGIC
            ggml_set_name(t, et.name.c_str());
        }
    }

    // 4. Map Standard Tensors (Populate layer structs)
    model->layers.resize(model->hparams.n_layer);

    auto get_tensor = [&](const std::string& name) -> struct ggml_tensor* {
        return ggml_get_tensor(model->ctx_w, name.c_str());
    };

    model->tok_embeddings = get_tensor("token_embd.weight");
    model->output_norm = get_tensor("output_norm.weight");
    model->output = get_tensor("output.weight");
    if (model->arch_flags.is_gemma4) {
        model->gemma4_per_layer_model_projection = get_tensor("per_layer_model_proj.weight");
        model->gemma4_per_layer_projection_norm = get_tensor("per_layer_proj_norm.weight");
        model->gemma4_per_layer_token_embeddings = get_tensor("per_layer_token_embd.weight");
        const bool has_any_per_layer_input_tensor = model->gemma4_per_layer_model_projection ||
                                                    model->gemma4_per_layer_projection_norm ||
                                                    model->gemma4_per_layer_token_embeddings;
        if (has_any_per_layer_input_tensor) {
            if (!model->gemma4_per_layer_model_projection || !model->gemma4_per_layer_projection_norm ||
                !model->gemma4_per_layer_token_embeddings) {
                std::cerr << "[DenseCore] FATAL: Gemma4 external tensor load has incomplete per-layer-input export"
                          << std::endl;
                delete model;
                return nullptr;
            }
            if (model->gemma4_hidden_size_per_layer_input <= 0) {
                std::cerr << "[DenseCore] FATAL: Gemma4 external tensor load requires explicit per-layer-input metadata"
                          << std::endl;
                delete model;
                return nullptr;
            }
        }
    }
    if (!model->output) model->output = get_tensor("lm_head.weight");

    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        std::string layer_prefix = "blk." + std::to_string(i) + ".";

        // Attempt to load standard implementation tensors (will be null if not found)
        model->layers[i].Set(model_keys::kAttnNorm, get_tensor(layer_prefix + "attn_norm.weight"));
        model->layers[i].Set(model_keys::kFfnNorm, get_tensor(layer_prefix + "ffn_norm.weight"));

        // Check for Fused QKV vs Separate
        model->layers[i].Set(model_keys::kAttnQWeight, get_tensor(layer_prefix + "attn_q.weight"));
        model->layers[i].Set(model_keys::kAttnKWeight, get_tensor(layer_prefix + "attn_k.weight"));
        model->layers[i].Set(model_keys::kAttnVWeight, get_tensor(layer_prefix + "attn_v.weight"));
        model->layers[i].Set(model_keys::kAttnOWeight, get_tensor(layer_prefix + "attn_output.weight"));

        model->layers[i].Set(model_keys::kFfnGate, get_tensor(layer_prefix + "ffn_gate.weight"));
        model->layers[i].Set(model_keys::kFfnDown, get_tensor(layer_prefix + "ffn_down.weight"));
        model->layers[i].Set(model_keys::kFfnUp, get_tensor(layer_prefix + "ffn_up.weight"));
    }

    model->decoder_spec = densecore::models::MakeDecoderModelSpec(model);
    std::cout << "[DenseCore] SmartLoader: Model loaded successfully (" << ctx_size / 1024 << " KB header)."
              << std::endl;
    return model;
}
bool IsMoELoaderDebugEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MOE_LOADER");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsQwen36MoEBindingDebugEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_QWEN36_MOE_BINDINGS");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}
