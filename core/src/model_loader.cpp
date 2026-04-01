#include "model_loader.h"

#include <ggml-cpu.h>
#ifdef __APPLE__
#include <ggml-metal.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <regex>
#include <string>
#include <thread>
#include <unordered_set>

#include "densecore/models/model_graph_bridge.h"  // Universal graph execution bridge
#include "hardware_topology.h"
#include "inference.h"  // For InitRoPETable
#include "matmul_backend.h"
#include "numa_allocator.h"
#include "tokenizer.h"

#if defined(__linux__)
#include <sys/mman.h>  // For mmap, MAP_HUGETLB
#endif

#include "apple_silicon.h"
#include "dtype_utils.h"
#include "qwen35_ssm_math.h"

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

    // Allocate a small separate context for view tensor metadata (expert slices, etc.).
    // gguf_init_from_file leaves no room for additional ggml_tensor structs in ctx_w,
    // so MoE view tensors (n_experts × n_layers × 3) must live in a dedicated pool.
    // 256 experts × 64 layers × 3 views × 512 bytes ≈ 24MB; allocate 32MB to be safe.
    {
        struct ggml_init_params vp = {
            /*.mem_size   =*/ 32LL * 1024LL * 1024LL,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
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

    std::string arch_lower = arch;
    std::transform(arch_lower.begin(), arch_lower.end(), arch_lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Set architecture enum and flags from detected string
    if (arch_lower == "llama") {
        model->arch = ModelArch::LLAMA;
    } else if (arch_lower == "qwen2" || arch_lower == "qwen2.5") {
        model->arch = ModelArch::QWEN2;
    } else if (arch_lower == "qwen35" || arch_lower == "qwen3.5" || arch_lower == "qwen35moe" ||
               arch_lower == "qwen35_moe" || arch_lower == "qwen3.5_moe" || arch_lower == "qwen3_5_moe" ||
               arch_lower == "qwen3_5_moe_text") {
        model->arch = ModelArch::QWEN35;
        model->arch_flags.is_hybrid_ssm = true;
        model->arch_flags.requires_q_norm = true;
        model->arch_flags.requires_k_norm = true;
    } else if (arch_lower == "qwen3" || arch_lower == "qwen3_moe" || arch_lower == "qwen3moe") {
        model->arch = ModelArch::QWEN3;
        model->arch_flags.requires_q_norm = true;
        model->arch_flags.requires_k_norm = true;
    } else if (arch_lower == "glm4_moe" || arch_lower == "glm4moe" || arch_lower == "glm4.5" ||
               arch_lower == "glm-4.5" || arch_lower == "glm4") {
        model->arch = ModelArch::GLM4_MOE;
        model->arch_flags.is_glm_moe = true;
    } else if (arch_lower == "glm_moe_dsa" || arch_lower == "glm5_dsa" || arch_lower == "glm5" ||
               arch_lower == "glm-5") {
        model->arch = ModelArch::GLM5_DSA;
        model->arch_flags.is_glm_moe = true;
        model->arch_flags.is_glm_dsa = true;
    } else if (arch_lower == "mistral") {
        model->arch = ModelArch::MISTRAL;
    } else if (arch_lower == "gemma" || arch_lower == "gemma2") {
        model->arch = ModelArch::GEMMA;
    } else if (arch_lower == "phi" || arch_lower == "phi3") {
        model->arch = ModelArch::PHI;
        // Vision Architectures
    } else if (arch_lower == "vit" || arch_lower == "vision_transformer") {
        model->arch = ModelArch::VIT;
    } else if (arch_lower == "clip" || arch_lower == "clip_vision") {
        model->arch = ModelArch::CLIP_VISION;
    } else if (arch_lower == "siglip") {
        model->arch = ModelArch::SIGLIP;
        // Audio Architectures
    } else if (arch_lower == "whisper") {
        model->arch = ModelArch::WHISPER;
        // Multimodal Architectures
    } else if (arch_lower == "llava") {
        model->arch = ModelArch::LLAVA;
    } else if (arch_lower == "qwen_vl" || arch_lower == "qwen2_vl") {
        model->arch = ModelArch::QWEN_VL;
    } else {
        model->arch = ModelArch::UNKNOWN;
        std::cerr << "[DenseCore] Warning: Unknown architecture '" << arch << "'. Model may not load correctly."
                  << std::endl;
    }

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
    std::string tokenizer_type = !tokenizer_pre.empty() ? tokenizer_pre : tokenizer_model;
    std::string tokenizer_lower = tokenizer_type;
    std::transform(tokenizer_lower.begin(), tokenizer_lower.end(), tokenizer_lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    model->tokenizer_type = tokenizer_type;
    int idx_chat_template = gguf_find_key(ctx_gguf, "tokenizer.chat_template");
    if (idx_chat_template != -1) {
        model->chat_template = gguf_get_val_str(ctx_gguf, idx_chat_template);
    }

    if (!tokenizer_type.empty()) {
        const std::vector<std::string> supported = {"llama",   "gpt2",  "qwen2", "qwen3", "qwen35",
                                                    "mistral", "gemma", "bpe",   "glm4",  "glm"};
        if (std::find(supported.begin(), supported.end(), tokenizer_lower) == supported.end()) {
            std::cerr << "[DenseCore] Warning: tokenizer model '" << tokenizer_type
                      << "' may not be fully compatible. Consider using external tokenization and input_ids."
                      << std::endl;
        }
    }

    // 2. Generic parameter loader using architecture prefix
    auto get_u32 = [&](const std::string& suffix, uint32_t& val) {
        // Try architecture-specific key first, then fallback to general
        std::string key = arch + "." + suffix;
        int idx = gguf_find_key(ctx_gguf, key.c_str());
        if (idx == -1) {
            // Fallback to "general." prefix
            key = "general." + suffix;
            idx = gguf_find_key(ctx_gguf, key.c_str());
        }
        if (idx != -1) {
            val = gguf_get_val_u32(ctx_gguf, idx);
        }
    };

    auto get_f32 = [&](const std::string& suffix, float& val) {
        std::string key = arch + "." + suffix;
        int idx = gguf_find_key(ctx_gguf, key.c_str());
        if (idx == -1) {
            key = "general." + suffix;
            idx = gguf_find_key(ctx_gguf, key.c_str());
        }
        if (idx != -1) {
            val = gguf_get_val_f32(ctx_gguf, idx);
        }
    };

    auto get_i32_arr4 = [&](const std::string& suffix, std::array<int32_t, 4>& vals) {
        std::string key = arch + "." + suffix;
        int idx = gguf_find_key(ctx_gguf, key.c_str());
        if (idx == -1) {
            key = "general." + suffix;
            idx = gguf_find_key(ctx_gguf, key.c_str());
        }
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
        std::string key = arch + "." + suffix;
        int idx = gguf_find_key(ctx_gguf, key.c_str());
        if (idx == -1) {
            key = "general." + suffix;
            idx = gguf_find_key(ctx_gguf, key.c_str());
        }
        if (idx != -1) {
            val = gguf_get_val_bool(ctx_gguf, idx);
        }
    };

    auto get_str_array = [&](const std::string& suffix, std::vector<std::string>& vals) {
        std::string key = arch + "." + suffix;
        int idx = gguf_find_key(ctx_gguf, key.c_str());
        if (idx == -1) {
            key = "general." + suffix;
            idx = gguf_find_key(ctx_gguf, key.c_str());
        }
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

    auto has_key = [&](const std::string& suffix) -> bool {
        std::string key = arch + "." + suffix;
        if (gguf_find_key(ctx_gguf, key.c_str()) != -1) {
            return true;
        }
        key = "general." + suffix;
        return gguf_find_key(ctx_gguf, key.c_str()) != -1;
    };

    // Load hyperparameters using dynamic architecture prefix
    get_u32("vocab_size", model->hparams.n_vocab);
    get_u32("embedding_length", model->hparams.n_embd);
    get_u32("block_count", model->hparams.n_layer);
    get_u32("attention.head_count", model->hparams.n_head);
    get_u32("attention.head_count_kv", model->hparams.n_head_kv);
    get_u32("context_length", model->hparams.n_ctx);

    if (model->hparams.n_head_kv == 0) model->hparams.n_head_kv = model->hparams.n_head;

    // llama.cpp style: Load head dimensions from GGUF
    get_u32("attention.key_length", model->hparams.n_embd_head_k);
    get_u32("attention.value_length", model->hparams.n_embd_head_v);

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

    // Load RMS epsilon
    model->hparams.f_norm_rms_eps = 1e-5f;
    get_f32("attention.layer_norm_rms_epsilon", model->hparams.f_norm_rms_eps);

    std::cout << "[DenseCore] Loaded RMS norm epsilon: " << model->hparams.f_norm_rms_eps << std::endl;

    // Load RoPE parameters
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

    // Load MoE parameters when present.
    if (model->arch_flags.is_glm_moe || has_key("n_routed_experts") || has_key("num_experts_per_tok") ||
        has_key("expert_count") || has_key("expert_used_count")) {
        uint32_t tmp = 0;

        tmp = model->hparams.n_experts;
        get_u32("n_routed_experts", tmp);
        if (tmp == 0) get_u32("num_local_experts", tmp);
        if (tmp == 0) get_u32("expert_count", tmp);  // qwen35moe naming
        model->hparams.n_experts = tmp;

        tmp = model->hparams.n_experts_used;
        get_u32("num_experts_per_tok", tmp);
        if (tmp == 0) get_u32("n_experts_used", tmp);
        if (tmp == 0) get_u32("expert_used_count", tmp);  // qwen35moe naming
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
        tmp = model->ssm_full_attn_interval;
        get_u32("full_attention_interval", tmp);
        model->ssm_full_attn_interval = static_cast<int>(tmp);

        const auto fail_hybrid_ssm = [&](const std::string& reason) {
            std::cerr << "[DenseCore] FATAL: invalid hybrid SSM configuration: " << reason << std::endl;
            if (model->backend) ggml_backend_free(model->backend);
            gguf_free(ctx_gguf);
            if (ctx_w) ggml_free(ctx_w);
            delete model;
            return static_cast<TransformerModel*>(nullptr);
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
        if (!layer_types.empty()) {
            model->hybrid_layer_is_ssm.assign(model->hparams.n_layer, 0);
            const size_t n = std::min(layer_types.size(), static_cast<size_t>(model->hparams.n_layer));
            for (size_t i = 0; i < n; ++i) {
                std::string type = layer_types[i];
                std::transform(type.begin(), type.end(), type.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                // Qwen3.5 GGUFs tag full-attention layers as "attention" (not "full_attention").
                // Treat both as non-SSM; only mark as SSM for explicitly recurrent types.
                const bool is_full_attn = (type == "full_attention" || type == "attention" ||
                                           type == "transformer" || type == "self_attention");
                model->hybrid_layer_is_ssm[i] = is_full_attn ? 0 : 1;
            }
        }

        // Initialize SSM runtime states
        const int conv_channels = model->ssm_inner_size + 2 * model->ssm_group_count * model->ssm_state_size;
        const int head_dim = model->ssm_inner_size / model->ssm_time_step_rank;
        if (conv_channels <= 0 || head_dim <= 0) {
            return fail_hybrid_ssm("derived conv_channels/head_dim must be positive");
        }
        int n_ssm_layers = 0;
        for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
            if (model->IsHybridSSMLayer(static_cast<int>(i))) {
                n_ssm_layers++;
            }
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
        if (model->hparams.rope_sections[0] > 0 || model->hparams.rope_sections[1] > 0) {
            std::cout << "[DenseCore] RoPE sections: [" << model->hparams.rope_sections[0] << ", "
                      << model->hparams.rope_sections[1] << ", " << model->hparams.rope_sections[2] << ", "
                      << model->hparams.rope_sections[3] << "]" << std::endl;
            std::cout << "[DenseCore] MRoPE interleaved: "
                      << (model->hparams.rope_mrope_interleaved ? "true" : "false") << std::endl;
        }
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

    // Check if model actually wants BOS added
    int idx_add_bos = gguf_find_key(ctx_gguf, "tokenizer.ggml.add_bos_token");
    bool add_bos = (idx_bos != -1);
    if (idx_add_bos != -1) {
        add_bos = gguf_get_val_bool(ctx_gguf, idx_add_bos);
    }

    if (model->bos_token_id < 0) {
        add_bos = false;
    }

    // Disable BOS if BOS equals PAD token
    int idx_pad = gguf_find_key(ctx_gguf, "tokenizer.ggml.padding_token_id");
    if (idx_pad != -1) {
        uint32_t pad_id = gguf_get_val_u32(ctx_gguf, idx_pad);
        if (model->bos_token_id == (int)pad_id) {
            std::cout << "[DenseCore] BOS == PAD, disabling BOS (model likely "
                         "doesn't use BOS)"
                      << std::endl;
            add_bos = false;
        }
    }

    if (!add_bos) {
        model->bos_token_id = -1;
    }

    std::cout << "[DenseCore] Model params: n_vocab=" << model->hparams.n_vocab << ", n_embd=" << model->hparams.n_embd
              << ", n_layer=" << model->hparams.n_layer << ", n_head=" << model->hparams.n_head
              << ", n_head_kv=" << model->hparams.n_head_kv << ", n_rot=" << model->hparams.n_rot
              << ", n_ctx=" << model->hparams.n_ctx << std::endl;
    std::cout << "[DenseCore] BOS=" << model->bos_token_id << ", EOS=" << model->eos_token_id
              << ", rope_freq=" << model->hparams.rope_freq_base << ", rope_scale=" << model->hparams.rope_freq_scale
              << std::endl;

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
            std::cout << "[DenseCore] Vocab size mismatch! Metadata=" << model->hparams.n_vocab << " but loaded "
                      << n_tokens << " tokens. Updating to " << n_tokens << std::endl;
            model->hparams.n_vocab = n_tokens;
        }
    }

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
        if (model->token_types.size() == model->vocab_tokens.size()) {
            for (size_t i = 0; i < model->token_types.size(); ++i) {
                if (model->token_types[i] != 3) continue;  // control
                const std::string& tok = model->vocab_tokens[i];
                if (tok.find("im_end") != std::string::npos || tok.find("eot") != std::string::npos ||
                    tok.find("eom") != std::string::npos || tok.find("endoftext") != std::string::npos) {
                    add_stop_id(static_cast<int32_t>(i));
                }
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
    model->output_norm = get_tensor("output_norm.weight");

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

    // Debug: Print tensor shapes
    if (model->tok_embeddings) {
        std::cout << "[DenseCore] tok_embeddings shape: [" << model->tok_embeddings->ne[0] << ", "
                  << model->tok_embeddings->ne[1] << "]" << std::endl;
    }
    if (model->output) {
        std::cout << "[DenseCore] output shape: [" << model->output->ne[0] << ", " << model->output->ne[1] << "]"
                  << std::endl;

        // Validate n_vocab matches output tensor shape
        uint32_t tensor_vocab_size = model->output->ne[1];
        if (tensor_vocab_size != model->hparams.n_vocab && tensor_vocab_size > 0) {
            std::cout << "[DenseCore] ERROR: Vocab size inconsistency! " << "Loaded vocab=" << model->hparams.n_vocab
                      << " but output tensor expects " << tensor_vocab_size << " tokens." << std::endl;
            std::cout << "[DenseCore] This will cause garbage output. " << "Check GGUF file integrity." << std::endl;

            if (tensor_vocab_size > model->hparams.n_vocab) {
                std::cout << "[DenseCore] WARNING: Tensor vocab larger than loaded vocab. "
                          << "Some tokens may not decode properly." << std::endl;
            }
        }
    }

    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        std::string layer_prefix = "blk." + std::to_string(i) + ".";

        // Common to all layer types: norms and FFN
        model->layers[i].Set(model_keys::kAttnNorm,
                             get_layer_tensor_any(i, {"attn_norm.weight", "input_layernorm.weight",
                                                      "attention_norm.weight", "self_attn_layernorm.weight"}));
        model->layers[i].Set(model_keys::kFfnNorm,
                             get_layer_tensor_any(i, {"ffn_norm.weight", "post_attention_layernorm.weight",
                                                      "post_attention_norm.weight", "mlp_layernorm.weight"}));
        model->layers[i].Set(model_keys::kPostAttnNorm, get_layer_tensor_any(i, {"post_attention_norm.weight",
                                                                                 "post_attention_layernorm.weight"}));
        // Fallback: Qwen3.5 uses post_attention_norm instead of ffn_norm
        if (!model->layers[i].Get(model_keys::kFfnNorm) && model->layers[i].Get(model_keys::kPostAttnNorm)) {
            model->layers[i].Set(model_keys::kFfnNorm, model->layers[i].Get(model_keys::kPostAttnNorm));
        }
        model->layers[i].Set(model_keys::kFfnGate,
                             get_layer_tensor_any(i, {"ffn_gate.weight", "ffn_gate_shexp.weight",
                                                      "mlp.gate_proj.weight", "gate_proj.weight"}));
        model->layers[i].Set(model_keys::kFfnDown,
                             get_layer_tensor_any(i, {"ffn_down.weight", "ffn_down_shexp.weight",
                                                      "mlp.down_proj.weight", "down_proj.weight"}));
        model->layers[i].Set(model_keys::kFfnUp,
                             get_layer_tensor_any(i, {"ffn_up.weight", "ffn_up_shexp.weight",
                                                      "mlp.up_proj.weight", "up_proj.weight"}));

        // Determine if this is an SSM layer or full attention layer
        const bool is_ssm = model->IsHybridSSMLayer(static_cast<int>(i));

        if (is_ssm) {
            // SSM/Mamba layer: fused QKV + SSM weights + gate
            model->layers[i].Set(model_keys::kAttnQkvWeight,
                                 get_layer_tensor_any(i, {"attn_qkv.weight", "linear_attn.in_proj_qkv.weight"}));
            model->layers[i].Set(model_keys::kAttnGate,
                                 get_layer_tensor_any(i, {"attn_gate.weight", "linear_attn.in_proj_z.weight"}));
            model->layers[i].Set(model_keys::kSSMConv1d,
                                 get_layer_tensor_any(i, {"ssm_conv1d.weight", "linear_attn.conv1d.weight"}));
            model->layers[i].Set(model_keys::kSSMA,
                                 get_layer_tensor_any(i, {"ssm_a", "linear_attn.A_log"}));
            model->layers[i].Set(model_keys::kSSMAlpha,
                                 get_layer_tensor_any(i, {"ssm_alpha.weight", "linear_attn.in_proj_a.weight"}));
            model->layers[i].Set(model_keys::kSSMBeta,
                                 get_layer_tensor_any(i, {"ssm_beta.weight", "linear_attn.in_proj_b.weight"}));
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

    auto ascii_lower_copy = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    };

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

        if (!layer.Get(model_keys::kFfnGate)) {
            layer.Set(model_keys::kFfnGate, find_layer_tensor_with_tokens(layer, {"shared_experts", "gate_proj"}));
        }
        if (!layer.Get(model_keys::kFfnUp)) {
            layer.Set(model_keys::kFfnUp, find_layer_tensor_with_tokens(layer, {"shared_experts", "up_proj"}));
        }
        if (!layer.Get(model_keys::kFfnDown)) {
            layer.Set(model_keys::kFfnDown, find_layer_tensor_with_tokens(layer, {"shared_experts", "down_proj"}));
        }
        if (!layer.Get(model_keys::kMoeGate)) {
            auto* t = find_layer_tensor_with_tokens(layer, {"mlp", "gate.weight"}, {"shared"});
            if (!t) t = find_layer_tensor_with_tokens(layer, {"ffn_gate_inp"}, {"shexp"});
            layer.Set(model_keys::kMoeGate, t);
        }
        if (!layer.Get(model_keys::kMoeCorrectionBias)) {
            layer.Set(model_keys::kMoeCorrectionBias,
                      find_layer_tensor_with_tokens(layer, {"e_score_correction_bias"}));
        }

        const struct ggml_tensor* packed_gate_up =
            find_layer_tensor_with_tokens(layer, {"experts", "gate_up_proj"}, {"shared"});
        const struct ggml_tensor* packed_down =
            find_layer_tensor_with_tokens(layer, {"experts", "down_proj"}, {"shared"});

        if (packed_gate_up && packed_down &&
            i >= static_cast<uint32_t>(std::max(0, model->moe_first_k_dense_replace))) {
            int packed_experts = 0;
            if (packed_gate_up->ne[2] > 0) {
                packed_experts = static_cast<int>(packed_gate_up->ne[2]);
            } else if (packed_gate_up->ne[3] > 0) {
                packed_experts = static_cast<int>(packed_gate_up->ne[3]);
            }
            if (model->hparams.n_experts == 0 && packed_experts > 0) {
                model->hparams.n_experts = static_cast<uint32_t>(packed_experts);
            }
            const int expert_count = std::min<int>(packed_experts, static_cast<int>(model->hparams.n_experts));
            if (expert_count > 0 && layer.Get(model_keys::kMoeGate)) {
                layer.is_moe = true;
                struct ggml_context* vctx = model->ctx_views ? model->ctx_views : model->ctx_w;
                for (int expert_idx = 0; expert_idx < expert_count; ++expert_idx) {
                    const size_t gate_up_offset = static_cast<size_t>(expert_idx) * packed_gate_up->nb[2];
                    struct ggml_tensor* gate_up_slice = ggml_view_2d(
                        vctx, const_cast<struct ggml_tensor*>(packed_gate_up), packed_gate_up->ne[0],
                        packed_gate_up->ne[1], packed_gate_up->nb[1], gate_up_offset);

                    const int64_t gate_up_rows = gate_up_slice->ne[1];
                    if (gate_up_rows < 2 || (gate_up_rows % 2) != 0) {
                        continue;
                    }
                    const int64_t intermediate = gate_up_rows / 2;
                    struct ggml_tensor* gate_w = ggml_view_2d(vctx, gate_up_slice, gate_up_slice->ne[0],
                                                              intermediate, gate_up_slice->nb[1], 0);
                    struct ggml_tensor* up_w =
                        ggml_view_2d(vctx, gate_up_slice, gate_up_slice->ne[0], intermediate,
                                     gate_up_slice->nb[1], static_cast<size_t>(intermediate) * gate_up_slice->nb[1]);
                    const size_t down_offset = static_cast<size_t>(expert_idx) * packed_down->nb[2];
                    struct ggml_tensor* down_w =
                        ggml_view_2d(vctx, const_cast<struct ggml_tensor*>(packed_down), packed_down->ne[0],
                                     packed_down->ne[1], packed_down->nb[1], down_offset);

                    layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kFfnGate, gate_w);
                    layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kFfnUp, up_w);
                    layer.SetExpert(static_cast<size_t>(expert_idx), model_keys::kFfnDown, down_w);
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
                continue;
            }
            if (std::regex_search(name, match, expert_up_re) && match.size() >= 2) {
                const size_t expert_idx = static_cast<size_t>(std::stoul(match[1].str()));
                layer.SetExpert(expert_idx, model_keys::kFfnUp, tensor);
                continue;
            }
            if (std::regex_search(name, match, expert_down_re) && match.size() >= 2) {
                const size_t expert_idx = static_cast<size_t>(std::stoul(match[1].str()));
                layer.SetExpert(expert_idx, model_keys::kFfnDown, tensor);
            }
        }

        // Handle separate stacked expert format: ffn_gate_exps / ffn_up_exps / ffn_down_exps
        // (used by qwen35moe bartowski GGUFs — experts stacked along ne[2] axis)
        if (layer.NumExperts() == 0 && layer.Get(model_keys::kMoeGate)) {
            const struct ggml_tensor* sep_gate = find_layer_tensor_with_tokens(layer, {"ffn_gate_exps"});
            const struct ggml_tensor* sep_up   = find_layer_tensor_with_tokens(layer, {"ffn_up_exps"});
            const struct ggml_tensor* sep_down  = find_layer_tensor_with_tokens(layer, {"ffn_down_exps"});
            if (sep_gate && sep_up && sep_down) {
                int n_exp = (sep_gate->ne[2] > 1) ? static_cast<int>(sep_gate->ne[2])
                          : (sep_gate->ne[3] > 0) ? static_cast<int>(sep_gate->ne[3]) : 0;
                if (model->hparams.n_experts == 0 && n_exp > 0) {
                    model->hparams.n_experts = static_cast<uint32_t>(n_exp);
                }
                int expert_count = std::min<int>(n_exp, static_cast<int>(model->hparams.n_experts));
                if (expert_count > 0 &&
                    i >= static_cast<uint32_t>(std::max(0, model->moe_first_k_dense_replace))) {
                    layer.is_moe = true;
                    struct ggml_context* vctx2 = model->ctx_views ? model->ctx_views : model->ctx_w;
                    for (int ei = 0; ei < expert_count; ++ei) {
                        const size_t g_off = static_cast<size_t>(ei) * sep_gate->nb[2];
                        const size_t u_off = static_cast<size_t>(ei) * sep_up->nb[2];
                        const size_t d_off = static_cast<size_t>(ei) * sep_down->nb[2];
                        struct ggml_tensor* gw = ggml_view_2d(vctx2,
                            const_cast<struct ggml_tensor*>(sep_gate),
                            sep_gate->ne[0], sep_gate->ne[1], sep_gate->nb[1], g_off);
                        struct ggml_tensor* uw = ggml_view_2d(vctx2,
                            const_cast<struct ggml_tensor*>(sep_up),
                            sep_up->ne[0], sep_up->ne[1], sep_up->nb[1], u_off);
                        struct ggml_tensor* dw = ggml_view_2d(vctx2,
                            const_cast<struct ggml_tensor*>(sep_down),
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

    // Auto-compute head dimensions from weight tensor shapes.
    // For hybrid SSM models (Qwen3.5), layer 0 may be SSM (no wk/wv),
    // so search for the first attention layer.
    struct ggml_tensor* wq_ref = nullptr;
    struct ggml_tensor* wk_ref = nullptr;
    struct ggml_tensor* wv_ref = nullptr;
    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        auto* wq_i = model->layers[i].Get(model_keys::kAttnQWeight);
        auto* wk_i = model->layers[i].Get(model_keys::kAttnKWeight);
        auto* wv_i = model->layers[i].Get(model_keys::kAttnVWeight);
        if (wq_i && !wq_ref) wq_ref = wq_i;
        if (wk_i && !wk_ref) wk_ref = wk_i;
        if (wv_i && !wv_ref) wv_ref = wv_i;
        if (wq_ref && wk_ref && wv_ref) break;
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
    if (model->hparams.n_embd_head_k == 0 && wk_ref) {
        model->hparams.n_embd_head_k = wk_ref->ne[1] / model->hparams.n_head_kv;
    }
    if (model->hparams.n_embd_head_v == 0 && wv_ref) {
        model->hparams.n_embd_head_v = wv_ref->ne[1] / model->hparams.n_head_kv;
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
        const int alpha_beta_expected = n_heads * static_cast<int>(model->hparams.n_embd);
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

                if (!canonicalize_conv1d("ssm_conv1d", conv, state.conv1d_f32, i) ||
                    !canonicalize_head_by_embd("ssm_alpha", alpha, state.alpha_f32, i) ||
                    !canonicalize_head_by_embd("ssm_beta", beta, state.beta_f32, i) ||
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
                    std::cout << "[DenseCore] Qwen3.5 SSM tensor types: conv1d=" << ggml_type_name(conv->type)
                              << " alpha=" << ggml_type_name(alpha->type) << " beta=" << ggml_type_name(beta->type)
                              << " dt_bias=" << ggml_type_name(dt->type) << " ssm_a=" << ggml_type_name(ssm_a->type)
                              << " norm=" << ggml_type_name(norm->type) << std::endl;
                    std::cout << "[DenseCore] Qwen3.5 SSM tensor shapes: conv1d=" << shape_string(conv)
                              << " alpha=" << shape_string(alpha) << " beta=" << shape_string(beta)
                              << " dt_bias=" << shape_string(dt) << " A_log=" << shape_string(ssm_a)
                              << " norm=" << shape_string(norm) << std::endl;
                    std::cout << "[DenseCore] Qwen3.5 SSM canonical layouts: conv1d=" << state.conv1d_f32.size()
                              << " alpha=" << state.alpha_f32.size() << " beta=" << state.beta_f32.size()
                              << " dt_bias=" << state.dt_bias_f32.size() << " A_log=" << state.a_log_f32.size()
                              << " norm=" << state.norm_f32.size() << " norm_semantics="
                              << (state.norm_layout == Qwen35SSMNormLayout::SHARED_HEAD_DIM ? "shared_head_dim"
                                  : state.norm_layout == Qwen35SSMNormLayout::FLATTENED_D_INNER ? "flattened_d_inner"
                                                                                                  : "invalid")
                              << std::endl;
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
    bool needs_rope = !model->has_vision && !model->has_whisper && model->arch != ModelArch::UNKNOWN;
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
        const char* graph_name = densecore::ModelGraphBridge::GetGraphName(model->arch);
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
    }

    // ==========================================================================
    // oneDNN Weight Prepack (Prefill MatMul Acceleration)
    // ==========================================================================
    auto prepare_weight = [&](struct ggml_tensor* t, const char* name) {
        if (!t || !t->data) return;
        if (model->int4_weight_bindings.find(t) != model->int4_weight_bindings.end()) return;
        if (model->fp8_weight_bindings.find(t) != model->fp8_weight_bindings.end()) return;
        const densecore::DType dtype = densecore::GgmlTypeToDType(t->type);
        if (dtype == densecore::DType::UNKNOWN) return;
        densecore::PrepareMatmulWeights(t->data, t->ne[0], t->ne[1], dtype, name);
        model->onednn_weights.push_back({t, dtype, t->ne[0], t->ne[1]});
    };

    prepare_weight(model->output, "output");
    for (size_t i = 0; i < model->layers.size(); ++i) {
        auto& layer = model->layers[i];
        prepare_weight(layer.Get(model_keys::kAttnQWeight), "wq");
        prepare_weight(layer.Get(model_keys::kAttnKWeight), "wk");
        prepare_weight(layer.Get(model_keys::kAttnVWeight), "wv");
        prepare_weight(layer.Get(model_keys::kAttnOWeight), "wo");
        prepare_weight(layer.Get(model_keys::kFfnGate), "w1");
        prepare_weight(layer.Get(model_keys::kFfnDown), "w2");
        prepare_weight(layer.Get(model_keys::kFfnUp), "w3");
    }

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

    std::cout << "[DenseCore] SmartLoader: Model loaded successfully (" << ctx_size / 1024 << " KB header)."
              << std::endl;
    return model;
}
