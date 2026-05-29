#include "densecore/runtime/inference.h"

#include "densecore/backend/flash_attention.h"
#include "densecore/backend/hardware_topology.h"  // For compute thread affinity
#include "densecore/backend/matmul_backend.h"
#include "densecore/models/model_graph_capabilities.h"
#include "densecore/models/transformer_graph_builder.h"  // Strategy Pattern for graph building
#include "densecore/runtime/optimization_bridge.h"       // Runtime SIMD dispatch
#include "ggml-cpu.h"                                    // For ggml_get_type_traits_cpu (vec_dot)
#include "ggml.h"                                        // Required for ggml_tensor definition
#include "models/model_inference_policy.h"
#include "runtime/inference_types_internal.h"  // Shared internal types

#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 32
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "densecore/backend/cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/kernels/paged_attention.h"
#include "densecore/memory/kv_cache.h"  // Added for KV cache
#include "densecore/memory/memory_pool.h"
#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/qwen35_ssm_math.h"
#include "densecore/moe/moe_routing.h"
#include "densecore/quantization/int4_types.h"  // For TensorInt4
#include "densecore/runtime/dtype_utils.h"      // For GgmlTypeToDType
#include "densecore/runtime/ggml_compute_policy.h"
#include "densecore/runtime/scheduler.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/attention/exec.h"
#include "llm/attention/internal.h"
#include "llm/config/runtime_config.h"
#include "llm/decoder/spec_runtime.h"
#include "llm/graph/planning.h"
#include "llm/models/common/family_internal.h"
#include "llm/runtime/deps.h"
#include "models/gemma4_packed_expert_layout.h"
#include "runtime/kernel_admission.h"
#include "runtime/runtime_env.h"

#ifndef DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER
#define DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER 1
#endif

#ifndef DENSECORE_DEFAULT_PRECOMPUTED_ROPE
#define DENSECORE_DEFAULT_PRECOMPUTED_ROPE 1
#endif

#ifndef DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM
#define DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM 0
#endif

#ifndef DENSECORE_DEFAULT_FUSED_QKV
#define DENSECORE_DEFAULT_FUSED_QKV 1
#endif

#include "runtime/inference_graph_support.inl"

// Keep these implementation chunks in the same translation unit so GGML custom
// callbacks, thread-local pools, and anonymous-namespace helpers preserve their
// original linkage and runtime behavior while still splitting this file by role.
#include "runtime/inference_runtime.inl"

#include "runtime/inference_matmul.inl"


#include "llm/moe/exec.inl"

static struct ggml_tensor* BuildTransformerGraphInlineImpl(TransformerModel* model, PagedKVCache* cache,
                                                           struct ggml_context* ctx_c, const BatchSpec& batch,
                                                           bool embedding_mode, struct ggml_cgraph* gf,
                                                           struct ggml_tensor** out_embd, struct ggml_tensor** out_pos);
static struct ggml_tensor* BuildBertEncoderEmbeddingGraph(TransformerModel* model, struct ggml_context* ctx_c,
                                                          const BatchSpec& batch, struct ggml_cgraph* gf,
                                                          struct ggml_tensor** out_embd, struct ggml_tensor** out_pos);

#include "runtime/inference_graph_dispatch.inl"

#include "runtime/inference_graph_inline_impl.inl"

#include "runtime/inference_hybrid_ssm_rebind.inl"


#include "runtime/inference_test_hooks.inl"
