// NOTE: MoEUserData is defined in runtime/inference_types_internal.h

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "densecore/models/lfm2_shortconv_math.h"

#include "llm/moe/exec_userdata.inl"
#include "llm/moe/glm_dsa_callbacks.inl"
#include "llm/moe/moe_expert_bindings.inl"
#include "llm/moe/moe_callback_support.inl"
#include "llm/moe/moe_callback_exec.inl"
#include "llm/moe/ssm_support.inl"
#include "llm/moe/debug_probe_callbacks.inl"
#include "llm/moe/ssm_reference_support.inl"
#include "llm/moe/ssm_conv_callbacks.inl"
#include "llm/moe/lfm2_shortconv_callbacks.inl"
#include "llm/moe/qwen_ssm_delta_callbacks.inl"
