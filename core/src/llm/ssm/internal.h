#pragma once
#include <limits>

// Recurrent diagnostics share arguments, never their mutable budget/storage.
struct Qwen35SSMDebugScalars {
    float alpha = std::numeric_limits<float>::quiet_NaN();
    float beta = std::numeric_limits<float>::quiet_NaN();
    float softplus_alpha = std::numeric_limits<float>::quiet_NaN();
    float exp_a_log = std::numeric_limits<float>::quiet_NaN();
    float g = std::numeric_limits<float>::quiet_NaN();
    float decay = std::numeric_limits<float>::quiet_NaN();
    float beta_gate = std::numeric_limits<float>::quiet_NaN();
    float q_sum_sq = std::numeric_limits<float>::quiet_NaN();
    float k_sum_sq = std::numeric_limits<float>::quiet_NaN();
    float rms = std::numeric_limits<float>::quiet_NaN();
};

float SigmoidStable(float x);
void CheckSSMFiniteVector(int layer_idx, int token_idx, int seq_idx, int head_idx, const char* stage,
                          const char* var_name, const float* values, int n, const Qwen35SSMDebugScalars& scalars);
void RunSSMConv1DReference(const float* conv_state, const float* input, const float* weight, float* output,
                           int channels, int kernel_size);
void LogSSMCoreReferenceDiff(int layer_idx, int token_idx, int seq_idx, int head_idx, const char* stage,
                             const char* var_name, const float* actual, const float* reference, int n);
[[noreturn]] void FatalQwen35SSMRuntimeError(int layer_idx, int token_idx, int seq_idx, int head_idx,
                                             const char* message);
bool IsDebugSSMCoreReferenceEnabled();
