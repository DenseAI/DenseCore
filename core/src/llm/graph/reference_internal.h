#pragma once
struct ggml_tensor;
void GatherTokenHeadContiguous(const struct ggml_tensor* src, int token_idx, int head_dim, int n_head_kv, float* out);
