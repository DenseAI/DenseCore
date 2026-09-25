#pragma once
#include "densecore/runtime/inference.h"
void WriteCurrentBatchKvToCache(const BatchSpec* batch, const ggml_tensor* src, PagedKVCache* cache, int layer,
                                int head_dim, int n_head_kv, bool is_k, int token_begin, int token_end, int* writes_ok,
                                int* writes_skipped);
