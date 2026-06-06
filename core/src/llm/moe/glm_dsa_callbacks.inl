void cb_pack_glm_dsa_q(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !dst->data || !src1->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int q_head_dim = ud->qk_nope_head_dim + ud->qk_rope_head_dim;
    const size_t src_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* src = reinterpret_cast<const float*>(src1->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* src_row = src + static_cast<size_t>(t) * src_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* src_head = src_row + static_cast<size_t>(h) * q_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * q_head_dim;
            memcpy(dst_head, src_head + ud->qk_nope_head_dim,
                   static_cast<size_t>(ud->qk_rope_head_dim) * sizeof(float));
            memcpy(dst_head + ud->qk_rope_head_dim, src_head,
                   static_cast<size_t>(ud->qk_nope_head_dim) * sizeof(float));
        }
    }
}

void cb_pack_glm_dsa_k(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                       const struct ggml_tensor* src2, int ith, int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !src2 || !dst->data || !src1->data || !src2->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int k_head_dim = ud->qk_nope_head_dim + ud->qk_rope_head_dim;
    const int kv_proj_head_dim = ud->qk_nope_head_dim + ud->v_head_dim;
    const size_t kv_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t rope_row_stride = static_cast<size_t>(src2->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* kv_src = reinterpret_cast<const float*>(src1->data);
    const float* rope_src = reinterpret_cast<const float*>(src2->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* kv_row = kv_src + static_cast<size_t>(t) * kv_row_stride;
        const float* rope_row = rope_src + static_cast<size_t>(t) * rope_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* kv_head = kv_row + static_cast<size_t>(h) * kv_proj_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * k_head_dim;
            memcpy(dst_head, rope_row, static_cast<size_t>(ud->qk_rope_head_dim) * sizeof(float));
            memcpy(dst_head + ud->qk_rope_head_dim, kv_head, static_cast<size_t>(ud->qk_nope_head_dim) * sizeof(float));
        }
    }
}

void cb_pack_glm_dsa_v(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !dst->data || !src1->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int kv_proj_head_dim = ud->qk_nope_head_dim + ud->v_head_dim;
    const size_t src_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* src = reinterpret_cast<const float*>(src1->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* src_row = src + static_cast<size_t>(t) * src_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* src_head = src_row + static_cast<size_t>(h) * kv_proj_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * ud->v_head_dim;
            memcpy(dst_head, src_head + ud->qk_nope_head_dim, static_cast<size_t>(ud->v_head_dim) * sizeof(float));
        }
    }
}
