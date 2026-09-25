#pragma once
struct ggml_tensor;

void cb_ssm_conv1d(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata);
void cb_ssm_conv1d_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_ssm_qwen35_delta_qkv_only(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                  void* userdata);
void cb_ssm_qwen35_delta_z_qkv(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                               int ith, int nth, void* userdata);
void cb_ssm_qwen35_delta_z_only(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                void* userdata);
void cb_ssm_qwen35_delta_z_qkv_alpha_beta(struct ggml_tensor* dst, const struct ggml_tensor* z,
                                          const struct ggml_tensor* qkv, const struct ggml_tensor* alpha_beta, int ith,
                                          int nth, void* userdata);
void cb_ssm_alpha_beta_qk_project_map3(struct ggml_tensor* dst, const struct ggml_tensor* placeholder,
                                       const struct ggml_tensor* input_src, const struct ggml_tensor* qkv, int ith,
                                       int nth, void* userdata);
void cb_ssm_qwen35_delta(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                         const struct ggml_tensor* c, int ith, int nth, void* userdata);
void cb_ssm_qwen35_delta_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_pack_glm_dsa_q(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata);
void cb_pack_glm_dsa_k(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                       const struct ggml_tensor* src2, int ith, int nth, void* userdata);
void cb_pack_glm_dsa_v(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata);
void cb_lfm2_shortconv(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                       int nth, void* userdata);
void cb_lfm2_shortconv_out_q4k_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_lfm2_shortconv_inout_q4k_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata);
