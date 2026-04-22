#ifndef DENSECORE_GEMMA4_PACKED_EXPERT_LAYOUT_H
#define DENSECORE_GEMMA4_PACKED_EXPERT_LAYOUT_H

#include "densecore/models/model_types.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace densecore::gemma4 {

enum class PackedGateUpLayoutKind : uint8_t {
    Unknown = 0,
    RowStacked3D,
    PlaneSeparated4D,
};

struct PackedProjectionView {
    ggml_tensor* tensor = nullptr;
    int expert_index = -1;
    int expert_axis = -1;
    int split_axis = -1;
    int split_index = -1;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t row_start = 0;
    size_t view_offset_bytes = 0;
};

struct PackedExpertViews {
    ggml_tensor* gate_up = nullptr;
    ggml_tensor* gate = nullptr;
    ggml_tensor* up = nullptr;
    ggml_tensor* down = nullptr;
    ggml_tensor* down_scale = nullptr;
    PackedProjectionView gate_view;
    PackedProjectionView up_view;
    PackedProjectionView down_view;
};

struct PackedExpertLayout {
    PackedGateUpLayoutKind gate_up_kind = PackedGateUpLayoutKind::Unknown;
    int num_experts = 0;
    int gate_up_expert_axis = -1;
    int down_expert_axis = -1;
    int64_t hidden_dim = 0;
    int64_t intermediate_dim = 0;
};

inline const char* PackedGateUpLayoutName(PackedGateUpLayoutKind kind) {
    switch (kind) {
    case PackedGateUpLayoutKind::RowStacked3D: return "row_stacked_3d";
    case PackedGateUpLayoutKind::PlaneSeparated4D: return "plane_separated_4d";
    default: return "unknown";
    }
}

inline bool InferPackedExpertLayout(const ggml_tensor* gate_up_root, const ggml_tensor* down_root,
                                    PackedExpertLayout* out, std::string* reason = nullptr) {
    if (!out) {
        if (reason) *reason = "missing output layout";
        return false;
    }
    *out = PackedExpertLayout{};
    if (!gate_up_root || !down_root) {
        if (reason) *reason = "missing packed expert roots";
        return false;
    }
    if (gate_up_root->ne[0] <= 0 || gate_up_root->ne[1] <= 0 || down_root->ne[0] <= 0 || down_root->ne[1] <= 0) {
        if (reason) *reason = "invalid packed expert dimensions";
        return false;
    }

    if (gate_up_root->ne[3] > 1 && gate_up_root->ne[2] == 2) {
        out->gate_up_kind = PackedGateUpLayoutKind::PlaneSeparated4D;
        out->gate_up_expert_axis = 3;
        out->num_experts = static_cast<int>(gate_up_root->ne[3]);
        out->hidden_dim = gate_up_root->ne[0];
        out->intermediate_dim = gate_up_root->ne[1];
    } else if (gate_up_root->ne[2] > 1 && gate_up_root->ne[1] >= 2 && (gate_up_root->ne[1] % 2) == 0) {
        out->gate_up_kind = PackedGateUpLayoutKind::RowStacked3D;
        out->gate_up_expert_axis = 2;
        out->num_experts = static_cast<int>(gate_up_root->ne[2]);
        out->hidden_dim = gate_up_root->ne[0];
        out->intermediate_dim = gate_up_root->ne[1] / 2;
    } else {
        if (reason) *reason = "unsupported Gemma4 packed gate_up root layout";
        return false;
    }

    if (down_root->ne[0] != out->intermediate_dim || down_root->ne[1] != out->hidden_dim) {
        if (reason) *reason = "packed down root dimensions do not match gate_up layout";
        return false;
    }

    if (down_root->ne[3] == out->num_experts && down_root->ne[3] > 1) {
        out->down_expert_axis = 3;
        return true;
    }
    if (down_root->ne[2] == out->num_experts && down_root->ne[2] > 1) {
        out->down_expert_axis = 2;
        return true;
    }

    if (reason) *reason = "unsupported Gemma4 packed down root layout";
    return false;
}

inline size_t ScaleMetaOffsetFloats(const ggml_tensor* meta_root, int expert_axis, int expert_index, int split_axis,
                                    int split_index, int64_t row_start) {
    if (!meta_root || !meta_root->data) {
        return 0;
    }
    size_t offset = static_cast<size_t>(row_start) * static_cast<size_t>(meta_root->nb[1] / sizeof(float));
    if (split_axis >= 0 && split_index >= 0 && meta_root->ne[split_axis] > 1) {
        offset += static_cast<size_t>(split_index) * static_cast<size_t>(meta_root->nb[split_axis] / sizeof(float));
    }
    if (expert_axis >= 0 && expert_index >= 0 && meta_root->ne[expert_axis] > 1) {
        offset += static_cast<size_t>(expert_index) * static_cast<size_t>(meta_root->nb[expert_axis] / sizeof(float));
    }
    return offset;
}

inline ggml_tensor* Make2DView(ggml_context* vctx, ggml_tensor* root, int64_t cols, int64_t rows, size_t offset_bytes) {
    if (!vctx || !root || cols <= 0 || rows <= 0) {
        return nullptr;
    }
    return ggml_view_2d(vctx, root, cols, rows, root->nb[1], offset_bytes);
}

inline bool ResolvePackedProjectionBinding(ggml_context* vctx, const TransformerModel::Int4WeightBinding& root_binding,
                                           const PackedProjectionView& slice, TransformerModel::Int4WeightBinding* out,
                                           std::string* reason = nullptr) {
    if (!out) {
        if (reason) *reason = "missing output binding";
        return false;
    }
    *out = TransformerModel::Int4WeightBinding{};
    if (!slice.tensor || !slice.tensor->data) {
        if (reason) *reason = "missing slice tensor";
        return false;
    }
    if (!root_binding.packed || !root_binding.scales || !root_binding.zeros || !root_binding.packed->data ||
        !root_binding.scales->data || !root_binding.zeros->data) {
        if (reason) *reason = "missing root binding tensors";
        return false;
    }
    if (root_binding.group_size <= 0 || root_binding.k != slice.cols || slice.rows <= 0) {
        if (reason) *reason = "root binding does not match canonical Gemma4 slice";
        return false;
    }

    ggml_tensor* scales_view =
        Make2DView(vctx, const_cast<ggml_tensor*>(root_binding.scales), root_binding.scales->ne[0], slice.rows,
                   ScaleMetaOffsetFloats(root_binding.scales, slice.expert_axis, slice.expert_index, slice.split_axis,
                                         slice.split_index, slice.row_start) *
                       sizeof(float));
    ggml_tensor* zeros_view =
        Make2DView(vctx, const_cast<ggml_tensor*>(root_binding.zeros), root_binding.zeros->ne[0], slice.rows,
                   ScaleMetaOffsetFloats(root_binding.zeros, slice.expert_axis, slice.expert_index, slice.split_axis,
                                         slice.split_index, slice.row_start) *
                       sizeof(float));
    if (!scales_view || !zeros_view) {
        if (reason) *reason = "failed to materialize Gemma4 scale/zero views";
        return false;
    }

    out->packed = slice.tensor;
    out->scales = scales_view;
    out->zeros = zeros_view;
    out->group_size = root_binding.group_size;
    out->k = slice.cols;
    out->n = slice.rows;
    return true;
}

inline bool MakePackedExpertViews(ggml_context* vctx, ggml_tensor* gate_up_root, ggml_tensor* down_root,
                                  ggml_tensor* down_scale_root, int expert_index, PackedExpertViews* out,
                                  std::string* reason = nullptr) {
    if (!out) {
        if (reason) *reason = "missing packed expert output";
        return false;
    }
    *out = PackedExpertViews{};

    PackedExpertLayout layout;
    if (!InferPackedExpertLayout(gate_up_root, down_root, &layout, reason)) {
        return false;
    }
    if (expert_index < 0 || expert_index >= layout.num_experts) {
        if (reason) *reason = "Gemma4 expert index out of range";
        return false;
    }

    size_t gate_up_expert_offset = 0;
    if (layout.gate_up_expert_axis >= 0) {
        gate_up_expert_offset =
            static_cast<size_t>(expert_index) * static_cast<size_t>(gate_up_root->nb[layout.gate_up_expert_axis]);
    }
    if (layout.gate_up_kind == PackedGateUpLayoutKind::RowStacked3D) {
        out->gate_up = ggml_view_2d(vctx, gate_up_root, layout.hidden_dim, layout.intermediate_dim * 2,
                                    gate_up_root->nb[1], gate_up_expert_offset);
        out->gate_view = {
            Make2DView(vctx, gate_up_root, layout.hidden_dim, layout.intermediate_dim, gate_up_expert_offset),
            expert_index,
            layout.gate_up_expert_axis,
            -1,
            -1,
            layout.intermediate_dim,
            layout.hidden_dim,
            0,
            gate_up_expert_offset};
        out->up_view = {Make2DView(vctx, gate_up_root, layout.hidden_dim, layout.intermediate_dim,
                                   gate_up_expert_offset + static_cast<size_t>(layout.intermediate_dim) *
                                                               static_cast<size_t>(gate_up_root->nb[1])),
                        expert_index,
                        layout.gate_up_expert_axis,
                        -1,
                        -1,
                        layout.intermediate_dim,
                        layout.hidden_dim,
                        layout.intermediate_dim,
                        gate_up_expert_offset +
                            static_cast<size_t>(layout.intermediate_dim) * static_cast<size_t>(gate_up_root->nb[1])};
    } else {
        out->gate_up = ggml_view_3d(vctx, gate_up_root, layout.hidden_dim, layout.intermediate_dim, 2,
                                    gate_up_root->nb[1], gate_up_root->nb[2], gate_up_expert_offset);
        out->gate_view = {
            Make2DView(vctx, gate_up_root, layout.hidden_dim, layout.intermediate_dim, gate_up_expert_offset),
            expert_index,
            layout.gate_up_expert_axis,
            2,
            0,
            layout.intermediate_dim,
            layout.hidden_dim,
            0,
            gate_up_expert_offset};
        out->up_view = {Make2DView(vctx, gate_up_root, layout.hidden_dim, layout.intermediate_dim,
                                   gate_up_expert_offset + static_cast<size_t>(gate_up_root->nb[2])),
                        expert_index,
                        layout.gate_up_expert_axis,
                        2,
                        1,
                        layout.intermediate_dim,
                        layout.hidden_dim,
                        0,
                        gate_up_expert_offset + static_cast<size_t>(gate_up_root->nb[2])};
    }
    out->gate = out->gate_view.tensor;
    out->up = out->up_view.tensor;

    const size_t down_offset =
        static_cast<size_t>(expert_index) * static_cast<size_t>(down_root->nb[layout.down_expert_axis]);
    out->down_view = {Make2DView(vctx, down_root, layout.intermediate_dim, layout.hidden_dim, down_offset),
                      expert_index,
                      layout.down_expert_axis,
                      -1,
                      -1,
                      layout.hidden_dim,
                      layout.intermediate_dim,
                      0,
                      down_offset};
    out->down = out->down_view.tensor;

    if (down_scale_root) {
        if (down_scale_root->type != GGML_TYPE_F32) {
            if (reason) *reason = "Gemma4 down scale sidecar must be F32";
            return false;
        }
        out->down_scale = Make2DView(vctx, down_scale_root, layout.intermediate_dim, layout.hidden_dim,
                                     static_cast<size_t>(expert_index) *
                                         static_cast<size_t>(down_scale_root->nb[layout.down_expert_axis]));
    }

    if (!out->gate || !out->up || !out->down || !out->gate_up) {
        if (reason) *reason = "failed to materialize Gemma4 packed expert views";
        return false;
    }
    return true;
}

}  // namespace densecore::gemma4

#endif  // DENSECORE_GEMMA4_PACKED_EXPERT_LAYOUT_H
