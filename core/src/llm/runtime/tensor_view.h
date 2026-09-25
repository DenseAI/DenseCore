#pragma once
#include <ggml.h>
#include "densecore/hal/tensor.h"
#include "densecore/runtime/dtype_utils.h"

inline densecore::Tensor GgmlToRowMajorTensor(const struct ggml_tensor* t) {
    if (!t) {
        return {};
    }

    densecore::Tensor out;
    out.data = t->data;
    out.dtype = densecore::GgmlTypeToDType(t->type);
    out.shape[0] = t->ne[1];
    out.shape[1] = t->ne[0];
    out.shape[2] = t->ne[2];
    out.shape[3] = t->ne[3];
    out.stride[0] = t->ne[0];
    out.stride[1] = 1;
    out.stride[2] = t->nb[2];
    out.stride[3] = t->nb[3];
    out.ndim = 2;
    out.device_type = densecore::DeviceType::CPU;
    return out;
}
