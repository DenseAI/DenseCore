/**
 * @file metal_universal_ops.mm
 * @brief Metal OpRegistry wrappers for Universal Transformer Operations
 *
 * Provides Metal GPU implementations of WindowAttention, TemporalAttention,
 * TriangularAttention, PatchEmbed3D, and Patchify/Unpatchify for Apple Silicon
 * acceleration.
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"

namespace densecore {
namespace kernels {
namespace {

MTLSize ChooseThreadgroupSize(id<MTLComputePipelineState> pipeline, MTLSize grid) {
    if (!pipeline) {
        return MTLSizeMake(1, 1, 1);
    }
    const NSUInteger simd_width = std::max<NSUInteger>(1, [pipeline threadExecutionWidth]);
    const NSUInteger max_threads =
        std::max<NSUInteger>(simd_width, [pipeline maxTotalThreadsPerThreadgroup]);

    const NSUInteger tx = std::max<NSUInteger>(1, std::min(grid.width, simd_width));
    const NSUInteger ty_cap = std::max<NSUInteger>(1, max_threads / tx);
    const NSUInteger ty = std::max<NSUInteger>(1, std::min(grid.height, ty_cap));
    const NSUInteger tz_cap = std::max<NSUInteger>(1, max_threads / (tx * ty));
    const NSUInteger tz = std::max<NSUInteger>(1, std::min(grid.depth, tz_cap));
    return MTLSizeMake(tx, ty, tz);
}

id<MTLBuffer> WrapSharedBuffer(id<MTLDevice> device, const void* data, size_t bytes) {
    return [device newBufferWithBytesNoCopy:(void*)data
                                     length:bytes
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

DenseCoreOp* GetCpuFallback(OpType op) {
    return OpRegistry::Instance().GetBest(op, DeviceType::CPU);
}

DenseCoreOp* GetExactDeviceOp(OpType op, DeviceType device) {
    return OpRegistry::Instance().Get(op, device);
}

template <OpType kOpType, int kPriority = 10>
class AppleNpuForwardOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        if (auto* metal = GetExactDeviceOp(kOpType, DeviceType::METAL)) {
            metal->Execute(inputs, outputs, params);
            return;
        }
        if (auto* cpu = GetExactDeviceOp(kOpType, DeviceType::CPU)) {
            cpu->Execute(inputs, outputs, params);
            return;
        }
        throw std::runtime_error(std::string("[AppleNpuForwardOp] No delegate registered for ") +
                                 OpTypeName(kOpType));
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::NPU; }

    OpCapabilities GetCapabilities() const override {
        // Explicit NPU registration for admission/dispatch ownership. This is not
        // a claim of ANE-native execution; it forwards to Metal when available
        // and otherwise preserves CPU fallback.
        return {.supports_fp16 = true, .priority = kPriority};
    }
};

// ============================================================================
// Metal Context Singleton
// ============================================================================

class MetalContext {
public:
    static MetalContext& Instance() {
        static MetalContext instance;
        return instance;
    }

    id<MTLDevice> device() { return device_; }
    id<MTLCommandQueue> queue() { return queue_; }
    id<MTLLibrary> library() { return library_; }

    id<MTLComputePipelineState> GetPipeline(const std::string& name) {
        auto it = pipelines_.find(name);
        if (it != pipelines_.end())
            return it->second;

        NSString* nsName = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> function = [library_ newFunctionWithName:nsName];
        if (!function) {
            for (id<MTLLibrary> lib : runtime_libraries_) {
                function = [lib newFunctionWithName:nsName];
                if (function)
                    break;
            }
        }
        if (!function)
            return nil;

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function
                                                                                      error:&error];
        if (pipeline) {
            pipelines_[name] = pipeline;
        }
        return pipeline;
    }

    bool IsAvailable() const {
        return device_ != nil && (library_ != nil || !runtime_libraries_.empty());
    }

private:
    MetalContext() {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_)
            return;

        queue_ = [device_ newCommandQueue];

        // Load shader library from bundle or default path
        NSError* error = nil;

        // Try loading from bundle first
        NSBundle* bundle = [NSBundle mainBundle];
        NSArray<NSString*>* libraryNames = @[ @"universal_ops", @"densecore" ];
        for (NSString* libraryName in libraryNames) {
            NSString* libraryPath = [bundle pathForResource:libraryName ofType:@"metallib"];
            if (!libraryPath) {
                continue;
            }
            library_ = [device_ newLibraryWithFile:libraryPath error:&error];
            if (library_) {
                break;
            }
        }

        // Fallback: compile shaders at runtime
        if (!library_) {
            NSURL* shaderURL = [[NSURL fileURLWithPath:[[NSBundle mainBundle] bundlePath]]
                URLByAppendingPathComponent:@"../shaders"];

            // Try to load individual shader files
            NSArray<NSString*>* shaderFiles = @[
                @"window_attention.metal", @"temporal_attention.metal",
                @"triangular_attention.metal", @"patch_embed_3d.metal", @"point_cloud_ops.metal"
            ];

            // Compile each shader if metallib not available
            for (NSString* shaderFile in shaderFiles) {
                NSURL* fileURL = [shaderURL URLByAppendingPathComponent:shaderFile];
                NSString* source = [NSString stringWithContentsOfURL:fileURL
                                                            encoding:NSUTF8StringEncoding
                                                               error:&error];
                if (source) {
                    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                    options.languageVersion = MTLLanguageVersion2_4;

                    id<MTLLibrary> shaderLib = [device_ newLibraryWithSource:source
                                                                     options:options
                                                                       error:&error];
                    if (shaderLib) {
                        runtime_libraries_.push_back(shaderLib);
                    }
                }
            }
            if (!runtime_libraries_.empty()) {
                library_ = runtime_libraries_.front();
            }
        }
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    std::vector<id<MTLLibrary>> runtime_libraries_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
};

// ============================================================================
// Metal Window Attention Op
// ============================================================================

class MetalWindowAttentionOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        if (inputs.size() < 3 || outputs.empty())
            return;

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable())
            return;

        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("window_attention_forward");
        if (!pipeline)
            return;

        const Tensor* query = inputs[0];
        const Tensor* key = inputs[1];
        const Tensor* value = inputs[2];
        const Tensor* bias = inputs.size() > 3 ? inputs[3] : nullptr;
        Tensor* output = outputs[0];

        // Extract params
        struct WinAttnParams {
            int window_size = 7;
            int shift_size = 0;
            int num_heads = 8;
            float scale = 0.0f;
        };
        WinAttnParams default_params;
        auto* p = params ? static_cast<const WinAttnParams*>(params) : &default_params;

        const uint window_size = p->window_size;
        const uint num_heads = p->num_heads;
        const uint head_dim = query->shape[3] / num_heads;
        const uint window_area = window_size * window_size;

        float scale = p->scale;
        if (scale == 0.0f) {
            scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        }

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Set buffers (wrapped for zero-copy on UMA)
        id<MTLBuffer> qBuffer =
            [ctx.device() newBufferWithBytesNoCopy:(void*)query->DataAs<float>()
                                            length:query->NumElements() * sizeof(float)
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
        id<MTLBuffer> kBuffer =
            [ctx.device() newBufferWithBytesNoCopy:(void*)key->DataAs<float>()
                                            length:key->NumElements() * sizeof(float)
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
        id<MTLBuffer> vBuffer =
            [ctx.device() newBufferWithBytesNoCopy:(void*)value->DataAs<float>()
                                            length:value->NumElements() * sizeof(float)
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
        id<MTLBuffer> oBuffer =
            [ctx.device() newBufferWithBytesNoCopy:output->DataAs<float>()
                                            length:output->NumElements() * sizeof(float)
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];

        [encoder setBuffer:qBuffer offset:0 atIndex:0];
        [encoder setBuffer:kBuffer offset:0 atIndex:1];
        [encoder setBuffer:vBuffer offset:0 atIndex:2];
        [encoder setBuffer:oBuffer offset:0 atIndex:3];

        if (bias && bias->IsValid()) {
            id<MTLBuffer> biasBuffer =
                [ctx.device() newBufferWithBytesNoCopy:(void*)bias->DataAs<float>()
                                                length:bias->NumElements() * sizeof(float)
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
            [encoder setBuffer:biasBuffer offset:0 atIndex:4];
        }

        [encoder setBytes:&scale length:sizeof(float) atIndex:5];
        [encoder setBytes:&window_area length:sizeof(uint) atIndex:6];
        [encoder setBytes:&head_dim length:sizeof(uint) atIndex:7];
        [encoder setBytes:&num_heads length:sizeof(uint) atIndex:8];

        uint use_bias = (bias && bias->IsValid()) ? 1 : 0;
        [encoder setBytes:&use_bias length:sizeof(uint) atIndex:9];

        // Dispatch
        const int64_t B = query->shape[0];
        const int64_t H = query->shape[1];
        const int64_t W = query->shape[2];
        const int64_t num_windows = (H / window_size) * (W / window_size);

        MTLSize gridSize = MTLSizeMake(window_area, num_heads, B * num_windows);
        MTLSize threadGroupSize = ChooseThreadgroupSize(pipeline, gridSize);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 32 * 1024 * 1024,
                .memory_bandwidth_gbps = 400,
                .priority = 100};
    }
};

// ============================================================================
// Metal Temporal Attention Op
// ============================================================================

class MetalTemporalAttentionOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        if (inputs.size() < 3 || outputs.empty())
            return;

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable())
            return;

        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("temporal_attention_forward");
        if (!pipeline)
            return;

        const Tensor* query = inputs[0];
        const Tensor* key = inputs[1];
        const Tensor* value = inputs[2];
        Tensor* output = outputs[0];

        struct TempAttnParams {
            int num_heads = 8;
            float scale = 0.0f;
            bool causal = false;
            int window_size = 0;
        };
        TempAttnParams default_params;
        auto* p = params ? static_cast<const TempAttnParams*>(params) : &default_params;

        const uint T = query->shape[1];  // time dimension
        const uint num_heads = p->num_heads;
        const uint head_dim = query->shape[2] / num_heads;

        float scale = p->scale;
        if (scale == 0.0f) {
            scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        }

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        auto wrapBuffer = [&ctx](const float* data, size_t count) {
            return [ctx.device() newBufferWithBytesNoCopy:(void*)data
                                                   length:count * sizeof(float)
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        };

        [encoder setBuffer:wrapBuffer(query->DataAs<float>(), query->NumElements())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:wrapBuffer(key->DataAs<float>(), key->NumElements()) offset:0 atIndex:1];
        [encoder setBuffer:wrapBuffer(value->DataAs<float>(), value->NumElements())
                    offset:0
                   atIndex:2];
        [encoder setBuffer:wrapBuffer(output->DataAs<float>(), output->NumElements())
                    offset:0
                   atIndex:3];

        [encoder setBytes:&scale length:sizeof(float) atIndex:4];
        [encoder setBytes:&T length:sizeof(uint) atIndex:5];
        [encoder setBytes:&head_dim length:sizeof(uint) atIndex:6];
        [encoder setBytes:&num_heads length:sizeof(uint) atIndex:7];

        uint causal = p->causal ? 1 : 0;
        uint window = p->window_size;
        [encoder setBytes:&causal length:sizeof(uint) atIndex:8];
        [encoder setBytes:&window length:sizeof(uint) atIndex:9];

        const int64_t batch_spatial = query->shape[0];

        MTLSize gridSize = MTLSizeMake(T, num_heads, batch_spatial);
        MTLSize threadGroupSize = ChooseThreadgroupSize(pipeline, gridSize);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 32 * 1024 * 1024,
                .memory_bandwidth_gbps = 400,
                .priority = 100};
    }
};

// ============================================================================
// Metal Triangular Attention Op
// ============================================================================

class MetalTriangularAttentionOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        if (inputs.size() < 4 || outputs.empty())
            return;

        auto cpu_fallback = [&]() {
            auto* cpu_op =
                OpRegistry::Instance().GetBest(OpType::TriangularAttention, DeviceType::CPU);
            if (cpu_op)
                cpu_op->Execute(inputs, outputs, params);
        };

        const TriangularAttentionParams* p = static_cast<const TriangularAttentionParams*>(params);
        TriangularAttentionParams default_params;
        if (!p)
            p = &default_params;

        const Tensor* pair = inputs[0];
        const Tensor* qw = inputs[1];
        const Tensor* kw = inputs[2];
        const Tensor* vw = inputs[3];
        Tensor* output = outputs[0];
        if (!pair || !qw || !kw || !vw || !output) {
            cpu_fallback();
            return;
        }

        const int64_t B = pair->shape[0];
        const int64_t L = pair->shape[1];
        const int64_t D = pair->shape[3];
        const int64_t H = p->num_heads;
        if (B <= 0 || L <= 0 || D <= 0 || H <= 0 || (D % H) != 0) {
            cpu_fallback();
            return;
        }
        const int64_t D_head = D / H;
        if (D_head <= 0 || D_head > 128) {
            cpu_fallback();
            return;
        }

        // Current Metal kernel only supports full-output layout [B, L, L, D].
        const bool has_tile = p->tile_row_start >= 0 && p->tile_row_end > p->tile_row_start &&
                              p->tile_col_start >= 0 && p->tile_col_end > p->tile_col_start;
        if (has_tile || output->shape[0] != B || output->shape[1] != L || output->shape[2] != L ||
            output->shape[3] != D) {
            cpu_fallback();
            return;
        }

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable()) {
            cpu_fallback();
            return;
        }

        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("triangular_attention_forward");
        if (!pipeline) {
            cpu_fallback();
            return;
        }

        float scale = p->scale;
        if (scale == 0.0f) {
            scale = 1.0f / std::sqrt(static_cast<float>(D_head));
        }

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        auto wrapBuffer = [&ctx](const float* data, size_t count) {
            return [ctx.device() newBufferWithBytesNoCopy:(void*)data
                                                   length:count * sizeof(float)
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        };

        [encoder setBuffer:wrapBuffer(pair->DataAs<float>(), pair->NumElements())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:wrapBuffer(qw->DataAs<float>(), qw->NumElements()) offset:0 atIndex:1];
        [encoder setBuffer:wrapBuffer(kw->DataAs<float>(), kw->NumElements()) offset:0 atIndex:2];
        [encoder setBuffer:wrapBuffer(vw->DataAs<float>(), vw->NumElements()) offset:0 atIndex:3];
        [encoder setBuffer:wrapBuffer(output->DataAs<float>(), output->NumElements())
                    offset:0
                   atIndex:4];

        const uint b = static_cast<uint>(B);
        const uint l = static_cast<uint>(L);
        const uint d = static_cast<uint>(D);
        const uint h = static_cast<uint>(H);
        const uint head_dim = static_cast<uint>(D_head);
        const uint starting = p->starting ? 1u : 0u;

        [encoder setBytes:&b length:sizeof(uint) atIndex:5];
        [encoder setBytes:&l length:sizeof(uint) atIndex:6];
        [encoder setBytes:&d length:sizeof(uint) atIndex:7];
        [encoder setBytes:&h length:sizeof(uint) atIndex:8];
        [encoder setBytes:&head_dim length:sizeof(uint) atIndex:9];
        [encoder setBytes:&starting length:sizeof(uint) atIndex:10];
        [encoder setBytes:&scale length:sizeof(float) atIndex:11];

        MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(L), static_cast<NSUInteger>(L),
                                       static_cast<NSUInteger>(B * H));
        MTLSize threadGroupSize = ChooseThreadgroupSize(pipeline, gridSize);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 0,
                .memory_bandwidth_gbps = 200,
                .priority = 100};
    }
};

// ============================================================================
// Metal PatchEmbed3D Op
// ============================================================================

class MetalPatchEmbed3DOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        if (inputs.size() < 2 || outputs.empty())
            return;

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable())
            return;

        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("patch_embed_3d");
        if (!pipeline)
            return;

        const Tensor* video = inputs[0];
        const Tensor* weights = inputs[1];
        const Tensor* bias = inputs.size() > 2 ? inputs[2] : nullptr;
        Tensor* output = outputs[0];

        struct PatchEmbed3DParams {
            int in_channels = 3;
            int embed_dim = 768;
            int temporal_patch = 2;
            int patch_h = 16;
            int patch_w = 16;
        };
        PatchEmbed3DParams default_params;
        auto* p = params ? static_cast<const PatchEmbed3DParams*>(params) : &default_params;

        const uint B = video->shape[0];
        const uint C = p->in_channels;
        const uint T = video->shape[1] / C;
        const uint H = video->shape[2];
        const uint W = video->shape[3];
        const uint Pt = p->temporal_patch;
        const uint Ph = p->patch_h;
        const uint Pw = p->patch_w;
        const uint embed_dim = p->embed_dim;

        const uint num_patches = (T / Pt) * (H / Ph) * (W / Pw);

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        auto wrapBuffer = [&ctx](const float* data, size_t count) {
            return [ctx.device() newBufferWithBytesNoCopy:(void*)data
                                                   length:count * sizeof(float)
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        };

        [encoder setBuffer:wrapBuffer(video->DataAs<float>(), video->NumElements())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:wrapBuffer(weights->DataAs<float>(), weights->NumElements())
                    offset:0
                   atIndex:1];

        if (bias && bias->IsValid()) {
            [encoder setBuffer:wrapBuffer(bias->DataAs<float>(), bias->NumElements())
                        offset:0
                       atIndex:2];
        }

        [encoder setBuffer:wrapBuffer(output->DataAs<float>(), output->NumElements())
                    offset:0
                   atIndex:3];

        [encoder setBytes:&B length:sizeof(uint) atIndex:4];
        [encoder setBytes:&C length:sizeof(uint) atIndex:5];
        [encoder setBytes:&T length:sizeof(uint) atIndex:6];
        [encoder setBytes:&H length:sizeof(uint) atIndex:7];
        [encoder setBytes:&W length:sizeof(uint) atIndex:8];
        [encoder setBytes:&Pt length:sizeof(uint) atIndex:9];
        [encoder setBytes:&Ph length:sizeof(uint) atIndex:10];
        [encoder setBytes:&Pw length:sizeof(uint) atIndex:11];
        [encoder setBytes:&embed_dim length:sizeof(uint) atIndex:12];

        uint use_bias = (bias && bias->IsValid()) ? 1 : 0;
        [encoder setBytes:&use_bias length:sizeof(uint) atIndex:13];

        MTLSize gridSize = MTLSizeMake(embed_dim, num_patches, B);
        MTLSize threadGroupSize = MTLSizeMake(32, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 32 * 1024 * 1024,
                .memory_bandwidth_gbps = 400,
                .priority = 100};
    }
};

// ============================================================================
// Metal Patchify/Unpatchify Ops
// ============================================================================

class MetalPatchifyOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        if (inputs.empty() || outputs.empty())
            return;

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable())
            return;

        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("patchify_2d");
        if (!pipeline)
            return;

        const Tensor* image = inputs[0];
        Tensor* patches = outputs[0];

        struct PatchParams {
            int patch_size = 16;
        };
        PatchParams default_params;
        auto* p = params ? static_cast<const PatchParams*>(params) : &default_params;

        const uint B = image->shape[0];
        const uint C = image->shape[1];
        const uint H = image->shape[2];
        const uint W = image->shape[3];
        const uint P = p->patch_size;
        const uint num_patches = (H / P) * (W / P);
        const uint patch_dim = C * P * P;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        auto wrapBuffer = [&ctx](const float* data, size_t count) {
            return [ctx.device() newBufferWithBytesNoCopy:(void*)data
                                                   length:count * sizeof(float)
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        };

        [encoder setBuffer:wrapBuffer(image->DataAs<float>(), image->NumElements())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:wrapBuffer(patches->DataAs<float>(), patches->NumElements())
                    offset:0
                   atIndex:1];

        [encoder setBytes:&B length:sizeof(uint) atIndex:2];
        [encoder setBytes:&C length:sizeof(uint) atIndex:3];
        [encoder setBytes:&H length:sizeof(uint) atIndex:4];
        [encoder setBytes:&W length:sizeof(uint) atIndex:5];
        [encoder setBytes:&P length:sizeof(uint) atIndex:6];

        MTLSize gridSize = MTLSizeMake(patch_dim, num_patches, B);
        MTLSize threadGroupSize = MTLSizeMake(64, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true, .priority = 100};
    }
};

class MetalUnpatchifyOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        if (inputs.empty() || outputs.empty())
            return;

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable())
            return;

        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("unpatchify_2d");
        if (!pipeline)
            return;

        const Tensor* patches = inputs[0];
        Tensor* image = outputs[0];

        struct UnpatchParams {
            int patch_size = 16;
            int height = 256;
            int width = 256;
            int channels = 3;
        };
        UnpatchParams default_params;
        auto* p = params ? static_cast<const UnpatchParams*>(params) : &default_params;

        const uint B = patches->shape[0];
        const uint C = p->channels;
        const uint H = p->height;
        const uint W = p->width;
        const uint P = p->patch_size;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        auto wrapBuffer = [&ctx](const float* data, size_t count) {
            return [ctx.device() newBufferWithBytesNoCopy:(void*)data
                                                   length:count * sizeof(float)
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        };

        [encoder setBuffer:wrapBuffer(patches->DataAs<float>(), patches->NumElements())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:wrapBuffer(image->DataAs<float>(), image->NumElements())
                    offset:0
                   atIndex:1];

        [encoder setBytes:&B length:sizeof(uint) atIndex:2];
        [encoder setBytes:&C length:sizeof(uint) atIndex:3];
        [encoder setBytes:&H length:sizeof(uint) atIndex:4];
        [encoder setBytes:&W length:sizeof(uint) atIndex:5];
        [encoder setBytes:&P length:sizeof(uint) atIndex:6];

        MTLSize gridSize = MTLSizeMake(C, H * W, B);
        MTLSize threadGroupSize = MTLSizeMake(64, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true, .priority = 100};
    }
};

// ============================================================================
// Metal Point-Cloud / NeRF Ops
// ============================================================================

class MetalPointCloudPatchifyOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        auto cpu_fallback = [&]() {
            if (auto* cpu = GetCpuFallback(OpType::PointCloudPatchify)) {
                cpu->Execute(inputs, outputs, params);
            }
        };
        if (inputs.size() < 2 || outputs.empty()) {
            cpu_fallback();
            return;
        }

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable()) {
            cpu_fallback();
            return;
        }
        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("point_cloud_patchify_forward");
        if (!pipeline) {
            cpu_fallback();
            return;
        }

        const auto* p = static_cast<const PointCloudPatchifyParams*>(params);
        PointCloudPatchifyParams default_params;
        if (!p)
            p = &default_params;

        const Tensor* points = inputs[0];
        const Tensor* features = inputs[1];
        Tensor* patches = outputs[0];
        const uint B = static_cast<uint>(points->shape[0]);
        const uint N = static_cast<uint>(points->shape[1]);
        const uint D = static_cast<uint>(features->shape[2]);
        const uint P = static_cast<uint>(p->num_patches);
        const uint patch_dim = static_cast<uint>(p->patch_dim);
        if (B == 0 || N == 0 || D == 0 || P == 0 || patch_dim == 0) {
            cpu_fallback();
            return;
        }

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), points->data, points->SizeBytes())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), features->data, features->SizeBytes())
                    offset:0
                   atIndex:1];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), patches->data, patches->SizeBytes())
                    offset:0
                   atIndex:2];
        [encoder setBytes:&B length:sizeof(uint) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint) atIndex:4];
        [encoder setBytes:&D length:sizeof(uint) atIndex:5];
        [encoder setBytes:&P length:sizeof(uint) atIndex:6];
        [encoder setBytes:&patch_dim length:sizeof(uint) atIndex:7];

        MTLSize gridSize = MTLSizeMake(patch_dim, P, B);
        MTLSize threadGroupSize = ChooseThreadgroupSize(pipeline, gridSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true, .priority = 80};
    }
};

class MetalPointCloudUnpatchifyOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        auto cpu_fallback = [&]() {
            if (auto* cpu = GetCpuFallback(OpType::PointCloudUnpatchify)) {
                cpu->Execute(inputs, outputs, params);
            }
        };
        if (inputs.size() < 2 || outputs.empty()) {
            cpu_fallback();
            return;
        }

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable()) {
            cpu_fallback();
            return;
        }
        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("point_cloud_unpatchify_forward");
        if (!pipeline) {
            cpu_fallback();
            return;
        }

        const auto* p = static_cast<const PointCloudPatchifyParams*>(params);
        PointCloudPatchifyParams default_params;
        if (!p)
            p = &default_params;

        const Tensor* patches = inputs[0];
        const Tensor* points = inputs[1];
        Tensor* features = outputs[0];
        const uint B = static_cast<uint>(points->shape[0]);
        const uint N = static_cast<uint>(points->shape[1]);
        const uint D = static_cast<uint>(features->shape[2]);
        const uint P = static_cast<uint>(p->num_patches);
        if (B == 0 || N == 0 || D == 0 || P == 0) {
            cpu_fallback();
            return;
        }

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), patches->data, patches->SizeBytes())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), points->data, points->SizeBytes())
                    offset:0
                   atIndex:1];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), features->data, features->SizeBytes())
                    offset:0
                   atIndex:2];
        [encoder setBytes:&B length:sizeof(uint) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint) atIndex:4];
        [encoder setBytes:&D length:sizeof(uint) atIndex:5];
        [encoder setBytes:&P length:sizeof(uint) atIndex:6];

        MTLSize gridSize = MTLSizeMake(D, N, B);
        MTLSize threadGroupSize = ChooseThreadgroupSize(pipeline, gridSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true, .priority = 80};
    }
};

class MetalNeRFPositionalEncodingOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        auto cpu_fallback = [&]() {
            if (auto* cpu = GetCpuFallback(OpType::NeRFPositionalEncoding)) {
                cpu->Execute(inputs, outputs, params);
            }
        };
        if (inputs.empty() || outputs.empty()) {
            cpu_fallback();
            return;
        }

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable()) {
            cpu_fallback();
            return;
        }
        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("nerf_positional_encoding_forward");
        if (!pipeline) {
            cpu_fallback();
            return;
        }

        const auto* p = static_cast<const NeRFPositionalEncodingParams*>(params);
        NeRFPositionalEncodingParams default_params;
        if (!p)
            p = &default_params;

        const Tensor* xyz = inputs[0];
        Tensor* output = outputs[0];
        const uint B = static_cast<uint>(xyz->shape[0]);
        const uint N = static_cast<uint>(xyz->shape[1]);
        const uint freq_bands = static_cast<uint>(p->num_frequencies);
        const uint include_input = p->include_input ? 1u : 0u;
        const uint out_dim = (include_input ? 3u : 0u) + 6u * freq_bands;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), xyz->data, xyz->SizeBytes())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), output->data, output->SizeBytes())
                    offset:0
                   atIndex:1];
        [encoder setBytes:&B length:sizeof(uint) atIndex:2];
        [encoder setBytes:&N length:sizeof(uint) atIndex:3];
        [encoder setBytes:&freq_bands length:sizeof(uint) atIndex:4];
        [encoder setBytes:&include_input length:sizeof(uint) atIndex:5];

        MTLSize gridSize = MTLSizeMake(out_dim, N, B);
        MTLSize threadGroupSize = ChooseThreadgroupSize(pipeline, gridSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true, .priority = 90};
    }
};

class MetalGaussianFourierFeaturesOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        auto cpu_fallback = [&]() {
            if (auto* cpu = GetCpuFallback(OpType::GaussianFourierFeatures)) {
                cpu->Execute(inputs, outputs, params);
            }
        };
        if (inputs.size() < 2 || outputs.empty()) {
            cpu_fallback();
            return;
        }

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable()) {
            cpu_fallback();
            return;
        }
        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("gaussian_fourier_features_forward");
        if (!pipeline) {
            cpu_fallback();
            return;
        }

        const Tensor* xyz = inputs[0];
        const Tensor* B_matrix = inputs[1];
        Tensor* output = outputs[0];
        const uint B = static_cast<uint>(xyz->shape[0]);
        const uint N = static_cast<uint>(xyz->shape[1]);
        const uint num_features = static_cast<uint>(output->shape[2]);
        if ((num_features % 2u) != 0u) {
            cpu_fallback();
            return;
        }

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), xyz->data, xyz->SizeBytes())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), B_matrix->data, B_matrix->SizeBytes())
                    offset:0
                   atIndex:1];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), output->data, output->SizeBytes())
                    offset:0
                   atIndex:2];
        [encoder setBytes:&B length:sizeof(uint) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint) atIndex:4];
        [encoder setBytes:&num_features length:sizeof(uint) atIndex:5];

        MTLSize gridSize = MTLSizeMake(num_features, N, B);
        MTLSize threadGroupSize = ChooseThreadgroupSize(pipeline, gridSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true, .priority = 90};
    }
};

class MetalFarthestPointSamplingOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        auto cpu_fallback = [&]() {
            if (auto* cpu = GetCpuFallback(OpType::FarthestPointSampling)) {
                cpu->Execute(inputs, outputs, params);
            }
        };
        if (inputs.empty() || outputs.empty()) {
            cpu_fallback();
            return;
        }

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable()) {
            cpu_fallback();
            return;
        }
        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("farthest_point_sampling_forward");
        if (!pipeline) {
            cpu_fallback();
            return;
        }

        const auto* p = static_cast<const FarthestPointSamplingParams*>(params);
        FarthestPointSamplingParams default_params;
        if (!p)
            p = &default_params;

        const Tensor* points = inputs[0];
        Tensor* indices = outputs[0];
        const uint B = static_cast<uint>(points->shape[0]);
        const uint N = static_cast<uint>(points->shape[1]);
        const uint samples = static_cast<uint>(p->num_samples);
        if (B == 0 || N == 0 || samples == 0) {
            cpu_fallback();
            return;
        }

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), points->data, points->SizeBytes())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), indices->data, indices->SizeBytes())
                    offset:0
                   atIndex:1];
        [encoder setBytes:&B length:sizeof(uint) atIndex:2];
        [encoder setBytes:&N length:sizeof(uint) atIndex:3];
        [encoder setBytes:&samples length:sizeof(uint) atIndex:4];

        MTLSize gridSize = MTLSizeMake(B, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false, .priority = 70};
    }
};

class MetalKNNQueryOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        auto cpu_fallback = [&]() {
            if (auto* cpu = GetCpuFallback(OpType::KNNQuery)) {
                cpu->Execute(inputs, outputs, params);
            }
        };
        if (inputs.size() < 2 || outputs.size() < 2) {
            cpu_fallback();
            return;
        }

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable()) {
            cpu_fallback();
            return;
        }
        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("knn_query_forward");
        if (!pipeline) {
            cpu_fallback();
            return;
        }

        const auto* p = static_cast<const KNNQueryParams*>(params);
        KNNQueryParams default_params;
        if (!p)
            p = &default_params;
        const uint k = static_cast<uint>(p->k);
        if (k == 0 || k > 64u) {
            cpu_fallback();
            return;
        }

        const Tensor* query = inputs[0];
        const Tensor* ref = inputs[1];
        Tensor* indices = outputs[0];
        Tensor* distances = outputs[1];
        const uint B = static_cast<uint>(query->shape[0]);
        const uint M = static_cast<uint>(query->shape[1]);
        const uint N = static_cast<uint>(ref->shape[1]);

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), query->data, query->SizeBytes())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), ref->data, ref->SizeBytes())
                    offset:0
                   atIndex:1];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), indices->data, indices->SizeBytes())
                    offset:0
                   atIndex:2];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), distances->data, distances->SizeBytes())
                    offset:0
                   atIndex:3];
        [encoder setBytes:&B length:sizeof(uint) atIndex:4];
        [encoder setBytes:&M length:sizeof(uint) atIndex:5];
        [encoder setBytes:&N length:sizeof(uint) atIndex:6];
        [encoder setBytes:&k length:sizeof(uint) atIndex:7];

        MTLSize gridSize = MTLSizeMake(M, 1, B);
        MTLSize threadGroupSize = ChooseThreadgroupSize(pipeline, gridSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false, .priority = 70};
    }
};

class MetalBallQueryOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        auto cpu_fallback = [&]() {
            if (auto* cpu = GetCpuFallback(OpType::BallQuery)) {
                cpu->Execute(inputs, outputs, params);
            }
        };
        if (inputs.size() < 2 || outputs.empty()) {
            cpu_fallback();
            return;
        }

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable()) {
            cpu_fallback();
            return;
        }
        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("ball_query_forward");
        if (!pipeline) {
            cpu_fallback();
            return;
        }

        const auto* p = static_cast<const BallQueryParams*>(params);
        BallQueryParams default_params;
        if (!p)
            p = &default_params;

        const Tensor* query = inputs[0];
        const Tensor* ref = inputs[1];
        Tensor* indices = outputs[0];
        const uint B = static_cast<uint>(query->shape[0]);
        const uint M = static_cast<uint>(query->shape[1]);
        const uint N = static_cast<uint>(ref->shape[1]);
        const float radius = p->radius;
        const uint max_samples = static_cast<uint>(p->max_samples);
        if (max_samples == 0) {
            cpu_fallback();
            return;
        }

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), query->data, query->SizeBytes())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), ref->data, ref->SizeBytes())
                    offset:0
                   atIndex:1];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), indices->data, indices->SizeBytes())
                    offset:0
                   atIndex:2];
        [encoder setBytes:&B length:sizeof(uint) atIndex:3];
        [encoder setBytes:&M length:sizeof(uint) atIndex:4];
        [encoder setBytes:&N length:sizeof(uint) atIndex:5];
        [encoder setBytes:&radius length:sizeof(float) atIndex:6];
        [encoder setBytes:&max_samples length:sizeof(uint) atIndex:7];

        MTLSize gridSize = MTLSizeMake(M, 1, B);
        MTLSize threadGroupSize = ChooseThreadgroupSize(pipeline, gridSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false, .priority = 70};
    }
};

class MetalDeformableAttentionOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        auto cpu_fallback = [&]() {
            if (auto* cpu = GetCpuFallback(OpType::DeformableAttention)) {
                cpu->Execute(inputs, outputs, params);
            }
        };
        if (inputs.size() < 6 || outputs.empty()) {
            cpu_fallback();
            return;
        }

        auto& ctx = MetalContext::Instance();
        if (!ctx.IsAvailable()) {
            cpu_fallback();
            return;
        }
        id<MTLComputePipelineState> pipeline = ctx.GetPipeline("deformable_attention_3d_forward");
        if (!pipeline) {
            cpu_fallback();
            return;
        }

        const auto* p = static_cast<const DeformableAttentionParams*>(params);
        DeformableAttentionParams default_params;
        if (!p)
            p = &default_params;

        const Tensor* query = inputs[0];
        const Tensor* key = inputs[1];
        const Tensor* value = inputs[2];
        const Tensor* reference = inputs[3];
        const Tensor* offsets = inputs[4];
        const Tensor* weights = inputs[5];
        const Tensor* key_points = inputs.size() >= 7 ? inputs[6] : nullptr;
        Tensor* output = outputs[0];

        const uint B = static_cast<uint>(query->shape[0]);
        const uint Q = static_cast<uint>(query->shape[1]);
        const uint N = static_cast<uint>(value->shape[1]);
        const uint D = static_cast<uint>(query->shape[2]);
        const uint H = static_cast<uint>(std::max(1, p->num_heads));
        const uint total_samples = static_cast<uint>(std::max(1, p->num_points));
        const uint offset_heads = offsets->ndim >= 4 ? static_cast<uint>(offsets->shape[2]) : 1u;
        const uint weight_heads = weights->ndim >= 4 ? static_cast<uint>(weights->shape[2]) : 1u;
        const uint use_key_points =
            (key_points && key_points->IsValid() && key_points->ndim == 3) ? 1u : 0u;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), query->data, query->SizeBytes())
                    offset:0
                   atIndex:0];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), key->data, key->SizeBytes())
                    offset:0
                   atIndex:1];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), value->data, value->SizeBytes())
                    offset:0
                   atIndex:2];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), reference->data, reference->SizeBytes())
                    offset:0
                   atIndex:3];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), offsets->data, offsets->SizeBytes())
                    offset:0
                   atIndex:4];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), weights->data, weights->SizeBytes())
                    offset:0
                   atIndex:5];
        [encoder
            setBuffer:WrapSharedBuffer(ctx.device(), use_key_points ? key_points->data : key->data,
                                       use_key_points ? key_points->SizeBytes() : key->SizeBytes())
               offset:0
              atIndex:6];
        [encoder setBuffer:WrapSharedBuffer(ctx.device(), output->data, output->SizeBytes())
                    offset:0
                   atIndex:7];
        [encoder setBytes:&B length:sizeof(uint) atIndex:8];
        [encoder setBytes:&Q length:sizeof(uint) atIndex:9];
        [encoder setBytes:&N length:sizeof(uint) atIndex:10];
        [encoder setBytes:&D length:sizeof(uint) atIndex:11];
        [encoder setBytes:&H length:sizeof(uint) atIndex:12];
        [encoder setBytes:&total_samples length:sizeof(uint) atIndex:13];
        [encoder setBytes:&offset_heads length:sizeof(uint) atIndex:14];
        [encoder setBytes:&weight_heads length:sizeof(uint) atIndex:15];
        [encoder setBytes:&use_key_points length:sizeof(uint) atIndex:16];

        MTLSize gridSize = MTLSizeMake(D, Q, B);
        MTLSize threadGroupSize = ChooseThreadgroupSize(pipeline, gridSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    bool Supports(DeviceType device) const override {
        return device == DeviceType::METAL && MetalContext::Instance().IsAvailable();
    }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true, .priority = 85};
    }
};

using NpuTemporalAttentionOp = AppleNpuForwardOp<OpType::TemporalAttention>;
using NpuPatchEmbed3DOp = AppleNpuForwardOp<OpType::PatchEmbed3D>;
using NpuPatchifyOp = AppleNpuForwardOp<OpType::Patchify>;
using NpuUnpatchifyOp = AppleNpuForwardOp<OpType::Unpatchify>;
using NpuPointCloudPatchifyOp = AppleNpuForwardOp<OpType::PointCloudPatchify>;
using NpuPointCloudUnpatchifyOp = AppleNpuForwardOp<OpType::PointCloudUnpatchify>;
using NpuDeformableAttentionOp = AppleNpuForwardOp<OpType::DeformableAttention>;
using NpuFarthestPointSamplingOp = AppleNpuForwardOp<OpType::FarthestPointSampling>;
using NpuKNNQueryOp = AppleNpuForwardOp<OpType::KNNQuery>;
using NpuBallQueryOp = AppleNpuForwardOp<OpType::BallQuery>;
using NpuNeRFPositionalEncodingOp = AppleNpuForwardOp<OpType::NeRFPositionalEncoding>;
using NpuGaussianFourierFeaturesOp = AppleNpuForwardOp<OpType::GaussianFourierFeatures>;
using NpuGridSampleOp = AppleNpuForwardOp<OpType::GridSample>;

// ============================================================================
// Op Registration
// ============================================================================

DENSECORE_REGISTER_OP(MetalWindowAttentionOp, OpType::WindowAttention, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalTemporalAttentionOp, OpType::TemporalAttention, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalTriangularAttentionOp, OpType::TriangularAttention, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalPatchEmbed3DOp, OpType::PatchEmbed3D, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalPatchifyOp, OpType::Patchify, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalUnpatchifyOp, OpType::Unpatchify, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalPointCloudPatchifyOp, OpType::PointCloudPatchify, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalPointCloudUnpatchifyOp, OpType::PointCloudUnpatchify, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalDeformableAttentionOp, OpType::DeformableAttention, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalFarthestPointSamplingOp, OpType::FarthestPointSampling,
                      DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalKNNQueryOp, OpType::KNNQuery, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalBallQueryOp, OpType::BallQuery, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalNeRFPositionalEncodingOp, OpType::NeRFPositionalEncoding,
                      DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalGaussianFourierFeaturesOp, OpType::GaussianFourierFeatures,
                      DeviceType::METAL);

DENSECORE_REGISTER_OP(NpuTemporalAttentionOp, OpType::TemporalAttention, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuPatchEmbed3DOp, OpType::PatchEmbed3D, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuPatchifyOp, OpType::Patchify, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuUnpatchifyOp, OpType::Unpatchify, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuPointCloudPatchifyOp, OpType::PointCloudPatchify, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuPointCloudUnpatchifyOp, OpType::PointCloudUnpatchify, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuDeformableAttentionOp, OpType::DeformableAttention, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuFarthestPointSamplingOp, OpType::FarthestPointSampling, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuKNNQueryOp, OpType::KNNQuery, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuBallQueryOp, OpType::BallQuery, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuNeRFPositionalEncodingOp, OpType::NeRFPositionalEncoding, DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuGaussianFourierFeaturesOp, OpType::GaussianFourierFeatures,
                      DeviceType::NPU);
DENSECORE_REGISTER_OP(NpuGridSampleOp, OpType::GridSample, DeviceType::NPU);

}  // namespace
}  // namespace kernels
}  // namespace densecore

#endif  // __APPLE__
