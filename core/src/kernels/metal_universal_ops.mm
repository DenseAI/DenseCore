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
        NSString* libraryPath = [bundle pathForResource:@"universal_ops" ofType:@"metallib"];
        if (libraryPath) {
            library_ = [device_ newLibraryWithFile:libraryPath error:&error];
        }

        // Fallback: compile shaders at runtime
        if (!library_) {
            NSURL* shaderURL = [[NSURL fileURLWithPath:[[NSBundle mainBundle] bundlePath]]
                URLByAppendingPathComponent:@"../shaders"];

            // Try to load individual shader files
            NSArray<NSString*>* shaderFiles = @[
                @"window_attention.metal", @"temporal_attention.metal",
                @"triangular_attention.metal", @"patch_embed_3d.metal"
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
// Op Registration
// ============================================================================

DENSECORE_REGISTER_OP(MetalWindowAttentionOp, OpType::WindowAttention, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalTemporalAttentionOp, OpType::TemporalAttention, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalTriangularAttentionOp, OpType::TriangularAttention, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalPatchEmbed3DOp, OpType::PatchEmbed3D, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalPatchifyOp, OpType::Patchify, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalUnpatchifyOp, OpType::Unpatchify, DeviceType::METAL);

}  // namespace
}  // namespace kernels
}  // namespace densecore

#endif  // __APPLE__
