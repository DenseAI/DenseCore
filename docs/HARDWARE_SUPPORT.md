# Hardware Support & Backends

DenseCore is built as a **heterogeneous inference runtime** — not a GPU-fallback or CPU-only engine. The Hardware Abstraction Layer (HAL) enables runtime kernel selection across x86, ARM64, and Apple Silicon, so the same binary adapts to the available hardware without recompilation.

This is the foundation of DenseCore's role as the **runtime substrate** of the Dense Series stack: one execution engine for edge nodes, ARM servers, cloud CPU instances, Apple Silicon workstations, and Jetson devices.

## Supported Architectures

### Apple Silicon
- **Metal GPU**: SIMD-group optimized compute shaders, FlashAttention prefill kernel
- **ANE (Neural Engine)**: CoreML-backed MatMul with dynamic bucketing (1-32K tokens)
- **Accelerate AMX**: Apple's matrix coprocessor via BLAS (M1-M4)
- **NEON FP16**: Native `vfmaq_f16` for 2x throughput on M3/M4

### x86 (Intel/AMD)
- **AVX-512**: 512-bit vectors with 8-way unrolling (Skylake-X+)
- **AVX-512 VNNI**: `vpdpbusd` INT8 dot products (Ice Lake+, Zen4+)
- **AMX**: BF16 tile matrix operations (Sapphire Rapids+)
- **AVX2 + FMA**: 256-bit vectors (Haswell+, Zen+)
- **AVX**: 256-bit vectors (Sandy Bridge+)
- **SSE4.1**: 128-bit vectors (Penryn+)

### ARM64 (AWS Graviton, Qualcomm)
- **SVE (256-bit+)**: Scalable vectors with `svdot_s32` (Graviton 3/4)
- **NEON DOTPROD**: `vdotq_s32` INT8 dot products (Graviton 2+)
- **NEON FP16**: `vfmaq_f16` for 2x throughput (Graviton 3+)
- **NEON**: Fixed 128-bit vectors (all ARM64)

## Runtime Detection
DenseCore automatically detects and selects the optimal kernel at runtime:
```
Intel Xeon (Sapphire Rapids) → AMX > AVX-512 VNNI > AVX-512
Apple M3 Max                 → Metal GPU + ANE + NEON FP16
AWS Graviton3                → SVE DotProd > NEON DOTPROD > NEON
AMD Zen4                     → AVX-512 VNNI > AVX2
```

## Supported Backends Status

| Backend | Target Devices | Status |
|---------|----------------|--------|
| **CPU** | All (x86, ARM64) | ✅ Production |
| **Metal** | Apple Silicon GPU | ✅ Production |
| **ANE** | Apple Neural Engine (M1-M4) | ✅ Production (Conditional) |
| **Accelerate** | Apple AMX (via BLAS) | ✅ Production |
| **Hybrid Scheduler** | Apple Silicon (CPU+GPU+ANE) | ✅ Production (Alpha) |
