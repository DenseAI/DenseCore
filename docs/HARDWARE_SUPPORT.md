# Hardware Support & Backends

DenseCore uses a Hardware Abstraction Layer (HAL) to support CPU-first inference across multiple hardware families. The design goal is not to force every workload onto a single fast path, but to keep the CPU baseline explicit while exposing backend-specific acceleration where it exists and where fallback remains correct.

## Core Principles

- **CPU-first baseline**: x86-64 and ARM64 CPU paths are the primary execution baseline.
- **Heterogeneous extensions**: Apple-specific Metal, ANE, and Accelerate paths can extend the baseline on supported hardware.
- **Graceful fallback**: unsupported or unavailable accelerators should fall back to CPU execution rather than turning into hard failure.
- **Explicit maturity**: backend status should be read as hardware-specific capability, not as a blanket claim for all deployment targets.

## Supported Architectures

### Apple Silicon
- **CPU baseline**: NEON FP16 and standard CPU execution remain the default baseline.
- **Metal GPU**: backend-specific compute kernels, including Metal paths for attention and matrix operations on supported Apple Silicon systems.
- **ANE (Neural Engine)**: CoreML-backed execution paths exist for selected operations and Apple-only workflows.
- **Accelerate**: BLAS-backed CPU-side acceleration is available where the platform provides it.
- **Unified Memory (UMA)**: Apple Silicon allows shared-memory integration between CPU and accelerator paths, which reduces transfer overhead on supported backends.

### x86-64 (Intel/AMD)
- **AVX-512**: 512-bit vectors with 8-way unrolling (Skylake-X+)
- **AVX-512 VNNI**: `vpdpbusd` INT8 dot products (Ice Lake+, Zen4+)
- **AMX**: BF16 tile matrix operations (Sapphire Rapids+)
- **AVX2 + FMA**: 256-bit vectors (Haswell+, Zen+)
- **AVX**: 256-bit vectors (Sandy Bridge+)
- **SSE4.1**: 128-bit vectors (Penryn+)

### ARM64 (AWS Graviton, Ampere, Qualcomm, Apple CPU path)
- **SVE (256-bit+)**: Scalable vectors with `svdot_s32` (Graviton 3/4)
- **NEON DOTPROD**: `vdotq_s32` INT8 dot products (Graviton 2+)
- **NEON FP16**: `vfmaq_f16` for 2x throughput (Graviton 3+)
- **NEON**: Fixed 128-bit vectors (all ARM64)
- **Highway dispatch**: portable SIMD kernels are used across ISA variants for selected operations such as RoPE, activation, and paged-attention related paths.

## Runtime Detection

DenseCore selects the best available path at runtime while preserving fallback behavior:

```
Intel Xeon (Sapphire Rapids) → AMX > AVX-512 VNNI > AVX-512
Apple M3 Max                 → Metal GPU + ANE + NEON FP16
AWS Graviton3                → SVE DotProd > NEON DOTPROD > NEON
AMD Zen4                     → AVX-512 VNNI > AVX2
```

## Software-Hardware Co-Optimization

DenseCore also includes runtime behavior that is hardware-aware even when execution stays on CPU.

### Paged KV Cache and Block Management

- **Paged KV cache**: keeps long-context memory growth explicit and avoids a single monolithic KV allocation strategy.
- **NUMA-aware allocation**: multi-socket systems can place memory with NUMA locality in mind rather than assuming uniform access costs.
- **Copy-on-Write support**: shared blocks can be duplicated only when a write is required, which helps prefix sharing and parallel generation workflows.

These mechanisms are part of the core runtime story, not optional accelerator features.

## Backend Status Matrix

| Backend | Capability | Status | Target Hardware |
|---------|------------|--------|-----------------|
| **CPU SIMD (x86-64)** | Full-model baseline execution | ✅ Production-ready baseline | Intel / AMD x86-64 |
| **CPU SIMD (ARM64)** | Full-model baseline execution | ✅ Production-ready baseline | Graviton, Ampere, Qualcomm, Apple CPU path |
| **Metal GPU** | Apple-specific accelerated kernels | 🟡 Beta / backend-specific | Apple Silicon |
| **ANE** | Selected CoreML-backed execution paths | 🟠 Experimental / conditional | Apple M1-M4 |
| **Accelerate** | BLAS-backed matrix acceleration | 🟡 Beta / conditional | Apple platforms with Accelerate |
| **Hybrid Scheduler** | Cross-device execution planning | 🟠 Experimental | Apple Silicon |
