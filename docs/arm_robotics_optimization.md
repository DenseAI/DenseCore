# ARM Robotics Optimization Guide

This guide documents DenseCore changes for low-power ARM robotics inference
(Jetson Orin Nano, Raspberry Pi 5, Qualcomm RB5 class devices).

## 1) Build Profiles (Tier 0)

DenseCore now supports `DENSEVLA_ARM_TARGET`:

- `generic` -> `armv8.2-a+dotprod`
- `jetson_orin` -> `armv8.7-a+sve2+i8mm+bf16+dotprod`
- `rpi5` -> `armv8.2-a+dotprod+fp16`
- `qualcomm_rb5` -> `armv8.2-a+dotprod+crypto`
- `custom` -> provide `DENSECORE_ARM_MARCH` / `GGML_CPU_ARM_ARCH` manually

These profiles also default GGML options for ARM:

- `GGML_NATIVE=OFF` when cross-compiling
- `GGML_LLAMAFILE=ON`
- `GGML_CPU_KLEIDIAI=ON`

### Example: CMake Cross Build

```bash
cmake -S core -B build-arm \
  -DCMAKE_TOOLCHAIN_FILE=core/cmake/aarch64-toolchain.cmake \
  -DDENSEVLA_ARM_TARGET=jetson_orin \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-arm -j
```

## 2) Action-Token Partial Sampling (Tier 1-3)

DenseCore sampler supports vocab subrange decoding for action heads:

- `DENSECORE_ACTION_TOKEN_START` (default `0`)
- `DENSECORE_ACTION_TOKEN_COUNT` (default `0`, disabled)

When enabled, sampling runs only in `[start, start + count)`, reducing
softmax/sampling overhead for robotics action-token heads.

## 3) Fused Vision Preprocess + Patchify (Tier 1-2)

`densecore::ops::ImageOps` now provides:

- `RGBToPatchTokensFused(...)`

This is a single-pass kernel:

`RGB uint8 NHWC -> normalize -> patch tokens [N, 3*P*P]`

It avoids intermediate CHW buffers and is optimized for ARM memory bandwidth.

## 4) Action Chunking Utility (Tier 2-1)

`densecore::robotics::ActionChunker` is available for low-frequency model
inference + high-frequency control-loop execution:

- resampling/interpolation to control rate
- bounded pending queue
- thread-safe producer/consumer usage

Header:

- `core/include/densecore/action_chunker.h`

## 5) ARM Runtime Controls (Tier 3)

`densecore::arm_runtime` utilities are available for:

- big/LITTLE core detection and pinning
- temporary CPU governor switching via RAII (`GovernorGuard`)

Header:

- `core/include/densecore/arm_runtime.h`
