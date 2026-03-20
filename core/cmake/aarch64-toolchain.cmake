# DenseCore ARM64 cross-compilation toolchain profile.
#
# Usage:
#   cmake -S core -B build-aarch64 \
#     -DCMAKE_TOOLCHAIN_FILE=core/cmake/aarch64-toolchain.cmake \
#     -DDENSECORE_ARM_TARGET=jetson_orin
#
# Supported DENSECORE_ARM_TARGET / DENSEVLA_ARM_TARGET values:
#   - generic
#   - jetson_orin
#   - rpi5
#   - qualcomm_rb5
#   - custom (set DENSECORE_ARM_MARCH manually)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Allow caller to override compilers, but provide sane defaults.
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/arm_profiles.cmake")

DenseCoreInitArmTargetProfiles()
DenseCoreResolveArmProfileMarch(DENSECORE_ARM_PROFILE_MARCH)

# Toolchain-level ISA activation:
# Ensure profile-derived -march is injected into initial C/CXX flags so
# standalone cross-compiles also get the intended ARM ISA extensions.
set(_densecore_effective_march "${DENSECORE_ARM_PROFILE_MARCH}")
if(DENSECORE_ARM_TARGET STREQUAL "custom" AND DENSECORE_ARM_MARCH)
    set(_densecore_effective_march "${DENSECORE_ARM_MARCH}")
endif()
if(_densecore_effective_march)
    set(_densecore_march_flag "-march=${_densecore_effective_march}")
    if(NOT CMAKE_C_FLAGS_INIT MATCHES "(^| )-march=")
        string(STRIP "${CMAKE_C_FLAGS_INIT} ${_densecore_march_flag}" CMAKE_C_FLAGS_INIT)
    endif()
    if(NOT CMAKE_CXX_FLAGS_INIT MATCHES "(^| )-march=")
        string(STRIP "${CMAKE_CXX_FLAGS_INIT} ${_densecore_march_flag}" CMAKE_CXX_FLAGS_INIT)
    endif()
endif()

# Keep profile-based targets deterministic even across CMake re-configures.
if(DENSECORE_ARM_TARGET STREQUAL "custom")
    if(NOT DEFINED DENSECORE_ARM_MARCH AND DENSECORE_ARM_PROFILE_MARCH)
        set(DENSECORE_ARM_MARCH "${DENSECORE_ARM_PROFILE_MARCH}" CACHE STRING
            "Explicit ARM architecture for -march" FORCE)
    endif()
elseif(DENSECORE_ARM_PROFILE_MARCH)
    set(DENSECORE_ARM_MARCH "${DENSECORE_ARM_PROFILE_MARCH}" CACHE STRING
        "Explicit ARM architecture for -march" FORCE)
endif()

if(DENSECORE_ARM_TARGET STREQUAL "custom")
    if(NOT DEFINED GGML_CPU_ARM_ARCH AND DENSECORE_ARM_PROFILE_GGML_ARCH)
        set(GGML_CPU_ARM_ARCH "${DENSECORE_ARM_PROFILE_GGML_ARCH}" CACHE STRING
            "ggml: CPU architecture for ARM" FORCE)
    endif()
elseif(DENSECORE_ARM_PROFILE_GGML_ARCH)
    set(GGML_CPU_ARM_ARCH "${DENSECORE_ARM_PROFILE_GGML_ARCH}" CACHE STRING
        "ggml: CPU architecture for ARM" FORCE)
endif()

# Cross-compilation must not use host-native CPU flags.
if(NOT DEFINED GGML_NATIVE)
    set(GGML_NATIVE OFF CACHE BOOL "Optimize for native CPU" FORCE)
endif()

if(NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE ON CACHE BOOL "Enable ggml llamafile kernels" FORCE)
endif()
if(NOT DEFINED GGML_CPU_KLEIDIAI)
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "Enable ggml KleidiAI kernels" FORCE)
endif()

message(STATUS "[DenseCore][Toolchain] ARM target profile: ${DENSECORE_ARM_TARGET}")
if(DENSECORE_ARM_MARCH)
    message(STATUS "[DenseCore][Toolchain] DENSECORE_ARM_MARCH: ${DENSECORE_ARM_MARCH}")
endif()
if(GGML_CPU_ARM_ARCH)
    message(STATUS "[DenseCore][Toolchain] GGML_CPU_ARM_ARCH: ${GGML_CPU_ARM_ARCH}")
endif()
if(CMAKE_C_FLAGS_INIT)
    message(STATUS "[DenseCore][Toolchain] CMAKE_C_FLAGS_INIT: ${CMAKE_C_FLAGS_INIT}")
endif()
if(CMAKE_CXX_FLAGS_INIT)
    message(STATUS "[DenseCore][Toolchain] CMAKE_CXX_FLAGS_INIT: ${CMAKE_CXX_FLAGS_INIT}")
endif()
