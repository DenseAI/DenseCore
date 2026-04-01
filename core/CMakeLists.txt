cmake_minimum_required(VERSION 3.14)

# ==========================================================================
# CMake 4.x + Ninja + OBJCXX Bug Workaround
# ==========================================================================
# CMake 4.x has a bug where CMAKE_OBJCXX_COMPILE_OBJECT is not set when
# enable_language(OBJCXX) is called after project() with Ninja generator.
# Workaround: Enable OBJCXX at project declaration time on Apple platforms.
# ==========================================================================
if(APPLE)
    project(DenseCore VERSION 1.0.0 LANGUAGES C CXX OBJC OBJCXX)
else()
    project(DenseCore VERSION 1.0.0 LANGUAGES C CXX)
endif()

# ==========================================================================
# Commercial / Open Source Toggle
# ==========================================================================
option(DENSECORE_COMMERCIAL "Enable commercial features (Graviton/Enterprise)" OFF)

set(DENSECORE_LINUX_ARM64 OFF)
if(NOT APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(DENSECORE_LINUX_ARM64 ON)
endif()

set(DENSECORE_ARM_CORRECTNESS_FIRST_DEFAULT OFF)
option(
    DENSECORE_ARM_CORRECTNESS_FIRST
    "Prefer correctness-first ggml defaults on Linux ARM (disables KleidiAI, llamafile, repack, and native ISA auto-tuning unless explicitly overridden)"
    ${DENSECORE_ARM_CORRECTNESS_FIRST_DEFAULT}
)

if(DENSECORE_COMMERCIAL)
    message(STATUS "[DenseCore] Commercial build ENABLED")
    add_compile_definitions(DENSECORE_COMMERCIAL)

    # Check for proprietary source directory
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/proprietary/CMakeLists.txt")
        add_subdirectory(src/proprietary)
    else()
        message(WARNING "[DenseCore] Commercial flag ON but src/proprietary not found or empty.")
    endif()
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==========================================================================
# Profile-Guided Optimization (PGO) + Link-Time Optimization (LTO)
# ==========================================================================
# PGO: Build optimized binaries using runtime profiling data.
#      Step 1: cmake -DDENSECORE_PGO_GENERATE=ON .. && make && ./benchmark
#      Step 2: cmake -DDENSECORE_PGO_USE=ON .. && make
#      The resulting binary runs 5-10% faster on the profiled workloads.
#
# LTO: Enable cross-module optimization for 2-5% performance boost.
#      Safe to enable for all release builds.
#
# These are used by our CI/CD to build optimized PyPI wheels, Docker images,
# and Helm charts. End users get pre-optimized binaries without compiling.
# ==========================================================================
option(DENSECORE_PGO_GENERATE "Build with PGO instrumentation (Step 1: collect profile)" OFF)
option(DENSECORE_PGO_USE "Build with PGO optimization (Step 2: use collected profile)" OFF)
option(DENSECORE_LTO "Enable Link-Time Optimization for release builds" OFF)

# PGO profile data directory (default: build directory)
set(DENSECORE_PGO_DIR "${CMAKE_BINARY_DIR}/pgo_data" CACHE PATH "Directory for PGO profile data")

if(DENSECORE_PGO_GENERATE AND DENSECORE_PGO_USE)
    message(FATAL_ERROR "Cannot enable both PGO_GENERATE and PGO_USE. Use one at a time.")
endif()

if(NOT MSVC)
    # GCC/Clang PGO flags
    if(DENSECORE_PGO_GENERATE)
        message(STATUS "[PGO] Step 1: Building instrumented binary for profile collection")
        message(STATUS "[PGO] After build, run representative workloads to generate profile data")
        message(STATUS "[PGO] Profile data will be written to: ${DENSECORE_PGO_DIR}")
        
        file(MAKE_DIRECTORY ${DENSECORE_PGO_DIR})
        
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang uses -fprofile-instr-generate
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate=${DENSECORE_PGO_DIR}/default.profraw")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-instr-generate=${DENSECORE_PGO_DIR}/default.profraw")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-instr-generate")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fprofile-instr-generate")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            # GCC uses -fprofile-generate
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate=${DENSECORE_PGO_DIR}")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-generate=${DENSECORE_PGO_DIR}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-generate=${DENSECORE_PGO_DIR}")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fprofile-generate=${DENSECORE_PGO_DIR}")
        endif()
    endif()

    if(DENSECORE_PGO_USE)
        message(STATUS "[PGO] Step 2: Building optimized binary using profile data")
        message(STATUS "[PGO] Using profile data from: ${DENSECORE_PGO_DIR}")
        
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang: merge .profraw files first, then use .profdata
            # User must run: llvm-profdata merge -output=${DENSECORE_PGO_DIR}/default.profdata ${DENSECORE_PGO_DIR}/*.profraw
            set(PGO_PROFDATA "${DENSECORE_PGO_DIR}/default.profdata")
            if(NOT EXISTS ${PGO_PROFDATA})
                message(WARNING "[PGO] Profile data not found at ${PGO_PROFDATA}")
                message(WARNING "[PGO] Run: llvm-profdata merge -output=${PGO_PROFDATA} ${DENSECORE_PGO_DIR}/*.profraw")
            endif()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-use=${PGO_PROFDATA}")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-instr-use=${PGO_PROFDATA}")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            # GCC: uses .gcda files directly
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use=${DENSECORE_PGO_DIR} -fprofile-correction")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-use=${DENSECORE_PGO_DIR} -fprofile-correction")
        endif()
    endif()

    # Link-Time Optimization (LTO)
    if(DENSECORE_LTO)
        message(STATUS "[LTO] Enabling Link-Time Optimization")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # ThinLTO: faster linking, better for iterative development
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=thin")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=thin")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=thin")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto=thin")
            message(STATUS "[LTO] Using Clang ThinLTO for faster linking")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            # Full LTO for GCC
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto")
            message(STATUS "[LTO] Using GCC full LTO")
        endif()
    endif()
endif()

# ==========================================================================
# Symbol Visibility Control
# ==========================================================================
# Hide all symbols by default, only expose functions marked with DENSECORE_API
# This reduces .so size and improves load time
# ==========================================================================
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_C_VISIBILITY_PRESET hidden)

# ==========================================================================
# Version Information Injection
# ==========================================================================
# Generate version header from CMake project version
# ==========================================================================
set(DENSECORE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(DENSECORE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(DENSECORE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(DENSECORE_VERSION_STRING "${PROJECT_VERSION}")

# Get build timestamp
string(TIMESTAMP DENSECORE_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)

# Try to get git commit hash
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE DENSECORE_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT DENSECORE_GIT_COMMIT)
        set(DENSECORE_GIT_COMMIT "unknown")
    endif()
else()
    set(DENSECORE_GIT_COMMIT "unknown")
endif()

# Generate version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/densecore_version.h
    @ONLY
)

# Output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ggml third party library path
set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)

# Check if ggml exists
if(NOT EXISTS ${GGML_DIR})
    message(FATAL_ERROR "ggml not found at ${GGML_DIR}. Please run scripts/setup_ggml.sh first.")
endif()

# Shared ARM target profile definitions.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/arm_profiles.cmake)

# ==========================================================================
# ARM target profiles for robotics devices (DenseVLA)
# ==========================================================================
# These profiles provide safe defaults for cross-compilation and allow
# overriding from CLI via -DDENSECORE_ARM_MARCH=... or -DGGML_CPU_ARM_ARCH=...
DenseCoreInitArmTargetProfiles()

set(DENSECORE_ARM_PROFILE_MARCH "")
if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    DenseCoreResolveArmProfileMarch(DENSECORE_ARM_PROFILE_MARCH)
endif()

set(DENSECORE_ARM_CORRECTNESS_FIRST_ACTIVE OFF)
if(DENSECORE_LINUX_ARM64 AND DENSECORE_ARM_CORRECTNESS_FIRST)
    set(DENSECORE_ARM_CORRECTNESS_FIRST_ACTIVE ON)
    message(STATUS "[DenseCore] ARM correctness-first defaults enabled")
endif()

# Configure ggml options before adding subdirectory
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Note: Do NOT use FORCE for these options - allow command-line override
if(NOT DEFINED GGML_NATIVE)
    if(DENSECORE_ARM_CORRECTNESS_FIRST_ACTIVE OR DENSECORE_PORTABLE OR CMAKE_CROSSCOMPILING)
        set(GGML_NATIVE OFF CACHE BOOL "Optimize for native CPU")
    else()
        set(GGML_NATIVE ON CACHE BOOL "Optimize for native CPU")
    endif()
endif()
set(DENSECORE_GGML_CPU_FEATURES_DEFAULT ON)
if(DENSECORE_ARM_CORRECTNESS_FIRST_ACTIVE)
    set(DENSECORE_GGML_CPU_FEATURES_DEFAULT OFF)
endif()
if(NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE ${DENSECORE_GGML_CPU_FEATURES_DEFAULT} CACHE BOOL "Enable ggml llamafile kernels")
endif()
if(NOT DEFINED GGML_CPU_REPACK)
    set(GGML_CPU_REPACK ${DENSECORE_GGML_CPU_FEATURES_DEFAULT} CACHE BOOL "Enable ggml runtime repack kernels")
endif()
set(DENSECORE_CAN_USE_KLEIDIAI OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$" AND NOT DENSECORE_ARM_CORRECTNESS_FIRST_ACTIVE)
    set(DENSECORE_CAN_USE_KLEIDIAI ON)
endif()
if(NOT DEFINED GGML_CPU_KLEIDIAI)
    set(GGML_CPU_KLEIDIAI ${DENSECORE_CAN_USE_KLEIDIAI} CACHE BOOL "Enable ggml KleidiAI kernels")
elseif(GGML_CPU_KLEIDIAI AND NOT DENSECORE_CAN_USE_KLEIDIAI)
    message(STATUS "[DenseCore] Disabling GGML_CPU_KLEIDIAI for non-ARM target: ${CMAKE_SYSTEM_PROCESSOR}")
    set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "Enable ggml KleidiAI kernels" FORCE)
endif()
if(NOT DEFINED GGML_CPU_ARM_ARCH)
    if(DENSECORE_ARM_CORRECTNESS_FIRST_ACTIVE)
        set(GGML_CPU_ARM_ARCH "armv8-a" CACHE STRING "ggml: CPU architecture for ARM")
    elseif(DENSECORE_ARM_PROFILE_GGML_ARCH AND (NOT GGML_NATIVE OR DENSECORE_PORTABLE OR CMAKE_CROSSCOMPILING))
        set(GGML_CPU_ARM_ARCH "${DENSECORE_ARM_PROFILE_GGML_ARCH}" CACHE STRING "ggml: CPU architecture for ARM")
    endif()
endif()
if(DENSECORE_PORTABLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(GGML_SSE42 OFF CACHE BOOL "ggml: enable SSE 4.2" FORCE)
    set(GGML_AVX OFF CACHE BOOL "ggml: enable AVX" FORCE)
    set(GGML_AVX2 OFF CACHE BOOL "ggml: enable AVX2" FORCE)
    set(GGML_AVX_VNNI OFF CACHE BOOL "ggml: enable AVX-VNNI" FORCE)
    set(GGML_BMI2 OFF CACHE BOOL "ggml: enable BMI2" FORCE)
    set(GGML_FMA OFF CACHE BOOL "ggml: enable FMA" FORCE)
    set(GGML_F16C OFF CACHE BOOL "ggml: enable F16C" FORCE)
    set(GGML_AVX512 OFF CACHE BOOL "ggml: enable AVX512F" FORCE)
    set(GGML_AVX512_VBMI OFF CACHE BOOL "ggml: enable AVX512-VBMI" FORCE)
    set(GGML_AVX512_VNNI OFF CACHE BOOL "ggml: enable AVX512-VNNI" FORCE)
    set(GGML_AVX512_BF16 OFF CACHE BOOL "ggml: enable AVX512-BF16" FORCE)
    set(GGML_AMX_TILE OFF CACHE BOOL "ggml: enable AMX-TILE" FORCE)
    set(GGML_AMX_INT8 OFF CACHE BOOL "ggml: enable AMX-INT8" FORCE)
    set(GGML_AMX_BF16 OFF CACHE BOOL "ggml: enable AMX-BF16" FORCE)
endif()
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# Add ggml as subdirectory
add_subdirectory(${GGML_DIR})

# ggml pulls simdjson; ensure symbols are exported from the shared lib even
# when the parent project uses hidden visibility defaults.
if(TARGET simdjson)
    set_target_properties(simdjson PROPERTIES
        C_VISIBILITY_PRESET default
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
    )
endif()

# ==========================================================================
# SIMDe Integration (for ARM64/non-x86 support)
# ==========================================================================
include(FetchContent)
FetchContent_Declare(
    simde
    GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
    GIT_TAG v0.8.2
)
FetchContent_MakeAvailable(simde)

# ==========================================================================
# Google Highway (portable SIMD — single source, all ISAs)
# ==========================================================================
FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.2.0
)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)


# ==========================================================================
# Microsoft Mimalloc (Performance Allocator)
# ==========================================================================
# Override standard malloc/free with mimalloc for performance.
# Recommended for rigorous, high-throughput production environments.
# ==========================================================================
if(NOT WIN32)
    option(DENSECORE_USE_MIMALLOC "Use mimalloc allocator override" ON)
else()
    option(DENSECORE_USE_MIMALLOC "Use mimalloc allocator override" OFF)
endif()

if(DENSECORE_USE_MIMALLOC)
    message(STATUS "[DenseCore] Mimalloc override ENABLED")
    
    set(MIMALLOC_OFF OFF CACHE BOOL "" FORCE)      # Build mimalloc
    set(MI_OVERRIDE ON CACHE BOOL "" FORCE)        # Override malloc/free
    set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)   # Static link preferred
    set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MI_INSTALL_TOPLEVEL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        mimalloc
        GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
        GIT_TAG v2.1.2
    )
    FetchContent_MakeAvailable(mimalloc)
    
    # We will link 'mimalloc-static' to the densecore target later
endif()

# ==========================================================================
# spdlog (Fast C++ Logging)
# ==========================================================================
option(DENSECORE_USE_SPDLOG "Enable spdlog for high-performance logging" ON)

if(DENSECORE_USE_SPDLOG)
    message(STATUS "[DenseCore] spdlog ENABLED")
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.12.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# ==========================================================================
# oneDNN (optional)
# ==========================================================================
option(DENSECORE_USE_ONEDNN "Enable oneDNN matmul backend" OFF)
option(DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER "Default to GGML-only threading for INT4 custom op" ON)
option(DENSECORE_DEFAULT_PREFILL_FLASH_MASK_REUSE "Default to prefill flash-mask reuse across layers" ON)
option(DENSECORE_DEFAULT_PRECOMPUTED_ROPE "Default to precomputed RoPE table custom-op path" ON)
option(DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM "Default to fused residual+RMSNorm pre-FFN path" OFF)
option(DENSECORE_DEFAULT_FUSED_QKV "Default to fused QKV projection path when F32 weights are available" ON)
option(DENSECORE_PORTABLE "Build portable binary for heterogeneous CPU fleets" OFF)

# ==========================================================================
# ARM64 Server Tuning (non-Apple)
# ==========================================================================
# - Default (non-cross compile): -mcpu=native
# - Cross compile: no default tuning flag; set one of the cache variables below
# - Use either DENSECORE_ARM_MCPU or DENSECORE_ARM_MARCH (mutually exclusive)
# ==========================================================================
set(DENSECORE_ARM_MCPU "" CACHE STRING "Explicit ARM CPU target for -mcpu (e.g. neoverse-v1, neoverse-v2, native)")
set(DENSECORE_ARM_MARCH "" CACHE STRING "Explicit ARM architecture for -march (e.g. armv8.2-a+sve+dotprod)")
set(DENSECORE_ARM_TUNE_FLAGS "")
set(DENSECORE_ARM_TUNE_ACTIVE FALSE)

if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    message(STATUS "[DenseCore][ARM64] Target profile: ${DENSEVLA_ARM_TARGET}")
    set(DENSECORE_ARM_NATIVE_AUTOTUNE FALSE)
    if(GGML_NATIVE AND NOT DENSECORE_PORTABLE AND NOT CMAKE_CROSSCOMPILING AND
       NOT DENSECORE_ARM_CORRECTNESS_FIRST_ACTIVE AND NOT DENSECORE_ARM_MCPU AND NOT DENSECORE_ARM_MARCH)
        set(DENSECORE_ARM_NATIVE_AUTOTUNE TRUE)
        message(STATUS "[DenseCore][ARM64] Native autotune active: preserving host ISA for DenseCore/ggml.")
    endif()
    if(GGML_CPU_ARM_ARCH)
        message(STATUS "[DenseCore][ARM64] GGML_CPU_ARM_ARCH: ${GGML_CPU_ARM_ARCH}")
    endif()

    if(DENSECORE_ARM_CORRECTNESS_FIRST_ACTIVE AND NOT DENSECORE_ARM_MCPU AND NOT DENSECORE_ARM_MARCH)
        set(DENSECORE_ARM_MARCH "armv8-a" CACHE STRING
            "Explicit ARM architecture for -march (e.g. armv8.2-a+sve+dotprod)" FORCE)
    elseif(NOT DENSECORE_ARM_NATIVE_AUTOTUNE AND
           NOT DENSECORE_ARM_MCPU AND NOT DENSECORE_ARM_MARCH AND DENSECORE_ARM_PROFILE_MARCH)
        set(DENSECORE_ARM_MARCH "${DENSECORE_ARM_PROFILE_MARCH}" CACHE STRING
            "Explicit ARM architecture for -march (e.g. armv8.2-a+sve+dotprod)" FORCE)
    endif()

    if(DENSECORE_ARM_MCPU AND DENSECORE_ARM_MARCH)
        message(FATAL_ERROR "Use either DENSECORE_ARM_MCPU or DENSECORE_ARM_MARCH, not both.")
    endif()

    set(DENSECORE_ARM_GLOBAL_ARCH_FLAG FALSE)
    if(CMAKE_C_FLAGS MATCHES "(^| )-march=" OR CMAKE_C_FLAGS MATCHES "(^| )-mcpu=" OR
       CMAKE_CXX_FLAGS MATCHES "(^| )-march=" OR CMAKE_CXX_FLAGS MATCHES "(^| )-mcpu=")
        set(DENSECORE_ARM_GLOBAL_ARCH_FLAG TRUE)
    endif()

    if(DENSECORE_ARM_GLOBAL_ARCH_FLAG)
        message(STATUS "[DenseCore][ARM64] Global ARM arch flags already set in CMAKE_C/CXX_FLAGS; "
                       "skipping target-level -march/-mcpu injection.")
    elseif(DENSECORE_ARM_MCPU)
        list(APPEND DENSECORE_ARM_TUNE_FLAGS "-mcpu=${DENSECORE_ARM_MCPU}")
    elseif(DENSECORE_ARM_MARCH)
        list(APPEND DENSECORE_ARM_TUNE_FLAGS "-march=${DENSECORE_ARM_MARCH}")
    elseif(CMAKE_CROSSCOMPILING)
        message(STATUS "[DenseCore][ARM64] Cross-compiling: no default -mcpu/-march. "
                       "Set DENSECORE_ARM_MCPU or DENSECORE_ARM_MARCH.")
    elseif(DENSECORE_ARM_CORRECTNESS_FIRST_ACTIVE)
        list(APPEND DENSECORE_ARM_TUNE_FLAGS "-march=armv8-a")
        message(STATUS "[DenseCore][ARM64] ARM correctness-first mode: using -march=armv8-a.")
    else()
        list(APPEND DENSECORE_ARM_TUNE_FLAGS "-mcpu=native")
        message(STATUS "[DenseCore][ARM64] Non-cross build: enabling -mcpu=native by default.")
    endif()

    if(DENSECORE_ARM_TUNE_FLAGS)
        set(DENSECORE_ARM_TUNE_ACTIVE TRUE)
        message(STATUS "[DenseCore][ARM64] Tuning flags: ${DENSECORE_ARM_TUNE_FLAGS}")
    endif()
endif()

set(DENSECORE_ONEDNN_LIB "")
if(DENSECORE_USE_ONEDNN)
    find_package(dnnl QUIET)
    if(NOT dnnl_FOUND)
        find_package(DNNL QUIET)
    endif()
    if(dnnl_FOUND)
        set(DENSECORE_ONEDNN_LIB dnnl)
    elseif(DNNL_FOUND)
        set(DENSECORE_ONEDNN_LIB DNNL::dnnl)
    else()
        message(FATAL_ERROR "DENSECORE_USE_ONEDNN=ON but oneDNN (dnnl) not found")
    endif()
endif()

# Source files (DenseCore specific only)
set(SOURCES
    src/engine.cpp
    src/worker.cpp
    src/worker_helpers.cpp
    src/inference.cpp
    src/model_types.cpp
    src/model_loader.cpp
    src/tokenizer.cpp
    src/kv_cache.cpp
    src/kv_cache_block_manager.cpp
    src/kv_cache_storage.cpp
    src/optimization_bridge.cpp
    src/save_model.cpp
    src/version.cpp
    src/error_api.cpp
    src/scheduler.cpp
    src/scheduler_policy.cpp
    src/scheduler_queue.cpp
    src/action_chunker.cpp
    src/arm_runtime.cpp
    src/operation_graph.cpp
    src/sampler/sampler.cpp
    # Backend abstraction
    src/op_registry.cpp
    src/plugin_loader.cpp
    src/graph_executor.cpp
    src/lora_storage.cpp
    src/cpu_backend.cpp
    src/backend/cpu_backend_moe_ops.cpp
    src/backend/cpu_backend_moe_registry.cpp
    src/backend_registry.cpp
    src/kernel_context.cpp
    src/async_cpu_backend.cpp
    src/qwen35_ssm_math.cpp
    src/matmul_backend.cpp
    src/matmul_backend_densecore.cpp
    src/matmul_backend_onednn.cpp
    # Kernels (optimized SIMD implementations)

    src/kv_cache_encoder.cpp
    src/kv_cache_decoder.cpp
    # MoE (Mixture of Experts) profiling and routing
    src/moe/profiler.cpp
    src/moe/moe_routing.cpp
    src/kernels/moe_ops.cpp
    src/kernels/matmul_cpu.cpp
    # Quantization module
    src/quantize_api.cpp
    src/quantizer.cpp
    src/quantization/max_quantizer.cpp
    src/quantization/awq_quantizer.cpp
    src/quantization/smoothquant_quantizer.cpp
    src/quantization/int4_quantizer.cpp
    # Pruning module
    src/pruner.cpp
    src/tensor_utils.cpp
    src/simd_ops.cpp  # Quantized dot product dispatcher
    src/pruning/depth_pruner.cpp
    src/pruning/width_pruner.cpp
    src/pruning/attention_pruner.cpp
    src/pruning/combined_pruner.cpp
    # Graph Optimization module
    src/graph_optimizer.cpp
    src/models/model_graph_bridge.cpp
    src/kernels/patch_embed_cpu.cpp
    # HAL Kernels (OpRegistry-based, vendor-extensible)
    src/kernels/activation_ops.cpp
    src/kernels/normalization_ops.cpp
    src/kernels/quantization_ops.cpp
    src/kernels/flash_attention.cpp
    src/kernels/memory_ops.cpp
    src/kernels/adaln.cpp
    src/kernels/groupnorm.cpp
    src/kernels/patch_embed_2d.cpp
    src/kernels/rope_1d.cpp
    src/kernels/rope_2d.cpp
    src/kernels/timestep_embed.cpp
    src/kernels/deformable_attention.cpp
    src/kernels/grid_sample.cpp
    src/kernels/image_ops_cpu.cpp
    src/kernels/cross_attention.cpp
    # HAL Backend Selector abstraction
    src/hal/backend_selector.cpp
    # Graph Builders (Strategy Pattern for BuildTransformerGraph)
    src/graph_builders/transformer_graph_builder_base.cpp
    src/graph_builders/whisper_graph_builder.cpp
    src/graph_builders/llm_config_generator.cpp
    src/graph/generic_graph_builder.cpp
    src/qnn_backend.cpp
    src/kernels/audio_ops.cpp
    # Universal Transformer Operations - Multimodal/Vision
    src/kernels/window_attention.cpp
    src/kernels/pos_interpolation.cpp
    # Universal Transformer Operations - Video DiT/SORA
    src/kernels/patch_embed_3d.cpp
    src/kernels/temporal_attention.cpp
    src/kernels/unpatchify.cpp
    src/kernels/patchify.cpp
    src/kernels/temporal_rope.cpp
    # Universal Transformer Operations - Diffusion
    src/kernels/noise_schedule.cpp
    # Universal Transformer Operations - Autonomous/3D
    src/kernels/bev_query.cpp
    src/kernels/scatter_reduce.cpp
    src/kernels/point_attention.cpp
    # Universal Transformer Operations - Bio/Science
    src/kernels/pair_representation.cpp
    src/kernels/triangular_attention.cpp
    src/kernels/invariant_point_attention.cpp
    src/kernels/triangular_mult_update.cpp
    src/kernels/attention_with_pair_bias.cpp
    # Universal Transformer Operations - Audio
    src/kernels/audio_conv1d.cpp
    # Highway SIMD kernels (portable single-source for all ISAs)
    src/kernels/hwy/hwy_rope.cc
    src/kernels/hwy/hwy_activation.cc
    src/kernels/hwy/hwy_image_ops.cc
    src/kernels/hwy/hwy_normalization.cc
    src/kernels/hwy/hwy_groupnorm.cc
    src/kernels/hwy/hwy_int4.cc
    src/kernels/hwy/hwy_matmul.cc
    src/kernels/hwy/hwy_fp8.cc
    src/kernels/hwy/hwy_adaln.cc
    src/kernels/hwy/hwy_attention_pair_bias.cc
    src/kernels/hwy/hwy_cross_attention.cc
    src/kernels/hwy/hwy_paged_attention.cc
    src/kernels/hwy/hwy_point_attention.cc
    src/kernels/hwy/hwy_window_attention.cc
    src/kernels/hwy/hwy_invariant_point_attention.cc
    src/kernels/hwy/hwy_triangular_attention.cc
    src/kernels/hwy/hwy_ssm.cc
)

# ==========================================================================
# Platform-Specific Source Additions (BEFORE add_library)
# ==========================================================================
# Must add platform-specific sources BEFORE add_library() is called.
# CMake captures the source list at add_library() time.
# ==========================================================================
    if(APPLE)
        # Metal source files (Objective-C++)
        set(METAL_SOURCES
            src/metal_backend.mm
            src/apple_silicon.mm
            src/ane_backend.mm
            src/hybrid_scheduler.mm
            src/hal/apple_backend_selector.mm
            src/kernels/metal_universal_ops.mm
            src/kernels/metal_linear_ops.mm
        )
    list(APPEND SOURCES ${METAL_SOURCES})
    message(STATUS "[DenseCore] Added Metal/Apple sources: ${METAL_SOURCES}")
endif()

# ==========================================================================
# SIMD Dispatch Strategy
# ==========================================================================
# x86: Uses __attribute__((target(...))) for function-level SIMD enabling.
# ARM:
#   - Apple Silicon: NEON only, tuned with -mcpu=apple-m1.
#   - Linux ARM64 (Graviton/Neoverse): tunable with DENSECORE_ARM_MCPU/-MARCH.
#     Default for non-cross builds is -mcpu=native.
# Highway and runtime SIMD detection still handle ISA selection at runtime.
# ==========================================================================


# ==========================================================================
# Create Shared Library (AFTER all source additions)
# ==========================================================================
add_library(densecore SHARED ${SOURCES})

# Include directories
target_include_directories(densecore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include  # Generated version header
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${GGML_DIR}/src/ggml-cpu
    ${simde_SOURCE_DIR} # Add SIMDe include path
    ${highway_SOURCE_DIR} # Google Highway SIMD
)

# Private include: src/ for internal headers (hwy kernels, etc.)
target_include_directories(densecore PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against ggml and Highway
target_link_libraries(densecore PRIVATE ggml hwy)
if(TARGET simdjson)
    target_link_libraries(densecore PRIVATE simdjson)
endif()
if(DENSECORE_USE_ONEDNN)
    target_link_libraries(densecore PRIVATE ${DENSECORE_ONEDNN_LIB})
    target_compile_definitions(densecore PRIVATE DENSECORE_USE_ONEDNN)
endif()

if(DENSECORE_USE_MIMALLOC)
    target_link_libraries(densecore PRIVATE mimalloc-static)
    target_compile_definitions(densecore PRIVATE DENSECORE_USE_MIMALLOC)
    # Ensure mimalloc headers are available if needed (though usually override is automatic)
    target_include_directories(densecore PRIVATE ${mimalloc_SOURCE_DIR}/include)
endif()

if(DENSECORE_USE_SPDLOG)
    target_link_libraries(densecore PRIVATE spdlog::spdlog)
    target_compile_definitions(densecore PRIVATE DENSECORE_USE_SPDLOG)
endif()

# Define DENSECORE_BUILD_SHARED for symbol export (Windows dllexport)
target_compile_definitions(densecore PRIVATE DENSECORE_BUILD_SHARED)
target_compile_definitions(densecore PRIVATE
    DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER=$<IF:$<BOOL:${DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER}>,1,0>
    DENSECORE_DEFAULT_PREFILL_FLASH_MASK_REUSE=$<IF:$<BOOL:${DENSECORE_DEFAULT_PREFILL_FLASH_MASK_REUSE}>,1,0>
    DENSECORE_DEFAULT_PRECOMPUTED_ROPE=$<IF:$<BOOL:${DENSECORE_DEFAULT_PRECOMPUTED_ROPE}>,1,0>
    DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM=$<IF:$<BOOL:${DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM}>,1,0>
    DENSECORE_DEFAULT_FUSED_QKV=$<IF:$<BOOL:${DENSECORE_DEFAULT_FUSED_QKV}>,1,0>
)
if(DENSECORE_ARM_CORRECTNESS_FIRST_ACTIVE)
    target_compile_definitions(densecore PRIVATE DENSECORE_ARM_CORRECTNESS_FIRST=1)
endif()

# AddressSanitizer option for memory debugging
option(ASAN_BUILD "Enable AddressSanitizer for memory error detection" OFF)
option(DENSECORE_BUILD_TESTS "Build DenseCore unit tests" OFF)
option(DENSECORE_BUILD_BENCHMARKS "Build DenseCore benchmarks" OFF)

if(ASAN_BUILD)
    message(STATUS "AddressSanitizer ENABLED - Building with -fsanitize=address")
    set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer -g -O1)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
endif()

# Compiler flags and optimizations
target_compile_options(densecore PRIVATE
    -Wall
    -Wextra
    $<IF:$<BOOL:${ASAN_BUILD}>,-O1,-O3>
    $<$<BOOL:${ASAN_BUILD}>:-fsanitize=address>
    $<$<BOOL:${ASAN_BUILD}>:-fno-omit-frame-pointer>
    $<$<BOOL:${ASAN_BUILD}>:-g>
)

# ==========================================================================
# Tests and Benchmarks (optional)
# ==========================================================================

# Helper function to create oneDNN-linked executables
function(add_densecore_onednn_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE densecore ggml ${DENSECORE_ONEDNN_LIB})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${GGML_DIR}/include
    )
endfunction()



if(DENSECORE_BUILD_BENCHMARKS AND DENSECORE_USE_ONEDNN)
    add_densecore_onednn_executable(densecore_onednn_matmul_bench ../benchmarks/onednn_matmul_bench.cpp)
endif()

if(DENSECORE_BUILD_BENCHMARKS)
    add_executable(alloc_bench ../benchmarks/alloc_bench.cpp)
    target_link_libraries(alloc_bench PRIVATE densecore Threads::Threads)
    if(DENSECORE_USE_MIMALLOC)
        target_compile_definitions(alloc_bench PRIVATE DENSECORE_USE_MIMALLOC)
    endif()

    foreach(_densecore_bench_target IN ITEMS bench_native quality_test qwen35_fast_qa)
        add_executable(${_densecore_bench_target} ../benchmarks/${_densecore_bench_target}.cpp)
        target_link_libraries(${_densecore_bench_target} PRIVATE densecore Threads::Threads)
        target_include_directories(${_densecore_bench_target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
            ${GGML_DIR}/include
        )
    endforeach()
endif()

# Platform-specific optimizations
if(APPLE)
    message(STATUS "Configuring for Apple platform - Using Accelerate framework")

    # Objective-C++ is enabled at project() declaration time
    # (see CMake 4.x + Ninja workaround at top of file)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

    # Set Objective-C++ flags for .mm files (METAL_SOURCES already added above)
    set_source_files_properties(${METAL_SOURCES} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )

    # ==========================================================================
    # Apple Frameworks
    # ==========================================================================
    # Accelerate (BLAS/LAPACK)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(densecore PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(densecore PRIVATE USE_ACCELERATE)
        message(STATUS "Accelerate framework found: ${ACCELERATE_FRAMEWORK}")
    else()
        message(WARNING "Accelerate framework not found")
    endif()

    # Metal (GPU compute)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)
    find_library(IOKIT_FRAMEWORK IOKit)

    if(METAL_FRAMEWORK)
        # Consolidate all Apple framework linking into a single call.
        # find_library sets variables to ...-NOTFOUND if not found,
        # and target_link_libraries ignores such values safely.
        target_link_libraries(densecore PRIVATE
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${METALPERFORMANCESHADERS_FRAMEWORK}
            ${COREML_FRAMEWORK}
            ${IOKIT_FRAMEWORK}
        )

        target_compile_definitions(densecore PRIVATE
            DENSECORE_USE_METAL
            DENSECORE_USE_ANE
            GGML_USE_METAL
        )
        message(STATUS "Metal framework enabled for GPU acceleration")
    else()
        message(WARNING "Metal framework not found - GPU acceleration disabled")
    endif()

    # ==========================================================================
    # Metal Shader Compilation
    # ==========================================================================
    set(METAL_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(METAL_SHADERS
        ${METAL_SHADER_DIR}/gemv.metal
        ${METAL_SHADER_DIR}/flash_attention.metal
        ${METAL_SHADER_DIR}/quantized_gemv.metal
        ${METAL_SHADER_DIR}/window_attention.metal
        ${METAL_SHADER_DIR}/temporal_attention.metal
        ${METAL_SHADER_DIR}/triangular_attention.metal
        ${METAL_SHADER_DIR}/patch_embed_3d.metal
        ${METAL_SHADER_DIR}/point_cloud_ops.metal
    )


    if(EXISTS ${METAL_SHADER_DIR})
        # Compile Metal shaders to metallib
        # Note: metal -c cannot use -o with multiple input files,
        # so we compile each .metal to .air separately, then link.
        set(METALLIB_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/densecore.metallib)
        set(AIR_FILES "")

        foreach(SHADER ${METAL_SHADERS})
            get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
            set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air)
            list(APPEND AIR_FILES ${AIR_FILE})

            add_custom_command(
                OUTPUT ${AIR_FILE}
                COMMAND xcrun -sdk macosx metal -c ${SHADER} -o ${AIR_FILE}
                DEPENDS ${SHADER}
                COMMENT "Compiling ${SHADER_NAME}.metal to AIR"
                VERBATIM
            )
        endforeach()

        add_custom_command(
            OUTPUT ${METALLIB_OUTPUT}
            COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB_OUTPUT}
            DEPENDS ${AIR_FILES}
            COMMENT "Linking Metal shaders into metallib"
            VERBATIM
        )

        add_custom_target(metal_shaders ALL DEPENDS ${METALLIB_OUTPUT})
        add_dependencies(densecore metal_shaders)

        # Install metallib alongside the library
        install(FILES ${METALLIB_OUTPUT} DESTINATION lib)

        message(STATUS "Metal shaders will be compiled from: ${METAL_SHADER_DIR}")
    endif()

    # ==========================================================================
    # Apple Silicon Optimizations
    # ==========================================================================
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        message(STATUS "Apple Silicon detected - enabling ARM NEON and Metal optimizations")
        target_compile_options(densecore PRIVATE
            -ffast-math
            -ffp-contract=fast
        )
        if(NOT DENSECORE_PORTABLE)
            target_compile_options(densecore PRIVATE -mcpu=apple-m1)
        endif()
        target_compile_definitions(densecore PRIVATE
            DENSECORE_APPLE_SILICON
            GGML_USE_CPU
            GGML_USE_ACCELERATE
        )
    else()
        # Intel Mac
        target_compile_definitions(densecore PRIVATE
            GGML_USE_CPU
            GGML_USE_ACCELERATE
        )
    endif()

else()
    message(STATUS "Configuring for non-Apple platform")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # Check for AVX support (compiler flags)
        include(CheckCXXCompilerFlag)

        if(MSVC)
            # MSVC uses different flags
            check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_AVX)
            check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
            check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
            set(AVX_FLAG "/arch:AVX")
            set(AVX2_FLAG "/arch:AVX2")
            set(AVX512_FLAG "/arch:AVX512")
        else()
            # GCC/Clang flags
            check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
            check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
            check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
            set(AVX_FLAG "-mavx")
            set(AVX2_FLAG "-mavx2" "-mfma")
            # AVX-512 flags including VNNI for INT8 acceleration (Ice Lake+, Sapphire Rapids, Zen4+)
            # -mavx512vnni enables vpdpbusd instruction for 4x INT8 dot products per lane
            set(AVX512_FLAG "-mavx512f" "-mavx512dq" "-mavx512vl" "-mavx512bw" "-mavx512vnni")
        endif()

        # Check CPU capabilities for AVX-512 (cross-platform approach)
        # We use a compile-and-run test to detect actual CPU support
        set(CPU_HAS_AVX512 FALSE)

        if(NOT MSVC)
            # For GCC/Clang: Try to compile and run an AVX-512 test program
            # This detects actual runtime CPU support, not just compiler support
            include(CheckCXXSourceRuns)
            set(CMAKE_REQUIRED_FLAGS "-mavx512f")
            check_cxx_source_runs("
                #include <immintrin.h>
                int main() {
                    __m512i a = _mm512_set1_epi32(1);
                    __m512i b = _mm512_add_epi32(a, a);
                    return _mm512_cvtsi512_si32(b) != 2 ? 1 : 0;
                }
            " AVX512_RUNTIME_SUPPORTED)
            unset(CMAKE_REQUIRED_FLAGS)

            if(AVX512_RUNTIME_SUPPORTED)
                set(CPU_HAS_AVX512 TRUE)
                message(STATUS "AVX-512 runtime support detected")
            else()
                # Fallback: check /proc/cpuinfo on Linux
                if(EXISTS "/proc/cpuinfo")
                    execute_process(
                        COMMAND grep -q avx512f /proc/cpuinfo
                        RESULT_VARIABLE GREP_RESULT
                        ERROR_QUIET OUTPUT_QUIET
                    )
                    if(GREP_RESULT EQUAL 0)
                        set(CPU_HAS_AVX512 TRUE)
                        message(STATUS "AVX-512 detected via /proc/cpuinfo")
                    endif()
                endif()
            endif()
        endif()

        # AMX Detection
        check_cxx_compiler_flag("-mamx-tile -mamx-int8 -mamx-bf16" COMPILER_SUPPORTS_AMX)
        if(COMPILER_SUPPORTS_AMX)
            message(STATUS "Enabling AMX optimizations support")
            set(AMX_FLAGS "-mamx-tile" "-mamx-int8" "-mamx-bf16")
        endif()

        # Apply optimizations based on detection
        if(DENSECORE_PORTABLE)
            message(STATUS "[DenseCore] PORTABLE mode: building baseline x86_64 binary with runtime dispatch only")
            target_compile_definitions(densecore PRIVATE GGML_USE_CPU)
        elseif(CPU_HAS_AVX512 AND COMPILER_SUPPORTS_AVX512)
            message(STATUS "Enabling AVX-512 optimizations (compiler and CPU)")
            target_compile_options(densecore PRIVATE ${AVX512_FLAG})
            if(COMPILER_SUPPORTS_AMX)
                 target_compile_options(densecore PRIVATE ${AMX_FLAGS})
                 target_compile_definitions(densecore PRIVATE __AMX_TILE__ __AMX_INT8__ __AMX_BF16__)
            endif()
            target_compile_definitions(densecore PRIVATE USE_AVX512 GGML_USE_CPU GGML_USE_AVX512)
            set(USING_AVX512 TRUE)
        elseif(COMPILER_SUPPORTS_AVX2)
            message(STATUS "Enabling AVX2 optimizations")
            target_compile_options(densecore PRIVATE ${AVX2_FLAG})
            target_compile_definitions(densecore PRIVATE USE_AVX2 GGML_USE_CPU GGML_USE_AVX2 GGML_USE_FMA)
            set(USING_AVX2 TRUE)
        elseif(COMPILER_SUPPORTS_AVX)
            message(STATUS "Enabling AVX optimizations")
            target_compile_options(densecore PRIVATE ${AVX_FLAG})
            target_compile_definitions(densecore PRIVATE USE_AVX GGML_USE_CPU GGML_USE_AVX)
            set(USING_AVX TRUE)
        else()
            message(STATUS "No AVX support detected - using standard optimizations")
            target_compile_definitions(densecore PRIVATE GGML_USE_CPU)
        endif()

        if(NOT MSVC AND NOT DENSECORE_PORTABLE)
            # Optimization: Enable -march=native to use all available instructions
            # This breaks portability (binary won't run on older CPUs) but maximizes performance
            message(STATUS "Enabling -march=native for maximum performance on this host")
            target_compile_options(densecore PRIVATE -march=native)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        message(STATUS "Configuring for non-Apple ARM64 platform")
        if(DENSECORE_ARM_TUNE_ACTIVE)
            target_compile_options(densecore PRIVATE ${DENSECORE_ARM_TUNE_FLAGS})
        endif()
        target_compile_definitions(densecore PRIVATE GGML_USE_CPU)
    else()
        message(STATUS "No architecture-specific SIMD flags configured for: ${CMAKE_SYSTEM_PROCESSOR}")
        target_compile_definitions(densecore PRIVATE GGML_USE_CPU)
    endif()
endif()

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(densecore PRIVATE Threads::Threads)
if(UNIX AND NOT APPLE)
    # Optional plugin loader uses dlopen/dlsym on Linux.
    target_link_libraries(densecore PRIVATE dl)
endif()

# ==========================================================================
# Threading Model: GGML Thread Pool + std::thread (NO OpenMP)
# ==========================================================================
# DenseCore uses a unified threading model:
# - Request-level parallelism: std::thread workers in worker.cpp
# - Compute-level parallelism: GGML's internal thread pool
#
# OpenMP is NOT used to avoid thread oversubscription (nested parallelism).
# All SIMD kernels in simd_ops.h are single-threaded and rely on the caller
# (GGML callbacks or worker.cpp) to manage parallel execution.
# ==========================================================================
message(STATUS "Threading: GGML thread pool + std::thread (OpenMP disabled)")


# ==========================================================================
# NUMA / hwloc Support (Optional)
# ==========================================================================
# Enables NUMA-aware memory allocation and thread affinity for multi-socket
# servers. Falls back to sysfs detection if libraries are not available.
# ==========================================================================
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(HWLOC QUIET hwloc)
    pkg_check_modules(NUMA QUIET numa)

    if(HWLOC_FOUND AND NUMA_FOUND)
        target_include_directories(densecore PRIVATE
            ${HWLOC_INCLUDE_DIRS} ${NUMA_INCLUDE_DIRS})
        target_link_libraries(densecore PRIVATE
            ${HWLOC_LIBRARIES} ${NUMA_LIBRARIES})
        target_compile_definitions(densecore PRIVATE DENSECORE_USE_HWLOC DENSECORE_HAS_NUMA)
        message(STATUS "NUMA support enabled: hwloc ${HWLOC_VERSION}, numa ${NUMA_VERSION}")
    elseif(HWLOC_FOUND)
        target_include_directories(densecore PRIVATE ${HWLOC_INCLUDE_DIRS})
        target_link_libraries(densecore PRIVATE ${HWLOC_LIBRARIES})
        target_compile_definitions(densecore PRIVATE DENSECORE_USE_HWLOC)
        message(STATUS "Partial NUMA support: hwloc ${HWLOC_VERSION} (libnuma not found)")
    else()
        message(STATUS "NUMA libs not found - using fallback sysfs detection")
    endif()
else()
    message(STATUS "pkg-config not found - NUMA support disabled")
endif()

# Position Independent Code (required for shared libraries)
set_target_properties(densecore PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Installation rules
install(TARGETS densecore
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Quantize tool
add_executable(quantize src/quantize.cpp)

target_include_directories(quantize PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${GGML_DIR}/src/ggml-cpu
    ${simde_SOURCE_DIR} # Add SIMDe include path
)

target_compile_definitions(quantize PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
    _GNU_SOURCE
)

# Reuse the same compile options/definitions as densecore for AVX/Accelerate
if(USE_ACCELERATE)
    target_link_libraries(quantize PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(quantize PRIVATE USE_ACCELERATE GGML_USE_CPU GGML_USE_ACCELERATE)
elseif(DENSECORE_PORTABLE)
    target_compile_definitions(quantize PRIVATE GGML_USE_CPU)
elseif(USING_AVX512)
    target_compile_options(quantize PRIVATE -mavx512f -mavx512dq -mavx512vl -mavx512bw)
    target_compile_definitions(quantize PRIVATE USE_AVX512 GGML_USE_CPU GGML_USE_AVX512)
elseif(USE_AVX2)
    target_compile_options(quantize PRIVATE -mavx2 -mfma)
    target_compile_definitions(quantize PRIVATE USE_AVX2 GGML_USE_CPU GGML_USE_AVX2 GGML_USE_FMA)
elseif(USE_AVX)
    target_compile_options(quantize PRIVATE -mavx)
    target_compile_definitions(quantize PRIVATE USE_AVX GGML_USE_CPU GGML_USE_AVX)
else()
    target_compile_definitions(quantize PRIVATE GGML_USE_CPU)
endif()

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" AND NOT DENSECORE_PORTABLE)
    target_compile_options(quantize PRIVATE -mcpu=apple-m1)
endif()
if(DENSECORE_ARM_TUNE_ACTIVE)
    target_compile_options(quantize PRIVATE ${DENSECORE_ARM_TUNE_FLAGS})
endif()

# Link against densecore (for CreateQuantizer, FreeINT4Data), ggml, and threads
target_link_libraries(quantize PRIVATE densecore ggml Threads::Threads)
set_target_properties(quantize PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

install(TARGETS quantize
    RUNTIME DESTINATION bin
)

install(FILES include/densecore.h
    DESTINATION include
)

# ==========================================
# Testing Infrastructure (GoogleTest)
# ==========================================
option(DENSECORE_BUILD_TESTS "Build DenseCore tests" ON)

if(DENSECORE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Prevent gtest from overriding compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # Test sources (use absolute paths to ensure correct resolution)
    set(TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_simd_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_qwen35_ssm_math.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_memory_pool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kv_cache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scheduler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_action_chunker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_arm_runtime.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hardware_topology.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_operation_graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_graph_capture.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_matmul_backend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_aligned_allocator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor_view.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernel_context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp8.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_async_backend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_int4.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_int4_gemv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_engine_kv_cache_config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_moe_profiler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_numa_sticky_routing.cpp
        # Phase 1: Universal HAL tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_transformer_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rope_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_deformable_attention.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_groupnorm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_op_registry.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_op_registry_di.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_qnn_backend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_activation_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_image_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sampling_range.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_normalization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_moe_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kv_cache_encoder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kv_cache_decoder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_graph_executor_integration.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_graph_domain_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_graph_domain_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_audio_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_whisper_e2e.cpp
        # Phase 2: Encoder-Decoder tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_attention_pair_bias_hwy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cross_attention.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_point_attention_hwy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_triangular_attention_hwy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_paged_attention.cpp
        # HAL Abstraction tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hal_abstraction.cpp
        # Universal Transformer Operations tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_universal_ops.cpp
        # Bio/Science kernel tests
        # ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bio_kernels.cpp
        # Logging tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_logging.cpp
        # Optional plugin loader security tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_plugin_loader.cpp
        # E2E smoke tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_engine_e2e.cpp
    )

    # Create test executable
    # NOTE: We include SOURCES here to compile core files directly into the test executable.
    # This allows tests to access "internal" (hidden visibility) symbols that are not exported
    # by the shared library.
    add_executable(densecore_tests ${TEST_SOURCES} ${SOURCES})

    target_include_directories(densecore_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/include  # Generated headers (densecore_version.h)
        ${GGML_DIR}/include
        ${GGML_DIR}/src
        ${GGML_DIR}/src/ggml-cpu
        ${simde_SOURCE_DIR}  # SIMDe for simd_platform.h
        ${highway_SOURCE_DIR}  # Google Highway SIMD
    )

    # Link dependencies directly (do NOT link densecore shared lib to avoid symbol conflicts/hiding)
    target_link_libraries(densecore_tests PRIVATE
        ggml
        hwy
        GTest::gtest_main
        Threads::Threads
    )

    # Enable mock model for E2E tests (allows InitEngine("mock", ...))
    target_compile_definitions(densecore_tests PRIVATE DENSECORE_TEST_BUILD)

    if(DENSECORE_USE_SPDLOG)
        target_link_libraries(densecore_tests PRIVATE spdlog::spdlog)
        target_compile_definitions(densecore_tests PRIVATE DENSECORE_USE_SPDLOG)
    endif()

    if(USE_ACCELERATE)
        target_link_libraries(densecore_tests PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()

    # Apply same compile flags as densecore
    if(DENSECORE_PORTABLE)
        target_compile_definitions(densecore_tests PRIVATE GGML_USE_CPU)
    elseif(CPU_HAS_AVX512 AND COMPILER_SUPPORTS_AVX512)
        target_compile_options(densecore_tests PRIVATE ${AVX512_FLAG})
        target_compile_definitions(densecore_tests PRIVATE USE_AVX512 GGML_USE_CPU GGML_USE_AVX512)
    elseif(COMPILER_SUPPORTS_AVX2)
        target_compile_options(densecore_tests PRIVATE ${AVX2_FLAG})
        target_compile_definitions(densecore_tests PRIVATE USE_AVX2 GGML_USE_CPU GGML_USE_AVX2 GGML_USE_FMA)
    elseif(COMPILER_SUPPORTS_AVX)
        target_compile_options(densecore_tests PRIVATE ${AVX_FLAG})
        target_compile_definitions(densecore_tests PRIVATE USE_AVX GGML_USE_CPU GGML_USE_AVX)
    else()
        target_compile_definitions(densecore_tests PRIVATE GGML_USE_CPU)
    endif()

    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" AND NOT DENSECORE_PORTABLE)
        target_compile_options(densecore_tests PRIVATE -mcpu=apple-m1)
        # Link Apple frameworks required by Metal/CoreML/Accelerate code
        target_link_libraries(densecore_tests PRIVATE
            ${ACCELERATE_FRAMEWORK}
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${METALPERFORMANCESHADERS_FRAMEWORK}
            ${COREML_FRAMEWORK}
            ${IOKIT_FRAMEWORK}
        )
    endif()
    if(DENSECORE_ARM_TUNE_ACTIVE)
        target_compile_options(densecore_tests PRIVATE ${DENSECORE_ARM_TUNE_FLAGS})
    endif()

    include(GoogleTest)
    gtest_discover_tests(densecore_tests)
endif()

# Print configuration summary
message(STATUS "==========================================")
message(STATUS "DenseCore Configuration Summary")
message(STATUS "==========================================")
message(STATUS "CMake Version: ${CMAKE_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "==========================================")
