# Build ownership follows source ownership. Keep SOURCES in CMakeLists.txt as
# the single inventory: newly extracted files are assigned by their module path.
# These are compilation boundaries, not claims that all C++ include dependencies
# have already been untangled. Unknown paths require an explicit ownership choice.
function(densecore_attach_modules consumer variant)
    set(modules common backend_kernels models_llm runtime api generic tools)
    set(seen_sources)
    set(module_manifest)
    foreach(source IN LISTS SOURCES)
        if(source IN_LIST seen_sources)
            message(FATAL_ERROR "Duplicate DenseCore source: ${source}")
        endif()
        list(APPEND seen_sources "${source}")
        if(source MATCHES "^src/api/")
            set(module api)
        elseif(source MATCHES "^src/(quantization|pruning|tools)/")
            set(module tools)
        elseif(source MATCHES "^src/(graph|graph_builders)/" OR
               source MATCHES "^src/runtime/graph_executor\\.cpp$" OR
               source MATCHES "^src/hal/(operation_graph|op_registry)\\.cpp$")
            set(module generic)
        elseif(source MATCHES "^src/(backend|kernels|simd)/")
            set(module backend_kernels)
        elseif(source MATCHES "^src/(models|llm|moe|sampler)/")
            set(module models_llm)
        elseif(source MATCHES "^src/runtime/")
            set(module runtime)
        elseif(source MATCHES "^src/(hal|common)/")
            set(module common)
        else()
            message(FATAL_ERROR "Assign DenseCore source to a build module: ${source}")
        endif()
        list(APPEND sources_${module} "${source}")
        string(APPEND module_manifest "${source}|${module}|densecore_${module}_${variant}\n")
    endforeach()

    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/densecore-module-map-${variant}.txt"
         CONTENT "${module_manifest}")

    foreach(module IN LISTS modules)
        if(NOT sources_${module})
            continue()
        endif()
        set(object_target densecore_${module}_${variant})
        add_library(${object_target} OBJECT ${sources_${module}})

        # Preserve each consumer's compile contract, including transitive usage
        # requirements from ggml/Highway/NUMA/spdlog and platform frameworks.
        # Do not link the consumer itself: that would create a dependency cycle.
        foreach(property IN ITEMS INCLUDE_DIRECTORIES SYSTEM_INCLUDE_DIRECTORIES
                COMPILE_DEFINITIONS COMPILE_OPTIONS COMPILE_FEATURES
                CXX_STANDARD CXX_STANDARD_REQUIRED CXX_EXTENSIONS
                OBJCXX_STANDARD OBJCXX_STANDARD_REQUIRED OBJCXX_EXTENSIONS
                INTERPROCEDURAL_OPTIMIZATION
                CXX_VISIBILITY_PRESET VISIBILITY_INLINES_HIDDEN
                POSITION_INDEPENDENT_CODE LINK_LIBRARIES)
            get_target_property(value ${consumer} ${property})
            if(NOT value STREQUAL "value-NOTFOUND")
                set_property(TARGET ${object_target} PROPERTY ${property} "${value}")
            else()
                # Clear defaults initialized later than the consumer (notably
                # Apple OBJCXX settings), rather than silently changing flags.
                set_property(TARGET ${object_target} PROPERTY ${property})
            endif()
        endforeach()

        # SHARED targets inject DEFINE_SYMBOL automatically; OBJECT targets do
        # not. Preserve the original production export preprocessor contract.
        get_target_property(consumer_type ${consumer} TYPE)
        if(consumer_type STREQUAL "SHARED_LIBRARY")
            get_target_property(export_symbol ${consumer} DEFINE_SYMBOL)
            if(export_symbol STREQUAL "export_symbol-NOTFOUND")
                set(export_symbol "${consumer}_EXPORTS")
            endif()
            if(export_symbol)
                target_compile_definitions(${object_target} PRIVATE "${export_symbol}")
            endif()
        endif()

        # Direct object inclusion retains every static registry initializer;
        # a static archive could discard otherwise-unreferenced registrations.
        target_sources(${consumer} PRIVATE $<TARGET_OBJECTS:${object_target}>)
    endforeach()
endfunction()
