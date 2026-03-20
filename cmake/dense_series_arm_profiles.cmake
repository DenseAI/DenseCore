include_guard(GLOBAL)

set(_dense_series_supported_arm_targets
    generic
    jetson_orin
    rpi5
    qualcomm_rb5
    custom
)

function(_dense_series_resolve_target out_var)
    if(NOT out_var)
        message(FATAL_ERROR "_dense_series_resolve_target requires an output variable.")
    endif()

    set(_dense_series_target "")
    if(DEFINED DENSECORE_ARM_TARGET AND NOT "${DENSECORE_ARM_TARGET}" STREQUAL "")
        set(_dense_series_target "${DENSECORE_ARM_TARGET}")
    endif()

    if(DEFINED DENSEVLA_ARM_TARGET AND NOT "${DENSEVLA_ARM_TARGET}" STREQUAL "")
        if(_dense_series_target AND NOT "${_dense_series_target}" STREQUAL "${DENSEVLA_ARM_TARGET}")
            message(FATAL_ERROR
                "DENSECORE_ARM_TARGET='${DENSECORE_ARM_TARGET}' and "
                "DENSEVLA_ARM_TARGET='${DENSEVLA_ARM_TARGET}' must match when both are set.")
        endif()
        set(_dense_series_target "${DENSEVLA_ARM_TARGET}")
    endif()

    if(NOT _dense_series_target)
        set(_dense_series_target "generic")
    endif()

    set(${out_var} "${_dense_series_target}" PARENT_SCOPE)
endfunction()

function(DenseSeriesInitArmTargetProfiles)
    _dense_series_resolve_target(_dense_series_target)

    set(_dense_series_target_help
        "ARM target profile: generic|jetson_orin|rpi5|qualcomm_rb5|custom")

    set(DENSECORE_ARM_TARGET "${_dense_series_target}" CACHE STRING
        "${_dense_series_target_help}" FORCE)
    set_property(CACHE DENSECORE_ARM_TARGET PROPERTY STRINGS
        ${_dense_series_supported_arm_targets})

    set(DENSEVLA_ARM_TARGET "${_dense_series_target}" CACHE STRING
        "${_dense_series_target_help}" FORCE)
    set_property(CACHE DENSEVLA_ARM_TARGET PROPERTY STRINGS
        ${_dense_series_supported_arm_targets})
endfunction()

function(DenseSeriesResolveArmProfile out_march out_ggml_arch out_is_jetson)
    if(NOT out_march OR NOT out_ggml_arch OR NOT out_is_jetson)
        message(FATAL_ERROR
            "DenseSeriesResolveArmProfile requires march, ggml arch, and jetson output variables.")
    endif()

    DenseSeriesInitArmTargetProfiles()

    set(_dense_series_arm_profile_march "")
    set(_dense_series_arm_profile_ggml "")
    set(_dense_series_arm_is_jetson FALSE)

    if("${DENSECORE_ARM_TARGET}" STREQUAL "jetson_orin")
        set(_dense_series_arm_profile_march "armv8.7-a+sve2+i8mm+bf16+dotprod")
        set(_dense_series_arm_profile_ggml "armv8.7-a+sve2+i8mm+bf16+dotprod")
        set(_dense_series_arm_is_jetson TRUE)
    elseif("${DENSECORE_ARM_TARGET}" STREQUAL "rpi5")
        set(_dense_series_arm_profile_march "armv8.2-a+dotprod+fp16")
        set(_dense_series_arm_profile_ggml "armv8.2-a+dotprod+fp16")
    elseif("${DENSECORE_ARM_TARGET}" STREQUAL "qualcomm_rb5")
        set(_dense_series_arm_profile_march "armv8.2-a+dotprod+crypto")
        set(_dense_series_arm_profile_ggml "armv8.2-a+dotprod+crypto")
    elseif("${DENSECORE_ARM_TARGET}" STREQUAL "generic")
        set(_dense_series_arm_profile_march "armv8.2-a+dotprod")
        set(_dense_series_arm_profile_ggml "armv8.2-a+dotprod")
    elseif("${DENSECORE_ARM_TARGET}" STREQUAL "custom")
        if(DEFINED DENSECORE_ARM_MARCH AND NOT "${DENSECORE_ARM_MARCH}" STREQUAL "")
            set(_dense_series_arm_profile_march "${DENSECORE_ARM_MARCH}")
        endif()
        if(DEFINED GGML_CPU_ARM_ARCH AND NOT "${GGML_CPU_ARM_ARCH}" STREQUAL "")
            set(_dense_series_arm_profile_ggml "${GGML_CPU_ARM_ARCH}")
        endif()
    else()
        message(FATAL_ERROR
            "Unknown ARM target profile '${DENSECORE_ARM_TARGET}'. "
            "Use one of: generic, jetson_orin, rpi5, qualcomm_rb5, custom")
    endif()

    set(${out_march} "${_dense_series_arm_profile_march}" PARENT_SCOPE)
    set(${out_ggml_arch} "${_dense_series_arm_profile_ggml}" PARENT_SCOPE)
    set(${out_is_jetson} "${_dense_series_arm_is_jetson}" PARENT_SCOPE)
endfunction()
