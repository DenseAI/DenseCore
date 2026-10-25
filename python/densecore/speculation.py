"""Versioned native initialization options, separate from request sampling."""

import ctypes


class NativeInitOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("spec_type", ctypes.c_int32),
        ("spec_draft_n_max", ctypes.c_int32),
    ]


def init_options(spec_type: str = "auto", spec_draft_n_max: int = 2) -> NativeInitOptions:
    modes = {"auto": 0, "none": 1, "mtp": 2}
    mode = spec_type.strip().lower() if isinstance(spec_type, str) else ""
    if mode not in modes:
        raise ValueError("spec_type must be auto, none or mtp")
    if type(spec_draft_n_max) is not int or spec_draft_n_max not in (1, 2, 3):
        raise ValueError("spec_draft_n_max must be 1, 2 or 3")
    return NativeInitOptions(ctypes.sizeof(NativeInitOptions), 1, modes[mode], spec_draft_n_max)


def initialize_native(lib, model_path: bytes, threads: int, kv_type: int, options: NativeInitOptions):
    """Use the additive API, while keeping ordinary decode on older libraries."""
    if hasattr(lib, "InitEngineWithOptions"):
        return lib.InitEngineWithOptions(model_path, None, threads, -1, 0, kv_type, ctypes.byref(options))
    if options.spec_type == 2:
        raise RuntimeError("explicit mtp requires a native library with InitEngineWithOptions")
    if kv_type != 0:
        if not hasattr(lib, "InitEngineWithKVType"):
            raise RuntimeError("selected KV cache type requires InitEngineWithKVType")
        return lib.InitEngineWithKVType(model_path, None, threads, -1, 0, kv_type)
    return lib.InitEngine(model_path, None, threads)


class NativeRuntimePlan(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("version", ctypes.c_uint32),
        ("effective_threads", ctypes.c_int32), ("effective_context", ctypes.c_int32),
        ("effective_concurrency", ctypes.c_int32), ("spec_requested", ctypes.c_int32),
        ("spec_selected", ctypes.c_int32), ("spec_draft_n_max", ctypes.c_int32),
        *[(name, ctypes.c_uint64) for name in (
            "available_bytes", "required_bytes", "kv_bytes", "graph_bytes", "repack_bytes",
            "recurrent_bytes", "prefix_snapshot_bytes", "mtp_bytes", "allocator_bytes")],
        ("spec_reason", ctypes.c_char * 192),
    ]


def runtime_plan(lib, handle) -> dict:
    if not hasattr(lib, "DenseCoreGetRuntimePlan"):
        raise RuntimeError("runtime diagnostics require DenseCoreGetRuntimePlan")
    result = NativeRuntimePlan()
    result.struct_size = ctypes.sizeof(result)
    result.version = 1
    if lib.DenseCoreGetRuntimePlan(handle, ctypes.byref(result)) != 0:
        raise RuntimeError("failed to read native runtime plan")
    values = {name: getattr(result, name) for name, _ in result._fields_}
    values["spec_reason"] = result.spec_reason.decode("utf-8", errors="replace")
    return values
