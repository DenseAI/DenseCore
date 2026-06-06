MoEUserData* AllocateMoEUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    if (ggml_get_no_alloc(ctx_c)) {
        thread_local MoEUserData dry_run_storage;
        dry_run_storage = MoEUserData{};
        return &dry_run_storage;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(MoEUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    std::memset(storage->data, 0, sizeof(MoEUserData));
    return reinterpret_cast<MoEUserData*>(storage->data);
}

static GLMDSAPackUserData* AllocateGLMDSAPackUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    if (ggml_get_no_alloc(ctx_c)) {
        thread_local GLMDSAPackUserData dry_run_storage;
        dry_run_storage = GLMDSAPackUserData{};
        return &dry_run_storage;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(GLMDSAPackUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    return reinterpret_cast<GLMDSAPackUserData*>(storage->data);
}

static HiddenSnapshotUserData* AllocateHiddenSnapshotUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    if (ggml_get_no_alloc(ctx_c)) {
        thread_local HiddenSnapshotUserData dry_run_storage;
        dry_run_storage = HiddenSnapshotUserData{};
        return &dry_run_storage;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(HiddenSnapshotUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    std::memset(storage->data, 0, sizeof(HiddenSnapshotUserData));
    return reinterpret_cast<HiddenSnapshotUserData*>(storage->data);
}

static Gemma4KVSummaryUserData* AllocateGemma4KVSummaryUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    if (ggml_get_no_alloc(ctx_c)) {
        thread_local Gemma4KVSummaryUserData dry_run_storage;
        dry_run_storage = Gemma4KVSummaryUserData{};
        return &dry_run_storage;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(Gemma4KVSummaryUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    std::memset(storage->data, 0, sizeof(Gemma4KVSummaryUserData));
    return reinterpret_cast<Gemma4KVSummaryUserData*>(storage->data);
}
