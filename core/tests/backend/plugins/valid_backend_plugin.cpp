#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>

#include "densecore/hal/compute_backend.h"

#if defined(_WIN32)
#include <malloc.h>
#include <windows.h>
#define DENSECORE_TEST_PLUGIN_EXPORT __declspec(dllexport)
#else
#define DENSECORE_TEST_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

namespace {

void WriteMarker(const char* env_name, const char* value) {
    const char* path = std::getenv(env_name);
    if (!path || path[0] == '\0') {
        return;
    }
    std::ofstream marker(path, std::ios::out | std::ios::trunc);
    marker << value;
}

bool MarkerExists(const char* env_name) {
    const char* path = std::getenv(env_name);
    if (!path || path[0] == '\0') {
        return false;
    }
    std::ifstream marker(path);
    return marker.good();
}

class TestPluginBackend final : public densecore::ComputeBackend {
public:
    ~TestPluginBackend() override { WriteMarker("DENSECORE_TEST_BACKEND_DESTROY_MARKER", "destroyed"); }

    const char* Name() const override { return "DenseCoreTestPlugin"; }
    densecore::DeviceType Device() const override { return densecore::DeviceType::ASIC; }

    void* AllocateDevice(size_t size_bytes, size_t alignment) override {
        if (alignment == 0) {
            alignment = alignof(std::max_align_t);
        }
#if defined(_WIN32)
        return _aligned_malloc(size_bytes, alignment);
#else
        void* ptr = nullptr;
        return posix_memalign(&ptr, alignment, size_bytes) == 0 ? ptr : nullptr;
#endif
    }

    void FreeDevice(void* ptr) override {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    void CopyToDevice(void* dst, const void* src, size_t size_bytes) override { std::memcpy(dst, src, size_bytes); }
    void CopyFromDevice(void* dst, const void* src, size_t size_bytes) override { std::memcpy(dst, src, size_bytes); }

    void MatMul(const densecore::Tensor&, const densecore::Tensor&, densecore::Tensor*) override {}
    void MatMulTransB(const densecore::Tensor&, const densecore::Tensor&, densecore::Tensor*) override {}
    void GemmInt4(const densecore::Tensor&, const densecore::Tensor&, const densecore::Tensor&,
                  const densecore::Tensor&, densecore::Tensor*, int) override {}
    void RMSNorm(const densecore::Tensor&, const densecore::Tensor&, densecore::Tensor*, float) override {}
    void AddRMSNorm(const densecore::Tensor&, const densecore::Tensor&, const densecore::Tensor&, densecore::Tensor*,
                    float) override {}
    void LayerNorm(const densecore::Tensor&, const densecore::Tensor&, const densecore::Tensor&, densecore::Tensor*,
                   float) override {}
    void Softmax(const densecore::Tensor&, densecore::Tensor*) override {}
    void SoftmaxInplace(densecore::Tensor*) override {}
    void SiLU(const densecore::Tensor&, densecore::Tensor*) override {}
    void GELU(const densecore::Tensor&, densecore::Tensor*) override {}
    void RoPE(const densecore::Tensor&, const densecore::Tensor&, const int*, densecore::Tensor*, int) override {}
    void FusedQKVProjection(const densecore::Tensor&, const densecore::Tensor&, const densecore::Tensor&,
                            const densecore::Tensor&, densecore::Tensor*, densecore::Tensor*,
                            densecore::Tensor*) override {}
    void FlashAttention(const densecore::Tensor&, const densecore::Tensor&, const densecore::Tensor&,
                        densecore::Tensor*, float, bool, int, int, float, uint32_t, int, int) override {}
    void Synchronize() override {}
};

}  // namespace

extern "C" DENSECORE_TEST_PLUGIN_EXPORT std::unique_ptr<densecore::ComputeBackend> CreateBackend() {
    return std::make_unique<TestPluginBackend>();
}

#if defined(_WIN32)
BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_DETACH) {
        WriteMarker("DENSECORE_TEST_PLUGIN_UNLOAD_MARKER", MarkerExists("DENSECORE_TEST_BACKEND_DESTROY_MARKER")
                                                               ? "unloaded-after-destroy"
                                                               : "unloaded-before-destroy");
    }
    return TRUE;
}
#else
__attribute__((destructor)) void OnPluginUnload() {
    WriteMarker("DENSECORE_TEST_PLUGIN_UNLOAD_MARKER", MarkerExists("DENSECORE_TEST_BACKEND_DESTROY_MARKER")
                                                           ? "unloaded-after-destroy"
                                                           : "unloaded-before-destroy");
}
#endif
