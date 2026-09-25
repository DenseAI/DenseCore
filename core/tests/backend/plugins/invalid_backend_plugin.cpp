#include <cstdlib>
#include <fstream>

#if defined(_WIN32)
#include <windows.h>
#define DENSECORE_TEST_PLUGIN_EXPORT __declspec(dllexport)
#else
#define DENSECORE_TEST_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

namespace {

void WriteUnloadMarker() {
    const char* path = std::getenv("DENSECORE_TEST_INVALID_PLUGIN_UNLOAD_MARKER");
    if (!path || path[0] == '\0') {
        return;
    }
    std::ofstream marker(path, std::ios::out | std::ios::trunc);
    marker << "unloaded";
}

}  // namespace

extern "C" DENSECORE_TEST_PLUGIN_EXPORT int DenseCoreInvalidBackendPluginMarker() {
    return 1;
}

#if defined(_WIN32)
BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_DETACH) {
        WriteUnloadMarker();
    }
    return TRUE;
}
#else
__attribute__((destructor)) void OnInvalidPluginUnload() {
    WriteUnloadMarker();
}
#endif
