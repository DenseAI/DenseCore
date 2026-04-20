/**
 * @file plugin_loader.cpp
 * @brief dlopen-based optional plugin loader implementation
 *
 * Discovers and loads libdensecore_ent.so at runtime, resolving the
 * plugin init symbol to activate optional features.
 * Gracefully fails when .so is absent (normal OSS operation continues).
 *
 * Complexity: O(1) — dlopen + dlsym + single function call
 */

#include "densecore/plugin_loader.h"
#include "densecore/confidential_compute.h"
#include "densecore/license_validator.h"
#include "densecore/numa_routing.h"
#include "densecore/telemetry_sink.h"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef DENSECORE_USE_SPDLOG
#include <spdlog/fmt/bundled/printf.h>
#include <spdlog/spdlog.h>
#define ENT_LOG_DEBUG(format_str, ...) spdlog::debug("{}", fmt::sprintf(format_str, ##__VA_ARGS__))
#define ENT_LOG_INFO(format_str, ...) spdlog::info("{}", fmt::sprintf(format_str, ##__VA_ARGS__))
#define ENT_LOG_WARN(format_str, ...) spdlog::warn("{}", fmt::sprintf(format_str, ##__VA_ARGS__))
#define ENT_LOG_ERROR(format_str, ...) spdlog::error("{}", fmt::sprintf(format_str, ##__VA_ARGS__))
#else
#include <cstdarg>
#include <cstdio>

namespace {

inline void EntLog(FILE* stream, const char* level, const char* fmt, ...) {
    std::fprintf(stream, "[ENT %s] ", level);
    if (fmt != nullptr) {
        va_list args;
        va_start(args, fmt);
        std::vfprintf(stream, fmt, args);
        va_end(args);
    } else {
        std::fputs("(null)", stream);
    }
    std::fputc('\n', stream);
}

}  // namespace

#define ENT_LOG_DEBUG(fmt, ...) EntLog(stdout, "DEBUG", fmt, ##__VA_ARGS__)
#define ENT_LOG_INFO(fmt, ...) EntLog(stdout, "INFO", fmt, ##__VA_ARGS__)
#define ENT_LOG_WARN(fmt, ...) EntLog(stderr, "WARN", fmt, ##__VA_ARGS__)
#define ENT_LOG_ERROR(fmt, ...) EntLog(stderr, "ERROR", fmt, ##__VA_ARGS__)
#endif

// =============================================================================
// Global State (module-internal)
// =============================================================================

namespace {
namespace fs = std::filesystem;

struct PluginState {
    void* dl_handle = nullptr;
    DenseCoreEntInitFn init_fn = nullptr;
    DenseCoreEntShutdownFn shutdown_fn = nullptr;
    const DenseCoreEntPluginInfo* info = nullptr;
    bool loaded = false;
};

// Single global plugin state — only one plugin is supported.
PluginState g_plugin;

// VTable registries (written once during init, read concurrently after).
// Using std::atomic for the pointer to ensure visibility across threads.
std::atomic<const DenseCoreEntTelemetryVTable*> g_telemetry_vtable{nullptr};
std::atomic<const DenseCoreEntLicenseVTable*> g_license_vtable{nullptr};
std::atomic<const DenseCoreEntNumaRoutingVTable*> g_numa_routing_vtable{nullptr};
std::atomic<const DenseCoreEntConfidentialVTable*> g_confidential_vtable{nullptr};

#ifdef _WIN32
constexpr const char* kDefaultPluginName = "densecore_ent.dll";
#elif defined(__APPLE__)
constexpr const char* kDefaultPluginName = "libdensecore_ent.dylib";
#else
constexpr const char* kDefaultPluginName = "libdensecore_ent.so";
#endif

constexpr int kPluginPathInvalid = -4;

std::string TrimWhitespace(const char* value) {
    if (!value) {
        return {};
    }

    std::string out(value);
    const size_t first = out.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return {};
    }
    const size_t last = out.find_last_not_of(" \t\n\r");
    return out.substr(first, last - first + 1);
}

bool IsPathWithinBase(const fs::path& candidate, const fs::path& base_dir) {
    const fs::path relative = candidate.lexically_relative(base_dir);
    if (relative.empty()) {
        return false;
    }
    for (const auto& component : relative) {
        if (component == "..") {
            return false;
        }
    }
    return true;
}

bool ResolveLoaderDir(fs::path* loader_dir) {
    if (!loader_dir) {
        return false;
    }

#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(&ResolveLoaderDir), &module)) {
        return false;
    }

    std::vector<char> module_path(512, '\0');
    for (;;) {
        const DWORD len = GetModuleFileNameA(module, module_path.data(), static_cast<DWORD>(module_path.size()));
        if (len == 0) {
            return false;
        }
        if (len < module_path.size() - 1) {
            module_path.resize(len);
            break;
        }
        module_path.resize(module_path.size() * 2, '\0');
    }
    fs::path module_file = fs::path(std::string(module_path.begin(), module_path.end()));
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&ResolveLoaderDir), &info) == 0 || !info.dli_fname ||
        info.dli_fname[0] == '\0') {
        return false;
    }
    fs::path module_file(info.dli_fname);
#endif

    std::error_code ec;
    const fs::path canonical_module_file = fs::canonical(module_file, ec);
    if (ec) {
        return false;
    }
    if (canonical_module_file.empty() || canonical_module_file.parent_path().empty()) {
        return false;
    }

    *loader_dir = canonical_module_file.parent_path();
    return true;
}

void ClearRegisteredVTables() {
    g_telemetry_vtable.store(nullptr, std::memory_order_release);
    g_license_vtable.store(nullptr, std::memory_order_release);
    g_numa_routing_vtable.store(nullptr, std::memory_order_release);
    g_confidential_vtable.store(nullptr, std::memory_order_release);
}

void CloseDynamicLibrary(void* dl_handle) {
    if (!dl_handle) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(dl_handle));
#else
    dlclose(dl_handle);
#endif
}

void RollbackFailedLoad() {
    CloseDynamicLibrary(g_plugin.dl_handle);
    g_plugin = PluginState{};
}

int ResolvePluginPath(const char* plugin_path, std::string* resolved_path) {
    if (!resolved_path) {
        return kPluginPathInvalid;
    }

    fs::path loader_dir;
    if (!ResolveLoaderDir(&loader_dir)) {
        ENT_LOG_ERROR("Unable to resolve DenseCore module directory for secure plugin loading");
        return kPluginPathInvalid;
    }

    std::error_code ec;
    const fs::path canonical_loader_dir = fs::canonical(loader_dir, ec);
    if (ec) {
        ENT_LOG_ERROR("Unable to canonicalize DenseCore module directory '%s'", loader_dir.string().c_str());
        return kPluginPathInvalid;
    }

    const std::string raw_input = TrimWhitespace(plugin_path);
    const bool has_custom_path = !raw_input.empty();
    fs::path candidate_path;
    if (!has_custom_path) {
        candidate_path = canonical_loader_dir / kDefaultPluginName;
    } else {
        fs::path requested(raw_input);
        if (requested.is_relative()) {
            requested = canonical_loader_dir / requested;
        }

        requested = requested.lexically_normal();
        if (!requested.is_absolute()) {
            ENT_LOG_ERROR("Rejected plugin path '%s': must resolve to an absolute path", raw_input.c_str());
            return kPluginPathInvalid;
        }
        if (requested.filename() != kDefaultPluginName) {
            ENT_LOG_ERROR("Rejected plugin path '%s': filename must be '%s'", raw_input.c_str(), kDefaultPluginName);
            return kPluginPathInvalid;
        }

        candidate_path = requested;
        if (!IsPathWithinBase(candidate_path, canonical_loader_dir)) {
            ENT_LOG_ERROR("Rejected plugin path '%s': outside trusted directory '%s'", raw_input.c_str(),
                          canonical_loader_dir.string().c_str());
            return kPluginPathInvalid;
        }
    }

    ec.clear();
    const bool exists = fs::exists(candidate_path, ec);
    if (ec) {
        ENT_LOG_ERROR("Failed to inspect plugin path '%s': %s", candidate_path.string().c_str(), ec.message().c_str());
        return kPluginPathInvalid;
    }
    if (!exists) {
        *resolved_path = candidate_path.string();
        return 1;
    }

    ec.clear();
    const bool is_regular_file = fs::is_regular_file(candidate_path, ec);
    if (ec) {
        ENT_LOG_ERROR("Failed to inspect plugin path '%s': %s", candidate_path.string().c_str(), ec.message().c_str());
        return kPluginPathInvalid;
    }
    if (!is_regular_file) {
        *resolved_path = candidate_path.string();
        return 1;
    }

    if (has_custom_path) {
        const fs::path canonical_candidate = fs::canonical(candidate_path, ec);
        if (ec) {
            ENT_LOG_ERROR("Failed to canonicalize plugin path '%s': %s", candidate_path.string().c_str(),
                          ec.message().c_str());
            return kPluginPathInvalid;
        }
        if (!IsPathWithinBase(canonical_candidate, canonical_loader_dir)) {
            ENT_LOG_ERROR("Rejected plugin path '%s': outside trusted directory '%s'",
                          canonical_candidate.string().c_str(), canonical_loader_dir.string().c_str());
            return kPluginPathInvalid;
        }
        candidate_path = canonical_candidate;
    }

    *resolved_path = candidate_path.string();
    return 0;
}

}  // namespace

// =============================================================================
// Plugin Loader Implementation
// =============================================================================

extern "C" {

int DenseCoreEntLoadPlugin(const char* plugin_path, void* engine) {
    if (g_plugin.loaded) {
        ENT_LOG_WARN("Plugin already loaded, ignoring duplicate load");
        return 0;
    }

    std::string resolved_path;
    const int path_result = ResolvePluginPath(plugin_path, &resolved_path);
    if (path_result == 1) {
        ENT_LOG_DEBUG("Plugin not found at '%s' (running in OSS mode)", resolved_path.c_str());
        return 1;
    }
    if (path_result < 0) {
        return path_result;
    }
    const char* path = resolved_path.c_str();

#ifdef _WIN32
    g_plugin.dl_handle = LoadLibraryA(path);
#else
    // RTLD_NOW: resolve all symbols immediately (fail fast on missing deps)
    // RTLD_LOCAL: don't pollute global symbol namespace
    g_plugin.dl_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif

    if (!g_plugin.dl_handle) {
#ifdef _WIN32
        ENT_LOG_ERROR("Plugin failed to load from '%s'", path);
#else
        ENT_LOG_ERROR("Plugin failed to load from '%s': %s", path, dlerror());
#endif
        return -1;
    }

    // Resolve init symbol
#ifdef _WIN32
    auto init_fn = reinterpret_cast<DenseCoreEntInitFn>(
        GetProcAddress(static_cast<HMODULE>(g_plugin.dl_handle), DENSECORE_ENT_PLUGIN_SYMBOL));
#else
    auto init_fn = reinterpret_cast<DenseCoreEntInitFn>(dlsym(g_plugin.dl_handle, DENSECORE_ENT_PLUGIN_SYMBOL));
#endif

    if (!init_fn) {
        ENT_LOG_ERROR("Plugin loaded but missing symbol: %s", DENSECORE_ENT_PLUGIN_SYMBOL);
        ClearRegisteredVTables();
        RollbackFailedLoad();
        return -1;
    }

    g_plugin.init_fn = init_fn;

    // Resolve optional shutdown symbol
#ifdef _WIN32
    g_plugin.shutdown_fn = reinterpret_cast<DenseCoreEntShutdownFn>(
        GetProcAddress(static_cast<HMODULE>(g_plugin.dl_handle), "DenseCoreEntPluginShutdown"));
#else
    g_plugin.shutdown_fn =
        reinterpret_cast<DenseCoreEntShutdownFn>(dlsym(g_plugin.dl_handle, "DenseCoreEntPluginShutdown"));
#endif

    // Call plugin init
    const DenseCoreEntPluginInfo* plugin_info = nullptr;
    int result = g_plugin.init_fn(engine, &plugin_info);

    if (result != 0 || !plugin_info) {
        ENT_LOG_ERROR("Plugin init failed (code=%d)", result);
        // Plugin init can register vtables before returning failure.
        ClearRegisteredVTables();
        RollbackFailedLoad();
        return -2;
    }

    // API version compatibility check
    uint32_t major = (plugin_info->api_version >> 16) & 0xFFFF;
    if (major != DENSECORE_ENT_API_VERSION_MAJOR) {
        ENT_LOG_ERROR("Plugin API version mismatch: host=%u.x, plugin=%u.%u", DENSECORE_ENT_API_VERSION_MAJOR, major,
                      plugin_info->api_version & 0xFFFF);
        // Plugin init already ran, so clear any registered callbacks before unload.
        ClearRegisteredVTables();
        RollbackFailedLoad();
        return -3;
    }

    g_plugin.info = plugin_info;
    g_plugin.loaded = true;

    char capability_hex[32];
    std::snprintf(capability_hex, sizeof(capability_hex), "0x%llX",
                  static_cast<unsigned long long>(plugin_info->capabilities));
    ENT_LOG_INFO("Plugin loaded: %s v%s (capabilities=%s)", plugin_info->name, plugin_info->version, capability_hex);

    return 0;
}

const DenseCoreEntPluginInfo* DenseCoreEntGetPluginInfo(void) {
    return g_plugin.loaded ? g_plugin.info : nullptr;
}

int DenseCoreEntHasCapability(uint64_t capability) {
    if (!g_plugin.loaded || !g_plugin.info) return 0;
    return (g_plugin.info->capabilities & capability) != 0 ? 1 : 0;
}

void DenseCoreEntUnloadPlugin(void) {
    if (!g_plugin.loaded) return;

    // Call plugin shutdown if available
    if (g_plugin.shutdown_fn) {
        g_plugin.shutdown_fn();
    }

    // Clear VTables before dlclose (prevents dangling function pointers)
    ClearRegisteredVTables();
    CloseDynamicLibrary(g_plugin.dl_handle);

    g_plugin = PluginState{};

    ENT_LOG_INFO("Plugin unloaded");
}

// =============================================================================
// VTable Registration (called by plugin during init)
// =============================================================================

void DenseCoreEntRegisterTelemetry(const DenseCoreEntTelemetryVTable* vtable) {
    g_telemetry_vtable.store(vtable, std::memory_order_release);
    ENT_LOG_INFO("Telemetry VTable registered");
}

const DenseCoreEntTelemetryVTable* DenseCoreEntGetTelemetryVTable(void) {
    return g_telemetry_vtable.load(std::memory_order_acquire);
}

void DenseCoreEntRegisterLicense(const DenseCoreEntLicenseVTable* vtable) {
    g_license_vtable.store(vtable, std::memory_order_release);
    ENT_LOG_INFO("License VTable registered");
}

const DenseCoreEntLicenseVTable* DenseCoreEntGetLicenseVTable(void) {
    return g_license_vtable.load(std::memory_order_acquire);
}

void DenseCoreEntRegisterNumaRouting(const DenseCoreEntNumaRoutingVTable* vtable) {
    g_numa_routing_vtable.store(vtable, std::memory_order_release);
    ENT_LOG_INFO("NUMA routing VTable registered");
}

const DenseCoreEntNumaRoutingVTable* DenseCoreEntGetNumaRoutingVTable(void) {
    return g_numa_routing_vtable.load(std::memory_order_acquire);
}

void DenseCoreEntRegisterConfidential(const DenseCoreEntConfidentialVTable* vtable) {
    g_confidential_vtable.store(vtable, std::memory_order_release);
    ENT_LOG_INFO("Confidential-computing VTable registered");
}

const DenseCoreEntConfidentialVTable* DenseCoreEntGetConfidentialVTable(void) {
    return g_confidential_vtable.load(std::memory_order_acquire);
}

}  // extern "C"
