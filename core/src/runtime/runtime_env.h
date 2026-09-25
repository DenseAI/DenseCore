#ifndef DENSECORE_RUNTIME_ENV_H
#define DENSECORE_RUNTIME_ENV_H

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

// Environment-variable access for the runtime.
//
// Two rules govern this header:
//
//  1. There is exactly one boolean spelling. `ParseBoolEnv` accepts
//     1/true/yes/on and 0/false/no/off, and falls back to the caller's default
//     for anything it does not recognise. Do not add a second boolean parser --
//     the previous split between "non-zero is true" and "only truthy words are
//     true" meant `VAR=false` enabled some features and disabled others.
//
//  2. Diagnostic knobs go through the `*Diagnostic*` accessors. Those compile
//     to their default in release builds, so they cost nothing on hot paths and
//     do not form part of the supported configuration surface. Configure with
//     -DDENSECORE_ENABLE_DEBUG_ENV=ON to get them back in a release build.
//
// Operational knobs -- the ones documented in docs/PERFORMANCE_TUNING.md -- use
// the plain accessors and are always live.

namespace densecore::env {

#if defined(DENSECORE_ENABLE_DEBUG_ENV)
inline constexpr bool kDiagnosticEnvEnabled = true;
#elif defined(NDEBUG)
inline constexpr bool kDiagnosticEnvEnabled = false;
#else
inline constexpr bool kDiagnosticEnvEnabled = true;
#endif

enum class RuntimeToggleMode { Off = 0, Auto = 1, On = 2 };

inline std::string AsciiLowerCopy(const char* value) {
    if (!value) {
        return {};
    }
    std::string lowered(value);
    for (char& ch : lowered) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return lowered;
}

// Unset, empty, or unrecognised values yield `default_value`. An unrecognised
// value never silently flips a feature the other way.
inline bool ParseBoolEnv(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_value;
    }

    const std::string lowered = AsciiLowerCopy(value);
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        return false;
    }
    return default_value;
}

inline int ParsePositiveEnvInt(const char* name, int default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }

    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

inline int ParseIntEnv(const char* name, int default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }

    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0') {
        return default_value;
    }
    if (parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

inline RuntimeToggleMode ParseRuntimeToggleModeValue(const char* value, RuntimeToggleMode default_mode) {
    if (!value || value[0] == '\0') {
        return default_mode;
    }

    const std::string mode = AsciiLowerCopy(value);
    if (mode == "off" || mode == "0" || mode == "false" || mode == "no") {
        return RuntimeToggleMode::Off;
    }
    if (mode == "on" || mode == "1" || mode == "true" || mode == "yes" || mode == "force") {
        return RuntimeToggleMode::On;
    }
    if (mode == "auto") {
        return RuntimeToggleMode::Auto;
    }
    return default_mode;
}

inline RuntimeToggleMode ParseRuntimeToggleMode(const char* name, RuntimeToggleMode default_mode) {
    return ParseRuntimeToggleModeValue(std::getenv(name), default_mode);
}

// ---------------------------------------------------------------------------
// Diagnostic accessors: compiled out of release builds.
// ---------------------------------------------------------------------------

inline bool ParseDiagnosticEnv(const char* name, bool default_value = false) {
    if constexpr (kDiagnosticEnvEnabled) {
        return ParseBoolEnv(name, default_value);
    } else {
        (void)name;
        return default_value;
    }
}

inline int ParseDiagnosticEnvInt(const char* name, int default_value) {
    if constexpr (kDiagnosticEnvEnabled) {
        return ParseIntEnv(name, default_value);
    } else {
        (void)name;
        return default_value;
    }
}

inline int ParseDiagnosticPositiveEnvInt(const char* name, int default_value) {
    if constexpr (kDiagnosticEnvEnabled) {
        return ParsePositiveEnvInt(name, default_value);
    } else {
        (void)name;
        return default_value;
    }
}

// Returns nullptr in release builds, so callers keep their "unset" path.
inline const char* GetDiagnosticEnv(const char* name) {
    if constexpr (kDiagnosticEnvEnabled) {
        return std::getenv(name);
    } else {
        (void)name;
        return nullptr;
    }
}

}  // namespace densecore::env

#endif  // DENSECORE_RUNTIME_ENV_H
