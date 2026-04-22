#ifndef DENSECORE_RUNTIME_ENV_H
#define DENSECORE_RUNTIME_ENV_H

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace densecore::env {

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

inline bool ParseNonZeroEnv(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }
    return std::strcmp(env, "0") != 0;
}

inline bool ParseTruthyEnv(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }

    const std::string lowered = AsciiLowerCopy(env);
    return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
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

}  // namespace densecore::env

#endif  // DENSECORE_RUNTIME_ENV_H
