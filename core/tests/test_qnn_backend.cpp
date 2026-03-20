#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "qnn_backend.h"

namespace densecore {
namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name ? name : "") {
        if (name_.empty()) {
            return;
        }

        const char* prev = std::getenv(name_.c_str());
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }

        Set(value);
    }

    ~ScopedEnvVar() {
        if (name_.empty()) {
            return;
        }

        if (had_prev_) {
            Set(prev_value_.c_str());
            return;
        }

        Set(nullptr);
    }

private:
    void Set(const char* value) {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), value ? value : "");
#else
        if (value) {
            setenv(name_.c_str(), value, 1);
        } else {
            unsetenv(name_.c_str());
        }
#endif
    }

    std::string name_;
    bool had_prev_ = false;
    std::string prev_value_;
};

TEST(QnnBackendAvailability, ForceEnableReportsAvailable) {
    ScopedEnvVar force_disable("DENSECORE_QNN_FORCE_DISABLE", nullptr);
    ScopedEnvVar force_enable("DENSECORE_QNN_FORCE_ENABLE", "1");
    EXPECT_TRUE(QnnBackend::CheckAvailability());
}

TEST(QnnBackendAvailability, ForceDisableReportsUnavailable) {
    ScopedEnvVar force_enable("DENSECORE_QNN_FORCE_ENABLE", nullptr);
    ScopedEnvVar force_disable("DENSECORE_QNN_FORCE_DISABLE", "1");
    EXPECT_FALSE(QnnBackend::CheckAvailability());
}

TEST(QnnBackendAvailability, ForceDisableTakesPriorityOverForceEnable) {
    ScopedEnvVar force_enable("DENSECORE_QNN_FORCE_ENABLE", "1");
    ScopedEnvVar force_disable("DENSECORE_QNN_FORCE_DISABLE", "1");
    EXPECT_FALSE(QnnBackend::CheckAvailability());
}

}  // namespace
}  // namespace densecore
