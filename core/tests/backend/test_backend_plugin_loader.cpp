#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "densecore.h"
#include "densecore/hal/backend_registry.h"

namespace densecore {
namespace {

class BackendPluginLoaderTest : public ::testing::Test {
protected:
    void SetUp() override { BackendRegistry::Instance().ResetForTesting(); }

    void TearDown() override {
        BackendRegistry::Instance().ResetForTesting();
        BackendRegistry::Instance().RegisterCpuBackend();
        UnsetEnv("DENSECORE_TEST_BACKEND_DESTROY_MARKER");
        UnsetEnv("DENSECORE_TEST_PLUGIN_UNLOAD_MARKER");
        UnsetEnv("DENSECORE_TEST_INVALID_PLUGIN_UNLOAD_MARKER");
    }

    static void SetEnv(const char* name, const std::string& value) {
#if defined(_WIN32)
        _putenv_s(name, value.c_str());
#else
        setenv(name, value.c_str(), 1);
#endif
    }

    static void UnsetEnv(const char* name) {
#if defined(_WIN32)
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
    }

    static std::string ReadFile(const std::filesystem::path& path) {
        std::ifstream input(path);
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
};

TEST_F(BackendPluginLoaderTest, MissingPluginReturnsFailure) {
    EXPECT_EQ(DenseCoreLoadPlugin("/densecore/tests/does-not-exist/backend-plugin.so"), DENSECORE_STATUS_BACKEND_ERROR);
    EXPECT_STRNE(DenseCoreGetLastError(), "");
}

TEST_F(BackendPluginLoaderTest, PluginWithoutCreateBackendReturnsFailure) {
    const auto marker = std::filesystem::temp_directory_path() / "densecore-invalid-backend-plugin-unloaded";
    std::filesystem::remove(marker);
    SetEnv("DENSECORE_TEST_INVALID_PLUGIN_UNLOAD_MARKER", marker.string());

    EXPECT_EQ(DenseCoreLoadPlugin(DENSECORE_TEST_INVALID_BACKEND_PLUGIN_PATH), DENSECORE_STATUS_BACKEND_ERROR);
    EXPECT_FALSE(BackendRegistry::Instance().IsRegistered(DeviceType::ASIC));
    EXPECT_EQ(ReadFile(marker), "unloaded");
    std::filesystem::remove(marker);
}

TEST_F(BackendPluginLoaderTest, ValidPluginRegistersBackendAndUnloadsAfterBackendDestruction) {
    const auto marker_dir = std::filesystem::temp_directory_path() / "densecore-backend-plugin-test";
    std::filesystem::create_directories(marker_dir);
    const auto destroy_marker = marker_dir / "backend-destroyed";
    const auto unload_marker = marker_dir / "plugin-unloaded";
    std::filesystem::remove(destroy_marker);
    std::filesystem::remove(unload_marker);
    SetEnv("DENSECORE_TEST_BACKEND_DESTROY_MARKER", destroy_marker.string());
    SetEnv("DENSECORE_TEST_PLUGIN_UNLOAD_MARKER", unload_marker.string());

    ASSERT_EQ(DenseCoreLoadPlugin(DENSECORE_TEST_BACKEND_PLUGIN_PATH), DENSECORE_STATUS_OK);
    EXPECT_STREQ(DenseCoreGetLastError(), "");
    ComputeBackend* backend = BackendRegistry::Instance().Get(DeviceType::ASIC);
    ASSERT_NE(backend, nullptr);
    EXPECT_STREQ(backend->Name(), "DenseCoreTestPlugin");
    EXPECT_FALSE(std::filesystem::exists(destroy_marker));
    EXPECT_FALSE(std::filesystem::exists(unload_marker));

    BackendRegistry::Instance().ResetForTesting();

    EXPECT_EQ(ReadFile(destroy_marker), "destroyed");
    EXPECT_EQ(ReadFile(unload_marker), "unloaded-after-destroy");
    std::filesystem::remove_all(marker_dir);
}

}  // namespace
}  // namespace densecore
