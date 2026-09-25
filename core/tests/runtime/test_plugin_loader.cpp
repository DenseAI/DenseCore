#include <gtest/gtest.h>

#include "densecore/plugin_loader.h"

namespace {

TEST(PluginLoaderSecurity, RejectsNonDefaultPluginFilename) {
    DenseCoreEntUnloadPlugin();

    const int rc = DenseCoreEntLoadPlugin("evil_plugin.so", nullptr);
    EXPECT_EQ(rc, -4);
    EXPECT_EQ(DenseCoreEntGetPluginInfo(), nullptr);
}

TEST(PluginLoaderSecurity, RejectsPathTraversalOutsideTrustedDirectory) {
    DenseCoreEntUnloadPlugin();

#ifdef _WIN32
    const char* traversal = "..\\densecore_ent.dll";
#elif defined(__APPLE__)
    const char* traversal = "../libdensecore_ent.dylib";
#else
    const char* traversal = "../libdensecore_ent.so";
#endif

    const int rc = DenseCoreEntLoadPlugin(traversal, nullptr);
    EXPECT_EQ(rc, -4);
    EXPECT_EQ(DenseCoreEntGetPluginInfo(), nullptr);
}

}  // namespace
