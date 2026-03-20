#include <gtest/gtest.h>

#include "densecore/enterprise_plugin.h"

namespace {

TEST(EnterprisePluginLoaderSecurity, RejectsNonDefaultPluginFilename) {
    DenseCoreEntUnloadPlugin();

    const int rc = DenseCoreEntLoadPlugin("evil_plugin.so", nullptr);
    EXPECT_EQ(rc, -4);
    EXPECT_EQ(DenseCoreEntGetPluginInfo(), nullptr);
}

TEST(EnterprisePluginLoaderSecurity, RejectsPathTraversalOutsideTrustedDirectory) {
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
