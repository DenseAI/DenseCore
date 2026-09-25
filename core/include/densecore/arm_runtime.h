/**
 * @file arm_runtime.h
 * @brief ARM Linux runtime controls for robotics deployments
 */

#ifndef DENSECORE_ARM_RUNTIME_H
#define DENSECORE_ARM_RUNTIME_H

#include <string>
#include <utility>
#include <vector>

namespace densecore {
namespace arm_runtime {

struct CoreClusters {
    std::vector<int> big_cores;
    std::vector<int> little_cores;
};

bool ParseCpuList(const std::string& cpu_list, std::vector<int>* out_cores);
CoreClusters PartitionCoresByMaxFrequency(const std::vector<std::pair<int, int>>& core_max_khz);
CoreClusters DetectCoreClustersLinux();

bool PinCurrentThreadToCores(const std::vector<int>& core_ids);
bool PinCurrentThreadToLittleCore(int little_index = 0);
bool PinCurrentThreadToBigCore(int big_index = 0);

enum class GovernorMode {
    Performance,
    Schedutil,
    Powersave,
};

const char* GovernorModeName(GovernorMode mode);

/**
 * @brief Temporarily switches CPU frequency governor and restores it on scope exit.
 *
 * On systems without permission (typical non-root), Enter* returns false and
 * the guard remains inactive.
 */
class GovernorGuard {
public:
    GovernorGuard() = default;
    ~GovernorGuard();

    bool EnterMode(GovernorMode mode);
    bool EnterPerformanceMode();
    void Restore();
    bool Active() const;

private:
    std::vector<std::pair<std::string, std::string>> previous_governors_;
    bool active_ = false;
};

}  // namespace arm_runtime
}  // namespace densecore

#endif  // DENSECORE_ARM_RUNTIME_H
