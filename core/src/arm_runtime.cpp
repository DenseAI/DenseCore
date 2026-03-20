#include "densecore/arm_runtime.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
#include <filesystem>
#include <pthread.h>
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace densecore {
namespace arm_runtime {
namespace {

std::string Trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

bool ReadFileString(const std::string& path, std::string* out) {
    if (!out) return false;
    std::ifstream ifs(path);
    if (!ifs.good()) return false;
    std::ostringstream ss;
    ss << ifs.rdbuf();
    *out = Trim(ss.str());
    return true;
}

bool WriteFileString(const std::string& path, const std::string& value) {
    std::ofstream ofs(path);
    if (!ofs.good()) return false;
    ofs << value;
    return ofs.good();
}

bool ReadIntFile(const std::string& path, int* out_value) {
    std::string text;
    if (!ReadFileString(path, &text)) return false;
    char* end = nullptr;
    const long v = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str()) return false;
    if (out_value) *out_value = static_cast<int>(v);
    return true;
}

#if defined(__linux__) && !defined(__ANDROID__)
std::vector<int> ReadOnlineCores() {
    std::vector<int> cores;
    std::string cpu_online;
    if (ReadFileString("/sys/devices/system/cpu/online", &cpu_online)) {
        ParseCpuList(cpu_online, &cores);
    }
    if (!cores.empty()) return cores;

    const int n = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    for (int i = 0; i < std::max(0, n); ++i) {
        cores.push_back(i);
    }
    return cores;
}

std::vector<std::pair<int, int>> DetectCoreMaxFreqKHzLinux() {
    std::vector<std::pair<int, int>> core_freq;
    const std::vector<int> online_cores = ReadOnlineCores();
    core_freq.reserve(online_cores.size());

    for (int core : online_cores) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(core) + "/cpufreq/";
        int freq_khz = -1;
        if (!ReadIntFile(base + "cpuinfo_max_freq", &freq_khz)) {
            ReadIntFile(base + "scaling_max_freq", &freq_khz);
        }
        core_freq.push_back({core, freq_khz});
    }
    return core_freq;
}

std::vector<std::string> GovernorPolicyPaths() {
    std::vector<std::string> paths;
    const std::filesystem::path cpufreq_root("/sys/devices/system/cpu/cpufreq");
    if (!std::filesystem::exists(cpufreq_root)) return paths;

    for (const auto& entry : std::filesystem::directory_iterator(cpufreq_root)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("policy", 0) != 0) continue;
        const std::filesystem::path gov = entry.path() / "scaling_governor";
        if (std::filesystem::exists(gov)) {
            paths.push_back(gov.string());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}
#endif

}  // namespace

bool ParseCpuList(const std::string& cpu_list, std::vector<int>* out_cores) {
    if (!out_cores) return false;
    out_cores->clear();

    const std::string trimmed = Trim(cpu_list);
    if (trimmed.empty()) return false;

    std::set<int> unique;
    std::stringstream ss(trimmed);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = Trim(token);
        if (token.empty()) continue;

        const size_t dash = token.find('-');
        if (dash == std::string::npos) {
            char* end = nullptr;
            const long cpu = std::strtol(token.c_str(), &end, 10);
            if (end == token.c_str() || cpu < 0) return false;
            unique.insert(static_cast<int>(cpu));
            continue;
        }

        const std::string lhs = token.substr(0, dash);
        const std::string rhs = token.substr(dash + 1);
        char* end_l = nullptr;
        char* end_r = nullptr;
        const long a = std::strtol(lhs.c_str(), &end_l, 10);
        const long b = std::strtol(rhs.c_str(), &end_r, 10);
        if (end_l == lhs.c_str() || end_r == rhs.c_str() || a < 0 || b < 0 || b < a) return false;
        for (long c = a; c <= b; ++c) {
            unique.insert(static_cast<int>(c));
        }
    }

    out_cores->assign(unique.begin(), unique.end());
    return !out_cores->empty();
}

CoreClusters PartitionCoresByMaxFrequency(const std::vector<std::pair<int, int>>& core_max_khz) {
    CoreClusters clusters;
    if (core_max_khz.empty()) return clusters;

    int min_freq = -1;
    int max_freq = -1;
    for (const auto& [core, freq] : core_max_khz) {
        (void)core;
        if (freq <= 0) continue;
        if (min_freq < 0 || freq < min_freq) min_freq = freq;
        if (max_freq < 0 || freq > max_freq) max_freq = freq;
    }

    if (min_freq <= 0 || max_freq <= 0 || min_freq == max_freq) {
        for (const auto& [core, freq] : core_max_khz) {
            (void)freq;
            clusters.big_cores.push_back(core);
        }
        std::sort(clusters.big_cores.begin(), clusters.big_cores.end());
        return clusters;
    }

    const int cutoff = min_freq + (max_freq - min_freq) / 2;
    for (const auto& [core, freq] : core_max_khz) {
        if (freq > 0 && freq <= cutoff) {
            clusters.little_cores.push_back(core);
        } else {
            clusters.big_cores.push_back(core);
        }
    }

    if (clusters.big_cores.empty() || clusters.little_cores.empty()) {
        clusters.big_cores.clear();
        clusters.little_cores.clear();
        for (const auto& [core, freq] : core_max_khz) {
            (void)freq;
            clusters.big_cores.push_back(core);
        }
    }

    std::sort(clusters.big_cores.begin(), clusters.big_cores.end());
    std::sort(clusters.little_cores.begin(), clusters.little_cores.end());
    return clusters;
}

CoreClusters DetectCoreClustersLinux() {
    CoreClusters clusters;
#if defined(__linux__) && !defined(__ANDROID__)
    clusters = PartitionCoresByMaxFrequency(DetectCoreMaxFreqKHzLinux());

    std::vector<int> env_little;
    std::vector<int> env_big;
    bool has_env_little = false;
    bool has_env_big = false;

    const char* little_env = std::getenv("DENSECORE_LITTLE_CORES");
    if (little_env && little_env[0] != '\0') {
        has_env_little = ParseCpuList(little_env, &env_little);
    }
    const char* big_env = std::getenv("DENSECORE_BIG_CORES");
    if (big_env && big_env[0] != '\0') {
        has_env_big = ParseCpuList(big_env, &env_big);
    }

    if (has_env_little) {
        clusters.little_cores = std::move(env_little);
    }
    if (has_env_big) {
        clusters.big_cores = std::move(env_big);
    }

    if (clusters.big_cores.empty() && clusters.little_cores.empty()) {
        clusters.big_cores = ReadOnlineCores();
    } else if (clusters.big_cores.empty()) {
        std::set<int> little(clusters.little_cores.begin(), clusters.little_cores.end());
        for (int core : ReadOnlineCores()) {
            if (little.find(core) == little.end()) {
                clusters.big_cores.push_back(core);
            }
        }
    } else if (clusters.little_cores.empty()) {
        std::set<int> big(clusters.big_cores.begin(), clusters.big_cores.end());
        for (int core : ReadOnlineCores()) {
            if (big.find(core) == big.end()) {
                clusters.little_cores.push_back(core);
            }
        }
    }

    std::sort(clusters.big_cores.begin(), clusters.big_cores.end());
    std::sort(clusters.little_cores.begin(), clusters.little_cores.end());
#endif
    return clusters;
}

bool PinCurrentThreadToCores(const std::vector<int>& core_ids) {
#if defined(__linux__) && !defined(__ANDROID__)
    if (core_ids.empty()) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int core : core_ids) {
        if (core < 0) continue;
        CPU_SET(core, &set);
    }
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)core_ids;
    return false;
#endif
}

bool PinCurrentThreadToLittleCore(int little_index) {
    const CoreClusters clusters = DetectCoreClustersLinux();
    if (clusters.little_cores.empty()) return false;
    const int idx = std::max(0, std::min<int>(little_index, static_cast<int>(clusters.little_cores.size()) - 1));
    return PinCurrentThreadToCores({clusters.little_cores[static_cast<size_t>(idx)]});
}

bool PinCurrentThreadToBigCore(int big_index) {
    const CoreClusters clusters = DetectCoreClustersLinux();
    if (clusters.big_cores.empty()) return false;
    const int idx = std::max(0, std::min<int>(big_index, static_cast<int>(clusters.big_cores.size()) - 1));
    return PinCurrentThreadToCores({clusters.big_cores[static_cast<size_t>(idx)]});
}

const char* GovernorModeName(GovernorMode mode) {
    switch (mode) {
    case GovernorMode::Performance: return "performance";
    case GovernorMode::Schedutil: return "schedutil";
    case GovernorMode::Powersave: return "powersave";
    default: return "schedutil";
    }
}

GovernorGuard::~GovernorGuard() {
    Restore();
}

bool GovernorGuard::EnterMode(GovernorMode mode) {
    Restore();

#if defined(__linux__) && !defined(__ANDROID__)
    const std::vector<std::string> paths = GovernorPolicyPaths();
    if (paths.empty()) return false;

    const std::string requested = GovernorModeName(mode);
    bool changed_any = false;

    for (const std::string& path : paths) {
        std::string previous;
        if (!ReadFileString(path, &previous)) {
            continue;
        }
        if (previous == requested) {
            continue;
        }
        if (WriteFileString(path, requested)) {
            previous_governors_.push_back({path, previous});
            changed_any = true;
        }
    }

    active_ = changed_any;
    return changed_any;
#else
    (void)mode;
    return false;
#endif
}

bool GovernorGuard::EnterPerformanceMode() {
    return EnterMode(GovernorMode::Performance);
}

void GovernorGuard::Restore() {
#if defined(__linux__) && !defined(__ANDROID__)
    for (auto it = previous_governors_.rbegin(); it != previous_governors_.rend(); ++it) {
        WriteFileString(it->first, it->second);
    }
#endif
    previous_governors_.clear();
    active_ = false;
}

bool GovernorGuard::Active() const {
    return active_;
}

}  // namespace arm_runtime
}  // namespace densecore
