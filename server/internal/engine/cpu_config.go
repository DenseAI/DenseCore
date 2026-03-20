package engine

import (
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
)

// Cgroup constants
const cgroupUnlimited = "max"

// CPUConfig holds CPU optimization settings
type CPUConfig struct {
	// Number of OS threads to use
	NumThreads int

	// GOMAXPROCS setting
	MaxProcs int

	// Enable CPU affinity (Linux only)
	EnableAffinity bool

	// CPU quota from cgroups (0 if not limited)
	CPUQuota float64

	// Memory limit from cgroups (0 if not limited)
	MemoryLimit int64
}

// DetectCPUConfig auto-detects optimal CPU configuration
func DetectCPUConfig() *CPUConfig {
	cfg := &CPUConfig{
		NumThreads:     runtime.NumCPU(),
		MaxProcs:       runtime.GOMAXPROCS(0),
		EnableAffinity: true,
	}

	// Detect cgroup limits (Kubernetes/Docker)
	cfg.CPUQuota = detectCPUQuota()
	cfg.MemoryLimit = detectMemoryLimit()

	// Adjust threads based on cgroup limits
	if cfg.CPUQuota > 0 {
		// Use CPU quota as thread count (rounded up)
		quotaThreads := int(cfg.CPUQuota + 0.5)
		if quotaThreads > 0 && quotaThreads < cfg.NumThreads {
			cfg.NumThreads = quotaThreads
			log.Printf("[CPU] Detected cgroup CPU quota: %.2f cores, using %d threads", cfg.CPUQuota, cfg.NumThreads)
		}
	}

	return cfg
}

// Apply applies the CPU configuration
func (c *CPUConfig) Apply() {
	// Set GOMAXPROCS
	if c.MaxProcs > 0 {
		prev := runtime.GOMAXPROCS(c.MaxProcs)
		log.Printf("[CPU] Set GOMAXPROCS: %d -> %d", prev, c.MaxProcs)
	}

	log.Printf("[CPU] Config: threads=%d, maxprocs=%d, affinity=%v, quota=%.2f, memory=%dMB",
		c.NumThreads, c.MaxProcs, c.EnableAffinity, c.CPUQuota, c.MemoryLimit/(1024*1024))
}

// OptimalThreadCount returns the optimal thread count for inference
func (c *CPUConfig) OptimalThreadCount() int {
	// For CPU inference, using all available threads is usually optimal
	// But leave 1-2 threads for Go runtime if we have many cores
	threads := c.NumThreads
	if threads > 8 {
		threads -= 2 // Leave 2 threads for Go runtime
	} else if threads > 4 {
		threads-- // Leave 1 thread for Go runtime
	}

	if threads < 1 {
		threads = 1
	}

	return threads
}

// detectCPUQuota reads CPU quota from cgroups v1 or v2
func detectCPUQuota() float64 {
	// Try cgroups v2 first
	if quota := readCgroupsV2CPUQuota(); quota > 0 {
		return quota
	}

	// Fall back to cgroups v1
	return readCgroupsV1CPUQuota()
}

// readCgroupsV2CPUQuota reads CPU quota from cgroups v2
func readCgroupsV2CPUQuota() float64 {
	// cgroups v2 uses cpu.max
	data, err := os.ReadFile("/sys/fs/cgroup/cpu.max")
	if err != nil {
		return 0
	}

	parts := strings.Fields(string(data))
	if len(parts) < 2 {
		return 0
	}

	if parts[0] == cgroupUnlimited {
		return 0 // Unlimited
	}

	quota, err := strconv.ParseFloat(parts[0], 64)
	if err != nil {
		return 0
	}

	period, err := strconv.ParseFloat(parts[1], 64)
	if err != nil || period == 0 {
		return 0
	}

	return quota / period
}

// readCgroupsV1CPUQuota reads CPU quota from cgroups v1
func readCgroupsV1CPUQuota() float64 {
	quotaData, err := os.ReadFile("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
	if err != nil {
		return 0
	}

	periodData, err := os.ReadFile("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
	if err != nil {
		return 0
	}

	quota, err := strconv.ParseFloat(strings.TrimSpace(string(quotaData)), 64)
	if err != nil || quota < 0 {
		return 0 // -1 means unlimited
	}

	period, err := strconv.ParseFloat(strings.TrimSpace(string(periodData)), 64)
	if err != nil || period == 0 {
		return 0
	}

	return quota / period
}

// detectMemoryLimit reads memory limit from cgroups
func detectMemoryLimit() int64 {
	// Try cgroups v2 first
	if limit := readCgroupsV2MemoryLimit(); limit > 0 {
		return limit
	}

	// Fall back to cgroups v1
	return readCgroupsV1MemoryLimit()
}

// readCgroupsV2MemoryLimit reads memory limit from cgroups v2
func readCgroupsV2MemoryLimit() int64 {
	data, err := os.ReadFile("/sys/fs/cgroup/memory.max")
	if err != nil {
		return 0
	}

	limit := strings.TrimSpace(string(data))
	if limit == "max" {
		return 0 // Unlimited
	}

	value, err := strconv.ParseInt(limit, 10, 64)
	if err != nil {
		return 0
	}

	return value
}

// readCgroupsV1MemoryLimit reads memory limit from cgroups v1
func readCgroupsV1MemoryLimit() int64 {
	data, err := os.ReadFile("/sys/fs/cgroup/memory/memory.limit_in_bytes")
	if err != nil {
		return 0
	}

	value, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0
	}

	// Check for "unlimited" (usually a very large number)
	if value > 1<<50 { // > 1 PB
		return 0
	}

	return value
}

// ============================================================================
// Memory Management
// ============================================================================

// MemoryStats holds memory statistics
type MemoryStats struct {
	Alloc      uint64 `json:"alloc_bytes"`
	TotalAlloc uint64 `json:"total_alloc_bytes"`
	Sys        uint64 `json:"sys_bytes"`
	NumGC      uint32 `json:"num_gc"`
	HeapAlloc  uint64 `json:"heap_alloc_bytes"`
	HeapSys    uint64 `json:"heap_sys_bytes"`
	HeapIdle   uint64 `json:"heap_idle_bytes"`
	HeapInuse  uint64 `json:"heap_inuse_bytes"`
}
