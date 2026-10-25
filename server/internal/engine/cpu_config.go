package engine

import (
	"log"
	"runtime"
)

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

	// Native owns mount-path, ancestor, quota, cpuset, and affinity interpretation.
	if threads, quota, memoryLimit, ok := nativeResourceBudget(); ok {
		cfg.NumThreads = threads
		cfg.CPUQuota = quota
		cfg.MemoryLimit = memoryLimit
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
	// Keep the serving thread budget aligned with the physical core budget.
	// Phase-specific runtime policy decides when prefill/decode should use
	// fewer threads; shaving the base budget here prevents those policies from
	// ever seeing the intended cap.
	threads := c.NumThreads
	if threads < 1 {
		threads = 1
	}

	return threads
}
