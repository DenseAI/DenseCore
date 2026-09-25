package server

import (
	"fmt"
	"os"
	"strings"

	"github.com/DenseAI/DenseCore/server/internal/config"
	"github.com/DenseAI/DenseCore/server/internal/workload"
)

// Config bootstrap ownership: environment config loading, CLI overrides, and workload defaults.
func loadBootstrapConfig(opts *Options) (*config.ServerConfig, workload.Profile, string, error) {
	logLevel := "info"
	if opts.Verbose {
		logLevel = "debug"
	}

	cfg, err := config.LoadFromEnv()
	if err != nil {
		return nil, workload.Profile{}, logLevel, err
	}
	if strings.TrimSpace(cfg.LogLevel) != "" {
		logLevel = cfg.LogLevel
	}

	if opts.Host != "" {
		cfg.Host = opts.Host
	}
	if opts.Port != 0 {
		cfg.Port = opts.Port
	}
	if opts.ModelPath != "" {
		cfg.MainModelPath = opts.ModelPath
	}
	if opts.Threads > 0 {
		cfg.Threads = opts.Threads
	}

	profile := workload.Resolve(cfg.WorkloadProfile)
	if strings.TrimSpace(cfg.WorkloadProfile) == "" {
		cfg.WorkloadProfile = profile.Name
	}
	if getEnvOrDefault("LLM_API_ENABLED", "") == "" {
		cfg.LLMAPIEnabled = profile.EnableLLMAPI
	}
	if getEnvOrDefault("MODEL_LIFECYCLE_PROBES_ENABLED", "") == "" {
		cfg.ModelLifecycleProbesEnabled = profile.EnableModelLifecycleProbes
	}
	if strings.TrimSpace(cfg.ServiceDescription) == "" {
		cfg.ServiceDescription = profile.Description
	}
	if strings.TrimSpace(cfg.ServiceName) == "" {
		cfg.ServiceName = "DenseCore API Server"
	}

	return cfg, profile, logLevel, nil
}

func validateFixedModelServeConfig(cfg *config.ServerConfig) error {
	if cfg == nil {
		return fmt.Errorf("server config is required")
	}
	if strings.TrimSpace(cfg.MainModelPath) == "" {
		return fmt.Errorf("fixed-model serve requires startup model via --model or MAIN_MODEL_PATH")
	}
	if strings.TrimSpace(cfg.DraftModelPath) != "" || strings.TrimSpace(os.Getenv("DENSECORE_DRAFT_MODEL_PATH")) != "" {
		return fmt.Errorf("fixed-model serve does not support draft_model_path, DRAFT_MODEL_PATH, or DENSECORE_DRAFT_MODEL_PATH")
	}
	return nil
}
