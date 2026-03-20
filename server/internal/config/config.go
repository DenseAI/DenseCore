package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Environment value constants to avoid magic strings
const envValueTrue = "true"

// ServerConfig holds all server configuration
type ServerConfig struct {
	// Server settings
	Port            int           `json:"port"`
	Host            string        `json:"host"`
	ReadTimeout     time.Duration `json:"read_timeout"`
	WriteTimeout    time.Duration `json:"write_timeout"`
	IdleTimeout     time.Duration `json:"idle_timeout"`
	ShutdownTimeout time.Duration `json:"shutdown_timeout"`

	// Rate Limiting
	RateLimitEnabled   bool `json:"rate_limit_enabled"`
	RateLimitReqPerSec int  `json:"rate_limit_rps"`
	RateLimitBurst     int  `json:"rate_limit_burst"`

	// Request settings
	MaxRequestBodySize int64         `json:"max_request_body_size"`
	RequestTimeout     time.Duration `json:"request_timeout"`

	// Model settings
	MainModelPath  string `json:"main_model_path"`
	DraftModelPath string `json:"draft_model_path"`
	Threads        int    `json:"threads"`

	// CPU Optimization
	EnableCPUAffinity bool `json:"cpu_affinity"`
	MaxConcurrency    int  `json:"max_concurrency"`

	// CORS
	CORSEnabled        bool     `json:"cors_enabled"`
	CORSAllowedOrigins []string `json:"cors_allowed_origins"`

	// Logging
	LogLevel  string `json:"log_level"`
	LogFormat string `json:"log_format"` // "json" or "text"

	// Metrics
	MetricsEnabled   bool   `json:"metrics_enabled"`
	MetricsPath      string `json:"metrics_path"`
	MetricsNamespace string `json:"metrics_namespace"`

	// Workload profile and API behavior
	WorkloadProfile             string `json:"workload_profile"`
	ServiceName                 string `json:"service_name"`
	ServiceDescription          string `json:"service_description"`
	LLMAPIEnabled               bool   `json:"llm_api_enabled"`
	ModelLifecycleProbesEnabled bool   `json:"model_lifecycle_probes_enabled"`
}

// DefaultConfig returns config with sensible defaults for production
func DefaultConfig() *ServerConfig {
	return &ServerConfig{
		Port:            8080,
		Host:            "0.0.0.0",
		ReadTimeout:     30 * time.Second,
		WriteTimeout:    120 * time.Second,
		IdleTimeout:     120 * time.Second,
		ShutdownTimeout: 30 * time.Second,

		RateLimitEnabled:   true,
		RateLimitReqPerSec: 100,
		RateLimitBurst:     200,

		MaxRequestBodySize: 10 * 1024 * 1024, // 10MB
		RequestTimeout:     120 * time.Second,

		Threads: 0, // Auto-detect

		EnableCPUAffinity: true,
		MaxConcurrency:    0, // Auto

		CORSEnabled:        true,
		CORSAllowedOrigins: []string{"*"},

		LogLevel:  "info",
		LogFormat: "json",

		MetricsEnabled:   true,
		MetricsPath:      "/metrics",
		MetricsNamespace: "densecore",

		WorkloadProfile:             "llm",
		ServiceName:                 "DenseCore API Server",
		ServiceDescription:          "",
		LLMAPIEnabled:               true,
		ModelLifecycleProbesEnabled: true,
	}
}

// LoadFromEnv loads configuration from environment variables
func LoadFromEnv() (*ServerConfig, error) {
	cfg := DefaultConfig()

	// Server
	if v := os.Getenv("PORT"); v != "" {
		if port, err := strconv.Atoi(v); err == nil {
			cfg.Port = port
		}
	}
	if v := os.Getenv("HOST"); v != "" {
		cfg.Host = v
	}
	if v := os.Getenv("READ_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			cfg.ReadTimeout = d
		}
	}
	if v := os.Getenv("WRITE_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			cfg.WriteTimeout = d
		}
	}
	if v := os.Getenv("REQUEST_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			cfg.RequestTimeout = d
		}
	}
	if v := os.Getenv("SHUTDOWN_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			cfg.ShutdownTimeout = d
		}
	}

	// Rate Limiting
	if v := os.Getenv("RATE_LIMIT_ENABLED"); v != "" {
		cfg.RateLimitEnabled = strings.ToLower(v) == envValueTrue || v == "1"
	}
	if v := os.Getenv("RATE_LIMIT_RPS"); v != "" {
		if rps, err := strconv.Atoi(v); err == nil {
			cfg.RateLimitReqPerSec = rps
		}
	}
	if v := os.Getenv("RATE_LIMIT_BURST"); v != "" {
		if burst, err := strconv.Atoi(v); err == nil {
			cfg.RateLimitBurst = burst
		}
	}

	// Request limits
	if v := os.Getenv("MAX_REQUEST_BODY_SIZE"); v != "" {
		if size, err := strconv.ParseInt(v, 10, 64); err == nil {
			cfg.MaxRequestBodySize = size
		}
	}

	// Model
	cfg.MainModelPath = os.Getenv("MAIN_MODEL_PATH")
	cfg.DraftModelPath = os.Getenv("DRAFT_MODEL_PATH")
	if v := os.Getenv("THREADS"); v != "" {
		if threads, err := strconv.Atoi(v); err == nil {
			cfg.Threads = threads
		}
	}
	// CPU Optimization
	if v := os.Getenv("CPU_AFFINITY"); v != "" {
		cfg.EnableCPUAffinity = strings.ToLower(v) == envValueTrue || v == "1"
	}
	if v := os.Getenv("MAX_CONCURRENCY"); v != "" {
		if conc, err := strconv.Atoi(v); err == nil {
			cfg.MaxConcurrency = conc
		}
	}

	// CORS
	if v := os.Getenv("CORS_ENABLED"); v != "" {
		cfg.CORSEnabled = strings.ToLower(v) == envValueTrue || v == "1"
	}
	if v := os.Getenv("CORS_ALLOWED_ORIGINS"); v != "" {
		cfg.CORSAllowedOrigins = strings.Split(v, ",")
	} else if v := os.Getenv("CORS_ORIGINS"); v != "" {
		cfg.CORSAllowedOrigins = strings.Split(v, ",")
	}

	// Logging
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}
	if v := os.Getenv("LOG_FORMAT"); v != "" {
		cfg.LogFormat = v
	}
	if v := os.Getenv("METRICS_ENABLED"); v != "" {
		cfg.MetricsEnabled = strings.ToLower(v) == envValueTrue || v == "1"
	}
	if v := os.Getenv("METRICS_PATH"); v != "" {
		cfg.MetricsPath = v
	}
	if v := os.Getenv("METRICS_NAMESPACE"); v != "" {
		cfg.MetricsNamespace = v
	}

	// Workload profile and API behavior
	if v := os.Getenv("WORKLOAD_PROFILE"); v != "" {
		cfg.WorkloadProfile = v
	}
	if v := os.Getenv("SERVICE_NAME"); v != "" {
		cfg.ServiceName = v
	}
	if v := os.Getenv("SERVICE_DESCRIPTION"); v != "" {
		cfg.ServiceDescription = v
	}
	if v := os.Getenv("LLM_API_ENABLED"); v != "" {
		cfg.LLMAPIEnabled = strings.ToLower(v) == envValueTrue || v == "1"
	}
	if v := os.Getenv("MODEL_LIFECYCLE_PROBES_ENABLED"); v != "" {
		cfg.ModelLifecycleProbesEnabled = strings.ToLower(v) == envValueTrue || v == "1"
	}

	return cfg, nil
}

// Validate checks that the configuration is valid
func (c *ServerConfig) Validate() error {
	if c.Port < 1 || c.Port > 65535 {
		return fmt.Errorf("invalid port: %d", c.Port)
	}
	if c.Threads < 0 {
		return fmt.Errorf("invalid threads: %d", c.Threads)
	}
	if c.RateLimitReqPerSec < 0 {
		return fmt.Errorf("invalid rate limit: %d", c.RateLimitReqPerSec)
	}
	return nil
}

// Address returns the server address string
func (c *ServerConfig) Address() string {
	return fmt.Sprintf("%s:%d", c.Host, c.Port)
}

// String returns a human-readable config summary
func (c *ServerConfig) String() string {
	return fmt.Sprintf(
		"Config{addr=%s, profile=%s, llm_api=%v, threads=%d, rate_limit=%v(%d/s), timeout=%v}",
		c.Address(), c.WorkloadProfile, c.LLMAPIEnabled, c.Threads, c.RateLimitEnabled, c.RateLimitReqPerSec, c.RequestTimeout,
	)
}
