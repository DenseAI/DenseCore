package server

import (
	"os"
	"strconv"
	"strings"
)

// Runtime utility ownership: environment parsing and small sizing helpers used by bootstrap phases.
func defaultGoWorkersForThreadCount(threads int) int {
	return defaultGoWorkersForEngineCapacity(threads, 4)
}

func defaultGoWorkersForEngineCapacity(threads int, maxNumSeqs int) int {
	target := maxInt(threads, maxNumSeqs)
	target = maxInt(target, 8)
	target = minInt(target, 32)
	return maxInt(target, 1)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func envFlagEnabled(key string) bool {
	value := strings.TrimSpace(os.Getenv(key))
	return value != "" && value != "0" && !strings.EqualFold(value, "false")
}

func envIntDefault(name string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

func parseEnvBool(key string, defaultVal bool) bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv(key)))
	if v == "" {
		return defaultVal
	}
	return v == "1" || v == envTrue || v == "yes" || v == "on"
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvOrDefaultInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.Atoi(value); err == nil {
			return i
		}
	}
	return defaultValue
}
