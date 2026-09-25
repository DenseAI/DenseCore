package server

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/middleware"
)

// Auth bootstrap ownership: API key store setup and Redis-backed key-store fallback.
func setupAPIKeyStore(opts *Options) (middleware.APIKeyStore, bool, error) {
	var apiKeyStore middleware.APIKeyStore
	authEnabled := opts.AuthEnabled || os.Getenv("AUTH_ENABLED") == envTrue
	if authEnabled {
		slog.Info("authentication enabled")
		redisURL := os.Getenv("REDIS_URL")
		redisKeystoreEnabled := os.Getenv("REDIS_KEYSTORE_ENABLED") == envTrue

		if redisURL != "" && redisKeystoreEnabled {
			cacheTTL, cacheSize, redisDB := parseRedisKeyStoreConfig()
			redisKeyStore, err := middleware.NewRedisKeyStore(middleware.RedisKeyStoreConfig{
				RedisURL:      redisURL,
				RedisPassword: os.Getenv("REDIS_PASSWORD"),
				RedisDB:       redisDB,
				CacheTTL:      cacheTTL,
				CacheSize:     cacheSize,
			})
			if err != nil {
				slog.Warn("failed to connect to Redis for key store, falling back to in-memory",
					slog.String("error", err.Error()))
			} else {
				slog.Info("using Redis key store for distributed API key management")
				apiKeyStore = redisKeyStore
			}
		}

		if apiKeyStore == nil {
			keyStore := middleware.NewInMemoryKeyStore()
			apiKeysEnv := os.Getenv("API_KEYS")
			if apiKeysEnv != "" {
				loadAPIKeys(keyStore, apiKeysEnv)
			} else {
				return nil, authEnabled, fmt.Errorf("AUTH_ENABLED=true but API_KEYS is empty and Redis keystore is not configured")
			}
			apiKeyStore = keyStore
		}
	} else {
		slog.Info("authentication disabled", slog.String("hint", "set AUTH_ENABLED=true to enable"))
	}
	return apiKeyStore, authEnabled, nil
}

func loadAPIKeys(store *middleware.InMemoryKeyStore, apiKeysEnv string) {
	if apiKeysEnv == "" {
		return
	}

	keys := strings.Split(apiKeysEnv, ",")
	for _, keyData := range keys {
		parts := strings.Split(strings.TrimSpace(keyData), ":")
		if len(parts) < 3 {
			slog.Warn("invalid API key format", slog.String("hint", "expected format: key:user:tier"))
			continue
		}

		key := strings.TrimSpace(parts[0])
		userID := strings.TrimSpace(parts[1])
		tier := strings.TrimSpace(parts[2])

		store.AddKey(key, userID, tier)
		slog.Info("loaded API key", slog.String("user_id", userID), slog.String("tier", tier))
	}
}

func parseRedisKeyStoreConfig() (time.Duration, int, int) {
	cacheTTLStr := getEnvOrDefault("REDIS_KEYSTORE_CACHE_TTL", "5m")
	cacheTTL, err := time.ParseDuration(cacheTTLStr)
	if err != nil {
		slog.Warn("invalid REDIS_KEYSTORE_CACHE_TTL, using default",
			slog.String("value", cacheTTLStr),
			slog.String("default", "5m"),
			slog.String("error", err.Error()))
		cacheTTL = 5 * time.Minute
	}

	cacheSizeStr := getEnvOrDefault("REDIS_KEYSTORE_CACHE_SIZE", "1000")
	cacheSize, err := strconv.Atoi(cacheSizeStr)
	if err != nil {
		slog.Warn("invalid REDIS_KEYSTORE_CACHE_SIZE, using default",
			slog.String("value", cacheSizeStr),
			slog.String("default", "1000"),
			slog.String("error", err.Error()))
		cacheSize = 1000
	}

	return cacheTTL, cacheSize, parseRedisDB()
}

func parseRedisDB() int {
	redisDBStr := getEnvOrDefault("REDIS_DB", "0")
	redisDB, err := strconv.Atoi(redisDBStr)
	if err != nil {
		slog.Warn("invalid REDIS_DB, using default",
			slog.String("value", redisDBStr),
			slog.String("default", "0"),
			slog.String("error", err.Error()))
		return 0
	}
	return redisDB
}
