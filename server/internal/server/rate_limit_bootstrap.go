package server

import (
	"log/slog"
	"os"
	"time"

	cloudmw "github.com/DenseAI/DenseCloud/go/middleware"

	"descore-server/internal/config"
	"descore-server/internal/middleware"
)

// Rate-limit bootstrap ownership: Redis rate limiter setup and in-memory fallback.
func setupRateLimiter(cfg *config.ServerConfig) cloudmw.RateLimiterInterface {
	var rateLimiter cloudmw.RateLimiterInterface
	if cfg.RateLimitEnabled {
		redisURL := os.Getenv("REDIS_URL")
		redisRateLimitEnabled := os.Getenv("REDIS_RATELIMIT_ENABLED") == envTrue

		if redisURL != "" && redisRateLimitEnabled {
			redisDB := parseRedisDB()
			redisRateLimiter, err := middleware.NewRedisRateLimiter(middleware.RedisRateLimiterConfig{
				RedisURL:          redisURL,
				RedisPassword:     os.Getenv("REDIS_PASSWORD"),
				RedisDB:           redisDB,
				RequestsPerSecond: cfg.RateLimitReqPerSec,
				Burst:             cfg.RateLimitBurst,
				FailureThreshold:  3,
				ResetTimeout:      30 * time.Second,
			})
			if err != nil {
				slog.Warn("failed to connect to Redis for rate limiting, falling back to in-memory",
					slog.String("error", err.Error()))
				rateLimiter = cloudmw.NewRateLimiter(cfg.RateLimitReqPerSec, cfg.RateLimitBurst)
			} else {
				slog.Info("using Redis rate limiter for distributed rate limiting")
				rateLimiter = redisRateLimiter
			}
		} else {
			rateLimiter = cloudmw.NewRateLimiter(cfg.RateLimitReqPerSec, cfg.RateLimitBurst)
		}
	}
	return rateLimiter
}
