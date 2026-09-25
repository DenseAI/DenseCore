package middleware

import cloudmw "github.com/DenseAI/DenseCloud/go/middleware"

// RedisRateLimiter is the shared distributed token bucket rate limiter from
// the DenseCloud chassis. It is re-exported here so that existing call sites
// inside this repository continue to compile without changes.
type RedisRateLimiter = cloudmw.RedisRateLimiter

// RedisRateLimiterConfig is the configuration for RedisRateLimiter.
type RedisRateLimiterConfig = cloudmw.RedisRateLimiterConfig

// NewRedisRateLimiter creates a new distributed rate limiter backed by Redis.
// Falls back to in-memory rate limiting when Redis is unavailable.
var NewRedisRateLimiter = cloudmw.NewRedisRateLimiter
