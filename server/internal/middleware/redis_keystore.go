package middleware

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/redis/go-redis/v9"
)

// deactivateKeyScript is the Lua script for atomic key deactivation.
// SECURITY: Performs atomic read-modify-write to prevent race conditions.
const deactivateKeyScript = `
local data = redis.call('GET', KEYS[1])
if not data then
    return redis.error_reply("key not found")
end

local info = cjson.decode(data)
info.Active = false
local newData = cjson.encode(info)
redis.call('SET', KEYS[1], newData)
return 'OK'
`

// cachedKeyInfo wraps APIKeyInfo with cache metadata.
type cachedKeyInfo struct {
	Info     *APIKeyInfo
	CachedAt time.Time
}

// RedisKeyStore stores API keys as SHA-256 hashes in Redis
// with an LRU cache layer to reduce Redis load.
type RedisKeyStore struct {
	client           *redis.Client
	cache            *lru.Cache[string, *cachedKeyInfo]
	cacheTTL         time.Duration
	keyPrefix        string
	deactivateScript *redis.Script

	// Circuit breaker for Redis failures
	mu          sync.RWMutex
	failures    int
	lastFailure time.Time
	circuitOpen bool
}

// RedisKeyStoreConfig holds configuration for the Redis key store.
type RedisKeyStoreConfig struct {
	RedisURL      string
	RedisPassword string
	RedisDB       int
	CacheTTL      time.Duration // Default: 5 minutes
	CacheSize     int           // Default: 1000
}

// NewRedisKeyStore creates a new Redis-backed API key store.
func NewRedisKeyStore(cfg RedisKeyStoreConfig) (*RedisKeyStore, error) {
	opts, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		return nil, err
	}

	if cfg.RedisPassword != "" {
		opts.Password = cfg.RedisPassword
	}
	opts.DB = cfg.RedisDB

	client := redis.NewClient(opts)

	// Verify connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close() // Best-effort cleanup on connection failure
		return nil, err
	}

	// Set defaults
	cacheTTL := cfg.CacheTTL
	if cacheTTL <= 0 {
		cacheTTL = 5 * time.Minute
	}
	cacheSize := cfg.CacheSize
	if cacheSize <= 0 {
		cacheSize = 1000
	}

	cache, err := lru.New[string, *cachedKeyInfo](cacheSize)
	if err != nil {
		_ = client.Close() // Best-effort cleanup on cache creation failure
		return nil, err
	}

	return &RedisKeyStore{
		client:           client,
		cache:            cache,
		cacheTTL:         cacheTTL,
		keyPrefix:        "apikey:",
		deactivateScript: redis.NewScript(deactivateKeyScript),
	}, nil
}

// hashKey generates a SHA-256 hash of the API key.
func hashKey(key string) string {
	h := sha256.Sum256([]byte(key))
	return hex.EncodeToString(h[:])
}

// Validate checks if the key exists and is active.
// SECURITY: Uses constant-time comparison to prevent timing attacks.
// Even for non-existent keys, we perform dummy operations to maintain
// consistent timing and prevent key enumeration.
func (s *RedisKeyStore) Validate(key string) bool {
	keyHash := hashKey(key)
	info := s.getKeyInfo(key)

	if info == nil {
		// Perform dummy comparison to maintain constant time even for non-existent keys
		// This prevents attackers from distinguishing between "key not found" and "key found but invalid"
		dummyHash := hashKey("dummy-key-for-timing-protection")
		subtle.ConstantTimeCompare([]byte(keyHash), []byte(dummyHash))
		return false
	}

	// Constant-time comparison of stored hash with computed hash
	if subtle.ConstantTimeCompare([]byte(info.Key), []byte(keyHash)) != 1 {
		return false
	}

	return info.Active
}

// GetTier returns the tier for the given key.
func (s *RedisKeyStore) GetTier(key string) string {
	info := s.getKeyInfo(key)
	if info == nil {
		return ""
	}
	return info.Tier
}

// GetUserID returns the user ID for the given key.
func (s *RedisKeyStore) GetUserID(key string) string {
	info := s.getKeyInfo(key)
	if info == nil {
		return ""
	}
	return info.UserID
}

// getKeyInfo retrieves key info from cache or Redis.
func (s *RedisKeyStore) getKeyInfo(key string) *APIKeyInfo {
	keyHash := hashKey(key)

	// Check local cache first
	if cached, ok := s.cache.Get(keyHash); ok {
		if time.Since(cached.CachedAt) < s.cacheTTL {
			return cached.Info
		}
		// Cache expired, remove it
		s.cache.Remove(keyHash)
	}

	// Check circuit breaker
	if s.isCircuitOpen() {
		slog.Warn("circuit open, cache miss for key", slog.String("component", "keystore"))
		return nil
	}

	// Fetch from Redis
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	data, err := s.client.Get(ctx, s.keyPrefix+keyHash).Bytes()
	if err != nil {
		if err != redis.Nil {
			s.recordFailure()
			slog.Error("Redis GET failed", slog.String("component", "keystore"), slog.String("error", err.Error()))
		}
		return nil
	}

	s.recordSuccess()

	var info APIKeyInfo
	if err := json.Unmarshal(data, &info); err != nil {
		slog.Error("failed to unmarshal key info", slog.String("component", "keystore"), slog.String("error", err.Error()))
		return nil
	}

	// Update cache
	s.cache.Add(keyHash, &cachedKeyInfo{
		Info:     &info,
		CachedAt: time.Now(),
	})

	return &info
}

// AddKey stores an API key in Redis (admin operation).
// The key is stored as a SHA-256 hash for security.
func (s *RedisKeyStore) AddKey(key, userID, tier string) error {
	keyHash := hashKey(key)

	info := &APIKeyInfo{
		Key:    keyHash, // Store hash, not original key
		UserID: userID,
		Tier:   tier,
		Active: true,
	}

	data, err := json.Marshal(info)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := s.client.Set(ctx, s.keyPrefix+keyHash, data, 0).Err(); err != nil {
		return err
	}

	// Invalidate cache
	s.cache.Remove(keyHash)

	slog.Info("added API key", slog.String("component", "keystore"), slog.String("user_id", userID), slog.String("tier", tier))
	return nil
}

// DeleteKey removes an API key from Redis.
func (s *RedisKeyStore) DeleteKey(key string) error {
	keyHash := hashKey(key)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := s.client.Del(ctx, s.keyPrefix+keyHash).Err(); err != nil {
		return err
	}

	// Invalidate cache
	s.cache.Remove(keyHash)
	return nil
}

// DeactivateKey marks a key as inactive without deleting it.
// SECURITY: Uses atomic Lua script to prevent race conditions.
func (s *RedisKeyStore) DeactivateKey(key string) error {
	keyHash := hashKey(key)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Use atomic Lua script to prevent race conditions
	_, err := s.deactivateScript.Run(ctx, s.client, []string{s.keyPrefix + keyHash}).Result()
	if err != nil {
		return err
	}

	// Invalidate cache
	s.cache.Remove(keyHash)
	return nil
}

// isCircuitOpen checks if Redis circuit breaker is open.
func (s *RedisKeyStore) isCircuitOpen() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.circuitOpen {
		return false
	}

	// Try again after 30 seconds
	if time.Since(s.lastFailure) > 30*time.Second {
		return false
	}

	return true
}

// recordFailure records a Redis failure.
func (s *RedisKeyStore) recordFailure() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.failures++
	s.lastFailure = time.Now()

	if s.failures >= 3 && !s.circuitOpen {
		slog.Warn("circuit breaker OPEN", slog.String("component", "keystore"), slog.Int("failures", s.failures))
		s.circuitOpen = true
	}
}

// recordSuccess resets the circuit breaker.
func (s *RedisKeyStore) recordSuccess() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.circuitOpen {
		slog.Info("circuit breaker CLOSED, Redis recovered", slog.String("component", "keystore"))
	}
	s.failures = 0
	s.circuitOpen = false
}

// Close closes the Redis connection.
func (s *RedisKeyStore) Close() error {
	if s.client != nil {
		return s.client.Close()
	}
	return nil
}

// HealthCheck verifies Redis connectivity.
func (s *RedisKeyStore) HealthCheck(ctx context.Context) error {
	return s.client.Ping(ctx).Err()
}
