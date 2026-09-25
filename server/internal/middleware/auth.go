package middleware

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"regexp"
	"strings"
	"unicode"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

// ============================================================================
// Security Constants
// ============================================================================

const (
	// MaxAPIKeyLength prevents DoS via extremely long keys
	MaxAPIKeyLength = 256

	// MinAPIKeyLength ensures meaningful key entropy
	MinAPIKeyLength = 8

	// MaxAuthHeaderLength prevents header-based attacks
	MaxAuthHeaderLength = 512
)

// validAPIKeyPattern matches alphanumeric keys with hyphens/underscores (e.g., sk-abc123_XYZ)
var validAPIKeyPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// ============================================================================
// APIKeyStore Interface
// ============================================================================

// APIKeyStore interface for validating API keys.
// SECURITY: Implementations MUST store SHA-256 hashes, not raw keys.
type APIKeyStore interface {
	// Validate checks if the key hash exists and is active.
	// The key parameter is the RAW key; implementations hash it internally.
	Validate(key string) bool

	// GetTier returns the tier for the given key.
	GetTier(key string) string

	// GetUserID returns the user ID for the given key.
	GetUserID(key string) string
}

// ============================================================================
// Secure In-Memory Key Store
// ============================================================================

// InMemoryKeyStore implements APIKeyStore with in-memory storage.
// SECURITY: Keys are stored as SHA-256 hashes, not plaintext.
type InMemoryKeyStore struct {
	// keys maps SHA-256(key) → APIKeyInfo
	keys map[string]*APIKeyInfo
}

// APIKeyInfo stores metadata about an API key.
// SECURITY: The Key field stores the hash, never the raw key.
type APIKeyInfo struct {
	Key    string // SHA-256 hash of the API key
	UserID string
	Tier   string // "free", "pro", "enterprise"
	Active bool
}

// NewInMemoryKeyStore creates a new secure in-memory key store.
func NewInMemoryKeyStore() *InMemoryKeyStore {
	return &InMemoryKeyStore{
		keys: make(map[string]*APIKeyInfo),
	}
}

// hashAPIKey generates a SHA-256 hash of the API key.
// This is used for secure storage and lookup.
func hashAPIKey(key string) string {
	h := sha256.Sum256([]byte(key))
	return hex.EncodeToString(h[:])
}

// AddKey stores an API key securely (hashed).
func (s *InMemoryKeyStore) AddKey(key, userID, tier string) {
	keyHash := hashAPIKey(key)
	s.keys[keyHash] = &APIKeyInfo{
		Key:    keyHash, // Store hash, not raw key
		UserID: userID,
		Tier:   tier,
		Active: true,
	}
}

// Validate checks if the key exists and is active using constant-time comparison.
// SECURITY: Uses crypto/subtle to prevent timing attacks.
func (s *InMemoryKeyStore) Validate(key string) bool {
	keyHash := hashAPIKey(key)
	info, exists := s.keys[keyHash]
	if !exists {
		// Perform dummy comparison to maintain constant time even for non-existent keys
		dummyHash := hashAPIKey("dummy-key-for-timing-protection")
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
func (s *InMemoryKeyStore) GetTier(key string) string {
	keyHash := hashAPIKey(key)
	if info, exists := s.keys[keyHash]; exists {
		return info.Tier
	}
	return ""
}

// GetUserID returns the user ID for the given key.
func (s *InMemoryKeyStore) GetUserID(key string) string {
	keyHash := hashAPIKey(key)
	if info, exists := s.keys[keyHash]; exists {
		return info.UserID
	}
	return ""
}

// ============================================================================
// Authentication Middleware
// ============================================================================

// APIKeyAuth creates a secure authentication middleware.
// SECURITY: Implements constant-time validation and strict header parsing.
func APIKeyAuth(store APIKeyStore) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Extract and validate Authorization header with strict parsing
			apiKey, err := parseAuthorizationHeader(r.Header.Get("Authorization"))
			if err != nil {
				writeAuthError(w, err.Error())
				return
			}

			// Validate API key (store handles hashing and constant-time comparison)
			if !store.Validate(apiKey) {
				writeAuthError(w, "Invalid API key")
				return
			}

			// Attach metadata to context for downstream handlers
			// SECURITY: Store hash in context, not raw key (defense in depth)
			tier := store.GetTier(apiKey)
			userID := store.GetUserID(apiKey)
			keyHash := hashAPIKey(apiKey)

			ctx := r.Context()
			ctx = context.WithValue(ctx, apiKeyContextKey, keyHash) // Hash, not raw key
			ctx = context.WithValue(ctx, apiTierContextKey, tier)
			ctx = context.WithValue(ctx, userIDContextKey, userID)

			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// parseAuthorizationHeader performs strict validation of the Authorization header.
// SECURITY: Prevents parsing vulnerabilities and injection attacks.
func parseAuthorizationHeader(header string) (string, error) {
	// Check header length to prevent DoS
	if len(header) == 0 {
		return "", &authError{"Missing Authorization header"}
	}
	if len(header) > MaxAuthHeaderLength {
		return "", &authError{"Authorization header too long"}
	}

	// Check for non-printable or control characters (prevent header injection)
	for _, r := range header {
		if r < 32 || r > 126 {
			return "", &authError{"Invalid characters in Authorization header"}
		}
	}

	// Split only on first space (SplitN with limit 2)
	parts := strings.SplitN(header, " ", 2)
	if len(parts) != 2 {
		return "", &authError{"Invalid Authorization format. Expected: Bearer <token>"}
	}

	scheme := parts[0]
	token := parts[1]

	// Strict scheme validation (case-sensitive per RFC 7235)
	if scheme != "Bearer" {
		return "", &authError{"Invalid authorization scheme. Expected: Bearer"}
	}

	// Validate token is not empty after trimming
	token = strings.TrimSpace(token)
	if token == "" {
		return "", &authError{"Empty API key"}
	}

	// Validate token length
	if len(token) < MinAPIKeyLength {
		return "", &authError{"API key too short"}
	}
	if len(token) > MaxAPIKeyLength {
		return "", &authError{"API key too long"}
	}

	// Validate token contains only allowed characters
	if !validAPIKeyPattern.MatchString(token) {
		return "", &authError{"API key contains invalid characters"}
	}

	// Check for whitespace within token (malformed input)
	if containsWhitespace(token) {
		return "", &authError{"API key contains whitespace"}
	}

	return token, nil
}

// ParseAuthorizationHeader validates and extracts a bearer token from Authorization header value.
// This helper allows non-HTTP transports (e.g., gRPC metadata) to reuse the same strict parser.
func ParseAuthorizationHeader(header string) (string, error) {
	return parseAuthorizationHeader(header)
}

// containsWhitespace checks if string contains any whitespace.
func containsWhitespace(s string) bool {
	for _, r := range s {
		if unicode.IsSpace(r) {
			return true
		}
	}
	return false
}

// authError is a simple error type for authentication failures.
type authError struct {
	message string
}

func (e *authError) Error() string {
	return e.message
}

// ============================================================================
// Context Keys (typed to prevent collisions)
// ============================================================================

type authContextKey string

const (
	//nolint:gosec // G101 false positive: this is a context key name, not a credential
	apiKeyContextKey  authContextKey = "api_key_hash"
	apiTierContextKey authContextKey = "api_tier"
	userIDContextKey  authContextKey = "user_id"
)

// ============================================================================
// Response Helpers
// ============================================================================

// writeAuthError writes a standardized authentication error response.
func writeAuthError(w http.ResponseWriter, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)

	err := domain.ErrAuthentication(message)
	response := map[string]interface{}{
		"error": map[string]interface{}{
			"message": err.Message,
			"type":    err.Code,
			"code":    nil,
		},
	}

	_ = json.NewEncoder(w).Encode(response)
}

// ============================================================================
// Context Extraction Helpers
// ============================================================================

// GetAPIKey extracts the API key hash from context.
// SECURITY: Returns the hash, not the raw key.
func GetAPIKey(ctx context.Context) string {
	if key, ok := ctx.Value(apiKeyContextKey).(string); ok {
		return key
	}
	return ""
}

// GetAPITier extracts the API tier from context.
func GetAPITier(ctx context.Context) string {
	if tier, ok := ctx.Value(apiTierContextKey).(string); ok {
		return tier
	}
	return "free"
}

// GetUserID extracts the user ID from context.
func GetUserID(ctx context.Context) string {
	if userID, ok := ctx.Value(userIDContextKey).(string); ok {
		return userID
	}
	return ""
}
