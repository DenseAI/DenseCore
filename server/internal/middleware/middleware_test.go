package middleware

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	cloudmw "github.com/DenseAI/DenseCloud/go/middleware"
)

// ============================================================================
// RateLimiter (In-Memory) Tests
// ============================================================================

func TestRateLimiter_Allow(t *testing.T) {
	limiter := cloudmw.NewRateLimiter(10, 5) // 10 req/sec, 5 burst

	// Should allow burst requests
	for i := 0; i < 5; i++ {
		if !limiter.Allow() {
			t.Errorf("Request %d should be allowed (within burst)", i+1)
		}
	}

	// 6th request should be denied
	if limiter.Allow() {
		t.Error("Request 6 should be denied (burst exceeded)")
	}
}

func TestRateLimiter_TokenRefill(t *testing.T) {
	limiter := cloudmw.NewRateLimiter(100, 1) // 100 req/sec, 1 burst

	// Use the token
	if !limiter.Allow() {
		t.Error("First request should be allowed")
	}

	// Immediately should be denied
	if limiter.Allow() {
		t.Error("Second immediate request should be denied")
	}

	// Wait for refill (at 100/sec, 1 token refills in 10ms)
	time.Sleep(15 * time.Millisecond)

	// Should be allowed now
	if !limiter.Allow() {
		t.Error("Request after refill should be allowed")
	}
}

func TestRateLimiter_Concurrent(t *testing.T) {
	limiter := cloudmw.NewRateLimiter(1000, 100)

	var wg sync.WaitGroup
	allowed := 0
	var mu sync.Mutex

	// Launch 200 concurrent requests
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if limiter.Allow() {
				mu.Lock()
				allowed++
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	// Should allow roughly burst amount (100), with some variance for timing
	if allowed < 90 || allowed > 110 {
		t.Errorf("Expected ~100 allowed requests, got %d", allowed)
	}
}

func TestRateLimitMiddleware(t *testing.T) {
	limiter := cloudmw.NewRateLimiter(10, 2)

	handler := cloudmw.RateLimit(limiter)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	// First 2 should succeed (burst)
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest("GET", "/", nil)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Request %d: expected 200, got %d", i+1, w.Code)
		}
	}

	// 3rd should be rate limited
	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusTooManyRequests {
		t.Errorf("Request 3: expected 429, got %d", w.Code)
	}

	// Check Retry-After header
	if w.Header().Get("Retry-After") != "1" {
		t.Error("Expected Retry-After header")
	}
}

// ============================================================================
// RateLimiterInterface Tests
// ============================================================================

type mockRateLimiter struct {
	shouldAllow bool
}

func (m *mockRateLimiter) Allow() bool {
	return m.shouldAllow
}

func TestRateLimitWithInterface(t *testing.T) {
	tests := []struct {
		name           string
		shouldAllow    bool
		expectedStatus int
	}{
		{"Allowed", true, http.StatusOK},
		{"Denied", false, http.StatusTooManyRequests},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			limiter := &mockRateLimiter{shouldAllow: tt.shouldAllow}

			handler := cloudmw.RateLimitWithInterface(limiter)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
			}))

			req := httptest.NewRequest("GET", "/", nil)
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("Expected %d, got %d", tt.expectedStatus, w.Code)
			}
		})
	}
}

// ============================================================================
// InMemoryKeyStore Tests (for comparison/baseline)
// ============================================================================

func TestInMemoryKeyStore(t *testing.T) {
	store := NewInMemoryKeyStore()
	store.AddKey("sk-test", "user1", "pro")

	// Test Validate
	if !store.Validate("sk-test") {
		t.Error("Key should be valid")
	}
	if store.Validate("sk-invalid") {
		t.Error("Invalid key should not validate")
	}

	// Test GetTier
	if store.GetTier("sk-test") != "pro" {
		t.Error("Tier should be 'pro'")
	}

	// Test GetUserID
	if store.GetUserID("sk-test") != "user1" {
		t.Error("UserID should be 'user1'")
	}
}

func TestAPIKeyAuthMiddleware(t *testing.T) {
	store := NewInMemoryKeyStore()
	store.AddKey("sk-valid123", "user1", "pro") // Meets MinAPIKeyLength

	handler := APIKeyAuth(store)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify context values
		// Note: GetAPIKey now returns hash, not raw key
		if GetAPIKey(r.Context()) == "" {
			t.Error("API key hash not in context")
		}
		if GetAPITier(r.Context()) != "pro" {
			t.Error("Tier not in context")
		}
		if GetUserID(r.Context()) != "user1" {
			t.Error("UserID not in context")
		}
		w.WriteHeader(http.StatusOK)
	}))

	tests := []struct {
		name           string
		authHeader     string
		expectedStatus int
	}{
		{"Valid key", "Bearer sk-valid123", http.StatusOK},
		{"Invalid key", "Bearer sk-invalid1", http.StatusUnauthorized},
		{"Missing header", "", http.StatusUnauthorized},
		{"Bad format", "Basic sk-valid123", http.StatusUnauthorized},
		{"Key too short", "Bearer short", http.StatusUnauthorized},
		{"Key with invalid chars", "Bearer sk-test!@#", http.StatusUnauthorized},
		{"Empty bearer token", "Bearer ", http.StatusUnauthorized},
		{"No space after Bearer", "Bearersk-valid123", http.StatusUnauthorized},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/", nil)
			if tt.authHeader != "" {
				req.Header.Set("Authorization", tt.authHeader)
			}
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("Expected %d, got %d", tt.expectedStatus, w.Code)
			}
		})
	}
}

// ============================================================================
// Strict Header Parsing Tests
// ============================================================================

func TestParseAuthorizationHeader(t *testing.T) {
	tests := []struct {
		name        string
		header      string
		expectError bool
		errorMsg    string
	}{
		{"Valid", "Bearer sk-valid-key123", false, ""},
		{"Missing header", "", true, "Missing Authorization header"},
		{"Wrong scheme", "Basic sk-valid-key123", true, "Invalid authorization scheme"},
		{"Lowercase bearer", "bearer sk-valid-key123", true, "Invalid authorization scheme"},
		{"Key too short", "Bearer short", true, "API key too short"},
		{"Key with spaces", "Bearer sk-va lid", true, "API key contains invalid characters"},
		{"Key with special chars", "Bearer sk-test!@#$", true, "API key contains invalid characters"},
		{"Empty token", "Bearer ", true, "Empty API key"},
		{"Control characters", "Bearer sk-test\x00key", true, "Invalid characters"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			token, err := parseAuthorizationHeader(tt.header)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				} else if tt.errorMsg != "" && !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("Expected error containing %q, got %q", tt.errorMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if token == "" {
					t.Error("Expected non-empty token")
				}
			}
		})
	}
}

func TestHashedKeyStorage(t *testing.T) {
	store := NewInMemoryKeyStore()
	rawKey := "sk-mysecretkey123"
	store.AddKey(rawKey, "user1", "enterprise")

	// Validate with raw key should work
	if !store.Validate(rawKey) {
		t.Error("Raw key validation should succeed")
	}

	// Validate with hash directly should fail (can't guess the key)
	keyHash := hashAPIKey(rawKey)
	if store.Validate(keyHash) {
		t.Error("Hash validation should fail (prevents hash-as-key attacks)")
	}

	// Different key should fail
	if store.Validate("sk-differentkey1") {
		t.Error("Different key should not validate")
	}
}
