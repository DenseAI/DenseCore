package middleware

import (
	"fmt"
	"sync"

	"descore-server/internal/domain"
)

// ============================================================================
// Validator Configuration
// ============================================================================

// ValidatorConfig holds configuration for the request validator.
type ValidatorConfig struct {
	MaxTokens      int // Maximum tokens per request
	MaxPromptLen   int // Maximum prompt length in characters
	MaxMessagesLen int // Maximum number of messages in a request
	MinTemp        float64
	MaxTemp        float64
}

// DefaultValidatorConfig returns sensible defaults.
func DefaultValidatorConfig() ValidatorConfig {
	return ValidatorConfig{
		MaxTokens:      4096,
		MaxPromptLen:   32000, // ~8k tokens @ 4 chars/token
		MaxMessagesLen: 100,
		MinTemp:        0.0,
		MaxTemp:        2.0,
	}
}

// ============================================================================
// Request Validator (Thread-Safe, Dynamic Limits)
// ============================================================================

// RequestValidator validates incoming API requests.
// Thread-safe: limits can be updated at runtime when models are loaded.
type RequestValidator struct {
	mu             sync.RWMutex
	maxTokens      int
	minTemp        float64
	maxTemp        float64
	maxPromptLen   int
	maxMessagesLen int
}

// NewRequestValidator creates a validator with default limits.
func NewRequestValidator() *RequestValidator {
	cfg := DefaultValidatorConfig()
	return NewRequestValidatorWithConfig(cfg)
}

// NewRequestValidatorWithConfig creates a validator with custom configuration.
func NewRequestValidatorWithConfig(cfg ValidatorConfig) *RequestValidator {
	return &RequestValidator{
		maxTokens:      cfg.MaxTokens,
		minTemp:        cfg.MinTemp,
		maxTemp:        cfg.MaxTemp,
		maxPromptLen:   cfg.MaxPromptLen,
		maxMessagesLen: cfg.MaxMessagesLen,
	}
}

// UpdateLimits updates the validator limits based on model context size.
// This should be called by ModelService when a new model is loaded.
// Uses a conservative 4 characters per token ratio.
func (v *RequestValidator) UpdateLimits(maxContextTokens int) {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Approximate: 4 characters per token (conservative for most tokenizers)
	v.maxPromptLen = maxContextTokens * 4
	v.maxTokens = maxContextTokens
}

// UpdateLimitsWithRatio updates limits with a custom char-to-token ratio.
func (v *RequestValidator) UpdateLimitsWithRatio(maxContextTokens int, charsPerToken int) {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.maxPromptLen = maxContextTokens * charsPerToken
	v.maxTokens = maxContextTokens
}

// GetLimits returns current limits (thread-safe read).
func (v *RequestValidator) GetLimits() (maxTokens, maxPromptLen int) {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.maxTokens, v.maxPromptLen
}

// ValidateChatRequest validates a chat completion request.
func (v *RequestValidator) ValidateChatRequest(req *domain.ChatCompletionRequest) error {
	v.mu.RLock()
	maxTokens := v.maxTokens
	maxPromptLen := v.maxPromptLen
	maxMessagesLen := v.maxMessagesLen
	minTemp := v.minTemp
	maxTemp := v.maxTemp
	v.mu.RUnlock()

	// Check messages or input IDs
	if len(req.Messages) == 0 && len(req.InputIDs) == 0 {
		return domain.ErrInvalidRequest("messages array cannot be empty unless input_ids is provided").
			WithParam("messages")
	}

	if len(req.Messages) > maxMessagesLen {
		return domain.ErrInvalidRequest(
			fmt.Sprintf("messages array cannot exceed %d items", maxMessagesLen),
		).WithParam("messages")
	}

	// Validate max_tokens
	if req.MaxTokens < 1 {
		req.MaxTokens = 100 // Default value
	}
	if req.MaxTokens > maxTokens {
		return domain.ErrInvalidRequest(
			fmt.Sprintf("max_tokens must be between 1 and %d", maxTokens),
		).WithParam("max_tokens")
	}

	// Validate temperature
	if req.Temperature < minTemp || req.Temperature > maxTemp {
		return domain.ErrInvalidRequest(
			fmt.Sprintf("temperature must be between %.1f and %.1f", minTemp, maxTemp),
		).WithParam("temperature")
	}

	// Check total prompt length for message-based requests
	if len(req.Messages) > 0 {
		totalLen := 0
		for _, msg := range req.Messages {
			if msg.Role == "" {
				return domain.ErrInvalidRequest("message role cannot be empty").
					WithParam("messages[].role")
			}
			if msg.Role != "system" && msg.Role != "developer" && msg.Role != "user" &&
				msg.Role != "assistant" && msg.Role != "tool" {
				return domain.ErrInvalidRequest(
					fmt.Sprintf("invalid message role: %s (must be system, developer, user, assistant, or tool)", msg.Role),
				).WithParam("messages[].role")
			}
			totalLen += len(msg.FlattenedText())
		}

		if totalLen > maxPromptLen {
			return domain.ErrInvalidRequest(
				fmt.Sprintf("total prompt length exceeds %d characters (model context limit)", maxPromptLen),
			).WithParam("messages")
		}
	}

	return nil
}

// ValidateEmbeddingRequest validates an embedding request.
func (v *RequestValidator) ValidateEmbeddingRequest(req *domain.EmbeddingRequest) error {
	v.mu.RLock()
	maxPromptLen := v.maxPromptLen
	v.mu.RUnlock()

	texts := req.GetInputTexts()

	if len(texts) == 0 {
		return domain.ErrInvalidRequest("input cannot be empty").WithParam("input")
	}

	if len(texts) > 100 {
		return domain.ErrInvalidRequest("input array cannot exceed 100 items").
			WithParam("input")
	}

	for i, text := range texts {
		if len(text) > maxPromptLen {
			return domain.ErrInvalidRequest(
				fmt.Sprintf("input[%d] exceeds %d characters", i, maxPromptLen),
			).WithParam("input")
		}
	}

	return nil
}
