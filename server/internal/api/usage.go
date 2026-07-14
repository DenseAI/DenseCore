package api

import (
	"context"
	"fmt"
)

// InferenceUsageRecorder is an optional neutral hook implemented by runtime
// extensions that account for successful inference usage.
type InferenceUsageRecorder interface {
	RecordInferenceUsage(ctx context.Context, promptTokens, completionTokens int) error
}

func WithInferenceUsageRecorder(recorder InferenceUsageRecorder) HandlerOption {
	return func(h *Handler) { h.usageRecorder = recorder }
}

func (h *Handler) recordInferenceUsage(ctx context.Context, promptTokens, completionTokens int) error {
	if h == nil || h.usageRecorder == nil {
		return nil
	}
	if promptTokens < 0 || completionTokens < 0 {
		return fmt.Errorf("inference token counts must be non-negative")
	}
	return h.usageRecorder.RecordInferenceUsage(ctx, promptTokens, completionTokens)
}
