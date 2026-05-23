package api

// Error ownership: OpenAI-compatible error responses and generation error classification.
import (
	"context"
	"descore-server/internal/domain"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
)

// Error codes for OpenAI-compatible error responses
const (
	ErrCodeInvalidRequest   = "invalid_request"
	ErrCodeInvalidJSON      = "invalid_json"
	ErrCodeMethodNotAllowed = "method_not_allowed"
	ErrCodeServerError      = "server_error"
	ErrCodeModelNotLoaded   = "model_not_loaded"
	ErrCodeRequestTimeout   = "request_timeout"
	ErrCodeRequestCanceled  = "request_canceled"
)

func sendError(w http.ResponseWriter, message, errType, code string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	errorResp := domain.ErrorResponse{
		Error: domain.ErrorDetail{
			Message: message,
			Type:    errType,
			Code:    code,
		},
	}

	if err := json.NewEncoder(w).Encode(errorResp); err != nil {
		slog.Debug("failed to encode error response", slog.String("error", err.Error()))
	}
}

func classifyGenerationError(err error) (message, errType, code string, statusCode int) {
	var appErr *domain.AppError
	switch {
	case errors.As(err, &appErr):
		return appErr.Message, string(appErr.Code), string(appErr.Code), appErr.StatusCode
	case errors.Is(err, context.DeadlineExceeded):
		return "request timed out before completion", "timeout_error", ErrCodeRequestTimeout, http.StatusGatewayTimeout
	case errors.Is(err, context.Canceled):
		return "request canceled", "canceled_error", ErrCodeRequestCanceled, 499
	default:
		return err.Error(), "internal_error", ErrCodeServerError, http.StatusInternalServerError
	}
}

func writeGenerationError(ctx context.Context, w http.ResponseWriter, flusher http.Flusher, model string, err error, streamStarted bool) {
	message, errType, code, statusCode := classifyGenerationError(err)
	logLevel := slog.LevelError
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		logLevel = slog.LevelWarn
	}
	slog.LogAttrs(ctx, logLevel, "generation failed",
		slog.String("model", model),
		slog.String("error", err.Error()),
		slog.Bool("stream_started", streamStarted),
	)
	if streamStarted {
		writeSSEError(w, flusher, message, errType, code)
		return
	}
	sendError(w, message, errType, code, statusCode)
}
