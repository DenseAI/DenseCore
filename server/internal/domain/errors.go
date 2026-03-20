package domain

import "fmt"

/**
 * Error codes for API operations.
 * Following OpenAI error code conventions for compatibility.
 */
type ErrorCode string

const (
	ErrCodeInvalidRequest     ErrorCode = "invalid_request_error"
	ErrCodeAuthentication     ErrorCode = "authentication_error"
	ErrCodePermission         ErrorCode = "permission_error"
	ErrCodeRateLimit          ErrorCode = "rate_limit_error"
	ErrCodeModelNotLoaded     ErrorCode = "model_not_loaded"
	ErrCodeOutOfMemory        ErrorCode = "out_of_memory"
	ErrCodeInternalError      ErrorCode = "internal_error"
	ErrCodeServiceUnavailable ErrorCode = "service_unavailable"
)

/**
 * Structured application error with context.
 */
type AppError struct {
	Code       ErrorCode `json:"code"`
	Message    string    `json:"message"`
	StatusCode int       `json:"-"` // HTTP status code (not serialized)
	RequestID  string    `json:"request_id,omitempty"`
	Param      string    `json:"param,omitempty"` // Parameter that caused error
	Cause      error     `json:"-"`               // Underlying error (not serialized)
}

/**
 * Error implements the error interface.
 */
func (e *AppError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%s] %s (caused by: %v)", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

/**
 * Create a new AppError.
 */
func NewError(code ErrorCode, message string, statusCode int) *AppError {
	return &AppError{
		Code:       code,
		Message:    message,
		StatusCode: statusCode,
	}
}

/**
 * Add request ID to error context.
 */
func (e *AppError) WithRequestID(id string) *AppError {
	e.RequestID = id
	return e
}

/**
 * Add parameter name to error context.
 */
func (e *AppError) WithParam(param string) *AppError {
	e.Param = param
	return e
}

/**
 * Add underlying cause to error.
 */
func (e *AppError) WithCause(cause error) *AppError {
	e.Cause = cause
	return e
}

/**
 * Common error constructors for convenience.
 */

func ErrInvalidRequest(message string) *AppError {
	return NewError(ErrCodeInvalidRequest, message, 400)
}

func ErrAuthentication(message string) *AppError {
	return NewError(ErrCodeAuthentication, message, 401)
}

func ErrPermission(message string) *AppError {
	return NewError(ErrCodePermission, message, 403)
}

func ErrRateLimit(message string) *AppError {
	return NewError(ErrCodeRateLimit, message, 429)
}

func ErrModelNotLoaded(message string) *AppError {
	return NewError(ErrCodeModelNotLoaded, message, 503)
}

func ErrOutOfMemory(message string) *AppError {
	return NewError(ErrCodeOutOfMemory, message, 503)
}

func ErrInternal(message string) *AppError {
	return NewError(ErrCodeInternalError, message, 500)
}

func ErrServiceUnavailable(message string) *AppError {
	return NewError(ErrCodeServiceUnavailable, message, 503)
}
