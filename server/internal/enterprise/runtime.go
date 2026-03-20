package enterprise

import (
	"context"
	"log/slog"
	"net/http"
)

// Runtime defines optional enterprise server integrations.
type Runtime interface {
	Enabled() bool
	APIMiddleware() []func(http.Handler) http.Handler
	RegisterRoutes(rootMux, apiMux *http.ServeMux)
	Startup(ctx context.Context) error
	Shutdown(ctx context.Context) error
}

// NewRuntime creates an enterprise runtime.
//
// The concrete implementation is selected by build tags.
func NewRuntime(logger *slog.Logger) (Runtime, error) {
	return newRuntime(logger)
}
