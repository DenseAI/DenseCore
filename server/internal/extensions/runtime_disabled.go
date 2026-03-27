//go:build !denseenterprise

package extensions

import (
	"context"
	"log/slog"
	"net/http"
)

type noopRuntime struct{}

func (noopRuntime) Enabled() bool { return false }

func (noopRuntime) APIMiddleware() []func(http.Handler) http.Handler { return nil }

func (noopRuntime) RegisterRoutes(_ *http.ServeMux, _ *http.ServeMux) {}

func (noopRuntime) Startup(context.Context) error { return nil }

func (noopRuntime) Shutdown(context.Context) error { return nil }

func newRuntime(_ *slog.Logger) (Runtime, error) {
	return noopRuntime{}, nil
}
