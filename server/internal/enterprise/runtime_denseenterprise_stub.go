//go:build denseenterprise && (!densecore_cgo || !cgo)

package enterprise

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
)

type invalidRuntime struct{}

func (invalidRuntime) Enabled() bool { return false }

func (invalidRuntime) APIMiddleware() []func(http.Handler) http.Handler { return nil }

func (invalidRuntime) RegisterRoutes(_ *http.ServeMux, _ *http.ServeMux) {}

func (invalidRuntime) Startup(context.Context) error {
	return fmt.Errorf("denseenterprise build requires CGO_ENABLED=1 and -tags densecore_cgo")
}

func (invalidRuntime) Shutdown(context.Context) error { return nil }

func newRuntime(_ *slog.Logger) (Runtime, error) {
	return invalidRuntime{}, nil
}
