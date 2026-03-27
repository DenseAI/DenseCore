//go:build denseenterprise && densecore_cgo && cgo

package extensions

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
)

type removedEnterpriseRuntime struct{}

func (removedEnterpriseRuntime) Enabled() bool { return false }

func (removedEnterpriseRuntime) APIMiddleware() []func(http.Handler) http.Handler { return nil }

func (removedEnterpriseRuntime) RegisterRoutes(_ *http.ServeMux, _ *http.ServeMux) {}

func (removedEnterpriseRuntime) Startup(context.Context) error {
	return fmt.Errorf("denseenterprise support has been removed from this module")
}

func (removedEnterpriseRuntime) Shutdown(context.Context) error { return nil }

func newRuntime(_ *slog.Logger) (Runtime, error) {
	return removedEnterpriseRuntime{}, nil
}
