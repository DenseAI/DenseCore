package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"descore-server/internal/config"
)

func TestOuterRuntimeMiddlewareRunsBeforeGenericAPIMiddleware(t *testing.T) {
	var order []string
	outer := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order = append(order, "enterprise")
			next.ServeHTTP(w, r)
		})
	}
	generic := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order = append(order, "generic")
			next.ServeHTTP(w, r)
		})
	}
	setup := runtimeExtensionsBootstrap{runtime: testNoopRuntime{}, outerAPI: []func(http.Handler) http.Handler{outer}}
	middleware := buildAPIMiddleware(config.DefaultConfig(), setup, nil, nil, false)
	middleware = append(middleware, generic)
	handler := wrapWithMiddleware(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		order = append(order, "handler")
	}), middleware...)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Content-Type", "application/json")
	handler.ServeHTTP(httptest.NewRecorder(), req)
	want := []string{"enterprise", "generic", "handler"}
	if !reflect.DeepEqual(order, want) {
		t.Fatalf("middleware order = %v, want %v", order, want)
	}
}

type testNoopRuntime struct{}

func (testNoopRuntime) Enabled() bool                                    { return false }
func (testNoopRuntime) APIMiddleware() []func(http.Handler) http.Handler { return nil }
func (testNoopRuntime) RegisterRoutes(*http.ServeMux, *http.ServeMux)    {}
func (testNoopRuntime) Startup(context.Context) error                    { return nil }
func (testNoopRuntime) Shutdown(context.Context) error                   { return nil }
