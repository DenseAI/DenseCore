package server

import (
	"encoding/json"
	"fmt"
	"net/http"

	cloudmw "github.com/DenseAI/DenseCloud/go/middleware"
	cloudserver "github.com/DenseAI/DenseCloud/go/server"
	"github.com/DenseAI/DenseCloud/go/telemetry"

	"descore-server/internal/api"
	"descore-server/internal/buildinfo"
	"descore-server/internal/config"
	"descore-server/internal/middleware"
)

// Route bootstrap ownership: root/API mux assembly and DenseCloud HTTPRuntime wiring.
type httpRuntimeBootstrap struct {
	runtime *cloudserver.HTTPRuntime
	server  *http.Server
}

func assembleHTTPRuntime(
	cfg *config.ServerConfig,
	runtimeSetup runtimeExtensionsBootstrap,
	services servicesBootstrap,
	httpMetrics *telemetry.HTTPMetrics,
	apiKeyStore middleware.APIKeyStore,
	rateLimiter cloudmw.RateLimiterInterface,
	authEnabled bool,
) (httpRuntimeBootstrap, error) {
	apiMiddleware := buildAPIMiddleware(cfg, runtimeSetup, apiKeyStore, rateLimiter, authEnabled)
	rootMux, apiMux := buildMuxes(cfg, runtimeSetup, services.handler, apiMiddleware, authEnabled)
	healthRegistry := buildDenseCoreHealthRegistry(cfg, services.modelService, apiKeyStore, rateLimiter)

	httpRuntime, err := cloudserver.NewHTTPRuntime(cloudserver.HTTPRuntimeConfig{
		ServiceName: "densecore",
		RootMux:     rootMux,
		APIMux:      apiMux,
		APIBasePath: "/v1",
		RootMiddleware: []func(http.Handler) http.Handler{
			cloudmw.RequestID(),
			cloudmw.Recovery(),
			cloudmw.RequestTimeout(cfg.RequestTimeout),
			cloudmw.Tracing("densecore-server"),
			cloudmw.Logging(),
		},
		APIMiddleware:               apiMiddleware,
		Health:                      healthRegistry,
		Metrics:                     httpMetrics,
		MetricsPath:                 cfg.MetricsPath,
		DisableMetricsRoute:         !cfg.MetricsEnabled,
		DisableRegisteredExtensions: true,
	})
	if err != nil {
		return httpRuntimeBootstrap{}, fmt.Errorf("failed to build densecloud http runtime: %w", err)
	}

	return httpRuntimeBootstrap{
		runtime: httpRuntime,
		server: &http.Server{
			Addr:         cfg.Address(),
			Handler:      httpRuntime.Handler(),
			ReadTimeout:  cfg.ReadTimeout,
			WriteTimeout: cfg.WriteTimeout,
			IdleTimeout:  cfg.IdleTimeout,
		},
	}, nil
}

func buildAPIMiddleware(
	cfg *config.ServerConfig,
	runtimeSetup runtimeExtensionsBootstrap,
	apiKeyStore middleware.APIKeyStore,
	rateLimiter cloudmw.RateLimiterInterface,
	authEnabled bool,
) []func(http.Handler) http.Handler {
	apiMiddleware := append([]func(http.Handler) http.Handler(nil), runtimeSetup.outerAPI...)
	if cfg.RateLimitEnabled && rateLimiter != nil {
		apiMiddleware = append(apiMiddleware, cloudmw.RateLimitWithInterface(rateLimiter))
	}
	if cfg.CORSEnabled {
		apiMiddleware = append(apiMiddleware, cloudmw.CORS(cfg.CORSAllowedOrigins))
	}
	if authEnabled {
		apiMiddleware = append(apiMiddleware, middleware.APIKeyAuth(apiKeyStore))
	}
	apiMiddleware = append(apiMiddleware, runtimeSetup.runtime.APIMiddleware()...)
	for _, ext := range runtimeSetup.chassisExt {
		apiMiddleware = append(apiMiddleware, ext.APIMiddleware()...)
	}
	apiMiddleware = append(apiMiddleware,
		cloudmw.MaxBodySize(cfg.MaxRequestBodySize),
		cloudmw.ContentType("application/json"),
	)
	return apiMiddleware
}

func buildMuxes(
	cfg *config.ServerConfig,
	runtimeSetup runtimeExtensionsBootstrap,
	handler *api.Handler,
	apiMiddleware []func(http.Handler) http.Handler,
	authEnabled bool,
) (*http.ServeMux, *http.ServeMux) {
	rootMux := http.NewServeMux()
	rootMux.Handle("/", makeRootHandler(rootHandlerConfig{
		ServiceName:                 cfg.ServiceName,
		ServiceDescription:          cfg.ServiceDescription,
		WorkloadProfile:             cfg.WorkloadProfile,
		LLMAPIEnabled:               cfg.LLMAPIEnabled,
		ModelLifecycleProbesEnabled: cfg.ModelLifecycleProbesEnabled,
		AuthEnabled:                 authEnabled,
		MetricsEnabled:              cfg.MetricsEnabled,
		MetricsPath:                 cfg.MetricsPath,
		Endpoints:                   buildEndpoints(cfg.LLMAPIEnabled),
	}))

	apiMux := http.NewServeMux()
	apiMux.HandleFunc("/runtime/profile", handler.RuntimeProfileHandler)
	if cfg.LLMAPIEnabled {
		apiMux.HandleFunc("/chat/completions", handler.ChatCompletionHandler)
		apiMux.HandleFunc("/completion", handler.CompletionHandler)
		apiMux.HandleFunc("/completions", handler.CompletionHandler)
		apiMux.HandleFunc("/embeddings", handler.EmbeddingsHandler)
		apiMux.HandleFunc("/rerank", handler.RerankHandler)
		apiMux.HandleFunc("/models", handler.ModelsHandler)
		apiMux.HandleFunc("/models/load", handler.LoadModelHandler)
		apiMux.HandleFunc("/models/unload", handler.UnloadModelHandler)
		rootMux.Handle("/completion", wrapWithMiddleware(http.HandlerFunc(handler.CompletionHandler), apiMiddleware...))
	}
	runtimeSetup.runtime.RegisterRoutes(rootMux, apiMux)
	for _, ext := range runtimeSetup.chassisExt {
		ext.RegisterRoutes(rootMux, apiMux)
	}
	return rootMux, apiMux
}

type rootHandlerConfig struct {
	ServiceName                 string
	ServiceDescription          string
	WorkloadProfile             string
	LLMAPIEnabled               bool
	ModelLifecycleProbesEnabled bool
	AuthEnabled                 bool
	MetricsEnabled              bool
	MetricsPath                 string
	Endpoints                   map[string]string
}

func buildEndpoints(llmAPIEnabled bool) map[string]string {
	endpoints := map[string]string{
		"health":  "/health",
		"runtime": "/v1/runtime/profile",
	}
	if llmAPIEnabled {
		endpoints["chat"] = "/v1/chat/completions"
		endpoints["completions"] = "/v1/completions"
		endpoints["embeddings"] = "/v1/embeddings"
		endpoints["rerank"] = "/v1/rerank"
		endpoints["models"] = "/v1/models"
	}
	return endpoints
}

func wrapWithMiddleware(handler http.Handler, middleware ...func(http.Handler) http.Handler) http.Handler {
	wrapped := handler
	for i := len(middleware) - 1; i >= 0; i-- {
		wrapped = middleware[i](wrapped)
	}
	return wrapped
}

func makeRootHandler(cfg rootHandlerConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		endpoints := cloneEndpoints(cfg.Endpoints)
		if len(endpoints) == 0 {
			endpoints = buildEndpoints(cfg.LLMAPIEnabled)
		}
		if cfg.MetricsEnabled {
			endpoints["metrics"] = cfg.MetricsPath
		}

		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"name":           cfg.ServiceName,
			"version":        buildinfo.Version,
			"description":    cfg.ServiceDescription,
			"profile":        cfg.WorkloadProfile,
			"llm_api":        cfg.LLMAPIEnabled,
			"authentication": cfg.AuthEnabled,
			"endpoints":      endpoints,
			"health_probes": map[string]string{
				"liveness":  "/health/live",
				"readiness": "/health/ready",
				"startup":   "/health/startup",
			},
			"probe_modes": map[string]bool{
				"model_lifecycle": cfg.ModelLifecycleProbesEnabled,
			},
		})
	}
}

func cloneEndpoints(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
