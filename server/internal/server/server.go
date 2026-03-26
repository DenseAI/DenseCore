package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	cloudmw "github.com/DenseAI/DenseCloud/go/middleware"
	cloudserver "github.com/DenseAI/DenseCloud/go/server"
	"github.com/DenseAI/DenseCloud/go/telemetry"

	"descore-server/internal/api"
	"descore-server/internal/buildinfo"
	"descore-server/internal/config"
	"descore-server/internal/domain"
	"descore-server/internal/engine"
	"descore-server/internal/enterprise"
	densecoregrpc "descore-server/internal/grpc"
	"descore-server/internal/middleware"
	"descore-server/internal/queue"
	"descore-server/internal/service"
	"descore-server/internal/workload"
)

const (
	Banner = `
==========================================
  DenseCore Server v%s
  Cloud-Native CPU Inference Engine
  OpenAI-Compatible REST + gRPC API
  Production-Ready with Authentication
==========================================
`
	// envTrue is the expected value for boolean environment variables
	envTrue = "true"
	// defaultEnterpriseMetricsPath is used when DENSECORE_ENT_METRICS_PATH is unset.
	defaultEnterpriseMetricsPath = "/metrics/enterprise"
	// startupRollbackTimeout bounds rollback cleanup on startup failures.
	startupRollbackTimeout = 30 * time.Second
)

// Options for starting the server.
type Options struct {
	Host         string
	Port         int
	ModelPath    string
	Verbose      bool
	LogOutput    io.Writer     // If nil, defaults to os.Stdout. Use io.Discard to silence logs.
	ShowBanner   bool          // Whether to print the ASCII banner
	Background   bool          // If true, don't wait for shutdown signals (caller manages lifecycle)
	ShutdownChan chan struct{} // Channel to signal shutdown in background mode
	AuthEnabled  bool          // Enable API key authentication

	// gRPC options. If GRPCEnabled is nil, environment config is used.
	GRPCEnabled *bool
	GRPCPort    int
}

// ServerInstance represents a running server instance for external control (TUI mode).
type ServerInstance struct {
	shutdownChan chan struct{}
	done         chan error
}

type denseCoreMetricsCollector struct {
	handler *api.Handler
}

func (c denseCoreMetricsCollector) AppendPrometheus(builder *strings.Builder) {
	if builder == nil || c.handler == nil {
		return
	}
	builder.WriteString(c.handler.RenderMetrics())
}

// Shutdown gracefully shuts down the server.
func (s *ServerInstance) Shutdown(ctx context.Context) error {
	if s == nil {
		return nil
	}
	close(s.shutdownChan)
	select {
	case err := <-s.done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Start initializes and starts the server in background mode, returning a controllable instance.
func Start(opts *Options) (*ServerInstance, error) {
	if opts == nil {
		opts = &Options{}
	}
	opts.Background = true
	shutdownChan := make(chan struct{})
	opts.ShutdownChan = shutdownChan

	done := make(chan error, 1)
	go func() {
		done <- Run(opts)
	}()

	return &ServerInstance{
		shutdownChan: shutdownChan,
		done:         done,
	}, nil
}

// Run starts the DenseCore server.
func Run(opts *Options) error {
	if opts == nil {
		opts = &Options{}
	}

	var startupRollbackHooks []cloudserver.ShutdownHook
	rollbackOnError := true
	defer func() {
		if !rollbackOnError {
			return
		}
		runStartupRollbackHooks(startupRollbackHooks)
	}()
	registerStartupRollback := func(name string, hook cloudserver.ShutdownHook) {
		startupRollbackHooks = append(startupRollbackHooks, func(ctx context.Context) error {
			if err := hook(ctx); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
			return nil
		})
	}

	logOutput := opts.LogOutput
	if logOutput == nil {
		logOutput = os.Stdout
	}

	// Determine log level from opts and config
	logLevel := "info"
	if opts.Verbose {
		logLevel = "debug"
	}

	// Load configuration
	cfg, err := config.LoadFromEnv()
	if err != nil {
		slog.Error("failed to load config", slog.String("error", err.Error()))
		return err
	}

	// Apply config log level if set
	if strings.TrimSpace(cfg.LogLevel) != "" {
		logLevel = cfg.LogLevel
	}

	// Initialize telemetry (logger setup via chassis)
	telemetry.Init(telemetry.Config{
		ServiceName: "densecore",
		Version:     buildinfo.Version,
		Level:       logLevel,
		Output:      logOutput,
	})

	// Override config with CLI options
	if opts.Host != "" {
		cfg.Host = opts.Host
	}
	if opts.Port != 0 {
		cfg.Port = opts.Port
	}
	if opts.ModelPath != "" {
		cfg.MainModelPath = opts.ModelPath
	}

	profile := workload.Resolve(cfg.WorkloadProfile)
	if strings.TrimSpace(cfg.WorkloadProfile) == "" {
		cfg.WorkloadProfile = profile.Name
	}
	if os.Getenv("LLM_API_ENABLED") == "" {
		cfg.LLMAPIEnabled = profile.EnableLLMAPI
	}
	if os.Getenv("MODEL_LIFECYCLE_PROBES_ENABLED") == "" {
		cfg.ModelLifecycleProbesEnabled = profile.EnableModelLifecycleProbes
	}
	if strings.TrimSpace(cfg.ServiceDescription) == "" {
		cfg.ServiceDescription = profile.Description
	}
	if strings.TrimSpace(cfg.ServiceName) == "" {
		cfg.ServiceName = "DenseCore API Server"
	}

	if err := cfg.Validate(); err != nil {
		slog.Error("invalid config", slog.String("error", err.Error()))
		return err
	}

	entRuntime, err := enterprise.NewRuntime(slog.Default())
	if err != nil {
		return fmt.Errorf("failed to initialize enterprise runtime: %w", err)
	}
	if err := validateEnterpriseMetricsPathConflict(cfg.MetricsEnabled, cfg.MetricsPath, entRuntime.Enabled()); err != nil {
		return err
	}
	if err := entRuntime.Startup(context.Background()); err != nil {
		return fmt.Errorf("enterprise runtime startup failed: %w", err)
	}
	registerStartupRollback("enterprise runtime", entRuntime.Shutdown)
	extensions := cloudserver.RuntimeExtensions()
	for _, ext := range extensions {
		extension := ext
		if err := extension.Startup(context.Background()); err != nil {
			return fmt.Errorf("extension startup failed (%s): %w", extension.Name(), err)
		}
		registerStartupRollback(fmt.Sprintf("extension %s", extension.Name()), extension.Shutdown)
	}

	// Initialize OpenTelemetry propagator and tracer provider.
	cloudmw.InitOTelPropagator()
	otelProvider, err := initOTelProviderFromEnv()
	if err != nil {
		return err
	}
	if otelProvider != nil {
		registerStartupRollback("otel provider", otelProvider.Shutdown)
	}

	if opts.ShowBanner {
		fmt.Printf(Banner, buildinfo.Version)
	}

	slog.Info("configuration loaded", slog.String("config", cfg.String()))

	// Detect and apply CPU configuration
	cpuCfg := engine.DetectCPUConfig()
	cpuCfg.Apply()

	threads := cfg.Threads
	if threads == 0 {
		threads = cpuCfg.OptimalThreadCount()
	}
	slog.Info("inference threads configured", slog.Int("threads", threads))

	// Initialize services
	modelService := service.NewModelService()
	registerStartupRollback("model service", func(context.Context) error {
		return modelService.UnloadModel()
	})

	// LLM profile runtime components (queue/worker/chat API) can be disabled
	// for non-LLM workloads while retaining shared cloud-native primitives.
	var requestQueue *queue.RequestQueue
	var workerPool *service.QueueProcessor
	var chatService *service.ChatService

	if cfg.LLMAPIEnabled {
		// Load initial model in background (non-blocking startup)
		if cfg.MainModelPath != "" {
			go func() {
				slog.Info("loading model in background", slog.String("model_path", cfg.MainModelPath))
				if err := modelService.LoadModel(cfg.MainModelPath, cfg.DraftModelPath, threads); err != nil {
					slog.Warn("failed to load model",
						slog.String("error", err.Error()),
						slog.String("hint", "use /v1/models/load to retry"),
					)
				} else {
					slog.Info("model loaded successfully, ready to serve requests")
				}
			}()
			slog.Info("model loading started in background, server will start immediately")
		} else {
			slog.Info("no model path specified, use /v1/models/load to load a model")
		}

		requestQueue = queue.NewRequestQueue(1024)
		workerPool = service.NewQueueProcessor(requestQueue, modelService)
		workerPool.Start(threads)
		registerStartupRollback("worker pool", func(context.Context) error {
			workerPool.Stop()
			return nil
		})
		chatService = service.NewChatService(modelService, requestQueue)
	} else {
		slog.Info("LLM API disabled by workload profile",
			slog.String("workload_profile", cfg.WorkloadProfile),
			slog.String("hint", "set LLM_API_ENABLED=true to enable OpenAI-compatible routes"),
		)
	}

	handlerOptions := []api.HandlerOption{
		api.WithWorkloadProfile(cfg.WorkloadProfile),
		api.WithMetricsNamespace(cfg.MetricsNamespace),
		api.WithModelLifecycleProbes(cfg.ModelLifecycleProbesEnabled),
		api.WithLLMAPIEnabled(cfg.LLMAPIEnabled),
	}
	if requestQueue != nil {
		handlerOptions = append(handlerOptions, api.WithQueueStatsProvider(requestQueue))
	}
	handler := api.NewHandler(chatService, modelService, handlerOptions...)
	grpcMetrics := telemetry.NewGRPCMetrics(telemetry.GRPCMetricsConfig{ServiceName: "densecore"})
	var httpMetrics *telemetry.HTTPMetrics
	if cfg.MetricsEnabled {
		httpMetrics = telemetry.NewHTTPMetrics(telemetry.HTTPMetricsConfig{
			ServiceName: "densecore",
			IgnorePaths: []string{
				"/health",
				"/health/live",
				"/health/ready",
				"/health/startup",
				cfg.MetricsPath,
			},
			Collectors: []telemetry.PrometheusCollector{
				grpcMetrics,
				denseCoreMetricsCollector{handler: handler},
			},
		})
	}

	// Initialize API key store
	var apiKeyStore middleware.APIKeyStore
	authEnabled := opts.AuthEnabled || os.Getenv("AUTH_ENABLED") == envTrue
	if authEnabled {
		slog.Info("authentication enabled")
		redisURL := os.Getenv("REDIS_URL")
		redisKeystoreEnabled := os.Getenv("REDIS_KEYSTORE_ENABLED") == envTrue

		if redisURL != "" && redisKeystoreEnabled {
			cacheTTL, cacheSize, redisDB := parseRedisKeyStoreConfig()
			redisKeyStore, err := middleware.NewRedisKeyStore(middleware.RedisKeyStoreConfig{
				RedisURL:      redisURL,
				RedisPassword: os.Getenv("REDIS_PASSWORD"),
				RedisDB:       redisDB,
				CacheTTL:      cacheTTL,
				CacheSize:     cacheSize,
			})
			if err != nil {
				slog.Warn("failed to connect to Redis for key store, falling back to in-memory",
					slog.String("error", err.Error()))
			} else {
				slog.Info("using Redis key store for distributed API key management")
				apiKeyStore = redisKeyStore
			}
		}

		if apiKeyStore == nil {
			keyStore := middleware.NewInMemoryKeyStore()
			apiKeysEnv := os.Getenv("API_KEYS")
			if apiKeysEnv != "" {
				loadAPIKeys(keyStore, apiKeysEnv)
			} else {
				return fmt.Errorf("AUTH_ENABLED=true but API_KEYS is empty and Redis keystore is not configured")
			}
			apiKeyStore = keyStore
		}
	} else {
		slog.Info("authentication disabled", slog.String("hint", "set AUTH_ENABLED=true to enable"))
	}

	// Create rate limiter (Redis or in-memory)
	var rateLimiter cloudmw.RateLimiterInterface
	if cfg.RateLimitEnabled {
		redisURL := os.Getenv("REDIS_URL")
		redisRateLimitEnabled := os.Getenv("REDIS_RATELIMIT_ENABLED") == envTrue

		if redisURL != "" && redisRateLimitEnabled {
			redisDB := parseRedisDB()
			redisRateLimiter, err := middleware.NewRedisRateLimiter(middleware.RedisRateLimiterConfig{
				RedisURL:          redisURL,
				RedisPassword:     os.Getenv("REDIS_PASSWORD"),
				RedisDB:           redisDB,
				RequestsPerSecond: cfg.RateLimitReqPerSec,
				Burst:             cfg.RateLimitBurst,
				FailureThreshold:  3,
				ResetTimeout:      30 * time.Second,
			})
			if err != nil {
				slog.Warn("failed to connect to Redis for rate limiting, falling back to in-memory",
					slog.String("error", err.Error()))
				rateLimiter = cloudmw.NewRateLimiter(cfg.RateLimitReqPerSec, cfg.RateLimitBurst)
			} else {
				slog.Info("using Redis rate limiter for distributed rate limiting")
				rateLimiter = redisRateLimiter
			}
		} else {
			rateLimiter = cloudmw.NewRateLimiter(cfg.RateLimitReqPerSec, cfg.RateLimitBurst)
		}
	}

	apiMiddleware := []func(http.Handler) http.Handler{}
	if cfg.RateLimitEnabled && rateLimiter != nil {
		apiMiddleware = append(apiMiddleware, cloudmw.RateLimitWithInterface(rateLimiter))
	}
	if cfg.CORSEnabled {
		apiMiddleware = append(apiMiddleware, cloudmw.CORS(cfg.CORSAllowedOrigins))
	}
	if authEnabled {
		apiMiddleware = append(apiMiddleware, middleware.APIKeyAuth(apiKeyStore))
	}
	apiMiddleware = append(apiMiddleware, entRuntime.APIMiddleware()...)
	for _, ext := range extensions {
		apiMiddleware = append(apiMiddleware, ext.APIMiddleware()...)
	}
	apiMiddleware = append(apiMiddleware,
		cloudmw.MaxBodySize(cfg.MaxRequestBodySize),
		cloudmw.ContentType("application/json"),
	)

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
		apiMux.HandleFunc("/embeddings", handler.EmbeddingsHandler)
		apiMux.HandleFunc("/rerank", handler.RerankHandler)
		apiMux.HandleFunc("/models", handler.ModelsHandler)
		apiMux.HandleFunc("/models/load", handler.LoadModelHandler)
		apiMux.HandleFunc("/models/unload", handler.UnloadModelHandler)
	}
	entRuntime.RegisterRoutes(rootMux, apiMux)
	for _, ext := range extensions {
		ext.RegisterRoutes(rootMux, apiMux)
	}
	healthRegistry := buildDenseCoreHealthRegistry(cfg, modelService, apiKeyStore, rateLimiter)
	httpRuntime, err := cloudserver.NewHTTPRuntime(cloudserver.HTTPRuntimeConfig{
		ServiceName: "densecore",
		RootMux:     rootMux,
		APIMux:      apiMux,
		APIBasePath: "/v1",
		RootMiddleware: []func(http.Handler) http.Handler{
			cloudmw.Recovery(),
			cloudmw.RequestID(),
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
		return fmt.Errorf("failed to build densecloud http runtime: %w", err)
	}

	httpServer := &http.Server{
		Addr:         cfg.Address(),
		Handler:      httpRuntime.Handler(),
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	grpcConfig := densecoregrpc.DefaultConfig()
	grpcConfig.Port = getEnvOrDefaultInt("GRPC_PORT", 50051)
	if opts.GRPCPort > 0 {
		grpcConfig.Port = opts.GRPCPort
	}
	grpcConfig.Enabled = profile.EnableGRPC && cfg.LLMAPIEnabled
	if os.Getenv("GRPC_ENABLED") != "" {
		grpcConfig.Enabled = parseEnvBool("GRPC_ENABLED", grpcConfig.Enabled)
	}
	if opts.GRPCEnabled != nil {
		grpcConfig.Enabled = *opts.GRPCEnabled
	}
	if !cfg.LLMAPIEnabled && grpcConfig.Enabled {
		slog.Warn("disabling gRPC because LLM API is disabled for current profile")
		grpcConfig.Enabled = false
	}
	grpcConfig.AuthEnabled = authEnabled
	grpcConfig.APIKeyStore = apiKeyStore
	grpcConfig.TLSEnabled = parseEnvBool("GRPC_TLS_ENABLED", false)
	grpcConfig.TLSCertFile = getEnvOrDefault("GRPC_TLS_CERT_FILE", "")
	grpcConfig.TLSKeyFile = getEnvOrDefault("GRPC_TLS_KEY_FILE", "")
	grpcConfig.TLSClientCAFile = getEnvOrDefault("GRPC_TLS_CLIENT_CA_FILE", "")
	grpcConfig.TLSRequireClientCert = parseEnvBool("GRPC_TLS_REQUIRE_CLIENT_CERT", false)
	grpcConfig.SharedMetrics = grpcMetrics

	var grpcServer *densecoregrpc.Server
	if cfg.LLMAPIEnabled {
		grpcServer = densecoregrpc.NewServer(chatService, modelService, grpcConfig)
	}

	// Build shutdown hooks for domain resources
	var shutdownHooks []cloudserver.ShutdownHook
	if workerPool != nil {
		shutdownHooks = append(shutdownHooks, func(context.Context) error {
			workerPool.Stop()
			return nil
		})
	}
	if modelService != nil {
		shutdownHooks = append(shutdownHooks, func(context.Context) error {
			return modelService.UnloadModel()
		})
	}
	if otelProvider != nil {
		shutdownHooks = append(shutdownHooks, func(ctx context.Context) error {
			return otelProvider.Shutdown(ctx)
		})
	}
	shutdownHooks = append(shutdownHooks, func(ctx context.Context) error {
		return entRuntime.Shutdown(ctx)
	})
	for _, ext := range extensions {
		extension := ext
		shutdownHooks = append(shutdownHooks, func(ctx context.Context) error {
			return extension.Shutdown(ctx)
		})
	}

	// Background mode: create context from ShutdownChan
	var ctx context.Context
	if opts.Background {
		if opts.ShutdownChan == nil {
			return fmt.Errorf("background mode requires ShutdownChan")
		}
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(context.Background())
		go func() {
			<-opts.ShutdownChan
			cancel()
		}()
	} else {
		ctx = context.Background()
	}

	// Use chassis Runner for server lifecycle
	var cloudGRPC cloudserver.GRPCServer
	if grpcServer != nil {
		cloudGRPC = grpcServer
	}

	runner, err := cloudserver.NewRunner(cloudserver.Options{
		HTTPServer:      httpServer,
		GRPCServer:      cloudGRPC,
		EnableGRPC:      grpcConfig.Enabled,
		ShutdownTimeout: 30 * time.Second,
		StartupHooks: []cloudserver.StartupHook{
			httpRuntime.Startup,
		},
		ShutdownHooks: append([]cloudserver.ShutdownHook{
			httpRuntime.Shutdown,
		}, shutdownHooks...),
	})
	if err != nil {
		return fmt.Errorf("failed to create server runner: %w", err)
	}
	rollbackOnError = false

	slog.Info("starting DenseCore server",
		slog.String("address", cfg.Address()),
		slog.Bool("auth_enabled", authEnabled),
		slog.Bool("grpc_enabled", grpcConfig.Enabled),
	)

	return runner.RunBlocking(ctx)
}

func runStartupRollbackHooks(hooks []cloudserver.ShutdownHook) {
	if len(hooks) == 0 {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), startupRollbackTimeout)
	defer cancel()

	for i := len(hooks) - 1; i >= 0; i-- {
		hook := hooks[i]
		if hook == nil {
			continue
		}
		if err := hook(ctx); err != nil {
			slog.Warn("startup rollback hook failed", slog.String("error", err.Error()))
		}
	}
}

func validateEnterpriseMetricsPathConflict(coreMetricsEnabled bool, coreMetricsPath string, enterpriseEnabled bool) error {
	if !coreMetricsEnabled || !enterpriseEnabled {
		return nil
	}

	entMetricsPath := strings.TrimSpace(os.Getenv("DENSECORE_ENT_METRICS_PATH"))
	if entMetricsPath == "" {
		entMetricsPath = defaultEnterpriseMetricsPath
	}
	coreMetricsPath = strings.TrimSpace(coreMetricsPath)
	if entMetricsPath == coreMetricsPath {
		return fmt.Errorf(
			"invalid metrics configuration: DENSECORE_ENT_METRICS_PATH (%q) conflicts with METRICS_PATH (%q)",
			entMetricsPath, coreMetricsPath,
		)
	}
	return nil
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
		endpoints["embeddings"] = "/v1/embeddings"
		endpoints["rerank"] = "/v1/rerank"
		endpoints["models"] = "/v1/models"
	}
	return endpoints
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

func loadAPIKeys(store *middleware.InMemoryKeyStore, apiKeysEnv string) {
	if apiKeysEnv == "" {
		return
	}

	keys := strings.Split(apiKeysEnv, ",")
	for _, keyData := range keys {
		parts := strings.Split(strings.TrimSpace(keyData), ":")
		if len(parts) < 3 {
			slog.Warn("invalid API key format", slog.String("hint", "expected format: key:user:tier"))
			continue
		}

		key := strings.TrimSpace(parts[0])
		userID := strings.TrimSpace(parts[1])
		tier := strings.TrimSpace(parts[2])

		store.AddKey(key, userID, tier)
		slog.Info("loaded API key", slog.String("user_id", userID), slog.String("tier", tier))
	}
}

func parseRedisKeyStoreConfig() (time.Duration, int, int) {
	cacheTTLStr := getEnvOrDefault("REDIS_KEYSTORE_CACHE_TTL", "5m")
	cacheTTL, err := time.ParseDuration(cacheTTLStr)
	if err != nil {
		slog.Warn("invalid REDIS_KEYSTORE_CACHE_TTL, using default",
			slog.String("value", cacheTTLStr),
			slog.String("default", "5m"),
			slog.String("error", err.Error()))
		cacheTTL = 5 * time.Minute
	}

	cacheSizeStr := getEnvOrDefault("REDIS_KEYSTORE_CACHE_SIZE", "1000")
	cacheSize, err := strconv.Atoi(cacheSizeStr)
	if err != nil {
		slog.Warn("invalid REDIS_KEYSTORE_CACHE_SIZE, using default",
			slog.String("value", cacheSizeStr),
			slog.String("default", "1000"),
			slog.String("error", err.Error()))
		cacheSize = 1000
	}

	return cacheTTL, cacheSize, parseRedisDB()
}

func parseRedisDB() int {
	redisDBStr := getEnvOrDefault("REDIS_DB", "0")
	redisDB, err := strconv.Atoi(redisDBStr)
	if err != nil {
		slog.Warn("invalid REDIS_DB, using default",
			slog.String("value", redisDBStr),
			slog.String("default", "0"),
			slog.String("error", err.Error()))
		return 0
	}
	return redisDB
}

func initOTelProviderFromEnv() (*telemetry.OTelProvider, error) {
	enabled := parseEnvBool("OTEL_ENABLED", false)
	if !enabled {
		return nil, nil
	}

	cfg := telemetry.DefaultOTelConfig()
	cfg.Enabled = true
	cfg.Endpoint = getEnvOrDefault("OTEL_ENDPOINT", cfg.Endpoint)
	cfg.ServiceName = getEnvOrDefault("OTEL_SERVICE_NAME", "densecore-server")
	cfg.ServiceVersion = buildinfo.Version
	cfg.Insecure = parseEnvBool("OTEL_INSECURE", cfg.Insecure)

	if sampling := os.Getenv("OTEL_SAMPLING_RATE"); sampling != "" {
		if v, err := strconv.ParseFloat(sampling, 64); err == nil {
			cfg.SamplingRate = v
		} else {
			slog.Warn("invalid OTEL_SAMPLING_RATE, using default",
				slog.String("value", sampling),
				slog.Float64("default", cfg.SamplingRate),
				slog.String("error", err.Error()))
		}
	}

	provider, err := telemetry.InitOTelTracer(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize OpenTelemetry: %w", err)
	}
	return provider, nil
}

func parseEnvBool(key string, defaultVal bool) bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv(key)))
	if v == "" {
		return defaultVal
	}
	return v == "1" || v == envTrue || v == "yes" || v == "on"
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func buildDenseCoreHealthRegistry(cfg *config.ServerConfig, modelService domain.ModelService, apiKeyStore middleware.APIKeyStore, rateLimiter cloudmw.RateLimiterInterface) *cloudserver.HealthRegistry {
	registry := cloudserver.NewHealthRegistry()

	if cfg.ModelLifecycleProbesEnabled {
		registry.RegisterStartup("model_loaded", func(context.Context) error {
			switch modelService.GetLoadingStatus() {
			case domain.StatusReady:
				return nil
			case domain.StatusLoading:
				return errors.New("model loading")
			case domain.StatusFailed:
				return modelLoadingErrorOrDefault(modelService, "model load failed")
			default:
				if modelService.GetEngine() != nil {
					return nil
				}
				return errors.New("no model configured")
			}
		})
		registry.RegisterReadiness("model_ready", func(context.Context) error {
			if modelService.IsLoading() {
				return errors.New("model loading")
			}
			if modelService.GetEngine() == nil {
				return modelLoadingErrorOrDefault(modelService, "model not loaded")
			}
			return nil
		})
	}

	registerHealthDependency(registry, "api_key_store", apiKeyStore)
	registerHealthDependency(registry, "rate_limiter", rateLimiter)
	return registry
}

func modelLoadingErrorOrDefault(modelService domain.ModelService, defaultMessage string) error {
	if err := modelService.GetLoadingError(); err != nil {
		return err
	}
	return errors.New(defaultMessage)
}

func registerHealthDependency(registry *cloudserver.HealthRegistry, name string, dependency any) {
	if registry == nil || dependency == nil {
		return
	}
	healthDependency, ok := dependency.(interface {
		HealthCheck(context.Context) error
	})
	if !ok {
		return
	}
	registry.RegisterDependency(name, healthDependency, cloudserver.HealthPhaseStartup, cloudserver.HealthPhaseReady)
}

func getEnvOrDefaultInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.Atoi(value); err == nil {
			return i
		}
	}
	return defaultValue
}
