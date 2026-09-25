package server

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	cloudserver "github.com/DenseAI/DenseCloud/go/server"

	"github.com/DenseAI/DenseCore/server/internal/api"
	"github.com/DenseAI/DenseCore/server/internal/config"
	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/engine"
	"github.com/DenseAI/DenseCore/server/internal/queue"
	"github.com/DenseAI/DenseCore/server/internal/service"
)

// Service bootstrap ownership: CPU tuning, model service, queue processor, and chat API assembly.
type runtimeTuning struct {
	threads   int
	goWorkers int
}

type servicesBootstrap struct {
	modelService *service.ModelService
	requestQueue *queue.RequestQueue
	workerPool   *service.QueueProcessor
	chatService  *service.ChatService
	handler      *api.Handler
}

func configureRuntimeTuning(cfg *config.ServerConfig) runtimeTuning {
	cpuCfg := engine.DetectCPUConfig()
	cpuCfg.Apply()
	applyBenchmarkProfileDefaults(cfg, cpuCfg)

	threads := cfg.EngineThreads
	if threads == 0 {
		threads = cfg.Threads
	}
	if threads == 0 {
		threads = cpuCfg.OptimalThreadCount()
	}
	goWorkers := cfg.GoWorkers
	if goWorkers == 0 {
		goWorkers = defaultGoWorkersForEngineCapacity(threads, cfg.MaxNumSeqs)
	}
	slog.Info("runtime tuning configured",
		slog.String("benchmark_profile", cfg.BenchmarkProfile),
		slog.Int("engine_threads", threads),
		slog.Int("go_workers", goWorkers),
		slog.Int("server_inflight", cfg.ServerInflight),
		slog.String("kv_type", cfg.KVType),
		slog.Int("max_num_seqs", cfg.MaxNumSeqs),
		slog.Int("max_seq_len", cfg.MaxSeqLen),
		slog.Int("kv_target_mb", cfg.KVTargetMB),
	)

	return runtimeTuning{threads: threads, goWorkers: goWorkers}
}

func assembleServices(cfg *config.ServerConfig, tuning runtimeTuning, usageRecorder api.InferenceUsageRecorder, registerRollback func(string, cloudserver.ShutdownHook)) (servicesBootstrap, error) {
	modelService := service.NewModelService()
	registerRollback("model service", func(ctx context.Context) error {
		return modelService.UnloadModelContext(ctx)
	})

	var requestQueue *queue.RequestQueue
	var workerPool *service.QueueProcessor
	var chatService *service.ChatService

	if cfg.LLMAPIEnabled {
		if err := loadStartupModel(cfg, modelService, tuning.threads); err != nil {
			return servicesBootstrap{}, err
		}
		requestQueue = queue.NewRequestQueue(cfg.ServerInflight)
		workerPool = service.NewQueueProcessor(requestQueue, modelService)
		workerPool.Start(tuning.goWorkers)
		registerRollback("worker pool", func(ctx context.Context) error {
			return workerPool.StopAndWait(ctx)
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
		api.WithRuntimeTuningProfile(api.RuntimeTuningProfile{
			BenchmarkProfile:         cfg.BenchmarkProfile,
			EngineThreads:            tuning.threads,
			GoWorkers:                tuning.goWorkers,
			ServerInflight:           cfg.ServerInflight,
			KVType:                   cfg.KVType,
			MaxNumSeqs:               cfg.MaxNumSeqs,
			MaxSeqLen:                cfg.MaxSeqLen,
			KVTargetMB:               cfg.KVTargetMB,
			TokenIDSubmit:            true,
			PrefixCacheAuto:          parseEnvBool("DENSECORE_PREFIX_CACHE_REUSE", true),
			CacheAffinityHeader:      api.CacheAffinityRequestHeader,
			CacheAffinityEnabled:     cfg.LLMAPIEnabled,
			SSMSnapshotAuto:          parseEnvBool("DENSECORE_HYBRID_SSM_SNAPSHOT_RESTORE", true),
			PrefillArenaReuse:        !envFlagEnabled("DENSECORE_DEBUG_DISABLE_GRAPH_CACHE_REUSE"),
			DecodeGraphCacheMaxBatch: envIntDefault("DENSECORE_DECODE_GRAPH_CACHE_MAX_BATCH", minInt(maxInt(cfg.MaxNumSeqs, 16), 32)),
			MoEDequantCacheMB:        envIntDefault("DENSECORE_MOE_DEQUANT_CACHE_MB", 512),
		}),
	}
	if requestQueue != nil {
		handlerOptions = append(handlerOptions, api.WithQueueStatsProvider(requestQueue))
	}
	if usageRecorder != nil {
		handlerOptions = append(handlerOptions, api.WithInferenceUsageRecorder(usageRecorder))
	}

	return servicesBootstrap{
		modelService: modelService,
		requestQueue: requestQueue,
		workerPool:   workerPool,
		chatService:  chatService,
		handler:      api.NewHandler(chatService, modelService, handlerOptions...),
	}, nil
}

func loadStartupModel(cfg *config.ServerConfig, modelService domain.ModelService, threads int) error {
	if cfg.MainModelPath == "" {
		return fmt.Errorf("fixed-model startup requires MAIN_MODEL_PATH or --model")
	}

	slog.Info("loading startup model", slog.String("model_path", cfg.MainModelPath))
	if err := modelService.LoadModel(cfg.MainModelPath, cfg.DraftModelPath, threads); err != nil {
		return fmt.Errorf("load startup model: %w", err)
	}

	executablePath, execErr := os.Executable()
	if execErr != nil {
		executablePath = ""
	}
	modelPath := cfg.MainModelPath
	if absPath, err := filepath.Abs(modelPath); err == nil {
		modelPath = absPath
	}
	tokenizerType := ""
	if loadedEngine := modelService.GetEngine(); loadedEngine != nil {
		tokenizerType = loadedEngine.GetTokenizerType()
	}
	slog.Info("startup model loaded; server may accept requests",
		slog.String("server_binary_path", executablePath),
		slog.String("densecore_shared_library_path", engine.ResolvedDenseCoreLibraryPath()),
		slog.String("model_path", modelPath),
		slog.String("tokenizer_type", tokenizerType),
		slog.Bool("gemma4_path", strings.EqualFold(tokenizerType, "gemma4")),
	)
	return nil
}

func applyBenchmarkProfileDefaults(cfg *config.ServerConfig, cpuCfg *engine.CPUConfig) {
	if cfg == nil {
		return
	}

	switch cfg.BenchmarkProfile {
	case "single-e2e":
		if strings.TrimSpace(os.Getenv("DENSECORE_ENGINE_THREADS")) == "" && cfg.EngineThreads <= 0 && cpuCfg != nil {
			cfg.EngineThreads = cpuCfg.OptimalThreadCount()
			if cfg.EngineThreads > 16 {
				cfg.EngineThreads = 16
			}
		}
		if strings.TrimSpace(os.Getenv("DENSECORE_GO_WORKERS")) == "" && cfg.GoWorkers <= 0 {
			cfg.GoWorkers = 1
		}
		if strings.TrimSpace(os.Getenv("DENSECORE_SERVER_INFLIGHT")) == "" && cfg.ServerInflight == 1024 {
			cfg.ServerInflight = 1
		}
	}
}

func buildShutdownHooks(services servicesBootstrap, otelProvider interface {
	Shutdown(context.Context) error
}, runtimeSetup runtimeExtensionsBootstrap) []cloudserver.ShutdownHook {
	var shutdownHooks []cloudserver.ShutdownHook
	if services.workerPool != nil {
		shutdownHooks = append(shutdownHooks, func(ctx context.Context) error {
			return services.workerPool.StopAndWait(ctx)
		})
	}
	if services.modelService != nil {
		shutdownHooks = append(shutdownHooks, func(ctx context.Context) error {
			return services.modelService.UnloadModelContext(ctx)
		})
	}
	if otelProvider != nil {
		shutdownHooks = append(shutdownHooks, func(ctx context.Context) error {
			return otelProvider.Shutdown(ctx)
		})
	}
	shutdownHooks = append(shutdownHooks, func(ctx context.Context) error {
		return runtimeSetup.runtime.Shutdown(ctx)
	})
	for _, ext := range runtimeSetup.chassisExt {
		extension := ext
		shutdownHooks = append(shutdownHooks, func(ctx context.Context) error {
			return extension.Shutdown(ctx)
		})
	}
	return shutdownHooks
}

func buildPreShutdownHooks(services servicesBootstrap) []cloudserver.ShutdownHook {
	if services.modelService == nil {
		return nil
	}
	return []cloudserver.ShutdownHook{func(context.Context) error {
		services.modelService.BeginEngineShutdown()
		return nil
	}}
}
