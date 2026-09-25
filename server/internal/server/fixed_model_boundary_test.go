package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/DenseAI/DenseCore/server/internal/api"
	"github.com/DenseAI/DenseCore/server/internal/config"
	"github.com/DenseAI/DenseCore/server/internal/domain"
)

type fixedModelTestRuntime struct{}

func (fixedModelTestRuntime) Enabled() bool { return false }

func (fixedModelTestRuntime) APIMiddleware() []func(http.Handler) http.Handler { return nil }

func (fixedModelTestRuntime) RegisterRoutes(_ *http.ServeMux, _ *http.ServeMux) {}

func (fixedModelTestRuntime) Startup(context.Context) error { return nil }

func (fixedModelTestRuntime) Shutdown(context.Context) error { return nil }

type fixedModelTestModelService struct{}

func (fixedModelTestModelService) LoadModel(string, string, int) error { return nil }
func (fixedModelTestModelService) UnloadModel() error                  { return nil }
func (fixedModelTestModelService) GetCurrentModel() string             { return "" }
func (fixedModelTestModelService) GetModelIdentity() (string, string, string) {
	return "densecore-v1", "densecore", ""
}
func (fixedModelTestModelService) GetEngine() domain.Engine               { return nil }
func (fixedModelTestModelService) GetLoadingStatus() domain.LoadingStatus { return domain.StatusReady }
func (fixedModelTestModelService) GetLoadingError() error                 { return nil }
func (fixedModelTestModelService) IsLoading() bool                        { return false }

type failingStartupModelService struct {
	fixedModelTestModelService
}

func (failingStartupModelService) LoadModel(string, string, int) error {
	return errors.New("invalid model artifact")
}

func TestValidateFixedModelServeConfigRequiresStartupModel(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.MainModelPath = ""

	err := validateFixedModelServeConfig(cfg)
	if err == nil {
		t.Fatal("expected missing startup model to be rejected")
	}
	if !strings.Contains(err.Error(), "startup model") {
		t.Fatalf("expected startup model error, got %v", err)
	}
}

func TestValidateFixedModelServeConfigRejectsDraftModel(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.MainModelPath = "/models/main.gguf"
	cfg.DraftModelPath = "/models/draft.gguf"

	err := validateFixedModelServeConfig(cfg)
	if err == nil {
		t.Fatal("expected draft model config to be rejected")
	}
	if !strings.Contains(err.Error(), "draft_model_path") {
		t.Fatalf("expected draft model error, got %v", err)
	}
}

func TestValidateFixedModelServeConfigRejectsCoreDraftModelEnv(t *testing.T) {
	t.Setenv("DENSECORE_DRAFT_MODEL_PATH", "/models/draft.gguf")
	cfg := config.DefaultConfig()
	cfg.MainModelPath = "/models/main.gguf"
	cfg.DraftModelPath = ""

	if err := validateFixedModelServeConfig(cfg); err == nil {
		t.Fatal("expected DENSECORE_DRAFT_MODEL_PATH to be rejected")
	}
}

func TestLoadStartupModelPropagatesInitialLoadFailure(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.MainModelPath = "/models/invalid.gguf"

	err := loadStartupModel(cfg, failingStartupModelService{}, 4)
	if err == nil {
		t.Fatal("expected initial model load failure")
	}
	if !strings.Contains(err.Error(), "invalid model artifact") {
		t.Fatalf("expected model load cause, got %v", err)
	}
}

func TestLoadBootstrapConfigAppliesCLIThreadOverride(t *testing.T) {
	cfg, _, _, err := loadBootstrapConfig(&Options{Threads: 12})
	if err != nil {
		t.Fatalf("loadBootstrapConfig() error = %v", err)
	}
	if cfg.Threads != 12 {
		t.Fatalf("Threads = %d, want 12", cfg.Threads)
	}
}

func TestBuildMuxesDoesNotMountModelLifecycleRoutes(t *testing.T) {
	cfg := config.DefaultConfig()
	handler := api.NewHandler(nil, fixedModelTestModelService{})
	_, apiMux := buildMuxes(
		cfg,
		runtimeExtensionsBootstrap{runtime: fixedModelTestRuntime{}},
		handler,
		nil,
		false,
	)

	t.Run("models remains mounted", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/models", nil)
		w := httptest.NewRecorder()
		apiMux.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("expected /models status 200, got %d", w.Code)
		}
	})

	for _, path := range []string{"/models/load", "/models/unload"} {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, path, nil)
			w := httptest.NewRecorder()
			apiMux.ServeHTTP(w, req)
			if w.Code != http.StatusNotFound {
				t.Fatalf("expected %s status 404, got %d", path, w.Code)
			}
		})
	}
}
