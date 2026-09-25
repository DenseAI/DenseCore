package server

import (
	"errors"
	"testing"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

type stubModelService struct {
	loadingErr error
}

func (s stubModelService) LoadModel(string, string, int) error        { return nil }
func (s stubModelService) UnloadModel() error                         { return nil }
func (s stubModelService) GetCurrentModel() string                    { return "" }
func (s stubModelService) GetModelIdentity() (string, string, string) { return "", "", "" }
func (s stubModelService) GetEngine() domain.Engine                   { return nil }
func (s stubModelService) GetLoadingStatus() domain.LoadingStatus     { return domain.StatusIdle }
func (s stubModelService) GetLoadingError() error                     { return s.loadingErr }
func (s stubModelService) IsLoading() bool                            { return false }

func TestModelLoadingErrorOrDefault(t *testing.T) {
	t.Run("returns loading error when present", func(t *testing.T) {
		loadErr := errors.New("load failed")

		err := modelLoadingErrorOrDefault(stubModelService{loadingErr: loadErr}, "fallback")
		if !errors.Is(err, loadErr) {
			t.Fatalf("expected loading error, got %v", err)
		}
	})

	t.Run("returns fallback when loading error missing", func(t *testing.T) {
		err := modelLoadingErrorOrDefault(stubModelService{}, "fallback")
		if err == nil || err.Error() != "fallback" {
			t.Fatalf("expected fallback error, got %v", err)
		}
	})
}
