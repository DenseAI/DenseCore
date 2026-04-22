package service

import "testing"

func TestDefaultModelManagerConfigKeepsAdaptiveThreadMode(t *testing.T) {
	cfg := DefaultModelManagerConfig()
	if cfg.DefaultThreads != 0 {
		t.Fatalf("expected DefaultThreads=0 for adaptive auto-detect, got %d", cfg.DefaultThreads)
	}
}

func TestNewModelManagerPreservesExplicitAutoThreadMode(t *testing.T) {
	mgr := NewModelManager(ModelManagerConfig{
		MaxModels:      1,
		DefaultThreads: 0,
		EvictionPolicy: "lru",
	})
	if mgr.config.DefaultThreads != 0 {
		t.Fatalf("expected manager to preserve DefaultThreads=0 for adaptive auto-detect, got %d", mgr.config.DefaultThreads)
	}
}
