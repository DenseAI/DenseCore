package engine

import "testing"

func TestOptimalThreadCountUsesFullDetectedBudget(t *testing.T) {
	cfg := &CPUConfig{NumThreads: 16}
	if got := cfg.OptimalThreadCount(); got != 16 {
		t.Fatalf("OptimalThreadCount() = %d, want 16", got)
	}
}

func TestOptimalThreadCountNeverDropsBelowOne(t *testing.T) {
	cfg := &CPUConfig{NumThreads: 0}
	if got := cfg.OptimalThreadCount(); got != 1 {
		t.Fatalf("OptimalThreadCount() = %d, want 1", got)
	}
}
