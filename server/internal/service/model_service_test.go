package service

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveModelLoadStrategy(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  modelLoadStrategy
	}{
		{name: "default", value: "", want: modelLoadStrategyAuto},
		{name: "auto", value: "auto", want: modelLoadStrategyAuto},
		{name: "blue_green", value: "blue_green", want: modelLoadStrategyBlueGreen},
		{name: "blue-green", value: "blue-green", want: modelLoadStrategyBlueGreen},
		{name: "force", value: "force", want: modelLoadStrategyForce},
		{name: "invalid", value: "nonsense", want: modelLoadStrategyAuto},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveModelLoadStrategy(tt.value); got != tt.want {
				t.Fatalf("resolveModelLoadStrategy(%q) = %v, want %v", tt.value, got, tt.want)
			}
		})
	}
}

func TestEstimateModelFileBytes(t *testing.T) {
	dir := t.TempDir()
	mainPath := filepath.Join(dir, "main.gguf")
	draftPath := filepath.Join(dir, "draft.gguf")
	if err := os.WriteFile(mainPath, []byte("12345"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(draftPath, []byte("123"), 0o600); err != nil {
		t.Fatal(err)
	}

	got, ok := estimateModelFileBytes(mainPath, draftPath)
	if !ok {
		t.Fatalf("estimateModelFileBytes returned ok=false")
	}
	if got != 8 {
		t.Fatalf("estimateModelFileBytes = %d, want 8", got)
	}

	if _, ok := estimateModelFileBytes(filepath.Join(dir, "missing.gguf")); ok {
		t.Fatalf("estimateModelFileBytes missing path returned ok=true")
	}
}

func TestLinuxMemAvailableBytes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "meminfo")
	if err := os.WriteFile(path, []byte("MemTotal:       1000 kB\nMemAvailable:    42 kB\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	got, ok := linuxMemAvailableBytes(path)
	if !ok {
		t.Fatalf("linuxMemAvailableBytes returned ok=false")
	}
	if got != 42*1024 {
		t.Fatalf("linuxMemAvailableBytes = %d, want %d", got, 42*1024)
	}
}
