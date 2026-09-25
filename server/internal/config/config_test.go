package config

import "testing"

func TestLoadFromEnvReadsDenseCorePerformanceKnobs(t *testing.T) {
	t.Setenv("DENSECORE_ENGINE_THREADS", "16")
	t.Setenv("DENSECORE_GO_WORKERS", "1")
	t.Setenv("DENSECORE_SERVER_INFLIGHT", "2")
	t.Setenv("DENSECORE_KV_TYPE", "q8_0")
	t.Setenv("DENSECORE_MAX_NUM_SEQS", "16")
	t.Setenv("DENSECORE_MAX_SEQ_LEN", "16384")
	t.Setenv("DENSECORE_KV_TARGET_MB", "4096")
	t.Setenv("DENSECORE_BENCHMARK_PROFILE", "single-e2e")

	cfg, err := LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv() error = %v", err)
	}

	if cfg.EngineThreads != 16 {
		t.Fatalf("EngineThreads = %d, want 16", cfg.EngineThreads)
	}
	if cfg.GoWorkers != 1 {
		t.Fatalf("GoWorkers = %d, want 1", cfg.GoWorkers)
	}
	if cfg.ServerInflight != 2 {
		t.Fatalf("ServerInflight = %d, want 2", cfg.ServerInflight)
	}
	if cfg.KVType != "q8_0" {
		t.Fatalf("KVType = %q, want q8_0", cfg.KVType)
	}
	if cfg.MaxNumSeqs != 16 {
		t.Fatalf("MaxNumSeqs = %d, want 16", cfg.MaxNumSeqs)
	}
	if cfg.MaxSeqLen != 16384 {
		t.Fatalf("MaxSeqLen = %d, want 16384", cfg.MaxSeqLen)
	}
	if cfg.KVTargetMB != 4096 {
		t.Fatalf("KVTargetMB = %d, want 4096", cfg.KVTargetMB)
	}
	if cfg.BenchmarkProfile != "single-e2e" {
		t.Fatalf("BenchmarkProfile = %q, want single-e2e", cfg.BenchmarkProfile)
	}
}

func TestDefaultConfigDisablesWildcardCORS(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.Host != "127.0.0.1" {
		t.Fatalf("expected loopback default host, got %q", cfg.Host)
	}
	if cfg.CORSEnabled {
		t.Fatal("expected CORS to be disabled by default")
	}
	if len(cfg.CORSAllowedOrigins) != 0 {
		t.Fatalf("expected no default CORS origins, got %v", cfg.CORSAllowedOrigins)
	}
}

func TestLoadFromEnvEnablesCORSWhenOriginsProvided(t *testing.T) {
	t.Setenv("CORS_ALLOWED_ORIGINS", "https://example.com, https://admin.example.com, ")

	cfg, err := LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv() error = %v", err)
	}
	if !cfg.CORSEnabled {
		t.Fatal("expected explicit CORS origins to enable CORS")
	}
	if len(cfg.CORSAllowedOrigins) != 2 {
		t.Fatalf("expected two CORS origins, got %v", cfg.CORSAllowedOrigins)
	}
	if cfg.CORSAllowedOrigins[1] != "https://admin.example.com" {
		t.Fatalf("expected trimmed CORS origin, got %q", cfg.CORSAllowedOrigins[1])
	}
}

func TestValidateRejectsUnknownDenseCorePerformanceKnobs(t *testing.T) {
	cfg := DefaultConfig()
	cfg.KVType = "bogus"
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected invalid kv type error")
	}

	cfg = DefaultConfig()
	cfg.BenchmarkProfile = "bogus"
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected invalid benchmark profile error")
	}
}
