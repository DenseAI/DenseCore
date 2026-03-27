package server

import (
	"strings"
	"testing"
)

func TestValidateMetricsPathConflict(t *testing.T) {
	tests := []struct {
		name               string
		coreMetricsEnabled bool
		coreMetricsPath    string
		runtimeEnabled     bool
		metricsPathEnv     string
		wantErr            bool
	}{
		{
			name:               "conflict with explicit enterprise path",
			coreMetricsEnabled: true,
			coreMetricsPath:    "/metrics",
			runtimeEnabled:     true,
			metricsPathEnv:     "/metrics",
			wantErr:            true,
		},
		{
			name:               "no conflict with default enterprise path",
			coreMetricsEnabled: true,
			coreMetricsPath:    "/metrics",
			runtimeEnabled:     true,
			metricsPathEnv:     "",
			wantErr:            false,
		},
		{
			name:               "no conflict when core metrics disabled",
			coreMetricsEnabled: false,
			coreMetricsPath:    "/metrics",
			runtimeEnabled:     true,
			metricsPathEnv:     "/metrics",
			wantErr:            false,
		},
		{
			name:               "no conflict when enterprise disabled",
			coreMetricsEnabled: true,
			coreMetricsPath:    "/metrics",
			runtimeEnabled:     false,
			metricsPathEnv:     "/metrics",
			wantErr:            false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("DENSECORE_ENT_METRICS_PATH", tt.metricsPathEnv)
			err := validateMetricsPathConflict(tt.coreMetricsEnabled, tt.coreMetricsPath, tt.runtimeEnabled)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected conflict error, got nil")
				}
				if !strings.Contains(err.Error(), "DENSECORE_ENT_METRICS_PATH") {
					t.Fatalf("expected error to mention DENSECORE_ENT_METRICS_PATH, got: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("expected no error, got: %v", err)
			}
		})
	}
}
