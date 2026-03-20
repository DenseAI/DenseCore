package server

import (
	"strings"
	"testing"
)

func TestValidateEnterpriseMetricsPathConflict(t *testing.T) {
	tests := []struct {
		name               string
		coreMetricsEnabled bool
		coreMetricsPath    string
		enterpriseEnabled  bool
		entMetricsPathEnv  string
		wantErr            bool
	}{
		{
			name:               "conflict with explicit enterprise path",
			coreMetricsEnabled: true,
			coreMetricsPath:    "/metrics",
			enterpriseEnabled:  true,
			entMetricsPathEnv:  "/metrics",
			wantErr:            true,
		},
		{
			name:               "no conflict with default enterprise path",
			coreMetricsEnabled: true,
			coreMetricsPath:    "/metrics",
			enterpriseEnabled:  true,
			entMetricsPathEnv:  "",
			wantErr:            false,
		},
		{
			name:               "no conflict when core metrics disabled",
			coreMetricsEnabled: false,
			coreMetricsPath:    "/metrics",
			enterpriseEnabled:  true,
			entMetricsPathEnv:  "/metrics",
			wantErr:            false,
		},
		{
			name:               "no conflict when enterprise disabled",
			coreMetricsEnabled: true,
			coreMetricsPath:    "/metrics",
			enterpriseEnabled:  false,
			entMetricsPathEnv:  "/metrics",
			wantErr:            false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("DENSECORE_ENT_METRICS_PATH", tt.entMetricsPathEnv)
			err := validateEnterpriseMetricsPathConflict(tt.coreMetricsEnabled, tt.coreMetricsPath, tt.enterpriseEnabled)
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
