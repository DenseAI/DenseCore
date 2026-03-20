package api

import "testing"

func TestSanitizeMetricsNamespace(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "empty uses default",
			in:   "",
			want: DefaultMetricsNamespace,
		},
		{
			name: "normalizes case and punctuation",
			in:   "Dense-Core Metrics",
			want: "dense_core_metrics",
		},
		{
			name: "keeps valid underscore prefix for leading digit",
			in:   "1core",
			want: "_1core",
		},
		{
			name: "invalid only falls back to default",
			in:   "---",
			want: DefaultMetricsNamespace,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sanitizeMetricsNamespace(tt.in)
			if got != tt.want {
				t.Fatalf("sanitizeMetricsNamespace(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
