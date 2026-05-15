package engine

import "testing"

func TestParseKVCacheType(t *testing.T) {
	tests := []struct {
		name       string
		value      string
		explicitKV bool
		wantErr    bool
	}{
		{name: "default empty", value: "", explicitKV: false},
		{name: "fp16", value: "fp16", explicitKV: true},
		{name: "f16 alias", value: "f16", explicitKV: true},
		{name: "q8", value: "q8_0", explicitKV: true},
		{name: "q4", value: "q4_0", explicitKV: true},
		{name: "invalid", value: "bad", wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, explicitKV, err := parseKVCacheType(tc.value)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if explicitKV != tc.explicitKV {
				t.Fatalf("explicitKV = %v, want %v", explicitKV, tc.explicitKV)
			}
		})
	}
}
