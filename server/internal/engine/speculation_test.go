package engine

import "testing"

func TestSpeculationOptions(t *testing.T) {
	for _, tc := range []struct {
		name, mode, limit   string
		wantMode, wantLimit int
		fail                bool
	}{
		{"defaults", "", "2", 0, 2, false},
		{"none", "none", "1", 1, 1, false},
		{"mtp", "mtp", "3", 2, 3, false},
		{"normalize", " AUTO ", "2", 0, 2, false},
		{"unknown mode", "draft", "2", 0, 0, true},
		{"zero draft", "auto", "0", 0, 0, true},
		{"large draft", "auto", "4", 0, 0, true},
		{"malformed draft", "auto", "two", 0, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("DENSECORE_SPEC_TYPE", tc.mode)
			t.Setenv("DENSECORE_SPEC_DRAFT_N_MAX", tc.limit)
			got, err := speculationOptionsFromEnv()
			if (err != nil) != tc.fail {
				t.Fatalf("options=%+v err=%v", got, err)
			}
			if !tc.fail && (got.mode != tc.wantMode || got.draftLimit != tc.wantLimit) {
				t.Fatalf("options=%+v, want mode=%d limit=%d", got, tc.wantMode, tc.wantLimit)
			}
		})
	}
}
