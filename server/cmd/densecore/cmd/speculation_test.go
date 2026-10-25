package cmd

import (
	"os"
	"testing"

	"github.com/spf13/cobra"
)

func TestSpeculationFlags(t *testing.T) {
	for _, tc := range []struct {
		name                string
		args                []string
		wantMode, wantLimit string
		fail                bool
	}{
		{"keep environment", nil, "none", "1", false},
		{"explicit mode", []string{"--spec-type", "auto"}, "auto", "1", false},
		{"explicit limit", []string{"--spec-draft-n-max", "3"}, "none", "3", false},
		{"invalid mode", []string{"--spec-type", "draft"}, "none", "1", true},
		{"invalid limit", []string{"--spec-draft-n-max", "0"}, "none", "1", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("DENSECORE_SPEC_TYPE", "none")
			t.Setenv("DENSECORE_SPEC_DRAFT_N_MAX", "1")
			root := &cobra.Command{Use: "test", PersistentPreRunE: configureSpeculationFlags, SilenceUsage: true, SilenceErrors: true}
			root.PersistentFlags().String("spec-type", "auto", "")
			root.PersistentFlags().Int("spec-draft-n-max", 2, "")
			root.AddCommand(&cobra.Command{Use: "serve", Run: func(*cobra.Command, []string) {}})
			root.SetArgs(append([]string{"serve"}, tc.args...))
			err := root.Execute()
			if (err != nil) != tc.fail {
				t.Fatalf("error=%v, want failure=%v", err, tc.fail)
			}
			if os.Getenv("DENSECORE_SPEC_TYPE") != tc.wantMode || os.Getenv("DENSECORE_SPEC_DRAFT_N_MAX") != tc.wantLimit {
				t.Fatalf("mode=%q limit=%q", os.Getenv("DENSECORE_SPEC_TYPE"), os.Getenv("DENSECORE_SPEC_DRAFT_N_MAX"))
			}
		})
	}
}
