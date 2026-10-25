package cmd

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

// Only explicit flags override environment settings. Defaults remain native
// defaults so embedding and other commands do not change process configuration.
func configureSpeculationFlags(cmd *cobra.Command, _ []string) error {
	if cmd.Name() != "run" && cmd.Name() != "serve" {
		return nil
	}
	mode, err := cmd.Flags().GetString("spec-type")
	if err != nil {
		return err
	}
	mode = strings.ToLower(strings.TrimSpace(mode))
	if mode != "auto" && mode != "none" && mode != "mtp" {
		return fmt.Errorf("--spec-type must be auto, none or mtp")
	}
	limit, err := cmd.Flags().GetInt("spec-draft-n-max")
	if err != nil {
		return err
	}
	if limit < 1 || limit > 3 {
		return fmt.Errorf("--spec-draft-n-max must be 1, 2 or 3")
	}
	if cmd.Flags().Changed("spec-type") {
		if err := os.Setenv("DENSECORE_SPEC_TYPE", mode); err != nil {
			return err
		}
	}
	if cmd.Flags().Changed("spec-draft-n-max") {
		return os.Setenv("DENSECORE_SPEC_DRAFT_N_MAX", strconv.Itoa(limit))
	}
	return nil
}
