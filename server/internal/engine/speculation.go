package engine

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

type speculationOptions struct {
	name       string
	mode       int
	draftLimit int
}

func speculationOptionsFromEnv() (speculationOptions, error) {
	name := strings.ToLower(strings.TrimSpace(os.Getenv("DENSECORE_SPEC_TYPE")))
	if name == "" {
		name = "auto"
	}
	result := speculationOptions{name: name, draftLimit: 2}
	switch name {
	case "auto":
		result.mode = 0
	case "none":
		result.mode = 1
	case "mtp":
		result.mode = 2
	default:
		return result, fmt.Errorf("DENSECORE_SPEC_TYPE must be auto, none or mtp, got %q", name)
	}
	if raw, ok := os.LookupEnv("DENSECORE_SPEC_DRAFT_N_MAX"); ok {
		limit, err := strconv.Atoi(strings.TrimSpace(raw))
		if err != nil || limit < 1 || limit > 3 {
			return result, fmt.Errorf("DENSECORE_SPEC_DRAFT_N_MAX must be 1, 2 or 3, got %q", raw)
		}
		result.draftLimit = limit
	}
	return result, nil
}
