package workload

import "strings"

const (
	ProfileLLM     = "llm"
	ProfileGeneric = "generic"
)

// Profile defines workload-level runtime behavior.
type Profile struct {
	Name                       string
	Description                string
	EnableLLMAPI               bool
	EnableModelLifecycleProbes bool
	EnableGRPC                 bool
}

// Resolve returns a built-in workload profile.
// Unknown values fall back to LLM for backward compatibility.
func Resolve(name string) Profile {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case ProfileGeneric:
		return Profile{
			Name:                       ProfileGeneric,
			Description:                "Cloud-native inference runtime (workload-agnostic profile)",
			EnableLLMAPI:               false,
			EnableModelLifecycleProbes: false,
			EnableGRPC:                 false,
		}
	default:
		return Profile{
			Name:                       ProfileLLM,
			Description:                "Cloud-native CPU inference engine with OpenAI-compatible API",
			EnableLLMAPI:               true,
			EnableModelLifecycleProbes: true,
			EnableGRPC:                 true,
		}
	}
}
