package service

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"descore-server/internal/domain"
	"descore-server/internal/util"
)

type promptProfileKind int

const (
	promptProfileKindGenericTranscript promptProfileKind = iota
	promptProfileKindChatML
	promptProfileKindTurnTags
)

type promptFamily int

const (
	promptFamilyGeneric promptFamily = iota
	promptFamilyQwen
	promptFamilyGemma
)

type promptProfile struct {
	family              promptFamily
	kind                promptProfileKind
	userRole            string
	assistantRole       string
	systemRole          string
	openTag             string
	closeTag            string
	defaultSystemPrompt string
	imageToken          string
	videoToken          string
	audioToken          string
}

type promptProfileFile struct {
	Profiles []promptProfileSpec `json:"profiles"`
}

type promptProfileSpec struct {
	Family          string   `json:"family"`
	Kind            string   `json:"kind"`
	MatchSubstrings []string `json:"match_substrings"`
	DefaultSystem   string   `json:"default_system_prompt"`
	Roles           struct {
		System    string `json:"system"`
		User      string `json:"user"`
		Assistant string `json:"assistant"`
	} `json:"roles"`
	Tags struct {
		Open  string `json:"open"`
		Close string `json:"close"`
	} `json:"tags"`
	Multimodal struct {
		Image string `json:"image"`
		Video string `json:"video"`
		Audio string `json:"audio"`
	} `json:"multimodal"`
}

var (
	promptProfileSpecsOnce sync.Once
	promptProfileSpecs     []promptProfileSpec
)

func resolvePromptProfile(modelHint string) promptProfile {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	for _, spec := range loadPromptProfileSpecs() {
		if spec.matches(lower) {
			return spec.toPromptProfile()
		}
	}
	return fallbackPromptProfile(modelHint)
}

func loadPromptProfileSpecs() []promptProfileSpec {
	promptProfileSpecsOnce.Do(func() {
		path := os.Getenv("DENSECORE_PROMPT_PROFILE_PATH")
		if path == "" {
			path = defaultPromptProfilePath()
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return
		}
		var decoded promptProfileFile
		if err := json.Unmarshal(content, &decoded); err != nil {
			return
		}
		promptProfileSpecs = decoded.Profiles
	})
	return promptProfileSpecs
}

func defaultPromptProfilePath() string {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	return filepath.Clean(filepath.Join(filepath.Dir(currentFile), "../../../python/densecore/prompt_profiles.json"))
}

func (s promptProfileSpec) matches(modelHint string) bool {
	for _, token := range s.MatchSubstrings {
		if token != "" && strings.Contains(modelHint, strings.ToLower(token)) {
			return true
		}
	}
	return false
}

func (s promptProfileSpec) toPromptProfile() promptProfile {
	return promptProfile{
		family:              parsePromptFamily(s.Family),
		kind:                parsePromptProfileKind(s.Kind),
		userRole:            s.Roles.User,
		assistantRole:       s.Roles.Assistant,
		systemRole:          s.Roles.System,
		openTag:             s.Tags.Open,
		closeTag:            s.Tags.Close,
		defaultSystemPrompt: firstNonEmpty(s.DefaultSystem, defaultChatSystemPrompt),
		imageToken:          s.Multimodal.Image,
		videoToken:          s.Multimodal.Video,
		audioToken:          s.Multimodal.Audio,
	}
}

func parsePromptFamily(family string) promptFamily {
	switch strings.ToLower(strings.TrimSpace(family)) {
	case "qwen":
		return promptFamilyQwen
	case "gemma":
		return promptFamilyGemma
	default:
		return promptFamilyGeneric
	}
}

func parsePromptProfileKind(kind string) promptProfileKind {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "chatml":
		return promptProfileKindChatML
	case "turn_tags":
		return promptProfileKindTurnTags
	default:
		return promptProfileKindGenericTranscript
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func fallbackPromptProfile(modelHint string) promptProfile {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	switch {
	case strings.Contains(lower, "qwen"):
		return promptProfile{
			family:              promptFamilyQwen,
			kind:                promptProfileKindChatML,
			userRole:            "user",
			assistantRole:       "assistant",
			systemRole:          "system",
			openTag:             "<|im_start|>",
			closeTag:            "<|im_end|>\n",
			defaultSystemPrompt: defaultChatSystemPrompt,
			imageToken:          "<|vision_start|><|image_pad|><|vision_end|>",
			videoToken:          "<|vision_start|><|video_pad|><|vision_end|>",
		}
	case strings.Contains(lower, "gemma"):
		return promptProfile{
			family:        promptFamilyGemma,
			kind:          promptProfileKindTurnTags,
			userRole:      "user",
			assistantRole: "model",
			systemRole:    "system",
			openTag:       "<|turn>",
			closeTag:      "<turn|>\n",
			imageToken:    "\n\n<|image|>\n\n",
			videoToken:    "\n\n<|video|>\n\n",
			audioToken:    "<|audio|>",
		}
	default:
		return promptProfile{
			family:              promptFamilyGeneric,
			kind:                promptProfileKindGenericTranscript,
			defaultSystemPrompt: defaultChatSystemPrompt,
		}
	}
}

func (p promptProfile) thinkingEnabled(modelHint string, templateKwargs *domain.ChatTemplateKwargs) bool {
	switch p.family {
	case promptFamilyQwen:
		return qwenThinkingEnabled(modelHint, templateKwargs)
	case promptFamilyGemma:
		return gemmaThinkingEnabled(modelHint, templateKwargs)
	default:
		return false
	}
}

func qwenThinkingEnabled(modelHint string, templateKwargs *domain.ChatTemplateKwargs) bool {
	if templateKwargs != nil && templateKwargs.EnableThinking != nil {
		return *templateKwargs.EnableThinking
	}
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	switch {
	case strings.Contains(lower, "qwen3.5") || strings.Contains(lower, "qwen3_5") ||
		strings.Contains(lower, "qwen3-5") || strings.Contains(lower, "qwen35"):
		return util.ParseBoolEnv("DENSECORE_QWEN35_ENABLE_THINKING", false)
	case strings.Contains(lower, "qwen3"):
		return util.ParseBoolEnv("DENSECORE_QWEN3_ENABLE_THINKING", true)
	default:
		return true
	}
}

func gemmaThinkingEnabled(_ string, templateKwargs *domain.ChatTemplateKwargs) bool {
	return templateKwargs != nil && templateKwargs.EnableThinking != nil && *templateKwargs.EnableThinking
}
