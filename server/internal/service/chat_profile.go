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
	promptFamilyLFM2
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
	return resolvePromptProfileWithMetadata(modelHint, "", "")
}

func resolvePromptProfileWithMetadata(modelHint, tokenizerType, chatTemplate string) promptProfile {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	tokenizerLower := strings.ToLower(strings.TrimSpace(tokenizerType))
	templateLower := strings.ToLower(strings.TrimSpace(chatTemplate))
	if strings.Contains(tokenizerLower, "lfm2") || strings.Contains(tokenizerLower, "lfm") ||
		strings.Contains(lower, "lfm2") || strings.Contains(lower, "lfm") {
		return fallbackPromptProfile("lfm2")
	}
	switch {
	case strings.Contains(templateLower, "<|im_start|>") || strings.Contains(templateLower, "<|im_end|>"):
		return fallbackPromptProfile("qwen")
	case strings.Contains(templateLower, "<|turn>") || strings.Contains(templateLower, "<turn|>"):
		return fallbackPromptProfile("gemma")
	}
	switch {
	case strings.Contains(tokenizerLower, "qwen35") || strings.Contains(tokenizerLower, "qwen3.5"):
		return fallbackPromptProfile("qwen3.5")
	case strings.Contains(tokenizerLower, "qwen"):
		return fallbackPromptProfile("qwen")
	case strings.Contains(tokenizerLower, "gemma"):
		return fallbackPromptProfile("gemma")
	}
	detectedFamily := inferPromptFamilyFromModelHint(lower)
	for _, spec := range loadPromptProfileSpecs() {
		if spec.matches(lower) {
			profile := spec.toPromptProfile()
			if profile.family == promptFamilyGeneric {
				switch detectedFamily {
				case promptFamilyQwen:
					return fallbackPromptProfile("qwen")
				case promptFamilyGemma:
					return fallbackPromptProfile("gemma")
				}
			}
			return profile
		}
	}
	return fallbackPromptProfile(modelHint)
}

func inferPromptFamilyFromModelHint(modelHint string) promptFamily {
	switch {
	case strings.Contains(modelHint, "lfm2") || strings.Contains(modelHint, "lfm"):
		return promptFamilyLFM2
	case strings.Contains(modelHint, "gemma"):
		return promptFamilyGemma
	case strings.Contains(modelHint, "qwen"):
		return promptFamilyQwen
	default:
		return promptFamilyGeneric
	}
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
	case "lfm2", "lfm":
		return promptFamilyLFM2
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
	case strings.Contains(lower, "lfm2") || strings.Contains(lower, "lfm"):
		return promptProfile{
			family:              promptFamilyLFM2,
			kind:                promptProfileKindChatML,
			userRole:            "user",
			assistantRole:       "assistant",
			systemRole:          "system",
			openTag:             "<|im_start|>",
			closeTag:            "<|im_end|>\n",
			defaultSystemPrompt: "You are a direct answer engine. Output only the final answer requested by the user. Do not quote, paraphrase, explain, analyze, or mention the request.",
		}
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

func (p promptProfile) preserveThinking(templateKwargs *domain.ChatTemplateKwargs) bool {
	if p.family != promptFamilyQwen {
		return true
	}
	if templateKwargs != nil && templateKwargs.PreserveThinking != nil {
		return *templateKwargs.PreserveThinking
	}
	return true
}

func qwenThinkingEnabled(modelHint string, templateKwargs *domain.ChatTemplateKwargs) bool {
	if templateKwargs != nil && templateKwargs.EnableThinking != nil {
		return *templateKwargs.EnableThinking
	}
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	if isQwen36ModelHint(lower) {
		if value, ok := os.LookupEnv("DENSECORE_QWEN36_ENABLE_THINKING"); ok {
			switch strings.TrimSpace(strings.ToLower(value)) {
			case "1", "true", "yes", "on":
				return true
			case "0", "false", "no", "off":
				return false
			}
		}
		return util.ParseBoolEnv("DENSECORE_QWEN35_ENABLE_THINKING", true)
	}
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
	if templateKwargs != nil && templateKwargs.EnableThinking != nil {
		return *templateKwargs.EnableThinking
	}
	if value, ok := os.LookupEnv("DENSECORE_GEMMA4_ENABLE_THINKING"); ok {
		switch strings.TrimSpace(strings.ToLower(value)) {
		case "0", "false", "no", "off":
			return false
		case "1", "true", "yes", "on":
			return true
		}
	}
	return false
}
