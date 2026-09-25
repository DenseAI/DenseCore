package service

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/util"
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
			defaultSystemPrompt: "You are a helpful assistant. Answer the user's request directly. Do not describe the prompt or your reasoning.",
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
	if isQwen36ModelHint(lower) || isQwen38ModelHint(lower) {
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

func isQwen36Request(modelHint string, modelVariant string) bool {
	variant := strings.ToLower(strings.TrimSpace(modelVariant))
	if variant == "qwen36" || variant == "qwen3.6" {
		return true
	}
	hint := strings.ToLower(modelHint)
	return strings.Contains(hint, "qwen3.6")
}

func isQwen35Request(modelHint string, modelVariant string) bool {
	variant := strings.ToLower(strings.TrimSpace(modelVariant))
	if variant == "qwen35" || variant == "qwen3.5" {
		return true
	}
	hint := strings.ToLower(modelHint)
	return strings.Contains(hint, "qwen3.5") || strings.Contains(hint, "qwen35")
}

func isQwenRenderedTokenPathRequest(modelHint string, modelVariant string) bool {
	return isQwen35Request(modelHint, modelVariant) || isQwen36Request(modelHint, modelVariant) ||
		isQwen38ModelHint(modelHint) || isQwen38ModelHint(modelVariant)
}

func IsQwenModelHint(modelHint string) bool {
	return isQwen35Request(modelHint, "") || isQwen36Request(modelHint, "") || isQwen38ModelHint(modelHint)
}

func isGemmaRenderedTokenPathRequest(modelHint string, modelVariant string, tokenizerType string) bool {
	variant := strings.ToLower(strings.TrimSpace(modelVariant))
	tokenizer := strings.ToLower(strings.TrimSpace(tokenizerType))
	hint := strings.ToLower(modelHint)
	return strings.Contains(variant, "gemma") || strings.Contains(tokenizer, "gemma") ||
		strings.Contains(hint, "gemma")
}

func isLFM2RenderedTokenPathRequest(modelHint string, modelVariant string, tokenizerType string) bool {
	variant := strings.ToLower(strings.TrimSpace(modelVariant))
	tokenizer := strings.ToLower(strings.TrimSpace(tokenizerType))
	hint := strings.ToLower(modelHint)
	return strings.Contains(variant, "lfm2") || strings.Contains(tokenizer, "lfm2") ||
		strings.Contains(hint, "lfm2")
}

func isRenderedTokenPathRequest(modelHint string, modelVariant string, tokenizerType string) bool {
	return isQwenRenderedTokenPathRequest(modelHint, modelVariant) ||
		isGemmaRenderedTokenPathRequest(modelHint, modelVariant, tokenizerType) ||
		isLFM2RenderedTokenPathRequest(modelHint, modelVariant, tokenizerType)
}

func inferModelVariantHint(modelHint string) string {
	if isQwen38ModelHint(modelHint) {
		return "qwen38"
	}
	if isQwen36ModelHint(modelHint) {
		return "qwen36"
	}
	if isQwen35Request(modelHint, "") {
		return "qwen35"
	}
	return ""
}

func promptFamilyName(family promptFamily) string {
	switch family {
	case promptFamilyQwen:
		return "chatml"
	case promptFamilyGemma:
		return "turn_tags"
	case promptFamilyLFM2:
		return "chatml"
	default:
		return "generic"
	}
}

func (s *ChatService) normalizeSampling(modelHint, tokenizerType, chatTemplate string, req domain.ChatCompletionRequest) (float64, float64, int, float64) {
	temperature := req.Temperature
	topP := req.TopP
	topK := req.TopK
	repetitionPenalty := req.RepetitionPenalty
	if req.ParityMode {
		return temperature, topP, topK, repetitionPenalty
	}
	profile := resolvePromptProfileWithMetadata(modelHint, tokenizerType, chatTemplate)
	isQwen := profile.family == promptFamilyQwen
	isGemma := profile.family == promptFamilyGemma
	isLFM2 := profile.family == promptFamilyLFM2 || strings.Contains(strings.ToLower(strings.TrimSpace(modelHint)), "lfm2")
	isQwen36 := isQwen && isQwen36ModelHint(modelHint)
	thinkingEnabled := profile.thinkingEnabled(modelHint, req.ChatTemplateKwargs)
	qwen36Default := resolveQwen36NoThinkingSamplingDefaults(req.MaxTokens)

	if !req.TemperatureSet {
		if isGemma {
			temperature = 0.2
		} else if isLFM2 {
			temperature = 0.2
		} else if isQwen36 && !thinkingEnabled {
			temperature = qwen36Default.temperature
		} else if isQwen && !thinkingEnabled {
			temperature = 0.7
		} else {
			temperature = 1.0
		}
	}
	if !req.TopPSet {
		if isGemma {
			topP = 0.95
		} else if isLFM2 {
			topP = 0.8
		} else if isQwen36 && !thinkingEnabled {
			topP = qwen36Default.topP
		} else if isQwen && thinkingEnabled {
			topP = 0.95
		} else if isQwen {
			topP = 0.8
		} else {
			topP = 1.0
		}
	}
	if !req.TopKSet {
		if isGemma {
			topK = 32
		} else if isLFM2 {
			topK = 20
		} else if isQwen36 && !thinkingEnabled {
			topK = qwen36Default.topK
		} else if isQwen {
			topK = 20
		} else {
			topK = 0
		}
	}
	if !req.RepetitionPenaltySet {
		if isGemma {
			repetitionPenalty = 1.05
		} else if isLFM2 {
			if req.MaxTokens >= 64 {
				repetitionPenalty = 1.12
			} else {
				repetitionPenalty = 1.05
			}
		} else if isQwen36 && !thinkingEnabled {
			repetitionPenalty = qwen36Default.repetitionPenalty
		} else if isQwen {
			repetitionPenalty = 1.05
		} else {
			repetitionPenalty = 1.0
		}
	}

	if temperature == 0.0 {
		if !req.TopPSet {
			topP = 1.0
		}
		if !req.TopKSet {
			topK = 1
		}
		if !req.RepetitionPenaltySet && !isLFM2 {
			repetitionPenalty = 1.0
		}
	}

	return temperature, topP, topK, repetitionPenalty
}

type qwen36SamplingDefaults struct {
	temperature       float64
	topP              float64
	topK              int
	repetitionPenalty float64
	qualityProfile    string
}

func resolveQwen36NoThinkingSamplingDefaults(maxTokens int) qwen36SamplingDefaults {
	if maxTokens >= 64 {
		return qwen36SamplingDefaults{
			temperature:       0.35,
			topP:              0.95,
			topK:              40,
			repetitionPenalty: 1.08,
			qualityProfile:    "qwen36_longform",
		}
	}
	return qwen36SamplingDefaults{
		temperature:       0.20,
		topP:              0.95,
		topK:              20,
		repetitionPenalty: 1.05,
		qualityProfile:    "qwen36_default",
	}
}

func resolveChatQualityProfile(modelHint, tokenizerType, chatTemplate string, req domain.ChatCompletionRequest) string {
	profile := resolvePromptProfileWithMetadata(modelHint, tokenizerType, chatTemplate)
	if profile.family != promptFamilyQwen || !isQwen36ModelHint(modelHint) {
		return "standard"
	}
	if profile.thinkingEnabled(modelHint, req.ChatTemplateKwargs) {
		return "standard"
	}
	return resolveQwen36NoThinkingSamplingDefaults(req.MaxTokens).qualityProfile
}

func isLFM2PreparedRequest(modelHint string, prepared preparedPrompt) bool {
	lowerHint := strings.ToLower(strings.TrimSpace(modelHint))
	lowerVariant := strings.ToLower(strings.TrimSpace(prepared.modelVariant))
	lowerTokenizer := strings.ToLower(strings.TrimSpace(prepared.tokenizerType))
	if strings.Contains(lowerHint, "lfm2") || strings.Contains(lowerVariant, "lfm2") ||
		strings.Contains(lowerTokenizer, "lfm2") {
		return true
	}
	profile := resolvePromptProfileWithMetadata(modelHint, prepared.tokenizerType, prepared.chatTemplate)
	return profile.family == promptFamilyLFM2
}

func defaultLFM2StopSequences() []string {
	return []string{"<|im_end|>", "<|endoftext|>", "<|startoftext|>", "<|im_start|>"}
}

func isQwen36ModelHint(modelHint string) bool {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	return strings.Contains(lower, "qwen3.6") || strings.Contains(lower, "qwen36")
}

func isQwen38ModelHint(modelHint string) bool {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	return strings.Contains(lower, "qwen3.8") || strings.Contains(lower, "qwen3_8") ||
		strings.Contains(lower, "qwen3-8") || strings.Contains(lower, "qwen38")
}

func isQwen35ModelHint(modelHint string) bool {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	return strings.Contains(lower, "qwen3.5") || strings.Contains(lower, "qwen3_5") ||
		strings.Contains(lower, "qwen3-5") || strings.Contains(lower, "qwen35")
}

func shouldPassThroughRawPrompt(modelHint, tokenizerType, chatTemplate string, messages []domain.Message,
	templateKwargs *domain.ChatTemplateKwargs) bool {
	if !allowDebugChatRawPassthrough() {
		return false
	}
	profile := resolvePromptProfileWithMetadata(modelHint, tokenizerType, chatTemplate)
	if profile.family == promptFamilyGemma {
		return false
	}
	if templateKwargs != nil && (templateKwargs.EnableThinking != nil || templateKwargs.PreserveThinking != nil) {
		return false
	}
	if len(messages) != 1 {
		return false
	}
	msg := messages[0]
	if strings.ToLower(strings.TrimSpace(msg.Role)) != roleUser {
		return false
	}
	if msg.HasStructuredContent() || len(msg.ToolCalls) > 0 || len(msg.ToolResponses) > 0 || msg.ReasoningContent != "" {
		return false
	}
	return strings.TrimSpace(msg.Content) != ""
}
