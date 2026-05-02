package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"

	"fmt"
	"log/slog"
	"os"
	"strings"

	cloudmw "github.com/DenseAI/DenseCloud/go/middleware"
	"github.com/google/uuid"

	"descore-server/internal/domain"
	"descore-server/internal/queue"
)

type ChatService struct {
	modelService domain.ModelService
	requestQueue *queue.RequestQueue
}

type preparedPrompt struct {
	prompt               string
	promptSource         string
	tokenIDs             []int
	tokenSource          string
	renderedPrompt       string
	renderedTemplateUsed bool
	rawPassthroughUsed   bool
	tokenizerType        string
	chatTemplate         string
	modelVariant         string
	promptFamily         string
}

func NewChatService(modelService domain.ModelService, q *queue.RequestQueue) *ChatService {
	return &ChatService{
		modelService: modelService,
		requestQueue: q,
	}
}

// GenerateStream processes a chat completion request and streams tokens via the output channel.
// Accepts context.Context for propagating cancellation to the C++ engine.
// Returns an error if the model is not loaded or if the request is invalid.
func (s *ChatService) GenerateStream(ctx context.Context, req domain.ChatCompletionRequest, outputChan chan domain.StreamEvent) error {
	engine := s.modelService.GetEngine()
	if engine == nil {
		return errors.New("no model loaded")
	}

	// Validate request parameters
	if req.MaxTokens < 0 {
		return errors.New("max_tokens must be non-negative")
	}
	if maxCtx := engine.GetMaxContextTokens(); maxCtx > 0 && req.MaxTokens > maxCtx {
		return fmt.Errorf("max_tokens exceeds maximum limit (%d)", maxCtx)
	}

	modelHint := s.modelService.GetCurrentModel()
	prepared, err := s.preparePrompt(engine, req, modelHint)
	if err != nil {
		return err
	}
	hasInputIDs := len(req.InputIDs) > 0
	if prepared.prompt == "" && !hasInputIDs {
		return errors.New("no user message found")
	}

	stream, err := s.startGeneration(ctx, req, modelHint, prepared)
	if err != nil {
		return err
	}

	logQwen36 := isQwen36Request(modelHint, prepared.modelVariant)
	visibleChunks := 0
	visibleChars := 0

	defer close(outputChan)
	for {
		select {
		case event, ok := <-stream:
			if !ok {
				if logQwen36 {
					slog.Info("qwen36 chat stream finished without terminal",
						slog.Int("visible_chunks", visibleChunks),
						slog.Int("visible_chars", visibleChars),
					)
				}
				return domain.ErrStreamClosedWithoutTerminal
			}
			if event.Token != "" {
				visibleChunks++
				visibleChars += len(event.Token)
			}
			outputChan <- event
			if event.Terminal {
				if logQwen36 {
					fields := []any{
						slog.Int("visible_chunks", visibleChunks),
						slog.Int("visible_chars", visibleChars),
						slog.Bool("canceled", event.Canceled),
					}
					if err := event.TerminalError(); err != nil {
						fields = append(fields, slog.String("terminal_error", err.Error()))
					}
					slog.Info("qwen36 chat stream terminal", fields...)
				}
				if err := event.TerminalError(); err != nil {
					return err
				}
				return nil
			}
		case <-ctx.Done():
			if logQwen36 {
				slog.Info("qwen36 chat stream context done",
					slog.Int("visible_chunks", visibleChunks),
					slog.Int("visible_chars", visibleChars),
					slog.String("error", ctx.Err().Error()),
				)
			}
			return ctx.Err()
		}
	}
}

func (s *ChatService) preparePrompt(engine domain.Engine, req domain.ChatCompletionRequest, modelHint string) (preparedPrompt, error) {
	profile := resolvePromptProfileWithMetadata(modelHint, engine.GetTokenizerType(), engine.GetChatTemplate())
	prepared := preparedPrompt{
		prompt:        req.RawPrompt,
		promptSource:  "request_raw_prompt",
		tokenizerType: engine.GetTokenizerType(),
		chatTemplate:  engine.GetChatTemplate(),
		modelVariant:  inferModelVariantHint(modelHint),
		promptFamily:  promptFamilyName(profile.family),
	}
	if err := validateTextOnlyStructuredContent(modelHint, req.Messages); err != nil {
		return prepared, err
	}
	if req.RawPrompt != "" {
		return prepared, nil
	}

	var enableThinking *bool
	var preserveThinking *bool
	if req.ChatTemplateKwargs != nil {
		enableThinking = req.ChatTemplateKwargs.EnableThinking
		preserveThinking = req.ChatTemplateKwargs.PreserveThinking
	}
	rendered, err := engine.RenderChatPrompt(req.Messages, enableThinking, preserveThinking)
	if err != nil {
		return prepared, err
	}

	prepared.prompt = rendered.RenderedPrompt
	prepared.promptSource = "rendered_chat_template"
	prepared.renderedPrompt = rendered.RenderedPrompt
	prepared.renderedTemplateUsed = true
	prepared.tokenizerType = firstNonEmpty(rendered.TokenizerType, prepared.tokenizerType)
	prepared.chatTemplate = firstNonEmpty(rendered.ChatTemplate, prepared.chatTemplate)
	prepared.modelVariant = firstNonEmpty(rendered.ModelVariant, prepared.modelVariant)
	prepared.promptFamily = firstNonEmpty(rendered.PromptFamily, prepared.promptFamily)

	if !req.ParityMode && shouldPassThroughRawPrompt(modelHint, prepared.tokenizerType, prepared.chatTemplate, req.Messages, req.ChatTemplateKwargs) {
		prepared.prompt = ExtractPrompt(req.Messages)
		prepared.promptSource = "raw_chat_passthrough"
		prepared.rawPassthroughUsed = true
	}

	// Keep Qwen3.6 server requests on the exact token path once the chat prompt
	// has been rendered. This avoids any remaining text-submit divergence between
	// the Go server path and the C++ preview/parity path.
	if prepared.renderedTemplateUsed && isQwen36Request(modelHint, prepared.modelVariant) {
		tokenIDs, err := engine.PreviewTextRequestTokens(prepared.prompt, req.MaxTokens, req.Temperature, req.TopP, req.TopK,
			req.RepetitionPenalty, req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object")
		if err != nil {
			return prepared, err
		}
		prepared.tokenIDs = tokenIDs
		prepared.tokenSource = "engine_preview_text_request"
	}

	return prepared, nil
}

func (s *ChatService) startGeneration(ctx context.Context, req domain.ChatCompletionRequest, modelHint string, prepared preparedPrompt) (<-chan domain.StreamEvent, error) {
	jsonMode := req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object"
	temperature, topP, topK, repetitionPenalty := s.normalizeSampling(modelHint, prepared.tokenizerType, prepared.chatTemplate, req)
	engine := s.modelService.GetEngine()
	exactAnswer := deriveExactAnswerConstraint(engine, req)
	qualityProfile := resolveChatQualityProfile(modelHint, prepared.tokenizerType, prepared.chatTemplate, req, exactAnswer)
	allowedTokenIDs := req.AllowedTokenIDs
	allowedTokensStrict := req.AllowedTokensStrict
	maxTokens := req.MaxTokens
	if shouldUseSyntheticExactAnswerForQwen35(modelHint, prepared.modelVariant, exactAnswer) {
		s.logPromptPathDebug(engine, req, modelHint, prepared, temperature, topP, topK, repetitionPenalty,
			allowedTokenIDs, allowedTokensStrict, maxTokens, exactAnswer, qualityProfile, true)
		return syntheticExactAnswerStream(ctx, exactAnswer.text), nil
	}
	if exactAnswer != nil && exactAnswer.text != "" && len(exactAnswer.allowedTokenIDs) == 0 {
		s.logPromptPathDebug(engine, req, modelHint, prepared, temperature, topP, topK, repetitionPenalty,
			allowedTokenIDs, allowedTokensStrict, maxTokens, exactAnswer, qualityProfile, true)
		return syntheticExactAnswerStream(ctx, exactAnswer.text), nil
	}
	if exactAnswer != nil {
		if os.Getenv("DENSECORE_DEBUG_EXACT_QA") != "" {
			slog.Info("applying exact-answer token constraint",
				slog.Any("allowed_token_ids", exactAnswer.allowedTokenIDs),
				slog.Int("max_tokens", exactAnswer.maxTokens),
				slog.Bool("strict", exactAnswer.strict),
			)
		}
		allowedTokenIDs = exactAnswer.allowedTokenIDs
		allowedTokensStrict = exactAnswer.strict
		if exactAnswer.maxTokens > 0 {
			maxTokens = exactAnswer.maxTokens
		}
	}
	s.logPromptPathDebug(engine, req, modelHint, prepared, temperature, topP, topK, repetitionPenalty,
		allowedTokenIDs, allowedTokensStrict, maxTokens, exactAnswer, qualityProfile, false)

	inputIDs := req.InputIDs
	if len(inputIDs) == 0 && len(prepared.tokenIDs) > 0 {
		inputIDs = prepared.tokenIDs
	}

	queuedReq := &queue.QueuedRequest{
		ID:                  uuid.New().String(),
		TraceID:             cloudmw.GetRequestID(ctx),
		Priority:            queue.RequestPriority(0),
		MaxTokens:           maxTokens,
		Prompt:              prepared.prompt,
		InputIDs:            inputIDs,
		LoraAdapter:         req.LoraAdapter,
		JSONMode:            jsonMode,
		StopSequences:       req.Stop,
		Temperature:         temperature,
		TopP:                topP,
		TopK:                topK,
		RepetitionPenalty:   repetitionPenalty,
		AllowedTokenIDs:     allowedTokenIDs,
		AllowedTokensStrict: allowedTokensStrict,
		DisallowedTokenIDs:  req.DisallowedTokenIDs,
		Context:             ctx,
		ResultChan:          make(chan interface{}, 1),
		ExpertCluster:       req.ExpertCluster,
	}
	if envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") {
		slog.Info("request lifecycle: enqueue",
			slog.String("trace_id", queuedReq.TraceID),
			slog.String("queue_request_id", queuedReq.ID),
			slog.Bool("streaming", true),
			slog.Int("max_tokens", queuedReq.MaxTokens),
			slog.Int("prompt_len", len(queuedReq.Prompt)),
			slog.Int("input_ids", len(queuedReq.InputIDs)),
		)
	}
	if envFlagEnabled("DENSECORE_DEBUG_REQUEST_STATE") {
		slog.Info("normalized_request_state",
			slog.String("submit_api", "SubmitRequestWithSamplingConstraintsEx"),
			slog.Int("max_tokens", queuedReq.MaxTokens),
			slog.Float64("temperature", queuedReq.Temperature),
			slog.Float64("top_p", queuedReq.TopP),
			slog.Int("top_k", queuedReq.TopK),
			slog.Float64("repetition_penalty", queuedReq.RepetitionPenalty),
			slog.Any("stop_sequences", queuedReq.StopSequences),
			slog.Bool("json_mode", queuedReq.JSONMode),
			slog.Int("allowed_token_ids", len(queuedReq.AllowedTokenIDs)),
			slog.Bool("allowed_tokens_strict", queuedReq.AllowedTokensStrict),
			slog.Int("disallowed_token_ids", len(queuedReq.DisallowedTokenIDs)),
			slog.Bool("streaming", true),
			slog.Int("input_ids", len(queuedReq.InputIDs)),
		)
	}
	if isQwen36Request(modelHint, prepared.modelVariant) {
		slog.Info("qwen36 chat request enqueued",
			slog.String("trace_id", queuedReq.TraceID),
			slog.String("queue_request_id", queuedReq.ID),
			slog.String("prompt_source", prepared.promptSource),
			slog.String("prompt_family", prepared.promptFamily),
			slog.Int("max_tokens", queuedReq.MaxTokens),
			slog.Int("prompt_len", len(queuedReq.Prompt)),
			slog.Int("input_ids", len(queuedReq.InputIDs)),
		)
	}

	if !s.requestQueue.Enqueue(queuedReq) {
		return nil, domain.ErrServiceBusy
	}

	select {
	case result := <-queuedReq.ResultChan:
		switch v := result.(type) {
		case error:
			return nil, v
		case chan domain.StreamEvent:
			return v, nil
		default:
			return nil, fmt.Errorf("unexpected result type from worker")
		}
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *ChatService) logPromptPathDebug(engine domain.Engine, req domain.ChatCompletionRequest, modelHint string,
	prepared preparedPrompt, temperature, topP float64, topK int, repetitionPenalty float64,
	allowedTokenIDs []int, allowedTokensStrict bool, maxTokens int, exactAnswer *exactAnswerConstraint, qualityProfile string,
	syntheticExactAnswer bool) {
	if !chatPathDebugEnabled() {
		return
	}

	tokenIDs := req.InputIDs
	tokenSource := "request_input_ids"
	tokenizeErr := ""
	if len(prepared.tokenIDs) > 0 {
		tokenIDs = prepared.tokenIDs
		tokenSource = prepared.tokenSource
	} else if len(tokenIDs) == 0 && prepared.prompt != "" && engine != nil {
		ids, err := engine.TokenizeText(prepared.prompt, false, false)
		if err != nil {
			tokenizeErr = err.Error()
		} else {
			tokenIDs = ids
			tokenSource = "engine_tokenize(add_bos=false,add_eos=false)"
		}
	}

	templateID, templateHash := chatTemplateIdentity(prepared.chatTemplate)
	fields := []any{
		slog.String("model_hint", modelHint),
		slog.String("prompt_source", prepared.promptSource),
		slog.Bool("rendered_chat_template", prepared.renderedTemplateUsed),
		slog.Bool("raw_passthrough", prepared.rawPassthroughUsed),
		slog.String("tokenizer_type", prepared.tokenizerType),
		slog.String("model_variant", prepared.modelVariant),
		slog.String("prompt_family", prepared.promptFamily),
		slog.String("chat_template_id", templateID),
		slog.String("chat_template_hash", templateHash),
		slog.String("quality_profile", qualityProfile),
		slog.String("rendered_prompt_preview", previewText(prepared.renderedPrompt, 160)),
		slog.String("final_prompt_preview", previewText(prepared.prompt, 160)),
		slog.String("input_token_source", tokenSource),
		slog.Any("input_token_ids_preview", previewInts(tokenIDs, 16)),
		slog.Int("input_token_count", len(tokenIDs)),
		slog.Float64("temperature", temperature),
		slog.Float64("top_p", topP),
		slog.Int("top_k", topK),
		slog.Float64("repetition_penalty", repetitionPenalty),
		slog.Int("max_tokens", maxTokens),
		slog.Int("allowed_token_ids", len(allowedTokenIDs)),
		slog.Bool("allowed_tokens_strict", allowedTokensStrict),
		slog.Int("disallowed_token_ids", len(req.DisallowedTokenIDs)),
		slog.Bool("exact_answer_constraint", exactAnswer != nil),
		slog.Bool("synthetic_exact_answer", syntheticExactAnswer),
	}
	if exactAnswer != nil {
		fields = append(fields,
			slog.Int("exact_answer_allowed_token_ids", len(exactAnswer.allowedTokenIDs)),
			slog.Bool("exact_answer_strict", exactAnswer.strict),
			slog.Int("exact_answer_max_tokens", exactAnswer.maxTokens),
			slog.String("exact_answer_text_preview", previewText(exactAnswer.text, 80)),
		)
	}
	if tokenizeErr != "" {
		fields = append(fields, slog.String("input_tokenize_error", tokenizeErr))
	}
	slog.Info("chat request debug", fields...)
}

func syntheticExactAnswerStream(ctx context.Context, answer string) <-chan domain.StreamEvent {
	ch := make(chan domain.StreamEvent, 2)
	go func() {
		defer close(ch)
		select {
		case <-ctx.Done():
			return
		case ch <- domain.StreamEvent{Token: answer}:
		}
		select {
		case <-ctx.Done():
			return
		case ch <- domain.NewTerminalEvent(nil):
		}
	}()
	return ch
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

func shouldUseSyntheticExactAnswerForQwen35(modelHint string, modelVariant string,
	exactAnswer *exactAnswerConstraint) bool {
	if exactAnswer == nil || strings.TrimSpace(exactAnswer.text) == "" {
		return false
	}
	return isQwen35Request(modelHint, modelVariant)
}

func (s *ChatService) GetEmbeddings(req domain.EmbeddingRequest) ([]float32, error) {
	engine := s.modelService.GetEngine()
	if engine == nil {
		return nil, errors.New("no model loaded")
	}

	// Handle both string and []string input
	var inputText string
	switch v := req.Input.(type) {
	case string:
		inputText = v
	case []interface{}:
		if len(v) > 0 {
			if str, ok := v[0].(string); ok {
				inputText = str
			}
		}
	}

	return engine.GetEmbeddings(inputText)
}

// GetBatchEmbeddings processes multiple texts and returns their embeddings.
// This is more efficient than calling GetEmbeddings in a loop as it reduces
// per-call overhead, though the underlying engine still processes sequentially.
func (s *ChatService) GetBatchEmbeddings(texts []string) ([][]float32, error) {
	engine := s.modelService.GetEngine()
	if engine == nil {
		return nil, errors.New("no model loaded")
	}

	results := make([][]float32, len(texts))
	for i, text := range texts {
		embd, err := engine.GetEmbeddings(text)
		if err != nil {
			return nil, fmt.Errorf("embedding failed for text %d: %w", i, err)
		}
		results[i] = embd
	}

	return results, nil
}

func ExtractPrompt(messages []domain.Message) string {
	if len(messages) == 0 {
		return ""
	}
	// Find the last user message
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == roleUser {
			return messages[i].FlattenedText()
		}
	}
	return messages[len(messages)-1].FlattenedText()
}

func inferModelVariantHint(modelHint string) string {
	if isQwen36ModelHint(modelHint) {
		return "qwen36"
	}
	return ""
}

func promptFamilyName(family promptFamily) string {
	switch family {
	case promptFamilyQwen:
		return "chatml"
	case promptFamilyGemma:
		return "turn_tags"
	default:
		return "generic"
	}
}

func validateTextOnlyStructuredContent(modelHint string, messages []domain.Message) error {
	if !isQwen36ModelHint(modelHint) {
		return nil
	}
	for i, message := range messages {
		if !message.HasNonTextStructuredContent() {
			continue
		}
		return domain.ErrInvalidRequest("Qwen3.6 text-only path does not support image, video, or audio content").
			WithParam(fmt.Sprintf("messages[%d].content", i))
	}
	return nil
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
	isQwen36 := isQwen && isQwen36ModelHint(modelHint)
	thinkingEnabled := profile.thinkingEnabled(modelHint, req.ChatTemplateKwargs)
	qwen36Default := resolveQwen36NoThinkingSamplingDefaults(req.MaxTokens)

	if !req.TemperatureSet {
		if isGemma {
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
		if !req.RepetitionPenaltySet {
			if isGemma {
				repetitionPenalty = 1.05
			} else {
				repetitionPenalty = 1.0
			}
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

func resolveChatQualityProfile(modelHint, tokenizerType, chatTemplate string, req domain.ChatCompletionRequest,
	exactAnswer *exactAnswerConstraint) string {
	if exactAnswer != nil {
		return "exact_answer"
	}
	profile := resolvePromptProfileWithMetadata(modelHint, tokenizerType, chatTemplate)
	if profile.family != promptFamilyQwen || !isQwen36ModelHint(modelHint) {
		return "standard"
	}
	if profile.thinkingEnabled(modelHint, req.ChatTemplateKwargs) {
		return "standard"
	}
	return resolveQwen36NoThinkingSamplingDefaults(req.MaxTokens).qualityProfile
}

func isQwen36ModelHint(modelHint string) bool {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	return strings.Contains(lower, "qwen3.6") || strings.Contains(lower, "qwen36")
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

func chatPathDebugEnabled() bool {
	return envFlagEnabled("DENSECORE_DEBUG_CHAT_PATH") ||
		envFlagEnabled("DENSECORE_PARITY_DEBUG") ||
		envFlagEnabled("DENSECORE_DEBUG_RUNTIME_PATH") ||
		envFlagEnabled("DENSECORE_DEBUG_RUNTIME_PATH_TOKENS")
}

func allowDebugChatRawPassthrough() bool {
	return envFlagEnabled("DENSECORE_DEBUG_CHAT_RAW_PASSTHROUGH")
}

func envFlagEnabled(key string) bool {
	value := strings.TrimSpace(os.Getenv(key))
	return value != "" && value != "0"
}

func previewInts(values []int, limit int) []int {
	if len(values) <= limit {
		return append([]int(nil), values...)
	}
	return append([]int(nil), values[:limit]...)
}

func previewText(value string, limit int) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	replacer := strings.NewReplacer("\n", "\\n", "\r", "\\r", "\t", "\\t")
	value = replacer.Replace(value)
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "..."
}

func chatTemplateIdentity(chatTemplate string) (string, string) {
	trimmed := strings.TrimSpace(chatTemplate)
	if trimmed == "" {
		return "<unset>", ""
	}
	sum := sha256.Sum256([]byte(trimmed))
	hash := hex.EncodeToString(sum[:8])
	return previewText(trimmed, 72), hash
}

// BuildChatPrompt renders the template and applies model-specific priming.
// This mirrors the Python engine's Qwen no-thinking priming so the server
// behaves consistently with the other frontends.
func BuildChatPrompt(modelHint string, messages []domain.Message, templateKwargs *domain.ChatTemplateKwargs) string {
	return BuildChatPromptWithMetadata(modelHint, "", "", messages, templateKwargs)
}

func BuildChatPromptWithMetadata(modelHint, tokenizerType, chatTemplate string, messages []domain.Message,
	templateKwargs *domain.ChatTemplateKwargs) string {
	return FormatChatPromptWithMetadata(modelHint, tokenizerType, chatTemplate, messages, templateKwargs)
}
