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
	"sync"
	"time"

	cloudmw "github.com/DenseAI/DenseCloud/go/middleware"
	"github.com/google/uuid"

	"descore-server/internal/domain"
	"descore-server/internal/promptcache"
	"descore-server/internal/queue"
	agenttools "descore-server/internal/tools"
)

var agentPromptCache = promptcache.NewManager(promptcache.ConfigFromEnv())

type ChatService struct {
	modelService domain.ModelService
	requestQueue *queue.RequestQueue
}

type preparedPrompt struct {
	prompt                   string
	promptSource             string
	tokenIDs                 []int
	tokenSource              string
	renderedChatSubmit       bool
	promptTokenCountDeferred bool
	promptTokenCountSource   string
	promptTokenIDs           []int
	promptTokenCount         int
	renderedPrompt           string
	renderedTemplateUsed     bool
	rawPassthroughUsed       bool
	tokenizerType            string
	chatTemplate             string
	modelVariant             string
	promptFamily             string
}

type generationStartMetrics struct {
	queueWaitMS    float64
	engineSubmitMS float64
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
	serverStart := time.Now()
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
	prepareStart := time.Now()
	prepared, err := s.preparePrompt(engine, req, modelHint)
	servicePrepareMS := durationMillis(time.Since(prepareStart))
	if err != nil {
		return err
	}
	hasInputIDs := len(req.InputIDs) > 0
	if prepared.prompt == "" && !hasInputIDs {
		return errors.New("no user message found")
	}
	if err := validatePreparedContextWindow(engine, prepared, req.InputIDs, req.MaxTokens); err != nil {
		return err
	}

	_, completionCh, done, startMetrics, err := s.startGeneration(ctx, req, modelHint, prepared, outputChan)
	if err != nil {
		return err
	}
	defer done()
	if completionCh == nil {
		logServerSubmitOverhead("stream", prepared.promptTokenCount, servicePrepareMS, startMetrics.engineSubmitMS,
			startMetrics.queueWaitMS, 0, 0, durationMillis(time.Since(serverStart)))
		return nil
	}
	select {
	case <-completionCh:
		logServerSubmitOverhead("stream", prepared.promptTokenCount, servicePrepareMS, startMetrics.engineSubmitMS,
			startMetrics.queueWaitMS, 0, 0, durationMillis(time.Since(serverStart)))
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// GenerateSync runs the same C++ generation path as streaming requests but
// collects tokens inside the service, avoiding the handler-side proxy channel
// used only for SSE formatting.
func (s *ChatService) GenerateSync(ctx context.Context, req domain.ChatCompletionRequest) (string, int, int, error) {
	serverStart := time.Now()
	engine := s.modelService.GetEngine()
	if engine == nil {
		return "", 0, 0, errors.New("no model loaded")
	}

	if req.MaxTokens < 0 {
		return "", 0, 0, errors.New("max_tokens must be non-negative")
	}
	if maxCtx := engine.GetMaxContextTokens(); maxCtx > 0 && req.MaxTokens > maxCtx {
		return "", 0, 0, fmt.Errorf("max_tokens exceeds maximum limit (%d)", maxCtx)
	}

	modelHint := s.modelService.GetCurrentModel()
	prepareStart := time.Now()
	prepared, err := s.preparePrompt(engine, req, modelHint)
	servicePrepareMS := durationMillis(time.Since(prepareStart))
	if err != nil {
		return "", 0, prepared.promptTokenCount, err
	}
	hasInputIDs := len(req.InputIDs) > 0
	if prepared.prompt == "" && !hasInputIDs {
		return "", 0, prepared.promptTokenCount, errors.New("no user message found")
	}
	if err := validatePreparedContextWindow(engine, prepared, req.InputIDs, req.MaxTokens); err != nil {
		return "", 0, prepared.promptTokenCount, err
	}

	submitStart := time.Now()
	stream, _, done, startMetrics, err := s.startGeneration(ctx, req, modelHint, prepared, nil)
	if err != nil {
		return "", 0, prepared.promptTokenCount, err
	}
	defer done()
	firstCallbackMS := 0.0
	lastCallbackMS := 0.0

	logQwen36 := isQwen36Request(modelHint, prepared.modelVariant)
	var responseBuilder strings.Builder
	completionTokens := 0
	visibleChunks := 0
	visibleChars := 0

	for {
		select {
		case event, ok := <-stream:
			if !ok {
				if logQwen36 {
					slog.Info("qwen36 chat sync finished without terminal",
						slog.Int("visible_chunks", visibleChunks),
						slog.Int("visible_chars", visibleChars),
					)
				}
				return "", completionTokens, prepared.promptTokenCount, domain.ErrStreamClosedWithoutTerminal
			}
			if event.Token != "" {
				if completionTokens == 0 {
					firstCallbackMS = durationMillis(time.Since(submitStart))
				}
				lastCallbackMS = durationMillis(time.Since(submitStart))
				responseBuilder.WriteString(event.Token)
				completionTokens++
				visibleChunks++
				visibleChars += len(event.Token)
			}
			if !event.Terminal {
				continue
			}
			if logQwen36 {
				fields := []any{
					slog.Int("visible_chunks", visibleChunks),
					slog.Int("visible_chars", visibleChars),
					slog.Bool("canceled", event.Canceled),
				}
				if err := event.TerminalError(); err != nil {
					fields = append(fields, slog.String("terminal_error", err.Error()))
				}
				slog.Info("qwen36 chat sync terminal", fields...)
			}
			if err := event.TerminalError(); err != nil {
				return "", completionTokens, prepared.promptTokenCount, err
			}
			logServerOverhead("sync", prepared.promptTokenCount, completionTokens, servicePrepareMS,
				startMetrics.engineSubmitMS, startMetrics.queueWaitMS, firstCallbackMS, lastCallbackMS,
				durationMillis(time.Since(serverStart)))
			return responseBuilder.String(), completionTokens, prepared.promptTokenCount, nil
		case <-ctx.Done():
			if logQwen36 {
				slog.Info("qwen36 chat sync context done",
					slog.Int("visible_chunks", visibleChunks),
					slog.Int("visible_chars", visibleChars),
					slog.String("error", ctx.Err().Error()),
				)
			}
			return "", completionTokens, prepared.promptTokenCount, ctx.Err()
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
		s.populatePromptTokenCount(engine, req, &prepared)
		return prepared, nil
	}

	var enableThinking *bool
	var preserveThinking *bool
	if req.ChatTemplateKwargs != nil {
		enableThinking = req.ChatTemplateKwargs.EnableThinking
		preserveThinking = req.ChatTemplateKwargs.PreserveThinking
	}
	messages := req.Messages
	if len(req.Tools) > 0 {
		renderedTools, normalized, err := (agenttools.DefaultRenderer{}).RenderTools(
			firstNonEmpty(modelHint, req.Model, prepared.modelVariant),
			req.Tools,
			req.ToolChoice,
		)
		if err != nil {
			return prepared, err
		}
		if renderedTools != "" {
			family := agenttools.ResolveParserFamily(agenttools.ModelDescriptor{
				ModelID:      firstNonEmpty(modelHint, req.Model),
				ModelVariant: prepared.modelVariant,
			})
			slog.Info("tool_parser_selected",
				slog.String("model_id", firstNonEmpty(req.Model, modelHint)),
				slog.String("model_family", string(family)),
				slog.String("parser_family", string(family)),
				slog.Int("tools", len(normalized.Tools)),
				slog.String("tool_schema_hash", normalized.SchemaHash),
			)
			messages = agenttools.InjectToolPrompt(req.Messages, renderedTools)
		}
	}

	rendered, err := engine.RenderChatPrompt(messages, enableThinking, preserveThinking)
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

	// Keep supported server requests on the exact token path once the chat prompt
	// has been rendered. This avoids any remaining text-submit divergence between
	// the Go server path and the C++ preview/parity path.
	if prepared.renderedTemplateUsed && isRenderedTokenPathRequest(modelHint, prepared.modelVariant, prepared.tokenizerType) {
		if _, ok := engine.(interface {
			GenerateStreamRenderedChatWithSamplingAwaitable(context.Context, string, int, string, bool, float64, float64, int, float64, []string, []int, bool, []int, chan domain.StreamEvent) (<-chan struct{}, error)
		}); ok {
			prepared.renderedChatSubmit = true
			prepared.tokenSource = "engine_submit_rendered_chat"
		} else {
			tokenIDs, err := engine.PreviewRenderedRequestTokens(prepared.prompt, req.MaxTokens, req.Temperature, req.TopP, req.TopK,
				req.RepetitionPenalty, req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object")
			if err != nil {
				return prepared, err
			}
			prepared.tokenIDs = tokenIDs
			prepared.tokenSource = "engine_preview_rendered_request"
		}
	}
	s.populatePromptTokenCount(engine, req, &prepared)

	return prepared, nil
}

func (s *ChatService) populatePromptTokenCount(engine domain.Engine, req domain.ChatCompletionRequest, prepared *preparedPrompt) {
	if prepared == nil {
		return
	}
	switch {
	case len(req.InputIDs) > 0:
		prepared.promptTokenCount = len(req.InputIDs)
		prepared.promptTokenIDs = append(prepared.promptTokenIDs[:0], req.InputIDs...)
	case len(prepared.tokenIDs) > 0:
		prepared.promptTokenCount = len(prepared.tokenIDs)
		prepared.promptTokenIDs = append(prepared.promptTokenIDs[:0], prepared.tokenIDs...)
	case prepared.renderedChatSubmit && req.Stream:
		prepared.promptTokenCountDeferred = true
		prepared.promptTokenCountSource = "rendered_chat_stream_runtime_usage"
	case prepared.prompt != "" && engine != nil:
		if ids, err := engine.TokenizeText(prepared.prompt, false, false); err == nil {
			prepared.promptTokenCount = len(ids)
			prepared.promptTokenIDs = ids
			prepared.promptTokenCountSource = "engine_tokenize"
		}
	}
}

func (s *ChatService) startGeneration(ctx context.Context, req domain.ChatCompletionRequest, modelHint string, prepared preparedPrompt, outputChan chan domain.StreamEvent) (<-chan domain.StreamEvent, <-chan struct{}, func(), generationStartMetrics, error) {
	jsonMode := req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object"
	temperature, topP, topK, repetitionPenalty := s.normalizeSampling(modelHint, prepared.tokenizerType, prepared.chatTemplate, req)
	engine := s.modelService.GetEngine()
	exactAnswer := deriveExactAnswerConstraint(engine, req)
	qualityProfile := resolveChatQualityProfile(modelHint, prepared.tokenizerType, prepared.chatTemplate, req, exactAnswer)
	allowedTokenIDs := req.AllowedTokenIDs
	allowedTokensStrict := req.AllowedTokensStrict
	maxTokens := req.MaxTokens
	if exactAnswer != nil {
		if len(exactAnswer.allowedTokenIDs) > 0 {
			if os.Getenv("DENSECORE_DEBUG_EXACT_QA") != "" {
				slog.Info("applying exact-answer token constraint",
					slog.Any("allowed_token_ids", exactAnswer.allowedTokenIDs),
					slog.Int("max_tokens", exactAnswer.maxTokens),
					slog.Bool("strict", exactAnswer.strict),
				)
			}
			allowedTokenIDs = exactAnswer.allowedTokenIDs
			allowedTokensStrict = exactAnswer.strict
		} else if os.Getenv("DENSECORE_DEBUG_EXACT_QA") != "" {
			slog.Info("exact-answer token constraint unavailable; using real model generation",
				slog.String("answer", exactAnswer.text),
				slog.Int("max_tokens", exactAnswer.maxTokens),
			)
		}
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
	cacheTokens := append([]int(nil), inputIDs...)
	if len(cacheTokens) == 0 && len(prepared.promptTokenIDs) > 0 {
		cacheTokens = append([]int(nil), prepared.promptTokenIDs...)
	}
	if len(cacheTokens) > 0 {
		identity := buildPromptCacheIdentity(req, prepared, modelHint)
		decision := agentPromptCache.LookupAndStore(identity, cacheTokens)
		logPromptCacheDecision(req, modelHint, decision)
	}

	if outputChan == nil {
		outputChan = make(chan domain.StreamEvent, defaultStreamEventBufferSize())
	}
	doneChan := make(chan struct{})
	var doneOnce sync.Once
	done := func() {
		doneOnce.Do(func() {
			close(doneChan)
		})
	}

	queuedReq := &queue.QueuedRequest{
		ID:                  uuid.New().String(),
		TraceID:             cloudmw.GetRequestID(ctx),
		Priority:            queue.RequestPriority(0),
		MaxTokens:           maxTokens,
		Prompt:              prepared.prompt,
		InputIDs:            inputIDs,
		RenderedChatSubmit:  prepared.renderedChatSubmit,
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
		OutputChan:          outputChan,
		DoneChan:            doneChan,
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
		submitAPI := "SubmitRequestWithSamplingConstraintsEx"
		if len(queuedReq.InputIDs) > 0 {
			if queuedReq.Prompt != "" {
				submitAPI = "SubmitRenderedRequestIdsWithSamplingConstraintsCallbackEx"
			} else {
				submitAPI = "SubmitRequestIdsWithSamplingConstraintsEx"
			}
		}
		slog.Info("normalized_request_state",
			slog.String("submit_api", submitAPI),
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
		done()
		return nil, nil, nil, generationStartMetrics{}, domain.ErrServiceBusy
	}

	select {
	case result := <-queuedReq.ResultChan:
		switch v := result.(type) {
		case error:
			done()
			return nil, nil, nil, generationStartMetrics{}, v
		case chan domain.StreamEvent:
			return v, nil, done, generationStartMetrics{}, nil
		case generationResult:
			return v.OutputChan, v.CompletionChan, done, generationStartMetrics{
				queueWaitMS:    v.QueueWaitMS,
				engineSubmitMS: v.EngineSubmitMS,
			}, nil
		default:
			done()
			return nil, nil, nil, generationStartMetrics{}, fmt.Errorf("unexpected result type from worker")
		}
	case <-ctx.Done():
		done()
		return nil, nil, nil, generationStartMetrics{}, ctx.Err()
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
	} else if len(tokenIDs) == 0 && prepared.prompt != "" && engine != nil && !prepared.renderedChatSubmit {
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
	return isQwen35Request(modelHint, modelVariant) || isQwen36Request(modelHint, modelVariant)
}

func isGemmaRenderedTokenPathRequest(modelHint string, modelVariant string, tokenizerType string) bool {
	variant := strings.ToLower(strings.TrimSpace(modelVariant))
	tokenizer := strings.ToLower(strings.TrimSpace(tokenizerType))
	hint := strings.ToLower(modelHint)
	return strings.Contains(variant, "gemma") || strings.Contains(tokenizer, "gemma") ||
		strings.Contains(hint, "gemma")
}

func isRenderedTokenPathRequest(modelHint string, modelVariant string, tokenizerType string) bool {
	return isQwenRenderedTokenPathRequest(modelHint, modelVariant) ||
		isGemmaRenderedTokenPathRequest(modelHint, modelVariant, tokenizerType)
}

func validatePreparedContextWindow(engine domain.Engine, prepared preparedPrompt, requestInputIDs []int, maxTokens int) error {
	if engine == nil {
		return nil
	}
	maxCtx := engine.GetMaxContextTokens()
	if maxCtx <= 0 {
		return nil
	}
	inputTokens := len(requestInputIDs)
	if inputTokens == 0 {
		inputTokens = len(prepared.tokenIDs)
	}
	if inputTokens == 0 && prepared.prompt != "" {
		count, err := engine.CountTokens(prepared.prompt, false, false)
		if err != nil {
			return fmt.Errorf("count prompt tokens: %w", err)
		}
		inputTokens = count
	}
	if inputTokens > 0 && inputTokens+maxTokens > maxCtx {
		return domain.ErrInvalidRequest(fmt.Sprintf("prompt tokens (%d) plus max_tokens (%d) exceeds model context limit (%d)",
			inputTokens, maxTokens, maxCtx)).WithParam("max_tokens")
	}
	return nil
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

	return engine.GetEmbeddingsWithOptions(inputText, normalizePoolingType(req.PoolingType), req.Normalize)
}

// GetBatchEmbeddings processes multiple texts and returns their embeddings.
// This is more efficient than calling GetEmbeddings in a loop as it reduces
// per-call overhead, though the underlying engine still processes sequentially.
func (s *ChatService) GetBatchEmbeddings(texts []string) ([][]float32, error) {
	return s.GetBatchEmbeddingsWithOptions(texts, "mean", nil)
}

// GetBatchEmbeddingsWithOptions processes multiple texts with explicit pooling and normalization.
func (s *ChatService) GetBatchEmbeddingsWithOptions(texts []string, poolingType string, normalize *bool) ([][]float32, error) {
	engine := s.modelService.GetEngine()
	if engine == nil {
		return nil, errors.New("no model loaded")
	}

	results := make([][]float32, len(texts))
	poolingType = normalizePoolingType(poolingType)
	for i, text := range texts {
		embd, err := engine.GetEmbeddingsWithOptions(text, poolingType, normalize)
		if err != nil {
			return nil, fmt.Errorf("embedding failed for text %d: %w", i, err)
		}
		results[i] = embd
	}

	return results, nil
}

func normalizePoolingType(poolingType string) string {
	switch strings.ToLower(strings.TrimSpace(poolingType)) {
	case "cls", "last", "max":
		return strings.ToLower(strings.TrimSpace(poolingType))
	default:
		return "mean"
	}
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
	if isQwen35Request(modelHint, "") {
		return "qwen35"
	}
	return ""
}

func buildPromptCacheIdentity(req domain.ChatCompletionRequest, prepared preparedPrompt, modelHint string) promptcache.Identity {
	modelID := firstNonEmpty(req.Model, modelHint)
	_, templateHash := chatTemplateIdentity(prepared.chatTemplate)
	toolSchemaHash := ""
	if len(req.Tools) > 0 {
		if normalized, err := agenttools.NormalizeTools(req.Tools, req.ToolChoice); err == nil {
			toolSchemaHash = normalized.SchemaHash
		}
	}
	systemPromptHash := hashSystemPrompt(req.Messages)
	parserFamily := agenttools.ResolveParserFamily(agenttools.ModelDescriptor{
		ModelID:       modelID,
		TokenizerType: prepared.tokenizerType,
		ChatTemplate:  prepared.chatTemplate,
		ModelVariant:  prepared.modelVariant,
	})
	identity := promptcache.Identity{
		ModelID:          modelID,
		ModelPath:        modelHint,
		ModelFingerprint: modelHint,
		TokenizerHash:    shortHash(prepared.tokenizerType),
		ChatTemplateHash: templateHash,
		ToolSchemaHash:   toolSchemaHash,
		SystemPromptHash: systemPromptHash,
		LoraAdapterID:    req.LoraAdapter,
		KVDType:          firstNonEmpty(os.Getenv("DENSECORE_KV_TYPE"), "fp16"),
		RopeConfig:       "engine_default",
		GraphFamily:      firstNonEmpty(prepared.promptFamily, "generic"),
		SlidingWindow:    "default",
		SSMPolicy:        "none",
		ParserFamily:     string(parserFamily),
		Supported:        true,
	}
	if req.CacheControl != nil {
		identity.ConversationID = req.CacheControl.ConversationID
		identity.CacheID = req.CacheControl.CacheID
		identity.AffinityKey = req.CacheControl.AffinityKey
	}
	if isQwen35Request(modelID, prepared.modelVariant) || isQwen36Request(modelID, prepared.modelVariant) {
		identity.RequiresSSM = true
		identity.HasSSMSnapshot = false
		identity.SSMPolicy = "hybrid_ssm_snapshot_required"
	}
	if strings.Contains(strings.ToLower(modelID+" "+prepared.modelVariant), "gemma4") || prepared.promptFamily == "turn_tags" {
		identity.GraphFamily = "decoder_sliding_sharedkv"
		identity.SlidingWindow = "enabled"
	}
	return identity
}

func hashSystemPrompt(messages []domain.Message) string {
	var parts []string
	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		if role == roleSystem || role == roleDeveloper {
			parts = append(parts, msg.FlattenedText())
		}
	}
	return shortHash(strings.Join(parts, "\n\n"))
}

func shortHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:12])
}

func logPromptCacheDecision(req domain.ChatCompletionRequest, modelHint string, decision promptcache.Decision) {
	fields := []any{
		slog.String("request_id", ""),
		slog.String("model_id", firstNonEmpty(req.Model, modelHint)),
		slog.String("cache_id", cacheIDForLog(req)),
		slog.Int("reuse_tokens", decision.ReusedTokens),
		slog.String("reason", decision.InvalidationReason),
	}
	switch {
	case !decision.Enabled:
		slog.Info("prompt_cache_disabled", fields...)
	case decision.Hit:
		slog.Info("prompt_cache_hit", fields...)
	default:
		slog.Info("prompt_cache_miss", fields...)
	}
}

func cacheIDForLog(req domain.ChatCompletionRequest) string {
	if req.CacheControl == nil {
		return ""
	}
	return firstNonEmpty(req.CacheControl.CacheID, req.CacheControl.ConversationID, req.CacheControl.AffinityKey)
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

func chatPathDebugEnabled() bool {
	return envFlagEnabled("DENSECORE_DEBUG_CHAT_PATH") ||
		envFlagEnabled("DENSECORE_PARITY_DEBUG") ||
		envFlagEnabled("DENSECORE_DEBUG_RUNTIME_PATH") ||
		envFlagEnabled("DENSECORE_DEBUG_RUNTIME_PATH_TOKENS")
}

func allowDebugChatRawPassthrough() bool {
	return envFlagEnabled("DENSECORE_DEBUG_CHAT_RAW_PASSTHROUGH")
}

func durationMillis(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000.0
}

func logServerOverhead(mode string, promptTokens, completionTokens int, servicePrepareMS, engineSubmitMS, queueWaitMS, firstCallbackMS, lastCallbackMS, serverTotalMS float64) {
	if !envFlagEnabled("DENSECORE_DEBUG_SERVER_OVERHEAD") && !envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") {
		return
	}
	slog.Info("server_overhead",
		slog.String("mode", mode),
		slog.Float64("service_prepare_ms", servicePrepareMS),
		slog.Float64("queue_wait_ms", queueWaitMS),
		slog.Float64("engine_submit_ms", engineSubmitMS),
		slog.Float64("first_callback_ms", firstCallbackMS),
		slog.Float64("last_callback_ms", lastCallbackMS),
		slog.Float64("server_total_ms", serverTotalMS),
		slog.Int("prompt_tokens", promptTokens),
		slog.Int("completion_tokens", completionTokens),
	)
}

func logServerSubmitOverhead(mode string, promptTokens int, servicePrepareMS, engineSubmitMS, queueWaitMS, firstCallbackMS, lastCallbackMS, serverTotalMS float64) {
	if !envFlagEnabled("DENSECORE_DEBUG_SERVER_OVERHEAD") && !envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") {
		return
	}
	slog.Info("server_submit_overhead",
		slog.String("mode", mode),
		slog.Float64("service_prepare_ms", servicePrepareMS),
		slog.Float64("queue_wait_ms", queueWaitMS),
		slog.Float64("engine_submit_ms", engineSubmitMS),
		slog.Float64("first_callback_ms", firstCallbackMS),
		slog.Float64("last_callback_ms", lastCallbackMS),
		slog.Float64("server_total_ms", serverTotalMS),
		slog.Int("prompt_tokens", promptTokens),
	)
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
