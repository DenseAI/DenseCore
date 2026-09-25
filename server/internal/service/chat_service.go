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
	"time"

	cloudmw "github.com/DenseAI/DenseCloud/go/middleware"
	"github.com/google/uuid"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/promptcache"
	"github.com/DenseAI/DenseCore/server/internal/queue"
	agenttools "github.com/DenseAI/DenseCore/server/internal/tools"
)

var agentPromptCache = promptcache.NewManager(promptcache.ConfigFromEnv())

type ChatService struct {
	modelService domain.ModelService
	requestQueue *queue.RequestQueue
}

type generationStartMetrics struct {
	queueWaitMS    float64
	engineSubmitMS float64
}

type normalizedGenerationRequest struct {
	jsonMode            bool
	temperature         float64
	topP                float64
	topK                int
	repetitionPenalty   float64
	maxTokens           int
	stopSequences       []string
	inputIDs            []int
	allowedTokenIDs     []int
	allowedTokensStrict bool
	disallowedTokenIDs  []int
}

func NewChatService(modelService domain.ModelService, q *queue.RequestQueue) *ChatService {
	return &ChatService{
		modelService: modelService,
		requestQueue: q,
	}
}

// SubmitGeneration prepares against one leased engine and returns its owned
// event stream. HTTP consumers read it directly, without a per-token proxy.
func (s *ChatService) SubmitGeneration(ctx context.Context, req domain.ChatCompletionRequest, eventBufferSize int) (*domain.Generation, error) {
	serverStart := time.Now()
	engine, releaseEngine := acquireRequestEngine(s.modelService)
	transferred := false
	defer func() {
		if !transferred {
			releaseEngine()
		}
	}()
	if engine == nil {
		return nil, errors.New("no model loaded")
	}

	// Validate request parameters
	if req.MaxTokens < 0 {
		return nil, errors.New("max_tokens must be non-negative")
	}
	if maxCtx := engine.GetMaxContextTokens(); maxCtx > 0 && req.MaxTokens > maxCtx {
		return nil, &domain.ContextLimitError{Limit: maxCtx}
	}

	modelHint := s.activeModelHint(req)
	prepareStart := time.Now()
	prepared, err := s.preparePrompt(engine, req, modelHint)
	servicePrepareMS := durationMillis(time.Since(prepareStart))
	if err != nil {
		return nil, err
	}
	hasInputIDs := len(req.InputIDs) > 0
	if prepared.prompt == "" && !hasInputIDs {
		return nil, errors.New("no user message found")
	}
	if err := validatePreparedContextWindow(engine, prepared, req.InputIDs, req.MaxTokens); err != nil {
		return nil, err
	}

	transferred = true
	stream, completionCh, done, startMetrics, err := s.startGeneration(ctx, engine, req, modelHint, prepared, releaseEngine, eventBufferSize)
	if err != nil {
		return nil, err
	}
	go func() {
		<-completionCh
		logServerSubmitOverhead("stream", prepared.promptTokenCount, servicePrepareMS, startMetrics.engineSubmitMS,
			startMetrics.queueWaitMS, 0, 0, durationMillis(time.Since(serverStart)))
	}()
	return &domain.Generation{Events: stream, Done: completionCh, Cancel: done}, nil
}

// GenerateStream adapts the owned generation stream for existing internal users.
func (s *ChatService) GenerateStream(ctx context.Context, req domain.ChatCompletionRequest, outputChan chan domain.StreamEvent) error {
	generation, err := s.SubmitGeneration(ctx, req, 0)
	if err != nil {
		return err
	}
	defer generation.Cancel()
	defer close(outputChan)
	for {
		select {
		case event, ok := <-generation.Events:
			if !ok {
				return nil
			}
			select {
			case outputChan <- event:
			case <-ctx.Done():
				return ctx.Err()
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// GenerateSync runs the same C++ generation path as streaming requests but
// collects tokens inside the service, avoiding the handler-side proxy channel
// used only for SSE formatting.
func (s *ChatService) GenerateSync(ctx context.Context, req domain.ChatCompletionRequest) (string, int, int, error) {
	text, tokens, promptTokens, _, err := s.GenerateSyncWithMetadata(ctx, req)
	return text, tokens, promptTokens, err
}

// GenerateSyncWithMetadata preserves authoritative native usage and termination.
func (s *ChatService) GenerateSyncWithMetadata(ctx context.Context, req domain.ChatCompletionRequest) (string, int, int, *domain.GenerationCompletion, error) {
	serverStart := time.Now()
	engine, releaseEngine := acquireRequestEngine(s.modelService)
	transferred := false
	defer func() {
		if !transferred {
			releaseEngine()
		}
	}()
	if engine == nil {
		return "", 0, 0, nil, errors.New("no model loaded")
	}

	if req.MaxTokens < 0 {
		return "", 0, 0, nil, errors.New("max_tokens must be non-negative")
	}
	if maxCtx := engine.GetMaxContextTokens(); maxCtx > 0 && req.MaxTokens > maxCtx {
		return "", 0, 0, nil, &domain.ContextLimitError{Limit: maxCtx}
	}

	modelHint := s.activeModelHint(req)
	prepareStart := time.Now()
	prepared, err := s.preparePrompt(engine, req, modelHint)
	servicePrepareMS := durationMillis(time.Since(prepareStart))
	if err != nil {
		return "", 0, prepared.promptTokenCount, nil, err
	}
	hasInputIDs := len(req.InputIDs) > 0
	if prepared.prompt == "" && !hasInputIDs {
		return "", 0, prepared.promptTokenCount, nil, errors.New("no user message found")
	}
	if err := validatePreparedContextWindow(engine, prepared, req.InputIDs, req.MaxTokens); err != nil {
		return "", 0, prepared.promptTokenCount, nil, err
	}

	submitStart := time.Now()
	transferred = true
	stream, _, done, startMetrics, err := s.startGeneration(ctx, engine, req, modelHint, prepared, releaseEngine)
	if err != nil {
		return "", 0, prepared.promptTokenCount, nil, err
	}
	defer done()
	firstCallbackMS := 0.0
	lastCallbackMS := 0.0

	logQwen36 := isQwen36Request(modelHint, prepared.modelVariant)
	var responseBuilder strings.Builder
	var qwenMarkerFilter *QwenVisibleControlMarkerFilter
	if isQwenRenderedTokenPathRequest(modelHint, prepared.modelVariant) {
		qwenMarkerFilter = NewQwenVisibleControlMarkerFilter()
	}
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
				return "", completionTokens, prepared.promptTokenCount, nil, domain.ErrStreamClosedWithoutTerminal
			}
			if event.Token != "" {
				token := event.Token
				if qwenMarkerFilter != nil {
					token = qwenMarkerFilter.Filter(token)
				}
				if completionTokens == 0 {
					firstCallbackMS = durationMillis(time.Since(submitStart))
				}
				lastCallbackMS = durationMillis(time.Since(submitStart))
				if token != "" {
					responseBuilder.WriteString(token)
					completionTokens++
					visibleChunks++
					visibleChars += len(token)
				}
			}
			if !event.Terminal {
				continue
			}
			if qwenMarkerFilter != nil {
				token := qwenMarkerFilter.Flush()
				if token != "" {
					responseBuilder.WriteString(token)
					completionTokens++
					visibleChunks++
					visibleChars += len(token)
				}
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
				return "", completionTokens, prepared.promptTokenCount, nil, err
			}
			if event.Completion != nil {
				completionTokens = event.Completion.Tokens
			}
			logServerOverhead("sync", prepared.promptTokenCount, completionTokens, servicePrepareMS,
				startMetrics.engineSubmitMS, startMetrics.queueWaitMS, firstCallbackMS, lastCallbackMS,
				durationMillis(time.Since(serverStart)))
			return responseBuilder.String(), completionTokens, prepared.promptTokenCount, event.Completion, nil
		case <-ctx.Done():
			if logQwen36 {
				slog.Info("qwen36 chat sync context done",
					slog.Int("visible_chunks", visibleChunks),
					slog.Int("visible_chars", visibleChars),
					slog.String("error", ctx.Err().Error()),
				)
			}
			return "", completionTokens, prepared.promptTokenCount, nil, ctx.Err()
		}
	}
}

func (s *ChatService) activeModelHint(req domain.ChatCompletionRequest) string {
	current := ""
	modelID := ""
	root := ""
	if s != nil && s.modelService != nil {
		current = s.modelService.GetCurrentModel()
		modelID, _, root = s.modelService.GetModelIdentity()
	}
	for _, candidate := range []string{root, modelID, current, req.Model} {
		lower := strings.ToLower(strings.TrimSpace(candidate))
		if strings.Contains(lower, "lfm2") || strings.Contains(lower, "lfm") || isQwen36ModelHint(candidate) ||
			isQwen38ModelHint(candidate) || strings.Contains(lower, "gemma") {
			return candidate
		}
	}
	return firstNonEmpty(current, root, modelID, req.Model)
}

func (s *ChatService) startGeneration(ctx context.Context, engine domain.Engine, req domain.ChatCompletionRequest, modelHint string, prepared preparedPrompt, releaseEngine func(), eventBufferSize ...int) (<-chan domain.StreamEvent, <-chan struct{}, func(), generationStartMetrics, error) {
	if releaseEngine == nil {
		releaseEngine = func() {}
	}
	normalized := s.normalizeGenerationRequest(modelHint, prepared, req)
	qualityProfile := resolveChatQualityProfile(modelHint, prepared.tokenizerType, prepared.chatTemplate, req)
	s.logPromptPathDebug(engine, req, modelHint, prepared, normalized.temperature, normalized.topP, normalized.topK,
		normalized.repetitionPenalty, normalized.allowedTokenIDs, normalized.allowedTokensStrict,
		normalized.maxTokens, qualityProfile)

	cacheTokens := append([]int(nil), normalized.inputIDs...)
	if len(cacheTokens) == 0 && len(prepared.promptTokenIDs) > 0 {
		cacheTokens = append([]int(nil), prepared.promptTokenIDs...)
	}
	if len(cacheTokens) > 0 {
		identity := buildPromptCacheIdentity(req, prepared, modelHint)
		decision := agentPromptCache.LookupAndStore(identity, cacheTokens)
		logPromptCacheDecision(req, modelHint, decision)
	}

	ctx, cancel := context.WithCancel(ctx)
	done := cancel

	queuedReq := &queue.QueuedRequest{
		GenerationRequest: domain.GenerationRequest{
			Input:       preparedGenerationInput(prepared, normalized.inputIDs),
			MaxTokens:   normalized.maxTokens,
			LoraAdapter: req.LoraAdapter,
			JSONMode:    normalized.jsonMode,
			Sampling: domain.SamplingOptions{
				Temperature: normalized.temperature, TopP: normalized.topP, TopK: normalized.topK,
				RepetitionPenalty: normalized.repetitionPenalty, StopSequences: normalized.stopSequences,
			},
			Constraints: domain.TokenConstraints{
				AllowedTokenIDs: normalized.allowedTokenIDs, AllowedTokensStrict: normalized.allowedTokensStrict,
				DisallowedTokenIDs: normalized.disallowedTokenIDs,
			},
		},

		ID:            uuid.New().String(),
		TraceID:       cloudmw.GetRequestID(ctx),
		Priority:      queue.RequestPriority(0),
		Context:       ctx,
		ResultChan:    make(chan queue.GenerationResult, 1),
		Engine:        engine,
		ReleaseEngine: releaseEngine,
		ExpertCluster: req.ExpertCluster,
	}
	if len(eventBufferSize) > 0 {
		queuedReq.EventBufferSize = eventBufferSize[0]
	}
	if envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") {
		slog.Info("request lifecycle: enqueue",
			slog.String("trace_id", queuedReq.TraceID),
			slog.String("queue_request_id", queuedReq.ID),
			slog.Bool("streaming", true),
			slog.Int("max_tokens", queuedReq.MaxTokens),
			slog.Int("prompt_len", len(queuedReq.Input.Text)),
			slog.Int("input_ids", len(queuedReq.Input.TokenIDs)),
		)
	}
	if envFlagEnabled("DENSECORE_DEBUG_REQUEST_STATE") {
		submitAPI := "SubmitRequestWithSamplingConstraintsEx"
		if len(queuedReq.Input.TokenIDs) > 0 {
			if queuedReq.Input.Text != "" {
				submitAPI = "SubmitRenderedRequestIdsWithSamplingConstraintsCallbackEx"
			} else {
				submitAPI = "SubmitRequestIdsWithSamplingConstraintsEx"
			}
		}
		slog.Info("normalized_request_state",
			slog.String("submit_api", submitAPI),
			slog.Int("max_tokens", queuedReq.MaxTokens),
			slog.Float64("temperature", queuedReq.Sampling.Temperature),
			slog.Float64("top_p", queuedReq.Sampling.TopP),
			slog.Int("top_k", queuedReq.Sampling.TopK),
			slog.Float64("repetition_penalty", queuedReq.Sampling.RepetitionPenalty),
			slog.Any("stop_sequences", queuedReq.Sampling.StopSequences),
			slog.Bool("json_mode", queuedReq.JSONMode),
			slog.Int("allowed_token_ids", len(queuedReq.Constraints.AllowedTokenIDs)),
			slog.Bool("allowed_tokens_strict", queuedReq.Constraints.AllowedTokensStrict),
			slog.Int("disallowed_token_ids", len(queuedReq.Constraints.DisallowedTokenIDs)),
			slog.Bool("streaming", true),
			slog.Int("input_ids", len(queuedReq.Input.TokenIDs)),
		)
	}
	if isQwen36Request(modelHint, prepared.modelVariant) {
		slog.Info("qwen36 chat request enqueued",
			slog.String("trace_id", queuedReq.TraceID),
			slog.String("queue_request_id", queuedReq.ID),
			slog.String("prompt_source", prepared.promptSource),
			slog.String("prompt_family", prepared.promptFamily),
			slog.Int("max_tokens", queuedReq.MaxTokens),
			slog.Int("prompt_len", len(queuedReq.Input.Text)),
			slog.Int("input_ids", len(queuedReq.Input.TokenIDs)),
		)
	}

	if !s.requestQueue.Enqueue(queuedReq) {
		releaseEngine()
		done()
		return nil, nil, nil, generationStartMetrics{}, domain.ErrServiceBusy
	}

	select {
	case result := <-queuedReq.ResultChan:
		if result.Err != nil {
			done()
			return nil, nil, nil, generationStartMetrics{}, result.Err
		}
		g := result.Generation
		return g.Events, g.Done, func() { g.Cancel(); done() }, generationStartMetrics{
			queueWaitMS: result.QueueWaitMS, engineSubmitMS: result.EngineSubmitMS,
		}, nil
	case <-ctx.Done():
		done()
		return nil, nil, nil, generationStartMetrics{}, ctx.Err()
	}
}

func (s *ChatService) normalizeGenerationRequest(
	modelHint string,
	prepared preparedPrompt,
	req domain.ChatCompletionRequest,
) normalizedGenerationRequest {
	temperature, topP, topK, repetitionPenalty :=
		s.normalizeSampling(modelHint, prepared.tokenizerType, prepared.chatTemplate, req)
	jsonMode := req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object"
	stopSequences := append([]string(nil), req.Stop...)
	if len(stopSequences) == 0 && isLFM2PreparedRequest(modelHint, prepared) && !jsonMode {
		stopSequences = append(stopSequences, defaultLFM2StopSequences()...)
	}
	inputIDs := req.InputIDs
	if len(inputIDs) == 0 {
		inputIDs = prepared.tokenIDs
	}

	return normalizedGenerationRequest{
		jsonMode:            jsonMode,
		temperature:         temperature,
		topP:                topP,
		topK:                topK,
		repetitionPenalty:   repetitionPenalty,
		maxTokens:           req.MaxTokens,
		stopSequences:       stopSequences,
		inputIDs:            append([]int(nil), inputIDs...),
		allowedTokenIDs:     append([]int(nil), req.AllowedTokenIDs...),
		allowedTokensStrict: req.AllowedTokensStrict,
		disallowedTokenIDs:  append([]int(nil), req.DisallowedTokenIDs...),
	}
}

func (s *ChatService) logPromptPathDebug(engine domain.Engine, req domain.ChatCompletionRequest, modelHint string,
	prepared preparedPrompt, temperature, topP float64, topK int, repetitionPenalty float64,
	allowedTokenIDs []int, allowedTokensStrict bool, maxTokens int, qualityProfile string) {
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
	}
	if tokenizeErr != "" {
		fields = append(fields, slog.String("input_tokenize_error", tokenizeErr))
	}
	slog.Info("chat request debug", fields...)
}

func (s *ChatService) GetEmbeddings(req domain.EmbeddingRequest) ([]float32, error) {
	engine, releaseEngine := acquireRequestEngine(s.modelService)
	defer releaseEngine()
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
	engine, releaseEngine := acquireRequestEngine(s.modelService)
	defer releaseEngine()
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
	if isQwen35Request(modelID, prepared.modelVariant) || isQwen36Request(modelID, prepared.modelVariant) ||
		isQwen38ModelHint(modelID) || isQwen38ModelHint(prepared.modelVariant) {
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
