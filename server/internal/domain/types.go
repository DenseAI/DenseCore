package domain

// StreamEvent represents a token event in the stream
type StreamEvent struct {
	Token      string
	IsFinished bool
}

// OpenAI-compatible request/response structures
type ChatCompletionRequest struct {
	Model             string          `json:"model"`
	Messages          []Message       `json:"messages"`
	InputIDs          []int           `json:"input_ids,omitempty"`
	LoraAdapter       string          `json:"lora_adapter,omitempty"`
	MaxTokens         int             `json:"max_tokens,omitempty"`
	Temperature       float64         `json:"temperature,omitempty"`
	TopP              float64         `json:"top_p,omitempty"`
	TopK              int             `json:"top_k,omitempty"`
	RepetitionPenalty float64         `json:"repetition_penalty,omitempty"`
	Stop              []string        `json:"stop,omitempty"`
	Stream            bool            `json:"stream,omitempty"`
	ResponseFormat    *ResponseFormat `json:"response_format,omitempty"`
	ExpertCluster     []int           `json:"expert_cluster,omitempty"`
}

type ResponseFormat struct {
	Type string `json:"type"` // "text" or "json_object"
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
}

type ChunkChoice struct {
	Index        int         `json:"index"`
	Delta        ChunkDelta  `json:"delta"`
	FinishReason interface{} `json:"finish_reason"`
}

type ChunkDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// Embedding structures
type EmbeddingRequest struct {
	Model          string      `json:"model"`
	Input          interface{} `json:"input"`                     // string or []string
	PoolingType    string      `json:"pooling_type,omitempty"`    // mean, cls, last, max
	Normalize      *bool       `json:"normalize,omitempty"`       // L2 normalize (default: true)
	EncodingFormat string      `json:"encoding_format,omitempty"` // float (default) or base64
}

// GetInputTexts extracts texts from Input field
func (r *EmbeddingRequest) GetInputTexts() []string {
	switch v := r.Input.(type) {
	case string:
		return []string{v}
	case []interface{}:
		texts := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				texts = append(texts, s)
			}
		}
		return texts
	default:
		return nil
	}
}

type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  Usage           `json:"usage"`
}

type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// DetailedMetrics structure
type DetailedMetrics struct {
	// Request metrics
	ActiveRequests    int   `json:"active_requests"`
	TotalRequests     int64 `json:"total_requests"`
	CompletedRequests int64 `json:"completed_requests"`
	FailedRequests    int64 `json:"failed_requests"`
	PendingRequests   int   `json:"pending_requests"`

	// Token metrics
	TotalTokensGenerated int64   `json:"total_tokens_generated"`
	TotalPromptTokens    int64   `json:"total_prompt_tokens"`
	TokensPerSecond      float32 `json:"tokens_per_second"`

	// Latency metrics (milliseconds)
	AvgTimeToFirstToken float32 `json:"avg_time_to_first_token_ms"`
	P50TimeToFirstToken float32 `json:"p50_time_to_first_token_ms"`
	P90TimeToFirstToken float32 `json:"p90_time_to_first_token_ms"`
	P99TimeToFirstToken float32 `json:"p99_time_to_first_token_ms"`

	AvgInterTokenLatency float32 `json:"avg_inter_token_latency_ms"`
	P50InterTokenLatency float32 `json:"p50_inter_token_latency_ms"`
	P90InterTokenLatency float32 `json:"p90_inter_token_latency_ms"`
	P99InterTokenLatency float32 `json:"p99_inter_token_latency_ms"`

	AvgQueueWaitTime float32 `json:"avg_queue_wait_time_ms"`
	P99QueueWaitTime float32 `json:"p99_queue_wait_time_ms"`

	// KV Cache metrics
	KVCacheUsageBlocks  int     `json:"kv_cache_usage_blocks"`
	KVCacheTotalBlocks  int     `json:"kv_cache_total_blocks"`
	KVCacheUsagePercent float32 `json:"kv_cache_usage_percent"`

	// Batch metrics
	AvgBatchSize     float32 `json:"avg_batch_size"`
	CurrentBatchSize int     `json:"current_batch_size"`

	// Error metrics
	OOMErrors     int `json:"oom_errors"`
	TimeoutErrors int `json:"timeout_errors"`
}

// Rerank structures (Cohere-compatible)
type RerankRequest struct {
	Model           string        `json:"model"`
	Query           string        `json:"query"`
	Documents       []interface{} `json:"documents"` // string or RerankDocument
	TopN            int           `json:"top_n,omitempty"`
	ReturnDocuments bool          `json:"return_documents,omitempty"`
}

// GetDocumentTexts extracts texts from Documents field
func (r *RerankRequest) GetDocumentTexts() []string {
	texts := make([]string, 0, len(r.Documents))
	for _, doc := range r.Documents {
		switch v := doc.(type) {
		case string:
			texts = append(texts, v)
		case map[string]interface{}:
			if text, ok := v["text"].(string); ok {
				texts = append(texts, text)
			}
		}
	}
	return texts
}

type RerankResponse struct {
	ID      string         `json:"id"`
	Results []RerankResult `json:"results"`
	Model   string         `json:"model"`
	Usage   Usage          `json:"usage,omitempty"`
}

type RerankResult struct {
	Index          int             `json:"index"`
	RelevanceScore float32         `json:"relevance_score"`
	Document       *RerankDocument `json:"document,omitempty"`
}

type RerankDocument struct {
	Text string `json:"text"`
}

// Function Calling structures
type Tool struct {
	Type     string       `json:"type"` // "function"
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // "function"
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// Extended Message with tool support
type MessageWithTools struct {
	Role       string     `json:"role"` // system, user, assistant, tool
	Content    *string    `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// Extended ChatCompletionRequest with tools
type ChatCompletionRequestWithTools struct {
	ChatCompletionRequest
	Tools      []Tool      `json:"tools,omitempty"`
	ToolChoice interface{} `json:"tool_choice,omitempty"` // "none", "auto", "required", or object
}
