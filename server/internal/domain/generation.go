package domain

import "context"

type GenerationInputKind uint8

const (
	GenerationInputText GenerationInputKind = iota
	GenerationInputTokenIDs
	GenerationInputRenderedText
	GenerationInputRenderedTokenIDs
)

// GenerationInput makes tokenization and template ownership explicit. Rendered
// token IDs retain Text for suppression metadata, never for re-tokenization.
type GenerationInput struct {
	Kind     GenerationInputKind
	Text     string
	TokenIDs []int
}

type SamplingOptions struct {
	Temperature       float64
	TopP              float64
	TopK              int
	RepetitionPenalty float64
	StopSequences     []string
}

type TokenConstraints struct {
	AllowedTokenIDs     []int
	AllowedTokensStrict bool
	DisallowedTokenIDs  []int
}

// GenerationRequest is the engine-facing request after protocol preparation.
// Slice storage must remain unchanged until SubmitGeneration returns.
type GenerationRequest struct {
	Input       GenerationInput
	Sampling    SamplingOptions
	Constraints TokenConstraints
	MaxTokens   int
	LoraAdapter string
	JSONMode    bool
	// EventBufferSize controls transport backpressure; zero uses the adapter default.
	EventBufferSize int
}

// Generation owns its event stream and cancellation. Only the adapter closes
// Events and Done. Done means terminal delivery (or cancellation discard) and
// callback registration cleanup, not native worker retirement. Engine teardown
// still joins native producers before releasing their resources.
type Generation struct {
	Events <-chan StreamEvent
	Done   <-chan struct{}
	Cancel context.CancelFunc
}

type GenerationEngine interface {
	SubmitGeneration(context.Context, GenerationRequest) (*Generation, error)
}
