package domain

import (
	"context"
	"errors"
)

var ErrServiceBusy = errors.New("server is busy, please try again later")
var ErrStreamClosedWithoutTerminal = errors.New("stream closed without terminal event")

// Engine defines the interface for the inference engine
type Engine interface {
	GenerateStream(ctx context.Context, prompt string, maxTokens int, outputChan chan StreamEvent) error
	GenerateStreamWithFormat(ctx context.Context, prompt string, maxTokens int, jsonMode bool, outputChan chan StreamEvent) error
	GenerateStreamWithSampling(ctx context.Context, prompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan StreamEvent) error
	GenerateStreamTokensWithSampling(ctx context.Context, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan StreamEvent) error
	RenderChatPrompt(messages []Message, enableThinking *bool, preserveThinking *bool) (*RenderedChatPrompt, error)
	GetEmbeddings(prompt string) ([]float32, error)
	GetEmbeddingsWithOptions(prompt string, poolingType string, normalize *bool) ([]float32, error)
	GetMetrics() map[string]interface{}
	GetDetailedMetrics() *DetailedMetrics
	GetMaxContextTokens() int
	CountTokens(text string, addBOS bool, addEOS bool) (int, error)
	TokenizeText(text string, addBOS bool, addEOS bool) ([]int, error)
	GetTokenizerType() string
	GetChatTemplate() string
	Close()
	CancelRequest(reqID uintptr)
}

// LoadingStatus represents the model loading state
type LoadingStatus int32

const (
	StatusIdle    LoadingStatus = 0
	StatusLoading LoadingStatus = 1
	StatusReady   LoadingStatus = 2
	StatusFailed  LoadingStatus = 3
)

// ModelService defines the interface for managing models
type ModelService interface {
	LoadModel(mainModelPath, draftModelPath string, threads int) error
	UnloadModel() error
	GetCurrentModel() string
	GetModelIdentity() (id, ownedBy, root string)
	GetEngine() Engine
	// Loading status methods for K8s probes
	GetLoadingStatus() LoadingStatus
	GetLoadingError() error
	IsLoading() bool
}
