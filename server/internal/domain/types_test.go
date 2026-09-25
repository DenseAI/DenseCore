package domain

import (
	"encoding/json"
	"testing"
)

func TestChatCompletionRequestUnmarshalTokenConstraints(t *testing.T) {
	var req ChatCompletionRequest
	err := json.Unmarshal([]byte(`{
		"model":"densecore",
		"messages":[{"role":"user","content":"The capital of France is"}],
		"max_tokens":1,
		"allowed_token_ids":[1,2,3],
		"allowed_tokens_strict":true,
		"disallowed_token_ids":[4,5]
	}`), &req)
	if err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}

	if len(req.AllowedTokenIDs) != 3 || req.AllowedTokenIDs[0] != 1 || req.AllowedTokenIDs[2] != 3 {
		t.Fatalf("expected allowed token ids to round-trip, got %+v", req.AllowedTokenIDs)
	}
	if !req.AllowedTokensStrict {
		t.Fatalf("expected allowed_tokens_strict to be true")
	}
	if len(req.DisallowedTokenIDs) != 2 || req.DisallowedTokenIDs[0] != 4 || req.DisallowedTokenIDs[1] != 5 {
		t.Fatalf("expected disallowed token ids to round-trip, got %+v", req.DisallowedTokenIDs)
	}
}
