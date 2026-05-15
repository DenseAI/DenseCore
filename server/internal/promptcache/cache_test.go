package promptcache

import (
	"testing"
	"time"
)

func baseIdentity() Identity {
	return Identity{
		ModelID:          "qwen-coder",
		ModelPath:        "/models/qwen.gguf",
		ModelFingerprint: "fp",
		TokenizerHash:    "tok",
		ChatTemplateHash: "tmpl",
		ToolSchemaHash:   "tools",
		SystemPromptHash: "system",
		KVDType:          "fp16",
		RopeConfig:       "rope",
		GraphFamily:      "decoder",
		SSMPolicy:        "none",
		ParserFamily:     "qwen_xml",
		Supported:        true,
	}
}

func TestSameRenderedPromptCacheHit(t *testing.T) {
	m := NewManager(Config{Mode: ModeOn, TTL: time.Hour})
	id := baseIdentity()
	tokens := []int{1, 2, 3}
	if got := m.LookupAndStore(id, tokens); got.Hit {
		t.Fatalf("first lookup should miss")
	}
	got := m.LookupAndStore(id, tokens)
	if !got.Hit || got.ReusedTokens != 3 || got.PrefillTokensSkipped != 3 {
		t.Fatalf("expected exact hit, got %#v", got)
	}
}

func TestIdentityChangesMiss(t *testing.T) {
	m := NewManager(Config{Mode: ModeOn, TTL: time.Hour})
	id := baseIdentity()
	m.LookupAndStore(id, []int{1, 2, 3})
	changed := id
	changed.ChatTemplateHash = "other"
	if got := m.LookupAndStore(changed, []int{1, 2, 3}); got.Hit {
		t.Fatalf("different chat template must miss")
	}
	changed = id
	changed.ToolSchemaHash = "other"
	if got := m.LookupAndStore(changed, []int{1, 2, 3}); got.Hit {
		t.Fatalf("different tool schema must miss")
	}
	changed = id
	changed.LoraAdapterID = "adapter"
	if got := m.LookupAndStore(changed, []int{1, 2, 3}); got.Hit {
		t.Fatalf("different LoRA must miss")
	}
	changed = id
	changed.ModelFingerprint = "other"
	if got := m.LookupAndStore(changed, []int{1, 2, 3}); got.Hit {
		t.Fatalf("different model fingerprint must miss")
	}
}

func TestPartialPrefixReuseAndTokenVerification(t *testing.T) {
	m := NewManager(Config{Mode: ModeOn, TTL: time.Hour})
	id := baseIdentity()
	m.LookupAndStore(id, []int{1, 2, 3, 4})
	got := m.LookupAndStore(id, []int{1, 2, 3, 9, 10})
	if !got.Hit || got.ReusedTokens != 3 {
		t.Fatalf("expected verified partial prefix hit, got %#v", got)
	}
	got = m.LookupAndStore(id, []int{7, 2, 3, 4})
	if got.Hit {
		t.Fatalf("nonmatching first token must not reuse")
	}
}

func TestTTLEvictionAndMemoryPressure(t *testing.T) {
	now := time.Unix(100, 0)
	m := NewManager(Config{Mode: ModeOn, TTL: time.Second, MaxBytes: 16, MaxSessions: 8})
	m.now = func() time.Time { return now }
	id := baseIdentity()
	m.LookupAndStore(id, []int{1, 2, 3})
	now = now.Add(2 * time.Second)
	if got := m.LookupAndStore(id, []int{1, 2, 3}); got.Hit {
		t.Fatalf("expired entry must miss")
	}
	m.Store(id, []int{1, 2, 3, 4, 5})
	if len(m.entries) != 0 {
		t.Fatalf("entry over max bytes should be evicted")
	}
}

func TestDisabledUnsupportedGraphAndSSMSafeMiss(t *testing.T) {
	m := NewManager(Config{Mode: ModeOff, TTL: time.Hour})
	if got := m.LookupAndStore(baseIdentity(), []int{1}); got.Enabled {
		t.Fatalf("off mode should disable cache")
	}
	m = NewManager(Config{Mode: ModeAuto, TTL: time.Hour})
	id := baseIdentity()
	id.GraphFamily = "decoder_sliding_sharedkv"
	if got := m.LookupAndStore(id, []int{1}); got.Enabled || got.InvalidationReason != "unsupported_graph_family" {
		t.Fatalf("unsafe graph should disable, got %#v", got)
	}
	id = baseIdentity()
	id.RequiresSSM = true
	id.HasSSMSnapshot = false
	if got := m.LookupAndStore(id, []int{1}); got.Hit || got.InvalidationReason != "ssm_snapshot_missing" {
		t.Fatalf("missing SSM snapshot should be a safe miss, got %#v", got)
	}
	id.HasSSMSnapshot = true
	m.LookupAndStore(id, []int{1, 2})
	if got := m.LookupAndStore(id, []int{1, 2}); !got.Hit {
		t.Fatalf("matching SSM snapshot boundary should hit, got %#v", got)
	}
}
