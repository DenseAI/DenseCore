package api

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strconv"
	"strings"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/promptcache"
)

type cacheAffinityDecision struct {
	Key      string
	Source   string
	Explicit bool
}

const maxRawAffinityMaterialBytes = 512

func applyCacheAffinity(w http.ResponseWriter, r *http.Request, req *domain.ChatCompletionRequest) cacheAffinityDecision {
	decision := buildCacheAffinityDecision(r, req)
	if decision.Key == "" {
		return decision
	}
	if w != nil {
		w.Header().Set(CacheAffinityResponseHeader, decision.Key)
		w.Header().Set(CacheAffinitySourceHeader, decision.Source)
		// Mirror the canonical key under the request-header name so clients can
		// replay it on follow-up requests without learning a second header.
		w.Header().Set(CacheAffinityRequestHeader, decision.Key)
	}
	promptcache.RecordAffinityKey(decision.Explicit)
	return decision
}

func buildCacheAffinityDecision(r *http.Request, req *domain.ChatCompletionRequest) cacheAffinityDecision {
	if req == nil {
		return cacheAffinityDecision{}
	}

	headerValue := ""
	if r != nil {
		headerValue = normalizeAffinityMaterial(r.Header.Get(CacheAffinityRequestHeader))
	}
	if headerValue != "" {
		ensureCacheControl(req).AffinityKey = headerValue
		return cacheAffinityDecision{
			Key:      hashedAffinityKey("header", req.Model, headerValue),
			Source:   "request_header",
			Explicit: true,
		}
	}

	if req.CacheControl != nil {
		if value := normalizeAffinityMaterial(req.CacheControl.AffinityKey); value != "" {
			req.CacheControl.AffinityKey = value
			return cacheAffinityDecision{
				Key:      hashedAffinityKey("affinity_key", req.Model, value),
				Source:   "cache_control.affinity_key",
				Explicit: true,
			}
		}
		if value := strings.TrimSpace(apiFirstNonEmpty(req.CacheControl.CacheID, req.CacheControl.ConversationID)); value != "" {
			return cacheAffinityDecision{
				Key:      hashedAffinityKey("cache_control", req.Model, value),
				Source:   "cache_control",
				Explicit: true,
			}
		}
	}

	if fallback := fallbackPromptAffinityMaterial(req); fallback != "" {
		return cacheAffinityDecision{
			Key:      hashedAffinityKey("prompt_fingerprint", req.Model, fallback),
			Source:   "prompt_fingerprint",
			Explicit: false,
		}
	}
	return cacheAffinityDecision{}
}

func ensureCacheControl(req *domain.ChatCompletionRequest) *domain.CacheControl {
	if req.CacheControl == nil {
		req.CacheControl = &domain.CacheControl{}
	}
	return req.CacheControl
}

func hashedAffinityKey(source, model, material string) string {
	sum := sha256.New()
	for _, part := range []string{
		"densecore-cache-affinity-v1",
		source,
		strings.TrimSpace(model),
		strings.TrimSpace(material),
	} {
		sum.Write([]byte(part))
		sum.Write([]byte{0})
	}
	return "dca1." + hex.EncodeToString(sum.Sum(nil))[:32]
}

func normalizeAffinityMaterial(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) <= maxRawAffinityMaterialBytes {
		return value
	}
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func fallbackPromptAffinityMaterial(req *domain.ChatCompletionRequest) string {
	if req == nil {
		return ""
	}
	var parts []string
	hasSystemOrDeveloper := false
	for _, msg := range req.Messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
		case "system", "developer":
			hasSystemOrDeveloper = true
			parts = append(parts, role, msg.FlattenedText())
		}
	}
	if hasSystemOrDeveloper {
		for _, msg := range req.Messages {
			role := strings.ToLower(strings.TrimSpace(msg.Role))
			if role == "system" || role == "developer" {
				continue
			}
			parts = append(parts, role, stablePrefixMaterial(msg.FlattenedText()))
			break
		}
	}
	if len(parts) == 0 && len(req.Messages) > 1 {
		// Route a normal multi-turn conversation by the already-established
		// prefix, not the current final user message that may change every turn.
		for _, msg := range req.Messages[:len(req.Messages)-1] {
			parts = append(parts, strings.ToLower(strings.TrimSpace(msg.Role)), msg.FlattenedText())
		}
	}
	if len(parts) == 0 && req.RawPrompt != "" {
		parts = append(parts, stablePrefixMaterial(req.RawPrompt))
	}
	if len(parts) == 0 && len(req.Messages) == 1 {
		parts = append(parts, strings.ToLower(strings.TrimSpace(req.Messages[0].Role)), stablePrefixMaterial(req.Messages[0].FlattenedText()))
	}
	if len(parts) == 0 && len(req.InputIDs) > 0 {
		builder := strings.Builder{}
		limit := len(req.InputIDs)
		if limit > 512 {
			limit = 512
		}
		for i := 0; i < limit; i++ {
			if i > 0 {
				builder.WriteByte(',')
			}
			builder.WriteString(strconv.Itoa(req.InputIDs[i]))
		}
		parts = append(parts, builder.String())
	}
	return strings.Join(parts, "\x00")
}

func stablePrefixMaterial(text string) string {
	text = strings.TrimSpace(text)
	const maxPrefixBytes = 8192
	if len(text) <= maxPrefixBytes {
		return text
	}
	return text[:maxPrefixBytes]
}
