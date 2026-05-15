package promptcache

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Mode string

const (
	ModeAuto Mode = "auto"
	ModeOn   Mode = "on"
	ModeOff  Mode = "off"
)

type Config struct {
	Mode        Mode
	MaxBytes    int64
	MaxSessions int
	TTL         time.Duration
	Debug       bool
}

type Identity struct {
	ModelID          string
	ModelPath        string
	ModelFingerprint string
	TokenizerHash    string
	ChatTemplateHash string
	ToolSchemaHash   string
	SystemPromptHash string
	LoraAdapterID    string
	KVDType          string
	RopeConfig       string
	GraphFamily      string
	SlidingWindow    string
	SSMPolicy        string
	ParserFamily     string
	ConversationID   string
	CacheID          string
	RequiresSSM      bool
	HasSSMSnapshot   bool
	Supported        bool
}

type Decision struct {
	Enabled              bool
	Hit                  bool
	CacheKey             string
	ReusedTokens         int
	PrefillTokensSkipped int
	RestoreMS            float64
	InvalidationReason   string
}

type entry struct {
	key            string
	lineage        string
	identity       Identity
	tokens         []int
	bytes          int64
	expiresAt      time.Time
	lastAccess     time.Time
	hasSSMSnapshot bool
}

type Manager struct {
	mu      sync.Mutex
	config  Config
	entries map[string]*entry
	lineage map[string][]string
	bytes   int64
	now     func() time.Time
}

func NewManager(config Config) *Manager {
	if config.Mode == "" {
		config.Mode = ModeAuto
	}
	if config.MaxBytes <= 0 {
		config.MaxBytes = 1 << 30
	}
	if config.MaxSessions <= 0 {
		config.MaxSessions = 1024
	}
	if config.TTL <= 0 {
		config.TTL = time.Hour
	}
	return &Manager{
		config:  config,
		entries: make(map[string]*entry),
		lineage: make(map[string][]string),
		now:     time.Now,
	}
}

func ConfigFromEnv() Config {
	return Config{
		Mode:        parseMode(os.Getenv("DENSECORE_AGENT_PROMPT_CACHE")),
		MaxBytes:    parseInt64Env("DENSECORE_AGENT_PROMPT_CACHE_MAX_BYTES", 1<<30),
		MaxSessions: int(parseInt64Env("DENSECORE_AGENT_PROMPT_CACHE_MAX_SESSIONS", 1024)),
		TTL:         time.Duration(parseInt64Env("DENSECORE_AGENT_PROMPT_CACHE_TTL_SECONDS", 3600)) * time.Second,
		Debug:       parseBoolEnv("DENSECORE_AGENT_PROMPT_CACHE_DEBUG"),
	}
}

func parseMode(raw string) Mode {
	switch Mode(strings.ToLower(strings.TrimSpace(raw))) {
	case ModeOn:
		return ModeOn
	case ModeOff:
		return ModeOff
	default:
		return ModeAuto
	}
}

func parseInt64Env(name string, fallback int64) int64 {
	if raw := strings.TrimSpace(os.Getenv(name)); raw != "" {
		if v, err := strconv.ParseInt(raw, 10, 64); err == nil && v > 0 {
			return v
		}
	}
	return fallback
}

func parseBoolEnv(name string) bool {
	raw := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
	return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

func (m *Manager) LookupAndStore(identity Identity, tokens []int) Decision {
	decision := m.Lookup(identity, tokens)
	if decision.Enabled && decision.InvalidationReason == "" {
		m.Store(identity, tokens)
	}
	return decision
}

func (m *Manager) Lookup(identity Identity, tokens []int) Decision {
	start := m.now()
	decision := Decision{Enabled: true}
	if m.config.Mode == ModeOff {
		decision.Enabled = false
		decision.InvalidationReason = "disabled_by_env"
		metrics.promptCacheDisabled.Add(1)
		return decision
	}
	if !identity.Supported && m.config.Mode == ModeAuto {
		decision.Enabled = false
		decision.InvalidationReason = "unsupported_model_family"
		metrics.promptCacheDisabled.Add(1)
		return decision
	}
	if isUnsafeGraphFamily(identity.GraphFamily, identity.SlidingWindow) {
		decision.Enabled = false
		decision.InvalidationReason = "unsupported_graph_family"
		metrics.promptCacheDisabled.Add(1)
		return decision
	}
	if identity.RequiresSSM && !identity.HasSSMSnapshot {
		decision.InvalidationReason = "ssm_snapshot_missing"
		metrics.ssmSnapshotMiss.Add(1)
		metrics.promptCacheMiss.Add(1)
		return decision
	}
	key := CacheKey(identity, tokens)
	decision.CacheKey = key

	m.mu.Lock()
	defer m.mu.Unlock()
	m.evictExpiredLocked()
	if e := m.entries[key]; e != nil && equalTokens(e.tokens, tokens) {
		e.lastAccess = m.now()
		decision.Hit = true
		decision.ReusedTokens = len(tokens)
		decision.PrefillTokensSkipped = len(tokens)
		metrics.promptCacheHit.Add(1)
		metrics.promptCacheTokensReused.Add(uint64(decision.ReusedTokens))
		metrics.promptCachePrefillSkipped.Add(uint64(decision.PrefillTokensSkipped))
		if identity.RequiresSSM {
			metrics.ssmSnapshotHit.Add(1)
		}
		metrics.promptCacheLookupNanos.Add(uint64(m.now().Sub(start).Nanoseconds()))
		return decision
	}
	if reused := m.longestVerifiedPrefixLocked(identity, tokens); reused > 0 {
		decision.Hit = true
		decision.ReusedTokens = reused
		decision.PrefillTokensSkipped = reused
		metrics.promptCacheHit.Add(1)
		metrics.promptCacheTokensReused.Add(uint64(reused))
		metrics.promptCachePrefillSkipped.Add(uint64(reused))
		metrics.promptCacheLookupNanos.Add(uint64(m.now().Sub(start).Nanoseconds()))
		return decision
	}
	metrics.promptCacheMiss.Add(1)
	metrics.promptCacheLookupNanos.Add(uint64(m.now().Sub(start).Nanoseconds()))
	return decision
}

func (m *Manager) Store(identity Identity, tokens []int) {
	if len(tokens) == 0 {
		return
	}
	key := CacheKey(identity, tokens)
	lineage := lineageKey(identity)
	copied := append([]int(nil), tokens...)
	e := &entry{
		key:            key,
		lineage:        lineage,
		identity:       identity,
		tokens:         copied,
		bytes:          int64(len(copied) * 4),
		expiresAt:      m.now().Add(m.config.TTL),
		lastAccess:     m.now(),
		hasSSMSnapshot: identity.HasSSMSnapshot,
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if old := m.entries[key]; old != nil {
		m.bytes -= old.bytes
	}
	m.entries[key] = e
	m.lineage[lineage] = append(m.lineage[lineage], key)
	if identity.ConversationID != "" || identity.CacheID != "" {
		sessionLineage := sessionLineageKey(identity)
		m.lineage[sessionLineage] = append(m.lineage[sessionLineage], key)
	}
	m.bytes += e.bytes
	m.evictPressureLocked()
}

func (m *Manager) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = make(map[string]*entry)
	m.lineage = make(map[string][]string)
	m.bytes = 0
}

func (m *Manager) longestVerifiedPrefixLocked(identity Identity, tokens []int) int {
	lineages := []string{lineageKey(identity)}
	if identity.ConversationID != "" || identity.CacheID != "" {
		lineages = append([]string{sessionLineageKey(identity)}, lineages...)
	}
	best := 0
	for _, line := range lineages {
		for _, key := range m.lineage[line] {
			e := m.entries[key]
			if e == nil || !compatibleIdentity(e.identity, identity) {
				continue
			}
			if identity.RequiresSSM && !e.hasSSMSnapshot {
				continue
			}
			if n := commonPrefix(e.tokens, tokens); n > best {
				best = n
			}
		}
	}
	return best
}

func (m *Manager) evictExpiredLocked() {
	now := m.now()
	for key, e := range m.entries {
		if now.After(e.expiresAt) {
			delete(m.entries, key)
			m.bytes -= e.bytes
			metrics.promptCacheEvictions.Add(1)
		}
	}
}

func (m *Manager) evictPressureLocked() {
	for m.bytes > m.config.MaxBytes || len(m.entries) > m.config.MaxSessions {
		var oldestKey string
		var oldest time.Time
		for key, e := range m.entries {
			if oldestKey == "" || e.lastAccess.Before(oldest) {
				oldestKey = key
				oldest = e.lastAccess
			}
		}
		if oldestKey == "" {
			return
		}
		m.bytes -= m.entries[oldestKey].bytes
		delete(m.entries, oldestKey)
		metrics.promptCacheEvictions.Add(1)
	}
}

func CacheKey(identity Identity, tokens []int) string {
	sum := sha256.New()
	for _, part := range []string{
		identity.ModelID,
		identity.ModelPath,
		identity.ModelFingerprint,
		identity.TokenizerHash,
		identity.ChatTemplateHash,
		identity.ToolSchemaHash,
		identity.SystemPromptHash,
		identity.LoraAdapterID,
		identity.KVDType,
		identity.RopeConfig,
		identity.GraphFamily,
		identity.SlidingWindow,
		identity.SSMPolicy,
		identity.ParserFamily,
	} {
		sum.Write([]byte(part))
		sum.Write([]byte{0})
	}
	for _, tok := range tokens {
		sum.Write([]byte(fmt.Sprintf("%d,", tok)))
	}
	return hex.EncodeToString(sum.Sum(nil))
}

func compatibleIdentity(a, b Identity) bool {
	return a.ModelID == b.ModelID &&
		a.ModelPath == b.ModelPath &&
		a.ModelFingerprint == b.ModelFingerprint &&
		a.TokenizerHash == b.TokenizerHash &&
		a.ChatTemplateHash == b.ChatTemplateHash &&
		a.ToolSchemaHash == b.ToolSchemaHash &&
		a.SystemPromptHash == b.SystemPromptHash &&
		a.LoraAdapterID == b.LoraAdapterID &&
		a.KVDType == b.KVDType &&
		a.RopeConfig == b.RopeConfig &&
		a.GraphFamily == b.GraphFamily &&
		a.SlidingWindow == b.SlidingWindow &&
		a.SSMPolicy == b.SSMPolicy &&
		a.ParserFamily == b.ParserFamily
}

func lineageKey(identity Identity) string {
	copy := identity
	copy.ConversationID = ""
	copy.CacheID = ""
	return CacheKey(copy, nil)
}

func sessionLineageKey(identity Identity) string {
	return lineageKey(identity) + "\x00" + identity.ConversationID + "\x00" + identity.CacheID
}

func commonPrefix(a, b []int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func equalTokens(a, b []int) bool {
	return len(a) == len(b) && commonPrefix(a, b) == len(a)
}

func isUnsafeGraphFamily(graphFamily, slidingWindow string) bool {
	lower := strings.ToLower(graphFamily + " " + slidingWindow)
	return strings.Contains(lower, "sliding") || strings.Contains(lower, "sharedkv") || strings.Contains(lower, "shared-kv")
}

type MetricsSnapshot struct {
	PromptCacheHitTotal            uint64
	PromptCacheMissTotal           uint64
	PromptCacheDisabledTotal       uint64
	PromptCacheEvictionsTotal      uint64
	PromptCacheTokensReusedTotal   uint64
	PromptCachePrefillSkippedTotal uint64
	PromptCacheLookupSeconds       float64
	PromptCacheRestoreSeconds      float64
	SSMSnapshotHitTotal            uint64
	SSMSnapshotMissTotal           uint64
	SSMSnapshotRestoreFailureTotal uint64
}

var metrics struct {
	promptCacheHit            atomic.Uint64
	promptCacheMiss           atomic.Uint64
	promptCacheDisabled       atomic.Uint64
	promptCacheEvictions      atomic.Uint64
	promptCacheTokensReused   atomic.Uint64
	promptCachePrefillSkipped atomic.Uint64
	promptCacheLookupNanos    atomic.Uint64
	promptCacheRestoreNanos   atomic.Uint64
	ssmSnapshotHit            atomic.Uint64
	ssmSnapshotMiss           atomic.Uint64
	ssmSnapshotRestoreFailure atomic.Uint64
}

func SnapshotMetrics() MetricsSnapshot {
	return MetricsSnapshot{
		PromptCacheHitTotal:            metrics.promptCacheHit.Load(),
		PromptCacheMissTotal:           metrics.promptCacheMiss.Load(),
		PromptCacheDisabledTotal:       metrics.promptCacheDisabled.Load(),
		PromptCacheEvictionsTotal:      metrics.promptCacheEvictions.Load(),
		PromptCacheTokensReusedTotal:   metrics.promptCacheTokensReused.Load(),
		PromptCachePrefillSkippedTotal: metrics.promptCachePrefillSkipped.Load(),
		PromptCacheLookupSeconds:       float64(metrics.promptCacheLookupNanos.Load()) / float64(time.Second),
		PromptCacheRestoreSeconds:      float64(metrics.promptCacheRestoreNanos.Load()) / float64(time.Second),
		SSMSnapshotHitTotal:            metrics.ssmSnapshotHit.Load(),
		SSMSnapshotMissTotal:           metrics.ssmSnapshotMiss.Load(),
		SSMSnapshotRestoreFailureTotal: metrics.ssmSnapshotRestoreFailure.Load(),
	}
}
