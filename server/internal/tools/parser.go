package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync/atomic"

	"descore-server/internal/domain"
)

type ParseResult struct {
	Content          string
	ReasoningContent string
	ToolCalls        []domain.ToolCall
	ParserFamily     ParserFamily
	Recovered        bool
}

type Parser struct {
	family     ParserFamily
	normalized NormalizedTools
	buffer     strings.Builder
}

func (p *Parser) Feed(delta string) ([]domain.ToolCall, error) {
	p.buffer.WriteString(delta)
	return nil, nil
}

func (p *Parser) Finalize() (ParseResult, error) {
	return ParseOutput(p.buffer.String(), p.family, p.normalized)
}

func ParseOutput(output string, family ParserFamily, normalized NormalizedTools) (ParseResult, error) {
	reasoned := splitReasoningBlocks(output)
	var calls []domain.ToolCall
	var content string
	var recovered bool
	var err error
	switch family {
	case ParserFamilyQwenXML:
		calls, content, recovered, err = parseQwenXML(reasoned.Content)
	default:
		calls, content, recovered, err = parseHermesJSON(reasoned.Content)
	}
	if err != nil {
		metrics.toolParseFailure.Add(1)
		return ParseResult{Content: reasoned.Content, ReasoningContent: reasoned.Reasoning, ParserFamily: family}, err
	}
	for i := range calls {
		if calls[i].ID == "" {
			calls[i].ID = fmt.Sprintf("call_%d", i)
		}
		if calls[i].Type == "" {
			calls[i].Type = FunctionToolType
		}
		if err := ValidateToolCall(calls[i], normalized); err != nil {
			metrics.toolParseFailure.Add(1)
			return ParseResult{Content: reasoned.Content, ReasoningContent: reasoned.Reasoning, ParserFamily: family}, err
		}
	}
	if len(calls) > 0 {
		metrics.toolParseSuccess.Add(1)
		metrics.toolCallEmitted.Add(uint64(len(calls)))
	}
	if recovered {
		metrics.toolParseRecovery.Add(1)
	}
	if reasoned.Reasoning != "" {
		metrics.reasoningBlocks.Add(1)
	}
	return ParseResult{
		Content:          strings.TrimSpace(content),
		ReasoningContent: reasoned.Reasoning,
		ToolCalls:        calls,
		ParserFamily:     family,
		Recovered:        recovered,
	}, nil
}

type rawToolCall struct {
	ID        string          `json:"id"`
	Type      string          `json:"type"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	Function  struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

func parseHermesJSON(text string) ([]domain.ToolCall, string, bool, error) {
	candidates := jsonCandidates(text)
	var lastErr error
	for _, c := range candidates {
		jsonText := c.jsonText
		candidateRecovered := c.recovered
		if repaired, ok := repairJSON(jsonText); ok {
			jsonText = repaired
			candidateRecovered = true
		}
		calls, err := decodeToolCandidate(jsonText)
		if err == nil && len(calls) > 0 {
			return calls, strings.TrimSpace(strings.Replace(text, c.fullText, "", 1)), candidateRecovered, nil
		}
		lastErr = err
		if repaired, ok := repairJSON(c.jsonText); ok {
			calls, err = decodeToolCandidate(repaired)
			if err == nil && len(calls) > 0 {
				return calls, strings.TrimSpace(strings.Replace(text, c.fullText, "", 1)), true, nil
			}
			lastErr = err
		}
	}
	if looksLikeToolCall(text) && lastErr != nil {
		return nil, text, false, lastErr
	}
	return nil, text, false, nil
}

type jsonCandidate struct {
	jsonText  string
	fullText  string
	recovered bool
}

func jsonCandidates(text string) []jsonCandidate {
	var out []jsonCandidate
	fenceRE := regexp.MustCompile("(?is)```(?:json)?\\s*([\\s\\S]*?)\\s*```")
	for _, match := range fenceRE.FindAllStringSubmatch(text, -1) {
		if strings.Contains(match[1], "tool") || strings.Contains(match[1], "function") {
			out = append(out, jsonCandidate{jsonText: strings.TrimSpace(match[1]), fullText: match[0]})
		}
	}
	for _, span := range balancedJSONSpans(text) {
		c := strings.TrimSpace(text[span[0]:span[1]])
		if strings.Contains(c, "tool") || strings.Contains(c, "function") || strings.Contains(c, "arguments") {
			out = append(out, jsonCandidate{jsonText: c, fullText: c})
		}
	}
	if strings.Contains(text, "tool_calls") {
		start := strings.Index(text, "{")
		end := strings.LastIndex(text, "}")
		if start >= 0 && end > start {
			c := strings.TrimSpace(text[start : end+1])
			out = append(out, jsonCandidate{jsonText: c, fullText: c, recovered: true})
		}
	}
	return out
}

func balancedJSONSpans(text string) [][2]int {
	var spans [][2]int
	inString := false
	escaped := false
	start := -1
	depth := 0
	for i, r := range text {
		if start >= 0 {
			if inString {
				if escaped {
					escaped = false
				} else if r == '\\' {
					escaped = true
				} else if r == '"' {
					inString = false
				}
				continue
			}
			if r == '"' {
				inString = true
				continue
			}
			if r == '{' || r == '[' {
				depth++
			}
			if r == '}' || r == ']' {
				depth--
				if depth == 0 {
					spans = append(spans, [2]int{start, i + len(string(r))})
					start = -1
				}
			}
			continue
		}
		if r == '{' || r == '[' {
			start = i
			depth = 1
		}
	}
	return spans
}

func decodeToolCandidate(src string) ([]domain.ToolCall, error) {
	src = strings.TrimSpace(src)
	if repaired, ok := repairJSON(src); ok {
		src = repaired
	}
	var envelope struct {
		ToolCalls []rawToolCall `json:"tool_calls"`
	}
	if err := json.Unmarshal([]byte(src), &envelope); err == nil && len(envelope.ToolCalls) > 0 {
		return rawCallsToDomain(envelope.ToolCalls)
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(src), &obj); err == nil {
		if rawList, ok := obj["tool_calls"]; ok {
			var calls []rawToolCall
			if err := json.Unmarshal(rawList, &calls); err != nil {
				return nil, err
			}
			if len(calls) > 0 {
				return rawCallsToDomain(calls)
			}
		}
	}
	var single rawToolCall
	if err := json.Unmarshal([]byte(src), &single); err == nil && rawCallName(single) != "" {
		return rawCallsToDomain([]rawToolCall{single})
	}
	var many []rawToolCall
	if err := json.Unmarshal([]byte(src), &many); err == nil && len(many) > 0 {
		return rawCallsToDomain(many)
	}
	return nil, errors.New("no tool_calls JSON shape found")
}

func rawCallsToDomain(raw []rawToolCall) ([]domain.ToolCall, error) {
	calls := make([]domain.ToolCall, 0, len(raw))
	for i, item := range raw {
		name := rawCallName(item)
		if name == "" {
			return nil, errors.New("tool call missing function name")
		}
		args, err := normalizeArguments(rawCallArguments(item))
		if err != nil {
			return nil, err
		}
		calls = append(calls, domain.ToolCall{
			ID:   firstNonEmpty(item.ID, fmt.Sprintf("call_%d", i)),
			Type: firstNonEmpty(item.Type, FunctionToolType),
			Function: domain.ToolCallFunction{
				Name:      name,
				Arguments: args,
			},
		})
	}
	return calls, nil
}

func rawCallName(item rawToolCall) string {
	if item.Function.Name != "" {
		return item.Function.Name
	}
	return item.Name
}

func rawCallArguments(item rawToolCall) json.RawMessage {
	if len(item.Function.Arguments) > 0 {
		return item.Function.Arguments
	}
	return item.Arguments
}

func normalizeArguments(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "{}", nil
	}
	var asString string
	if err := json.Unmarshal(raw, &asString); err == nil {
		raw = json.RawMessage(asString)
	}
	var obj interface{}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return "", err
	}
	b, err := json.Marshal(obj)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func repairJSON(src string) (string, bool) {
	re := regexp.MustCompile(`,\s*([}\]])`)
	repaired := src
	for {
		next := re.ReplaceAllString(repaired, "$1")
		if next == repaired {
			break
		}
		repaired = next
	}
	if repaired != src {
		return repaired, true
	}
	return "", false
}

func looksLikeToolCall(text string) bool {
	lower := strings.ToLower(text)
	return strings.Contains(lower, "tool_call") || strings.Contains(lower, "\"arguments\"") || strings.Contains(lower, "<tool_call")
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

type MetricsSnapshot struct {
	ToolParseSuccessTotal  uint64
	ToolParseFailureTotal  uint64
	ToolParseRecoveryTotal uint64
	ToolCallEmittedTotal   uint64
	ReasoningBlocksTotal   uint64
}

var metrics struct {
	toolParseSuccess  atomic.Uint64
	toolParseFailure  atomic.Uint64
	toolParseRecovery atomic.Uint64
	toolCallEmitted   atomic.Uint64
	reasoningBlocks   atomic.Uint64
}

func SnapshotMetrics() MetricsSnapshot {
	return MetricsSnapshot{
		ToolParseSuccessTotal:  metrics.toolParseSuccess.Load(),
		ToolParseFailureTotal:  metrics.toolParseFailure.Load(),
		ToolParseRecoveryTotal: metrics.toolParseRecovery.Load(),
		ToolCallEmittedTotal:   metrics.toolCallEmitted.Load(),
		ReasoningBlocksTotal:   metrics.reasoningBlocks.Load(),
	}
}
