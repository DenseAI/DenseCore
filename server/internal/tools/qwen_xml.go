package tools

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

func parseQwenXML(text string) ([]domain.ToolCall, string, bool, error) {
	blocks := extractToolBlocks(text)
	if len(blocks) == 0 {
		return parseHermesJSON(text)
	}
	var calls []domain.ToolCall
	recovered := false
	clean := text
	for idx, block := range blocks {
		call, blockRecovered, err := parseQwenBlock(block.body, idx)
		if err != nil {
			return nil, text, recovered || blockRecovered, err
		}
		recovered = recovered || blockRecovered || block.recovered
		calls = append(calls, call)
		clean = strings.Replace(clean, block.full, "", 1)
	}
	return calls, strings.TrimSpace(clean), recovered, nil
}

type toolBlock struct {
	full      string
	body      string
	recovered bool
}

func extractToolBlocks(text string) []toolBlock {
	var blocks []toolBlock
	lower := strings.ToLower(text)
	pos := 0
	for {
		start := strings.Index(lower[pos:], "<tool_call>")
		if start < 0 {
			break
		}
		start += pos
		bodyStart := start + len("<tool_call>")
		endRel := strings.Index(lower[bodyStart:], "</tool_call>")
		if endRel < 0 {
			full := text[start:]
			blocks = append(blocks, toolBlock{full: full, body: strings.TrimSpace(text[bodyStart:]), recovered: true})
			break
		}
		end := bodyStart + endRel
		fullEnd := end + len("</tool_call>")
		blocks = append(blocks, toolBlock{full: text[start:fullEnd], body: strings.TrimSpace(text[bodyStart:end])})
		pos = fullEnd
	}
	return blocks
}

func parseQwenBlock(body string, index int) (domain.ToolCall, bool, error) {
	if strings.HasPrefix(strings.TrimSpace(body), "{") {
		calls, err := decodeToolCandidate(body)
		if err != nil {
			if repaired, ok := repairJSON(body); ok {
				if calls, err = decodeToolCandidate(repaired); err == nil && len(calls) == 1 {
					return calls[0], true, nil
				}
			}
			return domain.ToolCall{}, false, err
		}
		if len(calls) != 1 {
			return domain.ToolCall{}, false, fmt.Errorf("expected one tool call in qwen block, got %d", len(calls))
		}
		return calls[0], false, nil
	}

	fnRe := regexp.MustCompile(`(?s)<function=([A-Za-z_][A-Za-z0-9_-]*)>\s*(.*?)\s*</function>`)
	match := fnRe.FindStringSubmatch(body)
	recovered := false
	if len(match) < 3 {
		openRe := regexp.MustCompile(`(?s)<function=([A-Za-z_][A-Za-z0-9_-]*)>\s*(.*)`)
		match = openRe.FindStringSubmatch(body)
		recovered = len(match) >= 3
	}
	if len(match) >= 3 {
		args := map[string]interface{}{}
		paramRe := regexp.MustCompile(`(?s)<parameter=([A-Za-z_][A-Za-z0-9_-]*)>\s*(.*?)\s*</parameter>`)
		for _, p := range paramRe.FindAllStringSubmatch(match[2], -1) {
			args[p[1]] = parseParameterValue(strings.TrimSpace(p[2]))
		}
		b, _ := json.Marshal(args)
		return domain.ToolCall{
			ID:   fmt.Sprintf("call_%d", index),
			Type: FunctionToolType,
			Function: domain.ToolCallFunction{
				Name:      match[1],
				Arguments: string(b),
			},
		}, recovered, nil
	}

	nameRe := regexp.MustCompile(`(?s)<name>\s*([^<]+?)\s*</name>`)
	argsRe := regexp.MustCompile(`(?s)<arguments>\s*(\{.*\})\s*</arguments>`)
	name := nameRe.FindStringSubmatch(body)
	args := argsRe.FindStringSubmatch(body)
	if len(name) >= 2 && len(args) >= 2 {
		normalized, err := normalizeArguments(json.RawMessage(args[1]))
		if err != nil {
			return domain.ToolCall{}, false, err
		}
		return domain.ToolCall{
			ID:   fmt.Sprintf("call_%d", index),
			Type: FunctionToolType,
			Function: domain.ToolCallFunction{
				Name:      strings.TrimSpace(name[1]),
				Arguments: normalized,
			},
		}, false, nil
	}
	return domain.ToolCall{}, false, fmt.Errorf("unrecognized qwen tool block")
}

func parseParameterValue(raw string) interface{} {
	var parsed interface{}
	if err := json.Unmarshal([]byte(raw), &parsed); err == nil {
		return parsed
	}
	return raw
}
