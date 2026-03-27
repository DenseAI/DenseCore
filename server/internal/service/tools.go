package service

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"

	"descore-server/internal/domain"
)

const toolCallTypeFunction = "function"

// ToolPromptFormatter formats tools into a prompt that the model can understand
type ToolPromptFormatter struct {
	// Model-specific format (e.g., "qwen", "llama", "generic")
	ModelType string
}

// NewToolPromptFormatter creates a new formatter
func NewToolPromptFormatter(modelType string) *ToolPromptFormatter {
	return &ToolPromptFormatter{ModelType: modelType}
}

// FormatToolsPrompt creates a system prompt section describing available tools
func (f *ToolPromptFormatter) FormatToolsPrompt(tools []domain.Tool) string {
	if len(tools) == 0 {
		return ""
	}

	switch f.ModelType {
	case "qwen":
		return f.formatQwenTools(tools)
	case "glm", "glm45", "glm47":
		return f.formatGLMTools(tools)
	case "llama":
		return f.formatLlamaTools(tools)
	default:
		return f.formatGenericTools(tools)
	}
}

// formatGenericTools creates a generic tool description
func (f *ToolPromptFormatter) formatGenericTools(tools []domain.Tool) string {
	var sb strings.Builder
	sb.WriteString("\n\n# Available Tools\n\n")
	sb.WriteString("You have access to the following tools. To use a tool, respond with a JSON object in the following format:\n")
	sb.WriteString("```json\n{\"tool_calls\": [{\"id\": \"call_<unique_id>\", \"type\": \"" + toolCallTypeFunction + "\", \"function\": {\"name\": \"<function_name>\", \"arguments\": \"<json_string>\"}}]}\n```\n\n")

	for _, tool := range tools {
		if tool.Type != toolCallTypeFunction {
			continue
		}
		sb.WriteString(fmt.Sprintf("## %s\n", tool.Function.Name))
		if tool.Function.Description != "" {
			sb.WriteString(fmt.Sprintf("%s\n\n", tool.Function.Description))
		}
		if tool.Function.Parameters != nil {
			params, err := json.MarshalIndent(tool.Function.Parameters, "", "  ")
			if err != nil {
				slog.Warn("failed to marshal tool parameters", slog.String("tool", tool.Function.Name), slog.String("error", err.Error()))
				continue
			}
			sb.WriteString("Parameters:\n```json\n")
			sb.WriteString(string(params))
			sb.WriteString("\n```\n\n")
		}
	}

	return sb.String()
}

// formatQwenTools creates Qwen-specific tool format
func (f *ToolPromptFormatter) formatQwenTools(tools []domain.Tool) string {
	var sb strings.Builder
	sb.WriteString("\n\n# Tools\n\n")
	sb.WriteString("You are a helpful assistant with access to the following tools:\n\n")

	for _, tool := range tools {
		if tool.Type != toolCallTypeFunction {
			continue
		}
		toolJSON, err := json.Marshal(tool)
		if err != nil {
			slog.Warn("failed to marshal tool", slog.String("tool", tool.Function.Name), slog.String("error", err.Error()))
			continue
		}
		sb.WriteString(string(toolJSON))
		sb.WriteString("\n")
	}

	sb.WriteString("\nTo call a tool, respond with:\n")
	sb.WriteString("<tool_call>\n{\"name\": \"function_name\", \"arguments\": {...}}\n</tool_call>\n")

	return sb.String()
}

// formatGLMTools creates a GLM-friendly XML-wrapped tool description.
func (f *ToolPromptFormatter) formatGLMTools(tools []domain.Tool) string {
	var sb strings.Builder
	sb.WriteString("\n\n# Tools\n\n")
	sb.WriteString("You may call tools by replying with:\n")
	sb.WriteString("<tool_call>\n{\"name\": \"function_name\", \"arguments\": {...}}\n</tool_call>\n\n")
	sb.WriteString("Available tools:\n<tools>\n")

	for _, tool := range tools {
		if tool.Type != toolCallTypeFunction {
			continue
		}
		toolJSON, err := json.Marshal(tool)
		if err != nil {
			slog.Warn("failed to marshal tool", slog.String("tool", tool.Function.Name), slog.String("error", err.Error()))
			continue
		}
		sb.WriteString(string(toolJSON))
		sb.WriteString("\n")
	}

	sb.WriteString("</tools>\n")
	sb.WriteString("If you include reasoning, wrap it in <think>...</think> and keep tool calls outside the reasoning block.\n")
	return sb.String()
}

// formatLlamaTools creates Llama-specific tool format
func (f *ToolPromptFormatter) formatLlamaTools(tools []domain.Tool) string {
	var sb strings.Builder
	sb.WriteString("\n\nYou have access to the following functions:\n\n")

	for _, tool := range tools {
		if tool.Type != toolCallTypeFunction {
			continue
		}
		sb.WriteString(fmt.Sprintf("Use the function '%s' to '%s':\n", tool.Function.Name, tool.Function.Description))
		params, err := json.Marshal(tool.Function.Parameters)
		if err != nil {
			slog.Warn("failed to marshal tool parameters", slog.String("tool", tool.Function.Name), slog.String("error", err.Error()))
			continue
		}
		sb.WriteString(string(params))
		sb.WriteString("\n\n")
	}

	sb.WriteString("If you choose to call a function, ONLY reply in the following format:\n")
	sb.WriteString("<function=function_name>{\"param\": \"value\"}</function>\n")

	return sb.String()
}

// ToolCallParser parses tool calls from model output
type ToolCallParser struct {
	ModelType string
}

// NewToolCallParser creates a new parser
func NewToolCallParser(modelType string) *ToolCallParser {
	return &ToolCallParser{ModelType: modelType}
}

// ParseToolCalls extracts tool calls from model output
func (p *ToolCallParser) ParseToolCalls(output string) ([]domain.ToolCall, string, error) {
	switch p.ModelType {
	case "qwen":
		return p.parseQwenToolCalls(output)
	case "glm", "glm45", "glm47":
		return p.parseGLMToolCalls(output)
	case "llama":
		return p.parseLlamaToolCalls(output)
	default:
		return p.parseGenericToolCalls(output)
	}
}

func stripReasoningBlocks(output string) string {
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`(?s)<think>.*?</think>`),
		regexp.MustCompile(`(?s)<reasoning>.*?</reasoning>`),
		regexp.MustCompile(`(?s)◁think▷.*?◁/think▷`),
	}

	cleaned := output
	for _, re := range patterns {
		cleaned = re.ReplaceAllString(cleaned, "")
	}
	return strings.TrimSpace(cleaned)
}

// parseGenericToolCalls parses JSON-based tool calls.
//
// Handles multiple output formats:
//  1. JSON in markdown code blocks (```json ... ```)
//  2. Raw JSON with tool_calls key
func (p *ToolCallParser) parseGenericToolCalls(output string) ([]domain.ToolCall, string, error) {
	type candidate struct {
		jsonStr   string
		fullMatch string
	}
	var candidates []candidate

	// Strategy 1: Extract JSON from markdown code blocks first
	// This handles: ```json\n{"tool_calls": [...]}\n```
	codeBlockRe := regexp.MustCompile("(?i)```(?:json)?\\s*\\n?([\\s\\S]*?)\\n?```")
	for _, match := range codeBlockRe.FindAllStringSubmatch(output, -1) {
		content := strings.TrimSpace(match[1])
		if strings.Contains(content, `"tool_calls"`) {
			candidates = append(candidates, candidate{jsonStr: content, fullMatch: match[0]})
		}
	}

	// Strategy 2: Direct JSON pattern (fallback)
	// Handles raw JSON without code blocks
	directRe := regexp.MustCompile(`\{[^{}]*"tool_calls"\s*:\s*\[[\s\S]*?\]\s*\}`)
	for _, match := range directRe.FindAllString(output, -1) {
		candidates = append(candidates, candidate{jsonStr: match, fullMatch: match})
	}

	// Try parsing each candidate
	for _, c := range candidates {
		var parsed struct {
			ToolCalls []domain.ToolCall `json:"tool_calls"`
		}
		if err := json.Unmarshal([]byte(c.jsonStr), &parsed); err != nil {
			continue // Try next candidate
		}

		if len(parsed.ToolCalls) == 0 {
			continue
		}

		// Remove the tool call JSON from output
		cleanOutput := strings.Replace(output, c.fullMatch, "", 1)
		cleanOutput = strings.TrimSpace(cleanOutput)

		// Generate IDs if missing
		for i := range parsed.ToolCalls {
			if parsed.ToolCalls[i].ID == "" {
				parsed.ToolCalls[i].ID = fmt.Sprintf("call_%d", i)
			}
			if parsed.ToolCalls[i].Type == "" {
				parsed.ToolCalls[i].Type = toolCallTypeFunction
			}
		}

		return parsed.ToolCalls, cleanOutput, nil
	}

	return nil, output, nil // No valid tool calls found
}

// parseQwenToolCalls parses Qwen-style <tool_call> format
func (p *ToolCallParser) parseQwenToolCalls(output string) ([]domain.ToolCall, string, error) {
	re := regexp.MustCompile(`<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>`)
	matches := re.FindAllStringSubmatch(output, -1)

	if len(matches) == 0 {
		return nil, output, nil
	}

	var toolCalls []domain.ToolCall
	for i, match := range matches {
		var parsed struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(match[1]), &parsed); err != nil {
			continue
		}

		toolCalls = append(toolCalls, domain.ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: toolCallTypeFunction,
			Function: domain.ToolCallFunction{
				Name:      parsed.Name,
				Arguments: string(parsed.Arguments),
			},
		})
	}

	// Remove tool calls from output
	cleanOutput := re.ReplaceAllString(output, "")
	cleanOutput = strings.TrimSpace(cleanOutput)

	return toolCalls, cleanOutput, nil
}

// parseGLMToolCalls parses GLM-style tool calls and strips reasoning tags first.
// Supported forms:
//  1. <tool_call>{"name":"fn","arguments":{...}}</tool_call>
//  2. <tool_call><name>fn</name><arguments>{...}</arguments></tool_call>
//  3. Generic JSON tool_calls fallback after reasoning removal
func (p *ToolCallParser) parseGLMToolCalls(output string) ([]domain.ToolCall, string, error) {
	cleaned := stripReasoningBlocks(output)

	// First, try the Qwen-compatible JSON-in-tool_call format.
	if toolCalls, cleanOutput, err := p.parseQwenToolCalls(cleaned); err != nil {
		return nil, output, err
	} else if len(toolCalls) > 0 {
		return toolCalls, stripReasoningBlocks(cleanOutput), nil
	}

	re := regexp.MustCompile(`(?s)<tool_call>\s*(.*?)\s*</tool_call>`)
	matches := re.FindAllStringSubmatch(cleaned, -1)
	if len(matches) == 0 {
		return p.parseGenericToolCalls(cleaned)
	}

	nameRe := regexp.MustCompile(`(?s)<name>\s*([^<]+?)\s*</name>`)
	argsRe := regexp.MustCompile(`(?s)<arguments>\s*(\{.*\})\s*</arguments>`)

	var toolCalls []domain.ToolCall
	for i, match := range matches {
		body := strings.TrimSpace(match[1])
		if strings.HasPrefix(body, "{") {
			var parsed struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}
			if err := json.Unmarshal([]byte(body), &parsed); err == nil && parsed.Name != "" {
				toolCalls = append(toolCalls, domain.ToolCall{
					ID:   fmt.Sprintf("call_%d", i),
					Type: toolCallTypeFunction,
					Function: domain.ToolCallFunction{
						Name:      parsed.Name,
						Arguments: string(parsed.Arguments),
					},
				})
			}
			continue
		}

		nameMatch := nameRe.FindStringSubmatch(body)
		argsMatch := argsRe.FindStringSubmatch(body)
		if len(nameMatch) < 2 || len(argsMatch) < 2 {
			continue
		}

		toolCalls = append(toolCalls, domain.ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: toolCallTypeFunction,
			Function: domain.ToolCallFunction{
				Name:      strings.TrimSpace(nameMatch[1]),
				Arguments: strings.TrimSpace(argsMatch[1]),
			},
		})
	}

	cleanOutput := re.ReplaceAllString(cleaned, "")
	cleanOutput = strings.TrimSpace(cleanOutput)
	if len(toolCalls) == 0 {
		return p.parseGenericToolCalls(cleanOutput)
	}

	return toolCalls, cleanOutput, nil
}

// parseLlamaToolCalls parses Llama-style <function=name>{...}</function> format
func (p *ToolCallParser) parseLlamaToolCalls(output string) ([]domain.ToolCall, string, error) {
	re := regexp.MustCompile(`<function=(\w+)>(\{[\s\S]*?\})</function>`)
	matches := re.FindAllStringSubmatch(output, -1)

	if len(matches) == 0 {
		return nil, output, nil
	}

	var toolCalls []domain.ToolCall
	for i, match := range matches {
		toolCalls = append(toolCalls, domain.ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: toolCallTypeFunction,
			Function: domain.ToolCallFunction{
				Name:      match[1],
				Arguments: match[2],
			},
		})
	}

	// Remove tool calls from output
	cleanOutput := re.ReplaceAllString(output, "")
	cleanOutput = strings.TrimSpace(cleanOutput)

	return toolCalls, cleanOutput, nil
}

// FormatToolResult formats a tool result message for the model
func FormatToolResult(toolCallID, functionName, result string) string {
	return fmt.Sprintf("Tool '%s' (call_id: %s) returned:\n%s", functionName, toolCallID, result)
}
