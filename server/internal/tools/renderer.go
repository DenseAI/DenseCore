package tools

import (
	"encoding/json"
	"fmt"
	"strings"

	"descore-server/internal/domain"
)

type ToolPromptRenderer interface {
	RenderTools(modelFamily string, tools []domain.Tool, toolChoice interface{}) (string, NormalizedTools, error)
}

type DefaultRenderer struct{}

func (DefaultRenderer) RenderTools(modelFamily string, rawTools []domain.Tool, rawChoice interface{}) (string, NormalizedTools, error) {
	normalized, err := NormalizeTools(rawTools, rawChoice)
	if err != nil {
		return "", normalized, err
	}
	if len(normalized.Tools) == 0 || normalized.Choice.Mode == ToolChoiceNone {
		return "", normalized, nil
	}
	family := ResolveParserFamily(ModelDescriptor{ModelID: modelFamily, ModelVariant: modelFamily})
	switch family {
	case ParserFamilyQwenXML:
		return renderQwenTools(normalized), normalized, nil
	default:
		return renderHermesTools(normalized), normalized, nil
	}
}

func InjectToolPrompt(messages []domain.Message, rendered string) []domain.Message {
	if strings.TrimSpace(rendered) == "" {
		return messages
	}
	out := append([]domain.Message(nil), messages...)
	for i := range out {
		role := strings.ToLower(strings.TrimSpace(out[i].Role))
		if role == "system" || role == "developer" {
			if out[i].Content != "" {
				out[i].Content += "\n\n" + rendered
			} else {
				out[i].Content = rendered
			}
			return out
		}
	}
	return append([]domain.Message{{Role: "system", Content: rendered}}, out...)
}

func renderHermesTools(normalized NormalizedTools) string {
	var sb strings.Builder
	sb.WriteString("You may call tools. When calling tools, respond with JSON only in this shape:\n")
	sb.WriteString(`{"tool_calls":[{"type":"function","function":{"name":"tool_name","arguments":{"arg":"value"}}}]}`)
	sb.WriteString("\nAvailable tools:\n")
	writeToolList(&sb, normalized.Tools)
	writeToolChoice(&sb, normalized.Choice)
	return sb.String()
}

func renderQwenTools(normalized NormalizedTools) string {
	var sb strings.Builder
	sb.WriteString("You may call tools. Available tools are listed as JSON schemas inside <tools>.\n<tools>\n")
	for _, tool := range normalized.Tools {
		b, _ := json.Marshal(tool)
		sb.Write(b)
		sb.WriteByte('\n')
	}
	sb.WriteString("</tools>\n")
	sb.WriteString("To call a tool, emit one or more blocks exactly like:\n")
	sb.WriteString("<tool_call>\n<function=tool_name>\n<parameter=arg_name>\nvalue\n</parameter>\n</function>\n</tool_call>\n")
	sb.WriteString("JSON-in-block form is also accepted: <tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>\n")
	writeToolChoice(&sb, normalized.Choice)
	return sb.String()
}

func writeToolList(sb *strings.Builder, tools []domain.Tool) {
	for _, tool := range tools {
		b, _ := json.Marshal(tool)
		sb.WriteString("- ")
		sb.Write(b)
		sb.WriteByte('\n')
	}
}

func writeToolChoice(sb *strings.Builder, choice ToolChoice) {
	switch choice.Mode {
	case ToolChoiceForced:
		fmt.Fprintf(sb, "You must call the function %q.\n", choice.ForcedName)
	case ToolChoiceRequired:
		sb.WriteString("You must call at least one tool.\n")
	case ToolChoiceNone:
		sb.WriteString("Do not call tools.\n")
	default:
		sb.WriteString("Call tools only when needed.\n")
	}
}
