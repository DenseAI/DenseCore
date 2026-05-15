package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"descore-server/internal/domain"
)

const FunctionToolType = "function"

var functionNameRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_-]{0,63}$`)

type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceForced   ToolChoiceMode = "forced"
)

type ToolChoice struct {
	Mode       ToolChoiceMode
	ForcedName string
}

type NormalizedTools struct {
	Tools      []domain.Tool
	ByName     map[string]domain.Tool
	Choice     ToolChoice
	SchemaHash string
}

func NormalizeToolChoice(raw interface{}) (ToolChoice, error) {
	if raw == nil {
		return ToolChoice{Mode: ToolChoiceAuto}, nil
	}
	switch v := raw.(type) {
	case string:
		mode := ToolChoiceMode(strings.ToLower(strings.TrimSpace(v)))
		switch mode {
		case "", ToolChoiceAuto:
			return ToolChoice{Mode: ToolChoiceAuto}, nil
		case ToolChoiceNone:
			return ToolChoice{Mode: ToolChoiceNone}, nil
		case ToolChoiceRequired:
			return ToolChoice{Mode: ToolChoiceRequired}, nil
		default:
			return ToolChoice{}, fmt.Errorf("unsupported tool_choice %q", v)
		}
	case map[string]interface{}:
		typ, _ := v["type"].(string)
		if typ != FunctionToolType {
			return ToolChoice{}, fmt.Errorf("unsupported forced tool_choice type %q", typ)
		}
		fn, ok := v["function"].(map[string]interface{})
		if !ok {
			return ToolChoice{}, errors.New("forced tool_choice requires function object")
		}
		name, _ := fn["name"].(string)
		if !functionNameRE.MatchString(name) {
			return ToolChoice{}, fmt.Errorf("invalid forced tool_choice function name %q", name)
		}
		return ToolChoice{Mode: ToolChoiceForced, ForcedName: name}, nil
	default:
		return ToolChoice{}, fmt.Errorf("unsupported tool_choice shape %T", raw)
	}
}

func NormalizeTools(tools []domain.Tool, rawChoice interface{}) (NormalizedTools, error) {
	choice, err := NormalizeToolChoice(rawChoice)
	if err != nil {
		return NormalizedTools{}, err
	}
	out := NormalizedTools{
		Tools:  make([]domain.Tool, 0, len(tools)),
		ByName: make(map[string]domain.Tool, len(tools)),
		Choice: choice,
	}
	for _, tool := range tools {
		if strings.ToLower(strings.TrimSpace(tool.Type)) != FunctionToolType {
			return NormalizedTools{}, fmt.Errorf("unsupported tool type %q", tool.Type)
		}
		tool.Type = FunctionToolType
		tool.Function.Name = strings.TrimSpace(tool.Function.Name)
		if !functionNameRE.MatchString(tool.Function.Name) {
			return NormalizedTools{}, fmt.Errorf("invalid tool function name %q", tool.Function.Name)
		}
		if _, exists := out.ByName[tool.Function.Name]; exists {
			return NormalizedTools{}, fmt.Errorf("duplicate tool function name %q", tool.Function.Name)
		}
		if tool.Function.Parameters != nil {
			typ, _ := tool.Function.Parameters["type"].(string)
			if typ != "" && strings.ToLower(typ) != "object" {
				return NormalizedTools{}, fmt.Errorf("tool %q parameters must be a JSON object schema", tool.Function.Name)
			}
		}
		out.Tools = append(out.Tools, tool)
		out.ByName[tool.Function.Name] = tool
	}
	if choice.Mode == ToolChoiceForced {
		if _, ok := out.ByName[choice.ForcedName]; !ok {
			return NormalizedTools{}, fmt.Errorf("forced tool_choice function %q is not declared", choice.ForcedName)
		}
	}
	out.SchemaHash = StableToolSchemaHash(out.Tools)
	return out, nil
}

func StableToolSchemaHash(tools []domain.Tool) string {
	if len(tools) == 0 {
		return ""
	}
	copyTools := append([]domain.Tool(nil), tools...)
	sort.Slice(copyTools, func(i, j int) bool {
		return copyTools[i].Function.Name < copyTools[j].Function.Name
	})
	b, _ := json.Marshal(copyTools)
	return ShortHashBytes(b)
}

func ValidateToolCall(call domain.ToolCall, normalized NormalizedTools) error {
	if len(normalized.ByName) == 0 {
		return errors.New("tool call emitted without declared tools")
	}
	if normalized.Choice.Mode == ToolChoiceForced && call.Function.Name != normalized.Choice.ForcedName {
		return fmt.Errorf("tool %q emitted while tool_choice forced %q", call.Function.Name, normalized.Choice.ForcedName)
	}
	declared, ok := normalized.ByName[call.Function.Name]
	if !ok {
		return fmt.Errorf("unknown tool name %q", call.Function.Name)
	}
	if call.Type != "" && call.Type != FunctionToolType {
		return fmt.Errorf("unsupported tool call type %q", call.Type)
	}
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return fmt.Errorf("invalid JSON arguments for tool %q: %w", call.Function.Name, err)
	}
	if declared.Function.Parameters != nil {
		if required, ok := declared.Function.Parameters["required"].([]interface{}); ok {
			for _, item := range required {
				name, _ := item.(string)
				if name == "" {
					continue
				}
				if _, exists := args[name]; !exists {
					return fmt.Errorf("tool %q missing required argument %q", call.Function.Name, name)
				}
			}
		}
	}
	return nil
}
