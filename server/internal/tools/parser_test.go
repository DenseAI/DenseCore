package tools

import (
	"strings"
	"testing"

	"descore-server/internal/domain"
)

func testTools() NormalizedTools {
	n, err := NormalizeTools([]domain.Tool{{
		Type: "function",
		Function: domain.ToolFunction{
			Name:        "read_file",
			Description: "Read a file",
			Parameters: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{"path": map[string]interface{}{"type": "string"}},
				"required":   []interface{}{"path"},
			},
		},
	}, {
		Type: "function",
		Function: domain.ToolFunction{
			Name:       "write_file",
			Parameters: map[string]interface{}{"type": "object"},
		},
	}}, "auto")
	if err != nil {
		panic(err)
	}
	return n
}

func TestHermesJSONSingleToolCall(t *testing.T) {
	result, err := ParseOutput(`{"tool_calls":[{"type":"function","function":{"name":"read_file","arguments":{"path":"src/main.cpp"}}}]}`, ParserFamilyHermes, testTools())
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].Function.Name != "read_file" {
		t.Fatalf("unexpected calls: %#v", result.ToolCalls)
	}
	if result.ToolCalls[0].Function.Arguments != `{"path":"src/main.cpp"}` {
		t.Fatalf("unexpected args %q", result.ToolCalls[0].Function.Arguments)
	}
}

func TestHermesJSONMultipleNestedEscapedAndRecovered(t *testing.T) {
	text := `prefix {"tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a\"b","meta":{"line":1}}}},{"function":{"name":"write_file","arguments":{"path":"x","body":"y",},},},],} suffix`
	result, err := ParseOutput(text, ParserFamilyHermes, testTools())
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 2 {
		t.Fatalf("expected two calls, got %#v", result.ToolCalls)
	}
	if !result.Recovered {
		t.Fatalf("expected malformed JSON recovery")
	}
}

func TestQwenXMLFunctionBlockAndSplitStreaming(t *testing.T) {
	p := NewParser(ModelDescriptor{ModelID: "qwen3.6-35b-a3b"}, testTools())
	for _, chunk := range []string{"<tool_", "call>\n<fun", "ction=read_", "file>\n<parameter=path>\nsrc/main.cpp\n</parameter>\n</function>\n</tool_call>"} {
		if _, err := p.Feed(chunk); err != nil {
			t.Fatal(err)
		}
	}
	result, err := p.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].Function.Arguments != `{"path":"src/main.cpp"}` {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestQwenXMLMultipleAndMalformedClosingRecovery(t *testing.T) {
	text := `<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call><tool_call><function=write_file><parameter=path>b</parameter>`
	result, err := ParseOutput(text, ParserFamilyQwenXML, testTools())
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 2 || !result.Recovered {
		t.Fatalf("expected two recovered calls, got %#v", result)
	}
}

func TestReasoningBlockRemovedFromContentWithTool(t *testing.T) {
	result, err := ParseOutput(`<think>private plan</think>{"tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a"}}}]}`, ParserFamilyHermes, testTools())
	if err != nil {
		t.Fatal(err)
	}
	if result.ReasoningContent != "private plan" || result.Content != "" || len(result.ToolCalls) != 1 {
		t.Fatalf("unexpected reasoning parse: %#v", result)
	}
}

func TestUnknownToolAndInvalidArgumentsRejected(t *testing.T) {
	if _, err := ParseOutput(`{"tool_calls":[{"function":{"name":"missing","arguments":{}}}]}`, ParserFamilyHermes, testTools()); err == nil {
		t.Fatalf("expected unknown tool rejection")
	}
	if _, err := ParseOutput(`{"tool_calls":[{"function":{"name":"read_file","arguments":{"other":1}}}]}`, ParserFamilyHermes, testTools()); err == nil {
		t.Fatalf("expected required argument rejection")
	}
}

func TestToolChoiceRendering(t *testing.T) {
	renderer := DefaultRenderer{}
	prompt, _, err := renderer.RenderTools("qwen3.6", testTools().Tools, map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name": "read_file",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if prompt == "" || !strings.Contains(prompt, "read_file") || !strings.Contains(prompt, "must call") {
		t.Fatalf("forced tool prompt did not render correctly: %q", prompt)
	}
	prompt, _, err = renderer.RenderTools("qwen3.6", testTools().Tools, "none")
	if err != nil {
		t.Fatal(err)
	}
	if prompt != "" {
		t.Fatalf("tool_choice none should suppress tool prompt, got %q", prompt)
	}
}
