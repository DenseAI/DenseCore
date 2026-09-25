package tools

import "strings"

type ParserFamily string

const (
	ParserFamilyHermes  ParserFamily = "hermes_json"
	ParserFamilyQwenXML ParserFamily = "qwen_xml"
	ParserFamilyGeneric ParserFamily = "generic_json"
)

type ModelDescriptor struct {
	ModelID       string
	TokenizerType string
	ChatTemplate  string
	ModelVariant  string
}

func ResolveParserFamily(desc ModelDescriptor) ParserFamily {
	joined := strings.ToLower(desc.ModelID + " " + desc.TokenizerType + " " + desc.ChatTemplate + " " + desc.ModelVariant)
	switch {
	case strings.Contains(joined, "qwen3.6"), strings.Contains(joined, "qwen36"),
		strings.Contains(joined, "qwen3.5"), strings.Contains(joined, "qwen35"),
		strings.Contains(joined, "qwen-coder"), strings.Contains(joined, "qwen_coder"):
		return ParserFamilyQwenXML
	case strings.Contains(joined, "hermes"), strings.Contains(joined, "openai"), strings.Contains(joined, "gemma4"),
		strings.Contains(joined, "gemma-4"), strings.Contains(joined, "deepseek"):
		return ParserFamilyHermes
	default:
		return ParserFamilyGeneric
	}
}

func NewParser(desc ModelDescriptor, normalized NormalizedTools) *Parser {
	return &Parser{
		family:     ResolveParserFamily(desc),
		normalized: normalized,
	}
}
