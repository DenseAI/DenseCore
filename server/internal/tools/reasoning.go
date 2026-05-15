package tools

import "strings"

type reasoningSplit struct {
	Content   string
	Reasoning string
}

func splitReasoningBlocks(text string) reasoningSplit {
	var reasoning strings.Builder
	var content strings.Builder
	rest := text
	for {
		start := strings.Index(strings.ToLower(rest), "<think>")
		if start < 0 {
			content.WriteString(rest)
			break
		}
		content.WriteString(rest[:start])
		afterOpen := rest[start+len("<think>"):]
		end := strings.Index(strings.ToLower(afterOpen), "</think>")
		if end < 0 {
			if reasoning.Len() > 0 {
				reasoning.WriteString("\n")
			}
			reasoning.WriteString(afterOpen)
			break
		}
		if reasoning.Len() > 0 {
			reasoning.WriteString("\n")
		}
		reasoning.WriteString(afterOpen[:end])
		rest = afterOpen[end+len("</think>"):]
	}
	return reasoningSplit{
		Content:   strings.TrimSpace(content.String()),
		Reasoning: strings.TrimSpace(reasoning.String()),
	}
}
