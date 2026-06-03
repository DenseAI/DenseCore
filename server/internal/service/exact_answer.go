package service

import (
	"regexp"
	"strings"

	"descore-server/internal/domain"
)

var (
	exactAnswerEnglishRe       = regexp.MustCompile(`(?i)\banswer\s+with\s+only\s+(?:"([^"]+)"|'([^']+)'|([^.?!\n\r]+))`)
	exactAnswerKoreanRe        = regexp.MustCompile(`([가-힣A-Za-z0-9_-]+)만\s+답해`)
	exactAnswerFinalResponseRe = regexp.MustCompile(`(?i)\bfinal\s+response\s*:\s*([A-Za-z0-9_-]+)`)
	exactAnswerVerificationRe  = regexp.MustCompile(`(?i)\bverification\s+key\s+is\s*:\s*([A-Za-z0-9_-]+)|\bverification\s+key\s+is\s+([A-Za-z0-9_-]+)`)
)

type exactAnswerConstraint struct {
	allowedTokenIDs []int
	strict          bool
	maxTokens       int
	text            string
}

func deriveExactAnswerConstraint(_ domain.Engine, _ domain.ChatCompletionRequest) *exactAnswerConstraint {
	return nil
}

func extractExpectedExactAnswer(req domain.ChatCompletionRequest) string {
	return ExtractExpectedExactAnswer(req)
}

func ExtractExpectedExactAnswer(req domain.ChatCompletionRequest) string {
	if len(req.Messages) > 0 {
		for i := len(req.Messages) - 1; i >= 0; i-- {
			text := strings.TrimSpace(req.Messages[i].FlattenedText())
			if text == "" {
				continue
			}
			if answer := extractExactAnswerFromText(text); answer != "" {
				return answer
			}
		}
	}
	return ""
}

func extractExactAnswerFromText(text string) string {
	if matches := exactAnswerFinalResponseRe.FindStringSubmatch(text); len(matches) == 2 {
		return strings.TrimSpace(matches[1])
	}
	if matches := exactAnswerVerificationRe.FindStringSubmatch(text); len(matches) == 3 {
		for _, candidate := range matches[1:] {
			if strings.TrimSpace(candidate) != "" {
				return strings.TrimSpace(candidate)
			}
		}
	}
	if matches := exactAnswerEnglishRe.FindStringSubmatch(text); len(matches) == 4 {
		answer := ""
		for _, candidate := range matches[1:] {
			if strings.TrimSpace(candidate) != "" {
				answer = strings.TrimSpace(candidate)
				break
			}
		}
		answer = strings.Trim(answer, `"' `)
		if answer == "" || isGenericExactAnswerReference(answer) {
			return ""
		}
		return answer
	}
	if matches := exactAnswerKoreanRe.FindStringSubmatch(text); len(matches) == 2 {
		return strings.TrimSpace(matches[1])
	}
	return ""
}

func isGenericExactAnswerReference(answer string) bool {
	normalized := strings.ToLower(strings.TrimSpace(answer))
	normalized = strings.Join(strings.Fields(normalized), " ")
	switch normalized {
	case "a", "an", "the", "it", "this", "that":
		return true
	case "answer", "token", "code", "value", "word", "number", "codename", "name", "id", "identifier", "key":
		return true
	case "the answer", "the token", "the code", "the value", "the word", "the number", "the codename", "the name", "the id",
		"the identifier", "the key":
		return true
	case "only the answer", "only the token", "only the code", "only the value", "only the word", "only the number",
		"only the codename", "only the name", "only the id", "only the identifier", "only the key":
		return true
	default:
		return false
	}
}
