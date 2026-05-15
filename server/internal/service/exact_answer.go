package service

import (
	"log/slog"
	"os"
	"regexp"
	"sort"
	"strings"

	"descore-server/internal/domain"
)

var (
	exactAnswerEnglishRe = regexp.MustCompile(`(?i)\banswer\s+with\s+only\s+(?:"([^"]+)"|'([^']+)'|([^.?!\n\r]+))`)
	exactAnswerKoreanRe  = regexp.MustCompile(`([가-힣A-Za-z0-9_-]+)만\s+답해`)
)

type exactAnswerConstraint struct {
	allowedTokenIDs []int
	strict          bool
	maxTokens       int
	text            string
}

func deriveExactAnswerConstraint(engine domain.Engine, req domain.ChatCompletionRequest) *exactAnswerConstraint {
	if engine == nil || len(req.AllowedTokenIDs) > 0 {
		return nil
	}

	answer := extractExpectedExactAnswer(req)
	debug := os.Getenv("DENSECORE_DEBUG_EXACT_QA") != ""
	if debug {
		slog.Info("exact-answer extraction", slog.String("answer", answer))
	}
	if answer == "" {
		return nil
	}

	var tokenIDs []int
	addTokens := func(text string) {
		ids, err := engine.TokenizeText(text, false, false)
		if debug {
			slog.Info("exact-answer tokenization", slog.String("text", text), slog.Any("ids", ids), slog.Any("err", err))
		}
		if err != nil || len(ids) != 1 {
			return
		}
		tokenIDs = append(tokenIDs, ids[0])
	}

	addTokens(answer)
	addTokens(" " + answer)
	if strings.TrimSpace(answer) != answer {
		addTokens(strings.TrimSpace(answer))
	}

	if len(tokenIDs) == 0 {
		return &exactAnswerConstraint{
			text:      answer,
			strict:    true,
			maxTokens: max(1, strings.Count(answer, " ")+1),
		}
	}

	sort.Ints(tokenIDs)
	tokenIDs = slicesCompact(tokenIDs)
	return &exactAnswerConstraint{
		allowedTokenIDs: tokenIDs,
		strict:          true,
		maxTokens:       1,
		text:            answer,
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func extractExpectedExactAnswer(req domain.ChatCompletionRequest) string {
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

func slicesCompact(values []int) []int {
	if len(values) == 0 {
		return values
	}
	out := values[:1]
	for _, v := range values[1:] {
		if v != out[len(out)-1] {
			out = append(out, v)
		}
	}
	return out
}
