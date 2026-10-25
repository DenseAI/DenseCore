package harmony

import (
	"strings"
	"testing"
)

func TestEveryByteSplitPreservesChannels(t *testing.T) {
	text := "<|channel|>analysis<|message|>계산 중<|end|><|start|>assistant<|channel|>final<|message|>정답은 4.<|return|>ignored"
	for step := 1; step <= len(text); step++ {
		d := NewDecoder()
		var content, reasoning string
		for i := 0; i < len(text); i += step {
			end := i + step
			if end > len(text) {
				end = len(text)
			}
			part, err := d.Write(text[i:end])
			if err != nil {
				t.Fatalf("step%d:%v", step, err)
			}
			content += part.Content
			reasoning += part.Reasoning
		}
		last, err := d.Finish()
		if err != nil {
			t.Fatal(err)
		}
		content += last.Content
		reasoning += last.Reasoning
		if content != "정답은 4." || reasoning != "계산 중" {
			t.Fatalf("step%d content=%q reasoning=%q", step, content, reasoning)
		}
	}
}
func TestIncompleteAnalysisNeverBecomesFinal(t *testing.T) {
	got, err := Split("<|channel|>analysis<|message|>Paris")
	if err != nil || got.Content != "" || got.Reasoning != "Paris" {
		t.Fatalf("%+v %v", got, err)
	}
}
func TestLegacySpellings(t *testing.T) {
	got, err := Split("<|meta_sep|>analysis<|im_sep|>plan<|im_end|><|im_start|>assistant<|meta_sep|>final<|im_sep|>answer<|fim_suffix|>")
	if err != nil || got.Content != "answer" || got.Reasoning != "plan" {
		t.Fatalf("%+v %v", got, err)
	}
}
func TestUnsupportedRecipientsAndMalformedHeaders(t *testing.T) {
	for _, text := range []string{" to=python<|channel|>analysis<|message|>secret", "<|channel|>unknown<|message|>secret", "<|channel|>final", "<|channel|>commentary<|message|>secret<|call|>"} {
		if got, err := Split(text); err == nil || got.Content != "" {
			t.Fatalf("%q %+v %v", text, got, err)
		}
	}
}
func TestCommentaryDoesNotQualifyAsFinal(t *testing.T) {
	got, err := Split("<|channel|>commentary<|message|>Paris<|end|>")
	if err != nil || got.Content != "" {
		t.Fatalf("%+v %v", got, err)
	}
}
func TestHeaderBoundedAndInvalidUTF8Rejected(t *testing.T) {
	d := NewDecoder()
	s := make([]byte, 4097)
	for i := range s {
		s[i] = 'a'
	}
	if _, err := d.Write(string(s)); err == nil {
		t.Fatal("unbounded header")
	}
	if _, err := Split("<|message|>bad\xff"); err == nil {
		t.Fatal("invalid utf8 accepted")
	}
}

func TestRejectTurnEndInsideHeader(t *testing.T) {
	for _, text := range []string{"<|return|>", "<|channel|>final<|return|>", "<|channel|>final<|fim_suffix|>"} {
		if _, err := Split(text); err == nil {
			t.Fatalf("accepted incomplete header %q", text)
		}
	}
}
func TestInvalidUtf8FailsWithoutAccumulatingStream(t *testing.T) {
	decoder := NewDecoder()
	if _, err := decoder.Write("<|message|>"); err != nil {
		t.Fatal(err)
	}
	if _, err := decoder.Write(strings.Repeat("\xff", 5000)); err == nil {
		t.Fatal("accepted permanently invalid utf8")
	}
}
