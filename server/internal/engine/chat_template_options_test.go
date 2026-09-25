package engine

import "testing"

func TestChatTemplateOptionValue(t *testing.T) {
	if got := chatTemplateOptionValue(nil); got != -1 {
		t.Fatalf("chatTemplateOptionValue(nil) = %d, want -1", got)
	}

	value := true
	if got := chatTemplateOptionValue(&value); got != 1 {
		t.Fatalf("chatTemplateOptionValue(true) = %d, want 1", got)
	}

	value = false
	if got := chatTemplateOptionValue(&value); got != 0 {
		t.Fatalf("chatTemplateOptionValue(false) = %d, want 0", got)
	}
}
