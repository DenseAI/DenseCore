package util

import "testing"

func TestParseBoolEnv(t *testing.T) {
	t.Setenv("TEST_BOOL", "true")
	if got := ParseBoolEnv("TEST_BOOL", false); !got {
		t.Fatal("expected true")
	}

	t.Setenv("TEST_BOOL", "OFF")
	if got := ParseBoolEnv("TEST_BOOL", true); got {
		t.Fatal("expected false")
	}

	t.Setenv("TEST_BOOL", "invalid")
	if got := ParseBoolEnv("TEST_BOOL", true); !got {
		t.Fatal("expected default true for invalid value")
	}

	t.Setenv("TEST_BOOL", "")
	if got := ParseBoolEnv("TEST_BOOL", false); got {
		t.Fatal("expected default false for empty value")
	}
}
