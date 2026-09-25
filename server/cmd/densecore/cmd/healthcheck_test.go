package cmd

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"
)

func TestCheckHealthAcceptsSuccessfulResponse(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path != "/health/live" {
			t.Fatalf("path = %q, want /health/live", request.URL.Path)
		}
		return healthResponse(http.StatusOK), nil
	}), Timeout: time.Second}

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())
	if err := checkHealth(cmd, client, "http://densecore.test/health/live"); err != nil {
		t.Fatalf("checkHealth() error = %v", err)
	}
}

func TestCheckHealthRejectsUnhealthyResponse(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return healthResponse(http.StatusServiceUnavailable), nil
	}), Timeout: time.Second}

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())
	if err := checkHealth(cmd, client, "http://densecore.test/health/live"); err == nil {
		t.Fatal("checkHealth() error = nil, want non-2xx failure")
	}
}

func TestCheckHealthReturnsTransportError(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return nil, errors.New("dial failed")
	}), Timeout: time.Second}

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())
	if err := checkHealth(cmd, client, "http://densecore.test/health/live"); err == nil {
		t.Fatal("checkHealth() error = nil, want transport failure")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func healthResponse(status int) *http.Response {
	return &http.Response{
		StatusCode: status,
		Status:     http.StatusText(status),
		Body:       io.NopCloser(strings.NewReader("")),
		Header:     make(http.Header),
	}
}
