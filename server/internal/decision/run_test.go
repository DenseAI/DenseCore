package decision

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

type listeningLog chan string

func (l listeningLog) Write(p []byte) (int, error) { l <- string(p); return len(p), nil }

func TestServeDrainsNativeBeforeClosingHTTPListener(t *testing.T) {
	b := &blockingBackend{fakeBackend: fixtureBackend(), entered: make(chan struct{}, 1), release: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var release sync.Once
	defer release.Do(func() { close(b.release) })
	log := make(listeningLog, 1)
	served := make(chan error, 1)
	go func() { served <- ServeWithOptions(ctx, b, "127.0.0.1:0", log, ServeOptions{QueueCapacity: 1}) }()
	var address string
	select {
	case line := <-log:
		fields := strings.Fields(line)
		address = "http://" + fields[len(fields)-1]
	case err := <-served:
		t.Fatalf("server startup: %v", err)
	case <-time.After(3 * time.Second):
		t.Fatal("server did not start")
	}
	client := &http.Client{Timeout: 3 * time.Second}
	request := make(chan error, 1)
	go func() {
		response, err := client.Post(address+"/v1/systemone", "application/json", strings.NewReader(decisionPayload))
		if err == nil {
			defer response.Body.Close()
			_, _ = io.Copy(io.Discard, response.Body)
			if response.StatusCode != 200 {
				err = fmt.Errorf("decision status %d", response.StatusCode)
			}
		}
		request <- err
	}()
	select {
	case <-b.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("native inference did not start")
	}
	cancel()
	// During native drain health reports 503 while the listener is still open.
	// Starting HTTP Shutdown first would immediately close the listener instead.
	deadline := time.Now().Add(time.Second)
	for {
		response, err := client.Get(address + "/health")
		if err != nil {
			t.Fatalf("listener closed before native drain: %v", err)
		}
		status := response.StatusCode
		response.Body.Close()
		if status == 503 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("admission did not drain")
		}
		time.Sleep(time.Millisecond)
	}
	time.Sleep(30 * time.Millisecond)
	response, err := client.Get(address + "/health")
	if err != nil {
		t.Fatalf("network shutdown started before native drain: %v", err)
	}
	response.Body.Close()
	if response.StatusCode != 503 {
		t.Fatal(response.StatusCode)
	}
	select {
	case err := <-served:
		t.Fatalf("Serve returned while native work blocked: %v", err)
	default:
	}
	release.Do(func() { close(b.release) })
	select {
	case err := <-request:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("admitted request did not finish")
	}
	select {
	case err := <-served:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("server did not drain")
	}
}
