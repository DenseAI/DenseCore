package engine

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

// This exercises the real CGO boundary: the Go race detector cannot observe
// reuse of the C++ thread-local strings returned by the rendering API.
func TestRenderChatPromptConcurrentBorrowedBuffers(t *testing.T) {
	model := os.Getenv("DENSECORE_TEST_MODEL")
	if model == "" {
		t.Skip("set DENSECORE_TEST_MODEL to a local GGUF for the CGO lifetime regression")
	}
	t.Setenv("DENSECORE_MAX_SEQ_LEN", "2048")
	t.Setenv("DENSECORE_KV_TARGET_MB", "512")
	t.Setenv("DENSECORE_GRAPH_CTX_MIN_MB", "512")
	t.Setenv("DENSECORE_GRAPH_CTX_MAX_MB", "2048")
	engine, err := NewDenseEngine(model, "", 2)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	const workers = 16
	errors := make(chan error, workers)
	start := make(chan struct{})
	var wg sync.WaitGroup
	thinking := false
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			<-start
			for iteration := 0; iteration < 256; iteration++ {
				marker := fmt.Sprintf("unique_prompt_%d_%d", worker, iteration)
				content := marker + " " + strings.Repeat("rendering lifetime boundary ", 32*(1+iteration%8)) + marker
				result, err := engine.RenderChatPrompt([]domain.Message{{Role: "user", Content: content}}, &thinking, nil)
				if err != nil {
					errors <- err
					return
				}
				if result == nil || !strings.Contains(result.RenderedPrompt, content) {
					errors <- fmt.Errorf("rendered prompt lost or replaced content for %s", marker)
					return
				}
			}
		}(worker)
	}
	close(start)
	wg.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}
