package decision

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"
)

func RunJSON(ctx context.Context, b Backend, in io.Reader, out io.Writer) error {
	raw, err := io.ReadAll(io.LimitReader(in, MaxRequestBytes+1))
	if err != nil {
		return err
	}
	if len(raw) > MaxRequestBytes {
		return fmt.Errorf("request exceeds 1 MiB")
	}
	req, err := Parse(raw)
	if err != nil {
		return err
	}
	result, err := Predict(ctx, b, req)
	if err != nil {
		return err
	}
	return json.NewEncoder(out).Encode(result)
}

// Serve owns the listener, while the caller owns the backend. On cancellation it
// stops admission and waits for admitted native work before starting the bounded
// HTTP connection shutdown. Returning guarantees the backend may be closed.
func Serve(ctx context.Context, b Backend, address string, log io.Writer) error {
	return ServeWithOptions(ctx, b, address, log, ServeOptions{})
}

func ServeWithOptions(ctx context.Context, b Backend, address string, log io.Writer, options ServeOptions) error {
	h, err := NewHandlerWithOptions(b, options)
	if err != nil {
		return err
	}
	defer func() {
		h.Drain()
		_ = h.Wait(context.Background())
	}()
	srv := &http.Server{Addr: address, Handler: h, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 32 << 10}
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return err
	}
	done := make(chan error, 1)
	go func() { done <- srv.Serve(listener) }()
	fmt.Fprintf(log, "%s decision server listening on %s\n", b.Config().ModelID(), listener.Addr())
	select {
	case err = <-done:
		if err == http.ErrServerClosed {
			return nil
		}
		return err
	case <-ctx.Done():
		h.Drain()
		// Native latency is deliberately unbounded. The HTTP network deadline
		// starts only after every admitted inference and queued job has drained.
		_ = h.Wait(context.Background())
		shutdown, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err = srv.Shutdown(shutdown); err != nil {
			_ = srv.Close()
			return err
		}
		return nil
	}
}
