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
// stops admission and drains requests; Close on the native backend waits for any
// C inference still running after the HTTP drain deadline.
func Serve(ctx context.Context, b Backend, address string, log io.Writer) error {
	h := NewHandler(b)
	srv := &http.Server{Addr: address, Handler: h, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 32 << 10}
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return err
	}
	done := make(chan error, 1)
	go func() { done <- srv.Serve(listener) }()
	fmt.Fprintf(log, "Laya decision server listening on %s\n", listener.Addr())
	select {
	case err = <-done:
		if err == http.ErrServerClosed {
			return nil
		}
		return err
	case <-ctx.Done():
		h.Drain()
		shutdown, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err = srv.Shutdown(shutdown); err != nil {
			_ = srv.Close()
			return err
		}
		return nil
	}
}
