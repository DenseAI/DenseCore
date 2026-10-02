// densecore-decision is the GGML-only Laya executable. It does not link the
// autoregressive model registry or load a Python runtime.
package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"syscall"

	"github.com/DenseAI/DenseCore/server/internal/decision"
)

func run() error {
	if len(os.Args) < 2 {
		return fmt.Errorf("usage: densecore-decision predict|serve --model model.gguf [--threads N] [--host 127.0.0.1 --port 8080]")
	}
	mode := os.Args[1]
	if mode != "predict" && mode != "serve" {
		return fmt.Errorf("unknown command %q: use predict or serve", mode)
	}
	fs := flag.NewFlagSet(mode, flag.ContinueOnError)
	model := fs.String("model", "", "Laya GGUF path (required)")
	threads := fs.Int("threads", min(runtime.NumCPU(), 16), "Native CPU threads")
	host := fs.String("host", "127.0.0.1", "Bind address")
	port := fs.Int("port", 8080, "HTTP port")
	if err := fs.Parse(os.Args[2:]); err != nil {
		return err
	}
	if *model == "" || fs.NArg() != 0 {
		return fmt.Errorf("--model is required; positional arguments are not supported")
	}
	b, err := decision.Open(*model, *threads)
	if err != nil {
		return err
	}
	defer b.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if mode == "predict" {
		return decision.RunJSON(ctx, b, os.Stdin, os.Stdout)
	}
	return decision.Serve(ctx, b, net.JoinHostPort(*host, strconv.Itoa(*port)), os.Stderr)
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
