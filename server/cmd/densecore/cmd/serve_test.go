package cmd

import (
	"testing"

	"github.com/spf13/cobra"
)

func TestBuildServeOptionsUsesLoopbackWhenHostUnset(t *testing.T) {
	t.Setenv("HOST", "")
	cmd := newServeCommandForTest()

	opts, err := buildServeOptions(cmd)
	if err != nil {
		t.Fatalf("buildServeOptions() error = %v", err)
	}
	if opts.Host != defaultCLIHost {
		t.Fatalf("Host = %q, want %q", opts.Host, defaultCLIHost)
	}
}

func TestBuildServeOptionsPreservesEnvConfiguredHost(t *testing.T) {
	t.Setenv("HOST", "0.0.0.0")
	cmd := newServeCommandForTest()

	opts, err := buildServeOptions(cmd)
	if err != nil {
		t.Fatalf("buildServeOptions() error = %v", err)
	}
	if opts.Host != "" {
		t.Fatalf("expected empty Host to preserve env-configured host, got %q", opts.Host)
	}
}

func TestBuildServeOptionsThreadsAndGRPCFlags(t *testing.T) {
	t.Setenv("HOST", "")
	cmd := newServeCommandForTest()
	if err := cmd.Flags().Set("threads", "16"); err != nil {
		t.Fatalf("set threads flag: %v", err)
	}
	if err := cmd.Flags().Set("grpc", "true"); err != nil {
		t.Fatalf("set grpc flag: %v", err)
	}

	opts, err := buildServeOptions(cmd)
	if err != nil {
		t.Fatalf("buildServeOptions() error = %v", err)
	}
	if opts.Threads != 16 {
		t.Fatalf("Threads = %d, want 16", opts.Threads)
	}
	if opts.GRPCEnabled == nil || !*opts.GRPCEnabled {
		t.Fatalf("expected explicit grpc flag to set GRPCEnabled=true, got %+v", opts.GRPCEnabled)
	}
}

func TestBuildServeOptionsPreservesModelEnvAlias(t *testing.T) {
	t.Setenv("HOST", "")
	t.Setenv("MODEL", "/models/model.gguf")
	cmd := newServeCommandForTest()

	opts, err := buildServeOptions(cmd)
	if err != nil {
		t.Fatalf("buildServeOptions() error = %v", err)
	}
	if opts.ModelPath != "/models/model.gguf" {
		t.Fatalf("ModelPath = %q, want MODEL alias", opts.ModelPath)
	}
}

func TestBuildServeOptionsLeavesGRPCToEnvWhenFlagUnchanged(t *testing.T) {
	t.Setenv("HOST", "")
	cmd := newServeCommandForTest()

	opts, err := buildServeOptions(cmd)
	if err != nil {
		t.Fatalf("buildServeOptions() error = %v", err)
	}
	if opts.GRPCEnabled != nil {
		t.Fatalf("expected nil GRPCEnabled when grpc flag is unchanged, got %+v", opts.GRPCEnabled)
	}
}

func newServeCommandForTest() *cobra.Command {
	cmd := &cobra.Command{Use: "serve"}
	cmd.Flags().String("host", defaultCLIHost, "Bind address")
	cmd.Flags().Int("port", 8080, "Port number")
	cmd.Flags().Bool("verbose", false, "Enable verbose output")
	cmd.Flags().String("model", "", "Path to GGUF model file")
	cmd.Flags().Int("threads", 0, "Number of inference threads")
	cmd.Flags().Bool("auth", false, "Enable API Key authentication")
	cmd.Flags().Bool("grpc", false, "Enable gRPC server")
	cmd.Flags().Int("grpc-port", 50051, "gRPC server port")
	return cmd
}
