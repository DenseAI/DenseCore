package cmd

import (
	"os"

	"github.com/DenseAI/DenseCore/server/internal/server"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the API server",
	Long: `Start the DenseCore HTTP API server.
Direct CLI runs bind to loopback by default; use --host 0.0.0.0 or HOST=0.0.0.0 for remote exposure.
Authentication remains disabled unless --auth or AUTH_ENABLED=true is set.

Examples:
  densecore serve --model ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
  densecore serve --host 0.0.0.0 --auth --model ./model.gguf
  densecore serve --port 9090 --model /path/to/model.gguf
  densecore serve --auth --model ./model.gguf`,
	RunE: func(cmd *cobra.Command, args []string) error {
		opts, err := buildServeOptions(cmd)
		if err != nil {
			return err
		}
		return server.Run(opts)
	},
}

func init() {
	serveCmd.Flags().StringP("model", "m", "", "Path to GGUF model file")
	serveCmd.Flags().IntP("threads", "t", 0, "Number of inference threads (0 = auto-detect)")
	serveCmd.Flags().Bool("auth", false, "Enable API Key authentication")
	serveCmd.Flags().Bool("grpc", false, "Enable gRPC server")
	serveCmd.Flags().Int("grpc-port", 50051, "gRPC server port")

	cobra.CheckErr(viper.BindPFlag("model", serveCmd.Flags().Lookup("model")))
	cobra.CheckErr(viper.BindPFlag("threads", serveCmd.Flags().Lookup("threads")))

	rootCmd.AddCommand(serveCmd)
}

func buildServeOptions(cmd *cobra.Command) (*server.Options, error) {
	auth, err := cmd.Flags().GetBool("auth")
	if err != nil {
		return nil, err
	}
	verbose, err := cmd.Flags().GetBool("verbose")
	if err != nil {
		return nil, err
	}
	modelPath, err := cmd.Flags().GetString("model")
	if err != nil {
		return nil, err
	}
	if !cmd.Flags().Changed("model") && modelPath == "" {
		modelPath = os.Getenv("MODEL")
	}
	threads, err := cmd.Flags().GetInt("threads")
	if err != nil {
		return nil, err
	}

	var host string
	if cmd.Flags().Changed("host") {
		host, err = cmd.Flags().GetString("host")
		if err != nil {
			return nil, err
		}
	} else if os.Getenv("HOST") == "" {
		host = defaultCLIHost
	}

	port := 0
	if cmd.Flags().Changed("port") {
		port, err = cmd.Flags().GetInt("port")
		if err != nil {
			return nil, err
		}
	}

	var grpcEnabled *bool
	if cmd.Flags().Changed("grpc") {
		enabled, err := cmd.Flags().GetBool("grpc")
		if err != nil {
			return nil, err
		}
		grpcEnabled = &enabled
	}

	grpcPort := 0
	if cmd.Flags().Changed("grpc-port") {
		grpcPort, err = cmd.Flags().GetInt("grpc-port")
		if err != nil {
			return nil, err
		}
	}

	return &server.Options{
		Host:        host,
		Port:        port,
		ModelPath:   modelPath,
		Threads:     threads,
		Verbose:     verbose,
		LogOutput:   os.Stdout,
		ShowBanner:  true,
		Background:  false,
		AuthEnabled: auth,
		GRPCEnabled: grpcEnabled,
		GRPCPort:    grpcPort,
	}, nil
}
