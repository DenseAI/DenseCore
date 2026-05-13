package cmd

import (
	"os"

	"descore-server/internal/server"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the API server (Production Mode)",
	Long: `Start the DenseCore HTTP API server for production use.
This mode emits structured JSON logs and handles proper signal handling for Kubernetes.

Examples:
  densecore serve --model ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
  densecore serve --port 9090 --model /path/to/model.gguf
  densecore serve --auth --model ./model.gguf`,
	RunE: func(cmd *cobra.Command, args []string) error {
		auth, _ := cmd.Flags().GetBool("auth")

		grpcEnabled, _ := cmd.Flags().GetBool("grpc")

		grpcPort := 0
		if cmd.Flags().Changed("grpc-port") {
			grpcPort, _ = cmd.Flags().GetInt("grpc-port")
		}

		opts := &server.Options{
			Host:        viper.GetString("host"),
			Port:        viper.GetInt("port"),
			Verbose:     viper.GetBool("verbose"),
			ModelPath:   viper.GetString("model"),
			LogOutput:   os.Stdout,
			ShowBanner:  true,
			Background:  false,
			AuthEnabled: auth, // Pass directly, env var fallback handled in server.Start()
			GRPCEnabled: &grpcEnabled,
			GRPCPort:    grpcPort,
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
