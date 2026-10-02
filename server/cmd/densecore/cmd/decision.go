package cmd

import (
	"net"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"syscall"

	"github.com/DenseAI/DenseCore/server/internal/decision"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func init() {
	var model string
	var threads int
	command := &cobra.Command{Use: "decision", Short: "Run Laya typed decisions from a GGUF model"}
	command.PersistentFlags().StringVar(&model, "model", "", "Path to a Laya GGUF model (required)")
	command.PersistentFlags().IntVar(&threads, "threads", min(runtime.NumCPU(), 16), "Native CPU threads")
	_ = command.MarkPersistentFlagRequired("model")
	predict := &cobra.Command{Use: "predict", Args: cobra.NoArgs, Short: "Read a System One JSON request from stdin and write the result", RunE: func(cmd *cobra.Command, args []string) error {
		b, e := decision.Open(model, threads)
		if e != nil {
			return e
		}
		defer b.Close()
		return decision.RunJSON(cmd.Context(), b, cmd.InOrStdin(), cmd.OutOrStdout())
	}}
	serve := &cobra.Command{Use: "serve", Args: cobra.NoArgs, Short: "Serve POST /v1/systemone, /health and /v1/models", RunE: func(cmd *cobra.Command, args []string) error {
		b, e := decision.Open(model, threads)
		if e != nil {
			return e
		}
		defer b.Close()
		ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		return decision.Serve(ctx, b, net.JoinHostPort(viper.GetString("host"), strconv.Itoa(viper.GetInt("port"))), cmd.ErrOrStderr())
	}}
	command.AddCommand(predict, serve)
	rootCmd.AddCommand(command)
}
