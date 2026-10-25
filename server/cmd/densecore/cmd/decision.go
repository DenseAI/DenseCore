package cmd

import (
	"net"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"syscall"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/decision"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func init() {
	var model, precision string
	var threads int
	var batchSize, queueCapacity int
	var batchWait time.Duration
	command := &cobra.Command{Use: "decision", Short: "Run typed decisions from a GGUF model"}
	command.PersistentFlags().StringVar(&model, "model", "", "Path to a decision GGUF model (required)")
	command.PersistentFlags().IntVar(&threads, "threads", min(runtime.NumCPU(), 16), "Native CPU threads")
	command.PersistentFlags().StringVar(&precision, "precision", "fp32", "Projection precision: fp32 or experimental q8_0")
	_ = command.MarkPersistentFlagRequired("model")
	predict := &cobra.Command{Use: "predict", Args: cobra.NoArgs, Short: "Read a System One JSON request from stdin and write the result", RunE: func(cmd *cobra.Command, args []string) error {
		b, e := decision.OpenWithOptions(model, threads, decision.OpenOptions{Precision: precision})
		if e != nil {
			return e
		}
		defer b.Close()
		return decision.RunJSON(cmd.Context(), b, cmd.InOrStdin(), cmd.OutOrStdout())
	}}
	serve := &cobra.Command{Use: "serve", Args: cobra.NoArgs, Short: "Serve POST /v1/systemone, /health and /v1/models", RunE: func(cmd *cobra.Command, args []string) error {
		b, e := decision.OpenWithOptions(model, threads, decision.OpenOptions{Precision: precision})
		if e != nil {
			return e
		}
		defer b.Close()
		ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		return decision.ServeWithOptions(ctx, b, net.JoinHostPort(viper.GetString("host"), strconv.Itoa(viper.GetInt("port"))), cmd.ErrOrStderr(), decision.ServeOptions{BatchSize: batchSize, BatchWait: batchWait, QueueCapacity: queueCapacity})
	}}
	serve.Flags().IntVar(&batchSize, "batch-size", 1, "Maximum native batch size")
	serve.Flags().DurationVar(&batchWait, "batch-wait", 0, "Maximum batch collection wait")
	serve.Flags().IntVar(&queueCapacity, "queue-capacity", 0, "Maximum pending requests beyond batch capacity")
	command.AddCommand(predict, serve)
	rootCmd.AddCommand(command)
}
