package cmd

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

const defaultCLIHost = "127.0.0.1"

var rootCmd = &cobra.Command{
	Use:               "densecore",
	Short:             "DenseCore: cloud-native inference runtime",
	PersistentPreRunE: configureSpeculationFlags,
	Long: `DenseCore is a high-performance cloud-native inference runtime.
It provides an OpenAI-compatible LLM API in the default profile and can run locally or in the cloud.

DenseCore uses runtime-qualified CPU kernels and explicit portable or target-specific build profiles.`,
}

func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().String("spec-type", "auto", "Speculation: auto (qualified profiles), none, mtp (explicit correctness reference)")
	rootCmd.PersistentFlags().Int("spec-draft-n-max", 2, "Maximum MTP draft tokens (1, 2 or 3)")
	rootCmd.PersistentFlags().BoolP("verbose", "v", false, "Enable verbose output")
	rootCmd.PersistentFlags().String("host", defaultCLIHost, "Bind address")
	rootCmd.PersistentFlags().Int("port", 8080, "Port number")

	cobra.CheckErr(viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose")))
	cobra.CheckErr(viper.BindPFlag("host", rootCmd.PersistentFlags().Lookup("host")))
	cobra.CheckErr(viper.BindPFlag("port", rootCmd.PersistentFlags().Lookup("port")))

	viper.AutomaticEnv() // Read from environment variables
}
