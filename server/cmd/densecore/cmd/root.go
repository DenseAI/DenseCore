package cmd

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var rootCmd = &cobra.Command{
	Use:   "densecore",
	Short: "DenseCore: cloud-native inference runtime",
	Long: `DenseCore is a high-performance cloud-native inference runtime.
It provides an OpenAI-compatible LLM API in the default profile and can run locally or in the cloud.

DenseCore uses specialized AVX-512 and AMX kernels to deliver GPU-like performance on modern CPUs.`,
}

func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().BoolP("verbose", "v", false, "Enable verbose output")
	rootCmd.PersistentFlags().String("host", "0.0.0.0", "Bind address")
	rootCmd.PersistentFlags().Int("port", 8080, "Port number")

	cobra.CheckErr(viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose")))
	cobra.CheckErr(viper.BindPFlag("host", rootCmd.PersistentFlags().Lookup("host")))
	cobra.CheckErr(viper.BindPFlag("port", rootCmd.PersistentFlags().Lookup("port")))

	viper.AutomaticEnv() // Read from environment variables
}
