package cmd

import (
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/spf13/cobra"
)

const defaultHealthcheckURL = "http://127.0.0.1:8080/health/live"

var healthcheckCmd = &cobra.Command{
	Use:          "healthcheck",
	Short:        "Check the local server liveness endpoint",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		url, err := cmd.Flags().GetString("url")
		if err != nil {
			return err
		}
		timeout, err := cmd.Flags().GetDuration("timeout")
		if err != nil {
			return err
		}

		client := &http.Client{Timeout: timeout}
		return checkHealth(cmd, client, url)
	},
}

func init() {
	healthcheckCmd.Flags().String("url", defaultHealthcheckURL, "Health endpoint URL")
	healthcheckCmd.Flags().Duration("timeout", 5*time.Second, "Request timeout")
	rootCmd.AddCommand(healthcheckCmd)
}

func checkHealth(cmd *cobra.Command, client *http.Client, url string) error {
	request, err := http.NewRequestWithContext(cmd.Context(), http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("create health request: %w", err)
	}

	response, err := client.Do(request)
	if err != nil {
		return fmt.Errorf("health request failed: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	_, _ = io.Copy(io.Discard, response.Body)

	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("health endpoint returned %s", response.Status)
	}
	return nil
}
