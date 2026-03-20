package main

import (
	"fmt"
	"os"

	"descore-server/internal/server"
)

func main() {
	opts := &server.Options{
		ShowBanner: true,
		LogOutput:  os.Stdout,
	}

	if err := server.Run(opts); err != nil {
		fmt.Fprintf(os.Stderr, "server error: %v\n", err)
		os.Exit(1)
	}
}
