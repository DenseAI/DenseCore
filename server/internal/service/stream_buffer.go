package service

import (
	"os"
	"strconv"
	"strings"
)

const defaultStreamEventBuffer = 4096

func defaultStreamEventBufferSize() int {
	raw := strings.TrimSpace(os.Getenv("DENSECORE_STREAM_CHANNEL_BUFFER"))
	if raw == "" {
		return defaultStreamEventBuffer
	}
	size, err := strconv.Atoi(raw)
	if err != nil || size <= 0 {
		return defaultStreamEventBuffer
	}
	return size
}
