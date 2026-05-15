package tools

import (
	"crypto/sha256"
	"encoding/hex"
)

func ShortHashBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:12])
}
