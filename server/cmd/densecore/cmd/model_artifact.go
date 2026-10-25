package cmd

import (
	"crypto/sha256"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const defaultModelRepo = "lmstudio-community/Qwen3.5-0.8B-GGUF"
const defaultModelFile = "Qwen3.5-0.8B-Q4_K_M.gguf"
const defaultModelRevision = "7925ccdc665d4efdb1034791e6b553e11128e6f8"
const defaultModelSHA256 = "f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec"

type modelArtifact struct{ Repo, Filename, Revision, SHA256 string }

func resolveModelArtifact(repo, filename, revision, checksum string) (modelArtifact, error) {
	if repo == defaultModelRepo && (filename == "" || filename == defaultModelFile) {
		filename = defaultModelFile
		if revision == "" {
			revision = defaultModelRevision
		}
		if checksum == "" {
			checksum = defaultModelSHA256
		}
	}
	cleanParts := func(value string) bool {
		if value == "" || strings.ContainsAny(value, "\\?#") || strings.HasPrefix(value, "/") {
			return false
		}
		for _, part := range strings.Split(value, "/") {
			if part == "" || part == "." || part == ".." {
				return false
			}
		}
		return true
	}
	if !cleanParts(repo) || !cleanParts(filename) {
		return modelArtifact{}, fmt.Errorf("model repository and filename must be relative paths without traversal")
	}
	if !regexp.MustCompile(`^[0-9a-fA-F]{40}$`).MatchString(revision) {
		return modelArtifact{}, fmt.Errorf("--revision must be a full immutable 40-character commit hash")
	}
	if !regexp.MustCompile(`^[0-9a-fA-F]{64}$`).MatchString(checksum) {
		return modelArtifact{}, fmt.Errorf("--sha256 must contain the expected 64-character SHA-256")
	}
	return modelArtifact{repo, filename, strings.ToLower(revision), strings.ToLower(checksum)}, nil
}

func (a modelArtifact) cachePath(root string) string {
	return filepath.Join(root, filepath.FromSlash(a.Repo), a.Revision, filepath.FromSlash(a.Filename))
}

func (a modelArtifact) downloadURL() string {
	parts := strings.Split(a.Filename, "/")
	for index := range parts {
		parts[index] = url.PathEscape(parts[index])
	}
	return "https://huggingface.co/" + a.Repo + "/resolve/" + a.Revision + "/" + strings.Join(parts, "/")
}

func verifyModelFile(path, expected string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return err
	}
	actual := fmt.Sprintf("%x", hash.Sum(nil))
	if actual != expected {
		return fmt.Errorf("model SHA-256 mismatch: expected %s, got %s", expected, actual)
	}
	return nil
}
