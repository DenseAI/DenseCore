package cmd

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultModelIsReadmePinned(t *testing.T) {
	artifact, err := resolveModelArtifact(defaultModelRepo, "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Filename != defaultModelFile || artifact.Revision != defaultModelRevision || artifact.SHA256 != defaultModelSHA256 {
		t.Fatalf("unexpected artifact: %+v", artifact)
	}
	if strings.Contains(artifact.downloadURL(), "/main/") {
		t.Fatal("mutable URL")
	}
}

func TestCustomModelsRequireImmutableIntegrity(t *testing.T) {
	cases := []struct{ repo, file, revision, hash string }{
		{"owner/repo", "model.gguf", "main", strings.Repeat("a", 64)},
		{"owner/repo", "model.gguf", strings.Repeat("b", 40), ""},
		{"../repo", "model.gguf", strings.Repeat("b", 40), strings.Repeat("a", 64)},
		{"owner/repo", "../model.gguf", strings.Repeat("b", 40), strings.Repeat("a", 64)},
	}
	for _, c := range cases {
		if _, err := resolveModelArtifact(c.repo, c.file, c.revision, c.hash); err == nil {
			t.Fatalf("accepted unsafe artifact: %+v", c)
		}
	}
}

func TestArtifactCacheSeparatesRevisionsAndChecksBytes(t *testing.T) {
	root := t.TempDir()
	payload := []byte("model")
	checksum := fmt.Sprintf("%x", sha256.Sum256(payload))
	first, err := resolveModelArtifact("owner/repo", "model.gguf", strings.Repeat("b", 40), checksum)
	if err != nil {
		t.Fatal(err)
	}
	second := first
	second.Revision = strings.Repeat("c", 40)
	if first.cachePath(root) == second.cachePath(root) {
		t.Fatal("revisions share cache")
	}
	path := first.cachePath(root)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, payload, 0644); err != nil {
		t.Fatal(err)
	}
	if err := verifyModelFile(path, checksum); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("truncated"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := verifyModelFile(path, checksum); err == nil {
		t.Fatal("corruption was accepted")
	}
}

func TestNestedArtifactURLRetainsRepositoryPath(t *testing.T) {
	artifact := modelArtifact{Repo: "owner/repo", Filename: "folder/model file.gguf", Revision: strings.Repeat("a", 40)}
	if !strings.HasSuffix(artifact.downloadURL(), "/folder/model%20file.gguf") {
		t.Fatal(artifact.downloadURL())
	}
}
