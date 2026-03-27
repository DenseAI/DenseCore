//go:build denseenterprise && densecore_cgo && cgo

package extensions

import (
	"io"
	"log/slog"
	"path/filepath"
	"testing"
)

func TestResolveAuditLogPath_DefaultAndNested(t *testing.T) {
	t.Chdir(t.TempDir())

	got, err := resolveAuditLogPath("")
	if err != nil {
		t.Fatalf("resolveAuditLogPath default failed: %v", err)
	}
	want := filepath.Join(enterpriseAuditLogDir, defaultEnterpriseAuditPath)
	if got != want {
		t.Fatalf("unexpected default path: got %q want %q", got, want)
	}

	got, err = resolveAuditLogPath("tenant-a/audit.log")
	if err != nil {
		t.Fatalf("resolveAuditLogPath nested failed: %v", err)
	}
	want = filepath.Join(enterpriseAuditLogDir, "tenant-a", "audit.log")
	if got != want {
		t.Fatalf("unexpected nested path: got %q want %q", got, want)
	}
}

func TestResolveAuditLogPath_RejectsTraversalAndAbsolute(t *testing.T) {
	t.Chdir(t.TempDir())

	if _, err := resolveAuditLogPath("../audit.log"); err == nil {
		t.Fatal("expected traversal path to be rejected")
	}

	absPath := filepath.Join(t.TempDir(), "audit.log")
	if _, err := resolveAuditLogPath(absPath); err == nil {
		t.Fatal("expected absolute path to be rejected")
	}
}

func TestLoadOIDCConfigFromEnv_ValidationAndAllowList(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	t.Setenv("DENSECORE_ENT_OIDC_ISSUER", "https://idp.example.com/realms/main")
	t.Setenv("DENSECORE_ENT_OIDC_CLIENT_ID", "densecore-enterprise")
	t.Setenv("DENSECORE_ENT_OIDC_ALLOWED_ISSUERS", "https://idp.example.com/realms/main")
	t.Setenv("DENSECORE_ENT_OIDC_ALLOWED_CLIENT_IDS", "densecore-enterprise")

	if _, err := loadOIDCConfigFromEnv(logger); err != nil {
		t.Fatalf("expected valid OIDC config, got error: %v", err)
	}

	t.Setenv("DENSECORE_ENT_OIDC_ISSUER", "http://idp.example.com/realms/main")
	if _, err := loadOIDCConfigFromEnv(logger); err == nil {
		t.Fatal("expected insecure issuer scheme to be rejected")
	}
	t.Setenv("DENSECORE_ENT_OIDC_ISSUER", "https://idp.example.com/realms/main")

	t.Setenv("DENSECORE_ENT_OIDC_ALLOWED_ISSUERS", "https://different.example.com/realms/main")
	if _, err := loadOIDCConfigFromEnv(logger); err == nil {
		t.Fatal("expected issuer allow-list mismatch to be rejected")
	}
}
