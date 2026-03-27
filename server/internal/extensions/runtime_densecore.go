//go:build denseenterprise && densecore_cgo && cgo

package extensions

import (
	"context"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	entlicense "github.com/denseseries/dense-enterprise/enterprise/license"
	entmw "github.com/denseseries/dense-enterprise/enterprise/middleware"
	enttelemetry "github.com/denseseries/dense-enterprise/enterprise/telemetry"

	"descore-server/internal/util"
)

type denseEnterpriseRuntime struct {
	enabled        bool
	logger         *slog.Logger
	metricsPath    string
	authMiddleware []func(http.Handler) http.Handler
	apiMiddleware  []func(http.Handler) http.Handler

	licensePath    string
	licenseManager *entlicense.Manager
	validator      *entlicense.CgoValidator
	exporter       *enttelemetry.Exporter
	auditSink      *entmw.HashChainFileSink
	auditProcessor *entmw.AsyncAuditProcessor
}

func newRuntime(logger *slog.Logger) (Runtime, error) {
	if logger == nil {
		logger = slog.Default()
	}

	rt := &denseEnterpriseRuntime{
		enabled: util.ParseBoolEnv("DENSECORE_ENTERPRISE_ENABLED", false),
		logger:  logger,
	}
	if !rt.enabled {
		return rt, nil
	}

	oidcCfg, err := loadOIDCConfigFromEnv(logger)
	if err != nil {
		return nil, err
	}

	validator, err := entmw.NewCoreOSValidator(context.Background(), oidcCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize OIDC validator: %w", err)
	}
	oidcAuthMiddleware := entmw.OIDCAuth(oidcCfg, validator)
	rt.authMiddleware = append(rt.authMiddleware, oidcAuthMiddleware)
	rt.apiMiddleware = append(rt.apiMiddleware, oidcAuthMiddleware)

	auditPath, err := resolveAuditLogPath(strings.TrimSpace(os.Getenv("DENSECORE_ENT_AUDIT_LOG_PATH")))
	if err != nil {
		return nil, err
	}
	auditSink, err := entmw.NewHashChainFileSink(auditPath)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize audit sink: %w", err)
	}
	rt.auditSink = auditSink
	rt.auditProcessor = entmw.NewAsyncAuditProcessor(entmw.AsyncAuditProcessorConfig{
		Logger:             logger,
		Sink:               auditSink,
		EnablePIIRedaction: true,
		QueueSize:          parseIntEnv("DENSECORE_ENT_AUDIT_QUEUE_SIZE", 4096),
		Workers:            parseIntEnv("DENSECORE_ENT_AUDIT_WORKERS", 2),
	})
	rt.apiMiddleware = append(rt.apiMiddleware, entmw.AuditLog(entmw.AuditConfig{
		Logger:             logger,
		Sink:               auditSink,
		EnablePIIRedaction: true,
		LogRequestBody:     util.ParseBoolEnv("DENSECORE_ENT_AUDIT_LOG_BODY", false),
		MaxBodyLogSize:     parseIntEnv("DENSECORE_ENT_AUDIT_MAX_BODY_BYTES", 1024),
		AsyncProcessor:     rt.auditProcessor,
	}))

	pubkey, err := parseLicensePubKey()
	if err != nil {
		return nil, err
	}
	rt.validator = entlicense.NewCgoValidator()
	rt.licenseManager = entlicense.NewManager(pubkey, rt.validator, logger)
	rt.licensePath = strings.TrimSpace(os.Getenv("DENSECORE_ENT_LICENSE_PATH"))
	if rt.licensePath == "" {
		return nil, fmt.Errorf("enterprise enabled but DENSECORE_ENT_LICENSE_PATH is not set")
	}

	rt.exporter = enttelemetry.NewExporter(enttelemetry.NewCgoSnapshotProvider(), logger)
	rt.metricsPath = strings.TrimSpace(os.Getenv("DENSECORE_ENT_METRICS_PATH"))
	if rt.metricsPath == "" {
		rt.metricsPath = "/metrics/enterprise"
	}

	return rt, nil
}

func (r *denseEnterpriseRuntime) Enabled() bool {
	return r.enabled
}

func (r *denseEnterpriseRuntime) APIMiddleware() []func(http.Handler) http.Handler {
	if !r.enabled {
		return nil
	}
	return r.apiMiddleware
}

func (r *denseEnterpriseRuntime) RegisterRoutes(rootMux, _ *http.ServeMux) {
	if !r.enabled || rootMux == nil || r.exporter == nil {
		return
	}
	h := r.exporter.Handler()
	h = chainMiddleware(h, r.authMiddleware)
	rootMux.Handle(r.metricsPath, h)
}

func (r *denseEnterpriseRuntime) Startup(_ context.Context) error {
	if !r.enabled {
		return nil
	}
	if err := r.licenseManager.LoadAndActivate(r.licensePath); err != nil {
		return fmt.Errorf("license activation failed: %w", err)
	}
	r.logger.Info("enterprise runtime activated",
		slog.String("metrics_path", r.metricsPath),
		slog.String("license_path", r.licensePath))
	return nil
}

func (r *denseEnterpriseRuntime) Shutdown(ctx context.Context) error {
	if !r.enabled {
		return nil
	}
	if r.licenseManager != nil {
		r.licenseManager.Shutdown()
	}
	if r.validator != nil {
		if err := r.validator.InvalidateLicense(); err != nil {
			r.logger.Warn("failed to invalidate license on shutdown",
				slog.String("error", err.Error()))
		}
	}
	if r.auditProcessor != nil {
		if err := r.auditProcessor.Shutdown(ctx); err != nil {
			r.logger.Warn("failed to shutdown audit processor", slog.String("error", err.Error()))
		}
	}
	if r.auditSink != nil {
		if err := r.auditSink.Close(); err != nil {
			r.logger.Warn("failed to close audit sink", slog.String("error", err.Error()))
		}
	}
	return nil
}

func parseLicensePubKey() ([]byte, error) {
	hexValue := strings.TrimSpace(os.Getenv("DENSECORE_ENT_LICENSE_PUBKEY_HEX"))
	if hexValue != "" {
		b, err := hex.DecodeString(hexValue)
		if err != nil {
			return nil, fmt.Errorf("invalid DENSECORE_ENT_LICENSE_PUBKEY_HEX: %w", err)
		}
		if len(b) != 32 {
			return nil, fmt.Errorf("invalid pubkey length from hex: expected 32, got %d", len(b))
		}
		return b, nil
	}

	b64Value := strings.TrimSpace(os.Getenv("DENSECORE_ENT_LICENSE_PUBKEY_B64"))
	if b64Value == "" {
		return nil, fmt.Errorf("enterprise enabled but no license pubkey is configured (set DENSECORE_ENT_LICENSE_PUBKEY_HEX or DENSECORE_ENT_LICENSE_PUBKEY_B64)")
	}

	decoders := []*base64.Encoding{
		base64.StdEncoding,
		base64.RawStdEncoding,
		base64.URLEncoding,
		base64.RawURLEncoding,
	}
	for _, enc := range decoders {
		b, err := enc.DecodeString(b64Value)
		if err != nil {
			continue
		}
		if len(b) != 32 {
			return nil, fmt.Errorf("invalid pubkey length from base64: expected 32, got %d", len(b))
		}
		return b, nil
	}
	return nil, fmt.Errorf("invalid DENSECORE_ENT_LICENSE_PUBKEY_B64 format")
}

func parseRoleMapping(raw string) map[string]string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}

	// JSON map format is preferred.
	var asJSON map[string]string
	if err := json.Unmarshal([]byte(raw), &asJSON); err == nil {
		return asJSON
	}

	// Fallback: "group=role,group2=role2"
	out := map[string]string{}
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		parts := strings.SplitN(item, "=", 2)
		if len(parts) != 2 {
			continue
		}
		group := strings.TrimSpace(parts[0])
		role := strings.TrimSpace(parts[1])
		if group != "" && role != "" {
			out[group] = role
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func parseIntEnv(key string, defaultValue int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return defaultValue
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return defaultValue
	}
	return n
}

func chainMiddleware(handler http.Handler, middleware []func(http.Handler) http.Handler) http.Handler {
	if handler == nil {
		return nil
	}
	for i := len(middleware) - 1; i >= 0; i-- {
		if middleware[i] == nil {
			continue
		}
		handler = middleware[i](handler)
	}
	return handler
}

var oidcClientIDPattern = regexp.MustCompile(`^[A-Za-z0-9._:@/\-]{1,128}$`)

func loadOIDCConfigFromEnv(logger *slog.Logger) (entmw.OIDCConfig, error) {
	if logger == nil {
		logger = slog.Default()
	}

	cfg := entmw.OIDCConfig{
		IssuerURL:   strings.TrimSpace(os.Getenv("DENSECORE_ENT_OIDC_ISSUER")),
		ClientID:    strings.TrimSpace(os.Getenv("DENSECORE_ENT_OIDC_CLIENT_ID")),
		TenantClaim: strings.TrimSpace(os.Getenv("DENSECORE_ENT_OIDC_TENANT_CLAIM")),
		GroupsClaim: strings.TrimSpace(os.Getenv("DENSECORE_ENT_OIDC_GROUPS_CLAIM")),
		RoleMapping: parseRoleMapping(os.Getenv("DENSECORE_ENT_OIDC_ROLE_MAPPING")),
		Logger:      logger,
	}
	if cfg.IssuerURL == "" || cfg.ClientID == "" {
		return entmw.OIDCConfig{}, fmt.Errorf("enterprise enabled but OIDC config is incomplete (set DENSECORE_ENT_OIDC_ISSUER and DENSECORE_ENT_OIDC_CLIENT_ID)")
	}
	if err := validateOIDCIssuerURL(cfg.IssuerURL); err != nil {
		return entmw.OIDCConfig{}, fmt.Errorf("invalid DENSECORE_ENT_OIDC_ISSUER: %w", err)
	}
	if err := validateOIDCClientID(cfg.ClientID); err != nil {
		return entmw.OIDCConfig{}, fmt.Errorf("invalid DENSECORE_ENT_OIDC_CLIENT_ID: %w", err)
	}
	if err := validateOIDCClaimName(cfg.TenantClaim, "DENSECORE_ENT_OIDC_TENANT_CLAIM"); err != nil {
		return entmw.OIDCConfig{}, err
	}
	if err := validateOIDCClaimName(cfg.GroupsClaim, "DENSECORE_ENT_OIDC_GROUPS_CLAIM"); err != nil {
		return entmw.OIDCConfig{}, err
	}

	issuerAllowList := parseNormalizedIssuerSet(os.Getenv("DENSECORE_ENT_OIDC_ALLOWED_ISSUERS"))
	if len(issuerAllowList) > 0 && !issuerAllowList[normalizeIssuerURL(cfg.IssuerURL)] {
		return entmw.OIDCConfig{}, fmt.Errorf(
			"DENSECORE_ENT_OIDC_ISSUER is not in DENSECORE_ENT_OIDC_ALLOWED_ISSUERS")
	}
	clientAllowList := parseTrimmedSet(os.Getenv("DENSECORE_ENT_OIDC_ALLOWED_CLIENT_IDS"))
	if len(clientAllowList) > 0 && !clientAllowList[cfg.ClientID] {
		return entmw.OIDCConfig{}, fmt.Errorf(
			"DENSECORE_ENT_OIDC_CLIENT_ID is not in DENSECORE_ENT_OIDC_ALLOWED_CLIENT_IDS")
	}

	if len(issuerAllowList) == 0 {
		logger.Warn(
			"OIDC issuer allow-list is not configured; set DENSECORE_ENT_OIDC_ALLOWED_ISSUERS to harden identity provider trust")
	}
	if len(clientAllowList) == 0 {
		logger.Warn(
			"OIDC client ID allow-list is not configured; set DENSECORE_ENT_OIDC_ALLOWED_CLIENT_IDS to harden audience trust")
	}

	return cfg, nil
}

func validateOIDCIssuerURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("must be a valid URL: %w", err)
	}
	if u.Scheme != "https" {
		return fmt.Errorf("must use https scheme")
	}
	if u.Host == "" || u.Hostname() == "" {
		return fmt.Errorf("must include a host")
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return fmt.Errorf("must not include query parameters or fragments")
	}
	host := strings.ToLower(u.Hostname())
	if host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return fmt.Errorf("localhost issuers are not allowed")
	}
	if ip := net.ParseIP(host); ip != nil {
		if ip.IsLoopback() || ip.IsUnspecified() {
			return fmt.Errorf("loopback/unspecified issuer IPs are not allowed")
		}
	}
	return nil
}

func validateOIDCClientID(raw string) error {
	if !oidcClientIDPattern.MatchString(raw) {
		return fmt.Errorf("must match %s", oidcClientIDPattern.String())
	}
	return nil
}

func validateOIDCClaimName(raw string, envKey string) error {
	if raw == "" {
		return nil
	}
	if !oidcClientIDPattern.MatchString(raw) {
		return fmt.Errorf("invalid %s: must match %s", envKey, oidcClientIDPattern.String())
	}
	return nil
}

func parseTrimmedSet(raw string) map[string]bool {
	out := map[string]bool{}
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		out[part] = true
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func parseNormalizedIssuerSet(raw string) map[string]bool {
	out := map[string]bool{}
	for _, part := range strings.Split(raw, ",") {
		part = normalizeIssuerURL(part)
		if part == "" {
			continue
		}
		out[part] = true
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func normalizeIssuerURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}

	u, err := url.Parse(raw)
	if err == nil {
		u.Scheme = strings.ToLower(u.Scheme)
		u.Host = strings.ToLower(u.Host)
		u.RawQuery = ""
		u.Fragment = ""
		return strings.TrimSuffix(u.String(), "/")
	}

	return strings.TrimSuffix(raw, "/")
}

const (
	enterpriseAuditLogDir      = "logs/enterprise"
	defaultEnterpriseAuditPath = "audit.log"
)

func resolveAuditLogPath(raw string) (string, error) {
	normalized := strings.TrimSpace(raw)
	if normalized == "" {
		normalized = defaultEnterpriseAuditPath
	}
	normalized = strings.ReplaceAll(normalized, "\\", "/")
	normalized = filepath.Clean(filepath.FromSlash(normalized))
	if normalized == "." || normalized == ".." || filepath.IsAbs(normalized) {
		return "", fmt.Errorf(
			"invalid DENSECORE_ENT_AUDIT_LOG_PATH: use a relative path under %q", enterpriseAuditLogDir)
	}
	prefix := ".." + string(filepath.Separator)
	if strings.HasPrefix(normalized, prefix) {
		return "", fmt.Errorf(
			"invalid DENSECORE_ENT_AUDIT_LOG_PATH: path traversal is not allowed")
	}

	baseDir := filepath.Clean(enterpriseAuditLogDir)
	resolved := filepath.Clean(filepath.Join(baseDir, normalized))
	rel, err := filepath.Rel(baseDir, resolved)
	if err != nil {
		return "", fmt.Errorf("invalid DENSECORE_ENT_AUDIT_LOG_PATH: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, prefix) {
		return "", fmt.Errorf(
			"invalid DENSECORE_ENT_AUDIT_LOG_PATH: path traversal is not allowed")
	}

	if err := os.MkdirAll(filepath.Dir(resolved), 0o750); err != nil {
		return "", fmt.Errorf("failed to prepare audit log directory: %w", err)
	}
	return resolved, nil
}
