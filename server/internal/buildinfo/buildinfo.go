package buildinfo

// Version is the server version string used across HTTP root/health/banner.
// It can be overridden at build time via -ldflags "-X descore-server/internal/buildinfo.Version=<ver>".
var Version = "2.1.0"
