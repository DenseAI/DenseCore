package buildinfo

// Version is the server version string used across HTTP root/health/banner.
// It can be overridden at build time via -ldflags "-X github.com/DenseAI/DenseCore/server/internal/buildinfo.Version=<ver>".
var Version = "0.1.0"
