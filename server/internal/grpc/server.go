// Package grpc provides gRPC server implementation for DenseCore.
// gRPC 서버는 저지연 바이너리 프로토콜을 제공하며, HTTP 서버와 병렬로 실행됩니다.
package grpc

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"log/slog"
	"net"
	"os"
	"sync"

	cloudtelemetry "github.com/DenseAI/DenseCloud/go/telemetry"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	"descore-server/internal/middleware"
	"descore-server/internal/service"
)

// Config holds gRPC server configuration
type Config struct {
	// Port is the gRPC server port (default: 50051)
	Port int

	// Enabled determines if gRPC server should start
	Enabled bool

	// MaxRecvMsgSize is the maximum message size in bytes (default: 4MB)
	MaxRecvMsgSize int

	// MaxSendMsgSize is the maximum send message size in bytes (default: 4MB)
	MaxSendMsgSize int

	// EnableReflection enables gRPC reflection for grpcurl support
	EnableReflection bool

	// AuthEnabled determines if API key authentication is required.
	AuthEnabled bool

	// APIKeyStore validates API keys when AuthEnabled is true.
	APIKeyStore middleware.APIKeyStore

	// TLSEnabled determines whether to enable TLS for gRPC transport.
	TLSEnabled bool

	// TLSCertFile is the server certificate file path.
	TLSCertFile string

	// TLSKeyFile is the server private key file path.
	TLSKeyFile string

	// TLSClientCAFile is the optional client CA bundle for mTLS.
	TLSClientCAFile string

	// TLSRequireClientCert enforces mTLS client certificate verification.
	TLSRequireClientCert bool

	// SharedMetrics records gRPC RED metrics into DenseCloud's shared `/metrics` endpoint.
	SharedMetrics *cloudtelemetry.GRPCMetrics
}

// DefaultConfig returns the default gRPC server configuration
func DefaultConfig() Config {
	return Config{
		Port:             50051,
		Enabled:          true,
		MaxRecvMsgSize:   4 * 1024 * 1024, // 4MB
		MaxSendMsgSize:   4 * 1024 * 1024, // 4MB
		EnableReflection: true,
		AuthEnabled:      false,
		TLSEnabled:       false,
	}
}

// Server wraps the gRPC server and service implementations
type Server struct {
	chatService  *service.ChatService
	modelService *service.ModelService
	grpcServer   *grpc.Server
	healthServer *health.Server
	config       Config
	mu           sync.Mutex
	started      bool
}

// NewServer creates a new gRPC server instance.
// gRPC 서버 인스턴스를 생성합니다. chatService와 modelService를 주입받아 사용합니다.
func NewServer(chat *service.ChatService, model *service.ModelService, cfg Config) *Server {
	return &Server{
		chatService:  chat,
		modelService: model,
		config:       cfg,
	}
}

// Start initializes and starts the gRPC server.
// gRPC 서버를 초기화하고 시작합니다. 블로킹 호출이므로 별도 고루틴에서 실행해야 합니다.
func (s *Server) Start() error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return fmt.Errorf("gRPC server already started")
	}

	if !s.config.Enabled {
		s.mu.Unlock()
		slog.Info("gRPC server disabled")
		return nil
	}
	if s.config.AuthEnabled && s.config.APIKeyStore == nil {
		s.mu.Unlock()
		return fmt.Errorf("gRPC authentication enabled but API key store is not configured")
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.config.Port))
	if err != nil {
		s.mu.Unlock()
		return fmt.Errorf("failed to listen on port %d: %w", s.config.Port, err)
	}

	// Server options with interceptors
	opts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(s.config.MaxRecvMsgSize),
		grpc.MaxSendMsgSize(s.config.MaxSendMsgSize),
		grpc.ChainUnaryInterceptor(
			RecoveryUnaryInterceptor(),
			TracingUnaryInterceptor(),
			LoggingUnaryInterceptor(),
			AuthUnaryInterceptor(s.config.AuthEnabled, s.config.APIKeyStore),
			MetricsUnaryInterceptor(s.config.SharedMetrics),
		),
		grpc.ChainStreamInterceptor(
			RecoveryStreamInterceptor(),
			TracingStreamInterceptor(),
			LoggingStreamInterceptor(),
			AuthStreamInterceptor(s.config.AuthEnabled, s.config.APIKeyStore),
			MetricsStreamInterceptor(s.config.SharedMetrics),
		),
	}

	if s.config.TLSEnabled {
		creds, err := buildServerTLSCredentials(s.config)
		if err != nil {
			s.mu.Unlock()
			return fmt.Errorf("failed to configure gRPC TLS: %w", err)
		}
		opts = append(opts, grpc.Creds(creds))
	} else {
		slog.Warn("gRPC TLS is disabled; traffic is unencrypted")
	}

	s.grpcServer = grpc.NewServer(opts...)

	// Register DenseCore service
	handler := NewDenseCoreHandler(s.chatService, s.modelService)
	RegisterDenseCoreServiceServer(s.grpcServer, handler)

	// Register health check service
	s.healthServer = health.NewServer()
	healthpb.RegisterHealthServer(s.grpcServer, s.healthServer)
	s.healthServer.SetServingStatus("densecore.DenseCoreService", healthpb.HealthCheckResponse_SERVING)

	// Enable reflection for grpcurl support
	if s.config.EnableReflection {
		reflection.Register(s.grpcServer)
	}

	s.started = true
	s.mu.Unlock()

	slog.Info("gRPC server starting",
		slog.Int("port", s.config.Port),
		slog.Bool("reflection", s.config.EnableReflection),
		slog.Bool("auth_enabled", s.config.AuthEnabled),
		slog.Bool("tls_enabled", s.config.TLSEnabled),
	)

	return s.grpcServer.Serve(lis)
}

// Stop gracefully shuts down the gRPC server.
// gRPC 서버를 안전하게 종료합니다. 진행 중인 요청이 완료될 때까지 대기합니다.
func (s *Server) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.started || s.grpcServer == nil {
		return
	}

	slog.Info("stopping gRPC server gracefully")

	// Set health status to NOT_SERVING before shutdown
	if s.healthServer != nil {
		s.healthServer.SetServingStatus("densecore.DenseCoreService", healthpb.HealthCheckResponse_NOT_SERVING)
	}

	s.grpcServer.GracefulStop()
	s.started = false

	slog.Info("gRPC server stopped")
}

// ForceStop immediately stops the gRPC server without waiting for active connections.
// 진행 중인 요청을 기다리지 않고 즉시 gRPC 서버를 중지합니다.
func (s *Server) ForceStop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.started || s.grpcServer == nil {
		return
	}

	slog.Warn("force stopping gRPC server")
	s.grpcServer.Stop()
	s.started = false
}

// SetServingStatus updates the health check serving status.
// 헬스 체크 상태를 업데이트합니다.
func (s *Server) SetServingStatus(serving bool) {
	if s.healthServer == nil {
		return
	}

	status := healthpb.HealthCheckResponse_NOT_SERVING
	if serving {
		status = healthpb.HealthCheckResponse_SERVING
	}
	s.healthServer.SetServingStatus("densecore.DenseCoreService", status)
}

// IsStarted returns whether the server has been started.
func (s *Server) IsStarted() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.started
}

func buildServerTLSCredentials(cfg Config) (credentials.TransportCredentials, error) {
	if cfg.TLSCertFile == "" || cfg.TLSKeyFile == "" {
		return nil, fmt.Errorf("GRPC_TLS_CERT_FILE and GRPC_TLS_KEY_FILE are required when TLS is enabled")
	}

	certificate, err := tls.LoadX509KeyPair(cfg.TLSCertFile, cfg.TLSKeyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load server cert/key: %w", err)
	}

	tlsCfg := &tls.Config{
		MinVersion:   tls.VersionTLS12,
		Certificates: []tls.Certificate{certificate},
	}

	if cfg.TLSRequireClientCert {
		if cfg.TLSClientCAFile == "" {
			return nil, fmt.Errorf("GRPC_TLS_CLIENT_CA_FILE is required when GRPC_TLS_REQUIRE_CLIENT_CERT=true")
		}

		caPem, err := os.ReadFile(cfg.TLSClientCAFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read client CA bundle: %w", err)
		}

		certPool := x509.NewCertPool()
		if ok := certPool.AppendCertsFromPEM(caPem); !ok {
			return nil, fmt.Errorf("failed to parse client CA bundle")
		}

		tlsCfg.ClientAuth = tls.RequireAndVerifyClientCert
		tlsCfg.ClientCAs = certPool
	}

	return credentials.NewTLS(tlsCfg), nil
}
