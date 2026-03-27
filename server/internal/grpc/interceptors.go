package grpc

import (
	"context"
	"errors"
	"log/slog"
	"runtime/debug"
	"strings"
	"time"

	"descore-server/internal/middleware"

	cloudtelemetry "github.com/DenseAI/DenseCloud/go/telemetry"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	oteltrace "go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
)

// ============================================================================
// Recovery Interceptors
// ============================================================================

// RecoveryUnaryInterceptor handles panics in unary calls.
// 패닉 발생 시 복구하고 Internal 에러를 반환합니다.
func RecoveryUnaryInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("gRPC panic recovered",
					slog.Any("panic", r),
					slog.String("method", info.FullMethod),
					slog.String("stack", string(debug.Stack())),
				)
				err = status.Errorf(codes.Internal, "internal server error")
			}
		}()
		return handler(ctx, req)
	}
}

// RecoveryStreamInterceptor handles panics in streaming calls.
// 스트리밍 호출에서 패닉 발생 시 복구합니다.
func RecoveryStreamInterceptor() grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) (err error) {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("gRPC stream panic recovered",
					slog.Any("panic", r),
					slog.String("method", info.FullMethod),
					slog.String("stack", string(debug.Stack())),
				)
				err = status.Errorf(codes.Internal, "internal server error")
			}
		}()
		return handler(srv, ss)
	}
}

// ============================================================================
// Authentication Interceptors
// ============================================================================

const denseCoreServiceMethodPrefix = "/densecore.DenseCoreService/"

// AuthUnaryInterceptor validates API keys from gRPC metadata when enabled.
func AuthUnaryInterceptor(authEnabled bool, store middleware.APIKeyStore) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		if !authEnabled || !isProtectedDenseCoreMethod(info.FullMethod) {
			return handler(ctx, req)
		}

		if store == nil {
			return nil, status.Error(codes.Internal, "authentication is enabled but gRPC key store is not configured")
		}

		apiKey, err := getAPIKeyFromMetadata(ctx)
		if err != nil {
			return nil, status.Error(codes.Unauthenticated, err.Error())
		}
		if !store.Validate(apiKey) {
			return nil, status.Error(codes.Unauthenticated, "invalid API key")
		}

		return handler(ctx, req)
	}
}

// AuthStreamInterceptor validates API keys from gRPC metadata for stream RPCs when enabled.
func AuthStreamInterceptor(authEnabled bool, store middleware.APIKeyStore) grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if !authEnabled || !isProtectedDenseCoreMethod(info.FullMethod) {
			return handler(srv, ss)
		}

		if store == nil {
			return status.Error(codes.Internal, "authentication is enabled but gRPC key store is not configured")
		}

		apiKey, err := getAPIKeyFromMetadata(ss.Context())
		if err != nil {
			return status.Error(codes.Unauthenticated, err.Error())
		}
		if !store.Validate(apiKey) {
			return status.Error(codes.Unauthenticated, "invalid API key")
		}

		return handler(srv, ss)
	}
}

// ============================================================================
// Tracing Interceptors
// ============================================================================

// TracingUnaryInterceptor propagates trace context and creates a span per RPC.
func TracingUnaryInterceptor() grpc.UnaryServerInterceptor {
	tracer := otel.Tracer("densecore-grpc")
	propagator := otel.GetTextMapPropagator()

	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		md, ok := metadata.FromIncomingContext(ctx)
		if !ok {
			md = metadata.New(nil)
		}

		ctx = propagator.Extract(ctx, metadataCarrier(md))
		ctx, span := tracer.Start(ctx, info.FullMethod)
		defer span.End()

		setUnaryTraceHeaders(ctx, span.SpanContext())

		resp, err := handler(ctx, req)
		recordSpanStatus(span, err)

		return resp, err
	}
}

// TracingStreamInterceptor propagates trace context and creates a span per stream RPC.
func TracingStreamInterceptor() grpc.StreamServerInterceptor {
	tracer := otel.Tracer("densecore-grpc")
	propagator := otel.GetTextMapPropagator()

	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		ctx := ss.Context()
		md, ok := metadata.FromIncomingContext(ctx)
		if !ok {
			md = metadata.New(nil)
		}

		ctx = propagator.Extract(ctx, metadataCarrier(md))
		ctx, span := tracer.Start(ctx, info.FullMethod)
		defer span.End()

		wrapped := &wrappedServerStream{
			ServerStream: ss,
			ctx:          ctx,
		}
		setStreamTraceHeaders(wrapped, span.SpanContext())

		err := handler(srv, wrapped)
		recordSpanStatus(span, err)
		return err
	}
}

// ============================================================================
// Logging Interceptors
// ============================================================================

// LoggingUnaryInterceptor logs unary RPC calls with timing and status.
// Unary RPC 호출을 소요 시간 및 상태와 함께 로깅합니다.
func LoggingUnaryInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		start := time.Now()

		// Extract peer info
		var peerAddr string
		if p, ok := peer.FromContext(ctx); ok {
			peerAddr = p.Addr.String()
		}

		// Extract request ID from metadata
		var requestID string
		if md, ok := metadata.FromIncomingContext(ctx); ok {
			if ids := md.Get("x-request-id"); len(ids) > 0 {
				requestID = ids[0]
			}
		}

		resp, err := handler(ctx, req)
		duration := time.Since(start)

		// Determine log level based on error
		level := slog.LevelInfo
		if err != nil {
			level = slog.LevelError
		}

		attrs := []slog.Attr{
			slog.String("method", info.FullMethod),
			slog.Duration("duration", duration),
			slog.String("peer", peerAddr),
		}
		if sc := oteltrace.SpanFromContext(ctx).SpanContext(); sc.HasTraceID() {
			attrs = append(attrs, slog.String("trace_id", sc.TraceID().String()))
			attrs = append(attrs, slog.String("span_id", sc.SpanID().String()))
		}

		if requestID != "" {
			attrs = append(attrs, slog.String("request_id", requestID))
		}

		if err != nil {
			st, _ := status.FromError(err)
			attrs = append(attrs, slog.String("code", st.Code().String()))
			attrs = append(attrs, slog.String("error", st.Message()))
		} else {
			attrs = append(attrs, slog.String("code", codes.OK.String()))
		}

		slog.LogAttrs(ctx, level, "gRPC request", attrs...)
		return resp, err
	}
}

// LoggingStreamInterceptor logs streaming RPC calls.
// 스트리밍 RPC 호출을 로깅합니다.
func LoggingStreamInterceptor() grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		start := time.Now()

		// Extract peer info
		var peerAddr string
		if p, ok := peer.FromContext(ss.Context()); ok {
			peerAddr = p.Addr.String()
		}

		err := handler(srv, ss)
		duration := time.Since(start)

		level := slog.LevelInfo
		if err != nil {
			level = slog.LevelError
		}

		attrs := []slog.Attr{
			slog.String("method", info.FullMethod),
			slog.Duration("duration", duration),
			slog.String("peer", peerAddr),
			slog.Bool("client_stream", info.IsClientStream),
			slog.Bool("server_stream", info.IsServerStream),
		}
		if sc := oteltrace.SpanFromContext(ss.Context()).SpanContext(); sc.HasTraceID() {
			attrs = append(attrs, slog.String("trace_id", sc.TraceID().String()))
			attrs = append(attrs, slog.String("span_id", sc.SpanID().String()))
		}

		if err != nil {
			st, _ := status.FromError(err)
			attrs = append(attrs, slog.String("code", st.Code().String()))
			attrs = append(attrs, slog.String("error", st.Message()))
		} else {
			attrs = append(attrs, slog.String("code", codes.OK.String()))
		}

		slog.LogAttrs(ss.Context(), level, "gRPC stream", attrs...)
		return err
	}
}

// ============================================================================
// Metrics Interceptors
// ============================================================================

// MetricsUnaryInterceptor tracks metrics for unary calls.
// Unary 호출에 대한 메트릭을 추적합니다. (placeholder for Prometheus integration)
func MetricsUnaryInterceptor(metrics *cloudtelemetry.GRPCMetrics) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		if metrics == nil {
			return handler(ctx, req)
		}
		start := time.Now()
		metrics.BeginRPC()
		resp, err := handler(ctx, req)
		metrics.ObserveRPC(info.FullMethod, "unary", status.Code(err).String(), time.Since(start).Seconds())
		return resp, err
	}
}

// MetricsStreamInterceptor tracks metrics for streaming calls.
// 스트리밍 호출에 대한 메트릭을 추적합니다. (placeholder for Prometheus integration)
func MetricsStreamInterceptor(metrics *cloudtelemetry.GRPCMetrics) grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if metrics == nil {
			return handler(srv, ss)
		}
		start := time.Now()
		metrics.BeginRPC()
		err := handler(srv, ss)
		metrics.ObserveRPC(info.FullMethod, "stream", status.Code(err).String(), time.Since(start).Seconds())
		return err
	}
}

// ============================================================================
// Context Helpers
// ============================================================================

// contextKey is a type for context keys to avoid collisions
type contextKey string

const (
	// RequestIDKey is the context key for request ID
	RequestIDKey contextKey = "request_id"
)

// GetRequestIDFromContext extracts request ID from context.
// 컨텍스트에서 요청 ID를 추출합니다.
func GetRequestIDFromContext(ctx context.Context) string {
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if ids := md.Get("x-request-id"); len(ids) > 0 {
			return ids[0]
		}
	}
	return ""
}

type metadataCarrier metadata.MD

func (mc metadataCarrier) Get(key string) string {
	values := metadata.MD(mc).Get(strings.ToLower(key))
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

func (mc metadataCarrier) Set(key, value string) {
	metadata.MD(mc).Set(strings.ToLower(key), value)
}

func (mc metadataCarrier) Keys() []string {
	out := make([]string, 0, len(mc))
	for k := range mc {
		out = append(out, k)
	}
	return out
}

type wrappedServerStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (w *wrappedServerStream) Context() context.Context {
	return w.ctx
}

func setUnaryTraceHeaders(ctx context.Context, sc oteltrace.SpanContext) {
	if !sc.HasTraceID() {
		return
	}
	_ = grpc.SetHeader(ctx, metadata.Pairs(
		"x-trace-id", sc.TraceID().String(),
		"x-span-id", sc.SpanID().String(),
	))
}

func setStreamTraceHeaders(ss grpc.ServerStream, sc oteltrace.SpanContext) {
	if !sc.HasTraceID() {
		return
	}
	_ = ss.SetHeader(metadata.Pairs(
		"x-trace-id", sc.TraceID().String(),
		"x-span-id", sc.SpanID().String(),
	))
}

func recordSpanStatus(span oteltrace.Span, err error) {
	if span == nil {
		return
	}

	if err != nil {
		st, _ := status.FromError(err)
		span.RecordError(err)
		span.SetAttributes(attribute.String("rpc.grpc.status_code", st.Code().String()))
		return
	}

	span.SetAttributes(attribute.String("rpc.grpc.status_code", codes.OK.String()))
}

func isProtectedDenseCoreMethod(fullMethod string) bool {
	return strings.HasPrefix(fullMethod, denseCoreServiceMethodPrefix)
}

func getAPIKeyFromMetadata(ctx context.Context) (string, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return "", errors.New("missing metadata")
	}

	values := md.Get("authorization")
	if len(values) == 0 {
		return "", errors.New("missing authorization metadata")
	}

	apiKey, err := middleware.ParseAuthorizationHeader(values[0])
	if err != nil {
		return "", err
	}
	return apiKey, nil
}
