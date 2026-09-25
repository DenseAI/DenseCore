package grpc

import (
	"context"
	"errors"
	"strings"

	"github.com/DenseAI/DenseCore/server/internal/middleware"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

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

func isProtectedDenseCoreMethod(fullMethod string) bool {
	return strings.HasPrefix(fullMethod, denseCoreServiceMethodPrefix)
}

type wrappedServerStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (w *wrappedServerStream) Context() context.Context {
	return w.ctx
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
