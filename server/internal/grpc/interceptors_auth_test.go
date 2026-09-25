package grpc

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type testAPIKeyStore struct {
	valid map[string]bool
}

func (s *testAPIKeyStore) Validate(key string) bool {
	return s.valid[key]
}

func (s *testAPIKeyStore) GetTier(key string) string {
	return ""
}

func (s *testAPIKeyStore) GetUserID(key string) string {
	return ""
}

func TestAuthUnaryInterceptorRejectsMissingMetadata(t *testing.T) {
	store := &testAPIKeyStore{valid: map[string]bool{"valid_key_123": true}}
	interceptor := AuthUnaryInterceptor(true, store)

	_, err := interceptor(
		context.Background(),
		nil,
		&grpc.UnaryServerInfo{FullMethod: DenseCoreService_ListModels_FullMethodName},
		func(ctx context.Context, req interface{}) (interface{}, error) {
			return "ok", nil
		},
	)
	if err == nil {
		t.Fatal("expected unauthenticated error")
	}
	if status.Code(err) != codes.Unauthenticated {
		t.Fatalf("expected Unauthenticated, got %v", status.Code(err))
	}
}

func TestAuthUnaryInterceptorRejectsInvalidAPIKey(t *testing.T) {
	store := &testAPIKeyStore{valid: map[string]bool{"valid_key_123": true}}
	interceptor := AuthUnaryInterceptor(true, store)

	ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs("authorization", "Bearer invalid_key_999"))
	_, err := interceptor(
		ctx,
		nil,
		&grpc.UnaryServerInfo{FullMethod: DenseCoreService_ListModels_FullMethodName},
		func(ctx context.Context, req interface{}) (interface{}, error) {
			return "ok", nil
		},
	)
	if err == nil {
		t.Fatal("expected unauthenticated error")
	}
	if status.Code(err) != codes.Unauthenticated {
		t.Fatalf("expected Unauthenticated, got %v", status.Code(err))
	}
}

func TestAuthUnaryInterceptorAcceptsValidAPIKey(t *testing.T) {
	store := &testAPIKeyStore{valid: map[string]bool{"valid_key_123": true}}
	interceptor := AuthUnaryInterceptor(true, store)

	ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs("authorization", "Bearer valid_key_123"))
	resp, err := interceptor(
		ctx,
		nil,
		&grpc.UnaryServerInfo{FullMethod: DenseCoreService_ListModels_FullMethodName},
		func(ctx context.Context, req interface{}) (interface{}, error) {
			return "ok", nil
		},
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp != "ok" {
		t.Fatalf("expected response ok, got %v", resp)
	}
}

func TestAuthUnaryInterceptorBypassesNonDenseCoreMethods(t *testing.T) {
	store := &testAPIKeyStore{valid: map[string]bool{}}
	interceptor := AuthUnaryInterceptor(true, store)

	called := false
	resp, err := interceptor(
		context.Background(),
		nil,
		&grpc.UnaryServerInfo{FullMethod: "/grpc.health.v1.Health/Check"},
		func(ctx context.Context, req interface{}) (interface{}, error) {
			called = true
			return "ok", nil
		},
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !called || resp != "ok" {
		t.Fatalf("expected handler call and ok response, called=%v resp=%v", called, resp)
	}
}

func TestAuthStreamInterceptorRejectsMissingMetadata(t *testing.T) {
	store := &testAPIKeyStore{valid: map[string]bool{"valid_key_123": true}}
	interceptor := AuthStreamInterceptor(true, store)

	err := interceptor(
		nil,
		&wrappedServerStream{ctx: context.Background()},
		&grpc.StreamServerInfo{FullMethod: DenseCoreService_StreamChatCompletion_FullMethodName},
		func(srv interface{}, stream grpc.ServerStream) error {
			return nil
		},
	)
	if err == nil {
		t.Fatal("expected unauthenticated error")
	}
	if status.Code(err) != codes.Unauthenticated {
		t.Fatalf("expected Unauthenticated, got %v", status.Code(err))
	}
}

func TestAuthUnaryInterceptorDisabled(t *testing.T) {
	interceptor := AuthUnaryInterceptor(false, nil)

	resp, err := interceptor(
		context.Background(),
		nil,
		&grpc.UnaryServerInfo{FullMethod: DenseCoreService_ListModels_FullMethodName},
		func(ctx context.Context, req interface{}) (interface{}, error) {
			return "ok", errors.New("app error")
		},
	)
	if resp != "ok" {
		t.Fatalf("expected response ok, got %v", resp)
	}
	if err == nil || err.Error() != "app error" {
		t.Fatalf("expected passthrough app error, got %v", err)
	}
}
