package grpc

import (
	"context"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestFixedModelServerRejectsGRPCModelLifecycle(t *testing.T) {
	handler := NewDenseCoreHandler(nil, nil)

	if _, err := handler.LoadModel(context.Background(), &LoadModelRequest{ModelPath: "/models/other.gguf"}); status.Code(err) != codes.Unimplemented {
		t.Fatalf("LoadModel status=%v want %v", status.Code(err), codes.Unimplemented)
	}
	if _, err := handler.UnloadModel(context.Background(), &UnloadModelRequest{}); status.Code(err) != codes.Unimplemented {
		t.Fatalf("UnloadModel status=%v want %v", status.Code(err), codes.Unimplemented)
	}
}
