package server

import (
	"context"
	"errors"
	"reflect"
	"testing"

	cloudserver "github.com/DenseCore/DenseCloud/go/server"
)

func TestRunStartupRollbackHooksRunsInReverseOrder(t *testing.T) {
	var calls []string
	hooks := []cloudserver.ShutdownHook{
		func(context.Context) error {
			calls = append(calls, "first")
			return nil
		},
		func(context.Context) error {
			calls = append(calls, "second")
			return nil
		},
		func(context.Context) error {
			calls = append(calls, "third")
			return nil
		},
	}

	runStartupRollbackHooks(hooks)

	want := []string{"third", "second", "first"}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("rollback call order = %v, want %v", calls, want)
	}
}

func TestRunStartupRollbackHooksContinuesAfterError(t *testing.T) {
	var calls []string
	hooks := []cloudserver.ShutdownHook{
		func(context.Context) error {
			calls = append(calls, "first")
			return nil
		},
		func(context.Context) error {
			calls = append(calls, "second")
			return errors.New("boom")
		},
		func(context.Context) error {
			calls = append(calls, "third")
			return nil
		},
	}

	runStartupRollbackHooks(hooks)

	want := []string{"third", "second", "first"}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("rollback calls = %v, want %v", calls, want)
	}
}
