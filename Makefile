.PHONY: all clean lib server run test help format lint format-check docs proto grpc-deps ensure-golangci-lint

# Default target
all: lib server

# Directories
BUILD_DIR := build
CORE_DIR := core
SERVER_DIR := server
BIN_DIR := bin
PYTHON_DIR := python
CLANG_FORMAT ?= clang-format
RUFF ?= ruff
GO_BIN := $(or $(shell go env GOPATH 2>/dev/null),$(HOME)/go)/bin
GOLANGCI_LINT_VERSION ?= v2.1.6
GOLANGCI_LINT ?= $(GO_BIN)/golangci-lint
GOLANGCI_LINT_RUN_FLAGS ?= --timeout=5m
GOLANGCI_LINT_RUN_TARGETS ?= ./cmd/... ./internal/...
GO_BUILD_VALIDATE_TARGETS ?= .

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT := dylib
    LIB_PREFIX := lib
else ifeq ($(UNAME_S),Linux)
    LIB_EXT := so
    LIB_PREFIX := lib
else
    LIB_EXT := dll
    LIB_PREFIX :=
endif

LIB_NAME := $(LIB_PREFIX)densecore.$(LIB_EXT)

help:
	@echo "==========================================="
	@echo "DenseCore Build System"
	@echo "==========================================="
	@echo "Build targets:"
	@echo "  make all      - Build library and server (default)"
	@echo "  make lib      - Build C++ core library"
	@echo "  make server   - Build Go server"
	@echo "  make run      - Build and run the server"
	@echo "  make test     - Run a test request"
	@echo "  make clean    - Remove build artifacts"
	@echo ""
	@echo "Code quality:"
	@echo "  make format       - Auto-fix formatting (C++, Python, Go)"
	@echo "  make lint         - Run linters (no auto-fix)"
	@echo "  make format-check - Check formatting without fixing"
	@echo ""
	@echo "Environment variables:"
	@echo "  MAIN_MODEL_PATH  - Path to main model"
	@echo "  DRAFT_MODEL_PATH - Path to draft model (optional)"
	@echo "  PORT             - Server port (default: 8080)"
	@echo "==========================================="

ensure-golangci-lint:
	@command -v go >/dev/null 2>&1 || { echo "❌ Go not found. Please install Go 1.24+"; exit 1; }
	@if [ ! -x "$(GOLANGCI_LINT)" ] || ! "$(GOLANGCI_LINT)" version 2>/dev/null | grep -q "version $(GOLANGCI_LINT_VERSION)"; then \
		echo "🔧 Installing golangci-lint $(GOLANGCI_LINT_VERSION)..."; \
		mkdir -p "$(GO_BIN)"; \
		GOBIN="$(GO_BIN)" go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION); \
	fi

#===========================================
# Code Quality Targets
#===========================================

# Auto-fix formatting for all languages
format: format-cpp format-python format-go
	@echo "✅ All formatting complete!"

format-cpp:
	@echo "🔧 Formatting C++ code..."
	@command -v $(CLANG_FORMAT) >/dev/null 2>&1 || { echo "❌ clang-format not found"; exit 1; }
	@find $(CORE_DIR)/src $(CORE_DIR)/include \( -name "*.cpp" -o -name "*.h" -o -name "*.mm" \) 2>/dev/null | \
		xargs -r $(CLANG_FORMAT) -i
	@echo "   C++ formatting done"

format-python:
	@echo "🔧 Formatting Python code..."
	@command -v $(RUFF) >/dev/null 2>&1 || { echo "❌ ruff not found"; exit 1; }
	@cd $(PYTHON_DIR) && $(RUFF) format .
	@cd $(PYTHON_DIR) && $(RUFF) check --fix .
	@echo "   Python formatting done"

format-go: ensure-golangci-lint
	@echo "🔧 Formatting and auto-fixing Go code..."
	@cd $(SERVER_DIR) && "$(GOLANGCI_LINT)" fmt
	@cd $(SERVER_DIR) && "$(GOLANGCI_LINT)" run --fix $(GOLANGCI_LINT_RUN_FLAGS) $(GOLANGCI_LINT_RUN_TARGETS)
	@echo "   Go formatting and lint auto-fix done"

# Check formatting without fixing (for CI)
format-check: format-check-cpp format-check-python format-check-go
	@echo "✅ All format checks passed!"

format-check-cpp:
	@echo "🔍 Checking C++ formatting..."
	@command -v $(CLANG_FORMAT) >/dev/null 2>&1 || { echo "❌ clang-format not found"; exit 1; }
	@find $(CORE_DIR)/src $(CORE_DIR)/include \( -name "*.cpp" -o -name "*.h" -o -name "*.mm" \) 2>/dev/null | \
		xargs -r $(CLANG_FORMAT) --dry-run --Werror

format-check-python:
	@echo "🔍 Checking Python formatting..."
	@command -v $(RUFF) >/dev/null 2>&1 || { echo "❌ ruff not found"; exit 1; }
	@cd $(PYTHON_DIR) && $(RUFF) format --check .
	@cd $(PYTHON_DIR) && $(RUFF) check .

format-check-go: ensure-golangci-lint
	@echo "🔍 Checking Go formatting..."
	@cd $(SERVER_DIR) && "$(GOLANGCI_LINT)" fmt --diff

# Run linters (no auto-fix)
lint: lint-cpp lint-python lint-go
	@echo "✅ All linting complete!"

lint-cpp:
	@echo "🔍 Linting C++ code..."
	@command -v $(CLANG_FORMAT) >/dev/null 2>&1 || { echo "❌ clang-format not found"; exit 1; }
	@find $(CORE_DIR)/src $(CORE_DIR)/include \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | \
		xargs -r $(CLANG_FORMAT) --dry-run --Werror

lint-python:
	@echo "🔍 Linting Python code..."
	@command -v $(RUFF) >/dev/null 2>&1 || { echo "❌ ruff not found"; exit 1; }
	@cd $(PYTHON_DIR) && $(RUFF) check .

lint-go: ensure-golangci-lint
	@echo "🔍 Linting Go code..."
	@cd $(SERVER_DIR) && go build $(GO_BUILD_VALIDATE_TARGETS)
	@cd $(SERVER_DIR) && "$(GOLANGCI_LINT)" run $(GOLANGCI_LINT_RUN_FLAGS) $(GOLANGCI_LINT_RUN_TARGETS)

#===========================================
# Build Targets
#===========================================

# Build C++ shared library
lib:
	@echo "=========================================="
	@echo "Building DenseCore C++ Library"
	@echo "=========================================="
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ../$(CORE_DIR) -DCMAKE_BUILD_TYPE=Release
	@cd $(BUILD_DIR) && $(MAKE)
	@echo ""
	@echo "Library built successfully: $(BUILD_DIR)/$(LIB_NAME)"
	@echo "=========================================="

# Build quantize tool
quantize:
	@echo "=========================================="
	@echo "Building Quantize Tool"
	@echo "=========================================="
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ../$(CORE_DIR) -DCMAKE_BUILD_TYPE=Release
	@cd $(BUILD_DIR) && $(MAKE) quantize
	@echo ""
	@echo "Quantize tool built successfully: $(BIN_DIR)/quantize"
	@echo "=========================================="

# Build Go server
server: lib
	@echo "=========================================="
	@echo "Building DenseCore Go Server"
	@echo "=========================================="
	@mkdir -p $(BIN_DIR)
	@cd $(SERVER_DIR) && \
		CGO_ENABLED=1 \
		CGO_LDFLAGS="-L../$(BUILD_DIR) -ldensecore -lstdc++" \
		go build -o ../$(BIN_DIR)/densecore-server .
	@echo ""
	@echo "Server built successfully: $(BIN_DIR)/densecore-server"
	@echo "=========================================="

# Build and run the server
run: server
	@echo "=========================================="
	@echo "Starting DenseCore Server"
	@echo "=========================================="
ifeq ($(UNAME_S),Darwin)
	@DYLD_LIBRARY_PATH=$(BUILD_DIR):$$DYLD_LIBRARY_PATH $(BIN_DIR)/densecore-server
else
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(BIN_DIR)/densecore-server
endif

# Test the server with a sample request
test:
	@echo "=========================================="
	@echo "Testing DenseCore Server"
	@echo "=========================================="
	@echo "Sending test request to http://localhost:8080/v1/chat/completions"
	@echo ""
	@curl -s -X POST http://localhost:8080/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{ \
			"model": "densecore-v1", \
			"messages": [{"role": "user", "content": "Hello, how are you?"}], \
			"max_tokens": 100 \
		}' | python3 -m json.tool || echo "Note: Install python3 for pretty JSON output"
	@echo ""
	@echo "=========================================="

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BIN_DIR)
	@rm -f $(SERVER_DIR)/densecore-server
	@echo "Clean complete"

# Install dependencies (for development)
deps:
	@echo "=========================================="
	@echo "Checking Dependencies"
	@echo "=========================================="
	@command -v cmake >/dev/null 2>&1 || { echo "❌ CMake not found. Please install CMake 3.14+"; exit 1; }
	@command -v go >/dev/null 2>&1 || { echo "❌ Go not found. Please install Go 1.19+"; exit 1; }
	@command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || { echo "❌ C++ compiler not found"; exit 1; }
	@echo "✓ CMake found: $$(cmake --version | head -n1)"
	@echo "✓ Go found: $$(go version)"
	@echo "✓ C++ compiler found"
	@echo "=========================================="
	@echo "All dependencies satisfied!"
	@echo "=========================================="

# Build in debug mode
debug:
	@echo "Building in debug mode..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ../$(CORE_DIR) -DCMAKE_BUILD_TYPE=Debug
	@cd $(BUILD_DIR) && $(MAKE)
	@cd $(SERVER_DIR) && \
		CGO_ENABLED=1 \
		CGO_LDFLAGS="-L../$(BUILD_DIR) -ldensecore -lstdc++" \
		go build -gcflags="all=-N -l" -o ../$(BIN_DIR)/densecore-server .
	@echo "Debug build complete"

# Generate API documentation
docs:
	@echo "=========================================="
	@echo "Generating API Documentation"
	@echo "=========================================="
	@mkdir -p docs/api
	@doxygen Doxyfile
	@echo ""
	@echo "Documentation generated at: docs/api/html/index.html"
	@echo "=========================================="

#===========================================
# Proto Generation
#===========================================

# Install gRPC Go plugins
grpc-deps:
	@echo "🔧 Installing gRPC Go plugins..."
	@go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.34.2
	@go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1
	@echo "   gRPC plugins installed"

# Generate Proto files
proto: grpc-deps
	@echo "=========================================="
	@echo "Generating Proto files"
	@echo "=========================================="
	@protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		proto/densecore.proto
	@echo "Proto generation complete"
	@echo "=========================================="
