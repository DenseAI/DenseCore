.PHONY: all build docker-build docker-push install uninstall deploy manifests generate fmt vet

# Image URL to use for building/pushing
IMG ?= densecore/operator:latest

# Build manager binary
all: build

build: generate fmt vet
	go build -o bin/manager cmd/main.go

# Run against the configured cluster
run: generate fmt vet
	go run cmd/main.go

# Install CRDs into cluster
install: manifests
	kubectl apply -f config/crd/bases

# Uninstall CRDs from cluster
uninstall:
	kubectl delete -f config/crd/bases

# Deploy operator into cluster
deploy: manifests
	kubectl apply -f config/crd/bases
	kubectl apply -f config/rbac/role.yaml
	kubectl apply -f config/rbac/role_binding.yaml
	kubectl apply -f config/manager/manager.yaml

# Generate manifests (CRD, RBAC)
manifests:
	controller-gen rbac:roleName=manager-role crd webhook paths="./..." output:crd:artifacts:config=config/crd/bases

# Generate code (deepcopy)
generate:
	controller-gen object:headerFile="hack/boilerplate.go.txt" paths="./..."

# Format code
fmt:
	go fmt ./...

# Vet code
vet:
	go vet ./...

# Build docker image
docker-build:
	docker build -t ${IMG} .

# Push docker image
docker-push:
	docker push ${IMG}

# Run tests
test:
	go test ./... -coverprofile cover.out
