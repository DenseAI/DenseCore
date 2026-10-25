//go:build !cgo

package decision

import "fmt"

func Open(path string, threads int) (Backend, error) {
	return nil, fmt.Errorf("native decision inference requires a cgo build of DenseCore")
}

func OpenWithOptions(path string, threads int, options OpenOptions) (Backend, error) {
	if _, err := options.precision(); err != nil {
		return nil, err
	}
	return Open(path, threads)
}
