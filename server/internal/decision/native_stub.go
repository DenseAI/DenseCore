//go:build !cgo

package decision

import "fmt"

func Open(path string, threads int) (Backend, error) {
	return nil, fmt.Errorf("native Laya inference requires a cgo build of DenseCore")
}
