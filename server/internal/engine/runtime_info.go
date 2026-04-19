package engine

import (
	"bufio"
	"os"
	"strings"
)

// ResolvedDenseCoreLibraryPath returns the mapped libdensecore shared library path
// for the current process on Linux. If the mapping cannot be found, it returns
// an empty string.
func ResolvedDenseCoreLibraryPath() string {
	file, err := os.Open("/proc/self/maps")
	if err != nil {
		return ""
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		slash := strings.IndexByte(line, '/')
		if slash < 0 {
			continue
		}
		path := strings.TrimSpace(line[slash:])
		if strings.Contains(path, "/libdensecore.so") {
			return path
		}
	}

	return ""
}
