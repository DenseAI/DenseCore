package decision

import "fmt"

// OpenOptions selects precision explicitly. Q8_0 is an experimental candidate
// and requires its own quality evaluation before any parity or accuracy claim.
type OpenOptions struct{ Precision string }

func (o OpenOptions) precision() (int, error) {
	switch o.Precision {
	case "", "fp32":
		return 0, nil
	case "q8_0":
		return 1, nil
	default:
		return 0, fmt.Errorf("precision must be fp32 or q8_0")
	}
}
