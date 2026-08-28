//go:build !linux

package benchmark

// NewProcessTreeResourceSampler fails closed on platforms without the Linux
// /proc counters. GPU and request timing evidence remain available.
func NewProcessTreeResourceSampler(int) func() ResourceSnapshot {
	return func() ResourceSnapshot { return ResourceSnapshot{} }
}
