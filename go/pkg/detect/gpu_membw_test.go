package detect

import "testing"

// The three cards this was designed against, with the clocks nvidia-smi
// actually reports for them.
func TestMemoryBandwidthMatchesKnownCards(t *testing.T) {
	cases := []struct {
		name    string
		clock   int
		wantGBs int // nominal spec bandwidth, GB/s
	}{
		{"NVIDIA GeForce RTX 3090 Ti", 10501, 1008},
		{"NVIDIA GeForce RTX 4070", 10501, 504},
		{"NVIDIA GeForce RTX 3060", 7501, 360},
	}
	for _, tc := range cases {
		got := memoryBandwidthMBps(tc.name, tc.clock)
		if gbs := got / 1000; gbs != tc.wantGBs {
			t.Errorf("%s: %d MB/s (%d GB/s), want ~%d GB/s", tc.name, got, gbs, tc.wantGBs)
		}
	}
}

// Longest-match wins, so a Ti/Super variant is never shadowed by the shorter
// base name sitting earlier in the table.
func TestMemoryBusWidthPrefersLongestMatch(t *testing.T) {
	cases := []struct {
		name string
		want int
	}{
		{"NVIDIA GeForce RTX 3090 Ti", 384},
		{"NVIDIA GeForce RTX 3080", 320},
		{"NVIDIA GeForce RTX 3080 Ti", 384},
		{"NVIDIA GeForce RTX 3060", 192},
		{"NVIDIA GeForce RTX 3060 Ti", 256},
		{"NVIDIA GeForce RTX 4070 Ti", 192},
		{"NVIDIA GeForce RTX 4070 Ti SUPER", 256},
	}
	for _, tc := range cases {
		if got := memoryBusWidthBits(tc.name); got != tc.want {
			t.Errorf("%s: bus width %d, want %d", tc.name, got, tc.want)
		}
	}
}

// An unknown card must yield 0 rather than a guess: 0 is the signal that keeps
// placement on its existing PCIe-derived weighting.
func TestMemoryBandwidthUnknownCardIsZero(t *testing.T) {
	if got := memoryBandwidthMBps("Some Future GPU 9999", 12000); got != 0 {
		t.Errorf("unknown card returned %d, want 0", got)
	}
	if got := memoryBandwidthMBps("", 12000); got != 0 {
		t.Errorf("empty name returned %d, want 0", got)
	}
	if got := memoryBandwidthMBps("NVIDIA GeForce RTX 3090 Ti", 0); got != 0 {
		t.Errorf("missing clock returned %d, want 0", got)
	}
}
