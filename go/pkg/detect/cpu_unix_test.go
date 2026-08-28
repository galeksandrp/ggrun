//go:build !windows

package detect

import "testing"

func TestParseLinuxCPUList(t *testing.T) {
	got := parseLinuxCPUList("0-2,8,10-11")
	for _, id := range []int{0, 1, 2, 8, 10, 11} {
		if !got[id] {
			t.Fatalf("allowed CPU %d missing from %#v", id, got)
		}
	}
	for _, id := range []int{3, 9, 12} {
		if got[id] {
			t.Fatalf("CPU %d incorrectly allowed in %#v", id, got)
		}
	}
}

func TestParseLinuxCPUListRejectsMalformedRanges(t *testing.T) {
	got := parseLinuxCPUList("4-2,-1,nope,7")
	if len(got) != 1 || !got[7] {
		t.Fatalf("malformed list was not rejected conservatively: %#v", got)
	}
}
