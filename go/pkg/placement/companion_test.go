package placement

import "testing"

// The reviewer's reservation was a constant that overshot its real footprint by
// 486 MiB, permanently, on whichever GPU the planner considered least valuable
// -- here a 12 GB card where an expert layer is 1371 MiB. The measurement has to
// survive to the next launch or the constant simply returns.
func TestCompanionVRAMRoundTripsAndOnlyGrows(t *testing.T) {
	dir := t.TempDir()
	const name = "claude-auto-reviewer"

	if got := MeasuredCompanionVRAMMB(dir, name); got != 0 {
		t.Fatalf("unmeasured companion reported %d MiB, want 0 so the caller keeps its bound", got)
	}
	if err := RecordCompanionVRAM(dir, name, 2114); err != nil {
		t.Fatal(err)
	}
	if got := MeasuredCompanionVRAMMB(dir, name); got != 2114 {
		t.Errorf("measurement = %d MiB, want the recorded 2114", got)
	}
	// A later sample taken before the reviewer's KV filled must not shrink the
	// reservation, or the next long conversation overruns it.
	if err := RecordCompanionVRAM(dir, name, 1600); err != nil {
		t.Fatal(err)
	}
	if got := MeasuredCompanionVRAMMB(dir, name); got != 2114 {
		t.Errorf("a smaller sample lowered the reservation to %d MiB", got)
	}
	// A genuinely larger footprint must win.
	if err := RecordCompanionVRAM(dir, name, 2400); err != nil {
		t.Fatal(err)
	}
	if got := MeasuredCompanionVRAMMB(dir, name); got != 2400 {
		t.Errorf("measurement = %d MiB, want the larger 2400", got)
	}
}
