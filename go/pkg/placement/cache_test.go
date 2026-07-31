package placement

import (
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// A plan computed while the previous server was still releasing VRAM pins the
// expert count at the old, tighter world -- and the cache replays it on every
// restart. Measured live: 8.4 GB idle on the split owner, --n-cpu-moe 30 where
// a fresh plan chooses ~25.
func TestPlacementCacheRejectsPlanMadeUnderTighterVRAM(t *testing.T) {
	path := filepath.Join(t.TempDir(), "p.place")
	entry := &CacheEntry{
		OTString:     "exps=CPU",
		TensorSplit:  []float64{1, 0, 0},
		SplitMode:    "layer",
		NCPUMoE:      30,
		BatchSize:    2048,
		UBatchSize:   512,
		Parallel:     1,
		MMap:         false,
		PlanFreeVRAM: map[int]int{0: 2000},
	}
	if err := SavePlacementCache(path, entry); err != nil {
		t.Fatal(err)
	}
	richer := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564, VRAMUsedMB: 24564 - 10400}}}
	if _, err := LoadPlacementCache(path, richer, 0); err == nil {
		t.Fatal("a plan made against 2000MB free must not be replayed when 10400MB is free")
	}
	sameWorld := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564, VRAMUsedMB: 24564 - 2300}}}
	loaded, err := LoadPlacementCache(path, sameWorld, 0)
	if err != nil || loaded == nil {
		t.Fatalf("within slack the plan must replay: %v", err)
	}
	if loaded.PlanFreeVRAM[0] != 2000 {
		t.Errorf("plan-time reading did not survive the round-trip: %v", loaded.PlanFreeVRAM)
	}
}

// Entries from before the field carry no reading to compare; the deficit
// direction is still covered by the launch preflight, so they stay valid.
func TestPlacementCacheGrandfathersEntriesWithoutPlanVRAM(t *testing.T) {
	path := filepath.Join(t.TempDir(), "p.place")
	legacy := &CacheEntry{
		OTString: "exps=CPU", TensorSplit: []float64{1}, SplitMode: "layer",
		NCPUMoE: 30, BatchSize: 2048, UBatchSize: 512, Parallel: 1,
	}
	if err := SavePlacementCache(path, legacy); err != nil {
		t.Fatal(err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564, VRAMUsedMB: 100}}}
	if _, err := LoadPlacementCache(path, caps, 0); err != nil {
		t.Fatalf("legacy entry without the field must stay valid: %v", err)
	}
}
