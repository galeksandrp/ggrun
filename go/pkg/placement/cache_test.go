package placement

import (
	"os"
	"path/filepath"
	"strings"
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

// A placement cache must be able to find and remove every cache file a model
// created — probe caches (hashed .probe names with a basename header), measured
// KV rate/geometry (kv_<basename>_<size>.cache), calibration decisions
// (cal-<hash>.json with a model marker), and placement caches
// (hashed .place files tagged with CACHED_MODEL_BASENAME) — while leaving a
// different model's caches untouched and keeping the GGUF itself in place.
func TestClearModelCachesRemovesOnlyThisModelsCaches(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{
		Path: "models/My-Model-Q4_K_M.gguf", Basename: "My-Model-Q4_K_M",
		SizeBytes: 1234, TotalSizeMB: 1,
		NumLayers: 12, NumExperts: 16, EmbeddingLength: 2048, FeedForwardLength: 4096,
	}
	other := &ModelProfile{Path: "models/Other-Model-Q4_K_M.gguf", Basename: "Other-Model-Q4_K_M"}

	// Probe cache: hashed filename, header names the basename.
	probe := probeCachePath(dir, model, 4096, 512, "mid", "gpu", "llama", nil, 1)
	if err := os.MkdirAll(filepath.Dir(probe), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(probe, []byte("# Probe cache for My-Model-Q4_K_M.gguf\nPROBED_COMPUTE_BUF_MB=2048\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Probe for the OTHER model must survive.
	otherProbe := probeCachePath(dir, other, 4096, 512, "mid", "gpu", "llama", nil, 1)
	if err := os.WriteFile(otherProbe, []byte("# Probe cache for Other-Model-Q4_K_M.gguf\nPROBED_COMPUTE_BUF_MB=2048\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Placement cache: hashed .place name with CACHED_MODEL_BASENAME tag.
	place := PlacementCachePathFor(dir, model, 4096, 512, "mid", "gpu", "llama", nil, 1, "", false)
	entry := &CacheEntry{OTString: "exps=CPU", TensorSplit: []float64{1}, SplitMode: "layer",
		NCPUMoE: 30, BatchSize: 2048, UBatchSize: 512, Parallel: 1, MMap: true, ModelBasename: "My-Model-Q4_K_M"}
	if err := SavePlacementCache(place, entry); err != nil {
		t.Fatal(err)
	}
	// Untagged placement cache (pre-marker) must be left alone.
	legacy := PlacementCachePathFor(dir, other, 4096, 512, "mid", "gpu", "llama", nil, 1, "", false)
	legacyEntry := &CacheEntry{OTString: "exps=CPU", TensorSplit: []float64{1}, SplitMode: "layer",
		NCPUMoE: 30, BatchSize: 2048, UBatchSize: 512, Parallel: 1, MMap: true}
	if err := SavePlacementCache(legacy, legacyEntry); err != nil {
		t.Fatal(err)
	}

	// KV rate cache: kv_<basename>_<size>.cache.
	kv := kvCachePath(dir, model)
	if err := os.MkdirAll(filepath.Dir(kv), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(kv, []byte("KV_BYTES_PER_TOK_f16=6912.2500\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Calibration decision with the model marker.
	calDir := filepath.Join(dir, "calibration")
	if err := os.MkdirAll(calDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := SaveCalibrationDecision(dir, CalibrationDecision{
		ScopeKey: "somehash", Winner: "default", ModelBasename: "My-Model-Q4_K_M",
	}); err != nil {
		t.Fatal(err)
	}

	// The GGUF stays.
	ggufPath := filepath.Join(dir, "My-Model-Q4_K_M.gguf")
	if err := os.WriteFile(ggufPath, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	removed, err := ClearModelCaches(dir, model)
	if err != nil {
		t.Fatalf("ClearModelCaches: %v", err)
	}
	if removed < 4 {
		t.Fatalf("expected at least 4 cache files removed (probe, place, kv, cal), got %d", removed)
	}

	if _, err := os.Stat(probe); !os.IsNotExist(err) {
		t.Errorf("model probe cache was not removed (stat err=%v)", err)
	}
	if _, err := os.Stat(place); !os.IsNotExist(err) {
		t.Errorf("model placement cache was not removed (stat err=%v)", err)
	}
	if _, err := os.Stat(kv); !os.IsNotExist(err) {
		t.Errorf("model KV cache was not removed (stat err=%v)", err)
	}
	if ents, _ := os.ReadDir(calDir); len(ents) != 0 {
		t.Errorf("model calibration decision was not removed; remaining %d entries", len(ents))
	}

	// Other model's caches survive.
	if _, err := os.Stat(otherProbe); err != nil {
		t.Errorf("other model's probe cache was removed: %v", err)
	}
	if _, err := os.Stat(legacy); err != nil {
		t.Errorf("untagged placement cache was removed: %v", err)
	}
	if _, err := os.Stat(ggufPath); err != nil {
		t.Errorf("the model GGUF itself must be kept: %v", err)
	}
}

// The clear helper must tolerate a cache directory with unrelated files
// (system probes, tune misses, catalog JSON) and no matching caches at all.
func TestClearModelCachesNoMatchesIsNoop(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "models/Only-Model.gguf", Basename: "Only-Model"}
	if err := os.WriteFile(filepath.Join(dir, "system_abc123.cache"), []byte("CUDA_OVERHEAD=512"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "catalog.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	removed, err := ClearModelCaches(dir, model)
	if err != nil {
		t.Fatalf("ClearModelCaches on a cold dir must not error: %v", err)
	}
	if removed != 0 {
		t.Fatalf("no model caches existed; removed=%d", removed)
	}
	if _, err := os.Stat(filepath.Join(dir, "system_abc123.cache")); err != nil {
		t.Errorf("system probe must be kept: %v", err)
	}
}

// probeCacheForModel matches the header line that writeProbeCacheForModel
// emits (`# Probe cache for <basename>`) and the CACHED_MODEL_BASENAME marker.
// The matcher accepts either the Basename or the file-basename form.
func TestProbeCacheForModelHeaderMatch(t *testing.T) {
	match := func(forms ...string) func(string) bool {
		return func(s string) bool {
			for _, f := range forms {
				if s == f {
					return true
				}
			}
			return false
		}
	}
	cases := []struct {
		content string
		matcher func(string) bool
		want    bool
	}{
		{"# Probe cache for My-Model-Q4_K_M.gguf\nPROBED_COMPUTE_BUF_MB=2048\n", match("My-Model-Q4_K_M.gguf", "My-Model-Q4_K_M"), true},
		{"# Probe cache for My-Model-Q4_K_M.gguf\nPROBED_COMPUTE_BUF_MB=2048\n", match("Other-Model.gguf"), false},
		{"CACHED_MODEL_BASENAME=\"My-Model-Q4_K_M\"\nCACHED_MMAP=\"1\"\n", match("My-Model-Q4_K_M.gguf", "My-Model-Q4_K_M"), true},
		{"CACHED_MODEL_BASENAME=\"Other\"\n", match("My-Model-Q4_K_M"), false},
		{"no header at all\n", match("My-Model"), false},
	}
	for _, c := range cases {
		if got := probeCacheForModel(c.content, c.matcher); got != c.want {
			t.Errorf("probeCacheForModel(%q)=%v, want %v", c.content, got, c.want)
		}
	}
}

// SavePlacementCache must round-trip the model marker so a later clear can
// match it, and an untagged legacy file must still load (grandfathered).
func TestPlacementCacheRoundTripsModelBasename(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "p.place")
	entry := &CacheEntry{OTString: "exps=CPU", TensorSplit: []float64{1}, SplitMode: "layer",
		NCPUMoE: 30, BatchSize: 2048, UBatchSize: 512, Parallel: 1, MMap: false, ModelBasename: "My-Model"}
	if err := SavePlacementCache(path, entry); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `CACHED_MODEL_BASENAME="My-Model"`) {
		t.Fatalf("placement cache file lacks the model marker: %s", data)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564, VRAMUsedMB: 100}}}
	loaded, err := LoadPlacementCache(path, caps, 0)
	if err != nil {
		t.Fatalf("tagged placement cache must load: %v", err)
	}
	if loaded.ModelBasename != "My-Model" {
		t.Fatalf("model basename did not round-trip: %q", loaded.ModelBasename)
	}
}
