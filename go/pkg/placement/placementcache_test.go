package placement

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func countOTLayersByDevice(ot string) map[int]int {
	out := map[int]int{}
	for _, part := range strings.Split(ot, ",") {
		m := otDevicePattern.FindStringSubmatch(part)
		if m == nil {
			continue
		}
		dev, _ := strconv.Atoi(m[2])
		out[dev] += len(strings.Split(m[1], "|"))
	}
	return out
}

func TestReplanAfterOOMReducesFailedDevice(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "3090 Ti", VRAMTotalMB: 24576, VRAMUsedMB: 800, BandwidthMBps: 31504},
			{Index: 1, Name: "3060", VRAMTotalMB: 12288, VRAMUsedMB: 600, BandwidthMBps: 12000},
			{Index: 2, Name: "4070", VRAMTotalMB: 12288, VRAMUsedMB: 600, BandwidthMBps: 25203},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "V4.gguf", TotalSizeMB: 146 * 1024, SizeBytes: 146 * 1024 * 1024 * 1024,
		NumLayers: 43, IsMoE: true, NumExperts: 256,
		ExpertBytes: int64(43 * 3289 * 1024 * 1024), NonExpertBytes: int64(7680 * 1024 * 1024),
		ContextSize: 32768, EmbeddingLength: 4096, HeadCountKV: 1, KeyLength: 512, ValueLength: 512,
		ExpertUsedCount: 6, ExpertFF: 2048,
	}
	opts := Options{ContextSize: 32768, KVQuality: "low", KVPlacement: "cpu", CacheDir: t.TempDir()}

	base, err := Compute(caps, model, opts)
	if err != nil || base.Type != MoEOffload {
		t.Fatalf("base compute: type=%v err=%v", base.Type, err)
	}
	baseC := countOTLayersByDevice(base.OTString)

	// Device 2 OOM'd by ~8 GB — the re-plan must shed layers from it.
	replan, err := ReplanAfterOOM(caps, model, opts, map[int]int{2: 8000})
	if err != nil || replan == nil {
		t.Fatalf("replan: %v", err)
	}
	newC := countOTLayersByDevice(replan.OTString)
	if newC[2] >= baseC[2] {
		t.Errorf("re-plan should reduce device 2 layers: base=%d new=%d (ot=%s)", baseC[2], newC[2], replan.OTString)
	}
	if replan.OTString == "" || !strings.Contains(replan.OTString, "exps=CPU") {
		t.Errorf("re-plan produced no valid MoE -ot: %q", replan.OTString)
	}
}

func TestPlacementCachePathFor_KeyedByKVAndCtx(t *testing.T) {
	m := &ModelProfile{Path: "model.gguf", NumLayers: 43, NumExperts: 256, EmbeddingLength: 4096}
	gpus := []detect.GPU{{Index: 0, Name: "3090 Ti"}, {Index: 1, Name: "3060"}}
	dir := "/tmp/cache"

	gpuKV := PlacementCachePathFor(dir, m, 1048576, 512, "mid", "gpu", "llama", gpus, 0, "", false)
	cpuKV := PlacementCachePathFor(dir, m, 1048576, 512, "mid", "cpu", "llama", gpus, 0, "", false)
	smallCtx := PlacementCachePathFor(dir, m, 131072, 512, "mid", "gpu", "llama", gpus, 0, "", false)

	if gpuKV == cpuKV {
		t.Errorf("kv=gpu and kv=cpu must not share a placement cache file:\n  %s", gpuKV)
	}
	if gpuKV == smallCtx {
		t.Errorf("different context sizes must not share a placement cache file")
	}
	// Deterministic + lands in the cache dir with a .place extension.
	if again := PlacementCachePathFor(dir, m, 1048576, 512, "mid", "gpu", "llama", gpus, 0, "", false); again != gpuKV {
		t.Errorf("path must be deterministic: %s != %s", again, gpuKV)
	}
	if filepath.Dir(gpuKV) != dir || filepath.Ext(gpuKV) != ".place" {
		t.Errorf("unexpected path %s (want %s/*.place)", gpuKV, dir)
	}
}

func TestPlacementAndProbeCachesIncludeEveryShardIdentity(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "model-00001-of-00002.gguf")
	second := filepath.Join(dir, "model-00002-of-00002.gguf")
	if err := os.WriteFile(first, []byte("first"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(second, []byte("second"), 0o644); err != nil {
		t.Fatal(err)
	}
	model := &ModelProfile{Path: first, SizeBytes: 11, TotalSizeMB: 1, NumLayers: 4, NumExperts: 8, EmbeddingLength: 16}
	gpus := []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 12000}}
	placeBefore := PlacementCachePathFor(dir, model, 4096, 128, "q8_0", "gpu", "build", gpus, 1, "", false)
	probeBefore := probeCachePath(dir, model, 4096, 128, "q8_0", "gpu", "build", gpus, 1)

	// Only the second shard changes. First-shard-only keys would incorrectly
	// reuse the old exact tensor placement and allocation measurement.
	if err := os.WriteFile(second, []byte("second shard replaced"), 0o644); err != nil {
		t.Fatal(err)
	}
	placeAfter := PlacementCachePathFor(dir, model, 4096, 128, "q8_0", "gpu", "build", gpus, 1, "", false)
	probeAfter := probeCachePath(dir, model, 4096, 128, "q8_0", "gpu", "build", gpus, 1)
	if placeBefore == placeAfter || probeBefore == probeAfter {
		t.Fatalf("second-shard replacement reused evidence: place %q/%q probe %q/%q", placeBefore, placeAfter, probeBefore, probeAfter)
	}
}

func TestWorkloadProfileScopesPlacementAndProbeCacheIdentity(t *testing.T) {
	model := &ModelProfile{Path: "model.gguf", NumLayers: 43, NumExperts: 256, EmbeddingLength: 4096}
	gpus := []detect.GPU{{Index: 0, Name: "3090 Ti"}}
	dir := t.TempDir()
	legacyTag := ScopedBackendCacheTag("llama", "")
	interactiveTag := ScopedBackendCacheTag("llama", "claude-agent-interactive-v1:custom-test")
	parallelTag := ScopedBackendCacheTag("llama", "claude-agent-parallel-v1:custom-test")
	if legacyTag != "llama" {
		t.Fatalf("legacy generic tag changed unexpectedly: %q", legacyTag)
	}
	if interactiveTag == legacyTag || parallelTag == legacyTag || interactiveTag == parallelTag {
		t.Fatalf("workload tags are not isolated: legacy=%q interactive=%q parallel=%q", legacyTag, interactiveTag, parallelTag)
	}
	legacyPlace := PlacementCachePathFor(dir, model, 65536, 512, "high", "gpu", legacyTag, gpus, 1, "", false)
	interactivePlace := PlacementCachePathFor(dir, model, 65536, 512, "high", "gpu", interactiveTag, gpus, 1, "", false)
	parallelPlace := PlacementCachePathFor(dir, model, 65536, 512, "high", "gpu", parallelTag, gpus, 4, "", false)
	if legacyPlace == interactivePlace || legacyPlace == parallelPlace || interactivePlace == parallelPlace {
		t.Fatalf("placement cache paths are not workload-scoped: %q %q %q", legacyPlace, interactivePlace, parallelPlace)
	}
	legacyProbe := probeCachePath(dir, model, 65536, 512, "high", "gpu", legacyTag, gpus, 1)
	interactiveProbe := probeCachePath(dir, model, 65536, 512, "high", "gpu", interactiveTag, gpus, 1)
	parallelProbe := probeCachePath(dir, model, 65536, 512, "high", "gpu", parallelTag, gpus, 4)
	if legacyProbe == interactiveProbe || legacyProbe == parallelProbe || interactiveProbe == parallelProbe {
		t.Fatalf("probe cache paths are not workload-scoped: %q %q %q", legacyProbe, interactiveProbe, parallelProbe)
	}
	if got := backendCacheTag(Options{BackendCacheTag: "llama", WorkloadProfile: "claude-agent-interactive-v1:custom-test"}); got != interactiveTag {
		t.Fatalf("options workload tag=%q, want %q", got, interactiveTag)
	}
}

func TestPlacementCachePathIsolatedBySpecMode(t *testing.T) {
	base := "/tmp/model.place"
	if got := placementCachePathForSpecMode(base, "off"); got != base {
		t.Fatalf("spec-off path changed: %s", got)
	}
	auto := placementCachePathForSpecMode(base, "auto")
	dflash := placementCachePathForSpecMode(base, "dflash")
	if auto == base || dflash == base || auto == dflash {
		t.Fatalf("spec placements are not isolated: base=%s auto=%s dflash=%s", base, auto, dflash)
	}
}

func TestSpecCacheIdentityIncludesCompanionArtifact(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "draft-a.gguf")
	b := filepath.Join(dir, "draft-b.gguf")
	if err := os.WriteFile(a, []byte("a"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(b, []byte("different"), 0o600); err != nil {
		t.Fatal(err)
	}
	draftA := &DraftConfig{Type: DraftDFlash, Path: a, DraftGPU: 0, CTXSizeDraft: 1048576, KVTypeDraft: "q4_0", VRAMMB: 7500}
	draftB := &DraftConfig{Type: DraftDFlash, Path: b, DraftGPU: 0, CTXSizeDraft: 1048576, KVTypeDraft: "q4_0", VRAMMB: 7500}
	if placementCachePathForSpec("model.place", "dflash", draftA) == placementCachePathForSpec("model.place", "dflash", draftB) {
		t.Fatal("different DFlash artifacts shared a placement cache")
	}
	if SpecWorkloadProfile("claude", draftA) == SpecWorkloadProfile("claude", draftB) {
		t.Fatal("different DFlash artifacts shared probe evidence")
	}
}

func TestPlacementCacheHitStillResolvesSpeculativeMode(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 65536},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "cached-moe.gguf", TotalSizeMB: 32768, SizeBytes: 32768 * 1024 * 1024,
		NumLayers: 32, IsMoE: true, NumExperts: 64, ContextSize: 32768,
		ExpertBytes: 28 * 1024 * 1024 * 1024, NonExpertBytes: 4 * 1024 * 1024 * 1024,
	}
	opts := Options{ContextSize: 32768, KVQuality: "low", KVPlacement: "cpu", CacheDir: t.TempDir()}
	base, err := Compute(caps, model, opts)
	if err != nil || base.PlacementCachePath == "" {
		t.Fatalf("base placement: strategy=%#v err=%v", base, err)
	}
	if err := SavePlacementCache(base.PlacementCachePath, StrategyToCacheEntry(base)); err != nil {
		t.Fatalf("save placement: %v", err)
	}
	opts.SpecMode = "ngram"
	opts.ForceSpecMoE = true
	cached, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatalf("cached placement: %v", err)
	}
	if cached.Draft == nil || cached.Draft.Type != DraftNgram {
		t.Fatalf("cache hit silently dropped speculative mode: %#v", cached.Draft)
	}
}

func TestPlacementCacheRejectsLegacyMissingMMap(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.place")
	content := `CACHED_OT_STRING="exps=CPU"
CACHED_TENSOR_SPLIT="0.86,0.03,0.11"
CACHED_SPLIT_MODE="layer"
CACHED_NCPUMOE="34"
CACHED_BATCH="2048"
CACHED_UBATCH="512"
CACHED_PARALLEL="4"
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	if _, err := LoadPlacementCache(path, caps, 0); err == nil || !strings.Contains(err.Error(), "CACHED_MMAP") {
		t.Fatalf("expected legacy cache without mmap mode to be rejected, got %v", err)
	}
}

func TestPlacementCacheRoundTripPreservesSubPins(t *testing.T) {
	// A placement WITH sub-layer gate+up pins must survive save -> load exactly,
	// so the squeeze isn't silently dropped on the next launch.
	ot := `blk\.(0|1|2)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,` +
		`blk\.(8)\.ffn_(gate_up|up_gate|gate|up)_exps.*=CUDA0,exps=CPU`
	strat := &Strategy{
		Type:        MoEOffload,
		OTString:    ot,
		TensorSplit: []float64{0.86, 0.03, 0.11},
		SplitMode:   "layer",
		NCPUMoE:     33,
		BatchSize:   2048,
		UBatchSize:  512,
		Parallel:    1,
		MMap:        false,
	}
	entry := StrategyToCacheEntry(strat)
	if entry.OTString != ot {
		t.Fatalf("StrategyToCacheEntry dropped OTString")
	}

	path := filepath.Join(t.TempDir(), "x.place")
	if err := SavePlacementCache(path, entry); err != nil {
		t.Fatalf("save: %v", err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	loaded, err := LoadPlacementCache(path, caps, 0)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.OTString != ot {
		t.Errorf("sub-pin -ot not preserved:\n got=%s\nwant=%s", loaded.OTString, ot)
	}
	if loaded.NCPUMoE != 33 || len(loaded.TensorSplit) != 3 {
		t.Errorf("cache round-trip lost fields: ncpumoe=%d split=%v", loaded.NCPUMoE, loaded.TensorSplit)
	}
}

// A --swa-full launch allocates a different KV cache than a plain one at the
// same context -- 4x on an interleaved sliding-window model -- so it must not
// be handed a plan validated without it. The key carried every other input that
// moves the allocation but not this one.
func TestPlacementCacheKeySeparatesSWAFull(t *testing.T) {
	dir := t.TempDir()
	m := &ModelProfile{Path: "/models/laguna.gguf", NumLayers: 48, NumExperts: 128,
		EmbeddingLength: 4096, FeedForwardLength: 14336}
	gpus := []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564}}

	plain := PlacementCachePathFor(dir, m, 81920, 512, "mid", "gpu", "laguna", gpus, 1, "", false)
	full := PlacementCachePathFor(dir, m, 81920, 512, "mid", "gpu", "laguna", gpus, 1, "", true)

	if plain == "" || full == "" {
		t.Fatal("expected a cache path for both")
	}
	if plain == full {
		t.Errorf("--swa-full shares a placement cache entry with a plain launch: %s", plain)
	}
	if again := PlacementCachePathFor(dir, m, 81920, 512, "mid", "gpu", "laguna", gpus, 1, "", true); again != full {
		t.Errorf("key is not stable for the same inputs: %s vs %s", again, full)
	}
}

// A "launch without cached config" must derive fresh: Compute with
// SkipCachedConfig ignores an existing placement cache, probe cache, and
// measured KV rate that a normal Compute would reuse, without deleting the
// files (they stay on disk for later launches).
func TestSkipCachedConfigIgnoresExistingCaches(t *testing.T) {
	dir := t.TempDir()
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "3090 Ti", VRAMTotalMB: 24576, VRAMUsedMB: 800, BandwidthMBps: 31504},
			{Index: 1, Name: "3060", VRAMTotalMB: 12288, VRAMUsedMB: 600, BandwidthMBps: 12000},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 120000},
		CPU: detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: "V4.gguf", TotalSizeMB: 80 * 1024, SizeBytes: 80 * 1024 * 1024 * 1024,
		NumLayers: 43, IsMoE: true, NumExperts: 256,
		ExpertBytes: int64(43 * 1800 * 1024 * 1024), NonExpertBytes: int64(7680 * 1024 * 1024),
		ContextSize: 32768, EmbeddingLength: 4096, HeadCountKV: 1, KeyLength: 512, ValueLength: 512,
		ExpertUsedCount: 6, ExpertFF: 2048,
	}
	opts := Options{ContextSize: 32768, KVQuality: "low", KVPlacement: "cpu", CacheDir: dir}

	// Establish the cached evidence a normal launch would reuse: a measured KV
	// rate and a placement cache at the exact path Compute derives.
	// writeMeasuredKVRate would also set the in-memory model field, so write the
	// cache file directly to keep the model clean for the skip-cached check.
	kvPath := kvCachePath(dir, model)
	if err := os.MkdirAll(filepath.Dir(kvPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(kvPath, []byte("KV_BYTES_PER_TOK_q8_0=2048.0000\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Compute the placement path (derives the key) then write a tagged cache.
	place := PlacementCachePathFor(dir, model, 32768, opts.UBatchSize, "low", "cpu", "llama", caps.GPUs, 1, "", false)
	if err := SavePlacementCache(place, &CacheEntry{
		OTString: "exps=CPU", TensorSplit: []float64{1}, SplitMode: "layer",
		NCPUMoE: 30, BatchSize: 2048, UBatchSize: 512, Parallel: 1, MMap: true, ModelBasename: "V4.gguf",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(place); err != nil {
		t.Fatalf("expected placement cache file: %v", err)
	}

	// A Compute with SkipCachedConfig must not load the measured KV rate.
	freshModel := *model
	skipOpts := opts
	skipOpts.SkipCachedConfig = true
	if _, err := Compute(caps, &freshModel, skipOpts); err != nil {
		t.Fatalf("skip-cached compute: %v", err)
	}
	if freshModel.MeasuredKVBytesPerTok != nil {
		t.Errorf("SkipCachedConfig must not load measured KV rates, got %v", freshModel.MeasuredKVBytesPerTok)
	}

	// The cache files themselves must still be on disk (non-destructive).
	if _, err := os.Stat(kvPath); err != nil {
		t.Errorf("SkipCachedConfig must leave the KV cache on disk, stat err=%v", err)
	}
	if _, err := os.Stat(place); err != nil {
		t.Errorf("SkipCachedConfig must leave the placement cache on disk, stat err=%v", err)
	}
}
