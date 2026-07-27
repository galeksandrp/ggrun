package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The exact lines llama.cpp printed for Laguna at 1M context, q4_0 K/V.
const lagunaKVLog = `
0.48.540.324 I llama_kv_cache_iswa: creating non-SWA KV cache, size = 1048576 cells
0.48.547.909 I llama_kv_cache:        CPU KV buffer size = 13824.00 MiB
0.49.565.811 I llama_kv_cache: size = 13824.00 MiB (1048576 cells,  12 layers,  1/1 seqs), K (q4_0): 6912.00 MiB, V (q4_0): 6912.00 MiB
0.49.565.836 I llama_kv_cache_iswa: creating     SWA KV cache, size = 1024 cells
0.49.565.927 I llama_kv_cache:        CPU KV buffer size =    40.50 MiB
0.49.569.136 I llama_kv_cache: size =   40.50 MiB (  1024 cells,  36 layers,  1/1 seqs), K (q4_0):   20.25 MiB, V (q4_0):   20.25 MiB
`

func TestKVGeometryMatchesTheBackend(t *testing.T) {
	g, ok := ParseKVGeometry(lagunaKVLog)
	if !ok {
		t.Fatal("did not parse the backend's own KV layout")
	}
	if g.FullLayers != 12 || g.SWALayers != 36 || g.SWACells != 1024 {
		t.Errorf("layout = %d full / %d swa / %d swa cells, want 12/36/1024",
			g.FullLayers, g.SWALayers, g.SWACells)
	}
	// n_embd_k_gqa 1024 x 0.5625 B (q4_0) x 2 (K and V) = 1152 bytes.
	if g.BytesPerCellPerLayer != 1152 {
		t.Errorf("bytes per cell per layer = %.2f, want 1152", g.BytesPerCellPerLayer)
	}
	// Reproduce the allocation the backend actually made.
	if got := g.TotalMB(1048576, false); got != 13864 && got != 13865 {
		t.Errorf("TotalMB(1M) = %d, want 13864 (13824 + 40.5)", got)
	}
	// --swa-full gives all 48 layers the full context.
	if got, want := g.TotalMB(1048576, true), 48*1152*1048576/(1024*1024); got != want {
		t.Errorf("TotalMB(1M, swa-full) = %d MiB, want %d", got, want)
	}
	// The linear bytes-per-token model ggrun used cannot express this: it would
	// predict the same 13864 with and without --swa-full.
	if g.TotalMB(1048576, true) <= g.TotalMB(1048576, false)*3 {
		t.Error("swa-full must be several times larger; a linear rate would miss it")
	}
	// Halving the context does not halve the total, because the windowed cache
	// is fixed-depth.
	half := g.TotalMB(524288, false)
	if half != 6952 && half != 6953 {
		t.Errorf("TotalMB(512k) = %d, want 6952 (6912 + 40.5), not half of 13864", half)
	}
}

// A model without sliding-window layers prints one cache line and must land on
// the same accounting rather than a special case.
func TestKVGeometryHandlesNonSWAModels(t *testing.T) {
	const log = `llama_kv_cache: size =  6912.00 MiB (1048576 cells,  12 layers,  1/1 seqs), K (f16): 3456.00 MiB`
	g, ok := ParseKVGeometry(log)
	if !ok || g.FullLayers != 12 || g.SWALayers != 0 {
		t.Fatalf("non-SWA layout = %+v, ok=%v", g, ok)
	}
	if a, b := g.TotalMB(1048576, false), g.TotalMB(1048576, true); a != b {
		t.Errorf("swa-full changed a model with no SWA layers: %d -> %d", a, b)
	}
	if got := g.TotalMB(524288, false); got != 3456 {
		t.Errorf("TotalMB(512k) = %d, want exactly half of 6912", got)
	}
}

// The geometry has to survive the round trip through the cache file, or every
// launch re-measures and placement keeps using the estimate in between.
func TestKVGeometryRoundTripsThroughTheCacheFile(t *testing.T) {
	dir := t.TempDir()
	model := &ModelProfile{Path: "m.gguf", Basename: "m.gguf", SizeBytes: 4242}
	RunPostLaunchKVProbe(dir, model, 1048576, "q4_0", lagunaKVLog)

	if got := model.MeasuredKVGeometry["q4_0"]; got.FullLayers != 12 || got.SWALayers != 36 {
		t.Fatalf("probe did not record the geometry: %+v", got)
	}
	reloaded := loadMeasuredKVGeometry(dir, &ModelProfile{Path: "m.gguf", Basename: "m.gguf", SizeBytes: 4242})
	g, ok := reloaded["q4_0"]
	if !ok {
		t.Fatal("geometry did not survive the cache file")
	}
	if g != model.MeasuredKVGeometry["q4_0"] {
		t.Errorf("reloaded %+v, wrote %+v", g, model.MeasuredKVGeometry["q4_0"])
	}
	// And it must beat the linear rate that the same probe also stored.
	fresh := &ModelProfile{Path: "m.gguf", Basename: "m.gguf", SizeBytes: 4242,
		MeasuredKVBytesPerTok: map[string]float64{"q4_0": 13864.5},
		MeasuredKVGeometry:    reloaded}
	if withFull, plain := computeKVTotalMB(fresh, 1048576, "q4_0", true), computeKVTotalMB(fresh, 1048576, "q4_0", false); withFull <= plain {
		t.Errorf("--swa-full priced at %d MiB against %d without: the rate is still winning", withFull, plain)
	}
}

// A flat 1024 MiB compute floor on expert-only GPUs is a fixed margin in a
// planner built on measurement. Measured on this project: 99 MiB against that
// floor, so 925 MiB was withheld -- 67% of a 1371 MiB expert layer, and always
// on the smallest card, since that is where the reviewer is seated.
//
// The per-GPU value was distrusted because a GPU measured as expert-only may be
// a split owner next time. The magnitude settles it: a split owner measured
// 4267 MiB on the same launch, so a value 43x smaller can only have come from a
// run where that GPU was already expert-only.
func TestExpertOnlyComputeReserveTrustsAnUnambiguousMeasurement(t *testing.T) {
	for _, tc := range []struct {
		name         string
		splitOwnerMB int
		measuredMB   int
		wantAtMost   int
		wantExactly  int
	}{
		{"the real measurement is accepted with headroom", 4267, 99, computeFloorMB, 148},
		{"no measurement keeps the floor", 4267, 0, computeFloorMB, computeFloorMB},
		{"no aggregate keeps the floor", 0, 99, computeFloorMB, computeFloorMB},
		{"ambiguous magnitude keeps the floor", 4267, 1000, computeFloorMB, computeFloorMB},
		{"a large measurement never exceeds the floor", 40000, 4000, computeFloorMB, computeFloorMB},
	} {
		got := expertOnlyComputeReserveMB(tc.splitOwnerMB, tc.measuredMB)
		if got != tc.wantExactly {
			t.Errorf("%s: reserve = %d MiB, want %d", tc.name, got, tc.wantExactly)
		}
		if got > tc.wantAtMost {
			t.Errorf("%s: reserve %d exceeds the floor %d", tc.name, got, tc.wantAtMost)
		}
	}
	// The point of the change: the measured case must free most of a layer.
	if freed := computeFloorMB - expertOnlyComputeReserveMB(4267, 99); freed < 800 {
		t.Errorf("only %d MiB reclaimed; the floor withheld 925", freed)
	}
}

// A context checkpoint is only useful if one sits at or before the point a
// later turn resumes from. The backend builds each checkpoint 2*n_swa wide and
// rejects any whose pos_max overshoots the resume point, so spacing decides
// whether the mechanism can work at all.
//
// Measured on a 16k prompt at llama.cpp's 8192 default: a 92% prefix match was
// found, the only two checkpoints were [14728,15751] and [15240,16263], the
// resume point was ~14906, and both were rejected for overshooting -- one by
// just 845 tokens. All 164,358 prompt tokens were re-processed.
func TestCheckpointSpacingTilesTheContext(t *testing.T) {
	swa := &ModelProfile{SlidingWindow: 512}
	// The backend built the SWA cache at exactly this depth on this project:
	// "creating SWA KV cache, size = 1024 cells" with n_swa 512 and -ub 512,
	// and every checkpoint spanned pos_max-pos_min = 1023.
	if got, want := checkpointMinStep(swa, 512), 1024; got != want {
		t.Errorf("spacing at ub=512 = %d, want the cache depth %d", got, want)
	}
	// Not 2*n_swa: that coincided only because the window equalled the ubatch.
	if got, want := checkpointMinStep(swa, 256), 768; got != want {
		t.Errorf("spacing at ub=256 = %d, want n_swa+n_ubatch = %d; 2*n_swa would be a third too coarse", got, want)
	}
	if got := checkpointMinStep(swa, 512); got > 8192 {
		t.Errorf("spacing %d is no better than llama.cpp's default", got)
	}

	// Replay the failing case. With contiguous spacing a checkpoint must land
	// at or before the resume point rather than straddling it.
	const resume, width = 14906, 1024
	step := checkpointMinStep(swa, 512)
	best := -1
	for posMin := 0; posMin+width <= resume; posMin += step {
		best = posMin
	}
	if best < 0 {
		t.Fatal("no checkpoint lands before the resume point")
	}
	if posMax := best + width - 1; posMax > resume {
		t.Errorf("checkpoint [%d,%d] still overshoots %d", best, posMax, resume)
	}
	if skipped := float64(best) / resume; skipped < 0.8 {
		t.Errorf("resuming at %d skips only %.0f%% of the prefill", best, skipped*100)
	}

	// A model without a sliding window keeps llama.cpp's default: the depth is
	// not derived this way there.
	if got := checkpointMinStep(&ModelProfile{}, 512); got != 0 {
		t.Errorf("non-SWA model got spacing %d, want the backend default", got)
	}
	if got := checkpointMinStep(nil, 512); got != 0 {
		t.Errorf("nil model got spacing %d", got)
	}
}

// The system CUDA overhead probe measured a whole device's usage minus the
// server's own logged buffers, so anything else resident on that card became
// permanent "overhead". ggrun seats the Auto reviewer on the least valuable
// GPU by design, so the card least able to afford it is exactly the one that
// got charged: 2299 MiB recorded on a 12 GB card against 397 and 255 on the
// other two, then applied on every later placement on top of the reviewer's own
// reservation. At 1371 MiB per expert layer that is a full layer, permanently.
func TestContaminatedSystemOverheadIsDiscarded(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{
		{Index: 0, Name: "RTX 3090 Ti", Driver: "580"},
		{Index: 1, Name: "RTX 3060", Driver: "580"},
		{Index: 2, Name: "RTX 4070", Driver: "580"},
	}
	path := filepath.Join(dir, fmt.Sprintf("system_%s.cache", gpuSignatureHash(gpus)))
	// Exactly what this project recorded, without a schema marker.
	body := "SYS_CUDA_OVERHEAD_MB_CUDA0=397\nSYS_CUDA_OVERHEAD_MB_CUDA1=2299\nSYS_CUDA_OVERHEAD_MB_CUDA2=255\nSYS_CUDA_OVERHEAD_MB=2299\n"
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	got := SystemCUDAOverheadByGPU(dir, gpus)
	if _, ok := got[1]; ok {
		t.Errorf("kept the contaminated 2299 MiB reading for CUDA1: %v", got)
	}
	for _, idx := range []int{0, 2} {
		if got[idx] == 0 {
			t.Errorf("discarded the plausible reading for CUDA%d: %v", idx, got)
		}
	}
}

// Plausible spreads must survive, or every multi-GPU host re-measures forever.
func TestPlausibleSystemOverheadSpreadIsKept(t *testing.T) {
	dir := t.TempDir()
	gpus := []detect.GPU{
		{Index: 0, Name: "RTX 3090 Ti", Driver: "580"},
		{Index: 1, Name: "RTX 3060", Driver: "580"},
	}
	path := filepath.Join(dir, fmt.Sprintf("system_%s.cache", gpuSignatureHash(gpus)))
	if err := os.WriteFile(path, []byte("SYS_CUDA_OVERHEAD_MB_CUDA0=488\nSYS_CUDA_OVERHEAD_MB_CUDA1=311\nSYS_CUDA_OVERHEAD_MB=488\n"), 0644); err != nil {
		t.Fatal(err)
	}
	got := SystemCUDAOverheadByGPU(dir, gpus)
	if got[0] != 488 || got[1] != 311 {
		t.Errorf("discarded a plausible spread: %v", got)
	}
}
