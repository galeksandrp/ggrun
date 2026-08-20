package placement

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The reporting box: a 4070 and a 3060 flanking a 3090 Ti. PCIe bandwidth is
// identical on the 4070 and the 3090 Ti, which is exactly why the PCIe-weighted
// split could not tell them apart; VRAM bandwidth differs 2x.
func benchBox() *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282, VRAMUsedMB: 300,
				BandwidthMBps: 15760, MemBandwidthMBps: 504048},
			{Index: 1, Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 300,
				BandwidthMBps: 15760, MemBandwidthMBps: 1008096},
			{Index: 2, Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288, VRAMUsedMB: 300,
				BandwidthMBps: 7880, MemBandwidthMBps: 360048},
		},
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 24},
	}
}

func qwen27B() *ModelProfile {
	return &ModelProfile{
		Path:        "Qwen3.8-27B-UD-Q4_K_XL.gguf",
		TotalSizeMB: 17093,
		SizeBytes:   17093 * 1024 * 1024,
		NumLayers:   64,
		IsMoE:       false,
		ContextSize: 262144,
		HiddenSize:  5120,
		CTXTrain:    262144,
		HeadCountKV: 8,
		KeyLength:   128,
		ValueLength: 128,
	}
}

// Layer split runs the devices sequentially, so time per token is the SUM of
// each device's weight read. The fastest VRAM must therefore carry the largest
// share -- not the card that merely has the most free VRAM after a fair split.
func TestDenseSplitLoadsFastestVRAMFirst(t *testing.T) {
	caps, model := benchBox(), qwen27B()
	strat, err := Compute(caps, model, Options{
		ContextSize: 131072, KVPlacement: "gpu", KVQuality: "low", CacheDir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	if len(strat.TensorSplit) != 3 {
		t.Fatalf("want a 3-entry split, got %v", strat.TensorSplit)
	}
	split := strat.TensorSplit
	t.Logf("split = %v", split)
	// GPU1 is the 3090 Ti at 1008 GB/s; it must hold strictly more than either
	// slower card, and more than both together.
	if split[1] <= split[0] || split[1] <= split[2] {
		t.Errorf("fastest GPU (idx1) did not get the largest share: %v", split)
	}
	if split[1] <= split[0]+split[2] {
		t.Errorf("fastest GPU should carry the majority, got %v", split)
	}
	// The slowest card must not be preferred over the mid card.
	if split[2] > split[0] {
		t.Errorf("slowest GPU (idx2, 360 GB/s) got more than idx0 (504 GB/s): %v", split)
	}
}

// Filling fastest-first must never over-book a card: each GPU's own share of
// weights + KV plus its fixed overhead has to fit in its free VRAM. This is the
// property whose absence caused the original OOM.
func TestDenseSplitSharesStillFitEachGPU(t *testing.T) {
	caps, model := benchBox(), qwen27B()
	const ctx = 131072
	strat, err := Compute(caps, model, Options{
		ContextSize: ctx, KVPlacement: "gpu", KVQuality: "low", CacheDir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	kvTotal := computeKVTotalMB(model, ctx, strat.KVType, false)
	cudaOH := measuredCUDAOverheadMB(loadSystemProbe(t.TempDir(), caps.GPUs))
	for gi, share := range strat.TensorSplit {
		if share == 0 {
			continue // deliberately unused card
		}
		need := int(float64(model.TotalSizeMB)*share) + int(float64(kvTotal)*share) + cudaOH + computeFloorMB
		if free := caps.GPUs[gi].VRAMFreeMB(); need > free {
			t.Errorf("GPU%d needs %d MiB but has %d free (share=%.2f)", gi, need, free, share)
		}
	}
}

// Unknown hardware must keep the previous behaviour exactly: a card with no VRAM
// bandwidth reading anywhere in the set falls back to the PCIe-weighted balanced
// split, where every contributing GPU keeps a non-zero share.
func TestDenseSplitFallsBackWhenBandwidthUnknown(t *testing.T) {
	caps := benchBox()
	caps.GPUs[2].MemBandwidthMBps = 0 // one unrecognised card
	if denseMemBandwidthKnown(caps.GPUs) {
		t.Fatal("a set containing an unknown card must not count as known")
	}
	strat, err := Compute(caps, qwen27B(), Options{
		ContextSize: 131072, KVPlacement: "gpu", KVQuality: "low", CacheDir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("compute failed: %v", err)
	}
	for gi, share := range strat.TensorSplit {
		if share <= 0 {
			t.Errorf("fallback path zeroed GPU%d; balanced split must keep every GPU: %v",
				gi, strat.TensorSplit)
		}
	}
}

func TestOrderGPUsByMemoryBandwidthRanksFastestFirst(t *testing.T) {
	order := orderGPUsByMemoryBandwidth(benchBox().GPUs)
	if want := []int{1, 0, 2}; len(order) != 3 || order[0] != want[0] || order[1] != want[1] || order[2] != want[2] {
		t.Errorf("order = %v, want %v (3090Ti, 4070, 3060)", order, want)
	}
}
