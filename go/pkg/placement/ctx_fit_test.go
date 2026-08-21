package placement

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// "fit" must size the context against the VRAM the KV will live in. A RAM-rich
// box used to hand the whole of free system RAM to the KV budget, so fit picked
// the model's native context and the resulting KV had to be offloaded to the
// host -- the speed loss fit exists to prevent.
func TestAutoContextFitBudgetsVRAMNotSystemRAM(t *testing.T) {
	model := &ModelProfile{
		Path: "small.gguf", TotalSizeMB: 7000, SizeBytes: 7000 * 1024 * 1024,
		NumLayers: 32, ContextSize: 262144, CTXTrain: 262144,
		HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	// One modest GPU, a lot of system RAM: the shape that made fit pick max.
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{
			Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282, VRAMUsedMB: 300,
			BandwidthMBps: 15760, MemBandwidthMBps: 504048,
		}},
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 24},
	}
	ctx, kvType := computeAutoContextSize(caps, model, model.TotalSizeMB, "q8_0", Options{})
	if ctx >= model.CTXTrain {
		t.Errorf("fit chose %d (native is %d): the KV budget still counts system RAM", ctx, model.CTXTrain)
	}
	// Whatever it chose must actually fit the VRAM left after the weights.
	kvMB := computeKVTotalMB(model, ctx, kvType, false)
	if free := caps.GPUs[0].VRAMFreeMB(); model.TotalSizeMB+kvMB > free {
		t.Errorf("fit chose ctx=%d needing %d MiB of KV; only %d MiB free after weights",
			ctx, kvMB, free-model.TotalSizeMB)
	}
}

// With no GPU the KV genuinely lives in host RAM, so CPU-only callers must keep
// budgeting against RAM rather than collapsing to the floor.
func TestAutoContextStillBudgetsRAMWithoutGPUs(t *testing.T) {
	model := &ModelProfile{
		Path: "small.gguf", TotalSizeMB: 7000, SizeBytes: 7000 * 1024 * 1024,
		NumLayers: 32, ContextSize: 262144, CTXTrain: 262144,
		HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 24},
	}
	ctx, _ := computeAutoContextSize(caps, model, model.TotalSizeMB, "q8_0", Options{})
	if ctx <= 32768 {
		t.Errorf("CPU-only fit chose %d; plenty of RAM was available for a larger window", ctx)
	}
}

// A box with enough VRAM for the native window should still get it: the fix
// narrows the budget, it does not cap the context.
func TestAutoContextStillReachesNativeWhenVRAMAllows(t *testing.T) {
	model := &ModelProfile{
		Path: "small.gguf", TotalSizeMB: 4000, SizeBytes: 4000 * 1024 * 1024,
		NumLayers: 32, ContextSize: 65536, CTXTrain: 65536,
		HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 300,
				BandwidthMBps: 15760, MemBandwidthMBps: 1008096},
			{Index: 1, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282, VRAMUsedMB: 300,
				BandwidthMBps: 15760, MemBandwidthMBps: 504048},
		},
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 24},
	}
	if ctx, _ := computeAutoContextSize(caps, model, model.TotalSizeMB, "q8_0", Options{}); ctx != model.CTXTrain {
		t.Errorf("fit chose %d, want the native %d: VRAM was ample", ctx, model.CTXTrain)
	}
}

// Filling the fastest card first makes a bigger window nearly free: at ~210k
// tokens only a few hundred MB spills onto the second card, so decode barely
// moves and the extra context is worth taking. (The 1.78x penalty measured on
// this box came from the OLD balanced split, which put 26% of the weights on the
// slowest cards -- spreading a little is cheap, spreading evenly is not.)
func TestSmallSpillIsCheapSoTheBiggerWindowWins(t *testing.T) {
	caps := benchBox()
	const weightsMB = 16128 // Qwen3.8-27B Q4_K_XL
	const kv210k = 7004
	ratio := spreadDecodeCostRatio(caps, weightsMB, weightsMB+kv210k, 1500)
	t.Logf("spread decode cost at ~210k tokens = %.3fx", ratio)
	if ratio > 1.10 {
		t.Errorf("a few hundred MB of spill cost %.2fx; fill-fastest-first should make it nearly free", ratio)
	}
	if !contextGainOutweighsSpread(caps, weightsMB, 1500, 150016, 4981, 210944, kv210k) {
		t.Errorf("refused 1.41x the window for only a %.2fx slowdown", ratio)
	}
}

// When the fastest card has little room to spare the weights really do land on
// slow memory, and then a modest window increase is not worth it.
func TestEvenSpreadIsRejectedWhenItCostsRealSpeed(t *testing.T) {
	caps := benchBox()
	// Something else already occupies most of the 3090 Ti, so the split has to
	// lean on the 4070 and 3060.
	caps.GPUs[1].VRAMUsedMB = caps.GPUs[1].VRAMTotalMB - 8000
	const weightsMB = 6000
	ratio := spreadDecodeCostRatio(caps, weightsMB, weightsMB+9000, 1500)
	t.Logf("spread decode cost with a crowded fastest card = %.3fx", ratio)
	if ratio <= 1.10 {
		t.Fatalf("expected a real slowdown once weights land on slower cards, got %.2fx", ratio)
	}
	// A 1.1x window is not worth a slowdown larger than that.
	if contextGainOutweighsSpread(caps, weightsMB, 1500, 100000, 5000, 110000, 9000) {
		t.Errorf("took a %.2fx slowdown to gain only 1.1x context", ratio)
	}
}

// With no single-GPU option there is nothing to protect, so the comparison must
// not veto the only plan that fits.
func TestSpreadIsAcceptedWhenTheModelCannotFitOneCard(t *testing.T) {
	caps := benchBox()
	const hugeWeightsMB = 40000 // larger than any single card here
	if ratio := spreadDecodeCostRatio(caps, hugeWeightsMB, hugeWeightsMB+2000, 1500); ratio != 1 {
		t.Errorf("ratio = %.2f, want 1 when no card can hold the model alone", ratio)
	}
	if !contextGainOutweighsSpread(caps, hugeWeightsMB, 1500, 32768, 1000, 65536, 2000) {
		t.Error("rejected the only plan that fits")
	}
}

// Unknown hardware must keep the previous prefer-the-larger-window behaviour
// rather than being vetoed on a comparison that cannot be made.
func TestSpreadComparisonIsNeutralWithoutBandwidthData(t *testing.T) {
	caps := benchBox()
	for i := range caps.GPUs {
		caps.GPUs[i].MemBandwidthMBps = 0
	}
	if ratio := spreadDecodeCostRatio(caps, 16128, 16128+7004, 1500); ratio != 1 {
		t.Errorf("ratio = %.2f, want 1 when VRAM bandwidth is unknown", ratio)
	}
	if !contextGainOutweighsSpread(caps, 16128, 1500, 150016, 4981, 210944, 7004) {
		t.Error("vetoed a larger window on hardware it cannot compare")
	}
}
