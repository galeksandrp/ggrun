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
