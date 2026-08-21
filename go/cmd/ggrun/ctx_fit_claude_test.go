package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func fitTestModel(ctxTrain, sizeMB int) *placement.ModelProfile {
	return &placement.ModelProfile{
		Path: "m.gguf", TotalSizeMB: sizeMB, SizeBytes: int64(sizeMB) * 1024 * 1024,
		NumLayers: 32, ContextSize: ctxTrain, CTXTrain: ctxTrain,
		HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
}

// placementOptionsFromRequestCaps reads backend identity fields, so tests supply
// a minimal real backend rather than nil.
func fitTestBackend() *backendInfo {
	return &backendInfo{Path: "/usr/bin/llama-server", Dialect: "llama", Tag: "llama", Identity: "test"}
}

func fitTestCaps(vramMB int) *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: []detect.GPU{{
			Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: vramMB, VRAMUsedMB: 300,
			BandwidthMBps: 15760, MemBandwidthMBps: 504048,
		}},
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 24},
	}
}

// Claude Code's default context used to be the model's native window outright,
// so "fit" meant "max" and a KV cache far too large for VRAM was planned and
// then offloaded to the host. The default must be bounded by what fits.
func TestClaudeCodeDefaultContextIsBoundedByVRAM(t *testing.T) {
	model := fitTestModel(262144, 7000)
	caps := fitTestCaps(12282) // no room for a 262144 KV alongside 7 GB of weights
	req := &launchRequest{ClaudeCode: true, CtxFlag: "fit"}

	opts := placementOptionsFromRequestCaps(req, model, fitTestBackend(), t.TempDir(), caps)
	if opts.ContextSize >= model.CTXTrain {
		t.Errorf("Claude Code fit chose %d (native %d); it ignored the VRAM budget",
			opts.ContextSize, model.CTXTrain)
	}
	if opts.ContextSize <= 0 {
		t.Fatalf("no context chosen: %d", opts.ContextSize)
	}
	// The exact size that fits is asserted in the placement package; here the
	// contract is that the Claude Code default consults it at all.
	if fit, _ := placement.AutoContextFitVRAM(caps, model, model.TotalSizeMB, "", placement.Options{}); opts.ContextSize > fit {
		t.Errorf("chose ctx=%d above the VRAM fit of %d", opts.ContextSize, fit)
	}
}

// Ample VRAM must still get the large window Claude Code wants: the bound is a
// ceiling, not a cap.
func TestClaudeCodeDefaultContextKeepsNativeWhenVRAMAllows(t *testing.T) {
	model := fitTestModel(65536, 4000)
	caps := fitTestCaps(24564)
	req := &launchRequest{ClaudeCode: true, CtxFlag: "fit"}

	opts := placementOptionsFromRequestCaps(req, model, fitTestBackend(), t.TempDir(), caps)
	if opts.ContextSize != model.CTXTrain {
		t.Errorf("context = %d, want the native %d; VRAM was ample", opts.ContextSize, model.CTXTrain)
	}
}

// An explicit context is the user's call and must never be lowered, however
// badly it fits -- the launcher warns instead.
func TestExplicitContextIsNeverLowered(t *testing.T) {
	model := fitTestModel(262144, 7000)
	caps := fitTestCaps(12282)
	req := &launchRequest{ClaudeCode: true, CtxFlag: "262144"}

	opts := placementOptionsFromRequestCaps(req, model, fitTestBackend(), t.TempDir(), caps)
	if opts.ContextSize != 262144 {
		t.Errorf("explicit --ctx 262144 became %d; a user override must stand", opts.ContextSize)
	}
}

// "max" is an explicit choice too, and must survive intact.
func TestExplicitMaxContextIsNeverLowered(t *testing.T) {
	model := fitTestModel(262144, 7000)
	caps := fitTestCaps(12282)
	req := &launchRequest{ClaudeCode: true, CtxFlag: "max"}

	opts := placementOptionsFromRequestCaps(req, model, fitTestBackend(), t.TempDir(), caps)
	if opts.ContextSize != model.CTXTrain {
		t.Errorf("--ctx max became %d, want the native %d", opts.ContextSize, model.CTXTrain)
	}
}
