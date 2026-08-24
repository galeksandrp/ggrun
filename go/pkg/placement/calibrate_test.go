package placement

import (
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestCalibrationCandidatesSingleGPUOffloadedMoEAddsUBatchLadder(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}}}
	model := &ModelProfile{Path: "m.gguf", IsMoE: true}
	base := &Strategy{Type: MoEOffload, KVPlacement: "cpu", BatchSize: 2048, UBatchSize: 512}
	got := CalibrationCandidates(caps, model, base, Options{})
	if len(got) != 3 || got[0].Name != "default" || got[1].Name != "ubatch-1024" || got[2].Name != "ubatch-2048" {
		t.Fatalf("single-GPU offloaded MoE must measure bounded ubatches, got %+v", got)
	}
}

func TestCalibrationCandidatesCPUOnlyIsNoOp(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288},
	}}
	base := &Strategy{Type: CPUOnly}
	got := CalibrationCandidates(caps, &ModelProfile{Path: "m.gguf"}, base, Options{})
	if len(got) != 1 {
		t.Fatalf("CPU-only must not produce alternatives, got %+v", got)
	}
}

func TestCalibrationCandidatesMoEAddsKVAlternate(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000},
	}}
	model := &ModelProfile{
		Path: "m.gguf", IsMoE: true, TotalSizeMB: 64 * 1024, NumLayers: 60, NumExperts: 128,
		ExpertBytes: 56 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
		ContextSize: 8192, HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	caps.RAM = detect.RAMInfo{TotalMB: 131072, FreeMB: 131072}
	caps.CPU = detect.CPUInfo{Cores: 16}
	base := &Strategy{
		Type: MoEOffload, KVPlacement: "cpu", KVQuality: "mid", KVType: "q8_0",
		ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		NCPUMoE: 40, OTString: "blk.ffn=CPU",
	}
	got := CalibrationCandidates(caps, model, base, Options{ContextSize: 8192, KVPlacement: "cpu", KVQuality: "mid", Parallel: 1})
	if len(got) != 4 {
		t.Fatalf("MoE multi-GPU should offer two ubatch candidates and a KV alternate, got %d candidates: %+v", len(got), got)
	}
	if got[0].Name != "default" || got[1].Name != "ubatch-1024" || got[2].Name != "ubatch-2048" || got[3].Name != "kv-alternate" {
		t.Fatalf("unexpected candidate names: %+v", got)
	}
	alt := got[3].Strategy
	if alt.KVPlacement != "gpu" {
		t.Fatalf("KV alternate should flip cpu->gpu, got %q", alt.KVPlacement)
	}
	// The alternate must not alias the base's expert split.
	if alt == base {
		t.Fatal("candidate aliases the base strategy")
	}
	if base.KVPlacement != "cpu" {
		t.Fatalf("base mutated to %q", base.KVPlacement)
	}
}

func TestCalibrationCandidatesMoEKVGPUFlipsToCPU(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288},
	}}
	caps.RAM = detect.RAMInfo{TotalMB: 131072, FreeMB: 131072}
	caps.CPU = detect.CPUInfo{Cores: 16}
	base := &Strategy{Type: MoEOffload, KVPlacement: "gpu"}
	model := &ModelProfile{
		Path: "m.gguf", IsMoE: true, TotalSizeMB: 64 * 1024, NumLayers: 60, NumExperts: 128,
		ExpertBytes: 56 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
	}
	got := CalibrationCandidates(caps, model, base, Options{ContextSize: 8192, KVPlacement: "gpu"})
	found := false
	for _, candidate := range got {
		if candidate.Name == "kv-alternate" && candidate.Strategy.KVPlacement == "cpu" {
			found = true
		}
	}
	if !found {
		t.Fatalf("KV=gpu should alternate to cpu, got %+v", got)
	}
}

func TestCalibrationCandidatesMoEUBatchRespectsExplicitRequest(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 12288}},
		RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU:  detect.CPUInfo{Cores: 16},
	}
	model := &ModelProfile{
		Path: "m.gguf", IsMoE: true, TotalSizeMB: 64 * 1024, NumLayers: 60, NumExperts: 128,
		ExpertBytes: 56 * 1024 * 1024 * 1024, NonExpertBytes: 8 * 1024 * 1024 * 1024,
		ContextSize: 8192, HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}
	base := &Strategy{
		Type: MoEOffload, KVPlacement: "cpu", KVQuality: "mid", KVType: "q8_0",
		ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
	}
	got := CalibrationCandidates(caps, model, base, Options{
		ContextSize: 8192, KVPlacement: "cpu", KVQuality: "mid", Parallel: 1,
		UBatchSize: 512, UBatchSizeExplicit: true,
	})
	for _, candidate := range got {
		if candidate.Name == "ubatch-1024" || candidate.Name == "ubatch-2048" {
			t.Fatalf("explicit ubatch was challenged by %q", candidate.Name)
		}
	}
}

func TestCalibrationScopeSeparatesExplicitAndAutomaticUBatch(t *testing.T) {
	model := &ModelProfile{Path: "m.gguf"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "gpu", VRAMTotalMB: 24576}}}
	base := &Strategy{ContextSize: 8192, Parallel: 1, BatchSize: 2048, UBatchSize: 512}
	auto := NewCalibrationScopeKey(model, caps, Options{}, base).String()
	explicit := NewCalibrationScopeKey(model, caps, Options{UBatchSizeExplicit: true}, base).String()
	if auto == explicit {
		t.Fatal("automatic and user-explicit ubatch shared a calibration scope")
	}
}

func TestCalibrationCandidatesDenseSplitInversion(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000},
	}}
	base := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{0.75, 0.25}, MainGPU: 0}
	got := CalibrationCandidates(caps, &ModelProfile{Path: "m.gguf"}, base, Options{})
	if len(got) != 2 || got[1].Name != "split-inverted" {
		t.Fatalf("dense multi-GPU should offer a split inversion, got %+v", got)
	}
	inv := got[1].Strategy
	if inv.TensorSplit[0] != 0.25 || inv.TensorSplit[1] != 0.75 {
		t.Fatalf("split not inverted: %v", inv.TensorSplit)
	}
	if inv.MainGPU != 1 {
		t.Fatalf("main GPU should follow the larger share, got %d", inv.MainGPU)
	}
	// Base untouched.
	if base.TensorSplit[0] != 0.75 || base.MainGPU != 0 {
		t.Fatal("base strategy mutated by inversion")
	}
}

func TestCalibrationCandidatesDenseSymmetricSplitSkipped(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576}, {Index: 1, VRAMTotalMB: 24576},
	}}
	base := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{0.5, 0.5}, MainGPU: 0}
	got := CalibrationCandidates(caps, &ModelProfile{Path: "m.gguf"}, base, Options{})
	if len(got) != 1 {
		t.Fatalf("a symmetric split inverts to itself and must be skipped, got %+v", got)
	}
}

func TestCalibrationDecisionRoundTrip(t *testing.T) {
	dir := t.TempDir()
	key := CalibrationScopeKey{
		ModelIdentity: "m", BackendIdentity: "b", HardwareID: "h",
		WorkloadProfile: "claude-agent-parallel-v1", ContextSize: 131072,
		Parallel: 4, UBatchSize: 256, KVQuality: "mid",
	}.String()
	d := CalibrationDecision{
		ScopeKey: key, Winner: "kv-alternate",
		DefaultTPS: 20.5, WinnerTPS: 24.1, Improvement: 17.5,
	}
	if _, err := SaveCalibrationDecision(dir, d); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := LoadCalibrationDecision(dir, key)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.Winner != "kv-alternate" || loaded.WinnerTPS != 24.1 {
		t.Fatalf("round trip lost decision: %+v", loaded)
	}
	// A different scope key must not read this decision.
	if _, err := LoadCalibrationDecision(dir, "other-scope"); err == nil {
		t.Fatal("stale/foreign scope must not load a decision")
	}
	// The file lives under the calibration namespace.
	if filepath.Dir(CalibrationPath(dir, key)) != filepath.Join(dir, "calibration") {
		t.Fatalf("unexpected calibration path %q", CalibrationPath(dir, key))
	}
}

func TestDeleteCalibrationDecisionsForModel(t *testing.T) {
	dir := t.TempDir()
	for _, decision := range []CalibrationDecision{
		{ScopeKey: "scope-a", ModelBasename: "a.gguf", Winner: "ubatch-1024"},
		{ScopeKey: "scope-b", ModelBasename: "b.gguf", Winner: "default"},
	} {
		if _, err := SaveCalibrationDecision(dir, decision); err != nil {
			t.Fatalf("save %s: %v", decision.ScopeKey, err)
		}
	}
	if err := DeleteCalibrationDecisionsForModel(dir, &ModelProfile{Path: "/models/a.gguf"}); err != nil {
		t.Fatalf("delete model decisions: %v", err)
	}
	if _, err := LoadCalibrationDecision(dir, "scope-a"); err == nil {
		t.Fatal("target model calibration decision survived deletion")
	}
	if _, err := LoadCalibrationDecision(dir, "scope-b"); err != nil {
		t.Fatalf("other model calibration decision was removed: %v", err)
	}
}
