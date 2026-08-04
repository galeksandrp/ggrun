package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func preflightRecoveryFixture() (*placement.ModelProfile, *detect.Capabilities, []string, *placement.Strategy) {
	model := &placement.ModelProfile{
		NumLayers:    60,
		LeadingDense: 3,
		ExpertBytes:  int64(57 * 2500 * 1024 * 1024),
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24576},
		{Index: 1, VRAMTotalMB: 12288},
	}}
	ot := `blk\.(3|4|5|6)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,blk\.(7|8|9|10)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA1,exps=CPU`
	args := []string{
		"llama-server", "-b", "2048", "-ub", "512",
		"--tensor-split", "0.67,0.33", "--split-mode", "layer",
		"-ot", ot, "--n-cpu-moe", "49",
	}
	strategy := &placement.Strategy{
		BatchSize: 2048, UBatchSize: 512, TensorSplit: []float64{0.67, 0.33},
		SplitMode: "layer", OTString: ot, NCPUMoE: 49,
	}
	return model, caps, args, strategy
}

func TestAllocationOOMOutcomePreservesExactRecoveryEvidence(t *testing.T) {
	got := allocationOOMOutcome(preflightOutcome{Device: -1}, &ikAllocationOOMError{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !got.DoesNotFit || got.Device != 0 || got.AllocMB != 2132 || got.DeficitMB != 74 || !got.IsComputeBuffer {
		t.Fatalf("allocation outcome lost exact evidence: %+v", got)
	}
}

func TestUnchangedComputeReplanMovesExpertBeforeLoweringUBatch(t *testing.T) {
	model, caps, args, unchanged := preflightRecoveryFixture()
	next, entry, method, ok := selectChangedPreflightRecovery(args, unchanged, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !ok || method != "expert-derate" {
		t.Fatalf("compute recovery = method %q ok=%v", method, ok)
	}
	fingerprint := effectiveMemoryArgsFingerprint(next)
	if entry == nil || entry.NCPUMoE != 50 || !strings.Contains(fingerprint, "n-cpu-moe=50") || !strings.Contains(fingerprint, "ubatch=512") {
		t.Fatalf("compute recovery did not move one expert while preserving ubatch: entry=%+v args=%v", entry, next)
	}
}

func TestComputeRecoveryNeverRaisesDeratedUBatch(t *testing.T) {
	model, caps, args, candidate := preflightRecoveryFixture()
	args = replaceUBatchArg(args, 256)
	candidate.UBatchSize = 512
	candidate.NCPUMoE++
	next, entry, method, ok := selectChangedPreflightRecovery(args, candidate, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !ok || method != "expert-derate" || entry == nil || entry.NCPUMoE != 50 || !strings.Contains(effectiveMemoryArgsFingerprint(next), "ubatch=256") {
		t.Fatalf("recovery raised a derated ubatch: method=%q entry=%+v ok=%v args=%v", method, entry, ok, next)
	}
}

func TestComputeRecoveryLowersUBatchWhenFailedDeviceHasNoExpert(t *testing.T) {
	model, caps, args, _ := preflightRecoveryFixture()
	for i := range args {
		if args[i] == "-ot" && i+1 < len(args) {
			args[i+1] = strings.ReplaceAll(args[i+1], "CUDA1", "CUDA0")
			break
		}
	}
	next, entry, method, ok := selectChangedPreflightRecovery(args, nil, model, caps, preflightOutcome{
		Device: 1, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !ok || method != "ubatch-derate" || entry == nil || entry.UBatchSize != 256 {
		t.Fatalf("compute fallback = method %q entry=%+v ok=%v args=%v", method, entry, ok, next)
	}
}

func TestUnchangedWeightReplanMovesExpertLayer(t *testing.T) {
	model, caps, args, unchanged := preflightRecoveryFixture()
	next, entry, method, ok := selectChangedPreflightRecovery(args, unchanged, model, caps, preflightOutcome{
		Device: 1, AllocMB: 2500, DeficitMB: 74,
	})
	if !ok || method != "expert-derate" {
		t.Fatalf("weight recovery = method %q ok=%v", method, ok)
	}
	if entry == nil || entry.NCPUMoE != 50 || !strings.Contains(effectiveMemoryArgsFingerprint(next), "n-cpu-moe=50") {
		t.Fatalf("weight recovery did not move one expert layer: entry=%+v args=%v", entry, next)
	}
}

func TestComputeRecoveryMovesExpertBeforeChangedPackerLowersUBatch(t *testing.T) {
	model, caps, args, candidate := preflightRecoveryFixture()
	candidate.UBatchSize = 256
	next, entry, method, ok := selectChangedPreflightRecovery(args, candidate, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if !ok || method != "expert-derate" || entry == nil || entry.NCPUMoE != 50 {
		t.Fatalf("layer-first compute recovery = method %q entry=%+v ok=%v", method, entry, ok)
	}
	if !strings.Contains(effectiveMemoryArgsFingerprint(next), "ubatch=512") {
		t.Fatalf("compute recovery lowered ubatch before moving an expert: %v", next)
	}
}

func TestSharedMemoryRecoveryAppliesLayerFirstStrategyAndArgsTogether(t *testing.T) {
	model, caps, args, current := preflightRecoveryFixture()
	candidate := *current
	candidate.UBatchSize = 256

	next, nextArgs, method, ok := applyMemoryRecoverySelection(
		nil, current, args, &candidate, model, caps,
		preflightOutcome{Device: 0, AllocMB: 2132, DeficitMB: 1, IsComputeBuffer: true},
	)
	if !ok || method != "expert-derate" || next != current {
		t.Fatalf("shared recovery = strategy %p current %p method %q ok=%v", next, current, method, ok)
	}
	fingerprint := effectiveMemoryArgsFingerprint(nextArgs)
	if next.NCPUMoE != 50 || next.UBatchSize != 512 ||
		!strings.Contains(fingerprint, "n-cpu-moe=50") || !strings.Contains(fingerprint, "ubatch=512") {
		t.Fatalf("strategy/argv recovery drifted: strategy=%+v args=%v", next, nextArgs)
	}
}

func TestEffectiveDuplicateOverrideForbidsIdenticalRetry(t *testing.T) {
	model, caps, args, candidate := preflightRecoveryFixture()
	args = replaceUBatchArg(args, 64)
	for i := range args {
		if args[i] == "-ot" && i+1 < len(args) {
			args[i+1] = strings.ReplaceAll(args[i+1], "CUDA0", "CUDA1")
			break
		}
	}
	args = append(args, "-ub", "64") // later user value remains authoritative
	candidate.UBatchSize = 256
	_, _, _, ok := selectChangedPreflightRecovery(args, candidate, model, caps, preflightOutcome{
		Device: 0, AllocMB: 2132, DeficitMB: 74, IsComputeBuffer: true,
	})
	if ok {
		t.Fatal("recovery must reject a syntactically changed but effectively identical retry")
	}
}

func TestNoGenericWeightLeverFailsClosed(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 32}
	args := []string{"llama-server", "-ub", "512"}
	_, _, _, ok := selectChangedPreflightRecovery(args, nil, model, &detect.Capabilities{}, preflightOutcome{
		Device: 0, AllocMB: 1024, DeficitMB: 1,
	})
	if ok {
		t.Fatal("weight OOM without a placement lever must fail closed")
	}
}

// A ggrun-generated --swa-full is withdrawn before any weight is shed. It is
// the largest reclaimable block on a memory failure (measured 2026-08-03:
// 6196 MiB of KV on CUDA0 versus 871 MiB without) and it was buying no prefix
// reuse on this model. A --swa-full the user typed is an instruction and stays.
func TestGeneratedSWAFullIsWithdrawnBeforeSheddingWeights(t *testing.T) {
	baseArgs := []string{
		"llama-server", "-m", "model.gguf", "-ub", "512", "--swa-full", "--n-cpu-moe", "39",
	}
	model := &placement.ModelProfile{ModelArch: "deepseek4", IsMoE: true, NumLayers: 43}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24564}}}
	outcome := preflightOutcome{Device: 0, AllocMB: 9307, AllocMBMeasured: true, DeficitMB: 1200, IsComputeBuffer: true}

	// Config-sourced: ExtraArgs carries it, OriginalArgs does not.
	generated := &launchRequest{ExtraArgs: []string{"--swa-full"}}
	_, nextArgs, method, ok := applyMemoryRecoverySelection(
		generated, &placement.Strategy{UBatchSize: 512, NCPUMoE: 39}, baseArgs, nil, model, caps, outcome)
	if !ok || method != "swa-full-withdrawn" {
		t.Fatalf("generated --swa-full was not withdrawn first: method=%q ok=%v", method, ok)
	}
	if hasArg(nextArgs, "--swa-full") {
		t.Fatalf("--swa-full survived withdrawal: %v", nextArgs)
	}
	if hasArg(generated.ExtraArgs, "--swa-full") {
		t.Fatalf("--swa-full must also leave the request so rebuilds cannot reintroduce it: %v", generated.ExtraArgs)
	}

	// Typed on the command line: never withdrawn, fall through to the ladder.
	typed := &launchRequest{ExtraArgs: []string{"--swa-full"}, OriginalArgs: []string{"--swa-full"}}
	_, _, method, _ = applyMemoryRecoverySelection(
		typed, &placement.Strategy{UBatchSize: 512, NCPUMoE: 39}, baseArgs, nil, model, caps, outcome)
	if method == "swa-full-withdrawn" {
		t.Fatal("an explicitly typed --swa-full must not be withdrawn")
	}
}
