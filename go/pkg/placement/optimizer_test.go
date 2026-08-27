package placement

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func optimizerTestGPUs() []detect.GPU {
	return []detect.GPU{
		{Index: 0, Name: "fast", VRAMTotalMB: 32768, MemBandwidthMBps: 1_000_000, BandwidthMBps: 24000},
		{Index: 1, Name: "slow", VRAMTotalMB: 32768, MemBandwidthMBps: 100_000, BandwidthMBps: 8000},
	}
}

func TestMeasuredAllocationMustMatchTensorDistribution(t *testing.T) {
	gpus := optimizerTestGPUs()
	model := &ModelProfile{TotalSizeMB: 10000, SizeBytes: 10000 * 1024 * 1024}
	strategy := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{0.8, 0.2}, KVPlacement: "cpu"}
	matching := MeasuredAllocation{
		ContextTotalMB: 100, ContextHostMB: 100,
		ModelByGPU: map[int]int{0: 8000, 1: 2000},
	}
	if !measuredAllocationMatchesStrategy(matching, model, strategy, gpus) {
		t.Fatal("matching allocation distribution was rejected")
	}
	reversed := matching
	reversed.ModelByGPU = map[int]int{0: 2000, 1: 8000}
	if measuredAllocationMatchesStrategy(reversed, model, strategy, gpus) {
		t.Fatal("allocation from the reversed tensor split became exact evidence")
	}
}

func TestBatchAndParallelFrontiersRespectIntentWithoutFixedPairs(t *testing.T) {
	base := &Strategy{ContextSize: 196608, Parallel: 2, BatchSize: 2048, UBatchSize: 512}
	pairs := calibrationBatchNeighbors(base, Options{BatchSizeExplicit: true})
	seenLargeUBatch := false
	for _, pair := range pairs {
		if pair.batch != base.BatchSize || pair.ubatch > pair.batch {
			t.Fatalf("explicit batch or invariant changed: %+v", pair)
		}
		seenLargeUBatch = seenLargeUBatch || pair.ubatch == 1024
	}
	if !seenLargeUBatch {
		t.Fatal("bounded frontier did not challenge a larger ubatch")
	}
	if got := calibrationBatchNeighbors(base, Options{BatchSizeExplicit: true, UBatchSizeExplicit: true}); len(got) != 0 {
		t.Fatalf("both explicit batch coordinates were challenged: %+v", got)
	}

	parallel := calibrationParallelNeighbors(base, Options{ParallelSlotTarget: 32768})
	seen := map[int]bool{}
	for _, value := range parallel {
		seen[value] = true
	}
	for _, want := range []int{1, 3, 4, 5, 6} {
		if !seen[want] {
			t.Fatalf("legal parallel=%d missing from frontier %v", want, parallel)
		}
	}
}

func TestEqualCostFrontierPreservesNearestGeneratorOrder(t *testing.T) {
	frontier := AnalyzeCandidateFrontier(nil, nil, Options{}, []CalibrationCandidate{
		{Name: "default"},
		{Name: "batch-near", Estimate: CandidateEstimate{AgentCost: 1, Feasible: true}},
		{Name: "batch-lexically-earlier-but-far", Estimate: CandidateEstimate{AgentCost: 1, Feasible: true}},
	})
	if len(frontier) != 3 || frontier[1].Name != "batch-near" {
		t.Fatalf("equal-cost frontier lost generator proximity order: %+v", frontier)
	}
}

func TestMaterialHeadroomChangeIsDirectional(t *testing.T) {
	base := &Strategy{BatchSize: 2048, UBatchSize: 512, Parallel: 2, KVPlacement: "gpu"}
	lowerSlots := cloneStrategy(base)
	lowerSlots.Parallel = 1
	if materialHeadroomChange(base, lowerSlots) {
		t.Fatal("lower-resource slot count was classified as residual headroom")
	}
	largerGraph := cloneStrategy(base)
	largerGraph.UBatchSize = 1024
	if !materialHeadroomChange(base, largerGraph) {
		t.Fatal("larger physical microbatch did not consume headroom")
	}
}

func TestAnalyzeFrontierUsesExactHeadroomAndPersistsBoundary(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	model := &ModelProfile{
		Path: modelPath, TotalSizeMB: 10000, SizeBytes: 10000 * 1024 * 1024,
		NumLayers: 40, EmbeddingLength: 4096,
	}
	caps := &detect.Capabilities{
		GPUs:                    optimizerTestGPUs(),
		RAM:                     detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:                     detect.CPUInfo{Cores: 16, Threads: 32},
		HostMemoryBandwidthMBps: 50000,
	}
	opts := Options{CacheDir: dir, BackendCacheTag: "test", WorkloadProfile: "agent", WorkloadConcurrency: 1}
	base := &Strategy{
		Type: MultiGPUDense, TensorSplit: []float64{0.5, 0.5}, MainGPU: 0,
		ContextSize: 32768, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		KVPlacement: "cpu", KVQuality: "high", KVType: "q8_0",
		PlanFreeVRAM: map[int]int{0: 32768, 1: 32768},
	}
	if err := RecordMeasuredAllocation(
		dir, model, base.ContextSize, base.UBatchSize, base.KVQuality,
		base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel,
		MeasuredAllocation{
			Evidence: "backend-measured", ContextTotalMB: 100, ContextHostMB: 100,
			ModelByGPU: map[int]int{0: 5000, 1: 5000},
		},
	); err != nil {
		t.Fatal(err)
	}
	fastHeavy := cloneStrategy(base)
	fastHeavy.TensorSplit = []float64{0.9, 0.1}
	frontier := AnalyzeCandidateFrontier(caps, model, opts, []CalibrationCandidate{
		{Name: "default", Strategy: base},
		{Name: "fast-heavy", Strategy: fastHeavy},
	})
	if base.ResourceLedger == nil || !base.ResourceLedger.Exact {
		t.Fatalf("baseline did not consume its exact allocation: %+v", base.ResourceLedger)
	}
	if frontier[1].Strategy.ResourceLedger == nil || frontier[1].Strategy.ResourceLedger.Exact {
		t.Fatal("different split reused the baseline's exact allocation")
	}
	if base.Residency != ResidencyRoomy {
		t.Fatalf("exact headroom with a faster feasible neighbor classified as %q", base.Residency)
	}
	if base.OptimizationBoundary == nil || base.OptimizationBoundary.CandidateCount != 2 ||
		base.OptimizationBoundary.FeasibleCount != 2 || base.OptimizationBoundary.ExactCount != 1 {
		t.Fatalf("incomplete explored boundary: %+v", base.OptimizationBoundary)
	}
}

func TestAnalyzeFrontierClassifiesRoomyFromExactBatchHeadroom(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	model := &ModelProfile{
		Path: modelPath, TotalSizeMB: 8000, SizeBytes: 8000 * 1024 * 1024,
		NumLayers: 32, EmbeddingLength: 4096,
	}
	caps := &detect.Capabilities{
		GPUs: optimizerTestGPUs()[:1],
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 16, Threads: 32},
	}
	opts := Options{CacheDir: dir, BackendCacheTag: "test"}
	base := &Strategy{
		Type: SingleGPU, MainGPU: 0, ContextSize: 32768, Parallel: 1,
		BatchSize: 2048, UBatchSize: 512, KVPlacement: "cpu", KVQuality: "high", KVType: "q8_0",
		PlanFreeVRAM: map[int]int{0: 32768},
	}
	if err := RecordMeasuredAllocation(
		dir, model, base.ContextSize, base.UBatchSize, base.KVQuality,
		base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel,
		MeasuredAllocation{
			Evidence: "backend-measured", ContextTotalMB: 100, ContextHostMB: 100,
			ModelByGPU: map[int]int{0: 8000},
		},
	); err != nil {
		t.Fatal(err)
	}
	largerBatch := cloneStrategy(base)
	largerBatch.BatchSize = 4096
	frontier := AnalyzeCandidateFrontier(caps, model, opts, []CalibrationCandidate{
		{Name: "default", Strategy: base},
		{Name: "batch-4096-ubatch-512", Strategy: largerBatch},
	})
	if !frontier[1].Estimate.Feasible {
		t.Fatalf("larger batch did not fit the exact slack: %+v", frontier[1].Estimate)
	}
	if base.Residency != ResidencyRoomy {
		t.Fatalf("exact slack that admits a larger batch classified as %q", base.Residency)
	}
	if !frontier[1].Estimate.ActionableGain {
		t.Fatal("larger batch was not marked as actionable headroom")
	}
}

func TestPackedMoEHasPerformanceHeadroomWhenHostCanAbsorbAnExpertLayer(t *testing.T) {
	model := &ModelProfile{
		TotalSizeMB: 120000, SizeBytes: 120000 * 1024 * 1024,
		NumLayers: 40, IsMoE: true, ExpertBytes: 100000 * 1024 * 1024,
	}
	base := &Strategy{
		Type: MoEOffload, ContextSize: 32768, Parallel: 2, BatchSize: 128, UBatchSize: 64,
		KVPlacement: "gpu", KVQuality: "high", KVType: "f16", NCPUMoE: 34,
		TensorSplit: []float64{0.3, 0.7}, PlacementPolicy: "link",
		ResourceLedger: &ResourceLedger{Exact: true, Fits: true, Host: HostResourceLedger{SlackMB: 30000}},
	}
	alternate := cloneStrategy(base)
	alternate.TensorSplit = []float64{0.2, 0.8}
	alternate.PlacementPolicy = "memory"
	frontier := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true}},
		{Name: "moe-topology-memory", Strategy: alternate, Estimate: CandidateEstimate{Feasible: true}},
	}
	if !moeTopologyExperimentHeadroom(model, base, frontier) {
		t.Fatal("system-roomy MoE was confused with an intentionally packed GPU")
	}
	// The threshold is one layer derived from this model, not a fixed cushion.
	base.ResourceLedger.Host.SlackMB = 2499
	if moeTopologyExperimentHeadroom(model, base, frontier) {
		t.Fatal("genuinely tight host was admitted without one expert layer of fallback room")
	}
}

func TestEstimateStrategyCostPrefersFewerCPUExpertLayers(t *testing.T) {
	model := &ModelProfile{
		TotalSizeMB: 120000, SizeBytes: 120000 * 1024 * 1024,
		NumLayers: 40, IsMoE: true, ExpertBytes: 100000 * 1024 * 1024,
		NumExperts: 256, ExpertUsedCount: 8,
	}
	caps := &detect.Capabilities{
		GPUs: optimizerTestGPUs(), HostMemoryBandwidthMBps: 50000,
	}
	ledger := ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
		{GPU: 0, Active: true, ModelMB: 1000, RequiredMB: 3000, BandwidthMBps: 1_000_000},
		{GPU: 1, Active: true, ModelMB: 1000, RequiredMB: 3000, BandwidthMBps: 100_000},
	}}
	base := &Strategy{
		Type: MoEOffload, Parallel: 2, UBatchSize: 64, NCPUMoE: 36,
		TensorSplit: []float64{0.5, 0.5},
	}
	denser := cloneStrategy(base)
	denser.NCPUMoE = 32
	opts := Options{WorkloadConcurrency: 2, WorkloadProfile: "claude-agent-parallel-v4:test"}
	baseEstimate := EstimateStrategyCost(caps, model, base, opts, ledger)
	denserEstimate := EstimateStrategyCost(caps, model, denser, opts, ledger)
	if baseEstimate.Bottleneck != "CPU expert bandwidth" || denserEstimate.AgentCost >= baseEstimate.AgentCost {
		t.Fatalf("denser GPU expert pack was not preferred: base=%+v denser=%+v", baseEstimate, denserEstimate)
	}
}

func TestTightLiveCandidatesKeepProvenShape(t *testing.T) {
	base := &Strategy{
		Type: MultiGPUDense, MainGPU: 0, TensorSplit: []float64{0.5, 0.5},
		BatchSize: 2048, UBatchSize: 512, Residency: ResidencyTight,
		ResourceLedger: &ResourceLedger{Exact: true, Fits: true},
	}
	same := cloneStrategy(base)
	same.BatchSize = 4096
	same.Residency = ""
	reshuffle := cloneStrategy(base)
	reshuffle.TensorSplit = []float64{0.9, 0.1}
	reshuffle.Residency = ""
	got := TightLiveCandidates([]CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true}},
		{Name: "topology", Strategy: reshuffle, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.5}},
		{Name: "batch-4096", Strategy: same, Estimate: CandidateEstimate{Feasible: true, AgentCost: 1}},
	})
	if len(got) != 2 || got[1].Name != "batch-4096" {
		t.Fatalf("tight live set should keep the proven split: %+v", got)
	}
}

func TestTightLiveCandidatesKeepDenserGPUExpertPack(t *testing.T) {
	base := &Strategy{
		Type: MoEOffload, KVPlacement: "gpu", NCPUMoE: 38,
		TensorSplit:    []float64{0.26, 0.63, 0.11},
		OTString:       "blk.(0|1|2)=CUDA1",
		Residency:      ResidencyTight,
		ResourceLedger: &ResourceLedger{Exact: true, Fits: true},
		VRAMLedger: []GPULedgerEntry{
			{GPU: 1, ExpertLayers: 3}, {GPU: 0, ExpertLayers: 1}, {GPU: 2, ExpertLayers: 1},
		},
	}
	denser := cloneStrategy(base)
	denser.NCPUMoE = 34
	denser.TensorSplit = []float64{0.27, 0.65, 0.09}
	denser.OTString = "blk.(0|1|2|3|4)=CUDA1,blk.(5|6)=CUDA0,blk.(7|8)=CUDA2"
	denser.VRAMLedger = []GPULedgerEntry{
		{GPU: 1, ExpertLayers: 5}, {GPU: 0, ExpertLayers: 2}, {GPU: 2, ExpertLayers: 2},
	}
	mmap := cloneStrategy(denser)
	mmap.MMapRequired = true
	got := TightLiveCandidates([]CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true}},
		{Name: "moe-topology-memory", Strategy: denser, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.8}},
		{Name: "mmap", Strategy: mmap, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.7}},
	})
	if len(got) != 2 || got[1].Name != "moe-topology-memory" {
		t.Fatalf("tight live set dropped the denser GPU expert pack: %+v", got)
	}
}

func TestDeviceBalanceBottleneckFlagsIdleComputeGPU(t *testing.T) {
	s := &Strategy{Type: MultiGPUDense, TensorSplit: []float64{0.27, 0.65, 0.09}}
	got := DeviceBalanceBottleneck(s, []GPUUtilSample{
		{GPU: 0, SMPercent: 4, MemPercent: 80},
		{GPU: 1, SMPercent: 2, MemPercent: 70},
		{GPU: 2, SMPercent: 96, MemPercent: 90},
	})
	if got != "GPU 2 saturated (96% SM) while GPU 1 is idle (2% SM)" {
		t.Fatalf("imbalanced topology bottleneck=%q", got)
	}
	if got := DeviceBalanceBottleneck(s, nil); got != "" {
		t.Fatalf("missing samples must fail closed, got %q", got)
	}
}

func TestSelectDeviceBalanceFinalistRelievesMeasuredBusyGPU(t *testing.T) {
	strategy := func(model0, model1, batch int) *Strategy {
		return &Strategy{
			Type: MultiGPUDense, ContextSize: 32768, Parallel: 2,
			BatchSize: batch, UBatchSize: 128, KVPlacement: "gpu", KVQuality: "mid", KVType: "q8_0",
			ResourceLedger: &ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
				{GPU: 0, ModelMB: model0, Active: true}, {GPU: 1, ModelMB: model1, Active: true},
			}},
		}
	}
	base := strategy(7000, 3000, 256)
	wrongCoordinate := strategy(3000, 7000, 512)
	balanced := strategy(4000, 6000, 256)
	frontier := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true, AgentCost: 1}},
		{Name: "batch-512", Strategy: wrongCoordinate, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.5}},
		{Name: "topology-capacity", Strategy: balanced, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.9}},
	}
	signal := AnalyzeDeviceBalance(base, []GPUUtilSample{{GPU: 0, SMPercent: 96}, {GPU: 1, SMPercent: 3}})
	got, ok := SelectDeviceBalanceFinalist(frontier, signal)
	if !ok || got.Name != "topology-capacity" {
		t.Fatalf("telemetry finalist=%+v ok=%t", got, ok)
	}
}

func TestSelectDeviceBalanceFinalistUsesMoEComputeRoleNotStoredBytes(t *testing.T) {
	strategy := func(split []float64, modelMB []int) *Strategy {
		return &Strategy{
			Type: MoEOffload, ContextSize: 1048576, Parallel: 2,
			BatchSize: 128, UBatchSize: 64, KVPlacement: "gpu", KVQuality: "high", KVType: "bf16",
			NCPUMoE: 36, TensorSplit: split,
			ResourceLedger: &ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
				{GPU: 0, ModelMB: modelMB[0], Active: true},
				{GPU: 1, ModelMB: modelMB[1], Active: true},
				{GPU: 2, ModelMB: modelMB[2], Active: true},
			}},
		}
	}
	base := strategy([]float64{0.23, 0.67, 0.10}, []int{3000, 9000, 1000})
	reweighted := strategy([]float64{0.15, 0.75, 0.10}, []int{2000, 10000, 1000})
	// CUDA0 stores more routed experts in this candidate, but owns no ordinary
	// layers. Sparse expert bytes must not hide removal of its serial backbone.
	owner1 := strategy([]float64{0, 1, 0}, []int{5000, 7000, 1000})
	owner1.PlacementPolicy = "owner-1"
	frontier := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true, AgentCost: 1}},
		{Name: "moe-topology-memory", Strategy: reweighted, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.8}},
		{Name: "moe-owner-1", Strategy: owner1, Estimate: CandidateEstimate{Feasible: true, AgentCost: 0.9}},
	}
	signal := AnalyzeDeviceBalance(base, []GPUUtilSample{
		{GPU: 0, SMPercent: 98}, {GPU: 1, SMPercent: 1}, {GPU: 2, SMPercent: 2},
	})
	got, ok := SelectDeviceBalanceFinalist(frontier, signal)
	if !ok || got.Name != "moe-owner-1" {
		t.Fatalf("role-aware finalist=%+v ok=%t", got, ok)
	}
}

func TestSelectDeviceBalanceFinalistPreservesTightMoEResidency(t *testing.T) {
	strategy := func(model0, model1, cpuMoE int) *Strategy {
		return &Strategy{
			Type: MoEOffload, ContextSize: 32768, Parallel: 1,
			BatchSize: 256, UBatchSize: 128, KVPlacement: "gpu", KVQuality: "mid", KVType: "q8_0",
			NCPUMoE: cpuMoE,
			ResourceLedger: &ResourceLedger{Fits: true, Devices: []DeviceResourceLedger{
				{GPU: 0, ModelMB: model0, Active: true}, {GPU: 1, ModelMB: model1, Active: true},
			}},
		}
	}
	base := strategy(7000, 3000, 30)
	moreCPU := strategy(3000, 7000, 31)
	frontier := []CalibrationCandidate{
		{Name: "default", Strategy: base, Estimate: CandidateEstimate{Feasible: true}},
		{Name: "more-cpu-offload", Strategy: moreCPU, Estimate: CandidateEstimate{Feasible: true}},
	}
	signal := AnalyzeDeviceBalance(base, []GPUUtilSample{{GPU: 0, SMPercent: 90}, {GPU: 1, SMPercent: 4}})
	if got, ok := SelectDeviceBalanceFinalist(frontier, signal); ok {
		t.Fatalf("less-resident topology became a finalist: %+v", got)
	}
}

func TestDenseSubsetSplitRejectsIdleMembersOfMask(t *testing.T) {
	gpus := optimizerTestGPUs()
	free := []int{32768, 32768}
	fixed := []int{1024, 1024}
	if split, ok := denseSubsetSplit(gpus, free, fixed, 8000, 0b11, "critical"); ok {
		t.Fatalf("faster card can hold the model, but a zero-weight second GPU was accepted: %v", split)
	}
	split, ok := denseSubsetSplit(gpus, free, fixed, 40000, 0b11, "critical")
	if !ok || len(split) < 2 || split[0] <= 0 || split[1] <= 0 {
		t.Fatalf("needed both GPUs, got ok=%v split=%v", ok, split)
	}
	zeroCapacity := []int{32768, 1024}
	if split, ok = denseSubsetSplit(gpus, zeroCapacity, fixed, 8000, 0b11, "capacity"); ok {
		t.Fatalf("mask member with no remaining capacity was accepted: %v", split)
	}
}

func TestTopologyRankingPrefersFastSingleGPUWhenItFits(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: optimizerTestGPUs(),
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 16, Threads: 32},
	}
	model := &ModelProfile{Path: "model.gguf", TotalSizeMB: 8000, SizeBytes: 8000 * 1024 * 1024, NumLayers: 32, EmbeddingLength: 4096}
	base := &Strategy{
		Type: MultiGPUDense, TensorSplit: []float64{0.5, 0.5}, MainGPU: 0,
		ContextSize: 32768, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		KVPlacement: "cpu", KVQuality: "high", KVType: "q8_0",
	}
	alternates := denseTopologyCandidates(caps, model, base, Options{})
	frontier := []CalibrationCandidate{{Name: "default", Strategy: base}}
	frontier = append(frontier, alternates...)
	frontier = AnalyzeCandidateFrontier(caps, model, Options{}, frontier)
	if len(frontier) < 3 {
		t.Fatalf("single-device topology candidates missing: %+v", frontier)
	}
	if frontier[1].Name != "single-gpu-0" {
		t.Fatalf("calculated finalist=%q, want fastest fitting single GPU", frontier[1].Name)
	}
}
