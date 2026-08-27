package placement

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// ResidencyClass is an internal state of the one standard-launch controller.
// It deliberately does not expose separate product modes: every launch first
// proves fit, then spends only actionable headroom on measured performance.
type ResidencyClass string

const (
	ResidencyRoomy       ResidencyClass = "roomy-resident"
	ResidencyTight       ResidencyClass = "tight-resident"
	ResidencyNonResident ResidencyClass = "non-resident"
)

// DeviceResourceLedger is one device's complete placement account. Estimated
// components are useful for screening; an Exact ledger is populated only when
// the backend reported the allocation for this exact runtime signature.
type DeviceResourceLedger struct {
	GPU           int    `json:"gpu"`
	Role          string `json:"role,omitempty"`
	FreeMB        int    `json:"free_mb"`
	ModelMB       int    `json:"model_mb,omitempty"`
	ContextMB     int    `json:"context_mb,omitempty"`
	GraphMB       int    `json:"graph_mb,omitempty"`
	RuntimeMB     int    `json:"runtime_growth_mb,omitempty"`
	RequiredMB    int    `json:"required_mb"`
	SlackMB       int    `json:"slack_mb"`
	Active        bool   `json:"active"`
	Evidence      string `json:"evidence,omitempty"`
	BandwidthMBps int    `json:"bandwidth_mbps,omitempty"`
}

// HostResourceLedger is the host half of the same placement account.
type HostResourceLedger struct {
	FreeMB      int    `json:"free_mb"`
	ModelMB     int    `json:"model_mb,omitempty"`
	ContextMB   int    `json:"context_mb,omitempty"`
	RuntimeMB   int    `json:"runtime_mb,omitempty"`
	RequiredMB  int    `json:"required_mb"`
	SlackMB     int    `json:"slack_mb"`
	Reclaimable int    `json:"reclaimable_mb,omitempty"`
	Evidence    string `json:"evidence,omitempty"`
}

// ResourceLedger keeps memory feasibility independent from the performance
// model. Exact means every active device was backed by observed backend
// allocation rows; otherwise the ledger is a conservative GGUF/probe estimate
// which still has to pass contained admission before launch.
type ResourceLedger struct {
	Devices  []DeviceResourceLedger `json:"devices,omitempty"`
	Host     HostResourceLedger     `json:"host"`
	Exact    bool                   `json:"exact"`
	Evidence string                 `json:"evidence"`
	Fits     bool                   `json:"fits"`
}

// CandidateEstimate predicts only enough to order a bounded finalist set. It
// is intentionally not a benchmark result and cannot make a candidate eligible
// for automatic promotion.
type CandidateEstimate struct {
	AgentCost      float64 `json:"agent_cost"`
	DecodeCost     float64 `json:"decode_cost"`
	PrefillCost    float64 `json:"prefill_cost"`
	TransferCost   float64 `json:"transfer_cost,omitempty"`
	ActiveGPUs     int     `json:"active_gpus"`
	Confidence     string  `json:"confidence"`
	Bottleneck     string  `json:"bottleneck"`
	Feasible       bool    `json:"feasible"`
	ActionableGain bool    `json:"actionable_gain"`
}

// OptimizationBoundary is the finite calculated region explored before one
// candidate is selected for a live comparison. Persisting it makes a
// baseline-won result meaningful: later launches can see what was ruled out
// instead of repeating an opaque search.
type OptimizationBoundary struct {
	CandidateCount int         `json:"candidate_count"`
	FeasibleCount  int         `json:"feasible_count"`
	ExactCount     int         `json:"exact_count,omitempty"`
	MinBatch       int         `json:"min_batch,omitempty"`
	MaxBatch       int         `json:"max_batch,omitempty"`
	MinUBatch      int         `json:"min_ubatch,omitempty"`
	MaxUBatch      int         `json:"max_ubatch,omitempty"`
	MinParallel    int         `json:"min_parallel,omitempty"`
	MaxParallel    int         `json:"max_parallel,omitempty"`
	Topologies     []string    `json:"topologies,omitempty"`
	DeviceSlackMB  map[int]int `json:"device_slack_mb,omitempty"`
	HostSlackMB    int         `json:"host_slack_mb"`
	Evidence       string      `json:"evidence,omitempty"`
}

// AnalyzeStrategy builds the common resource ledger and a topology-aware
// relative cost estimate for one complete strategy. It mutates only diagnostic
// fields on s; emitted backend arguments are unchanged.
func AnalyzeStrategy(caps *detect.Capabilities, model *ModelProfile, s *Strategy, opts Options) CandidateEstimate {
	if s == nil {
		return CandidateEstimate{Confidence: "unknown", Bottleneck: "no strategy"}
	}
	ledger := BuildResourceLedger(caps, model, s, opts)
	s.ResourceLedger = &ledger
	estimate := EstimateStrategyCost(caps, model, s, opts, ledger)
	s.EstimatedAgentCost = estimate.AgentCost
	s.EstimateConfidence = estimate.Confidence
	s.OptimizationBottleneck = estimate.Bottleneck
	return estimate
}

// BuildResourceLedger prices one exact resolved launch. When the backend has
// already reported model/context/unaccounted rows for this exact signature,
// those rows are authoritative. Missing rows fall back to GGUF tensor shares
// plus measured-or-planned graph reserves and remain explicitly non-exact.
func BuildResourceLedger(caps *detect.Capabilities, model *ModelProfile, s *Strategy, opts Options) ResourceLedger {
	ledger := ResourceLedger{Evidence: "gguf-and-probe-estimate", Fits: true}
	if caps == nil || model == nil || s == nil {
		ledger.Fits = false
		ledger.Evidence = "incomplete-input"
		return ledger
	}

	runtimeCaps, err := restrictGPUs(caps, opts.GPUs)
	if err != nil || runtimeCaps == nil {
		ledger.Fits = false
		ledger.Evidence = "invalid-device-set"
		return ledger
	}
	gpus := runtimeCaps.GPUs
	ledger.Devices = make([]DeviceResourceLedger, len(gpus))
	for i, gpu := range gpus {
		free := gpu.VRAMFreeMB()
		if planned, ok := s.PlanFreeVRAM[gpu.Index]; ok && planned >= 0 {
			free = planned
		}
		ledger.Devices[i] = DeviceResourceLedger{
			GPU: gpu.Index, FreeMB: free, SlackMB: free,
			BandwidthMBps: effectiveMemoryBandwidth(gpu), Evidence: ledger.Evidence,
		}
	}

	backendTag := backendCacheTag(opts)
	allocation, allocationOK := LoadMeasuredAllocation(
		opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality,
		s.KVPlacement, backendTag, gpus, s.Parallel,
	)
	if allocationOK && observedAllocationEvidence(allocation.Evidence) &&
		measuredAllocationMatchesStrategy(allocation, model, s, gpus) {
		complete := true
		for i, gpu := range gpus {
			modelMB := allocation.ModelByGPU[gpu.Index]
			contextMB := allocation.ContextByGPU[gpu.Index]
			graphMB := allocation.UnaccountedByGPU[gpu.Index]
			required := modelMB + contextMB + graphMB
			active := strategyUsesGPUAt(s, i, gpu.Index)
			if active && required <= 0 {
				complete = false
				break
			}
			ledger.Devices[i].ModelMB = modelMB
			ledger.Devices[i].ContextMB = contextMB
			ledger.Devices[i].GraphMB = graphMB
			ledger.Devices[i].RequiredMB = required
			ledger.Devices[i].SlackMB = ledger.Devices[i].FreeMB - required
			ledger.Devices[i].Active = active || required > 0
			ledger.Devices[i].Role = strategyGPURoleAt(s, i, gpu.Index)
			ledger.Devices[i].Evidence = allocation.Evidence
			if ledger.Devices[i].SlackMB < 0 {
				ledger.Fits = false
			}
		}
		if complete {
			ledger.Exact = true
			ledger.Evidence = allocation.Evidence
			ledger.Host = HostResourceLedger{
				FreeMB:  runtimeCaps.RAM.FreeMB,
				ModelMB: allocation.ModelHostMB, ContextMB: allocation.ContextHostMB,
				RuntimeMB:   allocation.UnaccountedHostMB,
				RequiredMB:  allocation.ModelHostMB + allocation.ContextHostMB + allocation.UnaccountedHostMB,
				Reclaimable: s.ReclaimableHostWeightsMB, Evidence: allocation.Evidence,
			}
			ledger.Host.SlackMB = ledger.Host.FreeMB - ledger.Host.RequiredMB
			if ledger.Host.SlackMB < 0 && !s.MMapRequired {
				ledger.Fits = false
			}
			return ledger
		}
	}

	totalSizeMB := model.TotalSizeMB
	if totalSizeMB <= 0 {
		totalSizeMB = int((model.SizeBytes + 1048575) / 1048576)
	}
	kvMB := computeKVTotalMB(model, s.ContextSize, s.KVType, s.SWAFull)
	if s.ContextAllocationMB > 0 {
		kvMB = s.ContextAllocationMB
	}
	overhead := SystemCUDAOverheadByGPU(opts.CacheDir, gpus)
	pc := opts.loadProbeCacheForStrategy(model, s, gpus)
	modelShares := estimatedModelShares(model, s, gpus, totalSizeMB)
	contextShares := estimatedContextShares(s, gpus, kvMB)
	order := orderGPUsByBandwidth(gpus)
	for i, gpu := range gpus {
		computeMB := 0
		runtimeMB := 0
		if pc != nil {
			computeMB = pc.ComputeBufByGPU[gpu.Index]
			if computeMB <= 0 && strategyUsesGPUAt(s, i, gpu.Index) {
				computeMB = pc.ComputeBufMB
			}
			runtimeMB = pc.RuntimeGraphGrowthByGPU[gpu.Index]
		}
		if computeMB <= 0 && strategyUsesGPUAt(s, i, gpu.Index) && !opts.RequireMeasuredBuffers {
			computeMB = firstLaunchComputeBufMBForGPUParallelAtContext(
				model, s.UBatchSize, max(1, s.Parallel), s.ContextSize, i, order,
			)
		}
		modelMB := modelShares[i]
		contextMB := contextShares[i]
		graphMB := overhead[gpu.Index] + computeMB
		required := modelMB + contextMB + graphMB + runtimeMB
		ledger.Devices[i].ModelMB = modelMB
		ledger.Devices[i].ContextMB = contextMB
		ledger.Devices[i].GraphMB = graphMB
		ledger.Devices[i].RuntimeMB = runtimeMB
		ledger.Devices[i].RequiredMB = required
		ledger.Devices[i].SlackMB = ledger.Devices[i].FreeMB - required
		ledger.Devices[i].Active = strategyUsesGPUAt(s, i, gpu.Index) || required > 0
		ledger.Devices[i].Role = strategyGPURoleAt(s, i, gpu.Index)
		if ledger.Devices[i].SlackMB < 0 {
			ledger.Fits = false
		}
	}

	hostRequired := s.PlannedHostFootprintMB
	if hostRequired <= 0 {
		hostRequired = plannedRAMRuntimeOverheadMB(runtimeCaps, model, s.UBatchSize, totalSizeMB, opts)
		if s.Type == CPUOnly || s.Type == DenseCPUOffload {
			hostRequired += totalSizeMB
		}
	}
	ledger.Host = HostResourceLedger{
		FreeMB: runtimeCaps.RAM.FreeMB, RequiredMB: hostRequired,
		RuntimeMB: hostRequired, Reclaimable: s.ReclaimableHostWeightsMB,
		Evidence: ledger.Evidence,
	}
	ledger.Host.SlackMB = ledger.Host.FreeMB - hostRequired
	if ledger.Host.SlackMB < 0 && !s.MMapRequired {
		ledger.Fits = false
	}
	return ledger
}

// measuredAllocationMatchesStrategy prevents an allocation observed for one
// tensor placement from becoming "exact" evidence for a different topology.
// Probe-cache keys intentionally let logical batch share a physical graph, but
// older keys do not encode tensor split. Distribution matching closes that gap
// while preserving reuse for batch-only candidates.
func measuredAllocationMatchesStrategy(allocation MeasuredAllocation, model *ModelProfile, s *Strategy, gpus []detect.GPU) bool {
	totalSizeMB := 0
	if model != nil {
		totalSizeMB = model.TotalSizeMB
		if totalSizeMB <= 0 {
			totalSizeMB = int((model.SizeBytes + 1048575) / 1048576)
		}
	}
	expectedModel := estimatedModelShares(model, s, gpus, totalSizeMB)
	if !allocationDistributionCompatible(expectedModel, allocation.ModelByGPU, gpus) {
		return false
	}

	expectedContextTotal := 0
	if s != nil && s.KVPlacement != "cpu" {
		expectedContextTotal = allocation.ContextTotalMB
		if expectedContextTotal <= 0 && model != nil {
			expectedContextTotal = computeKVTotalMB(model, s.ContextSize, s.KVType, s.SWAFull)
		}
	}
	expectedContext := estimatedContextShares(s, gpus, expectedContextTotal)
	return allocationDistributionCompatible(expectedContext, allocation.ContextByGPU, gpus)
}

func allocationDistributionCompatible(expected []int, actual map[int]int, gpus []detect.GPU) bool {
	expectedTotal, actualTotal := 0, 0
	for _, value := range expected {
		expectedTotal += max(0, value)
	}
	for _, gpu := range gpus {
		actualTotal += max(0, actual[gpu.Index])
	}
	if expectedTotal <= 0 {
		return actualTotal <= 64
	}
	if actualTotal <= 0 {
		return false
	}
	// Layer quantization, duplicated output tensors, and backend bookkeeping can
	// legitimately skew the GGUF byte estimate. A quarter of the observed total
	// tolerates those effects but still rejects a materially different owner or
	// device subset (for example 80/20 evidence applied to 20/80).
	tolerance := max(128, actualTotal/4)
	for i, gpu := range gpus {
		scaledExpected := expected[i] * actualTotal / expectedTotal
		if absInt(actual[gpu.Index]-scaledExpected) > tolerance {
			return false
		}
	}
	return true
}

func estimatedModelShares(model *ModelProfile, s *Strategy, gpus []detect.GPU, totalMB int) []int {
	n := len(gpus)
	out := make([]int, n)
	if n == 0 || s == nil {
		return out
	}
	switch s.Type {
	case CPUOnly, DenseCPUOffload:
		return out
	case SingleGPU:
		if ordinal := gpuOrdinal(gpus, s.MainGPU); ordinal >= 0 {
			out[ordinal] = totalMB
		}
		return out
	}
	shares := normalizedStrategySplit(s, n)
	for i := range out {
		out[i] = int(math.Ceil(float64(totalMB) * shares[i]))
	}
	if s.Type == MoEOffload && model != nil && model.ExpertBytes > 0 {
		// TensorSplit owns the backbone, while explicit expert pins own only a
		// subset of routed experts.
		nonExpert := bytesToMiBCeil(model.NonExpertBytes - model.TokenEmbdBytes)
		if nonExpert < 0 {
			nonExpert = 0
		}
		for i := range out {
			out[i] = int(math.Ceil(float64(nonExpert) * shares[i]))
		}
		moeLayers := max(1, model.NumLayers-max(0, model.LeadingDense))
		perLayer := ceilDivInt(bytesToMiBCeil(model.ExpertBytes-model.ShexpBytes), moeLayers)
		for _, entry := range s.VRAMLedger {
			if ordinal := gpuOrdinal(gpus, entry.GPU); ordinal >= 0 && entry.ExpertLayers > 0 {
				out[ordinal] += entry.ExpertLayers * perLayer
			}
		}
	}
	return out
}

func estimatedContextShares(s *Strategy, gpus []detect.GPU, totalMB int) []int {
	n := len(gpus)
	out := make([]int, n)
	if s == nil || n == 0 || totalMB <= 0 || s.KVPlacement == "cpu" || s.Type == CPUOnly {
		return out
	}
	if s.Type == SingleGPU {
		if ordinal := gpuOrdinal(gpus, s.MainGPU); ordinal >= 0 {
			out[ordinal] = totalMB
		}
		return out
	}
	shares := normalizedStrategySplit(s, n)
	for i := range out {
		out[i] = int(math.Ceil(float64(totalMB) * shares[i]))
	}
	return out
}

func normalizedStrategySplit(s *Strategy, n int) []float64 {
	out := make([]float64, n)
	if s == nil || n == 0 {
		return out
	}
	for i := 0; i < n && i < len(s.TensorSplit); i++ {
		if s.TensorSplit[i] > 0 {
			out[i] = s.TensorSplit[i]
		}
	}
	total := 0.0
	for _, value := range out {
		total += value
	}
	if total <= 0 {
		for i := range out {
			out[i] = 1 / float64(n)
		}
		return out
	}
	for i := range out {
		out[i] /= total
	}
	return out
}

func strategyUsesGPUAt(s *Strategy, ordinal, gpu int) bool {
	if s == nil || ordinal < 0 || gpu < 0 {
		return false
	}
	for _, entry := range s.VRAMLedger {
		if entry.GPU == gpu && entry.ExpertLayers > 0 {
			return true
		}
	}
	switch s.Type {
	case CPUOnly:
		return false
	case SingleGPU:
		return s.MainGPU == gpu
	default:
		if ordinal < len(s.TensorSplit) {
			return s.TensorSplit[ordinal] > 0
		}
		return true
	}
}

func strategyGPURoleAt(s *Strategy, ordinal, gpu int) string {
	if s == nil {
		return "unused"
	}
	if !strategyUsesGPUAt(s, ordinal, gpu) {
		for _, entry := range s.VRAMLedger {
			if entry.GPU == gpu && entry.ExpertLayers > 0 {
				return "experts"
			}
		}
		return "unused"
	}
	if s.Type == SingleGPU {
		return "model"
	}
	for _, entry := range s.VRAMLedger {
		if entry.GPU == gpu && entry.ExpertOnly {
			return "experts"
		}
	}
	if s.MainGPU == gpu {
		return "main/layers"
	}
	return "layers"
}

func gpuOrdinal(gpus []detect.GPU, gpu int) int {
	for i := range gpus {
		if gpus[i].Index == gpu {
			return i
		}
	}
	return -1
}

func effectiveMemoryBandwidth(gpu detect.GPU) int {
	if gpu.MemBandwidthMBps > 0 {
		return gpu.MemBandwidthMBps
	}
	return gpu.BandwidthMBps
}

// EstimateStrategyCost estimates the critical path from assigned bytes and the
// measured hardware ceilings. It chooses a finalist only; the live agent
// workload remains the sole performance authority.
func EstimateStrategyCost(caps *detect.Capabilities, model *ModelProfile, s *Strategy, opts Options, ledger ResourceLedger) CandidateEstimate {
	est := CandidateEstimate{Feasible: ledger.Fits, Confidence: "derived"}
	if caps == nil || model == nil || s == nil || !ledger.Fits {
		est.Confidence = "unknown"
		est.Bottleneck = "memory admission"
		est.AgentCost = math.Inf(1)
		return est
	}
	byGPU := make(map[int]detect.GPU, len(caps.GPUs))
	for _, gpu := range caps.GPUs {
		byGPU[gpu.Index] = gpu
	}
	maxDeviceCost := 0.0
	for _, entry := range ledger.Devices {
		if !entry.Active || entry.RequiredMB <= 0 {
			continue
		}
		est.ActiveGPUs++
		bw := entry.BandwidthMBps
		if bw <= 0 {
			bw = effectiveMemoryBandwidth(byGPU[entry.GPU])
		}
		if bw <= 0 {
			bw = 1
			est.Confidence = "low"
		}
		deviceCost := float64(max(1, entry.ModelMB)) / float64(bw)
		est.DecodeCost += deviceCost
		if deviceCost > maxDeviceCost {
			maxDeviceCost = deviceCost
			est.Bottleneck = fmt.Sprintf("GPU %d layer service", entry.GPU)
		}
	}

	if s.Type == MoEOffload && s.NCPUMoE > 0 && model.NumLayers > 0 {
		moeLayers := max(1, model.NumLayers-max(0, model.LeadingDense))
		cpuExpertMB := float64(bytesToMiBCeil(model.ExpertBytes-model.ShexpBytes)) *
			float64(min(s.NCPUMoE, moeLayers)) / float64(moeLayers)
		activeFraction := 1.0
		if model.NumExperts > 0 && model.ExpertUsedCount > 0 {
			activeFraction = float64(model.ExpertUsedCount) / float64(model.NumExperts)
		}
		hostBW := caps.HostMemoryBandwidthMBps
		if hostBW <= 0 {
			hostBW = 1
			est.Confidence = "low"
		}
		hostCost := cpuExpertMB * activeFraction / float64(hostBW)
		est.TransferCost += hostCost
		est.DecodeCost += hostCost
		if hostCost > maxDeviceCost {
			est.Bottleneck = "CPU expert bandwidth"
		}
	}

	if est.DecodeCost <= 0 {
		est.DecodeCost = 1
		est.Confidence = "low"
	}
	ubatch := max(32, s.UBatchSize)
	ubatchGain := 1.0 + 0.28*math.Log2(float64(ubatch)/32.0)
	if ubatchGain < 1 {
		ubatchGain = 1
	}
	if ubatchGain > 3.5 {
		ubatchGain = 3.5
	}
	est.PrefillCost = est.DecodeCost / ubatchGain

	targetConcurrency := max(1, opts.WorkloadConcurrency)
	if strings.Contains(strings.ToLower(opts.WorkloadProfile), "parallel") && targetConcurrency < 2 {
		targetConcurrency = max(2, opts.Parallel)
	}
	activeLanes := min(targetConcurrency, max(1, s.Parallel))
	waves := (targetConcurrency + activeLanes - 1) / activeLanes
	// Concurrent lanes share kernels and the weight stream, so they are not free;
	// nevertheless, serving N requested agents in fewer scheduler waves is the
	// central benefit of --parallel. The live workload measures the real curve.
	contention := 1 + 0.07*float64(activeLanes-1)
	est.AgentCost = (0.58*est.PrefillCost + 0.42*est.DecodeCost) * float64(waves) * contention
	if waves > 1 {
		est.Bottleneck = "agent admission queue"
	}
	if est.ActiveGPUs > 1 {
		est.AgentCost *= 1 + 0.015*float64(est.ActiveGPUs-1)
	}
	if ledger.Exact && est.Confidence != "low" {
		est.Confidence = "allocation-measured"
	}
	if est.Bottleneck == "" {
		est.Bottleneck = "scheduler/graph"
	}
	return est
}

// AnalyzeCandidateFrontier annotates every candidate, orders alternates by
// predicted agent cost, and derives the baseline residency state from whether a
// materially faster legal neighbor exists. Candidate zero remains the baseline.
func AnalyzeCandidateFrontier(caps *detect.Capabilities, model *ModelProfile, opts Options, candidates []CalibrationCandidate) []CalibrationCandidate {
	if len(candidates) == 0 {
		return candidates
	}
	for i := range candidates {
		if candidates[i].Strategy == nil {
			continue
		}
		candidates[i].Estimate = AnalyzeStrategy(caps, model, candidates[i].Strategy, opts)
	}
	if len(candidates) > 2 {
		sort.SliceStable(candidates[1:], func(i, j int) bool {
			a := candidates[1+i].Estimate
			b := candidates[1+j].Estimate
			if a.Feasible != b.Feasible {
				return a.Feasible
			}
			if a.AgentCost != b.AgentCost {
				return a.AgentCost < b.AgentCost
			}
			// CalibrationCandidates already emits equal-cost coordinates nearest
			// the stable baseline first. Preserve that deterministic order; a name
			// sort silently promoted farther batch rungs when the cost model quite
			// correctly treated logical batch as a tie.
			return false
		})
	}
	base := candidates[0].Strategy
	if base == nil {
		return candidates
	}
	if base.MMapRequired || base.Type == DenseCPUOffload || base.Type == CPUOnly {
		base.Residency = ResidencyNonResident
		base.OptimizationBoundary = SummarizeCandidateFrontier(candidates)
		return candidates
	}
	base.Residency = ResidencyTight
	baseCost := candidates[0].Estimate.AgentCost
	exact := base.ResourceLedger != nil && base.ResourceLedger.Exact
	actionable := false
	for i := 1; i < len(candidates); i++ {
		candidate := &candidates[i]
		if !candidate.Estimate.Feasible || math.IsInf(candidate.Estimate.AgentCost, 1) {
			continue
		}
		costGain := baseCost > 0 && candidate.Estimate.AgentCost < baseCost*0.98
		headroom := materialHeadroomChange(base, candidate.Strategy)
		candidate.Estimate.ActionableGain = costGain || headroom
		if candidate.Estimate.ActionableGain {
			actionable = true
		}
	}
	if exact && moeTopologyExperimentHeadroom(model, base, candidates) {
		// A GPU being full because the planner packed it is not proof that the
		// system is a tight fit. For a CPU-expert MoE, one model-derived expert
		// layer of exact host fallback room plus a feasible same-workload topology
		// is enough to enter the performance lane. This distinguishes the current
		// roomy server from the same checkpoint on the older genuinely tight host
		// without a model-size or fixed-MiB threshold.
		actionable = true
	}
	// Roomy is an exact-ledger fact: residual VRAM/RAM admits at least one
	// materially larger runtime or faster legal placement. Predicted cost can
	// rank a finalist, but it cannot invent headroom that the ledger does not
	// show, and a larger batch that the cost model ignores still counts.
	if exact && actionable {
		base.Residency = ResidencyRoomy
	}
	base.OptimizationBoundary = SummarizeCandidateFrontier(candidates)
	return candidates
}

func moeTopologyExperimentHeadroom(model *ModelProfile, base *Strategy, candidates []CalibrationCandidate) bool {
	if model == nil || base == nil || base.Type != MoEOffload || base.MMapRequired ||
		base.ResourceLedger == nil || !base.ResourceLedger.Exact || base.NCPUMoE <= 0 {
		return false
	}
	moeLayers := max(1, model.NumLayers-max(0, model.LeadingDense))
	expertBytes := model.ExpertBytes - model.ShexpBytes
	if expertBytes <= 0 {
		expertBytes = model.ExpertBytes
	}
	perLayerMB := ceilDivInt(bytesToMiBCeil(expertBytes), moeLayers)
	if perLayerMB <= 0 || base.ResourceLedger.Host.SlackMB < perLayerMB {
		return false
	}
	for i := 1; i < len(candidates); i++ {
		candidate := &candidates[i]
		if candidate.Strategy == nil || !candidate.Estimate.Feasible ||
			!balanceTopologyCompatible(base, candidate.Strategy) ||
			!materialTopologyChange(base, candidate.Strategy) {
			continue
		}
		candidate.Estimate.ActionableGain = true
		return true
	}
	return false
}

func materialTopologyChange(base, candidate *Strategy) bool {
	if base == nil || candidate == nil {
		return false
	}
	return base.Type != candidate.Type || base.MainGPU != candidate.MainGPU ||
		base.NCPUMoE != candidate.NCPUMoE || base.OTString != candidate.OTString ||
		base.PlacementPolicy != candidate.PlacementPolicy ||
		splitCompactKey(base.TensorSplit) != splitCompactKey(candidate.TensorSplit)
}

// materialHeadroomChange reports whether the neighbor spends residual capacity
// on a different runtime or topology. Cost ranking is independent: logical
// batch is a fairness/throughput knob the relative cost model does not price.
func materialHeadroomChange(base, candidate *Strategy) bool {
	if base == nil || candidate == nil {
		return false
	}
	if candidate.UBatchSize > base.UBatchSize || candidate.BatchSize > base.BatchSize {
		return true
	}
	if candidate.Parallel > base.Parallel {
		return true
	}
	if base.KVPlacement == "cpu" && candidate.KVPlacement == "gpu" {
		return true
	}
	if denserGPUExpertPack(base, candidate) {
		return true
	}
	return false
}

// sameProvenShape is the tight-resident live-search gate: keep the allocation-
// proven topology and only admit monotonic batch/ubatch/slot experiments.
func sameProvenShape(base, candidate *Strategy) bool {
	if base == nil || candidate == nil {
		return false
	}
	return base.Type == candidate.Type &&
		base.ContextSize == candidate.ContextSize &&
		base.MainGPU == candidate.MainGPU &&
		base.KVPlacement == candidate.KVPlacement &&
		base.KVType == candidate.KVType &&
		base.SWAFull == candidate.SWAFull &&
		base.MMap == candidate.MMap &&
		base.MMapRequired == candidate.MMapRequired &&
		base.NCPUMoE == candidate.NCPUMoE &&
		base.SplitMode == candidate.SplitMode &&
		base.OTString == candidate.OTString &&
		splitCompactKey(base.TensorSplit) == splitCompactKey(candidate.TensorSplit)
}

// denserGPUExpertPack is a monotonic recovery from a too-conservative MoE
// baseline: fewer CPU expert layers, same residency class, no new mmap. A
// slightly different split or -ot is expected because those strings are the
// expert pin list, not a topology experiment.
func denserGPUExpertPack(base, candidate *Strategy) bool {
	if base == nil || candidate == nil {
		return false
	}
	if candidate.Type != base.Type || candidate.KVPlacement != base.KVPlacement {
		return false
	}
	if candidate.MMapRequired && !base.MMapRequired {
		return false
	}
	if base.NCPUMoE > 0 && candidate.NCPUMoE > 0 && candidate.NCPUMoE < base.NCPUMoE {
		return true
	}
	return gpuExpertLayers(candidate) > gpuExpertLayers(base)
}

func gpuExpertLayers(s *Strategy) int {
	if s == nil {
		return 0
	}
	n := 0
	for _, entry := range s.VRAMLedger {
		n += entry.ExpertLayers
	}
	return n
}

func tightLiveEligible(base, candidate *Strategy) bool {
	return sameProvenShape(base, candidate) || denserGPUExpertPack(base, candidate)
}

// TightLiveCandidates keeps the baseline plus same-shape neighbors and any
// denser GPU-expert pack. Inverting a split or changing KV/mmap stays in the
// calculated boundary and is not live-tested on a tight launch.
func TightLiveCandidates(candidates []CalibrationCandidate) []CalibrationCandidate {
	if len(candidates) < 2 || candidates[0].Strategy == nil ||
		candidates[0].Strategy.Residency != ResidencyTight ||
		candidates[0].Strategy.ResourceLedger == nil ||
		!candidates[0].Strategy.ResourceLedger.Exact {
		return candidates
	}
	out := []CalibrationCandidate{candidates[0]}
	base := candidates[0].Strategy
	for _, candidate := range candidates[1:] {
		if candidate.Estimate.Feasible && tightLiveEligible(base, candidate.Strategy) {
			out = append(out, candidate)
		}
	}
	return out
}

// GPUUtilSample is one observed device during a bounded agent workload.
type GPUUtilSample struct {
	GPU        int `json:"gpu"`
	SMPercent  int `json:"sm_percent"`
	MemPercent int `json:"mem_percent"`
}

// DeviceBalanceSignal is measured evidence that one device owns the critical
// compute stage while another active placement device waits. It does not prove
// that a different topology is faster; it only makes one topology challenger
// worth contained admission and a live workload comparison.
type DeviceBalanceSignal struct {
	Observed   bool `json:"observed"`
	Imbalanced bool `json:"imbalanced"`
	BusyGPU    int  `json:"busy_gpu"`
	IdleGPU    int  `json:"idle_gpu"`
	BusySM     int  `json:"busy_sm_percent"`
	IdleSM     int  `json:"idle_sm_percent"`
}

// AnalyzeDeviceBalance derives a typed signal from active-workload samples.
// Missing samples and single-device placements fail closed.
func AnalyzeDeviceBalance(s *Strategy, samples []GPUUtilSample) DeviceBalanceSignal {
	signal := DeviceBalanceSignal{BusyGPU: -1, IdleGPU: -1}
	if s == nil || len(samples) == 0 {
		return signal
	}
	maxSM, minSM := -1, 101
	activeDevices := 0
	for _, sample := range samples {
		if !deviceHoldsWork(s, sample.GPU) {
			continue
		}
		activeDevices++
		if sample.SMPercent > maxSM {
			maxSM = sample.SMPercent
			signal.BusyGPU = sample.GPU
		}
		if sample.SMPercent < minSM {
			minSM = sample.SMPercent
			signal.IdleGPU = sample.GPU
		}
	}
	if activeDevices < 2 || signal.BusyGPU < 0 || signal.IdleGPU < 0 || signal.BusyGPU == signal.IdleGPU {
		return signal
	}
	signal.Observed = true
	signal.BusySM = maxSM
	signal.IdleSM = minSM
	signal.Imbalanced = maxSM >= 70 && minSM <= 10
	return signal
}

// DeviceBalanceBottleneck turns per-device SM samples into a topology signal.
// A card that holds weights but does no compute is storage, not a speed plan.
// Missing samples fail closed: no observation is not evidence of balance.
func DeviceBalanceBottleneck(s *Strategy, samples []GPUUtilSample) string {
	signal := AnalyzeDeviceBalance(s, samples)
	if !signal.Imbalanced {
		return ""
	}
	return fmt.Sprintf("GPU %d saturated (%d%% SM) while GPU %d is idle (%d%% SM)",
		signal.BusyGPU, signal.BusySM, signal.IdleGPU, signal.IdleSM)
}

// SelectDeviceBalanceFinalist chooses one complete topology that removes a
// material share of model work from the measured busy device. All workload and
// quality coordinates stay fixed, and a tight MoE may not move more experts to
// CPU. The result is only a finalist: exact admission and the live agent screen
// remain the promotion authority.
func SelectDeviceBalanceFinalist(candidates []CalibrationCandidate, signal DeviceBalanceSignal) (CalibrationCandidate, bool) {
	if len(candidates) < 2 || !signal.Imbalanced || candidates[0].Strategy == nil {
		return CalibrationCandidate{}, false
	}
	base := candidates[0].Strategy
	baseBusy := deviceModelFraction(base, signal.BusyGPU)
	baseBackbone := deviceBackboneFraction(base, signal.BusyGPU)
	if baseBusy <= 0 && baseBackbone <= 0 {
		return CalibrationCandidate{}, false
	}
	baseIdle := deviceModelFraction(base, signal.IdleGPU)
	baseIdleBackbone := deviceBackboneFraction(base, signal.IdleGPU)
	bestIndex := -1
	bestScore := 0.0
	for i := 1; i < len(candidates); i++ {
		candidate := candidates[i]
		if candidate.Strategy == nil || !candidate.Estimate.Feasible ||
			!balanceTopologyCompatible(base, candidate.Strategy) {
			continue
		}
		busy := deviceModelFraction(candidate.Strategy, signal.BusyGPU)
		modelRelief := baseBusy - busy
		backboneRelief := baseBackbone - deviceBackboneFraction(candidate.Strategy, signal.BusyGPU)
		// Two percentage points is large enough to cross an ordinary layer or
		// expert-placement boundary without reacting to ledger rounding. For MoE,
		// backbone ownership is the primary signal: stored expert bytes execute
		// only when routed, while every ordinary layer lies on the serial path.
		materialRelief := modelRelief >= 0.02
		if base.Type == MoEOffload {
			materialRelief = materialRelief || backboneRelief >= 0.02
		}
		if !materialRelief {
			continue
		}
		idleGain := deviceModelFraction(candidate.Strategy, signal.IdleGPU) - baseIdle
		idleBackboneGain := deviceBackboneFraction(candidate.Strategy, signal.IdleGPU) - baseIdleBackbone
		score := math.Max(0, modelRelief) + 0.25*math.Max(0, idleGain)
		if base.Type == MoEOffload {
			score += 2*math.Max(0, backboneRelief) + 0.5*math.Max(0, idleBackboneGain)
		}
		if bestIndex < 0 || score > bestScore+1e-9 ||
			(math.Abs(score-bestScore) <= 1e-9 && candidate.Estimate.AgentCost < candidates[bestIndex].Estimate.AgentCost) {
			bestIndex = i
			bestScore = score
		}
	}
	if bestIndex < 0 {
		return CalibrationCandidate{}, false
	}
	return candidates[bestIndex], true
}

// deviceBackboneFraction reports the share of ordinary layer slots owned by a
// physical GPU. ResourceLedger.Devices preserves capability order, which is
// also tensor-split order; using it avoids assuming physical indexes are dense.
// This is a better compute-role proxy for sparse MoE than resident model bytes,
// because routed experts may occupy most VRAM while executing only sparsely.
func deviceBackboneFraction(s *Strategy, gpu int) float64 {
	if s == nil || gpu < 0 || s.Type == CPUOnly || s.Type == DenseCPUOffload {
		return 0
	}
	if s.Type == SingleGPU {
		if s.MainGPU == gpu {
			return 1
		}
		return 0
	}
	n := len(s.TensorSplit)
	ordinal := -1
	if s.ResourceLedger != nil && len(s.ResourceLedger.Devices) > 0 {
		n = len(s.ResourceLedger.Devices)
		for i, entry := range s.ResourceLedger.Devices {
			if entry.GPU == gpu {
				ordinal = i
				break
			}
		}
	}
	if ordinal < 0 && gpu < len(s.TensorSplit) {
		ordinal = gpu
	}
	if ordinal < 0 || ordinal >= n {
		return 0
	}
	return normalizedStrategySplit(s, n)[ordinal]
}

func balanceTopologyCompatible(base, candidate *Strategy) bool {
	if base == nil || candidate == nil || candidate.Type == CPUOnly || candidate.Type == DenseCPUOffload {
		return false
	}
	if base.ContextSize != candidate.ContextSize || base.Parallel != candidate.Parallel ||
		base.BatchSize != candidate.BatchSize || base.UBatchSize != candidate.UBatchSize ||
		base.KVPlacement != candidate.KVPlacement || base.KVQuality != candidate.KVQuality ||
		base.KVType != candidate.KVType || base.SWAFull != candidate.SWAFull ||
		base.MMap != candidate.MMap || base.MMapRequired != candidate.MMapRequired {
		return false
	}
	if base.Type == MoEOffload {
		return candidate.Type == MoEOffload && candidate.NCPUMoE <= base.NCPUMoE
	}
	return candidate.Type == base.Type || candidate.Type == SingleGPU || candidate.Type == MultiGPUDense
}

func deviceModelFraction(s *Strategy, gpu int) float64 {
	if s == nil || s.ResourceLedger == nil || gpu < 0 {
		return 0
	}
	total, selected := 0, 0
	for _, entry := range s.ResourceLedger.Devices {
		modelMB := max(0, entry.ModelMB)
		total += modelMB
		if entry.GPU == gpu {
			selected = modelMB
		}
	}
	if total <= 0 {
		return 0
	}
	return float64(selected) / float64(total)
}

func deviceHoldsWork(s *Strategy, gpu int) bool {
	if s == nil {
		return false
	}
	if s.Type == CPUOnly {
		return false
	}
	if s.Type == SingleGPU {
		return s.MainGPU == gpu
	}
	if s.ResourceLedger != nil {
		for _, entry := range s.ResourceLedger.Devices {
			if entry.GPU == gpu {
				return entry.Active
			}
		}
	}
	for _, entry := range s.VRAMLedger {
		if entry.GPU == gpu && entry.ExpertLayers > 0 {
			return true
		}
	}
	if gpu >= 0 && gpu < len(s.TensorSplit) {
		return s.TensorSplit[gpu] > 0
	}
	return false
}

// SummarizeCandidateFrontier records the complete calculated search boundary,
// not merely the one alternate selected for live measurement.
func SummarizeCandidateFrontier(candidates []CalibrationCandidate) *OptimizationBoundary {
	if len(candidates) == 0 {
		return nil
	}
	boundary := &OptimizationBoundary{CandidateCount: len(candidates)}
	topologies := map[string]bool{}
	first := true
	for _, candidate := range candidates {
		s := candidate.Strategy
		if s == nil {
			continue
		}
		if candidate.Estimate.Feasible {
			boundary.FeasibleCount++
		}
		if s.ResourceLedger != nil && s.ResourceLedger.Exact {
			boundary.ExactCount++
		}
		parallel := max(1, s.Parallel)
		if first {
			boundary.MinBatch, boundary.MaxBatch = s.BatchSize, s.BatchSize
			boundary.MinUBatch, boundary.MaxUBatch = s.UBatchSize, s.UBatchSize
			boundary.MinParallel, boundary.MaxParallel = parallel, parallel
			first = false
		} else {
			boundary.MinBatch, boundary.MaxBatch = min(boundary.MinBatch, s.BatchSize), max(boundary.MaxBatch, s.BatchSize)
			boundary.MinUBatch, boundary.MaxUBatch = min(boundary.MinUBatch, s.UBatchSize), max(boundary.MaxUBatch, s.UBatchSize)
			boundary.MinParallel, boundary.MaxParallel = min(boundary.MinParallel, parallel), max(boundary.MaxParallel, parallel)
		}
		topology := string(s.Type)
		if s.PlacementPolicy != "" {
			topology += ":" + s.PlacementPolicy
		}
		if len(s.TensorSplit) > 0 {
			topology += ":gpu-" + splitGPUSet(s.TensorSplit)
		} else if s.Type == SingleGPU {
			topology += fmt.Sprintf(":gpu-%d", s.MainGPU)
		}
		topologies[topology] = true
	}
	for topology := range topologies {
		boundary.Topologies = append(boundary.Topologies, topology)
	}
	sort.Strings(boundary.Topologies)
	base := candidates[0].Strategy
	if base != nil && base.ResourceLedger != nil {
		boundary.DeviceSlackMB = make(map[int]int, len(base.ResourceLedger.Devices))
		for _, device := range base.ResourceLedger.Devices {
			boundary.DeviceSlackMB[device.GPU] = device.SlackMB
		}
		boundary.HostSlackMB = base.ResourceLedger.Host.SlackMB
		boundary.Evidence = base.ResourceLedger.Evidence
	}
	return boundary
}
