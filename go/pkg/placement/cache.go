package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// CacheEntry holds a validated placement cache entry for MoE.
type CacheEntry struct {
	GPUAssignments []GPUAssignment `json:"gpu_assignments"`     // cuda_idx:start:count
	OTString       string          `json:"ot_string,omitempty"` // exact -ot (preserves sub-layer pins)
	TensorSplit    []float64       `json:"tensor_split,omitempty"`
	SplitMode      string          `json:"split_mode,omitempty"`
	NCPUMoE        int             `json:"n_cpu_moe"`
	BatchSize      int             `json:"batch_size"`
	UBatchSize     int             `json:"ubatch_size"`
	Parallel       int             `json:"parallel"`
	KVUnified      bool            `json:"kv_unified"`
	NoPinned       bool            `json:"no_pinned"`
	MMap           bool            `json:"mmap"`
	// PlanFreeVRAM records the free VRAM each GPU showed when this plan was
	// computed, by CUDA index. A plan is only as good as that reading: one
	// computed while the previous server was still releasing 43 GB pinned
	// --n-cpu-moe at the old context's value and left 8.4 GB of the split
	// owner idle, and because the plan was cached, every restart replayed the
	// mistake. Deficits fix themselves -- the launch preflight measures and
	// re-plans -- but nothing downstream ever notices a plan that asks for too
	// little, so the cache has to.
	PlanFreeVRAM map[int]int `json:"plan_free_vram,omitempty"`
}

// planFreeVRAMSlackMB is how much more free VRAM the machine may show before a
// cached plan is considered stale-pessimistic. An expert layer costs ~1.4 GB,
// so a full gigabyte of unclaimed VRAM is the scale at which the plan starts
// leaving whole layers on the CPU; ordinary allocator jitter is far below it.
const planFreeVRAMSlackMB = 1024

// GPUAssignment describes layers assigned to a GPU.
type GPUAssignment struct {
	CUDAIndex int `json:"cuda_index"`
	Start     int `json:"start"`
	Count     int `json:"count"`
}

// LoadPlacementCache attempts to load a validated placement cache file.
func LoadPlacementCache(cachePath string, caps *detect.Capabilities, kvTotalMB int) (*CacheEntry, error) {
	data, err := os.ReadFile(cachePath)
	if err != nil {
		return nil, err
	}
	content := string(data)

	// Parse the legacy key=value cache format
	entry := &CacheEntry{
		BatchSize:  1024,
		UBatchSize: 512,
		Parallel:   2,
	}
	hasMMap := false
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.Trim(strings.TrimSpace(parts[1]), `"`)
		switch key {
		case "CACHED_GPU_ASSIGNMENTS":
			entry.GPUAssignments = parseGPUAssignments(val)
		case "CACHED_OT_STRING":
			entry.OTString = val
		case "CACHED_TENSOR_SPLIT":
			entry.TensorSplit = parseTensorSplit(val)
		case "CACHED_SPLIT_MODE":
			entry.SplitMode = val
		case "CACHED_NCPUMOE":
			entry.NCPUMoE, _ = strconv.Atoi(val)
		case "CACHED_BATCH":
			entry.BatchSize, _ = strconv.Atoi(val)
		case "CACHED_UBATCH":
			entry.UBatchSize, _ = strconv.Atoi(val)
		case "CACHED_PARALLEL":
			entry.Parallel, _ = strconv.Atoi(val)
		case "CACHED_KVUNIFIED":
			entry.KVUnified = val == "1"
		case "CACHED_NO_PINNED":
			entry.NoPinned = val == "1"
		case "CACHED_MMAP":
			hasMMap = true
			entry.MMap = val == "1"
		case "CACHED_PLAN_FREEVRAM":
			entry.PlanFreeVRAM = parsePlanFreeVRAM(val)
		}
	}

	// A plan computed under tighter VRAM than the machine now shows would
	// replay its pessimism forever; recompute instead. Entries from before this
	// field carry no reading to compare, and are grandfathered because the
	// deficit direction is still covered by the launch preflight.
	for idx, plannedFree := range entry.PlanFreeVRAM {
		for _, g := range caps.GPUs {
			if g.Index == idx && g.VRAMFreeMB() > plannedFree+planFreeVRAMSlackMB {
				return nil, fmt.Errorf(
					"cached plan is stale: GPU%d now has %dMB free but the plan was computed against %dMB",
					idx, g.VRAMFreeMB(), plannedFree)
			}
		}
	}

	// Validate: each GPU must have enough VRAM for assigned layers + KV share
	for _, assign := range entry.GPUAssignments {
		found := false
		for _, g := range caps.GPUs {
			if g.Index == assign.CUDAIndex {
				found = true
				// We can't validate exact layer MB without model info here,
				// but we can check that the GPU exists
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("cached assignment references unknown GPU %d", assign.CUDAIndex)
		}
	}
	if len(entry.GPUAssignments) == 0 && len(entry.TensorSplit) == 0 && entry.OTString == "" {
		return nil, fmt.Errorf("cache has no MoE placement data")
	}
	if !hasMMap {
		return nil, fmt.Errorf("cache missing CACHED_MMAP")
	}
	if len(entry.GPUAssignments) > 0 && len(entry.TensorSplit) == 0 {
		return nil, fmt.Errorf("cached MoE GPU assignments missing CACHED_TENSOR_SPLIT")
	}

	return entry, nil
}

// StrategyToCacheEntry captures a computed MoE placement as a cache entry so a
// launch that loaded cleanly can be persisted (via the launcher's success hook)
// and reused verbatim next time. Stores the exact -ot so sub-layer gate+up pins
// survive the round-trip.
func StrategyToCacheEntry(s *Strategy) *CacheEntry {
	if s == nil {
		return nil
	}
	entry := &CacheEntry{
		OTString:    s.OTString,
		TensorSplit: append([]float64(nil), s.TensorSplit...),
		SplitMode:   s.SplitMode,
		NCPUMoE:     s.NCPUMoE,
		BatchSize:   s.BatchSize,
		UBatchSize:  s.UBatchSize,
		Parallel:    s.Parallel,
		MMap:        s.MMap,
		KVUnified:   s.KVPlacement == "gpu",
	}
	if len(s.PlanFreeVRAM) > 0 {
		entry.PlanFreeVRAM = make(map[int]int, len(s.PlanFreeVRAM))
		for idx, free := range s.PlanFreeVRAM {
			entry.PlanFreeVRAM[idx] = free
		}
	}
	return entry
}

// SavePlacementCache writes a placement cache file in bash-compatible format.
func SavePlacementCache(cachePath string, entry *CacheEntry) error {
	_ = os.MkdirAll(filepath.Dir(cachePath), 0755)
	var parts []string
	parts = append(parts, fmt.Sprintf("# ggrun placement cache (%s)", time.Now().UTC().Format("2006-01-02T15:04:05Z")))
	if len(entry.GPUAssignments) > 0 {
		var assigns []string
		for _, a := range entry.GPUAssignments {
			assigns = append(assigns, fmt.Sprintf("%d:%d:%d", a.CUDAIndex, a.Start, a.Count))
		}
		parts = append(parts, fmt.Sprintf("CACHED_GPU_ASSIGNMENTS=\"%s\"", strings.Join(assigns, " ")))
	}
	if entry.OTString != "" {
		parts = append(parts, fmt.Sprintf("CACHED_OT_STRING=\"%s\"", entry.OTString))
	}
	if len(entry.TensorSplit) > 0 {
		var split []string
		for _, v := range entry.TensorSplit {
			split = append(split, fmt.Sprintf("%.2f", v))
		}
		parts = append(parts, fmt.Sprintf("CACHED_TENSOR_SPLIT=\"%s\"", strings.Join(split, ",")))
	}
	if entry.SplitMode != "" {
		parts = append(parts, fmt.Sprintf("CACHED_SPLIT_MODE=\"%s\"", entry.SplitMode))
	}
	if entry.NCPUMoE > 0 {
		parts = append(parts, fmt.Sprintf("CACHED_NCPUMOE=\"%d\"", entry.NCPUMoE))
	}
	parts = append(parts, fmt.Sprintf("CACHED_BATCH=\"%d\"", entry.BatchSize))
	parts = append(parts, fmt.Sprintf("CACHED_UBATCH=\"%d\"", entry.UBatchSize))
	parts = append(parts, fmt.Sprintf("CACHED_PARALLEL=\"%d\"", entry.Parallel))
	if entry.KVUnified {
		parts = append(parts, "CACHED_KVUNIFIED=\"1\"")
	}
	if entry.NoPinned {
		parts = append(parts, "CACHED_NO_PINNED=\"1\"")
	}
	if entry.MMap {
		parts = append(parts, "CACHED_MMAP=\"1\"")
	} else {
		parts = append(parts, "CACHED_MMAP=\"0\"")
	}
	if len(entry.PlanFreeVRAM) > 0 {
		idxs := make([]int, 0, len(entry.PlanFreeVRAM))
		for idx := range entry.PlanFreeVRAM {
			idxs = append(idxs, idx)
		}
		sort.Ints(idxs)
		var tokens []string
		for _, idx := range idxs {
			tokens = append(tokens, fmt.Sprintf("%d:%d", idx, entry.PlanFreeVRAM[idx]))
		}
		parts = append(parts, fmt.Sprintf("CACHED_PLAN_FREEVRAM=\"%s\"", strings.Join(tokens, " ")))
	}
	return os.WriteFile(cachePath, []byte(strings.Join(parts, "\n")+"\n"), 0644)
}

// snapshotPlanFreeVRAM captures the planning view of free VRAM so it can be
// stored beside the plan it produced.
func snapshotPlanFreeVRAM(caps *detect.Capabilities) map[int]int {
	if caps == nil || len(caps.GPUs) == 0 {
		return nil
	}
	out := make(map[int]int, len(caps.GPUs))
	for _, g := range caps.GPUs {
		out[g.Index] = g.VRAMFreeMB()
	}
	return out
}

func parsePlanFreeVRAM(s string) map[int]int {
	out := map[int]int{}
	for _, tok := range strings.Fields(s) {
		parts := strings.Split(tok, ":")
		if len(parts) != 2 {
			continue
		}
		idx, err1 := strconv.Atoi(parts[0])
		free, err2 := strconv.Atoi(parts[1])
		if err1 != nil || err2 != nil || free < 0 {
			continue
		}
		out[idx] = free
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func parseTensorSplit(s string) []float64 {
	fields := strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ' ' || r == ':' })
	out := make([]float64, 0, len(fields))
	for _, f := range fields {
		if f == "" {
			continue
		}
		v, err := strconv.ParseFloat(f, 64)
		if err != nil || v < 0 {
			continue
		}
		out = append(out, v)
	}
	return out
}

func parseGPUAssignments(s string) []GPUAssignment {
	var out []GPUAssignment
	for _, tok := range strings.Fields(s) {
		parts := strings.Split(tok, ":")
		if len(parts) != 3 {
			continue
		}
		ci, _ := strconv.Atoi(parts[0])
		st, _ := strconv.Atoi(parts[1])
		ct, _ := strconv.Atoi(parts[2])
		out = append(out, GPUAssignment{CUDAIndex: ci, Start: st, Count: ct})
	}
	return out
}
