package placement

import (
	"encoding/json"
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
	// ModelBasename is the cache-clearing identity written beside the plan so a
	// "clear caches" action can find every placement file a model created without
	// reconstructing every variant of the opaque cache key. Old files without the
	// line stay valid (LoadPlacementCache ignores it); they simply cannot be
	// matched for clearing.
	ModelBasename string `json:"model_basename,omitempty"`
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
		case "CACHED_MODEL_BASENAME":
			entry.ModelBasename = val
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
		OTString:     s.OTString,
		TensorSplit:  append([]float64(nil), s.TensorSplit...),
		SplitMode:    s.SplitMode,
		NCPUMoE:      s.NCPUMoE,
		BatchSize:    s.BatchSize,
		UBatchSize:   s.UBatchSize,
		Parallel:     s.Parallel,
		MMap:         s.MMap,
		KVUnified:    s.KVPlacement == "gpu",
		ModelBasename: s.ModelBasename,
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
	if entry.ModelBasename != "" {
		parts = append(parts, fmt.Sprintf("CACHED_MODEL_BASENAME=%q", entry.ModelBasename))
	}
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
	return atomicWriteFile(cachePath, []byte(strings.Join(parts, "\n")+"\n"), 0o644)
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

// ClearModelCaches removes every cached configuration ggrun learned about one
// model — probe caches, measured KV rate/geometry, calibration decisions, and
// placement caches — while keeping the model's GGUF file untouched. It is the
// placement-package half of the TUI's "Clear caches" action and the "launch
// without cached config" escape hatch, and is safe to call repeatedly.
//
// cacheDir is the ggrun cache directory (config.CacheDir). model identifies the
// model the same way every cache writer does, so the wrong model's data is
// never removed. Only files whose header/tag line names this exact basename are
// deleted; placement caches written before the CACHED_MODEL_BASENAME marker are
// skipped rather than risk deleting another model's plan (recompute is cheap).
func ClearModelCaches(cacheDir string, model *ModelProfile) (files int, err error) {
	if model == nil {
		return 0, nil
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	base := model.Basename
	if base == "" {
		base = filepath.Base(model.Path)
	}
	// The .probe writer headers with filepath.Base(model.Path) (which may carry
	// the .gguf extension) while the KV/placement writers use model.Basename.
	// Match either form so a model whose Basename differs from its file name by
	// an extension is still cleared.
	pathBase := filepath.Base(model.Path)
	if pathBase == base {
		pathBase = ""
	}
	if base == "" && pathBase == "" {
		return 0, nil
	}
	matchesModel := func(header string) bool {
		if base != "" && header == base {
			return true
		}
		return pathBase != "" && header == pathBase
	}

	removed := 0
	remove := func(path string) {
		if path == "" {
			return
		}
		if err := os.Remove(path); err == nil {
			removed++
		}
		// os.IsNotExist and any other failure (stale lock, read-only dir) are
		// both tolerated so one bad file does not abort the whole clear.
	}

	// Measured KV rate/geometry: kv_<basename>_<size>.cache in the cache root.
	remove(kvCachePath(cacheDir, model))

	// Probe caches and placement caches live in the cache root (and, for
	// historical installs, ~/.cache/ggrun/probes). Both are hashed filenames, so
	// match by the model-identity header line written inside the file rather
	// than by name.
	entries, readErr := os.ReadDir(cacheDir)
	if readErr == nil {
		for _, ent := range entries {
			if ent.IsDir() {
				continue
			}
			name := ent.Name()
			if !strings.HasSuffix(name, ".probe") && !strings.HasSuffix(name, ".place") {
				continue
			}
			content, readErr := os.ReadFile(filepath.Join(cacheDir, name))
			if readErr != nil {
				continue
			}
			if probeCacheForModel(string(content), matchesModel) {
				remove(filepath.Join(cacheDir, name))
			}
		}
	}
	// Legacy probe directory (~/.cache/ggrun/probes) may hold the same model's
	// older probes. Walk it with the same header match.
	if probesDir := filepath.Join(cacheDir, "probes"); probesDir != cacheDir {
		if entries, readErr := os.ReadDir(probesDir); readErr == nil {
			for _, ent := range entries {
				if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".probe") {
					continue
				}
				content, readErr := os.ReadFile(filepath.Join(probesDir, ent.Name()))
				if readErr != nil {
					continue
				}
				if probeCacheForModel(string(content), matchesModel) {
					remove(filepath.Join(probesDir, ent.Name()))
				}
			}
		}
	}

	// Calibration decisions: cal-<scopehash>.json under cacheDir/calibration.
	// Each file records the scope hash but not the model basename, so match by
	// reading the file and reconstructing the scope key from the model's cache
	// identity via the load path that rejects scope mismatches.
	calibrationDir := filepath.Join(cacheDir, "calibration")
	if entries, readErr := os.ReadDir(calibrationDir); readErr == nil {
		for _, ent := range entries {
			if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".json") {
				continue
			}
			data, readErr := os.ReadFile(filepath.Join(calibrationDir, ent.Name()))
			if readErr != nil {
				continue
			}
			if calibrationDecisionForModel(data, model, base) {
				remove(filepath.Join(calibrationDir, ent.Name()))
			}
		}
	}

	// Verified configs: verified-<scopehash>.json under cacheDir/verified-configs.
	// Each record stores the model basename (like the placement cache) so a
	// "clear caches" action can remove every verified config a model created.
	verifiedDir := filepath.Join(cacheDir, "verified-configs")
	if entries, readErr := os.ReadDir(verifiedDir); readErr == nil {
		for _, ent := range entries {
			if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".json") {
				continue
			}
			data, readErr := os.ReadFile(filepath.Join(verifiedDir, ent.Name()))
			if readErr != nil {
				continue
			}
			var vc VerifiedConfig
			if json.Unmarshal(data, &vc) == nil && vc.ModelBasename != "" && matchesModel(vc.ModelBasename) {
				remove(filepath.Join(verifiedDir, ent.Name()))
			}
		}
	}

	return removed, nil
}

// probeCacheForModel reports whether a probe/placement cache file's header names
// this model. Both formats write a model-identity line — the probe writer emits
// `# Probe cache for <basename>`, the placement writer emits
// `CACHED_MODEL_BASENAME="<basename>"` — so either match qualifies. matcher
// accepts both the Basename and filepath.Base(Path) forms.
func probeCacheForModel(content string, matcher func(string) bool) bool {
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "# Probe cache for "):
			if matcher(strings.TrimSpace(strings.TrimPrefix(line, "# Probe cache for "))) {
				return true
			}
		case strings.HasPrefix(line, "CACHED_MODEL_BASENAME="):
			val := strings.Trim(strings.TrimSpace(strings.TrimPrefix(line, "CACHED_MODEL_BASENAME=")), `"`)
			if matcher(val) {
				return true
			}
		}
	}
	return false
}

// calibrationDecisionForModel reports whether a stored calibration decision's
// model marker names this model. Decisions written after ClearModelCaches
// support carry an explicit ModelBasename field (the scope key is an opaque
// hash of model+backend+hardware that cannot be reversed), so the field is the
// authoritative match; the file header is a best-effort fallback for records
// that predate the marker.
func calibrationDecisionForModel(data []byte, model *ModelProfile, base string) bool {
	var d CalibrationDecision
	if json.Unmarshal(data, &d) == nil && d.ModelBasename != "" {
		return d.ModelBasename == base
	}
	// Fallback: the launcher writes `cal-<hash>.json` with no model text, so
	// only a header that explicitly names the model can be trusted. Save path
	// never wrote one, but tolerate a hand-edited marker.
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# model: ") && line == "# model: "+base {
			return true
		}
	}
	return false
}
