package placement

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// VerifiedConfigSchemaVersion bumps whenever the record shape or its semantics
// change, so a record saved under older semantics is never applied to a launch
// the new code would have planned differently.
const VerifiedConfigSchemaVersion = 1

// VerifiedConfig is a sibling of CacheEntry that stores the *whole* serving
// decision — placement identity, runtime knobs, and non-flag provenance — so
// the next launch of the exact same scope can start directly from the flags
// that already proved themselves, without re-estimating, re-probing, or
// re-calibrating.
//
// It is deliberately discriminated by StrategyType: for MoEOffload the
// GPUAssignments/OTString/NCPUMoE family is populated, for SingleGPU /
// MultiGPUDense the MainGPU / TensorSplit family, mirroring the existing JSON
// tags on Strategy so serialization is mechanical. Per-launch volatiles — the
// port, log/--log-file path, claudeRouterURL, and the tuned
// --chat-template-file temp path — are excluded; those are rebuilt at each
// launch.
type VerifiedConfig struct {
	SchemaVersion int           `json:"schema_version"`
	ScopeKey      string        `json:"scope_key"` // opaque hash, filename basis
	ModelBasename string        `json:"model_basename,omitempty"` // cache-clear matching, like cache.go:34
	StrategyType  StrategyType  `json:"strategy_type"` // discriminator

	// Placement identity (per strategy type — one family is populated)
	GPUAssignments []GPUAssignment `json:"gpu_assignments,omitempty"` // MoEOffload
	OTString       string          `json:"ot_string,omitempty"`       // MoEOffload
	NCPUMoE        int             `json:"n_cpu_moe,omitempty"`       // MoEOffload
	TensorSplit    []float64       `json:"tensor_split,omitempty"`    // MultiGPUDense (+ MoE)
	SplitMode      string          `json:"split_mode,omitempty"`
	MainGPU        int             `json:"main_gpu,omitempty"` // SingleGPU/MultiGPUDense

	// Runtime knobs that change the emitted flags (buildLaunchServerArgs)
	ContextSize    int    `json:"context_size"`
	KVPlacement    string `json:"kv_placement"`
	KVType         string `json:"kv_type,omitempty"`
	KVTypeV        string `json:"kv_type_v,omitempty"`
	BatchSize      int    `json:"batch_size"`
	UBatchSize     int    `json:"ubatch_size"`
	Parallel       int    `json:"parallel"`
	Threads        int    `json:"threads,omitempty"`
	ThreadsBatch   int    `json:"threads_batch,omitempty"`
	MMap           bool   `json:"mmap"`
	MLock          bool   `json:"mlock,omitempty"`
	FlashAttention bool   `json:"flash_attention"`
	SWAFull        bool   `json:"swa_full"`
	CRAM           int    `json:"cram,omitempty"`
	MaxCheckpoints int    `json:"max_checkpoints,omitempty"`
	CheckpointMinStep int `json:"checkpoint_min_step,omitempty"`
	UseCUDAGraphs  bool   `json:"use_cuda_graphs,omitempty"`
	Host           string `json:"host,omitempty"`
	NoJinja        bool   `json:"no_jinja,omitempty"`
	ReasoningOff   bool   `json:"reasoning_off"`
	MMProjPath     string `json:"mmproj_path,omitempty"`
	Draft          *DraftConfig `json:"draft,omitempty"`
	BackendTag     string `json:"backend_tag,omitempty"` // emitted dialect

	// Non-flag provenance (identity / evidence, not emitted)
	BackendIdentity string         `json:"backend_identity"`   // be.Identity
	BackendPath     string         `json:"backend_path"`       // be.Path
	ChatTemplate    string         `json:"chat_template,omitempty"` // catalog Entry.Name
	Reviewer        string         `json:"reviewer,omitempty"` // claudeCompanionProfile.Name
	PlanFreeVRAM    map[int]int    `json:"plan_free_vram,omitempty"` // stale-plan guard (cache.go:121-129)
	PlannedHostFootprintMB int     `json:"planned_host_footprint_mb,omitempty"`
	MeasuredAt      string         `json:"measured_at"`
}

// VerifiedConfigPath returns the cache file for one verified-config scope.
func VerifiedConfigPath(cacheDir, scopeKey string) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	if len(scopeKey) >= 12 {
		scopeKey = scopeKey[:12]
	}
	return filepath.Join(cacheDir, "verified-configs", "verified-"+scopeKey+".json")
}

// SaveVerifiedConfig persists a verified serving config atomically. It is only
// called from the promotion boundary (verifyAndActivateLaunch) after the
// functional canary and StateActive, so the record is always evidence of a
// launch that actually served. Failures degrade to a stderr log — a save
// failure must never fail the launch.
func SaveVerifiedConfig(cacheDir string, vc VerifiedConfig) (string, error) {
	if vc.SchemaVersion == 0 {
		vc.SchemaVersion = VerifiedConfigSchemaVersion
	}
	if vc.MeasuredAt == "" {
		vc.MeasuredAt = time.Now().UTC().Format(time.RFC3339)
	}
	path := VerifiedConfigPath(cacheDir, vc.ScopeKey)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(vc, "", "  ")
	if err != nil {
		return "", err
	}
	release, err := acquirePlacementLock(path+".lock", 5*time.Second)
	if err != nil {
		return "", err
	}
	defer release()
	if err := atomicWriteFile(path, append(data, '\n'), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

// LoadVerifiedConfig reads a prior verified config for the scope, rejecting
// stale-schema or mismatched keys so a record for a different launch shape is
// never applied. A missing file is a clean miss (caller falls through to the
// estimate path).
func LoadVerifiedConfig(cacheDir, scopeKey string) (*VerifiedConfig, error) {
	path := VerifiedConfigPath(cacheDir, scopeKey)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var vc VerifiedConfig
	if err := json.Unmarshal(data, &vc); err != nil {
		return nil, err
	}
	if vc.SchemaVersion != VerifiedConfigSchemaVersion || vc.ScopeKey != scopeKey {
		return nil, fmt.Errorf("verified config scope mismatch")
	}
	return &vc, nil
}

// DeleteVerifiedConfig removes a verified config record for one scope. A
// runtime OOM is evidence the saved config is unsafe at runtime, so the record
// must not be replayed on the next launch.
func DeleteVerifiedConfig(cacheDir, scopeKey string) error {
	path := VerifiedConfigPath(cacheDir, scopeKey)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// VerifiedToStrategy applies every field of a VerifiedConfig to a fresh
// Strategy so the next launch can start directly from the saved flags. The
// caller's request values win for the knobs the MoE placement-cache path
// already treats as request-owned: Parallel, BatchSize, UBatchSize, and an
// explicit user --no-mmap / --no-cached-config. Runtime-only strategy fields
// (PlacementCacheHit, CompanionPlacements, Backend* dialect probes) are not
// restored — the caller sets what it needs.
func VerifiedToStrategy(vc *VerifiedConfig, opts Options, caps *detect.Capabilities) *Strategy {
	s := &Strategy{
		Type:             vc.StrategyType,
		ContextSize:      vc.ContextSize,
		KVPlacement:      vc.KVPlacement,
		KVQuality:        kvTypeToQuality(vc.KVType),
		KVType:           vc.KVType,
		KVTypeV:          vc.KVTypeV,
		NCPUMoE:          vc.NCPUMoE,
		OTString:         vc.OTString,
		MainGPU:          vc.MainGPU,
		SplitMode:        vc.SplitMode,
		Threads:          vc.Threads,
		ThreadsBatch:     vc.ThreadsBatch,
		MMap:             vc.MMap,
		MLock:            vc.MLock,
		FlashAttention:   vc.FlashAttention,
		SWAFull:          vc.SWAFull,
		CRAM:             vc.CRAM,
		MaxCheckpoints:   vc.MaxCheckpoints,
		CheckpointMinStep: vc.CheckpointMinStep,
		UseCUDAGraphs:    vc.UseCUDAGraphs,
		Host:             vc.Host,
		NoJinja:          vc.NoJinja,
		ReasoningOff:     vc.ReasoningOff,
		MMProjPath:       vc.MMProjPath,
		BackendTag:       vc.BackendTag,
		GPULayers:        999,
		ModelBasename:    vc.ModelBasename,
		PlanFreeVRAM:     vc.PlanFreeVRAM,
		PlannedHostFootprintMB: vc.PlannedHostFootprintMB,
	}
	if vc.TensorSplit != nil {
		s.TensorSplit = append([]float64(nil), vc.TensorSplit...)
	}
	if vc.Draft != nil {
		d := *vc.Draft
		s.Draft = &d
	}
	// Request-owned knobs win (placement.go:1026-1034 generalized).
	if opts.Parallel > 0 {
		s.Parallel = opts.Parallel
	} else if vc.Parallel > 0 {
		s.Parallel = vc.Parallel
	}
	if opts.BatchSize > 0 {
		s.BatchSize = opts.BatchSize
	} else {
		s.BatchSize = vc.BatchSize
	}
	if opts.UBatchSize > 0 {
		s.UBatchSize = opts.UBatchSize
	} else {
		s.UBatchSize = vc.UBatchSize
	}
	if opts.Threads > 0 {
		s.Threads = opts.Threads
		s.ThreadsBatch = opts.Threads
	}
	if opts.Host != "" {
		s.Host = opts.Host
	}
	// Explicit user --no-mmap must still win.
	if opts.NoMMap {
		s.MMap = false
	}
	if opts.ForceMMap {
		s.MMap = true
	}
	if s.Host == "" {
		s.Host = "127.0.0.1"
	}
	if caps != nil {
		if s.Threads <= 0 {
			s.Threads = caps.CPU.Cores
		}
		if s.ThreadsBatch <= 0 {
			s.ThreadsBatch = caps.CPU.Cores
		}
	}
	return s
}

// VerifiedConfigToRecord builds a VerifiedConfig from a resolved, promoted
// Strategy plus the non-flag provenance the launcher holds (backend identity,
// chat template, reviewer). It is the inverse of VerifiedToStrategy and is used
// only on the save path — after verifyAndActivateLaunch reached StateActive.
// The strategy passed in is expected to be the final, post-applyCalibration
// strategy that actually served, so the record captures the full decision.
func VerifiedConfigToRecord(scopeKey, modelBasename string, s *Strategy, backendIdentity, backendPath, chatTemplate, reviewer string) VerifiedConfig {
	vc := VerifiedConfig{
		SchemaVersion:    VerifiedConfigSchemaVersion,
		ScopeKey:         scopeKey,
		ModelBasename:    modelBasename,
		StrategyType:     s.Type,
		OTString:         s.OTString,
		NCPUMoE:          s.NCPUMoE,
		SplitMode:        s.SplitMode,
		MainGPU:          s.MainGPU,
		ContextSize:      s.ContextSize,
		KVPlacement:      s.KVPlacement,
		KVType:           s.KVType,
		KVTypeV:          s.KVTypeV,
		BatchSize:        s.BatchSize,
		UBatchSize:       s.UBatchSize,
		Parallel:         s.Parallel,
		Threads:          s.Threads,
		ThreadsBatch:     s.ThreadsBatch,
		MMap:             s.MMap,
		MLock:            s.MLock,
		FlashAttention:   s.FlashAttention,
		SWAFull:          s.SWAFull,
		CRAM:             s.CRAM,
		MaxCheckpoints:   s.MaxCheckpoints,
		CheckpointMinStep: s.CheckpointMinStep,
		UseCUDAGraphs:    s.UseCUDAGraphs,
		Host:             s.Host,
		NoJinja:          s.NoJinja,
		ReasoningOff:     s.ReasoningOff,
		MMProjPath:       s.MMProjPath,
		BackendTag:       s.BackendTag,
		BackendIdentity:  backendIdentity,
		BackendPath:      backendPath,
		ChatTemplate:     chatTemplate,
		Reviewer:         reviewer,
		PlanFreeVRAM:     s.PlanFreeVRAM,
		PlannedHostFootprintMB: s.PlannedHostFootprintMB,
	}
	if s.TensorSplit != nil {
		vc.TensorSplit = append([]float64(nil), s.TensorSplit...)
	}
	if s.Draft != nil {
		d := *s.Draft
		vc.Draft = &d
	}
	return vc
}

// rebuildOTStringFromAssignments reconstructs the -ot flag from a saved
// GPU-assignment layout when the exact OTString was not persisted (legacy
// fallback; the modern save always writes OTString).
func rebuildOTStringFromAssignments(vc *VerifiedConfig, caps *detect.Capabilities, numLayers int) string {
	if vc == nil || len(vc.GPUAssignments) == 0 || numLayers <= 0 {
		return ""
	}
	return buildOTStringFromAssignments(vc.GPUAssignments, caps.GPUs, numLayers, vc.BackendTag)
}

// verifiedConfigFreeVRAMStale revalidates the stale-plan free-VRAM guard: a
// config computed under tighter VRAM than the machine now shows would replay
// its pessimism forever. Mirrors cache.go:121-129. It returns false (not
// stale) when the config carries no reading, mirroring the placement cache's
// grandfathering of entries without the field.
func verifiedConfigFreeVRAMStale(vc *VerifiedConfig, caps *detect.Capabilities) (string, bool) {
	if vc == nil || caps == nil || len(vc.PlanFreeVRAM) == 0 {
		return "", false
	}
	for idx, plannedFree := range vc.PlanFreeVRAM {
		for _, g := range caps.GPUs {
			if g.Index == idx && g.VRAMFreeMB() > plannedFree+planFreeVRAMSlackMB {
				return fmt.Sprintf(
					"verified config is stale: GPU%d now has %dMB free but the config was verified against %dMB",
					idx, g.VRAMFreeMB(), plannedFree), true
			}
		}
	}
	return "", false
}

// kvTypeToQuality is the inverse of NormalizeKVType: it recovers a KVQuality
// string from the resolved KVType a config saved. Unknown/missing types map to
// an empty string; the caller's request KV quality still wins downstream.
func kvTypeToQuality(kvType string) string {
	switch kvType {
	case "f16":
		return "high"
	case "q8_0":
		return "mid"
	case "q4_0", "q4_1", "q5_0", "q5_1", "q6_k", "q8_1", "bf16":
		return kvType
	default:
		return ""
	}
}
