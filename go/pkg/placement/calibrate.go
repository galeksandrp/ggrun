package placement

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// CalibrationSchemaVersion bumps whenever the candidate set or scoring changes,
// so a decision measured under older semantics is never applied after an
// upgrade changes what "fastest" means.
const CalibrationSchemaVersion = 5

// CalibrationDecision records which candidate won a measured first-launch
// calibration for one scope, with the numbers that decided it. The winner is
// stored by name and re-derived from the deterministic candidate generator on
// later launches, so the full placement is reproduced exactly rather than
// partially deserialized.
type CalibrationDecision struct {
	SchemaVersion int    `json:"schema_version"`
	ScopeKey      string `json:"scope_key"`
	// ModelBasename is the basename of the model this decision measured, written
	// beside the scope hash so a "clear caches" action can find every decision a
	// model created. The scope key is an opaque hash of model+backend+hardware
	// that cannot be reversed into a model identity, so clearing by model needs
	// this explicit marker. Old decisions without the field stay valid for
	// loading (scope-key validation is unchanged); they just cannot be matched
	// for clearing.
	ModelBasename    string  `json:"model_basename,omitempty"`
	Winner           string  `json:"winner"` // candidate Name, e.g. "default" or "kv-alternate"
	DefaultTPS       float64 `json:"default_tps"`
	DefaultPromptTPS float64 `json:"default_prompt_tps"`
	DefaultScore     float64 `json:"default_score"`
	WinnerTPS        float64 `json:"winner_tps"`
	WinnerPromptTPS  float64 `json:"winner_prompt_tps"`
	WinnerScore      float64 `json:"winner_score"`
	Improvement      float64 `json:"improvement_pct"`
	MeasuredAt       string  `json:"measured_at"`
}

// CalibrationPath returns the cache file for one calibration scope.
func CalibrationPath(cacheDir, scopeKey string) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	return filepath.Join(cacheDir, "calibration", "cal-"+scopeKey+".json")
}

// SaveCalibrationDecision persists a calibration result atomically.
func SaveCalibrationDecision(cacheDir string, d CalibrationDecision) (string, error) {
	if d.SchemaVersion == 0 {
		d.SchemaVersion = CalibrationSchemaVersion
	}
	if d.MeasuredAt == "" {
		d.MeasuredAt = time.Now().UTC().Format(time.RFC3339)
	}
	path := CalibrationPath(cacheDir, d.ScopeKey)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(d, "", "  ")
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

// LoadCalibrationDecision reads a prior calibration for the scope, rejecting
// stale-schema or mismatched keys so an old decision is never silently applied.
func LoadCalibrationDecision(cacheDir, scopeKey string) (*CalibrationDecision, error) {
	path := CalibrationPath(cacheDir, scopeKey)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var d CalibrationDecision
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, err
	}
	if d.SchemaVersion != CalibrationSchemaVersion || d.ScopeKey != scopeKey {
		return nil, fmt.Errorf("calibration decision scope mismatch")
	}
	return &d, nil
}

// DeleteCalibrationDecision removes a cached calibration result for one scope.
// A runtime OOM is evidence the declared winner is unsafe at runtime, so the
// decision must not re-declare it the winner on the next launch.
func DeleteCalibrationDecision(cacheDir, scopeKey string) error {
	path := CalibrationPath(cacheDir, scopeKey)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// DeleteCalibrationDecisionsForModel removes every named calibration decision
// for a model. A promoted candidate can have a different placement/ubatch scope
// from the default key under which its decision was saved; model-scoped cleanup
// is the fail-closed runtime-OOM path that prevents that winner being replayed.
func DeleteCalibrationDecisionsForModel(cacheDir string, model *ModelProfile) error {
	if model == nil {
		return nil
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	want := filepath.Base(model.Path)
	if want == "." || want == "" {
		want = filepath.Base(model.Basename)
	}
	if want == "." || want == "" {
		return nil
	}
	dir := filepath.Join(cacheDir, "calibration")
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		var decision CalibrationDecision
		if json.Unmarshal(data, &decision) != nil || filepath.Base(decision.ModelBasename) != want {
			continue
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

// CalibrationCandidate is one alternative placement to measure at first launch.
// The base strategy (index 0 in the slice returned by CalibrationCandidates) is
// the estimated default; the rest are deliberate, bounded variations. Placement
// alternatives are generated from the same Compute ledger; larger MoE ubatches
// must prove themselves through the launcher's contained allocation preflight.
// Context, slot count, and KV type remain fixed. Offloaded MoE plans may
// additionally challenge an automatic cold microbatch with bounded larger
// values; the base microbatch and whether it was explicit remain in the scope.
type CalibrationCandidate struct {
	Name     string
	Strategy *Strategy
}

// CalibrationCandidates returns the estimated default plus the alternative
// placements worth measuring on this hardware. It returns just the default
// (length 1) — i.e. "nothing to calibrate" — whenever the alternatives collapse
// onto the default or the planner cannot prove they fit:
//
//   - CPU-only: GPU microbatch/placement calibration has no meaning
//   - non-MoE single-GPU: there is only one place for the weights to go
//   - no bounded ubatch exists and no relocation survives the placement ledger
//
// The launcher measures each candidate with the same micro-probe and keeps the
// fastest under the scope key; the default is always candidate 0 so a failed or
// inconclusive calibration degrades to today's behavior.
func CalibrationCandidates(caps *detect.Capabilities, model *ModelProfile, base *Strategy, opts Options) []CalibrationCandidate {
	if caps == nil || model == nil || base == nil {
		return nil
	}
	out := []CalibrationCandidate{{Name: "default", Strategy: base}}
	if base.Type == CPUOnly {
		return out
	}

	switch base.Type {
	case MoEOffload:
		// A cold offloaded-MoE plan commonly lands on ubatch 512 because the
		// whole model cannot fit on one GPU, even when the distributed graph has
		// room for a much larger prefill microbatch. Challenge 1024 and 2048 and
		// measure them. This is never a blind default promotion: the launcher
		// still subjects each candidate to
		// allocation preflight, live benchmarking, lifecycle canaries, and the
		// cached-decision invalidation path.
		if !opts.UBatchSizeExplicit {
			for _, ubatch := range []int{512, 1024, 2048} {
				if ubatch <= base.UBatchSize || ubatch > base.BatchSize {
					continue
				}
				alt := cloneStrategy(base)
				alt.UBatchSize = ubatch
				alt.CheckpointMinStep = checkpointMinStep(model, ubatch)
				out = append(out, CalibrationCandidate{Name: fmt.Sprintf("ubatch-%d", ubatch), Strategy: alt})
			}
		}

		if len(caps.GPUs) < 2 {
			return out
		}
		// KV-alternate: move the KV cache between GPU and CPU while keeping the
		// expert policy. Changing KV placement changes both VRAM expert capacity
		// and host RAM, so this must be a fresh Compute pass. Flipping the field on
		// a cloned strategy produced candidates that had never passed either
		// memory ledger.
		altKV := "cpu"
		if base.KVPlacement == "cpu" {
			altKV = "gpu"
		}
		altOpts := calibrationBaseOptions(opts, base)
		altOpts.KVPlacement = altKV
		if alt, err := Compute(caps, model, altOpts); err == nil && alt != nil && alt.Type == MoEOffload && alt.KVPlacement == altKV {
			out = append(out, CalibrationCandidate{Name: "kv-alternate", Strategy: alt})
		}
	case MultiGPUDense:
		// Dense on multiple GPUs has exactly one real choice: which GPU owns the
		// output head and the largest split share. The default weights ownership
		// by bandwidth; try the VRAM-weighted inverse only when the fastest GPU is
		// not also the roomiest, which is the case where the estimate is most
		// likely to be wrong about end-to-end speed.
		if alt := invertDenseSplit(base); alt != nil {
			out = append(out, CalibrationCandidate{Name: "split-inverted", Strategy: alt})
		}
	}
	return out
}

func calibrationBaseOptions(opts Options, base *Strategy) Options {
	alt := opts
	alt.SkipPlacementCache = true
	alt.CacheFile = ""
	if base == nil {
		return alt
	}
	if base.ContextSize > 0 {
		alt.ContextSize = base.ContextSize
	}
	if base.Parallel > 0 {
		alt.Parallel = base.Parallel
	}
	if base.BatchSize > 0 {
		alt.BatchSize = base.BatchSize
	}
	if base.UBatchSize > 0 {
		alt.UBatchSize = base.UBatchSize
	}
	if base.KVQuality != "" {
		alt.KVQuality = base.KVQuality
	}
	return alt
}

// cloneStrategy deep-copies the placement-affecting fields of a strategy so a
// candidate can diverge without aliasing the base's slices.
func cloneStrategy(s *Strategy) *Strategy {
	if s == nil {
		return nil
	}
	c := *s
	if s.TensorSplit != nil {
		c.TensorSplit = append([]float64(nil), s.TensorSplit...)
	}
	if s.Draft != nil {
		d := *s.Draft
		c.Draft = &d
	}
	if s.CompanionPlacements != nil {
		c.CompanionPlacements = append([]CompanionPlacement(nil), s.CompanionPlacements...)
	}
	return &c
}

// invertDenseSplit returns a copy of a multi-GPU dense strategy with the split
// ratio reversed across devices, or nil when there is nothing meaningful to
// invert (single share, or an already-symmetric split).
func invertDenseSplit(s *Strategy) *Strategy {
	if s == nil || len(s.TensorSplit) < 2 {
		return nil
	}
	reversed := make([]float64, len(s.TensorSplit))
	for i, v := range s.TensorSplit {
		reversed[len(s.TensorSplit)-1-i] = v
	}
	// An inversion that reproduces the same split is not a distinct candidate.
	same := true
	for i := range reversed {
		if reversed[i] != s.TensorSplit[i] {
			same = false
			break
		}
	}
	if same {
		return nil
	}
	c := cloneStrategy(s)
	c.TensorSplit = reversed
	// The output head follows the largest share, so the main GPU moves too.
	if len(reversed) > 0 {
		best := 0
		for i, v := range reversed {
			if v > reversed[best] {
				best = i
			}
		}
		c.MainGPU = best
	}
	return c
}

// CalibrationScopeKey identifies the exact launch shape a calibration decision
// is valid for. Any field change — model, backend, hardware, workload, or the
// runtime knobs a candidate shares with the default — must produce a different
// key, or a stale decision could be applied to a launch it never measured.
type CalibrationScopeKey struct {
	ModelIdentity   string
	BackendIdentity string
	HardwareID      string
	WorkloadProfile string
	ContextSize     int
	Parallel        int
	BatchSize       int
	UBatchSize      int
	UBatchExplicit  bool
	KVQuality       string
	KVType          string
	GPUSet          string
	BasePlacement   string
	MemoryPolicy    string
	// SWAFull is part of the scope because it changes the KV allocation without
	// changing anything else the key already carries (see placement.go:6640-6643):
	// a config verified at the default window sizing is not a config for a
	// --swa-full launch, and reusing it hands the backend a layout that cannot
	// fit.
	SWAFull bool
	// ChatTemplate is part of the scope because a forced chat-template override
	// (catalog Entry.Name) is a different serving contract — the flags emitted
	// for --chat-template-file differ from an auto-matched template.
	ChatTemplate string
}

// NewCalibrationScopeKey builds the key from the same identity sources the
// speculative performance profile uses, so a calibration decision and a spec
// decision for the same launch can never disagree about what launch they
// describe.
func NewCalibrationScopeKey(model *ModelProfile, caps *detect.Capabilities, opts Options, base *Strategy) CalibrationScopeKey {
	contextSize := opts.ContextSize
	parallel := opts.Parallel
	batchSize := opts.BatchSize
	ubatchSize := opts.UBatchSize
	kvQuality := opts.KVQuality
	kvType := ""
	basePlacement := ""
	if base != nil {
		if base.ContextSize > 0 {
			contextSize = base.ContextSize
		}
		if base.Parallel > 0 {
			parallel = base.Parallel
		}
		if base.BatchSize > 0 {
			batchSize = base.BatchSize
		}
		if base.UBatchSize > 0 {
			ubatchSize = base.UBatchSize
		}
		if base.KVQuality != "" {
			kvQuality = base.KVQuality
		}
		kvType = base.KVType
		basePlacement = specHash(
			string(base.Type), base.KVPlacement, fmt.Sprintf("%t", base.MMap),
			fmt.Sprintf("%d", base.NCPUMoE), splitCompactKey(base.TensorSplit), base.OTString,
		)
	}
	return CalibrationScopeKey{
		ModelIdentity:   SpecTargetIdentity(model),
		BackendIdentity: opts.BackendIdentity,
		HardwareID:      SpecHardwareIdentity(caps),
		WorkloadProfile: opts.WorkloadProfile,
		ContextSize:     contextSize,
		Parallel:        parallel,
		BatchSize:       batchSize,
		UBatchSize:      ubatchSize,
		UBatchExplicit:  opts.UBatchSizeExplicit,
		KVQuality:       kvQuality,
		KVType:          kvType,
		GPUSet:          specGPUSet(opts.GPUs),
		BasePlacement:   basePlacement,
		MemoryPolicy: fmt.Sprintf("ram=%d,pct=%d,ram-head=%d,vram-head=%d,no-mmap=%t,force-mmap=%t,measured-buffers=%t",
			opts.RamBudgetMB, opts.RAMLimitPercent, opts.RAMHeadroomMB, opts.VRAMHeadroomMB, opts.NoMMap, opts.ForceMMap, opts.RequireMeasuredBuffers),
		SWAFull:      opts.SWAFull,
		ChatTemplate: opts.ChatTemplate,
	}
}

// String renders the key as a stable, opaque hash for use as a cache filename.
func (k CalibrationScopeKey) String() string {
	return specHash(
		k.ModelIdentity, k.BackendIdentity, k.HardwareID, k.WorkloadProfile,
		fmt.Sprintf("%d", k.ContextSize), fmt.Sprintf("%d", k.Parallel),
		fmt.Sprintf("%d", k.BatchSize), fmt.Sprintf("%d", k.UBatchSize),
		fmt.Sprintf("%t", k.UBatchExplicit),
		k.KVQuality, k.KVType, k.GPUSet, k.BasePlacement, k.MemoryPolicy,
		fmt.Sprintf("%t", k.SWAFull), k.ChatTemplate,
	)
}
