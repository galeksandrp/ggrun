package placement

import (
	"crypto/md5"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// VRAM and compute sizing constants
const (
	vramOverheadPercent = 130  // model size * this / 100 = estimated VRAM needed
	computePerGPUMB     = 512  // legacy; non-MoE single-GPU sizing only
	computeFloorMB      = 1024 // cited llama.cpp compute floor; CUDA overhead measured separately
	minCramMB           = 512
	// cramQuantumMB coarsens the derived prompt-cache budget, which is computed
	// from free RAM and therefore drifts by a few MiB between otherwise
	// identical runs (9753, 9752, 9742 on three consecutive launches here).
	// Prompt-cache precision below a few hundred MiB buys nothing, so the drift
	// is pure churn in every artifact keyed on the launch command.
	//
	// This narrows that window; it does not close it. Free RAM still crosses a
	// boundary now and then, which is why nothing that must survive a re-plan
	// may key on a derived value -- see recoveryLaunchIdentity, which excludes
	// this flag outright.
	cramQuantumMB = 512
	// Hybrid and recurrent prompt restoration needs several retained checkpoints:
	// one rolling checkpoint is erased before a branch older than the newest
	// boundary can restore it. Reserve conservatively per checkpoint and slot,
	// then bound the controller-owned policy to a useful 4..16 range.
	hybridCheckpointReservePerSlotMB = 128
	hybridCheckpointMinimum          = 4
	hybridCheckpointMaximum          = 16
	// Cards below this fraction of the fastest PCIe link are too slow to own
	// regular layer slots in MoE layer-split mode, but can still be useful as
	// expert-only VRAM when one or more whole expert layers fit.
	// Cards at one-third or less of the fastest link are better used as whole-
	// expert storage. On the measured x16/x4/x1 host, letting the x4 card own a
	// tiny dense split consumed an 8.7 GiB graph and left no complete-expert
	// capacity; expert-only placement produced the fastest stable service.
	expertOnlyMaxBandwidthRatio = 0.33
)

// StrategyType selects how the model is placed.
type StrategyType string

const (
	CPUOnly         StrategyType = "cpu_only"
	SingleGPU       StrategyType = "single_gpu"
	MultiGPUDense   StrategyType = "multi_gpu_dense"
	DenseCPUOffload StrategyType = "dense_cpu_offload"
	MoEOffload      StrategyType = "moe_offload"
)

// Strategy represents the computed placement for a model on this hardware.
type Strategy struct {
	Type        StrategyType `json:"type"`
	ContextSize int          `json:"context_size"`
	GPULayers   int          `json:"gpu_layers"` // always 999; llama-server decides
	TensorSplit []float64    `json:"tensor_split,omitempty"`
	SplitMode   string       `json:"split_mode,omitempty"` // graph, layer, row
	MainGPU     int          `json:"main_gpu,omitempty"`
	KVPlacement string       `json:"kv_placement"`        // gpu, cpu, auto
	KVQuality   string       `json:"kv_quality"`          // high, mid, low
	KVType      string       `json:"kv_type"`             // f16, q8_0, q4_0
	NCPUMoE     int          `json:"n_cpu_moe,omitempty"` // for MoE offload
	OTString    string       `json:"ot_string,omitempty"` // -ot override-tensor flags
	// PlacementCachePath is the keyed file where this exact placement (model +
	// ctx + ubatch + kv placement + backend + GPU set) is persisted, so a load
	// that lands right — or is corrected by OOM-recovery — is reused next launch
	// instead of re-predicted. Runtime-only; not part of the serialized strategy.
	PlacementCachePath string `json:"-"`
	// PlacementCacheHit distinguishes a validated last-known-good placement from
	// a fresh estimate. Post-load auto calibration must not bypass that proof and
	// immediately promote an unverified denser layout; users can still request a
	// forced calibration explicitly.
	PlacementCacheHit bool `json:"-"`
	// BackendSupportsFit is true when the backend's --help lists -fit/--fit.
	// Some backends accept an explicit on/off value while older compatible forks
	// expose a simple boolean --fit. Keep the dialect separate: sending
	// "--fit off" to a boolean-only fork makes it reject the whole launch.
	BackendSupportsFit   bool `json:"-"`
	BackendFitTakesValue bool `json:"-"`
	// BackendSupportsKVOffload reports whether the backend accepts the positive
	// --kv-offload switch. GPU KV is the backend default, so a backend which
	// only exposes --no-kv-offload must receive no positive flag at all.
	BackendSupportsKVOffload bool `json:"-"`
	// BackendCheckpointMinStepFlag is the help-probed spelling for checkpoint
	// spacing. Mainline and ik_llama expose the same control under different
	// names; emitting mainline's spelling to ik makes the server exit in argument
	// parsing before the memory preflight can even begin.
	BackendCheckpointMinStepFlag string `json:"-"`
	MMap                         bool   `json:"mmap"`
	MMapRequired                 bool   `json:"mmap_required,omitempty"`
	MLock                        bool   `json:"mlock"`
	FlashAttention               bool   `json:"flash_attention"`
	Threads                      int    `json:"threads"`
	BatchSize                    int    `json:"batch_size"`
	UBatchSize                   int    `json:"ubatch_size"`
	BackendTag                   string `json:"backend_tag,omitempty"` // "llama" or "ik_llama"
	IsMoE                        bool   `json:"is_moe"`
	ReasoningOff                 bool   `json:"reasoning_off"`      // default off for OpenAI compat
	NoJinja                      bool   `json:"no_jinja,omitempty"` // omit `--jinja`; model template unparseable by the backend Jinja engine
	ThreadsBatch                 int    `json:"threads_batch"`      // batch threads (logical cores)
	Parallel                     int    `json:"parallel,omitempty"`
	CRAM                         int    `json:"cram,omitempty"` // prompt cache MB
	// VRAMLedger explains, per GPU, how the expert placement was arrived at.
	// Every question of the form "why is that card not full?" has been answered
	// by reverse-engineering this arithmetic from the emitted -ot string and
	// nvidia-smi; printing it turns an afternoon into one line.
	VRAMLedger []GPULedgerEntry `json:"vram_ledger,omitempty"`
	// SWAFull records that this launch gives sliding-window layers the full
	// context. It belongs on the strategy rather than only on the request
	// because it changes the KV size every later stage prices, not just the
	// flag that gets emitted.
	SWAFull bool `json:"swa_full,omitempty"`
	// ContextAllocationMB/Evidence expose exact, launch-scoped backend evidence
	// when it replaced the architecture estimate. They are diagnostic fields;
	// the emitted flags remain unchanged.
	ContextAllocationMB       int    `json:"context_allocation_mb,omitempty"`
	ContextAllocationEvidence string `json:"context_allocation_evidence,omitempty"`
	// PlanFreeVRAM is the free VRAM each GPU showed when this plan was
	// computed, by CUDA index. The placement cache stores it so a plan made
	// under transiently tight VRAM -- a previous server mid-teardown -- is
	// recomputed once the VRAM is back, instead of replaying its pessimism.
	PlanFreeVRAM map[int]int `json:"plan_free_vram,omitempty"`
	// MeasuredPromptCacheBPT and PromptCacheTypicalTokens carry a measured
	// prompt-cache cost into CRAM sizing. Zero means nothing was measured, and
	// the derived budget stands.
	MeasuredPromptCacheBPT   float64 `json:"measured_prompt_cache_bpt,omitempty"`
	PromptCacheTypicalTokens int     `json:"prompt_cache_typical_tokens,omitempty"`
	// MeasuredPromptCacheEntryMB is what one saved conversation occupied, taken
	// from the backend rather than derived. It is preferred over the per-token
	// cost because it needs no assumption about how long a typical turn is.
	MeasuredPromptCacheEntryMB float64 `json:"measured_prompt_cache_entry_mb,omitempty"`
	MaxCheckpoints             int     `json:"max_checkpoints,omitempty"`
	// PlannedHostFootprintMB is the host RAM this plan has already spoken for:
	// CPU-resident expert weights, token embeddings, any CPU-side KV, and the
	// runtime overhead — the same quantity the mmap decision is made against.
	// The prompt cache must be sized against what is left after it, not against
	// free RAM, and not against `totalSize - totalVRAM`: that older subtraction
	// charged the model for every byte of VRAM installed rather than the bytes
	// the weights actually occupy there, and VRAM also carries KV, compute
	// buffers and CUDA overhead. On this 3-GPU host serving DeepSeek-V4 it put
	// 88793 MiB of host weights at 73128 MiB, so the cache was handed ~15 GiB of
	// RAM that was never free and the backend was OOM-killed by its own memory
	// scope while saving prompt state. Zero means the plan did not derive one and
	// the fallback stands.
	PlannedHostFootprintMB int `json:"-"`
	// CheckpointMinStep is the token spacing between context checkpoints. It
	// decides whether one can sit at the point a later turn resumes from.
	CheckpointMinStep int          `json:"checkpoint_min_step,omitempty"`
	UseCUDAGraphs     bool         `json:"use_cuda_graphs,omitempty"`
	Host              string       `json:"host,omitempty"`        // listen address
	HasSSM            bool         `json:"has_ssm,omitempty"`     // SSM/Mamba hybrid flag
	Draft             *DraftConfig `json:"draft,omitempty"`       // speculative decoding config
	MMProjPath        string       `json:"mmproj_path,omitempty"` // vision projector GGUF
	MMProjSizeMB      int          `json:"-"`                     // mmproj VRAM on primary GPU
	// CompanionPlacements records where each Options.Companions reservation was
	// placed: GPU index, or -1 for CPU. Runtime-only; the launcher starts the
	// helper on the device the planner chose.
	CompanionPlacements []CompanionPlacement `json:"-"`
}

// CompanionPlacement is the resolved seat for one CompanionReservation.
type CompanionPlacement struct {
	Name string `json:"name"`
	GPU  int    `json:"gpu"` // physical GPU index, or -1 for CPU
}

// ModelProfile describes the GGUF model.
type ModelProfile struct {
	Path                   string  `json:"path"`
	Name                   string  `json:"name,omitempty"`         // GGUF metadata: model name
	Basename               string  `json:"basename,omitempty"`     // GGUF metadata: model basename
	QuantizedBy            string  `json:"quantized_by,omitempty"` // GGUF metadata: quantizer (e.g. "unsloth")
	SizeBytes              int64   `json:"size_bytes"`
	TotalSizeMB            int     `json:"total_size_mb"` // includes multi-part shards
	NumLayers              int     `json:"num_layers"`
	NumParams              int64   `json:"num_params"`
	IsMoE                  bool    `json:"is_moe"`
	NumExperts             int     `json:"num_experts,omitempty"`
	ContextSize            int     `json:"context_size"`
	HiddenSize             int     `json:"hidden_size"`
	HeadCount              int     `json:"head_count"`
	HeadCountKV            int     `json:"head_count_kv"`
	KeyLength              int     `json:"key_length"`
	ValueLength            int     `json:"value_length"`
	VocabSize              int     `json:"vocab_size"`
	TokenizerModel         string  `json:"tokenizer_model,omitempty"`
	TokenizerPre           string  `json:"tokenizer_pre,omitempty"`
	TokenizerHash          string  `json:"tokenizer_hash,omitempty"`
	QuantType              string  `json:"quant_type"`
	ExpertBytes            int64   `json:"expert_bytes"`
	NonExpertBytes         int64   `json:"non_expert_bytes"`
	TokenEmbdBytes         int64   `json:"token_embd_bytes,omitempty"` // subset of NonExpertBytes; always host-resident
	OutputBytes            int64   `json:"output_bytes,omitempty"`     // subset of NonExpertBytes; whole on the last split device
	ShexpBytes             int64   `json:"shexp_bytes,omitempty"`      // subset of ExpertBytes; stays on the layer's device
	ExpertAuxBytes         int64   `json:"expert_aux_bytes,omitempty"` // subset of NonExpertBytes; routing tensors that follow expert pins
	ExpertLayerBytes       []int64 `json:"expert_layer_bytes,omitempty"`
	RoutedExpertLayerBytes []int64 `json:"routed_expert_layer_bytes,omitempty"`
	ShexpLayerBytes        []int64 `json:"shexp_layer_bytes,omitempty"`
	ExpertAuxLayerBytes    []int64 `json:"expert_aux_layer_bytes,omitempty"`
	NonExpertLayerBytes    []int64 `json:"non_expert_layer_bytes,omitempty"`
	Fused                  int     `json:"fused"`
	EmbeddingLength        int     `json:"embedding_length"`
	FeedForwardLength      int     `json:"feed_forward_length"`
	KVLoraRank             int     `json:"kv_lora_rank"`
	QLoraRank              int     `json:"q_lora_rank"`
	RopeDim                int     `json:"rope_dim"`
	KeyLengthMLA           int     `json:"key_length_mla"`
	ValueLengthMLA         int     `json:"value_length_mla"`
	HasSSM                 int     `json:"has_ssm"`
	SlidingWindow          int     `json:"sliding_window"`
	FullAttnInterval       int     `json:"full_attn_interval"`
	HasShexp               int     `json:"has_shexp"`
	CTXTrain               int     `json:"ctx_train"`
	ModelArch              string  `json:"model_arch"`
	ExpertUsedCount        int     `json:"expert_used_count,omitempty"`
	// MeasuredKVBytesPerTok maps a KV cache type (e.g. "q8_0") to the KV cache
	// bytes-per-token that llama.cpp ACTUALLY allocated on a previous launch of
	// this model, read back from the backend log. It is the ground truth for
	// compressed-attention models (MLA/CSA-HCA/SWA) where the GGUF formula is
	// unreliable; computeKVTotalMB prefers it over the formula when present.
	MeasuredKVBytesPerTok map[string]float64 `json:"-"`
	// MeasuredKVGeometry is the layout llama.cpp actually built, per KV type:
	// how many layers attend over the whole context, how many over a window,
	// and how deep that window's cache is. It supersedes the rate above because
	// it predicts a context the model was never launched at, and because it is
	// the only form that can price --swa-full.
	MeasuredKVGeometry        map[string]KVGeometry `json:"-"`
	ExpertFF                  int                   `json:"expert_ff,omitempty"`
	ExpertSharedFF            int                   `json:"expert_shared_ff,omitempty"`
	ExpertSharedCount         int                   `json:"expert_shared_count,omitempty"`
	ExpertSharedCountInferred bool                  `json:"expert_shared_count_inferred,omitempty"`
	LeadingDense              int                   `json:"leading_dense,omitempty"`
	LeadingDenseInferred      bool                  `json:"leading_dense_inferred,omitempty"`
	NextNPredictLayers        int                   `json:"nextn_predict_layers,omitempty"`
}

// GPULedgerEntry is one card's expert-placement arithmetic.
type GPULedgerEntry struct {
	GPU int `json:"gpu"`
	// FreeMB is what the hardware scan reported, already net of any companion
	// reservation seated on this card.
	FreeMB int `json:"free_mb"`
	// FixedMB is everything charged before experts: CUDA overhead, the compute
	// buffer (or its floor), measured runtime graph growth, and on a split owner
	// the attention and norm weights plus its KV share.
	FixedMB int `json:"fixed_mb"`
	// RoomMB is FreeMB minus FixedMB: the budget experts were packed into.
	RoomMB int `json:"room_mb"`
	// ExpertLayers is how many whole layers fit in RoomMB.
	ExpertLayers int `json:"expert_layers"`
	// StrandedMB is what RoomMB could not spend, because an expert layer is
	// indivisible. It is the honest answer to "why is that card not full?".
	StrandedMB int `json:"stranded_mb"`
	// ExpertOnly marks a card that carries pinned expert tensors but no
	// attention, norms or KV.
	ExpertOnly bool `json:"expert_only,omitempty"`
}

// checkpointMinStep is the token spacing between context checkpoints.
//
// A checkpoint covers the SWA cache's own depth, which llama.cpp builds as
// n_swa + n_ubatch padded to 256:
//
//	size_swa = GGML_PAD(min(size_base, hparams.n_swa + n_ubatch), 256)
//
// and the checkpoint search rejects any candidate whose pos_max overshoots the
// point a later turn resumes from. Spacing wider than that depth leaves gaps a
// resume point can fall into, and llama.cpp's 8192 default is far wider.
//
// Measured on a 16k prompt at the default: a 92% prefix match was found, the
// only two checkpoints were [14728,15751] and [15240,16263], the resume point
// was ~14906, and both were rejected for overshooting -- one by 845 tokens.
// All 164,358 prompt tokens were re-processed.
//
// Deriving the spacing from the depth makes consecutive checkpoints contiguous.
// Note this is not 2*n_swa: that matched only because this project runs
// -ub 512 with a 512 window, and would be a third too coarse at -ub 256.
//
// Returns 0 when the model has no sliding window, leaving llama.cpp's default
// in place -- without SWA the checkpoint depth is not derived this way.
func checkpointMinStep(model *ModelProfile, ubatch int) int {
	if model == nil || model.SlidingWindow <= 0 {
		return 0
	}
	if ubatch <= 0 {
		// llama.cpp's own default physical batch.
		ubatch = 512
	}
	step := model.SlidingWindow + ubatch
	// The backend pads the cache to a 256-cell boundary; match it so the
	// spacing cannot be marginally wider than the depth it is tiling.
	if rem := step % 256; rem != 0 {
		step -= rem
	}
	if step < checkpointMinStepFloor {
		step = checkpointMinStepFloor
	}
	return step
}

// checkpointMinStepFloor keeps a tiny sliding window from producing a
// checkpoint every few hundred tokens: each one costs real host memory (20.257
// MiB measured on this project) and --ctx-checkpoints caps how many survive, so
// spacing far below the depth buys nothing but eviction churn.
const checkpointMinStepFloor = 512

// CompanionReservation reserves VRAM on one GPU for a co-launched helper model
// (e.g. the Claude Auto safety reviewer) inside the same placement ledger as the
// main model, instead of placing the helper with a separate heuristic and
// re-detecting hardware afterwards. VRAMMB is the helper's total on-device
// footprint (weights + KV + compute), measured after its first real launch and
// conservative before that.
type CompanionReservation struct {
	Name   string // cache/scope key component, e.g. "claude-auto-reviewer"
	VRAMMB int    // total on-device footprint to reserve; <=0 disables the reservation
	// GPUPreference orders candidate GPUs by physical device index, most-preferred
	// first (e.g. slowest-link first, main GPU last). Empty means the planner
	// chooses: it tries the least-bandwidth GPU that still fits, keeping the main
	// GPU free for the model.
	GPUPreference []int
	// AllowCPU permits placing the companion on CPU when no GPU fits. When false,
	// a companion that fits nowhere is an error, not a silent CPU fallback.
	AllowCPU bool
}

// Options allows user overrides.
type Options struct {
	ContextSize int
	KVPlacement string // auto, gpu, cpu
	KVQuality   string // high, mid, low
	GPUs        []int  // restrict to specific GPUs
	// Companions reserves VRAM for co-launched helper models before the main
	// model's split is computed, so the main plan packs around real helper
	// footprints instead of discovering them after launch.
	Companions      []CompanionReservation
	CPUMode         bool
	RamBudgetMB     int
	RAMLimitPercent int // whole-host RAM utilisation target; fixed RamBudgetMB wins
	VRAMHeadroomMB  int // hold back this much total VRAM as a safety margin
	RAMHeadroomMB   int // hold back this much system RAM as a safety margin
	// RequireMeasuredBuffers removes cold-start compute/host buffer estimates
	// from authoritative fit decisions. The contained allocation preflight then
	// supplies exact evidence before ggrun permits a real launch.
	RequireMeasuredBuffers bool
	BackendTag             string // "llama" or "ik_llama"
	BackendCacheTag        string // backend identity for probe/cache isolation; defaults to BackendTag
	BackendIdentity        string // exact backend build/commit identity for speculative performance profiles
	SamplingProfile        string // default, greedy, recommended, or a hash of explicit sampling overrides
	// WorkloadProfile scopes placement and probe evidence to scheduler semantics
	// that can change the effective slot, batch, or cache behavior. An empty
	// value preserves the legacy generic-serving cache namespace; callers that
	// select an agent/workflow profile must provide a stable non-empty value.
	WorkloadProfile string
	NoMMap          bool
	ForceMMap       bool
	Parallel        int
	// Threads overrides the CPU thread count, which otherwise follows physical
	// cores. Physical cores is the right default -- MoE experts on CPU are
	// bandwidth-bound, and SMT siblings share the ports that bound them -- but
	// it is a claim worth being able to measure on a given box rather than
	// assume, and it cannot be measured without a way to set it.
	Threads int
	// SWAFull mirrors the backend's --swa-full: every sliding-window layer gets
	// the full context instead of a window-sized cache. It has to reach
	// placement because it is not a small correction -- on Laguna it takes KV
	// from 13.8 GB to 54.0 GB at 1M context, which decides how many expert
	// layers fit. It is the only setting that lets a windowed model reuse a
	// prompt prefix at all, so the cost has to be predictable in advance
	// rather than discovered by an out-of-memory abort.
	SWAFull bool
	// CacheRAMMB overrides the host prompt-cache budget (-cram). The derived
	// value takes a tenth of free RAM capped at 16 GiB, which is blind to what a
	// single entry costs: on this project one conversation's entry measured
	// ~6.5 GiB, so the cache held one and every new conversation evicted the
	// previous one -- 17.5% cache-read against 88.3% for a single-slot server on
	// the same box. Sizing that correctly needs a measurement per model, so the
	// override exists to take it.
	CacheRAMMB int
	// BatchSize and UBatchSize are explicit launcher requests. A positive value
	// must be accounted for before placement is chosen; treating it as a late
	// backend override can make the emitted server graph exceed the plan.
	BatchSize   int
	UBatchSize  int
	CacheFile   string // path to placement cache for MoE recovery
	CacheDir    string // path to ggrun cache dir (for probes)
	Host        string // listen address (default 127.0.0.1)
	VisionAuto  bool   // auto-detect mmproj for vision
	MMProjPath  string // explicit vision projector GGUF
	SpecMode    string // off, auto, draft, eagle3, dflash, ngram, ngram-mod, ngram-k4v, mtp
	BackendHelp string // llama-server --help output for dialect-specific flags
	// SpecCandidateValidator asks the selected backend to load a proposed
	// companion without allocating model buffers. GGUF metadata establishes
	// target compatibility; this hook establishes runtime compatibility for
	// backend-specific architectures and quant types before ggrun serves them.
	SpecCandidateValidator func(path string) error
	ForceSpecMoE           bool // allow speculative decoding on MoE despite default gate
	ReasoningOff           bool // emit `--reasoning off` (benchmark/tune only; normal serving keeps the model's thinking)
	// SkipPlacementCache disables loading the keyed .place cache for this Compute.
	// Set during a corrective OOM re-plan so it derives fresh from the penalized
	// VRAM instead of reloading the placement that just OOM'd.
	SkipPlacementCache bool
}

// ScopedBackendCacheTag returns the cache/probe namespace for a backend and
// workload. The workload is intentionally part of the opaque hashed cache key:
// a successful generic or parallel-serving launch is not proof that a different
// agent scheduler (with different slots or prefill batching) is safe.
//
// Keep an empty workload unscoped for backward-compatible generic serving. New
// workload-aware callers must pass a non-empty, versioned profile so old cache
// files are never mistaken for validation of the new workload.
func ScopedBackendCacheTag(backendTag, workloadProfile string) string {
	backendTag = strings.TrimSpace(backendTag)
	if backendTag == "" {
		backendTag = "llama"
	}
	workloadProfile = strings.TrimSpace(workloadProfile)
	if workloadProfile == "" {
		return backendTag
	}
	return backendTag + "|workload=" + workloadProfile
}

func SpecWorkloadProfile(base string, draft *DraftConfig) string {
	if draft == nil || draft.Type == DraftNone || draft.Path == "" {
		return base
	}
	identity := strings.Join([]string{
		"spec-v1", string(draft.Type), SpecCompanionIdentity(draft.Path), draft.KVTypeDraft,
		strconv.Itoa(draft.CTXSizeDraft), strconv.Itoa(draft.VRAMMB), strconv.Itoa(draft.DraftGPU),
	}, ":")
	if strings.TrimSpace(base) == "" {
		return identity
	}
	return base + "|" + identity
}

func backendCacheTag(opts Options) string {
	tag := strings.TrimSpace(opts.BackendCacheTag)
	if tag == "" {
		tag = opts.BackendTag
	}
	return ScopedBackendFeatureTag(ScopedBackendCacheTag(tag, opts.WorkloadProfile), opts.SWAFull)
}

// ScopedBackendFeatureTag isolates allocation evidence that changes when a
// backend feature changes without changing the model, context, or KV type.
// --swa-full is the first such feature: on a windowed model it can multiply the
// context allocation while every older probe-cache key field stays identical.
// Always recording both states deliberately invalidates the old, ambiguous
// namespace instead of treating a pre-feature cache entry as evidence for the
// plain state.
func ScopedBackendFeatureTag(backendTag string, swaFull bool) string {
	backendTag = strings.TrimSpace(backendTag)
	if backendTag == "" {
		backendTag = "llama"
	}
	if strings.Contains(backendTag, "|swa-full=") {
		return backendTag
	}
	if !swaFull {
		return backendTag
	}
	return backendTag + "|swa-full=true"
}

// backendFitTakesValue distinguishes current mainline's "--fit [on|off]"
// dialect from forks that expose --fit as a boolean. The latter must not receive
// a value: argparse-style parsing treats "off" as an unknown positional option.
func backendFitTakesValue(help string) bool {
	help = strings.ToLower(help)
	return strings.Contains(help, "--fit [on|off]") ||
		strings.Contains(help, "--fit (on|off)") ||
		strings.Contains(help, "--fit <on|off>")
}

func backendCheckpointMinStepFlag(help, backendTag string) string {
	switch {
	case backendHelpSupports(help, "--checkpoint-min-step"):
		return "--checkpoint-min-step"
	case backendHelpSupports(help, "--ctx-checkpoints-interval"):
		return "--ctx-checkpoints-interval"
	case strings.TrimSpace(help) != "":
		// A successfully collected help surface that advertises neither option
		// must not receive a guessed flag.
		return "-"
	case strings.EqualFold(strings.TrimSpace(backendTag), "ik_llama"):
		// Direct package callers and old cached strategies have no help text.
		// Preserve a dialect-safe fallback for those cases.
		return "--ctx-checkpoints-interval"
	default:
		return "--checkpoint-min-step"
	}
}

// applyCompanionReservations seats each companion helper on one GPU (or CPU when
// allowed and no GPU fits) and returns a capabilities copy whose chosen GPUs show
// the reservation as already-used VRAM. The main model's split is then computed
// against the remaining capacity, so helpers stop being invisible competitors
// discovered only after launch. The input slice is left untouched; callers keep
// their original caps for hardware display while placement sees the reserved view.
func applyCompanionReservations(caps *detect.Capabilities, comps []CompanionReservation) (*detect.Capabilities, []CompanionPlacement, error) {
	if caps == nil {
		return caps, nil, nil
	}
	reserved := *caps
	reserved.GPUs = append([]detect.GPU(nil), caps.GPUs...)
	placements := make([]CompanionPlacement, 0, len(comps))
	for _, comp := range comps {
		if comp.VRAMMB <= 0 {
			continue
		}
		name := comp.Name
		if name == "" {
			name = "companion"
		}
		gpu := chooseCompanionGPU(reserved.GPUs, comp)
		if gpu < 0 && !comp.AllowCPU {
			return nil, nil, fmt.Errorf("companion %s needs %d MB VRAM but no GPU has room", name, comp.VRAMMB)
		}
		if gpu >= 0 {
			for i := range reserved.GPUs {
				if reserved.GPUs[i].Index == gpu {
					reserved.GPUs[i].VRAMUsedMB += comp.VRAMMB
					break
				}
			}
		}
		placements = append(placements, CompanionPlacement{Name: name, GPU: gpu})
	}
	return &reserved, placements, nil
}

// chooseCompanionGPU picks the seat for one helper. An explicit GPUPreference
// order wins. Otherwise it takes the least-bandwidth GPU that still fits the
// reservation, keeping fast-link and main GPUs free for the model's weights and
// expert traffic. Returns the physical GPU index, or -1 when none fits.
func chooseCompanionGPU(gpus []detect.GPU, comp CompanionReservation) int {
	if len(gpus) == 0 {
		return -1
	}
	fits := func(idx int) bool {
		for _, g := range gpus {
			if g.Index == idx {
				return g.VRAMFreeMB() >= comp.VRAMMB
			}
		}
		return false
	}
	seen := map[int]bool{}
	for _, idx := range comp.GPUPreference {
		seen[idx] = true
		if fits(idx) {
			return idx
		}
	}
	if len(comp.GPUPreference) > 0 {
		return -1
	}
	candidates := append([]detect.GPU(nil), gpus...)
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].BandwidthMBps != candidates[j].BandwidthMBps {
			return candidates[i].BandwidthMBps < candidates[j].BandwidthMBps
		}
		if candidates[i].VRAMTotalMB != candidates[j].VRAMTotalMB {
			return candidates[i].VRAMTotalMB < candidates[j].VRAMTotalMB
		}
		return candidates[i].Index < candidates[j].Index
	})
	for _, g := range candidates {
		if g.VRAMFreeMB() >= comp.VRAMMB {
			return g.Index
		}
	}
	return -1
}

func speculativeCompanionReservation(draft *DraftConfig) (CompanionReservation, bool) {
	if draft == nil || draft.Type == DraftNone || draft.Path == "" || draft.VRAMMB <= 0 || draft.DraftGPU < 0 {
		return CompanionReservation{}, false
	}
	return CompanionReservation{
		Name:          "spec-" + string(draft.Type),
		VRAMMB:        draft.VRAMMB,
		GPUPreference: []int{draft.DraftGPU},
	}, true
}

func applyRAMBudget(caps *detect.Capabilities, budgetMB int) *detect.Capabilities {
	if budgetMB <= 0 || caps == nil {
		return caps
	}
	capped := *caps
	capped.RAM.FreeMB = budgetMB
	capped.RAM.TotalMB = budgetMB
	return &capped
}

func applyRAMPolicy(caps *detect.Capabilities, opts Options) *detect.Capabilities {
	if opts.RamBudgetMB > 0 {
		return applyRAMBudget(caps, opts.RamBudgetMB)
	}
	return detect.ApplyRAMLimitPercent(caps, opts.RAMLimitPercent)
}

// Compute builds a Strategy from hardware capabilities and model profile.
func Compute(caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	var err error
	caps, err = restrictGPUs(caps, opts.GPUs)
	if err != nil {
		return nil, err
	}

	caps = applyRAMPolicy(caps, opts)
	caps = detect.ApplyVRAMHeadroom(caps, opts.VRAMHeadroomMB)
	caps = detect.ApplyRAMHeadroom(caps, opts.RAMHeadroomMB)

	// Load any KV cache size llama.cpp measured for this model on a prior launch,
	// so context sizing uses measured truth (exact for compressed attention).
	if model.MeasuredKVBytesPerTok == nil {
		if rates := loadMeasuredKVRates(opts.CacheDir, model); rates != nil {
			model.MeasuredKVBytesPerTok = rates
		}
	}
	if model.MeasuredKVGeometry == nil {
		if g := loadMeasuredKVGeometry(opts.CacheDir, model); g != nil {
			model.MeasuredKVGeometry = g
		}
	}

	resolvedKVQuality, err := resolveKVQuality(model, opts.KVQuality, opts.BackendTag)
	if err != nil {
		return nil, err
	}

	s := &Strategy{
		ContextSize:    opts.ContextSize,
		SWAFull:        opts.SWAFull,
		PlanFreeVRAM:   snapshotPlanFreeVRAM(caps),
		KVPlacement:    opts.KVPlacement,
		KVQuality:      resolvedKVQuality,
		MMap:           opts.ForceMMap || !opts.NoMMap,
		MLock:          false,
		Threads:        caps.CPU.Cores, // physical cores
		ThreadsBatch:   caps.CPU.Cores, // physical cores
		BackendTag:     opts.BackendTag,
		IsMoE:          model.IsMoE,
		GPULayers:      999,
		FlashAttention: true, // finalized against the resolved KV placement before each return
		// Thinking stays ON for normal serving (backend default `--reasoning auto`);
		// only benchmark/tune opt in to `--reasoning off` for clean, fast measurement.
		ReasoningOff: opts.ReasoningOff,
		// Nanbeige's chat template contains a Jinja `raise_exception` macro the
		// backend's Jinja engine cannot parse, so the model 400s on every request
		// when --jinja is passed. The built-in (non-Jinja) template path handles
		// it fine, so nanbeige launches omit --jinja.
		NoJinja: strings.EqualFold(model.ModelArch, "nanbeige"),
		// DeepSeek4 uses non-shiftable recurrent memory even though current GGUFs
		// do not expose the generic SSM metadata bit. Treat it like other hybrid
		// models for context shifting and checkpoint restoration.
		HasSSM: model.HasSSM == 1 || strings.EqualFold(model.ModelArch, "deepseek4"),
		Host:   opts.Host,
		// ggrun sets explicit placement (-ngl/-ot/--tensor-split), so the backend's
		// own auto memory-fitting (-fit) is redundant with this explicit plan.
		// Only value-taking dialects need an explicit "off"; boolean backends are
		// already disabled when the flag is absent.
		BackendSupportsFit:           backendHelpSupports(opts.BackendHelp, "-fit"),
		BackendFitTakesValue:         backendFitTakesValue(opts.BackendHelp),
		BackendSupportsKVOffload:     backendHelpSupports(opts.BackendHelp, "--kv-offload"),
		BackendCheckpointMinStepFlag: backendCheckpointMinStepFlag(opts.BackendHelp, opts.BackendTag),
	}

	if s.ContextSize <= 0 {
		s.ContextSize = defaultContextSize(model, caps)
	}
	if opts.KVPlacement == "" {
		s.KVPlacement = "auto"
	}
	if opts.Parallel > 0 {
		s.Parallel = opts.Parallel
	}
	if opts.Threads > 0 {
		s.Threads = opts.Threads
		s.ThreadsBatch = opts.Threads
	}

	// Vision: use an explicit projector, or auto-detect one when --vision is set.
	if opts.MMProjPath != "" {
		if err := validateMMProj(opts.MMProjPath, model.Name, model.Basename); err != nil {
			return nil, err
		}
		s.MMProjPath = opts.MMProjPath
		if fi, err := os.Stat(opts.MMProjPath); err == nil {
			s.MMProjSizeMB = int(fi.Size() / 1024 / 1024)
		}
	} else if opts.VisionAuto && model.Path != "" {
		if path, err := findOrDownloadMMProj(model.Path, opts.CacheDir, model.Name, model.Basename, model.QuantizedBy); err == nil {
			s.MMProjPath = path
			if fi, err := os.Stat(path); err == nil {
				s.MMProjSizeMB = int(fi.Size() / 1024 / 1024)
			}
		} else {
			fmt.Fprintf(os.Stderr, "[vision] %v\n", err)
		}
	}

	// Total size MB (model + mmproj if vision)
	totalSizeMB := model.TotalSizeMB + s.MMProjSizeMB
	if totalSizeMB <= 0 {
		totalSizeMB = int(model.SizeBytes / 1024 / 1024)
	}

	// KV cache type selection. Besides the friendly high/mid/low presets,
	// users can select one of llama.cpp's supported cache types directly (for
	// example q5_1). That exact type must drive both the memory plan and the
	// emitted server flags; treating it as q8_0 here can make a plan needlessly
	// small, while treating an unknown type optimistically can cause an OOM.
	var kvErr error
	s.KVType, kvErr = NormalizeKVType(s.KVQuality)
	if kvErr != nil {
		return nil, fmt.Errorf("KV cache type: %w", kvErr)
	}

	// Resolve KV placement "auto" → concrete value up-front so every caller and
	// every explicit-context retry sees the same placement policy.
	if s.KVPlacement == "auto" || s.KVPlacement == "" {
		if opts.CPUMode || len(caps.GPUs) == 0 {
			s.KVPlacement = "cpu"
		} else {
			sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
			perGPUOH := plannedPerGPUVRAMOverheadMB(sysProbe, 0, opts)
			kvNeedMB := computeKVTotalMB(model, s.ContextSize, s.KVType, opts.SWAFull)
			s.KVPlacement = resolveAutoKVPlacement(caps, model, totalSizeMB, kvNeedMB, perGPUOH*len(caps.GPUs))
		}
	}

	// Auto-fit context: compute both single-GPU and multi-GPU, pick the larger.
	if opts.ContextSize <= 0 {
		if opts.CPUMode || len(caps.GPUs) == 0 {
			cpuCaps := *caps
			cpuCaps.GPUs = nil
			s.ContextSize, s.KVType = computeAutoContextSize(&cpuCaps, model, totalSizeMB, s.KVType, opts)
		} else {
			sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
			// Per-GPU non-weight VRAM overhead, derived per-component (measured
			// CUDA overhead + compute buffer) — no flat guess, no extra margin.
			perGPUOH := plannedPerGPUVRAMOverheadMB(sysProbe, 0, opts)

			bestFree := 0
			for _, g := range caps.GPUs {
				if g.VRAMFreeMB() > bestFree {
					bestFree = g.VRAMFreeMB()
				}
			}

			// Single-GPU estimate
			singleCtx, singleKV := computeAutoContextSizeSingleGPU(caps, model, totalSizeMB, s.KVType, opts)
			singleKVM := computeKVTotalMB(model, singleCtx, singleKV, opts.SWAFull)
			singleFits := (totalSizeMB+perGPUOH+singleKVM) <= bestFree && singleCtx >= 32768

			// Multi-GPU estimate
			multiCtx, multiKV := computeAutoContextSize(caps, model, totalSizeMB, s.KVType, opts)
			multiKVM := computeKVTotalMB(model, multiCtx, multiKV, opts.SWAFull)
			multiFree := 0
			for _, g := range caps.GPUs {
				multiFree += g.VRAMFreeMB()
			}
			multiFits := (totalSizeMB+perGPUOH*len(caps.GPUs)+multiKVM) <= multiFree && multiCtx >= 32768

			if multiFits && multiCtx > singleCtx {
				s.ContextSize, s.KVType = multiCtx, multiKV
			} else if singleFits {
				s.ContextSize, s.KVType = singleCtx, singleKV
			} else if multiFits {
				s.ContextSize, s.KVType = multiCtx, multiKV
			} else {
				// The model doesn't fit wholly in VRAM (a big MoE offloading experts
				// to CPU). Don't collapse to the 32768/q4_0 floor — size the context
				// by where its KV will actually live. --kv-placement drives it:
				// gpu → VRAM-bounded (safe, experts offload); cpu → RAM-bounded
				// (large window); auto → gpu if it fits, else cpu for a big MoE.
				placement := s.KVPlacement
				if placement == "auto" || placement == "" {
					placement = resolveAutoKVPlacement(caps, model, totalSizeMB, computeKVTotalMB(model, s.ContextSize, s.KVType, opts.SWAFull), perGPUOH*len(caps.GPUs))
				}
				s.KVPlacement = placement
				s.ContextSize, s.KVType = computeAutoContextSizeKVPlacement(caps, model, totalSizeMB, s.KVType, placement, opts)
			}
		}
	}

	// Compute KV cache size
	kvTotalMB := computeKVTotalMB(model, s.ContextSize, s.KVType, opts.SWAFull)

	// Batch sizes based on fit
	bestGPUFree := 0
	for _, g := range caps.GPUs {
		if g.VRAMFreeMB() > bestGPUFree {
			bestGPUFree = g.VRAMFreeMB()
		}
	}

	// Batch tier by exact fit on the best single GPU: model + measured CUDA
	// overhead + KV + that tier's actual compute buffer must fit in VRAM. No
	// percentage guess, no fixed headroom — each tier's compute buffer is the
	// real cost of running that (u)batch (firstLaunchComputeBufMB).
	batchBaseMB := totalSizeMB + measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs)) + kvTotalMB
	compute1024MB := firstLaunchComputeBufMB(model, 1024)
	compute512MB := firstLaunchComputeBufMB(model, 512)
	if opts.RequireMeasuredBuffers {
		compute1024MB = 0
		compute512MB = 0
	}
	switch {
	case batchBaseMB+compute1024MB <= bestGPUFree:
		s.BatchSize, s.UBatchSize = 8192, 1024
	case batchBaseMB+compute512MB <= bestGPUFree:
		s.BatchSize, s.UBatchSize = 4096, 512
	default:
		s.BatchSize, s.UBatchSize = 2048, 512
	}
	// Apply caller-selected values before deriving the placement-cache identity
	// and before sizing MoE buffers. This is intentionally earlier than server
	// argument generation: an explicit -ub changes graph allocation and cannot
	// safely be appended as an unplanned override.
	if opts.BatchSize > 0 {
		s.BatchSize = opts.BatchSize
	}
	if opts.UBatchSize > 0 {
		s.UBatchSize = opts.UBatchSize
	}

	// Resolve external speculation before target placement. Its metadata-derived
	// weights and KV footprint become occupied VRAM in the same ledger, so the
	// target packer cannot assign those bytes a second time.
	if opts.SpecMode != "" && opts.SpecMode != "off" {
		draftOpts := opts
		draftOpts.ContextSize = s.ContextSize
		s.Draft = ComputeDraft(model, caps, draftOpts)
	}
	opts.WorkloadProfile = SpecWorkloadProfile(opts.WorkloadProfile, s.Draft)
	// The backend's exact allocation for this complete launch signature is the
	// authority. Do not turn it into a global bytes/token rate: hybrid models can
	// have several independently-sized cache regions, and auxiliary regions do
	// not necessarily scale with either context or K/V quantization. Exact
	// evidence is consumed only under the key it was measured with.
	kvTotalMB = scopedContextAllocationMB(kvTotalMB, model, s, caps, opts)
	companions := append([]CompanionReservation(nil), opts.Companions...)
	if draftReservation, ok := speculativeCompanionReservation(s.Draft); ok {
		companions = append([]CompanionReservation{draftReservation}, companions...)
	}
	if len(companions) > 0 {
		var compErr error
		caps, s.CompanionPlacements, compErr = applyCompanionReservations(caps, companions)
		if compErr != nil {
			return nil, compErr
		}
	}

	// Persist/reuse this exact placement under a key that includes kv placement,
	// ctx, ubatch, backend, and the GPU set — computed from the now-resolved
	// strategy so the launcher (save) and this load agree byte-for-byte.
	cacheSplitKey := splitCompactKey(s.TensorSplit)
	s.PlacementCachePath = PlacementCachePathFor(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel, cacheSplitKey, opts.SWAFull)
	s.PlacementCachePath = placementCachePathForSpec(s.PlacementCachePath, opts.SpecMode, s.Draft)

	// Try cached placement first (MoE only). Prefer the keyed placement cache
	// (remembers a load that landed right or was OOM-corrected); fall back to a
	// tune-cache file if one was selected.
	placeFile := opts.CacheFile
	if s.PlacementCachePath != "" {
		if _, statErr := os.Stat(s.PlacementCachePath); statErr == nil {
			placeFile = s.PlacementCachePath
		}
	}
	if opts.SkipPlacementCache {
		placeFile = ""
	}
	if placeFile != "" && model.IsMoE {
		cache, err := LoadPlacementCache(placeFile, caps, kvTotalMB)
		cacheMMap, cacheMMapRequired := false, false
		if err == nil && cache != nil {
			var fits bool
			var cacheHostFootprintMB int
			fits, cacheMMap, cacheMMapRequired, cacheHostFootprintMB = cachedMoEHostMemoryFits(caps, model, s, cache, totalSizeMB, kvTotalMB, opts)
			if !fits {
				cache = nil
			} else {
				s.PlannedHostFootprintMB = cacheHostFootprintMB
			}
		}
		if err == nil && cache != nil {
			s.PlacementCacheHit = true
			s.Type = MoEOffload
			s.BatchSize = cache.BatchSize
			s.UBatchSize = cache.UBatchSize
			s.Parallel = cache.Parallel
			// The caller's requested slot count wins over what happened to be
			// cached: parallel shapes slots and compute buffers, not the weight
			// layout the cache exists to remember. Without this, a placement
			// cached by a --parallel 1 CLI run silently stripped Claude Code
			// mode down to a single slot.
			if opts.Parallel > 0 && opts.Parallel != cache.Parallel {
				s.Parallel = opts.Parallel
			}
			if opts.BatchSize > 0 {
				s.BatchSize = opts.BatchSize
			}
			if opts.UBatchSize > 0 {
				s.UBatchSize = opts.UBatchSize
			}
			s.NCPUMoE = cache.NCPUMoE
			// Restore the cached mmap decision, don't reset it: the entry was
			// saved from a load that landed right, and CACHED_MMAP=0 means it
			// loaded resident (--no-mmap). Resetting to mmap-allowed here made
			// a cache hit silently drop --no-mmap and page 100GB+ of experts
			// from disk. An explicit user --no-mmap still always wins.
			s.MMap = cacheMMap
			s.MMapRequired = cacheMMapRequired
			if cache.KVUnified {
				s.KVPlacement = "gpu"
			}
			if len(cache.TensorSplit) > 0 {
				s.TensorSplit = normalizeSplit(cache.TensorSplit)
				s.SplitMode = cache.SplitMode
				if s.SplitMode == "" {
					s.SplitMode = "layer"
				}
			}
			// Prefer the exact cached -ot (preserves sub-layer gate+up pins);
			// fall back to rebuilding from GPU assignments for legacy caches.
			if cache.OTString != "" {
				s.OTString = cache.OTString
			} else if len(cache.GPUAssignments) > 0 {
				otString := buildOTStringFromAssignments(cache.GPUAssignments, caps.GPUs, model.NumLayers, opts.BackendTag)
				if otString != "" {
					s.OTString = otString
				}
			}
			s.FlashAttention = defaultFlashAttention(model, opts, s.KVPlacement)
			// Placement caches intentionally persist only weight placement. Runtime
			// cache policy depends on current free RAM, slot count, and architecture,
			// so recompute it on every launch instead of inheriting the zero-value
			// checkpoint policy from the early cache-hit return.
			LoadMeasuredPromptCache(opts.CacheDir, model, s, backendCacheTag(opts), caps.GPUs)
			s.CRAM, s.MaxCheckpoints = computeCRAM(caps, model, s, totalSizeMB, kvTotalMB)
			s.CheckpointMinStep = checkpointMinStep(model, s.UBatchSize)
			if opts.CacheRAMMB > 0 {
				s.CRAM = opts.CacheRAMMB
			}
			if s.Host == "" {
				s.Host = "127.0.0.1"
			}
			return s, nil
		}
	}

	// Strategy selection
	strategy := chooseStrategy(caps, model, s, totalSizeMB, kvTotalMB, opts)
	s.Type = strategy

	// Vision override: mmproj needs extra VRAM — force multi-GPU
	if s.MMProjPath != "" && strategy == SingleGPU && len(caps.GPUs) > 1 {
		if model.IsMoE {
			strategy = MoEOffload
			s.Type = MoEOffload
		} else {
			strategy = MultiGPUDense
			s.Type = MultiGPUDense
		}
	}

	switch strategy {
	case CPUOnly:
		s, err = buildCPUOnly(s, caps, model, opts)
	case SingleGPU:
		s, err = buildSingleGPU(s, caps, model, totalSizeMB, kvTotalMB, opts)
	case MultiGPUDense:
		s, err = buildMultiGPUDense(s, caps, model, totalSizeMB, kvTotalMB, opts)
	case DenseCPUOffload:
		s, err = buildDenseCPUOffload(s, caps, model, totalSizeMB, kvTotalMB, opts)
	case MoEOffload:
		preUBatch := *s // buildMoEOffload returns (nil, err) on hard failure, losing s.UBatchSize
		s, err = buildMoEOffload(s, caps, model, totalSizeMB, kvTotalMB, opts)
		s, err = maximizeMoEGPUFitByUBatch(&preUBatch, s, err, caps, model, totalSizeMB, kvTotalMB, opts)
		if err != nil && opts.ContextSize <= 0 {
			s, kvTotalMB, err = retryMoEWithLowerAutoContext(&preUBatch, err, caps, model, totalSizeMB, opts)
		}
	}
	if err != nil {
		return nil, err
	}

	// OOM guard: refuse if model+KV+compute don't fit (non-MoE only)
	if strategy != MoEOffload {
		if err := checkMemoryOrDie(caps, model, s, totalSizeMB, kvTotalMB, opts); err != nil {
			return nil, err
		}
	}

	// Compute CRAM (prompt cache)
	LoadMeasuredPromptCache(opts.CacheDir, model, s, backendCacheTag(opts), caps.GPUs)
	cram, maxCheckpoints := computeCRAM(caps, model, s, totalSizeMB, kvTotalMB)
	if opts.CacheRAMMB > 0 {
		cram = opts.CacheRAMMB
	}
	s.CRAM = cram
	s.MaxCheckpoints = maxCheckpoints
	s.CheckpointMinStep = checkpointMinStep(model, s.UBatchSize)

	// Default host
	if s.Host == "" {
		s.Host = "127.0.0.1"
	}

	s.FlashAttention = defaultFlashAttention(model, opts, s.KVPlacement)

	return s, nil
}

// resolveKVQuality applies architecture/backend correctness rules before the
// generic memory-quality policy. DeepSeek-V4's current mainline llama.cpp path
// (CUDA, Vulkan, or Metal) produces incorrect output with compressed KV; a
// configuration that fits but returns garbage must fail before placement or
// launch.
func resolveKVQuality(model *ModelProfile, requested, backendTag string) (string, error) {
	requested = strings.TrimSpace(requested)
	if model != nil && strings.EqualFold(model.ModelArch, "deepseek4") &&
		!strings.EqualFold(backendTag, "ik_llama") {
		if requested != "" && !strings.EqualFold(requested, "auto") && !strings.EqualFold(requested, "mid") {
			kvType, err := NormalizeKVType(requested)
			if err != nil {
				return "", fmt.Errorf("KV cache type: %w", err)
			}
			if kvType != "f16" {
				return "", fmt.Errorf("DeepSeek-V4 on mainline llama.cpp requires f16 KV for correct output; %s is unsupported", kvType)
			}
		}
		return "high", nil
	}
	if model != nil && strings.EqualFold(model.ModelArch, "deepseek4") &&
		strings.EqualFold(backendTag, "ik_llama") &&
		requested != "" && !strings.EqualFold(requested, "auto") {
		kvType, err := NormalizeKVType(requested)
		if err != nil {
			return "", fmt.Errorf("KV cache type: %w", err)
		}
		if kvType != "f16" && kvType != "bf16" && kvType != "q8_0" {
			return "", fmt.Errorf("DeepSeek-V4 on ik_llama supports only f16, bf16, or q8_0 K-cache; %s is unsupported", kvType)
		}
	}
	if requested == "" || strings.EqualFold(requested, "auto") {
		// q8_0 KV cache: near-lossless for architectures without a stricter
		// correctness rule. The fitting logic falls back to q4_0 only when VRAM
		// genuinely cannot hold q8_0.
		return "mid", nil
	}
	return requested, nil
}

// ResolveKVQuality exposes the same backend/model correctness gate used by
// Compute so architecture-aware backend selection can reject a candidate that
// cannot honor the requested KV profile before committing to it.
func ResolveKVQuality(model *ModelProfile, requested, backendTag string) (string, error) {
	return resolveKVQuality(model, requested, backendTag)
}

// Target placement differs when a separate speculative model reserves VRAM.
// Keep those successful placements away from the faster spec-off cache (and
// vice versa); otherwise launching DFlash once could permanently CPU-offload
// extra target experts even on later non-speculative launches.
func placementCachePathForSpec(path, mode string, draft *DraftConfig) string {
	mode = normalizeSpecMode(mode)
	if path == "" || mode == "off" {
		return path
	}
	ext := filepath.Ext(path)
	base := strings.TrimSuffix(path, ext)
	suffix := sanitizeFilename(mode)
	if draft != nil && draft.Type != DraftNone {
		identity := strings.Join([]string{
			string(draft.Type), SpecCompanionIdentity(draft.Path), draft.KVTypeDraft,
			strconv.Itoa(draft.CTXSizeDraft), strconv.Itoa(draft.VRAMMB), strconv.Itoa(draft.DraftGPU),
		}, ":")
		suffix += "-" + md5Hash12(identity)
	}
	return base + "-spec-" + suffix + ext
}

func placementCachePathForSpecMode(path, mode string) string {
	return placementCachePathForSpec(path, mode, nil)
}

// restrictGPUs filters caps.GPUs to the user-selected device indices (--gpus).
// Devices are renumbered from 0 because the launcher restricts visibility via
// CUDA_VISIBLE_DEVICES / GGML_VK_VISIBLE_DEVICES, so the backend enumerates
// only the selected devices starting at index 0.
func restrictGPUs(caps *detect.Capabilities, want []int) (*detect.Capabilities, error) {
	if caps == nil || len(want) == 0 || len(caps.GPUs) == 0 {
		return caps, nil
	}
	wanted := make(map[int]bool, len(want))
	for _, idx := range want {
		wanted[idx] = true
	}
	filtered := *caps
	filtered.GPUs = nil
	for _, g := range caps.GPUs {
		if wanted[g.Index] {
			gg := g
			gg.Index = len(filtered.GPUs)
			filtered.GPUs = append(filtered.GPUs, gg)
		}
	}
	if len(filtered.GPUs) == 0 {
		return nil, fmt.Errorf("--gpus %v matches no detected GPU (have %d GPUs)", want, len(caps.GPUs))
	}
	return &filtered, nil
}

// chooseStrategy selects the placement strategy from hardware and model size.
func chooseStrategy(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int, opts Options) StrategyType {
	numGPUs := len(caps.GPUs)

	if opts.CPUMode || numGPUs == 0 {
		return CPUOnly
	}

	// Load measured CUDA overhead. Missing probe data is unknown and contributes 0;
	// preflight/startup OOM recording supplies measured data for later launches.
	cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

	// Load model probe for compute buffer
	computeBufMB := computeFloorMB // 1024 default
	if opts.RequireMeasuredBuffers {
		computeBufMB = 0
	}
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel)
	if pc != nil && pc.ComputeBufMB > 0 {
		computeBufMB = pc.ComputeBufMB
	}

	// Single GPU: model + overhead fits in best GPU
	// Use FREE VRAM (desktop/compositor uses some VRAM)
	bestFreeVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMFreeMB() > bestFreeVRAM {
			bestFreeVRAM = g.VRAMFreeMB()
		}
	}
	// Use measured overhead: model weights + CUDA overhead + compute buffer + KV.
	// Do not subtract a static reserve from free VRAM here; free VRAM is already
	// the measured allocator-visible capacity after resident desktop/process usage.
	singleGPUNeeded := totalSizeMB + cudaOverheadMB + computeBufMB + kvTotalMB
	if singleGPUNeeded <= bestFreeVRAM {
		return SingleGPU
	}

	// Multi-GPU dense: model fits across ALL GPUs (sum of FREE VRAM)
	if !model.IsMoE {
		totalFreeVRAM := 0
		for _, g := range caps.GPUs {
			totalFreeVRAM += g.VRAMFreeMB()
		}
		// Use measured overhead per GPU: model + (cudaOverhead + computeBuf) * numGPUs + KV
		vramNeeded := totalSizeMB + (cudaOverheadMB+computeBufMB)*numGPUs + kvTotalMB
		if vramNeeded <= totalFreeVRAM {
			return MultiGPUDense
		}
	}

	// MoE expert offload
	if model.IsMoE {
		return MoEOffload
	}

	// Dense model with CPU spill
	return DenseCPUOffload
}

func buildCPUOnly(s *Strategy, caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	s.GPULayers = 0
	s.MMap = !opts.NoMMap
	s.BatchSize = 512
	s.UBatchSize = 256
	return s, nil
}

func buildSingleGPU(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	gpuOrder := orderGPUsByBandwidth(caps.GPUs)
	if len(gpuOrder) == 0 {
		return nil, fmt.Errorf("single-GPU strategy requires a GPU")
	}

	cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))
	computeBufMB := computeFloorMB
	if opts.RequireMeasuredBuffers {
		computeBufMB = 0
	}
	if pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel); pc != nil {
		if pc.ComputeBufMB > 0 {
			computeBufMB = pc.ComputeBufMB
		}
	}
	neededMB := totalSizeMB + cudaOverheadMB + computeBufMB + kvTotalMB
	for _, mainIdx := range gpuOrder {
		if caps.GPUs[mainIdx].VRAMFreeMB() >= neededMB {
			s.MainGPU = caps.GPUs[mainIdx].Index
			return s, nil
		}
	}
	return nil, fmt.Errorf("single-GPU strategy no longer fits any selected GPU (need %d MiB)", neededMB)
}

func buildMultiGPUDense(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)

	// Load measured CUDA overhead. Missing probe data is unknown and contributes 0.
	cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

	// Load model probe for compute buffer (same as MoE path)
	probeHit := false
	computeBufMB := computeFloorMB // 1024 default
	if opts.RequireMeasuredBuffers {
		computeBufMB = 0
	}
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel)
	if pc != nil && pc.ComputeBufMB > 0 {
		probeHit = true
		computeBufMB = pc.ComputeBufMB
	}

	// Per-layer costs
	weightPerLayerMB := totalSizeMB / model.NumLayers
	if weightPerLayerMB <= 0 {
		weightPerLayerMB = 1
	}
	kvPerLayerMB := kvTotalMB / model.NumLayers
	if kvPerLayerMB < 1 && kvTotalMB > 0 {
		kvPerLayerMB = 1
	}

	// KV-first GPU reserve: VRAM-proportional (KV reads are VRAM-local)
	gpuKVReserveMB := kvReserveByBandwidth(kvTotalMB, caps.GPUs, seqRange(numGPUs), kvPerLayerMB)

	// Tensor-split: proportional to free VRAM only.
	// llama-server distributes BOTH model weights AND KV cache by this ratio.
	// Using effective free (subtracting KV reserve) causes OOM because
	// llama-server puts KV back proportionally to the split anyway.
	gpuOrder := orderGPUsByBandwidth(caps.GPUs)
	split := make([]float64, numGPUs)
	totalFree := 0.0
	for _, g := range caps.GPUs {
		totalFree += float64(g.VRAMFreeMB())
	}
	if totalFree > 0 {
		for _, gi := range gpuOrder {
			free := float64(caps.GPUs[gi].VRAMFreeMB())
			if gi == gpuOrder[0] && s.MMProjSizeMB > 0 {
				free -= float64(s.MMProjSizeMB)
				if free < 0 {
					free = 0
				}
			}
			split[gi] = free / totalFree
		}
	}
	s.TensorSplit = normalizeSplit(split)

	// Find smallest GPU subset that fits the model
	// Use effective capacity (free - overhead) not just total VRAM
	gpuOrderBW := orderGPUsByBandwidth(caps.GPUs)
	bestGPUCount := numGPUs
	for n := 2; n <= numGPUs; n++ {
		subsetCapacity := 0
		for j := 0; j < n; j++ {
			gi := gpuOrderBW[j]
			g := caps.GPUs[gi]
			overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
			effective := g.VRAMFreeMB() - overhead
			if effective < 0 {
				effective = 0
			}
			subsetCapacity += effective
		}
		modelWeightMB := totalSizeMB + kvTotalMB/2 // model weights + partial KV overhead
		if modelWeightMB <= subsetCapacity {
			bestGPUCount = n
			break
		}
	}

	// Zero out GPUs not in the selected subset
	if bestGPUCount < numGPUs {
		for idx := bestGPUCount; idx < numGPUs; idx++ {
			gi := gpuOrderBW[idx]
			split[gi] = 0
		}
		s.TensorSplit = normalizeSplit(split)
	}

	// Layer split is the portable default for heterogeneous GPUs. The tensor
	// split path uses NCCL collectives during graph construction and can abort
	// before health on systems without working peer access (observed with an
	// Ampere PCIe x1 + Ada PCIe x4 pair). Row split is also unsafe for some GQA
	// models. Layer split avoids both failure modes and is supported by mainline,
	// ik_llama, and the registered forks.
	s.SplitMode = "layer"

	_ = probeHit // used for logging/debugging

	return s, nil
}

func buildDenseCPUOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	if s.BackendSupportsFit {
		// Unlike the wholly resident paths, dense CPU offload does not have an
		// explicit layer plan. Leave n_gpu_layers and tensor_split unset so
		// llama.cpp can measure real tensor sizes and place the maximum safe
		// number of layers. Supplying -ngl 999 or a tensor split makes its fit
		// pass abort and then OOM during the real load.
		s.TensorSplit = nil
		s.SplitMode = ""
		s.MMap = false
		return s, nil
	}

	numGPUs := len(caps.GPUs)
	if numGPUs > 1 {
		// Load measured CUDA overhead. Missing probe data is unknown and contributes 0.
		cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

		// Load model probe for compute buffer
		computeBufMB := computeFloorMB
		if opts.RequireMeasuredBuffers {
			computeBufMB = 0
		}
		pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel)
		if pc != nil && pc.ComputeBufMB > 0 {
			computeBufMB = pc.ComputeBufMB
		}

		// KV per layer for reserve
		kvPerLayerMB := kvTotalMB / model.NumLayers
		if kvPerLayerMB < 1 && kvTotalMB > 0 {
			kvPerLayerMB = 1
		}

		// KV-first GPU reserve: weighted by VRAM * PCIe bandwidth
		gpuKVReserveMB := kvReserveByBandwidth(kvTotalMB, caps.GPUs, nil, kvPerLayerMB)

		// Tensor split proportional to effective free VRAM * bandwidth
		split := make([]float64, numGPUs)
		gpuOrder := orderGPUsByBandwidth(caps.GPUs)
		totalWeighted := 0.0
		for idx := 0; idx < numGPUs; idx++ {
			gi := gpuOrder[idx]
			g := caps.GPUs[gi]
			overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
			effective := g.VRAMFreeMB() - overhead
			if effective < 0 {
				effective = 0
			}
			bw := float64(g.BandwidthMBps)
			if bw <= 0 {
				bw = 1.0
			}
			totalWeighted += float64(effective) * bw
		}
		for idx := 0; idx < numGPUs; idx++ {
			gi := gpuOrder[idx]
			g := caps.GPUs[gi]
			overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
			effective := g.VRAMFreeMB() - overhead
			if effective < 0 {
				effective = 0
			}
			bw := float64(g.BandwidthMBps)
			if bw <= 0 {
				bw = 1.0
			}
			if totalWeighted > 0 {
				split[gi] = float64(effective) * bw / totalWeighted
			}
		}
		s.TensorSplit = normalizeSplit(split)

		// Keep the same portable heterogeneous-GPU policy as the fully resident
		// dense path. Tensor/row splits can require peer collectives that are not
		// available on many consumer multi-GPU topologies.
		s.SplitMode = "layer"
	}
	s.MMap = false
	return s, nil
}

// buildMoEOffload computes a fully specified multi-GPU MoE plan: tensor split
// for non-expert/KV tensors plus override-tensor pins for expert tensors.
// mmapCanPageCPUExperts reports whether the selected backend leaves CPU-offloaded
// expert tensors file-backed. ik_llama's CUDA path converts exps=CPU into one
// anonymous CUDA_Host buffer; --mmap, GGML_CUDA_NO_PINNED, and --defer-experts
// do not make that allocation reclaimable.
func mmapCanPageCPUExperts(opts Options) bool {
	return !strings.EqualFold(strings.TrimSpace(opts.BackendTag), "ik_llama")
}

type moeLayerMemoryMB struct {
	routed int
	shared int
	aux    int
}

func (m moeLayerMemoryMB) whole() int { return m.routed + m.shared + m.aux }

// moeLayerMemoryCosts returns file-anchored costs when the GGUF parser exposed
// every routed layer. The fallback preserves compatibility with old parser
// output and synthetic ModelProfiles used by callers, while staying
// conservative through per-layer MiB ceiling.
func moeLayerMemoryCosts(model *ModelProfile, start, count int) ([]moeLayerMemoryMB, bool) {
	if model == nil || count <= 0 {
		return nil, false
	}
	end := start + count
	exact := start >= 0 && end <= len(model.RoutedExpertLayerBytes)
	costs := make([]moeLayerMemoryMB, count)
	if exact {
		for i := range costs {
			layer := start + i
			costs[i] = moeLayerMemoryMB{
				routed: bytesToMiBCeil(model.RoutedExpertLayerBytes[layer]),
				shared: layerBytesMiBCeil(model.ShexpLayerBytes, layer),
				aux:    layerBytesMiBCeil(model.ExpertAuxLayerBytes, layer),
			}
			if costs[i].routed <= 0 {
				exact = false
				break
			}
		}
	}
	if exact {
		return costs, true
	}

	routedTotal := bytesToMiBCeil(model.ExpertBytes) - bytesToMiBCeil(model.ShexpBytes)
	if routedTotal < 0 {
		routedTotal = 0
	}
	routed := ceilDivInt(routedTotal, count)
	shared := ceilDivInt(bytesToMiBCeil(model.ShexpBytes), count)
	aux := ceilDivInt(bytesToMiBCeil(model.ExpertAuxBytes), count)
	for i := range costs {
		costs[i] = moeLayerMemoryMB{routed: routed, shared: shared, aux: aux}
	}
	return costs, false
}

func layerBytesMiBCeil(values []int64, layer int) int {
	if layer < 0 || layer >= len(values) {
		return 0
	}
	return bytesToMiBCeil(values[layer])
}

func sumRoutedLayerMB(costs []moeLayerMemoryMB, start int) int {
	if start < 0 {
		start = 0
	}
	total := 0
	for i := start; i < len(costs); i++ {
		total += costs[i].routed
	}
	return total
}

// mmapFreesCPUExperts reports whether leaving CPU-side expert tensors
// file-backed actually makes them reclaimable. This is not a property of the
// flag; it is a property of the loader.
//
// Measured 2026-08-07 on this machine. Under ik_llama, MiniMax-M3's CPU experts
// land in anonymous memory whether or not --mmap is passed — the mmap and
// resident rows' cgroups matched to the byte (anon 113363 MB, file 449 MB), so
// the flag changed nothing at all. ik_llama's mmap fast path
// (src/llama.cpp:4457) is taken only when the buffer type is the default-CPU or
// plain-CPU buffer type; with "-ot ...exps=CPU" the expert tensors match neither
// and fall through to an ordinary allocation. Mainline llama.cpp maps them and
// reports CPU_Mapped buffers instead, which is why DeepSeek-V4 under the same
// flags shows anon 1.9 GB against file 113 GB.
func mmapFreesCPUExperts(backendTag string) bool {
	return !strings.EqualFold(strings.TrimSpace(backendTag), "ik_llama")
}

// hostFootprintForCache reports how much host RAM a plan denies the prompt
// cache. With the weights resident that is the whole footprint. Under mmap the
// expert bytes are clean, file-backed pages the kernel evicts under pressure,
// so only the anonymous working set — runtime overhead plus any CPU-side KV —
// is genuinely unavailable; charging the cache for reclaimable pages there
// would disable it on exactly the hosts that page happily today.
//
// That reclaimability is conditional on the backend actually mapping those
// bytes. Where it does not, charging only the working set understates the plan
// by the entire expert set: for MiniMax-M3 it predicted 458 MB against a real
// 113 GB, a 250x error that the memory preflight then had to catch at launch.
func hostFootprintForCache(residentMB, workingSetMB int, mmap bool, backendTag string) int {
	footprint := residentMB
	if mmap && mmapFreesCPUExperts(backendTag) {
		footprint = workingSetMB
	}
	if footprint < 0 {
		return 0
	}
	return footprint
}

func cachedMoEHostMemoryFits(caps *detect.Capabilities, model *ModelProfile, s *Strategy, cache *CacheEntry, totalSizeMB, kvTotalMB int, opts Options) (fits, mmap, mmapRequired bool, hostFootprintMB int) {
	if caps == nil || model == nil || s == nil || cache == nil {
		return false, false, false, 0
	}
	moeLayers := model.NumLayers - model.LeadingDense
	if moeLayers <= 0 {
		moeLayers = model.NumLayers
	}
	moeStart := model.NumLayers - moeLayers
	layerCosts, _ := moeLayerMemoryCosts(model, moeStart, moeLayers)
	cpuStart := moeLayers - cache.NCPUMoE
	cpuExpertMB := sumRoutedLayerMB(layerCosts, cpuStart)
	cpuKVMB := 0
	if !cache.KVUnified && s.KVPlacement == "cpu" {
		cpuKVMB = kvTotalMB
	}
	runtimeMB := plannedRAMRuntimeOverheadMB(caps, model, cache.UBatchSize, totalSizeMB, opts)
	tokenEmbdMB := bytesToMiBCeil(model.TokenEmbdBytes)
	residentMB := cpuExpertMB + cpuKVMB + runtimeMB + tokenEmbdMB
	residentFits := residentMB <= caps.RAM.FreeMB

	mmap = cache.MMap
	if opts.NoMMap {
		mmap = false
	} else if opts.ForceMMap {
		mmap = true
	}
	workingSetFloor := runtimeMB + cpuKVMB
	if residentFits {
		return true, mmap, false, hostFootprintForCache(residentMB, workingSetFloor, mmap, opts.BackendTag)
	}
	if !mmap || !mmapCanPageCPUExperts(opts) {
		return false, mmap, false, 0
	}
	if workingSetFloor > caps.RAM.FreeMB {
		return false, mmap, false, 0
	}
	return true, true, true, hostFootprintForCache(residentMB, workingSetFloor, true, opts.BackendTag)
}

func buildMoEOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)
	if numGPUs == 0 {
		return buildCPUOnly(s, caps, model, opts)
	}
	if model.NumLayers <= 0 {
		return nil, fmt.Errorf("MoE placement requires model layer count")
	}

	gpuOrder := orderGPUsByBandwidth(caps.GPUs)
	s.MainGPU = caps.GPUs[gpuOrder[0]].Index
	if numGPUs > 1 {
		s.SplitMode = "layer"
	}

	moeStartLayer := model.LeadingDense
	if moeStartLayer < 0 || moeStartLayer >= model.NumLayers {
		moeStartLayer = 0
	}
	moeLayerCount := model.NumLayers - moeStartLayer
	if moeLayerCount <= 0 {
		moeLayerCount = model.NumLayers
		moeStartLayer = 0
	}

	// Per-layer costs
	expertTotalMB := bytesToMiBCeil(model.ExpertBytes)
	if expertTotalMB <= 0 {
		expertTotalMB = totalSizeMB * 90 / 100
	}
	nonExpertTotalMB := bytesToMiBCeil(model.NonExpertBytes)
	if nonExpertTotalMB <= 0 {
		nonExpertTotalMB = totalSizeMB - expertTotalMB
	}
	if nonExpertTotalMB < 0 {
		nonExpertTotalMB = 0
	}
	// Input embeddings never leave host memory (llama.cpp keeps the input layer
	// on CPU), so they must not be charged against per-GPU VRAM budgets; they
	// are counted on the RAM side below instead.
	tokenEmbdMB := bytesToMiBCeil(model.TokenEmbdBytes)
	if tokenEmbdMB > 0 && tokenEmbdMB < nonExpertTotalMB {
		nonExpertTotalMB -= tokenEmbdMB
	} else {
		tokenEmbdMB = 0
	}
	// The output head is not pro-rata: llama.cpp assigns it to the device that
	// owns the last of its n_layer+1 split slots. Charging it proportionally
	// once OOM'd the smallest GPU (output.weight ~1GB landed whole on the 4070
	// while placement had budgeted an 0.11 share of it).
	outputMB := bytesToMiBCeil(model.OutputBytes)
	if outputMB > 0 && outputMB < nonExpertTotalMB {
		nonExpertTotalMB -= outputMB
	} else {
		outputMB = 0
	}
	// Shared experts ride with their layer's owning device — the exps=CPU
	// catch-all does not match "_shexp", so even CPU-offloaded layers keep
	// their shared expert in VRAM. GPU whole-layer pins already include shexp
	// in expertPerLayerMB; the CPU side must exclude it.
	shexpTotalMB := bytesToMiBCeil(model.ShexpBytes)
	if shexpTotalMB < 0 || shexpTotalMB >= expertTotalMB {
		shexpTotalMB = 0
	}
	layerCosts, exactLayerCosts := moeLayerMemoryCosts(model, moeStartLayer, moeLayerCount)
	if len(layerCosts) > 0 && layerCosts[0].whole() == 0 {
		routedFallback := ceilDivInt(expertTotalMB-shexpTotalMB, moeLayerCount)
		sharedFallback := ceilDivInt(shexpTotalMB, moeLayerCount)
		auxFallback := ceilDivInt(bytesToMiBCeil(model.ExpertAuxBytes), moeLayerCount)
		for i := range layerCosts {
			layerCosts[i] = moeLayerMemoryMB{routed: routedFallback, shared: sharedFallback, aux: auxFallback}
		}
	}
	expertPerLayerMB := 0
	for _, cost := range layerCosts {
		if cost.whole() > expertPerLayerMB {
			expertPerLayerMB = cost.whole()
		}
	}
	if expertPerLayerMB <= 0 {
		expertPerLayerMB = 1
	}
	if os.Getenv("GGRUN_TRACE_PLACEMENT") != "" {
		fmt.Fprintf(os.Stderr, "[trace] model expertTotalMB=%d nonExpertTotalMB=%d moeLayerCount=%d expertPerLayerMB=%d exact=%v totalSize=%d shexp=%d\n",
			expertTotalMB, nonExpertTotalMB, moeLayerCount, expertPerLayerMB, exactLayerCosts, totalSizeMB, shexpTotalMB)
		for i, c := range layerCosts {
			if i < 8 || i >= moeLayerCount-2 || c.whole() > 2700 {
				fmt.Fprintf(os.Stderr, "[trace] layer[%d] routed=%d shared=%d aux=%d whole=%d\n", moeStartLayer+i, c.routed, c.shared, c.aux, c.whole())
			}
		}
	}
	// RAM (and gate+up chunk) cost of a CPU-offloaded expert layer: routed
	// experts only, shared expert stays on the owning GPU.
	expertCPUPerLayerMB := ceilDivInt(expertTotalMB-shexpTotalMB, moeLayerCount)
	if expertCPUPerLayerMB <= 0 {
		expertCPUPerLayerMB = expertPerLayerMB
	}
	nonExpertForGPU := nonExpertTotalMB
	if nonExpertForGPU < 0 {
		nonExpertForGPU = 0
	}
	nonExpertPerLayerMB := ceilDivInt(nonExpertForGPU, model.NumLayers)
	if nonExpertPerLayerMB <= 0 {
		nonExpertPerLayerMB = 1
	}

	// Load measured CUDA overhead per GPU. Missing probe entries are unknown
	// components, not free VRAM; they contribute 0 to whole-layer fitting but
	// block optional remainder squeeze below until a successful launch measures
	// them.
	sysCUDAOverheadByGPU := SystemCUDAOverheadByGPU(opts.CacheDir, caps.GPUs)

	// Load per-model/runtime probe cache. Until a model has completed one launch
	// with these settings, use a first-launch fallback that keeps the main GPU
	// conservative without charging the full prompt graph to every secondary GPU.
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel)
	fixedPerGPU := make([]int, numGPUs)
	expertOnlyFixedPerGPU := make([]int, numGPUs)
	for i, g := range caps.GPUs {
		computeBufMB := firstLaunchComputeBufMBForGPUParallelAtContext(model, s.UBatchSize, s.Parallel, s.ContextSize, i, gpuOrder)
		measuredThisGPU := 0
		if pc != nil {
			measuredThisGPU = pc.ComputeBufByGPU[g.Index]
		}
		aggregate := computeBufMB
		if pc != nil && pc.ComputeBufMB > aggregate {
			aggregate = pc.ComputeBufMB
		}
		expertOnlyComputeMB := expertOnlyComputeReserveMB(aggregate, measuredThisGPU)
		if opts.RequireMeasuredBuffers {
			computeBufMB = 0
			// Under RequireMeasuredBuffers only measured values are trusted, so
			// an expert-only GPU's compute buffer is restored below from the
			// probe cache's per-GPU value — NOT zeroed. Zeroing it made every
			// expert-only GPU budget 0 MiB of compute, over-packing layers that
			// then OOM'd at runtime (the 599.86 MiB GPU1 failure). The reserve
			// stays as the cold-start fallback when nothing is measured.
			expertOnlyComputeMB = 0
		}
		runtimeGrowthMB := 0
		if pc != nil {
			// Use the aggregate (primary) compute buffer for split-owner cost
			// accounting. The per-GPU values in the probe cache are measured
			// for the specific placement that ran (e.g. GPU1 expert-only =
			// 299 MB, GPU0 split-owner = 33 GB at ub=256). When the placement
			// changes between measurement and this computation — which is
			// exactly what a re-plan does — the per-GPU values are wrong: a
			// GPU that was expert-only (299 MB) might become a split-owner
			// (needing the full ~33 GB), or vice versa. The aggregate
			// (primary's compute buffer) is the maximum any split-owner would
			// need, so using it for all GPUs is conservative but correct.
			// Expert-only GPUs use expertOnlyComputeReserveMB which caps at
			// the compute floor, so the aggregate doesn't affect them.
			if pc.ComputeBufMB > 0 {
				computeBufMB = pc.ComputeBufMB
			}
			// Restore the expert-only compute buffer from the measured per-GPU
			// value (the actual expert-only compute cost for THIS device in the
			// placement that was measured). This is the fix for the zeroed
			// expertOnlyComputeMB: without it an expert-only GPU is budgeted 0
			// MiB of compute and gets over-packed with expert layers that OOM
			// at runtime. expertOnlyComputeReserveMB above stays as the
			// cold-start fallback when nothing is measured for this key.
			if measuredPerGPU := pc.ComputeBufByGPU[g.Index]; measuredPerGPU > 0 {
				expertOnlyComputeMB = measuredPerGPU
			}
			// Real long requests can need more than the load-time graph
			// reserve accounts for — llama-server's context-checkpoint state
			// save (tools/server/server-context.cpp create_checkpoint) lives
			// entirely outside llama_context::sched_reserve(), so no no-alloc
			// measurement (ours or llama.cpp's own --fit) ever sees it.
			// Reproduced 2026-07-08: DeepSeek-V4 crashed needing +1679MB on
			// CUDA2 at 16384 real tokens despite a startup reserve sized for
			// the full max context. Once a canary or a real crash measures
			// this growth (RecordRuntimeGraphGrowth/-FromOOM), reserve it here
			// so the next placement for this exact key packs around it
			// instead of rediscovering the deficit by crashing again.
			runtimeGrowthMB = pc.RuntimeGraphGrowthByGPU[g.Index]
		}
		// Cold-start carry: on a key with no measured runtime growth, reserve the
		// largest MEASURED (non-estimated) growth from a related key of the same
		// model/GPU-slot set. Measured-not-static; self-heals once this key
		// records its own value. Runs OUTSIDE the pc!=nil guard so a genuinely
		// cold key (pc==nil) still gets the reserve.
		if runtimeGrowthMB == 0 {
			if related := RelatedModelRuntimeGraphGrowth(opts.CacheDir, model, caps.GPUs, s.Parallel, opts.BackendTag); related != nil {
				runtimeGrowthMB = related[g.Index]
			}
		}
		fixedPerGPU[i] = sysCUDAOverheadByGPU[g.Index] + computeBufMB + runtimeGrowthMB
		expertOnlyFixedPerGPU[i] = sysCUDAOverheadByGPU[g.Index] + expertOnlyComputeMB + runtimeGrowthMB
		if os.Getenv("GGRUN_TRACE_PLACEMENT") != "" {
			fmt.Fprintf(os.Stderr,
				"[trace] gpu%d probeHit=%v ctx=%d ub=%d kvq=%s par=%d free=%d sysOH=%d compute=%d growth=%d fixed=%d\n",
				g.Index, pc != nil, s.ContextSize, s.UBatchSize, s.KVQuality, s.Parallel,
				g.VRAMFreeMB(), sysCUDAOverheadByGPU[g.Index], computeBufMB, runtimeGrowthMB, fixedPerGPU[i])
		}
	}
	expertOnlyGPU := expertOnlySlowGPUs(caps.GPUs, fixedPerGPU, expertOnlyFixedPerGPU, expertPerLayerMB, nonExpertPerLayerMB)

	// Use only GPUs that can carry CUDA/compute overhead plus their emitted
	// tensor-split share of all non-expert weights and KV. The split is
	// bandwidth-weighted: under --split-mode layer, the GPU that owns a
	// layer's non-expert weights also computes that layer — including
	// streaming its CPU-resident experts over PCIe. Weighting the split by
	// measured PCIe bandwidth (not just free VRAM) concentrates layer
	// ownership on the fastest-link GPU, so CPU-expert streaming avoids
	// bottlenecking on a slow PCIe link (e.g. a card stuck at x1).
	// When bandwidth is unknown or uniform across GPUs, this degenerates
	// to free-VRAM-proportional (the previous behaviour).
	// Only reserve GPU VRAM for the KV cache when it actually lives on the GPU.
	// With KV on CPU (big-MoE auto-placement, or --kv-placement cpu) the experts
	// must be free to fill that VRAM instead — otherwise every GPU sits half-empty
	// holding room for a KV cache that's in system RAM.
	gpuKVTotalMB := kvTotalMB
	if s.KVPlacement == "cpu" {
		gpuKVTotalMB = 0
	}
	var split []float64
	var ownedLayers []int
	outputDev := -1

	// nonExpertChargeMB is the exact VRAM a GPU pays for non-expert weights
	// under llama.cpp's real slot assignment: its owned repeating layers times
	// the per-layer non-expert bytes, the shared experts of its owned layers,
	// and the whole output head iff it owns the last slot. This replaces the
	// old pro-rata split share, which spread the ~1GB output head across all
	// GPUs and OOM'd whichever small card actually received it.
	nonExpertChargeMB := func(owned []int, outputDev, gi int) int {
		if exactLayerCosts && len(model.NonExpertLayerBytes) >= model.NumLayers {
			charge := 0
			layerDevices, _ := layerDeviceAssignments(split, model.NumLayers)
			for layer, dev := range layerDevices {
				if dev != gi {
					continue
				}
				charge += layerBytesMiBCeil(model.NonExpertLayerBytes, layer)
				if layer >= moeStartLayer {
					cost := layerCosts[layer-moeStartLayer]
					charge += cost.shared + cost.aux
				}
			}
			layerBytes := int64(0)
			for _, value := range model.NonExpertLayerBytes {
				layerBytes += value
			}
			globalBytes := model.NonExpertBytes - model.TokenEmbdBytes - model.OutputBytes - model.ExpertAuxBytes - layerBytes
			if globalBytes < 0 {
				globalBytes = 0
			}
			if gi == outputDev {
				charge += outputMB + bytesToMiBCeil(globalBytes)
			}
			return charge
		}
		perLayer := float64(nonExpertTotalMB) / float64(model.NumLayers)
		perShexp := 0.0
		if moeLayerCount > 0 {
			perShexp = float64(shexpTotalMB) / float64(moeLayerCount)
		}
		ownedMoELayers := 0
		layerDevices, _ := layerDeviceAssignments(split, model.NumLayers)
		for layer, dev := range layerDevices {
			if layer >= moeStartLayer && dev == gi {
				ownedMoELayers++
			}
		}
		charge := float64(owned[gi])*perLayer + float64(ownedMoELayers)*perShexp
		if gi == outputDev {
			charge += float64(outputMB)
		}
		return int(charge + 0.999)
	}

	used := make([]bool, numGPUs)
	for i, g := range caps.GPUs {
		free := g.VRAMFreeMB()
		fixed := fixedPerGPU[i]
		used[i] = !expertOnlyGPU[i] && free > fixed
		if os.Getenv("GGRUN_TRACE_PLACEMENT") != "" {
			fmt.Fprintf(os.Stderr, "[trace] used init gpu%d expOnly=%v free=%d fixed=%d used=%v\n", i, expertOnlyGPU[i], free, fixed, used[i])
		}
	}
	for {
		rawSplit := make([]float64, numGPUs)
		totalWeighted := 0.0
		for i, g := range caps.GPUs {
			if used[i] {
				// Weight by VRAM available AFTER the fixed per-GPU CUDA/compute
				// reserve, not raw free VRAM, so high-overhead GPUs are not
				// over-allocated.
				avail := float64(g.VRAMFreeMB() - fixedPerGPU[i])
				totalWeighted += avail * gpuSplitWeight(g)
			}
		}
		if totalWeighted <= 0 {
			return nil, fmt.Errorf("Model does not fit on this system: no GPU has free VRAM after CUDA/compute overhead")
		}
		for i, g := range caps.GPUs {
			if used[i] {
				avail := float64(g.VRAMFreeMB() - fixedPerGPU[i])
				rawSplit[i] = avail * gpuSplitWeight(g) / totalWeighted
			}
		}
		split = normalizeSplit(rawSplit)
		ownedLayers, outputDev = layerOwnership(split, model.NumLayers)
		if os.Getenv("GGRUN_TRACE_PLACEMENT") != "" {
			fmt.Fprintf(os.Stderr, "[trace] split iter used=%v rawSplit=%v split=%v owned=%v outputDev=%d\n", used, rawSplit, split, ownedLayers, outputDev)
		}

		removed := false
		for i, g := range caps.GPUs {
			if !used[i] {
				continue
			}
			kvShareMB := ownedShareMB(gpuKVTotalMB, ownedLayers, model.NumLayers, i)
			charge := fixedPerGPU[i] + nonExpertChargeMB(ownedLayers, outputDev, i) + kvShareMB
			if os.Getenv("GGRUN_TRACE_PLACEMENT") != "" {
				fmt.Fprintf(os.Stderr, "[trace] split check gpu%d fixed=%d nonExp=%d kv=%d charge=%d free=%d\n", i, fixedPerGPU[i], nonExpertChargeMB(ownedLayers, outputDev, i), kvShareMB, charge, g.VRAMFreeMB())
			}
			if charge > g.VRAMFreeMB() {
				used[i] = false
				removed = true
				if os.Getenv("GGRUN_TRACE_PLACEMENT") != "" {
					fmt.Fprintf(os.Stderr, "[trace] split REMOVED gpu%d\n", i)
				}
			}
		}
		if !removed {
			break
		}
	}
	// Post-split retrofit: a GPU eliminated from the split (KV or compute buffer
	// exceeded its share) but not already classified expert-only can still hold
	// whole expert layers. Without this, such a GPU sits idle while its VRAM
	// could offload expert layers from CPU RAM. Reclassify it as expert-only
	// when it can fit at least one complete expert layer under the expert-only
	// reserve, so the expert-capacity loop below picks it up. Capacity always uses
	// current free VRAM: existing workloads and accumulated OOM penalties remain
	// real even if the card changes from split-owner to expert-only.
	for i, g := range caps.GPUs {
		if expertOnlyGPU[i] || used[i] || split[i] > 0 {
			continue
		}
		if i >= len(expertOnlyFixedPerGPU) {
			continue
		}
		if g.VRAMFreeMB()-expertOnlyFixedPerGPU[i] >= expertPerLayerMB {
			expertOnlyGPU[i] = true
		}
	}
	if numGPUs > 1 {
		s.TensorSplit = split
	}

	// Per-GPU expert capacity under the exact emitted split. roomMBPer keeps the
	// exact VRAM budget for experts on each GPU so sub-layer packing can later
	// fill the remainder that whole-layer flooring leaves stranded.
	roomMBPer := make([]int, numGPUs)
	for _, gi := range gpuOrder {
		if split[gi] <= 0 && !expertOnlyGPU[gi] {
			continue
		}
		g := caps.GPUs[gi]
		kvShareMB := ownedShareMB(gpuKVTotalMB, ownedLayers, model.NumLayers, gi)
		fixedMB := fixedPerGPU[gi]
		if expertOnlyGPU[gi] {
			fixedMB = expertOnlyFixedPerGPU[gi]
		}
		// Expert-only GPUs run the explicitly pinned expert projection, routing,
		// and shared-expert tensors, but do not receive attention or norm weights:
		// those stay on the regular split owner. Keep the compute-buffer floor;
		// the probe data shows the
		// actual expert-only compute buffer is small (~299 MB for 3 layers on
		// DeepSeek-V4), well under the 1024 MB floor.
		if expertOnlyGPU[gi] && !opts.RequireMeasuredBuffers && fixedMB < computeFloorMB {
			fixedMB = computeFloorMB
		}
		nonExpertCharge := nonExpertChargeMB(ownedLayers, outputDev, gi)
		if expertOnlyGPU[gi] {
			nonExpertCharge = 0 // zero here; per-layer cost accounts for it below
		}
		var roomMB int
		if expertOnlyGPU[gi] {
			// Expert-only GPUs don't own dense layers or KV, but other processes,
			// CUDA overhead, and any observed OOM penalty still consume real VRAM.
			roomMB = g.VRAMFreeMB() - fixedMB
		} else {
			roomMB = g.VRAMFreeMB() - fixedMB - nonExpertCharge - kvShareMB
		}
		if roomMB < 0 {
			roomMB = 0
		}
		roomMBPer[gi] = roomMB
		if os.Getenv("GGRUN_TRACE_PLACEMENT") != "" {
			fmt.Fprintf(os.Stderr, "[trace] room gpu%d expOnly=%v free=%d fixedMB=%d nonExpertCharge=%d kvShareMB=%d room=%d split=%.3f\n",
				gi, expertOnlyGPU[gi], g.VRAMFreeMB(), fixedMB, nonExpertCharge, kvShareMB, roomMB, split[gi])
		}
	}

	// Pack the emitted contiguous expert ranges using each layer's actual
	// routed/shared/router bytes. nonExpertChargeMB initially places shared and
	// router tensors on the regular owner. A cross-device expert pin moves those
	// bytes, so release them from the owner exactly once while charging the full
	// layer to the expert GPU. A pin to the existing owner adds only routed
	// experts because its shared/router bytes are already in that owner's base.
	layersPerGPU := make([]int, numGPUs)
	layerDevices, _ := layerDeviceAssignments(split, model.NumLayers)
	nextMoELayer := 0
	for _, gi := range gpuOrder {
		for nextMoELayer < moeLayerCount {
			absoluteLayer := moeStartLayer + nextMoELayer
			owner := -1
			if absoluteLayer >= 0 && absoluteLayer < len(layerDevices) {
				owner = layerDevices[absoluteLayer]
			}
			cost := layerCosts[nextMoELayer]
			charge := cost.whole()
			if owner == gi {
				charge = cost.routed
			}
			if roomMBPer[gi] < charge {
				break
			}
			roomMBPer[gi] -= charge
			if owner >= 0 && owner < len(roomMBPer) && owner != gi {
				roomMBPer[owner] += cost.shared + cost.aux
			}
			layersPerGPU[gi]++
			nextMoELayer++
		}
	}
	maxGPULayers := nextMoELayer

	// Hard ceilings: _recompute_cpu_layer_caps
	kvPlacementEffective := s.KVPlacement
	if kvPlacementEffective == "auto" {
		kvPlacementEffective = "gpu"
	}
	cpuKVRAMMB := 0
	if kvPlacementEffective == "cpu" {
		cpuKVRAMMB = kvTotalMB
	}

	// Strict ceiling (--no-mmap path). Uses the real measured-overhead
	// formula (not a hand-copied subset) so this ceiling and the later real
	// RAM check (checkMemoryOrDie -> ramRuntimeOverheadMB) can't disagree —
	// a stale copy here was missing the cpuActMB term, letting this ceiling
	// accept a CPU-layer count the real check would then reject.
	ramOverheadPreMB := plannedRAMRuntimeOverheadMB(caps, model, s.UBatchSize, totalSizeMB, opts)
	cpuBudgetStrict := caps.RAM.FreeMB - ramOverheadPreMB - cpuKVRAMMB - tokenEmbdMB
	if cpuBudgetStrict < 0 {
		cpuBudgetStrict = 0
	}
	maxCPULayersStrict := 0
	strictUsedMB := 0
	for i := maxGPULayers; i < len(layerCosts); i++ {
		if strictUsedMB+layerCosts[i].routed > cpuBudgetStrict {
			break
		}
		strictUsedMB += layerCosts[i].routed
		maxCPULayersStrict++
	}

	// Mmap-aware ceiling.
	//
	// Expert weight bytes live in a read-only, file-backed mmap: the kernel can
	// always evict those clean pages under memory pressure and re-fault them
	// from the model file, so they never have to be simultaneously resident.
	// What CANNOT be reclaimed is the runtime's own anonymous memory — CUDA
	// host staging, graph scratch, the mmap page-table, CPU activation buffers
	// (ramOverheadPreMB), and the KV cache (cpuKVRAMMB, continuously
	// read/written, not file-backed) — if that doesn't fit in free RAM, no
	// amount of expert-page eviction helps, and mmap would thrash from the
	// first token. That is the real (measured, not guessed) floor: it replaces
	// both the old flat "8 layers" floor (too lenient — let a model whose
	// working set vastly exceeds even non-reclaimable RAM through) and the
	// "full footprint must fit" floor (too strict — defeated the entire point
	// of mmap, which is to hold LESS than the full footprint resident).
	preWorkingSetFloor := ramOverheadPreMB + cpuKVRAMMB
	mmapReclaimable := mmapCanPageCPUExperts(opts)
	maxCPULayersMMap := 0
	if mmapReclaimable && caps.RAM.FreeMB >= preWorkingSetFloor {
		maxCPULayersMMap = moeLayerCount - maxGPULayers
	}

	maxCPULayers := maxCPULayersStrict
	ceilCPULabel := "resident RAM"
	if opts.NoMMap {
		ceilCPULabel = "strict --no-mmap"
	} else if !mmapReclaimable {
		ceilCPULabel = fmt.Sprintf("resident RAM (%s CPU experts are anonymous)", opts.BackendTag)
	}
	if maxCPULayersStrict < moeLayerCount-maxGPULayers &&
		!opts.NoMMap &&
		maxCPULayersMMap > maxCPULayersStrict {
		maxCPULayers = maxCPULayersMMap
		ceilCPULabel = "mmap (page-cache)"
	}

	// Does-not-fit guard
	if maxCPULayers < moeLayerCount-maxGPULayers {
		gap := moeLayerCount - maxGPULayers - maxCPULayers
		gapVRAMMB := 0
		gapRAMMB := 0
		for i := maxGPULayers + maxCPULayers; i < moeLayerCount; i++ {
			gapVRAMMB += layerCosts[i].whole()
			gapRAMMB += layerCosts[i].routed
		}
		mmapAdvice := "Drop --no-mmap so kernel can page experts on demand"
		if !mmapReclaimable {
			mmapAdvice = fmt.Sprintf("Selected %s backend allocates CPU experts in anonymous CUDA-host memory; mmap cannot reduce this RAM requirement", opts.BackendTag)
		}
		exactCPUExpertMB := sumRoutedLayerMB(layerCosts, maxGPULayers)
		return nil, fmt.Errorf(
			"Model does not fit on this system.\n"+
				"  Required:    %d MoE layers\n"+
				"  GPU cap:     %d layers across %d GPU(s)\n"+
				"  CPU cap:     %d layers (%s)\n"+
				"  RAM ledger:  %dMB limit - %dMB runtime - %dMB CPU KV - %dMB fixed CPU weights = %dMB for resident experts\n"+
				"  Expert cost: %dMB average routed bytes per CPU layer; %dMB exact for %d CPU layers required after GPU placement\n"+
				"  Gap:         %d layers — need ~%dMB more free VRAM or ~%dMB more RAM\n"+
				"\n  Options:\n"+
				"    1. Free VRAM (close other GPU workloads, --gpus to add a card)\n"+
				"    2. %s\n"+
				"    3. Use a smaller quantization or smaller model",
			moeLayerCount, maxGPULayers, numGPUs, maxCPULayers, ceilCPULabel,
			caps.RAM.FreeMB, ramOverheadPreMB, cpuKVRAMMB, tokenEmbdMB, cpuBudgetStrict,
			expertCPUPerLayerMB, exactCPUExpertMB, moeLayerCount-maxGPULayers,
			gap, gapVRAMMB, gapRAMMB, mmapAdvice)
	}

	totalGPULayers := maxGPULayers
	layersCPU := moeLayerCount - totalGPULayers

	// Sub-layer expert packing (GPU squeeze): whole-layer packing floors each
	// GPU's capacity, stranding up to ~expertPerLayerMB of VRAM per card. Fill
	// that remainder with the gate+up projections (2/3 of a layer) of the next
	// CPU-bound layers — down stays on CPU — so stranded VRAM becomes expert
	// residency: more experts on GPU (faster) and that weight leaves system RAM
	// (breathing room). Sizing is exact: gate+up is 2/3 of the file-anchored
	// per-layer ROUTED expert bytes — the shared expert is not part of a chunk
	// (it already lives on the layer's owning GPU).
	gateUpChunkMB := 2 * expertCPUPerLayerMB / 3
	remainderMB := make([]int, numGPUs)
	for gi := range caps.GPUs {
		r := roomMBPer[gi]
		// Expert-only slow-link GPUs are used for whole expert layers only. A
		// partial gate+up pin on a PCIe x1 device creates cross-device expert
		// traffic without making a full layer self-contained.
		if expertOnlyGPU[gi] {
			r = 0
		}
		if r < 0 {
			r = 0
		}
		remainderMB[gi] = r
	}
	var subPins []subExpertPin
	movedOffCPUMB := 0
	if enableAutomaticSubLayerExpertPins &&
		hasMeasuredCUDAOverheadForActiveGPUs(sysCUDAOverheadByGPU, caps.GPUs, split) {
		subPins, movedOffCPUMB = packGateUpChunks(remainderMB, gpuOrder, gateUpChunkMB, moeStartLayer+totalGPULayers, layersCPU)
	}

	// RAM safety check. Sub-layer pins move gate+up off the CPU, so the resident
	// CPU expert footprint shrinks by exactly the packed bytes. CPU layers cost
	// only their routed experts — shared experts stay in VRAM.
	cpuExpertMB := sumRoutedLayerMB(layerCosts, totalGPULayers) - movedOffCPUMB
	if cpuExpertMB < 0 {
		cpuExpertMB = 0
	}

	// Non-weight RAM overhead, derived per-component (see ramRuntimeOverheadMB):
	// CUDA host staging + graph scratch + mmap page table + CPU activation.
	ramOverheadMB := plannedRAMRuntimeOverheadMB(caps, model, s.UBatchSize, totalSizeMB, opts)

	cpuKVMB := 0
	if kvPlacementEffective == "cpu" {
		cpuKVMB = kvTotalMB
	}
	ramNeeded := cpuExpertMB + cpuKVMB + ramOverheadMB + tokenEmbdMB
	ramAvailMB := caps.RAM.FreeMB

	// Mmap decision for MoE — VRAM-aware, no guessed margin.
	//
	// mmap is a question about RAM, not total model size. Expert layers placed
	// on the GPUs live in VRAM and cost zero system RAM, so what matters is the
	// CPU-resident footprint only: CPU-side experts + CPU-side KV + activation/
	// compute overhead (that is exactly ramNeeded). Compare it against the real
	// available RAM (MemAvailable, already net of any user --ram-headroom) with
	// no fudge factor — a guessed % or fixed-MB cushion breaks at scale (trivial
	// on a 1TB box, wasteful as a percentage on a huge model). If the resident
	// footprint fits, load resident (fast, no SSD paging); otherwise mmap while
	// the working set fits. A deliberate reserve is the user's --ram-headroom.
	//
	// The old test keyed off totalSizeMB, which ignored VRAM entirely: a 146GB
	// model with ~40GB of experts on the GPUs (leaving ~100GB on CPU, well under
	// 122GB RAM) was forced onto mmap and paged from SSD for no reason.
	if opts.NoMMap {
		s.MMap = false
	} else if ramNeeded > ramAvailMB {
		// The CPU-resident expert bytes counted in ramNeeded are clean,
		// file-backed mmap pages — evictable and re-fault-able from the model
		// file, so they don't have to all be resident at once. What must
		// actually fit is the runtime's non-reclaimable anonymous memory: KV
		// cache (continuously read/written) plus compute/activation overhead.
		// (Same reasoning as preWorkingSetFloor above; keep both in sync.)
		workingSetFloor := ramOverheadMB + cpuKVMB
		if mmapReclaimable && ramAvailMB >= workingSetFloor {
			s.MMap = true
			s.MMapRequired = true
		} else if !mmapReclaimable {
			return nil, fmt.Errorf("Model does not fit on this system: %s CPU expert offload needs %dMB resident RAM but the configured limit is %dMB; mmap cannot reclaim anonymous CUDA-host buffers", opts.BackendTag, ramNeeded, ramAvailMB)
		}
	} else if opts.ForceMMap {
		s.MMap = true
	} else {
		s.MMap = false
	}

	// Hand the same footprint to the prompt cache that the mmap decision was
	// just made against, so the cache is sized against RAM that will really be
	// free rather than against RAM the weights are about to take.
	s.PlannedHostFootprintMB = hostFootprintForCache(ramNeeded, ramOverheadMB+cpuKVMB, s.MMap, opts.BackendTag)
	if os.Getenv("GGRUN_TRACE_PLACEMENT") != "" {
		fmt.Fprintf(os.Stderr, "[trace] host footprint=%d (cpuExperts=%d tokenEmbd=%d cpuKV=%d overhead=%d mmap=%v) freeRAM=%d\n",
			s.PlannedHostFootprintMB, cpuExpertMB, tokenEmbdMB, cpuKVMB, ramOverheadMB, s.MMap, ramAvailMB)
	}

	// Build -ot string. Always include the exps=CPU catch-all so expert
	// tensors never follow the backend's default layer split by accident.
	// Record the arithmetic before it is thrown away. Stranded VRAM is not a
	// defect on its own -- an expert layer is indivisible, so a card will always
	// end with less than one layer unspent -- but stranded space larger than a
	// layer means something reserved it, and that is worth being able to see.
	s.VRAMLedger = s.VRAMLedger[:0]
	for _, gi := range gpuOrder {
		if gi < 0 || gi >= len(caps.GPUs) {
			continue
		}
		spent := 0
		for i := 0; i < layersPerGPU[gi] && i < len(layerCosts); i++ {
			spent += layerCosts[i].whole()
		}
		if layersPerGPU[gi] > 0 && expertPerLayerMB > 0 {
			spent = layersPerGPU[gi] * expertPerLayerMB
		}
		stranded := roomMBPer[gi] - spent
		if stranded < 0 {
			stranded = 0
		}
		fixed := fixedPerGPU[gi]
		if expertOnlyGPU[gi] {
			fixed = expertOnlyFixedPerGPU[gi]
		}
		s.VRAMLedger = append(s.VRAMLedger, GPULedgerEntry{
			GPU:          caps.GPUs[gi].Index,
			FreeMB:       caps.GPUs[gi].VRAMFreeMB(),
			FixedMB:      fixed,
			RoomMB:       roomMBPer[gi],
			ExpertLayers: layersPerGPU[gi],
			StrandedMB:   stranded,
			ExpertOnly:   expertOnlyGPU[gi],
		})
	}

	otString := buildOTStringWithSubPins(layersPerGPU, subPins, caps.GPUs, gpuOrder, moeStartLayer, opts.BackendTag)
	if otString != "" {
		s.OTString = otString
	}

	// NCPUMoE is a CPU expert-layer count, not an expert count.
	if layersCPU > 0 {
		s.NCPUMoE = layersCPU
	}

	_ = nonExpertPerLayerMB

	return s, nil
}

// UBatchFitLadder are ubatch sizes tried, largest first, when the default
// ubatch's compute buffer leaves literally zero VRAM for any MoE expert
// layer. Flash-attention's compute buffer scales with ubatch at large
// context — measured on DeepSeek-V4 at ctx 1048576, f16 KV, 3-way tensor
// split: ~17-20GB/GPU at ubatch 512 vs ~2.5-2.8GB/GPU at ubatch 64. A
// smaller ubatch trades prefill batch size for GPU-resident experts, which
// is the better trade once every decode step would otherwise stream all
// experts from CPU RAM. Exported so the launch-time preflight (cmd/ggrun)
// can pre-measure every rung via the fit-params oracle before this ladder
// runs, instead of the retry silently falling back to the first-launch
// heuristic for ubatch values that were never actually measured.
var UBatchFitLadder = []int{256, 128, 64}

// Automatic partial expert projection pins are intentionally disabled. Live
// parallel-4 benchmarks showed that gate+up-only packing raised serial decode
// 2.4% but reduced prompt-heavy aggregate throughput 17-19% because every
// partial layer adds a CPU/GPU boundary. Keep the parser/cache support for
// explicit or legacy plans, but the general planner emits complete expert
// layers only until a topology-aware benchmark can prove partial pins faster.
const enableAutomaticSubLayerExpertPins = false

// numGPUsExcluded counts GPUs that got no tensor-split share and no explicit
// expert pins in a multi-GPU MoE placement. A zero split is acceptable for a
// deliberately expert-only slow PCIe GPU; it is only broken when the GPU is
// completely unused.
func numGPUsExcluded(s *Strategy, gpus []detect.GPU, numLayers int) int {
	numGPUs := len(gpus)
	if s == nil || numGPUs <= 1 || len(s.TensorSplit) != numGPUs {
		return 0
	}
	excluded := 0
	for i, v := range s.TensorSplit {
		if otStringUsesDevice(s.OTString, gpus[i].Index) {
			continue
		}
		// A share is only participation if it maps to at least one real layer.
		// A bare v > 0 test counted a 0.01 share of 43 layers as "GPU in use",
		// which is 0 layers and 0 bytes. Measured 2026-08-02: the ladder walked
		// ubatch 512 -> 64 to reach "3/3 GPUs used", and the plan it chose gave
		// CUDA1 a 0.01 share, no expert range, and nothing to do — 8x the prefill
		// batch spent on a cosmetic number.
		if numLayers <= 0 || v*float64(numLayers) < 1 {
			excluded++
		}
	}
	return excluded
}

// maximizeMoEGPUFitByUBatch retries buildMoEOffload at a smaller ubatch when
// the default ubatch produces a placement that wastes GPU capacity: (a)
// starves every expert layer off every GPU (NCPUMoE == total MoE layers),
// (b) fails to fit at all — flash-attention's compute buffer can be large
// enough at big contexts that compute+KV alone exceed a GPU's VRAM before any
// expert weight is even considered, surfacing as a hard "does not fit" error
// rather than a soft zero-experts result — or (c) excludes a whole GPU from
// the tensor split even though other GPUs are in use. base carries the
// pre-call strategy fields (ContextSize, KVPlacement, UBatchSize, ...) since
// a hard failure returns (nil, err) and loses them. Each retry reuses
// buildMoEOffload's own measured-or-heuristic compute accounting for that
// ubatch — no new margin is invented, only a different, real ubatch is
// tried — and stops at the largest ladder rung that measurably improves the
// placement, preserving as much prefill batching as the VRAM allows.
func maximizeMoEGPUFitByUBatch(base, s *Strategy, err error, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	if base == nil || model == nil {
		return s, err
	}
	_, moeCount := moeLayerRange(model)
	if moeCount <= 0 {
		return s, err
	}
	numGPUs := 0
	if caps != nil {
		numGPUs = len(caps.GPUs)
	}
	var gpus []detect.GPU
	if caps != nil {
		gpus = caps.GPUs
	}
	baseExcluded := numGPUsExcluded(s, gpus, model.NumLayers)
	if err == nil && s != nil && s.NCPUMoE < moeCount {
		// The largest ubatch already has a usable whole-layer plan. More GPU
		// experts do not automatically mean a faster service: live MoE tests
		// showed the smaller ubatch/denser placement can lose prompt and parallel
		// throughput. Preserve the largest proven-fit prefill batch.
		//
		// A stranded GPU deliberately does NOT reopen the ladder. Descending a
		// rung costs real prefill throughput on every request for the life of
		// the server, while the GPU it rescues contributes at most a couple of
		// expert layers. Measured 2026-08-02 on DeepSeek-V4-Flash: chasing
		// "3/3 GPUs used" drove ubatch 512 -> 64, an 8x smaller prefill batch
		// (47 t/s -> 16 t/s, a 30k-token prompt going from ~11 to 31 minutes),
		// and the plan it settled on still left the third GPU with no expert
		// range at all. Fit the weights around the ubatch, not the reverse.
		return s, nil
	}
	best, bestErr, bestExcluded := s, err, baseExcluded
	bestNCPUMoE := moeCount + 1
	if err == nil && s != nil {
		bestNCPUMoE = s.NCPUMoE
	}
	for _, ub := range UBatchFitLadder {
		if ub >= base.UBatchSize {
			continue
		}
		cand := *base
		cand.UBatchSize = ub
		if cand.BatchSize < ub {
			cand.BatchSize = ub
		}
		// Allocation evidence is keyed by ubatch. A copied strategy may carry an
		// exact total from the previous rung; never reuse it after changing the
		// signature. The formula remains the conservative fallback when this rung
		// has not yet been measured.
		candidateContextMB := scopedContextAllocationMB(
			computeKVTotalMB(model, cand.ContextSize, cand.KVType, opts.SWAFull),
			model, &cand, caps, opts,
		)
		next, cerr := buildMoEOffload(&cand, caps, model, totalSizeMB, candidateContextMB, opts)
		if cerr != nil {
			continue
		}
		nextExcluded := numGPUsExcluded(next, gpus, model.NumLayers)
		// The base rung produced no usable plan, so any rung that does is an
		// improvement. Take the largest such rung and stop: descending further
		// only buys expert layers, and it buys them with prefill throughput.
		if next.NCPUMoE < moeCount {
			fmt.Fprintf(os.Stderr,
				"[placement] ubatch %d did not yield a usable whole-layer MoE plan — using ubatch %d instead (%d expert layer(s) on GPU, %d/%d GPUs used)\n",
				base.UBatchSize, ub, moeCount-next.NCPUMoE, numGPUs, numGPUs)
			return next, nil
		}
		if next.NCPUMoE >= bestNCPUMoE && nextExcluded >= bestExcluded {
			continue // not an improvement over the best rung found so far
		}
		reason := "left no VRAM for MoE experts"
		if baseExcluded > 0 {
			reason = fmt.Sprintf("stranded %d of %d GPUs with zero tensor-split share", baseExcluded, numGPUs)
		}
		fmt.Fprintf(os.Stderr,
			"[placement] ubatch %d %s at this context/KV type — using ubatch %d instead (%d expert layer(s) on GPU, %d/%d GPUs used)\n",
			base.UBatchSize, reason, ub, moeCount-next.NCPUMoE, numGPUs-nextExcluded, numGPUs)
		best, bestErr, bestNCPUMoE, bestExcluded = next, nil, next.NCPUMoE, nextExcluded
	}
	return best, bestErr
}

func retryMoEWithLowerAutoContext(base *Strategy, originalErr error, caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, opts Options) (*Strategy, int, error) {
	if base == nil || model == nil || base.ContextSize <= 32768 {
		return nil, 0, originalErr
	}
	for _, ctx := range lowerContextRungs(base.ContextSize) {
		cand := *base
		cand.ContextSize = ctx
		kvTotalMB := computeKVTotalMB(model, cand.ContextSize, cand.KVType, opts.SWAFull)
		kvTotalMB = scopedContextAllocationMB(kvTotalMB, model, &cand, caps, opts)
		cand.PlacementCachePath = placementCachePathForStrategy(&cand, caps, model, opts)
		preUBatch := cand
		next, err := buildMoEOffload(&cand, caps, model, totalSizeMB, kvTotalMB, opts)
		next, err = maximizeMoEGPUFitByUBatch(&preUBatch, next, err, caps, model, totalSizeMB, kvTotalMB, opts)
		if err == nil && next != nil {
			fmt.Fprintf(os.Stderr, "[placement] auto context lowered to %d after larger context did not fit\n", ctx)
			return next, kvTotalMB, nil
		}
	}
	return nil, 0, originalErr
}

// scopedContextAllocationMB returns backend-authoritative context allocation
// only for the exact launch signature represented by s. The strategy fields are
// cleared before lookup because callers commonly copy a strategy and then
// change context or ubatch; retaining the prior evidence would turn a scoped
// measurement back into the global extrapolation this mechanism replaces.
func scopedContextAllocationMB(fallback int, model *ModelProfile, s *Strategy, caps *detect.Capabilities, opts Options) int {
	if s == nil {
		return fallback
	}
	s.ContextAllocationMB = 0
	s.ContextAllocationEvidence = ""
	if model == nil || caps == nil {
		return fallback
	}
	allocation, ok := LoadMeasuredAllocation(opts.CacheDir, model, s.ContextSize, s.UBatchSize,
		s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel)
	if !ok {
		return fallback
	}
	s.ContextAllocationMB = allocation.ContextTotalMB
	s.ContextAllocationEvidence = allocation.Evidence
	return allocation.ContextTotalMB
}

func lowerContextRungs(ctx int) []int {
	rungs := []int{4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768}
	out := make([]int, 0, len(rungs))
	for _, rung := range rungs {
		if rung < ctx {
			out = append(out, rung)
		}
	}
	return out
}

func placementCachePathForStrategy(s *Strategy, caps *detect.Capabilities, model *ModelProfile, opts Options) string {
	if s == nil || model == nil || caps == nil {
		return ""
	}
	cacheSplitKey := splitCompactKey(s.TensorSplit)
	path := PlacementCachePathFor(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel, cacheSplitKey, opts.SWAFull)
	return placementCachePathForSpec(path, opts.SpecMode, s.Draft)
}

func bytesToMiBCeil(n int64) int {
	if n <= 0 {
		return 0
	}
	return int((n + 1048576 - 1) / 1048576)
}

func ceilDivInt(n, d int) int {
	if n <= 0 || d <= 0 {
		return 0
	}
	return (n + d - 1) / d
}

// layerOwnership mirrors llama.cpp's tensor-split slot assignment exactly
// (llama-model.cpp): n_layer+1 slots — the repeating layers plus the output
// head — are distributed by upper_bound over the cumulative normalized split.
// Returns the owned repeating-layer count per device and the index of the
// device that owns the output slot (-1 if no device has a share). The input
// layer (token embeddings) always stays on the CPU and owns no slot.
func layerOwnership(split []float64, numLayers int) (owned []int, outputDev int) {
	owned = make([]int, len(split))
	layerDevices, outputDev := layerDeviceAssignments(split, numLayers)
	for _, dev := range layerDevices {
		if dev >= 0 && dev < len(owned) {
			owned[dev]++
		}
	}
	return owned, outputDev
}

// layerDeviceAssignments returns the exact device index for every repeating
// layer plus the output slot. Keeping the per-layer ownership is needed when a
// cost applies only to the MoE suffix (for example shared experts).
func layerDeviceAssignments(split []float64, numLayers int) (layerDevices []int, outputDev int) {
	layerDevices = make([]int, numLayers)
	for i := range layerDevices {
		layerDevices[i] = -1
	}
	outputDev = -1
	sum := 0.0
	for _, v := range split {
		if v > 0 {
			sum += v
		}
	}
	if sum <= 0 || numLayers <= 0 {
		return
	}
	cum := make([]float64, len(split))
	c := 0.0
	for i, v := range split {
		if v > 0 {
			c += v
		}
		cum[i] = c / sum
	}
	slots := numLayers + 1
	for slot := 0; slot < slots; slot++ {
		f := float64(slot) / float64(slots)
		dev := -1
		for i := range cum {
			if cum[i] > f {
				dev = i
				break
			}
		}
		if dev < 0 {
			dev = len(split) - 1
		}
		if slot == numLayers {
			outputDev = dev
		} else {
			layerDevices[slot] = dev
		}
	}
	return layerDevices, outputDev
}

// ownedShareMB charges a device its owned-layer fraction of a per-layer total
// (e.g. the KV cache, which llama.cpp allocates on each layer's device).
func ownedShareMB(totalMB int, owned []int, numLayers, idx int) int {
	if totalMB <= 0 || numLayers <= 0 || idx < 0 || idx >= len(owned) || owned[idx] <= 0 {
		return 0
	}
	return int(math.Ceil(float64(totalMB) * float64(owned[idx]) / float64(numLayers)))
}

// hasMeasuredCUDAOverheadForActiveGPUs gates optional sub-layer squeeze.
// VERIFICATION: cold-cache placement must not treat an unmeasured CUDA context
// as free remainder; no percentage/static reserve is hidden here.
func hasMeasuredCUDAOverheadForActiveGPUs(overheadByGPU map[int]int, gpus []detect.GPU, split []float64) bool {
	if len(gpus) == 0 || len(split) == 0 {
		return false
	}
	for i, g := range gpus {
		if i >= len(split) || split[i] <= 0 {
			continue
		}
		if overheadByGPU[g.Index] <= 0 {
			return false
		}
	}
	return true
}

func splitShareMB(totalMB int, split []float64, idx int) int {
	if totalMB <= 0 || idx < 0 || idx >= len(split) || split[idx] <= 0 {
		return 0
	}
	totalSplit := 0.0
	for _, v := range split {
		if v > 0 {
			totalSplit += v
		}
	}
	if totalSplit <= 0 {
		return 0
	}
	return int(math.Ceil(float64(totalMB) * split[idx] / totalSplit))
}

// ReplanAfterOOM recomputes the full placement after a cudaMalloc OOM, with the
// failed device(s) penalized by how much they overshot. Because it re-runs the
// real packer, the correction is fill-preserving: it refits the failed card
// tightly (partial gate+up chunks, not whole layers) AND reclaims stranded VRAM
// on the other cards via the sub-pin squeeze — so experts move off system RAM
// instead of a blind whole-layer drop that over-corrects and erases the squeeze.
// penaltyMB is keyed by GPU Index and accumulates across retries. Returns the new
// strategy, or an error if there's nothing to replan or it no longer fits.
func ReplanAfterOOM(caps *detect.Capabilities, model *ModelProfile, opts Options, penaltyMB map[int]int) (*Strategy, error) {
	if caps == nil || model == nil || len(penaltyMB) == 0 {
		return nil, fmt.Errorf("replan: nothing to do")
	}
	c2 := *caps
	c2.GPUs = append([]detect.GPU(nil), caps.GPUs...)
	any := false
	for i := range c2.GPUs {
		if p := penaltyMB[c2.GPUs[i].Index]; p > 0 {
			c2.GPUs[i].VRAMUsedMB += p // shrink usable VRAM by the overshoot
			any = true
		}
	}
	if !any {
		return nil, fmt.Errorf("replan: no matching device")
	}
	o := opts
	o.SkipPlacementCache = true // derive fresh, don't reload the placement that OOM'd
	o.CacheFile = ""
	return Compute(&c2, model, o)
}

// CurrentUBatch exposes the effective micro-batch size for launch recovery.
func CurrentUBatch(args []string) int {
	return currentUBatch(args)
}

// currentUBatch reads the launch args' current -ub/--ubatch-size value, or 0
// if unset/unparseable.
func currentUBatch(args []string) int {
	idx := argIndex(args, "-ub", "--ubatch-size")
	if idx < 0 || idx+1 >= len(args) {
		return 0
	}
	v, _ := strconv.Atoi(args[idx+1])
	return v
}

// nextUBatchDown returns the next smaller rung on the same fit ladder
// maximizeMoEGPUFitByUBatch uses at placement time, or ok=false if current is
// already at or below the ladder's floor.
func nextUBatchDown(current int) (int, bool) {
	for _, rung := range UBatchFitLadder {
		if rung < current {
			return rung, true
		}
	}
	return 0, false
}

// DerateCUDAOOMArgs recovers from a cudaMalloc load failure. isComputeBuffer
// distinguishes the two failure classes the caller can observe in the
// backend log: a graph_reserve/gallocr (compute-buffer) OOM scales with
// ubatch, not expert-layer placement, so shrinking ubatch one rung down the
// same ladder used at placement time is tried first; a model-weight
// allocation failure (isComputeBuffer=false) goes straight to moving expert
// layers from the failed device back to CPU, since ubatch has no bearing on
// weight tensor size.
func DerateCUDAOOMArgs(args []string, model *ModelProfile, caps *detect.Capabilities, device, allocMB int, isComputeBuffer bool) ([]string, *CacheEntry, bool) {
	if model == nil || model.NumLayers <= 0 || allocMB <= 0 {
		return nil, nil, false
	}
	if isComputeBuffer {
		if next, ok := nextUBatchDown(currentUBatch(args)); ok {
			newArgs := append([]string(nil), args...)
			setOrAppendArg(&newArgs, "-ub", strconv.Itoa(next))
			// Keep the in-memory Strategy's UBatchSize in sync with serverArgs —
			// applyDeratedPlacementEntry applies this to strategy, which is what
			// the success path persists to the .place cache. Without it, a cache
			// hit later would resurrect the OOM'd, too-large ubatch.
			return newArgs, &CacheEntry{UBatchSize: next}, true
		}
	}
	_, moeLayers := moeLayerRange(model)
	expertPerLayerMB := ceilDivInt(bytesToMiBCeil(model.ExpertBytes), moeLayers)
	if expertPerLayerMB <= 0 {
		return nil, nil, false
	}
	overshootMB := allocMB
	if caps != nil {
		for _, g := range caps.GPUs {
			if g.Index == device && allocMB > g.VRAMFreeMB() {
				overshootMB = allocMB - g.VRAMFreeMB()
				break
			}
		}
	}
	dropLayers := ceilDivInt(overshootMB, expertPerLayerMB)
	if dropLayers <= 0 {
		dropLayers = 1
	}

	otIdx := argIndex(args, "-ot", "--override-tensor")
	if otIdx < 0 || otIdx+1 >= len(args) {
		return nil, nil, false
	}
	assignments := parseOTAssignments(args[otIdx+1])
	if len(assignments) == 0 {
		return nil, nil, false
	}
	remainingDrop := dropLayers
	actualDrop := 0
	for i := range assignments {
		if assignments[i].CUDAIndex != device || assignments[i].Count <= 0 {
			continue
		}
		drop := remainingDrop
		if drop > assignments[i].Count {
			drop = assignments[i].Count
		}
		assignments[i].Count -= drop
		actualDrop += drop
		remainingDrop -= drop
		if remainingDrop == 0 {
			break
		}
	}
	if actualDrop == 0 {
		return nil, nil, false
	}

	newArgs := append([]string(nil), args...)
	newArgs[otIdx+1] = buildOTStringFromAssignments(assignments, nil, model.NumLayers, "")
	setOrAppendArg(&newArgs, "--n-cpu-moe", strconv.Itoa(currentNCPUMoE(args)+actualDrop))

	entry := cacheEntryFromArgs(newArgs, assignments)
	return newArgs, entry, true
}

func moeLayerRange(model *ModelProfile) (int, int) {
	if model == nil || model.NumLayers <= 0 {
		return 0, 0
	}
	start := model.LeadingDense
	if start < 0 || start >= model.NumLayers {
		start = 0
	}
	count := model.NumLayers - start
	if count <= 0 {
		return 0, model.NumLayers
	}
	return start, count
}

var otAssignmentPattern = regexp.MustCompile(`blk\\\.\(([^)]*)\).*=(?:CUDA|Vulkan)(\d+)`)

func parseOTAssignments(ot string) []GPUAssignment {
	var out []GPUAssignment
	for _, part := range strings.Split(ot, ",") {
		m := otAssignmentPattern.FindStringSubmatch(part)
		if m == nil {
			continue
		}
		device, err := strconv.Atoi(m[2])
		if err != nil {
			continue
		}
		layers := strings.Split(m[1], "|")
		if len(layers) == 0 {
			continue
		}
		start, err := strconv.Atoi(layers[0])
		if err != nil {
			continue
		}
		out = append(out, GPUAssignment{CUDAIndex: device, Start: start, Count: len(layers)})
	}
	return out
}

func argIndex(args []string, names ...string) int {
	for i, arg := range args {
		for _, name := range names {
			if arg == name {
				return i
			}
		}
	}
	return -1
}

func setOrAppendArg(args *[]string, name, value string) {
	if idx := argIndex(*args, name); idx >= 0 {
		if idx+1 < len(*args) {
			(*args)[idx+1] = value
			return
		}
	}
	*args = append(*args, name, value)
}

func currentNCPUMoE(args []string) int {
	idx := argIndex(args, "--n-cpu-moe")
	if idx < 0 || idx+1 >= len(args) {
		return 0
	}
	v, _ := strconv.Atoi(args[idx+1])
	return v
}

func cacheEntryFromArgs(args []string, assignments []GPUAssignment) *CacheEntry {
	entry := &CacheEntry{GPUAssignments: positiveAssignments(assignments)}
	// The exact override is authoritative. GPUAssignments can reproduce only
	// whole-layer runs; preserving argv also keeps backend-specific tensor regex
	// variants and sub-layer pins. Without this field, applyDeratedPlacementEntry
	// left Strategy.OTString at the pre-OOM layout and the success path cached the
	// failed GPU layer again even though the live retry used the corrected argv.
	if idx := argIndex(args, "-ot", "--override-tensor"); idx >= 0 && idx+1 < len(args) {
		entry.OTString = args[idx+1]
	}
	if idx := argIndex(args, "--tensor-split"); idx >= 0 && idx+1 < len(args) {
		entry.TensorSplit = parseTensorSplit(args[idx+1])
	}
	if idx := argIndex(args, "--split-mode"); idx >= 0 && idx+1 < len(args) {
		entry.SplitMode = args[idx+1]
	}
	if idx := argIndex(args, "--n-cpu-moe"); idx >= 0 && idx+1 < len(args) {
		entry.NCPUMoE, _ = strconv.Atoi(args[idx+1])
	}
	if idx := argIndex(args, "-b", "--batch-size"); idx >= 0 && idx+1 < len(args) {
		entry.BatchSize, _ = strconv.Atoi(args[idx+1])
	}
	if idx := argIndex(args, "-ub", "--ubatch-size"); idx >= 0 && idx+1 < len(args) {
		entry.UBatchSize, _ = strconv.Atoi(args[idx+1])
	}
	if idx := argIndex(args, "--parallel", "-np"); idx >= 0 && idx+1 < len(args) {
		entry.Parallel, _ = strconv.Atoi(args[idx+1])
	}
	entry.MMap = argIndex(args, "--no-mmap") < 0
	// Persist the resolved KV placement so a cache hit re-applies it:
	// without this, no .place cache carries CACHED_KVUNIFIED and the load-side
	// check at placement.go:397 never fires.
	// GPU KV is the server default. Some compatible backends do not expose the
	// positive --kv-offload flag, so absence means GPU rather than "unknown".
	// This keeps a derated placement cache correct for ik_llama as well as
	// mainline llama.cpp.
	entry.KVUnified = argIndex(args, "--no-kv-offload") < 0
	return entry
}

func positiveAssignments(assignments []GPUAssignment) []GPUAssignment {
	out := make([]GPUAssignment, 0, len(assignments))
	for _, a := range assignments {
		if a.Count > 0 {
			out = append(out, a)
		}
	}
	return out
}

// buildOTString builds the -ot override-tensor string for MoE.
// Builds the -ot override-tensor string: explicit layer list with escaped dots.
func buildOTString(layersPerGPU []int, gpus []detect.GPU, gpuOrder []int, backendTag string) string {
	return buildOTStringFromStart(layersPerGPU, gpus, gpuOrder, 0, backendTag)
}

const expertTensorPattern = `ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b)`

func buildOTStringFromStart(layersPerGPU []int, gpus []detect.GPU, gpuOrder []int, startLayer int, backendTag string) string {
	var parts []string

	nextLayer := startLayer
	for _, gi := range gpuOrder {
		count := layersPerGPU[gi]
		if count > 0 {
			start := nextLayer
			last := start + count - 1
			cudaIdx := gpus[gi].Index
			// Build explicit layer list, e.g. 0|1|2|...|31
			var layerParts []string
			for l := start; l <= last; l++ {
				layerParts = append(layerParts, fmt.Sprintf("%d", l))
			}
			layerRange := stringsJoin(layerParts, "|")
			parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, layerRange, expertTensorPattern, deviceName(backendTag, cudaIdx)))
			nextLayer += count
		}
	}
	parts = append(parts, "exps=CPU")

	return stringsJoin(parts, ",")
}

// subExpertPin pins one MoE layer's gate+up expert projections (2/3 of the
// layer's expert weight) to a specific GPU. The layer's down projection is left
// unpinned so the exps=CPU catch-all keeps it in system RAM.
type subExpertPin struct {
	Layer int // absolute layer index
	GI    int // position in caps.GPUs
}

// packGateUpChunks fills the VRAM that whole-layer packing floors away. Whole
// layers cost ~expertPerLayerMB each, so each GPU strands up to that much VRAM
// (its remainder). A layer's gate+up projections are 2/3 of its expert weight;
// pinning them onto a GPU with leftover room — down stays on CPU — turns stranded
// VRAM into expert residency: more experts on the GPU, that much less in system
// RAM. Greedy by remainder: each CPU-bound layer goes to the GPU (bandwidth
// order for ties) with the most room that can still hold a gate+up chunk.
// remainderMB is indexed by caps.GPUs position; gpuOrder gives the fill order.
// Returns the pins and the total expert MB moved off the CPU.
func packGateUpChunks(remainderMB []int, gpuOrder []int, gateUpChunkMB, cpuStartLayer, cpuLayerCount int) ([]subExpertPin, int) {
	if gateUpChunkMB <= 0 || cpuLayerCount <= 0 {
		return nil, 0
	}
	rem := make([]int, len(remainderMB))
	copy(rem, remainderMB)
	var pins []subExpertPin
	movedMB := 0
	for i := 0; i < cpuLayerCount; i++ {
		best := -1
		for _, gi := range gpuOrder {
			if gi < 0 || gi >= len(rem) {
				continue
			}
			if rem[gi] >= gateUpChunkMB && (best < 0 || rem[gi] > rem[best]) {
				best = gi
			}
		}
		if best < 0 {
			break // no GPU can hold another gate+up chunk
		}
		pins = append(pins, subExpertPin{Layer: cpuStartLayer + i, GI: best})
		rem[best] -= gateUpChunkMB
		movedMB += gateUpChunkMB
	}
	return pins, movedMB
}

// buildOTStringWithSubPins is buildOTStringFromStart plus optional sub-layer
// gate+up pins. Whole-layer pins come first, then the gate+up pins, then the
// exps=CPU catch-all — first-match-wins (see llama.cpp arg.cpp) keeps each
// partial layer's gate+up on its GPU while down falls through to CPU. With no
// sub-pins the output is identical to buildOTStringFromStart.
func buildOTStringWithSubPins(layersPerGPU []int, subPins []subExpertPin, gpus []detect.GPU, gpuOrder []int, startLayer int, backendTag string) string {
	var parts []string
	// Match expert weight tensors (routed *_exps, shared *_shexp) AND the
	// per-layer routing tensors (ffn_gate_inp for routed-gate layers,
	// ffn_gate_tid2eid + ffn_exp_probs_b for hash-routed early layers). The
	// routing tensors must ride with their expert weights on the same CUDA
	// device, otherwise llama.cpp's MoE dispatch cannot send the expert
	// compute to that GPU and the layer silently falls back to CPU/GPU0 —
	// leaving the expert GPU idle (e.g. GPU2 at 0% util with 9GB loaded).
	gateUpPattern := `ffn_(gate_up|up_gate|gate|up)_(ch|)exps`

	nextLayer := startLayer
	for _, gi := range gpuOrder {
		count := layersPerGPU[gi]
		if count > 0 {
			start := nextLayer
			last := start + count - 1
			var layerParts []string
			for l := start; l <= last; l++ {
				layerParts = append(layerParts, fmt.Sprintf("%d", l))
			}
			parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, stringsJoin(layerParts, "|"), expertTensorPattern, deviceName(backendTag, gpus[gi].Index)))
			nextLayer += count
		}
	}

	// Group sub-pins by GPU, preserving first-seen order, one compact rule per
	// GPU, emitted before the exps=CPU catch-all.
	if len(subPins) > 0 {
		byGPU := map[int][]int{}
		var order []int
		for _, p := range subPins {
			if _, ok := byGPU[p.GI]; !ok {
				order = append(order, p.GI)
			}
			byGPU[p.GI] = append(byGPU[p.GI], p.Layer)
		}
		for _, gi := range order {
			var layerParts []string
			for _, l := range byGPU[gi] {
				layerParts = append(layerParts, fmt.Sprintf("%d", l))
			}
			parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, stringsJoin(layerParts, "|"), gateUpPattern, deviceName(backendTag, gpus[gi].Index)))
		}
	}

	parts = append(parts, "exps=CPU")
	return stringsJoin(parts, ",")
}

func buildOTStringFromAssignments(assignments []GPUAssignment, gpus []detect.GPU, numLayers int, backendTag string) string {
	var parts []string
	// Match expert weight tensors (routed *_exps, shared *_shexp) AND the
	// per-layer routing tensors (ffn_gate_inp for routed-gate layers,
	// ffn_gate_tid2eid + ffn_exp_probs_b for hash-routed early layers). The
	// routing tensors must ride with their expert weights on the same CUDA
	// device, otherwise llama.cpp's MoE dispatch cannot send the expert
	// compute to that GPU and the layer silently falls back to CPU/GPU0 —
	// leaving the expert GPU idle (e.g. GPU2 at 0% util with 9GB loaded).

	nextLayer := 0
	for _, assign := range assignments {
		if assign.Count <= 0 {
			continue
		}
		start := assign.Start
		last := start + assign.Count - 1
		var layerParts []string
		for l := start; l <= last; l++ {
			layerParts = append(layerParts, fmt.Sprintf("%d", l))
		}
		layerRange := stringsJoin(layerParts, "|")
		parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, layerRange, expertTensorPattern, deviceName(backendTag, assign.CUDAIndex)))
		nextLayer += assign.Count
	}
	parts = append(parts, "exps=CPU")
	return stringsJoin(parts, ",")
}

func otStringUsesDevice(ot string, index int) bool {
	return strings.Contains(ot, fmt.Sprintf("=CUDA%d", index)) ||
		strings.Contains(ot, fmt.Sprintf("=Vulkan%d", index))
}

func deviceName(backendTag string, index int) string {
	if strings.EqualFold(backendTag, "vulkan") {
		return fmt.Sprintf("Vulkan%d", index)
	}
	return fmt.Sprintf("CUDA%d", index)
}

func stringsJoin(parts []string, sep string) string {
	if len(parts) == 0 {
		return ""
	}
	result := parts[0]
	for i := 1; i < len(parts); i++ {
		result += sep + parts[i]
	}
	return result
}

// computeKVTotalMB calculates exact KV cache size.
func computeKVTotalMB(model *ModelProfile, ctxSize int, kvType string, swaFull bool) int {
	// Prefer the KV size llama.cpp actually allocated on a previous launch (read
	// back from its log) — it is exact for every attention scheme, including the
	// compressed ones (MLA / CSA-HCA / sliding-window) the formula below can't
	// model. Falls through to the per-arch estimate when we have no measurement.
	// A measured geometry beats a measured rate. The rate is bytes per token of
	// context, which assumes KV grows linearly with the context -- true for a
	// uniform model, wrong for an interleaved sliding-window one whose windowed
	// layers are fixed-depth, and unable to express --swa-full at all. Laguna
	// measured 13864 MiB at 1M and 55296 with --swa-full; one rate cannot be
	// both.
	globalMeasurementSafe := !requiresScopedContextEvidence(model)
	if g, ok := model.MeasuredKVGeometry[strings.ToLower(kvType)]; globalMeasurementSafe && ok && g.Measured() {
		if mb := g.TotalMB(ctxSize, swaFull); mb > 0 {
			return mb
		}
	}
	// A geometry measured at one KV type still describes this model at every
	// other one: the layer split and the window depth are properties of the
	// architecture, and only the per-cell width follows the quantisation. So
	// rescale rather than fall through to the formula -- a measured layout at
	// the wrong type predicts --swa-full far better than an estimate at the
	// right one, and launches routinely pick a type the model was never
	// measured at (Laguna is measured at q4_0 and planned at q8_0).
	if g, from, ok := anyMeasuredKVGeometry(model); globalMeasurementSafe && ok {
		if scaled, ok := rescaleKVGeometry(g, from, kvType); ok {
			if mb := scaled.TotalMB(ctxSize, swaFull); mb > 0 {
				return mb
			}
		}
	}
	// The rate cannot express --swa-full at all (it is bytes per token of
	// context, and swa-full changes which layers scale with context), so it is
	// only safe when the windowed layers keep their fixed depth.
	if r, ok := model.MeasuredKVBytesPerTok[strings.ToLower(kvType)]; globalMeasurementSafe && ok && r > 0 && !swaFull {
		return int(r*float64(ctxSize)/1048576.0 + 0.5)
	}

	var kvElemsTotal int

	hasMLA := model.KVLoraRank > 0
	hasSSM := model.HasSSM == 1
	hasISWA := model.SlidingWindow > 0

	if hasMLA {
		// MLA: compressed c^{KV} + RoPE'd key once per layer
		kvElemsTotal = model.NumLayers * ctxSize * (model.KVLoraRank + model.RopeDim)
	} else if hasSSM {
		var attnLayers int
		if model.FullAttnInterval > 0 {
			attnLayers = model.NumLayers / model.FullAttnInterval
			if attnLayers < 1 {
				attnLayers = 1
			}
		} else if model.HeadCountKV == 0 {
			attnLayers = 0
		} else {
			attnLayers = (model.NumLayers + 1) / 2
		}
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = attnLayers * ctxSize * kvBytesPerLayerPerToken
	} else if hasISWA {
		swaPeriod := 6
		switch model.ModelArch {
		case "gemma2", "cohere2", "exaone4", "llama4":
			swaPeriod = 4
		case "gemma3":
			swaPeriod = 6
		case "plamo3":
			swaPeriod = 8
		}
		fullLayers := (model.NumLayers + swaPeriod - 1) / swaPeriod
		swaLayers := model.NumLayers - fullLayers
		// --swa-full sets size_swa = size_base, so the windowed layers stop being
		// fixed-depth and scale with the context like the full ones. Clamping to
		// the sliding window here under-predicts by the SWA layer count -- for
		// Laguna, 36 of 48 layers, a 4x miss that surfaces as an OOM at load
		// rather than as a rejected plan.
		swaCtx := ctxSize
		if !swaFull && swaCtx > model.SlidingWindow {
			swaCtx = model.SlidingWindow
		}
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = fullLayers*ctxSize*kvBytesPerLayerPerToken + swaLayers*swaCtx*kvBytesPerLayerPerToken
	} else {
		// Standard GQA/MQA
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = model.NumLayers * ctxSize * kvBytesPerLayerPerToken
	}

	bytesPerElem, ok := kvTypeBytesPerElement(kvType)
	if !ok {
		// Compute normally receives a validated type. Preserve the old
		// conservative q8_0 fallback for direct package callers.
		bytesPerElem = 1.0625
	}

	return int(float64(kvElemsTotal) * bytesPerElem / 1024 / 1024)
}

// requiresScopedContextEvidence identifies layouts whose cache is composed of
// independently-sized recurrent/attention regions. Their measured total is
// valuable for the exact launch but is not a model-wide rate or a uniformly
// quantized geometry that can be extrapolated safely.
func requiresScopedContextEvidence(model *ModelProfile) bool {
	if model == nil {
		return false
	}
	return model.HasSSM != 0 || strings.EqualFold(model.ModelArch, "deepseek4")
}

// EstimateKVCacheMB exposes the same KV arithmetic used by placement to
// read-only callers such as the TUI. It deliberately accepts a ModelProfile so
// measured geometry, when available, remains more authoritative than a generic
// architecture formula.
func EstimateKVCacheMB(model *ModelProfile, ctxSize int, kvType string, swaFull bool) int {
	if model == nil || ctxSize <= 0 {
		return 0
	}
	return computeKVTotalMB(model, ctxSize, kvType, swaFull)
}

// NormalizeKVType resolves ggrun's quality presets and the cache types accepted
// by llama.cpp's --cache-type-k/--cache-type-v flags. The returned spelling is
// safe to pass straight to llama-server and to use as the probe/cache key.
func NormalizeKVType(value string) (string, error) {
	typeName := strings.ToLower(strings.TrimSpace(value))
	typeName = strings.TrimPrefix(typeName, "ggml_")
	switch typeName {
	case "", "auto", "mid":
		return "q8_0", nil
	case "high":
		return "f16", nil
	case "low":
		return "q4_0", nil
	case "fp16":
		return "f16", nil
	case "fp32":
		return "f32", nil
	case "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1":
		return typeName, nil
	default:
		return "", fmt.Errorf("unsupported type %q (use auto, high, mid, low, f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, or q5_1)", value)
	}
}

func kvTypeFromQuality(quality string) string {
	typeName, err := NormalizeKVType(quality)
	if err != nil {
		return "q8_0"
	}
	return typeName
}

func exactKVTypeRequested(quality string) bool {
	switch strings.ToLower(strings.TrimSpace(quality)) {
	case "", "high", "mid", "low":
		return false
	default:
		return true
	}
}

func kvTypesForAutoContext(preferred, quality string) []string {
	values := []string{preferred}
	if !exactKVTypeRequested(quality) {
		values = append(values, "q8_0", "q4_0")
	}
	seen := make(map[string]bool, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value != "" && !seen[value] {
			seen[value] = true
			out = append(out, value)
		}
	}
	return out
}

func fallbackKVType(preferred, quality string) string {
	if exactKVTypeRequested(quality) {
		return preferred
	}
	return "q4_0"
}

func kvTypeBytesPerElement(kvType string) (float64, bool) {
	typeName, err := NormalizeKVType(kvType)
	if err != nil {
		return 0, false
	}
	switch typeName {
	case "f32":
		return 4, true
	case "f16", "bf16":
		return 2, true
	case "q8_0":
		return 1.0625, true // 34-byte block / 32 values
	case "q5_1":
		return 0.75, true // 24-byte block / 32 values
	case "q5_0":
		return 0.6875, true // 22-byte block / 32 values
	case "q4_1":
		return 0.625, true // 20-byte block / 32 values
	case "q4_0", "iq4_nl":
		return 0.5625, true // 18-byte block / 32 values
	default:
		return 0, false
	}
}

func orderGPUsByBandwidth(gpus []detect.GPU) []int {
	indices := make([]int, len(gpus))
	for i := range gpus {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		gi := gpus[indices[i]]
		gj := gpus[indices[j]]
		// Primary: bandwidth DESC
		if gi.BandwidthMBps != gj.BandwidthMBps {
			return gi.BandwidthMBps > gj.BandwidthMBps
		}
		// Secondary: VRAM total DESC
		if gi.VRAMTotalMB != gj.VRAMTotalMB {
			return gi.VRAMTotalMB > gj.VRAMTotalMB
		}
		// Tertiary: PCI index ASC (lower index = closer to CPU)
		return gi.Index < gj.Index
	})
	return indices
}

// kvReserveByBandwidth distributes KV cache proportionally to free VRAM.
// KV reads are VRAM-local — PCIe bandwidth does not affect KV access speed.
func kvReserveByBandwidth(kvTotalMB int, gpus []detect.GPU, order []int, kvPerLayerMB int) []int {
	reserve := make([]int, len(gpus))
	totalFree := 0
	for _, g := range gpus {
		totalFree += g.VRAMFreeMB()
	}
	if kvTotalMB <= 0 || totalFree <= 0 {
		return reserve
	}
	useOrder := order
	if len(useOrder) == 0 {
		useOrder = seqRange(len(gpus))
	}
	for _, gi := range useOrder {
		share := (kvTotalMB*gpus[gi].VRAMFreeMB() + totalFree - 1) / totalFree
		if kvPerLayerMB > 0 {
			share = ((share + kvPerLayerMB - 1) / kvPerLayerMB) * kvPerLayerMB
		}
		reserve[gi] = share
	}
	return reserve
}

func seqRange(n int) []int {
	r := make([]int, n)
	for i := range r {
		r[i] = i
	}
	return r
}

func orderGPUsByFreeVRAM(gpus []detect.GPU) []int {
	indices := make([]int, len(gpus))
	for i := range gpus {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		vi := gpus[indices[i]].VRAMFreeMB()
		vj := gpus[indices[j]].VRAMFreeMB()
		if vi == vj {
			return gpus[indices[i]].Index < gpus[indices[j]].Index
		}
		return vi > vj
	})
	return indices
}

func normalizeSplit(split []float64) []float64 {
	var total float64
	for _, v := range split {
		total += v
	}
	if total == 0 {
		return split
	}
	for i := range split {
		split[i] = math.Round(split[i]/total*100) / 100
	}
	return split
}

// gpuSplitWeight returns the weight applied to a GPU's free VRAM when
// computing the tensor-split for MoE offload. Under --split-mode layer the
// GPU that owns a layer's non-expert weights computes that layer, so
// CPU-resident experts stream over that GPU's PCIe link. Weighting by PCIe
// bandwidth concentrates ownership on fast-link GPUs. Returns 1.0 when
// bandwidth is unknown so the split degenerates to free-VRAM-proportional.
func gpuSplitWeight(g detect.GPU) float64 {
	if g.BandwidthMBps <= 0 {
		return 1.0
	}
	return float64(g.BandwidthMBps)
}

// expertOnlySlowGPUs classifies GPUs as expert-only devices: they must not own
// regular layer slots, but their VRAM can still hold whole expert layers. A GPU
// is an expert-only candidate when EITHER (a) its PCIe link is very slow
// relative to the fastest GPU (bandwidth ratio <= expertOnlyMaxBandwidthRatio),
// so owning dense layers would bottleneck the layer pipeline, OR (b) it cannot
// fit the split-owner compute reserve plus one dense layer's non-expert weight
// (capacity trigger), so a dense split slot would be wasted on it. The
// classification uses the normal split-owner reserve to make sure at least one
// true layer owner remains, and the expert-only reserve to decide whether a
// candidate card can carry at least one complete expert layer.
func expertOnlySlowGPUs(gpus []detect.GPU, splitFixedPerGPU, expertOnlyFixedPerGPU []int, expertPerLayerMB, nonExpertPerLayerMB int) []bool {
	expertOnly := make([]bool, len(gpus))
	if len(gpus) <= 1 || expertPerLayerMB <= 0 {
		return expertOnly
	}
	maxBandwidth := 0
	for _, g := range gpus {
		if g.BandwidthMBps > maxBandwidth {
			maxBandwidth = g.BandwidthMBps
		}
	}
	if maxBandwidth <= 0 {
		return expertOnly
	}

	splitCandidates := 0
	for i, g := range gpus {
		if i < len(splitFixedPerGPU) && g.VRAMFreeMB() > splitFixedPerGPU[i] {
			splitCandidates++
		}
	}
	candidates := make([]int, 0, len(gpus))
	for i, g := range gpus {
		if i >= len(splitFixedPerGPU) || i >= len(expertOnlyFixedPerGPU) || g.BandwidthMBps <= 0 {
			continue
		}
		slowLink := float64(g.BandwidthMBps)/float64(maxBandwidth) <= expertOnlyMaxBandwidthRatio
		cantFitDense := nonExpertPerLayerMB > 0 && g.VRAMFreeMB()-splitFixedPerGPU[i] < nonExpertPerLayerMB
		if !slowLink && !cantFitDense {
			continue
		}
		if g.VRAMFreeMB()-expertOnlyFixedPerGPU[i] < expertPerLayerMB {
			continue
		}
		candidates = append(candidates, i)
	}
	sort.Slice(candidates, func(i, j int) bool {
		gi := gpus[candidates[i]]
		gj := gpus[candidates[j]]
		if gi.BandwidthMBps != gj.BandwidthMBps {
			return gi.BandwidthMBps < gj.BandwidthMBps
		}
		if gi.VRAMFreeMB() != gj.VRAMFreeMB() {
			return gi.VRAMFreeMB() > gj.VRAMFreeMB()
		}
		return gi.Index < gj.Index
	})
	for _, i := range candidates {
		if splitCandidates <= 1 {
			break
		}
		expertOnly[i] = true
		splitCandidates--
	}
	return expertOnly
}

// expertOnlyComputeReserveMB is the compute margin for a GPU that will only run
// explicitly pinned experts. It keeps measured/recorded CUDA and runtime growth
// outside this helper and caps the prompt-graph reserve at llama.cpp's compute
// floor, because an expert-only GPU does not own regular prompt-processing
// layer slots or KV. The probe cache's per-GPU compute buffer is measured for
// the placement that actually ran (split-owner with dense layers), which is
// dramatically larger than what an expert-only GPU needs (e.g. DeepSeek-V4:
// ~9.8 GB split-owner vs ~299 MB expert-only on the same card). Using the
// split-owner value for an expert-only GPU blocks it from receiving any expert
// layers at all. Cap at the compute floor: it is a conservative reserve for
// the small expert-projection graph, and the preflight gate catches any real
// overflow before the load.
// expertOnlyComputeReserveMB sizes the compute buffer for a GPU that carries
// pinned expert tensors but no attention, norms or KV.
//
// It used to return a flat computeFloorMB, which is a fixed margin in a planner
// whose whole premise is that reserved VRAM must be measured. On this project
// the measurement was 99 MiB against a 1024 MiB floor -- 925 MiB withheld, 67%
// of an expert layer, and always on the smallest card, because that is where
// the reviewer is seated and where a withheld layer costs the most.
//
// The reason the per-GPU measurement was distrusted is real and documented: a
// GPU that was expert-only when measured may be a split owner on the next plan,
// and 99 MiB would then be catastrophically short. But the magnitude settles
// that question by itself. A split owner's buffer is orders of magnitude larger
// -- measured 4267 MiB against 99 on the same launch -- so a per-GPU value that
// small could only have come from a run where that GPU was already expert-only.
//
// Accept it under that test, with headroom, and fall back to the floor whenever
// the measurement is absent or large enough to be ambiguous.
func expertOnlyComputeReserveMB(splitOwnerComputeMB, measuredThisGPUMB int) int {
	if measuredThisGPUMB <= 0 || splitOwnerComputeMB <= 0 {
		return computeFloorMB
	}
	// Ambiguous: not obviously smaller than a split owner's buffer, so it may
	// have been measured while this GPU owned a split.
	if measuredThisGPUMB*expertOnlyComputeRoleRatio > splitOwnerComputeMB {
		return computeFloorMB
	}
	reserve := measuredThisGPUMB * expertOnlyComputeHeadroomNum / expertOnlyComputeHeadroomDen
	if reserve > computeFloorMB {
		return computeFloorMB
	}
	return reserve
}

const (
	// expertOnlyComputeRoleRatio is how much smaller than the split owner's
	// buffer a per-GPU measurement must be before it is accepted as proof that
	// the GPU was expert-only when measured. 4267 against 99 is a factor of 43;
	// requiring 8 leaves a wide margin for a smaller model.
	expertOnlyComputeRoleRatio = 8
	// Headroom on an accepted measurement, since a later plan can pin a few more
	// expert layers to the same card than the measured one did.
	expertOnlyComputeHeadroomNum = 3
	expertOnlyComputeHeadroomDen = 2
)

// modelAwareHeadroom estimates the non-weight VRAM/RAM the runtime needs beyond
// the model weights (prompt-graph compute buffer + a small runtime-growth
// margin). Replaces the flat 8 GiB guess previously hard-coded in the auto
// context-size paths so large models reserve enough and small models don't
// waste context capacity.
func modelAwareHeadroom(model *ModelProfile) int {
	return firstLaunchComputeBufMB(model, 512) + 1024
}

func plannedModelAwareHeadroom(model *ModelProfile, opts Options) int {
	if opts.RequireMeasuredBuffers {
		return 0
	}
	return modelAwareHeadroom(model)
}

// firstLaunchComputeBufMB is a conservative compute-buffer reservation used until
// the post-launch probe measures the real value for this model + settings. The
// prompt-processing graph scales with ubatch AND model shape: activation working
// set per token is roughly proportional to hidden_size * num_layers. The old flat
// ~4 MiB/ubatch estimate (ub 512 -> 2048 MiB) under-reserved large MoE graphs by
// ~4.4x — once expert-packing filled the GPU, llama.cpp's compute-buffer alloc
// OOM'd ("failed to create context", V4 needs ~9020 MiB at ub 512). We now size
// from the model: bytes per (token·hidden·layer) ≈ 42, calibrated so V4 reserves
// ~9412 MiB (covers the measured ~9020). Over-estimating is safe — the probe cache
// overrides this with the measured value after the first launch; under-estimating
// is fatal (OOM crash). A nil model falls back to the old per-ubatch heuristic.
func firstLaunchComputeBufMB(model *ModelProfile, uBatch int) int {
	return firstLaunchComputeBufMBParallel(model, uBatch, 1)
}

// firstLaunchComputeBufMBParallel estimates the graph reserve for a cold-cache
// launch. MoE routed-expert activation grows with experts selected per token;
// parallel slots divide the physical ubatch graph approximately evenly. The
// 128-byte fan-out coefficient is calibrated against llama-fit-params for
// DeepSeek-V4 (6 active experts: 33.9 GiB at ub256/parallel1 and 8.9 GiB at
// ub256/parallel4). Dense/unknown models keep the prior 42-byte coefficient.
func firstLaunchComputeBufMBParallel(model *ModelProfile, uBatch, parallel int) int {
	est := uBatch * 4
	if model != nil && model.HiddenSize > 0 && model.NumLayers > 0 {
		coefficient := 42.0
		// DeepSeek4's routed MLA graph is substantially larger than a generic
		// MoE graph. Do not project that architecture-specific coefficient onto
		// Kimi/Mixtral/etc.; their exact backend preflight will calibrate them.
		if strings.EqualFold(model.ModelArch, "deepseek4") && model.ExpertUsedCount > 0 {
			moeCoefficient := float64(model.ExpertUsedCount) * 128.0
			if moeCoefficient > coefficient {
				coefficient = moeCoefficient
			}
		}
		per := float64(model.HiddenSize) * float64(model.NumLayers) * coefficient / 1e6
		est = int(float64(uBatch) * per)
		if parallel > 1 {
			est = ceilDivInt(est, parallel)
		}
	}
	if est < computeFloorMB {
		est = computeFloorMB
	}
	return est
}

func firstLaunchComputeBufMBForGPU(model *ModelProfile, uBatch, gpuPos int, order []int) int {
	return firstLaunchComputeBufMBForGPUParallel(model, uBatch, 1, gpuPos, order)
}

func firstLaunchComputeBufMBForGPUParallel(model *ModelProfile, uBatch, parallel, gpuPos int, order []int) int {
	// Every device that owns regular split layers may need the full graph.
	// llama-fit-params measured 8.9 GiB on CUDA0 and 8.7 GiB on CUDA2 for the
	// same DeepSeek-V4 ub256/parallel4 plan. Discounting secondary owners by 50%
	// overcommitted 12 GiB cards. Expert-only devices are capped separately by
	// expertOnlyComputeReserveMB after their classification is known.
	_ = gpuPos
	_ = order
	return firstLaunchComputeBufMBParallel(model, uBatch, parallel)
}

// firstLaunchComputeBufMBForGPUParallelAtContext applies DeepSeek4's measured
// context scaling to the conservative 1M-context graph estimate. The original
// fan-out calibration came from ctx=1,048,576 and was previously charged
// unchanged at ctx=65,536. That predicted 33.9 GiB for ubatch 256 even though
// llama-fit-params measures 2,429 MiB for the exact p1/65k placement. Keep a
// 1 GiB graph floor and scale only the remainder; this predicts about 3 GiB at
// 65k, deliberately above the backend measurement while no longer rejecting a
// configuration already proven through a 60k request.
func firstLaunchComputeBufMBForGPUParallelAtContext(model *ModelProfile, uBatch, parallel, contextSize, gpuPos int, order []int) int {
	est := firstLaunchComputeBufMBForGPUParallel(model, uBatch, parallel, gpuPos, order)
	if model == nil || !strings.EqualFold(model.ModelArch, "deepseek4") || contextSize <= 0 {
		return est
	}
	const referenceContext = 1048576
	if est <= computeFloorMB {
		return est
	}
	scaledRemainder := int64(est-computeFloorMB) * int64(contextSize) / referenceContext
	scaled := int64(computeFloorMB) + scaledRemainder
	if scaled < computeFloorMB {
		return computeFloorMB
	}
	maxInt := int64(^uint(0) >> 1)
	if scaled > maxInt {
		return int(maxInt)
	}
	return int(scaled)
}

// perGPUVRAMOverheadMB is the non-weight VRAM a single GPU needs at runtime:
// measured CUDA context/allocator overhead plus the compute buffer. Missing
// CUDA probe data is unknown and contributes 0; no static fallback margin is
// hidden here.
func perGPUVRAMOverheadMB(sysProbe *systemProbe, uBatch int) int {
	return measuredCUDAOverheadMB(sysProbe) + firstLaunchComputeBufMB(nil, uBatch)
}

func plannedPerGPUVRAMOverheadMB(sysProbe *systemProbe, uBatch int, opts Options) int {
	if opts.RequireMeasuredBuffers {
		return measuredCUDAOverheadMB(sysProbe)
	}
	return perGPUVRAMOverheadMB(sysProbe, uBatch)
}

// measuredCUDAOverheadMB is the measured CUDA context/allocator overhead per GPU.
// Missing probe data contributes 0 so callers cannot accidentally depend on a
// fabricated reserve.
func measuredCUDAOverheadMB(sysProbe *systemProbe) int {
	if sysProbe != nil && sysProbe.CUDAOverheadMB > 0 {
		return sysProbe.CUDAOverheadMB
	}
	return 0
}

// ramRuntimeOverheadMB is the non-weight system RAM the runtime needs: CUDA host
// pinned staging, compute-graph scratch, the mmap page table, and CPU activation
// buffers. Each term is derived from model dims / file size rather than a flat
// guessed reserve. (mmapPT is exact: a 4KB-page table is fileSize/4096*8B ≈
// fileSize/512. The host/graph terms are the last constants still pending a
// measurement probe — kept here as the single place to swap in measured values.)
func ramRuntimeOverheadMB(model *ModelProfile, uBatch, totalSizeMB int) int {
	const cudaHostMB = 1024
	const graphScratchMB = 2048
	mmapPTMB := totalSizeMB / 500

	actFFN := model.FeedForwardLength
	if model.NumExperts > 0 && model.ExpertUsedCount > 0 && model.ExpertFF > 0 {
		actFFN = model.ExpertUsedCount * model.ExpertFF
		if model.ExpertSharedFF > 0 {
			actFFN += model.ExpertSharedFF
		}
	}
	if model.KVLoraRank > 0 {
		actFFN += model.KVLoraRank + model.QLoraRank
	}
	cpuActMB := uBatch * (model.EmbeddingLength + actFFN) * 4 * 2 / 1048576
	if cpuActMB < 64 {
		cpuActMB = 64
	}
	return cudaHostMB + graphScratchMB + mmapPTMB + cpuActMB
}

func plannedRAMRuntimeOverheadMB(caps *detect.Capabilities, model *ModelProfile, uBatch, totalSizeMB int, opts Options) int {
	if opts.RequireMeasuredBuffers {
		// A measurement satisfies the measured-buffers contract; an estimate does
		// not. Returning 0 unconditionally is what left ~1.9 GiB of real host
		// overhead outside every plan on this rig -- correct in refusing to guess,
		// but wrong once the number is actually known. See hostOverheadMB.
		if caps != nil {
			if sp := loadSystemProbe(opts.CacheDir, caps.GPUs); sp != nil && sp.HostOverheadMB > 0 {
				return sp.HostOverheadMB
			}
		}
		return 0
	}
	return ramRuntimeOverheadMB(model, uBatch, totalSizeMB)
}

// checkMemoryOrDie refuses to launch when model + KV + compute buffers exceed the pool.
// OOM guard: refuse to launch if model+KV+compute don't fit.
func checkMemoryOrDie(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int, opts Options) error {
	numGPUs := len(caps.GPUs)

	// Load measured CUDA overhead. Missing probe data is unknown and contributes 0;
	// preflight/startup OOM recording supplies measured data for later launches.
	cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

	// Load model probe for compute buffer
	computeBufMB := computeFloorMB // 1024 default
	if opts.RequireMeasuredBuffers {
		computeBufMB = 0
	}
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel)
	if pc != nil && pc.ComputeBufMB > 0 {
		computeBufMB = pc.ComputeBufMB
	}

	// Model weights + per-GPU overhead (CUDA context + compute buffer)
	// For single GPU, only count 1 GPU's overhead
	overheadGPUs := numGPUs
	if s.Type == CPUOnly {
		overheadGPUs = 0
	} else if s.Type == SingleGPU {
		overheadGPUs = 1
	}
	modelOverheadMB := totalSizeMB + (cudaOverheadMB+computeBufMB)*overheadGPUs
	neededMB := modelOverheadMB + kvTotalMB
	ramOverheadMB := 0
	if s.Type == CPUOnly || s.Type == DenseCPUOffload {
		ramOverheadMB = plannedRAMRuntimeOverheadMB(caps, model, s.UBatchSize, totalSizeMB, opts)
		neededMB += ramOverheadMB
	}

	var poolMB int
	var poolLabel string

	switch s.Type {
	case SingleGPU:
		// Best GPU's free VRAM
		bestFree := 0
		for _, g := range caps.GPUs {
			if g.VRAMFreeMB() > bestFree {
				bestFree = g.VRAMFreeMB()
			}
		}
		poolMB = bestFree
		poolLabel = "best GPU"
	case MultiGPUDense:
		// Total free VRAM across all GPUs
		for _, g := range caps.GPUs {
			poolMB += g.VRAMFreeMB()
		}
		poolLabel = "all GPUs"
	case CPUOnly:
		poolMB = caps.RAM.FreeMB
		poolLabel = "RAM"
	case DenseCPUOffload:
		// Total system memory (GPU + RAM) since model is split
		for _, g := range caps.GPUs {
			poolMB += g.VRAMFreeMB()
		}
		poolMB += caps.RAM.FreeMB
		poolLabel = "system memory"
	}

	if neededMB > poolMB {
		// Back-solve max safe context
		maxKVMB := poolMB - modelOverheadMB - ramOverheadMB
		if maxKVMB < 0 {
			maxKVMB = 0
		}
		maxCtx := 0
		if kvTotalMB > 0 {
			maxCtx = maxKVMB * s.ContextSize / kvTotalMB
		}

		msg := fmt.Sprintf(
			"ERROR: Model does not fit in %s.\n"+
				"  Model weights:          %dMB\n"+
				"  CUDA overhead (%d GPU): %dMB\n"+
				"  Compute buffers (%d):   %dMB\n"+
				"  KV cache (ctx=%d):      %dMB\n"+
				"  Host runtime buffers:   %dMB\n"+
				"  -----------------------------\n"+
				"  Total needed:          %dMB\n"+
				"  Available (%s):        %dMB\n"+
				"  Shortfall:             %dMB\n",
			poolLabel, totalSizeMB, overheadGPUs, cudaOverheadMB*overheadGPUs,
			overheadGPUs, computeBufMB*overheadGPUs, s.ContextSize, kvTotalMB,
			ramOverheadMB, neededMB, poolLabel, poolMB, neededMB-poolMB)

		if maxCtx > 0 {
			msg += fmt.Sprintf("\n  Max safe context at this memory: --ctx-size %d", maxCtx)
		} else if totalSizeMB > poolMB {
			msg += "\n  Model weights alone exceed available memory."
		} else {
			msg += "\n  Fixed runtime buffers leave no safe space for the requested KV cache."
		}
		msg += "\n  Or use a smaller quantization / model."
		return fmt.Errorf("%s", msg)
	}
	return nil
}

// computeCRAM calculates prompt cache size from remaining memory after load.
func computeCRAM(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int) (int, int) {
	numGPUs := len(caps.GPUs)

	// Fits on GPU? (model fits entirely in VRAM)
	fitsOnGPU := false
	switch s.Type {
	case SingleGPU, MultiGPUDense:
		fitsOnGPU = true
	}

	// RAM remaining after weights load
	var ramAfterLoad int
	switch {
	case fitsOnGPU:
		ramAfterLoad = caps.RAM.FreeMB
	case s.PlannedHostFootprintMB > 0:
		// What this plan actually puts in host RAM, taken from the plan rather
		// than re-derived here. See Strategy.PlannedHostFootprintMB.
		ramAfterLoad = caps.RAM.FreeMB - s.PlannedHostFootprintMB
	default:
		// No plan-derived figure (CPU-only, dense CPU offload, or a strategy
		// built by a path that does not compute one). Charging the model for
		// the whole VRAM install is wrong whenever VRAM holds anything besides
		// weights, but it is the only estimate available here and it errs
		// toward a larger cache, so keep it bounded by the two-thirds rule
		// below rather than trusted on its own.
		ramOnCPU := totalSizeMB - caps.TotalVRAM()
		if ramOnCPU < 0 {
			ramOnCPU = 0
		}
		ramAfterLoad = caps.RAM.FreeMB - ramOnCPU
	}
	if ramAfterLoad < 0 {
		ramAfterLoad = 0
	}

	// Size the cache to what an entry actually costs, not to an arbitrary
	// fraction of RAM. An entry is one slot's saved KV, so a cache that cannot
	// hold one per slot evicts a conversation's prefix to admit the next and
	// every turn re-prefills from zero. Measured on this project: entries of
	// ~6.5 GiB against a 9761 MiB budget gave 17.5% cache-read on the 4-slot
	// server while a 1-slot server on the same box, same client, reached 88.3%.
	//
	// The old tenth-of-free-RAM rule with a 16 GiB ceiling could not express
	// that: at four slots it was short by a factor of two before the ceiling
	// even bound. Keeping it as a floor means hosts whose KV estimate is small
	// or missing are no worse off than before.
	slots := s.Parallel
	if slots < 1 {
		slots = 1
	}
	cram := ramAfterLoad / 10
	if want := kvTotalMB; want > cram {
		cram = want
	}
	// A measurement of what an entry really costs beats both: the estimate above
	// is KV, and an entry is target state + draft state + checkpoints. Sizing is
	// per slot, against the context a turn actually carries rather than the
	// configured maximum, because a 1M-token allowance is not what gets cached.
	//
	// Until one of these is measured the budget does not vary with the slot
	// count at all, which is why it could not be compared across slot settings:
	// the same 9728 MiB came out at one slot and at four.
	if measured := measuredPromptCacheBudgetMB(s, slots); measured > cram {
		cram = measured
	}
	// Never take RAM the weights and their working set need: the backend runs
	// inside a memory scope, and a cache that pushes it over the limit trades a
	// slow launch for a failed one.
	if budget := ramAfterLoad * 2 / 3; cram > budget {
		cram = budget
	}
	// Round down, never up: the budget above is a ceiling and quantizing must
	// not push the cache past it.
	cram -= cram % cramQuantumMB
	if cram < minCramMB {
		cram = 0
	}

	// -1 means "not computed for this strategy type" (single-GPU/CPU-only):
	// leave the backend's own default alone rather than silently emitting a
	// disable decision nothing actually derived. The multi-GPU branch below
	// always overwrites this with a real, headroom-based value (0 included).
	maxCheckpoints := -1

	// The prompt cache lives in host RAM, not VRAM: server_prompt_cache copies
	// slot state out through llama_state_seq_get_data into host buffers and
	// caps itself in MiB. Sizing it from VRAM headroom therefore disabled it
	// exactly when it is most valuable -- a model large enough to fill VRAM is
	// also the one whose re-prefill costs the most. On a 3-GPU host serving a
	// 68 GiB MoE this produced `-cram 0`, which in turn disabled
	// --cache-idle-slots, so an agent evicted from a slot lost its whole
	// prefix instead of parking it in the ~99 GiB of free RAM.
	//
	// The host-RAM budget computed above is correct for every strategy, so the
	// multi-GPU case only adds a checkpoint policy on top of it.
	if numGPUs > 1 && s.Type != CPUOnly {
		if cram < minCramMB {
			maxCheckpoints = 0
		} else {
			maxCheckpoints = cram / 200
			if maxCheckpoints < 2 {
				maxCheckpoints = 2
			}
			if maxCheckpoints > 16 {
				maxCheckpoints = 16
			}
		}
	}

	// A zero checkpoint policy makes every append/branch agent turn re-evaluate
	// the complete prompt on a hybrid or recurrent model. One rolling checkpoint
	// is also insufficient: once a newer boundary is saved, a branch before it
	// has no restorable recurrent state. Keep as many as measured host headroom
	// can safely carry, with four the minimum useful branch window and sixteen a
	// bound against unaccounted allocator growth.
	if s.HasSSM {
		slots := s.Parallel
		if slots < 1 {
			slots = 1
		}
		checkpointHeadroom := ramAfterLoad - cram
		if checkpointHeadroom < 0 {
			checkpointHeadroom = 0
		}
		capacity := checkpointHeadroom / (slots * hybridCheckpointReservePerSlotMB)
		if capacity >= hybridCheckpointMinimum {
			if capacity > hybridCheckpointMaximum {
				capacity = hybridCheckpointMaximum
			}
			maxCheckpoints = capacity
		} else {
			maxCheckpoints = 0
		}
	}

	return cram, maxCheckpoints
}

// defaultFlashAttention decides whether ggrun forces `--flash-attn on`. The
// decision must follow the resolved KV placement because the CUDA FA kernel
// requires GPU-resident KV.
func defaultFlashAttention(model *ModelProfile, opts Options, kvPlacement string) bool {
	// llama.cpp auto-disables flash attention whenever the KV cache isn't
	// GPU-resident — the FA CUDA kernel
	// needs its KV tensor on the same device doing the attention compute.
	// Claiming FlashAttention here when kvPlacement=="cpu" would emit a
	// self-contradicting `--flash-attn on --no-kv-offload` command; for
	// deepseek4 specifically that also silently re-opens the unbounded
	// compute-buffer growth this flag exists to prevent (see [[Task #10]]).
	return kvPlacement != "cpu"
}

func defaultContextSize(model *ModelProfile, caps *detect.Capabilities) int {
	if model.ContextSize > 0 && model.ContextSize < 32768 {
		return model.ContextSize
	}
	return 32768
}

// computeAutoContextSizeSingleGPU computes the largest context that fits on
// a SINGLE GPU (the best one). Used to prefer single-GPU mode (faster).
func computeAutoContextSizeSingleGPU(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType string, opts Options) (int, string) {
	// Find best single GPU by total VRAM
	bestVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMTotalMB > bestVRAM {
			bestVRAM = g.VRAMTotalMB
		}
	}

	// Total hardware for single GPU: best GPU VRAM + up to 4GB RAM (not entire system)
	// Single GPU context shouldn't use entire system RAM — the model must fit on ONE GPU.
	totalHWMB := bestVRAM + 4096

	// Fixed overhead: model weights + model-aware headroom.
	fixedOverheadMB := totalSizeMB + plannedModelAwareHeadroom(model, opts)

	// If model doesn't fit at all, return minimum
	if totalHWMB <= fixedOverheadMB {
		return 32768, preferredKVType
	}

	// KV budget = total hardware - model - headroom
	kvBudgetMB := totalHWMB - fixedOverheadMB
	if kvBudgetMB <= 0 {
		return 32768, preferredKVType
	}

	orderedTypes := kvTypesForAutoContext(preferredKVType, opts.KVQuality)

	for _, kvType := range orderedTypes {
		refCtx := 32768
		refKVTotalMB := computeKVTotalMB(model, refCtx, kvType, opts.SWAFull)
		if refKVTotalMB <= 0 {
			continue
		}
		kvBytesPerToken := float64(refKVTotalMB) * 1048576.0 / float64(refCtx)
		maxCtxRaw := int(float64(kvBudgetMB) * 1048576.0 / kvBytesPerToken)

		hwCapCtx := maxCtxRaw
		if model.CTXTrain > 0 && model.CTXTrain < hwCapCtx {
			hwCapCtx = model.CTXTrain
		}

		powerOfTwoValues := []int{32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304}
		suggestedCtx := 32768
		for _, c := range powerOfTwoValues {
			if c <= hwCapCtx {
				suggestedCtx = c
			}
		}

		if suggestedCtx >= 32768 {
			return suggestedCtx, kvType
		}
	}

	return 32768, fallbackKVType(preferredKVType, opts.KVQuality)
}

// resolveAutoKVPlacement decides gpu vs cpu for the KV cache when --kv-placement
// is "auto". A model that fits in VRAM keeps its KV on GPU (fast, VRAM to spare).
// A big MoE whose experts must offload to CPU puts KV on CPU instead: that frees
// VRAM for more expert layers (the decode-bandwidth bottleneck) and unlocks a much
// larger context. A dense model bigger than VRAM keeps KV on GPU (its only spot).
func resolveAutoKVPlacement(caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB, vramOverheadMB int) string {
	freeVRAM := 0
	for _, g := range caps.GPUs {
		freeVRAM += g.VRAMFreeMB()
	}
	if totalSizeMB+kvTotalMB+vramOverheadMB <= freeVRAM {
		return "gpu"
	}
	if model.IsMoE {
		if strings.EqualFold(model.ModelArch, "deepseek4") {
			// KV on CPU makes llama.cpp auto-disable flash attention, and
			// deepseek4's non-FA graph materializes score tensors that grow
			// with real token position (~98 KiB/token measured 2026-07-09) —
			// no load-time reserve can cover that. Mainline therefore keeps
			// DeepSeek4 KV on GPU and trades expert VRAM for a bounded FA graph.
			return "gpu"
		}
		return "cpu"
	}
	return "gpu"
}

// computeAutoContextSizeKVPlacement computes the largest context whose KV cache
// fits in the memory implied by placement: VRAM for "gpu", system RAM for "cpu".
// For a MoE, "gpu" keeps only the non-expert weights on GPU and reserves the rest
// of VRAM for KV (experts offload to CPU), while "cpu" leaves VRAM for experts and
// puts the (large) KV in RAM. This is what makes --kv-placement drive the context
// ceiling instead of a fixed VRAM+RAM budget that can overflow a GPU-pinned KV.
func computeAutoContextSizeKVPlacement(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType, placement string, opts Options) (int, string) {
	freeVRAM := 0
	for _, g := range caps.GPUs {
		freeVRAM += g.VRAMFreeMB()
	}
	// Overhead is derived per-component, not a flat guess: VRAM side = measured
	// CUDA overhead + compute buffer per GPU; RAM side = host/graph/mmap/activation.
	sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
	vramOverheadMB := plannedPerGPUVRAMOverheadMB(sysProbe, 0, opts) * len(caps.GPUs)

	var kvBudgetMB int
	if placement == "cpu" {
		// KV lives in RAM, sharing it with the weights that don't fit in VRAM.
		weightsInRAM := totalSizeMB - freeVRAM
		if weightsInRAM < 0 {
			weightsInRAM = 0
		}
		kvBudgetMB = caps.RAM.FreeMB - weightsInRAM - plannedRAMRuntimeOverheadMB(caps, model, 0, totalSizeMB, opts)
	} else {
		// KV lives in VRAM alongside the GPU-resident weights. Dense: the whole
		// model. MoE: only the non-expert weights (experts offload to CPU), so the
		// rest of VRAM is free for KV.
		gpuResident := totalSizeMB
		if model.IsMoE {
			if ne := bytesToMiBCeil(model.NonExpertBytes); ne > 0 {
				gpuResident = ne
				// Input embeddings stay in host memory, never in VRAM.
				if te := bytesToMiBCeil(model.TokenEmbdBytes); te > 0 && te < ne {
					gpuResident = ne - te
				}
			}
		}
		kvBudgetMB = freeVRAM - gpuResident - vramOverheadMB
	}
	if kvBudgetMB <= 0 {
		return 32768, preferredKVType
	}

	for _, kvType := range kvTypesForAutoContext(preferredKVType, opts.KVQuality) {
		refCtx := 32768
		refKVMB := computeKVTotalMB(model, refCtx, kvType, opts.SWAFull)
		if refKVMB <= 0 {
			continue
		}
		kvBytesPerToken := float64(refKVMB) * 1048576.0 / float64(refCtx)
		maxCtx := int(float64(kvBudgetMB) * 1048576.0 / kvBytesPerToken)
		if model.CTXTrain > 0 && model.CTXTrain < maxCtx {
			maxCtx = model.CTXTrain
		}
		best := 0
		for _, c := range []int{32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304} {
			if c <= maxCtx {
				best = c
			}
		}
		if best >= 32768 {
			return best, kvType
		}
	}
	return 32768, fallbackKVType(preferredKVType, opts.KVQuality)
}

// computeAutoContextSize computes the largest context that fits in available
// hardware memory, .
// Uses TOTAL_VRAM + RAM_AVAIL.
func computeAutoContextSize(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType string, opts Options) (int, string) {
	// Total hardware = all GPU VRAM + free RAM
	totalVRAM := 0
	for _, g := range caps.GPUs {
		totalVRAM += g.VRAMTotalMB
	}
	totalHWMB := totalVRAM + caps.RAM.FreeMB

	// Fixed overhead: model weights + model-aware headroom.
	fixedOverheadMB := totalSizeMB + plannedModelAwareHeadroom(model, opts)

	// If model doesn't fit at all, return minimum
	if totalHWMB <= fixedOverheadMB {
		return 32768, preferredKVType
	}

	// KV budget = total hardware - model - headroom
	kvBudgetMB := totalHWMB - fixedOverheadMB
	if kvBudgetMB <= 0 {
		return 32768, preferredKVType
	}

	orderedTypes := kvTypesForAutoContext(preferredKVType, opts.KVQuality)

	for _, kvType := range orderedTypes {
		refCtx := 32768
		refKVTotalMB := computeKVTotalMB(model, refCtx, kvType, opts.SWAFull)
		if refKVTotalMB <= 0 {
			continue
		}
		kvBytesPerToken := float64(refKVTotalMB) * 1048576.0 / float64(refCtx)
		maxCtxRaw := int(float64(kvBudgetMB) * 1048576.0 / kvBytesPerToken)

		hwCapCtx := maxCtxRaw
		if model.CTXTrain > 0 && model.CTXTrain < hwCapCtx {
			hwCapCtx = model.CTXTrain
		}

		powerOfTwoValues := []int{32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304}
		suggestedCtx := 32768
		for _, c := range powerOfTwoValues {
			if c <= hwCapCtx {
				suggestedCtx = c
			}
		}

		if suggestedCtx >= 32768 {
			return suggestedCtx, kvType
		}
	}

	// Preset qualities may fall back to the compact type. An exact llama.cpp
	// type is user-owned, so preserve it and lower context instead.
	return 32768, fallbackKVType(preferredKVType, opts.KVQuality)
}

// Args converts a Strategy into llama-server command-line arguments.
func (s *Strategy) Args(modelPath string, port int) []string {
	host := s.Host
	if host == "" {
		host = "127.0.0.1"
	}
	args := []string{
		"-m", modelPath,
		"--host", host,
		"--port", fmt.Sprintf("%d", port),
		"--ctx-size", fmt.Sprintf("%d", s.ContextSize),
	}
	if s.FlashAttention {
		args = append(args, "--flash-attn", "on")
	}
	args = append(args,
		"-b", fmt.Sprintf("%d", s.BatchSize),
		"-ub", fmt.Sprintf("%d", s.UBatchSize),
		"--cache-type-k", s.KVType,
		"--cache-type-v", s.KVType,
	)
	// Tool calls require --jinja (the backend returns "tools param requires
	// --jinja flag" without it). A model whose template the Jinja engine cannot
	// auto-parse (nanbeige: raise_exception macro) gets a corrected template
	// override via --chat-template-file (see nanbeigeTemplateArgs in main.go),
	// keeping --jinja on. Always pass --jinja here.
	args = append(args, "--jinja")
	args = append(args,
		"--threads", fmt.Sprintf("%d", s.Threads),
		"--threads-batch", fmt.Sprintf("%d", s.ThreadsBatch),
	)

	if s.KVPlacement == "cpu" {
		args = append(args, "--no-kv-offload")
	} else if s.KVPlacement == "gpu" && s.BackendSupportsKVOffload {
		args = append(args, "--kv-offload")
	}

	if s.ReasoningOff {
		args = append(args, "--reasoning", "off")
	}

	// SSM/Mamba models need --no-context-shift
	if s.HasSSM {
		args = append(args, "--no-context-shift")
	}

	if s.Parallel > 0 {
		args = append(args, "--parallel", fmt.Sprintf("%d", s.Parallel))
	} else {
		args = append(args, "--parallel", "1")
	}

	// Vision support: auto-detected mmproj
	if s.MMProjPath != "" {
		args = append(args, "--mmproj", s.MMProjPath)
	}

	// GPU offloading. CPU-only still
	// prints -ngl 0 so compatibility tests and user scripts can see the mode.
	if s.Type == CPUOnly {
		args = append(args, "-ngl", "0")
	} else if s.Type == DenseCPUOffload && s.BackendSupportsFit {
		// Keep n_gpu_layers and tensor_split unset. Backend fit uses exact GGUF
		// tensor sizes to choose a safe GPU/CPU layer boundary at the requested
		// context. Explicit values make llama.cpp fit abort without changing them.
		args = append(args, "--fit")
		if s.BackendFitTakesValue {
			args = append(args, "on")
		}
	} else if len(s.TensorSplit) > 0 || s.Type != CPUOnly {
		args = append(args, "-ngl", "999")
		// Disable the backend's own memory auto-fit: ggrun already sets explicit
		// placement, so a second fit pass is redundant. Only emit the option when
		// the selected backend supports it.
		if s.BackendSupportsFit && s.BackendFitTakesValue {
			args = append(args, "--fit", "off")
		}
		// Metal has exactly one logical device — device-routing flags are
		// CUDA/Vulkan concepts and llama-server rejects unknown device names.
		if s.MainGPU >= 0 && len(s.TensorSplit) == 0 && !strings.EqualFold(s.BackendTag, "metal") {
			args = append(args, "-mg", fmt.Sprintf("%d", s.MainGPU))
			if s.Type == SingleGPU {
				args = append(args, "--device", deviceName(s.BackendTag, s.MainGPU))
			}
		}
	}

	if len(s.TensorSplit) > 0 {
		var splitStr string
		for i, v := range s.TensorSplit {
			if i > 0 {
				splitStr += ","
			}
			splitStr += fmt.Sprintf("%.2f", v)
		}
		args = append(args, "--tensor-split", splitStr)
	}

	if s.SplitMode != "" {
		args = append(args, "--split-mode", s.SplitMode)
	}

	if s.OTString != "" {
		args = append(args, "-ot", s.OTString)
	}

	if s.NCPUMoE > 0 {
		args = append(args, "--n-cpu-moe", fmt.Sprintf("%d", s.NCPUMoE))
	}

	if !s.MMap {
		args = append(args, "--no-mmap")
	}

	if s.MLock {
		args = append(args, "--mlock")
	}

	// CRAM is always a real, derived decision (never "not applicable") — 0
	// must reach the backend as an explicit "-cram 0" (disable), not silence
	// that lets llama-server fall back to its own 8192 MiB default. Same for
	// MaxCheckpoints when computeCRAM actually evaluated it (>= 0): nesting
	// this inside "CRAM > 0" used to mean a correctly-computed "0, disable
	// checkpoints — VRAM is too tight" was silently dropped, leaving
	// llama-server's default of 32 checkpoints active. That default's context
	// checkpoint save (tools/server/server-context.cpp create_checkpoint)
	// needs backend memory sched_reserve() never accounts for, and is exactly
	// what crashed DeepSeek-V4 mid-request on 2026-07-08 despite a placement
	// that had loaded clean and passed health check.
	args = append(args, "-cram", fmt.Sprintf("%d", s.CRAM))
	if s.MaxCheckpoints >= 0 {
		args = append(args, "--ctx-checkpoints", fmt.Sprintf("%d", s.MaxCheckpoints))
		// Spacing decides whether a checkpoint can ever sit at the point a
		// later turn resumes from, and llama.cpp's 8192 default is too coarse
		// for an agent workload.
		//
		// A checkpoint spans 2*n_swa tokens, and the search rejects any whose
		// pos_max overshoots the resume point. Measured here on a 16k prompt
		// with the default spacing: a 92% prefix match was found, the only two
		// checkpoints were [14728,15751] and [15240,16263], the resume point
		// was ~14906, and both were rejected for overshooting it by as little
		// as 845 tokens. Every one of 164,358 prompt tokens was re-processed.
		//
		// Spacing equal to the checkpoint's own width tiles the context without
		// gaps, so no resume point can fall between two of them. It is derived
		// from the model rather than chosen: the width is what the backend
		// already uses.
		if step := s.CheckpointMinStep; step > 0 {
			flag := s.BackendCheckpointMinStepFlag
			if flag == "" {
				flag = backendCheckpointMinStepFlag("", s.BackendTag)
			}
			if flag != "-" {
				args = append(args, flag, fmt.Sprintf("%d", step))
			}
		}
	}

	// ik_llama.cpp fork specific flags
	if s.BackendTag == "ik_llama" {
		args = append(args, "--run-time-repack")
		args = append(args, "-khad")
		args = append(args, "--defrag-thold", "0.1")

		if s.IsMoE {
			args = append(args, "-muge")
			args = append(args, "-ger")
		}

		if len(s.TensorSplit) > 0 || s.Type != CPUOnly {
			args = append(args, "-mqkv")
		}
	}

	// Speculative decoding flags (MTP, EAGLE-3, draft model, or explicit ngram)
	if s.Draft != nil && s.Draft.Type != DraftNone {
		args = append(args, DraftFlags(s.Draft)...)
	}

	// Server --timeout: slow local models and queued Workflow agents must not be
	// killed by a backend's 600s/3600s socket default. Backends do not agree that
	// zero means disabled, so use the largest signed 32-bit seconds value instead
	// (about 68 years). Process health and client disconnects still stop real work.
	args = append(args, "--timeout", "2147483647")

	return args
}

// systemProbeSchema is bumped when the meaning of a stored value changes, so a
// file written by an older method is re-measured rather than trusted.
const systemProbeSchema = 2

// systemProbeOutlierRatio is how far above its peers a stored per-GPU overhead
// must sit before it is treated as contaminated rather than measured. Real
// variation between a 24 GB and a 12 GB card is well under 2x; the contaminated
// reading on this project was 9x.
const systemProbeOutlierRatio = 4

// loadSystemProbe tries to load measured CUDA overhead from cache.
// Keys the probe cache by a GPU-signature hash.
// SystemCUDAOverheadMB returns the legacy measured CUDA context overhead. New
// placement/preflight code must use SystemCUDAOverheadByGPU so headroom remains
// per-device and measured-only; this helper is kept for old call sites/tests.
func SystemCUDAOverheadMB(cacheDir string, gpus []detect.GPU) int {
	if p := loadSystemProbe(cacheDir, gpus); p != nil && p.CUDAOverheadMB > 0 {
		return p.CUDAOverheadMB
	}
	return 0
}

// SystemCUDAOverheadByGPU returns measured CUDA context/allocator overhead keyed
// by CUDA device index. Missing entries are intentionally absent: callers must
// not fill them with static margins.
func SystemCUDAOverheadByGPU(cacheDir string, gpus []detect.GPU) map[int]int {
	sp := loadSystemProbe(cacheDir, gpus)
	if sp == nil {
		return nil
	}
	out := map[int]int{}
	for k, v := range sp.CUDAOverheadByGPU {
		if v > 0 {
			out[k] = v
		}
	}
	if len(out) == 0 && sp.CUDAOverheadMB > 0 {
		for _, g := range gpus {
			out[g.Index] = sp.CUDAOverheadMB
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func loadSystemProbe(cacheDir string, gpus []detect.GPU) *systemProbe {
	explicitCacheDir := cacheDir != ""
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	// Compute GPU signature hash: sort(names+drivers), MD5, take first 12 chars
	gpuSig := gpuSignatureHash(gpus)
	path := filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuSig))
	data, err := os.ReadFile(path)
	if err != nil && explicitCacheDir {
		// App-local installs used to read ~/.cache/ggrun before LLM_APP_HOME
		// became authoritative. Preserve those measured CUDA values and migrate
		// them lazily instead of treating the missing new-path file as zero.
		home, _ := os.UserHomeDir()
		legacyPath := filepath.Join(home, ".cache", "ggrun", fmt.Sprintf("system_%s.cache", gpuSig))
		if legacyPath != path {
			if legacyData, legacyErr := os.ReadFile(legacyPath); legacyErr == nil {
				data, err = legacyData, nil
				if mkErr := os.MkdirAll(filepath.Dir(path), 0755); mkErr == nil {
					_ = atomicWriteFile(path, legacyData, 0o644)
				}
			}
		}
	}
	if err != nil {
		return nil
	}
	sp := &systemProbe{CUDAOverheadByGPU: map[int]int{}}
	sysSchema := 1
	lines := strings.Split(string(data), "\n")
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
		val := strings.TrimSpace(strings.Trim(parts[1], `"`))
		switch {
		case key == "SYS_CUDA_OVERHEAD_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				sp.CUDAOverheadMB = v
			}
		case key == "SYS_PROBE_SCHEMA":
			if v, err := strconv.Atoi(val); err == nil {
				sysSchema = v
			}
		case key == "SYS_HOST_OVERHEAD_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				sp.HostOverheadMB = v
			}
		case strings.HasPrefix(key, "SYS_CUDA_OVERHEAD_MB_CUDA"):
			idxRaw := strings.TrimPrefix(key, "SYS_CUDA_OVERHEAD_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				sp.CUDAOverheadByGPU[idx] = v
			}
		}
	}
	if sp.CUDAOverheadMB == 0 && len(sp.CUDAOverheadByGPU) == 0 {
		return nil
	}
	// Schema 1 measured CUDA overhead as the whole device's usage minus the
	// server's own logged buffers, so anything else resident on that card was
	// recorded as permanent overhead. On this project the Auto reviewer, seated
	// on the least valuable GPU by design, was written in as 2299 MiB of
	// "overhead" against 397 and 255 on the other two cards -- then charged on
	// every later placement on top of the reviewer's own reservation, costing
	// the smallest GPU a full expert layer.
	//
	// Only an outlier is discarded, not every legacy value: CUDA context
	// overhead is broadly similar across devices sharing a driver, so a card
	// several times its peers is measuring something else. A single-GPU record
	// has no peer to compare against and is kept.
	if sysSchema < systemProbeSchema && len(sp.CUDAOverheadByGPU) > 1 {
		lo := 0
		for _, v := range sp.CUDAOverheadByGPU {
			if v > 0 && (lo == 0 || v < lo) {
				lo = v
			}
		}
		if lo > 0 {
			for idx, v := range sp.CUDAOverheadByGPU {
				if v > lo*systemProbeOutlierRatio {
					delete(sp.CUDAOverheadByGPU, idx)
				}
			}
		}
	}
	return sp
}

// gpuSignatureHash computes MD5 hash of sorted GPU name+driver pairs.
// GPU signature: nvidia-smi --query-gpu=name,driver_version | sort | md5sum | cut -c1-12
func gpuSignatureHash(gpus []detect.GPU) string {
	var parts []string
	for _, g := range gpus {
		// Stable hardware identity only: never include current free/used VRAM,
		// but do include topology and capacity. Two same-name cards on x1 and
		// x16 links are materially different placement hardware, as are two
		// revisions carrying different VRAM sizes.
		parts = append(parts, fmt.Sprintf("%d|%s|%d|%s|%s|%s|gen%d|x%d|bw%d",
			g.Index, g.Name, g.VRAMTotalMB, g.Driver, g.ComputeCap, g.PCIBusID,
			g.PCIGen, g.PCILanes, g.BandwidthMBps))
	}
	sort.Strings(parts)
	input := strings.Join(parts, "\n") + "\n"
	h := md5.New()
	h.Write([]byte(input))
	return fmt.Sprintf("%x", h.Sum(nil))[:12]
}

// RunPostLaunchProbe measures actual CUDA overhead after a successful server launch.
// It reads current VRAM usage from nvidia-smi, parses buffer sizes from the
// server's captured stderr log, and caches the result for future launches.
// Parses the server log after launch to record measured overhead.
// kvCachePath is the per-model cache of measured KV bytes-per-token. Keyed by
// model basename + byte size so requantizations/different models never collide.
func kvCachePath(cacheDir string, model *ModelProfile) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	base := model.Basename
	if base == "" {
		base = filepath.Base(model.Path)
	}
	return filepath.Join(cacheDir, fmt.Sprintf("kv_%s_%d.cache", base, model.SizeBytes))
}

// loadMeasuredKVRates reads the per-model KV cache into a kvType→bytes/token map.
func loadMeasuredKVRates(cacheDir string, model *ModelProfile) map[string]float64 {
	path := kvCachePath(cacheDir, model)
	data, err := os.ReadFile(path)
	if err != nil && cacheDir != "" {
		// See loadSystemProbe: older installs wrote exact KV measurements to the
		// user cache even when the current app uses an app-local cache directory.
		// A compressed-attention model can be overestimated by many GiB without
		// this value, so migrate the measurement rather than reverting to formula.
		legacyPath := kvCachePath("", model)
		if legacyPath != path {
			if legacyData, legacyErr := os.ReadFile(legacyPath); legacyErr == nil {
				data, err = legacyData, nil
				if mkErr := os.MkdirAll(filepath.Dir(path), 0755); mkErr == nil {
					_ = atomicWriteFile(path, legacyData, 0o644)
				}
			}
		}
	}
	if err != nil {
		return nil
	}
	out := map[string]float64{}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// format: KV_BYTES_PER_TOK_<kvtype>=<float>
		const pfx = "KV_BYTES_PER_TOK_"
		if !strings.HasPrefix(line, pfx) {
			continue
		}
		kv := strings.SplitN(strings.TrimPrefix(line, pfx), "=", 2)
		if len(kv) != 2 {
			continue
		}
		if v, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64); err == nil && v > 0 {
			out[strings.ToLower(strings.TrimSpace(kv[0]))] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// loadMeasuredKVGeometry reads the per-KV-type cache layout recorded by a
// previous launch. Kept separate from the rate loader so an older cache file
// that predates the geometry still yields its rate rather than nothing.
func loadMeasuredKVGeometry(cacheDir string, model *ModelProfile) map[string]KVGeometry {
	data, err := os.ReadFile(kvCachePath(cacheDir, model))
	if err != nil {
		return nil
	}
	out := map[string]KVGeometry{}
	const pfx = "KV_GEOMETRY_"
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, pfx) {
			continue
		}
		kv := strings.SplitN(strings.TrimPrefix(line, pfx), "=", 2)
		if len(kv) != 2 {
			continue
		}
		f := strings.Split(kv[1], ",")
		if len(f) != 4 {
			continue
		}
		full, e1 := strconv.Atoi(strings.TrimSpace(f[0]))
		swa, e2 := strconv.Atoi(strings.TrimSpace(f[1]))
		cells, e3 := strconv.Atoi(strings.TrimSpace(f[2]))
		bpc, e4 := strconv.ParseFloat(strings.TrimSpace(f[3]), 64)
		if e1 != nil || e2 != nil || e3 != nil || e4 != nil {
			continue
		}
		g := KVGeometry{FullLayers: full, SWALayers: swa, SWACells: cells, BytesPerCellPerLayer: bpc}
		if g.Measured() {
			out[strings.ToLower(strings.TrimSpace(kv[0]))] = g
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// parseKVBufferTotalMB extracts the model's TOTAL KV cache allocation (MiB) at the
// launched context from a backend log. llama.cpp's wording varies across versions
// and backends, so match all known forms: an aggregate "KV self size = X MiB" /
// "KV cache size = X MiB" line (already the total — take it directly), otherwise
// SUM the per-device "... KV buffer size = X MiB" lines across CUDA devices + CPU.
// Returns 0 when the log carries no KV line (caller falls back to the formula or
// the VRAM-delta probe).
func parseKVBufferTotalMB(log string) float64 {
	var aggregate, bufSum float64
	for _, line := range strings.Split(log, "\n") {
		low := strings.ToLower(line)
		if !strings.Contains(low, "=") {
			continue
		}
		switch {
		case strings.Contains(low, "kv self size"), strings.Contains(low, "kv cache size"):
			if v := parseMiB(line); v > aggregate {
				aggregate = v // aggregate line: the total, printed once
			}
		case strings.Contains(low, "kv buffer size"):
			bufSum += parseMiB(line) // per-device: sum across GPUs + CPU
		}
	}
	if aggregate > 0 {
		return aggregate
	}
	return bufSum
}

// kvBytesPerTokenFromVRAMDelta derives KV bytes-per-token from two launches that
// differ ONLY in context size. Weights, compute buffers, and CUDA overhead are
// identical across the two, so the VRAM difference is pure KV cache — exact for
// every architecture and independent of whether the backend logs its KV size at
// all. Returns 0 if the samples are unusable.
func kvBytesPerTokenFromVRAMDelta(ctxA, vramA_MB, ctxB, vramB_MB int) float64 {
	dCtx := ctxB - ctxA
	dVRAM := vramB_MB - vramA_MB
	if dCtx < 0 {
		dCtx, dVRAM = -dCtx, -dVRAM
	}
	if dCtx == 0 || dVRAM <= 0 {
		return 0
	}
	return float64(dVRAM) * 1048576.0 / float64(dCtx)
}

// setCtxSizeArg returns a copy of args with --ctx-size set to ctx (adding it if
// absent). Used by the VRAM-delta probe to launch the same placement twice at
// different contexts.
func setCtxSizeArg(args []string, ctx int) []string {
	out := make([]string, len(args))
	copy(out, args)
	for i := 0; i < len(out)-1; i++ {
		if out[i] == "--ctx-size" || out[i] == "-c" {
			out[i+1] = strconv.Itoa(ctx)
			return out
		}
	}
	return append(out, "--ctx-size", strconv.Itoa(ctx))
}

// measureLoadedVRAM launches backendPath with args, waits for VRAM to plateau
// (the model + KV finished allocating), returns total VRAM used across gpus (MiB),
// then kills the process. Log-independent — it reads nvidia-smi, not stderr.
func measureLoadedVRAM(backendPath string, args []string, gpus []detect.GPU, timeout time.Duration) int {
	cmd := exec.Command(backendPath, args...)
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return 0
	}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()

	deadline := time.Now().Add(timeout)
	prev, stable := -1, 0
	for time.Now().Before(deadline) {
		time.Sleep(1500 * time.Millisecond)
		total := 0
		for _, g := range gpus {
			total += QueryVRAMUsed(g.Index)
		}
		// Plateau = two consecutive readings within 64 MiB, above a floor.
		if total > 512 && prev > 512 && absInt(total-prev) <= 64 {
			stable++
			if stable >= 2 {
				return total
			}
		} else {
			stable = 0
		}
		prev = total
	}
	if prev > 512 {
		return prev
	}
	return 0
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// ProbeKVViaVRAMDelta measures a model's KV bytes-per-token by launching the same
// placement (baseArgs, minus the binary) twice — at a small and a larger context —
// and attributing the VRAM difference entirely to KV. It is the log-independent
// fallback for backends that don't print their KV buffer size. Requires roughly
// idle GPUs for a clean delta. Writes the per-model KV cache and returns true on
// success. Best-effort: returns false if either launch fails to allocate.
//
// NOTE: the two-launch flow needs live validation on real hardware before it is
// auto-invoked; the arithmetic (kvBytesPerTokenFromVRAMDelta) and cache round-trip
// are unit-tested.
func ProbeKVViaVRAMDelta(backendPath string, baseArgs []string, gpus []detect.GPU, cacheDir string, model *ModelProfile, kvType string) bool {
	if backendPath == "" || model == nil || len(gpus) == 0 {
		return false
	}
	if kvType == "" {
		kvType = "q8_0"
	}
	const ctxA, ctxB = 8192, 65536
	loadTimeout := 15 * time.Minute

	vramA := measureLoadedVRAM(backendPath, setCtxSizeArg(baseArgs, ctxA), gpus, loadTimeout)
	if vramA <= 0 {
		return false
	}
	vramB := measureLoadedVRAM(backendPath, setCtxSizeArg(baseArgs, ctxB), gpus, loadTimeout)
	if vramB <= 0 {
		return false
	}
	rate := kvBytesPerTokenFromVRAMDelta(ctxA, vramA, ctxB, vramB)
	if rate <= 0 {
		return false
	}
	writeMeasuredKVRate(cacheDir, model, strings.ToLower(kvType), rate,
		fmt.Sprintf("VRAM-delta probe: ctx %d=%dMB, ctx %d=%dMB", ctxA, vramA, ctxB, vramB))
	return true
}

// RunPostLaunchKVProbe reads the KV cache size llama.cpp actually allocated at
// ctxSize from the backend log and caches it as bytes-per-token for this model +
// kvType, so future launches size the context from measured truth instead of the
// per-arch GGUF formula. A successful launch deliberately refreshes a preflight
// estimate because the final buffer log is the most precise measurement.
func RunPostLaunchKVProbe(cacheDir string, model *ModelProfile, ctxSize int, kvType, serverLog string) {
	if model == nil || ctxSize <= 0 || serverLog == "" {
		return
	}
	if requiresScopedContextEvidence(model) {
		// recordMeasuredLaunchProbes stores this total in the exact runtime key.
		// A global rate cannot represent this layout and used to poison every
		// later DeepSeek-V4 plan after one parallel or --swa-full launch.
		return
	}
	if kvType == "" {
		kvType = "q8_0"
	}
	kvType = strings.ToLower(kvType)
	totalKVMB := parseKVBufferTotalMB(serverLog)
	if totalKVMB <= 0 {
		return
	}
	bytesPerTok := totalKVMB * 1048576.0 / float64(ctxSize)
	if bytesPerTok <= 0 {
		return
	}
	// The geometry is strictly better than the rate and comes from the same
	// log, so record it whenever the backend printed its per-cache breakdown.
	g, ok := ParseKVGeometry(serverLog)
	if !ok && kvCacheLogIsSWAFlattened(serverLog) {
		// A --swa-full launch allocates every layer at full depth, so both the
		// layout and the rate it produces describe that launch and no other.
		// Storing either one poisons the next plain launch, which would then
		// reserve the swa-full total: measured here as 55296 B/token against a
		// true 13864, a 4x over-reservation that costs expert layers on every
		// GPU. Keep whatever an earlier non-flattened launch measured.
		return
	}
	if ok {
		if model.MeasuredKVGeometry == nil {
			model.MeasuredKVGeometry = map[string]KVGeometry{}
		}
		model.MeasuredKVGeometry[kvType] = g
	}
	writeMeasuredKVRate(cacheDir, model, kvType, bytesPerTok,
		fmt.Sprintf("launch log: ctx=%d total_kv=%.0fMB", ctxSize, totalKVMB))
}

// RecordPostLaunchContextAllocation stores the exact total context allocation
// printed by a healthy backend under the full runtime signature. It complements
// fit-params/guarded preflight on forks that expose no complete no-allocation
// oracle, without promoting the number to a model-wide extrapolation.
func RecordPostLaunchContextAllocation(cacheDir string, model *ModelProfile, strategy *Strategy,
	backendTag string, gpus []detect.GPU, serverLog string,
) bool {
	if model == nil || strategy == nil || strings.TrimSpace(serverLog) == "" {
		return false
	}
	total := int(parseKVBufferTotalMB(serverLog) + 0.5)
	if total <= 0 {
		return false
	}
	err := RecordMeasuredAllocation(cacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, backendTag, gpus, strategy.Parallel,
		MeasuredAllocation{Evidence: "live-allocated", ContextTotalMB: total})
	return err == nil
}

// MeasuredAllocation is the fixed memory ledger reported by the selected
// backend for one exact launch signature. Device maps use the CUDA indices the
// backend sees after device filtering. Evidence is intentionally a string so
// cmd/ggrun can preserve the backend probe's richer level without creating a
// package dependency cycle.
type MeasuredAllocation struct {
	Evidence          string
	ContextTotalMB    int
	ContextByGPU      map[int]int
	ContextHostMB     int
	ModelByGPU        map[int]int
	ModelHostMB       int
	UnaccountedByGPU  map[int]int
	UnaccountedHostMB int
}

// RecordMeasuredAllocation stores a backend allocation under the same exact
// model/runtime/backend/hardware key consumed by Compute. It does not update
// model.MeasuredKVBytesPerTok or KVGeometry: those model-wide extrapolations
// cannot represent heterogeneous recurrent/attention cache regions safely.
func RecordMeasuredAllocation(cacheDir string, model *ModelProfile, ctxSize, ubatch int,
	kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int,
	allocation MeasuredAllocation,
) error {
	if model == nil || ctxSize <= 0 || ubatch <= 0 {
		return nil
	}
	if allocation.ContextTotalMB <= 0 {
		allocation.ContextTotalMB = allocation.ContextHostMB
		for _, value := range allocation.ContextByGPU {
			if value > 0 {
				allocation.ContextTotalMB += value
			}
		}
	}
	if allocation.ContextTotalMB <= 0 {
		return nil
	}
	if strings.TrimSpace(allocation.Evidence) == "" {
		allocation.Evidence = "backend-measured"
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement,
		backendTag, gpus, parallel, nil, nil, nil, 0, probeMeasurements{
			AllocationSet:      true,
			ContextTotalMB:     allocation.ContextTotalMB,
			ContextByGPU:       copyProbeIntMap(allocation.ContextByGPU),
			ContextHostMB:      allocation.ContextHostMB,
			ModelByGPU:         copyProbeIntMap(allocation.ModelByGPU),
			ModelHostMB:        allocation.ModelHostMB,
			UnaccountedByGPU:   copyProbeIntMap(allocation.UnaccountedByGPU),
			UnaccountedHostMB:  allocation.UnaccountedHostMB,
			AllocationEvidence: allocation.Evidence,
		})
}

// LoadMeasuredAllocation returns immutable copies of exact allocation evidence
// for this signature. A different context, ubatch, slot count, backend build,
// hardware signature, workload, KV placement, or feature state is a cache miss.
func LoadMeasuredAllocation(cacheDir string, model *ModelProfile, ctxSize, ubatch int,
	kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int,
) (MeasuredAllocation, bool) {
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if pc == nil || pc.ContextTotalMB <= 0 {
		return MeasuredAllocation{}, false
	}
	return MeasuredAllocation{
		Evidence:          pc.AllocationEvidence,
		ContextTotalMB:    pc.ContextTotalMB,
		ContextByGPU:      copyProbeIntMap(pc.ContextByGPU),
		ContextHostMB:     pc.ContextHostMB,
		ModelByGPU:        copyProbeIntMap(pc.ModelByGPU),
		ModelHostMB:       pc.ModelHostMB,
		UnaccountedByGPU:  copyProbeIntMap(pc.UnaccountedByGPU),
		UnaccountedHostMB: pc.UnaccountedHostMB,
	}, true
}

// RecordMeasuredContextMB records the backend-authoritative total context
// allocation reported by llama-fit-params. For placement purposes this is the
// exact quantity computeKVTotalMB must reserve: summing every device row keeps
// GPU-, CPU-, and mixed-KV placements on the same accounting path. The model is
// updated in memory as well as on disk so an immediate preflight re-plan sees
// the measurement without waiting for another process or launch.
func RecordMeasuredContextMB(cacheDir string, model *ModelProfile, ctxSize int, kvType string, totalContextMB int, swaFull bool) {
	if model == nil || ctxSize <= 0 || totalContextMB <= 0 {
		return
	}
	if requiresScopedContextEvidence(model) {
		return
	}
	// Same hazard the launch-log probe already refuses, reached by the other
	// path. Under --swa-full every windowed layer is allocated at full depth, so
	// the total describes that launch alone; stored as a plain per-token rate it
	// makes the next launch reserve the swa-full figure. Measured on Laguna:
	// 6912 MB at ctx 131072 recorded 55296 B/token against a true 13864, and the
	// resulting 4x over-reservation pushed auto-fit to force KV onto the host.
	if swaFull && model.SlidingWindow > 0 {
		return
	}
	if kvType == "" {
		kvType = "q8_0"
	}
	kvType = strings.ToLower(kvType)
	bytesPerTok := float64(totalContextMB) * 1048576.0 / float64(ctxSize)
	if bytesPerTok <= 0 {
		return
	}
	writeMeasuredKVRate(cacheDir, model, kvType, bytesPerTok,
		fmt.Sprintf("fit-params preflight: ctx=%d total_context=%dMB", ctxSize, totalContextMB))
}

// RecordMeasuredComputeBuffers records per-GPU compute-buffer MiB measured by
// the no-alloc fit-params preflight (cmd/ggrun/preflight.go) for the exact
// ctx/ubatch/KV/backend/GPU-set key, merging with (not clobbering) any
// runtime-growth or KV-per-layer data already cached for that same key. This
// lets the FIRST launch attempt for a shape use real numbers instead of the
// first-launch heuristic (ubatch*4, clamped 1024-4096) — which measurably
// under-estimates flash-attention's compute buffer at large context (e.g.
// ~17-20GB actual vs a 4096MB clamp for DeepSeek-V4 at ctx 1M, f16 KV).
func RecordMeasuredComputeBuffers(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, computeByGPU map[int]int) error {
	if model == nil || ctxSize <= 0 || ubatch <= 0 || len(computeByGPU) == 0 {
		return nil
	}
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	growth := map[int]int{}
	estimatedByGPU := map[int]bool{}
	kvPerLayerMB := 0
	if pc != nil {
		for k, v := range pc.RuntimeGraphGrowthByGPU {
			growth[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthEstimatedByGPU {
			estimatedByGPU[k] = v
		}
		kvPerLayerMB = pc.KVPerLayerMB
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, computeByGPU, growth, estimatedByGPU, kvPerLayerMB)
}

// RunPostLaunchModelProbe records measured compute-buffer data for the exact
// model/runtime placement that just loaded. Future placement can use this instead
// of the first-launch compute estimate.
// RuntimeGraphGrowthByGPU returns measured post-health graph growth keyed by CUDA
// device. These values are populated only from observed runtime allocation growth
// or exact cudaMalloc failures for the same runtime signature; missing means
// unknown, not zero-margin proof.
func RuntimeGraphGrowthByGPU(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) map[int]int {
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if pc == nil || len(pc.RuntimeGraphGrowthByGPU) == 0 {
		return nil
	}
	out := map[int]int{}
	for k, v := range pc.RuntimeGraphGrowthByGPU {
		if v > 0 {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// RelatedModelRuntimeGraphGrowth carries a MEASURED (non-estimated) runtime-growth
// value from a RELATED probe key of the same model artifact into a cold key, so
// the first real launch on a new context/ubatch (which cannot measure growth
// before a request exists) does not launch blind and OOM. It is NOT a static
// margin: the value is a recorded backend observation, filtered through the
// existing estimated-bit machinery, applied only to the device(s) that actually
// carried the growth, and it self-heals once the cold key records its own value.
//
// Match width: SpecTargetIdentity (via the probe header model basename) +
// gpuSignatureHash. ctx/ubatch/kv_quality/kv_placement are relaxed — runtime
// growth is model-graph state that scales with model shape, not context/ubatch.
//
// parallel is NOT relaxed. A measurement from a different slot count is not
// evidence for this launch, and the reason is stronger than it looks: the
// recorder (RecordRuntimeGraphGrowthFromOOM) files the size of WHATEVER
// allocation failed, so a probe can hold a KV-buffer figure mislabelled as
// growth. Verified 2026-08-05 on this project: CUDA0 carried 5504, which is
// exactly that plan's "CUDA0 KV buffer size = 5504.00 MiB". KV is budgeted
// separately, so carrying such an entry double-counts it and would strand
// several GB. Widening the match only spreads a bad measurement further; the
// fix belongs in the recorder, not here.
//
// This does NOT relax the compute-buffer cache either -- slot count really does
// change that measurement.
func RelatedModelRuntimeGraphGrowth(cacheDir string, model *ModelProfile, gpus []detect.GPU, parallel int, backendTag string) map[int]int {
	if model == nil || cacheDir == "" {
		return nil
	}
	modelBase := filepath.Base(model.Path)
	wantSig := gpuSignatureHash(gpus)
	wantParallel := probeParallelKey(parallel)
	exactByDevice := map[int]int{}
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		return nil
	}
	for _, ent := range entries {
		if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".probe") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(cacheDir, ent.Name()))
		if err != nil {
			continue
		}
		lines := strings.Split(string(data), "\n")
		var headerKV, sig string
		var par int
		fileSchema := 1
		matched := false
		hasEstimate := map[int]bool{}
		growth := map[int]int{}
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			if strings.HasPrefix(line, "PROBE_CACHE_SCHEMA=") {
				if v, convErr := strconv.Atoi(strings.TrimPrefix(line, "PROBE_CACHE_SCHEMA=")); convErr == nil && v > 0 {
					fileSchema = v
				}
			}
			// Model identity: the first header line is "# Probe cache for <basename>".
			if strings.HasPrefix(line, "# Probe cache for ") {
				headerKV = strings.TrimPrefix(line, "# Probe cache for ")
			}
			// Key line: "# ctx=.. ubatch=.. kv_quality=.. kv_placement=.. backend=.. gpu_sig=.. parallel=.."
			if strings.HasPrefix(line, "# ctx=") {
				for _, kv := range strings.Fields(line) {
					switch {
					case strings.HasPrefix(kv, "gpu_sig="):
						sig = strings.TrimPrefix(kv, "gpu_sig=")
					case strings.HasPrefix(kv, "parallel="):
						fmt.Sscanf(strings.TrimPrefix(kv, "parallel="), "%d", &par)
					}
				}
			}
			// Match on model basename + gpu_sig. backend is part of the backend=
			// value in the key line but its exact form varies; relax it here (the
			// gpu_sig + same-model match is the load-bearing filter). Slot count
			// is recorded rather than required, so the two tiers can be split
			// after parsing.
			if headerKV == modelBase && sig == wantSig {
				matched = true
			}
			if strings.HasPrefix(line, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA") {
				var dev int
				var v int
				if _, err := fmt.Sscanf(line, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA%d=%d", &dev, &v); err == nil {
					growth[dev] = v
				}
			}
			if strings.HasPrefix(line, "PROBED_RUNTIME_GRAPH_GROWTH_ESTIMATED_CUDA") {
				var dev int
				var v int
				if _, err := fmt.Sscanf(line, "PROBED_RUNTIME_GRAPH_GROWTH_ESTIMATED_CUDA%d=%d", &dev, &v); err == nil && v != 0 {
					hasEstimate[dev] = true
				}
			}
		}
		if !matched {
			continue
		}
		// Same reasoning as loadProbeCache: an ungated growth figure carried
		// across models is a reserve nothing can justify.
		if growthPredatesServingGate(fileSchema) {
			continue
		}
		if probeParallelKey(par) != wantParallel {
			continue
		}
		for dev, v := range growth {
			if hasEstimate[dev] {
				continue // estimated (guessed) growth is not evidence for a carry
			}
			if v > exactByDevice[dev] {
				exactByDevice[dev] = v
			}
		}
	}
	byDevice := exactByDevice
	if len(byDevice) == 0 {
		return nil
	}
	return byDevice
}

// HasRuntimeGraphGrowthProbe reports whether all active GPUs have measured
// runtime-growth data for this exact runtime signature. This is a verification
// marker for future agents: no static fallback is hidden behind this predicate.
func HasRuntimeGraphGrowthProbe(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) bool {
	growth := RuntimeGraphGrowthByGPU(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if len(gpus) == 0 || len(growth) == 0 {
		return false
	}
	for _, g := range gpus {
		if _, ok := growth[g.Index]; !ok {
			return false
		}
	}
	return true
}

// growthPredatesServingGate reports whether a probe file's runtime-growth
// entries were written before the recorder learned to distinguish a failure at
// runtime from one during load. Such entries are indistinguishable from a
// misfiled load-time allocation and are dropped rather than reserved forever.
func growthPredatesServingGate(schemaVersion int) bool {
	// Pinned to the version that introduced the gate, not to the current schema:
	// a later bump for an unrelated reason must not start discarding growth that
	// was gated correctly.
	return schemaVersion < probeGrowthGateSchema
}

// probeGrowthGateSchema is the schema at which runtime-graph-growth entries
// began passing the post-serving window gate.
const probeGrowthGateSchema = 6

// RecordRuntimeGraphGrowth stores per-device runtime graph growth for the
// current runtime signature.
//
// A measured value is an exact cudaMalloc request parsed from the backend log,
// or a VRAM delta after a canary. Those accumulate by maximum, because a larger
// observed allocation is strictly better evidence than a smaller one.
//
// An estimated value is a fallback used when a CUDA VMM failure carries no
// allocation size: a flat fraction of the card, which is a guess about the
// hardware rather than an observation of it. Estimates must not accumulate. Two
// aborts on a 24 GiB card previously compounded to 4914 MiB -- 3.6 expert
// layers withheld permanently -- and one of those aborts was a malformed launch
// that proved nothing about capacity. A second guess is not twice the evidence.
//
// A measurement always supersedes an estimate, whatever their sizes: a real
// allocation size is better evidence than a fraction of the card even when it
// is smaller. Backing off is safe here because a CUDA out-of-memory aborts the
// backend process, which ggrun's recovery derates and restarts; it does not
// take the host down.
func RecordRuntimeGraphGrowth(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, growthByGPU map[int]int) error {
	return recordRuntimeGraphGrowth(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, growthByGPU, false)
}

func recordRuntimeGraphGrowth(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, growthByGPU map[int]int, estimated bool) error {
	if model == nil || ctxSize <= 0 || ubatch <= 0 || len(growthByGPU) == 0 {
		return nil
	}
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	computeByGPU := map[int]int{}
	kvPerLayerMB := 0
	mergedGrowth := map[int]int{}
	mergedEstimated := map[int]bool{}
	if pc != nil {
		for k, v := range pc.ComputeBufByGPU {
			computeByGPU[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthByGPU {
			mergedGrowth[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthEstimatedByGPU {
			mergedEstimated[k] = v
		}
		kvPerLayerMB = pc.KVPerLayerMB
	}
	for idx, v := range growthByGPU {
		prior, had := mergedGrowth[idx]
		priorEstimated := mergedEstimated[idx]
		switch {
		case !had:
			// Nothing known yet: take it, remembering how it was obtained.
			mergedGrowth[idx] = v
			mergedEstimated[idx] = estimated
		case !estimated && priorEstimated:
			// Measurement replaces a guess outright, even downwards.
			mergedGrowth[idx] = v
			mergedEstimated[idx] = false
		case estimated && !priorEstimated:
			// A guess must not raise a value that was actually observed.
		default:
			// Same kind of evidence on both sides: keep the larger. For
			// estimates this caps them rather than summing, which is what made
			// repeated aborts compound.
			if v > prior {
				mergedGrowth[idx] = v
			}
		}
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, computeByGPU, mergedGrowth, mergedEstimated, kvPerLayerMB)
}

// RecordRuntimeGraphGrowthFromOOM records a runtime graph allocation observed in
// a cudaMalloc OOM line. When estimated is false, allocMB is the size the
// backend actually asked for, which is measured accounting rather than a
// margin. When it is true the backend gave no size and allocMB is a fallback
// fraction of the card, which is stored as such so a later measurement can
// replace it.
func RecordRuntimeGraphGrowthFromOOM(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel, device, allocMB int, estimated bool) error {
	if device < 0 || allocMB <= 0 {
		return nil
	}
	return recordRuntimeGraphGrowth(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, map[int]int{device: allocMB}, estimated)
}

// ClearRuntimeGraphGrowth removes learned growth for one runtime signature so it
// can be derived again. Nothing else shrinks these values, so a scope taxed by a
// crash that no longer reproduces has no other way back.
func ClearRuntimeGraphGrowth(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) error {
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if pc == nil {
		return nil
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel,
		pc.ComputeBufByGPU, map[int]int{}, map[int]bool{}, pc.KVPerLayerMB, probeMeasurements{ClearRuntimeGrowth: true})
}

// RunPostLaunchModelProbeVRAMDelta writes per-GPU compute-buffer probe cache from
// nvidia-smi VRAM delta (current - baseline) instead of log parsing. Log-independent;
// works even when the server binary suppresses LLAMA_LOG_INFO. It estimates model
// weight per GPU from the placement OT assignments, subtracts that from the VRAM
// delta, and stores the remainder as the measured compute-buffer value for that GPU.
func computeBuffersFromVRAMDelta(
	model *ModelProfile, strategy *Strategy, gpus []detect.GPU,
	baselineVRAMByGPU, usedVRAMByGPU, overheadByGPU map[int]int,
) map[int]int {
	if model == nil || strategy == nil || model.NumLayers <= 0 || len(gpus) == 0 {
		return nil
	}
	// Partial gate+up pins do not expose their byte size in the OT string. A
	// guessed model subtraction would turn this fallback into poisoned compute
	// data, so rely on fit/log probes for those placements.
	if strings.Contains(strategy.OTString, `ffn_(gate_up|up_gate|gate|up)_(ch|)exps`) {
		return nil
	}

	assignments := parseOTAssignments(strategy.OTString)
	expertLayersByGPU := map[int]int{}
	for _, a := range assignments {
		expertLayersByGPU[a.CUDAIndex] += a.Count
	}
	moeLayers := model.NumLayers - model.LeadingDense
	if moeLayers <= 0 {
		moeLayers = model.NumLayers
	}
	expertPerLayerMB := ceilDivInt(bytesToMiBCeil(model.ExpertBytes), moeLayers)

	nonExpertTotalMB := bytesToMiBCeil(model.NonExpertBytes)
	if tokenMB := bytesToMiBCeil(model.TokenEmbdBytes); tokenMB > 0 && tokenMB < nonExpertTotalMB {
		nonExpertTotalMB -= tokenMB
	}
	outputMB := bytesToMiBCeil(model.OutputBytes)
	if outputMB > 0 && outputMB < nonExpertTotalMB {
		nonExpertTotalMB -= outputMB
	} else {
		outputMB = 0
	}
	shexpTotalMB := bytesToMiBCeil(model.ShexpBytes)
	if shexpTotalMB < 0 || shexpTotalMB >= bytesToMiBCeil(model.ExpertBytes) {
		shexpTotalMB = 0
	}
	owned, outputDev := layerOwnership(strategy.TensorSplit, model.NumLayers)
	layerDevices, _ := layerDeviceAssignments(strategy.TensorSplit, model.NumLayers)
	moeOwned := make([]int, len(strategy.TensorSplit))
	moeStart := model.LeadingDense
	if moeStart < 0 || moeStart >= model.NumLayers {
		moeStart = 0
	}
	for layer := moeStart; layer < len(layerDevices); layer++ {
		if dev := layerDevices[layer]; dev >= 0 && dev < len(moeOwned) {
			moeOwned[dev]++
		}
	}

	kvTotalMB := 0
	if strings.EqualFold(strategy.KVPlacement, "gpu") && strategy.ContextSize > 0 {
		kvTotalMB = computeKVTotalMB(model, strategy.ContextSize, strategy.KVType, strategy.SWAFull)
	}

	computeByGPU := map[int]int{}
	for gi, g := range gpus {
		overheadMB, measured := overheadByGPU[g.Index]
		if !measured {
			continue
		}
		usedMB := usedVRAMByGPU[g.Index]
		baselineMB := baselineVRAMByGPU[g.Index]
		if usedMB <= baselineMB {
			continue
		}

		modelMB := expertLayersByGPU[g.Index] * expertPerLayerMB
		if gi < len(owned) && owned[gi] > 0 {
			modelMB += ownedShareMB(nonExpertTotalMB, owned, model.NumLayers, gi)
		}
		if gi < len(moeOwned) && moeOwned[gi] > 0 {
			modelMB += int(math.Ceil(float64(shexpTotalMB) * float64(moeOwned[gi]) / float64(moeLayers)))
		}
		if gi == outputDev {
			modelMB += outputMB
		}
		kvShareMB := ownedShareMB(kvTotalMB, owned, model.NumLayers, gi)
		bufMB := usedMB - baselineMB - overheadMB - modelMB - kvShareMB
		if bufMB > 0 {
			computeByGPU[g.Index] = bufMB
		}
	}
	if len(computeByGPU) == 0 {
		return nil
	}
	return computeByGPU
}

func RunPostLaunchModelProbeVRAMDelta(
	cacheDir string, model *ModelProfile, strategy *Strategy,
	backendTag string, gpus []detect.GPU, baselineVRAMByGPU map[int]int,
) bool {
	if model == nil || strategy == nil || len(gpus) == 0 || len(baselineVRAMByGPU) == 0 {
		return false
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	// This is strictly a fallback for backends that expose neither fit-params nor
	// compute-buffer log rows. Never replace an authoritative probe already
	// recorded by preflight for this exact runtime signature.
	existing := loadProbeCache(cacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, backendTag, gpus, strategy.Parallel)
	if existing != nil && len(existing.ComputeBufByGPU) > 0 {
		return false
	}

	usedVRAMByGPU := map[int]int{}
	for _, g := range gpus {
		usedMB := QueryVRAMUsed(g.Index)
		if usedMB > 0 {
			usedVRAMByGPU[g.Index] = usedMB
		}
	}
	computeByGPU := computeBuffersFromVRAMDelta(model, strategy, gpus, baselineVRAMByGPU, usedVRAMByGPU, SystemCUDAOverheadByGPU(cacheDir, gpus))
	if len(computeByGPU) == 0 {
		return false
	}

	// Preserve any runtime-growth history from a previous OOM so the probe
	// cache does not silently erase it (audit cross-check #3).
	var mergedGrowth map[int]int
	var mergedEstimated map[int]bool
	if existing != nil {
		mergedGrowth = existing.RuntimeGraphGrowthByGPU
		mergedEstimated = existing.RuntimeGraphGrowthEstimatedByGPU
	}

	if err := writeProbeCacheForModel(cacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, backendTag, gpus, strategy.Parallel, computeByGPU, mergedGrowth, mergedEstimated, 0); err == nil {
		indices := make([]int, 0, len(computeByGPU))
		for idx := range computeByGPU {
			indices = append(indices, idx)
		}
		sort.Ints(indices)
		parts := make([]string, 0, len(indices))
		for _, idx := range indices {
			parts = append(parts, fmt.Sprintf("CUDA%d=%dMB", idx, computeByGPU[idx]))
		}
		fmt.Fprintf(os.Stderr, "  VRAM probe: compute_buf %s\n", strings.Join(parts, ", "))
		return true
	}
	return false
}

func RunPostLaunchModelProbe(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, serverLog string) bool {
	if model == nil || ctxSize <= 0 || ubatch <= 0 || serverLog == "" {
		return false
	}
	computeByGPU := ParseComputeBuffersByGPU(serverLog)
	_, kvPerLayerMB := ParseLogForProbe(serverLog)
	if len(computeByGPU) == 0 && kvPerLayerMB <= 0 {
		return false
	}
	if err := writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, computeByGPU, nil, nil, kvPerLayerMB); err == nil {
		if len(computeByGPU) == 0 {
			return true
		}
		parts := make([]string, 0, len(computeByGPU))
		indices := make([]int, 0, len(computeByGPU))
		for idx := range computeByGPU {
			indices = append(indices, idx)
		}
		sort.Ints(indices)
		for _, idx := range indices {
			parts = append(parts, fmt.Sprintf("CUDA%d=%dMB", idx, computeByGPU[idx]))
		}
		fmt.Fprintf(os.Stderr, "  Model probe written: compute_buf %s\n", strings.Join(parts, ", "))
		return true
	}
	return false
}

// writeMeasuredKVRate records a measured KV bytes-per-token for model+kvType,
// merging with any rates already cached for other kvTypes.
func writeMeasuredKVRate(cacheDir string, model *ModelProfile, kvType string, bytesPerTok float64, note string) {
	kvType = strings.ToLower(kvType)
	path := kvCachePath(cacheDir, model)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return
	}
	release, err := acquirePlacementLock(path+".lock", 5*time.Second)
	if err != nil {
		return
	}
	defer release()
	rates := loadMeasuredKVRates(cacheDir, model)
	if rates == nil {
		rates = map[string]float64{}
	}
	for k, v := range model.MeasuredKVBytesPerTok {
		if v > 0 && rates[k] <= 0 {
			rates[k] = v
		}
	}
	rates[kvType] = bytesPerTok
	model.MeasuredKVBytesPerTok = rates

	var b strings.Builder
	fmt.Fprintf(&b, "# Measured KV cache for %s (%s)\n", model.Basename, note)
	for k, v := range rates {
		fmt.Fprintf(&b, "KV_BYTES_PER_TOK_%s=%.4f\n", k, v)
	}
	for k, g := range model.MeasuredKVGeometry {
		if !g.Measured() {
			continue
		}
		fmt.Fprintf(&b, "KV_GEOMETRY_%s=%d,%d,%d,%.4f\n", k, g.FullLayers, g.SWALayers, g.SWACells, g.BytesPerCellPerLayer)
	}
	if err := atomicWriteFile(path, []byte(b.String()), 0o644); err == nil {
		fmt.Fprintf(os.Stderr, "  KV probe: %s = %.0f bytes/token (%s)\n", kvType, bytesPerTok, note)
	}
}

// parseMemoryBreakdownTable parses llama.cpp's common_memory_breakdown_print
// table from a server log. That table is printed on every startup (and on
// shutdown), so unlike a live-PID probe it is available even when the server
// crashed. Each GPU row carries the per-device ground truth:
//
//	| - CUDA1 (RTX 3060) | 11909 = 5749 + (     0 =      0 +       0 +       0) +        6160 |
//
// total/free are the addressable device totals (nvidia-smi minus the BAR
// reserve); self = model + context + compute is llama.cpp's own attribution of
// what it placed on that device; unaccounted = total - free - self is the
// allocator peak llama.cpp could not attribute to model/context/compute buffers
// — the CUDA context overhead ggrun wants to budget. Returns per-physical-GPU
// unaccounted MiB, keyed by the CUDA index.
func parseMemoryBreakdownTable(log string) map[int]int {
	out := map[int]int{}
	lines := strings.Split(log, "\n")
	for _, line := range lines {
		// Row shape (after the "common_memory_breakdown_print:" prefix):
		//   |  - CUDA1 (RTX 3060) | 11909 = 5749 + (  0 = 0 + 0 + 0) +  6160 |
		idx := strings.Index(line, "CUDA")
		if idx < 0 {
			continue
		}
		// Extract the trailing device index right after "CUDA".
		rest := line[idx+4:]
		sp := strings.IndexByte(rest, ' ')
		if sp <= 0 {
			continue
		}
		gpuIdx, err := strconv.Atoi(rest[:sp])
		if err != nil {
			continue
		}
		// Find the "| total = free + ( self = model + context + compute ) + unaccounted |"
		// and read the LAST number before the closing pipe — the unaccounted column.
		// Row format from llama.cpp (template_gpu):
		//   "%s = %s + (%s = %s + %s + %s) + %s |"
		// NOTE: guard pipe BEFORE slicing line[:pipe]; a "CUDA" mention in an
		// unrelated line (model buffer size, device enumeration) has no pipe and
		// LastIndex returns -1, which would panic the slice.
		pipe := strings.LastIndex(line, "|")
		if pipe <= 0 {
			continue
		}
		open := strings.LastIndex(line[:pipe], "(")
		if open < 0 || open > pipe {
			continue
		}
		tail := line[open+1 : pipe]
		// tail is " 0 = 0 + 0 + 0 ) + 6160". The last whitespace-delimited number is unaccounted.
		fields := strings.Fields(tail)
		if len(fields) == 0 {
			continue
		}
		last := fields[len(fields)-1]
		// Strip any trailing punctuation (e.g. the "|" or "+" if spacing differs).
		// Trim from the RIGHT only: TrimFunc from both ends would also strip a
		// leading "-", silently converting a negative unaccounted into a positive.
		v, err := strconv.Atoi(strings.TrimRightFunc(last, func(r rune) bool { return r < '0' || r > '9' }))
		if err != nil || v < 0 {
			continue
		}
		out[gpuIdx] = v
	}
	return out
}

// companionVRAMByGPU is the on-device VRAM ggrun itself reserved for helper
// companions (Auto reviewer, worker), keyed by physical GPU index. The
// breakdown table's unaccounted column includes this VRAM (a separate process
// llama.cpp does not attribute), so the probe nets it out before writing the
// system cache — otherwise the companion latches as permanent CUDA overhead on
// its card (the 2916 MiB bug from 2026-08-02, and the 6160 MiB GPU1 column in
// the current logs).
func RunPostLaunchProbe(cacheDir string, gpus []detect.GPU, serverLog string, serverPID int, companionVRAMByGPU map[int]int) {
	if len(gpus) == 0 || serverLog == "" {
		return
	}
	// Measure per device, and keep measuring devices that have no entry yet.
	//
	// This used to return as soon as ANY device had a value, which made the
	// probe all-or-nothing on a file that is never refreshed. On this project a
	// single entry written 2026-08-02T19:25:30Z (CUDA1, and wrong — it was the
	// companion) permanently prevented CUDA0 and CUDA2 from ever being measured,
	// so both cards carried a 0 MiB CUDA-overhead reserve forever. Their real
	// cost is visible in llama.cpp's own shutdown breakdown: 944 MiB on CUDA0
	// and 450 MiB on CUDA2. Every plan was therefore optimistic by roughly half
	// a gigabyte per card, which is how a placement that passes the no-alloc fit
	// oracle still dies on the real load: attempt 1 of the 23:11 launch packed
	// CUDA2 to within 97 MiB of free and needed 450.
	existing := map[int]int{}
	existingHostOverhead := 0
	if sp := loadSystemProbe(cacheDir, gpus); sp != nil {
		for idx, v := range sp.CUDAOverheadByGPU {
			if v > 0 {
				existing[idx] = v
			}
		}
		existingHostOverhead = sp.HostOverheadMB
	}
	measuredEvery := len(existing) > 0
	for _, gpu := range gpus {
		if _, ok := existing[gpu.Index]; !ok {
			measuredEvery = false
			break
		}
	}
	// The host term is measured on its own schedule: a rig whose cards were all
	// measured by an earlier launch would otherwise return here and never learn
	// its host overhead at all.
	if measuredEvery && existingHostOverhead > 0 {
		return
	}

	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}

	overheadByGPU := map[int]int{}
	// When every card already has a value and only the host term is missing,
	// take the cards as they are. Re-measuring them here would overwrite good
	// numbers with ones taken under whatever workload happens to be running.
	//
	// A whole-device delta charges everything it cannot attribute to "CUDA
	// overhead", and what it cannot attribute depends on the launch: a
	// single-GPU Qwen3.6-27B at -b 8192 -ub 1024 produced CUDA0=3850 MiB where
	// DeepSeek-V4 at -b 2048 -ub 512 had measured 1327, and applying that to V4
	// cost CUDA0 its only expert layer (9 GPU layers -> 8). The device values
	// are only replaced when a device has none.
	if measuredEvery {
		for idx, v := range existing {
			overheadByGPU[idx] = v
		}
	}
	// Primary source: llama.cpp's own shutdown/startup breakdown table. Its
	// unaccounted column (total - free - self) is the allocator peak the server
	// holds that is not model/context/compute buffers — the CUDA context overhead
	// we want. Unlike the PID probe it does not need the server alive, so a
	// crashed launch (serverPID dead) still yields measured data instead of
	// leaving the term budgeted 0 forever. It is also already net of any
	// companion's VRAM (the companion sits in "free"), so it cannot re-latch the
	// 2916 MiB companion-as-overhead bug.
	tableOverhead := parseMemoryBreakdownTable(serverLog)
	if len(tableOverhead) > 0 {
		for _, gpu := range gpus {
			if _, done := overheadByGPU[gpu.Index]; done {
				continue // already measured by an earlier launch; do not overwrite
			}
			v, ok := tableOverhead[gpu.Index]
			if !ok || v <= 0 {
				continue
			}
			// Net out ggrun's own companion VRAM on this card. The breakdown's
			// unaccounted column is total - free - self; the companion occupies
			// "free" from the server's perspective but is not in "self", so it
			// falls entirely into unaccounted. Charging it as permanent CUDA
			// overhead would re-create the 2916 MiB latch (and turn GPU1's
			// 6160 MiB column into a permanent 6160 MiB reserve).
			if compMB := companionVRAMByGPU[gpu.Index]; compMB > 0 {
				v -= compMB
			}
			// Sanity: after netting the companion, the real CUDA context is a
			// modest amount (a few hundred MiB). Reject anything that still
			// looks like a latched foreign process (e.g. a second companion or
			// another workload llama.cpp could not attribute).
			if v > 0 && v < gpu.VRAMTotalMB/systemProbeOutlierRatio {
				overheadByGPU[gpu.Index] = v
			}
		}
	}
	// Secondary source: a live whole-device reading via the server's PID. Only
	// needed when the log had no breakdown table (older backends). Prefer what
	// the server itself holds on this device, and skip a device the server does
	// not use rather than guessing. A whole-device reading includes every other
	// process, and ggrun deliberately seats the Auto reviewer on the least
	// valuable card -- so the card most hurt by an inflated overhead is exactly
	// the one that gets it. Recorded 2026-08-02T19:25:30Z on this project:
	// SYS_CUDA_OVERHEAD_MB_CUDA1=2916, the NanoBeige worker booked as permanent
	// system overhead on the RTX 3060. Because the probe is written once and
	// never re-measured, that latched: the card was charged 2916 MiB on every
	// later plan, could no longer afford a 2935 MiB expert layer, held no server
	// tensors as a result, and so re-measured the same way. A card the server
	// does not use is skipped, not guessed at.
	for _, gpu := range gpus {
		if _, ok := overheadByGPU[gpu.Index]; ok {
			continue // breakdown table already provided this device
		}
		usedMB := QueryVRAMUsedByPIDOnGPU(serverPID, gpu.Index)
		if usedMB <= 0 {
			continue
		}
		modelBufMB, kvBufMB, computeBufMB := parseBuffersFromLog(serverLog, gpu.Index)
		accounted := modelBufMB + kvBufMB + computeBufMB
		if accounted <= 0 {
			continue
		}
		cudaOverhead := usedMB - accounted
		if cudaOverhead <= 0 || cudaOverhead >= usedMB {
			continue
		}
		overheadByGPU[gpu.Index] = cudaOverhead
	}
	// Tertiary source: a whole-device reading minus what the backend says it
	// holds. This exists because the two sources above can both be silent on a
	// perfectly healthy launch, and then the term stays 0 forever.
	//
	// The breakdown table is not printed by every build -- the mainline
	// llama-server used here emits it zero times, so that source never fires.
	// The per-PID query answers 0 whenever the recorded PID is not the process
	// holding the memory, which is what happens when the launch goes through the
	// cgroup scope wrapper: the wrapper is the child ggrun waited on, the server
	// is its child. Between them, a rig can serve for hours and still have no
	// measurement.
	//
	// Measured here 2026-08-05 while the server was live: CUDA0 held 23922 MiB
	// against 23151 MiB of reported buffers, CUDA2 11473 against 11036 --
	// roughly 770 and 436 MiB of CUDA context that no plan reserved. The planner
	// had packed CUDA0 to 97.4% of the card and left 642 MiB for allocations it
	// could not see; the next context checkpoint killed it.
	//
	// A whole-device reading counts every process, so ggrun's own companion is
	// subtracted (a separate, already-budgeted tenant) and the result is held to
	// the same outlier ceiling as the other sources. Anything larger is another
	// workload, not a CUDA context, and must not latch.
	for _, gpu := range gpus {
		if _, ok := overheadByGPU[gpu.Index]; ok {
			continue // an earlier source already answered for this device
		}
		liveUsedMB := QueryVRAMUsed(gpu.Index)
		if liveUsedMB <= 0 {
			continue
		}
		modelBufMB, kvBufMB, computeBufMB := parseBuffersFromLog(serverLog, gpu.Index)
		accounted := modelBufMB + kvBufMB + computeBufMB
		if accounted <= 0 {
			continue // the server does not use this card; skip, never guess
		}
		if delta, ok := wholeDeviceOverheadMB(liveUsedMB, accounted, companionVRAMByGPU[gpu.Index], gpu.VRAMTotalMB); ok {
			overheadByGPU[gpu.Index] = delta
		}
	}
	// Carry forward devices measured by earlier launches: this launch may not
	// have occupied every card, and a device the server does not use is skipped
	// rather than re-measured, so its prior value is the best evidence there is.
	for idx, v := range existing {
		if _, fresh := overheadByGPU[idx]; !fresh {
			overheadByGPU[idx] = v
		}
	}

	// Host side, same shape as the whole-device source above: what the memory
	// scope really holds, minus the host buffers the backend declared.
	hostOverhead := existingHostOverhead
	if live := queryHostNonReclaimableMB(serverPID); live > 0 {
		if accounted := parseHostBuffersFromLog(serverLog); accounted > 0 {
			if v, ok := hostOverheadMB(live, accounted); ok {
				hostOverhead = v
			}
		}
	}

	if len(overheadByGPU) == 0 && hostOverhead <= 0 {
		return
	}

	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return
	}
	gpuSig := gpuSignatureHash(gpus)
	path := filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuSig))
	var b strings.Builder
	fmt.Fprintf(&b, "# System probe (post-launch per-device measurement)\n")
	fmt.Fprintf(&b, "SYS_PROBE_SCHEMA=%d\n", systemProbeSchema)
	fmt.Fprintf(&b, "# Generated: %s\n", time.Now().Format(time.RFC3339))
	indices := make([]int, 0, len(overheadByGPU))
	for idx := range overheadByGPU {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	legacyMax := 0
	parts := make([]string, 0, len(indices))
	for _, idx := range indices {
		v := overheadByGPU[idx]
		if v > legacyMax {
			legacyMax = v
		}
		fmt.Fprintf(&b, "SYS_CUDA_OVERHEAD_MB_CUDA%d=%d\n", idx, v)
		parts = append(parts, fmt.Sprintf("CUDA%d=%dMB", idx, v))
	}
	// Compatibility for older readers. This is still measured data, not a margin.
	fmt.Fprintf(&b, "SYS_CUDA_OVERHEAD_MB=%d\n", legacyMax)
	if hostOverhead > 0 {
		fmt.Fprintf(&b, "SYS_HOST_OVERHEAD_MB=%d\n", hostOverhead)
		parts = append(parts, fmt.Sprintf("host=%dMB", hostOverhead))
	}
	if err := atomicWriteFile(path, []byte(b.String()), 0o644); err == nil {
		fmt.Fprintf(os.Stderr, "  System probe written: cuda_overhead %s\n", strings.Join(parts, ", "))
	}
}

// parseHostBuffersFromLog sums the host-side buffers the backend reported:
// CUDA_Host (pinned or plain host allocations backing CPU experts) and CPU
// (host KV when the cache is not offloaded). Maxima, not sums, because the log
// accumulates across launch attempts within one ggrun run -- the first attempt
// here reported 86136 MiB before --swa-full was withdrawn and the surviving one
// 91448 MiB, and adding them would invent 84 GiB that never existed.
func parseHostBuffersFromLog(log string) int {
	var maxModel, maxCompute, maxOutput, maxKV float64
	for _, line := range strings.Split(log, "\n") {
		if !strings.Contains(line, "CUDA_Host") && !strings.Contains(line, "CPU ") {
			continue
		}
		if !strings.Contains(line, "buffer size =") {
			continue
		}
		v := parseMiB(line)
		if v <= 0 {
			continue
		}
		switch {
		case strings.Contains(line, "KV buffer size ="):
			if v > maxKV {
				maxKV = v
			}
		case strings.Contains(line, "compute buffer size ="):
			if v > maxCompute {
				maxCompute = v
			}
		case strings.Contains(line, "output buffer size ="):
			if v > maxOutput {
				maxOutput = v
			}
		default:
			if v > maxModel {
				maxModel = v
			}
		}
	}
	return int(maxModel + maxCompute + maxOutput + maxKV + 0.5)
}

// queryHostNonReclaimableMB reports the non-reclaimable host memory held by the
// cgroup the backend runs in. ggrun launches the backend inside its own memory
// scope, so the cgroup is the exact boundary: it counts the server and nothing
// else, and it does not care that the recorded PID is the scope wrapper rather
// than the server itself -- the same mismatch that makes the per-PID VRAM query
// answer 0. Returns 0 when the process is gone or the cgroup is unreadable.
func queryHostNonReclaimableMB(pid int) int {
	if pid <= 0 {
		return 0
	}
	cgData, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return 0
	}
	cgPath := ""
	for _, line := range strings.Split(string(cgData), "\n") {
		parts := strings.SplitN(strings.TrimSpace(line), ":", 3)
		if len(parts) == 3 && parts[0] == "0" {
			cgPath = parts[2]
			break
		}
	}
	if cgPath == "" {
		return 0
	}
	statData, err := os.ReadFile(filepath.Join("/sys/fs/cgroup", cgPath, "memory.stat"))
	if err != nil {
		return 0
	}
	// anon: the backend's own allocations. shmem: how the CUDA host buffers for
	// CPU experts land under --no-mmap. slab: kernel structures charged to it.
	// "file" is deliberately excluded -- that is page cache for the GGUF, which
	// the kernel reclaims under pressure and which no plan should reserve.
	var totalBytes int64
	for _, line := range strings.Split(string(statData), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		switch fields[0] {
		case "anon", "shmem", "slab":
			if v, convErr := strconv.ParseInt(fields[1], 10, 64); convErr == nil && v > 0 {
				totalBytes += v
			}
		}
	}
	return int(totalBytes / (1024 * 1024))
}

// QueryVRAMUsed returns current nvidia-smi memory.used for a given GPU index.
func QueryVRAMUsed(gpuIndex int) int {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.used", "--format=csv,noheader,nounits",
		"-i", fmt.Sprintf("%d", gpuIndex),
	).Output()
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(string(out))
	v, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return v
}

// QueryVRAMUsedByPIDOnGPU returns the VRAM one process holds on one device.
//
// It exists because a whole-device reading cannot separate the model from
// anything else ggrun launched beside it. The system CUDA overhead probe
// subtracted the server's own logged buffers from the card's total usage, so on
// a card that also hosted the Auto reviewer the reviewer's memory landed in
// "system overhead" -- recorded as 2299 MiB on a 12 GB card against 397 and 255
// on the other two, and then charged on every later placement on top of the
// reviewer's own reservation. That cost the smallest GPU a full expert layer.
//
// Returns 0 when nvidia-smi is unavailable or the process holds nothing there,
// which leaves the caller on its previous behaviour.
func QueryVRAMUsedByPIDOnGPU(pid, gpuIndex int) int {
	if pid <= 0 || gpuIndex < 0 {
		return 0
	}
	uuid := gpuUUIDForIndex(gpuIndex)
	if uuid == "" {
		return 0
	}
	out, err := exec.Command("nvidia-smi",
		"--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0
	}
	total := 0
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Split(line, ",")
		if len(f) != 3 {
			continue
		}
		gotPID, err1 := strconv.Atoi(strings.TrimSpace(f[0]))
		mb, err2 := strconv.Atoi(strings.TrimSpace(f[2]))
		if err1 != nil || err2 != nil || gotPID != pid || mb <= 0 {
			continue
		}
		if strings.TrimSpace(f[1]) != uuid {
			continue
		}
		total += mb
	}
	return total
}

func gpuUUIDForIndex(gpuIndex int) string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader").Output()
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Split(line, ",")
		if len(f) != 2 {
			continue
		}
		if idx, err := strconv.Atoi(strings.TrimSpace(f[0])); err == nil && idx == gpuIndex {
			return strings.TrimSpace(f[1])
		}
	}
	return ""
}

// QueryVRAMUsedByPID returns the VRAM a single process holds, summed across
// devices. nvidia-smi reports one row per (process, device), so a process with
// contexts on several cards appears more than once.
//
// This exists so a companion's reservation can be a measurement rather than a
// constant: memory.used for a whole GPU cannot separate the companion from the
// model it sits beside.
func QueryVRAMUsedByPID(pid int) int {
	if pid <= 0 {
		return 0
	}
	out, err := exec.Command("nvidia-smi",
		"--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0
	}
	total := 0
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Split(line, ",")
		if len(fields) != 2 {
			continue
		}
		got, err1 := strconv.Atoi(strings.TrimSpace(fields[0]))
		mb, err2 := strconv.Atoi(strings.TrimSpace(fields[1]))
		if err1 != nil || err2 != nil || got != pid || mb <= 0 {
			continue
		}
		total += mb
	}
	return total
}

// parseBuffersFromLog parses llama-server log for CUDA buffer sizes on a specific GPU.
// Returns modelBufMB, kvBufMB, computeBufMB.
// Handles both mainline ("CUDAN model buffer size = X MiB") and
// ik_llama ("CUDAN buffer size = X MiB") formats.
func parseBuffersFromLog(log string, gpuIndex int) (modelBufMB, kvBufMB, computeBufMB int) {
	cudaTag := fmt.Sprintf("CUDA%d", gpuIndex)

	var maxModelBuf, maxComputeBuf float64
	var totalKVBuf float64
	var kvCount int

	lines := strings.Split(log, "\n")
	for _, line := range lines {
		if !strings.Contains(line, cudaTag) {
			continue
		}

		// Model buffer: "CUDA0 model buffer size = X MiB" or "CUDA0 buffer size = X MiB"
		if strings.Contains(line, "buffer size =") && !strings.Contains(line, "KV") && !strings.Contains(line, "compute") {
			if v := parseMiB(line); v > maxModelBuf {
				maxModelBuf = v
			}
		}

		// KV buffer: "CUDA0 KV buffer size = X MiB"
		if strings.Contains(line, "KV buffer size =") {
			if v := parseMiB(line); v > 0 {
				totalKVBuf += v
				kvCount++
			}
		}

		// Compute buffer: "CUDA0 compute buffer size = X MiB"
		if strings.Contains(line, "compute buffer size =") {
			if v := parseMiB(line); v > maxComputeBuf {
				maxComputeBuf = v
			}
		}
	}

	modelBufMB = int(maxModelBuf + 0.5)
	computeBufMB = int(maxComputeBuf + 0.5)
	if totalKVBuf > 0 && kvCount > 0 {
		kvBufMB = int(totalKVBuf/float64(kvCount) + 0.5)
	}
	return
}

// parseMiB extracts a floating-point MiB value from a log line containing "X MiB".
func parseMiB(line string) float64 {
	idx := strings.LastIndex(line, "=")
	if idx < 0 {
		return 0
	}
	// Take the number between "=" and the FIRST "MiB" after it, so lines with
	// trailing detail (e.g. "KV self size = X MiB, K (f16): Y MiB, ...") parse the
	// aggregate X and not the per-component values.
	rest := line[idx+1:]
	mib := strings.Index(rest, "MiB")
	if mib < 0 {
		return 0
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(rest[:mib]), 64)
	if err != nil {
		return 0
	}
	return v
}

// loadProbeCache tries to load per-model/runtime probe data.
// Keys the probe cache file by an MD5 of the model + placement runtime signature.
func loadProbeCache(cacheDir string, model *ModelProfile, ctxSize int, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) *probeCache {
	path := probeCachePath(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, probeParallelKey(parallel))
	if path == "" {
		return nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	content := string(data)
	pc := &probeCache{
		ComputeBufByGPU:                  map[int]int{},
		RuntimeGraphGrowthByGPU:          map[int]int{},
		RuntimeGraphGrowthEstimatedByGPU: map[int]bool{},
		ContextByGPU:                     map[int]int{},
		ModelByGPU:                       map[int]int{},
		UnaccountedByGPU:                 map[int]int{},
	}
	schemaVersion := 1
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
		k := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(strings.Trim(parts[1], `"`))
		switch {
		case k == "PROBE_CACHE_SCHEMA":
			if v, err := strconv.Atoi(val); err == nil && v > 0 {
				schemaVersion = v
			}
		case k == "PROBED_COMPUTE_BUF_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.ComputeBufMB = v
			}
		case k == "PROBED_PROMPT_CACHE_BYTES_PER_TOKEN":
			if v, err := strconv.ParseFloat(val, 64); err == nil && v > 0 {
				pc.PromptCacheBytesPerToken = v
			}
		case k == "PROBED_PROMPT_CACHE_ENTRY_MB":
			if v, err := strconv.ParseFloat(val, 64); err == nil && v > 0 {
				pc.PromptCacheEntryMB = v
			}
		case k == "PROBED_CONTEXT_TOTAL_MB":
			if v, err := strconv.Atoi(val); err == nil && v > 0 {
				pc.ContextTotalMB = v
			}
		case k == "PROBED_CONTEXT_MB_HOST":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.ContextHostMB = v
			}
		case k == "PROBED_MODEL_MB_HOST":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.ModelHostMB = v
			}
		case k == "PROBED_UNACCOUNTED_MB_HOST":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.UnaccountedHostMB = v
			}
		case k == "PROBED_ALLOCATION_EVIDENCE":
			pc.AllocationEvidence = val
		case strings.HasPrefix(k, "PROBED_CONTEXT_MB_CUDA"):
			idxRaw := strings.TrimPrefix(k, "PROBED_CONTEXT_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				pc.ContextByGPU[idx] = v
			}
		case strings.HasPrefix(k, "PROBED_MODEL_MB_CUDA"):
			idxRaw := strings.TrimPrefix(k, "PROBED_MODEL_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				pc.ModelByGPU[idx] = v
			}
		case strings.HasPrefix(k, "PROBED_UNACCOUNTED_MB_CUDA"):
			idxRaw := strings.TrimPrefix(k, "PROBED_UNACCOUNTED_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				pc.UnaccountedByGPU[idx] = v
			}
		case strings.HasPrefix(k, "PROBED_COMPUTE_BUF_MB_CUDA"):
			idxRaw := strings.TrimPrefix(k, "PROBED_COMPUTE_BUF_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				pc.ComputeBufByGPU[idx] = v
			}
		case strings.HasPrefix(k, "PROBED_RUNTIME_GRAPH_GROWTH_ESTIMATED_CUDA"):
			idxRaw := strings.TrimPrefix(k, "PROBED_RUNTIME_GRAPH_GROWTH_ESTIMATED_CUDA")
			if idx, err := strconv.Atoi(idxRaw); err == nil && idx >= 0 {
				pc.RuntimeGraphGrowthEstimatedByGPU[idx] = val == "1" || strings.EqualFold(val, "true")
			}
		case strings.HasPrefix(k, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA"):
			idxRaw := strings.TrimPrefix(k, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				pc.RuntimeGraphGrowthByGPU[idx] = v
			}
		case k == "PROBED_KV_PER_LAYER_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.KVPerLayerMB = v
			}
		case k == "PROBED_FREE_VRAM":
			pc.FreeVRAMAtProbe = parsePlanFreeVRAM(val)
		}
	}
	// Reject the whole file, not just one field: a backend boxed into a
	// degenerate placement reserves a different graph, sizes a different
	// context, and reports different buffers. Every number in it is
	// conditional on how busy the machine was.
	//
	// Below schema 7 nothing recorded those conditions, so no such file can be
	// judged and all of them are discarded. From schema 7 on, the writer always
	// tries: an absent field then means the free-VRAM reading itself was
	// unavailable, which is a different thing from a busy machine and must not
	// cost the cache.
	if schemaVersion < probeCacheSchema {
		return nil
	}
	if len(pc.FreeVRAMAtProbe) > 0 && probeMeasuredUnderDuress(pc.FreeVRAMAtProbe, gpus) {
		return nil
	}
	if growthPredatesServingGate(schemaVersion) {
		// Nothing in the file says whether a growth figure was learned after the
		// model started serving or from an allocation that failed during load.
		// Keeping an unverifiable one is not the safe choice it looks like: it
		// withholds VRAM permanently and silently. Dropping a genuine one costs
		// a single re-learn, because a real runtime OOM aborts the backend and
		// ggrun's recovery records it again -- this time through the gate.
		pc.RuntimeGraphGrowthByGPU = map[int]int{}
		pc.RuntimeGraphGrowthEstimatedByGPU = map[int]bool{}
	}
	// A schema-1 repair used to live here: before schema 2 a startup
	// graph-reserve OOM was written once as the compute buffer and again as
	// "runtime growth" from the same cudaMalloc line, and summing both excluded
	// otherwise viable GPUs. The conditions gate above now discards every
	// pre-schema-7 file outright, so that repair became unreachable and was
	// removed rather than left to imply a guarantee it no longer provides.
	if pc.ComputeBufMB > 0 || len(pc.ComputeBufByGPU) > 0 || len(pc.RuntimeGraphGrowthByGPU) > 0 ||
		pc.KVPerLayerMB > 0 || pc.ContextTotalMB > 0 || pc.PromptCacheBytesPerToken > 0 || pc.PromptCacheEntryMB > 0 {
		return pc
	}
	return nil
}

func probeCachePath(cacheDir string, model *ModelProfile, ctxSize int, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) string {
	if model == nil {
		return ""
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun", "probes")
	}
	if kvPlacement == "" {
		kvPlacement = "auto"
	}
	if kvQuality == "" {
		kvQuality = "mid"
	}
	if backendTag == "" {
		backendTag = "llama"
	}
	// MD5 hash key over model/runtime/placement. Compute buffers differ with KV
	// placement, backend, GPU set, and parallel slots. Tensor split is derived
	// after probes are loaded, so it cannot honestly participate in this key.
	modelIdentity := SpecTargetIdentity(model)
	// Keep the historical trailing separator so serial (parallel key 0) cache
	// paths remain compatible with measurements from before slot isolation.
	key := fmt.Sprintf("probe:v%d:%s:%d:%d:%d:%d:%d:%d:%s:%s:%s:%s:%d:",
		placementProbeCacheVersion, modelIdentity, model.NumLayers, model.NumExperts,
		model.EmbeddingLength, model.FeedForwardLength,
		ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpuSignatureHash(gpus), parallel)
	hash := md5Hash12(key)
	return filepath.Join(cacheDir, hash+".probe")
}

// PlacementCachePathFor returns the keyed file that persists the validated MoE
// placement for this exact model + runtime + hardware. Keyed identically to the
// probe cache (kv placement, ctx, ubatch, backend, GPU set all part of the key)
// so kv=gpu vs kv=cpu — or two context sizes — never share a cache entry. Both
// the fit (load) and OOM-recovery / success (save) use this same path, so a
// placement that loads cleanly is remembered instead of re-predicted.
// Placement and measurement versions are deliberately separate. Changing the
// packing algorithm must invalidate saved .place files without throwing away
// valid compute/KV measurements that the new generic optimizer needs.
// Version 6 adds exact feature-scoped allocation evidence and a complete
// stable hardware signature. Older keys cannot prove either property.
const placementProbeCacheVersion = 7

// Bump whenever placement semantics can change emitted expert residency.
// Version 6 removes the architecture-specific split-owner exclusion and lets
// every MoE use the same capacity-driven whole-layer packing path.
// Version 5 had also invalidated v4 entries planned against an under-sized KV:
// sliding-window layers were priced at their window depth even under --swa-full,
// and a geometry measured at one KV type was not reused for another, so plans
// were validated against an allocation the backend would never make.
const placementPlanCacheVersion = 7

// swaFull belongs in the key because it changes the KV allocation without
// changing anything else the key already carries: on Laguna the same context
// costs 4x more with it, so a plan cached from a plain launch is not a plan for
// a --swa-full one, and reusing it hands the backend a layout that cannot fit.
func PlacementCachePathFor(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, tensorSplit string, swaFull bool) string {
	if model == nil {
		return ""
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	if kvPlacement == "" {
		kvPlacement = "auto"
	}
	if kvQuality == "" {
		kvQuality = "mid"
	}
	if backendTag == "" {
		backendTag = "llama"
	}
	key := fmt.Sprintf("place:v%d:%s:%d:%d:%d:%d:%d:%d:%s:%s:%s:%s:%d:%s:swafull=%t",
		placementPlanCacheVersion, SpecTargetIdentity(model), model.NumLayers, model.NumExperts,
		model.EmbeddingLength, model.FeedForwardLength,
		ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpuSignatureHash(gpus), parallel, tensorSplit, swaFull)
	return filepath.Join(cacheDir, md5Hash12(key)+".place")
}

// md5Hash12 computes first 12 chars of MD5 hash of input string.
func md5Hash12(input string) string {
	h := md5.New()
	h.Write([]byte(input))
	return fmt.Sprintf("%x", h.Sum(nil))[:12]
}

// internal types for probe loading
type systemProbe struct {
	CUDAOverheadMB    int
	CUDAOverheadByGPU map[int]int
	// HostOverheadMB is host RAM the backend holds that is not a buffer it
	// reported: allocator arenas, thread stacks, the CUDA host-side runtime.
	// Zero means it was never measured.
	HostOverheadMB int
}

// hostOverheadMB is the host analogue of wholeDeviceOverheadMB: what the
// backend's memory scope really holds, minus the host buffers the backend
// itself reported. plannedRAMRuntimeOverheadMB returns 0 under
// RequireMeasuredBuffers because the project takes no static margins, so
// without this the whole term was simply absent from every host plan.
//
// Measured on this host with DeepSeek-V4 serving at --n-cpu-moe 37:
//
//	non-reclaimable (anon 1680 + shmem 91522 + slab 212) = 93414 MiB
//	backend-declared host buffers (91448.56 + 36.32 + 0.99) = 91486 MiB
//	                                              overhead =  1928 MiB
//
// llama-server's own VmRSS was 93878 MiB, agreeing to within half a percent.
//
// Only non-reclaimable bytes count. Page cache for the GGUF is file-backed and
// evictable -- it was 15.5 GiB here -- and charging the plan for it would
// invent a shortage. The outlier ceiling is a fraction of the live reading
// itself: real runtime overhead is a small remainder beside the weights, so
// anything approaching the same order means the buffers were not parsed and
// the difference is mostly model bytes, not overhead.
func hostOverheadMB(liveNonReclaimableMB, accountedHostMB int) (int, bool) {
	if liveNonReclaimableMB <= 0 || accountedHostMB <= 0 {
		return 0, false
	}
	delta := liveNonReclaimableMB - accountedHostMB
	if delta <= 0 || delta >= liveNonReclaimableMB/systemProbeOutlierRatio {
		return 0, false
	}
	return delta, true
}

type probeCache struct {
	ComputeBufMB    int
	ComputeBufByGPU map[int]int
	// ContextTotalMB is the backend-authoritative context allocation for this
	// exact runtime signature. Unlike the legacy model-wide bytes/token cache it
	// is never extrapolated to another context, backend build, slot count, KV
	// placement, workload, or --swa-full state.
	ContextTotalMB int
	ContextByGPU   map[int]int
	ContextHostMB  int
	// Model and unaccounted rows complete the fixed allocation ledger. They are
	// retained for convergence reports and candidate validation even where the
	// current packer can only consume the context total directly.
	ModelByGPU        map[int]int
	ModelHostMB       int
	UnaccountedByGPU  map[int]int
	UnaccountedHostMB int
	// AllocationEvidence is "oracle-planned", "guarded-allocated", or
	// "live-allocated". It makes confidence explicit instead of flattening a
	// dry plan and an observed allocator peak into the same boolean.
	AllocationEvidence string
	// RuntimeGraphGrowthByGPU is VRAM a real request needed beyond the
	// load-time graph reserve, keyed by GPU index.
	RuntimeGraphGrowthByGPU map[int]int
	// RuntimeGraphGrowthEstimatedByGPU marks entries that were guessed rather
	// than read from the backend.
	//
	// A CUDA VMM out-of-memory line carries no allocation size, so ggrun falls
	// back to a flat fraction of the card. Storing that indistinguishably from
	// a measured size meant a guess could never be corrected, and repeated
	// crashes compounded it: on a 24 GiB card two aborts accumulated 4914 MiB,
	// which is 3.6 expert layers withheld permanently. One of those aborts was
	// a malformed launch that proved nothing about capacity.
	RuntimeGraphGrowthEstimatedByGPU map[int]bool
	KVPerLayerMB                     int
	// PromptCacheBytesPerToken is what one cached prompt actually costs per
	// token of prefix, read from the backend rather than derived.
	//
	// A saved entry is target state + draft state + checkpoints, and no formula
	// over KV predicted it: KV for a whole 1M context estimated 9238 MiB while
	// one entry for a 126k-token prompt measured 6494. Sizing the budget from
	// the estimate left room for a single entry, so a four-slot server evicted
	// one conversation's prefix to admit the next and re-prefilled ~126k tokens
	// on the turn after -- 17.5% cache-read against 88.3% on a single-slot
	// server sharing the same box and client.
	PromptCacheBytesPerToken float64
	// PromptCacheEntryMB is what one saved conversation actually occupied,
	// read from the backend's own eviction and skip lines.
	//
	// This is the quantity the budget formula needs -- a cache holding fewer
	// than one entry per slot evicts a conversation's prefix to admit the next
	// -- and unlike the per-token cost it is printed at warning level, so it
	// arrives without raising verbosity. It only appears once the cache is
	// already under pressure, which is exactly when the budget is wrong.
	PromptCacheEntryMB float64
	// FreeVRAMAtProbe is the free VRAM each GPU showed when these numbers were
	// measured. Every value above is conditional on it: a backend boxed into a
	// degenerate placement by a busy card reserves a completely different graph
	// than the same backend on an idle one. See probeMeasuredUnderDuress.
	FreeVRAMAtProbe map[int]int
}

// probeMeasuredUnderDuress reports whether a probe's numbers were measured on a
// machine materially busier than it is now, which makes them stale-pessimistic
// rather than wrong-at-the-time.
//
// ggrun learns from every launch but had no notion of whether a launch was
// healthy, so a probe taken while the previous server was still releasing VRAM
// was cached and then believed. Measured here: launching before teardown
// finished recorded a CUDA2 compute buffer of 8641 MiB against a healthy
// 149 MiB. The next launch read that as fact --
//
//	12282 total - 8641 compute - 359 overhead = 3282 MiB room
//	one expert layer                          = 3292 MiB
//
// -- and gave the card zero expert layers, missing by 10 MiB. Three of 43
// layers landed on GPU, 107 GB of weights went to host RAM, and that plan was
// cached in turn. Each degraded launch taught the next one to be worse.
//
// A probe with no recorded conditions cannot be judged and is treated as
// suspect: the unjudgeable ones are exactly the population this exists to
// contain. The re-probe that follows costs one preflight.
// formatProbeFreeVRAM renders the current per-GPU free VRAM for the probe
// header, in the same "idx:MB" shape the placement cache uses.
func formatProbeFreeVRAM(gpus []detect.GPU) string {
	idxs := make([]int, 0, len(gpus))
	free := make(map[int]int, len(gpus))
	for _, g := range gpus {
		v := g.VRAMFreeMB()
		if v <= 0 {
			continue
		}
		idxs = append(idxs, g.Index)
		free[g.Index] = v
	}
	if len(idxs) == 0 {
		return ""
	}
	sort.Ints(idxs)
	tokens := make([]string, 0, len(idxs))
	for _, idx := range idxs {
		tokens = append(tokens, fmt.Sprintf("%d:%d", idx, free[idx]))
	}
	return strings.Join(tokens, " ")
}

func probeMeasuredUnderDuress(freeAtProbe map[int]int, gpus []detect.GPU) bool {
	if len(freeAtProbe) == 0 {
		return true
	}
	for _, g := range gpus {
		recorded, ok := freeAtProbe[g.Index]
		if !ok {
			return true
		}
		if g.VRAMFreeMB() > recorded+planFreeVRAMSlackMB {
			return true
		}
	}
	return false
}

// probeMeasurements carries optional measurements into the shared probe
// writer. AllocationSet distinguishes "this caller did not measure allocation"
// from an authoritative measurement whose per-device maps happen to be empty.
type probeMeasurements struct {
	BytesPerToken      float64
	EntryMB            float64
	AllocationSet      bool
	ContextTotalMB     int
	ContextByGPU       map[int]int
	ContextHostMB      int
	ModelByGPU         map[int]int
	ModelHostMB        int
	UnaccountedByGPU   map[int]int
	UnaccountedHostMB  int
	AllocationEvidence string
	ClearRuntimeGrowth bool
}

// probeCacheSchema 7 is the first version that records the conditions its
// numbers were measured under (PROBED_FREE_VRAM), so a reader can tell a
// healthy measurement from one taken while the machine was still busy. Files
// below it are discarded outright: see probeMeasuredUnderDuress.
//
// Schema 6 was the first whose runtime-graph-growth entries are
// known to have passed the post-serving window gate. Before it, the recorder
// filed whatever cudaMalloc size had failed, so a load-time KV or compute-buffer
// allocation became "growth" and was reserved on every later plan for that
// signature. Measured here: 5504 MiB on CUDA0, exactly that plan's
// `CUDA0 KV buffer size = 5504.00 MiB`, which cost four expert layers to the CPU
// and roughly half the decode rate -- and re-probing carried it forward rather
// than retiring it. See growthPredatesServingGate.
const probeCacheSchema = 7

// probeParallelKey preserves the legacy serial key (0) for normal --parallel 1
// launches while isolating multi-slot graph measurements such as Claude Code's
// --parallel 4. Graph allocation changes with slot count; sharing those probes
// was another form of compute-buffer double accounting across launch modes.
func probeParallelKey(parallel int) int {
	if parallel <= 1 {
		return 0
	}
	return parallel
}

// splitCompactKey returns a compact string for a tensor-split vector, suitable
// for cache-key inclusion so placements with different split ratios don't share
// cached compute-buffer measurements (audit cross-check #6).
func splitCompactKey(split []float64) string {
	if len(split) == 0 {
		return "0"
	}
	parts := make([]string, len(split))
	for i, v := range split {
		parts[i] = fmt.Sprintf("%.2f", v)
	}
	return strings.Join(parts, ",")
}

// WriteProbeCache writes a legacy probe cache. Prefer WriteProbeCacheForModel,
// which writes to the same runtime-signature key that loadProbeCache reads.
func WriteProbeCache(cacheDir, modelName string, computeBufMB, kvPerLayerMB int) error {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun", "probes")
	}
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return err
	}
	// Sanitize model name for filename
	safeName := strings.ReplaceAll(modelName, "/", "_")
	path := filepath.Join(cacheDir, safeName+".probe")
	content := fmt.Sprintf(
		"# Probe cache for %s\n"+
			"# Generated: %s\n"+
			"PROBED_COMPUTE_BUF_MB=%d\n"+
			"PROBED_KV_PER_LAYER_MB=%d\n",
		modelName, time.Now().Format(time.RFC3339), computeBufMB, kvPerLayerMB,
	)
	return atomicWriteFile(path, []byte(content), 0o644)
}

// WriteProbeCacheForModel writes measured compute-buffer and KV sizes to the
// per-model/runtime cache consumed by placement. computeByGPU is keyed by CUDA
// device index as emitted in the backend log.
func WriteProbeCacheForModel(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, computeByGPU map[int]int, kvPerLayerMB int) error {
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, 0, computeByGPU, nil, nil, kvPerLayerMB)
}

// measurements is variadic so the many focused writers stay untouched. A
// missing tail means preserve any prompt-cache and allocation evidence already
// stored for this exact key; it never means erase it.
func writeProbeCacheForModel(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, computeByGPU map[int]int, runtimeGrowthByGPU map[int]int, estimatedByGPU map[int]bool, kvPerLayerMB int, measurements ...probeMeasurements) error {
	parallelKey := probeParallelKey(parallel)
	var measured probeMeasurements
	if len(measurements) > 0 {
		measured = measurements[0]
	}
	path := probeCachePath(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallelKey)
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	release, err := acquirePlacementLock(path+".lock", 5*time.Second)
	if err != nil {
		return err
	}
	defer release()
	previous := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if previous != nil {
		if measured.BytesPerToken <= 0 {
			measured.BytesPerToken = previous.PromptCacheBytesPerToken
		}
		if measured.EntryMB <= 0 {
			measured.EntryMB = previous.PromptCacheEntryMB
		}
		if !measured.AllocationSet {
			measured.ContextTotalMB = previous.ContextTotalMB
			measured.ContextHostMB = previous.ContextHostMB
			measured.ModelHostMB = previous.ModelHostMB
			measured.UnaccountedHostMB = previous.UnaccountedHostMB
			measured.AllocationEvidence = previous.AllocationEvidence
			measured.ContextByGPU = copyProbeIntMap(previous.ContextByGPU)
			measured.ModelByGPU = copyProbeIntMap(previous.ModelByGPU)
			measured.UnaccountedByGPU = copyProbeIntMap(previous.UnaccountedByGPU)
		}
	}
	// One runtime signature can produce several tensor placements while the
	// preflight/recovery loop converges. Graph sizes are placement-dependent, so
	// retain the maximum ever observed per device instead of letting a later,
	// smaller graph erase the reserve that another valid placement required.
	mergedCompute := map[int]int{}
	for idx, v := range computeByGPU {
		mergedCompute[idx] = v
	}
	mergedGrowth := map[int]int{}
	mergedEstimated := map[int]bool{}
	if existing := previous; existing != nil {
		for idx, v := range existing.ComputeBufByGPU {
			if v > mergedCompute[idx] {
				mergedCompute[idx] = v
			}
		}
		if !measured.ClearRuntimeGrowth {
			for idx, v := range existing.RuntimeGraphGrowthByGPU {
				mergedGrowth[idx] = v
			}
			for idx, value := range existing.RuntimeGraphGrowthEstimatedByGPU {
				mergedEstimated[idx] = value
			}
		}
		if existing.KVPerLayerMB > kvPerLayerMB {
			kvPerLayerMB = existing.KVPerLayerMB
		}
	}
	if !measured.ClearRuntimeGrowth {
		// Merge under the file lock. Callers can have stale pre-lock snapshots;
		// measurements beat estimates even when smaller, and like-for-like
		// evidence retains the larger observed allocation.
		for idx, value := range runtimeGrowthByGPU {
			incomingEstimated := estimatedByGPU[idx]
			prior, exists := mergedGrowth[idx]
			priorEstimated := mergedEstimated[idx]
			switch {
			case !exists:
				mergedGrowth[idx], mergedEstimated[idx] = value, incomingEstimated
			case !incomingEstimated && priorEstimated:
				mergedGrowth[idx], mergedEstimated[idx] = value, false
			case incomingEstimated && !priorEstimated:
				// Never replace measured evidence with an estimate.
			case value > prior:
				mergedGrowth[idx], mergedEstimated[idx] = value, incomingEstimated
			}
		}
	}
	maxCompute := 0
	for _, v := range mergedCompute {
		if v > maxCompute {
			maxCompute = v
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# Probe cache for %s\n", filepath.Base(model.Path))
	fmt.Fprintf(&b, "# Generated: %s\n", time.Now().Format(time.RFC3339))
	fmt.Fprintf(&b, "# ctx=%d ubatch=%d kv_quality=%s kv_placement=%s backend=%s gpu_sig=%s parallel=%d\n", ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpuSignatureHash(gpus), parallelKey)
	fmt.Fprintf(&b, "PROBE_CACHE_SCHEMA=%d\n", probeCacheSchema)
	// The conditions every number below was measured under. Without them a
	// reader cannot tell a healthy measurement from one taken while the previous
	// server was still releasing the cards.
	if freeParts := formatProbeFreeVRAM(gpus); freeParts != "" {
		fmt.Fprintf(&b, "PROBED_FREE_VRAM=\"%s\"\n", freeParts)
	}
	if maxCompute > 0 {
		fmt.Fprintf(&b, "PROBED_COMPUTE_BUF_MB=%d\n", maxCompute)
	}
	// Outside the compute-buffer branch on purpose: these were nested inside it,
	// so a prompt-cache measurement was discarded on every write that had no
	// compute buffer to record alongside it.
	if measured.BytesPerToken > 0 {
		fmt.Fprintf(&b, "PROBED_PROMPT_CACHE_BYTES_PER_TOKEN=%.3f\n", measured.BytesPerToken)
	}
	if measured.EntryMB > 0 {
		fmt.Fprintf(&b, "PROBED_PROMPT_CACHE_ENTRY_MB=%.1f\n", measured.EntryMB)
	}
	if measured.ContextTotalMB > 0 {
		fmt.Fprintf(&b, "PROBED_CONTEXT_TOTAL_MB=%d\n", measured.ContextTotalMB)
		if measured.AllocationEvidence != "" {
			fmt.Fprintf(&b, "PROBED_ALLOCATION_EVIDENCE=%s\n", measured.AllocationEvidence)
		}
		fmt.Fprintf(&b, "PROBED_CONTEXT_MB_HOST=%d\n", max(measured.ContextHostMB, 0))
		fmt.Fprintf(&b, "PROBED_MODEL_MB_HOST=%d\n", max(measured.ModelHostMB, 0))
		fmt.Fprintf(&b, "PROBED_UNACCOUNTED_MB_HOST=%d\n", max(measured.UnaccountedHostMB, 0))
		writeProbeIntMap(&b, "PROBED_CONTEXT_MB_CUDA", measured.ContextByGPU)
		writeProbeIntMap(&b, "PROBED_MODEL_MB_CUDA", measured.ModelByGPU)
		writeProbeIntMap(&b, "PROBED_UNACCOUNTED_MB_CUDA", measured.UnaccountedByGPU)
	}
	indices := make([]int, 0, len(mergedCompute))
	for idx := range mergedCompute {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	for _, idx := range indices {
		if mergedCompute[idx] > 0 {
			fmt.Fprintf(&b, "PROBED_COMPUTE_BUF_MB_CUDA%d=%d\n", idx, mergedCompute[idx])
		}
	}
	growthIndices := make([]int, 0, len(mergedGrowth))
	for idx := range mergedGrowth {
		growthIndices = append(growthIndices, idx)
	}
	sort.Ints(growthIndices)
	for _, idx := range growthIndices {
		if mergedGrowth[idx] > 0 {
			fmt.Fprintf(&b, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA%d=%d\n", idx, mergedGrowth[idx])
			if mergedEstimated[idx] {
				fmt.Fprintf(&b, "PROBED_RUNTIME_GRAPH_GROWTH_ESTIMATED_CUDA%d=1\n", idx)
			}
		}
	}
	if kvPerLayerMB > 0 {
		fmt.Fprintf(&b, "PROBED_KV_PER_LAYER_MB=%d\n", kvPerLayerMB)
	}
	return atomicWriteFile(path, []byte(b.String()), 0o644)
}

func copyProbeIntMap(in map[int]int) map[int]int {
	if in == nil {
		return nil
	}
	out := make(map[int]int, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func writeProbeIntMap(b *strings.Builder, prefix string, values map[int]int) {
	indices := make([]int, 0, len(values))
	for index := range values {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	for _, index := range indices {
		if values[index] >= 0 {
			fmt.Fprintf(b, "%s%d=%d\n", prefix, index, values[index])
		}
	}
}

var memoryBreakdownComputePattern = regexp.MustCompile(`CUDA([0-9]+).*?\(\s*-?[0-9]+\s*=\s*-?[0-9]+\s*\+\s*-?[0-9]+\s*\+\s*([0-9]+)\s*\)`)

// ParseLogForProbe extracts compute_buf and kv_per_layer from server log output.
// Looks for lines like: "CUDA0 compute buffer size = 1410.12 MiB" and fit
// breakdown rows like: "CUDA0 ... ( self = model + context + compute )".
func ParseLogForProbe(logData string) (computeBufMB, kvPerLayerMB int) {
	computeByGPU := ParseComputeBuffersByGPU(logData)
	for _, v := range computeByGPU {
		if v > computeBufMB {
			computeBufMB = v
		}
	}

	// Sum KV buffer sizes across CUDA devices
	var totalKVBuf float64
	var kvCount int
	lines := strings.Split(logData, "\n")
	for _, line := range lines {
		if idx := strings.Index(line, "KV buffer size ="); idx >= 0 {
			rest := line[idx+len("KV buffer size ="):]
			rest = strings.TrimSpace(rest)
			rest = strings.TrimSuffix(rest, " MiB")
			if v, err := strconv.ParseFloat(rest, 64); err == nil {
				totalKVBuf += v
				kvCount++
			}
		}
	}

	if totalKVBuf > 0 && kvCount > 0 {
		// Approximate per-device KV share by kvCount (devices holding KV)
		kvPerLayerMB = int(totalKVBuf/float64(kvCount) + 0.5)
	}

	return
}

// ParseComputeBuffersByGPU extracts measured/planned compute-buffer MiB by CUDA
// index. It accepts both final llama.cpp buffer lines and the memory breakdown
// table printed during fit.
func ParseComputeBuffersByGPU(logData string) map[int]int {
	out := map[int]int{}
	for _, line := range strings.Split(logData, "\n") {
		if idx := strings.Index(line, "compute buffer size ="); idx >= 0 {
			cudaIdx := cudaIndexFromLine(line)
			if cudaIdx < 0 {
				continue
			}
			rest := line[idx+len("compute buffer size ="):]
			rest = strings.TrimSpace(rest)
			rest = strings.TrimSuffix(rest, " MiB")
			if v, err := strconv.ParseFloat(rest, 64); err == nil && v > 0 {
				mb := int(v + 0.5)
				if mb > out[cudaIdx] {
					out[cudaIdx] = mb
				}
			}
			continue
		}
		if m := memoryBreakdownComputePattern.FindStringSubmatch(line); m != nil {
			cudaIdx, idxErr := strconv.Atoi(m[1])
			mb, mbErr := strconv.Atoi(m[2])
			if idxErr == nil && mbErr == nil && mb > out[cudaIdx] {
				out[cudaIdx] = mb
			}
		}
	}
	return out
}

func cudaIndexFromLine(line string) int {
	idx := strings.Index(line, "CUDA")
	if idx < 0 {
		return -1
	}
	start := idx + len("CUDA")
	end := start
	for end < len(line) && line[end] >= '0' && line[end] <= '9' {
		end++
	}
	if end == start {
		return -1
	}
	n, err := strconv.Atoi(line[start:end])
	if err != nil {
		return -1
	}
	return n
}

// wholeDeviceOverheadMB derives per-device CUDA context overhead from a live
// whole-device VRAM reading and what the backend reported it holds.
//
// liveUsedMB counts every process on the card, so any companion ggrun seated
// there is subtracted: it is a separate tenant with its own budget line, and
// charging it here is how a 12 GB card once latched a 2916 MiB "overhead" that
// it could then never afford to contradict.
//
// The result is rejected unless it is positive and below the same outlier
// ceiling the other probe sources use. A larger figure is another workload, not
// a CUDA context, and a wrong value written here is not one bad launch -- it
// taxes every future plan on this hardware signature.
func wholeDeviceOverheadMB(liveUsedMB, accountedMB, companionMB, vramTotalMB int) (int, bool) {
	if liveUsedMB <= 0 || accountedMB <= 0 || vramTotalMB <= 0 {
		return 0, false
	}
	delta := liveUsedMB - accountedMB - companionMB
	if delta <= 0 || delta >= vramTotalMB/systemProbeOutlierRatio {
		return 0, false
	}
	return delta, true
}
