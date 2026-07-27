package placement

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The host prompt cache decides how much of a conversation's prefix survives to
// its next turn, and ggrun sized it from a KV formula that does not describe it.
// A saved entry is target state + draft state + checkpoints; on this project KV
// for a whole 1M context estimated 9238 MiB while one entry for a 126k-token
// prompt measured 6494. Sized from the estimate, the budget held one entry, so a
// four-slot server evicted one conversation to admit the next and re-prefilled
// ~126k tokens on the turn after. Measured 17.5% cache-read on that server
// against 88.3% on a single-slot server sharing the same box and client.
//
// The backend already knows the true figure and prints it, so ggrun reads it
// rather than modelling it -- the same choice already made for compute buffers
// and runtime graph growth.

// promptCacheStateRe matches the server's own cache accounting:
//
//	srv  update: - cache state: 3 prompts, 19484.211 MiB (limits: 9761.000 MiB, 0 tokens, 4096 est)
//
// It is logged at trace level, one above llama.cpp's default of info, so a
// launch that wants this measurement must raise verbosity.
var promptCacheStateRe = regexp.MustCompile(
	`cache state:\s*(\d+)\s+prompts,\s*([0-9.]+)\s*MiB\s*\(limits:\s*([0-9.]+)\s*MiB,\s*(\d+)\s*tokens`)

// promptCacheEntryRe matches the eviction and skip lines, which carry an entry's
// size directly. These appear at warning level, so they are visible by default,
// but only when the cache is already under pressure -- by then the re-prefill
// has been paid.
var promptCacheEntryRe = regexp.MustCompile(`entry \(size = ([0-9.]+) MiB\)`)

// promptCacheSkipRe matches an entry rejected for exceeding the whole budget.
// This one matters more than it looks: the backend skips silently as far as the
// cache is concerned, so the prompt is never stored and every turn re-prefills
// with no eviction line to explain why.
var promptCacheSkipRe = regexp.MustCompile(
	`prompt state size ([0-9.]+) MiB exceeds cache size limit ([0-9.]+) MiB`)

// PromptCacheObservation is what a backend log revealed about the prompt cache.
type PromptCacheObservation struct {
	// BytesPerToken is total cached bytes over total cached tokens. Zero when
	// the log had no trace-level cache state to read.
	BytesPerToken float64
	// LargestEntryMB is the biggest single entry seen, from eviction or skip
	// lines. Useful on its own: the budget must exceed one entry or nothing is
	// ever cached.
	LargestEntryMB float64
	// Prompts and LimitMB are the last reported cache occupancy and budget.
	Prompts int
	LimitMB float64
	// Skipped counts entries rejected for exceeding the budget outright.
	Skipped int
	// Evicted counts entries dropped to make room.
	Evicted int
}

// Measured reports whether the observation carries a usable per-token cost.
func (o PromptCacheObservation) Measured() bool { return o.BytesPerToken > 0 }

// ObservePromptCache extracts prompt-cache accounting from backend log output.
// It reads whatever is present: the trace-level state line gives the per-token
// cost directly, while eviction and skip lines still bound the entry size when
// verbosity was left at the default.
func ObservePromptCache(logText string) PromptCacheObservation {
	var obs PromptCacheObservation
	if strings.TrimSpace(logText) == "" {
		return obs
	}

	// Later lines describe a warmer cache, so the last state wins.
	for _, m := range promptCacheStateRe.FindAllStringSubmatch(logText, -1) {
		prompts, err1 := strconv.Atoi(m[1])
		sizeMB, err2 := strconv.ParseFloat(m[2], 64)
		limitMB, err3 := strconv.ParseFloat(m[3], 64)
		tokens, err4 := strconv.Atoi(m[4])
		if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
			continue
		}
		obs.Prompts, obs.LimitMB = prompts, limitMB
		if tokens > 0 && sizeMB > 0 {
			obs.BytesPerToken = sizeMB * 1024 * 1024 / float64(tokens)
		}
	}

	for _, m := range promptCacheEntryRe.FindAllStringSubmatch(logText, -1) {
		obs.Evicted++
		if v, err := strconv.ParseFloat(m[1], 64); err == nil && v > obs.LargestEntryMB {
			obs.LargestEntryMB = v
		}
	}
	for _, m := range promptCacheSkipRe.FindAllStringSubmatch(logText, -1) {
		obs.Skipped++
		if v, err := strconv.ParseFloat(m[1], 64); err == nil && v > obs.LargestEntryMB {
			obs.LargestEntryMB = v
		}
	}
	return obs
}

// promptCacheBudgetMB sizes the host prompt cache from a measured per-token cost
// so that every slot can keep one conversation resident.
//
// Slots is the right multiplier because a slot is exactly what gets evicted: a
// conversation returning to a slot whose prefix was dropped re-prefills from
// zero. Beyond one entry per slot there is no further hit rate to buy, so this
// deliberately does not grow to fill available RAM -- that memory is wanted by
// the expert layers and by the scope the backend runs inside.
//
// Returns 0 when nothing was measured, so the caller keeps its derived value
// rather than acting on a guess dressed as a measurement.
func promptCacheBudgetMB(bytesPerToken float64, promptTokens, slots int) int {
	if bytesPerToken <= 0 || promptTokens <= 0 {
		return 0
	}
	if slots < 1 {
		slots = 1
	}
	perEntryMB := bytesPerToken * float64(promptTokens) / (1024 * 1024)
	// One spare entry: a conversation is stored before the one it replaces is
	// dropped, so a budget of exactly n_slots entries still thrashes at the edge.
	return int(perEntryMB * float64(slots+1))
}

// RecordMeasuredPromptCache stores a per-token prompt-cache cost read from
// backend output, keyed exactly like the other probes so a different context,
// KV type, slot count or backend cannot inherit it.
func RecordMeasuredPromptCache(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, bytesPerToken float64) error {
	if model == nil || bytesPerToken <= 0 || ctxSize <= 0 || ubatch <= 0 {
		return nil
	}
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	compute := map[int]int{}
	growth := map[int]int{}
	estimated := map[int]bool{}
	kvPerLayerMB := 0
	if pc != nil {
		for k, v := range pc.ComputeBufByGPU {
			compute[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthByGPU {
			growth[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthEstimatedByGPU {
			estimated[k] = v
		}
		kvPerLayerMB = pc.KVPerLayerMB
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel,
		compute, growth, estimated, kvPerLayerMB, bytesPerToken)
}

// MeasuredPromptCacheBytesPerToken returns a stored measurement, or 0 when this
// shape has never been observed.
func MeasuredPromptCacheBytesPerToken(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) float64 {
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if pc == nil {
		return 0
	}
	return pc.PromptCacheBytesPerToken
}

// LargestRoutedExpertLayerMB is the size of the biggest routed-expert layer,
// which is the unit placement actually moves between GPU and CPU.
//
// It exists so a CUDA out-of-memory abort can be answered with a real quantity.
// The abort at graph instantiation carries no allocation size, and the previous
// response to that was to reserve a tenth of the card -- a number derived from
// nothing, which either overshoots and withholds several layers permanently or
// undershoots and crashes again. Moving exactly one layer is the smallest step
// that changes the outcome, and its cost is already known exactly from the GGUF
// ledger, so the retry converges instead of guessing.
//
// Returns 0 when the ledger is unavailable, so the caller can fall back rather
// than reserve zero.
func LargestRoutedExpertLayerMB(model *ModelProfile) int {
	if model == nil {
		return 0
	}
	layers := model.RoutedExpertLayerBytes
	if len(layers) == 0 {
		layers = model.ExpertLayerBytes
	}
	var maxBytes int64
	for _, b := range layers {
		if b > maxBytes {
			maxBytes = b
		}
	}
	if maxBytes <= 0 {
		return 0
	}
	return int((maxBytes + (1 << 20) - 1) / (1 << 20))
}
