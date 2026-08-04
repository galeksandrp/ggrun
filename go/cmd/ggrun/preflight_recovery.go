package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// launchMemoryRecovery is the single state machine shared by contained
// preflight and real-startup allocation recovery. The two paths produce
// different evidence, but neither may resurrect an argv that this launch has
// already disproved.
type launchMemoryRecovery struct {
	rejected map[string]struct{}
}

func newLaunchMemoryRecovery() *launchMemoryRecovery {
	return &launchMemoryRecovery{rejected: map[string]struct{}{}}
}

func (r *launchMemoryRecovery) reject(args []string) {
	if r == nil {
		return
	}
	if r.rejected == nil {
		r.rejected = map[string]struct{}{}
	}
	r.rejected[launchArgsIdentity(args)] = struct{}{}
}

func (r *launchMemoryRecovery) isRejected(args []string) bool {
	if r == nil {
		return false
	}
	_, rejected := r.rejected[launchArgsIdentity(args)]
	return rejected
}

func (r *launchMemoryRecovery) hasRejections() bool {
	return r != nil && len(r.rejected) > 0
}

func (r *launchMemoryRecovery) rejectionCount() int {
	if r == nil {
		return 0
	}
	return len(r.rejected)
}

// recomputeDecision reports whether next changes the exact process argv and,
// if so, whether that argv has already failed this launch's memory checks.
func (r *launchMemoryRecovery) recomputeDecision(current, next []string) (changed, rejected bool) {
	if launchArgsIdentity(current) == launchArgsIdentity(next) {
		return false, false
	}
	if r == nil {
		return true, false
	}
	_, rejected = r.rejected[launchArgsIdentity(next)]
	return true, rejected
}

// launchArgsIdentity is deliberately the complete argv, not only the common
// CUDA flags: backend-specific switches such as full SWA cache and checkpoint
// modes also change memory. NUL is not valid inside an argv element, so this is
// collision-free for process arguments.
func launchArgsIdentity(args []string) string {
	return strings.Join(args, "\x00")
}

// recoverPreflightOOM converts one measured allocation failure into a strictly
// different launch shape. Failed graph allocations are useful component
// measurements, but are never promoted to complete allocation evidence.
func recoverPreflightOOM(
	req *launchRequest,
	cfg *config.Config,
	model *placement.ModelProfile,
	be *backendInfo,
	caps, runtimeCaps *detect.Capabilities,
	visibleToPhysical map[int]int,
	strategy *placement.Strategy,
	serverArgs []string,
	oomPenalty map[int]int,
	outcome preflightOutcome,
) (*placement.Strategy, []string, string, error) {
	if !outcome.DoesNotFit || outcome.Device < 0 || model == nil || strategy == nil {
		return nil, nil, "", fmt.Errorf("invalid preflight allocation failure")
	}
	if outcome.AllocMB <= 0 {
		outcome.AllocMB = maxPreflightInt(outcome.DeficitMB, 1)
		outcome.AllocMBMeasured = false
	}
	if outcome.DeficitMB <= 0 {
		outcome.DeficitMB = 1
	}

	var candidate *placement.Strategy
	var replanErr error
	if outcome.IsComputeBuffer && outcome.AllocMBMeasured && cfg != nil && be != nil && runtimeCaps != nil {
		cacheBackendTag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
		recordErr := placement.RecordMeasuredComputeBuffers(
			cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize,
			strategy.KVQuality, strategy.KVPlacement, cacheBackendTag,
			runtimeCaps.GPUs, strategy.Parallel, map[int]int{outcome.Device: outcome.AllocMB},
		)
		if recordErr == nil {
			opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
			// Preserve a prior derating: a retry at ubatch 256 must never be
			// recomputed from the original automatic request back to 512.
			opts.UBatchSize = strategy.UBatchSize
			opts.SkipPlacementCache = true
			opts.CacheFile = ""
			candidate, replanErr = placement.Compute(caps, model, opts)
		}
	}

	if candidate == nil && !outcome.IsComputeBuffer {
		physicalDev := physicalGPUIndex(outcome.Device, visibleToPhysical)
		oomPenalty[physicalDev] += outcome.DeficitMB
		candidate, replanErr = placement.ReplanAfterOOM(
			caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir), oomPenalty,
		)
	}

	nextStrategy, nextArgs, method, changed := applyMemoryRecoverySelection(
		req, strategy, serverArgs, candidate, model, runtimeCaps, outcome,
	)
	if !changed {
		detail := ""
		if replanErr != nil {
			detail = ": " + replanErr.Error()
		}
		return nil, nil, "", fmt.Errorf(
			"CUDA%d allocation of %d MiB exceeded the guard by %d MiB, but neither exact re-planning nor deterministic derating changed the effective memory configuration%s",
			outcome.Device, outcome.AllocMB, outcome.DeficitMB, detail,
		)
	}
	return nextStrategy, nextArgs, method, nil
}

// applyMemoryRecoverySelection is the only place that turns a selected memory
// recovery action into the next Strategy+argv pair. Both preflight failures and
// real allocator OOMs use it, so policy ordering and state mutation cannot
// drift between the two loops again.
func applyMemoryRecoverySelection(
	req *launchRequest,
	current *placement.Strategy,
	currentArgs []string,
	candidate *placement.Strategy,
	model *placement.ModelProfile,
	caps *detect.Capabilities,
	outcome preflightOutcome,
) (*placement.Strategy, []string, string, bool) {
	// Withdraw a ggrun-generated --swa-full before shedding any weights.
	//
	// It is the largest single reclaimable block on a memory failure and the
	// cheapest to give up: measured 2026-08-03 on DeepSeek-V4-Flash, it cost
	// 6196 MiB of KV on CUDA0 against 871 MiB without — 5.3 GiB, more than an
	// expert layer and a half — while the very runs that carried it still logged
	// "forcing full prompt re-processing due to lack of cache data", so it was
	// buying no prefix reuse at all. Dropping it only ever frees VRAM, so the
	// placement computed against the larger cache stays valid; the next loop
	// re-plans around the headroom. A --swa-full typed on the command line is an
	// instruction and is never withdrawn here.
	if req != nil && hasArg(currentArgs, "--swa-full") && !userExplicitBackendFlag(req, "--swa-full") {
		// Compare the argv directly, not the memory fingerprint: --swa-full is
		// not one of the canonical memory args, so the fingerprint is identical
		// with and without it even though the KV allocation differs sevenfold.
		withdrawn := setPassthroughBoolFlag(currentArgs, "--swa-full", false)
		if len(withdrawn) != len(currentArgs) {
			req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", false)
			disableBackendFlag(req, "--swa-full", "withdrawn after a memory failure; it reserves a full-context KV cache this model does not reuse")
			return current, withdrawn, "swa-full-withdrawn", true
		}
	}
	nextArgs, entry, method, changed := selectChangedPreflightRecovery(
		currentArgs, candidate, model, caps, outcome,
	)
	if !changed {
		return nil, nil, "", false
	}
	if method == "replanned" {
		if candidate == nil {
			return nil, nil, "", false
		}
		return candidate, nextArgs, method, true
	}
	if current == nil || entry == nil {
		return nil, nil, "", false
	}
	applyDeratedPlacementEntry(current, entry)
	return current, nextArgs, method, true
}

// selectChangedPreflightRecovery accepts a packer result only when it changes
// the backend's effective memory flags. Otherwise it applies a monotonic
// black-box fallback. A known failed CUDA device first loses one expert layer;
// only when that device has no movable expert layer does a graph OOM lower
// ubatch. Returning changed=false forbids an identical reload.
func selectChangedPreflightRecovery(
	currentArgs []string,
	candidate *placement.Strategy,
	model *placement.ModelProfile,
	caps *detect.Capabilities,
	outcome preflightOutcome,
) ([]string, *placement.CacheEntry, string, bool) {
	currentFingerprint := effectiveMemoryArgsFingerprint(currentArgs)
	candidateLowersUBatch := false
	if candidate != nil {
		currentUB := placement.CurrentUBatch(currentArgs)
		if outcome.IsComputeBuffer && currentUB > 0 {
			switch {
			case candidate.UBatchSize > currentUB:
				candidate = nil
			case candidate.UBatchSize > 0 && candidate.UBatchSize < currentUB:
				candidateLowersUBatch = true
			}
		}
	}
	var candidateArgs []string
	if candidate != nil {
		candidateArgs = patchPlacementArgs(currentArgs, candidate)
		if effectiveMemoryArgsFingerprint(candidateArgs) == currentFingerprint {
			candidateArgs = nil
		}
	}
	// A same-ubatch re-pack that moves weights is preferable to a blind layer
	// drop. A candidate whose only way forward is a smaller ubatch waits until
	// the failed device has first lost one routed expert layer.
	if candidateArgs != nil && !candidateLowersUBatch {
		return candidateArgs, nil, "replanned", true
	}

	// The exact deficit is the minimum memory that must be reclaimed from the
	// loaded state. Passing the full allocation here could drop many expert
	// layers even when the guard was exceeded by only a few MiB.
	derateMB := maxPreflightInt(outcome.DeficitMB, 1)
	if outcome.IsComputeBuffer {
		// The failed device is stronger evidence than a generic compute-buffer
		// classification. Move one routed expert layer off that exact GPU before
		// reducing global compute throughput via ubatch.
		nextArgs, entry, ok := placement.DerateCUDAOOMArgs(
			currentArgs, model, caps, outcome.Device, derateMB, false,
		)
		if ok && effectiveMemoryArgsFingerprint(nextArgs) != currentFingerprint {
			return nextArgs, entry, "expert-derate", true
		}
	}
	if candidateArgs != nil {
		return candidateArgs, nil, "replanned", true
	}
	nextArgs, entry, ok := placement.DerateCUDAOOMArgs(
		currentArgs, model, caps, outcome.Device, derateMB, outcome.IsComputeBuffer,
	)
	if !ok || effectiveMemoryArgsFingerprint(nextArgs) == currentFingerprint {
		return nil, nil, "", false
	}
	method := "expert-derate"
	if outcome.IsComputeBuffer && entry != nil && entry.UBatchSize > 0 {
		method = "ubatch-derate"
	}
	return nextArgs, entry, method, true
}

var memoryArgCanonical = map[string]string{
	"-m": "model", "--model": "model",
	"-c": "ctx", "--ctx-size": "ctx", "--ctx": "ctx",
	"-b": "batch", "--batch-size": "batch",
	"-ub": "ubatch", "--ubatch-size": "ubatch",
	"-ctk": "cache-k", "--cache-type-k": "cache-k",
	"-ctv": "cache-v", "--cache-type-v": "cache-v",
	"-np": "parallel", "--parallel": "parallel",
	"-ngl": "gpu-layers", "--n-gpu-layers": "gpu-layers", "--gpu-layers": "gpu-layers",
	"-ts": "tensor-split", "--tensor-split": "tensor-split",
	"-sm": "split-mode", "--split-mode": "split-mode",
	"-ot": "override-tensor", "--override-tensor": "override-tensor",
	"-ncmoe": "n-cpu-moe", "--n-cpu-moe": "n-cpu-moe",
	"-fa": "flash-attn", "--flash-attn": "flash-attn",
	"-mg": "main-gpu", "--main-gpu": "main-gpu",
	"-dev": "device", "--device": "device",
}

// effectiveMemoryArgsFingerprint follows the backend's last-value-wins argv
// behavior. This catches retries where ggrun changed an earlier generated flag
// but a later user override kept the effective placement identical.
func effectiveMemoryArgsFingerprint(args []string) string {
	values := map[string]string{}
	for i := 0; i < len(args); i++ {
		canonical, ok := memoryArgCanonical[args[i]]
		if !ok || i+1 >= len(args) || strings.HasPrefix(args[i+1], "-") {
			continue
		}
		values[canonical] = args[i+1]
		i++
	}
	for _, flag := range []string{"--no-kv-offload", "--no-mmap", "--mmap"} {
		if hasArg(args, flag) {
			values[flag] = "true"
		}
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var parts []string
	for _, key := range keys {
		parts = append(parts, key+"="+values[key])
	}
	return strings.Join(parts, "\n")
}
