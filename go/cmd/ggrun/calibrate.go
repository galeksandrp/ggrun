package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/raketenkater/ggrun/pkg/advisor"
	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// calibrationMinImprovementPct is the margin a challenger must beat the default
// by before it is cached as the winner. Below this the two placements are
// indistinguishable at micro-probe precision and the estimated default stands.
const calibrationMinImprovementPct = 3.0

const (
	// Auto mode gets one live challenger regardless of model size. This is the
	// critical first-live feedback path for large MoE models, while the bounded
	// start/failure/time budget prevents an unattended restart spiral.
	calibrationAutoMaxCandidates   = 3 // default plus the bounded 1024/2048 MoE ladder
	calibrationForcedMaxCandidates = 4
	calibrationAutoFailureBudget   = 1
	calibrationForcedFailureBudget = 2
	calibrationAutoMaxElapsed      = 20 * time.Minute
	calibrationForcedMaxElapsed    = 60 * time.Minute
	calibrationAdvisorTiePct       = 3.0
)

type calibrationBudget struct {
	MaxCandidates int
	MaxFailures   int
	MaxElapsed    time.Duration
}

func calibrationBudgetFor(mode string) calibrationBudget {
	if mode == calibrateOn {
		return calibrationBudget{calibrationForcedMaxCandidates, calibrationForcedFailureBudget, calibrationForcedMaxElapsed}
	}
	return calibrationBudget{calibrationAutoMaxCandidates, calibrationAutoFailureBudget, calibrationAutoMaxElapsed}
}

// calibrationPlan decides whether this launch should run first-launch
// calibration and, if so, the candidate set to measure. It is a pure function
// of the request, model, hardware, and whether a prior decision exists, so the
// policy is unit-testable without starting a server.
//
// Returns nil when there is nothing to do: calibration disabled, a decision
// already cached for this exact scope, or the placement has no alternatives
// (single GPU, CPU-only, non-MoE single-GPU, symmetric dense split).
func calibrationPlan(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) []placement.CalibrationCandidate {
	if req == nil || cfg == nil || model == nil || be == nil || caps == nil || strategy == nil {
		return nil
	}
	mode := req.Calibrate
	if mode == "" {
		mode = calibrateAuto
	}
	if mode == calibrateOff {
		return nil
	}
	candidates := calibrationCandidates(req, cfg, model, be, caps, strategy)
	if len(candidates) < 2 {
		return nil
	}
	budget := calibrationBudgetFor(mode)
	if len(candidates) > budget.MaxCandidates {
		candidates = candidates[:budget.MaxCandidates]
	}
	// One decision per scope: once any candidate has won, later launches apply
	// it directly instead of re-measuring.
	scopeKey := calibrationScopeKey(req, model, be, caps, strategy)
	if _, err := placement.LoadCalibrationDecision(cfg.CacheDir, scopeKey); err == nil {
		return nil
	}
	return candidates
}

// applyCalibrationDecision returns the strategy to serve with when a prior
// calibration decision exists for this scope. The winner is re-derived by name
// from the deterministic candidate generator rather than deserialized, so the
// full placement (KV placement, main GPU, split) is reproduced exactly and can
// never be a partial overlay. Context, batch, and slots always come from the
// current request. Returns the input strategy unchanged when no decision
// applies or the winner's candidate no longer exists for this hardware.
func applyCalibrationDecision(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) *placement.Strategy {
	if req == nil || cfg == nil || model == nil || be == nil || caps == nil || strategy == nil {
		return strategy
	}
	if req.Calibrate == calibrateOff {
		return strategy
	}
	// --no-cached-config is the escape hatch for a stale cached decision: do not
	// re-apply a measured calibration winner to this launch either.
	if req.NoCachedConfig {
		return strategy
	}
	scopeKey := calibrationScopeKey(req, model, be, caps, strategy)
	decision, err := placement.LoadCalibrationDecision(cfg.CacheDir, scopeKey)
	if err != nil || decision.Winner == "" || decision.Winner == "default" {
		return strategy
	}
	for _, cand := range calibrationCandidates(req, cfg, model, be, caps, strategy) {
		if cand.Name == decision.Winner {
			fmt.Printf("[calibrate] applying cached winner %s (%.1f vs default %.1f tok/s)\n", decision.Winner, decision.WinnerTPS, decision.DefaultTPS)
			return cand.Strategy
		}
	}
	return strategy
}

// recomputeAndApplyCalibration re-derives placement from current evidence and
// re-applies the cached calibration winner on top, so a corrective recompute
// never replaces the applied winner with an unbenchmarked estimate. Mirror of
// the computeStrategy path in main(); SkipPlacementCache and CacheFile are
// caller-managed because a mid-retry placement is never persisted.
func recomputeAndApplyCalibration(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, s *placement.Strategy) *placement.Strategy {
	return applyCalibrationDecision(req, cfg, model, be, caps, s)
}

// calibrationScopeKey builds the opaque cache key for this launch shape. It
// mirrors placement.NewCalibrationScopeKey on the request's resolved options so
// a decision is valid only for the exact model + backend + hardware + workload
// + runtime knobs it was measured under.
func calibrationCandidates(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) []placement.CalibrationCandidate {
	cacheDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
	}
	opts := placementOptionsFromRequest(req, model, be, cacheDir)
	candidates := placement.CalibrationCandidates(caps, model, strategy, opts)
	if req == nil || req.ForceMMap || len(candidates) < 2 {
		return candidates
	}
	// Do not stop a resident server and then ask for a new disk-paging policy
	// halfway through calibration. An mmap-dependent alternate is eligible only
	// when the user approved mmap on the original command line or launch prompt.
	out := candidates[:1]
	for _, cand := range candidates[1:] {
		if cand.Strategy != nil && !cand.Strategy.MMapRequired {
			out = append(out, cand)
		}
	}
	return out
}

func calibrationScopeKey(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy) string {
	opts := placementOptionsFromRequest(req, model, be, "")
	key := placement.NewCalibrationScopeKey(model, caps, opts, strategy)
	return key.String()
}

type calibrationMeasurement struct {
	Name     string
	Strategy *placement.Strategy
	Args     []string
	Result   *benchmark.Result
	Score    float64
}

func exactCalibrationCandidate(candidateArgs, measuredArgs []string) bool {
	return formatCommand(candidateArgs) == formatCommand(measuredArgs)
}

func validCalibrationResult(result *benchmark.Result) bool {
	if result == nil || result.GenTPS <= 0 || result.PromptTPS <= 0 || result.GenTokens <= 0 || result.PromptTokens <= 0 {
		return false
	}
	return !math.IsNaN(result.GenTPS) && !math.IsInf(result.GenTPS, 0) &&
		!math.IsNaN(result.PromptTPS) && !math.IsInf(result.PromptTPS, 0)
}

// calibrationScore balances interactive decode and prompt ingestion. Ratios
// keep the two different units comparable, and the measured default is exactly
// 1.0. Correctness remains a hard gate: a candidate without a complete valid
// benchmark never receives a score and is never offered to the advisor.
func calibrationScore(result, baseline *benchmark.Result) float64 {
	if !validCalibrationResult(result) || !validCalibrationResult(baseline) {
		return 0
	}
	return 0.75*(result.GenTPS/baseline.GenTPS) + 0.25*(result.PromptTPS/baseline.PromptTPS)
}

func boundedCalibrationTimeout(configured, remaining time.Duration) time.Duration {
	if configured <= 0 || configured > remaining {
		return remaining
	}
	return configured
}

// runCalibration measures each candidate placement with a live micro-benchmark
// and returns the strategy to actually serve with. Automatic mode tries one
// challenger even for very large models, but a start/failure/elapsed budget
// makes the controller finite. The optional support optimizer only sees typed,
// already-measured candidates after every main-model process has stopped.
//
// The decision is returned pending: the caller persists it only after the
// winner passes the normal functional/cache/profile lifecycle and becomes
// active. A benchmark alone must never become last-known-good state.
func runCalibration(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, serverArgs []string, timeout time.Duration, p *server.Process, memoryRecovery *launchMemoryRecovery, resourceBaseline *detect.Capabilities) (*server.Process, *placement.Strategy, []string, *placement.CalibrationDecision) {
	candidates := calibrationPlan(req, cfg, model, be, caps, strategy)
	if len(candidates) < 2 {
		return p, strategy, serverArgs, nil
	}
	mode := req.Calibrate
	if mode == "" {
		mode = calibrateAuto
	}
	budget := calibrationBudgetFor(mode)
	scopeKey := calibrationScopeKey(req, model, be, caps, strategy)
	startedAt := time.Now()
	fmt.Printf("[calibrate] first live profile: measuring at most %d placements (failures=%d, elapsed=%s)\n",
		budget.MaxCandidates, budget.MaxFailures, budget.MaxElapsed)

	baseURL := fmt.Sprintf("http://localhost:%d", req.Port)
	bench := func() (*benchmark.Result, error) {
		remaining := budget.MaxElapsed - time.Since(startedAt)
		if remaining <= 3*time.Second {
			return nil, fmt.Errorf("calibration elapsed-time budget exhausted")
		}
		// Runner performs three requests. Dividing the remaining wall budget keeps
		// a stalled endpoint from multiplying the overall controller deadline.
		requestTimeout := 5 * time.Minute
		if perRequest := remaining / 3; perRequest < requestTimeout {
			requestTimeout = perRequest
		}
		runner := &benchmark.Runner{BaseURL: baseURL, Model: model.Basename, Timeout: requestTimeout}
		res, err := runner.Run()
		if err != nil {
			return nil, err
		}
		if !validCalibrationResult(res) {
			return nil, fmt.Errorf("benchmark returned incomplete metrics")
		}
		return res, nil
	}

	// The default is already running: measure it in place.
	defaultResult, err := bench()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[calibrate] baseline measurement failed (%v); serving default placement\n", err)
		return p, strategy, serverArgs, nil
	}
	measurements := []calibrationMeasurement{{
		Name: "default", Strategy: strategy, Args: serverArgs,
		Result: defaultResult, Score: 1,
	}}
	fmt.Printf("[calibrate] default: decode %.1f tok/s, prefill %.1f tok/s, score %.3f\n",
		defaultResult.GenTPS, defaultResult.PromptTPS, 1.0)

	curP := p
	failures := 0
	for _, cand := range candidates[1:] {
		remaining := budget.MaxElapsed - time.Since(startedAt)
		if remaining <= 3*time.Second {
			fmt.Fprintln(os.Stderr, "[calibrate] elapsed-time budget reached; stopping candidate search")
			break
		}
		candArgs := buildLaunchServerArgs(req, cfg, be, caps, model, cand.Strategy)
		if fmt.Sprintf("%v", candArgs) == fmt.Sprintf("%v", serverArgs) {
			continue // candidate serializes identically to the default; nothing to measure
		}
		if memoryRecovery.isRejected(candArgs) {
			fmt.Printf("[calibrate] skipping %s: its exact argv already failed memory admission in this launch\n", cand.Name)
			continue
		}
		fmt.Printf("[calibrate] measuring %s...\n", cand.Name)
		if curP != nil {
			if !stopCalibrationProcessAndWait(curP, "default before "+cand.Name, resourceBaseline, 30*time.Second) {
				return curP, strategy, serverArgs, nil
			}
			curP = nil
		}
		candidateTimeout := boundedCalibrationTimeout(timeout, remaining)
		cp, measuredStrategy, measuredArgs, serr := startLaunchWithCUDAOOMRecoveryState(req, cfg, model, cand.Strategy, be, caps, candArgs, candidateTimeout, memoryRecovery)
		if serr != nil {
			fmt.Fprintf(os.Stderr, "[calibrate] %s failed to start (%v); skipping\n", cand.Name, serr)
			failures++
			if failures >= budget.MaxFailures {
				fmt.Fprintln(os.Stderr, "[calibrate] failure budget reached; stopping candidate search")
				break
			}
			continue
		}
		// A larger ubatch candidate is useful only when that exact candidate
		// passed admission. Recovery may make it start by moving experts or
		// derating ubatch; measuring that different argv and caching the original
		// candidate name would replay an unverified shape on the next launch.
		if !exactCalibrationCandidate(candArgs, measuredArgs) {
			fmt.Fprintf(os.Stderr, "[calibrate] %s needed memory recovery; skipping the altered candidate\n", cand.Name)
			if !stopCalibrationProcessAndWait(cp, cand.Name+" after memory recovery", resourceBaseline, 30*time.Second) {
				return cp, measuredStrategy, measuredArgs, nil
			}
			failures++
			if failures >= budget.MaxFailures {
				fmt.Fprintln(os.Stderr, "[calibrate] failure budget reached; stopping candidate search")
				break
			}
			continue
		}
		result, berr := bench()
		if berr != nil {
			fmt.Fprintf(os.Stderr, "[calibrate] %s measurement failed (%v); skipping\n", cand.Name, berr)
			if !stopCalibrationProcessAndWait(cp, cand.Name+" after failed measurement", resourceBaseline, 30*time.Second) {
				return cp, measuredStrategy, measuredArgs, nil
			}
			failures++
			if failures >= budget.MaxFailures {
				fmt.Fprintln(os.Stderr, "[calibrate] failure budget reached; stopping candidate search")
				break
			}
			continue
		}
		score := calibrationScore(result, defaultResult)
		fmt.Printf("[calibrate] %s: decode %.1f tok/s, prefill %.1f tok/s, score %.3f\n",
			cand.Name, result.GenTPS, result.PromptTPS, score)
		measurements = append(measurements, calibrationMeasurement{
			Name: cand.Name, Strategy: measuredStrategy, Args: measuredArgs,
			Result: result, Score: score,
		})
		if !stopCalibrationProcessAndWait(cp, "measured candidate "+cand.Name, resourceBaseline, 30*time.Second) {
			return cp, measuredStrategy, measuredArgs, nil
		}
	}

	// A failed challenger still leaves the known-good default stopped. Bring it
	// back immediately; no incomplete candidate is ever cached or advised.
	if len(measurements) < 2 {
		if curP != nil {
			return curP, strategy, serverArgs, nil
		}
		curP = restartPlacement(req, cfg, model, strategy, be, caps, serverArgs, timeout, memoryRecovery)
		return curP, strategy, serverArgs, nil
	}

	best := measurements[0]
	for _, measured := range measurements[1:] {
		if measured.Score > best.Score*(1+calibrationMinImprovementPct/100) {
			best = measured
		}
	}
	selected, optimized, optimizerErr := maybeOptimizeCalibration(req, cfg, model, be, caps, scopeKey, measurements)
	if optimizerErr != nil {
		fmt.Fprintf(os.Stderr, "[calibrate] refusing main-model restart: %v\n", optimizerErr)
		return nil, strategy, serverArgs, nil
	}
	if optimized {
		for _, measured := range measurements {
			if measured.Name != selected {
				continue
			}
			floor := best.Score * (1 - calibrationAdvisorTiePct/100)
			if measured.Score >= floor {
				fmt.Printf("[calibrate] support optimizer selected measured near-tie %s (score %.3f)\n", measured.Name, measured.Score)
				best = measured
			} else {
				fmt.Fprintf(os.Stderr, "[calibrate] rejected advisor selection %s: score %.3f is below deterministic floor %.3f\n", measured.Name, measured.Score, floor)
			}
			break
		}
	}

	// The support helper (when enabled) has stopped and verified resource release.
	// Only now may the winning main-model process be started.
	curP = restartPlacement(req, cfg, model, best.Strategy, be, caps, best.Args, timeout, memoryRecovery)
	if curP == nil && best.Name != "default" {
		fmt.Fprintln(os.Stderr, "[calibrate] winner restart failed; restoring measured default")
		best = measurements[0]
		curP = restartPlacement(req, cfg, model, best.Strategy, be, caps, best.Args, timeout, memoryRecovery)
	}
	if curP == nil {
		return nil, best.Strategy, best.Args, nil
	}

	pending := &placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: best.Name,
		ModelBasename:    filepath.Base(model.Path),
		DefaultTPS:       defaultResult.GenTPS,
		DefaultPromptTPS: defaultResult.PromptTPS,
		DefaultScore:     1,
		WinnerTPS:        best.Result.GenTPS,
		WinnerPromptTPS:  best.Result.PromptTPS,
		WinnerScore:      best.Score,
		Improvement:      (best.Score - 1) * 100,
	}
	fmt.Printf("[calibrate] provisional winner %s (decode %.1f, prefill %.1f, score %.3f); awaiting lifecycle verification\n",
		best.Name, best.Result.GenTPS, best.Result.PromptTPS, best.Score)
	return curP, best.Strategy, best.Args, pending
}

func stopCalibrationProcess(process *server.Process, label string) bool {
	if process == nil {
		return true
	}
	err := process.Stop()
	if process.IsRunning() {
		fmt.Fprintf(os.Stderr, "[calibrate] %s did not stop; refusing overlapping candidate/helper start", label)
		if err != nil {
			fmt.Fprintf(os.Stderr, ": %v", err)
		}
		fmt.Fprintln(os.Stderr)
		return false
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "[calibrate] %s exited with stop warning: %v\n", label, err)
	}
	return true
}

func stopCalibrationProcessAndWait(process *server.Process, label string, baseline *detect.Capabilities, timeout time.Duration) bool {
	if !stopCalibrationProcess(process, label) {
		return false
	}
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	deadline := time.Now().Add(timeout)
	for {
		if launchResourcesAtBaseline(baseline) {
			return true
		}
		if time.Now().After(deadline) {
			fmt.Fprintf(os.Stderr, "[calibrate] %s released its process but RAM/VRAM did not return to the pre-main baseline within %s; refusing an overlapping start\n", label, timeout)
			return false
		}
		time.Sleep(250 * time.Millisecond)
	}
}

// maybeOptimizeCalibration invokes NanoBeige only when it is explicitly on or
// fully installed in auto mode. Its candidate IDs and metrics come exclusively
// from successful ggrun benchmark runs, and the caller retains a deterministic
// near-tie guard even after schema validation.
func maybeOptimizeCalibration(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, scopeKey string, measurements []calibrationMeasurement) (string, bool, error) {
	if cfg == nil || len(measurements) < 2 {
		return "", false, nil
	}
	mode, online := cfg.SupportExpert, cfg.SupportOnline
	if req != nil {
		if req.SupportExpert != "" {
			mode = req.SupportExpert
		}
		online = req.SupportOnline
	}
	if mode == "off" {
		return "", false, nil
	}
	if mode == "auto" && !currentSupportStatus(cfg, caps).Ready {
		return "", false, nil
	}
	incident := calibrationAdvisorIncident(req, model, be, caps, scopeKey, measurements)
	preferred := ""
	if be != nil {
		preferred = be.Path
	}
	decision, report, runErr := runSupportIncidentFn(context.Background(), cfg, caps, preferred, incident, online)
	var decisionPtr *advisor.Decision
	if runErr == nil {
		decisionPtr = &decision
	}
	history, historyErr := advisor.SaveAnalysis(cfg.CacheDir, incident, decisionPtr, report, runErr)
	if historyErr != nil {
		fmt.Fprintf(os.Stderr, "[support] could not persist optimizer analysis: %v\n", historyErr)
	}
	if runErr != nil {
		fmt.Fprintf(os.Stderr, "[support] optimizer unavailable; deterministic measured score remains authoritative: %v\n", runErr)
		if errors.Is(runErr, advisor.ErrResourceReleaseUnverified) {
			return "", false, runErr
		}
		return "", false, nil
	}
	fmt.Printf("[support] optimizer decision=%s confidence=%.2f; helper release verified; record=%s\n",
		decision.Action, decision.Confidence, history)
	if decision.Action != advisor.ActionSelectCandidate {
		return "", false, nil
	}
	return decision.CandidateID, true, nil
}

func calibrationAdvisorIncident(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, scopeKey string, measurements []calibrationMeasurement) advisor.Incident {
	arch, family, identity := "", "", ""
	if model != nil {
		arch = model.ModelArch
	}
	if be != nil {
		family, identity = backendDialect(be), evidenceBackendCacheTag(be)
	}
	hardware := map[string]string{}
	if caps != nil {
		hardware["ram_total_mb"] = strconv.Itoa(caps.RAM.TotalMB)
		hardware["ram_free_mb"] = strconv.Itoa(caps.RAM.FreeMB)
		for _, gpu := range caps.GPUs {
			prefix := fmt.Sprintf("gpu%d_", gpu.Index)
			hardware[prefix+"name"] = gpu.Name
			hardware[prefix+"total_mb"] = strconv.Itoa(gpu.VRAMTotalMB)
			hardware[prefix+"free_mb"] = strconv.Itoa(gpu.VRAMFreeMB())
		}
	}
	settings := map[string]string{}
	if req != nil {
		settings["context"] = req.CtxFlag
		settings["kv_quality"] = req.KVQuality
		settings["parallel"] = strconv.Itoa(req.Parallel)
	}
	candidates := make([]advisor.Candidate, 0, len(measurements))
	observations := make([]advisor.Observation, 0, len(measurements))
	for index, measured := range measurements {
		properties := map[string]string{}
		if measured.Strategy != nil {
			properties["placement_type"] = string(measured.Strategy.Type)
			properties["kv_placement"] = measured.Strategy.KVPlacement
			properties["cpu_moe_layers"] = strconv.Itoa(measured.Strategy.NCPUMoE)
			properties["main_gpu"] = strconv.Itoa(measured.Strategy.MainGPU)
			properties["mmap"] = strconv.FormatBool(measured.Strategy.MMap)
		}
		candidates = append(candidates, advisor.Candidate{
			ID: measured.Name, Verified: true, Properties: properties,
			Metrics: map[string]float64{
				"decode_tps": measured.Result.GenTPS, "prefill_tps": measured.Result.PromptTPS,
				"decode_seconds": measured.Result.GenTimeS, "prefill_seconds": measured.Result.PromptTimeS,
				"balanced_score": measured.Score,
			},
		})
		observations = append(observations, advisor.Observation{
			Code: fmt.Sprintf("candidate_%d_measured", index), Component: "calibration",
			Value: measured.Score, Unit: "relative_score", Source: "ggrun_benchmark", Confidence: "measured",
			Attributes: map[string]string{"candidate_id": measured.Name},
		})
	}
	sum := sha256.Sum256([]byte(scopeKey + "|optimizer"))
	incident := advisor.Incident{
		ID: "opt-" + hex.EncodeToString(sum[:8]), Mode: advisor.ModeOptimizer,
		Architecture: arch, BackendFamily: family, BackendIdentity: identity,
		Workload: requestWorkloadProfile(req, model), ProfileState: "performance_measured",
		Hardware: hardware, Settings: settings, Observations: observations,
		AllowedActions: []advisor.ActionID{advisor.ActionNoAction, advisor.ActionSelectCandidate},
		Candidates:     candidates,
	}
	_ = incident.Normalize()
	return incident
}

// restartPlacement brings the selected measured strategy back up after the
// losing candidate stopped it. A nil return means the restart failed; the
// caller then reports the error and stops any reviewer before exiting, since
// leaving the user with no server after a calibration that already measured a
// working default is worse than failing loudly.
func restartPlacement(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities, serverArgs []string, timeout time.Duration, memoryRecovery *launchMemoryRecovery) *server.Process {
	p, _, _, err := restoreLaunchWithCUDAOOMRecoveryState(req, cfg, model, strategy, be, caps, serverArgs, timeout, memoryRecovery)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[calibrate] restart of best placement failed: %v\n", err)
		return nil
	}
	return p
}
