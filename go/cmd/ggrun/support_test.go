package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/advisor"
	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestLaunchFailureIncidentContainsTypedEvidenceOnly(t *testing.T) {
	req := &launchRequest{
		CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1,
		ExtraArgs: []string{"--swa-full"},
	}
	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	strategy := &placement.Strategy{ContextSize: 131072, UBatchSize: 512, NCPUMoE: 37, MaxCheckpoints: 8}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	secret := "SECRET_RAW_LOG_DO_NOT_COPY"
	incident := launchFailureIncident(req, model, &backendInfo{Tag: "llama", Identity: "build"}, caps, strategy,
		errors.New("memory preflight failed: "+secret), "backend_start")
	if len(incident.Observations) != 1 || incident.Observations[0].Code != "memory_preflight_failed" {
		t.Fatalf("typed classification lost: %#v", incident.Observations)
	}
	for _, value := range incident.Settings {
		if value == secret {
			t.Fatal("raw backend error leaked into advisor settings")
		}
	}
	if !containsAdvisorAction(incident.AllowedActions, advisor.ActionLowerUBatch) ||
		!containsAdvisorAction(incident.AllowedActions, advisor.ActionRemoveGeneratedFeature) {
		t.Fatalf("safe controller actions missing: %v", incident.AllowedActions)
	}
	for _, unsupported := range []advisor.ActionID{
		advisor.ActionMoveExpertLayer, advisor.ActionSelectCompatibleBackend, advisor.ActionRefreshBackend,
	} {
		if containsAdvisorAction(incident.AllowedActions, unsupported) {
			t.Fatalf("incident advertised controller action %q that launch retry cannot execute", unsupported)
		}
	}
}

// A device-scoped CUDA OOM is the one failure where shedding a layer from the
// failing card beats cutting global prefill throughput, and it is what the
// deterministic ladder already prefers. Before this, the advisor could only ever
// answer "lower ubatch" no matter what the evidence said.
func TestLaunchFailureIncidentOffersExpertMoveOnDeviceScopedOOM(t *testing.T) {
	req := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "f16", Parallel: 1}
	model := &placement.ModelProfile{
		ModelArch: "deepseek4", IsMoE: true, NumLayers: 43, LeadingDense: 0,
		ExpertBytes: 43 * 2900 * 1024 * 1024,
	}
	strategy := &placement.Strategy{ContextSize: 131072, UBatchSize: 512, NCPUMoE: 37}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	incident := launchFailureIncident(req, model, &backendInfo{Tag: "llama", Identity: "build"}, caps, strategy,
		errors.New("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9348.32 MiB on device 0: cudaMalloc failed: out of memory"), "backend_start")
	if !containsAdvisorAction(incident.AllowedActions, advisor.ActionMoveExpertLayer) {
		t.Fatalf("device-scoped OOM did not offer an expert move: %v", incident.AllowedActions)
	}

	// The decision must reach the packer as a VRAM budget reduction, never as argv.
	if !applyAdvisorDecision(req, model, strategy, advisor.Decision{
		Action: advisor.ActionMoveExpertLayer, Device: 0, Count: 2,
	}) {
		t.Fatal("bounded expert-move decision was rejected")
	}
	if got := req.AdvisorVRAMPenaltyMB[0]; got != 2*expertLayerVRAMMB(model) {
		t.Fatalf("expert-move penalty = %d MiB, want %d", got, 2*expertLayerVRAMMB(model))
	}

	// Without a usable per-layer cost the action must be withheld, not guessed.
	if applyAdvisorDecision(&launchRequest{}, &placement.ModelProfile{}, strategy, advisor.Decision{
		Action: advisor.ActionMoveExpertLayer, Device: 0, Count: 1,
	}) {
		t.Fatal("expert move was applied without a known per-layer expert cost")
	}
}

func TestApplyAdvisorDecisionIsBoundedAndProtectsUserFlags(t *testing.T) {
	req := &launchRequest{ExtraArgs: []string{"--swa-full"}}
	strategy := &placement.Strategy{UBatchSize: 512}
	if !applyAdvisorDecision(req, nil, strategy, advisor.Decision{Action: advisor.ActionLowerUBatch, UBatch: 128}) || req.UBatchSize != 128 {
		t.Fatalf("controller ubatch rung was not applied: %#v", req)
	}
	if applyAdvisorDecision(req, nil, strategy, advisor.Decision{Action: advisor.ActionLowerUBatch, UBatch: 64}) {
		t.Fatal("advisor was allowed a second ubatch mutation")
	}

	explicit := &launchRequest{ExtraArgs: []string{"--swa-full"}, OriginalArgs: []string{"--swa-full"}}
	if applyAdvisorDecision(explicit, nil, strategy, advisor.Decision{Action: advisor.ActionRemoveGeneratedFeature, Feature: advisor.FeatureSWAFull}) {
		t.Fatal("explicit user SWA flag was removed")
	}
	generated := &launchRequest{ExtraArgs: []string{"--swa-full"}}
	if !applyAdvisorDecision(generated, nil, strategy, advisor.Decision{Action: advisor.ActionRemoveGeneratedFeature, Feature: advisor.FeatureSWAFull}) || hasArg(generated.ExtraArgs, "--swa-full") {
		t.Fatalf("generated SWA feature was not removed: %v", generated.ExtraArgs)
	}
	if !applyAdvisorDecision(&launchRequest{}, nil, strategy, advisor.Decision{Action: advisor.ActionRemeasureAllocation}) {
		t.Fatal("remeasure action did not enter the deterministic cache-bypassing recompute path")
	}
}

// propose_ubatch applies the same strict-derate discipline as lower_ubatch but
// rides the {256,128,64} ladder. It must never be applied when the user set the
// ubatch, when the proposal is not strictly smaller, or after a prior ubatch
// mutation.
func TestApplyAdvisorDecisionProposeUBatch(t *testing.T) {
	req := &launchRequest{}
	strategy := &placement.Strategy{UBatchSize: 512}
	if !applyAdvisorDecision(req, nil, strategy, advisor.Decision{Action: advisor.ActionProposeUBatch, UBatch: 128}) || req.UBatchSize != 128 {
		t.Fatalf("propose_ubatch derate not applied: %#v", req)
	}
	if applyAdvisorDecision(req, nil, strategy, advisor.Decision{Action: advisor.ActionProposeUBatch, UBatch: 64}) {
		t.Fatal("advisor was allowed a second ubatch mutation")
	}
	// Never when the user set the ubatch.
	if applyAdvisorDecision(&launchRequest{UBatchSize: 256, UBatchSizeSet: true}, nil, strategy, advisor.Decision{Action: advisor.ActionProposeUBatch, UBatch: 64}) {
		t.Fatal("propose_ubatch mutated a user-set ubatch")
	}
	// Never when the proposal is not strictly smaller than the current rung.
	if applyAdvisorDecision(&launchRequest{}, nil, &placement.Strategy{UBatchSize: 128}, advisor.Decision{Action: advisor.ActionProposeUBatch, UBatch: 128}) {
		t.Fatal("propose_ubatch applied a non-derate proposal")
	}
}

// propose_layer_distribution reaches the packer as an AdvisorVRAMPenaltyMB
// reduction exactly like move_expert_layer, and is withheld without a known
// per-layer expert cost.
func TestApplyAdvisorDecisionProposeLayerDistribution(t *testing.T) {
	model := &placement.ModelProfile{
		IsMoE: true, NumLayers: 43, LeadingDense: 0,
		ExpertBytes: 43 * 2900 * 1024 * 1024,
	}
	req := &launchRequest{}
	if !applyAdvisorDecision(req, model, nil, advisor.Decision{Action: advisor.ActionProposeLayerDistribution, Device: 1, Count: 2}) {
		t.Fatal("layer distribution proposal was rejected")
	}
	if got := req.AdvisorVRAMPenaltyMB[1]; got != 2*expertLayerVRAMMB(model) {
		t.Fatalf("layer distribution penalty = %d MiB, want %d", got, 2*expertLayerVRAMMB(model))
	}
	// Without a usable per-layer cost the action must be withheld, not guessed.
	if applyAdvisorDecision(&launchRequest{}, &placement.ModelProfile{}, nil, advisor.Decision{Action: advisor.ActionProposeLayerDistribution, Device: 0, Count: 1}) {
		t.Fatal("layer distribution was applied without a known per-layer expert cost")
	}
}

// toggle_swa_full flips a controller-GENERATED swa-full in either direction. A
// user-explicit choice is never touched, and a decision that does not name
// swa-full is never applied.
func TestApplyAdvisorDecisionToggleSWAFull(t *testing.T) {
	generated := &launchRequest{ExtraArgs: []string{"--swa-full"}}
	if !applyAdvisorDecision(generated, nil, nil, advisor.Decision{Action: advisor.ActionToggleSWAFull, Feature: advisor.FeatureSWAFull}) || hasArg(generated.ExtraArgs, "--swa-full") {
		t.Fatalf("toggle_swa_full did not remove the generated feature: %v", generated.ExtraArgs)
	}
	// A true value confirms the generated feature (no-op removal of the negative
	// form) and must still pass through setPassthroughBoolFlag deterministically.
	reconfirm := &launchRequest{ExtraArgs: []string{"--swa-full"}}
	if !applyAdvisorDecision(reconfirm, nil, nil, advisor.Decision{Action: advisor.ActionToggleSWAFull, Feature: advisor.FeatureSWAFull, Value: true}) || !hasArg(reconfirm.ExtraArgs, "--swa-full") {
		t.Fatalf("toggle_swa_full did not preserve a true target: %v", reconfirm.ExtraArgs)
	}
	explicit := &launchRequest{ExtraArgs: []string{"--swa-full"}, OriginalArgs: []string{"--swa-full"}}
	if applyAdvisorDecision(explicit, nil, nil, advisor.Decision{Action: advisor.ActionToggleSWAFull, Feature: advisor.FeatureSWAFull}) {
		t.Fatal("toggle_swa_full touched a user-explicit swa-full flag")
	}
	absent := &launchRequest{}
	if applyAdvisorDecision(absent, nil, nil, advisor.Decision{Action: advisor.ActionToggleSWAFull, Feature: advisor.FeatureSWAFull}) {
		t.Fatal("toggle_swa_full created a feature that was never generated")
	}
}

// launchFailureIncident must grant the new actions only under their precise
// conditions: propose_ubatch above the derate ladder, propose_layer_distribution
// when the per-layer expert cost is known, toggle_swa_full only for generated
// swa-full.
func TestLaunchFailureIncidentGrantsNewActions(t *testing.T) {
	req := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1, ExtraArgs: []string{"--swa-full"}}
	model := &placement.ModelProfile{ModelArch: "deepseek4", IsMoE: true, NumLayers: 43, LeadingDense: 0, ExpertBytes: 43 * 2900 * 1024 * 1024}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	be := &backendInfo{Tag: "llama", Identity: "build"}

	// Strategy at the top rung: propose_ubatch is offered (all ladder values are
	// strict derates); a user-set ubatch withholds it.
	atTop := launchFailureIncident(req, model, be, caps, &placement.Strategy{ContextSize: 131072, UBatchSize: 512}, errors.New("unclassified"), "backend_start")
	if !containsAdvisorAction(atTop.AllowedActions, advisor.ActionProposeUBatch) {
		t.Fatalf("top-rung strategy did not offer propose_ubatch: %v", atTop.AllowedActions)
	}
	if !containsAdvisorAction(atTop.AllowedActions, advisor.ActionProposeLayerDistribution) {
		t.Fatalf("MoE model with known per-layer cost did not offer propose_layer_distribution: %v", atTop.AllowedActions)
	}
	if !containsAdvisorAction(atTop.AllowedActions, advisor.ActionToggleSWAFull) {
		t.Fatalf("generated swa-full did not offer toggle_swa_full: %v", atTop.AllowedActions)
	}

	userSet := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1, UBatchSize: 256, UBatchSizeSet: true}
	belowTop := launchFailureIncident(userSet, model, be, caps, &placement.Strategy{ContextSize: 131072, UBatchSize: 128}, errors.New("unclassified"), "backend_start")
	if containsAdvisorAction(belowTop.AllowedActions, advisor.ActionProposeUBatch) {
		t.Fatalf("user-set ubatch still offered propose_ubatch: %v", belowTop.AllowedActions)
	}

	// A non-MoE model with no known per-layer cost must not offer the distribution.
	dense := launchFailureIncident(req, &placement.ModelProfile{ModelArch: "deepseek4"}, be, caps, &placement.Strategy{ContextSize: 131072, UBatchSize: 512}, errors.New("unclassified"), "backend_start")
	if containsAdvisorAction(dense.AllowedActions, advisor.ActionProposeLayerDistribution) {
		t.Fatalf("dense model offered propose_layer_distribution: %v", dense.AllowedActions)
	}

	// User-explicit swa-full withholds toggle_swa_full.
	explicitSWA := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1, ExtraArgs: []string{"--swa-full"}, OriginalArgs: []string{"--swa-full"}}
	explicitIncident := launchFailureIncident(explicitSWA, model, be, caps, &placement.Strategy{ContextSize: 131072, UBatchSize: 512}, errors.New("unclassified"), "backend_start")
	if containsAdvisorAction(explicitIncident.AllowedActions, advisor.ActionToggleSWAFull) {
		t.Fatalf("user-explicit swa-full still offered toggle_swa_full: %v", explicitIncident.AllowedActions)
	}
}

func TestFindSupportBackendRequiresArchitectureLiteral(t *testing.T) {
	dir := t.TempDir()
	unsupported := filepath.Join(dir, "old-server")
	if err := os.WriteFile(unsupported, []byte("llama\x00deepseek4\x00"), 0o700); err != nil {
		t.Fatal(err)
	}
	supported := filepath.Join(dir, "new-server")
	if err := os.WriteFile(supported, []byte("llama\x00nanbeige\x00"), 0o700); err != nil {
		t.Fatal(err)
	}
	cfg := config.Defaults()
	cfg.AppHome = dir
	cfg.LlamaServer = unsupported
	got := findSupportBackend(cfg, nil, supported)
	if got == "" || got == unsupported {
		t.Fatalf("support backend=%q, want a verified architecture-capable binary", got)
	}
	if ok, probed := backends.BackendSupportsArch(got, "nanbeige"); !probed || !ok {
		t.Fatalf("selected backend does not prove nanbeige support: %q", got)
	}
}

func TestReviewedSupportBackendRejectsUnpinnedLegacyRecord(t *testing.T) {
	appHome := t.TempDir()
	t.Setenv("LLM_APP_HOME", appHome)
	binary := filepath.Join(appHome, "legacy-server")
	if err := os.WriteFile(binary, []byte("llama\x00nanbeige\x00"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := backends.Save([]backends.Backend{{
		Tag: "nanbeige42", Path: binary, RouteArch: "nanbeige",
	}}); err != nil {
		t.Fatal(err)
	}
	if _, err := reviewedSupportBackend(); err == nil {
		t.Fatal("architecture string alone must not verify helper provenance")
	}
}

func TestParseSupportLaunchFlags(t *testing.T) {
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "missing"))
	req, err := parseLaunchArgs([]string{"model.gguf", "--support-expert", "on", "--support-online"})
	if err != nil {
		t.Fatal(err)
	}
	if req.SupportExpert != "on" || !req.SupportOnline {
		t.Fatalf("support flags lost: %#v", req)
	}
	if !req.SupportOnlineSet {
		t.Fatalf("explicit --support-online did not mark SupportOnlineSet: %#v", req)
	}
	// --no-support-online is a user instruction: it sets online=false AND marks
	// the tri-state set, so a later escalation must NOT force online research.
	req, err = parseLaunchArgs([]string{"model.gguf", "--no-support-online"})
	if err != nil {
		t.Fatal(err)
	}
	if req.SupportOnline || !req.SupportOnlineSet {
		t.Fatalf("--no-support-online lost: online=%t set=%t", req.SupportOnline, req.SupportOnlineSet)
	}
	if _, err := parseLaunchArgs([]string{"model.gguf", "--support-expert", "shell"}); err == nil {
		t.Fatal("invalid support policy accepted")
	}
}

// The Layer-2 gate is an OR: a novel/unclassified failure escalates on its own,
// while a classified failure escalates only when the deterministic retry budget
// is exhausted AND the same class recurred. The AND side must not let a single
// symptom turn into an advisor process on the very first deterministic failure.
func TestShouldEscalateToAdvisorIsORGate(t *testing.T) {
	for _, signal := range []escalationSignal{
		{},
		{retriesExhausted: true},
		{sameCodeRecurred: true},
		{retriesExhausted: true, sameCodeRecurred: true},
	} {
		if !shouldEscalateToAdvisor("unclassified_launch_failure", signal) {
			t.Fatalf("unclassified failure must escalate with signal %+v", signal)
		}
	}
	if shouldEscalateToAdvisor("cuda_oom_after_deterministic_recovery", escalationSignal{}) {
		t.Fatal("classified failure escalated without any retry evidence")
	}
	if shouldEscalateToAdvisor("cuda_oom_after_deterministic_recovery", escalationSignal{retriesExhausted: true}) {
		t.Fatal("classified failure escalated with retries exhausted but no recurrence")
	}
	if shouldEscalateToAdvisor("cuda_oom_after_deterministic_recovery", escalationSignal{sameCodeRecurred: true}) {
		t.Fatal("classified failure escalated with recurrence but retries not exhausted")
	}
	if !shouldEscalateToAdvisor("cuda_oom_after_deterministic_recovery", escalationSignal{retriesExhausted: true, sameCodeRecurred: true}) {
		t.Fatal("classified recurring failure did not escalate with retries exhausted")
	}
}

// maybeAnalyzeLaunchFailure must consult the advisor only through the seam and
// only when the Layer-2 gate passes. A classified, non-recurring failure must
// not spawn a helper process at all.
func TestMaybeAnalyzeLaunchFailureEscalationGate(t *testing.T) {
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	cfg.SupportExpert = "on"

	req := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1}
	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	be := &backendInfo{Tag: "llama", Identity: "build"}

	original := runSupportIncidentFn
	defer func() { runSupportIncidentFn = original }()
	called := 0
	runSupportIncidentFn = func(_ context.Context, _ *config.Config, _ *detect.Capabilities, _ string, _ advisor.Incident, _ bool) (advisor.Decision, advisor.RunReport, error) {
		called++
		return advisor.Decision{
			Action: advisor.ActionNoAction, Confidence: 0.5,
			Rationale: "seam", EvidenceCodes: []string{"internal:seam"},
		}, advisor.RunReport{ReleaseVerified: true}, nil
	}

	// A classified, non-recurring failure must NOT reach the advisor.
	if decision, escalated := maybeAnalyzeLaunchFailure(req, cfg, model, be, caps, nil,
		errors.New("memory preflight failed: too tight"), "placement", escalationSignal{}); escalated {
		t.Fatalf("classified non-recurring failure escalated past the Layer-2 gate: %s", decision.Action)
	}
	if called != 0 {
		t.Fatalf("advisor consulted %d times for a non-escalating failure", called)
	}

	// An unclassified failure escalates regardless of retry state.
	decision, escalated := maybeAnalyzeLaunchFailure(req, cfg, model, be, caps, nil,
		errors.New("some novel failure"), "placement", escalationSignal{})
	if !escalated {
		t.Fatal("unclassified failure did not escalate")
	}
	if decision.Action != advisor.ActionNoAction || called != 1 {
		t.Fatalf("seam decision not returned: action=%s called=%d", decision.Action, called)
	}

	// A classified recurring failure with the deterministic budget exhausted
	// escalates too.
	if _, escalated := maybeAnalyzeLaunchFailure(req, cfg, model, be, caps, nil,
		errors.New("cudaMalloc failed: out of memory"), "backend_start",
		escalationSignal{retriesExhausted: true, sameCodeRecurred: true}); !escalated {
		t.Fatal("classified recurring failure with exhausted retries did not escalate")
	}
	if called != 2 {
		t.Fatalf("advisor consulted %d times, want 2", called)
	}
}

// The deterministic placement-side replan must follow the OOM recovery option
// discipline (SkipPlacementCache + cleared CacheFile + preserved derated
// ubatch), not merely SkipPlacementCache. On a CPU-only fixture it must produce
// a valid strategy while preserving an explicit prior ubatch.
func TestDeterministicReplanOnPlacementFailure(t *testing.T) {
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	req := &launchRequest{CtxFlag: "32768", Parallel: 1}
	model := &placement.ModelProfile{SizeBytes: 1, NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	be := &backendInfo{Tag: "llama", Identity: "build"}
	caps := &detect.Capabilities{CPU: detect.CPUInfo{Cores: 4}, RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 16384}}

	strategy, err := deterministicReplanOnPlacementFailure(req, model, be, cfg, caps, nil)
	if err != nil {
		t.Fatalf("deterministic replan failed: %v", err)
	}
	if strategy == nil || strategy.Type != placement.CPUOnly {
		t.Fatalf("replan did not produce a CPU-only strategy: %+v", strategy)
	}
	if strategy.PlacementCacheHit {
		t.Fatal("deterministic replan reused a placement cache entry despite the bypass")
	}

	// A prior strategy's derated ubatch must be forwarded into the recompute so
	// a GPU MoE retry does not bounce back up to the rung that already failed.
	// (buildCPUOnly re-pins its own ubatch, so the forwarded value is only
	// observable as the opts the packer receives; the helper is what preserves
	// it for the GPU paths.)
	prior := &placement.Strategy{UBatchSize: 128}
	strategy, err = deterministicReplanOnPlacementFailure(req, model, be, cfg, caps, prior)
	if err != nil {
		t.Fatalf("deterministic replan with prior ubatch failed: %v", err)
	}
	if strategy == nil {
		t.Fatal("replan with prior returned no strategy")
	}
}

func containsAdvisorAction(actions []advisor.ActionID, want advisor.ActionID) bool {
	for _, action := range actions {
		if action == want {
			return true
		}
	}
	return false
}

// --- Layer-3 consent gate ---

func TestAdvisorConsentPlanClassifiesEscalationCost(t *testing.T) {
	if plan := advisorConsentPlanForEscalation("unclassified_launch_failure"); plan.Tier != ConsentTierExpensive {
		t.Fatalf("unclassified failure tier = %v, want expensive", plan.Tier)
	}
	if plan := advisorConsentPlanForEscalation("cuda_oom_after_deterministic_recovery"); plan.Tier != ConsentTierExpensive {
		t.Fatalf("recurring CUDA OOM tier = %v, want expensive", plan.Tier)
	}
	if plan := advisorConsentPlanForEscalation("memory_preflight_failed"); plan.Tier != ConsentTierCheap {
		t.Fatalf("classified recurring failure tier = %v, want cheap", plan.Tier)
	}
	for _, code := range []string{"unclassified_launch_failure", "cuda_oom_after_deterministic_recovery", "memory_preflight_failed"} {
		if plan := advisorConsentPlanForEscalation(code); strings.TrimSpace(plan.Summary) == "" {
			t.Fatalf("escalation %q has an empty plan summary", code)
		}
	}
}

// The three new actions are deterministic placement reshaples: their consent
// tier is cheap, so an interactive launch auto-approves the bounded action after
// the consultation, and the plan names the exact knob being turned.
func TestAdvisorConsentPlanForNewActionsIsCheapAndNamed(t *testing.T) {
	for _, action := range []advisor.ActionID{
		advisor.ActionProposeUBatch, advisor.ActionProposeLayerDistribution, advisor.ActionToggleSWAFull,
	} {
		plan := advisorConsentPlanForAction(action, advisor.FeatureSWAFull)
		if plan.Tier != ConsentTierCheap {
			t.Fatalf("new action %s tier = %v, want cheap placement reshape", action, plan.Tier)
		}
		if strings.TrimSpace(plan.Summary) == "" {
			t.Fatalf("new action %s has an empty plan summary", action)
		}
	}
}

func TestConfirmAdvisorActionEnvOnAutoApproves(t *testing.T) {
	t.Setenv("GGRUN_ADVISOR_CONSENT", "on")
	var output bytes.Buffer
	if err := confirmAdvisorAction(
		AdvisorConsentPlan{Tier: ConsentTierExpensive, Summary: "run its helper model"},
		strings.NewReader(""), &output, false,
	); err != nil {
		t.Fatalf("GGRUN_ADVISOR_CONSENT=on did not auto-approve even non-interactively: %v", err)
	}
	if output.Len() != 0 {
		t.Fatalf("auto-approval wrote a prompt: %q", output.String())
	}
}

func TestConfirmAdvisorActionEnvOffDeclines(t *testing.T) {
	t.Setenv("GGRUN_ADVISOR_CONSENT", "off")
	if err := confirmAdvisorAction(
		AdvisorConsentPlan{Tier: ConsentTierCheap, Summary: "reshape placement"},
		strings.NewReader(""), io.Discard, true,
	); !errors.Is(err, ErrAdvisorDeclined) {
		t.Fatalf("GGRUN_ADVISOR_CONSENT=off did not decline: %v", err)
	}
}

func TestConfirmAdvisorActionInvalidEnvFailsClosedToPrompt(t *testing.T) {
	t.Setenv("GGRUN_ADVISOR_CONSENT", "definitely-not-a-policy")
	var output bytes.Buffer
	// Invalid policy must fall back to "prompt", never silently auto-approve:
	// a non-interactive launch must still fail closed with a rerun hint.
	if err := confirmAdvisorAction(
		AdvisorConsentPlan{Tier: ConsentTierExpensive, Summary: "run its helper model"},
		strings.NewReader(""), &output, false,
	); err == nil || !strings.Contains(err.Error(), "GGRUN_ADVISOR_CONSENT=on") {
		t.Fatalf("invalid env policy did not fail closed with a rerun hint: %v", err)
	}
}

func TestConfirmAdvisorActionFailsClosedWhenNonInteractive(t *testing.T) {
	var output bytes.Buffer
	// The TUI does not redirect stdin permanently, so the fail-closed test is
	// the interactive bool, not "stdin redirected".
	err := confirmAdvisorAction(
		AdvisorConsentPlan{Tier: ConsentTierCheap, Summary: "reshape placement"},
		strings.NewReader("yes\n"), &output, false,
	)
	if err == nil || !strings.Contains(err.Error(), "GGRUN_ADVISOR_CONSENT=on") {
		t.Fatalf("non-interactive consent did not fail closed with a rerun hint: %v", err)
	}
}

func TestConfirmAdvisorActionCheapAutoApprovesWhenInteractive(t *testing.T) {
	var output bytes.Buffer
	if err := confirmAdvisorAction(
		AdvisorConsentPlan{Tier: ConsentTierCheap, Summary: "reshape placement"},
		strings.NewReader(""), &output, true,
	); err != nil {
		t.Fatalf("cheap interactive consultation was not auto-approved: %v", err)
	}
	if output.Len() != 0 {
		t.Fatalf("cheap interactive approval wrote a prompt: %q", output.String())
	}
}

func TestConfirmAdvisorActionExpensivePromptsWhenInteractive(t *testing.T) {
	var output bytes.Buffer
	if err := confirmAdvisorAction(
		AdvisorConsentPlan{Tier: ConsentTierExpensive, Summary: "run its helper model"},
		strings.NewReader("yes\n"), &output, true,
	); err != nil {
		t.Fatalf("explicit approval was rejected: %v", err)
	}
	if !strings.Contains(output.String(), "Support advisor wants to run its helper model") {
		t.Fatalf("expensive consultation did not prompt: %q", output.String())
	}
}

func TestConfirmAdvisorActionExpensiveDecline(t *testing.T) {
	if err := confirmAdvisorAction(
		AdvisorConsentPlan{Tier: ConsentTierExpensive, Summary: "run its helper model"},
		strings.NewReader("no\n"), io.Discard, true,
	); !errors.Is(err, ErrAdvisorDeclined) {
		t.Fatalf("negative answer did not decline: %v", err)
	}
}

func TestConfirmAdvisorActionSWAFullHint(t *testing.T) {
	// The deterministic --no-swa-full alternative is surfaced when the blocked
	// consultation would remove the generated SWA-full feature.
	var output bytes.Buffer
	err := confirmAdvisorAction(
		AdvisorConsentPlan{Action: advisor.ActionRemoveGeneratedFeature, Feature: advisor.FeatureSWAFull, Summary: "remove swa-full"},
		strings.NewReader(""), &output, false,
	)
	if err == nil || !strings.Contains(err.Error(), "--no-swa-full") {
		t.Fatalf("SWA-full hint missing: %v", err)
	}
}

// retryStartWithAdvisor must consult the advisor only through the seam, and the
// Layer-3 consent gate must sit between the caller's stop/verify and the helper.
// A declined consultation must return the ORIGINAL launch error untouched; a
// GGRUN_ADVISOR_CONSENT=off launch must not spawn the helper process at all.
func TestRetryStartWithAdvisorConsentGate(t *testing.T) {
	t.Setenv("GGRUN_ADVISOR_CONSENT", "off")
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	cfg.SupportExpert = "on"

	req := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1}
	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	be := &backendInfo{Tag: "llama", Identity: "build"}
	strategy := &placement.Strategy{UBatchSize: 512}
	cause := errors.New("some novel launch failure")

	original := runSupportIncidentFn
	defer func() { runSupportIncidentFn = original }()
	called := 0
	runSupportIncidentFn = func(_ context.Context, _ *config.Config, _ *detect.Capabilities, _ string, _ advisor.Incident, _ bool) (advisor.Decision, advisor.RunReport, error) {
		called++
		return advisor.Decision{}, advisor.RunReport{}, errors.New("helper must never run with consent off")
	}

	_, _, _, _, retryErr := retryStartWithAdvisor(req, cfg, model, be, caps, strategy, cause, time.Minute, nil)
	if !errors.Is(retryErr, cause) {
		t.Fatalf("declined consultation error=%v, want original launch error", retryErr)
	}
	if called != 0 {
		t.Fatalf("helper consulted %d times with GGRUN_ADVISOR_CONSENT=off", called)
	}
}

// With GGRUN_ADVISOR_CONSENT=on the seam decision flows through to the bounded
// application path. A classified non-escalating failure still never reaches the
// helper even with consent on: the Layer-2 gate owns escalation.
func TestRetryStartWithAdvisorConsentOnRunsSeam(t *testing.T) {
	t.Setenv("GGRUN_ADVISOR_CONSENT", "on")
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	cfg.SupportExpert = "on"

	req := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1}
	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	be := &backendInfo{Tag: "llama", Identity: "build"}
	strategy := &placement.Strategy{UBatchSize: 512}

	original := runSupportIncidentFn
	defer func() { runSupportIncidentFn = original }()
	called := 0
	runSupportIncidentFn = func(_ context.Context, _ *config.Config, _ *detect.Capabilities, _ string, _ advisor.Incident, _ bool) (advisor.Decision, advisor.RunReport, error) {
		called++
		return advisor.Decision{
			Action: advisor.ActionNoAction, Confidence: 0.5,
			Rationale: "seam", EvidenceCodes: []string{"internal:seam"},
		}, advisor.RunReport{ReleaseVerified: true}, nil
	}

	// A classified, non-recurring failure must not reach the helper even with
	// consent on: Layer 2 gates escalation, Layer 3 gates approval.
	classified := errors.New("memory preflight failed: too tight")
	_, _, _, _, err := retryStartWithAdvisor(req, cfg, model, be, caps, strategy, classified, time.Minute, nil)
	if !errors.Is(err, classified) {
		t.Fatalf("classified non-escalating failure error=%v, want original", err)
	}
	if called != 0 {
		t.Fatalf("helper consulted %d times for a non-escalating failure", called)
	}

	// A novel failure with consent on is consulted through the seam. The helper
	// returns no_action, which retryStartWithAdvisor maps back to the original
	// error (nothing to apply), so the caller still sees the real failure.
	novel := errors.New("some novel launch failure")
	_, _, _, _, err = retryStartWithAdvisor(req, cfg, model, be, caps, strategy, novel, time.Minute, nil)
	if !errors.Is(err, novel) {
		t.Fatalf("no_action on novel failure error=%v, want original", err)
	}
	if called != 1 {
		t.Fatalf("helper consulted %d times, want 1", called)
	}
}

// escalatePlacementFailure is the placement-side Layer-2/Layer-3 boundary. With
// GGRUN_ADVISOR_CONSENT=off the consultation must fail closed and return the
// ORIGINAL replan error without ever consulting the helper; with the consent
// env on, a bounded decision flows through to the injected computeStrategy.
func TestEscalatePlacementFailureConsentGate(t *testing.T) {
	t.Setenv("GGRUN_ADVISOR_CONSENT", "off")
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	cfg.SupportExpert = "on"

	req := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1}
	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	be := &backendInfo{Tag: "llama", Identity: "build"}
	replanErr := errors.New("some novel placement failure")

	original := runSupportIncidentFn
	defer func() { runSupportIncidentFn = original }()
	called := 0
	runSupportIncidentFn = func(_ context.Context, _ *config.Config, _ *detect.Capabilities, _ string, _ advisor.Incident, _ bool) (advisor.Decision, advisor.RunReport, error) {
		called++
		return advisor.Decision{}, advisor.RunReport{}, errors.New("helper must never run with consent off")
	}

	computeCalled := 0
	computeStrategy := func(candidateReq *launchRequest) (*placement.Strategy, error) {
		computeCalled++
		return &placement.Strategy{UBatchSize: 64}, nil
	}

	got, err := escalatePlacementFailure(req, cfg, model, be, caps, "unclassified_launch_failure", replanErr, computeStrategy)
	if !errors.Is(err, replanErr) {
		t.Fatalf("declined escalation error=%v, want original replan error", err)
	}
	if got != nil {
		t.Fatalf("declined escalation returned a strategy: %+v", got)
	}
	if called != 0 {
		t.Fatalf("helper consulted %d times with GGRUN_ADVISOR_CONSENT=off", called)
	}
	if computeCalled != 0 {
		t.Fatalf("placement recomputed %d times with consent off", computeCalled)
	}
}

// With consent on, a novel placement failure is consulted through the seam and a
// bounded decision is applied before the injected computeStrategy recomputes. A
// classified, non-recurring failure must not reach the helper at all even with
// consent on: the Layer-2 gate owns escalation.
func TestEscalatePlacementFailureConsentOnRunsSeam(t *testing.T) {
	t.Setenv("GGRUN_ADVISOR_CONSENT", "on")
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	cfg.SupportExpert = "on"

	req := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1}
	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	be := &backendInfo{Tag: "llama", Identity: "build"}

	original := runSupportIncidentFn
	defer func() { runSupportIncidentFn = original }()
	called := 0
	runSupportIncidentFn = func(_ context.Context, _ *config.Config, _ *detect.Capabilities, _ string, _ advisor.Incident, _ bool) (advisor.Decision, advisor.RunReport, error) {
		called++
		// remeasure_allocation is the only action applyAdvisorDecision accepts with
		// a nil strategy (the placement failure means there is no prior strategy to
		// derate): it just enters the deterministic cache-bypassing recompute path,
		// exactly like the real placement escalation.
		return advisor.Decision{
			Action: advisor.ActionRemeasureAllocation, Confidence: 0.5,
			Rationale: "seam", EvidenceCodes: []string{"internal:seam"},
		}, advisor.RunReport{ReleaseVerified: true}, nil
	}

	computeCalled := 0
	computeStrategy := func(candidateReq *launchRequest) (*placement.Strategy, error) {
		computeCalled++
		return &placement.Strategy{UBatchSize: 128}, nil
	}

	// A classified, non-recurring placement failure must not reach the helper:
	// the first compute failed with one class and the deterministic replan with a
	// DIFFERENT class, so sameCodeRecurred is false and the Layer-2 gate keeps it
	// with the controller.
	classified := errors.New("memory preflight failed: too tight")
	if got, err := escalatePlacementFailure(req, cfg, model, be, caps, "cuda_oom_after_deterministic_recovery", classified, computeStrategy); !errors.Is(err, classified) || got != nil {
		t.Fatalf("classified non-recurring escalation error=%v strategy=%+v, want original/nil", err, got)
	}
	if called != 0 {
		t.Fatalf("helper consulted %d times for a non-escalating placement failure", called)
	}
	if computeCalled != 0 {
		t.Fatalf("placement recomputed for a non-escalating placement failure")
	}

	// A novel placement failure escalates; the seam returns the bounded
	// remeasure action, which must enter the deterministic recompute path.
	novel := errors.New("some novel placement failure")
	got, err := escalatePlacementFailure(req, cfg, model, be, caps, "unclassified_launch_failure", novel, computeStrategy)
	if err != nil {
		t.Fatalf("escalated placement failure returned %v", err)
	}
	if got == nil || got.UBatchSize != 128 {
		t.Fatalf("escalated placement strategy=%+v, want recomputed strategy", got)
	}
	if called != 1 {
		t.Fatalf("helper consulted %d times, want 1", called)
	}
	if computeCalled != 1 {
		t.Fatalf("placement recomputed %d times, want 1", computeCalled)
	}
}

// --- STEP 8: force-online research on escalation + pre-flight taxonomy ---

// The pre-flight environment classes (port collision, permission boundary,
// missing cgroup containment, a flag the repair loop cannot remove) are
// deterministic user-action failures. No bounded placement reshape fixes them,
// so they must never escalate to the advisor — even with the retry budget
// exhausted and the same class recurring.
func TestShouldEscalateToAdvisorNeverEscalatesEnvironmentClasses(t *testing.T) {
	for _, code := range []string{
		"port_in_use", "permission_denied", "memory_cgroup_limit", "flag_rejected_no_fix",
	} {
		if shouldEscalateToAdvisor(code, escalationSignal{retriesExhausted: true, sameCodeRecurred: true}) {
			t.Fatalf("deterministic class %q escalated to the advisor", code)
		}
	}
}

// classifyAdvisorFailure must route container/port/permission/memory-cgroup
// rejects to deterministic classes BEFORE backend_start_failed (a bind failure
// surfaces wrapped inside "server not ready"), and a flag the repair loop
// cannot remove to flag_rejected_no_fix instead of the repairable
// backend_flag_rejected class.
func TestClassifyAdvisorFailurePreFlightProbe(t *testing.T) {
	cases := []struct {
		name  string
		cause error
		want  string
	}{
		{"bind address already in use", errors.New("server not ready: server process exited during startup: bind: address already in use"), "port_in_use"},
		{"port already in use guard", errors.New("port 8081 is already in use; choose a free --port"), "port_in_use"},
		{"cannot assign requested address", errors.New("server process exited during startup: listen tcp 127.0.0.1:8081: bind: cannot assign requested address"), "port_in_use"},
		{"permission denied", errors.New("start server: fork/exec /path/llama-server: permission denied"), "permission_denied"},
		{"operation not permitted", errors.New("start server: fork/exec /path/llama-server: operation not permitted"), "permission_denied"},
		{"missing systemd-run", errors.New("backend memory containment requires systemd-run: exec: \"systemd-run\": executable file not found"), "memory_cgroup_limit"},
		{"cgroup containment", errors.New("allocation probe needs Linux cgroup v2 containment: no such file or directory"), "memory_cgroup_limit"},
		{"repair loop stalled", errors.New("backend argument repair repeated without progress: backend /x rejected the generated launch command before model load: unknown argument: --swa-full"), "flag_rejected_no_fix"},
		{"user-explicit flag rejected", errors.New("backend rejected explicitly supplied --defrag-thold; refusing to change user input: backend /x rejected the generated launch command before model load: unknown argument"), "flag_rejected_no_fix"},
		{"repairable generated flag", errors.New("backend /x rejected the generated launch command before model load: unknown argument: --swa-full"), "backend_flag_rejected"},
		{"plain backend start", errors.New("server not ready: timeout waiting for server on port 8081"), "backend_start_failed"},
		{"cuda oom", errors.New("ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory"), "cuda_oom_after_deterministic_recovery"},
	}
	for _, tc := range cases {
		if got := classifyAdvisorFailure(tc.cause); got != tc.want {
			t.Fatalf("%s: classifyAdvisorFailure = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// A backendArgValidationError naming a NON-repairable flag must classify as
// flag_rejected_no_fix even though its message contains "unknown argument".
func TestClassifyAdvisorFailureStructuredNonRepairableFlag(t *testing.T) {
	err := &backendArgValidationError{
		Backend:    "/path/llama-server",
		Flag:       "--user-only-flag",
		Diagnostic: "unknown argument: --user-only-flag",
	}
	if got := classifyAdvisorFailure(err); got != "flag_rejected_no_fix" {
		t.Fatalf("non-repairable structured flag classified as %q, want flag_rejected_no_fix", got)
	}

	repairable := &backendArgValidationError{
		Backend:    "/path/llama-server",
		Flag:       "--swa-full",
		Diagnostic: "unknown argument: --swa-full",
	}
	if got := classifyAdvisorFailure(repairable); got != "backend_flag_rejected" {
		t.Fatalf("repairable structured flag classified as %q, want backend_flag_rejected", got)
	}
}

// maybeAnalyzeLaunchFailure forces online research on escalation unless the
// user explicitly named --no-support-online. The seam records the online flag.
func TestMaybeAnalyzeLaunchFailureForcesOnlineOnEscalation(t *testing.T) {
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	cfg.SupportExpert = "on"
	cfg.SupportOnline = false // config default: offline

	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	be := &backendInfo{Tag: "llama", Identity: "build"}

	original := runSupportIncidentFn
	defer func() { runSupportIncidentFn = original }()

	for _, tc := range []struct {
		name string
		req  *launchRequest
		want bool
	}{
		{"default forces online", &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1}, true},
		{"--support-online forces online", &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1, SupportOnline: true, SupportOnlineSet: true}, true},
		{"--no-support-online honored", &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1, SupportOnline: false, SupportOnlineSet: true}, false},
	} {
		got := ""
		runSupportIncidentFn = func(_ context.Context, _ *config.Config, _ *detect.Capabilities, _ string, _ advisor.Incident, online bool) (advisor.Decision, advisor.RunReport, error) {
			got = strconv.FormatBool(online)
			return advisor.Decision{
				Action: advisor.ActionNoAction, Confidence: 0.5,
				Rationale: "seam", EvidenceCodes: []string{"internal:seam"},
			}, advisor.RunReport{ReleaseVerified: true}, nil
		}
		_, escalated := maybeAnalyzeLaunchFailure(tc.req, cfg, model, be, caps, nil,
			errors.New("some novel failure"), "placement", escalationSignal{})
		if !escalated {
			t.Fatalf("%s: novel failure did not escalate", tc.name)
		}
		if got != strconv.FormatBool(tc.want) {
			t.Fatalf("%s: online=%s, want %t", tc.name, got, tc.want)
		}
	}
}

// A deterministic environment failure must not reach the helper even when it is
// the only failure observed: Layer 2 gates escalation, and no reshape fixes a
// port collision.
func TestMaybeAnalyzeLaunchFailureNeverConsultsForEnvironmentClass(t *testing.T) {
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	cfg.SupportExpert = "on"

	req := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1}
	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	be := &backendInfo{Tag: "llama", Identity: "build"}

	original := runSupportIncidentFn
	defer func() { runSupportIncidentFn = original }()
	called := 0
	runSupportIncidentFn = func(_ context.Context, _ *config.Config, _ *detect.Capabilities, _ string, _ advisor.Incident, _ bool) (advisor.Decision, advisor.RunReport, error) {
		called++
		return advisor.Decision{}, advisor.RunReport{}, errors.New("helper must never run for an environment class")
	}

	if _, escalated := maybeAnalyzeLaunchFailure(req, cfg, model, be, caps, nil,
		errors.New("server not ready: server process exited during startup: bind: address already in use"), "backend_start",
		escalationSignal{retriesExhausted: true, sameCodeRecurred: true}); escalated {
		t.Fatal("environment class escalated to the advisor")
	}
	if called != 0 {
		t.Fatalf("helper consulted %d times for an environment class", called)
	}
}

// A deterministic environment failure on the placement side must never escalate
// through escalatePlacementFailure either: the Layer-2 gate routes port,
// permission, cgroup, and un-repairable-flag classes to the controller with the
// ORIGINAL replan error and no helper process.
func TestEscalatePlacementFailureNeverConsultsForEnvironmentClass(t *testing.T) {
	t.Setenv("GGRUN_ADVISOR_CONSENT", "on") // consent would approve; the gate must still block
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	cfg.SupportExpert = "on"

	req := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1}
	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	be := &backendInfo{Tag: "llama", Identity: "build"}

	original := runSupportIncidentFn
	defer func() { runSupportIncidentFn = original }()
	called := 0
	runSupportIncidentFn = func(_ context.Context, _ *config.Config, _ *detect.Capabilities, _ string, _ advisor.Incident, _ bool) (advisor.Decision, advisor.RunReport, error) {
		called++
		return advisor.Decision{}, advisor.RunReport{}, errors.New("helper must never run for an environment class")
	}
	computeCalled := 0
	computeStrategy := func(candidateReq *launchRequest) (*placement.Strategy, error) {
		computeCalled++
		return &placement.Strategy{UBatchSize: 64}, nil
	}

	replanErr := errors.New("server not ready: server process exited during startup: bind: address already in use")
	got, err := escalatePlacementFailure(req, cfg, model, be, caps, "port_in_use", replanErr, computeStrategy)
	if !errors.Is(err, replanErr) {
		t.Fatalf("environment class error=%v, want original replan error", err)
	}
	if got != nil {
		t.Fatalf("environment class returned a strategy: %+v", got)
	}
	if called != 0 {
		t.Fatalf("helper consulted %d times for an environment class", called)
	}
	if computeCalled != 0 {
		t.Fatalf("placement recomputed for an environment class")
	}
}
