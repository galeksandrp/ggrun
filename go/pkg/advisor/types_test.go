package advisor

import (
	"strings"
	"testing"
)

func normalizedIncident(t *testing.T, mode Mode, actions ...ActionID) Incident {
	t.Helper()
	incident := Incident{
		ID: "incident-1", Mode: mode, Architecture: "deepseek4",
		BackendFamily: "llama", BackendIdentity: "build-abc", Workload: "claude-code",
		ProfileState: "load_healthy", AllowedActions: actions,
		Observations: []Observation{{Code: "cuda0_underfilled", Component: "placement", Source: "nvidia-smi", Confidence: "measured"}},
		Candidates:   []Candidate{{ID: "candidate-safe", Verified: true}},
	}
	if err := incident.Normalize(); err != nil {
		t.Fatal(err)
	}
	return incident
}

func TestDecodeDecisionRejectsUnknownCommandField(t *testing.T) {
	incident := normalizedIncident(t, ModeSupport, ActionNoAction)
	_, err := DecodeDecision([]byte(`{"schema_version":1,"action":"no_action","confidence":1,"rationale":"measured","command":"rm -rf /"}`), incident)
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("malicious command field was not rejected: %v", err)
	}
}

func TestDecodeDecisionRejectsUnofferedActionAndTrailingJSON(t *testing.T) {
	incident := normalizedIncident(t, ModeSupport, ActionNoAction)
	_, err := DecodeDecision([]byte(`{"schema_version":1,"action":"lower_ubatch","ubatch":128,"confidence":0.8,"rationale":"try it"}`), incident)
	if err == nil || !strings.Contains(err.Error(), "not offered") {
		t.Fatalf("unoffered action was not rejected: %v", err)
	}

	_, err = DecodeDecision([]byte(`{"schema_version":1,"action":"no_action","confidence":1,"rationale":"done"} {"action":"refresh_backend"}`), incident)
	if err == nil || !strings.Contains(err.Error(), "more than one JSON") {
		t.Fatalf("trailing JSON was not rejected: %v", err)
	}
}

func TestDecodeDecisionMapsSemanticConfidenceTokens(t *testing.T) {
	incident := normalizedIncident(t, ModeSupport, ActionNoAction)
	cases := []struct {
		token string
		want  float64
	}{
		{`"high"`, 0.9},
		{`"medium"`, 0.6},
		{`"low"`, 0.3},
	}
	for _, tc := range cases {
		input := `{"schema_version":1,"action":"no_action","confidence":` + tc.token + `,"rationale":"measured"}`
		decision, err := DecodeDecision([]byte(input), incident)
		if err != nil {
			t.Fatalf("confidence %s: unexpected error: %v", tc.token, err)
		}
		if float64(decision.Confidence) != tc.want {
			t.Fatalf("confidence %s: got %v, want %v", tc.token, decision.Confidence, tc.want)
		}
	}
	// The bare numeric form must keep working alongside the semantic tokens.
	decision, err := DecodeDecision([]byte(`{"schema_version":1,"action":"no_action","confidence":0.42,"rationale":"measured"}`), incident)
	if err != nil || float64(decision.Confidence) != 0.42 {
		t.Fatalf("bare numeric confidence regressed: decision=%v err=%v", decision, err)
	}
}

func TestDecodeDecisionDowngradesUnparsableConfidence(t *testing.T) {
	incident := normalizedIncident(t, ModeSupport, ActionNoAction)
	decision, err := DecodeDecision([]byte(`{"schema_version":1,"action":"no_action","confidence":"bogus","rationale":"measured"}`), incident)
	if err != nil {
		t.Fatalf("unparsable confidence must downgrade, not reject: %v", err)
	}
	if float64(decision.Confidence) != 0.5 {
		t.Fatalf("downgraded confidence: got %v, want 0.5", decision.Confidence)
	}
	found := false
	for _, code := range decision.EvidenceCodes {
		if strings.Contains(code, "confidence_unparsable") {
			found = true
		}
	}
	if !found {
		t.Fatalf("downgrade did not record the internal evidence marker: %#v", decision.EvidenceCodes)
	}
	// The downgraded decision must survive re-validation (e.g. SaveAnalysis).
	if err := ValidateDecision(incident, decision); err != nil {
		t.Fatalf("downgraded decision does not re-validate: %v", err)
	}
}

func TestDecodeDecisionAcceptsQuotedUBatch(t *testing.T) {
	incident := normalizedIncident(t, ModeSupport, ActionLowerUBatch)
	decision, err := DecodeDecision([]byte(`{"schema_version":1,"action":"lower_ubatch","ubatch":"128","confidence":0.8,"rationale":"shrink footprint"}`), incident)
	if err != nil {
		t.Fatalf("quoted ubatch must decode: %v", err)
	}
	if decision.UBatch != 128 {
		t.Fatalf("quoted ubatch: got %v, want 128", decision.UBatch)
	}
}

func TestDecodeDecisionRejectsSemanticTokenInLenientInt(t *testing.T) {
	incident := normalizedIncident(t, ModeSupport, ActionLowerUBatch)
	_, err := DecodeDecision([]byte(`{"schema_version":1,"action":"lower_ubatch","ubatch":"high","confidence":0.8,"rationale":"shrink footprint"}`), incident)
	if err == nil || !strings.Contains(err.Error(), "not a number") {
		t.Fatalf("lenientInt must stay strict against semantic tokens: %v", err)
	}
}

func TestOptimizerCanSelectOnlyControllerCandidate(t *testing.T) {
	incident := normalizedIncident(t, ModeOptimizer, ActionSelectCandidate)
	good := Decision{SchemaVersion: 1, Action: ActionSelectCandidate, CandidateID: "candidate-safe", Confidence: 0.7, Rationale: "best verified score"}
	if err := ValidateDecision(incident, good); err != nil {
		t.Fatalf("controller candidate rejected: %v", err)
	}
	bad := good
	bad.CandidateID = "--ctx-size=1"
	if err := ValidateDecision(incident, bad); err == nil {
		t.Fatal("model-selected arbitrary candidate was accepted")
	}
	incident.Candidates = append(incident.Candidates, Candidate{ID: "candidate-unverified"})
	unverified := good
	unverified.CandidateID = "candidate-unverified"
	if err := ValidateDecision(incident, unverified); err == nil {
		t.Fatal("generated but unverified candidate was accepted")
	}
}

// The propose_ubatch ladder is {256,128,64} and NEVER 512, because the action is
// only offered for a strategy sitting above the derate ladder: accepting 512
// would let the advisor propose a no-op (or worse, a raise). ValidateDecision
// must reject 512 and every out-of-ladder rung even when the action is offered.
func TestValidateDecisionProposeUBatchLadderExcludes512(t *testing.T) {
	incident := normalizedIncident(t, ModeSupport, ActionProposeUBatch)
	for _, rung := range []lenientInt{64, 128, 256} {
		decision := Decision{SchemaVersion: 1, Action: ActionProposeUBatch, UBatch: rung, Confidence: 0.8, Rationale: "derate"}
		if err := ValidateDecision(incident, decision); err != nil {
			t.Fatalf("propose_ubatch rung %d rejected: %v", rung, err)
		}
	}
	for _, rung := range []lenientInt{512, 32, 1024, 0} {
		decision := Decision{SchemaVersion: 1, Action: ActionProposeUBatch, UBatch: rung, Confidence: 0.8, Rationale: "derate"}
		if err := ValidateDecision(incident, decision); err == nil {
			t.Fatalf("propose_ubatch rung %d accepted, must be outside the {256,128,64} ladder", rung)
		}
	}
}

// propose_layer_distribution shares the move_expert_layer shape: a device target
// and a 1..2 layer count. Anything outside that typed envelope is rejected.
func TestValidateDecisionProposeLayerDistributionBounds(t *testing.T) {
	incident := normalizedIncident(t, ModeSupport, ActionProposeLayerDistribution)
	good := Decision{SchemaVersion: 1, Action: ActionProposeLayerDistribution, Device: 0, Count: 1, Confidence: 0.7, Rationale: "rebalance"}
	if err := ValidateDecision(incident, good); err != nil {
		t.Fatalf("valid layer distribution rejected: %v", err)
	}
	for _, bad := range []Decision{
		{SchemaVersion: 1, Action: ActionProposeLayerDistribution, Device: -1, Count: 1, Confidence: 0.7, Rationale: "rebalance"},
		{SchemaVersion: 1, Action: ActionProposeLayerDistribution, Device: 0, Count: 0, Confidence: 0.7, Rationale: "rebalance"},
		{SchemaVersion: 1, Action: ActionProposeLayerDistribution, Device: 0, Count: 3, Confidence: 0.7, Rationale: "rebalance"},
	} {
		if err := ValidateDecision(incident, bad); err == nil {
			t.Fatalf("invalid layer distribution accepted: %+v", bad)
		}
	}
}

// toggle_swa_full only ever targets the controller-owned swa-full feature. A
// decision that names a different feature (or none) is rejected.
func TestValidateDecisionToggleSWAFullTargetsSWAOnly(t *testing.T) {
	incident := normalizedIncident(t, ModeSupport, ActionToggleSWAFull)
	good := Decision{SchemaVersion: 1, Action: ActionToggleSWAFull, Feature: FeatureSWAFull, Confidence: 0.7, Rationale: "drop generated swa-full"}
	if err := ValidateDecision(incident, good); err != nil {
		t.Fatalf("valid swa-full toggle rejected: %v", err)
	}
	for _, feature := range []FeatureID{FeatureKHadamard, FeatureSpeculation, ""} {
		bad := Decision{SchemaVersion: 1, Action: ActionToggleSWAFull, Feature: feature, Confidence: 0.7, Rationale: "drop feature"}
		if err := ValidateDecision(incident, bad); err == nil {
			t.Fatalf("toggle_swa_full targeting %q was accepted", feature)
		}
	}
}

func TestIncidentNormalizeSanitizesTypedEvidence(t *testing.T) {
	incident := Incident{
		ID: "x\ncommand", Mode: ModeSupport,
		Hardware:       map[string]string{"gpu\n0": strings.Repeat("x", 500)},
		AllowedActions: []ActionID{ActionNoAction, "shell"},
		Observations:   []Observation{{Code: "oom\x00now", Component: "memory", Source: "probe", Confidence: "measured"}},
	}
	if err := incident.Normalize(); err != nil {
		t.Fatal(err)
	}
	if strings.ContainsAny(incident.ID, "\n\x00") || len(incident.Hardware["gpu 0"]) > 96 {
		t.Fatalf("incident was not sanitized: %#v", incident)
	}
	if len(incident.AllowedActions) != 1 || incident.AllowedActions[0] != ActionNoAction {
		t.Fatalf("invalid action survived normalization: %v", incident.AllowedActions)
	}
}
