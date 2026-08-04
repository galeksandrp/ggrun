// Package advisor implements ggrun's optional support expert. The model is a
// replaceable reasoning component inside a typed controller protocol: it never
// receives shell authority and its output cannot directly mutate a launch.
package advisor

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

const IncidentSchemaVersion = 1

type Mode string

const (
	ModeSupport   Mode = "support"
	ModeOptimizer Mode = "optimizer"
)

type ActionID string

const (
	ActionNoAction                ActionID = "no_action"
	ActionRemeasureAllocation     ActionID = "remeasure_allocation"
	ActionRemoveGeneratedFeature  ActionID = "remove_generated_feature"
	ActionSelectCompatibleBackend ActionID = "select_compatible_backend"
	ActionMoveExpertLayer         ActionID = "move_expert_layer"
	ActionLowerUBatch             ActionID = "lower_ubatch"
	ActionIncreaseCheckpoints     ActionID = "increase_checkpoints"
	ActionRefreshBackend          ActionID = "refresh_backend"
	ActionRunCacheCanary          ActionID = "run_cache_canary"
	ActionSelectCandidate         ActionID = "select_candidate"
	// ActionProposeUBatch asks the advisor to name a concrete micro-batch rung
	// from the deterministic derate ladder {256,128,64}. Unlike lower_ubatch it
	// can NEVER propose 512, so any accepted proposal is a genuine derate of a
	// strategy already sitting above the derate ladder's top rung.
	ActionProposeUBatch ActionID = "propose_ubatch"
	// ActionProposeLayerDistribution lets the advisor rebalance expert layers
	// across devices, expressed as a device target and a 1..2 layer count. It is
	// the general sibling of move_expert_layer (which is offered only for a
	// device-scoped OOM); both reach the packer as an AdvisorVRAMPenaltyMB
	// budget reduction, never as argv.
	ActionProposeLayerDistribution ActionID = "propose_layer_distribution"
	// ActionToggleSWAFull flips the controller-GENERATED swa-full feature in
	// either direction via setPassthroughBoolFlag. It is never offered for a
	// user-explicit --swa-full/--no-swa-full choice, so a toggle only ever
	// mutates a knob the controller itself materialized.
	ActionToggleSWAFull ActionID = "toggle_swa_full"
)

type FeatureID string

const (
	FeatureSWAFull     FeatureID = "swa-full"
	FeatureKHadamard   FeatureID = "k-hadamard"
	FeatureSpeculation FeatureID = "speculation"
)

type Observation struct {
	Code       string            `json:"code"`
	Component  string            `json:"component"`
	Device     int               `json:"device,omitempty"`
	Bytes      uint64            `json:"bytes,omitempty"`
	Value      float64           `json:"value,omitempty"`
	Unit       string            `json:"unit,omitempty"`
	Source     string            `json:"source"`
	Confidence string            `json:"confidence"`
	Attributes map[string]string `json:"attributes,omitempty"`
}

type Candidate struct {
	ID         string             `json:"id"`
	Properties map[string]string  `json:"properties,omitempty"`
	Metrics    map[string]float64 `json:"metrics,omitempty"`
	Verified   bool               `json:"verified,omitempty"`
}

type Source struct {
	Title       string    `json:"title"`
	URL         string    `json:"url"`
	RetrievedAt time.Time `json:"retrieved_at"`
	SHA256      string    `json:"sha256,omitempty"`
	// SearchPath records how this source was obtained: the search endpoint
	// queried (official issue index, model-card pin) or the local cache path the
	// material was written to. Purely for reproducibility/audit — it is not an
	// executable and never influences a launch action.
	SearchPath string `json:"search_path,omitempty"`
	Excerpt    string `json:"excerpt"`
}

type Incident struct {
	SchemaVersion   int               `json:"schema_version"`
	ID              string            `json:"id"`
	Mode            Mode              `json:"mode"`
	Architecture    string            `json:"architecture"`
	BackendFamily   string            `json:"backend_family"`
	BackendIdentity string            `json:"backend_identity"`
	Workload        string            `json:"workload"`
	ProfileState    string            `json:"profile_state"`
	Hardware        map[string]string `json:"hardware"`
	Settings        map[string]string `json:"settings"`
	Observations    []Observation     `json:"observations"`
	AllowedActions  []ActionID        `json:"allowed_actions"`
	Candidates      []Candidate       `json:"candidates,omitempty"`
	Knowledge       []Source          `json:"knowledge,omitempty"`
}

type Decision struct {
	SchemaVersion int        `json:"schema_version"`
	Action        ActionID   `json:"action"`
	Feature       FeatureID  `json:"feature,omitempty"`
	Device        lenientInt `json:"device,omitempty"`
	Count         lenientInt `json:"count,omitempty"`
	UBatch        lenientInt `json:"ubatch,omitempty"`
	Checkpoints   lenientInt `json:"checkpoints,omitempty"`
	BackendFamily string     `json:"backend_family,omitempty"`
	CandidateID   string     `json:"candidate_id,omitempty"`
	// Value carries a typed boolean target for controller-owned boolean knobs
	// (toggle_swa_full). It is a typed placement decision, never an argv string:
	// setPassthroughBoolFlag materializes it deterministically and only after
	// ValidateDecision. A false/omitted value means the conservative direction.
	Value         bool         `json:"value,omitempty"`
	Confidence    lenientFloat `json:"confidence"`
	Rationale     string       `json:"rationale"`
	EvidenceCodes []string     `json:"evidence_codes,omitempty"`
}

// internalEvidencePrefix marks controller-generated evidence notes. The support
// expert only ever cites observation codes that exist on the incident; ggrun
// itself writes internal notes (e.g. the confidence-downgrade marker) to explain
// a decision in the persisted record. These are exempt from the
// unknown-evidence check that guards against model-cited hallucinated codes.
// The prefix grants no authority: evidence codes are advisory and never gate a
// launch action (action/feature/ubatch/device are the authority fields).
const internalEvidencePrefix = "internal:"

// The support expert is a 3B model, not a JSON encoder. It reliably picks the
// right action and then writes "confidence": "0.9" instead of 0.9 — which cost
// a real incident on 2026-08-03T07:19:32Z, where a correct decision was thrown
// away with `cannot unmarshal string into field Decision.confidence`. Numbers
// are accepted quoted or bare; everything that decides behaviour (the action
// itself, the feature, the bounds) still goes through ValidateDecision
// unchanged, so leniency here widens parsing, never authority.
type lenientFloat float64

func (v *lenientFloat) UnmarshalJSON(data []byte) error {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "null" {
		return nil
	}
	unquoted := strings.Trim(trimmed, `"`)
	// A 3B reasoning model sometimes writes a semantic confidence token instead
	// of a number ("high", "medium", "low"). Map those to the same range the
	// deterministic controller expects, BEFORE the numeric parse. This widens
	// parsing only — ValidateDecision still bounds confidence to [0,1], and the
	// action/feature/ubatch fields (lenientInt) stay strict.
	switch strings.ToLower(strings.TrimSpace(unquoted)) {
	case "high":
		*v = lenientFloat(0.9)
		return nil
	case "medium":
		*v = lenientFloat(0.6)
		return nil
	case "low":
		*v = lenientFloat(0.3)
		return nil
	}
	parsed, err := strconv.ParseFloat(strings.TrimSpace(unquoted), 64)
	if err != nil {
		return fmt.Errorf("advisor confidence %s is not a number", trimmed)
	}
	*v = lenientFloat(parsed)
	return nil
}

type lenientInt int

func (v *lenientInt) UnmarshalJSON(data []byte) error {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "null" {
		return nil
	}
	unquoted := strings.Trim(trimmed, `"`)
	parsed, err := strconv.ParseFloat(strings.TrimSpace(unquoted), 64)
	if err != nil {
		return fmt.Errorf("advisor integer field %s is not a number", trimmed)
	}
	*v = lenientInt(int(parsed))
	return nil
}

func (incident *Incident) Normalize() error {
	if incident == nil {
		return errors.New("nil advisor incident")
	}
	incident.SchemaVersion = IncidentSchemaVersion
	if incident.Mode == "" {
		incident.Mode = ModeSupport
	}
	if incident.Mode != ModeSupport && incident.Mode != ModeOptimizer {
		return fmt.Errorf("invalid advisor mode %q", incident.Mode)
	}
	incident.ID = cleanToken(incident.ID, 96)
	incident.Architecture = cleanToken(incident.Architecture, 64)
	incident.BackendFamily = cleanToken(incident.BackendFamily, 64)
	incident.BackendIdentity = cleanToken(incident.BackendIdentity, 160)
	incident.Workload = cleanToken(incident.Workload, 96)
	incident.ProfileState = cleanToken(incident.ProfileState, 48)
	incident.Hardware = sanitizeStringMap(incident.Hardware, 24, 96)
	incident.Settings = sanitizeStringMap(incident.Settings, 32, 128)
	if len(incident.Observations) > 96 {
		incident.Observations = incident.Observations[:96]
	}
	for i := range incident.Observations {
		obs := &incident.Observations[i]
		obs.Code = cleanToken(obs.Code, 80)
		obs.Component = cleanToken(obs.Component, 64)
		obs.Source = cleanToken(obs.Source, 64)
		obs.Confidence = cleanToken(obs.Confidence, 24)
		obs.Unit = cleanToken(obs.Unit, 24)
		obs.Attributes = sanitizeStringMap(obs.Attributes, 16, 160)
	}
	allowed := map[ActionID]bool{}
	for _, action := range incident.AllowedActions {
		if validAction(action) {
			allowed[action] = true
		}
	}
	incident.AllowedActions = incident.AllowedActions[:0]
	for action := range allowed {
		incident.AllowedActions = append(incident.AllowedActions, action)
	}
	sort.Slice(incident.AllowedActions, func(i, j int) bool { return incident.AllowedActions[i] < incident.AllowedActions[j] })
	if len(incident.AllowedActions) == 0 {
		incident.AllowedActions = []ActionID{ActionNoAction}
	}
	if len(incident.Candidates) > 24 {
		incident.Candidates = incident.Candidates[:24]
	}
	for i := range incident.Candidates {
		incident.Candidates[i].ID = cleanToken(incident.Candidates[i].ID, 96)
		incident.Candidates[i].Properties = sanitizeStringMap(incident.Candidates[i].Properties, 24, 128)
	}
	return nil
}

func ValidateDecision(incident Incident, decision Decision) error {
	if decision.SchemaVersion != IncidentSchemaVersion {
		return fmt.Errorf("advisor decision schema %d is unsupported", decision.SchemaVersion)
	}
	if !validAction(decision.Action) {
		return fmt.Errorf("advisor action %q is not recognized", decision.Action)
	}
	allowed := false
	for _, action := range incident.AllowedActions {
		if action == decision.Action {
			allowed = true
			break
		}
	}
	if !allowed {
		return fmt.Errorf("advisor action %q was not offered by the deterministic controller", decision.Action)
	}
	confidence := float64(decision.Confidence)
	if math.IsNaN(confidence) || math.IsInf(confidence, 0) || confidence < 0 || confidence > 1 {
		return errors.New("advisor confidence must be between zero and one")
	}
	if len(strings.TrimSpace(decision.Rationale)) == 0 || len(decision.Rationale) > 1200 {
		return errors.New("advisor rationale is empty or too long")
	}
	switch decision.Action {
	case ActionNoAction, ActionRemeasureAllocation, ActionRefreshBackend, ActionRunCacheCanary:
	case ActionRemoveGeneratedFeature:
		if decision.Feature != FeatureSWAFull && decision.Feature != FeatureKHadamard && decision.Feature != FeatureSpeculation {
			return fmt.Errorf("feature %q is not controller-owned", decision.Feature)
		}
	case ActionSelectCompatibleBackend:
		if decision.BackendFamily != "llama" && decision.BackendFamily != "ik_llama" && decision.BackendFamily != "registered" {
			return errors.New("invalid backend family")
		}
	case ActionMoveExpertLayer, ActionProposeLayerDistribution:
		if decision.Device < 0 || decision.Count < 1 || decision.Count > 2 {
			return errors.New("layer distribution must target a device and one or two layers")
		}
	case ActionLowerUBatch:
		if decision.UBatch != 64 && decision.UBatch != 128 && decision.UBatch != 256 && decision.UBatch != 512 {
			return errors.New("advisor ubatch is outside the deterministic ladder")
		}
	case ActionProposeUBatch:
		// The proposal ladder is {256,128,64} and NEVER 512: a propose_ubatch
		// decision is only offered for a strategy sitting above the derate ladder,
		// so every accepted value is a genuine micro-batch derate.
		if decision.UBatch != 64 && decision.UBatch != 128 && decision.UBatch != 256 {
			return errors.New("advisor ubatch proposal is outside the derate ladder {256,128,64}")
		}
	case ActionToggleSWAFull:
		if decision.Feature != FeatureSWAFull {
			return fmt.Errorf("toggle_swa_full targets feature %q, only swa-full is toggleable", decision.Feature)
		}
	case ActionIncreaseCheckpoints:
		if decision.Checkpoints < 4 || decision.Checkpoints > 16 {
			return errors.New("advisor checkpoints must be in the controller range 4..16")
		}
	case ActionSelectCandidate:
		found := false
		for _, candidate := range incident.Candidates {
			if candidate.ID == decision.CandidateID {
				found = candidate.Verified
				break
			}
		}
		if !found || decision.CandidateID == "" {
			return errors.New("advisor selected a candidate that was not measured and verified by ggrun")
		}
	}
	knownEvidence := map[string]bool{}
	for _, obs := range incident.Observations {
		knownEvidence[obs.Code] = true
	}
	for _, code := range decision.EvidenceCodes {
		if !knownEvidence[code] && !strings.HasPrefix(code, internalEvidencePrefix) {
			return fmt.Errorf("advisor cited unknown evidence code %q", code)
		}
	}
	return nil
}

func DecodeDecision(data []byte, incident Incident) (Decision, error) {
	var decision Decision
	trimmed := strings.TrimSpace(string(data))
	trimmed = strings.TrimPrefix(trimmed, "```json")
	trimmed = strings.TrimPrefix(trimmed, "```")
	trimmed = strings.TrimSuffix(trimmed, "```")
	decoder := json.NewDecoder(strings.NewReader(strings.TrimSpace(trimmed)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decision); err != nil {
		// Confidence is advisory — a model that cannot produce a numeric
		// confidence for an otherwise-valid decision must not throw the whole
		// run away (the action/feature/ubatch fields are the ones that gate
		// authority and they stay strict). Downgrade ONLY a confidence decode
		// failure to a default and continue.
		if strings.Contains(err.Error(), "confidence") {
			// Decode again with confidence treated as an unknown field by
			// stripping it from the raw JSON, then validate with the default.
			if d2, ok := decodeDecisionWithoutConfidence(trimmed); ok {
				decision = d2
				decision.Confidence = 0.5
				decision.EvidenceCodes = append(decision.EvidenceCodes,
					internalEvidencePrefix+"confidence_unparsable: treated as 0.5")
				return decision, ValidateDecision(incident, decision)
			}
		}
		return Decision{}, fmt.Errorf("decode advisor decision: %w", err)
	}
	var trailing interface{}
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return Decision{}, errors.New("advisor decision contains more than one JSON value")
		}
		return Decision{}, fmt.Errorf("decode trailing advisor output: %w", err)
	}
	if err := ValidateDecision(incident, decision); err != nil {
		return Decision{}, err
	}
	return decision, nil
}

// decodeDecisionWithoutConfidence re-decodes advisor output with the confidence
// field removed, so a model that emits an unparseable confidence token can still
// yield its (valid) action. Only used on the confidence-downgrade path; every
// other field is decoded strictly. The caller sets a 0.5 confidence and an
// internal evidence marker after re-decoding.
func decodeDecisionWithoutConfidence(raw string) (Decision, bool) {
	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &rawMap); err != nil {
		return Decision{}, false
	}
	delete(rawMap, "confidence")
	stripped, err := json.Marshal(rawMap)
	if err != nil {
		return Decision{}, false
	}
	var d Decision
	dec := json.NewDecoder(bytes.NewReader(stripped))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&d); err != nil {
		return Decision{}, false
	}
	return d, true
}

func validAction(action ActionID) bool {
	switch action {
	case ActionNoAction, ActionRemeasureAllocation, ActionRemoveGeneratedFeature,
		ActionSelectCompatibleBackend, ActionMoveExpertLayer, ActionLowerUBatch,
		ActionIncreaseCheckpoints, ActionRefreshBackend, ActionRunCacheCanary,
		ActionSelectCandidate, ActionProposeUBatch, ActionProposeLayerDistribution,
		ActionToggleSWAFull:
		return true
	default:
		return false
	}
}

func cleanToken(value string, limit int) string {
	value = strings.Join(strings.Fields(strings.Map(func(r rune) rune {
		if r < 32 || r == 127 {
			return -1
		}
		return r
	}, value)), " ")
	if len(value) > limit {
		value = value[:limit]
	}
	return value
}

func sanitizeStringMap(input map[string]string, maxItems, maxValue int) map[string]string {
	if len(input) == 0 {
		return nil
	}
	keys := make([]string, 0, len(input))
	for key := range input {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	if len(keys) > maxItems {
		keys = keys[:maxItems]
	}
	out := make(map[string]string, len(keys))
	for _, key := range keys {
		out[cleanToken(key, 64)] = cleanToken(input[key], maxValue)
	}
	return out
}
