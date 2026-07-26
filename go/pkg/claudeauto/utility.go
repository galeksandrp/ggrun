package claudeauto

import (
	"bytes"
	"encoding/json"
	"strings"
)

// Claude Code chooses a model tier per call: a cheap tier for background and
// mechanical work, the main tier for reasoning. ggrun pointed all five tiers at
// one alias so that no request could leave for the vendor API, which is right,
// but it also collapsed that choice -- every background summary and classifier
// call landed on the main model.
//
// Serving a second alias from the companion backend restores the distinction
// without ggrun guessing anything about a request: the harness has already
// decided which tier the work belongs to, and the alias carries that decision.
//
// This is deliberately independent of any particular model, backend fork or
// machine. The companion is whatever GGRUN_CLAUDE_REVIEWER_MODEL names, and the
// routing is HTTP-level, so any llama.cpp-compatible server can serve it.

// UtilityAlias is the model name Claude Code's cheap tiers are pointed at. It
// is a routing label, not a model: what actually answers is the companion
// backend ggrun already runs.
const UtilityAlias = "local-fast"

// routeUtility labels the lane a request should take.
const routeUtility = "utility"

// requestModel returns the model field of an Anthropic request body.
func requestModel(body []byte) string {
	var request struct {
		Model string `json:"model"`
	}
	if json.Unmarshal(body, &request) != nil {
		return ""
	}
	return strings.TrimSpace(request.Model)
}

// IsUtilityRequest reports whether Claude Code addressed the cheap tier.
//
// This is an exact test on a field the harness set deliberately, not an
// inference from request shape. A heuristic that guessed "this looks
// mechanical" would eventually misroute a user's foreground turn to a smaller
// model, which is a correctness regression disguised as an optimisation.
func IsUtilityRequest(body []byte) bool {
	return strings.EqualFold(requestModel(body), UtilityAlias)
}

// retargetModel rewrites the model field so the companion backend sees the
// alias it was launched with. The backend is not required to know about
// ggrun's routing labels.
func retargetModel(body []byte, alias string) []byte {
	if alias == "" {
		return body
	}
	var payload map[string]any
	if json.Unmarshal(body, &payload) != nil {
		return body
	}
	if _, ok := payload["model"]; !ok {
		return body
	}
	payload["model"] = alias
	out, err := json.Marshal(payload)
	if err != nil {
		return body
	}
	return out
}

// utilityBody prepares a cheap-tier request for the companion backend.
func utilityBody(body []byte, backendAlias string) []byte {
	return retargetModel(body, backendAlias)
}

// bodyReader is a small helper so callers can reset a request body after the
// model field is rewritten.
func bodyReader(body []byte) *bytes.Reader { return bytes.NewReader(body) }
