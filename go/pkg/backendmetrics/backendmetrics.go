// Package backendmetrics reads llama.cpp's own timing counters.
//
// A proxy in front of the backend can only time whole HTTP requests, and that
// is not the same thing as time spent generating. A cancelled request, a client
// that stops reading, or a connection held open after the last token all
// inflate the wall-clock measurement: ggrun's router once derived 0.26 tok/s
// from request timings while the backend reported 4.13 for the same traffic.
//
// The backend counts only cycles it actually spent, so its counters are the
// authority for throughput. Everything here is derived from them.
package backendmetrics

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Metrics is llama.cpp's /metrics snapshot, in the fields ggrun uses.
//
// Counters are cumulative since the server started, so a rate computed from a
// single snapshot describes the whole run. Two snapshots give an interval rate.
type Metrics struct {
	PromptTokens      float64 // tokens evaluated during prefill
	PromptSeconds     float64 // seconds spent in prefill
	PredictedTokens   float64 // tokens generated
	PredictedSeconds  float64 // seconds spent generating
	DecodeCalls       float64 // llama_decode calls: prefill micro-batches plus decode steps
	BusySlotsPerCall  float64 // average slots active per decode call
	RequestsRunning   float64
	RequestsDeferred  float64
	MaxTokensObserved float64
}

// counterFields maps the exported metric name onto the destination field.
var counterFields = map[string]func(*Metrics, float64){
	"llamacpp:prompt_tokens_total":            func(m *Metrics, v float64) { m.PromptTokens = v },
	"llamacpp:prompt_seconds_total":           func(m *Metrics, v float64) { m.PromptSeconds = v },
	"llamacpp:tokens_predicted_total":         func(m *Metrics, v float64) { m.PredictedTokens = v },
	"llamacpp:tokens_predicted_seconds_total": func(m *Metrics, v float64) { m.PredictedSeconds = v },
	"llamacpp:n_decode_total":                 func(m *Metrics, v float64) { m.DecodeCalls = v },
	"llamacpp:n_busy_slots_per_decode":        func(m *Metrics, v float64) { m.BusySlotsPerCall = v },
	"llamacpp:requests_processing":            func(m *Metrics, v float64) { m.RequestsRunning = v },
	"llamacpp:requests_deferred":              func(m *Metrics, v float64) { m.RequestsDeferred = v },
	"llamacpp:n_tokens_max":                   func(m *Metrics, v float64) { m.MaxTokensObserved = v },
}

// Parse reads the Prometheus-style exposition llama.cpp serves. Unknown metrics
// and comment lines are ignored so a backend that adds or drops counters does
// not break the parse.
func Parse(body string) Metrics {
	var m Metrics
	scanner := bufio.NewScanner(strings.NewReader(body))
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name, value, ok := strings.Cut(line, " ")
		if !ok {
			continue
		}
		set, known := counterFields[name]
		if !known {
			continue
		}
		v, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
		if err != nil {
			continue
		}
		set(&m, v)
	}
	return m
}

// Fetch reads /metrics from a backend base URL.
func Fetch(ctx context.Context, client *http.Client, baseURL string) (Metrics, error) {
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Second}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/metrics", nil)
	if err != nil {
		return Metrics{}, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return Metrics{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Metrics{}, fmt.Errorf("backend metrics: HTTP %s", resp.Status)
	}
	// A backend built without metrics support answers with something small and
	// unparseable rather than an error, so cap the read and let Parse decide.
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return Metrics{}, err
	}
	return Parse(string(body)), nil
}

// Available reports whether the snapshot carries usable timing. A backend
// without the metrics endpoint enabled yields zeros, and reporting a rate of
// zero as though it were measured would be worse than reporting nothing.
func (m Metrics) Available() bool {
	return m.PromptSeconds > 0 || m.PredictedSeconds > 0
}

// PromptTokensPerSecond is prefill throughput as the backend measured it.
func (m Metrics) PromptTokensPerSecond() float64 {
	if m.PromptSeconds <= 0 {
		return 0
	}
	return m.PromptTokens / m.PromptSeconds
}

// DecodeTokensPerSecond is generation throughput as the backend measured it.
func (m Metrics) DecodeTokensPerSecond() float64 {
	if m.PredictedSeconds <= 0 {
		return 0
	}
	return m.PredictedTokens / m.PredictedSeconds
}

// PassCost separates what a forward pass costs before it processes anything
// from what each additional token in that pass costs.
//
// This is the measurement that explains a RAM-resident MoE. Decode puts one
// token through a pass; prefill puts a whole micro-batch through the same pass.
// On a machine whose expert weights live in system RAM, the pass has to touch
// those weights either way, so the fixed part dominates decode and vanishes
// into prefill. Observed on a 118B MoE: 235 ms fixed against 6.94 ms per token,
// which is 97% of a decode step spent before the token is even considered.
//
// It also tells the user whether speculative decoding is worth trying, since
// speculation's entire value is putting more tokens through each pass.
type PassCost struct {
	FixedMS    float64 // cost of a pass carrying no tokens
	MarginalMS float64 // additional cost per token in the pass
	UBatch     int     // micro-batch the prefill figure was measured at
}

// PassCostFrom derives the decomposition from a snapshot and the configured
// micro-batch. It returns false when the snapshot cannot support it: both
// phases must have run, and prefill must genuinely amortise (a micro-batch of
// one carries no information about the split).
func PassCostFrom(m Metrics, ubatch int) (PassCost, bool) {
	if ubatch < 2 || !m.Available() {
		return PassCost{}, false
	}
	promptPerToken := 0.0
	if m.PromptTokens > 0 {
		promptPerToken = m.PromptSeconds * 1000 / m.PromptTokens
	}
	decodePerToken := 0.0
	if m.PredictedTokens > 0 {
		decodePerToken = m.PredictedSeconds * 1000 / m.PredictedTokens
	}
	if promptPerToken <= 0 || decodePerToken <= 0 {
		return PassCost{}, false
	}
	costBatch := promptPerToken * float64(ubatch) // one full micro-batch pass
	costOne := decodePerToken                     // one single-token pass
	if costBatch <= costOne {
		// Prefill is not amortising, so there is no fixed cost to attribute and
		// the model below would produce a negative marginal.
		return PassCost{}, false
	}
	marginal := (costBatch - costOne) / float64(ubatch-1)
	fixed := costOne - marginal
	if fixed <= 0 {
		return PassCost{}, false
	}
	return PassCost{FixedMS: fixed, MarginalMS: marginal, UBatch: ubatch}, true
}

// FixedShare is the fraction of a single-token decode step spent before the
// token is processed.
func (p PassCost) FixedShare() float64 {
	step := p.FixedMS + p.MarginalMS
	if step <= 0 {
		return 0
	}
	return p.FixedMS / step
}

// ProjectedTokensPerSecond is the rate if n tokens went through each pass.
//
// It is an upper bound, and on a mixture-of-experts model a loose one: more
// tokens per pass route to more distinct experts, so the "fixed" cost grows
// with token diversity rather than staying flat. Tokens continuing one sequence
// should share more experts than tokens from unrelated requests, but by how
// much is a property of the model and has to be measured, not assumed.
func (p PassCost) ProjectedTokensPerSecond(n int) float64 {
	if n < 1 {
		return 0
	}
	cost := p.FixedMS + p.MarginalMS*float64(n)
	if cost <= 0 {
		return 0
	}
	return float64(n) / (cost / 1000)
}
