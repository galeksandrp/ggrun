package backendmetrics

import (
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Captured verbatim from a running llama.cpp fork serving a 118B MoE.
const realExposition = `# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 249473
# HELP llamacpp:prompt_seconds_total Prompt process time
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 1850.65
llamacpp:tokens_predicted_total 348
llamacpp:tokens_predicted_seconds_total 84.304
llamacpp:n_decode_total 953
llamacpp:n_tokens_max 126308
llamacpp:prompt_tokens_seconds 134.803
llamacpp:predicted_tokens_seconds 4.12792
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
llamacpp:n_busy_slots_per_decode 1
llamacpp:some_future_counter_we_do_not_know 42
`

func closeTo(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %.4f, want %.4f (+/- %g)", label, got, want, tol)
	}
}

func TestParseReadsTheBackendCounters(t *testing.T) {
	m := Parse(realExposition)
	closeTo(t, "PromptTokens", m.PromptTokens, 249473, 0)
	closeTo(t, "PromptSeconds", m.PromptSeconds, 1850.65, 0.001)
	closeTo(t, "PredictedTokens", m.PredictedTokens, 348, 0)
	closeTo(t, "PredictedSeconds", m.PredictedSeconds, 84.304, 0.001)
	closeTo(t, "DecodeCalls", m.DecodeCalls, 953, 0)
	closeTo(t, "BusySlotsPerCall", m.BusySlotsPerCall, 1, 0)
	if !m.Available() {
		t.Error("snapshot with timing reported as unavailable")
	}
}

// The rates the backend measured are the ones ggrun must report: a proxy timing
// whole requests once derived 0.26 tok/s while the backend reported 4.13.
func TestRatesMatchTheBackendsOwnFigures(t *testing.T) {
	m := Parse(realExposition)
	// The backend also publishes these directly, so they are a cross-check.
	closeTo(t, "prompt tok/s", m.PromptTokensPerSecond(), 134.803, 0.01)
	closeTo(t, "decode tok/s", m.DecodeTokensPerSecond(), 4.12792, 0.001)
}

func TestParseIgnoresCommentsUnknownCountersAndJunk(t *testing.T) {
	m := Parse(`# a comment
llamacpp:prompt_tokens_total 100
llamacpp:prompt_seconds_total 10
llamacpp:unknown_thing 999
not_a_metric_line
llamacpp:tokens_predicted_total notanumber

`)
	closeTo(t, "PromptTokens", m.PromptTokens, 100, 0)
	closeTo(t, "PromptSeconds", m.PromptSeconds, 10, 0)
	if m.PredictedTokens != 0 {
		t.Errorf("unparseable value was accepted: %v", m.PredictedTokens)
	}
}

// A backend without the metrics endpoint must report nothing rather than a
// measured-looking zero.
func TestUnavailableWhenNoTimingIsExposed(t *testing.T) {
	if Parse("").Available() {
		t.Error("empty body reported as available")
	}
	if Parse("llamacpp:requests_processing 3\n").Available() {
		t.Error("a snapshot with no timing reported as available")
	}
	if got := Parse("").DecodeTokensPerSecond(); got != 0 {
		t.Errorf("DecodeTokensPerSecond on empty = %v, want 0", got)
	}
}

// The decomposition that explains a RAM-resident MoE: a forward pass costs
// almost the same whether it carries one token or a whole micro-batch.
func TestPassCostSeparatesFixedFromMarginal(t *testing.T) {
	m := Parse(realExposition)
	pc, ok := PassCostFrom(m, 512)
	if !ok {
		t.Fatal("PassCostFrom refused a snapshot with both phases measured")
	}
	// prompt 7.418 ms/token * 512 = 3798 ms for a full pass; decode 242.3 ms
	// for a one-token pass; the difference spread over 511 tokens.
	closeTo(t, "marginal ms/token", pc.MarginalMS, 6.96, 0.1)
	closeTo(t, "fixed ms/pass", pc.FixedMS, 235.3, 1.0)
	if pc.FixedShare() < 0.90 {
		t.Errorf("fixed share = %.2f, want >0.90 for this machine", pc.FixedShare())
	}
	if pc.UBatch != 512 {
		t.Errorf("UBatch = %d, want 512", pc.UBatch)
	}
}

func TestPassCostRefusesWhatItCannotDerive(t *testing.T) {
	m := Parse(realExposition)
	// A micro-batch of one carries no information about the split.
	if _, ok := PassCostFrom(m, 1); ok {
		t.Error("derived a decomposition from ubatch=1")
	}
	// Only prefill measured.
	prefillOnly := Metrics{PromptTokens: 1000, PromptSeconds: 10}
	if _, ok := PassCostFrom(prefillOnly, 512); ok {
		t.Error("derived a decomposition with no decode measurement")
	}
	// No metrics at all.
	if _, ok := PassCostFrom(Metrics{}, 512); ok {
		t.Error("derived a decomposition from an empty snapshot")
	}
	// A backend where prefill does not amortise has no fixed cost to attribute;
	// the naive algebra would return a negative marginal.
	flat := Metrics{PromptTokens: 512, PromptSeconds: 512 * 0.5, PredictedTokens: 10, PredictedSeconds: 10 * 0.5}
	if _, ok := PassCostFrom(flat, 512); ok {
		t.Error("derived a decomposition where prefill does not amortise")
	}
}

func TestProjectionGrowsWithTokensPerPass(t *testing.T) {
	pc := PassCost{FixedMS: 235, MarginalMS: 6.94, UBatch: 512}
	one := pc.ProjectedTokensPerSecond(1)
	closeTo(t, "1 token/pass", one, 4.13, 0.05)
	eight := pc.ProjectedTokensPerSecond(8)
	if eight <= one {
		t.Fatalf("8 tokens/pass (%.2f) not faster than 1 (%.2f)", eight, one)
	}
	closeTo(t, "8 tokens/pass", eight, 27.5, 0.5)
	// Nonsense input must not produce a confident-looking number.
	if got := pc.ProjectedTokensPerSecond(0); got != 0 {
		t.Errorf("projection for 0 tokens = %v, want 0", got)
	}
}

func TestFetchReadsALiveEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(realExposition))
	}))
	defer srv.Close()

	m, err := Fetch(context.Background(), srv.Client(), srv.URL)
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	closeTo(t, "decode tok/s", m.DecodeTokensPerSecond(), 4.12792, 0.001)

	// A trailing slash on the base URL must not produce a double slash.
	if _, err := Fetch(context.Background(), srv.Client(), srv.URL+"/"); err != nil {
		t.Errorf("Fetch with trailing slash: %v", err)
	}
}

func TestFetchReportsABackendWithoutMetrics(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()
	if _, err := Fetch(context.Background(), srv.Client(), srv.URL); err == nil {
		t.Error("Fetch reported success against a backend with no metrics endpoint")
	}
}
