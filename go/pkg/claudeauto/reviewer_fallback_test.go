package claudeauto

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// countingBackend is a stand-in backend that records how many routed requests
// actually reached it, so a test can prove where a review was sent rather than
// trusting the recorded route label alone.
func countingBackend(t *testing.T, hits *atomic.Int64) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[{"type":"text","text":"<block>no</block>"}],"usage":{"input_tokens":11,"output_tokens":2}}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// classifierBody builds a review request carrying the marker Claude Code uses,
// padded to an approximate prompt size so context-overflow can be exercised.
func classifierBody(padBytes int) string {
	return fmt.Sprintf(`{"model":"local","system":[{"type":"text","text":%q}],`+
		`"messages":[{"role":"user","content":%q}]}`,
		ClassifierMarker+" Review it.", strings.Repeat("x", padBytes))
}

func TestRouterSelfClassifiesWhenNoSeparateReviewer(t *testing.T) {
	var hits atomic.Int64
	backend := countingBackend(t, &hits)

	// Same backend for both roles: no separate reviewer was seated.
	router, path := newTestRouter(t, backend.URL, 1)
	post(t, router, classifierBody(16))

	records := waitForRecords(t, path, 1)
	if records[0].Route != routeMain {
		t.Errorf("route = %q, want %q (main model self-classifies)", records[0].Route, routeMain)
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("backend hits = %d, want 1", got)
	}
}

func TestRouterKeepsReviewerRouteWhenPromptFitsContext(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := countingBackend(t, &reviewerHits)

	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	router.SetReviewerContext(65536)
	post(t, router, classifierBody(64))

	records := waitForRecords(t, path, 1)
	if records[0].Route != routeReviewer {
		t.Errorf("route = %q, want %q", records[0].Route, routeReviewer)
	}
	if reviewerHits.Load() != 1 || mainHits.Load() != 0 {
		t.Errorf("reviewer hits = %d, main hits = %d; want 1 and 0",
			reviewerHits.Load(), mainHits.Load())
	}
}

func TestRouterSelfClassifiesWhenPromptExceedsReviewerContext(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := countingBackend(t, &reviewerHits)

	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	// A window far smaller than the padded prompt below, so the review cannot
	// fit in the reviewer and must fall back to the main model.
	router.SetReviewerContext(64)
	post(t, router, classifierBody(4096))

	records := waitForRecords(t, path, 1)
	if records[0].Route != routeMain {
		t.Errorf("route = %q, want %q (reviewer overflow)", records[0].Route, routeMain)
	}
	if mainHits.Load() != 1 || reviewerHits.Load() != 0 {
		t.Errorf("main hits = %d, reviewer hits = %d; want 1 and 0",
			mainHits.Load(), reviewerHits.Load())
	}
}

func TestRouterKeepsReviewerRouteWhenOverflowFallbackDisabled(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := countingBackend(t, &reviewerHits)

	// Never told the reviewer's window, so there is nothing to overflow against
	// and a large review must still go to the reviewer rather than silently
	// diverting to the main model.
	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	post(t, router, classifierBody(8192))

	records := waitForRecords(t, path, 1)
	if records[0].Route != routeReviewer {
		t.Errorf("route = %q, want %q with the overflow fallback disabled",
			records[0].Route, routeReviewer)
	}
	if reviewerHits.Load() != 1 {
		t.Errorf("reviewer hits = %d, want 1", reviewerHits.Load())
	}
}

// A self-classified review runs on the main backend, so it has to take a main
// admission slot; counting it as free would oversubscribe the main model.
func TestSelfClassifiedReviewTakesMainAdmissionSlot(t *testing.T) {
	var hits atomic.Int64
	backend := countingBackend(t, &hits)

	router, path := newTestRouter(t, backend.URL, 1)
	post(t, router, classifierBody(16))

	records := waitForRecords(t, path, 1)
	if records[0].Route != routeMain {
		t.Fatalf("route = %q, want %q", records[0].Route, routeMain)
	}
	if router.sched == nil {
		t.Fatal("main scheduler missing, so no slot could have been taken")
	}
	// The slot must be released again once the request completes, otherwise the
	// next main turn would queue behind a review that already finished.
	if active := router.mainActive.Load(); active != 0 {
		t.Errorf("main active = %d after the review completed, want 0", active)
	}
}

func TestEstimatedPromptTokensNeverUnderCountsAtFourBytesPerToken(t *testing.T) {
	for _, size := range []int{0, 1, 3, 100, 65536, 1 << 20} {
		got := estimatedPromptTokens(make([]byte, size))
		// Real tokenizers average well over 3 bytes per token, so a 3-byte
		// divisor must never come out below a 4-byte-per-token count; an
		// underestimate is what would send an oversized prompt to the reviewer.
		if want := size / 4; got < want {
			t.Errorf("estimatedPromptTokens(%d bytes) = %d, want >= %d", size, got, want)
		}
	}
}

// A seated reviewer that is down must not take the review with it: the router
// retries on the main model and the client sees a normal answer, never a 502.
func TestReviewerFailureFallsBackToMainModel(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	// A reviewer that is reachable but broken, the shape a backend takes when it
	// cannot serve the prompt it was given.
	reviewerBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reviewerHits.Add(1)
		http.Error(w, "context too small", http.StatusInternalServerError)
	}))
	t.Cleanup(reviewerBackend.Close)

	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json",
		strings.NewReader(classifierBody(16)))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	payload, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200; the reviewer's failure reached the client", resp.StatusCode)
	}
	if !strings.Contains(string(payload), "input_tokens") {
		t.Errorf("client got %q, want the main model's answer", string(payload))
	}
	// The attempt is withheld until it is known good, so the reviewer's error
	// body must never have reached the client on its way to being abandoned.
	if strings.Contains(string(payload), "context too small") {
		t.Errorf("the failed reviewer's error leaked to the client: %q", string(payload))
	}
	if reviewerHits.Load() != 1 {
		t.Errorf("reviewer hits = %d, want 1 (it must be tried first)", reviewerHits.Load())
	}
	if mainHits.Load() != 1 {
		t.Errorf("main hits = %d, want 1 (the fallback)", mainHits.Load())
	}
	records := waitForRecords(t, path, 2)
	last := records[len(records)-1]
	if last.Route != routeMain {
		t.Errorf("final route = %q, want %q for a fallback", last.Route, routeMain)
	}
	if records[0].Route != "reviewer-rejected/unusable-response" {
		t.Errorf("first route = %q, want the rejected reviewer attempt recorded", records[0].Route)
	}
}

// An unreachable reviewer is the other failure shape: nothing listening at all.
func TestUnreachableReviewerFallsBackToMainModel(t *testing.T) {
	var mainHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	dead := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	deadURL := dead.URL
	dead.Close() // nothing is listening on deadURL any more

	router, _ := newTestRouterPair(t, mainBackend.URL, deadURL, 1)
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json",
		strings.NewReader(classifierBody(16)))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200 via the main model", resp.StatusCode)
	}
	if mainHits.Load() != 1 {
		t.Errorf("main hits = %d, want 1", mainHits.Load())
	}
}

func TestReviewerClientErrorFallsBackToMainModel(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reviewerHits.Add(1)
		http.Error(w, "template rejected prompt", http.StatusBadRequest)
	}))
	t.Cleanup(reviewerBackend.Close)

	router, _ := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(classifierBody(16)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	payload, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(payload), "<block>no</block>") {
		t.Fatalf("client got status %d body %q, want main-model verdict", resp.StatusCode, string(payload))
	}
	if reviewerHits.Load() != 1 || mainHits.Load() != 1 {
		t.Fatalf("reviewer=%d main=%d; want one rejected attempt and one fallback", reviewerHits.Load(), mainHits.Load())
	}
}

func TestReviewerUnusableHTTP200FallsBackToMainModel(t *testing.T) {
	for _, tc := range []struct {
		name        string
		contentType string
		body        string
	}{
		{"empty-json", "application/json", `{}`},
		{"thinking-only", "application/json", `{"content":[{"type":"thinking","thinking":"looks safe"}]}`},
		{"tool-only", "application/json", `{"content":[{"type":"tool_use","name":"ggrun_canary_noop","input":{}}]}`},
		{"prose-only", "application/json", `{"content":[{"type":"text","text":"This looks safe."}]}`},
		{"ambiguous", "application/json", `{"content":[{"type":"text","text":"<block>no</block><block>yes</block>"}]}`},
		{"tool-only-stream", "text/event-stream", "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"name\":\"noop\"}}\n\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mainHits, reviewerHits atomic.Int64
			mainBackend := countingBackend(t, &mainHits)
			reviewerBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				reviewerHits.Add(1)
				w.Header().Set("Content-Type", tc.contentType)
				_, _ = w.Write([]byte(tc.body))
			}))
			t.Cleanup(reviewerBackend.Close)

			router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
			resp, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(classifierBody(16)))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			payload, _ := io.ReadAll(resp.Body)
			if !strings.Contains(string(payload), "<block>no</block>") {
				t.Fatalf("client got %q, want the main-model verdict", string(payload))
			}
			if reviewerHits.Load() != 1 || mainHits.Load() != 1 {
				t.Fatalf("reviewer=%d main=%d; want one unusable attempt and one fallback", reviewerHits.Load(), mainHits.Load())
			}
			records := waitForRecords(t, path, 2)
			if records[len(records)-1].Route != routeMain {
				t.Fatalf("final route = %q, want %q after fallback", records[len(records)-1].Route, routeMain)
			}
		})
	}
}

func TestReviewerValidStreamingVerdictAnswersItself(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reviewerHits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"<block>no\"}}\n\n"))
		_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"</block>\"}}\n\n"))
	}))
	t.Cleanup(reviewerBackend.Close)

	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(classifierBody(16)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)
	if reviewerHits.Load() != 1 || mainHits.Load() != 0 {
		t.Fatalf("reviewer=%d main=%d; want reviewer-only streaming verdict", reviewerHits.Load(), mainHits.Load())
	}
	if records := waitForRecords(t, path, 1); records[0].Route != routeReviewer {
		t.Fatalf("route = %q, want %q", records[0].Route, routeReviewer)
	}
}

// A healthy reviewer must still answer normally: the fallback must not divert
// working reviews to the main model.
func TestHealthyReviewerStillAnswersItself(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := countingBackend(t, &reviewerHits)

	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json",
		strings.NewReader(classifierBody(16)))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)
	// Headers are buffered with the body while the attempt is undecided; a
	// successful review has to replay them, not drop them.
	if got := resp.Header.Get("Content-Type"); got == "" {
		t.Error("withheld response headers were not replayed on success")
	}

	if reviewerHits.Load() != 1 || mainHits.Load() != 0 {
		t.Errorf("reviewer=%d main=%d; want 1 and 0", reviewerHits.Load(), mainHits.Load())
	}
	records := waitForRecords(t, path, 1)
	if records[0].Route != routeReviewer {
		t.Errorf("route = %q, want %q", records[0].Route, routeReviewer)
	}
}

// Cheap-tier work should prefer the companion, but companion failure must not
// fail the agent turn. Buffer the failed response and retry on main just like a
// classifier review, using the utility response contract rather than requiring
// a <block> verdict.
func TestUtilityTrafficFallsBackWhenReviewerFails(t *testing.T) {
	var mainHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "companion exploded", http.StatusInternalServerError)
	}))
	t.Cleanup(reviewerBackend.Close)

	router, _ := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	router.SetCompanion("local", true)

	resp, err := http.Post(router.URL()+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"local-fast","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want the successful main-model fallback", resp.StatusCode)
	}
	if got := mainHits.Load(); got != 1 {
		t.Errorf("utility failure reached the main model %d time(s); want 1", got)
	}
}

// A review-only reviewer is still a second backend. With no worker-serving
// companion, cheap-tier traffic must use it rather than queue behind long
// main-model streams on a single slot — that queueing is what timed out the
// permission classifier and blocked tool calls on the live 262k-ctx MoE.
func TestUtilityFallsThroughToSeatedReviewerWithoutCompanion(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := countingBackend(t, &reviewerHits)

	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	router.SetReviewerContext(65536)
	// No SetCompanion call: the profile is review-only, exactly like the live
	// Qwen3.5-2B reviewer deployment.
	post(t, router, `{"model":"local-fast","messages":[{"role":"user","content":"summarize"}]}`)

	records := waitForRecords(t, path, 1)
	if records[0].Route != routeUtility {
		t.Errorf("route = %q, want %q (seated reviewer serves the cheap tier)", records[0].Route, routeUtility)
	}
	if reviewerHits.Load() != 1 || mainHits.Load() != 0 {
		t.Errorf("reviewer hits = %d, main hits = %d; want 1 and 0",
			reviewerHits.Load(), mainHits.Load())
	}
}

// The utility lane keeps the same context-overflow guard as the classifier
// lane: a prompt too large for the reviewer cannot be made to fit by routing
// it there anyway, so it still goes to the main model.
func TestUtilityOverflowStillGoesToMainWithoutCompanion(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := countingBackend(t, &reviewerHits)

	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	router.SetReviewerContext(64)
	post(t, router, `{"model":"local-fast","messages":[{"role":"user","content":"`+strings.Repeat("x", 4096)+`"}]}`)

	records := waitForRecords(t, path, 1)
	if records[0].Route != routeMain {
		t.Errorf("route = %q, want %q (utility prompt overflows the reviewer)", records[0].Route, routeMain)
	}
	if mainHits.Load() != 1 || reviewerHits.Load() != 0 {
		t.Errorf("main hits = %d, reviewer hits = %d; want 1 and 0",
			mainHits.Load(), reviewerHits.Load())
	}
}

// A reviewer answer that does not carry a parseable verdict is recorded so the
// metrics log shows why the review leaked to the main model instead of leaving
// a template or format mismatch indistinguishable from reviewer unavailability.
func TestRejectedReviewerVerdictIsRecorded(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	// Answers, but with prose instead of the <block> contract.
	reviewerBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reviewerHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[{"type":"text","text":"I think this looks fine to me."}]}`))
	}))
	t.Cleanup(reviewerBackend.Close)

	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	router.SetReviewerContext(65536)
	post(t, router, classifierBody(64))

	records := waitForRecords(t, path, 2)
	if records[0].Route != routeMain && records[1].Route != routeMain {
		t.Errorf("neither record shows the main-model retry: %+v", records)
	}
	found := false
	for _, rec := range records {
		if strings.HasPrefix(rec.Route, "reviewer-rejected/") {
			found = true
			if !strings.HasSuffix(rec.Route, "invalid-verdict") {
				t.Errorf("rejection reason = %q, want invalid-verdict (the reviewer answered)", rec.Route)
			}
			if rec.Status != http.StatusOK {
				t.Errorf("rejected record status = %d, want the reviewer's 200", rec.Status)
			}
		}
	}
	if !found {
		t.Errorf("no reviewer-rejected record among: %+v", records)
	}
}

// The token estimate must err high: a 65,675-token review arrived in a body
// that the previous bytes/3 divisor scored below a 65,536 window and overflowed
// the reviewer with a 400. bytes/2 catches dense prompts at the cost of
// over-admitting mostly-ASCII prose, which the reviewer serves fine.
func TestEstimatedPromptTokensErrsHigh(t *testing.T) {
	if got := estimatedPromptTokens([]byte(strings.Repeat("a", 65536*2))); got < 65536 {
		t.Errorf("a %d-byte body estimated at %d tokens, want >= 65536", 65536*2, got)
	}
	if got := estimatedPromptTokens(nil); got != 0 {
		t.Errorf("empty body estimated at %d tokens, want 0", got)
	}
}
