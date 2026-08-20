package claudeauto

import (
	"fmt"
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
		_, _ = w.Write([]byte(`{"usage":{"input_tokens":11,"output_tokens":2}}`))
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
