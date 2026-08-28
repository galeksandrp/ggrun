package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type testProgressLog string

func (l testProgressLog) Tail(int) string { return string(l) }

func TestParsePromptLogProgressKeepsLatestPerTask(t *testing.T) {
	log := `
slot print_timing: id  0 | task 4 | prompt processing, n_tokens =  2049, progress = 0.03, t = 125.00 s / 16.39 tokens per second
slot print_timing: id  1 | task 7 | prompt processing, n_tokens =  1025, progress = 0.25, t = 50.00 s / 20.50 tokens per second
slot print_timing: id  0 | task 4 | prompt processing, n_tokens = 32769, progress = 0.55, t = 1113.88 s / 29.42 tokens per second
`
	items := parsePromptLogProgress(log)
	if len(items) != 2 {
		t.Fatalf("got %d tasks, want 2: %+v", len(items), items)
	}
	var main promptLogProgress
	for _, item := range items {
		if item.Task == 4 {
			main = item
		}
	}
	if main.Processed != 32769 || main.Fraction != 0.55 || main.Rate != 29.42 {
		t.Fatalf("latest task progress not retained: %+v", main)
	}
}

func TestParseClaudeLogSnapshotTracksLifecycle(t *testing.T) {
	log := `
slot launch_slot_: id  3 | task 196 | processing task, is_child = 0
slot print_timing: id  3 | task 196 | prompt processing, n_tokens =   6144, progress = 0.16, t = 235.14 s / 26.13 tokens per second
slot      release: id  3 | task 196 | stop processing: n_tokens = 8192, truncated = 0
slot launch_slot_: id  3 | task 401 | processing task, is_child = 0
`
	snapshot := parseClaudeLogSnapshot(log)
	if len(snapshot.Active) != 1 || snapshot.Active[401].Stage != "working" || !snapshot.Released[196] {
		t.Fatalf("unexpected lifecycle snapshot: %+v", snapshot)
	}
	if snapshot.TotalSlots != 4 {
		t.Fatalf("total slots=%d, want 4", snapshot.TotalSlots)
	}
}

func TestPollAndFormatClaudeProgress(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/slots", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[
  {"id":0,"id_task":4,"is_processing":true,"n_ctx":262144,"n_prompt_tokens":32769,"n_prompt_tokens_processed":32769,"next_token":{"has_next_token":false,"n_remain":16,"n_decoded":0}},
  {"id":1,"is_processing":false,"n_ctx":262144}
]`))
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("llamacpp:prompt_tokens_seconds 29.42\nllamacpp:predicted_tokens_seconds 5.88\nllamacpp:requests_deferred 3\n"))
	})
	mux.HandleFunc("/ggrun/router", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"active":1,"queued":2,"limit":1}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	log := testProgressLog("slot print_timing: id  0 | task 4 | prompt processing, n_tokens = 32769, progress = 0.55, t = 1113.88 s / 29.42 tokens per second\n")
	state := pollClaudeProgress(srv.Client(), u.Hostname(), port, log)
	if state.Active != 1 || state.Queued != 5 || state.TotalSlots != 2 || len(state.Requests) != 1 {
		t.Fatalf("unexpected state: %+v", state)
	}
	got := formatClaudeProgress(state)
	for _, want := range []string{"██████░░░░", "55%", "S0 prefill 32,769/~59,580", "29.4 tok/s", "5 queued"} {
		if !strings.Contains(got, want) {
			t.Fatalf("status %q missing %q", got, want)
		}
	}
}

func TestFormatClaudeProgressGenerationAndReady(t *testing.T) {
	generated := formatClaudeProgress(claudeProgressState{
		TotalSlots: 4,
		Active:     2,
		Queued:     1,
		Requests: []claudeRequestProgress{{
			Slot: 2, Stage: "generating", Generated: 17, TokensPerSecond: 5.88, ElapsedSeconds: 134,
		}},
	})
	for _, want := range []string{"S2 generating 17 tok", "5.9 tok/s", "2m14s", "2 active", "1 queued"} {
		if !strings.Contains(generated, want) {
			t.Fatalf("generation status %q missing %q", generated, want)
		}
	}
	if got := formatClaudeProgress(claudeProgressState{TotalSlots: 4}); got != "ggrun · local ready · 4 slots" {
		t.Fatalf("ready status = %q", got)
	}
	if got := formatClaudeProgress(claudeProgressState{Event: "request completed"}); got != "ggrun · request completed" {
		t.Fatalf("completion status = %q", got)
	}
}

func TestFormatClaudeProgressSplitRates(t *testing.T) {
	// Prefill-only: only the prompt-eval rate is known.
	prefill := formatClaudeProgress(claudeProgressState{
		TotalSlots: 4,
		Active:     1,
		Requests: []claudeRequestProgress{{
			Slot: 1, Stage: "prefill", PromptFraction: 0.5, PromptProcessed: 2049,
			PrefillTPS: 29.42,
		}},
	})
	for _, want := range []string{"S1 prefill 2,049", "29.4 tok/s"} {
		if !strings.Contains(prefill, want) {
			t.Fatalf("prefill status %q missing %q", prefill, want)
		}
	}
	if strings.Contains(prefill, "decode") {
		t.Fatalf("prefill-only status %q must not mention decode", prefill)
	}

	// Decode-only: only the eval rate is known.
	decode := formatClaudeProgress(claudeProgressState{
		TotalSlots: 4,
		Active:     1,
		Requests: []claudeRequestProgress{{
			Slot: 2, Stage: "generating", Generated: 17, DecodeTPS: 5.88,
		}},
	})
	for _, want := range []string{"S2 generating 17 tok", "5.9 tok/s"} {
		if !strings.Contains(decode, want) {
			t.Fatalf("decode status %q missing %q", decode, want)
		}
	}
	if strings.Contains(decode, "prefill") {
		t.Fatalf("decode-only status %q must not mention prefill", decode)
	}

	// Both rates known over the sample window: render a split two-part line.
	both := formatClaudeProgress(claudeProgressState{
		TotalSlots: 4,
		Active:     1,
		Requests: []claudeRequestProgress{{
			Slot: 3, Stage: "generating", Generated: 512, PrefillTPS: 29.42, DecodeTPS: 5.88,
		}},
	})
	for _, want := range []string{"prefill 29.4 tok/s · decode 5.9 tok/s", "S3 generating 512 tok"} {
		if !strings.Contains(both, want) {
			t.Fatalf("both-rates status %q missing %q", both, want)
		}
	}

	// Backward compatibility: a struct carrying only the blended rate still
	// renders the legacy single-rate line.
	legacy := formatClaudeProgress(claudeProgressState{
		TotalSlots: 4,
		Active:     1,
		Requests: []claudeRequestProgress{{
			Slot: 0, Stage: "generating", Generated: 3, TokensPerSecond: 7.5,
		}},
	})
	if !strings.Contains(legacy, "7.5 tok/s") {
		t.Fatalf("legacy blended-rate status %q missing %q", legacy, "7.5 tok/s")
	}
}

func TestClaudeProgressFallsBackToPassiveLogsWhenSlotsBusy(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/slots", func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(100 * time.Millisecond)
		_, _ = w.Write([]byte(`[]`))
	})
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Timeout: 20 * time.Millisecond}
	log := testProgressLog(`
slot launch_slot_: id  3 | task 401 | processing task, is_child = 0
slot print_timing: id  3 | task 401 | prompt processing, n_tokens =   6144, progress = 0.16, t = 235.14 s / 26.13 tokens per second
`)
	state, structuredOK := pollClaudeProgressResilient(client, u.Hostname(), port, log, claudeProgressState{TotalSlots: 4}, true)
	if structuredOK || state.Error != "" || !state.StatusDelayed || state.Active != 1 {
		t.Fatalf("expected healthy passive fallback, got ok=%v state=%+v", structuredOK, state)
	}
	status := formatClaudeProgress(state)
	for _, want := range []string{"16%", "S3 prefill 6,144/~38,400", "26.1 tok/s", "log estimate"} {
		if !strings.Contains(status, want) {
			t.Fatalf("passive status %q missing %q", status, want)
		}
	}
}

func TestClaudeProgressPassiveFallbackPreservesLastKnownRequest(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	previous := claudeProgressState{
		TotalSlots: 4,
		Active:     1,
		Requests: []claudeRequestProgress{{
			Slot: 1, Task: 77, Stage: "generating", Generated: 23, TokensPerSecond: 5.5,
		}},
	}
	state, structuredOK := pollClaudeProgressResilient(srv.Client(), u.Hostname(), port, testProgressLog(""), previous, false)
	if structuredOK || !state.StatusDelayed || state.Active != 1 || state.Requests[0].Task != 77 {
		t.Fatalf("last known request was not retained: ok=%v state=%+v", structuredOK, state)
	}
}

func TestClaudeProgressTrackerElapsedAndLifecycle(t *testing.T) {
	now := time.Unix(1000, 0)
	tracker := newClaudeProgressTracker()
	tracker.now = func() time.Time { return now }
	active := claudeProgressState{Active: 1, Requests: []claudeRequestProgress{{Task: 4, Stage: "prefill"}}}
	tracker.enrich(&active)
	now = now.Add(74 * time.Second)
	tracker.enrich(&active)
	if active.Requests[0].ElapsedSeconds != 74 {
		t.Fatalf("elapsed=%d, want 74", active.Requests[0].ElapsedSeconds)
	}
	completed := claudeProgressState{TotalSlots: 4}
	tracker.enrich(&completed)
	if completed.Event != "request completed" {
		t.Fatalf("completion event=%q", completed.Event)
	}

	now = now.Add(6 * time.Second)
	tracker.enrich(&completed)
	if completed.Event != "" {
		t.Fatalf("stale completion event still visible: %q", completed.Event)
	}

	failedActive := claudeProgressState{Active: 1, Requests: []claudeRequestProgress{{Task: 5, Stage: "generating"}}}
	tracker.enrich(&failedActive)
	failed := claudeProgressState{Error: "connection refused"}
	tracker.enrich(&failed)
	if failed.Event != "request failed" || formatClaudeProgress(failed) != "ggrun · request failed" {
		t.Fatalf("failed request transition missing: %+v", failed)
	}
}

func TestClaudeCodeProgressArgs(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("GGRUN_CLAUDE_PROGRESS", "")
	args, ok := claudeCodeProgressClientArgs([]string{"--model", "local"}, 8123)
	if !ok || len(args) < 3 || args[0] != "--settings" {
		t.Fatalf("status line was not injected: ok=%v args=%v", ok, args)
	}
	var settings map[string]map[string]interface{}
	if err := json.Unmarshal([]byte(args[1]), &settings); err != nil {
		t.Fatal(err)
	}
	status := settings["statusLine"]
	command, _ := status["command"].(string)
	if !strings.Contains(command, "claude-status --port 8123") || status["refreshInterval"] != float64(2) {
		t.Fatalf("unexpected status settings: %v", status)
	}
	if !strings.Contains(args[1], `"matcher":"Workflow"`) || !strings.Contains(args[1], "claude-workflow-hook") {
		t.Fatalf("Workflow no-timeout hook missing from session settings: %s", args[1])
	}

	args, ok = claudeCodeProgressClientArgs([]string{"--settings", "mine.json"}, 8123)
	if ok || len(args) != 2 {
		t.Fatalf("user settings must win: ok=%v args=%v", ok, args)
	}

	withMetrics := claudeCodeProgressServerArgs([]string{"llama-server"}, true, "--slots --metrics")
	if !hasArg(withMetrics, "--metrics") {
		t.Fatalf("metrics flag missing: %v", withMetrics)
	}
	withoutSupport := claudeCodeProgressServerArgs([]string{"llama-server"}, true, "--slots")
	if hasArg(withoutSupport, "--metrics") {
		t.Fatalf("unsupported metrics flag added: %v", withoutSupport)
	}

	t.Setenv("GGRUN_CLAUDE_PROGRESS", "off")
	args, ok = claudeCodeProgressClientArgs(nil, 8123)
	if ok {
		t.Fatal("progress status line should not be injected when explicitly disabled")
	}
	if len(args) < 2 || !strings.Contains(args[1], "claude-workflow-hook") || strings.Contains(args[1], "statusLine") {
		t.Fatalf("disabling progress must keep the Workflow hook only: %v", args)
	}
}

func TestClaudeProgressStateRoundTripAndStaleness(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	port := 18765
	want := claudeProgressState{UpdatedAt: time.Now(), TotalSlots: 4, Active: 1}
	if err := writeClaudeProgressState(port, want); err != nil {
		t.Fatal(err)
	}
	got, err := readClaudeProgressState(port)
	if err != nil || got.TotalSlots != 4 || got.Active != 1 {
		t.Fatalf("round trip: got=%+v err=%v", got, err)
	}
	want.UpdatedAt = time.Now().Add(-claudeProgressStaleAfter - time.Second)
	if err := writeClaudeProgressState(port, want); err != nil {
		t.Fatal(err)
	}
	if _, err := readClaudeProgressState(port); err == nil {
		t.Fatal("expected stale state to be rejected")
	}
	_ = os.Remove(claudeProgressStatePath(port))
}

func TestClaudeProgressMonitorPublishesAndCleansState(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	mux := http.NewServeMux()
	mux.HandleFunc("/slots", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[{"id":0,"id_task":9,"is_processing":true,"n_prompt_tokens_processed":2049,"next_token":{"n_decoded":0}}]`))
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("llamacpp:prompt_tokens_seconds 20\nllamacpp:requests_deferred 0\n"))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	log := testProgressLog("slot print_timing: id  0 | task 9 | prompt processing, n_tokens =  2049, progress = 0.25, t = 100.00 s / 20.49 tokens per second\n")
	stop := startClaudeProgressMonitor(u.Hostname(), port, log, false)
	deadline := time.Now().Add(2 * time.Second)
	for {
		state, readErr := readClaudeProgressState(port)
		if readErr == nil && state.Active == 1 && len(state.Requests) == 1 {
			break
		}
		if time.Now().After(deadline) {
			stop()
			t.Fatalf("monitor did not publish active progress: state=%+v err=%v", state, readErr)
		}
		time.Sleep(10 * time.Millisecond)
	}
	stop()
	if _, err := os.Stat(claudeProgressStatePath(port)); !os.IsNotExist(err) {
		t.Fatalf("progress state was not cleaned up: %v", err)
	}
}

func TestParseClaudeLogSnapshotDecodesFromLog(t *testing.T) {
	log := `
slot launch_slot_: id  0 | task 96151 | processing task, is_child = 0
slot print_timing: id  0 | task 96151 | prompt processing, n_tokens = 61528, progress = 0.94, t = 444.66 s / 138.41 tokens per second
slot print_timing: id  0 | task 96151 | n_gen =    230, tg =   9.94 t/s, tg_3s =  10.78 t/s
slot print_timing: id  0 | task 96151 | n_gen =    296, tg =  10.04 t/s, tg_3s =  10.78 t/s
`
	snapshot := parseClaudeLogSnapshot(log)
	req, ok := snapshot.Active[96151]
	if !ok {
		t.Fatalf("decode task missing from snapshot: %+v", snapshot)
	}
	if req.Stage != "generating" || req.Generated != 296 {
		t.Fatalf("unexpected decode state: %+v", req)
	}
	if req.DecodeTPS != 10.78 {
		t.Fatalf("decode rate = %v, want windowed tg_3s 10.78", req.DecodeTPS)
	}
	// The prefill rate observed earlier in the same tail must survive the
	// merge so the status can show both rates.
	if req.PrefillTPS != 138.41 {
		t.Fatalf("prefill rate = %v, want 138.41 retained across decode merge", req.PrefillTPS)
	}
}

func TestParseDecodeLogProgressKeepsLatestPerTask(t *testing.T) {
	log := `
slot print_timing: id  0 | task 100 | n_gen =    100, tg =  10.69 t/s, tg_3s =  10.80 t/s
slot print_timing: id  1 | task 101 | n_gen =    152, tg =   8.10 t/s, tg_3s =   8.35 t/s
slot print_timing: id  0 | task 100 | n_gen =    296, tg =  10.04 t/s, tg_3s =  10.78 t/s
`
	items := parseDecodeLogProgress(log)
	if len(items) != 2 {
		t.Fatalf("got %d tasks, want 2: %+v", len(items), items)
	}
	var main decodeLogProgress
	for _, item := range items {
		if item.Task == 100 {
			main = item
		}
	}
	if main.Decoded != 296 || main.WindowRate != 10.78 {
		t.Fatalf("latest decode progress not retained: %+v", main)
	}
}

func TestPollClaudeProgressPrefersWindowedDecodeRate(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/slots", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[
  {"id":0,"id_task":9,"is_processing":true,"n_ctx":262144,"n_prompt_tokens":65193,"n_prompt_tokens_processed":61528,"next_token":{"has_next_token":true,"n_remain":30241,"n_decoded":1759}}
]`))
	})
	// /metrics either fails (the scheduler-busy case) or returns a stale
	// whole-run average; either way the log's windowed tg_3s must win.
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	mux.HandleFunc("/ggrun/router", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"active":1,"queued":0,"limit":1}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	log := testProgressLog("slot print_timing: id  0 | task 9 | prompt processing, n_tokens = 61528, progress = 0.94, t = 444.66 s / 138.41 tokens per second\nslot print_timing: id  0 | task 9 | n_gen =    296, tg =  10.04 t/s, tg_3s =  10.78 t/s\n")
	state := pollClaudeProgress(srv.Client(), u.Hostname(), port, log)
	if state.Active != 1 || len(state.Requests) != 1 {
		t.Fatalf("unexpected state: %+v", state)
	}
	req := state.Requests[0]
	if req.Stage != "generating" {
		t.Fatalf("stage = %q, want generating", req.Stage)
	}
	if req.DecodeTPS != 10.78 {
		t.Fatalf("decode tps = %v, want 10.78 from tg_3s log line", req.DecodeTPS)
	}
	if req.PrefillTPS != 138.41 {
		t.Fatalf("prefill tps = %v, want 138.41 retained", req.PrefillTPS)
	}
	got := formatClaudeProgress(state)
	if !strings.Contains(got, "prefill 138.4 tok/s") || !strings.Contains(got, "decode 10.8 tok/s") {
		t.Fatalf("status %q missing split rates", got)
	}
}

func TestPollClaudeProgressCancelsSlowMetricsAndCountsRouterOnce(t *testing.T) {
	var active atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/slots", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[{"id":0,"is_processing":false}]`))
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		active.Add(1)
		defer active.Add(-1)
		<-r.Context().Done()
	})
	mux.HandleFunc("/ggrun/router", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"queued":3}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	state := pollClaudeProgress(srv.Client(), u.Hostname(), port, nil)
	if elapsed := time.Since(started); elapsed > claudeProgressMetricsGrace+time.Second {
		t.Fatalf("slow metrics request was not bounded: %s", elapsed)
	}
	if !state.metricsAttempted || state.metricsOK {
		t.Fatalf("metrics backoff signal=%+v", state)
	}
	if state.Queued != 3 {
		t.Fatalf("router queue was counted %d times, want exactly 3", state.Queued)
	}
	deadline := time.Now().Add(time.Second)
	for active.Load() != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if active.Load() != 0 {
		t.Fatal("cancelled metrics request remained active")
	}
}

func TestPollClaudeProgressMetricsBackoffSkipsEndpoint(t *testing.T) {
	var metricsCalls atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/slots", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[]`))
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		metricsCalls.Add(1)
		_, _ = w.Write([]byte("llamacpp:requests_deferred 99\n"))
	})
	mux.HandleFunc("/ggrun/router", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"queued":2}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	state := pollClaudeProgressWithMetrics(srv.Client(), u.Hostname(), port, nil, false)
	if metricsCalls.Load() != 0 || state.metricsAttempted || state.Queued != 2 {
		t.Fatalf("metrics backoff was not isolated: calls=%d state=%+v", metricsCalls.Load(), state)
	}
}
