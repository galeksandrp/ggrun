package claudeauto

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestIsUtilityRequestMatchesOnlyTheAlias(t *testing.T) {
	if !IsUtilityRequest([]byte(`{"model":"` + UtilityAlias + `","messages":[]}`)) {
		t.Error("cheap-tier alias not recognised")
	}
	if !IsUtilityRequest([]byte(`{"model":"LOCAL-FAST","messages":[]}`)) {
		t.Error("alias match should be case-insensitive")
	}
	// The main tier, an unknown model and a malformed body must never be
	// downgraded: routing a user's foreground turn to a smaller model is a
	// correctness regression, not an optimisation.
	for _, body := range []string{
		`{"model":"local","messages":[]}`,
		`{"model":"claude-opus-5","messages":[]}`,
		`{"messages":[]}`,
		`not json`,
		`{"model":"local-fast-but-different"}`,
	} {
		if IsUtilityRequest([]byte(body)) {
			t.Errorf("non-utility body routed to the cheap tier: %s", body)
		}
	}
}

func TestRetargetModelRewritesOnlyTheModelField(t *testing.T) {
	in := []byte(`{"model":"local-fast","max_tokens":128,"messages":[{"role":"user","content":"hi"}]}`)
	out := retargetModel(in, "local")
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("rewritten body is not valid JSON: %v", err)
	}
	if got["model"] != "local" {
		t.Errorf("model = %v, want the backend alias", got["model"])
	}
	if got["max_tokens"] != float64(128) {
		t.Errorf("max_tokens lost: %v", got["max_tokens"])
	}
	msgs, ok := got["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Errorf("messages lost: %v", got["messages"])
	}
	// A body with no model field must pass through untouched rather than gain one.
	if out := retargetModel([]byte(`{"messages":[]}`), "local"); strings.Contains(string(out), "model") {
		t.Errorf("retargetModel invented a model field: %s", out)
	}
}

// With no companion backend the alias must fall back to the main model, or
// cheap-tier work would be routed into a lane that loops to the same server.
func TestUtilityLaneDisabledWithoutACompanion(t *testing.T) {
	var gotMain, gotCompanion int
	main := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		gotMain++
		_, _ = w.Write([]byte(`{"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer main.Close()
	companion := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		gotCompanion++
		_, _ = w.Write([]byte(`{"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer companion.Close()

	body := `{"model":"` + UtilityAlias + `","messages":[{"role":"user","content":"hi"}]}`

	off, err := StartRouter(main.URL, companion.URL, true, 1)
	if err != nil {
		t.Fatal(err)
	}
	off.SetCompanion("local", false) // no separate companion
	if err := postErr(off, body); err != nil {
		t.Fatal(err)
	}
	_ = off.Close()
	if gotMain != 1 || gotCompanion != 0 {
		t.Errorf("with no companion: main=%d companion=%d, want 1/0", gotMain, gotCompanion)
	}

	gotMain, gotCompanion = 0, 0
	on, err := StartRouter(main.URL, companion.URL, true, 1)
	if err != nil {
		t.Fatal(err)
	}
	on.SetCompanion("local", true)
	if err := postErr(on, body); err != nil {
		t.Fatal(err)
	}
	_ = on.Close()
	if gotCompanion != 1 || gotMain != 0 {
		t.Errorf("with a companion: main=%d companion=%d, want 0/1", gotMain, gotCompanion)
	}
}

// The companion must receive the alias it was launched with, not ggrun's label.
func TestCompanionReceivesItsOwnAlias(t *testing.T) {
	var seen string
	companion := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		seen = body.Model
		_, _ = w.Write([]byte(`{"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer companion.Close()
	main := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	}))
	defer main.Close()

	router, err := StartRouter(main.URL, companion.URL, true, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	router.SetCompanion("my-backend-alias", true)
	if err := postErr(router, `{"model":"`+UtilityAlias+`","messages":[]}`); err != nil {
		t.Fatal(err)
	}
	if seen != "my-backend-alias" {
		t.Errorf("companion saw model %q, want its own alias", seen)
	}
}

// Safety review must keep priority over the cheap tier: both share the
// companion backend, and a review blocks the user's tool call.
func TestClassifierStillWinsOverUtility(t *testing.T) {
	body := `{"model":"` + UtilityAlias + `","system":[{"type":"text","text":"` + ClassifierMarker + `"}],"messages":[]}`
	if !IsClassifierRequest([]byte(body)) {
		t.Fatal("classifier marker not detected when the cheap alias is also set")
	}
}

// The status endpoint must never fetch inline. A status line refreshing once a
// second would otherwise become a request per second against a backend that is
// already saturated, and would block on it.
func TestStatusEndpointDoesNotTouchTheBackend(t *testing.T) {
	var hits int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&hits, 1)
		_, _ = w.Write([]byte("llamacpp:prompt_seconds_total 1\n"))
	}))
	defer backend.Close()

	router, err := StartRouter(backend.URL, backend.URL, true, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	// Polling is not started, so nothing should reach the backend.
	for i := 0; i < 5; i++ {
		resp, err := http.Get(router.URL() + "/ggrun/router")
		if err != nil {
			t.Fatal(err)
		}
		_ = resp.Body.Close()
	}
	if n := atomic.LoadInt32(&hits); n != 0 {
		t.Errorf("status polling made %d backend requests, want 0", n)
	}
}

func TestBackendSnapshotAppearsOncePolled(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(
			"llamacpp:prompt_tokens_total 249473\n" +
				"llamacpp:prompt_seconds_total 1850.65\n" +
				"llamacpp:tokens_predicted_total 348\n" +
				"llamacpp:tokens_predicted_seconds_total 84.304\n"))
	}))
	defer backend.Close()

	router, err := StartRouter(backend.URL, backend.URL, true, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	if _, ok := router.BackendSnapshot(); ok {
		t.Error("snapshot available before polling started")
	}
	router.SetUBatch(512)
	router.StartBackendPolling(50 * time.Millisecond)

	deadline := time.Now().Add(3 * time.Second)
	var snap map[string]any
	for time.Now().Before(deadline) {
		if s, ok := router.BackendSnapshot(); ok {
			snap = s
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if snap == nil {
		t.Fatal("no backend snapshot after polling")
	}
	// The backend's measured decode rate, not one derived from request timing.
	if got, _ := snap["decode_tokens_per_s"].(float64); got < 4.0 || got > 4.3 {
		t.Errorf("decode_tokens_per_s = %v, want ~4.13", got)
	}
	pc, ok := snap["pass_cost"].(map[string]any)
	if !ok {
		t.Fatal("pass_cost missing from the snapshot")
	}
	if share, _ := pc["fixed_share"].(float64); share < 0.90 {
		t.Errorf("fixed_share = %v, want >0.90 for this fixture", share)
	}
}
