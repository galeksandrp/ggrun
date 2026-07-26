package claudeauto

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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
