package advisor

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/server"
)

func TestSupportBackendEnvUsesHelpersOwnLibraryTree(t *testing.T) {
	buildBin := filepath.Join(t.TempDir(), "build-cpu", "bin")
	if err := os.MkdirAll(buildBin, 0o755); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(buildBin, "llama-server")
	if err := os.WriteFile(binary, []byte("binary"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(buildBin, "libllama-server-impl.so"), []byte("library"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_SERVER_LIB_HUB", "/wrong/main/backend")
	env := supportBackendEnv(binary)
	joined := strings.Join(env, "\n")
	if !strings.Contains(joined, "LD_LIBRARY_PATH="+buildBin) {
		t.Fatalf("helper environment did not select its own libraries: %v", env)
	}
	if strings.Contains(joined, "/wrong/main/backend") {
		t.Fatalf("helper environment inherited the main backend hub: %v", env)
	}
}

func TestWaitForResourceReleaseRejectsLiveProcess(t *testing.T) {
	cmd := exec.Command("sleep", "10")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start test process: %v", err)
	}
	process := &server.Process{Cmd: cmd}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()

	if waitForResourceRelease(process, captureResources(), 20*time.Millisecond) {
		t.Fatal("live helper process must block resource-release verification")
	}
}

func TestCompletedRunErrorPrioritizesUnverifiedRelease(t *testing.T) {
	queryErr := errors.New("query failed")
	stopErr := errors.New("stop failed")
	err := completedRunError(false, queryErr, stopErr)
	if !errors.Is(err, ErrResourceReleaseUnverified) {
		t.Fatalf("unverified release must be the hard failure, got %v", err)
	}
	if errors.Is(err, queryErr) || errors.Is(err, stopErr) {
		t.Fatalf("ordinary helper errors must not mask or wrap the release boundary: %v", err)
	}

	if err := completedRunError(true, queryErr, stopErr); !errors.Is(err, queryErr) {
		t.Fatalf("verified release should preserve query error, got %v", err)
	}
}

// The 2026-08-02 incident: NanoBeige spent its whole budget inside <think>, so
// llama.cpp returned reasoning_content with an empty content and the controller
// reported a bare "EOF" after three wasted minutes.
func TestReasoningOffArgsPrefersTheFlagTheBackendAdvertises(t *testing.T) {
	realHelp := "-rea,  --reasoning [on|off|auto]   Use reasoning/thinking in the chat\n" +
		"--reasoning-budget N    token budget for thinking\n" +
		"--reasoning-format FORMAT   controls thought tags\n"
	if got := reasoningOffArgs(realHelp); len(got) != 2 || got[0] != "--reasoning" || got[1] != "off" {
		t.Fatalf("reasoningOffArgs(real help) = %v, want [--reasoning off]", got)
	}
	budgetOnly := "--reasoning-budget N   token budget for thinking\n--reasoning-format FORMAT  x\n"
	if got := reasoningOffArgs(budgetOnly); len(got) != 2 || got[0] != "--reasoning-budget" || got[1] != "0" {
		t.Fatalf("reasoningOffArgs(budget-only help) = %v, want [--reasoning-budget 0]", got)
	}
	// An unknown flag aborts argument parsing, so a backend that advertises
	// neither spelling must get nothing rather than a guess.
	if got := reasoningOffArgs("--ctx-size N\n--parallel N\n"); got != nil {
		t.Fatalf("reasoningOffArgs(no reasoning flags) = %v, want nil", got)
	}
	if got := reasoningOffArgs(""); got != nil {
		t.Fatalf("reasoningOffArgs(unprobeable help) = %v, want nil", got)
	}
}

func TestExtractJSONObjectRecoversADecisionFromAReasoningTrace(t *testing.T) {
	trace := `I should check the device. {"schema_version":1,"action":"no_action"} that is my answer.`
	if got := extractJSONObject(trace); got != `{"schema_version":1,"action":"no_action"}` {
		t.Fatalf("extractJSONObject = %q", got)
	}
	// Braces inside strings must not close the object early.
	nested := `x {"a":"}{","b":{"c":1}} y`
	if got := extractJSONObject(nested); got != `{"a":"}{","b":{"c":1}}` {
		t.Fatalf("extractJSONObject(nested) = %q", got)
	}
	if got := extractJSONObject("no json here"); got != "" {
		t.Fatalf("extractJSONObject(none) = %q, want empty", got)
	}
	// A truncated object is not a decision.
	if got := extractJSONObject(`{"schema_version":1,"action":`); got != "" {
		t.Fatalf("extractJSONObject(truncated) = %q, want empty", got)
	}
}

// TestResearchKnowledgeFallbackChainWithDegradedNotes exercises the
// Official -> ModelCard -> Bundled chain end to end. Both online tiers are
// pointed at local httptest servers via the researchBase seam, so no network is
// touched; the bundled tier is always present. It verifies the chain appends
// sources and records a "research_degraded" settings note when a tier fails.
func TestResearchKnowledgeFallbackChainWithDegradedNotes(t *testing.T) {
	// Official tier serves one matching issue; model-card tier serves its README.
	official := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/search/issues") {
			t.Fatalf("unexpected official request path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"items": []map[string]any{{
				"title":      "Fix a llama.cpp scheduling bug",
				"html_url":   "https://github.com/ggml-org/llama.cpp/issues/42",
				"body":       "the real issue body",
				"updated_at": "2026-07-30T12:00:00Z",
			}},
		})
	}))
	defer official.Close()
	card := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/README.md") {
			t.Fatalf("unexpected model card request path %q", r.URL.Path)
		}
		_, _ = w.Write([]byte("# Nanbeige4.2\n\nKnown-good context size is 16k."))
	}))
	defer card.Close()

	original := researchBase
	researchBase.Official = official.URL
	researchBase.ModelCard = card.URL
	defer func() { researchBase = original }()

	incident := Incident{
		SchemaVersion: IncidentSchemaVersion, Mode: ModeSupport, Architecture: "nanbeige",
		Settings: map[string]string{"stage": "backend_start"},
	}
	if err := incident.Normalize(); err != nil {
		t.Fatal(err)
	}
	// Bundled tier is always seeded before research starts.
	incident.Knowledge = append(incident.Knowledge, BundledKnowledge(incident)...)

	var report RunReport
	degraded := researchKnowledge(context.Background(), official.Client(), &incident, &report)

	if len(degraded) != 0 {
		t.Fatalf("healthy chain reported degradation: %v", degraded)
	}
	if report.OnlineSources != 1 {
		t.Fatalf("online sources=%d, want 1", report.OnlineSources)
	}
	sawOfficial, sawCard := false, false
	for _, source := range incident.Knowledge {
		if strings.HasPrefix(source.URL, "https://github.com/ggml-org/llama.cpp/") {
			sawOfficial = true
		}
		if strings.HasPrefix(source.URL, "https://huggingface.co/Nanbeige/") {
			sawCard = true
		}
	}
	if !sawOfficial || !sawCard {
		t.Fatalf("fallback chain missing tiers: official=%t card=%t", sawOfficial, sawCard)
	}
}

// TestResearchKnowledgeChainDegradesWhenOfficialUnavailable verifies the
// degraded note path: when the official tier returns HTTP 403 and the model card
// tier is unpinned, the chain still returns with a "research_degraded" reason
// and the analysis is NOT aborted.
func TestResearchKnowledgeChainDegradesWhenOfficialUnavailable(t *testing.T) {
	official := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	}))
	defer official.Close()

	original := researchBase
	researchBase.Official = official.URL
	researchBase.ModelCard = "https://unused.invalid" // pin-only lookups never fetch when unpinned
	defer func() { researchBase = original }()

	incident := Incident{
		SchemaVersion: IncidentSchemaVersion, Mode: ModeSupport, Architecture: "qwen3.6",
		Settings: map[string]string{},
	}
	if err := incident.Normalize(); err != nil {
		t.Fatal(err)
	}
	incident.Knowledge = append(incident.Knowledge, BundledKnowledge(incident)...)

	var report RunReport
	degraded := researchKnowledge(context.Background(), official.Client(), &incident, &report)

	if len(degraded) == 0 {
		t.Fatal("expected a degraded note when the official tier fails")
	}
	if !strings.Contains(strings.Join(degraded, "; "), "official issue index: official research returned HTTP 403") {
		t.Fatalf("degraded note does not name the failing tier: %v", degraded)
	}
	if !strings.Contains(strings.Join(degraded, "; "), "model card: not pinned") {
		t.Fatalf("degraded note does not explain the unpinned card tier: %v", degraded)
	}
	// The bundled tier must survive so the analysis has evidence regardless.
	sawBundled := false
	for _, source := range incident.Knowledge {
		if strings.HasPrefix(source.URL, "ggrun://knowledge/") {
			sawBundled = true
		}
	}
	if !sawBundled {
		t.Fatal("bundled knowledge was lost on degradation")
	}
}
