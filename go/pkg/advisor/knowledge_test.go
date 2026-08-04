package advisor

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestResearchModelCardParsesPinnedCard verifies the model-card tier: the pinned
// README is parsed into a single Source with a bounded excerpt, the SHA-256 of
// the card body is recorded for provenance, and the exact fetch path is kept on
// SearchPath for reproducibility.
func TestResearchModelCardParsesPinnedCard(t *testing.T) {
	longCard := strings.Repeat("card text ", 900) // ~9 KB, past the 1800-char budget
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/Nanbeige/Nanbeige4.2-3B/raw/main/README.md" {
			t.Fatalf("unexpected model card path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/markdown")
		_, _ = w.Write([]byte(longCard))
	}))
	defer server.Close()

	incident := Incident{
		SchemaVersion: IncidentSchemaVersion, Mode: ModeSupport, Architecture: "nanbeige",
		Settings: map[string]string{"stage": "backend_start"},
	}
	if err := incident.Normalize(); err != nil {
		t.Fatal(err)
	}
	sources, err := (Researcher{Client: server.Client()}).ResearchModelCard(context.Background(), incident, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) != 1 {
		t.Fatalf("sources=%d, want exactly one pinned card", len(sources))
	}
	source := sources[0]
	if !strings.HasPrefix(source.URL, "https://huggingface.co/Nanbeige/") {
		t.Fatalf("unexpected card URL %q", source.URL)
	}
	if len(source.Excerpt) > 1800 {
		t.Fatalf("excerpt exceeds 1800-char budget: %d", len(source.Excerpt))
	}
	if len(source.SHA256) != 64 {
		t.Fatalf("model card did not record a SHA-256 digest: %q", source.SHA256)
	}
	if source.SearchPath == "" || !strings.Contains(source.SearchPath, "/README.md") {
		t.Fatalf("model card did not record its fetch path: %q", source.SearchPath)
	}
}

// TestResearchModelCardUnpinnedArchitecture ensures an architecture without a
// pinned card contributes nothing rather than fetching an arbitrary repo.
func TestResearchModelCardUnpinnedArchitecture(t *testing.T) {
	incident := Incident{Mode: ModeSupport, Architecture: "qwen3.6", Settings: map[string]string{}}
	_ = incident.Normalize()
	sources, err := (Researcher{}).ResearchModelCard(context.Background(), incident, "https://unused.invalid")
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) != 0 {
		t.Fatalf("unpinned architecture returned sources: %v", sources)
	}
}

// TestResearchModelCardCustomSettingsPin verifies the only route to an unseeded
// architecture: an explicit settings pin still points at a fixed repo@revision.
func TestResearchModelCardCustomSettingsPin(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/Someone/else/raw/somecommit/README.md" {
			t.Fatalf("unexpected custom card path %q", r.URL.Path)
		}
		_, _ = w.Write([]byte("custom card body"))
	}))
	defer server.Close()
	incident := Incident{
		Mode: ModeSupport, Architecture: "qwen3.6",
		Settings: map[string]string{"model_card_pin": "Someone/else@somecommit"},
	}
	_ = incident.Normalize()
	sources, err := (Researcher{Client: server.Client()}).ResearchModelCard(context.Background(), incident, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) != 1 || !strings.Contains(sources[0].SearchPath, "/Someone/else/raw/somecommit/") {
		t.Fatalf("custom settings pin not honored: %+v", sources)
	}
}

// TestResearchModelCardRejectsNonOK ensures the HTTP status gate fires through
// the injected base URL, which the fallback chain turns into a degraded note.
func TestResearchModelCardRejectsNonOK(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	}))
	defer server.Close()
	incident := Incident{Mode: ModeSupport, Architecture: "nanbeige"}
	_ = incident.Normalize()
	_, err := (Researcher{Client: server.Client()}).ResearchModelCard(context.Background(), incident, server.URL)
	if err == nil || !strings.Contains(err.Error(), "403") {
		t.Fatalf("expected HTTP 403 error, got %v", err)
	}
}

// TestResearchOfficialParsesTypedIssueIndex verifies the injected-base-URL seam:
// the official issue index is parsed into Sources whose excerpt is bounded at
// 1800 chars, only github.com/ggml-org/llama.cpp/ URLs survive, and each source
// records the exact search path it was fetched from.
func TestResearchOfficialParsesTypedIssueIndex(t *testing.T) {
	longBody := strings.Repeat("x", 5000) + " real signal"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/search/issues" {
			t.Fatalf("unexpected request path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"items": []map[string]any{
				{
					"title":      "Fix a llama.cpp scheduling bug",
					"html_url":   "https://github.com/ggml-org/llama.cpp/issues/42",
					"body":       longBody,
					"updated_at": "2026-07-30T12:00:00Z",
				},
				{
					"title":      "Off-topic: unrelated repo chatter",
					"html_url":   "https://github.com/someone-else/other/issues/7",
					"body":       "should be filtered",
					"updated_at": "2026-07-29T12:00:00Z",
				},
				{
					"title":      "CUDA OOM under 128 ubatch",
					"html_url":   "https://github.com/ggml-org/llama.cpp/pull/9001",
					"body":       "",
					"updated_at": "2026-07-28T12:00:00Z",
				},
			},
		})
	}))
	defer server.Close()

	incident := Incident{
		SchemaVersion: IncidentSchemaVersion, Mode: ModeSupport, Architecture: "llama",
		Observations: []Observation{{Code: "cuda0_oom", Component: "memory", Source: "nvidia-smi", Confidence: "measured"}},
	}
	if err := incident.Normalize(); err != nil {
		t.Fatal(err)
	}
	sources, err := (Researcher{Client: server.Client()}).ResearchOfficial(context.Background(), incident, server.URL)
	if err != nil {
		t.Fatal(err)
	}

	if len(sources) != 2 {
		t.Fatalf("sources=%d, want 2 (only official-owner URLs survive)", len(sources))
	}
	for _, source := range sources {
		if !strings.HasPrefix(source.URL, "https://github.com/ggml-org/llama.cpp/") {
			t.Fatalf("non-official URL survived filtering: %q", source.URL)
		}
		if len(source.Excerpt) > 1800 {
			t.Fatalf("excerpt exceeds 1800-char budget for %q: %d", source.URL, len(source.Excerpt))
		}
		if source.SearchPath == "" {
			t.Fatalf("source %q did not record its search path", source.URL)
		}
		if !strings.HasPrefix(source.SearchPath, server.URL+"/search/issues?q=") {
			t.Fatalf("search path does not record the queried endpoint: %q", source.SearchPath)
		}
	}

	// The long body must be truncated, not silently dropped or retained whole.
	if got := sources[0].Excerpt; len(got) > 1800 || got == longBody {
		t.Fatalf("long body was not clipped: len=%d", len(got))
	}
	// An item with an empty body falls back to a clipped title excerpt.
	if got := sources[1].Excerpt; got == "" {
		t.Fatal("item with empty body did not fall back to title excerpt")
	}
}

// TestResearchOfficialRejectsNonOK ensures the HTTP status gate still fires
// through the injected base URL.
func TestResearchOfficialRejectsNonOK(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	}))
	defer server.Close()
	incident := Incident{Mode: ModeSupport, Architecture: "llama"}
	_ = incident.Normalize()
	_, err := (Researcher{Client: server.Client()}).ResearchOfficial(context.Background(), incident, server.URL)
	if err == nil || !strings.Contains(err.Error(), "403") {
		t.Fatalf("expected HTTP 403 error, got %v", err)
	}
}

// TestBundledKnowledgeIncludesCacheOnlyForCacheShapedIncidents locks the branch
// that keeps the offline knowledge bundle stable.
func TestBundledKnowledgeIncludesCacheOnlyForCacheShapedIncidents(t *testing.T) {
	plain := Incident{Mode: ModeSupport, Architecture: "llama", ProfileState: "load_healthy"}
	_ = plain.Normalize()
	base := BundledKnowledge(plain)
	for _, source := range base {
		if source.URL == "ggrun://knowledge/cache" {
			t.Fatal("cache knowledge bundled for a non-cache incident")
		}
	}

	cached := Incident{Mode: ModeSupport, Architecture: "deepseek4", ProfileState: "cache_stale"}
	_ = cached.Normalize()
	sawCache := false
	for _, source := range BundledKnowledge(cached) {
		if source.URL == "ggrun://knowledge/cache" {
			sawCache = true
		}
	}
	if !sawCache {
		t.Fatal("cache knowledge missing for a cache-shaped incident")
	}
}
