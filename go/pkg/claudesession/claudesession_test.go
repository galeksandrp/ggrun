package claudesession

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestNewSessionIDIsAValidUUIDAndUnique(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 50; i++ {
		id, err := NewSessionID()
		if err != nil {
			t.Fatalf("NewSessionID: %v", err)
		}
		if !ValidSessionID(id) {
			t.Fatalf("generated id %q is not a valid session id", id)
		}
		// Claude Code requires a v4 UUID; a non-conforming id is rejected at
		// launch, which would silently lose the resume handle.
		if id[14] != '4' {
			t.Fatalf("id %q is not version 4", id)
		}
		if variant := id[19]; variant != '8' && variant != '9' && variant != 'a' && variant != 'b' {
			t.Fatalf("id %q has wrong variant nibble %q", id, variant)
		}
		if seen[id] {
			t.Fatalf("duplicate session id %q", id)
		}
		seen[id] = true
	}
}

func TestValidSessionIDRejectsPathEscapes(t *testing.T) {
	for _, id := range []string{
		"", "not-a-uuid", "../../etc/passwd",
		"../../../home/mik/.claude/creds.json",
		"12345678-1234-1234-1234-12345678901", // too short
		"12345678-1234-1234-1234-1234567890123",
		"12345678/1234-1234-1234-123456789012",
		"gggggggg-1234-1234-1234-123456789012",
	} {
		if ValidSessionID(id) {
			t.Errorf("ValidSessionID(%q) = true, want false", id)
		}
	}
	if !ValidSessionID("072e63a1-819a-4682-a742-559695c3cd76") {
		t.Error("rejected a real session id")
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	rec := Record{
		SessionID:  "072e63a1-819a-4682-a742-559695c3cd76",
		WorkDir:    "/home/mik/ggrun-project/ggrun",
		ModelPath:  "/models/Laguna.gguf",
		Port:       8081,
		ServerArgs: []string{"--ctx-size", "1048576", "--parallel", "4"},
		Workflow:   &Workflow{RunID: "wf_894b5285-5d3", Name: "deep-research"},
	}
	if err := Save(dir, rec); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, err := Load(dir, rec.SessionID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.ModelPath != rec.ModelPath || got.Port != rec.Port {
		t.Errorf("round trip lost fields: %+v", got)
	}
	if got.Workflow == nil || got.Workflow.RunID != "wf_894b5285-5d3" {
		t.Errorf("workflow not preserved: %+v", got.Workflow)
	}
	if got.Recorded.IsZero() {
		t.Error("Recorded not stamped on save")
	}
	// A partial write must not survive as a loadable record.
	if entries, _ := os.ReadDir(Dir(dir)); len(entries) != 1 {
		t.Errorf("want 1 record file, got %d", len(entries))
	}
}

func TestSaveRejectsInvalidSessionID(t *testing.T) {
	dir := t.TempDir()
	if err := Save(dir, Record{SessionID: "../escape"}); err == nil {
		t.Fatal("Save accepted an invalid session id")
	}
	if _, err := os.Stat(Dir(dir)); err == nil {
		t.Error("Save created the records directory for an invalid id")
	}
}

func TestListAndLatestAreScopedToWorkDirAndOrdered(t *testing.T) {
	dir := t.TempDir()
	older := Record{
		SessionID: "11111111-1111-4111-8111-111111111111",
		WorkDir:   "/project/a", Recorded: time.Now().Add(-2 * time.Hour),
	}
	newer := Record{
		SessionID: "22222222-2222-4222-8222-222222222222",
		WorkDir:   "/project/a", Recorded: time.Now().Add(-1 * time.Hour),
	}
	other := Record{
		SessionID: "33333333-3333-4333-8333-333333333333",
		WorkDir:   "/project/b", Recorded: time.Now(),
	}
	for _, rec := range []Record{older, newer, other} {
		if err := Save(dir, rec); err != nil {
			t.Fatalf("Save: %v", err)
		}
	}
	list, err := List(dir, "/project/a")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("want 2 records for /project/a, got %d", len(list))
	}
	if list[0].SessionID != newer.SessionID {
		t.Errorf("List not newest-first: got %s", list[0].SessionID)
	}
	latest, err := Latest(dir, "/project/a")
	if err != nil {
		t.Fatalf("Latest: %v", err)
	}
	if latest.SessionID != newer.SessionID {
		t.Errorf("Latest = %s, want %s", latest.SessionID, newer.SessionID)
	}
	// A different project must not resume into this one's session.
	if _, err := Latest(dir, "/project/c"); err == nil {
		t.Error("Latest returned a record for an unrelated work dir")
	}
}

func TestListOnMissingDirectoryIsEmptyNotAnError(t *testing.T) {
	list, err := List(t.TempDir(), "")
	if err != nil {
		t.Fatalf("List on empty cache: %v", err)
	}
	if len(list) != 0 {
		t.Errorf("want no records, got %d", len(list))
	}
}

func TestShapeMismatchesDetectSettingsThatChangeOutput(t *testing.T) {
	rec := Record{ServerArgs: []string{
		"--ctx-size", "1048576", "--parallel", "4",
		"--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
		"--n-cpu-moe", "24", "-b", "2048",
	}}
	same := rec.ServerArgs
	if diff := rec.ShapeMismatches(same); len(diff) != 0 {
		t.Errorf("identical shape reported %d mismatches: %v", len(diff), diff)
	}

	// Changing KV type reinterprets a cache built under another quant.
	changed := []string{
		"--ctx-size", "1048576", "--parallel", "4",
		"--cache-type-k", "f16", "--cache-type-v", "q4_0",
		"--n-cpu-moe", "24", "-b", "2048",
	}
	diff := rec.ShapeMismatches(changed)
	if len(diff) != 1 || diff[0].Key != "--cache-type-k" {
		t.Fatalf("want one --cache-type-k mismatch, got %v", diff)
	}
	if diff[0].Recorded != "q4_0" || diff[0].Proposed != "f16" {
		t.Errorf("mismatch detail wrong: %v", diff[0])
	}

	// A dropped setting is a mismatch, not a match against empty.
	dropped := []string{"--ctx-size", "1048576", "--parallel", "4"}
	if diff := rec.ShapeMismatches(dropped); len(diff) == 0 {
		t.Error("dropping KV and placement flags reported no mismatch")
	}
}

func TestShapeIgnoresSettingsThatCannotChangeOutput(t *testing.T) {
	rec := Record{ServerArgs: []string{"--ctx-size", "65536", "--port", "8081", "--alias", "local"}}
	// Port and alias differ; neither alters model output, so neither may block
	// a resume.
	other := []string{"--ctx-size", "65536", "--port", "9090", "--alias", "other"}
	if diff := rec.ShapeMismatches(other); len(diff) != 0 {
		t.Errorf("non-output settings blocked a resume: %v", diff)
	}
}

func TestProjectKeyMatchesClaudeCodeLayout(t *testing.T) {
	cases := map[string]string{
		"/home/mik/ggrun-project/ggrun":   "-home-mik-ggrun-project-ggrun",
		"/home/mik/ggrun/.src/llm-server": "-home-mik-ggrun--src-llm-server",
		"/home/mik/ultra-zen":             "-home-mik-ultra-zen",
	}
	for in, want := range cases {
		if got := ProjectKey(in); got != want {
			t.Errorf("ProjectKey(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestJournalPathAndCachedAgents(t *testing.T) {
	projects := t.TempDir()
	workDir := "/home/mik/ggrun-project/ggrun"
	session := "072e63a1-819a-4682-a742-559695c3cd76"
	run := "wf_894b5285-5d3"
	path := JournalPath(projects, workDir, session, run)
	want := filepath.Join(projects, "-home-mik-ggrun-project-ggrun", session,
		"subagents", "workflows", run, "journal.jsonl")
	if path != want {
		t.Fatalf("JournalPath = %q, want %q", path, want)
	}

	if got := CachedAgents(path); got != 0 {
		t.Errorf("missing journal reported %d cached agents, want 0", got)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	journal := `{"type":"started","key":"a","agentId":"1"}
{"type":"result","key":"a","agentId":"1","result":{}}
{"type":"started","key":"b","agentId":"2"}
{"type":"result","key":"b","agentId":"2","result":{}}
{"type":"started","key":"c","agentId":"3"}
`
	if err := os.WriteFile(path, []byte(journal), 0o600); err != nil {
		t.Fatal(err)
	}
	// Two results and three starts: only the finished agents replay.
	if got := CachedAgents(path); got != 2 {
		t.Errorf("CachedAgents = %d, want 2", got)
	}
}
