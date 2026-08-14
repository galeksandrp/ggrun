package modelusage

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRecordLaunchPersistsAndBumps(t *testing.T) {
	dir := t.TempDir()
	RecordLaunch(dir, "/models/alpha.gguf")
	RecordLaunch(dir, "/models/alpha.gguf")
	RecordLaunch(dir, "/models/beta.gguf")

	models := Load(dir)
	alpha := models["/models/alpha.gguf"]
	if alpha.Launches != 2 {
		t.Fatalf("alpha launches = %d, want 2", alpha.Launches)
	}
	beta := models["/models/beta.gguf"]
	if beta.Launches != 1 {
		t.Fatalf("beta launches = %d, want 1", beta.Launches)
	}
	if alpha.LastUsedAt.IsZero() || beta.LastUsedAt.IsZero() {
		t.Fatal("last-used timestamps must be set")
	}
	// Writes are back-to-back, so timestamps can be equal down to the
	// nanosecond. The final write belongs to beta, so beta must never be older.
	if beta.LastUsedAt.Before(alpha.LastUsedAt) {
		t.Fatalf("beta (final write) must not be older than alpha: beta=%v alpha=%v", beta.LastUsedAt, alpha.LastUsedAt)
	}
}

func TestLoadEmptyDirReturnsNil(t *testing.T) {
	if got := Load(t.TempDir()); got != nil {
		t.Fatalf("expected nil for missing cache, got %v", got)
	}
}

func TestRecordLaunchIgnoresEmptyInput(t *testing.T) {
	dir := t.TempDir()
	RecordLaunch("", "/models/alpha.gguf")
	RecordLaunch(dir, "  ")
	if got := Load(dir); got != nil {
		t.Fatalf("empty-key launch must not create a cache file, got %v", got)
	}
}

func TestCorruptCacheFallsBackToEmpty(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, cacheFile), []byte("not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := Load(dir); got != nil {
		t.Fatalf("corrupt cache must return nil, got %v", got)
	}
	// A later real launch must recover instead of deadlocking on the bad file.
	RecordLaunch(dir, "/models/alpha.gguf")
	if models := Load(dir); models["/models/alpha.gguf"].Launches != 1 {
		t.Fatalf("record after corrupt cache failed: %+v", models)
	}
}

func TestTimestampRoundTrip(t *testing.T) {
	dir := t.TempDir()
	RecordLaunch(dir, "/models/alpha.gguf")
	models := Load(dir)
	if _, err := time.Parse(time.RFC3339, models["/models/alpha.gguf"].LastUsedAt.UTC().Format(time.RFC3339)); err != nil {
		t.Fatalf("timestamp did not survive JSON round trip: %v", err)
	}
}
