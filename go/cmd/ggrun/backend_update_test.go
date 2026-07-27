package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/backends"
)

func TestBackendLayoutFor(t *testing.T) {
	ok, err := backendLayoutFor(backends.Backend{
		Tag:  "laguna",
		Path: "/src/fork-llama.cpp-laguna/build-cuda/bin/llama-server",
	})
	if err != nil {
		t.Fatalf("recognised layout rejected: %v", err)
	}
	if ok.srcDir != "/src/fork-llama.cpp-laguna" || ok.buildDir != "/src/fork-llama.cpp-laguna/build-cuda" || ok.accel != "cuda" {
		t.Errorf("layout = %+v", ok)
	}

	// A hand-registered binary has no build tree, and updating it would rebuild
	// something ggrun never built.
	for _, bad := range []string{
		"/usr/local/bin/llama-server",
		"/src/fork/build-cuda/llama-server",
		"/src/fork/out/bin/llama-server",
		"",
	} {
		if _, err := backendLayoutFor(backends.Backend{Tag: "x", Path: bad}); err == nil {
			t.Errorf("path %q accepted as a ggrun build layout", bad)
		}
	}
}

func TestArchiveBuildDirPreservesAndDoesNotCollide(t *testing.T) {
	root := t.TempDir()
	build := filepath.Join(root, "build-cuda")
	if err := os.MkdirAll(filepath.Join(build, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	marker := filepath.Join(build, "bin", "llama-server")
	if err := os.WriteFile(marker, []byte("first"), 0o755); err != nil {
		t.Fatal(err)
	}

	first, err := archiveBuildDir(build, "04b2b72cb54048ead292884adbe11f284e3ec950")
	if err != nil {
		t.Fatalf("archive: %v", err)
	}
	if !strings.HasSuffix(first, "build-cuda-prev-04b2b72cb540") {
		t.Errorf("archive name = %q", first)
	}
	// The point of archiving is that the old binary survives intact.
	if got, _ := os.ReadFile(filepath.Join(first, "bin", "llama-server")); string(got) != "first" {
		t.Errorf("archived binary content = %q, want %q", got, "first")
	}
	if _, err := os.Stat(build); !os.IsNotExist(err) {
		t.Error("build dir should have been moved aside, not copied")
	}

	// A second archive at the same commit must not overwrite the first: that
	// would destroy the only build known to work.
	if err := os.MkdirAll(filepath.Join(build, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(marker, []byte("second"), 0o755); err != nil {
		t.Fatal(err)
	}
	second, err := archiveBuildDir(build, "04b2b72cb54048ead292884adbe11f284e3ec950")
	if err != nil {
		t.Fatalf("second archive: %v", err)
	}
	if second == first {
		t.Fatal("second archive reused the first path")
	}
	if got, _ := os.ReadFile(filepath.Join(first, "bin", "llama-server")); string(got) != "first" {
		t.Error("first archive was clobbered by the second")
	}
}

func TestArchiveBuildDirUnknownCommit(t *testing.T) {
	root := t.TempDir()
	build := filepath.Join(root, "build-cuda")
	if err := os.MkdirAll(build, 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := archiveBuildDir(build, "")
	if err != nil {
		t.Fatalf("archive: %v", err)
	}
	if !strings.HasSuffix(got, "build-cuda-prev-unknown") {
		t.Errorf("archive name = %q", got)
	}
}

func TestPruneBuildDirRefusesNonArchives(t *testing.T) {
	root := t.TempDir()
	// Guard against deleting a live build tree, or anything else, by name.
	for _, name := range []string{"build-cuda", "src", ""} {
		p := filepath.Join(root, name)
		if name != "" {
			if err := os.MkdirAll(p, 0o755); err != nil {
				t.Fatal(err)
			}
		} else {
			p = ""
		}
		if err := pruneBuildDir(p); err == nil {
			t.Errorf("pruneBuildDir(%q) was allowed", p)
		}
		if name != "" {
			if _, err := os.Stat(p); err != nil {
				t.Errorf("%q was removed despite the refusal", name)
			}
		}
	}

	archive := filepath.Join(root, "build-cuda-prev-abc123")
	if err := os.MkdirAll(archive, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := pruneBuildDir(archive); err != nil {
		t.Fatalf("pruning a real archive failed: %v", err)
	}
	if _, err := os.Stat(archive); !os.IsNotExist(err) {
		t.Error("archive still present after prune")
	}
}

func TestShortCommit(t *testing.T) {
	if got := shortCommit("04b2b72cb54048ead292884adbe11f284e3ec950"); got != "04b2b72cb540" {
		t.Errorf("shortCommit = %q", got)
	}
	if got := shortCommit(""); got != "(unknown)" {
		t.Errorf("empty commit = %q", got)
	}
	if got := shortCommit("abc123"); got != "abc123" {
		t.Errorf("short input = %q", got)
	}
}
