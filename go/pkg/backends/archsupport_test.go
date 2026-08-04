package backends

import (
	"os"
	"path/filepath"
	"testing"
)

// writeProbeFile builds a file whose bytes surround the given literals the way
// a loader's string table does, so the probe is exercised against the shape it
// actually meets rather than a bare substring.
func writeProbeFile(t *testing.T, name string, body []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatalf("write probe file: %v", err)
	}
	return path
}

func TestFileHasLiteralRequiresNULBrackets(t *testing.T) {
	// A build directory embedded in a binary is the exact false positive the
	// NUL-bracketing exists to reject.
	body := []byte("\x00/home/u/src/fork-llama.cpp-add-laguna/build/x.o\x00\x00qwen3moe\x00\x00llama\x00")
	path := writeProbeFile(t, "strings.bin", body)

	cases := []struct {
		arch string
		want bool
	}{
		{"qwen3moe", true},
		{"llama", true},
		// Present only inside a path string, never as its own literal.
		{"laguna", false},
		// A prefix of a real literal must not match.
		{"qwen3", false},
		{"bogusarch", false},
	}
	for _, c := range cases {
		needle := append(append([]byte{0}, c.arch...), 0)
		got, err := fileHasLiteral(path, needle)
		if err != nil {
			t.Fatalf("%s: %v", c.arch, err)
		}
		if got != c.want {
			t.Errorf("fileHasLiteral(%q) = %v, want %v", c.arch, got, c.want)
		}
	}
}

func TestFileHasLiteralAcrossChunkBoundary(t *testing.T) {
	// A literal straddling the read boundary is found only if the carry overlap
	// is right, and that is precisely the bug a single-chunk test would miss.
	needle := append(append([]byte{0}, "laguna"...), 0)

	for _, offset := range []int{-3, -1, 0, 1} {
		start := archProbeChunk + offset
		body := make([]byte, archProbeChunk*2)
		for i := range body {
			body[i] = 'x'
		}
		copy(body[start:], needle)

		path := writeProbeFile(t, "boundary.bin", body)
		got, err := fileHasLiteral(path, needle)
		if err != nil {
			t.Fatalf("offset %d: %v", offset, err)
		}
		if !got {
			t.Errorf("literal at chunk offset %d not found", offset)
		}
	}
}

func TestBackendSupportsArchCannotProbe(t *testing.T) {
	// "Could not probe" must stay distinct from "not supported": a caller that
	// conflated them would block launches on an unreadable file.
	if sup, probed := BackendSupportsArch("/nonexistent/llama-server", "laguna"); sup || probed {
		t.Errorf("missing binary: got (%v,%v), want (false,false)", sup, probed)
	}
	if sup, probed := BackendSupportsArch("", "laguna"); sup || probed {
		t.Errorf("empty path: got (%v,%v), want (false,false)", sup, probed)
	}
	// Too short to be matched safely.
	real := filepath.Join(t.TempDir(), "llama-server")
	if err := os.WriteFile(real, []byte("\x00a\x00"), 0o600); err != nil {
		t.Fatal(err)
	}
	if sup, probed := BackendSupportsArch(real, "a"); sup || probed {
		t.Errorf("short arch: got (%v,%v), want (false,false)", sup, probed)
	}
}

func TestMainRouteCatalogExcludesHelperOnlyNanoBeige(t *testing.T) {
	if got := RecipesForArch("nanbeige"); len(got) != 0 {
		t.Fatalf("helper-only NanoBeige recipe was offered as a main-model route: %#v", got)
	}
	if got := RecipesForArch("laguna"); len(got) == 0 {
		t.Fatal("main-routable recipes disappeared while filtering helpers")
	}

	appHome := t.TempDir()
	t.Setenv("LLM_APP_HOME", appHome)
	if err := Save([]Backend{{Tag: "helper", RouteArch: "nanbeige", HelperOnly: true}, {Tag: "main", RouteArch: "nanbeige"}}); err != nil {
		t.Fatal(err)
	}
	got := RegisteredForArch("nanbeige")
	if len(got) != 1 || got[0].Tag != "main" {
		t.Fatalf("registered main route catalog leaked helper: %#v", got)
	}
}

func TestFindLibInBuildTree(t *testing.T) {
	// The ik_llama layout: llama-server in build/bin, libllama.so in build/src.
	root := t.TempDir()
	srcDir := filepath.Join(root, "src")
	if err := os.MkdirAll(filepath.Join(root, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(srcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	lib := filepath.Join(srcDir, "libllama.so")
	if err := os.WriteFile(lib, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := findLibInBuildTree(root, "libllama.so"); got != lib {
		t.Errorf("findLibInBuildTree = %q, want %q", got, lib)
	}
	if got := findLibInBuildTree(root, "libabsent.so"); got != "" {
		t.Errorf("absent lib = %q, want empty", got)
	}
}

// TestRealBackendProbe checks the probe against backends actually built from
// source, which is the only way to cover what it exists to do: follow a fork's
// library graph to the object holding the architecture table. Synthetic files
// cannot reproduce that, so the test runs against a real build tree when
// GGRUN_TEST_BACKEND_SRC points at one (the directory holding the llama.cpp and
// fork checkouts) and skips otherwise.
func TestRealBackendProbe(t *testing.T) {
	root := os.Getenv("GGRUN_TEST_BACKEND_SRC")
	if root == "" {
		t.Skip("set GGRUN_TEST_BACKEND_SRC to a directory of built llama.cpp checkouts to run this")
	}
	var (
		lag = filepath.Join(root, "fork-llama.cpp-add-laguna/build-cuda/bin/llama-server")
		ik  = filepath.Join(root, "ik_llama.cpp/build/bin/llama-server")
		hy3 = filepath.Join(root, "fork-ik-llama-hy3-hy3-support/build-cuda/bin/llama-server")
		mai = filepath.Join(root, "llama.cpp/build-cuda/bin/llama-server")
	)
	cases := []struct {
		label, path, arch string
		want              bool
	}{
		{"laguna fork serves laguna", lag, "laguna", true},
		{"laguna fork serves dflash", lag, "dflash", true},
		{"laguna fork keeps mainline arches", lag, "qwen3moe", true},
		{"unknown arch stays unknown", lag, "bogusarch", false},
		// The case the feature exists for: mainline cannot serve laguna.
		{"mainline lacks laguna", mai, "laguna", false},
		{"mainline has generic dflash", mai, "dflash", true},
		{"ik_llama serves qwen3moe", ik, "qwen3moe", true},
		{"hy3 fork serves hy_v3", hy3, "hy_v3", true},
		// Over-report from the tokenizer pre-type list; see the file comment.
		{"ik_llama over-reports laguna", ik, "laguna", true},
	}
	for _, c := range cases {
		if _, err := os.Stat(c.path); err != nil {
			t.Skipf("backend not built on this machine: %s", c.path)
		}
		sup, probed := BackendSupportsArch(c.path, c.arch)
		if !probed {
			t.Errorf("%s: probe failed on a readable binary", c.label)
			continue
		}
		if sup != c.want {
			t.Errorf("%s: supported=%v, want %v", c.label, sup, c.want)
		}
	}
}
