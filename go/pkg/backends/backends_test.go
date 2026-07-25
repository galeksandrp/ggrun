package backends

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestHY3RecipeIsPinnedAndRouted(t *testing.T) {
	recipe := RecipeByName("hy3")
	if recipe == nil {
		t.Fatal("HY3 recipe missing")
	}
	if recipe.RouteArch != "hy_v3" {
		t.Fatalf("HY3 route arch = %q, want hy_v3", recipe.RouteArch)
	}
	if recipe.Branch != "hy3-support" || len(recipe.Commit) != 40 {
		t.Fatalf("HY3 source is not reproducibly pinned: %#v", recipe)
	}
	if recipe.GitURL != "https://github.com/noonr48/ik_llama-hy3.git" {
		t.Fatalf("unexpected HY3 fork: %s", recipe.GitURL)
	}
	if got := recipe.PatchNames(); len(got) != 1 || got[0] != "hy3/0001-fix-router-tensor-name" {
		t.Fatalf("HY3 recipe patches = %#v", got)
	}

	// Callers receive a copy and cannot mutate the built-in catalog.
	recipe.Commit = "changed"
	again := RecipeByName("HY3")
	if again == nil || again.Commit == "changed" {
		t.Fatal("recipe lookup leaked mutable catalog state")
	}
}

func TestMiniMaxM3RecipeIsPinnedAndRouted(t *testing.T) {
	recipe := RecipeByName("minimax-m3")
	if recipe == nil {
		t.Fatal("MiniMax-M3 recipe missing")
	}
	if recipe.RouteArch != "minimax-m3" {
		t.Fatalf("MiniMax-M3 route arch = %q, want minimax-m3", recipe.RouteArch)
	}
	if recipe.Branch != "minimax-m3" || len(recipe.Commit) != 40 {
		t.Fatalf("MiniMax-M3 source is not reproducibly pinned: %#v", recipe)
	}
	if recipe.GitURL != "https://github.com/danielhanchen/llama.cpp.git" {
		t.Fatalf("unexpected MiniMax-M3 fork: %s", recipe.GitURL)
	}
	if got := recipe.PatchNames(); len(got) != 0 {
		t.Fatalf("MiniMax-M3 recipe unexpectedly carries patches: %#v", got)
	}

	byTag := RecipeByName("MiniMax-M3")
	if byTag == nil || byTag.Commit != recipe.Commit {
		t.Fatal("MiniMax-M3 tag lookup did not resolve the pinned recipe")
	}
}

func TestLagunaRecipeIsPinnedAndRouted(t *testing.T) {
	recipe := RecipeByName("laguna")
	if recipe == nil {
		t.Fatal("Laguna recipe missing")
	}
	if recipe.RouteArch != "laguna" || recipe.Branch != "add-laguna" {
		t.Fatalf("unexpected Laguna route/branch: %#v", recipe)
	}
	if recipe.GitURL != "https://github.com/joerowell/llama.cpp.git" || len(recipe.Commit) != 40 {
		t.Fatalf("Laguna source is not reproducibly pinned: %#v", recipe)
	}
}

func TestHY3RecipePatchAppliesAndRevertsCleanly(t *testing.T) {
	recipe := RecipeByName("hy3")
	if recipe == nil {
		t.Fatal("HY3 recipe missing")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "src", "llama-model.cpp")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	original := `static const std::map<llm_arch, std::map<llm_tensor, std::string>> LLM_TENSOR_NA
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
`
	if err := os.WriteFile(target, []byte(original), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := recipe.ApplyPatches(dir); err != nil {
		t.Fatalf("apply HY3 patch: %v", err)
	}
	patched, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(patched), `LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp"`) {
		t.Fatalf("router tensor name was not patched:\n%s", patched)
	}
	if err := recipe.ApplyPatches(dir); err != nil {
		t.Fatalf("reapplying HY3 patch must be idempotent: %v", err)
	}
	if err := recipe.RevertPatches(dir); err != nil {
		t.Fatalf("revert HY3 patch: %v", err)
	}
	restored, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(restored) != original {
		t.Fatalf("patch revert changed source:\n%s", restored)
	}
}

// The installer's default target is ~/.local/bin, whose parent holds no ggrun
// state. Claiming it as an app home made a second copy of ggrun report no
// models and no backends while the real install sat untouched.
func TestAppHomeFromExeIgnoresAStatelessBinParent(t *testing.T) {
	root := t.TempDir()
	stateless := filepath.Join(root, ".local")
	if err := os.MkdirAll(filepath.Join(stateless, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := AppHomeFromExe(filepath.Join(stateless, "bin", "ggrun")); got != "" {
		t.Errorf("AppHomeFromExe claimed a stateless bin parent: %q", got)
	}
}

// A real self-contained install must still be recognised from its own binary,
// with no environment variable set.
func TestAppHomeFromExeAcceptsARealInstall(t *testing.T) {
	install := t.TempDir()
	if err := os.MkdirAll(filepath.Join(install, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(install, ".config", "backends.json"), []byte("[]"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, binDir := range []string{".bin", "bin"} {
		exe := filepath.Join(install, binDir, "ggrun")
		if got := AppHomeFromExe(exe); got != install {
			t.Errorf("AppHomeFromExe(%s) = %q, want %q", binDir, got, install)
		}
	}
}

// Anything outside a bin directory says nothing about the app home.
func TestAppHomeFromExeIgnoresUnrelatedLocations(t *testing.T) {
	if got := AppHomeFromExe("/home/mik/ggrun-project/ggrun/ggrun"); got != "" {
		t.Errorf("AppHomeFromExe = %q, want empty for a repo-local binary", got)
	}
	if got := AppHomeFromExe(""); got != "" {
		t.Errorf("AppHomeFromExe(\"\") = %q, want empty", got)
	}
}

func TestHasStateRecognisesRealInstallsOnly(t *testing.T) {
	dir := t.TempDir()
	if HasState(dir) {
		t.Error("empty directory reported as an app home")
	}
	if err := os.MkdirAll(filepath.Join(dir, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if HasState(dir) {
		t.Error("bare .config directory reported as an app home")
	}
	if err := os.WriteFile(filepath.Join(dir, ".config", "backends.json"), []byte("[]"), 0o600); err != nil {
		t.Fatal(err)
	}
	if !HasState(dir) {
		t.Error("a directory holding backends.json is an app home")
	}
}

// Discovery is what saves a user whose shell no longer exports LLM_APP_HOME.
func TestDiscoverAppHomeFindsANestedInstall(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	install := filepath.Join(home, "my-project", "ggrun-productions")
	if err := os.MkdirAll(filepath.Join(install, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(install, ".config", "backends.json"), []byte("[]"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := DiscoverAppHome(); got != install {
		t.Errorf("DiscoverAppHome() = %q, want %q", got, install)
	}
}

func TestDiscoverAppHomePrefersTheConventionalLayout(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	for _, dir := range []string{
		filepath.Join(home, "ggrun-productions"),
		filepath.Join(home, "elsewhere", "ggrun-productions"),
	} {
		if err := os.MkdirAll(filepath.Join(dir, ".config"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, ".config", "backends.json"), []byte("[]"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if got := DiscoverAppHome(); got != filepath.Join(home, "ggrun-productions") {
		t.Errorf("DiscoverAppHome() = %q, want the conventional ~/ggrun-productions", got)
	}
}

func TestDiscoverAppHomeReturnsEmptyWhenNoInstallExists(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := os.MkdirAll(filepath.Join(home, "unrelated", "stuff"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := DiscoverAppHome(); got != "" {
		t.Errorf("DiscoverAppHome() = %q, want empty", got)
	}
}

// A recorded pointer must survive the install being deleted without sending
// every later lookup to a dead path.
func TestPointerIsIgnoredWhenTheRecordedInstallIsGone(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("LLM_APP_HOME", "")
	install := filepath.Join(home, "gone")
	if err := os.MkdirAll(filepath.Join(install, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(install, ".config", "backends.json"), []byte("[]"), 0o600); err != nil {
		t.Fatal(err)
	}
	RecordAppHome(install)
	if got := AppHome(); got != install {
		t.Fatalf("AppHome() = %q, want the recorded %q", got, install)
	}
	if err := os.RemoveAll(install); err != nil {
		t.Fatal(err)
	}
	if got := AppHome(); got == install {
		t.Error("AppHome() still points at a deleted install")
	}
}

func TestScanIsBounded(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	for i := 0; i < 60; i++ {
		if err := os.MkdirAll(filepath.Join(home, "d"+strconv.Itoa(i), "sub"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// No install anywhere: must terminate and report nothing rather than walk
	// the tree indefinitely.
	if got := DiscoverAppHome(); got != "" {
		t.Errorf("DiscoverAppHome() = %q, want empty", got)
	}
}
