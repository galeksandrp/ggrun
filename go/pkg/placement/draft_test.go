package placement

import (
	"os"
	"path/filepath"
	"testing"
)

// A DFlash speculator is a different architecture from its target by design.
// poolside publishes Laguna's companion as general.architecture "dflash", but
// the rule demanded the target's name as a prefix ("laguna-dflash"), so the
// only DFlash drafter that exists for this target was rejected while sitting
// in the model directory.
func TestDFlashCompanionArchitectureNeedNotNameTheTarget(t *testing.T) {
	target := &ModelProfile{ModelArch: "laguna", EmbeddingLength: 3072}
	if !specializedArchitectureCompatibleForBackend(target, "dflash", "dflash", "llama") {
		t.Error("rejected poolside's real DFlash companion architecture")
	}
	if !specializedArchitectureCompatibleForBackend(target, "dflash", "laguna-dflash", "llama") {
		t.Error("rejected a target-prefixed DFlash architecture")
	}
	// A companion that is not a DFlash speculator at all must still be refused;
	// dimensional identity is checked separately by the caller.
	for _, arch := range []string{"laguna", "qwen3moe", "", "unknown"} {
		if specializedArchitectureCompatibleForBackend(target, "dflash", arch, "llama") {
			t.Errorf("accepted non-DFlash architecture %q as a DFlash companion", arch)
		}
	}
}

// ggrun's downloader stores each model in a per-repository directory, and
// quantisations often sit one level deeper again. A flat scan of the model
// directory could never see a companion ggrun itself had placed there, so every
// locally held drafter was invisible and speculation fell through to a download.
func TestSpecCandidateFilesFindsNestedModels(t *testing.T) {
	root := t.TempDir()
	want := map[string]bool{}
	for _, rel := range []string{
		"top.gguf",
		"Repo-GGUF/companion.gguf",
		"Repo-GGUF/UD-Q4_K_XL/shard-00001-of-00003.gguf",
	} {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
		want[p] = true
	}
	// Noise that must not be returned.
	if err := os.WriteFile(filepath.Join(root, "notes.txt"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	got := specCandidateFiles(root)
	for _, p := range got {
		if !want[p] {
			t.Errorf("unexpected candidate %q", p)
		}
		delete(want, p)
	}
	for p := range want {
		t.Errorf("missed candidate %q", p)
	}
}

// Relaxing the name rule must not let a speculator built for one family pair
// with another target just because both mention "dflash".
func TestDFlashCrossFamilyCompanionStillRejected(t *testing.T) {
	deepseek := &ModelProfile{ModelArch: "deepseek4"}
	if specializedArchitectureCompatibleForBackend(deepseek, "dflash", "qwen35-dflash-draft", "llama") {
		t.Error("a Qwen DFlash speculator was accepted for a DeepSeek target")
	}
	if !specializedArchitectureCompatibleForBackend(deepseek, "dflash", "deepseek4-dflash-draft", "llama") {
		t.Error("same-family DFlash architecture was rejected")
	}
	// Bare "dflash" names no family and must fall through to dimensional proof.
	if !specializedArchitectureCompatibleForBackend(deepseek, "dflash", "dflash", "llama") {
		t.Error("bare dflash architecture was rejected")
	}
}
