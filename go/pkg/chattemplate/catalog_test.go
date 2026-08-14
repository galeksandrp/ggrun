package chattemplate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveMatchesQwen38ByArch(t *testing.T) {
	e, ok := Resolve("qwen35", "Qwen3.8-27B-UD-Q8_K_XL.gguf", "<broken raise_exception>", true)
	if !ok {
		t.Fatal("catalog must resolve a corrected template for qwen35 arch")
	}
	if e.Name != "qwen3.8-27b" {
		t.Fatalf("resolved entry = %q, want qwen3.8-27b", e.Name)
	}
	tpl, err := TemplateFor(e)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(tpl, "<|im_start|>") {
		t.Fatalf("Qwen3.8 corrected template lacks im_start tokens")
	}
	// The specific guard that fires on llama.cpp's injected tool-instruction
	// system message must be gone from the corrected template.
	if strings.Contains(tpl, "System message must be at the beginning") {
		t.Fatalf("corrected Qwen3.8 template must not raise on a late system message")
	}
}

func TestResolveMatchesNanbeigeByBasename(t *testing.T) {
	e, ok := Resolve("nanbeige", "Nanbeige4.2-3B-Q4_K_M.gguf", "<broken raise_exception>", true)
	if !ok {
		t.Fatal("catalog must resolve a corrected template for nanbeige")
	}
	if e.Name != "nanbeige4.2-3b" {
		t.Fatalf("resolved entry = %q, want nanbeige4.2-3b", e.Name)
	}
}

func TestResolveModelWithoutCatalogEntryUntouched(t *testing.T) {
	// A model whose arch/basename has no catalog entry must not resolve.
	if _, ok := Resolve("deepseek4", "DeepSeek-V4-Q4_K_M.gguf", "<broken raise_exception>", true); ok {
		t.Fatal("deepseek4 must not match a catalog entry")
	}
	// Even when the embedded template is broken, no entry means no override.
	if _, ok := Resolve("some-arch", "Whatever.gguf", "<broken raise_exception>", true); ok {
		t.Fatal("unknown model must not resolve a catalog override")
	}
}

func TestResolveRequiresBrokenTemplateUnlessForced(t *testing.T) {
	// An entry whose arch matches but whose GGUF template does NOT carry
	// raise_exception must be skipped when requireBroken=true: the model's own
	// template parses fine and must not be clobbered.
	if _, ok := Resolve("qwen35", "Qwen3.8-27B.gguf", "<healthy template>", true); ok {
		t.Fatal("entry must not apply to a model whose template parses (requireBroken)")
	}
	// requireBroken=false (forced) applies regardless.
	if _, ok := Resolve("qwen35", "Qwen3.8-27B.gguf", "<healthy template>", false); !ok {
		t.Fatal("forced entry must apply regardless of the embedded template")
	}
}

func TestResolveOverrideByName(t *testing.T) {
	e, ok := ResolveOverride("nanbeige4.2-3b")
	if !ok || e.Name != "nanbeige4.2-3b" {
		t.Fatalf("ResolveOverride(nanbeige4.2-3b) = %+v, %v", e, ok)
	}
	if _, ok := ResolveOverride("does-not-exist"); ok {
		t.Fatal("unknown override name must not resolve")
	}
}

func TestMaterializeWritesTemplateFile(t *testing.T) {
	dir := t.TempDir()
	e, ok := Resolve("qwen35", "Qwen3.8-27B.gguf", "<raise_exception>", true)
	if !ok {
		t.Fatal("no catalog entry")
	}
	path, err := Materialize(dir, e)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(path, ".jinja") {
		t.Fatalf("materialized path %q must end in .jinja", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatal("materialized template is empty")
	}
	// Idempotent: second call returns the same path.
	again, err := Materialize(dir, e)
	if err != nil {
		t.Fatal(err)
	}
	if again != path {
		t.Fatalf("materialize must be idempotent: %q != %q", again, path)
	}
}

func TestDisabledEntryNotReturnedByEnabled(t *testing.T) {
	for _, e := range Enabled() {
		if !e.Enabled {
			t.Fatalf("Enabled() returned disabled entry %q", e.Name)
		}
	}
}

func TestMissingOrInvalidTemplateFileGraceful(t *testing.T) {
	bad := Entry{Name: "bad", TemplateFile: "does-not-exist.jinja", Enabled: true}
	if _, err := TemplateFor(bad); err == nil {
		t.Fatal("TemplateFor of a missing template must error")
	}
	if _, err := TemplateFor(Entry{Name: "no-file"}); err == nil {
		t.Fatal("TemplateFor of an entry without template_file must error")
	}
}

func TestLoadParsesEmbeddedCatalog(t *testing.T) {
	entries := Load()
	if len(entries) == 0 {
		t.Fatal("embedded catalog must not be empty")
	}
	for _, e := range entries {
		if e.TemplateFile == "" {
			t.Fatalf("entry %q has no template_file", e.Name)
		}
		if _, err := templatesFS.ReadFile("templates/" + e.TemplateFile); err != nil {
			t.Fatalf("entry %q references missing template %q: %v", e.Name, e.TemplateFile, err)
		}
	}
}

func TestWriteTemplateFileSanitizesName(t *testing.T) {
	dir := t.TempDir()
	e := Entry{Name: "Weird/Name!?.jinja", TemplateFile: "Nanbeige4.2-3B.jinja"}
	content, _ := TemplateFor(e)
	path, err := writeTemplateFile(dir, e, content)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(filepath.Dir(path)) != "chat-templates" {
		t.Fatalf("template must be materialized under chat-templates/: %q", path)
	}
	if base := filepath.Base(path); base != "Weird_Name__.jinja.jinja" {
		t.Fatalf("entry name must be sanitized into the file basename, got %q", base)
	}
}
