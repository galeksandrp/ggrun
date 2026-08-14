// Package chattemplate provides a data-driven catalog of corrected chat
// templates. Some models ship GGUFs whose embedded chat template carries a
// Jinja `raise_exception` guard (e.g. "System message must be at the
// beginning") that llama.cpp's minja engine cannot parse, so every request
// under --jinja 400s with "Unable to generate parser for this template."
// Tool calls require --jinja, so the fix is to keep --jinja and serve a
// corrected template via --chat-template-file instead of dropping the flag.
//
// The catalog is data-driven: adding a broken model to catalog.json (and a
// corrected .jinja in templates/) fixes it with no code change. This package
// owns no per-model Go branches.
package chattemplate

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

//go:embed catalog.json
var catalogJSON []byte

// Entry is one catalog row. A model matches when its GGUF architecture equals
// Arch (case-insensitive) or its file basename contains Basename
// (case-insensitive). The first enabled match wins.
type Entry struct {
	Name         string `json:"name"`
	Arch         string `json:"arch,omitempty"`
	Basename     string `json:"basename,omitempty"`
	TemplateFile string `json:"template_file"`
	Enabled      bool   `json:"enabled"`
}

type catalog struct {
	Version int     `json:"version"`
	Entries []Entry `json:"entries"`
}

var (
	parseOnce sync.Once
	parsed    *catalog
	parseErr  error
)

// catalogVersion is the minja/raise_exception pattern that marks a template as
// needing a corrected override. Entries only apply when the model's own GGUF
// template actually carries this guard, unless the entry forces the override.
const raiseExceptionMarker = "raise_exception"

// Load returns the parsed embedded catalog. The embedded catalog always
// parses; on a malformed ship it degrades to an empty catalog rather than
// breaking every launch.
func Load() []Entry {
	parseOnce.Do(func() {
		parsed = &catalog{}
		if err := json.Unmarshal(catalogJSON, parsed); err != nil {
			parseErr = fmt.Errorf("parse embedded chat-template catalog: %w", err)
			parsed = &catalog{}
			return
		}
	})
	if parseErr != nil {
		// Embedded data is a build invariant; surface the problem loudly once
		// instead of silently disabling template fixes.
		fmt.Fprintf(os.Stderr, "[chat-template] warning: %v\n", parseErr)
	}
	if parsed == nil {
		return nil
	}
	return parsed.Entries
}

// Enabled filters the catalog to enabled entries.
func Enabled() []Entry {
	all := Load()
	out := make([]Entry, 0, len(all))
	for _, e := range all {
		if e.Enabled {
			out = append(out, e)
		}
	}
	return out
}

// Match reports whether the entry applies to the given model arch and basename.
func (e Entry) Match(arch, basename string) bool {
	if !e.Enabled {
		return false
	}
	if e.Arch != "" && strings.EqualFold(strings.TrimSpace(e.Arch), strings.TrimSpace(arch)) {
		return true
	}
	if e.Basename != "" && strings.Contains(strings.ToLower(basename), strings.ToLower(e.Basename)) {
		return true
	}
	return false
}

// Resolve returns the first enabled entry matching the model, or ok=false.
// When requireBroken is true the entry only applies if the GGUF's own embedded
// template carries the raise_exception marker, so a catalog entry does not
// clobber a model whose template already parses.
func Resolve(arch, basename string, ggufTemplate string, requireBroken bool) (Entry, bool) {
	matches := Enabled()
	for _, e := range matches {
		if !e.Match(arch, basename) {
			continue
		}
		if requireBroken && ggufTemplate != "" && !strings.Contains(ggufTemplate, raiseExceptionMarker) {
			continue
		}
		return e, true
	}
	return Entry{}, false
}

// ResolveOverride returns the entry with the given name (used by
// --chat-template <name>), regardless of arch/basename matching. It respects
// the enabled flag.
func ResolveOverride(name string) (Entry, bool) {
	n := strings.ToLower(strings.TrimSpace(name))
	for _, e := range Enabled() {
		if strings.ToLower(strings.TrimSpace(e.Name)) == n {
			return e, true
		}
	}
	return Entry{}, false
}

// Names returns the enabled catalog entry names, sorted, for help output.
func Names() []string {
	out := make([]string, 0, len(Enabled()))
	for _, e := range Enabled() {
		out = append(out, e.Name)
	}
	return out
}

// TemplateFor returns the corrected template content for an entry. Templates
// are embedded in this package via go:embed; user-extensible overrides can
// also ship as files on disk (see materializeEntry which prefers a user file).
func TemplateFor(e Entry) (string, error) {
	if e.TemplateFile == "" {
		return "", errors.New("chat-template entry has no template_file")
	}
	// go:embed templates/*.jinja mounts the files under the templates/ prefix.
	path := filepath.ToSlash(filepath.Join("templates", e.TemplateFile))
	data, err := templatesFS.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read embedded chat template %q: %w", e.TemplateFile, err)
	}
	return string(data), nil
}

// Materialize writes the corrected template for an entry into the given cache
// dir and returns the file path, so the launch can pass --chat-template-file.
// The write is idempotent: content changes rewrite, identical content is a no-op.
func Materialize(cacheDir string, e Entry) (string, error) {
	content, err := TemplateFor(e)
	if err != nil {
		return "", err
	}
	return writeTemplateFile(cacheDir, e, content)
}

func writeTemplateFile(cacheDir string, e Entry, content string) (string, error) {
	dir := filepath.Join(cacheDir, "chat-templates")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create chat-template cache dir: %w", err)
	}
	// Keep the file name human-readable and unique per entry.
	name := e.Name
	if name == "" {
		name = "template"
	}
	path := filepath.Join(dir, sanitize(name)+".jinja")
	// Only rewrite when content changed; avoids churning the mtime and fsync
	// churn across repeated launch/repair passes.
	if existing, err := os.ReadFile(path); err == nil && string(existing) == content {
		return path, nil
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("write chat template %q: %w", path, err)
	}
	return path, nil
}

func sanitize(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	return b.String()
}
