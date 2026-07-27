// Package backends manages user-registered llama.cpp fork backends. New model
// architectures usually land in forks before mainline, so ggrun lets the user
// add one (`ggrun backend add <git-url>`) and optionally route a model
// architecture to it automatically. The manifest is shared by the CLI (backend
// selection/routing) and the TUI (backend picker).
package backends

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Backend is a registered fork backend.
type Backend struct {
	Tag       string `json:"tag"`                  // selection name (--backend <tag>)
	Path      string `json:"path"`                 // path to the built llama-server binary
	RouteArch string `json:"route_arch,omitempty"` // auto-select for models of this arch
	GitURL    string `json:"git_url,omitempty"`
	Branch    string `json:"branch,omitempty"`
	Commit    string `json:"commit,omitempty"`
	// AppliedPatches identifies reviewed source fixes applied while building the
	// backend. It makes a recipe-built binary auditable from backends.json.
	AppliedPatches []string `json:"applied_patches,omitempty"`
	// Previous is the build this backend replaced, kept so a bad update can be
	// undone. One level is deliberate: the value of a rollback is getting back
	// to the build that was working an hour ago, and each retained build costs a
	// full compiled tree on disk.
	Previous *BackendVersion `json:"previous,omitempty"`
}

// BackendVersion is a superseded build, retained only if its binary still
// exists. Recording a path whose tree had been rebuilt over would make rollback
// claim a recovery it cannot perform.
type BackendVersion struct {
	Path           string   `json:"path"`
	Commit         string   `json:"commit,omitempty"`
	AppliedPatches []string `json:"applied_patches,omitempty"`
	ReplacedAt     string   `json:"replaced_at,omitempty"`
}

// RecipePatch is a narrow, reviewed source correction applied to a pinned fork.
// Its contents are embedded in ggrun so an install is deterministic and does not
// depend on an unrecorded edit in the user's fork checkout.
type RecipePatch struct {
	Name     string
	contents []byte
}

// Recipe is a reviewed, reproducible fork integration. Recipes keep new-model
// support declarative: the CLI owns clone/build/register/routing once, while a
// model-specific entry supplies only source identity and architecture.
type Recipe struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Tag         string        `json:"tag"`
	GitURL      string        `json:"git_url"`
	Branch      string        `json:"branch"`
	Commit      string        `json:"commit"`
	RouteArch   string        `json:"route_arch"`
	Accel       string        `json:"accel,omitempty"`
	Patches     []RecipePatch `json:"-"`
}

//go:embed patches/hy3/0001-fix-router-tensor-name.patch
var hy3RouterTensorNamePatch []byte

// common/speculative.cpp calls std::isfinite without including <cmath>. The
// declaration reaches poolside's compiler transitively and ours not at all, so
// the fork builds for them and fails here at 95% with "'isfinite' is not a
// member of 'std'". Adding the include is the whole fix and changes no
// behaviour.
//
//go:embed patches/laguna/0001-include-cmath-for-isfinite.patch
var lagunaCmathPatch []byte

var builtinRecipes = []Recipe{
	{
		Name:        "hy3",
		Description: "Tencent Hy3 / hy_v3 support with built-in MTP and its reviewed chat template",
		Tag:         "hy3",
		GitURL:      "https://github.com/noonr48/ik_llama-hy3.git",
		Branch:      "hy3-support",
		Commit:      "f46c95ee90d8c8200b0147c646b883405020b482",
		RouteArch:   "hy_v3",
		Accel:       "",
		Patches: []RecipePatch{{
			Name:     "hy3/0001-fix-router-tensor-name",
			contents: hy3RouterTensorNamePatch,
		}},
	},
	{
		Name:        "minimax-m3",
		Description: "Preliminary MiniMax-M3 support with native structured tool-call parsing",
		Tag:         "minimax-m3",
		GitURL:      "https://github.com/danielhanchen/llama.cpp.git",
		Branch:      "minimax-m3",
		Commit:      "66f43aa655a07999c7746fe9ff5ede94835e921e",
		RouteArch:   "minimax-m3",
		Accel:       "",
	},
	{
		// Poolside's own fork, not the upstream pull request.
		//
		// The upstream PR (joerowell/llama.cpp @ add-laguna) serves Laguna
		// correctly but implements only the target architecture. Its loader
		// builds 69 of the 76 tensors the published DFlash drafter contains, so
		// any speculative launch dies during model load:
		//
		//   done_getting_tensors: wrong number of tensors; expected 76, got 69
		//
		// Poolside state it directly: "Upstream llama.cpp ships the generic
		// DFlash framework but not the Laguna decoder contract this draft model
		// needs." Speculation is the largest measured lever on a RAM-resident
		// MoE -- a forward pass costs nearly the same carrying one token or a
		// micro-batch -- so a recipe that cannot load the drafter forfeits it.
		Name:        "laguna",
		Description: "Poolside Laguna support including the DFlash drafter (poolside fork, not the upstream PR)",
		Tag:         "laguna",
		GitURL:      "https://github.com/poolsideai/llama.cpp.git",
		Branch:      "laguna",
		Commit:      "04b2b72cb54048ead292884adbe11f284e3ec950",
		RouteArch:   "laguna",
		Accel:       "",
		Patches: []RecipePatch{{
			Name:     "laguna/0001-include-cmath-for-isfinite",
			contents: lagunaCmathPatch,
		}},
	},
	{
		// Kept selectable: it is the branch heading for upstream, so it is the
		// one to test against when the PR merges, and it remains a working
		// target-only fallback if the poolside fork regresses.
		Name:        "laguna-upstream-pr",
		Description: "Laguna target architecture from the open upstream llama.cpp PR (no DFlash drafter)",
		Tag:         "laguna-upstream-pr",
		GitURL:      "https://github.com/joerowell/llama.cpp.git",
		Branch:      "add-laguna",
		Commit:      "54f214a09b8c4e709357ae661a77925edb154f13",
		RouteArch:   "laguna",
		Accel:       "",
	},
}

// Recipes returns a copy of the reviewed built-in recipe catalog.
func Recipes() []Recipe {
	return append([]Recipe(nil), builtinRecipes...)
}

// RecipeByName resolves a recipe by name or tag, case-insensitively.
func RecipeByName(name string) *Recipe {
	name = strings.ToLower(strings.TrimSpace(name))
	for _, recipe := range builtinRecipes {
		if strings.ToLower(recipe.Name) == name || strings.ToLower(recipe.Tag) == name {
			copy := recipe
			return &copy
		}
	}
	return nil
}

// PatchNames returns stable IDs for the reviewed patches included by a recipe.
func (r Recipe) PatchNames() []string {
	names := make([]string, 0, len(r.Patches))
	for _, patch := range r.Patches {
		names = append(names, patch.Name)
	}
	return names
}

// ApplyPatches applies a recipe's reviewed source corrections. It is
// idempotent: a patch that is already present is left untouched, while an
// unexpected source mismatch fails loudly instead of silently building an
// unverified backend.
func (r Recipe) ApplyPatches(srcDir string) error {
	for _, patch := range r.Patches {
		if len(patch.contents) == 0 {
			return fmt.Errorf("recipe patch %q is empty", patch.Name)
		}
		if err := gitApply(srcDir, patch.contents, false, true); err == nil {
			if err := gitApply(srcDir, patch.contents, false, false); err != nil {
				return fmt.Errorf("apply recipe patch %q: %w", patch.Name, err)
			}
			continue
		}
		if err := gitApply(srcDir, patch.contents, true, true); err == nil {
			continue // exact patch already present
		}
		return fmt.Errorf("recipe patch %q does not apply to this checkout", patch.Name)
	}
	return nil
}

// RevertPatches removes only the exact reviewed patches before a pinned
// checkout is refreshed. Any other local edits remain and are rejected by the
// caller, preserving the user's work.
func (r Recipe) RevertPatches(srcDir string) error {
	for i := len(r.Patches) - 1; i >= 0; i-- {
		patch := r.Patches[i]
		if err := gitApply(srcDir, patch.contents, true, true); err != nil {
			return fmt.Errorf("local edits do not match managed recipe patch %q", patch.Name)
		}
		if err := gitApply(srcDir, patch.contents, true, false); err != nil {
			return fmt.Errorf("revert recipe patch %q: %w", patch.Name, err)
		}
	}
	return nil
}

func gitApply(srcDir string, patch []byte, reverse, check bool) error {
	args := []string{"apply"}
	if reverse {
		args = append(args, "--reverse")
	}
	if check {
		args = append(args, "--check")
	}
	cmd := exec.Command("git", args...)
	cmd.Dir = srcDir
	cmd.Stdin = bytes.NewReader(patch)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git %s: %s", strings.Join(args, " "), strings.TrimSpace(string(out)))
	}
	return nil
}

// PointerFile records the app home for binaries that cannot derive it. The
// installer's default target is ~/.local/bin, whose parent ~/.local holds no
// ggrun state, so a second copy of ggrun installed there would otherwise see no
// models and no registered backends while the real install sat untouched
// elsewhere. Silently falling back to defaults is worse than any error: it
// looks like the install lost its configuration.
const PointerFile = "apphome"

// HasState reports whether dir is a real ggrun app home rather than a directory
// that merely happens to contain the binary.
func HasState(dir string) bool {
	if dir == "" {
		return false
	}
	for _, marker := range []string{
		filepath.Join(dir, ".config", "backends.json"),
		filepath.Join(dir, ".config", "config"),
		filepath.Join(dir, ".config", "ggrun", "config"),
	} {
		if st, err := os.Stat(marker); err == nil && !st.IsDir() {
			return true
		}
	}
	return false
}

// pointerPath is where the app home pointer lives, always under the user's own
// config directory so every binary can find it regardless of install location.
func pointerPath() string {
	home := os.Getenv("HOME")
	if home == "" {
		return ""
	}
	return filepath.Join(home, ".config", "ggrun", PointerFile)
}

// RecordAppHome saves the app home so any other ggrun binary resolves the same
// install. Best effort: failing to write it only costs discoverability.
func RecordAppHome(dir string) {
	p := pointerPath()
	if p == "" || dir == "" || dir == os.Getenv("HOME") || !HasState(dir) {
		return
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return
	}
	_ = os.WriteFile(p, []byte(dir+"\n"), 0o644)
}

// pointedAppHome returns the recorded app home if it still holds ggrun state.
func pointedAppHome() string {
	p := pointerPath()
	if p == "" {
		return ""
	}
	body, err := os.ReadFile(p)
	if err != nil {
		return ""
	}
	dir := strings.TrimSpace(string(body))
	if HasState(dir) {
		return dir
	}
	return ""
}

// maxDiscoveryDirs bounds the search so startup cost stays predictable on a
// home directory with many entries.
const maxDiscoveryDirs = 400

// AppHomeFromExe derives the app home from the running binary's location, or
// returns "" when that location says nothing about it.
//
// The parent of a .bin/bin directory is only an app home if it actually holds
// ggrun state. The installer's default target is ~/.local/bin, and treating
// ~/.local as an app home made a second copy of ggrun report no models and no
// registered backends while the real install sat untouched elsewhere.
//
// Split out from AppHome so it is testable: os.Executable() inside a test
// returns the test binary, which can never exercise this path.
func AppHomeFromExe(exe string) string {
	if exe == "" {
		return ""
	}
	exeDir := filepath.Dir(exe)
	switch filepath.Base(exeDir) {
	case ".bin", "bin":
		if parent := filepath.Dir(exeDir); HasState(parent) {
			return parent
		}
	}
	return ""
}

// DiscoverAppHome looks for a ggrun install tree when nothing else identifies
// one. It checks the user's home directory and one level below it, which covers
// both ~/ggrun-productions and the common ~/<project>/ggrun-productions layout,
// and stops at the first directory holding real ggrun state.
//
// This is what keeps a fresh shell, a second install, or a machine that lost an
// exported LLM_APP_HOME from presenting an empty ggrun with no models and no
// registered backends.
func DiscoverAppHome() string {
	home := os.Getenv("HOME")
	if home == "" {
		return ""
	}
	// Named layouts first: cheap, and correct in the overwhelming majority of
	// installs.
	for _, name := range []string{"ggrun-productions", ".ggrun", filepath.Join(".local", "share", "ggrun")} {
		if dir := filepath.Join(home, name); HasState(dir) {
			return dir
		}
	}
	// Then the home directory, then the usual places an install can be moved
	// to, including mounted volumes. The budget is shared across every root so
	// total work stays bounded no matter how many roots exist.
	budget := maxDiscoveryDirs
	roots := append([]string{home}, systemSearchRoots()...)
	for _, root := range roots {
		if dir := scanForState(root, 2, &budget); dir != "" {
			return dir
		}
	}
	return ""
}

// systemSearchRoots are the non-home locations an install may have been moved
// to. Deliberately a short list rather than a whole-filesystem walk: an
// unbounded scan would be slow on every startup and could wander into network
// or removable mounts.
func systemSearchRoots() []string {
	return []string{"/opt", "/srv", "/mnt", "/media", "/usr/local/share"}
}

// scanForState looks for a ggrun app home under root, up to depth levels deep,
// consuming from a shared budget.
func scanForState(root string, depth int, budget *int) string {
	if depth <= 0 || *budget <= 0 {
		return ""
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return ""
	}
	var subdirs []string
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		// Skip dot directories: ggrun's own state lives in a .config inside a
		// normal directory, never in a hidden top-level one, and descending
		// into caches and VCS metadata wastes the budget.
		if strings.HasPrefix(name, ".") {
			continue
		}
		if *budget--; *budget <= 0 {
			return ""
		}
		dir := filepath.Join(root, name)
		if HasState(dir) {
			return dir
		}
		subdirs = append(subdirs, dir)
	}
	for _, dir := range subdirs {
		if found := scanForState(dir, depth-1, budget); found != "" {
			return found
		}
	}
	return ""
}

// AppHome resolves the ggrun app home (holds .bin, .src, .config).
//
// Order: LLM_APP_HOME, then the parent of the .bin/bin directory the running
// binary lives in but only when that parent actually holds ggrun state, then
// the recorded pointer, then a bounded search of the home directory.
//
// A binary sitting in a plain install directory such as ~/.local/bin -- the
// installer's own default -- does not get to claim an app home. Before this
// ordering it did, which made a second copy of ggrun report no models and no
// backends while the real install sat untouched, indistinguishable from a lost
// configuration.
func AppHome() string {
	if h := os.Getenv("LLM_APP_HOME"); h != "" {
		return h
	}
	if exe, err := os.Executable(); err == nil {
		if home := AppHomeFromExe(exe); home != "" {
			return home
		}
	}
	if p := pointedAppHome(); p != "" {
		return p
	}
	if d := DiscoverAppHome(); d != "" {
		// Remember it so the next run skips the search entirely.
		RecordAppHome(d)
		return d
	}
	if h := os.Getenv("HOME"); h != "" {
		return h
	}
	return "."
}

// ManifestPath is the on-disk location of the backends manifest.
func ManifestPath() string {
	return filepath.Join(AppHome(), ".config", "backends.json")
}

// Load returns the registered fork backends (empty if none/unreadable).
func Load() []Backend {
	data, err := os.ReadFile(ManifestPath())
	if err != nil {
		return nil
	}
	var list []Backend
	if json.Unmarshal(data, &list) != nil {
		return nil
	}
	return list
}

// Save writes the manifest.
func Save(list []Backend) error {
	p := ManifestPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(p, data, 0o644)
}

// ByTag returns the registered backend with this tag (case-insensitive), or nil.
func ByTag(tag string) *Backend {
	tag = strings.TrimSpace(strings.ToLower(tag))
	if tag == "" {
		return nil
	}
	list := Load()
	for i := range list {
		if strings.ToLower(list[i].Tag) == tag {
			return &list[i]
		}
	}
	return nil
}

// ForArch returns a registered backend that routes this model architecture,
// if its binary exists on disk. Case-insensitive on arch.
func ForArch(arch string) *Backend {
	arch = strings.TrimSpace(strings.ToLower(arch))
	if arch == "" {
		return nil
	}
	list := Load()
	for i := range list {
		if strings.ToLower(list[i].RouteArch) != arch {
			continue
		}
		if _, err := os.Stat(list[i].Path); err == nil {
			return &list[i]
		}
	}
	return nil
}

// Upsert adds or replaces a backend by tag.
func Upsert(be Backend) error {
	list := Load()
	for i := range list {
		if strings.EqualFold(list[i].Tag, be.Tag) {
			list[i] = be
			return Save(list)
		}
	}
	return Save(append(list, be))
}

// Remove drops a backend by tag; returns false if not found.
func Remove(tag string) (bool, error) {
	list := Load()
	out := list[:0:0]
	found := false
	for _, b := range list {
		if strings.EqualFold(b.Tag, tag) {
			found = true
			continue
		}
		out = append(out, b)
	}
	if !found {
		return false, nil
	}
	return found, Save(out)
}

// Tags returns the registered backend tags (for pickers).
func Tags() []string {
	list := Load()
	tags := make([]string, 0, len(list))
	for _, b := range list {
		tags = append(tags, b.Tag)
	}
	return tags
}
