package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"
)

// Updating a backend used to mean reinstalling it, which rebuilt over the only
// working binary. When the new revision failed to build -- the poolside Laguna
// fork did exactly that, dying at 95% on a missing include -- there was nothing
// to go back to, and the machine was left with no backend for the architecture
// at all.
//
// So an update preserves the build it replaces instead of overwriting it. The
// old tree is renamed aside before the new one is built, which is a rename on
// the same filesystem rather than a copy of several hundred megabytes, and the
// registration only moves once the new build has produced a binary.
//
// Nothing here is specific to a model or a fork: the source directory,
// accelerator and pin all come from the backend's own record, so any registered
// backend built from source can be updated and rolled back the same way.

// backendBuildLayout recovers where a backend was built from its binary path,
// which is the only thing backends.json records. The install path builds into
// <srcDir>/build-<accel>/bin/llama-server, so the layout is recoverable, and a
// path not matching that shape means the backend was registered by hand and
// cannot be rebuilt.
type backendBuildLayout struct {
	srcDir   string
	buildDir string
	accel    string
}

func backendLayoutFor(be backends.Backend) (backendBuildLayout, error) {
	if be.Path == "" {
		return backendBuildLayout{}, fmt.Errorf("backend %q has no binary path", be.Tag)
	}
	binDir := filepath.Dir(be.Path)
	if filepath.Base(binDir) != "bin" {
		return backendBuildLayout{}, fmt.Errorf("backend %q was not built by ggrun (binary is not in a build-*/bin directory)", be.Tag)
	}
	buildDir := filepath.Dir(binDir)
	name := filepath.Base(buildDir)
	if !strings.HasPrefix(name, "build-") {
		return backendBuildLayout{}, fmt.Errorf("backend %q was not built by ggrun (found %q, expected build-<accel>)", be.Tag, name)
	}
	return backendBuildLayout{
		srcDir:   filepath.Dir(buildDir),
		buildDir: buildDir,
		accel:    strings.TrimPrefix(name, "build-"),
	}, nil
}

// archiveBuildDir moves a build aside so the new one can take its place. The
// commit is part of the name so an archived tree is identifiable, and a
// collision is resolved rather than overwritten -- silently discarding the one
// build that still works would defeat the entire point.
func archiveBuildDir(buildDir, commit string) (string, error) {
	short := strings.TrimSpace(commit)
	if len(short) > 12 {
		short = short[:12]
	}
	if short == "" {
		short = "unknown"
	}
	base := buildDir + "-prev-" + short
	dest := base
	for i := 1; ; i++ {
		if _, err := os.Stat(dest); os.IsNotExist(err) {
			break
		}
		dest = fmt.Sprintf("%s.%d", base, i)
		if i > 50 {
			return "", fmt.Errorf("too many archived builds beside %s", buildDir)
		}
	}
	if err := os.Rename(buildDir, dest); err != nil {
		return "", fmt.Errorf("archive previous build: %w", err)
	}
	return dest, nil
}

// pruneBuildDir removes a superseded build tree. It refuses anything that is
// not one of our own archives, because this deletes a directory outright.
func pruneBuildDir(path string) error {
	if path == "" || !strings.Contains(filepath.Base(path), "-prev-") {
		return fmt.Errorf("refusing to remove %q: not a ggrun archived build", path)
	}
	return os.RemoveAll(path)
}

// cmdBackendUpdate rebuilds a backend at the newest revision of its recipe (or
// its recorded branch), keeping the current build so the change is reversible.
func cmdBackendUpdate(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "update needs a backend tag; see: ggrun backend list")
		os.Exit(2)
	}
	tag := args[0]
	be := backends.ByTag(tag)
	if be == nil {
		fmt.Fprintf(os.Stderr, "unknown backend %q; see: ggrun backend list\n", tag)
		os.Exit(2)
	}
	layout, err := backendLayoutFor(*be)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot update: %v\n", err)
		os.Exit(1)
	}
	if _, err := os.Stat(layout.srcDir); err != nil {
		fmt.Fprintf(os.Stderr, "cannot update: source checkout %s is gone; reinstall instead\n", layout.srcDir)
		os.Exit(1)
	}

	// A recipe is the authority on where a backend should be: it carries the
	// reviewed commit pin and the patches. Falling back to the record's own
	// branch keeps hand-added forks updatable too.
	recipe := backends.RecipeByName(tag)
	branch, commit := be.Branch, be.Commit
	if recipe != nil {
		branch, commit = recipe.Branch, recipe.Commit
		if commit != "" && strings.EqualFold(commit, be.Commit) {
			fmt.Printf("[backend] %s is already at the reviewed commit %s; nothing to update.\n", tag, shortCommit(commit))
			return
		}
	} else if commit != "" {
		// A hand-pinned backend has nothing to advance to. Tracking the branch
		// instead would silently discard the pin the user chose.
		fmt.Printf("[backend] %s is pinned to commit %s with no recipe to advance it.\n", tag, shortCommit(commit))
		fmt.Println("[backend] re-add it with a new --commit to move the pin.")
		return
	}

	fmt.Printf("[backend] updating %s (%s)\n", tag, layout.srcDir)
	if err := prepareForkCheckoutRecipe(layout.srcDir, branch, commit, recipe); err != nil {
		fmt.Fprintf(os.Stderr, "source checkout failed: %v\n", err)
		os.Exit(1)
	}
	newCommit, _ := gitOutput(layout.srcDir, "rev-parse", "HEAD")
	newCommit = strings.TrimSpace(newCommit)
	if newCommit != "" && strings.EqualFold(newCommit, be.Commit) {
		fmt.Printf("[backend] %s already at %s; nothing to rebuild.\n", tag, shortCommit(newCommit))
		return
	}

	archived := ""
	if _, err := os.Stat(layout.buildDir); err == nil {
		archived, err = archiveBuildDir(layout.buildDir, be.Commit)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		fmt.Printf("[backend] previous build kept at %s\n", archived)
	}

	fmt.Printf("[backend] building (%s)… this can take 30–60 min\n", layout.accel)
	bin, err := buildLlamaFork(layout.srcDir, layout.accel, "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "build failed: %v\n", err)
		// The registration still points at the archived build, so restoring it
		// leaves the machine exactly as the update found it.
		if archived != "" {
			if rerr := os.Rename(archived, layout.buildDir); rerr == nil {
				fmt.Fprintf(os.Stderr, "[backend] restored previous build; %s still works.\n", tag)
			} else {
				fmt.Fprintf(os.Stderr, "[backend] could not restore previous build: %v\n", rerr)
				fmt.Fprintf(os.Stderr, "[backend] it is intact at %s\n", archived)
			}
		}
		os.Exit(1)
	}

	prev := &backends.BackendVersion{
		Path:           filepath.Join(archived, "bin", filepath.Base(be.Path)),
		Commit:         be.Commit,
		AppliedPatches: be.AppliedPatches,
		ReplacedAt:     time.Now().UTC().Format(time.RFC3339),
	}
	if archived == "" {
		prev = nil
	}
	updated := *be
	updated.Path = bin
	updated.Commit = newCommit
	updated.Previous = prev
	if recipe != nil {
		updated.AppliedPatches = recipe.PatchNames()
		if recipe.GitURL != "" {
			updated.GitURL = recipe.GitURL
		}
		updated.Branch = recipe.Branch
	}
	if err := backends.Upsert(updated); err != nil {
		fmt.Fprintf(os.Stderr, "register failed: %v\n", err)
		os.Exit(1)
	}
	// Retention is one level, so the tree this update pushed out of history is
	// now unreachable and would otherwise accumulate a compiled checkout per
	// update. It is only removed after the new build succeeded and registered.
	if be.Previous != nil && be.Previous.Path != "" {
		stale := filepath.Dir(filepath.Dir(be.Previous.Path))
		if err := pruneBuildDir(stale); err == nil {
			fmt.Printf("[backend] removed superseded build %s\n", stale)
		}
	}
	fmt.Printf("Updated backend %q → %s\n", tag, shortCommit(newCommit))
	if prev != nil {
		fmt.Printf("Roll back with: ggrun backend rollback %s\n", tag)
	}
}

// cmdBackendRollback re-points a backend at the build it replaced.
func cmdBackendRollback(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "rollback needs a backend tag; see: ggrun backend list")
		os.Exit(2)
	}
	tag := args[0]
	be := backends.ByTag(tag)
	if be == nil {
		fmt.Fprintf(os.Stderr, "unknown backend %q; see: ggrun backend list\n", tag)
		os.Exit(2)
	}
	if be.Previous == nil || be.Previous.Path == "" {
		fmt.Fprintf(os.Stderr, "no previous build recorded for %q; nothing to roll back to.\n", tag)
		os.Exit(1)
	}
	if _, err := os.Stat(be.Previous.Path); err != nil {
		fmt.Fprintf(os.Stderr, "previous build for %q is gone (%s); reinstall instead.\n", tag, be.Previous.Path)
		os.Exit(1)
	}

	// The swap is symmetric so a rollback can itself be undone: someone who
	// rolls back to escape one failure and meets a worse one can get back.
	current := &backends.BackendVersion{
		Path:           be.Path,
		Commit:         be.Commit,
		AppliedPatches: be.AppliedPatches,
		ReplacedAt:     time.Now().UTC().Format(time.RFC3339),
	}
	restored := *be
	restored.Path = be.Previous.Path
	restored.Commit = be.Previous.Commit
	restored.AppliedPatches = be.Previous.AppliedPatches
	restored.Previous = current
	if err := backends.Upsert(restored); err != nil {
		fmt.Fprintf(os.Stderr, "register failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Rolled back backend %q → %s\n", tag, restored.Path)
	if restored.Commit != "" {
		fmt.Printf("  commit: %s\n", shortCommit(restored.Commit))
	}
	fmt.Printf("Undo with: ggrun backend rollback %s\n", tag)
}

func shortCommit(c string) string {
	c = strings.TrimSpace(c)
	if len(c) > 12 {
		return c[:12]
	}
	if c == "" {
		return "(unknown)"
	}
	return c
}
