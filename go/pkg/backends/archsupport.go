package backends

import (
	"bytes"
	"debug/elf"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// A backend that cannot serve a model's architecture fails at model load with a
// message from deep inside the loader, and nothing tells the user that a
// reviewed fork exists which can. On this project a Laguna launch silently
// routed to mainline, whose loader has no laguna architecture at all, and the
// only signal was a tensor-count error.
//
// The check here is a real capability probe rather than a guess: llama.cpp
// registers every architecture as a C string literal in the loader
// (`{ LLM_ARCH_DFLASH, "dflash" }`), so the name is present exactly when the
// build knows the architecture.
//
// Two details decide whether the probe is right or useless:
//
// The literal usually is not in the executable. Mainline-style builds link
// llama-server against libllama, so the executable carries no architecture at
// all while the shared object carries all of them; ik_llama places libllama in
// a sibling directory, and the poolside build reaches libllama only through
// libllama-server-impl. Scanning just the given path reports "unsupported" for
// every architecture on every dynamically linked backend, so the probe follows
// the library graph.
//
// A plain substring is not evidence. Build directories end up embedded in the
// binaries, so a fork built in .../fork-llama.cpp-add-laguna/ contains dozens
// of copies of "laguna" inside path strings -- libggml-cuda alone had 69, none
// of them an architecture. Matching the name NUL-bracketed keeps only genuine
// standalone C literals, which drops every one of those paths while still
// finding each real architecture.
//
// One inaccuracy survives, and it is the safe one. Architecture names are not
// the only NUL-terminated literals in a loader: the ik_llama forks list
// "laguna" among their tokenizer pre-types, so a probe of those backends
// reports the architecture as known when it is not. Nothing distinguishes the
// two tables by content, so the probe over-reports rather than under-reports.
// That direction is deliberate -- an over-report costs the user the suggestion
// they would not have had anyway, while an under-report would talk them out of
// a backend that works. Mainline llama.cpp carries no such literal, so the case
// this feature exists for, a fork-only architecture launched on mainline, still
// probes correctly.

// archProbeChunk is the read size for scanning a file. Backend libraries reach
// hundreds of megabytes; streaming keeps memory bounded regardless.
const archProbeChunk = 4 << 20

// archProbeMaxFiles bounds the dependency walk. Real backends pull in a handful
// of llama objects; the cap only stops a pathological or cyclic graph.
const archProbeMaxFiles = 24

// BackendSupportsArch reports whether a built backend knows an architecture.
//
// The second return value is false when the question could not be answered --
// an unreadable binary, or an architecture name too short to match safely. A
// caller must not treat "could not probe" as "not supported": refusing a launch
// on a failed probe would be worse than the cryptic loader error it replaces.
func BackendSupportsArch(binaryPath, arch string) (supported, probed bool) {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if binaryPath == "" || len(arch) < 2 {
		return false, false
	}
	// Genuine architecture literals are NUL-terminated and follow the previous
	// string's terminator, so requiring both rejects substrings of paths and of
	// longer architecture names alike.
	needle := make([]byte, 0, len(arch)+2)
	needle = append(needle, 0)
	needle = append(needle, arch...)
	needle = append(needle, 0)

	files := archProbeFiles(binaryPath)
	scannedAny := false
	for _, f := range files {
		hit, err := fileHasLiteral(f, needle)
		if err != nil {
			continue
		}
		scannedAny = true
		if hit {
			return true, true
		}
	}
	if !scannedAny {
		return false, false
	}
	return false, true
}

// archProbeFiles returns the binary plus the llama libraries it loads, directly
// or transitively. Traversal stays on objects whose name carries "llama": the
// architecture table lives there, and following everything else would mean
// scanning libc and the CUDA runtime to no purpose.
func archProbeFiles(binaryPath string) []string {
	out := []string{binaryPath}
	seen := map[string]bool{binaryPath: true}

	for i := 0; i < len(out) && len(out) < archProbeMaxFiles; i++ {
		for _, dep := range elfLlamaDeps(out[i]) {
			if seen[dep] {
				continue
			}
			seen[dep] = true
			out = append(out, dep)
			if len(out) >= archProbeMaxFiles {
				break
			}
		}
	}
	return out
}

// elfLlamaDeps resolves the llama-named DT_NEEDED entries of one object against
// its DT_RUNPATH/DT_RPATH. Only $ORIGIN-relative and absolute entries are
// resolved -- backends built from source always locate their own libraries that
// way, and guessing at system search paths would invite the wrong libllama.
func elfLlamaDeps(path string) []string {
	f, err := elf.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	needed, err := f.ImportedLibraries()
	if err != nil || len(needed) == 0 {
		return nil
	}
	origin := filepath.Dir(path)

	var dirs []string
	for _, tag := range []elf.DynTag{elf.DT_RUNPATH, elf.DT_RPATH} {
		vals, err := f.DynString(tag)
		if err != nil {
			continue
		}
		for _, v := range vals {
			for _, entry := range strings.Split(v, ":") {
				if entry = strings.TrimSpace(entry); entry == "" {
					continue
				}
				entry = strings.ReplaceAll(entry, "${ORIGIN}", origin)
				entry = strings.ReplaceAll(entry, "$ORIGIN", origin)
				if filepath.IsAbs(entry) {
					dirs = append(dirs, entry)
				}
			}
		}
	}
	// The object's own directory is where a build most often keeps its
	// libraries, and it stays valid when RUNPATH was stripped.
	dirs = append(dirs, origin)

	var out []string
	for _, name := range needed {
		if !strings.Contains(strings.ToLower(name), "llama") {
			continue
		}
		found := ""
		for _, dir := range dirs {
			cand := filepath.Join(dir, name)
			if st, err := os.Stat(cand); err == nil && !st.IsDir() {
				found = cand
				break
			}
		}
		if found == "" {
			// Some builds ship no RUNPATH at all and rely on the launcher's
			// LD_LIBRARY_PATH -- the ik_llama fork keeps llama-server in
			// build/bin while its libllama.so stays in build/src. Searching the
			// build tree finds those without encoding any one fork's layout.
			found = findLibInBuildTree(filepath.Dir(origin), name)
		}
		if found != "" {
			out = append(out, found)
		}
	}
	return out
}

// archProbeSearchDirs bounds the fallback search so an unexpected directory
// tree cannot turn a probe into a full-disk walk.
const archProbeSearchDirs = 512

// findLibInBuildTree looks for a library by exact name under a build root,
// shallowly. Build layouts keep their objects a couple of levels down, so a
// small depth limit finds them while keeping the walk cheap.
func findLibInBuildTree(root, name string) string {
	if root == "" || name == "" {
		return ""
	}
	rootDepth := strings.Count(filepath.Clean(root), string(os.PathSeparator))
	found := ""
	visited := 0

	filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || found != "" {
			if found != "" {
				return filepath.SkipAll
			}
			return nil
		}
		if d.IsDir() {
			visited++
			if visited > archProbeSearchDirs {
				return filepath.SkipAll
			}
			if strings.Count(filepath.Clean(path), string(os.PathSeparator))-rootDepth >= 3 {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Name() == name {
			found = path
			return filepath.SkipAll
		}
		return nil
	})
	return found
}

// fileHasLiteral streams a file looking for an exact byte sequence, overlapping
// successive reads so a match straddling a chunk boundary is still found.
func fileHasLiteral(path string, needle []byte) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()

	buf := make([]byte, archProbeChunk)
	overlap := len(needle) - 1
	carry := make([]byte, 0, overlap)
	for {
		n, readErr := f.Read(buf)
		if n > 0 {
			window := append(append(make([]byte, 0, len(carry)+n), carry...), buf[:n]...)
			if bytes.Contains(window, needle) {
				return true, nil
			}
			if len(window) > overlap {
				carry = append(carry[:0], window[len(window)-overlap:]...)
			} else {
				carry = append(carry[:0], window...)
			}
		}
		if readErr == io.EOF {
			return false, nil
		}
		if readErr != nil {
			return false, readErr
		}
	}
}

// RecipesForArch returns reviewed recipes that route an architecture, so a
// launch that cannot proceed can name the exact fix instead of describing the
// problem. Generic by construction: it consults the catalog rather than
// special-casing any model.
func RecipesForArch(arch string) []Recipe {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if arch == "" {
		return nil
	}
	var out []Recipe
	for _, r := range Recipes() {
		if strings.EqualFold(strings.TrimSpace(r.RouteArch), arch) {
			out = append(out, r)
		}
	}
	return out
}

// RegisteredForArch returns registered backends routing an architecture.
func RegisteredForArch(arch string) []Backend {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if arch == "" {
		return nil
	}
	var out []Backend
	for _, b := range Load() {
		if strings.EqualFold(strings.TrimSpace(b.RouteArch), arch) {
			out = append(out, b)
		}
	}
	return out
}
