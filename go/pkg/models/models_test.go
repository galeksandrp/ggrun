package models

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func writeModel(t *testing.T, root, name string, size int) {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestListGroupsShardsAndSorts(t *testing.T) {
	root := t.TempDir()
	writeModel(t, root, "zeta.gguf", 11)
	writeModel(t, root, "nested/alpha-00001-of-00002.gguf", 7)
	writeModel(t, root, "nested/alpha-00002-of-00002.gguf", 13)
	writeModel(t, root, "nested/notes.txt", 100)

	got, err := List(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("models = %#v, want two", got)
	}
	if got[0].Name != filepath.Join("nested", "alpha.gguf") || got[0].Bytes != 20 || len(got[0].Files) != 2 {
		t.Fatalf("grouped shards = %#v", got[0])
	}
	if got[1].Name != "zeta.gguf" || got[1].Bytes != 11 {
		t.Fatalf("single model = %#v", got[1])
	}
}

func TestResolveGGUFShardFilesRequiresEveryDeclaredShard(t *testing.T) {
	root := t.TempDir()
	first := filepath.Join(root, "model-00001-of-00003.gguf")
	writeModel(t, root, "model-00001-of-00003.gguf", 7)
	writeModel(t, root, "model-00002-of-00003.gguf", 11)

	if _, sharded, err := ResolveGGUFShardFiles(first); !sharded || err == nil {
		t.Fatalf("incomplete shard set must fail: sharded=%v err=%v", sharded, err)
	}
	writeModel(t, root, "model-00003-of-00003.gguf", 13)
	files, sharded, err := ResolveGGUFShardFiles(first)
	if err != nil || !sharded || len(files) != 3 {
		t.Fatalf("complete shard set = %#v, sharded=%v, err=%v", files, sharded, err)
	}
	if _, _, err := ResolveGGUFShardFiles(filepath.Join(root, "model-00002-of-00003.gguf")); err == nil {
		t.Fatal("a non-first shard must not be accepted as the model entrypoint")
	}

	single := filepath.Join(root, "single.gguf")
	writeModel(t, root, "single.gguf", 5)
	files, sharded, err = ResolveGGUFShardFiles(single)
	if err != nil || sharded || len(files) != 1 || files[0] != single {
		t.Fatalf("single GGUF resolution = %#v, sharded=%v, err=%v", files, sharded, err)
	}
}

func TestRemoveDeletesOnlySelectedShardedModel(t *testing.T) {
	root := t.TempDir()
	writeModel(t, root, "bundle-00001-of-00002.gguf", 7)
	writeModel(t, root, "bundle-00002-of-00002.gguf", 13)
	writeModel(t, root, "keep.gguf", 5)

	removed, err := Remove(root, "bundle.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if removed.Bytes != 20 || len(removed.Files) != 2 {
		t.Fatalf("removed = %#v", removed)
	}
	for _, shard := range removed.Files {
		if _, err := os.Lstat(filepath.Join(root, shard)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("shard %s still exists: %v", shard, err)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "keep.gguf")); err != nil {
		t.Fatalf("unrelated model was removed: %v", err)
	}
}

func TestRemoveRejectsTraversalAndUnknownModels(t *testing.T) {
	root := t.TempDir()
	writeModel(t, root, "keep.gguf", 5)

	for _, name := range []string{"../keep.gguf", "not-a-model", filepath.Join(root, "keep.gguf")} {
		if _, err := Remove(root, name); err == nil {
			t.Fatalf("Remove(%q) succeeded", name)
		}
	}
	if _, err := Remove(root, "missing.gguf"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing model error = %v, want ErrNotFound", err)
	}
}

func TestRemoveRejectsFilesThroughSymlinkedDirectory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink permissions are not portable on Windows CI")
	}
	root := t.TempDir()
	outside := t.TempDir()
	writeModel(t, outside, "outside.gguf", 5)
	if err := os.Symlink(outside, filepath.Join(root, "outside")); err != nil {
		t.Fatal(err)
	}

	if _, err := Remove(root, filepath.Join("outside", "outside.gguf")); err == nil {
		t.Fatal("Remove followed a symlinked directory outside root")
	}
	if _, err := os.Stat(filepath.Join(outside, "outside.gguf")); err != nil {
		t.Fatalf("outside model was removed: %v", err)
	}
}

func TestRemoveExternalDeletesExplicitOutsideFile(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	writeModel(t, outside, "external.gguf", 9)
	keep := filepath.Join(outside, "keep.gguf")
	writeModel(t, outside, "keep.gguf", 4)

	removed, err := RemoveExternal(root, filepath.Join(outside, "external.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	if removed.Bytes != 9 || len(removed.Files) != 1 || removed.Files[0] != filepath.Join(outside, "external.gguf") {
		t.Fatalf("removed = %#v", removed)
	}
	if _, err := os.Stat(filepath.Join(outside, "external.gguf")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("external model still exists: %v", err)
	}
	if _, err := os.Stat(keep); err != nil {
		t.Fatalf("unrelated external file was removed: %v", err)
	}
	if _, err := os.Stat(root); err != nil {
		t.Fatalf("primary directory must be untouched: %v", err)
	}
}

func TestRemoveExternalRejectsPathResolvingInsidePrimaryDir(t *testing.T) {
	root := t.TempDir()
	writeModel(t, root, "inside.gguf", 5)
	outsideroot := t.TempDir()
	// A symlink inside the primary dir pointing at another file inside the
	// primary dir must not be removable as "external": the resolved target is
	// inside, so this API refuses to bypass the normal in-dir Remove path.
	link := filepath.Join(outsideroot, "link.gguf")
	if err := os.Symlink(filepath.Join(root, "inside.gguf"), link); err != nil {
		t.Skip("symlink permissions are not portable on this platform")
	}

	if _, err := RemoveExternal(root, link); err == nil {
		t.Fatal("RemoveExternal must refuse a path whose target resolves inside the primary directory")
	}
	if _, err := os.Stat(filepath.Join(root, "inside.gguf")); err != nil {
		t.Fatalf("in-dir model was removed through the external path: %v", err)
	}
}

func TestRemoveExternalFollowsSymlinkToRealTargetOnConfirmation(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink permissions are not portable on Windows CI")
	}
	root := t.TempDir()
	outside := t.TempDir()
	realTarget := filepath.Join(outside, "real-target.gguf")
	writeModel(t, outside, "real-target.gguf", 7)
	linkDir := t.TempDir()
	link := filepath.Join(linkDir, "link.gguf")
	if err := os.Symlink(realTarget, link); err != nil {
		t.Fatal(err)
	}

	removed, err := RemoveExternal(root, link)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(realTarget); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("confirmed external removal must delete the symlink's real target, stat err=%v", err)
	}
	if _, err := os.Lstat(link); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("the symlink itself must be removed too, err=%v", err)
	}
	if removed.Bytes != 7 || removed.Name != "real-target.gguf" {
		t.Fatalf("removed = %#v", removed)
	}
}

func TestRemoveExternalRejectsMissingAndDirectoryPaths(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	dirPath := filepath.Join(outside, "adir")
	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{filepath.Join(outside, "missing.gguf"), dirPath} {
		if _, err := RemoveExternal(root, path); err == nil {
			t.Fatalf("RemoveExternal(%q) succeeded", path)
		}
	}
}
