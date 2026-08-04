package advisor

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

func tinyArtifact(t *testing.T, url string, data []byte) Artifact {
	t.Helper()
	sum := sha256.Sum256(data)
	return Artifact{
		Name: "support.gguf", Model: "test", Architecture: "nanbeige", Quantization: "Q4_K_M",
		URL: url, SizeBytes: int64(len(data)), SHA256: hex.EncodeToString(sum[:]), License: "Apache-2.0",
	}
}

func TestInstallerVerifiesChecksumAndPermissions(t *testing.T) {
	payload := []byte("small verified advisor artifact")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(payload)
	}))
	defer server.Close()
	artifact := tinyArtifact(t, server.URL, payload)
	path, err := (Installer{Client: server.Client()}).Install(context.Background(), t.TempDir(), artifact, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyArtifact(path, artifact); err != nil {
		t.Fatal(err)
	}
	if info, err := os.Stat(path); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("artifact permissions = %v, err=%v", info.Mode().Perm(), err)
	}
}

func TestInstallerRejectsChecksumAndLeavesNoArtifact(t *testing.T) {
	payload := []byte("corrupt")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(payload)
	}))
	defer server.Close()
	artifact := tinyArtifact(t, server.URL, payload)
	artifact.SHA256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	cacheDir := t.TempDir()
	if _, err := (Installer{Client: server.Client()}).Install(context.Background(), cacheDir, artifact, nil); err == nil {
		t.Fatal("checksum mismatch was accepted")
	}
	if _, err := os.Stat(ArtifactPath(cacheDir, artifact)); !os.IsNotExist(err) {
		t.Fatalf("failed download left an activatable artifact: %v", err)
	}
}

func TestConcurrentInstallDownloadsOnce(t *testing.T) {
	payload := []byte("one shared verified download")
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		_, _ = w.Write(payload)
	}))
	defer server.Close()
	artifact := tinyArtifact(t, server.URL, payload)
	cacheDir := t.TempDir()
	installer := Installer{Client: server.Client()}
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := installer.Install(context.Background(), cacheDir, artifact, nil)
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := requests.Load(); got != 1 {
		t.Fatalf("download requests=%d, want one", got)
	}
	if matches, _ := filepath.Glob(filepath.Join(cacheDir, "advisor", "models", ".advisor-*.download")); len(matches) != 0 {
		t.Fatalf("temporary downloads leaked: %v", matches)
	}
}
