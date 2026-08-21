package claudeauto

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

func testArtifact(t *testing.T, size int) ([]byte, string) {
	t.Helper()
	payload := make([]byte, size)
	for i := range payload {
		payload[i] = byte(i % 251)
	}
	sum := sha256.Sum256(payload)
	return payload, hex.EncodeToString(sum[:])
}

// serveRange answers with the requested byte range, so the client under test
// exercises the same resume path a real CDN provides.
func serveRange(w http.ResponseWriter, r *http.Request, payload []byte) {
	rangeHeader := r.Header.Get("Range")
	if rangeHeader == "" {
		w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
		return
	}
	var start int64
	_, _ = fmt.Sscanf(rangeHeader, "bytes=%d-", &start)
	if start >= int64(len(payload)) {
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
		return
	}
	w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(payload)-1, len(payload)))
	w.WriteHeader(http.StatusPartialContent)
	_, _ = w.Write(payload[start:])
}

func TestDownloadModelVerifiesAndInstalls(t *testing.T) {
	payload, sum := testArtifact(t, 64*1024)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serveRange(w, r, payload)
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "reviewer.gguf")
	spec := ModelSpec{URL: srv.URL, Name: "reviewer.gguf", Size: int64(len(payload)), SHA256: sum}
	if err := downloadModel(context.Background(), srv.Client(), spec, dest, nil); err != nil {
		t.Fatalf("downloadModel: %v", err)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("read installed artifact: %v", err)
	}
	if len(got) != len(payload) {
		t.Errorf("installed %d bytes, want %d", len(got), len(payload))
	}
	// The partial file must not survive a successful install.
	if _, err := os.Stat(dest + ".part"); !os.IsNotExist(err) {
		t.Errorf(".part file still present after a successful install")
	}
}

// A download cut off mid-transfer must resume from what is already on disk
// rather than re-fetching the whole artifact. This is what makes a multi-GB
// reviewer installable on a link that cannot hold a connection open.
func TestDownloadModelResumesAfterInterruption(t *testing.T) {
	payload, sum := testArtifact(t, 256*1024)
	var attempts atomic.Int64
	var sawRange atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if n := attempts.Add(1); n == 1 {
			// First attempt: hand over a prefix, then drop the connection.
			w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(payload[:64*1024])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			panic(http.ErrAbortHandler) // sever the connection mid-body
		}
		if r.Header.Get("Range") != "" {
			sawRange.Store(true)
		}
		serveRange(w, r, payload)
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "reviewer.gguf")
	spec := ModelSpec{URL: srv.URL, Name: "reviewer.gguf", Size: int64(len(payload)), SHA256: sum}
	if err := downloadModel(context.Background(), srv.Client(), spec, dest, io.Discard); err != nil {
		t.Fatalf("downloadModel did not recover from an interrupted transfer: %v", err)
	}
	if !sawRange.Load() {
		t.Error("retry did not send a Range header; the interrupted bytes were re-fetched")
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("read installed artifact: %v", err)
	}
	gotSum := sha256.Sum256(got)
	if hex.EncodeToString(gotSum[:]) != sum {
		t.Error("resumed artifact does not match the pinned checksum")
	}
}

// A server that ignores Range answers 200 with the whole artifact. The partial
// file has to be truncated first, or the resumed bytes would be appended to a
// prefix that is already there and the file would be too long.
func TestDownloadModelHandlesServerIgnoringRange(t *testing.T) {
	payload, sum := testArtifact(t, 128*1024)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	}))
	defer srv.Close()

	dir := t.TempDir()
	dest := filepath.Join(dir, "reviewer.gguf")
	// Pre-seed a partial file so the download starts by asking for a range.
	if err := os.WriteFile(dest+".part", payload[:32*1024], 0o644); err != nil {
		t.Fatalf("seed partial: %v", err)
	}
	spec := ModelSpec{URL: srv.URL, Name: "reviewer.gguf", Size: int64(len(payload)), SHA256: sum}
	if err := downloadModel(context.Background(), srv.Client(), spec, dest, nil); err != nil {
		t.Fatalf("downloadModel: %v", err)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("read installed artifact: %v", err)
	}
	if len(got) != len(payload) {
		t.Fatalf("installed %d bytes, want %d; the ignored Range was appended", len(got), len(payload))
	}
	gotSum := sha256.Sum256(got)
	if hex.EncodeToString(gotSum[:]) != sum {
		t.Error("artifact does not match the pinned checksum")
	}
}

// A checksum mismatch must not install, and must not leave a poisoned partial
// behind for the next run to "resume" into.
func TestDownloadModelRejectsAndDiscardsCorruptArtifact(t *testing.T) {
	payload, _ := testArtifact(t, 32*1024)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serveRange(w, r, payload)
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "reviewer.gguf")
	spec := ModelSpec{
		URL: srv.URL, Name: "reviewer.gguf", Size: int64(len(payload)),
		SHA256: strings.Repeat("00", 32), // deliberately wrong
	}
	err := downloadModel(context.Background(), srv.Client(), spec, dest, nil)
	if err == nil {
		t.Fatal("a checksum mismatch was installed")
	}
	if !strings.Contains(err.Error(), "checksum mismatch") {
		t.Errorf("error = %v, want a checksum mismatch", err)
	}
	if _, statErr := os.Stat(dest); !os.IsNotExist(statErr) {
		t.Error("a corrupt artifact was installed at the destination")
	}
	if _, statErr := os.Stat(dest + ".part"); !os.IsNotExist(statErr) {
		t.Error("the corrupt partial was kept and would be resumed next run")
	}
}

// The client must carry no whole-request deadline: the pinned reviewer is 2.55
// GiB, and a 30-minute cap made it uninstallable below ~1.5 MB/s no matter how
// healthy the connection was.
func TestReviewerHTTPClientHasNoWholeRequestDeadline(t *testing.T) {
	if got := reviewerHTTPClient().Timeout; got != 0 {
		t.Errorf("client timeout = %v, want 0; a large artifact on a slow link is not a hung request", got)
	}
}
