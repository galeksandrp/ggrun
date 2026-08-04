package advisor

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type Artifact struct {
	Name           string `json:"name"`
	Model          string `json:"model"`
	Architecture   string `json:"architecture"`
	Quantization   string `json:"quantization"`
	URL            string `json:"url"`
	SourceRevision string `json:"source_revision"`
	SizeBytes      int64  `json:"size_bytes"`
	SHA256         string `json:"sha256"`
	License        string `json:"license"`
	LicenseURL     string `json:"license_url"`
	MinimumBackend string `json:"minimum_backend"`
}

// DefaultArtifact is intentionally a manifest, not bytes embedded in ggrun.
// The 2.57 GB model is installed on demand, verified by size and SHA-256, and
// can be replaced independently of the launcher. The GGUF is an unofficial
// conversion of the official Apache-2.0 checkpoint; both identities are pinned.
var DefaultArtifact = Artifact{
	Name:           "Nanbeige4.2-3B-Q4_K_M.gguf",
	Model:          "Nanbeige4.2-3B",
	Architecture:   "nanbeige",
	Quantization:   "Q4_K_M",
	URL:            "https://huggingface.co/Tdamre/Nanbeige4.2-3B-GGUF/resolve/main/Nanbeige4.2-3B-Q4_K_M.gguf",
	SourceRevision: "451ed48c3273ecef7ea8faaa43c31ce529763bb1",
	SizeBytes:      2574807840,
	SHA256:         "99c7bfb88907f7eee0a04c4314f1c46bca391819478d8cb90b3e164f09576489",
	License:        "Apache-2.0",
	LicenseURL:     "https://huggingface.co/Nanbeige/Nanbeige4.2-3B",
	MinimumBackend: "ggrun nanbeige42 recipe at b77d646751d01c0962bc203b6809e9d94f7d50b7 with the reviewed loop_count GGUF compatibility patch",
}

type Installer struct {
	Client *http.Client
}

func ArtifactPath(cacheDir string, artifact Artifact) string {
	return filepath.Join(cacheDir, "advisor", "models", filepath.Base(artifact.Name))
}

func VerifyArtifact(path string, artifact Artifact) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if artifact.SizeBytes > 0 && info.Size() != artifact.SizeBytes {
		return fmt.Errorf("advisor artifact size %d does not match manifest %d", info.Size(), artifact.SizeBytes)
	}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return err
	}
	actual := hex.EncodeToString(hash.Sum(nil))
	if !strings.EqualFold(actual, artifact.SHA256) {
		return fmt.Errorf("advisor artifact SHA-256 %s does not match manifest %s", actual, artifact.SHA256)
	}
	return nil
}

func (installer Installer) Install(ctx context.Context, cacheDir string, artifact Artifact, progress func(received, total int64)) (string, error) {
	if strings.TrimSpace(cacheDir) == "" {
		return "", errors.New("advisor install requires cache directory")
	}
	if artifact.URL == "" || artifact.Name == "" || artifact.SizeBytes <= 0 || len(artifact.SHA256) != 64 {
		return "", errors.New("advisor artifact manifest is incomplete")
	}
	destination := ArtifactPath(cacheDir, artifact)
	if err := VerifyArtifact(destination, artifact); err == nil {
		return destination, nil
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
		return "", err
	}
	release, err := acquireArtifactLock(ctx, destination+".lock", 30*time.Second)
	if err != nil {
		return "", err
	}
	defer release()
	if err := VerifyArtifact(destination, artifact); err == nil {
		return destination, nil
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, artifact.URL, nil)
	if err != nil {
		return "", err
	}
	client := installer.Client
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Hour}
	}
	response, err := client.Do(request)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("advisor download returned HTTP %d", response.StatusCode)
	}
	if response.ContentLength > 0 && response.ContentLength != artifact.SizeBytes {
		return "", fmt.Errorf("advisor download length %d does not match manifest %d", response.ContentLength, artifact.SizeBytes)
	}
	tmp, err := os.CreateTemp(filepath.Dir(destination), ".advisor-*.download")
	if err != nil {
		return "", err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	hash := sha256.New()
	reader := io.TeeReader(io.LimitReader(response.Body, artifact.SizeBytes+1), hash)
	written, err := copyWithProgress(tmp, reader, artifact.SizeBytes, progress)
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return "", err
	}
	if written != artifact.SizeBytes {
		return "", fmt.Errorf("advisor download wrote %d bytes; want %d", written, artifact.SizeBytes)
	}
	actual := hex.EncodeToString(hash.Sum(nil))
	if !strings.EqualFold(actual, artifact.SHA256) {
		return "", fmt.Errorf("advisor download SHA-256 %s does not match manifest %s", actual, artifact.SHA256)
	}
	if err := os.Chmod(tmpName, 0o600); err != nil {
		return "", err
	}
	if syncFile, err := os.OpenFile(tmpName, os.O_RDWR, 0); err != nil {
		return "", err
	} else if err := syncFile.Sync(); err != nil {
		_ = syncFile.Close()
		return "", err
	} else if err := syncFile.Close(); err != nil {
		return "", err
	}
	if err := os.Rename(tmpName, destination); err != nil {
		return "", err
	}
	if dir, err := os.Open(filepath.Dir(destination)); err == nil {
		_ = dir.Sync()
		_ = dir.Close()
	}
	return destination, nil
}

func copyWithProgress(dst io.Writer, src io.Reader, total int64, progress func(received, total int64)) (int64, error) {
	buffer := make([]byte, 4<<20)
	var received int64
	for {
		n, readErr := src.Read(buffer)
		if n > 0 {
			written, writeErr := dst.Write(buffer[:n])
			received += int64(written)
			if progress != nil {
				progress(received, total)
			}
			if writeErr != nil {
				return received, writeErr
			}
			if written != n {
				return received, io.ErrShortWrite
			}
		}
		if readErr == io.EOF {
			return received, nil
		}
		if readErr != nil {
			return received, readErr
		}
	}
}

func acquireArtifactLock(ctx context.Context, path string, timeout time.Duration) (func(), error) {
	deadline := time.Now().Add(timeout)
	for {
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_, _ = fmt.Fprintf(file, "%d\n", os.Getpid())
			_ = file.Close()
			return func() { _ = os.Remove(path) }, nil
		}
		if !os.IsExist(err) {
			return nil, err
		}
		if info, statErr := os.Stat(path); statErr == nil && time.Since(info.ModTime()) > 4*time.Hour {
			_ = os.Remove(path)
			continue
		}
		if time.Now().After(deadline) {
			return nil, errors.New("timed out waiting for advisor artifact lock")
		}
		select {
		case <-time.After(100 * time.Millisecond):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}
