// Package modelusage records per-model launch counts and last-used timestamps so
// the TUI can sort the model list by actual usage (most-used and most-recent
// first). It is a small cache file updated once per launch — never per request —
// so it stays cheap and never touches the launch hot path.
package modelusage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	cacheVersion = 1
	cacheFile    = "model-usage.json"
)

// Record is one model's launch usage history.
type Record struct {
	Launches   int       `json:"launches"`
	LastUsedAt time.Time `json:"last_used_at,omitempty"`
}

type usageIndex struct {
	Version int               `json:"version"`
	Models  map[string]Record `json:"models"`
}

// Load reads the persisted usage index from cacheDir. It returns an empty index
// when the file is missing or invalid; usage data is an enhancement, never a
// launch blocker.
func Load(cacheDir string) map[string]Record {
	cacheDir = strings.TrimSpace(cacheDir)
	if cacheDir == "" {
		return nil
	}
	data, err := os.ReadFile(filepath.Join(cacheDir, cacheFile))
	if err != nil {
		return nil
	}
	var idx usageIndex
	if json.Unmarshal(data, &idx) != nil || idx.Version != cacheVersion {
		return nil
	}
	return idx.Models
}

// RecordLaunch bumps the launch count and refreshes the last-used timestamp for
// one model path, then writes the whole index back atomically. A write failure
// is deliberately swallowed: losing usage history must not abort a launch.
// key is the canonical model identity (usually a normalized path).
func RecordLaunch(cacheDir, key string) {
	key = strings.TrimSpace(key)
	cacheDir = strings.TrimSpace(cacheDir)
	if key == "" || cacheDir == "" {
		return
	}
	models := Load(cacheDir)
	if models == nil {
		models = make(map[string]Record)
	}
	rec := models[key]
	rec.Launches++
	rec.LastUsedAt = time.Now().UTC()
	models[key] = rec

	idx := usageIndex{Version: cacheVersion, Models: models}
	data, err := json.MarshalIndent(idx, "", "  ")
	if err != nil {
		return
	}
	if err := os.MkdirAll(cacheDir, 0o700); err != nil {
		return
	}
	tmp, err := os.CreateTemp(cacheDir, ".model-usage-*.tmp")
	if err != nil {
		return
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return
	}
	if err := tmp.Close(); err != nil {
		return
	}
	destination := filepath.Join(cacheDir, cacheFile)
	if runtime.GOOS == "windows" {
		_ = os.Remove(destination)
	}
	_ = os.Rename(tmpPath, destination)
}
