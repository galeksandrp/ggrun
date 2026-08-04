package advisor

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type AnalysisRecord struct {
	SchemaVersion int       `json:"schema_version"`
	CreatedAt     time.Time `json:"created_at"`
	Incident      Incident  `json:"incident"`
	Decision      *Decision `json:"decision,omitempty"`
	Report        RunReport `json:"report"`
	Error         string    `json:"error,omitempty"`
}

// SaveAnalysis persists the sanitized typed incident and validated decision.
// Raw backend logs are deliberately absent: the only log material retained is
// the bounded helper-backend tail in RunReport, never fed back into a launch.
func SaveAnalysis(cacheDir string, incident Incident, decision *Decision, report RunReport, runErr error) (string, error) {
	if strings.TrimSpace(cacheDir) == "" {
		return "", errors.New("advisor history requires cache directory")
	}
	if err := incident.Normalize(); err != nil {
		return "", err
	}
	if decision != nil {
		copyDecision := *decision
		if err := ValidateDecision(incident, copyDecision); err != nil {
			return "", err
		}
		decision = &copyDecision
	}
	record := AnalysisRecord{SchemaVersion: IncidentSchemaVersion, CreatedAt: time.Now().UTC(), Incident: incident, Decision: decision, Report: report}
	if runErr != nil {
		record.Error = cleanToken(runErr.Error(), 600)
	}
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return "", err
	}
	data = append(data, '\n')
	dir := filepath.Join(cacheDir, "advisor", "history")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	name := fmt.Sprintf("%s-%s.json", record.CreatedAt.Format("20060102T150405.000000000Z"), cleanFilename(incident.ID))
	path := filepath.Join(dir, name)
	if err := atomicWriteFile(path, data, 0o600); err != nil {
		return "", err
	}
	if err := atomicWriteFile(filepath.Join(cacheDir, "advisor", "latest.json"), data, 0o600); err != nil {
		return path, err
	}
	return path, nil
}

func atomicWriteFile(path string, data []byte, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".advisor-record-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Chmod(mode); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return err
	}
	if dir, err := os.Open(filepath.Dir(path)); err == nil {
		_ = dir.Sync()
		_ = dir.Close()
	}
	return nil
}

func cleanFilename(value string) string {
	value = cleanToken(value, 80)
	var b strings.Builder
	for _, r := range value {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		}
	}
	if b.Len() == 0 {
		return "incident"
	}
	return b.String()
}
