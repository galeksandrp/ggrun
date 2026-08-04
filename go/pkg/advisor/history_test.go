package advisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSaveAnalysisWritesValidatedLatestAtomically(t *testing.T) {
	dir := t.TempDir()
	incident := normalizedIncident(t, ModeSupport, ActionNoAction)
	decision := Decision{SchemaVersion: 1, Action: ActionNoAction, Confidence: 1, Rationale: "deterministic recovery exhausted"}
	path, err := SaveAnalysis(dir, incident, &decision, RunReport{ReleaseVerified: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range []string{path, filepath.Join(dir, "advisor", "latest.json")} {
		data, err := os.ReadFile(candidate)
		if err != nil {
			t.Fatal(err)
		}
		var record AnalysisRecord
		if err := json.Unmarshal(data, &record); err != nil {
			t.Fatalf("invalid record %s: %v", candidate, err)
		}
		if record.Decision == nil || record.Decision.Action != ActionNoAction || !record.Report.ReleaseVerified {
			t.Fatalf("record lost validated data: %#v", record)
		}
	}
}
