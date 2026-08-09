package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// Scratch adversarial check: does invalidateRuntimeOOMLaunch delete the cached
// calibration decision when the OOM'd strategy is the *winner* (whose scope key
// differs from the default's base-placement-derived key)?
func TestScratchF10WinnerScopeKey(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()

	// Default strategy (what calibrationPlan saves the decision under).
	def, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		t.Fatalf("default compute: %v", err)
	}
	defaultKey := calibrationScopeKey(req, model, be, caps, def)
	t.Logf("default scope key: %s", defaultKey)

	// What calibration would measure as a winner: kv-alternate candidate.
	cands := calibrationCandidates(req, cfg, model, be, caps, def)
	if len(cands) < 2 {
		t.Fatalf("no calibration challenger for default strategy")
	}
	winner := cands[1].Strategy
	winnerKey := calibrationScopeKey(req, model, be, caps, winner)
	t.Logf("winner (%s) scope key: %s", cands[1].Name, winnerKey)

	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: defaultKey, Winner: cands[1].Name, DefaultTPS: 20, WinnerTPS: 24,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}

	// A runtime OOM on the WINNER strategy (the actual serving placement).
	if err := invalidateRuntimeOOMLaunch(req, cfg, model, be, caps, winner, nil, "runtime oom"); err != nil {
		t.Fatalf("invalidate: %v", err)
	}

	if _, err := placement.LoadCalibrationDecision(cfg.CacheDir, defaultKey); err == nil {
		t.Logf("F10 WEAKNESS: OOM on winner (%s) did NOT delete the decision cached under the default scope key", cands[1].Name)
	} else {
		t.Logf("decision deleted OK on winner OOM")
	}
}

// Scratch check: does the winner even get applied after a runtime-OOM relaunch
// path replans? replanAfterRuntimeOOM does NOT re-apply calibration.
func TestScratchRuntimeOOMReplanNoCalibration(t *testing.T) {
	t.Log("replanAfterRuntimeOOM (line 4859) calls placement.Compute with SkipPlacementCache; no applyCalibrationDecision. After F10 deletes the decision, the relaunch uses the raw estimate (not the winner).")
}
