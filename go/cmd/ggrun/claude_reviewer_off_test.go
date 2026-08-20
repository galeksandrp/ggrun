package main

import (
	"strings"
	"testing"
)

func TestParseClaudeReviewerAcceptsOff(t *testing.T) {
	got, err := parseClaudeReviewer("--claude-reviewer", "off")
	if err != nil {
		t.Fatalf("parseClaudeReviewer(off): %v", err)
	}
	if got != claudeReviewerOff {
		t.Errorf("got %q, want %q", got, claudeReviewerOff)
	}
}

func TestParseClaudeReviewerRejectsUnknownAndNamesOff(t *testing.T) {
	_, err := parseClaudeReviewer("--claude-reviewer", "nope")
	if err == nil {
		t.Fatal("want an error for an unknown reviewer value")
	}
	// The error is the only place a user learns the valid set, so it has to
	// list "off" now that "off" is selectable.
	if got := err.Error(); !strings.Contains(got, `"off"`) {
		t.Errorf("error %q does not mention the off value", got)
	}
}

// Choosing "off" is the user asking for self-review up front, so it must
// disable the reviewer at parse time -- every downstream gate keys off
// ClaudeReviewerDisabled, and none of them should need a special case.
func TestClaudeReviewerOffDisablesTheReviewer(t *testing.T) {
	req, err := parseLaunchArgs([]string{"--claude-code", "--claude-reviewer", "off"})
	if err != nil {
		t.Fatalf("parseLaunchArgs: %v", err)
	}
	if !req.ClaudeReviewerDisabled {
		t.Error("ClaudeReviewerDisabled = false, want true for --claude-reviewer off")
	}
}

// Every other reviewer value still seats a reviewer; only "off" opts out.
func TestOtherClaudeReviewerValuesKeepTheReviewer(t *testing.T) {
	for _, value := range []string{"auto", "qwen", "qwen2b", "nanbeige"} {
		req, err := parseLaunchArgs([]string{"--claude-code", "--claude-reviewer", value})
		if err != nil {
			t.Fatalf("parseLaunchArgs(%s): %v", value, err)
		}
		if req.ClaudeReviewerDisabled {
			t.Errorf("--claude-reviewer %s disabled the reviewer, want it seated", value)
		}
	}
}

// With the reviewer off there is no reservation to place, which is what keeps
// the main model from having VRAM held back for a companion that never starts.
func TestClaudeReviewerOffMakesNoReservation(t *testing.T) {
	req, err := parseLaunchArgs([]string{"--claude-code", "--claude-reviewer", "off"})
	if err != nil {
		t.Fatalf("parseLaunchArgs: %v", err)
	}
	if got := claudeReviewerReservation(req, nil, ""); got != nil {
		t.Errorf("reservation = %+v, want nil with the reviewer off", got)
	}
}
