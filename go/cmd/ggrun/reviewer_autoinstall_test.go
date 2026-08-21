package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/claudeauto"
)

// Each reviewer profile must install the artifact it actually runs. The 2B
// profile is planned with a 2600 MB reservation; installing the 4B under it
// leaves the companion ~2000 MB under-reserved, which is an OOM the planner
// cannot see coming.
func TestReviewerProfileInstallsItsOwnArtifact(t *testing.T) {
	cases := []struct {
		override string
		wantFile string
		wantMB   int
	}{
		{"", claudeauto.DefaultReviewerFile, claudeReviewerReservationVRAMMB},
		{claudeReviewerAuto, claudeauto.DefaultReviewerFile, claudeReviewerReservationVRAMMB},
		{claudeReviewerQwen, claudeauto.DefaultReviewerFile, claudeReviewerReservationVRAMMB},
		{claudeReviewerQwen2B, claudeauto.DefaultSmallReviewerFile, claudeSmallReviewerReservationVRAMMB},
	}
	for _, tc := range cases {
		req := &launchRequest{ClaudeCode: true, ClaudeReviewerOverride: tc.override}
		profile := resolveClaudeCompanionProfile(req, "")
		if profile == nil {
			t.Fatalf("--claude-reviewer %q resolved no profile", tc.override)
		}
		if !strings.HasSuffix(profile.ModelPath, tc.wantFile) {
			t.Errorf("--claude-reviewer %q: profile model = %q, want it to end in %q",
				tc.override, profile.ModelPath, tc.wantFile)
		}
		if profile.ReservationVRAMMB != tc.wantMB {
			t.Errorf("--claude-reviewer %q: reservation = %d MB, want %d",
				tc.override, profile.ReservationVRAMMB, tc.wantMB)
		}
		// The artifact the launcher installs must be the one the profile plans
		// for, otherwise the reservation describes a different model.
		spec := claudeReviewerArtifactSpec(profile)
		if spec.Name != tc.wantFile {
			t.Errorf("--claude-reviewer %q: would install %q, but the profile runs %q",
				tc.override, spec.Name, tc.wantFile)
		}
		if spec.URL == "" || spec.Size <= 0 || spec.SHA256 == "" {
			t.Errorf("--claude-reviewer %q: artifact spec is not fully pinned: %+v", tc.override, spec)
		}
	}
}

// The pinned 2B must be a genuinely different artifact from the 4B; if the specs
// ever collapsed to the same file the mismatch above would go unnoticed.
func TestSmallReviewerSpecIsDistinctFromDefault(t *testing.T) {
	small := claudeauto.SmallReviewerSpec()
	full := claudeauto.DefaultReviewerSpec()
	if small.Name == full.Name || small.SHA256 == full.SHA256 || small.URL == full.URL {
		t.Errorf("small and default reviewer specs are not distinct: small=%+v full=%+v", small, full)
	}
	if small.Size >= full.Size {
		t.Errorf("small reviewer (%d bytes) is not smaller than the default (%d bytes)", small.Size, full.Size)
	}
}
