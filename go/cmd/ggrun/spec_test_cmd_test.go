package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestSpecTestComparisonMath(t *testing.T) {
	if got := percentGain(12, 10); got < 19.99 || got > 20.01 {
		t.Fatalf("percentGain = %f", got)
	}
	if got := inversePercentGain(8, 10); got < 24.99 || got > 25.01 {
		t.Fatalf("inversePercentGain = %f", got)
	}
	if got := percentRegression(95, 100); got < 4.99 || got > 5.01 {
		t.Fatalf("percentRegression = %f", got)
	}
	if got := percentRegression(105, 100); got != 0 {
		t.Fatalf("faster prompt processing reported regression: %f", got)
	}
	if got := absolutePercentDelta(90, 100); got < 9.99 || got > 10.01 {
		t.Fatalf("absolutePercentDelta = %f", got)
	}
}

func TestSpecTestIsKnownCommand(t *testing.T) {
	if !knownCommand("spec-test") {
		t.Fatal("spec-test must bypass compatibility launch dispatch")
	}
}

func TestSpecLaunchIdentityIgnoresNetworkAddressButTracksRuntimeFlags(t *testing.T) {
	a := specLaunchIdentity([]string{"server-a", "--model", "m.gguf", "--port", "8081", "--host", "127.0.0.1", "-ub", "256"})
	b := specLaunchIdentity([]string{"server-b", "--model", "m.gguf", "--port", "9090", "--host", "0.0.0.0", "-ub", "256"})
	if a != b {
		t.Fatal("network-only changes invalidated launch identity")
	}
	c := specLaunchIdentity([]string{"server-a", "--model", "m.gguf", "--port", "8081", "-ub", "128"})
	if c == a {
		t.Fatal("performance flag change did not invalidate launch identity")
	}
}

// spec-test hardcoded MTP throughout, so the only harness able to prove a
// speculative win could not evaluate DFlash -- the path that matters most on a
// MoE served from system RAM, where a forward pass costs nearly the same
// whether it carries one token or many.
func TestResolveSpecTestModeCoversEverySpeculativePath(t *testing.T) {
	for flag, want := range map[string]placement.DraftType{
		"mtp":    placement.DraftMTP,
		"dflash": placement.DraftDFlash,
		"eagle3": placement.DraftEagle3,
		"draft":  placement.DraftModel,
	} {
		mode, err := resolveSpecTestMode(flag)
		if err != nil {
			t.Errorf("resolveSpecTestMode(%q): %v", flag, err)
			continue
		}
		if mode.Draft != want {
			t.Errorf("%s -> draft type %q, want %q", flag, mode.Draft, want)
		}
		if mode.Flag != flag || mode.Label == "" {
			t.Errorf("%s -> flag %q label %q", flag, mode.Flag, mode.Label)
		}
	}
	// Case should not matter.
	if m, err := resolveSpecTestMode("DFlash"); err != nil || m.Draft != placement.DraftDFlash {
		t.Errorf("uppercase mode not resolved: %v %v", m, err)
	}
}

// MTP stays the default so an existing invocation keeps its meaning.
func TestResolveSpecTestModeDefaultsToMTP(t *testing.T) {
	for _, in := range []string{"", "auto", "off", "  "} {
		mode, err := resolveSpecTestMode(in)
		if err != nil {
			t.Fatalf("resolveSpecTestMode(%q): %v", in, err)
		}
		if mode.Draft != placement.DraftMTP {
			t.Errorf("%q defaulted to %q, want MTP", in, mode.Draft)
		}
	}
}

// An unsupported mode must name the alternatives rather than silently testing
// something the user did not ask for.
func TestResolveSpecTestModeRejectsUnknownAndListsOptions(t *testing.T) {
	_, err := resolveSpecTestMode("ngram")
	if err == nil {
		t.Fatal("spec-test accepted a mode it cannot evaluate")
	}
	for _, want := range []string{"ngram", "dflash", "mtp", "eagle3", "draft"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not mention %q: %v", want, err)
		}
	}
}
