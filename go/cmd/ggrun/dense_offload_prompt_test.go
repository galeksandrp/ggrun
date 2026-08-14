package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestConfirmDenseCPUOffloadPrompt covers the launcher's ASK hook for dense
// models that cannot fit on GPU even at reduced context. assumeYes and
// non-terminal callers must preserve today's silent host-offload behavior;
// a terminal user can accept or decline.
func TestConfirmDenseCPUOffloadPrompt(t *testing.T) {
	cases := []struct {
		name       string
		assumeYes  bool
		terminal   bool
		input      string
		want       bool
		wantPrompt bool
	}{
		{"assume-yes accepts silently", true, true, "", true, false},
		{"non-terminal accepts silently", false, false, "", true, false},
		{"terminal yes", false, true, "y\n", true, true},
		{"terminal yes-word", false, true, "yes\n", true, true},
		{"terminal no declines", false, true, "n\n", false, true},
		{"terminal enter declines", false, true, "\n", false, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			in := strings.NewReader(tc.input)
			hook := confirmDenseCPUOffloadWith(tc.assumeYes, in, &out, tc.terminal)
			got := hook(30804, 41509)
			if got != tc.want {
				t.Fatalf("hook(%d, %d) = %v, want %v", 30804, 41509, got, tc.want)
			}
			prompted := strings.Contains(out.String(), "offloading to system RAM")
			if prompted != tc.wantPrompt {
				t.Fatalf("prompted = %v, want %v (output: %q)", prompted, tc.wantPrompt, out.String())
			}
		})
	}
}
