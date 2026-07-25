package main

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/claudesession"
)

func TestClaudeSessionArgsPinsAMintedSessionID(t *testing.T) {
	spec := &claudeSessionSpec{ID: "072e63a1-819a-4682-a742-559695c3cd76"}
	got, err := claudeSessionArgs(spec, nil, []string{"--append-system-prompt", "x"})
	if err != nil {
		t.Fatalf("claudeSessionArgs: %v", err)
	}
	if len(got) < 2 || got[0] != "--session-id" || got[1] != spec.ID {
		t.Fatalf("want --session-id %s first, got %v", spec.ID, got)
	}
	// The rest of the invocation must survive.
	if got[2] != "--append-system-prompt" {
		t.Errorf("existing args lost: %v", got)
	}
}

func TestClaudeSessionArgsResumesWithTheOriginalID(t *testing.T) {
	spec := &claudeSessionSpec{ID: "072e63a1-819a-4682-a742-559695c3cd76", Resume: true}
	got, err := claudeSessionArgs(spec, nil, nil)
	if err != nil {
		t.Fatalf("claudeSessionArgs: %v", err)
	}
	if len(got) != 2 || got[0] != "--resume" || got[1] != spec.ID {
		t.Fatalf("want --resume %s, got %v", spec.ID, got)
	}
}

// Forking mints a new session ID, which moves the journal path. Every cached
// agent would be silently discarded, so this must fail loudly.
func TestClaudeSessionArgsRefusesForkSessionOnResume(t *testing.T) {
	spec := &claudeSessionSpec{ID: "072e63a1-819a-4682-a742-559695c3cd76", Resume: true}
	_, err := claudeSessionArgs(spec, []string{"--fork-session"}, nil)
	if err == nil {
		t.Fatal("--fork-session was accepted alongside a resume")
	}
	if !strings.Contains(err.Error(), "fork-session") {
		t.Errorf("error does not name the offending flag: %v", err)
	}
}

func TestClaudeSessionArgsYieldsToAUserSuppliedSession(t *testing.T) {
	spec := &claudeSessionSpec{ID: "072e63a1-819a-4682-a742-559695c3cd76"}
	for _, user := range [][]string{{"--session-id", "x"}, {"--resume", "y"}, {"-r"}} {
		got, err := claudeSessionArgs(spec, user, []string{"keep"})
		if err != nil {
			t.Fatalf("claudeSessionArgs(%v): %v", user, err)
		}
		if len(got) != 1 || got[0] != "keep" {
			t.Errorf("ggrun overrode the user's own session flag %v: %v", user, got)
		}
	}
}

func TestClaudeResumeSpecRefusesAChangedBackendShape(t *testing.T) {
	rec := claudesession.Record{
		SessionID:  "072e63a1-819a-4682-a742-559695c3cd76",
		ServerArgs: []string{"--ctx-size", "1048576", "--cache-type-k", "q4_0"},
	}
	changed := []string{"--ctx-size", "524288", "--cache-type-k", "q4_0"}

	if _, err := claudeResumeSpec(rec, changed, false); err == nil {
		t.Fatal("resume accepted a changed context size")
	} else if !strings.Contains(err.Error(), "--ctx-size") {
		t.Errorf("error does not name the changed setting: %v", err)
	}

	// The override exists so the user can accept the risk deliberately.
	spec, err := claudeResumeSpec(rec, changed, true)
	if err != nil {
		t.Fatalf("forced resume rejected: %v", err)
	}
	if !spec.Resume || spec.ID != rec.SessionID {
		t.Errorf("forced resume produced a wrong spec: %+v", spec)
	}
}

func TestClaudeResumeSpecAcceptsAnIdenticalShape(t *testing.T) {
	args := []string{"--ctx-size", "1048576", "--parallel", "4", "--cache-type-k", "q4_0"}
	rec := claudesession.Record{SessionID: "072e63a1-819a-4682-a742-559695c3cd76", ServerArgs: args}
	spec, err := claudeResumeSpec(rec, args, false)
	if err != nil {
		t.Fatalf("identical shape rejected: %v", err)
	}
	if !spec.Resume {
		t.Error("spec is not marked as a resume")
	}
}

func TestClaudeStripResumeArgsPreventsChainedResumes(t *testing.T) {
	in := []string{
		"model.gguf", "--claude-code", "--claude-resume", "abc",
		"--ctx-size", "fit", "--claude-resume-force", "--claude-resume=latest",
	}
	got := claudeStripResumeArgs(in)
	want := []string{"model.gguf", "--claude-code", "--ctx-size", "fit"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("claudeStripResumeArgs = %v, want %v", got, want)
	}
}

func TestClaudeResumePromptNamesTheRunAndForbidsEdits(t *testing.T) {
	spec := &claudeSessionSpec{
		ID:     "072e63a1-819a-4682-a742-559695c3cd76",
		Resume: true,
		Workflow: &claudesession.Workflow{
			RunID:      "wf_894b5285-5d3",
			ScriptPath: "/tmp/deep-research.js",
		},
	}
	prompt := claudeResumePrompt(spec)
	for _, want := range []string{"wf_894b5285-5d3", "resumeFromRunId", "/tmp/deep-research.js"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt missing %q: %s", want, prompt)
		}
	}
	// Editing the script invalidates the cache, so the prompt must say so.
	if !strings.Contains(prompt, "Do not change the script") {
		t.Errorf("prompt does not warn against editing: %s", prompt)
	}

	if got := claudeResumePrompt(&claudeSessionSpec{ID: "x", Resume: true}); got != "" {
		t.Errorf("prompt produced for a session with no workflow: %q", got)
	}
}

func TestDescribeClaudeResumeReportsRecoverableWork(t *testing.T) {
	spec := &claudeSessionSpec{
		ID:       "072e63a1-819a-4682-a742-559695c3cd76",
		Resume:   true,
		Cached:   51,
		Workflow: &claudesession.Workflow{RunID: "wf_894b5285-5d3"},
	}
	got := describeClaudeResume(spec)
	if !strings.Contains(got, "51") || !strings.Contains(got, "wf_894b5285-5d3") {
		t.Errorf("summary does not state what is recovered: %s", got)
	}
	// A fresh launch has nothing to announce.
	if got := describeClaudeResume(&claudeSessionSpec{ID: "x"}); got != "" {
		t.Errorf("fresh launch produced a resume summary: %q", got)
	}
}

func TestParseLaunchArgsAcceptsResumeInBothFormsAndImpliesClaudeCode(t *testing.T) {
	for _, args := range [][]string{
		{"model.gguf", "--claude-resume", "latest"},
		{"model.gguf", "--claude-resume=latest"},
	} {
		req, err := parseLaunchArgs(args)
		if err != nil {
			t.Fatalf("parseLaunchArgs(%v): %v", args, err)
		}
		if req.ClaudeResume != "latest" {
			t.Errorf("ClaudeResume = %q, want latest (args %v)", req.ClaudeResume, args)
		}
		// Resuming a Claude session outside Claude Code mode is meaningless.
		if !req.ClaudeCode {
			t.Errorf("--claude-resume did not imply --claude-code (args %v)", args)
		}
		if len(req.OriginalArgs) != len(args) {
			t.Errorf("OriginalArgs = %v, want %v", req.OriginalArgs, args)
		}
	}
}

func TestParseLaunchArgsResumeForceIsOffByDefault(t *testing.T) {
	req, err := parseLaunchArgs([]string{"model.gguf", "--claude-resume", "latest"})
	if err != nil {
		t.Fatalf("parseLaunchArgs: %v", err)
	}
	if req.ClaudeResumeForce {
		t.Error("resume force defaulted to on; a changed shape must be refused by default")
	}
	forced, err := parseLaunchArgs([]string{"model.gguf", "--claude-resume", "latest", "--claude-resume-force"})
	if err != nil {
		t.Fatalf("parseLaunchArgs: %v", err)
	}
	if !forced.ClaudeResumeForce {
		t.Error("--claude-resume-force was not parsed")
	}
}

func TestParseLaunchArgsRejectsResumeWithoutAValue(t *testing.T) {
	if _, err := parseLaunchArgs([]string{"model.gguf", "--claude-resume"}); err == nil {
		t.Fatal("--claude-resume without a value was accepted")
	}
}

// Ctrl+C reaches the whole foreground process group while Claude Code holds the
// terminal. ggrun must absorb it rather than die, or it orphans the backend and
// never records the session.
func TestHoldInterruptForClaudeAbsorbsSIGINT(t *testing.T) {
	release := holdInterruptForClaude()
	proc, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatalf("FindProcess: %v", err)
	}
	for i := 0; i < 3; i++ {
		if err := proc.Signal(os.Interrupt); err != nil {
			t.Fatalf("signal: %v", err)
		}
	}
	// Reaching here at all means the default action did not terminate us.
	time.Sleep(50 * time.Millisecond)
	release()

	// Release must stop delivery so the normal shutdown handler can own SIGINT
	// again once Claude Code has exited.
	release()
}
