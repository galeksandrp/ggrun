package main

import (
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"

	"github.com/raketenkater/ggrun/pkg/claudesession"
	"github.com/raketenkater/ggrun/pkg/config"
)

// Claude Code's workflow resume cache is keyed by session ID on disk, so the
// session ID is the whole resume handle. ggrun mints it at launch with
// --session-id instead of discovering it afterwards by scanning transcripts,
// and stores it with the launch shape that produced it.

// claudeSessionSpec carries the resume decision from the launch path into the
// client invocation.
type claudeSessionSpec struct {
	ID       string
	Resume   bool
	Workflow *claudesession.Workflow
	Cached   int
}

// claudeProjectsDir is where Claude Code keeps per-project session state.
func claudeProjectsDir() string {
	if dir := os.Getenv("CLAUDE_CONFIG_DIR"); dir != "" {
		return filepath.Join(dir, "projects")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".claude", "projects")
}

// newClaudeSessionSpec mints a session ID for a fresh launch.
func newClaudeSessionSpec() (*claudeSessionSpec, error) {
	id, err := claudesession.NewSessionID()
	if err != nil {
		return nil, err
	}
	return &claudeSessionSpec{ID: id}, nil
}

// resolveClaudeResume loads a recorded session for --claude-resume. The value
// is a session ID, or "latest" for the newest session in this directory.
func resolveClaudeResume(cacheDir, workDir, value string) (claudesession.Record, error) {
	value = strings.TrimSpace(value)
	if value == "" || strings.EqualFold(value, "latest") || strings.EqualFold(value, "last") {
		return claudesession.Latest(cacheDir, workDir)
	}
	return claudesession.Load(cacheDir, value)
}

// claudeResumeSpec turns a recorded session into a launch spec, refusing the
// resume when the proposed backend shape no longer matches the recorded one.
// Reusing a conversation and its workflow cache under a different context,
// quant, KV type or placement does not error at runtime -- it silently
// reinterprets state built under other settings.
func claudeResumeSpec(rec claudesession.Record, serverArgs []string, force bool) (*claudeSessionSpec, error) {
	if mismatches := rec.ShapeMismatches(serverArgs); len(mismatches) > 0 && !force {
		var lines []string
		for _, m := range mismatches {
			lines = append(lines, "  "+m.String())
		}
		return nil, fmt.Errorf(
			"backend shape changed since session %s was recorded:\n%s\n"+
				"Resuming would reuse state built under the recorded settings. "+
				"Relaunch with the recorded settings, or pass --claude-resume-force to accept the risk.",
			rec.SessionID, strings.Join(lines, "\n"))
	}
	spec := &claudeSessionSpec{ID: rec.SessionID, Resume: true, Workflow: rec.Workflow}
	if spec.Workflow != nil && spec.Workflow.RunID != "" {
		journal := claudesession.JournalPath(claudeProjectsDir(), rec.WorkDir, rec.SessionID, spec.Workflow.RunID)
		spec.Cached = claudesession.CachedAgents(journal)
		return spec, nil
	}
	// The run ID is assigned inside Claude Code, so discover the session's
	// newest recoverable run instead of requiring it to have been recorded.
	if wf, cached := claudesession.LatestRun(claudeProjectsDir(), rec.WorkDir, rec.SessionID); wf != nil {
		spec.Workflow, spec.Cached = wf, cached
	}
	return spec, nil
}

// recordClaudeSession stores the session and the exact shape it ran under.
// Recording is evidence, not a dependency: a failure must not stop a launch.
func recordClaudeSession(cacheDir string, spec *claudeSessionSpec, modelPath, backend string, port int, launchArgs, serverArgs []string) {
	if spec == nil || cacheDir == "" {
		return
	}
	workDir, err := os.Getwd()
	if err != nil {
		return
	}
	rec := claudesession.Record{
		SessionID:  spec.ID,
		WorkDir:    workDir,
		ModelPath:  modelPath,
		Backend:    backend,
		Port:       port,
		LaunchArgs: launchArgs,
		ServerArgs: serverArgs,
		Workflow:   spec.Workflow,
	}
	if err := claudesession.Save(cacheDir, rec); err != nil {
		fmt.Fprintf(os.Stderr, "[claude-code] could not record session for resume: %v\n", err)
	}
}

// claudeLaunchSession resolves the session for this launch: either a recorded
// one being resumed, or a freshly minted ID that is recorded for next time.
func claudeLaunchSession(cfg *config.Config, req *launchRequest, serverArgs []string) (*claudeSessionSpec, error) {
	cacheDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
	}
	if req != nil && req.ClaudeResume != "" {
		workDir, err := os.Getwd()
		if err != nil {
			return nil, err
		}
		rec, err := resolveClaudeResume(cacheDir, workDir, req.ClaudeResume)
		if err != nil {
			return nil, err
		}
		return claudeResumeSpec(rec, serverArgs, req.ClaudeResumeForce)
	}
	spec, err := newClaudeSessionSpec()
	if err != nil {
		// A missing session ID only costs the ability to resume later, so it
		// must not block the launch itself.
		fmt.Fprintf(os.Stderr, "[claude-code] could not mint a session id: %v\n", err)
		return nil, nil
	}
	var (
		modelPath  string
		backend    string
		port       int
		launchArgs []string
	)
	if req != nil {
		modelPath, backend, port = req.ModelPath, req.Backend, req.Port
		// The argv as given, so `ggrun claude resume` reproduces this launch
		// exactly instead of re-deriving flags that may since have changed.
		launchArgs = claudeStripResumeArgs(req.OriginalArgs)
	}
	recordClaudeSession(cacheDir, spec, modelPath, backend, port, launchArgs, serverArgs)
	return spec, nil
}

// claudeSessionArgs prepends the session flags to the client invocation.
//
// A resume must reuse the original session ID. --fork-session mints a new one,
// which moves the journal path and silently discards every cached agent, so it
// is refused rather than quietly honoured.
func claudeSessionArgs(spec *claudeSessionSpec, extraArgs, args []string) ([]string, error) {
	if spec == nil || spec.ID == "" {
		return args, nil
	}
	if hasArg(extraArgs, "--session-id") || hasArg(extraArgs, "--resume") || hasArg(extraArgs, "-r") {
		// The user pinned their own session; do not fight them for it.
		return args, nil
	}
	if !spec.Resume {
		return append([]string{"--session-id", spec.ID}, args...), nil
	}
	if hasArg(extraArgs, "--fork-session") {
		return nil, fmt.Errorf(
			"--fork-session cannot be combined with a workflow resume: forking mints a new " +
				"session ID, which moves the workflow journal path and discards every cached agent")
	}
	return append([]string{"--resume", spec.ID}, args...), nil
}

// claudeResumePrompt asks Claude Code to continue the recorded workflow from
// its journal. Cached agents replay without a model call; anything that was
// still in flight when the session stopped re-runs.
func claudeResumePrompt(spec *claudeSessionSpec) string {
	if spec == nil || spec.Workflow == nil || spec.Workflow.RunID == "" {
		return ""
	}
	wf := spec.Workflow
	var b strings.Builder
	fmt.Fprintf(&b, "Resume the interrupted workflow run %s.", wf.RunID)
	if wf.ScriptPath != "" {
		fmt.Fprintf(&b, " Call Workflow({scriptPath: %q, resumeFromRunId: %q}).", wf.ScriptPath, wf.RunID)
	} else {
		fmt.Fprintf(&b, " Call Workflow with resumeFromRunId: %q and the same script and args as before.", wf.RunID)
	}
	b.WriteString(" Do not change the script or args: agents whose prompt and options are unchanged replay from cache," +
		" and any edit re-runs everything after the first changed call.")
	return b.String()
}

// holdInterruptForClaude keeps Ctrl+C from killing ggrun while Claude Code owns
// the terminal.
//
// The client runs in the foreground, so Ctrl+C reaches the whole process group.
// Claude Code handles it itself -- the first interrupt cancels its current turn
// rather than ending the session -- but ggrun installs no handler until after
// the client returns, so the default action would terminate ggrun mid-session,
// orphan the backend and skip the session record that makes the run resumable.
//
// signal.Notify is used rather than signal.Ignore on purpose: an ignored
// disposition survives exec and would leave Claude Code unable to see Ctrl+C at
// all, whereas a caught signal is reset to default in the child.
func holdInterruptForClaude() func() {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	done := make(chan struct{})
	go func() {
		for {
			select {
			case <-sigCh:
				// Absorbed: the interrupt belongs to Claude Code.
			case <-done:
				return
			}
		}
	}()
	// Idempotent: the caller defers this and may also release it explicitly on
	// an early return, and a second close would panic.
	var once sync.Once
	return func() {
		once.Do(func() {
			signal.Stop(sigCh)
			close(done)
		})
	}
}

// refreshClaudeSessionRecord re-records the session once Claude Code exits, so
// a workflow started during the session is part of the resume handle. The run
// ID is assigned inside Claude Code and cannot be known at launch.
func refreshClaudeSessionRecord(cacheDir string, spec *claudeSessionSpec, modelPath, backend string, port int, launchArgs, serverArgs []string) {
	if spec == nil || spec.ID == "" || cacheDir == "" {
		return
	}
	workDir, err := os.Getwd()
	if err != nil {
		return
	}
	if wf, cached := claudesession.LatestRun(claudeProjectsDir(), workDir, spec.ID); wf != nil {
		spec.Workflow, spec.Cached = wf, cached
		fmt.Printf("[claude-code] Session %s recorded: workflow %s has %d completed agents cached. "+
			"Resume with: ggrun claude resume\n", spec.ID, wf.RunID, cached)
	}
	// Recording on exit also repairs a launch-time record that could not be
	// written, so a resumable session is not lost to a transient error.
	if spec.Resume {
		// A resumed session keeps its original recorded shape; only the
		// workflow pointer is refreshed above.
		if rec, err := claudesession.Load(cacheDir, spec.ID); err == nil {
			rec.Workflow = spec.Workflow
			if saveErr := claudesession.Save(cacheDir, rec); saveErr != nil {
				fmt.Fprintf(os.Stderr, "[claude-code] could not update session record: %v\n", saveErr)
			}
			return
		}
	}
	recordClaudeSession(cacheDir, spec, modelPath, backend, port, launchArgs, serverArgs)
}

// claudeStripResumeArgs removes resume flags from a recorded launch so
// replaying it cannot chain a resume of a resume.
func claudeStripResumeArgs(args []string) []string {
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		switch {
		case args[i] == "--claude-resume":
			i++ // also drop its value
		case strings.HasPrefix(args[i], "--claude-resume="):
		case args[i] == "--claude-resume-force":
		default:
			out = append(out, args[i])
		}
	}
	return out
}

// cmdClaude implements `ggrun claude list|resume`.
func cmdClaude(args []string) {
	sub := "resume"
	if len(args) > 0 {
		sub = args[0]
		args = args[1:]
	}
	switch sub {
	case "list", "ls", "sessions":
		cmdClaudeList()
	case "resume", "continue":
		target, force := parseClaudeResumeArgs(args)
		cmdClaudeResume(target, force)
	default:
		fmt.Fprintln(os.Stderr, "Usage: ggrun claude [list | resume [session-id|latest] [--force]]")
		os.Exit(2)
	}
}

// parseClaudeResumeArgs splits `ggrun claude resume` arguments into the target
// session and the shape-override flag. The override matters because placement
// legitimately recomputes between launches -- a reboot alone can move an expert
// layer -- so a refused resume must have a recovery path that does not require
// reconstructing the long-form launch command by hand.
func parseClaudeResumeArgs(args []string) (target string, force bool) {
	for _, arg := range args {
		switch arg {
		case "--force", "-f", "--claude-resume-force":
			force = true
		default:
			if target == "" {
				target = arg
			}
		}
	}
	return target, force
}

func cmdClaudeList() {
	cfg := loadConfigOrExit()
	workDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	records, err := claudesession.List(cfg.CacheDir, workDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if len(records) == 0 {
		fmt.Printf("No recorded Claude Code sessions for %s\n", workDir)
		return
	}
	fmt.Printf("Recorded Claude Code sessions for %s:\n\n", workDir)
	for _, rec := range records {
		wf, cached := claudesession.LatestRun(claudeProjectsDir(), rec.WorkDir, rec.SessionID)
		fmt.Printf("  %s  %s  %s\n", rec.SessionID,
			rec.Recorded.Local().Format("2006-01-02 15:04"), filepath.Base(rec.ModelPath))
		if wf != nil {
			fmt.Printf("      workflow %s (%s): %d completed agents cached\n", wf.RunID, wf.Name, cached)
		}
	}
	fmt.Println("\nResume the newest with: ggrun claude resume")
}

// cmdClaudeResume relaunches the recorded backend shape and reopens the
// session. It replays the recorded launch argv rather than re-deriving flags,
// because a placement or KV default that changed since would silently
// reinterpret the cached conversation.
func cmdClaudeResume(target string, force bool) {
	cfg := loadConfigOrExit()
	workDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	rec, err := resolveClaudeResume(cfg.CacheDir, workDir, target)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintln(os.Stderr, "List recorded sessions with: ggrun claude list")
		os.Exit(1)
	}
	if wf, cached := claudesession.LatestRun(claudeProjectsDir(), rec.WorkDir, rec.SessionID); wf != nil {
		fmt.Printf("[claude-code] Session %s, workflow %s (%s): %d completed agents will replay from cache.\n",
			rec.SessionID, wf.RunID, wf.Name, cached)
	}
	launchArgs := claudeStripResumeArgs(rec.LaunchArgs)
	if len(launchArgs) == 0 {
		if rec.ModelPath == "" {
			fmt.Fprintf(os.Stderr, "Error: session %s has no recorded launch to reproduce\n", rec.SessionID)
			os.Exit(1)
		}
		launchArgs = []string{rec.ModelPath, "--claude-code"}
	}
	launchArgs = append(launchArgs, "--claude-resume", rec.SessionID)
	if force {
		launchArgs = append(launchArgs, "--claude-resume-force")
	}
	cmdLaunch(launchArgs)
}

// describeClaudeResume reports what a resume will actually recover, so the cost
// is visible before the backend is loaded.
func describeClaudeResume(spec *claudeSessionSpec) string {
	if spec == nil || !spec.Resume {
		return ""
	}
	if spec.Workflow == nil || spec.Workflow.RunID == "" {
		return fmt.Sprintf("[claude-code] Resuming session %s (no recorded workflow run).", spec.ID)
	}
	return fmt.Sprintf(
		"[claude-code] Resuming session %s, workflow %s: %d completed agents replay from cache; "+
			"agents still running when it stopped re-run.",
		spec.ID, spec.Workflow.RunID, spec.Cached)
}
