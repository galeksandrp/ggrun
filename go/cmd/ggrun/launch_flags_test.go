package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// parseLaunchArgs handles "--flag=value" and "--flag value" in two separate
// switches, and a flag added to only one of them silently does nothing in the
// other form. That is how --threads shipped: `--threads=16` worked, `--threads
// 16` parsed to zero, and a launch that looked correct ran on the default
// thread count. Both forms are covered here for every flag that takes a value
// and was added late, because the failure is silent by construction.
func TestValueFlagsParseInBothForms(t *testing.T) {
	cases := []struct {
		name  string
		args  []string
		check func(*launchRequest) (int, int) // got, want
	}{
		{"threads long spaced", []string{"m.gguf", "--threads", "16"},
			func(r *launchRequest) (int, int) { return r.Threads, 16 }},
		{"threads long equals", []string{"m.gguf", "--threads=16"},
			func(r *launchRequest) (int, int) { return r.Threads, 16 }},
		{"threads short spaced", []string{"m.gguf", "-t", "16"},
			func(r *launchRequest) (int, int) { return r.Threads, 16 }},
		{"cache-ram long spaced", []string{"m.gguf", "--cache-ram", "32768"},
			func(r *launchRequest) (int, int) { return r.CacheRAMMB, 32768 }},
		{"cache-ram long equals", []string{"m.gguf", "--cache-ram=32768"},
			func(r *launchRequest) (int, int) { return r.CacheRAMMB, 32768 }},
		{"cache-ram short spaced", []string{"m.gguf", "-cram", "32768"},
			func(r *launchRequest) (int, int) { return r.CacheRAMMB, 32768 }},
		{"claude-max-active spaced", []string{"m.gguf", "--claude-max-active", "4"},
			func(r *launchRequest) (int, int) { return r.ClaudeMaxActive, 4 }},
		{"claude-max-active equals", []string{"m.gguf", "--claude-max-active=4"},
			func(r *launchRequest) (int, int) { return r.ClaudeMaxActive, 4 }},
	}
	for _, c := range cases {
		req, err := parseLaunchArgs(c.args)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if got, want := c.check(req); got != want {
			t.Errorf("%s: got %d, want %d", c.name, got, want)
		}
	}
}

// Adding --threads and --cache-ram between the "--parallel" case and its
// trailing `req.ParallelSet = true` reattached that line to --cache-ram. The
// value flags still parsed, so the both-forms test above passed, but
// `--parallel 4` stopped marking itself explicit and `--cache-ram N` started
// marking it instead -- which would have silently pinned the slot count during
// the prompt-cache experiment that flag exists for. Set-flags need their own
// assertions; a value that arrives correctly proves nothing about the bookkeeping
// beside it.
func TestExplicitParallelIsMarkedSetAndCacheRAMIsNot(t *testing.T) {
	for _, args := range [][]string{
		{"m.gguf", "--parallel", "4"},
		{"m.gguf", "--parallel=4"},
	} {
		req, err := parseLaunchArgs(args)
		if err != nil {
			t.Fatalf("%v: %v", args, err)
		}
		if req.Parallel != 4 || !req.ParallelSet {
			t.Errorf("%v: Parallel=%d ParallelSet=%v, want 4/true", args, req.Parallel, req.ParallelSet)
		}
	}
	for _, args := range [][]string{
		{"m.gguf", "--cache-ram", "32768"},
		{"m.gguf", "-cram", "32768"},
		{"m.gguf", "--threads", "16"},
	} {
		req, err := parseLaunchArgs(args)
		if err != nil {
			t.Fatalf("%v: %v", args, err)
		}
		if req.ParallelSet {
			t.Errorf("%v marked the slot count as explicitly chosen", args)
		}
	}
}

// The resume override path is how these flags are actually passed in practice,
// so it is worth asserting end to end rather than trusting the merge alone.
func TestResumeOverridesReachLaunchRequest(t *testing.T) {
	recorded := []string{"m.gguf", "--claude-code", "--ctx-size", "1048576", "--spec", "dflash"}

	_, _, overrides := parseClaudeResumeArgs([]string{"latest", "--spec", "off", "--threads", "16", "--cache-ram", "32768"})
	req, err := parseLaunchArgs(claudeApplyResumeOverrides(recorded, overrides))
	if err != nil {
		t.Fatalf("parse merged launch: %v", err)
	}
	if req.Threads != 16 {
		t.Errorf("Threads = %d, want 16", req.Threads)
	}
	if req.CacheRAMMB != 32768 {
		t.Errorf("CacheRAMMB = %d, want 32768", req.CacheRAMMB)
	}
	// The override must replace the recorded value, not append beside it.
	if req.SpecMode != "off" {
		t.Errorf("SpecMode = %q, want off", req.SpecMode)
	}
	// Untouched recorded flags survive.
	if req.CtxFlag != "1048576" {
		t.Errorf("CtxFlag = %q, want the recorded 1048576", req.CtxFlag)
	}
}

// A graph-instantiation OOM carries no allocation size, so the reserve has to
// come from somewhere else. One routed expert layer is the unit placement moves
// and its size is exact; a tenth of the card is not derived from anything.
func TestRuntimeOOMReservesOneMeasuredExpertLayer(t *testing.T) {
	const log = `0.36.091.423 E CUDA error: out of memory
0.36.091.432 E   current device: 0, in function ggml_cuda_graph_evaluate_and_capture at ggml-cuda.cu:104
0.36.091.433 E   cudaGraphInstantiate(&graph->instance, graph->graph, __null, __null, 0)`

	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}}
	// 1400 MiB layers: the ledger value must win over the 2457 MiB tenth-of-card.
	model := &placement.ModelProfile{
		Path:                   "m.gguf",
		RoutedExpertLayerBytes: []int64{1300 << 20, 1400 << 20, 1200 << 20},
	}

	device, reserveMB, estimated, ok := runtimeLogCUDAOOM(log, caps, model, nil)
	if !ok {
		t.Fatal("a sizeless CUDA OOM was not recognized")
	}
	if device != 0 {
		t.Errorf("device = %d, want 0", device)
	}
	if reserveMB != 1400 {
		t.Errorf("reserve = %d MiB, want one 1400 MiB layer", reserveMB)
	}
	if !estimated {
		t.Error("a sizeless OOM must stay marked estimated")
	}

	// A repeat at the same shape moves a second layer rather than standing still.
	_, repeat, _, ok := runtimeLogCUDAOOM(log, caps, model, map[int]int{0: reserveMB})
	if !ok || repeat <= reserveMB {
		t.Errorf("repeat reserve = %d, want more than %d", repeat, reserveMB)
	}

	// Without a ledger it must still reserve something: an OOM proves
	// something has to give.
	if _, fallback, _, ok := runtimeLogCUDAOOM(log, caps, nil, nil); !ok || fallback <= 0 {
		t.Errorf("fallback reserve = %d, want positive", fallback)
	}
}

// The recovery log is only reachable if two runs of the same workload agree on
// its name. Three real launches crashed at --n-cpu-moe 23 in a row because their
// argv differed by ten MiB of derived prompt-cache budget, which was enough to
// give each crash its own scope: every retry then read a path that had never
// been written and re-planned the placement that had just aborted.
func TestRecoveryIdentityIgnoresPlanDerivedFlags(t *testing.T) {
	base := []string{"llama-server", "-m", "m.gguf", "--ctx-size", "1048576",
		"-b", "2048", "-ub", "512", "--parallel", "4", "--threads", "16"}
	plan := func(ot, moe, cram, ckpt, ngl, split string) []string {
		return append(append([]string(nil), base...),
			"-ot", ot, "--n-cpu-moe", moe, "-cram", cram,
			"--ctx-checkpoints", ckpt, "-ngl", ngl, "--tensor-split", split,
			"--host", "0.0.0.0", "--port", "8081")
	}
	crashed := plan("blk.1=CUDA0", "23", "9752", "16", "999", "1.00,0.00,0.00")
	replanned := plan("blk.1=CUDA0,blk.2=CPU", "24", "9742", "15", "999", "0.90,0.10,0.00")

	if got, want := recoveryLaunchIdentity(replanned), recoveryLaunchIdentity(crashed); got != want {
		t.Errorf("re-planned launch got identity %s, want the crashed launch's %s", got, want)
	}

	// The shape must still separate genuinely different workloads, or an OOM
	// from a 1M-context run would be applied to a 256k one.
	for _, change := range [][2]string{
		{"--ctx-size", "262144"}, {"--parallel", "2"}, {"-ub", "256"}, {"--threads", "8"},
	} {
		other := append([]string(nil), crashed...)
		for i, tok := range other {
			if tok == change[0] {
				other[i+1] = change[1]
			}
		}
		if recoveryLaunchIdentity(other) == recoveryLaunchIdentity(crashed) {
			t.Errorf("%s %s shares an identity with the crashed launch", change[0], change[1])
		}
	}

	// specLaunchIdentity verifies a speculative result came from an exact
	// command line, so it must keep reacting to everything.
	if specLaunchIdentity(replanned) == specLaunchIdentity(crashed) {
		t.Error("specLaunchIdentity must stay exact; only the recovery identity is coarsened")
	}
}

// ggrun reads its measurements out of the backend's log, and the ones that
// matter most are not printed at llama.cpp's default level. Prefix-reuse
// decisions ("forcing full prompt re-processing", "restored context
// checkpoint") and the host prompt cache's own accounting are all trace-level.
// This project re-prefilled every turn for weeks -- 0 tokens reused out of
// 1.16 million, measured only in hindsight -- with nothing in the log to say so.
func TestBackendVerbosityDefaultsToTrace(t *testing.T) {
	help := "  -lv, --verbosity N   set the verbosity level"
	got := backendVerbosityArgs([]string{"llama-server", "-m", "m.gguf"}, help)
	found := ""
	for i, a := range got {
		if a == "-lv" && i+1 < len(got) {
			found = got[i+1]
		}
	}
	if found != "4" {
		t.Errorf("verbosity = %q, want 4 (trace); args=%v", found, got)
	}
	for _, explicit := range [][]string{
		{"llama-server", "-lv", "2"},
		{"llama-server", "--verbosity", "2"},
		{"llama-server", "-lv=2"},
	} {
		if out := backendVerbosityArgs(explicit, help); len(out) != len(explicit) {
			t.Errorf("%v: ggrun overrode an explicit verbosity -> %v", explicit, out)
		}
	}
	// A backend that does not advertise the flag must not receive it.
	if out := backendVerbosityArgs([]string{"srv"}, "--some-other-flag"); len(out) != 1 {
		t.Errorf("added -lv to a backend that does not support it: %v", out)
	}
}
