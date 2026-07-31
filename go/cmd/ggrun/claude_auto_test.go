package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestClaudeReviewerGPUCandidatesPreservesLargestGPU(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564, BandwidthMBps: 15754},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 985},
		{Index: 2, VRAMTotalMB: 12282, BandwidthMBps: 3938},
	}}
	got := claudeReviewerGPUCandidates(caps, &launchRequest{})
	want := []int{1, 2, 0}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestClaudeMainMaxActiveSerializesHostOffload(t *testing.T) {
	req := &launchRequest{ClaudeCode: true}
	for _, strategyType := range []placement.StrategyType{placement.MoEOffload, placement.DenseCPUOffload} {
		strategy := &placement.Strategy{Type: strategyType, Parallel: 4}
		if got := claudeMainMaxActive(req, strategy); got != 1 {
			t.Fatalf("strategy %s max active=%d, want 1", strategyType, got)
		}
	}
}

func TestClaudeMainMaxActiveLeavesGPUResidentParallel(t *testing.T) {
	for _, tc := range []struct {
		req      *launchRequest
		strategy *placement.Strategy
	}{
		{&launchRequest{ClaudeCode: true}, &placement.Strategy{Type: placement.MultiGPUDense, Parallel: 4}},
		{&launchRequest{}, &placement.Strategy{Type: placement.MoEOffload, Parallel: 4}},
	} {
		if got := claudeMainMaxActive(tc.req, tc.strategy); got != 0 {
			t.Fatalf("unexpected admission cap %d for req=%+v strategy=%+v", got, tc.req, tc.strategy)
		}
	}
}

// A single slot used to return 0, and 0 means "build no scheduler at all" --
// so the one configuration that most needs ordering got none: lane priority,
// affinity and aging all off, with llama.cpp queueing the fan-out FIFO. One
// slot serves one request either way; the limit is what keeps a permission
// review from waiting behind bulk work.
func TestClaudeMainMaxActiveStillSchedulesAtOneSlot(t *testing.T) {
	for _, strategyType := range []placement.StrategyType{placement.MoEOffload, placement.MultiGPUDense} {
		strategy := &placement.Strategy{Type: strategyType, Parallel: 1}
		if got := claudeMainMaxActive(&launchRequest{ClaudeCode: true}, strategy); got != 1 {
			t.Errorf("strategy %s at one slot: max active=%d, want 1", strategyType, got)
		}
	}
}

// The flag is how the serialized default gets tested against real concurrency,
// so it has to actually override -- and it must never exceed the slot count,
// because admitting more than there are slots moves the queue into llama.cpp
// rather than removing it.
func TestClaudeMaxActiveOverrideIsClampedToSlots(t *testing.T) {
	moe := func(parallel int) *placement.Strategy {
		return &placement.Strategy{Type: placement.MoEOffload, Parallel: parallel}
	}
	req := func(limit int) *launchRequest {
		return &launchRequest{ClaudeCode: true, ClaudeMaxActive: limit, ClaudeMaxActiveSet: true}
	}
	for _, tc := range []struct {
		name     string
		req      *launchRequest
		strategy *placement.Strategy
		want     int
	}{
		{"override raises the host-offload default", req(4), moe(4), 4},
		{"clamped to the available slots", req(8), moe(4), 4},
		{"zero is an explicit opt out", req(0), moe(4), 0},
		{"override lowers below the default", req(1), moe(4), 1},
		{"unset keeps the measured default", &launchRequest{ClaudeCode: true}, moe(4), 1},
	} {
		if got := claudeMainMaxActive(tc.req, tc.strategy); got != tc.want {
			t.Errorf("%s: max active=%d, want %d", tc.name, got, tc.want)
		}
	}
}

func TestClaudeReviewerGPUCandidatesKeepSparsePhysicalSelection(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0}, {Index: 1}, {Index: 2}}}
	got := claudeReviewerGPUCandidates(caps, &launchRequest{GPUsFlag: "2,1,2,9"})
	want := []int{2, 1}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("got %v, want physical selection %v", got, want)
	}
}

func TestClaudeReviewerArgsUsesIsolatedDeviceAsLocalMain(t *testing.T) {
	args := claudeReviewerArgs("server", "reviewer.gguf", 1234, "CUDA7", "--reasoning ARG --cache-type-k TYPE --cache-type-v TYPE")
	for _, want := range []string{"--device", "CUDA7", "-mg", "0", "--reasoning", "off", "--ctx-size", "65536", "--cache-type-k", "q8_0", "--cache-type-v"} {
		if !hasArg(args, want) {
			t.Fatalf("missing %q in %v", want, args)
		}
	}
	for _, flag := range []string{"--cache-type-k", "--cache-type-v"} {
		if !hasArgValue(args, flag, "q8_0") {
			t.Fatalf("expected %s q8_0 in %v", flag, args)
		}
	}
}

func TestClaudeReviewerArgsKeepsOlderBackendCompatibility(t *testing.T) {
	args := claudeReviewerArgs("server", "reviewer.gguf", 1234, "", "--reasoning ARG")
	for _, unsupported := range []string{"--cache-type-k", "--cache-type-v"} {
		if hasArg(args, unsupported) {
			t.Fatalf("unexpected unsupported %q in %v", unsupported, args)
		}
	}
}

func TestClaudeReviewerGPUDeviceUsesAdvertisedName(t *testing.T) {
	dir := t.TempDir()
	binary := filepath.Join(dir, "llama-server")
	script := "#!/bin/sh\nprintf 'Available devices:\\n  CUDA3: Test GPU\\n'\n"
	if err := os.WriteFile(binary, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	got, err := claudeReviewerGPUDevice(binary, []string{"CUDA_VISIBLE_DEVICES=2"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "CUDA3" {
		t.Fatalf("got %q, want backend-advertised CUDA3", got)
	}
}

func TestClaudeReviewerGPUDeviceRejectsBackendWithoutCUDA(t *testing.T) {
	dir := t.TempDir()
	binary := filepath.Join(dir, "llama-server")
	script := "#!/bin/sh\nprintf 'Available devices:\\n  Vulkan0: Test GPU\\n'\n"
	if err := os.WriteFile(binary, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := claudeReviewerGPUDevice(binary, nil); err == nil || !strings.Contains(err.Error(), "no CUDA device") {
		t.Fatalf("expected clear missing-CUDA error, got %v", err)
	}
}

func TestFindClaudeReviewerBackendSkipsVulkanForCUDA(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("LLM_APP_HOME", "")
	for _, tc := range []struct {
		path    string
		devices string
	}{
		{filepath.Join(home, "llama.cpp", "build-vulkan", "bin", "llama-server"), "Vulkan0: Test GPU"},
		{filepath.Join(home, "llama.cpp", "build", "bin", "llama-server"), "CUDA0: Test GPU"},
	} {
		if err := os.MkdirAll(filepath.Dir(tc.path), 0755); err != nil {
			t.Fatal(err)
		}
		script := "#!/bin/sh\nif [ \"$1\" = --help ]; then printf '%s\\n' '--reasoning ARG'; else printf 'Available devices:\\n  %s\\n' '" + tc.devices + "'; fi\n"
		if err := os.WriteFile(tc.path, []byte(script), 0755); err != nil {
			t.Fatal(err)
		}
	}
	got := findClaudeReviewerBackend(nil)
	want := filepath.Join(home, "llama.cpp", "build", "bin", "llama-server")
	if got == nil || got.Path != want {
		t.Fatalf("got %#v, want CUDA backend %q", got, want)
	}
}

func TestClaudeReviewerCPUFallbackHidesAccelerators(t *testing.T) {
	got := claudeReviewerCPUEnv()
	for _, want := range []string{"CUDA_VISIBLE_DEVICES=-1", "HIP_VISIBLE_DEVICES=-1", "ROCR_VISIBLE_DEVICES=-1"} {
		if !hasArg(got, want) {
			t.Fatalf("missing %q in %v", want, got)
		}
	}
}

func TestClaudeReviewerBackendEnvAddsResolvedLibraryPath(t *testing.T) {
	root := t.TempDir()
	binDir := filepath.Join(root, "build-cuda", "bin")
	linkDir := filepath.Join(root, ".bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(linkDir, 0755); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(binDir, "llama-server")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(binDir, "libllama-server-impl.so"), []byte("lib"), 0644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(linkDir, "llama-server-cuda")
	if err := os.Symlink(binary, link); err != nil {
		t.Fatal(err)
	}
	got := claudeReviewerBackendEnv(link, []string{"CUDA_VISIBLE_DEVICES=2"})
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "CUDA_VISIBLE_DEVICES=2") {
		t.Fatalf("reviewer env lost GPU isolation: %v", got)
	}
	if !strings.Contains(joined, "LD_LIBRARY_PATH="+binDir) {
		t.Fatalf("reviewer env missing resolved backend lib dir %q: %v", binDir, got)
	}
}

func TestClaudeAutoReviewerNeededDefaultsOnForAuto(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	if !claudeAutoReviewerNeeded(nil) {
		t.Fatal("default local Auto launch must start its reviewer")
	}
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "acceptEdits")
	if claudeAutoReviewerNeeded(nil) {
		t.Fatal("non-Auto permission mode should not spend memory on a reviewer")
	}
}

func TestClaudeReviewerReservationBuildsCompanion(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564, BandwidthMBps: 15754},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 985},
	}}
	res := claudeReviewerReservation(&launchRequest{ClaudeCode: true}, caps, "")
	if res == nil {
		t.Fatal("Claude Code launch with GPUs must reserve the reviewer")
	}
	if res.Name != claudeReviewerCompanionName {
		t.Fatalf("companion name = %q, want %q", res.Name, claudeReviewerCompanionName)
	}
	if res.VRAMMB <= 0 {
		t.Fatalf("reservation must carry a positive VRAM footprint, got %d", res.VRAMMB)
	}
	if !res.AllowCPU {
		t.Fatal("a full-GPU host must keep fail-closed Auto working via CPU")
	}
	// Preference order mirrors the legacy walk: slow GPU first, main last.
	if len(res.GPUPreference) != 2 || res.GPUPreference[0] != 1 || res.GPUPreference[1] != 0 {
		t.Fatalf("GPU preference = %v, want [1 0]", res.GPUPreference)
	}
}

func TestClaudeReviewerReservationSkipsNonClaudeAndCPU(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}}
	if res := claudeReviewerReservation(&launchRequest{}, caps, ""); res != nil {
		t.Fatal("non-Claude launch must not reserve a reviewer")
	}
	if res := claudeReviewerReservation(&launchRequest{ClaudeCode: true, CPUMode: true}, caps, ""); res != nil {
		t.Fatal("CPU-mode launch must not reserve GPU VRAM for the reviewer")
	}
	if res := claudeReviewerReservation(&launchRequest{ClaudeCode: true}, &detect.Capabilities{}, ""); res != nil {
		t.Fatal("GPU-less host must not reserve GPU VRAM for the reviewer")
	}
}

// A reviewer left running by a previous ggrun is already inside the VRAM the
// hardware scan reports as used. Adding the full reservation on top charges one
// process twice: measured here as 2096 MiB resident plus a 2600 MiB reservation,
// ~4.7 GB withheld for a 2.1 GB helper. On the 12 GB card that seat is on, and
// at 1371 MiB per expert layer, it cost two layers -- the card took 4 where an
// unoccupied one took 6.
func TestReviewerReservationIsNotChargedTwice(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564, BandwidthMBps: 15754},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 985},
	}}
	req := &launchRequest{ClaudeCode: true}
	restore := residentReviewerVRAM
	defer func() { residentReviewerVRAM = restore }()

	// Nothing resident: the full bound is reserved.
	residentReviewerVRAM = func() int { return 0 }
	got := claudeReviewerReservation(req, caps, "")
	if got == nil || got.VRAMMB != claudeReviewerReservationVRAMMB {
		t.Fatalf("with no reviewer running, reservation = %+v, want %d MiB", got, claudeReviewerReservationVRAMMB)
	}

	// A stored measurement supersedes the constant.
	dir := t.TempDir()
	if err := placement.RecordCompanionVRAM(dir, claudeReviewerCompanionName, 2096); err != nil {
		t.Fatal(err)
	}
	if got := claudeReviewerReservation(req, caps, dir); got == nil || got.VRAMMB != 2096 {
		t.Errorf("measured reservation = %+v, want 2096 MiB", got)
	}

	// A leftover reviewer already occupies that VRAM, so nothing more is owed.
	residentReviewerVRAM = func() int { return 2096 }
	if got := claudeReviewerReservation(req, caps, dir); got != nil {
		t.Errorf("reservation = %+v, want none: the seat is already occupied", got)
	}
	// A partially covered seat reserves only the difference.
	residentReviewerVRAM = func() int { return 1500 }
	if got := claudeReviewerReservation(req, caps, dir); got == nil || got.VRAMMB != 596 {
		t.Errorf("reservation = %+v, want the uncovered 596 MiB", got)
	}
}

// --parallel is an instruction. Every slot costs context whether or not it is
// ever used, so allocating the slots and then admitting one request is the one
// outcome nobody asked for: observed live as two 131072 slots, one permanently
// idle, at the throughput a single 262144 slot already delivered.
func TestExplicitParallelRaisesAdmissionOnHostOffload(t *testing.T) {
	req := &launchRequest{ClaudeCode: true, ParallelSet: true}
	for _, strategyType := range []placement.StrategyType{placement.MoEOffload, placement.DenseCPUOffload} {
		strategy := &placement.Strategy{Type: strategyType, Parallel: 2}
		if got := claudeMainMaxActive(req, strategy); got != 2 {
			t.Fatalf("strategy %s with explicit --parallel 2: max active=%d, want 2", strategyType, got)
		}
	}
}

// The conservative default still stands when ggrun picked the slot count.
func TestImplicitParallelKeepsHostOffloadSerialized(t *testing.T) {
	req := &launchRequest{ClaudeCode: true}
	strategy := &placement.Strategy{Type: placement.MoEOffload, Parallel: 4}
	if got := claudeMainMaxActive(req, strategy); got != 1 {
		t.Fatalf("max active=%d, want the conservative default of 1", got)
	}
}

// An explicit --claude-max-active is more specific than --parallel and wins.
func TestExplicitMaxActiveOverridesExplicitParallel(t *testing.T) {
	req := &launchRequest{ClaudeCode: true, ParallelSet: true, ClaudeMaxActive: 1, ClaudeMaxActiveSet: true}
	strategy := &placement.Strategy{Type: placement.MoEOffload, Parallel: 4}
	if got := claudeMainMaxActive(req, strategy); got != 1 {
		t.Fatalf("max active=%d, want the explicit 1", got)
	}
}

// "--reasoning" must not be satisfied by "--reasoning-format": mainline rejects
// `--reasoning off` and an unknown flag kills the reviewer at startup. Recent
// mainline disables thinking with --reasoning-budget 0 instead.
func TestClaudeReviewerArgsMatchReasoningFlagsExactly(t *testing.T) {
	legacy := strings.Join(claudeReviewerArgs("bin", "m.gguf", 1, "", "--reasoning FMT --cache-type-k T"), " ")
	if !strings.Contains(legacy, "--reasoning off") {
		t.Errorf("binary with exact --reasoning must get `--reasoning off`: %s", legacy)
	}
	mainline := strings.Join(claudeReviewerArgs("bin", "m.gguf", 1, "", "--reasoning-format FMT --reasoning-budget N"), " ")
	if strings.Contains(mainline, "--reasoning off") {
		t.Errorf("substring match sent a fatal flag to a mainline binary: %s", mainline)
	}
	if !strings.Contains(mainline, "--reasoning-budget 0") {
		t.Errorf("mainline binary must disable thinking via --reasoning-budget 0: %s", mainline)
	}
}

// A reviewer model only a fork can load needs the binary named alongside it.
func TestReviewerBinaryOverrideWinsResolution(t *testing.T) {
	dir := t.TempDir()
	stub := filepath.Join(dir, "fork-server")
	script := "#!/bin/sh\ncase \"$1\" in\n--help) echo '--reasoning-budget N --cache-type-k T' ;;\n--list-devices) echo 'CUDA0: stub' ;;\nesac\nexit 0\n"
	if err := os.WriteFile(stub, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GGRUN_CLAUDE_REVIEWER_BIN", stub)
	be := findClaudeReviewerBackend(&detect.Capabilities{})
	if be == nil || be.Path != stub {
		t.Fatalf("GGRUN_CLAUDE_REVIEWER_BIN was not honoured: %+v", be)
	}
}
