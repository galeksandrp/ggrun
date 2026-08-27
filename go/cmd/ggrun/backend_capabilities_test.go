package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestApplyRequestDisabledBackendFlagsRemovesOptionValue(t *testing.T) {
	req := &launchRequest{
		DisabledBackendFlags:      map[string]string{"--ctx-checkpoints-interval": "rejected"},
		DisabledBackendFlagValues: map[string]bool{"--ctx-checkpoints-interval": true},
	}
	got := applyRequestDisabledBackendFlags([]string{"server", "-m", "model.gguf", "--ctx-checkpoints-interval", "512", "--metrics"}, req)
	want := []string{"server", "-m", "model.gguf", "--metrics"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("value-aware filtered argv = %#v, want %#v", got, want)
	}
}

func TestBackendCapabilityCacheIsExactAndProtectsExplicitFlags(t *testing.T) {
	dir := t.TempDir()
	model := &placement.ModelProfile{Path: "/models/a.gguf", SizeBytes: 1234, ModelArch: "deepseek4", NumLayers: 61}
	backend := &backendInfo{Path: "/bin/llama-server", Identity: "build-one", Tag: "llama"}
	if err := persistBackendCapability(dir, model, backend, "--ctx-checkpoints-interval", "parser rejected it", true); err != nil {
		t.Fatal(err)
	}
	req := &launchRequest{}
	applyCachedBackendCapabilities(req, dir, model, backend)
	if req.DisabledBackendFlags["--ctx-checkpoints-interval"] == "" || !req.DisabledBackendFlagValues["--ctx-checkpoints-interval"] {
		t.Fatalf("cached capability was not restored: %#v", req)
	}
	different := &launchRequest{}
	applyCachedBackendCapabilities(different, dir, model, &backendInfo{Path: backend.Path, Identity: "build-two", Tag: "llama"})
	if len(different.DisabledBackendFlags) != 0 {
		t.Fatalf("capability leaked across backend builds: %#v", different.DisabledBackendFlags)
	}
	explicit := &launchRequest{OriginalArgs: []string{"model.gguf", "--ctx-checkpoints-interval", "512"}}
	applyCachedBackendCapabilities(explicit, dir, model, backend)
	if len(explicit.DisabledBackendFlags) != 0 {
		t.Fatalf("cached rejection overrode explicit user flag: %#v", explicit.DisabledBackendFlags)
	}
}

func TestBackendCapabilityCachePersistsExactMMapContradiction(t *testing.T) {
	dir := t.TempDir()
	model := &placement.ModelProfile{Path: "/models/m3.gguf", SizeBytes: 4321, ModelArch: "minimax-m3", NumLayers: 62}
	backend := &backendInfo{Path: "/bin/llama-server", Identity: "mapped-build", Tag: "llama"}
	if err := persistBackendCapability(dir, model, backend, "--metrics", "parser rejected it", false); err != nil {
		t.Fatal(err)
	}
	evidence := "live cgroup showed CPU experts in anonymous memory"
	if err := persistBackendCPUExpertMMapCapability(dir, model, backend, placement.CPUExpertMMapAnonymous, evidence); err != nil {
		t.Fatal(err)
	}
	replay := &backendInfo{Path: backend.Path, Identity: backend.Identity, Tag: backend.Tag}
	req := &launchRequest{}
	applyCachedBackendCapabilities(req, dir, model, replay)
	if replay.CPUExpertMMapCapability != placement.CPUExpertMMapAnonymous || replay.CPUExpertMMapEvidence != evidence {
		t.Fatalf("mmap contradiction was not restored: %+v", replay)
	}
	if req.DisabledBackendFlags["--metrics"] == "" {
		t.Fatal("persisting mmap evidence erased existing parser capabilities")
	}
	differentBuild := &backendInfo{Path: backend.Path, Identity: "new-build", Tag: backend.Tag}
	applyCachedBackendCapabilities(&launchRequest{}, dir, model, differentBuild)
	if differentBuild.CPUExpertMMapCapability != "" {
		t.Fatalf("mmap evidence leaked across exact backend builds: %+v", differentBuild)
	}
}

func TestValidateAndRepairBackendArgsRemovesGeneratedOptionalFlag(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "llama-server")
	script := "#!/bin/sh\nfor arg in \"$@\"; do\n" +
		"  if [ \"$arg\" = \"--ctx-checkpoints-interval\" ]; then echo 'error: unknown argument: --ctx-checkpoints-interval' >&2; exit 1; fi\n" +
		"  if [ \"$arg\" = \"--version\" ]; then echo version; exit 0; fi\n" +
		"done\nexit 0\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := config.Defaults()
	cfg.CacheDir = filepath.Join(dir, "cache")
	model := &placement.ModelProfile{Path: filepath.Join(dir, "model.gguf"), Basename: "model", SizeBytes: 1234, NumLayers: 2, ModelArch: "test"}
	be := &backendInfo{Path: bin, Help: "--version", Identity: "test-build", Tag: "llama"}
	req := &launchRequest{
		ModelPath: model.Path, Port: 8081,
		ExtraArgs: []string{"--ctx-checkpoints-interval", "512"},
	}
	strategy := &placement.Strategy{Type: placement.CPUOnly, ContextSize: 4096, BatchSize: 128, UBatchSize: 64, Parallel: 1, KVType: "f16"}
	args := []string{bin, "-m", model.Path, "--ctx-checkpoints-interval", "512"}
	nextStrategy, nextArgs, err := validateAndRepairBackendArgs(req, cfg, model, be, nil, strategy, args)
	if err != nil {
		t.Fatalf("generated optional flag was not repaired: %v", err)
	}
	if nextStrategy != strategy || hasArg(nextArgs, "--ctx-checkpoints-interval") || !req.DisabledBackendFlagValues["--ctx-checkpoints-interval"] {
		t.Fatalf("unexpected repair result: strategy=%p/%p args=%#v req=%#v", nextStrategy, strategy, nextArgs, req)
	}
	if _, err := os.Stat(backendCapabilityPath(cfg.CacheDir, model, be)); err != nil {
		t.Fatalf("measured capability was not persisted: %v", err)
	}
	// A second request should avoid the rejected parser attempt from cache.
	replay := &launchRequest{ExtraArgs: []string{"--ctx-checkpoints-interval", "512"}}
	applyCachedBackendCapabilities(replay, cfg.CacheDir, model, be)
	filtered := applyRequestDisabledBackendFlags([]string{bin, "--ctx-checkpoints-interval", "512", "--version"}, replay)
	if hasArg(filtered, "--ctx-checkpoints-interval") {
		t.Fatalf("cached repair was not replayed: %#v", filtered)
	}
}

func TestCriticalGeneratedFlagIsNotSilentlyRemoved(t *testing.T) {
	if repairableGeneratedBackendFlag("--flash-attn") {
		t.Fatal("flash attention is a model/placement requirement, not an optional repairable flag")
	}
}

func TestBackendArgumentDiagnosticRecognizesInvalidArgument(t *testing.T) {
	diagnostic := backendArgumentDiagnostic("error: invalid argument: --ctx-checkpoints-interval\n")
	if diagnostic == "" {
		t.Fatal("common invalid-argument dialect was not recognized")
	}
	if got := rejectedBackendFlag(diagnostic); got != "--ctx-checkpoints-interval" {
		t.Fatalf("rejected flag = %q", got)
	}
	if len(repairableGeneratedBackendFlags) <= 8 {
		t.Fatalf("test no longer exercises the former eight-attempt ceiling: %d flags", len(repairableGeneratedBackendFlags))
	}
}
