package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func freeTokenTestCaps() *detect.Capabilities {
	return &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, Name: "NVIDIA GeForce RTX 4070", PCIBusID: "0000:17:00.0", VRAMTotalMB: 12288},
		{Index: 1, Name: "NVIDIA GeForce RTX 3090 Ti", PCIBusID: "0000:65:00.0", VRAMTotalMB: 24576},
		{Index: 2, Name: "NVIDIA GeForce RTX 3060", PCIBusID: "0000:b3:00.0", VRAMTotalMB: 12288},
	}}
}

func TestParseFreeTokenArgsKeepsAdapterAndNativeFlagsSeparate(t *testing.T) {
	req, err := parseFreeTokenArgs([]string{
		"Qwen/Qwen3.6-35B-A3B", "--gpu=2", "--ctx", "131072",
		"--parallel", "2", "--moe-backend", "hybrid", "--port", "1920",
		"--", "--memory-ratio", "0.86", "--moe-cache-auto",
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.Model != "Qwen/Qwen3.6-35B-A3B" || req.GPU != 2 || !req.GPUSet ||
		req.ContextSize != 131072 || req.Parallel != 2 || req.MoEBackend != "hybrid" || req.Port != 1920 {
		t.Fatalf("unexpected request: %+v", req)
	}
	wantExtra := []string{"--memory-ratio", "0.86", "--moe-cache-auto"}
	if !reflect.DeepEqual(req.ExtraArgs, wantExtra) {
		t.Fatalf("passthrough = %v, want %v", req.ExtraArgs, wantExtra)
	}
}

func TestParseFreeTokenArgsRejectsUnknownAdapterFlag(t *testing.T) {
	_, err := parseFreeTokenArgs([]string{"model", "--memory-ratio", "0.8"})
	if err == nil || !strings.Contains(err.Error(), "after --") {
		t.Fatalf("expected native-flag boundary error, got %v", err)
	}
}

func TestParseFreeTokenArgsRejectsPassthroughTensorParallelism(t *testing.T) {
	for _, args := range [][]string{
		{"model", "--", "--tensor-parallel-size", "2"},
		{"model", "--", "--tp-size=2"},
		{"model", "--", "--port", "9999"},
	} {
		if _, err := parseFreeTokenArgs(args); err == nil || !strings.Contains(err.Error(), "adapter-owned") {
			t.Errorf("args %v: expected adapter-owned error, got %v", args, err)
		}
	}
}

func TestSelectFreeTokenGPURequiresOneExplicitGPUOnMultiGPUHost(t *testing.T) {
	caps := freeTokenTestCaps()
	if _, err := selectFreeTokenGPU(caps, -1, false); err == nil || !strings.Contains(err.Error(), "choose exactly one") {
		t.Fatalf("expected explicit-selection error, got %v", err)
	}
	gpu, err := selectFreeTokenGPU(caps, 2, true)
	if err != nil {
		t.Fatalf("select GPU 2: %v", err)
	}
	if gpu.Index != 2 || !strings.Contains(gpu.Name, "3060") {
		t.Fatalf("selected wrong GPU: %+v", gpu)
	}
}

func TestSelectFreeTokenGPUAutoSelectsOnlyNVIDIADevice(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, Name: "AMD Radeon", PCIBusID: "0000:01:00.0"},
		{Index: 1, Name: "NVIDIA RTX 4090", PCIBusID: "0000:02:00.0"},
	}}
	gpu, err := selectFreeTokenGPU(caps, -1, false)
	if err != nil || gpu.Index != 1 {
		t.Fatalf("single NVIDIA selection = %+v, %v", gpu, err)
	}
}

func TestBuildFreeTokenCommandMapsOnlySupportedAdapterFields(t *testing.T) {
	req := defaultFreeTokenRequest()
	req.Model = "/models/Qwen"
	req.Port = 1920
	req.ContextSize = 65536
	req.Parallel = 3
	req.MoEBackend = "offload"
	req.ExtraArgs = []string{"--memory-ratio", "0.85"}
	got := buildFreeTokenCommand(req, "/venv/bin/ft")
	want := []string{
		"/venv/bin/ft", "serve", "--model", "/models/Qwen",
		"--host", "127.0.0.1", "--port", "1920",
		"--max-running-requests", "3", "--moe-backend", "offload",
		"--max-seq-len-override", "65536", "--memory-ratio", "0.85",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("command = %v\nwant    = %v", got, want)
	}
}

func TestHelpAdvertisesExperimentalFreeTokenAdapter(t *testing.T) {
	if help := usageText(); !strings.Contains(help, "freetoken <model>") {
		t.Fatalf("main help does not advertise FreeToken adapter")
	}
}
