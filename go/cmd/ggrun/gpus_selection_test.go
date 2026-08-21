package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func gpuSelectionBox() *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282, VRAMUsedMB: 163},
			{Index: 1, Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 22648},
			{Index: 2, Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288, VRAMUsedMB: 113},
		},
	}
}

// Reserving a card for other work is the case where silently using it anyway is
// worst, so a --gpus index this machine does not have must fail loudly.
func TestGPUSelectionRejectsUnknownIndex(t *testing.T) {
	for _, flag := range []string{"5", "0,3", "1,9"} {
		err := validateRequestedGPUs(gpuSelectionBox(), &launchRequest{GPUsFlag: flag})
		if err == nil {
			t.Errorf("--gpus %s was accepted; it names hardware that is not present", flag)
			continue
		}
		// The message has to say what IS available, or the user cannot correct it.
		if !strings.Contains(err.Error(), "3090 Ti") {
			t.Errorf("--gpus %s error does not list the available GPUs: %v", flag, err)
		}
	}
}

func TestGPUSelectionAcceptsPresentIndices(t *testing.T) {
	for _, flag := range []string{"0,2", "1", "0,1,2", ""} {
		if err := validateRequestedGPUs(gpuSelectionBox(), &launchRequest{GPUsFlag: flag}); err != nil {
			t.Errorf("--gpus %q rejected: %v", flag, err)
		}
	}
}

// The selection must restrict placement to exactly those cards, renumbered the
// way CUDA_VISIBLE_DEVICES will present them to the backend.
func TestGPUSelectionRestrictsAndRenumbers(t *testing.T) {
	caps, mapping := runtimeGPUCapabilities(gpuSelectionBox(), &launchRequest{GPUsFlag: "0,2"})
	if len(caps.GPUs) != 2 {
		t.Fatalf("got %d cards, want 2", len(caps.GPUs))
	}
	if caps.GPUs[0].Name != "NVIDIA GeForce RTX 4070" || caps.GPUs[1].Name != "NVIDIA GeForce RTX 3060" {
		t.Errorf("wrong cards selected: %s, %s", caps.GPUs[0].Name, caps.GPUs[1].Name)
	}
	// Visible 0 -> physical 0, visible 1 -> physical 2.
	if mapping[0] != 0 || mapping[1] != 2 {
		t.Errorf("visible->physical mapping = %v, want map[0:0 1:2]", mapping)
	}
	// The reserved card must not appear at all.
	for _, g := range caps.GPUs {
		if strings.Contains(g.Name, "3090 Ti") {
			t.Error("the excluded GPU is still in the plan")
		}
	}
}

// A syntactically invalid selection is still rejected with a clear message.
func TestGPUSelectionRejectsMalformed(t *testing.T) {
	for _, flag := range []string{"abc", "-1", "0,0"} {
		if err := validateRequestedGPUs(gpuSelectionBox(), &launchRequest{GPUsFlag: flag}); err == nil {
			t.Errorf("--gpus %q was accepted", flag)
		}
	}
}
