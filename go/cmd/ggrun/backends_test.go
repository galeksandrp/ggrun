package main

import (
	"strings"
	"testing"
)

// A CUDA fork must be built with the flash-attention kernels for every
// quantized KV type. Without them the planner can emit a K/V quantization the
// binary has no kernel for: the model loads its weights across the GPUs and then
// dies at context creation, which is how the Inkling fork failed on first
// registration (-ctk q4_0 -ctv q4_0 --flash-attn on).
//
// collectCMakeFlags already preserves this flag on update, so leaving it out of
// the initial build made a freshly registered fork worse than the same source
// after `ggrun backend update`.
func TestForkCUDABuildCompilesAllQuantFlashAttentionKernels(t *testing.T) {
	cfg, err := forkCMakeConfigureArgs("/b", "cuda", "86;89")
	if err != nil {
		t.Fatalf("cuda configure: %v", err)
	}
	joined := strings.Join(cfg, " ")
	for _, want := range []string{
		"-DGGML_CUDA=ON",
		"-DGGML_CUDA_FA_ALL_QUANTS=ON",
		"-DCMAKE_CUDA_ARCHITECTURES=86;89",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %s in %q", want, joined)
		}
	}
}

// An unspecified arch list must still produce a valid build rather than an
// empty CMAKE_CUDA_ARCHITECTURES.
func TestForkCUDABuildDefaultsArchToNative(t *testing.T) {
	cfg, err := forkCMakeConfigureArgs("/b", "cuda", "")
	if err != nil {
		t.Fatalf("cuda configure: %v", err)
	}
	if !strings.Contains(strings.Join(cfg, " "), "-DCMAKE_CUDA_ARCHITECTURES=native") {
		t.Errorf("expected native default, got %v", cfg)
	}
}

// The quant kernels are a CUDA concern; other accelerators must not gain the
// flag, and an unknown accelerator must still be rejected.
func TestForkNonCUDABuilds(t *testing.T) {
	cfg, err := forkCMakeConfigureArgs("/b", "vulkan", "")
	if err != nil {
		t.Fatalf("vulkan configure: %v", err)
	}
	if strings.Contains(strings.Join(cfg, " "), "FA_ALL_QUANTS") {
		t.Error("FA_ALL_QUANTS is CUDA-only")
	}
	if _, err := forkCMakeConfigureArgs("/b", "cpu", ""); err != nil {
		t.Fatalf("cpu configure: %v", err)
	}
	if _, err := forkCMakeConfigureArgs("/b", "bogus", ""); err == nil {
		t.Error("unknown accelerator must be rejected")
	}
}
