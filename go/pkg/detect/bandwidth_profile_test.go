package detect

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func bandwidthTestCaps() *Capabilities {
	return &Capabilities{
		OS:   "linux",
		Arch: "amd64",
		CPU:  CPUInfo{Model: "Test CPU", Cores: 8, Threads: 16},
		RAM:  RAMInfo{TotalMB: 65536, FreeMB: 42000},
		GPUs: []GPU{
			{Index: 0, Name: "GPU A", VRAMTotalMB: 24576, PCIBusID: "00000000:01:00.0", ComputeCap: "8.6", Driver: "1", BandwidthMBps: 1000},
			{Index: 1, Name: "GPU B", VRAMTotalMB: 12288, PCIBusID: "0000:04:00.0", ComputeCap: "8.9", Driver: "1", BandwidthMBps: 2000},
		},
	}
}

func bandwidthTestProfile(caps *Capabilities) *BandwidthProfile {
	return &BandwidthProfile{
		Schema:             bandwidthProfileSchema,
		HardwareKey:        HardwareBandwidthKey(caps),
		MeasuredAt:         time.Now().UTC(),
		Source:             "cuda_driver_pinned_copy",
		Bytes:              bandwidthProbeBytes,
		MinIterations:      4,
		HostCopyMBps:       48000,
		HostCopyIterations: 100,
		HostCopyWorkers:    8,
		GPUs: []GPUBandwidthMeasurement{
			{PCIBusID: "0000:04:00.0", H2DMBps: 7100, D2HMBps: 6900, H2DIterations: 8, D2HIterations: 8},
			{PCIBusID: "0000:01:00.0", H2DMBps: 24500, D2HMBps: 23800, H2DIterations: 16, D2HIterations: 16},
		},
	}
}

func TestHardwareBandwidthKeyIgnoresVolatileState(t *testing.T) {
	caps := bandwidthTestCaps()
	want := HardwareBandwidthKey(caps)
	caps.RAM.FreeMB = 100
	caps.GPUs[0].VRAMUsedMB = 20000
	caps.GPUs[0].BandwidthMBps = 99999
	caps.GPUs[0].BandwidthSource = "changed"
	caps.GPUs[0].Driver = "new-driver"
	if got := HardwareBandwidthKey(caps); got != want {
		t.Fatalf("volatile state changed hardware key: %s != %s", got, want)
	}
}

func TestHardwareBandwidthKeyChangesWithPhysicalGPU(t *testing.T) {
	caps := bandwidthTestCaps()
	want := HardwareBandwidthKey(caps)
	caps.GPUs[1].PCIBusID = "0000:05:00.0"
	if got := HardwareBandwidthKey(caps); got == want {
		t.Fatal("moving a GPU must invalidate the bandwidth profile")
	}
}

func TestApplyBandwidthProfileUsesBusIDAndMeasuredHostCopy(t *testing.T) {
	caps := bandwidthTestCaps()
	profile := bandwidthTestProfile(caps)
	if !ApplyBandwidthProfile(caps, profile) {
		t.Fatal("valid profile was not applied")
	}
	if got := caps.GPUs[0].BandwidthMBps; got != 24500 {
		t.Fatalf("GPU 0 bandwidth = %d, want 24500", got)
	}
	if got := caps.GPUs[1].BandwidthMBps; got != 7100 {
		t.Fatalf("GPU 1 bandwidth = %d, want 7100", got)
	}
	for _, gpu := range caps.GPUs {
		if gpu.BandwidthSource != "measured_pinned_h2d" {
			t.Fatalf("unexpected bandwidth source: %q", gpu.BandwidthSource)
		}
	}
	if caps.HostMemoryBandwidthMBps != 48000 || caps.HostMemoryBandwidthSource != "measured_parallel_memcpy" {
		t.Fatalf("host measurement not applied: %+v", caps)
	}
}

func TestApplyBandwidthProfileRejectsPartialWithoutMutation(t *testing.T) {
	caps := bandwidthTestCaps()
	profile := bandwidthTestProfile(caps)
	profile.GPUs = profile.GPUs[:1]
	if ApplyBandwidthProfile(caps, profile) {
		t.Fatal("partial profile should be rejected")
	}
	if caps.GPUs[0].BandwidthMBps != 1000 || caps.GPUs[1].BandwidthMBps != 2000 {
		t.Fatalf("rejected profile partially mutated capabilities: %+v", caps.GPUs)
	}
}

func TestApplyBandwidthProfileRejectsHardwareMismatch(t *testing.T) {
	caps := bandwidthTestCaps()
	profile := bandwidthTestProfile(caps)
	profile.HardwareKey = "other-hardware"
	if ApplyBandwidthProfile(caps, profile) {
		t.Fatal("mismatched profile should be rejected")
	}
	if caps.HostMemoryBandwidthMBps != 0 {
		t.Fatal("mismatched profile mutated host bandwidth")
	}
}

func TestSaveLoadAndApplyCachedBandwidthProfile(t *testing.T) {
	caps := bandwidthTestCaps()
	profile := bandwidthTestProfile(caps)
	path := filepath.Join(t.TempDir(), "nested", "profile.json")
	if err := SaveBandwidthProfile(path, profile); err != nil {
		t.Fatalf("save profile: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat profile: %v", err)
	}
	if info.Mode().Perm()&0o077 != 0 {
		t.Fatalf("profile permissions = %o, want private", info.Mode().Perm())
	}
	loaded, err := LoadBandwidthProfile(path)
	if err != nil {
		t.Fatalf("load profile: %v", err)
	}
	if loaded.HardwareKey != profile.HardwareKey || loaded.HostCopyMBps != profile.HostCopyMBps {
		t.Fatalf("loaded profile changed: %+v", loaded)
	}

	t.Setenv("LLM_BANDWIDTH_PROFILE", path)
	fresh := bandwidthTestCaps()
	if err := ApplyCachedBandwidthProfile(fresh); err != nil {
		t.Fatalf("apply cached profile: %v", err)
	}
	if fresh.GPUs[0].BandwidthMBps != 24500 {
		t.Fatalf("cached profile was not applied: %+v", fresh.GPUs)
	}
}

func TestCanonicalPCIBusIDNormalizesNVIDIADomainWidth(t *testing.T) {
	if got := canonicalPCIBusID("00000000:01:00.0"); got != "0000:01:00.0" {
		t.Fatalf("canonical bus ID = %q", got)
	}
}
