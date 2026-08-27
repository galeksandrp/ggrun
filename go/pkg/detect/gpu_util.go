package detect

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const gpuUtilizationCommandTimeout = 2 * time.Second

// GPUUtilization is one nvidia-smi observation keyed by PCI bus id so CUDA
// device order cannot remap it onto the wrong card. Index is filled only after
// MapUtilizationToIndexes.
type GPUUtilization struct {
	Index      int    `json:"index,omitempty"`
	PCIBusID   string `json:"pci_bus_id,omitempty"`
	SMPercent  int    `json:"sm_percent"`
	MemPercent int    `json:"mem_percent"`
}

// SampleGPUUtilization returns current SM/memory utilization, or nil when the
// vendor tool is missing. Callers must fail closed on an empty result.
func SampleGPUUtilization() []GPUUtilization {
	ctx, cancel := context.WithTimeout(context.Background(), gpuUtilizationCommandTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=pci.bus_id,utilization.gpu,utilization.memory",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	return parseNVIDIAUtilization(string(out))
}

func parseNVIDIAUtilization(csv string) []GPUUtilization {
	var out []GPUUtilization
	for _, line := range strings.Split(strings.TrimSpace(csv), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ", ")
		if len(parts) < 3 {
			parts = strings.Split(line, ",")
		}
		if len(parts) < 3 {
			continue
		}
		sm, errSM := strconv.Atoi(strings.TrimSpace(parts[1]))
		mem, errMem := strconv.Atoi(strings.TrimSpace(parts[2]))
		if errSM != nil || errMem != nil {
			continue
		}
		out = append(out, GPUUtilization{
			PCIBusID:   strings.TrimSpace(parts[0]),
			SMPercent:  sm,
			MemPercent: mem,
		})
	}
	return out
}

// MapUtilizationToIndexes translates PCI-keyed samples onto ggrun GPU indexes.
func MapUtilizationToIndexes(gpus []GPU, samples []GPUUtilization) []GPUUtilization {
	if len(gpus) == 0 || len(samples) == 0 {
		return nil
	}
	byPCI := make(map[string]int, len(gpus))
	for _, gpu := range gpus {
		if gpu.PCIBusID != "" {
			byPCI[canonicalPCIBusID(gpu.PCIBusID)] = gpu.Index
		}
	}
	out := make([]GPUUtilization, 0, len(samples))
	for _, sample := range samples {
		idx, ok := byPCI[canonicalPCIBusID(sample.PCIBusID)]
		if !ok {
			continue
		}
		sample.Index = idx
		out = append(out, sample)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
