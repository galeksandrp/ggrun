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
	Index       int    `json:"index,omitempty"`
	NVIDIAIndex int    `json:"-"`
	PCIBusID    string `json:"pci_bus_id,omitempty"`
	SMPercent   int    `json:"sm_percent"`
	MemPercent  int    `json:"mem_percent"`
	PCIeRXMBps  int    `json:"pcie_rx_mbps,omitempty"`
	PCIeTXMBps  int    `json:"pcie_tx_mbps,omitempty"`
}

// SampleGPUUtilization returns current SM/memory utilization, or nil when the
// vendor tool is missing. Callers must fail closed on an empty result.
func SampleGPUUtilization() []GPUUtilization {
	ctx, cancel := context.WithTimeout(context.Background(), gpuUtilizationCommandTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=index,pci.bus_id,utilization.gpu,utilization.memory",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	samples := parseNVIDIAUtilization(string(out))
	transferOut, transferErr := exec.CommandContext(ctx, "nvidia-smi", "dmon", "-s", "t", "-c", "1").Output()
	if transferErr == nil {
		transfers := parseNVIDIATransfers(string(transferOut))
		for i := range samples {
			if pair, ok := transfers[samples[i].NVIDIAIndex]; ok {
				samples[i].PCIeRXMBps, samples[i].PCIeTXMBps = pair[0], pair[1]
			}
		}
	}
	return samples
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
		offset, nvidiaIndex := 0, -1
		if len(parts) >= 4 {
			if parsed, err := strconv.Atoi(strings.TrimSpace(parts[0])); err == nil {
				nvidiaIndex, offset = parsed, 1
			}
		}
		sm, errSM := strconv.Atoi(strings.TrimSpace(parts[offset+1]))
		mem, errMem := strconv.Atoi(strings.TrimSpace(parts[offset+2]))
		if errSM != nil || errMem != nil {
			continue
		}
		out = append(out, GPUUtilization{
			NVIDIAIndex: nvidiaIndex,
			PCIBusID:    strings.TrimSpace(parts[offset]),
			SMPercent:   sm,
			MemPercent:  mem,
		})
	}
	return out
}

func parseNVIDIATransfers(output string) map[int][2]int {
	out := map[int][2]int{}
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		index, errIndex := strconv.Atoi(fields[0])
		rx, errRX := strconv.Atoi(fields[1])
		tx, errTX := strconv.Atoi(fields[2])
		if errIndex == nil && errRX == nil && errTX == nil {
			out[index] = [2]int{rx, tx}
		}
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
