package detect

import "strings"

// VRAM memory bandwidth, not PCIe link bandwidth.
//
// For a fully-resident dense model the weights are read from VRAM once per
// generated token and almost nothing crosses PCIe, so per-card memory bandwidth
// -- not link width -- is what decides how long that card's layers take. nvidia-smi
// reports the memory clock but not the bus width, so the width comes from this
// table and the bandwidth is derived:
//
//	MB/s = clockMHz * 2 (DDR) * busWidthBits / 8
//
// e.g. an RTX 3090 Ti at 10501 MHz on 384 bits -> 1,008,096 MB/s.
//
// A card missing from the table yields 0, and every caller falls back to the
// PCIe-derived BandwidthMBps it used before. Unknown hardware therefore keeps
// today's behaviour rather than getting a wrong number.
var gpuMemoryBusWidthBits = []struct {
	match string // lowercased substring, matched longest-first
	bits  int
}{
	// Blackwell / Ada / Ampere / Turing consumer
	{"rtx 5090", 512}, {"rtx 5080", 256}, {"rtx 5070 ti", 256}, {"rtx 5070", 192},
	{"rtx 4090", 384}, {"rtx 4080 super", 256}, {"rtx 4080", 256},
	{"rtx 4070 ti super", 256}, {"rtx 4070 ti", 192}, {"rtx 4070 super", 192}, {"rtx 4070", 192},
	{"rtx 4060 ti", 128}, {"rtx 4060", 128},
	{"rtx 3090 ti", 384}, {"rtx 3090", 384}, {"rtx 3080 ti", 384}, {"rtx 3080", 320},
	{"rtx 3070 ti", 256}, {"rtx 3070", 256}, {"rtx 3060 ti", 256}, {"rtx 3060", 192},
	{"rtx 3050", 128},
	{"rtx 2080 ti", 352}, {"rtx 2080", 256}, {"rtx 2070", 256}, {"rtx 2060", 192},
	{"gtx 1080 ti", 352}, {"gtx 1080", 256}, {"gtx 1070", 256}, {"gtx 1060", 192},
	// Workstation / datacenter
	{"rtx 6000 ada", 384}, {"rtx 5000 ada", 256}, {"rtx 4500 ada", 192}, {"rtx 4000 ada", 160},
	{"a100", 5120}, {"h100", 5120}, {"h200", 6144},
	{"a6000", 384}, {"a5000", 384}, {"a4500", 320}, {"a4000", 256}, {"a2000", 192},
	{"l40s", 384}, {"l40", 384}, {"l4", 192},
	{"v100", 4096}, {"titan rtx", 384},
	{"tesla p40", 384}, {"tesla p100", 4096}, {"tesla t4", 256},
}

// memoryBusWidthBits returns the VRAM bus width for a GPU marketing name, or 0
// when the card is not in the table. Longest match wins so "rtx 3090 ti" is not
// shadowed by "rtx 3090", and "rtx 4070 ti super" not by "rtx 4070 ti".
func memoryBusWidthBits(name string) int {
	lower := strings.ToLower(strings.TrimSpace(name))
	if lower == "" {
		return 0
	}
	best, bestLen := 0, 0
	for _, entry := range gpuMemoryBusWidthBits {
		if len(entry.match) > bestLen && strings.Contains(lower, entry.match) {
			best, bestLen = entry.bits, len(entry.match)
		}
	}
	return best
}

// memoryBandwidthMBps derives VRAM bandwidth from the reported memory clock and
// the card's bus width. Returns 0 when either input is unknown, which is the
// signal for callers to fall back to PCIe-derived weighting.
func memoryBandwidthMBps(name string, clockMHz int) int {
	if clockMHz <= 0 {
		return 0
	}
	bits := memoryBusWidthBits(name)
	if bits <= 0 {
		return 0
	}
	// DDR transfers twice per clock; bits/8 converts to bytes per transfer.
	return clockMHz * 2 * bits / 8
}
