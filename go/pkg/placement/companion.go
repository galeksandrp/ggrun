package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// A companion's VRAM footprint was a constant in the launcher: 2600 MiB for the
// Claude reviewer, described in its own comment as "a conservative bound" whose
// real usage "is visible in the normal probe paths after a real launch". Nothing
// read it back, so the bound stood forever.
//
// Measured on this project the reviewer occupies 2114 MiB, so the reservation
// overshot by 486 MiB. That is small in isolation and not small in placement:
// the reviewer is deliberately seated on the least valuable GPU, which here is a
// 12 GB card, and an expert layer is 1371 MiB. A permanent 486 MiB withheld from
// the smallest device is 35% of a layer that never comes back.
//
// This is the same correction already applied to compute buffers, runtime graph
// growth, KV geometry and the prompt cache: read what the process actually took
// rather than reserving a number derived from nothing.

// companionVRAMPath is one file per companion name, since a companion is defined
// by its role and model rather than by the target model it accompanies.
func companionVRAMPath(cacheDir, name string) string {
	if strings.TrimSpace(name) == "" {
		return ""
	}
	safe := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		}
		return '-'
	}, name)
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	return filepath.Join(cacheDir, fmt.Sprintf("companion_%s.vram", safe))
}

// RecordCompanionVRAM stores a companion's measured on-device footprint.
//
// Keeps the largest sample: the reviewer's usage grows as its KV fills, and a
// reservation sized from an idle moment would be revised downward every launch
// and overrun on the next long conversation.
func RecordCompanionVRAM(cacheDir, name string, usedMB int) error {
	path := companionVRAMPath(cacheDir, name)
	if path == "" || usedMB <= 0 {
		return nil
	}
	if prev := MeasuredCompanionVRAMMB(cacheDir, name); prev >= usedMB {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	body := fmt.Sprintf("# Measured VRAM for companion %s\nCOMPANION_VRAM_MB=%d\n", name, usedMB)
	return os.WriteFile(path, []byte(body), 0644)
}

// MeasuredCompanionVRAMMB returns a stored measurement, or 0 when this companion
// has never been observed running.
func MeasuredCompanionVRAMMB(cacheDir, name string) int {
	path := companionVRAMPath(cacheDir, name)
	if path == "" {
		return 0
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "COMPANION_VRAM_MB=") {
			continue
		}
		if v, err := strconv.Atoi(strings.TrimPrefix(line, "COMPANION_VRAM_MB=")); err == nil && v > 0 {
			return v
		}
	}
	return 0
}
