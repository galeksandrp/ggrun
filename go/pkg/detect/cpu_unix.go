//go:build !windows

package detect

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

// detectPhysicalCores returns the number of physical CPU cores on Linux/Darwin.
func detectPhysicalCores() int {
	// Darwin: use sysctl
	if runtime.GOOS == "darwin" {
		out, err := exec.Command("sysctl", "-n", "hw.physicalcpu").Output()
		if err == nil {
			if n, e := strconv.Atoi(strings.TrimSpace(string(out))); e == nil && n > 0 {
				return n
			}
		}
	}

	// Linux: parse /proc/cpuinfo for physical cores
	data, err := os.ReadFile("/proc/cpuinfo")
	if err == nil {
		seen := make(map[string]bool)
		var physID, coreID string
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			switch {
			case strings.HasPrefix(line, "physical id"):
				if parts := strings.SplitN(line, ":", 2); len(parts) == 2 {
					physID = strings.TrimSpace(parts[1])
				}
			case strings.HasPrefix(line, "core id"):
				if parts := strings.SplitN(line, ":", 2); len(parts) == 2 {
					coreID = strings.TrimSpace(parts[1])
				}
				if physID != "" && coreID != "" {
					seen[physID+":"+coreID] = true
					physID = ""
					coreID = ""
				}
			}
		}
		if n := len(seen); n > 0 {
			return n
		}
	}

	// Fallback: logical cores / 2 (HT assumption)
	n := runtime.NumCPU()
	if n >= 4 {
		return n / 2
	}
	return n
}

func detectPhysicalCPUList() []int {
	// Linux sysfs plus the process affinity mask are the only topology source we
	// can map safely to llama.cpp CPU IDs. Darwin's physical-core count does not
	// identify P/E cores or logical IDs, so leave affinity unspecified there.
	if runtime.GOOS != "linux" {
		return nil
	}
	entries, err := os.ReadDir("/sys/devices/system/cpu")
	if err != nil {
		return nil
	}
	allowed := linuxAllowedCPUSet()
	type coreKey struct {
		pkg  string
		core string
	}
	first := map[coreKey]int{}
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasPrefix(name, "cpu") {
			continue
		}
		id, err := strconv.Atoi(strings.TrimPrefix(name, "cpu"))
		if err != nil {
			continue
		}
		if len(allowed) > 0 && !allowed[id] {
			continue
		}
		if online, err := os.ReadFile(filepath.Join("/sys/devices/system/cpu", name, "online")); err == nil && strings.TrimSpace(string(online)) == "0" {
			continue
		}
		base := filepath.Join("/sys/devices/system/cpu", name, "topology")
		pkg, err := os.ReadFile(filepath.Join(base, "physical_package_id"))
		if err != nil {
			continue
		}
		core, err := os.ReadFile(filepath.Join(base, "core_id"))
		if err != nil {
			continue
		}
		key := coreKey{pkg: strings.TrimSpace(string(pkg)), core: strings.TrimSpace(string(core))}
		if prev, ok := first[key]; !ok || id < prev {
			first[key] = id
		}
	}
	if len(first) == 0 {
		return nil
	}
	ids := make([]int, 0, len(first))
	for _, id := range first {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}

func linuxAllowedCPUSet() map[int]bool {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return nil
	}
	for _, line := range strings.Split(string(data), "\n") {
		name, value, found := strings.Cut(line, ":")
		if found && strings.TrimSpace(name) == "Cpus_allowed_list" {
			return parseLinuxCPUList(value)
		}
	}
	return nil
}

func parseLinuxCPUList(value string) map[int]bool {
	out := map[int]bool{}
	for _, item := range strings.Split(strings.TrimSpace(value), ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		loText, hiText, ranged := strings.Cut(item, "-")
		lo, err := strconv.Atoi(loText)
		if err != nil || lo < 0 {
			continue
		}
		hi := lo
		if ranged {
			hi, err = strconv.Atoi(hiText)
			if err != nil || hi < lo {
				continue
			}
		}
		for id := lo; id <= hi; id++ {
			out[id] = true
		}
	}
	return out
}

// detectRAMFreeMB returns available RAM in MB on Linux/Darwin.
func detectRAMFreeMB() int {
	if runtime.GOOS == "darwin" {
		// macOS: use sysctl + vm_stat (approximate)
		out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
		if err == nil {
			totalBytes, _ := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
			// Return ~80% of total as "free" (macOS manages memory aggressively)
			return int(totalBytes / 1024 / 1024 * 80 / 100)
		}
	}
	// Linux: /proc/meminfo
	data, err := os.ReadFile("/proc/meminfo")
	if err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "MemAvailable:") {
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					kb, _ := strconv.Atoi(parts[1])
					return kb / 1024
				}
			}
		}
	}
	return 4096 // fallback
}

func detectRAMWindows() RAMInfo {
	return RAMInfo{}
}
