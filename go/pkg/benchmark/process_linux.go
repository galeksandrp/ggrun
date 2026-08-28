//go:build linux

package benchmark

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Linux exposes process times in USER_HZ units through /proc/<pid>/stat. The
// kernel userspace ABI defines USER_HZ as 100 on supported Linux architectures,
// independently of the scheduler's internal tick rate.
const linuxUserHZ = 100.0

// NewProcessTreeResourceSampler returns a non-blocking /proc sampler for a
// launched backend and its stable wrapper descendants. Discovering the family
// once avoids scanning all of /proc at every 250 ms observation.
func NewProcessTreeResourceSampler(rootPID int) func() ResourceSnapshot {
	pids := linuxProcessFamily(rootPID)
	return func() ResourceSnapshot {
		process := sampleLinuxProcessFamily(pids)
		if !process.Available && rootPID > 0 {
			pids = linuxProcessFamily(rootPID)
			process = sampleLinuxProcessFamily(pids)
		}
		return ResourceSnapshot{Process: process}
	}
}

func sampleLinuxProcessFamily(pids []int) ProcessResourceSnapshot {
	var out ProcessResourceSnapshot
	for _, pid := range pids {
		stat, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
		if err != nil {
			continue
		}
		cpuSeconds, _, ok := parseLinuxProcStat(stat)
		if !ok {
			continue
		}
		out.Available = true
		out.CPUTimeSeconds += cpuSeconds
		if status, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid)); err == nil {
			out.RSSMB += parseLinuxProcRSSMB(status)
		}
		if ioData, err := os.ReadFile(fmt.Sprintf("/proc/%d/io", pid)); err == nil {
			readBytes, writeBytes := parseLinuxProcIO(ioData)
			out.ReadBytes += readBytes
			out.WriteBytes += writeBytes
		}
	}
	return out
}

func linuxProcessFamily(rootPID int) []int {
	if rootPID <= 0 {
		return nil
	}
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return []int{rootPID}
	}
	children := map[int][]int{}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		stat, err := os.ReadFile(filepath.Join("/proc", entry.Name(), "stat"))
		if err != nil {
			continue
		}
		_, ppid, ok := parseLinuxProcStat(stat)
		if ok && ppid > 0 {
			children[ppid] = append(children[ppid], pid)
		}
	}
	seen := map[int]bool{rootPID: true}
	out := []int{rootPID}
	for cursor := 0; cursor < len(out); cursor++ {
		for _, child := range children[out[cursor]] {
			if seen[child] {
				continue
			}
			seen[child] = true
			out = append(out, child)
		}
	}
	return out
}

// parseLinuxProcStat returns aggregate CPU seconds and PPID. The executable
// name is parenthesized and may itself contain spaces or ')' characters, so the
// parser anchors on the final ')' before indexing the documented fields.
func parseLinuxProcStat(data []byte) (cpuSeconds float64, ppid int, ok bool) {
	closeParen := bytes.LastIndexByte(data, ')')
	if closeParen < 0 || closeParen+1 >= len(data) {
		return 0, 0, false
	}
	fields := strings.Fields(string(data[closeParen+1:]))
	if len(fields) <= 12 {
		return 0, 0, false
	}
	parsedPPID, err := strconv.Atoi(fields[1])
	if err != nil {
		return 0, 0, false
	}
	userTicks, userErr := strconv.ParseUint(fields[11], 10, 64)
	systemTicks, systemErr := strconv.ParseUint(fields[12], 10, 64)
	if userErr != nil || systemErr != nil {
		return 0, 0, false
	}
	return float64(userTicks+systemTicks) / linuxUserHZ, parsedPPID, true
}

func parseLinuxProcRSSMB(data []byte) int {
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "VmRSS:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kb, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return 0
		}
		return int(kb / 1024)
	}
	return 0
}

func parseLinuxProcIO(data []byte) (readBytes, writeBytes uint64) {
	for _, line := range strings.Split(string(data), "\n") {
		name, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		parsed, err := strconv.ParseUint(strings.TrimSpace(value), 10, 64)
		if err != nil {
			continue
		}
		switch strings.TrimSpace(name) {
		case "read_bytes":
			readBytes = parsed
		case "write_bytes":
			writeBytes = parsed
		}
	}
	return readBytes, writeBytes
}
