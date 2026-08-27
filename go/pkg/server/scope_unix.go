//go:build linux

package server

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const scopedParentWatchScript = `parent=$1
shift
cg=$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup 2>/dev/null)
if [ -n "$cg" ] && [ -w "/sys/fs/cgroup${cg}/memory.oom.group" ]; then
  printf '1\n' > "/sys/fs/cgroup${cg}/memory.oom.group" 2>/dev/null || true
fi
setsid "$@" &
child=$!
watcher=
terminate_child() {
  trap - HUP INT TERM
  if [ -n "$watcher" ]; then
    kill "$watcher" 2>/dev/null || true
  fi
  kill -TERM -"$child" 2>/dev/null || true
  sleep 2
  kill -KILL -"$child" 2>/dev/null || true
}
trap 'terminate_child; exit 143' HUP INT TERM
(
  while kill -0 "$parent" 2>/dev/null; do
    sleep 1
  done
  kill -TERM -"$child" 2>/dev/null
  sleep 2
  kill -KILL -"$child" 2>/dev/null
) &
watcher=$!
wait "$child"
status=$?
trap - HUP INT TERM
kill "$watcher" 2>/dev/null
wait "$watcher" 2>/dev/null
exit "$status"`

func scopedCommandArgs(args []string, memoryMaxMB int) ([]string, error) {
	return scopedCommandArgsWithUnit(args, memoryMaxMB, "ggrun-test.scope")
}

func scopedCommandArgsWithUnit(args []string, memoryMaxMB int, unit string) ([]string, error) {
	return scopedCommandArgsWithLimits(args, 0, memoryMaxMB, unit)
}

func scopedCommandArgsWithLimits(args []string, memoryHighMB, memoryMaxMB int, unit string) ([]string, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("start server: empty argv")
	}
	if memoryMaxMB <= 0 {
		return args, nil
	}
	systemdRun, err := exec.LookPath("systemd-run")
	if err != nil {
		return nil, fmt.Errorf("backend memory containment requires systemd-run: %w", err)
	}
	out := []string{
		systemdRun,
		"--user",
		"--scope",
		"--quiet",
	}
	if unit != "" {
		// Do not use systemd-run --collect here. A failed/OOM-killed transient
		// scope can otherwise disappear before Process.Stop reads memory.peak and
		// memory.events, losing the evidence needed to classify the failure.
		out = append(out, "--unit", unit)
	}
	out = append(out,
		"-p", "MemoryAccounting=yes",
	)
	if memoryHighMB > 0 {
		if memoryHighMB > memoryMaxMB {
			memoryHighMB = memoryMaxMB
		}
		out = append(out, "-p", fmt.Sprintf("MemoryHigh=%dM", memoryHighMB))
	}
	out = append(out,
		"-p", fmt.Sprintf("MemoryMax=%dM", memoryMaxMB),
		"-p", "MemorySwapMax=0",
		"-p", "OOMPolicy=kill",
		// SIGTERM only the wrapper; its trap terminates the setsid child group
		// once. control-group also signalled the child directly, so the wrapper's
		// own TERM became llama-server's fatal \"second interrupt\".
		"-p", "KillMode=mixed",
		"--",
	)
	wrapper := []string{"/bin/sh", "-c", scopedParentWatchScript, "ggrun-scope-watch", strconv.Itoa(os.Getpid())}
	out = append(out, wrapper...)
	return append(out, args...), nil
}

func stopScopeUnit(unit string) error {
	if unit == "" {
		return nil
	}
	err := exec.Command("systemctl", "--user", "stop", unit).Run()
	// A transient scope may disappear between the activity check and stop.
	// systemctl returns exit 5 for that already-stopped state; teardown has
	// nevertheless achieved its only required outcome.
	if err != nil && !scopeUnitActive(unit) {
		return nil
	}
	return err
}

func resetFailedScopeUnit(unit string) error {
	if unit == "" {
		return nil
	}
	return exec.Command("systemctl", "--user", "reset-failed", unit).Run()
}

func waitScopeUnitStopped(unit string, timeout time.Duration) error {
	if unit == "" {
		return nil
	}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !scopeUnitActive(unit) {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("systemd scope %s did not stop within %s", unit, timeout)
}

func scopeUnitActive(unit string) bool {
	return exec.Command("systemctl", "--user", "is-active", "--quiet", unit).Run() == nil
}

func scopeMemoryPeakBytes(unit string) (uint64, error) {
	peak, _, peakErr, _ := scopeMemoryStats(unit)
	return peak, peakErr
}

func scopeMemoryStats(unit string) (uint64, uint64, error, error) {
	cgroup, err := scopeControlGroup(unit)
	if err != nil {
		return scopeUnitMemoryStats(unit)
	}
	data, err := os.ReadFile("/sys/fs/cgroup" + cgroup + "/memory.peak")
	if err != nil {
		return scopeUnitMemoryStats(unit)
	}
	peak, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0, 0, fmt.Errorf("parse scope memory.peak: %w", err), nil
	}
	oomKills, oomErr := scopeMemoryOOMKillCountAt(cgroup)
	if oomErr != nil {
		return scopeUnitMemoryStats(unit)
	}
	return peak, oomKills, nil, oomErr
}

func scopeUnitMemoryStats(unit string) (uint64, uint64, error, error) {
	peakOut, peakErr := exec.Command("systemctl", "--user", "show", "--property=MemoryPeak", "--value", unit).Output()
	peak := uint64(0)
	if peakErr == nil {
		peak, peakErr = strconv.ParseUint(strings.TrimSpace(string(peakOut)), 10, 64)
	}
	if peakErr != nil {
		peakErr = fmt.Errorf("read scope MemoryPeak property: %w", peakErr)
	}
	resultOut, resultErr := exec.Command("systemctl", "--user", "show", "--property=Result", "--value", unit).Output()
	oomKills := uint64(0)
	if resultErr == nil && strings.TrimSpace(string(resultOut)) == "oom-kill" {
		oomKills = 1
	}
	if resultErr != nil {
		resultErr = fmt.Errorf("read scope Result property: %w", resultErr)
	}
	return peak, oomKills, peakErr, resultErr
}

func scopeControlGroup(unit string) (string, error) {
	if unit == "" {
		return "", fmt.Errorf("empty systemd scope unit")
	}
	out, err := exec.Command("systemctl", "--user", "show", "--property=ControlGroup", "--value", unit).Output()
	if err != nil {
		return "", fmt.Errorf("read scope control group: %w", err)
	}
	cgroup := strings.TrimSpace(string(out))
	if cgroup == "" {
		return "", fmt.Errorf("scope %s has no control group", unit)
	}
	return cgroup, nil
}

func scopeMemoryOOMKillCount(unit string) (uint64, error) {
	cgroup, err := scopeControlGroup(unit)
	if err != nil {
		return 0, err
	}
	return scopeMemoryOOMKillCountAt(cgroup)
}

// ScopeNonReclaimableMB reports the backend's current non-reclaimable host
// footprint (anon + shmem + slab) from its own cgroup's memory.stat. This is
// the measured post-launch footprint Fix B sizes the running scope's MemoryMax
// against: model/server anon RSS, CPU-expert CUDA host buffers (shmem), KV,
// context checkpoints, and prompt cache. Page cache is excluded as reclaimable.
func (p *Process) ScopeNonReclaimableMB() (int, error) {
	if p == nil || p.scopeUnit == "" {
		return 0, fmt.Errorf("backend has no memory scope")
	}
	cgroup, err := scopeControlGroup(p.scopeUnit)
	if err != nil {
		return 0, err
	}
	return scopeNonReclaimableMB(cgroup)
}

// SetMemoryMaxMB raises the running backend scope's hard ceiling to memoryMaxMB
// MiB. It is the post-launch half of measured-footprint containment: once the
// backend is healthy and the canary has run, ggrun re-sizes the scope to the
// real measured footprint plus headroom instead of the pre-launch plan estimate.
// The clamp to the whole-host ceiling is the caller's responsibility.
func (p *Process) SetMemoryMaxMB(memoryMaxMB int) error {
	if p == nil || p.scopeUnit == "" {
		return fmt.Errorf("backend has no memory scope")
	}
	return setScopeMemoryMaxMB(p.scopeUnit, memoryMaxMB)
}

// SetMemoryHighMB updates the running backend scope's reclaim/throttle
// boundary without lowering its hard ceiling. Mmap-backed launches use this
// after the first real canary: measured anonymous state belongs below
// memory.high, while clean model pages still need the larger memory.max band in
// which the kernel can evict and re-fault them.
func (p *Process) SetMemoryHighMB(memoryHighMB int) error {
	if p == nil || p.scopeUnit == "" {
		return fmt.Errorf("backend has no memory scope")
	}
	return setScopeMemoryHighMB(p.scopeUnit, memoryHighMB)
}

func scopeMemoryOOMKillCountAt(cgroup string) (uint64, error) {
	data, err := os.ReadFile("/sys/fs/cgroup" + cgroup + "/memory.events")
	if err != nil {
		return 0, fmt.Errorf("read scope memory.events: %w", err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 || fields[0] != "oom_kill" {
			continue
		}
		return strconv.ParseUint(fields[1], 10, 64)
	}
	return 0, nil
}

// scopeNonReclaimableMB reports the scope's non-reclaimable host memory:
// anon + shmem + slab from memory.stat. This is exactly what the backend's
// resident footprint is -- model/server anonymous RSS, CUDA host buffers for
// CPU experts (shmem), KV, context checkpoints, and prompt cache. Page cache
// (file-backed experts) is deliberately excluded: the kernel can evict it
// under pressure, so no plan or limit should reserve it.
func scopeNonReclaimableMB(cgroup string) (int, error) {
	if cgroup == "" {
		return 0, fmt.Errorf("empty scope control group")
	}
	statData, err := os.ReadFile("/sys/fs/cgroup" + cgroup + "/memory.stat")
	if err != nil {
		return 0, fmt.Errorf("read scope memory.stat: %w", err)
	}
	var totalBytes int64
	for _, line := range strings.Split(string(statData), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		switch fields[0] {
		case "anon", "shmem", "slab":
			if v, convErr := strconv.ParseInt(fields[1], 10, 64); convErr == nil && v > 0 {
				totalBytes += v
			}
		}
	}
	return int(totalBytes / (1024 * 1024)), nil
}

// setScopeMemoryMaxMB raises the running scope's hard memory ceiling by writing
// the cgroup's memory.max directly. The user owns the scope, so a direct write
// is allowed and avoids systemd's transient-unit set-property quirks; a
// systemctl --user set-property fallback keeps DBus in the loop when the
// direct write is not permitted.
func setScopeMemoryMaxMB(unit string, memoryMaxMB int) error {
	return setScopeMemoryLimitMB(unit, "memory.max", "MemoryMax", memoryMaxMB)
}

func setScopeMemoryHighMB(unit string, memoryHighMB int) error {
	return setScopeMemoryLimitMB(unit, "memory.high", "MemoryHigh", memoryHighMB)
}

func setScopeMemoryLimitMB(unit, cgroupFile, systemdProperty string, limitMB int) error {
	if unit == "" {
		return fmt.Errorf("empty scope unit")
	}
	if limitMB <= 0 {
		return fmt.Errorf("invalid %s %d MiB", cgroupFile, limitMB)
	}
	cgroup, err := scopeControlGroup(unit)
	if err != nil {
		return err
	}
	bytes := uint64(limitMB) * 1024 * 1024
	path := "/sys/fs/cgroup" + cgroup + "/" + cgroupFile
	if err := os.WriteFile(path, []byte(strconv.FormatUint(bytes, 10)), 0o644); err == nil {
		return nil
	}
	// Fallback: systemctl set-property on the transient unit. This goes through
	// DBus and may fail for a scope that was never fully registered, which the
	// caller treats as a non-fatal signal.
	if exec.Command("systemctl", "--user", "set-property", unit, systemdProperty, fmt.Sprintf("%d", bytes)).Run() == nil {
		return nil
	}
	return fmt.Errorf("set scope %s at %s", cgroupFile, path)
}
