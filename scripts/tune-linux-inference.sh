#!/usr/bin/env bash
# Report (and optionally apply) Linux knobs that help CPU-expert MoE prefill
# and decode. Capable CPU-expert launches pin llama.cpp to a safe contiguous
# physical-core set; this script is the remaining host policy that needs root.
#
# Usage:
#   scripts/tune-linux-inference.sh           # report only
#   sudo scripts/tune-linux-inference.sh --apply
set -euo pipefail

apply=0
if [[ "${1:-}" == "--apply" ]]; then
  apply=1
fi

need_root() {
  if [[ $apply -eq 1 && ${EUID} -ne 0 ]]; then
    echo "error: --apply needs root (sudo $0 --apply)" >&2
    exit 1
  fi
}

report() {
  local title="$1" current="$2" want="$3"
  if [[ "$current" == *"$want"* ]]; then
    printf 'ok    %s: %s\n' "$title" "$current"
  else
    printf 'tune  %s: %s  (want %s)\n' "$title" "$current" "$want"
  fi
}

echo "=== ggrun Linux inference host ==="
echo "physical cores pin is a llama-server flag (--cpu-range/--cpu-strict); this"
echo "script only covers sysfs/sysctl/nvidia that ggrun cannot set as a user."
echo

gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo missing)
report "cpufreq governor" "$gov" "performance"

thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo missing)
report "THP enabled" "$thp" "[always]"

defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo missing)
report "THP defrag" "$defrag" "[madvise]"

swap=$(cat /proc/sys/vm/swappiness 2>/dev/null || echo missing)
report "vm.swappiness" "$swap" "0"

persist="missing"
if command -v nvidia-smi >/dev/null 2>&1; then
  persist=$(nvidia-smi -q 2>/dev/null | awk -F: '/Persistence Mode/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' || true)
  persist=${persist:-missing}
fi
report "nvidia persistence" "$persist" "Enabled"

echo
echo "Not changed here (reboot or dedicated inference kernel): isolcpus/nohz_full,"
echo "SMT off, Spectre mitigations=off. Those are not required to prove the"
echo "CPU-expert DRAM bound; they are last-percent host policy."

if [[ $apply -eq 0 ]]; then
  echo
  echo "Re-run with --apply as root to set governor, THP defrag=madvise, swappiness=0,"
  echo "and nvidia persistence."
  exit 0
fi

need_root

if [[ -d /sys/devices/system/cpu ]]; then
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -w $g ]] && echo performance >"$g" || true
  done
fi
if [[ -w /sys/kernel/mm/transparent_hugepage/defrag ]]; then
  echo madvise >/sys/kernel/mm/transparent_hugepage/defrag
fi
sysctl -w vm.swappiness=0 >/dev/null
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -pm 1 >/dev/null || true
fi
echo "applied."
