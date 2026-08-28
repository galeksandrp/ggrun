//go:build linux

package benchmark

import (
	"os"
	"testing"
)

func TestParseLinuxProcessCounters(t *testing.T) {
	stat := []byte("123 (llama server) R 42 1 1 0 -1 4194560 10 0 1 0 250 50 0 0 20 0 8 0")
	cpuSeconds, ppid, ok := parseLinuxProcStat(stat)
	if !ok || cpuSeconds != 3 || ppid != 42 {
		t.Fatalf("stat parse=(%f,%d,%t)", cpuSeconds, ppid, ok)
	}
	if rss := parseLinuxProcRSSMB([]byte("Name:\tllama\nVmRSS:\t65536 kB\n")); rss != 64 {
		t.Fatalf("rss=%d MB", rss)
	}
	readBytes, writeBytes := parseLinuxProcIO([]byte("rchar: 10\nread_bytes: 8192\nwrite_bytes: 1024\n"))
	if readBytes != 8192 || writeBytes != 1024 {
		t.Fatalf("io=(%d,%d)", readBytes, writeBytes)
	}
}

func TestProcessTreeSamplerObservesCurrentProcess(t *testing.T) {
	sample := NewProcessTreeResourceSampler(os.Getpid())()
	if !sample.Process.Available || sample.Process.RSSMB <= 0 {
		t.Fatalf("current process was not observable: %+v", sample.Process)
	}
}
