package placement

import (
	"math"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestObservePromptCacheReconstructsEntryBeforeFailedSave(t *testing.T) {
	logText := `
slot create_check: id  0 | task 58 | created context checkpoint 1 of 16 (pos_min = 0, pos_max = 2047, n_tokens = 2048, size = 871.657 MiB)
slot create_check: id  0 | task 58 | created context checkpoint 2 of 16 (pos_min = 0, pos_max = 4095, n_tokens = 4096, size = 871.673 MiB)
slot create_check: id  0 | task 77 | created context checkpoint 5 of 16 (pos_min = 0, pos_max = 6361, n_tokens = 6362, size = 871.690 MiB)
slot create_check: id  0 | task 77 | created context checkpoint 6 of 16 (pos_min = 0, pos_max = 6371, n_tokens = 6372, size = 871.690 MiB)
slot create_check: id  0 | task 87 | created context checkpoint 3 of 16 (pos_min = 0, pos_max = 4245, n_tokens = 4246, size = 871.674 MiB)
slot create_check: id  0 | task 97 | created context checkpoint 4 of 16 (pos_min = 0, pos_max = 4255, n_tokens = 4256, size = 871.674 MiB)
 - saving prompt with length 4256, total state size = 1050.393 MiB
`

	obs := ObservePromptCache(logText)
	want := 1050.393 + 6*871.690
	if math.Abs(obs.LargestEntryMB-want) > 0.001 {
		t.Fatalf("LargestEntryMB = %.3f, want %.3f", obs.LargestEntryMB, want)
	}
	if obs.Evicted != 0 || obs.Skipped != 0 {
		t.Fatalf("save-only observation counted pressure: evicted=%d skipped=%d", obs.Evicted, obs.Skipped)
	}
}

func TestObservePromptCacheDoesNotUndercutBackendFullEntryMeasurement(t *testing.T) {
	logText := `
slot create_check: id 0 | task 1 | created context checkpoint 16 of 16 (size = 74.930 MiB)
srv prompt_save: - saving prompt with length 86885, total state size = 2451.686 MiB (draft: 0.000 MiB)
srv alloc: - prompt state size 3650.558 MiB exceeds cache size limit 3584.000 MiB, skipping
`

	obs := ObservePromptCache(logText)
	if obs.LargestEntryMB < 3650.558 || obs.LargestEntryMB > 3650.6 {
		t.Fatalf("LargestEntryMB = %.3f, want at least backend measurement 3650.558", obs.LargestEntryMB)
	}
	if obs.Skipped != 1 {
		t.Fatalf("Skipped = %d, want 1", obs.Skipped)
	}
}

func TestMeasuredPromptCacheTargetRoundsUpWithoutBreakingHostCap(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576}},
		RAM:  detect.RAMInfo{TotalMB: 60000, FreeMB: 60000},
	}
	s := &Strategy{
		Type:                       MoEOffload,
		Parallel:                   1,
		PlannedHostFootprintMB:     30000,
		MeasuredPromptCacheEntryMB: 3650.558,
	}

	cram, _ := computeCRAM(caps, &ModelProfile{}, s, 0, 0)
	if cram != 7680 {
		t.Fatalf("CRAM = %d MiB, want 7680 MiB (ceil of two measured entries)", cram)
	}

	// The capacity rounding must never override the existing two-thirds host
	// safety ceiling. With only 9 GiB after load, the quantized cap is 5632 MiB.
	s.PlannedHostFootprintMB = 51000
	cram, _ = computeCRAM(caps, &ModelProfile{}, s, 0, 0)
	if cram != 5632 {
		t.Fatalf("capped CRAM = %d MiB, want 5632 MiB", cram)
	}
}
