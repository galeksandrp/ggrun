package benchmark

import (
	"math"
	"sync"
	"testing"
	"time"
)

func TestPhaseResourceObservationRetainsBoundaryDeltas(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	runner := &Runner{
		ResourceSamplingInterval: time.Hour,
		SampleResources: func() ResourceSnapshot {
			mu.Lock()
			defer mu.Unlock()
			calls++
			return ResourceSnapshot{
				GPUUtilization: []GPUUtilization{{GPU: 1, SMPercent: 80, MemPercent: 40}},
				Process: ProcessResourceSnapshot{
					Available: true, CPUTimeSeconds: float64(calls) * 0.05,
					RSSMB: 100 + calls, ReadBytes: uint64(calls * 4096),
					WriteBytes: uint64(calls * 512),
				},
				Queue: QueueResourceSnapshot{Available: true, Active: calls, Deferred: calls - 1},
			}
		},
	}
	stop := startPhaseResourceObservation(runner, AgentPhasePrefill)
	time.Sleep(5 * time.Millisecond)
	got := stop()
	if repeated := stop(); repeated.Observations != got.Observations {
		t.Fatalf("stopper was not idempotent: first=%+v repeated=%+v", got, repeated)
	}
	if got.Phase != AgentPhasePrefill || got.Observations != 2 || got.ProcessObservations != 2 {
		t.Fatalf("phase boundary was not sampled exactly: %+v", got)
	}
	if math.Abs(got.ProcessCPUTimeS-0.05) > 1e-9 || got.ProcessCPUPercent <= 0 {
		t.Fatalf("CPU delta was not derived: %+v", got)
	}
	if got.ProcessReadBytes != 4096 || got.ProcessWriteBytes != 512 || got.ProcessRSSPeakMB != 102 {
		t.Fatalf("process counters were not retained: %+v", got)
	}
	if len(got.GPUUtilization) != 1 || got.GPUUtilization[0].GPU != 1 ||
		got.GPUUtilization[0].Observations != 2 {
		t.Fatalf("GPU phase evidence was not retained: %+v", got.GPUUtilization)
	}
	if got.QueueObservations != 2 || got.QueueActiveAvg != 1.5 || got.QueueDeferredMax != 1 {
		t.Fatalf("queue evidence was not aggregated: %+v", got)
	}
}

func TestMergeAndAveragePhaseUtilizationPreservesRates(t *testing.T) {
	got := mergePhaseUtilization(
		[]PhaseUtilization{{
			Phase: AgentPhaseDecode, DurationS: 2, ProcessCPUTimeS: 1,
			ProcessReadBytes: 200, ProcessObservations: 2,
			GPUUtilization: []GPUUtilization{{GPU: 0, SMPercent: 90, Observations: 2}},
		}},
		[]PhaseUtilization{{
			Phase: AgentPhaseDecode, DurationS: 4, ProcessCPUTimeS: 1,
			ProcessReadBytes: 400, ProcessObservations: 2,
			GPUUtilization: []GPUUtilization{{GPU: 0, SMPercent: 30, Observations: 1}},
		}},
	)
	averagePhaseUtilization(got, 2)
	if len(got) != 1 || got[0].DurationS != 3 || got[0].ProcessCPUTimeS != 1 ||
		got[0].ProcessReadBytes != 300 {
		t.Fatalf("phase average=%+v", got)
	}
	if math.Abs(got[0].ProcessCPUPercent-100.0/3.0) > 1e-9 {
		t.Fatalf("CPU rate=%f, want %f", got[0].ProcessCPUPercent, 100.0/3.0)
	}
	if len(got[0].GPUUtilization) != 1 || got[0].GPUUtilization[0].SMPercent != 70 {
		t.Fatalf("weighted GPU merge=%+v", got[0].GPUUtilization)
	}
}
