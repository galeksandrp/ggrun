package benchmark

import (
	"sort"
	"sync"
	"time"
)

// AgentPhase identifies one materially different part of an agent workload.
// Keeping these phases separate prevents a long prefill from hiding an idle
// decode path (or a short decode from hiding a saturated prefill path).
type AgentPhase string

const (
	AgentPhasePrefill AgentPhase = "prefill"
	AgentPhaseAppend  AgentPhase = "append"
	AgentPhaseDecode  AgentPhase = "decode"
	AgentPhaseMixed   AgentPhase = "mixed"
)

// ProcessResourceSnapshot contains cumulative process counters plus current
// resident memory. Available distinguishes a measured zero from no permission
// or a backend which disappeared between samples.
type ProcessResourceSnapshot struct {
	Available      bool    `json:"available"`
	CPUTimeSeconds float64 `json:"cpu_time_seconds,omitempty"`
	RSSMB          int     `json:"rss_mb,omitempty"`
	ReadBytes      uint64  `json:"read_bytes,omitempty"`
	WriteBytes     uint64  `json:"write_bytes,omitempty"`
}

// QueueResourceSnapshot is optional because llama.cpp's /metrics request uses
// the serving task queue on some backends. Callers should populate it only from
// a non-perturbing source (for example an already-running router poll).
type QueueResourceSnapshot struct {
	Available bool `json:"available"`
	Active    int  `json:"active,omitempty"`
	Deferred  int  `json:"deferred,omitempty"`
}

// ResourceSnapshot is one point-in-time observation. GPU counters are
// instantaneous; process CPU and I/O counters are cumulative so the observer
// can derive rates over exactly the phase boundary.
type ResourceSnapshot struct {
	GPUUtilization []GPUUtilization        `json:"gpu_utilization,omitempty"`
	Process        ProcessResourceSnapshot `json:"process"`
	Queue          QueueResourceSnapshot   `json:"queue"`
}

// PhaseUtilization is the bounded evidence retained for one agent phase.
// ProcessCPUPercent uses 100% per fully occupied logical CPU, so a multithreaded
// backend may legitimately exceed 100%.
type PhaseUtilization struct {
	Phase               AgentPhase       `json:"phase"`
	DurationS           float64          `json:"duration_s,omitempty"`
	GPUUtilization      []GPUUtilization `json:"gpu_utilization,omitempty"`
	ProcessCPUTimeS     float64          `json:"process_cpu_time_s,omitempty"`
	ProcessCPUPercent   float64          `json:"process_cpu_percent,omitempty"`
	ProcessRSSPeakMB    int              `json:"process_rss_peak_mb,omitempty"`
	ProcessReadBytes    uint64           `json:"process_read_bytes,omitempty"`
	ProcessWriteBytes   uint64           `json:"process_write_bytes,omitempty"`
	ProcessObservations int              `json:"process_observations,omitempty"`
	QueueActiveAvg      float64          `json:"queue_active_avg,omitempty"`
	QueueActiveMax      int              `json:"queue_active_max,omitempty"`
	QueueDeferredAvg    float64          `json:"queue_deferred_avg,omitempty"`
	QueueDeferredMax    int              `json:"queue_deferred_max,omitempty"`
	QueueObservations   int              `json:"queue_observations,omitempty"`
	Observations        int              `json:"observations,omitempty"`
}

// startPhaseResourceObservation samples one phase until the returned function
// is called. The stopper is idempotent. A final sample is taken on stop so the
// cumulative CPU and I/O deltas cover short phases which finish before the
// first ticker interval.
func startPhaseResourceObservation(r *Runner, phase AgentPhase) func() PhaseUtilization {
	if r == nil || (r.SampleResources == nil && r.SampleGPUUtilization == nil) {
		return func() PhaseUtilization { return PhaseUtilization{Phase: phase} }
	}
	interval := r.ResourceSamplingInterval
	if interval <= 0 {
		interval = r.GPUUtilizationInterval
	}
	if interval <= 0 {
		interval = defaultGPUUtilInterval
	}

	type gpuAggregate struct {
		sm, memory, observations int
	}
	started := time.Now()
	stop := make(chan struct{})
	done := make(chan PhaseUtilization, 1)
	go func() {
		byGPU := map[int]gpuAggregate{}
		var processFirst, processLast ProcessResourceSnapshot
		var processFirstAt, processLastAt time.Time
		processObservations := 0
		processRSSPeakMB := 0
		queueActiveSum, queueDeferredSum := 0, 0
		queueActiveMax, queueDeferredMax := 0, 0
		queueObservations := 0
		observations := 0

		observe := func() {
			now := time.Now()
			sample := sampleRunnerResources(r)
			observations++

			maxSM := 0
			for _, gpu := range sample.GPUUtilization {
				if gpu.SMPercent > maxSM {
					maxSM = gpu.SMPercent
				}
			}
			// All-idle GPU frames are not topology evidence. Process and queue
			// counters are still retained for CPU-only and stalled paths.
			if maxSM >= activeGPUUtilThreshold {
				for _, gpu := range sample.GPUUtilization {
					agg := byGPU[gpu.GPU]
					agg.sm += gpu.SMPercent
					agg.memory += gpu.MemPercent
					agg.observations++
					byGPU[gpu.GPU] = agg
				}
			}

			if sample.Process.Available {
				if processObservations == 0 {
					processFirst = sample.Process
					processFirstAt = now
				}
				processLast = sample.Process
				processLastAt = now
				processObservations++
				if sample.Process.RSSMB > processRSSPeakMB {
					processRSSPeakMB = sample.Process.RSSMB
				}
			}

			if sample.Queue.Available {
				queueObservations++
				queueActiveSum += sample.Queue.Active
				queueDeferredSum += sample.Queue.Deferred
				if sample.Queue.Active > queueActiveMax {
					queueActiveMax = sample.Queue.Active
				}
				if sample.Queue.Deferred > queueDeferredMax {
					queueDeferredMax = sample.Queue.Deferred
				}
			}
		}

		observe()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				observe()
			case <-stop:
				observe()
				out := PhaseUtilization{
					Phase: phase, DurationS: time.Since(started).Seconds(),
					ProcessRSSPeakMB:    processRSSPeakMB,
					ProcessObservations: processObservations,
					QueueActiveMax:      queueActiveMax, QueueDeferredMax: queueDeferredMax,
					QueueObservations: queueObservations, Observations: observations,
				}
				for gpu, agg := range byGPU {
					if agg.observations <= 0 {
						continue
					}
					out.GPUUtilization = append(out.GPUUtilization, GPUUtilization{
						GPU: gpu, SMPercent: agg.sm / agg.observations,
						MemPercent:   agg.memory / agg.observations,
						Observations: agg.observations,
					})
				}
				sort.Slice(out.GPUUtilization, func(i, j int) bool {
					return out.GPUUtilization[i].GPU < out.GPUUtilization[j].GPU
				})
				if processObservations >= 2 {
					if processLast.CPUTimeSeconds >= processFirst.CPUTimeSeconds {
						out.ProcessCPUTimeS = processLast.CPUTimeSeconds - processFirst.CPUTimeSeconds
					}
					processWall := processLastAt.Sub(processFirstAt).Seconds()
					if processWall > 0 {
						out.ProcessCPUPercent = out.ProcessCPUTimeS / processWall * 100
					}
					if processLast.ReadBytes >= processFirst.ReadBytes {
						out.ProcessReadBytes = processLast.ReadBytes - processFirst.ReadBytes
					}
					if processLast.WriteBytes >= processFirst.WriteBytes {
						out.ProcessWriteBytes = processLast.WriteBytes - processFirst.WriteBytes
					}
				}
				if queueObservations > 0 {
					out.QueueActiveAvg = float64(queueActiveSum) / float64(queueObservations)
					out.QueueDeferredAvg = float64(queueDeferredSum) / float64(queueObservations)
				}
				done <- out
				return
			}
		}
	}()

	var once sync.Once
	var result PhaseUtilization
	return func() PhaseUtilization {
		once.Do(func() {
			close(stop)
			result = <-done
		})
		return result
	}
}

func sampleRunnerResources(r *Runner) ResourceSnapshot {
	var sample ResourceSnapshot
	if r != nil && r.SampleResources != nil {
		sample = r.SampleResources()
	}
	if r != nil && len(sample.GPUUtilization) == 0 && r.SampleGPUUtilization != nil {
		sample.GPUUtilization = r.SampleGPUUtilization()
	}
	return sample
}

// startGPUUtilizationObservation preserves the original package-level helper
// for callers and tests while using the phase-capable observer internally.
func startGPUUtilizationObservation(r *Runner) func() []GPUUtilization {
	stop := startPhaseResourceObservation(r, "")
	return func() []GPUUtilization { return stop().GPUUtilization }
}

func mergePhaseUtilization(dst, src []PhaseUtilization) []PhaseUtilization {
	byPhase := make(map[AgentPhase]*PhaseUtilization, len(dst)+len(src))
	order := make([]AgentPhase, 0, len(dst)+len(src))
	add := func(sample PhaseUtilization) {
		out, ok := byPhase[sample.Phase]
		if !ok {
			copySample := PhaseUtilization{Phase: sample.Phase}
			out = &copySample
			byPhase[sample.Phase] = out
			order = append(order, sample.Phase)
		}
		oldQueueObservations := out.QueueObservations
		out.DurationS += sample.DurationS
		out.ProcessCPUTimeS += sample.ProcessCPUTimeS
		out.ProcessReadBytes += sample.ProcessReadBytes
		out.ProcessWriteBytes += sample.ProcessWriteBytes
		out.ProcessObservations += sample.ProcessObservations
		if sample.ProcessRSSPeakMB > out.ProcessRSSPeakMB {
			out.ProcessRSSPeakMB = sample.ProcessRSSPeakMB
		}
		if sample.QueueActiveMax > out.QueueActiveMax {
			out.QueueActiveMax = sample.QueueActiveMax
		}
		if sample.QueueDeferredMax > out.QueueDeferredMax {
			out.QueueDeferredMax = sample.QueueDeferredMax
		}
		out.QueueObservations += sample.QueueObservations
		if out.QueueObservations > 0 {
			out.QueueActiveAvg = (out.QueueActiveAvg*float64(oldQueueObservations) +
				sample.QueueActiveAvg*float64(sample.QueueObservations)) / float64(out.QueueObservations)
			out.QueueDeferredAvg = (out.QueueDeferredAvg*float64(oldQueueObservations) +
				sample.QueueDeferredAvg*float64(sample.QueueObservations)) / float64(out.QueueObservations)
		}
		out.Observations += sample.Observations
		out.GPUUtilization = mergeGPUUtilization(out.GPUUtilization, sample.GPUUtilization)
		if out.DurationS > 0 {
			out.ProcessCPUPercent = out.ProcessCPUTimeS / out.DurationS * 100
		}
	}
	for _, sample := range dst {
		add(sample)
	}
	for _, sample := range src {
		add(sample)
	}
	out := make([]PhaseUtilization, 0, len(order))
	for _, phase := range order {
		out = append(out, *byPhase[phase])
	}
	return out
}

func averagePhaseUtilization(samples []PhaseUtilization, trials int) {
	if trials <= 1 {
		return
	}
	for i := range samples {
		samples[i].DurationS /= float64(trials)
		samples[i].ProcessCPUTimeS /= float64(trials)
		samples[i].ProcessReadBytes /= uint64(trials)
		samples[i].ProcessWriteBytes /= uint64(trials)
		if samples[i].DurationS > 0 {
			samples[i].ProcessCPUPercent = samples[i].ProcessCPUTimeS / samples[i].DurationS * 100
		}
	}
}
