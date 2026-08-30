package detect

import "testing"

func TestParseNVIDIAUtilizationMapsPCISamples(t *testing.T) {
	samples := parseNVIDIAUtilization("0000:01:00.0, 96, 70\n0000:02:00.0, 4, 80\n")
	if len(samples) != 2 || samples[0].PCIBusID != "0000:01:00.0" || samples[0].SMPercent != 96 {
		t.Fatalf("parsed %+v", samples)
	}
	gpus := []GPU{
		{Index: 0, PCIBusID: "00000000:01:00.0"},
		{Index: 2, PCIBusID: "0000:02:00.0"},
	}
	mapped := MapUtilizationToIndexes(gpus, samples)
	if len(mapped) != 2 || mapped[0].Index != 0 || mapped[1].Index != 2 || mapped[1].SMPercent != 4 {
		t.Fatalf("mapped %+v", mapped)
	}
}

func TestParseNVIDIAUtilizationAndTransferCounters(t *testing.T) {
	samples := parseNVIDIAUtilization("0, 0000:01:00.0, 96, 70\n1, 0000:02:00.0, 4, 80\n")
	if len(samples) != 2 || samples[1].NVIDIAIndex != 1 || samples[1].PCIBusID != "0000:02:00.0" {
		t.Fatalf("indexed utilization parsed %+v", samples)
	}
	transfers := parseNVIDIATransfers("# gpu rxpci txpci\n# Idx MB/s MB/s\n 0 366 35\n 1 380 43\n")
	if transfers[0] != [2]int{366, 35} || transfers[1] != [2]int{380, 43} {
		t.Fatalf("transfers parsed %+v", transfers)
	}
}

func TestMapUtilizationToIndexesFailsClosedWithoutPCI(t *testing.T) {
	if got := MapUtilizationToIndexes([]GPU{{Index: 0}}, []GPUUtilization{{SMPercent: 90}}); got != nil {
		t.Fatalf("unkeyed samples became indexes: %+v", got)
	}
}
