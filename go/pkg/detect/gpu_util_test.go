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

func TestMapUtilizationToIndexesFailsClosedWithoutPCI(t *testing.T) {
	if got := MapUtilizationToIndexes([]GPU{{Index: 0}}, []GPUUtilization{{SMPercent: 90}}); got != nil {
		t.Fatalf("unkeyed samples became indexes: %+v", got)
	}
}
