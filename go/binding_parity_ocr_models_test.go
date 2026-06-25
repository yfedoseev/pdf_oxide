//go:build cgo

package pdfoxide

// cgo binding-parity test: the OCR model provisioning trio the other
// bindings expose must also exist here. Network-free — only the
// air-gapped manifest is asserted (no downloads; those belong to the
// model-gated Rust lane).

import (
	"strings"
	"testing"
)

func TestModelManifest(t *testing.T) {
	manifest := ModelManifest()
	if !strings.Contains(manifest, "det.onnx") {
		t.Errorf("ModelManifest() must list the shared detector det.onnx; got %q", manifest)
	}
	if !strings.Contains(manifest, "english") {
		t.Errorf("ModelManifest() must list the english recognition model; got %q", manifest)
	}

	// PrefetchAvailable() is a pure feature probe (no I/O); just
	// exercise the call path and signature.
	_ = PrefetchAvailable() // bool — matches the purego signature
}
