//go:build cgo

package pdfoxide

// cgo binding-parity tests: the standalone document sanitization and
// document-scoped PAdES-B-LTA reader signal that the other bindings
// expose must also exist here.

import (
	"strings"
	"testing"
)

func TestDocumentEditor_SanitizeDocument(t *testing.T) {
	editor, cleanup := openEditorForTest(t, "# Sanitize\n\nconfidential body")
	defer cleanup()

	removed, err := editor.SanitizeDocument()
	if err != nil {
		t.Fatalf("SanitizeDocument: %v", err)
	}
	if removed < 0 {
		t.Errorf("SanitizeDocument: negative removed count %d", removed)
	}
	out, err := editor.SaveToBytes()
	if err != nil {
		t.Fatalf("SaveToBytes after sanitize: %v", err)
	}
	if !strings.HasPrefix(string(out[:5]), "%PDF-") {
		t.Errorf("sanitized output is not a PDF: %q", out[:5])
	}
}

func TestPdfDocument_HasDocumentTimestamp_PlainPDF(t *testing.T) {
	path := makeTempPDF(t, "# LTA probe\n\nplain document")
	doc, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer doc.Close()

	has, err := doc.HasDocumentTimestamp()
	if err != nil {
		t.Fatalf("HasDocumentTimestamp: %v", err)
	}
	if has {
		t.Error("a freshly-built PDF must not report a /DocTimeStamp (B-LTA)")
	}
}

func TestPAdESLevel_FrozenEnumMapping(t *testing.T) {
	// The integer mapping is frozen and shared with the C ABI / every binding.
	if PAdESBB != 0 || PAdESBT != 1 || PAdESBLt != 2 || PAdESBLta != 3 {
		t.Fatalf("PAdES level enum drifted: %d %d %d %d",
			PAdESBB, PAdESBT, PAdESBLt, PAdESBLta)
	}
}

func TestCryptoGovernance_PolicyAndCbom(t *testing.T) {
	// CryptoPolicy()/CryptoInventory()/CryptoCBOM() are process-wide
	// readers (no handle); they must be callable and well-typed.
	if p := CryptoPolicy(); p == "" {
		t.Error("CryptoPolicy() returned empty string")
	}
	_ = CryptoInventory() // []string; may be empty early in the process
	cbom := CryptoCBOM()
	if !strings.Contains(cbom, "CycloneDX") {
		t.Errorf("CryptoCBOM() is not a CycloneDX document: %.60q", cbom)
	}
}
