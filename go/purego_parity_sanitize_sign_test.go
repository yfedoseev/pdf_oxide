//go:build !cgo

package pdfoxide

// purego-backend parity tests: the cgo-free build must now expose
// crypto-governance, sanitize, the PAdES read side, and
// split-by-bookmarks with the SAME signatures as the cgo backend.
// Runtime tests need the shared lib (skipped without it); the
// frozen-enum check is pure-Go and always runs.

import (
	"os"
	"strings"
	"testing"
)

func TestPurego_PAdESLevel_FrozenEnumMapping(t *testing.T) {
	if PAdESBB != 0 || PAdESBT != 1 || PAdESBLt != 2 || PAdESBLta != 3 {
		t.Fatalf("PAdES level enum drifted: %d %d %d %d",
			PAdESBB, PAdESBT, PAdESBLt, PAdESBLta)
	}
}

func TestPurego_CryptoGovernance(t *testing.T) {
	requireLib(t)
	if p := CryptoPolicy(); p == "" {
		t.Error("CryptoPolicy() returned empty string")
	}
	_ = CryptoInventory() // []string — matches the cgo signature
	if cbom := CryptoCBOM(); !strings.Contains(cbom, "CycloneDX") {
		t.Errorf("CryptoCBOM() is not a CycloneDX document: %.60q", cbom)
	}
}

func TestPurego_SanitizeAndBLtaReader(t *testing.T) {
	requireLib(t)
	path := makePDF(t)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	editor, err := OpenEditorFromBytes(data)
	if err != nil {
		t.Fatalf("OpenEditorFromBytes: %v", err)
	}
	defer editor.Close()
	if _, err := editor.SanitizeDocument(); err != nil {
		t.Fatalf("SanitizeDocument: %v", err)
	}

	doc, err := OpenFromBytes(data)
	if err != nil {
		t.Fatalf("OpenFromBytes: %v", err)
	}
	defer doc.Close()
	has, err := doc.HasDocumentTimestamp()
	if err != nil {
		t.Fatalf("HasDocumentTimestamp: %v", err)
	}
	if has {
		t.Error("a freshly-built PDF must not report a /DocTimeStamp")
	}
	if _, err := doc.PlanSplitByBookmarks(SplitByBookmarksOptions{}); err == nil {
		// A no-outline document returns an error or empty plan; either
		// is acceptable — we only assert the call path is wired.
		t.Log("PlanSplitByBookmarks returned no error (document had an outline)")
	}
}
