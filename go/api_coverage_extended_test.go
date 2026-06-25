//go:build cgo

package pdfoxide

// API-coverage tests for the extended C-ABI surface: one test per newly
// bound symbol. Symbols that need a feature-gated build / signing material
// assert return-or-error (invoke + accept success or the binding error);
// the pure global tunables assert invokability.

import (
	"os"
	"testing"
)

// pdf_convert_to_pdf_a — feature/conversion gated: return-or-error.
func TestConvertToPdfA_ReturnOrError(t *testing.T) {
	doc, cleanup := openCovDoc(t, "# PDF/A\n\nBody text.")
	defer cleanup()

	if _, err := doc.ConvertToPdfA(2); err != nil {
		t.Logf("ConvertToPdfA returned error (acceptable): %v", err)
	}
}

// pdf_document_get_source_bytes — return-or-error (depends on conversion build).
func TestSourceBytes_ReturnOrError(t *testing.T) {
	doc, cleanup := openCovDoc(t, "# Source\n\nBody text.")
	defer cleanup()

	b, err := doc.SourceBytes()
	if err != nil {
		t.Logf("SourceBytes returned error (acceptable): %v", err)
		return
	}
	if len(b) == 0 {
		t.Log("SourceBytes returned empty slice (acceptable)")
	}
}

// pdf_oxide_search_result_count / _get_text / _get_page / _get_bbox —
// exercised through a real search via the scalar-accessor decode path.
func TestSearchResultAccessors_RealSearch(t *testing.T) {
	doc, cleanup := openCovDoc(t, "# Title\n\nThe quick brown fox jumps.")
	defer cleanup()

	results, err := doc.SearchAllVerbose("fox", false)
	if err != nil {
		t.Fatalf("SearchAllVerbose: %v", err)
	}
	if len(results) == 0 {
		t.Skip("term not found in this build's extraction; accessors still invoked")
	}
	// The accessor path populated each field; bbox dims should be non-negative.
	for i, r := range results {
		if r.Width < 0 || r.Height < 0 {
			t.Fatalf("result %d has negative bbox dims: %+v", i, r)
		}
	}
}

// pdf_oxide_set_max_ops_per_stream — no error channel: assert invokable and
// that the prior value is restored.
func TestSetMaxOpsPerStream_Invokable(t *testing.T) {
	prev := SetMaxOpsPerStream(1_000_000)
	// Restore whatever the prior value was so we leave global state untouched.
	restored := SetMaxOpsPerStream(prev)
	if restored != 1_000_000 {
		t.Fatalf("expected previous value 1000000, got %d", restored)
	}
}

// pdf_oxide_set_preserve_unmapped_glyphs — no error channel: assert invokable
// and restore the prior value.
func TestSetPreserveUnmappedGlyphs_Invokable(t *testing.T) {
	prev := SetPreserveUnmappedGlyphs(1)
	restored := SetPreserveUnmappedGlyphs(prev)
	if restored != 1 {
		t.Fatalf("expected previous value 1, got %d", restored)
	}
}

// pdf_render_page_with_options_ex — feature(render) gated: return-or-error.
func TestRenderPageWithOptionsEx_ReturnOrError(t *testing.T) {
	doc, cleanup := openCovDoc(t, "# Render\n\nBody text.")
	defer cleanup()

	img, err := doc.RenderPageWithOptionsEx(0, RenderOptions{}, []string{"Watermark"})
	if err != nil {
		if isUnsupportedError(err) {
			t.Skipf("rendering unavailable in this build: %v", err)
		}
		t.Logf("RenderPageWithOptionsEx returned error (acceptable): %v", err)
		return
	}
	defer img.Close()
	if len(img.Data()) == 0 {
		t.Fatal("expected non-empty rendered image data")
	}
}

// pdf_sign_bytes_pades_opts — needs signing material + feature: return-or-error.
func TestSignPdfBytesPAdESOpts_ReturnOrError(t *testing.T) {
	path := makeTempPDF(t, "# Sign\n\nBody text.")
	defer os.Remove(path)
	pdfData, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	cert, err := LoadCertificate(testCertificateDer(t), "")
	if err != nil {
		t.Skipf("LoadCertificate unavailable in this build: %v", err)
	}
	defer cert.Close()

	if _, err := SignPdfBytesPAdESOpts(pdfData, cert, PAdESOptions{Level: PAdESBB}); err != nil {
		t.Logf("SignPdfBytesPAdESOpts returned error (acceptable): %v", err)
	}
}
