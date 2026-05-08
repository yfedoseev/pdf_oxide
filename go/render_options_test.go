//go:build pdf_oxide_dev
// +build pdf_oxide_dev

package pdfoxide

import (
	"bytes"
	"sync"
	"testing"
)

// Regression target: Go's RenderPage only took a `format int` argument,
// so callers could not pick DPI, background, annotations, or JPEG
// quality. These tests exercise the full Rust `RenderOptions` surface
// exposed via RenderPageWithOptions.

func pngMagic(b []byte) bool {
	return len(b) >= 8 && b[0] == 0x89 && b[1] == 0x50 && b[2] == 0x4E && b[3] == 0x47
}

func jpegMagic(b []byte) bool {
	return len(b) >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF
}

func makeDocForRender(t *testing.T) *PdfDocument {
	t.Helper()
	path := createTestPDF(t, "# Render me\n\nBody.")
	doc, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { doc.Close() })
	return doc
}

// skipIfRenderUnavailable skips the test when the native lib was compiled
// without the rendering feature. Mirrors the certificate_test.go pattern.
func skipIfRenderUnavailable(t *testing.T, doc *PdfDocument) {
	t.Helper()
	_, err := doc.RenderPageWithOptions(0, RenderOptions{})
	if err != nil && isUnsupportedError(err) {
		t.Skipf("RenderPageWithOptions unavailable in this build: %v", err)
	}
}

func TestRenderPageWithOptions_DefaultPng(t *testing.T) {
	doc := makeDocForRender(t)
	skipIfRenderUnavailable(t, doc)
	img, err := doc.RenderPageWithOptions(0, RenderOptions{})
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	defer img.Close()
	if !pngMagic(img.Data()) {
		t.Fatal("expected PNG magic on default options")
	}
}

func TestRenderPageWithOptions_JpegFormat(t *testing.T) {
	doc := makeDocForRender(t)
	skipIfRenderUnavailable(t, doc)
	img, err := doc.RenderPageWithOptions(0, RenderOptions{
		Format: RenderFormatJpeg,
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	defer img.Close()
	if !jpegMagic(img.Data()) {
		t.Fatal("expected JPEG magic with Format=Jpeg")
	}
}

func TestRenderPageWithOptions_HigherDpi_Bigger(t *testing.T) {
	doc := makeDocForRender(t)
	skipIfRenderUnavailable(t, doc)
	small, err := doc.RenderPageWithOptions(0, RenderOptions{Dpi: 72})
	if err != nil {
		t.Fatalf("small: %v", err)
	}
	defer small.Close()
	large, err := doc.RenderPageWithOptions(0, RenderOptions{Dpi: 300})
	if err != nil {
		t.Fatalf("large: %v", err)
	}
	defer large.Close()
	if len(large.Data()) <= len(small.Data()) {
		t.Fatalf("expected 300 dpi bytes > 72 dpi bytes, got %d vs %d",
			len(large.Data()), len(small.Data()))
	}
}

func TestRenderPageWithOptions_InvalidDpi_Error(t *testing.T) {
	doc := makeDocForRender(t)
	_, err := doc.RenderPageWithOptions(0, RenderOptions{Dpi: -1})
	if err == nil {
		t.Fatal("expected error for Dpi=-1")
	}
	// Contract: any non-nil error is acceptable — the FFI layer may
	// classify invalid DPI under different sentinels across releases.
}

func TestRenderPageWithOptions_TransparentBackground_StillPng(t *testing.T) {
	doc := makeDocForRender(t)
	skipIfRenderUnavailable(t, doc)
	img, err := doc.RenderPageWithOptions(0, RenderOptions{
		TransparentBackground: true,
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	defer img.Close()
	if !pngMagic(img.Data()) {
		t.Fatal("expected PNG magic on transparent render")
	}
	// Ensure PNG bytes didn't accidentally get written with extra header
	if !bytes.HasPrefix(img.Data(), []byte{0x89, 0x50, 0x4E, 0x47}) {
		t.Fatal("unexpected prefix")
	}
}

func TestRenderPageWithOptions_InvalidJpegQuality_Error(t *testing.T) {
	// Zero JpegQuality is the Go-idiomatic "use default 85" sentinel;
	// out-of-range values (>100 or <1 non-zero) must surface as errors.
	doc := makeDocForRender(t)
	_, err := doc.RenderPageWithOptions(0, RenderOptions{
		Format:      RenderFormatJpeg,
		JpegQuality: 200,
	})
	if err == nil {
		t.Fatal("expected error for JpegQuality=200")
	}
}

// ── Raw RGBA tests (issue #446) ──────────────────────────────────────────────

func TestRenderPageRaw_SizeMatchesWidthTimesHeight(t *testing.T) {
	doc := makeDocForRender(t)
	skipIfRenderUnavailable(t, doc)
	px, err := doc.RenderPageRaw(0, 72)
	if err != nil {
		if isUnsupportedError(err) {
			t.Skip("RenderPageRaw unavailable in this build")
		}
		t.Fatalf("RenderPageRaw: %v", err)
	}
	if px.Width <= 0 || px.Height <= 0 {
		t.Fatalf("unexpected dimensions: %dx%d", px.Width, px.Height)
	}
	expected := px.Width * px.Height * 4
	if len(px.Data) != expected {
		t.Fatalf("data length %d != Width*Height*4 = %d", len(px.Data), expected)
	}
}

func TestRenderPageRaw_NotPngEncoded(t *testing.T) {
	doc := makeDocForRender(t)
	skipIfRenderUnavailable(t, doc)
	px, err := doc.RenderPageRaw(0, 72)
	if err != nil {
		if isUnsupportedError(err) {
			t.Skip("RenderPageRaw unavailable")
		}
		t.Fatalf("RenderPageRaw: %v", err)
	}
	if bytes.HasPrefix(px.Data, []byte{0x89, 0x50, 0x4E, 0x47}) {
		t.Fatal("RenderPageRaw returned PNG-encoded data, expected raw RGBA pixels")
	}
}

func TestRenderPageRaw_DimensionsMatchPng(t *testing.T) {
	doc := makeDocForRender(t)
	skipIfRenderUnavailable(t, doc)
	pngImg, err := doc.RenderPageWithOptions(0, RenderOptions{Dpi: 72})
	if err != nil {
		t.Fatalf("RenderPageWithOptions: %v", err)
	}
	defer pngImg.Close()
	px, err := doc.RenderPageRaw(0, 72)
	if err != nil {
		if isUnsupportedError(err) {
			t.Skip("RenderPageRaw unavailable")
		}
		t.Fatalf("RenderPageRaw: %v", err)
	}
	if px.Width != pngImg.Width || px.Height != pngImg.Height {
		t.Fatalf("dimension mismatch: raw %dx%d vs png %dx%d",
			px.Width, px.Height, pngImg.Width, pngImg.Height)
	}
}

func TestRenderPageRaw_InvalidDpi_Error(t *testing.T) {
	doc := makeDocForRender(t)
	_, err := doc.RenderPageRaw(0, 0)
	if err == nil {
		t.Fatal("expected error for dpi=0")
	}
}

// ── Concurrency tests (issue #481) ──────────────────────────────────────────

func TestConcurrentRenders_RenderPage(t *testing.T) {
	doc := makeDocForRender(t)
	skipIfRenderUnavailable(t, doc)
	pages, err := doc.PageCount()
	if err != nil {
		t.Fatalf("PageCount: %v", err)
	}
	const goroutines = 8
	var wg sync.WaitGroup
	errs := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			img, err := doc.RenderPage(p%pages, 0)
			if err != nil {
				errs <- err
				return
			}
			img.Close()
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent RenderPage error: %v", err)
	}
}

func TestConcurrentRenders_RenderPageFit(t *testing.T) {
	doc := makeDocForRender(t)
	skipIfRenderUnavailable(t, doc)
	pages, err := doc.PageCount()
	if err != nil {
		t.Fatalf("PageCount: %v", err)
	}
	const goroutines = 8
	var wg sync.WaitGroup
	errs := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			img, err := doc.RenderPageFit(p%pages, 800, 1200, 0)
			if err != nil {
				errs <- err
				return
			}
			img.Close()
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent RenderPageFit error: %v", err)
	}
}
