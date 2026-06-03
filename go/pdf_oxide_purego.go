//go:build !cgo

// Package pdfoxide — purego backend (CGO_ENABLED=0).
//
// This file provides a cgo-free implementation of the Go bindings that
// dynamically loads `libpdf_oxide.{so,dylib,dll}` at init via
// github.com/ebitengine/purego. It unlocks `CGO_ENABLED=0` cross-compiles
// and pure-Go toolchain builds at the cost of a small per-call dispatch
// overhead compared to the cgo backend (pdf_oxide.go).
//
// Scope (v0.3.38): read-side PdfDocument API + a minimal PdfCreator
// for tests. Editor, DocumentBuilder, barcodes, signatures, TSA,
// rendering, forms, OCR, and other surfaces remain cgo-only for now
// — constructing them under this build tag will yield a compile error
// since the owning types/functions are tagged `//go:build cgo`.
//
// Runtime lookup:
//
//	PDF_OXIDE_LIB_PATH — if set, purego.Dlopen is called with that path.
//	Otherwise we search (in order):
//	  $XDG_CACHE_HOME/pdf_oxide/v<ver>/lib/<GOOS_GOARCH>/libpdf_oxide.<ext>
//	  <os.UserCacheDir>/pdf_oxide/v<ver>/lib/<GOOS_GOARCH>/libpdf_oxide.<ext>
//	  (Linux only) system loader — `libpdf_oxide.so` via dlopen RTLD lookup
//
// If none resolve, every public entry point returns an error wrapping
// ErrPuregoLibraryNotFound.

package pdfoxide

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ErrPuregoLibraryNotFound is returned when the shared lib can't be located
// at build time. Setting PDF_OXIDE_LIB_PATH or running `go run
// github.com/yfedoseev/pdf_oxide/go/cmd/install@latest -shared` fixes it.
var ErrPuregoLibraryNotFound = errors.New("pdf_oxide: could not locate libpdf_oxide shared library (set PDF_OXIDE_LIB_PATH or run the installer with -shared)")

// ─── Library loading ─────────────────────────────────────────────────────────

var (
	libOnce sync.Once
	libErr  error
	libHdl  uintptr
)

// loadLib ensures libpdf_oxide is dlopened exactly once. Subsequent calls
// are no-ops. Errors are sticky so we don't retry a failed load.
func loadLib() error {
	libOnce.Do(func() {
		path, err := locateLib()
		if err != nil {
			libErr = err
			return
		}
		handle, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			libErr = fmt.Errorf("pdf_oxide: dlopen %s: %w", path, err)
			return
		}
		libHdl = handle
		registerFFI(handle)
	})
	return libErr
}

// libExtension returns the per-platform shared-object extension.
func libExtension() string {
	switch runtime.GOOS {
	case "darwin":
		return ".dylib"
	case "windows":
		return ".dll"
	default:
		return ".so"
	}
}

// libBasename is the platform-appropriate basename the release workflow ships.
func libBasename() string {
	if runtime.GOOS == "windows" {
		return "pdf_oxide.dll"
	}
	return "libpdf_oxide" + libExtension()
}

// locateLib returns an absolute path to libpdf_oxide — PDF_OXIDE_LIB_PATH
// first, then the installer's cache layout, then falls back to the system
// loader (bare basename; Dlopen will use LD_LIBRARY_PATH / rpath).
func locateLib() (string, error) {
	if env := os.Getenv("PDF_OXIDE_LIB_PATH"); env != "" {
		return env, nil
	}
	// Match the installer's layout: <user-cache>/pdf_oxide/v<VER>/lib/<GOOS_GOARCH>/<libname>
	// We don't know the version here, so we scan the directory for any vX.Y.Z.
	if cacheDir, err := os.UserCacheDir(); err == nil {
		root := filepath.Join(cacheDir, "pdf_oxide")
		if entries, err := os.ReadDir(root); err == nil {
			sub := runtime.GOOS + "_" + runtime.GOARCH
			for _, e := range entries {
				if !e.IsDir() {
					continue
				}
				candidate := filepath.Join(root, e.Name(), "lib", sub, libBasename())
				if _, err := os.Stat(candidate); err == nil {
					return candidate, nil
				}
			}
		}
	}
	// Last-resort: system loader lookup. Will fail cleanly if the user has
	// neither set PDF_OXIDE_LIB_PATH nor installed it via -shared.
	return libBasename(), nil
}

// ─── FFI function table ──────────────────────────────────────────────────────
//
// Purego requires Go-native types in function signatures; cgo types like
// C.int are replaced by int32, C-strings by `string` (automatically marshalled
// to NUL-terminated cstring by purego), and opaque handles by `uintptr`.
//
// Strings returned *from* C come back as `uintptr` (we don't want purego to
// allocate+copy without giving us a chance to call free_string). We convert
// via goStringAndFree below.

//nolint:gochecknoglobals // registered once at init, read-only thereafter
var (
	// Memory management — first arg is the raw C pointer we got back from
	// a string/bytes-returning function. We pass it back as *byte so
	// purego forwards the same address without extra copies.
	ffiFreeString func(p *byte)
	ffiFreeBytes  func(p *byte)

	// PdfDocument — read-side
	ffiPdfDocumentOpen             func(path string, errCode *int32) uintptr
	ffiPdfDocumentOpenFromBytes    func(data []byte, length uintptr, errCode *int32) uintptr
	ffiPdfDocumentOpenWithPassword func(path, password string, errCode *int32) uintptr
	ffiPdfDocumentFree             func(handle uintptr)
	ffiPdfDocumentGetPageCount     func(handle uintptr, errCode *int32) int32
	ffiPdfDocumentGetVersion       func(handle uintptr, major, minor *uint8)
	ffiPdfDocumentHasStructureTree func(handle uintptr) bool
	ffiPdfDocumentIsEncrypted      func(handle uintptr) bool
	ffiPdfDocumentAuthenticate     func(handle uintptr, password string, errCode *int32) bool
	ffiPdfDocumentExtractText      func(handle uintptr, pageIndex int32, errCode *int32) *byte
	ffiPdfDocumentExtractAllText   func(handle uintptr, errCode *int32) *byte
	ffiPdfDocumentToMarkdown       func(handle uintptr, pageIndex int32, errCode *int32) *byte
	ffiPdfDocumentToHtml           func(handle uintptr, pageIndex int32, errCode *int32) *byte
	ffiPdfDocumentToPlainText      func(handle uintptr, pageIndex int32, errCode *int32) *byte
	ffiPdfDocumentToMarkdownAll    func(handle uintptr, errCode *int32) *byte
	ffiPdfDocumentToHtmlAll        func(handle uintptr, errCode *int32) *byte
	ffiPdfDocumentToPlainTextAll   func(handle uintptr, errCode *int32) *byte
	// #517 comprehensive auto extraction (frozen JSON envelope).
	ffiPdfDocumentClassifyPage     func(handle uintptr, pageIndex int32, errCode *int32) *byte
	ffiPdfDocumentClassifyDocument func(handle uintptr, errCode *int32) *byte
	ffiPdfDocumentExtractTextAuto  func(handle uintptr, pageIndex int32, errCode *int32) *byte
	ffiPdfDocumentExtractPageAuto  func(handle uintptr, pageIndex int32, optionsJSON string, errCode *int32) *byte
	// #536 structure-tree-ordered structured extraction (JSON StructuredPage).
	ffiPdfDocumentExtractStructuredToJSON func(handle uintptr, pageIndex int32, errCode *int32) *byte

	// PdfCreator — minimal, enough to generate test fixtures via FromMarkdown.
	ffiPdfFromMarkdown func(markdown string, errCode *int32) uintptr
	ffiPdfFromHtml     func(html string, errCode *int32) uintptr
	ffiPdfFromText     func(text string, errCode *int32) uintptr
	ffiPdfSave         func(handle uintptr, path string, errCode *int32) int32
	ffiPdfGetPageCount func(handle uintptr, errCode *int32) int32
	ffiPdfFree         func(handle uintptr)

	// Bulk JSON extractors — Fonts / Annotations / Elements / Search.
	// Each returns an opaque list handle, then `<thing>_to_json` serialises
	// the whole list in one FFI call; we unmarshal on the Go side.
	ffiPdfDocumentGetEmbeddedFonts   func(handle uintptr, pageIndex int32, errCode *int32) uintptr
	ffiPdfOxideFontListFree          func(handle uintptr)
	ffiPdfOxideFontsToJSON           func(fonts uintptr, errCode *int32) *byte
	ffiPdfDocumentGetPageAnnotations func(handle uintptr, pageIndex int32, errCode *int32) uintptr
	ffiPdfOxideAnnotationListFree    func(handle uintptr)
	ffiPdfOxideAnnotationsToJSON     func(ann uintptr, errCode *int32) *byte
	ffiPdfPageGetElements            func(handle uintptr, pageIndex int32, errCode *int32) uintptr
	ffiPdfOxideElementsFree          func(handle uintptr)
	ffiPdfOxideElementsToJSON        func(elements uintptr, errCode *int32) *byte
	ffiPdfDocumentSearchPage         func(handle uintptr, pageIndex int32, term string, caseSensitive bool, errCode *int32) uintptr
	ffiPdfDocumentSearchAll          func(handle uintptr, term string, caseSensitive bool, errCode *int32) uintptr
	ffiPdfOxideSearchResultFree      func(handle uintptr)
	ffiPdfOxideSearchResultsToJSON   func(results uintptr, errCode *int32) *byte

	// Page info — width/height/rotation/boxes.
	ffiPdfPageGetWidth    func(handle uintptr, pageIndex int32, errCode *int32) float32
	ffiPdfPageGetHeight   func(handle uintptr, pageIndex int32, errCode *int32) float32
	ffiPdfPageGetRotation func(handle uintptr, pageIndex int32, errCode *int32) int32

	// Logging
	ffiSetLogLevel func(level int32)
	ffiGetLogLevel func() int32

	// OCR model provisioning (#519) — process-wide, no handle.
	ffiPrefetchModels    func(languagesCSV string, errCode *int32) *byte
	ffiModelManifest     func() *byte
	ffiPrefetchAvailable func() int32

	// Runtime crypto-governance policy (#230) — process-wide, no handle.
	ffiCryptoSetPolicy func(spec string) int32
	ffiCryptoPolicy    func() *byte
	ffiCryptoInventory func() *byte
	ffiCryptoCbom      func() *byte

	// Split-by-bookmarks (#482) — plans against a PdfDocument handle.
	ffiPdfDocumentPlanSplitByBookmarks func(handle uintptr, optionsJSON string, errCode *int32) *byte

	// DocumentEditor + destructive redaction / sanitize (#231).
	ffiDocumentEditorOpen          func(path string, errCode *int32) uintptr
	ffiDocumentEditorOpenFromBytes func(data []byte, length uintptr, errCode *int32) uintptr
	ffiDocumentEditorSaveToBytes   func(handle uintptr, outLen *uintptr, errCode *int32) *byte
	ffiDocumentEditorSave          func(handle uintptr, path string, errCode *int32) int32
	ffiDocumentEditorFree          func(handle uintptr)
	ffiRedactionAdd                func(handle uintptr, page uintptr, x1, y1, x2, y2, r, g, b float64, errCode *int32) int32
	ffiRedactionCount              func(handle uintptr, page uintptr, errCode *int32) int32
	ffiRedactionApply              func(handle uintptr, scrub bool, r, g, b float64, errCode *int32) int32
	ffiRedactionScrubMetadata      func(handle uintptr, errCode *int32) int32

	// PAdES signing + DSS read side (#235).
	ffiCertificateLoadFromBytes func(certBytes []byte, certLen int32, password string, errCode *int32) uintptr
	ffiCertificateLoadFromPem   func(certPem, keyPem string, errCode *int32) uintptr
	ffiCertificateFree          func(handle uintptr)
	ffiSignBytes                func(pdf []byte, pdfLen uintptr, cert uintptr, reason, location string, outLen *uintptr, errCode *int32) *byte
	// 5-arg struct-options variant. The 18-arg pdf_sign_bytes_pades
	// exceeds purego's SysV/AMD64 argument limit (panics at
	// registration), so the purego backend uses the collapsed
	// pdf_sign_bytes_pades_opts (parity with cgo via the same core).
	ffiSignBytesPadesOpts func(
		pdf []byte, pdfLen uintptr, opts *padesSignOptsC,
		outLen *uintptr, errCode *int32) *byte
	ffiDocumentGetSignatureCount func(handle uintptr, errCode *int32) int32
	ffiDocumentGetSignature      func(handle uintptr, index int32, errCode *int32) uintptr
	ffiSignatureGetPadesLevel    func(handle uintptr, errCode *int32) int32
	ffiSignatureFree             func(handle uintptr)
	ffiDocumentGetDss            func(handle uintptr, errCode *int32) uintptr
	ffiDocumentHasTimestamp      func(handle uintptr, errCode *int32) int32
	ffiDssCertCount              func(dss uintptr) int32
	ffiDssCrlCount               func(dss uintptr) int32
	ffiDssOcspCount              func(dss uintptr) int32
	ffiDssVriCount               func(dss uintptr) int32
	ffiDssGetCert                func(dss uintptr, index int32, outLen *uintptr, errCode *int32) *byte
	ffiDssGetCrl                 func(dss uintptr, index int32, outLen *uintptr, errCode *int32) *byte
	ffiDssGetOcsp                func(dss uintptr, index int32, outLen *uintptr, errCode *int32) *byte
	ffiDssFree                   func(dss uintptr)
)

func registerFFI(lib uintptr) {
	r := func(target any, name string) {
		purego.RegisterLibFunc(target, lib, name)
	}
	r(&ffiFreeString, "free_string")
	r(&ffiFreeBytes, "free_bytes")

	r(&ffiPdfDocumentOpen, "pdf_document_open")
	r(&ffiPdfDocumentOpenFromBytes, "pdf_document_open_from_bytes")
	r(&ffiPdfDocumentOpenWithPassword, "pdf_document_open_with_password")
	r(&ffiPdfDocumentFree, "pdf_document_free")
	r(&ffiPdfDocumentGetPageCount, "pdf_document_get_page_count")
	r(&ffiPdfDocumentGetVersion, "pdf_document_get_version")
	r(&ffiPdfDocumentHasStructureTree, "pdf_document_has_structure_tree")
	r(&ffiPdfDocumentIsEncrypted, "pdf_document_is_encrypted")
	r(&ffiPdfDocumentAuthenticate, "pdf_document_authenticate")
	r(&ffiPdfDocumentExtractText, "pdf_document_extract_text")
	r(&ffiPdfDocumentExtractAllText, "pdf_document_extract_all_text")
	r(&ffiPdfDocumentToMarkdown, "pdf_document_to_markdown")
	r(&ffiPdfDocumentToHtml, "pdf_document_to_html")
	r(&ffiPdfDocumentToPlainText, "pdf_document_to_plain_text")
	r(&ffiPdfDocumentToMarkdownAll, "pdf_document_to_markdown_all")
	r(&ffiPdfDocumentClassifyPage, "pdf_document_classify_page")
	r(&ffiPdfDocumentClassifyDocument, "pdf_document_classify_document")
	r(&ffiPdfDocumentExtractTextAuto, "pdf_document_extract_text_auto")
	r(&ffiPdfDocumentExtractPageAuto, "pdf_document_extract_page_auto")
	r(&ffiPdfDocumentExtractStructuredToJSON, "pdf_document_extract_structured_to_json")
	r(&ffiPdfDocumentToHtmlAll, "pdf_document_to_html_all")
	r(&ffiPdfDocumentToPlainTextAll, "pdf_document_to_plain_text_all")

	r(&ffiPdfFromMarkdown, "pdf_from_markdown")
	r(&ffiPdfFromHtml, "pdf_from_html")
	r(&ffiPdfFromText, "pdf_from_text")
	r(&ffiPdfSave, "pdf_save")
	r(&ffiPdfGetPageCount, "pdf_get_page_count")
	r(&ffiPdfFree, "pdf_free")

	r(&ffiPdfDocumentGetEmbeddedFonts, "pdf_document_get_embedded_fonts")
	r(&ffiPdfOxideFontListFree, "pdf_oxide_font_list_free")
	r(&ffiPdfOxideFontsToJSON, "pdf_oxide_fonts_to_json")
	r(&ffiPdfDocumentGetPageAnnotations, "pdf_document_get_page_annotations")
	r(&ffiPdfOxideAnnotationListFree, "pdf_oxide_annotation_list_free")
	r(&ffiPdfOxideAnnotationsToJSON, "pdf_oxide_annotations_to_json")
	r(&ffiPdfPageGetElements, "pdf_page_get_elements")
	r(&ffiPdfOxideElementsFree, "pdf_oxide_elements_free")
	r(&ffiPdfOxideElementsToJSON, "pdf_oxide_elements_to_json")
	r(&ffiPdfDocumentSearchPage, "pdf_document_search_page")
	r(&ffiPdfDocumentSearchAll, "pdf_document_search_all")
	r(&ffiPdfOxideSearchResultFree, "pdf_oxide_search_result_free")
	r(&ffiPdfOxideSearchResultsToJSON, "pdf_oxide_search_results_to_json")

	r(&ffiPdfPageGetWidth, "pdf_page_get_width")
	r(&ffiPdfPageGetHeight, "pdf_page_get_height")
	r(&ffiPdfPageGetRotation, "pdf_page_get_rotation")

	r(&ffiSetLogLevel, "pdf_oxide_set_log_level")
	r(&ffiGetLogLevel, "pdf_oxide_get_log_level")

	r(&ffiPrefetchModels, "pdf_oxide_prefetch_models")
	r(&ffiModelManifest, "pdf_oxide_model_manifest")
	r(&ffiPrefetchAvailable, "pdf_oxide_prefetch_available")

	r(&ffiCryptoSetPolicy, "pdf_oxide_crypto_set_policy")
	r(&ffiCryptoPolicy, "pdf_oxide_crypto_policy")
	r(&ffiCryptoInventory, "pdf_oxide_crypto_inventory")
	r(&ffiCryptoCbom, "pdf_oxide_crypto_cbom")

	r(&ffiPdfDocumentPlanSplitByBookmarks, "pdf_document_plan_split_by_bookmarks")

	r(&ffiDocumentEditorOpen, "document_editor_open")
	r(&ffiDocumentEditorOpenFromBytes, "document_editor_open_from_bytes")
	r(&ffiDocumentEditorSaveToBytes, "document_editor_save_to_bytes")
	r(&ffiDocumentEditorSave, "document_editor_save")
	r(&ffiDocumentEditorFree, "document_editor_free")
	r(&ffiRedactionAdd, "pdf_redaction_add")
	r(&ffiRedactionCount, "pdf_redaction_count")
	r(&ffiRedactionApply, "pdf_redaction_apply")
	r(&ffiRedactionScrubMetadata, "pdf_redaction_scrub_metadata")

	r(&ffiCertificateLoadFromBytes, "pdf_certificate_load_from_bytes")
	r(&ffiCertificateLoadFromPem, "pdf_certificate_load_from_pem")
	r(&ffiCertificateFree, "pdf_certificate_free")
	r(&ffiSignBytes, "pdf_sign_bytes")
	r(&ffiSignBytesPadesOpts, "pdf_sign_bytes_pades_opts")
	r(&ffiDocumentGetSignatureCount, "pdf_document_get_signature_count")
	r(&ffiDocumentGetSignature, "pdf_document_get_signature")
	r(&ffiSignatureGetPadesLevel, "pdf_signature_get_pades_level")
	r(&ffiSignatureFree, "pdf_signature_free")
	r(&ffiDocumentGetDss, "pdf_document_get_dss")
	r(&ffiDocumentHasTimestamp, "pdf_document_has_timestamp")
	r(&ffiDssCertCount, "pdf_dss_cert_count")
	r(&ffiDssCrlCount, "pdf_dss_crl_count")
	r(&ffiDssOcspCount, "pdf_dss_ocsp_count")
	r(&ffiDssVriCount, "pdf_dss_vri_count")
	r(&ffiDssGetCert, "pdf_dss_get_cert")
	r(&ffiDssGetCrl, "pdf_dss_get_crl")
	r(&ffiDssGetOcsp, "pdf_dss_get_ocsp")
	r(&ffiDssFree, "pdf_dss_free")
}

// goStringAndFree copies a NUL-terminated C string at p into a Go string and
// then calls free_string on p. Safe when p == nil; returns "".
func goStringAndFree(p *byte) string {
	if p == nil {
		return ""
	}
	// Walk until NUL using unsafe.Add (idiomatic pointer arithmetic).
	base := unsafe.Pointer(p)
	var n int
	for *(*byte)(unsafe.Add(base, n)) != 0 {
		n++
	}
	// unsafe.Slice is the Go 1.17+ way to build a slice from a C buffer.
	// string() copies the bytes into Go-managed memory so it's safe to
	// free the C side immediately after.
	s := string(unsafe.Slice(p, n))
	ffiFreeString(p)
	return s
}

// ─── PdfDocument ─────────────────────────────────────────────────────────────

// PdfDocument represents an open PDF document.
// It is safe for concurrent use by multiple goroutines.
type PdfDocument struct {
	mu     sync.RWMutex
	handle uintptr
	closed bool
}

func (doc *PdfDocument) acquireRead() error {
	doc.mu.Lock()
	if doc.closed {
		doc.mu.Unlock()
		return ErrDocumentClosed
	}
	return nil
}

// Open opens a PDF document from file path.
func Open(path string) (*PdfDocument, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	var ec int32
	h := ffiPdfDocumentOpen(path, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to open document: %w", ErrInternal)
	}
	return &PdfDocument{handle: h}, nil
}

// OpenFromBytes opens a PDF document from an in-memory byte slice.
func OpenFromBytes(data []byte) (*PdfDocument, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, ErrEmptyContent
	}
	var ec int32
	h := ffiPdfDocumentOpenFromBytes(data, uintptr(len(data)), &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to open document from bytes: %w", ErrInternal)
	}
	return &PdfDocument{handle: h}, nil
}

// OpenWithPassword opens an encrypted PDF document with the given password.
func OpenWithPassword(path, password string) (*PdfDocument, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	var ec int32
	h := ffiPdfDocumentOpenWithPassword(path, password, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to open document: %w", ErrInternal)
	}
	return &PdfDocument{handle: h}, nil
}

// Close releases document resources. Safe to call multiple times.
func (doc *PdfDocument) Close() error {
	doc.mu.Lock()
	defer doc.mu.Unlock()
	if !doc.closed && doc.handle != 0 {
		ffiPdfDocumentFree(doc.handle)
		doc.closed = true
		doc.handle = 0
	}
	return nil
}

// IsClosed returns whether the document is closed.
func (doc *PdfDocument) IsClosed() bool {
	doc.mu.Lock()
	defer doc.mu.Unlock()
	return doc.closed
}

// PageCount returns the number of pages in the document.
func (doc *PdfDocument) PageCount() (int, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()
	var ec int32
	n := ffiPdfDocumentGetPageCount(doc.handle, &ec)
	if ec != 0 {
		return 0, ffiErrorFromInt(int(ec))
	}
	return int(n), nil
}

// Version returns the PDF spec version as (major, minor).
func (doc *PdfDocument) Version() (uint8, uint8, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, 0, err
	}
	defer doc.mu.Unlock()
	var major, minor uint8
	ffiPdfDocumentGetVersion(doc.handle, &major, &minor)
	return major, minor, nil
}

// HasStructureTree reports whether the document has a Tagged PDF structure tree.
func (doc *PdfDocument) HasStructureTree() (bool, error) {
	if err := doc.acquireRead(); err != nil {
		return false, err
	}
	defer doc.mu.Unlock()
	return ffiPdfDocumentHasStructureTree(doc.handle), nil
}

// IsEncrypted reports whether the document is encrypted.
func (doc *PdfDocument) IsEncrypted() (bool, error) {
	if err := doc.acquireRead(); err != nil {
		return false, err
	}
	defer doc.mu.Unlock()
	return ffiPdfDocumentIsEncrypted(doc.handle), nil
}

// Authenticate attempts to decrypt the document with the given password.
func (doc *PdfDocument) Authenticate(password string) (bool, error) {
	doc.mu.Lock()
	defer doc.mu.Unlock()
	if doc.closed {
		return false, ErrDocumentClosed
	}
	var ec int32
	ok := ffiPdfDocumentAuthenticate(doc.handle, password, &ec)
	if ec != 0 {
		return false, ffiErrorFromInt(int(ec))
	}
	return ok, nil
}

// ExtractText extracts plain text from the given page (0-based).
func (doc *PdfDocument) ExtractText(pageIndex int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var ec int32
	p := ffiPdfDocumentExtractText(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ExtractAllText extracts plain text from every page, concatenated.
func (doc *PdfDocument) ExtractAllText() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var ec int32
	p := ffiPdfDocumentExtractAllText(doc.handle, &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ClassifyPage — see the cgo backend for docs (#517). Signature-
// identical so the build-tag split is transparent.
func (doc *PdfDocument) ClassifyPage(pageIndex int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	if pageIndex < 0 {
		return "", ErrInvalidPageIndex
	}
	var ec int32
	p := ffiPdfDocumentClassifyPage(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ClassifyDocument — JSON DocumentClassification (#517).
func (doc *PdfDocument) ClassifyDocument() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var ec int32
	p := ffiPdfDocumentClassifyDocument(doc.handle, &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ExtractTextAuto — graceful text-vs-OCR (#513/#517).
func (doc *PdfDocument) ExtractTextAuto(pageIndex int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	if pageIndex < 0 {
		return "", ErrInvalidPageIndex
	}
	var ec int32
	p := ffiPdfDocumentExtractTextAuto(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ExtractPageAuto — JSON PageExtraction; functional options (#517).
func (doc *PdfDocument) ExtractPageAuto(pageIndex int, opts ...AutoOption) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	if pageIndex < 0 {
		return "", ErrInvalidPageIndex
	}
	var ec int32
	p := ffiPdfDocumentExtractPageAuto(doc.handle, int32(pageIndex), autoOptionsJSON(opts), &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ExtractStructured returns a structure-tree-ordered extraction of the page
// as a JSON StructuredPage string ({page_index, page_width, page_height,
// regions:[{kind, text, bbox, spans, column_index}]}). Callers unmarshal the
// JSON themselves (#536).
func (doc *PdfDocument) ExtractStructured(page int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	if page < 0 {
		return "", ErrInvalidPageIndex
	}
	var ec int32
	p := ffiPdfDocumentExtractStructuredToJSON(doc.handle, int32(page), &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ToMarkdown renders the given page as Markdown.
func (doc *PdfDocument) ToMarkdown(pageIndex int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var ec int32
	p := ffiPdfDocumentToMarkdown(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ToHtml renders the given page as HTML.
func (doc *PdfDocument) ToHtml(pageIndex int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var ec int32
	p := ffiPdfDocumentToHtml(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ToPlainText renders the given page as plain text.
func (doc *PdfDocument) ToPlainText(pageIndex int) (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var ec int32
	p := ffiPdfDocumentToPlainText(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ToMarkdownAll renders all pages as Markdown, concatenated.
func (doc *PdfDocument) ToMarkdownAll() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var ec int32
	p := ffiPdfDocumentToMarkdownAll(doc.handle, &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ToHtmlAll renders all pages as HTML, concatenated.
func (doc *PdfDocument) ToHtmlAll() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var ec int32
	p := ffiPdfDocumentToHtmlAll(doc.handle, &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ToPlainTextAll renders all pages as plain text, concatenated.
func (doc *PdfDocument) ToPlainTextAll() (string, error) {
	if err := doc.acquireRead(); err != nil {
		return "", err
	}
	defer doc.mu.Unlock()
	var ec int32
	p := ffiPdfDocumentToPlainTextAll(doc.handle, &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	return goStringAndFree(p), nil
}

// ─── Page — lightweight handle for a single page of a PdfDocument ──────────

// Page represents a single page of a PdfDocument.
type Page struct {
	doc   *PdfDocument
	Index int
}

// Page returns a handle to the page at the given zero-based index.
func (doc *PdfDocument) Page(index int) (*Page, error) {
	count, err := doc.PageCount()
	if err != nil {
		return nil, err
	}
	if index < 0 || index >= count {
		return nil, fmt.Errorf("page index %d out of range [0, %d)", index, count)
	}
	return &Page{doc: doc, Index: index}, nil
}

// Pages returns all pages as a slice.
func (doc *PdfDocument) Pages() ([]*Page, error) {
	count, err := doc.PageCount()
	if err != nil {
		return nil, err
	}
	pages := make([]*Page, count)
	for i := 0; i < count; i++ {
		pages[i] = &Page{doc: doc, Index: i}
	}
	return pages, nil
}

// Text extracts plain text from the page.
func (p *Page) Text() (string, error) { return p.doc.ExtractText(p.Index) }

// Markdown renders the page as Markdown.
func (p *Page) Markdown() (string, error) { return p.doc.ToMarkdown(p.Index) }

// Html renders the page as HTML.
func (p *Page) Html() (string, error) { return p.doc.ToHtml(p.Index) }

// PlainText renders the page as plain text.
func (p *Page) PlainText() (string, error) { return p.doc.ToPlainText(p.Index) }

// ─── PdfCreator — minimal subset for test fixtures ──────────────────────────

// PdfCreator represents a generated PDF. Only enough of the API is implemented
// under the purego backend to let test helpers build fixtures in-memory; the
// richer editor/builder APIs remain cgo-only.
type PdfCreator struct {
	mu     sync.Mutex
	handle uintptr
	closed bool
}

// FromMarkdown builds a PDF from a Markdown string.
func FromMarkdown(markdown string) (*PdfCreator, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	if markdown == "" {
		return nil, ErrEmptyContent
	}
	var ec int32
	h := ffiPdfFromMarkdown(markdown, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to create pdf from markdown: %w", ErrInternal)
	}
	return &PdfCreator{handle: h}, nil
}

// FromHtml builds a PDF from an HTML string.
func FromHtml(html string) (*PdfCreator, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	if html == "" {
		return nil, ErrEmptyContent
	}
	var ec int32
	h := ffiPdfFromHtml(html, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to create pdf from html: %w", ErrInternal)
	}
	return &PdfCreator{handle: h}, nil
}

// FromText builds a PDF from a plain text string.
func FromText(text string) (*PdfCreator, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	if text == "" {
		return nil, ErrEmptyContent
	}
	var ec int32
	h := ffiPdfFromText(text, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to create pdf from text: %w", ErrInternal)
	}
	return &PdfCreator{handle: h}, nil
}

// Save writes the generated PDF to disk.
func (c *PdfCreator) Save(path string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return ErrCreatorClosed
	}
	var ec int32
	rc := ffiPdfSave(c.handle, path, &ec)
	if ec != 0 {
		return ffiErrorFromInt(int(ec))
	}
	if rc != 0 {
		return fmt.Errorf("pdf_oxide: save returned %d: %w", rc, ErrInternal)
	}
	return nil
}

// PageCount returns how many pages the generated PDF contains.
func (c *PdfCreator) PageCount() (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0, ErrCreatorClosed
	}
	var ec int32
	n := ffiPdfGetPageCount(c.handle, &ec)
	if ec != 0 {
		return 0, ffiErrorFromInt(int(ec))
	}
	return int(n), nil
}

// Close releases creator resources. Safe to call multiple times.
func (c *PdfCreator) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.closed && c.handle != 0 {
		ffiPdfFree(c.handle)
		c.closed = true
		c.handle = 0
	}
	return nil
}

// ─── Bulk JSON extractors — Fonts / Annotations / PageElements / Search ────

// unmarshalListJSON is the common tail shared by every *-to-JSON extractor:
// decode the returned C string into dst, free_string the C pointer, and
// forward any error_code from the extractor.
func unmarshalListJSON(p *byte, ec int32, dst any) error {
	if ec != 0 {
		return ffiErrorFromInt(int(ec))
	}
	if p == nil {
		return nil
	}
	s := goStringAndFree(p)
	if s == "" {
		return nil
	}
	return json.Unmarshal([]byte(s), dst)
}

// Fonts returns all fonts embedded in or used by the given page. One FFI
// call per page — the list is JSON-encoded on the Rust side.
func (doc *PdfDocument) Fonts(pageIndex int) ([]Font, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var ec int32
	h := ffiPdfDocumentGetEmbeddedFonts(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to get fonts: %w", ErrInternal)
	}
	defer ffiPdfOxideFontListFree(h)

	var jec int32
	p := ffiPdfOxideFontsToJSON(h, &jec)
	var out []Font
	if err := unmarshalListJSON(p, jec, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Annotations returns all annotations on the given page.
func (doc *PdfDocument) Annotations(pageIndex int) ([]Annotation, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var ec int32
	h := ffiPdfDocumentGetPageAnnotations(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to get annotations: %w", ErrInternal)
	}
	defer ffiPdfOxideAnnotationListFree(h)

	var jec int32
	p := ffiPdfOxideAnnotationsToJSON(h, &jec)
	var out []Annotation
	if err := unmarshalListJSON(p, jec, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// PageElements returns all layout elements (text spans with bbox) on the
// given page.
func (doc *PdfDocument) PageElements(pageIndex int) ([]Element, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var ec int32
	h := ffiPdfPageGetElements(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to get elements: %w", ErrInternal)
	}
	defer ffiPdfOxideElementsFree(h)

	var jec int32
	p := ffiPdfOxideElementsToJSON(h, &jec)
	var out []Element
	if err := unmarshalListJSON(p, jec, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// SearchPage searches the given page for `term` and returns all matches.
func (doc *PdfDocument) SearchPage(pageIndex int, term string, caseSensitive bool) ([]SearchResult, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var ec int32
	h := ffiPdfDocumentSearchPage(doc.handle, int32(pageIndex), term, caseSensitive, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, ErrInternal
	}
	defer ffiPdfOxideSearchResultFree(h)

	var jec int32
	p := ffiPdfOxideSearchResultsToJSON(h, &jec)
	var out []SearchResult
	if err := unmarshalListJSON(p, jec, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// SearchAll searches the whole document for `term`.
func (doc *PdfDocument) SearchAll(term string, caseSensitive bool) ([]SearchResult, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var ec int32
	h := ffiPdfDocumentSearchAll(doc.handle, term, caseSensitive, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, ErrInternal
	}
	defer ffiPdfOxideSearchResultFree(h)

	var jec int32
	p := ffiPdfOxideSearchResultsToJSON(h, &jec)
	var out []SearchResult
	if err := unmarshalListJSON(p, jec, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// ─── Page dimensions ─────────────────────────────────────────────────────────

// PageWidth returns the given page's width in PDF points.
func (doc *PdfDocument) PageWidth(pageIndex int) (float32, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()
	var ec int32
	w := ffiPdfPageGetWidth(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return 0, ffiErrorFromInt(int(ec))
	}
	return w, nil
}

// PageHeight returns the given page's height in PDF points.
func (doc *PdfDocument) PageHeight(pageIndex int) (float32, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()
	var ec int32
	h := ffiPdfPageGetHeight(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return 0, ffiErrorFromInt(int(ec))
	}
	return h, nil
}

// PageRotation returns the rotation (degrees) of the given page.
func (doc *PdfDocument) PageRotation(pageIndex int) (int, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()
	var ec int32
	r := ffiPdfPageGetRotation(doc.handle, int32(pageIndex), &ec)
	if ec != 0 {
		return 0, ffiErrorFromInt(int(ec))
	}
	return int(r), nil
}

// ─── Logging ─────────────────────────────────────────────────────────────────

// SetLogLevel sets the global log level.
func SetLogLevel(level LogLevel) {
	if err := loadLib(); err != nil {
		return
	}
	ffiSetLogLevel(int32(level))
}

// GetLogLevel returns the current log level.
func GetLogLevel() LogLevel {
	if err := loadLib(); err != nil {
		return LogOff
	}
	return LogLevel(ffiGetLogLevel())
}

// ─── OCR model provisioning (#519) ───────────────────────────────────────────

// PrefetchModels downloads the shared OCR detector plus the
// recognition model and dictionary for each requested language code
// (e.g. "english", "chinese", "arabic") into the model cache dir
// ($PDF_OXIDE_MODEL_DIR / the platform cache) and returns that dir.
// No languages → English. Unknown codes are skipped. Idempotent.
// Actual download requires the native lib built with the ocr feature;
// without it the cache dir is still created (no fetch) — query
// PrefetchAvailable. The signature matches the cgo backend exactly.
func PrefetchModels(langs ...string) (string, error) {
	if err := loadLib(); err != nil {
		return "", err
	}
	var ec int32
	p := ffiPrefetchModels(strings.Join(langs, ","), &ec)
	if ec != 0 {
		return "", ffiErrorFromInt(int(ec))
	}
	if p == nil {
		return "", ErrInternal
	}
	return goStringAndFree(p), nil
}

// ModelManifest returns the air-gapped OCR model manifest as JSON
// (detector + every supported language's cache filenames and source
// URLs). Never errors. Signature matches the cgo backend.
func ModelManifest() string {
	if err := loadLib(); err != nil {
		return ""
	}
	return goStringAndFree(ffiModelManifest())
}

// PrefetchAvailable reports whether this build can actually download
// models (compiled with the ocr feature). When false, PrefetchModels
// only creates the cache dir (no fetch). Signature matches the cgo
// backend.
func PrefetchAvailable() bool {
	if err := loadLib(); err != nil {
		return false
	}
	return ffiPrefetchAvailable() != 0
}

// goBytesAndFree copies n bytes from the C buffer at p into a fresh Go
// slice and then calls free_bytes(p). Safe when p == nil (returns nil).
func goBytesAndFree(p *byte, n int) []byte {
	if p == nil || n <= 0 {
		if p != nil {
			ffiFreeBytes(p)
		}
		return nil
	}
	out := make([]byte, n)
	copy(out, unsafe.Slice(p, n))
	ffiFreeBytes(p)
	return out
}

// ─── Runtime crypto-governance policy (#230) ─────────────────────────────────

// SetCryptoPolicy installs the process-wide runtime crypto policy from
// its grammar string ("compat"|"strict"|"fips-strict"[;…]). Fail-closed:
// returns ErrCryptoPolicyParse on an unparseable spec or
// ErrCryptoPolicyAlreadySet if a policy is already set. The signature
// matches the cgo backend exactly.
func SetCryptoPolicy(spec string) error {
	if err := loadLib(); err != nil {
		return err
	}
	switch ffiCryptoSetPolicy(spec) {
	case 0:
		return nil
	case 1:
		return ErrCryptoPolicyInvalidArg
	case 2:
		return ErrCryptoPolicyParse
	case 3:
		return ErrCryptoPolicyAlreadySet
	default:
		return fmt.Errorf("pdf_oxide_crypto_set_policy returned unknown error code")
	}
}

// CryptoPolicy returns the active crypto policy as its canonical
// grammar string (default "compat" when never set or the library
// cannot be loaded). Signature matches the cgo backend.
func CryptoPolicy() string {
	if err := loadLib(); err != nil {
		return "compat"
	}
	s := goStringAndFree(ffiCryptoPolicy())
	if s == "" {
		return "compat"
	}
	return s
}

// CryptoInventory returns the cryptographic algorithm tokens exercised
// so far this process (governance report). Signature matches the cgo
// backend (CSV from the C ABI, split on ',').
func CryptoInventory() []string {
	if err := loadLib(); err != nil {
		return nil
	}
	joined := goStringAndFree(ffiCryptoInventory())
	if joined == "" {
		return nil
	}
	return splitCSV(joined)
}

// CryptoCBOM returns a CycloneDX 1.6 Cryptographic Bill of Materials
// (JSON) of the algorithms exercised so far this process (#230).
// Signature matches the cgo backend.
func CryptoCBOM() string {
	if err := loadLib(); err != nil {
		return ""
	}
	return goStringAndFree(ffiCryptoCbom())
}

func splitCSV(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == ',' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	out = append(out, s[start:])
	return out
}

// ─── Split a PDF by bookmarks (#482) ─────────────────────────────────────────

// SplitSegment is one planned output segment of a bookmark split (#482).
type SplitSegment struct {
	Index     int     `json:"index"`
	StartPage int     `json:"start_page"`
	EndPage   int     `json:"end_page"`
	Title     *string `json:"title"`
	FileStem  string  `json:"file_stem"`
	PageLabel *string `json:"page_label"`
}

// SplitByBookmarksOptions controls a bookmark split. Level: 0 = all
// depths, 1 = top-level only (default), n = up to depth n.
type SplitByBookmarksOptions struct {
	TitlePrefix        *string
	IgnoreCase         bool
	Level              int
	IncludeFrontMatter bool
}

// PlanSplitByBookmarks plans (does not produce) a split of the document
// at outline/bookmark boundaries (#482), mirroring the core
// plan_split_by_bookmarks.
func (doc *PdfDocument) PlanSplitByBookmarks(opts SplitByBookmarksOptions) ([]SplitSegment, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	optJSON, err := json.Marshal(map[string]interface{}{
		"title_prefix":         opts.TitlePrefix,
		"ignore_case":          opts.IgnoreCase,
		"level":                opts.Level,
		"include_front_matter": opts.IncludeFrontMatter,
	})
	if err != nil {
		return nil, fmt.Errorf("pdf_oxide: marshal split options: %w", err)
	}
	var ec int32
	p := ffiPdfDocumentPlanSplitByBookmarks(doc.handle, string(optJSON), &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	text := goStringAndFree(p)
	var segs []SplitSegment
	if err := json.Unmarshal([]byte(text), &segs); err != nil {
		return nil, fmt.Errorf("pdf_oxide: parse split plan JSON: %w", err)
	}
	return segs, nil
}

// ─── DocumentEditor + destructive redaction / sanitize (#231) ────────────────

// DocumentEditor is a mutable PDF document for the purego backend. It is
// safe for concurrent use by multiple goroutines.
type DocumentEditor struct {
	mu     sync.RWMutex
	handle uintptr
	closed bool
}

func (editor *DocumentEditor) acquireWrite() error {
	editor.mu.Lock()
	if editor.closed {
		editor.mu.Unlock()
		return ErrEditorClosed
	}
	return nil
}

// OpenEditor opens a PDF document from a file path for editing.
func OpenEditor(path string) (*DocumentEditor, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	var ec int32
	h := ffiDocumentEditorOpen(path, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to open document editor: %w", ErrInternal)
	}
	return &DocumentEditor{handle: h}, nil
}

// OpenEditorFromBytes opens a PDF document from memory for editing.
func OpenEditorFromBytes(data []byte) (*DocumentEditor, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("pdf_oxide: data must not be empty: %w", ErrInvalidPath)
	}
	var ec int32
	h := ffiDocumentEditorOpenFromBytes(data, uintptr(len(data)), &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, fmt.Errorf("pdf_oxide: failed to open editor from bytes: %w", ErrInternal)
	}
	return &DocumentEditor{handle: h}, nil
}

// Close releases editor resources. Safe to call multiple times.
func (editor *DocumentEditor) Close() {
	editor.mu.Lock()
	defer editor.mu.Unlock()
	if !editor.closed && editor.handle != 0 {
		ffiDocumentEditorFree(editor.handle)
		editor.closed = true
		editor.handle = 0
	}
}

// Save writes the edited document to a file path.
func (editor *DocumentEditor) Save(path string) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	var ec int32
	ffiDocumentEditorSave(editor.handle, path, &ec)
	if ec != 0 {
		return ffiErrorFromInt(int(ec))
	}
	return nil
}

// SaveToBytes serialises the edited document into an in-memory slice.
func (editor *DocumentEditor) SaveToBytes() ([]byte, error) {
	if err := editor.acquireWrite(); err != nil {
		return nil, err
	}
	defer editor.mu.Unlock()
	var outLen uintptr
	var ec int32
	p := ffiDocumentEditorSaveToBytes(editor.handle, &outLen, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if p == nil {
		return nil, fmt.Errorf("pdf_oxide: failed to save editor to bytes: %w", ErrInternal)
	}
	return goBytesAndFree(p, int(outLen)), nil
}

// AddRedaction queues an explicit destructive redaction rectangle (page
// user space). The content is physically removed by ApplyRedactions —
// not a cosmetic overlay (ISO 32000-1:2008 §12.5.6.23). fill is an
// optional DeviceRGB [r,g,b]; nil uses black.
func (editor *DocumentEditor) AddRedaction(page int, rect [4]float64, fill *[3]float64) error {
	if err := editor.acquireWrite(); err != nil {
		return err
	}
	defer editor.mu.Unlock()
	r, g, b := 0.0, 0.0, 0.0
	if fill != nil {
		r, g, b = fill[0], fill[1], fill[2]
	}
	var ec int32
	ffiRedactionAdd(editor.handle, uintptr(page),
		rect[0], rect[1], rect[2], rect[3], r, g, b, &ec)
	if ec != 0 {
		return ffiErrorFromInt(int(ec))
	}
	return nil
}

// RedactionCount returns the number of redaction regions queued for a
// page (annotations + programmatic rectangles).
func (editor *DocumentEditor) RedactionCount(page int) (int, error) {
	if err := editor.acquireWrite(); err != nil {
		return 0, err
	}
	defer editor.mu.Unlock()
	var ec int32
	n := ffiRedactionCount(editor.handle, uintptr(page), &ec)
	if ec != 0 {
		return 0, ffiErrorFromInt(int(ec))
	}
	return int(n), nil
}

// ApplyRedactions destructively applies all queued redactions (true
// content removal). Returns the number of glyphs physically removed.
func (editor *DocumentEditor) ApplyRedactions(scrubMetadata bool) (int, error) {
	if err := editor.acquireWrite(); err != nil {
		return 0, err
	}
	defer editor.mu.Unlock()
	var ec int32
	removed := ffiRedactionApply(editor.handle, scrubMetadata, 0, 0, 0, &ec)
	if ec != 0 {
		return 0, ffiErrorFromInt(int(ec))
	}
	return int(removed), nil
}

// SanitizeDocument performs standalone document sanitization (no
// geometric redaction): it strips the /Info dictionary, the catalog
// XMP /Metadata stream, document JavaScript (/OpenAction, /AA,
// /Names/JavaScript) and /Names/EmbeddedFiles, hard-excluding the
// removed object subtrees from the rewritten file. Returns the number
// of annotations removed (issue #231).
func (editor *DocumentEditor) SanitizeDocument() (int, error) {
	if err := editor.acquireWrite(); err != nil {
		return 0, err
	}
	defer editor.mu.Unlock()
	var ec int32
	removed := ffiRedactionScrubMetadata(editor.handle, &ec)
	if ec != 0 {
		return 0, ffiErrorFromInt(int(ec))
	}
	return int(removed), nil
}

// ─── PAdES LTV signing + DSS read side (#235) ────────────────────────────────

// PAdESLevel is the PAdES baseline level. The integer mapping
// (PAdESBB=0, PAdESBT=1, PAdESBLt=2, PAdESBLta=3) is frozen and shared
// with the C ABI and every binding — never renumber.
type PAdESLevel int32

const (
	// PAdESBB is CAdES-B-B (signed attrs incl. ESS signing-certificate-v2).
	PAdESBB PAdESLevel = 0
	// PAdESBT is B-B + an RFC 3161 signature-time-stamp unsigned attr.
	PAdESBT PAdESLevel = 1
	// PAdESBLt is B-T + a Document Security Store (DSS/VRI).
	PAdESBLt PAdESLevel = 2
	// PAdESBLta is B-LT + a document-scoped /DocTimeStamp archival timestamp.
	PAdESBLta PAdESLevel = 3
)

// RevocationMaterial is the offline B-LT validation set: DER X.509
// certificates, CRLs, and OCSP responses.
type RevocationMaterial struct {
	Certs [][]byte
	CRLs  [][]byte
	OCSPs [][]byte
}

// PAdESOptions configures SignPdfBytesPAdES. TSAURL is required for
// Level >= PAdESBT (the RFC 3161 source). Reason/Location are optional.
// Revocation supplies the B-LT DSS material.
type PAdESOptions struct {
	Level      PAdESLevel
	TSAURL     string
	Reason     string
	Location   string
	Revocation *RevocationMaterial
}

// Certificate holds a loaded signing certificate (purego backend).
type Certificate struct {
	handle uintptr
}

// LoadCertificate loads a PKCS#12 certificate from bytes.
func LoadCertificate(data []byte, password string) (*Certificate, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("pdf_oxide: certificate data is empty: %w", ErrEmptyContent)
	}
	var ec int32
	h := ffiCertificateLoadFromBytes(data, int32(len(data)), password, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	return &Certificate{handle: h}, nil
}

// LoadCertificateFromPem loads signing credentials from PEM-encoded
// certificate and private-key strings.
func LoadCertificateFromPem(certPem, keyPem string) (*Certificate, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	var ec int32
	h := ffiCertificateLoadFromPem(certPem, keyPem, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	return &Certificate{handle: h}, nil
}

// Close releases certificate resources.
func (cert *Certificate) Close() {
	if cert.handle != 0 {
		ffiCertificateFree(cert.handle)
		cert.handle = 0
	}
}

// SignPdfBytes applies a CMS/PKCS#7 detached signature to pdfData and
// returns the signed PDF. The Certificate must carry a private key.
func SignPdfBytes(pdfData []byte, cert *Certificate, reason, location string) ([]byte, error) {
	if cert == nil || cert.handle == 0 {
		return nil, ErrInternal
	}
	if len(pdfData) == 0 {
		return nil, ErrEmptyContent
	}
	var outLen uintptr
	var ec int32
	p := ffiSignBytes(pdfData, uintptr(len(pdfData)), cert.handle, reason, location, &outLen, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if p == nil {
		return nil, ErrInternal
	}
	return goBytesAndFree(p, int(outLen)), nil
}

// blobArrays builds parallel (ptrs, lens) uintptr slices over blobs for
// the *_pades FFI. The caller MUST keep `blobs`, the returned slices,
// and the originating RevocationMaterial alive across the FFI call
// (runtime.KeepAlive) — the slices hold raw data addresses.
func blobArrays(blobs [][]byte) (ptrs, lens []uintptr) {
	if len(blobs) == 0 {
		return nil, nil
	}
	ptrs = make([]uintptr, len(blobs))
	lens = make([]uintptr, len(blobs))
	for i, b := range blobs {
		if len(b) == 0 {
			continue
		}
		ptrs[i] = uintptr(unsafe.Pointer(&b[0]))
		lens[i] = uintptr(len(b))
	}
	return ptrs, lens
}

func firstPtr(s []uintptr) *uintptr {
	if len(s) == 0 {
		return nil
	}
	return &s[0]
}

// padesSignOptsC mirrors the Rust `#[repr(C)] PadesSignOptionsC`
// EXACTLY (field order + pointer-sized types); passed by pointer to
// pdf_sign_bytes_pades_opts so the purego call stays at 5 arguments.
type padesSignOptsC struct {
	certHandle uintptr
	certs      uintptr
	certLens   uintptr
	nCerts     uintptr
	crls       uintptr
	crlLens    uintptr
	nCRLs      uintptr
	ocsps      uintptr
	ocspLens   uintptr
	nOCSPs     uintptr
	tsaURL     uintptr
	reason     uintptr
	location   uintptr
	level      int32
}

// cBytes returns a NUL-terminated copy of s and its first-byte address.
// The caller MUST runtime.KeepAlive the returned slice across the call.
func cBytes(s string) ([]byte, uintptr) {
	b := append([]byte(s), 0)
	return b, uintptr(unsafe.Pointer(&b[0]))
}

// sliceAddr is the address of s[0], or 0 for an empty slice.
func sliceAddr(s []uintptr) uintptr {
	if len(s) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&s[0]))
}

// SignPdfBytesPAdES signs pdfData at a PAdES baseline level and returns
// the signed PDF. The Certificate must carry a private key. For
// PAdESBT/PAdESBLt a TSAURL is required.
func SignPdfBytesPAdES(pdfData []byte, cert *Certificate, opts PAdESOptions) ([]byte, error) {
	if cert == nil || cert.handle == 0 {
		return nil, ErrInternal
	}
	if len(pdfData) == 0 {
		return nil, ErrEmptyContent
	}
	var certsP, certsL, crlsP, crlsL, ocspsP, ocspsL []uintptr
	var nCerts, nCRLs, nOCSPs uintptr
	if r := opts.Revocation; r != nil {
		certsP, certsL = blobArrays(r.Certs)
		crlsP, crlsL = blobArrays(r.CRLs)
		ocspsP, ocspsL = blobArrays(r.OCSPs)
		nCerts = uintptr(len(r.Certs))
		nCRLs = uintptr(len(r.CRLs))
		nOCSPs = uintptr(len(r.OCSPs))
	}
	tsaB, tsaP := cBytes(opts.TSAURL)
	rsnB, rsnP := cBytes(opts.Reason)
	locB, locP := cBytes(opts.Location)
	o := padesSignOptsC{
		certHandle: cert.handle,
		certs:      sliceAddr(certsP),
		certLens:   sliceAddr(certsL),
		nCerts:     nCerts,
		crls:       sliceAddr(crlsP),
		crlLens:    sliceAddr(crlsL),
		nCRLs:      nCRLs,
		ocsps:      sliceAddr(ocspsP),
		ocspLens:   sliceAddr(ocspsL),
		nOCSPs:     nOCSPs,
		tsaURL:     tsaP,
		reason:     rsnP,
		location:   locP,
		level:      int32(opts.Level),
	}
	var outLen uintptr
	var ec int32
	p := ffiSignBytesPadesOpts(pdfData, uintptr(len(pdfData)), &o, &outLen, &ec)
	if opts.Revocation != nil {
		runtime.KeepAlive(opts.Revocation.Certs)
		runtime.KeepAlive(opts.Revocation.CRLs)
		runtime.KeepAlive(opts.Revocation.OCSPs)
		runtime.KeepAlive(certsP)
		runtime.KeepAlive(certsL)
		runtime.KeepAlive(crlsP)
		runtime.KeepAlive(crlsL)
		runtime.KeepAlive(ocspsP)
		runtime.KeepAlive(ocspsL)
	}
	runtime.KeepAlive(pdfData)
	runtime.KeepAlive(tsaB)
	runtime.KeepAlive(rsnB)
	runtime.KeepAlive(locB)
	runtime.KeepAlive(&o)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if p == nil {
		return nil, ErrInternal
	}
	return goBytesAndFree(p, int(outLen)), nil
}

// Signature is a live handle to an existing PDF digital signature
// returned by PdfDocument.Signatures. Close() must be called.
type Signature struct {
	handle uintptr
}

// Close releases the underlying native signature handle.
func (s *Signature) Close() {
	if s != nil && s.handle != 0 {
		ffiSignatureFree(s.handle)
		s.handle = 0
	}
}

// PAdESLevel classifies this signature from its CMS attributes alone
// (PAdESBB vs PAdESBT). PAdESBLt additionally needs the document /DSS —
// read it via (*PdfDocument).DSS and re-classify there.
func (s *Signature) PAdESLevel() (PAdESLevel, error) {
	if s == nil || s.handle == 0 {
		return PAdESBB, ErrInternal
	}
	var ec int32
	lvl := ffiSignatureGetPadesLevel(s.handle, &ec)
	if ec != 0 {
		return PAdESBB, ffiErrorFromInt(int(ec))
	}
	return PAdESLevel(lvl), nil
}

// SignatureCount returns the number of existing digital signatures in
// the document (0 when none — not an error).
func (doc *PdfDocument) SignatureCount() (int, error) {
	if err := doc.acquireRead(); err != nil {
		return 0, err
	}
	defer doc.mu.Unlock()
	var ec int32
	n := ffiDocumentGetSignatureCount(doc.handle, &ec)
	if ec != 0 {
		return 0, ffiErrorFromInt(int(ec))
	}
	return int(n), nil
}

// Signatures returns a snapshot of every signature on the document.
// Each Signature must be Close()d by the caller.
func (doc *PdfDocument) Signatures() ([]*Signature, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var ec int32
	n := ffiDocumentGetSignatureCount(doc.handle, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	out := make([]*Signature, 0, n)
	for i := int32(0); i < n; i++ {
		var e int32
		h := ffiDocumentGetSignature(doc.handle, i, &e)
		if e != 0 {
			for _, s := range out {
				s.Close()
			}
			return nil, ffiErrorFromInt(int(e))
		}
		if h == 0 {
			for _, s := range out {
				s.Close()
			}
			return nil, fmt.Errorf("pdf_oxide: pdf_document_get_signature(%d) returned null", i)
		}
		out = append(out, &Signature{handle: h})
	}
	return out, nil
}

// DSS is a parsed Document Security Store (/DSS, ISO 32000-2 §12.8.4.3).
type DSS struct {
	Certs    [][]byte
	CRLs     [][]byte
	OCSPs    [][]byte
	VRICount int
}

// DSS reads the document's Document Security Store, or nil if the PDF
// has no /DSS (not an error). Mirrors Rust signatures::read_dss.
func (doc *PdfDocument) DSS() (*DSS, error) {
	if err := doc.acquireRead(); err != nil {
		return nil, err
	}
	defer doc.mu.Unlock()
	var ec int32
	h := ffiDocumentGetDss(doc.handle, &ec)
	if ec != 0 {
		return nil, ffiErrorFromInt(int(ec))
	}
	if h == 0 {
		return nil, nil // no DSS present
	}
	defer ffiDssFree(h)

	read := func(count func(uintptr) int32, get func(uintptr, int32, *uintptr, *int32) *byte) ([][]byte, error) {
		nn := int(count(h))
		if nn <= 0 {
			return nil, nil
		}
		blobs := make([][]byte, 0, nn)
		for i := 0; i < nn; i++ {
			var l uintptr
			var e int32
			p := get(h, int32(i), &l, &e)
			if e != 0 {
				return nil, ffiErrorFromInt(int(e))
			}
			if p == nil {
				continue
			}
			blobs = append(blobs, goBytesAndFree(p, int(l)))
		}
		return blobs, nil
	}

	certs, err := read(ffiDssCertCount, ffiDssGetCert)
	if err != nil {
		return nil, err
	}
	crls, err := read(ffiDssCrlCount, ffiDssGetCrl)
	if err != nil {
		return nil, err
	}
	ocsps, err := read(ffiDssOcspCount, ffiDssGetOcsp)
	if err != nil {
		return nil, err
	}
	return &DSS{
		Certs:    certs,
		CRLs:     crls,
		OCSPs:    ocsps,
		VRICount: int(ffiDssVriCount(h)),
	}, nil
}

// HasDocumentTimestamp reports whether the document carries a
// document-scoped RFC 3161 /DocTimeStamp archival timestamp
// (PAdES-B-LTA, ISO 32000-2:2020 §12.8.5). This is the document-level
// reader signal; (*Signature).PAdESLevel is signature-scoped and tops
// out at B-LT by design.
func (doc *PdfDocument) HasDocumentTimestamp() (bool, error) {
	if err := doc.acquireRead(); err != nil {
		return false, err
	}
	defer doc.mu.Unlock()
	var ec int32
	r := ffiDocumentHasTimestamp(doc.handle, &ec)
	if ec != 0 {
		return false, ffiErrorFromInt(int(ec))
	}
	return r == 1, nil
}
