// Package pdfoxide provides Go bindings to the pdf_oxide Rust PDF toolkit.
//
// This file holds the build-tag-agnostic type surface: error sentinels,
// the structured *Error value, and error-code mapping. Both the cgo
// backend (pdf_oxide.go, //go:build cgo) and the purego backend
// (pdf_oxide_purego.go, //go:build !cgo) build on top of these.
package pdfoxide

import (
	"encoding/json"
	"errors"
	"fmt"
)

// ─── Comprehensive auto extraction (#517) — functional options ─────────────
//
// Go-idiomatic functional options (the locked decision); they reduce to
// the same `AutoExtractOptions` JSON the frozen C-ABI parses, so cgo and
// purego are signature-identical and consistent with every other
// binding. Build-tag-agnostic (shared by both backends).

type autoOpts struct {
	Mode                   string   `json:"mode,omitempty"`
	ReconstructImageTables *bool    `json:"reconstruct_image_tables,omitempty"`
	EmitPlaceholders       *bool    `json:"emit_placeholders,omitempty"`
	OcrLanguages           []string `json:"ocr_languages,omitempty"`
	MinTextConfidence      *float32 `json:"min_text_confidence,omitempty"`
	TableConfidence        *float32 `json:"table_confidence,omitempty"`
	ForceOcrPages          []uint   `json:"force_ocr_pages,omitempty"`
}

// AutoOption configures an auto-extraction call. Pass zero or more to
// ExtractPageAuto; none = the `balanced` defaults.
type AutoOption func(*autoOpts)

// WithMode sets the text-vs-OCR mode ("text_only" | "auto" | "force_ocr").
func WithMode(mode string) AutoOption { return func(o *autoOpts) { o.Mode = mode } }

// WithImageTables toggles image-table reconstruction.
func WithImageTables(yes bool) AutoOption {
	return func(o *autoOpts) { o.ReconstructImageTables = &yes }
}

// WithPlaceholders toggles figure/table placeholders in the text flow.
func WithPlaceholders(yes bool) AutoOption {
	return func(o *autoOpts) { o.EmitPlaceholders = &yes }
}

// WithOCRLanguages sets OCR language hints.
func WithOCRLanguages(langs ...string) AutoOption {
	return func(o *autoOpts) { o.OcrLanguages = langs }
}

// WithForceOCRPages forces OCR on the given 0-based page indices
// (additive on the mode; does not change it).
func WithForceOCRPages(pages ...uint) AutoOption {
	return func(o *autoOpts) { o.ForceOcrPages = pages }
}

// autoOptionsJSON reduces the functional options to the C-ABI JSON
// (empty string when no options were given → C-ABI uses defaults).
func autoOptionsJSON(opts []AutoOption) string {
	if len(opts) == 0 {
		return ""
	}
	var o autoOpts
	for _, f := range opts {
		f(&o)
	}
	b, err := json.Marshal(&o)
	if err != nil {
		return ""
	}
	return string(b)
}

// Sentinel errors for errors.Is comparisons. Every failure path in this
// package reports one of these wrapped in an *Error for FFI errors, or
// returns the sentinel directly for non-FFI failures.
var (
	// ErrInvalidPath indicates the path argument was invalid. FFI code 1.
	ErrInvalidPath = errors.New("pdf_oxide: invalid path")
	// ErrDocumentNotFound indicates the document could not be opened. FFI code 2.
	ErrDocumentNotFound = errors.New("pdf_oxide: document not found")
	// ErrInvalidFormat indicates the PDF could not be parsed. FFI code 3
	// (ERR_PARSE). Historically documented as "FFI code 3" with the
	// aliased ErrParseError also matching code 5 — see below.
	ErrInvalidFormat = errors.New("pdf_oxide: invalid PDF format")
	// ErrExtractionFailed indicates extraction failed. FFI code 4.
	ErrExtractionFailed = errors.New("pdf_oxide: extraction failed")
	// ErrParseError is a legacy alias for parse-time failures. Kept for
	// backward compatibility with v0.3.38 code that matched against
	// `errors.Is(err, ErrParseError)` when Rust emitted ERR_INTERNAL
	// (code 5) — the v0.3.38 behaviour. As of v0.3.39 the canonical
	// sentinel for code 5 is ErrInternal; code-3 failures match both
	// ErrInvalidFormat (new canonical) and ErrParseError (alias).
	//
	// Deprecated: use ErrInvalidFormat for parse failures or ErrInternal
	// for native-layer failures.
	ErrParseError = ErrInvalidFormat
	// ErrInvalidPageIndex indicates an out-of-range page index. FFI code 6.
	ErrInvalidPageIndex = errors.New("pdf_oxide: invalid page index")
	// ErrSearchFailed indicates a search operation failed. FFI code 7.
	ErrSearchFailed = errors.New("pdf_oxide: search failed")
	// ErrInternal indicates an internal/unknown error. FFI code 5
	// (ERR_INTERNAL). Fixed in v0.3.39 per #398 — previously
	// misreported as "FFI code 8" in the docstring and `sentinelForCode`
	// mapped code 5 to ErrParseError, masking native-layer races
	// behind a misleading parse-error message.
	ErrInternal = errors.New("pdf_oxide: internal error")

	// ErrDocumentClosed indicates the document has been closed.
	ErrDocumentClosed = errors.New("pdf_oxide: document is closed")
	// ErrEditorClosed indicates the editor has been closed.
	ErrEditorClosed = errors.New("pdf_oxide: editor is closed")
	// ErrCreatorClosed indicates the PDF creator has been closed.
	ErrCreatorClosed = errors.New("pdf_oxide: creator is closed")
	// ErrIndexOutOfBounds indicates an out-of-range index.
	ErrIndexOutOfBounds = errors.New("pdf_oxide: index out of bounds")
	// ErrEmptyContent indicates required content was empty.
	ErrEmptyContent = errors.New("pdf_oxide: content must not be empty")

	// ErrNotImplementedInPurego is returned by methods that exist in the
	// cgo backend but have not yet been ported to the purego backend.
	// Build with CGO_ENABLED=1 to use them.
	ErrNotImplementedInPurego = errors.New("pdf_oxide: not implemented in pure-Go (purego) build; rebuild with CGO_ENABLED=1")

	// ErrCryptoPolicyInvalidArg is returned by SetCryptoPolicy for a
	// null/non-UTF-8 spec (#230).
	ErrCryptoPolicyInvalidArg = errors.New("invalid crypto policy spec (not valid UTF-8)")
	// ErrCryptoPolicyParse is returned by SetCryptoPolicy when the spec
	// string is rejected (fail-closed: the policy is NOT installed).
	ErrCryptoPolicyParse = errors.New("crypto policy spec rejected (parse error)")
	// ErrCryptoPolicyAlreadySet is returned by SetCryptoPolicy when a
	// policy was already installed (set-once).
	ErrCryptoPolicyAlreadySet = errors.New("crypto policy already set")
)

// Error is a structured PDF error that carries an FFI error code alongside a
// canonical sentinel. It implements Unwrap so errors.Is works with the exported
// Err* sentinels, and Is so two *Error values with the same Code compare equal.
type Error struct {
	Code     int
	Message  string
	sentinel error
}

// Error returns a human-readable description of the error.
func (e *Error) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("pdf_oxide: error %d", e.Code)
	}
	return fmt.Sprintf("pdf_oxide: %s (code %d)", e.Message, e.Code)
}

// Unwrap returns the canonical sentinel so errors.Is(err, ErrInvalidPath) works.
func (e *Error) Unwrap() error { return e.sentinel }

// Is reports whether target is the same canonical sentinel, or another *Error
// carrying the same Code.
func (e *Error) Is(target error) bool {
	if e.sentinel != nil && target == e.sentinel {
		return true
	}
	var other *Error
	if errors.As(target, &other) {
		return e.Code == other.Code
	}
	return false
}

// ffiErrorFromInt wraps a plain int FFI error code into a fully populated
// *Error. Used by the purego backend, which speaks plain int32 rather
// than C.int. The cgo backend has its own typed wrapper (ffiError) that
// converts C.int before calling this.
func ffiErrorFromInt(code int) error {
	sentinel := sentinelForCode(code)
	return &Error{
		Code:     code,
		Message:  sentinel.Error(),
		sentinel: sentinel,
	}
}

// sentinelForCode returns the canonical sentinel for an FFI error code,
// matching the Rust-side table at `src/ffi.rs:48-56`:
//
//	0 ERR_SUCCESS       — no sentinel
//	1 ERR_INVALID_ARG   — ErrInvalidPath (historical name; is really
//	                     "invalid argument", not path-specific)
//	2 ERR_IO            — ErrDocumentNotFound (historical name; is
//	                     really generic IO)
//	3 ERR_PARSE         — ErrInvalidFormat
//	4 ERR_EXTRACTION    — ErrExtractionFailed
//	5 ERR_INTERNAL      — ErrInternal (fixed in v0.3.39 per #398;
//	                     previously mapped to ErrParseError, which
//	                     masked native-layer races as "parse error")
//	6 ERR_INVALID_PAGE  — ErrInvalidPageIndex
//	7 ERR_SEARCH        — ErrSearchFailed
//	8 _ERR_UNSUPPORTED  — ErrInternal (no dedicated sentinel yet)
//
// The historical names at codes 1, 2, and the former code-5/8 swap are
// legacy API commitments kept for backward compatibility. Renaming to
// accurate labels is tracked as a v0.3.40 cleanup.
func sentinelForCode(code int) error {
	switch code {
	case 1:
		return ErrInvalidPath
	case 2:
		return ErrDocumentNotFound
	case 3:
		return ErrInvalidFormat
	case 4:
		return ErrExtractionFailed
	case 5:
		return ErrInternal
	case 6:
		return ErrInvalidPageIndex
	case 7:
		return ErrSearchFailed
	case 8:
		return ErrInternal
	default:
		return ErrInternal
	}
}

// ─── Extraction result types ────────────────────────────────────────────────
//
// These types are marshaled from JSON payloads returned by the Rust FFI's
// bulk extractors (`pdf_oxide_*_to_json`). The JSON tags match the Rust
// schema so one FFI call per list is enough for the Go layer.

// SearchResult represents a single search hit.
type SearchResult struct {
	Text   string  `json:"text"`
	Page   int     `json:"page"`
	X      float32 `json:"x"`
	Y      float32 `json:"y"`
	Width  float32 `json:"width"`
	Height float32 `json:"height"`
}

// Font represents a font embedded in or used by a PDF page.
type Font struct {
	Name       string  `json:"name"`
	Type       string  `json:"type"`
	Encoding   string  `json:"encoding"`
	IsEmbedded bool    `json:"isEmbedded"`
	IsSubset   bool    `json:"isSubset"`
	Size       float32 `json:"size"`
}

// Annotation represents a single annotation on a PDF page with all its
// metadata already materialized.
type Annotation struct {
	Type             string  `json:"type"`
	Subtype          string  `json:"subtype"`
	Content          string  `json:"content"`
	X                float32 `json:"x"`
	Y                float32 `json:"y"`
	Width            float32 `json:"width"`
	Height           float32 `json:"height"`
	Author           string  `json:"author"`
	BorderWidth      float32 `json:"borderWidth"`
	Color            uint32  `json:"color"`
	CreationDate     int64   `json:"creationDate"`
	ModificationDate int64   `json:"modificationDate"`
	LinkURI          string  `json:"linkURI"`
	TextIconName     string  `json:"textIconName"`
	IsHidden         bool    `json:"isHidden"`
	IsPrintable      bool    `json:"isPrintable"`
	IsReadOnly       bool    `json:"isReadOnly"`
	IsMarkedDeleted  bool    `json:"isMarkedDeleted"`
}

// Element represents a layout element on a PDF page (text block, image, etc.).
type Element struct {
	Type   string  `json:"type"`
	Text   string  `json:"text"`
	X      float32 `json:"x"`
	Y      float32 `json:"y"`
	Width  float32 `json:"width"`
	Height float32 `json:"height"`
	// Provenance is the ISO 32000-1 §9.10.2 mapping tier the span's font
	// offered ("to_unicode"/"encoding"/"predefined_cmap"/"embedded_cmap"/
	// "actual_text"/"fallback"), or "" when unknown. "fallback" means the
	// text is a fabricated glyph-index echo, not read from the file.
	Provenance string `json:"provenance,omitempty"`
}

// ─── DocumentBuilder write-side value types ──────────────────────────────
//
// These types are pure data + tag-agnostic so both backends see them even
// though only the cgo backend currently wires them to the FFI. The purego
// backend lacks DocumentBuilder entirely (the types here are therefore
// just field carriers until purego parity lands).

// Alignment encodes the horizontal alignment of text inside a rectangle
// or column. Values mirror the FFI encoding (0/1/2).
type Alignment int

const (
	// AlignLeft left-aligns text within the rect or column (default).
	AlignLeft Alignment = 0
	// AlignCenter horizontally centers text within the rect or column.
	AlignCenter Alignment = 1
	// AlignRight right-aligns text within the rect or column.
	AlignRight Alignment = 2
)

// Column describes one column of a Table or StreamingTable.
//
//   - Header is the header-row label. For Table with HasHeader=false, the
//     field is ignored. For StreamingTable, the header row is emitted at
//     the top of each new page when RepeatHeader is true.
//   - Width is the column width in PDF points.
//   - Align is the per-column body alignment. Headers always center by
//     default in the underlying writer.
type Column struct {
	Header string
	Width  float32
	Align  Alignment
}

// TableSpec is the buffered-table surface consumed by PageBuilder.Table.
// The whole row matrix is held in managed memory until the page commits.
//
//   - Columns drive column widths + alignments. If HasHeader is true the
//     column headers are promoted into a bold header row.
//   - Rows is a row-major matrix of cell strings; len(row) must equal
//     len(Columns) for each row.
//   - HasHeader toggles the header row synthesized from Columns[i].Header.
type TableSpec struct {
	Columns   []Column
	Rows      [][]string
	HasHeader bool
}

// TableModeKind selects the column-sizing strategy for a StreamingTable.
type TableModeKind int

const (
	// TableModeFixed uses the Width from each Column as-is (default).
	TableModeFixed TableModeKind = 0
	// TableModeSample buffers the first N rows, measures content widths,
	// then freezes column widths for the remainder of the stream.
	TableModeSample TableModeKind = 1
)

// TableMode configures how a StreamingTable sizes its columns (issue #400).
// Use the TableModeFixed or TableModeSample constants for Kind.
type TableMode struct {
	Kind          TableModeKind
	SampleRows    int     // used when Kind == TableModeSample (default 20)
	MinColWidthPt float32 // used when Kind == TableModeSample (default 0)
	MaxColWidthPt float32 // used when Kind == TableModeSample (default 9999)
}

// StreamingTableConfig configures a StreamingTable — the row-at-a-time
// adapter returned by PageBuilder.StreamingTable.
//
//   - Columns drives widths, alignments, and the header labels.
//   - RepeatHeader repeats the header row on every page break.
//   - Mode selects the column-sizing strategy (default: TableModeFixed).
//   - MaxRowspan allows cells to span multiple rows via PushRowSpan.
//     0 or 1 disables rowspan (default); ≥2 enables it.
//   - BatchSize is the maximum number of rows accumulated in the
//     client-side buffer before an automatic flush to the native layer.
//     0 or unset defaults to 256.
type StreamingTableConfig struct {
	Columns      []Column
	RepeatHeader bool
	Mode         TableMode
	MaxRowspan   int
	BatchSize    int
}

// LogLevel represents the log verbosity level.
type LogLevel int

const (
	// LogOff disables all logging.
	LogOff LogLevel = 0
	// LogError enables error messages only.
	LogError LogLevel = 1
	// LogWarn enables warnings and errors.
	LogWarn LogLevel = 2
	// LogInfo enables informational messages.
	LogInfo LogLevel = 3
	// LogDebug enables debug messages.
	LogDebug LogLevel = 4
	// LogTrace enables verbose trace messages.
	LogTrace LogLevel = 5
)
