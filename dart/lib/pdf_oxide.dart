// pdf_oxide — idiomatic Dart bindings over the C ABI via dart:ffi.
//
// Loads the native cdylib (libpdf_oxide.{so,dylib,dll}) at runtime and exposes
// PdfDocument (extraction) + Pdf (builder). Handles are freed by NativeFinalizer
// (and explicit close()); C strings/buffers are copied to Dart and freed via
// free_string. C-ABI error codes are thrown as PdfOxideError.
//
// API surface mirrors the other language bindings; coverage is asserted by
// test/api_coverage_test.dart (one test per public method).
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

/// Thrown on any non-success C-ABI error code.
class PdfOxideError implements Exception {
  PdfOxideError(this.code, this.op);
  final int code;
  final String op;
  @override
  String toString() => 'PdfOxideError: $op failed (error code $code)';
}

// ── native signatures ────────────────────────────────────────────────────────
typedef _OpenC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _OpenBytesC = Pointer<Void> Function(
    Pointer<Uint8>, IntPtr, Pointer<Int32>);
typedef _OpenPwC = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _FreeC = Void Function(Pointer<Void>);
typedef _FreeD = void Function(Pointer<Void>);
typedef _PageCountC = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _PageCountD = int Function(Pointer<Void>, Pointer<Int32>);
typedef _VersionC = Void Function(
    Pointer<Void>, Pointer<Uint8>, Pointer<Uint8>);
typedef _VersionD = void Function(
    Pointer<Void>, Pointer<Uint8>, Pointer<Uint8>);
typedef _BoolC = Bool Function(Pointer<Void>);
typedef _BoolD = bool Function(Pointer<Void>);
typedef _TextC = Pointer<Utf8> Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _TextD = Pointer<Utf8> Function(Pointer<Void>, int, Pointer<Int32>);
typedef _TextAllC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _TextAllD = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _AuthC = Bool Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _AuthD = bool Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _FromStrC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _SaveC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _SaveD = int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _SaveBytesC = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<Int32>, Pointer<Int32>);
typedef _SaveBytesD = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<Int32>, Pointer<Int32>);
typedef _FreeStringC = Void Function(Pointer<Utf8>);
typedef _FreeStringD = void Function(Pointer<Utf8>);
typedef _FreeBytesC = Void Function(Pointer<Uint8>);
typedef _FreeBytesD = void Function(Pointer<Uint8>);

// element-extraction (Phase 1): each list is opened on a document handle, read
// element-by-element, then freed once with its `*_list_free`.
typedef _ExtractC = Pointer<Void> Function(
    Pointer<Void>, Int32, Pointer<Int32>);
typedef _ExtractD = Pointer<Void> Function(Pointer<Void>, int, Pointer<Int32>);
typedef _ListCountC = Int32 Function(Pointer<Void>);
typedef _ListCountD = int Function(Pointer<Void>);
typedef _ListStrC = Pointer<Utf8> Function(
    Pointer<Void>, Int32, Pointer<Int32>);
typedef _ListStrD = Pointer<Utf8> Function(Pointer<Void>, int, Pointer<Int32>);
typedef _ListF32C = Float Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _ListF32D = double Function(Pointer<Void>, int, Pointer<Int32>);
typedef _ListI32C = Int32 Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _ListI32D = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _ListU32C = Uint32 Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _ListU32D = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _ListBoolC = Bool Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _ListBoolD = bool Function(Pointer<Void>, int, Pointer<Int32>);
typedef _ListBboxC = Void Function(Pointer<Void>, Int32, Pointer<Float>,
    Pointer<Float>, Pointer<Float>, Pointer<Float>, Pointer<Int32>);
typedef _ListBboxD = void Function(Pointer<Void>, int, Pointer<Float>,
    Pointer<Float>, Pointer<Float>, Pointer<Float>, Pointer<Int32>);
typedef _ListFreeC = Void Function(Pointer<Void>);
typedef _ListFreeD = void Function(Pointer<Void>);
typedef _CellC = Pointer<Utf8> Function(
    Pointer<Void>, Int32, Int32, Int32, Pointer<Int32>);
typedef _CellD = Pointer<Utf8> Function(
    Pointer<Void>, int, int, int, Pointer<Int32>);

// element-extraction (Phase 2). `font_is_embedded`/`is_subset` return int32 in
// the C ABI (0/1); image data uses an int32 data_len out-param + free_bytes;
// search lists open with a term string (+ case-sensitive bool) and free with
// `pdf_oxide_search_result_free` (NOT a `*_list_free`).
typedef _ListBytesC = Pointer<Uint8> Function(
    Pointer<Void>, Int32, Pointer<Int32>, Pointer<Int32>);
typedef _ListBytesD = Pointer<Uint8> Function(
    Pointer<Void>, int, Pointer<Int32>, Pointer<Int32>);
typedef _SearchPageC = Pointer<Void> Function(
    Pointer<Void>, Int32, Pointer<Utf8>, Bool, Pointer<Int32>);
typedef _SearchPageD = Pointer<Void> Function(
    Pointer<Void>, int, Pointer<Utf8>, bool, Pointer<Int32>);
typedef _SearchAllC = Pointer<Void> Function(
    Pointer<Void>, Pointer<Utf8>, Bool, Pointer<Int32>);
typedef _SearchAllD = Pointer<Void> Function(
    Pointer<Void>, Pointer<Utf8>, bool, Pointer<Int32>);

/// Resolved native library + bound functions (loaded once).
class _Native {
  _Native(this.lib)
      : open = lib.lookupFunction<_OpenC, _OpenD>('pdf_document_open'),
        openBytes = lib.lookupFunction<_OpenBytesC, _OpenBytesD>(
            'pdf_document_open_from_bytes'),
        openPw = lib.lookupFunction<_OpenPwC, _OpenPwD>(
            'pdf_document_open_with_password'),
        docFree = lib.lookupFunction<_FreeC, _FreeD>('pdf_document_free'),
        pageCount = lib.lookupFunction<_PageCountC, _PageCountD>(
            'pdf_document_get_page_count'),
        version = lib
            .lookupFunction<_VersionC, _VersionD>('pdf_document_get_version'),
        isEncrypted =
            lib.lookupFunction<_BoolC, _BoolD>('pdf_document_is_encrypted'),
        hasTree = lib
            .lookupFunction<_BoolC, _BoolD>('pdf_document_has_structure_tree'),
        extractText =
            lib.lookupFunction<_TextC, _TextD>('pdf_document_extract_text'),
        toPlain =
            lib.lookupFunction<_TextC, _TextD>('pdf_document_to_plain_text'),
        toMd = lib.lookupFunction<_TextC, _TextD>('pdf_document_to_markdown'),
        toHtml = lib.lookupFunction<_TextC, _TextD>('pdf_document_to_html'),
        toMdAll = lib.lookupFunction<_TextAllC, _TextAllD>(
            'pdf_document_to_markdown_all'),
        toHtmlAll = lib
            .lookupFunction<_TextAllC, _TextAllD>('pdf_document_to_html_all'),
        toPlainAll = lib.lookupFunction<_TextAllC, _TextAllD>(
            'pdf_document_to_plain_text_all'),
        authenticate =
            lib.lookupFunction<_AuthC, _AuthD>('pdf_document_authenticate'),
        structJson = lib.lookupFunction<_TextC, _TextD>(
            'pdf_document_extract_structured_to_json'),
        fromMarkdown =
            lib.lookupFunction<_FromStrC, _OpenD>('pdf_from_markdown'),
        fromHtml = lib.lookupFunction<_FromStrC, _OpenD>('pdf_from_html'),
        fromText = lib.lookupFunction<_FromStrC, _OpenD>('pdf_from_text'),
        pdfFree = lib.lookupFunction<_FreeC, _FreeD>('pdf_free'),
        save = lib.lookupFunction<_SaveC, _SaveD>('pdf_save'),
        saveBytes =
            lib.lookupFunction<_SaveBytesC, _SaveBytesD>('pdf_save_to_bytes'),
        freeString =
            lib.lookupFunction<_FreeStringC, _FreeStringD>('free_string'),
        freeBytes = lib.lookupFunction<_FreeBytesC, _FreeBytesD>('free_bytes'),
        // chars
        extractChars = lib
            .lookupFunction<_ExtractC, _ExtractD>('pdf_document_extract_chars'),
        charCount = lib
            .lookupFunction<_ListCountC, _ListCountD>('pdf_oxide_char_count'),
        charGetChar =
            lib.lookupFunction<_ListU32C, _ListU32D>('pdf_oxide_char_get_char'),
        charGetBbox = lib
            .lookupFunction<_ListBboxC, _ListBboxD>('pdf_oxide_char_get_bbox'),
        charGetFontName = lib.lookupFunction<_ListStrC, _ListStrD>(
            'pdf_oxide_char_get_font_name'),
        charGetFontSize = lib.lookupFunction<_ListF32C, _ListF32D>(
            'pdf_oxide_char_get_font_size'),
        charListFree = lib
            .lookupFunction<_ListFreeC, _ListFreeD>('pdf_oxide_char_list_free'),
        // words
        extractWords = lib
            .lookupFunction<_ExtractC, _ExtractD>('pdf_document_extract_words'),
        wordCount = lib
            .lookupFunction<_ListCountC, _ListCountD>('pdf_oxide_word_count'),
        wordGetText =
            lib.lookupFunction<_ListStrC, _ListStrD>('pdf_oxide_word_get_text'),
        wordGetBbox = lib
            .lookupFunction<_ListBboxC, _ListBboxD>('pdf_oxide_word_get_bbox'),
        wordGetFontName = lib.lookupFunction<_ListStrC, _ListStrD>(
            'pdf_oxide_word_get_font_name'),
        wordGetFontSize = lib.lookupFunction<_ListF32C, _ListF32D>(
            'pdf_oxide_word_get_font_size'),
        wordIsBold = lib
            .lookupFunction<_ListBoolC, _ListBoolD>('pdf_oxide_word_is_bold'),
        wordListFree = lib
            .lookupFunction<_ListFreeC, _ListFreeD>('pdf_oxide_word_list_free'),
        // text lines
        extractLines = lib.lookupFunction<_ExtractC, _ExtractD>(
            'pdf_document_extract_text_lines'),
        lineCount = lib
            .lookupFunction<_ListCountC, _ListCountD>('pdf_oxide_line_count'),
        lineGetText =
            lib.lookupFunction<_ListStrC, _ListStrD>('pdf_oxide_line_get_text'),
        lineGetBbox = lib
            .lookupFunction<_ListBboxC, _ListBboxD>('pdf_oxide_line_get_bbox'),
        lineGetWordCount = lib.lookupFunction<_ListI32C, _ListI32D>(
            'pdf_oxide_line_get_word_count'),
        lineListFree = lib
            .lookupFunction<_ListFreeC, _ListFreeD>('pdf_oxide_line_list_free'),
        // tables
        extractTables = lib.lookupFunction<_ExtractC, _ExtractD>(
            'pdf_document_extract_tables'),
        tableCount = lib
            .lookupFunction<_ListCountC, _ListCountD>('pdf_oxide_table_count'),
        tableGetRowCount = lib.lookupFunction<_ListI32C, _ListI32D>(
            'pdf_oxide_table_get_row_count'),
        tableGetColCount = lib.lookupFunction<_ListI32C, _ListI32D>(
            'pdf_oxide_table_get_col_count'),
        tableGetCellText =
            lib.lookupFunction<_CellC, _CellD>('pdf_oxide_table_get_cell_text'),
        tableHasHeader = lib.lookupFunction<_ListBoolC, _ListBoolD>(
            'pdf_oxide_table_has_header'),
        tableListFree = lib.lookupFunction<_ListFreeC, _ListFreeD>(
            'pdf_oxide_table_list_free'),
        // fonts
        extractFonts = lib.lookupFunction<_ExtractC, _ExtractD>(
            'pdf_document_get_embedded_fonts'),
        fontCount = lib
            .lookupFunction<_ListCountC, _ListCountD>('pdf_oxide_font_count'),
        fontGetName =
            lib.lookupFunction<_ListStrC, _ListStrD>('pdf_oxide_font_get_name'),
        fontGetType =
            lib.lookupFunction<_ListStrC, _ListStrD>('pdf_oxide_font_get_type'),
        fontGetEncoding = lib.lookupFunction<_ListStrC, _ListStrD>(
            'pdf_oxide_font_get_encoding'),
        fontIsEmbedded = lib
            .lookupFunction<_ListI32C, _ListI32D>('pdf_oxide_font_is_embedded'),
        fontIsSubset = lib
            .lookupFunction<_ListI32C, _ListI32D>('pdf_oxide_font_is_subset'),
        fontListFree = lib
            .lookupFunction<_ListFreeC, _ListFreeD>('pdf_oxide_font_list_free'),
        // images
        extractImages = lib.lookupFunction<_ExtractC, _ExtractD>(
            'pdf_document_get_embedded_images'),
        imageCount = lib
            .lookupFunction<_ListCountC, _ListCountD>('pdf_oxide_image_count'),
        imageGetWidth = lib
            .lookupFunction<_ListI32C, _ListI32D>('pdf_oxide_image_get_width'),
        imageGetHeight = lib
            .lookupFunction<_ListI32C, _ListI32D>('pdf_oxide_image_get_height'),
        imageGetBitsPerComponent = lib.lookupFunction<_ListI32C, _ListI32D>(
            'pdf_oxide_image_get_bits_per_component'),
        imageGetFormat = lib
            .lookupFunction<_ListStrC, _ListStrD>('pdf_oxide_image_get_format'),
        imageGetColorspace = lib.lookupFunction<_ListStrC, _ListStrD>(
            'pdf_oxide_image_get_colorspace'),
        imageGetData = lib.lookupFunction<_ListBytesC, _ListBytesD>(
            'pdf_oxide_image_get_data'),
        imageListFree = lib.lookupFunction<_ListFreeC, _ListFreeD>(
            'pdf_oxide_image_list_free'),
        // annotations
        extractAnnotations = lib.lookupFunction<_ExtractC, _ExtractD>(
            'pdf_document_get_page_annotations'),
        annotationCount = lib.lookupFunction<_ListCountC, _ListCountD>(
            'pdf_oxide_annotation_count'),
        annotationGetType = lib.lookupFunction<_ListStrC, _ListStrD>(
            'pdf_oxide_annotation_get_type'),
        annotationGetSubtype = lib.lookupFunction<_ListStrC, _ListStrD>(
            'pdf_oxide_annotation_get_subtype'),
        annotationGetContent = lib.lookupFunction<_ListStrC, _ListStrD>(
            'pdf_oxide_annotation_get_content'),
        annotationGetAuthor = lib.lookupFunction<_ListStrC, _ListStrD>(
            'pdf_oxide_annotation_get_author'),
        annotationGetRect = lib.lookupFunction<_ListBboxC, _ListBboxD>(
            'pdf_oxide_annotation_get_rect'),
        annotationGetBorderWidth = lib.lookupFunction<_ListF32C, _ListF32D>(
            'pdf_oxide_annotation_get_border_width'),
        annotationListFree = lib.lookupFunction<_ListFreeC, _ListFreeD>(
            'pdf_oxide_annotation_list_free'),
        // paths
        extractPaths = lib
            .lookupFunction<_ExtractC, _ExtractD>('pdf_document_extract_paths'),
        pathCount = lib
            .lookupFunction<_ListCountC, _ListCountD>('pdf_oxide_path_count'),
        pathGetBbox = lib
            .lookupFunction<_ListBboxC, _ListBboxD>('pdf_oxide_path_get_bbox'),
        pathGetStrokeWidth = lib.lookupFunction<_ListF32C, _ListF32D>(
            'pdf_oxide_path_get_stroke_width'),
        pathHasStroke = lib.lookupFunction<_ListBoolC, _ListBoolD>(
            'pdf_oxide_path_has_stroke'),
        pathHasFill = lib
            .lookupFunction<_ListBoolC, _ListBoolD>('pdf_oxide_path_has_fill'),
        pathGetOperationCount = lib.lookupFunction<_ListI32C, _ListI32D>(
            'pdf_oxide_path_get_operation_count'),
        pathListFree = lib
            .lookupFunction<_ListFreeC, _ListFreeD>('pdf_oxide_path_list_free'),
        // search
        searchPage = lib.lookupFunction<_SearchPageC, _SearchPageD>(
            'pdf_document_search_page'),
        searchAll = lib.lookupFunction<_SearchAllC, _SearchAllD>(
            'pdf_document_search_all'),
        searchResultCount = lib.lookupFunction<_ListCountC, _ListCountD>(
            'pdf_oxide_search_result_count'),
        searchResultGetText = lib.lookupFunction<_ListStrC, _ListStrD>(
            'pdf_oxide_search_result_get_text'),
        searchResultGetPage = lib.lookupFunction<_ListI32C, _ListI32D>(
            'pdf_oxide_search_result_get_page'),
        searchResultGetBbox = lib.lookupFunction<_ListBboxC, _ListBboxD>(
            'pdf_oxide_search_result_get_bbox'),
        searchResultFree = lib.lookupFunction<_ListFreeC, _ListFreeD>(
            'pdf_oxide_search_result_free');

  final DynamicLibrary lib;
  final _OpenD open;
  final _OpenBytesD openBytes;
  final _OpenPwD openPw;
  final _FreeD docFree;
  final _PageCountD pageCount;
  final _VersionD version;
  final _BoolD isEncrypted;
  final _BoolD hasTree;
  final _TextD extractText, toPlain, toMd, toHtml, structJson;
  final _TextAllD toMdAll, toHtmlAll, toPlainAll;
  final _AuthD authenticate;
  final _OpenD fromMarkdown, fromHtml, fromText;
  final _FreeD pdfFree;
  final _SaveD save;
  final _SaveBytesD saveBytes;
  final _FreeStringD freeString;
  final _FreeBytesD freeBytes;
  // chars
  final _ExtractD extractChars;
  final _ListCountD charCount;
  final _ListU32D charGetChar;
  final _ListBboxD charGetBbox;
  final _ListStrD charGetFontName;
  final _ListF32D charGetFontSize;
  final _ListFreeD charListFree;
  // words
  final _ExtractD extractWords;
  final _ListCountD wordCount;
  final _ListStrD wordGetText;
  final _ListBboxD wordGetBbox;
  final _ListStrD wordGetFontName;
  final _ListF32D wordGetFontSize;
  final _ListBoolD wordIsBold;
  final _ListFreeD wordListFree;
  // text lines
  final _ExtractD extractLines;
  final _ListCountD lineCount;
  final _ListStrD lineGetText;
  final _ListBboxD lineGetBbox;
  final _ListI32D lineGetWordCount;
  final _ListFreeD lineListFree;
  // tables
  final _ExtractD extractTables;
  final _ListCountD tableCount;
  final _ListI32D tableGetRowCount;
  final _ListI32D tableGetColCount;
  final _CellD tableGetCellText;
  final _ListBoolD tableHasHeader;
  final _ListFreeD tableListFree;
  // fonts
  final _ExtractD extractFonts;
  final _ListCountD fontCount;
  final _ListStrD fontGetName;
  final _ListStrD fontGetType;
  final _ListStrD fontGetEncoding;
  final _ListI32D fontIsEmbedded;
  final _ListI32D fontIsSubset;
  final _ListFreeD fontListFree;
  // images
  final _ExtractD extractImages;
  final _ListCountD imageCount;
  final _ListI32D imageGetWidth;
  final _ListI32D imageGetHeight;
  final _ListI32D imageGetBitsPerComponent;
  final _ListStrD imageGetFormat;
  final _ListStrD imageGetColorspace;
  final _ListBytesD imageGetData;
  final _ListFreeD imageListFree;
  // annotations
  final _ExtractD extractAnnotations;
  final _ListCountD annotationCount;
  final _ListStrD annotationGetType;
  final _ListStrD annotationGetSubtype;
  final _ListStrD annotationGetContent;
  final _ListStrD annotationGetAuthor;
  final _ListBboxD annotationGetRect;
  final _ListF32D annotationGetBorderWidth;
  final _ListFreeD annotationListFree;
  // paths
  final _ExtractD extractPaths;
  final _ListCountD pathCount;
  final _ListBboxD pathGetBbox;
  final _ListF32D pathGetStrokeWidth;
  final _ListBoolD pathHasStroke;
  final _ListBoolD pathHasFill;
  final _ListI32D pathGetOperationCount;
  final _ListFreeD pathListFree;
  // search
  final _SearchPageD searchPage;
  final _SearchAllD searchAll;
  final _ListCountD searchResultCount;
  final _ListStrD searchResultGetText;
  final _ListI32D searchResultGetPage;
  final _ListBboxD searchResultGetBbox;
  final _ListFreeD searchResultFree;
}

typedef _OpenD = Pointer<Void> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _OpenBytesD = Pointer<Void> Function(
    Pointer<Uint8>, int, Pointer<Int32>);
typedef _OpenPwD = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);

_Native? _cached;

/// Locate and load the native library. Override the path with the
/// `PDF_OXIDE_LIB_PATH` environment variable, else search common build dirs.
DynamicLibrary _load() {
  final env = Platform.environment['PDF_OXIDE_LIB_PATH'];
  if (env != null && File(env).existsSync()) return DynamicLibrary.open(env);
  final name = Platform.isMacOS
      ? 'libpdf_oxide.dylib'
      : Platform.isWindows
          ? 'pdf_oxide.dll'
          : 'libpdf_oxide.so';
  for (final dir in [
    Platform.environment['PDF_OXIDE_LIB_DIR'],
    '../target/release',
    'target/release',
  ]) {
    if (dir == null) continue;
    final p = '$dir/$name';
    if (File(p).existsSync()) return DynamicLibrary.open(p);
  }
  return DynamicLibrary.open(name); // fall back to system loader path
}

_Native get _n => _cached ??= _Native(_load());

String _takeString(Pointer<Utf8> p, int code, String op) {
  if (p == nullptr) throw PdfOxideError(code, op);
  final s = p.toDartString();
  _n.freeString(p);
  return s;
}

/// PDF version (e.g. 1.7).
class PdfVersion {
  const PdfVersion(this.major, this.minor);
  final int major;
  final int minor;
  @override
  String toString() => '$major.$minor';
}

/// An axis-aligned bounding box in PDF user-space points.
class Bbox {
  const Bbox(this.x, this.y, this.width, this.height);
  final double x;
  final double y;
  final double width;
  final double height;
  @override
  String toString() => 'Bbox($x, $y, $width, $height)';
}

/// A single extracted glyph. [character] is the Unicode codepoint.
class Char {
  const Char(this.character, this.bbox, this.fontName, this.fontSize);

  /// The Unicode codepoint of this glyph.
  final int character;
  final Bbox bbox;
  final String fontName;
  final double fontSize;
}

/// A single extracted word.
class Word {
  const Word(this.text, this.bbox, this.fontName, this.fontSize, this.bold);
  final String text;
  final Bbox bbox;
  final String fontName;
  final double fontSize;
  final bool bold;
}

/// A single extracted line of text.
class TextLine {
  const TextLine(this.text, this.bbox, this.wordCount);
  final String text;
  final Bbox bbox;
  final int wordCount;
}

/// A single extracted table. Cells are read lazily via [cell].
class Table {
  const Table(this.rowCount, this.colCount, this.hasHeader, this._cell);
  final int rowCount;
  final int colCount;
  final bool hasHeader;
  final String Function(int row, int col) _cell;

  /// Text of the cell at 0-based [row]/[col].
  String cell(int row, int col) => _cell(row, col);
}

/// An embedded font referenced by a page.
class Font {
  const Font(this.name, this.type, this.encoding, this.embedded, this.subset);
  final String name;
  final String type;
  final String encoding;
  final bool embedded;
  final bool subset;
}

/// An embedded image. [data] holds the raw image bytes.
class Image {
  const Image(this.width, this.height, this.bitsPerComponent, this.format,
      this.colorspace, this.data);
  final int width;
  final int height;
  final int bitsPerComponent;
  final String format;
  final String colorspace;
  final Uint8List data;
}

/// A page annotation.
class Annotation {
  const Annotation(this.type, this.subtype, this.content, this.author,
      this.rect, this.borderWidth);
  final String type;
  final String subtype;
  final String content;
  final String author;
  final Bbox rect;
  final double borderWidth;
}

/// A vector path (graphics) element on a page.
class Path {
  const Path(this.bbox, this.strokeWidth, this.hasStroke, this.hasFill,
      this.operationCount);
  final Bbox bbox;
  final double strokeWidth;
  final bool hasStroke;
  final bool hasFill;
  final int operationCount;
}

/// A single search hit.
class SearchResult {
  const SearchResult(this.text, this.page, this.bbox);
  final String text;
  final int page;
  final Bbox bbox;
}

/// An opened PDF for extraction/inspection. Call [close] when done (or rely on
/// the finalizer).
class PdfDocument implements Finalizable {
  PdfDocument._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_FreeC>>('pdf_document_free'));
  Pointer<Void> _handle;

  /// Open a PDF from a filesystem path.
  static PdfDocument open(String path) {
    final cPath = path.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.open(cPath, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'open');
      return PdfDocument._(h);
    } finally {
      calloc.free(cPath);
      calloc.free(code);
    }
  }

  /// Open a PDF from in-memory bytes.
  static PdfDocument openFromBytes(Uint8List data) {
    final buf = calloc<Uint8>(data.length);
    buf.asTypedList(data.length).setAll(0, data);
    final code = calloc<Int32>();
    try {
      final h = _n.openBytes(buf, data.length, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'openFromBytes');
      return PdfDocument._(h);
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  /// Open a password-protected PDF.
  static PdfDocument openWithPassword(String path, String password) {
    final cPath = path.toNativeUtf8();
    final cPw = password.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.openPw(cPath, cPw, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'openWithPassword');
      return PdfDocument._(h);
    } finally {
      calloc.free(cPath);
      calloc.free(cPw);
      calloc.free(code);
    }
  }

  void _check() {
    if (_handle == nullptr) throw StateError('PdfDocument is closed');
  }

  int get pageCount {
    _check();
    final code = calloc<Int32>();
    try {
      final n = _n.pageCount(_handle, code);
      if (n < 0) throw PdfOxideError(code.value, 'pageCount');
      return n;
    } finally {
      calloc.free(code);
    }
  }

  PdfVersion get version {
    _check();
    final maj = calloc<Uint8>();
    final min = calloc<Uint8>();
    try {
      _n.version(_handle, maj, min);
      return PdfVersion(maj.value, min.value);
    } finally {
      calloc.free(maj);
      calloc.free(min);
    }
  }

  bool isEncrypted() {
    _check();
    return _n.isEncrypted(_handle);
  }

  bool hasStructureTree() {
    _check();
    return _n.hasTree(_handle);
  }

  String _strPage(_TextD fn, int page, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(fn(_handle, page, code), code.value, op);
    } finally {
      calloc.free(code);
    }
  }

  String extractText(int page) => _strPage(_n.extractText, page, 'extractText');
  String toPlainText(int page) => _strPage(_n.toPlain, page, 'toPlainText');
  String toMarkdown(int page) => _strPage(_n.toMd, page, 'toMarkdown');
  String toHtml(int page) => _strPage(_n.toHtml, page, 'toHtml');
  String extractStructuredJson(int page) =>
      _strPage(_n.structJson, page, 'extractStructuredJson');

  // ── element extraction (Phase 1) ───────────────────────────────────────────

  /// Read a bbox out-param tuple for element [i] from a list [handle].
  Bbox _bbox(_ListBboxD fn, Pointer<Void> handle, int i, String op) {
    final x = calloc<Float>();
    final y = calloc<Float>();
    final w = calloc<Float>();
    final h = calloc<Float>();
    final code = calloc<Int32>();
    try {
      fn(handle, i, x, y, w, h, code);
      if (code.value != 0) throw PdfOxideError(code.value, op);
      return Bbox(x.value, y.value, w.value, h.value);
    } finally {
      calloc.free(x);
      calloc.free(y);
      calloc.free(w);
      calloc.free(h);
      calloc.free(code);
    }
  }

  /// Open an element list on this document for [page], or throw on error.
  Pointer<Void> _openList(_ExtractD fn, int page, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = fn(_handle, page, code);
      if (h == nullptr) throw PdfOxideError(code.value, op);
      return h;
    } finally {
      calloc.free(code);
    }
  }

  /// Extract individual glyphs from 0-based [page].
  List<Char> extractChars(int page) {
    final list = _openList(_n.extractChars, page, 'extractChars');
    final code = calloc<Int32>();
    try {
      final n = _n.charCount(list);
      final out = <Char>[];
      for (var i = 0; i < n; i++) {
        final cp = _n.charGetChar(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractChars');
        final bbox = _bbox(_n.charGetBbox, list, i, 'extractChars');
        final fontName = _takeString(
            _n.charGetFontName(list, i, code), code.value, 'extractChars');
        final fontSize = _n.charGetFontSize(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractChars');
        out.add(Char(cp, bbox, fontName, fontSize));
      }
      return out;
    } finally {
      _n.charListFree(list);
      calloc.free(code);
    }
  }

  /// Extract words from 0-based [page].
  List<Word> extractWords(int page) {
    final list = _openList(_n.extractWords, page, 'extractWords');
    final code = calloc<Int32>();
    try {
      final n = _n.wordCount(list);
      final out = <Word>[];
      for (var i = 0; i < n; i++) {
        final text = _takeString(
            _n.wordGetText(list, i, code), code.value, 'extractWords');
        final bbox = _bbox(_n.wordGetBbox, list, i, 'extractWords');
        final fontName = _takeString(
            _n.wordGetFontName(list, i, code), code.value, 'extractWords');
        final fontSize = _n.wordGetFontSize(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractWords');
        final bold = _n.wordIsBold(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractWords');
        out.add(Word(text, bbox, fontName, fontSize, bold));
      }
      return out;
    } finally {
      _n.wordListFree(list);
      calloc.free(code);
    }
  }

  /// Extract text lines from 0-based [page].
  List<TextLine> extractTextLines(int page) {
    final list = _openList(_n.extractLines, page, 'extractTextLines');
    final code = calloc<Int32>();
    try {
      final n = _n.lineCount(list);
      final out = <TextLine>[];
      for (var i = 0; i < n; i++) {
        final text = _takeString(
            _n.lineGetText(list, i, code), code.value, 'extractTextLines');
        final bbox = _bbox(_n.lineGetBbox, list, i, 'extractTextLines');
        final wordCount = _n.lineGetWordCount(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'extractTextLines');
        }
        out.add(TextLine(text, bbox, wordCount));
      }
      return out;
    } finally {
      _n.lineListFree(list);
      calloc.free(code);
    }
  }

  /// Extract tables from 0-based [page]. Each [Table] exposes its cells lazily
  /// via [Table.cell]; the underlying list is copied/closed before returning.
  List<Table> extractTables(int page) {
    final list = _openList(_n.extractTables, page, 'extractTables');
    final code = calloc<Int32>();
    try {
      final n = _n.tableCount(list);
      final out = <Table>[];
      for (var i = 0; i < n; i++) {
        final rowCount = _n.tableGetRowCount(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractTables');
        final colCount = _n.tableGetColCount(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractTables');
        final hasHeader = _n.tableHasHeader(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractTables');
        // Eagerly read all cells so the table outlives the freed list.
        final cells = <String>[];
        for (var r = 0; r < rowCount; r++) {
          for (var c = 0; c < colCount; c++) {
            cells.add(_takeString(_n.tableGetCellText(list, i, r, c, code),
                code.value, 'extractTables'));
          }
        }
        out.add(Table(
            rowCount, colCount, hasHeader, (r, c) => cells[r * colCount + c]));
      }
      return out;
    } finally {
      _n.tableListFree(list);
      calloc.free(code);
    }
  }

  // ── element extraction (Phase 2) ───────────────────────────────────────────

  /// Embedded fonts referenced by 0-based [page].
  List<Font> embeddedFonts(int page) {
    final list = _openList(_n.extractFonts, page, 'embeddedFonts');
    final code = calloc<Int32>();
    try {
      final n = _n.fontCount(list);
      final out = <Font>[];
      for (var i = 0; i < n; i++) {
        final name = _takeString(
            _n.fontGetName(list, i, code), code.value, 'embeddedFonts');
        final type = _takeString(
            _n.fontGetType(list, i, code), code.value, 'embeddedFonts');
        final encoding = _takeString(
            _n.fontGetEncoding(list, i, code), code.value, 'embeddedFonts');
        final embedded = _n.fontIsEmbedded(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'embeddedFonts');
        final subset = _n.fontIsSubset(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'embeddedFonts');
        out.add(Font(name, type, encoding, embedded != 0, subset != 0));
      }
      return out;
    } finally {
      _n.fontListFree(list);
      calloc.free(code);
    }
  }

  /// Embedded images on 0-based [page].
  List<Image> embeddedImages(int page) {
    final list = _openList(_n.extractImages, page, 'embeddedImages');
    final code = calloc<Int32>();
    final len = calloc<Int32>();
    try {
      final n = _n.imageCount(list);
      final out = <Image>[];
      for (var i = 0; i < n; i++) {
        final width = _n.imageGetWidth(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'embeddedImages');
        final height = _n.imageGetHeight(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'embeddedImages');
        final bpc = _n.imageGetBitsPerComponent(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'embeddedImages');
        final format = _takeString(
            _n.imageGetFormat(list, i, code), code.value, 'embeddedImages');
        final colorspace = _takeString(
            _n.imageGetColorspace(list, i, code), code.value, 'embeddedImages');
        final p = _n.imageGetData(list, i, len, code);
        if (p == nullptr) throw PdfOxideError(code.value, 'embeddedImages');
        final data =
            Uint8List.fromList(p.asTypedList(len.value < 0 ? 0 : len.value));
        _n.freeBytes(p);
        out.add(Image(width, height, bpc, format, colorspace, data));
      }
      return out;
    } finally {
      _n.imageListFree(list);
      calloc.free(code);
      calloc.free(len);
    }
  }

  /// Annotations on 0-based [page].
  List<Annotation> pageAnnotations(int page) {
    final list = _openList(_n.extractAnnotations, page, 'pageAnnotations');
    final code = calloc<Int32>();
    try {
      final n = _n.annotationCount(list);
      final out = <Annotation>[];
      for (var i = 0; i < n; i++) {
        final type = _takeString(
            _n.annotationGetType(list, i, code), code.value, 'pageAnnotations');
        final subtype = _takeString(_n.annotationGetSubtype(list, i, code),
            code.value, 'pageAnnotations');
        final content = _takeString(_n.annotationGetContent(list, i, code),
            code.value, 'pageAnnotations');
        final author = _takeString(_n.annotationGetAuthor(list, i, code),
            code.value, 'pageAnnotations');
        final rect = _bbox(_n.annotationGetRect, list, i, 'pageAnnotations');
        final borderWidth = _n.annotationGetBorderWidth(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'pageAnnotations');
        }
        out.add(Annotation(type, subtype, content, author, rect, borderWidth));
      }
      return out;
    } finally {
      _n.annotationListFree(list);
      calloc.free(code);
    }
  }

  /// Vector paths on 0-based [page].
  List<Path> extractPaths(int page) {
    final list = _openList(_n.extractPaths, page, 'extractPaths');
    final code = calloc<Int32>();
    try {
      final n = _n.pathCount(list);
      final out = <Path>[];
      for (var i = 0; i < n; i++) {
        final bbox = _bbox(_n.pathGetBbox, list, i, 'extractPaths');
        final strokeWidth = _n.pathGetStrokeWidth(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractPaths');
        final hasStroke = _n.pathHasStroke(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractPaths');
        final hasFill = _n.pathHasFill(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractPaths');
        final operationCount = _n.pathGetOperationCount(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'extractPaths');
        out.add(Path(bbox, strokeWidth, hasStroke, hasFill, operationCount));
      }
      return out;
    } finally {
      _n.pathListFree(list);
      calloc.free(code);
    }
  }

  /// Read a search-results list (already opened) into [SearchResult]s, then
  /// free it via `pdf_oxide_search_result_free`.
  List<SearchResult> _readSearch(Pointer<Void> list, String op) {
    final code = calloc<Int32>();
    try {
      final n = _n.searchResultCount(list);
      final out = <SearchResult>[];
      for (var i = 0; i < n; i++) {
        final text =
            _takeString(_n.searchResultGetText(list, i, code), code.value, op);
        final hitPage = _n.searchResultGetPage(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, op);
        final bbox = _bbox(_n.searchResultGetBbox, list, i, op);
        out.add(SearchResult(text, hitPage, bbox));
      }
      return out;
    } finally {
      _n.searchResultFree(list);
      calloc.free(code);
    }
  }

  /// Search a single 0-based [page] for [term].
  List<SearchResult> search(int page, String term, bool caseSensitive) {
    _check();
    final cTerm = term.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final list = _n.searchPage(_handle, page, cTerm, caseSensitive, code);
      if (list == nullptr) throw PdfOxideError(code.value, 'search');
      return _readSearch(list, 'search');
    } finally {
      calloc.free(cTerm);
      calloc.free(code);
    }
  }

  /// Search the whole document for [term].
  List<SearchResult> searchAll(String term, bool caseSensitive) {
    _check();
    final cTerm = term.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final list = _n.searchAll(_handle, cTerm, caseSensitive, code);
      if (list == nullptr) throw PdfOxideError(code.value, 'searchAll');
      return _readSearch(list, 'searchAll');
    } finally {
      calloc.free(cTerm);
      calloc.free(code);
    }
  }

  String toMarkdownAll() {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(
          _n.toMdAll(_handle, code), code.value, 'toMarkdownAll');
    } finally {
      calloc.free(code);
    }
  }

  String toHtmlAll() {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(_n.toHtmlAll(_handle, code), code.value, 'toHtmlAll');
    } finally {
      calloc.free(code);
    }
  }

  String toPlainTextAll() {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(
          _n.toPlainAll(_handle, code), code.value, 'toPlainTextAll');
    } finally {
      calloc.free(code);
    }
  }

  /// Authenticate against an encrypted PDF. Returns `true` on success and
  /// `false` for a wrong password (without throwing); throws [PdfOxideError]
  /// only on an actual error.
  bool authenticate(String password) {
    _check();
    final cPw = password.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final ok = _n.authenticate(_handle, cPw, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'authenticate');
      return ok;
    } finally {
      calloc.free(cPw);
      calloc.free(code);
    }
  }

  /// A lightweight view of a single 0-based page. The returned [Page] keeps a
  /// reference to this document and must not be used after [close].
  Page page(int index) => Page._(this, index);

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.docFree(_handle);
      _handle = nullptr;
    }
  }
}

/// A lightweight, 0-based view of a single page of a [PdfDocument]. Holds a
/// strong reference to its document (so the document is not collected while the
/// page is alive); extraction delegates to the document's per-page methods.
class Page {
  Page._(this._doc, this.index);

  final PdfDocument _doc;

  /// 0-based page index.
  final int index;

  String text() => _doc.extractText(index);
  String markdown() => _doc.toMarkdown(index);
  String html() => _doc.toHtml(index);
  String plainText() => _doc.toPlainText(index);
}

/// A PDF produced by a builder. Call [close] when done.
class Pdf implements Finalizable {
  Pdf._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer =
      NativeFinalizer(_n.lib.lookup<NativeFunction<_FreeC>>('pdf_free'));
  Pointer<Void> _handle;

  static Pdf _from(_OpenD fn, String input, String op) {
    final c = input.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = fn(c, code);
      if (h == nullptr) throw PdfOxideError(code.value, op);
      return Pdf._(h);
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  static Pdf fromMarkdown(String md) =>
      _from(_n.fromMarkdown, md, 'fromMarkdown');
  static Pdf fromHtml(String html) => _from(_n.fromHtml, html, 'fromHtml');
  static Pdf fromText(String text) => _from(_n.fromText, text, 'fromText');

  void _check() {
    if (_handle == nullptr) throw StateError('Pdf is closed');
  }

  void save(String path) {
    _check();
    final c = path.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.save(_handle, c, code) != 0) {
        throw PdfOxideError(code.value, 'save');
      }
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  Uint8List toBytes() {
    _check();
    final len = calloc<Int32>();
    final code = calloc<Int32>();
    try {
      final p = _n.saveBytes(_handle, len, code);
      if (p == nullptr) throw PdfOxideError(code.value, 'toBytes');
      final out =
          Uint8List.fromList(p.asTypedList(len.value < 0 ? 0 : len.value));
      _n.freeBytes(p);
      return out;
    } finally {
      calloc.free(len);
      calloc.free(code);
    }
  }

  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.pdfFree(_handle);
      _handle = nullptr;
    }
  }
}
