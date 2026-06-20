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
            'pdf_oxide_table_list_free');

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
