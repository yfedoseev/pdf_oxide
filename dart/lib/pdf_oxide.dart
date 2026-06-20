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
        freeBytes = lib.lookupFunction<_FreeBytesC, _FreeBytesD>('free_bytes');

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
  final _TextAllD toMdAll;
  final _OpenD fromMarkdown, fromHtml, fromText;
  final _FreeD pdfFree;
  final _SaveD save;
  final _SaveBytesD saveBytes;
  final _FreeStringD freeString;
  final _FreeBytesD freeBytes;
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

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.docFree(_handle);
      _handle = nullptr;
    }
  }
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

  Uint8List saveToBytes() {
    _check();
    final len = calloc<Int32>();
    final code = calloc<Int32>();
    try {
      final p = _n.saveBytes(_handle, len, code);
      if (p == nullptr) throw PdfOxideError(code.value, 'saveToBytes');
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
