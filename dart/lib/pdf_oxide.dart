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

// page rendering (Phase 3). Render entry points open an FfiRenderedImage handle
// (NULL on error); accessors read width/height/data; `pdf_save_rendered_image`
// writes the encoded bytes to disk; the handle is freed via
// `pdf_rendered_image_free` (NOT free_bytes — only the data buffer is).
typedef _RenderPageC = Pointer<Void> Function(
    Pointer<Void>, Int32, Int32, Pointer<Int32>);
typedef _RenderPageD = Pointer<Void> Function(
    Pointer<Void>, int, int, Pointer<Int32>);
typedef _RenderZoomC = Pointer<Void> Function(
    Pointer<Void>, Int32, Float, Int32, Pointer<Int32>);
typedef _RenderZoomD = Pointer<Void> Function(
    Pointer<Void>, int, double, int, Pointer<Int32>);
typedef _RenderThumbC = Pointer<Void> Function(
    Pointer<Void>, Int32, Int32, Int32, Pointer<Int32>);
typedef _RenderThumbD = Pointer<Void> Function(
    Pointer<Void>, int, int, int, Pointer<Int32>);
typedef _RenderedDimC = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _RenderedDimD = int Function(Pointer<Void>, Pointer<Int32>);
typedef _RenderedDataC = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<Int32>, Pointer<Int32>);
typedef _RenderedDataD = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<Int32>, Pointer<Int32>);
typedef _RenderedSaveC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _RenderedSaveD = int Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _RenderedFreeC = Void Function(Pointer<Void>);
typedef _RenderedFreeD = void Function(Pointer<Void>);

// document editing. The DocumentEditor handle is an opaque pointer freed via
// `document_editor_free`. Page indices follow each C signature exactly: some
// args are uintptr_t (IntPtr), some int32_t (Int32). String returns are owned
// char* (-> free_string); byte returns are uint8* with a uintptr_t out-len
// (-> free_bytes). int32 status returns are 0 = success.
typedef _DeOpenC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _DeOpenD = Pointer<Void> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _DeOpenBytesC = Pointer<Void> Function(
    Pointer<Uint8>, IntPtr, Pointer<Int32>);
typedef _DeOpenBytesD = Pointer<Void> Function(
    Pointer<Uint8>, int, Pointer<Int32>);
typedef _DeFreeC = Void Function(Pointer<Void>);
typedef _DeFreeD = void Function(Pointer<Void>);
typedef _DeBoolC = Bool Function(Pointer<Void>);
typedef _DeBoolD = bool Function(Pointer<Void>);
typedef _DeStrC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _DeStrD = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _DeVersionC = Void Function(
    Pointer<Void>, Pointer<Uint8>, Pointer<Uint8>);
typedef _DeVersionD = void Function(
    Pointer<Void>, Pointer<Uint8>, Pointer<Uint8>);
typedef _DeI32C = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _DeI32D = int Function(Pointer<Void>, Pointer<Int32>);
typedef _DeSetStrC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _DeSetStrD = int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _DeSaveC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _DeSaveD = int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _DeSaveBytesC = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<IntPtr>, Pointer<Int32>);
typedef _DeSaveBytesD = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<IntPtr>, Pointer<Int32>);
typedef _DeSaveBytesOptsC = Pointer<Uint8> Function(
    Pointer<Void>, Bool, Bool, Bool, Pointer<IntPtr>, Pointer<Int32>);
typedef _DeSaveBytesOptsD = Pointer<Uint8> Function(
    Pointer<Void>, bool, bool, bool, Pointer<IntPtr>, Pointer<Int32>);
typedef _DeExtractPagesC = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<Int32>, IntPtr, Pointer<IntPtr>, Pointer<Int32>);
typedef _DeExtractPagesD = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<Int32>, int, Pointer<IntPtr>, Pointer<Int32>);
typedef _DeConvertPdfAC = Int32 Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _DeConvertPdfAD = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _DeSaveEncBytesC = Pointer<Uint8> Function(Pointer<Void>, Pointer<Utf8>,
    Pointer<Utf8>, Pointer<IntPtr>, Pointer<Int32>);
typedef _DeSaveEncBytesD = Pointer<Uint8> Function(Pointer<Void>, Pointer<Utf8>,
    Pointer<Utf8>, Pointer<IntPtr>, Pointer<Int32>);
typedef _DeMergeBytesC = Int32 Function(
    Pointer<Void>, Pointer<Uint8>, IntPtr, Pointer<Int32>);
typedef _DeMergeBytesD = int Function(
    Pointer<Void>, Pointer<Uint8>, int, Pointer<Int32>);
typedef _DeEmbedFileC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Uint8>, IntPtr, Pointer<Int32>);
typedef _DeEmbedFileD = int Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Uint8>, int, Pointer<Int32>);
typedef _DeUsizeI32C = Int32 Function(Pointer<Void>, IntPtr, Pointer<Int32>);
typedef _DeUsizeI32D = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _DeI32I32C = Int32 Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _DeI32I32D = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _DeI32I32I32C = Int32 Function(
    Pointer<Void>, Int32, Int32, Pointer<Int32>);
typedef _DeI32I32I32D = int Function(Pointer<Void>, int, int, Pointer<Int32>);
typedef _DeRotateByC = Int32 Function(
    Pointer<Void>, IntPtr, Int32, Pointer<Int32>);
typedef _DeRotateByD = int Function(Pointer<Void>, int, int, Pointer<Int32>);
typedef _DeGetBoxC = Int32 Function(Pointer<Void>, IntPtr, Pointer<Double>,
    Pointer<Double>, Pointer<Double>, Pointer<Double>, Pointer<Int32>);
typedef _DeGetBoxD = int Function(Pointer<Void>, int, Pointer<Double>,
    Pointer<Double>, Pointer<Double>, Pointer<Double>, Pointer<Int32>);
typedef _DeSetBoxC = Int32 Function(
    Pointer<Void>, IntPtr, Double, Double, Double, Double, Pointer<Int32>);
typedef _DeSetBoxD = int Function(
    Pointer<Void>, int, double, double, double, double, Pointer<Int32>);
typedef _DeEraseRegionsC = Int32 Function(
    Pointer<Void>, IntPtr, Pointer<Double>, IntPtr, Pointer<Int32>);
typedef _DeEraseRegionsD = int Function(
    Pointer<Void>, int, Pointer<Double>, int, Pointer<Int32>);
typedef _DeEraseRegionC = Int32 Function(
    Pointer<Void>, Int32, Float, Float, Float, Float, Pointer<Int32>);
typedef _DeEraseRegionD = int Function(
    Pointer<Void>, int, double, double, double, double, Pointer<Int32>);
typedef _DeIsMarkedC = Int32 Function(Pointer<Void>, IntPtr);
typedef _DeIsMarkedD = int Function(Pointer<Void>, int);
typedef _DeCropMarginsC = Int32 Function(
    Pointer<Void>, Float, Float, Float, Float, Pointer<Int32>);
typedef _DeCropMarginsD = int Function(
    Pointer<Void>, double, double, double, double, Pointer<Int32>);
typedef _DeMergeFromC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _DeMergeFromD = int Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _DeSaveEncC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _DeSaveEncD = int Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _DeSetFormC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _DeSetFormD = int Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _DeWarnCountC = Int32 Function(Pointer<Void>);
typedef _DeWarnCountD = int Function(Pointer<Void>);
typedef _DeWarnC = Pointer<Utf8> Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _DeWarnD = Pointer<Utf8> Function(Pointer<Void>, int, Pointer<Int32>);

// PDF creation builder. Three opaque handle types:
//   DocumentBuilder  — `FfiDocumentBuilder` from pdf_document_builder_create
//   PageBuilder      — `FfiPageBuilder` from pdf_document_builder_page/_letter/_a4
//   EmbeddedFont     — `EmbeddedFont` from pdf_embedded_font_from_file/_bytes
// int32 returns are status codes (0 = success); a non-zero return OR a non-zero
// error_code raises. byte returns use a `uintptr_t` out-len + free_bytes. String
// arrays are `const char* const*`; float/int32 arrays marshal from Dart lists.
//
// Ownership: pdf_document_builder_register_embedded_font CONSUMES the font on
// success — the EmbeddedFont wrapper nulls its handle so close()/finalizer won't
// double-free. pdf_page_builder_done CONSUMES the page; _free drops an
// uncommitted one. build()/save()/to_bytes_encrypted consume builder STATE but
// the handle must still be freed via pdf_document_builder_free.

// EmbeddedFont
typedef _EfFromFileC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _EfFromFileD = Pointer<Void> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _EfFromBytesC = Pointer<Void> Function(
    Pointer<Uint8>, IntPtr, Pointer<Utf8>, Pointer<Int32>);
typedef _EfFromBytesD = Pointer<Void> Function(
    Pointer<Uint8>, int, Pointer<Utf8>, Pointer<Int32>);
typedef _EfFreeC = Void Function(Pointer<Void>);
typedef _EfFreeD = void Function(Pointer<Void>);

// DocumentBuilder
typedef _DbCreateC = Pointer<Void> Function(Pointer<Int32>);
typedef _DbCreateD = Pointer<Void> Function(Pointer<Int32>);
typedef _DbFreeC = Void Function(Pointer<Void>);
typedef _DbFreeD = void Function(Pointer<Void>);
typedef _DbSetStrC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _DbSetStrD = int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _DbStatus0C = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _DbStatus0D = int Function(Pointer<Void>, Pointer<Int32>);
typedef _DbRoleMapC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _DbRoleMapD = int Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _DbRegisterFontC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Void>, Pointer<Int32>);
typedef _DbRegisterFontD = int Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Void>, Pointer<Int32>);
typedef _DbPageC = Pointer<Void> Function(Pointer<Void>, Pointer<Int32>);
typedef _DbPageD = Pointer<Void> Function(Pointer<Void>, Pointer<Int32>);
typedef _DbPageSizeC = Pointer<Void> Function(
    Pointer<Void>, Float, Float, Pointer<Int32>);
typedef _DbPageSizeD = Pointer<Void> Function(
    Pointer<Void>, double, double, Pointer<Int32>);
typedef _DbBuildC = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<IntPtr>, Pointer<Int32>);
typedef _DbBuildD = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<IntPtr>, Pointer<Int32>);
typedef _DbSaveC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _DbSaveD = int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _DbSaveEncC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _DbSaveEncD = int Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _DbToBytesEncC = Pointer<Uint8> Function(Pointer<Void>, Pointer<Utf8>,
    Pointer<Utf8>, Pointer<IntPtr>, Pointer<Int32>);
typedef _DbToBytesEncD = Pointer<Uint8> Function(Pointer<Void>, Pointer<Utf8>,
    Pointer<Utf8>, Pointer<IntPtr>, Pointer<Int32>);

// PageBuilder — grouped by C signature shape.
typedef _PbFreeC = Void Function(Pointer<Void>);
typedef _PbFreeD = void Function(Pointer<Void>);
typedef _PbStatus0C = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _PbStatus0D = int Function(Pointer<Void>, Pointer<Int32>);
typedef _PbStrC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _PbStrD = int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _PbFontC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Float, Pointer<Int32>);
typedef _PbFontD = int Function(
    Pointer<Void>, Pointer<Utf8>, double, Pointer<Int32>);
typedef _PbAtC = Int32 Function(Pointer<Void>, Float, Float, Pointer<Int32>);
typedef _PbAtD = int Function(Pointer<Void>, double, double, Pointer<Int32>);
typedef _PbHeadingC = Int32 Function(
    Pointer<Void>, Uint8, Pointer<Utf8>, Pointer<Int32>);
typedef _PbHeadingD = int Function(
    Pointer<Void>, int, Pointer<Utf8>, Pointer<Int32>);
typedef _PbSpaceC = Int32 Function(Pointer<Void>, Float, Pointer<Int32>);
typedef _PbSpaceD = int Function(Pointer<Void>, double, Pointer<Int32>);
typedef _PbLinkPageC = Int32 Function(Pointer<Void>, IntPtr, Pointer<Int32>);
typedef _PbLinkPageD = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _PbRgbC = Int32 Function(
    Pointer<Void>, Float, Float, Float, Pointer<Int32>);
typedef _PbRgbD = int Function(
    Pointer<Void>, double, double, double, Pointer<Int32>);
typedef _PbStickyAtC = Int32 Function(
    Pointer<Void>, Float, Float, Pointer<Utf8>, Pointer<Int32>);
typedef _PbStickyAtD = int Function(
    Pointer<Void>, double, double, Pointer<Utf8>, Pointer<Int32>);
typedef _PbFreetextC = Int32 Function(
    Pointer<Void>, Float, Float, Float, Float, Pointer<Utf8>, Pointer<Int32>);
typedef _PbFreetextD = int Function(Pointer<Void>, double, double, double,
    double, Pointer<Utf8>, Pointer<Int32>);
typedef _PbImageC = Int32 Function(Pointer<Void>, Pointer<Uint8>, IntPtr, Float,
    Float, Float, Float, Pointer<Int32>);
typedef _PbImageD = int Function(Pointer<Void>, Pointer<Uint8>, int, double,
    double, double, double, Pointer<Int32>);
typedef _PbImageAltC = Int32 Function(Pointer<Void>, Pointer<Uint8>, IntPtr,
    Float, Float, Float, Float, Pointer<Utf8>, Pointer<Int32>);
typedef _PbImageAltD = int Function(Pointer<Void>, Pointer<Uint8>, int, double,
    double, double, double, Pointer<Utf8>, Pointer<Int32>);
typedef _PbRectC = Int32 Function(
    Pointer<Void>, Float, Float, Float, Float, Pointer<Int32>);
typedef _PbRectD = int Function(
    Pointer<Void>, double, double, double, double, Pointer<Int32>);
typedef _PbFilledRectC = Int32 Function(Pointer<Void>, Float, Float, Float,
    Float, Float, Float, Float, Pointer<Int32>);
typedef _PbFilledRectD = int Function(Pointer<Void>, double, double, double,
    double, double, double, double, Pointer<Int32>);
typedef _PbLineC = Int32 Function(
    Pointer<Void>, Float, Float, Float, Float, Pointer<Int32>);
typedef _PbLineD = int Function(
    Pointer<Void>, double, double, double, double, Pointer<Int32>);
typedef _PbStrokeRectC = Int32 Function(Pointer<Void>, Float, Float, Float,
    Float, Float, Float, Float, Float, Pointer<Int32>);
typedef _PbStrokeRectD = int Function(Pointer<Void>, double, double, double,
    double, double, double, double, double, Pointer<Int32>);
typedef _PbStrokeLineC = Int32 Function(Pointer<Void>, Float, Float, Float,
    Float, Float, Float, Float, Float, Pointer<Int32>);
typedef _PbStrokeLineD = int Function(Pointer<Void>, double, double, double,
    double, double, double, double, double, Pointer<Int32>);
typedef _PbStrokeRectDashedC = Int32 Function(
    Pointer<Void>,
    Float,
    Float,
    Float,
    Float,
    Float,
    Float,
    Float,
    Float,
    Pointer<Float>,
    IntPtr,
    Float,
    Pointer<Int32>);
typedef _PbStrokeRectDashedD = int Function(
    Pointer<Void>,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    Pointer<Float>,
    int,
    double,
    Pointer<Int32>);
typedef _PbTextInRectC = Int32 Function(Pointer<Void>, Float, Float, Float,
    Float, Pointer<Utf8>, Int32, Pointer<Int32>);
typedef _PbTextInRectD = int Function(Pointer<Void>, double, double, double,
    double, Pointer<Utf8>, int, Pointer<Int32>);
typedef _PbFootnoteC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _PbFootnoteD = int Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _PbColumnsC = Int32 Function(
    Pointer<Void>, Uint32, Float, Pointer<Utf8>, Pointer<Int32>);
typedef _PbColumnsD = int Function(
    Pointer<Void>, int, double, Pointer<Utf8>, Pointer<Int32>);
typedef _PbInlineColorC = Int32 Function(
    Pointer<Void>, Float, Float, Float, Pointer<Utf8>, Pointer<Int32>);
typedef _PbInlineColorD = int Function(
    Pointer<Void>, double, double, double, Pointer<Utf8>, Pointer<Int32>);
typedef _PbBarcode1dC = Int32 Function(Pointer<Void>, Int32, Pointer<Utf8>,
    Float, Float, Float, Float, Pointer<Int32>);
typedef _PbBarcode1dD = int Function(Pointer<Void>, int, Pointer<Utf8>, double,
    double, double, double, Pointer<Int32>);
typedef _PbBarcodeQrC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Float, Float, Float, Pointer<Int32>);
typedef _PbBarcodeQrD = int Function(
    Pointer<Void>, Pointer<Utf8>, double, double, double, Pointer<Int32>);
typedef _PbTextFieldC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Float,
    Float, Float, Float, Pointer<Utf8>, Pointer<Int32>);
typedef _PbTextFieldD = int Function(Pointer<Void>, Pointer<Utf8>, double,
    double, double, double, Pointer<Utf8>, Pointer<Int32>);
typedef _PbCheckboxC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Float,
    Float, Float, Float, Int32, Pointer<Int32>);
typedef _PbCheckboxD = int Function(Pointer<Void>, Pointer<Utf8>, double,
    double, double, double, int, Pointer<Int32>);
typedef _PbSigFieldC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Float, Float, Float, Float, Pointer<Int32>);
typedef _PbSigFieldD = int Function(Pointer<Void>, Pointer<Utf8>, double,
    double, double, double, Pointer<Int32>);
typedef _PbPushButtonC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Float,
    Float, Float, Float, Pointer<Utf8>, Pointer<Int32>);
typedef _PbPushButtonD = int Function(Pointer<Void>, Pointer<Utf8>, double,
    double, double, double, Pointer<Utf8>, Pointer<Int32>);
typedef _PbComboBoxC = Int32 Function(
    Pointer<Void>,
    Pointer<Utf8>,
    Float,
    Float,
    Float,
    Float,
    Pointer<Pointer<Utf8>>,
    IntPtr,
    Pointer<Utf8>,
    Pointer<Int32>);
typedef _PbComboBoxD = int Function(
    Pointer<Void>,
    Pointer<Utf8>,
    double,
    double,
    double,
    double,
    Pointer<Pointer<Utf8>>,
    int,
    Pointer<Utf8>,
    Pointer<Int32>);
typedef _PbRadioGroupC = Int32 Function(
    Pointer<Void>,
    Pointer<Utf8>,
    Pointer<Pointer<Utf8>>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    IntPtr,
    Pointer<Utf8>,
    Pointer<Int32>);
typedef _PbRadioGroupD = int Function(
    Pointer<Void>,
    Pointer<Utf8>,
    Pointer<Pointer<Utf8>>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    int,
    Pointer<Utf8>,
    Pointer<Int32>);
typedef _PbStampC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _PbStampD = int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _PbTableC = Int32 Function(Pointer<Void>, IntPtr, Pointer<Float>,
    Pointer<Int32>, IntPtr, Pointer<Pointer<Utf8>>, Int32, Pointer<Int32>);
typedef _PbTableD = int Function(Pointer<Void>, int, Pointer<Float>,
    Pointer<Int32>, int, Pointer<Pointer<Utf8>>, int, Pointer<Int32>);

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
            'pdf_oxide_search_result_free'),
        // page rendering (Phase 3)
        renderPage =
            lib.lookupFunction<_RenderPageC, _RenderPageD>('pdf_render_page'),
        renderPageZoom = lib
            .lookupFunction<_RenderZoomC, _RenderZoomD>('pdf_render_page_zoom'),
        renderPageThumbnail = lib.lookupFunction<_RenderThumbC, _RenderThumbD>(
            'pdf_render_page_thumbnail'),
        renderedImageWidth = lib.lookupFunction<_RenderedDimC, _RenderedDimD>(
            'pdf_get_rendered_image_width'),
        renderedImageHeight = lib.lookupFunction<_RenderedDimC, _RenderedDimD>(
            'pdf_get_rendered_image_height'),
        renderedImageData = lib.lookupFunction<_RenderedDataC, _RenderedDataD>(
            'pdf_get_rendered_image_data'),
        renderedImageSave = lib.lookupFunction<_RenderedSaveC, _RenderedSaveD>(
            'pdf_save_rendered_image'),
        renderedImageFree = lib.lookupFunction<_RenderedFreeC, _RenderedFreeD>(
            'pdf_rendered_image_free'),
        // document editing
        deOpen = lib.lookupFunction<_DeOpenC, _DeOpenD>('document_editor_open'),
        deOpenBytes = lib.lookupFunction<_DeOpenBytesC, _DeOpenBytesD>(
            'document_editor_open_from_bytes'),
        deFree = lib.lookupFunction<_DeFreeC, _DeFreeD>('document_editor_free'),
        deIsModified = lib
            .lookupFunction<_DeBoolC, _DeBoolD>('document_editor_is_modified'),
        deGetSourcePath = lib.lookupFunction<_DeStrC, _DeStrD>(
            'document_editor_get_source_path'),
        deGetVersion = lib.lookupFunction<_DeVersionC, _DeVersionD>(
            'document_editor_get_version'),
        dePageCount = lib
            .lookupFunction<_DeI32C, _DeI32D>('document_editor_get_page_count'),
        deGetProducer = lib
            .lookupFunction<_DeStrC, _DeStrD>('document_editor_get_producer'),
        deSetProducer = lib.lookupFunction<_DeSetStrC, _DeSetStrD>(
            'document_editor_set_producer'),
        deGetCreationDate = lib.lookupFunction<_DeStrC, _DeStrD>(
            'document_editor_get_creation_date'),
        deSetCreationDate = lib.lookupFunction<_DeSetStrC, _DeSetStrD>(
            'document_editor_set_creation_date'),
        deSave = lib.lookupFunction<_DeSaveC, _DeSaveD>('document_editor_save'),
        deSaveToBytes = lib.lookupFunction<_DeSaveBytesC, _DeSaveBytesD>(
            'document_editor_save_to_bytes'),
        deSaveToBytesWithOptions =
            lib.lookupFunction<_DeSaveBytesOptsC, _DeSaveBytesOptsD>(
                'document_editor_save_to_bytes_with_options'),
        deExtractPagesToBytes =
            lib.lookupFunction<_DeExtractPagesC, _DeExtractPagesD>(
                'document_editor_extract_pages_to_bytes'),
        deConvertToPdfA = lib.lookupFunction<_DeConvertPdfAC, _DeConvertPdfAD>(
            'document_editor_convert_to_pdf_a'),
        deSaveEncryptedToBytes =
            lib.lookupFunction<_DeSaveEncBytesC, _DeSaveEncBytesD>(
                'document_editor_save_encrypted_to_bytes'),
        deMergeFromBytes = lib.lookupFunction<_DeMergeBytesC, _DeMergeBytesD>(
            'document_editor_merge_from_bytes'),
        deEmbedFile = lib.lookupFunction<_DeEmbedFileC, _DeEmbedFileD>(
            'document_editor_embed_file'),
        deApplyPageRedactions = lib.lookupFunction<_DeUsizeI32C, _DeUsizeI32D>(
            'document_editor_apply_page_redactions'),
        deApplyAllRedactions = lib.lookupFunction<_DeI32C, _DeI32D>(
            'document_editor_apply_all_redactions'),
        deRotateAllPages = lib.lookupFunction<_DeI32I32C, _DeI32I32D>(
            'document_editor_rotate_all_pages'),
        deRotatePageBy = lib.lookupFunction<_DeRotateByC, _DeRotateByD>(
            'document_editor_rotate_page_by'),
        deGetPageMediaBox = lib.lookupFunction<_DeGetBoxC, _DeGetBoxD>(
            'document_editor_get_page_media_box'),
        deSetPageMediaBox = lib.lookupFunction<_DeSetBoxC, _DeSetBoxD>(
            'document_editor_set_page_media_box'),
        deGetPageCropBox = lib.lookupFunction<_DeGetBoxC, _DeGetBoxD>(
            'document_editor_get_page_crop_box'),
        deSetPageCropBox = lib.lookupFunction<_DeSetBoxC, _DeSetBoxD>(
            'document_editor_set_page_crop_box'),
        deEraseRegions = lib.lookupFunction<_DeEraseRegionsC, _DeEraseRegionsD>(
            'document_editor_erase_regions'),
        deClearEraseRegions = lib.lookupFunction<_DeUsizeI32C, _DeUsizeI32D>(
            'document_editor_clear_erase_regions'),
        deIsPageMarkedForFlatten =
            lib.lookupFunction<_DeIsMarkedC, _DeIsMarkedD>(
                'document_editor_is_page_marked_for_flatten'),
        deUnmarkPageForFlatten = lib.lookupFunction<_DeUsizeI32C, _DeUsizeI32D>(
            'document_editor_unmark_page_for_flatten'),
        deIsPageMarkedForRedaction =
            lib.lookupFunction<_DeIsMarkedC, _DeIsMarkedD>(
                'document_editor_is_page_marked_for_redaction'),
        deUnmarkPageForRedaction =
            lib.lookupFunction<_DeUsizeI32C, _DeUsizeI32D>(
                'document_editor_unmark_page_for_redaction'),
        deDeletePage = lib.lookupFunction<_DeI32I32C, _DeI32I32D>(
            'document_editor_delete_page'),
        deMovePage = lib.lookupFunction<_DeI32I32I32C, _DeI32I32I32D>(
            'document_editor_move_page'),
        deGetPageRotation = lib.lookupFunction<_DeI32I32C, _DeI32I32D>(
            'document_editor_get_page_rotation'),
        deSetPageRotation = lib.lookupFunction<_DeI32I32I32C, _DeI32I32I32D>(
            'document_editor_set_page_rotation'),
        deEraseRegion = lib.lookupFunction<_DeEraseRegionC, _DeEraseRegionD>(
            'document_editor_erase_region'),
        deFlattenAnnotations = lib.lookupFunction<_DeI32I32C, _DeI32I32D>(
            'document_editor_flatten_annotations'),
        deFlattenAllAnnotations = lib.lookupFunction<_DeI32C, _DeI32D>(
            'document_editor_flatten_all_annotations'),
        deCropMargins = lib.lookupFunction<_DeCropMarginsC, _DeCropMarginsD>(
            'document_editor_crop_margins'),
        deMergeFrom = lib.lookupFunction<_DeMergeFromC, _DeMergeFromD>(
            'document_editor_merge_from'),
        deSaveEncrypted = lib.lookupFunction<_DeSaveEncC, _DeSaveEncD>(
            'document_editor_save_encrypted'),
        deSetFormFieldValue = lib.lookupFunction<_DeSetFormC, _DeSetFormD>(
            'document_editor_set_form_field_value'),
        deFlattenForms = lib
            .lookupFunction<_DeI32C, _DeI32D>('document_editor_flatten_forms'),
        deFlattenFormsOnPage = lib.lookupFunction<_DeI32I32C, _DeI32I32D>(
            'document_editor_flatten_forms_on_page'),
        deFlattenWarningsCount =
            lib.lookupFunction<_DeWarnCountC, _DeWarnCountD>(
                'document_editor_flatten_warnings_count'),
        deFlattenWarning = lib.lookupFunction<_DeWarnC, _DeWarnD>(
            'document_editor_flatten_warning'),
        // PDF creation — EmbeddedFont
        efFromFile = lib.lookupFunction<_EfFromFileC, _EfFromFileD>(
            'pdf_embedded_font_from_file'),
        efFromBytes = lib.lookupFunction<_EfFromBytesC, _EfFromBytesD>(
            'pdf_embedded_font_from_bytes'),
        efFree =
            lib.lookupFunction<_EfFreeC, _EfFreeD>('pdf_embedded_font_free'),
        // PDF creation — DocumentBuilder
        dbCreate = lib.lookupFunction<_DbCreateC, _DbCreateD>(
            'pdf_document_builder_create'),
        dbFree =
            lib.lookupFunction<_DbFreeC, _DbFreeD>('pdf_document_builder_free'),
        dbSetTitle = lib.lookupFunction<_DbSetStrC, _DbSetStrD>(
            'pdf_document_builder_set_title'),
        dbSetAuthor = lib.lookupFunction<_DbSetStrC, _DbSetStrD>(
            'pdf_document_builder_set_author'),
        dbSetSubject = lib.lookupFunction<_DbSetStrC, _DbSetStrD>(
            'pdf_document_builder_set_subject'),
        dbSetKeywords = lib.lookupFunction<_DbSetStrC, _DbSetStrD>(
            'pdf_document_builder_set_keywords'),
        dbSetCreator = lib.lookupFunction<_DbSetStrC, _DbSetStrD>(
            'pdf_document_builder_set_creator'),
        dbOnOpen = lib.lookupFunction<_DbSetStrC, _DbSetStrD>(
            'pdf_document_builder_on_open'),
        dbTaggedPdfUa1 = lib.lookupFunction<_DbStatus0C, _DbStatus0D>(
            'pdf_document_builder_tagged_pdf_ua1'),
        dbLanguage = lib.lookupFunction<_DbSetStrC, _DbSetStrD>(
            'pdf_document_builder_language'),
        dbRoleMap = lib.lookupFunction<_DbRoleMapC, _DbRoleMapD>(
            'pdf_document_builder_role_map'),
        dbRegisterFont = lib.lookupFunction<_DbRegisterFontC, _DbRegisterFontD>(
            'pdf_document_builder_register_embedded_font'),
        dbA4Page = lib
            .lookupFunction<_DbPageC, _DbPageD>('pdf_document_builder_a4_page'),
        dbLetterPage = lib.lookupFunction<_DbPageC, _DbPageD>(
            'pdf_document_builder_letter_page'),
        dbPage = lib.lookupFunction<_DbPageSizeC, _DbPageSizeD>(
            'pdf_document_builder_page'),
        dbBuild = lib
            .lookupFunction<_DbBuildC, _DbBuildD>('pdf_document_builder_build'),
        dbSave =
            lib.lookupFunction<_DbSaveC, _DbSaveD>('pdf_document_builder_save'),
        dbSaveEncrypted = lib.lookupFunction<_DbSaveEncC, _DbSaveEncD>(
            'pdf_document_builder_save_encrypted'),
        dbToBytesEncrypted = lib.lookupFunction<_DbToBytesEncC, _DbToBytesEncD>(
            'pdf_document_builder_to_bytes_encrypted'),
        // PDF creation — PageBuilder
        pbFree =
            lib.lookupFunction<_PbFreeC, _PbFreeD>('pdf_page_builder_free'),
        pbDone = lib
            .lookupFunction<_PbStatus0C, _PbStatus0D>('pdf_page_builder_done'),
        pbFont =
            lib.lookupFunction<_PbFontC, _PbFontD>('pdf_page_builder_font'),
        pbAt = lib.lookupFunction<_PbAtC, _PbAtD>('pdf_page_builder_at'),
        pbText = lib.lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_text'),
        pbHeading = lib.lookupFunction<_PbHeadingC, _PbHeadingD>(
            'pdf_page_builder_heading'),
        pbParagraph =
            lib.lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_paragraph'),
        pbSpace =
            lib.lookupFunction<_PbSpaceC, _PbSpaceD>('pdf_page_builder_space'),
        pbHorizontalRule = lib.lookupFunction<_PbStatus0C, _PbStatus0D>(
            'pdf_page_builder_horizontal_rule'),
        pbLinkUrl =
            lib.lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_link_url'),
        pbLinkPage = lib.lookupFunction<_PbLinkPageC, _PbLinkPageD>(
            'pdf_page_builder_link_page'),
        pbLinkNamed =
            lib.lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_link_named'),
        pbLinkJavascript = lib.lookupFunction<_PbStrC, _PbStrD>(
            'pdf_page_builder_link_javascript'),
        pbOnOpen =
            lib.lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_on_open'),
        pbOnClose =
            lib.lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_on_close'),
        pbFieldKeystroke = lib.lookupFunction<_PbStrC, _PbStrD>(
            'pdf_page_builder_field_keystroke'),
        pbFieldFormat = lib
            .lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_field_format'),
        pbFieldValidate = lib.lookupFunction<_PbStrC, _PbStrD>(
            'pdf_page_builder_field_validate'),
        pbFieldCalculate = lib.lookupFunction<_PbStrC, _PbStrD>(
            'pdf_page_builder_field_calculate'),
        pbHighlight =
            lib.lookupFunction<_PbRgbC, _PbRgbD>('pdf_page_builder_highlight'),
        pbUnderline =
            lib.lookupFunction<_PbRgbC, _PbRgbD>('pdf_page_builder_underline'),
        pbStrikeout =
            lib.lookupFunction<_PbRgbC, _PbRgbD>('pdf_page_builder_strikeout'),
        pbSquiggly =
            lib.lookupFunction<_PbRgbC, _PbRgbD>('pdf_page_builder_squiggly'),
        pbStickyNote = lib
            .lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_sticky_note'),
        pbStickyNoteAt = lib.lookupFunction<_PbStickyAtC, _PbStickyAtD>(
            'pdf_page_builder_sticky_note_at'),
        pbWatermark =
            lib.lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_watermark'),
        pbWatermarkConfidential = lib.lookupFunction<_PbStatus0C, _PbStatus0D>(
            'pdf_page_builder_watermark_confidential'),
        pbWatermarkDraft = lib.lookupFunction<_PbStatus0C, _PbStatus0D>(
            'pdf_page_builder_watermark_draft'),
        pbStamp =
            lib.lookupFunction<_PbStampC, _PbStampD>('pdf_page_builder_stamp'),
        pbFreetext = lib.lookupFunction<_PbFreetextC, _PbFreetextD>(
            'pdf_page_builder_freetext'),
        pbTextField = lib.lookupFunction<_PbTextFieldC, _PbTextFieldD>(
            'pdf_page_builder_text_field'),
        pbCheckbox = lib.lookupFunction<_PbCheckboxC, _PbCheckboxD>(
            'pdf_page_builder_checkbox'),
        pbComboBox = lib.lookupFunction<_PbComboBoxC, _PbComboBoxD>(
            'pdf_page_builder_combo_box'),
        pbRadioGroup = lib.lookupFunction<_PbRadioGroupC, _PbRadioGroupD>(
            'pdf_page_builder_radio_group'),
        pbPushButton = lib.lookupFunction<_PbPushButtonC, _PbPushButtonD>(
            'pdf_page_builder_push_button'),
        pbSignatureField = lib.lookupFunction<_PbSigFieldC, _PbSigFieldD>(
            'pdf_page_builder_signature_field'),
        pbFootnote = lib.lookupFunction<_PbFootnoteC, _PbFootnoteD>(
            'pdf_page_builder_footnote'),
        pbColumns = lib.lookupFunction<_PbColumnsC, _PbColumnsD>(
            'pdf_page_builder_columns'),
        pbInline =
            lib.lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_inline'),
        pbInlineBold = lib
            .lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_inline_bold'),
        pbInlineItalic = lib
            .lookupFunction<_PbStrC, _PbStrD>('pdf_page_builder_inline_italic'),
        pbInlineColor = lib.lookupFunction<_PbInlineColorC, _PbInlineColorD>(
            'pdf_page_builder_inline_color'),
        pbNewline = lib.lookupFunction<_PbStatus0C, _PbStatus0D>(
            'pdf_page_builder_newline'),
        pbBarcode1d = lib.lookupFunction<_PbBarcode1dC, _PbBarcode1dD>(
            'pdf_page_builder_barcode_1d'),
        pbBarcodeQr = lib.lookupFunction<_PbBarcodeQrC, _PbBarcodeQrD>(
            'pdf_page_builder_barcode_qr'),
        pbImage =
            lib.lookupFunction<_PbImageC, _PbImageD>('pdf_page_builder_image'),
        pbImageWithAlt = lib.lookupFunction<_PbImageAltC, _PbImageAltD>(
            'pdf_page_builder_image_with_alt'),
        pbImageArtifact = lib.lookupFunction<_PbImageC, _PbImageD>(
            'pdf_page_builder_image_artifact'),
        pbRect =
            lib.lookupFunction<_PbRectC, _PbRectD>('pdf_page_builder_rect'),
        pbFilledRect = lib.lookupFunction<_PbFilledRectC, _PbFilledRectD>(
            'pdf_page_builder_filled_rect'),
        pbLine =
            lib.lookupFunction<_PbLineC, _PbLineD>('pdf_page_builder_line'),
        pbStrokeRect = lib.lookupFunction<_PbStrokeRectC, _PbStrokeRectD>(
            'pdf_page_builder_stroke_rect'),
        pbStrokeLine = lib.lookupFunction<_PbStrokeLineC, _PbStrokeLineD>(
            'pdf_page_builder_stroke_line'),
        pbStrokeRectDashed =
            lib.lookupFunction<_PbStrokeRectDashedC, _PbStrokeRectDashedD>(
                'pdf_page_builder_stroke_rect_dashed'),
        pbStrokeLineDashed =
            lib.lookupFunction<_PbStrokeRectDashedC, _PbStrokeRectDashedD>(
                'pdf_page_builder_stroke_line_dashed'),
        pbTextInRect = lib.lookupFunction<_PbTextInRectC, _PbTextInRectD>(
            'pdf_page_builder_text_in_rect'),
        pbNewPageSameSize = lib.lookupFunction<_PbStatus0C, _PbStatus0D>(
            'pdf_page_builder_new_page_same_size'),
        pbTable =
            lib.lookupFunction<_PbTableC, _PbTableD>('pdf_page_builder_table'),
        // ── Phase 6: signatures / PKI / timestamps / TSA / DSS / validation ──
        certLoadFromBytes =
            lib.lookupFunction<_CertLoadBytesC, _CertLoadBytesD>(
                'pdf_certificate_load_from_bytes'),
        certLoadFromPem = lib.lookupFunction<_CertLoadPemC, _CertLoadPemD>(
            'pdf_certificate_load_from_pem'),
        certGetSubject = lib.lookupFunction<_CertStrC, _CertStrD>(
            'pdf_certificate_get_subject'),
        certGetIssuer = lib
            .lookupFunction<_CertStrC, _CertStrD>('pdf_certificate_get_issuer'),
        certGetSerial = lib
            .lookupFunction<_CertStrC, _CertStrD>('pdf_certificate_get_serial'),
        certGetValidity = lib.lookupFunction<_CertValidityC, _CertValidityD>(
            'pdf_certificate_get_validity'),
        certIsValid = lib
            .lookupFunction<_CertI32C, _CertI32D>('pdf_certificate_is_valid'),
        certFree =
            lib.lookupFunction<_PtrFreeC, _PtrFreeD>('pdf_certificate_free'),
        signBytes =
            lib.lookupFunction<_SignBytesC, _SignBytesD>('pdf_sign_bytes'),
        signBytesPades = lib
            .lookupFunction<_SignPadesC, _SignPadesD>('pdf_sign_bytes_pades'),
        signBytesPadesOpts =
            lib.lookupFunction<_SignPadesOptsC, _SignPadesOptsD>(
                'pdf_sign_bytes_pades_opts'),
        sigGetSignerName = lib.lookupFunction<_SigStrC, _SigStrD>(
            'pdf_signature_get_signer_name'),
        sigGetSigningReason = lib.lookupFunction<_SigStrC, _SigStrD>(
            'pdf_signature_get_signing_reason'),
        sigGetSigningLocation = lib.lookupFunction<_SigStrC, _SigStrD>(
            'pdf_signature_get_signing_location'),
        sigGetSigningTime = lib.lookupFunction<_SigI64C, _SigI64D>(
            'pdf_signature_get_signing_time'),
        sigGetCertificate = lib.lookupFunction<_SigPtrC, _SigPtrD>(
            'pdf_signature_get_certificate'),
        sigGetPadesLevel = lib.lookupFunction<_SigI32C, _SigI32D>(
            'pdf_signature_get_pades_level'),
        sigHasTimestamp = lib.lookupFunction<_SigBoolC, _SigBoolD>(
            'pdf_signature_has_timestamp'),
        sigGetTimestamp = lib
            .lookupFunction<_SigPtrC, _SigPtrD>('pdf_signature_get_timestamp'),
        sigAddTimestamp = lib.lookupFunction<_SigAddTsC, _SigAddTsD>(
            'pdf_signature_add_timestamp'),
        sigVerify =
            lib.lookupFunction<_SigI32C, _SigI32D>('pdf_signature_verify'),
        sigVerifyDetached =
            lib.lookupFunction<_SigVerifyDetachedC, _SigVerifyDetachedD>(
                'pdf_signature_verify_detached'),
        sigFree =
            lib.lookupFunction<_PtrFreeC, _PtrFreeD>('pdf_signature_free'),
        tsParse =
            lib.lookupFunction<_TsParseC, _TsParseD>('pdf_timestamp_parse'),
        tsGetToken = lib.lookupFunction<_TsConstBytesC, _TsConstBytesD>(
            'pdf_timestamp_get_token'),
        tsGetMessageImprint =
            lib.lookupFunction<_TsConstBytesC, _TsConstBytesD>(
                'pdf_timestamp_get_message_imprint'),
        tsGetTime =
            lib.lookupFunction<_TsI64C, _TsI64D>('pdf_timestamp_get_time'),
        tsGetSerial =
            lib.lookupFunction<_TsStrC, _TsStrD>('pdf_timestamp_get_serial'),
        tsGetTsaName =
            lib.lookupFunction<_TsStrC, _TsStrD>('pdf_timestamp_get_tsa_name'),
        tsGetPolicyOid = lib
            .lookupFunction<_TsStrC, _TsStrD>('pdf_timestamp_get_policy_oid'),
        tsGetHashAlgorithm = lib.lookupFunction<_TsI32C, _TsI32D>(
            'pdf_timestamp_get_hash_algorithm'),
        tsVerify =
            lib.lookupFunction<_TsBoolC, _TsBoolD>('pdf_timestamp_verify'),
        tsFree = lib.lookupFunction<_PtrFreeC, _PtrFreeD>('pdf_timestamp_free'),
        tsaClientCreate = lib
            .lookupFunction<_TsaCreateC, _TsaCreateD>('pdf_tsa_client_create'),
        tsaRequestTimestamp =
            lib.lookupFunction<_TsaReqC, _TsaReqD>('pdf_tsa_request_timestamp'),
        tsaRequestTimestampHash =
            lib.lookupFunction<_TsaReqHashC, _TsaReqHashD>(
                'pdf_tsa_request_timestamp_hash'),
        tsaClientFree =
            lib.lookupFunction<_PtrFreeC, _PtrFreeD>('pdf_tsa_client_free'),
        dssCertCount =
            lib.lookupFunction<_DssCountC, _DssCountD>('pdf_dss_cert_count'),
        dssCrlCount =
            lib.lookupFunction<_DssCountC, _DssCountD>('pdf_dss_crl_count'),
        dssOcspCount =
            lib.lookupFunction<_DssCountC, _DssCountD>('pdf_dss_ocsp_count'),
        dssVriCount =
            lib.lookupFunction<_DssCountC, _DssCountD>('pdf_dss_vri_count'),
        dssGetCert = lib.lookupFunction<_DssGetC, _DssGetD>('pdf_dss_get_cert'),
        dssGetCrl = lib.lookupFunction<_DssGetC, _DssGetD>('pdf_dss_get_crl'),
        dssGetOcsp = lib.lookupFunction<_DssGetC, _DssGetD>('pdf_dss_get_ocsp'),
        dssFree = lib.lookupFunction<_PtrFreeC, _PtrFreeD>('pdf_dss_free'),
        validatePdfALevel = lib
            .lookupFunction<_ValidateC, _ValidateD>('pdf_validate_pdf_a_level'),
        validatePdfUa =
            lib.lookupFunction<_ValidateC, _ValidateD>('pdf_validate_pdf_ua'),
        validatePdfXLevel = lib
            .lookupFunction<_ValidateC, _ValidateD>('pdf_validate_pdf_x_level'),
        pdfAIsCompliant =
            lib.lookupFunction<_ValBoolC, _ValBoolD>('pdf_pdf_a_is_compliant'),
        pdfAErrorCount =
            lib.lookupFunction<_ValCountC, _ValCountD>('pdf_pdf_a_error_count'),
        pdfAWarningCount = lib
            .lookupFunction<_ValCountC, _ValCountD>('pdf_pdf_a_warning_count'),
        pdfAGetError =
            lib.lookupFunction<_ValGetC, _ValGetD>('pdf_pdf_a_get_error'),
        pdfAResultsFree =
            lib.lookupFunction<_ValFreeC, _ValFreeD>('pdf_pdf_a_results_free'),
        pdfUaIsAccessible = lib
            .lookupFunction<_ValBoolC, _ValBoolD>('pdf_pdf_ua_is_accessible'),
        pdfUaErrorCount = lib
            .lookupFunction<_ValCountC, _ValCountD>('pdf_pdf_ua_error_count'),
        pdfUaWarningCount = lib
            .lookupFunction<_ValCountC, _ValCountD>('pdf_pdf_ua_warning_count'),
        pdfUaGetError =
            lib.lookupFunction<_ValGetC, _ValGetD>('pdf_pdf_ua_get_error'),
        pdfUaGetWarning =
            lib.lookupFunction<_ValGetC, _ValGetD>('pdf_pdf_ua_get_warning'),
        pdfUaGetStats =
            lib.lookupFunction<_UaStatsC, _UaStatsD>('pdf_pdf_ua_get_stats'),
        pdfUaResultsFree =
            lib.lookupFunction<_ValFreeC, _ValFreeD>('pdf_pdf_ua_results_free'),
        pdfXIsCompliant =
            lib.lookupFunction<_ValBoolC, _ValBoolD>('pdf_pdf_x_is_compliant'),
        pdfXErrorCount =
            lib.lookupFunction<_ValCountC, _ValCountD>('pdf_pdf_x_error_count'),
        pdfXGetError =
            lib.lookupFunction<_ValGetC, _ValGetD>('pdf_pdf_x_get_error'),
        pdfXResultsFree =
            lib.lookupFunction<_ValFreeC, _ValFreeD>('pdf_pdf_x_results_free'),
        setLogLevel =
            lib.lookupFunction<_SetLogC, _SetLogD>('pdf_oxide_set_log_level'),
        getLogLevel =
            lib.lookupFunction<_GetLogC, _GetLogD>('pdf_oxide_get_log_level'),
        // ── Phase 7: barcodes / OCR / render variants / redaction / from_* ──
        generateQrCode =
            lib.lookupFunction<_GenQrC, _GenQrD>('pdf_generate_qr_code'),
        generateBarcode = lib
            .lookupFunction<_GenBarcodeC, _GenBarcodeD>('pdf_generate_barcode'),
        barcodeGetData =
            lib.lookupFunction<_BcStrC, _BcStrD>('pdf_barcode_get_data'),
        barcodeGetFormat =
            lib.lookupFunction<_BcI32C, _BcI32D>('pdf_barcode_get_format'),
        barcodeGetConfidence =
            lib.lookupFunction<_BcF32C, _BcF32D>('pdf_barcode_get_confidence'),
        barcodeGetImagePng =
            lib.lookupFunction<_BcPngC, _BcPngD>('pdf_barcode_get_image_png'),
        barcodeGetSvg =
            lib.lookupFunction<_BcSvgC, _BcSvgD>('pdf_barcode_get_svg'),
        barcodeFree =
            lib.lookupFunction<_PtrFreeC, _PtrFreeD>('pdf_barcode_free'),
        addBarcodeToPage = lib.lookupFunction<_AddBarcodeC, _AddBarcodeD>(
            'pdf_add_barcode_to_page'),
        ocrEngineCreate = lib
            .lookupFunction<_OcrCreateC, _OcrCreateD>('pdf_ocr_engine_create'),
        ocrEngineFree =
            lib.lookupFunction<_PtrFreeC, _PtrFreeD>('pdf_ocr_engine_free'),
        ocrPageNeedsOcr = lib
            .lookupFunction<_OcrNeedsC, _OcrNeedsD>('pdf_ocr_page_needs_ocr'),
        ocrExtractText = lib
            .lookupFunction<_OcrExtractC, _OcrExtractD>('pdf_ocr_extract_text'),
        renderPageWithOptions = lib.lookupFunction<_RenderOptsC, _RenderOptsD>(
            'pdf_render_page_with_options'),
        renderPageWithOptionsEx =
            lib.lookupFunction<_RenderOptsExC, _RenderOptsExD>(
                'pdf_render_page_with_options_ex'),
        renderPageRegion = lib.lookupFunction<_RenderRegionC, _RenderRegionD>(
            'pdf_render_page_region'),
        renderPageFit =
            lib.lookupFunction<_RenderFitC, _RenderFitD>('pdf_render_page_fit'),
        renderPageRaw =
            lib.lookupFunction<_RenderRawC, _RenderRawD>('pdf_render_page_raw'),
        createRenderer = lib.lookupFunction<_CreateRendererC, _CreateRendererD>(
            'pdf_create_renderer'),
        rendererFree =
            lib.lookupFunction<_PtrFreeC, _PtrFreeD>('pdf_renderer_free'),
        estimateRenderTime =
            lib.lookupFunction<_EstimateRenderC, _EstimateRenderD>(
                'pdf_estimate_render_time'),
        redactionAdd = lib.lookupFunction<_RedactionAddC, _RedactionAddD>(
            'pdf_redaction_add'),
        redactionCount = lib.lookupFunction<_RedactionCountC, _RedactionCountD>(
            'pdf_redaction_count'),
        redactionApply = lib.lookupFunction<_RedactionApplyC, _RedactionApplyD>(
            'pdf_redaction_apply'),
        redactionScrubMetadata =
            lib.lookupFunction<_RedactionScrubC, _RedactionScrubD>(
                'pdf_redaction_scrub_metadata'),
        fromImage =
            lib.lookupFunction<_FromImageC, _FromImageD>('pdf_from_image'),
        fromImageBytes = lib.lookupFunction<_FromImageBytesC, _FromImageBytesD>(
            'pdf_from_image_bytes'),
        fromHtmlCss = lib
            .lookupFunction<_FromHtmlCssC, _FromHtmlCssD>('pdf_from_html_css'),
        fromHtmlCssWithFonts =
            lib.lookupFunction<_FromHtmlCssFontsC, _FromHtmlCssFontsD>(
                'pdf_from_html_css_with_fonts'),
        merge = lib.lookupFunction<_MergeC, _MergeD>('pdf_merge'),
        pageGetWidth =
            lib.lookupFunction<_PageF32C, _PageF32D>('pdf_page_get_width'),
        pageGetHeight =
            lib.lookupFunction<_PageF32C, _PageF32D>('pdf_page_get_height'),
        pageGetRotation =
            lib.lookupFunction<_PageI32C, _PageI32D>('pdf_page_get_rotation'),
        pageGetElements = lib.lookupFunction<_PageElementsC, _PageElementsD>(
            'pdf_page_get_elements'),
        elementCount = lib.lookupFunction<_ElemCountC, _ElemCountD>(
            'pdf_oxide_element_count'),
        elementGetType = lib
            .lookupFunction<_ListStrC, _ListStrD>('pdf_oxide_element_get_type'),
        elementGetText = lib
            .lookupFunction<_ListStrC, _ListStrD>('pdf_oxide_element_get_text'),
        elementGetRect = lib.lookupFunction<_ElemRectC, _ElemRectD>(
            'pdf_oxide_element_get_rect'),
        elementsFree = lib
            .lookupFunction<_ListFreeC, _ListFreeD>('pdf_oxide_elements_free'),
        elementsToJson = lib.lookupFunction<_ElemJsonC, _ElemJsonD>(
            'pdf_oxide_elements_to_json'),
        addTimestamp = lib.lookupFunction<_AddTimestampC, _AddTimestampD>(
            'pdf_add_timestamp'),
        // ── Phase 8 ───────────────────────────────────────────────────────────
        // office I/O
        openFromDocx = lib.lookupFunction<_OpenBytesC, _OpenBytesD>(
            'pdf_document_open_from_docx_bytes'),
        openFromPptx = lib.lookupFunction<_OpenBytesC, _OpenBytesD>(
            'pdf_document_open_from_pptx_bytes'),
        openFromXlsx = lib.lookupFunction<_OpenBytesC, _OpenBytesD>(
            'pdf_document_open_from_xlsx_bytes'),
        toDocx =
            lib.lookupFunction<_DocBytesC, _DocBytesD>('pdf_document_to_docx'),
        toPptx =
            lib.lookupFunction<_DocBytesC, _DocBytesD>('pdf_document_to_pptx'),
        toXlsx =
            lib.lookupFunction<_DocBytesC, _DocBytesD>('pdf_document_to_xlsx'),
        // in-rect extractors
        extractTextInRect = lib.lookupFunction<_RectStrC, _RectStrD>(
            'pdf_document_extract_text_in_rect'),
        extractWordsInRect = lib.lookupFunction<_RectListC, _RectListD>(
            'pdf_document_extract_words_in_rect'),
        extractLinesInRect = lib.lookupFunction<_RectListC, _RectListD>(
            'pdf_document_extract_lines_in_rect'),
        extractTablesInRect = lib.lookupFunction<_RectListC, _RectListD>(
            'pdf_document_extract_tables_in_rect'),
        extractImagesInRect = lib.lookupFunction<_RectListC, _RectListD>(
            'pdf_document_extract_images_in_rect'),
        // auto / classify
        extractTextAuto = lib
            .lookupFunction<_TextC, _TextD>('pdf_document_extract_text_auto'),
        extractAllText = lib.lookupFunction<_TextAllC, _TextAllD>(
            'pdf_document_extract_all_text'),
        extractPageAuto = lib.lookupFunction<_PageAutoC, _PageAutoD>(
            'pdf_document_extract_page_auto'),
        classifyPage =
            lib.lookupFunction<_TextC, _TextD>('pdf_document_classify_page'),
        classifyDocument = lib.lookupFunction<_TextAllC, _TextAllD>(
            'pdf_document_classify_document'),
        // furniture
        eraseHeader = lib.lookupFunction<_DeI32I32C, _DeI32I32D>(
            'pdf_document_erase_header'),
        eraseFooter = lib.lookupFunction<_DeI32I32C, _DeI32I32D>(
            'pdf_document_erase_footer'),
        eraseArtifacts = lib.lookupFunction<_DeI32I32C, _DeI32I32D>(
            'pdf_document_erase_artifacts'),
        removeHeaders = lib
            .lookupFunction<_RemoveC, _RemoveD>('pdf_document_remove_headers'),
        removeFooters = lib
            .lookupFunction<_RemoveC, _RemoveD>('pdf_document_remove_footers'),
        removeArtifacts = lib.lookupFunction<_RemoveC, _RemoveD>(
            'pdf_document_remove_artifacts'),
        // forms
        getFormFields = lib.lookupFunction<_FormFieldsC, _FormFieldsD>(
            'pdf_document_get_form_fields'),
        formFieldCount = lib
            .lookupFunction<_FfCountC, _FfCountD>('pdf_oxide_form_field_count'),
        formFieldGetName = lib
            .lookupFunction<_FfStrC, _FfStrD>('pdf_oxide_form_field_get_name'),
        formFieldGetValue = lib
            .lookupFunction<_FfStrC, _FfStrD>('pdf_oxide_form_field_get_value'),
        formFieldGetType = lib
            .lookupFunction<_FfStrC, _FfStrD>('pdf_oxide_form_field_get_type'),
        formFieldIsReadonly = lib.lookupFunction<_FfBoolC, _FfBoolD>(
            'pdf_oxide_form_field_is_readonly'),
        formFieldIsRequired = lib.lookupFunction<_FfBoolC, _FfBoolD>(
            'pdf_oxide_form_field_is_required'),
        formFieldListFree = lib.lookupFunction<_FfFreeC, _FfFreeD>(
            'pdf_oxide_form_field_list_free'),
        exportFormData = lib.lookupFunction<_ExportFormC, _ExportFormD>(
            'pdf_document_export_form_data_to_bytes'),
        importFormData = lib.lookupFunction<_ImportFormC, _ImportFormD>(
            'pdf_document_import_form_data'),
        importFdfBytes = lib.lookupFunction<_ImportFdfC, _ImportFdfD>(
            'pdf_editor_import_fdf_bytes'),
        importXfdfBytes = lib.lookupFunction<_ImportFdfC, _ImportFdfD>(
            'pdf_editor_import_xfdf_bytes'),
        formImportFromFile =
            lib.lookupFunction<_FormImportFileC, _FormImportFileD>(
                'pdf_form_import_from_file'),
        // doc structure / metadata
        getOutline = lib
            .lookupFunction<_TextAllC, _TextAllD>('pdf_document_get_outline'),
        getPageLabels = lib.lookupFunction<_TextAllC, _TextAllD>(
            'pdf_document_get_page_labels'),
        getXmpMetadata = lib.lookupFunction<_TextAllC, _TextAllD>(
            'pdf_document_get_xmp_metadata'),
        getSourceBytes = lib.lookupFunction<_DocBytesC, _DocBytesD>(
            'pdf_document_get_source_bytes'),
        hasXfa = lib.lookupFunction<_BoolC, _BoolD>('pdf_document_has_xfa'),
        pdfGetPageCount =
            lib.lookupFunction<_PageCountC, _PageCountD>('pdf_get_page_count'),
        planSplitByBookmarks = lib.lookupFunction<_PlanSplitC, _PlanSplitD>(
            'pdf_document_plan_split_by_bookmarks'),
        // doc-level signatures
        docSign = lib.lookupFunction<_DocSignC, _DocSignD>('pdf_document_sign'),
        docGetSignature = lib.lookupFunction<_DocGetSigC, _DocGetSigD>(
            'pdf_document_get_signature'),
        docGetSignatureCount = lib.lookupFunction<_DocSigCountC, _DocSigCountD>(
            'pdf_document_get_signature_count'),
        docVerifyAllSignatures =
            lib.lookupFunction<_DocSigCountC, _DocSigCountD>(
                'pdf_document_verify_all_signatures'),
        docHasTimestamp = lib.lookupFunction<_DocSigCountC, _DocSigCountD>(
            'pdf_document_has_timestamp'),
        docGetDss = lib
            .lookupFunction<_DocGetDssC, _DocGetDssD>('pdf_document_get_dss'),
        // annotation extras
        annGetColor = lib.lookupFunction<_AnnU32C, _AnnU32D>(
            'pdf_oxide_annotation_get_color'),
        annGetCreationDate = lib.lookupFunction<_AnnI64C, _AnnI64D>(
            'pdf_oxide_annotation_get_creation_date'),
        annGetModificationDate = lib.lookupFunction<_AnnI64C, _AnnI64D>(
            'pdf_oxide_annotation_get_modification_date'),
        annIsHidden = lib.lookupFunction<_AnnBoolC, _AnnBoolD>(
            'pdf_oxide_annotation_is_hidden'),
        annIsMarkedDeleted = lib.lookupFunction<_AnnBoolC, _AnnBoolD>(
            'pdf_oxide_annotation_is_marked_deleted'),
        annIsPrintable = lib.lookupFunction<_AnnBoolC, _AnnBoolD>(
            'pdf_oxide_annotation_is_printable'),
        annIsReadOnly = lib.lookupFunction<_AnnBoolC, _AnnBoolD>(
            'pdf_oxide_annotation_is_read_only'),
        annotationsToJson = lib.lookupFunction<_AnnJsonC, _AnnJsonD>(
            'pdf_oxide_annotations_to_json'),
        highlightQuadCount = lib.lookupFunction<_QuadCountC, _QuadCountD>(
            'pdf_oxide_highlight_annotation_get_quad_points_count'),
        highlightQuadPoint = lib.lookupFunction<_QuadPointC, _QuadPointD>(
            'pdf_oxide_highlight_annotation_get_quad_point'),
        linkGetUri = lib.lookupFunction<_AnnStrC, _AnnStrD>(
            'pdf_oxide_link_annotation_get_uri'),
        textAnnotGetIcon = lib.lookupFunction<_AnnStrC, _AnnStrD>(
            'pdf_oxide_text_annotation_get_icon_name'),
        // list -> json / font size
        fontsToJson = lib
            .lookupFunction<_ListJsonC, _ListJsonD>('pdf_oxide_fonts_to_json'),
        searchResultsToJson = lib.lookupFunction<_ListJsonC, _ListJsonD>(
            'pdf_oxide_search_results_to_json'),
        fontGetSize = lib
            .lookupFunction<_FontSizeC, _FontSizeD>('pdf_oxide_font_get_size'),
        // crypto / FIPS
        cryptoActiveProvider = lib.lookupFunction<_NullStrC, _NullStrD>(
            'pdf_oxide_crypto_active_provider'),
        cryptoCbom =
            lib.lookupFunction<_NullStrC, _NullStrD>('pdf_oxide_crypto_cbom'),
        cryptoInventory = lib
            .lookupFunction<_NullStrC, _NullStrD>('pdf_oxide_crypto_inventory'),
        cryptoPolicy =
            lib.lookupFunction<_NullStrC, _NullStrD>('pdf_oxide_crypto_policy'),
        cryptoFipsAvailable = lib.lookupFunction<_NullI32C, _NullI32D>(
            'pdf_oxide_crypto_fips_available'),
        cryptoUseFips = lib
            .lookupFunction<_NullI32C, _NullI32D>('pdf_oxide_crypto_use_fips'),
        cryptoSetPolicy = lib.lookupFunction<_StrArgI32C, _StrArgI32D>(
            'pdf_oxide_crypto_set_policy'),
        // models / config
        modelManifest = lib
            .lookupFunction<_NullStrC, _NullStrD>('pdf_oxide_model_manifest'),
        prefetchAvailable = lib.lookupFunction<_NullI32C, _NullI32D>(
            'pdf_oxide_prefetch_available'),
        prefetchModels = lib.lookupFunction<_PrefetchC, _PrefetchD>(
            'pdf_oxide_prefetch_models'),
        setMaxOpsPerStream = lib.lookupFunction<_SetI64C, _SetI64D>(
            'pdf_oxide_set_max_ops_per_stream'),
        setPreserveUnmappedGlyphs = lib.lookupFunction<_SetI32C, _SetI32D>(
            'pdf_oxide_set_preserve_unmapped_glyphs'),
        convertToPdfA =
            lib.lookupFunction<_ConvPdfAC, _ConvPdfAD>('pdf_convert_to_pdf_a'),
        // streaming tables
        stBegin = lib.lookupFunction<_StBeginC, _StBeginD>(
            'pdf_page_builder_streaming_table_begin'),
        stBeginV2 = lib.lookupFunction<_StBeginV2C, _StBeginV2D>(
            'pdf_page_builder_streaming_table_begin_v2'),
        stPushRow = lib.lookupFunction<_StPushRowC, _StPushRowD>(
            'pdf_page_builder_streaming_table_push_row'),
        stPushRowV2 = lib.lookupFunction<_StPushRowV2C, _StPushRowV2D>(
            'pdf_page_builder_streaming_table_push_row_v2'),
        stFlush = lib.lookupFunction<_StStatusC, _StStatusD>(
            'pdf_page_builder_streaming_table_flush'),
        stFinish = lib.lookupFunction<_StStatusC, _StStatusD>(
            'pdf_page_builder_streaming_table_finish'),
        stSetBatchSize = lib.lookupFunction<_StSetBatchC, _StSetBatchD>(
            'pdf_page_builder_streaming_table_set_batch_size'),
        stBatchCount = lib.lookupFunction<_StUsizeC, _StUsizeD>(
            'pdf_page_builder_streaming_table_batch_count'),
        stPendingRowCount = lib.lookupFunction<_StUsizeC, _StUsizeD>(
            'pdf_page_builder_streaming_table_pending_row_count');

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
  // page rendering (Phase 3)
  final _RenderPageD renderPage;
  final _RenderZoomD renderPageZoom;
  final _RenderThumbD renderPageThumbnail;
  final _RenderedDimD renderedImageWidth;
  final _RenderedDimD renderedImageHeight;
  final _RenderedDataD renderedImageData;
  final _RenderedSaveD renderedImageSave;
  final _RenderedFreeD renderedImageFree;
  // document editing
  final _DeOpenD deOpen;
  final _DeOpenBytesD deOpenBytes;
  final _DeFreeD deFree;
  final _DeBoolD deIsModified;
  final _DeStrD deGetSourcePath;
  final _DeVersionD deGetVersion;
  final _DeI32D dePageCount;
  final _DeStrD deGetProducer;
  final _DeSetStrD deSetProducer;
  final _DeStrD deGetCreationDate;
  final _DeSetStrD deSetCreationDate;
  final _DeSaveD deSave;
  final _DeSaveBytesD deSaveToBytes;
  final _DeSaveBytesOptsD deSaveToBytesWithOptions;
  final _DeExtractPagesD deExtractPagesToBytes;
  final _DeConvertPdfAD deConvertToPdfA;
  final _DeSaveEncBytesD deSaveEncryptedToBytes;
  final _DeMergeBytesD deMergeFromBytes;
  final _DeEmbedFileD deEmbedFile;
  final _DeUsizeI32D deApplyPageRedactions;
  final _DeI32D deApplyAllRedactions;
  final _DeI32I32D deRotateAllPages;
  final _DeRotateByD deRotatePageBy;
  final _DeGetBoxD deGetPageMediaBox;
  final _DeSetBoxD deSetPageMediaBox;
  final _DeGetBoxD deGetPageCropBox;
  final _DeSetBoxD deSetPageCropBox;
  final _DeEraseRegionsD deEraseRegions;
  final _DeUsizeI32D deClearEraseRegions;
  final _DeIsMarkedD deIsPageMarkedForFlatten;
  final _DeUsizeI32D deUnmarkPageForFlatten;
  final _DeIsMarkedD deIsPageMarkedForRedaction;
  final _DeUsizeI32D deUnmarkPageForRedaction;
  final _DeI32I32D deDeletePage;
  final _DeI32I32I32D deMovePage;
  final _DeI32I32D deGetPageRotation;
  final _DeI32I32I32D deSetPageRotation;
  final _DeEraseRegionD deEraseRegion;
  final _DeI32I32D deFlattenAnnotations;
  final _DeI32D deFlattenAllAnnotations;
  final _DeCropMarginsD deCropMargins;
  final _DeMergeFromD deMergeFrom;
  final _DeSaveEncD deSaveEncrypted;
  final _DeSetFormD deSetFormFieldValue;
  final _DeI32D deFlattenForms;
  final _DeI32I32D deFlattenFormsOnPage;
  final _DeWarnCountD deFlattenWarningsCount;
  final _DeWarnD deFlattenWarning;
  // PDF creation — EmbeddedFont
  final _EfFromFileD efFromFile;
  final _EfFromBytesD efFromBytes;
  final _EfFreeD efFree;
  // PDF creation — DocumentBuilder
  final _DbCreateD dbCreate;
  final _DbFreeD dbFree;
  final _DbSetStrD dbSetTitle;
  final _DbSetStrD dbSetAuthor;
  final _DbSetStrD dbSetSubject;
  final _DbSetStrD dbSetKeywords;
  final _DbSetStrD dbSetCreator;
  final _DbSetStrD dbOnOpen;
  final _DbStatus0D dbTaggedPdfUa1;
  final _DbSetStrD dbLanguage;
  final _DbRoleMapD dbRoleMap;
  final _DbRegisterFontD dbRegisterFont;
  final _DbPageD dbA4Page;
  final _DbPageD dbLetterPage;
  final _DbPageSizeD dbPage;
  final _DbBuildD dbBuild;
  final _DbSaveD dbSave;
  final _DbSaveEncD dbSaveEncrypted;
  final _DbToBytesEncD dbToBytesEncrypted;
  // PDF creation — PageBuilder
  final _PbFreeD pbFree;
  final _PbStatus0D pbDone;
  final _PbFontD pbFont;
  final _PbAtD pbAt;
  final _PbStrD pbText;
  final _PbHeadingD pbHeading;
  final _PbStrD pbParagraph;
  final _PbSpaceD pbSpace;
  final _PbStatus0D pbHorizontalRule;
  final _PbStrD pbLinkUrl;
  final _PbLinkPageD pbLinkPage;
  final _PbStrD pbLinkNamed;
  final _PbStrD pbLinkJavascript;
  final _PbStrD pbOnOpen;
  final _PbStrD pbOnClose;
  final _PbStrD pbFieldKeystroke;
  final _PbStrD pbFieldFormat;
  final _PbStrD pbFieldValidate;
  final _PbStrD pbFieldCalculate;
  final _PbRgbD pbHighlight;
  final _PbRgbD pbUnderline;
  final _PbRgbD pbStrikeout;
  final _PbRgbD pbSquiggly;
  final _PbStrD pbStickyNote;
  final _PbStickyAtD pbStickyNoteAt;
  final _PbStrD pbWatermark;
  final _PbStatus0D pbWatermarkConfidential;
  final _PbStatus0D pbWatermarkDraft;
  final _PbStampD pbStamp;
  final _PbFreetextD pbFreetext;
  final _PbTextFieldD pbTextField;
  final _PbCheckboxD pbCheckbox;
  final _PbComboBoxD pbComboBox;
  final _PbRadioGroupD pbRadioGroup;
  final _PbPushButtonD pbPushButton;
  final _PbSigFieldD pbSignatureField;
  final _PbFootnoteD pbFootnote;
  final _PbColumnsD pbColumns;
  final _PbStrD pbInline;
  final _PbStrD pbInlineBold;
  final _PbStrD pbInlineItalic;
  final _PbInlineColorD pbInlineColor;
  final _PbStatus0D pbNewline;
  final _PbBarcode1dD pbBarcode1d;
  final _PbBarcodeQrD pbBarcodeQr;
  final _PbImageD pbImage;
  final _PbImageAltD pbImageWithAlt;
  final _PbImageD pbImageArtifact;
  final _PbRectD pbRect;
  final _PbFilledRectD pbFilledRect;
  final _PbLineD pbLine;
  final _PbStrokeRectD pbStrokeRect;
  final _PbStrokeLineD pbStrokeLine;
  final _PbStrokeRectDashedD pbStrokeRectDashed;
  final _PbStrokeRectDashedD pbStrokeLineDashed;
  final _PbTextInRectD pbTextInRect;
  final _PbStatus0D pbNewPageSameSize;
  final _PbTableD pbTable;
  // Phase 6: signatures / PKI / timestamps / TSA / DSS / validation
  final _CertLoadBytesD certLoadFromBytes;
  final _CertLoadPemD certLoadFromPem;
  final _CertStrD certGetSubject, certGetIssuer, certGetSerial;
  final _CertValidityD certGetValidity;
  final _CertI32D certIsValid;
  final _PtrFreeD certFree;
  final _SignBytesD signBytes;
  final _SignPadesD signBytesPades;
  final _SignPadesOptsD signBytesPadesOpts;
  final _SigStrD sigGetSignerName, sigGetSigningReason, sigGetSigningLocation;
  final _SigI64D sigGetSigningTime;
  final _SigPtrD sigGetCertificate, sigGetTimestamp;
  final _SigI32D sigGetPadesLevel, sigVerify;
  final _SigBoolD sigHasTimestamp;
  final _SigAddTsD sigAddTimestamp;
  final _SigVerifyDetachedD sigVerifyDetached;
  final _PtrFreeD sigFree;
  final _TsParseD tsParse;
  final _TsConstBytesD tsGetToken, tsGetMessageImprint;
  final _TsI64D tsGetTime;
  final _TsStrD tsGetSerial, tsGetTsaName, tsGetPolicyOid;
  final _TsI32D tsGetHashAlgorithm;
  final _TsBoolD tsVerify;
  final _PtrFreeD tsFree;
  final _TsaCreateD tsaClientCreate;
  final _TsaReqD tsaRequestTimestamp;
  final _TsaReqHashD tsaRequestTimestampHash;
  final _PtrFreeD tsaClientFree;
  final _DssCountD dssCertCount, dssCrlCount, dssOcspCount, dssVriCount;
  final _DssGetD dssGetCert, dssGetCrl, dssGetOcsp;
  final _PtrFreeD dssFree;
  final _ValidateD validatePdfALevel, validatePdfUa, validatePdfXLevel;
  final _ValBoolD pdfAIsCompliant, pdfUaIsAccessible, pdfXIsCompliant;
  final _ValCountD pdfAErrorCount, pdfAWarningCount;
  final _ValCountD pdfUaErrorCount, pdfUaWarningCount, pdfXErrorCount;
  final _ValGetD pdfAGetError, pdfUaGetError, pdfUaGetWarning, pdfXGetError;
  final _UaStatsD pdfUaGetStats;
  final _ValFreeD pdfAResultsFree, pdfUaResultsFree, pdfXResultsFree;
  final _SetLogD setLogLevel;
  final _GetLogD getLogLevel;
  // Phase 7: barcodes / OCR / render variants / redaction / from_* / page getters
  final _GenQrD generateQrCode;
  final _GenBarcodeD generateBarcode;
  final _BcStrD barcodeGetData;
  final _BcI32D barcodeGetFormat;
  final _BcF32D barcodeGetConfidence;
  final _BcPngD barcodeGetImagePng;
  final _BcSvgD barcodeGetSvg;
  final _PtrFreeD barcodeFree;
  final _AddBarcodeD addBarcodeToPage;
  final _OcrCreateD ocrEngineCreate;
  final _PtrFreeD ocrEngineFree;
  final _OcrNeedsD ocrPageNeedsOcr;
  final _OcrExtractD ocrExtractText;
  final _RenderOptsD renderPageWithOptions;
  final _RenderOptsExD renderPageWithOptionsEx;
  final _RenderRegionD renderPageRegion;
  final _RenderFitD renderPageFit;
  final _RenderRawD renderPageRaw;
  final _CreateRendererD createRenderer;
  final _PtrFreeD rendererFree;
  final _EstimateRenderD estimateRenderTime;
  final _RedactionAddD redactionAdd;
  final _RedactionCountD redactionCount;
  final _RedactionApplyD redactionApply;
  final _RedactionScrubD redactionScrubMetadata;
  final _FromImageD fromImage;
  final _FromImageBytesD fromImageBytes;
  final _FromHtmlCssD fromHtmlCss;
  final _FromHtmlCssFontsD fromHtmlCssWithFonts;
  final _MergeD merge;
  final _PageF32D pageGetWidth;
  final _PageF32D pageGetHeight;
  final _PageI32D pageGetRotation;
  final _PageElementsD pageGetElements;
  final _ElemCountD elementCount;
  final _ListStrD elementGetType;
  final _ListStrD elementGetText;
  final _ElemRectD elementGetRect;
  final _ListFreeD elementsFree;
  final _ElemJsonD elementsToJson;
  final _AddTimestampD addTimestamp;
  // ── Phase 8 ─────────────────────────────────────────────────────────────────
  final _OpenBytesD openFromDocx, openFromPptx, openFromXlsx;
  final _DocBytesD toDocx, toPptx, toXlsx, getSourceBytes;
  final _RectStrD extractTextInRect;
  final _RectListD extractWordsInRect,
      extractLinesInRect,
      extractTablesInRect,
      extractImagesInRect;
  final _TextD extractTextAuto, classifyPage;
  final _TextAllD extractAllText,
      classifyDocument,
      getOutline,
      getPageLabels,
      getXmpMetadata;
  final _PageAutoD extractPageAuto;
  final _DeI32I32D eraseHeader, eraseFooter, eraseArtifacts;
  final _RemoveD removeHeaders, removeFooters, removeArtifacts;
  final _FormFieldsD getFormFields;
  final _FfCountD formFieldCount;
  final _FfStrD formFieldGetName, formFieldGetValue, formFieldGetType;
  final _FfBoolD formFieldIsReadonly, formFieldIsRequired;
  final _FfFreeD formFieldListFree;
  final _ExportFormD exportFormData;
  final _ImportFormD importFormData;
  final _ImportFdfD importFdfBytes, importXfdfBytes;
  final _FormImportFileD formImportFromFile;
  final _BoolD hasXfa;
  final _PageCountD pdfGetPageCount;
  final _PlanSplitD planSplitByBookmarks;
  final _DocSignD docSign;
  final _DocGetSigD docGetSignature;
  final _DocSigCountD docGetSignatureCount,
      docVerifyAllSignatures,
      docHasTimestamp;
  final _DocGetDssD docGetDss;
  final _AnnU32D annGetColor;
  final _AnnI64D annGetCreationDate, annGetModificationDate;
  final _AnnBoolD annIsHidden,
      annIsMarkedDeleted,
      annIsPrintable,
      annIsReadOnly;
  final _AnnJsonD annotationsToJson;
  final _QuadCountD highlightQuadCount;
  final _QuadPointD highlightQuadPoint;
  final _AnnStrD linkGetUri, textAnnotGetIcon;
  final _ListJsonD fontsToJson, searchResultsToJson;
  final _FontSizeD fontGetSize;
  final _NullStrD cryptoActiveProvider,
      cryptoCbom,
      cryptoInventory,
      cryptoPolicy,
      modelManifest;
  final _NullI32D cryptoFipsAvailable, cryptoUseFips, prefetchAvailable;
  final _StrArgI32D cryptoSetPolicy;
  final _PrefetchD prefetchModels;
  final _SetI64D setMaxOpsPerStream;
  final _SetI32D setPreserveUnmappedGlyphs;
  final _ConvPdfAD convertToPdfA;
  final _StBeginD stBegin;
  final _StBeginV2D stBeginV2;
  final _StPushRowD stPushRow;
  final _StPushRowV2D stPushRowV2;
  final _StStatusD stFlush, stFinish;
  final _StSetBatchD stSetBatchSize;
  final _StUsizeD stBatchCount, stPendingRowCount;
}

typedef _OpenD = Pointer<Void> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _OpenBytesD = Pointer<Void> Function(
    Pointer<Uint8>, int, Pointer<Int32>);
typedef _OpenPwD = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);

// ── Phase 7: barcodes / QR / OCR / render variants / redaction / from_* ──────
// Conventions match earlier phases: opaque handles are `Pointer<Void>` freed via
// their `*_free` symbol (on close()/finalizer + closed-handle guards); owned
// `char*` go through `_takeString` (+ `free_string`); owned `uint8*` buffers are
// copied then released with `free_bytes`. Render variants return the same
// `FfiRenderedImage` handle wrapped by [RenderedImage]. int32 status returns are
// 0 = success. String / parallel-array params are marshalled C-side (copied).

// barcodes / QR
typedef _GenQrC = Pointer<Void> Function(
    Pointer<Utf8>, Int32, Int32, Pointer<Int32>);
typedef _GenQrD = Pointer<Void> Function(
    Pointer<Utf8>, int, int, Pointer<Int32>);
typedef _GenBarcodeC = Pointer<Void> Function(
    Pointer<Utf8>, Int32, Int32, Pointer<Int32>);
typedef _GenBarcodeD = Pointer<Void> Function(
    Pointer<Utf8>, int, int, Pointer<Int32>);
typedef _BcStrC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _BcStrD = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _BcI32C = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _BcI32D = int Function(Pointer<Void>, Pointer<Int32>);
typedef _BcF32C = Float Function(Pointer<Void>, Pointer<Int32>);
typedef _BcF32D = double Function(Pointer<Void>, Pointer<Int32>);
typedef _BcPngC = Pointer<Uint8> Function(
    Pointer<Void>, Int32, Pointer<Int32>, Pointer<Int32>);
typedef _BcPngD = Pointer<Uint8> Function(
    Pointer<Void>, int, Pointer<Int32>, Pointer<Int32>);
typedef _BcSvgC = Pointer<Utf8> Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _BcSvgD = Pointer<Utf8> Function(Pointer<Void>, int, Pointer<Int32>);
typedef _AddBarcodeC = Int32 Function(Pointer<Void>, Int32, Pointer<Void>,
    Float, Float, Float, Float, Pointer<Int32>);
typedef _AddBarcodeD = int Function(Pointer<Void>, int, Pointer<Void>, double,
    double, double, double, Pointer<Int32>);

// OCR
typedef _OcrCreateC = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _OcrCreateD = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _OcrNeedsC = Bool Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _OcrNeedsD = bool Function(Pointer<Void>, int, Pointer<Int32>);
typedef _OcrExtractC = Pointer<Utf8> Function(
    Pointer<Void>, Int32, Pointer<Void>, Pointer<Int32>);
typedef _OcrExtractD = Pointer<Utf8> Function(
    Pointer<Void>, int, Pointer<Void>, Pointer<Int32>);

// render variants (all return FfiRenderedImage*)
typedef _RenderOptsC = Pointer<Void> Function(Pointer<Void>, Int32, Int32,
    Int32, Float, Float, Float, Float, Int32, Int32, Int32, Pointer<Int32>);
typedef _RenderOptsD = Pointer<Void> Function(Pointer<Void>, int, int, int,
    double, double, double, double, int, int, int, Pointer<Int32>);
typedef _RenderOptsExC = Pointer<Void> Function(
    Pointer<Void>,
    Int32,
    Int32,
    Int32,
    Float,
    Float,
    Float,
    Float,
    Int32,
    Int32,
    Int32,
    Pointer<Pointer<Utf8>>,
    IntPtr,
    Pointer<Int32>);
typedef _RenderOptsExD = Pointer<Void> Function(
    Pointer<Void>,
    int,
    int,
    int,
    double,
    double,
    double,
    double,
    int,
    int,
    int,
    Pointer<Pointer<Utf8>>,
    int,
    Pointer<Int32>);
typedef _RenderRegionC = Pointer<Void> Function(
    Pointer<Void>, Int32, Float, Float, Float, Float, Int32, Pointer<Int32>);
typedef _RenderRegionD = Pointer<Void> Function(
    Pointer<Void>, int, double, double, double, double, int, Pointer<Int32>);
typedef _RenderFitC = Pointer<Void> Function(
    Pointer<Void>, Int32, Int32, Int32, Int32, Pointer<Int32>);
typedef _RenderFitD = Pointer<Void> Function(
    Pointer<Void>, int, int, int, int, Pointer<Int32>);
typedef _RenderRawC = Pointer<Void> Function(Pointer<Void>, Int32, Int32,
    Pointer<Int32>, Pointer<Int32>, Pointer<Int32>);
typedef _RenderRawD = Pointer<Void> Function(
    Pointer<Void>, int, int, Pointer<Int32>, Pointer<Int32>, Pointer<Int32>);
typedef _CreateRendererC = Pointer<Void> Function(
    Int32, Int32, Int32, Bool, Pointer<Int32>);
typedef _CreateRendererD = Pointer<Void> Function(
    int, int, int, bool, Pointer<Int32>);
typedef _EstimateRenderC = Int32 Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _EstimateRenderD = int Function(Pointer<Void>, int, Pointer<Int32>);

// redaction (on DocumentEditor)
typedef _RedactionAddC = Int32 Function(Pointer<Void>, IntPtr, Double, Double,
    Double, Double, Double, Double, Double, Pointer<Int32>);
typedef _RedactionAddD = int Function(Pointer<Void>, int, double, double,
    double, double, double, double, double, Pointer<Int32>);
typedef _RedactionCountC = Int32 Function(
    Pointer<Void>, IntPtr, Pointer<Int32>);
typedef _RedactionCountD = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _RedactionApplyC = Int32 Function(
    Pointer<Void>, Bool, Double, Double, Double, Pointer<Int32>);
typedef _RedactionApplyD = int Function(
    Pointer<Void>, bool, double, double, double, Pointer<Int32>);
typedef _RedactionScrubC = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _RedactionScrubD = int Function(Pointer<Void>, Pointer<Int32>);

// constructors (return Pdf*)
typedef _FromImageC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _FromImageD = Pointer<Void> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _FromImageBytesC = Pointer<Void> Function(
    Pointer<Uint8>, Int32, Pointer<Int32>);
typedef _FromImageBytesD = Pointer<Void> Function(
    Pointer<Uint8>, int, Pointer<Int32>);
typedef _FromHtmlCssC = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Uint8>, IntPtr, Pointer<Int32>);
typedef _FromHtmlCssD = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Uint8>, int, Pointer<Int32>);
typedef _FromHtmlCssFontsC = Pointer<Void> Function(
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<Pointer<Utf8>>,
    Pointer<Pointer<Uint8>>,
    Pointer<IntPtr>,
    IntPtr,
    Pointer<Int32>);
typedef _FromHtmlCssFontsD = Pointer<Void> Function(
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<Pointer<Utf8>>,
    Pointer<Pointer<Uint8>>,
    Pointer<IntPtr>,
    int,
    Pointer<Int32>);
typedef _MergeC = Pointer<Uint8> Function(
    Pointer<Pointer<Utf8>>, Int32, Pointer<Int32>, Pointer<Int32>);
typedef _MergeD = Pointer<Uint8> Function(
    Pointer<Pointer<Utf8>>, int, Pointer<Int32>, Pointer<Int32>);

// page getters + element list (on PdfDocument)
typedef _PageF32C = Float Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _PageF32D = double Function(Pointer<Void>, int, Pointer<Int32>);
typedef _PageI32C = Int32 Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _PageI32D = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _PageElementsC = Pointer<Void> Function(
    Pointer<Void>, Int32, Pointer<Int32>);
typedef _PageElementsD = Pointer<Void> Function(
    Pointer<Void>, int, Pointer<Int32>);
typedef _ElemCountC = Int32 Function(Pointer<Void>);
typedef _ElemCountD = int Function(Pointer<Void>);
typedef _ElemRectC = Void Function(Pointer<Void>, Int32, Pointer<Float>,
    Pointer<Float>, Pointer<Float>, Pointer<Float>, Pointer<Int32>);
typedef _ElemRectD = void Function(Pointer<Void>, int, Pointer<Float>,
    Pointer<Float>, Pointer<Float>, Pointer<Float>, Pointer<Int32>);
typedef _ElemJsonC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _ElemJsonD = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);

// timestamp (top-level)
typedef _AddTimestampC = Bool Function(Pointer<Uint8>, IntPtr, Int32,
    Pointer<Utf8>, Pointer<Pointer<Uint8>>, Pointer<IntPtr>, Pointer<Int32>);
typedef _AddTimestampD = bool Function(Pointer<Uint8>, int, int, Pointer<Utf8>,
    Pointer<Pointer<Uint8>>, Pointer<IntPtr>, Pointer<Int32>);

// ── Phase 6: digital signatures / PKI / timestamps / TSA / PDF-A,X,UA ─────────
// Conventions match earlier phases: opaque handles are `Pointer<Void>` freed via
// their `*_free` symbol (on close()/finalizer + closed-handle guards); owned
// `char*` go through `_takeString` + `free_string`; owned `uint8*` buffers are
// copied then released with `free_bytes`. A `const uint8*` return (timestamp
// token / message imprint) is COPIED only — never `free_bytes`'d. Validation
// result handles are freed by their dedicated `*_results_free`. The PAdES sign
// entry points marshal three parallel DER byte-array arrays.

// PadesSignOptionsC mirrors the #[repr(C)] struct in the header (14 fields).
final class _PadesSignOptionsC extends Struct {
  external Pointer<Void> certificateHandle;
  external Pointer<Pointer<Uint8>> certs;
  external Pointer<IntPtr> certLens;
  @IntPtr()
  external int nCerts;
  external Pointer<Pointer<Uint8>> crls;
  external Pointer<IntPtr> crlLens;
  @IntPtr()
  external int nCrls;
  external Pointer<Pointer<Uint8>> ocsps;
  external Pointer<IntPtr> ocspLens;
  @IntPtr()
  external int nOcsps;
  external Pointer<Utf8> tsaUrl;
  external Pointer<Utf8> reason;
  external Pointer<Utf8> location;
  @Int32()
  external int level;
}

// certificate / signing
typedef _CertLoadBytesC = Pointer<Void> Function(
    Pointer<Uint8>, Int32, Pointer<Utf8>, Pointer<Int32>);
typedef _CertLoadBytesD = Pointer<Void> Function(
    Pointer<Uint8>, int, Pointer<Utf8>, Pointer<Int32>);
typedef _CertLoadPemC = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _CertLoadPemD = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _CertStrC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _CertStrD = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _CertValidityC = Void Function(
    Pointer<Void>, Pointer<Int64>, Pointer<Int64>, Pointer<Int32>);
typedef _CertValidityD = void Function(
    Pointer<Void>, Pointer<Int64>, Pointer<Int64>, Pointer<Int32>);
typedef _CertI32C = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _CertI32D = int Function(Pointer<Void>, Pointer<Int32>);
typedef _PtrFreeC = Void Function(Pointer<Void>);
typedef _PtrFreeD = void Function(Pointer<Void>);

typedef _SignBytesC = Pointer<Uint8> Function(
    Pointer<Uint8>,
    IntPtr,
    Pointer<Void>,
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<IntPtr>,
    Pointer<Int32>);
typedef _SignBytesD = Pointer<Uint8> Function(
    Pointer<Uint8>,
    int,
    Pointer<Void>,
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<IntPtr>,
    Pointer<Int32>);
typedef _SignPadesC = Pointer<Uint8> Function(
    Pointer<Uint8>,
    IntPtr,
    Pointer<Void>,
    Int32,
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<Pointer<Uint8>>,
    Pointer<IntPtr>,
    IntPtr,
    Pointer<Pointer<Uint8>>,
    Pointer<IntPtr>,
    IntPtr,
    Pointer<Pointer<Uint8>>,
    Pointer<IntPtr>,
    IntPtr,
    Pointer<IntPtr>,
    Pointer<Int32>);
typedef _SignPadesD = Pointer<Uint8> Function(
    Pointer<Uint8>,
    int,
    Pointer<Void>,
    int,
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<Pointer<Uint8>>,
    Pointer<IntPtr>,
    int,
    Pointer<Pointer<Uint8>>,
    Pointer<IntPtr>,
    int,
    Pointer<Pointer<Uint8>>,
    Pointer<IntPtr>,
    int,
    Pointer<IntPtr>,
    Pointer<Int32>);
typedef _SignPadesOptsC = Pointer<Uint8> Function(Pointer<Uint8>, IntPtr,
    Pointer<_PadesSignOptionsC>, Pointer<IntPtr>, Pointer<Int32>);
typedef _SignPadesOptsD = Pointer<Uint8> Function(Pointer<Uint8>, int,
    Pointer<_PadesSignOptionsC>, Pointer<IntPtr>, Pointer<Int32>);

// signature info
typedef _SigStrC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _SigStrD = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _SigI64C = Int64 Function(Pointer<Void>, Pointer<Int32>);
typedef _SigI64D = int Function(Pointer<Void>, Pointer<Int32>);
typedef _SigPtrC = Pointer<Void> Function(Pointer<Void>, Pointer<Int32>);
typedef _SigPtrD = Pointer<Void> Function(Pointer<Void>, Pointer<Int32>);
typedef _SigI32C = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _SigI32D = int Function(Pointer<Void>, Pointer<Int32>);
typedef _SigBoolC = Bool Function(Pointer<Void>, Pointer<Int32>);
typedef _SigBoolD = bool Function(Pointer<Void>, Pointer<Int32>);
typedef _SigAddTsC = Bool Function(
    Pointer<Void>, Pointer<Void>, Pointer<Int32>);
typedef _SigAddTsD = bool Function(
    Pointer<Void>, Pointer<Void>, Pointer<Int32>);
typedef _SigVerifyDetachedC = Int32 Function(
    Pointer<Void>, Pointer<Uint8>, IntPtr, Pointer<Int32>);
typedef _SigVerifyDetachedD = int Function(
    Pointer<Void>, Pointer<Uint8>, int, Pointer<Int32>);

// timestamp
typedef _TsParseC = Pointer<Void> Function(
    Pointer<Uint8>, IntPtr, Pointer<Int32>);
typedef _TsParseD = Pointer<Void> Function(Pointer<Uint8>, int, Pointer<Int32>);
typedef _TsConstBytesC = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<IntPtr>, Pointer<Int32>);
typedef _TsConstBytesD = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<IntPtr>, Pointer<Int32>);
typedef _TsI64C = Int64 Function(Pointer<Void>, Pointer<Int32>);
typedef _TsI64D = int Function(Pointer<Void>, Pointer<Int32>);
typedef _TsStrC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _TsStrD = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _TsI32C = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _TsI32D = int Function(Pointer<Void>, Pointer<Int32>);
typedef _TsBoolC = Bool Function(Pointer<Void>, Pointer<Int32>);
typedef _TsBoolD = bool Function(Pointer<Void>, Pointer<Int32>);

// TSA client
typedef _TsaCreateC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>,
    Pointer<Utf8>, Int32, Int32, Bool, Bool, Pointer<Int32>);
typedef _TsaCreateD = Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>,
    Pointer<Utf8>, int, int, bool, bool, Pointer<Int32>);
typedef _TsaReqC = Pointer<Void> Function(
    Pointer<Void>, Pointer<Uint8>, IntPtr, Pointer<Int32>);
typedef _TsaReqD = Pointer<Void> Function(
    Pointer<Void>, Pointer<Uint8>, int, Pointer<Int32>);
typedef _TsaReqHashC = Pointer<Void> Function(
    Pointer<Void>, Pointer<Uint8>, IntPtr, Int32, Pointer<Int32>);
typedef _TsaReqHashD = Pointer<Void> Function(
    Pointer<Void>, Pointer<Uint8>, int, int, Pointer<Int32>);

// DSS
typedef _DssCountC = Int32 Function(Pointer<Void>);
typedef _DssCountD = int Function(Pointer<Void>);
typedef _DssGetC = Pointer<Uint8> Function(
    Pointer<Void>, Int32, Pointer<IntPtr>, Pointer<Int32>);
typedef _DssGetD = Pointer<Uint8> Function(
    Pointer<Void>, int, Pointer<IntPtr>, Pointer<Int32>);

// validation
typedef _ValidateC = Pointer<Void> Function(
    Pointer<Void>, Int32, Pointer<Int32>);
typedef _ValidateD = Pointer<Void> Function(Pointer<Void>, int, Pointer<Int32>);
typedef _ValBoolC = Bool Function(Pointer<Void>, Pointer<Int32>);
typedef _ValBoolD = bool Function(Pointer<Void>, Pointer<Int32>);
typedef _ValCountC = Int32 Function(Pointer<Void>);
typedef _ValCountD = int Function(Pointer<Void>);
typedef _ValGetC = Pointer<Utf8> Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _ValGetD = Pointer<Utf8> Function(Pointer<Void>, int, Pointer<Int32>);
typedef _ValFreeC = Void Function(Pointer<Void>);
typedef _ValFreeD = void Function(Pointer<Void>);
typedef _UaStatsC = Bool Function(
    Pointer<Void>,
    Pointer<Int32>,
    Pointer<Int32>,
    Pointer<Int32>,
    Pointer<Int32>,
    Pointer<Int32>,
    Pointer<Int32>,
    Pointer<Int32>);
typedef _UaStatsD = bool Function(
    Pointer<Void>,
    Pointer<Int32>,
    Pointer<Int32>,
    Pointer<Int32>,
    Pointer<Int32>,
    Pointer<Int32>,
    Pointer<Int32>,
    Pointer<Int32>);

// log level
typedef _SetLogC = Void Function(Int32);
typedef _SetLogD = void Function(int);
typedef _GetLogC = Int32 Function();
typedef _GetLogD = int Function();

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

/// An interactive form field (AcroForm) as returned by
/// [PdfDocument.getFormFields].
class FormField {
  const FormField(
      this.name, this.value, this.type, this.readonly, this.required);

  /// The fully-qualified field name.
  final String name;

  /// The current field value (may be empty).
  final String value;

  /// The field type (e.g. `Tx`, `Btn`, `Ch`, `Sig`).
  final String type;

  /// Whether the field is read-only.
  final bool readonly;

  /// Whether the field is required.
  final bool required;
}

/// A single highlight-annotation quad (four corner points, in PDF user space).
class QuadPoint {
  const QuadPoint(
      this.x1, this.y1, this.x2, this.y2, this.x3, this.y3, this.x4, this.y4);
  final double x1, y1, x2, y2, x3, y3, x4, y4;
}

/// Extended attributes of a page annotation (flags, dates, colour, and
/// subtype-specific data) as returned by [PdfDocument.pageAnnotationDetails].
class AnnotationDetails {
  const AnnotationDetails(
      this.type,
      this.subtype,
      this.content,
      this.rect,
      this.color,
      this.creationDate,
      this.modificationDate,
      this.hidden,
      this.markedDeleted,
      this.printable,
      this.readOnly,
      this.linkUri,
      this.iconName,
      this.quadPoints);

  final String type;
  final String subtype;
  final String content;
  final Bbox rect;

  /// Packed ARGB colour value.
  final int color;

  /// Creation timestamp (Unix epoch seconds; 0 if absent).
  final int creationDate;

  /// Modification timestamp (Unix epoch seconds; 0 if absent).
  final int modificationDate;
  final bool hidden;
  final bool markedDeleted;
  final bool printable;
  final bool readOnly;

  /// The target URI for link annotations (empty otherwise).
  final String linkUri;

  /// The icon name for text annotations (empty otherwise).
  final String iconName;

  /// Quad points for highlight annotations (empty otherwise).
  final List<QuadPoint> quadPoints;
}

/// A rasterised page image produced by [PdfDocument.renderPage] (and friends).
///
/// Owns the native `FfiRenderedImage` handle; [width], [height] and [data] are
/// read through it on demand, and [save] writes the encoded image to disk via
/// the native saver. Call [close] when done (or rely on the finalizer). The
/// encoded image bytes returned by [data] are copied into Dart and the native
/// buffer is freed via `free_bytes`.
class RenderedImage implements Finalizable {
  RenderedImage._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_RenderedFreeC>>('pdf_rendered_image_free'));
  Pointer<Void> _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('RenderedImage is closed');
  }

  /// Image width in pixels.
  int get width {
    _check();
    final code = calloc<Int32>();
    try {
      final w = _n.renderedImageWidth(_handle, code);
      if (code.value != 0)
        throw PdfOxideError(code.value, 'renderedImageWidth');
      return w;
    } finally {
      calloc.free(code);
    }
  }

  /// Image height in pixels.
  int get height {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.renderedImageHeight(_handle, code);
      if (code.value != 0) {
        throw PdfOxideError(code.value, 'renderedImageHeight');
      }
      return h;
    } finally {
      calloc.free(code);
    }
  }

  /// Encoded image bytes (e.g. PNG). Copied into Dart; the native buffer is
  /// freed via `free_bytes`.
  Uint8List get data {
    _check();
    final len = calloc<Int32>();
    final code = calloc<Int32>();
    try {
      final p = _n.renderedImageData(_handle, len, code);
      if (p == nullptr) throw PdfOxideError(code.value, 'renderedImageData');
      final out =
          Uint8List.fromList(p.asTypedList(len.value < 0 ? 0 : len.value));
      _n.freeBytes(p);
      return out;
    } finally {
      calloc.free(len);
      calloc.free(code);
    }
  }

  /// Write the encoded image to [path] using the native saver.
  void save(String path) {
    _check();
    final c = path.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.renderedImageSave(_handle, c, code) != 0) {
        throw PdfOxideError(code.value, 'saveRenderedImage');
      }
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.renderedImageFree(_handle);
      _handle = nullptr;
    }
  }
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

  static PdfDocument _openOffice(_OpenBytesD fn, Uint8List data, String op) {
    final buf = calloc<Uint8>(data.isEmpty ? 1 : data.length);
    if (data.isNotEmpty) buf.asTypedList(data.length).setAll(0, data);
    final code = calloc<Int32>();
    try {
      final h = fn(buf, data.length, code);
      if (h == nullptr) throw PdfOxideError(code.value, op);
      return PdfDocument._(h);
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  /// Open a document from DOCX (Office Open XML) bytes.
  static PdfDocument openFromDocxBytes(Uint8List data) =>
      _openOffice(_n.openFromDocx, data, 'openFromDocxBytes');

  /// Open a document from PPTX bytes.
  static PdfDocument openFromPptxBytes(Uint8List data) =>
      _openOffice(_n.openFromPptx, data, 'openFromPptxBytes');

  /// Open a document from XLSX bytes.
  static PdfDocument openFromXlsxBytes(Uint8List data) =>
      _openOffice(_n.openFromXlsx, data, 'openFromXlsxBytes');

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

  /// Embedded fonts on 0-based [page] serialized to JSON.
  String embeddedFontsJson(int page) {
    final list = _openList(_n.extractFonts, page, 'embeddedFontsJson');
    final code = calloc<Int32>();
    try {
      return _takeString(
          _n.fontsToJson(list, code), code.value, 'embeddedFontsJson');
    } finally {
      _n.fontListFree(list);
      calloc.free(code);
    }
  }

  /// The point size of each embedded font on 0-based [page].
  List<double> embeddedFontSizes(int page) {
    final list = _openList(_n.extractFonts, page, 'embeddedFontSizes');
    final code = calloc<Int32>();
    try {
      final n = _n.fontCount(list);
      final out = <double>[];
      for (var i = 0; i < n; i++) {
        final s = _n.fontGetSize(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'embeddedFontSizes');
        }
        out.add(s);
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

  /// Read a single highlight-annotation quad [quadIndex] from annotation [i].
  QuadPoint _quadPoint(Pointer<Void> list, int i, int quadIndex, String op) {
    final x1 = calloc<Float>();
    final y1 = calloc<Float>();
    final x2 = calloc<Float>();
    final y2 = calloc<Float>();
    final x3 = calloc<Float>();
    final y3 = calloc<Float>();
    final x4 = calloc<Float>();
    final y4 = calloc<Float>();
    final code = calloc<Int32>();
    try {
      _n.highlightQuadPoint(
          list, i, quadIndex, x1, y1, x2, y2, x3, y3, x4, y4, code);
      if (code.value != 0) throw PdfOxideError(code.value, op);
      return QuadPoint(x1.value, y1.value, x2.value, y2.value, x3.value,
          y3.value, x4.value, y4.value);
    } finally {
      calloc.free(x1);
      calloc.free(y1);
      calloc.free(x2);
      calloc.free(y2);
      calloc.free(x3);
      calloc.free(y3);
      calloc.free(x4);
      calloc.free(y4);
      calloc.free(code);
    }
  }

  /// Extended annotation attributes on 0-based [page] (flags, dates, colour,
  /// link URIs, icon names and highlight quad points).
  List<AnnotationDetails> pageAnnotationDetails(int page) {
    final list =
        _openList(_n.extractAnnotations, page, 'pageAnnotationDetails');
    const op = 'pageAnnotationDetails';
    final code = calloc<Int32>();
    try {
      final n = _n.annotationCount(list);
      final out = <AnnotationDetails>[];
      for (var i = 0; i < n; i++) {
        final type =
            _takeString(_n.annotationGetType(list, i, code), code.value, op);
        final subtype =
            _takeString(_n.annotationGetSubtype(list, i, code), code.value, op);
        final content =
            _takeString(_n.annotationGetContent(list, i, code), code.value, op);
        final rect = _bbox(_n.annotationGetRect, list, i, op);
        final color = _n.annGetColor(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, op);
        final creationDate = _n.annGetCreationDate(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, op);
        final modificationDate = _n.annGetModificationDate(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, op);
        final hidden = _n.annIsHidden(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, op);
        final markedDeleted = _n.annIsMarkedDeleted(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, op);
        final printable = _n.annIsPrintable(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, op);
        final readOnly = _n.annIsReadOnly(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, op);
        final linkUri =
            _takeString(_n.linkGetUri(list, i, code), code.value, op);
        final iconName =
            _takeString(_n.textAnnotGetIcon(list, i, code), code.value, op);
        final quadCount = _n.highlightQuadCount(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, op);
        final quads = <QuadPoint>[];
        for (var q = 0; q < quadCount; q++) {
          quads.add(_quadPoint(list, i, q, op));
        }
        out.add(AnnotationDetails(
            type,
            subtype,
            content,
            rect,
            color,
            creationDate,
            modificationDate,
            hidden,
            markedDeleted,
            printable,
            readOnly,
            linkUri,
            iconName,
            quads));
      }
      return out;
    } finally {
      _n.annotationListFree(list);
      calloc.free(code);
    }
  }

  /// Annotations on 0-based [page] serialized to JSON.
  String annotationsToJson(int page) {
    final list = _openList(_n.extractAnnotations, page, 'annotationsToJson');
    final code = calloc<Int32>();
    try {
      return _takeString(
          _n.annotationsToJson(list, code), code.value, 'annotationsToJson');
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

  /// Search a single 0-based [page] for [term] and serialize the hits to JSON.
  String searchResultsToJson(int page, String term, bool caseSensitive) {
    _check();
    final cTerm = term.toNativeUtf8();
    final code = calloc<Int32>();
    Pointer<Void> list = nullptr;
    try {
      list = _n.searchPage(_handle, page, cTerm, caseSensitive, code);
      if (list == nullptr)
        throw PdfOxideError(code.value, 'searchResultsToJson');
      return _takeString(_n.searchResultsToJson(list, code), code.value,
          'searchResultsToJson');
    } finally {
      if (list != nullptr) _n.searchResultFree(list);
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

  // ── page rendering (Phase 3) ───────────────────────────────────────────────

  /// Render 0-based [pageIndex] to a [RenderedImage]. [format] is an image
  /// format (0 = PNG, the default).
  RenderedImage renderPage(int pageIndex, [int format = 0]) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.renderPage(_handle, pageIndex, format, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'renderPage');
      return RenderedImage._(h);
    } finally {
      calloc.free(code);
    }
  }

  /// Render 0-based [pageIndex] at the given [zoom] factor. [format] is an
  /// image format (0 = PNG, the default).
  RenderedImage renderPageZoom(int pageIndex, double zoom, [int format = 0]) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.renderPageZoom(_handle, pageIndex, zoom, format, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'renderPageZoom');
      return RenderedImage._(h);
    } finally {
      calloc.free(code);
    }
  }

  /// Render a thumbnail of 0-based [pageIndex] fitting within [size] pixels.
  /// [format] is an image format (0 = PNG, the default).
  RenderedImage renderPageThumbnail(int pageIndex, int size, [int format = 0]) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.renderPageThumbnail(_handle, pageIndex, size, format, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'renderPageThumbnail');
      return RenderedImage._(h);
    } finally {
      calloc.free(code);
    }
  }

  // ── Phase 7: render variants ───────────────────────────────────────────────

  /// Render 0-based [pageIndex] with the full RenderOptions surface.
  ///
  /// [background] channels are 0.0..1.0; set [transparentBackground] to drop the
  /// fill. [format] 0=PNG 1=JPEG; [dpi] resolution; [jpegQuality] 1..100.
  RenderedImage renderPageWithOptions(
    int pageIndex, {
    int dpi = 150,
    int format = 0,
    double backgroundR = 1.0,
    double backgroundG = 1.0,
    double backgroundB = 1.0,
    double backgroundA = 1.0,
    bool transparentBackground = false,
    bool renderAnnotations = true,
    int jpegQuality = 90,
  }) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.renderPageWithOptions(
        _handle,
        pageIndex,
        dpi,
        format,
        backgroundR,
        backgroundG,
        backgroundB,
        backgroundA,
        transparentBackground ? 1 : 0,
        renderAnnotations ? 1 : 0,
        jpegQuality,
        code,
      );
      if (h == nullptr) {
        throw PdfOxideError(code.value, 'renderPageWithOptions');
      }
      return RenderedImage._(h);
    } finally {
      calloc.free(code);
    }
  }

  /// Like [renderPageWithOptions] plus suppression of the named OCG layers in
  /// [excludedLayers]. Pass an empty list to disable filtering.
  RenderedImage renderPageWithOptionsEx(
    int pageIndex, {
    int dpi = 150,
    int format = 0,
    double backgroundR = 1.0,
    double backgroundG = 1.0,
    double backgroundB = 1.0,
    double backgroundA = 1.0,
    bool transparentBackground = false,
    bool renderAnnotations = true,
    int jpegQuality = 90,
    List<String> excludedLayers = const [],
  }) {
    _check();
    final code = calloc<Int32>();
    final n = excludedLayers.length;
    final arr = n == 0 ? nullptr : calloc<Pointer<Utf8>>(n);
    for (var i = 0; i < n; i++) {
      arr[i] = excludedLayers[i].toNativeUtf8();
    }
    try {
      final h = _n.renderPageWithOptionsEx(
        _handle,
        pageIndex,
        dpi,
        format,
        backgroundR,
        backgroundG,
        backgroundB,
        backgroundA,
        transparentBackground ? 1 : 0,
        renderAnnotations ? 1 : 0,
        jpegQuality,
        arr,
        n,
        code,
      );
      if (h == nullptr) {
        throw PdfOxideError(code.value, 'renderPageWithOptionsEx');
      }
      return RenderedImage._(h);
    } finally {
      for (var i = 0; i < n; i++) {
        calloc.free(arr[i]);
      }
      if (arr != nullptr) calloc.free(arr);
      calloc.free(code);
    }
  }

  /// Render a rectangular region of 0-based [pageIndex]. Crop coordinates are in
  /// PDF user-space points (origin bottom-left). [format] 0=PNG 1=JPEG.
  RenderedImage renderPageRegion(
    int pageIndex,
    double cropX,
    double cropY,
    double cropWidth,
    double cropHeight, [
    int format = 0,
  ]) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.renderPageRegion(_handle, pageIndex, cropX, cropY, cropWidth,
          cropHeight, format, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'renderPageRegion');
      return RenderedImage._(h);
    } finally {
      calloc.free(code);
    }
  }

  /// Render 0-based [pageIndex] to fit inside [width]×[height] pixels,
  /// preserving aspect ratio. [format] 0=PNG 1=JPEG.
  RenderedImage renderPageFit(int pageIndex, int width, int height,
      [int format = 0]) {
    _check();
    final code = calloc<Int32>();
    try {
      final h =
          _n.renderPageFit(_handle, pageIndex, width, height, format, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'renderPageFit');
      return RenderedImage._(h);
    } finally {
      calloc.free(code);
    }
  }

  /// Render 0-based [pageIndex] to a raw premultiplied RGBA8888 [RenderedImage]
  /// at [dpi]. The pixel buffer is row-major, top-left origin; read the raw
  /// bytes via [RenderedImage.data].
  RenderedImage renderPageRaw(int pageIndex, [int dpi = 150]) {
    _check();
    final w = calloc<Int32>();
    final h = calloc<Int32>();
    final code = calloc<Int32>();
    try {
      final img = _n.renderPageRaw(_handle, pageIndex, dpi, w, h, code);
      if (img == nullptr) throw PdfOxideError(code.value, 'renderPageRaw');
      return RenderedImage._(img);
    } finally {
      calloc.free(w);
      calloc.free(h);
      calloc.free(code);
    }
  }

  /// Estimate the render time (in milliseconds) for 0-based [pageIndex].
  int estimateRenderTime(int pageIndex) {
    _check();
    final code = calloc<Int32>();
    try {
      final ms = _n.estimateRenderTime(_handle, pageIndex, code);
      if (code.value != 0)
        throw PdfOxideError(code.value, 'estimateRenderTime');
      return ms;
    } finally {
      calloc.free(code);
    }
  }

  // ── Phase 7: page getters ──────────────────────────────────────────────────

  /// Width of 0-based [pageIndex] in PDF user-space points.
  double pageWidth(int pageIndex) {
    _check();
    final code = calloc<Int32>();
    try {
      final v = _n.pageGetWidth(_handle, pageIndex, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'pageWidth');
      return v;
    } finally {
      calloc.free(code);
    }
  }

  /// Height of 0-based [pageIndex] in PDF user-space points.
  double pageHeight(int pageIndex) {
    _check();
    final code = calloc<Int32>();
    try {
      final v = _n.pageGetHeight(_handle, pageIndex, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'pageHeight');
      return v;
    } finally {
      calloc.free(code);
    }
  }

  /// Rotation (degrees) of 0-based [pageIndex].
  int pageRotation(int pageIndex) {
    _check();
    final code = calloc<Int32>();
    try {
      final v = _n.pageGetRotation(_handle, pageIndex, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'pageRotation');
      return v;
    } finally {
      calloc.free(code);
    }
  }

  /// Extract the layout elements of 0-based [pageIndex] as an [ElementList].
  /// Call [ElementList.close] when done (or rely on the finalizer).
  ElementList pageElements(int pageIndex) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.pageGetElements(_handle, pageIndex, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'pageElements');
      return ElementList._(h);
    } finally {
      calloc.free(code);
    }
  }

  // ── Phase 7: OCR ───────────────────────────────────────────────────────────

  /// Whether 0-based [pageIndex] needs OCR (i.e. is scanned/hybrid).
  bool pageNeedsOcr(int pageIndex) {
    _check();
    final code = calloc<Int32>();
    try {
      final v = _n.ocrPageNeedsOcr(_handle, pageIndex, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'pageNeedsOcr');
      return v;
    } finally {
      calloc.free(code);
    }
  }

  /// Extract text from 0-based [pageIndex] using OCR. [engine] may be null (then
  /// native text extraction only is used).
  String ocrExtractText(int pageIndex, [OcrEngine? engine]) {
    _check();
    final code = calloc<Int32>();
    try {
      final enginePtr = engine?._handlePtr ?? nullptr;
      return _takeString(_n.ocrExtractText(_handle, pageIndex, enginePtr, code),
          code.value, 'ocrExtractText');
    } finally {
      calloc.free(code);
    }
  }

  /// A lightweight view of a single 0-based page. The returned [Page] keeps a
  /// reference to this document and must not be used after [close].
  Page page(int index) => Page._(this, index);

  // ── Phase 6: conformance validation ─────────────────────────────────────────

  /// Validate against PDF/A at the given [level] (an integer conformance code).
  PdfAResults validatePdfA(int level) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.validatePdfALevel(_handle, level, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'validatePdfA');
      return PdfAResults._(h);
    } finally {
      calloc.free(code);
    }
  }

  /// Validate against PDF/UA accessibility at the given [level].
  UaResults validatePdfUa(int level) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.validatePdfUa(_handle, level, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'validatePdfUa');
      return UaResults._(h);
    } finally {
      calloc.free(code);
    }
  }

  /// Validate against PDF/X at the given [level] (an integer conformance code).
  PdfXResults validatePdfX(int level) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.validatePdfXLevel(_handle, level, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'validatePdfX');
      return PdfXResults._(h);
    } finally {
      calloc.free(code);
    }
  }

  // ── Phase 8: PDF/A conversion ──────────────────────────────────────────────

  /// Convert this document in place to PDF/A at the given [level] (an integer
  /// conformance code). Returns whether conversion succeeded.
  bool convertToPdfA(int level) {
    _check();
    final code = calloc<Int32>();
    try {
      final ok = _n.convertToPdfA(_handle, level, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'convertToPdfA');
      return ok;
    } finally {
      calloc.free(code);
    }
  }

  // ── Phase 8: office export ─────────────────────────────────────────────────

  Uint8List _docBytes(_DocBytesD fn, String op) {
    _check();
    final len = calloc<IntPtr>();
    final code = calloc<Int32>();
    try {
      final p = fn(_handle, len, code);
      return _takeBytes(p, len.value, code.value, op);
    } finally {
      calloc.free(len);
      calloc.free(code);
    }
  }

  /// Convert this document to a DOCX (Office Open XML) byte buffer.
  Uint8List toDocx() => _docBytes(_n.toDocx, 'toDocx');

  /// Convert this document to a PPTX byte buffer.
  Uint8List toPptx() => _docBytes(_n.toPptx, 'toPptx');

  /// Convert this document to an XLSX byte buffer.
  Uint8List toXlsx() => _docBytes(_n.toXlsx, 'toXlsx');

  /// The original source bytes this document was loaded from.
  Uint8List sourceBytes() => _docBytes(_n.getSourceBytes, 'sourceBytes');

  // ── Phase 8: in-rect extraction ────────────────────────────────────────────

  String _strRect(_RectStrD fn, int page, double x, double y, double w,
      double h, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(fn(_handle, page, x, y, w, h, code), code.value, op);
    } finally {
      calloc.free(code);
    }
  }

  Pointer<Void> _openRectList(_RectListD fn, int page, double x, double y,
      double w, double h, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      final list = fn(_handle, page, x, y, w, h, code);
      if (list == nullptr) throw PdfOxideError(code.value, op);
      return list;
    } finally {
      calloc.free(code);
    }
  }

  /// Extract plain text within rectangle [x],[y],[w],[h] on 0-based [page].
  String extractTextInRect(int page, double x, double y, double w, double h) =>
      _strRect(_n.extractTextInRect, page, x, y, w, h, 'extractTextInRect');

  /// Extract [Word]s within rectangle [x],[y],[w],[h] on 0-based [page].
  List<Word> extractWordsInRect(
      int page, double x, double y, double w, double h) {
    final list = _openRectList(
        _n.extractWordsInRect, page, x, y, w, h, 'extractWordsInRect');
    final code = calloc<Int32>();
    try {
      final n = _n.wordCount(list);
      final out = <Word>[];
      for (var i = 0; i < n; i++) {
        final text = _takeString(
            _n.wordGetText(list, i, code), code.value, 'extractWordsInRect');
        final bbox = _bbox(_n.wordGetBbox, list, i, 'extractWordsInRect');
        final fontName = _takeString(_n.wordGetFontName(list, i, code),
            code.value, 'extractWordsInRect');
        final fontSize = _n.wordGetFontSize(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'extractWordsInRect');
        }
        final bold = _n.wordIsBold(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'extractWordsInRect');
        }
        out.add(Word(text, bbox, fontName, fontSize, bold));
      }
      return out;
    } finally {
      _n.wordListFree(list);
      calloc.free(code);
    }
  }

  /// Extract [TextLine]s within rectangle [x],[y],[w],[h] on 0-based [page].
  List<TextLine> extractLinesInRect(
      int page, double x, double y, double w, double h) {
    final list = _openRectList(
        _n.extractLinesInRect, page, x, y, w, h, 'extractLinesInRect');
    final code = calloc<Int32>();
    try {
      final n = _n.lineCount(list);
      final out = <TextLine>[];
      for (var i = 0; i < n; i++) {
        final text = _takeString(
            _n.lineGetText(list, i, code), code.value, 'extractLinesInRect');
        final bbox = _bbox(_n.lineGetBbox, list, i, 'extractLinesInRect');
        final wordCount = _n.lineGetWordCount(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'extractLinesInRect');
        }
        out.add(TextLine(text, bbox, wordCount));
      }
      return out;
    } finally {
      _n.lineListFree(list);
      calloc.free(code);
    }
  }

  /// Extract [Table]s within rectangle [x],[y],[w],[h] on 0-based [page].
  List<Table> extractTablesInRect(
      int page, double x, double y, double w, double h) {
    final list = _openRectList(
        _n.extractTablesInRect, page, x, y, w, h, 'extractTablesInRect');
    final code = calloc<Int32>();
    try {
      final n = _n.tableCount(list);
      final out = <Table>[];
      for (var i = 0; i < n; i++) {
        final rowCount = _n.tableGetRowCount(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'extractTablesInRect');
        }
        final colCount = _n.tableGetColCount(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'extractTablesInRect');
        }
        final hasHeader = _n.tableHasHeader(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'extractTablesInRect');
        }
        final cells = <String>[];
        for (var r = 0; r < rowCount; r++) {
          for (var c = 0; c < colCount; c++) {
            cells.add(_takeString(_n.tableGetCellText(list, i, r, c, code),
                code.value, 'extractTablesInRect'));
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

  /// Extract [Image]s within rectangle [x],[y],[w],[h] on 0-based [page].
  List<Image> extractImagesInRect(
      int page, double x, double y, double w, double h) {
    final list = _openRectList(
        _n.extractImagesInRect, page, x, y, w, h, 'extractImagesInRect');
    final code = calloc<Int32>();
    final len = calloc<Int32>();
    try {
      final n = _n.imageCount(list);
      final out = <Image>[];
      for (var i = 0; i < n; i++) {
        final width = _n.imageGetWidth(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'extractImagesInRect');
        }
        final height = _n.imageGetHeight(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'extractImagesInRect');
        }
        final bpc = _n.imageGetBitsPerComponent(list, i, code);
        if (code.value != 0) {
          throw PdfOxideError(code.value, 'extractImagesInRect');
        }
        final format = _takeString(_n.imageGetFormat(list, i, code), code.value,
            'extractImagesInRect');
        final colorspace = _takeString(_n.imageGetColorspace(list, i, code),
            code.value, 'extractImagesInRect');
        final p = _n.imageGetData(list, i, len, code);
        if (p == nullptr)
          throw PdfOxideError(code.value, 'extractImagesInRect');
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

  // ── Phase 8: auto extraction / classification ──────────────────────────────

  /// Auto-extract text from 0-based [page] (native + image OCR as needed).
  String extractTextAuto(int page) =>
      _strPage(_n.extractTextAuto, page, 'extractTextAuto');

  /// Auto-extract a single 0-based [page] with optional [optionsJson].
  String extractPageAuto(int page, [String optionsJson = '']) {
    _check();
    final c = optionsJson.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      return _takeString(_n.extractPageAuto(_handle, page, c, code), code.value,
          'extractPageAuto');
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  /// Extract all pages' text as a single string.
  String extractAllText() {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(
          _n.extractAllText(_handle, code), code.value, 'extractAllText');
    } finally {
      calloc.free(code);
    }
  }

  /// Classify a single 0-based [page]; returns a JSON description.
  String classifyPage(int page) =>
      _strPage(_n.classifyPage, page, 'classifyPage');

  /// Classify the whole document; returns a JSON description.
  String classifyDocument() {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(
          _n.classifyDocument(_handle, code), code.value, 'classifyDocument');
    } finally {
      calloc.free(code);
    }
  }

  // ── Phase 8: header / footer / artifact removal ────────────────────────────

  int _erasePage(_DeI32I32D fn, int page, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      final n = fn(_handle, page, code);
      if (code.value != 0) throw PdfOxideError(code.value, op);
      return n;
    } finally {
      calloc.free(code);
    }
  }

  int _remove(_RemoveD fn, double threshold, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      final n = fn(_handle, threshold, code);
      if (code.value != 0) throw PdfOxideError(code.value, op);
      return n;
    } finally {
      calloc.free(code);
    }
  }

  /// Erase the header region of 0-based [page]; returns items erased.
  int eraseHeader(int page) => _erasePage(_n.eraseHeader, page, 'eraseHeader');

  /// Erase the footer region of 0-based [page]; returns items erased.
  int eraseFooter(int page) => _erasePage(_n.eraseFooter, page, 'eraseFooter');

  /// Erase marked artifacts on 0-based [page]; returns items erased.
  int eraseArtifacts(int page) =>
      _erasePage(_n.eraseArtifacts, page, 'eraseArtifacts');

  /// Remove repeating headers document-wide using detection [threshold].
  int removeHeaders([double threshold = 0.5]) =>
      _remove(_n.removeHeaders, threshold, 'removeHeaders');

  /// Remove repeating footers document-wide using detection [threshold].
  int removeFooters([double threshold = 0.5]) =>
      _remove(_n.removeFooters, threshold, 'removeFooters');

  /// Remove artifacts document-wide using detection [threshold].
  int removeArtifacts([double threshold = 0.5]) =>
      _remove(_n.removeArtifacts, threshold, 'removeArtifacts');

  // ── Phase 8: forms ─────────────────────────────────────────────────────────

  /// The interactive (AcroForm) [FormField]s in this document (may be empty).
  List<FormField> getFormFields() {
    _check();
    final code = calloc<Int32>();
    final list = _n.getFormFields(_handle, code);
    if (list == nullptr) throw PdfOxideError(code.value, 'getFormFields');
    try {
      final n = _n.formFieldCount(list);
      final out = <FormField>[];
      for (var i = 0; i < n; i++) {
        final name = _takeString(
            _n.formFieldGetName(list, i, code), code.value, 'getFormFields');
        final value = _takeString(
            _n.formFieldGetValue(list, i, code), code.value, 'getFormFields');
        final type = _takeString(
            _n.formFieldGetType(list, i, code), code.value, 'getFormFields');
        final readonly = _n.formFieldIsReadonly(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'getFormFields');
        final required = _n.formFieldIsRequired(list, i, code);
        if (code.value != 0) throw PdfOxideError(code.value, 'getFormFields');
        out.add(FormField(name, value, type, readonly, required));
      }
      return out;
    } finally {
      _n.formFieldListFree(list);
      calloc.free(code);
    }
  }

  /// Export AcroForm data in [formatType] (0=FDF, 1=XFDF, 2=JSON) as bytes.
  Uint8List exportFormDataToBytes([int formatType = 0]) {
    _check();
    final len = calloc<IntPtr>();
    final code = calloc<Int32>();
    try {
      final p = _n.exportFormData(_handle, formatType, len, code);
      return _takeBytes(p, len.value, code.value, 'exportFormDataToBytes');
    } finally {
      calloc.free(len);
      calloc.free(code);
    }
  }

  /// Import AcroForm data from the FDF/XFDF/JSON file at [dataPath].
  void importFormData(String dataPath) {
    _check();
    final c = dataPath.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.importFormData(_handle, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'importFormData');
      }
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  /// Import AcroForm field values from a file (FDF/XFDF) at [filename].
  /// Returns whether anything was imported.
  bool importFormFromFile(String filename) {
    _check();
    final c = filename.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final ok = _n.formImportFromFile(_handle, c, code);
      if (code.value != 0)
        throw PdfOxideError(code.value, 'importFormFromFile');
      return ok;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  // ── Phase 8: document structure / metadata ─────────────────────────────────

  String _docStr(_TextAllD fn, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(fn(_handle, code), code.value, op);
    } finally {
      calloc.free(code);
    }
  }

  /// The document outline (bookmarks) as JSON.
  String getOutline() => _docStr(_n.getOutline, 'getOutline');

  /// The document page labels as JSON.
  String getPageLabels() => _docStr(_n.getPageLabels, 'getPageLabels');

  /// The document XMP metadata (XML), if present.
  String getXmpMetadata() => _docStr(_n.getXmpMetadata, 'getXmpMetadata');

  /// Whether this document contains an XFA form.
  bool hasXfa() {
    _check();
    return _n.hasXfa(_handle);
  }

  /// Plan a split-by-bookmarks operation, returning a JSON plan. [optionsJson]
  /// configures bookmark level / naming.
  String planSplitByBookmarks([String optionsJson = '']) {
    _check();
    final c = optionsJson.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      return _takeString(_n.planSplitByBookmarks(_handle, c, code), code.value,
          'planSplitByBookmarks');
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  // ── Phase 8: document-level signatures ─────────────────────────────────────

  /// Digitally sign this document with [certificate], embedding [reason] and
  /// [location]. Returns the C-ABI status code.
  int sign(Certificate certificate,
      {String reason = '', String location = ''}) {
    _check();
    final cReason = reason.toNativeUtf8();
    final cLoc = location.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final rc = _n.docSign(_handle, certificate.handle, cReason, cLoc, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'sign');
      return rc;
    } finally {
      calloc.free(cReason);
      calloc.free(cLoc);
      calloc.free(code);
    }
  }

  /// The number of signatures present in this document.
  int getSignatureCount() {
    _check();
    final code = calloc<Int32>();
    try {
      final n = _n.docGetSignatureCount(_handle, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'getSignatureCount');
      return n;
    } finally {
      calloc.free(code);
    }
  }

  /// The [SignatureInfo] at 0-based [index].
  SignatureInfo getSignature(int index) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.docGetSignature(_handle, index, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'getSignature');
      return SignatureInfo.fromHandle(h);
    } finally {
      calloc.free(code);
    }
  }

  /// Verify all signatures; returns the aggregate C-ABI status.
  int verifyAllSignatures() {
    _check();
    final code = calloc<Int32>();
    try {
      final rc = _n.docVerifyAllSignatures(_handle, code);
      if (code.value != 0) {
        throw PdfOxideError(code.value, 'verifyAllSignatures');
      }
      return rc;
    } finally {
      calloc.free(code);
    }
  }

  /// Whether any signature in this document carries a timestamp.
  bool hasTimestamp() {
    _check();
    final code = calloc<Int32>();
    try {
      final r = _n.docHasTimestamp(_handle, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'hasTimestamp');
      return r != 0;
    } finally {
      calloc.free(code);
    }
  }

  /// The document's [Dss] (document security store).
  Dss getDss() {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.docGetDss(_handle, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'getDss');
      return Dss.fromHandle(h);
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

  // ── Phase 7: image / HTML+CSS constructors ─────────────────────────────────

  /// Build a single-page PDF wrapping the image file at [path].
  static Pdf fromImage(String path) => _from(_n.fromImage, path, 'fromImage');

  /// Build a single-page PDF wrapping the in-memory image [data].
  static Pdf fromImageBytes(Uint8List data) {
    final buf = calloc<Uint8>(data.isEmpty ? 1 : data.length);
    if (data.isNotEmpty) buf.asTypedList(data.length).setAll(0, data);
    final code = calloc<Int32>();
    try {
      final h = _n.fromImageBytes(buf, data.length, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'fromImageBytes');
      return Pdf._(h);
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  /// Build a PDF from [html] + [css] with an optional single embedded
  /// [fontBytes] (TTF/OTF).
  static Pdf fromHtmlCss(String html, String css, [Uint8List? fontBytes]) {
    final cHtml = html.toNativeUtf8();
    final cCss = css.toNativeUtf8();
    final font = fontBytes ?? Uint8List(0);
    final buf = calloc<Uint8>(font.isEmpty ? 1 : font.length);
    if (font.isNotEmpty) buf.asTypedList(font.length).setAll(0, font);
    final code = calloc<Int32>();
    try {
      final h = _n.fromHtmlCss(cHtml, cCss, buf, font.length, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'fromHtmlCss');
      return Pdf._(h);
    } finally {
      calloc.free(cHtml);
      calloc.free(cCss);
      calloc.free(buf);
      calloc.free(code);
    }
  }

  /// Build a PDF from [html] + [css] with a multi-font cascade. [families] and
  /// [fonts] are parallel lists of the same length (family name -> font bytes).
  static Pdf fromHtmlCssWithFonts(
    String html,
    String css,
    List<String> families,
    List<Uint8List> fonts,
  ) {
    if (families.length != fonts.length) {
      throw ArgumentError('families and fonts must have equal length');
    }
    final n = families.length;
    final cHtml = html.toNativeUtf8();
    final cCss = css.toNativeUtf8();
    final famArr = n == 0 ? nullptr : calloc<Pointer<Utf8>>(n);
    for (var i = 0; i < n; i++) {
      famArr[i] = families[i].toNativeUtf8();
    }
    final fontArr = _ByteArrayArray(fonts);
    final code = calloc<Int32>();
    try {
      final h = _n.fromHtmlCssWithFonts(
          cHtml, cCss, famArr, fontArr.ptrs, fontArr.lens, n, code);
      if (h == nullptr) {
        throw PdfOxideError(code.value, 'fromHtmlCssWithFonts');
      }
      return Pdf._(h);
    } finally {
      calloc.free(cHtml);
      calloc.free(cCss);
      for (var i = 0; i < n; i++) {
        calloc.free(famArr[i]);
      }
      if (famArr != nullptr) calloc.free(famArr);
      fontArr.free();
      calloc.free(code);
    }
  }

  void _check() {
    if (_handle == nullptr) throw StateError('Pdf is closed');
  }

  /// The page count of this builder-produced PDF via the `pdf_get_page_count`
  /// entry point.
  int get pageCount {
    _check();
    final code = calloc<Int32>();
    try {
      final n = _n.pdfGetPageCount(_handle, code);
      if (n < 0) throw PdfOxideError(code.value, 'pageCount');
      return n;
    } finally {
      calloc.free(code);
    }
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

/// An opened PDF for editing — wraps the native `DocumentEditor` handle.
///
/// Mirrors [PdfDocument]/[Pdf]: opened via the static [open]/[openFromBytes]
/// factories, owns the native handle (freed on [close], by the finalizer, or
/// implicitly on dealloc), and shares the same error/string/byte conventions.
/// Page indices are 0-based. C status codes are 0 = success; any non-zero
/// status or set `error_code` raises [PdfOxideError]. The `is_*` queries return
/// `bool` (1 = true). Call [close] when done (or rely on the finalizer).
class DocumentEditor implements Finalizable {
  DocumentEditor._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_DeFreeC>>('document_editor_free'));
  Pointer<Void> _handle;

  /// Open a PDF for editing from a filesystem path.
  static DocumentEditor open(String path) {
    final cPath = path.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.deOpen(cPath, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'openEditor');
      return DocumentEditor._(h);
    } finally {
      calloc.free(cPath);
      calloc.free(code);
    }
  }

  /// Open a PDF for editing from in-memory bytes.
  static DocumentEditor openFromBytes(Uint8List data) {
    final buf = calloc<Uint8>(data.length);
    buf.asTypedList(data.length).setAll(0, data);
    final code = calloc<Int32>();
    try {
      final h = _n.deOpenBytes(buf, data.length, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'openFromBytes');
      return DocumentEditor._(h);
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  void _check() {
    if (_handle == nullptr) throw StateError('DocumentEditor is closed');
  }

  // ── small helpers (shared shapes) ──────────────────────────────────────────

  /// Read an owned char* + free_string.
  String _str(_DeStrD fn, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(fn(_handle, code), code.value, op);
    } finally {
      calloc.free(code);
    }
  }

  /// Set an owned UTF-8 string-valued field (status return).
  void _setStr(_DeSetStrD fn, String value, String op) {
    _check();
    final c = value.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (fn(_handle, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  /// Run a `(handle, error_code) -> int32` status function.
  void _status0(_DeI32D fn, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      if (fn(_handle, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Run a `(handle, int32, error_code) -> int32` status function.
  void _statusI32(_DeI32I32D fn, int a, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      if (fn(_handle, a, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Run a `(handle, usize, error_code) -> int32` status function.
  void _statusUsize(_DeUsizeI32D fn, int page, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      if (fn(_handle, page, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Read a double-quad box out-param (x, y, w, h) into a [Bbox].
  Bbox _getBox(_DeGetBoxD fn, int page, String op) {
    _check();
    final x = calloc<Double>();
    final y = calloc<Double>();
    final w = calloc<Double>();
    final h = calloc<Double>();
    final code = calloc<Int32>();
    try {
      if (fn(_handle, page, x, y, w, h, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
      return Bbox(x.value, y.value, w.value, h.value);
    } finally {
      calloc.free(x);
      calloc.free(y);
      calloc.free(w);
      calloc.free(h);
      calloc.free(code);
    }
  }

  /// Set a double-quad box (x, y, w, h).
  void _setBox(_DeSetBoxD fn, int page, Bbox box, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      if (fn(_handle, page, box.x, box.y, box.width, box.height, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Read an owned uint8* byte buffer with a `uintptr_t` out-len, copy into
  /// Dart, and free via `free_bytes`.
  Uint8List _bytesOut(
      Pointer<Uint8> Function(Pointer<IntPtr>, Pointer<Int32>) call,
      String op) {
    final len = calloc<IntPtr>();
    final code = calloc<Int32>();
    try {
      final p = call(len, code);
      if (p == nullptr) throw PdfOxideError(code.value, op);
      final out =
          Uint8List.fromList(p.asTypedList(len.value < 0 ? 0 : len.value));
      _n.freeBytes(p);
      return out;
    } finally {
      calloc.free(len);
      calloc.free(code);
    }
  }

  // ── document metadata / queries ────────────────────────────────────────────

  int get pageCount {
    _check();
    final code = calloc<Int32>();
    try {
      final n = _n.dePageCount(_handle, code);
      if (n < 0 || code.value != 0)
        throw PdfOxideError(code.value, 'pageCount');
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
      _n.deGetVersion(_handle, maj, min);
      return PdfVersion(maj.value, min.value);
    } finally {
      calloc.free(maj);
      calloc.free(min);
    }
  }

  /// `true` if the editor has unsaved modifications.
  bool isModified() {
    _check();
    return _n.deIsModified(_handle);
  }

  /// Source path the editor was opened from (empty/absent allowed).
  String getSourcePath() => _str(_n.deGetSourcePath, 'getSourcePath');

  String getProducer() => _str(_n.deGetProducer, 'getProducer');
  void setProducer(String value) =>
      _setStr(_n.deSetProducer, value, 'setProducer');

  String getCreationDate() => _str(_n.deGetCreationDate, 'getCreationDate');
  void setCreationDate(String dateStr) =>
      _setStr(_n.deSetCreationDate, dateStr, 'setCreationDate');

  // ── page operations ────────────────────────────────────────────────────────

  void deletePage(int pageIndex) =>
      _statusI32(_n.deDeletePage, pageIndex, 'deletePage');

  void movePage(int from, int to) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.deMovePage(_handle, from, to, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'movePage');
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Rotate a single page by [degrees] (additive). Page is `uintptr_t`.
  void rotatePageBy(int page, int degrees) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.deRotatePageBy(_handle, page, degrees, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'rotatePageBy');
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Rotate all pages by [degrees] (relative).
  void rotateAllPages(int degrees) =>
      _statusI32(_n.deRotateAllPages, degrees, 'rotateAllPages');

  /// Set absolute rotation for a page.
  void setPageRotation(int page, int degrees) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.deSetPageRotation(_handle, page, degrees, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'setPageRotation');
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Current rotation (degrees) of a page.
  int getPageRotation(int page) {
    _check();
    final code = calloc<Int32>();
    try {
      final r = _n.deGetPageRotation(_handle, page, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'getPageRotation');
      return r;
    } finally {
      calloc.free(code);
    }
  }

  /// Crop the document margins (points) on every page.
  void cropMargins(double left, double right, double top, double bottom) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.deCropMargins(_handle, left, right, top, bottom, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'cropMargins');
      }
    } finally {
      calloc.free(code);
    }
  }

  Bbox getPageCropBox(int page) =>
      _getBox(_n.deGetPageCropBox, page, 'getPageCropBox');
  void setPageCropBox(int page, Bbox box) =>
      _setBox(_n.deSetPageCropBox, page, box, 'setPageCropBox');
  Bbox getPageMediaBox(int page) =>
      _getBox(_n.deGetPageMediaBox, page, 'getPageMediaBox');
  void setPageMediaBox(int page, Bbox box) =>
      _setBox(_n.deSetPageMediaBox, page, box, 'setPageMediaBox');

  // ── redaction / erase ──────────────────────────────────────────────────────

  void applyAllRedactions() =>
      _status0(_n.deApplyAllRedactions, 'applyAllRedactions');
  void applyPageRedactions(int page) =>
      _statusUsize(_n.deApplyPageRedactions, page, 'applyPageRedactions');

  /// Erase a single rectangular region on [page] (float coords). Page int32.
  void eraseRegion(int page, double x, double y, double w, double h) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.deEraseRegion(_handle, page, x, y, w, h, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'eraseRegion');
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Erase multiple rectangles on [page]. [rects] is a list of (x, y, w, h)
  /// quads flattened to a contiguous f64 array.
  void eraseRegions(int page, List<List<double>> rects) {
    _check();
    final buf = calloc<Double>(rects.length * 4);
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < rects.length; i++) {
        final r = rects[i];
        buf[i * 4 + 0] = r[0];
        buf[i * 4 + 1] = r[1];
        buf[i * 4 + 2] = r[2];
        buf[i * 4 + 3] = r[3];
      }
      if (_n.deEraseRegions(_handle, page, buf, rects.length, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'eraseRegions');
      }
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  void clearEraseRegions(int page) =>
      _statusUsize(_n.deClearEraseRegions, page, 'clearEraseRegions');

  /// `true` if [page] is marked for redaction (1 = true).
  bool isPageMarkedForRedaction(int page) {
    _check();
    final r = _n.deIsPageMarkedForRedaction(_handle, page);
    if (r < 0) throw PdfOxideError(r, 'isPageMarkedForRedaction');
    return r == 1;
  }

  void unmarkPageForRedaction(int page) =>
      _statusUsize(_n.deUnmarkPageForRedaction, page, 'unmarkPageForRedaction');

  // ── flatten (forms + annotations) ──────────────────────────────────────────

  void flattenForms() => _status0(_n.deFlattenForms, 'flattenForms');
  void flattenFormsOnPage(int pageIndex) =>
      _statusI32(_n.deFlattenFormsOnPage, pageIndex, 'flattenFormsOnPage');
  void flattenAnnotations(int page) =>
      _statusI32(_n.deFlattenAnnotations, page, 'flattenAnnotations');
  void flattenAllAnnotations() =>
      _status0(_n.deFlattenAllAnnotations, 'flattenAllAnnotations');

  /// Number of warnings collected during the last form-flatten save.
  int flattenWarningsCount() {
    _check();
    return _n.deFlattenWarningsCount(_handle);
  }

  /// The [index]-th flatten warning message.
  String flattenWarning(int index) {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(_n.deFlattenWarning(_handle, index, code), code.value,
          'flattenWarning');
    } finally {
      calloc.free(code);
    }
  }

  /// `true` if [page] is marked for annotation-flatten (1 = true).
  bool isPageMarkedForFlatten(int page) {
    _check();
    final r = _n.deIsPageMarkedForFlatten(_handle, page);
    if (r < 0) throw PdfOxideError(r, 'isPageMarkedForFlatten');
    return r == 1;
  }

  void unmarkPageForFlatten(int page) =>
      _statusUsize(_n.deUnmarkPageForFlatten, page, 'unmarkPageForFlatten');

  // ── forms ──────────────────────────────────────────────────────────────────

  void setFormFieldValue(String name, String value) {
    _check();
    final cName = name.toNativeUtf8();
    final cValue = value.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.deSetFormFieldValue(_handle, cName, cValue, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'setFormFieldValue');
      }
    } finally {
      calloc.free(cName);
      calloc.free(cValue);
      calloc.free(code);
    }
  }

  // ── merge / convert / embed / extract ──────────────────────────────────────

  /// Merge pages from a PDF on disk into this document.
  void mergeFrom(String sourcePath) {
    _check();
    final c = sourcePath.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.deMergeFrom(_handle, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'mergeFrom');
      }
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  /// Merge pages from an in-memory PDF into this document.
  void mergeFromBytes(Uint8List data) {
    _check();
    final buf = calloc<Uint8>(data.length);
    buf.asTypedList(data.length).setAll(0, data);
    final code = calloc<Int32>();
    try {
      if (_n.deMergeFromBytes(_handle, buf, data.length, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'mergeFromBytes');
      }
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  /// Convert the document to PDF/A in place. [level]: 0=A1b 1=A1a 2=A2b 3=A2a
  /// 4=A2u 5=A3b 6=A3a 7=A3u.
  void convertToPdfA(int level) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.deConvertToPdfA(_handle, level, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'convertToPdfA');
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Embed a file attachment named [name] with the given [data].
  void embedFile(String name, Uint8List data) {
    _check();
    final cName = name.toNativeUtf8();
    final buf = calloc<Uint8>(data.length);
    buf.asTypedList(data.length).setAll(0, data);
    final code = calloc<Int32>();
    try {
      if (_n.deEmbedFile(_handle, cName, buf, data.length, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'embedFile');
      }
    } finally {
      calloc.free(cName);
      calloc.free(buf);
      calloc.free(code);
    }
  }

  /// Extract a subset of 0-based [pages] to a new in-memory PDF.
  Uint8List extractPagesToBytes(List<int> pages) {
    _check();
    final buf = calloc<Int32>(pages.length);
    for (var i = 0; i < pages.length; i++) {
      buf[i] = pages[i];
    }
    try {
      return _bytesOut(
          (len, code) =>
              _n.deExtractPagesToBytes(_handle, buf, pages.length, len, code),
          'extractPagesToBytes');
    } finally {
      calloc.free(buf);
    }
  }

  // ── save ───────────────────────────────────────────────────────────────────

  /// Save the edited document to a filesystem [path].
  void save(String path) {
    _check();
    final c = path.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.deSave(_handle, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'save');
      }
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  /// Save the edited document to an in-memory byte buffer.
  Uint8List saveToBytes() {
    _check();
    return _bytesOut(
        (len, code) => _n.deSaveToBytes(_handle, len, code), 'saveToBytes');
  }

  /// Save to bytes with explicit compression / GC / linearize options.
  Uint8List saveToBytesWithOptions(
      bool compress, bool garbageCollect, bool linearize) {
    _check();
    return _bytesOut(
        (len, code) => _n.deSaveToBytesWithOptions(
            _handle, compress, garbageCollect, linearize, len, code),
        'saveToBytesWithOptions');
  }

  /// Save with AES-256 encryption to a filesystem [path].
  void saveEncrypted(String path, String userPassword, String ownerPassword) {
    _check();
    final cPath = path.toNativeUtf8();
    final cUser = userPassword.toNativeUtf8();
    final cOwner = ownerPassword.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.deSaveEncrypted(_handle, cPath, cUser, cOwner, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'saveEncrypted');
      }
    } finally {
      calloc.free(cPath);
      calloc.free(cUser);
      calloc.free(cOwner);
      calloc.free(code);
    }
  }

  /// Save with AES-256 encryption to an in-memory byte buffer.
  Uint8List saveEncryptedToBytes(String userPassword, String ownerPassword) {
    _check();
    final cUser = userPassword.toNativeUtf8();
    final cOwner = ownerPassword.toNativeUtf8();
    try {
      return _bytesOut(
          (len, code) =>
              _n.deSaveEncryptedToBytes(_handle, cUser, cOwner, len, code),
          'saveEncryptedToBytes');
    } finally {
      calloc.free(cUser);
      calloc.free(cOwner);
    }
  }

  // ── Phase 7: redaction ─────────────────────────────────────────────────────

  /// Queue a redaction rectangle on 0-based [page]. Coordinates and the overlay
  /// fill colour ([r]/[g]/[b], 0.0..1.0) are in page user-space / DeviceRGB.
  void redactionAdd(
    int page,
    double x1,
    double y1,
    double x2,
    double y2, {
    double r = 0.0,
    double g = 0.0,
    double b = 0.0,
  }) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.redactionAdd(_handle, page, x1, y1, x2, y2, r, g, b, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'redactionAdd');
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Number of queued redaction regions for 0-based [page].
  int redactionCount(int page) {
    _check();
    final code = calloc<Int32>();
    try {
      final n = _n.redactionCount(_handle, page, code);
      if (n < 0) throw PdfOxideError(code.value, 'redactionCount');
      return n;
    } finally {
      calloc.free(code);
    }
  }

  /// Destructively apply all queued redactions, painting an overlay in
  /// [r]/[g]/[b] (0.0..1.0). Returns the number of glyphs physically removed.
  int redactionApply({
    bool scrubMetadata = false,
    double r = 0.0,
    double g = 0.0,
    double b = 0.0,
  }) {
    _check();
    final code = calloc<Int32>();
    try {
      final n = _n.redactionApply(_handle, scrubMetadata, r, g, b, code);
      if (n < 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'redactionApply');
      }
      return n;
    } finally {
      calloc.free(code);
    }
  }

  /// Strip document metadata / JavaScript / embedded files without geometric
  /// redaction. Returns the number of top-level constructs removed.
  int redactionScrubMetadata() {
    _check();
    final code = calloc<Int32>();
    try {
      final n = _n.redactionScrubMetadata(_handle, code);
      if (n < 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'redactionScrubMetadata');
      }
      return n;
    } finally {
      calloc.free(code);
    }
  }

  // ── Phase 7: barcodes ──────────────────────────────────────────────────────

  /// Stamp [barcode] onto 0-based [pageIndex] at ([x], [y]) sized
  /// [width]×[height] (PDF user-space points).
  void addBarcodeToPage(
    int pageIndex,
    BarcodeImage barcode,
    double x,
    double y,
    double width,
    double height,
  ) {
    _check();
    barcode._check();
    final code = calloc<Int32>();
    try {
      if (_n.addBarcodeToPage(_handle, pageIndex, barcode._handle, x, y, width,
                  height, code) !=
              0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'addBarcodeToPage');
      }
    } finally {
      calloc.free(code);
    }
  }

  // ── Phase 8: FDF / XFDF form-data import ───────────────────────────────────

  void _importBytes(_ImportFdfD fn, Uint8List data, String op) {
    _check();
    final buf = calloc<Uint8>(data.isEmpty ? 1 : data.length);
    if (data.isNotEmpty) buf.asTypedList(data.length).setAll(0, data);
    final code = calloc<Int32>();
    try {
      if (fn(_handle, buf, data.length, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  /// Import AcroForm field values from FDF [data] bytes.
  void importFdfBytes(Uint8List data) =>
      _importBytes(_n.importFdfBytes, data, 'importFdfBytes');

  /// Import AcroForm field values from XFDF [data] bytes.
  void importXfdfBytes(Uint8List data) =>
      _importBytes(_n.importXfdfBytes, data, 'importXfdfBytes');

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.deFree(_handle);
      _handle = nullptr;
    }
  }
}

// ── PDF creation (builder API) ───────────────────────────────────────────────

/// A loaded TTF/OTF font for embedding via [DocumentBuilder.registerEmbeddedFont].
///
/// Owns the native `EmbeddedFont` handle (freed on [close], the finalizer, or
/// dealloc). After a *successful* [DocumentBuilder.registerEmbeddedFont] the
/// builder takes ownership of the underlying handle, so the wrapper's handle is
/// nulled and must not be freed again (the contract in the C header).
class EmbeddedFont implements Finalizable {
  EmbeddedFont._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_EfFreeC>>('pdf_embedded_font_free'));
  Pointer<Void> _handle;

  /// Load a TTF/OTF font from a filesystem [path].
  static EmbeddedFont fromFile(String path) {
    final cPath = path.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.efFromFile(cPath, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'embeddedFontFromFile');
      return EmbeddedFont._(h);
    } finally {
      calloc.free(cPath);
      calloc.free(code);
    }
  }

  /// Load a font from in-memory [data]. [name] may be null to use the
  /// PostScript name from the font face.
  static EmbeddedFont fromBytes(Uint8List data, [String? name]) {
    final buf = calloc<Uint8>(data.length);
    buf.asTypedList(data.length).setAll(0, data);
    final cName = name == null ? nullptr : name.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.efFromBytes(buf, data.length, cName.cast(), code);
      if (h == nullptr)
        throw PdfOxideError(code.value, 'embeddedFontFromBytes');
      return EmbeddedFont._(h);
    } finally {
      calloc.free(buf);
      if (cName != nullptr) calloc.free(cName);
      calloc.free(code);
    }
  }

  void _check() {
    if (_handle == nullptr) throw StateError('EmbeddedFont is closed');
  }

  // Internal: surrender the raw handle to a builder that has taken ownership.
  // Detaches the finalizer and nulls the field so close()/dealloc won't free it.
  Pointer<Void> _surrender() {
    _check();
    _finalizer.detach(this);
    final h = _handle;
    _handle = nullptr;
    return h;
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.efFree(_handle);
      _handle = nullptr;
    }
  }
}

/// A fluent builder for a single page, created by [DocumentBuilder.page],
/// [DocumentBuilder.letterPage], or [DocumentBuilder.a4Page].
///
/// Owns the native `FfiPageBuilder` handle. Each op returns `this` for
/// chaining. Call [done] to commit the buffered ops to the parent builder
/// (this consumes the handle), or [close] to discard them. After either, the
/// handle is null and further ops throw [StateError]. The finalizer frees an
/// uncommitted handle if you forget to call [done]/[close].
class PageBuilder implements Finalizable {
  PageBuilder._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_PbFreeC>>('pdf_page_builder_free'));
  Pointer<Void> _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('PageBuilder is closed');
  }

  // ── shared op shapes ───────────────────────────────────────────────────────

  PageBuilder _status0(_PbStatus0D fn, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      if (fn(_handle, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  PageBuilder _str(_PbStrD fn, String value, String op) {
    _check();
    final c = value.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (fn(_handle, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  PageBuilder _rgb(_PbRgbD fn, double r, double g, double b, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      if (fn(_handle, r, g, b, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  // ── text / layout ──────────────────────────────────────────────────────────

  PageBuilder font(String name, double size) {
    _check();
    final c = name.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbFont(_handle, c, size, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'font');
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  PageBuilder at(double x, double y) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.pbAt(_handle, x, y, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'at');
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  PageBuilder text(String value) => _str(_n.pbText, value, 'text');

  PageBuilder heading(int level, String value) {
    _check();
    final c = value.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbHeading(_handle, level, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'heading');
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  PageBuilder paragraph(String value) =>
      _str(_n.pbParagraph, value, 'paragraph');

  PageBuilder space(double points) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.pbSpace(_handle, points, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'space');
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  PageBuilder horizontalRule() =>
      _status0(_n.pbHorizontalRule, 'horizontalRule');

  PageBuilder columns(int columnCount, double gapPt, String value) {
    _check();
    final c = value.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbColumns(_handle, columnCount, gapPt, c, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'columns');
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  PageBuilder inline(String value) => _str(_n.pbInline, value, 'inline');
  PageBuilder inlineBold(String value) =>
      _str(_n.pbInlineBold, value, 'inlineBold');
  PageBuilder inlineItalic(String value) =>
      _str(_n.pbInlineItalic, value, 'inlineItalic');

  PageBuilder inlineColor(double r, double g, double b, String value) {
    _check();
    final c = value.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbInlineColor(_handle, r, g, b, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'inlineColor');
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  PageBuilder newline() => _status0(_n.pbNewline, 'newline');

  PageBuilder footnote(String refMark, String noteText) {
    _check();
    final cRef = refMark.toNativeUtf8();
    final cNote = noteText.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbFootnote(_handle, cRef, cNote, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'footnote');
      }
      return this;
    } finally {
      calloc.free(cRef);
      calloc.free(cNote);
      calloc.free(code);
    }
  }

  PageBuilder textInRect(
      double x, double y, double w, double h, String value, int align) {
    _check();
    final c = value.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbTextInRect(_handle, x, y, w, h, c, align, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'textInRect');
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  PageBuilder newPageSameSize() =>
      _status0(_n.pbNewPageSameSize, 'newPageSameSize');

  // ── links ──────────────────────────────────────────────────────────────────

  PageBuilder linkUrl(String url) => _str(_n.pbLinkUrl, url, 'linkUrl');

  PageBuilder linkPage(int page) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.pbLinkPage(_handle, page, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'linkPage');
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  PageBuilder linkNamed(String destination) =>
      _str(_n.pbLinkNamed, destination, 'linkNamed');
  PageBuilder linkJavascript(String script) =>
      _str(_n.pbLinkJavascript, script, 'linkJavascript');

  // ── actions / fields ───────────────────────────────────────────────────────

  PageBuilder onOpen(String script) => _str(_n.pbOnOpen, script, 'onOpen');
  PageBuilder onClose(String script) => _str(_n.pbOnClose, script, 'onClose');

  PageBuilder fieldKeystroke(String script) =>
      _str(_n.pbFieldKeystroke, script, 'fieldKeystroke');
  PageBuilder fieldFormat(String script) =>
      _str(_n.pbFieldFormat, script, 'fieldFormat');
  PageBuilder fieldValidate(String script) =>
      _str(_n.pbFieldValidate, script, 'fieldValidate');
  PageBuilder fieldCalculate(String script) =>
      _str(_n.pbFieldCalculate, script, 'fieldCalculate');

  // ── text markup ────────────────────────────────────────────────────────────

  PageBuilder highlight(double r, double g, double b) =>
      _rgb(_n.pbHighlight, r, g, b, 'highlight');
  PageBuilder underline(double r, double g, double b) =>
      _rgb(_n.pbUnderline, r, g, b, 'underline');
  PageBuilder strikeout(double r, double g, double b) =>
      _rgb(_n.pbStrikeout, r, g, b, 'strikeout');
  PageBuilder squiggly(double r, double g, double b) =>
      _rgb(_n.pbSquiggly, r, g, b, 'squiggly');

  // ── annotations / watermark / stamp ────────────────────────────────────────

  PageBuilder stickyNote(String value) =>
      _str(_n.pbStickyNote, value, 'stickyNote');

  PageBuilder stickyNoteAt(double x, double y, String value) {
    _check();
    final c = value.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbStickyNoteAt(_handle, x, y, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'stickyNoteAt');
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  PageBuilder watermark(String value) =>
      _str(_n.pbWatermark, value, 'watermark');
  PageBuilder watermarkConfidential() =>
      _status0(_n.pbWatermarkConfidential, 'watermarkConfidential');
  PageBuilder watermarkDraft() =>
      _status0(_n.pbWatermarkDraft, 'watermarkDraft');
  PageBuilder stamp(String typeName) => _str(_n.pbStamp, typeName, 'stamp');

  PageBuilder freetext(double x, double y, double w, double h, String value) {
    _check();
    final c = value.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbFreetext(_handle, x, y, w, h, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'freetext');
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  // ── images ─────────────────────────────────────────────────────────────────

  PageBuilder _imageCall(_PbImageD fn, Uint8List bytes, double x, double y,
      double w, double h, String op) {
    _check();
    final buf = calloc<Uint8>(bytes.length);
    buf.asTypedList(bytes.length).setAll(0, bytes);
    final code = calloc<Int32>();
    try {
      if (fn(_handle, buf, bytes.length, x, y, w, h, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
      return this;
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  PageBuilder image(Uint8List bytes, double x, double y, double w, double h) =>
      _imageCall(_n.pbImage, bytes, x, y, w, h, 'image');

  PageBuilder imageArtifact(
          Uint8List bytes, double x, double y, double w, double h) =>
      _imageCall(_n.pbImageArtifact, bytes, x, y, w, h, 'imageArtifact');

  PageBuilder imageWithAlt(
      Uint8List bytes, double x, double y, double w, double h, String altText) {
    _check();
    final buf = calloc<Uint8>(bytes.length);
    buf.asTypedList(bytes.length).setAll(0, bytes);
    final cAlt = altText.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbImageWithAlt(
                  _handle, buf, bytes.length, x, y, w, h, cAlt, code) !=
              0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'imageWithAlt');
      }
      return this;
    } finally {
      calloc.free(buf);
      calloc.free(cAlt);
      calloc.free(code);
    }
  }

  PageBuilder barcode1d(
      int barcodeType, String data, double x, double y, double w, double h) {
    _check();
    final c = data.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbBarcode1d(_handle, barcodeType, c, x, y, w, h, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'barcode1d');
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  PageBuilder barcodeQr(String data, double x, double y, double size) {
    _check();
    final c = data.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbBarcodeQr(_handle, c, x, y, size, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'barcodeQr');
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  // ── vector graphics ────────────────────────────────────────────────────────

  PageBuilder rect(double x, double y, double w, double h) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.pbRect(_handle, x, y, w, h, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'rect');
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  PageBuilder filledRect(
      double x, double y, double w, double h, double r, double g, double b) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.pbFilledRect(_handle, x, y, w, h, r, g, b, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'filledRect');
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  PageBuilder line(double x1, double y1, double x2, double y2) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.pbLine(_handle, x1, y1, x2, y2, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'line');
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  PageBuilder strokeRect(double x, double y, double w, double h, double width,
      double r, double g, double b) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.pbStrokeRect(_handle, x, y, w, h, width, r, g, b, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'strokeRect');
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  PageBuilder strokeLine(double x1, double y1, double x2, double y2,
      double width, double r, double g, double b) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.pbStrokeLine(_handle, x1, y1, x2, y2, width, r, g, b, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'strokeLine');
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  PageBuilder _strokeDashed(
      _PbStrokeRectDashedD fn,
      double a,
      double b,
      double c,
      double d,
      double width,
      double r,
      double g,
      double bl,
      List<double> dashArray,
      double phase,
      String op) {
    _check();
    final n = dashArray.length;
    final dash = n == 0 ? nullptr : calloc<Float>(n);
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < n; i++) {
        dash[i] = dashArray[i];
      }
      if (fn(_handle, a, b, c, d, width, r, g, bl, dash.cast(), n, phase,
                  code) !=
              0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
      return this;
    } finally {
      if (dash != nullptr) calloc.free(dash);
      calloc.free(code);
    }
  }

  PageBuilder strokeRectDashed(double x, double y, double w, double h,
          double width, double r, double g, double b, List<double> dashArray,
          [double phase = 0]) =>
      _strokeDashed(_n.pbStrokeRectDashed, x, y, w, h, width, r, g, b,
          dashArray, phase, 'strokeRectDashed');

  PageBuilder strokeLineDashed(double x1, double y1, double x2, double y2,
          double width, double r, double g, double b, List<double> dashArray,
          [double phase = 0]) =>
      _strokeDashed(_n.pbStrokeLineDashed, x1, y1, x2, y2, width, r, g, b,
          dashArray, phase, 'strokeLineDashed');

  // ── form fields ────────────────────────────────────────────────────────────

  PageBuilder textField(String name, double x, double y, double w, double h,
      [String? defaultValue]) {
    _check();
    final cName = name.toNativeUtf8();
    final cDef = defaultValue == null ? nullptr : defaultValue.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbTextField(_handle, cName, x, y, w, h, cDef.cast(), code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'textField');
      }
      return this;
    } finally {
      calloc.free(cName);
      if (cDef != nullptr) calloc.free(cDef);
      calloc.free(code);
    }
  }

  PageBuilder checkbox(
      String name, double x, double y, double w, double h, bool checked) {
    _check();
    final cName = name.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbCheckbox(_handle, cName, x, y, w, h, checked ? 1 : 0, code) !=
              0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'checkbox');
      }
      return this;
    } finally {
      calloc.free(cName);
      calloc.free(code);
    }
  }

  PageBuilder comboBox(String name, double x, double y, double w, double h,
      List<String> options, int count,
      [String? selected]) {
    _check();
    final cName = name.toNativeUtf8();
    final cSel = selected == null ? nullptr : selected.toNativeUtf8();
    final opts = calloc<Pointer<Utf8>>(count);
    final cStrings = <Pointer<Utf8>>[];
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < count; i++) {
        final s = options[i].toNativeUtf8();
        cStrings.add(s);
        opts[i] = s;
      }
      if (_n.pbComboBox(
                  _handle, cName, x, y, w, h, opts, count, cSel.cast(), code) !=
              0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'comboBox');
      }
      return this;
    } finally {
      for (final s in cStrings) {
        calloc.free(s);
      }
      calloc.free(opts);
      calloc.free(cName);
      if (cSel != nullptr) calloc.free(cSel);
      calloc.free(code);
    }
  }

  PageBuilder radioGroup(String name, List<String> values, List<double> xs,
      List<double> ys, List<double> ws, List<double> hs, int count,
      [String? selected]) {
    _check();
    final cName = name.toNativeUtf8();
    final cSel = selected == null ? nullptr : selected.toNativeUtf8();
    final vals = calloc<Pointer<Utf8>>(count);
    final cStrings = <Pointer<Utf8>>[];
    final pxs = calloc<Float>(count);
    final pys = calloc<Float>(count);
    final pws = calloc<Float>(count);
    final phs = calloc<Float>(count);
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < count; i++) {
        final s = values[i].toNativeUtf8();
        cStrings.add(s);
        vals[i] = s;
        pxs[i] = xs[i];
        pys[i] = ys[i];
        pws[i] = ws[i];
        phs[i] = hs[i];
      }
      if (_n.pbRadioGroup(_handle, cName, vals, pxs, pys, pws, phs, count,
                  cSel.cast(), code) !=
              0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'radioGroup');
      }
      return this;
    } finally {
      for (final s in cStrings) {
        calloc.free(s);
      }
      calloc.free(vals);
      calloc.free(pxs);
      calloc.free(pys);
      calloc.free(pws);
      calloc.free(phs);
      calloc.free(cName);
      if (cSel != nullptr) calloc.free(cSel);
      calloc.free(code);
    }
  }

  PageBuilder pushButton(
      String name, double x, double y, double w, double h, String caption) {
    _check();
    final cName = name.toNativeUtf8();
    final cCap = caption.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbPushButton(_handle, cName, x, y, w, h, cCap, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'pushButton');
      }
      return this;
    } finally {
      calloc.free(cName);
      calloc.free(cCap);
      calloc.free(code);
    }
  }

  PageBuilder signatureField(
      String name, double x, double y, double w, double h) {
    _check();
    final cName = name.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.pbSignatureField(_handle, cName, x, y, w, h, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'signatureField');
      }
      return this;
    } finally {
      calloc.free(cName);
      calloc.free(code);
    }
  }

  // ── table ──────────────────────────────────────────────────────────────────

  /// Buffer a static table. [widths]/[aligns] are length [nCols]; [cellStrings]
  /// is row-major (`cellStrings[row * nCols + col]`) of length `nCols * nRows`.
  /// [aligns] encodes 0=Left, 1=Center, 2=Right.
  PageBuilder table(int nCols, List<double> widths, List<int> aligns, int nRows,
      List<String> cellStrings, bool hasHeader) {
    _check();
    final pw = calloc<Float>(nCols);
    final pa = calloc<Int32>(nCols);
    final cells = calloc<Pointer<Utf8>>(nCols * nRows);
    final cStrings = <Pointer<Utf8>>[];
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < nCols; i++) {
        pw[i] = widths[i];
        pa[i] = aligns[i];
      }
      for (var i = 0; i < nCols * nRows; i++) {
        final s = cellStrings[i].toNativeUtf8();
        cStrings.add(s);
        cells[i] = s;
      }
      if (_n.pbTable(_handle, nCols, pw, pa, nRows, cells, hasHeader ? 1 : 0,
                  code) !=
              0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'table');
      }
      return this;
    } finally {
      for (final s in cStrings) {
        calloc.free(s);
      }
      calloc.free(cells);
      calloc.free(pw);
      calloc.free(pa);
      calloc.free(code);
    }
  }

  // ── Phase 8: streaming tables ──────────────────────────────────────────────

  /// Begin a streaming table with [headers] (one per column), per-column
  /// [widths] and [aligns], repeating the header row across page breaks when
  /// [repeatHeader] is set.
  PageBuilder streamingTableBegin(List<String> headers, List<double> widths,
      List<int> aligns, bool repeatHeader) {
    _check();
    final n = headers.length;
    final hdrs = calloc<Pointer<Utf8>>(n == 0 ? 1 : n);
    final pw = calloc<Float>(n == 0 ? 1 : n);
    final pa = calloc<Int32>(n == 0 ? 1 : n);
    final cStrings = <Pointer<Utf8>>[];
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < n; i++) {
        final s = headers[i].toNativeUtf8();
        cStrings.add(s);
        hdrs[i] = s;
        pw[i] = widths[i];
        pa[i] = aligns[i];
      }
      if (_n.stBegin(_handle, n, hdrs, pw, pa, repeatHeader ? 1 : 0, code) !=
              0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'streamingTableBegin');
      }
      return this;
    } finally {
      for (final s in cStrings) {
        calloc.free(s);
      }
      calloc.free(hdrs);
      calloc.free(pw);
      calloc.free(pa);
      calloc.free(code);
    }
  }

  /// Begin a streaming table (v2) with layout tuning: column-width [mode],
  /// [sampleRows] to auto-size from, [minColWidthPt]/[maxColWidthPt] bounds and
  /// [maxRowspan].
  PageBuilder streamingTableBeginV2(List<String> headers, List<double> widths,
      List<int> aligns, bool repeatHeader,
      {int mode = 0,
      int sampleRows = 0,
      double minColWidthPt = 0,
      double maxColWidthPt = 0,
      int maxRowspan = 1}) {
    _check();
    final n = headers.length;
    final hdrs = calloc<Pointer<Utf8>>(n == 0 ? 1 : n);
    final pw = calloc<Float>(n == 0 ? 1 : n);
    final pa = calloc<Int32>(n == 0 ? 1 : n);
    final cStrings = <Pointer<Utf8>>[];
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < n; i++) {
        final s = headers[i].toNativeUtf8();
        cStrings.add(s);
        hdrs[i] = s;
        pw[i] = widths[i];
        pa[i] = aligns[i];
      }
      if (_n.stBeginV2(_handle, n, hdrs, pw, pa, repeatHeader ? 1 : 0, mode,
                  sampleRows, minColWidthPt, maxColWidthPt, maxRowspan, code) !=
              0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'streamingTableBeginV2');
      }
      return this;
    } finally {
      for (final s in cStrings) {
        calloc.free(s);
      }
      calloc.free(hdrs);
      calloc.free(pw);
      calloc.free(pa);
      calloc.free(code);
    }
  }

  /// Push one row of [cells] into the open streaming table.
  PageBuilder streamingTablePushRow(List<String> cells) {
    _check();
    final n = cells.length;
    final arr = calloc<Pointer<Utf8>>(n == 0 ? 1 : n);
    final cStrings = <Pointer<Utf8>>[];
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < n; i++) {
        final s = cells[i].toNativeUtf8();
        cStrings.add(s);
        arr[i] = s;
      }
      if (_n.stPushRow(_handle, n, arr, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'streamingTablePushRow');
      }
      return this;
    } finally {
      for (final s in cStrings) {
        calloc.free(s);
      }
      calloc.free(arr);
      calloc.free(code);
    }
  }

  /// Push one row of [cells] with per-cell [rowspans] into the streaming table.
  PageBuilder streamingTablePushRowV2(List<String> cells, List<int> rowspans) {
    _check();
    final n = cells.length;
    final arr = calloc<Pointer<Utf8>>(n == 0 ? 1 : n);
    final spans = calloc<IntPtr>(n == 0 ? 1 : n);
    final cStrings = <Pointer<Utf8>>[];
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < n; i++) {
        final s = cells[i].toNativeUtf8();
        cStrings.add(s);
        arr[i] = s;
        spans[i] = rowspans[i];
      }
      if (_n.stPushRowV2(_handle, n, arr, spans, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'streamingTablePushRowV2');
      }
      return this;
    } finally {
      for (final s in cStrings) {
        calloc.free(s);
      }
      calloc.free(arr);
      calloc.free(spans);
      calloc.free(code);
    }
  }

  /// Flush any buffered/pending rows of the streaming table.
  PageBuilder streamingTableFlush() =>
      _status0(_n.stFlush, 'streamingTableFlush');

  /// Finish (close) the open streaming table.
  PageBuilder streamingTableFinish() =>
      _status0(_n.stFinish, 'streamingTableFinish');

  /// Set the streaming-table flush [batchSize] (rows buffered before flush).
  PageBuilder streamingTableSetBatchSize(int batchSize) {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.stSetBatchSize(_handle, batchSize, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'streamingTableSetBatchSize');
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  /// The number of batches flushed so far for the streaming table.
  int streamingTableBatchCount() {
    _check();
    return _n.stBatchCount(_handle);
  }

  /// The number of rows pending (not yet flushed) in the streaming table.
  int streamingTablePendingRowCount() {
    _check();
    return _n.stPendingRowCount(_handle);
  }

  // ── lifecycle ──────────────────────────────────────────────────────────────

  /// Commit this page's buffered ops to the parent builder. **Consumes** the
  /// handle — after this the page builder is closed (do not call further ops).
  void done() {
    _check();
    final code = calloc<Int32>();
    try {
      final rc = _n.pbDone(_handle, code);
      // _done consumes the native handle regardless of success; detach the
      // finalizer and null our field so we never call _free on it.
      _finalizer.detach(this);
      _handle = nullptr;
      if (rc != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'done');
      }
    } finally {
      calloc.free(code);
    }
  }

  /// Discard this page (drops buffered ops) and free the handle (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.pbFree(_handle);
      _handle = nullptr;
    }
  }
}

/// A builder for assembling a brand-new PDF document.
///
/// Owns the native `FfiDocumentBuilder` handle. Set metadata, register fonts,
/// open pages via [page]/[letterPage]/[a4Page], then [build]/[save]/etc. The
/// terminal ops consume the builder *state* but the handle is still freed via
/// [close]/the finalizer/dealloc (per the C contract). C status returns are
/// 0 = success; a non-zero return or set error_code raises [PdfOxideError].
class DocumentBuilder implements Finalizable {
  DocumentBuilder._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_DbFreeC>>('pdf_document_builder_free'));
  Pointer<Void> _handle;

  /// Create a fresh document builder.
  static DocumentBuilder create() {
    final code = calloc<Int32>();
    try {
      final h = _n.dbCreate(code);
      if (h == nullptr) throw PdfOxideError(code.value, 'create');
      return DocumentBuilder._(h);
    } finally {
      calloc.free(code);
    }
  }

  void _check() {
    if (_handle == nullptr) throw StateError('DocumentBuilder is closed');
  }

  // ── metadata (fluent) ──────────────────────────────────────────────────────

  DocumentBuilder _setStr(_DbSetStrD fn, String value, String op) {
    _check();
    final c = value.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (fn(_handle, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, op);
      }
      return this;
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  DocumentBuilder setTitle(String value) =>
      _setStr(_n.dbSetTitle, value, 'setTitle');
  DocumentBuilder setAuthor(String value) =>
      _setStr(_n.dbSetAuthor, value, 'setAuthor');
  DocumentBuilder setSubject(String value) =>
      _setStr(_n.dbSetSubject, value, 'setSubject');
  DocumentBuilder setKeywords(String value) =>
      _setStr(_n.dbSetKeywords, value, 'setKeywords');
  DocumentBuilder setCreator(String value) =>
      _setStr(_n.dbSetCreator, value, 'setCreator');
  DocumentBuilder onOpen(String script) =>
      _setStr(_n.dbOnOpen, script, 'onOpen');
  DocumentBuilder language(String lang) =>
      _setStr(_n.dbLanguage, lang, 'language');

  /// Enable PDF/UA-1 tagged PDF mode.
  DocumentBuilder taggedPdfUa1() {
    _check();
    final code = calloc<Int32>();
    try {
      if (_n.dbTaggedPdfUa1(_handle, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'taggedPdfUa1');
      }
      return this;
    } finally {
      calloc.free(code);
    }
  }

  /// Add a role-map entry: [custom] structure type → [standard] PDF type.
  DocumentBuilder roleMap(String custom, String standard) {
    _check();
    final cCustom = custom.toNativeUtf8();
    final cStd = standard.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.dbRoleMap(_handle, cCustom, cStd, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'roleMap');
      }
      return this;
    } finally {
      calloc.free(cCustom);
      calloc.free(cStd);
      calloc.free(code);
    }
  }

  /// Register an [EmbeddedFont] under [name]. On success the builder takes
  /// ownership of the font's native handle (the [EmbeddedFont] is consumed and
  /// must not be used or freed afterwards). On error the font remains valid.
  DocumentBuilder registerEmbeddedFont(String name, EmbeddedFont font) {
    _check();
    font._check();
    final cName = name.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final rc = _n.dbRegisterFont(_handle, cName, font._handle, code);
      if (rc != 0 || code.value != 0) {
        // Font NOT consumed on error — leave the wrapper owning it.
        throw PdfOxideError(code.value, 'registerEmbeddedFont');
      }
      // Consumed: surrender so the EmbeddedFont won't double-free.
      font._surrender();
      return this;
    } finally {
      calloc.free(cName);
      calloc.free(code);
    }
  }

  // ── pages ──────────────────────────────────────────────────────────────────

  PageBuilder _openPage(_DbPageD fn, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = fn(_handle, code);
      if (h == nullptr) throw PdfOxideError(code.value, op);
      return PageBuilder._(h);
    } finally {
      calloc.free(code);
    }
  }

  PageBuilder a4Page() => _openPage(_n.dbA4Page, 'a4Page');
  PageBuilder letterPage() => _openPage(_n.dbLetterPage, 'letterPage');

  PageBuilder page(double width, double height) {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.dbPage(_handle, width, height, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'page');
      return PageBuilder._(h);
    } finally {
      calloc.free(code);
    }
  }

  // ── build / save ───────────────────────────────────────────────────────────

  /// Build the PDF and return its bytes.
  Uint8List build() {
    _check();
    final len = calloc<IntPtr>();
    final code = calloc<Int32>();
    try {
      final p = _n.dbBuild(_handle, len, code);
      if (p == nullptr) throw PdfOxideError(code.value, 'build');
      final out =
          Uint8List.fromList(p.asTypedList(len.value < 0 ? 0 : len.value));
      _n.freeBytes(p);
      return out;
    } finally {
      calloc.free(len);
      calloc.free(code);
    }
  }

  /// Build and save the PDF to [path].
  void save(String path) {
    _check();
    final c = path.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.dbSave(_handle, c, code) != 0 || code.value != 0) {
        throw PdfOxideError(code.value, 'save');
      }
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  /// Build and save with AES-256 encryption to [path].
  void saveEncrypted(String path, String userPassword, String ownerPassword) {
    _check();
    final cPath = path.toNativeUtf8();
    final cUser = userPassword.toNativeUtf8();
    final cOwner = ownerPassword.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      if (_n.dbSaveEncrypted(_handle, cPath, cUser, cOwner, code) != 0 ||
          code.value != 0) {
        throw PdfOxideError(code.value, 'saveEncrypted');
      }
    } finally {
      calloc.free(cPath);
      calloc.free(cUser);
      calloc.free(cOwner);
      calloc.free(code);
    }
  }

  /// Build encrypted bytes (AES-256).
  Uint8List toBytesEncrypted(String userPassword, String ownerPassword) {
    _check();
    final cUser = userPassword.toNativeUtf8();
    final cOwner = ownerPassword.toNativeUtf8();
    final len = calloc<IntPtr>();
    final code = calloc<Int32>();
    try {
      final p = _n.dbToBytesEncrypted(_handle, cUser, cOwner, len, code);
      if (p == nullptr) throw PdfOxideError(code.value, 'toBytesEncrypted');
      final out =
          Uint8List.fromList(p.asTypedList(len.value < 0 ? 0 : len.value));
      _n.freeBytes(p);
      return out;
    } finally {
      calloc.free(cUser);
      calloc.free(cOwner);
      calloc.free(len);
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.dbFree(_handle);
      _handle = nullptr;
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Phase 6 — digital signatures / PKI / timestamps / TSA / DSS / validation.
//
// Style matches earlier phases: every opaque native handle is wrapped in a
// `Finalizable` class freed via its `*_free` symbol on `close()` (idempotent)
// and a `NativeFinalizer`; calls after close throw `StateError`. Owned `char*`
// returns go through `_takeString` (+ `free_string`); owned `uint8*` buffers are
// copied then released with `free_bytes`. `const uint8*` returns (timestamp
// token / message imprint) are COPIED ONLY — never `free_bytes`'d. Validation
// result handles use their dedicated `*_results_free`.
// ════════════════════════════════════════════════════════════════════════════

/// Copy an owned native `uint8*` buffer into Dart and release it via `free_bytes`.
Uint8List _takeBytes(Pointer<Uint8> p, int len, int code, String op) {
  if (p == nullptr) throw PdfOxideError(code, op);
  final out = Uint8List.fromList(p.asTypedList(len < 0 ? 0 : len));
  _n.freeBytes(p);
  return out;
}

/// Marshal a list of DER byte buffers into the parallel
/// `(const uint8* const*, const uintptr*, count)` triple the PAdES signer
/// expects. Returns the two allocated pointer arrays plus every per-element
/// buffer so the caller can [calloc.free] them after the call returns.
class _ByteArrayArray {
  _ByteArrayArray(List<Uint8List> items)
      : count = items.length,
        ptrs = items.isEmpty ? nullptr : calloc<Pointer<Uint8>>(items.length),
        lens = items.isEmpty ? nullptr : calloc<IntPtr>(items.length) {
    for (var i = 0; i < items.length; i++) {
      final item = items[i];
      final buf = calloc<Uint8>(item.isEmpty ? 1 : item.length);
      if (item.isNotEmpty) buf.asTypedList(item.length).setAll(0, item);
      _bufs.add(buf);
      ptrs[i] = buf;
      lens[i] = item.length;
    }
  }

  final int count;
  final Pointer<Pointer<Uint8>> ptrs;
  final Pointer<IntPtr> lens;
  final List<Pointer<Uint8>> _bufs = [];

  void free() {
    for (final b in _bufs) {
      calloc.free(b);
    }
    if (ptrs != nullptr) calloc.free(ptrs);
    if (lens != nullptr) calloc.free(lens);
  }
}

/// Validity window of a [Certificate] as Unix epoch seconds.
class CertificateValidity {
  const CertificateValidity(this.notBefore, this.notAfter);
  final int notBefore;
  final int notAfter;
  @override
  String toString() => 'CertificateValidity($notBefore..$notAfter)';
}

/// Signing credentials / X.509 certificate. Created via [loadFromBytes]
/// (PKCS#12) or [loadFromPem]; freed via [close] (or the finalizer).
class Certificate implements Finalizable {
  Certificate._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_PtrFreeC>>('pdf_certificate_free'));
  Pointer<Void> _handle;

  /// The raw native handle (advanced/interop use).
  Pointer<Void> get handle => _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('Certificate is closed');
  }

  /// Load credentials from a PKCS#12 (.p12/.pfx) byte buffer with [password].
  static Certificate loadFromBytes(Uint8List bytes, String password) {
    final buf = calloc<Uint8>(bytes.isEmpty ? 1 : bytes.length);
    if (bytes.isNotEmpty) buf.asTypedList(bytes.length).setAll(0, bytes);
    final cPw = password.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.certLoadFromBytes(buf, bytes.length, cPw, code);
      if (h == nullptr) {
        throw PdfOxideError(code.value, 'certificateLoadFromBytes');
      }
      return Certificate._(h);
    } finally {
      calloc.free(buf);
      calloc.free(cPw);
      calloc.free(code);
    }
  }

  /// Load credentials from PEM-encoded certificate + private-key strings.
  static Certificate loadFromPem(String certPem, String keyPem) {
    final cCert = certPem.toNativeUtf8();
    final cKey = keyPem.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.certLoadFromPem(cCert, cKey, code);
      if (h == nullptr) {
        throw PdfOxideError(code.value, 'certificateLoadFromPem');
      }
      return Certificate._(h);
    } finally {
      calloc.free(cCert);
      calloc.free(cKey);
      calloc.free(code);
    }
  }

  String _str(_CertStrD fn, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(fn(_handle, code), code.value, op);
    } finally {
      calloc.free(code);
    }
  }

  /// The certificate subject (distinguished name).
  String get subject => _str(_n.certGetSubject, 'certificateGetSubject');

  /// The certificate issuer (distinguished name).
  String get issuer => _str(_n.certGetIssuer, 'certificateGetIssuer');

  /// The certificate serial number (decimal/hex string).
  String get serial => _str(_n.certGetSerial, 'certificateGetSerial');

  /// The validity window (notBefore/notAfter, Unix epoch seconds).
  CertificateValidity get validity {
    _check();
    final nb = calloc<Int64>();
    final na = calloc<Int64>();
    final code = calloc<Int32>();
    try {
      _n.certGetValidity(_handle, nb, na, code);
      if (code.value != 0) {
        throw PdfOxideError(code.value, 'certificateGetValidity');
      }
      return CertificateValidity(nb.value, na.value);
    } finally {
      calloc.free(nb);
      calloc.free(na);
      calloc.free(code);
    }
  }

  /// Whether the certificate is currently valid (not expired / not before).
  bool isValid() {
    _check();
    final code = calloc<Int32>();
    try {
      final r = _n.certIsValid(_handle, code);
      if (code.value != 0) {
        throw PdfOxideError(code.value, 'certificateIsValid');
      }
      return r != 0;
    } finally {
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.certFree(_handle);
      _handle = nullptr;
    }
  }
}

/// An RFC 3161 timestamp token, parsed via [parse]. Freed via [close].
class Timestamp implements Finalizable {
  Timestamp._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_PtrFreeC>>('pdf_timestamp_free'));
  Pointer<Void> _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('Timestamp is closed');
  }

  /// Parse a DER-encoded RFC 3161 TimeStampToken (or bare TSTInfo).
  static Timestamp parse(Uint8List bytes) {
    final buf = calloc<Uint8>(bytes.isEmpty ? 1 : bytes.length);
    if (bytes.isNotEmpty) buf.asTypedList(bytes.length).setAll(0, bytes);
    final code = calloc<Int32>();
    try {
      final h = _n.tsParse(buf, bytes.length, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'timestampParse');
      return Timestamp._(h);
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  // The token / message-imprint returns are `const uint8*` owned by the handle:
  // copy them out, never free_bytes.
  Uint8List _constBytes(_TsConstBytesD fn, String op) {
    _check();
    final len = calloc<IntPtr>();
    final code = calloc<Int32>();
    try {
      final p = fn(_handle, len, code);
      if (p == nullptr) throw PdfOxideError(code.value, op);
      return Uint8List.fromList(p.asTypedList(len.value < 0 ? 0 : len.value));
    } finally {
      calloc.free(len);
      calloc.free(code);
    }
  }

  /// The full DER TimeStampToken bytes (copied; not owned by the caller).
  Uint8List get token => _constBytes(_n.tsGetToken, 'timestampGetToken');

  /// The hashed message imprint bytes (copied; not owned by the caller).
  Uint8List get messageImprint =>
      _constBytes(_n.tsGetMessageImprint, 'timestampGetMessageImprint');

  String _str(_TsStrD fn, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(fn(_handle, code), code.value, op);
    } finally {
      calloc.free(code);
    }
  }

  /// The timestamp time (Unix epoch seconds).
  int get time {
    _check();
    final code = calloc<Int32>();
    try {
      final t = _n.tsGetTime(_handle, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'timestampGetTime');
      return t;
    } finally {
      calloc.free(code);
    }
  }

  /// The token serial number.
  String get serial => _str(_n.tsGetSerial, 'timestampGetSerial');

  /// The TSA (timestamp authority) name.
  String get tsaName => _str(_n.tsGetTsaName, 'timestampGetTsaName');

  /// The TSA policy OID.
  String get policyOid => _str(_n.tsGetPolicyOid, 'timestampGetPolicyOid');

  /// The message-imprint hash algorithm code.
  int get hashAlgorithm {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.tsGetHashAlgorithm(_handle, code);
      if (code.value != 0) {
        throw PdfOxideError(code.value, 'timestampGetHashAlgorithm');
      }
      return h;
    } finally {
      calloc.free(code);
    }
  }

  /// Cryptographically verify the timestamp token.
  bool verify() {
    _check();
    final code = calloc<Int32>();
    try {
      final r = _n.tsVerify(_handle, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'timestampVerify');
      return r;
    } finally {
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.tsFree(_handle);
      _handle = nullptr;
    }
  }
}

/// An RFC 3161 TSA (timestamp authority) client. Created via [create]; freed via
/// [close]. Requests return owned [Timestamp] handles.
class TsaClient implements Finalizable {
  TsaClient._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_PtrFreeC>>('pdf_tsa_client_free'));
  Pointer<Void> _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('TsaClient is closed');
  }

  /// Create a TSA client for [url]. [username]/[password] are optional HTTP
  /// basic-auth credentials (pass empty strings to omit).
  static TsaClient create(
    String url, {
    String username = '',
    String password = '',
    int timeout = 30,
    int hashAlgo = 0,
    bool useNonce = true,
    bool certReq = true,
  }) {
    final cUrl = url.toNativeUtf8();
    final cUser = username.toNativeUtf8();
    final cPw = password.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.tsaClientCreate(
          cUrl, cUser, cPw, timeout, hashAlgo, useNonce, certReq, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'tsaClientCreate');
      return TsaClient._(h);
    } finally {
      calloc.free(cUrl);
      calloc.free(cUser);
      calloc.free(cPw);
      calloc.free(code);
    }
  }

  /// Request a timestamp over raw [data] (the client hashes it).
  Timestamp requestTimestamp(Uint8List data) {
    _check();
    final buf = calloc<Uint8>(data.isEmpty ? 1 : data.length);
    if (data.isNotEmpty) buf.asTypedList(data.length).setAll(0, data);
    final code = calloc<Int32>();
    try {
      final h = _n.tsaRequestTimestamp(_handle, buf, data.length, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'tsaRequestTimestamp');
      return Timestamp._(h);
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  /// Request a timestamp over a precomputed [hash] of the given [hashAlgo].
  Timestamp requestTimestampHash(Uint8List hash, int hashAlgo) {
    _check();
    final buf = calloc<Uint8>(hash.isEmpty ? 1 : hash.length);
    if (hash.isNotEmpty) buf.asTypedList(hash.length).setAll(0, hash);
    final code = calloc<Int32>();
    try {
      final h =
          _n.tsaRequestTimestampHash(_handle, buf, hash.length, hashAlgo, code);
      if (h == nullptr) {
        throw PdfOxideError(code.value, 'tsaRequestTimestampHash');
      }
      return Timestamp._(h);
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.tsaClientFree(_handle);
      _handle = nullptr;
    }
  }
}

/// Information about a digital signature embedded in a PDF. Wraps an
/// `FfiSignatureInfo*` handle (e.g. from `pdf_signature_get_timestamp`'s peers);
/// freed via [close].
class SignatureInfo implements Finalizable {
  /// Adopt a raw `FfiSignatureInfo*` handle (advanced/interop use).
  SignatureInfo.fromHandle(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_PtrFreeC>>('pdf_signature_free'));
  Pointer<Void> _handle;

  /// The raw native handle (advanced/interop use).
  Pointer<Void> get handle => _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('SignatureInfo is closed');
  }

  String _str(_SigStrD fn, String op) {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(fn(_handle, code), code.value, op);
    } finally {
      calloc.free(code);
    }
  }

  /// The signer name.
  String get signerName => _str(_n.sigGetSignerName, 'signatureGetSignerName');

  /// The signing reason.
  String get signingReason =>
      _str(_n.sigGetSigningReason, 'signatureGetSigningReason');

  /// The signing location.
  String get signingLocation =>
      _str(_n.sigGetSigningLocation, 'signatureGetSigningLocation');

  /// The signing time (Unix epoch seconds).
  int get signingTime {
    _check();
    final code = calloc<Int32>();
    try {
      final t = _n.sigGetSigningTime(_handle, code);
      if (code.value != 0) {
        throw PdfOxideError(code.value, 'signatureGetSigningTime');
      }
      return t;
    } finally {
      calloc.free(code);
    }
  }

  /// The signer's [Certificate].
  Certificate get certificate {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.sigGetCertificate(_handle, code);
      if (h == nullptr) {
        throw PdfOxideError(code.value, 'signatureGetCertificate');
      }
      return Certificate._(h);
    } finally {
      calloc.free(code);
    }
  }

  /// The PAdES baseline level code (B-B/B-T/...), or `-1` if not PAdES.
  int get padesLevel {
    _check();
    final code = calloc<Int32>();
    try {
      return _n.sigGetPadesLevel(_handle, code);
    } finally {
      calloc.free(code);
    }
  }

  /// Whether this signature carries an embedded timestamp.
  bool hasTimestamp() {
    _check();
    final code = calloc<Int32>();
    try {
      final r = _n.sigHasTimestamp(_handle, code);
      if (code.value != 0) {
        throw PdfOxideError(code.value, 'signatureHasTimestamp');
      }
      return r;
    } finally {
      calloc.free(code);
    }
  }

  /// The embedded [Timestamp].
  Timestamp get timestamp {
    _check();
    final code = calloc<Int32>();
    try {
      final h = _n.sigGetTimestamp(_handle, code);
      if (h == nullptr) {
        throw PdfOxideError(code.value, 'signatureGetTimestamp');
      }
      return Timestamp._(h);
    } finally {
      calloc.free(code);
    }
  }

  /// Attach a [ts] timestamp to this signature. Returns whether it succeeded.
  bool addTimestamp(Timestamp ts) {
    _check();
    ts._check();
    final code = calloc<Int32>();
    try {
      final r = _n.sigAddTimestamp(_handle, ts._handle, code);
      if (code.value != 0) {
        throw PdfOxideError(code.value, 'signatureAddTimestamp');
      }
      return r;
    } finally {
      calloc.free(code);
    }
  }

  /// Run the signer-attributes crypto check. Returns 1 valid / 0 invalid /
  /// -1 unknown-or-unsupported.
  int verify() {
    _check();
    final code = calloc<Int32>();
    try {
      return _n.sigVerify(_handle, code);
    } finally {
      calloc.free(code);
    }
  }

  /// Verify end-to-end against the full [pdf] file bytes. Returns 1 valid /
  /// 0 invalid / -1 unknown-or-unsupported.
  int verifyDetached(Uint8List pdf) {
    _check();
    final buf = calloc<Uint8>(pdf.isEmpty ? 1 : pdf.length);
    if (pdf.isNotEmpty) buf.asTypedList(pdf.length).setAll(0, pdf);
    final code = calloc<Int32>();
    try {
      return _n.sigVerifyDetached(_handle, buf, pdf.length, code);
    } finally {
      calloc.free(buf);
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.sigFree(_handle);
      _handle = nullptr;
    }
  }
}

/// A document `/DSS` (document security store). Freed via [close].
class Dss implements Finalizable {
  /// Adopt a raw DSS handle (advanced/interop use).
  Dss.fromHandle(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer =
      NativeFinalizer(_n.lib.lookup<NativeFunction<_PtrFreeC>>('pdf_dss_free'));
  Pointer<Void> _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('Dss is closed');
  }

  /// Number of certificates in the DSS.
  int get certCount {
    _check();
    return _n.dssCertCount(_handle);
  }

  /// Number of CRLs in the DSS.
  int get crlCount {
    _check();
    return _n.dssCrlCount(_handle);
  }

  /// Number of OCSP responses in the DSS.
  int get ocspCount {
    _check();
    return _n.dssOcspCount(_handle);
  }

  /// Number of VRI (validation-related info) entries in the DSS.
  int get vriCount {
    _check();
    return _n.dssVriCount(_handle);
  }

  Uint8List _get(_DssGetD fn, int index, String op) {
    _check();
    final len = calloc<IntPtr>();
    final code = calloc<Int32>();
    try {
      final p = fn(_handle, index, len, code);
      return _takeBytes(p, len.value, code.value, op);
    } finally {
      calloc.free(len);
      calloc.free(code);
    }
  }

  /// The DER bytes of the certificate at [index].
  Uint8List getCert(int index) => _get(_n.dssGetCert, index, 'dssGetCert');

  /// The DER bytes of the CRL at [index].
  Uint8List getCrl(int index) => _get(_n.dssGetCrl, index, 'dssGetCrl');

  /// The DER bytes of the OCSP response at [index].
  Uint8List getOcsp(int index) => _get(_n.dssGetOcsp, index, 'dssGetOcsp');

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.dssFree(_handle);
      _handle = nullptr;
    }
  }
}

/// PDF/UA accessibility element statistics.
class UaStats {
  const UaStats(this.structElements, this.images, this.tables, this.forms,
      this.annotations, this.pages);
  final int structElements;
  final int images;
  final int tables;
  final int forms;
  final int annotations;
  final int pages;
  @override
  String toString() =>
      'UaStats(struct=$structElements, images=$images, tables=$tables, '
      'forms=$forms, annotations=$annotations, pages=$pages)';
}

/// PDF/A validation results. Freed via [close].
class PdfAResults implements Finalizable {
  PdfAResults._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_ValFreeC>>('pdf_pdf_a_results_free'));
  Pointer<Void> _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('PdfAResults is closed');
  }

  /// Whether the document is PDF/A compliant.
  bool isCompliant() {
    _check();
    final code = calloc<Int32>();
    try {
      final r = _n.pdfAIsCompliant(_handle, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'pdfAIsCompliant');
      return r;
    } finally {
      calloc.free(code);
    }
  }

  /// The validation error messages.
  List<String> errors() {
    _check();
    final n = _n.pdfAErrorCount(_handle);
    final out = <String>[];
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < n; i++) {
        out.add(_takeString(
            _n.pdfAGetError(_handle, i, code), code.value, 'pdfAGetError'));
      }
      return out;
    } finally {
      calloc.free(code);
    }
  }

  /// The validation warning messages (PDF/A exposes warning counts only).
  List<String> warnings() {
    _check();
    // The C ABI exposes a warning count for PDF/A but no per-warning getter;
    // surface placeholder entries so warnings() stays a List like the others.
    final n = _n.pdfAWarningCount(_handle);
    return List<String>.generate(n, (i) => 'warning $i');
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.pdfAResultsFree(_handle);
      _handle = nullptr;
    }
  }
}

/// PDF/UA accessibility validation results. Freed via [close].
class UaResults implements Finalizable {
  UaResults._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_ValFreeC>>('pdf_pdf_ua_results_free'));
  Pointer<Void> _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('UaResults is closed');
  }

  /// Whether the document is PDF/UA accessible.
  bool isAccessible() {
    _check();
    final code = calloc<Int32>();
    try {
      final r = _n.pdfUaIsAccessible(_handle, code);
      if (code.value != 0) {
        throw PdfOxideError(code.value, 'pdfUaIsAccessible');
      }
      return r;
    } finally {
      calloc.free(code);
    }
  }

  /// The accessibility error messages.
  List<String> errors() {
    _check();
    final n = _n.pdfUaErrorCount(_handle);
    final out = <String>[];
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < n; i++) {
        out.add(_takeString(
            _n.pdfUaGetError(_handle, i, code), code.value, 'pdfUaGetError'));
      }
      return out;
    } finally {
      calloc.free(code);
    }
  }

  /// The accessibility warning messages.
  List<String> warnings() {
    _check();
    final n = _n.pdfUaWarningCount(_handle);
    final out = <String>[];
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < n; i++) {
        out.add(_takeString(_n.pdfUaGetWarning(_handle, i, code), code.value,
            'pdfUaGetWarning'));
      }
      return out;
    } finally {
      calloc.free(code);
    }
  }

  /// Element statistics gathered during accessibility validation.
  UaStats uaStats() {
    _check();
    final s = calloc<Int32>();
    final im = calloc<Int32>();
    final t = calloc<Int32>();
    final f = calloc<Int32>();
    final a = calloc<Int32>();
    final p = calloc<Int32>();
    final code = calloc<Int32>();
    try {
      final ok = _n.pdfUaGetStats(_handle, s, im, t, f, a, p, code);
      if (!ok || code.value != 0) {
        throw PdfOxideError(code.value, 'pdfUaGetStats');
      }
      return UaStats(s.value, im.value, t.value, f.value, a.value, p.value);
    } finally {
      calloc.free(s);
      calloc.free(im);
      calloc.free(t);
      calloc.free(f);
      calloc.free(a);
      calloc.free(p);
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.pdfUaResultsFree(_handle);
      _handle = nullptr;
    }
  }
}

/// PDF/X validation results. Freed via [close].
class PdfXResults implements Finalizable {
  PdfXResults._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_ValFreeC>>('pdf_pdf_x_results_free'));
  Pointer<Void> _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('PdfXResults is closed');
  }

  /// Whether the document is PDF/X compliant.
  bool isCompliant() {
    _check();
    final code = calloc<Int32>();
    try {
      final r = _n.pdfXIsCompliant(_handle, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'pdfXIsCompliant');
      return r;
    } finally {
      calloc.free(code);
    }
  }

  /// The validation error messages.
  List<String> errors() {
    _check();
    final n = _n.pdfXErrorCount(_handle);
    final out = <String>[];
    final code = calloc<Int32>();
    try {
      for (var i = 0; i < n; i++) {
        out.add(_takeString(
            _n.pdfXGetError(_handle, i, code), code.value, 'pdfXGetError'));
      }
      return out;
    } finally {
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.pdfXResultsFree(_handle);
      _handle = nullptr;
    }
  }
}

// ── top-level Phase 6 signing / logging entry points ─────────────────────────

/// Sign raw PDF [pdf] bytes with [cert]; returns the signed PDF bytes.
Uint8List signBytes(
  Uint8List pdf,
  Certificate cert, {
  String reason = '',
  String location = '',
}) {
  cert._check();
  final buf = calloc<Uint8>(pdf.isEmpty ? 1 : pdf.length);
  if (pdf.isNotEmpty) buf.asTypedList(pdf.length).setAll(0, pdf);
  final cReason = reason.toNativeUtf8();
  final cLocation = location.toNativeUtf8();
  final len = calloc<IntPtr>();
  final code = calloc<Int32>();
  try {
    final p = _n.signBytes(
        buf, pdf.length, cert._handle, cReason, cLocation, len, code);
    return _takeBytes(p, len.value, code.value, 'signBytes');
  } finally {
    calloc.free(buf);
    calloc.free(cReason);
    calloc.free(cLocation);
    calloc.free(len);
    calloc.free(code);
  }
}

/// Sign raw PDF [pdf] bytes at a PAdES baseline [level] (0=B-B 1=B-T 2=B-LT).
/// [tsaUrl] is required for level >= 1. [certs]/[crls]/[ocsps] carry B-LT
/// revocation material (DER). Returns the signed PDF bytes.
Uint8List signBytesPades(
  Uint8List pdf,
  Certificate cert,
  int level, {
  String? tsaUrl,
  String reason = '',
  String location = '',
  List<Uint8List> certs = const [],
  List<Uint8List> crls = const [],
  List<Uint8List> ocsps = const [],
}) {
  cert._check();
  final buf = calloc<Uint8>(pdf.isEmpty ? 1 : pdf.length);
  if (pdf.isNotEmpty) buf.asTypedList(pdf.length).setAll(0, pdf);
  final cTsa = tsaUrl == null ? nullptr : tsaUrl.toNativeUtf8();
  final cReason = reason.toNativeUtf8();
  final cLocation = location.toNativeUtf8();
  final certArr = _ByteArrayArray(certs);
  final crlArr = _ByteArrayArray(crls);
  final ocspArr = _ByteArrayArray(ocsps);
  final len = calloc<IntPtr>();
  final code = calloc<Int32>();
  try {
    final p = _n.signBytesPades(
      buf,
      pdf.length,
      cert._handle,
      level,
      cTsa.cast(),
      cReason,
      cLocation,
      certArr.ptrs,
      certArr.lens,
      certArr.count,
      crlArr.ptrs,
      crlArr.lens,
      crlArr.count,
      ocspArr.ptrs,
      ocspArr.lens,
      ocspArr.count,
      len,
      code,
    );
    return _takeBytes(p, len.value, code.value, 'signBytesPades');
  } finally {
    calloc.free(buf);
    if (cTsa != nullptr) calloc.free(cTsa);
    calloc.free(cReason);
    calloc.free(cLocation);
    certArr.free();
    crlArr.free();
    ocspArr.free();
    calloc.free(len);
    calloc.free(code);
  }
}

/// Struct-options variant of [signBytesPades] — identical behaviour, marshalled
/// through the `PadesSignOptionsC` struct.
Uint8List signBytesPadesOpts(
  Uint8List pdf,
  Certificate cert,
  int level, {
  String? tsaUrl,
  String reason = '',
  String location = '',
  List<Uint8List> certs = const [],
  List<Uint8List> crls = const [],
  List<Uint8List> ocsps = const [],
}) {
  cert._check();
  final buf = calloc<Uint8>(pdf.isEmpty ? 1 : pdf.length);
  if (pdf.isNotEmpty) buf.asTypedList(pdf.length).setAll(0, pdf);
  final cTsa = tsaUrl == null ? nullptr : tsaUrl.toNativeUtf8();
  final cReason = reason.toNativeUtf8();
  final cLocation = location.toNativeUtf8();
  final certArr = _ByteArrayArray(certs);
  final crlArr = _ByteArrayArray(crls);
  final ocspArr = _ByteArrayArray(ocsps);
  final opts = calloc<_PadesSignOptionsC>();
  final len = calloc<IntPtr>();
  final code = calloc<Int32>();
  try {
    final o = opts.ref;
    o.certificateHandle = cert._handle;
    o.certs = certArr.ptrs;
    o.certLens = certArr.lens;
    o.nCerts = certArr.count;
    o.crls = crlArr.ptrs;
    o.crlLens = crlArr.lens;
    o.nCrls = crlArr.count;
    o.ocsps = ocspArr.ptrs;
    o.ocspLens = ocspArr.lens;
    o.nOcsps = ocspArr.count;
    o.tsaUrl = cTsa.cast();
    o.reason = cReason;
    o.location = cLocation;
    o.level = level;
    final p = _n.signBytesPadesOpts(buf, pdf.length, opts, len, code);
    return _takeBytes(p, len.value, code.value, 'signBytesPadesOpts');
  } finally {
    calloc.free(buf);
    if (cTsa != nullptr) calloc.free(cTsa);
    calloc.free(cReason);
    calloc.free(cLocation);
    certArr.free();
    crlArr.free();
    ocspArr.free();
    calloc.free(opts);
    calloc.free(len);
    calloc.free(code);
  }
}

/// Set the global library log level (0=Off 1=Error 2=Warn 3=Info 4=Debug 5=Trace).
void setLogLevel(int level) => _n.setLogLevel(level);

/// Get the current global library log level (0-5).
int getLogLevel() => _n.getLogLevel();

// ════════════════════════════════════════════════════════════════════════════
// Phase 7 — barcodes / QR / OCR / render-variants / redaction / from_* / page
// getters / element-lists / timestamp.
//
// Style matches earlier phases: every opaque native handle is wrapped in a
// `Finalizable` class freed via its `*_free` symbol on `close()` (idempotent) and
// a `NativeFinalizer`; calls after close throw `StateError`. Owned `char*` go
// through `_takeString` (+ `free_string`); owned `uint8*` buffers are copied then
// released with `free_bytes`. Render variants reuse [RenderedImage]; barcode
// stamping and redaction are methods on [DocumentEditor]; image / HTML+CSS
// constructors are static factories on [Pdf].
// ════════════════════════════════════════════════════════════════════════════

/// A generated 1-D barcode or QR code. Created via [BarcodeImage.qr] /
/// [BarcodeImage.barcode]; freed via [close] (or the finalizer). Stamp onto a
/// page with [DocumentEditor.addBarcodeToPage].
class BarcodeImage implements Finalizable {
  BarcodeImage._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_PtrFreeC>>('pdf_barcode_free'));
  Pointer<Void> _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('BarcodeImage is closed');
  }

  /// Generate a QR code encoding [data]. [errorCorrection] is the EC level code;
  /// [sizePx] the target pixel size.
  static BarcodeImage qr(String data,
      {int errorCorrection = 0, int sizePx = 256}) {
    final c = data.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.generateQrCode(c, errorCorrection, sizePx, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'generateQrCode');
      return BarcodeImage._(h);
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  /// Generate a 1-D barcode encoding [data] in the given [format] code.
  /// [sizePx] is the target pixel size.
  static BarcodeImage barcode(String data, int format, {int sizePx = 256}) {
    final c = data.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.generateBarcode(c, format, sizePx, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'generateBarcode');
      return BarcodeImage._(h);
    } finally {
      calloc.free(c);
      calloc.free(code);
    }
  }

  /// The encoded payload string.
  String get data {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(
          _n.barcodeGetData(_handle, code), code.value, 'barcodeGetData');
    } finally {
      calloc.free(code);
    }
  }

  /// The barcode format code.
  int get format {
    _check();
    final code = calloc<Int32>();
    try {
      final v = _n.barcodeGetFormat(_handle, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'barcodeGetFormat');
      return v;
    } finally {
      calloc.free(code);
    }
  }

  /// The detection confidence (0.0..1.0) for decoded barcodes.
  double get confidence {
    _check();
    final code = calloc<Int32>();
    try {
      final v = _n.barcodeGetConfidence(_handle, code);
      if (code.value != 0) {
        throw PdfOxideError(code.value, 'barcodeGetConfidence');
      }
      return v;
    } finally {
      calloc.free(code);
    }
  }

  /// Encode the barcode as PNG bytes at [sizePx] pixels.
  Uint8List imagePng({int sizePx = 256}) {
    _check();
    final len = calloc<Int32>();
    final code = calloc<Int32>();
    try {
      final p = _n.barcodeGetImagePng(_handle, sizePx, len, code);
      return _takeBytes(p, len.value, code.value, 'barcodeGetImagePng');
    } finally {
      calloc.free(len);
      calloc.free(code);
    }
  }

  /// Render the barcode as an SVG document string at [sizePx] pixels.
  String svg({int sizePx = 256}) {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(
          _n.barcodeGetSvg(_handle, sizePx, code), code.value, 'barcodeGetSvg');
    } finally {
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.barcodeFree(_handle);
      _handle = nullptr;
    }
  }
}

/// An OCR engine backed by detection/recognition model files. Created via
/// [OcrEngine.create]; freed via [close] (or the finalizer). Pass to
/// [PdfDocument.ocrExtractText].
class OcrEngine implements Finalizable {
  OcrEngine._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_PtrFreeC>>('pdf_ocr_engine_free'));
  Pointer<Void> _handle;

  Pointer<Void> get _handlePtr {
    if (_handle == nullptr) throw StateError('OcrEngine is closed');
    return _handle;
  }

  /// Create an OCR engine from the detection model, recognition model and
  /// dictionary file paths. Throws [PdfOxideError] if the models cannot load.
  static OcrEngine create(
      String detModelPath, String recModelPath, String dictPath) {
    final cDet = detModelPath.toNativeUtf8();
    final cRec = recModelPath.toNativeUtf8();
    final cDict = dictPath.toNativeUtf8();
    final code = calloc<Int32>();
    try {
      final h = _n.ocrEngineCreate(cDet, cRec, cDict, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'ocrEngineCreate');
      return OcrEngine._(h);
    } finally {
      calloc.free(cDet);
      calloc.free(cRec);
      calloc.free(cDict);
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.ocrEngineFree(_handle);
      _handle = nullptr;
    }
  }
}

/// A single layout element read from an [ElementList].
class Element {
  const Element(this.type, this.text, this.rect);

  /// The element type label (e.g. `Text`, `Image`, `Table`).
  final String type;

  /// The element's text content (may be empty).
  final String text;

  /// The element's bounding box in page user-space points.
  final Bbox rect;

  @override
  String toString() => 'Element($type, $rect)';
}

/// The layout elements of a page, produced by [PdfDocument.pageElements].
///
/// Owns the native `FfiElementList` handle (freed on [close] or by the
/// finalizer). Read [count] elements via [operator []] / [toList], or dump the
/// whole list to JSON via [toJson].
class ElementList implements Finalizable {
  ElementList._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_ListFreeC>>('pdf_oxide_elements_free'));
  Pointer<Void> _handle;

  void _check() {
    if (_handle == nullptr) throw StateError('ElementList is closed');
  }

  /// Number of elements in the list.
  int get count {
    _check();
    return _n.elementCount(_handle);
  }

  String _str(_ListStrD fn, int index, String op) {
    final code = calloc<Int32>();
    try {
      final p = fn(_handle, index, code);
      if (p == nullptr) {
        // Empty/absent strings are returned as null with code 0.
        if (code.value != 0) throw PdfOxideError(code.value, op);
        return '';
      }
      final s = p.toDartString();
      _n.freeString(p);
      return s;
    } finally {
      calloc.free(code);
    }
  }

  /// Read the element at [index].
  Element operator [](int index) {
    _check();
    final type = _str(_n.elementGetType, index, 'elementGetType');
    final text = _str(_n.elementGetText, index, 'elementGetText');
    final x = calloc<Float>();
    final y = calloc<Float>();
    final w = calloc<Float>();
    final h = calloc<Float>();
    final code = calloc<Int32>();
    try {
      _n.elementGetRect(_handle, index, x, y, w, h, code);
      if (code.value != 0) throw PdfOxideError(code.value, 'elementGetRect');
      return Element(type, text, Bbox(x.value, y.value, w.value, h.value));
    } finally {
      calloc.free(x);
      calloc.free(y);
      calloc.free(w);
      calloc.free(h);
      calloc.free(code);
    }
  }

  /// All elements as a list.
  List<Element> toList() => [for (var i = 0; i < count; i++) this[i]];

  /// Serialise the whole list to a JSON string.
  String toJson() {
    _check();
    final code = calloc<Int32>();
    try {
      return _takeString(
          _n.elementsToJson(_handle, code), code.value, 'elementsToJson');
    } finally {
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.elementsFree(_handle);
      _handle = nullptr;
    }
  }
}

/// A reusable page renderer configured with fixed DPI/format/quality. Created
/// via [Renderer.create]; freed via [close] (or the finalizer).
class Renderer implements Finalizable {
  Renderer._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final _finalizer = NativeFinalizer(
      _n.lib.lookup<NativeFunction<_PtrFreeC>>('pdf_renderer_free'));
  Pointer<Void> _handle;

  /// Create a renderer with the given [dpi], [format] (0=PNG 1=JPEG),
  /// [quality] (1..100) and [antiAlias] flag.
  static Renderer create(
      {int dpi = 150,
      int format = 0,
      int quality = 90,
      bool antiAlias = true}) {
    final code = calloc<Int32>();
    try {
      final h = _n.createRenderer(dpi, format, quality, antiAlias, code);
      if (h == nullptr) throw PdfOxideError(code.value, 'createRenderer');
      return Renderer._(h);
    } finally {
      calloc.free(code);
    }
  }

  /// Free the native handle now (idempotent).
  void close() {
    if (_handle != nullptr) {
      _finalizer.detach(this);
      _n.rendererFree(_handle);
      _handle = nullptr;
    }
  }
}

// ── Phase 8: office I/O / in-rect / auto / classify / furniture / forms /
//    doc structure / doc-level signatures / annotation extras / *_to_json /
//    crypto / models / config / streaming tables ──────────────────────────────
// Conventions match earlier phases: opaque handles are `Pointer<Void>`; owned
// `char*` go through `_takeString` (+ `free_string`); owned `uint8*` buffers go
// through `_takeBytes` (+ `free_bytes`); list handles are freed via their
// `*_list_free` symbol; out-params are `calloc`'d and freed in `finally`.

// office: open_from_{docx,pptx,xlsx}_bytes reuse _OpenBytesD; export reuse below.
typedef _DocBytesC = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<IntPtr>, Pointer<Int32>);
typedef _DocBytesD = Pointer<Uint8> Function(
    Pointer<Void>, Pointer<IntPtr>, Pointer<Int32>);

// in-rect: text returns char*, lists return a list handle.
typedef _RectStrC = Pointer<Utf8> Function(
    Pointer<Void>, Int32, Float, Float, Float, Float, Pointer<Int32>);
typedef _RectStrD = Pointer<Utf8> Function(
    Pointer<Void>, int, double, double, double, double, Pointer<Int32>);
typedef _RectListC = Pointer<Void> Function(
    Pointer<Void>, Int32, Float, Float, Float, Float, Pointer<Int32>);
typedef _RectListD = Pointer<Void> Function(
    Pointer<Void>, int, double, double, double, double, Pointer<Int32>);

// auto / classify: text_auto reuses _TextD; all_text / classify_document /
// outline / page_labels / xmp reuse _TextAllD; extract_page_auto needs options.
typedef _PageAutoC = Pointer<Utf8> Function(
    Pointer<Void>, Int32, Pointer<Utf8>, Pointer<Int32>);
typedef _PageAutoD = Pointer<Utf8> Function(
    Pointer<Void>, int, Pointer<Utf8>, Pointer<Int32>);

// furniture: erase_{header,footer,artifacts} reuse _DeI32I32D (handle,page,err).
typedef _RemoveC = Int32 Function(Pointer<Void>, Float, Pointer<Int32>);
typedef _RemoveD = int Function(Pointer<Void>, double, Pointer<Int32>);

// forms.
typedef _ExportFormC = Pointer<Uint8> Function(
    Pointer<Void>, Int32, Pointer<IntPtr>, Pointer<Int32>);
typedef _ExportFormD = Pointer<Uint8> Function(
    Pointer<Void>, int, Pointer<IntPtr>, Pointer<Int32>);
typedef _ImportFormC = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _ImportFormD = int Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _ImportFdfC = Int32 Function(
    Pointer<Void>, Pointer<Uint8>, IntPtr, Pointer<Int32>);
typedef _ImportFdfD = int Function(
    Pointer<Void>, Pointer<Uint8>, int, Pointer<Int32>);
typedef _FormImportFileC = Bool Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _FormImportFileD = bool Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _FormFieldsC = Pointer<Void> Function(Pointer<Void>, Pointer<Int32>);
typedef _FormFieldsD = Pointer<Void> Function(Pointer<Void>, Pointer<Int32>);
typedef _FfCountC = Int32 Function(Pointer<Void>);
typedef _FfCountD = int Function(Pointer<Void>);
typedef _FfStrC = Pointer<Utf8> Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _FfStrD = Pointer<Utf8> Function(Pointer<Void>, int, Pointer<Int32>);
typedef _FfBoolC = Bool Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _FfBoolD = bool Function(Pointer<Void>, int, Pointer<Int32>);
typedef _FfFreeC = Void Function(Pointer<Void>);
typedef _FfFreeD = void Function(Pointer<Void>);

// doc structure: outline/page_labels/xmp reuse _TextAllD; source_bytes reuse
// _DocBytesD; has_xfa reuse _BoolD; plan_split_by_bookmarks needs options.
typedef _PlanSplitC = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);
typedef _PlanSplitD = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Int32>);

// doc-level signatures.
typedef _DocSignC = Int32 Function(
    Pointer<Void>, Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _DocSignD = int Function(
    Pointer<Void>, Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Int32>);
typedef _DocGetSigC = Pointer<Void> Function(
    Pointer<Void>, Int32, Pointer<Int32>);
typedef _DocGetSigD = Pointer<Void> Function(
    Pointer<Void>, int, Pointer<Int32>);
typedef _DocSigCountC = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _DocSigCountD = int Function(Pointer<Void>, Pointer<Int32>);
typedef _DocGetDssC = Pointer<Void> Function(Pointer<Void>, Pointer<Int32>);
typedef _DocGetDssD = Pointer<Void> Function(Pointer<Void>, Pointer<Int32>);

// annotation extras (list, index, err).
typedef _AnnU32C = Uint32 Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _AnnU32D = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _AnnI64C = Int64 Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _AnnI64D = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _AnnBoolC = Bool Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _AnnBoolD = bool Function(Pointer<Void>, int, Pointer<Int32>);
typedef _AnnStrC = Pointer<Utf8> Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _AnnStrD = Pointer<Utf8> Function(Pointer<Void>, int, Pointer<Int32>);
typedef _AnnJsonC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _AnnJsonD = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _QuadCountC = Int32 Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _QuadCountD = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _QuadPointC = Void Function(
    Pointer<Void>,
    Int32,
    Int32,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Int32>);
typedef _QuadPointD = void Function(
    Pointer<Void>,
    int,
    int,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Int32>);

// list -> json + font size.
typedef _ListJsonC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _ListJsonD = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int32>);
typedef _FontSizeC = Float Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _FontSizeD = double Function(Pointer<Void>, int, Pointer<Int32>);

// crypto / models / config: nullary-string / nullary-int / string-arg.
typedef _NullStrC = Pointer<Utf8> Function();
typedef _NullStrD = Pointer<Utf8> Function();
typedef _NullI32C = Int32 Function();
typedef _NullI32D = int Function();
typedef _StrArgI32C = Int32 Function(Pointer<Utf8>);
typedef _StrArgI32D = int Function(Pointer<Utf8>);
typedef _PrefetchC = Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _PrefetchD = Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Int32>);
typedef _SetI64C = Int64 Function(Int64);
typedef _SetI64D = int Function(int);
typedef _SetI32C = Int32 Function(Int32);
typedef _SetI32D = int Function(int);
typedef _ConvPdfAC = Bool Function(Pointer<Void>, Int32, Pointer<Int32>);
typedef _ConvPdfAD = bool Function(Pointer<Void>, int, Pointer<Int32>);

// streaming tables (on FfiPageBuilder).
typedef _StBeginC = Int32 Function(
    Pointer<Void>,
    IntPtr,
    Pointer<Pointer<Utf8>>,
    Pointer<Float>,
    Pointer<Int32>,
    Int32,
    Pointer<Int32>);
typedef _StBeginD = int Function(Pointer<Void>, int, Pointer<Pointer<Utf8>>,
    Pointer<Float>, Pointer<Int32>, int, Pointer<Int32>);
typedef _StBeginV2C = Int32 Function(
    Pointer<Void>,
    IntPtr,
    Pointer<Pointer<Utf8>>,
    Pointer<Float>,
    Pointer<Int32>,
    Int32,
    Int32,
    IntPtr,
    Float,
    Float,
    IntPtr,
    Pointer<Int32>);
typedef _StBeginV2D = int Function(
    Pointer<Void>,
    int,
    Pointer<Pointer<Utf8>>,
    Pointer<Float>,
    Pointer<Int32>,
    int,
    int,
    int,
    double,
    double,
    int,
    Pointer<Int32>);
typedef _StPushRowC = Int32 Function(
    Pointer<Void>, IntPtr, Pointer<Pointer<Utf8>>, Pointer<Int32>);
typedef _StPushRowD = int Function(
    Pointer<Void>, int, Pointer<Pointer<Utf8>>, Pointer<Int32>);
typedef _StPushRowV2C = Int32 Function(Pointer<Void>, IntPtr,
    Pointer<Pointer<Utf8>>, Pointer<IntPtr>, Pointer<Int32>);
typedef _StPushRowV2D = int Function(Pointer<Void>, int, Pointer<Pointer<Utf8>>,
    Pointer<IntPtr>, Pointer<Int32>);
typedef _StStatusC = Int32 Function(Pointer<Void>, Pointer<Int32>);
typedef _StStatusD = int Function(Pointer<Void>, Pointer<Int32>);
typedef _StSetBatchC = Int32 Function(Pointer<Void>, IntPtr, Pointer<Int32>);
typedef _StSetBatchD = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef _StUsizeC = IntPtr Function(Pointer<Void>);
typedef _StUsizeD = int Function(Pointer<Void>);

// ── top-level Phase 7 entry points ───────────────────────────────────────────

/// Merge the PDFs at [paths] (in order) into a single PDF; returns its bytes.
Uint8List pdfMerge(List<String> paths) {
  final n = paths.length;
  final arr = n == 0 ? nullptr : calloc<Pointer<Utf8>>(n);
  for (var i = 0; i < n; i++) {
    arr[i] = paths[i].toNativeUtf8();
  }
  final len = calloc<Int32>();
  final code = calloc<Int32>();
  try {
    final p = _n.merge(arr, n, len, code);
    return _takeBytes(p, len.value, code.value, 'pdfMerge');
  } finally {
    for (var i = 0; i < n; i++) {
      calloc.free(arr[i]);
    }
    if (arr != nullptr) calloc.free(arr);
    calloc.free(len);
    calloc.free(code);
  }
}

/// Apply an RFC 3161 timestamp to the signature at [sigIndex] of the PDF [pdf]
/// bytes, contacting the TSA at [tsaUrl]; returns the timestamped PDF bytes.
Uint8List addTimestamp(Uint8List pdf, int sigIndex, String tsaUrl) {
  final buf = calloc<Uint8>(pdf.isEmpty ? 1 : pdf.length);
  if (pdf.isNotEmpty) buf.asTypedList(pdf.length).setAll(0, pdf);
  final cUrl = tsaUrl.toNativeUtf8();
  final outData = calloc<Pointer<Uint8>>();
  final outLen = calloc<IntPtr>();
  final code = calloc<Int32>();
  try {
    final ok =
        _n.addTimestamp(buf, pdf.length, sigIndex, cUrl, outData, outLen, code);
    if (!ok || outData.value == nullptr) {
      throw PdfOxideError(code.value, 'addTimestamp');
    }
    final out = Uint8List.fromList(
        outData.value.asTypedList(outLen.value < 0 ? 0 : outLen.value));
    _n.freeBytes(outData.value);
    return out;
  } finally {
    calloc.free(buf);
    calloc.free(cUrl);
    calloc.free(outData);
    calloc.free(outLen);
    calloc.free(code);
  }
}

// ── Phase 8: crypto / FIPS, models / prefetch, global config ─────────────────

String _nullStr(_NullStrD fn, String op) {
  final p = fn();
  if (p == nullptr) throw PdfOxideError(0, op);
  final s = p.toDartString();
  _n.freeString(p);
  return s;
}

/// The name of the active cryptographic provider.
String cryptoActiveProvider() =>
    _nullStr(_n.cryptoActiveProvider, 'cryptoActiveProvider');

/// The cryptographic bill of materials (CBOM) as JSON.
String cryptoCbom() => _nullStr(_n.cryptoCbom, 'cryptoCbom');

/// The cryptographic inventory as JSON.
String cryptoInventory() => _nullStr(_n.cryptoInventory, 'cryptoInventory');

/// The active cryptographic policy as JSON.
String cryptoPolicy() => _nullStr(_n.cryptoPolicy, 'cryptoPolicy');

/// Whether a FIPS-validated provider is available (non-zero = available).
int cryptoFipsAvailable() => _n.cryptoFipsAvailable();

/// Switch to a FIPS-validated provider; returns the C-ABI status.
int cryptoUseFips() => _n.cryptoUseFips();

/// Set the cryptographic policy from a [spec] string; returns the status.
int cryptoSetPolicy(String spec) {
  final c = spec.toNativeUtf8();
  try {
    return _n.cryptoSetPolicy(c);
  } finally {
    calloc.free(c);
  }
}

/// The bundled model manifest as JSON.
String modelManifest() => _nullStr(_n.modelManifest, 'modelManifest');

/// Whether model prefetch is available in this build (non-zero = available).
int prefetchAvailable() => _n.prefetchAvailable();

/// Prefetch OCR/layout models for the comma-separated [languagesCsv]; returns a
/// JSON report.
String prefetchModels(String languagesCsv) {
  final c = languagesCsv.toNativeUtf8();
  final code = calloc<Int32>();
  try {
    return _takeString(
        _n.prefetchModels(c, code), code.value, 'prefetchModels');
  } finally {
    calloc.free(c);
    calloc.free(code);
  }
}

/// Set the global per-content-stream operator [limit]; returns the prior value.
int setMaxOpsPerStream(int limit) => _n.setMaxOpsPerStream(limit);

/// Globally toggle preservation of unmapped glyphs ([preserve] != 0 = on);
/// returns the prior value.
int setPreserveUnmappedGlyphs(int preserve) =>
    _n.setPreserveUnmappedGlyphs(preserve);
