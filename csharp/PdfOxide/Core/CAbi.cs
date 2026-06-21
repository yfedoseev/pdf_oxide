using System;
using System.Runtime.InteropServices;
using PdfOxide.Internal;

namespace PdfOxide.Core
{
    /// <summary>
    /// Thin idiomatic wrappers over the canonical snake_case C-ABI symbols
    /// exported by the native library (<c>pdf_document_open</c>,
    /// <c>pdf_from_markdown</c>, …).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The high-level <see cref="PdfDocument"/>, <see cref="Pdf"/> and
    /// <see cref="DocumentEditor"/> types bind the PascalCase symbol aliases and
    /// manage handle lifetime via <c>SafeHandle</c>. This class exposes the
    /// equivalent raw C symbols directly for callers that want to drive the
    /// C-ABI without going through those wrappers — each method owns and frees
    /// any native resource it allocates within the call, so it does not change
    /// the existing lifetime model.
    /// </para>
    /// </remarks>
    public static class CAbi
    {
        // ── Document lifecycle ───────────────────────────────────────────────

        /// <summary>Opens a PDF from a file path via <c>pdf_document_open</c>.</summary>
        /// <returns>An opaque document handle; free it with <see cref="DocumentFree"/>.</returns>
        public static IntPtr DocumentOpen(string path)
        {
            var handle = NativeMethods.pdf_document_open(path, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            return handle;
        }

        /// <summary>Frees a document handle via <c>pdf_document_free</c> (null is a no-op).</summary>
        public static void DocumentFree(IntPtr handle) => NativeMethods.pdf_document_free(handle);

        /// <summary>Returns the page count via <c>pdf_document_get_page_count</c>.</summary>
        public static int DocumentGetPageCount(IntPtr handle)
        {
            var count = NativeMethods.pdf_document_get_page_count(handle, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            return count;
        }

        // ── Per-page extraction / conversion ─────────────────────────────────

        /// <summary>Extracts a page's text via <c>pdf_document_extract_text</c>.</summary>
        public static string DocumentExtractText(IntPtr handle, int pageIndex)
        {
            var ptr = NativeMethods.pdf_document_extract_text(handle, pageIndex, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            return StringMarshaler.PtrToStringAndFree(ptr);
        }

        /// <summary>Converts a page to Markdown via <c>pdf_document_to_markdown</c>.</summary>
        public static string DocumentToMarkdown(IntPtr handle, int pageIndex)
        {
            var ptr = NativeMethods.pdf_document_to_markdown(handle, pageIndex, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            return StringMarshaler.PtrToStringAndFree(ptr);
        }

        /// <summary>Converts a page to HTML via <c>pdf_document_to_html</c>.</summary>
        public static string DocumentToHtml(IntPtr handle, int pageIndex)
        {
            var ptr = NativeMethods.pdf_document_to_html(handle, pageIndex, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            return StringMarshaler.PtrToStringAndFree(ptr);
        }

        /// <summary>Converts a page to plain text via <c>pdf_document_to_plain_text</c>.</summary>
        public static string DocumentToPlainText(IntPtr handle, int pageIndex)
        {
            var ptr = NativeMethods.pdf_document_to_plain_text(handle, pageIndex, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            return StringMarshaler.PtrToStringAndFree(ptr);
        }

        /// <summary>Copies the document's current source bytes via <c>pdf_document_get_source_bytes</c>.</summary>
        public static byte[] DocumentGetSourceBytes(IntPtr handle)
        {
            var ptr = NativeMethods.pdf_document_get_source_bytes(handle, out var len, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            if (ptr == IntPtr.Zero)
                return Array.Empty<byte>();
            try
            {
                var buffer = new byte[(int)len];
                Marshal.Copy(ptr, buffer, 0, buffer.Length);
                return buffer;
            }
            finally
            {
                NativeMethods.FreeBytes(ptr);
            }
        }

        // ── DocumentEditor ───────────────────────────────────────────────────

        /// <summary>Opens a PDF for editing via <c>document_editor_open</c>.</summary>
        /// <returns>An opaque editor handle; free it with <see cref="EditorFree"/>.</returns>
        public static IntPtr EditorOpen(string path)
        {
            var handle = NativeMethods.document_editor_open(path, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            return handle;
        }

        /// <summary>Frees an editor handle via <c>document_editor_free</c> (null is a no-op).</summary>
        public static void EditorFree(IntPtr handle) => NativeMethods.document_editor_free(handle);

        /// <summary>Saves an edited document to a path via <c>document_editor_save</c>.</summary>
        public static void EditorSave(IntPtr handle, string path)
        {
            NativeMethods.document_editor_save(handle, path, out var ec);
            ExceptionMapper.ThrowIfError(ec);
        }

        // ── Pdf builder ──────────────────────────────────────────────────────

        /// <summary>Builds a Pdf from Markdown via <c>pdf_from_markdown</c>.</summary>
        /// <returns>An opaque Pdf handle; free it with <see cref="PdfFree"/>.</returns>
        public static IntPtr PdfFromMarkdown(string markdown)
        {
            var handle = NativeMethods.pdf_from_markdown(markdown, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            return handle;
        }

        /// <summary>Builds a Pdf from HTML via <c>pdf_from_html</c>.</summary>
        public static IntPtr PdfFromHtml(string html)
        {
            var handle = NativeMethods.pdf_from_html(html, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            return handle;
        }

        /// <summary>Builds a Pdf from plain text via <c>pdf_from_text</c>.</summary>
        public static IntPtr PdfFromText(string text)
        {
            var handle = NativeMethods.pdf_from_text(text, out var ec);
            ExceptionMapper.ThrowIfError(ec);
            return handle;
        }

        /// <summary>Saves a Pdf handle to a path via <c>pdf_save</c>.</summary>
        public static void PdfSave(IntPtr handle, string path)
        {
            NativeMethods.pdf_save(handle, path, out var ec);
            ExceptionMapper.ThrowIfError(ec);
        }

        /// <summary>Frees a Pdf handle via <c>pdf_free</c> (null is a no-op).</summary>
        public static void PdfFree(IntPtr handle) => NativeMethods.pdf_free(handle);

        // ── Global toggles (return previous value, no error channel) ─────────

        /// <summary>
        /// Sets the global content-stream operator cap via
        /// <c>pdf_oxide_set_max_ops_per_stream</c>. Returns the previous cap.
        /// </summary>
        public static long SetMaxOpsPerStream(long limit) =>
            NativeMethods.pdf_oxide_set_max_ops_per_stream(limit);

        /// <summary>
        /// Toggles preservation of unmapped glyphs (U+FFFD) via
        /// <c>pdf_oxide_set_preserve_unmapped_glyphs</c>. Returns the previous value.
        /// </summary>
        public static int SetPreserveUnmappedGlyphs(bool preserve) =>
            NativeMethods.pdf_oxide_set_preserve_unmapped_glyphs(preserve ? 1 : 0);

        // ── Streaming table ──────────────────────────────────────────────────

        /// <summary>
        /// Opens a streaming table on a page builder via
        /// <c>pdf_page_builder_streaming_table_begin</c>. The three column arrays
        /// (<paramref name="headers"/>, <paramref name="widths"/>,
        /// <paramref name="aligns"/>) must be the same length.
        /// </summary>
        public static unsafe void PageBuilderStreamingTableBegin(
            IntPtr pageBuilder,
            string[] headers,
            float[] widths,
            int[] aligns,
            bool repeatHeader)
        {
            if (headers is null) throw new ArgumentNullException(nameof(headers));
            if (widths is null) throw new ArgumentNullException(nameof(widths));
            if (aligns is null) throw new ArgumentNullException(nameof(aligns));
            if (widths.Length != headers.Length || aligns.Length != headers.Length)
                throw new ArgumentException("headers, widths and aligns must have the same length.");

            var headerPtrs = new IntPtr[headers.Length];
            try
            {
                for (int i = 0; i < headers.Length; i++)
                    headerPtrs[i] = Marshal.StringToCoTaskMemUTF8(headers[i] ?? string.Empty);

                fixed (IntPtr* hp = headerPtrs)
                fixed (float* wp = widths)
                fixed (int* ap = aligns)
                {
                    NativeMethods.pdf_page_builder_streaming_table_begin(
                        pageBuilder,
                        (nuint)headers.Length,
                        (byte**)hp,
                        wp,
                        ap,
                        repeatHeader ? 1 : 0,
                        out var ec);
                    ExceptionMapper.ThrowIfError(ec);
                }
            }
            finally
            {
                foreach (var p in headerPtrs)
                    if (p != IntPtr.Zero) Marshal.FreeCoTaskMem(p);
            }
        }

        // ── Render with OCG layer filtering ──────────────────────────────────

        /// <summary>
        /// Renders a page with the full options surface plus OCG layer filtering
        /// via <c>pdf_render_page_with_options_ex</c>. Pass an empty / null
        /// <paramref name="excludedLayers"/> to disable filtering.
        /// </summary>
        /// <returns>An opaque rendered-image handle (free via the image API), or <see cref="IntPtr.Zero"/>.</returns>
        public static unsafe IntPtr RenderPageWithOptionsEx(
            IntPtr doc,
            int pageIndex,
            int dpi,
            int format,
            float bgR,
            float bgG,
            float bgB,
            float bgA,
            bool transparentBackground,
            bool renderAnnotations,
            int jpegQuality,
            string[]? excludedLayers)
        {
            var layers = excludedLayers ?? Array.Empty<string>();
            var layerPtrs = new IntPtr[layers.Length];
            try
            {
                for (int i = 0; i < layers.Length; i++)
                    layerPtrs[i] = Marshal.StringToCoTaskMemUTF8(layers[i] ?? string.Empty);

                fixed (IntPtr* lp = layerPtrs)
                {
                    var img = NativeMethods.pdf_render_page_with_options_ex(
                        doc,
                        pageIndex,
                        dpi,
                        format,
                        bgR, bgG, bgB, bgA,
                        transparentBackground ? 1 : 0,
                        renderAnnotations ? 1 : 0,
                        jpegQuality,
                        layers.Length == 0 ? null : (byte**)lp,
                        (nuint)layers.Length,
                        out var ec);
                    ExceptionMapper.ThrowIfError(ec);
                    return img;
                }
            }
            finally
            {
                foreach (var p in layerPtrs)
                    if (p != IntPtr.Zero) Marshal.FreeCoTaskMem(p);
            }
        }

        // ── PAdES (struct-options variant) ───────────────────────────────────

        /// <summary>
        /// Signs raw PDF bytes via the struct-options variant
        /// <c>pdf_sign_bytes_pades_opts</c>. <paramref name="options"/> must point
        /// to a <c>PadesSignOptionsC</c>. Returns the signed bytes.
        /// </summary>
        public static unsafe byte[] SignBytesPadesOpts(ReadOnlySpan<byte> pdfData, IntPtr options)
        {
            fixed (byte* p = pdfData)
            {
                byte* result = NativeMethods.pdf_sign_bytes_pades_opts(
                    p,
                    (nuint)pdfData.Length,
                    (void*)options,
                    out var outLen,
                    out var ec);
                ExceptionMapper.ThrowIfError(ec);
                if (result == null)
                    return Array.Empty<byte>();
                try
                {
                    var buffer = new byte[(int)outLen];
                    Marshal.Copy((IntPtr)result, buffer, 0, buffer.Length);
                    return buffer;
                }
                finally
                {
                    NativeMethods.FreeBytes((IntPtr)result);
                }
            }
        }
    }
}
