using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace PdfOxide.Internal
{
    /// <summary>
    /// P/Invoke declarations for the native pdf_oxide library.
    /// </summary>
    /// <remarks>
    /// This class declares all the FFI functions exported from the Rust library.
    /// All functions are blittable and use standard calling conventions for maximum compatibility.
    /// </remarks>
    internal static partial class NativeMethods
    {
        private const string LibName = "pdf_oxide";

        #region PdfDocument API

        /// <summary>
        /// Opens a PDF document from a file path.
        /// </summary>
        /// <param name="path">UTF-8 null-terminated file path.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Opaque handle to the PDF document, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle PdfDocumentOpen(
            string path,
            out int errorCode);

        /// <summary>
        /// Opens a PDF document from a memory buffer.
        /// </summary>
        /// <param name="data">Pointer to PDF bytes.</param>
        /// <param name="length">Length of the data buffer.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Opaque handle to the PDF document, or null on error.</returns>
        [LibraryImport(LibName, EntryPoint = "pdf_document_open_from_bytes", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle PdfDocumentOpenFromBytes(
            [In] byte[] data,
            int length,
            out int errorCode);

        /// <summary>
        /// Zero-copy overload of <c>pdf_document_open_from_bytes</c> that accepts a
        /// pinned-byte reference instead of a managed array. Used by
        /// <see cref="PdfOxide.Core.PdfDocument.Open(System.ReadOnlySpan{byte})"/>
        /// to forward a <see cref="System.ReadOnlySpan{T}"/> without the
        /// <see cref="System.IO.MemoryStream"/> hop the Stream overload takes.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_open_from_bytes", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle PdfDocumentOpenFromBytesRef(
            ref byte data,
            int length,
            out int errorCode);

        /// <summary>
        /// Frees a PdfDocument handle.
        /// </summary>
        /// <param name="handle">The handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfDocumentFree(IntPtr handle);

        /// <summary>
        /// Gets the PDF version.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="major">Output parameter for major version number.</param>
        /// <param name="minor">Output parameter for minor version number.</param>
        [LibraryImport(LibName, EntryPoint = "pdf_document_get_version", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfDocumentGetVersion(
            NativeHandle handle,
            out byte major,
            out byte minor);

        /// <summary>
        /// Gets the number of pages in the document.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The page count, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentGetPageCount(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Checks if the document has a structure tree (Tagged PDF).
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if the document has a structure tree, false otherwise.</returns>
        [LibraryImport(LibName, EntryPoint = "pdf_document_has_structure_tree", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool PdfDocumentHasStructureTree(NativeHandle handle);

        /// <summary>
        /// Extracts text from a page.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer (must be freed with FreeString).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentExtractText(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Converts a page to Markdown format.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer (must be freed with FreeString).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentToMarkdown(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Converts all pages to Markdown format.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer (must be freed with FreeString).</returns>
        [LibraryImport(LibName, EntryPoint = "pdf_document_to_markdown_all", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentToMarkdownAll(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Converts a page to HTML format.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer (must be freed with FreeString).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentToHtml(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Converts a page to plain text format.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer (must be freed with FreeString).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentToPlainText(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        #endregion

        #region Memory Management

        /// <summary>
        /// Frees a UTF-8 string allocated by Rust.
        /// </summary>
        /// <param name="ptr">Pointer to the string to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void FreeString(IntPtr ptr);

        /// <summary>
        /// Frees a byte buffer allocated by Rust via the C system allocator
        /// (malloc).  No length argument needed — malloc tracks the size.
        /// </summary>
        /// <param name="ptr">Pointer to the buffer to free.</param>
        [LibraryImport(LibName, EntryPoint = "free_bytes", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void FreeBytes(IntPtr ptr);

        #endregion

        #region JSON Bulk Extractors (cross-language DRY — one FFI crossing per list)

        /// <summary>Serializes a font list handle to a UTF-8 JSON C string (must be freed with <see cref="FreeString"/>).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_fonts_to_json", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfOxideFontsToJson(IntPtr fonts, out int errorCode);

        /// <summary>Serializes an annotation list handle to a UTF-8 JSON C string (must be freed with <see cref="FreeString"/>).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotations_to_json", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfOxideAnnotationsToJson(IntPtr annotations, out int errorCode);

        /// <summary>Serializes an element list handle to a UTF-8 JSON C string (must be freed with <see cref="FreeString"/>).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_elements_to_json", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfOxideElementsToJson(IntPtr elements, out int errorCode);

        /// <summary>Serializes a search results handle to a UTF-8 JSON C string (must be freed with <see cref="FreeString"/>).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_search_results_to_json", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfOxideSearchResultsToJson(IntPtr results, out int errorCode);

        #endregion

        #region Logging API

        /// <summary>
        /// Sets the global log level for the native library.
        /// </summary>
        /// <param name="level">Log level: 0=Off, 1=Error, 2=Warn, 3=Info, 4=Debug, 5=Trace.</param>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_set_log_level", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfOxideSetLogLevel(int level);

        /// <summary>
        /// Gets the current log level.
        /// </summary>
        /// <returns>Current log level (0-5).</returns>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_get_log_level", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfOxideGetLogLevel();

        // ── Crypto provider (issue #236) ─────────────────────────────

        /// <summary>
        /// Returns the name of the active cryptographic provider as
        /// a native UTF-8 string. Caller must free with FreeString.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_crypto_active_provider")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial nint PdfOxideCryptoActiveProvider();

        /// <summary>
        /// 1 if the FIPS-validated aws-lc-rs provider was compiled
        /// into this binary; 0 otherwise.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_crypto_fips_available")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfOxideCryptoFipsAvailable();

        /// <summary>
        /// Install the FIPS-validated aws-lc-rs provider as the
        /// process-wide active backend. Returns 0 on success, 1 if
        /// FIPS not compiled in, 2 if a provider is already set.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_crypto_use_fips")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfOxideCryptoUseFips();

        #endregion

        #region Pdf Creation API

        /// <summary>
        /// Creates a PDF from Markdown content.
        /// </summary>
        /// <param name="markdown">UTF-8 null-terminated Markdown content.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Opaque handle to Pdf, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle PdfFromMarkdown(
            string markdown,
            out int errorCode);

        /// <summary>
        /// Creates a PDF from HTML content.
        /// </summary>
        /// <param name="html">UTF-8 null-terminated HTML content.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Opaque handle to Pdf, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle PdfFromHtml(
            string html,
            out int errorCode);

        /// <summary>
        /// Creates a PDF from plain text content.
        /// </summary>
        /// <param name="text">UTF-8 null-terminated text content.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Opaque handle to Pdf, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle PdfFromText(
            string text,
            out int errorCode);

        /// <summary>
        /// Saves a PDF to file.
        /// </summary>
        /// <param name="handle">The PDF handle.</param>
        /// <param name="path">UTF-8 null-terminated output file path.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>0 on success, non-zero on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfSave(
            NativeHandle handle,
            string path,
            out int errorCode);

        /// <summary>
        /// Saves a PDF to memory buffer.
        /// </summary>
        /// <param name="handle">The PDF handle.</param>
        /// <param name="dataLen">Output parameter for buffer length.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Pointer to the PDF byte buffer — must be freed with <see cref="FreeBytes"/>, or <see cref="IntPtr.Zero"/> on error.</returns>
        [LibraryImport(LibName, EntryPoint = "pdf_save_to_bytes", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfSaveToBytes(
            NativeHandle handle,
            out int dataLen,
            out int errorCode);

        /// <summary>
        /// Gets the page count from a Pdf handle.
        /// </summary>
        /// <param name="handle">The PDF handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The page count, or -1 on error.</returns>
        [LibraryImport(LibName, EntryPoint = "pdf_get_page_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfGetPageCount(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Frees a Pdf handle.
        /// </summary>
        /// <param name="handle">The handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfFree(IntPtr handle);

        #endregion

        #region DocumentEditor API

        /// <summary>
        /// Opens a PDF document for editing.
        /// </summary>
        /// <param name="path">UTF-8 null-terminated file path.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Opaque handle to DocumentEditor, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle DocumentEditorOpen(
            string path,
            out int errorCode);

        /// <summary>
        /// Frees a DocumentEditor handle.
        /// </summary>
        /// <param name="handle">The handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void DocumentEditorFree(IntPtr handle);

        /// <summary>
        /// Checks if the document has been modified.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <returns>True if modified, false otherwise.</returns>
        [LibraryImport(LibName, EntryPoint = "document_editor_is_modified", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool DocumentEditorIsModified(IntPtr handle);

        /// <summary>
        /// Gets the source file path.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer (must be freed with FreeString).</returns>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_source_path", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr DocumentEditorGetSourcePath(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the PDF version.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <param name="major">Output parameter for major version number.</param>
        /// <param name="minor">Output parameter for minor version number.</param>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_version", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void DocumentEditorGetVersion(
            IntPtr handle,
            out byte major,
            out byte minor);

        /// <summary>
        /// Gets the number of pages in the document.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The page count, or -1 on error.</returns>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_page_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int DocumentEditorGetPageCount(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the document title.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer (must be freed with FreeString), or null if not set.</returns>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_title", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr DocumentEditorGetTitle(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Sets the document title.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <param name="title">UTF-8 null-terminated title string.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void DocumentEditorSetTitle(
            IntPtr handle,
            string title,
            out int errorCode);

        /// <summary>
        /// Gets the document author.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer (must be freed with FreeString), or null if not set.</returns>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_author", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr DocumentEditorGetAuthor(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Sets the document author.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <param name="author">UTF-8 null-terminated author string.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void DocumentEditorSetAuthor(
            IntPtr handle,
            string author,
            out int errorCode);

        /// <summary>
        /// Gets the document subject.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer (must be freed with FreeString), or null if not set.</returns>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_subject", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr DocumentEditorGetSubject(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Sets the document subject.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <param name="subject">UTF-8 null-terminated subject string.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        [LibraryImport(LibName, EntryPoint = "document_editor_set_subject", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void DocumentEditorSetSubject(
            IntPtr handle,
            string subject,
            out int errorCode);

        /// <summary>
        /// Saves the document to a file.
        /// </summary>
        /// <param name="handle">The editor handle.</param>
        /// <param name="path">UTF-8 null-terminated output file path.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>0 on success, non-zero on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int DocumentEditorSave(
            IntPtr handle,
            string path,
            out int errorCode);

        #endregion

        #region DOM API

        /// <summary>
        /// Gets the width of a page.
        /// </summary>
        /// <param name="handle">The page handle.</param>
        /// <returns>The page width in points, or 0 if invalid.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float PdfPageGetWidth(IntPtr handle);

        /// <summary>
        /// Gets the height of a page.
        /// </summary>
        /// <param name="handle">The page handle.</param>
        /// <returns>The page height in points, or 0 if invalid.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float PdfPageGetHeight(IntPtr handle);

        /// <summary>
        /// Gets the page index.
        /// </summary>
        /// <param name="handle">The page handle.</param>
        /// <returns>The page index (0-based), or -1 if invalid.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageGetIndex(IntPtr handle);

        /// <summary>
        /// Gets the page dimensions.
        /// </summary>
        /// <param name="handle">The page handle.</param>
        /// <param name="widthOut">Output parameter for page width.</param>
        /// <param name="heightOut">Output parameter for page height.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfPageGetDimensions(
            IntPtr handle,
            out float widthOut,
            out float heightOut);

        /// <summary>
        /// Frees a PdfPage handle.
        /// </summary>
        /// <param name="handle">The handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfPageFree(IntPtr handle);

        /// <summary>
        /// Gets the page dimensions from document by page index.
        /// </summary>
        /// <param name="documentHandle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="width">Output parameter for page width.</param>
        /// <param name="height">Output parameter for page height.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if successful, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_get_page_dimensions(
            IntPtr documentHandle,
            int pageIndex,
            out float width,
            out float height,
            out int errorCode);

        /// <summary>
        /// Gets the page rotation in degrees.
        /// </summary>
        /// <param name="documentHandle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Rotation angle (0, 90, 180, or 270 degrees).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_page_rotation(
            IntPtr documentHandle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Gets the page media box (full page size).
        /// </summary>
        /// <param name="documentHandle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="llx">Output lower-left X coordinate.</param>
        /// <param name="lly">Output lower-left Y coordinate.</param>
        /// <param name="urx">Output upper-right X coordinate.</param>
        /// <param name="ury">Output upper-right Y coordinate.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if successful, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_get_page_media_box(
            IntPtr documentHandle,
            int pageIndex,
            out float llx,
            out float lly,
            out float urx,
            out float ury,
            out int errorCode);

        /// <summary>
        /// Gets the page crop box (visible area).
        /// </summary>
        /// <param name="documentHandle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="llx">Output lower-left X coordinate.</param>
        /// <param name="lly">Output lower-left Y coordinate.</param>
        /// <param name="urx">Output upper-right X coordinate.</param>
        /// <param name="ury">Output upper-right Y coordinate.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if successful, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_get_page_crop_box(
            IntPtr documentHandle,
            int pageIndex,
            out float llx,
            out float lly,
            out float urx,
            out float ury,
            out int errorCode);

        #endregion

        #region Element API

        /// <summary>
        /// Gets the number of elements of a specific type on a page.
        /// </summary>
        /// <param name="handle">The page handle.</param>
        /// <param name="elementType">The type of element to count (ELEMENT_TYPE_*).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The number of elements found, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageFindElementsCount(
            IntPtr handle,
            int elementType,
            out int errorCode);

        /// <summary>
        /// Gets the text content of a text element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfTextElementGetContent(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the font size of a text element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The font size in points.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float PdfTextElementGetFontSize(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the font name of a text element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated font name string. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfTextElementGetFontName(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the text color of a text element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="r">Output parameter for red component (0-255).</param>
        /// <param name="g">Output parameter for green component (0-255).</param>
        /// <param name="b">Output parameter for blue component (0-255).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfTextElementGetColor(
            IntPtr handle,
            out byte r,
            out byte g,
            out byte b,
            out int errorCode);

        /// <summary>
        /// Gets whether a text element is bold.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if bold, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool PdfTextElementGetIsBold(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets whether a text element is italic.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if italic, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool PdfTextElementGetIsItalic(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the bounding box of an element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="x">Output parameter for x coordinate.</param>
        /// <param name="y">Output parameter for y coordinate.</param>
        /// <param name="width">Output parameter for width.</param>
        /// <param name="height">Output parameter for height.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfElementGetBbox(
            IntPtr handle,
            out float x,
            out float y,
            out float width,
            out float height);

        /// <summary>
        /// Gets the element type as an integer constant.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <returns>The element type constant (ELEMENT_TYPE_*), or -1 if invalid.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfElementGetType(IntPtr handle);

        /// <summary>
        /// Frees an element handle.
        /// </summary>
        /// <param name="handle">The element handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfElementFree(IntPtr handle);

        /// <summary>
        /// Gets the format of an image element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The image format constant (0=JPEG, 1=PNG, 2=TIFF, 3=Unknown).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfImageElementGetFormat(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the dimensions of an image element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="width">Output parameter for width in pixels.</param>
        /// <param name="height">Output parameter for height in pixels.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfImageElementGetDimensions(
            IntPtr handle,
            out uint width,
            out uint height,
            out int errorCode);

        /// <summary>
        /// Gets the size of the raw image data.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The size in bytes of the image data, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfImageElementGetDataSize(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the raw image data from an image element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="data">Output buffer for image data.</param>
        /// <param name="maxLen">Maximum length of data buffer.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The number of bytes written to data buffer, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfImageElementGetData(
            IntPtr handle,
            byte[] data,
            int maxLen,
            out int errorCode);

        /// <summary>
        /// Gets the alternative text of an image element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated alt text string. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfImageElementGetAltText(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the horizontal DPI of an image element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The horizontal DPI, or -1 if not available.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float PdfImageElementGetHorizontalDpi(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the vertical DPI of an image element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The vertical DPI, or -1 if not available.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float PdfImageElementGetVerticalDpi(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets whether an image element is grayscale.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if the image is grayscale, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool PdfImageElementGetIsGrayscale(
            IntPtr handle,
            out int errorCode);

        #region Path Element API

        /// <summary>
        /// Gets the stroke color of a path element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="r">Output parameter for red component (0-255).</param>
        /// <param name="g">Output parameter for green component (0-255).</param>
        /// <param name="b">Output parameter for blue component (0-255).</param>
        /// <param name="hasColor">Output parameter for whether the path has a stroke color.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfPathElementGetStrokeColor(
            IntPtr handle,
            out byte r,
            out byte g,
            out byte b,
            [MarshalAs(UnmanagedType.I1)] out bool hasColor,
            out int errorCode);

        /// <summary>
        /// Gets the fill color of a path element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="r">Output parameter for red component (0-255).</param>
        /// <param name="g">Output parameter for green component (0-255).</param>
        /// <param name="b">Output parameter for blue component (0-255).</param>
        /// <param name="hasColor">Output parameter for whether the path has a fill color.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfPathElementGetFillColor(
            IntPtr handle,
            out byte r,
            out byte g,
            out byte b,
            [MarshalAs(UnmanagedType.I1)] out bool hasColor,
            out int errorCode);

        /// <summary>
        /// Gets the line width of a path element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The line width in points, or 0 if not stroked.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float PdfPathElementGetLineWidth(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the fill mode of a path element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The fill mode (PathFillMode enum value).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPathElementGetFillMode(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the stroke style of a path element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The stroke style (PathStrokeStyle enum value).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPathElementGetStrokeStyle(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the number of rows in a table element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The number of rows, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfTableElementGetRowCount(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the number of columns in a table element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The number of columns, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfTableElementGetColumnCount(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the structure type of a structure element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfStructureElementGetStructureType(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the alt text of a structure element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer, or null if not set. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfStructureElementGetAltText(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the actual text of a structure element.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer, or null if not set. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfStructureElementGetActualText(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets whether a structure element is marked as removed.
        /// </summary>
        /// <param name="handle">The element handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if marked as removed, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool PdfStructureElementGetIsRemoved(
            IntPtr handle,
            out int errorCode);

        #endregion

        #endregion

        #region Annotation API

        // === Document-level annotation list (pdf_oxide_annotation_* / pdf_document_get_page_annotations) ===

        /// <summary>
        /// Gets the annotation list handle for a given page of a document.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Opaque handle to the annotation list. Must be freed with <see cref="PdfOxideAnnotationListFree"/>.</returns>
        [LibraryImport(LibName, EntryPoint = "pdf_document_get_page_annotations", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentGetPageAnnotations(
            IntPtr handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Returns the number of annotations in an annotation list.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfOxideAnnotationCount(IntPtr annotations);

        /// <summary>
        /// Returns the type string of the annotation at <paramref name="index"/> as a UTF-8 pointer (must be freed with <see cref="FreeString"/>).
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_type", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfOxideAnnotationGetType(
            IntPtr annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Returns the content string of the annotation at <paramref name="index"/> as a UTF-8 pointer (must be freed with <see cref="FreeString"/>).
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_content", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfOxideAnnotationGetContent(
            IntPtr annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the bounding rectangle of the annotation at <paramref name="index"/>.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_rect", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfOxideAnnotationGetRect(
            IntPtr annotations,
            int index,
            out float x,
            out float y,
            out float width,
            out float height,
            out int errorCode);

        /// <summary>
        /// Frees an annotation list handle obtained from <see cref="PdfDocumentGetPageAnnotations"/>.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_list_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfOxideAnnotationListFree(IntPtr handle);

        // === Per-annotation API (pdf_annotation_*) ===

        /// <summary>
        /// Gets the number of annotations on a page.
        /// </summary>
        /// <param name="handle">The page handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The number of annotations found, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageGetAnnotationsCount(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the count of annotations of a specific type on a page.
        /// </summary>
        /// <param name="handle">The page handle.</param>
        /// <param name="annotationType">The type of annotation to count (ANNOTATION_TYPE_*).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The number of annotations of that type, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageGetAnnotationsByTypeCount(
            IntPtr handle,
            int annotationType,
            out int errorCode);

        /// <summary>
        /// Gets the type of an annotation.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <returns>The annotation type constant (ANNOTATION_TYPE_*), or -1 if invalid.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfAnnotationGetType(IntPtr handle);

        /// <summary>
        /// Gets the contents/text of an annotation.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfAnnotationGetContents(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the subject of an annotation.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfAnnotationGetSubject(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the author of an annotation.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfAnnotationGetAuthor(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the bounding box of an annotation.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="x">Output parameter for x coordinate.</param>
        /// <param name="y">Output parameter for y coordinate.</param>
        /// <param name="width">Output parameter for width.</param>
        /// <param name="height">Output parameter for height.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfAnnotationGetBbox(
            IntPtr handle,
            out float x,
            out float y,
            out float width,
            out float height);

        /// <summary>
        /// Gets the color of an annotation as RGB values (0.0-1.0).
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="r">Output parameter for red component.</param>
        /// <param name="g">Output parameter for green component.</param>
        /// <param name="b">Output parameter for blue component.</param>
        /// <param name="hasColor">Output parameter for whether color was found (1=yes, 0=no).</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfAnnotationGetColor(
            IntPtr handle,
            out float r,
            out float g,
            out float b,
            out int hasColor);

        /// <summary>
        /// Gets the opacity of an annotation (0.0-1.0).
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The opacity value (1.0 if not set).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float PdfAnnotationGetOpacity(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets flags for an annotation (visibility, printability, etc.).
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The flags as a bitmask.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfAnnotationGetFlags(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Text annotation specific: Gets the icon type.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The icon type code (0=Comment, 1=Key, 2=Note, 3=Help, etc., -1=Unknown).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfTextAnnotationGetIcon(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Text annotation specific: Gets whether the annotation is open.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>1 if open, 0 if closed.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfTextAnnotationGetOpen(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Link annotation specific: Gets the URI of a link.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfLinkAnnotationGetUri(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Link annotation specific: Gets the destination page index.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The page index, or -1 if not a page link or error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfLinkAnnotationGetPage(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Text markup annotation specific: Gets the markup type.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The markup type (0=Highlight, 1=Underline, 2=StrikeOut, 3=Squiggly, -1=Unknown).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfTextMarkupAnnotationGetType(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// FreeText annotation specific: Gets the font name.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfFreeTextAnnotationGetFontName(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// FreeText annotation specific: Gets the font size.
        /// </summary>
        /// <param name="handle">The annotation handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The font size in points.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float PdfFreeTextAnnotationGetFontSize(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Frees an annotation handle.
        /// </summary>
        /// <param name="handle">The annotation handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfAnnotationFree(IntPtr handle);

        #endregion

        #region Search API

        /// <summary>
        /// Searches for text on a page.
        /// </summary>
        /// <param name="pageHandle">The page handle.</param>
        /// <param name="searchTerm">The UTF-8 search term to find.</param>
        /// <param name="caseSensitive">Whether to match case (1=yes, 0=no).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The number of occurrences found, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageSearchText(
            IntPtr pageHandle,
            string searchTerm,
            int caseSensitive,
            out int errorCode);

        /// <summary>
        /// Gets the text content of a search result.
        /// </summary>
        /// <param name="handle">The search result handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfSearchResultGetText(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets the bounding box of a search result.
        /// </summary>
        /// <param name="handle">The search result handle.</param>
        /// <param name="x">Output parameter for x coordinate.</param>
        /// <param name="y">Output parameter for y coordinate.</param>
        /// <param name="width">Output parameter for width.</param>
        /// <param name="height">Output parameter for height.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfSearchResultGetBbox(
            IntPtr handle,
            out float x,
            out float y,
            out float width,
            out float height);

        /// <summary>
        /// Gets the page index of a search result.
        /// </summary>
        /// <param name="handle">The search result handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>The page index, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfSearchResultGetPage(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Frees a search result handle.
        /// </summary>
        /// <param name="handle">The search result handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfSearchResultFree(IntPtr handle);

        #endregion

        #region Utility Functions

        /// <summary>
        /// Allocates a string in Rust memory.
        /// </summary>
        /// <param name="s">UTF-8 null-terminated string pointer.</param>
        /// <returns>Allocated string pointer (must be freed with FreeString).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr AllocString(
            string s);

        #endregion

        #region Rendering API

        /// <summary>
        /// Creates a PDF renderer with specified options.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="dpi">Resolution in DPI (e.g., 150, 300).</param>
        /// <param name="colorSpace">Color space (0=RGB, 1=RGBA, 2=Grayscale, 3=CMYK).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Renderer handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_create_renderer(
            NativeHandle handle,
            float dpi,
            int colorSpace,
            out int errorCode);

        /// <summary>
        /// Renders a page to an image buffer.
        /// </summary>
        /// <param name="rendererHandle">The renderer handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Image handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_render_page(
            NativeHandle rendererHandle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Full-option render — mirrors Rust `RenderOptions` surface.
        /// Takes raw <see cref="IntPtr"/> to match the existing
        /// <c>pdf_render_page</c> basic overload's handle convention.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_render_page_with_options", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfRenderPageWithOptions(
            IntPtr docHandle,
            int pageIndex,
            int dpi,
            int format,
            float bgR,
            float bgG,
            float bgB,
            float bgA,
            int transparentBackground,
            int renderAnnotations,
            int jpegQuality,
            out int errorCode);

        /// <summary>
        /// Gets the width of a rendered image.
        /// </summary>
        /// <param name="imageHandle">The image handle.</param>
        /// <returns>Width in pixels.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_render_image_width(NativeHandle imageHandle);

        /// <summary>
        /// Gets the height of a rendered image.
        /// </summary>
        /// <param name="imageHandle">The image handle.</param>
        /// <returns>Height in pixels.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_render_image_height(NativeHandle imageHandle);

        /// <summary>
        /// Gets the image data as bytes.
        /// </summary>
        /// <param name="imageHandle">The image handle.</param>
        /// <param name="format">Output format (0=PNG, 1=JPEG, 2=BMP, 3=TIFF).</param>
        /// <param name="outputPtr">Output parameter for byte buffer pointer.</param>
        /// <param name="outputLen">Output parameter for buffer size.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_render_image_data(
            NativeHandle imageHandle,
            int format,
            out IntPtr outputPtr,
            out int outputLen,
            out int errorCode);

        /// <summary>
        /// Saves a rendered image to file.
        /// </summary>
        /// <param name="imageHandle">The image handle.</param>
        /// <param name="path">Output file path.</param>
        /// <param name="format">Output format (0=PNG, 1=JPEG, 2=BMP, 3=TIFF).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_render_image_save(
            NativeHandle imageHandle,
            string path,
            int format,
            out int errorCode);

        /// <summary>
        /// Frees a renderer handle.
        /// </summary>
        /// <param name="handle">The renderer handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_renderer_free(IntPtr handle);

        /// <summary>
        /// Frees a rendered image handle.
        /// </summary>
        /// <param name="handle">The image handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_render_image_free(IntPtr handle);

        #endregion

        #region OCR API
        // v0.3.27: replaced hallucinated P/Invoke declarations with the real
        // FFI bridge added to src/ffi.rs. These 4 functions match the Rust
        // OcrEngine API and the Go/Node.js bindings. When the crate is built
        // without --features ocr (the default), each function returns
        // ERR_UNSUPPORTED (6).

        /// <summary>
        /// Creates an OCR engine from ONNX model file paths.
        /// Returns ERR_UNSUPPORTED when the native library was built without OCR support.
        /// </summary>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_ocr_engine_create(
            string detModelPath,
            string recModelPath,
            string dictPath,
            out int errorCode);

        /// <summary>
        /// Frees an OCR engine handle created by pdf_ocr_engine_create.
        /// </summary>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_ocr_engine_free(IntPtr engine);

        /// <summary>
        /// Checks whether a page would benefit from OCR (scanned image, no text layer).
        /// </summary>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_ocr_page_needs_ocr(
            NativeHandle documentHandle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Runs OCR on a page and returns the recognized text as a UTF-8 C string.
        /// The engine parameter may be IntPtr.Zero, in which case the function returns an error.
        /// Caller must free the returned string with FreeString().
        /// </summary>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_ocr_extract_text(
            NativeHandle documentHandle,
            int pageIndex,
            IntPtr engine,
            out int errorCode);

        #endregion

        #region Compliance API

        /// <summary>
        /// Validates document against PDF/A standard.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="level">PDF/A level (0=1b, 1=1a, 2=2b, 3=2a, 4=2u, 5=3b, 6=3a, 7=3u).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Validation result handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_validate_pdf_a(
            NativeHandle handle,
            int level,
            out int errorCode);

        /// <summary>
        /// Validates document against PDF/X standard.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="level">PDF/X level (0=1a, 1=3, 2=4).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Validation result handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_validate_pdf_x(
            NativeHandle handle,
            int level,
            out int errorCode);

        /// <summary>
        /// Validates document against PDF/UA standard.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="level">PDF/UA level (0=1, 1=2).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Validation result handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_validate_pdf_ua(
            NativeHandle handle,
            int level,
            out int errorCode);

        /// <summary>
        /// Checks if validation result is valid (compliant).
        /// </summary>
        /// <param name="resultHandle">The validation result handle.</param>
        /// <returns>True if compliant, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_validation_result_is_valid(NativeHandle resultHandle);

        /// <summary>
        /// Gets the number of issues from validation result.
        /// </summary>
        /// <param name="resultHandle">The validation result handle.</param>
        /// <returns>Number of issues found.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_validation_result_issue_count(NativeHandle resultHandle);

        /// <summary>
        /// Gets issue at specified index from validation result.
        /// </summary>
        /// <param name="resultHandle">The validation result handle.</param>
        /// <param name="index">Issue index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Issue handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_validation_result_get_issue(
            NativeHandle resultHandle,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the severity of a compliance issue.
        /// </summary>
        /// <param name="issueHandle">The issue handle.</param>
        /// <returns>Severity level (0=Error, 1=Warning, 2=Info).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_compliance_issue_get_severity(NativeHandle issueHandle);

        /// <summary>
        /// Gets the message of a compliance issue.
        /// </summary>
        /// <param name="issueHandle">The issue handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_compliance_issue_get_message(
            NativeHandle issueHandle,
            out int errorCode);

        /// <summary>
        /// Gets the rule identifier of a compliance issue.
        /// </summary>
        /// <param name="issueHandle">The issue handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_compliance_issue_get_rule(
            NativeHandle issueHandle,
            out int errorCode);

        /// <summary>
        /// Gets the page number of a compliance issue.
        /// </summary>
        /// <param name="issueHandle">The issue handle.</param>
        /// <returns>Page index (0-based), or -1 for document-level issues.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_compliance_issue_get_page(NativeHandle issueHandle);

        /// <summary>
        /// Converts document to PDF/A compliance.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="level">Target PDF/A level.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_convert_to_pdf_a(
            NativeHandle handle,
            int level,
            out int errorCode);

        /// <summary>
        /// Auto-detects document's compliance level.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Compliance level, or -1 if none detected.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_compliance_level(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Frees a validation result handle.
        /// </summary>
        /// <param name="handle">The result handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_validation_result_free(IntPtr handle);

        /// <summary>
        /// Frees a compliance issue handle.
        /// </summary>
        /// <param name="handle">The issue handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_compliance_issue_free(IntPtr handle);

        #endregion

        #region Digital Signature API

        /// <summary>
        /// Gets the number of signatures in document.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Number of signatures.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_signature_count(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Gets signature at specified index.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="index">Signature index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Signature handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_get_signature(
            NativeHandle handle,
            int index,
            out int errorCode);

        /// <summary>
        /// Verifies a single signature.
        /// </summary>
        /// <param name="signatureHandle">The signature handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Verification status (0=Valid, 1=Invalid, 2=Unknown, 3=NotVerified).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_verify_signature(
            NativeHandle signatureHandle,
            out int errorCode);

        /// <summary>
        /// Gets signer name from signature.
        /// </summary>
        /// <param name="signatureHandle">The signature handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_signature_get_signer_name(
            NativeHandle signatureHandle,
            out int errorCode);

        /// <summary>
        /// Gets signing time as Unix timestamp.
        /// </summary>
        /// <param name="signatureHandle">The signature handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Unix timestamp of signing time.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_signature_get_signing_time(
            NativeHandle signatureHandle,
            out int errorCode);

        /// <summary>
        /// Gets reason for signing.
        /// </summary>
        /// <param name="signatureHandle">The signature handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_signature_get_reason(
            NativeHandle signatureHandle,
            out int errorCode);

        /// <summary>
        /// Gets signing location.
        /// </summary>
        /// <param name="signatureHandle">The signature handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_signature_get_location(
            NativeHandle signatureHandle,
            out int errorCode);

        /// <summary>
        /// Gets certificate from signature.
        /// </summary>
        /// <param name="signatureHandle">The signature handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Certificate handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_signature_get_certificate(
            NativeHandle signatureHandle,
            out int errorCode);

        /// <summary>
        /// Gets certificate subject.
        /// </summary>
        /// <param name="certHandle">The certificate handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_certificate_get_subject(
            NativeHandle certHandle,
            out int errorCode);

        /// <summary>
        /// Gets certificate issuer.
        /// </summary>
        /// <param name="certHandle">The certificate handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_certificate_get_issuer(
            NativeHandle certHandle,
            out int errorCode);

        /// <summary>
        /// Gets certificate serial number.
        /// </summary>
        /// <param name="certHandle">The certificate handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_certificate_get_serial(
            NativeHandle certHandle,
            out int errorCode);

        /// <summary>
        /// Gets certificate validity dates.
        /// </summary>
        /// <param name="certHandle">The certificate handle.</param>
        /// <param name="notBefore">Output parameter for not_before timestamp.</param>
        /// <param name="notAfter">Output parameter for not_after timestamp.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_certificate_get_validity(
            NativeHandle certHandle,
            out long notBefore,
            out long notAfter,
            out int errorCode);

        /// <summary>
        /// Checks if certificate is currently valid.
        /// </summary>
        /// <param name="certHandle">The certificate handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if valid, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_certificate_is_valid(
            NativeHandle certHandle,
            out int errorCode);

        /// <summary>
        /// Signs a document.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pfxPath">Path to PFX/P12 certificate file.</param>
        /// <param name="password">Certificate password.</param>
        /// <param name="reason">Optional signing reason (can be null).</param>
        /// <param name="location">Optional signing location (can be null).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_sign_document(
            NativeHandle handle,
            string pfxPath,
            string password,
            string reason,
            string location,
            out int errorCode);

        /// <summary>
        /// Verifies all signatures in document.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if all signatures are valid, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_verify_all_signatures(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Frees a signature handle.
        /// </summary>
        /// <param name="handle">The signature handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_signature_free(IntPtr handle);

        /// <summary>
        /// Frees a certificate handle.
        /// </summary>
        /// <param name="handle">The certificate handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_certificate_free(IntPtr handle);

        #endregion

        #region Barcode API

        /// <summary>
        /// Detects barcodes on a page.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Detection results handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_detect_barcodes_on_page(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Gets number of detected barcodes.
        /// </summary>
        /// <param name="resultsHandle">The detection results handle.</param>
        /// <returns>Number of barcodes found.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_barcode_results_count(NativeHandle resultsHandle);

        /// <summary>
        /// Gets barcode at specified index.
        /// </summary>
        /// <param name="resultsHandle">The detection results handle.</param>
        /// <param name="index">Barcode index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Barcode handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_barcode_results_get(
            NativeHandle resultsHandle,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets barcode format type.
        /// </summary>
        /// <param name="barcodeHandle">The barcode handle.</param>
        /// <returns>Format type (0=QR, 1=DataMatrix, 2=PDF417, etc.).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_barcode_get_format(NativeHandle barcodeHandle);

        /// <summary>
        /// Gets decoded barcode data.
        /// </summary>
        /// <param name="barcodeHandle">The barcode handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_barcode_get_data(
            NativeHandle barcodeHandle,
            out int errorCode);

        /// <summary>
        /// Gets barcode bounding box.
        /// </summary>
        /// <param name="barcodeHandle">The barcode handle.</param>
        /// <param name="x">Output parameter for x coordinate.</param>
        /// <param name="y">Output parameter for y coordinate.</param>
        /// <param name="width">Output parameter for width.</param>
        /// <param name="height">Output parameter for height.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_barcode_get_bounds(
            NativeHandle barcodeHandle,
            out float x,
            out float y,
            out float width,
            out float height,
            out int errorCode);

        /// <summary>
        /// Gets detection confidence score.
        /// </summary>
        /// <param name="barcodeHandle">The barcode handle.</param>
        /// <returns>Confidence score (0.0-1.0).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float pdf_barcode_get_confidence(NativeHandle barcodeHandle);

        /// <summary>
        /// Generates a QR code.
        /// </summary>
        /// <param name="data">Data to encode.</param>
        /// <param name="errorCorrection">Error correction level (0=Low, 1=Medium, 2=Quartile, 3=High).</param>
        /// <param name="size">Image size in pixels.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Barcode image handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_generate_qr_code(
            string data,
            int errorCorrection,
            int size,
            out int errorCode);

        /// <summary>
        /// Generates a barcode.
        /// </summary>
        /// <param name="data">Data to encode.</param>
        /// <param name="format">Barcode format (0=QR, 1=DataMatrix, etc.).</param>
        /// <param name="width">Image width in pixels.</param>
        /// <param name="height">Image height in pixels.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Barcode image handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_generate_barcode(
            string data,
            int format,
            int width,
            int height,
            out int errorCode);

        /// <summary>
        /// Gets barcode image data as bytes.
        /// </summary>
        /// <param name="imageHandle">The barcode image handle.</param>
        /// <param name="outputLen">Output parameter for buffer size.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Pointer to image data (PNG format).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_barcode_image_data(
            NativeHandle imageHandle,
            out int outputLen,
            out int errorCode);

        /// <summary>
        /// Gets barcode image dimensions.
        /// </summary>
        /// <param name="imageHandle">The barcode image handle.</param>
        /// <param name="width">Output parameter for width.</param>
        /// <param name="height">Output parameter for height.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_barcode_image_dimensions(
            NativeHandle imageHandle,
            out int width,
            out int height,
            out int errorCode);

        /// <summary>
        /// Saves barcode image to file.
        /// </summary>
        /// <param name="imageHandle">The barcode image handle.</param>
        /// <param name="path">Output file path.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_barcode_image_save(
            NativeHandle imageHandle,
            string path,
            out int errorCode);

        /// <summary>
        /// Frees barcode detection results.
        /// </summary>
        /// <param name="handle">The results handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_barcode_results_free(IntPtr handle);

        /// <summary>
        /// Frees a barcode handle.
        /// </summary>
        /// <param name="handle">The barcode handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_barcode_free(IntPtr handle);

        // Clean P/Invoke set for the public Core/Barcode.cs wrapper. Uses
        // IntPtr handles (no SafeHandle) and EntryPoint overrides so these
        // coexist with the existing barcode entries above that have
        // divergent legacy signatures.
        [LibraryImport(LibName, EntryPoint = "pdf_generate_barcode", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfGenerateBarcode(string data, int format, int sizePx, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_barcode_get_format", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfBarcodeGetFormat(IntPtr handle, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_barcode_get_data", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfBarcodeGetData(IntPtr handle, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_barcode_get_confidence", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float PdfBarcodeGetConfidence(IntPtr handle, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_barcode_get_image_png", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfBarcodeGetImagePng(IntPtr handle, int sizePx, out int outLen, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_barcode_get_svg", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfBarcodeGetSvg(IntPtr handle, int sizePx, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_generate_qr_code", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfGenerateQrCode(string data, int errorCorrection, int sizePx, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_barcode_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfBarcodeFree(IntPtr handle);

        // PdfValidator wrappers (Pascal-case, consistent IntPtr handles).
        // EntryPoint overrides let these coexist with the divergent legacy
        // declarations earlier in the file.
        [LibraryImport(LibName, EntryPoint = "pdf_validate_pdf_a_level", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfValidatePdfALevel(IntPtr document, int level, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_a_is_compliant", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool PdfPdfAIsCompliant(IntPtr results, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_a_error_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPdfAErrorCount(IntPtr results);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_a_warning_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPdfAWarningCount(IntPtr results);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_a_get_error", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfPdfAGetError(IntPtr results, int index, out int errorCode);

        // NOTE: Rust core exposes pdf_pdf_a_warning_count but no
        // pdf_pdf_a_get_warning today — PDF/A messages only surface as
        // errors. We read the count for parity and leave the list empty.

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_a_results_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfPdfAResultsFree(IntPtr results);

        [LibraryImport(LibName, EntryPoint = "pdf_validate_pdf_x_level", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfValidatePdfXLevel(IntPtr document, int level, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_x_is_compliant", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool PdfPdfXIsCompliant(IntPtr results, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_x_error_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPdfXErrorCount(IntPtr results);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_x_get_error", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfPdfXGetError(IntPtr results, int index, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_x_results_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfPdfXResultsFree(IntPtr results);

        [LibraryImport(LibName, EntryPoint = "pdf_validate_pdf_ua", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfValidatePdfUa(IntPtr document, int level, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_ua_is_accessible", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool PdfPdfUaIsAccessible(IntPtr results, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_ua_error_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPdfUaErrorCount(IntPtr results);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_ua_warning_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPdfUaWarningCount(IntPtr results);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_ua_get_error", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfPdfUaGetError(IntPtr results, int index, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_ua_get_warning", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfPdfUaGetWarning(IntPtr results, int index, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_pdf_ua_results_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfPdfUaResultsFree(IntPtr results);

        // OCR wrappers that take an IntPtr document handle (Core/OcrEngine.cs
        // uses IntPtr throughout). Parallel entries above use NativeHandle
        // and can't be called from the IntPtr code path directly.
        [LibraryImport(LibName, EntryPoint = "pdf_ocr_page_needs_ocr", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool OcrPageNeedsOcrByPtr(IntPtr documentHandle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_ocr_extract_text", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr OcrExtractTextByPtr(IntPtr documentHandle, int pageIndex, IntPtr engine, out int errorCode);

        /// <summary>
        /// Frees a barcode image handle.
        /// </summary>
        /// <param name="handle">The image handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_barcode_image_free(IntPtr handle);

        #endregion

        #region XFA API

        /// <summary>
        /// Checks if document contains XFA forms.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if XFA forms present, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_has_xfa(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Gets the XFA data packet.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="packetName">Name of the XFA packet (e.g., "template", "data", "config").</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated XML string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_xfa_get_packet(
            NativeHandle handle,
            string packetName,
            out int errorCode);

        /// <summary>
        /// Gets the XFA form type.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>XFA type (0=None, 1=Static, 2=Dynamic, 3=Hybrid).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_xfa_get_type(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Gets the number of XFA fields.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Number of XFA fields.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_xfa_get_field_count(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Gets an XFA field by index.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="index">Field index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Field handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_xfa_get_field(
            NativeHandle handle,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the name of an XFA field.
        /// </summary>
        /// <param name="fieldHandle">The field handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_xfa_field_get_name(
            NativeHandle fieldHandle,
            out int errorCode);

        /// <summary>
        /// Gets the value of an XFA field.
        /// </summary>
        /// <param name="fieldHandle">The field handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_xfa_field_get_value(
            NativeHandle fieldHandle,
            out int errorCode);

        /// <summary>
        /// Sets the value of an XFA field.
        /// </summary>
        /// <param name="fieldHandle">The field handle.</param>
        /// <param name="value">The new value.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_xfa_field_set_value(
            NativeHandle fieldHandle,
            string value,
            out int errorCode);

        /// <summary>
        /// Frees an XFA field handle.
        /// </summary>
        /// <param name="handle">The field handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_xfa_field_free(IntPtr handle);

        #endregion

        #region Hybrid ML API

        /// <summary>
        /// Creates a hybrid ML analyzer.
        /// </summary>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Analyzer handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_hybrid_ml_analyzer_create(out int errorCode);

        /// <summary>
        /// Analyzes a page using hybrid ML.
        /// </summary>
        /// <param name="analyzerHandle">The analyzer handle.</param>
        /// <param name="documentHandle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Analysis result handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_hybrid_ml_analyze_page(
            NativeHandle analyzerHandle,
            NativeHandle documentHandle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Gets the number of detected regions from analysis result.
        /// </summary>
        /// <param name="resultHandle">The analysis result handle.</param>
        /// <returns>Number of regions detected.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_hybrid_ml_result_region_count(NativeHandle resultHandle);

        /// <summary>
        /// Gets a region from analysis result.
        /// </summary>
        /// <param name="resultHandle">The analysis result handle.</param>
        /// <param name="index">Region index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Region handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_hybrid_ml_result_get_region(
            NativeHandle resultHandle,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the type of a detected region.
        /// </summary>
        /// <param name="regionHandle">The region handle.</param>
        /// <returns>Region type (0=Text, 1=Image, 2=Table, 3=Figure, etc.).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_hybrid_ml_region_get_type(NativeHandle regionHandle);

        /// <summary>
        /// Gets the bounding box of a detected region.
        /// </summary>
        /// <param name="regionHandle">The region handle.</param>
        /// <param name="x">Output parameter for x coordinate.</param>
        /// <param name="y">Output parameter for y coordinate.</param>
        /// <param name="width">Output parameter for width.</param>
        /// <param name="height">Output parameter for height.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_hybrid_ml_region_get_bounds(
            NativeHandle regionHandle,
            out float x,
            out float y,
            out float width,
            out float height);

        /// <summary>
        /// Gets the confidence of a detected region.
        /// </summary>
        /// <param name="regionHandle">The region handle.</param>
        /// <returns>Confidence score (0.0-1.0).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float pdf_hybrid_ml_region_get_confidence(NativeHandle regionHandle);

        /// <summary>
        /// Gets the extracted text from a region.
        /// </summary>
        /// <param name="regionHandle">The region handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_hybrid_ml_region_get_text(
            NativeHandle regionHandle,
            out int errorCode);

        /// <summary>
        /// Frees a hybrid ML analyzer handle.
        /// </summary>
        /// <param name="handle">The analyzer handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_hybrid_ml_analyzer_free(IntPtr handle);

        /// <summary>
        /// Frees a hybrid ML result handle.
        /// </summary>
        /// <param name="handle">The result handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_hybrid_ml_result_free(IntPtr handle);

        /// <summary>
        /// Frees a hybrid ML region handle.
        /// </summary>
        /// <param name="handle">The region handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_hybrid_ml_region_free(IntPtr handle);

        #endregion

        #region Layer API

        /// <summary>
        /// Gets the number of layers (OCGs) in document.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Number of layers.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_layer_count(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets layer name at specified index.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="index">Layer index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_layer_name(
            IntPtr handle,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if layer is visible.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="index">Layer index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if visible, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_layer_visible(
            IntPtr handle,
            int index,
            out int errorCode);

        /// <summary>
        /// Sets layer visibility.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="index">Layer index (0-based).</param>
        /// <param name="visible">Whether the layer should be visible.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_set_layer_visibility(
            IntPtr handle,
            int index,
            [MarshalAs(UnmanagedType.I1)] bool visible,
            out int errorCode);

        /// <summary>
        /// Checks if document has layers.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if document has layers, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_has_layers(IntPtr handle);

        #endregion

        #region Metadata API

        /// <summary>
        /// Gets document title.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer, or null if not set. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_title(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets document author.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer, or null if not set. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_author(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets document subject.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer, or null if not set. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_subject(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets document keywords.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer, or null if not set. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_keywords(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets document creator application.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer, or null if not set. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_creator(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets document producer application.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer, or null if not set. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_producer(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets document creation date as Unix timestamp.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Unix timestamp, or 0 if not set.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_get_creation_date(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets document modification date as Unix timestamp.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Unix timestamp, or 0 if not set.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_get_modification_date(
            IntPtr handle,
            out int errorCode);

        #endregion

        #region Outline API

        /// <summary>
        /// Gets the number of outlines (bookmarks) in document.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Number of outlines.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_outline_count(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Gets outline at specified index.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="index">Outline index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Outline handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_outline(
            IntPtr handle,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets outline title.
        /// </summary>
        /// <param name="outlineHandle">The outline handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_outline_get_title(
            IntPtr outlineHandle,
            out int errorCode);

        /// <summary>
        /// Gets outline target page.
        /// </summary>
        /// <param name="outlineHandle">The outline handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Page index (0-based), or -1 if not a page link.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_outline_get_page(
            IntPtr outlineHandle,
            out int errorCode);

        /// <summary>
        /// Gets the number of children for an outline.
        /// </summary>
        /// <param name="outlineHandle">The outline handle.</param>
        /// <returns>Number of child outlines.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_outline_get_child_count(IntPtr outlineHandle);

        /// <summary>
        /// Gets child outline at specified index.
        /// </summary>
        /// <param name="outlineHandle">The parent outline handle.</param>
        /// <param name="index">Child index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Outline handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_outline_get_child(
            IntPtr outlineHandle,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if document has outlines.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if document has outlines, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_has_outlines(IntPtr handle);

        /// <summary>
        /// Frees an outline handle.
        /// </summary>
        /// <param name="handle">The outline handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_outline_free(IntPtr handle);

        #endregion

        #region Security API

        /// <summary>
        /// Checks if document is encrypted.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if encrypted, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_encrypted(IntPtr handle);

        /// <summary>
        /// Gets the encryption level.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer (e.g., "RC4 40-bit", "AES 256-bit"). Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_encryption_level(
            IntPtr handle,
            out int errorCode);

        /// <summary>
        /// Checks if document requires password.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if password required, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_requires_password(IntPtr handle);

        /// <summary>
        /// Checks if document allows printing.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if printing allowed, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_can_print(IntPtr handle);

        /// <summary>
        /// Checks if document allows copying.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if copying allowed, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_can_copy(IntPtr handle);

        /// <summary>
        /// Checks if document allows modification.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if modification allowed, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_can_modify(IntPtr handle);

        /// <summary>
        /// Checks if document allows form filling.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if form filling allowed, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_can_fill_forms(IntPtr handle);

        /// <summary>
        /// Checks if document allows annotation.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if annotation allowed, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_can_annotate(IntPtr handle);

        /// <summary>
        /// Unlocks a password-protected document.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="password">The password.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if unlocked successfully, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_unlock(
            NativeHandle handle,
            string password,
            out int errorCode);

        #endregion

        #region Form Field API

        /// <summary>
        /// Gets the number of form fields in document.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Number of form fields.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_form_field_count(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Gets form field at specified index.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="index">Field index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Field handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_get_form_field(
            NativeHandle handle,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets form field by name.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="name">The field name.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Field handle, or null if not found.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_get_form_field_by_name(
            NativeHandle handle,
            string name,
            out int errorCode);

        /// <summary>
        /// Gets form field name.
        /// </summary>
        /// <param name="fieldHandle">The field handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_form_field_get_name(
            NativeHandle fieldHandle,
            out int errorCode);

        /// <summary>
        /// Gets form field type.
        /// </summary>
        /// <param name="fieldHandle">The field handle.</param>
        /// <returns>Field type (0=Text, 1=Checkbox, 2=Radio, 3=Combo, 4=List, 5=Button, 6=Signature).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_form_field_get_type(NativeHandle fieldHandle);

        /// <summary>
        /// Gets form field value.
        /// </summary>
        /// <param name="fieldHandle">The field handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_form_field_get_value(
            NativeHandle fieldHandle,
            out int errorCode);

        /// <summary>
        /// Sets form field value.
        /// </summary>
        /// <param name="fieldHandle">The field handle.</param>
        /// <param name="value">The new value.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_form_field_set_value(
            NativeHandle fieldHandle,
            string value,
            out int errorCode);

        /// <summary>
        /// Checks if form field is read-only.
        /// </summary>
        /// <param name="fieldHandle">The field handle.</param>
        /// <returns>True if read-only, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_form_field_is_readonly(NativeHandle fieldHandle);

        /// <summary>
        /// Checks if form field is required.
        /// </summary>
        /// <param name="fieldHandle">The field handle.</param>
        /// <returns>True if required, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_form_field_is_required(NativeHandle fieldHandle);

        /// <summary>
        /// Checks if document has form fields.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <returns>True if document has form fields, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_has_form_fields(NativeHandle handle);

        /// <summary>
        /// Frees a form field handle.
        /// </summary>
        /// <param name="handle">The field handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_form_field_free(IntPtr handle);

        #endregion

        #region Thumbnail API

        /// <summary>
        /// Generates a thumbnail for a page.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="width">Maximum thumbnail width.</param>
        /// <param name="height">Maximum thumbnail height.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Thumbnail image handle, or null on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_generate_thumbnail(
            NativeHandle handle,
            int pageIndex,
            int width,
            int height,
            out int errorCode);

        /// <summary>
        /// Gets thumbnail image data as bytes.
        /// </summary>
        /// <param name="thumbnailHandle">The thumbnail handle.</param>
        /// <param name="format">Output format (0=PNG, 1=JPEG).</param>
        /// <param name="outputPtr">Output parameter for byte buffer pointer.</param>
        /// <param name="outputLen">Output parameter for buffer size.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_thumbnail_get_data(
            NativeHandle thumbnailHandle,
            int format,
            out IntPtr outputPtr,
            out int outputLen,
            out int errorCode);

        /// <summary>
        /// Gets thumbnail dimensions.
        /// </summary>
        /// <param name="thumbnailHandle">The thumbnail handle.</param>
        /// <param name="width">Output parameter for width.</param>
        /// <param name="height">Output parameter for height.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_thumbnail_get_dimensions(
            NativeHandle thumbnailHandle,
            out int width,
            out int height);

        /// <summary>
        /// Saves thumbnail to file.
        /// </summary>
        /// <param name="thumbnailHandle">The thumbnail handle.</param>
        /// <param name="path">Output file path.</param>
        /// <param name="format">Output format (0=PNG, 1=JPEG).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_thumbnail_save(
            NativeHandle thumbnailHandle,
            string path,
            int format,
            out int errorCode);

        /// <summary>
        /// Frees a thumbnail handle.
        /// </summary>
        /// <param name="handle">The thumbnail handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_thumbnail_free(IntPtr handle);

        #endregion

        #region Content Analysis API

        /// <summary>
        /// Checks if a page has any content.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if page has content, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_page_has_content(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Checks if a page is blank.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if page is blank, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_page_is_blank(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Gets page content size in bytes.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Content size in bytes.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_page_get_content_size(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Gets page complexity score.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Complexity score (higher = more complex).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_page_get_complexity_score(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Gets the width of a page in PDF points.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_get_width", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float pdf_page_get_width(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Gets the height of a page in PDF points.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_get_height", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float pdf_page_get_height(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Checks if page likely has forms.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if page likely has forms, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_page_likely_has_forms(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Checks if page likely has tables.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if page likely has tables, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_page_likely_has_tables(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Checks if page likely has images.
        /// </summary>
        /// <param name="handle">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True if page likely has images, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_page_likely_has_images(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        #endregion

        // Phase 2 declarations moved to existing regions above (Barcode API, Digital Signature API, Rendering API)

        #region Document Security & Permissions (15 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_can_copy(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_can_print(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_can_modify(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_can_annotate(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_can_fill_forms(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_can_extract_text(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_can_assemble(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_can_print_high_quality(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_is_encrypted(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_get_security_revision(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_all_permissions(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_owner_password_status(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_user_password_status(NativeHandle handle, out int errorCode);

        #endregion

        #region Document Metadata (18 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_document_get_creation_date(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_document_get_modification_date(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_producer(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_creator(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_all_metadata(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_has_metadata_stream(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_metadata_xml(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_is_linearized(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_document_get_file_size(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_embedded_font_names(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_get_embedded_file_count(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_author(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_title(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_subject(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_keywords(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_pdf_version(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_encryption_algorithm(NativeHandle handle, out int errorCode);

        #endregion

        #region Element Finding & Discovery (17 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_element_count(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_element_content(NativeHandle handle, int pageIndex, int elementIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_element_type(NativeHandle handle, int pageIndex, int elementIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_has_structured_content(NativeHandle handle, int pageIndex);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_text_element_count(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_find_elements_by_type(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string elementType, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_page_elements_json(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_find_elements_in_rect(NativeHandle handle, int pageIndex, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_element_bounds(NativeHandle handle, int pageIndex, int elementIndex, out int errorCode);

        #endregion

        #region Page Operations (9 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_delete_page(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_move_page(NativeHandle handle, int fromIndex, int toIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_copy_page(NativeHandle handle, int pageIndex, int insertAfter, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_rotate_page(NativeHandle handle, int pageIndex, int degrees, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_page_blank(NativeHandle handle, int pageIndex);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_page_rotation(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_page_mediabox(NativeHandle handle, int pageIndex, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_page_cropbox(NativeHandle handle, int pageIndex, float x, float y, float width, float height, out int errorCode);

        #endregion

        #region Form Field Operations (29 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_create_text_field(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_create_checkbox(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_create_radio_button(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, IntPtr options, int optionCount, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_create_listbox(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, IntPtr items, int itemCount, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_create_combobox(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, IntPtr items, int itemCount, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_create_signature_field(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_field_readonly(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, [MarshalAs(UnmanagedType.I1)] bool readOnly, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_field_required(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, [MarshalAs(UnmanagedType.I1)] bool required, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_field_value(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, [MarshalAs(UnmanagedType.LPUTF8Str)] string value, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_field_default_value(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, [MarshalAs(UnmanagedType.LPUTF8Str)] string value, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_field_max_length(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, int maxLength, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_field_visibility(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, int visibility, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_field_type(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_field_value(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_field_required(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_field_readonly(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_field_max_length(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_field_options(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_field_bounds(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_validate_fields(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_field_errors(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_reset_field(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_reset_all_fields(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_export_field_data(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_import_field_data(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string data, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_flatten_form(NativeHandle handle, out int errorCode);

        // pdf_has_form_fields is defined in Form Field API region above - removed duplicate

        #endregion

        #region Form Export/Import (11 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_export_form_xml(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_export_form_json(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_export_form_csv(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_export_form_xfdf(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_import_form_json(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string jsonData, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_import_form_xml(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string xmlData, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_import_form_csv(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string csvData, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_import_form_xfdf(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string xfdfData, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_export_form_bytes(NativeHandle handle, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_import_form_bytes(NativeHandle handle, [In] byte[] data, int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_form_version(NativeHandle handle, out int errorCode);

        #endregion

        #region Search Operations (13 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_search_all(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string query, [MarshalAs(UnmanagedType.I1)] bool caseSensitive, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_search_page(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string query, [MarshalAs(UnmanagedType.I1)] bool caseSensitive, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_count_occurrences(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, [MarshalAs(UnmanagedType.I1)] bool caseSensitive, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_first_occurrence(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_last_occurrence(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_search_regex(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string pattern, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_count_regex_matches(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string pattern, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_search_regex_page(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string pattern, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_search_whole_word(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string word, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_search_with_context(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, int contextChars, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_search_term_frequency(NativeHandle handle, IntPtr terms, int termCount, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_search_near_point(NativeHandle handle, int pageIndex, double x, double y, double radius, out int errorCode);

        #endregion

        #region Annotation Operations (31 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_text_annotation(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string content, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_highlight(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string color, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_underline(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string color, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_strikeout(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string color, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_square_annotation(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string borderColor, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_circle_annotation(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string borderColor, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_line_annotation(NativeHandle handle, int pageIndex, double x1, double y1, double x2, double y2, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_freetext_annotation(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, [MarshalAs(UnmanagedType.LPUTF8Str)] string fontName, double fontSize, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_link_annotation(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string url, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_file_annotation(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string filePath, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_annotation_count_on_page(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_annotation(NativeHandle handle, int pageIndex, int annotationIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_page_annotations(NativeHandle handle, int pageIndex, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_annotations_by_type(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string type, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_annotation_content(NativeHandle handle, int pageIndex, int annotationIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_modify_annotation_content(NativeHandle handle, int pageIndex, int annotationIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string newContent, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_modify_annotation_color(NativeHandle handle, int pageIndex, int annotationIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string color, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_modify_annotation_opacity(NativeHandle handle, int pageIndex, int annotationIndex, double opacity, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_annotation_author(NativeHandle handle, int pageIndex, int annotationIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string author, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_annotation_date(NativeHandle handle, int pageIndex, int annotationIndex, long dateTime, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_annotation_flags(NativeHandle handle, int pageIndex, int annotationIndex, int flags, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_delete_annotation(NativeHandle handle, int pageIndex, int annotationIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_delete_all_page_annotations(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_delete_annotations_by_type(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string type, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_flatten_annotations(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_total_annotation_count(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_export_annotations(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_import_annotations(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, [MarshalAs(UnmanagedType.LPUTF8Str)] string data, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_export_annotations_bytes(NativeHandle handle, out int dataLen, out int errorCode);

        #endregion

        #region Barcode Operations (14 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_generate_barcode(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string data, int format, int width, int height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_barcode_to_page(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string data, int format, double x, double y, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_generate_qrcode(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string data, int size, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_generate_code128(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string data, int width, int height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_generate_code39(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string data, int width, int height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_generate_ean13(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string data, int width, int height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_detect_barcodes(NativeHandle handle, int pageIndex, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_read_barcode(NativeHandle handle, int pageIndex, int barcodeIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_count_barcodes(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_detect_all_barcodes(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_barcode_format(NativeHandle handle, int pageIndex, int barcodeIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_barcode_location(NativeHandle handle, int pageIndex, int barcodeIndex, out int errorCode);

        #endregion

        #region Rendering Operations (15 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_to_png(NativeHandle handle, int pageIndex, int dpi, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_to_jpeg(NativeHandle handle, int pageIndex, int dpi, int quality, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_to_pdf(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_render_colorspace(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string colorspace, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_render_scale(NativeHandle handle, double scale, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_render_transparency(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool enabled, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_page_range(NativeHandle handle, int startIndex, int endIndex, int format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_render_to_files(NativeHandle handle, int startIndex, int endIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string outputDir, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_page_dimensions(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_mediabox(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_cropbox(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_bleedbox(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_trimbox(NativeHandle handle, int pageIndex, out int errorCode);

        #endregion

        #region XFA Operations (10 functions)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_has_xfa(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_xfa_form_type(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_xfa_field_count(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_parse_xfa_form(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_xfa_fields(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_xfa_field_value(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_xfa_field_value(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, [MarshalAs(UnmanagedType.LPUTF8Str)] string value, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_xfa_dataset(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_export_xfa_dataset_xml(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_import_xfa_dataset_xml(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string xmlData, out int errorCode);

        #endregion

        #region Digital Signature Operations (13 functions)

        // pdf_sign_document is defined in Digital Signature API region above - removed duplicate

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_signature_field(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_sign_field(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, [MarshalAs(UnmanagedType.LPUTF8Str)] string certificatePath, [MarshalAs(UnmanagedType.LPUTF8Str)] string password, int algorithm, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_verify_signature(NativeHandle handle, int signatureIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_verify_all_signatures(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_signature_info(NativeHandle handle, int signatureIndex, out int errorCode);

        // pdf_get_signature_count is defined in Digital Signature API region above - removed duplicate

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_all_signatures(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_get_signature_date(NativeHandle handle, int signatureIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_signer_name(NativeHandle handle, int signatureIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_cert_expired(NativeHandle handle, int signatureIndex);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_remove_signature(NativeHandle handle, int signatureIndex, out int errorCode);

        #endregion

        #region Additional Manager Support Functions

        // Cache Operations
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_cache_get_size(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_get_entry_count(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_cache_get_statistics(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_cache_get_hit_count(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_cache_get_miss_count(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial double pdf_cache_get_hit_rate(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_cache_get_eviction_count(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_clear(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_clear_rendering(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_clear_fonts(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_clear_images(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_clear_text(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_clear_ocr(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_clear_all(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_set_max_size(long maxBytes, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_cache_get_max_size(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_set_enabled([MarshalAs(UnmanagedType.I1)] bool enabled, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_cache_is_enabled(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_cache_get_rendering_size(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_cache_get_font_size(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_cache_get_image_size(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_cache_get_text_size(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_trim(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_trim_to_size(long targetBytes, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_evict_older_than(int seconds, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_cache_get_document_size(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_clear_document(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_preload_pages(NativeHandle handle, int startPage, int endPage, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial double pdf_cache_get_warmth(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_optimize(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_cache_compact(out int errorCode);

        // Element Analysis Operations
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_find_elements(NativeHandle handle, int pageIndex, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_find_elements_by_type(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string type, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_find_elements_in_region(NativeHandle handle, int pageIndex, float x, float y, float width, float height, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_element_at_position(NativeHandle handle, int pageIndex, float x, float y, out int errorCode);

        // pdf_get_element_count is defined earlier - removed duplicate

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_element_count_by_type(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string type, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_find_tables(NativeHandle handle, int pageIndex, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_find_images(NativeHandle handle, int pageIndex, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_find_text_blocks(NativeHandle handle, int pageIndex, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_element_bounds(NativeHandle handle, int pageIndex, int elementIndex, ref float x, ref float y, ref float width, ref float height, out int errorCode);

        // pdf_get_element_type and pdf_get_element_content are defined earlier - removed duplicates

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_element_properties(NativeHandle handle, int pageIndex, int elementIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_table_cell_content(NativeHandle handle, int pageIndex, int tableIndex, int row, int column, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_table_dimensions(NativeHandle handle, int pageIndex, int tableIndex, ref int rows, ref int columns, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_extract_image_data(NativeHandle handle, int pageIndex, int imageIndex, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_image_format(NativeHandle handle, int pageIndex, int imageIndex, out int errorCode);

        // Format Conversion Operations
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_convert_to(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string outputPath, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, int quality, int dpi, [MarshalAs(UnmanagedType.I1)] bool embedFonts, [MarshalAs(UnmanagedType.I1)] bool compressImages, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_convert_to_bytes(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, int quality, int dpi, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_convert_to_pdfa(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string outputPath, [MarshalAs(UnmanagedType.LPUTF8Str)] string conformanceLevel, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_convert_to_pdfx(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string outputPath, [MarshalAs(UnmanagedType.LPUTF8Str)] string conformanceLevel, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_page_to_image(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, int dpi, int quality, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_convert_page_to_html(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_convert_to_html(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_convert_page_to_markdown(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_convert_to_markdown(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_convert_page_to_text(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_convert_to_text(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_convert_to_xml(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_convert_to_json(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_supported_formats(out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_format_supported([MarshalAs(UnmanagedType.LPUTF8Str)] string format);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_estimate_output_size(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, int quality, int dpi, out int errorCode);

        // Advanced Metadata Operations
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_metadata_field(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_metadata_field(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, [MarshalAs(UnmanagedType.LPUTF8Str)] string value, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_custom_property(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_custom_property(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, [MarshalAs(UnmanagedType.LPUTF8Str)] string value, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_remove_custom_property(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_custom_property_keys(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_xmp_metadata(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_xmp_metadata(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string xmpXml, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_xmp_property(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string namespace_, [MarshalAs(UnmanagedType.LPUTF8Str)] string property, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_xmp_property(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string namespace_, [MarshalAs(UnmanagedType.LPUTF8Str)] string property, [MarshalAs(UnmanagedType.LPUTF8Str)] string value, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_has_xmp_metadata(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_sync_metadata(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_clear_all_metadata(NativeHandle handle, out int errorCode);

        // Advanced Rendering Operations
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_page(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, int dpi, int quality, [MarshalAs(UnmanagedType.I1)] bool antiAliasing, [MarshalAs(UnmanagedType.I1)] bool renderAnnotations, [MarshalAs(UnmanagedType.I1)] bool renderFormFields, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_render_dimensions(NativeHandle handle, int pageIndex, int dpi, float scale, ref int width, ref int height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_region(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, int dpi, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_grayscale(NativeHandle handle, int pageIndex, int dpi, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_with_transparency(NativeHandle handle, int pageIndex, int dpi, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_with_background(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string backgroundColor, int dpi, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_rotated(NativeHandle handle, int pageIndex, int rotation, int dpi, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_scaled(NativeHandle handle, int pageIndex, float scale, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_page_pixels(NativeHandle handle, int pageIndex, int dpi, out int width, out int height, out int stride, out int errorCode);

        // Compliance Operations
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_check_compliance(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string standard, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_pdfa_compliant(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string level);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_pdfx_compliant(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string level);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_pdfua_compliant(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_pdfa_level(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_pdfx_level(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_validate_fonts(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_validate_colors(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_validate_images(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_validate_metadata(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_validate_accessibility(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_tagged(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_are_fonts_embedded(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_output_intent(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_generate_compliance_report(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string standard, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_compliance_summary(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_fix_suggestions(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string standard, out int count, out int errorCode);

        // Barcode Advanced Operations
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_scan_barcodes(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.I1)] bool tryHarder, [MarshalAs(UnmanagedType.I1)] bool multiplePerPage, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_scan_barcodes_in_region(NativeHandle handle, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.I1)] bool tryHarder, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_decode_barcode(NativeHandle handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string type, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_generate_barcode([MarshalAs(UnmanagedType.LPUTF8Str)] string data, [MarshalAs(UnmanagedType.LPUTF8Str)] string type, int width, int height, [MarshalAs(UnmanagedType.LPUTF8Str)] string errorCorrection, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_barcode_count(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_validate_barcode_data([MarshalAs(UnmanagedType.LPUTF8Str)] string data, [MarshalAs(UnmanagedType.LPUTF8Str)] string type);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_barcode_get_data(IntPtr barcodesPtr, int index, out int errorCode);

        // Digital Signature Advanced Operations
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_signatures(NativeHandle handle, out int count, out int errorCode);

        // pdf_get_signature is defined in Digital Signature API region above - removed duplicate

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_verify_signature_with_message(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string signatureName, out IntPtr messagePtr, out int errorCode);

        // pdf_sign_document extended version is considered a separate overload and renamed
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_sign_document_advanced(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string certificatePath, [MarshalAs(UnmanagedType.LPUTF8Str)] string password, [MarshalAs(UnmanagedType.LPUTF8Str)] string reason, [MarshalAs(UnmanagedType.LPUTF8Str)] string location, [MarshalAs(UnmanagedType.LPUTF8Str)] string contactInfo, int signatureType, int pageIndex, float x, float y, float width, float height, [MarshalAs(UnmanagedType.I1)] bool addTimestamp, [MarshalAs(UnmanagedType.LPUTF8Str)] string timestampUrl, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_sign_with_certificate_bytes(NativeHandle handle, IntPtr certPtr, int certLen, [MarshalAs(UnmanagedType.LPUTF8Str)] string password, [MarshalAs(UnmanagedType.LPUTF8Str)] string reason, [MarshalAs(UnmanagedType.LPUTF8Str)] string location, int signatureType, int pageIndex, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_add_timestamp(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string signatureName, [MarshalAs(UnmanagedType.LPUTF8Str)] string timestampUrl, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_remove_signature(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string signatureName, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_remove_all_signatures(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_signature_certificate(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string signatureName, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_modified_since_signing(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_signed(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_all_signatures_valid(NativeHandle handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_signature_field_names(NativeHandle handle, out int count, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_create_signature_field(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fieldName, int pageIndex, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_signature_get_name(IntPtr sigPtr, int index, out int errorCode);

        // Phase 1 Enterprise Signing: Credential Management

        /// <summary>
        /// Loads signing credentials from a PKCS#12 (.p12/.pfx) file.
        /// </summary>
        /// <param name="filePath">Path to the PKCS#12 file.</param>
        /// <param name="password">Password for the PKCS#12 file.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Opaque handle to signing credentials, or IntPtr.Zero on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_credentials_from_pkcs12(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string filePath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string password,
            out int errorCode);

        /// <summary>
        /// Loads signing credentials from PEM certificate and key files.
        /// </summary>
        /// <param name="certFile">Path to the certificate PEM file.</param>
        /// <param name="keyFile">Path to the private key PEM file (can be null).</param>
        /// <param name="keyPassword">Password for an encrypted key (can be null).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Opaque handle to signing credentials, or IntPtr.Zero on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_credentials_from_pem(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string certFile,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? keyFile,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? keyPassword,
            out int errorCode);

        /// <summary>
        /// Loads signing credentials from raw DER-encoded certificate and key bytes.
        /// </summary>
        /// <param name="certData">Pointer to DER-encoded certificate bytes.</param>
        /// <param name="certSize">Size of certificate data in bytes.</param>
        /// <param name="keyData">Pointer to DER-encoded private key bytes (can be IntPtr.Zero).</param>
        /// <param name="keySize">Size of key data in bytes.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Opaque handle to signing credentials, or IntPtr.Zero on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_credentials_from_der(
            IntPtr certData,
            UIntPtr certSize,
            IntPtr keyData,
            UIntPtr keySize,
            out int errorCode);

        /// <summary>
        /// Adds a certificate chain entry to signing credentials.
        /// </summary>
        /// <param name="credentials">Handle to the signing credentials.</param>
        /// <param name="certData">Pointer to DER-encoded certificate bytes.</param>
        /// <param name="certSize">Size of certificate data in bytes.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_credentials_add_chain_cert(
            IntPtr credentials,
            IntPtr certData,
            UIntPtr certSize,
            out int errorCode);

        /// <summary>
        /// Gets the certificate from signing credentials.
        /// </summary>
        /// <param name="credentials">Handle to the signing credentials.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Certificate handle, or IntPtr.Zero on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_credentials_get_certificate(
            IntPtr credentials,
            out int errorCode);

        /// <summary>
        /// Frees signing credentials.
        /// </summary>
        /// <param name="credentials">Handle to the signing credentials to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_credentials_free(IntPtr credentials);

        // Phase 1 Enterprise Signing: Certificate Inspection

        /// <summary>
        /// Loads a certificate from raw DER-encoded bytes for inspection.
        /// </summary>
        /// <param name="certData">Pointer to DER-encoded certificate bytes.</param>
        /// <param name="certSize">Size of certificate data in bytes.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Certificate handle, or IntPtr.Zero on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_certificate_load_from_bytes(
            IntPtr certData,
            UIntPtr certSize,
            out int errorCode);

        /// <summary>Load signing credentials from PEM-encoded certificate and private key strings.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_certificate_load_from_pem", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfCertificateLoadFromPem(
            string certPem, string keyPem, out int errorCode);

        /// <summary>
        /// Applies a CMS/PKCS#7 detached signature to raw PDF bytes and returns the signed PDF.
        /// The caller must free the returned buffer with <see cref="FreeBytes"/>.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_sign_bytes", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial byte* PdfSignBytes(
            byte* pdfData, nuint pdfLen,
            IntPtr certificateHandle,
            string? reason, string? location,
            out nuint outLen, out int errorCode);

        /// <summary>
        /// Gets the common name (CN) from a certificate handle.
        /// </summary>
        /// <param name="cert">Certificate handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_certificate_get_cn(
            IntPtr cert,
            out int errorCode);

        /// <summary>
        /// Gets the issuer name from a certificate handle (standalone certificate inspection).
        /// </summary>
        /// <param name="cert">Certificate handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>UTF-8 null-terminated string pointer. Must be freed with FreeString().</returns>
        [LibraryImport(LibName, EntryPoint = "pdf_certificate_get_issuer", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_certificate_handle_get_issuer(
            IntPtr cert,
            out int errorCode);

        /// <summary>
        /// Gets the size in bytes of a certificate's DER-encoded data.
        /// </summary>
        /// <param name="cert">Certificate handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Certificate size in bytes.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial UIntPtr pdf_certificate_get_size(
            IntPtr cert,
            out int errorCode);

        /// <summary>
        /// Adds a timestamp to a signed PDF document using a TSA (Time Stamp Authority).
        /// Operates on raw PDF byte data and returns timestamped PDF bytes.
        /// </summary>
        /// <param name="pdfData">Pointer to signed PDF data bytes.</param>
        /// <param name="pdfLen">Length of PDF data.</param>
        /// <param name="signatureIndex">Index of the signature to timestamp.</param>
        /// <param name="tsaUrl">URL of the Time Stamp Authority.</param>
        /// <param name="outData">Output pointer to timestamped PDF bytes.</param>
        /// <param name="outLen">Output length of timestamped PDF bytes.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, EntryPoint = "pdf_add_timestamp", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_add_timestamp_bytes(
            IntPtr pdfData,
            UIntPtr pdfLen,
            int signatureIndex,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string tsaUrl,
            out IntPtr outData,
            out UIntPtr outLen,
            out int errorCode);

        // Phase 1 Enterprise Signing: Document Signing

        /// <summary>
        /// Signs a PDF document in memory and returns the signed bytes.
        /// </summary>
        /// <param name="pdfData">Pointer to PDF data bytes.</param>
        /// <param name="pdfLen">Length of PDF data.</param>
        /// <param name="credentials">Handle to signing credentials.</param>
        /// <param name="reason">Signing reason (can be null).</param>
        /// <param name="location">Signing location (can be null).</param>
        /// <param name="contact">Contact info (can be null).</param>
        /// <param name="algorithm">Digest algorithm (0=SHA1, 1=SHA256, 2=SHA384, 3=SHA512).</param>
        /// <param name="subfilter">Signature subfilter (0=PKCS7_DETACHED, 1=PKCS7_SHA1, 2=CADES_DETACHED).</param>
        /// <param name="outData">Output pointer to signed PDF bytes.</param>
        /// <param name="outLen">Output length of signed PDF bytes.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_sign(
            IntPtr pdfData,
            UIntPtr pdfLen,
            IntPtr credentials,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? reason,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? location,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? contact,
            int algorithm,
            int subfilter,
            out IntPtr outData,
            out UIntPtr outLen,
            out int errorCode);

        /// <summary>
        /// Signs a PDF file and writes the signed output to another file.
        /// </summary>
        /// <param name="inputPath">Path to the input PDF file.</param>
        /// <param name="outputPath">Path for the signed output file.</param>
        /// <param name="credentials">Handle to signing credentials.</param>
        /// <param name="reason">Signing reason (can be null).</param>
        /// <param name="location">Signing location (can be null).</param>
        /// <param name="contact">Contact info (can be null).</param>
        /// <param name="algorithm">Digest algorithm.</param>
        /// <param name="subfilter">Signature subfilter.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_sign_file(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string inputPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string outputPath,
            IntPtr credentials,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? reason,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? location,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? contact,
            int algorithm,
            int subfilter,
            out int errorCode);

        /// <summary>
        /// Signs a PDF document with a visible signature appearance.
        /// </summary>
        /// <param name="pdfData">Pointer to PDF data bytes.</param>
        /// <param name="pdfLen">Length of PDF data.</param>
        /// <param name="credentials">Handle to signing credentials.</param>
        /// <param name="pageNum">Page number for the visible signature.</param>
        /// <param name="x">X coordinate of the signature rectangle.</param>
        /// <param name="y">Y coordinate of the signature rectangle.</param>
        /// <param name="width">Width of the signature rectangle.</param>
        /// <param name="height">Height of the signature rectangle.</param>
        /// <param name="reason">Signing reason (can be null).</param>
        /// <param name="location">Signing location (can be null).</param>
        /// <param name="contact">Contact info (can be null).</param>
        /// <param name="algorithm">Digest algorithm.</param>
        /// <param name="outData">Output pointer to signed PDF bytes.</param>
        /// <param name="outLen">Output length of signed PDF bytes.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_sign_with_appearance(
            IntPtr pdfData,
            UIntPtr pdfLen,
            IntPtr credentials,
            int pageNum,
            float x,
            float y,
            float width,
            float height,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? reason,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? location,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? contact,
            int algorithm,
            out IntPtr outData,
            out UIntPtr outLen,
            out int errorCode);

        /// <summary>
        /// Embeds LTV (Long-Term Validation) data into a signed PDF.
        /// </summary>
        /// <param name="pdfData">Pointer to signed PDF data bytes.</param>
        /// <param name="pdfLen">Length of PDF data.</param>
        /// <param name="ocspData">Pointer to OCSP response bytes (can be IntPtr.Zero).</param>
        /// <param name="ocspLen">Length of OCSP data.</param>
        /// <param name="crlData">Pointer to CRL bytes (can be IntPtr.Zero).</param>
        /// <param name="crlLen">Length of CRL data.</param>
        /// <param name="outData">Output pointer to result PDF bytes.</param>
        /// <param name="outLen">Output length of result PDF bytes.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_embed_ltv_data(
            IntPtr pdfData,
            UIntPtr pdfLen,
            IntPtr ocspData,
            UIntPtr ocspLen,
            IntPtr crlData,
            UIntPtr crlLen,
            out IntPtr outData,
            out UIntPtr outLen,
            out int errorCode);

        /// <summary>
        /// Saves signed PDF bytes to a file.
        /// </summary>
        /// <param name="pdfData">Pointer to signed PDF data bytes.</param>
        /// <param name="pdfLen">Length of PDF data.</param>
        /// <param name="outputPath">Path for the output file.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>True on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_save_signed(
            IntPtr pdfData,
            UIntPtr pdfLen,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string outputPath,
            out int errorCode);

        /// <summary>
        /// Frees a byte buffer returned by signing/LTV FFI functions.
        /// </summary>
        /// <param name="data">Pointer to the byte buffer.</param>
        /// <param name="len">Length of the byte buffer.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_signed_bytes_free(IntPtr data, UIntPtr len);

        // Security Write Operations
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_user_password(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string password, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_owner_password(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string password, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_passwords(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string userPassword, [MarshalAs(UnmanagedType.LPUTF8Str)] string ownerPassword, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_remove_user_password(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_remove_owner_password(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_remove_security(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_encryption_level(NativeHandle handle, int level, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_permissions(NativeHandle handle, int permissions, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_print_permission(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool allowed, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_copy_permission(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool allowed, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_modify_permission(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool allowed, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_annotate_permission(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool allowed, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_fill_forms_permission(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool allowed, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_extract_permission(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool allowed, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_assemble_permission(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool allowed, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_print_hq_permission(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool allowed, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_set_encrypt_metadata(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool encrypt, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_validate_password(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string password);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_is_owner_password(NativeHandle handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string password);

        // PDF Creator Operations
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_creator_new(out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_creator_free(IntPtr creatorHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_add_page(IntPtr creatorHandle, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_set_current_page(IntPtr creatorHandle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_draw_text(IntPtr creatorHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, float x, float y, [MarshalAs(UnmanagedType.LPUTF8Str)] string fontName, float fontSize, [MarshalAs(UnmanagedType.LPUTF8Str)] string color, [MarshalAs(UnmanagedType.I1)] bool bold, [MarshalAs(UnmanagedType.I1)] bool italic, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_draw_text_block(IntPtr creatorHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string fontName, float fontSize, [MarshalAs(UnmanagedType.LPUTF8Str)] string color, int alignment, float lineHeight, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_draw_image(IntPtr creatorHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string imagePath, float x, float y, float width, float height, [MarshalAs(UnmanagedType.I1)] bool preserveAspect, float opacity, float rotation, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_draw_image_bytes(IntPtr creatorHandle, IntPtr dataPtr, int dataLen, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, float x, float y, float width, float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_draw_rectangle(IntPtr creatorHandle, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string fillColor, [MarshalAs(UnmanagedType.LPUTF8Str)] string strokeColor, float strokeWidth, float opacity, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_draw_circle(IntPtr creatorHandle, float centerX, float centerY, float radius, [MarshalAs(UnmanagedType.LPUTF8Str)] string fillColor, [MarshalAs(UnmanagedType.LPUTF8Str)] string strokeColor, float strokeWidth, float opacity, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_draw_line(IntPtr creatorHandle, float x1, float y1, float x2, float y2, [MarshalAs(UnmanagedType.LPUTF8Str)] string color, float width, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_draw_polygon(IntPtr creatorHandle, float[] xCoords, float[] yCoords, int pointCount, [MarshalAs(UnmanagedType.LPUTF8Str)] string fillColor, [MarshalAs(UnmanagedType.LPUTF8Str)] string strokeColor, float strokeWidth, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_add_link(IntPtr creatorHandle, float x, float y, float width, float height, [MarshalAs(UnmanagedType.LPUTF8Str)] string url, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_add_page_link(IntPtr creatorHandle, float x, float y, float width, float height, int targetPage, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_set_title(IntPtr creatorHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string title, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_set_author(IntPtr creatorHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string author, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_set_subject(IntPtr creatorHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string subject, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_set_keywords(IntPtr creatorHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string keywords, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_add_bookmark(IntPtr creatorHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string title, int pageIndex, int parentIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_embed_font(IntPtr creatorHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string fontPath, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_set_compression(IntPtr creatorHandle, int level, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_save(IntPtr creatorHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string outputPath, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_creator_save_to_bytes(IntPtr creatorHandle, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_creator_get_page_count(IntPtr creatorHandle, out int errorCode);

        // FreeBytes is defined in Utility Functions region above - removed duplicate

        #endregion

        #region Accessibility Operations

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_accessibility_is_tagged(IntPtr documentHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_accessibility_get_structure_tree(IntPtr documentHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_accessibility_auto_tag(IntPtr documentHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string language, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_accessibility_set_alt_text(IntPtr documentHandle, int page, int mcid, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_accessibility_set_language(IntPtr documentHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string language, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_accessibility_set_title(IntPtr documentHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string title, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_structure_tree_free(IntPtr treeHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_struct_elem_free(IntPtr elemHandle);

        #endregion

        #region Optimization Operations

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_open_mmap([MarshalAs(UnmanagedType.LPUTF8Str)] string path, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_optimize_subset_fonts(IntPtr documentHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_optimize_downsample_images(IntPtr documentHandle, int dpi, int quality, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_optimize_deduplicate(IntPtr documentHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_optimize_full(IntPtr documentHandle, int dpi, int quality, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_optimization_result_bytes_saved(IntPtr resultHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_optimization_result_free(IntPtr resultHandle);

        #endregion

        #region Enterprise Operations (Bates, Comparison, Stamping)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_bates_apply(IntPtr documentHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string prefix, int startNumber, int numDigits, int position, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_bates_apply_advanced(IntPtr documentHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string prefix, [MarshalAs(UnmanagedType.LPUTF8Str)] string suffix, int startNumber, int numDigits, int position, float fontSize, float margin, int startPage, int endPage, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_compare_pages(IntPtr docA, int pageA, IntPtr docB, int pageB, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial double pdf_comparison_get_similarity(IntPtr comparisonHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_comparison_get_diff_count(IntPtr comparisonHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_comparison_get_diff(IntPtr comparisonHandle, int index);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_comparison_get_diff_type(IntPtr diffHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_compare_documents(IntPtr docA, IntPtr docB, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_comparison_free(IntPtr comparisonHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_document_comparison_free(IntPtr comparisonHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_stamp_header(IntPtr documentHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, int alignment, float size, float margin, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_stamp_footer(IntPtr documentHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, int alignment, float size, float margin, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_stamp_header_footer(IntPtr documentHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string headerText, [MarshalAs(UnmanagedType.LPUTF8Str)] string footerText, int alignment, float size, float margin, out int errorCode);

        #endregion

        #region TSA (Time Stamp Authority)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_tsa_client_create(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string url,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string username,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string password,
            int timeoutSeconds, int hashAlgorithm,
            [MarshalAs(UnmanagedType.I1)] bool useNonce,
            [MarshalAs(UnmanagedType.I1)] bool certReq,
            out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_tsa_client_free(IntPtr clientHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_tsa_request_timestamp(IntPtr clientHandle, IntPtr data, UIntPtr dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_tsa_request_timestamp_hash(IntPtr clientHandle, IntPtr hash, UIntPtr hashLen, int hashAlgorithm, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_timestamp_parse(IntPtr bytes, UIntPtr len, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_timestamp_get_token(IntPtr timestampHandle, out UIntPtr outLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_timestamp_get_time(IntPtr timestampHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_timestamp_get_serial(IntPtr timestampHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_timestamp_get_tsa_name(IntPtr timestampHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_timestamp_get_policy_oid(IntPtr timestampHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_timestamp_get_hash_algorithm(IntPtr timestampHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_timestamp_get_message_imprint(IntPtr timestampHandle, out UIntPtr outLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_timestamp_verify(IntPtr timestampHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_timestamp_free(IntPtr timestampHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_signature_add_timestamp(IntPtr signatureHandle, IntPtr timestampHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_signature_has_timestamp(IntPtr signatureHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_signature_get_timestamp(IntPtr signatureHandle, out int errorCode);

        // PAdES Level Enforcement

        /// <summary>
        /// Validates the PAdES level of a signature in a PDF document.
        /// </summary>
        /// <param name="handle">Document handle.</param>
        /// <param name="sigIndex">Zero-based signature index.</param>
        /// <param name="level">PAdES level (0=B-B, 1=B-T, 2=B-LT, 3=B-LTA).</param>
        /// <param name="errorCode">Output error code.</param>
        /// <returns>1 if valid, 0 if not valid, -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_pades_validate_level(IntPtr handle, int sigIndex, int level, out int errorCode);

        /// <summary>
        /// Signs a PDF with PAdES level enforcement.
        /// </summary>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_pades_sign(
            IntPtr pdfData, UIntPtr pdfLen,
            IntPtr credentials, int level,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string tsaUrl,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string reason,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string location,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string contact,
            out IntPtr outData, out UIntPtr outLen,
            out int errorCode);

        /// <summary>
        /// Detects the PAdES level of an existing signature.
        /// </summary>
        /// <param name="handle">Document handle.</param>
        /// <param name="sigIndex">Zero-based signature index.</param>
        /// <param name="errorCode">Output error code.</param>
        /// <returns>PAdES level (0-3) or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_pades_get_level(IntPtr handle, int sigIndex, out int errorCode);

        #endregion

        #region PDF/UA Validation (extended)

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_pdf_ua_warning_count(IntPtr resultsHandle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_pdf_ua_get_warning(IntPtr resultsHandle, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_pdf_ua_get_stats(
            IntPtr resultsHandle,
            out int outStructElements, out int outImages, out int outTables,
            out int outForms, out int outAnnotations, out int outPages,
            out int errorCode);

        #endregion

        #region FDF/XFDF Import/Export

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_import_form_data(IntPtr documentHandle, [MarshalAs(UnmanagedType.LPUTF8Str)] string dataPath, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_editor_import_fdf_bytes(IntPtr documentHandle, IntPtr data, UIntPtr dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_editor_import_xfdf_bytes(IntPtr documentHandle, IntPtr data, UIntPtr dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_export_form_data_to_bytes(IntPtr documentHandle, int formatType, out UIntPtr outLen, out int errorCode);

        #endregion

        #region Table Detection

        /// <summary>
        /// Detects tables on a specific page.
        /// </summary>
        /// <param name="document">The document handle.</param>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Table list handle, or IntPtr.Zero on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_detect_tables_on_page(IntPtr document, int pageIndex, out int errorCode);

        /// <summary>
        /// Gets number of detected tables in the list.
        /// </summary>
        /// <param name="list">The table list handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Number of tables found.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_table_list_count(IntPtr list, out int errorCode);

        /// <summary>
        /// Gets the row count for a table at the given index.
        /// </summary>
        /// <param name="list">The table list handle.</param>
        /// <param name="index">Table index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Number of rows, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_table_get_row_count(IntPtr list, int index, out int errorCode);

        /// <summary>
        /// Gets the column count for a table at the given index.
        /// </summary>
        /// <param name="list">The table list handle.</param>
        /// <param name="index">Table index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Number of columns, or -1 on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_table_get_col_count(IntPtr list, int index, out int errorCode);

        /// <summary>
        /// Gets the text content of a specific cell. Result must be freed with free_string().
        /// </summary>
        /// <param name="list">The table list handle.</param>
        /// <param name="tableIndex">Table index (0-based).</param>
        /// <param name="row">Row index (0-based).</param>
        /// <param name="col">Column index (0-based).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Cell text as a string pointer, or IntPtr.Zero on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_table_get_cell_text(IntPtr list, int tableIndex, int row, int col, out int errorCode);

        /// <summary>
        /// Gets the bounding box of a table.
        /// </summary>
        /// <param name="list">The table list handle.</param>
        /// <param name="index">Table index (0-based).</param>
        /// <param name="x">Output: x coordinate.</param>
        /// <param name="y">Output: y coordinate.</param>
        /// <param name="w">Output: width.</param>
        /// <param name="h">Output: height.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>true on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_table_get_bounds(IntPtr list, int index, out float x, out float y, out float w, out float h, out int errorCode);

        /// <summary>
        /// Frees a table list handle.
        /// </summary>
        /// <param name="list">The table list handle to free.</param>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_table_list_free(IntPtr list);

        #endregion

        #region Writer / Encryption

        /// <summary>
        /// Enable or disable compression for the writer (editing mode only).
        /// </summary>
        /// <param name="document">The document handle (must be in editing mode).</param>
        /// <param name="enable">true to enable compression, false to disable.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>true on success, false on error.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_writer_enable_compression(IntPtr document, [MarshalAs(UnmanagedType.I1)] bool enable, out int errorCode);

        /// <summary>
        /// Authenticate with the owner password.
        /// </summary>
        /// <param name="document">The document handle.</param>
        /// <param name="password">The owner password (UTF-8).</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>true if authentication succeeded, false otherwise.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_authenticate_owner(IntPtr document,
            string password, out int errorCode);

        /// <summary>
        /// Get document permissions as a bitmask. Returns -1 if not encrypted.
        /// </summary>
        /// <param name="document">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Permission bitmask, or -1 if not encrypted.</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_get_permissions(IntPtr document, out int errorCode);

        /// <summary>
        /// Get encryption algorithm info. 0=None, 1=RC4-40, 2=RC4-128, 3=AES-128, 4=AES-256.
        /// </summary>
        /// <param name="document">The document handle.</param>
        /// <param name="errorCode">Output parameter for error code.</param>
        /// <returns>Encryption algorithm code (0-4).</returns>
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_get_encryption_info(IntPtr document, out int errorCode);

        #endregion

        #region v0.3.23 New FFI Functions

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_extract_all_text(IntPtr handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_to_html_all(IntPtr handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_to_plain_text_all(IntPtr handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_is_encrypted(IntPtr handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_authenticate(IntPtr handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string password, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_document_has_xfa(IntPtr handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_extract_words(IntPtr handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_word_count(IntPtr words);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_word_get_text(IntPtr words, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_word_get_bbox(IntPtr words, int index, out float x, out float y, out float w, out float h, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_word_list_free(IntPtr handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_extract_text_lines(IntPtr handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_line_count(IntPtr lines);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_line_get_text(IntPtr lines, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_line_get_bbox(IntPtr lines, int index, out float x, out float y, out float w, out float h, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_line_list_free(IntPtr handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_extract_tables(IntPtr handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_table_count(IntPtr tables);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_table_get_row_count(IntPtr tables, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_table_get_col_count(IntPtr tables, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_table_get_cell_text(IntPtr tables, int tableIndex, int row, int col, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_table_list_free(IntPtr handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_extract_text_in_rect(IntPtr handle, int pageIndex, float x, float y, float w, float h, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_form_fields(IntPtr handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_form_field_count(IntPtr fields);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_form_field_get_name(IntPtr fields, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_form_field_get_type(IntPtr fields, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_form_field_get_value(IntPtr fields, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_form_field_list_free(IntPtr handle);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_remove_headers(IntPtr handle, float threshold, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_remove_footers(IntPtr handle, float threshold, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_remove_artifacts(IntPtr handle, float threshold, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_validate_pdf_a_level(IntPtr document, int level, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_pdf_a_is_compliant(IntPtr results, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_pdf_a_error_count(IntPtr results);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_pdf_a_get_error(IntPtr results, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_pdf_a_results_free(IntPtr results);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_validate_pdf_x_level(IntPtr document, int level, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_pdf_x_is_compliant(IntPtr results, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_pdf_x_error_count(IntPtr results);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_pdf_x_get_error(IntPtr results, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_pdf_x_results_free(IntPtr results);

        // Search
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_search_all(IntPtr handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string searchTerm, [MarshalAs(UnmanagedType.I1)] bool caseSensitive, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_search_page(IntPtr handle, int pageIndex, [MarshalAs(UnmanagedType.LPUTF8Str)] string searchTerm, [MarshalAs(UnmanagedType.I1)] bool caseSensitive, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_search_result_count(IntPtr results);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_search_result_get_text(IntPtr results, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_search_result_get_page(IntPtr results, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_search_result_get_bbox(IntPtr results, int index, out float x, out float y, out float width, out float height, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_search_result_free(IntPtr handle);

        // Paths
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_extract_paths(IntPtr handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_path_count(IntPtr paths);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_path_get_bbox(IntPtr paths, int index, out float x, out float y, out float w, out float h, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float pdf_oxide_path_get_stroke_width(IntPtr paths, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_path_list_free(IntPtr handle);

        // Metadata
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_page_labels(IntPtr handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_xmp_metadata(IntPtr handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_outline(IntPtr handle, out int errorCode);

        // Chars
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_extract_chars(IntPtr handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_char_count(IntPtr chars);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial uint pdf_oxide_char_get_char(IntPtr chars, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_char_get_bbox(IntPtr chars, int index, out float x, out float y, out float w, out float h, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_char_list_free(IntPtr handle);

        // PDF Creator from image (JPEG / PNG → single-page PDF wrapper).
        [LibraryImport(LibName, EntryPoint = "pdf_from_image", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle PdfFromImage(string path, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_from_image_bytes", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle PdfFromImageBytes(
            [In] byte[] data,
            int dataLen,
            out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_merge(IntPtr paths, int pathCount, out int dataLen, out int errorCode);

        // FDF/XFDF export
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_export_form_data_to_bytes(IntPtr document, int formatType, out int outLen, out int errorCode);

        // Fonts
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_embedded_fonts(IntPtr handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_font_count(IntPtr fonts);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_font_get_name(IntPtr fonts, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_font_list_free(IntPtr handle);

        // Region extraction
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_extract_words_in_rect(IntPtr handle, int pageIndex, float x, float y, float w, float h, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_extract_images_in_rect(IntPtr handle, int pageIndex, float x, float y, float w, float h, out int errorCode);

        // Rendering
        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_page(IntPtr doc, int pageIndex, int format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_page_zoom(IntPtr doc, int pageIndex, float zoom, int format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_page_fit(IntPtr doc, int pageIndex, int fitWidth, int fitHeight, int format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_page_thumbnail(IntPtr doc, int pageIndex, int size, int format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_render_page_raw(IntPtr doc, int pageIndex, int dpi, out int outWidth, out int outHeight, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_rendered_image_width(IntPtr img, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_rendered_image_height(IntPtr img, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_rendered_image_data(IntPtr img, out int dataLen, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_save_rendered_image(IntPtr img, [MarshalAs(UnmanagedType.LPUTF8Str)] string filePath, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_rendered_image_free(IntPtr handle);

        // Barcodes (already in existing NativeMethods for BarcodeManager)

        #endregion

        #region BarcodesSignaturesRenderingManager + OCRComplianceCacheManager Support

        // --- Barcode functions used by BarcodesManager ---

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_generate_barcode(
            string data,
            int format,
            int size,
            out int errorCode);

        // --- Certificate functions used by SignaturesManager ---

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_certificate_load_from_bytes(
            byte[] certData,
            int certSize,
            string? password,
            out int errorCode);

        // --- Signature functions used by SignaturesManager ---

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_get_signature_count(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_document_get_signature(NativeHandle handle, int index, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_verify_all_signatures(NativeHandle handle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_signature_verify(NativeHandle signature, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int pdf_signature_verify_detached(
            NativeHandle signature,
            byte* pdfData,
            nuint pdfLen,
            out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_signature_get_signing_reason(NativeHandle signatureHandle, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_signature_get_signing_location(NativeHandle signatureHandle, out int errorCode);

        // --- Rendering functions used by PageRenderingManager ---

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_estimate_render_time(NativeHandle handle, int pageIndex, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_create_renderer(int dpi, int format, int quality, [MarshalAs(UnmanagedType.I1)] bool antiAlias, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_render_page(NativeHandle handle, int pageIndex, int format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_render_page_region(NativeHandle handle, int pageIndex, float x, float y, float w, float h, int format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_render_page_fit(NativeHandle handle, int pageIndex, int w, int h, int format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_render_page_zoom(NativeHandle handle, int pageIndex, float zoom, int format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_render_page_thumbnail(NativeHandle handle, int pageIndex, int size, int format, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_rendered_image_width(NativeHandle img, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_get_rendered_image_height(NativeHandle img, out int errorCode);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_get_rendered_image_data(NativeHandle img, out int dataLen, out int errorCode);

        // --- Cache functions used by CacheManager ---

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_set_caching_enabled(NativeHandle handle, [MarshalAs(UnmanagedType.I1)] bool enabled, out int errorCode);

        #endregion

        #region DocumentEditor Extended Operations (17 functions)

        /// <summary>
        /// Gets the producer metadata from a DocumentEditor.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_producer", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr document_editor_get_producer(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Sets the producer metadata on a DocumentEditor.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_set_producer", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_set_producer(
            NativeHandle handle,
            string value,
            out int errorCode);

        /// <summary>
        /// Gets the creation date from a DocumentEditor.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_creation_date", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr document_editor_get_creation_date(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Sets the creation date on a DocumentEditor.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_set_creation_date", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_set_creation_date(
            NativeHandle handle,
            string dateStr,
            out int errorCode);

        /// <summary>
        /// Deletes a page from the document.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_delete_page", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_delete_page(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Moves a page from one position to another.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_move_page", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_move_page(
            NativeHandle handle,
            int from,
            int to,
            out int errorCode);

        /// <summary>
        /// Gets the rotation of a page in degrees.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_page_rotation", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_get_page_rotation(
            NativeHandle handle,
            int page,
            out int errorCode);

        /// <summary>
        /// Sets the rotation of a page in degrees.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_set_page_rotation", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_set_page_rotation(
            NativeHandle handle,
            int page,
            int degrees,
            out int errorCode);

        /// <summary>
        /// Erases content in a rectangular region on a page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_erase_region", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_erase_region(
            NativeHandle handle,
            int page,
            float x,
            float y,
            float w,
            float h,
            out int errorCode);

        /// <summary>
        /// Flattens annotations on a specific page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_flatten_annotations", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_flatten_annotations(
            NativeHandle handle,
            int page,
            out int errorCode);

        /// <summary>
        /// Flattens all annotations in the entire document.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_flatten_all_annotations", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_flatten_all_annotations(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Crops margins on all pages.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_crop_margins", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_crop_margins(
            NativeHandle handle,
            float left,
            float right,
            float top,
            float bottom,
            out int errorCode);

        /// <summary>
        /// Merges pages from another PDF file into the current document.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_merge_from", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_merge_from(
            NativeHandle handle,
            string sourcePath,
            out int errorCode);

        /// <summary>
        /// Saves the document with encryption (user and owner passwords).
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_save_encrypted", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_save_encrypted(
            NativeHandle handle,
            string path,
            string userPassword,
            string ownerPassword,
            out int errorCode);

        /// <summary>
        /// Sets a form field value by name.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_set_form_field_value", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_set_form_field_value(
            NativeHandle handle,
            string name,
            string value,
            out int errorCode);

        /// <summary>
        /// Flattens all form fields in the document (bakes values into page content).
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_flatten_forms", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_flatten_forms(
            NativeHandle handle,
            out int errorCode);

        /// <summary>
        /// Flattens form fields on a specific page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_flatten_forms_on_page", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_flatten_forms_on_page(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Returns the number of flatten warnings from the last save.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_flatten_warnings_count")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_flatten_warnings_count(NativeHandle handle);

        /// <summary>
        /// Returns the index-th flatten warning as a native string (must be freed with free_string).
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_flatten_warning")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr document_editor_flatten_warning(
            NativeHandle handle,
            int index,
            out int errorCode);

        #endregion

        #region DocumentEditor New Methods (v0.3.39)

        /// <summary>Opens a DocumentEditor from an in-memory byte buffer.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_open_from_bytes")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle document_editor_open_from_bytes(
            [In] byte[] data,
            nuint length,
            out int errorCode);

        /// <summary>Saves the editor to an in-memory byte buffer.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_save_to_bytes")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr document_editor_save_to_bytes(
            NativeHandle handle,
            out nuint outLen,
            out int errorCode);

        /// <summary>Extracts a subset of pages from the editor into a new in-memory PDF.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_extract_pages_to_bytes")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial IntPtr document_editor_extract_pages_to_bytes(
            NativeHandle handle, int* pages, nuint count, out nuint outLen, out int errorCode);

        /// <summary>Converts the document in-place to the given PDF/A conformance level.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_convert_to_pdf_a")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_convert_to_pdf_a(
            NativeHandle handle, int level, out int errorCode);

        /// <summary>Saves the document with AES-256 encryption and returns the bytes.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_save_encrypted_to_bytes")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr document_editor_save_encrypted_to_bytes(
            NativeHandle handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string userPassword,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string ownerPassword,
            out nuint outLen, out int errorCode);

        /// <summary>Saves the editor to bytes with compress / garbage-collect / linearize options.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_save_to_bytes_with_options")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr document_editor_save_to_bytes_with_options(
            NativeHandle handle,
            [MarshalAs(UnmanagedType.I1)] bool compress,
            [MarshalAs(UnmanagedType.I1)] bool garbageCollect,
            [MarshalAs(UnmanagedType.I1)] bool linearize,
            out nuint outLen,
            out int errorCode);

        /// <summary>Gets the keywords metadata from a DocumentEditor.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_keywords", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr document_editor_get_keywords(
            NativeHandle handle,
            out int errorCode);

        /// <summary>Sets the keywords metadata on a DocumentEditor.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_set_keywords", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_set_keywords(
            NativeHandle handle,
            string keywords,
            out int errorCode);

        /// <summary>Merges pages from an in-memory PDF byte buffer.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_merge_from_bytes")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_merge_from_bytes(
            NativeHandle handle,
            [In] byte[] data,
            nuint length,
            out int errorCode);

        /// <summary>Embeds a file attachment into the document.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_embed_file", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_embed_file(
            NativeHandle handle,
            string name,
            [In] byte[] data,
            nuint length,
            out int errorCode);

        /// <summary>Burns in redaction annotations on a single page.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_apply_page_redactions")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_apply_page_redactions(
            NativeHandle handle,
            nuint page,
            out int errorCode);

        /// <summary>Burns in all pending redaction annotations.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_apply_all_redactions")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_apply_all_redactions(
            NativeHandle handle,
            out int errorCode);

        /// <summary>Rotates all pages by degrees (additive).</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_rotate_all_pages")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_rotate_all_pages(
            NativeHandle handle,
            int degrees,
            out int errorCode);

        /// <summary>Rotates a single page by degrees (additive).</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_rotate_page_by")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_rotate_page_by(
            NativeHandle handle,
            nuint page,
            int degrees,
            out int errorCode);

        /// <summary>Gets the MediaBox of a page.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_page_media_box")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_get_page_media_box(
            NativeHandle handle,
            nuint page,
            out double x, out double y, out double w, out double h,
            out int errorCode);

        /// <summary>Sets the MediaBox of a page.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_set_page_media_box")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_set_page_media_box(
            NativeHandle handle,
            nuint page,
            double x, double y, double w, double h,
            out int errorCode);

        /// <summary>Gets the CropBox of a page. Returns zeros if no CropBox is set.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_get_page_crop_box")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_get_page_crop_box(
            NativeHandle handle,
            nuint page,
            out double x, out double y, out double w, out double h,
            out int errorCode);

        /// <summary>Sets the CropBox of a page.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_set_page_crop_box")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_set_page_crop_box(
            NativeHandle handle,
            nuint page,
            double x, double y, double w, double h,
            out int errorCode);

        /// <summary>
        /// Erases multiple rectangular regions on a page.
        /// rects is a flat [x,y,w,h,...] array; rectsCount is the number of rectangles.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_erase_regions")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_erase_regions(
            NativeHandle handle,
            nuint page,
            [In] double[] rects,
            nuint rectsCount,
            out int errorCode);

        /// <summary>Clears all pending erase-region entries for a page.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_clear_erase_regions")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_clear_erase_regions(
            NativeHandle handle,
            nuint page,
            out int errorCode);

        /// <summary>Returns 1 if marked for flatten, 0 if not, -1 on error.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_is_page_marked_for_flatten")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_is_page_marked_for_flatten(
            NativeHandle handle,
            nuint page);

        /// <summary>Removes the flatten mark from a page.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_unmark_page_for_flatten")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_unmark_page_for_flatten(
            NativeHandle handle,
            nuint page,
            out int errorCode);

        /// <summary>Returns 1 if marked for redaction, 0 if not, -1 on error.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_is_page_marked_for_redaction")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_is_page_marked_for_redaction(
            NativeHandle handle,
            nuint page);

        /// <summary>Removes the redaction mark from a page.</summary>
        [LibraryImport(LibName, EntryPoint = "document_editor_unmark_page_for_redaction")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int document_editor_unmark_page_for_redaction(
            NativeHandle handle,
            nuint page,
            out int errorCode);

        #endregion

        #region PdfDocument Extended Operations (8 functions)

        /// <summary>
        /// Opens a PDF document with a password.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_open_with_password", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_document_open_with_password(
            string path,
            string password,
            out int errorCode);

        /// <summary>
        /// Gets embedded images from a page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_get_embedded_images", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_document_get_embedded_images(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Gets annotations from a page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_get_page_annotations", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_document_get_page_annotations(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Erases the header on a specific page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_erase_header", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_erase_header(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Erases the footer on a specific page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_erase_footer", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_erase_footer(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Erases artifacts on a specific page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_erase_artifacts", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_document_erase_artifacts(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Extracts text lines within a rectangular region on a page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_extract_lines_in_rect", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_document_extract_lines_in_rect(
            NativeHandle handle,
            int pageIndex,
            float x,
            float y,
            float w,
            float h,
            out int errorCode);

        /// <summary>
        /// Extracts tables within a rectangular region on a page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_extract_tables_in_rect", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_document_extract_tables_in_rect(
            NativeHandle handle,
            int pageIndex,
            float x,
            float y,
            float w,
            float h,
            out int errorCode);

        #endregion

        #region PdfPage Extended Operations (2 functions)

        /// <summary>
        /// Gets page elements (text spans) for a page.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_get_elements", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial NativeHandle pdf_page_get_elements(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        /// <summary>
        /// Gets the rotation of a page in degrees.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_get_rotation", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_page_get_rotation(
            NativeHandle handle,
            int pageIndex,
            out int errorCode);

        #endregion

        #region PdfForm Operations (1 function)

        /// <summary>
        /// Imports form data from a file (FDF/XFDF). Currently unsupported via PdfDocument handle.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_form_import_from_file", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_form_import_from_file(
            IntPtr document,
            string filename,
            out int errorCode);

        #endregion

        #region Barcode Extended Operations (2 functions)

        /// <summary>
        /// Gets the PNG image data from a barcode handle. The returned pointer is a buffer
        /// of length <paramref name="outLen"/> bytes and must be freed with <see cref="FreeBytes"/>.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_barcode_get_image_png", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_barcode_get_image_png(
            NativeHandle barcodeHandle,
            int sizePx,
            out int outLen,
            out int errorCode);

        /// <summary>
        /// Gets the SVG representation of a barcode.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_barcode_get_svg", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_barcode_get_svg(
            NativeHandle barcodeHandle,
            int sizePx,
            out int errorCode);

        #endregion

        #region Annotation Accessors (15 functions)

        /// <summary>
        /// Gets the number of annotations in an annotation list.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_annotation_count(NativeHandle annotations);

        /// <summary>
        /// Gets the type string of an annotation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_type", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_annotation_get_type(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the content/text of an annotation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_content", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_annotation_get_content(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the bounding rectangle of an annotation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_rect", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_annotation_get_rect(
            NativeHandle annotations,
            int index,
            out float x,
            out float y,
            out float width,
            out float height,
            out int errorCode);

        /// <summary>
        /// Gets the subtype of an annotation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_subtype", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_annotation_get_subtype(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if an annotation is marked as deleted.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_is_marked_deleted", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_oxide_annotation_is_marked_deleted(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the creation date of an annotation as a Unix timestamp.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_creation_date", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_oxide_annotation_get_creation_date(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the modification date of an annotation as a Unix timestamp.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_modification_date", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial long pdf_oxide_annotation_get_modification_date(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the author of an annotation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_author", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_annotation_get_author(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the border width of an annotation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_border_width", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float pdf_oxide_annotation_get_border_width(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the color of an annotation as a packed RGB uint (0xRRGGBB).
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_get_color", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial uint pdf_oxide_annotation_get_color(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if an annotation is hidden.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_is_hidden", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_oxide_annotation_is_hidden(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if an annotation is printable.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_is_printable", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_oxide_annotation_is_printable(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if an annotation is read-only.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_is_read_only", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_oxide_annotation_is_read_only(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Frees an annotation list handle.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_annotation_list_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_annotation_list_free(IntPtr handle);

        #endregion

        #region Image Accessors (8 functions)

        /// <summary>
        /// Gets the number of images in an image list.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_image_count(NativeHandle images);

        /// <summary>
        /// Gets the width of an image.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_width", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_image_get_width(
            NativeHandle images,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the height of an image.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_height", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_image_get_height(
            NativeHandle images,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the format string of an image.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_format", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_image_get_format(
            NativeHandle images,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the color space string of an image.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_colorspace", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_image_get_colorspace(
            NativeHandle images,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the bits per component of an image.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_bits_per_component", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_image_get_bits_per_component(
            NativeHandle images,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the raw data of an image. Returns a pointer to the data buffer.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_data", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_image_get_data(
            NativeHandle images,
            int index,
            out int dataLen,
            out int errorCode);

        /// <summary>
        /// Frees an image list handle.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_list_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_image_list_free(IntPtr handle);

        #endregion

        #region Font Accessors (5 functions)

        /// <summary>
        /// Gets the type/subtype of a font.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_font_get_type", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_font_get_type(
            NativeHandle fonts,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the encoding of a font.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_font_get_encoding", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_font_get_encoding(
            NativeHandle fonts,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if a font is embedded. Returns 1 if embedded, 0 if not.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_font_is_embedded", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_font_is_embedded(
            NativeHandle fonts,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if a font is a subset. Returns 1 if subset, 0 if not.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_font_is_subset", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_font_is_subset(
            NativeHandle fonts,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the size of a font (returns 0.0 - size is context-dependent).
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_font_get_size", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float pdf_oxide_font_get_size(
            NativeHandle fonts,
            int index,
            out int errorCode);

        #endregion

        #region Word, Line, and Char Accessors (6 functions)

        /// <summary>
        /// Gets the font name of a word.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_word_get_font_name", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_word_get_font_name(
            NativeHandle words,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the font size of a word.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_word_get_font_size", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float pdf_oxide_word_get_font_size(
            NativeHandle words,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if a word is bold.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_word_is_bold", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_oxide_word_is_bold(
            NativeHandle words,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the word count within a text line.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_line_get_word_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_line_get_word_count(
            NativeHandle lines,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the font name of a character.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_char_get_font_name", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_char_get_font_name(
            NativeHandle chars,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the font size of a character.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_char_get_font_size", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial float pdf_oxide_char_get_font_size(
            NativeHandle chars,
            int index,
            out int errorCode);

        #endregion

        #region Element and Path Accessors (8 functions)

        /// <summary>
        /// Gets the number of elements in an element list.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_element_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_element_count(NativeHandle elements);

        /// <summary>
        /// Gets the type string of an element (e.g. "text").
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_element_get_type", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_element_get_type(
            NativeHandle elements,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the text content of an element.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_element_get_text", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_element_get_text(
            NativeHandle elements,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the bounding rectangle of an element.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_element_get_rect", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_element_get_rect(
            NativeHandle elements,
            int index,
            out float x,
            out float y,
            out float width,
            out float height,
            out int errorCode);

        /// <summary>
        /// Frees an element list handle.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_elements_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_elements_free(IntPtr handle);

        /// <summary>
        /// Gets the number of operations in a path.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_path_get_operation_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_path_get_operation_count(
            NativeHandle paths,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if a path has a fill color.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_path_has_fill", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_oxide_path_has_fill(
            NativeHandle paths,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if a path has a stroke color.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_path_has_stroke", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_oxide_path_has_stroke(
            NativeHandle paths,
            int index,
            out int errorCode);

        #endregion

        #region Other Accessors (7 functions)

        /// <summary>
        /// Checks if a form field is read-only.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_form_field_is_readonly", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_oxide_form_field_is_readonly(
            NativeHandle fields,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if a form field is required.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_form_field_is_required", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_oxide_form_field_is_required(
            NativeHandle fields,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets a quad point from a highlight annotation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_highlight_annotation_get_quad_point", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_highlight_annotation_get_quad_point(
            NativeHandle annotations,
            int index,
            int quadIndex,
            out float x1,
            out float y1,
            out float x2,
            out float y2,
            out float x3,
            out float y3,
            out float x4,
            out float y4,
            out int errorCode);

        /// <summary>
        /// Gets the number of quad points in a highlight annotation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_highlight_annotation_get_quad_points_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_highlight_annotation_get_quad_points_count(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the URI from a link annotation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_link_annotation_get_uri", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_link_annotation_get_uri(
            NativeHandle annotations,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if a table has a header row.
        /// </summary>
        // All other table API P/Invokes take `IntPtr tables` (see
        // `pdf_oxide_table_count`, `pdf_oxide_table_get_row_count`, etc.)
        // because the caller owns the raw handle returned by
        // `pdf_document_extract_tables`. Keeping a second `NativeHandle`
        // overload would be dead code.
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_table_has_header")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_oxide_table_has_header(
            IntPtr tables,
            int index,
            out int errorCode);

        /// <summary>
        /// Gets the icon name of a text annotation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_oxide_text_annotation_get_icon_name", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_text_annotation_get_icon_name(
            NativeHandle annotations,
            int index,
            out int errorCode);

        #endregion

        #region Compliance Extended Operations (5 functions)

        /// <summary>
        /// Gets the number of warnings from PDF/A validation results.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_pdf_a_warning_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_pdf_a_warning_count(NativeHandle results);

        /// <summary>
        /// Gets the number of errors from PDF/UA validation results.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_pdf_ua_error_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_pdf_ua_error_count(NativeHandle results);

        /// <summary>
        /// Gets an error message from PDF/UA validation results by index.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_pdf_ua_get_error", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_pdf_ua_get_error(
            NativeHandle results,
            int index,
            out int errorCode);

        /// <summary>
        /// Checks if the document is accessible according to PDF/UA validation.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_pdf_ua_is_accessible", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool pdf_pdf_ua_is_accessible(
            NativeHandle results,
            out int errorCode);

        /// <summary>
        /// Frees PDF/UA validation results.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_pdf_ua_results_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_pdf_ua_results_free(IntPtr results);

        #endregion

        #region IntPtr-based image list accessors (used by PdfDocument.ExtractImages)

        [LibraryImport(LibName, EntryPoint = "pdf_document_get_embedded_images", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_document_get_embedded_images(
            IntPtr handle,
            int pageIndex,
            out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_count", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_image_count_ptr(IntPtr images);

        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_width", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_image_get_width_ptr(IntPtr images, int index, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_height", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_image_get_height_ptr(IntPtr images, int index, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_bits_per_component", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int pdf_oxide_image_get_bits_per_component_ptr(IntPtr images, int index, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_format", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_image_get_format_ptr(IntPtr images, int index, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_colorspace", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_image_get_colorspace_ptr(IntPtr images, int index, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_get_data", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr pdf_oxide_image_get_data_ptr(IntPtr images, int index, out int dataLen, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_oxide_image_list_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void pdf_oxide_image_list_free_ptr(IntPtr handle);

        #endregion

        #region Write-side API (DocumentBuilder, fonts, HTML+CSS)

        /// <summary>Load a TTF/OTF font from a file path. Returns an opaque handle or <see cref="IntPtr.Zero"/>.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_embedded_font_from_file", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfEmbeddedFontFromFile(string path, out int errorCode);

        /// <summary>Load a TTF/OTF font from raw bytes. <paramref name="name"/> may be null to use the PostScript name.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_embedded_font_from_bytes", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfEmbeddedFontFromBytes([In] byte[] data, nuint len, string? name, out int errorCode);

        /// <summary>Free an EmbeddedFont handle. No-op on null. Do not call after a successful <c>PdfDocumentBuilderRegisterEmbeddedFont</c>.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_embedded_font_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfEmbeddedFontFree(IntPtr handle);

        /// <summary>Create a new DocumentBuilder.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_create", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentBuilderCreate(out int errorCode);

        /// <summary>Free a DocumentBuilder handle. Safe to call after a terminal (<c>_build</c> / <c>_save</c> / etc.) consumed it.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfDocumentBuilderFree(IntPtr handle);

        /// <summary>Set document title.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_set_title", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderSetTitle(IntPtr handle, string title, out int errorCode);

        /// <summary>Set document author.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_set_author", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderSetAuthor(IntPtr handle, string author, out int errorCode);

        /// <summary>Set document subject.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_set_subject", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderSetSubject(IntPtr handle, string subject, out int errorCode);

        /// <summary>Set document keywords.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_set_keywords", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderSetKeywords(IntPtr handle, string keywords, out int errorCode);

        /// <summary>Set the creator application name.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_set_creator", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderSetCreator(IntPtr handle, string creator, out int errorCode);

        /// <summary>Run JavaScript when the document is opened (/OpenAction).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_on_open", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderOnOpen(IntPtr handle, string script, out int errorCode);

        /// <summary>Enable PDF/UA-1 tagged PDF mode (Bundle F-1/F-2).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_tagged_pdf_ua1", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderTaggedPdfUa1(IntPtr handle, out int errorCode);

        /// <summary>Set the document's natural language tag, e.g. "en-US" (Bundle F-2).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_language", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderLanguage(IntPtr handle, string lang, out int errorCode);

        /// <summary>Add a role-map entry: custom structure type → standard PDF type (Bundle F-4).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_role_map", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderRoleMap(IntPtr handle, string custom, string standard, out int errorCode);

        /// <summary>Register a TTF/OTF font. CONSUMES <paramref name="font"/> on success.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_register_embedded_font", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderRegisterEmbeddedFont(IntPtr handle, string name, IntPtr font, out int errorCode);

        /// <summary>Start a new A4 page. Returns a page sub-handle. Only one page may be open per builder.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_a4_page", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentBuilderA4Page(IntPtr handle, out int errorCode);

        /// <summary>Start a new US Letter page.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_letter_page", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentBuilderLetterPage(IntPtr handle, out int errorCode);

        /// <summary>Start a page with custom dimensions in PDF points.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_page", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentBuilderPage(IntPtr handle, float width, float height, out int errorCode);

        // PageBuilder — content ops ---------------------------------------------

        /// <summary>Set the font + size for subsequent text on the page.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_font", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderFont(IntPtr page, string name, float size, out int errorCode);

        /// <summary>Move the cursor to absolute coordinates.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_at", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderAt(IntPtr page, float x, float y, out int errorCode);

        /// <summary>Emit a line of text at the current cursor position.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_text", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderText(IntPtr page, string text, out int errorCode);

        /// <summary>Emit a heading at the current cursor position.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_heading", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderHeading(IntPtr page, byte level, string text, out int errorCode);

        /// <summary>Emit a paragraph with automatic line wrapping.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_paragraph", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderParagraph(IntPtr page, string text, out int errorCode);

        /// <summary>Advance the cursor by the given number of points.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_space", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderSpace(IntPtr page, float points, out int errorCode);

        /// <summary>Draw a horizontal rule across the page.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_horizontal_rule", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderHorizontalRule(IntPtr page, out int errorCode);

        // PageBuilder — annotations (Phase 3) -----------------------------------

        /// <summary>Attach a URL link to the previously-emitted text element.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_link_url", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderLinkUrl(IntPtr page, string url, out int errorCode);

        /// <summary>Link the previous text to an internal page (zero-based).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_link_page", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderLinkPage(IntPtr page, nuint targetPage, out int errorCode);

        /// <summary>Link the previous text to a named destination.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_link_named", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderLinkNamed(IntPtr page, string destination, out int errorCode);

        /// <summary>Link the previous text to a JavaScript action.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_link_javascript", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderLinkJavascript(IntPtr page, string script, out int errorCode);

        /// <summary>Run JavaScript when this page is opened (/AA /O).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_on_open", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderOnOpen(IntPtr page, string script, out int errorCode);

        /// <summary>Run JavaScript when this page is closed (/AA /C).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_on_close", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderOnClose(IntPtr page, string script, out int errorCode);

        /// <summary>Set a keystroke JS action (/AA /K) on the last form field.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_field_keystroke", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderFieldKeystroke(IntPtr page, string script, out int errorCode);

        /// <summary>Set a format JS action (/AA /F) on the last form field.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_field_format", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderFieldFormat(IntPtr page, string script, out int errorCode);

        /// <summary>Set a validate JS action (/AA /V) on the last form field.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_field_validate", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderFieldValidate(IntPtr page, string script, out int errorCode);

        /// <summary>Set a calculate JS action (/AA /C) on the last form field.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_field_calculate", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderFieldCalculate(IntPtr page, string script, out int errorCode);

        /// <summary>Highlight the previous text (RGB channels 0–1).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_highlight", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderHighlight(IntPtr page, float r, float g, float b, out int errorCode);

        /// <summary>Underline the previous text.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_underline", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderUnderline(IntPtr page, float r, float g, float b, out int errorCode);

        /// <summary>Strikeout the previous text.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_strikeout", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderStrikeout(IntPtr page, float r, float g, float b, out int errorCode);

        /// <summary>Squiggly-underline the previous text.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_squiggly", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderSquiggly(IntPtr page, float r, float g, float b, out int errorCode);

        /// <summary>Attach a sticky-note annotation to the previous text.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_sticky_note", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderStickyNote(IntPtr page, string text, out int errorCode);

        /// <summary>Place a sticky note at an absolute position on the page.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_sticky_note_at", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderStickyNoteAt(IntPtr page, float x, float y, string text, out int errorCode);

        /// <summary>Apply a text watermark to the page.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_watermark", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderWatermark(IntPtr page, string text, out int errorCode);

        /// <summary>Apply the standard "CONFIDENTIAL" diagonal watermark.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_watermark_confidential", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderWatermarkConfidential(IntPtr page, out int errorCode);

        /// <summary>Apply the standard "DRAFT" diagonal watermark.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_watermark_draft", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderWatermarkDraft(IntPtr page, out int errorCode);

        /// <summary>Attach a standard stamp annotation at the cursor (150×50 default).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_stamp", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderStamp(IntPtr page, string typeName, out int errorCode);

        /// <summary>Place a free-flowing text annotation inside a rectangle.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_freetext", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderFreetext(IntPtr page, float x, float y, float w, float h, string text, out int errorCode);

        // Form fields

        /// <summary>Add a single-line text form field widget.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_text_field", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderTextField(
            IntPtr page, string name, float x, float y, float w, float h,
            string? defaultValue, out int errorCode);

        /// <summary>Add a checkbox form field widget.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_checkbox", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderCheckbox(
            IntPtr page, string name, float x, float y, float w, float h,
            [MarshalAs(UnmanagedType.I4)] bool checkedState, out int errorCode);

        /// <summary>Add a dropdown combo-box form field.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_combo_box", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderComboBox(
            IntPtr page, string name, float x, float y, float w, float h,
            byte** options, nuint optionsCount, string? selected,
            out int errorCode);

        /// <summary>Add a radio-button group.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_radio_group", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderRadioGroup(
            IntPtr page, string name,
            byte** values, float* xs, float* ys, float* ws, float* hs,
            nuint count, string? selected,
            out int errorCode);

        /// <summary>Add a push-button form field.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_push_button", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderPushButton(
            IntPtr page, string name, float x, float y, float w, float h,
            string caption, out int errorCode);

        /// <summary>Add an unsigned signature placeholder field.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_signature_field", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderSignatureField(
            IntPtr page, string name, float x, float y, float w, float h, out int errorCode);

        /// <summary>Add a footnote reference mark inline and record the body for page-end placement.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_footnote", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderFootnote(
            IntPtr page, string refMark, string noteText, out int errorCode);

        /// <summary>Lay out text as balanced multi-column flow.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_columns", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderColumns(
            IntPtr page, uint columnCount, float gapPt, string text, out int errorCode);

        /// <summary>Inline text run (advances cursorX only).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_inline", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderInline(IntPtr page, string text, out int errorCode);

        /// <summary>Inline bold run.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_inline_bold", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderInlineBold(IntPtr page, string text, out int errorCode);

        /// <summary>Inline italic run.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_inline_italic", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderInlineItalic(IntPtr page, string text, out int errorCode);

        /// <summary>Inline colored run (RGB 0.0–1.0).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_inline_color", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderInlineColor(
            IntPtr page, float r, float g, float b, string text, out int errorCode);

        /// <summary>Advance cursorY one line-height; reset cursorX to 72 pt.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_newline")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderNewline(IntPtr page, out int errorCode);

        /// <summary>Place a 1-D barcode image on the page.
        /// barcodeType: 0=Code128 1=Code39 2=EAN13 3=EAN8 4=UPCA 5=ITF 6=Code93 7=Codabar.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_barcode_1d", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderBarcode1d(
            IntPtr page, int barcodeType, string data,
            float x, float y, float w, float h, out int errorCode);

        /// <summary>Place a QR-code image on the page (square: size × size).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_barcode_qr", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderBarcodeQr(
            IntPtr page, string data, float x, float y, float size, out int errorCode);

        /// <summary>Embed an image at (x, y, w, h) without accessibility wrapper.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_image", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderImage(
            IntPtr page, byte* bytes, nuint len,
            float x, float y, float w, float h, out int errorCode);

        /// <summary>Embed an image with accessibility alt text (PDF/UA-1 Figure).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_image_with_alt", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderImageWithAlt(
            IntPtr page, byte* bytes, nuint len,
            float x, float y, float w, float h,
            string altText, out int errorCode);

        /// <summary>Embed a decorative image as an /Artifact (no alt text, PDF/UA-1).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_image_artifact", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderImageArtifact(
            IntPtr page, byte* bytes, nuint len,
            float x, float y, float w, float h, out int errorCode);

        /// <summary>Draw a stroked rectangle outline.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_rect", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderRect(IntPtr page, float x, float y, float w, float h, out int errorCode);

        /// <summary>Draw a filled rectangle in RGB colour.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_filled_rect", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderFilledRect(IntPtr page, float x, float y, float w, float h, float r, float g, float b, out int errorCode);

        /// <summary>Draw a line from (x1, y1) to (x2, y2).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_line", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderLine(IntPtr page, float x1, float y1, float x2, float y2, out int errorCode);

        // v0.3.39 primitives (#393) — buffered Table surface ------------------

        /// <summary>Draw a stroked rectangle with caller-supplied width + RGB colour.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_stroke_rect", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderStrokeRect(
            IntPtr page, float x, float y, float w, float h,
            float width, float r, float g, float b,
            out int errorCode);

        /// <summary>Draw a straight line with caller-supplied width + RGB colour.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_stroke_line", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderStrokeLine(
            IntPtr page, float x1, float y1, float x2, float y2,
            float width, float r, float g, float b,
            out int errorCode);

        /// <summary>Draw a dashed rectangle border. dashArray is alternating on/off lengths in points.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_stroke_rect_dashed")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderStrokeRectDashed(
            IntPtr page, float x, float y, float w, float h,
            float width, float r, float g, float b,
            float* dashArray, nuint nDash, float phase,
            out int errorCode);

        /// <summary>Draw a dashed line. dashArray is alternating on/off lengths in points.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_stroke_line_dashed")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderStrokeLineDashed(
            IntPtr page, float x1, float y1, float x2, float y2,
            float width, float r, float g, float b,
            float* dashArray, nuint nDash, float phase,
            out int errorCode);

        /// <summary>Place wrapped text inside a rectangle with horizontal alignment (0=Left,1=Center,2=Right).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_text_in_rect", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderTextInRect(
            IntPtr page, float x, float y, float w, float h,
            string text, int align, out int errorCode);

        /// <summary>Transition to a new page with the same dimensions as the current one.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_new_page_same_size", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderNewPageSameSize(IntPtr page, out int errorCode);

        /// <summary>
        /// Place a buffered table at the current cursor. <c>cellStrings</c>
        /// is a row-major array of UTF-8 C strings of length <c>nRows * nColumns</c>.
        /// </summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_table", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderTable(
            IntPtr page,
            nuint nColumns,
            float* widths,
            int* aligns,
            nuint nRows,
            byte** cellStrings,
            int hasHeader,
            out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_streaming_table_begin_v2")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderStreamingTableBeginV2(
            IntPtr page,
            nuint nColumns,
            byte** headers,
            float* widths,
            int* aligns,
            int repeatHeader,
            int mode,
            nuint sampleRows,
            float minColWidthPt,
            float maxColWidthPt,
            nuint maxRowspan,
            out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_streaming_table_push_row")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderStreamingTablePushRow(
            IntPtr page,
            nuint nCells,
            byte** cells,
            out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_streaming_table_push_row_v2")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial int PdfPageBuilderStreamingTablePushRowV2(
            IntPtr page,
            nuint nCells,
            byte** cells,
            nuint* rowspans,
            out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_streaming_table_set_batch_size")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderStreamingTableSetBatchSize(IntPtr page, nuint batchSize, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_streaming_table_pending_row_count")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial nuint PdfPageBuilderStreamingTablePendingRowCount(IntPtr page);

        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_streaming_table_batch_count")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial nuint PdfPageBuilderStreamingTableBatchCount(IntPtr page);

        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_streaming_table_flush")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderStreamingTableFlush(IntPtr page, out int errorCode);

        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_streaming_table_finish")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderStreamingTableFinish(IntPtr page, out int errorCode);

        /// <summary>Commit the page and CONSUME the page handle.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_done", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfPageBuilderDone(IntPtr page, out int errorCode);

        /// <summary>Drop an uncommitted page handle (error-recovery).</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_page_builder_free", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void PdfPageBuilderFree(IntPtr page);

        // DocumentBuilder — finalisation (each CONSUMES the handle) ------------

        /// <summary>Build the PDF and return the bytes. CONSUMES the handle. Free the output with <see cref="FreeBytes"/>.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_build", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentBuilderBuild(IntPtr handle, out nuint outLen, out int errorCode);

        /// <summary>Build and save the PDF to a file. CONSUMES the handle.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_save", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderSave(IntPtr handle, string path, out int errorCode);

        /// <summary>Build and save with AES-256 encryption. CONSUMES the handle.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_save_encrypted", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int PdfDocumentBuilderSaveEncrypted(IntPtr handle, string path, string userPassword, string ownerPassword, out int errorCode);

        /// <summary>Build encrypted bytes. CONSUMES the handle.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_document_builder_to_bytes_encrypted", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfDocumentBuilderToBytesEncrypted(IntPtr handle, string userPassword, string ownerPassword, out nuint outLen, out int errorCode);

        // HTML+CSS pipeline -----------------------------------------------------

        /// <summary>Build a PDF by rendering HTML + CSS with a single embedded font.</summary>
        [LibraryImport(LibName, EntryPoint = "pdf_from_html_css", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr PdfFromHtmlCss(string html, string css, [In] byte[] fontBytes, nuint fontLen, out int errorCode);

        /// <summary>Build a PDF from HTML+CSS with a multi-font cascade. Parallel arrays of length <paramref name="count"/>.</summary>
        // Arrays of pointers cross the boundary as `IntPtr*` rather
        // than `byte**` so the managed-side pinning of `IntPtr[]`
        // doesn't have to reinterpret element type. ABI is identical
        // on every supported platform (`sizeof(IntPtr) ==
        // sizeof(void*)`); the change just lets the caller skip the
        // pointer-type cast.
        [LibraryImport(LibName, EntryPoint = "pdf_from_html_css_with_fonts", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial IntPtr PdfFromHtmlCssWithFonts(
            string html, string css,
            IntPtr* families, IntPtr* fontBytes, nuint* fontLens,
            nuint count, out int errorCode);

        #endregion
    }
}
