using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using PdfOxide.Exceptions;
using PdfOxide.Internal;

namespace PdfOxide.Core
{
    /// <summary>
    /// Represents a PDF document for reading and text extraction.
    /// Provides read-only access with automatic reading order detection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// PdfDocument is the primary API for opening and reading existing PDF files.
    /// It supports:
    /// <list type="bullet">
    /// <item><description>Opening PDF files from disk or memory</description></item>
    /// <item><description>Extracting text with automatic reading order detection</description></item>
    /// <item><description>Converting pages to various formats (Markdown, HTML, PlainText)</description></item>
    /// <item><description>Accessing PDF metadata and structure information</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The document must be explicitly disposed to release native resources.
    /// Use 'using' statements for automatic cleanup.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// using (var doc = PdfDocument.Open("document.pdf"))
    /// {
    ///     // Get PDF version and page count
    ///     var version = doc.Version;
    ///     var pageCount = doc.PageCount;
    ///     Console.WriteLine($"PDF {version.Major}.{version.Minor}, {pageCount} pages");
    ///
    ///     // Extract text from first page
    ///     var text = doc.ExtractText(0);
    ///     Console.WriteLine(text);
    ///
    ///     // Convert to Markdown
    ///     var markdown = doc.ToMarkdown(0);
    ///     File.WriteAllText("output.md", markdown);
    /// }
    /// </code>
    /// </example>
    public sealed class PdfDocument : IDisposable
    {
        private NativeHandle _handle;
        private volatile bool _disposed;
        // SupportsRecursion: read methods may call other read-locked helpers (e.g. ExtractText
        // calling PageCount for bounds-checking). Read-read recursion is safe and non-deadlocking.
        private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim(LockRecursionPolicy.SupportsRecursion);

        private PdfDocument(NativeHandle handle)
        {
            _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        }

        /// <summary>
        /// Gets the native handle pointer for internal use by managers.
        /// Thread-safe: acquires a read lock.
        /// </summary>
        internal IntPtr Handle
        {
            get
            {
                _lock.EnterReadLock();
                try
                {
                    ThrowIfDisposed();
                    return _handle.Ptr;
                }
                finally
                {
                    _lock.ExitReadLock();
                }
            }
        }

        /// <summary>
        /// Gets the native handle for internal use by managers.
        /// Thread-safe: acquires a read lock.
        /// </summary>
        internal NativeHandle NativeHandle
        {
            get
            {
                _lock.EnterReadLock();
                try
                {
                    ThrowIfDisposed();
                    return _handle;
                }
                finally
                {
                    _lock.ExitReadLock();
                }
            }
        }

        /// <summary>
        /// Opens a PDF document from a file path.
        /// </summary>
        /// <param name="path">The file path to the PDF.</param>
        /// <returns>An opened PdfDocument.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="path"/> is null.</exception>
        /// <exception cref="PdfIoException">Thrown if the file cannot be opened.</exception>
        /// <exception cref="PdfParseException">Thrown if the PDF is invalid.</exception>
        public static PdfDocument Open(string path)
        {
            ArgumentNullException.ThrowIfNull(path);

            var handle = NativeMethods.PdfDocumentOpen(path, out var errorCode);
            if (handle.IsInvalid)
            {
                ExceptionMapper.ThrowIfError(errorCode);
            }

            return new PdfDocument(handle);
        }

        /// <summary>
        /// Opens a PDF document from a stream.
        /// </summary>
        /// <remarks>
        /// If the PDF is already in memory as a <see cref="byte"/>[] or
        /// <see cref="ReadOnlySpan{T}"/>, prefer the dedicated overloads —
        /// they skip the intermediate <see cref="MemoryStream"/> copy this
        /// overload has to make to produce a contiguous buffer for the FFI.
        /// </remarks>
        /// <param name="stream">The stream containing PDF data.</param>
        /// <returns>An opened PdfDocument.</returns>
        public static PdfDocument Open(Stream stream)
        {
            ArgumentNullException.ThrowIfNull(stream);

            byte[] data;
            using (var ms = new MemoryStream())
            {
                stream.CopyTo(ms);
                data = ms.ToArray();
            }

            return Open(data);
        }

        /// <summary>
        /// Opens a PDF document from a byte array.
        /// </summary>
        /// <remarks>
        /// Forwards <paramref name="data"/> directly to the FFI without the
        /// <see cref="MemoryStream"/> copy the <see cref="Open(Stream)"/>
        /// overload has to make.
        /// </remarks>
        /// <param name="data">The PDF bytes. Must be non-null and non-empty.</param>
        /// <returns>An opened PdfDocument.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="data"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown if <paramref name="data"/> is empty.</exception>
        public static PdfDocument Open(byte[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            if (data.Length == 0)
                throw new ArgumentException("PDF byte array must not be empty.", nameof(data));

            var handle = NativeMethods.PdfDocumentOpenFromBytes(data, data.Length, out var errorCode);
            if (handle.IsInvalid)
            {
                ExceptionMapper.ThrowIfError(errorCode);
            }
            return new PdfDocument(handle);
        }

        /// <summary>
        /// Opens a PDF document from a <see cref="ReadOnlySpan{T}"/> of bytes.
        /// </summary>
        /// <remarks>
        /// Zero-copy entry point: LibraryImport pins the span while the FFI
        /// call is in flight, so no managed-array allocation or
        /// <see cref="MemoryStream"/> hop is involved. Use when the PDF is
        /// already materialised in an un-pinned buffer you don't want to
        /// duplicate.
        /// </remarks>
        /// <param name="data">The PDF bytes. Must be non-empty.</param>
        /// <returns>An opened PdfDocument.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="data"/> is empty.</exception>
        public static PdfDocument Open(ReadOnlySpan<byte> data)
        {
            if (data.IsEmpty)
                throw new ArgumentException("PDF byte span must not be empty.", nameof(data));

            var handle = NativeMethods.PdfDocumentOpenFromBytesRef(
                ref System.Runtime.InteropServices.MemoryMarshal.GetReference(data),
                data.Length,
                out var errorCode);
            if (handle.IsInvalid)
            {
                ExceptionMapper.ThrowIfError(errorCode);
            }
            return new PdfDocument(handle);
        }

        /// <summary>
        /// Gets the PDF version as a tuple of (major, minor).
        /// </summary>
        /// <value>A tuple containing the major and minor version numbers.</value>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        public (byte Major, byte Minor) Version
        {
            get
            {
                _lock.EnterReadLock();
                try
                {
                    ThrowIfDisposed();
                    NativeMethods.PdfDocumentGetVersion(_handle, out var major, out var minor);
                    return (major, minor);
                }
                finally { _lock.ExitReadLock(); }
            }
        }

        /// <summary>
        /// Gets the number of pages in the document.
        /// </summary>
        /// <value>The page count.</value>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        /// <exception cref="PdfException">Thrown if page count cannot be determined.</exception>
        public int PageCount
        {
            get
            {
                _lock.EnterReadLock();
                try
                {
                    ThrowIfDisposed();
                    var count = NativeMethods.PdfDocumentGetPageCount(_handle, out var errorCode);
                    ExceptionMapper.ThrowIfError(errorCode);
                    return count;
                }
                finally { _lock.ExitReadLock(); }
            }
        }

        /// <summary>
        /// Gets the number of existing digital signatures in this document.
        /// Returns 0 for documents without an AcroForm or without
        /// signed signature fields. Surfaces the Rust
        /// <c>enumerate_signatures</c> walker.
        /// </summary>
        public int SignatureCount
        {
            get
            {
                _lock.EnterReadLock();
                try
                {
                    ThrowIfDisposed();
                    var count = NativeMethods.pdf_document_get_signature_count(_handle, out int err);
                    ExceptionMapper.ThrowIfError(err);
                    return count;
                }
                finally { _lock.ExitReadLock(); }
            }
        }

        /// <summary>
        /// Enumerate every existing digital signature in the document.
        /// Each <see cref="Signature"/> must be disposed by the caller;
        /// the returned list is a snapshot, not live-linked to the
        /// underlying PDF state.
        /// </summary>
        public System.Collections.Generic.IReadOnlyList<Signature> Signatures
        {
            get
            {
                _lock.EnterReadLock();
                try
                {
                    ThrowIfDisposed();
                    var count = NativeMethods.pdf_document_get_signature_count(_handle, out int err);
                    ExceptionMapper.ThrowIfError(err);
                    var list = new System.Collections.Generic.List<Signature>(count);
                    try
                    {
                        for (int i = 0; i < count; i++)
                        {
                            var sigHandle = NativeMethods.pdf_document_get_signature(_handle, i, out int e);
                            ExceptionMapper.ThrowIfError(e);
                            if (sigHandle.IsInvalid)
                            {
                                throw new PdfException(
                                    $"pdf_document_get_signature({i}) returned null with no error code");
                            }
                            list.Add(Signature.FromHandle(sigHandle));
                        }
                        return list;
                    }
                    catch
                    {
                        foreach (var s in list) s.Dispose();
                        throw;
                    }
                }
                finally { _lock.ExitReadLock(); }
            }
        }

        /// <summary>
        /// Gets a value indicating whether the document has a structure tree (Tagged PDF).
        /// </summary>
        /// <value>True if the document has a structure tree, false otherwise.</value>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        public bool HasStructureTree
        {
            get
            {
                _lock.EnterReadLock();
                try
                {
                    ThrowIfDisposed();
                    return NativeMethods.PdfDocumentHasStructureTree(_handle);
                }
                finally { _lock.ExitReadLock(); }
            }
        }

        /// <summary>
        /// Extracts text from a page with automatic reading order detection.
        /// </summary>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <returns>
        /// The extracted text as a managed <see cref="string"/>. The native
        /// layer returns UTF-8, which is decoded to .NET's native UTF-16 here,
        /// so <see cref="string.Length"/> reports UTF-16 code units, not
        /// bytes. Use <c>System.Text.Encoding.UTF8.GetByteCount(text)</c> if
        /// you need the byte count (e.g. to compare against Go's
        /// <c>len(string)</c> or Rust's <c>String::len()</c>).
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="pageIndex"/> is out of range.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        /// <exception cref="PdfException">Thrown if text extraction fails.</exception>
        public string ExtractText(int pageIndex)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                if (pageIndex < 0 || pageIndex >= PageCount)
                    throw new ArgumentOutOfRangeException(nameof(pageIndex));
                var ptr = NativeMethods.PdfDocumentExtractText(_handle, pageIndex, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>
        /// Asynchronously extracts text from a page.
        /// </summary>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>A task that yields the extracted text.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="pageIndex"/> is out of range.</exception>
        /// <exception cref="OperationCanceledException">Thrown if the operation is cancelled.</exception>
        public Task<string> ExtractTextAsync(int pageIndex, CancellationToken cancellationToken = default)
        {
            return Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return ExtractText(pageIndex);
            }, cancellationToken);
        }

        /// <summary>
        /// Converts a page to Markdown format.
        /// </summary>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <returns>The page content as Markdown.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="pageIndex"/> is out of range.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        /// <exception cref="PdfException">Thrown if conversion fails.</exception>
        public string ToMarkdown(int pageIndex)
        {
            // WriteLock because the Rust to_markdown lazily initialises structure-tree caches
            // via a double-checked pattern that is not safe under concurrent &self calls.
            _lock.EnterWriteLock();
            try
            {
                ThrowIfDisposed();
                if (pageIndex < 0 || pageIndex >= PageCount)
                    throw new ArgumentOutOfRangeException(nameof(pageIndex));
                var ptr = NativeMethods.PdfDocumentToMarkdown(_handle, pageIndex, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>
        /// Converts all pages to Markdown format.
        /// </summary>
        /// <returns>The document content as Markdown.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        /// <exception cref="PdfException">Thrown if conversion fails.</exception>
        public string ToMarkdownAll()
        {
            // WriteLock: same reason as ToMarkdown — lazy structure-tree cache is not concurrent-read-safe.
            _lock.EnterWriteLock();
            try
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.PdfDocumentToMarkdownAll(_handle, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>
        /// Converts a page to HTML format.
        /// </summary>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <returns>The page content as HTML.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="pageIndex"/> is out of range.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        /// <exception cref="PdfException">Thrown if conversion fails.</exception>
        public string ToHtml(int pageIndex)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                if (pageIndex < 0 || pageIndex >= PageCount)
                    throw new ArgumentOutOfRangeException(nameof(pageIndex));
                var ptr = NativeMethods.PdfDocumentToHtml(_handle, pageIndex, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>
        /// Converts a page to plain text format.
        /// </summary>
        /// <param name="pageIndex">The page index (0-based).</param>
        /// <returns>The page content as plain text.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="pageIndex"/> is out of range.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        /// <exception cref="PdfException">Thrown if conversion fails.</exception>
        public string ToPlainText(int pageIndex)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                if (pageIndex < 0 || pageIndex >= PageCount)
                    throw new ArgumentOutOfRangeException(nameof(pageIndex));
                var ptr = NativeMethods.PdfDocumentToPlainText(_handle, pageIndex, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitReadLock(); }
        }

        // ================================================================
        // v0.3.23 New Methods
        // ================================================================

        /// <summary>Extracts text from all pages.</summary>
        public string ExtractAllText()
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.pdf_document_extract_all_text(_handle.Ptr, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Converts all pages to HTML.</summary>
        public string ToHtmlAll()
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.pdf_document_to_html_all(_handle.Ptr, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Converts all pages to plain text.</summary>
        public string ToPlainTextAll()
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.pdf_document_to_plain_text_all(_handle.Ptr, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Checks if the document is encrypted.</summary>
        public bool IsEncrypted
        {
            get
            {
                _lock.EnterReadLock();
                try
                {
                    ThrowIfDisposed();
                    return NativeMethods.pdf_document_is_encrypted(_handle.Ptr);
                }
                finally { _lock.ExitReadLock(); }
            }
        }

        /// <summary>Authenticates with a password. Returns true if successful.</summary>
        public bool Authenticate(string password)
        {
            ArgumentNullException.ThrowIfNull(password);
            _lock.EnterWriteLock();
            try
            {
                ThrowIfDisposed();
                return NativeMethods.pdf_document_authenticate(_handle.Ptr, password, out _);
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Checks if the document has XFA forms.</summary>
        public bool HasXfa
        {
            get
            {
                _lock.EnterReadLock();
                try
                {
                    ThrowIfDisposed();
                    return NativeMethods.pdf_document_has_xfa(_handle.Ptr);
                }
                finally { _lock.ExitReadLock(); }
            }
        }

        /// <summary>Extracts text from a rectangular region on a page.</summary>
        public string ExtractTextInRect(int pageIndex, float x, float y, float width, float height)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.pdf_document_extract_text_in_rect(_handle.Ptr, pageIndex, x, y, width, height, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Removes repeated headers across pages. Returns count removed.</summary>
        public int RemoveHeaders(float threshold = 0.8f)
        {
            _lock.EnterWriteLock();
            try
            {
                ThrowIfDisposed();
                var n = NativeMethods.pdf_document_remove_headers(_handle.Ptr, threshold, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return n;
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Removes repeated footers across pages. Returns count removed.</summary>
        public int RemoveFooters(float threshold = 0.8f)
        {
            _lock.EnterWriteLock();
            try
            {
                ThrowIfDisposed();
                var n = NativeMethods.pdf_document_remove_footers(_handle.Ptr, threshold, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return n;
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Removes headers and footers. Returns count removed.</summary>
        public int RemoveArtifacts(float threshold = 0.8f)
        {
            _lock.EnterWriteLock();
            try
            {
                ThrowIfDisposed();
                var n = NativeMethods.pdf_document_remove_artifacts(_handle.Ptr, threshold, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return n;
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Opens a PDF with password.</summary>
        public static PdfDocument OpenWithPassword(string path, string password)
        {
            var doc = Open(path);
            doc.Authenticate(password);  // Authenticate now holds the write lock
            return doc;
        }

        /// <summary>Extracts words from a page. Returns handle-based results (use NativeMethods directly for now).</summary>
        public (string Text, float X, float Y, float W, float H)[] ExtractWords(int pageIndex)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var handle = NativeMethods.pdf_document_extract_words(_handle.Ptr, pageIndex, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (handle == IntPtr.Zero) return Array.Empty<(string, float, float, float, float)>();
                try
                {
                    var count = NativeMethods.pdf_oxide_word_count(handle);
                    var results = new (string, float, float, float, float)[count];
                    for (int i = 0; i < count; i++)
                    {
                        var textPtr = NativeMethods.pdf_oxide_word_get_text(handle, i, out _);
                        var text = StringMarshaler.PtrToStringAndFree(textPtr);
                        NativeMethods.pdf_oxide_word_get_bbox(handle, i, out var x, out var y, out var w, out var h, out _);
                        results[i] = (text, x, y, w, h);
                    }
                    return results;
                }
                finally { NativeMethods.pdf_oxide_word_list_free(handle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Extracts text lines from a page.</summary>
        public (string Text, float X, float Y, float W, float H)[] ExtractTextLines(int pageIndex)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var handle = NativeMethods.pdf_document_extract_text_lines(_handle.Ptr, pageIndex, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (handle == IntPtr.Zero) return Array.Empty<(string, float, float, float, float)>();
                try
                {
                    var count = NativeMethods.pdf_oxide_line_count(handle);
                    var results = new (string, float, float, float, float)[count];
                    for (int i = 0; i < count; i++)
                    {
                        var textPtr = NativeMethods.pdf_oxide_line_get_text(handle, i, out _);
                        var text = StringMarshaler.PtrToStringAndFree(textPtr);
                        NativeMethods.pdf_oxide_line_get_bbox(handle, i, out var x, out var y, out var w, out var h, out _);
                        results[i] = (text, x, y, w, h);
                    }
                    return results;
                }
                finally { NativeMethods.pdf_oxide_line_list_free(handle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Extracts tables from a page. Returns row/col counts per table.</summary>
        public Table[] ExtractTables(int pageIndex)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var handle = NativeMethods.pdf_document_extract_tables(_handle.Ptr, pageIndex, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (handle == IntPtr.Zero) return Array.Empty<Table>();
                try
                {
                    var count = NativeMethods.pdf_oxide_table_count(handle);
                    var results = new Table[count];
                    for (int i = 0; i < count; i++)
                    {
                        var rows = NativeMethods.pdf_oxide_table_get_row_count(handle, i, out _);
                        var cols = NativeMethods.pdf_oxide_table_get_col_count(handle, i, out _);
                        var hasHeader = NativeMethods.pdf_oxide_table_has_header(handle, i, out _);
                        var cells = new string[rows, cols];
                        for (int r = 0; r < rows; r++)
                            for (int c = 0; c < cols; c++)
                            {
                                var ptr = NativeMethods.pdf_oxide_table_get_cell_text(handle, i, r, c, out _);
                                if (ptr != IntPtr.Zero)
                                {
                                    cells[r, c] = System.Runtime.InteropServices.Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
                                    NativeMethods.FreeString(ptr);
                                }
                                else cells[r, c] = string.Empty;
                            }
                        results[i] = new Table(rows, cols, hasHeader, cells);
                    }
                    return results;
                }
                finally { NativeMethods.pdf_oxide_table_list_free(handle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Searches all pages for text. Returns results with page index and bounding box.</summary>
        public (int Page, string Text, float X, float Y, float W, float H)[] SearchAll(string text, bool caseSensitive = false)
        {
            ArgumentNullException.ThrowIfNull(text);
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var resultsHandle = NativeMethods.pdf_document_search_all(_handle.Ptr, text, caseSensitive, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (resultsHandle == IntPtr.Zero) return Array.Empty<(int, string, float, float, float, float)>();
                try
                {
                    return DecodeSearchResults(resultsHandle);
                }
                finally { NativeMethods.pdf_oxide_search_result_free(resultsHandle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Searches a specific page for text.</summary>
        public (int Page, string Text, float X, float Y, float W, float H)[] SearchPage(int pageIndex, string text, bool caseSensitive = false)
        {
            ArgumentNullException.ThrowIfNull(text);
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var resultsHandle = NativeMethods.pdf_document_search_page(_handle.Ptr, pageIndex, text, caseSensitive, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (resultsHandle == IntPtr.Zero) return Array.Empty<(int, string, float, float, float, float)>();
                try
                {
                    return DecodeSearchResults(resultsHandle);
                }
                finally { NativeMethods.pdf_oxide_search_result_free(resultsHandle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        // One FFI crossing → Rust serializes the entire result list to JSON →
        // System.Text.Json decodes it. Matches the Go binding pattern and is
        // O(1) FFI calls instead of O(count × 4) per-field calls.
        private static (int Page, string Text, float X, float Y, float W, float H)[] DecodeSearchResults(IntPtr handle)
        {
            var jsonPtr = NativeMethods.PdfOxideSearchResultsToJson(handle, out var jsonErr);
            ExceptionMapper.ThrowIfError(jsonErr);
            if (jsonPtr == IntPtr.Zero) return Array.Empty<(int, string, float, float, float, float)>();

            string json;
            try
            {
                json = StringMarshaler.PtrToString(jsonPtr);
            }
            finally
            {
                NativeMethods.FreeString(jsonPtr);
            }

            using var doc = System.Text.Json.JsonDocument.Parse(json);
            var arr = doc.RootElement;
            var results = new (int, string, float, float, float, float)[arr.GetArrayLength()];
            int idx = 0;
            foreach (var el in arr.EnumerateArray())
            {
                results[idx++] = (
                    el.GetProperty("page").GetInt32(),
                    el.GetProperty("text").GetString() ?? string.Empty,
                    el.GetProperty("x").GetSingle(),
                    el.GetProperty("y").GetSingle(),
                    el.GetProperty("width").GetSingle(),
                    el.GetProperty("height").GetSingle());
            }
            return results;
        }

        /// <summary>Gets page labels as JSON.</summary>
        public string GetPageLabels()
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.pdf_document_get_page_labels(_handle.Ptr, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Gets XMP metadata as JSON.</summary>
        public string GetXmpMetadata()
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.pdf_document_get_xmp_metadata(_handle.Ptr, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Gets document outline/bookmarks as JSON.</summary>
        public string GetOutline()
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.pdf_document_get_outline(_handle.Ptr, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return StringMarshaler.PtrToStringAndFree(ptr);
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Extracts individual characters from a page.</summary>
        public (char Char, float X, float Y, float W, float H)[] ExtractChars(int pageIndex)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var handle = NativeMethods.pdf_document_extract_chars(_handle.Ptr, pageIndex, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (handle == IntPtr.Zero) return Array.Empty<(char, float, float, float, float)>();
                try
                {
                    var count = NativeMethods.pdf_oxide_char_count(handle);
                    var results = new (char, float, float, float, float)[count];
                    for (int i = 0; i < count; i++)
                    {
                        var ch = NativeMethods.pdf_oxide_char_get_char(handle, i, out _);
                        NativeMethods.pdf_oxide_char_get_bbox(handle, i, out var x, out var y, out var w, out var h, out _);
                        results[i] = ((char)ch, x, y, w, h);
                    }
                    return results;
                }
                finally { NativeMethods.pdf_oxide_char_list_free(handle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Extracts paths from a page.</summary>
        public (float X, float Y, float W, float H, float StrokeWidth)[] ExtractPaths(int pageIndex)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var handle = NativeMethods.pdf_document_extract_paths(_handle.Ptr, pageIndex, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (handle == IntPtr.Zero) return Array.Empty<(float, float, float, float, float)>();
                try
                {
                    var count = NativeMethods.pdf_oxide_path_count(handle);
                    var results = new (float, float, float, float, float)[count];
                    for (int i = 0; i < count; i++)
                    {
                        NativeMethods.pdf_oxide_path_get_bbox(handle, i, out var x, out var y, out var w, out var h, out _);
                        var sw = NativeMethods.pdf_oxide_path_get_stroke_width(handle, i, out _);
                        results[i] = (x, y, w, h, sw);
                    }
                    return results;
                }
                finally { NativeMethods.pdf_oxide_path_list_free(handle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Extracts words from a rectangular region.</summary>
        public (string Text, float X, float Y, float W, float H)[] ExtractWordsInRect(int pageIndex, float x, float y, float width, float height)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var handle = NativeMethods.pdf_document_extract_words_in_rect(_handle.Ptr, pageIndex, x, y, width, height, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (handle == IntPtr.Zero) return Array.Empty<(string, float, float, float, float)>();
                try
                {
                    var count = NativeMethods.pdf_oxide_word_count(handle);
                    var results = new (string, float, float, float, float)[count];
                    for (int i = 0; i < count; i++)
                    {
                        var textPtr = NativeMethods.pdf_oxide_word_get_text(handle, i, out _);
                        var text = StringMarshaler.PtrToStringAndFree(textPtr);
                        NativeMethods.pdf_oxide_word_get_bbox(handle, i, out var wx, out var wy, out var ww, out var wh, out _);
                        results[i] = (text, wx, wy, ww, wh);
                    }
                    return results;
                }
                finally { NativeMethods.pdf_oxide_word_list_free(handle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Gets font names from a page.</summary>
        public string[] GetFonts(int pageIndex)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var handle = NativeMethods.pdf_document_get_embedded_fonts(_handle.Ptr, pageIndex, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (handle == IntPtr.Zero) return Array.Empty<string>();
                try
                {
                    var count = NativeMethods.pdf_oxide_font_count(handle);
                    var results = new string[count];
                    for (int i = 0; i < count; i++)
                    {
                        var namePtr = NativeMethods.pdf_oxide_font_get_name(handle, i, out _);
                        results[i] = StringMarshaler.PtrToStringAndFree(namePtr);
                    }
                    return results;
                }
                finally { NativeMethods.pdf_oxide_font_list_free(handle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Renders a page to PNG bytes. format: 0=PNG, 1=JPEG.</summary>
        public byte[] RenderPage(int pageIndex, int format = 0)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var imgHandle = NativeMethods.pdf_render_page(_handle.Ptr, pageIndex, format, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (imgHandle == IntPtr.Zero) return Array.Empty<byte>();
                try
                {
                    var data = NativeMethods.pdf_get_rendered_image_data(imgHandle, out var dataLen, out _);
                    if (data == IntPtr.Zero) return Array.Empty<byte>();
                    var bytes = new byte[dataLen];
                    System.Runtime.InteropServices.Marshal.Copy(data, bytes, 0, dataLen);
                    NativeMethods.FreeBytes(data);
                    return bytes;
                }
                finally { NativeMethods.pdf_rendered_image_free(imgHandle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>
        /// Renders a page with the full <see cref="RenderOptions"/> surface:
        /// DPI, output format, background colour or transparency,
        /// annotation toggle, and JPEG quality. The simpler
        /// <see cref="RenderPage(int, int)"/> overload only exposes the
        /// format knob.
        /// </summary>
        /// <param name="pageIndex">Page index, 0-based.</param>
        /// <param name="options">Render options; see <see cref="RenderOptions"/>.</param>
        /// <exception cref="ArgumentNullException">If <paramref name="options"/> is null.</exception>
        public byte[] RenderPage(int pageIndex, RenderOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            options.Validate();
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var imgHandle = NativeMethods.PdfRenderPageWithOptions(
                    _handle.Ptr,
                    pageIndex,
                    options.Dpi,
                    (int)options.Format,
                    options.Background.R,
                    options.Background.G,
                    options.Background.B,
                    options.Background.A,
                    options.TransparentBackground ? 1 : 0,
                    options.RenderAnnotations ? 1 : 0,
                    options.JpegQuality,
                    out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (imgHandle == IntPtr.Zero) return Array.Empty<byte>();
                try
                {
                    var data = NativeMethods.pdf_get_rendered_image_data(imgHandle, out var dataLen, out _);
                    if (data == IntPtr.Zero) return Array.Empty<byte>();
                    var bytes = new byte[dataLen];
                    System.Runtime.InteropServices.Marshal.Copy(data, bytes, 0, dataLen);
                    NativeMethods.FreeBytes(data);
                    return bytes;
                }
                finally { NativeMethods.pdf_rendered_image_free(imgHandle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Renders a page with zoom factor. Returns PNG bytes.</summary>
        public byte[] RenderPageZoom(int pageIndex, float zoom, int format = 0)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var imgHandle = NativeMethods.pdf_render_page_zoom(_handle.Ptr, pageIndex, zoom, format, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (imgHandle == IntPtr.Zero) return Array.Empty<byte>();
                try
                {
                    var data = NativeMethods.pdf_get_rendered_image_data(imgHandle, out var dataLen, out _);
                    if (data == IntPtr.Zero) return Array.Empty<byte>();
                    var bytes = new byte[dataLen];
                    System.Runtime.InteropServices.Marshal.Copy(data, bytes, 0, dataLen);
                    NativeMethods.FreeBytes(data);
                    return bytes;
                }
                finally { NativeMethods.pdf_rendered_image_free(imgHandle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>
        /// Renders a page to fit inside a <paramref name="fitWidth"/> × <paramref name="fitHeight"/>
        /// pixel box, preserving aspect ratio. Picks the largest DPI such that
        /// both rendered dimensions are ≤ the target box, so the output may be
        /// smaller than the requested box on one axis. Issue #448.
        /// </summary>
        /// <param name="pageIndex">Zero-based page index.</param>
        /// <param name="fitWidth">Target box width in pixels (must be &gt; 0).</param>
        /// <param name="fitHeight">Target box height in pixels (must be &gt; 0).</param>
        /// <param name="format">0 = PNG (default), 1 = JPEG.</param>
        public byte[] RenderPageFit(int pageIndex, int fitWidth, int fitHeight, int format = 0)
        {
            if (fitWidth <= 0)
                throw new ArgumentOutOfRangeException(nameof(fitWidth), fitWidth, $"fitWidth must be > 0, got {fitWidth}");
            if (fitHeight <= 0)
                throw new ArgumentOutOfRangeException(nameof(fitHeight), fitHeight, $"fitHeight must be > 0, got {fitHeight}");
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var imgHandle = NativeMethods.pdf_render_page_fit(_handle.Ptr, pageIndex, fitWidth, fitHeight, format, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (imgHandle == IntPtr.Zero) return Array.Empty<byte>();
                try
                {
                    var data = NativeMethods.pdf_get_rendered_image_data(imgHandle, out var dataLen, out _);
                    if (data == IntPtr.Zero) return Array.Empty<byte>();
                    var bytes = new byte[dataLen];
                    System.Runtime.InteropServices.Marshal.Copy(data, bytes, 0, dataLen);
                    NativeMethods.FreeBytes(data);
                    return bytes;
                }
                finally { NativeMethods.pdf_rendered_image_free(imgHandle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Renders a page thumbnail (72 DPI). Returns PNG bytes.</summary>
        public byte[] RenderThumbnail(int pageIndex, int format = 0)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var imgHandle = NativeMethods.pdf_render_page_thumbnail(_handle.Ptr, pageIndex, 72, format, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (imgHandle == IntPtr.Zero) return Array.Empty<byte>();
                try
                {
                    var data = NativeMethods.pdf_get_rendered_image_data(imgHandle, out var dataLen, out _);
                    if (data == IntPtr.Zero) return Array.Empty<byte>();
                    var bytes = new byte[dataLen];
                    System.Runtime.InteropServices.Marshal.Copy(data, bytes, 0, dataLen);
                    NativeMethods.FreeBytes(data);
                    return bytes;
                }
                finally { NativeMethods.pdf_rendered_image_free(imgHandle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>
        /// Renders a page as premultiplied RGBA8888 pixels without encoding to PNG or JPEG.
        /// Useful when passing pixel data directly to image-processing libraries (e.g. System.Drawing,
        /// ImageSharp, SkiaSharp). Data layout: row-major, top-left origin, 4 bytes per pixel,
        /// width × height × 4 bytes total. Alpha is premultiplied (PDF spec §11 transparency model).
        /// </summary>
        /// <param name="pageIndex">Zero-based page index.</param>
        /// <param name="dpi">Resolution in dots per inch (default 150).</param>
        /// <returns><see cref="RgbaPixmap"/> with the raw pixel bytes and dimensions.</returns>
        /// <exception cref="UnsupportedFeatureException">Native lib compiled without rendering feature.</exception>
        public RgbaPixmap RenderToRgba(int pageIndex, int dpi = 150)
        {
            if (dpi <= 0)
                throw new ArgumentOutOfRangeException(nameof(dpi), dpi, "dpi must be > 0");
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var imgHandle = NativeMethods.pdf_render_page_raw(
                    _handle.Ptr, pageIndex, dpi, out int w, out int h, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (imgHandle == IntPtr.Zero) return new RgbaPixmap(ReadOnlyMemory<byte>.Empty, 0, 0);
                try
                {
                    var data = NativeMethods.pdf_get_rendered_image_data(imgHandle, out int dataLen, out _);
                    if (data == IntPtr.Zero) return new RgbaPixmap(ReadOnlyMemory<byte>.Empty, w, h);
                    var bytes = new byte[dataLen];
                    System.Runtime.InteropServices.Marshal.Copy(data, bytes, 0, dataLen);
                    NativeMethods.FreeBytes(data);
                    return new RgbaPixmap(bytes, w, h);
                }
                finally { NativeMethods.pdf_rendered_image_free(imgHandle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Saves a rendered page to a file.</summary>
        public void SaveRenderedImage(int pageIndex, string filePath, int format = 0)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var imgHandle = NativeMethods.pdf_render_page(_handle.Ptr, pageIndex, format, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (imgHandle == IntPtr.Zero) return;
                try { NativeMethods.pdf_save_rendered_image(imgHandle, filePath, out _); }
                finally { NativeMethods.pdf_rendered_image_free(imgHandle); }
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>
        /// Disposes the document and releases native resources.
        /// Thread-safe: acquires a write lock to prevent concurrent access during disposal.
        /// </summary>
        public void Dispose()
        {
            _lock.EnterWriteLock();
            try
            {
                if (!_disposed)
                {
                    _handle?.Dispose();
                    _disposed = true;
                }
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Extracts embedded images from a page. Returns an empty array when the page has no images.
        /// </summary>
        /// <param name="pageIndex">Zero-based page index.</param>
        /// <returns>Array of embedded images with their pixel data and metadata.</returns>
        /// <exception cref="PdfException">Thrown if the native call fails.</exception>
        public ExtractedImage[] ExtractImages(int pageIndex)
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var list = NativeMethods.pdf_document_get_embedded_images(_handle.Ptr, pageIndex, out int err);
                if (err != 0)
                    throw new PdfException($"Failed to extract images: {PdfException.GetErrorMessage(err)}");
                if (list == IntPtr.Zero)
                    return Array.Empty<ExtractedImage>();
                try
                {
                    int count = NativeMethods.pdf_oxide_image_count_ptr(list);
                    var images = new ExtractedImage[count];
                    for (int i = 0; i < count; i++)
                    {
                        int w = NativeMethods.pdf_oxide_image_get_width_ptr(list, i, out int e1);
                        int h = NativeMethods.pdf_oxide_image_get_height_ptr(list, i, out int e2);
                        int bpc = NativeMethods.pdf_oxide_image_get_bits_per_component_ptr(list, i, out int e3);
                        string format = PtrToStringAndFree(NativeMethods.pdf_oxide_image_get_format_ptr(list, i, out int e4));
                        string colorspace = PtrToStringAndFree(NativeMethods.pdf_oxide_image_get_colorspace_ptr(list, i, out int e5));
                        var dataPtr = NativeMethods.pdf_oxide_image_get_data_ptr(list, i, out int dataLen, out int e6);
                        byte[] data = dataPtr != IntPtr.Zero && dataLen > 0
                            ? new byte[dataLen]
                            : Array.Empty<byte>();
                        if (dataPtr != IntPtr.Zero && dataLen > 0)
                        {
                            System.Runtime.InteropServices.Marshal.Copy(dataPtr, data, 0, dataLen);
                        }
                        images[i] = new ExtractedImage(w, h, format, colorspace, bpc, data);
                    }
                    return images;
                }
                finally
                {
                    NativeMethods.pdf_oxide_image_list_free_ptr(list);
                }
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        /// <summary>
        /// Reads all form (AcroForm) fields from the document.
        /// </summary>
        /// <returns>Array of form fields. Empty if the document has no form fields.</returns>
        /// <exception cref="PdfException">Thrown if the native call fails.</exception>
        public FormField[] GetFormFields()
        {
            _lock.EnterReadLock();
            try
            {
                ThrowIfDisposed();
                var list = NativeMethods.pdf_document_get_form_fields(_handle.Ptr, out int err);
                if (err != 0)
                    throw new PdfException($"Failed to get form fields: {PdfException.GetErrorMessage(err)}");
                if (list == IntPtr.Zero)
                    return Array.Empty<FormField>();
                try
                {
                    int count = NativeMethods.pdf_oxide_form_field_count(list);
                    var fields = new FormField[count];
                    for (int i = 0; i < count; i++)
                    {
                        string name = PtrToStringAndFree(NativeMethods.pdf_oxide_form_field_get_name(list, i, out _));
                        string type = PtrToStringAndFree(NativeMethods.pdf_oxide_form_field_get_type(list, i, out _));
                        string value = PtrToStringAndFree(NativeMethods.pdf_oxide_form_field_get_value(list, i, out _));
                        fields[i] = new FormField(name, type, value);
                    }
                    return fields;
                }
                finally
                {
                    NativeMethods.pdf_oxide_form_field_list_free(list);
                }
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        private static string PtrToStringAndFree(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                return string.Empty;
            try
            {
                return Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
            }
            finally
            {
                NativeMethods.FreeString(ptr);
            }
        }

        /// <summary>Gets all pages as a read-only list. Enables foreach and LINQ.</summary>
        public IReadOnlyList<Page> Pages
        {
            get
            {
                ThrowIfDisposed();
                var count = PageCount;
                var pages = new Page[count];
                for (int i = 0; i < count; i++)
                    pages[i] = new Page(this, i);
                return pages;
            }
        }

        /// <summary>Returns the page at the given zero-based index.</summary>
        public Page this[int pageIndex]
        {
            get
            {
                ThrowIfDisposed();
                if (pageIndex < 0 || pageIndex >= PageCount)
                    throw new ArgumentOutOfRangeException(nameof(pageIndex));
                return new Page(this, pageIndex);
            }
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }

        /// <summary>
        /// Sets the global log level for the native pdf_oxide library.
        /// </summary>
        /// <param name="level">Log level: 0=Off, 1=Error, 2=Warn, 3=Info, 4=Debug, 5=Trace</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if level is not between 0 and 5.</exception>
        public static void SetLogLevel(int level)
        {
            if (level < 0 || level > 5)
                throw new ArgumentOutOfRangeException(nameof(level), "Log level must be between 0 (Off) and 5 (Trace).");
            NativeMethods.PdfOxideSetLogLevel(level);
        }

        /// <summary>
        /// Gets the current log level of the native pdf_oxide library.
        /// </summary>
        /// <returns>Current log level (0=Off, 1=Error, 2=Warn, 3=Info, 4=Debug, 5=Trace).</returns>
        public static int GetLogLevel()
        {
            return NativeMethods.PdfOxideGetLogLevel();
        }

        /// <summary>
        /// Returns the name of the active cryptographic provider —
        /// <c>"rust-crypto"</c> for the default permissive provider,
        /// or <c>"aws-lc-rs"</c> for the FIPS-validated provider once
        /// installed via <see cref="UseFipsCryptoProvider"/>. See
        /// <see href="https://github.com/yfedoseev/pdf_oxide/issues/236"/>.
        /// </summary>
        public static string GetActiveCryptoProvider()
        {
            var ptr = NativeMethods.PdfOxideCryptoActiveProvider();
            if (ptr == IntPtr.Zero) return "unknown";
            try
            {
                return Marshal.PtrToStringUTF8(ptr) ?? "unknown";
            }
            finally
            {
                NativeMethods.FreeString(ptr);
            }
        }

        /// <summary>
        /// Whether the FIPS-validated <c>aws-lc-rs</c> provider was
        /// compiled into the native library. Build the native lib
        /// with <c>--features fips</c> to enable.
        /// </summary>
        public static bool IsFipsCryptoAvailable()
        {
            return NativeMethods.PdfOxideCryptoFipsAvailable() != 0;
        }

        /// <summary>
        /// Install the FIPS-validated <c>aws-lc-rs</c> provider as
        /// the process-wide active cryptographic backend. Must be
        /// called before any PDF operation that uses crypto.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// FIPS provider not compiled in, or a provider has already
        /// been installed.
        /// </exception>
        public static void UseFipsCryptoProvider()
        {
            var code = NativeMethods.PdfOxideCryptoUseFips();
            switch (code)
            {
                case 0: return;
                case 1:
                    throw new InvalidOperationException(
                        "FIPS provider not compiled into pdf_oxide native lib; " +
                        "rebuild with `cargo build --features fips`.");
                case 2:
                    throw new InvalidOperationException(
                        "Cryptographic provider already installed — " +
                        "UseFipsCryptoProvider must be called before any PDF operation.");
                default:
                    throw new InvalidOperationException(
                        $"pdf_oxide_crypto_use_fips returned unknown error code {code}.");
            }
        }
    }

    /// <summary>
    /// Premultiplied RGBA8888 pixel buffer returned by <see cref="PdfDocument.RenderToRgba"/>.
    /// Layout: row-major, top-left origin, 4 bytes (R,G,B,A) per pixel.
    /// <c>Data.Length == Width * Height * 4</c>.
    /// </summary>
    public sealed record RgbaPixmap(ReadOnlyMemory<byte> Data, int Width, int Height);

    /// <summary>
    /// A table extracted from a PDF page, with row/column dimensions and per-cell text.
    /// </summary>
    public sealed class Table
    {
        private readonly string[,] _cells;

        /// <summary>Number of rows.</summary>
        public int RowCount { get; }

        /// <summary>Number of columns.</summary>
        public int ColCount { get; }

        /// <summary>True if the first row is a header row.</summary>
        public bool HasHeader { get; }

        internal Table(int rowCount, int colCount, bool hasHeader, string[,] cells)
        {
            RowCount = rowCount;
            ColCount = colCount;
            HasHeader = hasHeader;
            _cells = cells;
        }

        /// <summary>Returns the text of the cell at (row, col). Both indices are zero-based.</summary>
        public string CellText(int row, int col) => _cells[row, col];
    }

    /// <summary>
    /// An embedded image extracted from a PDF page.
    /// </summary>
    public sealed class ExtractedImage
    {
        /// <summary>Image width in pixels.</summary>
        public int Width { get; }

        /// <summary>Image height in pixels.</summary>
        public int Height { get; }

        /// <summary>Container format (e.g. "Jpeg", "Png", "Raw").</summary>
        public string Format { get; }

        /// <summary>Color space string (e.g. "DeviceRGB", "DeviceGray", "DeviceCMYK").</summary>
        public string Colorspace { get; }

        /// <summary>Bits per component (typically 1, 8, or 16).</summary>
        public int BitsPerComponent { get; }

        /// <summary>Raw image bytes. Interpretation depends on <see cref="Format"/>.</summary>
        public byte[] Data { get; }

        internal ExtractedImage(int width, int height, string format, string colorspace, int bitsPerComponent, byte[] data)
        {
            Width = width;
            Height = height;
            Format = format;
            Colorspace = colorspace;
            BitsPerComponent = bitsPerComponent;
            Data = data;
        }
    }

    /// <summary>
    /// An AcroForm field read from a PDF document.
    /// </summary>
    public sealed class FormField
    {
        /// <summary>Fully-qualified field name (e.g. "employee.name").</summary>
        public string Name { get; }

        /// <summary>Field type string (e.g. "Text", "Button", "Choice", "Signature").</summary>
        public string FieldType { get; }

        /// <summary>Current value of the field as a string (empty for unset fields).</summary>
        public string Value { get; }

        internal FormField(string name, string fieldType, string value)
        {
            Name = name;
            FieldType = fieldType;
            Value = value;
        }
    }

    /// <summary>
    /// Represents a single page of a <see cref="PdfDocument"/>.
    /// All extraction methods dispatch to the parent document.
    /// </summary>
    public sealed class Page
    {
        private readonly PdfDocument _doc;

        /// <summary>Zero-based page index.</summary>
        public int Index { get; }

        internal Page(PdfDocument doc, int index)
        {
            _doc = doc;
            Index = index;
        }

        /// <summary>Extracts plain text from the page.</summary>
        public string ExtractText() => _doc.ExtractText(Index);

        /// <summary>Extracts plain text asynchronously.</summary>
        public Task<string> ExtractTextAsync(CancellationToken ct = default) =>
            _doc.ExtractTextAsync(Index, ct);

        /// <summary>Converts the page to Markdown.</summary>
        public string ToMarkdown() => _doc.ToMarkdown(Index);

        /// <summary>Converts the page to Markdown asynchronously.</summary>
        public Task<string> ToMarkdownAsync(CancellationToken ct = default) =>
            Task.Run(() => _doc.ToMarkdown(Index), ct);

        /// <summary>Converts the page to HTML.</summary>
        public string ToHtml() => _doc.ToHtml(Index);

        /// <summary>Converts the page to HTML asynchronously.</summary>
        public Task<string> ToHtmlAsync(CancellationToken ct = default) =>
            Task.Run(() => _doc.ToHtml(Index), ct);

        /// <summary>Converts the page to plain text.</summary>
        public string ToPlainText() => _doc.ToPlainText(Index);

        /// <summary>Converts the page to plain text asynchronously.</summary>
        public Task<string> ToPlainTextAsync(CancellationToken ct = default) =>
            Task.Run(() => _doc.ToPlainText(Index), ct);

        /// <summary>Extracts words with bounding boxes.</summary>
        public (string Text, float X, float Y, float W, float H)[] ExtractWords() =>
            _doc.ExtractWords(Index);

        /// <summary>Extracts text lines with bounding boxes.</summary>
        public (string Text, float X, float Y, float W, float H)[] ExtractLines() =>
            _doc.ExtractTextLines(Index);

        /// <summary>Extracts tables from the page.</summary>
        public Table[] ExtractTables() => _doc.ExtractTables(Index);

        /// <summary>Extracts embedded images from the page.</summary>
        public ExtractedImage[] ExtractImages() => _doc.ExtractImages(Index);

        /// <summary>Extracts characters with bounding boxes.</summary>
        public (char Char, float X, float Y, float W, float H)[] ExtractChars() =>
            _doc.ExtractChars(Index);

        /// <summary>Extracts path geometries from the page.</summary>
        public (float X, float Y, float W, float H, float StrokeWidth)[] ExtractPaths() =>
            _doc.ExtractPaths(Index);

        /// <summary>Returns font names used on the page.</summary>
        public string[] GetFonts() => _doc.GetFonts(Index);

        /// <summary>Searches for text on the page.</summary>
        public (int Page, string Text, float X, float Y, float W, float H)[] Search(
            string text, bool caseSensitive = false) =>
            _doc.SearchPage(Index, text, caseSensitive);

        /// <summary>Renders the page to image bytes.</summary>
        public byte[] Render(int format = 0) => _doc.RenderPage(Index, format);

        /// <summary>Renders a thumbnail of the page.</summary>
        public byte[] RenderThumbnail(int format = 0) => _doc.RenderThumbnail(Index, format);

        /// <inheritdoc/>
        public override string ToString() => $"Page(index={Index})";
    }
}
