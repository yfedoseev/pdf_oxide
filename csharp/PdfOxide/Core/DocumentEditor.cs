using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using PdfOxide.Exceptions;
using PdfOxide.Internal;

namespace PdfOxide.Core
{
    /// <summary>
    /// Represents a PDF document opened for editing.
    /// Provides capabilities to modify metadata, content, and save changes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// DocumentEditor is the editing API that provides:
    /// <list type="bullet">
    /// <item><description>Opening existing PDFs for editing</description></item>
    /// <item><description>Modifying document metadata (title, author, subject)</description></item>
    /// <item><description>Managing pages (add, remove, reorder)</description></item>
    /// <item><description>Modifying page content (text, images, annotations)</description></item>
    /// <item><description>Saving changes with incremental updates or full rewrite</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The document must be explicitly disposed to release native resources.
    /// Use 'using' statements for automatic cleanup.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Open a PDF for editing
    /// using (var editor = DocumentEditor.Open("document.pdf"))
    /// {
    ///     // Modify metadata
    ///     editor.Title = "Updated Title";
    ///     editor.Author = "New Author";
    ///
    ///     // Save changes
    ///     editor.Save("output.pdf");
    /// }
    /// </code>
    /// </example>
    public sealed class DocumentEditor : IDisposable
    {
        private NativeHandle _handle;
        private bool _disposed;

        private DocumentEditor(NativeHandle handle)
        {
            _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        }

        /// <summary>
        /// Opens a PDF document for editing.
        /// </summary>
        /// <param name="path">The file path to the PDF document.</param>
        /// <returns>A new DocumentEditor instance.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="path"/> is null.</exception>
        /// <exception cref="PdfException">Thrown if the document cannot be opened.</exception>
        /// <example>
        /// <code>
        /// using (var editor = DocumentEditor.Open("document.pdf"))
        /// {
        ///     Console.WriteLine($"Pages: {editor.PageCount}");
        /// }
        /// </code>
        /// </example>
        public static DocumentEditor Open(string path)
        {
            ArgumentNullException.ThrowIfNull(path);

            var handle = NativeMethods.DocumentEditorOpen(path, out var errorCode);
            if (handle.IsInvalid)
            {
                ExceptionMapper.ThrowIfError(errorCode);
            }

            return new DocumentEditor(handle);
        }

        /// <summary>
        /// Checks if the document has unsaved changes.
        /// </summary>
        /// <value>True if the document has been modified, false otherwise.</value>
        public bool IsModified
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.DocumentEditorIsModified(_handle.DangerousGetHandle());
            }
        }

        /// <summary>
        /// Gets the source file path for this document.
        /// </summary>
        /// <value>The file path where the document was opened from.</value>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        public string SourcePath
        {
            get
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.DocumentEditorGetSourcePath(_handle.DangerousGetHandle(), out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);

                try
                {
                    return StringMarshaler.PtrToString(ptr);
                }
                finally
                {
                    NativeMethods.FreeString(ptr);
                }
            }
        }

        /// <summary>
        /// Gets the PDF version as (major, minor).
        /// </summary>
        /// <value>A tuple containing the major and minor version numbers.</value>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        public (byte Major, byte Minor) Version
        {
            get
            {
                ThrowIfDisposed();
                NativeMethods.DocumentEditorGetVersion(_handle.DangerousGetHandle(),
                    out var major, out var minor);
                return (major, minor);
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
                ThrowIfDisposed();
                var count = NativeMethods.DocumentEditorGetPageCount(_handle.DangerousGetHandle(), out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                return count;
            }
        }

        /// <summary>
        /// Gets or sets the document title.
        /// </summary>
        /// <value>The document title, or <c>null</c> if not set.</value>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        /// <exception cref="PdfException">Thrown if the title cannot be retrieved or set.</exception>
        public string? Title
        {
            get
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.DocumentEditorGetTitle(_handle.DangerousGetHandle(), out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);

                if (ptr == IntPtr.Zero)
                    return null;

                try
                {
                    return StringMarshaler.PtrToString(ptr);
                }
                finally
                {
                    NativeMethods.FreeString(ptr);
                }
            }
            set
            {
                ThrowIfDisposed();
                NativeMethods.DocumentEditorSetTitle(_handle.DangerousGetHandle(), value, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
            }
        }

        /// <summary>
        /// Gets or sets the document author.
        /// </summary>
        /// <value>The document author, or <c>null</c> if not set.</value>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        /// <exception cref="PdfException">Thrown if the author cannot be retrieved or set.</exception>
        public string? Author
        {
            get
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.DocumentEditorGetAuthor(_handle.DangerousGetHandle(), out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);

                if (ptr == IntPtr.Zero)
                    return null;

                try
                {
                    return StringMarshaler.PtrToString(ptr);
                }
                finally
                {
                    NativeMethods.FreeString(ptr);
                }
            }
            set
            {
                ThrowIfDisposed();
                NativeMethods.DocumentEditorSetAuthor(_handle.DangerousGetHandle(), value, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
            }
        }

        /// <summary>
        /// Gets or sets the document subject.
        /// </summary>
        /// <value>The document subject, or <c>null</c> if not set.</value>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        /// <exception cref="PdfException">Thrown if the subject cannot be retrieved or set.</exception>
        public string? Subject
        {
            get
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.DocumentEditorGetSubject(_handle.DangerousGetHandle(), out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);

                if (ptr == IntPtr.Zero)
                    return null;

                try
                {
                    return StringMarshaler.PtrToString(ptr);
                }
                finally
                {
                    NativeMethods.FreeString(ptr);
                }
            }
            set
            {
                ThrowIfDisposed();
                NativeMethods.DocumentEditorSetSubject(_handle.DangerousGetHandle(), value, out var errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
            }
        }

        /// <summary>
        /// Saves the document to a file.
        /// </summary>
        /// <param name="path">The output file path.</param>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="path"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if the document has been disposed.</exception>
        /// <exception cref="PdfIoException">Thrown if the file cannot be written.</exception>
        /// <example>
        /// <code>
        /// using (var editor = DocumentEditor.Open("input.pdf"))
        /// {
        ///     editor.Title = "Modified";
        ///     editor.Save("output.pdf");
        /// }
        /// </code>
        /// </example>
        public void Save(string path)
        {
            ArgumentNullException.ThrowIfNull(path);

            ThrowIfDisposed();

            var result = NativeMethods.DocumentEditorSave(_handle.DangerousGetHandle(), path, out var errorCode);
            if (result != 0)
            {
                ExceptionMapper.ThrowIfError(errorCode);
            }
        }

        /// <summary>
        /// Sets the value of a form (AcroForm) field by its fully-qualified name.
        /// </summary>
        /// <param name="name">Fully-qualified field name (e.g. "employee.ssn").</param>
        /// <param name="value">New field value.</param>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="name"/> or <paramref name="value"/> is null.</exception>
        /// <exception cref="PdfException">Thrown if the native call fails.</exception>
        public void SetFormFieldValue(string name, string value)
        {
            ArgumentNullException.ThrowIfNull(name);
            ArgumentNullException.ThrowIfNull(value);
            ThrowIfDisposed();
            int rc = NativeMethods.document_editor_set_form_field_value(_handle, name, value, out int err);
            if (rc != 0)
                ExceptionMapper.ThrowIfError(err != 0 ? err : 11);
        }

        /// <summary>
        /// Flattens all form fields in the document, converting their rendered appearance into static content.
        /// After flattening, the fields are no longer editable.
        /// </summary>
        /// <exception cref="PdfException">Thrown if the native call fails.</exception>
        public void FlattenForms()
        {
            ThrowIfDisposed();
            int rc = NativeMethods.document_editor_flatten_forms(_handle, out int err);
            if (rc != 0)
                ExceptionMapper.ThrowIfError(err != 0 ? err : 11);
        }

        /// <summary>
        /// Asynchronously saves the document to a file.
        /// </summary>
        /// <param name="path">The output file path.</param>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>A task that completes when the file is saved.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="path"/> is null.</exception>
        /// <exception cref="OperationCanceledException">Thrown if the operation is cancelled.</exception>
        public Task SaveAsync(string path, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(path);

            return Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                Save(path);
            }, cancellationToken);
        }

        // ================================================================
        // Mutations. All 12 methods below are thin wrappers over the
        // `document_editor_*` P/Invoke declarations in NativeMethods.cs.
        // ================================================================

        /// <summary>
        /// Append all pages from another PDF file to the end of this document.
        /// </summary>
        /// <param name="sourcePath">Path of the PDF whose pages should be appended.</param>
        public void MergeFrom(string sourcePath)
        {
            ArgumentNullException.ThrowIfNull(sourcePath);
            ThrowIfDisposed();
            NativeMethods.document_editor_merge_from(_handle, sourcePath, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Delete the page at <paramref name="pageIndex"/> (zero-based).
        /// </summary>
        public void DeletePage(int pageIndex)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_delete_page(_handle, pageIndex, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Move a page from one position to another. Indices are zero-based
        /// and refer to positions <em>before</em> the move, matching the
        /// Rust / Python / Go contract.
        /// </summary>
        public void MovePage(int fromIndex, int toIndex)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_move_page(_handle, fromIndex, toIndex, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Get the rotation of a page in degrees (0, 90, 180, 270).
        /// </summary>
        public int GetPageRotation(int pageIndex)
        {
            ThrowIfDisposed();
            int degrees = NativeMethods.document_editor_get_page_rotation(_handle, pageIndex, out int err);
            ExceptionMapper.ThrowIfError(err);
            return degrees;
        }

        /// <summary>
        /// Set the rotation of a page. Valid values are 0, 90, 180, 270.
        /// </summary>
        public void SetPageRotation(int pageIndex, int degrees)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_set_page_rotation(_handle, pageIndex, degrees, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Crop the visible area of every page by subtracting margins (points).
        /// </summary>
        public void CropMargins(float left, float right, float top, float bottom)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_crop_margins(_handle, left, right, top, bottom, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Erase a rectangular region on <paramref name="pageIndex"/>.
        /// Origin is the page's bottom-left in PDF user space.
        /// </summary>
        public void EraseRegion(int pageIndex, float x, float y, float width, float height)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_erase_region(_handle, pageIndex, x, y, width, height, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Flatten annotations on a single page — bakes their visual
        /// representation into page content and removes the interactive
        /// annotation objects.
        /// </summary>
        public void FlattenAnnotations(int pageIndex)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_flatten_annotations(_handle, pageIndex, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Flatten annotations across every page in the document.
        /// </summary>
        public void FlattenAllAnnotations()
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_flatten_all_annotations(_handle, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Flatten form fields on a single page without touching the rest
        /// of the document.
        /// </summary>
        public void FlattenFormsOnPage(int pageIndex)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_flatten_forms_on_page(_handle, pageIndex, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Returns warnings collected during the last form-flattening save.
        /// Each entry names a widget field that had no <c>/AP</c> appearance
        /// stream; flattening it produces a blank rectangle.
        /// </summary>
        public string[] FlattenWarnings
        {
            get
            {
                ThrowIfDisposed();
                int count = NativeMethods.document_editor_flatten_warnings_count(_handle);
                if (count <= 0) return Array.Empty<string>();
                var result = new string[count];
                for (int i = 0; i < count; i++)
                {
                    IntPtr ptr = NativeMethods.document_editor_flatten_warning(_handle, i, out int _);
                    if (ptr != IntPtr.Zero)
                    {
                        result[i] = System.Runtime.InteropServices.Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
                        NativeMethods.FreeString(ptr);
                    }
                }
                return result;
            }
        }

        /// <summary>
        /// Save the document with AES-256 encryption using the supplied
        /// user and owner passwords.
        /// </summary>
        public void SaveEncrypted(string path, string userPassword, string ownerPassword)
        {
            ArgumentNullException.ThrowIfNull(path);
            ArgumentNullException.ThrowIfNull(userPassword);
            ArgumentNullException.ThrowIfNull(ownerPassword);
            ThrowIfDisposed();
            NativeMethods.document_editor_save_encrypted(_handle, path, userPassword, ownerPassword, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Gets or sets the document producer string (the tool that
        /// produced the PDF). Round-trips through the
        /// <c>/Info.Producer</c> metadata entry.
        /// </summary>
        public string? Producer
        {
            get
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.document_editor_get_producer(_handle, out int err);
                ExceptionMapper.ThrowIfError(err);
                if (ptr == IntPtr.Zero) return null;
                try { return StringMarshaler.PtrToString(ptr); }
                finally { NativeMethods.FreeString(ptr); }
            }
            set
            {
                ThrowIfDisposed();
                NativeMethods.document_editor_set_producer(_handle, value ?? string.Empty, out int err);
                ExceptionMapper.ThrowIfError(err);
            }
        }

        /// <summary>
        /// Gets or sets the raw PDF creation-date string
        /// (e.g. <c>D:20260421120000Z</c>).
        /// </summary>
        public string? CreationDate
        {
            get
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.document_editor_get_creation_date(_handle, out int err);
                ExceptionMapper.ThrowIfError(err);
                if (ptr == IntPtr.Zero) return null;
                try { return StringMarshaler.PtrToString(ptr); }
                finally { NativeMethods.FreeString(ptr); }
            }
            set
            {
                ThrowIfDisposed();
                NativeMethods.document_editor_set_creation_date(_handle, value ?? string.Empty, out int err);
                ExceptionMapper.ThrowIfError(err);
            }
        }

        // ================================================================
        // New methods (v0.3.39)
        // ================================================================

        /// <summary>
        /// Opens a DocumentEditor from an in-memory byte array.
        /// </summary>
        public static DocumentEditor OpenFromBytes(byte[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            if (data.Length == 0)
                throw new ArgumentException("Data must not be empty.", nameof(data));
            var handle = NativeMethods.document_editor_open_from_bytes(data, (nuint)data.Length, out var errorCode);
            ExceptionMapper.ThrowIfError(errorCode);
            return new DocumentEditor(handle);
        }

        /// <summary>
        /// Saves the editor contents to an in-memory byte array.
        /// </summary>
        public byte[] SaveToBytes()
        {
            ThrowIfDisposed();
            var ptr = NativeMethods.document_editor_save_to_bytes(_handle, out nuint outLen, out int errorCode);
            ExceptionMapper.ThrowIfError(errorCode);
            if (ptr == IntPtr.Zero || outLen == 0) return Array.Empty<byte>();
            try
            {
                var bytes = new byte[(int)outLen];
                System.Runtime.InteropServices.Marshal.Copy(ptr, bytes, 0, (int)outLen);
                return bytes;
            }
            finally
            {
                NativeMethods.FreeBytes(ptr);
            }
        }

        /// <summary>
        /// Extracts a subset of pages into a new in-memory PDF.
        /// </summary>
        /// <param name="pageIndices">Zero-based page indices to extract.</param>
        public unsafe byte[] ExtractPages(int[] pageIndices)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(pageIndices);
            if (pageIndices.Length == 0) return Array.Empty<byte>();
            fixed (int* pPages = pageIndices)
            {
                var ptr = NativeMethods.document_editor_extract_pages_to_bytes(
                    _handle, pPages, (nuint)pageIndices.Length, out nuint outLen, out int errorCode);
                ExceptionMapper.ThrowIfError(errorCode);
                if (ptr == IntPtr.Zero || outLen == 0) return Array.Empty<byte>();
                try
                {
                    var bytes = new byte[(int)outLen];
                    System.Runtime.InteropServices.Marshal.Copy(ptr, bytes, 0, (int)outLen);
                    return bytes;
                }
                finally
                {
                    NativeMethods.FreeBytes(ptr);
                }
            }
        }

        /// <summary>
        /// Converts the document to PDF/A in-place.
        /// </summary>
        /// <param name="level">PDF/A conformance level (default A2b).</param>
        public void ConvertToPdfA(PdfALevel level = PdfALevel.A2b)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_convert_to_pdf_a(_handle, (int)level, out int errorCode);
            ExceptionMapper.ThrowIfError(errorCode);
        }

        /// <summary>
        /// Saves the document with AES-256 encryption and returns the bytes.
        /// </summary>
        public byte[] SaveEncryptedToBytes(string userPassword, string ownerPassword)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(userPassword);
            if (string.IsNullOrEmpty(ownerPassword)) ownerPassword = userPassword;
            var ptr = NativeMethods.document_editor_save_encrypted_to_bytes(
                _handle, userPassword, ownerPassword, out nuint outLen, out int errorCode);
            ExceptionMapper.ThrowIfError(errorCode);
            if (ptr == IntPtr.Zero || outLen == 0) return Array.Empty<byte>();
            try
            {
                var bytes = new byte[(int)outLen];
                System.Runtime.InteropServices.Marshal.Copy(ptr, bytes, 0, (int)outLen);
                return bytes;
            }
            finally
            {
                NativeMethods.FreeBytes(ptr);
            }
        }

        /// <summary>
        /// Saves the editor contents to bytes with compression / garbage-collection / linearize flags.
        /// </summary>
        public byte[] SaveToBytesWithOptions(bool compress, bool garbageCollect, bool linearize)
        {
            ThrowIfDisposed();
            var ptr = NativeMethods.document_editor_save_to_bytes_with_options(
                _handle, compress, garbageCollect, linearize, out nuint outLen, out int errorCode);
            ExceptionMapper.ThrowIfError(errorCode);
            if (ptr == IntPtr.Zero || outLen == 0) return Array.Empty<byte>();
            try
            {
                var bytes = new byte[(int)outLen];
                System.Runtime.InteropServices.Marshal.Copy(ptr, bytes, 0, (int)outLen);
                return bytes;
            }
            finally
            {
                NativeMethods.FreeBytes(ptr);
            }
        }

        /// <summary>
        /// Gets or sets the document keywords metadata.
        /// </summary>
        public string? Keywords
        {
            get
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.document_editor_get_keywords(_handle, out int err);
                ExceptionMapper.ThrowIfError(err);
                if (ptr == IntPtr.Zero) return null;
                try { return StringMarshaler.PtrToString(ptr); }
                finally { NativeMethods.FreeString(ptr); }
            }
            set
            {
                ThrowIfDisposed();
                NativeMethods.document_editor_set_keywords(_handle, value ?? string.Empty, out int err);
                ExceptionMapper.ThrowIfError(err);
            }
        }

        /// <summary>
        /// Merges pages from an in-memory PDF byte array into this document.
        /// Returns the number of pages added.
        /// </summary>
        public int MergeFromBytes(byte[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            if (data.Length == 0)
                throw new ArgumentException("Data must not be empty.", nameof(data));
            ThrowIfDisposed();
            int n = NativeMethods.document_editor_merge_from_bytes(_handle, data, (nuint)data.Length, out int err);
            ExceptionMapper.ThrowIfError(err);
            return n;
        }

        /// <summary>
        /// Embeds a file attachment into the document.
        /// </summary>
        public void EmbedFile(string name, byte[] data)
        {
            ArgumentNullException.ThrowIfNull(name);
            ArgumentNullException.ThrowIfNull(data);
            ThrowIfDisposed();
            NativeMethods.document_editor_embed_file(_handle, name, data, (nuint)data.Length, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Burns in redaction annotations on a single page.
        /// </summary>
        public void ApplyPageRedactions(int pageIndex)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_apply_page_redactions(_handle, (nuint)pageIndex, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Burns in all pending redaction annotations across the document.
        /// </summary>
        public void ApplyAllRedactions()
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_apply_all_redactions(_handle, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Queues an explicit destructive redaction rectangle on a page
        /// (page user space). The underlying content is physically removed
        /// by <see cref="ApplyRedactions"/> — not a cosmetic overlay
        /// (ISO 32000-1:2008 §12.5.6.23).
        /// </summary>
        public void AddRedaction(int pageIndex, double x1, double y1, double x2, double y2,
            double r = 0, double g = 0, double b = 0)
        {
            ThrowIfDisposed();
            NativeMethods.pdf_redaction_add(_handle, (nuint)pageIndex, x1, y1, x2, y2, r, g, b, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Number of redaction regions queued for a page (annotations +
        /// programmatic rectangles).
        /// </summary>
        public int RedactionCount(int pageIndex)
        {
            ThrowIfDisposed();
            int n = NativeMethods.pdf_redaction_count(_handle, (nuint)pageIndex, out int err);
            ExceptionMapper.ThrowIfError(err);
            return n;
        }

        /// <summary>
        /// Destructively applies all queued redactions (true content
        /// removal). Returns the number of glyphs physically removed.
        /// </summary>
        public int ApplyRedactions(bool scrubMetadata = true,
            double r = 0, double g = 0, double b = 0)
        {
            ThrowIfDisposed();
            int removed = NativeMethods.pdf_redaction_apply(_handle, scrubMetadata, r, g, b, out int err);
            ExceptionMapper.ThrowIfError(err);
            return removed;
        }

        /// <summary>
        /// Standalone document sanitization (no geometric redaction): strips
        /// the /Info dictionary, the catalog XMP /Metadata stream, document
        /// JavaScript (/OpenAction, /AA, /Names/JavaScript) and
        /// /Names/EmbeddedFiles, hard-excluding the removed object subtrees
        /// from the rewritten file. Returns the number of annotations removed
        /// (issue #231).
        /// </summary>
        public int SanitizeDocument()
        {
            ThrowIfDisposed();
            int removed = NativeMethods.pdf_redaction_scrub_metadata(_handle, out int err);
            ExceptionMapper.ThrowIfError(err);
            return removed;
        }

        /// <summary>
        /// Rotates all pages by <paramref name="degrees"/> (additive, not absolute).
        /// </summary>
        public void RotateAllPages(int degrees)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_rotate_all_pages(_handle, degrees, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Rotates a single page by <paramref name="degrees"/> (additive, not absolute).
        /// </summary>
        public void RotatePageBy(int pageIndex, int degrees)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_rotate_page_by(_handle, (nuint)pageIndex, degrees, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Gets the MediaBox of a page as (x, y, width, height).
        /// </summary>
        public (double X, double Y, double Width, double Height) GetPageMediaBox(int pageIndex)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_get_page_media_box(
                _handle, (nuint)pageIndex, out double x, out double y, out double w, out double h, out int err);
            ExceptionMapper.ThrowIfError(err);
            return (x, y, w, h);
        }

        /// <summary>
        /// Sets the MediaBox of a page.
        /// </summary>
        public void SetPageMediaBox(int pageIndex, double x, double y, double width, double height)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_set_page_media_box(
                _handle, (nuint)pageIndex, x, y, width, height, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Gets the CropBox of a page as (x, y, width, height).
        /// Returns (0, 0, 0, 0) if no CropBox is set.
        /// </summary>
        public (double X, double Y, double Width, double Height) GetPageCropBox(int pageIndex)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_get_page_crop_box(
                _handle, (nuint)pageIndex, out double x, out double y, out double w, out double h, out int err);
            ExceptionMapper.ThrowIfError(err);
            return (x, y, w, h);
        }

        /// <summary>
        /// Sets the CropBox of a page.
        /// </summary>
        public void SetPageCropBox(int pageIndex, double x, double y, double width, double height)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_set_page_crop_box(
                _handle, (nuint)pageIndex, x, y, width, height, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Erases multiple rectangular regions on a page.
        /// Each element of <paramref name="rects"/> is [x, y, w, h].
        /// </summary>
        public void EraseRegions(int pageIndex, double[][] rects)
        {
            ArgumentNullException.ThrowIfNull(rects);
            ThrowIfDisposed();
            if (rects.Length == 0) return;
            var flat = new double[rects.Length * 4];
            for (int i = 0; i < rects.Length; i++)
            {
                if (rects[i].Length < 4)
                    throw new ArgumentException($"rects[{i}] must have 4 elements.", nameof(rects));
                flat[i * 4 + 0] = rects[i][0];
                flat[i * 4 + 1] = rects[i][1];
                flat[i * 4 + 2] = rects[i][2];
                flat[i * 4 + 3] = rects[i][3];
            }
            NativeMethods.document_editor_erase_regions(
                _handle, (nuint)pageIndex, flat, (nuint)rects.Length, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Clears all pending erase-region entries for a page.
        /// </summary>
        public void ClearEraseRegions(int pageIndex)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_clear_erase_regions(_handle, (nuint)pageIndex, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Returns true if the page is marked for annotation-flatten.
        /// </summary>
        public bool IsPageMarkedForFlatten(int pageIndex)
        {
            ThrowIfDisposed();
            return NativeMethods.document_editor_is_page_marked_for_flatten(_handle, (nuint)pageIndex) == 1;
        }

        /// <summary>
        /// Removes the flatten mark from a page.
        /// </summary>
        public void UnmarkPageForFlatten(int pageIndex)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_unmark_page_for_flatten(_handle, (nuint)pageIndex, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Returns true if the page is marked for redaction.
        /// </summary>
        public bool IsPageMarkedForRedaction(int pageIndex)
        {
            ThrowIfDisposed();
            return NativeMethods.document_editor_is_page_marked_for_redaction(_handle, (nuint)pageIndex) == 1;
        }

        /// <summary>
        /// Removes the redaction mark from a page.
        /// </summary>
        public void UnmarkPageForRedaction(int pageIndex)
        {
            ThrowIfDisposed();
            NativeMethods.document_editor_unmark_page_for_redaction(_handle, (nuint)pageIndex, out int err);
            ExceptionMapper.ThrowIfError(err);
        }

        /// <summary>
        /// Disposes the DocumentEditor and releases native resources.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _handle?.Dispose();
                _disposed = true;
            }
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
