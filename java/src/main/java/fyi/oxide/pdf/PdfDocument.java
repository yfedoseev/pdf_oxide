/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import fyi.oxide.pdf.exception.PdfInvalidStateException;
import fyi.oxide.pdf.exception.PdfIoException;
import fyi.oxide.pdf.internal.NativeLoader;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.Cleaner;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The primary read-only entry point to a PDF.
 *
 * <p><b>Lifecycle.</b> A {@code PdfDocument} owns native memory and
 * <b>must be closed</b> when no longer in use. The recommended idiom
 * is try-with-resources:
 *
 * <pre>{@code
 * try (PdfDocument doc = PdfDocument.open(Paths.get("invoice.pdf"))) {
 *     System.out.println(doc.extractText(0));
 * }
 * }</pre>
 *
 * <p>Calls to {@link #close()} are idempotent — a second call is a
 * no-op, NOT a JVM crash. A {@link java.lang.ref.Cleaner} backstop
 * is registered to free leaked handles and emit a warning, but
 * callers must not rely on it for timely cleanup; it runs on a
 * dedicated thread with no ordering guarantees.
 *
 * <p><b>Thread safety.</b> Instances are <b>not thread-safe</b>.
 * Open one document per worker. (Stateless static helpers like
 * {@link MarkdownConverter} and {@link PdfValidator} are thread-safe.)
 *
 * <p><b>Convenience helpers.</b> {@link #extractText(String)},
 * {@link #extractMarkdown(String)} and {@link #extractAuto(String)}
 * are static one-shots that open + extract + close in a single call.
 * Use them for the simple case; use {@link #open(Path)} +
 * try-with-resources for everything else.
 */
public final class PdfDocument implements AutoCloseable {

    static {
        NativeLoader.ensureLoaded();
    }

    /** Shared cleaner for leak detection (logs once per leaked handle). */
    private static final Cleaner CLEANER = Cleaner.create();

    /** Diagnostic: number of currently-live native handles. Test-only signal. */
    private static final AtomicLong LIVE_HANDLES = new AtomicLong(0);

    /**
     * Native handle state, **shared** between this {@code PdfDocument}
     * and its {@link HandleCleaner}. Stored in an {@link AtomicLong}
     * (not a {@code volatile long} field directly) so the cleaner
     * sees zero-ing done by {@link #close()} — captures-by-value
     * across the cleaner boundary would let the cleaner re-free a
     * pointer already freed by {@code close()} (the empirically-
     * observed glibc "double free or corruption (out)" — fixed by
     * this design).
     *
     * <p>The cleaner's reference to this object is OK for GC: the
     * cleaner holds only a reference to the {@code AtomicLong}, not
     * back to {@code PdfDocument}, so {@code PdfDocument} remains
     * GC-eligible once user code drops it.
     */
    private final AtomicLong handleState;

    /** Cleaner registration for leak detection. */
    private final Cleaner.Cleanable cleanable;

    /**
     * Internal constructor. Public callers go through {@link #open}.
     * The native side leaks a {@code Box<PdfDocument>} and returns the
     * raw pointer cast to {@code jlong}; the Java side stores it and
     * frees on {@link #close()}.
     */
    private PdfDocument(long handle) {
        this.handleState = new AtomicLong(handle);
        LIVE_HANDLES.incrementAndGet();
        this.cleanable = CLEANER.register(this, new HandleCleaner(this.handleState));
    }

    // ────────────────────── factories ──────────────────────

    /**
     * Open a PDF from a filesystem path.
     *
     * @param path absolute or relative path to a PDF file.
     * @return a non-closed {@code PdfDocument} — caller is responsible
     *         for invoking {@link #close()} (use try-with-resources).
     * @throws fyi.oxide.pdf.exception.PdfParseException for malformed PDFs.
     * @throws fyi.oxide.pdf.exception.PdfEncryptedException for password-required PDFs.
     * @throws fyi.oxide.pdf.exception.PdfIoException for filesystem failures.
     */
    public static PdfDocument open(Path path) {
        Objects.requireNonNull(path, "path");
        final long h = nativeOpenPath(path.toAbsolutePath().toString());
        return new PdfDocument(h);
    }

    /** Convenience overload taking a string path. */
    public static PdfDocument open(String path) {
        Objects.requireNonNull(path, "path");
        return open(Paths.get(path));
    }

    /** Open a PDF from an in-memory byte array. The bytes are copied. */
    public static PdfDocument open(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        final long h = nativeOpenBytes(bytes);
        return new PdfDocument(h);
    }

    /**
     * Open + authenticate in one call. Convenience for encrypted
     * PDFs where the password is known up front.
     *
     * @throws fyi.oxide.pdf.exception.PdfEncryptedException if the
     *         password is wrong (authentication returned false).
     */
    public static PdfDocument open(Path path, String password) {
        PdfDocument doc = open(path);
        try {
            if (!doc.authenticate(Objects.requireNonNull(password, "password"))) {
                throw new fyi.oxide.pdf.exception.PdfEncryptedException("wrong password for PDF: " + path);
            }
            return doc;
        } catch (RuntimeException | Error e) {
            doc.close();
            throw e;
        }
    }

    /** {@link #open(Path, String)} taking a string path. */
    public static PdfDocument open(String path, String password) {
        return open(Paths.get(Objects.requireNonNull(path, "path")), password);
    }

    /** {@link #open(Path, String)} taking in-memory bytes. */
    public static PdfDocument open(byte[] bytes, String password) {
        PdfDocument doc = open(bytes);
        try {
            if (!doc.authenticate(Objects.requireNonNull(password, "password"))) {
                throw new fyi.oxide.pdf.exception.PdfEncryptedException("wrong password for PDF (in-memory)");
            }
            return doc;
        } catch (RuntimeException | Error e) {
            doc.close();
            throw e;
        }
    }

    /** Open a PDF from an {@link InputStream}; reads to byte[] internally. */
    public static PdfDocument open(InputStream stream) {
        Objects.requireNonNull(stream, "stream");
        try {
            return open(readAll(stream));
        } catch (IOException e) {
            throw new PdfIoException("Failed reading InputStream: " + e.getMessage(), e);
        }
    }

    // ────────────────── static convenience ─────────────────

    /**
     * Open + extract page 0 text + close in one call. Convenience for
     * the most common case.
     */
    public static String extractText(String path) {
        try (PdfDocument doc = open(path)) {
            return doc.extractText(0);
        }
    }

    /** Same as {@link #extractText(String)} but accepting a {@link Path}. */
    public static String extractText(Path path) {
        try (PdfDocument doc = open(path)) {
            return doc.extractText(0);
        }
    }

    // ─────────────────────── instance ──────────────────────

    /**
     * Authenticate against this document's encryption with a password.
     *
     * <p>For unencrypted PDFs returns {@code true} immediately (no
     * authentication is needed). For encrypted PDFs returns
     * {@code true} on the correct password and {@code false} on the
     * wrong one.
     *
     * <p>Call once after {@link #open} before any extraction call —
     * subsequent calls on a successfully-authenticated document
     * succeed normally; calls before successful authentication on an
     * encrypted document throw {@link PdfEncryptedException}.
     *
     * @param password the password as bytes (UTF-8 typically; ISO 32000-1
     *                 §7.6.3 permits PDFDocEncoding for owner password).
     * @return {@code true} on success.
     * @throws PdfInvalidStateException if this document has been closed.
     */
    public boolean authenticate(byte[] password) {
        Objects.requireNonNull(password, "password");
        return nativeAuthenticate(checkHandle(), password);
    }

    /** Convenience: {@code authenticate(password.getBytes(StandardCharsets.UTF_8))}. */
    public boolean authenticate(String password) {
        Objects.requireNonNull(password, "password");
        return authenticate(password.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    /**
     * @return the number of pages in the document.
     * @throws PdfInvalidStateException if this document has been closed.
     */
    public int pageCount() {
        return nativePageCount(checkHandle());
    }

    /**
     * Auto-routed extraction for a single page (v0.3.51 #517).
     * Returns native text-layer content when present, OCR text for
     * scanned regions when the {@code ocr} feature is available, and
     * gracefully falls back to native + a logged warning when OCR is
     * unavailable — <b>never</b> throws
     * {@link fyi.oxide.pdf.exception.PdfOcrUnavailableException} on
     * this path (use {@link AutoExtractor#extractPage} with
     * {@code mode=FORCE_OCR} for the strict-OCR variant).
     *
     * @param pageIndex 0-based page index.
     * @return the extracted text; may be empty if the page has no text.
     */
    public String extractTextAuto(int pageIndex) {
        return nativeExtractTextAuto(checkHandle(), pageIndex);
    }

    /**
     * Render a page to PNG bytes at the default 150 DPI. Requires
     * the {@code rendering} Cargo feature on the {@code pdf_oxide_jni}
     * build (included in the {@code full} feature, which the
     * published fat-jar ships with).
     *
     * @param pageIndex 0-based page index.
     * @return PNG-encoded image bytes (decodable by {@link
     *         javax.imageio.ImageIO#read(java.io.InputStream)}).
     */
    public byte[] render(int pageIndex) {
        return nativeRenderPng(checkHandle(), pageIndex, 0);
    }

    /**
     * Render a page to PNG bytes at the supplied DPI.
     *
     * @param pageIndex 0-based page index.
     * @param dpi resolution in dots-per-inch (e.g. 72, 150, 300).
     *            Must be positive; {@code &le; 0} uses the default 150.
     */
    public byte[] render(int pageIndex, int dpi) {
        return nativeRenderPng(checkHandle(), pageIndex, dpi);
    }

    /**
     * @return the Document Info dictionary's {@code /Producer} entry,
     *         or {@link java.util.Optional#empty()} if missing.
     */
    public java.util.Optional<String> producer() {
        return java.util.Optional.ofNullable(nativeProducer(checkHandle()));
    }

    /**
     * @return the Document Info dictionary's {@code /Creator} entry,
     *         or {@link java.util.Optional#empty()} if missing.
     */
    public java.util.Optional<String> creator() {
        return java.util.Optional.ofNullable(nativeCreator(checkHandle()));
    }

    /**
     * @return all AcroForm fields in this document. v0.3.53
     *         limitation: each field's {@code pageIndex} is {@code -1}
     *         because pdf_oxide's form extractor doesn't yet expose
     *         per-field page placement; the field is identified by
     *         its {@code name} only.
     */
    public java.util.List<fyi.oxide.pdf.form.FormField> formFields() {
        return nativeFormFields(checkHandle());
    }

    /**
     * Search the document for a pattern (literal text by default;
     * regex when {@code regex=true}). Returns the matches in
     * document order with per-match page index, on-page bbox, and
     * the matched text.
     *
     * @param query           the pattern to search for.
     * @param caseInsensitive whether to ignore case.
     * @param regex           when true, treat {@code query} as a
     *                        regex; when false, treat as literal.
     * @param maxResults      cap on number of matches ({@code &le; 0}
     *                        means no cap).
     */
    public java.util.List<fyi.oxide.pdf.search.SearchMatch> search(
            String query, boolean caseInsensitive, boolean regex, int maxResults) {
        Objects.requireNonNull(query, "query");
        return nativeSearch(checkHandle(), query, caseInsensitive, !regex, maxResults);
    }

    /** {@link #search(String, boolean, boolean, int)} with defaults (literal, case-sensitive, no cap). */
    public java.util.List<fyi.oxide.pdf.search.SearchMatch> search(String query) {
        return search(query, false, false, 0);
    }

    /**
     * Convenience: convert this document to Markdown. Equivalent to
     * {@link MarkdownConverter#toMarkdown(PdfDocument)}.
     */
    public String toMarkdown() {
        return MarkdownConverter.toMarkdown(this);
    }

    /**
     * Convenience: convert one page to Markdown. Equivalent to
     * {@link MarkdownConverter#toMarkdown(PdfDocument, int)}.
     */
    public String toMarkdown(int pageIndex) {
        return MarkdownConverter.toMarkdown(this, pageIndex);
    }

    /**
     * Convenience: convert this document to HTML. Equivalent to
     * {@link MarkdownConverter#toHtml(PdfDocument)}.
     */
    public String toHtml() {
        return MarkdownConverter.toHtml(this);
    }

    /**
     * Convenience: convert one page to HTML. Equivalent to
     * {@link MarkdownConverter#toHtml(PdfDocument, int)}.
     */
    public String toHtml(int pageIndex) {
        return MarkdownConverter.toHtml(this, pageIndex);
    }

    /**
     * Get a lightweight view of the page at {@code index}. The
     * returned {@link PdfPage} borrows from this document — it is
     * invalidated when this document is closed.
     *
     * @param index 0-based page index.
     * @throws IndexOutOfBoundsException if {@code index} is out of range.
     * @throws PdfInvalidStateException if this document has been closed.
     */
    public PdfPage page(int index) {
        if (index < 0 || index >= pageCount()) {
            throw new IndexOutOfBoundsException("page index " + index + " out of range [0, " + pageCount() + ")");
        }
        return new PdfPage(this, index);
    }

    /**
     * @return all pages as a {@link java.util.List} (eager — for the
     *         lazy {@link java.util.stream.Stream} variant see
     *         {@link #pagesStream()}, which is preferred for large docs).
     */
    public java.util.List<PdfPage> pages() {
        final int n = pageCount();
        java.util.ArrayList<PdfPage> pages = new java.util.ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            pages.add(new PdfPage(this, i));
        }
        return pages;
    }

    /**
     * @return all pages as a lazy {@link java.util.stream.Stream}.
     *         The stream borrows from this document — fully consume
     *         it before closing the document.
     */
    public java.util.stream.Stream<PdfPage> pagesStream() {
        final int n = pageCount();
        return java.util.stream.IntStream.range(0, n).mapToObj(i -> new PdfPage(this, i));
    }

    /**
     * Extract plain text for a single page.
     *
     * @param pageIndex 0-based page index.
     * @return the extracted text. Empty string if the page has no text.
     * @throws IndexOutOfBoundsException if {@code pageIndex} is out of range.
     * @throws PdfInvalidStateException if this document has been closed.
     */
    public String extractText(int pageIndex) {
        return nativeExtractText(checkHandle(), pageIndex);
    }

    /**
     * Extract the structured layout of a single page as a JSON string
     * (#536). The returned JSON is a serialized {@code StructuredPage}:
     * {@code {page_index, page_width, page_height, regions:[{kind, text,
     * bbox, spans, column_index}]}}.
     *
     * <p>Like {@link #extractText(int)}, this returns the raw payload
     * (here, JSON) rather than parsing it — callers may deserialize with
     * the JSON library of their choice. This keeps the binding free of a
     * JSON-parser dependency.
     *
     * @param page 0-based page index.
     * @return the structured page serialized as a JSON string.
     * @throws IndexOutOfBoundsException if {@code page} is out of range.
     * @throws PdfInvalidStateException if this document has been closed.
     */
    public String extractStructured(int page) {
        return nativeExtractStructured(checkHandle(), page);
    }

    /**
     * @return true if this document is still open (handle has not
     *         been freed). Useful for diagnostics; in normal code paths
     *         prefer the try-with-resources pattern.
     */
    public boolean isOpen() {
        return handleState.get() != 0L;
    }

    /**
     * Free the native handle. Idempotent — calling more than once is
     * a no-op, not a JVM crash. Safe to call from a finally block.
     */
    @Override
    public void close() {
        // Atomically zero the handle and capture the prior value.
        // Two concurrent close() calls cooperate: only the winner of
        // the CAS frees; the loser sees 0 and bails.
        final long h = handleState.getAndSet(0L);
        if (h == 0L) {
            return; // already closed
        }
        nativeClose(h);
        LIVE_HANDLES.decrementAndGet();
        // The cleaner now sees handleState == 0 and skips its free.
        // Still call clean() to deregister so it doesn't keep the
        // PhantomReference alive longer than necessary. clean() is
        // idempotent in the JDK Cleaner.
        cleanable.clean();
    }

    private long checkHandle() {
        final long h = handleState.get();
        if (h == 0L) {
            throw new PdfInvalidStateException("PdfDocument has been closed");
        }
        return h;
    }

    /**
     * Package-private accessor used by sibling classes in
     * {@code fyi.oxide.pdf.*} (MarkdownConverter, AutoExtractor,
     * PdfSigner, …) that need the raw handle to pass to their own
     * JNI methods. Same precondition as {@link #checkHandle()}.
     *
     * @throws PdfInvalidStateException if this document has been closed.
     */
    long requireHandleForCallers() {
        return checkHandle();
    }

    /** Test-only: how many handles are currently outstanding across the JVM. */
    static long liveHandleCount() {
        return LIVE_HANDLES.get();
    }

    private static byte[] readAll(InputStream s) throws IOException {
        // Java 9+ has InputStream.readAllBytes() — JDK 11 floor allows it.
        return s.readAllBytes();
    }

    /**
     * Cleaner action for leaked handles. Holds the **same**
     * {@link AtomicLong} state as the {@link PdfDocument} (not a
     * captured-by-value long), so when {@link #close()} CAS-zeroes
     * the state, the cleaner sees 0 and skips — preventing the
     * double-free that bit the empirical first run of this binding.
     *
     * <p>Holding a reference to {@code AtomicLong} (not to
     * {@code PdfDocument}) keeps the cleaner registration GC-correct:
     * the outer object can still be collected even though the
     * cleaner action is reachable. Standard Cleaner pattern.
     */
    private static final class HandleCleaner implements Runnable {
        private final AtomicLong state;

        HandleCleaner(AtomicLong state) {
            this.state = state;
        }

        @Override
        public void run() {
            // CAS — race-free with close() running concurrently.
            final long h = state.getAndSet(0L);
            if (h == 0L) {
                return; // close() already freed it
            }
            nativeClose(h);
            LIVE_HANDLES.decrementAndGet();
            System.err.println("[pdf_oxide] WARN: PdfDocument leaked — close() was not called. "
                    + "Use try-with-resources to manage document lifetime.");
        }
    }

    // ─────────────────────── native ────────────────────────

    private static native long nativeOpenPath(String path);

    private static native long nativeOpenBytes(byte[] bytes);

    private static native void nativeClose(long handle);

    private static native int nativePageCount(long handle);

    private static native String nativeExtractText(long handle, int pageIndex);

    private static native boolean nativeAuthenticate(long handle, byte[] password);

    private static native String nativeProducer(long handle);

    private static native String nativeCreator(long handle);

    private static native String nativeExtractTextAuto(long handle, int pageIndex);

    private static native String nativeExtractStructured(long handle, int pageIndex);

    private static native byte[] nativeRenderPng(long handle, int pageIndex, int dpi);

    private static native java.util.List<fyi.oxide.pdf.form.FormField> nativeFormFields(long handle);

    private static native java.util.List<fyi.oxide.pdf.search.SearchMatch> nativeSearch(
            long handle, String pattern, boolean caseInsensitive, boolean literal, int maxResults);
}
