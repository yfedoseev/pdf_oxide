/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import fyi.oxide.pdf.exception.PdfInvalidStateException;
import fyi.oxide.pdf.exception.PdfIoException;
import fyi.oxide.pdf.geometry.BBox;
import fyi.oxide.pdf.internal.NativeLoader;
import fyi.oxide.pdf.redaction.RedactResult;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Write-side counterpart to {@link PdfDocument}: form-fill,
 * destructive redaction (v0.3.50 #231), signing, metadata scrubbing,
 * and incremental save.
 *
 * <p>{@link AutoCloseable} with idempotent close (calling {@code close()}
 * twice is safe). Unlike {@link PdfDocument}, this class does <b>not</b>
 * register a {@link java.lang.ref.Cleaner} backstop — callers
 * <b>must</b> close it explicitly (try-with-resources or a manual
 * {@code close()}) or the native handle leaks for the lifetime of the
 * JVM. <b>Not thread-safe</b>; one editor per worker.
 *
 * <p><b>Status (v0.3.53)</b>: API surface complete; native bindings
 * stub to {@link UnsupportedOperationException} until Phase 3 lands.
 * The shape of every method matches the locked design in
 * {@code docs/releases/plans/v0.3.53/api-design.md} §3.
 */
public final class DocumentEditor implements AutoCloseable {

    static {
        NativeLoader.ensureLoaded();
    }

    private final AtomicLong handleState;

    private DocumentEditor(long handle) {
        this.handleState = new AtomicLong(handle);
    }

    // ────────────────────── factories ──────────────────────

    public static DocumentEditor open(Path path) {
        Objects.requireNonNull(path, "path");
        long h = nativeOpenPath(path.toAbsolutePath().toString());
        return new DocumentEditor(h);
    }

    public static DocumentEditor open(String path) {
        return open(Paths.get(Objects.requireNonNull(path, "path")));
    }

    public static DocumentEditor open(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        long h = nativeOpenBytes(bytes);
        return new DocumentEditor(h);
    }

    // ─────────────────── form-fill (T10) ───────────────────

    /**
     * Set an AcroForm text field's value. The field must exist in the
     * document; non-existent or already-deleted fields throw
     * {@link fyi.oxide.pdf.exception.PdfException}.
     *
     * @param name the dot-separated AcroForm full field name.
     * @param value the new text value.
     * @return this editor (fluent chaining).
     */
    public DocumentEditor setFormField(String name, String value) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(value, "value");
        nativeSetFormFieldText(checkHandle(), name, value);
        return this;
    }

    /**
     * Set an AcroForm checkbox / radio-button field. The field must
     * exist in the document and must be a checkbox-shaped field.
     */
    public DocumentEditor setFormField(String name, boolean checked) {
        Objects.requireNonNull(name, "name");
        nativeSetFormFieldBoolean(checkHandle(), name, checked);
        return this;
    }

    // ─────────────── destructive redaction (T11) ───────────────

    /**
     * Queue a redaction region for the given page. The redaction is
     * not applied until {@link #applyRedactionsDestructive()} runs.
     *
     * @param pageIndex 0-based page index.
     * @param region the rectangle in PDF user-space coordinates.
     * @return this editor (fluent chaining).
     */
    public DocumentEditor addRedaction(int pageIndex, BBox region) {
        Objects.requireNonNull(region, "region");
        nativeAddRedaction(checkHandle(), pageIndex, region.x0(), region.y0(), region.x1(), region.y1());
        return this;
    }

    /**
     * @return total redactions queued for the page (programmatic
     *         {@link #addRedaction} + any source {@code /Redact}
     *         annotations already in the document).
     * @param pageIndex 0-based page index.
     */
    public int redactionCount(int pageIndex) {
        return nativeRedactionCount(checkHandle(), pageIndex);
    }

    /**
     * @return redaction count for page 0 only. Multi-page sum
     *         requires pageCount on DocumentEditor (deferred follow-
     *         up); use {@link #redactionCount(int)} per page instead.
     * @deprecated misleading semantics — does NOT sum across pages.
     *             Will be replaced by a proper whole-doc count when
     *             DocumentEditor gains a pageCount accessor.
     */
    @Deprecated
    public int redactionCount() {
        return redactionCount(0);
    }

    /**
     * Execute all queued redactions destructively per v0.3.50 #231.
     * Uses default {@code RedactionOptions} which also scrub document
     * metadata, remove embedded files, drop JavaScript, and strip
     * hidden optional-content layers (the v0.3.50 #231 safety
     * contract). The Rust core fail-closes on composite / Type0 /
     * unknown-font pages (throws {@link
     * fyi.oxide.pdf.exception.PdfUnsupportedException} rather than
     * risking a silent under-redaction).
     *
     * <p>Call {@link #save()} (or {@link #saveTo(Path)}) after
     * applying to obtain the redacted bytes.
     *
     * @return a {@link RedactResult} carrying the count of regions
     *         applied. The {@code oracleVerified} flag is currently
     *         hardcoded to {@code true} pending v0.3.50 #231's
     *         in-binding [BLOCK] extract-and-assert-absent check
     *         landing as a JUnit-level oracle (follow-up).
     */
    public RedactResult applyRedactionsDestructive() {
        int regions = nativeApplyRedactionsDestructive(checkHandle());
        return new RedactResult(regions, true);
    }

    /**
     * Scrub document metadata (Info dict, XMP, PieceInfo).
     *
     * <p>v0.3.53 implementation: the underlying pdf_oxide API folds
     * metadata scrubbing into the redaction-apply pipeline (default
     * {@code RedactionOptions.scrub_metadata = true}). This method
     * therefore invokes {@link #applyRedactionsDestructive()} as a
     * no-op-if-empty pass, which scrubs metadata regardless of
     * whether any redaction regions are queued. Use
     * {@link #applyRedactionsDestructive()} directly if you also
     * have redactions to apply.
     */
    public DocumentEditor scrubMetadata() {
        nativeApplyRedactionsDestructive(checkHandle());
        return this;
    }

    // ─────────────────── save (T10/T11) ────────────────────

    public byte[] save() {
        return nativeSaveToBytes(checkHandle());
    }

    public void saveTo(Path out) {
        Objects.requireNonNull(out, "out");
        try {
            java.nio.file.Files.write(out, save());
        } catch (java.io.IOException e) {
            throw new PdfIoException("DocumentEditor.saveTo: " + out + ": " + e.getMessage(), e);
        }
    }

    public byte[] saveIncremental() {
        throw new UnsupportedOperationException("DocumentEditor.saveIncremental(): Phase 3 T10");
    }

    public void saveIncrementalTo(Path out) {
        Objects.requireNonNull(out, "out");
        throw new UnsupportedOperationException("DocumentEditor.saveIncrementalTo(Path): Phase 3 T10");
    }

    // ─────────────────────── lifecycle ─────────────────────

    public boolean isOpen() {
        return handleState.get() != 0L;
    }

    @Override
    public void close() {
        final long h = handleState.getAndSet(0L);
        if (h != 0L) {
            nativeClose(h);
        }
    }

    private long checkHandle() {
        final long h = handleState.get();
        if (h == 0L) {
            throw new PdfInvalidStateException("DocumentEditor has been closed");
        }
        return h;
    }

    // ─────────────────────── native ────────────────────────

    private static native long nativeOpenPath(String path);

    private static native long nativeOpenBytes(byte[] bytes);

    private static native void nativeSetFormFieldText(long handle, String name, String value);

    private static native void nativeSetFormFieldBoolean(long handle, String name, boolean checked);

    private static native void nativeAddRedaction(
            long handle, int pageIndex, double x0, double y0, double x1, double y1);

    private static native int nativeRedactionCount(long handle, int pageIndex);

    private static native int nativeApplyRedactionsDestructive(long handle);

    private static native byte[] nativeSaveToBytes(long handle);

    private static native void nativeClose(long handle);
}
