/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import fyi.oxide.pdf.auto.AutoExtractConfig;
import fyi.oxide.pdf.auto.AutoResult;
import fyi.oxide.pdf.auto.ClassifyResult;
import fyi.oxide.pdf.internal.NativeLoader;
import java.util.Objects;

/**
 * The v0.3.51 typed-reason, graceful-fallback auto-extractor.
 *
 * <p>Given any {@link PdfDocument}, returns all recoverable text
 * (native AND OCR), per-page/per-region, with a typed
 * {@link fyi.oxide.pdf.auto.ExtractReason} naming every degraded
 * result. When OCR is unavailable, falls back to the native text
 * layer with a logged warning — never silent-empty, never throws
 * (the {@code feedback_extraction_graceful_fallback} contract).
 *
 * <p>Constructed once per (doc, config) pair via {@link #of(PdfDocument)}
 * or a preset factory ({@link #fast}/{@link #balanced}/{@link #highFidelity}).
 * Re-use the same {@code AutoExtractor} across many extractions on
 * the same document to amortise model-loading cost.
 *
 * <p><b>Status (v0.3.53)</b>: API surface is complete; the native
 * side is stubbed until the JSON-envelope wire format from v0.3.51's
 * C-ABI is plumbed through (Phase 2 T9). Calling any method on a
 * v0.3.53 build throws {@link UnsupportedOperationException} for now.
 */
public final class AutoExtractor {

    static {
        NativeLoader.ensureLoaded();
    }

    /** Owning document (lifetime-bound). */
    private final PdfDocument doc;
    /** Configured behaviour. */
    private final AutoExtractConfig config;

    private AutoExtractor(PdfDocument doc, AutoExtractConfig config) {
        this.doc = Objects.requireNonNull(doc, "doc");
        this.config = Objects.requireNonNull(config, "config");
    }

    /** Construct with default config (mode=AUTO, all margins at zero). */
    public static AutoExtractor of(PdfDocument doc) {
        return new AutoExtractor(doc, AutoExtractConfig.DEFAULT);
    }

    /** Construct with the supplied config. */
    public static AutoExtractor of(PdfDocument doc, AutoExtractConfig config) {
        return new AutoExtractor(doc, config);
    }

    /** Preset: prioritises speed over accuracy (no OCR, no image-tables). */
    public static AutoExtractor fast(PdfDocument doc) {
        return of(
                doc,
                AutoExtractConfig.builder()
                        .withMode(fyi.oxide.pdf.auto.ExtractMode.TEXT_ONLY)
                        .build());
    }

    /** Preset: default; OCR auto-routed; image-tables reconstructed. */
    public static AutoExtractor balanced(PdfDocument doc) {
        return of(
                doc,
                AutoExtractConfig.builder()
                        .withMode(fyi.oxide.pdf.auto.ExtractMode.AUTO)
                        .build());
    }

    /** Preset: forces OCR on every page; slowest but most thorough. */
    public static AutoExtractor highFidelity(PdfDocument doc) {
        return of(
                doc,
                AutoExtractConfig.builder()
                        .withMode(fyi.oxide.pdf.auto.ExtractMode.FORCE_OCR)
                        .build());
    }

    /**
     * Extract the entire document as plain text via the v0.3.51
     * graceful auto-routing path (text-layer where present, OCR for
     * scanned regions when the {@code ocr} feature is available,
     * graceful fallback otherwise). Concatenates per-page output
     * with a single newline between pages.
     *
     * <p>v0.3.53 surface: returns plain {@code String}. The richer
     * {@link AutoResult} with typed reasons + per-region regions +
     * confidence lands via the JSON-envelope follow-up.
     */
    public String extractText() {
        int n = doc.pageCount();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            if (i > 0) sb.append('\n');
            sb.append(doc.extractTextAuto(i));
        }
        return sb.toString();
    }

    /** Extract a single page's text via the auto-routing path. */
    public String extractTextForPage(int pageIndex) {
        if (pageIndex < 0 || pageIndex >= doc.pageCount()) {
            throw new IndexOutOfBoundsException("page " + pageIndex + " out of [0, " + doc.pageCount() + ")");
        }
        return doc.extractTextAuto(pageIndex);
    }

    /**
     * Extract a single page as a simplified {@link AutoResult}.
     *
     * <p><b>v0.3.53 limitation</b>: this surface returns text +
     * {@link fyi.oxide.pdf.auto.ExtractReason#OK} + confidence=1.0 +
     * ocrUsed=false + empty regions list + empty pagesNeedingOcr.
     * The full AutoResult with typed reasons per region + bbox +
     * confidence per region needs the JSON-envelope wire format
     * (v0.3.51 #517) which is a follow-up.
     *
     * <p>If OCR fallback was triggered, the underlying
     * {@link PdfDocument#extractTextAuto(int)} call still returns
     * the native text content (per v0.3.51 graceful-fallback
     * contract) — but this simplified surface doesn't surface that
     * via {@code reason=FALLBACK_FROM_OCR}; for that, use
     * {@link #extractTextForPage(int)} and check the {@code ocr}
     * feature was enabled at build time.
     */
    public AutoResult extractAutoPage(int pageIndex) {
        if (pageIndex < 0 || pageIndex >= doc.pageCount()) {
            throw new IndexOutOfBoundsException("page " + pageIndex + " out of [0, " + doc.pageCount() + ")");
        }
        String text = doc.extractTextAuto(pageIndex);
        return new AutoResult(
                text,
                null, // markdown
                null, // html
                fyi.oxide.pdf.auto.ExtractReason.OK,
                1.0, // confidence
                false, // ocrUsed
                java.util.Collections.emptyList(), // regions
                java.util.Collections.emptyList()); // pagesNeedingOcr
    }

    /**
     * Whole-document simplified {@link AutoResult}. See
     * {@link #extractAutoPage(int)} for the per-page surface and
     * v0.3.53 limitations.
     */
    public AutoResult extractAutoDocument() {
        return new AutoResult(
                extractText(),
                null,
                null,
                fyi.oxide.pdf.auto.ExtractReason.OK,
                1.0,
                false,
                java.util.Collections.emptyList(),
                java.util.Collections.emptyList());
    }

    /**
     * Extract the entire document as a typed {@link AutoResult} with
     * typed-reason regions, per-region bboxes, confidence, and the
     * pages-needing-ocr list. Delegates to {@link #extractDocumentJson()}
     * + parses via {@code org.json}.
     */
    public AutoResult extractDocument() {
        String json = extractDocumentJson();
        return parseDocumentExtraction(new org.json.JSONObject(json));
    }

    /** Extract a single page as a typed {@link AutoResult}. */
    public AutoResult extractPage(int pageIndex) {
        if (pageIndex < 0 || pageIndex >= doc.pageCount()) {
            throw new IndexOutOfBoundsException("page " + pageIndex + " out of [0, " + doc.pageCount() + ")");
        }
        String json = extractPageJson(pageIndex);
        return parsePageExtraction(new org.json.JSONObject(json), pageIndex);
    }

    // ────────────────────── JSON parsing helpers ──────────────────────

    /** Parse a serde-serialized v0.3.51 PageExtraction. */
    static AutoResult parsePageExtraction(org.json.JSONObject obj, int pageIndex) {
        String text = obj.optString("text", "");
        double confidence = obj.optDouble("confidence", 1.0);
        boolean ocrUsed = obj.optBoolean("ocr_used", false);
        fyi.oxide.pdf.auto.ExtractReason reason = parseReason(obj.optString("reason", "ok"));
        java.util.List<fyi.oxide.pdf.auto.RegionResult> regions = new java.util.ArrayList<>();
        org.json.JSONArray rArr = obj.optJSONArray("regions");
        if (rArr != null) {
            for (int i = 0; i < rArr.length(); i++) {
                regions.add(parseRegion(rArr.getJSONObject(i), pageIndex));
            }
        }
        return new AutoResult(
                text, null, null, reason, confidence, ocrUsed, regions, java.util.Collections.emptyList());
    }

    /** Parse a serde-serialized v0.3.51 DocumentExtraction. */
    static AutoResult parseDocumentExtraction(org.json.JSONObject obj) {
        StringBuilder text = new StringBuilder();
        java.util.List<fyi.oxide.pdf.auto.RegionResult> allRegions = new java.util.ArrayList<>();
        java.util.List<Integer> pagesNeedingOcr = new java.util.ArrayList<>();
        boolean anyOcrUsed = false;
        double minConfidence = 1.0;
        fyi.oxide.pdf.auto.ExtractReason worstReason = fyi.oxide.pdf.auto.ExtractReason.OK;
        org.json.JSONArray pages = obj.optJSONArray("pages");
        if (pages != null) {
            for (int i = 0; i < pages.length(); i++) {
                org.json.JSONObject p = pages.getJSONObject(i);
                int pageIdx = p.optInt("page", i);
                if (text.length() > 0) text.append('\n');
                text.append(p.optString("text", ""));
                org.json.JSONArray rArr = p.optJSONArray("regions");
                if (rArr != null) {
                    for (int j = 0; j < rArr.length(); j++) {
                        allRegions.add(parseRegion(rArr.getJSONObject(j), pageIdx));
                    }
                }
                anyOcrUsed |= p.optBoolean("ocr_used", false);
                double pc = p.optDouble("confidence", 1.0);
                if (pc < minConfidence) minConfidence = pc;
                fyi.oxide.pdf.auto.ExtractReason pr = parseReason(p.optString("reason", "ok"));
                if (pr != fyi.oxide.pdf.auto.ExtractReason.OK && worstReason == fyi.oxide.pdf.auto.ExtractReason.OK) {
                    worstReason = pr;
                }
            }
        }
        org.json.JSONArray needing = obj.optJSONArray("pages_needing_ocr");
        if (needing != null) {
            for (int i = 0; i < needing.length(); i++) {
                pagesNeedingOcr.add(needing.getInt(i));
            }
        }
        return new AutoResult(
                text.toString(), null, null, worstReason, minConfidence, anyOcrUsed, allRegions, pagesNeedingOcr);
    }

    private static fyi.oxide.pdf.auto.RegionResult parseRegion(org.json.JSONObject r, int pageIdx) {
        org.json.JSONObject b = r.optJSONObject("bbox");
        fyi.oxide.pdf.geometry.BBox bbox = b == null
                ? new fyi.oxide.pdf.geometry.BBox(0, 0, 0, 0)
                : new fyi.oxide.pdf.geometry.BBox(
                        b.optDouble("x", 0),
                        b.optDouble("y", 0),
                        b.optDouble("x", 0) + b.optDouble("width", 0),
                        b.optDouble("y", 0) + b.optDouble("height", 0));
        return new fyi.oxide.pdf.auto.RegionResult(
                pageIdx,
                bbox,
                r.optString("text", ""),
                parseReason(r.optString("reason", "ok")),
                r.optDouble("confidence", 1.0),
                r.optBoolean("ocr_used", false),
                null);
    }

    private static fyi.oxide.pdf.auto.ExtractReason parseReason(String s) {
        try {
            return fyi.oxide.pdf.auto.ExtractReason.valueOf(s.toUpperCase(java.util.Locale.ROOT));
        } catch (IllegalArgumentException ignored) {
            return fyi.oxide.pdf.auto.ExtractReason.OK;
        }
    }

    /** Classify the entire document (cheap preflight). */
    public ClassifyResult classifyDocument() {
        throw new UnsupportedOperationException(
                "AutoExtractor.classifyDocument: native wiring lands in Phase 2 T9 follow-up");
    }

    /**
     * Classify a single page — quick preflight that decides whether
     * OCR routing is needed. Returns the page's
     * {@link fyi.oxide.pdf.auto.PageClass}.
     *
     * <p>v0.3.53 surface: simplified single-value return. The full
     * {@link ClassifyResult} with confidence / typed reason / signals
     * lands in a follow-up via the v0.3.51 JSON-envelope wire format.
     */
    public fyi.oxide.pdf.auto.PageClass classifyPageKind(int pageIndex) {
        int ordinal = nativeClassifyPageOrdinal(doc.requireHandleForCallers(), pageIndex);
        return fyi.oxide.pdf.auto.PageClass.values()[ordinal];
    }

    /** @deprecated v0.3.53 ships {@link #classifyPageKind} as a simpler returning the enum. */
    @Deprecated
    public ClassifyResult classifyPage(int pageIndex) {
        throw new UnsupportedOperationException(
                "AutoExtractor.classifyPage: ClassifyResult marshaller pending — use classifyPageKind(int) for the v0.3.53 simplified surface");
    }

    /**
     * Classify every page in the document; returns a per-page
     * {@link fyi.oxide.pdf.auto.PageClass} list.
     *
     * <p>v0.3.53 surface: simplified list return. The richer
     * {@link ClassifyResult} (with pagesNeedingOcr / pagesWithChart /
     * pagesEncrypted sublists) lands via the JSON-envelope follow-up.
     */
    public java.util.List<fyi.oxide.pdf.auto.PageClass> classifyDocumentKinds() {
        int[] ords = nativeClassifyDocumentOrdinals(doc.requireHandleForCallers());
        fyi.oxide.pdf.auto.PageClass[] all = fyi.oxide.pdf.auto.PageClass.values();
        java.util.List<fyi.oxide.pdf.auto.PageClass> out = new java.util.ArrayList<>(ords.length);
        for (int o : ords) {
            out.add(all[o]);
        }
        return out;
    }

    /**
     * Escape-hatch: rich per-page extraction serialized as JSON. The
     * binding intentionally does NOT impose a JSON parser on the
     * consumer — parse with your preferred library (org.json,
     * jackson, gson, etc.).
     *
     * <p>JSON shape (v0.3.51 {@code PageExtraction}):
     * {@code {page, kind, text, regions:[{bbox, text, reason,
     * confidence, ocr_used, ...}], confidence, reason, ocr_used,
     * pages_needing_ocr}}.
     */
    public String extractPageJson(int pageIndex) {
        if (pageIndex < 0 || pageIndex >= doc.pageCount()) {
            throw new IndexOutOfBoundsException("page " + pageIndex + " out of [0, " + doc.pageCount() + ")");
        }
        return nativeExtractPageJson(doc.requireHandleForCallers(), pageIndex);
    }

    /** Escape-hatch: rich whole-document extraction as JSON. See {@link #extractPageJson(int)}. */
    public String extractDocumentJson() {
        return nativeExtractDocumentJson(doc.requireHandleForCallers());
    }

    private static native int nativeClassifyPageOrdinal(long handle, int pageIndex);

    private static native int[] nativeClassifyDocumentOrdinals(long handle);

    private static native String nativeExtractPageJson(long handle, int pageIndex);

    private static native String nativeExtractDocumentJson(long handle);

    /** @return the configured doc (read-only accessor). */
    public PdfDocument document() {
        return doc;
    }
    /** @return the configured behaviour. */
    public AutoExtractConfig config() {
        return config;
    }
}
