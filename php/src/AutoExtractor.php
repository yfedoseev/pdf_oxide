<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide;

use PdfOxide\FFI\FunctionBindings;

/**
 * The v0.3.51 typed-reason, graceful-fallback auto-extractor.
 *
 * Mirrors `fyi.oxide.pdf.AutoExtractor`. Given a {@see PdfDocument},
 * returns all recoverable text (native AND OCR), per-page, with a
 * typed reason naming every degraded result. When OCR is unavailable,
 * gracefully falls back to the native text layer with a logged warning
 * — never silent-empty, never throws (the
 * {@code feedback_extraction_graceful_fallback} contract).
 *
 * Construct once per document via {@see of()} or a preset factory
 * (`fast`/`balanced`/`highFidelity`).
 *
 * Status (v0.3.55): the simplified surface returns an
 * {@see AutoExtractResult} for per-page/per-document text. The rich
 * JSON envelope (typed reasons + per-region bbox + confidence per
 * region) is reachable via {@see extractPageJson()} /
 * {@see extractDocumentJson()}.
 */
final class AutoExtractor
{
    /** Mode constants — wire ints match the Rust ExtractMode enum. */
    public const MODE_AUTO = 0;

    public const MODE_TEXT_ONLY = 1;

    public const MODE_FORCE_OCR = 2;

    private readonly FunctionBindings $bindings;

    private function __construct(
        private readonly PdfDocument $doc,
        private readonly int $mode,
    ) {
        $this->bindings = new FunctionBindings();
    }

    // ────────────────────── factories ──────────────────────

    /** Construct with `mode=AUTO` (default). */
    public static function of(PdfDocument $doc, int $mode = self::MODE_AUTO): self
    {
        return new self($doc, $mode);
    }

    /** Preset: prioritises speed over accuracy (no OCR). */
    public static function fast(PdfDocument $doc): self
    {
        return new self($doc, self::MODE_TEXT_ONLY);
    }

    /** Preset: default; OCR auto-routed; image-tables reconstructed. */
    public static function balanced(PdfDocument $doc): self
    {
        return new self($doc, self::MODE_AUTO);
    }

    /** Preset: forces OCR on every page; slowest but most thorough. */
    public static function highFidelity(PdfDocument $doc): self
    {
        return new self($doc, self::MODE_FORCE_OCR);
    }

    // ─────────────── plain-text extraction ─────────────────

    /**
     * Extract the entire document as plain text via the v0.3.51
     * graceful auto-routing path. Concatenates per-page output with a
     * newline between pages.
     */
    public function extractText(): string
    {
        $n = $this->doc->pageCount();
        $out = '';
        for ($i = 0; $i < $n; ++$i) {
            if ($i > 0) {
                $out .= "\n";
            }
            $out .= $this->doc->extractTextAuto($i);
        }
        return $out;
    }

    /** Extract a single page's text via the auto-routing path. */
    public function extractTextForPage(int $pageIndex): string
    {
        $this->boundsCheck($pageIndex);
        return $this->doc->extractTextAuto($pageIndex);
    }

    // ─────────────── typed AutoExtractResult ───────────────

    /**
     * Extract a single page as a simplified {@see AutoExtractResult}.
     *
     * Limitation (parity with Java v0.3.53 surface): the returned
     * value carries text + {@code OK} reason + confidence=1.0 +
     * empty regions list. For the rich envelope (per-region bbox +
     * confidence per region) use {@see extractPageJson()}.
     */
    public function extractAutoPage(int $pageIndex): AutoExtractResult
    {
        $this->boundsCheck($pageIndex);
        $text = $this->doc->extractTextAuto($pageIndex);
        return new AutoExtractResult(
            text: $text,
            reason: AutoExtractResult::REASON_OK,
            confidence: 1.0,
            ocrUsed: false,
            regions: [],
            pagesNeedingOcr: [],
        );
    }

    /** Whole-document simplified extraction. */
    public function extractAutoDocument(): AutoExtractResult
    {
        return new AutoExtractResult(
            text: $this->extractText(),
            reason: AutoExtractResult::REASON_OK,
            confidence: 1.0,
            ocrUsed: false,
            regions: [],
            pagesNeedingOcr: [],
        );
    }

    // ─────────────── classification ────────────────────────

    /**
     * Cheap per-page classification — no OCR, no rasterisation. The
     * page's kind one of {@see AutoExtractResult::KIND_*}.
     */
    public function classifyPageKind(int $pageIndex): string
    {
        $this->boundsCheck($pageIndex);
        $json = $this->bindings->pdfDocumentClassifyPage($this->doc->getHandle(), $pageIndex);
        $decoded = self::decodeJson($json);
        return (string) ($decoded['kind'] ?? AutoExtractResult::KIND_MIXED);
    }

    /**
     * Classify every page; returns a list of per-page kinds.
     *
     * The Rust side serialises the document classification as
     * `{"pages": ["text_layer", "scanned", ...], ...}` (a flat list of
     * kind strings, not an array of objects); we honour both shapes
     * for forward-compat.
     *
     * @return array<int, string>
     */
    public function classifyDocumentKinds(): array
    {
        $json = $this->bindings->pdfDocumentClassifyDocument($this->doc->getHandle());
        $decoded = self::decodeJson($json);
        $pages = $decoded['pages'] ?? [];
        if (! is_array($pages)) {
            return [];
        }
        $out = [];
        foreach ($pages as $p) {
            if (is_string($p)) {
                $out[] = $p;
            } elseif (is_array($p) && isset($p['kind']) && is_string($p['kind'])) {
                $out[] = $p['kind'];
            } else {
                $out[] = AutoExtractResult::KIND_MIXED;
            }
        }
        return $out;
    }

    // ─────────────── rich JSON escape hatch ────────────────

    /**
     * Rich per-page extraction serialised as JSON. The binding
     * intentionally does NOT impose a JSON-decoder choice on the
     * consumer — parse with `json_decode()` or whatever's idiomatic.
     *
     * JSON shape (v0.3.51 PageExtraction):
     *   {page, kind, text, regions:[{bbox,text,reason,confidence,ocr_used,...}],
     *    confidence, reason, ocr_used, pages_needing_ocr}
     */
    public function extractPageJson(int $pageIndex): string
    {
        $this->boundsCheck($pageIndex);
        // pdfDocumentExtractPageAuto wraps the rich envelope.
        return $this->bindings->pdfDocumentExtractPageAuto($this->doc->getHandle(), $pageIndex, null);
    }

    /** Rich whole-document extraction as JSON. */
    public function extractDocumentJson(): string
    {
        return $this->bindings->pdfDocumentClassifyDocument($this->doc->getHandle());
    }

    // ─────────────── accessors ─────────────────────────────

    public function document(): PdfDocument
    {
        return $this->doc;
    }

    public function mode(): int
    {
        return $this->mode;
    }

    /**
     * @return array<string,mixed>
     */
    private static function decodeJson(string $json): array
    {
        if ($json === '') {
            return [];
        }
        try {
            $decoded = json_decode($json, true, 512, \JSON_THROW_ON_ERROR);
            return is_array($decoded) ? $decoded : [];
        } catch (\JsonException) {
            return [];
        }
    }

    private function boundsCheck(int $pageIndex): void
    {
        $n = $this->doc->pageCount();
        if ($pageIndex < 0 || $pageIndex >= $n) {
            throw new \OutOfRangeException("page index {$pageIndex} out of range [0, {$n})");
        }
    }
}
