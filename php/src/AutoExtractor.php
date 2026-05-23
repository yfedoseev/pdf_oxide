<?php

declare(strict_types=1);

namespace PdfOxide;

use PdfOxide\Enums\ExtractReason;
use PdfOxide\Enums\PageKind;
use PdfOxide\FFI\FunctionBindings;

/**
 * v0.3.51 #519 — auto-extraction with typed reasons.
 *
 * Mirrors the API surface of every other binding's `AutoExtractor`:
 *
 *   $extractor = new AutoExtractor();
 *   $result    = $extractor->extractPage($doc, 0);
 *   echo $result->text;
 *   if ($result->reason !== ExtractReason::Ok) {
 *       error_log("extraction degraded: " . $result->reason->value);
 *   }
 *
 * Graceful-fallback semantics
 * ---------------------------
 *
 * Extraction is NOT a security op (per
 * `feedback_extraction_graceful_fallback`): when the Rust side reports
 * an OCR-required state but {@see prefetchAvailable()} returns false,
 * this class returns the native text-layer with
 * {@see ExtractReason::OcrRequestedButUnavailable} instead of throwing.
 * That mirrors Python's `AutoExtractor` and Java's
 * `fyi.oxide.pdf.auto.AutoExtractor`.
 */
final class AutoExtractor
{
    private FunctionBindings $bindings;

    public function __construct()
    {
        $this->bindings = new FunctionBindings();
    }

    /**
     * Cheap per-page classification (no OCR, no rasterisation).
     * Returns the JSON envelope decoded to a typed {@see AutoExtractResult}.
     */
    public function classifyPage(PdfDocument $doc, int $pageIndex): AutoExtractResult
    {
        $json = $this->bindings->pdfDocumentClassifyPage($doc->getHandle(), $pageIndex);
        $decoded = self::decodeJson($json);

        $kind = PageKind::fromWire($decoded['kind'] ?? null);
        $reason = ExtractReason::fromWire($decoded['reason'] ?? null);
        $confidence = isset($decoded['confidence']) ? (float)$decoded['confidence'] : 0.0;

        return new AutoExtractResult(
            text: '',
            reason: $reason,
            kind: $kind,
            confidence: $confidence,
            classification: $decoded,
        );
    }

    /**
     * Whole-document classification — per-page kinds + a
     * `pages_needing_ocr` array. Returned as JSON envelope.
     */
    public function classifyDocument(PdfDocument $doc): array
    {
        $json = $this->bindings->pdfDocumentClassifyDocument($doc->getHandle());
        return self::decodeJson($json);
    }

    /**
     * One-shot auto text extraction (text-vs-OCR routing with graceful
     * native fallback). The page-level reason is derived from a follow-
     * up classification call (cheap; no second OCR pass).
     */
    public function extractText(PdfDocument $doc, int $pageIndex): AutoExtractResult
    {
        // The text-only call doesn't return a JSON envelope, so the
        // reason is inferred via the cheap classifier (Python does the
        // same thing).
        $text = $this->bindings->pdfDocumentExtractTextAuto($doc->getHandle(), $pageIndex);

        $reason = ExtractReason::Ok;
        $kind = PageKind::Mixed;
        $classification = null;
        try {
            $cls = $this->classifyPage($doc, $pageIndex);
            $reason = $cls->reason;
            $kind = $cls->kind;
            $classification = $cls->classification;
        } catch (\Throwable $e) {
            // Classification is best-effort; never let it mask extraction.
        }

        // Graceful-fallback hook: if the classifier wants OCR and the
        // build can't provide it, surface that as the reason regardless
        // of whether the native side already downgraded.
        if (! $this->bindings->pdfOxidePrefetchAvailable() && $kind === PageKind::Scanned) {
            $reason = ExtractReason::OcrRequestedButUnavailable;
        }

        return new AutoExtractResult(
            text: $text,
            reason: $reason,
            kind: $kind,
            confidence: 0.0,
            classification: $classification,
        );
    }

    /**
     * Rich per-page extraction — returns the full JSON `PageExtraction`
     * envelope (text + per-region bbox + reason + confidence + ocrUsed).
     *
     * @param array<string,mixed>|null $options AutoExtractOptions; null → defaults.
     */
    public function extractPage(PdfDocument $doc, int $pageIndex, ?array $options = null): AutoExtractResult
    {
        $optionsJson = $options === null ? null : json_encode($options, JSON_THROW_ON_ERROR);
        $json = $this->bindings->pdfDocumentExtractPageAuto($doc->getHandle(), $pageIndex, $optionsJson);
        $decoded = self::decodeJson($json);

        $kind = PageKind::fromWire($decoded['kind'] ?? null);
        $reason = ExtractReason::fromWire($decoded['reason'] ?? null);
        $confidence = isset($decoded['confidence']) ? (float)$decoded['confidence'] : 0.0;
        $text = (string)($decoded['text'] ?? '');

        return new AutoExtractResult(
            text: $text,
            reason: $reason,
            kind: $kind,
            confidence: $confidence,
            classification: $decoded,
        );
    }

    // ---------- Models subsystem (#519 provisioning trio) ----------

    /**
     * Provision OCR models for the given languages. Returns the cache
     * directory path; an empty string when the build was compiled
     * without the `ocr` feature.
     *
     * @param array<int,string> $languages e.g. ["eng", "rus"]
     */
    public function prefetchModels(array $languages): string
    {
        return $this->bindings->pdfOxidePrefetchModels(implode(',', $languages));
    }

    /**
     * Returns the model manifest as a decoded array. May be empty if
     * the build lacks the `ocr` feature.
     *
     * @return array<string,mixed>
     */
    public function modelManifest(): array
    {
        $json = $this->bindings->pdfOxideModelManifest();
        if ($json === '') {
            return [];
        }
        try {
            $decoded = json_decode($json, true, 512, JSON_THROW_ON_ERROR);
            return is_array($decoded) ? $decoded : [];
        } catch (\JsonException) {
            return [];
        }
    }

    /**
     * Whether the build supports OCR provisioning (and a model cache
     * appears reachable). Used by AutoExtractor's graceful-fallback
     * decision.
     */
    public function prefetchAvailable(): bool
    {
        return $this->bindings->pdfOxidePrefetchAvailable();
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
            $decoded = json_decode($json, true, 512, JSON_THROW_ON_ERROR);
            return is_array($decoded) ? $decoded : [];
        } catch (\JsonException) {
            return [];
        }
    }
}
