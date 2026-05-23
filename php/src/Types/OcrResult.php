<?php

declare(strict_types=1);

namespace PdfOxide\Types;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Represents OCR results from a page.
 *
 * Contains extracted text and character-level details with confidence scoring.
 */
class OcrResult
{
    private CData $handle;
    private FunctionBindings $bindings;
    private ?string $cachedText = null;
    private ?array $cachedSpans = null;
    private ?float $cachedConfidence = null;

    public function __construct(CData $handle, FunctionBindings $bindings)
    {
        $this->handle = $handle;
        $this->bindings = $bindings;
    }

    /**
     * Get extracted text.
     *
     * @return string OCR-recognized text from the page
     */
    public function getText(): string
    {
        if ($this->cachedText === null) {
            $this->cachedText = $this->bindings->pdfOcrExtractText($this->handle);
        }
        return $this->cachedText;
    }

    /**
     * Get number of text spans.
     *
     * A span is typically a word or unit of recognized text.
     *
     * @return int Number of spans
     */
    public function getSpanCount(): int
    {
        return $this->bindings->pdfOcrResultsCount($this->handle);
    }

    /**
     * Get text span by index.
     *
     * @param int $index Span index
     * @return OcrSpan Text span with confidence and position
     */
    public function getSpan(int $index): OcrSpan
    {
        $spanHandle = $this->bindings->pdfOcrResultsGetSpan($this->handle, $index);
        return new OcrSpan($spanHandle, $this->bindings);
    }

    /**
     * Get all text spans.
     *
     * @return OcrSpan[] Array of text spans
     */
    public function getSpans(): array
    {
        if ($this->cachedSpans === null) {
            $this->cachedSpans = [];
            $count = $this->getSpanCount();
            for ($i = 0; $i < $count; $i++) {
                $this->cachedSpans[] = $this->getSpan($i);
            }
        }
        return $this->cachedSpans;
    }

    /**
     * Get average confidence score.
     *
     * Confidence ranges from 0.0 (no confidence) to 1.0 (perfect confidence).
     *
     * @return float Average confidence (0.0-1.0)
     */
    public function getAverageConfidence(): float
    {
        if ($this->cachedConfidence === null) {
            $this->cachedConfidence = $this->bindings->pdfOcrResultsAverageConfidence($this->handle);
        }
        return $this->cachedConfidence;
    }

    /**
     * Check if recognition quality is high.
     *
     * @param float $threshold Confidence threshold (default 0.8 = 80%)
     * @return bool True if average confidence meets threshold
     */
    public function isHighQuality(float $threshold = 0.8): bool
    {
        return $this->getAverageConfidence() >= $threshold;
    }

    /**
     * Get confidence statistics.
     *
     * @return array Confidence data (min, max, avg)
     */
    public function getConfidenceStats(): array
    {
        $spans = $this->getSpans();

        if (empty($spans)) {
            return [
                'min' => 0.0,
                'max' => 0.0,
                'avg' => 0.0,
                'total_spans' => 0,
            ];
        }

        $confidences = [];
        foreach ($spans as $span) {
            $confidences[] = $span->getConfidence();
        }

        return [
            'min' => min($confidences),
            'max' => max($confidences),
            'avg' => array_sum($confidences) / count($confidences),
            'total_spans' => count($spans),
        ];
    }

    /**
     * Get OCR result as array.
     *
     * @return array Result data
     */
    public function toArray(): array
    {
        return [
            'text' => $this->getText(),
            'span_count' => $this->getSpanCount(),
            'average_confidence' => $this->getAverageConfidence(),
            'confidence_stats' => $this->getConfidenceStats(),
        ];
    }

    /**
     * Free OCR results resources.
     */
    public function __destruct()
    {
        $this->bindings->pdfOcrResultsFree($this->handle);
    }
}
