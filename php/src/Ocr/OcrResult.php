<?php

declare(strict_types=1);

namespace PdfOxide\Ocr;

/**
 * Aggregate OCR results from a page or document.
 *
 * Contains multiple OcrSpan objects and provides summary statistics
 * such as average confidence and total text.
 *
 * Example:
 *     $result = new OcrResult($spans);
 *     echo "Found " . $result->getCount() . " text spans";
 *     echo "Average confidence: " . ($result->getAverageConfidence() * 100) . "%";
 *     echo "Full text: " . $result->getText();
 *
 * @since 0.4.0
 */
class OcrResult
{
    /**
     * Create OcrResult from array of spans.
     *
     * @param OcrSpan[] $spans Array of OcrSpan objects
     */
    public function __construct(
        private readonly array $spans
    ) {}

    /**
     * Get number of text spans.
     *
     * @return int Count of spans in this result
     *
     * Example:
     *     $count = $result->getCount();
     */
    public function getCount(): int
    {
        return count($this->spans);
    }

    /**
     * Get span by index.
     *
     * @param int $index Span index (0-based)
     * @return OcrSpan The span at the given index
     * @throws \OutOfRangeException If index is out of bounds
     *
     * Example:
     *     $firstSpan = $result->getSpan(0);
     */
    public function getSpan(int $index): OcrSpan
    {
        if ($index < 0 || $index >= count($this->spans)) {
            throw new \OutOfRangeException("Span index out of bounds: {$index}");
        }

        return $this->spans[$index];
    }

    /**
     * Get all spans.
     *
     * @return OcrSpan[] All spans in this result
     *
     * Example:
     *     foreach ($result->getSpans() as $span) {
     *         echo $span->getText();
     *     }
     */
    public function getSpans(): array
    {
        return $this->spans;
    }

    /**
     * Get combined text from all spans.
     *
     * Concatenates text from all spans with space separators.
     *
     * @return string Combined text
     *
     * Example:
     *     $text = $result->getText();
     */
    public function getText(): string
    {
        $texts = array_map(fn(OcrSpan $span) => $span->getText(), $this->spans);
        return implode(' ', $texts);
    }

    /**
     * Get average confidence across all spans.
     *
     * Calculates the mean confidence score of all spans.
     *
     * @return float Average confidence (0.0-1.0)
     *
     * Example:
     *     $avgConfidence = $result->getAverageConfidence();
     *     echo "Average confidence: " . ($avgConfidence * 100) . "%";
     */
    public function getAverageConfidence(): float
    {
        if (count($this->spans) === 0) {
            return 0.0;
        }

        $sum = 0.0;
        foreach ($this->spans as $span) {
            $sum += $span->getConfidence();
        }

        return $sum / count($this->spans);
    }

    /**
     * Get minimum confidence among all spans.
     *
     * @return float|null Minimum confidence, or null if no spans
     *
     * Example:
     *     $minConfidence = $result->getMinConfidence();
     */
    public function getMinConfidence(): ?float
    {
        if (count($this->spans) === 0) {
            return null;
        }

        $confidences = array_map(fn(OcrSpan $span) => $span->getConfidence(), $this->spans);
        return min($confidences);
    }

    /**
     * Get maximum confidence among all spans.
     *
     * @return float|null Maximum confidence, or null if no spans
     *
     * Example:
     *     $maxConfidence = $result->getMaxConfidence();
     */
    public function getMaxConfidence(): ?float
    {
        if (count($this->spans) === 0) {
            return null;
        }

        $confidences = array_map(fn(OcrSpan $span) => $span->getConfidence(), $this->spans);
        return max($confidences);
    }

    /**
     * Filter spans by minimum confidence threshold.
     *
     * @param float $minConfidence Minimum confidence threshold (0.0-1.0)
     * @return OcrSpan[] Spans meeting the threshold
     * @throws \InvalidArgumentException If threshold is invalid
     *
     * Example:
     *     $highConfidence = $result->filterByConfidence(0.8);
     */
    public function filterByConfidence(float $minConfidence): array
    {
        if ($minConfidence < 0.0 || $minConfidence > 1.0) {
            throw new \InvalidArgumentException(
                "Confidence threshold must be 0.0-1.0, got {$minConfidence}"
            );
        }

        return array_filter(
            $this->spans,
            fn(OcrSpan $span) => $span->getConfidence() >= $minConfidence
        );
    }

    /**
     * Convert to array for serialization.
     *
     * @return array Array representation of result
     */
    public function toArray(): array
    {
        return [
            'count' => $this->getCount(),
            'text' => $this->getText(),
            'averageConfidence' => $this->getAverageConfidence(),
            'minConfidence' => $this->getMinConfidence(),
            'maxConfidence' => $this->getMaxConfidence(),
            'spans' => array_map(fn(OcrSpan $span) => $span->toArray(), $this->spans),
        ];
    }

    /**
     * Implement ArrayAccess-like behavior for spans.
     *
     * @return \ArrayIterator
     */
    public function getIterator(): \ArrayIterator
    {
        return new \ArrayIterator($this->spans);
    }

    /**
     * Count spans (implements Countable).
     *
     * @return int Number of spans
     */
    public function count(): int
    {
        return count($this->spans);
    }
}
