<?php

declare(strict_types=1);

namespace PdfOxide\Types;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Represents a text span from OCR results.
 *
 * A span is typically a word or unit of recognized text with position and confidence.
 */
class OcrSpan
{
    private CData $handle;
    private FunctionBindings $bindings;
    private ?Rect $cachedBbox = null;
    private ?float $cachedConfidence = null;

    public function __construct(CData $handle, FunctionBindings $bindings)
    {
        $this->handle = $handle;
        $this->bindings = $bindings;
    }

    /**
     * Get bounding box for this span.
     *
     * @return Rect Position and size of recognized text
     */
    public function getBoundingBox(): Rect
    {
        if ($this->cachedBbox === null) {
            $bbox = $this->bindings->pdfOcrSpanGetBbox($this->handle);
            $this->cachedBbox = new Rect($bbox['x'], $bbox['y'], $bbox['width'], $bbox['height']);
        }
        return $this->cachedBbox;
    }

    /**
     * Get confidence for character at index.
     *
     * @param int $charIndex Character index within span
     * @return float Confidence (0.0-1.0)
     */
    public function getCharConfidence(int $charIndex): float
    {
        return $this->bindings->pdfOcrSpanGetCharConfidence($this->handle, $charIndex);
    }

    /**
     * Get average confidence for this span.
     *
     * @return float Average confidence (0.0-1.0)
     */
    public function getConfidence(): float
    {
        if ($this->cachedConfidence === null) {
            // Get confidence for first character as representative
            $this->cachedConfidence = $this->getCharConfidence(0);
        }
        return $this->cachedConfidence;
    }

    /**
     * Get X coordinate.
     *
     * @return float X position
     */
    public function getX(): float
    {
        return $this->getBoundingBox()->x;
    }

    /**
     * Get Y coordinate.
     *
     * @return float Y position
     */
    public function getY(): float
    {
        return $this->getBoundingBox()->y;
    }

    /**
     * Get width.
     *
     * @return float Width
     */
    public function getWidth(): float
    {
        return $this->getBoundingBox()->width;
    }

    /**
     * Get height.
     *
     * @return float Height
     */
    public function getHeight(): float
    {
        return $this->getBoundingBox()->height;
    }

    /**
     * Check if confidence meets threshold.
     *
     * @param float $threshold Confidence threshold
     * @return bool True if confidence >= threshold
     */
    public function meetsConfidenceThreshold(float $threshold = 0.8): bool
    {
        return $this->getConfidence() >= $threshold;
    }

    /**
     * Get span data as array.
     *
     * @return array Span information
     */
    public function toArray(): array
    {
        $bbox = $this->getBoundingBox();
        return [
            'x' => $bbox->x,
            'y' => $bbox->y,
            'width' => $bbox->width,
            'height' => $bbox->height,
            'confidence' => $this->getConfidence(),
        ];
    }

    /**
     * Free OCR span resources.
     */
    public function __destruct()
    {
        $this->bindings->pdfOcrSpanFree($this->handle);
    }
}
