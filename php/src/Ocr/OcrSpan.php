<?php

declare(strict_types=1);

namespace PdfOxide\Ocr;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * A span of OCR-recognized text with position and confidence.
 *
 * Represents a single word or text segment recognized by OCR, including
 * its bounding box coordinates, confidence score, and optional per-character
 * confidence values.
 *
 * Example:
 *     $span = new OcrSpan($spanHandle, $bindings);
 *     echo "Text: " . $span->getText();
 *     echo "Confidence: " . ($span->getConfidence() * 100) . "%";
 *     [$x, $y, $w, $h] = $span->getBbox();
 *     echo "Position: ({$x}, {$y}) Size: ({$w}x{$h})";
 *
 * @since 0.4.0
 */
class OcrSpan
{
    private string $text;
    private array $bbox;
    private float $confidence;
    private ?array $charConfidences = null;

    /**
     * Create OcrSpan from FFI handle.
     *
     * Extracts all span data from the native handle immediately, then
     * frees the handle. Immutable after construction.
     *
     * @param CData $spanHandle FFI span handle
     * @param FunctionBindings $bindings Function bindings for FFI calls
     * @throws RuntimeException If data extraction fails
     */
    public function __construct(CData $spanHandle, FunctionBindings $bindings)
    {
        try {
            // Extract all data from handle
            $this->text = $bindings->pdfOcrSpanGetText($spanHandle);
            $this->bbox = $bindings->pdfOcrSpanGetBbox($spanHandle);
            $this->confidence = $bindings->pdfOcrSpanGetConfidence($spanHandle);

            // Extract per-character confidences if available
            $this->charConfidences = $this->extractCharConfidences($spanHandle, $bindings);

            // Free the handle
            $bindings->pdfOcrSpanFree($spanHandle);
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to create OcrSpan from handle: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Get recognized text content.
     *
     * @return string The text content of this span
     *
     * Example:
     *     $text = $span->getText();
     */
    public function getText(): string
    {
        return $this->text;
    }

    /**
     * Get bounding box coordinates.
     *
     * Returns the bounding box as an associative array with coordinates
     * in PDF point units (72 points per inch).
     *
     * @return array Associative array with keys: 'x', 'y', 'width', 'height'
     *
     * Example:
     *     $bbox = $span->getBbox();
     *     echo "Position: ({$bbox['x']}, {$bbox['y']})";
     *     echo "Size: {$bbox['width']}x{$bbox['height']}";
     */
    public function getBbox(): array
    {
        return $this->bbox;
    }

    /**
     * Get overall recognition confidence.
     *
     * Returns a confidence score between 0.0 (no confidence) and 1.0 (certain).
     *
     * @return float Confidence score (0.0-1.0)
     *
     * Example:
     *     $confidence = $span->getConfidence();
     *     if ($confidence < 0.7) {
     *         echo "Low confidence text: {$span->getText()}";
     *     }
     */
    public function getConfidence(): float
    {
        return $this->confidence;
    }

    /**
     * Get per-character confidence scores.
     *
     * Returns an array of confidence scores, one for each character in the
     * recognized text. Each score is between 0.0 and 1.0.
     *
     * @return float[]|null Array of per-character confidence scores, or null if not available
     *
     * Example:
     *     $charConfidences = $span->getCharConfidences();
     *     if ($charConfidences !== null) {
     *         foreach ($charConfidences as $index => $confidence) {
     *             echo $text[$index] . ": {$confidence}";
     *         }
     *     }
     */
    public function getCharConfidences(): ?array
    {
        return $this->charConfidences;
    }

    /**
     * Get confidence for a specific character.
     *
     * Returns the confidence score for the character at the given index.
     *
     * @param int $charIndex Character index (0-based)
     * @return float Character confidence (0.0-1.0)
     * @throws RuntimeException If character index is invalid
     *
     * Example:
     *     $firstCharConfidence = $span->getCharConfidence(0);
     */
    public function getCharConfidence(int $charIndex): float
    {
        if ($this->charConfidences === null) {
            throw new \RuntimeException('Character confidences not available for this span');
        }

        if ($charIndex < 0 || $charIndex >= count($this->charConfidences)) {
            throw new \RuntimeException(
                "Character index out of bounds: {$charIndex}"
            );
        }

        return $this->charConfidences[$charIndex];
    }

    /**
     * Get span data as array.
     *
     * Returns all span data as an associative array, useful for serialization
     * or passing to APIs that expect arrays.
     *
     * @return array Associative array with keys: text, bbox, confidence, charConfidences
     *
     * Example:
     *     $data = $span->toArray();
     *     $json = json_encode($data);
     */
    public function toArray(): array
    {
        return [
            'text' => $this->text,
            'bbox' => $this->bbox,
            'confidence' => $this->confidence,
            'charConfidences' => $this->charConfidences,
        ];
    }

    /**
     * Extract per-character confidences from span.
     *
     * Iterates through each character and retrieves its confidence score.
     *
     * @param CData $spanHandle FFI span handle
     * @param FunctionBindings $bindings Function bindings
     * @return array|null Array of character confidences, or null if extraction fails
     */
    private function extractCharConfidences(CData $spanHandle, FunctionBindings $bindings): ?array
    {
        try {
            $charConfidences = [];
            $textLength = strlen($this->text);

            for ($i = 0; $i < $textLength; $i++) {
                $confidence = $bindings->pdfOcrSpanGetCharConfidence($spanHandle, $i);
                $charConfidences[] = $confidence;
            }

            return $charConfidences;
        } catch (\Exception) {
            // Character confidences may not always be available
            return null;
        }
    }

    /**
     * String representation of the span.
     *
     * @return string
     */
    public function __toString(): string
    {
        return $this->text;
    }
}
