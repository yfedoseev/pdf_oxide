<?php

declare(strict_types=1);

namespace PdfOxide\Types;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Represents page analysis results.
 *
 * Contains metrics about page complexity, content distribution, and detected elements.
 */
class AnalysisResult
{
    private CData $handle;
    private FunctionBindings $bindings;
    private ?float $cachedComplexity = null;
    private ?string $cachedContentType = null;
    private ?float $cachedTextDensity = null;
    private ?float $cachedImageDensity = null;

    public function __construct(CData $handle, FunctionBindings $bindings)
    {
        $this->handle = $handle;
        $this->bindings = $bindings;
    }

    /**
     * Get complexity score.
     *
     * Higher scores indicate more complex processing requirements.
     *
     * @return float Complexity (0.0-1.0)
     */
    public function getComplexityScore(): float
    {
        if ($this->cachedComplexity === null) {
            $this->cachedComplexity = $this->bindings->pdfAnalysisGetComplexityScore($this->handle);
        }
        return $this->cachedComplexity;
    }

    /**
     * Get page content type.
     *
     * @return string Content type classification
     */
    public function getContentType(): string
    {
        if ($this->cachedContentType === null) {
            $this->cachedContentType = $this->bindings->pdfAnalysisGetContentType($this->handle);
        }
        return $this->cachedContentType;
    }

    /**
     * Get text density.
     *
     * @return float Proportion of page with text (0.0-1.0)
     */
    public function getTextDensity(): float
    {
        if ($this->cachedTextDensity === null) {
            $this->cachedTextDensity = $this->bindings->pdfAnalysisGetTextDensity($this->handle);
        }
        return $this->cachedTextDensity;
    }

    /**
     * Get image density.
     *
     * @return float Proportion of page with images (0.0-1.0)
     */
    public function getImageDensity(): float
    {
        if ($this->cachedImageDensity === null) {
            $this->cachedImageDensity = $this->bindings->pdfAnalysisGetImageDensity($this->handle);
        }
        return $this->cachedImageDensity;
    }

    /**
     * Check if page is primarily text.
     *
     * @param float $threshold Text density threshold (default 0.7)
     * @return bool True if text-heavy
     */
    public function isTextHeavy(float $threshold = 0.7): bool
    {
        return $this->getTextDensity() >= $threshold;
    }

    /**
     * Check if page is primarily image-based.
     *
     * @param float $threshold Image density threshold (default 0.7)
     * @return bool True if image-heavy
     */
    public function isImageHeavy(float $threshold = 0.7): bool
    {
        return $this->getImageDensity() >= $threshold;
    }

    /**
     * Check if page is highly complex.
     *
     * @param float $threshold Complexity threshold (default 0.7)
     * @return bool True if highly complex
     */
    public function isHighlyComplex(float $threshold = 0.7): bool
    {
        return $this->getComplexityScore() >= $threshold;
    }

    /**
     * Check if page is relatively simple.
     *
     * @param float $threshold Complexity threshold (default 0.3)
     * @return bool True if simple
     */
    public function isSimple(float $threshold = 0.3): bool
    {
        return $this->getComplexityScore() <= $threshold;
    }

    /**
     * Get whitespace ratio.
     *
     * @return float Proportion of page without content (0.0-1.0)
     */
    public function getWhitespaceRatio(): float
    {
        return 1.0 - ($this->getTextDensity() + $this->getImageDensity());
    }

    /**
     * Get analysis summary.
     *
     * @return array Summary metrics
     */
    public function getSummary(): array
    {
        return [
            'complexity_score' => $this->getComplexityScore(),
            'content_type' => $this->getContentType(),
            'text_density' => $this->getTextDensity(),
            'image_density' => $this->getImageDensity(),
            'whitespace_ratio' => $this->getWhitespaceRatio(),
            'is_text_heavy' => $this->isTextHeavy(),
            'is_image_heavy' => $this->isImageHeavy(),
            'is_complex' => $this->isHighlyComplex(),
            'is_simple' => $this->isSimple(),
        ];
    }

    /**
     * Convert to array.
     *
     * @return array Full analysis data
     */
    public function toArray(): array
    {
        return $this->getSummary();
    }

    /**
     * Free analysis result resources.
     */
    public function __destruct()
    {
        $this->bindings->pdfAnalysisResultFree($this->handle);
    }
}
