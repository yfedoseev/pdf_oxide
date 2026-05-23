<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\AnalysisResult;

/**
 * Manages PDF document and page analysis operations.
 *
 * Provides intelligent document processing insights including complexity analysis,
 * content type detection, column/table detection, and performance estimation.
 */
class AnalysisManager
{
    private FunctionBindings $bindings;
    private CData $handle;
    private ?array $cachedAnalysis = [];

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Analyze a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return AnalysisResult Analysis results for the page
     */
    public function analyzePage(int $pageIndex): AnalysisResult
    {
        if (isset($this->cachedAnalysis[$pageIndex])) {
            return $this->cachedAnalysis[$pageIndex];
        }

        $resultHandle = $this->bindings->pdfAnalyzePage($this->handle, $pageIndex);
        $result = new AnalysisResult($resultHandle, $this->bindings);
        $this->cachedAnalysis[$pageIndex] = $result;
        return $result;
    }

    /**
     * Get page complexity score.
     *
     * Simple metric for quick assessment of page processing difficulty.
     *
     * @param int $pageIndex Zero-based page index
     * @return float Complexity (0.0 = simple, 1.0 = very complex)
     */
    public function getPageComplexity(int $pageIndex): float
    {
        return $this->analyzePage($pageIndex)->getComplexityScore();
    }

    /**
     * Get page content type.
     *
     * @param int $pageIndex Zero-based page index
     * @return string Content type (text, image, mixed, vector, etc.)
     */
    public function getContentType(int $pageIndex): string
    {
        return $this->analyzePage($pageIndex)->getContentType();
    }

    /**
     * Get text density.
     *
     * Proportion of page covered by text vs. whitespace.
     *
     * @param int $pageIndex Zero-based page index
     * @return float Text density (0.0-1.0)
     */
    public function getTextDensity(int $pageIndex): float
    {
        return $this->analyzePage($pageIndex)->getTextDensity();
    }

    /**
     * Get image density.
     *
     * Proportion of page covered by images vs. text.
     *
     * @param int $pageIndex Zero-based page index
     * @return float Image density (0.0-1.0)
     */
    public function getImageDensity(int $pageIndex): float
    {
        return $this->analyzePage($pageIndex)->getImageDensity();
    }

    /**
     * Estimate processing time for a page.
     *
     * Useful for progress estimation and resource planning.
     *
     * @param int $pageIndex Zero-based page index
     * @return int Estimated milliseconds
     */
    public function estimateProcessingTime(int $pageIndex): int
    {
        return $this->bindings->pdfEstimateProcessingTime($this->handle, $pageIndex);
    }

    /**
     * Detect number of columns in page.
     *
     * Useful for optimal text extraction layout.
     *
     * @param int $pageIndex Zero-based page index
     * @return int Number of columns detected (0 if no columns detected)
     */
    public function detectColumns(int $pageIndex): int
    {
        return $this->bindings->pdfDetectColumns($this->handle, $pageIndex);
    }

    /**
     * Detect number of tables in page.
     *
     * Useful for structured data extraction.
     *
     * @param int $pageIndex Zero-based page index
     * @return int Number of tables detected (0 if no tables detected)
     */
    public function detectTables(int $pageIndex): int
    {
        return $this->bindings->pdfDetectTables($this->handle, $pageIndex);
    }

    /**
     * Get ML model availability status.
     *
     * @return string Status (available, downloading, unavailable)
     */
    public function getMlStatus(): string
    {
        return $this->bindings->pdfMlGetStatus();
    }

    /**
     * Check if specific ML model is available.
     *
     * @param string $modelName Model name (table_detection, column_detection, etc.)
     * @return bool True if model is available
     */
    public function isModelAvailable(string $modelName): bool
    {
        return $this->bindings->pdfMlModelAvailable($modelName);
    }

    /**
     * Get available ML models.
     *
     * @return string[] List of available/possible model names
     */
    public static function getAvailableModels(): array
    {
        return [
            'table_detection',
            'column_detection',
            'text_detection',
            'layout_analysis',
        ];
    }

    /**
     * Analyze entire document and get summary.
     *
     * Note: This is a utility method that would need page count knowledge.
     *
     * @return array Document analysis summary
     */
    public function analyzeDocument(): array
    {
        return [
            'ml_status' => $this->getMlStatus(),
            'models_available' => array_filter(
                self::getAvailableModels(),
                fn($m) => $this->isModelAvailable($m)
            ),
        ];
    }

    /**
     * Get analysis summary for document.
     *
     * @return array Analysis capabilities and status
     */
    public function getSummary(): array
    {
        return [
            'analysis_capabilities' => [
                'complexity_scoring' => true,
                'content_type_detection' => true,
                'text_density_analysis' => true,
                'image_density_analysis' => true,
                'processing_time_estimation' => true,
                'column_detection' => true,
                'table_detection' => true,
            ],
            'ml_status' => $this->getMlStatus(),
        ];
    }

    /**
     * Clear cached analysis results.
     *
     * @internal
     */
    public function clearCache(): void
    {
        $this->cachedAnalysis = [];
    }
}
