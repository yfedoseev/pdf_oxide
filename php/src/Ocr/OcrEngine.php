<?php

declare(strict_types=1);

namespace PdfOxide\Ocr;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\PdfDocument;
use PdfOxide\PdfPage;

/**
 * OCR engine for text recognition in scanned PDFs.
 *
 * Provides optical character recognition capabilities with support for:
 * - Single-page text extraction
 * - Text spans with bounding boxes and confidence scores
 * - Batch processing of multiple pages
 * - Page OCR need detection
 *
 * Example:
 *     $engine = new OcrEngine();
 *     $doc = new PdfDocument('scanned.pdf');
 *
 *     // Extract plain text
 *     $text = $engine->extractText($doc->getPage(0));
 *
 *     // Extract with position and confidence
 *     $spans = $engine->extractSpans($doc->getPage(0));
 *     foreach ($spans as $span) {
 *         echo $span->getText() . " ({$span->getConfidence()})";
 *     }
 *
 *     // Check if OCR is needed
 *     if ($engine->pageNeedsOcr($doc->getPage(0))) {
 *         echo "Page requires OCR processing";
 *     }
 *
 * @since 0.4.0
 */
class OcrEngine
{
    private ?CData $handle = null;
    private FunctionBindings $bindings;
    private bool $closed = false;

    /**
     * Create OCR engine instance.
     *
     * @param string|null $modelPath Optional path to OCR model directory
     * @throws RuntimeException If engine creation fails
     */
    public function __construct(?string $modelPath = null)
    {
        $this->bindings = new FunctionBindings();

        try {
            $this->handle = $this->bindings->pdfOcrEngineCreate();
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to create OCR engine: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Destructor - cleanup OCR engine resources.
     */
    public function __destruct()
    {
        $this->close();
    }

    /**
     * Close and cleanup OCR engine resources.
     *
     * Automatically called in destructor, but can be called manually
     * to free resources before the object is garbage collected.
     */
    public function close(): void
    {
        if ($this->handle !== null && !$this->closed) {
            try {
                $this->bindings->pdfOcrEngineFree($this->handle);
            } catch (\Exception) {
                // Silently ignore errors during cleanup
            }
            $this->handle = null;
            $this->closed = true;
        }
    }

    /**
     * Extract plain text from a page using OCR.
     *
     * Performs optical character recognition on the page and returns
     * all recognized text as a single string.
     *
     * @param PdfPage $page Page to extract from
     * @return string Extracted text
     * @throws RuntimeException If extraction fails
     */
    public function extractText(PdfPage $page): string
    {
        $this->ensureOpen();

        try {
            $results = $this->bindings->pdfOcrRecognizePage(
                $this->handle,
                $page->getHandle(),
                $page->getIndex()
            );

            $text = $this->bindings->pdfOcrExtractText($results);
            $this->bindings->pdfOcrResultsFree($results);

            return $text;
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "OCR text extraction failed: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Extract text spans with positions and confidence scores.
     *
     * Returns an array of OcrSpan objects, each representing a recognized
     * word or text segment with its bounding box and confidence score.
     *
     * @param PdfPage $page Page to extract from
     * @return OcrSpan[] Array of text spans
     * @throws RuntimeException If extraction fails
     *
     * Example:
     *     $spans = $engine->extractSpans($page);
     *     foreach ($spans as $span) {
     *         [$x, $y, $width, $height] = $span->getBbox();
     *         echo "{$span->getText()} at ({$x}, {$y}) - Confidence: {$span->getConfidence()}";
     *     }
     */
    public function extractSpans(PdfPage $page): array
    {
        $this->ensureOpen();

        try {
            $results = $this->bindings->pdfOcrRecognizePage(
                $this->handle,
                $page->getHandle(),
                $page->getIndex()
            );

            $spans = [];
            $count = $this->bindings->pdfOcrResultsCount($results);

            for ($i = 0; $i < $count; $i++) {
                $spanHandle = $this->bindings->pdfOcrResultsGetSpan($results, $i);
                $spans[] = new OcrSpan($spanHandle, $this->bindings);
            }

            $this->bindings->pdfOcrResultsFree($results);

            return $spans;
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "OCR span extraction failed: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Extract text spans from multiple pages.
     *
     * Processes an array of pages and returns an array of span arrays,
     * indexed by page order.
     *
     * @param PdfPage[] $pages Pages to process
     * @return array<int, OcrSpan[]> Array of span arrays indexed by page order
     * @throws RuntimeException If extraction fails
     *
     * Example:
     *     $pages = [$doc->getPage(0), $doc->getPage(1), $doc->getPage(2)];
     *     $allSpans = $engine->extractPages($pages);
     *     foreach ($allSpans as $pageIndex => $spans) {
     *         echo "Page {$pageIndex}: " . count($spans) . " spans";
     *     }
     */
    public function extractPages(array $pages): array
    {
        $results = [];
        foreach ($pages as $page) {
            $results[] = $this->extractSpans($page);
        }
        return $results;
    }

    /**
     * Extract text spans from a page range.
     *
     * Processes all pages from startPage to endPage (inclusive) and returns
     * their extracted spans.
     *
     * @param PdfDocument $doc Document containing the pages
     * @param int $startPage Start page index (0-based, inclusive)
     * @param int $endPage End page index (0-based, inclusive)
     * @return array<int, OcrSpan[]> Array of span arrays
     * @throws RuntimeException If extraction fails
     *
     * Example:
     *     $spans = $engine->extractRange($doc, 0, 4);  // Pages 0-4
     *     foreach ($spans as $pageSpans) {
     *         echo "Found " . count($pageSpans) . " spans";
     *     }
     */
    public function extractRange(PdfDocument $doc, int $startPage, int $endPage): array
    {
        $results = [];
        for ($i = $startPage; $i <= $endPage; $i++) {
            $results[] = $this->extractSpans($doc->getPage($i));
        }
        return $results;
    }

    /**
     * Check if a page needs OCR (has no embedded text).
     *
     * Returns true if the page contains only images or scanned content
     * without embedded text, indicating that OCR would be needed.
     *
     * @param PdfPage $page Page to check
     * @return bool True if page needs OCR, false if it has embedded text
     * @throws RuntimeException If detection fails
     *
     * Example:
     *     if ($engine->pageNeedsOcr($page)) {
     *         $text = $engine->extractText($page);  // Use OCR
     *     } else {
     *         $text = $page->extractText();  // Use native extraction
     *     }
     */
    public function pageNeedsOcr(PdfPage $page): bool
    {
        $this->ensureOpen();

        try {
            return $this->bindings->pdfOcrPageNeedsOcr(
                $page->getHandle(),
                $page->getIndex()
            );
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "OCR page detection failed: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Get OCR engine version.
     *
     * Returns the version string of the OCR engine implementation.
     *
     * @return string Version string (e.g., "4.2.0")
     * @throws RuntimeException If version retrieval fails
     *
     * Example:
     *     $version = OcrEngine::getVersion();
     *     echo "OCR Engine v{$version}";
     */
    public static function getVersion(): string
    {
        $bindings = new FunctionBindings();

        try {
            // Create temporary engine to get version
            $engine = $bindings->pdfOcrEngineCreate();
            $version = $bindings->pdfOcrEngineGetVersion($engine);
            $bindings->pdfOcrEngineFree($engine);

            return $version;
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to get OCR engine version: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Get OCR engine status/health information.
     *
     * Returns status information about the OCR engine including
     * initialization state and any warnings.
     *
     * @return string Status string describing engine state
     * @throws RuntimeException If status retrieval fails
     *
     * Example:
     *     $status = $engine->getStatus();
     *     echo "Engine status: " . $status;
     */
    public function getStatus(): string
    {
        $this->ensureOpen();

        try {
            return $this->bindings->pdfOcrEngineGetStatus($this->handle);
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to get OCR engine status: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Check if engine is still open and operational.
     *
     * @return bool True if engine is open, false if closed
     */
    public function isOpen(): bool
    {
        return $this->handle !== null && !$this->closed;
    }

    /**
     * Ensure engine is open and throw if closed.
     *
     * @throws RuntimeException If engine is closed
     */
    private function ensureOpen(): void
    {
        if (!$this->isOpen()) {
            throw new \RuntimeException('OCR engine has been closed');
        }
    }

    /**
     * Get the underlying FFI handle.
     *
     * For advanced use only. Direct handle manipulation may cause issues.
     *
     * @internal
     * @return CData The FFI handle
     */
    public function getHandle(): CData
    {
        $this->ensureOpen();
        return $this->handle;
    }

    /**
     * Get the FunctionBindings instance.
     *
     * @internal
     * @return FunctionBindings
     */
    public function getBindings(): FunctionBindings
    {
        return $this->bindings;
    }
}
