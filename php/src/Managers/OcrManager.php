<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI;
use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\ErrorHandler;
use PdfOxide\FFI\StringMarshaller;

/**
 * Manages PDF OCR (Optical Character Recognition) operations.
 *
 * Handles OCR processing for scanned PDFs and mixed content documents.
 * Supports page detection, recognition, confidence scoring, region-based
 * text detection, multi-language support, and batch processing.
 * Uses PHP 8+ features for clean, type-safe implementation.
 */
class OcrManager
{
    private readonly FunctionBindings $bindings;
    private readonly CData $handle;
    private readonly FFI $ffi;
    private ?CData $ocrEngine = null;
    private OcrConfiguration $config;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
        $this->ffi = NativeLibrary::getInstance();
        $this->config = new OcrConfiguration();
    }

    // ==================== ENGINE MANAGEMENT ====================

    /**
     * Initialize OCR engine with configuration.
     *
     * @param OcrConfiguration|null $config Configuration options
     * @return void
     */
    public function initialize(?OcrConfiguration $config = null): void
    {
        if ($config !== null) {
            $this->config = $config;
        }

        if ($this->ocrEngine !== null) {
            $this->freeEngine();
        }

        $errorCode = FFI::new('int');
        $cConfig = $this->createNativeConfig($this->config);

        try {
            $this->ocrEngine = $this->ffi->pdf_ocr_engine_create($cConfig, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_ocr_engine_create');
        } finally {
            unset($cConfig);
        }
    }

    /**
     * Initialize OCR engine (simple, no config).
     *
     * @return void
     */
    public function initializeEngine(): void
    {
        if ($this->ocrEngine === null) {
            $this->initialize();
        }
    }

    /**
     * Create native configuration structure.
     */
    private function createNativeConfig(OcrConfiguration $config): CData
    {
        $cConfig = $this->ffi->new('PdfOcrConfig');
        $cConfig->detection_threshold = $config->detectionThreshold;
        $cConfig->recognition_threshold = $config->recognitionThreshold;
        $cConfig->max_side_len = $config->maxSideLength;
        $cConfig->use_gpu = $config->useGpu;
        $cConfig->gpu_device_id = $config->gpuDeviceId;
        return FFI::addr($cConfig);
    }

    /**
     * Get or create OCR engine.
     */
    private function getEngine(): CData
    {
        if ($this->ocrEngine === null) {
            $this->initialize();
        }
        return $this->ocrEngine;
    }

    /**
     * Get engine version.
     *
     * @return string Version string
     */
    public function getEngineVersion(): string
    {
        $errorCode = FFI::new('int');
        $version = $this->ffi->pdf_ocr_engine_get_version($this->getEngine(), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_engine_get_version');
        return StringMarshaller::fromCString($version);
    }

    /**
     * Get engine status.
     *
     * @return OcrEngineStatus Engine status information
     */
    public function getEngineStatus(): OcrEngineStatus
    {
        $modelsLoaded = FFI::new('bool');
        $gpuAvailable = FFI::new('bool');
        $errorCode = FFI::new('int');

        $statusPtr = $this->ffi->pdf_ocr_engine_get_status(
            $this->getEngine(),
            FFI::addr($modelsLoaded),
            FFI::addr($gpuAvailable),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_engine_get_status');

        $statusStr = StringMarshaller::fromCString($statusPtr);

        return new OcrEngineStatus(
            status: $statusStr,
            modelsLoaded: (bool)$modelsLoaded->cdata,
            gpuAvailable: (bool)$gpuAvailable->cdata
        );
    }

    // ==================== PAGE DETECTION ====================

    /**
     * Check if page needs OCR.
     *
     * @param int $pageIndex Zero-based page index
     * @return bool True if page is scanned/image-based
     */
    public function pageNeedsOcr(int $pageIndex): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_ocr_page_needs_ocr($this->handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_page_needs_ocr', ['page' => $pageIndex]);
        return (bool)$result;
    }

    /**
     * Detect text regions on a page (fast, no recognition).
     *
     * @param int $pageIndex Zero-based page index
     * @return OcrDetectionResult Detection results
     */
    public function detectPage(int $pageIndex): OcrDetectionResult
    {
        $errorCode = FFI::new('int');
        $resultsHandle = $this->ffi->pdf_ocr_detect_page(
            $this->handle,
            $pageIndex,
            $this->getEngine(),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_detect_page', ['page' => $pageIndex]);

        return new OcrDetectionResult($resultsHandle, $this->ffi, $pageIndex);
    }

    /**
     * Recognize text on a page (full OCR).
     *
     * @param int $pageIndex Zero-based page index
     * @return OcrRecognitionResult Recognition results
     */
    public function recognizePage(int $pageIndex): OcrRecognitionResult
    {
        $errorCode = FFI::new('int');
        $resultsHandle = $this->ffi->pdf_ocr_recognize_page(
            $this->handle,
            $pageIndex,
            $this->getEngine(),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_recognize_page', ['page' => $pageIndex]);

        return new OcrRecognitionResult($resultsHandle, $this->ffi, $pageIndex);
    }

    // ==================== TEXT EXTRACTION ====================

    /**
     * Extract text from page with OCR.
     *
     * @param int $pageIndex Zero-based page index
     * @param bool $forceOcr Force OCR even if page has text
     * @return string Extracted text
     */
    public function extractText(int $pageIndex, bool $forceOcr = false): string
    {
        $errorCode = FFI::new('int');
        $textPtr = $this->ffi->pdf_ocr_extract_text(
            $this->handle,
            $pageIndex,
            $this->getEngine(),
            $forceOcr,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_extract_text', ['page' => $pageIndex]);

        return StringMarshaller::fromCString($textPtr);
    }

    /**
     * Extract text spans from page.
     *
     * @param int $pageIndex Zero-based page index
     * @return OcrSpanCollection Collection of text spans
     */
    public function extractSpans(int $pageIndex): OcrSpanCollection
    {
        $errorCode = FFI::new('int');
        $resultsHandle = $this->ffi->pdf_ocr_extract_spans(
            $this->handle,
            $pageIndex,
            $this->getEngine(),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_extract_spans', ['page' => $pageIndex]);

        return new OcrSpanCollection($resultsHandle, $this->ffi, $pageIndex);
    }

    // ==================== BATCH PROCESSING ====================

    /**
     * Check if any page in range needs OCR.
     *
     * @param int $startPage Start page index
     * @param int $endPage End page index
     * @return bool True if any page needs OCR
     */
    public function rangeNeedsOcr(int $startPage, int $endPage): bool
    {
        for ($i = $startPage; $i <= $endPage; $i++) {
            if ($this->pageNeedsOcr($i)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Perform OCR on all pages.
     *
     * @return OcrRecognitionResult[] Results for each page
     */
    public function recognizeAll(): array
    {
        $results = [];
        $errorCode = FFI::new('int');
        $pageCount = $this->ffi->pdf_document_get_page_count($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_page_count');

        for ($i = 0; $i < $pageCount; $i++) {
            $results[] = $this->recognizePage($i);
        }

        return $results;
    }

    /**
     * Process a page range with OCR.
     *
     * @param int $startPage Start page index
     * @param int $endPage End page index
     * @param bool $skipNonScanned Skip pages that don't need OCR
     * @return OcrBatchResult Batch processing results
     */
    public function processPageRange(int $startPage, int $endPage, bool $skipNonScanned = true): OcrBatchResult
    {
        $errorCode = FFI::new('int');
        $resultsHandle = $this->ffi->pdf_ocr_extract_pages(
            $this->handle,
            $startPage,
            $endPage,
            $this->getEngine(),
            $skipNonScanned,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_extract_pages', [
            'start' => $startPage,
            'end' => $endPage,
        ]);

        return new OcrBatchResult($resultsHandle, $this->ffi, $startPage, $endPage);
    }

    /**
     * Extract OCR text with aggregated statistics from page range.
     *
     * @param int $startPage Starting page index
     * @param int $endPage Ending page index
     * @param bool $skipNonScanned Skip pages without selectable text
     * @return array Aggregated OCR statistics
     */
    public function extractPageRange(int $startPage, int $endPage, bool $skipNonScanned = true): array
    {
        $totalSpans = 0;
        $confidenceSum = 0.0;
        $skippedPages = 0;

        for ($pageIdx = $startPage; $pageIdx <= $endPage; $pageIdx++) {
            if ($skipNonScanned) {
                if (!$this->pageNeedsOcr($pageIdx)) {
                    $skippedPages++;
                    continue;
                }
            }

            try {
                $result = $this->recognizePage($pageIdx);
                $spanCount = $result->getSpanCount();
                $totalSpans += $spanCount > 0 ? $spanCount : 1;
                $confidence = $result->getAverageConfidence();
                $confidenceSum += $confidence;
            } catch (\Exception $e) {
                continue;
            }
        }

        $processedPages = ($endPage - $startPage + 1) - $skippedPages;
        $avgConfidence = $processedPages > 0 ? $confidenceSum / $processedPages : 0.0;

        return [
            'startPage' => $startPage,
            'endPage' => $endPage,
            'totalPages' => $endPage - $startPage + 1,
            'totalSpans' => $totalSpans,
            'averageConfidence' => $avgConfidence,
            'skippedPages' => $skippedPages,
        ];
    }

    // ==================== REGION-BASED OCR ====================

    /**
     * Detect text in a specific region.
     *
     * @param int $pageIndex Zero-based page index
     * @param float $x Region X coordinate
     * @param float $y Region Y coordinate
     * @param float $width Region width
     * @param float $height Region height
     * @return OcrRegionResult Region OCR results
     */
    public function detectTextInRegion(
        int $pageIndex,
        float $x,
        float $y,
        float $width,
        float $height
    ): OcrRegionResult {
        $errorCode = FFI::new('int');
        $resultsHandle = $this->ffi->pdf_ocr_detect_region(
            $this->handle,
            $pageIndex,
            $x,
            $y,
            $width,
            $height,
            $this->getEngine(),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_detect_region', ['page' => $pageIndex]);

        return new OcrRegionResult(
            $resultsHandle,
            $this->ffi,
            $pageIndex,
            new OcrRegion($x, $y, $width, $height)
        );
    }

    /**
     * Extract text from a specific region.
     *
     * @param int $pageIndex Zero-based page index
     * @param float $x Region X coordinate
     * @param float $y Region Y coordinate
     * @param float $width Region width
     * @param float $height Region height
     * @return string Extracted text
     */
    public function extractTextFromRegion(
        int $pageIndex,
        float $x,
        float $y,
        float $width,
        float $height
    ): string {
        $result = $this->detectTextInRegion($pageIndex, $x, $y, $width, $height);
        return $result->getText();
    }

    // ==================== LANGUAGE DETECTION ====================

    /**
     * Detect language on a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return LanguageDetectionResult Language detection results
     */
    public function detectLanguage(int $pageIndex): LanguageDetectionResult
    {
        $errorCode = FFI::new('int');
        $resultHandle = $this->ffi->pdf_ocr_detect_language(
            $this->handle,
            $pageIndex,
            $this->getEngine(),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_detect_language', ['page' => $pageIndex]);

        return new LanguageDetectionResult($resultHandle, $this->ffi);
    }

    /**
     * Get supported languages.
     *
     * @return array<string> Array of language codes
     */
    public function getSupportedLanguages(): array
    {
        $errorCode = FFI::new('int');
        $outCount = FFI::new('int');

        $langsPtr = $this->ffi->pdf_ocr_get_supported_languages(
            $this->getEngine(),
            FFI::addr($outCount),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_get_supported_languages');

        $languages = [];
        $count = (int)$outCount->cdata;
        for ($i = 0; $i < $count; $i++) {
            $languages[] = StringMarshaller::fromCString($langsPtr[$i], false);
        }

        return $languages;
    }

    /**
     * Set OCR language.
     *
     * @param string $languageCode Language code (e.g., 'en', 'zh', 'ja')
     * @return void
     */
    public function setLanguage(string $languageCode): void
    {
        $cLang = StringMarshaller::toCString($languageCode);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_ocr_set_language($this->getEngine(), $cLang, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_ocr_set_language');
        } finally {
            unset($cLang);
        }
    }

    // ==================== CONFIGURATION ====================

    /**
     * Set detection threshold.
     *
     * @param float $threshold Threshold (0.0 - 1.0)
     */
    public function setDetectionThreshold(float $threshold): void
    {
        $this->config = new OcrConfiguration(
            detectionThreshold: $threshold,
            recognitionThreshold: $this->config->recognitionThreshold,
            maxSideLength: $this->config->maxSideLength,
            useGpu: $this->config->useGpu,
            gpuDeviceId: $this->config->gpuDeviceId
        );

        if ($this->ocrEngine !== null) {
            $this->initialize($this->config);
        }
    }

    /**
     * Set recognition threshold.
     *
     * @param float $threshold Threshold (0.0 - 1.0)
     */
    public function setRecognitionThreshold(float $threshold): void
    {
        $this->config = new OcrConfiguration(
            detectionThreshold: $this->config->detectionThreshold,
            recognitionThreshold: $threshold,
            maxSideLength: $this->config->maxSideLength,
            useGpu: $this->config->useGpu,
            gpuDeviceId: $this->config->gpuDeviceId
        );

        if ($this->ocrEngine !== null) {
            $this->initialize($this->config);
        }
    }

    /**
     * Enable GPU acceleration.
     *
     * @param int $deviceId GPU device ID
     */
    public function enableGpu(int $deviceId = 0): void
    {
        $this->config = new OcrConfiguration(
            detectionThreshold: $this->config->detectionThreshold,
            recognitionThreshold: $this->config->recognitionThreshold,
            maxSideLength: $this->config->maxSideLength,
            useGpu: true,
            gpuDeviceId: $deviceId
        );

        if ($this->ocrEngine !== null) {
            $this->initialize($this->config);
        }
    }

    /**
     * Disable GPU acceleration.
     */
    public function disableGpu(): void
    {
        $this->config = new OcrConfiguration(
            detectionThreshold: $this->config->detectionThreshold,
            recognitionThreshold: $this->config->recognitionThreshold,
            maxSideLength: $this->config->maxSideLength,
            useGpu: false,
            gpuDeviceId: 0
        );

        if ($this->ocrEngine !== null) {
            $this->initialize($this->config);
        }
    }

    // ==================== CLEANUP ====================

    /**
     * Free OCR engine resources.
     */
    public function freeEngine(): void
    {
        if ($this->ocrEngine !== null) {
            $this->ffi->pdf_ocr_engine_free($this->ocrEngine);
            $this->ocrEngine = null;
        }
    }

    /**
     * Get OCR summary for document.
     *
     * @return array Summary with engine info and capabilities
     */
    public function getSummary(): array
    {
        $status = $this->getEngineStatus();
        return [
            'version' => $this->getEngineVersion(),
            'status' => $status->toArray(),
            'initialized' => $this->ocrEngine !== null,
            'configuration' => $this->config->toArray(),
            'capabilities' => [
                'page_detection' => true,
                'page_recognition' => true,
                'batch_processing' => true,
                'region_ocr' => true,
                'language_detection' => true,
                'gpu_acceleration' => $status->gpuAvailable,
            ],
        ];
    }

    /**
     * Free resources on destruct.
     */
    public function __destruct()
    {
        $this->freeEngine();
    }
}

// ==================== SUPPORTING CLASSES ====================

/**
 * OCR configuration options.
 */
readonly class OcrConfiguration
{
    public function __construct(
        public float $detectionThreshold = 0.5,
        public float $recognitionThreshold = 0.5,
        public int $maxSideLength = 1920,
        public bool $useGpu = false,
        public int $gpuDeviceId = 0
    ) {}

    public function toArray(): array
    {
        return [
            'detection_threshold' => $this->detectionThreshold,
            'recognition_threshold' => $this->recognitionThreshold,
            'max_side_length' => $this->maxSideLength,
            'use_gpu' => $this->useGpu,
            'gpu_device_id' => $this->gpuDeviceId,
        ];
    }
}

/**
 * OCR engine status.
 */
readonly class OcrEngineStatus
{
    public function __construct(
        public string $status,
        public bool $modelsLoaded,
        public bool $gpuAvailable
    ) {}

    public function isReady(): bool
    {
        return $this->modelsLoaded;
    }

    public function toArray(): array
    {
        return [
            'status' => $this->status,
            'models_loaded' => $this->modelsLoaded,
            'gpu_available' => $this->gpuAvailable,
            'ready' => $this->isReady(),
        ];
    }
}

/**
 * OCR region definition.
 */
readonly class OcrRegion
{
    public function __construct(
        public float $x,
        public float $y,
        public float $width,
        public float $height
    ) {}

    public function toArray(): array
    {
        return [
            'x' => $this->x,
            'y' => $this->y,
            'width' => $this->width,
            'height' => $this->height,
        ];
    }
}

/**
 * OCR detection result (without text recognition).
 */
class OcrDetectionResult
{
    private CData $handle;
    private FFI $ffi;
    private int $pageIndex;

    public function __construct(CData $handle, FFI $ffi, int $pageIndex)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
        $this->pageIndex = $pageIndex;
    }

    public function getSpanCount(): int
    {
        return (int)$this->ffi->pdf_ocr_results_count($this->handle);
    }

    public function getAverageConfidence(): float
    {
        return (float)$this->ffi->pdf_ocr_results_average_confidence($this->handle);
    }

    public function getPageIndex(): int
    {
        return $this->pageIndex;
    }

    public function __destruct()
    {
        $this->ffi->pdf_ocr_results_free($this->handle);
    }
}

/**
 * OCR recognition result (with text).
 */
class OcrRecognitionResult extends OcrDetectionResult
{
    public function getText(): string
    {
        $errorCode = FFI::new('int');
        $textPtr = $this->ffi->pdf_ocr_results_get_text($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_results_get_text');
        return StringMarshaller::fromCString($textPtr);
    }

    public function getSpan(int $index): OcrTextSpan
    {
        $errorCode = FFI::new('int');
        $spanPtr = $this->ffi->pdf_ocr_results_get_span($this->handle, $index, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_results_get_span');
        return new OcrTextSpan($spanPtr, $this->ffi);
    }

    public function getAllSpans(): array
    {
        $spans = [];
        $count = $this->getSpanCount();
        for ($i = 0; $i < $count; $i++) {
            $spans[] = $this->getSpan($i);
        }
        return $spans;
    }
}

/**
 * OCR text span.
 */
class OcrTextSpan
{
    private CData $handle;
    private FFI $ffi;

    public function __construct(CData $handle, FFI $ffi)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
    }

    public function getText(): string
    {
        return (string)FFI::string($this->handle->text);
    }

    public function getConfidence(): float
    {
        return (float)$this->handle->confidence;
    }

    public function getCharCount(): int
    {
        return (int)$this->handle->char_count;
    }

    public function getPolygon(): array
    {
        $polygon = [];
        for ($i = 0; $i < 8; $i++) {
            $polygon[] = (float)$this->handle->polygon[$i];
        }
        return $polygon;
    }

    public function getBoundingBox(): array
    {
        $x = FFI::new('float');
        $y = FFI::new('float');
        $width = FFI::new('float');
        $height = FFI::new('float');

        $this->ffi->pdf_ocr_span_get_bbox($this->handle, FFI::addr($x), FFI::addr($y), FFI::addr($width), FFI::addr($height));

        return [
            'x' => (float)$x->cdata,
            'y' => (float)$y->cdata,
            'width' => (float)$width->cdata,
            'height' => (float)$height->cdata,
        ];
    }

    public function getCharConfidence(int $charIndex): float
    {
        return (float)$this->ffi->pdf_ocr_span_get_char_confidence($this->handle, $charIndex);
    }

    public function toArray(): array
    {
        return [
            'text' => $this->getText(),
            'confidence' => $this->getConfidence(),
            'char_count' => $this->getCharCount(),
            'bounding_box' => $this->getBoundingBox(),
        ];
    }

    public function __destruct()
    {
        $this->ffi->pdf_ocr_span_free($this->handle);
    }
}

/**
 * OCR span collection.
 */
class OcrSpanCollection extends OcrRecognitionResult
{
    public function filter(float $minConfidence): array
    {
        return array_filter(
            $this->getAllSpans(),
            fn(OcrTextSpan $span) => $span->getConfidence() >= $minConfidence
        );
    }

    public function toArray(): array
    {
        return array_map(fn(OcrTextSpan $span) => $span->toArray(), $this->getAllSpans());
    }
}

/**
 * OCR batch result.
 */
class OcrBatchResult
{
    private CData $handle;
    private FFI $ffi;
    private int $startPage;
    private int $endPage;

    public function __construct(CData $handle, FFI $ffi, int $startPage, int $endPage)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
        $this->startPage = $startPage;
        $this->endPage = $endPage;
    }

    public function getTotalSpanCount(): int
    {
        return (int)$this->ffi->pdf_ocr_results_count($this->handle);
    }

    public function getAverageConfidence(): float
    {
        return (float)$this->ffi->pdf_ocr_results_average_confidence($this->handle);
    }

    public function getPageForSpan(int $spanIndex): int
    {
        return (int)$this->ffi->pdf_ocr_batch_results_get_page($this->handle, $spanIndex);
    }

    public function getStartPage(): int
    {
        return $this->startPage;
    }

    public function getEndPage(): int
    {
        return $this->endPage;
    }

    public function getPageCount(): int
    {
        return $this->endPage - $this->startPage + 1;
    }

    public function __destruct()
    {
        $this->ffi->pdf_ocr_results_free($this->handle);
    }
}

/**
 * OCR region result.
 */
class OcrRegionResult extends OcrRecognitionResult
{
    private OcrRegion $region;

    public function __construct(CData $handle, FFI $ffi, int $pageIndex, OcrRegion $region)
    {
        parent::__construct($handle, $ffi, $pageIndex);
        $this->region = $region;
    }

    public function getRegion(): OcrRegion
    {
        return $this->region;
    }
}

/**
 * Language detection result.
 */
class LanguageDetectionResult
{
    private CData $handle;
    private FFI $ffi;

    public function __construct(CData $handle, FFI $ffi)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
    }

    public function getPrimaryLanguage(): string
    {
        $langPtr = $this->ffi->pdf_language_result_get_primary($this->handle);
        return StringMarshaller::fromCString($langPtr, false);
    }

    public function getConfidence(): float
    {
        return (float)$this->ffi->pdf_language_result_get_confidence($this->handle);
    }

    public function getDetectedLanguages(): array
    {
        $outCount = FFI::new('int');
        $langsPtr = $this->ffi->pdf_language_result_get_all($this->handle, FFI::addr($outCount));

        $languages = [];
        $count = (int)$outCount->cdata;
        for ($i = 0; $i < $count; $i++) {
            $languages[] = StringMarshaller::fromCString($langsPtr[$i], false);
        }

        return $languages;
    }

    public function toArray(): array
    {
        return [
            'primary' => $this->getPrimaryLanguage(),
            'confidence' => $this->getConfidence(),
            'detected' => $this->getDetectedLanguages(),
        ];
    }

    public function __destruct()
    {
        $this->ffi->pdf_language_result_free($this->handle);
    }
}
