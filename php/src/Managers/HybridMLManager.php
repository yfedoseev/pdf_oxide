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
 * Manages Hybrid ML operations for intelligent PDF analysis.
 *
 * Provides ML-based page analysis, layout detection, content classification,
 * table detection, column detection, and extraction strategy recommendations.
 * Uses PHP 8+ features for clean, type-safe implementation.
 */
class HybridMLManager
{
    private readonly FunctionBindings $bindings;
    private readonly CData $handle;
    private readonly FFI $ffi;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
        $this->ffi = NativeLibrary::getInstance();
    }

    // ==================== PAGE ANALYSIS ====================

    /**
     * Analyze a page to determine its characteristics.
     *
     * Returns comprehensive analysis including complexity, content type,
     * text/image density, and extraction strategy recommendations.
     *
     * @param int $pageIndex Zero-based page index
     * @return PageAnalysisResult Analysis result
     */
    public function analyzePage(int $pageIndex): PageAnalysisResult
    {
        $errorCode = FFI::new('int');
        $resultHandle = $this->ffi->pdf_analyze_page($this->handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_analyze_page', ['page' => $pageIndex]);

        return new PageAnalysisResult($resultHandle, $this->ffi);
    }

    /**
     * Analyze entire document complexity.
     *
     * Returns an overall complexity score for the document.
     *
     * @return float Document complexity score (0.0 - 1.0)
     */
    public function analyzeDocument(): float
    {
        $errorCode = FFI::new('int');
        $score = $this->ffi->pdf_analyze_document($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_analyze_document');
        return (float)$score;
    }

    /**
     * Estimate processing time for an operation on a page.
     *
     * @param int $pageIndex Zero-based page index
     * @param OperationType $operation Operation type
     * @return int Estimated time in milliseconds
     */
    public function estimateProcessingTime(int $pageIndex, OperationType $operation = OperationType::TEXT_EXTRACTION): int
    {
        $errorCode = FFI::new('int');
        $time = $this->ffi->pdf_estimate_processing_time($this->handle, $operation->value, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_estimate_processing_time', ['page' => $pageIndex]);
        return (int)$time;
    }

    // ==================== EXTRACTION STRATEGY ====================

    /**
     * Create an extraction strategy for a page.
     *
     * The strategy recommends the best approach for text extraction
     * based on page content analysis.
     *
     * @param int $pageIndex Zero-based page index
     * @return ExtractionStrategy Strategy recommendation
     */
    public function createExtractionStrategy(int $pageIndex): ExtractionStrategy
    {
        $errorCode = FFI::new('int');
        $strategyHandle = $this->ffi->pdf_create_extraction_strategy($this->handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_create_extraction_strategy', ['page' => $pageIndex]);

        return new ExtractionStrategy($strategyHandle, $this->ffi);
    }

    /**
     * Get extraction strategy description.
     *
     * @param int $pageIndex Zero-based page index
     * @return string Human-readable strategy description
     */
    public function getStrategyDescription(int $pageIndex): string
    {
        $strategy = $this->createExtractionStrategy($pageIndex);
        return $strategy->getDescription();
    }

    /**
     * Check if OCR is recommended for a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return bool True if OCR is recommended
     */
    public function recommendsOcr(int $pageIndex): bool
    {
        $strategy = $this->createExtractionStrategy($pageIndex);
        return $strategy->recommendsOcr();
    }

    // ==================== LAYOUT DETECTION ====================

    /**
     * Detect columns in a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return ColumnDetectionResult Column detection results with boundaries
     */
    public function detectColumns(int $pageIndex): ColumnDetectionResult
    {
        $outCount = FFI::new('int');
        $errorCode = FFI::new('int');

        $columnsPtr = $this->ffi->pdf_detect_columns($this->handle, $pageIndex, FFI::addr($outCount), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_detect_columns', ['page' => $pageIndex]);

        $count = (int)$outCount->cdata;
        $columns = [];

        // Each column has 4 floats: x, y, width, height
        for ($i = 0; $i < $count; $i++) {
            $offset = $i * 4;
            $columns[] = new ColumnBounds(
                x: (float)$columnsPtr[$offset],
                y: (float)$columnsPtr[$offset + 1],
                width: (float)$columnsPtr[$offset + 2],
                height: (float)$columnsPtr[$offset + 3]
            );
        }

        // Free the native memory
        if ($count > 0) {
            $this->ffi->free_bytes(FFI::cast('uint8_t*', $columnsPtr));
        }

        return new ColumnDetectionResult($count, $columns);
    }

    /**
     * Detect tables in a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return TableDetectionResult Table detection results with boundaries
     */
    public function detectTables(int $pageIndex): TableDetectionResult
    {
        $outCount = FFI::new('int');
        $errorCode = FFI::new('int');

        $tablesPtr = $this->ffi->pdf_detect_tables($this->handle, $pageIndex, FFI::addr($outCount), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_detect_tables', ['page' => $pageIndex]);

        $count = (int)$outCount->cdata;
        $tables = [];

        // Each table has 4 floats: x, y, width, height
        for ($i = 0; $i < $count; $i++) {
            $offset = $i * 4;
            $tables[] = new TableBounds(
                x: (float)$tablesPtr[$offset],
                y: (float)$tablesPtr[$offset + 1],
                width: (float)$tablesPtr[$offset + 2],
                height: (float)$tablesPtr[$offset + 3]
            );
        }

        // Free the native memory
        if ($count > 0) {
            $this->ffi->free_bytes(FFI::cast('uint8_t*', $tablesPtr));
        }

        return new TableDetectionResult($count, $tables);
    }

    // ==================== ML STATUS ====================

    /**
     * Get ML subsystem status.
     *
     * @return MlStatus ML availability status
     */
    public function getMlStatus(): MlStatus
    {
        $errorCode = FFI::new('int');
        $statusPtr = $this->ffi->pdf_ml_get_status(FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ml_get_status');

        $status = StringMarshaller::fromCString($statusPtr);
        return MlStatus::from($status);
    }

    /**
     * Check if a specific ML model is available.
     *
     * @param string $modelName Model name (e.g., 'table_detection', 'column_detection')
     * @return bool True if model is available
     */
    public function isModelAvailable(string $modelName): bool
    {
        $cName = StringMarshaller::toCString($modelName);
        try {
            return (bool)$this->ffi->pdf_ml_model_available($cName);
        } finally {
            unset($cName);
        }
    }

    /**
     * Get available ML models.
     *
     * @return array<string, bool> Model availability map
     */
    public function getAvailableModels(): array
    {
        $models = [
            'table_detection',
            'column_detection',
            'layout_analysis',
            'content_classification',
            'text_extraction',
            'ocr_enhancement',
        ];

        $availability = [];
        foreach ($models as $model) {
            $availability[$model] = $this->isModelAvailable($model);
        }

        return $availability;
    }

    // ==================== BATCH OPERATIONS ====================

    /**
     * Analyze multiple pages in batch.
     *
     * @param int $startPage Start page index
     * @param int $endPage End page index
     * @return array<int, PageAnalysisResult> Analysis results by page index
     */
    public function analyzePageRange(int $startPage, int $endPage): array
    {
        $results = [];
        for ($i = $startPage; $i <= $endPage; $i++) {
            $results[$i] = $this->analyzePage($i);
        }
        return $results;
    }

    /**
     * Get content type statistics for the document.
     *
     * @return ContentTypeStatistics Statistics about content types
     */
    public function getContentTypeStatistics(): ContentTypeStatistics
    {
        $textOnly = 0;
        $textImages = 0;
        $tables = 0;
        $mixed = 0;
        $scanned = 0;
        $forms = 0;
        $vectorGraphics = 0;

        // This would need page count from document
        // For now, return empty statistics
        return new ContentTypeStatistics(
            textOnly: $textOnly,
            textImages: $textImages,
            tables: $tables,
            mixedLayout: $mixed,
            scanned: $scanned,
            forms: $forms,
            vectorGraphics: $vectorGraphics
        );
    }

    /**
     * Get summary of ML capabilities.
     *
     * @return array Summary information
     */
    public function getSummary(): array
    {
        return [
            'status' => $this->getMlStatus()->value,
            'models' => $this->getAvailableModels(),
            'capabilities' => [
                'page_analysis' => true,
                'document_analysis' => true,
                'column_detection' => true,
                'table_detection' => true,
                'extraction_strategy' => true,
            ],
        ];
    }
}

// ==================== SUPPORTING CLASSES ====================

/**
 * Operation types for processing time estimation.
 */
enum OperationType: int
{
    case TEXT_EXTRACTION = 0;
    case RENDERING = 1;
    case OCR = 2;
    case TABLE_EXTRACTION = 3;
    case LAYOUT_ANALYSIS = 4;
}

/**
 * ML subsystem status.
 */
enum MlStatus: string
{
    case AVAILABLE = 'available';
    case DOWNLOADING = 'downloading';
    case UNAVAILABLE = 'unavailable';
    case ERROR = 'error';
}

/**
 * Page content types.
 */
enum ContentType: int
{
    case TEXT_ONLY = 0;
    case TEXT_IMAGES = 1;
    case TABLES = 2;
    case MIXED_LAYOUT = 3;
    case SCANNED = 4;
    case FORM = 5;
    case VECTOR_GRAPHICS = 6;

    public function getDescription(): string
    {
        return match($this) {
            self::TEXT_ONLY => 'Text-only content',
            self::TEXT_IMAGES => 'Text with images',
            self::TABLES => 'Contains tables',
            self::MIXED_LAYOUT => 'Mixed layout',
            self::SCANNED => 'Scanned document',
            self::FORM => 'Form document',
            self::VECTOR_GRAPHICS => 'Vector graphics',
        };
    }
}

/**
 * Page complexity levels.
 */
enum PageComplexity: int
{
    case SIMPLE = 0;
    case MODERATE = 1;
    case COMPLEX = 2;
    case VERY_COMPLEX = 3;

    public function getDescription(): string
    {
        return match($this) {
            self::SIMPLE => 'Simple page layout',
            self::MODERATE => 'Moderate complexity',
            self::COMPLEX => 'Complex layout',
            self::VERY_COMPLEX => 'Very complex layout',
        };
    }
}

/**
 * Page analysis result.
 */
readonly class PageAnalysisResult
{
    private CData $handle;
    private FFI $ffi;

    public function __construct(CData $handle, FFI $ffi)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
    }

    public function getComplexity(): PageComplexity
    {
        $value = (int)$this->ffi->pdf_analysis_get_complexity($this->handle);
        return PageComplexity::from($value);
    }

    public function getComplexityScore(): float
    {
        return (float)$this->ffi->pdf_analysis_get_complexity_score($this->handle);
    }

    public function getContentType(): ContentType
    {
        $value = (int)$this->ffi->pdf_analysis_get_content_type($this->handle);
        return ContentType::from($value);
    }

    public function getTextDensity(): float
    {
        return (float)$this->ffi->pdf_analysis_get_text_density($this->handle);
    }

    public function getImageDensity(): float
    {
        return (float)$this->ffi->pdf_analysis_get_image_density($this->handle);
    }

    public function toArray(): array
    {
        return [
            'complexity' => $this->getComplexity()->name,
            'complexity_score' => $this->getComplexityScore(),
            'content_type' => $this->getContentType()->name,
            'text_density' => $this->getTextDensity(),
            'image_density' => $this->getImageDensity(),
        ];
    }

    public function __destruct()
    {
        $this->ffi->pdf_analysis_result_free($this->handle);
    }
}

/**
 * Extraction strategy recommendation.
 */
readonly class ExtractionStrategy
{
    private CData $handle;
    private FFI $ffi;

    public function __construct(CData $handle, FFI $ffi)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
    }

    public function getDescription(): string
    {
        $errorCode = FFI::new('int');
        $desc = $this->ffi->pdf_strategy_get_description($this->handle, FFI::addr($errorCode));
        return StringMarshaller::fromCString($desc);
    }

    public function recommendsOcr(): bool
    {
        return (bool)$this->ffi->pdf_strategy_recommends_ocr($this->handle);
    }

    public function __destruct()
    {
        $this->ffi->pdf_strategy_free($this->handle);
    }
}

/**
 * Column bounds.
 */
readonly class ColumnBounds
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
 * Column detection result.
 */
readonly class ColumnDetectionResult
{
    /**
     * @param int $count Number of columns detected
     * @param array<ColumnBounds> $columns Column boundaries
     */
    public function __construct(
        public int $count,
        public array $columns
    ) {}

    public function hasColumns(): bool
    {
        return $this->count > 0;
    }

    public function isMultiColumn(): bool
    {
        return $this->count > 1;
    }

    public function toArray(): array
    {
        return [
            'count' => $this->count,
            'columns' => array_map(fn($c) => $c->toArray(), $this->columns),
        ];
    }
}

/**
 * Table bounds.
 */
readonly class TableBounds
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
 * Table detection result.
 */
readonly class TableDetectionResult
{
    /**
     * @param int $count Number of tables detected
     * @param array<TableBounds> $tables Table boundaries
     */
    public function __construct(
        public int $count,
        public array $tables
    ) {}

    public function hasTables(): bool
    {
        return $this->count > 0;
    }

    public function toArray(): array
    {
        return [
            'count' => $this->count,
            'tables' => array_map(fn($t) => $t->toArray(), $this->tables),
        ];
    }
}

/**
 * Content type statistics.
 */
readonly class ContentTypeStatistics
{
    public function __construct(
        public int $textOnly,
        public int $textImages,
        public int $tables,
        public int $mixedLayout,
        public int $scanned,
        public int $forms,
        public int $vectorGraphics
    ) {}

    public function total(): int
    {
        return $this->textOnly + $this->textImages + $this->tables +
               $this->mixedLayout + $this->scanned + $this->forms + $this->vectorGraphics;
    }

    public function toArray(): array
    {
        return [
            'text_only' => $this->textOnly,
            'text_images' => $this->textImages,
            'tables' => $this->tables,
            'mixed_layout' => $this->mixedLayout,
            'scanned' => $this->scanned,
            'forms' => $this->forms,
            'vector_graphics' => $this->vectorGraphics,
            'total' => $this->total(),
        ];
    }
}
