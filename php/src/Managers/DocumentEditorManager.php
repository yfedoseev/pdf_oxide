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
 * Manages PDF document editing operations.
 *
 * Handles page manipulation (insert, delete, rotate, reorder),
 * document merging, splitting, and content modification.
 * Uses PHP 8+ features for clean, type-safe implementation.
 */
class DocumentEditorManager
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

    // ==================== PAGE MANIPULATION ====================

    /**
     * Delete a page from the document.
     *
     * @param int $pageIndex Zero-based page index to delete
     * @return bool True on success
     */
    public function deletePage(int $pageIndex): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_delete_page($this->handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_delete_page', ['page' => $pageIndex]);
        return (bool)$result;
    }

    /**
     * Delete multiple pages from the document.
     *
     * @param array<int> $pageIndices Page indices to delete (sorted descending)
     * @return int Number of pages deleted
     */
    public function deletePages(array $pageIndices): int
    {
        // Sort descending to avoid index shifting issues
        rsort($pageIndices);

        $deleted = 0;
        foreach ($pageIndices as $pageIndex) {
            if ($this->deletePage($pageIndex)) {
                $deleted++;
            }
        }
        return $deleted;
    }

    /**
     * Insert a blank page at specified position.
     *
     * @param int $position Position to insert (0 = beginning)
     * @param float $width Page width in points (default: A4)
     * @param float $height Page height in points (default: A4)
     * @return bool True on success
     */
    public function insertBlankPage(int $position, float $width = 595.0, float $height = 842.0): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_insert_blank_page(
            $this->handle,
            $position,
            $width,
            $height,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_insert_blank_page', ['position' => $position]);
        return (bool)$result;
    }

    /**
     * Rotate a page.
     *
     * @param int $pageIndex Zero-based page index
     * @param PageRotation $rotation Rotation angle
     * @return bool True on success
     */
    public function rotatePage(int $pageIndex, PageRotation $rotation): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_rotate_page(
            $this->handle,
            $pageIndex,
            $rotation->value,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_rotate_page', ['page' => $pageIndex]);
        return (bool)$result;
    }

    /**
     * Rotate multiple pages.
     *
     * @param array<int> $pageIndices Page indices to rotate
     * @param PageRotation $rotation Rotation angle
     * @return int Number of pages rotated
     */
    public function rotatePages(array $pageIndices, PageRotation $rotation): int
    {
        $rotated = 0;
        foreach ($pageIndices as $pageIndex) {
            if ($this->rotatePage($pageIndex, $rotation)) {
                $rotated++;
            }
        }
        return $rotated;
    }

    /**
     * Move a page to a new position.
     *
     * @param int $fromIndex Source page index
     * @param int $toIndex Destination page index
     * @return bool True on success
     */
    public function movePage(int $fromIndex, int $toIndex): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_move_page(
            $this->handle,
            $fromIndex,
            $toIndex,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_move_page', [
            'from' => $fromIndex,
            'to' => $toIndex,
        ]);
        return (bool)$result;
    }

    /**
     * Reorder pages according to new order.
     *
     * @param array<int> $newOrder Array where value at index i is the old page index
     * @return bool True on success
     */
    public function reorderPages(array $newOrder): bool
    {
        $errorCode = FFI::new('int');
        $count = count($newOrder);

        // Create C array for page order
        $cOrder = FFI::new("int[{$count}]");
        for ($i = 0; $i < $count; $i++) {
            $cOrder[$i] = $newOrder[$i];
        }

        $result = $this->ffi->pdf_document_reorder_pages(
            $this->handle,
            $cOrder,
            $count,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_reorder_pages');
        return (bool)$result;
    }

    /**
     * Duplicate a page.
     *
     * @param int $pageIndex Page index to duplicate
     * @param int|null $insertAt Position to insert copy (null = after original)
     * @return bool True on success
     */
    public function duplicatePage(int $pageIndex, ?int $insertAt = null): bool
    {
        $insertAt ??= $pageIndex + 1;

        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_duplicate_page(
            $this->handle,
            $pageIndex,
            $insertAt,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_duplicate_page', ['page' => $pageIndex]);
        return (bool)$result;
    }

    // ==================== PAGE SIZE OPERATIONS ====================

    /**
     * Resize a page.
     *
     * @param int $pageIndex Zero-based page index
     * @param float $width New width in points
     * @param float $height New height in points
     * @return bool True on success
     */
    public function resizePage(int $pageIndex, float $width, float $height): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_resize_page(
            $this->handle,
            $pageIndex,
            $width,
            $height,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_resize_page', ['page' => $pageIndex]);
        return (bool)$result;
    }

    /**
     * Set page crop box.
     *
     * @param int $pageIndex Zero-based page index
     * @param float $x Crop box X
     * @param float $y Crop box Y
     * @param float $width Crop box width
     * @param float $height Crop box height
     * @return bool True on success
     */
    public function setCropBox(int $pageIndex, float $x, float $y, float $width, float $height): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_set_crop_box(
            $this->handle,
            $pageIndex,
            $x,
            $y,
            $width,
            $height,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_set_crop_box', ['page' => $pageIndex]);
        return (bool)$result;
    }

    // ==================== DOCUMENT MERGE/SPLIT ====================

    /**
     * Merge another PDF into this document.
     *
     * @param string $otherPdfPath Path to PDF to merge
     * @param int|null $insertAt Position to insert (null = end)
     * @return int Number of pages added
     */
    public function mergeDocument(string $otherPdfPath, ?int $insertAt = null): int
    {
        $cPath = StringMarshaller::toCString($otherPdfPath);
        $errorCode = FFI::new('int');

        try {
            $pagesAdded = $this->ffi->pdf_document_merge(
                $this->handle,
                $cPath,
                $insertAt ?? -1,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_merge', ['path' => $otherPdfPath]);
            return (int)$pagesAdded;
        } finally {
            unset($cPath);
        }
    }

    /**
     * Merge specific pages from another PDF.
     *
     * @param string $otherPdfPath Path to PDF to merge from
     * @param array<int> $pageIndices Page indices to merge
     * @param int|null $insertAt Position to insert (null = end)
     * @return int Number of pages added
     */
    public function mergePages(string $otherPdfPath, array $pageIndices, ?int $insertAt = null): int
    {
        $cPath = StringMarshaller::toCString($otherPdfPath);
        $errorCode = FFI::new('int');
        $count = count($pageIndices);

        // Create C array for page indices
        $cIndices = FFI::new("int[{$count}]");
        for ($i = 0; $i < $count; $i++) {
            $cIndices[$i] = $pageIndices[$i];
        }

        try {
            $pagesAdded = $this->ffi->pdf_document_merge_pages(
                $this->handle,
                $cPath,
                $cIndices,
                $count,
                $insertAt ?? -1,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_merge_pages', ['path' => $otherPdfPath]);
            return (int)$pagesAdded;
        } finally {
            unset($cPath);
        }
    }

    /**
     * Split document into separate PDFs.
     *
     * @param string $outputPrefix Output file prefix
     * @param SplitMode $mode Split mode
     * @param int $pagesPerFile Pages per file (for EVERY_N_PAGES mode)
     * @return array<string> Array of created file paths
     */
    public function splitDocument(string $outputPrefix, SplitMode $mode = SplitMode::EVERY_PAGE, int $pagesPerFile = 1): array
    {
        $cPrefix = StringMarshaller::toCString($outputPrefix);
        $errorCode = FFI::new('int');
        $outCount = FFI::new('int');

        try {
            $filesPtr = $this->ffi->pdf_document_split(
                $this->handle,
                $cPrefix,
                $mode->value,
                $pagesPerFile,
                FFI::addr($outCount),
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_split');

            $files = [];
            $count = (int)$outCount->cdata;
            for ($i = 0; $i < $count; $i++) {
                $files[] = StringMarshaller::fromCString($filesPtr[$i], false);
            }

            return $files;
        } finally {
            unset($cPrefix);
        }
    }

    /**
     * Extract specific pages to a new PDF.
     *
     * @param array<int> $pageIndices Page indices to extract
     * @param string $outputPath Output file path
     * @return bool True on success
     */
    public function extractPages(array $pageIndices, string $outputPath): bool
    {
        $cPath = StringMarshaller::toCString($outputPath);
        $errorCode = FFI::new('int');
        $count = count($pageIndices);

        // Create C array for page indices
        $cIndices = FFI::new("int[{$count}]");
        for ($i = 0; $i < $count; $i++) {
            $cIndices[$i] = $pageIndices[$i];
        }

        try {
            $result = $this->ffi->pdf_document_extract_pages(
                $this->handle,
                $cIndices,
                $count,
                $cPath,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_extract_pages');
            return (bool)$result;
        } finally {
            unset($cPath);
        }
    }

    // ==================== CONTENT OPERATIONS ====================

    /**
     * Flatten all form fields.
     *
     * Converts interactive form fields to static content.
     *
     * @return bool True on success
     */
    public function flattenForms(): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_flatten_forms($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_flatten_forms');
        return (bool)$result;
    }

    /**
     * Flatten all annotations.
     *
     * Converts annotations to static content.
     *
     * @return bool True on success
     */
    public function flattenAnnotations(): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_flatten_annotations($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_flatten_annotations');
        return (bool)$result;
    }

    /**
     * Remove all annotations.
     *
     * @return int Number of annotations removed
     */
    public function removeAnnotations(): int
    {
        $errorCode = FFI::new('int');
        $count = $this->ffi->pdf_document_remove_annotations($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_remove_annotations');
        return (int)$count;
    }

    /**
     * Remove all JavaScript.
     *
     * @return bool True on success
     */
    public function removeJavaScript(): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_remove_javascript($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_remove_javascript');
        return (bool)$result;
    }

    /**
     * Optimize document for web viewing (linearize).
     *
     * @return bool True on success
     */
    public function linearize(): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_linearize($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_linearize');
        return (bool)$result;
    }

    /**
     * Compress document to reduce file size.
     *
     * @param CompressionLevel $level Compression level
     * @return CompressionResult Compression statistics
     */
    public function compress(CompressionLevel $level = CompressionLevel::MEDIUM): CompressionResult
    {
        $errorCode = FFI::new('int');
        $originalSize = FFI::new('int64_t');
        $compressedSize = FFI::new('int64_t');

        $result = $this->ffi->pdf_document_compress(
            $this->handle,
            $level->value,
            FFI::addr($originalSize),
            FFI::addr($compressedSize),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_compress');

        return new CompressionResult(
            success: (bool)$result,
            originalSize: (int)$originalSize->cdata,
            compressedSize: (int)$compressedSize->cdata
        );
    }

    // ==================== SAVE OPERATIONS ====================

    /**
     * Save document to file.
     *
     * @param string $filePath Output file path
     * @return bool True on success
     */
    public function save(string $filePath): bool
    {
        $cPath = StringMarshaller::toCString($filePath);
        $errorCode = FFI::new('int');

        try {
            $result = $this->ffi->pdf_document_save($this->handle, $cPath, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_save');
            return (bool)$result;
        } finally {
            unset($cPath);
        }
    }

    /**
     * Save document to bytes.
     *
     * @return string PDF document as binary string
     */
    public function saveToBytes(): string
    {
        $errorCode = FFI::new('int');
        $outSize = FFI::new('size_t');

        $dataPtr = $this->ffi->pdf_document_save_to_bytes($this->handle, FFI::addr($outSize), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_save_to_bytes');

        $size = (int)$outSize->cdata;
        $data = FFI::string($dataPtr, $size);

        // Free the native memory
        $this->ffi->free_bytes($dataPtr);

        return $data;
    }

    // ==================== UTILITIES ====================

    /**
     * Get editor capabilities summary.
     *
     * @return array Summary information
     */
    public function getSummary(): array
    {
        return [
            'capabilities' => [
                'page_manipulation' => [
                    'delete' => true,
                    'insert' => true,
                    'rotate' => true,
                    'move' => true,
                    'reorder' => true,
                    'duplicate' => true,
                    'resize' => true,
                ],
                'document_operations' => [
                    'merge' => true,
                    'split' => true,
                    'extract' => true,
                    'compress' => true,
                    'linearize' => true,
                ],
                'content_operations' => [
                    'flatten_forms' => true,
                    'flatten_annotations' => true,
                    'remove_annotations' => true,
                    'remove_javascript' => true,
                ],
            ],
        ];
    }
}

// ==================== SUPPORTING CLASSES ====================

/**
 * Page rotation angles.
 */
enum PageRotation: int
{
    case NONE = 0;
    case CLOCKWISE_90 = 90;
    case CLOCKWISE_180 = 180;
    case CLOCKWISE_270 = 270;

    public function getDescription(): string
    {
        return match($this) {
            self::NONE => 'No rotation',
            self::CLOCKWISE_90 => '90 degrees clockwise',
            self::CLOCKWISE_180 => '180 degrees',
            self::CLOCKWISE_270 => '270 degrees clockwise',
        };
    }
}

/**
 * Document split modes.
 */
enum SplitMode: int
{
    case EVERY_PAGE = 0;
    case EVERY_N_PAGES = 1;
    case BY_BOOKMARKS = 2;
    case BY_SIZE = 3;

    public function getDescription(): string
    {
        return match($this) {
            self::EVERY_PAGE => 'Split every page into separate file',
            self::EVERY_N_PAGES => 'Split every N pages',
            self::BY_BOOKMARKS => 'Split by bookmark boundaries',
            self::BY_SIZE => 'Split by file size limit',
        };
    }
}

/**
 * Compression levels.
 */
enum CompressionLevel: int
{
    case NONE = 0;
    case LOW = 1;
    case MEDIUM = 2;
    case HIGH = 3;
    case MAXIMUM = 4;

    public function getDescription(): string
    {
        return match($this) {
            self::NONE => 'No compression',
            self::LOW => 'Low compression, fast',
            self::MEDIUM => 'Balanced compression',
            self::HIGH => 'High compression',
            self::MAXIMUM => 'Maximum compression, slow',
        };
    }
}

/**
 * Compression result.
 */
readonly class CompressionResult
{
    public function __construct(
        public bool $success,
        public int $originalSize,
        public int $compressedSize
    ) {}

    public function getSavedBytes(): int
    {
        return $this->originalSize - $this->compressedSize;
    }

    public function getCompressionRatio(): float
    {
        if ($this->originalSize === 0) {
            return 0.0;
        }
        return 1.0 - ($this->compressedSize / $this->originalSize);
    }

    public function getCompressionPercentage(): float
    {
        return $this->getCompressionRatio() * 100;
    }

    public function toArray(): array
    {
        return [
            'success' => $this->success,
            'original_size' => $this->originalSize,
            'compressed_size' => $this->compressedSize,
            'saved_bytes' => $this->getSavedBytes(),
            'compression_ratio' => $this->getCompressionRatio(),
            'compression_percentage' => $this->getCompressionPercentage(),
        ];
    }
}
