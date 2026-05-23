<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI;
use FFI\CData;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\ErrorHandler;

/**
 * Manages PDF optimization operations.
 *
 * Provides font subsetting, image downsampling, content deduplication,
 * and combined optimization for reducing PDF file size.
 *
 * Example:
 *     $manager = new OptimizationManager($documentHandle);
 *     $saved = $manager->subsetFonts();
 *     $result = $manager->optimizeFull(targetDpi: 150, quality: 85);
 */
class OptimizationManager
{
    private readonly CData $handle;
    private readonly FFI $ffi;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->ffi = NativeLibrary::getInstance();
    }

    // ==================== FONT OPTIMIZATION ====================

    /**
     * Subset fonts to remove unused glyphs.
     *
     * @return int Estimated bytes saved
     * @throws \PdfOxide\Exceptions\OptimizationException If the operation fails
     */
    public function subsetFonts(): int
    {
        $errorCode = FFI::new('int');
        $saved = $this->ffi->pdf_optimize_subset_fonts(
            $this->handle,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_optimize_subset_fonts');
        return (int)$saved;
    }

    // ==================== IMAGE OPTIMIZATION ====================

    /**
     * Downsample images to reduce file size.
     *
     * @param int $targetDpi Target DPI for downsampling (default: 150)
     * @param int $quality JPEG quality 1-100 (default: 85)
     * @return int Estimated bytes saved
     * @throws \PdfOxide\Exceptions\OptimizationException If the operation fails
     */
    public function downsampleImages(int $targetDpi = 150, int $quality = 85): int
    {
        $errorCode = FFI::new('int');
        $saved = $this->ffi->pdf_optimize_downsample_images(
            $this->handle,
            $targetDpi,
            $quality,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_optimize_downsample_images', [
            'target_dpi' => $targetDpi,
            'quality' => $quality,
        ]);
        return (int)$saved;
    }

    // ==================== DEDUPLICATION ====================

    /**
     * Deduplicate identical content streams and objects.
     *
     * @return int Estimated bytes saved
     * @throws \PdfOxide\Exceptions\OptimizationException If the operation fails
     */
    public function deduplicate(): int
    {
        $errorCode = FFI::new('int');
        $saved = $this->ffi->pdf_optimize_deduplicate(
            $this->handle,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_optimize_deduplicate');
        return (int)$saved;
    }

    // ==================== FULL OPTIMIZATION ====================

    /**
     * Run full optimization pipeline (fonts + images + dedup).
     *
     * @param int $targetDpi Target DPI for image downsampling (default: 150)
     * @param int $quality JPEG quality 1-100 (default: 85)
     * @return OptimizationFullResult Result with total bytes saved
     * @throws \PdfOxide\Exceptions\OptimizationException If the operation fails
     */
    public function optimizeFull(int $targetDpi = 150, int $quality = 85): OptimizationFullResult
    {
        $errorCode = FFI::new('int');
        $resultHandle = $this->ffi->pdf_optimize_full(
            $this->handle,
            $targetDpi,
            $quality,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_optimize_full', [
            'target_dpi' => $targetDpi,
            'quality' => $quality,
        ]);

        $bytesSaved = (int)$this->ffi->pdf_optimization_result_bytes_saved($resultHandle);
        $this->ffi->pdf_optimization_result_free($resultHandle);

        return new OptimizationFullResult(
            bytesSaved: $bytesSaved
        );
    }

    // ==================== SUMMARY ====================

    /**
     * Get optimization capabilities summary.
     *
     * @return array Summary of optimization capabilities
     */
    public function getSummary(): array
    {
        return [
            'capabilities' => [
                'subset_fonts' => true,
                'downsample_images' => true,
                'deduplicate' => true,
                'full_optimization' => true,
            ],
        ];
    }
}

// ==================== SUPPORTING CLASSES ====================

/**
 * Result of a full optimization run.
 */
readonly class OptimizationFullResult
{
    public function __construct(
        public int $bytesSaved,
        public string $details = ''
    ) {}

    public function toArray(): array
    {
        return [
            'bytes_saved' => $this->bytesSaved,
            'details' => $this->details,
        ];
    }
}
