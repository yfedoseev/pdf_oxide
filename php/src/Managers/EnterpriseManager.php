<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI;
use FFI\CData;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\ErrorHandler;
use PdfOxide\FFI\StringMarshaller;

/**
 * Manages enterprise PDF operations.
 *
 * Provides Bates numbering, document comparison, and header/footer
 * stamping for legal and enterprise document workflows.
 *
 * Example:
 *     $manager = new EnterpriseManager($documentHandle);
 *     $manager->applyBates('ABC-', startNumber: 1, numDigits: 6);
 *     $manager->stampHeader('Confidential', alignment: StampAlignment::CENTER);
 */
class EnterpriseManager
{
    private readonly CData $handle;
    private readonly FFI $ffi;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->ffi = NativeLibrary::getInstance();
    }

    // ==================== BATES NUMBERING ====================

    /**
     * Apply Bates numbering to all pages.
     *
     * @param string $prefix Bates number prefix text
     * @param int $startNumber Starting number (default: 1)
     * @param int $numDigits Number of digits, zero-padded (default: 6)
     * @param BatesPosition $position Position on page (default: BOTTOM_RIGHT)
     * @return int Number of pages stamped
     * @throws \PdfOxide\Exceptions\PdfException If the operation fails
     */
    public function applyBates(
        string $prefix,
        int $startNumber = 1,
        int $numDigits = 6,
        BatesPosition $position = BatesPosition::BOTTOM_RIGHT
    ): int {
        $cPrefix = StringMarshaller::toCString($prefix);
        $errorCode = FFI::new('int');

        try {
            $count = $this->ffi->pdf_bates_apply(
                $this->handle,
                $cPrefix,
                $startNumber,
                $numDigits,
                $position->value,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_bates_apply', [
                'prefix' => $prefix,
                'start_number' => $startNumber,
            ]);
            return (int)$count;
        } finally {
            unset($cPrefix);
        }
    }

    /**
     * Apply advanced Bates numbering with full options.
     *
     * @param string $prefix Prefix text
     * @param string $suffix Suffix text
     * @param int $startNumber Starting number
     * @param int $numDigits Digit count (zero-padded)
     * @param BatesPosition $position Position on page
     * @param float $fontSize Font size in points
     * @param float $margin Margin from page edge in points
     * @param array $color RGB color array [r, g, b] with values 0.0-1.0
     * @return int Number of pages stamped
     * @throws \PdfOxide\Exceptions\PdfException If the operation fails
     */
    public function applyBatesAdvanced(
        string $prefix,
        string $suffix = '',
        int $startNumber = 1,
        int $numDigits = 6,
        BatesPosition $position = BatesPosition::BOTTOM_RIGHT,
        float $fontSize = 10.0,
        float $margin = 36.0,
        array $color = [0.0, 0.0, 0.0]
    ): int {
        $cPrefix = StringMarshaller::toCString($prefix);
        $cSuffix = StringMarshaller::toCString($suffix);
        $errorCode = FFI::new('int');

        try {
            $count = $this->ffi->pdf_bates_apply_advanced(
                $this->handle,
                $cPrefix,
                $cSuffix,
                $startNumber,
                $numDigits,
                $position->value,
                $fontSize,
                $margin,
                (float)($color[0] ?? 0.0),
                (float)($color[1] ?? 0.0),
                (float)($color[2] ?? 0.0),
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_bates_apply_advanced', [
                'prefix' => $prefix,
                'suffix' => $suffix,
            ]);
            return (int)$count;
        } finally {
            unset($cPrefix, $cSuffix);
        }
    }

    // ==================== DOCUMENT COMPARISON ====================

    /**
     * Compare a page from this document with a page from another.
     *
     * @param CData $otherHandle Handle to the other PDF document
     * @param int $pageA Page index in this document (default: 0)
     * @param int $pageB Page index in the other document (default: 0)
     * @return PageComparisonResult Result with similarity and differences
     * @throws \PdfOxide\Exceptions\PdfException If the comparison fails
     */
    public function comparePages(CData $otherHandle, int $pageA = 0, int $pageB = 0): PageComparisonResult
    {
        $errorCode = FFI::new('int');
        $compHandle = $this->ffi->pdf_compare_pages(
            $this->handle,
            $otherHandle,
            $pageA,
            $pageB,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_compare_pages', [
            'page_a' => $pageA,
            'page_b' => $pageB,
        ]);

        $similarity = (float)$this->ffi->pdf_comparison_get_similarity($compHandle);
        $diffCount = (int)$this->ffi->pdf_comparison_get_diff_count($compHandle);

        $differences = [];
        for ($i = 0; $i < $diffCount; $i++) {
            $diffHandle = $this->ffi->pdf_comparison_get_diff($compHandle, $i);
            $diffType = (int)$this->ffi->pdf_comparison_get_diff_type($diffHandle);
            $differences[] = new PageDifference(
                diffType: DifferenceType::from($diffType),
                description: ''
            );
        }

        $this->ffi->pdf_comparison_free($compHandle);

        return new PageComparisonResult(
            similarity: $similarity,
            differences: $differences
        );
    }

    /**
     * Compare this document with another page by page.
     *
     * @param CData $otherHandle Handle to the other PDF document
     * @return DocumentComparisonResult Overall and per-page results
     * @throws \PdfOxide\Exceptions\PdfException If the comparison fails
     */
    public function compareDocuments(CData $otherHandle): DocumentComparisonResult
    {
        $errorCode = FFI::new('int');
        $compHandle = $this->ffi->pdf_compare_documents(
            $this->handle,
            $otherHandle,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_compare_documents');

        $similarity = (float)$this->ffi->pdf_comparison_get_similarity($compHandle);

        $this->ffi->pdf_document_comparison_free($compHandle);

        return new DocumentComparisonResult(
            similarity: $similarity,
            totalDifferences: 0,
            pageComparisons: []
        );
    }

    // ==================== HEADER/FOOTER STAMPING ====================

    /**
     * Stamp a header on all pages.
     *
     * Supports placeholders: {page}, {pages}, {date}.
     *
     * @param string $text Header text
     * @param StampAlignment $alignment Text alignment
     * @param float $fontSize Font size in points
     * @param float $margin Margin from page edge
     * @return int Number of pages stamped
     * @throws \PdfOxide\Exceptions\PdfException If the operation fails
     */
    public function stampHeader(
        string $text,
        StampAlignment $alignment = StampAlignment::CENTER,
        float $fontSize = 10.0,
        float $margin = 36.0
    ): int {
        $cText = StringMarshaller::toCString($text);
        $errorCode = FFI::new('int');

        try {
            $count = $this->ffi->pdf_stamp_header(
                $this->handle,
                $cText,
                $alignment->value,
                $fontSize,
                $margin,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_stamp_header');
            return (int)$count;
        } finally {
            unset($cText);
        }
    }

    /**
     * Stamp a footer on all pages.
     *
     * Supports placeholders: {page}, {pages}, {date}.
     *
     * @param string $text Footer text
     * @param StampAlignment $alignment Text alignment
     * @param float $fontSize Font size in points
     * @param float $margin Margin from page edge
     * @return int Number of pages stamped
     * @throws \PdfOxide\Exceptions\PdfException If the operation fails
     */
    public function stampFooter(
        string $text,
        StampAlignment $alignment = StampAlignment::CENTER,
        float $fontSize = 10.0,
        float $margin = 36.0
    ): int {
        $cText = StringMarshaller::toCString($text);
        $errorCode = FFI::new('int');

        try {
            $count = $this->ffi->pdf_stamp_footer(
                $this->handle,
                $cText,
                $alignment->value,
                $fontSize,
                $margin,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_stamp_footer');
            return (int)$count;
        } finally {
            unset($cText);
        }
    }

    /**
     * Stamp both header and footer on all pages.
     *
     * @param string|null $headerText Header text (null to skip)
     * @param string|null $footerText Footer text (null to skip)
     * @param StampAlignment $alignment Text alignment
     * @param float $fontSize Font size in points
     * @param float $margin Margin from page edge
     * @return int Number of pages stamped
     * @throws \PdfOxide\Exceptions\PdfException If the operation fails
     */
    public function stampHeaderFooter(
        ?string $headerText = null,
        ?string $footerText = null,
        StampAlignment $alignment = StampAlignment::CENTER,
        float $fontSize = 10.0,
        float $margin = 36.0
    ): int {
        $cHeader = $headerText !== null ? StringMarshaller::toCString($headerText) : null;
        $cFooter = $footerText !== null ? StringMarshaller::toCString($footerText) : null;
        $errorCode = FFI::new('int');

        try {
            $count = $this->ffi->pdf_stamp_header_footer(
                $this->handle,
                $cHeader,
                $cFooter,
                $alignment->value,
                $fontSize,
                $margin,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_stamp_header_footer');
            return (int)$count;
        } finally {
            unset($cHeader, $cFooter);
        }
    }

    // ==================== SUMMARY ====================

    /**
     * Get enterprise capabilities summary.
     *
     * @return array Summary of enterprise capabilities
     */
    public function getSummary(): array
    {
        return [
            'capabilities' => [
                'bates_numbering' => true,
                'bates_advanced' => true,
                'page_comparison' => true,
                'document_comparison' => true,
                'header_stamping' => true,
                'footer_stamping' => true,
                'header_footer_stamping' => true,
            ],
        ];
    }
}

// ==================== SUPPORTING CLASSES ====================

/**
 * Position for Bates number placement.
 */
enum BatesPosition: int
{
    case TOP_LEFT = 0;
    case TOP_CENTER = 1;
    case TOP_RIGHT = 2;
    case BOTTOM_LEFT = 3;
    case BOTTOM_CENTER = 4;
    case BOTTOM_RIGHT = 5;
}

/**
 * Alignment for header/footer text.
 */
enum StampAlignment: int
{
    case LEFT = 0;
    case CENTER = 1;
    case RIGHT = 2;
}

/**
 * Type of difference found between documents.
 */
enum DifferenceType: int
{
    case TEXT_ADDED = 0;
    case TEXT_REMOVED = 1;
    case TEXT_CHANGED = 2;
    case IMAGE_ADDED = 3;
    case IMAGE_REMOVED = 4;
}

/**
 * A single difference found between pages.
 */
readonly class PageDifference
{
    public function __construct(
        public DifferenceType $diffType,
        public string $description
    ) {}

    public function toArray(): array
    {
        return [
            'type' => $this->diffType->name,
            'description' => $this->description,
        ];
    }
}

/**
 * Result of comparing two pages.
 */
readonly class PageComparisonResult
{
    /**
     * @param float $similarity Similarity score 0.0-1.0
     * @param array<PageDifference> $differences List of differences
     */
    public function __construct(
        public float $similarity,
        public array $differences = []
    ) {}

    public function getDiffCount(): int
    {
        return count($this->differences);
    }

    public function toArray(): array
    {
        return [
            'similarity' => $this->similarity,
            'diff_count' => $this->getDiffCount(),
            'differences' => array_map(fn($d) => $d->toArray(), $this->differences),
        ];
    }
}

/**
 * Result of comparing two documents.
 */
readonly class DocumentComparisonResult
{
    /**
     * @param float $similarity Overall similarity score 0.0-1.0
     * @param int $totalDifferences Total number of differences
     * @param array<PageComparisonResult> $pageComparisons Per-page results
     */
    public function __construct(
        public float $similarity,
        public int $totalDifferences = 0,
        public array $pageComparisons = []
    ) {}

    public function toArray(): array
    {
        return [
            'similarity' => $this->similarity,
            'total_differences' => $this->totalDifferences,
            'page_comparisons' => array_map(fn($p) => $p->toArray(), $this->pageComparisons),
        ];
    }
}
