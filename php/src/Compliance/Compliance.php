<?php

declare(strict_types=1);

namespace PdfOxide\Compliance;

use PdfOxide\PdfDocument;
use PdfOxide\Types\ComplianceResult;

/**
 * Static utility for PDF compliance operations.
 *
 * Provides convenient static methods for converting and validating PDF documents
 * against various compliance standards including PDF/A, PDF/X, and PDF/UA.
 *
 * Example:
 *     // Convert to PDF/A-2B
 *     $doc = new PdfDocument('input.pdf');
 *     $pdfaBytes = Compliance::convertToPdfA($doc, '2b');
 *     file_put_contents('output_a2b.pdf', $pdfaBytes);
 *
 *     // Validate against PDF/A-3A
 *     $result = Compliance::validatePdfA($doc, '3a');
 *     if ($result->isCompliant()) {
 *         echo "Document is PDF/A-3A compliant";
 *     }
 *
 *     // Convert to PDF/UA (accessible)
 *     $uaBytes = Compliance::convertToPdfUa($doc);
 *
 *     // Validate PDF/X-4
 *     $xResult = Compliance::validatePdfX($doc, '4');
 *
 * @since 0.4.0
 */
final class Compliance
{
    /**
     * Supported PDF/A levels.
     */
    public const PDFA_LEVELS = ['1a', '1b', '2a', '2b', '3a', '3b'];

    /**
     * Supported PDF/X standards.
     */
    public const PDFX_STANDARDS = ['1a', '3', '4'];

    /**
     * Convert PDF document to PDF/A format.
     *
     * @param PdfDocument $doc Document to convert
     * @param string $level PDF/A level ('1a', '1b', '2a', '2b', '3a', '3b')
     * @return string Binary PDF/A document bytes
     * @throws \InvalidArgumentException If level is invalid
     * @throws \RuntimeException If conversion fails
     *
     * PDF/A is an ISO-standardized subset of PDF designed for long-term archival.
     * Different levels offer varying features and requirements.
     *
     * Level Details:
     * - 1a: PDF/A-1, part A (archived), highest conformance with structure
     * - 1b: PDF/A-1, part B (basic), visual appearance preservation
     * - 2a: PDF/A-2, part A, supports transparency and JPEG2000
     * - 2b: PDF/A-2, part B, more lenient than 2a
     * - 3a: PDF/A-3, part A, allows embedding of arbitrary files
     * - 3b: PDF/A-3, part B, most permissive level
     *
     * Example:
     *     $bytes = Compliance::convertToPdfA($doc, '2b');
     *     file_put_contents('archived.pdf', $bytes);
     */
    public static function convertToPdfA(PdfDocument $doc, string $level): string
    {
        $level = strtolower(trim($level));
        self::validatePdfALevel($level);

        try {
            $doc->compliance()->convertToPdfA($level);
            return $doc->toBytes();
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to convert document to PDF/A-{$level}: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Convert PDF document to PDF/UA (accessible) format.
     *
     * @param PdfDocument $doc Document to convert
     * @return string Binary PDF/UA document bytes
     * @throws \RuntimeException If conversion fails
     *
     * PDF/UA (Universal Accessibility) ensures PDFs are accessible to users
     * with disabilities by requiring proper document structure, tagged content,
     * and metadata.
     *
     * Example:
     *     $bytes = Compliance::convertToPdfUa($doc);
     *     file_put_contents('accessible.pdf', $bytes);
     */
    public static function convertToPdfUa(PdfDocument $doc): string
    {
        try {
            $doc->compliance()->convertToPdfUa();
            return $doc->toBytes();
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to convert document to PDF/UA: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Convert PDF document to PDF/X format.
     *
     * @param PdfDocument $doc Document to convert
     * @param string $standard PDF/X standard ('1a', '3', '4')
     * @return string Binary PDF/X document bytes
     * @throws \InvalidArgumentException If standard is invalid
     * @throws \RuntimeException If conversion fails
     *
     * PDF/X is an ISO standard for print production, ensuring reliable
     * data exchange in printing workflows.
     *
     * Standard Details:
     * - 1a: PDF/X-1:2001, requires CMYK colors, no transparency
     * - 3: PDF/X-3:2002, allows spot colors and RGB, no transparency
     * - 4: PDF/X-4:2010, allows transparency and external color profiles
     *
     * Example:
     *     $bytes = Compliance::convertToPdfX($doc, '4');
     *     file_put_contents('print_ready.pdf', $bytes);
     */
    public static function convertToPdfX(PdfDocument $doc, string $standard): string
    {
        $standard = strtolower(trim($standard));
        self::validatePdfXStandard($standard);

        try {
            $doc->compliance()->convertToPdfX($standard);
            return $doc->toBytes();
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to convert document to PDF/X-{$standard}: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Validate PDF/A compliance.
     *
     * @param PdfDocument $doc Document to validate
     * @param string $level PDF/A level to validate against ('1a', '1b', '2a', '2b', '3a', '3b')
     * @return ComplianceResult Validation result with errors and warnings
     * @throws \InvalidArgumentException If level is invalid
     * @throws \RuntimeException If validation fails
     *
     * Example:
     *     $result = Compliance::validatePdfA($doc, '2a');
     *     if ($result->isCompliant()) {
     *         echo "Document meets PDF/A-2A standard";
     *     } else {
     *         foreach ($result->getErrors() as $error) {
     *             echo "Error: {$error}\n";
     *         }
     *     }
     */
    public static function validatePdfA(PdfDocument $doc, string $level): ComplianceResult
    {
        $level = strtolower(trim($level));
        self::validatePdfALevel($level);

        try {
            return $doc->compliance()->validatePdfA($level);
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "PDF/A validation failed: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Validate PDF/UA (accessibility) compliance.
     *
     * @param PdfDocument $doc Document to validate
     * @return ComplianceResult Validation result with errors and warnings
     * @throws \RuntimeException If validation fails
     *
     * Example:
     *     $result = Compliance::validatePdfUa($doc);
     *     echo "Accessibility issues: " . $result->getErrorCount();
     */
    public static function validatePdfUa(PdfDocument $doc): ComplianceResult
    {
        try {
            return $doc->compliance()->validatePdfUa();
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "PDF/UA validation failed: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Validate PDF/X compliance.
     *
     * @param PdfDocument $doc Document to validate
     * @param string $standard PDF/X standard to validate against ('1a', '3', '4')
     * @return ComplianceResult Validation result with errors and warnings
     * @throws \InvalidArgumentException If standard is invalid
     * @throws \RuntimeException If validation fails
     *
     * Example:
     *     $result = Compliance::validatePdfX($doc, '4');
     *     if (!$result->isCompliant()) {
     *         echo "Print production issues: " . $result->getErrorCount();
     *     }
     */
    public static function validatePdfX(PdfDocument $doc, string $standard): ComplianceResult
    {
        $standard = strtolower(trim($standard));
        self::validatePdfXStandard($standard);

        try {
            return $doc->compliance()->validatePdfX($standard);
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "PDF/X validation failed: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Check if PDF/A level is valid.
     *
     * @param string $level Level to validate
     * @return bool True if valid
     *
     * Example:
     *     if (Compliance::isValidPdfALevel('2b')) {
     *         // Process level
     *     }
     */
    public static function isValidPdfALevel(string $level): bool
    {
        return in_array(strtolower(trim($level)), self::PDFA_LEVELS);
    }

    /**
     * Check if PDF/X standard is valid.
     *
     * @param string $standard Standard to validate
     * @return bool True if valid
     *
     * Example:
     *     if (Compliance::isValidPdfXStandard('4')) {
     *         // Process standard
     *     }
     */
    public static function isValidPdfXStandard(string $standard): bool
    {
        return in_array(strtolower(trim($standard)), self::PDFX_STANDARDS);
    }

    /**
     * Get supported PDF/A levels.
     *
     * @return string[] Array of supported levels
     */
    public static function getSupportedPdfALevels(): array
    {
        return self::PDFA_LEVELS;
    }

    /**
     * Get supported PDF/X standards.
     *
     * @return string[] Array of supported standards
     */
    public static function getSupportedPdfXStandards(): array
    {
        return self::PDFX_STANDARDS;
    }

    /**
     * Validate PDF/A level and throw if invalid.
     *
     * @param string $level Level to validate
     * @throws \InvalidArgumentException If level is invalid
     */
    private static function validatePdfALevel(string $level): void
    {
        if (!self::isValidPdfALevel($level)) {
            throw new \InvalidArgumentException(
                "Invalid PDF/A level: {$level}. Supported levels: " . implode(', ', self::PDFA_LEVELS)
            );
        }
    }

    /**
     * Validate PDF/X standard and throw if invalid.
     *
     * @param string $standard Standard to validate
     * @throws \InvalidArgumentException If standard is invalid
     */
    private static function validatePdfXStandard(string $standard): void
    {
        if (!self::isValidPdfXStandard($standard)) {
            throw new \InvalidArgumentException(
                "Invalid PDF/X standard: {$standard}. Supported standards: " . implode(', ', self::PDFX_STANDARDS)
            );
        }
    }

    /**
     * Private constructor to prevent instantiation.
     * This is a static utility class.
     */
    private function __construct()
    {
        // Static class, not instantiable
    }
}
