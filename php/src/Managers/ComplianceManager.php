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
 * Manages PDF compliance and validation operations.
 *
 * Handles comprehensive PDF/A (archival), PDF/X (print), and PDF/UA (accessibility)
 * validation, conversion, and issue reporting.
 * Uses PHP 8+ features for clean, type-safe implementation.
 */
class ComplianceManager
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

    // ==================== PDF/A VALIDATION ====================

    /**
     * Validate PDF/A compliance.
     *
     * @param PdfALevel $level PDF/A conformance level
     * @return PdfAValidationResult Validation result
     */
    public function validatePdfA(PdfALevel $level): PdfAValidationResult
    {
        $errorCode = FFI::new('int');
        $resultHandle = $this->ffi->pdf_validate_pdf_a($this->handle, $level->value, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_validate_pdf_a');

        return new PdfAValidationResult($resultHandle, $this->ffi, $level);
    }

    /**
     * Quick check if PDF/A compliant.
     *
     * @param PdfALevel $level PDF/A level
     * @return bool True if compliant
     */
    public function isPdfACompliant(PdfALevel $level): bool
    {
        $result = $this->validatePdfA($level);
        return $result->isCompliant();
    }

    /**
     * Convert document to PDF/A.
     *
     * @param PdfALevel $level Target PDF/A level
     * @return ConversionResult Conversion result
     */
    public function convertToPdfA(PdfALevel $level = PdfALevel::A2B): ConversionResult
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_convert_to_pdf_a($this->handle, $level->value, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_convert_to_pdf_a');

        return new ConversionResult(
            success: (bool)$result,
            standard: 'PDF/A',
            level: $level->name
        );
    }

    /**
     * Validate all PDF/A levels and return best match.
     *
     * @return array<PdfALevel, PdfAValidationResult> Results by level
     */
    public function validateAllPdfALevels(): array
    {
        $results = [];
        foreach (PdfALevel::cases() as $level) {
            try {
                $results[$level->name] = $this->validatePdfA($level);
            } catch (\Exception $e) {
                // Skip levels that fail validation
            }
        }
        return $results;
    }

    /**
     * Get best PDF/A conformance level.
     *
     * @return PdfALevel|null Best compliant level or null
     */
    public function getBestPdfALevel(): ?PdfALevel
    {
        // Check from highest to lowest level
        $levels = [
            PdfALevel::A3A,
            PdfALevel::A3U,
            PdfALevel::A3B,
            PdfALevel::A2A,
            PdfALevel::A2U,
            PdfALevel::A2B,
            PdfALevel::A1A,
            PdfALevel::A1B,
        ];

        foreach ($levels as $level) {
            if ($this->isPdfACompliant($level)) {
                return $level;
            }
        }

        return null;
    }

    // ==================== PDF/X VALIDATION ====================

    /**
     * Validate PDF/X compliance.
     *
     * @param PdfXLevel $level PDF/X conformance level
     * @return PdfXValidationResult Validation result
     */
    public function validatePdfX(PdfXLevel $level): PdfXValidationResult
    {
        $errorCode = FFI::new('int');
        $resultHandle = $this->ffi->pdf_validate_pdf_x($this->handle, $level->value, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_validate_pdf_x');

        return new PdfXValidationResult($resultHandle, $this->ffi, $level);
    }

    /**
     * Quick check if PDF/X compliant.
     *
     * @param PdfXLevel $level PDF/X level
     * @return bool True if compliant
     */
    public function isPdfXCompliant(PdfXLevel $level): bool
    {
        $result = $this->validatePdfX($level);
        return $result->isCompliant();
    }

    /**
     * Convert document to PDF/X.
     *
     * @param PdfXLevel $level Target PDF/X level
     * @return ConversionResult Conversion result
     */
    public function convertToPdfX(PdfXLevel $level = PdfXLevel::X4): ConversionResult
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_convert_to_pdf_x($this->handle, $level->value, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_convert_to_pdf_x');

        return new ConversionResult(
            success: (bool)$result,
            standard: 'PDF/X',
            level: $level->name
        );
    }

    // ==================== PDF/UA VALIDATION ====================

    /**
     * Validate PDF/UA accessibility.
     *
     * @param PdfUaLevel $level PDF/UA conformance level
     * @return PdfUaValidationResult Validation result
     */
    public function validatePdfUa(PdfUaLevel $level = PdfUaLevel::UA1): PdfUaValidationResult
    {
        $errorCode = FFI::new('int');
        $resultHandle = $this->ffi->pdf_validate_pdf_ua($this->handle, $level->value, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_validate_pdf_ua');

        return new PdfUaValidationResult($resultHandle, $this->ffi, $level);
    }

    /**
     * Quick check if PDF/UA accessible.
     *
     * @param PdfUaLevel $level PDF/UA level
     * @return bool True if accessible
     */
    public function isPdfUaAccessible(PdfUaLevel $level = PdfUaLevel::UA1): bool
    {
        $result = $this->validatePdfUa($level);
        return $result->isAccessible();
    }

    /**
     * Convert document to PDF/UA.
     *
     * @param PdfUaLevel $level Target PDF/UA level
     * @return ConversionResult Conversion result
     */
    public function convertToPdfUa(PdfUaLevel $level = PdfUaLevel::UA1): ConversionResult
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_convert_to_pdf_ua($this->handle, $level->value, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_convert_to_pdf_ua');

        return new ConversionResult(
            success: (bool)$result,
            standard: 'PDF/UA',
            level: $level->name
        );
    }

    // ==================== COMPREHENSIVE VALIDATION ====================

    /**
     * Validate all standards at once.
     *
     * @return ComprehensiveValidationResult Combined validation results
     */
    public function validateAll(): ComprehensiveValidationResult
    {
        $pdfA = null;
        $pdfX = null;
        $pdfUa = null;

        try {
            $pdfA = $this->validatePdfA(PdfALevel::A2B);
        } catch (\Exception $e) {
            // PDF/A validation failed
        }

        try {
            $pdfX = $this->validatePdfX(PdfXLevel::X4);
        } catch (\Exception $e) {
            // PDF/X validation failed
        }

        try {
            $pdfUa = $this->validatePdfUa(PdfUaLevel::UA1);
        } catch (\Exception $e) {
            // PDF/UA validation failed
        }

        return new ComprehensiveValidationResult($pdfA, $pdfX, $pdfUa);
    }

    /**
     * Get supported PDF/A levels.
     *
     * @return string[] Available PDF/A levels
     */
    public static function getPdfALevels(): array
    {
        return array_map(fn($l) => $l->name, PdfALevel::cases());
    }

    /**
     * Get supported PDF/X standards.
     *
     * @return string[] Available PDF/X standards
     */
    public static function getPdfXStandards(): array
    {
        return array_map(fn($l) => $l->name, PdfXLevel::cases());
    }

    /**
     * Get compliance summary.
     *
     * @return array Summary information
     */
    public function getSummary(): array
    {
        return [
            'pdf_a' => [
                'levels' => self::getPdfALevels(),
                'description' => 'Archival and long-term preservation',
            ],
            'pdf_x' => [
                'levels' => self::getPdfXStandards(),
                'description' => 'Print production and prepress',
            ],
            'pdf_ua' => [
                'levels' => array_map(fn($l) => $l->name, PdfUaLevel::cases()),
                'description' => 'Universal accessibility',
            ],
            'capabilities' => [
                'validation' => true,
                'conversion' => true,
                'issue_reporting' => true,
                'auto_fix' => true,
            ],
        ];
    }
}

// ==================== SUPPORTING CLASSES ====================

/**
 * PDF/A conformance levels.
 */
enum PdfALevel: int
{
    case A1B = 0;  // PDF/A-1b (basic)
    case A1A = 1;  // PDF/A-1a (accessible)
    case A2B = 2;  // PDF/A-2b (basic)
    case A2A = 3;  // PDF/A-2a (accessible)
    case A2U = 4;  // PDF/A-2u (unicode)
    case A3B = 5;  // PDF/A-3b (basic + attachments)
    case A3A = 6;  // PDF/A-3a (accessible + attachments)
    case A3U = 7;  // PDF/A-3u (unicode + attachments)

    public function getDescription(): string
    {
        return match($this) {
            self::A1B => 'PDF/A-1b: Basic visual appearance preservation',
            self::A1A => 'PDF/A-1a: Accessible (tagged, unicode mapped)',
            self::A2B => 'PDF/A-2b: Basic with JPEG2000, transparency',
            self::A2A => 'PDF/A-2a: Accessible (tagged, unicode mapped)',
            self::A2U => 'PDF/A-2u: Unicode text mapping',
            self::A3B => 'PDF/A-3b: Basic with embedded files',
            self::A3A => 'PDF/A-3a: Accessible with embedded files',
            self::A3U => 'PDF/A-3u: Unicode with embedded files',
        };
    }

    public function getIsoReference(): string
    {
        return match($this) {
            self::A1B, self::A1A => 'ISO 19005-1:2005',
            self::A2B, self::A2A, self::A2U => 'ISO 19005-2:2011',
            self::A3B, self::A3A, self::A3U => 'ISO 19005-3:2012',
        };
    }
}

/**
 * PDF/X conformance levels.
 */
enum PdfXLevel: int
{
    case X1A_2001 = 0;
    case X1A_2003 = 1;
    case X3_2003 = 2;
    case X4 = 3;
    case X5 = 4;
    case X6 = 5;

    public function getDescription(): string
    {
        return match($this) {
            self::X1A_2001 => 'PDF/X-1a:2001: CMYK/spot colors only',
            self::X1A_2003 => 'PDF/X-1a:2003: CMYK/spot colors only (updated)',
            self::X3_2003 => 'PDF/X-3:2003: Color-managed workflows',
            self::X4 => 'PDF/X-4: Live transparency, ICC-based colors',
            self::X5 => 'PDF/X-5: External graphics references',
            self::X6 => 'PDF/X-6: Variable data printing',
        };
    }
}

/**
 * PDF/UA conformance levels.
 */
enum PdfUaLevel: int
{
    case UA1 = 0;

    public function getDescription(): string
    {
        return match($this) {
            self::UA1 => 'PDF/UA-1: Universal accessibility (ISO 14289-1)',
        };
    }
}

/**
 * Compliance issue severity.
 */
enum ComplianceSeverity: string
{
    case ERROR = 'error';
    case WARNING = 'warning';
    case INFO = 'info';
}

/**
 * Compliance issue.
 */
readonly class ComplianceIssue
{
    public function __construct(
        public int $code,
        public string $message,
        public ComplianceSeverity $severity,
        public ?int $pageIndex = null,
        public ?string $objectId = null
    ) {}

    public function toArray(): array
    {
        return [
            'code' => $this->code,
            'message' => $this->message,
            'severity' => $this->severity->value,
            'page_index' => $this->pageIndex,
            'object_id' => $this->objectId,
        ];
    }
}

/**
 * PDF/A validation result.
 */
class PdfAValidationResult
{
    private CData $handle;
    private FFI $ffi;
    private PdfALevel $level;
    private ?array $cachedErrors = null;
    private ?array $cachedWarnings = null;

    public function __construct(CData $handle, FFI $ffi, PdfALevel $level)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
        $this->level = $level;
    }

    public function isCompliant(): bool
    {
        return (bool)$this->ffi->pdf_pdf_a_is_compliant($this->handle);
    }

    public function getLevel(): PdfALevel
    {
        return $this->level;
    }

    public function getErrorCount(): int
    {
        return (int)$this->ffi->pdf_pdf_a_error_count($this->handle);
    }

    public function getWarningCount(): int
    {
        return (int)$this->ffi->pdf_pdf_a_warning_count($this->handle);
    }

    public function getErrors(): array
    {
        if ($this->cachedErrors !== null) {
            return $this->cachedErrors;
        }

        $this->cachedErrors = [];
        $count = $this->getErrorCount();

        for ($i = 0; $i < $count; $i++) {
            $errorCode = FFI::new('int');
            $issuePtr = $this->ffi->pdf_pdf_a_get_error($this->handle, $i, FFI::addr($errorCode));

            if ($issuePtr !== null) {
                $this->cachedErrors[] = new ComplianceIssue(
                    code: (int)$issuePtr->code,
                    message: StringMarshaller::fromCString($issuePtr->message, false),
                    severity: ComplianceSeverity::ERROR
                );
                $this->ffi->pdf_compliance_issue_free($issuePtr);
            }
        }

        return $this->cachedErrors;
    }

    public function getWarnings(): array
    {
        if ($this->cachedWarnings !== null) {
            return $this->cachedWarnings;
        }

        $this->cachedWarnings = [];
        $count = $this->getWarningCount();

        for ($i = 0; $i < $count; $i++) {
            $errorCode = FFI::new('int');
            $issuePtr = $this->ffi->pdf_pdf_a_get_warning($this->handle, $i, FFI::addr($errorCode));

            if ($issuePtr !== null) {
                $this->cachedWarnings[] = new ComplianceIssue(
                    code: (int)$issuePtr->code,
                    message: StringMarshaller::fromCString($issuePtr->message, false),
                    severity: ComplianceSeverity::WARNING
                );
                $this->ffi->pdf_compliance_issue_free($issuePtr);
            }
        }

        return $this->cachedWarnings;
    }

    public function getReport(): string
    {
        $errorCode = FFI::new('int');
        $reportPtr = $this->ffi->pdf_pdf_a_get_report($this->handle, FFI::addr($errorCode));
        return StringMarshaller::fromCString($reportPtr);
    }

    public function toArray(): array
    {
        return [
            'level' => $this->level->name,
            'compliant' => $this->isCompliant(),
            'error_count' => $this->getErrorCount(),
            'warning_count' => $this->getWarningCount(),
            'errors' => array_map(fn($e) => $e->toArray(), $this->getErrors()),
            'warnings' => array_map(fn($w) => $w->toArray(), $this->getWarnings()),
        ];
    }

    public function __destruct()
    {
        $this->ffi->pdf_pdf_a_results_free($this->handle);
    }
}

/**
 * PDF/X validation result.
 */
class PdfXValidationResult
{
    private CData $handle;
    private FFI $ffi;
    private PdfXLevel $level;

    public function __construct(CData $handle, FFI $ffi, PdfXLevel $level)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
        $this->level = $level;
    }

    public function isCompliant(): bool
    {
        return (bool)$this->ffi->pdf_pdf_x_is_compliant($this->handle);
    }

    public function getLevel(): PdfXLevel
    {
        return $this->level;
    }

    public function getErrorCount(): int
    {
        return (int)$this->ffi->pdf_pdf_x_error_count($this->handle);
    }

    public function getWarningCount(): int
    {
        return (int)$this->ffi->pdf_pdf_x_warning_count($this->handle);
    }

    public function getErrors(): array
    {
        $errors = [];
        $count = $this->getErrorCount();

        for ($i = 0; $i < $count; $i++) {
            $errorCode = FFI::new('int');
            $issuePtr = $this->ffi->pdf_pdf_x_get_error($this->handle, $i, FFI::addr($errorCode));

            if ($issuePtr !== null) {
                $errors[] = new ComplianceIssue(
                    code: (int)$issuePtr->code,
                    message: StringMarshaller::fromCString($issuePtr->message, false),
                    severity: ComplianceSeverity::ERROR
                );
                $this->ffi->pdf_compliance_issue_free($issuePtr);
            }
        }

        return $errors;
    }

    public function getReport(): string
    {
        $errorCode = FFI::new('int');
        $reportPtr = $this->ffi->pdf_pdf_x_get_report($this->handle, FFI::addr($errorCode));
        return StringMarshaller::fromCString($reportPtr);
    }

    public function toArray(): array
    {
        return [
            'level' => $this->level->name,
            'compliant' => $this->isCompliant(),
            'error_count' => $this->getErrorCount(),
            'warning_count' => $this->getWarningCount(),
            'errors' => array_map(fn($e) => $e->toArray(), $this->getErrors()),
        ];
    }

    public function __destruct()
    {
        $this->ffi->pdf_pdf_x_results_free($this->handle);
    }
}

/**
 * PDF/UA validation result.
 */
class PdfUaValidationResult
{
    private CData $handle;
    private FFI $ffi;
    private PdfUaLevel $level;

    public function __construct(CData $handle, FFI $ffi, PdfUaLevel $level)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
        $this->level = $level;
    }

    public function isAccessible(): bool
    {
        return (bool)$this->ffi->pdf_pdf_ua_is_accessible($this->handle);
    }

    public function getLevel(): PdfUaLevel
    {
        return $this->level;
    }

    public function getErrorCount(): int
    {
        return (int)$this->ffi->pdf_pdf_ua_error_count($this->handle);
    }

    public function getErrors(): array
    {
        $errors = [];
        $count = $this->getErrorCount();

        for ($i = 0; $i < $count; $i++) {
            $errorCode = FFI::new('int');
            $issuePtr = $this->ffi->pdf_pdf_ua_get_error($this->handle, $i, FFI::addr($errorCode));

            if ($issuePtr !== null) {
                $errors[] = new ComplianceIssue(
                    code: (int)$issuePtr->code,
                    message: StringMarshaller::fromCString($issuePtr->message, false),
                    severity: ComplianceSeverity::ERROR
                );
                $this->ffi->pdf_compliance_issue_free($issuePtr);
            }
        }

        return $errors;
    }

    public function toArray(): array
    {
        return [
            'level' => $this->level->name,
            'accessible' => $this->isAccessible(),
            'error_count' => $this->getErrorCount(),
            'errors' => array_map(fn($e) => $e->toArray(), $this->getErrors()),
        ];
    }

    public function __destruct()
    {
        $this->ffi->pdf_pdf_ua_results_free($this->handle);
    }
}

/**
 * Conversion result.
 */
readonly class ConversionResult
{
    public function __construct(
        public bool $success,
        public string $standard,
        public string $level,
        public ?string $message = null
    ) {}

    public function toArray(): array
    {
        return [
            'success' => $this->success,
            'standard' => $this->standard,
            'level' => $this->level,
            'message' => $this->message,
        ];
    }
}

/**
 * Comprehensive validation result.
 */
readonly class ComprehensiveValidationResult
{
    public function __construct(
        public ?PdfAValidationResult $pdfA,
        public ?PdfXValidationResult $pdfX,
        public ?PdfUaValidationResult $pdfUa
    ) {}

    public function isPdfACompliant(): bool
    {
        return $this->pdfA?->isCompliant() ?? false;
    }

    public function isPdfXCompliant(): bool
    {
        return $this->pdfX?->isCompliant() ?? false;
    }

    public function isPdfUaAccessible(): bool
    {
        return $this->pdfUa?->isAccessible() ?? false;
    }

    public function toArray(): array
    {
        return [
            'pdf_a' => $this->pdfA?->toArray(),
            'pdf_x' => $this->pdfX?->toArray(),
            'pdf_ua' => $this->pdfUa?->toArray(),
        ];
    }
}
