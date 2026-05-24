<?php

declare(strict_types=1);

namespace PdfOxide\FFI;

use PdfOxide\Exceptions\{
    PdfException,
    ParseException,
    IoException,
    EncryptionException,
    ValidationException,
    ComplianceException,
    NotFoundException,
    SignatureException,
    RedactionException,
    AccessibilityException,
    OptimizationException
};

/**
 * Handles error code mapping and exception throwing.
 *
 * Maps Rust error codes to PHP exceptions.
 */
class ErrorHandler
{
    // Error code constants (must match pdf_oxide.h)
    public const SUCCESS = 0;
    public const INVALID_ARG = 1;
    public const IO_ERROR = 2;
    public const PARSE_ERROR = 3;
    public const NOT_FOUND = 4;
    public const PERMISSION_DENIED = 5;
    public const UNSUPPORTED = 6;
    public const INTERNAL = 7;
    public const SIGNATURE_ERROR = 8;
    public const REDACTION_ERROR = 9;
    public const COMPLIANCE_ERROR = 10;
    public const ACCESSIBILITY_ERROR = 11;
    public const OPTIMIZATION_ERROR = 12;

    /**
     * Check error code and throw appropriate exception if error occurred.
     *
     * @param int $errorCode The error code from FFI call
     * @param string $operation The operation being performed
     * @param array $context Additional context information
     * @throws PdfException on error
     */
    public static function check(int $errorCode, string $operation = '', array $context = []): void
    {
        if ($errorCode === self::SUCCESS) {
            return;
        }

        $exception = self::createException($errorCode, $operation, $context);
        throw $exception;
    }

    /**
     * Create an exception from an error code.
     *
     * @param int $errorCode The error code
     * @param string $operation The operation being performed
     * @param array $context Additional context
     */
    public static function createException(int $errorCode, string $operation = '', array $context = []): PdfException
    {
        $message = self::getErrorMessage($errorCode);
        if ($operation) {
            $message .= " (during {$operation})";
        }

        return match ($errorCode) {
            self::PARSE_ERROR => new ParseException($message, $context),
            self::IO_ERROR => new IoException($message, $context),
            self::INVALID_ARG => new ValidationException($message, $context),
            self::NOT_FOUND => new NotFoundException($message, $context),
            self::PERMISSION_DENIED => new EncryptionException($message, $context),
            self::UNSUPPORTED => new ValidationException($message, $context),
            self::INTERNAL => new PdfException($message, 'INTERNAL_ERROR', $context),
            self::SIGNATURE_ERROR => new SignatureException($message, $context),
            self::REDACTION_ERROR => new RedactionException($message, $context),
            self::COMPLIANCE_ERROR => new ComplianceException($message, $context),
            self::ACCESSIBILITY_ERROR => new AccessibilityException($message, $context),
            self::OPTIMIZATION_ERROR => new OptimizationException($message, $context),
            default => new PdfException(
                "Unknown error: {$errorCode} {$message}",
                'UNKNOWN_ERROR',
                $context
            ),
        };
    }

    /**
     * Get human-readable error message for an error code.
     */
    public static function getErrorMessage(int $errorCode): string
    {
        return match ($errorCode) {
            self::SUCCESS => 'Operation completed successfully',
            self::INVALID_ARG => 'Invalid argument provided',
            self::IO_ERROR => 'I/O error occurred',
            self::PARSE_ERROR => 'Failed to parse PDF',
            self::NOT_FOUND => 'Resource not found',
            self::PERMISSION_DENIED => 'Permission denied',
            self::UNSUPPORTED => 'Operation not supported',
            self::INTERNAL => 'Internal error occurred',
            self::SIGNATURE_ERROR => 'Digital signature operation failed',
            self::REDACTION_ERROR => 'Redaction operation failed',
            self::COMPLIANCE_ERROR => 'Compliance operation failed',
            self::ACCESSIBILITY_ERROR => 'Accessibility operation failed',
            self::OPTIMIZATION_ERROR => 'Optimization operation failed',
            default => "Unknown error code: {$errorCode}",
        };
    }

}
