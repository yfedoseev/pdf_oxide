<?php

declare(strict_types=1);

namespace PdfOxide\FFI;

use PdfOxide\Exceptions\{
    PdfException,
    ParseException,
    IoException,
    ValidationException
};

/**
 * Handles error code mapping and exception throwing.
 *
 * Maps Rust error codes to PHP exceptions.
 */
class ErrorHandler
{
    // Error code constants — MUST mirror src/ffi.rs:98 (the cdylib's
    // canonical error encoding). Previously these were
    // alphabetical-natural and silently mismapped: e.g. cdylib
    // returned 4 (ERR_EXTRACTION) but PHP threw NotFoundException.
    // C# / Ruby / Go all follow the Rust ordering; this brings PHP
    // into line.
    public const SUCCESS = 0;
    public const INVALID_ARG = 1;
    public const IO_ERROR = 2;
    public const PARSE_ERROR = 3;
    public const EXTRACTION_ERROR = 4;
    public const INTERNAL = 5;
    public const INVALID_PAGE = 6;
    public const SEARCH_ERROR = 7;
    public const UNSUPPORTED = 8;

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
            self::INVALID_ARG => new ValidationException($message, $context),
            self::IO_ERROR => new IoException($message, $context),
            self::PARSE_ERROR => new ParseException($message, $context),
            // ERR_EXTRACTION (4) — layout-analysis / text-extraction
            // failure. Per `feedback_extraction_graceful_fallback`,
            // surfaces as a typed PdfException, not as a Validation
            // or NotFound miscategorisation.
            self::EXTRACTION_ERROR => new PdfException($message, 'EXTRACTION_ERROR', $context),
            self::INTERNAL => new PdfException($message, 'INTERNAL_ERROR', $context),
            self::INVALID_PAGE => new ValidationException($message, $context),
            self::SEARCH_ERROR => new PdfException($message, 'SEARCH_ERROR', $context),
            self::UNSUPPORTED => new ValidationException($message, $context),
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
            self::EXTRACTION_ERROR => 'Text or layout extraction failed',
            self::INTERNAL => 'Internal error occurred',
            self::INVALID_PAGE => 'Invalid page index',
            self::SEARCH_ERROR => 'Search operation failed',
            self::UNSUPPORTED => 'Operation not supported',
            default => "Unknown error code: {$errorCode}",
        };
    }

}
