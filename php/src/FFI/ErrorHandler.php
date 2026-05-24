<?php

declare(strict_types=1);

namespace PdfOxide\FFI;

use PdfOxide\Exceptions\{
    InternalError,
    IoException,
    ParseException,
    PdfException,
    SearchException,
    UnsupportedException,
    ValidationException
};

/**
 * Maps cdylib int32 error codes to PHP exceptions.
 *
 * <p>The mapping table mirrors {@code src/ffi.rs:98-106} exactly —
 * the same 9-code surface the C# binding ({@code
 * PdfOxide.Internal.ExceptionMapper}), Ruby binding
 * ({@code PdfOxide::PdfDocument#raise_for_code}), and Go binding
 * all use. PHP follows the C# pattern: one exception class per
 * code, no aliasing.
 *
 * <p>Pre-v0.3.55 PHP had alphabetical-natural codes
 * ({@code NOT_FOUND=4, PERMISSION_DENIED=5, …}) that mismapped
 * against the cdylib's wire format — cdylib returned 4
 * (ERR_EXTRACTION), PHP threw NotFoundException; returned 5
 * (ERR_INTERNAL), PHP threw EncryptionException; returned 8
 * (ERR_UNSUPPORTED), PHP threw SignatureException. C# fixed the
 * same bug class in an earlier release (see comment block in
 * {@code csharp/PdfOxide/Internal/ExceptionMapper.cs}); this
 * brings PHP into line.
 */
class ErrorHandler
{
    // Error codes — MUST stay byte-for-byte identical to
    // src/ffi.rs:98-106. CI cross-binding parity tests catch drift.
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
     * Throw the appropriate {@see PdfException} subclass when
     * {@code $errorCode} is non-zero.
     *
     * @param int $errorCode Code returned by the cdylib (`*err`).
     * @param string $operation Native function name, included in
     *                          the message for traceability.
     * @param array $context Optional structured context.
     * @throws PdfException
     */
    public static function check(int $errorCode, string $operation = '', array $context = []): void
    {
        if ($errorCode === self::SUCCESS) {
            return;
        }

        throw self::createException($errorCode, $operation, $context);
    }

    /**
     * Build (but don't throw) the exception for an error code. Used
     * by call sites that want to inspect the typed exception before
     * deciding to raise or degrade (e.g. signature-aware paths).
     *
     * @param int $errorCode
     * @param string $operation
     * @param array $context
     */
    public static function createException(int $errorCode, string $operation = '', array $context = []): PdfException
    {
        $message = self::getErrorMessage($errorCode);
        if ($operation !== '') {
            $message .= " (during {$operation})";
        }

        // 1-to-1 mapping matching csharp/PdfOxide/Internal/ExceptionMapper.cs.
        return match ($errorCode) {
            self::INVALID_ARG       => new ValidationException($message, $context),
            self::IO_ERROR          => new IoException($message, $context),
            self::PARSE_ERROR       => new ParseException($message, $context),
            self::EXTRACTION_ERROR  => new ParseException($message, $context),
            self::INTERNAL          => new InternalError($message, $context),
            self::INVALID_PAGE      => new ValidationException($message, $context),
            self::SEARCH_ERROR      => new SearchException($message, $context),
            self::UNSUPPORTED       => new UnsupportedException($message, $context),
            default                 => new PdfException(
                "Unknown error code: {$errorCode} ({$message})",
                'UNKNOWN_ERROR',
                $context
            ),
        };
    }

    /**
     * Human-readable description for a cdylib error code. Messages
     * mirror the C# binding so log lines are recognisable across
     * languages.
     */
    public static function getErrorMessage(int $errorCode): string
    {
        return match ($errorCode) {
            self::SUCCESS           => 'Operation completed successfully',
            self::INVALID_ARG       => 'Invalid argument: one or more arguments were invalid',
            self::IO_ERROR          => 'I/O error: file not found, permission denied, or read/write failed',
            self::PARSE_ERROR       => 'Parse error: invalid PDF structure or content stream',
            self::EXTRACTION_ERROR  => 'Extraction failed: page content could not be extracted',
            self::INTERNAL          => 'Internal error: unexpected failure in the core library',
            self::INVALID_PAGE      => 'Invalid page index: page out of range for this document',
            self::SEARCH_ERROR      => 'Search error: search operation failed',
            self::UNSUPPORTED       => 'Unsupported feature: this build was compiled without support for the requested operation',
            default                 => "Unknown error code: {$errorCode}",
        };
    }
}
