<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Unit;

use PdfOxide\Exceptions\InternalError;
use PdfOxide\Exceptions\IoException;
use PdfOxide\Exceptions\ParseException;
use PdfOxide\Exceptions\PdfException;
use PdfOxide\Exceptions\SearchException;
use PdfOxide\Exceptions\UnsupportedException;
use PdfOxide\Exceptions\ValidationException;
use PdfOxide\FFI\ErrorHandler;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

/**
 * Locks in the cdylib int32 error code → PHP exception mapping
 * against src/ffi.rs:98-106. Mirrors the C# binding's
 * ExceptionMapperTests and the Ruby spec
 * ffi_signature_regression_spec.rb#raise_for_code section.
 *
 * Pre-v0.3.55 PHP had alphabetical-natural codes
 * (PDFA codes 4..12 all wrong); a previously-mapped 4=NOT_FOUND
 * fired NotFoundException when the cdylib actually returned
 * ERR_EXTRACTION. This test fails fast if the mapping ever drifts
 * again.
 */
final class ErrorHandlerMappingTest extends TestCase
{
    public function testConstantsMirrorRustABI(): void
    {
        // src/ffi.rs:98 — these MUST stay identical across every
        // pdf_oxide binding (PHP, Ruby, C#, Go).
        $this->assertSame(0, ErrorHandler::SUCCESS);
        $this->assertSame(1, ErrorHandler::INVALID_ARG);
        $this->assertSame(2, ErrorHandler::IO_ERROR);
        $this->assertSame(3, ErrorHandler::PARSE_ERROR);
        $this->assertSame(4, ErrorHandler::EXTRACTION_ERROR);
        $this->assertSame(5, ErrorHandler::INTERNAL);
        $this->assertSame(6, ErrorHandler::INVALID_PAGE);
        $this->assertSame(7, ErrorHandler::SEARCH_ERROR);
        $this->assertSame(8, ErrorHandler::UNSUPPORTED);
    }

    public function testCheckIsNoOpForSuccess(): void
    {
        $this->expectNotToPerformAssertions();
        ErrorHandler::check(0, 'irrelevant');
    }

    #[DataProvider('codeToExceptionMapProvider')]
    public function testCodeMapsToTypedException(int $code, string $expectedClass): void
    {
        $ex = ErrorHandler::createException($code, "test_op_{$code}");
        $this->assertInstanceOf($expectedClass, $ex);
        $this->assertInstanceOf(PdfException::class, $ex);
        $this->assertStringContainsString("test_op_{$code}", $ex->getMessage());
    }

    /**
     * @return iterable<array{int, class-string<PdfException>}>
     */
    public static function codeToExceptionMapProvider(): iterable
    {
        // Same one-to-one mapping as csharp/PdfOxide/Internal/ExceptionMapper.cs.
        yield 'ERR_INVALID_ARG → ValidationException' => [1, ValidationException::class];
        yield 'ERR_IO → IoException' => [2, IoException::class];
        yield 'ERR_PARSE → ParseException' => [3, ParseException::class];
        yield 'ERR_EXTRACTION → ParseException' => [4, ParseException::class];
        yield 'ERR_INTERNAL → InternalError' => [5, InternalError::class];
        yield 'ERR_INVALID_PAGE → ValidationException' => [6, ValidationException::class];
        yield 'ERR_SEARCH → SearchException' => [7, SearchException::class];
        yield '_ERR_UNSUPPORTED → UnsupportedException' => [8, UnsupportedException::class];
    }

    public function testUnknownCodeFallsBackToBasePdfException(): void
    {
        $ex = ErrorHandler::createException(99, 'weird_op');
        $this->assertInstanceOf(PdfException::class, $ex);
        // The fallback uses the generic base class, not a typed subclass.
        $this->assertSame(PdfException::class, $ex::class);
        $this->assertStringContainsString('99', $ex->getMessage());
    }

    public function testGetErrorMessageMirrorsCsharpMessages(): void
    {
        // The wording of each message matches
        // csharp/PdfOxide/Internal/ExceptionMapper.cs so log lines
        // are recognisable across language boundaries.
        $this->assertStringContainsString('Invalid argument', ErrorHandler::getErrorMessage(1));
        $this->assertStringContainsString('I/O error', ErrorHandler::getErrorMessage(2));
        $this->assertStringContainsString('Parse error', ErrorHandler::getErrorMessage(3));
        $this->assertStringContainsString('Extraction failed', ErrorHandler::getErrorMessage(4));
        $this->assertStringContainsString('Internal error', ErrorHandler::getErrorMessage(5));
        $this->assertStringContainsString('Invalid page', ErrorHandler::getErrorMessage(6));
        $this->assertStringContainsString('Search error', ErrorHandler::getErrorMessage(7));
        $this->assertStringContainsString('Unsupported', ErrorHandler::getErrorMessage(8));
    }
}
