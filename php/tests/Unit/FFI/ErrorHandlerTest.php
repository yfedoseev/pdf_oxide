<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\FFI;

use PHPUnit\Framework\TestCase;
use PdfOxide\FFI\ErrorHandler;
use PdfOxide\Exceptions\{
    ParseException,
    IoException,
    ValidationException,
    NotFoundException,
    PdfException
};

/**
 * Tests for ErrorHandler
 */
class ErrorHandlerTest extends TestCase
{
    public function testSuccessReturns(): void
    {
        // Should not throw
        ErrorHandler::check(ErrorHandler::SUCCESS);
        $this->assertTrue(ErrorHandler::isSuccess(ErrorHandler::SUCCESS));
    }

    public function testParseErrorThrows(): void
    {
        $this->expectException(ParseException::class);
        ErrorHandler::check(ErrorHandler::PARSE_ERROR, 'test_operation');
    }

    public function testIoErrorThrows(): void
    {
        $this->expectException(IoException::class);
        ErrorHandler::check(ErrorHandler::IO_ERROR);
    }

    public function testInvalidArgThrows(): void
    {
        $this->expectException(ValidationException::class);
        ErrorHandler::check(ErrorHandler::INVALID_ARG);
    }

    public function testNotFoundThrows(): void
    {
        $this->expectException(NotFoundException::class);
        ErrorHandler::check(ErrorHandler::NOT_FOUND);
    }

    public function testErrorMessageGeneration(): void
    {
        $msg = ErrorHandler::getErrorMessage(ErrorHandler::PARSE_ERROR);
        $this->assertStringContainsString('parse', strtolower($msg));

        $msg = ErrorHandler::getErrorMessage(ErrorHandler::IO_ERROR);
        $this->assertStringContainsString('I/O', $msg);
    }

    public function testErrorCodeNames(): void
    {
        $this->assertEquals('SUCCESS', ErrorHandler::getErrorCodeName(ErrorHandler::SUCCESS));
        $this->assertEquals('PARSE_ERROR', ErrorHandler::getErrorCodeName(ErrorHandler::PARSE_ERROR));
        $this->assertEquals('IO_ERROR', ErrorHandler::getErrorCodeName(ErrorHandler::IO_ERROR));
    }

    public function testExceptionCreation(): void
    {
        $exception = ErrorHandler::createException(
            ErrorHandler::PARSE_ERROR,
            'test_op',
            ['detail' => 'test']
        );

        $this->assertInstanceOf(ParseException::class, $exception);
        $this->assertStringContainsString('test_op', $exception->getMessage());
        $this->assertEquals('PARSE_ERROR', $exception->getErrorCode());
        $this->assertArrayHasKey('detail', $exception->getContext());
    }

    public function testUnknownErrorCode(): void
    {
        $exception = ErrorHandler::createException(999);
        $this->assertInstanceOf(PdfException::class, $exception);
        $this->assertStringContainsString('Unknown', $exception->getMessage());
    }
}
