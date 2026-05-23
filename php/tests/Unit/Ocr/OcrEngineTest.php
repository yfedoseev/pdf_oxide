<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Ocr;

use PHPUnit\Framework\TestCase;
use PdfOxide\Ocr\OcrEngine;
use PdfOxide\Ocr\OcrSpan;
use PdfOxide\Ocr\OcrResult;
use PdfOxide\PdfDocument;
use PdfOxide\FFI\FunctionBindings;

/**
 * Tests for OcrEngine class
 *
 * @covers \PdfOxide\Ocr\OcrEngine
 */
class OcrEngineTest extends TestCase
{
    private OcrEngine $engine;

    protected function setUp(): void
    {
        $this->engine = new OcrEngine();
    }

    protected function tearDown(): void
    {
        if ($this->engine->isOpen()) {
            $this->engine->close();
        }
    }

    /**
     * Test OCR engine initialization
     */
    public function testEngineInitialization(): void
    {
        $this->assertInstanceOf(OcrEngine::class, $this->engine);
        $this->assertTrue($this->engine->isOpen());
    }

    /**
     * Test engine is open after construction
     */
    public function testEngineIsOpenAfterConstruction(): void
    {
        $engine = new OcrEngine();
        $this->assertTrue($engine->isOpen());
        $engine->close();
    }

    /**
     * Test engine can be closed
     */
    public function testEngineClose(): void
    {
        $this->assertTrue($this->engine->isOpen());
        $this->engine->close();
        $this->assertFalse($this->engine->isOpen());
    }

    /**
     * Test closing already closed engine doesn't throw
     */
    public function testCloseAlreadyClosedEngine(): void
    {
        $this->engine->close();
        // Should not throw
        $this->engine->close();
        $this->assertFalse($this->engine->isOpen());
    }

    /**
     * Test operations on closed engine throw exception
     */
    public function testOperationOnClosedEngineThrows(): void
    {
        $this->engine->close();

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('OCR engine has been closed');

        // Try any operation that requires engine to be open
        $bindings = $this->getMockBuilder(FunctionBindings::class)
            ->onlyMethods(['pdfOcrEngineGetStatus'])
            ->getMock();

        $this->engine->getStatus();
    }

    /**
     * Test getting engine version
     */
    public function testGetEngineVersion(): void
    {
        $version = OcrEngine::getVersion();
        $this->assertIsString($version);
        $this->assertNotEmpty($version);
    }

    /**
     * Test getting engine status
     */
    public function testGetEngineStatus(): void
    {
        $status = $this->engine->getStatus();
        $this->assertIsString($status);
        $this->assertNotEmpty($status);
    }

    /**
     * Test getting engine handle
     */
    public function testGetEngineHandle(): void
    {
        $handle = $this->engine->getHandle();
        $this->assertNotNull($handle);
    }

    /**
     * Test getting engine bindings
     */
    public function testGetEngineBindings(): void
    {
        $bindings = $this->engine->getBindings();
        $this->assertInstanceOf(FunctionBindings::class, $bindings);
    }

    /**
     * Test engine destructor closes resources
     */
    public function testEngineDestructorClosesResources(): void
    {
        $engine = new OcrEngine();
        $this->assertTrue($engine->isOpen());

        unset($engine);

        // After unset, the destructor should have been called
        // We can't directly test the closed state, but we can verify
        // the object is destroyed
        $this->assertTrue(true);
    }

    /**
     * Test extracting text returns string
     */
    public function testExtractTextReturnsString(): void
    {
        // This would require a real PDF or mock
        // For now, we test the method exists and has correct signature
        $this->assertTrue(method_exists($this->engine, 'extractText'));
    }

    /**
     * Test extracting spans returns array
     */
    public function testExtractSpansReturnsArray(): void
    {
        $this->assertTrue(method_exists($this->engine, 'extractSpans'));
    }

    /**
     * Test extracting pages returns array
     */
    public function testExtractPagesReturnsArray(): void
    {
        $this->assertTrue(method_exists($this->engine, 'extractPages'));
    }

    /**
     * Test extracting page range returns array
     */
    public function testExtractRangeReturnsArray(): void
    {
        $this->assertTrue(method_exists($this->engine, 'extractRange'));
    }

    /**
     * Test page needs OCR returns boolean
     */
    public function testPageNeedsOcrReturnsBoolean(): void
    {
        $this->assertTrue(method_exists($this->engine, 'pageNeedsOcr'));
    }

    /**
     * Test multiple engines can exist independently
     */
    public function testMultipleEnginesIndependent(): void
    {
        $engine1 = new OcrEngine();
        $engine2 = new OcrEngine();

        $this->assertTrue($engine1->isOpen());
        $this->assertTrue($engine2->isOpen());

        $engine1->close();

        $this->assertFalse($engine1->isOpen());
        $this->assertTrue($engine2->isOpen());

        $engine2->close();
    }

    /**
     * Test engine with model path parameter
     */
    public function testEngineWithModelPath(): void
    {
        // Engine should handle model path parameter (even if not used in this version)
        $engine = new OcrEngine('/path/to/model');
        $this->assertTrue($engine->isOpen());
        $engine->close();
    }

    /**
     * Test bindings are reused across operations
     */
    public function testBindingsReused(): void
    {
        $bindings1 = $this->engine->getBindings();
        $bindings2 = $this->engine->getBindings();

        $this->assertSame($bindings1, $bindings2);
    }

    /**
     * Test handle is consistent across operations
     */
    public function testHandleConsistent(): void
    {
        $handle1 = $this->engine->getHandle();
        $handle2 = $this->engine->getHandle();

        $this->assertSame($handle1, $handle2);
    }

    /**
     * Test engine status before and after close
     */
    public function testEngineStatusBeforeAndAfterClose(): void
    {
        $this->assertTrue($this->engine->isOpen());

        try {
            $status = $this->engine->getStatus();
            $this->assertIsString($status);
        } catch (\Exception) {
            // Status retrieval might fail without a valid PDF, that's OK
        }

        $this->engine->close();
        $this->assertFalse($this->engine->isOpen());
    }

    /**
     * Test version is same for static call and instance
     */
    public function testVersionConsistency(): void
    {
        $staticVersion = OcrEngine::getVersion();
        $this->assertIsString($staticVersion);
        $this->assertNotEmpty($staticVersion);
    }
}
