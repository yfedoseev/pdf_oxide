<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Ocr;

use PHPUnit\Framework\TestCase;
use PdfOxide\Ocr\OcrSpan;
use PdfOxide\FFI\FunctionBindings;
use FFI\CData;

/**
 * Tests for OcrSpan class
 *
 * @covers \PdfOxide\Ocr\OcrSpan
 */
class OcrSpanTest extends TestCase
{
    private FunctionBindings $bindingsMock;
    private CData $spanHandleMock;

    protected function setUp(): void
    {
        $this->bindingsMock = $this->getMockBuilder(FunctionBindings::class)
            ->disableOriginalConstructor()
            ->onlyMethods([
                'pdfOcrSpanGetText',
                'pdfOcrSpanGetBbox',
                'pdfOcrSpanGetConfidence',
                'pdfOcrSpanGetCharConfidence',
                'pdfOcrSpanFree'
            ])
            ->getMock();

        $this->spanHandleMock = $this->getMockBuilder(CData::class)
            ->getMock();
    }

    /**
     * Create an OcrSpan with mock data
     */
    private function createOcrSpan(
        string $text = 'Hello',
        array $bbox = ['x' => 10, 'y' => 20, 'width' => 50, 'height' => 15],
        float $confidence = 0.95,
        ?array $charConfidences = null
    ): OcrSpan {
        $this->bindingsMock
            ->method('pdfOcrSpanGetText')
            ->willReturn($text);

        $this->bindingsMock
            ->method('pdfOcrSpanGetBbox')
            ->willReturn($bbox);

        $this->bindingsMock
            ->method('pdfOcrSpanGetConfidence')
            ->willReturn($confidence);

        if ($charConfidences !== null) {
            $this->bindingsMock
                ->method('pdfOcrSpanGetCharConfidence')
                ->willReturnOnConsecutiveCalls(...$charConfidences);
        } else {
            $this->bindingsMock
                ->method('pdfOcrSpanGetCharConfidence')
                ->willThrowException(new \Exception('No char confidences'));
        }

        $this->bindingsMock
            ->method('pdfOcrSpanFree')
            ->with($this->spanHandleMock);

        return new OcrSpan($this->spanHandleMock, $this->bindingsMock);
    }

    /**
     * Test OcrSpan creation
     */
    public function testOcrSpanCreation(): void
    {
        $span = $this->createOcrSpan();
        $this->assertInstanceOf(OcrSpan::class, $span);
    }

    /**
     * Test getText method
     */
    public function testGetText(): void
    {
        $text = 'Hello World';
        $span = $this->createOcrSpan($text);

        $this->assertEquals($text, $span->getText());
    }

    /**
     * Test getText with special characters
     */
    public function testGetTextWithSpecialCharacters(): void
    {
        $text = 'Héllo Wørld 123!@#';
        $span = $this->createOcrSpan($text);

        $this->assertEquals($text, $span->getText());
    }

    /**
     * Test getText with empty string
     */
    public function testGetTextEmpty(): void
    {
        $span = $this->createOcrSpan('');
        $this->assertEquals('', $span->getText());
    }

    /**
     * Test getBbox method
     */
    public function testGetBbox(): void
    {
        $bbox = ['x' => 100, 'y' => 200, 'width' => 300, 'height' => 50];
        $span = $this->createOcrSpan('Text', $bbox);

        $result = $span->getBbox();
        $this->assertEquals($bbox, $result);
    }

    /**
     * Test getBbox has correct keys
     */
    public function testGetBboxKeys(): void
    {
        $bbox = ['x' => 10, 'y' => 20, 'width' => 100, 'height' => 30];
        $span = $this->createOcrSpan('Text', $bbox);

        $result = $span->getBbox();
        $this->assertArrayHasKey('x', $result);
        $this->assertArrayHasKey('y', $result);
        $this->assertArrayHasKey('width', $result);
        $this->assertArrayHasKey('height', $result);
    }

    /**
     * Test getBbox with float coordinates
     */
    public function testGetBboxWithFloats(): void
    {
        $bbox = ['x' => 10.5, 'y' => 20.7, 'width' => 50.3, 'height' => 15.9];
        $span = $this->createOcrSpan('Text', $bbox);

        $result = $span->getBbox();
        $this->assertEquals(10.5, $result['x']);
        $this->assertEquals(20.7, $result['y']);
    }

    /**
     * Test getConfidence method
     */
    public function testGetConfidence(): void
    {
        $confidence = 0.85;
        $span = $this->createOcrSpan('Text', ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0], $confidence);

        $this->assertEquals($confidence, $span->getConfidence());
    }

    /**
     * Test getConfidence with perfect confidence
     */
    public function testGetConfidencePerfect(): void
    {
        $span = $this->createOcrSpan('Text', ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0], 1.0);
        $this->assertEquals(1.0, $span->getConfidence());
    }

    /**
     * Test getConfidence with zero confidence
     */
    public function testGetConfidenceZero(): void
    {
        $span = $this->createOcrSpan('Text', ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0], 0.0);
        $this->assertEquals(0.0, $span->getConfidence());
    }

    /**
     * Test getCharConfidences when available
     */
    public function testGetCharConfidencesAvailable(): void
    {
        $charConfidences = [0.95, 0.92, 0.88, 0.90, 0.87];
        $span = $this->createOcrSpan('Hello', ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0], 0.90, $charConfidences);

        $result = $span->getCharConfidences();
        $this->assertEquals($charConfidences, $result);
    }

    /**
     * Test getCharConfidences when not available
     */
    public function testGetCharConfidencesNotAvailable(): void
    {
        $span = $this->createOcrSpan('Text');
        $this->assertNull($span->getCharConfidences());
    }

    /**
     * Test getCharConfidence at specific index
     */
    public function testGetCharConfidenceAtIndex(): void
    {
        $charConfidences = [0.95, 0.92, 0.88, 0.90, 0.87];
        $span = $this->createOcrSpan('Hello', ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0], 0.90, $charConfidences);

        $this->assertEquals(0.95, $span->getCharConfidence(0));
        $this->assertEquals(0.92, $span->getCharConfidence(1));
        $this->assertEquals(0.87, $span->getCharConfidence(4));
    }

    /**
     * Test getCharConfidence throws when confidences not available
     */
    public function testGetCharConfidenceThrowsWhenNotAvailable(): void
    {
        $span = $this->createOcrSpan('Text');

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('Character confidences not available');

        $span->getCharConfidence(0);
    }

    /**
     * Test getCharConfidence with negative index throws
     */
    public function testGetCharConfidenceNegativeIndexThrows(): void
    {
        $charConfidences = [0.95, 0.92];
        $span = $this->createOcrSpan('Hi', ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0], 0.90, $charConfidences);

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('out of bounds');

        $span->getCharConfidence(-1);
    }

    /**
     * Test getCharConfidence with index out of bounds throws
     */
    public function testGetCharConfidenceOutOfBoundsThrows(): void
    {
        $charConfidences = [0.95, 0.92];
        $span = $this->createOcrSpan('Hi', ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0], 0.90, $charConfidences);

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('out of bounds');

        $span->getCharConfidence(5);
    }

    /**
     * Test toArray method
     */
    public function testToArray(): void
    {
        $text = 'Test';
        $bbox = ['x' => 10, 'y' => 20, 'width' => 50, 'height' => 15];
        $confidence = 0.92;

        $span = $this->createOcrSpan($text, $bbox, $confidence);
        $result = $span->toArray();

        $this->assertIsArray($result);
        $this->assertEquals($text, $result['text']);
        $this->assertEquals($bbox, $result['bbox']);
        $this->assertEquals($confidence, $result['confidence']);
        $this->assertNull($result['charConfidences']);
    }

    /**
     * Test toArray with character confidences
     */
    public function testToArrayWithCharConfidences(): void
    {
        $charConfidences = [0.95, 0.92, 0.88, 0.90];
        $span = $this->createOcrSpan('Test', ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0], 0.91, $charConfidences);

        $result = $span->toArray();
        $this->assertEquals($charConfidences, $result['charConfidences']);
    }

    /**
     * Test __toString method
     */
    public function testToString(): void
    {
        $text = 'Hello';
        $span = $this->createOcrSpan($text);

        $this->assertEquals($text, (string)$span);
    }

    /**
     * Test span is immutable after construction
     */
    public function testSpanImmutable(): void
    {
        $span = $this->createOcrSpan('Text');

        // Try to set properties (should fail)
        $this->expectException(\Exception::class);

        $reflection = new \ReflectionClass($span);
        $property = $reflection->getProperty('text');
        $property->setAccessible(true);
        $property->setValue($span, 'Modified');
    }

    /**
     * Test FFI handle is freed after construction
     */
    public function testHandleFreedAfterConstruction(): void
    {
        $this->bindingsMock
            ->method('pdfOcrSpanGetText')
            ->willReturn('Text');

        $this->bindingsMock
            ->method('pdfOcrSpanGetBbox')
            ->willReturn(['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0]);

        $this->bindingsMock
            ->method('pdfOcrSpanGetConfidence')
            ->willReturn(0.9);

        $this->bindingsMock
            ->method('pdfOcrSpanGetCharConfidence')
            ->willThrowException(new \Exception('No char confidences'));

        $this->bindingsMock
            ->expects($this->once())
            ->method('pdfOcrSpanFree')
            ->with($this->spanHandleMock);

        new OcrSpan($this->spanHandleMock, $this->bindingsMock);
    }

    /**
     * Test span construction with exception handling
     */
    public function testConstructionWithExceptionHandling(): void
    {
        $bindingsFail = $this->getMockBuilder(FunctionBindings::class)
            ->disableOriginalConstructor()
            ->onlyMethods(['pdfOcrSpanGetText'])
            ->getMock();

        $bindingsFail
            ->method('pdfOcrSpanGetText')
            ->willThrowException(new \Exception('FFI error'));

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('Failed to create OcrSpan');

        new OcrSpan($this->spanHandleMock, $bindingsFail);
    }

    /**
     * Test multiple spans can coexist
     */
    public function testMultipleSpansCoexist(): void
    {
        $span1 = $this->createOcrSpan('Hello', ['x' => 0, 'y' => 0, 'width' => 50, 'height' => 15]);
        $span2 = $this->createOcrSpan('World', ['x' => 60, 'y' => 0, 'width' => 50, 'height' => 15]);

        $this->assertEquals('Hello', $span1->getText());
        $this->assertEquals('World', $span2->getText());
        $this->assertNotEquals($span1->getText(), $span2->getText());
    }
}
