<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Ocr;

use PHPUnit\Framework\TestCase;
use PdfOxide\Ocr\OcrResult;
use PdfOxide\Ocr\OcrSpan;
use FFI\CData;

/**
 * Tests for OcrResult class
 *
 * @covers \PdfOxide\Ocr\OcrResult
 */
class OcrResultTest extends TestCase
{
    /**
     * Create a mock OcrSpan with specified data
     */
    private function createMockSpan(
        string $text = 'Word',
        float $confidence = 0.9
    ): OcrSpan {
        $span = $this->getMockBuilder(OcrSpan::class)
            ->disableOriginalConstructor()
            ->getMock();

        $span->method('getText')->willReturn($text);
        $span->method('getConfidence')->willReturn($confidence);
        $span->method('toArray')->willReturn([
            'text' => $text,
            'bbox' => ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0],
            'confidence' => $confidence,
            'charConfidences' => null,
        ]);

        return $span;
    }

    /**
     * Test OcrResult creation with empty spans
     */
    public function testResultCreationEmpty(): void
    {
        $result = new OcrResult([]);
        $this->assertInstanceOf(OcrResult::class, $result);
        $this->assertEquals(0, $result->getCount());
    }

    /**
     * Test OcrResult creation with spans
     */
    public function testResultCreationWithSpans(): void
    {
        $spans = [
            $this->createMockSpan('Hello'),
            $this->createMockSpan('World'),
        ];

        $result = new OcrResult($spans);
        $this->assertEquals(2, $result->getCount());
    }

    /**
     * Test getCount method
     */
    public function testGetCount(): void
    {
        $spans = [
            $this->createMockSpan('One'),
            $this->createMockSpan('Two'),
            $this->createMockSpan('Three'),
        ];

        $result = new OcrResult($spans);
        $this->assertEquals(3, $result->getCount());
    }

    /**
     * Test getCount on empty result
     */
    public function testGetCountEmpty(): void
    {
        $result = new OcrResult([]);
        $this->assertEquals(0, $result->getCount());
    }

    /**
     * Test getSpan by index
     */
    public function testGetSpan(): void
    {
        $span1 = $this->createMockSpan('First');
        $span2 = $this->createMockSpan('Second');

        $result = new OcrResult([$span1, $span2]);

        $this->assertSame($span1, $result->getSpan(0));
        $this->assertSame($span2, $result->getSpan(1));
    }

    /**
     * Test getSpan with invalid index throws
     */
    public function testGetSpanInvalidIndexThrows(): void
    {
        $span = $this->createMockSpan('Word');
        $result = new OcrResult([$span]);

        $this->expectException(\OutOfRangeException::class);
        $this->expectExceptionMessage('out of bounds');

        $result->getSpan(5);
    }

    /**
     * Test getSpan with negative index throws
     */
    public function testGetSpanNegativeIndexThrows(): void
    {
        $span = $this->createMockSpan('Word');
        $result = new OcrResult([$span]);

        $this->expectException(\OutOfRangeException::class);
        $result->getSpan(-1);
    }

    /**
     * Test getSpans returns all spans
     */
    public function testGetSpans(): void
    {
        $spans = [
            $this->createMockSpan('One'),
            $this->createMockSpan('Two'),
            $this->createMockSpan('Three'),
        ];

        $result = new OcrResult($spans);
        $retrieved = $result->getSpans();

        $this->assertEquals(count($spans), count($retrieved));
        $this->assertSame($spans[0], $retrieved[0]);
        $this->assertSame($spans[1], $retrieved[1]);
        $this->assertSame($spans[2], $retrieved[2]);
    }

    /**
     * Test getText combines all span texts
     */
    public function testGetText(): void
    {
        $spans = [
            $this->createMockSpan('Hello'),
            $this->createMockSpan('World'),
            $this->createMockSpan('Test'),
        ];

        $result = new OcrResult($spans);
        $this->assertEquals('Hello World Test', $result->getText());
    }

    /**
     * Test getText with empty result
     */
    public function testGetTextEmpty(): void
    {
        $result = new OcrResult([]);
        $this->assertEquals('', $result->getText());
    }

    /**
     * Test getText with single span
     */
    public function testGetTextSingleSpan(): void
    {
        $result = new OcrResult([$this->createMockSpan('OnlyWord')]);
        $this->assertEquals('OnlyWord', $result->getText());
    }

    /**
     * Test getAverageConfidence
     */
    public function testGetAverageConfidence(): void
    {
        $spans = [
            $this->createMockSpan('One', 0.9),
            $this->createMockSpan('Two', 0.8),
            $this->createMockSpan('Three', 0.7),
        ];

        $result = new OcrResult($spans);
        $average = $result->getAverageConfidence();

        // (0.9 + 0.8 + 0.7) / 3 = 2.4 / 3 = 0.8
        $this->assertAlmostEquals(0.8, $average, 5);
    }

    /**
     * Test getAverageConfidence with empty result
     */
    public function testGetAverageConfidenceEmpty(): void
    {
        $result = new OcrResult([]);
        $this->assertEquals(0.0, $result->getAverageConfidence());
    }

    /**
     * Test getAverageConfidence with single span
     */
    public function testGetAverageConfidenceSingleSpan(): void
    {
        $result = new OcrResult([$this->createMockSpan('Word', 0.75)]);
        $this->assertEquals(0.75, $result->getAverageConfidence());
    }

    /**
     * Test getMinConfidence
     */
    public function testGetMinConfidence(): void
    {
        $spans = [
            $this->createMockSpan('One', 0.95),
            $this->createMockSpan('Two', 0.65),
            $this->createMockSpan('Three', 0.85),
        ];

        $result = new OcrResult($spans);
        $this->assertEquals(0.65, $result->getMinConfidence());
    }

    /**
     * Test getMinConfidence with empty result
     */
    public function testGetMinConfidenceEmpty(): void
    {
        $result = new OcrResult([]);
        $this->assertNull($result->getMinConfidence());
    }

    /**
     * Test getMinConfidence with single span
     */
    public function testGetMinConfidenceSingleSpan(): void
    {
        $result = new OcrResult([$this->createMockSpan('Word', 0.42)]);
        $this->assertEquals(0.42, $result->getMinConfidence());
    }

    /**
     * Test getMaxConfidence
     */
    public function testGetMaxConfidence(): void
    {
        $spans = [
            $this->createMockSpan('One', 0.75),
            $this->createMockSpan('Two', 0.92),
            $this->createMockSpan('Three', 0.68),
        ];

        $result = new OcrResult($spans);
        $this->assertEquals(0.92, $result->getMaxConfidence());
    }

    /**
     * Test getMaxConfidence with empty result
     */
    public function testGetMaxConfidenceEmpty(): void
    {
        $result = new OcrResult([]);
        $this->assertNull($result->getMaxConfidence());
    }

    /**
     * Test getMaxConfidence with single span
     */
    public function testGetMaxConfidenceSingleSpan(): void
    {
        $result = new OcrResult([$this->createMockSpan('Word', 0.88)]);
        $this->assertEquals(0.88, $result->getMaxConfidence());
    }

    /**
     * Test filterByConfidence
     */
    public function testFilterByConfidence(): void
    {
        $span1 = $this->createMockSpan('High', 0.95);
        $span2 = $this->createMockSpan('Low', 0.60);
        $span3 = $this->createMockSpan('Medium', 0.80);

        $result = new OcrResult([$span1, $span2, $span3]);
        $filtered = $result->filterByConfidence(0.75);

        // Should include span1 (0.95) and span3 (0.80), exclude span2 (0.60)
        $this->assertCount(2, $filtered);
    }

    /**
     * Test filterByConfidence with threshold 0
     */
    public function testFilterByConfidenceMinThreshold(): void
    {
        $spans = [
            $this->createMockSpan('One', 0.1),
            $this->createMockSpan('Two', 0.5),
            $this->createMockSpan('Three', 0.9),
        ];

        $result = new OcrResult($spans);
        $filtered = $result->filterByConfidence(0.0);

        $this->assertCount(3, $filtered);
    }

    /**
     * Test filterByConfidence with threshold 1
     */
    public function testFilterByConfidenceMaxThreshold(): void
    {
        $spans = [
            $this->createMockSpan('One', 0.9),
            $this->createMockSpan('Two', 1.0),
            $this->createMockSpan('Three', 0.95),
        ];

        $result = new OcrResult($spans);
        $filtered = $result->filterByConfidence(1.0);

        $this->assertCount(1, $filtered);
    }

    /**
     * Test filterByConfidence with invalid threshold throws
     */
    public function testFilterByConfidenceInvalidThresholdLow(): void
    {
        $result = new OcrResult([$this->createMockSpan('Word')]);

        $this->expectException(\InvalidArgumentException::class);
        $result->filterByConfidence(-0.1);
    }

    /**
     * Test filterByConfidence with invalid threshold throws
     */
    public function testFilterByConfidenceInvalidThresholdHigh(): void
    {
        $result = new OcrResult([$this->createMockSpan('Word')]);

        $this->expectException(\InvalidArgumentException::class);
        $result->filterByConfidence(1.1);
    }

    /**
     * Test toArray method
     */
    public function testToArray(): void
    {
        $spans = [
            $this->createMockSpan('Hello', 0.9),
            $this->createMockSpan('World', 0.85),
        ];

        $result = new OcrResult($spans);
        $array = $result->toArray();

        $this->assertIsArray($array);
        $this->assertArrayHasKey('count', $array);
        $this->assertArrayHasKey('text', $array);
        $this->assertArrayHasKey('averageConfidence', $array);
        $this->assertArrayHasKey('minConfidence', $array);
        $this->assertArrayHasKey('maxConfidence', $array);
        $this->assertArrayHasKey('spans', $array);

        $this->assertEquals(2, $array['count']);
        $this->assertEquals('Hello World', $array['text']);
    }

    /**
     * Test toArray with empty result
     */
    public function testToArrayEmpty(): void
    {
        $result = new OcrResult([]);
        $array = $result->toArray();

        $this->assertEquals(0, $array['count']);
        $this->assertEquals('', $array['text']);
        $this->assertEquals(0.0, $array['averageConfidence']);
        $this->assertNull($array['minConfidence']);
        $this->assertNull($array['maxConfidence']);
        $this->assertEmpty($array['spans']);
    }

    /**
     * Test Countable interface
     */
    public function testCountableInterface(): void
    {
        $spans = [
            $this->createMockSpan('One'),
            $this->createMockSpan('Two'),
            $this->createMockSpan('Three'),
        ];

        $result = new OcrResult($spans);

        // count() function should work
        $this->assertEquals(3, count($result));
    }

    /**
     * Test iteration over result
     */
    public function testIterationOverResult(): void
    {
        $span1 = $this->createMockSpan('One');
        $span2 = $this->createMockSpan('Two');
        $span3 = $this->createMockSpan('Three');

        $result = new OcrResult([$span1, $span2, $span3]);

        $iterator = $result->getIterator();
        $this->assertInstanceOf(\ArrayIterator::class, $iterator);

        $collected = iterator_to_array($iterator);
        $this->assertCount(3, $collected);
        $this->assertSame($span1, $collected[0]);
        $this->assertSame($span2, $collected[1]);
        $this->assertSame($span3, $collected[2]);
    }

    /**
     * Test result is immutable
     */
    public function testResultImmutable(): void
    {
        $span = $this->createMockSpan('Word');
        $result = new OcrResult([$span]);

        // The internal array should be read-only due to readonly property
        $reflection = new \ReflectionClass($result);
        $property = $reflection->getProperty('spans');

        $this->assertTrue($property->isReadOnly());
    }

    /**
     * Test result with many spans
     */
    public function testResultWithManySpans(): void
    {
        $spans = array_map(
            fn($i) => $this->createMockSpan("Word$i", 0.5 + ($i * 0.01)),
            range(0, 99)
        );

        $result = new OcrResult($spans);

        $this->assertEquals(100, $result->getCount());
        $this->assertGreater($result->getMaxConfidence(), $result->getMinConfidence());
        $this->assertGreater(0, strlen($result->getText()));
    }

    /**
     * Test filterByConfidence with many spans
     */
    public function testFilterByConfidenceWithManySpans(): void
    {
        $spans = array_map(
            fn($i) => $this->createMockSpan("Word$i", ($i / 100)),
            range(0, 99)
        );

        $result = new OcrResult($spans);
        $filtered = $result->filterByConfidence(0.5);

        // Should include only spans with confidence >= 0.5
        $this->assertGreaterThan(0, count($filtered));
        $this->assertLessThan(100, count($filtered));
    }

    /**
     * Test confidence statistics are accurate
     */
    public function testConfidenceStatisticsAccuracy(): void
    {
        $spans = [
            $this->createMockSpan('Low', 0.3),
            $this->createMockSpan('Mid', 0.5),
            $this->createMockSpan('High', 0.9),
        ];

        $result = new OcrResult($spans);

        $this->assertEquals(0.3, $result->getMinConfidence());
        $this->assertEquals(0.9, $result->getMaxConfidence());
        $this->assertAlmostEquals(0.5667, $result->getAverageConfidence(), 3);
    }
}
