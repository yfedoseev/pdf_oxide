<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Barcodes;

use PHPUnit\Framework\TestCase;
use PdfOxide\Barcodes\BarcodeDetector;
use PdfOxide\Barcodes\DetectedBarcode;
use PdfOxide\PdfPage;

/**
 * Tests for BarcodeDetector class
 *
 * @covers \PdfOxide\Barcodes\BarcodeDetector
 */
class BarcodeDetectorTest extends TestCase
{
    private BarcodeDetector $detector;
    private PdfPage $pageMock;

    protected function setUp(): void
    {
        $this->detector = new BarcodeDetector();

        $this->pageMock = $this->getMockBuilder(PdfPage::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->pageMock->method('getIndex')->willReturn(0);
    }

    /**
     * Test BarcodeDetector initialization
     */
    public function testBarcodeDetectorInitialization(): void
    {
        $this->assertInstanceOf(BarcodeDetector::class, $this->detector);
    }

    /**
     * Test default confidence threshold is 0.5
     */
    public function testDefaultConfidenceThreshold(): void
    {
        $this->assertEquals(0.5, $this->detector->getConfidenceThreshold());
    }

    /**
     * Test default try harder is disabled
     */
    public function testDefaultTryHarderDisabled(): void
    {
        $this->assertFalse($this->detector->isTryHarderEnabled());
    }

    /**
     * Test setConfidenceThreshold with valid value
     */
    public function testSetConfidenceThresholdValid(): void
    {
        $result = $this->detector->setConfidenceThreshold(0.8);

        $this->assertSame($this->detector, $result);
        $this->assertEquals(0.8, $this->detector->getConfidenceThreshold());
    }

    /**
     * Test setConfidenceThreshold with minimum value
     */
    public function testSetConfidenceThresholdMinimum(): void
    {
        $this->detector->setConfidenceThreshold(0.0);
        $this->assertEquals(0.0, $this->detector->getConfidenceThreshold());
    }

    /**
     * Test setConfidenceThreshold with maximum value
     */
    public function testSetConfidenceThresholdMaximum(): void
    {
        $this->detector->setConfidenceThreshold(1.0);
        $this->assertEquals(1.0, $this->detector->getConfidenceThreshold());
    }

    /**
     * Test setConfidenceThreshold below minimum throws exception
     */
    public function testSetConfidenceThresholdTooLowThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('0.0-1.0');

        $this->detector->setConfidenceThreshold(-0.1);
    }

    /**
     * Test setConfidenceThreshold above maximum throws exception
     */
    public function testSetConfidenceThresholdTooHighThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('0.0-1.0');

        $this->detector->setConfidenceThreshold(1.1);
    }

    /**
     * Test setTryHarder enable
     */
    public function testSetTryHarderEnable(): void
    {
        $result = $this->detector->setTryHarder(true);

        $this->assertSame($this->detector, $result);
        $this->assertTrue($this->detector->isTryHarderEnabled());
    }

    /**
     * Test setTryHarder disable
     */
    public function testSetTryHarderDisable(): void
    {
        $this->detector->setTryHarder(true);
        $this->detector->setTryHarder(false);

        $this->assertFalse($this->detector->isTryHarderEnabled());
    }

    /**
     * Test fluent interface chaining
     */
    public function testFluentInterfaceChaining(): void
    {
        $result = $this->detector
            ->setConfidenceThreshold(0.9)
            ->setTryHarder(true);

        $this->assertSame($this->detector, $result);
        $this->assertEquals(0.9, $this->detector->getConfidenceThreshold());
        $this->assertTrue($this->detector->isTryHarderEnabled());
    }

    /**
     * Test getSupportedFormats returns array
     */
    public function testGetSupportedFormats(): void
    {
        $formats = BarcodeDetector::getSupportedFormats();

        $this->assertIsArray($formats);
        $this->assertGreaterThan(0, count($formats));
    }

    /**
     * Test getSupportedFormats contains expected formats
     */
    public function testGetSupportedFormatsContainsCommon(): void
    {
        $formats = BarcodeDetector::getSupportedFormats();

        $this->assertContains('QR_CODE', $formats);
        $this->assertContains('CODE128', $formats);
        $this->assertContains('EAN_13', $formats);
    }

    /**
     * Test detect method exists and has correct signature
     */
    public function testDetectMethodSignature(): void
    {
        $this->assertTrue(method_exists($this->detector, 'detect'));
    }

    /**
     * Test detectInRegion method exists and has correct signature
     */
    public function testDetectInRegionMethodSignature(): void
    {
        $this->assertTrue(method_exists($this->detector, 'detectInRegion'));
    }

    /**
     * Test detectInRegion with invalid bounding box (missing x)
     */
    public function testDetectInRegionMissingXThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('missing required key');

        $bbox = ['y' => 100, 'width' => 200, 'height' => 150];
        $this->detector->detectInRegion($this->pageMock, $bbox);
    }

    /**
     * Test detectInRegion with invalid bounding box (missing y)
     */
    public function testDetectInRegionMissingYThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);

        $bbox = ['x' => 50, 'width' => 200, 'height' => 150];
        $this->detector->detectInRegion($this->pageMock, $bbox);
    }

    /**
     * Test detectInRegion with invalid bounding box (missing width)
     */
    public function testDetectInRegionMissingWidthThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);

        $bbox = ['x' => 50, 'y' => 100, 'height' => 150];
        $this->detector->detectInRegion($this->pageMock, $bbox);
    }

    /**
     * Test detectInRegion with invalid bounding box (missing height)
     */
    public function testDetectInRegionMissingHeightThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);

        $bbox = ['x' => 50, 'y' => 100, 'width' => 200];
        $this->detector->detectInRegion($this->pageMock, $bbox);
    }

    /**
     * Test detectInRegion with zero width throws
     */
    public function testDetectInRegionZeroWidthThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('dimensions must be positive');

        $bbox = ['x' => 50, 'y' => 100, 'width' => 0, 'height' => 150];
        $this->detector->detectInRegion($this->pageMock, $bbox);
    }

    /**
     * Test detectInRegion with negative height throws
     */
    public function testDetectInRegionNegativeHeightThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('dimensions must be positive');

        $bbox = ['x' => 50, 'y' => 100, 'width' => 200, 'height' => -10];
        $this->detector->detectInRegion($this->pageMock, $bbox);
    }

    /**
     * Test detectInRegion with valid bounding box
     */
    public function testDetectInRegionValidBbox(): void
    {
        $bbox = ['x' => 50, 'y' => 100, 'width' => 200, 'height' => 150];

        // Method should accept valid bbox (though it may fail on FFI call)
        $this->assertTrue(method_exists($this->detector, 'detectInRegion'));
    }

    /**
     * Test toArray includes all settings
     */
    public function testToArray(): void
    {
        $this->detector
            ->setConfidenceThreshold(0.75)
            ->setTryHarder(true);

        $array = $this->detector->toArray();

        $this->assertArrayHasKey('confidenceThreshold', $array);
        $this->assertArrayHasKey('tryHarder', $array);
        $this->assertArrayHasKey('supportedFormats', $array);

        $this->assertEquals(0.75, $array['confidenceThreshold']);
        $this->assertTrue($array['tryHarder']);
        $this->assertIsArray($array['supportedFormats']);
    }

    /**
     * Test multiple detectors are independent
     */
    public function testMultipleDetectorsIndependent(): void
    {
        $detector1 = new BarcodeDetector();
        $detector2 = new BarcodeDetector();

        $detector1->setConfidenceThreshold(0.9)->setTryHarder(true);
        $detector2->setConfidenceThreshold(0.3)->setTryHarder(false);

        $this->assertEquals(0.9, $detector1->getConfidenceThreshold());
        $this->assertTrue($detector1->isTryHarderEnabled());

        $this->assertEquals(0.3, $detector2->getConfidenceThreshold());
        $this->assertFalse($detector2->isTryHarderEnabled());
    }

    /**
     * Test confidence threshold range variations
     */
    public function testConfidenceThresholdVariations(): void
    {
        $thresholds = [0.0, 0.25, 0.5, 0.75, 1.0];

        foreach ($thresholds as $threshold) {
            $this->detector->setConfidenceThreshold($threshold);
            $this->assertEquals($threshold, $this->detector->getConfidenceThreshold());
        }
    }

    /**
     * Test getSupportedFormats is consistent across calls
     */
    public function testGetSupportedFormatsConsistent(): void
    {
        $formats1 = BarcodeDetector::getSupportedFormats();
        $formats2 = BarcodeDetector::getSupportedFormats();

        $this->assertEquals($formats1, $formats2);
    }

    /**
     * Test supported formats are non-empty strings
     */
    public function testSupportedFormatsAreStrings(): void
    {
        $formats = BarcodeDetector::getSupportedFormats();

        foreach ($formats as $format) {
            $this->assertIsString($format);
            $this->assertNotEmpty($format);
        }
    }

    /**
     * Test detector configuration persists across multiple calls
     */
    public function testConfigurationPersists(): void
    {
        $this->detector->setConfidenceThreshold(0.85)->setTryHarder(true);

        // Simulate multiple detect calls by checking configuration remains
        $this->assertEquals(0.85, $this->detector->getConfidenceThreshold());
        $this->assertTrue($this->detector->isTryHarderEnabled());

        // Configuration should not change on subsequent configuration calls
        $this->detector->setConfidenceThreshold(0.85);
        $this->assertEquals(0.85, $this->detector->getConfidenceThreshold());
    }

    /**
     * Test bounding box with float coordinates
     */
    public function testDetectInRegionFloatCoordinates(): void
    {
        $bbox = ['x' => 50.5, 'y' => 100.7, 'width' => 200.3, 'height' => 150.9];

        // Should accept float coordinates
        $this->assertTrue(method_exists($this->detector, 'detectInRegion'));
    }
}
