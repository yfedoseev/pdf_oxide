<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Barcodes;

use PHPUnit\Framework\TestCase;
use PdfOxide\Barcodes\DetectedBarcode;
use PdfOxide\FFI\FunctionBindings;
use FFI\CData;

/**
 * Tests for DetectedBarcode class
 *
 * @covers \PdfOxide\Barcodes\DetectedBarcode
 */
class DetectedBarcodeTest extends TestCase
{
    private FunctionBindings $bindingsMock;
    private CData $barcodeMock;

    protected function setUp(): void
    {
        $this->bindingsMock = $this->getMockBuilder(FunctionBindings::class)
            ->disableOriginalConstructor()
            ->onlyMethods([
                'pdfDetectedBarcodeGetFormat',
                'pdfDetectedBarcodeGetData',
                'pdfDetectedBarcodeGetBbox',
                'pdfDetectedBarcodeGetConfidence',
                'pdfDetectedBarcodeFree',
            ])
            ->getMock();

        $this->barcodeMock = $this->getMockBuilder(CData::class)->getMock();
    }

    /**
     * Create a DetectedBarcode with mock data
     */
    private function createDetectedBarcode(
        string $format = 'QR_CODE',
        string $data = 'https://example.com',
        array $bbox = ['x' => 10, 'y' => 20, 'width' => 100, 'height' => 100],
        float $confidence = 0.95
    ): DetectedBarcode {
        $this->bindingsMock
            ->method('pdfDetectedBarcodeGetFormat')
            ->willReturn($format);

        $this->bindingsMock
            ->method('pdfDetectedBarcodeGetData')
            ->willReturn($data);

        $this->bindingsMock
            ->method('pdfDetectedBarcodeGetBbox')
            ->willReturn($bbox);

        $this->bindingsMock
            ->method('pdfDetectedBarcodeGetConfidence')
            ->willReturn($confidence);

        $this->bindingsMock
            ->method('pdfDetectedBarcodeFree')
            ->with($this->barcodeMock);

        return new DetectedBarcode($this->barcodeMock, $this->bindingsMock);
    }

    /**
     * Test DetectedBarcode creation
     */
    public function testDetectedBarcodeCreation(): void
    {
        $barcode = $this->createDetectedBarcode();
        $this->assertInstanceOf(DetectedBarcode::class, $barcode);
    }

    /**
     * Test getFormat returns correct format
     */
    public function testGetFormat(): void
    {
        $barcode = $this->createDetectedBarcode('CODE128');
        $this->assertEquals('CODE128', $barcode->getFormat());
    }

    /**
     * Test getFormat with QR code
     */
    public function testGetFormatQrCode(): void
    {
        $barcode = $this->createDetectedBarcode('QR_CODE');
        $this->assertEquals('QR_CODE', $barcode->getFormat());
    }

    /**
     * Test getFormat with EAN-13
     */
    public function testGetFormatEan13(): void
    {
        $barcode = $this->createDetectedBarcode('EAN_13');
        $this->assertEquals('EAN_13', $barcode->getFormat());
    }

    /**
     * Test getData returns correct data
     */
    public function testGetData(): void
    {
        $data = 'https://example.com/product/12345';
        $barcode = $this->createDetectedBarcode('QR_CODE', $data);

        $this->assertEquals($data, $barcode->getData());
    }

    /**
     * Test getData with numeric barcode
     */
    public function testGetDataNumeric(): void
    {
        $data = '5901234123457';  // EAN-13
        $barcode = $this->createDetectedBarcode('EAN_13', $data);

        $this->assertEquals($data, $barcode->getData());
    }

    /**
     * Test getData with special characters
     */
    public function testGetDataSpecialChars(): void
    {
        $data = 'Product-SKU:ABC-123/XYZ';
        $barcode = $this->createDetectedBarcode('CODE128', $data);

        $this->assertEquals($data, $barcode->getData());
    }

    /**
     * Test getBbox returns correct bounding box
     */
    public function testGetBbox(): void
    {
        $bbox = ['x' => 50, 'y' => 100, 'width' => 200, 'height' => 150];
        $barcode = $this->createDetectedBarcode('QR_CODE', 'data', $bbox);

        $result = $barcode->getBbox();
        $this->assertEquals($bbox, $result);
    }

    /**
     * Test getBbox has required keys
     */
    public function testGetBboxHasRequiredKeys(): void
    {
        $barcode = $this->createDetectedBarcode();
        $bbox = $barcode->getBbox();

        $this->assertArrayHasKey('x', $bbox);
        $this->assertArrayHasKey('y', $bbox);
        $this->assertArrayHasKey('width', $bbox);
        $this->assertArrayHasKey('height', $bbox);
    }

    /**
     * Test getConfidence returns correct value
     */
    public function testGetConfidence(): void
    {
        $barcode = $this->createDetectedBarcode('QR_CODE', 'data', ['x' => 0, 'y' => 0, 'width' => 100, 'height' => 100], 0.87);
        $this->assertEquals(0.87, $barcode->getConfidence());
    }

    /**
     * Test getConfidence with perfect confidence
     */
    public function testGetConfidencePerfect(): void
    {
        $barcode = $this->createDetectedBarcode('QR_CODE', 'data', ['x' => 0, 'y' => 0, 'width' => 100, 'height' => 100], 1.0);
        $this->assertEquals(1.0, $barcode->getConfidence());
    }

    /**
     * Test getConfidence with low confidence
     */
    public function testGetConfidenceLow(): void
    {
        $barcode = $this->createDetectedBarcode('QR_CODE', 'data', ['x' => 0, 'y' => 0, 'width' => 100, 'height' => 100], 0.3);
        $this->assertEquals(0.3, $barcode->getConfidence());
    }

    /**
     * Test isQrCode returns true for QR codes
     */
    public function testIsQrCodeTrue(): void
    {
        $barcode = $this->createDetectedBarcode('QR_CODE');
        $this->assertTrue($barcode->isQrCode());
    }

    /**
     * Test isQrCode returns false for other formats
     */
    public function testIsQrCodeFalse(): void
    {
        $barcode = $this->createDetectedBarcode('CODE128');
        $this->assertFalse($barcode->isQrCode());
    }

    /**
     * Test is1D returns true for 1D barcodes
     */
    public function testIs1DTrue(): void
    {
        $oneDFormats = ['CODE128', 'CODE39', 'EAN_13', 'EAN_8', 'UPC_A'];

        foreach ($oneDFormats as $format) {
            $barcode = $this->createDetectedBarcode($format);
            $this->assertTrue($barcode->is1D(), "Format $format should be 1D");
        }
    }

    /**
     * Test is1D returns false for 2D barcodes
     */
    public function testIs1DFalse(): void
    {
        $barcode = $this->createDetectedBarcode('QR_CODE');
        $this->assertFalse($barcode->is1D());
    }

    /**
     * Test is2D returns true for 2D barcodes
     */
    public function testIs2DTrue(): void
    {
        $twoDFormats = ['QR_CODE', 'PDF417', 'DATA_MATRIX'];

        foreach ($twoDFormats as $format) {
            $barcode = $this->createDetectedBarcode($format);
            $this->assertTrue($barcode->is2D(), "Format $format should be 2D");
        }
    }

    /**
     * Test is2D returns false for 1D barcodes
     */
    public function testIs2DFalse(): void
    {
        $barcode = $this->createDetectedBarcode('CODE128');
        $this->assertFalse($barcode->is2D());
    }

    /**
     * Test toArray includes all fields
     */
    public function testToArray(): void
    {
        $barcode = $this->createDetectedBarcode('QR_CODE', 'data', ['x' => 10, 'y' => 20, 'width' => 100, 'height' => 100], 0.92);
        $array = $barcode->toArray();

        $this->assertArrayHasKey('format', $array);
        $this->assertArrayHasKey('data', $array);
        $this->assertArrayHasKey('bbox', $array);
        $this->assertArrayHasKey('confidence', $array);
        $this->assertArrayHasKey('is_qr_code', $array);
        $this->assertArrayHasKey('is_1d', $array);
        $this->assertArrayHasKey('is_2d', $array);
    }

    /**
     * Test toArray values are correct
     */
    public function testToArrayValues(): void
    {
        $barcode = $this->createDetectedBarcode('CODE128', 'test', ['x' => 5, 'y' => 10, 'width' => 50, 'height' => 30], 0.88);
        $array = $barcode->toArray();

        $this->assertEquals('CODE128', $array['format']);
        $this->assertEquals('test', $array['data']);
        $this->assertEquals(['x' => 5, 'y' => 10, 'width' => 50, 'height' => 30], $array['bbox']);
        $this->assertEquals(0.88, $array['confidence']);
        $this->assertFalse($array['is_qr_code']);
        $this->assertTrue($array['is_1d']);
        $this->assertFalse($array['is_2d']);
    }

    /**
     * Test __toString method
     */
    public function testToString(): void
    {
        $barcode = $this->createDetectedBarcode('QR_CODE', 'https://example.com');
        $str = (string)$barcode;

        $this->assertStringContainsString('QR_CODE', $str);
        $this->assertStringContainsString('https://example.com', $str);
    }

    /**
     * Test barcode is immutable after construction
     */
    public function testBarcodeImmutable(): void
    {
        $barcode = $this->createDetectedBarcode();

        $this->expectException(\Exception::class);

        $reflection = new \ReflectionClass($barcode);
        $property = $reflection->getProperty('format');
        $property->setAccessible(true);
        $property->setValue($barcode, 'MODIFIED');
    }

    /**
     * Test FFI handle is freed after construction
     */
    public function testHandleFreedAfterConstruction(): void
    {
        $this->bindingsMock
            ->method('pdfDetectedBarcodeGetFormat')
            ->willReturn('QR_CODE');

        $this->bindingsMock
            ->method('pdfDetectedBarcodeGetData')
            ->willReturn('data');

        $this->bindingsMock
            ->method('pdfDetectedBarcodeGetBbox')
            ->willReturn(['x' => 0, 'y' => 0, 'width' => 100, 'height' => 100]);

        $this->bindingsMock
            ->method('pdfDetectedBarcodeGetConfidence')
            ->willReturn(0.9);

        $this->bindingsMock
            ->expects($this->once())
            ->method('pdfDetectedBarcodeFree')
            ->with($this->barcodeMock);

        new DetectedBarcode($this->barcodeMock, $this->bindingsMock);
    }

    /**
     * Test multiple barcodes coexist independently
     */
    public function testMultipleBarcodesIndependent(): void
    {
        $barcode1 = $this->createDetectedBarcode('QR_CODE', 'data1');
        $barcode2 = $this->createDetectedBarcode('CODE128', 'data2');

        $this->assertEquals('QR_CODE', $barcode1->getFormat());
        $this->assertEquals('data1', $barcode1->getData());

        $this->assertEquals('CODE128', $barcode2->getFormat());
        $this->assertEquals('data2', $barcode2->getData());
    }

    /**
     * Test all supported barcode formats
     */
    public function testAllBarcodeFormats(): void
    {
        $formats = ['QR_CODE', 'CODE128', 'CODE39', 'EAN_13', 'EAN_8', 'UPC_A', 'PDF417', 'DATA_MATRIX'];

        foreach ($formats as $format) {
            $barcode = $this->createDetectedBarcode($format);
            $this->assertEquals($format, $barcode->getFormat());
        }
    }

    /**
     * Test toArray with all barcode types
     */
    public function testToArrayWithAllTypes(): void
    {
        // QR Code (2D)
        $qr = $this->createDetectedBarcode('QR_CODE', 'qr_data');
        $qrArray = $qr->toArray();
        $this->assertTrue($qrArray['is_qr_code']);
        $this->assertTrue($qrArray['is_2d']);
        $this->assertFalse($qrArray['is_1d']);

        // Code128 (1D)
        $code = $this->createDetectedBarcode('CODE128', 'code_data');
        $codeArray = $code->toArray();
        $this->assertFalse($codeArray['is_qr_code']);
        $this->assertTrue($codeArray['is_1d']);
        $this->assertFalse($codeArray['is_2d']);
    }
}
