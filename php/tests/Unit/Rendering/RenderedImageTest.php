<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Rendering;

use PHPUnit\Framework\TestCase;
use PdfOxide\Types\RenderedImage;
use PdfOxide\Builders\RenderingOptions;
use FFI\CData;

/**
 * Tests for RenderedImage class
 *
 * @covers \PdfOxide\Types\RenderedImage
 */
class RenderedImageTest extends TestCase
{
    private RenderedImage $image;
    private RenderingOptions $optionsMock;

    protected function setUp(): void
    {
        $this->optionsMock = new RenderingOptions();
        $this->optionsMock->imageFormat('png');

        $this->image = new RenderedImage(null, $this->optionsMock);
    }

    /**
     * Test RenderedImage creation with null handle
     */
    public function testRenderedImageCreationWithNullHandle(): void
    {
        $this->assertInstanceOf(RenderedImage::class, $this->image);
    }

    /**
     * Test getFormat returns correct format
     */
    public function testGetFormatPng(): void
    {
        $this->assertEquals('png', $this->image->getFormat());
    }

    /**
     * Test getFormat with JPEG
     */
    public function testGetFormatJpeg(): void
    {
        $options = new RenderingOptions();
        $options->imageFormat('jpeg');
        $image = new RenderedImage(null, $options);

        $this->assertEquals('jpeg', $image->getFormat());
    }

    /**
     * Test getFormat with WebP
     */
    public function testGetFormatWebp(): void
    {
        $options = new RenderingOptions();
        $options->imageFormat('webp');
        $image = new RenderedImage(null, $options);

        $this->assertEquals('webp', $image->getFormat());
    }

    /**
     * Test getWidth with null handle returns 0
     */
    public function testGetWidthNullHandle(): void
    {
        $this->assertEquals(0, $this->image->getWidth());
    }

    /**
     * Test getHeight with null handle returns 0
     */
    public function testGetHeightNullHandle(): void
    {
        $this->assertEquals(0, $this->image->getHeight());
    }

    /**
     * Test getSize with null handle returns 0
     */
    public function testGetSizeNullHandle(): void
    {
        $this->assertEquals(0, $this->image->getSize());
    }

    /**
     * Test getData with null handle returns empty string
     */
    public function testGetDataNullHandle(): void
    {
        $this->assertEquals('', $this->image->getData());
    }

    /**
     * Test getAspectRatio with zero dimensions
     */
    public function testGetAspectRatioZeroDimensions(): void
    {
        $this->assertEquals(0, $this->image->getAspectRatio());
    }

    /**
     * Test getMimeTypePng
     */
    public function testGetMimeTypePng(): void
    {
        $this->assertEquals('image/png', $this->image->getMimeType());
    }

    /**
     * Test getMimeTypeJpeg
     */
    public function testGetMimeTypeJpeg(): void
    {
        $options = new RenderingOptions();
        $options->imageFormat('jpeg');
        $image = new RenderedImage(null, $options);

        $this->assertEquals('image/jpeg', $image->getMimeType());
    }

    /**
     * Test getMimeTypeWebp
     */
    public function testGetMimeTypeWebp(): void
    {
        $options = new RenderingOptions();
        $options->imageFormat('webp');
        $image = new RenderedImage(null, $options);

        $this->assertEquals('image/webp', $image->getMimeType());
    }

    /**
     * Test getExtensionPng
     */
    public function testGetExtensionPng(): void
    {
        $this->assertEquals('png', $this->image->getExtension());
    }

    /**
     * Test getExtensionJpeg
     */
    public function testGetExtensionJpeg(): void
    {
        $options = new RenderingOptions();
        $options->imageFormat('jpeg');
        $image = new RenderedImage(null, $options);

        $this->assertEquals('jpg', $image->getExtension());
    }

    /**
     * Test getExtensionWebp
     */
    public function testGetExtensionWebp(): void
    {
        $options = new RenderingOptions();
        $options->imageFormat('webp');
        $image = new RenderedImage(null, $options);

        $this->assertEquals('webp', $image->getExtension());
    }

    /**
     * Test hasData with null handle
     */
    public function testHasDataNullHandle(): void
    {
        $this->assertFalse($this->image->hasData());
    }

    /**
     * Test toBase64 without MIME prefix
     */
    public function testToBase64WithoutMimePrefix(): void
    {
        $base64 = $this->image->toBase64();
        $this->assertEquals('', $base64);  // Empty data returns empty base64
    }

    /**
     * Test toBase64 with MIME prefix for PNG
     */
    public function testToBase64WithMimePrefixPng(): void
    {
        $base64 = $this->image->toBase64(true);
        // Empty data still produces valid data URL
        $this->assertStringStartsWith('data:image/png;base64,', $base64);
    }

    /**
     * Test toBase64 with MIME prefix for JPEG
     */
    public function testToBase64WithMimePrefixJpeg(): void
    {
        $options = new RenderingOptions();
        $options->imageFormat('jpeg');
        $image = new RenderedImage(null, $options);

        $base64 = $image->toBase64(true);
        $this->assertStringStartsWith('data:image/jpeg;base64,', $base64);
    }

    /**
     * Test toBase64 with MIME prefix for WebP
     */
    public function testToBase64WithMimePrefixWebp(): void
    {
        $options = new RenderingOptions();
        $options->imageFormat('webp');
        $image = new RenderedImage(null, $options);

        $base64 = $image->toBase64(true);
        $this->assertStringStartsWith('data:image/webp;base64,', $base64);
    }

    /**
     * Test toArray includes expected keys
     */
    public function testToArrayStructure(): void
    {
        $array = $this->image->toArray();

        $this->assertArrayHasKey('width', $array);
        $this->assertArrayHasKey('height', $array);
        $this->assertArrayHasKey('format', $array);
        $this->assertArrayHasKey('aspect_ratio', $array);
        $this->assertArrayHasKey('size_bytes', $array);
        $this->assertArrayHasKey('mime_type', $array);
        $this->assertArrayHasKey('extension', $array);
    }

    /**
     * Test toArray with null handle
     */
    public function testToArrayWithNullHandle(): void
    {
        $array = $this->image->toArray();

        $this->assertEquals(0, $array['width']);
        $this->assertEquals(0, $array['height']);
        $this->assertEquals('png', $array['format']);
        $this->assertEquals(0, $array['aspect_ratio']);
        $this->assertEquals(0, $array['size_bytes']);
        $this->assertEquals('image/png', $array['mime_type']);
        $this->assertEquals('png', $array['extension']);
    }

    /**
     * Test convertFormat changes format
     */
    public function testConvertFormatPngToJpeg(): void
    {
        $converted = $this->image->convertFormat('jpeg');

        $this->assertInstanceOf(RenderedImage::class, $converted);
        $this->assertEquals('jpeg', $converted->getFormat());
    }

    /**
     * Test convertFormat preserves data
     */
    public function testConvertFormatPreservesDataStructure(): void
    {
        $original = $this->image;
        $converted = $original->convertFormat('webp');

        // Original should remain unchanged
        $this->assertEquals('png', $original->getFormat());
        // Converted should have new format
        $this->assertEquals('webp', $converted->getFormat());
    }

    /**
     * Test MIME type is valid for all formats
     */
    public function testMimeTypeValidity(): void
    {
        $formats = ['png', 'jpeg', 'webp'];

        foreach ($formats as $format) {
            $options = new RenderingOptions();
            $options->imageFormat($format);
            $image = new RenderedImage(null, $options);

            $mime = $image->getMimeType();
            $this->assertStringContainsString('/', $mime);
        }
    }

    /**
     * Test extension is lowercase for all formats
     */
    public function testExtensionIsLowercase(): void
    {
        $formats = ['png', 'jpeg', 'webp'];

        foreach ($formats as $format) {
            $options = new RenderingOptions();
            $options->imageFormat($format);
            $image = new RenderedImage(null, $options);

            $ext = $image->getExtension();
            $this->assertEquals(strtolower($ext), $ext);
        }
    }

    /**
     * Test hasData is false initially
     */
    public function testHasDataInitiallyFalse(): void
    {
        $this->assertFalse($this->image->hasData());
    }

    /**
     * Test getData is empty initially
     */
    public function testGetDataInitiallyEmpty(): void
    {
        $this->assertEquals('', $this->image->getData());
    }

    /**
     * Test getSize is zero initially
     */
    public function testGetSizeInitiallyZero(): void
    {
        $this->assertEquals(0, $this->image->getSize());
    }

    /**
     * Test multiple getFormat calls return same value
     */
    public function testMultipleGetFormatCallsConsistent(): void
    {
        $format1 = $this->image->getFormat();
        $format2 = $this->image->getFormat();

        $this->assertEquals($format1, $format2);
    }

    /**
     * Test toBase64 with different MIME prefix modes
     */
    public function testToBase64MimePrefixModes(): void
    {
        $withoutPrefix = $this->image->toBase64(false);
        $withPrefix = $this->image->toBase64(true);

        // With prefix should start with data URL format
        $this->assertStringStartsWith('data:image/png;base64,', $withPrefix);
        // Without prefix should be just base64
        $this->assertNotStringStartsWith('data:', $withoutPrefix);
    }

    /**
     * Test toArray values are correct types
     */
    public function testToArrayValueTypes(): void
    {
        $array = $this->image->toArray();

        $this->assertIsInt($array['width']);
        $this->assertIsInt($array['height']);
        $this->assertIsString($array['format']);
        $this->assertIsFloat($array['aspect_ratio']);
        $this->assertIsInt($array['size_bytes']);
        $this->assertIsString($array['mime_type']);
        $this->assertIsString($array['extension']);
    }

    /**
     * Test multiple format conversions
     */
    public function testMultipleFormatConversions(): void
    {
        $png = $this->image;
        $jpeg = $png->convertFormat('jpeg');
        $webp = $jpeg->convertFormat('webp');

        $this->assertEquals('png', $png->getFormat());
        $this->assertEquals('jpeg', $jpeg->getFormat());
        $this->assertEquals('webp', $webp->getFormat());
    }

    /**
     * Test convertFormat with all supported formats
     */
    public function testConvertFormatAllSupported(): void
    {
        $formats = ['png', 'jpeg', 'webp'];

        foreach ($formats as $targetFormat) {
            $converted = $this->image->convertFormat($targetFormat);
            $this->assertEquals($targetFormat, $converted->getFormat());
        }
    }

    /**
     * Test toArray after format conversion
     */
    public function testToArrayAfterFormatConversion(): void
    {
        $converted = $this->image->convertFormat('jpeg');
        $array = $converted->toArray();

        $this->assertEquals('jpeg', $array['format']);
        $this->assertEquals('image/jpeg', $array['mime_type']);
        $this->assertEquals('jpg', $array['extension']);
    }

    /**
     * Test SaveToFile with empty data
     */
    public function testSaveToFileEmptyDataFallback(): void
    {
        $tmpFile = tempnam(sys_get_temp_dir(), 'rendered_');

        try {
            $this->image->saveToFile($tmpFile);
            // File should be created even with empty data
            $this->assertTrue(file_exists($tmpFile));
            $this->assertEquals(0, filesize($tmpFile));
        } finally {
            if (file_exists($tmpFile)) {
                unlink($tmpFile);
            }
        }
    }

    /**
     * Test MIME type fallback for unknown formats
     */
    public function testMimeTypeFallback(): void
    {
        // Create options that might have invalid format (shouldn't happen, but test safety)
        $options = new RenderingOptions();
        $options->imageFormat('png');  // Valid format
        $image = new RenderedImage(null, $options);

        $mime = $image->getMimeType();
        $this->assertNotEmpty($mime);
    }
}
