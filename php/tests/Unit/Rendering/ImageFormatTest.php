<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Rendering;

use PHPUnit\Framework\TestCase;
use PdfOxide\Enums\ImageFormat;

/**
 * Tests for ImageFormat enum
 *
 * @covers \PdfOxide\Enums\ImageFormat
 */
class ImageFormatTest extends TestCase
{
    /**
     * Test PNG enum case
     */
    public function testPngEnumCase(): void
    {
        $format = ImageFormat::PNG;
        $this->assertEquals('png', $format->value);
    }

    /**
     * Test JPEG enum case
     */
    public function testJpegEnumCase(): void
    {
        $format = ImageFormat::JPEG;
        $this->assertEquals('jpeg', $format->value);
    }

    /**
     * Test WEBP enum case
     */
    public function testWebpEnumCase(): void
    {
        $format = ImageFormat::WEBP;
        $this->assertEquals('webp', $format->value);
    }

    /**
     * Test PNG MIME type
     */
    public function testPngMimeType(): void
    {
        $this->assertEquals('image/png', ImageFormat::PNG->mimeType());
    }

    /**
     * Test JPEG MIME type
     */
    public function testJpegMimeType(): void
    {
        $this->assertEquals('image/jpeg', ImageFormat::JPEG->mimeType());
    }

    /**
     * Test WEBP MIME type
     */
    public function testWebpMimeType(): void
    {
        $this->assertEquals('image/webp', ImageFormat::WEBP->mimeType());
    }

    /**
     * Test PNG extension
     */
    public function testPngExtension(): void
    {
        $this->assertEquals('png', ImageFormat::PNG->extension());
    }

    /**
     * Test JPEG extension (jpg)
     */
    public function testJpegExtension(): void
    {
        $this->assertEquals('jpg', ImageFormat::JPEG->extension());
    }

    /**
     * Test WEBP extension
     */
    public function testWebpExtension(): void
    {
        $this->assertEquals('webp', ImageFormat::WEBP->extension());
    }

    /**
     * Test PNG description
     */
    public function testPngDescription(): void
    {
        $desc = ImageFormat::PNG->description();
        $this->assertStringContainsString('PNG', $desc);
        $this->assertStringContainsString('lossless', strtolower($desc));
    }

    /**
     * Test JPEG description
     */
    public function testJpegDescription(): void
    {
        $desc = ImageFormat::JPEG->description();
        $this->assertStringContainsString('JPEG', $desc);
        $this->assertStringContainsString('lossy', strtolower($desc));
    }

    /**
     * Test WEBP description
     */
    public function testWebpDescription(): void
    {
        $desc = ImageFormat::WEBP->description();
        $this->assertStringContainsString('WebP', $desc);
    }

    /**
     * Test PNG is lossless
     */
    public function testPngIsLossless(): void
    {
        $this->assertTrue(ImageFormat::PNG->isLossless());
    }

    /**
     * Test JPEG is not lossless
     */
    public function testJpegIsNotLossless(): void
    {
        $this->assertFalse(ImageFormat::JPEG->isLossless());
    }

    /**
     * Test WEBP can be lossless
     */
    public function testWebpIsLossless(): void
    {
        $this->assertTrue(ImageFormat::WEBP->isLossless());
    }

    /**
     * Test PNG supports transparency
     */
    public function testPngSupportsTransparency(): void
    {
        $this->assertTrue(ImageFormat::PNG->supportsTransparency());
    }

    /**
     * Test JPEG does not support transparency
     */
    public function testJpegNoTransparency(): void
    {
        $this->assertFalse(ImageFormat::JPEG->supportsTransparency());
    }

    /**
     * Test WEBP supports transparency
     */
    public function testWebpSupportsTransparency(): void
    {
        $this->assertTrue(ImageFormat::WEBP->supportsTransparency());
    }

    /**
     * Test PNG default quality
     */
    public function testPngDefaultQuality(): void
    {
        $this->assertEquals(0, ImageFormat::PNG->defaultQuality());
    }

    /**
     * Test JPEG default quality
     */
    public function testJpegDefaultQuality(): void
    {
        $quality = ImageFormat::JPEG->defaultQuality();
        $this->assertGreaterThan(0, $quality);
        $this->assertLessThanOrEqual(100, $quality);
        $this->assertEquals(85, $quality);
    }

    /**
     * Test WEBP default quality
     */
    public function testWebpDefaultQuality(): void
    {
        $quality = ImageFormat::WEBP->defaultQuality();
        $this->assertGreaterThan(0, $quality);
        $this->assertLessThanOrEqual(100, $quality);
        $this->assertEquals(80, $quality);
    }

    /**
     * Test tryFromString with 'png'
     */
    public function testTryFromStringPng(): void
    {
        $format = ImageFormat::tryFromString('png');
        $this->assertSame(ImageFormat::PNG, $format);
    }

    /**
     * Test tryFromString with 'jpeg'
     */
    public function testTryFromStringJpeg(): void
    {
        $format = ImageFormat::tryFromString('jpeg');
        $this->assertSame(ImageFormat::JPEG, $format);
    }

    /**
     * Test tryFromString with 'jpg' (alias)
     */
    public function testTryFromStringJpg(): void
    {
        $format = ImageFormat::tryFromString('jpg');
        $this->assertSame(ImageFormat::JPEG, $format);
    }

    /**
     * Test tryFromString with 'webp'
     */
    public function testTryFromStringWebp(): void
    {
        $format = ImageFormat::tryFromString('webp');
        $this->assertSame(ImageFormat::WEBP, $format);
    }

    /**
     * Test tryFromString with uppercase
     */
    public function testTryFromStringUppercase(): void
    {
        $format = ImageFormat::tryFromString('PNG');
        $this->assertSame(ImageFormat::PNG, $format);
    }

    /**
     * Test tryFromString with mixed case
     */
    public function testTryFromStringMixedCase(): void
    {
        $format = ImageFormat::tryFromString('JpEg');
        $this->assertSame(ImageFormat::JPEG, $format);
    }

    /**
     * Test tryFromString with whitespace
     */
    public function testTryFromStringWithWhitespace(): void
    {
        $format = ImageFormat::tryFromString('  png  ');
        $this->assertSame(ImageFormat::PNG, $format);
    }

    /**
     * Test tryFromString with invalid format returns null
     */
    public function testTryFromStringInvalidReturnsNull(): void
    {
        $format = ImageFormat::tryFromString('bmp');
        $this->assertNull($format);
    }

    /**
     * Test tryFromString with empty string returns null
     */
    public function testTryFromStringEmptyReturnsNull(): void
    {
        $format = ImageFormat::tryFromString('');
        $this->assertNull($format);
    }

    /**
     * Test isValid with valid formats
     */
    public function testIsValidWithValidFormats(): void
    {
        $this->assertTrue(ImageFormat::isValid('png'));
        $this->assertTrue(ImageFormat::isValid('jpeg'));
        $this->assertTrue(ImageFormat::isValid('jpg'));
        $this->assertTrue(ImageFormat::isValid('webp'));
    }

    /**
     * Test isValid with invalid formats
     */
    public function testIsValidWithInvalidFormats(): void
    {
        $this->assertFalse(ImageFormat::isValid('bmp'));
        $this->assertFalse(ImageFormat::isValid('gif'));
        $this->assertFalse(ImageFormat::isValid('tiff'));
        $this->assertFalse(ImageFormat::isValid(''));
    }

    /**
     * Test isValid is case insensitive
     */
    public function testIsValidCaseInsensitive(): void
    {
        $this->assertTrue(ImageFormat::isValid('PNG'));
        $this->assertTrue(ImageFormat::isValid('JpEg'));
        $this->assertTrue(ImageFormat::isValid('WEBP'));
    }

    /**
     * Test all enum cases are accessible
     */
    public function testAllCasesAccessible(): void
    {
        $cases = ImageFormat::cases();
        $this->assertCount(3, $cases);
        $this->assertContains(ImageFormat::PNG, $cases);
        $this->assertContains(ImageFormat::JPEG, $cases);
        $this->assertContains(ImageFormat::WEBP, $cases);
    }

    /**
     * Test enum values match expected strings
     */
    public function testEnumValuesMatchStrings(): void
    {
        $this->assertEquals('png', ImageFormat::PNG->value);
        $this->assertEquals('jpeg', ImageFormat::JPEG->value);
        $this->assertEquals('webp', ImageFormat::WEBP->value);
    }

    /**
     * Test MIME type format is valid
     */
    public function testMimeTypeFormat(): void
    {
        foreach (ImageFormat::cases() as $format) {
            $mime = $format->mimeType();
            $this->assertStringContainsString('/', $mime);
            $this->assertStringStartsWith('image/', $mime);
        }
    }

    /**
     * Test extension does not include leading dot
     */
    public function testExtensionNoLeadingDot(): void
    {
        foreach (ImageFormat::cases() as $format) {
            $ext = $format->extension();
            $this->assertStringNotStartsWith('.', $ext);
        }
    }

    /**
     * Test extension is lowercase
     */
    public function testExtensionIsLowercase(): void
    {
        foreach (ImageFormat::cases() as $format) {
            $ext = $format->extension();
            $this->assertEquals(strtolower($ext), $ext);
        }
    }

    /**
     * Test description is non-empty
     */
    public function testDescriptionNonEmpty(): void
    {
        foreach (ImageFormat::cases() as $format) {
            $desc = $format->description();
            $this->assertNotEmpty($desc);
            $this->assertIsString($desc);
        }
    }
}
