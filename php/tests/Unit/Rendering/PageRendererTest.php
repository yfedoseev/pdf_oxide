<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Rendering;

use PHPUnit\Framework\TestCase;
use PdfOxide\Rendering\PageRenderer;
use PdfOxide\Enums\ImageFormat;
use PdfOxide\PdfPage;
use PdfOxide\Types\RenderedImage;
use FFI\CData;

/**
 * Tests for PageRenderer class
 *
 * @covers \PdfOxide\Rendering\PageRenderer
 * @covers \PdfOxide\Enums\ImageFormat
 */
class PageRendererTest extends TestCase
{
    private PageRenderer $renderer;
    private PdfPage $pageMock;

    protected function setUp(): void
    {
        $this->pageMock = $this->getMockBuilder(PdfPage::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->pageMock->method('getIndex')->willReturn(0);

        $this->renderer = new PageRenderer($this->pageMock);
    }

    /**
     * Test PageRenderer initialization
     */
    public function testRendererInitialization(): void
    {
        $this->assertInstanceOf(PageRenderer::class, $this->renderer);
        $this->assertSame($this->pageMock, $this->renderer->getPage());
    }

    /**
     * Test default format is PNG
     */
    public function testDefaultFormatIsPng(): void
    {
        $this->assertEquals(ImageFormat::PNG, $this->renderer->getFormat());
    }

    /**
     * Test default DPI is 150
     */
    public function testDefaultDpiIs150(): void
    {
        $this->assertEquals(150, $this->renderer->getDpi());
    }

    /**
     * Test default quality is 85
     */
    public function testDefaultQualityIs85(): void
    {
        $this->assertEquals(85, $this->renderer->getQuality());
    }

    /**
     * Test default antialiasing is enabled
     */
    public function testDefaultAntialiasingEnabled(): void
    {
        $this->assertTrue($this->renderer->isAntialiasingEnabled());
    }

    /**
     * Test setFormat with PNG
     */
    public function testSetFormatPng(): void
    {
        $result = $this->renderer->setFormat(ImageFormat::PNG);

        $this->assertSame($this->renderer, $result);
        $this->assertEquals(ImageFormat::PNG, $this->renderer->getFormat());
    }

    /**
     * Test setFormat with JPEG
     */
    public function testSetFormatJpeg(): void
    {
        $this->renderer->setFormat(ImageFormat::JPEG);
        $this->assertEquals(ImageFormat::JPEG, $this->renderer->getFormat());
    }

    /**
     * Test setFormat with WebP
     */
    public function testSetFormatWebp(): void
    {
        $this->renderer->setFormat(ImageFormat::WEBP);
        $this->assertEquals(ImageFormat::WEBP, $this->renderer->getFormat());
    }

    /**
     * Test setDpi with minimum value
     */
    public function testSetDpiMinimum(): void
    {
        $this->renderer->setDpi(1);
        $this->assertEquals(1, $this->renderer->getDpi());
    }

    /**
     * Test setDpi with maximum value
     */
    public function testSetDpiMaximum(): void
    {
        $this->renderer->setDpi(600);
        $this->assertEquals(600, $this->renderer->getDpi());
    }

    /**
     * Test setDpi with typical values
     */
    public function testSetDpiTypicalValues(): void
    {
        $this->renderer->setDpi(72);
        $this->assertEquals(72, $this->renderer->getDpi());

        $this->renderer->setDpi(300);
        $this->assertEquals(300, $this->renderer->getDpi());
    }

    /**
     * Test setDpi below minimum throws exception
     */
    public function testSetDpiTooLowThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('DPI must be between 1 and 600');

        $this->renderer->setDpi(0);
    }

    /**
     * Test setDpi above maximum throws exception
     */
    public function testSetDpiTooHighThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('DPI must be between 1 and 600');

        $this->renderer->setDpi(601);
    }

    /**
     * Test setQuality with minimum value
     */
    public function testSetQualityMinimum(): void
    {
        $this->renderer->setQuality(1);
        $this->assertEquals(1, $this->renderer->getQuality());
    }

    /**
     * Test setQuality with maximum value
     */
    public function testSetQualityMaximum(): void
    {
        $this->renderer->setQuality(100);
        $this->assertEquals(100, $this->renderer->getQuality());
    }

    /**
     * Test setQuality below minimum throws exception
     */
    public function testSetQualityTooLowThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Quality must be between 1 and 100');

        $this->renderer->setQuality(0);
    }

    /**
     * Test setQuality above maximum throws exception
     */
    public function testSetQualityTooHighThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Quality must be between 1 and 100');

        $this->renderer->setQuality(101);
    }

    /**
     * Test setAntialiasing enable
     */
    public function testSetAntialiasingTrue(): void
    {
        $this->renderer->setAntialiasing(true);
        $this->assertTrue($this->renderer->isAntialiasingEnabled());
    }

    /**
     * Test setAntialiasing disable
     */
    public function testSetAntialiasingFalse(): void
    {
        $this->renderer->setAntialiasing(false);
        $this->assertFalse($this->renderer->isAntialiasingEnabled());
    }

    /**
     * Test setBackgroundColor with valid hex
     */
    public function testSetBackgroundColorValid(): void
    {
        $this->renderer->setBackgroundColor('#FFFFFF');
        $this->assertEquals('#FFFFFF', $this->renderer->getBackgroundColor());
    }

    /**
     * Test setBackgroundColor with lowercase hex
     */
    public function testSetBackgroundColorLowercase(): void
    {
        $this->renderer->setBackgroundColor('#ffffff');
        $this->assertEquals('#ffffff', $this->renderer->getBackgroundColor());
    }

    /**
     * Test setBackgroundColor with null
     */
    public function testSetBackgroundColorNull(): void
    {
        $this->renderer->setBackgroundColor('#FFFFFF');
        $this->renderer->setBackgroundColor(null);
        $this->assertNull($this->renderer->getBackgroundColor());
    }

    /**
     * Test setBackgroundColor with invalid format throws
     */
    public function testSetBackgroundColorInvalidThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid color format');

        $this->renderer->setBackgroundColor('FFFFFF');  // Missing #
    }

    /**
     * Test setBackgroundColor with wrong hex length throws
     */
    public function testSetBackgroundColorWrongLengthThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);

        $this->renderer->setBackgroundColor('#FFF');  // Too short
    }

    /**
     * Test fluent interface chaining
     */
    public function testFluentInterfaceChaining(): void
    {
        $result = $this->renderer
            ->setFormat(ImageFormat::JPEG)
            ->setDpi(300)
            ->setQuality(90)
            ->setAntialiasing(false);

        $this->assertSame($this->renderer, $result);
        $this->assertEquals(ImageFormat::JPEG, $this->renderer->getFormat());
        $this->assertEquals(300, $this->renderer->getDpi());
        $this->assertEquals(90, $this->renderer->getQuality());
        $this->assertFalse($this->renderer->isAntialiasingEnabled());
    }

    /**
     * Test toArray includes all settings
     */
    public function testToArrayIncludesAllSettings(): void
    {
        $this->renderer
            ->setFormat(ImageFormat::WEBP)
            ->setDpi(200)
            ->setQuality(95)
            ->setAntialiasing(false)
            ->setBackgroundColor('#000000');

        $array = $this->renderer->toArray();

        $this->assertArrayHasKey('format', $array);
        $this->assertArrayHasKey('dpi', $array);
        $this->assertArrayHasKey('quality', $array);
        $this->assertArrayHasKey('antialiasing', $array);
        $this->assertArrayHasKey('backgroundColor', $array);
        $this->assertArrayHasKey('pageIndex', $array);

        $this->assertEquals('webp', $array['format']);
        $this->assertEquals(200, $array['dpi']);
        $this->assertEquals(95, $array['quality']);
        $this->assertFalse($array['antialiasing']);
        $this->assertEquals('#000000', $array['backgroundColor']);
    }

    /**
     * Test toArray with default settings
     */
    public function testToArrayDefaultSettings(): void
    {
        $array = $this->renderer->toArray();

        $this->assertEquals('png', $array['format']);
        $this->assertEquals(150, $array['dpi']);
        $this->assertEquals(85, $array['quality']);
        $this->assertTrue($array['antialiasing']);
        $this->assertNull($array['backgroundColor']);
        $this->assertEquals(0, $array['pageIndex']);
    }

    /**
     * Test getPage returns correct page
     */
    public function testGetPageReturnsCorrectPage(): void
    {
        $page = $this->renderer->getPage();
        $this->assertSame($this->pageMock, $page);
    }

    /**
     * Test format from different instances are independent
     */
    public function testMultipleRenderersIndependent(): void
    {
        $page1 = $this->getMockBuilder(PdfPage::class)
            ->disableOriginalConstructor()
            ->getMock();

        $page2 = $this->getMockBuilder(PdfPage::class)
            ->disableOriginalConstructor()
            ->getMock();

        $renderer1 = new PageRenderer($page1);
        $renderer2 = new PageRenderer($page2);

        $renderer1->setFormat(ImageFormat::JPEG)->setDpi(100);
        $renderer2->setFormat(ImageFormat::PNG)->setDpi(300);

        $this->assertEquals(ImageFormat::JPEG, $renderer1->getFormat());
        $this->assertEquals(100, $renderer1->getDpi());

        $this->assertEquals(ImageFormat::PNG, $renderer2->getFormat());
        $this->assertEquals(300, $renderer2->getDpi());
    }

    /**
     * Test renderFitWithInvalidWidthThrows
     */
    public function testRenderFitInvalidWidthThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Width and height must be positive');

        // Mock document to avoid full render
        $docMock = $this->getMockBuilder(\PdfOxide\PdfDocument::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->pageMock->method('getDocument')->willReturn($docMock);

        $this->renderer->renderFit(0, 600);
    }

    /**
     * Test renderFitWithInvalidHeightThrows
     */
    public function testRenderFitInvalidHeightThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);

        $docMock = $this->getMockBuilder(\PdfOxide\PdfDocument::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->pageMock->method('getDocument')->willReturn($docMock);

        $this->renderer->renderFit(800, -1);
    }

    /**
     * Test thumbnailWithInvalidSizeThrows
     */
    public function testThumbnailInvalidSizeThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Thumbnail size must be positive');

        $docMock = $this->getMockBuilder(\PdfOxide\PdfDocument::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->pageMock->method('getDocument')->willReturn($docMock);

        $this->renderer->thumbnail(0);
    }

    /**
     * Test renderToFileWithInvalidPathThrows
     */
    public function testRenderToFileEmptyPathThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('File path cannot be empty');

        $this->renderer->renderToFile('');
    }

    /**
     * Test renderToFileWithNonexistentDirectoryThrows
     */
    public function testRenderToFileNonexistentDirThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Directory does not exist');

        $this->renderer->renderToFile('/nonexistent/path/file.png');
    }

    /**
     * Test renderToFileWithReadOnlyDirectoryThrows
     */
    public function testRenderToFileReadOnlyDirThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Directory is not writable');

        $nonWritableDir = '/root/.ssh/';  // Typically not writable
        if (is_writable($nonWritableDir)) {
            $this->markTestSkipped('Test requires a non-writable directory');
        }

        $this->renderer->renderToFile($nonWritableDir . 'test.png');
    }

    /**
     * Test DPI range with all common values
     */
    public function testCommonDpiValues(): void
    {
        $commonValues = [72, 96, 150, 200, 300, 600];

        foreach ($commonValues as $dpi) {
            $this->renderer->setDpi($dpi);
            $this->assertEquals($dpi, $this->renderer->getDpi());
        }
    }

    /**
     * Test quality range with common values
     */
    public function testCommonQualityValues(): void
    {
        $commonValues = [50, 75, 85, 90, 95];

        foreach ($commonValues as $quality) {
            $this->renderer->setQuality($quality);
            $this->assertEquals($quality, $this->renderer->getQuality());
        }
    }

    /**
     * Test all format enum values
     */
    public function testAllFormatEnumValues(): void
    {
        foreach (ImageFormat::cases() as $format) {
            $this->renderer->setFormat($format);
            $this->assertSame($format, $this->renderer->getFormat());
        }
    }
}
