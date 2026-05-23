<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Builders;

use PHPUnit\Framework\TestCase;
use PdfOxide\Builders\ConversionOptions;

/**
 * Tests for ConversionOptions builder
 */
class ConversionOptionsTest extends TestCase
{
    public function testDefaults(): void
    {
        $options = new ConversionOptions();

        $this->assertTrue($options->isPreservingLayout());
        $this->assertTrue($options->isDetectingHeadings());
        $this->assertFalse($options->isDetectingTables());
        $this->assertEquals('markdown', $options->getOutputFormat());
        $this->assertEquals(85, $options->getImageQuality());
    }

    public function testFluentInterface(): void
    {
        $options = (new ConversionOptions())
            ->preserveLayout(false)
            ->detectTables(true)
            ->imageQuality(90)
            ->outputFormat('html');

        $this->assertFalse($options->isPreservingLayout());
        $this->assertTrue($options->isDetectingTables());
        $this->assertEquals(90, $options->getImageQuality());
        $this->assertEquals('html', $options->getOutputFormat());
    }

    public function testToArray(): void
    {
        $options = new ConversionOptions();
        $array = $options->toArray();

        $this->assertIsArray($array);
        $this->assertArrayHasKey('preserve_layout', $array);
        $this->assertArrayHasKey('detect_headings', $array);
        $this->assertArrayHasKey('image_format', $array);
        $this->assertArrayHasKey('output_format', $array);
    }

    public function testFromArray(): void
    {
        $array = [
            'preserve_layout' => false,
            'detect_tables' => true,
            'image_quality' => 95,
            'output_format' => 'plain',
        ];

        $options = ConversionOptions::fromArray($array);

        $this->assertFalse($options->isPreservingLayout());
        $this->assertTrue($options->isDetectingTables());
        $this->assertEquals(95, $options->getImageQuality());
        $this->assertEquals('plain', $options->getOutputFormat());
    }

    public function testImageQualityBounds(): void
    {
        $options = (new ConversionOptions())
            ->imageQuality(-50);

        $this->assertEquals(0, $options->getImageQuality());

        $options->imageQuality(150);
        $this->assertEquals(100, $options->getImageQuality());
    }
}
