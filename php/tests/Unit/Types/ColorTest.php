<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Types;

use PHPUnit\Framework\TestCase;
use PdfOxide\Types\Color;

/**
 * Tests for Color type
 */
class ColorTest extends TestCase
{
    public function testColorCreation(): void
    {
        $color = new Color(255, 128, 0);
        $this->assertEquals(255, $color->red);
        $this->assertEquals(128, $color->green);
        $this->assertEquals(0, $color->blue);
        $this->assertEquals(255, $color->alpha);
    }

    public function testColorWithAlpha(): void
    {
        $color = new Color(255, 128, 0, 128);
        $this->assertEquals(128, $color->alpha);
    }

    public function testInvalidRed(): void
    {
        $this->expectException(\ValueError::class);
        new Color(256, 128, 0);
    }

    public function testInvalidGreen(): void
    {
        $this->expectException(\ValueError::class);
        new Color(255, -1, 0);
    }

    public function testFromHex(): void
    {
        $color = Color::fromHex('#FF8000');
        $this->assertEquals(255, $color->red);
        $this->assertEquals(128, $color->green);
        $this->assertEquals(0, $color->blue);
    }

    public function testFromHexWithoutHash(): void
    {
        $color = Color::fromHex('FF8000');
        $this->assertEquals(255, $color->red);
        $this->assertEquals(128, $color->green);
        $this->assertEquals(0, $color->blue);
    }

    public function testFromHexWithAlpha(): void
    {
        $color = Color::fromHex('#FF800080');
        $this->assertEquals(255, $color->red);
        $this->assertEquals(128, $color->green);
        $this->assertEquals(0, $color->blue);
        $this->assertEquals(128, $color->alpha);
    }

    public function testToHex(): void
    {
        $color = new Color(255, 128, 0);
        $this->assertEquals('#FF8000', $color->toHex());
        $this->assertEquals('#FF8000FF', $color->toHex(includeAlpha: true));
    }

    public function testToArgb(): void
    {
        $color = new Color(255, 128, 0, 200);
        $argb = $color->toArgb();
        // ARGB format: AA RR GG BB
        $this->assertIsInt($argb);
    }

    public function testToRgba(): void
    {
        $color = new Color(255, 128, 0, 200);
        $rgba = $color->toRgba();
        $this->assertIsInt($rgba);
    }

    public function testCommonColors(): void
    {
        $black = Color::black();
        $this->assertEquals(0, $black->red);
        $this->assertEquals(0, $black->green);
        $this->assertEquals(0, $black->blue);

        $white = Color::white();
        $this->assertEquals(255, $white->red);
        $this->assertEquals(255, $white->green);
        $this->assertEquals(255, $white->blue);

        $red = Color::red();
        $this->assertEquals(255, $red->red);
        $this->assertEquals(0, $red->green);
        $this->assertEquals(0, $red->blue);
    }

    public function testToArray(): void
    {
        $color = new Color(255, 128, 0, 200);
        $array = $color->toArray();

        $this->assertIsArray($array);
        $this->assertEquals(255, $array['red']);
        $this->assertEquals(128, $array['green']);
        $this->assertEquals(0, $array['blue']);
        $this->assertEquals(200, $array['alpha']);
    }
}
