<?php

declare(strict_types=1);

namespace PdfOxide\Enums;

/**
 * PDF blend modes for compositing operations.
 */
enum BlendMode: string
{
    // Basic modes
    case NORMAL = 'Normal';
    case MULTIPLY = 'Multiply';
    case SCREEN = 'Screen';
    case OVERLAY = 'Overlay';

    // Darkening modes
    case DARKEN = 'Darken';
    case COLOR_DODGE = 'ColorDodge';
    case COLOR_BURN = 'ColorBurn';
    case HARD_LIGHT = 'HardLight';
    case SOFT_LIGHT = 'SoftLight';
    case DIFFERENCE = 'Difference';
    case EXCLUSION = 'Exclusion';

    // Lightening modes
    case LIGHTEN = 'Lighten';

    // Comparative modes
    case HUE = 'Hue';
    case SATURATION = 'Saturation';
    case COLOR = 'Color';
    case LUMINOSITY = 'Luminosity';

    /**
     * Get human-readable description.
     */
    public function description(): string
    {
        return match ($this) {
            self::NORMAL => 'Normal blending',
            self::MULTIPLY => 'Multiply (darkens)',
            self::SCREEN => 'Screen (lightens)',
            self::OVERLAY => 'Overlay',
            self::DARKEN => 'Darken',
            self::COLOR_DODGE => 'Color Dodge (brightens)',
            self::COLOR_BURN => 'Color Burn (darkens)',
            self::HARD_LIGHT => 'Hard Light',
            self::SOFT_LIGHT => 'Soft Light',
            self::DIFFERENCE => 'Difference',
            self::EXCLUSION => 'Exclusion',
            self::LIGHTEN => 'Lighten',
            self::HUE => 'Hue',
            self::SATURATION => 'Saturation',
            self::COLOR => 'Color',
            self::LUMINOSITY => 'Luminosity',
        };
    }

    /**
     * Check if blend mode darkens the image.
     */
    public function darkens(): bool
    {
        return in_array($this, [
            self::MULTIPLY, self::DARKEN, self::COLOR_BURN, self::HARD_LIGHT,
        ]);
    }

    /**
     * Check if blend mode lightens the image.
     */
    public function lightens(): bool
    {
        return in_array($this, [
            self::SCREEN, self::LIGHTEN, self::COLOR_DODGE,
        ]);
    }
}
