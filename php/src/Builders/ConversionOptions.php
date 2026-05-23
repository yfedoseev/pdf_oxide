<?php

declare(strict_types=1);

namespace PdfOxide\Builders;

/**
 * Options for PDF format conversion operations.
 *
 * Provides fluent interface for configuring conversion behavior.
 */
class ConversionOptions
{
    private bool $preserveLayout = true;
    private bool $detectHeadings = true;
    private bool $detectTables = false;
    private bool $detectColumns = false;
    private int $maxWidth = 0; // 0 = no limit
    private string $imageFormat = 'png';
    private int $imageQuality = 85;
    private bool $includeImages = true;
    private bool $includeText = true;
    private string $outputFormat = 'markdown'; // markdown, html, plain

    /**
     * Set whether to preserve original layout.
     */
    public function preserveLayout(bool $preserve): self
    {
        $this->preserveLayout = $preserve;
        return $this;
    }

    /**
     * Set whether to detect and format headings.
     */
    public function detectHeadings(bool $detect): self
    {
        $this->detectHeadings = $detect;
        return $this;
    }

    /**
     * Set whether to detect and format tables.
     */
    public function detectTables(bool $detect): self
    {
        $this->detectTables = $detect;
        return $this;
    }

    /**
     * Set whether to detect columns.
     */
    public function detectColumns(bool $detect): self
    {
        $this->detectColumns = $detect;
        return $this;
    }

    /**
     * Set maximum width for wrapped text.
     */
    public function maxWidth(int $width): self
    {
        $this->maxWidth = max(0, $width);
        return $this;
    }

    /**
     * Set image export format.
     */
    public function imageFormat(string $format): self
    {
        $this->imageFormat = strtolower($format);
        return $this;
    }

    /**
     * Set image quality (0-100).
     */
    public function imageQuality(int $quality): self
    {
        $this->imageQuality = max(0, min(100, $quality));
        return $this;
    }

    /**
     * Set whether to include images.
     */
    public function includeImages(bool $include): self
    {
        $this->includeImages = $include;
        return $this;
    }

    /**
     * Set whether to include text.
     */
    public function includeText(bool $include): self
    {
        $this->includeText = $include;
        return $this;
    }

    /**
     * Set output format (markdown, html, plain).
     */
    public function outputFormat(string $format): self
    {
        $this->outputFormat = strtolower($format);
        return $this;
    }

    // Getters
    public function isPreservingLayout(): bool { return $this->preserveLayout; }
    public function isDetectingHeadings(): bool { return $this->detectHeadings; }
    public function isDetectingTables(): bool { return $this->detectTables; }
    public function isDetectingColumns(): bool { return $this->detectColumns; }
    public function getMaxWidth(): int { return $this->maxWidth; }
    public function getImageFormat(): string { return $this->imageFormat; }
    public function getImageQuality(): int { return $this->imageQuality; }
    public function isIncludingImages(): bool { return $this->includeImages; }
    public function isIncludingText(): bool { return $this->includeText; }
    public function getOutputFormat(): string { return $this->outputFormat; }

    /**
     * Convert to array for FFI calls.
     */
    public function toArray(): array
    {
        return [
            'preserve_layout' => $this->preserveLayout,
            'detect_headings' => $this->detectHeadings,
            'detect_tables' => $this->detectTables,
            'detect_columns' => $this->detectColumns,
            'max_width' => $this->maxWidth,
            'image_format' => $this->imageFormat,
            'image_quality' => $this->imageQuality,
            'include_images' => $this->includeImages,
            'include_text' => $this->includeText,
            'output_format' => $this->outputFormat,
        ];
    }

    /**
     * Create from array.
     */
    public static function fromArray(array $options): self
    {
        $opts = new self();

        if (isset($options['preserve_layout'])) {
            $opts->preserveLayout($options['preserve_layout']);
        }
        if (isset($options['detect_headings'])) {
            $opts->detectHeadings($options['detect_headings']);
        }
        if (isset($options['detect_tables'])) {
            $opts->detectTables($options['detect_tables']);
        }
        if (isset($options['detect_columns'])) {
            $opts->detectColumns($options['detect_columns']);
        }
        if (isset($options['max_width'])) {
            $opts->maxWidth($options['max_width']);
        }
        if (isset($options['image_format'])) {
            $opts->imageFormat($options['image_format']);
        }
        if (isset($options['image_quality'])) {
            $opts->imageQuality($options['image_quality']);
        }
        if (isset($options['include_images'])) {
            $opts->includeImages($options['include_images']);
        }
        if (isset($options['include_text'])) {
            $opts->includeText($options['include_text']);
        }
        if (isset($options['output_format'])) {
            $opts->outputFormat($options['output_format']);
        }

        return $opts;
    }
}
