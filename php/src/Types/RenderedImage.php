<?php

declare(strict_types=1);

namespace PdfOxide\Types;

use FFI\CData;
use PdfOxide\Builders\RenderingOptions;
use PdfOxide\FFI\FunctionBindings;

/**
 * Represents a rendered PDF page image.
 *
 * Handles the image data in memory with support for multiple formats.
 */
class RenderedImage
{
    private ?CData $handle;
    private RenderingOptions $options;
    private ?string $cachedData = null;
    private ?int $width = null;
    private ?int $height = null;
    private FunctionBindings $bindings;

    public function __construct(?CData $handle, RenderingOptions $options)
    {
        $this->handle = $handle;
        $this->options = $options;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Get image width in pixels.
     *
     * @return int Width
     */
    public function getWidth(): int
    {
        if ($this->width === null && $this->handle !== null) {
            $this->width = $this->bindings->pdfRenderedImageWidth($this->handle);
        }
        return $this->width ?? 0;
    }

    /**
     * Get image height in pixels.
     *
     * @return int Height
     */
    public function getHeight(): int
    {
        if ($this->height === null && $this->handle !== null) {
            $this->height = $this->bindings->pdfRenderedImageHeight($this->handle);
        }
        return $this->height ?? 0;
    }

    /**
     * Get image format.
     *
     * @return string Format (png, jpeg, webp)
     */
    public function getFormat(): string
    {
        return $this->options->getImageFormat();
    }

    /**
     * Get aspect ratio.
     *
     * @return float Aspect ratio (width/height)
     */
    public function getAspectRatio(): float
    {
        return $this->height > 0 ? $this->width / $this->height : 0;
    }

    /**
     * Get image data size in bytes.
     *
     * @return int Size
     */
    public function getSize(): int
    {
        if ($this->cachedData !== null) {
            return strlen($this->cachedData);
        }

        if ($this->handle !== null) {
            return $this->bindings->pdfRenderedImageSize($this->handle);
        }

        return 0;
    }

    /**
     * Get raw image data.
     *
     * @return string Binary image data
     */
    public function getData(): string
    {
        if ($this->cachedData !== null) {
            return $this->cachedData;
        }

        if ($this->handle !== null) {
            $this->cachedData = $this->bindings->pdfRenderedImageData($this->handle);
            return $this->cachedData;
        }

        return '';
    }

    /**
     * Save image to file.
     *
     * @param string $filePath Path where to save
     * @return void
     * @throws \PdfOxide\Exceptions\IoException on error
     */
    public function saveToFile(string $filePath): void
    {
        if ($this->handle !== null) {
            // Use FFI for better error handling
            $this->bindings->pdfRenderedImageSave($this->handle, $filePath);
        } else {
            // Fallback: write cached data
            file_put_contents($filePath, $this->getData());
        }
    }

    /**
     * Convert image to base64 string.
     *
     * @param bool $withMimePrefix Include data:image/... prefix
     * @return string Base64 encoded image
     */
    public function toBase64(bool $withMimePrefix = false): string
    {
        $data = $this->getData();
        $b64 = base64_encode($data);

        if ($withMimePrefix) {
            $mimeType = match ($this->getFormat()) {
                'png' => 'image/png',
                'jpeg' => 'image/jpeg',
                'webp' => 'image/webp',
                default => 'application/octet-stream',
            };
            return "data:{$mimeType};base64,{$b64}";
        }

        return $b64;
    }

    /**
     * Convert to different format.
     *
     * @param string $newFormat New image format (png, jpeg, webp)
     * @return RenderedImage Converted image
     */
    public function convertFormat(string $newFormat): RenderedImage
    {
        $newOptions = new RenderingOptions();
        $newOptions->imageFormat($newFormat);

        if ($this->handle !== null) {
            $newHandle = $this->bindings->pdfRenderedImageConvert($this->handle, $newFormat);
            return new RenderedImage($newHandle, $newOptions);
        }

        return new RenderedImage($this->handle, $newOptions);
    }

    /**
     * Get MIME type of image.
     *
     * @return string MIME type
     */
    public function getMimeType(): string
    {
        return match ($this->getFormat()) {
            'png' => 'image/png',
            'jpeg' => 'image/jpeg',
            'webp' => 'image/webp',
            default => 'application/octet-stream',
        };
    }

    /**
     * Get file extension for format.
     *
     * @return string Extension (without dot)
     */
    public function getExtension(): string
    {
        return match ($this->getFormat()) {
            'png' => 'png',
            'jpeg' => 'jpg',
            'webp' => 'webp',
            default => 'bin',
        };
    }

    /**
     * Check if image has valid data.
     *
     * @return bool True if image data exists
     */
    public function hasData(): bool
    {
        return $this->handle !== null || $this->cachedData !== null;
    }

    /**
     * Get image info as array.
     *
     * @return array Image information
     */
    public function toArray(): array
    {
        return [
            'width' => $this->getWidth(),
            'height' => $this->getHeight(),
            'format' => $this->getFormat(),
            'aspect_ratio' => $this->getAspectRatio(),
            'size_bytes' => $this->getSize(),
            'mime_type' => $this->getMimeType(),
            'extension' => $this->getExtension(),
        ];
    }

    /**
     * Free resources on destruct.
     */
    public function __destruct()
    {
        if ($this->handle !== null) {
            $this->bindings->pdfRenderedImageFree($this->handle);
        }
    }
}
