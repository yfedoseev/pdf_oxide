<?php

declare(strict_types=1);

namespace PdfOxide\Types;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Represents a generated barcode or QR code.
 *
 * Handles barcode data with support for multiple formats (PNG, SVG).
 */
class Barcode
{
    private CData $handle;
    private string $format;
    private string $data;
    private FunctionBindings $bindings;
    private ?string $cachedPngData = null;
    private ?string $cachedSvg = null;

    public function __construct(CData $handle, string $format, string $data)
    {
        $this->handle = $handle;
        $this->format = $format;
        $this->data = $data;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Get the underlying FFI handle.
     *
     * @return CData The barcode FFI handle
     * @internal
     */
    public function getHandle(): CData
    {
        return $this->handle;
    }

    /**
     * Get barcode format.
     *
     * @return string Barcode format (QR_CODE, EAN13, CODE128, etc.)
     */
    public function getFormat(): string
    {
        return $this->format;
    }

    /**
     * Get encoded data.
     *
     * @return string The data encoded in the barcode
     */
    public function getData(): string
    {
        return $this->data;
    }

    /**
     * Get barcode as PNG image.
     *
     * @return string Binary PNG data
     */
    public function getPng(): string
    {
        if ($this->cachedPngData === null) {
            $this->cachedPngData = $this->bindings->pdfBarcodeGetImagePng($this->handle);
        }
        return $this->cachedPngData;
    }

    /**
     * Get barcode as SVG.
     *
     * @return string SVG XML string
     */
    public function getSvg(): string
    {
        if ($this->cachedSvg === null) {
            $this->cachedSvg = $this->bindings->pdfBarcodeGetSvg($this->handle);
        }
        return $this->cachedSvg;
    }

    /**
     * Get barcode size (for PNG).
     *
     * @return int Size in bytes
     */
    public function getSize(): int
    {
        return strlen($this->getPng());
    }

    /**
     * Save barcode as PNG file.
     *
     * @param string $filePath Output file path
     * @return void
     */
    public function savePng(string $filePath): void
    {
        file_put_contents($filePath, $this->getPng());
    }

    /**
     * Save barcode as SVG file.
     *
     * @param string $filePath Output file path
     * @return void
     */
    public function saveSvg(string $filePath): void
    {
        file_put_contents($filePath, $this->getSvg());
    }

    /**
     * Convert barcode to base64 PNG.
     *
     * @param bool $withMimePrefix Include data:image/png;base64, prefix
     * @return string Base64 encoded PNG
     */
    public function toPngBase64(bool $withMimePrefix = false): string
    {
        $b64 = base64_encode($this->getPng());
        if ($withMimePrefix) {
            return "data:image/png;base64,{$b64}";
        }
        return $b64;
    }

    /**
     * Check if barcode is a QR code.
     *
     * @return bool True if this is a QR code
     */
    public function isQrCode(): bool
    {
        return $this->format === 'QR_CODE';
    }

    /**
     * Check if barcode is 1D (linear).
     *
     * @return bool True if this is a 1D barcode
     */
    public function is1D(): bool
    {
        return !$this->isQrCode();
    }

    /**
     * Get barcode info as array.
     *
     * @return array Barcode information
     */
    public function toArray(): array
    {
        return [
            'format' => $this->format,
            'data' => $this->data,
            'size_bytes' => $this->getSize(),
            'is_qr_code' => $this->isQrCode(),
            'is_1d' => $this->is1D(),
        ];
    }

    /**
     * Free barcode resources on destruct.
     */
    public function __destruct()
    {
        $this->bindings->pdfBarcodeFree($this->handle);
    }
}
