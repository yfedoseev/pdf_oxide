<?php

declare(strict_types=1);

namespace PdfOxide\Barcodes;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * A detected barcode from PDF page scanning.
 *
 * Represents a single barcode detected by the BarcodeDetector,
 * including its format, decoded data, position, and confidence score.
 *
 * Example:
 *     $detector = new BarcodeDetector();
 *     $barcodes = $detector->detect($page);
 *     foreach ($barcodes as $barcode) {
 *         echo "Type: " . $barcode->getFormat();
 *         echo "Data: " . $barcode->getData();
 *         echo "Confidence: " . ($barcode->getConfidence() * 100) . "%";
 *         [$x, $y, $w, $h] = $barcode->getBbox();
 *         echo "Position: ({$x}, {$y}) Size: ({$w}x{$h})";
 *     }
 *
 * @since 0.4.0
 */
class DetectedBarcode
{
    private string $format;
    private string $data;
    private array $bbox;
    private float $confidence;

    /**
     * Create DetectedBarcode from FFI handle.
     *
     * Extracts all barcode data from the native handle immediately, then
     * frees the handle. Immutable after construction.
     *
     * @param CData $barcodeHandle FFI barcode handle
     * @param FunctionBindings $bindings Function bindings for FFI calls
     * @throws \RuntimeException If data extraction fails
     */
    public function __construct(CData $barcodeHandle, FunctionBindings $bindings)
    {
        try {
            // Extract all data from handle
            $this->format = $bindings->pdfDetectedBarcodeGetFormat($barcodeHandle);
            $this->data = $bindings->pdfDetectedBarcodeGetData($barcodeHandle);
            $this->bbox = $bindings->pdfDetectedBarcodeGetBbox($barcodeHandle);
            $this->confidence = $bindings->pdfDetectedBarcodeGetConfidence($barcodeHandle);

            // Free the handle
            $bindings->pdfDetectedBarcodeFree($barcodeHandle);
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to create DetectedBarcode from handle: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Get barcode format.
     *
     * @return string The barcode format (e.g., 'QR_CODE', 'CODE128', 'EAN_13')
     *
     * Example:
     *     $format = $barcode->getFormat();
     *     if ($format === 'QR_CODE') {
     *         echo "This is a QR code";
     *     }
     */
    public function getFormat(): string
    {
        return $this->format;
    }

    /**
     * Get decoded barcode data.
     *
     * @return string The decoded data from the barcode
     *
     * For text barcodes (QR codes, Code128, etc.), this is the text content.
     * For binary barcodes, this may contain raw bytes.
     *
     * Example:
     *     $data = $barcode->getData();
     *     echo "Barcode contains: " . $data;
     */
    public function getData(): string
    {
        return $this->data;
    }

    /**
     * Get bounding box of detected barcode.
     *
     * @return array Associative array with keys: 'x', 'y', 'width', 'height'
     *
     * Coordinates are in PDF points (72 points = 1 inch).
     *
     * Example:
     *     $bbox = $barcode->getBbox();
     *     echo "Position: ({$bbox['x']}, {$bbox['y']})";
     *     echo "Size: {$bbox['width']}x{$bbox['height']}";
     */
    public function getBbox(): array
    {
        return $this->bbox;
    }

    /**
     * Get detection confidence score.
     *
     * @return float Confidence score (0.0-1.0)
     *
     * 1.0 indicates high confidence in the detection,
     * while lower values indicate the detection may be ambiguous.
     *
     * Example:
     *     $confidence = $barcode->getConfidence();
     *     if ($confidence < 0.8) {
     *         echo "Low confidence detection, verify manually";
     *     }
     */
    public function getConfidence(): float
    {
        return $this->confidence;
    }

    /**
     * Check if barcode is a QR code.
     *
     * @return bool True if format is QR_CODE
     */
    public function isQrCode(): bool
    {
        return $this->format === 'QR_CODE';
    }

    /**
     * Check if barcode is 1D (linear).
     *
     * @return bool True if barcode is 1D format
     */
    public function is1D(): bool
    {
        $oneDFormats = ['CODE128', 'CODE39', 'EAN_13', 'EAN_8', 'UPC_A'];
        return in_array($this->format, $oneDFormats);
    }

    /**
     * Check if barcode is 2D (matrix).
     *
     * @return bool True if barcode is 2D format
     */
    public function is2D(): bool
    {
        $twoDFormats = ['QR_CODE', 'PDF417', 'DATA_MATRIX'];
        return in_array($this->format, $twoDFormats);
    }

    /**
     * Get barcode as array representation.
     *
     * @return array Associative array with barcode data
     *
     * Example:
     *     $data = $barcode->toArray();
     *     $json = json_encode($data);
     */
    public function toArray(): array
    {
        return [
            'format' => $this->format,
            'data' => $this->data,
            'bbox' => $this->bbox,
            'confidence' => $this->confidence,
            'is_qr_code' => $this->isQrCode(),
            'is_1d' => $this->is1D(),
            'is_2d' => $this->is2D(),
        ];
    }

    /**
     * String representation of detected barcode.
     *
     * @return string
     */
    public function __toString(): string
    {
        return "{$this->format}: {$this->data}";
    }
}
