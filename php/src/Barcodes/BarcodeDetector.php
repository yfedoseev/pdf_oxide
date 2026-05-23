<?php

declare(strict_types=1);

namespace PdfOxide\Barcodes;

use PdfOxide\PdfPage;
use PdfOxide\FFI\FunctionBindings;
use FFI\CData;

/**
 * Detector for barcodes and QR codes in PDF pages.
 *
 * Scans PDF pages to detect and decode various barcode formats including
 * QR codes, 1D codes (Code128, EAN-13, UPC-A, etc.), and 2D codes (PDF417, DataMatrix).
 *
 * Example:
 *     $detector = new BarcodeDetector();
 *     $barcodes = $detector->detect($page);
 *     foreach ($barcodes as $barcode) {
 *         echo "Format: {$barcode->getFormat()}";
 *         echo "Data: {$barcode->getData()}";
 *         echo "Confidence: {$barcode->getConfidence()}";
 *     }
 *
 *     // Detect only high-confidence barcodes
 *     $barcodes = $detector->setConfidenceThreshold(0.9)->detect($page);
 *
 *     // Detect in specific region
 *     $bbox = ['x' => 50, 'y' => 100, 'width' => 200, 'height' => 150];
 *     $barcodes = $detector->detectInRegion($page, $bbox);
 *
 * @since 0.4.0
 */
class BarcodeDetector
{
    private float $confidenceThreshold = 0.5;
    private bool $tryHarder = false;
    private FunctionBindings $bindings;

    /**
     * Supported barcode formats.
     */
    public const FORMATS = [
        'QR_CODE',
        'CODE128',
        'CODE39',
        'EAN_13',
        'EAN_8',
        'UPC_A',
        'PDF417',
        'DATA_MATRIX',
    ];

    /**
     * Create a barcode detector instance.
     */
    public function __construct()
    {
        $this->bindings = new FunctionBindings();
    }

    /**
     * Set confidence threshold for barcode detection.
     *
     * @param float $threshold Confidence threshold (0.0-1.0)
     * @return self Fluent interface
     * @throws \InvalidArgumentException If threshold is invalid
     *
     * Barcodes with confidence below this threshold are excluded from results.
     * Valid range: 0.0 (accept all) to 1.0 (require perfect detection)
     * Default: 0.5
     *
     * Example:
     *     $detector->setConfidenceThreshold(0.8);  // Only high-confidence results
     */
    public function setConfidenceThreshold(float $threshold): self
    {
        if ($threshold < 0.0 || $threshold > 1.0) {
            throw new \InvalidArgumentException(
                "Confidence threshold must be 0.0-1.0, got {$threshold}"
            );
        }
        $this->confidenceThreshold = $threshold;
        return $this;
    }

    /**
     * Set "try harder" mode for detection.
     *
     * @param bool $enabled Enable or disable harder detection
     * @return self Fluent interface
     *
     * When enabled, uses more intensive algorithms for detection at the cost
     * of increased processing time. Useful for challenging conditions or rotated barcodes.
     *
     * Example:
     *     $detector->setTryHarder(true);  // Better detection, slower processing
     */
    public function setTryHarder(bool $enabled): self
    {
        $this->tryHarder = $enabled;
        return $this;
    }

    /**
     * Detect barcodes in a PDF page.
     *
     * Scans the entire page for barcodes and returns all detected results
     * that meet the configured confidence threshold.
     *
     * @param PdfPage $page Page to scan for barcodes
     * @return DetectedBarcode[] Array of detected barcodes
     * @throws \RuntimeException If detection fails
     *
     * Example:
     *     $barcodes = $detector->detect($page);
     *     echo "Found " . count($barcodes) . " barcodes";
     */
    public function detect(PdfPage $page): array
    {
        try {
            // Create configuration handle
            $config = $this->createDetectorConfig();

            // Detect on page
            $results = $this->bindings->pdfDetectBarcodes(
                $page->getHandle(),
                $page->getIndex(),
                $config
            );

            // Extract barcodes from results
            $barcodes = $this->extractBarcodes($results);

            // Free results
            $this->bindings->pdfDetectionResultsFree($results);

            return $barcodes;
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Barcode detection failed: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Detect barcodes in a specific region of a page.
     *
     * Scans only the specified region for barcodes. Useful for detecting
     * barcodes in known locations on a page.
     *
     * @param PdfPage $page Page to scan
     * @param array $bbox Bounding box with keys: 'x', 'y', 'width', 'height'
     * @return DetectedBarcode[] Array of detected barcodes in region
     * @throws \InvalidArgumentException If bounding box is invalid
     * @throws \RuntimeException If detection fails
     *
     * Bounding box coordinates are in PDF points (72 points = 1 inch).
     *
     * Example:
     *     $bbox = ['x' => 50, 'y' => 100, 'width' => 200, 'height' => 150];
     *     $barcodes = $detector->detectInRegion($page, $bbox);
     */
    public function detectInRegion(PdfPage $page, array $bbox): array
    {
        $this->validateBbox($bbox);

        try {
            // Create configuration handle
            $config = $this->createDetectorConfig();

            // Detect in region
            $results = $this->bindings->pdfDetectBarcodesInRegion(
                $page->getHandle(),
                $page->getIndex(),
                $bbox['x'],
                $bbox['y'],
                $bbox['width'],
                $bbox['height'],
                $config
            );

            // Extract barcodes from results
            $barcodes = $this->extractBarcodes($results);

            // Free results
            $this->bindings->pdfDetectionResultsFree($results);

            return $barcodes;
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Barcode detection in region failed: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Get supported barcode formats.
     *
     * @return string[] Array of supported format names
     *
     * Example:
     *     $formats = BarcodeDetector::getSupportedFormats();
     *     // ['QR_CODE', 'CODE128', 'CODE39', 'EAN_13', 'EAN_8', 'UPC_A', 'PDF417', 'DATA_MATRIX']
     */
    public static function getSupportedFormats(): array
    {
        return self::FORMATS;
    }

    /**
     * Get current confidence threshold.
     *
     * @return float Confidence threshold (0.0-1.0)
     */
    public function getConfidenceThreshold(): float
    {
        return $this->confidenceThreshold;
    }

    /**
     * Check if try harder mode is enabled.
     *
     * @return bool True if try harder is enabled
     */
    public function isTryHarderEnabled(): bool
    {
        return $this->tryHarder;
    }

    /**
     * Create detector configuration handle.
     *
     * @return CData Configuration handle for FFI calls
     */
    private function createDetectorConfig(): CData
    {
        $config = $this->bindings->pdfBarcodeDetectorConfigCreate();

        // Set confidence threshold
        $this->bindings->pdfBarcodeDetectorConfigSetConfidenceThreshold(
            $config,
            $this->confidenceThreshold
        );

        // Set try harder flag
        if ($this->tryHarder) {
            $this->bindings->pdfBarcodeDetectorConfigSetTryHarder($config);
        }

        return $config;
    }

    /**
     * Extract detected barcodes from results.
     *
     * @param CData $results Detection results handle
     * @return DetectedBarcode[] Array of detected barcodes
     */
    private function extractBarcodes(CData $results): array
    {
        $barcodes = [];
        $count = $this->bindings->pdfDetectionResultsCount($results);

        for ($i = 0; $i < $count; $i++) {
            $barcodeHandle = $this->bindings->pdfDetectionResultsGetBarcode($results, $i);

            try {
                $barcode = new DetectedBarcode($barcodeHandle, $this->bindings);

                // Filter by confidence threshold
                if ($barcode->getConfidence() >= $this->confidenceThreshold) {
                    $barcodes[] = $barcode;
                }
            } catch (\Exception) {
                // Skip invalid barcodes
                continue;
            }
        }

        return $barcodes;
    }

    /**
     * Validate bounding box structure.
     *
     * @param array $bbox Bounding box to validate
     * @throws \InvalidArgumentException If bounding box is invalid
     */
    private function validateBbox(array $bbox): void
    {
        $required = ['x', 'y', 'width', 'height'];

        foreach ($required as $key) {
            if (!array_key_exists($key, $bbox)) {
                throw new \InvalidArgumentException(
                    "Bounding box missing required key: {$key}"
                );
            }
        }

        if ($bbox['width'] <= 0 || $bbox['height'] <= 0) {
            throw new \InvalidArgumentException(
                "Bounding box dimensions must be positive"
            );
        }
    }

    /**
     * Get detector configuration as array.
     *
     * @return array Configuration array
     */
    public function toArray(): array
    {
        return [
            'confidenceThreshold' => $this->confidenceThreshold,
            'tryHarder' => $this->tryHarder,
            'supportedFormats' => self::FORMATS,
        ];
    }
}
