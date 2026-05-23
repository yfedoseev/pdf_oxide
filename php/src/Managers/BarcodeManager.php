<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI;
use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\ErrorHandler;
use PdfOxide\FFI\StringMarshaller;

/**
 * Manages barcode and QR code operations.
 *
 * Handles barcode/QR code generation, detection, and embedding.
 * Supports multiple formats: QR, EAN, UPC, Code128, Code39, etc.
 * Uses PHP 8+ features for clean, type-safe implementation.
 */
class BarcodeManager
{
    private readonly FunctionBindings $bindings;
    private readonly CData $handle;
    private readonly FFI $ffi;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
        $this->ffi = NativeLibrary::getInstance();
    }

    // ==================== QR CODE GENERATION ====================

    /**
     * Generate a QR code.
     *
     * @param string $data Data to encode
     * @param QrErrorCorrection $errorCorrection Error correction level
     * @return GeneratedBarcode Generated barcode
     */
    public function generateQrCode(
        string $data,
        QrErrorCorrection $errorCorrection = QrErrorCorrection::MEDIUM
    ): GeneratedBarcode {
        $cData = StringMarshaller::toCString($data);
        $errorCode = FFI::new('int');

        try {
            $barcodeHandle = $this->ffi->pdf_generate_qr_code(
                $cData,
                $errorCorrection->value,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_generate_qr_code');

            return new GeneratedBarcode(
                $barcodeHandle,
                $this->ffi,
                BarcodeFormat::QR_CODE,
                $data
            );
        } finally {
            unset($cData);
        }
    }

    /**
     * Generate a QR code with custom options.
     *
     * @param string $data Data to encode
     * @param QrCodeOptions $options QR code options
     * @return GeneratedBarcode Generated barcode
     */
    public function generateQrCodeWithOptions(string $data, QrCodeOptions $options): GeneratedBarcode
    {
        return $this->generateQrCode($data, $options->errorCorrection);
    }

    // ==================== BARCODE GENERATION ====================

    /**
     * Generate a barcode.
     *
     * @param string $data Data to encode
     * @param BarcodeFormat $format Barcode format
     * @return GeneratedBarcode Generated barcode
     */
    public function generateBarcode(string $data, BarcodeFormat $format = BarcodeFormat::CODE128): GeneratedBarcode
    {
        $cData = StringMarshaller::toCString($data);
        $errorCode = FFI::new('int');

        try {
            $barcodeHandle = $this->ffi->pdf_generate_barcode(
                $format->value,
                $cData,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_generate_barcode');

            return new GeneratedBarcode($barcodeHandle, $this->ffi, $format, $data);
        } finally {
            unset($cData);
        }
    }

    /**
     * Generate an EAN-13 barcode.
     *
     * @param string $data 13-digit EAN number
     * @return GeneratedBarcode Generated barcode
     */
    public function generateEan13(string $data): GeneratedBarcode
    {
        return $this->generateBarcode($data, BarcodeFormat::EAN13);
    }

    /**
     * Generate an EAN-8 barcode.
     *
     * @param string $data 8-digit EAN number
     * @return GeneratedBarcode Generated barcode
     */
    public function generateEan8(string $data): GeneratedBarcode
    {
        return $this->generateBarcode($data, BarcodeFormat::EAN8);
    }

    /**
     * Generate a UPC-A barcode.
     *
     * @param string $data 12-digit UPC number
     * @return GeneratedBarcode Generated barcode
     */
    public function generateUpcA(string $data): GeneratedBarcode
    {
        return $this->generateBarcode($data, BarcodeFormat::UPC_A);
    }

    /**
     * Generate a UPC-E barcode.
     *
     * @param string $data UPC-E number
     * @return GeneratedBarcode Generated barcode
     */
    public function generateUpcE(string $data): GeneratedBarcode
    {
        return $this->generateBarcode($data, BarcodeFormat::UPC_E);
    }

    /**
     * Generate a Code128 barcode.
     *
     * @param string $data Data to encode
     * @return GeneratedBarcode Generated barcode
     */
    public function generateCode128(string $data): GeneratedBarcode
    {
        return $this->generateBarcode($data, BarcodeFormat::CODE128);
    }

    /**
     * Generate a Code39 barcode.
     *
     * @param string $data Data to encode
     * @return GeneratedBarcode Generated barcode
     */
    public function generateCode39(string $data): GeneratedBarcode
    {
        return $this->generateBarcode($data, BarcodeFormat::CODE39);
    }

    /**
     * Generate a Codabar barcode.
     *
     * @param string $data Data to encode
     * @return GeneratedBarcode Generated barcode
     */
    public function generateCodabar(string $data): GeneratedBarcode
    {
        return $this->generateBarcode($data, BarcodeFormat::CODABAR);
    }

    /**
     * Generate an ITF barcode.
     *
     * @param string $data Data to encode
     * @return GeneratedBarcode Generated barcode
     */
    public function generateItf(string $data): GeneratedBarcode
    {
        return $this->generateBarcode($data, BarcodeFormat::ITF);
    }

    // ==================== BARCODE DETECTION ====================

    /**
     * Detect barcodes on a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return BarcodeDetectionResult Detection results
     */
    public function detectOnPage(int $pageIndex): BarcodeDetectionResult
    {
        $errorCode = FFI::new('int');
        $outCount = FFI::new('int');

        $resultsPtr = $this->ffi->pdf_detect_barcodes(
            $this->handle,
            $pageIndex,
            FFI::addr($outCount),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_detect_barcodes', ['page' => $pageIndex]);

        return new BarcodeDetectionResult($resultsPtr, $this->ffi, (int)$outCount->cdata, $pageIndex);
    }

    /**
     * Detect barcodes in a region.
     *
     * @param int $pageIndex Zero-based page index
     * @param float $x Region X
     * @param float $y Region Y
     * @param float $width Region width
     * @param float $height Region height
     * @return BarcodeDetectionResult Detection results
     */
    public function detectInRegion(
        int $pageIndex,
        float $x,
        float $y,
        float $width,
        float $height
    ): BarcodeDetectionResult {
        $errorCode = FFI::new('int');
        $outCount = FFI::new('int');

        $resultsPtr = $this->ffi->pdf_detect_barcodes_in_region(
            $this->handle,
            $pageIndex,
            $x,
            $y,
            $width,
            $height,
            FFI::addr($outCount),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_detect_barcodes_in_region', ['page' => $pageIndex]);

        return new BarcodeDetectionResult($resultsPtr, $this->ffi, (int)$outCount->cdata, $pageIndex);
    }

    /**
     * Detect all barcodes in document.
     *
     * @return array<int, BarcodeDetectionResult> Results by page index
     */
    public function detectAll(): array
    {
        $results = [];
        $errorCode = FFI::new('int');
        $pageCount = $this->ffi->pdf_document_get_page_count($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_page_count');

        for ($i = 0; $i < $pageCount; $i++) {
            $result = $this->detectOnPage($i);
            if ($result->getCount() > 0) {
                $results[$i] = $result;
            }
        }

        return $results;
    }

    // ==================== BARCODE EMBEDDING ====================

    /**
     * Add barcode to page.
     *
     * @param int $pageIndex Zero-based page index
     * @param GeneratedBarcode $barcode Barcode to add
     * @param float $x X coordinate
     * @param float $y Y coordinate
     * @param float $width Barcode width
     * @param float $height Barcode height
     * @return bool True on success
     */
    public function addToPage(
        int $pageIndex,
        GeneratedBarcode $barcode,
        float $x,
        float $y,
        float $width,
        float $height
    ): bool {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_add_barcode_to_page(
            $this->handle,
            $pageIndex,
            $barcode->getHandle(),
            $x,
            $y,
            $width,
            $height,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_add_barcode_to_page', ['page' => $pageIndex]);
        return (bool)$result;
    }

    /**
     * Add QR code to page.
     *
     * @param int $pageIndex Zero-based page index
     * @param string $data QR code data
     * @param float $x X coordinate
     * @param float $y Y coordinate
     * @param float $size QR code size (width = height)
     * @param QrErrorCorrection $errorCorrection Error correction level
     * @return bool True on success
     */
    public function addQrCodeToPage(
        int $pageIndex,
        string $data,
        float $x,
        float $y,
        float $size,
        QrErrorCorrection $errorCorrection = QrErrorCorrection::MEDIUM
    ): bool {
        $qrCode = $this->generateQrCode($data, $errorCorrection);
        return $this->addToPage($pageIndex, $qrCode, $x, $y, $size, $size);
    }

    /**
     * Add barcode to all pages.
     *
     * @param GeneratedBarcode $barcode Barcode to add
     * @param float $x X coordinate
     * @param float $y Y coordinate
     * @param float $width Barcode width
     * @param float $height Barcode height
     * @return int Number of pages modified
     */
    public function addToAllPages(
        GeneratedBarcode $barcode,
        float $x,
        float $y,
        float $width,
        float $height
    ): int {
        $errorCode = FFI::new('int');
        $pageCount = $this->ffi->pdf_document_get_page_count($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_page_count');

        $modified = 0;
        for ($i = 0; $i < $pageCount; $i++) {
            if ($this->addToPage($i, $barcode, $x, $y, $width, $height)) {
                $modified++;
            }
        }

        return $modified;
    }

    // ==================== UTILITIES ====================

    /**
     * Get supported barcode formats.
     *
     * @return array<BarcodeFormat> Supported formats
     */
    public function getSupportedFormats(): array
    {
        return BarcodeFormat::cases();
    }

    /**
     * Get common barcode format names.
     *
     * @return string[] Available barcode format names
     */
    public static function getAvailableFormats(): array
    {
        return [
            'QR_CODE',
            'EAN13',
            'EAN8',
            'UPC_A',
            'UPC_E',
            'CODE128',
            'CODE39',
            'CODE93',
            'CODABAR',
            'ITF',
        ];
    }

    /**
     * Validate barcode data for format.
     *
     * @param string $data Data to validate
     * @param BarcodeFormat $format Target format
     * @return BarcodeValidationResult Validation result
     */
    public function validateData(string $data, BarcodeFormat $format): BarcodeValidationResult
    {
        $isValid = true;
        $error = null;

        switch ($format) {
            case BarcodeFormat::EAN13:
                if (!preg_match('/^\d{13}$/', $data)) {
                    $isValid = false;
                    $error = 'EAN-13 must be exactly 13 digits';
                }
                break;

            case BarcodeFormat::EAN8:
                if (!preg_match('/^\d{8}$/', $data)) {
                    $isValid = false;
                    $error = 'EAN-8 must be exactly 8 digits';
                }
                break;

            case BarcodeFormat::UPC_A:
                if (!preg_match('/^\d{12}$/', $data)) {
                    $isValid = false;
                    $error = 'UPC-A must be exactly 12 digits';
                }
                break;

            case BarcodeFormat::CODE39:
                if (!preg_match('/^[A-Z0-9\-\.\ \$\/\+\%]+$/', strtoupper($data))) {
                    $isValid = false;
                    $error = 'Code39 contains invalid characters';
                }
                break;

            case BarcodeFormat::QR_CODE:
                // QR codes can contain any data
                break;

            default:
                // Basic validation for other formats
                break;
        }

        return new BarcodeValidationResult($isValid, $error);
    }

    /**
     * Get summary information.
     *
     * @return array Summary
     */
    public function getSummary(): array
    {
        return [
            'supported_formats' => array_map(fn($f) => $f->name, $this->getSupportedFormats()),
            'capabilities' => [
                'generation' => true,
                'detection' => true,
                'embedding' => true,
                'qr_codes' => true,
                '1d_barcodes' => true,
            ],
        ];
    }
}

// ==================== SUPPORTING CLASSES ====================

/**
 * Barcode formats.
 */
enum BarcodeFormat: int
{
    case QR_CODE = 0;
    case EAN13 = 1;
    case EAN8 = 2;
    case UPC_A = 3;
    case UPC_E = 4;
    case CODE128 = 5;
    case CODE39 = 6;
    case CODABAR = 7;
    case ITF = 8;

    public function getDescription(): string
    {
        return match($this) {
            self::QR_CODE => 'QR Code (2D)',
            self::EAN13 => 'EAN-13 (European Article Number)',
            self::EAN8 => 'EAN-8 (Short EAN)',
            self::UPC_A => 'UPC-A (Universal Product Code)',
            self::UPC_E => 'UPC-E (Compact UPC)',
            self::CODE128 => 'Code 128 (Alphanumeric)',
            self::CODE39 => 'Code 39 (Code 3 of 9)',
            self::CODABAR => 'Codabar',
            self::ITF => 'ITF (Interleaved 2 of 5)',
        };
    }

    public function is2D(): bool
    {
        return $this === self::QR_CODE;
    }
}

/**
 * QR error correction levels.
 */
enum QrErrorCorrection: int
{
    case LOW = 0;       // 7% recovery
    case MEDIUM = 1;    // 15% recovery
    case QUARTILE = 2;  // 25% recovery
    case HIGH = 3;      // 30% recovery

    public function getRecoveryPercentage(): int
    {
        return match($this) {
            self::LOW => 7,
            self::MEDIUM => 15,
            self::QUARTILE => 25,
            self::HIGH => 30,
        };
    }

    public function getDescription(): string
    {
        return match($this) {
            self::LOW => 'Low (7% recovery)',
            self::MEDIUM => 'Medium (15% recovery)',
            self::QUARTILE => 'Quartile (25% recovery)',
            self::HIGH => 'High (30% recovery)',
        };
    }
}

/**
 * QR code options.
 */
readonly class QrCodeOptions
{
    public function __construct(
        public QrErrorCorrection $errorCorrection = QrErrorCorrection::MEDIUM,
        public int $size = 10,
        public string $foregroundColor = '#000000',
        public string $backgroundColor = '#FFFFFF'
    ) {}
}

/**
 * Generated barcode.
 */
class GeneratedBarcode
{
    private CData $handle;
    private FFI $ffi;
    private BarcodeFormat $format;
    private string $data;

    public function __construct(CData $handle, FFI $ffi, BarcodeFormat $format, string $data)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
        $this->format = $format;
        $this->data = $data;
    }

    public function getHandle(): CData
    {
        return $this->handle;
    }

    public function getFormat(): BarcodeFormat
    {
        return $this->format;
    }

    public function getData(): string
    {
        return $this->data;
    }

    /**
     * Get barcode as PNG image.
     *
     * @param int $size Size in pixels
     * @return string PNG binary data
     */
    public function toPng(int $size = 200): string
    {
        $outSize = FFI::new('size_t');
        $errorCode = FFI::new('int');

        $dataPtr = $this->ffi->pdf_barcode_get_image_png(
            $this->handle,
            $size,
            FFI::addr($outSize),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_barcode_get_image_png');

        $size = (int)$outSize->cdata;
        $data = FFI::string($dataPtr, $size);

        // Free native memory
        $this->ffi->free_bytes($dataPtr);

        return $data;
    }

    /**
     * Get barcode as SVG.
     *
     * @param int $size Size in pixels
     * @return string SVG XML
     */
    public function toSvg(int $size = 200): string
    {
        $errorCode = FFI::new('int');
        $svgPtr = $this->ffi->pdf_barcode_get_svg($this->handle, $size, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_barcode_get_svg');

        return StringMarshaller::fromCString($svgPtr);
    }

    /**
     * Save barcode to PNG file.
     *
     * @param string $filePath Output file path
     * @param int $size Size in pixels
     * @return bool True on success
     */
    public function saveToPng(string $filePath, int $size = 200): bool
    {
        $png = $this->toPng($size);
        return file_put_contents($filePath, $png) !== false;
    }

    /**
     * Save barcode to SVG file.
     *
     * @param string $filePath Output file path
     * @param int $size Size in pixels
     * @return bool True on success
     */
    public function saveToSvg(string $filePath, int $size = 200): bool
    {
        $svg = $this->toSvg($size);
        return file_put_contents($filePath, $svg) !== false;
    }

    /**
     * Get base64-encoded PNG.
     *
     * @param int $size Size in pixels
     * @param bool $withDataUri Include data URI prefix
     * @return string Base64 string
     */
    public function toBase64(int $size = 200, bool $withDataUri = true): string
    {
        $png = $this->toPng($size);
        $base64 = base64_encode($png);

        if ($withDataUri) {
            return 'data:image/png;base64,' . $base64;
        }

        return $base64;
    }

    public function __destruct()
    {
        $this->ffi->pdf_barcode_free($this->handle);
    }
}

/**
 * Detected barcode.
 */
readonly class DetectedBarcode
{
    public function __construct(
        public BarcodeFormat $format,
        public string $data,
        public float $x,
        public float $y,
        public float $width,
        public float $height,
        public float $confidence
    ) {}

    public function toArray(): array
    {
        return [
            'format' => $this->format->name,
            'data' => $this->data,
            'x' => $this->x,
            'y' => $this->y,
            'width' => $this->width,
            'height' => $this->height,
            'confidence' => $this->confidence,
        ];
    }
}

/**
 * Barcode detection result.
 */
class BarcodeDetectionResult
{
    private CData $handle;
    private FFI $ffi;
    private int $count;
    private int $pageIndex;
    private ?array $cachedBarcodes = null;

    public function __construct(CData $handle, FFI $ffi, int $count, int $pageIndex)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
        $this->count = $count;
        $this->pageIndex = $pageIndex;
    }

    public function getCount(): int
    {
        return $this->count;
    }

    public function getPageIndex(): int
    {
        return $this->pageIndex;
    }

    public function getBarcodes(): array
    {
        if ($this->cachedBarcodes !== null) {
            return $this->cachedBarcodes;
        }

        $this->cachedBarcodes = [];

        for ($i = 0; $i < $this->count; $i++) {
            $format = (int)$this->ffi->pdf_detected_barcode_get_format($this->handle, $i);
            $dataPtr = $this->ffi->pdf_detected_barcode_get_data($this->handle, $i);
            $data = StringMarshaller::fromCString($dataPtr, false);

            $x = (float)$this->ffi->pdf_detected_barcode_get_x($this->handle, $i);
            $y = (float)$this->ffi->pdf_detected_barcode_get_y($this->handle, $i);
            $width = (float)$this->ffi->pdf_detected_barcode_get_width($this->handle, $i);
            $height = (float)$this->ffi->pdf_detected_barcode_get_height($this->handle, $i);
            $confidence = (float)$this->ffi->pdf_detected_barcode_get_confidence($this->handle, $i);

            $this->cachedBarcodes[] = new DetectedBarcode(
                BarcodeFormat::from($format),
                $data,
                $x,
                $y,
                $width,
                $height,
                $confidence
            );
        }

        return $this->cachedBarcodes;
    }

    public function toArray(): array
    {
        return [
            'count' => $this->count,
            'page_index' => $this->pageIndex,
            'barcodes' => array_map(fn($b) => $b->toArray(), $this->getBarcodes()),
        ];
    }

    public function __destruct()
    {
        $this->ffi->pdf_detected_barcodes_free($this->handle);
    }
}

/**
 * Barcode validation result.
 */
readonly class BarcodeValidationResult
{
    public function __construct(
        public bool $isValid,
        public ?string $error = null
    ) {}

    public function toArray(): array
    {
        return [
            'is_valid' => $this->isValid,
            'error' => $this->error,
        ];
    }
}
