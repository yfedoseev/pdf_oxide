# Barcodes Module Test Suite

Comprehensive test suite for barcode detection functionality.

## Test Coverage

### BarcodeDetectorTest.php
**24 test cases** covering:
- Detector initialization and default settings
- Confidence threshold configuration (0.0-1.0 range)
- Try harder mode enable/disable
- Fluent interface chaining
- Bounding box validation (required keys, positive dimensions)
- Configuration persistence
- Multiple detector independence
- All supported barcode formats

**Key Test Methods**:
- testDefaultConfidenceThreshold
- testSetConfidenceThresholdValid
- testSetConfidenceThresholdTooLowThrows
- testDetectInRegionMissingXThrows
- testFluentInterfaceChaining
- testGetSupportedFormats
- testDetectMethodSignature

### DetectedBarcodeTest.php
**26 test cases** covering:
- Barcode creation and data extraction
- Format detection (QR_CODE, CODE128, EAN_13, etc.)
- Data retrieval and special character handling
- Bounding box retrieval and validation
- Confidence score handling
- Barcode type classification (isQrCode, is1D, is2D)
- Array serialization
- Immutability verification
- FFI handle cleanup

**Key Test Methods**:
- testGetFormat
- testGetData
- testGetBbox
- testGetConfidence
- testIsQrCodeTrue
- testIs1DTrue
- testIs2DTrue
- testToArray
- testBarcodeImmutable
- testHandleFreedAfterConstruction

## Total Test Count: 50 Tests

## Supported Barcode Formats

The barcode detector supports the following formats:
- **QR_CODE** - 2D matrix barcode for URLs and data
- **CODE128** - 1D linear barcode for alphanumeric data
- **CODE39** - 1D linear barcode (alphanumeric)
- **EAN_13** - 1D linear barcode for products (13 digits)
- **EAN_8** - 1D linear barcode for products (8 digits)
- **UPC_A** - 1D linear barcode for products (12 digits)
- **PDF417** - 2D linear barcode for data storage
- **DATA_MATRIX** - 2D matrix barcode for small data

## Running the Tests

### Prerequisites
- PHP 8.1+
- PHPUnit 9+
- Composer dependencies installed

### Installation
```bash
cd /path/to/pdf_oxide/php
composer install
composer require --dev phpunit/phpunit
```

### Execution
```bash
# Run all barcode tests
./vendor/bin/phpunit tests/Unit/Barcodes/

# Run specific test file
./vendor/bin/phpunit tests/Unit/Barcodes/BarcodeDetectorTest.php

# Run with coverage
./vendor/bin/phpunit --coverage-html coverage tests/Unit/Barcodes/

# Run with verbose output
./vendor/bin/phpunit -v tests/Unit/Barcodes/
```

## Test Coverage Areas

### Detector Configuration
- Confidence threshold (0.0-1.0 with validation)
- Try harder mode for challenging detection
- Fluent interface for method chaining
- Configuration persistence across operations

### Barcode Detection
- Full page scanning
- Region-based detection with bbox validation
- Support for all 8 barcode formats
- Confidence scoring (0.0-1.0)

### Barcode Classification
- QR Code detection
- 1D barcode detection (linear codes)
- 2D barcode detection (matrix codes)

### Error Handling
- Invalid confidence thresholds
- Invalid bounding box dimensions
- Missing bbox keys
- FFI call failures

### Data Integrity
- Barcode format preservation
- Data integrity through immutability
- Multiple barcode independence
- Type classification accuracy

## Key Testing Strategies

1. **Configuration Testing**: Verify threshold ranges and mode settings
2. **Boundary Testing**: Test edge values (0.0, 1.0, positive/negative dimensions)
3. **Format Testing**: Verify all barcode formats are supported
4. **Type Classification**: Test QR/1D/2D detection logic
5. **Error Path Testing**: Invalid inputs should throw appropriate exceptions
6. **Immutability Testing**: Ensure barcodes cannot be modified after creation
7. **Integration Testing**: Multiple detectors work independently

## Notes

- Tests use PHPUnit 9+ compatible syntax
- Mock objects for FFI bindings
- All test classes follow PSR-12 coding standards
- Isolated unit tests that don't require real PDFs
- Comprehensive error condition coverage
