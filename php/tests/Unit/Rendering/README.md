# Rendering Module Test Suite

Comprehensive test suite for the PHP rendering module (PageRenderer, ImageFormat, RenderedImage classes).

## Test Coverage

### PageRendererTest.php
**42 test cases** covering:
- Renderer initialization and page reference
- Default configuration values (PNG format, 150 DPI, quality 85)
- Format configuration (PNG, JPEG, WebP)
- DPI configuration (1-600 range validation)
- Quality configuration (1-100 range validation)
- Antialiasing enable/disable
- Background color validation (hex format)
- Fluent interface chaining
- Rendering method signatures
- File I/O validation
- Error handling for invalid parameters

**Key Test Methods**:
- testRendererInitialization
- testDefaultFormatIsPng
- testSetDpiRange (minimum, maximum, typical values)
- testSetQualityRange
- testSetBackgroundColorValid
- testFluentInterfaceChaining
- testRenderFitInvalidDimensionsThrow
- testThumbnailInvalidSizeThrows
- testCommonDpiValues
- testAllFormatEnumValues

### ImageFormatTest.php
**39 test cases** covering:
- Enum value verification (PNG, JPEG, WebP)
- MIME type mapping
- File extension mapping (jpg alias for jpeg)
- Human-readable descriptions
- Lossless/lossy format detection
- Transparency support detection
- Default quality values
- Format string parsing (case-insensitive)
- Format validation
- Whitespace handling
- All enum cases accessibility
- Format characteristics (compression, transparency, quality)

**Key Test Methods**:
- testPngEnumCase
- testPngMimeType
- testJpegExtension
- testPngIsLossless
- testJpegNoTransparency
- testJpegDefaultQuality
- testTryFromStringWithVariations (uppercase, lowercase, whitespace)
- testIsValidWithValidFormats
- testAllCasesAccessible
- testMimeTypeFormat

### RenderedImageTest.php
**41 test cases** covering:
- Image creation with null handle
- Format retrieval
- Dimension handling (width, height, aspect ratio)
- Data retrieval and size calculation
- MIME type and extension mapping
- Base64 encoding (with and without MIME prefix)
- Data URL generation
- Format conversion
- Array serialization
- File operations
- Data caching and consistency
- Type validation

**Key Test Methods**:
- testRenderedImageCreation
- testGetFormatVariations
- testGetWidthWithNullHandle
- testGetAspectRatioZeroDimensions
- testToBase64WithMimePrefix
- testConvertFormatPngToJpeg
- testToArrayStructure
- testMimeTypeValidity
- testExtensionIsLowercase
- testSaveToFileEmptyData
- testMultipleFormatConversions

## Total Test Count: 122 Tests

## Test Organization

```
/php/tests/Unit/Rendering/
├── PageRendererTest.php       (42 tests)
├── ImageFormatTest.php         (39 tests)
├── RenderedImageTest.php       (41 tests)
└── README.md                   (this file)
```

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
# Run all rendering tests
./vendor/bin/phpunit tests/Unit/Rendering/

# Run specific test file
./vendor/bin/phpunit tests/Unit/Rendering/PageRendererTest.php

# Run with coverage
./vendor/bin/phpunit --coverage-html coverage tests/Unit/Rendering/

# Run with verbose output
./vendor/bin/phpunit -v tests/Unit/Rendering/
```

## Test Coverage Areas

### Configuration Validation
- DPI range (1-600 with validation)
- Quality range (1-100 with validation)
- Color format validation (hex codes)
- Format enum validation

### Format Support
- PNG: lossless, supports transparency, 0 quality
- JPEG: lossy, no transparency, default quality 85
- WebP: lossless-capable, supports transparency, default quality 80

### Error Handling
- Invalid DPI values (below 1, above 600)
- Invalid quality values (below 1, above 100)
- Invalid color format (non-hex, wrong length)
- Invalid file paths (non-existent directories, non-writable)
- Invalid render dimensions (zero or negative)

### Data Integrity
- Multiple renderer instances are independent
- Format conversions preserve image structure
- Fluent interface returns correct object
- MIME types match file formats
- Base64 encoding with/without prefix

### Enum Capabilities
- Case-insensitive format string parsing
- Whitespace trimming in format strings
- Alias support (jpg for jpeg)
- Format validation
- All enum cases accessible

## Key Testing Strategies

1. **Boundary Testing**: DPI (1, 600), Quality (1, 100), colors (#000000, #FFFFFF)
2. **Parameter Validation**: Verify all error conditions throw appropriate exceptions
3. **Type Consistency**: Ensure correct return types across all methods
4. **Format Coverage**: Test all three supported image formats
5. **Fluent Interface**: Verify method chaining works correctly
6. **Data Integrity**: Check that conversions preserve data structure
7. **Enum Parsing**: Test case-insensitive, whitespace-tolerant format parsing
8. **Error Paths**: Verify exceptions for invalid inputs

## Notes

- Tests use PHPUnit 9+ compatible syntax
- Mock objects are created where needed for dependencies
- All test classes follow PSR-12 coding standards
- Tests are isolated and can run in any order
- Fixtures are created dynamically in setUp methods
- Both unit and integration aspects covered

## Phase 2 Summary

- ✅ ImageFormat enum: 3 cases (PNG, JPEG, WebP) with full capabilities
- ✅ PageRenderer class: Fluent builder with validation
- ✅ RenderedImage class: Already implemented with full functionality
- ✅ Test coverage: 122 comprehensive test cases
- ✅ All error conditions covered
- ✅ All format variations tested
