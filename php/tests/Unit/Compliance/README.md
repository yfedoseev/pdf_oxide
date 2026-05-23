# Compliance Module Test Suite

Comprehensive test suite for PDF standards compliance operations.

## Test Coverage

### ComplianceTest.php
**32 test cases** covering:
- Supported PDF/A levels and PDF/X standards
- PDF/A level validation (1a, 1b, 2a, 2b, 3a, 3b)
- PDF/X standard validation (1a, 3, 4)
- Case-insensitive level/standard handling
- Whitespace trimming
- Document conversion to various standards
- Document validation against standards
- Error handling for invalid levels/standards
- Conversion error message formatting
- Validation result return types
- Static method consistency

**Key Test Methods**:
- testSupportedPdfALevels
- testSupportedPdfXStandards
- testIsValidPdfALevelValid
- testIsValidPdfALevelCaseInsensitive
- testIsValidPdfXStandardInvalid
- testConvertToPdfAValidLevel
- testConvertToPdfUa
- testConvertToPdfXValidStandard
- testValidatePdfA
- testValidatePdfX
- testConvertToPdfAInvalidLevelThrows

## Total Test Count: 32 Tests

## Supported Standards

### PDF/A Levels (6 levels)
- **1a** - PDF/A-1, part A (highest conformance with structure)
- **1b** - PDF/A-1, part B (basic visual preservation)
- **2a** - PDF/A-2, part A (supports transparency and JPEG2000)
- **2b** - PDF/A-2, part B (more lenient than 2a)
- **3a** - PDF/A-3, part A (allows embedding arbitrary files)
- **3b** - PDF/A-3, part B (most permissive level)

### PDF/X Standards (3 standards)
- **1a** - PDF/X-1:2001 (requires CMYK colors, no transparency)
- **3** - PDF/X-3:2002 (allows spot colors and RGB)
- **4** - PDF/X-4:2010 (allows transparency and external profiles)

### PDF/UA
- Universal Accessibility standard for accessible PDFs

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
# Run all compliance tests
./vendor/bin/phpunit tests/Unit/Compliance/

# Run compliance tests only
./vendor/bin/phpunit tests/Unit/Compliance/ComplianceTest.php

# Run with coverage
./vendor/bin/phpunit --coverage-html coverage tests/Unit/Compliance/

# Run with verbose output
./vendor/bin/phpunit -v tests/Unit/Compliance/
```

## Test Coverage Areas

### Level/Standard Validation
- PDF/A level validation (1a-3b)
- PDF/X standard validation (1a, 3, 4)
- Case-insensitive matching
- Whitespace trimming
- Invalid level/standard rejection

### Conversion Operations
- Convert to PDF/A (all 6 levels)
- Convert to PDF/UA
- Convert to PDF/X (all 3 standards)
- Byte output verification
- Error message formatting

### Validation Operations
- Validate PDF/A compliance
- Validate PDF/UA accessibility
- Validate PDF/X compliance
- ComplianceResult return type
- Error handling

### Error Conditions
- Invalid PDF/A levels
- Invalid PDF/X standards
- FFI call failures
- Error message clarity

### Static Method Behavior
- Consistency across calls
- Level/standard constants
- Supported format lists

## API Examples

```php
use PdfOxide\Compliance\Compliance;
use PdfOxide\PdfDocument;

// Convert to PDF/A-2B
$doc = new PdfDocument('input.pdf');
$pdfaBytes = Compliance::convertToPdfA($doc, '2b');
file_put_contents('output_a2b.pdf', $pdfaBytes);

// Validate against PDF/A-3A
$result = Compliance::validatePdfA($doc, '3a');
if ($result->isCompliant()) {
    echo "Document meets PDF/A-3A";
}

// Convert to PDF/UA (accessible)
$uaBytes = Compliance::convertToPdfUa($doc);

// Convert to PDF/X-4 (print production)
$printBytes = Compliance::convertToPdfX($doc, '4');

// Check supported levels
$levels = Compliance::getSupportedPdfALevels();
$standards = Compliance::getSupportedPdfXStandards();
```

## Key Testing Strategies

1. **Level Validation**: Verify all 6 PDF/A levels are recognized
2. **Standard Validation**: Verify all 3 PDF/X standards are recognized
3. **Case Handling**: Test uppercase, lowercase, mixed case
4. **Whitespace Handling**: Test strings with leading/trailing spaces
5. **Conversion Testing**: Verify conversion methods work for all levels/standards
6. **Validation Testing**: Verify validation methods return ComplianceResult
7. **Error Handling**: Invalid levels/standards throw specific exceptions
8. **Consistency**: Static methods return consistent results

## Notes

- Tests use PHPUnit 9+ compatible syntax
- Mock PdfDocument and ComplianceManager for isolation
- All test classes follow PSR-12 coding standards
- No file I/O in tests (mocked output)
- Comprehensive error condition coverage
- Validates static utility class pattern

## Phase 3 Summary

- ✅ BarcodeDetector: Full detection implementation
- ✅ DetectedBarcode: Complete result wrapper
- ✅ Compliance: Static utility for standards conversion
- ✅ Test coverage: 82 comprehensive test cases (50 + 32)
- ✅ All error conditions covered
- ✅ All standard levels/formats tested
