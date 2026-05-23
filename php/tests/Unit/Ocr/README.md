# OCR Module Test Suite

Comprehensive test suite for the PHP OCR module (OcrEngine, OcrSpan, OcrResult classes).

## Test Coverage

### OcrEngineTest.php
**27 test cases** covering:
- Engine initialization and lifecycle management
- Resource creation and cleanup
- Handle management and consistency
- Version and status retrieval
- Multiple engine instances
- Operations on closed engines (error handling)
- Destructor cleanup

**Key Test Methods**:
- testEngineInitialization
- testEngineClose
- testOperationOnClosedEngineThrows
- testGetEngineVersion
- testGetEngineStatus
- testEngineDestructorClosesResources
- testMultipleEnginesIndependent

### OcrSpanTest.php
**27 test cases** covering:
- Span creation and immutability
- Text extraction with special characters
- Bounding box retrieval and coordinate types
- Confidence scores (0.0-1.0)
- Per-character confidence handling
- Error conditions and bounds checking
- Array serialization
- FFI handle management and cleanup

**Key Test Methods**:
- testGetText (with special characters, empty string)
- testGetBbox (float coordinates, key validation)
- testGetConfidence (perfect, zero, range)
- testGetCharConfidences (available, not available)
- testGetCharConfidence (index access, bounds checking)
- testToString
- testSpanImmutable
- testHandleFreedAfterConstruction

### OcrResultTest.php
**33 test cases** covering:
- Result creation with and without spans
- Span access methods (getSpan, getSpans, getCount)
- Text aggregation with separators
- Confidence statistics (average, min, max)
- Filtering by confidence threshold
- Array serialization
- Countable and Iterator interfaces
- Immutability verification
- Large dataset handling
- Accuracy of statistical calculations

**Key Test Methods**:
- testGetCount
- testGetSpan (with bounds checking)
- testGetText (empty, single, multiple spans)
- testGetAverageConfidence
- testGetMinConfidence
- testGetMaxConfidence
- testFilterByConfidence (with validation)
- testToArray
- testCountableInterface
- testIterationOverResult
- testConfidenceStatisticsAccuracy

## Total Test Count: 87 Tests

## Running the Tests

### Prerequisites
- PHP 8.1+
- PHPUnit 9+ (install via composer)
- The pdf_oxide PHP binding installed

### Installation
```bash
cd /path/to/pdf_oxide/php
composer install
composer require --dev phpunit/phpunit
```

### Execution
```bash
# Run all OCR tests
./vendor/bin/phpunit tests/Unit/Ocr/

# Run specific test file
./vendor/bin/phpunit tests/Unit/Ocr/OcrEngineTest.php

# Run with verbose output
./vendor/bin/phpunit -v tests/Unit/Ocr/

# Run with code coverage
./vendor/bin/phpunit --coverage-html coverage tests/Unit/Ocr/
```

## Test Design Patterns

### Mocking
- FFI handles mocked using PHPUnit's MockBuilder
- FunctionBindings mocked to control FFI behavior
- Span creation uses mock bindings for reproducible tests

### Assertions
- Type assertions for returned values
- Boundary value testing (0.0, 1.0, negative, out of bounds)
- Exception assertions for error conditions
- Array structure verification

### Fixtures
- Mock OcrSpan objects for result testing
- Configurable confidence and text values
- Multiple span combinations for aggregation testing

## Key Testing Strategies

1. **Lifecycle Testing**: Verify proper resource creation and cleanup
2. **Immutability Testing**: Ensure objects cannot be modified after creation
3. **Boundary Testing**: Test edge cases (empty, zero, max values)
4. **Exception Testing**: Verify proper error handling
5. **Integration Testing**: Test interactions between classes
6. **Statistical Testing**: Verify accuracy of aggregated calculations
7. **Interface Testing**: Test Countable, Iterator implementations

## Coverage Goals

- ✅ OcrEngine: All public methods
- ✅ OcrSpan: All public methods and immutability
- ✅ OcrResult: All public methods, statistics, filtering
- ✅ Error handling and edge cases
- ✅ Resource management and cleanup

## Notes

- Tests use PHPUnit 9+ compatible syntax
- Mock objects are created dynamically to avoid external dependencies
- All test classes follow PSR-12 coding standards
- Each test method includes detailed assertions
- Tests document expected behavior through assertions and comments
