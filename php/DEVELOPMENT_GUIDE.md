# PHP Binding Development Guide

Guide for continuing development of the PDF Oxide PHP binding.

## Quick Start for Developers

### Setup Development Environment

```bash
cd php

# Install dependencies
composer install

# Enable FFI extension
php -m | grep ffi

# Verify installation
php -r "require 'vendor/autoload.php'; \
  echo PdfOxide\FFI\NativeLibrary::getPlatformInfo() ? 'OK' : 'FAIL';"
```

### Running Tests

```bash
# Run all tests
composer test

# Run specific test file
composer test -- tests/Unit/FFI/ErrorHandlerTest.php

# Generate coverage report
composer test-coverage
```

### Code Quality

```bash
# Check code style (PSR-12)
composer cs-check

# Fix code style automatically
composer cs-fix

# Run static analysis
composer psalm

# Run all checks
composer check
```

## Architecture Overview

### Layer 1: FFI Foundation
Core interaction with native library via PHP FFI.

**Key Classes**:
- `NativeLibrary`: Cross-platform library loading
- `FunctionBindings`: Type-safe FFI wrappers
- `ErrorHandler`: Error code to exception mapping
- `StringMarshaller`: String encoding/decoding
- `HandleManager`: Resource lifecycle management

**Key Concept**: All Rust FFI functions are wrapped here with proper error handling.

### Layer 2: Main Classes
High-level PHP APIs for common operations.

**Current**:
- `PdfDocument`: Reading and analysis (COMPLETE ✅)

**Planned**:
- `Pdf`: Creation
- `DocumentEditor`: Editing

### Layer 3: Managers & Utilities
Specialized operations and builder patterns.

**Planned**:
- ExtractionManager, SearchManager, RenderingManager
- AnnotationManager, FormManager, MetadataManager
- PageManager, OCRManager, SignatureManager, BarcodeManager
- Various Option builders

### Layer 4: Types & Enums
Data structures and enumerations.

**Current** (7 types):
- Rect, Point, Color, SearchResult, Font, Image, Annotation

**Current** (1 enum):
- PageSize

**Planned**:
- PageInfo, Metadata, FormField, etc. (10+ more types)
- AnnotationType, FormFieldType, BlendMode, etc. (5+ more enums)

## Adding New Functionality

### Pattern 1: Adding a Simple Type

Create a new readonly class in `src/Types/`:

```php
<?php
namespace PdfOxide\Types;

readonly class MyType
{
    public function __construct(
        public string $field1,
        public int $field2
    ) {}

    public function toArray(): array
    {
        return [
            'field1' => $this->field1,
            'field2' => $this->field2,
        ];
    }
}
```

### Pattern 2: Adding FFI Functions

1. Declare in C header (`php/include/pdf_oxide.h`)
2. Add wrapper in `FunctionBindings.php`:

```php
public function myNewFunction(CData $handle, string $param): string
{
    $cParam = StringMarshaller::toCString($param);
    $errorCode = FFI::new('int');

    try {
        $result = $this->ffi->ffi_function_name($handle, $cParam, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'ffi_function_name', ['param' => $param]);
        return StringMarshaller::fromCString($result);
    } finally {
        unset($cParam);
    }
}
```

3. Use in higher-level class

### Pattern 3: Adding a Manager

Create in `src/Managers/`:

```php
<?php
namespace PdfOxide\Managers;

use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\MyType;

class MyManager
{
    private FunctionBindings $bindings;
    private CData $handle;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    public function operation(): MyType
    {
        // Use bindings to call FFI functions
        $result = $this->bindings->myNewFunction($this->handle, 'param');
        return new MyType($result);
    }
}
```

### Pattern 4: Adding Tests

Create in `tests/Unit/`:

```php
<?php
namespace PdfOxide\Tests\Unit;

use PHPUnit\Framework\TestCase;
use PdfOxide\MyClass;

class MyClassTest extends TestCase
{
    public function testSomething(): void
    {
        $obj = new MyClass();
        $this->assertEquals('expected', $obj->method());
    }
}
```

Run:
```bash
composer test -- tests/Unit/MyClassTest.php
```

## Important Files to Know

| File | Purpose |
|------|---------|
| `include/pdf_oxide.h` | C header for FFI definitions |
| `src/FFI/NativeLibrary.php` | Library loading (start here for platform issues) |
| `src/FFI/FunctionBindings.php` | FFI wrappers (add new functions here) |
| `src/Exceptions/PdfException.php` | Base exception (reference for error handling) |
| `src/PdfDocument.php` | Reading API (reference implementation for main classes) |
| `phpunit.xml` | Test configuration |
| `psalm.xml` | Static analysis configuration |
| `.php-cs-fixer.php` | Code style configuration |

## Common Tasks

### Task: Add a new FFI function

1. Check if function exists in C header (`php/include/pdf_oxide.h`)
2. If not, add declaration
3. Add wrapper method in `FunctionBindings.php`
4. Create test in `tests/Unit/FFI/FunctionBindingsTest.php`
5. Use in higher-level class (e.g., PdfDocument, Manager)
6. Add test for higher-level usage

### Task: Add a new Exception type

1. Create class in `src/Exceptions/`
2. Extend `PdfException`
3. Add to `ErrorHandler::createException()` switch
4. Add test case
5. Document in README.md

### Task: Add a new Manager

1. Create class in `src/Managers/`
2. Implement methods using FunctionBindings
3. Create data types if needed in `src/Types/`
4. Add accessor method to main class (PdfDocument, Pdf, etc.)
5. Create test file
6. Add example usage

### Task: Fix a Bug

1. Create a minimal test case that reproduces it
2. Make test fail: `composer test`
3. Fix the code
4. Make test pass: `composer test`
5. Run full checks: `composer check`
6. Commit with reference to issue

## Code Style Guidelines

- **PHP Version**: 8.1+
- **Strict Types**: Use `declare(strict_types=1);` at top of every file
- **Naming**: PSR-12 (classes: PascalCase, methods: camelCase, constants: UPPER_SNAKE_CASE)
- **Formatting**: Run `composer cs-fix` before committing
- **Analysis**: Run `composer psalm` to check for type issues
- **Documentation**: Include PHPDoc comments for public methods

## Performance Considerations

- **Handle Management**: Always close handles or they'll be freed on shutdown
- **String Marshaling**: Avoid converting large strings repeatedly
- **Memory**: Be aware of document size (typically 10-20MB per document)
- **Caching**: Consider caching expensive operations (e.g., page count)

## Debugging Tips

### Debug FFI Initialization

```php
$info = PdfOxide\FFI\NativeLibrary::getPlatformInfo();
print_r($info);

echo "Header: " . PdfOxide\FFI\NativeLibrary::getHeaderFile() . "\n";
echo "Library: " . PdfOxide\FFI\NativeLibrary::getLibraryFile() . "\n";
```

### Debug Handle Management

```php
$stats = PdfOxide\FFI\HandleManager::getStatistics();
print_r($stats);
```

### Debug String Marshaling

```php
use PdfOxide\FFI\StringMarshaller;

$valid = StringMarshaller::isValidUtf8($str);
$safe = StringMarshaller::ensureUtf8($str);
```

### Debug Exceptions

```php
try {
    // operation
} catch (\PdfOxide\Exceptions\PdfException $e) {
    echo "Code: " . $e->getErrorCode() . "\n";
    print_r($e->getContext());
    echo "Trace:\n" . $e->getTraceAsString();
}
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes
# Add tests
# Run checks
composer check

# Commit (descriptive message)
git commit -m "feat: Add new feature

- Detail 1
- Detail 2

Closes #123"

# Push and create PR
git push origin feature/my-feature
```

## Continuous Integration

The project uses GitHub Actions for CI/CD. Configuration in `.github/workflows/ci.yml`:

- Runs tests on multiple PHP versions (8.1+)
- Checks code style (PSR-12)
- Runs static analysis (Psalm)
- Generates coverage reports

All checks must pass before merging PR.

## Release Process

1. Update version in `composer.json`
2. Update `IMPLEMENTATION_STATUS.md`
3. Create GitHub release with notes
4. Tag: `git tag v0.x.x && git push --tags`
5. Update Packagist (auto via webhook)

## Resources

- **PHP FFI Docs**: https://www.php.net/manual/en/book.ffi.php
- **PSR-12**: https://www.php-fig.org/psr/psr-12/
- **PHPUnit**: https://phpunit.de/
- **Psalm**: https://psalm.dev/
- **Rust FFI**: `src/ffi/` (Rust source code)

## Getting Help

- Check existing tests for examples
- Review PdfDocument.php for patterns
- Look at Go binding for reference: `go/internal/binding/`
- Ask in GitHub discussions

---

**Happy coding! 🚀**
