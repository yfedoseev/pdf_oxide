# PDF Oxide PHP - Quick Reference

Fast lookup for common operations.

## Installation

```bash
# Install package
composer require pdf-oxide/pdf-oxide

# Verify
php -r "require 'vendor/autoload.php'; \
  echo PdfOxide\FFI\NativeLibrary::getPlatformInfo() ? 'OK' : 'FAIL';"
```

## Reading PDFs

### Open and Close

```php
use PdfOxide\PdfDocument;

$pdf = new PdfDocument('file.pdf');
$pageCount = $pdf->getPageCount();
$pdf->close();
// or auto-closes on destruct: unset($pdf);
```

### Extract Text

```php
// From specific page
$text = $pdf->extractText(0);

// All pages
$text = $pdf->extractTextAll();

// As Markdown
$md = $pdf->toMarkdown(0);
$md = $pdf->toMarkdownAll();

// As HTML
$html = $pdf->toHtml(0);

// Plain text with layout
$text = $pdf->toPlainText(0);
```

### Search

```php
// Search entire document
$results = $pdf->searchAll('keyword');

// Search specific page
$results = $pdf->searchPage('keyword', 0);

// Case-sensitive
$results = $pdf->searchAll('keyword', caseSensitive: true);

// Process results
foreach ($results as $result) {
    echo $result->text;           // The found text
    echo $result->pageIndex;      // Page (0-based)
    echo $result->position;       // Character position
    echo $result->boundingBox->x; // X coordinate
}
```

### Extract Content

```php
// Fonts
$fonts = $pdf->getFonts(0);
foreach ($fonts as $font) {
    echo $font->name;      // Font name
    echo $font->type;      // Font type
    echo $font->embedded;  // Is embedded
}

// Images
$images = $pdf->getImages(0);
foreach ($images as $image) {
    echo $image->format;      // Format (PNG, JPEG, etc.)
    echo $image->width;       // Width in pixels
    echo $image->height;      // Height in pixels
    echo $image->getAspectRatio(); // Aspect ratio
}

// Annotations
$annotations = $pdf->getAnnotations(0);
foreach ($annotations as $ann) {
    echo $ann->type;          // Annotation type
    echo $ann->content;       // Annotation content
}
```

### Document Info

```php
$metadata = $pdf->getMetadata();
echo $metadata['file_path'];
echo $metadata['file_size'];
echo $metadata['page_count'];
echo $metadata['version']['major']; // 1
echo $metadata['version']['minor']; // 4
echo $metadata['has_structure_tree']; // bool
```

## Data Types

### Color

```php
use PdfOxide\Types\Color;

// Create from RGB (0-255)
$color = new Color(255, 128, 0);
$color = new Color(255, 128, 0, 200); // with alpha

// From hex
$color = Color::fromHex('#FF8000');
$color = Color::fromHex('FF8000');
$color = Color::fromHex('#FF8000FF'); // with alpha

// Common colors
Color::black();
Color::white();
Color::red();
Color::green();
Color::blue();

// Convert
$hex = $color->toHex();        // "#FF8000"
$rgba = $color->toRgba();      // 32-bit integer
$argb = $color->toArgb();      // 32-bit integer
$array = $color->toArray();    // ['red' => 255, ...]
```

### Rect & Point

```php
use PdfOxide\Types\Rect;
use PdfOxide\Types\Point;

$rect = new Rect(10, 20, 100, 50);
$point = new Point(50, 30);

// Operations
$rect->contains($point);         // bool
$rect->intersects($other_rect);  // bool
$rect->getRight();               // x + width
$rect->getBottom();              // y + height
$rect->toArray();                // ['x' => 10, ...]

$point->distanceTo($other_point);
$point->toArray();
```

### SearchResult

```php
// From search results
foreach ($results as $result) {
    $result->text;
    $result->pageIndex;
    $result->position;
    $result->boundingBox;  // Rect object

    // Convert to array
    $result->toArray();
}
```

## Exception Handling

```php
use PdfOxide\Exceptions\{
    PdfException,
    ParseException,
    IoException,
    SearchException,
    ValidationException
};

try {
    $pdf = new PdfDocument('file.pdf');
} catch (IoException $e) {
    echo "File error: " . $e->getMessage();
    echo "Code: " . $e->getErrorCode();
    print_r($e->getContext());
} catch (ParseException $e) {
    echo "PDF parse error: " . $e->getMessage();
} catch (PdfException $e) {
    echo "PDF error: " . $e->getMessage();
}
```

## Enums

### PageSize

```php
use PdfOxide\Enums\PageSize;

// Available sizes
PageSize::A0;      // 2384 x 3370 pt
PageSize::A1;      // 1684 x 2384 pt
PageSize::A2;      // 1191 x 1684 pt
PageSize::A3;      // 842 x 1191 pt
PageSize::A4;      // 595 x 842 pt (default)
PageSize::A5;      // 420 x 595 pt
PageSize::A6;      // 298 x 420 pt
PageSize::LETTER;  // 612 x 792 pt
PageSize::LEGAL;   // 612 x 1008 pt
PageSize::TABLOID; // 792 x 1224 pt
PageSize::LEDGER;  // 1224 x 792 pt

// Get dimensions
$dims = PageSize::A4->getDimensions();
echo $dims['width'];   // 595
echo $dims['height'];  // 842

$widthMm = PageSize::A4->getWidthMm();   // 210
$heightMm = PageSize::A4->getHeightMm(); // 297
```

## Testing Your Installation

### Run Examples

```bash
cd php

# Basic reading
php examples/01_basic_reading.php sample.pdf

# Text extraction & search
php examples/02_text_extraction.php sample.pdf "search term"
```

### Run Tests

```bash
# All tests
composer test

# Specific test
composer test -- tests/Unit/Types/ColorTest.php

# With coverage
composer test-coverage
```

### Code Quality

```bash
# Check style
composer cs-check

# Fix style
composer cs-fix

# Static analysis
composer psalm

# All checks
composer check
```

## Common Patterns

### Process All Pages

```php
$pdf = new PdfDocument('file.pdf');
for ($i = 0; $i < $pdf->getPageCount(); $i++) {
    $text = $pdf->extractText($i);
    // Process page text
}
```

### Search and Extract Context

```php
$results = $pdf->searchAll('keyword');
foreach ($results as $result) {
    $page = $result->pageIndex;
    $text = $pdf->extractText($page);

    // Get surrounding context (rough)
    $pos = $result->position;
    $start = max(0, $pos - 100);
    $context = substr($text, $start, 200);

    echo "...{$context}...";
}
```

### Export to File

```php
// Export as Markdown
$markdown = $pdf->toMarkdownAll();
file_put_contents('output.md', $markdown);

// Export as HTML
for ($i = 0; $i < $pdf->getPageCount(); $i++) {
    $html = $pdf->toHtml($i);
    file_put_contents("page_{$i}.html", $html);
}
```

### Error-Safe Processing

```php
try {
    $pdf = new PdfDocument('file.pdf');

    for ($i = 0; $i < $pdf->getPageCount(); $i++) {
        try {
            $text = $pdf->extractText($i);
            // Process safely
        } catch (\Exception $e) {
            echo "Page $i error: " . $e->getMessage();
            continue; // Skip this page
        }
    }

    $pdf->close();
} catch (\Exception $e) {
    echo "Error: " . $e->getMessage();
}
```

## Troubleshooting

### FFI Not Loaded
```bash
php -m | grep ffi
# Should show: ffi

# If not:
php -r "echo phpinfo();" | grep -i ffi
```

### Library Not Found
```bash
# Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Add to path
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
```

### UTF-8 Issues
```php
// Force UTF-8 encoding
$text = mb_convert_encoding($text, 'UTF-8', 'ISO-8859-1');
```

## Performance Tips

1. **Reuse Document Object**: Don't open/close repeatedly
2. **Cache Results**: Save extracted text/metadata
3. **Close When Done**: Free memory: `$pdf->close()`
4. **Batch Operations**: Process multiple pages in loops
5. **Check Page Count**: Validate page index before access

```php
// ✅ Good
$pdf = new PdfDocument('file.pdf');
for ($i = 0; $i < $pdf->getPageCount(); $i++) {
    $text = $pdf->extractText($i);
}
$pdf->close();

// ❌ Avoid
for ($i = 0; $i < 100; $i++) {
    $pdf = new PdfDocument('file.pdf');
    $text = $pdf->extractText(0);
    $pdf->close();
}
```

## Resources

- **README.md** - Feature overview
- **INSTALLATION.md** - Setup guide
- **DEVELOPMENT_GUIDE.md** - Extending the library
- **examples/** - Working code examples
- **tests/** - Usage patterns in tests

## API Completeness

**Currently Implemented** ✅
- PdfDocument (reading)
- 7 data types
- 10 exception types
- Error handling
- Resource management

**Coming Soon** 🔄
- Pdf (creation)
- DocumentEditor (editing)
- Managers (specialized operations)
- More enums and builders

---

**For more info**: See full documentation in README.md and docs/
