<?php

declare(strict_types=1);

namespace PdfOxide;

use FFI\CData;
use PdfOxide\FFI\{FunctionBindings, HandleManager, StringMarshaller};
use PdfOxide\Exceptions\InvalidStateException;
use PdfOxide\Types\{Color, Point, Rect};
use PdfOxide\Enums\PageSize;

/**
 * Main class for creating and modifying PDF documents.
 *
 * Provides fluent interface for building PDFs with text, images, shapes, and styling.
 */
class Pdf
{
    private ?CData $handle = null;
    private bool $closed = false;
    private FunctionBindings $bindings;
    private bool $isNew = false;

    private Color $currentColor;
    private string $currentFont = 'Helvetica';
    private float $currentFontSize = 12.0;

    /**
     * Private constructor - use static factory methods.
     */
    private function __construct()
    {
        $this->bindings = new FunctionBindings();
        $this->currentColor = Color::black();
    }

    /**
     * Create a new blank PDF document.
     *
     * @return self A new Pdf instance
     */
    public static function create(): self
    {
        $pdf = new self();
        $pdf->isNew = true;
        // Document handle would be created here with FFI call
        // For now, placeholder for phase 3 continuation
        return $pdf;
    }

    /**
     * Create a PDF from Markdown content.
     *
     * @param string $markdown Markdown formatted text
     * @return self A new Pdf instance
     */
    public static function fromMarkdown(string $markdown): self
    {
        $pdf = new self();
        $pdf->isNew = true;
        try {
            $pdf->handle = $pdf->bindings->pdfFromMarkdown($markdown);
            HandleManager::register($pdf->handle, 'PdfHandle', 'generated-from-markdown');
        } catch (\Exception $e) {
            throw new \PdfOxide\Exceptions\ParseException("Failed to create PDF from Markdown: " . $e->getMessage());
        }
        return $pdf;
    }

    /**
     * Create a PDF from HTML content.
     *
     * @param string $html HTML formatted text
     * @return self A new Pdf instance
     */
    public static function fromHtml(string $html): self
    {
        $pdf = new self();
        $pdf->isNew = true;
        try {
            $pdf->handle = $pdf->bindings->pdfFromHtml($html);
            HandleManager::register($pdf->handle, 'PdfHandle', 'generated-from-html');
        } catch (\Exception $e) {
            throw new \PdfOxide\Exceptions\ParseException("Failed to create PDF from HTML: " . $e->getMessage());
        }
        return $pdf;
    }

    /**
     * Create a PDF from plain text.
     *
     * @param string $text Plain text content
     * @return self A new Pdf instance
     */
    public static function fromText(string $text): self
    {
        $pdf = new self();
        $pdf->isNew = true;
        try {
            $pdf->handle = $pdf->bindings->pdfFromText($text);
            HandleManager::register($pdf->handle, 'PdfHandle', 'generated-from-text');
        } catch (\Exception $e) {
            throw new \PdfOxide\Exceptions\ParseException("Failed to create PDF from text: " . $e->getMessage());
        }
        return $pdf;
    }

    /**
     * Open an existing PDF document for editing.
     *
     * @param string $filePath Path to the PDF file
     * @return self The Pdf instance
     */
    public static function open(string $filePath): self
    {
        if (!file_exists($filePath)) {
            throw new \PdfOxide\Exceptions\IoException(
                "PDF file not found: {$filePath}",
                ['file' => $filePath]
            );
        }

        $pdf = new self();
        $pdf->isNew = false;
        // Load document with FFI call
        // For now, placeholder for phase 3 continuation
        return $pdf;
    }

    /**
     * Add a new page to the document.
     *
     * @param float|null $width Page width in points (default: A4 width)
     * @param float|null $height Page height in points (default: A4 height)
     * @return self Fluent interface
     */
    public function addPage(?float $width = null, ?float $height = null): self
    {
        $this->ensureOpen();

        // Default to A4 size
        if ($width === null || $height === null) {
            $dims = PageSize::A4->getDimensions();
            $width ??= $dims['width'];
            $height ??= $dims['height'];
        }

        // Add page via FFI
        // Call: this->bindings->pdfAddPage(this->handle, (int)$width, (int)$height);

        return $this;
    }

    /**
     * Add a page using a predefined page size.
     *
     * @param PageSize $size The page size enum
     * @return self Fluent interface
     */
    public function addPageWithSize(PageSize $size): self
    {
        $dims = $size->getDimensions();
        return $this->addPage($dims['width'], $dims['height']);
    }

    /**
     * Remove a page from the document.
     *
     * @param int $pageIndex Zero-based page index
     * @return self Fluent interface
     */
    public function removePage(int $pageIndex): self
    {
        $this->ensureOpen();
        // Call FFI: this->bindings->pdfRemovePage(this->handle, $pageIndex);
        return $this;
    }

    /**
     * Add text to the current page.
     *
     * @param string $text The text to add
     * @param float $x X coordinate in points
     * @param float $y Y coordinate in points
     * @param float|null $fontSize Font size (uses current if null)
     * @return self Fluent interface
     */
    public function text(
        string $text,
        float $x,
        float $y,
        ?float $fontSize = null
    ): self {
        $this->ensureOpen();

        $fontSize ??= $this->currentFontSize;

        // Convert text to C string
        $cText = StringMarshaller::toCString($text);
        $cFont = StringMarshaller::toCString($this->currentFont);

        try {
            // Call FFI: this->bindings->pdfAddText(
            //     this->handle, $cText, $x, $y, $fontSize,
            //     this->currentColor->red, this->currentColor->green,
            //     this->currentColor->blue, this->currentColor->alpha
            // );
        } finally {
            unset($cText, $cFont);
        }

        return $this;
    }

    /**
     * Add an image to the current page.
     *
     * @param string $imagePath Path to image file
     * @param float $x X coordinate in points
     * @param float $y Y coordinate in points
     * @param float $width Image width in points
     * @param float|null $height Image height in points (maintains aspect ratio if null)
     * @return self Fluent interface
     */
    public function image(
        string $imagePath,
        float $x,
        float $y,
        float $width,
        ?float $height = null
    ): self {
        $this->ensureOpen();

        if (!file_exists($imagePath)) {
            throw new \PdfOxide\Exceptions\IoException(
                "Image file not found: {$imagePath}",
                ['file' => $imagePath]
            );
        }

        $cPath = StringMarshaller::toCString($imagePath);

        try {
            // Call FFI: this->bindings->pdfAddImage(
            //     this->handle, $cPath, $x, $y, $width, $height ?? 0
            // );
        } finally {
            unset($cPath);
        }

        return $this;
    }

    /**
     * Draw a line on the current page.
     *
     * @param float $x1 Start X coordinate
     * @param float $y1 Start Y coordinate
     * @param float $x2 End X coordinate
     * @param float $y2 End Y coordinate
     * @param float $lineWidth Line width in points
     * @return self Fluent interface
     */
    public function line(
        float $x1,
        float $y1,
        float $x2,
        float $y2,
        float $lineWidth = 1.0
    ): self {
        $this->ensureOpen();

        // Call FFI: this->bindings->pdfDrawLine(
        //     this->handle, $x1, $y1, $x2, $y2, $lineWidth,
        //     this->currentColor->red, this->currentColor->green,
        //     this->currentColor->blue, this->currentColor->alpha
        // );

        return $this;
    }

    /**
     * Draw a rectangle on the current page.
     *
     * @param float $x Top-left X coordinate
     * @param float $y Top-left Y coordinate
     * @param float $width Rectangle width
     * @param float $height Rectangle height
     * @param bool $fill Whether to fill the rectangle
     * @param float|null $lineWidth Line width for outline
     * @return self Fluent interface
     */
    public function rect(
        float $x,
        float $y,
        float $width,
        float $height,
        bool $fill = false,
        ?float $lineWidth = 1.0
    ): self {
        $this->ensureOpen();

        // Call FFI: this->bindings->pdfDrawRect(
        //     this->handle, $x, $y, $width, $height, $fill ? 1 : 0,
        //     $lineWidth ?? 0,
        //     this->currentColor->red, this->currentColor->green,
        //     this->currentColor->blue, this->currentColor->alpha
        // );

        return $this;
    }

    /**
     * Draw a circle on the current page.
     *
     * @param float $centerX Center X coordinate
     * @param float $centerY Center Y coordinate
     * @param float $radius Circle radius
     * @param bool $fill Whether to fill the circle
     * @param float $lineWidth Line width for outline
     * @return self Fluent interface
     */
    public function circle(
        float $centerX,
        float $centerY,
        float $radius,
        bool $fill = false,
        float $lineWidth = 1.0
    ): self {
        $this->ensureOpen();

        // Call FFI: this->bindings->pdfDrawCircle(
        //     this->handle, $centerX, $centerY, $radius, $fill ? 1 : 0, $lineWidth,
        //     this->currentColor->red, this->currentColor->green,
        //     this->currentColor->blue, this->currentColor->alpha
        // );

        return $this;
    }

    /**
     * Set the current font name and size.
     *
     * @param string $fontName Font name (e.g., 'Helvetica', 'Times-Roman')
     * @param float $fontSize Font size in points
     * @return self Fluent interface
     */
    public function setFont(string $fontName, float $fontSize = 12.0): self
    {
        $this->ensureOpen();
        $this->currentFont = $fontName;
        $this->currentFontSize = $fontSize;

        // Call FFI: this->bindings->pdfSetFont(this->handle, $fontName, $fontSize);

        return $this;
    }

    /**
     * Set the current text color.
     *
     * @param Color $color The color to use
     * @return self Fluent interface
     */
    public function setColor(Color $color): self
    {
        $this->ensureOpen();
        $this->currentColor = $color;

        // Call FFI: this->bindings->pdfSetColor(
        //     this->handle, $color->red, $color->green,
        //     $color->blue, $color->alpha
        // );

        return $this;
    }

    /**
     * Set the line width for drawing operations.
     *
     * @param float $width Line width in points
     * @return self Fluent interface
     */
    public function setLineWidth(float $width): self
    {
        $this->ensureOpen();

        // Call FFI: this->bindings->pdfSetLineWidth(this->handle, $width);

        return $this;
    }

    /**
     * Save the document to a file.
     *
     * @param string $filePath Path where to save the PDF
     * @throws \PdfOxide\Exceptions\IoException on save error
     */
    public function save(string $filePath): void
    {
        $this->ensureOpen();
        $this->bindings->pdfSave($this->handle, $filePath);
    }

    /**
     * Save the document to a string (in-memory PDF).
     *
     * @return string The PDF document as a string
     */
    public function saveToString(): string
    {
        $this->ensureOpen();
        return $this->bindings->pdfSaveToBytes($this->handle);
    }

    /**
     * Get the current font name.
     *
     * @return string Current font name
     */
    public function getCurrentFont(): string
    {
        return $this->currentFont;
    }

    /**
     * Get the current font size.
     *
     * @return float Current font size
     */
    public function getCurrentFontSize(): float
    {
        return $this->currentFontSize;
    }

    /**
     * Get the current drawing color.
     *
     * @return Color Current color
     */
    public function getCurrentColor(): Color
    {
        return $this->currentColor;
    }

    /**
     * Check if document is still open.
     *
     * @return bool True if document is open
     */
    public function isOpen(): bool
    {
        return !$this->closed && $this->handle !== null;
    }

    /**
     * Close the document and free resources.
     *
     * @return void
     */
    public function close(): void
    {
        if ($this->handle !== null && !$this->closed) {
            HandleManager::unregister($this->handle);
            // Call FFI: this->bindings->pdfFree(this->handle);
            $this->closed = true;
            $this->handle = null;
        }
    }

    /**
     * Ensure the document is still open.
     *
     * @throws InvalidStateException if document is closed
     * @internal
     */
    private function ensureOpen(): void
    {
        if (!$this->isOpen()) {
            throw new InvalidStateException(
                'PDF document is closed',
                ['is_new' => $this->isNew]
            );
        }
    }

    /**
     * Close document on destruct.
     */
    public function __destruct()
    {
        $this->close();
    }
}
