<?php

declare(strict_types=1);

namespace PdfOxide;

use FFI\CData;
use PdfOxide\FFI\{FunctionBindings, HandleManager, NativeLibrary};
use PdfOxide\Exceptions\InvalidStateException;
use PdfOxide\Types\{Rect, Point, Color, SearchResult, Font, Image, Annotation};
use PdfOxide\Managers\{
    MetadataManager,
    OutlineManager,
    LayerManager,
    CacheManager,
    SignatureManager,
    ComplianceManager,
    RenderingManager,
    OcrManager,
    FormManager,
    AnnotationManager,
    BarcodeManager,
    ExtractionManager,
    HybridMLManager,
    XfaFormManager,
    DocumentEditorManager
};
use PdfOxide\Xfa\XfaForm;

/**
 * Main class for reading and analyzing PDF documents.
 *
 * Provides methods for text extraction, search, and content analysis.
 */
class PdfDocument
{
    private ?CData $handle = null;
    private bool $closed = false;
    private FunctionBindings $bindings;
    private string $filePath;

    // Lazy-loaded managers
    private ?MetadataManager $metadataManager = null;
    private ?OutlineManager $outlineManager = null;
    private ?LayerManager $layerManager = null;
    private ?CacheManager $cacheManager = null;
    private ?SignatureManager $signatureManager = null;
    private ?ComplianceManager $complianceManager = null;
    private ?RenderingManager $renderingManager = null;
    private ?OcrManager $ocrManager = null;
    private ?FormManager $formManager = null;
    private ?AnnotationManager $annotationManager = null;
    private ?BarcodeManager $barcodeManager = null;
    private ?ExtractionManager $extractionManager = null;
    private ?HybridMLManager $hybridMLManager = null;
    private ?XfaFormManager $xfaFormManager = null;
    private ?DocumentEditorManager $editorManager = null;

    /**
     * Open a PDF document for reading.
     *
     * @param string $filePath Path to the PDF file
     * @throws \PdfOxide\Exceptions\PdfException on error
     */
    public function __construct(string $filePath)
    {
        $this->filePath = $filePath;
        $this->bindings = new FunctionBindings();

        if (!file_exists($filePath)) {
            throw new \PdfOxide\Exceptions\IoException(
                "PDF file not found: {$filePath}",
                ['file' => $filePath]
            );
        }

        // Open the document
        $this->handle = $this->bindings->pdfDocumentOpen($filePath);
        HandleManager::register($this->handle, 'PdfDocumentHandle', $filePath);
    }

    /**
     * Get the number of pages in the document.
     *
     * @return int Number of pages
     */
    public function getPageCount(): int
    {
        $this->ensureOpen();
        return $this->bindings->pdfDocumentGetPageCount($this->handle);
    }

    /**
     * Get the PDF version.
     *
     * @return array Array with 'major' and 'minor' version numbers
     */
    public function getVersion(): array
    {
        $this->ensureOpen();
        return $this->bindings->pdfDocumentGetVersion($this->handle);
    }

    /**
     * Check if document has a structure tree (for accessibility).
     *
     * @return bool True if document has structure tree
     */
    public function hasStructureTree(): bool
    {
        $this->ensureOpen();
        return $this->bindings->pdfDocumentHasStructureTree($this->handle);
    }

    // ==================== MANAGERS ====================

    /**
     * Get metadata manager for accessing document properties.
     */
    public function metadata(): MetadataManager
    {
        $this->ensureOpen();
        if ($this->metadataManager === null) {
            $this->metadataManager = new MetadataManager($this->handle);
        }
        return $this->metadataManager;
    }

    /**
     * Get outline manager for accessing bookmarks.
     */
    public function outlines(): OutlineManager
    {
        $this->ensureOpen();
        if ($this->outlineManager === null) {
            $this->outlineManager = new OutlineManager($this->handle);
        }
        return $this->outlineManager;
    }

    /**
     * Get layer manager for accessing OCG/layers.
     */
    public function layers(): LayerManager
    {
        $this->ensureOpen();
        if ($this->layerManager === null) {
            $this->layerManager = new LayerManager($this->handle);
        }
        return $this->layerManager;
    }

    /**
     * Get cache manager for cache control.
     */
    public function cache(): CacheManager
    {
        $this->ensureOpen();
        if ($this->cacheManager === null) {
            $this->cacheManager = new CacheManager($this->handle);
        }
        return $this->cacheManager;
    }

    /**
     * Get signature manager for digital signatures.
     */
    public function signatures(): SignatureManager
    {
        $this->ensureOpen();
        if ($this->signatureManager === null) {
            $this->signatureManager = new SignatureManager($this->handle);
        }
        return $this->signatureManager;
    }

    /**
     * Get compliance manager for PDF standards conversion.
     */
    public function compliance(): ComplianceManager
    {
        $this->ensureOpen();
        if ($this->complianceManager === null) {
            $this->complianceManager = new ComplianceManager($this->handle);
        }
        return $this->complianceManager;
    }

    /**
     * Get rendering manager for page rendering to images.
     */
    public function rendering(): RenderingManager
    {
        $this->ensureOpen();
        if ($this->renderingManager === null) {
            $this->renderingManager = new RenderingManager($this->handle);
        }
        return $this->renderingManager;
    }

    /**
     * Get OCR manager for text recognition.
     */
    public function ocr(): OcrManager
    {
        $this->ensureOpen();
        if ($this->ocrManager === null) {
            $this->ocrManager = new OcrManager($this->handle);
        }
        return $this->ocrManager;
    }

    /**
     * Get form manager for AcroForm operations.
     */
    public function forms(): FormManager
    {
        $this->ensureOpen();
        if ($this->formManager === null) {
            $this->formManager = new FormManager($this->handle);
        }
        return $this->formManager;
    }

    /**
     * Get annotation manager for annotations.
     */
    public function annotations(): AnnotationManager
    {
        $this->ensureOpen();
        if ($this->annotationManager === null) {
            $this->annotationManager = new AnnotationManager($this->handle);
        }
        return $this->annotationManager;
    }

    /**
     * Get barcode manager for barcode generation.
     */
    public function barcodes(): BarcodeManager
    {
        $this->ensureOpen();
        if ($this->barcodeManager === null) {
            $this->barcodeManager = new BarcodeManager($this->handle);
        }
        return $this->barcodeManager;
    }

    /**
     * Get extraction manager for content extraction.
     */
    public function extraction(): ExtractionManager
    {
        $this->ensureOpen();
        if ($this->extractionManager === null) {
            $this->extractionManager = new ExtractionManager($this->handle);
        }
        return $this->extractionManager;
    }

    /**
     * Get hybrid ML manager for intelligent analysis.
     */
    public function hybridML(): HybridMLManager
    {
        $this->ensureOpen();
        if ($this->hybridMLManager === null) {
            $this->hybridMLManager = new HybridMLManager($this->handle);
        }
        return $this->hybridMLManager;
    }

    /**
     * Get XFA form manager for XFA operations.
     */
    public function xfa(): XfaFormManager
    {
        $this->ensureOpen();
        if ($this->xfaFormManager === null) {
            $this->xfaFormManager = new XfaFormManager($this->handle);
        }
        return $this->xfaFormManager;
    }

    /**
     * Get document editor manager for page manipulation.
     */
    public function editor(): DocumentEditorManager
    {
        $this->ensureOpen();
        if ($this->editorManager === null) {
            $this->editorManager = new DocumentEditorManager($this->handle);
        }
        return $this->editorManager;
    }

    /**
     * Check if document has XFA form.
     *
     * @return bool True if document contains XFA form
     */
    public function hasXfa(): bool
    {
        $this->ensureOpen();
        return $this->bindings->pdfDocumentHasXfa($this->handle);
    }

    /**
     * Get XFA form if document contains one.
     *
     * @return ?XfaForm The XFA form, or null if not present
     */
    public function getXfaForm(): ?XfaForm
    {
        $this->ensureOpen();

        if (!$this->hasXfa()) {
            return null;
        }

        $handle = $this->bindings->pdfParseXfaForm($this->handle);
        return new XfaForm($handle, $this->bindings);
    }

    /**
     * Extract plain text from a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return string Extracted text
     */
    public function extractText(int $pageIndex): string
    {
        $this->ensureOpen();
        $this->validatePageIndex($pageIndex);
        return $this->bindings->pdfDocumentExtractText($this->handle, $pageIndex);
    }

    /**
     * Extract text from all pages.
     *
     * @return string Concatenated text from all pages
     */
    public function extractTextAll(): string
    {
        $text = '';
        for ($i = 0; $i < $this->getPageCount(); $i++) {
            if ($text) {
                $text .= "\n\n---PAGE-BREAK---\n\n";
            }
            $text .= $this->extractText($i);
        }
        return $text;
    }

    /**
     * Convert a page to Markdown format.
     *
     * @param int $pageIndex Zero-based page index
     * @return string Markdown formatted text
     */
    public function toMarkdown(int $pageIndex): string
    {
        $this->ensureOpen();
        $this->validatePageIndex($pageIndex);
        return $this->bindings->pdfDocumentToMarkdown($this->handle, $pageIndex);
    }

    /**
     * Convert entire document to Markdown format.
     *
     * @return string Markdown formatted text
     */
    public function toMarkdownAll(): string
    {
        $this->ensureOpen();
        return $this->bindings->pdfDocumentToMarkdownAll($this->handle);
    }

    /**
     * Convert a page to HTML format.
     *
     * @param int $pageIndex Zero-based page index
     * @return string HTML content
     */
    public function toHtml(int $pageIndex): string
    {
        $this->ensureOpen();
        $this->validatePageIndex($pageIndex);
        return $this->bindings->pdfDocumentToHtml($this->handle, $pageIndex);
    }

    /**
     * Convert a page to plain text (layout-preserving).
     *
     * @param int $pageIndex Zero-based page index
     * @return string Plain text with layout preserved
     */
    public function toPlainText(int $pageIndex): string
    {
        $this->ensureOpen();
        $this->validatePageIndex($pageIndex);
        return $this->bindings->pdfDocumentToPlainText($this->handle, $pageIndex);
    }

    /**
     * Search for text in a specific page.
     *
     * @param string $searchTerm The text to search for
     * @param int $pageIndex Zero-based page index
     * @param bool $caseSensitive Whether search is case-sensitive
     * @return SearchResult[] Array of search results
     */
    public function searchPage(string $searchTerm, int $pageIndex, bool $caseSensitive = false): array
    {
        $this->ensureOpen();
        $this->validatePageIndex($pageIndex);

        $resultsHandle = $this->bindings->pdfDocumentSearchPage(
            $this->handle,
            $searchTerm,
            $pageIndex,
            $caseSensitive
        );

        try {
            return $this->parseSearchResults($resultsHandle);
        } finally {
            $this->bindings->oxideSearchResultFree($resultsHandle);
        }
    }

    /**
     * Search for text in entire document.
     *
     * @param string $searchTerm The text to search for
     * @param bool $caseSensitive Whether search is case-sensitive
     * @return SearchResult[] Array of search results
     */
    public function searchAll(string $searchTerm, bool $caseSensitive = false): array
    {
        $this->ensureOpen();

        $resultsHandle = $this->bindings->pdfDocumentSearchAll(
            $this->handle,
            $searchTerm,
            $caseSensitive
        );

        try {
            return $this->parseSearchResults($resultsHandle);
        } finally {
            $this->bindings->oxideSearchResultFree($resultsHandle);
        }
    }

    /**
     * Get embedded fonts from a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return Font[] Array of fonts
     */
    public function getFonts(int $pageIndex): array
    {
        $this->ensureOpen();
        $this->validatePageIndex($pageIndex);

        $fontsHandle = $this->bindings->pdfDocumentGetEmbeddedFonts($this->handle, $pageIndex);

        try {
            $fonts = [];
            $count = $this->bindings->oxideFontCount($fontsHandle);

            for ($i = 0; $i < $count; $i++) {
                $fonts[] = new Font(
                    $this->bindings->oxideFontGetName($fontsHandle, $i),
                    $this->bindings->oxideFontGetType($fontsHandle, $i),
                    $this->bindings->oxideFontIsEmbedded($fontsHandle, $i)
                );
            }

            return $fonts;
        } finally {
            $this->bindings->oxideFontFree($fontsHandle);
        }
    }

    /**
     * Get embedded images from a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return Image[] Array of images
     */
    public function getImages(int $pageIndex): array
    {
        $this->ensureOpen();
        $this->validatePageIndex($pageIndex);

        $imagesHandle = $this->bindings->pdfDocumentGetEmbeddedImages($this->handle, $pageIndex);

        try {
            $images = [];
            $count = $this->bindings->oxideImageCount($imagesHandle);

            for ($i = 0; $i < $count; $i++) {
                $images[] = new Image(
                    $this->bindings->oxideImageGetFormat($imagesHandle, $i),
                    $this->bindings->oxideImageGetWidth($imagesHandle, $i),
                    $this->bindings->oxideImageGetHeight($imagesHandle, $i)
                );
            }

            return $images;
        } finally {
            $this->bindings->oxideImageFree($imagesHandle);
        }
    }

    /**
     * Get annotations from a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return Annotation[] Array of annotations
     */
    public function getAnnotations(int $pageIndex): array
    {
        $this->ensureOpen();
        $this->validatePageIndex($pageIndex);

        $annotationsHandle = $this->bindings->pdfDocumentGetAnnotations($this->handle, $pageIndex);

        try {
            $annotations = [];
            $count = $this->bindings->oxideAnnotationCount($annotationsHandle);

            for ($i = 0; $i < $count; $i++) {
                $annotations[] = new Annotation(
                    $this->bindings->oxideAnnotationGetType($annotationsHandle, $i),
                    $this->bindings->oxideAnnotationGetContent($annotationsHandle, $i)
                );
            }

            return $annotations;
        } finally {
            $this->bindings->oxideAnnotationFree($annotationsHandle);
        }
    }

    /**
     * Get the file path of this document.
     *
     * @return string File path
     */
    public function getFilePath(): string
    {
        return $this->filePath;
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
            $this->bindings->pdfDocumentFree($this->handle);
            $this->closed = true;
            $this->handle = null;
        }
    }

    /**
     * Get document metadata.
     *
     * @return array Metadata array
     */
    public function getMetadata(): array
    {
        $this->ensureOpen();

        return [
            'file_path' => $this->filePath,
            'file_size' => filesize($this->filePath),
            'page_count' => $this->getPageCount(),
            'version' => $this->getVersion(),
            'has_structure_tree' => $this->hasStructureTree(),
        ];
    }

    /**
     * Parse search results from handle.
     *
     * @param CData $resultsHandle The search results handle
     * @return SearchResult[] Array of results
     * @internal
     */
    private function parseSearchResults(CData $resultsHandle): array
    {
        $results = [];
        $count = $this->bindings->oxideSearchResultCount($resultsHandle);

        for ($i = 0; $i < $count; $i++) {
            $bbox = $this->bindings->oxideSearchResultGetBbox($resultsHandle, $i);
            $results[] = new SearchResult(
                $this->bindings->oxideSearchResultGetText($resultsHandle, $i),
                $this->bindings->oxideSearchResultGetPage($resultsHandle, $i),
                $this->bindings->oxideSearchResultGetPosition($resultsHandle, $i),
                new Rect(
                    $bbox['x'],
                    $bbox['y'],
                    $bbox['width'],
                    $bbox['height']
                )
            );
        }

        return $results;
    }

    /**
     * Validate that a page index is within bounds.
     *
     * @param int $pageIndex The page index to validate
     * @throws InvalidStateException if page index is invalid
     * @internal
     */
    private function validatePageIndex(int $pageIndex): void
    {
        $pageCount = $this->getPageCount();
        if ($pageIndex < 0 || $pageIndex >= $pageCount) {
            $maxPage = $pageCount - 1;
            throw new InvalidStateException(
                "Page index {$pageIndex} out of bounds (0-{$maxPage})",
                ['page_index' => $pageIndex, 'page_count' => $pageCount]
            );
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
                ['file' => $this->filePath]
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
