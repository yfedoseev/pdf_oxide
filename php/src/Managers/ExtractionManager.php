<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\{Font, Image, Annotation};

/**
 * Manages content extraction from PDF pages.
 *
 * Handles extraction of fonts, images, annotations, and other page resources.
 */
class ExtractionManager
{
    private FunctionBindings $bindings;
    private CData $handle;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Extract all fonts from a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return Font[] Array of fonts found on the page
     */
    public function fonts(int $pageIndex): array
    {
        $fontsHandle = $this->bindings->pdfDocumentGetEmbeddedFonts($this->handle, $pageIndex);

        try {
            $fonts = [];
            $count = $this->bindings->pdfOxideFontCount($fontsHandle);

            for ($i = 0; $i < $count; $i++) {
                $fonts[] = new Font(
                    $this->bindings->pdfOxideFontGetName($fontsHandle, $i),
                    $this->bindings->pdfOxideFontGetType($fontsHandle, $i),
                    $this->bindings->pdfOxideFontIsEmbedded($fontsHandle, $i),
                    $this->bindings->pdfOxideFontGetEncoding($fontsHandle, $i),
                    $this->bindings->pdfOxideFontIsSubset($fontsHandle, $i),
                    $this->bindings->pdfOxideFontGetSize($fontsHandle, $i)
                );
            }

            return $fonts;
        } finally {
            $this->bindings->pdfOxideFontListFree($fontsHandle);
        }
    }

    /**
     * Extract all images from a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return Image[] Array of images found on the page
     */
    public function images(int $pageIndex): array
    {
        $imagesHandle = $this->bindings->pdfDocumentGetEmbeddedImages($this->handle, $pageIndex);

        try {
            $images = [];
            $count = $this->bindings->pdfOxideImageCount($imagesHandle);

            for ($i = 0; $i < $count; $i++) {
                $images[] = new Image(
                    $this->bindings->pdfOxideImageGetFormat($imagesHandle, $i),
                    $this->bindings->pdfOxideImageGetWidth($imagesHandle, $i),
                    $this->bindings->pdfOxideImageGetHeight($imagesHandle, $i),
                    $this->bindings->pdfOxideImageGetColorspace($imagesHandle, $i),
                    $this->bindings->pdfOxideImageGetBitsPerComponent($imagesHandle, $i)
                );
            }

            return $images;
        } finally {
            $this->bindings->pdfOxideImageListFree($imagesHandle);
        }
    }

    /**
     * Extract all annotations from a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return Annotation[] Array of annotations on the page
     */
    public function annotations(int $pageIndex): array
    {
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
     * Extract fonts from all pages.
     *
     * @return array Fonts indexed by page
     */
    public function allFonts(): array
    {
        // This would require knowing page count from parent document
        // For now, returns empty - would be called from PdfDocument
        return [];
    }

    /**
     * Extract images from all pages.
     *
     * @return array Images indexed by page
     */
    public function allImages(): array
    {
        return [];
    }

    /**
     * Extract annotations from all pages.
     *
     * @return array Annotations indexed by page
     */
    public function allAnnotations(): array
    {
        return [];
    }

    /**
     * Get unique font names from a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return string[] Array of unique font names
     */
    public function uniqueFontNames(int $pageIndex): array
    {
        $fonts = $this->fonts($pageIndex);
        return array_unique(array_map(fn($f) => $f->name, $fonts));
    }

    /**
     * Get embedded fonts count on a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return int Number of embedded fonts
     */
    public function embeddedFontCount(int $pageIndex): int
    {
        $fonts = $this->fonts($pageIndex);
        return count(array_filter($fonts, fn($f) => $f->embedded));
    }

    /**
     * Get total image pixel area on a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return int Total pixel area (width * height)
     */
    public function totalImagePixelArea(int $pageIndex): int
    {
        $images = $this->images($pageIndex);
        return array_sum(array_map(fn($i) => $i->width * $i->height, $images));
    }
}
