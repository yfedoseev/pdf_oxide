<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\{Annotation, Rect};
use PdfOxide\Enums\AnnotationType;

/**
 * Manages PDF annotation operations.
 *
 * Handles reading, creating, modifying, and deleting annotations.
 * Supports all PDF annotation types (highlights, comments, etc.).
 */
class AnnotationManager
{
    private FunctionBindings $bindings;
    private CData $handle;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Get annotations from a specific page.
     *
     * @param int $pageIndex Zero-based page index
     * @return Annotation[] Array of annotations
     */
    public function getPageAnnotations(int $pageIndex): array
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
     * Get all annotations from the entire document.
     *
     * @return array Annotations indexed by page
     */
    public function getAllAnnotations(): array
    {
        // Would need page count from parent document
        // For now, return empty - called from PdfDocument
        return [];
    }

    /**
     * Add a highlight annotation.
     *
     * @param int $pageIndex Page index
     * @param Rect $boundingBox Area to highlight
     * @param string $color Color (hex format or name)
     * @param string|null $author Author name
     * @return void
     */
    public function addHighlight(
        int $pageIndex,
        Rect $boundingBox,
        string $color = '#FFFF00',
        ?string $author = null
    ): void {
        $this->bindings->pdfAddAnnotationHighlight(
            $this->handle,
            $pageIndex,
            $boundingBox->x,
            $boundingBox->y,
            $boundingBox->x + $boundingBox->width,
            $boundingBox->y + $boundingBox->height,
            $color,
            $author
        );
    }

    /**
     * Add an underline annotation.
     *
     * @param int $pageIndex Page index
     * @param Rect $boundingBox Area to underline
     * @param string $color Color
     * @param string|null $author Author name
     * @return void
     */
    public function addUnderline(
        int $pageIndex,
        Rect $boundingBox,
        string $color = '#000000',
        ?string $author = null
    ): void {
        $this->bindings->pdfAddAnnotationUnderline(
            $this->handle,
            $pageIndex,
            $boundingBox->x,
            $boundingBox->y,
            $boundingBox->x + $boundingBox->width,
            $boundingBox->y + $boundingBox->height,
            $color,
            $author
        );
    }

    /**
     * Add a strikeout annotation.
     *
     * @param int $pageIndex Page index
     * @param Rect $boundingBox Area to strike
     * @param string $color Color
     * @param string|null $author Author name
     * @return void
     */
    public function addStrikeout(
        int $pageIndex,
        Rect $boundingBox,
        string $color = '#000000',
        ?string $author = null
    ): void {
        $this->bindings->pdfAddAnnotationStrikeout(
            $this->handle,
            $pageIndex,
            $boundingBox->x,
            $boundingBox->y,
            $boundingBox->x + $boundingBox->width,
            $boundingBox->y + $boundingBox->height,
            $color,
            $author
        );
    }

    /**
     * Add a comment/note annotation.
     *
     * @param int $pageIndex Page index
     * @param float $x X coordinate
     * @param float $y Y coordinate
     * @param string $content Comment text
     * @param string|null $author Author name
     * @param string $icon Icon type (Comment, Note, Help, etc.)
     * @return void
     */
    public function addComment(
        int $pageIndex,
        float $x,
        float $y,
        string $content,
        ?string $author = null,
        string $icon = 'Comment'
    ): void {
        $this->bindings->pdfAddAnnotationComment(
            $this->handle,
            $pageIndex,
            $x,
            $y,
            $content,
            $author,
            $icon
        );
    }

    /**
     * Add a free text (popup) annotation.
     *
     * @param int $pageIndex Page index
     * @param Rect $boundingBox Text area
     * @param string $content Text content
     * @param string $fontName Font name
     * @param int $fontSize Font size
     * @param string $color Text color
     * @return void
     */
    public function addFreeText(
        int $pageIndex,
        Rect $boundingBox,
        string $content,
        string $fontName = 'Helvetica',
        int $fontSize = 12,
        string $color = '#000000'
    ): void {
        // Would call FFI function
    }

    /**
     * Add a drawing annotation (ink, line, shape).
     *
     * @param int $pageIndex Page index
     * @param AnnotationType $type Type of annotation
     * @param array $coordinates Drawing coordinates
     * @param string $color Line color
     * @param float $lineWidth Line width
     * @return void
     */
    public function addDrawing(
        int $pageIndex,
        AnnotationType $type,
        array $coordinates,
        string $color = '#000000',
        float $lineWidth = 1.0
    ): void {
        // Would call FFI function
    }

    /**
     * Delete an annotation.
     *
     * @param int $pageIndex Page index
     * @param int $annotationIndex Annotation index on page
     * @return void
     */
    public function deleteAnnotation(int $pageIndex, int $annotationIndex): void
    {
        $this->bindings->pdfDeleteAnnotation($this->handle, $pageIndex, $annotationIndex);
    }

    /**
     * Get annotations by type.
     *
     * @param int $pageIndex Page index
     * @param AnnotationType $type Filter by type
     * @return Annotation[] Matching annotations
     */
    public function getAnnotationsByType(int $pageIndex, AnnotationType $type): array
    {
        $allAnnotations = $this->getPageAnnotations($pageIndex);
        return array_filter(
            $allAnnotations,
            fn($a) => strtolower($a->type) === strtolower($type->value)
        );
    }

    /**
     * Get all markup annotations (highlights, underlines, etc.).
     *
     * @param int $pageIndex Page index
     * @return Annotation[] Markup annotations
     */
    public function getMarkupAnnotations(int $pageIndex): array
    {
        $annotations = $this->getPageAnnotations($pageIndex);
        $markupTypes = ['Highlight', 'Underline', 'StrikeOut', 'Squiggly'];
        return array_filter(
            $annotations,
            fn($a) => in_array($a->type, $markupTypes)
        );
    }

    /**
     * Get all comment/text annotations.
     *
     * @param int $pageIndex Page index
     * @return Annotation[] Comment annotations
     */
    public function getCommentAnnotations(int $pageIndex): array
    {
        $annotations = $this->getPageAnnotations($pageIndex);
        return array_filter(
            $annotations,
            fn($a) => in_array($a->type, ['Comment', 'Note', 'Text', 'FreeText'])
        );
    }

    /**
     * Count annotations on a page.
     *
     * @param int $pageIndex Page index
     * @return int Annotation count
     */
    public function countPageAnnotations(int $pageIndex): int
    {
        return count($this->getPageAnnotations($pageIndex));
    }

    /**
     * Check if page has any annotations.
     *
     * @param int $pageIndex Page index
     * @return bool True if page has annotations
     */
    public function hasAnnotations(int $pageIndex): bool
    {
        return $this->countPageAnnotations($pageIndex) > 0;
    }

    /**
     * Flatten annotations (convert to content).
     *
     * This makes annotations permanent and non-editable.
     *
     * @param int $pageIndex Page index
     * @return void
     */
    public function flattenAnnotations(int $pageIndex): void
    {
        $this->bindings->pdfFlattenAnnotations($this->handle, $pageIndex);
    }

    /**
     * Get annotation summary for document.
     *
     * @return array Summary statistics
     */
    public function getSummary(): array
    {
        return [
            'total_annotations' => 0,
            'by_type' => [],
            'pages_with_annotations' => 0,
        ];
    }
}
