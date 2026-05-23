<?php

declare(strict_types=1);

namespace PdfOxide\FFI;

use FFI;
use FFI\CData;

/**
 * Type-safe wrappers for all FFI function calls.
 *
 * Provides a PHP interface to the Rust FFI layer.
 */
class FunctionBindings
{
    private FFI $ffi;

    public function __construct()
    {
        $this->ffi = NativeLibrary::getInstance();
    }

    /**
     * Open a PDF document.
     *
     * @param string $path Path to the PDF file
     * @return CData|null Document handle or null on error
     */
    public function pdfDocumentOpen(string $path): ?CData
    {
        $cPath = StringMarshaller::toCString($path);
        $errorCode = FFI::new('int');

        try {
            $handle = $this->ffi->pdf_document_open($cPath, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_open', ['path' => $path]);
            return $handle;
        } finally {
            unset($cPath);
        }
    }

    /**
     * Free a document handle.
     *
     * @param CData $handle The document handle to free
     */
    public function pdfDocumentFree(CData $handle): void
    {
        $this->ffi->pdf_document_free($handle);
    }

    /**
     * Get PDF version.
     *
     * @param CData $handle The document handle
     * @return array [major, minor] version numbers
     */
    public function pdfDocumentGetVersion(CData $handle): array
    {
        $major = FFI::new('uint8_t');
        $minor = FFI::new('uint8_t');

        $this->ffi->pdf_document_get_version($handle, FFI::addr($major), FFI::addr($minor));

        return [
            'major' => (int)$major->cdata,
            'minor' => (int)$minor->cdata,
        ];
    }

    /**
     * Get page count.
     *
     * @param CData $handle The document handle
     * @return int Number of pages
     */
    public function pdfDocumentGetPageCount(CData $handle): int
    {
        $errorCode = FFI::new('int');
        $count = $this->ffi->pdf_document_get_page_count($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_page_count');
        return (int)$count;
    }

    /**
     * Check if document has structure tree.
     *
     * @param CData $handle The document handle
     * @return bool True if document has structure tree
     */
    public function pdfDocumentHasStructureTree(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_has_structure_tree($handle);
    }

    /**
     * Extract text from a page.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @return string Extracted text
     */
    public function pdfDocumentExtractText(CData $handle, int $pageIndex): string
    {
        $errorCode = FFI::new('int');
        $text = $this->ffi->pdf_document_extract_text($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_extract_text', ['page' => $pageIndex]);
        return StringMarshaller::fromCString($text);
    }

    /**
     * Convert page to Markdown.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @return string Markdown text
     */
    public function pdfDocumentToMarkdown(CData $handle, int $pageIndex): string
    {
        $errorCode = FFI::new('int');
        $markdown = $this->ffi->pdf_document_to_markdown($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_to_markdown', ['page' => $pageIndex]);
        return StringMarshaller::fromCString($markdown);
    }

    /**
     * Convert entire document to Markdown.
     *
     * @param CData $handle The document handle
     * @return string Markdown text
     */
    public function pdfDocumentToMarkdownAll(CData $handle): string
    {
        $errorCode = FFI::new('int');
        $markdown = $this->ffi->pdf_document_to_markdown_all($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_to_markdown_all');
        return StringMarshaller::fromCString($markdown);
    }

    /**
     * Convert page to HTML.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @return string HTML content
     */
    public function pdfDocumentToHtml(CData $handle, int $pageIndex): string
    {
        $errorCode = FFI::new('int');
        $html = $this->ffi->pdf_document_to_html($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_to_html', ['page' => $pageIndex]);
        return StringMarshaller::fromCString($html);
    }

    /**
     * Convert page to plain text.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @return string Plain text
     */
    public function pdfDocumentToPlainText(CData $handle, int $pageIndex): string
    {
        $errorCode = FFI::new('int');
        $text = $this->ffi->pdf_document_to_plain_text($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_to_plain_text', ['page' => $pageIndex]);
        return StringMarshaller::fromCString($text);
    }

    /**
     * Search in a specific page.
     *
     * @param CData $handle The document handle
     * @param string $searchTerm The text to search for
     * @param int $pageIndex Zero-based page index
     * @param bool $caseSensitive Whether search is case-sensitive
     * @return CData Search results handle
     */
    public function pdfDocumentSearchPage(
        CData $handle,
        string $searchTerm,
        int $pageIndex,
        bool $caseSensitive = false
    ): CData {
        $cTerm = StringMarshaller::toCString($searchTerm);
        $errorCode = FFI::new('int');

        try {
            $results = $this->ffi->pdf_document_search_page(
                $handle,
                $cTerm,
                $pageIndex,
                $caseSensitive ? 1 : 0,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_search_page', [
                'term' => $searchTerm,
                'page' => $pageIndex,
            ]);
            return $results;
        } finally {
            unset($cTerm);
        }
    }

    /**
     * Search entire document.
     *
     * @param CData $handle The document handle
     * @param string $searchTerm The text to search for
     * @param bool $caseSensitive Whether search is case-sensitive
     * @return CData Search results handle
     */
    public function pdfDocumentSearchAll(
        CData $handle,
        string $searchTerm,
        bool $caseSensitive = false
    ): CData {
        $cTerm = StringMarshaller::toCString($searchTerm);
        $errorCode = FFI::new('int');

        try {
            $results = $this->ffi->pdf_document_search_all(
                $handle,
                $cTerm,
                $caseSensitive ? 1 : 0,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_search_all', [
                'term' => $searchTerm,
            ]);
            return $results;
        } finally {
            unset($cTerm);
        }
    }

    /**
     * Get embedded fonts from a page.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @return CData Font list handle
     */
    public function pdfDocumentGetEmbeddedFonts(CData $handle, int $pageIndex): CData
    {
        $errorCode = FFI::new('int');
        $fonts = $this->ffi->pdf_document_get_embedded_fonts($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_embedded_fonts', ['page' => $pageIndex]);
        return $fonts;
    }

    /**
     * Get embedded images from a page.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @return CData Image list handle
     */
    public function pdfDocumentGetEmbeddedImages(CData $handle, int $pageIndex): CData
    {
        $errorCode = FFI::new('int');
        $images = $this->ffi->pdf_document_get_embedded_images($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_embedded_images', ['page' => $pageIndex]);
        return $images;
    }

    /**
     * Get annotations from a page.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @return CData Annotation list handle
     */
    public function pdfDocumentGetAnnotations(CData $handle, int $pageIndex): CData
    {
        $errorCode = FFI::new('int');
        $annotations = $this->ffi->pdf_document_get_annotations($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_annotations', ['page' => $pageIndex]);
        return $annotations;
    }

    /**
     * Get search result count.
     *
     * @param CData $resultsHandle The search results handle
     * @return int Number of results
     */
    public function oxideSearchResultCount(CData $resultsHandle): int
    {
        return (int)$this->ffi->pdf_oxide_search_result_count($resultsHandle);
    }

    /**
     * Get search result text.
     *
     * @param CData $resultsHandle The search results handle
     * @param int $index Result index
     * @return string The result text
     */
    public function oxideSearchResultGetText(CData $resultsHandle, int $index): string
    {
        $errorCode = FFI::new('int');
        $text = $this->ffi->pdf_oxide_search_result_get_text($resultsHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_oxide_search_result_get_text', ['index' => $index]);
        return StringMarshaller::fromCString($text);
    }

    /**
     * Get search result page number.
     *
     * @param CData $resultsHandle The search results handle
     * @param int $index Result index
     * @return int The page number
     */
    public function oxideSearchResultGetPage(CData $resultsHandle, int $index): int
    {
        return (int)$this->ffi->pdf_oxide_search_result_get_page($resultsHandle, $index);
    }

    /**
     * Get search result position.
     *
     * @param CData $resultsHandle The search results handle
     * @param int $index Result index
     * @return int The character position
     */
    public function oxideSearchResultGetPosition(CData $resultsHandle, int $index): int
    {
        return (int)$this->ffi->pdf_oxide_search_result_get_position($resultsHandle, $index);
    }

    /**
     * Get search result bounding box.
     *
     * @param CData $resultsHandle The search results handle
     * @param int $index Result index
     * @return array [x, y, width, height] coordinates
     */
    public function oxideSearchResultGetBbox(CData $resultsHandle, int $index): array
    {
        $x = FFI::new('float');
        $y = FFI::new('float');
        $width = FFI::new('float');
        $height = FFI::new('float');

        $this->ffi->pdf_oxide_search_result_get_bbox(
            $resultsHandle,
            $index,
            FFI::addr($x),
            FFI::addr($y),
            FFI::addr($width),
            FFI::addr($height)
        );

        return [
            'x' => (float)$x->cdata,
            'y' => (float)$y->cdata,
            'width' => (float)$width->cdata,
            'height' => (float)$height->cdata,
        ];
    }

    /**
     * Free search results.
     *
     * @param CData $resultsHandle The search results handle
     */
    public function oxideSearchResultFree(CData $resultsHandle): void
    {
        $this->ffi->pdf_oxide_search_result_free($resultsHandle);
    }

    /**
     * Get annotation count.
     *
     * @param CData $listHandle The annotation list handle
     * @return int Number of annotations
     */
    public function oxideAnnotationCount(CData $listHandle): int
    {
        return (int)$this->ffi->pdf_oxide_annotation_count($listHandle);
    }

    /**
     * Get annotation type.
     *
     * @param CData $listHandle The annotation list handle
     * @param int $index Annotation index
     * @return string The annotation type
     */
    public function oxideAnnotationGetType(CData $listHandle, int $index): string
    {
        $type = $this->ffi->pdf_oxide_annotation_get_type($listHandle, $index);
        return StringMarshaller::fromCString($type, false);
    }

    /**
     * Get annotation content.
     *
     * @param CData $listHandle The annotation list handle
     * @param int $index Annotation index
     * @return string The annotation content
     */
    public function oxideAnnotationGetContent(CData $listHandle, int $index): string
    {
        $content = $this->ffi->pdf_oxide_annotation_get_content($listHandle, $index);
        return StringMarshaller::fromCString($content, false);
    }

    /**
     * Free annotation list.
     *
     * @param CData $listHandle The annotation list handle
     */
    public function oxideAnnotationFree(CData $listHandle): void
    {
        $this->ffi->pdf_oxide_annotation_free($listHandle);
    }

    /**
     * Get font count.
     *
     * @param CData $listHandle The font list handle
     * @return int Number of fonts
     */
    public function oxideFontCount(CData $listHandle): int
    {
        return (int)$this->ffi->pdf_oxide_font_count($listHandle);
    }

    /**
     * Get font name.
     *
     * @param CData $listHandle The font list handle
     * @param int $index Font index
     * @return string The font name
     */
    public function oxideFontGetName(CData $listHandle, int $index): string
    {
        $name = $this->ffi->pdf_oxide_font_get_name($listHandle, $index);
        return StringMarshaller::fromCString($name, false);
    }

    /**
     * Get font type.
     *
     * @param CData $listHandle The font list handle
     * @param int $index Font index
     * @return string The font type
     */
    public function oxideFontGetType(CData $listHandle, int $index): string
    {
        $type = $this->ffi->pdf_oxide_font_get_type($listHandle, $index);
        return StringMarshaller::fromCString($type, false);
    }

    /**
     * Check if font is embedded.
     *
     * @param CData $listHandle The font list handle
     * @param int $index Font index
     * @return bool True if font is embedded
     */
    public function oxideFontIsEmbedded(CData $listHandle, int $index): bool
    {
        return (bool)$this->ffi->pdf_oxide_font_is_embedded($listHandle, $index);
    }

    /**
     * Free font list.
     *
     * @param CData $listHandle The font list handle
     */
    public function oxideFontFree(CData $listHandle): void
    {
        $this->ffi->pdf_oxide_font_free($listHandle);
    }

    /**
     * Get image count.
     *
     * @param CData $listHandle The image list handle
     * @return int Number of images
     */
    public function oxideImageCount(CData $listHandle): int
    {
        return (int)$this->ffi->pdf_oxide_image_count($listHandle);
    }

    /**
     * Get image width.
     *
     * @param CData $listHandle The image list handle
     * @param int $index Image index
     * @return int Image width
     */
    public function oxideImageGetWidth(CData $listHandle, int $index): int
    {
        return (int)$this->ffi->pdf_oxide_image_get_width($listHandle, $index);
    }

    /**
     * Get image height.
     *
     * @param CData $listHandle The image list handle
     * @param int $index Image index
     * @return int Image height
     */
    public function oxideImageGetHeight(CData $listHandle, int $index): int
    {
        return (int)$this->ffi->pdf_oxide_image_get_height($listHandle, $index);
    }

    /**
     * Get image format.
     *
     * @param CData $listHandle The image list handle
     * @param int $index Image index
     * @return string Image format
     */
    public function oxideImageGetFormat(CData $listHandle, int $index): string
    {
        $format = $this->ffi->pdf_oxide_image_get_format($listHandle, $index);
        return StringMarshaller::fromCString($format, false);
    }

    /**
     * Free image list.
     *
     * @param CData $listHandle The image list handle
     */
    public function oxideImageFree(CData $listHandle): void
    {
        $this->ffi->pdf_oxide_image_free($listHandle);
    }

    /**
     * Render a page to an image.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @param CData|null $options Rendering options (NULL for defaults)
     * @return CData Image handle
     */
    public function pdfRenderPage(CData $handle, int $pageIndex, ?CData $options = null): CData
    {
        $errorCode = FFI::new('int');
        $imageHandle = $this->ffi->pdf_render_page($handle, $pageIndex, $options, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_render_page');
        return $imageHandle;
    }

    /**
     * Render a page to a file.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @param string $filePath Output file path
     * @param CData|null $options Rendering options
     */
    public function pdfRenderPageToFile(CData $handle, int $pageIndex, string $filePath, ?CData $options = null): void
    {
        $cPath = StringMarshaller::toCString($filePath);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_render_page_to_file($handle, $pageIndex, $cPath, $options, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_render_page_to_file');
        } finally {
            unset($cPath);
        }
    }

    /**
     * Render a page range to files.
     *
     * @param CData $handle The document handle
     * @param int $startPage Start page index
     * @param int $endPage End page index
     * @param string $filePrefix Output file prefix
     * @param CData|null $options Rendering options
     * @return int Number of pages rendered
     */
    public function pdfRenderPageRange(CData $handle, int $startPage, int $endPage, string $filePrefix, ?CData $options = null): int
    {
        $cPrefix = StringMarshaller::toCString($filePrefix);
        $errorCode = FFI::new('int');

        try {
            $count = $this->ffi->pdf_render_page_range($handle, $startPage, $endPage, $cPrefix, $options, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_render_page_range');
            return (int)$count;
        } finally {
            unset($cPrefix);
        }
    }

    /**
     * Render entire document to files.
     *
     * @param CData $handle The document handle
     * @param string $filePrefix Output file prefix
     * @param CData|null $options Rendering options
     * @return int Total number of pages rendered
     */
    public function pdfRenderDocument(CData $handle, string $filePrefix, ?CData $options = null): int
    {
        $cPrefix = StringMarshaller::toCString($filePrefix);
        $errorCode = FFI::new('int');

        try {
            $count = $this->ffi->pdf_render_document($handle, $cPrefix, $options, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_render_document');
            return (int)$count;
        } finally {
            unset($cPrefix);
        }
    }

    /**
     * Render a page region (crop).
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @param float $x Crop region X coordinate
     * @param float $y Crop region Y coordinate
     * @param float $width Crop region width
     * @param float $height Crop region height
     * @param CData|null $options Rendering options
     * @return CData Image handle
     */
    public function pdfRenderPageRegion(CData $handle, int $pageIndex, float $x, float $y, float $width, float $height, ?CData $options = null): CData
    {
        $errorCode = FFI::new('int');
        $imageHandle = $this->ffi->pdf_render_page_region($handle, $pageIndex, $x, $y, $width, $height, $options, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_render_page_region');
        return $imageHandle;
    }

    /**
     * Render a page with zoom.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @param float $zoomLevel Zoom level (1.0 = 100%)
     * @param CData|null $options Rendering options
     * @return CData Image handle
     */
    public function pdfRenderPageZoom(CData $handle, int $pageIndex, float $zoomLevel, ?CData $options = null): CData
    {
        $errorCode = FFI::new('int');
        $imageHandle = $this->ffi->pdf_render_page_zoom($handle, $pageIndex, $zoomLevel, $options, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_render_page_zoom');
        return $imageHandle;
    }

    /**
     * Render a page fitted to specific dimensions.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @param int $fitWidth Target width in pixels
     * @param int $fitHeight Target height in pixels
     * @param CData|null $options Rendering options
     * @return CData Image handle
     */
    public function pdfRenderPageFit(CData $handle, int $pageIndex, int $fitWidth, int $fitHeight, ?CData $options = null): CData
    {
        $errorCode = FFI::new('int');
        $imageHandle = $this->ffi->pdf_render_page_fit($handle, $pageIndex, $fitWidth, $fitHeight, $options, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_render_page_fit');
        return $imageHandle;
    }

    /**
     * Render a page thumbnail.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @param int $maxSize Maximum width/height in pixels
     * @param CData|null $options Rendering options
     * @return CData Image handle
     */
    public function pdfRenderPageThumbnail(CData $handle, int $pageIndex, int $maxSize, ?CData $options = null): CData
    {
        $errorCode = FFI::new('int');
        $imageHandle = $this->ffi->pdf_render_page_thumbnail($handle, $pageIndex, $maxSize, $options, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_render_page_thumbnail');
        return $imageHandle;
    }

    /**
     * Get rendered image width.
     *
     * @param CData $imageHandle The image handle
     * @return int Image width in pixels
     */
    public function pdfRenderedImageWidth(CData $imageHandle): int
    {
        return (int)$this->ffi->pdf_rendered_image_width($imageHandle);
    }

    /**
     * Get rendered image height.
     *
     * @param CData $imageHandle The image handle
     * @return int Image height in pixels
     */
    public function pdfRenderedImageHeight(CData $imageHandle): int
    {
        return (int)$this->ffi->pdf_rendered_image_height($imageHandle);
    }

    /**
     * Get rendered image format.
     *
     * @param CData $imageHandle The image handle
     * @return string Image format (png, jpeg, webp)
     */
    public function pdfRenderedImageFormat(CData $imageHandle): string
    {
        $format = $this->ffi->pdf_rendered_image_format($imageHandle);
        return StringMarshaller::fromCString($format, false);
    }

    /**
     * Get rendered image data size.
     *
     * @param CData $imageHandle The image handle
     * @return int Data size in bytes
     */
    public function pdfRenderedImageSize(CData $imageHandle): int
    {
        return (int)$this->ffi->pdf_rendered_image_size($imageHandle);
    }

    /**
     * Get rendered image data.
     *
     * @param CData $imageHandle The image handle
     * @return string Binary image data
     */
    public function pdfRenderedImageData(CData $imageHandle): string
    {
        $dataPtr = $this->ffi->pdf_rendered_image_data($imageHandle);
        $size = $this->pdfRenderedImageSize($imageHandle);
        return FFI::string($dataPtr, $size);
    }

    /**
     * Save rendered image to file.
     *
     * @param CData $imageHandle The image handle
     * @param string $filePath Output file path
     */
    public function pdfRenderedImageSave(CData $imageHandle, string $filePath): void
    {
        $cPath = StringMarshaller::toCString($filePath);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_rendered_image_save($imageHandle, $cPath, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_rendered_image_save');
        } finally {
            unset($cPath);
        }
    }

    /**
     * Convert rendered image to different format.
     *
     * @param CData $imageHandle The image handle
     * @param string $newFormat New format (png, jpeg, webp)
     * @return CData New image handle
     */
    public function pdfRenderedImageConvert(CData $imageHandle, string $newFormat): CData
    {
        $cFormat = StringMarshaller::toCString($newFormat);
        $errorCode = FFI::new('int');

        try {
            $newHandle = $this->ffi->pdf_rendered_image_convert($imageHandle, $cFormat, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_rendered_image_convert');
            return $newHandle;
        } finally {
            unset($cFormat);
        }
    }

    /**
     * Free rendered image.
     *
     * @param CData $imageHandle The image handle
     */
    public function pdfRenderedImageFree(CData $imageHandle): void
    {
        $this->ffi->pdf_rendered_image_free($imageHandle);
    }

    /**
     * Estimate rendering time for a page.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @param CData|null $options Rendering options
     * @return int Estimated time in milliseconds
     */
    public function pdfEstimateRenderTime(CData $handle, int $pageIndex, ?CData $options = null): int
    {
        return (int)$this->ffi->pdf_estimate_render_time($handle, $pageIndex, $options);
    }

    /**
     * Get renderer statistics.
     *
     * @param CData $handle The document handle
     * @return array Statistics array with 'pages_rendered', 'total_time_ms', 'avg_time_ms'
     */
    public function pdfRendererGetStatistics(CData $handle): array
    {
        $pagesRendered = FFI::new('int');
        $totalTimeMs = FFI::new('long');
        $avgTimeMs = FFI::new('long');

        $this->ffi->pdf_renderer_get_statistics($handle, FFI::addr($pagesRendered), FFI::addr($totalTimeMs), FFI::addr($avgTimeMs));

        return [
            'pages_rendered' => (int)$pagesRendered->cdata,
            'total_time_ms' => (int)$totalTimeMs->cdata,
            'avg_time_ms' => (int)$avgTimeMs->cdata,
        ];
    }

    /**
     * Reset renderer statistics.
     *
     * @param CData $handle The document handle
     */
    public function pdfRendererResetStatistics(CData $handle): void
    {
        $this->ffi->pdf_renderer_reset_statistics($handle);
    }

    /**
     * Check if document has form fields.
     *
     * @param CData $handle The document handle
     * @return bool True if document has form fields
     */
    public function pdfDocumentHasFormFields(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_has_form_fields($handle);
    }

    /**
     * Get form field count.
     *
     * @param CData $handle The document handle
     * @return int Number of form fields
     */
    public function pdfDocumentGetFormFieldCount(CData $handle): int
    {
        $errorCode = FFI::new('int');
        $count = $this->ffi->pdf_document_get_form_field_count($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_form_field_count');
        return (int)$count;
    }

    /**
     * Get form field by index.
     *
     * @param CData $handle The document handle
     * @param int $index Field index
     * @return CData Form field handle
     */
    public function pdfDocumentGetFormField(CData $handle, int $index): CData
    {
        $errorCode = FFI::new('int');
        $fieldHandle = $this->ffi->pdf_document_get_form_field($handle, $index, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_form_field');
        return $fieldHandle;
    }

    /**
     * Get form field by name.
     *
     * @param CData $handle The document handle
     * @param string $fieldName Field name
     * @return CData|null Form field handle or null if not found
     */
    public function pdfDocumentFindFormField(CData $handle, string $fieldName): ?CData
    {
        $cName = StringMarshaller::toCString($fieldName);
        $errorCode = FFI::new('int');

        try {
            $fieldHandle = $this->ffi->pdf_document_find_form_field($handle, $cName, FFI::addr($errorCode));
            if ($errorCode->cdata === 0) {
                return $fieldHandle;
            }
            return null;
        } finally {
            unset($cName);
        }
    }

    /**
     * Get form field name.
     *
     * @param CData $fieldHandle The form field handle
     * @return string Field name
     */
    public function pdfFormFieldGetName(CData $fieldHandle): string
    {
        $name = $this->ffi->pdf_form_field_get_name($fieldHandle);
        return StringMarshaller::fromCString($name, false);
    }

    /**
     * Get form field type.
     *
     * @param CData $fieldHandle The form field handle
     * @return string Field type
     */
    public function pdfFormFieldGetType(CData $fieldHandle): string
    {
        $type = $this->ffi->pdf_form_field_get_type($fieldHandle);
        return StringMarshaller::fromCString($type, false);
    }

    /**
     * Get form field value.
     *
     * @param CData $fieldHandle The form field handle
     * @return string Field value
     */
    public function pdfFormFieldGetValue(CData $fieldHandle): string
    {
        $value = $this->ffi->pdf_form_field_get_value($fieldHandle);
        return StringMarshaller::fromCString($value, false);
    }

    /**
     * Set form field value.
     *
     * @param CData $fieldHandle The form field handle
     * @param string $value New value
     */
    public function pdfFormFieldSetValue(CData $fieldHandle, string $value): void
    {
        $cValue = StringMarshaller::toCString($value);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_form_field_set_value($fieldHandle, $cValue, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_form_field_set_value');
        } finally {
            unset($cValue);
        }
    }

    /**
     * Check if form field is required.
     *
     * @param CData $fieldHandle The form field handle
     * @return bool True if field is required
     */
    public function pdfFormFieldIsRequired(CData $fieldHandle): bool
    {
        return (bool)$this->ffi->pdf_form_field_is_required($fieldHandle);
    }

    /**
     * Get form field options count.
     *
     * @param CData $fieldHandle The form field handle
     * @return int Number of options
     */
    public function pdfFormFieldGetOptionsCount(CData $fieldHandle): int
    {
        return (int)$this->ffi->pdf_form_field_get_options_count($fieldHandle);
    }

    /**
     * Get form field option by index.
     *
     * @param CData $fieldHandle The form field handle
     * @param int $index Option index
     * @return string Option value
     */
    public function pdfFormFieldGetOption(CData $fieldHandle, int $index): string
    {
        $option = $this->ffi->pdf_form_field_get_option($fieldHandle, $index);
        return StringMarshaller::fromCString($option, false);
    }

    /**
     * Get form field page index.
     *
     * @param CData $fieldHandle The form field handle
     * @return int Page index
     */
    public function pdfFormFieldGetPageIndex(CData $fieldHandle): int
    {
        return (int)$this->ffi->pdf_form_field_get_page_index($fieldHandle);
    }

    /**
     * Free form field handle.
     *
     * @param CData $fieldHandle The form field handle
     */
    public function pdfFormFieldFree(CData $fieldHandle): void
    {
        $this->ffi->pdf_form_field_free($fieldHandle);
    }

    /**
     * Check if document has XFA form.
     *
     * @param CData $handle The document handle
     * @return bool True if XFA form exists
     */
    public function pdfDocumentHasXfaForm(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_has_xfa_form($handle);
    }

    /**
     * Convert XFA form to AcroForm.
     *
     * @param CData $handle The document handle
     */
    public function pdfDocumentConvertXfaToAcroForm(CData $handle): void
    {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_document_convert_xfa_to_acroform($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_convert_xfa_to_acroform');
    }

    /**
     * Add highlight annotation.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Page index
     * @param float $x1 Top-left X coordinate
     * @param float $y1 Top-left Y coordinate
     * @param float $x2 Bottom-right X coordinate
     * @param float $y2 Bottom-right Y coordinate
     * @param string $color Color (hex or name)
     * @param string|null $author Author name
     */
    public function pdfAddAnnotationHighlight(CData $handle, int $pageIndex, float $x1, float $y1, float $x2, float $y2, string $color, ?string $author = null): void
    {
        $cColor = StringMarshaller::toCString($color);
        $cAuthor = $author ? StringMarshaller::toCString($author) : null;
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_add_annotation_highlight($handle, $pageIndex, $x1, $y1, $x2, $y2, $cColor, $cAuthor, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_add_annotation_highlight');
        } finally {
            unset($cColor, $cAuthor);
        }
    }

    /**
     * Add underline annotation.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Page index
     * @param float $x1 Top-left X coordinate
     * @param float $y1 Top-left Y coordinate
     * @param float $x2 Bottom-right X coordinate
     * @param float $y2 Bottom-right Y coordinate
     * @param string $color Color (hex or name)
     * @param string|null $author Author name
     */
    public function pdfAddAnnotationUnderline(CData $handle, int $pageIndex, float $x1, float $y1, float $x2, float $y2, string $color, ?string $author = null): void
    {
        $cColor = StringMarshaller::toCString($color);
        $cAuthor = $author ? StringMarshaller::toCString($author) : null;
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_add_annotation_underline($handle, $pageIndex, $x1, $y1, $x2, $y2, $cColor, $cAuthor, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_add_annotation_underline');
        } finally {
            unset($cColor, $cAuthor);
        }
    }

    /**
     * Add strikeout annotation.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Page index
     * @param float $x1 Top-left X coordinate
     * @param float $y1 Top-left Y coordinate
     * @param float $x2 Bottom-right X coordinate
     * @param float $y2 Bottom-right Y coordinate
     * @param string $color Color (hex or name)
     * @param string|null $author Author name
     */
    public function pdfAddAnnotationStrikeout(CData $handle, int $pageIndex, float $x1, float $y1, float $x2, float $y2, string $color, ?string $author = null): void
    {
        $cColor = StringMarshaller::toCString($color);
        $cAuthor = $author ? StringMarshaller::toCString($author) : null;
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_add_annotation_strikeout($handle, $pageIndex, $x1, $y1, $x2, $y2, $cColor, $cAuthor, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_add_annotation_strikeout');
        } finally {
            unset($cColor, $cAuthor);
        }
    }

    /**
     * Add comment annotation.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Page index
     * @param float $x X coordinate
     * @param float $y Y coordinate
     * @param string $content Comment text
     * @param string|null $author Author name
     * @param string $icon Icon type
     */
    public function pdfAddAnnotationComment(CData $handle, int $pageIndex, float $x, float $y, string $content, ?string $author = null, string $icon = 'Comment'): void
    {
        $cContent = StringMarshaller::toCString($content);
        $cAuthor = $author ? StringMarshaller::toCString($author) : null;
        $cIcon = StringMarshaller::toCString($icon);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_add_annotation_comment($handle, $pageIndex, $x, $y, $cContent, $cAuthor, $cIcon, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_add_annotation_comment');
        } finally {
            unset($cContent, $cAuthor, $cIcon);
        }
    }

    /**
     * Delete annotation.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Page index
     * @param int $annotationIndex Annotation index
     */
    public function pdfDeleteAnnotation(CData $handle, int $pageIndex, int $annotationIndex): void
    {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_delete_annotation($handle, $pageIndex, $annotationIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_delete_annotation');
    }

    /**
     * Flatten annotations.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Page index
     */
    public function pdfFlattenAnnotations(CData $handle, int $pageIndex): void
    {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_flatten_annotations($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_flatten_annotations');
    }

    /**
     * Generate a QR code.
     *
     * @param string $data Data to encode in QR code
     * @param int $size Size (1-40)
     * @return CData Barcode handle
     */
    public function pdfGenerateQrCode(string $data, int $size = 10): CData
    {
        $cData = StringMarshaller::toCString($data);
        $errorCode = FFI::new('int');

        try {
            $barcodeHandle = $this->ffi->pdf_generate_qr_code($cData, $size, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_generate_qr_code');
            return $barcodeHandle;
        } finally {
            unset($cData);
        }
    }

    /**
     * Generate a barcode.
     *
     * @param string $data Data to encode
     * @param string $format Barcode format (EAN13, UPC_A, CODE128, etc.)
     * @return CData Barcode handle
     */
    public function pdfGenerateBarcode(string $data, string $format): CData
    {
        $cData = StringMarshaller::toCString($data);
        $cFormat = StringMarshaller::toCString($format);
        $errorCode = FFI::new('int');

        try {
            $barcodeHandle = $this->ffi->pdf_generate_barcode($cData, $cFormat, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_generate_barcode');
            return $barcodeHandle;
        } finally {
            unset($cData, $cFormat);
        }
    }

    /**
     * Get barcode image as PNG.
     *
     * @param CData $barcodeHandle Barcode handle
     * @return string PNG binary data
     */
    public function pdfBarcodeGetImagePng(CData $barcodeHandle): string
    {
        $sizePtr = FFI::new('int');
        $dataPtr = $this->ffi->pdf_barcode_get_image_png($barcodeHandle, FFI::addr($sizePtr));
        $size = (int)$sizePtr->cdata;
        return FFI::string($dataPtr, $size);
    }

    /**
     * Get barcode as SVG.
     *
     * @param CData $barcodeHandle Barcode handle
     * @return string SVG XML string
     */
    public function pdfBarcodeGetSvg(CData $barcodeHandle): string
    {
        $svg = $this->ffi->pdf_barcode_get_svg($barcodeHandle);
        return StringMarshaller::fromCString($svg, false);
    }

    /**
     * Add barcode to page.
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Page index
     * @param CData $barcodeHandle Barcode handle
     * @param float $x X coordinate
     * @param float $y Y coordinate
     * @param float $width Width
     * @param float $height Height
     */
    public function pdfAddBarcodeToPage(CData $handle, int $pageIndex, CData $barcodeHandle, float $x, float $y, float $width, float $height): void
    {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_add_barcode_to_page($handle, $pageIndex, $barcodeHandle, $x, $y, $width, $height, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_add_barcode_to_page');
    }

    /**
     * Free barcode handle.
     *
     * @param CData $barcodeHandle The barcode handle
     */
    public function pdfBarcodeFree(CData $barcodeHandle): void
    {
        $this->ffi->pdf_barcode_free($barcodeHandle);
    }

    /**
     * Create an OCR engine.
     *
     * @return CData OCR engine handle
     */
    public function pdfOcrEngineCreate(): CData
    {
        $errorCode = FFI::new('int');
        $engine = $this->ffi->pdf_ocr_engine_create(FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_engine_create');
        return $engine;
    }

    /**
     * Free OCR engine.
     *
     * @param CData $engine OCR engine handle
     */
    public function pdfOcrEngineFree(CData $engine): void
    {
        $this->ffi->pdf_ocr_engine_free($engine);
    }

    /**
     * Get OCR engine version.
     *
     * @param CData $engine OCR engine handle
     * @return string Version string
     */
    public function pdfOcrEngineGetVersion(CData $engine): string
    {
        $version = $this->ffi->pdf_ocr_engine_get_version($engine);
        return StringMarshaller::fromCString($version, false);
    }

    /**
     * Get OCR engine status.
     *
     * @param CData $engine OCR engine handle
     * @return string Status string
     */
    public function pdfOcrEngineGetStatus(CData $engine): string
    {
        $status = $this->ffi->pdf_ocr_engine_get_status($engine);
        return StringMarshaller::fromCString($status, false);
    }

    /**
     * Check if page needs OCR.
     *
     * @param CData $handle Document handle
     * @param int $pageIndex Page index
     * @return bool True if page needs OCR
     */
    public function pdfOcrPageNeedsOcr(CData $handle, int $pageIndex): bool
    {
        return (bool)$this->ffi->pdf_ocr_page_needs_ocr($handle, $pageIndex);
    }

    /**
     * Detect page for OCR (analyze without recognizing).
     *
     * @param CData $engine OCR engine handle
     * @param CData $handle Document handle
     * @param int $pageIndex Page index
     * @return CData OCR results handle
     */
    public function pdfOcrDetectPage(CData $engine, CData $handle, int $pageIndex): CData
    {
        $errorCode = FFI::new('int');
        $results = $this->ffi->pdf_ocr_detect_page($engine, $handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_detect_page');
        return $results;
    }

    /**
     * Recognize page with OCR (full OCR process).
     *
     * @param CData $engine OCR engine handle
     * @param CData $handle Document handle
     * @param int $pageIndex Page index
     * @return CData OCR results handle
     */
    public function pdfOcrRecognizePage(CData $engine, CData $handle, int $pageIndex): CData
    {
        $errorCode = FFI::new('int');
        $results = $this->ffi->pdf_ocr_recognize_page($engine, $handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_recognize_page');
        return $results;
    }

    /**
     * Extract OCR text from results.
     *
     * @param CData $results OCR results handle
     * @return string Extracted text
     */
    public function pdfOcrExtractText(CData $results): string
    {
        $text = $this->ffi->pdf_ocr_extract_text($results);
        return StringMarshaller::fromCString($text, false);
    }

    /**
     * Get OCR results count.
     *
     * @param CData $results OCR results handle
     * @return int Number of results
     */
    public function pdfOcrResultsCount(CData $results): int
    {
        return (int)$this->ffi->pdf_ocr_results_count($results);
    }

    /**
     * Get OCR span by index.
     *
     * @param CData $results OCR results handle
     * @param int $index Span index
     * @return CData OCR span handle
     */
    public function pdfOcrResultsGetSpan(CData $results, int $index): CData
    {
        return $this->ffi->pdf_ocr_results_get_span($results, $index);
    }

    /**
     * Get average confidence from OCR results.
     *
     * @param CData $results OCR results handle
     * @return float Average confidence (0.0-1.0)
     */
    public function pdfOcrResultsAverageConfidence(CData $results): float
    {
        return (float)$this->ffi->pdf_ocr_results_average_confidence($results);
    }

    /**
     * Get character confidence from OCR span.
     *
     * @param CData $span OCR span handle
     * @param int $charIndex Character index
     * @return float Character confidence (0.0-1.0)
     */
    public function pdfOcrSpanGetCharConfidence(CData $span, int $charIndex): float
    {
        return (float)$this->ffi->pdf_ocr_span_get_char_confidence($span, $charIndex);
    }

    /**
     * Get bounding box from OCR span.
     *
     * @param CData $span OCR span handle
     * @return array Bounding box [x, y, width, height]
     */
    public function pdfOcrSpanGetBbox(CData $span): array
    {
        $x = FFI::new('float');
        $y = FFI::new('float');
        $width = FFI::new('float');
        $height = FFI::new('float');

        $this->ffi->pdf_ocr_span_get_bbox($span, FFI::addr($x), FFI::addr($y), FFI::addr($width), FFI::addr($height));

        return [
            'x' => (float)$x->cdata,
            'y' => (float)$y->cdata,
            'width' => (float)$width->cdata,
            'height' => (float)$height->cdata,
        ];
    }

    /**
     * Free OCR results.
     *
     * @param CData $results OCR results handle
     */
    public function pdfOcrResultsFree(CData $results): void
    {
        $this->ffi->pdf_ocr_results_free($results);
    }

    /**
     * Free OCR span.
     *
     * @param CData $span OCR span handle
     */
    public function pdfOcrSpanFree(CData $span): void
    {
        $this->ffi->pdf_ocr_span_free($span);
    }

    /**
     * Get text from OCR span.
     *
     * @param CData $span OCR span handle
     * @return string Text content of the span
     */
    public function pdfOcrSpanGetText(CData $span): string
    {
        $text = $this->ffi->pdf_ocr_span_get_text($span);
        return StringMarshaller::fromCString($text, false);
    }

    /**
     * Get confidence from OCR span.
     *
     * @param CData $span OCR span handle
     * @return float Confidence score (0.0-1.0)
     */
    public function pdfOcrSpanGetConfidence(CData $span): float
    {
        return (float)$this->ffi->pdf_ocr_span_get_confidence($span);
    }

    /**
     * Validate PDF/A compliance.
     *
     * @param CData $handle Document handle
     * @param string $level PDF/A level (A1A, A1B, A2A, A2B, A3A, A3B, A4, A4e)
     * @return CData Validation result handle
     */
    public function pdfValidatePdfA(CData $handle, string $level): CData
    {
        $cLevel = StringMarshaller::toCString($level);
        $errorCode = FFI::new('int');

        try {
            $resultHandle = $this->ffi->pdf_validate_pdf_a($handle, $cLevel, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_validate_pdf_a');
            return $resultHandle;
        } finally {
            unset($cLevel);
        }
    }

    /**
     * Check if PDF/A compliant.
     *
     * @param CData $resultHandle Validation result handle
     * @return bool True if document is PDF/A compliant
     */
    public function pdfPdfAIsCompliant(CData $resultHandle): bool
    {
        return (bool)$this->ffi->pdf_pdf_a_is_compliant($resultHandle);
    }

    /**
     * Get PDF/A error count.
     *
     * @param CData $resultHandle Validation result handle
     * @return int Number of errors
     */
    public function pdfPdfAErrorCount(CData $resultHandle): int
    {
        return (int)$this->ffi->pdf_pdf_a_error_count($resultHandle);
    }

    /**
     * Get PDF/A warning count.
     *
     * @param CData $resultHandle Validation result handle
     * @return int Number of warnings
     */
    public function pdfPdfAWarningCount(CData $resultHandle): int
    {
        return (int)$this->ffi->pdf_pdf_a_warning_count($resultHandle);
    }

    /**
     * Get PDF/A error by index.
     *
     * @param CData $resultHandle Validation result handle
     * @param int $index Error index
     * @return string Error message
     */
    public function pdfPdfAGetError(CData $resultHandle, int $index): string
    {
        $error = $this->ffi->pdf_pdf_a_get_error($resultHandle, $index);
        return StringMarshaller::fromCString($error, false);
    }

    /**
     * Get PDF/A warning by index.
     *
     * @param CData $resultHandle Validation result handle
     * @param int $index Warning index
     * @return string Warning message
     */
    public function pdfPdfAGetWarning(CData $resultHandle, int $index): string
    {
        $warning = $this->ffi->pdf_pdf_a_get_warning($resultHandle, $index);
        return StringMarshaller::fromCString($warning, false);
    }

    /**
     * Get PDF/A validation report.
     *
     * @param CData $resultHandle Validation result handle
     * @return string Full report
     */
    public function pdfPdfAGetReport(CData $resultHandle): string
    {
        $report = $this->ffi->pdf_pdf_a_get_report($resultHandle);
        return StringMarshaller::fromCString($report, false);
    }

    /**
     * Free PDF/A validation result.
     *
     * @param CData $resultHandle Validation result handle
     */
    public function pdfPdfAResultFree(CData $resultHandle): void
    {
        $this->ffi->pdf_pdf_a_result_free($resultHandle);
    }

    /**
     * Validate PDF/X compliance.
     *
     * @param CData $handle Document handle
     * @param string $standard PDF/X standard (1a, 1b, 2, 3, 4, 4p)
     * @return CData Validation result handle
     */
    public function pdfValidatePdfX(CData $handle, string $standard): CData
    {
        $cStandard = StringMarshaller::toCString($standard);
        $errorCode = FFI::new('int');

        try {
            $resultHandle = $this->ffi->pdf_validate_pdf_x($handle, $cStandard, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_validate_pdf_x');
            return $resultHandle;
        } finally {
            unset($cStandard);
        }
    }

    /**
     * Check if PDF/X compliant.
     *
     * @param CData $resultHandle Validation result handle
     * @return bool True if document is PDF/X compliant
     */
    public function pdfPdfXIsCompliant(CData $resultHandle): bool
    {
        return (bool)$this->ffi->pdf_pdf_x_is_compliant($resultHandle);
    }

    /**
     * Get PDF/X error count.
     *
     * @param CData $resultHandle Validation result handle
     * @return int Number of errors
     */
    public function pdfPdfXErrorCount(CData $resultHandle): int
    {
        return (int)$this->ffi->pdf_pdf_x_error_count($resultHandle);
    }

    /**
     * Free PDF/X validation result.
     *
     * @param CData $resultHandle Validation result handle
     */
    public function pdfPdfXResultFree(CData $resultHandle): void
    {
        $this->ffi->pdf_pdf_x_result_free($resultHandle);
    }

    /**
     * Validate PDF/UA accessibility.
     *
     * @param CData $handle Document handle
     * @return CData Validation result handle
     */
    public function pdfValidatePdfUa(CData $handle): CData
    {
        $errorCode = FFI::new('int');
        $resultHandle = $this->ffi->pdf_validate_pdf_ua($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_validate_pdf_ua');
        return $resultHandle;
    }

    /**
     * Check if PDF/UA accessible.
     *
     * @param CData $resultHandle Validation result handle
     * @return bool True if document is PDF/UA accessible
     */
    public function pdfPdfUaIsAccessible(CData $resultHandle): bool
    {
        return (bool)$this->ffi->pdf_pdf_ua_is_accessible($resultHandle);
    }

    /**
     * Get PDF/UA error count.
     *
     * @param CData $resultHandle Validation result handle
     * @return int Number of accessibility issues
     */
    public function pdfPdfUaErrorCount(CData $resultHandle): int
    {
        return (int)$this->ffi->pdf_pdf_ua_error_count($resultHandle);
    }

    /**
     * Free PDF/UA validation result.
     *
     * @param CData $resultHandle Validation result handle
     */
    public function pdfPdfUaResultFree(CData $resultHandle): void
    {
        $this->ffi->pdf_pdf_ua_result_free($resultHandle);
    }

    /**
     * Convert document to PDF/A.
     *
     * @param CData $handle Document handle
     * @param string $level PDF/A level
     */
    public function pdfConvertToPdfA(CData $handle, string $level): void
    {
        $cLevel = StringMarshaller::toCString($level);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_convert_to_pdf_a($handle, $cLevel, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_convert_to_pdf_a');
        } finally {
            unset($cLevel);
        }
    }

    /**
     * Convert document to PDF/X.
     *
     * @param CData $handle Document handle
     * @param string $standard PDF/X standard
     */
    public function pdfConvertToPdfX(CData $handle, string $standard): void
    {
        $cStandard = StringMarshaller::toCString($standard);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_convert_to_pdf_x($handle, $cStandard, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_convert_to_pdf_x');
        } finally {
            unset($cStandard);
        }
    }

    /**
     * Convert document to PDF/UA.
     *
     * @param CData $handle Document handle
     */
    public function pdfConvertToPdfUa(CData $handle): void
    {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_convert_to_pdf_ua($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_convert_to_pdf_ua');
    }

    /**
     * Check if document has signatures.
     *
     * @param CData $handle Document handle
     * @return bool True if document has signatures
     */
    public function pdfDocumentHasSignatures(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_has_signatures($handle);
    }

    /**
     * Get signature count.
     *
     * @param CData $handle Document handle
     * @return int Number of signatures
     */
    public function pdfDocumentGetSignatureCount(CData $handle): int
    {
        return (int)$this->ffi->pdf_document_get_signature_count($handle);
    }

    /**
     * Get signature by index.
     *
     * @param CData $handle Document handle
     * @param int $index Signature index
     * @return CData Signature handle
     */
    public function pdfDocumentGetSignature(CData $handle, int $index): CData
    {
        $errorCode = FFI::new('int');
        $signatureHandle = $this->ffi->pdf_document_get_signature($handle, $index, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_signature');
        return $signatureHandle;
    }

    /**
     * Get signature reason.
     *
     * @param CData $signatureHandle Signature handle
     * @return string Signature reason
     */
    public function pdfSignatureGetReason(CData $signatureHandle): string
    {
        $reason = $this->ffi->pdf_signature_get_reason($signatureHandle);
        return StringMarshaller::fromCString($reason, false);
    }

    /**
     * Get signer name.
     *
     * @param CData $signatureHandle Signature handle
     * @return string Signer name
     */
    public function pdfSignatureGetSigner(CData $signatureHandle): string
    {
        $signer = $this->ffi->pdf_signature_get_signer($signatureHandle);
        return StringMarshaller::fromCString($signer, false);
    }

    /**
     * Get signature date.
     *
     * @param CData $signatureHandle Signature handle
     * @return string ISO 8601 date string
     */
    public function pdfSignatureGetDate(CData $signatureHandle): string
    {
        $date = $this->ffi->pdf_signature_get_date($signatureHandle);
        return StringMarshaller::fromCString($date, false);
    }

    /**
     * Verify signature.
     *
     * @param CData $signatureHandle Signature handle
     * @param CData $certificateHandle Certificate handle (optional)
     * @return bool True if signature is valid
     */
    public function pdfSignatureVerify(CData $signatureHandle, ?CData $certificateHandle = null): bool
    {
        $errorCode = FFI::new('int');
        $valid = $this->ffi->pdf_signature_verify($signatureHandle, $certificateHandle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_signature_verify');
        return (bool)$valid;
    }

    /**
     * Free signature handle.
     *
     * @param CData $signatureHandle Signature handle
     */
    public function pdfSignatureFree(CData $signatureHandle): void
    {
        $this->ffi->pdf_signature_free($signatureHandle);
    }

    /**
     * Load certificate from bytes.
     *
     * @param string $certData Certificate data (PEM or DER format)
     * @param string $password Certificate password (if encrypted)
     * @return CData Certificate handle
     */
    public function pdfCertificateLoadFromBytes(string $certData, string $password = ''): CData
    {
        $cData = StringMarshaller::toCString($certData);
        $cPassword = StringMarshaller::toCString($password);
        $errorCode = FFI::new('int');

        try {
            $certHandle = $this->ffi->pdf_certificate_load_from_bytes($cData, strlen($certData), $cPassword, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_certificate_load_from_bytes');
            return $certHandle;
        } finally {
            unset($cData, $cPassword);
        }
    }

    /**
     * Load certificate from file.
     *
     * @param string $filePath Path to certificate file
     * @param string $password Certificate password (if encrypted)
     * @return CData Certificate handle
     */
    public function pdfCertificateLoadFromFile(string $filePath, string $password = ''): CData
    {
        $cPath = StringMarshaller::toCString($filePath);
        $cPassword = StringMarshaller::toCString($password);
        $errorCode = FFI::new('int');

        try {
            $certHandle = $this->ffi->pdf_certificate_load_from_file($cPath, $cPassword, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_certificate_load_from_file');
            return $certHandle;
        } finally {
            unset($cPath, $cPassword);
        }
    }

    /**
     * Get certificate subject.
     *
     * @param CData $certificateHandle Certificate handle
     * @return string Certificate subject DN
     */
    public function pdfCertificateGetSubject(CData $certificateHandle): string
    {
        $subject = $this->ffi->pdf_certificate_get_subject($certificateHandle);
        return StringMarshaller::fromCString($subject, false);
    }

    /**
     * Get certificate issuer.
     *
     * @param CData $certificateHandle Certificate handle
     * @return string Certificate issuer DN
     */
    public function pdfCertificateGetIssuer(CData $certificateHandle): string
    {
        $issuer = $this->ffi->pdf_certificate_get_issuer($certificateHandle);
        return StringMarshaller::fromCString($issuer, false);
    }

    /**
     * Free certificate handle.
     *
     * @param CData $certificateHandle Certificate handle
     */
    public function pdfCertificateFree(CData $certificateHandle): void
    {
        $this->ffi->pdf_certificate_free($certificateHandle);
    }

    /**
     * Sign document.
     *
     * @param CData $handle Document handle
     * @param CData $certificateHandle Certificate handle
     * @param string $reason Signature reason
     * @param string $location Signature location
     * @param int $pageIndex Page to sign on
     * @param float $x X coordinate
     * @param float $y Y coordinate
     * @param float $width Signature box width
     * @param float $height Signature box height
     */
    public function pdfSignDocument(
        CData $handle,
        CData $certificateHandle,
        string $reason,
        string $location,
        int $pageIndex,
        float $x,
        float $y,
        float $width,
        float $height
    ): void {
        $cReason = StringMarshaller::toCString($reason);
        $cLocation = StringMarshaller::toCString($location);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_sign_document(
                $handle,
                $certificateHandle,
                $cReason,
                $cLocation,
                $pageIndex,
                $x,
                $y,
                $width,
                $height,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_sign_document');
        } finally {
            unset($cReason, $cLocation);
        }
    }

    /**
     * Analyze page.
     *
     * @param CData $handle Document handle
     * @param int $pageIndex Page index
     * @return CData Analysis result handle
     */
    public function pdfAnalyzePage(CData $handle, int $pageIndex): CData
    {
        $errorCode = FFI::new('int');
        $resultHandle = $this->ffi->pdf_analyze_page($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_analyze_page');
        return $resultHandle;
    }

    /**
     * Get page complexity score.
     *
     * @param CData $analysisHandle Analysis result handle
     * @return float Complexity score (0.0-1.0)
     */
    public function pdfAnalysisGetComplexityScore(CData $analysisHandle): float
    {
        return (float)$this->ffi->pdf_analysis_get_complexity_score($analysisHandle);
    }

    /**
     * Get page content type.
     *
     * @param CData $analysisHandle Analysis result handle
     * @return string Content type (text, image, mixed, etc.)
     */
    public function pdfAnalysisGetContentType(CData $analysisHandle): string
    {
        $type = $this->ffi->pdf_analysis_get_content_type($analysisHandle);
        return StringMarshaller::fromCString($type, false);
    }

    /**
     * Get text density.
     *
     * @param CData $analysisHandle Analysis result handle
     * @return float Text density (0.0-1.0)
     */
    public function pdfAnalysisGetTextDensity(CData $analysisHandle): float
    {
        return (float)$this->ffi->pdf_analysis_get_text_density($analysisHandle);
    }

    /**
     * Get image density.
     *
     * @param CData $analysisHandle Analysis result handle
     * @return float Image density (0.0-1.0)
     */
    public function pdfAnalysisGetImageDensity(CData $analysisHandle): float
    {
        return (float)$this->ffi->pdf_analysis_get_image_density($analysisHandle);
    }

    /**
     * Free analysis result.
     *
     * @param CData $analysisHandle Analysis result handle
     */
    public function pdfAnalysisResultFree(CData $analysisHandle): void
    {
        $this->ffi->pdf_analysis_result_free($analysisHandle);
    }

    /**
     * Estimate processing time for page.
     *
     * @param CData $handle Document handle
     * @param int $pageIndex Page index
     * @return int Estimated milliseconds
     */
    public function pdfEstimateProcessingTime(CData $handle, int $pageIndex): int
    {
        return (int)$this->ffi->pdf_estimate_processing_time($handle, $pageIndex);
    }

    /**
     * Detect columns in page.
     *
     * @param CData $handle Document handle
     * @param int $pageIndex Page index
     * @return int Number of columns detected
     */
    public function pdfDetectColumns(CData $handle, int $pageIndex): int
    {
        $errorCode = FFI::new('int');
        $count = $this->ffi->pdf_detect_columns($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_detect_columns');
        return (int)$count;
    }

    /**
     * Detect tables in page.
     *
     * @param CData $handle Document handle
     * @param int $pageIndex Page index
     * @return int Number of tables detected
     */
    public function pdfDetectTables(CData $handle, int $pageIndex): int
    {
        $errorCode = FFI::new('int');
        $count = $this->ffi->pdf_detect_tables($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_detect_tables');
        return (int)$count;
    }

    /**
     * Get ML model availability status.
     *
     * @return string Status (available, downloading, unavailable)
     */
    public function pdfMlGetStatus(): string
    {
        $status = $this->ffi->pdf_ml_get_status();
        return StringMarshaller::fromCString($status, false);
    }

    /**
     * Check if specific ML model is available.
     *
     * @param string $modelName Model name (table_detection, column_detection, etc.)
     * @return bool True if model is available
     */
    public function pdfMlModelAvailable(string $modelName): bool
    {
        $cName = StringMarshaller::toCString($modelName);
        $errorCode = FFI::new('int');

        try {
            $available = $this->ffi->pdf_ml_model_available($cName, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_ml_model_available');
            return (bool)$available;
        } finally {
            unset($cName);
        }
    }

    // ==================== METADATA OPERATIONS ====================

    /**
     * Get document keywords.
     */
    public function pdfDocumentGetKeywords(CData $handle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_get_keywords($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_keywords');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get document creator application.
     */
    public function pdfDocumentGetCreator(CData $handle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_get_creator($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_creator');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get document producer application.
     */
    public function pdfDocumentGetProducer(CData $handle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_get_producer($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_producer');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get document creation date.
     */
    public function pdfDocumentGetCreationDate(CData $handle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_get_creation_date($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_creation_date');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get document modification date.
     */
    public function pdfDocumentGetModDate(CData $handle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_get_mod_date($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_mod_date');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    // ==================== SECURITY & PERMISSIONS ====================

    /**
     * Check if document is encrypted.
     */
    public function pdfDocumentIsEncrypted(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_is_encrypted($handle);
    }

    /**
     * Get encryption algorithm.
     */
    public function pdfDocumentGetEncryptionAlgorithm(CData $handle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_get_encryption_algorithm($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_encryption_algorithm');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Check if document requires password.
     */
    public function pdfDocumentRequiresPassword(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_requires_password($handle);
    }

    /**
     * Check if document can be printed.
     */
    public function pdfDocumentCanPrint(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_can_print($handle);
    }

    /**
     * Check if document content can be copied.
     */
    public function pdfDocumentCanCopy(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_can_copy($handle);
    }

    /**
     * Check if document can be modified.
     */
    public function pdfDocumentCanModify(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_can_modify($handle);
    }

    /**
     * Check if forms can be filled.
     */
    public function pdfDocumentCanFillForms(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_can_fill_forms($handle);
    }

    /**
     * Check if document can be annotated.
     */
    public function pdfDocumentCanAnnotate(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_can_annotate($handle);
    }

    // ==================== OUTLINES / BOOKMARKS ====================

    /**
     * Get outline (bookmark) count.
     */
    public function pdfDocumentGetOutlineCount(CData $handle): int
    {
        $errorCode = FFI::new('int');
        try {
            $count = (int)$this->ffi->pdf_document_get_outline_count($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_outline_count');
            return $count;
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Get outline title.
     */
    public function pdfDocumentGetOutlineTitle(CData $handle, int $index): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_get_outline_title($handle, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_outline_title');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get outline target page.
     */
    public function pdfDocumentGetOutlinePage(CData $handle, int $index): int
    {
        $errorCode = FFI::new('int');
        try {
            $page = (int)$this->ffi->pdf_document_get_outline_page($handle, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_outline_page');
            return $page;
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Get outline nesting level.
     */
    public function pdfDocumentGetOutlineLevel(CData $handle, int $index): int
    {
        $errorCode = FFI::new('int');
        try {
            $level = (int)$this->ffi->pdf_document_get_outline_level($handle, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_outline_level');
            return $level;
        } finally {
            unset($errorCode);
        }
    }

    // ==================== LAYERS / OCG ====================

    /**
     * Get layer count.
     */
    public function pdfDocumentGetLayerCount(CData $handle): int
    {
        $errorCode = FFI::new('int');
        try {
            $count = (int)$this->ffi->pdf_document_get_layer_count($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_layer_count');
            return $count;
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Get layer name.
     */
    public function pdfDocumentGetLayerName(CData $handle, int $index): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_get_layer_name($handle, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_layer_name');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Check if layer is visible.
     */
    public function pdfDocumentIsLayerVisible(CData $handle, int $index): bool
    {
        $errorCode = FFI::new('int');
        try {
            $visible = (bool)$this->ffi->pdf_document_is_layer_visible($handle, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_is_layer_visible');
            return $visible;
        } finally {
            unset($errorCode);
        }
    }

    // ==================== CACHE MANAGEMENT ====================

    /**
     * Clear all document cache.
     */
    public function pdfCacheClear(CData $handle): void
    {
        $errorCode = FFI::new('int');
        try {
            $this->ffi->pdf_cache_clear($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_cache_clear');
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Invalidate cache for a specific page.
     */
    public function pdfCacheInvalidatePage(CData $handle, int $pageIndex): void
    {
        $errorCode = FFI::new('int');
        try {
            $this->ffi->pdf_cache_invalidate_page($handle, $pageIndex, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_cache_invalidate_page');
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Set cache maximum size.
     */
    public function pdfCacheSetMaxSize(CData $handle, int $maxSize): void
    {
        $errorCode = FFI::new('int');
        try {
            $this->ffi->pdf_cache_set_max_size($handle, $maxSize, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_cache_set_max_size');
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Get cache statistics.
     */
    public function pdfCacheGetStatistics(CData $handle): array
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_cache_get_statistics($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_cache_get_statistics');
            return \json_decode(StringMarshaller::fromCString($cStr, true), true) ?? [];
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get cache statistics as JSON.
     */
    public function pdfCacheGetStatisticsJson(CData $handle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_cache_get_statistics_json($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_cache_get_statistics_json');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    // ==================== PAGE DOM OPERATIONS ====================

    /**
     * Get page width in user units.
     */
    public function pdfPageGetWidth(CData $pageHandle): float
    {
        return (float)$this->ffi->pdf_page_get_width($pageHandle);
    }

    /**
     * Get page height in user units.
     */
    public function pdfPageGetHeight(CData $pageHandle): float
    {
        return (float)$this->ffi->pdf_page_get_height($pageHandle);
    }

    /**
     * Get page index.
     */
    public function pdfPageGetIndex(CData $pageHandle): int
    {
        return (int)$this->ffi->pdf_page_get_index($pageHandle);
    }

    /**
     * Get both page dimensions.
     */
    public function pdfPageGetDimensions(CData $pageHandle): array
    {
        $width = FFI::new('float');
        $height = FFI::new('float');
        $errorCode = FFI::new('int');

        try {
            $success = (bool)$this->ffi->pdf_page_get_dimensions(
                $pageHandle,
                FFI::addr($width),
                FFI::addr($height),
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_page_get_dimensions');
            return ['width' => $width->cdata, 'height' => $height->cdata];
        } finally {
            unset($width, $height, $errorCode);
        }
    }

    /**
     * Free page handle.
     */
    public function pdfPageFree(CData $pageHandle): void
    {
        $this->ffi->pdf_page_free($pageHandle);
    }

    // ==================== DOCUMENT EDITOR EXTENSIONS ====================

    /**
     * Check if document editor has unsaved changes.
     */
    public function pdfDocumentEditorIsModified(CData $editorHandle): bool
    {
        return (bool)$this->ffi->pdf_document_editor_is_modified($editorHandle);
    }

    /**
     * Get source path from document editor.
     */
    public function pdfDocumentEditorGetSourcePath(CData $editorHandle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_editor_get_source_path($editorHandle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_get_source_path');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get PDF version from editor.
     */
    public function pdfDocumentEditorGetVersion(CData $editorHandle): array
    {
        $major = FFI::new('uint8_t');
        $minor = FFI::new('uint8_t');

        try {
            $this->ffi->pdf_document_editor_get_version(
                $editorHandle,
                FFI::addr($major),
                FFI::addr($minor)
            );
            return ['major' => $major->cdata, 'minor' => $minor->cdata];
        } finally {
            unset($major, $minor);
        }
    }

    /**
     * Get page count from editor.
     */
    public function pdfDocumentEditorGetPageCount(CData $editorHandle): int
    {
        $errorCode = FFI::new('int');
        try {
            $count = (int)$this->ffi->pdf_document_editor_get_page_count($editorHandle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_get_page_count');
            return $count;
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Open document for editing.
     */
    public function pdfDocumentEditorOpen(string $path): ?CData
    {
        $cPath = StringMarshaller::toCString($path);
        $errorCode = FFI::new('int');

        try {
            $handle = $this->ffi->pdf_document_editor_open($cPath, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_open', ['path' => $path]);
            return $handle;
        } finally {
            unset($cPath, $errorCode);
        }
    }

    /**
     * Free document editor handle.
     */
    public function pdfDocumentEditorFree(CData $editorHandle): void
    {
        $this->ffi->pdf_document_editor_free($editorHandle);
    }

    /**
     * Save document editor changes.
     */
    public function pdfDocumentEditorSave(CData $editorHandle, string $path): void
    {
        $cPath = StringMarshaller::toCString($path);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_document_editor_save($editorHandle, $cPath, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_save', ['path' => $path]);
        } finally {
            unset($cPath, $errorCode);
        }
    }

    // ==================== PDF CREATION FUNCTIONS ====================

    /**
     * Create PDF from Markdown content.
     */
    public function pdfFromMarkdown(string $markdown): ?CData
    {
        $cMarkdown = StringMarshaller::toCString($markdown);
        $errorCode = FFI::new('int');

        try {
            $handle = $this->ffi->pdf_from_markdown($cMarkdown, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_from_markdown');
            return $handle;
        } finally {
            unset($cMarkdown, $errorCode);
        }
    }

    /**
     * Create PDF from HTML content.
     */
    public function pdfFromHtml(string $html): ?CData
    {
        $cHtml = StringMarshaller::toCString($html);
        $errorCode = FFI::new('int');

        try {
            $handle = $this->ffi->pdf_from_html($cHtml, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_from_html');
            return $handle;
        } finally {
            unset($cHtml, $errorCode);
        }
    }

    /**
     * Create PDF from plain text.
     */
    public function pdfFromText(string $text): ?CData
    {
        $cText = StringMarshaller::toCString($text);
        $errorCode = FFI::new('int');

        try {
            $handle = $this->ffi->pdf_from_text($cText, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_from_text');
            return $handle;
        } finally {
            unset($cText, $errorCode);
        }
    }

    /**
     * Save PDF to file.
     */
    public function pdfSave(CData $handle, string $path): void
    {
        $cPath = StringMarshaller::toCString($path);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_save($handle, $cPath, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_save', ['path' => $path]);
        } finally {
            unset($cPath, $errorCode);
        }
    }

    /**
     * Save PDF to bytes.
     */
    public function pdfSaveToBytes(CData $handle): string
    {
        $outputPtr = FFI::new('char**');
        $outputLen = FFI::new('size_t');
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_save_to_bytes($handle, FFI::addr($outputPtr), FFI::addr($outputLen), FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_save_to_bytes');

            if ($outputPtr->cdata === null || $outputLen->cdata === 0) {
                return '';
            }

            return \FFI::string($outputPtr->cdata, $outputLen->cdata);
        } finally {
            unset($outputPtr, $outputLen, $errorCode);
        }
    }

    /**
     * Get page count from PDF handle.
     */
    public function pdfGetPageCount(CData $handle): int
    {
        $errorCode = FFI::new('int');
        try {
            $count = (int)$this->ffi->pdf_get_page_count($handle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_get_page_count');
            return $count;
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Free PDF handle.
     */
    public function pdfFree(CData $handle): void
    {
        $this->ffi->pdf_free($handle);
    }

    // ==================== DOCUMENT EDITOR METADATA OPERATIONS ====================

    /**
     * Get title from editor.
     */
    public function pdfDocumentEditorGetTitle(CData $editorHandle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_editor_get_title($editorHandle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_get_title');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Set title in editor.
     */
    public function pdfDocumentEditorSetTitle(CData $editorHandle, string $title): void
    {
        $cTitle = StringMarshaller::toCString($title);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_document_editor_set_title($editorHandle, $cTitle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_set_title');
        } finally {
            unset($cTitle, $errorCode);
        }
    }

    /**
     * Get author from editor.
     */
    public function pdfDocumentEditorGetAuthor(CData $editorHandle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_editor_get_author($editorHandle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_get_author');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Set author in editor.
     */
    public function pdfDocumentEditorSetAuthor(CData $editorHandle, string $author): void
    {
        $cAuthor = StringMarshaller::toCString($author);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_document_editor_set_author($editorHandle, $cAuthor, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_set_author');
        } finally {
            unset($cAuthor, $errorCode);
        }
    }

    /**
     * Get subject from editor.
     */
    public function pdfDocumentEditorGetSubject(CData $editorHandle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_document_editor_get_subject($editorHandle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_get_subject');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Set subject in editor.
     */
    public function pdfDocumentEditorSetSubject(CData $editorHandle, string $subject): void
    {
        $cSubject = StringMarshaller::toCString($subject);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_document_editor_set_subject($editorHandle, $cSubject, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_set_subject');
        } finally {
            unset($cSubject, $errorCode);
        }
    }

    // ==================== FONT ACCESSORS ====================

    /**
     * Get font list count.
     */
    public function pdfOxideFontCount(CData $fontList): int
    {
        return (int)$this->ffi->pdf_oxide_font_count($fontList);
    }

    /**
     * Get font name by index.
     */
    public function pdfOxideFontGetName(CData $fontList, int $index): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_oxide_font_get_name($fontList, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_oxide_font_get_name');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get font type by index.
     */
    public function pdfOxideFontGetType(CData $fontList, int $index): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_oxide_font_get_type($fontList, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_oxide_font_get_type');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get font encoding by index.
     */
    public function pdfOxideFontGetEncoding(CData $fontList, int $index): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_oxide_font_get_encoding($fontList, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_oxide_font_get_encoding');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Check if font is embedded.
     */
    public function pdfOxideFontIsEmbedded(CData $fontList, int $index): bool
    {
        return (bool)$this->ffi->pdf_oxide_font_is_embedded($fontList, $index);
    }

    /**
     * Check if font is subset.
     */
    public function pdfOxideFontIsSubset(CData $fontList, int $index): bool
    {
        return (bool)$this->ffi->pdf_oxide_font_is_subset($fontList, $index);
    }

    /**
     * Get font size by index.
     */
    public function pdfOxideFontGetSize(CData $fontList, int $index): float
    {
        return (float)$this->ffi->pdf_oxide_font_get_size($fontList, $index);
    }

    /**
     * Free font list.
     */
    public function pdfOxideFontListFree(CData $fontList): void
    {
        $this->ffi->pdf_oxide_font_list_free($fontList);
    }

    // ==================== IMAGE ACCESSORS ====================

    /**
     * Get image list count.
     */
    public function pdfOxideImageCount(CData $imageList): int
    {
        return (int)$this->ffi->pdf_oxide_image_count($imageList);
    }

    /**
     * Get image width by index.
     */
    public function pdfOxideImageGetWidth(CData $imageList, int $index): int
    {
        return (int)$this->ffi->pdf_oxide_image_get_width($imageList, $index);
    }

    /**
     * Get image height by index.
     */
    public function pdfOxideImageGetHeight(CData $imageList, int $index): int
    {
        return (int)$this->ffi->pdf_oxide_image_get_height($imageList, $index);
    }

    /**
     * Get image format by index.
     */
    public function pdfOxideImageGetFormat(CData $imageList, int $index): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_oxide_image_get_format($imageList, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_oxide_image_get_format');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get image colorspace by index.
     */
    public function pdfOxideImageGetColorspace(CData $imageList, int $index): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_oxide_image_get_colorspace($imageList, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_oxide_image_get_colorspace');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get image bits per component by index.
     */
    public function pdfOxideImageGetBitsPerComponent(CData $imageList, int $index): int
    {
        return (int)$this->ffi->pdf_oxide_image_get_bits_per_component($imageList, $index);
    }

    /**
     * Get image data by index.
     */
    public function pdfOxideImageGetData(CData $imageList, int $index): string
    {
        $outSize = FFI::new('size_t');
        $errorCode = FFI::new('int');

        try {
            $dataPtr = $this->ffi->pdf_oxide_image_get_data($imageList, $index, FFI::addr($outSize), FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_oxide_image_get_data');

            if ($dataPtr === null || $outSize->cdata === 0) {
                return '';
            }

            return \FFI::string($dataPtr, $outSize->cdata);
        } finally {
            unset($outSize, $errorCode);
        }
    }

    /**
     * Free image list.
     */
    public function pdfOxideImageListFree(CData $imageList): void
    {
        $this->ffi->pdf_oxide_image_list_free($imageList);
    }

    // ==================== SEARCH RESULT ACCESSORS ====================

    /**
     * Get search result count.
     */
    public function pdfOxideSearchResultCount(CData $results): int
    {
        return (int)$this->ffi->pdf_oxide_search_result_count($results);
    }

    /**
     * Get search result text by index.
     */
    public function pdfOxideSearchResultGetText(CData $results, int $index): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_oxide_search_result_get_text($results, $index, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_oxide_search_result_get_text');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode, $cStr);
        }
    }

    /**
     * Get search result page number by index.
     */
    public function pdfOxideSearchResultGetPage(CData $results, int $index): int
    {
        return (int)$this->ffi->pdf_oxide_search_result_get_page($results, $index);
    }

    /**
     * Get search result position by index.
     */
    public function pdfOxideSearchResultGetPosition(CData $results, int $index): int
    {
        return (int)$this->ffi->pdf_oxide_search_result_get_position($results, $index);
    }

    /**
     * Get search result bounding box by index.
     */
    public function pdfOxideSearchResultGetBbox(CData $results, int $index): array
    {
        $x = FFI::new('float');
        $y = FFI::new('float');
        $width = FFI::new('float');
        $height = FFI::new('float');
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_oxide_search_result_get_bbox(
                $results,
                $index,
                FFI::addr($x),
                FFI::addr($y),
                FFI::addr($width),
                FFI::addr($height),
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_oxide_search_result_get_bbox');
            return [
                'x' => $x->cdata,
                'y' => $y->cdata,
                'width' => $width->cdata,
                'height' => $height->cdata,
            ];
        } finally {
            unset($x, $y, $width, $height, $errorCode);
        }
    }

    /**
     * Free search results.
     */
    public function pdfOxideSearchResultFree(CData $results): void
    {
        $this->ffi->pdf_oxide_search_result_free($results);
    }

    // ==================== ANNOTATION LIST OPERATIONS ====================

    /**
     * Get annotation count on a page.
     */
    public function pdfPageGetAnnotationsCount(CData $pageHandle): int
    {
        $errorCode = FFI::new('int');
        try {
            $count = (int)$this->ffi->pdf_page_get_annotations_count($pageHandle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_page_get_annotations_count');
            return $count;
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Get annotation count by type on a page.
     */
    public function pdfPageGetAnnotationsByTypeCount(CData $pageHandle, int $annotationType): int
    {
        $errorCode = FFI::new('int');
        try {
            $count = (int)$this->ffi->pdf_page_get_annotations_by_type_count($pageHandle, $annotationType, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_page_get_annotations_by_type_count');
            return $count;
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Get annotation type.
     */
    public function pdfAnnotationGetType(CData $annotationHandle): int
    {
        return (int)$this->ffi->pdf_annotation_get_type($annotationHandle);
    }

    /**
     * Get annotation contents.
     */
    public function pdfAnnotationGetContents(CData $annotationHandle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_annotation_get_contents($annotationHandle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_annotation_get_contents');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Get annotation subject.
     */
    public function pdfAnnotationGetSubject(CData $annotationHandle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_annotation_get_subject($annotationHandle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_annotation_get_subject');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Get annotation author.
     */
    public function pdfAnnotationGetAuthor(CData $annotationHandle): string
    {
        $errorCode = FFI::new('int');
        try {
            $cStr = $this->ffi->pdf_annotation_get_author($annotationHandle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_annotation_get_author');
            return StringMarshaller::fromCString($cStr, true);
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Get annotation bounding box.
     */
    public function pdfAnnotationGetBbox(CData $annotationHandle): array
    {
        $x = FFI::new('float');
        $y = FFI::new('float');
        $width = FFI::new('float');
        $height = FFI::new('float');

        try {
            $this->ffi->pdf_annotation_get_bbox(
                $annotationHandle,
                FFI::addr($x),
                FFI::addr($y),
                FFI::addr($width),
                FFI::addr($height)
            );
            return [
                'x' => $x->cdata,
                'y' => $y->cdata,
                'width' => $width->cdata,
                'height' => $height->cdata,
            ];
        } finally {
            unset($x, $y, $width, $height);
        }
    }

    /**
     * Get annotation color (RGB).
     */
    public function pdfAnnotationGetColor(CData $annotationHandle): array
    {
        $r = FFI::new('float');
        $g = FFI::new('float');
        $b = FFI::new('float');
        $a = FFI::new('float');

        try {
            $this->ffi->pdf_annotation_get_color(
                $annotationHandle,
                FFI::addr($r),
                FFI::addr($g),
                FFI::addr($b),
                FFI::addr($a)
            );
            return [
                'r' => $r->cdata,
                'g' => $g->cdata,
                'b' => $b->cdata,
                'a' => $a->cdata,
            ];
        } finally {
            unset($r, $g, $b, $a);
        }
    }

    /**
     * Get annotation opacity.
     */
    public function pdfAnnotationGetOpacity(CData $annotationHandle): float
    {
        return (float)$this->ffi->pdf_annotation_get_opacity($annotationHandle);
    }

    /**
     * Get annotation flags.
     */
    public function pdfAnnotationGetFlags(CData $annotationHandle): int
    {
        return (int)$this->ffi->pdf_annotation_get_flags($annotationHandle);
    }

    /**
     * Free annotation handle.
     */
    public function pdfAnnotationFree(CData $annotationHandle): void
    {
        $this->ffi->pdf_annotation_free($annotationHandle);
    }

    // ==================== RENDERING FUNCTIONS (EXTENDED) ====================

    /**
     * Get default render options.
     */
    public function pdfRenderOptionsDefault(): ?CData
    {
        return $this->ffi->pdf_render_options_default();
    }

    /**
     * Create page renderer.
     */
    public function pdfPageRendererCreate(?CData $options = null): ?CData
    {
        $errorCode = FFI::new('int');
        try {
            $renderer = $this->ffi->pdf_page_renderer_create($options, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_page_renderer_create');
            return $renderer;
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Set page renderer options.
     */
    public function pdfPageRendererSetOptions(CData $renderer, CData $options): void
    {
        $errorCode = FFI::new('int');
        try {
            $this->ffi->pdf_page_renderer_set_options($renderer, $options, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_page_renderer_set_options');
        } finally {
            unset($errorCode);
        }
    }

    /**
     * Free page renderer.
     */
    public function pdfPageRendererFree(CData $renderer): void
    {
        $this->ffi->pdf_page_renderer_free($renderer);
    }

    /**
     * Get image format MIME type.
     */
    public function pdfImageFormatMimeType(string $format): string
    {
        $cFormat = StringMarshaller::toCString($format);
        try {
            $cMime = $this->ffi->pdf_image_format_mime_type($cFormat);
            return StringMarshaller::fromCString($cMime, false);
        } finally {
            unset($cFormat);
        }
    }

    /**
     * Get image format file extension.
     */
    public function pdfImageFormatExtension(string $format): string
    {
        $cFormat = StringMarshaller::toCString($format);
        try {
            $cExt = $this->ffi->pdf_image_format_extension($cFormat);
            return StringMarshaller::fromCString($cExt, false);
        } finally {
            unset($cFormat);
        }
    }

    // ==================== PHASE 4: Document Page Operations (15 functions) ====================

    /**
     * Insert a blank page at specified index.
     */
    public function pdfDocumentInsertPage(\FFI\CData $handle, int $index, float $width, float $height): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_insert_page($handle, $index, $width, $height, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Delete a page at specified index.
     */
    public function pdfDocumentDeletePage(\FFI\CData $handle, int $index): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_delete_page($handle, $index, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Rotate page by specified degrees (90, 180, 270).
     */
    public function pdfDocumentRotatePage(\FFI\CData $handle, int $pageIndex, int $degrees): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_rotate_page($handle, $pageIndex, $degrees, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Get page media box (width and height).
     */
    public function pdfDocumentGetPageMediaBox(\FFI\CData $handle, int $pageIndex): array
    {
        $error = \FFI::new('int');
        $x = \FFI::new('double');
        $y = \FFI::new('double');
        $w = \FFI::new('double');
        $h = \FFI::new('double');

        $this->ffi->pdf_document_get_page_media_box($handle, $pageIndex, $x, $y, $w, $h, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);

        return [
            'x' => $x->cdata,
            'y' => $y->cdata,
            'width' => $w->cdata,
            'height' => $h->cdata
        ];
    }

    /**
     * Set page media box dimensions.
     */
    public function pdfDocumentSetPageMediaBox(\FFI\CData $handle, int $pageIndex, float $x, float $y, float $width, float $height): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_page_media_box($handle, $pageIndex, $x, $y, $width, $height, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Get page crop box dimensions.
     */
    public function pdfDocumentGetPageCropBox(\FFI\CData $handle, int $pageIndex): array
    {
        $error = \FFI::new('int');
        $x = \FFI::new('double');
        $y = \FFI::new('double');
        $w = \FFI::new('double');
        $h = \FFI::new('double');

        $this->ffi->pdf_document_get_page_crop_box($handle, $pageIndex, $x, $y, $w, $h, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);

        return [
            'x' => $x->cdata,
            'y' => $y->cdata,
            'width' => $w->cdata,
            'height' => $h->cdata
        ];
    }

    /**
     * Set page crop box dimensions.
     */
    public function pdfDocumentSetPageCropBox(\FFI\CData $handle, int $pageIndex, float $x, float $y, float $width, float $height): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_page_crop_box($handle, $pageIndex, $x, $y, $width, $height, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Get page ArtBox dimensions.
     */
    public function pdfDocumentGetPageArtBox(\FFI\CData $handle, int $pageIndex): array
    {
        $error = \FFI::new('int');
        $x = \FFI::new('double');
        $y = \FFI::new('double');
        $w = \FFI::new('double');
        $h = \FFI::new('double');

        $this->ffi->pdf_document_get_page_art_box($handle, $pageIndex, $x, $y, $w, $h, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);

        return [
            'x' => $x->cdata,
            'y' => $y->cdata,
            'width' => $w->cdata,
            'height' => $h->cdata
        ];
    }

    /**
     * Set page ArtBox dimensions.
     */
    public function pdfDocumentSetPageArtBox(\FFI\CData $handle, int $pageIndex, float $x, float $y, float $width, float $height): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_page_art_box($handle, $pageIndex, $x, $y, $width, $height, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Get page BleedBox dimensions.
     */
    public function pdfDocumentGetPageBleedBox(\FFI\CData $handle, int $pageIndex): array
    {
        $error = \FFI::new('int');
        $x = \FFI::new('double');
        $y = \FFI::new('double');
        $w = \FFI::new('double');
        $h = \FFI::new('double');

        $this->ffi->pdf_document_get_page_bleed_box($handle, $pageIndex, $x, $y, $w, $h, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);

        return [
            'x' => $x->cdata,
            'y' => $y->cdata,
            'width' => $w->cdata,
            'height' => $h->cdata
        ];
    }

    /**
     * Set page BleedBox dimensions.
     */
    public function pdfDocumentSetPageBleedBox(\FFI\CData $handle, int $pageIndex, float $x, float $y, float $width, float $height): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_page_bleed_box($handle, $pageIndex, $x, $y, $width, $height, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Get page TrimBox dimensions.
     */
    public function pdfDocumentGetPageTrimBox(\FFI\CData $handle, int $pageIndex): array
    {
        $error = \FFI::new('int');
        $x = \FFI::new('double');
        $y = \FFI::new('double');
        $w = \FFI::new('double');
        $h = \FFI::new('double');

        $this->ffi->pdf_document_get_page_trim_box($handle, $pageIndex, $x, $y, $w, $h, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);

        return [
            'x' => $x->cdata,
            'y' => $y->cdata,
            'width' => $w->cdata,
            'height' => $h->cdata
        ];
    }

    /**
     * Set page TrimBox dimensions.
     */
    public function pdfDocumentSetPageTrimBox(\FFI\CData $handle, int $pageIndex, float $x, float $y, float $width, float $height): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_page_trim_box($handle, $pageIndex, $x, $y, $width, $height, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Duplicate/clone a page.
     */
    public function pdfDocumentDuplicatePage(\FFI\CData $handle, int $pageIndex): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_duplicate_page($handle, $pageIndex, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Move page to new position.
     */
    public function pdfDocumentMovePage(\FFI\CData $handle, int $fromIndex, int $toIndex): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_move_page($handle, $fromIndex, $toIndex, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    // ==================== PHASE 5: Content Drawing (20 functions) ====================

    /**
     * Add text at position on page.
     */
    public function pdfDocumentAddText(\FFI\CData $handle, int $pageIndex, string $text, float $x, float $y, float $size): void
    {
        $error = \FFI::new('int');
        $cText = StringMarshaller::toCString($text);
        try {
            $this->ffi->pdf_document_add_text($handle, $pageIndex, $cText, $x, $y, $size, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
        } finally {
            unset($cText);
        }
    }

    /**
     * Add styled text at position.
     */
    public function pdfDocumentAddStyledText(\FFI\CData $handle, int $pageIndex, string $text, float $x, float $y, float $size, int $fontSize, bool $bold, bool $italic): void
    {
        $error = \FFI::new('int');
        $cText = StringMarshaller::toCString($text);
        try {
            $this->ffi->pdf_document_add_styled_text($handle, $pageIndex, $cText, $x, $y, $size, $fontSize, $bold ? 1 : 0, $italic ? 1 : 0, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
        } finally {
            unset($cText);
        }
    }

    /**
     * Add rectangle shape on page.
     */
    public function pdfDocumentAddRectangle(\FFI\CData $handle, int $pageIndex, float $x, float $y, float $width, float $height, bool $fill): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_add_rectangle($handle, $pageIndex, $x, $y, $width, $height, $fill ? 1 : 0, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Add circle shape on page.
     */
    public function pdfDocumentAddCircle(\FFI\CData $handle, int $pageIndex, float $x, float $y, float $radius, bool $fill): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_add_circle($handle, $pageIndex, $x, $y, $radius, $fill ? 1 : 0, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Add ellipse shape on page.
     */
    public function pdfDocumentAddEllipse(\FFI\CData $handle, int $pageIndex, float $x, float $y, float $radiusX, float $radiusY, bool $fill): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_add_ellipse($handle, $pageIndex, $x, $y, $radiusX, $radiusY, $fill ? 1 : 0, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Add line on page.
     */
    public function pdfDocumentAddLine(\FFI\CData $handle, int $pageIndex, float $x1, float $y1, float $x2, float $y2, float $width): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_add_line($handle, $pageIndex, $x1, $y1, $x2, $y2, $width, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Add path on page (polyline).
     */
    public function pdfDocumentAddPath(\FFI\CData $handle, int $pageIndex, array $points, float $width, bool $close): void
    {
        $error = \FFI::new('int');
        $count = count($points);
        $cPoints = \FFI::new("double[$count]");
        for ($i = 0; $i < $count; $i++) {
            $cPoints[$i] = (float)$points[$i];
        }
        try {
            $this->ffi->pdf_document_add_path($handle, $pageIndex, $cPoints, $count, $width, $close ? 1 : 0, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
        } finally {
            unset($cPoints);
        }
    }

    /**
     * Set drawing color (RGB).
     */
    public function pdfDocumentSetDrawColor(\FFI\CData $handle, int $red, int $green, int $blue): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_draw_color($handle, $red, $green, $blue, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Set fill color (RGB).
     */
    public function pdfDocumentSetFillColor(\FFI\CData $handle, int $red, int $green, int $blue): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_fill_color($handle, $red, $green, $blue, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Set text color (RGB).
     */
    public function pdfDocumentSetTextColor(\FFI\CData $handle, int $red, int $green, int $blue): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_text_color($handle, $red, $green, $blue, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Set line width.
     */
    public function pdfDocumentSetLineWidth(\FFI\CData $handle, float $width): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_line_width($handle, $width, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Set line cap style (0=butt, 1=round, 2=square).
     */
    public function pdfDocumentSetLineCapStyle(\FFI\CData $handle, int $style): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_line_cap_style($handle, $style, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Set line join style (0=miter, 1=round, 2=bevel).
     */
    public function pdfDocumentSetLineJoinStyle(\FFI\CData $handle, int $style): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_line_join_style($handle, $style, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Add watermark text to page.
     */
    public function pdfDocumentAddWatermark(\FFI\CData $handle, int $pageIndex, string $text, float $opacity): void
    {
        $error = \FFI::new('int');
        $cText = StringMarshaller::toCString($text);
        try {
            $this->ffi->pdf_document_add_watermark($handle, $pageIndex, $cText, $opacity, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
        } finally {
            unset($cText);
        }
    }

    /**
     * Add image at position on page.
     */
    public function pdfDocumentAddImage(\FFI\CData $handle, int $pageIndex, string $imagePath, float $x, float $y, float $width, float $height): void
    {
        $error = \FFI::new('int');
        $cPath = StringMarshaller::toCString($imagePath);
        try {
            $this->ffi->pdf_document_add_image($handle, $pageIndex, $cPath, $x, $y, $width, $height, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
        } finally {
            unset($cPath);
        }
    }

    /**
     * Add image from memory.
     */
    public function pdfDocumentAddImageFromBytes(\FFI\CData $handle, int $pageIndex, string $imageData, float $x, float $y, float $width, float $height): void
    {
        $error = \FFI::new('int');
        $len = strlen($imageData);
        $cData = \FFI::new("uint8_t[$len]");
        for ($i = 0; $i < $len; $i++) {
            $cData[$i] = ord($imageData[$i]);
        }
        try {
            $this->ffi->pdf_document_add_image_from_bytes($handle, $pageIndex, $cData, $len, $x, $y, $width, $height, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
        } finally {
            unset($cData);
        }
    }

    /**
     * Set blend mode for subsequent drawing operations.
     */
    public function pdfDocumentSetBlendMode(\FFI\CData $handle, int $blendMode): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_blend_mode($handle, $blendMode, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Set opacity for subsequent drawing operations.
     */
    public function pdfDocumentSetOpacity(\FFI\CData $handle, float $opacity): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_set_opacity($handle, $opacity, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    // ==================== PHASE 6: Search Advanced (10 functions) ====================

    /**
     * Search with regex pattern.
     */
    public function pdfDocumentSearchRegex(\FFI\CData $handle, string $pattern): ?\FFI\CData
    {
        $error = \FFI::new('int');
        $cPattern = StringMarshaller::toCString($pattern);
        try {
            $result = $this->ffi->pdf_document_search_regex($handle, $cPattern, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
            return $result;
        } finally {
            unset($cPattern);
        }
    }

    /**
     * Search for text on specific page.
     */
    public function pdfDocumentSearchOnPage(\FFI\CData $handle, int $pageIndex, string $query): ?\FFI\CData
    {
        $error = \FFI::new('int');
        $cQuery = StringMarshaller::toCString($query);
        try {
            $result = $this->ffi->pdf_document_search_on_page($handle, $pageIndex, $cQuery, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
            return $result;
        } finally {
            unset($cQuery);
        }
    }

    /**
     * Search case-insensitive.
     */
    public function pdfDocumentSearchCaseInsensitive(\FFI\CData $handle, string $query): ?\FFI\CData
    {
        $error = \FFI::new('int');
        $cQuery = StringMarshaller::toCString($query);
        try {
            $result = $this->ffi->pdf_document_search_case_insensitive($handle, $cQuery, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
            return $result;
        } finally {
            unset($cQuery);
        }
    }

    /**
     * Search in annotations text.
     */
    public function pdfDocumentSearchAnnotations(\FFI\CData $handle, string $query): ?\FFI\CData
    {
        $error = \FFI::new('int');
        $cQuery = StringMarshaller::toCString($query);
        try {
            $result = $this->ffi->pdf_document_search_annotations($handle, $cQuery, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
            return $result;
        } finally {
            unset($cQuery);
        }
    }

    /**
     * Search by position/area on page.
     */
    public function pdfDocumentSearchInArea(\FFI\CData $handle, int $pageIndex, string $query, float $x, float $y, float $width, float $height): ?\FFI\CData
    {
        $error = \FFI::new('int');
        $cQuery = StringMarshaller::toCString($query);
        try {
            $result = $this->ffi->pdf_document_search_in_area($handle, $pageIndex, $cQuery, $x, $y, $width, $height, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
            return $result;
        } finally {
            unset($cQuery);
        }
    }

    /**
     * Get next search result.
     */
    public function pdfSearchResultNext(?\FFI\CData $handle): ?\FFI\CData
    {
        $error = \FFI::new('int');
        if ($handle === null) {
            return null;
        }
        $result = $this->ffi->pdf_search_result_next($handle, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return $result;
    }

    /**
     * Replace text in document.
     */
    public function pdfDocumentReplaceText(\FFI\CData $handle, string $find, string $replace): int
    {
        $error = \FFI::new('int');
        $cFind = StringMarshaller::toCString($find);
        $cReplace = StringMarshaller::toCString($replace);
        try {
            $count = $this->ffi->pdf_document_replace_text($handle, $cFind, $cReplace, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
            return $count;
        } finally {
            unset($cFind, $cReplace);
        }
    }

    /**
     * Replace text on page.
     */
    public function pdfDocumentReplaceTextOnPage(\FFI\CData $handle, int $pageIndex, string $find, string $replace): int
    {
        $error = \FFI::new('int');
        $cFind = StringMarshaller::toCString($find);
        $cReplace = StringMarshaller::toCString($replace);
        try {
            $count = $this->ffi->pdf_document_replace_text_on_page($handle, $pageIndex, $cFind, $cReplace, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
            return $count;
        } finally {
            unset($cFind, $cReplace);
        }
    }

    /**
     * Get text from specific area.
     */
    public function pdfDocumentGetTextFromArea(\FFI\CData $handle, int $pageIndex, float $x, float $y, float $width, float $height): string
    {
        $error = \FFI::new('int');
        $cText = $this->ffi->pdf_document_get_text_from_area($handle, $pageIndex, $x, $y, $width, $height, \FFI::addr($error));
        try {
            ErrorHandler::checkError($error->cdata);
            return StringMarshaller::fromCString($cText, true);
        } finally {
            // StringMarshaller::fromCString frees the C string
        }
    }

    // ==================== PHASE 7: Format Utilities (10 functions) ====================

    /**
     * Detect PDF file format version from bytes.
     */
    public function pdfDetectFormatVersion(string $data): string
    {
        $len = strlen($data);
        $cData = \FFI::new("uint8_t[$len]");
        for ($i = 0; $i < $len; $i++) {
            $cData[$i] = ord($data[$i]);
        }
        try {
            $cVersion = $this->ffi->pdf_detect_format_version($cData, $len);
            return StringMarshaller::fromCString($cVersion, false);
        } finally {
            unset($cData);
        }
    }

    /**
     * Check if data appears to be PDF.
     */
    public function pdfIsPdfData(string $data): bool
    {
        $len = strlen($data);
        $cData = \FFI::new("uint8_t[$len]");
        for ($i = 0; $i < $len; $i++) {
            $cData[$i] = ord($data[$i]);
        }
        try {
            $result = $this->ffi->pdf_is_pdf_data($cData, $len);
            return (bool)$result;
        } finally {
            unset($cData);
        }
    }

    /**
     * Convert page to specific image format.
     */
    public function pdfDocumentRenderPageAsFormat(\FFI\CData $handle, int $pageIndex, string $format, int $quality): string
    {
        $error = \FFI::new('int');
        $cFormat = StringMarshaller::toCString($format);
        try {
            $cImage = $this->ffi->pdf_document_render_page_as_format($handle, $pageIndex, $cFormat, $quality, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
            // This returns a pointer to a byte buffer that we need to convert to string
            $len = $this->ffi->pdf_get_last_buffer_size();
            $imageData = '';
            $pData = $cImage;
            for ($i = 0; $i < $len; $i++) {
                $imageData .= chr($pData[$i]);
            }
            $this->ffi->pdf_free_buffer($cImage);
            return $imageData;
        } finally {
            unset($cFormat);
        }
    }

    /**
     * Optimize PDF for web (linearization).
     */
    public function pdfDocumentOptimizeForWeb(\FFI\CData $handle): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_optimize_for_web($handle, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Compress PDF content streams.
     */
    public function pdfDocumentCompress(\FFI\CData $handle): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_compress($handle, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Get estimated uncompressed size.
     */
    public function pdfDocumentGetUncompressedSize(\FFI\CData $handle): int
    {
        $error = \FFI::new('int');
        $size = $this->ffi->pdf_document_get_uncompressed_size($handle, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return $size;
    }

    /**
     * Merge multiple PDF documents.
     */
    public function pdfDocumentMerge(\FFI\CData $handle, \FFI\CData $otherHandle): void
    {
        $error = \FFI::new('int');
        $this->ffi->pdf_document_merge($handle, $otherHandle, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
    }

    /**
     * Split PDF into separate documents.
     */
    public function pdfDocumentSplit(\FFI\CData $handle, int $pageIndex): ?\FFI\CData
    {
        $error = \FFI::new('int');
        $newHandle = $this->ffi->pdf_document_split($handle, $pageIndex, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return $newHandle;
    }

    /**
     * Extract range of pages as new PDF.
     */
    public function pdfDocumentExtractPages(\FFI\CData $handle, int $startPage, int $endPage): ?\FFI\CData
    {
        $error = \FFI::new('int');
        $newHandle = $this->ffi->pdf_document_extract_pages($handle, $startPage, $endPage, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return $newHandle;
    }

    /**
     * Get file size in bytes.
     */
    public function pdfDocumentGetFileSize(\FFI\CData $handle): int
    {
        $error = \FFI::new('int');
        $size = $this->ffi->pdf_document_get_file_size($handle, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return $size;
    }

    // ==================== PHASE 8: DOM Elements (10 functions) ====================

    /**
     * Get text elements from page.
     */
    public function pdfDocumentGetPageTextElements(\FFI\CData $handle, int $pageIndex): ?\FFI\CData
    {
        $error = \FFI::new('int');
        $elements = $this->ffi->pdf_document_get_page_text_elements($handle, $pageIndex, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return $elements;
    }

    /**
     * Get image elements from page.
     */
    public function pdfDocumentGetPageImageElements(\FFI\CData $handle, int $pageIndex): ?\FFI\CData
    {
        $error = \FFI::new('int');
        $elements = $this->ffi->pdf_document_get_page_image_elements($handle, $pageIndex, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return $elements;
    }

    /**
     * Get count of text elements.
     */
    public function pdfTextElementCount(\FFI\CData $handle): int
    {
        $error = \FFI::new('int');
        $count = $this->ffi->pdf_text_element_count($handle, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return $count;
    }

    /**
     * Get text element content.
     */
    public function pdfTextElementGetContent(\FFI\CData $handle, int $index): string
    {
        $error = \FFI::new('int');
        $cContent = $this->ffi->pdf_text_element_get_content($handle, $index, \FFI::addr($error));
        try {
            ErrorHandler::checkError($error->cdata);
            return StringMarshaller::fromCString($cContent, true);
        } finally {
            // StringMarshaller::fromCString frees the C string
        }
    }

    /**
     * Get text element font name.
     */
    public function pdfTextElementGetFont(\FFI\CData $handle, int $index): string
    {
        $error = \FFI::new('int');
        $cFont = $this->ffi->pdf_text_element_get_font($handle, $index, \FFI::addr($error));
        try {
            ErrorHandler::checkError($error->cdata);
            return StringMarshaller::fromCString($cFont, true);
        } finally {
            // StringMarshaller::fromCString frees the C string
        }
    }

    /**
     * Get text element position.
     */
    public function pdfTextElementGetPosition(\FFI\CData $handle, int $index): array
    {
        $error = \FFI::new('int');
        $x = \FFI::new('double');
        $y = \FFI::new('double');
        $this->ffi->pdf_text_element_get_position($handle, $index, $x, $y, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return [
            'x' => $x->cdata,
            'y' => $y->cdata
        ];
    }

    /**
     * Get image element dimensions.
     */
    public function pdfImageElementGetDimensions(\FFI\CData $handle, int $index): array
    {
        $error = \FFI::new('int');
        $width = \FFI::new('double');
        $height = \FFI::new('double');
        $this->ffi->pdf_image_element_get_dimensions($handle, $index, $width, $height, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return [
            'width' => $width->cdata,
            'height' => $height->cdata
        ];
    }

    /**
     * Get image element position.
     */
    public function pdfImageElementGetPosition(\FFI\CData $handle, int $index): array
    {
        $error = \FFI::new('int');
        $x = \FFI::new('double');
        $y = \FFI::new('double');
        $this->ffi->pdf_image_element_get_position($handle, $index, $x, $y, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return [
            'x' => $x->cdata,
            'y' => $y->cdata
        ];
    }

    /**
     * Free text/image elements list.
     */
    public function pdfElementListFree(?\FFI\CData $handle): void
    {
        if ($handle !== null) {
            $this->ffi->pdf_element_list_free($handle);
        }
    }

    // ==================== PHASE 9: Final Edge Cases & Platform Utilities (15+ functions) ====================

    // ========== Advanced Rendering Options ==========

    /**
     * Create advanced rendering options with all parameters.
     */
    public function pdfCreateRenderingOptions(float $dpi, int $format, int $quality): ?\FFI\CData
    {
        $error = \FFI::new('int');
        $opts = $this->ffi->pdf_create_rendering_options($dpi, $format, $quality, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return $opts;
    }

    /**
     * Set rendering option by key-value.
     */
    public function pdfRenderingOptionsSet(?\FFI\CData $opts, string $key, string $value): void
    {
        if ($opts === null) {
            return;
        }
        $error = \FFI::new('int');
        $cKey = StringMarshaller::toCString($key);
        $cValue = StringMarshaller::toCString($value);
        try {
            $this->ffi->pdf_rendering_options_set($opts, $cKey, $cValue, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
        } finally {
            unset($cKey, $cValue);
        }
    }

    /**
     * Render page with custom interpolation method.
     */
    public function pdfDocumentRenderPageWithInterpolation(\FFI\CData $handle, int $pageIndex, int $interpolationMethod): string
    {
        $error = \FFI::new('int');
        $cImage = $this->ffi->pdf_document_render_page_with_interpolation($handle, $pageIndex, $interpolationMethod, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        $len = $this->ffi->pdf_get_last_buffer_size();
        $imageData = '';
        for ($i = 0; $i < $len; $i++) {
            $imageData .= chr($cImage[$i]);
        }
        $this->ffi->pdf_free_buffer($cImage);
        return $imageData;
    }

    /**
     * Get supported image formats.
     */
    public function pdfGetSupportedImageFormats(): array
    {
        $error = \FFI::new('int');
        $cFormats = $this->ffi->pdf_get_supported_image_formats(\FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return explode(',', StringMarshaller::fromCString($cFormats, true));
    }

    // ========== Platform & System Utilities ==========

    /**
     * Get platform/OS information.
     */
    public function pdfGetPlatformInfo(): string
    {
        $cInfo = $this->ffi->pdf_get_platform_info();
        return StringMarshaller::fromCString($cInfo, false);
    }

    /**
     * Get library version string.
     */
    public function pdfGetLibraryVersion(): string
    {
        $cVersion = $this->ffi->pdf_get_library_version();
        return StringMarshaller::fromCString($cVersion, false);
    }

    /**
     * Check if feature is supported on this platform.
     */
    public function pdfIsFeatureSupported(string $feature): bool
    {
        $error = \FFI::new('int');
        $cFeature = StringMarshaller::toCString($feature);
        try {
            $result = $this->ffi->pdf_is_feature_supported($cFeature, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
            return (bool)$result;
        } finally {
            unset($cFeature);
        }
    }

    /**
     * Get available CPU count for parallel processing.
     */
    public function pdfGetCpuCount(): int
    {
        return (int)$this->ffi->pdf_get_cpu_count();
    }

    /**
     * Get available memory in bytes.
     */
    public function pdfGetAvailableMemory(): int
    {
        return (int)$this->ffi->pdf_get_available_memory();
    }

    // ========== Encoding & Binary Utilities ==========

    /**
     * Encode string as hex.
     */
    public function pdfEncodeHex(string $data): string
    {
        $len = strlen($data);
        $cData = \FFI::new("uint8_t[$len]");
        for ($i = 0; $i < $len; $i++) {
            $cData[$i] = ord($data[$i]);
        }
        try {
            $cHex = $this->ffi->pdf_encode_hex($cData, $len);
            return StringMarshaller::fromCString($cHex, true);
        } finally {
            unset($cData);
        }
    }

    /**
     * Decode hex string.
     */
    public function pdfDecodeHex(string $hex): string
    {
        $cHex = StringMarshaller::toCString($hex);
        try {
            $cData = $this->ffi->pdf_decode_hex($cHex);
            $len = $this->ffi->pdf_get_last_buffer_size();
            $data = '';
            for ($i = 0; $i < $len; $i++) {
                $data .= chr($cData[$i]);
            }
            $this->ffi->pdf_free_buffer($cData);
            return $data;
        } finally {
            unset($cHex);
        }
    }

    /**
     * Encode data as Base64.
     */
    public function pdfEncodeBase64(string $data): string
    {
        $len = strlen($data);
        $cData = \FFI::new("uint8_t[$len]");
        for ($i = 0; $i < $len; $i++) {
            $cData[$i] = ord($data[$i]);
        }
        try {
            $cB64 = $this->ffi->pdf_encode_base64($cData, $len);
            return StringMarshaller::fromCString($cB64, true);
        } finally {
            unset($cData);
        }
    }

    /**
     * Decode Base64 data.
     */
    public function pdfDecodeBase64(string $base64): string
    {
        $cB64 = StringMarshaller::toCString($base64);
        try {
            $cData = $this->ffi->pdf_decode_base64($cB64);
            $len = $this->ffi->pdf_get_last_buffer_size();
            $data = '';
            for ($i = 0; $i < $len; $i++) {
                $data .= chr($cData[$i]);
            }
            $this->ffi->pdf_free_buffer($cData);
            return $data;
        } finally {
            unset($cB64);
        }
    }

    // ========== Advanced Format Conversion ==========

    /**
     * Convert PDF to alternative format with options.
     */
    public function pdfDocumentConvertToFormat(\FFI\CData $handle, string $targetFormat, array $options): string
    {
        $error = \FFI::new('int');
        $cFormat = StringMarshaller::toCString($targetFormat);

        // Serialize options as JSON or key-value pairs
        $optionsStr = json_encode($options);
        $cOptions = StringMarshaller::toCString($optionsStr);

        try {
            $cResult = $this->ffi->pdf_document_convert_to_format($handle, $cFormat, $cOptions, \FFI::addr($error));
            ErrorHandler::checkError($error->cdata);
            $len = $this->ffi->pdf_get_last_buffer_size();
            $result = '';
            for ($i = 0; $i < $len; $i++) {
                $result .= chr($cResult[$i]);
            }
            $this->ffi->pdf_free_buffer($cResult);
            return $result;
        } finally {
            unset($cFormat, $cOptions);
        }
    }

    /**
     * Get document statistics and metrics.
     */
    public function pdfDocumentGetStatistics(\FFI\CData $handle): array
    {
        $error = \FFI::new('int');
        $cStats = $this->ffi->pdf_document_get_statistics($handle, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        $statsJson = StringMarshaller::fromCString($cStats, true);
        return json_decode($statsJson, true) ?? [];
    }

    /**
     * Validate PDF structure integrity.
     */
    public function pdfDocumentValidateStructure(\FFI\CData $handle): bool
    {
        $error = \FFI::new('int');
        $isValid = $this->ffi->pdf_document_validate_structure($handle, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        return (bool)$isValid;
    }

    /**
     * Get detailed validation errors.
     */
    public function pdfDocumentGetValidationErrors(\FFI\CData $handle): array
    {
        $error = \FFI::new('int');
        $cErrors = $this->ffi->pdf_document_get_validation_errors($handle, \FFI::addr($error));
        ErrorHandler::checkError($error->cdata);
        $errorsJson = StringMarshaller::fromCString($cErrors, true);
        return json_decode($errorsJson, true) ?? [];
    }

    // ========== Performance & Optimization ==========

    /**
     * Enable/disable caching for performance.
     */
    public function pdfSetCachingEnabled(bool $enabled): void
    {
        $this->ffi->pdf_set_caching_enabled($enabled ? 1 : 0);
    }

    /**
     * Set thread pool size for parallel processing.
     */
    public function pdfSetThreadPoolSize(int $size): void
    {
        if ($size > 0) {
            $this->ffi->pdf_set_thread_pool_size($size);
        }
    }

    /**
     * Get performance metrics.
     */
    public function pdfGetPerformanceMetrics(): array
    {
        $cMetrics = $this->ffi->pdf_get_performance_metrics();
        $metricsJson = StringMarshaller::fromCString($cMetrics, false);
        return json_decode($metricsJson, true) ?? [];
    }

    /**
     * Reset performance counters.
     */
    public function pdfResetPerformanceMetrics(): void
    {
        $this->ffi->pdf_reset_performance_metrics();
    }

    // ========== Barcode Detection Functions ==========

    /**
     * Create barcode detector configuration.
     */
    public function pdfBarcodeDetectorConfigCreate(): CData
    {
        return $this->ffi->pdf_barcode_detector_config_create();
    }

    /**
     * Set confidence threshold for detector.
     */
    public function pdfBarcodeDetectorConfigSetConfidenceThreshold(CData $config, float $threshold): void
    {
        $this->ffi->pdf_barcode_detector_config_set_confidence_threshold($config, $threshold);
    }

    /**
     * Set try harder flag for detector.
     */
    public function pdfBarcodeDetectorConfigSetTryHarder(CData $config): void
    {
        $this->ffi->pdf_barcode_detector_config_set_try_harder($config);
    }

    /**
     * Detect barcodes on a page.
     */
    public function pdfDetectBarcodes(CData $handle, int $pageIndex, CData $config): CData
    {
        return $this->ffi->pdf_detect_barcodes($handle, $pageIndex, $config);
    }

    /**
     * Detect barcodes in a specific region.
     */
    public function pdfDetectBarcodesInRegion(CData $handle, int $pageIndex, float $x, float $y, float $width, float $height, CData $config): CData
    {
        return $this->ffi->pdf_detect_barcodes_in_region($handle, $pageIndex, $x, $y, $width, $height, $config);
    }

    /**
     * Get number of detected barcodes.
     */
    public function pdfDetectionResultsCount(CData $results): int
    {
        return (int)$this->ffi->pdf_detection_results_count($results);
    }

    /**
     * Get a specific detected barcode.
     */
    public function pdfDetectionResultsGetBarcode(CData $results, int $index): CData
    {
        return $this->ffi->pdf_detection_results_get_barcode($results, $index);
    }

    /**
     * Get format of detected barcode.
     */
    public function pdfDetectedBarcodeGetFormat(CData $barcode): string
    {
        $cFormat = $this->ffi->pdf_detected_barcode_get_format($barcode);
        return StringMarshaller::fromCString($cFormat, false);
    }

    /**
     * Get decoded data from detected barcode.
     */
    public function pdfDetectedBarcodeGetData(CData $barcode): string
    {
        $cData = $this->ffi->pdf_detected_barcode_get_data($barcode);
        return StringMarshaller::fromCString($cData, false);
    }

    /**
     * Get bounding box of detected barcode.
     */
    public function pdfDetectedBarcodeGetBbox(CData $barcode): array
    {
        $x = \FFI::new('float');
        $y = \FFI::new('float');
        $width = \FFI::new('float');
        $height = \FFI::new('float');

        $this->ffi->pdf_detected_barcode_get_bbox($barcode, \FFI::addr($x), \FFI::addr($y), \FFI::addr($width), \FFI::addr($height));

        return [
            'x' => (float)$x->cdata,
            'y' => (float)$y->cdata,
            'width' => (float)$width->cdata,
            'height' => (float)$height->cdata,
        ];
    }

    /**
     * Get confidence score of detected barcode.
     */
    public function pdfDetectedBarcodeGetConfidence(CData $barcode): float
    {
        return (float)$this->ffi->pdf_detected_barcode_get_confidence($barcode);
    }

    /**
     * Free detected barcode.
     */
    public function pdfDetectedBarcodeFree(CData $barcode): void
    {
        $this->ffi->pdf_detected_barcode_free($barcode);
    }

    /**
     * Free detection results.
     */
    public function pdfDetectionResultsFree(CData $results): void
    {
        $this->ffi->pdf_detection_results_free($results);
    }

    // ========== XFA Form Functions ==========

    /**
     * Check if document has XFA form.
     */
    public function pdfDocumentHasXfa(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_has_xfa($handle);
    }

    /**
     * Parse XFA form from document.
     */
    public function pdfParseXfaForm(CData $handle): CData
    {
        return $this->ffi->pdf_parse_xfa_form($handle);
    }

    /**
     * Get number of fields in XFA form.
     */
    public function pdfXfaFormFieldCount(CData $form): int
    {
        return (int)$this->ffi->pdf_xfa_form_field_count($form);
    }

    /**
     * Get a specific field from XFA form.
     */
    public function pdfXfaFormGetField(CData $form, int $index): CData
    {
        return $this->ffi->pdf_xfa_form_get_field($form, $index);
    }

    /**
     * Get XFA field name.
     */
    public function pdfXfaFieldGetName(CData $field): string
    {
        $cName = $this->ffi->pdf_xfa_field_get_name($field);
        return StringMarshaller::fromCString($cName, false);
    }

    /**
     * Get XFA field type.
     */
    public function pdfXfaFieldGetType(CData $field): string
    {
        $cType = $this->ffi->pdf_xfa_field_get_type($field);
        return StringMarshaller::fromCString($cType, false);
    }

    /**
     * Get XFA field value.
     */
    public function pdfXfaFieldGetValue(CData $field): string
    {
        $cValue = $this->ffi->pdf_xfa_field_get_value($field);
        return StringMarshaller::fromCString($cValue, false);
    }

    /**
     * Set XFA field value.
     */
    public function pdfXfaFieldSetValue(CData $field, string $value): void
    {
        $cValue = StringMarshaller::toCString($value);
        try {
            $this->ffi->pdf_xfa_field_set_value($field, $cValue);
        } finally {
            unset($cValue);
        }
    }

    /**
     * Free XFA field.
     */
    public function pdfXfaFieldFree(CData $field): void
    {
        $this->ffi->pdf_xfa_field_free($field);
    }

    /**
     * Free XFA form.
     */
    public function pdfXfaFormFree(CData $form): void
    {
        $this->ffi->pdf_xfa_form_free($form);
    }

    // ========== Advanced OCR Functions (unique additions) ==========

    /**
     * Detect text in a specific region.
     */
    public function pdfOcrDetectRegion(CData $handle, int $pageIndex, float $x, float $y, float $width, float $height, CData $engine): CData
    {
        $errorCode = FFI::new('int');
        $results = $this->ffi->pdf_ocr_detect_region($handle, $pageIndex, $x, $y, $width, $height, $engine, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_detect_region');
        return $results;
    }

    /**
     * Detect language on a page.
     */
    public function pdfOcrDetectLanguage(CData $handle, int $pageIndex, CData $engine): CData
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_ocr_detect_language($handle, $pageIndex, $engine, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_detect_language');
        return $result;
    }

    /**
     * Set OCR language.
     */
    public function pdfOcrSetLanguage(CData $engine, string $lang): void
    {
        $cLang = StringMarshaller::toCString($lang);
        $errorCode = FFI::new('int');
        try {
            $this->ffi->pdf_ocr_set_language($engine, $cLang, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_ocr_set_language');
        } finally {
            unset($cLang);
        }
    }

    /**
     * Get supported OCR languages.
     */
    public function pdfOcrGetSupportedLanguages(CData $engine): array
    {
        $outCount = FFI::new('int');
        $errorCode = FFI::new('int');
        $langsPtr = $this->ffi->pdf_ocr_get_supported_languages($engine, FFI::addr($outCount), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_ocr_get_supported_languages');

        $languages = [];
        $count = (int)$outCount->cdata;
        for ($i = 0; $i < $count; $i++) {
            $languages[] = StringMarshaller::fromCString($langsPtr[$i], false);
        }
        return $languages;
    }

    // ========== Advanced Signature Functions (unique additions) ==========

    /**
     * Sign document with visual signature.
     */
    public function pdfDocumentSignVisual(CData $handle, CData $cert, int $pageIndex, float $x, float $y, float $width, float $height, string $reason, string $location): bool
    {
        $cReason = StringMarshaller::toCString($reason);
        $cLocation = StringMarshaller::toCString($location);
        $errorCode = FFI::new('int');
        try {
            $result = $this->ffi->pdf_document_sign_visual($handle, $cert, $pageIndex, $x, $y, $width, $height, $cReason, $cLocation, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_sign_visual');
            return (bool)$result;
        } finally {
            unset($cReason, $cLocation);
        }
    }

    /**
     * Add timestamp to document.
     */
    public function pdfDocumentAddTimestamp(CData $handle, string $tsaUrl): bool
    {
        $cUrl = StringMarshaller::toCString($tsaUrl);
        $errorCode = FFI::new('int');
        try {
            $result = $this->ffi->pdf_document_add_timestamp($handle, $cUrl, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_add_timestamp');
            return (bool)$result;
        } finally {
            unset($cUrl);
        }
    }

    /**
     * Check if document was modified since signing.
     */
    public function pdfDocumentWasModifiedSinceSigning(CData $handle): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_was_modified_since_signing($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_was_modified_since_signing');
        return (bool)$result;
    }

    /**
     * Load PKCS12 certificate.
     */
    public function pdfCertificateLoadPkcs12(string $path, string $password): CData
    {
        $cPath = StringMarshaller::toCString($path);
        $cPassword = StringMarshaller::toCString($password);
        $errorCode = FFI::new('int');
        try {
            $cert = $this->ffi->pdf_certificate_load_pkcs12($cPath, $cPassword, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_certificate_load_pkcs12');
            return $cert;
        } finally {
            unset($cPath, $cPassword);
        }
    }

    /**
     * Get certificate serial number.
     */
    public function pdfCertificateGetSerial(CData $cert): string
    {
        $serial = $this->ffi->pdf_certificate_get_serial($cert);
        return StringMarshaller::fromCString($serial, false);
    }

    /**
     * Get certificate not-before date.
     */
    public function pdfCertificateGetNotBefore(CData $cert): string
    {
        $date = $this->ffi->pdf_certificate_get_not_before($cert);
        return StringMarshaller::fromCString($date, false);
    }

    /**
     * Get certificate not-after date.
     */
    public function pdfCertificateGetNotAfter(CData $cert): string
    {
        $date = $this->ffi->pdf_certificate_get_not_after($cert);
        return StringMarshaller::fromCString($date, false);
    }

    /**
     * Get signature location.
     */
    public function pdfSignatureGetLocation(CData $sig): string
    {
        $location = $this->ffi->pdf_signature_get_location($sig);
        return StringMarshaller::fromCString($location, false);
    }

    /**
     * Get signature contact info.
     */
    public function pdfSignatureGetContactInfo(CData $sig): string
    {
        $contact = $this->ffi->pdf_signature_get_contact_info($sig);
        return StringMarshaller::fromCString($contact, false);
    }

    /**
     * Get signature signing time.
     */
    public function pdfSignatureGetSigningTime(CData $sig): string
    {
        $time = $this->ffi->pdf_signature_get_signing_time($sig);
        return StringMarshaller::fromCString($time, false);
    }

    // ========== Credential & Signing Functions (Phase 1) ==========

    /**
     * Create signing credentials from a PKCS#12 file.
     *
     * @param string $filePath Path to the P12/PFX file
     * @param string $password Password for the PKCS#12 bundle
     * @return CData Credentials handle
     */
    public function pdfCredentialsFromPkcs12(string $filePath, string $password): CData
    {
        $cPath = StringMarshaller::toCString($filePath);
        $cPassword = StringMarshaller::toCString($password);
        $errorCode = FFI::new('int');

        try {
            $handle = $this->ffi->pdf_credentials_from_pkcs12($cPath, $cPassword, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_credentials_from_pkcs12', ['path' => $filePath]);
            return $handle;
        } finally {
            unset($cPath, $cPassword);
        }
    }

    /**
     * Create signing credentials from PEM certificate and key files.
     *
     * @param string $certFile Path to the PEM certificate file
     * @param string $keyFile Path to the PEM private key file
     * @param string $keyPassword Password for the private key (empty if unencrypted)
     * @return CData Credentials handle
     */
    public function pdfCredentialsFromPem(string $certFile, string $keyFile, string $keyPassword = ''): CData
    {
        $cCertFile = StringMarshaller::toCString($certFile);
        $cKeyFile = StringMarshaller::toCString($keyFile);
        $cKeyPassword = StringMarshaller::toCString($keyPassword);
        $errorCode = FFI::new('int');

        try {
            $handle = $this->ffi->pdf_credentials_from_pem(
                $cCertFile,
                $cKeyFile,
                $cKeyPassword,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_credentials_from_pem', [
                'cert_file' => $certFile,
                'key_file' => $keyFile,
            ]);
            return $handle;
        } finally {
            unset($cCertFile, $cKeyFile, $cKeyPassword);
        }
    }

    /**
     * Create signing credentials from DER-encoded certificate and key data.
     *
     * @param string $certData DER-encoded certificate bytes
     * @param string $keyData DER-encoded private key bytes
     * @return CData Credentials handle
     */
    public function pdfCredentialsFromDer(string $certData, string $keyData): CData
    {
        $errorCode = FFI::new('int');

        $handle = $this->ffi->pdf_credentials_from_der(
            $certData,
            strlen($certData),
            $keyData,
            strlen($keyData),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_credentials_from_der');
        return $handle;
    }

    /**
     * Add an intermediate certificate to the credentials chain.
     *
     * @param CData $credentials Credentials handle
     * @param string $certData DER-encoded certificate bytes
     * @return bool True on success
     */
    public function pdfCredentialsAddChainCert(CData $credentials, string $certData): bool
    {
        $errorCode = FFI::new('int');

        $result = $this->ffi->pdf_credentials_add_chain_cert(
            $credentials,
            $certData,
            strlen($certData),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_credentials_add_chain_cert');
        return (bool)$result;
    }

    /**
     * Get the certificate from a credentials handle.
     *
     * @param CData $credentials Credentials handle
     * @return CData Certificate handle
     */
    public function pdfCredentialsGetCertificate(CData $credentials): CData
    {
        $errorCode = FFI::new('int');

        $certHandle = $this->ffi->pdf_credentials_get_certificate($credentials, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_credentials_get_certificate');
        return $certHandle;
    }

    /**
     * Get the Common Name (CN) from a certificate handle.
     *
     * @param CData $cert Certificate handle
     * @return string Certificate common name
     */
    public function pdfCertificateGetCn(CData $cert): string
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_certificate_get_cn($cert, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_certificate_get_cn');
        return FFI::string($result);
    }

    /**
     * Get the size of a certificate in bytes.
     *
     * @param CData $cert Certificate handle
     * @return int Certificate size in bytes
     */
    public function pdfCertificateGetSize(CData $cert): int
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_certificate_get_size($cert, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_certificate_get_size');
        return (int)$result;
    }

    /**
     * Verify a specific signature using the Rust verification engine.
     *
     * @param CData $document Document handle
     * @param int $signatureIndex Zero-based signature index
     * @return int Verification status (0=Valid, 1=Invalid, 2=Unknown, 3=ValidWithWarnings)
     */
    public function pdfVerifySignature(CData $document, int $signatureIndex): int
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_verify_signature($document, $signatureIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_verify_signature');
        return (int)$result;
    }

    /**
     * Free a credentials handle.
     *
     * @param CData $credentials Credentials handle
     */
    public function pdfCredentialsFree(CData $credentials): void
    {
        $this->ffi->pdf_credentials_free($credentials);
    }

    /**
     * Sign PDF data in memory using credentials.
     *
     * @param string $pdfData Raw PDF bytes
     * @param CData $credentials Credentials handle
     * @param string $reason Signing reason
     * @param string $location Signing location
     * @param string $contact Contact information
     * @param int $algorithm Signature algorithm (0=RSA, 1=ECDSA)
     * @param int $subfilter CMS subfilter type
     * @return string Signed PDF bytes
     */
    public function pdfDocumentSignWithCredentials(
        string $pdfData,
        CData $credentials,
        string $reason = '',
        string $location = '',
        string $contact = '',
        int $algorithm = 0,
        int $subfilter = 0
    ): string {
        $cReason = StringMarshaller::toCString($reason);
        $cLocation = StringMarshaller::toCString($location);
        $cContact = StringMarshaller::toCString($contact);
        $errorCode = FFI::new('int');
        $outData = $this->ffi->new('uint8_t*');
        $outLen = FFI::new('size_t');

        try {
            $result = $this->ffi->pdf_document_sign_with_credentials(
                $pdfData,
                strlen($pdfData),
                $credentials,
                $cReason,
                $cLocation,
                $cContact,
                $algorithm,
                $subfilter,
                FFI::addr($outData),
                FFI::addr($outLen),
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_sign_with_credentials');

            $length = (int)$outLen->cdata;
            $signedPdf = FFI::string($outData, $length);

            // Free the native buffer
            $this->ffi->pdf_signed_bytes_free($outData, $length);

            return $signedPdf;
        } finally {
            unset($cReason, $cLocation, $cContact);
        }
    }

    /**
     * Sign a PDF file on disk using credentials.
     *
     * @param string $inputPath Path to input PDF
     * @param string $outputPath Path to write signed PDF
     * @param CData $credentials Credentials handle
     * @param string $reason Signing reason
     * @param string $location Signing location
     * @param string $contact Contact information
     * @param int $algorithm Signature algorithm (0=RSA, 1=ECDSA)
     * @param int $subfilter CMS subfilter type
     */
    public function pdfDocumentSignFile(
        string $inputPath,
        string $outputPath,
        CData $credentials,
        string $reason = '',
        string $location = '',
        string $contact = '',
        int $algorithm = 0,
        int $subfilter = 0
    ): void {
        $cInputPath = StringMarshaller::toCString($inputPath);
        $cOutputPath = StringMarshaller::toCString($outputPath);
        $cReason = StringMarshaller::toCString($reason);
        $cLocation = StringMarshaller::toCString($location);
        $cContact = StringMarshaller::toCString($contact);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_document_sign_file(
                $cInputPath,
                $cOutputPath,
                $credentials,
                $cReason,
                $cLocation,
                $cContact,
                $algorithm,
                $subfilter,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_sign_file', [
                'input' => $inputPath,
                'output' => $outputPath,
            ]);
        } finally {
            unset($cInputPath, $cOutputPath, $cReason, $cLocation, $cContact);
        }
    }

    /**
     * Sign a PDF document with visual signature appearance.
     *
     * Places a visible signature rectangle at the given coordinates on the
     * specified page and signs the document with the provided credentials.
     *
     * @param string $pdfData Raw PDF bytes
     * @param CData $credentials Credentials handle
     * @param int $pageNum Page number (0-based)
     * @param float $x X coordinate of signature rectangle
     * @param float $y Y coordinate of signature rectangle
     * @param float $width Width of signature rectangle
     * @param float $height Height of signature rectangle
     * @param string $reason Signing reason
     * @param string $location Signing location
     * @param string $contact Contact information
     * @param int $algorithm Signature algorithm
     * @return string Signed PDF bytes
     */
    public function pdfDocumentSignWithAppearance(
        string $pdfData,
        CData $credentials,
        int $pageNum = 0,
        float $x = 50.0,
        float $y = 700.0,
        float $width = 200.0,
        float $height = 50.0,
        string $reason = '',
        string $location = '',
        string $contact = '',
        int $algorithm = 0
    ): string {
        $cReason = StringMarshaller::toCString($reason);
        $cLocation = StringMarshaller::toCString($location);
        $cContact = StringMarshaller::toCString($contact);
        $errorCode = FFI::new('int');
        $outData = $this->ffi->new('uint8_t*');
        $outLen = FFI::new('size_t');

        try {
            $result = $this->ffi->pdf_document_sign_with_appearance(
                $pdfData,
                strlen($pdfData),
                $credentials,
                $pageNum,
                $x,
                $y,
                $width,
                $height,
                $cReason,
                $cLocation,
                $cContact,
                $algorithm,
                FFI::addr($outData),
                FFI::addr($outLen),
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_sign_with_appearance');

            $length = (int)$outLen->cdata;
            $signedPdf = FFI::string($outData, $length);

            // Free the native buffer
            $this->ffi->pdf_signed_bytes_free($outData, $length);

            return $signedPdf;
        } finally {
            unset($cReason, $cLocation, $cContact);
        }
    }

    /**
     * Embed LTV (Long-Term Validation) data into a signed PDF.
     *
     * @param string $pdfData Raw PDF bytes
     * @param string|null $ocspData OCSP response data (or null)
     * @param string|null $crlData CRL data (or null)
     * @return string PDF bytes with embedded LTV data
     */
    public function pdfEmbedLtvData(string $pdfData, ?string $ocspData = null, ?string $crlData = null): string
    {
        $errorCode = FFI::new('int');
        $outData = $this->ffi->new('uint8_t*');
        $outLen = FFI::new('size_t');

        $ocspLen = $ocspData !== null ? strlen($ocspData) : 0;
        $crlLen = $crlData !== null ? strlen($crlData) : 0;

        $result = $this->ffi->pdf_embed_ltv_data(
            $pdfData,
            strlen($pdfData),
            $ocspData,
            $ocspLen,
            $crlData,
            $crlLen,
            FFI::addr($outData),
            FFI::addr($outLen),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_embed_ltv_data');

        $length = (int)$outLen->cdata;
        $resultPdf = FFI::string($outData, $length);

        // Free the native buffer
        $this->ffi->pdf_signed_bytes_free($outData, $length);

        return $resultPdf;
    }

    /**
     * Save signed PDF data to a file.
     *
     * @param string $pdfData Raw signed PDF bytes
     * @param string $outputPath Path to write the file
     */
    public function pdfDocumentSaveSigned(string $pdfData, string $outputPath): void
    {
        $cOutputPath = StringMarshaller::toCString($outputPath);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_document_save_signed(
                $pdfData,
                strlen($pdfData),
                $cOutputPath,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_save_signed', ['output' => $outputPath]);
        } finally {
            unset($cOutputPath);
        }
    }

    /**
     * Free signed bytes allocated by native code.
     *
     * @param CData $data Pointer to signed bytes
     * @param int $len Length of the data
     */
    public function pdfSignedBytesFree(CData $data, int $len): void
    {
        $this->ffi->pdf_signed_bytes_free($data, $len);
    }

    // ========== PAdES Level Enforcement ==========

    /**
     * Validate a signature against a PAdES conformance level.
     *
     * @param CData $handle Document handle
     * @param int $signatureIndex Zero-based signature index
     * @param int $level PAdES level (0=B-B, 1=B-T, 2=B-LT, 3=B-LTA)
     * @return bool True if the signature meets the specified level
     */
    public function pdfPadesValidateLevel(CData $handle, int $signatureIndex, int $level): bool
    {
        $errorCode = FFI::new('int');

        $result = $this->ffi->pdf_pades_validate_level(
            $handle,
            $signatureIndex,
            $level,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_pades_validate_level');

        return ((int)$result) === 1;
    }

    /**
     * Sign PDF data with PAdES conformance at a specified level.
     *
     * @param string $pdfData Raw PDF bytes
     * @param CData $credentials Credentials handle
     * @param int $level PAdES level (0=B-B, 1=B-T, 2=B-LT, 3=B-LTA)
     * @param string|null $tsaUrl TSA URL (required for B-T and above)
     * @param string|null $reason Signing reason
     * @param string|null $location Signing location
     * @param string|null $contact Contact information
     * @return string Signed PDF bytes
     */
    public function pdfPadesSign(
        string $pdfData,
        CData $credentials,
        int $level,
        ?string $tsaUrl = null,
        ?string $reason = null,
        ?string $location = null,
        ?string $contact = null
    ): string {
        $errorCode = FFI::new('int');
        $outData = $this->ffi->new('uint8_t*');
        $outLen = FFI::new('size_t');

        $cTsaUrl = $tsaUrl !== null ? StringMarshaller::toCString($tsaUrl) : null;
        $cReason = $reason !== null ? StringMarshaller::toCString($reason) : null;
        $cLocation = $location !== null ? StringMarshaller::toCString($location) : null;
        $cContact = $contact !== null ? StringMarshaller::toCString($contact) : null;

        try {
            $result = $this->ffi->pdf_pades_sign(
                $pdfData,
                strlen($pdfData),
                $credentials,
                $level,
                $cTsaUrl,
                $cReason,
                $cLocation,
                $cContact,
                FFI::addr($outData),
                FFI::addr($outLen),
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_pades_sign');

            $length = (int)$outLen->cdata;
            $signedPdf = FFI::string($outData, $length);

            // Free the native buffer
            $this->ffi->pdf_signed_bytes_free($outData, $length);

            return $signedPdf;
        } finally {
            unset($cTsaUrl, $cReason, $cLocation, $cContact);
        }
    }

    /**
     * Detect the PAdES conformance level of a signature.
     *
     * @param CData $handle Document handle
     * @param int $signatureIndex Zero-based signature index
     * @return int PAdES level (0=B-B, 1=B-T, 2=B-LT, 3=B-LTA) or -1 on error
     */
    public function pdfPadesGetLevel(CData $handle, int $signatureIndex): int
    {
        $errorCode = FFI::new('int');

        $result = $this->ffi->pdf_pades_get_level(
            $handle,
            $signatureIndex,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_pades_get_level');

        return (int)$result;
    }

    // ========== Accessibility Functions ==========

    /**
     * Check if document is tagged.
     */
    public function pdfAccessibilityIsTagged(CData $handle): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_accessibility_is_tagged($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_is_tagged');
        return (bool)$result;
    }

    /**
     * Get the document structure tree.
     */
    public function pdfAccessibilityGetStructureTree(CData $handle): CData
    {
        $errorCode = FFI::new('int');
        $tree = $this->ffi->pdf_accessibility_get_structure_tree($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_get_structure_tree');
        return $tree;
    }

    /**
     * Automatically tag the document.
     */
    public function pdfAccessibilityAutoTag(CData $handle, ?string $language = null): int
    {
        $cLanguage = $language !== null ? StringMarshaller::toCString($language) : null;
        $errorCode = FFI::new('int');
        try {
            $count = $this->ffi->pdf_accessibility_auto_tag($handle, $cLanguage, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_auto_tag');
            return (int)$count;
        } finally {
            unset($cLanguage);
        }
    }

    /**
     * Set alt text on a structure element.
     */
    public function pdfAccessibilitySetAltText(CData $handle, int $page, int $mcid, string $text): void
    {
        $cText = StringMarshaller::toCString($text);
        $errorCode = FFI::new('int');
        try {
            $this->ffi->pdf_accessibility_set_alt_text($handle, $page, $mcid, $cText, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_set_alt_text');
        } finally {
            unset($cText);
        }
    }

    /**
     * Set the document language.
     */
    public function pdfAccessibilitySetLanguage(CData $handle, string $language): void
    {
        $cLanguage = StringMarshaller::toCString($language);
        $errorCode = FFI::new('int');
        try {
            $this->ffi->pdf_accessibility_set_language($handle, $cLanguage, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_set_language');
        } finally {
            unset($cLanguage);
        }
    }

    /**
     * Set the document title for accessibility.
     */
    public function pdfAccessibilitySetTitle(CData $handle, string $title): void
    {
        $cTitle = StringMarshaller::toCString($title);
        $errorCode = FFI::new('int');
        try {
            $this->ffi->pdf_accessibility_set_title($handle, $cTitle, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_set_title');
        } finally {
            unset($cTitle);
        }
    }

    /**
     * Free a structure tree handle.
     */
    public function pdfStructureTreeFree(CData $handle): void
    {
        $this->ffi->pdf_structure_tree_free($handle);
    }

    /**
     * Free a structure element handle.
     */
    public function pdfStructElemFree(CData $handle): void
    {
        $this->ffi->pdf_struct_elem_free($handle);
    }

    // ========== Optimization Functions ==========

    /**
     * Open a document with mmap for optimized I/O.
     */
    public function pdfDocumentOpenMmap(string $path): CData
    {
        $cPath = StringMarshaller::toCString($path);
        $errorCode = FFI::new('int');
        try {
            $handle = $this->ffi->pdf_document_open_mmap($cPath, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_open_mmap', ['path' => $path]);
            return $handle;
        } finally {
            unset($cPath);
        }
    }

    /**
     * Subset fonts to remove unused glyphs.
     */
    public function pdfOptimizeSubsetFonts(CData $handle): int
    {
        $errorCode = FFI::new('int');
        $saved = $this->ffi->pdf_optimize_subset_fonts($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_optimize_subset_fonts');
        return (int)$saved;
    }

    /**
     * Downsample images.
     */
    public function pdfOptimizeDownsampleImages(CData $handle, int $targetDpi, int $quality): int
    {
        $errorCode = FFI::new('int');
        $saved = $this->ffi->pdf_optimize_downsample_images($handle, $targetDpi, $quality, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_optimize_downsample_images');
        return (int)$saved;
    }

    /**
     * Deduplicate content streams.
     */
    public function pdfOptimizeDeduplicate(CData $handle): int
    {
        $errorCode = FFI::new('int');
        $saved = $this->ffi->pdf_optimize_deduplicate($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_optimize_deduplicate');
        return (int)$saved;
    }

    /**
     * Run full optimization pipeline.
     */
    public function pdfOptimizeFull(CData $handle, int $targetDpi, int $quality): CData
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_optimize_full($handle, $targetDpi, $quality, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_optimize_full');
        return $result;
    }

    /**
     * Get bytes saved from optimization result.
     */
    public function pdfOptimizationResultBytesSaved(CData $resultHandle): int
    {
        return (int)$this->ffi->pdf_optimization_result_bytes_saved($resultHandle);
    }

    /**
     * Free optimization result handle.
     */
    public function pdfOptimizationResultFree(CData $resultHandle): void
    {
        $this->ffi->pdf_optimization_result_free($resultHandle);
    }

    // ========== Enterprise Functions ==========

    /**
     * Apply Bates numbering.
     */
    public function pdfBatesApply(CData $handle, string $prefix, int $startNumber, int $numDigits, int $position): int
    {
        $cPrefix = StringMarshaller::toCString($prefix);
        $errorCode = FFI::new('int');
        try {
            $count = $this->ffi->pdf_bates_apply($handle, $cPrefix, $startNumber, $numDigits, $position, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_bates_apply');
            return (int)$count;
        } finally {
            unset($cPrefix);
        }
    }

    /**
     * Apply advanced Bates numbering.
     */
    public function pdfBatesApplyAdvanced(
        CData $handle,
        string $prefix,
        string $suffix,
        int $startNumber,
        int $numDigits,
        int $position,
        float $fontSize,
        float $margin,
        float $r,
        float $g,
        float $b
    ): int {
        $cPrefix = StringMarshaller::toCString($prefix);
        $cSuffix = StringMarshaller::toCString($suffix);
        $errorCode = FFI::new('int');
        try {
            $count = $this->ffi->pdf_bates_apply_advanced(
                $handle, $cPrefix, $cSuffix,
                $startNumber, $numDigits, $position,
                $fontSize, $margin, $r, $g, $b,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_bates_apply_advanced');
            return (int)$count;
        } finally {
            unset($cPrefix, $cSuffix);
        }
    }

    /**
     * Compare pages from two documents.
     */
    public function pdfComparePages(CData $handleA, CData $handleB, int $pageA, int $pageB): CData
    {
        $errorCode = FFI::new('int');
        $comp = $this->ffi->pdf_compare_pages($handleA, $handleB, $pageA, $pageB, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_compare_pages');
        return $comp;
    }

    /**
     * Compare two documents.
     */
    public function pdfCompareDocuments(CData $handleA, CData $handleB): CData
    {
        $errorCode = FFI::new('int');
        $comp = $this->ffi->pdf_compare_documents($handleA, $handleB, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_compare_documents');
        return $comp;
    }

    /**
     * Stamp header on all pages.
     */
    public function pdfStampHeader(CData $handle, string $text, int $alignment, float $fontSize, float $margin): int
    {
        $cText = StringMarshaller::toCString($text);
        $errorCode = FFI::new('int');
        try {
            $count = $this->ffi->pdf_stamp_header($handle, $cText, $alignment, $fontSize, $margin, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_stamp_header');
            return (int)$count;
        } finally {
            unset($cText);
        }
    }

    /**
     * Stamp footer on all pages.
     */
    public function pdfStampFooter(CData $handle, string $text, int $alignment, float $fontSize, float $margin): int
    {
        $cText = StringMarshaller::toCString($text);
        $errorCode = FFI::new('int');
        try {
            $count = $this->ffi->pdf_stamp_footer($handle, $cText, $alignment, $fontSize, $margin, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_stamp_footer');
            return (int)$count;
        } finally {
            unset($cText);
        }
    }

    /**
     * Stamp header and footer on all pages.
     */
    public function pdfStampHeaderFooter(
        CData $handle,
        ?string $headerText,
        ?string $footerText,
        int $alignment,
        float $fontSize,
        float $margin
    ): int {
        $cHeader = $headerText !== null ? StringMarshaller::toCString($headerText) : null;
        $cFooter = $footerText !== null ? StringMarshaller::toCString($footerText) : null;
        $errorCode = FFI::new('int');
        try {
            $count = $this->ffi->pdf_stamp_header_footer(
                $handle, $cHeader, $cFooter, $alignment, $fontSize, $margin, FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_stamp_header_footer');
            return (int)$count;
        } finally {
            unset($cHeader, $cFooter);
        }
    }

    // ========== Utility & Helper Functions ==========

    /**
     * Free bytes allocated by native code.
     */
    public function freeBytes(CData $ptr): void
    {
        $this->ffi->free_bytes($ptr);
    }

    /**
     * Get the FFI instance directly for advanced usage.
     *
     * @return FFI The FFI instance
     * @internal
     */
    public function getFfi(): FFI
    {
        return $this->ffi;
    }

    // ============================================================
    // Phase 6 / v0.3.50-v0.3.54 bindings — see
    // docs/releases/plans/v0.3.55/feature-php-binding.md §6.
    // Every wrapper below calls a symbol that EXISTS in
    // php/include/pdf_oxide.h (verified at scaffold time);
    // callers must hold the corresponding handle's lifetime.
    // ============================================================

    // -------- Auto-extraction (v0.3.51 #519) --------

    /**
     * #519: cheap per-page text-vs-OCR classification → JSON envelope.
     * Caller `json_decode`s the returned string.
     */
    public function pdfDocumentClassifyPage(CData $handle, int $pageIndex): string
    {
        $errorCode = FFI::new('int');
        $json = $this->ffi->pdf_document_classify_page($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_classify_page', ['page' => $pageIndex]);
        return StringMarshaller::fromCString($json);
    }

    /**
     * #519: whole-document classification → JSON
     * (per-page kinds + `pages_needing_ocr`).
     */
    public function pdfDocumentClassifyDocument(CData $handle): string
    {
        $errorCode = FFI::new('int');
        $json = $this->ffi->pdf_document_classify_document($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_classify_document');
        return StringMarshaller::fromCString($json);
    }

    /**
     * #519: one-shot auto text extraction — auto-routes text vs OCR
     * with graceful native fallback. Never returns the opaque OCR
     * error #513; per spec the fallback is logged + reflected in the
     * caller's ExtractReason.
     */
    public function pdfDocumentExtractTextAuto(CData $handle, int $pageIndex): string
    {
        $errorCode = FFI::new('int');
        $text = $this->ffi->pdf_document_extract_text_auto($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_extract_text_auto', ['page' => $pageIndex]);
        return StringMarshaller::fromCString($text);
    }

    /**
     * #519: rich per-page extraction → JSON `PageExtraction`
     * (per-region bbox + typed reason; never bare-empty). `$optionsJson`
     * is `{}`-tolerant `AutoExtractOptions`; empty / null → defaults.
     */
    public function pdfDocumentExtractPageAuto(CData $handle, int $pageIndex, ?string $optionsJson = null): string
    {
        $errorCode = FFI::new('int');
        $cOpts = StringMarshaller::toCString($optionsJson ?? '{}');
        try {
            $json = $this->ffi->pdf_document_extract_page_auto(
                $handle,
                $pageIndex,
                $cOpts,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_extract_page_auto', ['page' => $pageIndex]);
            return StringMarshaller::fromCString($json);
        } finally {
            unset($cOpts);
        }
    }

    /**
     * Provision OCR models for the given languages (CSV: "eng,rus").
     * Returns the cache directory path. NOTE: per the Rust contract,
     * this only *prepares* the cache dir; downloads happen lazily on
     * first OCR call. Returns "" gracefully if the build lacks OCR.
     */
    public function pdfOxidePrefetchModels(string $languagesCsv): string
    {
        $errorCode = FFI::new('int');
        $cCsv = StringMarshaller::toCString($languagesCsv);
        try {
            $path = $this->ffi->pdf_oxide_prefetch_models($cCsv, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_oxide_prefetch_models', ['languages' => $languagesCsv]);
            return StringMarshaller::fromCString($path);
        } finally {
            unset($cCsv);
        }
    }

    /**
     * Return the model manifest as JSON. Always returns a string —
     * empty / minimal JSON if the `ocr` cargo feature is off.
     */
    public function pdfOxideModelManifest(): string
    {
        $json = $this->ffi->pdf_oxide_model_manifest();
        return StringMarshaller::fromCString($json);
    }

    /**
     * Whether the build was compiled with the `ocr` feature AND a model
     * cache appears available. Used by AutoExtractor's graceful-fallback
     * decision: false → ExtractReason::OcrRequestedButUnavailable.
     */
    public function pdfOxidePrefetchAvailable(): bool
    {
        return $this->ffi->pdf_oxide_prefetch_available() !== 0;
    }

    // -------- Document editor open/free (correct ABI names) --------

    /**
     * Open a document for editing — returns a `DocumentEditor*` handle.
     *
     * NOTE: The scaffold's {@see self::pdfDocumentEditorOpen()} calls
     * a symbol named `pdf_document_editor_open` which does NOT exist
     * in the v0.3.55 C ABI. The correct symbol is bare
     * `document_editor_open`. This wrapper uses the right name.
     */
    public function documentEditorOpen(string $path): ?CData
    {
        $cPath = StringMarshaller::toCString($path);
        $errorCode = FFI::new('int');
        try {
            $handle = $this->ffi->document_editor_open($cPath, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'document_editor_open', ['path' => $path]);
            return $handle;
        } finally {
            unset($cPath);
        }
    }

    /** Free a `DocumentEditor*` handle. */
    public function documentEditorFree(CData $editor): void
    {
        $this->ffi->document_editor_free($editor);
    }

    // -------- Destructive redaction (v0.3.50 #231) --------

    /**
     * Mark a rectangle for destructive redaction on the given page.
     * Coordinates are PDF points; color is the fill that replaces
     * the redacted region after {@see pdfRedactionApply()}.
     */
    public function pdfRedactionAdd(
        CData $editor,
        int $page,
        float $x1,
        float $y1,
        float $x2,
        float $y2,
        float $r = 0.0,
        float $g = 0.0,
        float $b = 0.0
    ): int {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_redaction_add(
            $editor,
            $page,
            $x1,
            $y1,
            $x2,
            $y2,
            $r,
            $g,
            $b,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_add', ['page' => $page]);
        return (int)$result;
    }

    /**
     * Number of pending redaction marks on a page.
     */
    public function pdfRedactionCount(CData $editor, int $page): int
    {
        $errorCode = FFI::new('int');
        $n = $this->ffi->pdf_redaction_count($editor, $page, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_count', ['page' => $page]);
        return (int)$n;
    }

    /**
     * Apply all pending redactions destructively (byte-level scrub).
     * SECURITY OP: throws RedactionException on any non-zero error_code
     * — never silently swallows.
     */
    public function pdfRedactionApply(
        CData $editor,
        bool $scrubMetadata = true,
        float $r = 0.0,
        float $g = 0.0,
        float $b = 0.0
    ): int {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_redaction_apply(
            $editor,
            $scrubMetadata,
            $r,
            $g,
            $b,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_apply');
        return (int)$result;
    }

    /**
     * Destructively wipe all document metadata (Info dict, XMP, etc.).
     * Independent of any pending rect redactions.
     */
    public function pdfRedactionScrubMetadata(CData $editor): int
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_redaction_scrub_metadata($editor, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_scrub_metadata');
        return (int)$result;
    }

    /**
     * Apply redactions for a single page only (granular variant).
     */
    public function documentEditorApplyPageRedactions(CData $editor, int $page): int
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->document_editor_apply_page_redactions(
            $editor,
            $page,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'document_editor_apply_page_redactions', ['page' => $page]);
        return (int)$result;
    }

    /**
     * Apply redactions across every marked page.
     */
    public function documentEditorApplyAllRedactions(CData $editor): int
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->document_editor_apply_all_redactions($editor, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'document_editor_apply_all_redactions');
        return (int)$result;
    }

    // -------- PAdES signature shim (v0.3.51) --------

    /**
     * Sign PDF bytes with PAdES — the 5-arg shim added in v0.3.51 for
     * binders that can't handle the legacy 18-arg `pdf_sign_bytes_pades`
     * call. PHP can do either, but the shim is the canonical entry.
     *
     * `$optionsBlob` must be a packed `PadesSignOptionsC` struct (PHP
     * FFI CData). Built by {@see \PdfOxide\Managers\SignatureManager}.
     */
    public function pdfSignBytesPadesOpts(string $pdfData, CData $optionsBlob): string
    {
        $errorCode = FFI::new('int');
        $outLen = FFI::new('size_t');
        $pdfLen = strlen($pdfData);

        $pdfBuf = FFI::new('uint8_t[' . ($pdfLen > 0 ? $pdfLen : 1) . ']', false);
        if ($pdfLen > 0) {
            FFI::memcpy($pdfBuf, $pdfData, $pdfLen);
        }

        $out = $this->ffi->pdf_sign_bytes_pades_opts(
            FFI::cast('uint8_t*', $pdfBuf),
            $pdfLen,
            FFI::addr($optionsBlob),
            FFI::addr($outLen),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_sign_bytes_pades_opts');

        $length = (int)$outLen->cdata;
        $signed = FFI::string($out, $length);
        // Free native buffer.
        $this->ffi->free_bytes(FFI::cast('uint8_t*', $out));
        FFI::free($pdfBuf);

        return $signed;
    }

    /**
     * Read back the detected PAdES level (B-B/B-T/B-LT/B-LTA) of a
     * signature handle. Returns the integer ordinal.
     */
    public function pdfSignatureGetPadesLevel(CData $signatureHandle): int
    {
        $errorCode = FFI::new('int');
        $level = $this->ffi->pdf_signature_get_pades_level($signatureHandle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_signature_get_pades_level');
        return (int)$level;
    }

    /**
     * Whether the document has at least one document-timestamp
     * (B-T or above).
     *
     * Behavior on builds without the `signatures` cargo feature:
     * the ABI returns ERR_UNSUPPORTED. Per
     * `feedback_extraction_graceful_fallback`, this read-only
     * inspection is NOT a security op — degrade to `false` rather
     * than raise.
     */
    public function pdfDocumentHasTimestamp(CData $documentHandle): bool
    {
        $errorCode = FFI::new('int');
        $r = $this->ffi->pdf_document_has_timestamp($documentHandle, FFI::addr($errorCode));
        $code = (int)$errorCode->cdata;
        if ($code === ErrorHandler::UNSUPPORTED) {
            return false;
        }
        if ($code === ErrorHandler::SIGNATURE_ERROR) {
            // Documents with no signatures: degrade rather than throw.
            return false;
        }
        ErrorHandler::check($code, 'pdf_document_has_timestamp');
        return $r !== 0;
    }

    // -------- Office converter (v0.3.48 #159) --------

    /**
     * Open a PDF document from raw DOCX bytes (converts in-memory).
     */
    public function pdfDocumentOpenFromDocxBytes(string $data): ?CData
    {
        $errorCode = FFI::new('int');
        $len = strlen($data);
        $buf = FFI::new('uint8_t[' . ($len > 0 ? $len : 1) . ']', false);
        if ($len > 0) {
            FFI::memcpy($buf, $data, $len);
        }
        try {
            $handle = $this->ffi->pdf_document_open_from_docx_bytes(
                FFI::cast('uint8_t*', $buf),
                $len,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_open_from_docx_bytes');
            return $handle;
        } finally {
            FFI::free($buf);
        }
    }

    public function pdfDocumentOpenFromPptxBytes(string $data): ?CData
    {
        $errorCode = FFI::new('int');
        $len = strlen($data);
        $buf = FFI::new('uint8_t[' . ($len > 0 ? $len : 1) . ']', false);
        if ($len > 0) {
            FFI::memcpy($buf, $data, $len);
        }
        try {
            $handle = $this->ffi->pdf_document_open_from_pptx_bytes(
                FFI::cast('uint8_t*', $buf),
                $len,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_open_from_pptx_bytes');
            return $handle;
        } finally {
            FFI::free($buf);
        }
    }

    public function pdfDocumentOpenFromXlsxBytes(string $data): ?CData
    {
        $errorCode = FFI::new('int');
        $len = strlen($data);
        $buf = FFI::new('uint8_t[' . ($len > 0 ? $len : 1) . ']', false);
        if ($len > 0) {
            FFI::memcpy($buf, $data, $len);
        }
        try {
            $handle = $this->ffi->pdf_document_open_from_xlsx_bytes(
                FFI::cast('uint8_t*', $buf),
                $len,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_open_from_xlsx_bytes');
            return $handle;
        } finally {
            FFI::free($buf);
        }
    }

    /**
     * Export the PDF as DOCX bytes (forward conversion).
     */
    public function pdfDocumentToDocxBytes(CData $handle): string
    {
        $errorCode = FFI::new('int');
        $outLen = FFI::new('size_t');
        $out = $this->ffi->pdf_document_to_docx($handle, FFI::addr($outLen), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_to_docx');
        $length = (int)$outLen->cdata;
        $bytes = FFI::string($out, $length);
        $this->ffi->free_bytes(FFI::cast('uint8_t*', $out));
        return $bytes;
    }

    public function pdfDocumentToPptxBytes(CData $handle): string
    {
        $errorCode = FFI::new('int');
        $outLen = FFI::new('size_t');
        $out = $this->ffi->pdf_document_to_pptx($handle, FFI::addr($outLen), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_to_pptx');
        $length = (int)$outLen->cdata;
        $bytes = FFI::string($out, $length);
        $this->ffi->free_bytes(FFI::cast('uint8_t*', $out));
        return $bytes;
    }

    public function pdfDocumentToXlsxBytes(CData $handle): string
    {
        $errorCode = FFI::new('int');
        $outLen = FFI::new('size_t');
        $out = $this->ffi->pdf_document_to_xlsx($handle, FFI::addr($outLen), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_to_xlsx');
        $length = (int)$outLen->cdata;
        $bytes = FFI::string($out, $length);
        $this->ffi->free_bytes(FFI::cast('uint8_t*', $out));
        return $bytes;
    }

    // -------- Split by bookmarks (v0.3.50) --------

    /**
     * Plan a split by outline bookmarks. Returns a JSON envelope the
     * caller can feed to the binding-side splitter; native side does
     * the planning only (per the v0.3.50 design — keep the cdylib
     * lean).
     *
     * `$optionsJson` is a JSON object: `{ "min_level": 1, "max_level": 2 }`.
     * `null` / empty → defaults.
     */
    public function pdfDocumentPlanSplitByBookmarks(CData $handle, ?string $optionsJson = null): string
    {
        $errorCode = FFI::new('int');
        $cOpts = StringMarshaller::toCString($optionsJson ?? '{}');
        try {
            $json = $this->ffi->pdf_document_plan_split_by_bookmarks(
                $handle,
                $cOpts,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_plan_split_by_bookmarks');
            return StringMarshaller::fromCString($json);
        } finally {
            unset($cOpts);
        }
    }

    // -------- Watermark / stamp builder ops --------

    /**
     * Append a watermark with custom text to the current page-builder.
     */
    public function pdfPageBuilderWatermark(CData $pageBuilder, string $text): int
    {
        $errorCode = FFI::new('int');
        $cText = StringMarshaller::toCString($text);
        try {
            $result = $this->ffi->pdf_page_builder_watermark($pageBuilder, $cText, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_page_builder_watermark');
            return (int)$result;
        } finally {
            unset($cText);
        }
    }

    /** "CONFIDENTIAL" preset watermark. */
    public function pdfPageBuilderWatermarkConfidential(CData $pageBuilder): int
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_page_builder_watermark_confidential($pageBuilder, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_page_builder_watermark_confidential');
        return (int)$result;
    }

    /** "DRAFT" preset watermark. */
    public function pdfPageBuilderWatermarkDraft(CData $pageBuilder): int
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_page_builder_watermark_draft($pageBuilder, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_page_builder_watermark_draft');
        return (int)$result;
    }
}
