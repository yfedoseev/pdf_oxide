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
        $errorCode = $this->ffi->new('int');

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
        $major = $this->ffi->new('uint8_t');
        $minor = $this->ffi->new('uint8_t');

        $this->ffi->pdf_document_get_version($handle, FFI::addr($major), FFI::addr($minor));

        return [
            'major' => (int) $major->cdata,
            'minor' => (int) $minor->cdata,
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
        $errorCode = $this->ffi->new('int');
        $count = $this->ffi->pdf_document_get_page_count($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_page_count');
        return (int) $count;
    }

    /**
     * Check if document has structure tree.
     *
     * @param CData $handle The document handle
     * @return bool True if document has structure tree
     */
    public function pdfDocumentHasStructureTree(CData $handle): bool
    {
        return (bool) $this->ffi->pdf_document_has_structure_tree($handle);
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
        $errorCode = $this->ffi->new('int');
        $text = $this->ffi->pdf_document_extract_text($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_extract_text', ['page' => $pageIndex]);
        return StringMarshaller::fromCString($text);
    }

    /**
     * Extract structured page layout as a JSON string (#536).
     *
     * @param CData $handle The document handle
     * @param int $pageIndex Zero-based page index
     * @return string Serialized StructuredPage JSON
     */
    public function pdfDocumentExtractStructuredToJson(CData $handle, int $pageIndex): string
    {
        $errorCode = $this->ffi->new('int');
        $json = $this->ffi->pdf_document_extract_structured_to_json($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_extract_structured_to_json', ['page' => $pageIndex]);
        return StringMarshaller::fromCString($json);
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');

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
        $errorCode = $this->ffi->new('int');

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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
        $images = $this->ffi->pdf_document_get_embedded_images($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_embedded_images', ['page' => $pageIndex]);
        return $images;
    }


    /**
     * Get search result count.
     *
     * @param CData $resultsHandle The search results handle
     * @return int Number of results
     */
    public function oxideSearchResultCount(CData $resultsHandle): int
    {
        return (int) $this->ffi->pdf_oxide_search_result_count($resultsHandle);
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
        $errorCode = $this->ffi->new('int');
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
        // C: int32_t pdf_oxide_search_result_get_page(results, index, *err).
        // Pre-fix omitted the err pointer → cdylib wrote through register
        // garbage; same root cause as the v0.3.55 Ruby aarch64 segfaults
        // (#547, commit a9cff143).
        $errorCode = $this->ffi->new('int32_t');
        $page = $this->ffi->pdf_oxide_search_result_get_page($resultsHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_search_result_get_page');
        return (int) $page;
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
        $x = $this->ffi->new('float');
        $y = $this->ffi->new('float');
        $width = $this->ffi->new('float');
        $height = $this->ffi->new('float');
        $errorCode = $this->ffi->new('int32_t');

        // C: void pdf_oxide_search_result_get_bbox(results, index,
        //                                          *x, *y, *w, *h, *err)
        $this->ffi->pdf_oxide_search_result_get_bbox(
            $resultsHandle,
            $index,
            FFI::addr($x),
            FFI::addr($y),
            FFI::addr($width),
            FFI::addr($height),
            FFI::addr($errorCode)
        );
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_search_result_get_bbox');

        return [
            'x' => (float) $x->cdata,
            'y' => (float) $y->cdata,
            'width' => (float) $width->cdata,
            'height' => (float) $height->cdata,
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
        return (int) $this->ffi->pdf_oxide_annotation_count($listHandle);
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
        // C: char *pdf_oxide_annotation_get_type(annotations, index, *err)
        $errorCode = $this->ffi->new('int32_t');
        $type = $this->ffi->pdf_oxide_annotation_get_type($listHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_annotation_get_type');
        return StringMarshaller::fromCString($type);
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
        $errorCode = $this->ffi->new('int32_t');
        $content = $this->ffi->pdf_oxide_annotation_get_content($listHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_annotation_get_content');
        return StringMarshaller::fromCString($content);
    }

    /**
     * Free annotation list.
     *
     * @param CData $listHandle The annotation list handle
     */
    public function oxideAnnotationFree(CData $listHandle): void
    {
        $this->ffi->pdf_oxide_annotation_list_free($listHandle);
    }

    /**
     * Get font count.
     *
     * @param CData $listHandle The font list handle
     * @return int Number of fonts
     */
    public function oxideFontCount(CData $listHandle): int
    {
        return (int) $this->ffi->pdf_oxide_font_count($listHandle);
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
        $errorCode = $this->ffi->new('int32_t');
        $name = $this->ffi->pdf_oxide_font_get_name($listHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_font_get_name');
        return StringMarshaller::fromCString($name);
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
        $errorCode = $this->ffi->new('int32_t');
        $type = $this->ffi->pdf_oxide_font_get_type($listHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_font_get_type');
        return StringMarshaller::fromCString($type);
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
        $errorCode = $this->ffi->new('int32_t');
        $embedded = $this->ffi->pdf_oxide_font_is_embedded($listHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_font_is_embedded');
        return ((int) $embedded) !== 0;
    }

    /**
     * Free font list.
     *
     * @param CData $listHandle The font list handle
     */
    public function oxideFontFree(CData $listHandle): void
    {
        $this->ffi->pdf_oxide_font_list_free($listHandle);
    }

    /**
     * Get image count.
     *
     * @param CData $listHandle The image list handle
     * @return int Number of images
     */
    public function oxideImageCount(CData $listHandle): int
    {
        return (int) $this->ffi->pdf_oxide_image_count($listHandle);
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
        $errorCode = $this->ffi->new('int32_t');
        $w = $this->ffi->pdf_oxide_image_get_width($listHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_image_get_width');
        return (int) $w;
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
        $errorCode = $this->ffi->new('int32_t');
        $h = $this->ffi->pdf_oxide_image_get_height($listHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_image_get_height');
        return (int) $h;
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
        $errorCode = $this->ffi->new('int32_t');
        $format = $this->ffi->pdf_oxide_image_get_format($listHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_image_get_format');
        return StringMarshaller::fromCString($format);
    }

    /**
     * Free image list.
     *
     * @param CData $listHandle The image list handle
     */
    public function oxideImageFree(CData $listHandle): void
    {
        $this->ffi->pdf_oxide_image_list_free($listHandle);
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
        $errorCode = $this->ffi->new('int');
        $imageHandle = $this->ffi->pdf_render_page($handle, $pageIndex, $options, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_render_page');
        return $imageHandle;
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
        $imageHandle = $this->ffi->pdf_render_page_thumbnail($handle, $pageIndex, $maxSize, $options, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_render_page_thumbnail');
        return $imageHandle;
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
        // C: int32_t pdf_estimate_render_time(const void *_doc,
        //                                     int32_t _page_index,
        //                                     int32_t *error_code)
        // The underscore-prefixed params signal Rust phantoms — the
        // function is a stub. Pre-fix passed `$options` (a CData
        // pointer) into the *err slot, so the cdylib wrote the error
        // code through wherever `$options` pointed. Pass a proper
        // err buffer here; `$options` retained on the wrapper API for
        // forward-compat once the cdylib implementation lands.
        $errorCode = $this->ffi->new('int32_t');
        unset($options);
        $estimate = $this->ffi->pdf_estimate_render_time($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_estimate_render_time');
        return (int) $estimate;
    }


    /**
     * Generate a QR code.
     *
     * @param string $data Data to encode in QR code
     * @param int $errorCorrection Error-correction level (0=L, 1=M, 2=Q, 3=H)
     * @param int $sizePx Output bitmap edge in pixels
     * @return CData Barcode handle
     */
    public function pdfGenerateQrCode(string $data, int $errorCorrection = 1, int $sizePx = 256): CData
    {
        // C: FfiBarcodeImage *pdf_generate_qr_code(const char *data,
        //        int32_t error_correction, int32_t size_px, int32_t *err)
        // Pre-fix omitted `error_correction` AND `size_px` — `$size`
        // landed in the EC slot, `*err` landed in `size_px`, and the
        // cdylib wrote to whatever was in the actual `*err` register.
        $cData = StringMarshaller::toCString($data);
        $errorCode = $this->ffi->new('int32_t');

        try {
            $barcodeHandle = $this->ffi->pdf_generate_qr_code(
                $cData,
                $errorCorrection,
                $sizePx,
                FFI::addr($errorCode)
            );
            ErrorHandler::check((int) $errorCode->cdata, 'pdf_generate_qr_code');
            return $barcodeHandle;
        } finally {
            unset($cData);
        }
    }

    /**
     * Generate a barcode.
     *
     * @param string $data Data to encode
     * @param int $format Barcode format ordinal (matches Rust BarcodeFormat enum)
     * @param int $sizePx Output bitmap edge in pixels
     * @return CData Barcode handle
     */
    public function pdfGenerateBarcode(string $data, int $format, int $sizePx = 256): CData
    {
        // C: FfiBarcodeImage *pdf_generate_barcode(const char *data,
        //        int32_t format, int32_t size_px, int32_t *err)
        // Pre-fix omitted `size_px` AND the format was passed as a
        // string instead of the int32 ordinal.
        $cData = StringMarshaller::toCString($data);
        $errorCode = $this->ffi->new('int32_t');

        try {
            $barcodeHandle = $this->ffi->pdf_generate_barcode(
                $cData,
                $format,
                $sizePx,
                FFI::addr($errorCode)
            );
            ErrorHandler::check((int) $errorCode->cdata, 'pdf_generate_barcode');
            return $barcodeHandle;
        } finally {
            unset($cData);
        }
    }

    /**
     * Get barcode image as PNG.
     *
     * @param CData $barcodeHandle Barcode handle
     * @param int $sizePx Output PNG edge in pixels
     * @return string PNG binary data
     */
    public function pdfBarcodeGetImagePng(CData $barcodeHandle, int $sizePx = 256): string
    {
        // C: uint8_t *pdf_barcode_get_image_png(handle, int32_t _size_px,
        //                                       uintptr_t *out_len, int32_t *err)
        // Pre-fix passed only `(handle, FFI::addr($sizePtr))` — `*out_len`
        // landed in `_size_px` slot, `*err` slot was register garbage.
        $outLen = $this->ffi->new('size_t');
        $errorCode = $this->ffi->new('int32_t');
        $dataPtr = $this->ffi->pdf_barcode_get_image_png(
            $barcodeHandle,
            $sizePx,
            FFI::addr($outLen),
            FFI::addr($errorCode)
        );
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_barcode_get_image_png');
        $size = (int) $outLen->cdata;
        $bytes = FFI::string($dataPtr, $size);
        // PNG buffer is heap-allocated by the cdylib — must free.
        $this->ffi->free_bytes($this->ffi->cast('uint8_t*', $dataPtr));
        return $bytes;
    }

    /**
     * Get barcode as SVG.
     *
     * @param CData $barcodeHandle Barcode handle
     * @param int $sizePx SVG canvas edge in pixels (sizing hint)
     * @return string SVG XML string
     */
    public function pdfBarcodeGetSvg(CData $barcodeHandle, int $sizePx = 256): string
    {
        // C: char *pdf_barcode_get_svg(handle, int32_t _size_px, int32_t *err)
        $errorCode = $this->ffi->new('int32_t');
        $svg = $this->ffi->pdf_barcode_get_svg($barcodeHandle, $sizePx, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_barcode_get_svg');
        return StringMarshaller::fromCString($svg);
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
        $errorCode = $this->ffi->new('int');
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
     * Create an OCR engine from model assets on disk.
     *
     * @param string $detectionModelPath PaddleOCR detection ONNX model path
     * @param string $recognitionModelPath PaddleOCR recognition ONNX model path
     * @param string $dictionaryPath Character dictionary path
     * @return CData OCR engine handle
     */
    public function pdfOcrEngineCreate(
        string $detectionModelPath,
        string $recognitionModelPath,
        string $dictionaryPath
    ): CData {
        // C: void *pdf_ocr_engine_create(const char *det_model_path,
        //     const char *rec_model_path, const char *dict_path,
        //     int32_t *err)
        // Pre-fix passed only `FFI::addr($err)` — the err pointer landed
        // in the det_model_path slot (read as `const char*` → crash on
        // first use).
        $cDet = StringMarshaller::toCString($detectionModelPath);
        $cRec = StringMarshaller::toCString($recognitionModelPath);
        $cDict = StringMarshaller::toCString($dictionaryPath);
        $errorCode = $this->ffi->new('int32_t');
        try {
            $engine = $this->ffi->pdf_ocr_engine_create(
                $cDet,
                $cRec,
                $cDict,
                FFI::addr($errorCode)
            );
            ErrorHandler::check((int) $errorCode->cdata, 'pdf_ocr_engine_create');
            return $engine;
        } finally {
            unset($cDet, $cRec, $cDict);
        }
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
     * Check if page needs OCR.
     *
     * @param CData $handle Document handle
     * @param int $pageIndex Page index
     * @return bool True if page needs OCR
     */
    public function pdfOcrPageNeedsOcr(CData $handle, int $pageIndex): bool
    {
        $errorCode = $this->ffi->new('int32_t');
        $needs = $this->ffi->pdf_ocr_page_needs_ocr($handle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_ocr_page_needs_ocr');
        return (bool) $needs;
    }


    /**
     * Extract OCR'd text from a single page.
     *
     * @param CData $docHandle Document handle
     * @param int $pageIndex 0-based page index
     * @param CData $engine OCR engine handle (from {@see pdfOcrEngineCreate})
     * @return string Extracted text
     */
    public function pdfOcrExtractText(CData $docHandle, int $pageIndex, CData $engine): string
    {
        // C: char *pdf_ocr_extract_text(PdfDocument *doc,
        //     int32_t page_index, const void *engine, int32_t *err)
        // Pre-fix passed only `(results)` — treated as `doc`,
        // remaining 3 slots were register garbage.
        $errorCode = $this->ffi->new('int32_t');
        $text = $this->ffi->pdf_ocr_extract_text(
            $docHandle,
            $pageIndex,
            $engine,
            FFI::addr($errorCode)
        );
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_ocr_extract_text');
        return StringMarshaller::fromCString($text);
    }


    /**
     * Check if PDF/A compliant.
     *
     * @param CData $resultHandle Validation result handle
     * @return bool True if document is PDF/A compliant
     */
    public function pdfPdfAIsCompliant(CData $resultHandle): bool
    {
        // C: bool pdf_pdf_a_is_compliant(const FfiPdfAResults *results, int32_t *err)
        $errorCode = $this->ffi->new('int32_t');
        $compliant = $this->ffi->pdf_pdf_a_is_compliant($resultHandle, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_pdf_a_is_compliant');
        return (bool) $compliant;
    }

    /**
     * Get PDF/A error count.
     *
     * @param CData $resultHandle Validation result handle
     * @return int Number of errors
     */
    public function pdfPdfAErrorCount(CData $resultHandle): int
    {
        return (int) $this->ffi->pdf_pdf_a_error_count($resultHandle);
    }

    /**
     * Get PDF/A warning count.
     *
     * @param CData $resultHandle Validation result handle
     * @return int Number of warnings
     */
    public function pdfPdfAWarningCount(CData $resultHandle): int
    {
        return (int) $this->ffi->pdf_pdf_a_warning_count($resultHandle);
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
        $errorCode = $this->ffi->new('int32_t');
        $error = $this->ffi->pdf_pdf_a_get_error($resultHandle, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_pdf_a_get_error');
        return StringMarshaller::fromCString($error);
    }


    /**
     * Check if PDF/X compliant.
     *
     * @param CData $resultHandle Validation result handle
     * @return bool True if document is PDF/X compliant
     */
    public function pdfPdfXIsCompliant(CData $resultHandle): bool
    {
        $errorCode = $this->ffi->new('int32_t');
        $compliant = $this->ffi->pdf_pdf_x_is_compliant($resultHandle, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_pdf_x_is_compliant');
        return (bool) $compliant;
    }

    /**
     * Get PDF/X error count.
     *
     * @param CData $resultHandle Validation result handle
     * @return int Number of errors
     */
    public function pdfPdfXErrorCount(CData $resultHandle): int
    {
        return (int) $this->ffi->pdf_pdf_x_error_count($resultHandle);
    }


    /**
     * Validate PDF/UA accessibility at the requested level.
     *
     * @param CData $handle Document handle
     * @param int $level PDF/UA level ordinal (see {@see \PdfOxide\PdfValidator::PDFUA_1} / `PDFUA_2`)
     * @return CData Validation result handle
     */
    public function pdfValidatePdfUa(CData $handle, int $level = 1): CData
    {
        // C: FfiUaResults *pdf_validate_pdf_ua(PdfDocument *document,
        //                                      int32_t level, int32_t *err)
        // Pre-fix omitted `level` — `*err` landed in the level slot
        // and *err slot was register garbage. The cdylib defaults
        // level == 2 → UA-2, else UA-1.
        $errorCode = $this->ffi->new('int32_t');
        $resultHandle = $this->ffi->pdf_validate_pdf_ua($handle, $level, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_validate_pdf_ua');
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
        $errorCode = $this->ffi->new('int32_t');
        $accessible = $this->ffi->pdf_pdf_ua_is_accessible($resultHandle, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_pdf_ua_is_accessible');
        return (bool) $accessible;
    }

    /**
     * Get PDF/UA error count.
     *
     * @param CData $resultHandle Validation result handle
     * @return int Number of accessibility issues
     */
    public function pdfPdfUaErrorCount(CData $resultHandle): int
    {
        return (int) $this->ffi->pdf_pdf_ua_error_count($resultHandle);
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
        $errorCode = $this->ffi->new('int');

        try {
            $this->ffi->pdf_convert_to_pdf_a($handle, $cLevel, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_convert_to_pdf_a');
        } finally {
            unset($cLevel);
        }
    }


    /**
     * Get signature count.
     *
     * @param CData $handle Document handle
     * @return int Number of signatures
     */
    public function pdfDocumentGetSignatureCount(CData $handle): int
    {
        // C: int32_t pdf_document_get_signature_count(const PdfDocument *handle, int32_t *err)
        $errorCode = $this->ffi->new('int32_t');
        $count = $this->ffi->pdf_document_get_signature_count($handle, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_document_get_signature_count');
        return (int) $count;
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
        $errorCode = $this->ffi->new('int');
        $signatureHandle = $this->ffi->pdf_document_get_signature($handle, $index, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_signature');
        return $signatureHandle;
    }


    /**
     * Verify signature against the embedded cert chain.
     *
     * <p>The cdylib's signature handle already carries the cert chain
     * — the wire format does NOT take a separate certificate handle.
     * The pre-v0.3.55 PHP wrapper accepted (and silently mispositioned)
     * one; callers passing a non-null cert handle ended up with the
     * cdylib writing the error code into wherever the cert handle
     * pointer landed.
     *
     * @param CData $signatureHandle Signature handle
     * @return bool True if signature is valid
     */
    public function pdfSignatureVerify(CData $signatureHandle): bool
    {
        // C: bool pdf_signature_verify(const void *signature_handle, int32_t *err)
        $errorCode = $this->ffi->new('int32_t');
        $valid = $this->ffi->pdf_signature_verify($signatureHandle, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_signature_verify');
        return (bool) $valid;
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
        // PKCS#12 / PEM cert data is BINARY (contains bytes >= 0x80).
        // Pre-v0.3.55 this went through StringMarshaller::toCString
        // which allocates `char[N+1]` and forces a `$this->ffi->cast('uint8_t*', …)`
        // — on PHP 8.5 with `char` defaulting to signed, that cast
        // segfaults the moment the cdylib touches a byte with the high
        // bit set. Allocate the input as `uint8_t[N]` directly so no
        // sign-aware cast is needed.
        //
        // Diagnosis: /tmp/php_signer_repro.php with `char[N+1]` SEGV's
        // every time; with `uint8_t[N]` (owned or unowned) it returns
        // err=0 + non-null cert handle. The C ABI doesn't expect NUL
        // termination on a length-prefixed buffer.
        $certLen = strlen($certData);
        $cData = $this->ffi->new('uint8_t[' . ($certLen > 0 ? $certLen : 1) . ']');
        if ($certLen > 0) {
            FFI::memcpy($cData, $certData, $certLen);
        }
        $cPassword = StringMarshaller::toCString($password); // password is a text string — toCString is correct here.
        $errorCode = $this->ffi->new('int32_t');

        try {
            $certHandle = $this->ffi->pdf_certificate_load_from_bytes(
                $cData,
                $certLen,
                $cPassword,
                FFI::addr($errorCode)
            );
            ErrorHandler::check((int) $errorCode->cdata, 'pdf_certificate_load_from_bytes');
            return $certHandle;
        } finally {
            unset($cData, $cPassword);
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
        // C: char *pdf_certificate_get_subject(const void *cert, int32_t *err)
        $errorCode = $this->ffi->new('int32_t');
        $subject = $this->ffi->pdf_certificate_get_subject($certificateHandle, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_certificate_get_subject');
        return StringMarshaller::fromCString($subject);
    }

    /**
     * Get certificate issuer.
     *
     * @param CData $certificateHandle Certificate handle
     * @return string Certificate issuer DN
     */
    public function pdfCertificateGetIssuer(CData $certificateHandle): string
    {
        $errorCode = $this->ffi->new('int32_t');
        $issuer = $this->ffi->pdf_certificate_get_issuer($certificateHandle, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_certificate_get_issuer');
        return StringMarshaller::fromCString($issuer);
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


    // ==================== SECURITY & PERMISSIONS ====================

    /**
     * Check if document is encrypted.
     */
    public function pdfDocumentIsEncrypted(CData $handle): bool
    {
        return (bool) $this->ffi->pdf_document_is_encrypted($handle);
    }


    // ==================== PAGE DOM OPERATIONS ====================

    /**
     * Get page width in user units.
     *
     * <p>The cdylib has no "page handle" concept — width/height live
     * on the document and are indexed by page number. The pre-v0.3.55
     * PHP wrapper took a {@code $pageHandle} and passed it as if it
     * were a {@code PdfDocument*}; the two missing args were read as
     * register garbage.
     */
    public function pdfPageGetWidth(CData $docHandle, int $pageIndex): float
    {
        // C: float pdf_page_get_width(const PdfDocument *handle, int32_t page_index, int32_t *err)
        $errorCode = $this->ffi->new('int32_t');
        $width = $this->ffi->pdf_page_get_width($docHandle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_page_get_width');
        return (float) $width;
    }

    /**
     * Get page height in user units.
     */
    public function pdfPageGetHeight(CData $docHandle, int $pageIndex): float
    {
        $errorCode = $this->ffi->new('int32_t');
        $height = $this->ffi->pdf_page_get_height($docHandle, $pageIndex, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_page_get_height');
        return (float) $height;
    }


    // ==================== PDF CREATION FUNCTIONS ====================

    /**
     * Create PDF from Markdown content.
     */
    public function pdfFromMarkdown(string $markdown): ?CData
    {
        $cMarkdown = StringMarshaller::toCString($markdown);
        $errorCode = $this->ffi->new('int');

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
        $errorCode = $this->ffi->new('int');

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
        $errorCode = $this->ffi->new('int');

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
        $errorCode = $this->ffi->new('int');

        try {
            $this->ffi->pdf_save($handle, $cPath, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_save', ['path' => $path]);
        } finally {
            unset($cPath, $errorCode);
        }
    }

    /**
     * Save PDF to bytes.
     *
     * <p>C: {@code uint8_t *pdf_save_to_bytes(Pdf *handle,
     * uintptr_t *data_len, int32_t *err)} — the buffer is the RETURN
     * VALUE, with the length written through {@code *data_len}. The
     * pre-v0.3.55 wrapper modelled a phantom {@code char**}
     * out-parameter that doesn't exist in the C ABI; reading
     * {@code $outputPtr->cdata} after the call read uninitialised
     * memory because the cdylib never wrote there. {@see \PdfOxide\Pdf::save}
     * already calls the C symbol directly and correctly — this
     * wrapper now uses the same shape so it can be a drop-in.
     */
    public function pdfSaveToBytes(CData $handle): string
    {
        $outputLen = $this->ffi->new('size_t');
        $errorCode = $this->ffi->new('int32_t');

        $ptr = $this->ffi->pdf_save_to_bytes(
            $handle,
            FFI::addr($outputLen),
            FFI::addr($errorCode)
        );
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_save_to_bytes');

        $len = (int) $outputLen->cdata;
        if ($ptr === null || $len === 0) {
            return '';
        }
        $bytes = FFI::string($ptr, $len);
        // Owned uint8_t* — free via the cdylib's free_bytes.
        $this->ffi->free_bytes($this->ffi->cast('uint8_t*', $ptr));
        return $bytes;
    }

    /**
     * Get page count from PDF handle.
     */
    public function pdfGetPageCount(CData $handle): int
    {
        $errorCode = $this->ffi->new('int');
        try {
            $count = (int) $this->ffi->pdf_get_page_count($handle, FFI::addr($errorCode));
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


    // ==================== FONT ACCESSORS ====================

    /**
     * Get font list count.
     */
    public function pdfOxideFontCount(CData $fontList): int
    {
        return (int) $this->ffi->pdf_oxide_font_count($fontList);
    }

    /**
     * Get font name by index.
     */
    public function pdfOxideFontGetName(CData $fontList, int $index): string
    {
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        // C: int32_t pdf_oxide_font_is_embedded(const FfiFontList *fonts, int32_t index, int32_t *err)
        $errorCode = $this->ffi->new('int32_t');
        $embedded = $this->ffi->pdf_oxide_font_is_embedded($fontList, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_font_is_embedded');
        return ((int) $embedded) !== 0;
    }

    /**
     * Check if font is subset.
     */
    public function pdfOxideFontIsSubset(CData $fontList, int $index): bool
    {
        $errorCode = $this->ffi->new('int32_t');
        $subset = $this->ffi->pdf_oxide_font_is_subset($fontList, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_font_is_subset');
        return ((int) $subset) !== 0;
    }

    /**
     * Get font size by index.
     */
    public function pdfOxideFontGetSize(CData $fontList, int $index): float
    {
        $errorCode = $this->ffi->new('int32_t');
        $size = $this->ffi->pdf_oxide_font_get_size($fontList, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_font_get_size');
        return (float) $size;
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
        return (int) $this->ffi->pdf_oxide_image_count($imageList);
    }

    /**
     * Get image width by index.
     */
    public function pdfOxideImageGetWidth(CData $imageList, int $index): int
    {
        $errorCode = $this->ffi->new('int32_t');
        $w = $this->ffi->pdf_oxide_image_get_width($imageList, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_image_get_width');
        return (int) $w;
    }

    /**
     * Get image height by index.
     */
    public function pdfOxideImageGetHeight(CData $imageList, int $index): int
    {
        $errorCode = $this->ffi->new('int32_t');
        $h = $this->ffi->pdf_oxide_image_get_height($imageList, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_image_get_height');
        return (int) $h;
    }

    /**
     * Get image format by index.
     */
    public function pdfOxideImageGetFormat(CData $imageList, int $index): string
    {
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int32_t');
        $bpc = $this->ffi->pdf_oxide_image_get_bits_per_component($imageList, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_image_get_bits_per_component');
        return (int) $bpc;
    }

    /**
     * Get image data by index.
     */
    public function pdfOxideImageGetData(CData $imageList, int $index): string
    {
        $outSize = $this->ffi->new('size_t');
        $errorCode = $this->ffi->new('int32_t');

        $dataPtr = $this->ffi->pdf_oxide_image_get_data(
            $imageList,
            $index,
            FFI::addr($outSize),
            FFI::addr($errorCode)
        );
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_image_get_data');

        $size = (int) $outSize->cdata;
        if ($dataPtr === null || $size === 0) {
            return '';
        }
        $bytes = FFI::string($dataPtr, $size);
        // Owned uint8_t* — free via the cdylib's free_bytes.
        $this->ffi->free_bytes($this->ffi->cast('uint8_t*', $dataPtr));
        return $bytes;
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
        return (int) $this->ffi->pdf_oxide_search_result_count($results);
    }

    /**
     * Get search result text by index.
     */
    public function pdfOxideSearchResultGetText(CData $results, int $index): string
    {
        $errorCode = $this->ffi->new('int');
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
        // C: int32_t pdf_oxide_search_result_get_page(results, index, *err)
        // Same off-by-one trailing-err bug class as #547 round 3.
        $errorCode = $this->ffi->new('int32_t');
        $page = $this->ffi->pdf_oxide_search_result_get_page($results, $index, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_oxide_search_result_get_page');
        return (int) $page;
    }


    /**
     * Get search result bounding box by index.
     */
    public function pdfOxideSearchResultGetBbox(CData $results, int $index): array
    {
        $x = $this->ffi->new('float');
        $y = $this->ffi->new('float');
        $width = $this->ffi->new('float');
        $height = $this->ffi->new('float');
        $errorCode = $this->ffi->new('int');

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


    // ========== XFA Form Functions ==========

    /**
     * Check if document has XFA form.
     */
    public function pdfDocumentHasXfa(CData $handle): bool
    {
        return (bool) $this->ffi->pdf_document_has_xfa($handle);
    }


    // ========== Advanced Signature Functions (unique additions) ==========


    /**
     * Get certificate serial number.
     */
    public function pdfCertificateGetSerial(CData $cert): string
    {
        // C: char *pdf_certificate_get_serial(const void *cert, int32_t *err)
        $errorCode = $this->ffi->new('int32_t');
        $serial = $this->ffi->pdf_certificate_get_serial($cert, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_certificate_get_serial');
        return StringMarshaller::fromCString($serial);
    }


    /**
     * Get signature signing time.
     */
    public function pdfSignatureGetSigningTime(CData $sig): string
    {
        $errorCode = $this->ffi->new('int32_t');
        $time = $this->ffi->pdf_signature_get_signing_time($sig, FFI::addr($errorCode));
        ErrorHandler::check((int) $errorCode->cdata, 'pdf_signature_get_signing_time');
        return StringMarshaller::fromCString($time);
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        $errorCode = $this->ffi->new('int');
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
        return (int) $result;
    }

    /**
     * Number of pending redaction marks on a page.
     */
    public function pdfRedactionCount(CData $editor, int $page): int
    {
        $errorCode = $this->ffi->new('int');
        $n = $this->ffi->pdf_redaction_count($editor, $page, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_count', ['page' => $page]);
        return (int) $n;
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
        $errorCode = $this->ffi->new('int');
        $result = $this->ffi->pdf_redaction_apply(
            $editor,
            $scrubMetadata,
            $r,
            $g,
            $b,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_apply');
        return (int) $result;
    }

    /**
     * Destructively wipe all document metadata (Info dict, XMP, etc.).
     * Independent of any pending rect redactions.
     */
    public function pdfRedactionScrubMetadata(CData $editor): int
    {
        $errorCode = $this->ffi->new('int');
        $result = $this->ffi->pdf_redaction_scrub_metadata($editor, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_scrub_metadata');
        return (int) $result;
    }

    /**
     * Apply redactions for a single page only (granular variant).
     */
    public function documentEditorApplyPageRedactions(CData $editor, int $page): int
    {
        $errorCode = $this->ffi->new('int');
        $result = $this->ffi->document_editor_apply_page_redactions(
            $editor,
            $page,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'document_editor_apply_page_redactions', ['page' => $page]);
        return (int) $result;
    }

    /**
     * Apply redactions across every marked page.
     */
    public function documentEditorApplyAllRedactions(CData $editor): int
    {
        $errorCode = $this->ffi->new('int');
        $result = $this->ffi->document_editor_apply_all_redactions($editor, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'document_editor_apply_all_redactions');
        return (int) $result;
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
        $errorCode = $this->ffi->new('int');
        $outLen = $this->ffi->new('size_t');
        $pdfLen = strlen($pdfData);

        $pdfBuf = $this->ffi->new('uint8_t[' . ($pdfLen > 0 ? $pdfLen : 1) . ']', false);
        if ($pdfLen > 0) {
            FFI::memcpy($pdfBuf, $pdfData, $pdfLen);
        }

        $out = $this->ffi->pdf_sign_bytes_pades_opts(
            $this->ffi->cast('uint8_t*', $pdfBuf),
            $pdfLen,
            FFI::addr($optionsBlob),
            FFI::addr($outLen),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_sign_bytes_pades_opts');

        $length = (int) $outLen->cdata;
        $signed = FFI::string($out, $length);
        // Free native buffer.
        $this->ffi->free_bytes($this->ffi->cast('uint8_t*', $out));
        FFI::free($pdfBuf);

        return $signed;
    }

    /**
     * Read back the detected PAdES level (B-B/B-T/B-LT/B-LTA) of a
     * signature handle. Returns the integer ordinal.
     */
    public function pdfSignatureGetPadesLevel(CData $signatureHandle): int
    {
        $errorCode = $this->ffi->new('int');
        $level = $this->ffi->pdf_signature_get_pades_level($signatureHandle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_signature_get_pades_level');
        return (int) $level;
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
        $errorCode = $this->ffi->new('int');
        $r = $this->ffi->pdf_document_has_timestamp($documentHandle, FFI::addr($errorCode));
        $code = (int) $errorCode->cdata;
        if ($code === ErrorHandler::UNSUPPORTED) {
            // Documents with no signatures: degrade rather than throw.
            // (Pre-v0.3.55 also handled cdylib code 8 here under the
            //  alias SIGNATURE_ERROR — the rename to UNSUPPORTED
            //  realigns PHP with the C ABI in src/ffi.rs:98.)
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
        $errorCode = $this->ffi->new('int');
        $len = strlen($data);
        $buf = $this->ffi->new('uint8_t[' . ($len > 0 ? $len : 1) . ']', false);
        if ($len > 0) {
            FFI::memcpy($buf, $data, $len);
        }
        try {
            $handle = $this->ffi->pdf_document_open_from_docx_bytes(
                $this->ffi->cast('uint8_t*', $buf),
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
        $errorCode = $this->ffi->new('int');
        $len = strlen($data);
        $buf = $this->ffi->new('uint8_t[' . ($len > 0 ? $len : 1) . ']', false);
        if ($len > 0) {
            FFI::memcpy($buf, $data, $len);
        }
        try {
            $handle = $this->ffi->pdf_document_open_from_pptx_bytes(
                $this->ffi->cast('uint8_t*', $buf),
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
        $errorCode = $this->ffi->new('int');
        $len = strlen($data);
        $buf = $this->ffi->new('uint8_t[' . ($len > 0 ? $len : 1) . ']', false);
        if ($len > 0) {
            FFI::memcpy($buf, $data, $len);
        }
        try {
            $handle = $this->ffi->pdf_document_open_from_xlsx_bytes(
                $this->ffi->cast('uint8_t*', $buf),
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
        $errorCode = $this->ffi->new('int');
        $outLen = $this->ffi->new('size_t');
        $out = $this->ffi->pdf_document_to_docx($handle, FFI::addr($outLen), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_to_docx');
        $length = (int) $outLen->cdata;
        $bytes = FFI::string($out, $length);
        $this->ffi->free_bytes($this->ffi->cast('uint8_t*', $out));
        return $bytes;
    }

    public function pdfDocumentToPptxBytes(CData $handle): string
    {
        $errorCode = $this->ffi->new('int');
        $outLen = $this->ffi->new('size_t');
        $out = $this->ffi->pdf_document_to_pptx($handle, FFI::addr($outLen), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_to_pptx');
        $length = (int) $outLen->cdata;
        $bytes = FFI::string($out, $length);
        $this->ffi->free_bytes($this->ffi->cast('uint8_t*', $out));
        return $bytes;
    }

    public function pdfDocumentToXlsxBytes(CData $handle): string
    {
        $errorCode = $this->ffi->new('int');
        $outLen = $this->ffi->new('size_t');
        $out = $this->ffi->pdf_document_to_xlsx($handle, FFI::addr($outLen), FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_to_xlsx');
        $length = (int) $outLen->cdata;
        $bytes = FFI::string($out, $length);
        $this->ffi->free_bytes($this->ffi->cast('uint8_t*', $out));
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
        $errorCode = $this->ffi->new('int');
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

    /**
     * Real outline accessor — returns the full bookmark tree as a JSON
     * array of `{title, dest, children}` records.
     *
     * Replaces the pre-v0.3.55 scaffold's `pdfDocumentGetOutlineCount`
     * / `_Title` / `_Page` / `_Level` family, none of which exist in
     * the real C ABI. Always returns valid JSON (possibly `[]`) — the
     * native side promotes outline-read errors to an empty array.
     */
    public function pdfDocumentGetOutline(CData $handle): string
    {
        $errorCode = $this->ffi->new('int');
        $json = $this->ffi->pdf_document_get_outline($handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_outline');
        return StringMarshaller::fromCString($json);
    }

    // -------- Watermark / stamp builder ops --------

    /**
     * Append a watermark with custom text to the current page-builder.
     */
    public function pdfPageBuilderWatermark(CData $pageBuilder, string $text): int
    {
        $errorCode = $this->ffi->new('int');
        $cText = StringMarshaller::toCString($text);
        try {
            $result = $this->ffi->pdf_page_builder_watermark($pageBuilder, $cText, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_page_builder_watermark');
            return (int) $result;
        } finally {
            unset($cText);
        }
    }

    /** "CONFIDENTIAL" preset watermark. */
    public function pdfPageBuilderWatermarkConfidential(CData $pageBuilder): int
    {
        $errorCode = $this->ffi->new('int');
        $result = $this->ffi->pdf_page_builder_watermark_confidential($pageBuilder, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_page_builder_watermark_confidential');
        return (int) $result;
    }

    /** "DRAFT" preset watermark. */
    public function pdfPageBuilderWatermarkDraft(CData $pageBuilder): int
    {
        $errorCode = $this->ffi->new('int');
        $result = $this->ffi->pdf_page_builder_watermark_draft($pageBuilder, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_page_builder_watermark_draft');
        return (int) $result;
    }

    /**
     * Set the global content-stream operator cap.
     *
     * `limit < 0` restores the default (1,000,000); any non-negative value
     * (including 0) is used as the explicit cap. There is no error channel —
     * the call returns the previous cap (or -1 if the default was active).
     *
     * @param int $limit New per-stream operator cap (negative = restore default)
     * @return int The previous cap (-1 if the default was active)
     */
    public function pdfOxideSetMaxOpsPerStream(int $limit): int
    {
        return (int) $this->ffi->pdf_oxide_set_max_ops_per_stream($limit);
    }

    /**
     * Toggle the global U+FFFD preservation flag for the high-level
     * extract_text / extract_words / extract_spans accessors.
     *
     * `1` = preserve FFFD chars; `0` = filter (v0.3.54 default). There is no
     * error channel — the call returns the previous value as `0` or `1`.
     *
     * @param int $preserve 1 to preserve unmapped glyphs, 0 to filter them
     * @return int The previous flag value (0 or 1)
     */
    public function pdfOxideSetPreserveUnmappedGlyphs(int $preserve): int
    {
        return (int) $this->ffi->pdf_oxide_set_preserve_unmapped_glyphs($preserve);
    }

    /**
     * Render a page with the full RenderOptions surface plus OCG layer
     * filtering.
     *
     * `$excludedLayers` is a list of Optional Content Group `/Name`s to
     * suppress; pass an empty array to disable filtering (matching
     * {@see pdfRenderPage} behaviour).
     *
     * @param CData       $handle               The document handle
     * @param int         $pageIndex            Zero-based page index
     * @param int         $dpi                  Render resolution
     * @param int         $format               0=PNG, 1=JPEG
     * @param float       $bgR                  Background red   (0.0..=1.0)
     * @param float       $bgG                  Background green (0.0..=1.0)
     * @param float       $bgB                  Background blue  (0.0..=1.0)
     * @param float       $bgA                  Background alpha (0.0..=1.0)
     * @param int         $transparentBackground 1 to drop the fill entirely
     * @param int         $renderAnnotations    1 to render annotations
     * @param int         $jpegQuality          JPEG quality (when format=1)
     * @param list<string> $excludedLayers      OCG `/Name`s to suppress
     * @return CData Image handle
     */
    public function pdfRenderPageWithOptionsEx(
        CData $handle,
        int $pageIndex,
        int $dpi,
        int $format,
        float $bgR,
        float $bgG,
        float $bgB,
        float $bgA,
        int $transparentBackground,
        int $renderAnnotations,
        int $jpegQuality,
        array $excludedLayers = []
    ): CData {
        $errorCode = $this->ffi->new('int32_t');

        $count = count($excludedLayers);
        $layersPtr = null;
        $cStrings = [];
        if ($count > 0) {
            // Build a `const char *const *` — an array of NUL-terminated
            // C strings. Retain the per-string CData in $cStrings so they
            // outlive the native call.
            $arr = $this->ffi->new("char*[{$count}]");
            $i = 0;
            foreach ($excludedLayers as $name) {
                $cStr = StringMarshaller::toCString((string) $name);
                $cStrings[] = $cStr;
                $arr[$i] = $this->ffi->cast('char*', FFI::addr($cStr));
                $i++;
            }
            $layersPtr = $this->ffi->cast('char**', FFI::addr($arr));
        }

        try {
            $imageHandle = $this->ffi->pdf_render_page_with_options_ex(
                $handle,
                $pageIndex,
                $dpi,
                $format,
                $bgR,
                $bgG,
                $bgB,
                $bgA,
                $transparentBackground,
                $renderAnnotations,
                $jpegQuality,
                $layersPtr,
                $count,
                FFI::addr($errorCode)
            );
            ErrorHandler::check((int) $errorCode->cdata, 'pdf_render_page_with_options_ex');
            return $imageHandle;
        } finally {
            unset($cStrings);
        }
    }
}
