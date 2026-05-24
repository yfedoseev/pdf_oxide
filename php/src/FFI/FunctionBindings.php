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
        $errorCode = FFI::new('int');
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


    // ==================== SECURITY & PERMISSIONS ====================

    /**
     * Check if document is encrypted.
     */
    public function pdfDocumentIsEncrypted(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_is_encrypted($handle);
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


    // ========== XFA Form Functions ==========

    /**
     * Check if document has XFA form.
     */
    public function pdfDocumentHasXfa(CData $handle): bool
    {
        return (bool)$this->ffi->pdf_document_has_xfa($handle);
    }


    // ========== Advanced Signature Functions (unique additions) ==========


    /**
     * Get certificate serial number.
     */
    public function pdfCertificateGetSerial(CData $cert): string
    {
        $serial = $this->ffi->pdf_certificate_get_serial($cert);
        return StringMarshaller::fromCString($serial, false);
    }


    /**
     * Get signature signing time.
     */
    public function pdfSignatureGetSigningTime(CData $sig): string
    {
        $time = $this->ffi->pdf_signature_get_signing_time($sig);
        return StringMarshaller::fromCString($time, false);
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
        $errorCode = FFI::new('int');
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
