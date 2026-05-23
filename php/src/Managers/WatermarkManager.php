<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI;
use FFI\CData;
use PdfOxide\FFI\ErrorHandler;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\StringMarshaller;

/**
 * v0.3.55 — watermark + stamp + freetext annotations on a page builder.
 *
 * Wraps the page-builder C-ABI watermark surface
 * (`pdf_page_builder_watermark*`, `pdf_page_builder_stamp`,
 * `pdf_page_builder_freetext`). These are builder ops — they apply to
 * the currently-being-constructed page rather than to an already-open
 * document, matching the same design in C# / Java / Python.
 *
 * Get one from {@see \PdfOxide\Builders\PageBuilder::watermarks()} (when
 * the creation API lands in v0.3.56). Until then this class is
 * directly constructible from a raw page-builder CData handle for
 * power-user direct-FFI builder flows and integration tests.
 */
final class WatermarkManager
{
    private readonly FunctionBindings $bindings;
    private readonly CData $pageBuilderHandle;
    private readonly FFI $ffi;

    public function __construct(CData $pageBuilderHandle)
    {
        $this->pageBuilderHandle = $pageBuilderHandle;
        $this->bindings = new FunctionBindings();
        $this->ffi = NativeLibrary::getInstance();
    }

    /**
     * Add a custom text watermark to the current page builder.
     *
     * Returns the number of glyphs / characters laid out (per the
     * Rust contract); 0 is a successful no-op (empty text).
     */
    public function addText(string $text): int
    {
        return $this->bindings->pdfPageBuilderWatermark($this->pageBuilderHandle, $text);
    }

    /** Add the preset "CONFIDENTIAL" diagonal watermark. */
    public function addConfidential(): int
    {
        return $this->bindings->pdfPageBuilderWatermarkConfidential($this->pageBuilderHandle);
    }

    /** Add the preset "DRAFT" diagonal watermark. */
    public function addDraft(): int
    {
        return $this->bindings->pdfPageBuilderWatermarkDraft($this->pageBuilderHandle);
    }

    /**
     * Add a "rubber stamp" annotation by type name
     * (e.g. "Approved", "NotApproved", "Confidential"; per the PDF spec).
     */
    public function addStamp(string $typeName): int
    {
        $errorCode = FFI::new('int');
        $cType = StringMarshaller::toCString($typeName);
        try {
            $result = $this->ffi->pdf_page_builder_stamp($this->pageBuilderHandle, $cType, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_page_builder_stamp', ['type' => $typeName]);
            return (int)$result;
        } finally {
            unset($cType);
        }
    }

    /**
     * Add a free-text annotation in a rectangle.
     */
    public function addFreetext(float $x, float $y, float $width, float $height, string $text): int
    {
        $errorCode = FFI::new('int');
        $cText = StringMarshaller::toCString($text);
        try {
            $result = $this->ffi->pdf_page_builder_freetext(
                $this->pageBuilderHandle,
                $x,
                $y,
                $width,
                $height,
                $cText,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_page_builder_freetext');
            return (int)$result;
        } finally {
            unset($cText);
        }
    }
}
