<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\PdfDocument;

/**
 * v0.3.48 #159 — Office <-> PDF conversion.
 *
 * Two surfaces:
 *  - Static factories that take Office bytes and return a
 *    {@see PdfDocument} (DOCX/PPTX/XLSX -> PDF).
 *  - Instance methods that take an open document and emit Office bytes
 *    (PDF -> DOCX/PPTX/XLSX).
 *
 * Static factories are also mirrored on {@see PdfDocument::fromDocxBytes()}
 * etc. for ergonomic parity with Python's
 * `pdf_oxide.PdfDocument.from_docx_bytes`.
 */
final class OfficeConverter
{
    private readonly FunctionBindings $bindings;
    private readonly CData $documentHandle;

    public function __construct(CData $documentHandle)
    {
        $this->documentHandle = $documentHandle;
        $this->bindings = new FunctionBindings();
    }

    /** Static: import a DOCX document and return the corresponding PDF. */
    public static function importDocx(string $data): PdfDocument
    {
        return PdfDocument::fromDocxBytes($data);
    }

    /** Static: import a PPTX document. */
    public static function importPptx(string $data): PdfDocument
    {
        return PdfDocument::fromPptxBytes($data);
    }

    /** Static: import an XLSX document. */
    public static function importXlsx(string $data): PdfDocument
    {
        return PdfDocument::fromXlsxBytes($data);
    }

    /**
     * Export the open document as DOCX bytes.
     *
     * Caller is responsible for writing the bytes to disk (e.g.
     * `file_put_contents`); we deliberately don't shell-out to libreoffice
     * or perform any IO here.
     */
    public function toDocx(): string
    {
        return $this->bindings->pdfDocumentToDocxBytes($this->documentHandle);
    }

    /** Export the open document as PPTX bytes. */
    public function toPptx(): string
    {
        return $this->bindings->pdfDocumentToPptxBytes($this->documentHandle);
    }

    /** Export the open document as XLSX bytes. */
    public function toXlsx(): string
    {
        return $this->bindings->pdfDocumentToXlsxBytes($this->documentHandle);
    }
}
