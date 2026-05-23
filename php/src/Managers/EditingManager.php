<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI;
use FFI\CData;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\ErrorHandler;

/**
 * Manages PDF editing operations: redaction, flattening, and compliance.
 *
 * Provides methods for:
 * - Adding and applying content redactions
 * - Scrubbing sensitive metadata
 * - Flattening form fields and annotations into static content
 * - PDF/A compliance conversion and validation
 *
 * Example:
 *     $manager = new EditingManager($documentHandle);
 *     $manager->addRedaction(0, 100.0, 200.0, 300.0, 50.0);
 *     $manager->applyRedactions();
 *     $manager->flattenForms();
 */
class EditingManager
{
    private readonly CData $handle;
    private readonly FFI $ffi;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->ffi = NativeLibrary::getInstance();
    }

    // ==================== REDACTION ====================

    /**
     * Add a redaction annotation to a page.
     *
     * The redaction marks an area to be permanently removed when
     * applyRedactions() is called. Until applied, the content is
     * still visible but marked for redaction.
     *
     * @param int $page Zero-based page index
     * @param float $x X coordinate of the redaction rectangle
     * @param float $y Y coordinate of the redaction rectangle
     * @param float $width Width of the redaction rectangle
     * @param float $height Height of the redaction rectangle
     * @param array $color Fill color as [r, g, b] with values 0-255, default [0, 0, 0]
     * @throws \PdfOxide\Exceptions\RedactionException If adding the redaction fails
     */
    public function addRedaction(
        int $page,
        float $x,
        float $y,
        float $width,
        float $height,
        array $color = [0, 0, 0]
    ): void {
        $r = (int)($color[0] ?? 0);
        $g = (int)($color[1] ?? 0);
        $b = (int)($color[2] ?? 0);

        $errorCode = FFI::new('int');
        $this->ffi->pdf_redaction_add(
            $this->handle,
            $page,
            $x,
            $y,
            $width,
            $height,
            $r,
            $g,
            $b,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_add', [
            'page' => $page,
        ]);
    }

    /**
     * Apply all pending redactions, permanently removing marked content.
     *
     * This operation is irreversible. Once applied, the redacted content
     * is permanently removed from the document.
     *
     * @param bool $scrubMetadata If true, also scrub document metadata
     * @param array $fillColor Fill color for redacted areas as [r, g, b], default [0, 0, 0]
     * @throws \PdfOxide\Exceptions\RedactionException If applying redactions fails
     */
    public function applyRedactions(bool $scrubMetadata = false, array $fillColor = [0, 0, 0]): void
    {
        $r = (int)($fillColor[0] ?? 0);
        $g = (int)($fillColor[1] ?? 0);
        $b = (int)($fillColor[2] ?? 0);

        $errorCode = FFI::new('int');
        $this->ffi->pdf_redaction_apply(
            $this->handle,
            $scrubMetadata,
            $r,
            $g,
            $b,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_apply');
    }

    /**
     * Scrub sensitive metadata from the document.
     *
     * Removes various types of metadata that may contain sensitive
     * information such as author names, creation software, or
     * embedded scripts.
     *
     * @param bool $removeInfo Remove document Info dictionary (Title, Author, etc.)
     * @param bool $removeXmp Remove XMP metadata streams
     * @param bool $removeJs Remove embedded JavaScript
     * @throws \PdfOxide\Exceptions\RedactionException If metadata scrubbing fails
     */
    public function scrubMetadata(
        bool $removeInfo = true,
        bool $removeXmp = true,
        bool $removeJs = true
    ): void {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_redaction_scrub_metadata(
            $this->handle,
            $removeInfo,
            $removeXmp,
            $removeJs,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_scrub_metadata');
    }

    /**
     * Get the number of pending redaction annotations.
     *
     * @return int Number of pending redactions not yet applied
     * @throws \PdfOxide\Exceptions\RedactionException If retrieving the count fails
     */
    public function getRedactionCount(): int
    {
        $errorCode = FFI::new('int');
        $count = $this->ffi->pdf_redaction_count(
            $this->handle,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_redaction_count');
        return (int)$count;
    }

    // ==================== FLATTENING ====================

    /**
     * Flatten all form fields in the document.
     *
     * Converts interactive form fields into static content. After
     * flattening, form fields are no longer editable but their
     * values are preserved as visible content.
     *
     * @throws \PdfOxide\Exceptions\PdfException If flattening fails
     */
    public function flattenForms(): void
    {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_document_editor_flatten_forms(
            $this->handle,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_flatten_forms');
    }

    /**
     * Flatten form fields on a specific page.
     *
     * @param int $page Zero-based page index
     * @throws \PdfOxide\Exceptions\PdfException If flattening fails
     */
    public function flattenFormsPage(int $page): void
    {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_document_editor_flatten_forms_page(
            $this->handle,
            $page,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_flatten_forms_page', [
            'page' => $page,
        ]);
    }

    /**
     * Flatten all annotations in the document.
     *
     * Converts interactive annotations (highlights, stamps, notes, etc.)
     * into static page content. After flattening, annotations are no
     * longer interactive.
     *
     * @throws \PdfOxide\Exceptions\PdfException If flattening fails
     */
    public function flattenAnnotations(): void
    {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_document_editor_flatten_annotations(
            $this->handle,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_flatten_annotations');
    }

    /**
     * Flatten annotations on a specific page.
     *
     * @param int $page Zero-based page index
     * @throws \PdfOxide\Exceptions\PdfException If flattening fails
     */
    public function flattenAnnotationsPage(int $page): void
    {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_document_editor_flatten_annotations_page(
            $this->handle,
            $page,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_editor_flatten_annotations_page', [
            'page' => $page,
        ]);
    }

    // ==================== COMPLIANCE ====================

    /**
     * Convert the document to PDF/A format.
     *
     * @param int $level PDF/A conformance level (0=1B, 1=1A, 2=2B, 3=2A, etc.)
     * @throws \PdfOxide\Exceptions\ComplianceException If conversion fails
     */
    public function convertToPdfA(int $level = 2): void
    {
        $errorCode = FFI::new('int');
        $this->ffi->pdf_convert_to_pdf_a(
            $this->handle,
            $level,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_convert_to_pdf_a', [
            'level' => $level,
        ]);
    }

    /**
     * Validate the document against a PDF/A conformance level.
     *
     * @param int $level PDF/A conformance level to validate against
     * @return int Validation result code (0 = compliant)
     * @throws \PdfOxide\Exceptions\ComplianceException If validation fails
     */
    public function validatePdfA(int $level = 2): int
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_validate_pdfa(
            $this->handle,
            $level,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_validate_pdfa', [
            'level' => $level,
        ]);
        return (int)$result;
    }

    // ==================== SUMMARY ====================

    /**
     * Get editing capabilities summary.
     *
     * @return array Summary of editing capabilities
     */
    public function getSummary(): array
    {
        return [
            'redaction_count' => $this->getRedactionCount(),
            'capabilities' => [
                'redaction' => true,
                'flatten_forms' => true,
                'flatten_annotations' => true,
                'scrub_metadata' => true,
                'pdf_a_conversion' => true,
                'pdf_a_validation' => true,
            ],
        ];
    }
}
