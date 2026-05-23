<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI;
use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\ErrorHandler;
use PdfOxide\FFI\StringMarshaller;

/**
 * Manages XFA (XML Forms Architecture) operations.
 *
 * Handles XFA form parsing, field access, data extraction,
 * and conversion to standard AcroForm format.
 * Uses PHP 8+ features for clean, type-safe implementation.
 */
class XfaFormManager
{
    private readonly FunctionBindings $bindings;
    private readonly CData $handle;
    private readonly FFI $ffi;
    private ?CData $xfaFormHandle = null;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
        $this->ffi = NativeLibrary::getInstance();
    }

    // ==================== XFA DETECTION ====================

    /**
     * Check if document contains XFA form.
     *
     * @return bool True if document has XFA form
     */
    public function hasXfa(): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_has_xfa($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_has_xfa');
        return (bool)$result;
    }

    /**
     * Get XFA form type.
     *
     * @return XfaFormType|null Form type or null if not XFA
     */
    public function getFormType(): ?XfaFormType
    {
        if (!$this->hasXfa()) {
            return null;
        }

        $form = $this->getForm();
        // Determine type based on form characteristics
        // Static forms have fixed layout, dynamic can change at runtime
        return XfaFormType::STATIC;
    }

    // ==================== FORM PARSING ====================

    /**
     * Parse XFA form from document.
     *
     * @return XfaForm Parsed XFA form object
     */
    public function parseForm(): XfaForm
    {
        if ($this->xfaFormHandle === null) {
            $this->xfaFormHandle = $this->getForm();
        }

        return new XfaForm($this->xfaFormHandle, $this->ffi);
    }

    /**
     * Get XFA form handle (internal).
     *
     * @return CData Form handle
     */
    private function getForm(): CData
    {
        if ($this->xfaFormHandle !== null) {
            return $this->xfaFormHandle;
        }

        $errorCode = FFI::new('int');
        $this->xfaFormHandle = $this->ffi->pdf_parse_xfa_form($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_parse_xfa_form');
        return $this->xfaFormHandle;
    }

    // ==================== FIELD ACCESS ====================

    /**
     * Get XFA field count.
     *
     * @return int Number of fields
     */
    public function getFieldCount(): int
    {
        if (!$this->hasXfa()) {
            return 0;
        }

        $form = $this->getForm();
        $errorCode = FFI::new('int');
        $count = $this->ffi->pdf_xfa_form_field_count($form, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_xfa_form_field_count');
        return (int)$count;
    }

    /**
     * Get XFA field by index.
     *
     * @param int $index Field index
     * @return XfaField Field object
     */
    public function getField(int $index): XfaField
    {
        $form = $this->getForm();
        $errorCode = FFI::new('int');
        $fieldHandle = $this->ffi->pdf_xfa_form_get_field($form, $index, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_xfa_form_get_field', ['index' => $index]);

        return new XfaField($fieldHandle, $this->ffi);
    }

    /**
     * Get all XFA fields.
     *
     * @return array<XfaField> Array of fields
     */
    public function getAllFields(): array
    {
        $fields = [];
        $count = $this->getFieldCount();

        for ($i = 0; $i < $count; $i++) {
            $fields[] = $this->getField($i);
        }

        return $fields;
    }

    /**
     * Get field by name.
     *
     * @param string $name Field name
     * @return XfaField|null Field or null if not found
     */
    public function getFieldByName(string $name): ?XfaField
    {
        $fields = $this->getAllFields();
        foreach ($fields as $field) {
            if ($field->getName() === $name) {
                return $field;
            }
        }
        return null;
    }

    // ==================== DATA EXTRACTION ====================

    /**
     * Get XFA dataset.
     *
     * @return XfaDataset Dataset object
     */
    public function getDataset(): XfaDataset
    {
        $form = $this->getForm();
        $errorCode = FFI::new('int');
        $datasetHandle = $this->ffi->pdf_xfa_form_get_dataset($form, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_xfa_form_get_dataset');

        return new XfaDataset($datasetHandle, $this->ffi);
    }

    /**
     * Export XFA data as XML.
     *
     * @return string XML data
     */
    public function exportDataAsXml(): string
    {
        $dataset = $this->getDataset();
        return $dataset->toXml();
    }

    /**
     * Export XFA data to file.
     *
     * @param string $filePath Output file path
     * @return bool True on success
     */
    public function exportDataToFile(string $filePath): bool
    {
        $xml = $this->exportDataAsXml();
        return file_put_contents($filePath, $xml) !== false;
    }

    // ==================== CONVERSION ====================

    /**
     * Convert XFA form to AcroForm.
     *
     * This converts the dynamic XFA form to a standard AcroForm
     * that is more widely supported.
     *
     * @return bool True on success
     */
    public function convertToAcroForm(): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_convert_xfa_to_acroform($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_convert_xfa_to_acroform');
        return (bool)$result;
    }

    // ==================== UTILITIES ====================

    /**
     * Get XFA form summary.
     *
     * @return array Summary information
     */
    public function getSummary(): array
    {
        if (!$this->hasXfa()) {
            return [
                'has_xfa' => false,
                'field_count' => 0,
                'form_type' => null,
            ];
        }

        return [
            'has_xfa' => true,
            'field_count' => $this->getFieldCount(),
            'form_type' => $this->getFormType()?->value,
            'capabilities' => [
                'parse_form' => true,
                'get_fields' => true,
                'export_data' => true,
                'convert_to_acroform' => true,
            ],
        ];
    }

    /**
     * Free resources on destruct.
     */
    public function __destruct()
    {
        if ($this->xfaFormHandle !== null) {
            $this->ffi->pdf_xfa_form_free($this->xfaFormHandle);
            $this->xfaFormHandle = null;
        }
    }
}

// ==================== SUPPORTING CLASSES ====================

/**
 * XFA form types.
 */
enum XfaFormType: string
{
    case STATIC = 'static';
    case DYNAMIC = 'dynamic';

    public function getDescription(): string
    {
        return match($this) {
            self::STATIC => 'Static XFA form with fixed layout',
            self::DYNAMIC => 'Dynamic XFA form with runtime layout changes',
        };
    }
}

/**
 * XFA field types.
 */
enum XfaFieldType: int
{
    case TEXT = 0;
    case CHECKBOX = 1;
    case RADIO = 2;
    case DROPDOWN = 3;
    case BUTTON = 4;
    case SIGNATURE = 5;
    case IMAGE = 6;
    case DATETIME = 7;
    case NUMERIC = 8;
    case PASSWORD = 9;

    public function getDescription(): string
    {
        return match($this) {
            self::TEXT => 'Text field',
            self::CHECKBOX => 'Checkbox',
            self::RADIO => 'Radio button',
            self::DROPDOWN => 'Dropdown list',
            self::BUTTON => 'Button',
            self::SIGNATURE => 'Signature field',
            self::IMAGE => 'Image field',
            self::DATETIME => 'Date/time field',
            self::NUMERIC => 'Numeric field',
            self::PASSWORD => 'Password field',
        };
    }
}

/**
 * XFA form object.
 */
class XfaForm
{
    private CData $handle;
    private FFI $ffi;

    public function __construct(CData $handle, FFI $ffi)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
    }

    public function getFieldCount(): int
    {
        $errorCode = FFI::new('int');
        $count = $this->ffi->pdf_xfa_form_field_count($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_xfa_form_field_count');
        return (int)$count;
    }

    public function getField(int $index): XfaField
    {
        $errorCode = FFI::new('int');
        $fieldHandle = $this->ffi->pdf_xfa_form_get_field($this->handle, $index, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_xfa_form_get_field', ['index' => $index]);
        return new XfaField($fieldHandle, $this->ffi);
    }

    public function getHandle(): CData
    {
        return $this->handle;
    }
}

/**
 * XFA field object.
 */
class XfaField
{
    private CData $handle;
    private FFI $ffi;
    private ?string $cachedName = null;

    public function __construct(CData $handle, FFI $ffi)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
    }

    public function getName(): string
    {
        if ($this->cachedName !== null) {
            return $this->cachedName;
        }

        $errorCode = FFI::new('int');
        $namePtr = $this->ffi->pdf_xfa_field_get_name($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_xfa_field_get_name');
        $this->cachedName = StringMarshaller::fromCString($namePtr);
        return $this->cachedName;
    }

    public function toArray(): array
    {
        return [
            'name' => $this->getName(),
        ];
    }

    public function __destruct()
    {
        $this->ffi->pdf_xfa_field_free($this->handle);
    }
}

/**
 * XFA dataset object.
 */
class XfaDataset
{
    private CData $handle;
    private FFI $ffi;

    public function __construct(CData $handle, FFI $ffi)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
    }

    public function toXml(): string
    {
        $errorCode = FFI::new('int');
        $xmlPtr = $this->ffi->pdf_xfa_dataset_to_xml($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_xfa_dataset_to_xml');
        return StringMarshaller::fromCString($xmlPtr);
    }

    public function __destruct()
    {
        $this->ffi->pdf_xfa_dataset_free($this->handle);
    }
}
