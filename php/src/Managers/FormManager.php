<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\FormField;
use PdfOxide\Enums\FormFieldType;

/**
 * Manages PDF form field operations.
 *
 * Handles reading, filling, and modifying form fields in PDF documents.
 * Supports both AcroForm and XFA forms.
 */
class FormManager
{
    private FunctionBindings $bindings;
    private CData $handle;
    private ?array $cachedFields = null;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Get all form fields from the document.
     *
     * @return FormField[] Array of form fields
     */
    public function getFields(): array
    {
        if ($this->cachedFields !== null) {
            return $this->cachedFields;
        }

        $this->cachedFields = [];

        if (!$this->bindings->pdfDocumentHasFormFields($this->handle)) {
            return $this->cachedFields;
        }

        $count = $this->bindings->pdfDocumentGetFormFieldCount($this->handle);

        for ($i = 0; $i < $count; $i++) {
            $fieldHandle = $this->bindings->pdfDocumentGetFormField($this->handle, $i);

            try {
                $name = $this->bindings->pdfFormFieldGetName($fieldHandle);
                $type = $this->bindings->pdfFormFieldGetType($fieldHandle);
                $value = $this->bindings->pdfFormFieldGetValue($fieldHandle);
                $required = $this->bindings->pdfFormFieldIsRequired($fieldHandle);
                $pageIndex = $this->bindings->pdfFormFieldGetPageIndex($fieldHandle);

                // Get options for dropdown/checkbox fields
                $options = [];
                $optionsCount = $this->bindings->pdfFormFieldGetOptionsCount($fieldHandle);
                for ($j = 0; $j < $optionsCount; $j++) {
                    $options[] = $this->bindings->pdfFormFieldGetOption($fieldHandle, $j);
                }

                // Create FormField object
                $field = new FormField();
                $field->name = $name;
                $field->type = FormFieldType::from($type);
                $field->value = $value ?: null;
                $field->required = $required;
                $field->pageIndex = $pageIndex;
                $field->options = $options;

                $this->cachedFields[] = $field;
            } finally {
                $this->bindings->pdfFormFieldFree($fieldHandle);
            }
        }

        return $this->cachedFields;
    }

    /**
     * Get form fields by page.
     *
     * @param int $pageIndex Zero-based page index
     * @return FormField[] Form fields on this page
     */
    public function getFieldsByPage(int $pageIndex): array
    {
        $allFields = $this->getFields();
        return array_filter($allFields, fn($f) => $f->pageIndex === $pageIndex);
    }

    /**
     * Get a form field by name.
     *
     * @param string $name Field name
     * @return FormField|null The field or null if not found
     */
    public function getField(string $name): ?FormField
    {
        $fields = $this->getFields();
        foreach ($fields as $field) {
            if ($field->name === $name) {
                return $field;
            }
        }
        return null;
    }

    /**
     * Get form field value.
     *
     * @param string $fieldName Field name
     * @return string|array|null Field value
     */
    public function getFieldValue(string $fieldName): string|array|null
    {
        $field = $this->getField($fieldName);
        return $field?->value;
    }

    /**
     * Set form field value.
     *
     * @param string $fieldName Field name
     * @param string|array $value New value
     * @return void
     * @throws \PdfOxide\Exceptions\ValidationException if field doesn't exist
     */
    public function setFieldValue(string $fieldName, string|array $value): void
    {
        $fieldHandle = $this->bindings->pdfDocumentFindFormField($this->handle, $fieldName);
        if ($fieldHandle === null) {
            throw new \PdfOxide\Exceptions\ValidationException(
                "Form field not found: {$fieldName}",
                ['field_name' => $fieldName]
            );
        }

        try {
            // Convert value to string for FFI
            $strValue = is_array($value) ? implode(',', $value) : (string)$value;
            $this->bindings->pdfFormFieldSetValue($fieldHandle, $strValue);
            $this->clearCache();
        } finally {
            $this->bindings->pdfFormFieldFree($fieldHandle);
        }
    }

    /**
     * Fill multiple form fields at once.
     *
     * @param array<string, string|array> $values Field names => values
     * @return void
     */
    public function fillFields(array $values): void
    {
        foreach ($values as $name => $value) {
            $this->setFieldValue($name, $value);
        }
    }

    /**
     * Get field count.
     *
     * @return int Total number of form fields
     */
    public function count(): int
    {
        return count($this->getFields());
    }

    /**
     * Check if document has form fields.
     *
     * @return bool True if form exists
     */
    public function hasFields(): bool
    {
        return $this->count() > 0;
    }

    /**
     * Get fields by type.
     *
     * @param FormFieldType $type Field type to filter by
     * @return FormField[] Fields of specified type
     */
    public function getFieldsByType(FormFieldType $type): array
    {
        $fields = $this->getFields();
        return array_filter($fields, fn($f) => $f->type === $type);
    }

    /**
     * Get all text fields.
     *
     * @return FormField[] Text fields
     */
    public function getTextFields(): array
    {
        return $this->getFieldsByType(FormFieldType::TEXT);
    }

    /**
     * Get all checkbox fields.
     *
     * @return FormField[] Checkbox fields
     */
    public function getCheckboxFields(): array
    {
        return $this->getFieldsByType(FormFieldType::CHECKBOX);
    }

    /**
     * Get all dropdown/select fields.
     *
     * @return FormField[] Dropdown fields
     */
    public function getDropdownFields(): array
    {
        $dropdowns = $this->getFieldsByType(FormFieldType::DROPDOWN);
        $combos = $this->getFieldsByType(FormFieldType::COMBO_BOX);
        return array_merge($dropdowns, $combos);
    }

    /**
     * Get required form fields.
     *
     * @return FormField[] Required fields
     */
    public function getRequiredFields(): array
    {
        $fields = $this->getFields();
        return array_filter($fields, fn($f) => $f->required);
    }

    /**
     * Check if all required fields are filled.
     *
     * @return bool True if all required fields have values
     */
    public function areRequiredFieldsFilled(): bool
    {
        $required = $this->getRequiredFields();
        foreach ($required as $field) {
            if (empty($field->value)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Get unfilled required fields.
     *
     * @return FormField[] Unfilled required fields
     */
    public function getUnfilledRequiredFields(): array
    {
        $required = $this->getRequiredFields();
        return array_filter($required, fn($f) => empty($f->value));
    }

    /**
     * Check if document has XFA form.
     *
     * @return bool True if XFA form exists
     */
    public function hasXfaForm(): bool
    {
        return $this->bindings->pdfDocumentHasXfaForm($this->handle);
    }

    /**
     * Convert XFA form to AcroForm.
     *
     * This is useful for compatibility with PDF readers that don't support XFA.
     *
     * @return void
     * @throws \PdfOxide\Exceptions\ValidationException if no XFA form exists
     */
    public function convertXfaToAcroForm(): void
    {
        if (!$this->hasXfaForm()) {
            throw new \PdfOxide\Exceptions\ValidationException(
                'Document does not contain an XFA form',
                ['has_xfa' => false]
            );
        }

        $this->bindings->pdfDocumentConvertXfaToAcroForm($this->handle);
        $this->clearCache();
    }

    /**
     * Get form fields as array for serialization.
     *
     * @return array Array of field data
     */
    public function toArray(): array
    {
        return array_map(fn($f) => $f->toArray(), $this->getFields());
    }

    /**
     * Clear the field cache.
     *
     * @internal
     */
    public function clearCache(): void
    {
        $this->cachedFields = null;
    }

    /**
     * Get form summary.
     *
     * @return array Summary statistics
     */
    public function getSummary(): array
    {
        $fields = $this->getFields();

        $byType = [];
        foreach ($fields as $field) {
            $type = $field->type->value;
            $byType[$type] = ($byType[$type] ?? 0) + 1;
        }

        return [
            'total_fields' => count($fields),
            'required_count' => count($this->getRequiredFields()),
            'filled_count' => count(array_filter($fields, fn($f) => !empty($f->value))),
            'by_type' => $byType,
            'has_xfa' => $this->hasXfaForm(),
        ];
    }
}
