<?php

declare(strict_types=1);

namespace PdfOxide\Xfa;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * XFA (XML Forms Architecture) form representation.
 *
 * Provides access to form fields defined in XFA format within PDF documents.
 * XFA is an advanced form technology more powerful than traditional AcroForms.
 *
 * Example:
 *     $doc = new PdfDocument('form.pdf');
 *     if ($doc->hasXfa()) {
 *         $form = $doc->getXfaForm();
 *         echo "Fields: " . $form->getFieldCount();
 *         foreach ($form->getFields() as $field) {
 *             echo $field->getName() . " = " . $field->getValue();
 *         }
 *     }
 *
 * @since 0.4.0
 */
class XfaForm
{
    private CData $handle;
    private FunctionBindings $bindings;
    private ?array $cachedFields = null;

    /**
     * Create XfaForm from FFI handle.
     *
     * @param CData $handle FFI form handle
     * @param FunctionBindings $bindings Function bindings
     */
    public function __construct(CData $handle, FunctionBindings $bindings)
    {
        $this->handle = $handle;
        $this->bindings = $bindings;
    }

    /**
     * Get total number of form fields.
     *
     * @return int Field count
     */
    public function getFieldCount(): int
    {
        return (int)$this->bindings->pdfXfaFormFieldCount($this->handle);
    }

    /**
     * Get form field by index.
     *
     * @param int $index Field index (0-based)
     * @return XfaField The field
     * @throws \OutOfRangeException If index is invalid
     *
     * Example:
     *     $field = $form->getField(0);
     */
    public function getField(int $index): XfaField
    {
        if ($index < 0 || $index >= $this->getFieldCount()) {
            throw new \OutOfRangeException("Field index out of bounds: {$index}");
        }

        $fieldHandle = $this->bindings->pdfXfaFormGetField($this->handle, $index);
        return new XfaField($fieldHandle, $this->bindings);
    }

    /**
     * Get form field by name.
     *
     * @param string $name Field name
     * @return ?XfaField The field, or null if not found
     *
     * Example:
     *     $field = $form->getFieldByName('firstName');
     */
    public function getFieldByName(string $name): ?XfaField
    {
        $count = $this->getFieldCount();
        for ($i = 0; $i < $count; $i++) {
            $field = $this->getField($i);
            if ($field->getName() === $name) {
                return $field;
            }
        }
        return null;
    }

    /**
     * Get all form fields.
     *
     * @return XfaField[] Array of all fields
     *
     * Example:
     *     foreach ($form->getFields() as $field) {
     *         echo $field->getName();
     *     }
     */
    public function getFields(): array
    {
        if ($this->cachedFields !== null) {
            return $this->cachedFields;
        }

        $fields = [];
        $count = $this->getFieldCount();

        for ($i = 0; $i < $count; $i++) {
            $fields[] = $this->getField($i);
        }

        $this->cachedFields = $fields;
        return $fields;
    }

    /**
     * Get all field names.
     *
     * @return string[] Array of field names
     */
    public function getFieldNames(): array
    {
        return array_map(fn(XfaField $field) => $field->getName(), $this->getFields());
    }

    /**
     * Get all field values as associative array.
     *
     * @return array Field names mapped to values
     *
     * Example:
     *     $values = $form->getFieldValues();
     *     $firstName = $values['firstName'] ?? null;
     */
    public function getFieldValues(): array
    {
        $values = [];
        foreach ($this->getFields() as $field) {
            $values[$field->getName()] = $field->getValue();
        }
        return $values;
    }

    /**
     * Set a field value.
     *
     * @param string $name Field name
     * @param mixed $value New value
     * @throws \InvalidArgumentException If field not found
     *
     * Example:
     *     $form->setFieldValue('email', 'user@example.com');
     */
    public function setFieldValue(string $name, mixed $value): void
    {
        $field = $this->getFieldByName($name);
        if ($field === null) {
            throw new \InvalidArgumentException("Field not found: {$name}");
        }

        $field->setValue((string)$value);
    }

    /**
     * Get form as array.
     *
     * @return array Form data as array
     */
    public function toArray(): array
    {
        return [
            'fieldCount' => $this->getFieldCount(),
            'fieldNames' => $this->getFieldNames(),
            'fieldValues' => $this->getFieldValues(),
            'fields' => array_map(fn(XfaField $f) => $f->toArray(), $this->getFields()),
        ];
    }

    /**
     * Free form resources.
     */
    public function __destruct()
    {
        if ($this->handle !== null) {
            try {
                $this->bindings->pdfXfaFormFree($this->handle);
            } catch (\Exception) {
                // Ignore errors during cleanup
            }
        }
    }
}
