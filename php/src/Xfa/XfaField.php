<?php

declare(strict_types=1);

namespace PdfOxide\Xfa;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * A single XFA form field.
 *
 * Represents one field in an XFA form with type, value, and metadata.
 *
 * Example:
 *     $field = $form->getField(0);
 *     echo "Type: " . $field->getType();
 *     echo "Value: " . $field->getValue();
 *     $field->setValue('new value');
 *
 * @since 0.4.0
 */
class XfaField
{
    private CData $handle;
    private FunctionBindings $bindings;
    private ?string $cachedName = null;
    private ?string $cachedType = null;
    private ?string $cachedValue = null;

    /**
     * Create XfaField from FFI handle.
     *
     * @param CData $handle FFI field handle
     * @param FunctionBindings $bindings Function bindings
     */
    public function __construct(CData $handle, FunctionBindings $bindings)
    {
        $this->handle = $handle;
        $this->bindings = $bindings;
    }

    /**
     * Get field name.
     *
     * @return string Field name
     *
     * Example:
     *     echo $field->getName();  // e.g., "firstName"
     */
    public function getName(): string
    {
        if ($this->cachedName === null) {
            $this->cachedName = $this->bindings->pdfXfaFieldGetName($this->handle);
        }
        return $this->cachedName;
    }

    /**
     * Get field type.
     *
     * @return string Field type (e.g., 'text', 'checkbox', 'radio', 'dropdown')
     *
     * Example:
     *     if ($field->getType() === 'checkbox') {
     *         // Handle checkbox field
     *     }
     */
    public function getType(): string
    {
        if ($this->cachedType === null) {
            $this->cachedType = $this->bindings->pdfXfaFieldGetType($this->handle);
        }
        return $this->cachedType;
    }

    /**
     * Get field value.
     *
     * @return string Field value as string
     *
     * Example:
     *     $value = $field->getValue();
     */
    public function getValue(): string
    {
        if ($this->cachedValue === null) {
            $this->cachedValue = $this->bindings->pdfXfaFieldGetValue($this->handle);
        }
        return $this->cachedValue;
    }

    /**
     * Set field value.
     *
     * @param string $value New value
     *
     * Example:
     *     $field->setValue('John');
     */
    public function setValue(string $value): void
    {
        $this->bindings->pdfXfaFieldSetValue($this->handle, $value);
        $this->cachedValue = $value;
    }

    /**
     * Check if field is a text field.
     *
     * @return bool True if type is text-based
     */
    public function isTextField(): bool
    {
        return in_array($this->getType(), ['text', 'password']);
    }

    /**
     * Check if field is a checkbox.
     *
     * @return bool True if type is checkbox
     */
    public function isCheckbox(): bool
    {
        return $this->getType() === 'checkbox';
    }

    /**
     * Check if field is a radio button.
     *
     * @return bool True if type is radio
     */
    public function isRadio(): bool
    {
        return $this->getType() === 'radio';
    }

    /**
     * Check if field is a dropdown/select.
     *
     * @return bool True if type is dropdown or listbox
     */
    public function isDropdown(): bool
    {
        return in_array($this->getType(), ['dropdown', 'listbox']);
    }

    /**
     * Get field as array.
     *
     * @return array Field data
     */
    public function toArray(): array
    {
        return [
            'name' => $this->getName(),
            'type' => $this->getType(),
            'value' => $this->getValue(),
        ];
    }

    /**
     * String representation of field.
     *
     * @return string
     */
    public function __toString(): string
    {
        return "{$this->getName()} ({$this->getType()}): {$this->getValue()}";
    }

    /**
     * Free field resources.
     */
    public function __destruct()
    {
        if ($this->handle !== null) {
            try {
                $this->bindings->pdfXfaFieldFree($this->handle);
            } catch (\Exception) {
                // Ignore errors during cleanup
            }
        }
    }
}
