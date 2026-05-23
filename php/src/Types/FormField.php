<?php

declare(strict_types=1);

namespace PdfOxide\Types;

use PdfOxide\Enums\FormFieldType;

/**
 * Represents a form field in a PDF.
 */
readonly class FormField
{
    public function __construct(
        public string $name,
        public FormFieldType $type,
        public ?string $value = null,
        public bool $required = false,
        public bool $readOnly = false,
        public ?Rect $boundingBox = null,
        public int $pageIndex = 0,
        public array $options = [] // For dropdown/listbox
    ) {}

    /**
     * Check if field is a text field.
     */
    public function isTextField(): bool
    {
        return $this->type === FormFieldType::TEXT;
    }

    /**
     * Check if field is a checkbox or radio button.
     */
    public function isBoolean(): bool
    {
        return in_array($this->type, [FormFieldType::CHECKBOX, FormFieldType::RADIO]);
    }

    /**
     * Check if field has options (dropdown, listbox, etc.).
     */
    public function hasOptions(): bool
    {
        return in_array($this->type, [FormFieldType::DROPDOWN, FormFieldType::LISTBOX, FormFieldType::COMBO_BOX]);
    }

    /**
     * Convert to array format.
     */
    public function toArray(): array
    {
        return [
            'name' => $this->name,
            'type' => $this->type->value,
            'value' => $this->value,
            'required' => $this->required,
            'read_only' => $this->readOnly,
            'bounding_box' => $this->boundingBox?->toArray(),
            'page_index' => $this->pageIndex,
            'options' => $this->options,
        ];
    }
}
