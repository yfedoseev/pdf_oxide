<?php

declare(strict_types=1);

namespace PdfOxide\Enums;

/**
 * PDF form field types.
 */
enum FormFieldType: string
{
    case TEXT = 'Text';
    case CHECKBOX = 'Checkbox';
    case RADIO = 'Radio';
    case DROPDOWN = 'Dropdown';
    case LISTBOX = 'ListBox';
    case SIGNATURE = 'Signature';
    case DATE = 'Date';
    case TIME = 'Time';
    case BUTTON = 'Button';
    case COMBO_BOX = 'ComboBox';

    /**
     * Get human-readable name.
     */
    public function label(): string
    {
        return match ($this) {
            self::TEXT => 'Text Field',
            self::CHECKBOX => 'Checkbox',
            self::RADIO => 'Radio Button',
            self::DROPDOWN => 'Dropdown List',
            self::LISTBOX => 'List Box',
            self::SIGNATURE => 'Signature Field',
            self::DATE => 'Date Field',
            self::TIME => 'Time Field',
            self::BUTTON => 'Button',
            self::COMBO_BOX => 'Combo Box',
        };
    }

    /**
     * Check if field type supports multiple values.
     */
    public function supportsMultipleValues(): bool
    {
        return in_array($this, [self::CHECKBOX, self::LISTBOX]);
    }

    /**
     * Check if field type is editable by user.
     */
    public function isEditable(): bool
    {
        return in_array($this, [
            self::TEXT, self::CHECKBOX, self::RADIO, self::DROPDOWN,
            self::LISTBOX, self::SIGNATURE, self::DATE, self::TIME,
            self::COMBO_BOX,
        ]);
    }
}
