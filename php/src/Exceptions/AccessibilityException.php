<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when accessibility operations fail.
 */
class AccessibilityException extends PdfException
{
    public function __construct(
        string $message = 'Accessibility operation failed',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'ACCESSIBILITY_ERROR', $context, 11, $previous);
    }
}
