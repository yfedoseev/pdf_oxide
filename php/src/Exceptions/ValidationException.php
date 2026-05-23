<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when validation operations fail.
 */
class ValidationException extends PdfException
{
    public function __construct(
        string $message = 'Validation failed',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'VALIDATION_ERROR', $context, 1, $previous);
    }
}
