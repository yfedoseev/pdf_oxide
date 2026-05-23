<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when an operation is attempted on an invalid object state.
 */
class InvalidStateException extends PdfException
{
    public function __construct(
        string $message = 'Invalid object state',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'INVALID_STATE', $context, 1, $previous);
    }
}
