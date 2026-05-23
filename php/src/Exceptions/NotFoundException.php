<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when a requested resource is not found.
 */
class NotFoundException extends PdfException
{
    public function __construct(
        string $message = 'Resource not found',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'NOT_FOUND', $context, 4, $previous);
    }
}
