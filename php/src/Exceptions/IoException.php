<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when an I/O error occurs during PDF operations.
 */
class IoException extends PdfException
{
    public function __construct(
        string $message = 'I/O error during PDF operation',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'IO_ERROR', $context, 2, $previous);
    }
}
