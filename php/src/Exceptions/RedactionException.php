<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when redaction operations fail.
 */
class RedactionException extends PdfException
{
    public function __construct(
        string $message = 'Redaction operation failed',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'REDACTION_ERROR', $context, 9, $previous);
    }
}
