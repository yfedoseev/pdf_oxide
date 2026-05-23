<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when a PDF cannot be parsed.
 */
class ParseException extends PdfException
{
    public function __construct(
        string $message = 'Failed to parse PDF document',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'PARSE_ERROR', $context, 3, $previous);
    }
}
