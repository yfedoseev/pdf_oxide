<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when search operations fail.
 */
class SearchException extends PdfException
{
    public function __construct(
        string $message = 'Search operation failed',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'SEARCH_ERROR', $context, 4, $previous);
    }
}
