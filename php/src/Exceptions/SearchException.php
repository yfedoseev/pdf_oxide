<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

use Exception;

/**
 * Thrown when a text-search operation fails inside the cdylib
 * (cdylib error code 7 / `ERR_SEARCH`). Mirrors the C# binding's
 * `PdfOxide.Exceptions.SearchException`.
 *
 * <p>This is distinct from {@see ValidationException} (bad search
 * arguments) — it indicates the search engine itself failed after
 * accepting the inputs.
 */
class SearchException extends PdfException
{
    public function __construct(
        string $message = 'Search operation failed',
        array $context = [],
        ?Exception $previous = null
    ) {
        parent::__construct($message, 'SEARCH_ERROR', $context, 7, $previous);
    }
}
