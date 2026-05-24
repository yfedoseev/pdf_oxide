<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

use Exception;

/**
 * Thrown when the pdf_oxide core hits an unexpected failure
 * (cdylib error code 5 / `ERR_INTERNAL`). Mirrors the C# binding's
 * `PdfOxide.Exceptions.InternalError` and the Ruby binding's
 * `PdfOxide::InternalError`.
 *
 * <p>Per `feedback_extraction_graceful_fallback`: this surfaces a
 * fail-closed condition from the core library — there is no
 * recovery path other than to retry against a different input.
 */
class InternalError extends PdfException
{
    public function __construct(
        string $message = 'Internal error in pdf_oxide core',
        array $context = [],
        ?Exception $previous = null
    ) {
        parent::__construct($message, 'INTERNAL_ERROR', $context, 5, $previous);
    }
}
