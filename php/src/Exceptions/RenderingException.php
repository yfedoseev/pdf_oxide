<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when PDF rendering operations fail.
 */
class RenderingException extends PdfException
{
    public function __construct(
        string $message = 'PDF rendering failed',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'RENDERING_ERROR', $context, 6, $previous);
    }
}
