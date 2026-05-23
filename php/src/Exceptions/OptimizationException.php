<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when optimization operations fail.
 */
class OptimizationException extends PdfException
{
    public function __construct(
        string $message = 'Optimization operation failed',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'OPTIMIZATION_ERROR', $context, 12, $previous);
    }
}
