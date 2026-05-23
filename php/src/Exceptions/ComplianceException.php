<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when compliance operations fail.
 */
class ComplianceException extends PdfException
{
    public function __construct(
        string $message = 'Compliance check failed',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'COMPLIANCE_ERROR', $context, 5, $previous);
    }
}
