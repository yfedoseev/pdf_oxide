<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when digital signature operations fail.
 */
class SignatureException extends PdfException
{
    public function __construct(
        string $message = 'Signature operation failed',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'SIGNATURE_ERROR', $context, 8, $previous);
    }
}
