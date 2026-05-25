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
        // -1 puts SignatureException out-of-band of the cdylib wire codes
        // (0-8): signing failures have no dedicated wire code and would
        // otherwise collide with UnsupportedException (code 8). Matches
        // EncryptionException's convention.
        parent::__construct($message, 'SIGNATURE_ERROR', $context, -1, $previous);
    }
}
