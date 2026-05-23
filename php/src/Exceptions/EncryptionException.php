<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

/**
 * Exception thrown when encryption/decryption operations fail.
 */
class EncryptionException extends PdfException
{
    public function __construct(
        string $message = 'Encryption operation failed',
        array $context = [],
        ?PdfException $previous = null
    ) {
        parent::__construct($message, 'ENCRYPTION_ERROR', $context, 3, $previous);
    }
}
