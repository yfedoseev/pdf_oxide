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
        // No dedicated cdylib code for encryption errors — the C ABI
        // routes "permission_denied" / "encrypted password missing"
        // through ERR_PARSE (3) or ERR_INTERNAL (5) per origin. Use 0
        // here so the base-Exception numeric code is unambiguous in
        // PHP's exception chain; the symbolic 'ENCRYPTION_ERROR'
        // class code remains the routing key. Was 3 which collided
        // with ParseException's 3.
        parent::__construct($message, 'ENCRYPTION_ERROR', $context, 0, $previous);
    }
}
