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
        // through ERR_PARSE (3) or ERR_INTERNAL (5) per origin. -1
        // is deliberately out-of-band w.r.t. the 0..8 cdylib codes
        // so anyone inspecting Exception::getCode() can tell the
        // difference from a real cdylib error. The symbolic
        // 'ENCRYPTION_ERROR' (via PdfException::getErrorCode()) is
        // the routing key. Was 3 which collided with ParseException;
        // then briefly 0 which collided with SUCCESS.
        parent::__construct($message, 'ENCRYPTION_ERROR', $context, -1, $previous);
    }
}
