<?php

declare(strict_types=1);

namespace PdfOxide\Types;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Represents a digital signature in a PDF.
 *
 * Contains signature details, signer information, and verification capability.
 */
class Signature
{
    private CData $handle;
    private FunctionBindings $bindings;
    private ?string $cachedReason = null;
    private ?string $cachedSigner = null;
    private ?string $cachedDate = null;
    private ?bool $cachedValid = null;

    public function __construct(CData $handle, FunctionBindings $bindings)
    {
        $this->handle = $handle;
        $this->bindings = $bindings;
    }

    /**
     * Get the underlying FFI handle.
     *
     * @return CData The signature FFI handle
     * @internal
     */
    public function getHandle(): CData
    {
        return $this->handle;
    }

    /**
     * Get signature reason.
     *
     * @return string Reason for signing
     */
    public function getReason(): string
    {
        if ($this->cachedReason === null) {
            $this->cachedReason = $this->bindings->pdfSignatureGetReason($this->handle);
        }
        return $this->cachedReason;
    }

    /**
     * Get signer name.
     *
     * @return string Name of person/entity that signed
     */
    public function getSigner(): string
    {
        if ($this->cachedSigner === null) {
            $this->cachedSigner = $this->bindings->pdfSignatureGetSigner($this->handle);
        }
        return $this->cachedSigner;
    }

    /**
     * Get signature date.
     *
     * @return string ISO 8601 date string
     */
    public function getDate(): string
    {
        if ($this->cachedDate === null) {
            $this->cachedDate = $this->bindings->pdfSignatureGetDate($this->handle);
        }
        return $this->cachedDate;
    }

    /**
     * Verify signature validity.
     *
     * @param Certificate|null $trustedCertificate Trusted certificate for verification
     * @return bool True if signature is valid
     */
    public function verify(?Certificate $trustedCertificate = null): bool
    {
        if ($this->cachedValid === null) {
            $certHandle = $trustedCertificate ? $trustedCertificate->getHandle() : null;
            $this->cachedValid = $this->bindings->pdfSignatureVerify($this->handle, $certHandle);
        }
        return $this->cachedValid;
    }

    /**
     * Get signature info as array.
     *
     * @return array Signature information
     */
    public function toArray(): array
    {
        return [
            'signer' => $this->getSigner(),
            'reason' => $this->getReason(),
            'date' => $this->getDate(),
            'valid' => $this->verify(),
        ];
    }

    /**
     * Free signature resources.
     */
    public function __destruct()
    {
        $this->bindings->pdfSignatureFree($this->handle);
    }
}
