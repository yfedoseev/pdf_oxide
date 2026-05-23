<?php

declare(strict_types=1);

namespace PdfOxide\Types;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Represents a digital certificate (X.509).
 *
 * Used for signing PDFs and verifying signatures.
 * Loaded from PEM or DER format files.
 */
class Certificate
{
    private CData $handle;
    private FunctionBindings $bindings;
    private ?string $cachedSubject = null;
    private ?string $cachedIssuer = null;

    public function __construct(CData $handle, FunctionBindings $bindings)
    {
        $this->handle = $handle;
        $this->bindings = $bindings;
    }

    /**
     * Get the underlying FFI handle.
     *
     * @return CData The certificate FFI handle
     * @internal
     */
    public function getHandle(): CData
    {
        return $this->handle;
    }

    /**
     * Get certificate subject.
     *
     * @return string Subject Distinguished Name (DN)
     */
    public function getSubject(): string
    {
        if ($this->cachedSubject === null) {
            $this->cachedSubject = $this->bindings->pdfCertificateGetSubject($this->handle);
        }
        return $this->cachedSubject;
    }

    /**
     * Get certificate issuer.
     *
     * @return string Issuer Distinguished Name (DN)
     */
    public function getIssuer(): string
    {
        if ($this->cachedIssuer === null) {
            $this->cachedIssuer = $this->bindings->pdfCertificateGetIssuer($this->handle);
        }
        return $this->cachedIssuer;
    }

    /**
     * Extract common name from subject DN.
     *
     * @return string|null Common name or null if not found
     */
    public function getCommonName(): ?string
    {
        return $this->extractDnField($this->getSubject(), 'CN');
    }

    /**
     * Extract organization from subject DN.
     *
     * @return string|null Organization or null if not found
     */
    public function getOrganization(): ?string
    {
        return $this->extractDnField($this->getSubject(), 'O');
    }

    /**
     * Extract organization unit from subject DN.
     *
     * @return string|null Organization unit or null if not found
     */
    public function getOrganizationUnit(): ?string
    {
        return $this->extractDnField($this->getSubject(), 'OU');
    }

    /**
     * Extract country from subject DN.
     *
     * @return string|null Country code or null if not found
     */
    public function getCountry(): ?string
    {
        return $this->extractDnField($this->getSubject(), 'C');
    }

    /**
     * Extract field from DN string.
     *
     * @param string $dn Distinguished Name string
     * @param string $field Field name (CN, O, OU, C, etc.)
     * @return string|null Field value or null if not found
     */
    private function extractDnField(string $dn, string $field): ?string
    {
        if (preg_match('/' . preg_quote($field) . '=([^,]+)/', $dn, $matches)) {
            return $matches[1];
        }
        return null;
    }

    /**
     * Check if certificate is self-signed.
     *
     * @return bool True if subject equals issuer
     */
    public function isSelfSigned(): bool
    {
        return $this->getSubject() === $this->getIssuer();
    }

    /**
     * Get certificate info as array.
     *
     * @return array Certificate information
     */
    public function toArray(): array
    {
        return [
            'subject' => $this->getSubject(),
            'issuer' => $this->getIssuer(),
            'common_name' => $this->getCommonName(),
            'organization' => $this->getOrganization(),
            'organization_unit' => $this->getOrganizationUnit(),
            'country' => $this->getCountry(),
            'self_signed' => $this->isSelfSigned(),
        ];
    }

    /**
     * Free certificate resources.
     */
    public function __destruct()
    {
        $this->bindings->pdfCertificateFree($this->handle);
    }
}
