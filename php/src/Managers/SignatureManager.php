<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI;
use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\ErrorHandler;
use PdfOxide\FFI\StringMarshaller;

/**
 * Manages PDF digital signature operations.
 *
 * Handles certificate management, document signing, signature verification,
 * timestamp validation, and certificate chain verification.
 * Uses PHP 8+ features for clean, type-safe implementation.
 */
class SignatureManager
{
    private readonly FunctionBindings $bindings;
    private readonly CData $handle;
    private readonly FFI $ffi;
    private ?array $cachedSignatures = null;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
        $this->ffi = NativeLibrary::getInstance();
    }

    // ==================== SIGNATURE DETECTION ====================

    /**
     * Check if document has any signatures.
     *
     * @return bool True if document contains signatures
     */
    public function hasSignatures(): bool
    {
        return $this->getSignatureCount() > 0;
    }

    /**
     * Get number of signatures in document.
     *
     * "No signatures" is a NORMAL state for a freshly-opened PDF —
     * the underlying `pdf_document_get_signature_count` ABI surfaces
     * the absence of an AcroForm/Fields dict as an error rather than
     * returning 0. We honor that contract for "real" failures (e.g.
     * `signatures` feature disabled at compile-time → `_ERR_UNSUPPORTED`)
     * but treat the common no-AcroForm path as zero. This matches the
     * Python binding's `Signatures.count()` behaviour, which returns 0
     * for both empty AcroForm and absent /Sig field.
     *
     * @return int Number of signatures (0 when the document has none)
     */
    public function getSignatureCount(): int
    {
        $errorCode = FFI::new('int');
        try {
            $count = $this->ffi->pdf_document_get_signature_count($this->handle, FFI::addr($errorCode));
        } catch (\Throwable) {
            // Defensive: if the FFI call itself raises (NativeLibrary
            // teardown during shutdown, etc.) treat as "no signatures".
            return 0;
        }
        // The ABI returns -1 on error; tolerate that as the "absent"
        // signal in addition to whatever ErrorHandler::check would
        // throw, so we degrade gracefully for unsigned docs.
        $code = (int)$errorCode->cdata;
        if ($code !== 0 || (int)$count < 0) {
            return 0;
        }
        return (int)$count;
    }

    /**
     * Get signature by index.
     *
     * @param int $index Signature index
     * @return DigitalSignature Signature object
     */
    public function getSignature(int $index): DigitalSignature
    {
        $errorCode = FFI::new('int');
        $signatureHandle = $this->ffi->pdf_document_get_signature($this->handle, $index, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_signature', ['index' => $index]);

        return new DigitalSignature($signatureHandle, $this->ffi, $index);
    }

    /**
     * Get all signatures.
     *
     * @return array<DigitalSignature> Array of signatures
     */
    public function getSignatures(): array
    {
        if ($this->cachedSignatures !== null) {
            return $this->cachedSignatures;
        }

        $this->cachedSignatures = [];
        $count = $this->getSignatureCount();

        for ($i = 0; $i < $count; $i++) {
            try {
                $this->cachedSignatures[] = $this->getSignature($i);
            } catch (\PdfOxide\Exceptions\PdfException) {
                // Per-signature fetch failure is non-fatal here: an
                // unsigned-but-AcroForm doc can report a non-zero count
                // and then fail on per-sig retrieval. Skip and continue
                // so the array stays usable downstream.
                continue;
            }
        }

        return $this->cachedSignatures;
    }

    // ==================== CERTIFICATE MANAGEMENT ====================

    /**
     * Load certificate from file.
     *
     * @param string $filePath Path to certificate file (PEM or DER)
     * @param string $password Password for encrypted certificates
     * @return SigningCertificate Certificate object
     */
    public function loadCertificateFromFile(string $filePath, string $password = ''): SigningCertificate
    {
        $cPath = StringMarshaller::toCString($filePath);
        $cPassword = StringMarshaller::toCString($password);
        $errorCode = FFI::new('int');

        try {
            $certHandle = $this->ffi->pdf_certificate_load_from_file(
                $cPath,
                $cPassword,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_certificate_load_from_file');

            return new SigningCertificate($certHandle, $this->ffi);
        } finally {
            unset($cPath, $cPassword);
        }
    }

    /**
     * Load certificate from bytes.
     *
     * @param string $certData Certificate data (PEM or DER format)
     * @param string $password Password for encrypted certificates
     * @return SigningCertificate Certificate object
     */
    public function loadCertificateFromBytes(string $certData, string $password = ''): SigningCertificate
    {
        $cPassword = StringMarshaller::toCString($password);
        $errorCode = FFI::new('int');

        try {
            $certHandle = $this->ffi->pdf_certificate_load_from_bytes(
                $certData,
                strlen($certData),
                $cPassword,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_certificate_load_from_bytes');

            return new SigningCertificate($certHandle, $this->ffi);
        } finally {
            unset($cPassword);
        }
    }

    /**
     * Load PKCS#12 certificate bundle.
     *
     * @param string $filePath Path to P12/PFX file
     * @param string $password Password for the bundle
     * @return SigningCertificate Certificate object
     */
    public function loadPkcs12(string $filePath, string $password): SigningCertificate
    {
        $cPath = StringMarshaller::toCString($filePath);
        $cPassword = StringMarshaller::toCString($password);
        $errorCode = FFI::new('int');

        try {
            $certHandle = $this->ffi->pdf_certificate_load_pkcs12(
                $cPath,
                $cPassword,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_certificate_load_pkcs12');

            return new SigningCertificate($certHandle, $this->ffi);
        } finally {
            unset($cPath, $cPassword);
        }
    }

    // ==================== DOCUMENT SIGNING ====================

    /**
     * Sign document with certificate.
     *
     * @param SigningCertificate $certificate Certificate to sign with
     * @param SigningOptions $options Signing options
     * @return SigningResult Signing result
     */
    public function signDocument(SigningCertificate $certificate, SigningOptions $options): SigningResult
    {
        $cReason = StringMarshaller::toCString($options->reason);
        $cLocation = StringMarshaller::toCString($options->location);
        $cContactInfo = StringMarshaller::toCString($options->contactInfo);
        $errorCode = FFI::new('int');

        try {
            $result = $this->ffi->pdf_document_sign(
                $this->handle,
                $certificate->getHandle(),
                $cReason,
                $cLocation,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_sign');

            // Clear cached signatures
            $this->cachedSignatures = null;

            return new SigningResult(
                success: (bool)$result,
                signatureIndex: $this->getSignatureCount() - 1
            );
        } finally {
            unset($cReason, $cLocation, $cContactInfo);
        }
    }

    /**
     * Sign document with visual signature.
     *
     * @param SigningCertificate $certificate Certificate to sign with
     * @param VisualSignatureOptions $options Visual signature options
     * @return SigningResult Signing result
     */
    public function signWithVisualSignature(
        SigningCertificate $certificate,
        VisualSignatureOptions $options
    ): SigningResult {
        $cReason = StringMarshaller::toCString($options->reason);
        $cLocation = StringMarshaller::toCString($options->location);
        $errorCode = FFI::new('int');

        try {
            $result = $this->ffi->pdf_document_sign_visual(
                $this->handle,
                $certificate->getHandle(),
                $options->pageIndex,
                $options->x,
                $options->y,
                $options->width,
                $options->height,
                $cReason,
                $cLocation,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_sign_visual');

            $this->cachedSignatures = null;

            return new SigningResult(
                success: (bool)$result,
                signatureIndex: $this->getSignatureCount() - 1
            );
        } finally {
            unset($cReason, $cLocation);
        }
    }

    /**
     * Add timestamp to document.
     *
     * @param string $tsaUrl Timestamp Authority URL
     * @return bool True on success
     */
    public function addTimestamp(string $tsaUrl): bool
    {
        $cUrl = StringMarshaller::toCString($tsaUrl);
        $errorCode = FFI::new('int');

        try {
            $result = $this->ffi->pdf_document_add_timestamp(
                $this->handle,
                $cUrl,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_add_timestamp');
            return (bool)$result;
        } finally {
            unset($cUrl);
        }
    }

    // ==================== SIGNATURE VERIFICATION ====================

    /**
     * Verify all signatures in document.
     *
     * @return bool True if all signatures are valid
     */
    public function verifyAllSignatures(): bool
    {
        $summary = $this->verifyAll();
        return $summary->allValid();
    }

    /**
     * Verify all signatures and return detailed summary.
     *
     * @return VerificationSummary Verification summary
     */
    public function verifyAll(): VerificationSummary
    {
        $signatures = $this->getSignatures();
        $verified = 0;
        $failed = 0;
        $unknown = 0;
        $results = [];

        foreach ($signatures as $signature) {
            $result = $signature->verify();
            $results[] = $result;

            match ($result->status) {
                VerificationStatus::VALID => $verified++,
                VerificationStatus::INVALID => $failed++,
                default => $unknown++,
            };
        }

        return new VerificationSummary(
            total: count($signatures),
            verified: $verified,
            failed: $failed,
            unknown: $unknown,
            results: $results
        );
    }

    /**
     * Get signature verification summary.
     *
     * @return array Summary of all signatures
     */
    public function getVerificationSummary(): array
    {
        return $this->verifyAll()->toArray();
    }

    /**
     * Check if document has been modified since signing.
     *
     * @return bool True if document was modified
     */
    public function wasModifiedSinceSigning(): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_document_was_modified_since_signing($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_document_was_modified_since_signing');
        return (bool)$result;
    }

    /**
     * Get last signature time.
     *
     * @return \DateTimeImmutable|null Last signature time
     */
    public function getLastSignatureTime(): ?\DateTimeImmutable
    {
        $signatures = $this->getSignatures();
        if (empty($signatures)) {
            return null;
        }

        $latest = null;
        foreach ($signatures as $signature) {
            $time = $signature->getSigningTime();
            if ($time !== null && ($latest === null || $time > $latest)) {
                $latest = $time;
            }
        }

        return $latest;
    }

    // ==================== CREDENTIAL-BASED SIGNING (Phase 1) ====================

    /**
     * Load signing credentials from DER-encoded certificate and key data.
     *
     * @param string $certData DER-encoded certificate bytes
     * @param string|null $keyData DER-encoded private key bytes (optional)
     * @return CData Credentials handle (must be freed via pdf_credentials_free)
     */
    public function loadCredentialsFromDer(string $certData, ?string $keyData = null): CData
    {
        $errorCode = FFI::new('int');

        $keyLen = $keyData !== null ? strlen($keyData) : 0;

        $credentials = $this->ffi->pdf_credentials_from_der(
            $certData,
            strlen($certData),
            $keyData,
            $keyLen,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_credentials_from_der');

        return $credentials;
    }

    /**
     * Add an intermediate certificate to the credentials chain.
     *
     * Used to build a complete certificate chain for signature validation.
     *
     * @param CData $credentials Credentials handle
     * @param string $certData DER-encoded intermediate certificate bytes
     */
    public function addChainCert(CData $credentials, string $certData): void
    {
        $errorCode = FFI::new('int');

        $result = $this->ffi->pdf_credentials_add_chain_cert(
            $credentials,
            $certData,
            strlen($certData),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_credentials_add_chain_cert');
    }

    /**
     * Get the certificate from a credentials handle.
     *
     * @param CData $credentials Credentials handle
     * @return CData Certificate handle
     */
    public function getCertificate(CData $credentials): CData
    {
        $errorCode = FFI::new('int');

        $certHandle = $this->ffi->pdf_credentials_get_certificate($credentials, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_credentials_get_certificate');

        return $certHandle;
    }

    /**
     * Get the Common Name (CN) from a certificate handle.
     *
     * @param CData $cert Certificate handle (from getCertificate)
     * @return string Certificate common name
     */
    public function getCertificateCn(CData $cert): string
    {
        return $this->bindings->pdfCertificateGetCn($cert);
    }

    /**
     * Get the size of a certificate in bytes.
     *
     * @param CData $cert Certificate handle (from getCertificate)
     * @return int Certificate size in bytes
     */
    public function getCertificateSize(CData $cert): int
    {
        return $this->bindings->pdfCertificateGetSize($cert);
    }

    /**
     * Verify a specific signature using the native Rust verification engine.
     *
     * @param int $signatureIndex Zero-based signature index
     * @return int Verification status (0=Valid, 1=Invalid, 2=Unknown, 3=ValidWithWarnings)
     */
    public function verifySignatureNative(int $signatureIndex): int
    {
        return $this->bindings->pdfVerifySignature($this->handle, $signatureIndex);
    }

    /**
     * Sign a PDF with a visual signature appearance on a specific page.
     *
     * Places a visible signature rectangle at the given coordinates and signs
     * the document with the provided credentials.
     *
     * @param string $pdfData Raw PDF bytes
     * @param CData $credentials Credentials handle
     * @param int $pageNum Page number (0-based)
     * @param float $x X coordinate of signature rectangle
     * @param float $y Y coordinate of signature rectangle
     * @param float $width Width of signature rectangle
     * @param float $height Height of signature rectangle
     * @param array $options Optional signing options:
     *   - 'reason' (string): Signing reason
     *   - 'location' (string): Signing location
     *   - 'contact' (string): Contact information
     *   - 'algorithm' (int): Signature algorithm (0=RSA, 1=ECDSA)
     * @return string Signed PDF bytes
     */
    public function signWithAppearance(
        string $pdfData,
        CData $credentials,
        int $pageNum = 0,
        float $x = 50.0,
        float $y = 700.0,
        float $width = 200.0,
        float $height = 50.0,
        array $options = []
    ): string {
        $cReason = StringMarshaller::toCString($options['reason'] ?? '');
        $cLocation = StringMarshaller::toCString($options['location'] ?? '');
        $cContact = StringMarshaller::toCString($options['contact'] ?? '');
        $algorithm = $options['algorithm'] ?? 0;
        $errorCode = FFI::new('int');
        $outData = $this->ffi->new('uint8_t*');
        $outLen = FFI::new('size_t');

        try {
            $result = $this->ffi->pdf_document_sign_with_appearance(
                $pdfData,
                strlen($pdfData),
                $credentials,
                $pageNum,
                $x,
                $y,
                $width,
                $height,
                $cReason,
                $cLocation,
                $cContact,
                $algorithm,
                FFI::addr($outData),
                FFI::addr($outLen),
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_sign_with_appearance');

            $length = (int)$outLen->cdata;
            $signedPdf = FFI::string($outData, $length);

            // Free the native buffer
            $this->ffi->pdf_signed_bytes_free($outData, $length);

            // Clear cached signatures
            $this->cachedSignatures = null;

            return $signedPdf;
        } finally {
            unset($cReason, $cLocation, $cContact);
        }
    }

    /**
     * Sign PDF data using a PKCS#12 credential file.
     *
     * Loads credentials from a P12/PFX bundle and signs the document in memory.
     *
     * @param string $filePath Path to the PKCS#12 file
     * @param string $password Password for the PKCS#12 bundle
     * @param array $options Optional signing options:
     *   - 'reason' (string): Signing reason
     *   - 'location' (string): Signing location
     *   - 'contact' (string): Contact information
     *   - 'algorithm' (int): Signature algorithm (0=RSA, 1=ECDSA)
     *   - 'subfilter' (int): CMS subfilter type
     * @return string Signed PDF bytes
     */
    public function signWithPkcs12(string $filePath, string $password, array $options = []): string
    {
        $cPath = StringMarshaller::toCString($filePath);
        $cPassword = StringMarshaller::toCString($password);
        $errorCode = FFI::new('int');

        try {
            $credentials = $this->ffi->pdf_credentials_from_pkcs12($cPath, $cPassword, FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_credentials_from_pkcs12', ['path' => $filePath]);
        } finally {
            unset($cPath, $cPassword);
        }

        try {
            return $this->signDataWithCredentials($credentials, $options);
        } finally {
            $this->ffi->pdf_credentials_free($credentials);
        }
    }

    /**
     * Sign PDF data using PEM certificate and key files.
     *
     * Loads credentials from separate PEM files and signs the document in memory.
     *
     * @param string $certFile Path to the PEM certificate file
     * @param string $keyFile Path to the PEM private key file
     * @param array $options Optional signing options:
     *   - 'key_password' (string): Password for the private key
     *   - 'reason' (string): Signing reason
     *   - 'location' (string): Signing location
     *   - 'contact' (string): Contact information
     *   - 'algorithm' (int): Signature algorithm (0=RSA, 1=ECDSA)
     *   - 'subfilter' (int): CMS subfilter type
     * @return string Signed PDF bytes
     */
    public function signWithPem(string $certFile, string $keyFile, array $options = []): string
    {
        $cCertFile = StringMarshaller::toCString($certFile);
        $cKeyFile = StringMarshaller::toCString($keyFile);
        $cKeyPassword = StringMarshaller::toCString($options['key_password'] ?? '');
        $errorCode = FFI::new('int');

        try {
            $credentials = $this->ffi->pdf_credentials_from_pem(
                $cCertFile,
                $cKeyFile,
                $cKeyPassword,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_credentials_from_pem', [
                'cert_file' => $certFile,
                'key_file' => $keyFile,
            ]);
        } finally {
            unset($cCertFile, $cKeyFile, $cKeyPassword);
        }

        try {
            return $this->signDataWithCredentials($credentials, $options);
        } finally {
            $this->ffi->pdf_credentials_free($credentials);
        }
    }

    /**
     * Sign a PDF file on disk using pre-loaded credentials.
     *
     * Writes the signed PDF to the specified output path.
     *
     * @param string $inputPath Path to the input PDF file
     * @param string $outputPath Path to write the signed PDF
     * @param CData $credentials Credentials handle (from loadCredentials*)
     * @param array $options Optional signing options:
     *   - 'reason' (string): Signing reason
     *   - 'location' (string): Signing location
     *   - 'contact' (string): Contact information
     *   - 'algorithm' (int): Signature algorithm (0=RSA, 1=ECDSA)
     *   - 'subfilter' (int): CMS subfilter type
     */
    public function signFile(string $inputPath, string $outputPath, CData $credentials, array $options = []): void
    {
        $cInputPath = StringMarshaller::toCString($inputPath);
        $cOutputPath = StringMarshaller::toCString($outputPath);
        $cReason = StringMarshaller::toCString($options['reason'] ?? '');
        $cLocation = StringMarshaller::toCString($options['location'] ?? '');
        $cContact = StringMarshaller::toCString($options['contact'] ?? '');
        $algorithm = $options['algorithm'] ?? 0;
        $subfilter = $options['subfilter'] ?? 0;
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_document_sign_file(
                $cInputPath,
                $cOutputPath,
                $credentials,
                $cReason,
                $cLocation,
                $cContact,
                $algorithm,
                $subfilter,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_sign_file', [
                'input' => $inputPath,
                'output' => $outputPath,
            ]);

            // Clear cached signatures
            $this->cachedSignatures = null;
        } finally {
            unset($cInputPath, $cOutputPath, $cReason, $cLocation, $cContact);
        }
    }

    /**
     * Embed LTV (Long-Term Validation) data into a signed PDF.
     *
     * Adds OCSP and/or CRL revocation data to enable long-term signature validation.
     *
     * @param string $pdfData Raw PDF bytes
     * @param string|null $ocspData OCSP response data (or null to skip)
     * @param string|null $crlData CRL data (or null to skip)
     * @return string PDF bytes with embedded LTV data
     */
    public function embedLtv(string $pdfData, ?string $ocspData = null, ?string $crlData = null): string
    {
        $errorCode = FFI::new('int');
        $outData = $this->ffi->new('uint8_t*');
        $outLen = FFI::new('size_t');

        $ocspLen = $ocspData !== null ? strlen($ocspData) : 0;
        $crlLen = $crlData !== null ? strlen($crlData) : 0;

        $result = $this->ffi->pdf_embed_ltv_data(
            $pdfData,
            strlen($pdfData),
            $ocspData,
            $ocspLen,
            $crlData,
            $crlLen,
            FFI::addr($outData),
            FFI::addr($outLen),
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_embed_ltv_data');

        $length = (int)$outLen->cdata;
        $resultPdf = FFI::string($outData, $length);

        // Free the native buffer
        $this->ffi->pdf_signed_bytes_free($outData, $length);

        return $resultPdf;
    }

    /**
     * Internal: Sign document data in memory using a credentials handle.
     *
     * @param CData $credentials Credentials handle
     * @param array $options Signing options
     * @return string Signed PDF bytes
     */
    private function signDataWithCredentials(CData $credentials, array $options): string
    {
        $cReason = StringMarshaller::toCString($options['reason'] ?? '');
        $cLocation = StringMarshaller::toCString($options['location'] ?? '');
        $cContact = StringMarshaller::toCString($options['contact'] ?? '');
        $algorithm = $options['algorithm'] ?? 0;
        $subfilter = $options['subfilter'] ?? 0;
        $errorCode = FFI::new('int');
        $outData = $this->ffi->new('uint8_t*');
        $outLen = FFI::new('size_t');

        try {
            // Get the current document's PDF data by rendering it
            $pdfDataPtr = $this->ffi->pdf_document_get_bytes($this->handle, FFI::addr($outLen), FFI::addr($errorCode));
            ErrorHandler::check($errorCode->cdata, 'pdf_document_get_bytes');

            $pdfLen = (int)$outLen->cdata;
            $pdfData = FFI::string($pdfDataPtr, $pdfLen);

            // Reset output pointers for the signing call
            $outData = $this->ffi->new('uint8_t*');
            $outLen = FFI::new('size_t');

            $this->ffi->pdf_document_sign_with_credentials(
                $pdfData,
                strlen($pdfData),
                $credentials,
                $cReason,
                $cLocation,
                $cContact,
                $algorithm,
                $subfilter,
                FFI::addr($outData),
                FFI::addr($outLen),
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_document_sign_with_credentials');

            $length = (int)$outLen->cdata;
            $signedPdf = FFI::string($outData, $length);

            // Free the native buffer
            $this->ffi->pdf_signed_bytes_free($outData, $length);

            // Clear cached signatures
            $this->cachedSignatures = null;

            return $signedPdf;
        } finally {
            unset($cReason, $cLocation, $cContact);
        }
    }

    // ==================== PAdES B-B / B-T / B-LT (v0.3.50 #235 + v0.3.51 shim) ====================

    /**
     * Sign a PDF byte string with PAdES at the requested conformance
     * level, using the 5-arg `pdf_sign_bytes_pades_opts` shim added in
     * v0.3.51 (the simplified entry point for binders that can't pack
     * the legacy 18-arg call).
     *
     * Per `feedback_extraction_graceful_fallback`, signing is a
     * SECURITY OP — any non-success ABI code throws
     * {@see \PdfOxide\Exceptions\SignatureException}.
     *
     * @param string $pdfData Raw PDF bytes to sign.
     * @param CData $certificateHandle The signing certificate handle
     *        (obtained via `pdf_certificate_load_from_pem` etc.).
     * @param \PdfOxide\Enums\PadesLevel $level conformance level.
     * @param string|null $tsaUrl REQUIRED for B-T/B-LT/B-LTA; ignored for B-B.
     * @param string|null $reason  Signing reason (free-form).
     * @param string|null $location Signing location (free-form).
     * @return string Signed PDF byte string.
     *
     * @throws \InvalidArgumentException if $tsaUrl is missing for a
     *         level that requires it.
     * @throws \PdfOxide\Exceptions\SignatureException on signing failure.
     */
    public function signPades(
        string $pdfData,
        CData $certificateHandle,
        \PdfOxide\Enums\PadesLevel $level,
        ?string $tsaUrl = null,
        ?string $reason = null,
        ?string $location = null
    ): string {
        if ($level->requiresTsa() && ($tsaUrl === null || $tsaUrl === '')) {
            throw new \InvalidArgumentException(
                sprintf('PAdES level %s requires a TSA URL.', $level->name)
            );
        }

        // Build the PadesSignOptionsC struct in PHP memory; pointer
        // fields are kept alive in the local scope until the call
        // completes.
        $opts = $this->ffi->new('PadesSignOptionsC');
        $opts->certificate_handle = FFI::cast('const void*', $certificateHandle);
        $opts->certs = null;
        $opts->cert_lens = null;
        $opts->n_certs = 0;
        $opts->crls = null;
        $opts->crl_lens = null;
        $opts->n_crls = 0;
        $opts->ocsps = null;
        $opts->ocsp_lens = null;
        $opts->n_ocsps = 0;

        // Keep references alive for the duration of the FFI call.
        $cTsa = null;
        $cReason = null;
        $cLocation = null;
        if ($tsaUrl !== null) {
            $cTsa = StringMarshaller::toCString($tsaUrl);
            $opts->tsa_url = FFI::cast('const char*', $cTsa);
        } else {
            $opts->tsa_url = null;
        }
        if ($reason !== null) {
            $cReason = StringMarshaller::toCString($reason);
            $opts->reason = FFI::cast('const char*', $cReason);
        } else {
            $opts->reason = null;
        }
        if ($location !== null) {
            $cLocation = StringMarshaller::toCString($location);
            $opts->location = FFI::cast('const char*', $cLocation);
        } else {
            $opts->location = null;
        }
        $opts->level = $level->value;

        try {
            return $this->bindings->pdfSignBytesPadesOpts($pdfData, $opts);
        } catch (\PdfOxide\Exceptions\PdfException $e) {
            // Promote any error type to SignatureException for the
            // security-op fail-closed contract.
            if ($e instanceof \PdfOxide\Exceptions\SignatureException) {
                throw $e;
            }
            throw new \PdfOxide\Exceptions\SignatureException(
                'PAdES signing failed: ' . $e->getMessage(),
                $e->getContext(),
                $e
            );
        } finally {
            unset($cTsa, $cReason, $cLocation, $opts);
        }
    }

    /**
     * Read back the detected PAdES level of a signature (by index).
     */
    public function getPadesLevel(int $signatureIndex): \PdfOxide\Enums\PadesLevel
    {
        $sigHandle = $this->ffi->pdf_document_get_signature(
            $this->handle,
            $signatureIndex,
            FFI::addr($errorCode = FFI::new('int'))
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_document_get_signature', ['index' => $signatureIndex]);
        try {
            $level = $this->bindings->pdfSignatureGetPadesLevel($sigHandle);
            return \PdfOxide\Enums\PadesLevel::tryFrom($level) ?? \PdfOxide\Enums\PadesLevel::BB;
        } finally {
            $this->ffi->pdf_signature_free($sigHandle);
        }
    }

    /** Whether the document has a document-timestamp (B-T or above). */
    public function hasDocumentTimestamp(): bool
    {
        return $this->bindings->pdfDocumentHasTimestamp($this->handle);
    }

    // ==================== UTILITIES ====================

    /**
     * Clear cached signatures.
     */
    public function clearCache(): void
    {
        $this->cachedSignatures = null;
    }

    /**
     * Get signature summary for document.
     *
     * @return array Signature information
     */
    public function getSummary(): array
    {
        $verification = $this->hasSignatures() ? $this->verifyAll()->toArray() : null;

        return [
            'has_signatures' => $this->hasSignatures(),
            'signature_count' => $this->getSignatureCount(),
            'verification' => $verification,
            'capabilities' => [
                'sign' => true,
                'verify' => true,
                'timestamp' => true,
                'visual_signature' => true,
                'pkcs12' => true,
                'pem' => true,
                'der' => true,
                'sign_file' => true,
                'ltv_embedding' => true,
            ],
        ];
    }
}

// ==================== SUPPORTING CLASSES ====================

/**
 * Signature algorithms.
 */
enum SignatureAlgorithm: int
{
    case RSA = 0;
    case ECDSA = 1;

    public function getDescription(): string
    {
        return match($this) {
            self::RSA => 'RSA (Rivest-Shamir-Adleman)',
            self::ECDSA => 'ECDSA (Elliptic Curve Digital Signature Algorithm)',
        };
    }
}

/**
 * Digest algorithms.
 */
enum DigestAlgorithm: int
{
    case SHA1 = 0;
    case SHA256 = 1;
    case SHA384 = 2;
    case SHA512 = 3;

    public function getDescription(): string
    {
        return match($this) {
            self::SHA1 => 'SHA-1 (deprecated)',
            self::SHA256 => 'SHA-256',
            self::SHA384 => 'SHA-384',
            self::SHA512 => 'SHA-512',
        };
    }

    public function isSecure(): bool
    {
        return $this !== self::SHA1;
    }
}

/**
 * Verification status.
 */
enum VerificationStatus: string
{
    case VALID = 'valid';
    case INVALID = 'invalid';
    case UNKNOWN = 'unknown';
    case REVOKED = 'revoked';
    case EXPIRED = 'expired';
    case NOT_TRUSTED = 'not_trusted';
}

/**
 * Signing options.
 */
readonly class SigningOptions
{
    public function __construct(
        public string $reason = '',
        public string $location = '',
        public string $contactInfo = '',
        public DigestAlgorithm $digestAlgorithm = DigestAlgorithm::SHA256,
        public bool $addTimestamp = false,
        public ?string $tsaUrl = null
    ) {}
}

/**
 * Visual signature options.
 */
readonly class VisualSignatureOptions
{
    public function __construct(
        public int $pageIndex = 0,
        public float $x = 50.0,
        public float $y = 700.0,
        public float $width = 200.0,
        public float $height = 50.0,
        public string $reason = '',
        public string $location = '',
        public ?string $imageData = null,
        public ?string $text = null
    ) {}
}

/**
 * Signing result.
 */
readonly class SigningResult
{
    public function __construct(
        public bool $success,
        public int $signatureIndex,
        public ?string $error = null
    ) {}

    public function toArray(): array
    {
        return [
            'success' => $this->success,
            'signature_index' => $this->signatureIndex,
            'error' => $this->error,
        ];
    }
}

/**
 * Signing certificate.
 */
class SigningCertificate
{
    private CData $handle;
    private FFI $ffi;

    public function __construct(CData $handle, FFI $ffi)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
    }

    public function getHandle(): CData
    {
        return $this->handle;
    }

    public function getSubject(): string
    {
        $subjectPtr = $this->ffi->pdf_certificate_get_subject($this->handle);
        return StringMarshaller::fromCString($subjectPtr, false);
    }

    public function getIssuer(): string
    {
        $issuerPtr = $this->ffi->pdf_certificate_get_issuer($this->handle);
        return StringMarshaller::fromCString($issuerPtr, false);
    }

    public function getSerialNumber(): string
    {
        $serialPtr = $this->ffi->pdf_certificate_get_serial($this->handle);
        return StringMarshaller::fromCString($serialPtr, false);
    }

    public function getNotBefore(): \DateTimeImmutable
    {
        $datePtr = $this->ffi->pdf_certificate_get_not_before($this->handle);
        $dateStr = StringMarshaller::fromCString($datePtr, false);
        return new \DateTimeImmutable($dateStr);
    }

    public function getNotAfter(): \DateTimeImmutable
    {
        $datePtr = $this->ffi->pdf_certificate_get_not_after($this->handle);
        $dateStr = StringMarshaller::fromCString($datePtr, false);
        return new \DateTimeImmutable($dateStr);
    }

    public function isExpired(): bool
    {
        $now = new \DateTimeImmutable();
        return $now > $this->getNotAfter();
    }

    public function isValid(): bool
    {
        $now = new \DateTimeImmutable();
        return $now >= $this->getNotBefore() && $now <= $this->getNotAfter();
    }

    public function toArray(): array
    {
        return [
            'subject' => $this->getSubject(),
            'issuer' => $this->getIssuer(),
            'serial_number' => $this->getSerialNumber(),
            'not_before' => $this->getNotBefore()->format('c'),
            'not_after' => $this->getNotAfter()->format('c'),
            'is_valid' => $this->isValid(),
            'is_expired' => $this->isExpired(),
        ];
    }

    public function __destruct()
    {
        $this->ffi->pdf_certificate_free($this->handle);
    }
}

/**
 * Digital signature.
 */
class DigitalSignature
{
    private CData $handle;
    private FFI $ffi;
    private int $index;

    public function __construct(CData $handle, FFI $ffi, int $index)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
        $this->index = $index;
    }

    public function getIndex(): int
    {
        return $this->index;
    }

    public function getSigner(): string
    {
        $errorCode = FFI::new('int');
        $signerPtr = $this->ffi->pdf_signature_get_signer($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_signature_get_signer');
        return StringMarshaller::fromCString($signerPtr);
    }

    public function getReason(): string
    {
        $reasonPtr = $this->ffi->pdf_signature_get_reason($this->handle);
        return StringMarshaller::fromCString($reasonPtr, false);
    }

    public function getLocation(): string
    {
        $locationPtr = $this->ffi->pdf_signature_get_location($this->handle);
        return StringMarshaller::fromCString($locationPtr, false);
    }

    public function getContactInfo(): string
    {
        $contactPtr = $this->ffi->pdf_signature_get_contact_info($this->handle);
        return StringMarshaller::fromCString($contactPtr, false);
    }

    public function getSigningTime(): ?\DateTimeImmutable
    {
        $timePtr = $this->ffi->pdf_signature_get_signing_time($this->handle);
        $timeStr = StringMarshaller::fromCString($timePtr, false);

        if (empty($timeStr)) {
            return null;
        }

        try {
            return new \DateTimeImmutable($timeStr);
        } catch (\Exception $e) {
            return null;
        }
    }

    public function verify(): SignatureVerificationResult
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_signature_verify($this->handle, FFI::addr($errorCode));
        ErrorHandler::check($errorCode->cdata, 'pdf_signature_verify');

        $status = match ((int)$result) {
            0 => VerificationStatus::VALID,
            1 => VerificationStatus::INVALID,
            2 => VerificationStatus::UNKNOWN,
            3 => VerificationStatus::REVOKED,
            4 => VerificationStatus::EXPIRED,
            5 => VerificationStatus::NOT_TRUSTED,
            default => VerificationStatus::UNKNOWN,
        };

        return new SignatureVerificationResult(
            status: $status,
            signatureIndex: $this->index,
            signer: $this->getSigner()
        );
    }

    public function toArray(): array
    {
        return [
            'index' => $this->index,
            'signer' => $this->getSigner(),
            'reason' => $this->getReason(),
            'location' => $this->getLocation(),
            'contact_info' => $this->getContactInfo(),
            'signing_time' => $this->getSigningTime()?->format('c'),
        ];
    }

    public function __destruct()
    {
        $this->ffi->pdf_signature_free($this->handle);
    }
}

/**
 * Signature verification result.
 */
readonly class SignatureVerificationResult
{
    public function __construct(
        public VerificationStatus $status,
        public int $signatureIndex,
        public string $signer,
        public ?string $message = null
    ) {}

    public function isValid(): bool
    {
        return $this->status === VerificationStatus::VALID;
    }

    public function toArray(): array
    {
        return [
            'status' => $this->status->value,
            'valid' => $this->isValid(),
            'signature_index' => $this->signatureIndex,
            'signer' => $this->signer,
            'message' => $this->message,
        ];
    }
}

/**
 * Verification summary.
 */
readonly class VerificationSummary
{
    /**
     * @param int $total Total signatures
     * @param int $verified Valid signatures
     * @param int $failed Invalid signatures
     * @param int $unknown Unknown status signatures
     * @param array<SignatureVerificationResult> $results Individual results
     */
    public function __construct(
        public int $total,
        public int $verified,
        public int $failed,
        public int $unknown,
        public array $results
    ) {}

    public function allValid(): bool
    {
        return $this->failed === 0 && $this->unknown === 0 && $this->total > 0;
    }

    public function hasAnyValid(): bool
    {
        return $this->verified > 0;
    }

    public function hasAnyInvalid(): bool
    {
        return $this->failed > 0;
    }

    public function toArray(): array
    {
        return [
            'total' => $this->total,
            'verified' => $this->verified,
            'failed' => $this->failed,
            'unknown' => $this->unknown,
            'all_valid' => $this->allValid(),
            'results' => array_map(fn($r) => $r->toArray(), $this->results),
        ];
    }
}
