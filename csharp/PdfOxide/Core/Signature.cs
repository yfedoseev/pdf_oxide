using System;
using PdfOxide.Exceptions;
using PdfOxide.Internal;

namespace PdfOxide.Core
{
    /// <summary>
    /// A single existing digital signature on a PDF document, obtained
    /// via <see cref="PdfDocument.Signatures"/>.
    ///
    /// <see cref="Verify"/> runs the Rust-core RSA-PKCS#1 v1.5
    /// signer-attribute crypto check for SHA-1/256/384/512 signers;
    /// RSA-PSS and ECDSA still surface as
    /// <see cref="UnsupportedFeatureException"/> until their verifiers
    /// land. The <c>messageDigest</c> attribute vs document-bytes
    /// check is not yet performed on this path.
    /// </summary>
    public sealed class Signature : IDisposable
    {
        private NativeHandle _handle;
        private bool _disposed;

        private Signature(NativeHandle handle)
        {
            _handle = handle;
        }

        internal static Signature FromHandle(NativeHandle handle) => new(handle);

        /// <summary>
        /// The <c>/Name</c> of the signer as recorded in the signature
        /// dictionary, or <c>null</c> if the PDF left that field blank.
        /// </summary>
        public string? SignerName => ReadOptionalString(NativeMethods.pdf_signature_get_signer_name);

        /// <summary>
        /// The stated reason for signing (<c>/Reason</c>), or <c>null</c>
        /// if not supplied.
        /// </summary>
        public string? Reason => ReadOptionalString(NativeMethods.pdf_signature_get_signing_reason);

        /// <summary>
        /// The stated signing location (<c>/Location</c>), or <c>null</c>
        /// if not supplied.
        /// </summary>
        public string? Location => ReadOptionalString(NativeMethods.pdf_signature_get_signing_location);

        /// <summary>
        /// The parsed signing time (<c>/M</c>), or <c>null</c> if the
        /// dictionary had no <c>/M</c> entry or it was unparseable.
        /// </summary>
        public DateTimeOffset? SigningTime
        {
            get
            {
                ThrowIfDisposed();
                var epoch = NativeMethods.pdf_signature_get_signing_time(_handle, out int err);
                ExceptionMapper.ThrowIfError(err);
                return epoch == 0 ? null : DateTimeOffset.FromUnixTimeSeconds(epoch);
            }
        }

        /// <summary>
        /// Extract the signing certificate from the embedded
        /// PKCS#7/CMS SignedData blob. Parses the `/Contents` bytes
        /// using the Rust-core CMS helper and returns a live
        /// <see cref="Certificate"/> the caller owns and must dispose.
        /// </summary>
        /// <exception cref="PdfException">The signature dictionary had no <c>/Contents</c> entry
        /// or the bytes didn't parse as CMS SignedData.</exception>
        public Certificate GetCertificate()
        {
            ThrowIfDisposed();
            var certHandle = NativeMethods.pdf_signature_get_certificate(_handle, out int err);
            ExceptionMapper.ThrowIfError(err);
            if (certHandle.IsInvalid)
            {
                throw new PdfException("pdf_signature_get_certificate returned null with no error code");
            }
            return Certificate.FromHandle(certHandle);
        }

        /// <summary>
        /// Run the RFC 5652 §5.4 signer-attributes crypto check against
        /// the embedded signer certificate and return whether it
        /// succeeded. Today this covers RSA-PKCS#1 v1.5 over SHA-1 /
        /// SHA-256 / SHA-384 / SHA-512 — the padding used by
        /// ~every PDF signature in the wild.
        ///
        /// <para>
        /// A <c>true</c> result proves the signer held the private key
        /// matching the embedded certificate and that the signed-attribute
        /// bundle has not been tampered with. It does <b>not</b>
        /// verify the <c>messageDigest</c> attribute against the raw
        /// byte-range content of the PDF — call
        /// <see cref="VerifyDetached"/> for that end-to-end check.
        /// </para>
        /// </summary>
        /// <returns><c>true</c> if the RSA-PKCS#1 v1.5 check succeeded;
        /// <c>false</c> if it failed (wrong key or tampered attributes).</returns>
        /// <exception cref="UnsupportedFeatureException">The signer
        /// uses RSA-PSS, ECDSA, an unknown digest OID, or the CMS
        /// structure lacks the signed attributes required for
        /// verification.</exception>
        public bool Verify()
        {
            ThrowIfDisposed();
            var result = NativeMethods.pdf_signature_verify(_handle, out int err);
            ExceptionMapper.ThrowIfError(err);
            return result == 1;
        }

        /// <summary>
        /// The PAdES baseline level from this signature's CMS attributes
        /// alone (<see cref="PadesLevel.BB"/> vs <see cref="PadesLevel.BT"/>).
        /// <see cref="PadesLevel.BLt"/> additionally requires the
        /// document <c>/DSS</c> — read it via
        /// <see cref="PdfDocument.GetDss"/> and combine there.
        /// </summary>
        public PadesLevel PadesLevel
        {
            get
            {
                ThrowIfDisposed();
                int code = NativeMethods.pdf_signature_get_pades_level(_handle, out int err);
                ExceptionMapper.ThrowIfError(err);
                return (PadesLevel)code;
            }
        }

        /// <summary>
        /// End-to-end detached-signature verification: runs the
        /// signer-attributes RSA-PKCS#1 v1.5 crypto check AND the
        /// RFC 5652 §11.2 <c>messageDigest</c> attribute check against
        /// the bytes this signature protects (pulled out of
        /// <paramref name="pdfData"/> using the signature's
        /// <c>/ByteRange</c>).
        ///
        /// <para>
        /// <c>pdfData</c> must be the full PDF file. A <c>true</c>
        /// result means the signer is authentic AND the document
        /// bytes under the signature's ByteRange have not been
        /// altered since signing. A <c>false</c> result means either
        /// the signer check failed (wrong key / tampered attributes)
        /// or the document bytes were modified.
        /// </para>
        /// </summary>
        /// <param name="pdfData">The full PDF file bytes.</param>
        /// <returns><c>true</c> if both checks succeeded, <c>false</c>
        /// if either failed.</returns>
        /// <exception cref="UnsupportedFeatureException">Signer uses
        /// RSA-PSS, ECDSA, an unknown digest, or the CMS blob lacks
        /// <c>signed_attrs</c>/<c>messageDigest</c>.</exception>
        public bool VerifyDetached(ReadOnlySpan<byte> pdfData)
        {
            ThrowIfDisposed();
            if (pdfData.IsEmpty)
                throw new ArgumentException("pdfData must contain the full PDF bytes; got an empty span.", nameof(pdfData));
            int result;
            int err;
            unsafe
            {
                fixed (byte* pdfPtr = pdfData)
                {
                    result = NativeMethods.pdf_signature_verify_detached(
                        _handle,
                        pdfPtr,
                        (nuint)pdfData.Length,
                        out err);
                }
            }
            ExceptionMapper.ThrowIfError(err);
            return result == 1;
        }

        /// <inheritdoc />
        public void Dispose()
        {
            if (!_disposed)
            {
                _handle?.Dispose();
                _disposed = true;
            }
        }

        private delegate IntPtr NativeStringAccessor(NativeHandle handle, out int errorCode);

        private string? ReadOptionalString(NativeStringAccessor accessor)
        {
            ThrowIfDisposed();
            var ptr = accessor(_handle, out int err);
            ExceptionMapper.ThrowIfError(err);
            if (ptr == IntPtr.Zero) return null;
            try { return StringMarshaler.PtrToString(ptr); }
            finally { NativeMethods.FreeString(ptr); }
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
