using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using PdfOxide.Exceptions;
using PdfOxide.Internal;

namespace PdfOxide.Core
{
    /// <summary>
    /// Inspection-only X.509 certificate loaded from raw DER bytes.
    ///
    /// The DER path is functional against the <c>signatures</c> feature.
    /// PKCS#12 (certificate + private key) is not yet supported, so
    /// Load() currently only accepts raw DER-encoded X.509 bytes.
    /// </summary>
    public sealed class Certificate : IDisposable
    {
        private NativeHandle _handle;
        private bool _disposed;

        private Certificate(NativeHandle handle)
        {
            _handle = handle;
        }

        internal static Certificate FromHandle(NativeHandle handle) => new(handle);

        /// <summary>
        /// Load a signing credential from PEM-encoded certificate and private key strings.
        /// Both <paramref name="certPem"/> and <paramref name="keyPem"/> must be PEM text
        /// (-----BEGIN CERTIFICATE----- / -----BEGIN PRIVATE KEY-----).
        /// </summary>
        public static Certificate LoadFromPem(string certPem, string keyPem)
        {
            ArgumentNullException.ThrowIfNull(certPem);
            ArgumentNullException.ThrowIfNull(keyPem);

            var handle = NativeMethods.PdfCertificateLoadFromPem(certPem, keyPem, out int err);
            if (handle == IntPtr.Zero)
            {
                ExceptionMapper.ThrowIfError(err);
                throw new PdfException("pdf_certificate_load_from_pem returned null with no error code");
            }
            return new Certificate(new NativeHandle(handle, NativeMethods.pdf_certificate_free));
        }

        /// <summary>
        /// Load a certificate from raw DER-encoded X.509 bytes. If the
        /// blob is a PKCS#12 (.p12/.pfx), <paramref name="password"/>
        /// should be the matching password; otherwise pass null/empty.
        /// </summary>
        public static Certificate Load(byte[] data, string? password = null)
        {
            ArgumentNullException.ThrowIfNull(data);
            if (data.Length == 0)
                throw new ArgumentException("Certificate data must not be empty.", nameof(data));

            var handle = NativeMethods.pdf_certificate_load_from_bytes(
                data, data.Length, password, out int err);
            if (handle.IsInvalid)
            {
                ExceptionMapper.ThrowIfError(err);
                throw new PdfException("pdf_certificate_load_from_bytes returned null with no error code");
            }
            return new Certificate(handle);
        }

        /// <summary>Certificate subject Distinguished Name (RFC 5280).</summary>
        public string Subject
        {
            get
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.pdf_certificate_get_subject(_handle, out int err);
                ExceptionMapper.ThrowIfError(err);
                if (ptr == IntPtr.Zero) return string.Empty;
                try { return StringMarshaler.PtrToString(ptr); }
                finally { NativeMethods.FreeString(ptr); }
            }
        }

        /// <summary>Certificate issuer Distinguished Name.</summary>
        public string Issuer
        {
            get
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.pdf_certificate_get_issuer(_handle, out int err);
                ExceptionMapper.ThrowIfError(err);
                if (ptr == IntPtr.Zero) return string.Empty;
                try { return StringMarshaler.PtrToString(ptr); }
                finally { NativeMethods.FreeString(ptr); }
            }
        }

        /// <summary>Certificate serial number as a hex string.</summary>
        public string Serial
        {
            get
            {
                ThrowIfDisposed();
                var ptr = NativeMethods.pdf_certificate_get_serial(_handle, out int err);
                ExceptionMapper.ThrowIfError(err);
                if (ptr == IntPtr.Zero) return string.Empty;
                try { return StringMarshaler.PtrToString(ptr); }
                finally { NativeMethods.FreeString(ptr); }
            }
        }

        /// <summary>Certificate validity window as Unix timestamps.</summary>
        public (DateTimeOffset NotBefore, DateTimeOffset NotAfter) Validity
        {
            get
            {
                ThrowIfDisposed();
                NativeMethods.pdf_certificate_get_validity(
                    _handle, out long notBefore, out long notAfter, out int err);
                ExceptionMapper.ThrowIfError(err);
                return (
                    DateTimeOffset.FromUnixTimeSeconds(notBefore),
                    DateTimeOffset.FromUnixTimeSeconds(notAfter));
            }
        }

        /// <summary>
        /// True iff the certificate is within its validity window as
        /// of right now (system clock).
        /// </summary>
        public bool IsValid
        {
            get
            {
                ThrowIfDisposed();
                bool valid = NativeMethods.pdf_certificate_is_valid(_handle, out int err);
                ExceptionMapper.ThrowIfError(err);
                return valid;
            }
        }

        /// <summary>
        /// Applies a CMS/PKCS#7 detached signature to <paramref name="pdfData"/> and returns
        /// the signed PDF bytes. The certificate must have been loaded with a private key
        /// (e.g. via <see cref="LoadFromPem"/>).
        /// </summary>
        /// <param name="pdfData">Raw bytes of the PDF to sign.</param>
        /// <param name="reason">Optional signature reason (e.g. "Approved"). Pass null to omit.</param>
        /// <param name="location">Optional signing location (e.g. "Berlin"). Pass null to omit.</param>
        /// <returns>New byte array containing the signed PDF.</returns>
        public unsafe byte[] SignPdfBytes(byte[] pdfData, string? reason = null, string? location = null)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(pdfData);
            if (pdfData.Length == 0)
                throw new ArgumentException("PDF data must not be empty.", nameof(pdfData));

            IntPtr certPtr = _handle.Ptr;
            fixed (byte* pdfPtr = pdfData)
            {
                byte* outPtr = NativeMethods.PdfSignBytes(
                    pdfPtr, (nuint)pdfData.Length,
                    certPtr,
                    reason, location,
                    out nuint outLen, out int err);
                ExceptionMapper.ThrowIfError(err);
                if (outPtr == null)
                    throw new PdfException("pdf_sign_bytes returned null with no error code");
                try
                {
                    var result = new byte[(int)outLen];
                    Marshal.Copy((IntPtr)outPtr, result, 0, (int)outLen);
                    return result;
                }
                finally
                {
                    NativeMethods.FreeBytes((IntPtr)outPtr);
                }
            }
        }

        /// <summary>
        /// Signs <paramref name="pdfData"/> at a PAdES baseline level
        /// (<see cref="PadesSignOptions.Level"/>) and returns the signed
        /// PDF. <see cref="PadesLevel.BLta"/> is reserved (throws). For
        /// <see cref="PadesLevel.BT"/>/<see cref="PadesLevel.BLt"/> a
        /// <see cref="PadesSignOptions.TsaUrl"/> is required.
        /// </summary>
        public unsafe byte[] SignPdfBytesPades(byte[] pdfData, PadesSignOptions options)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(pdfData);
            ArgumentNullException.ThrowIfNull(options);
            if (pdfData.Length == 0)
                throw new ArgumentException("PDF data must not be empty.", nameof(pdfData));

            var rev = options.Revocation;
            IList<byte[]> certs = rev?.Certificates ?? (IList<byte[]>)Array.Empty<byte[]>();
            IList<byte[]> crls = rev?.Crls ?? (IList<byte[]>)Array.Empty<byte[]>();
            IList<byte[]> ocsps = rev?.OcspResponses ?? (IList<byte[]>)Array.Empty<byte[]>();

            var pins = new List<GCHandle>();
            try
            {
                var (cPtrs, cLens) = PinBlobs(certs, pins);
                var (rPtrs, rLens) = PinBlobs(crls, pins);
                var (oPtrs, oLens) = PinBlobs(ocsps, pins);

                IntPtr certPtr = _handle.Ptr;
                fixed (byte* pdfPtr = pdfData)
                fixed (IntPtr* cP = cPtrs)
                fixed (nuint* cL = cLens)
                fixed (IntPtr* rP = rPtrs)
                fixed (nuint* rL = rLens)
                fixed (IntPtr* oP = oPtrs)
                fixed (nuint* oL = oLens)
                {
                    byte* outPtr = NativeMethods.PdfSignBytesPades(
                        pdfPtr, (nuint)pdfData.Length,
                        certPtr,
                        (int)options.Level,
                        options.TsaUrl, options.Reason, options.Location,
                        (byte**)cP, (nuint*)cL, (nuint)certs.Count,
                        (byte**)rP, (nuint*)rL, (nuint)crls.Count,
                        (byte**)oP, (nuint*)oL, (nuint)ocsps.Count,
                        out nuint outLen, out int err);
                    ExceptionMapper.ThrowIfError(err);
                    if (outPtr == null)
                        throw new PdfException("pdf_sign_bytes_pades returned null with no error code");
                    try
                    {
                        var result = new byte[(int)outLen];
                        Marshal.Copy((IntPtr)outPtr, result, 0, (int)outLen);
                        return result;
                    }
                    finally
                    {
                        NativeMethods.FreeBytes((IntPtr)outPtr);
                    }
                }
            }
            finally
            {
                foreach (var h in pins)
                    h.Free();
            }
        }

        // Pin each DER blob (stable address while pinned) and return
        // parallel pointer/length arrays for the *_pades FFI.
        private static (IntPtr[] ptrs, nuint[] lens) PinBlobs(
            IList<byte[]> blobs, List<GCHandle> pins)
        {
            var ptrs = new IntPtr[blobs.Count];
            var lens = new nuint[blobs.Count];
            for (int i = 0; i < blobs.Count; i++)
            {
                var b = blobs[i] ?? Array.Empty<byte>();
                if (b.Length == 0)
                {
                    ptrs[i] = IntPtr.Zero;
                    lens[i] = 0;
                    continue;
                }
                var h = GCHandle.Alloc(b, GCHandleType.Pinned);
                pins.Add(h);
                ptrs[i] = h.AddrOfPinnedObject();
                lens[i] = (nuint)b.Length;
            }
            return (ptrs, lens);
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

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
