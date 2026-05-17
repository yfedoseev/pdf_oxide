using System;
using System.IO;
using PdfOxide.Core;
using PdfOxide.Exceptions;
using Xunit;

namespace PdfOxide.Tests
{
    /// <summary>
    /// Cross-binding mirror of the Rust <c>tests/pades_ltv.rs</c> /
    /// <c>signatures::sign_bytes</c> PAdES suite (#235). Covers the
    /// frozen <see cref="PadesLevel"/> ABI contract and the B-B signing
    /// round-trip.
    ///
    /// When the native library is built without the <c>signatures</c>
    /// feature (or is absent) the signing calls throw and the tests
    /// pass vacuously — same discipline as <see cref="SignatureTests"/>.
    /// </summary>
    public class PadesTests
    {
        private const string CertPem = "../../../../../tests/fixtures/test_signing_cert.pem";
        private const string KeyPem = "../../../../../tests/fixtures/test_signing_key.pem";
        private const string Pdf = "../../../../../tests/fixtures/simple.pdf";

        // The integer mapping is part of the stable C ABI and every
        // binding depends on it — never renumber.
        [Fact]
        public void PadesLevel_IntegerMapping_IsFrozen()
        {
            Assert.Equal(0, (int)PadesLevel.BB);
            Assert.Equal(1, (int)PadesLevel.BT);
            Assert.Equal(2, (int)PadesLevel.BLt);
            Assert.Equal(3, (int)PadesLevel.BLta);
        }

        private static Certificate? TryLoadCert()
        {
            try
            {
                return Certificate.LoadFromPem(
                    File.ReadAllText(CertPem), File.ReadAllText(KeyPem));
            }
            catch (UnsupportedFeatureException)
            {
                return null; // signatures feature not compiled in
            }
        }

        [Fact]
        public void SignPdfBytesPades_BB_ProducesSignedPdf()
        {
            using var cert = TryLoadCert();
            if (cert is null)
                return;

            byte[] pdf = File.ReadAllBytes(Pdf);
            byte[] signed;
            try
            {
                signed = cert.SignPdfBytesPades(
                    pdf, new PadesSignOptions { Level = PadesLevel.BB });
            }
            catch (UnsupportedFeatureException)
            {
                return;
            }

            Assert.NotNull(signed);
            Assert.True(signed.Length > pdf.Length, "signed PDF must be larger");
            // The incremental update appends a signature dictionary.
            Assert.Contains("/Type /Sig", System.Text.Encoding.Latin1.GetString(signed));
        }

        [Fact]
        public void SignPdfBytesPades_BT_WithoutTsa_FailsClosed()
        {
            using var cert = TryLoadCert();
            if (cert is null)
                return;

            byte[] pdf = File.ReadAllBytes(Pdf);
            try
            {
                // B-T without a TsaUrl must fail closed, never silently
                // produce a B-B file.
                Assert.ThrowsAny<Exception>(() =>
                    cert.SignPdfBytesPades(
                        pdf, new PadesSignOptions { Level = PadesLevel.BT }));
            }
            catch (UnsupportedFeatureException)
            {
                // signatures feature absent — vacuous pass
            }
        }

        [Fact]
        public void SignPdfBytesPades_BLta_IsUnsupported()
        {
            using var cert = TryLoadCert();
            if (cert is null)
                return;

            byte[] pdf = File.ReadAllBytes(Pdf);
            try
            {
                Assert.ThrowsAny<Exception>(() =>
                    cert.SignPdfBytesPades(
                        pdf, new PadesSignOptions { Level = PadesLevel.BLta }));
            }
            catch (UnsupportedFeatureException)
            {
                // signatures feature absent — vacuous pass
            }
        }

        [Fact]
        public void GetDss_UnsignedPdf_ReturnsNull()
        {
            using var doc = PdfDocument.Open(Pdf);
            try
            {
                Assert.Null(doc.GetDss());
            }
            catch (UnsupportedFeatureException)
            {
                // signatures feature absent — vacuous pass
            }
        }
    }
}
