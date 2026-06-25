using System;
using System.IO;
using PdfOxide.Core;
using PdfOxide.Exceptions;
using Xunit;

namespace PdfOxide.Tests
{
    /// <summary>
    /// Cross-binding API-parity tests: the standalone document
    /// sanitization and the document-scoped PAdES-B-LTA reader signal
    /// the other bindings expose must also exist in the managed
    /// surface, plus the frozen PAdES level enum and the process-wide
    /// crypto-governance readers.
    ///
    /// The native-touching cases swallow <see cref="UnsupportedFeatureException"/>
    /// so they no-op on the bare-features regression-guard build (the
    /// lib compiled without the optional feature), matching the
    /// established pattern in CertificateTests / PadesTests.
    /// </summary>
    public class BindingParitySanitizeSignTests
    {
        private static string CreateTestPdf(string markdown = "# Parity\n\nConfidential body.")
        {
            using var pdf = Pdf.FromMarkdown(markdown);
            var path = Path.Combine(Path.GetTempPath(), $"pdfoxide-parity-{Guid.NewGuid():N}.pdf");
            pdf.Save(path);
            return path;
        }

        [Fact]
        public void DocumentEditor_SanitizeDocument_RunsAndRewrites()
        {
            var path = CreateTestPdf();
            try
            {
                using var editor = DocumentEditor.Open(path);
                int removed = editor.SanitizeDocument();
                Assert.True(removed >= 0);
                var bytes = editor.SaveToBytes();
                Assert.True(bytes.Length > 50);
                Assert.Equal((byte)'%', bytes[0]);
            }
            catch (UnsupportedFeatureException)
            {
                // bare-features build: sanitize unavailable — skip.
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void PdfDocument_HasDocumentTimestamp_FalseForPlainPdf()
        {
            var path = CreateTestPdf("# LTA probe\n\nplain");
            try
            {
                using var doc = PdfDocument.Open(path);
                Assert.False(doc.HasDocumentTimestamp());
            }
            catch (UnsupportedFeatureException)
            {
                // bare-features build: signatures off, B-LTA reader
                // returns the unsupported sentinel — skip.
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void PadesLevel_FrozenEnumMapping()
        {
            // Pure managed enum — no native call, always asserts.
            Assert.Equal(0, (int)PadesLevel.BB);
            Assert.Equal(1, (int)PadesLevel.BT);
            Assert.Equal(2, (int)PadesLevel.BLt);
            Assert.Equal(3, (int)PadesLevel.BLta);
        }

        [Fact]
        public void CryptoGovernance_PolicyAndCbom_Callable()
        {
            try
            {
                Assert.False(string.IsNullOrEmpty(PdfDocument.CryptoPolicy()));
                Assert.NotNull(PdfDocument.CryptoInventory());
                Assert.Contains("CycloneDX", PdfDocument.CryptoCbom());
            }
            catch (UnsupportedFeatureException)
            {
                // bare-features build: governance surface unavailable — skip.
            }
        }
    }
}
