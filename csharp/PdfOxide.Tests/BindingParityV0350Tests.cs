using System;
using System.IO;
using PdfOxide.Core;
using Xunit;

namespace PdfOxide.Tests
{
    /// <summary>
    /// v0.3.50 cross-binding API-parity tests: the standalone document
    /// sanitization (#231) and the document-scoped PAdES-B-LTA reader
    /// signal (#235) the other bindings expose must also exist in the
    /// managed surface, plus the frozen PAdES level enum (#235) and the
    /// process-wide crypto-governance readers (#230).
    /// </summary>
    public class BindingParityV0350Tests
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
            finally { File.Delete(path); }
        }

        [Fact]
        public void PadesLevel_FrozenEnumMapping()
        {
            Assert.Equal(0, (int)PadesLevel.BB);
            Assert.Equal(1, (int)PadesLevel.BT);
            Assert.Equal(2, (int)PadesLevel.BLt);
            Assert.Equal(3, (int)PadesLevel.BLta);
        }

        [Fact]
        public void CryptoGovernance_PolicyAndCbom_Callable()
        {
            Assert.False(string.IsNullOrEmpty(PdfDocument.CryptoPolicy()));
            Assert.NotNull(PdfDocument.CryptoInventory());
            Assert.Contains("CycloneDX", PdfDocument.CryptoCbom());
        }
    }
}
