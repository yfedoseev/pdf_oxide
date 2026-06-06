using System;
using System.Text;
using PdfOxide.Core;
using PdfOxide.Exceptions;
using Xunit;

namespace PdfOxide.Tests
{
    /// <summary>
    /// Core functional test-parity suite (C#) — mirrors the shared
    /// cross-language spec
    /// (<c>docs/releases/plans/v0.3.61/core-test-parity-spec.md</c>) with the
    /// idiomatic .NET API. Every binding asserts the same behaviors.
    ///
    /// Each test is self-contained: it builds its own input via
    /// <see cref="Pdf.FromText"/> and opens it from bytes, so the suite has no
    /// fixture-file dependency.
    /// </summary>
    public class CoreParityTests
    {
        private static byte[] BuildBytes()
        {
            using var pdf = Pdf.FromText("Core parity across all bindings.\nSecond line of text.");
            return pdf.SaveToBytes();
        }

        private static PdfDocument Open() => PdfDocument.Open(BuildBytes());

        [Fact]
        public void OpenAndPageCount()
        {
            using var doc = Open();
            Assert.True(doc.PageCount >= 1);
        }

        [Fact]
        public void ExtractTextReturnsString()
        {
            using var doc = Open();
            Assert.NotNull(doc.ExtractText(0));
        }

        [Fact]
        public void ConvertMarkdownHtmlPlain()
        {
            using var doc = Open();
            Assert.NotNull(doc.ToMarkdown(0));
            Assert.NotNull(doc.ToHtml(0));
            Assert.NotNull(doc.ToPlainText(0));
        }

        [Fact]
        public void SearchReturnsResults()
        {
            using var doc = Open();
            Assert.NotNull(doc.SearchAll("parity"));
        }

        [Fact]
        public void StructuredExtraction()
        {
            using var doc = Open();
            Assert.NotNull(doc.ExtractStructured(0));
        }

        [Fact]
        public void CreatePdfFromText()
        {
            var bytes = BuildBytes();
            Assert.True(bytes.Length > 4);
            Assert.Equal("%PDF-", Encoding.ASCII.GetString(bytes, 0, 5));
        }

        [Fact]
        public void OpenFromBytesPageCount()
        {
            using var doc = PdfDocument.Open(BuildBytes());
            Assert.True(doc.PageCount >= 1);
        }

        [Fact]
        public void OpeningMissingPathThrows()
        {
            Assert.ThrowsAny<PdfException>(() => PdfDocument.Open("/no/such/file/does/not/exist.pdf"));
        }

        [Fact]
        public void ExposesVersion()
        {
            using var doc = Open();
            var (major, _) = doc.Version;
            Assert.True(major >= 1);
        }
    }
}
