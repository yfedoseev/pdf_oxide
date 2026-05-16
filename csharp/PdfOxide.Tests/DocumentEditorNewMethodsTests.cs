using System;
using System.IO;
using PdfOxide.Core;
using PdfOxide.Exceptions;
using Xunit;

namespace PdfOxide.Tests
{
    /// <summary>
    /// Tests for the DocumentEditor methods added in v0.3.39:
    ///   OpenFromBytes, SaveToBytes, SaveToBytesWithOptions, Keywords,
    ///   MergeFromBytes, EmbedFile, ApplyPageRedactions, ApplyAllRedactions,
    ///   RotateAllPages, RotatePageBy, GetPageMediaBox, SetPageMediaBox,
    ///   GetPageCropBox, SetPageCropBox, EraseRegions, ClearEraseRegions,
    ///   IsPageMarkedForFlatten, UnmarkPageForFlatten,
    ///   IsPageMarkedForRedaction, UnmarkPageForRedaction.
    /// </summary>
    public class DocumentEditorNewMethodsTests
    {
        // ---------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------

        private static string CreateTestPdf(string markdown = "# Test\n\nContent.")
        {
            using var pdf = Pdf.FromMarkdown(markdown);
            var path = Path.Combine(Path.GetTempPath(), $"pdfoxide-ednew-{Guid.NewGuid():N}.pdf");
            pdf.Save(path);
            return path;
        }

        private static byte[] CreateTestPdfBytes(string markdown = "# Test\n\nContent.")
        {
            using var pdf = Pdf.FromMarkdown(markdown);
            return pdf.SaveToBytes();
        }

        private static (DocumentEditor editor, string path) OpenTestEditor(string markdown = "# Test\n\nContent.")
        {
            var path = CreateTestPdf(markdown);
            return (DocumentEditor.Open(path), path);
        }

        // ---------------------------------------------------------------
        // SaveToBytes
        // ---------------------------------------------------------------

        [Fact]
        public void SaveToBytes_ReturnsPdfBytes()
        {
            var path = CreateTestPdf("# SaveToBytes");
            try
            {
                using var editor = DocumentEditor.Open(path);
                var bytes = editor.SaveToBytes();
                Assert.NotNull(bytes);
                Assert.True(bytes.Length > 100);
                Assert.Equal((byte)'%', bytes[0]);
                Assert.Equal((byte)'P', bytes[1]);
                Assert.Equal((byte)'D', bytes[2]);
                Assert.Equal((byte)'F', bytes[3]);
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // SaveToBytesWithOptions
        // ---------------------------------------------------------------

        [Fact]
        public void SaveToBytesWithOptions_ReturnsPdfBytes()
        {
            var path = CreateTestPdf("# SaveToBytesWithOptions");
            try
            {
                using var editor = DocumentEditor.Open(path);
                var bytes = editor.SaveToBytesWithOptions(compress: true, garbageCollect: true, linearize: false);
                Assert.NotNull(bytes);
                Assert.True(bytes.Length > 100);
                Assert.Equal((byte)'%', bytes[0]);
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // OpenFromBytes
        // ---------------------------------------------------------------

        [Fact]
        public void OpenFromBytes_RoundTrip()
        {
            // Create editor from file, save to bytes, re-open from bytes
            var path = CreateTestPdf("# OpenFromBytes round-trip");
            try
            {
                byte[] pdfBytes;
                int pagesBefore;
                using (var editor = DocumentEditor.Open(path))
                {
                    pagesBefore = editor.PageCount;
                    pdfBytes = editor.SaveToBytes();
                }

                using var editor2 = DocumentEditor.OpenFromBytes(pdfBytes);
                Assert.Equal(pagesBefore, editor2.PageCount);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void OpenFromBytes_NullThrows()
        {
            Assert.Throws<ArgumentNullException>(() => DocumentEditor.OpenFromBytes(null!));
        }

        [Fact]
        public void OpenFromBytes_EmptyThrows()
        {
            Assert.Throws<ArgumentException>(() => DocumentEditor.OpenFromBytes(Array.Empty<byte>()));
        }

        [Fact]
        public void OpenFromBytes_InvalidDataThrows()
        {
            var garbage = new byte[] { 0x00, 0x01, 0x02, 0x03 };
            Assert.ThrowsAny<PdfException>(() => DocumentEditor.OpenFromBytes(garbage));
        }

        // ---------------------------------------------------------------
        // Keywords
        // ---------------------------------------------------------------

        [Fact]
        public void Keywords_SetAndGet_RoundTrips()
        {
            var path = CreateTestPdf("# Keywords");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.Keywords = "csharp, test, pdf";
                Assert.Equal("csharp, test, pdf", editor.Keywords);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Keywords_SetEmpty_ClearsValue()
        {
            var path = CreateTestPdf("# Keywords empty");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.Keywords = "initial";
                editor.Keywords = string.Empty;
                var kw = editor.Keywords;
                // Either null or empty string is acceptable after clearing
                Assert.True(kw == null || kw == string.Empty);
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // MergeFromBytes
        // ---------------------------------------------------------------

        [Fact]
        public void MergeFromBytes_IncreasesPageCount()
        {
            var pathA = CreateTestPdf("# A");
            var pathB = CreateTestPdf("# B");
            try
            {
                using var editor = DocumentEditor.Open(pathA);
                int before = editor.PageCount;
                var bytesB = File.ReadAllBytes(pathB);
                int added = editor.MergeFromBytes(bytesB);
                Assert.True(added >= 1, $"Expected at least 1 page added, got {added}");
                Assert.True(editor.PageCount > before);
            }
            finally
            {
                File.Delete(pathA);
                File.Delete(pathB);
            }
        }

        [Fact]
        public void MergeFromBytes_NullThrows()
        {
            var path = CreateTestPdf();
            try
            {
                using var editor = DocumentEditor.Open(path);
                Assert.Throws<ArgumentNullException>(() => editor.MergeFromBytes(null!));
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // EmbedFile
        // ---------------------------------------------------------------

        [Fact]
        public void EmbedFile_DoesNotThrow()
        {
            var path = CreateTestPdf("# EmbedFile");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.EmbedFile("hello.txt", System.Text.Encoding.UTF8.GetBytes("hello embedded world"));
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // ApplyPageRedactions / ApplyAllRedactions
        // ---------------------------------------------------------------

        [Fact]
        public void ApplyPageRedactions_NoRedactionsDoesNotThrow()
        {
            var path = CreateTestPdf("# ApplyPageRedactions");
            try
            {
                using var editor = DocumentEditor.Open(path);
                // No redactions marked — should be a no-op
                editor.ApplyPageRedactions(0);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void ApplyAllRedactions_NoRedactionsDoesNotThrow()
        {
            var path = CreateTestPdf("# ApplyAllRedactions");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.ApplyAllRedactions();
            }
            finally { File.Delete(path); }
        }

        // Destructive redaction (#231): AddRedaction / RedactionCount / ApplyRedactions
        [Fact]
        public void DestructiveRedaction_Parity()
        {
            var path = CreateTestPdf("# Destructive redact parity");
            try
            {
                using var editor = DocumentEditor.Open(path);
                Assert.Equal(0, editor.RedactionCount(0));
                editor.AddRedaction(0, 0, 0, 5000, 5000);
                Assert.Equal(1, editor.RedactionCount(0));
                // Must not throw on a simple-font document.
                editor.ApplyRedactions(scrubMetadata: true);
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // RotateAllPages / RotatePageBy
        // ---------------------------------------------------------------

        [Fact]
        public void RotateAllPages_DoesNotThrow()
        {
            var path = CreateTestPdf("# RotateAllPages");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.RotateAllPages(90);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void RotatePageBy_DoesNotThrow()
        {
            var path = CreateTestPdf("# RotatePageBy");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.RotatePageBy(0, 180);
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // GetPageMediaBox / SetPageMediaBox
        // ---------------------------------------------------------------

        [Fact]
        public void GetPageMediaBox_ReturnsPositiveDimensions()
        {
            var path = CreateTestPdf("# MediaBox");
            try
            {
                using var editor = DocumentEditor.Open(path);
                var box = editor.GetPageMediaBox(0);
                Assert.True(box.Width > 0, $"Expected positive width, got {box.Width}");
                Assert.True(box.Height > 0, $"Expected positive height, got {box.Height}");
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void SetPageMediaBox_DoesNotThrow()
        {
            var path = CreateTestPdf("# SetMediaBox");
            try
            {
                using var editor = DocumentEditor.Open(path);
                var box = editor.GetPageMediaBox(0);
                // Set it back to same values — should be idempotent
                editor.SetPageMediaBox(0, box.X, box.Y, box.Width, box.Height);
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // GetPageCropBox / SetPageCropBox
        // ---------------------------------------------------------------

        [Fact]
        public void GetPageCropBox_DoesNotThrow()
        {
            var path = CreateTestPdf("# CropBox");
            try
            {
                using var editor = DocumentEditor.Open(path);
                // May return (0,0,0,0) if no CropBox set — that's fine
                var box = editor.GetPageCropBox(0);
                Assert.True(box.Width >= 0);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void SetPageCropBox_DoesNotThrow()
        {
            var path = CreateTestPdf("# SetCropBox");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.SetPageCropBox(0, 10, 10, 500, 700);
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // EraseRegions / ClearEraseRegions
        // ---------------------------------------------------------------

        [Fact]
        public void EraseRegions_DoesNotThrow()
        {
            var path = CreateTestPdf("# EraseRegions");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.EraseRegions(0, new double[][]
                {
                    new double[] { 10, 10, 100, 50 },
                    new double[] { 200, 200, 80, 40 },
                });
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void EraseRegions_EmptyArray_DoesNotThrow()
        {
            var path = CreateTestPdf("# EraseRegions empty");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.EraseRegions(0, Array.Empty<double[]>());
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void EraseRegions_ShortInnerArray_Throws()
        {
            var path = CreateTestPdf("# EraseRegions short");
            try
            {
                using var editor = DocumentEditor.Open(path);
                Assert.Throws<ArgumentException>(() =>
                    editor.EraseRegions(0, new double[][] { new double[] { 10, 10 } }));
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void ClearEraseRegions_DoesNotThrow()
        {
            var path = CreateTestPdf("# ClearEraseRegions");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.ClearEraseRegions(0);
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // IsPageMarkedForFlatten / UnmarkPageForFlatten
        // ---------------------------------------------------------------

        [Fact]
        public void IsPageMarkedForFlatten_DefaultFalse()
        {
            var path = CreateTestPdf("# FlattenMark");
            try
            {
                using var editor = DocumentEditor.Open(path);
                Assert.False(editor.IsPageMarkedForFlatten(0));
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void UnmarkPageForFlatten_DoesNotThrow()
        {
            var path = CreateTestPdf("# UnmarkFlatten");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.UnmarkPageForFlatten(0);
            }
            finally { File.Delete(path); }
        }

        // ---------------------------------------------------------------
        // IsPageMarkedForRedaction / UnmarkPageForRedaction
        // ---------------------------------------------------------------

        [Fact]
        public void IsPageMarkedForRedaction_DefaultFalse()
        {
            var path = CreateTestPdf("# RedactionMark");
            try
            {
                using var editor = DocumentEditor.Open(path);
                Assert.False(editor.IsPageMarkedForRedaction(0));
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void UnmarkPageForRedaction_DoesNotThrow()
        {
            var path = CreateTestPdf("# UnmarkRedaction");
            try
            {
                using var editor = DocumentEditor.Open(path);
                editor.UnmarkPageForRedaction(0);
            }
            finally { File.Delete(path); }
        }
    }
}
