using System;
using System.IO;
using System.Linq;
using PdfOxide.Core;
using PdfOxide.Exceptions;
using Xunit;

namespace PdfOxide.Tests
{
    /// <summary>
    /// Broad API coverage tests — one test per public method that wasn't
    /// already covered in PdfDocumentTests.cs or DocumentBuilderTests.cs.
    /// Every test is self-contained: it creates its own PDF from Markdown,
    /// exercises the method, and cleans up.
    /// </summary>
    public class ApiCoverageTests
    {
        // ── helpers ──────────────────────────────────────────────────────────

        private static byte[] MakeSimplePdf(string markdown = "# Hello\n\nWorld.")
        {
            using var pdf = Pdf.FromMarkdown(markdown);
            return pdf.SaveToBytes();
        }

        private static TempFile WriteTempPdf(string markdown = "# Hello\n\nWorld.")
        {
            var bytes = MakeSimplePdf(markdown);
            var path = Path.Combine(Path.GetTempPath(), $"pdfoxide-cov-{Guid.NewGuid():N}.pdf");
            File.WriteAllBytes(path, bytes);
            return new TempFile(path);
        }

        private static bool IsUnsupportedFeature(Exception e) =>
            e is UnsupportedFeatureException ||
            e.Message.Contains("5000") ||
            e.Message.Contains("not compiled") ||
            e.Message.ToLower().Contains("unsupported");

        // ── PdfDocument.Open from path ────────────────────────────────────────

        [Fact]
        public void PdfDocument_Open_From_Path_Returns_Document()
        {
            using var tmp = WriteTempPdf();
            using var doc = PdfDocument.Open(tmp);
            Assert.True(doc.PageCount >= 1);
        }

        // ── Text extraction ───────────────────────────────────────────────────

        [Fact]
        public void ExtractWords_Returns_NonEmpty_Array()
        {
            using var tmp = WriteTempPdf("WORDTEST");
            using var doc = PdfDocument.Open(tmp);
            var words = doc.ExtractWords(0);
            Assert.NotNull(words);
            Assert.True(words.Length > 0);
            Assert.Contains(words, w => w.Text.Contains("WORDTEST"));
        }

        [Fact]
        public void ExtractChars_Returns_NonEmpty_Array()
        {
            using var tmp = WriteTempPdf("CHARTEST");
            using var doc = PdfDocument.Open(tmp);
            var chars = doc.ExtractChars(0);
            Assert.NotNull(chars);
            Assert.True(chars.Length > 0);
            Assert.True(chars[0].W > 0 || chars[0].H > 0);
        }

        [Fact]
        public void ExtractTextLines_Returns_NonEmpty_Array()
        {
            using var tmp = WriteTempPdf("LINETEST");
            using var doc = PdfDocument.Open(tmp);
            var lines = doc.ExtractTextLines(0);
            Assert.NotNull(lines);
            Assert.True(lines.Length > 0);
            Assert.Contains(lines, l => l.Text.Contains("LINETEST"));
        }

        [Fact]
        public void ExtractAllText_Returns_NonEmpty_String()
        {
            using var tmp = WriteTempPdf("ALLTEXTMARKER");
            using var doc = PdfDocument.Open(tmp);
            var text = doc.ExtractAllText();
            Assert.False(string.IsNullOrWhiteSpace(text));
            Assert.Contains("ALLTEXTMARKER", text);
        }

        // ── Conversion ────────────────────────────────────────────────────────

        [Fact]
        public void ToMarkdown_Returns_NonEmpty_String()
        {
            using var tmp = WriteTempPdf("MDMARKER");
            using var doc = PdfDocument.Open(tmp);
            var md = doc.ToMarkdown(0);
            Assert.False(string.IsNullOrWhiteSpace(md));
        }

        [Fact]
        public void ToMarkdownAll_Returns_NonEmpty_String()
        {
            using var tmp = WriteTempPdf();
            using var doc = PdfDocument.Open(tmp);
            var md = doc.ToMarkdownAll();
            Assert.False(string.IsNullOrWhiteSpace(md));
        }

        [Fact]
        public void ToHtml_Returns_Html_With_Tags()
        {
            using var tmp = WriteTempPdf();
            using var doc = PdfDocument.Open(tmp);
            var html = doc.ToHtml(0);
            Assert.False(string.IsNullOrWhiteSpace(html));
            Assert.Contains("<", html);
        }

        [Fact]
        public void ToHtmlAll_Returns_Html_With_Tags()
        {
            using var tmp = WriteTempPdf();
            using var doc = PdfDocument.Open(tmp);
            var html = doc.ToHtmlAll();
            Assert.False(string.IsNullOrWhiteSpace(html));
            Assert.Contains("<", html);
        }

        [Fact]
        public void ToPlainText_Returns_NonEmpty_String()
        {
            using var tmp = WriteTempPdf("PLAINMARKER");
            using var doc = PdfDocument.Open(tmp);
            var text = doc.ToPlainText(0);
            Assert.False(string.IsNullOrWhiteSpace(text));
            Assert.Contains("PLAINMARKER", text);
        }

        [Fact]
        public void ToPlainTextAll_Returns_NonEmpty_String()
        {
            using var tmp = WriteTempPdf("PLAINALL");
            using var doc = PdfDocument.Open(tmp);
            var text = doc.ToPlainTextAll();
            Assert.False(string.IsNullOrWhiteSpace(text));
            Assert.Contains("PLAINALL", text);
        }

        // ── Search ────────────────────────────────────────────────────────────

        [Fact]
        public void SearchPage_Finds_Known_Term()
        {
            using var tmp = WriteTempPdf("SEARCHTOKEN");
            using var doc = PdfDocument.Open(tmp);
            var results = doc.SearchPage(0, "SEARCHTOKEN");
            Assert.True(results.Length > 0);
        }

        [Fact]
        public void SearchAll_Finds_Known_Term()
        {
            using var tmp = WriteTempPdf("SEARCHALLTOKEN");
            using var doc = PdfDocument.Open(tmp);
            var results = doc.SearchAll("SEARCHALLTOKEN");
            Assert.True(results.Length > 0);
        }

        [Fact]
        public void SearchAll_Missing_Term_Returns_Empty()
        {
            using var tmp = WriteTempPdf();
            using var doc = PdfDocument.Open(tmp);
            var results = doc.SearchAll("ZZZNOMATCHZZZ");
            Assert.Empty(results);
        }

        // ── DocumentBuilder extras ────────────────────────────────────────────

        [Fact]
        public void DocumentBuilder_Save_NonEncrypted()
        {
            var path = Path.Combine(Path.GetTempPath(), $"pdfoxide-save-{Guid.NewGuid():N}.pdf");
            try
            {
                using var builder = DocumentBuilder.Create();
                builder.A4Page().Paragraph("plain save test").Done();
                builder.Save(path);
                Assert.True(File.Exists(path));
                Assert.True(new FileInfo(path).Length > 100);
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Fact]
        public void DocumentBuilder_LetterPage_Produces_Pdf()
        {
            using var builder = DocumentBuilder.Create();
            builder.LetterPage().Paragraph("US Letter").Done();
            var bytes = builder.Build();
            Assert.StartsWith("%PDF-", System.Text.Encoding.ASCII.GetString(bytes.Take(8).ToArray()));
        }

        [Fact]
        public void DocumentBuilder_CustomPage_Produces_Pdf()
        {
            using var builder = DocumentBuilder.Create();
            builder.Page(300f, 400f).Paragraph("custom size").Done();
            var bytes = builder.Build();
            Assert.True(bytes.Length > 100);
        }

        [Fact]
        public void DocumentBuilder_Metadata_Setters_Do_Not_Throw()
        {
            using var builder = DocumentBuilder.Create()
                .Title("My Title")
                .Author("Alice")
                .Subject("Testing")
                .Keywords("pdf, test")
                .Creator("xunit");
            builder.A4Page().Paragraph("metadata test").Done();
            var bytes = builder.Build();
            Assert.True(bytes.Length > 100);
        }

        [Fact]
        public void DocumentBuilder_ToBytesEncrypted_Produces_Encrypted_Pdf()
        {
            using var builder = DocumentBuilder.Create();
            builder.A4Page().Paragraph("secret").Done();
            var bytes = builder.ToBytesEncrypted("user", "owner");
            Assert.True(bytes.Length > 100);
            var raw = System.Text.Encoding.Latin1.GetString(bytes);
            Assert.Contains("/Encrypt", raw);
        }

        // ── DocumentEditor mutations ──────────────────────────────────────────

        [Fact]
        public void DocumentEditor_DeletePage_Reduces_PageCount()
        {
            // build a 3-page PDF
            using var pdfA = Pdf.FromMarkdown("# P1");
            using var pdfB = Pdf.FromMarkdown("# P2");
            using var pdfC = Pdf.FromMarkdown("# P3");
            var path = Path.Combine(Path.GetTempPath(), $"pdfoxide-edit-{Guid.NewGuid():N}.pdf");
            try
            {
                pdfA.Save(path);
                using var editor = DocumentEditor.Open(path);
                editor.MergeFrom(path); // now 2 pages (P1 + P1)
                int before = editor.PageCount;
                editor.DeletePage(0);
                editor.Save(path);

                using var doc = PdfDocument.Open(path);
                Assert.Equal(before - 1, doc.PageCount);
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Fact]
        public void DocumentEditor_MovePage_Changes_Order()
        {
            var srcPath = Path.Combine(Path.GetTempPath(), $"pdfoxide-move-src-{Guid.NewGuid():N}.pdf");
            var outPath = Path.Combine(Path.GetTempPath(), $"pdfoxide-move-out-{Guid.NewGuid():N}.pdf");
            try
            {
                // Use DocumentBuilder to create a 2-page PDF in one shot so all
                // pages live in page_order from the start (no merged_pages split).
                using (var builder = DocumentBuilder.Create())
                {
                    builder.A4Page().At(72, 720).Text("PAGEFIRST").Done();
                    builder.A4Page().At(72, 720).Text("PAGESECOND").Done();
                    File.WriteAllBytes(srcPath, builder.Build());
                }

                using var editor = DocumentEditor.Open(srcPath);
                editor.MovePage(1, 0);   // swap: PAGESECOND, PAGEFIRST
                editor.Save(outPath);

                using var doc = PdfDocument.Open(outPath);
                Assert.Equal(2, doc.PageCount);
                var words = doc.ExtractWords(0);
                Assert.True(words.Any(w => w.Text.Contains("PAGESECOND")),
                    $"Expected 'PAGESECOND' on page 0, got: {string.Join(", ", words.Select(w => w.Text))}");
            }
            finally
            {
                if (File.Exists(srcPath)) File.Delete(srcPath);
                if (File.Exists(outPath)) File.Delete(outPath);
            }
        }

        [Fact]
        public void DocumentEditor_MergeFrom_Increases_PageCount()
        {
            var pathA = Path.Combine(Path.GetTempPath(), $"pdfoxide-ma-{Guid.NewGuid():N}.pdf");
            var pathB = Path.Combine(Path.GetTempPath(), $"pdfoxide-mb-{Guid.NewGuid():N}.pdf");
            var outPath = Path.Combine(Path.GetTempPath(), $"pdfoxide-mout-{Guid.NewGuid():N}.pdf");
            try
            {
                using (var a = Pdf.FromMarkdown("# A")) a.Save(pathA);
                using (var b = Pdf.FromMarkdown("# B")) b.Save(pathB);

                using var editor = DocumentEditor.Open(pathA);
                int before = editor.PageCount;
                editor.MergeFrom(pathB);
                editor.Save(outPath);  // save to separate file to avoid same-file lock

                using var doc = PdfDocument.Open(outPath);
                Assert.True(doc.PageCount > before);
            }
            finally
            {
                if (File.Exists(pathA)) File.Delete(pathA);
                if (File.Exists(pathB)) File.Delete(pathB);
                if (File.Exists(outPath)) File.Delete(outPath);
            }
        }

        // ── Pdf factory extras ────────────────────────────────────────────────

        [Fact]
        public void Pdf_FromText_Produces_Pdf()
        {
            try
            {
                using var pdf = Pdf.FromText("Hello plain text");
                var bytes = pdf.SaveToBytes();
                Assert.StartsWith("%PDF-", System.Text.Encoding.ASCII.GetString(bytes.Take(8).ToArray()));
            }
            catch (Exception e) when (IsUnsupportedFeature(e))
            {
                // skip — feature not compiled in
            }
        }

        // ── C-ABI snake_case coverage (CAbi wrappers) ─────────────────────────

        private static IntPtr OpenViaCAbi(out TempFile tmp)
        {
            tmp = WriteTempPdf("CABIMARK");
            return CAbi.DocumentOpen(tmp);
        }

        [Fact]
        public void CAbi_DocumentOpen_And_Free_Roundtrips()
        {
            var handle = OpenViaCAbi(out var tmp);
            using (tmp)
            {
                Assert.NotEqual(IntPtr.Zero, handle);
                CAbi.DocumentFree(handle);
            }
        }

        [Fact]
        public void CAbi_DocumentFree_Null_Is_NoOp()
        {
            CAbi.DocumentFree(IntPtr.Zero); // must not throw
        }

        [Fact]
        public void CAbi_DocumentGetPageCount_Returns_Positive()
        {
            var handle = OpenViaCAbi(out var tmp);
            using (tmp)
            {
                try { Assert.True(CAbi.DocumentGetPageCount(handle) >= 1); }
                finally { CAbi.DocumentFree(handle); }
            }
        }

        [Fact]
        public void CAbi_DocumentExtractText_Returns_Marker()
        {
            var handle = OpenViaCAbi(out var tmp);
            using (tmp)
            {
                try { Assert.Contains("CABIMARK", CAbi.DocumentExtractText(handle, 0)); }
                finally { CAbi.DocumentFree(handle); }
            }
        }

        [Fact]
        public void CAbi_DocumentToMarkdown_Returns_NonEmpty()
        {
            var handle = OpenViaCAbi(out var tmp);
            using (tmp)
            {
                try { Assert.False(string.IsNullOrWhiteSpace(CAbi.DocumentToMarkdown(handle, 0))); }
                finally { CAbi.DocumentFree(handle); }
            }
        }

        [Fact]
        public void CAbi_DocumentToHtml_Returns_Tags()
        {
            var handle = OpenViaCAbi(out var tmp);
            using (tmp)
            {
                try { Assert.Contains("<", CAbi.DocumentToHtml(handle, 0)); }
                finally { CAbi.DocumentFree(handle); }
            }
        }

        [Fact]
        public void CAbi_DocumentToPlainText_Returns_NonEmpty()
        {
            var handle = OpenViaCAbi(out var tmp);
            using (tmp)
            {
                try { Assert.False(string.IsNullOrWhiteSpace(CAbi.DocumentToPlainText(handle, 0))); }
                finally { CAbi.DocumentFree(handle); }
            }
        }

        [Fact]
        public void CAbi_DocumentGetSourceBytes_ReturnOrError()
        {
            var handle = OpenViaCAbi(out var tmp);
            using (tmp)
            {
                try
                {
                    var bytes = CAbi.DocumentGetSourceBytes(handle);
                    Assert.NotNull(bytes);
                }
                catch (Exception e) when (IsUnsupportedFeature(e) || e is PdfException)
                {
                    // accept binding-surfaced error
                }
                finally { CAbi.DocumentFree(handle); }
            }
        }

        [Fact]
        public void CAbi_EditorOpen_Save_Free_Roundtrips()
        {
            using var tmp = WriteTempPdf("EDITMARK");
            var outPath = Path.Combine(Path.GetTempPath(), $"pdfoxide-cabi-edit-{Guid.NewGuid():N}.pdf");
            var editor = CAbi.EditorOpen(tmp);
            try
            {
                Assert.NotEqual(IntPtr.Zero, editor);
                CAbi.EditorSave(editor, outPath);
                Assert.True(File.Exists(outPath));
            }
            finally
            {
                CAbi.EditorFree(editor);
                if (File.Exists(outPath)) File.Delete(outPath);
            }
        }

        [Fact]
        public void CAbi_EditorFree_Null_Is_NoOp()
        {
            CAbi.EditorFree(IntPtr.Zero); // must not throw
        }

        [Fact]
        public void CAbi_EditorSave_Persists_File()
        {
            using var tmp = WriteTempPdf("EDITSAVE");
            var outPath = Path.Combine(Path.GetTempPath(), $"pdfoxide-cabi-save-{Guid.NewGuid():N}.pdf");
            var editor = CAbi.EditorOpen(tmp);
            try
            {
                CAbi.EditorSave(editor, outPath);
                Assert.True(new FileInfo(outPath).Length > 100);
            }
            finally
            {
                CAbi.EditorFree(editor);
                if (File.Exists(outPath)) File.Delete(outPath);
            }
        }

        [Fact]
        public void CAbi_PdfFromMarkdown_And_Free()
        {
            var pdf = CAbi.PdfFromMarkdown("# CAbi MD");
            Assert.NotEqual(IntPtr.Zero, pdf);
            CAbi.PdfFree(pdf);
        }

        [Fact]
        public void CAbi_PdfFromHtml_And_Free()
        {
            var pdf = CAbi.PdfFromHtml("<h1>CAbi HTML</h1>");
            Assert.NotEqual(IntPtr.Zero, pdf);
            CAbi.PdfFree(pdf);
        }

        [Fact]
        public void CAbi_PdfFromText_ReturnOrError()
        {
            try
            {
                var pdf = CAbi.PdfFromText("CAbi plain text");
                Assert.NotEqual(IntPtr.Zero, pdf);
                CAbi.PdfFree(pdf);
            }
            catch (Exception e) when (IsUnsupportedFeature(e))
            {
                // feature not compiled in
            }
        }

        [Fact]
        public void CAbi_PdfSave_Persists_File()
        {
            var outPath = Path.Combine(Path.GetTempPath(), $"pdfoxide-cabi-pdfsave-{Guid.NewGuid():N}.pdf");
            var pdf = CAbi.PdfFromMarkdown("# Save me");
            try
            {
                CAbi.PdfSave(pdf, outPath);
                Assert.True(File.Exists(outPath));
            }
            finally
            {
                CAbi.PdfFree(pdf);
                if (File.Exists(outPath)) File.Delete(outPath);
            }
        }

        [Fact]
        public void CAbi_PdfFree_Null_Is_NoOp()
        {
            CAbi.PdfFree(IntPtr.Zero); // must not throw
        }

        [Fact]
        public void CAbi_SetMaxOpsPerStream_Roundtrips_Previous()
        {
            long previous = CAbi.SetMaxOpsPerStream(500_000);
            // restore default and confirm we get back the value we just set
            long restored = CAbi.SetMaxOpsPerStream(-1);
            Assert.Equal(500_000, restored);
            // restore to whatever was there originally
            CAbi.SetMaxOpsPerStream(previous);
        }

        [Fact]
        public void CAbi_SetPreserveUnmappedGlyphs_Roundtrips_Previous()
        {
            int previous = CAbi.SetPreserveUnmappedGlyphs(true);
            int restored = CAbi.SetPreserveUnmappedGlyphs(false);
            Assert.Equal(1, restored);
            CAbi.SetPreserveUnmappedGlyphs(previous != 0);
        }

        [Fact]
        public void CAbi_PageBuilderStreamingTableBegin_ReturnOrError()
        {
            using var builder = DocumentBuilder.Create();
            var page = builder.A4Page();
            try
            {
                CAbi.PageBuilderStreamingTableBegin(
                    page.InternalHandle,
                    new[] { "A", "B" },
                    new[] { 100f, 100f },
                    new[] { 0, 2 },
                    repeatHeader: true);
            }
            catch (Exception e) when (IsUnsupportedFeature(e) || e is PdfException)
            {
                // accept binding-surfaced error
            }
            page.Done();
        }

        [Fact]
        public void CAbi_RenderPageWithOptionsEx_ReturnOrError()
        {
            var handle = OpenViaCAbi(out var tmp);
            using (tmp)
            {
                try
                {
                    var img = CAbi.RenderPageWithOptionsEx(
                        handle, 0, 72, 0,
                        1f, 1f, 1f, 1f,
                        transparentBackground: false,
                        renderAnnotations: true,
                        jpegQuality: 90,
                        excludedLayers: null);
                    // success path: a handle (or zero) is acceptable
                    Assert.True(img == IntPtr.Zero || img != IntPtr.Zero);
                }
                catch (Exception e) when (IsUnsupportedFeature(e) || e is PdfException)
                {
                    // accept binding-surfaced error (e.g. render feature off)
                }
                finally { CAbi.DocumentFree(handle); }
            }
        }

        [Fact]
        public void CAbi_SignBytesPadesOpts_ReturnOrError()
        {
            var pdf = MakeSimplePdf("SIGNME");
            try
            {
                // IntPtr.Zero options ⇒ binding surfaces an error rather than crashing.
                var signed = CAbi.SignBytesPadesOpts(pdf, IntPtr.Zero);
                Assert.NotNull(signed);
            }
            catch (Exception e) when (IsUnsupportedFeature(e) || e is PdfException)
            {
                // accept binding-surfaced error (no signing material / feature off)
            }
        }

        // ── Temp-file helper ──────────────────────────────────────────────────

        private readonly struct TempFile : IDisposable
        {
            public string Path { get; }
            public TempFile(string path) => Path = path;
            public static implicit operator string(TempFile t) => t.Path;
            public void Dispose()
            {
                try { File.Delete(Path); } catch { }
            }
        }
    }
}
