using System;
using System.IO;
using PdfOxide.Core;
using PdfOxide.Exceptions;
using Xunit;

namespace PdfOxide.Tests
{
    /// <summary>
    /// Tests for <see cref="PdfDocument.RenderPage(int, RenderOptions)"/>,
    /// the full RenderOptions surface (DPI, background, annotations, JPEG
    /// quality) for the C# layer. Mirrors the Python surface.
    ///
    /// Tests that exercise the native render path are guarded: when the
    /// native library is compiled without the <c>rendering</c> feature (e.g.
    /// the bare-features CI job or an end-user build that omits it) the call
    /// throws <see cref="UnsupportedFeatureException"/> and the test passes
    /// vacuously. Argument-validation tests do not need guards because C#
    /// validates inputs before touching native code.
    /// </summary>
    public class RenderOptionsTests
    {
        private static PdfDocument CreateTestDoc()
        {
            using var pdf = Pdf.FromMarkdown("# Render me\n\nBody.");
            var bytes = pdf.SaveToBytes();
            return PdfDocument.Open(bytes);
        }

        private static bool IsPng(byte[] b) =>
            b.Length >= 8 && b[0] == 0x89 && b[1] == 0x50 && b[2] == 0x4E && b[3] == 0x47;

        private static bool IsJpeg(byte[] b) =>
            b.Length >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF;

        // Returns true if rendering is available, false (and discards bytes) if not.
        private static bool TryRender(PdfDocument doc, RenderOptions opts, out byte[] bytes)
        {
            try
            {
                bytes = doc.RenderPage(0, opts);
                return true;
            }
            catch (UnsupportedFeatureException)
            {
                bytes = Array.Empty<byte>();
                return false; // native lib compiled without rendering — skip assertion
            }
        }

        [Fact]
        public void RenderPage_WithOptions_DefaultIsPng()
        {
            using var doc = CreateTestDoc();
            if (!TryRender(doc, new RenderOptions(), out var bytes)) return;
            Assert.True(IsPng(bytes));
        }

        [Fact]
        public void RenderPage_WithOptions_JpegFormat_EmitsJpegMagic()
        {
            using var doc = CreateTestDoc();
            if (!TryRender(doc, new RenderOptions { Format = RenderImageFormat.Jpeg }, out var bytes)) return;
            Assert.True(IsJpeg(bytes));
        }

        [Fact]
        public void RenderPage_WithOptions_HigherDpiProducesBiggerOutput()
        {
            using var doc = CreateTestDoc();
            if (!TryRender(doc, new RenderOptions { Dpi = 72 }, out var small)) return;
            if (!TryRender(doc, new RenderOptions { Dpi = 300 }, out var large)) return;
            Assert.True(IsPng(small) && IsPng(large));
            Assert.True(large.Length > small.Length);
        }

        [Fact]
        public void RenderPage_WithOptions_LowerJpegQualityIsSmaller()
        {
            using var doc = CreateTestDoc();
            if (!TryRender(doc, new RenderOptions { Format = RenderImageFormat.Jpeg, JpegQuality = 20 }, out var low)) return;
            if (!TryRender(doc, new RenderOptions { Format = RenderImageFormat.Jpeg, JpegQuality = 95 }, out var high)) return;
            Assert.True(IsJpeg(low) && IsJpeg(high));
            Assert.True(low.Length <= high.Length);
        }

        [Fact]
        public void RenderPage_WithOptions_TransparentBackground_OK()
        {
            using var doc = CreateTestDoc();
            if (!TryRender(doc, new RenderOptions { TransparentBackground = true }, out var bytes)) return;
            Assert.True(IsPng(bytes));
        }

        [Fact]
        public void RenderPage_WithOptions_CustomBackground_OK()
        {
            using var doc = CreateTestDoc();
            if (!TryRender(doc, new RenderOptions { Background = (0f, 0f, 0f, 1f) }, out var bytes)) return;
            Assert.True(IsPng(bytes));
        }

        [Fact]
        public void RenderPage_WithOptions_RenderAnnotationsFalse_OK()
        {
            using var doc = CreateTestDoc();
            if (!TryRender(doc, new RenderOptions { RenderAnnotations = false }, out var bytes)) return;
            Assert.True(IsPng(bytes));
        }

        // Argument-validation tests: C# throws before reaching native code,
        // so no feature guard is needed.

        [Fact]
        public void RenderPage_WithOptions_Null_Throws()
        {
            using var doc = CreateTestDoc();
            Assert.Throws<ArgumentNullException>(() => doc.RenderPage(0, (RenderOptions)null!));
        }

        [Fact]
        public void RenderPage_WithOptions_InvalidDpi_Throws()
        {
            using var doc = CreateTestDoc();
            Assert.Throws<ArgumentException>(() =>
                doc.RenderPage(0, new RenderOptions { Dpi = 0 }));
        }

        [Fact]
        public void RenderPage_WithOptions_InvalidJpegQuality_Throws()
        {
            using var doc = CreateTestDoc();
            Assert.Throws<ArgumentException>(() =>
                doc.RenderPage(0, new RenderOptions
                {
                    Format = RenderImageFormat.Jpeg,
                    JpegQuality = 0,
                }));
        }

        // Helper for RGBA tests — returns false if rendering feature not compiled in.
        private static bool TryRenderRgba(PdfDocument doc, out RgbaPixmap px)
        {
            try
            {
                px = doc.RenderToRgba(0, 72);
                return true;
            }
            catch (UnsupportedFeatureException)
            {
                px = new RgbaPixmap(ReadOnlyMemory<byte>.Empty, 0, 0);
                return false;
            }
        }

        [Fact]
        public void RenderToRgba_SizeMatchesWidthTimesHeightTimes4()
        {
            using var doc = CreateTestDoc();
            if (!TryRenderRgba(doc, out var px)) return;
            Assert.True(px.Width > 0);
            Assert.True(px.Height > 0);
            Assert.Equal(px.Width * px.Height * 4, px.Data.Length);
        }

        [Fact]
        public void RenderToRgba_NotPngMagicBytes()
        {
            using var doc = CreateTestDoc();
            if (!TryRenderRgba(doc, out var px)) return;
            var span = px.Data.Span;
            Assert.True(span.Length >= 4);
            // PNG magic is 0x89 0x50 0x4E 0x47 — raw RGBA must not start with this
            Assert.False(span[0] == 0x89 && span[1] == 0x50 && span[2] == 0x4E && span[3] == 0x47,
                "RenderToRgba returned PNG-encoded data instead of raw RGBA pixels");
        }

        [Fact]
        public void RenderToRgba_DimensionsMatchPngRender()
        {
            using var doc = CreateTestDoc();
            if (!TryRender(doc, new RenderOptions { Dpi = 72 }, out _)) return;
            if (!TryRenderRgba(doc, out var px)) return;

            // Decode PNG dimensions for comparison
            var pngBytes = doc.RenderPage(0, new RenderOptions { Dpi = 72 });
            // PNG IHDR starts at byte 16; width at 16-19, height at 20-23 (big-endian)
            int pngW = (pngBytes[16] << 24) | (pngBytes[17] << 16) | (pngBytes[18] << 8) | pngBytes[19];
            int pngH = (pngBytes[20] << 24) | (pngBytes[21] << 16) | (pngBytes[22] << 8) | pngBytes[23];

            Assert.Equal(pngW, px.Width);
            Assert.Equal(pngH, px.Height);
        }

        [Fact]
        public void RenderToRgba_InvalidDpi_Throws()
        {
            using var doc = CreateTestDoc();
            Assert.Throws<ArgumentOutOfRangeException>(() => doc.RenderToRgba(0, 0));
        }
    }
}
