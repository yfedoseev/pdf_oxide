using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using PdfOxide.Core;
using PdfOxide.Exceptions;
using Xunit;

namespace PdfOxide.Tests
{
    /// <summary>
    /// Regression tests for issue #481: concurrent access to a single PdfDocument
    /// must not corrupt internal state or crash the process.
    /// </summary>
    public class ThreadSafetyTests
    {
        private static PdfDocument CreateTestDoc()
        {
            using var pdf = Pdf.FromMarkdown("# Thread Safety\n\nPage 1.\n\n---\n\nPage 2.\n\n---\n\nPage 3.");
            var bytes = pdf.SaveToBytes();
            return PdfDocument.Open(bytes);
        }

        [Fact]
        public void ExtractText_ParallelForEach_DoesNotThrow()
        {
            using var doc = CreateTestDoc();
            int pages = doc.PageCount;
            var exceptions = new ConcurrentBag<Exception>();

            Parallel.For(0, pages * 4, i =>
            {
                try { _ = doc.ExtractText(i % pages); }
                catch (Exception ex) { exceptions.Add(ex); }
            });

            Assert.Empty(exceptions);
        }

        [Fact]
        public void ToMarkdown_ParallelForEach_DoesNotThrow()
        {
            using var doc = CreateTestDoc();
            int pages = doc.PageCount;
            var exceptions = new ConcurrentBag<Exception>();

            Parallel.For(0, pages * 4, i =>
            {
                try { _ = doc.ToMarkdown(i % pages); }
                catch (Exception ex) { exceptions.Add(ex); }
            });

            Assert.Empty(exceptions);
        }

        [Fact]
        public void RenderPage_ParallelForEach_DoesNotThrow()
        {
            using var doc = CreateTestDoc();
            int pages = doc.PageCount;
            var exceptions = new ConcurrentBag<Exception>();

            Parallel.For(0, pages * 4, i =>
            {
                try { _ = doc.RenderPage(i % pages); }
                catch (UnsupportedFeatureException) { /* rendering not compiled in — skip */ }
                catch (Exception ex) { exceptions.Add(ex); }
            });

            Assert.Empty(exceptions);
        }

        [Fact]
        public void RenderPageFit_ParallelForEach_DoesNotThrow()
        {
            using var doc = CreateTestDoc();
            int pages = doc.PageCount;
            var exceptions = new ConcurrentBag<Exception>();

            Parallel.For(0, pages * 4, i =>
            {
                try { _ = doc.RenderPageFit(i % pages, 800, 1200); }
                catch (UnsupportedFeatureException) { /* rendering not compiled in — skip */ }
                catch (Exception ex) { exceptions.Add(ex); }
            });

            Assert.Empty(exceptions);
        }

        [Fact]
        public async Task MixedReadWrite_Concurrent_DoesNotThrow()
        {
            // Interleave reads (ExtractText) with mutations (RemoveArtifacts) from parallel threads.
            using var doc = CreateTestDoc();
            int pages = doc.PageCount;
            var exceptions = new ConcurrentBag<Exception>();

            var readers = Task.Run(() =>
                Parallel.For(0, pages * 8, i =>
                {
                    try { _ = doc.ExtractText(i % pages); }
                    catch (Exception ex) { exceptions.Add(ex); }
                }));

            var writers = Task.Run(() =>
            {
                for (int i = 0; i < 4; i++)
                {
                    try { _ = doc.RemoveArtifacts(); }
                    catch (Exception ex) { exceptions.Add(ex); }
                }
            });

            await Task.WhenAll(readers, writers);
            Assert.Empty(exceptions);
        }

        [Fact]
        public void PageCount_ParallelReads_ReturnsConsistentValue()
        {
            using var doc = CreateTestDoc();
            int expected = doc.PageCount;
            var results = new ConcurrentBag<int>();

            Parallel.For(0, 32, _ => results.Add(doc.PageCount));

            Assert.All(results, v => Assert.Equal(expected, v));
        }
    }
}
