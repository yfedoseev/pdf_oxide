using System;
using PdfOxide.Core;
using PdfOxide.Exceptions;
using Xunit;

namespace PdfOxide.Tests
{
    /// <summary>
    /// Cross-binding API-parity tests: the OCR model provisioning trio
    /// the other bindings expose must also exist in the managed
    /// surface. Network-free — only the air-gapped manifest is asserted
    /// (no downloads; those belong to the model-gated Rust lane).
    ///
    /// The native-touching cases swallow <see cref="UnsupportedFeatureException"/>
    /// so they no-op on the bare-features regression-guard build,
    /// matching the established pattern in BindingParitySanitizeSignTests.
    /// </summary>
    public class BindingParityOcrModelsTests
    {
        [Fact]
        public void ModelManifest_ListsDetectorAndEnglish()
        {
            try
            {
                var manifest = PdfDocument.ModelManifest();
                Assert.False(string.IsNullOrEmpty(manifest));
                Assert.Contains("det.onnx", manifest);
                Assert.Contains("english", manifest);
            }
            catch (UnsupportedFeatureException)
            {
                // bare-features build: provisioning surface unavailable — skip.
            }
        }

        [Fact]
        public void PrefetchAvailable_IsCallable()
        {
            try
            {
                // Pure feature probe (no I/O); just exercise the call
                // path and signature.
                _ = PdfDocument.PrefetchAvailable();
            }
            catch (UnsupportedFeatureException)
            {
                // bare-features build: provisioning surface unavailable — skip.
            }
        }
    }
}
