using PdfOxide.Core;
using Xunit;

namespace PdfOxide.Tests
{
    /// <summary>
    /// Locks in the PDF/A, PDF/X, and PDF/UA wire-format integer
    /// mapping against the cdylib's documented C ABI (
    /// <c>src/ffi.rs:1225, :5538, :7412</c>). Every pdf_oxide binding
    /// (C#, Java, Ruby, PHP, Go, Node.js) must send the SAME integer
    /// for the SAME compliance level — any future re-numbering
    /// surfaces here as a hard test failure rather than as a
    /// silently-wrong validation verdict.
    ///
    /// Companion to:
    ///   * <c>java/src/test/.../compliance/PdfLevelWireFormatTest.java</c>
    ///   * <c>php/tests/Unit/PdfValidatorLevelMappingTest.php</c>
    ///   * <c>ruby/spec/ffi_signature_regression_spec.rb</c>
    ///
    /// C# is the gold standard for this mapping (the underlying enum
    /// values have always been correct — see the documented
    /// "u/gevorgter Reddit regression" history in
    /// <c>ExceptionMapperTests.cs</c>). These tests exist to KEEP
    /// them correct: a future contributor renaming or renumbering
    /// PdfALevel without realising it's a C ABI surface would break
    /// every other binding silently.
    /// </summary>
    public class PdfLevelWireFormatTests
    {
        [Fact]
        public void PdfALevel_IntegerEncoding_MirrorsCdylibAbi()
        {
            // src/ffi.rs:1225 — `0=A1b 1=A1a 2=A2b 3=A2a 4=A2u
            //                    5=A3b 6=A3a 7=A3u`.
            // B comes before A within each level — the cdylib contract,
            // not an alphabetical choice.
            Assert.Equal(0, (int)PdfALevel.A1b);
            Assert.Equal(1, (int)PdfALevel.A1a);
            Assert.Equal(2, (int)PdfALevel.A2b);
            Assert.Equal(3, (int)PdfALevel.A2a);
            Assert.Equal(4, (int)PdfALevel.A2u);
            Assert.Equal(5, (int)PdfALevel.A3b);
            Assert.Equal(6, (int)PdfALevel.A3a);
            Assert.Equal(7, (int)PdfALevel.A3u);
        }

        [Fact]
        public void PdfUaLevel_IntegerEncoding_MirrorsCdylibAbi()
        {
            // src/ffi.rs:5538 — `level == 2 → UA-2, else UA-1`.
            // 1-indexed, not 0-indexed.
            Assert.Equal(1, (int)PdfUaLevel.Ua1);
            Assert.Equal(2, (int)PdfUaLevel.Ua2);
        }

        [Fact]
        public void PdfXLevel_IntegerEncoding_MirrorsCdylibAbi()
        {
            // src/ffi.rs:7412 — `0=X1a:2001, 1=X3:2002, 2=X4`.
            Assert.Equal(0, (int)PdfXLevel.X1a);
            Assert.Equal(1, (int)PdfXLevel.X3);
            Assert.Equal(2, (int)PdfXLevel.X4);
        }
    }
}
