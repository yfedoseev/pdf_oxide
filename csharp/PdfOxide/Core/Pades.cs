using System;
using System.Collections.Generic;

namespace PdfOxide.Core
{
    /// <summary>
    /// PAdES baseline level (ETSI EN 319 142-1 §5). The integer mapping
    /// (BB=0, BT=1, BLt=2, BLta=3) is frozen and shared with the C ABI
    /// and every binding — never renumber.
    /// </summary>
    public enum PadesLevel
    {
        /// <summary>CAdES-B-B: signed attrs incl. ESS signing-certificate-v2.</summary>
        BB = 0,
        /// <summary>B-B + an RFC 3161 signature-time-stamp unsigned attribute.</summary>
        BT = 1,
        /// <summary>B-T + a Document Security Store (DSS/VRI).</summary>
        BLt = 2,
        /// <summary>Reserved; producing this level is not supported in this release.</summary>
        BLta = 3,
    }

    /// <summary>
    /// Offline B-LT validation material: DER X.509 certificates, CRLs,
    /// and OCSP responses. Mirrors Rust <c>signatures::RevocationMaterial</c>.
    /// </summary>
    public sealed class RevocationMaterial
    {
        /// <summary>DER X.509 certificates (signer + TSA chain).</summary>
        public IList<byte[]> Certificates { get; } = new List<byte[]>();
        /// <summary>DER CRLs.</summary>
        public IList<byte[]> Crls { get; } = new List<byte[]>();
        /// <summary>DER OCSP responses (RFC 6960).</summary>
        public IList<byte[]> OcspResponses { get; } = new List<byte[]>();
    }

    /// <summary>
    /// Options for <see cref="Certificate.SignPdfBytesPades"/>.
    /// <see cref="TsaUrl"/> is required for <see cref="PadesLevel.BT"/>
    /// and <see cref="PadesLevel.BLt"/> (the RFC 3161 source).
    /// </summary>
    public sealed class PadesSignOptions
    {
        /// <summary>The target baseline level.</summary>
        public PadesLevel Level { get; set; } = PadesLevel.BB;
        /// <summary>RFC 3161 TSA URL (required for B-T/B-LT).</summary>
        public string? TsaUrl { get; set; }
        /// <summary>Optional <c>/Reason</c>.</summary>
        public string? Reason { get; set; }
        /// <summary>Optional <c>/Location</c>.</summary>
        public string? Location { get; set; }
        /// <summary>B-LT revocation material (optional).</summary>
        public RevocationMaterial? Revocation { get; set; }
    }

    /// <summary>
    /// A parsed Document Security Store (<c>/DSS</c>,
    /// ISO 32000-2:2020 §12.8.4.3). Document-level DER blobs plus the
    /// count of per-signature <c>/VRI</c> entries.
    /// </summary>
    public sealed class DocumentSecurityStore
    {
        /// <summary>Document-level DER certificates (<c>/Certs</c>).</summary>
        public IReadOnlyList<byte[]> Certificates { get; }
        /// <summary>Document-level DER CRLs (<c>/CRLs</c>).</summary>
        public IReadOnlyList<byte[]> Crls { get; }
        /// <summary>Document-level DER OCSP responses (<c>/OCSPs</c>).</summary>
        public IReadOnlyList<byte[]> OcspResponses { get; }
        /// <summary>Number of per-signature <c>/VRI</c> entries.</summary>
        public int VriCount { get; }

        internal DocumentSecurityStore(
            IReadOnlyList<byte[]> certs,
            IReadOnlyList<byte[]> crls,
            IReadOnlyList<byte[]> ocsps,
            int vriCount)
        {
            Certificates = certs;
            Crls = crls;
            OcspResponses = ocsps;
            VriCount = vriCount;
        }
    }
}
