//! Press-accurate OutputIntent CMYK ICC integration tests.
//!
//! Builds synthetic PDFs that declare an `/OutputIntents` array with a
//! CMYK `DestOutputProfile`, renders them through the composite path,
//! and pins that the resulting RGB values come from the qcms-driven
//! ICC conversion rather than the §10.3.5 additive-clamp fallback.
//!
//! The minimal CMYK ICC profile used here is synthesised in-test (see
//! `build_minimal_cmyk_to_rgb_lut8_profile` and the README in
//! `tests/fixtures/icc/`). It maps every CMYK input to a constant
//! `RGB(128, 128, 128)` so the pin is unambiguous: an OutputIntent-
//! driven render gives ~128 grey; an additive-clamp fallback gives the
//! §10.3.5 value for the input CMYK.

#![cfg(all(feature = "rendering", feature = "icc"))]
// Probe set grows across commits; the no-OutputIntent baseline
// builder lands ahead of its consumer.
#![allow(dead_code)]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};

// ===========================================================================
// QA round-1 tracking constants
// ===========================================================================
//
// Probes that lock behaviour the foundation does not yet ship are gated on
// `#[ignore = OUTPUT_INTENT_DEFER_*]` so a future engineer running the
// suite sees the open question by name instead of by silence. Each
// constant names the open question and the plan phase that will close
// it.
//
// Convention matches the wave-QA suites' `WAVE-DEFER-*` style so a
// `grep -RI 'OUTPUT_INTENT_DEFER_'` across the worktree pulls every pin
// that is currently on ice.

/// Caching of `Transform::new_srgb_target` calls. Each `k` / `K` operator
/// rebuilds the qcms transform today; the plan defers this to phase 7.
const OUTPUT_INTENT_DEFER_PHASE_7_CACHING: &str =
    "OUTPUT_INTENT_DEFER_PHASE_7_CACHING: plan phase 7 will cache compiled qcms transforms; \
     until then per-paint transform construction is the baseline";

/// Form XObject inheritance of /OutputIntents was originally tagged
/// against the same marker as /DefaultCMYK (round 1); the placeholder
/// probe lives below and stays ignored until a Form-XObject fixture
/// helper lands.
const OUTPUT_INTENT_DEFER_FORM_XOBJECT_INHERITANCE: &str =
    "OUTPUT_INTENT_DEFER_FORM_XOBJECT_INHERITANCE: needs a Form XObject test-fixture helper \
     before the inheritance probe can be wired";

// ===========================================================================
// Minimal CMYK ICC profile synthesis
// ===========================================================================
//
// ICC profile structure (per ICC.1:2004-10 §7 for v2; ICC.1:2010 §7
// for v4 — the layout is identical, only the version byte at offset
// 8..12 differs):
//   - 128-byte header
//   - 4-byte tag count
//   - tag table: N × 12 bytes (signature, offset, size)
//   - tag data: each section 4-byte aligned
//
// Minimum tags qcms's CMYK→RGB transform path needs:
//   - A2B0 (mft1 LUT8 type): CMYK→PCS lookup
// qcms reads the LUT8 (entry-size 1, fixed 256-entry input/output tables)
// per ICC.1 §10.8. Layout inside the LUT8 tag data:
//   bytes 0..4    type signature 'mft1' (0x6d667431)
//   bytes 4..8    reserved zero
//   bytes 8       input channels (4 for CMYK)
//   bytes 9       output channels (3 for RGB)
//   bytes 10      grid points per dimension
//   bytes 11      padding
//   bytes 12..48  9 × s15Fixed16 matrix entries (identity for CMYK)
//   bytes 48..    input tables (input_channels × 256 bytes)
//   then          CLUT (grid_points^input_channels × output_channels bytes)
//   then          output tables (output_channels × 256 bytes)

/// Build a minimal valid ICC v2 CMYK→Lab profile whose A2B0 LUT8 maps
/// every CMYK input to a fixed Lab tuple. The PCS is `Lab ` rather
/// than `XYZ ` because qcms's Lab→XYZ→sRGB chain decodes the 8-bit
/// LUT8 outputs as `L = byte/255*100`, `a = byte - 128`, `b = byte -
/// 128` — easier to point at "neutral grey" than to compute the
/// matching XYZ tuple and round it into a LUT8 byte.
///
/// The constant CLUT makes the test pin unambiguous: whichever CMYK
/// quadruple the renderer feeds the profile, the qcms-converted RGB
/// is the same near-neutral grey that Lab(target_L, 0, 0) projects to
/// through sRGB. That's distinct from the §10.3.5 additive-clamp
/// value for any non-degenerate CMYK input, so a fallback to
/// additive-clamp is immediately visible.
///
/// `target_l_byte` is the LUT8 byte for the L* channel — e.g. 135 ≈
/// L*53, which projects through sRGB to roughly mid-grey
/// `RGB(~128, ~128, ~128)`. a* and b* are pinned at 128 (decoded as
/// 0, the achromatic axis).
fn build_minimal_cmyk_to_rgb_lut8_profile(target_l_byte: u8) -> Vec<u8> {
    build_minimal_cmyk_to_rgb_lut8_profile_with_version(target_l_byte, IccProfileVersion::V2)
}

/// ICC profile header version byte (bytes 8..12 of the 128-byte header).
///
/// ICC.1:2004-10 §7.2.3 (Table 14): the first byte is the major
/// revision, the second the minor, bytes 10..12 are reserved (must be
/// zero). qcms 0.3.0's `check_profile_version` (iccread.rs:274) reads
/// the reserved bytes and rejects anything non-zero, but the version
/// comparison itself is commented out — both v2 (0x02400000) and v4
/// (0x04000000) profile headers parse provided the tag-data the qcms
/// CMM consumes is itself well-formed.
///
/// LUT8 (`mft1`) tag bodies are an ICC v2-era construct. ICC v4
/// introduces the `mAB ` tag form; qcms parses both for the `A2B0`
/// transform-direction tag, so a v4-versioned profile whose A2B0 body
/// is still an mft1 LUT8 is parseable end-to-end. A true v4 profile
/// with mAB tag bodies needs richer tag construction (curve sets,
/// matrices, a CLUT) — synthesising one in-test for the constant-CLUT
/// fixture trick gains nothing the version-byte flip already proves;
/// the LUT8 body is intent-invariant whether the header advertises v2
/// or v4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IccProfileVersion {
    /// 2.4.0.0 — the version qcms 0.3.0's iccread treats as the LUT8
    /// path.
    V2,
    /// 4.0.0.0 — the modern major revision. qcms 0.3.0 accepts the
    /// header (no major-revision check) and reads whatever A2B0 tag
    /// body is present; a v4 header with an mft1 LUT8 body is a
    /// legitimate forward-compatible encoding.
    V4,
}

impl IccProfileVersion {
    fn header_bytes(self) -> [u8; 4] {
        // ICC.1:2004-10 §7.2.3: major.minor with bytes 10..12 reserved
        // (must be zero per the spec; qcms enforces this in
        // check_profile_version).
        match self {
            // 2.4.0.0
            Self::V2 => 0x0240_0000u32.to_be_bytes(),
            // 4.0.0.0 — the modern major. qcms 0.3.0's
            // check_profile_version (iccread.rs:281-288) has the
            // major-revision check commented out with the comment
            // "Checking the version doesn't buy us anything"; only the
            // reserved bytes are validated.
            Self::V4 => 0x0400_0000u32.to_be_bytes(),
        }
    }
}

/// Like [`build_minimal_cmyk_to_rgb_lut8_profile`] but with an
/// explicit ICC version-byte choice in the 128-byte header. Used by
/// the phase-6 ICC v4 verification probe to assert qcms 0.3.0 accepts
/// the v4 header at parse time and drives the same constant-CLUT body
/// to the same byte-exact RGB reference.
fn build_minimal_cmyk_to_rgb_lut8_profile_with_version(
    target_l_byte: u8,
    version: IccProfileVersion,
) -> Vec<u8> {
    // LUT8 tag body for in=4 out=3 grid=2.
    // Sizes:
    //   header: 48
    //   input tables: 4 * 256 = 1024
    //   CLUT: 2^4 * 3 = 48
    //   output tables: 3 * 256 = 768
    //   total: 1888 bytes
    let in_chan: u8 = 4;
    let out_chan: u8 = 3;
    let grid: u8 = 2;
    let mut lut = Vec::with_capacity(1888);

    // Type signature 'mft1'.
    lut.extend_from_slice(&0x6d66_7431u32.to_be_bytes());
    // Reserved.
    lut.extend_from_slice(&0u32.to_be_bytes());
    lut.push(in_chan);
    lut.push(out_chan);
    lut.push(grid);
    lut.push(0); // padding

    // 9 × s15Fixed16 matrix entries (identity matrix). qcms reads these
    // off the LUT8 tag header at offsets 12..48 even for CMYK inputs;
    // they only matter for RGB inputs but qcms still parses them.
    // Identity matrix: 1.0 along diagonal.
    let identity: [i32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x0001_0000];
    for v in identity {
        lut.extend_from_slice(&(v as u32).to_be_bytes());
    }

    // Input tables — identity 0..255 for each of 4 input channels.
    for _ in 0..in_chan {
        for i in 0..256u16 {
            lut.push(i as u8);
        }
    }

    // CLUT: 2^4 × 3 = 16 grid points × 3 output channels.
    // Every grid point outputs Lab(target_L, 0, 0) — neutral grey at the
    // requested lightness. qcms decodes LUT8 outputs through the chain
    //   L = byte/255 * 100
    //   a = byte - 128
    //   b = byte - 128
    // so target_l_byte directly controls L*; a* and b* are pinned at
    // 128 (decoded as the achromatic axis 0).
    let grid_size = (grid as usize).pow(in_chan as u32);
    for _ in 0..grid_size {
        lut.push(target_l_byte);
        lut.push(128);
        lut.push(128);
    }

    // Output tables — identity 0..255 for each of 3 output channels.
    for _ in 0..out_chan {
        for i in 0..256u16 {
            lut.push(i as u8);
        }
    }

    debug_assert_eq!(lut.len(), 1888, "LUT8 body size mismatch");

    // ICC profile envelope: 128-byte header + tag table + tag data.
    // Total profile size: 128 (header) + 4 (count) + 12 (one tag entry)
    // + 1888 (A2B0 data) = 2032 bytes, with the A2B0 data starting at
    // offset 144.
    let mut profile = vec![0u8; 128];
    let total_size: u32 = 128 + 4 + 12 + lut.len() as u32;

    // Profile size at bytes 0..4.
    profile[0..4].copy_from_slice(&total_size.to_be_bytes());
    // Preferred CMM at bytes 4..8 — left zero (no preference).
    // Profile version at bytes 8..12. The version byte is determined
    // by `version` so the phase-6 v4 probe can flip just this field
    // while keeping the same constant-CLUT LUT8 body.
    profile[8..12].copy_from_slice(&version.header_bytes());
    // Device class: 'prtr' (output device).
    profile[12..16].copy_from_slice(b"prtr");
    // Colour space: 'CMYK'.
    profile[16..20].copy_from_slice(b"CMYK");
    // PCS: 'Lab ' — qcms's LABtoXYZ stage gives us a straightforward
    // mapping from "byte in CLUT" to "near-neutral grey at L*≈53".
    profile[20..24].copy_from_slice(b"Lab ");
    // Creation date (12 bytes) at 24..36 — all-zero.
    // Profile signature 'acsp' at 36..40.
    profile[36..40].copy_from_slice(b"acsp");
    // Primary platform at 40..44 — zero.
    // Flags / device manufacturer / model / attributes — all zero through
    // byte 100. Rendering intent at 64..68 (0 = perceptual).
    profile[64..68].copy_from_slice(&0u32.to_be_bytes());
    // Illuminant XYZ at 68..80 — D50 (0.9642, 1.0, 0.8249).
    profile[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes()); // X 0.9642
    profile[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes()); // Y 1.0
    profile[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes()); // Z 0.8249
                                                                    // Creator at 80..84 — zero.

    // Tag table: count = 1, then one entry (signature, offset, size).
    profile.extend_from_slice(&1u32.to_be_bytes());
    profile.extend_from_slice(&0x4132_4230u32.to_be_bytes()); // 'A2B0'
    profile.extend_from_slice(&144u32.to_be_bytes()); // offset
    profile.extend_from_slice(&(lut.len() as u32).to_be_bytes()); // size

    // A2B0 tag data.
    profile.extend_from_slice(&lut);

    profile
}

/// Build a minimal valid ICC v2 RGB→Lab profile whose A2B0 LUT8 maps
/// every RGB input to a fixed Lab tuple. Mirrors the CMYK builder's
/// constant-CLUT trick at `in_chan=3` so the qcms reference is a
/// stable point: whichever RGB the renderer feeds through the profile,
/// the output is the same near-neutral grey at L*=target_l_byte/255*100.
///
/// Used by the Phase 9 `/DefaultRGB` precedence probe: the override
/// declared as `[/ICCBased <stream>]` against bare /DeviceRGB paint
/// routes the components through this profile; the rendered pixel
/// matches the qcms reference RGB regardless of the painted RGB value.
///
/// qcms 0.3.0's LUT8 parser at `iccread.rs:760` accepts `in_chan ∈ {3,
/// 4}` only; this builder targets the `in_chan=3` slot. The synthesised
/// device class is `prtr` and the PCS is `Lab ` so the same Lab→sRGB
/// decoder runs for both CMYK and RGB fixture profiles — only the
/// input-channel count and CLUT-grid power differ.
fn build_minimal_rgb_to_lab_lut8_profile(target_l_byte: u8) -> Vec<u8> {
    // LUT8 tag body for in=3 out=3 grid=2.
    // Sizes:
    //   header: 48
    //   input tables: 3 * 256 = 768
    //   CLUT: 2^3 * 3 = 24
    //   output tables: 3 * 256 = 768
    //   total: 1608 bytes
    let in_chan: u8 = 3;
    let out_chan: u8 = 3;
    let grid: u8 = 2;
    let mut lut = Vec::with_capacity(1608);

    // Type signature 'mft1'.
    lut.extend_from_slice(&0x6d66_7431u32.to_be_bytes());
    // Reserved.
    lut.extend_from_slice(&0u32.to_be_bytes());
    lut.push(in_chan);
    lut.push(out_chan);
    lut.push(grid);
    lut.push(0); // padding

    // 9 × s15Fixed16 matrix entries (identity matrix). For RGB inputs
    // qcms applies this matrix to the raw input components before the
    // CLUT lookup; identity preserves them.
    let identity: [i32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x0001_0000];
    for v in identity {
        lut.extend_from_slice(&(v as u32).to_be_bytes());
    }

    // Input tables — identity 0..255 for each of 3 input channels.
    for _ in 0..in_chan {
        for i in 0..256u16 {
            lut.push(i as u8);
        }
    }

    // CLUT: 2^3 = 8 grid points × 3 output channels. Every grid point
    // outputs Lab(target_L, 0, 0) — the constant-CLUT trick that makes
    // the qcms reference unambiguous.
    let grid_size = (grid as usize).pow(in_chan as u32);
    for _ in 0..grid_size {
        lut.push(target_l_byte);
        lut.push(128);
        lut.push(128);
    }

    // Output tables — identity 0..255 for each of 3 output channels.
    for _ in 0..out_chan {
        for i in 0..256u16 {
            lut.push(i as u8);
        }
    }

    debug_assert_eq!(lut.len(), 1608, "RGB LUT8 body size mismatch");

    // ICC profile envelope: 128-byte header + 4 (count) + 12 (one tag
    // entry) + 1608 (A2B0 data) = 1752 bytes.
    let mut profile = vec![0u8; 128];
    let total_size: u32 = 128 + 4 + 12 + lut.len() as u32;
    profile[0..4].copy_from_slice(&total_size.to_be_bytes());
    profile[8..12].copy_from_slice(&IccProfileVersion::V2.header_bytes());
    profile[12..16].copy_from_slice(b"prtr");
    // Colour space: 'RGB ' — three-channel input.
    profile[16..20].copy_from_slice(b"RGB ");
    // PCS: 'Lab ' — same Lab→sRGB decoder as the CMYK fixture.
    profile[20..24].copy_from_slice(b"Lab ");
    profile[36..40].copy_from_slice(b"acsp");
    profile[64..68].copy_from_slice(&0u32.to_be_bytes());
    profile[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes()); // X 0.9642
    profile[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes()); // Y 1.0
    profile[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes()); // Z 0.8249

    profile.extend_from_slice(&1u32.to_be_bytes());
    profile.extend_from_slice(&0x4132_4230u32.to_be_bytes()); // 'A2B0'
    profile.extend_from_slice(&144u32.to_be_bytes()); // offset
    profile.extend_from_slice(&(lut.len() as u32).to_be_bytes()); // size
    profile.extend_from_slice(&lut);
    profile
}

/// Build a minimal valid ICC v2 Gray TRC profile.
///
/// Unlike the LUT8-based CMYK and RGB fixtures, Gray ICC profiles use
/// the simpler `kTRC` (Tone Reproduction Curve) tag — a single curve
/// mapping the device byte into the linear PCS. qcms 0.3.0's
/// `iccread.rs:1712-1714` reads only the `kTRC` tag for GRAY-signed
/// profiles; no A2B0 / B2A0 / colorant / matrix tags are needed.
///
/// The curve emitted here is a 256-entry `curv` table that linearly
/// ramps from 0 to 65535, corresponding to a gamma of 1.0 (the linear
/// identity). qcms's gray transform path then drives the destination
/// sRGB profile's output_gamma_lut_{r,g,b}: a linear input byte
/// becomes a linear PCS-Y value (`device/255`), which the sRGB
/// inverse-gamma encoding then converts to a perceptual sRGB byte.
/// The result is the canonical sRGB encoding of the linear gray:
/// `byte → sRGB_inv_gamma(byte/255) → sRGB byte`.
///
/// Using gamma 1.0 keeps the profile honest (a deliberate, not
/// accidental, identity in linear space) and produces a distinctive
/// reference value through the sRGB encoder that's nowhere near the
/// raw input byte for mid-tones — making a no-ICC fall-through
/// failure mode immediately visible.
fn build_minimal_gray_trc_profile() -> Vec<u8> {
    // ICC v2 `curveType` tag body shape (ICC.1:2004-10 §10.5):
    //   bytes 0..4   type signature 'curv' (0x63757276)
    //   bytes 4..8   reserved zero
    //   bytes 8..12  count (number of entries)
    //   bytes 12..   count × u16 entries (big-endian)
    //
    // 256-entry linear ramp 0..65535 — qcms reads this as the input
    // gamma table for the gray channel.
    let entry_count: u32 = 256;
    let mut curv = Vec::with_capacity(12 + (entry_count as usize) * 2);
    curv.extend_from_slice(&0x6375_7276u32.to_be_bytes()); // 'curv'
    curv.extend_from_slice(&0u32.to_be_bytes()); // reserved
    curv.extend_from_slice(&entry_count.to_be_bytes());
    for i in 0..entry_count {
        // Linear ramp: 0 → 0, 255 → 65535. This matches the encoding
        // qcms's `lut_interp_linear` expects (the table is sampled
        // linearly across [0, 1] and the entry value is treated as a
        // u16 in the linear PCS-Y representation).
        let v = ((i * 65535) / (entry_count - 1)) as u16;
        curv.extend_from_slice(&v.to_be_bytes());
    }

    // Envelope: 128-byte header + 4 (tag count) + 12 (one tag entry) +
    // curveType body. Tag data offset = 144.
    let mut profile = vec![0u8; 128];
    let total_size: u32 = 128 + 4 + 12 + curv.len() as u32;
    profile[0..4].copy_from_slice(&total_size.to_be_bytes());
    profile[8..12].copy_from_slice(&IccProfileVersion::V2.header_bytes());
    // Display device profile — qcms accepts mntr/scnr/prtr/spac for
    // the colour-space-profile arm that GRAY signatures take. mntr is
    // the most common shape for Gray ICC.
    profile[12..16].copy_from_slice(b"mntr");
    // Colour space: 'GRAY' — single channel input.
    profile[16..20].copy_from_slice(b"GRAY");
    // PCS: 'XYZ ' — qcms's gray pipeline expects an XYZ PCS for the
    // linear PCS-Y interpretation of the curve.
    profile[20..24].copy_from_slice(b"XYZ ");
    profile[36..40].copy_from_slice(b"acsp");
    // Illuminant XYZ at 68..80 — D50 (0.9642, 1.0, 0.8249).
    profile[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes());
    profile[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    profile[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes());

    profile.extend_from_slice(&1u32.to_be_bytes()); // tag count = 1
    profile.extend_from_slice(&0x6b54_5243u32.to_be_bytes()); // 'kTRC'
    profile.extend_from_slice(&144u32.to_be_bytes()); // offset
    profile.extend_from_slice(&(curv.len() as u32).to_be_bytes()); // size
    profile.extend_from_slice(&curv);
    profile
}

// ===========================================================================
// PDF construction helpers
// ===========================================================================

/// Build a one-page PDF with the given catalog entries and content
/// stream. The catalog entries string is spliced into the catalog
/// dictionary so callers can add `/OutputIntents [...]` without
/// reconstructing the whole envelope.
///
/// MediaBox is fixed at `[0 0 100 100]`; rendering at 72 DPI gives a
/// 100×100 pixel canvas so callers can pin pixels at known offsets.
fn build_pdf_with_catalog_entries_and_content(
    catalog_entries: &str,
    content_ops: &str,
    icc_profile_bytes: Option<&[u8]>,
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    let catalog =
        format!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R {} >>\nendobj\n", catalog_entries);
    buf.extend_from_slice(catalog.as_bytes());

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> /Contents 4 0 R >>\nendobj\n",
    );

    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content_ops.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content_ops.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let icc_off;
    let obj_count;
    if let Some(icc) = icc_profile_bytes {
        icc_off = buf.len();
        let icc_hdr = format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", icc.len());
        buf.extend_from_slice(icc_hdr.as_bytes());
        buf.extend_from_slice(icc);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
        obj_count = 6;
    } else {
        icc_off = 0;
        obj_count = 5;
    }

    let xref_off = buf.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    if icc_profile_bytes.is_some() {
        buf.extend_from_slice(format!("{:010} 00000 n \n", icc_off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// Build a PDF whose page paints CMYK(0.25, 0, 0, 0) into a 60×60
/// rect centred on the canvas and whose catalog declares
/// `/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX
/// /OutputCondition (Synthetic CMYK) /DestOutputProfile 5 0 R >>]`.
fn build_pdf_cmyk_with_output_intent(icc_profile_bytes: &[u8]) -> Vec<u8> {
    let catalog_entries = "/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK) /DestOutputProfile 5 0 R >>]";
    // PDF user space is bottom-left origin; the rect at (20, 20, 60, 60)
    // covers the canvas centre.
    let content_ops = "0.25 0 0 0 k\n20 20 60 60 re\nf\n";
    build_pdf_with_catalog_entries_and_content(
        catalog_entries,
        content_ops,
        Some(icc_profile_bytes),
    )
}

/// Same paint operator as `build_pdf_cmyk_with_output_intent` but with
/// no `/OutputIntents` in the catalog. Pins the §10.3.5 fallback.
fn build_pdf_cmyk_without_output_intent() -> Vec<u8> {
    let content_ops = "0.25 0 0 0 k\n20 20 60 60 re\nf\n";
    build_pdf_with_catalog_entries_and_content("", content_ops, None)
}

/// Build a PDF whose page paints a `/Separation` colour space (with a
/// Type-4 PostScript tint transform that produces CMYK(0, tint, 0, 0))
/// against a document-level `/OutputIntents` CMYK profile.
///
/// Object layout:
///   1 — Catalog (with /OutputIntents → 5 0 R)
///   2 — Pages
///   3 — Page (with Resources /ColorSpace /CS1 →
///       [/Separation /MagentaSpot /DeviceCMYK 6 0 R])
///   4 — Content stream
///   5 — OutputIntent profile stream
///   6 — Tint-transform Type-4 stream
///
/// The Type-4 program `{ 0.0 exch 0.0 0.0 }` lifts the input tint into
/// the M position so the alternate-space output is CMYK(0, tint, 0, 0).
fn build_pdf_separation_type4_devicecmyk_with_output_intent(
    output_intent_profile: &[u8],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    let catalog = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK) /DestOutputProfile 5 0 R >>] >>\nendobj\n";
    buf.extend_from_slice(catalog.as_bytes());

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    let page = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /CS1 [/Separation /MagentaSpot /DeviceCMYK 6 0 R] >> >> /Contents 4 0 R >>\nendobj\n";
    buf.extend_from_slice(page.as_bytes());

    let stream_off = buf.len();
    // Activate the Separation colour space and paint the rect with full
    // tint (1.0). With the Type-4 program below, the tint transform
    // produces CMYK(0, 1, 0, 0).
    let content = "/CS1 cs\n1.0 scn\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let icc_off = buf.len();
    let icc_hdr = format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", output_intent_profile.len());
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(output_intent_profile);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let tint_off = buf.len();
    // Type 4 PostScript tint transform. Stack semantics per the
    // resolver-side test in src/rendering/resolution/color.rs:697:
    // `{ 0.0 exch 0.0 0.0 }` consumes input tint and leaves the stack
    // bottom-to-top as [0, tint, 0, 0] — i.e. CMYK output (C=0, M=tint,
    // Y=0, K=0). Domain [0 1] is the input range; Range [0 1 0 1 0 1 0 1]
    // is the four-component CMYK output range.
    let tint_program: &[u8] = b"{ 0.0 exch 0.0 0.0 }";
    let tint_hdr = format!(
        "6 0 obj\n<< /FunctionType 4 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\nstream\n",
        tint_program.len()
    );
    buf.extend_from_slice(tint_hdr.as_bytes());
    buf.extend_from_slice(tint_program);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    let obj_count = 7;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, icc_off, tint_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// Same shape as `build_pdf_separation_type4_devicecmyk_with_output_intent`
/// but the colour space is a 2-colorant `/DeviceN` whose alternate is
/// `/DeviceCMYK` and whose Type-4 tint transform consumes the two input
/// tints and emits CMYK(0, tint0, 0, 0) — i.e. only the first input
/// drives the magenta component, the second is dropped. With content
/// `[1.0 0.5] scn` the input is (tint0=1.0, tint1=0.5) and the output is
/// CMYK(0, 1, 0, 0).
fn build_pdf_devicen_type4_devicecmyk_with_output_intent(output_intent_profile: &[u8]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    let catalog = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK) /DestOutputProfile 5 0 R >>] >>\nendobj\n";
    buf.extend_from_slice(catalog.as_bytes());

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    // DeviceN colorant array: two named spot inks. The tint-transform
    // function is referenced by indirect object.
    let page = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /CS1 [/DeviceN [/Magenta /Cyan] /DeviceCMYK 6 0 R] >> >> /Contents 4 0 R >>\nendobj\n";
    buf.extend_from_slice(page.as_bytes());

    let stream_off = buf.len();
    // Activate the DeviceN colour space and paint with two component
    // tints (1.0, 0.5). The Type-4 tint transform drops the second tint
    // and emits CMYK(0, tint0, 0, 0) = CMYK(0, 1, 0, 0).
    let content = "/CS1 cs\n1.0 0.5 scn\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let icc_off = buf.len();
    let icc_hdr = format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", output_intent_profile.len());
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(output_intent_profile);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let tint_off = buf.len();
    // Type 4 program with two inputs (the two DeviceN colorant tints).
    // Stack on entry: [tint0, tint1]. Program: `{ pop 0.0 exch 0.0 0.0 }`
    // pops tint1, then `0.0 exch 0.0 0.0` leaves stack bottom-to-top as
    // [0, tint0, 0, 0] (C=0, M=tint0, Y=0, K=0).
    let tint_program: &[u8] = b"{ pop 0.0 exch 0.0 0.0 }";
    let tint_hdr = format!(
        "6 0 obj\n<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\nstream\n",
        tint_program.len()
    );
    buf.extend_from_slice(tint_hdr.as_bytes());
    buf.extend_from_slice(tint_program);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    let obj_count = 7;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, icc_off, tint_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// Build a PDF that declares BOTH an `/OutputIntents` CMYK profile A and
/// a page-resources `/ColorSpace /CS1 [/ICCBased <stream>]` colour space
/// whose embedded N=4 profile B is a DIFFERENT minimal CMYK profile. The
/// content stream sets fill colour space to `/CS1` and paints with
/// `0.25 0 0 0 scn`.
///
/// Object layout:
///   1 — Catalog (with /OutputIntents → 5 0 R)
///   2 — Pages
///   3 — Page (with Resources /ColorSpace /CS1 → ICCBased referencing 6 0 R)
///   4 — Content stream
///   5 — OutputIntent profile A stream
///   6 — ICCBased embedded profile B stream
fn build_pdf_embedded_iccbased_with_different_output_intent(
    output_intent_profile_a: &[u8],
    embedded_iccbased_profile_b: &[u8],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    let catalog = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK A) /DestOutputProfile 5 0 R >>] >>\nendobj\n";
    buf.extend_from_slice(catalog.as_bytes());

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    // Resources declare an `ICCBased` colour space CS1 whose stream is
    // object 6 — the alternate profile B. Painting `0.25 0 0 0 scn`
    // against CS1 feeds the four components into the embedded profile.
    let page = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /CS1 [/ICCBased 6 0 R] >> >> /Contents 4 0 R >>\nendobj\n";
    buf.extend_from_slice(page.as_bytes());

    let stream_off = buf.len();
    // Set fill colour space to CS1, then paint a 60×60 rect at the centre
    // with the four CMYK components via `scn`. The integer-form fill
    // operator `cs` selects the named colour space.
    let content = "/CS1 cs\n0.25 0 0 0 scn\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let icc_a_off = buf.len();
    let icc_a_hdr =
        format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", output_intent_profile_a.len());
    buf.extend_from_slice(icc_a_hdr.as_bytes());
    buf.extend_from_slice(output_intent_profile_a);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let icc_b_off = buf.len();
    let icc_b_hdr =
        format!("6 0 obj\n<< /N 4 /Length {} >>\nstream\n", embedded_iccbased_profile_b.len());
    buf.extend_from_slice(icc_b_hdr.as_bytes());
    buf.extend_from_slice(embedded_iccbased_profile_b);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    let obj_count = 7;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [
        cat_off, pages_off, page_off, stream_off, icc_a_off, icc_b_off,
    ] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// Build a PDF whose page declares BOTH `/OutputIntents` profile A and
/// page Resources `/ColorSpace /DefaultCMYK [/ICCBased <profile B>]`,
/// then paints a bare `/DeviceCMYK` (`0.25 0 0 0 k`) rectangle.
///
/// ISO 32000-1:2008 §8.6.5.6: the `/DefaultCMYK` override redirects
/// bare /DeviceCMYK paint through the override's colour space —
/// independently of any document-level /OutputIntents profile. The
/// override therefore wins on the rendered pixel.
///
/// Object layout:
///   1 — Catalog (with /OutputIntents → 5 0 R)
///   2 — Pages
///   3 — Page (with Resources /ColorSpace /DefaultCMYK → ICCBased
///       referencing 6 0 R)
///   4 — Content stream
///   5 — OutputIntent profile A stream
///   6 — DefaultCMYK embedded profile B stream
fn build_pdf_default_cmyk_overrides_output_intent(
    output_intent_profile_a: &[u8],
    default_cmyk_profile_b: &[u8],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    let catalog = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK A) /DestOutputProfile 5 0 R >>] >>\nendobj\n";
    buf.extend_from_slice(catalog.as_bytes());

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    // /DefaultCMYK is a named entry in /Resources /ColorSpace per
    // §8.6.5.6. Its value is an ICCBased colour space wrapping profile B.
    let page = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /DefaultCMYK [/ICCBased 6 0 R] >> >> /Contents 4 0 R >>\nendobj\n";
    buf.extend_from_slice(page.as_bytes());

    let stream_off = buf.len();
    // Bare /DeviceCMYK paint via `k` — the canonical case §8.6.5.6
    // redirects through /DefaultCMYK.
    let content = "0.25 0 0 0 k\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let icc_a_off = buf.len();
    let icc_a_hdr =
        format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", output_intent_profile_a.len());
    buf.extend_from_slice(icc_a_hdr.as_bytes());
    buf.extend_from_slice(output_intent_profile_a);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let icc_b_off = buf.len();
    let icc_b_hdr =
        format!("6 0 obj\n<< /N 4 /Length {} >>\nstream\n", default_cmyk_profile_b.len());
    buf.extend_from_slice(icc_b_hdr.as_bytes());
    buf.extend_from_slice(default_cmyk_profile_b);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    let obj_count = 7;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [
        cat_off, pages_off, page_off, stream_off, icc_a_off, icc_b_off,
    ] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// Build a PDF whose page declares `/Resources /ColorSpace /DefaultRGB
/// [/ICCBased <stream>]` and paints bare `/DeviceRGB` with `rg`.
/// §8.6.5.6 redirects the bare paint through the /DefaultRGB override.
///
/// No /OutputIntents declared — RGB OutputIntents aren't carried by the
/// pipeline at all (only CMYK /N=4), so the only thing that can
/// influence bare-DeviceRGB rendering through this fixture is the
/// /DefaultRGB consumer the resolver gains in this phase.
///
/// Object layout:
///   1 — Catalog
///   2 — Pages
///   3 — Page (with Resources /ColorSpace /DefaultRGB → ICCBased
///       referencing 5 0 R)
///   4 — Content stream
///   5 — DefaultRGB embedded N=3 profile stream
fn build_pdf_default_rgb_overrides_bare_device_rgb(default_rgb_profile: &[u8]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    let page = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /DefaultRGB [/ICCBased 5 0 R] >> >> /Contents 4 0 R >>\nendobj\n";
    buf.extend_from_slice(page.as_bytes());

    let stream_off = buf.len();
    // Bare /DeviceRGB paint via `rg` — `0.8 0.2 0.5 rg` is RGB(204, 51, 128)
    // in raw bytes. With the override active the renderer routes the
    // three components through the override profile; without it, the
    // raw bytes land on the canvas directly.
    let content = "0.8 0.2 0.5 rg\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let icc_off = buf.len();
    // RGB ICC profile stream: /N 3 (not 4 — this is a 3-channel input
    // profile).
    let icc_hdr = format!("5 0 obj\n<< /N 3 /Length {} >>\nstream\n", default_rgb_profile.len());
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(default_rgb_profile);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    let obj_count = 6;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, icc_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// Build a PDF whose page declares `/Resources /ColorSpace /DefaultGray
/// [/Separation /MagentaSpot /DeviceCMYK <Type-4>]` and paints bare
/// `/DeviceGray` with `g`. §8.6.5.6 redirects the bare paint through the
/// /DefaultGray override.
///
/// The Separation tint transform `{ 0.0 exch 0.0 0.0 }` consumes the
/// single gray input and emits CMYK(0, gray, 0, 0). For gray=0.5 the
/// alternate is CMYK(0, 0.5, 0, 0); §10.3.5 projects that to
/// RGB(255, 127, 255) — a magenta that's clearly distinct from the
/// literal grey RGB(127, 127, 127) the bare paint would produce
/// without the override.
///
/// Object layout:
///   1 — Catalog
///   2 — Pages
///   3 — Page (with Resources /ColorSpace /DefaultGray → Separation
///       array referencing 5 0 R)
///   4 — Content stream
///   5 — Tint-transform Type-4 stream
fn build_pdf_default_gray_routes_bare_device_gray() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    // /DefaultGray is a Separation that lifts the gray input into the
    // M channel; the alternate is /DeviceCMYK so §10.3.5 yields a
    // magenta-channel-only pixel.
    let page = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /DefaultGray [/Separation /MagentaSpot /DeviceCMYK 5 0 R] >> >> /Contents 4 0 R >>\nendobj\n";
    buf.extend_from_slice(page.as_bytes());

    let stream_off = buf.len();
    // Bare /DeviceGray paint via `g` at gray=0.5.
    let content = "0.5 g\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let tint_off = buf.len();
    let tint_program: &[u8] = b"{ 0.0 exch 0.0 0.0 }";
    let tint_hdr = format!(
        "5 0 obj\n<< /FunctionType 4 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\nstream\n",
        tint_program.len()
    );
    buf.extend_from_slice(tint_hdr.as_bytes());
    buf.extend_from_slice(tint_program);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    let obj_count = 6;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, tint_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    buf
}

fn render_rgba(doc: &PdfDocument) -> Vec<u8> {
    let opts = RenderOptions::with_dpi(72).as_raw();
    let img = render_page(doc, 0, &opts).expect("render_page");
    assert_eq!(img.format, ImageFormat::RawRgba8);
    img.data
}

fn pixel_at(rgba: &[u8], x: u32, y: u32) -> (u8, u8, u8, u8) {
    let w = 100u32;
    let h = 100u32;
    assert_eq!(rgba.len() as u32, w * h * 4);
    assert!(x < w && y < h);
    let off = ((y * w + x) * 4) as usize;
    (rgba[off], rgba[off + 1], rgba[off + 2], rgba[off + 3])
}

// ===========================================================================
// Phase 2 positive test
// ===========================================================================

/// Pin that a /DeviceCMYK fill on a page whose document declares a
/// CMYK `/OutputIntents` profile is rendered via the qcms-driven ICC
/// path rather than ISO 32000-1:2008 §10.3.5's additive-clamp formula.
///
/// Fixture details:
///   - CMYK input: (0.25, 0, 0, 0) — modest cyan tint.
///   - Profile: minimal in-test CMYK→RGB LUT8 that maps every CMYK input
///     to constant `RGB(128, 128, 128)`. With the OutputIntent path
///     live, every pixel inside the rect must be ~128 grey on every
///     channel. With the additive-clamp fallback the pixel would be
///     `(191, 255, 255)` — `1 - (C + K)`, `1 - (M + K)`, `1 - (Y + K)`
///     scaled to bytes.
#[test]
fn device_cmyk_paint_with_output_intent_renders_via_icc_not_additive_clamp() {
    // L*53 maps roughly to sRGB(128, 128, 128) — a clear non-additive-
    // clamp anchor for CMYK(0.25, 0, 0, 0).
    let target_l_byte: u8 = 135;
    let icc = build_minimal_cmyk_to_rgb_lut8_profile(target_l_byte);
    // First sanity-check the synthesised profile compiles into a real
    // qcms transform — otherwise the test would silently degrade to
    // the §10.3.5 fallback and the assertion below would fail for the
    // wrong reason. The transform-build path is the same one the
    // composite renderer will exercise on this profile.
    {
        use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
        use std::sync::Arc;
        let prof = Arc::new(
            IccProfile::parse(icc.clone(), 4)
                .expect("synthesised profile parses through IccProfile::parse"),
        );
        let t = Transform::new_srgb_target(prof, RenderingIntent::RelativeColorimetric);
        assert!(
            t.has_cmm(),
            "synthesised profile must compile into a real qcms transform; \
             without it the OutputIntent test degrades to the additive-clamp \
             fallback and asserts the wrong thing"
        );
        // Sanity-pin the constant CLUT actually drives qcms: with this
        // profile every CMYK input must produce roughly (128, 128, 128).
        // qcms tetra-CLUT interpolation on a 2^4 grid with constant
        // output should be exact to within rounding.
        let rgb = t.convert_cmyk_pixel(64, 0, 0, 0);
        // Lab(53, 0, 0) → sRGB ≈ (128, 128, 128) within rounding. Tolerate
        // ±10 per channel — Lab→XYZ→sRGB through the qcms pipeline rounds
        // at multiple steps and ICC v2 Lab encoding has its own scale
        // quantisation.
        let near = |a: u8, b: u8| (a as i32 - b as i32).abs() <= 10;
        assert!(
            near(rgb[0], 128) && near(rgb[1], 128) && near(rgb[2], 128),
            "qcms must drive the constant CLUT: got {rgb:?}, want ~(128, 128, 128) \
             ±10 (Lab(53,0,0) → sRGB grey)"
        );
    }

    let pdf = build_pdf_cmyk_with_output_intent(&icc);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    // Re-confirm the document accessor finds the OutputIntent. If this
    // returns None the test isn't actually probing the OutputIntent
    // path — it'd just probe the no-OutputIntent baseline.
    let oi = doc
        .output_intent_cmyk_profile()
        .expect("synthetic catalog declares a CMYK OutputIntent");
    assert_eq!(oi.n_components(), 4, "OutputIntent must be /N=4");

    let rgba = render_rgba(&doc);
    let (r, g, b, _a) = pixel_at(&rgba, 50, 50);

    // Additive-clamp value for CMYK(0.25, 0, 0, 0) is RGB(0.75, 1.0, 1.0)
    // = (191, 255, 255). The qcms-converted value is ~(128, 128, 128).
    // Tolerance ±10 absorbs Lab → XYZ → sRGB rounding through the chain.
    let near_const = |v: u8| (v as i32 - 128).abs() <= 10;
    assert!(
        near_const(r) && near_const(g) && near_const(b),
        "OutputIntent /DeviceCMYK paint expected qcms-converted RGB ~(128, 128, 128); \
         got ({r}, {g}, {b}). RGB(191, 255, 255) would mean the §10.3.5 additive-clamp \
         fallback fired — the resolver is not consulting ctx.output_intent_cmyk."
    );
}

// ===========================================================================
// Negative pin: no OutputIntent → §10.3.5 additive-clamp preserved
// ===========================================================================

/// Pin that a /DeviceCMYK fill on a page whose document declares no
/// `/OutputIntents` array is rendered through ISO 32000-1:2008
/// §10.3.5's additive-clamp formula, byte-for-byte, as it shipped
/// before OutputIntent threading landed.
///
/// This is the contrapositive of the positive test: when
/// `ctx.output_intent_cmyk` is `None`, the resolver MUST fall through
/// to the shipped behaviour. A bug that unconditionally consulted
/// some other ICC profile (or that flipped the precedence rules) would
/// surface here as the wrong colour.
#[test]
fn device_cmyk_paint_without_output_intent_renders_additive_clamp() {
    let pdf = build_pdf_cmyk_without_output_intent();
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    // Cross-check the catalog has no OutputIntent — if it did, this
    // test would conflate "no OI" with "OI that happens to produce
    // additive-clamp values" and could pass for the wrong reason.
    assert!(
        doc.output_intent_cmyk_profile().is_none(),
        "fixture must declare no /OutputIntents in catalog"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, _a) = pixel_at(&rgba, 50, 50);

    // CMYK(0.25, 0, 0, 0) → additive-clamp:
    //   R = 1 - (0.25 + 0) = 0.75 → 191
    //   G = 1 - (0.00 + 0) = 1.00 → 255
    //   B = 1 - (0.00 + 0) = 1.00 → 255
    assert_eq!(
        (r, g, b),
        (191, 255, 255),
        "without /OutputIntents the §10.3.5 additive-clamp fallback must \
         be preserved byte-for-byte; got ({r}, {g}, {b})"
    );
}

// ===========================================================================
// QA: byte-exact Lab→sRGB pin (replaces the ±10 hand-wave)
// ===========================================================================

/// Byte-exact pin of the qcms reference value the synthesised
/// `target_l_byte=135` profile yields.
///
/// The existing positive test (`device_cmyk_paint_with_output_intent_*`)
/// asserts the rendered pixel falls within `(128, 128, 128) ± 10` per
/// channel — that's a hand-wave that hides up to a ~9-byte channel-by-
/// channel drift. Derived against qcms 0.3.0 (the version pinned in
/// Cargo.lock at this commit), the byte-exact reference for
/// `target_l_byte=135` + CMYK(64,0,0,0) at `RelativeColorimetric` is
/// `(126, 126, 126)`. The rendered pixel at (50, 50) through the
/// composite pipeline is `(126, 126, 126, 255)`. We pin both — any
/// drift in the qcms chain (Lab→XYZ→sRGB), the LUT8 tetra-interp, or
/// the resolver's 8-bit round-trip surfaces here byte-for-byte.
///
/// If a future qcms upgrade shifts the reference, the right answer is
/// to re-derive the value here, not to widen the tolerance — `±10` was
/// the impl-agent's tolerance for an unmeasured target; this probe pins
/// the actual measured target.
#[test]
fn output_intent_render_pixel_is_byte_exact_against_qcms_reference() {
    use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
    use std::sync::Arc;

    let target_l_byte: u8 = 135;
    let icc = build_minimal_cmyk_to_rgb_lut8_profile(target_l_byte);

    // Standalone transform: pin the qcms output byte-for-byte against
    // the derived reference. CMYK(64, 0, 0, 0) is the input the
    // positive integration test feeds for its sanity check.
    {
        let prof = Arc::new(IccProfile::parse(icc.clone(), 4).expect("parse"));
        let t = Transform::new_srgb_target(prof, RenderingIntent::RelativeColorimetric);
        let rgb = t.convert_cmyk_pixel(64, 0, 0, 0);
        assert_eq!(
            rgb,
            [126u8, 126, 126],
            "qcms 0.3.0 byte-exact reference for target_l_byte=135 + CMYK(64,0,0,0): \
             expected (126, 126, 126); got {rgb:?}. Re-derive (see plan errata) if qcms \
             ever changes its Lab→sRGB chain — do not widen tolerance."
        );
    }

    // Through the composite renderer: pin the rendered pixel at the
    // centre of the painted rect byte-for-byte.
    let pdf = build_pdf_cmyk_with_output_intent(&icc);
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (126u8, 126, 126, 255),
        "rendered pixel must match the qcms reference byte-for-byte; got ({r},{g},{b},{a}). \
         (191,255,255,_) means the §10.3.5 fallback fired."
    );
}

/// Pin the qcms reference value is intent-independent for the synthesised
/// constant-CLUT profile.
///
/// The constant-CLUT shape of the synthesised profile means a CMM whose
/// gamut compression depends on rendering intent (which is the whole
/// point of having intents) should still produce the same value — there's
/// no out-of-gamut excursion to compress. If qcms ever starts producing
/// different values per intent on a constant CLUT that's a CMM bug
/// worth surfacing.
#[test]
fn output_intent_constant_clut_is_invariant_across_rendering_intents() {
    use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
    use std::sync::Arc;

    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let prof = Arc::new(IccProfile::parse(icc, 4).expect("parse"));
    let mut last: Option<[u8; 3]> = None;
    for intent in [
        RenderingIntent::Perceptual,
        RenderingIntent::RelativeColorimetric,
        RenderingIntent::Saturation,
        RenderingIntent::AbsoluteColorimetric,
    ] {
        let t = Transform::new_srgb_target(Arc::clone(&prof), intent);
        let rgb = t.convert_cmyk_pixel(64, 0, 0, 0);
        if let Some(prev) = last {
            assert_eq!(
                prev, rgb,
                "constant-CLUT qcms output must be identical across rendering intents; \
                 first intent yielded {prev:?}, intent={intent:?} yielded {rgb:?}"
            );
        }
        last = Some(rgb);
    }
}

// ===========================================================================
// QA: qcms validation fragility — bad-profile fall-through
// ===========================================================================

/// Pin that a syntactically-shaped but tag-table-truncated CMYK profile
/// declared on `/OutputIntents` does not crash the renderer and produces
/// the §10.3.5 fallback colour byte-for-byte.
///
/// This is the impl-agent's open-question #1 surfaced as a probe: when
/// qcms refuses to compile the OutputIntent profile, `Transform::
/// convert_cmyk_pixel` devolves internally — but the renderer-level
/// behaviour must be (a) no panic and (b) the same RGB the no-
/// OutputIntent fixture produces, so a malformed `/OutputIntents`
/// degrades gracefully.
#[test]
fn output_intent_with_unparseable_profile_falls_through_to_additive_clamp() {
    // Header-only profile: parses through `IccProfile::parse` (which
    // only validates the 128-byte header), but qcms refuses at build
    // time because there's no tag table. Mirrors the stub the in-source
    // unit test in color.rs uses but reaches the rasteriser end-to-end.
    let mut header_only = vec![0u8; 128];
    header_only[8..12].copy_from_slice(&0x0400_0000u32.to_be_bytes());
    header_only[12..16].copy_from_slice(b"prtr");
    header_only[16..20].copy_from_slice(b"CMYK");
    header_only[20..24].copy_from_slice(b"Lab ");
    header_only[36..40].copy_from_slice(b"acsp");

    let pdf = build_pdf_cmyk_with_output_intent(&header_only);
    let doc = PdfDocument::from_bytes(pdf).expect("open");

    // Sanity-pin: the document-level accessor still hands back the
    // parsed-header profile, so the renderer DOES see a Some on
    // `ctx.output_intent_cmyk` — the fall-through has to happen inside
    // `convert_cmyk_pixel`, not by the accessor returning None.
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "header-only stub must parse through IccProfile::parse; fall-through must \
         happen inside Transform::convert_cmyk_pixel, not at the accessor"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (191u8, 255, 255, 255),
        "unparseable OutputIntent profile must fall through to §10.3.5 byte-exact; \
         got ({r},{g},{b},{a})"
    );
}

/// Pin that an OutputIntent profile whose ICC header declares a non-CMYK
/// colour space (`RGB `, `GRAY`, `Lab `) is filtered out by
/// `IccProfile::parse`'s cross-check, even though the stream dict's
/// `/N 4` would otherwise let it through the accessor.
///
/// `IccProfile::parse(bytes, declared_n)` at `src/color.rs:159` requires
/// that the ICC header's implied component count match the stream
/// dict's `/N`. An `RGB ` header implies `n=3`; `declared_n=4` → reject.
/// `output_intent_cmyk_profile` then returns `None`, and the renderer
/// falls back to §10.3.5 byte-for-byte.
///
/// This is the strongest gate: a malformed profile that lied about
/// colour space in the ICC header gets rejected before reaching qcms.
/// A regression that loosened the cross-check would let the qcms layer
/// see CMYK bytes through an RGB profile — at best garbage, at worst a
/// panic in the CMM.
#[test]
fn output_intent_with_mismatched_icc_header_colour_space_is_rejected_at_parse() {
    let mut header_only = vec![0u8; 128];
    header_only[8..12].copy_from_slice(&0x0400_0000u32.to_be_bytes());
    header_only[12..16].copy_from_slice(b"prtr");
    header_only[16..20].copy_from_slice(b"RGB "); // intentionally mismatched
    header_only[20..24].copy_from_slice(b"Lab ");
    header_only[36..40].copy_from_slice(b"acsp");

    let pdf = build_pdf_cmyk_with_output_intent(&header_only);
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    // IccProfile::parse rejects the mismatch (header→n=3 vs declared_n=4);
    // the accessor surfaces None.
    assert!(
        doc.output_intent_cmyk_profile().is_none(),
        "IccProfile::parse must reject when ICC header colour-space \
         tag implies a different component count than the stream's /N"
    );
    // Renderer falls through to §10.3.5 byte-for-byte.
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (191u8, 255, 255, 255),
        "mismatched-header OutputIntent must fall through to §10.3.5; got ({r},{g},{b},{a})"
    );
}

// ===========================================================================
// QA: helper-level consistency (§10.3.5 source-of-truth probe)
// ===========================================================================

/// Pin that `crate::extractors::images::cmyk_pixel_to_rgb` and the
/// resolver helper's no-OutputIntent arm produce the same RGB bytes on
/// the same CMYK quadruple.
///
/// This is the HONEST_GAP the impl agent flagged in
/// `cmyk_to_rgb_via_intent_falls_back_when_profile_has_no_cmm`. Verified
/// here at the public-API level by routing both paths through a known
/// CMYK input and comparing byte-for-byte. If a future refactor diverges
/// the two §10.3.5 implementations, the fallback path inside qcms's
/// no-CMM arm could disagree with the resolver's bare-fallback arm even
/// though both intend the spec formula.
///
/// The probe iterates over a handful of representative inputs — pure
/// process inks, the test fixture's input, and a few interior CMYK
/// quadruples. Every input must agree.
#[test]
fn additive_clamp_consistency_between_extractors_helper_and_no_output_intent_arm() {
    use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
    use std::sync::Arc;

    // Build a header-only stub: qcms refuses, Transform::convert_cmyk_pixel
    // devolves to crate::extractors::images::cmyk_pixel_to_rgb internally
    // (verified at src/color.rs:301). That's the reference "no-CMM
    // fallback" path.
    let mut header_only = vec![0u8; 128];
    header_only[8..12].copy_from_slice(&0x0400_0000u32.to_be_bytes());
    header_only[12..16].copy_from_slice(b"prtr");
    header_only[16..20].copy_from_slice(b"CMYK");
    header_only[20..24].copy_from_slice(b"Lab ");
    header_only[36..40].copy_from_slice(b"acsp");
    let prof = Arc::new(IccProfile::parse(header_only, 4).expect("parse"));
    let t = Transform::new_srgb_target(prof, RenderingIntent::RelativeColorimetric);

    // The §10.3.5 formula in plain Rust — re-derived here so we don't
    // import the crate-private helper. Both the Transform no-CMM arm
    // and the resolver fallback must agree with this.
    fn spec_additive_clamp(c: u8, m: u8, y: u8, k: u8) -> [u8; 3] {
        let cf = c as f32 / 255.0;
        let mf = m as f32 / 255.0;
        let yf = y as f32 / 255.0;
        let kf = k as f32 / 255.0;
        let r = ((1.0 - (cf + kf).min(1.0)) * 255.0).round() as u8;
        let g = ((1.0 - (mf + kf).min(1.0)) * 255.0).round() as u8;
        let b = ((1.0 - (yf + kf).min(1.0)) * 255.0).round() as u8;
        [r, g, b]
    }

    for (c, m, y, k) in [
        (0u8, 0, 0, 0),
        (255, 0, 0, 0),
        (0, 255, 0, 0),
        (0, 0, 255, 0),
        (0, 0, 0, 255),
        (64, 0, 0, 0), // fixture input
        (128, 128, 128, 128),
        (200, 100, 50, 25),
    ] {
        let from_transform = t.convert_cmyk_pixel(c, m, y, k);
        let from_spec = spec_additive_clamp(c, m, y, k);
        assert_eq!(
            from_transform, from_spec,
            "Transform no-CMM fallback must agree with §10.3.5 spec on CMYK({c},{m},{y},{k}); \
             transform={from_transform:?}, spec={from_spec:?}"
        );
    }
}

// ===========================================================================
// QA: foundation coverage probes (q/Q, alpha edges, deferred placeholders)
// ===========================================================================

/// Pin that DeviceCMYK paint inside a `q ... Q` save-restore bracket
/// still routes through the OutputIntent ICC.
///
/// `q`/`Q` push/pop the graphics state; a regression that re-built the
/// resolution context inside the bracket without re-attaching the
/// OutputIntent borrow would lose the ICC routing on the inner paint
/// even though it's the same page.
#[test]
fn output_intent_survives_graphics_state_save_restore() {
    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);
    // q / fill / Q bracket performing the CMYK paint inside a fresh
    // graphics-state scope. The inner paint must still hit ICC.
    let catalog_entries =
        "/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (S) /DestOutputProfile 5 0 R >>]";
    let content = "q\n0.25 0 0 0 k\n20 20 60 60 re\nf\nQ\n";
    let pdf = build_pdf_with_catalog_entries_and_content(catalog_entries, content, Some(&icc));
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (126u8, 126, 126, 255),
        "DeviceCMYK paint inside q/Q must still route through OutputIntent ICC; got ({r},{g},{b},{a})"
    );
}

/// Pin that a fully-opaque DeviceCMYK paint at the alpha=1 edge resolves
/// to the qcms reference without any zero-coverage shortcut intercepting
/// the conversion before it reaches the helper.
///
/// The composite path has multiple alpha-aware shortcuts (zero-alpha
/// skip, fully-opaque skip, etc.). A regression that bypassed the
/// colour stage on the opaque edge would silently produce the
/// uncomposited additive-clamp value.
#[test]
fn output_intent_renders_at_alpha_one_edge() {
    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);
    // Default content stream has no explicit alpha — that's alpha=1.
    let pdf = build_pdf_cmyk_with_output_intent(&icc);
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(a, 255, "alpha=1 paint must produce fully-opaque pixel");
    assert_eq!(
        (r, g, b),
        (126u8, 126, 126),
        "alpha=1 paint must still route through OutputIntent ICC; got ({r},{g},{b})"
    );
}

/// Pin that a subsequent opaque RGB over-paint obscures the prior CMYK
/// ICC paint cleanly — the OutputIntent path doesn't leak ICC-converted
/// pixels into a later non-CMYK paint scope.
#[test]
fn output_intent_does_not_leak_into_subsequent_rgb_overpaint() {
    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let catalog_entries =
        "/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (S) /DestOutputProfile 5 0 R >>]";
    // CMYK paint, then white RGB paint covering the same rect.
    let content = "0.25 0 0 0 k\n20 20 60 60 re\nf\n1 1 1 rg\n20 20 60 60 re\nf\n";
    let pdf = build_pdf_with_catalog_entries_and_content(catalog_entries, content, Some(&icc));
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (255u8, 255, 255, 255),
        "white RGB over-paint must obscure the CMYK paint regardless of OutputIntent; \
         got ({r},{g},{b},{a})"
    );
}

/// Pin that DeviceCMYK painted inside a Form XObject inherits the
/// document-level OutputIntent. Form XObjects share the document's
/// colour-policy state by spec (§14.8.3) — a regression that built a
/// fresh resolution context for the XObject scope without re-threading
/// the OutputIntent borrow would lose the ICC routing on every spot
/// CMYK paint nested inside the XObject.
///
/// Currently `#[ignore]`-ed pending a Form-XObject test-fixture helper;
/// the marker captures the gap so a follow-up audit picks it up.
#[test]
#[ignore = "OUTPUT_INTENT_DEFER_FORM_XOBJECT_INHERITANCE"]
fn output_intent_inherited_by_form_xobject_paint() {
    panic!("placeholder: needs a Form XObject test-fixture helper");
}

/// Pin the page-level `/DefaultCMYK` override precedence over the
/// document `/OutputIntents` profile for bare `/DeviceCMYK` paint.
///
/// ISO 32000-1:2008 §8.6.5.6 says: when a page declares
/// `/Resources /ColorSpace /DefaultCMYK <override>`, any bare
/// `/DeviceCMYK` paint operator (`k`/`K`/`scn` against a DeviceCMYK
/// alias) MUST be interpreted as if it specified `<override>` instead
/// of the device family default. The override therefore wins over the
/// document-level `/OutputIntents` profile when present, because the
/// override IS the page's declared CMYK colour space and OutputIntent
/// only applies as the default when no override has been declared.
///
/// Fixture geometry:
///   - Catalog declares /OutputIntents → profile A (target_l_byte=135 →
///     qcms reference RGB(126,126,126)).
///   - Page Resources /ColorSpace /DefaultCMYK → [/ICCBased <stream B>]
///     where profile B has target_l_byte=200 → qcms reference
///     RGB(194,194,194).
///   - Content stream: `0.25 0 0 0 k   20 20 60 60 re   f`. The `k`
///     operator paints bare /DeviceCMYK, exactly the case §8.6.5.6
///     redirects through /DefaultCMYK.
///
/// Three observable outcomes:
///   - (194, 194, 194, 255): /DefaultCMYK override won — pass.
///   - (126, 126, 126, 255): OutputIntent won — fail (precedence inverted;
///     §8.6.5.6 says /DefaultCMYK takes precedence over the document
///     default).
///   - (191, 255, 255, 255): §10.3.5 additive-clamp fired — neither
///     profile consulted (an even worse regression).
#[test]
fn page_level_default_cmyk_takes_precedence_over_output_intent() {
    let profile_a = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let profile_b = build_minimal_cmyk_to_rgb_lut8_profile(PROFILE_B_TARGET_L_BYTE);

    // Sanity-pin both profiles compile through qcms and produce the
    // expected byte-exact references — without this gate a regression
    // in profile B's transform would make the integration assertion
    // fire for the wrong reason.
    {
        use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
        use std::sync::Arc;
        let prof_a = Arc::new(IccProfile::parse(profile_a.clone(), 4).expect("parse A"));
        let prof_b = Arc::new(IccProfile::parse(profile_b.clone(), 4).expect("parse B"));
        let t_a = Transform::new_srgb_target(prof_a, RenderingIntent::RelativeColorimetric);
        let t_b = Transform::new_srgb_target(prof_b, RenderingIntent::RelativeColorimetric);
        assert_eq!(
            t_a.convert_cmyk_pixel(64, 0, 0, 0),
            [126u8, 126, 126],
            "profile A reference must be (126,126,126); fixture is invalid otherwise"
        );
        assert_eq!(
            t_b.convert_cmyk_pixel(64, 0, 0, 0),
            [194u8, 194, 194],
            "profile B reference must be (194,194,194); fixture is invalid otherwise"
        );
    }

    let pdf = build_pdf_default_cmyk_overrides_output_intent(&profile_a, &profile_b);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "fixture must declare a CMYK OutputIntent so the precedence is actually contested"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    let (br, bg, bb) = PROFILE_B_RGB_AT_FIXTURE_INPUT;
    assert_eq!(
        (r, g, b, a),
        (br, bg, bb, 255),
        "page-level /DefaultCMYK override must take precedence over /OutputIntents \
         on bare /DeviceCMYK paint; expected B's qcms reference {:?}; got ({r},{g},{b},{a}). \
         (126,126,126,_) means OutputIntent won — §8.6.5.6 precedence is inverted. \
         (191,255,255,_) means neither profile was consulted and §10.3.5 fired.",
        (br, bg, bb, 255u8)
    );
}

/// Negative pin: when the page declares NO `/DefaultCMYK` override,
/// bare `/DeviceCMYK` paint must fall through to the document
/// `/OutputIntents` profile (the round-1 behaviour). This is the
/// contrapositive of the precedence test above — it pins that the
/// override consumer doesn't fire spuriously when no override is
/// declared, otherwise a regression that always routed through some
/// hard-coded path would pass the positive test by coincidence.
#[test]
fn no_default_cmyk_falls_through_to_output_intent() {
    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let pdf = build_pdf_cmyk_with_output_intent(&icc);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (126u8, 126, 126, 255),
        "no /DefaultCMYK declared → bare DeviceCMYK paint must route through \
         /OutputIntents (round-1 behaviour); got ({r},{g},{b},{a})"
    );
}

/// Pin the page-level `/DefaultRGB` override drives bare `/DeviceRGB`
/// paint through the override's colour space, even when no `/OutputIntents`
/// is present. §8.6.5.6 redirects bare device-family paint through the
/// /Default<DeviceFamily> override; for RGB this is the only place an
/// override can influence rendering (OutputIntent in our pipeline only
/// carries CMYK).
///
/// Fixture: /DefaultRGB → [/ICCBased <stream>] where the embedded N=3
/// LUT8 profile maps every RGB input to constant `Lab(target_l_byte=200,
/// 0, 0)` → qcms reference RGB(194, 194, 194). Content paints
/// `0.8 0.2 0.5 rg`. Without the override the rendered pixel would be
/// the literal RGB(0.8, 0.2, 0.5) = (204, 51, 128). With the override
/// active the rendered pixel must be the qcms reference value.
#[test]
fn page_level_default_rgb_routes_bare_device_rgb_through_override() {
    let profile = build_minimal_rgb_to_lab_lut8_profile(PROFILE_B_TARGET_L_BYTE);

    // Sanity-pin the synthesised RGB profile actually compiles and
    // produces the expected reference. Without this gate the integration
    // assertion below could fail for the wrong reason (e.g. profile
    // rejected → resolver fall-through → literal RGB observed).
    let rgb_ref = {
        use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
        use std::sync::Arc;
        let prof = Arc::new(IccProfile::parse(profile.clone(), 3).expect("RGB profile parses"));
        let t = Transform::new_srgb_target(prof, RenderingIntent::RelativeColorimetric);
        assert!(
            t.has_cmm(),
            "synthesised RGB LUT8 profile must compile into a real qcms CMM; \
             without it the /DefaultRGB test degrades to additive fall-through and \
             asserts the wrong thing"
        );
        // For RGB input qcms uses convert_rgb_buffer (3 bytes in → 3 bytes
        // out). Get the reference for a representative input.
        let mut out = [0u8; 3];
        out.copy_from_slice(&t.convert_rgb_buffer(&[204u8, 51, 128]));
        out
    };

    let pdf = build_pdf_default_rgb_overrides_bare_device_rgb(&profile);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (rgb_ref[0], rgb_ref[1], rgb_ref[2], 255),
        "page-level /DefaultRGB override must route bare /DeviceRGB paint \
         through the override profile; expected qcms reference {:?}; \
         got ({r},{g},{b},{a}). (204,51,128,_) means the override was not \
         consulted and the literal RGB landed on the canvas directly.",
        rgb_ref
    );
}

/// Pin the page-level `/DefaultGray` override drives bare `/DeviceGray`
/// paint through the override's colour space.
///
/// Fixture: /DefaultGray → [/Separation /MagentaSpot /DeviceCMYK <Type-4
/// tint transform>] that lifts the gray input into the M channel:
/// `gray → CMYK(0, gray, 0, 0)`. Painting `0.5 g` (bare DeviceGray)
/// without the override would produce literal RGB(127, 127, 127). With
/// the override active, the gray value routes through the Separation,
/// produces CMYK(0, 0.5, 0, 0), and projects to RGB(255, 127, 255) via
/// §10.3.5 (no OutputIntent declared in this fixture). The colour
/// change is visible and discriminates the routes.
///
/// **Coverage note:** this Separation route covers the dispatcher
/// edge of /DefaultGray (the override is consulted; the gray
/// component reaches the override's colour space). The complementary
/// N=1 ICC route — `/DefaultGray [/ICCBased <N=1 TRC stream>]` —
/// drives the qcms gray pipeline and is pinned by
/// `qa_round4_default_gray_iccbased_n1_routes_through_qcms` below.
/// The two probes together prove both ends of the §8.6.5.6
/// /DefaultGray contract: dispatch and ICC conversion.
#[test]
fn page_level_default_gray_routes_bare_device_gray_through_override() {
    let pdf = build_pdf_default_gray_routes_bare_device_gray();
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_none(),
        "fixture must declare no /OutputIntents — the override drives the route \
         entirely; an OutputIntent in play would confound the assertion"
    );
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    // The Separation tint transform `{ 0.0 exch 0.0 0.0 }` consumes the
    // single gray-component input and emits CMYK(0, gray, 0, 0). For
    // gray=0.5 the alternate is CMYK(0, 0.5, 0, 0) → §10.3.5 produces
    // R=1, G=1-0.5=0.5, B=1. The 0.5 channel rounds to 128 (round-half-
    // up via Rust f32 → u8 cast at the pixel-writer boundary). The
    // R=255, B=255 channels and the M-only behaviour are the
    // discriminating signal — they are byte-exact and would be
    // ABSENT if the override was bypassed (literal gray would produce
    // RGB(128, 128, 128) with no magenta channel).
    assert_eq!(
        r, 255,
        "/DefaultGray override → Separation magenta projection must produce R=255 \
         (additive-clamp of CMYK(0,0.5,0,0) leaves R=1.0); got ({r},{g},{b},{a}). \
         (128,*,*) means the override was bypassed and the literal gray landed."
    );
    assert_eq!(
        b, 255,
        "/DefaultGray override → Separation magenta projection must produce B=255; \
         got ({r},{g},{b},{a})"
    );
    // tiny-skia's f32 → u8 conversion at color.rs:444 is
    // `(c * 255.0 + 0.5) as u8` — round-half-up via truncation. For
    // c=0.5 that's 0.5 * 255.0 + 0.5 = 128.0 → 128, deterministic
    // across platforms and tiny-skia builds. Earlier rounds asserted
    // a (120..=130) tolerance against a supposed platform-dependent
    // rounding; the actual conversion is exact, so pin the byte.
    assert_eq!(
        g, 128,
        "/DefaultGray override → Separation magenta projection must produce \
         G=128 (additive-clamp of CMYK(0,0.5,0,0) gives G=0.5; tiny-skia's \
         f32→u8 conversion is (c*255.0+0.5) as u8 = 128, deterministic); \
         got G={g}, full pixel ({r},{g},{b},{a}). G=255 would mean no \
         magenta — override bypassed."
    );
    assert_eq!(a, 255, "alpha=1 paint must be fully opaque; got a={a}");
}

/// Pin `/DefaultGray [/ICCBased <N=1 TRC stream>]` drives bare
/// `/DeviceGray` paint through the qcms gray pipeline.
///
/// Round 3 documented "qcms 0.3.0's LUT8 parser only accepts in_chan ∈
/// {3, 4}; a 1-channel LUT8 would be rejected at compile time" — true
/// for LUT8 (`mft1`) bodies, but qcms's GRAY-signature arm at
/// `iccread.rs:1712-1714` is a *separate* path that reads the `kTRC`
/// (gray TRC) curveType tag, not a LUT8. A real N=1 Gray ICC profile
/// uses `kTRC`, qcms compiles it via `transform_create` →
/// `qcms_transform_data_gray_*` (`transform.rs:437-475`), and the
/// gray channel becomes RGB through the destination sRGB profile's
/// output gamma tables. The resolver previously had no N=1 arm at all
/// — `resolve_iccbased` fell straight to `first_as_gray(components)`,
/// emitting the literal gray byte without ever consulting qcms.
///
/// Fixture: a one-page PDF whose /DefaultGray is `[/ICCBased <N=1
/// linear-curv TRC stream>]`. The TRC is a 256-entry linear ramp
/// 0..65535 → effectively gamma 1.0 in the linear PCS-Y
/// representation. Painting `0.5 g` routes the 0.5 gray byte (128)
/// through the qcms transform; the linear PCS-Y is encoded back to
/// sRGB via the destination sRGB profile's inverse gamma. The
/// resulting RGB is the canonical sRGB encoding of linear gray 0.5 —
/// a value distinct from the no-override RGB(128, 128, 128) literal.
///
/// The expected RGB is derived empirically from the same qcms call
/// the resolver makes: build the profile, parse it through
/// `IccProfile::parse`, call `Transform::new_srgb_target` +
/// `convert_gray_buffer`, and compare byte-exact. No tolerance — the
/// renderer and the reference call use the same code path.
#[test]
fn qa_round4_default_gray_iccbased_n1_routes_through_qcms() {
    let profile = build_minimal_gray_trc_profile();

    // Sanity-pin the synthesised Gray profile parses through
    // IccProfile::parse (N=1) and compiles into a real qcms transform.
    // Without this gate the integration assertion below could fail
    // for the wrong reason (e.g. profile rejected → resolver
    // fall-through to literal gray).
    let gray_ref = {
        use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
        use std::sync::Arc;
        let prof = Arc::new(
            IccProfile::parse(profile.clone(), 1)
                .expect("Gray TRC profile parses through IccProfile::parse(_, 1)"),
        );
        let t = Transform::new_srgb_target(prof, RenderingIntent::RelativeColorimetric);
        assert!(
            t.has_cmm(),
            "synthesised Gray TRC profile must compile into a real qcms CMM; \
             without it the /DefaultGray ICC test degrades to fall-through and \
             asserts the wrong thing"
        );
        // Render reference: feed the single gray byte 128 (the painted
        // 0.5 quantised at the resolver boundary) and read back the
        // 3 RGB bytes qcms produces. The renderer's resolver runs the
        // same call inside the N=1 arm, so the rendered pixel must
        // match byte-exact.
        let out = t.convert_gray_buffer(&[128u8]);
        assert_eq!(out.len(), 3, "Gray8 → RGB8 conversion emits 3 bytes per input");
        [out[0], out[1], out[2]]
    };

    // Build the PDF: /DefaultGray → [/ICCBased <N=1 stream>], paint
    // `0.5 g` covering a 60×60 rect at the canvas centre. Object
    // layout mirrors `build_pdf_default_rgb_overrides_bare_device_rgb`
    // but with /N 1 on the ICC stream and a one-byte component.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    let page = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /DefaultGray [/ICCBased 5 0 R] >> >> /Contents 4 0 R >>\nendobj\n";
    buf.extend_from_slice(page.as_bytes());
    let stream_off = buf.len();
    let content = "0.5 g\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_off = buf.len();
    let icc_hdr = format!("5 0 obj\n<< /N 1 /Length {} >>\nstream\n", profile.len());
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(&profile);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, icc_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );

    let doc = PdfDocument::from_bytes(buf).expect("open synthetic PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_none(),
        "fixture must declare no /OutputIntents — the /DefaultGray ICC override \
         drives the route entirely"
    );
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (gray_ref[0], gray_ref[1], gray_ref[2], 255),
        "/DefaultGray [/ICCBased N=1] override must route bare /DeviceGray paint \
         through the qcms gray pipeline; expected qcms reference {:?}; got \
         ({r},{g},{b},{a}). (128,128,128,_) means the resolver fell through to \
         first_as_gray and never consulted qcms — the N=1 arm is missing.",
        gray_ref
    );
}

/// Pin a malformed `/DefaultCMYK <string>` entry falls through to the
/// document `/OutputIntents` profile rather than silently mis-rendering
/// the paint via first-component-as-gray.
///
/// A `/Default<Family>` entry per §8.6.5.6 MUST be a colour space —
/// a Name (device-family alias) or an Array (CalGray, ICCBased,
/// Separation, …). A PDF that declares `/DefaultCMYK (some string)`
/// is malformed; the renderer must decide between:
///   1. Honour the malformed entry by routing through
///      `resolve_spaced`'s catch-all `first_as_gray` arm. For
///      CMYK(0.25, 0, 0, 0) this produces RGB(64, 64, 64) — wrong
///      colour, silent mis-rendering, indistinguishable from a
///      buggy override.
///   2. Treat the malformed entry as "no override declared" and
///      fall through to the device-family path:
///      `ResolvedColor::Cmyk` → composite projection via
///      `cmyk_to_rgb_via_intent` → `ctx.output_intent_cmyk`. For
///      this fixture's constant-CLUT OutputIntent that's RGB
///      ~(128, 128, 128) — the press-target colour the OutputIntent
///      claims is right, which is the best fallback a renderer can
///      offer for a malformed override.
///
/// We pick option 2 — a malformed `/Default<Family>` is structurally
/// indistinguishable from the entry being absent, so honouring the
/// OutputIntent matches the §8.6.5.6 + §14.11.5 precedence cascade
/// the rest of the resolver implements.
///
/// Fixture: catalog declares /OutputIntents → constant-grey CMYK
/// profile; page declares /DefaultCMYK as a literal PDF string
/// (`/DefaultCMYK (not a colour space)`). Content paints
/// `0.25 0 0 0 k`. With the fix the pixel matches the OutputIntent
/// reference; without it the pixel is the literal-grey (64, 64, 64).
#[test]
fn qa_round4_malformed_default_cmyk_falls_through_to_output_intent() {
    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);

    // OutputIntent reference: feed CMYK(0.25, 0, 0, 0) through the
    // same constant-CLUT profile the catalog declares. With the
    // fall-through path firing, the rendered pixel must match this.
    let oi_ref = {
        use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
        use std::sync::Arc;
        let prof = Arc::new(IccProfile::parse(icc.clone(), 4).expect("CMYK profile parses"));
        let t = Transform::new_srgb_target(prof, RenderingIntent::RelativeColorimetric);
        assert!(t.has_cmm(), "synthesised CMYK profile must compile for the reference path");
        // Renderer quantises 0.25 → 64; the constant CLUT then produces
        // ~(128, 128, 128) regardless of the CMYK input.
        let rgb = t.convert_cmyk_pixel(64, 0, 0, 0);
        [rgb[0], rgb[1], rgb[2]]
    };

    // Build the PDF directly — none of the existing builders carry
    // a malformed /DefaultCMYK entry. /DefaultCMYK (string) is a
    // literal PDF string object: parses to Object::String, which is
    // neither a Name nor an Array.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    let catalog = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK) /DestOutputProfile 5 0 R >>] >>\nendobj\n";
    buf.extend_from_slice(catalog.as_bytes());
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    let page = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /DefaultCMYK (not a colour space) >> >> /Contents 4 0 R >>\nendobj\n";
    buf.extend_from_slice(page.as_bytes());
    let stream_off = buf.len();
    let content = "0.25 0 0 0 k\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_off = buf.len();
    let icc_hdr = format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", icc.len());
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(&icc);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, icc_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );

    let doc = PdfDocument::from_bytes(buf).expect("open synthetic PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "fixture must declare a CMYK /OutputIntents — without it the test \
         can't distinguish the fall-through from the malformed path"
    );
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (oi_ref[0], oi_ref[1], oi_ref[2], 255),
        "Malformed /DefaultCMYK (string) must fall through to the document \
         /OutputIntents path; expected qcms reference {:?}; got \
         ({r},{g},{b},{a}). RGB(64, 64, 64) means the resolver honoured the \
         malformed entry via first_as_gray — silent mis-rendering of CMYK \
         paint as a literal-grey gradient.",
        oi_ref
    );
}

/// Pin that 1000 same-colour `/DeviceCMYK` paint operators on a single
/// page build the qcms `Transform` exactly once. This is the cache
/// hit-rate assertion the plan calls for: without caching every
/// `k`/`f` pair rebuilds the qcms transform (an 17×17×17×17 CLUT
/// precomputation that dominates the per-paint cost). With the cache
/// the first paint builds; the remaining 999 hit.
///
/// The build count comes from the `IccTransformCache`'s own counter
/// (`PageRenderer::icc_transform_cache_build_count`), gated on
/// `#[cfg(feature = "test-support")]`. Reading the per-instance
/// counter avoids racing other concurrent integration tests that
/// might also call `Transform::new_srgb_target` on the same process —
/// the cache is local to the `PageRenderer` we construct here, so
/// nobody else touches it.
///
/// **Why a counter instead of wall-clock duration:** wall-clock
/// measurements are noisy (CPU thermal state, OS scheduling, debug-vs-
/// release builds) and would conflate caching with unrelated perf
/// drift. A counter is exact: 1 build proves the cache works, N builds
/// proves it doesn't.
///
/// **Feature gate:** the per-cache build counter is exposed only when
/// the `test-support` feature is on (production builds carry zero
/// overhead); the test runs under
/// `cargo test --features rendering,icc,test-support`.
#[cfg(feature = "test-support")]
#[test]
fn output_intent_thousand_cmyk_paints_build_one_transform() {
    use pdf_oxide::rendering::{PageRenderer, RenderOptions};

    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let mut ops = String::new();
    for i in 0..1000 {
        let y = i % 100;
        ops.push_str(&format!("0.25 0 0 0 k\n0 {y} 1 1 re\nf\n"));
    }
    let catalog_entries =
        "/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (S) /DestOutputProfile 5 0 R >>]";
    let pdf = build_pdf_with_catalog_entries_and_content(catalog_entries, &ops, Some(&icc));
    let doc = PdfDocument::from_bytes(pdf).expect("open");

    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72));
    let _ = renderer.render_page(&doc, 0).expect("render");

    let built = renderer.icc_transform_cache_build_count();
    assert_eq!(
        built, 1,
        "1000 same-colour /DeviceCMYK paints under one /OutputIntents profile \
         and one rendering intent must build qcms::Transform exactly once \
         (cache miss on first paint, hit on the next 999). Built {built} times — \
         the per-page CMYK transform cache regressed or is missing."
    );
}

/// Pin that two different rendering intents on the same page +
/// OutputIntent split the cache into two entries — each intent gets
/// its own `Transform`. Critical because qcms's `Transform::new_to`
/// takes an intent parameter; even though qcms 0.3.0 currently
/// ignores that parameter for CMYK (see HONEST_GAP in the phase 8
/// section below), the cache key MUST include intent so a future qcms
/// upgrade that honours intent doesn't silently emit the wrong colour
/// from a shared transform.
///
/// The fixture interleaves two `ri` operators (rendering-intent
/// overrides) inside a single page's content stream. The PDF spec's
/// §10.7.3 `ri` operator sets the graphics-state rendering intent —
/// pdf_oxide parses this and the colour stage threads it through
/// `ctx.rendering_intent`. With two distinct intents seen on the
/// page, the cache holds two `Transform` instances (one per intent),
/// not one shared across both.
///
/// HONEST_GAP: this probe also pins that the `ri` operator dispatch
/// is wired through `gs.rendering_intent`. If a regression removed
/// the `Operator::SetRenderingIntent` arm from the page renderer,
/// every paint would resolve under the default
/// /RelativeColorimetric intent and the cache would collapse to one
/// entry — surfaced here as a count of 1 instead of 2.
#[cfg(feature = "test-support")]
#[test]
fn qa_round3_cache_keys_include_rendering_intent() {
    use pdf_oxide::rendering::{PageRenderer, RenderOptions};

    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);
    // Two paints: first under /RI /RelativeColorimetric (default),
    // second after switching to /RI /Perceptual. The `ri` operator
    // takes a name argument; both pin different cache keys.
    let ops = "0.25 0 0 0 k\n10 10 20 20 re\nf\n\
               /Perceptual ri\n\
               0.50 0 0 0 k\n40 10 20 20 re\nf\n";
    let catalog_entries =
        "/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (S) /DestOutputProfile 5 0 R >>]";
    let pdf = build_pdf_with_catalog_entries_and_content(catalog_entries, ops, Some(&icc));
    let doc = PdfDocument::from_bytes(pdf).expect("open");

    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72));
    let _ = renderer.render_page(&doc, 0).expect("render");

    let built = renderer.icc_transform_cache_build_count();
    assert_eq!(
        built, 2,
        "Two distinct rendering intents on one page + one OutputIntent profile \
         must split the transform cache into two entries — one per intent. \
         Built {built} times; expected exactly 2. A count of 1 means the \
         cache key drops the intent (incorrect — qcms's Transform::new_to \
         takes intent as a parameter even if 0.3.0 ignores it internally); \
         a count > 2 means a regression."
    );
}

/// Pin that 1000 same-colour bare `/DeviceRGB` paints routed through a
/// page-level `/DefaultRGB [/ICCBased N=3]` override build the qcms
/// `Transform` exactly once. The cache is keyed on
/// `(profile.content_hash(), intent)` — n_components-agnostic by
/// design — so an N=3 profile routed through the resolver MUST hit the
/// cache after the first build, exactly as the N=4 CMYK path does.
///
/// Without the cache wiring, every bare `rg` paint reparses the
/// embedded profile and rebuilds the qcms transform; the per-page
/// counter would land at 1000 instead of 1. This probe is the
/// structural counterpart of `output_intent_thousand_cmyk_paints_*`
/// for the N=3 arm of `resolve_iccbased`.
///
/// Fixture: a one-page PDF whose `/Resources /ColorSpace /DefaultRGB`
/// is `[/ICCBased <constant-CLUT N=3 LUT8>]` and whose content stream
/// emits 1000 identical `0.8 0.2 0.5 rg` + `re` + `f` triples.
#[cfg(feature = "test-support")]
#[test]
fn qa_round4_thousand_rgb_paints_through_default_rgb_build_one_transform() {
    use pdf_oxide::rendering::{PageRenderer, RenderOptions};

    let profile = build_minimal_rgb_to_lab_lut8_profile(PROFILE_B_TARGET_L_BYTE);
    let mut ops = String::new();
    for i in 0..1000 {
        let y = i % 100;
        ops.push_str(&format!("0.8 0.2 0.5 rg\n0 {y} 1 1 re\nf\n"));
    }

    // Fixture: bake the /DefaultRGB override into a one-page PDF
    // whose ICC stream carries the N=3 profile we just built. Mirror
    // the shape of `build_pdf_default_rgb_overrides_bare_device_rgb`
    // but parameterise the content stream so the 1000-paint loop
    // composes here.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    let page = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /DefaultRGB [/ICCBased 5 0 R] >> >> /Contents 4 0 R >>\nendobj\n";
    buf.extend_from_slice(page.as_bytes());
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", ops.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(ops.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_off = buf.len();
    let icc_hdr = format!("5 0 obj\n<< /N 3 /Length {} >>\nstream\n", profile.len());
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(&profile);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, icc_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );

    let doc = PdfDocument::from_bytes(buf).expect("open");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72));
    let _ = renderer.render_page(&doc, 0).expect("render");

    let built = renderer.icc_transform_cache_build_count();
    assert_eq!(
        built, 1,
        "1000 same-colour bare /DeviceRGB paints routed through a page-level \
         /DefaultRGB [/ICCBased N=3] override must build qcms::Transform \
         exactly once. Built {built} times — the N=3 arm of resolve_iccbased \
         is bypassing the per-page transform cache (a build_count of 1000 means \
         every paint reparsed the profile and rebuilt the transform; the cache \
         exists to amortise this)."
    );
}

// ===========================================================================
// Phase 8: rendering-intent dispatch — qcms 0.3.0 limitation pin
// ===========================================================================
//
// ISO 32000-1:2008 §10.7.3 specifies per-paint rendering intent through
// the `/RI` operator. The graphics-state field flows through
// `gs.rendering_intent` → `ctx.rendering_intent` → `Transform::new_srgb_target`'s
// intent parameter → qcms.
//
// **Research surface: qcms 0.3.0 intent dispatch.** Reading
// `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/qcms-0.3.0/src/transform.rs`:
//   - line 1283: `pub fn transform_create(input, in_type, output, out_type,
//     _intent: Intent)` — the underscore prefix marks the parameter as
//     intentionally unused. qcms 0.3.0 discards the intent.
//   - line 1245: `fn transform_precacheLUT_cmyk_float(transform, input,
//     output, samples, in_type)` — the CMYK CLUT precomputation takes no
//     intent; the same CLUT is produced regardless of caller intent.
//   - The crate's `lib.rs` Intent enum docstring (line 32) notes that
//     "BPC brings an unacceptable performance overhead, so we go with
//     perceptual" — qcms 0.3.0 has no Black Point Compensation flag at
//     all. The §10.7.3 `/AbsoluteColorimetric` BPC-off behaviour cannot
//     be expressed through qcms 0.3.0.
//
// **HONEST_GAP_QCMS_INTENT_IGNORED**: per-channel intent dispatch IS
// wired through `ctx.rendering_intent` → `Transform::new_srgb_target`
// at the pdf_oxide layer, and the cache key separates entries by intent
// — but qcms 0.3.0 silently drops intent inside transform construction.
// Distinct intents produce byte-identical CMYK→RGB conversions through
// qcms 0.3.0; a future qcms upgrade that honours intent will surface the
// difference automatically because the dispatch chain already routes the
// per-intent value through. The cache invalidation guarantee holds:
// distinct intents get distinct Transform instances (so when qcms starts
// honouring intent, no shared-Transform cross-contamination happens).
//
// The probes below pin three claims:
//   1. `gs.rendering_intent` flows end-to-end into the cache key (covered
//      by `qa_round3_cache_keys_include_rendering_intent` above).
//   2. qcms 0.3.0's intent-invariance for CMYK conversions is observable
//      via the constant-CLUT fixture — same input, four intents, same
//      RGB. Documents the qcms version constraint.
//   3. The default intent fallback (§8.6.5.8: unrecognised intent →
//      /RelativeColorimetric) is honoured at the colour boundary.

/// Pin qcms 0.3.0's documented intent-invariance for CMYK profiles:
/// the same CMYK input under all four `RenderingIntent` values
/// produces byte-identical RGB through `Transform::convert_cmyk_pixel`.
///
/// **Why this probe is GREEN immediately:** qcms 0.3.0's
/// `transform_create` ignores the intent parameter (see the module
/// docstring above). A future qcms upgrade that honours intent for
/// CMYK→sRGB conversions would flip this probe to RED, surfacing the
/// behaviour change at upgrade time so a CHANGELOG entry can document
/// the §10.7.3 intent dispatch becoming externally observable.
///
/// **Constructed fixture caveat:** the constant-CLUT profile used here
/// produces the same RGB for every CMYK input by design — that's how
/// the test pin stays unambiguous. A real-world CMYK profile (CoatedFOGRA39
/// etc.) carries an intent-dependent CLUT that WOULD vary across intents
/// IF qcms honoured them. Synthesising such a fixture requires a real
/// CMM toolchain (curves + matrices + a true 4D CLUT) — deferred as
/// HONEST_GAP_INTENT_SENSITIVE_FIXTURE.
#[test]
fn qa_round3_qcms_030_treats_cmyk_intent_as_informational() {
    use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
    use std::sync::Arc;

    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let profile = Arc::new(IccProfile::parse(icc, 4).expect("parse"));

    let intents = [
        RenderingIntent::Perceptual,
        RenderingIntent::RelativeColorimetric,
        RenderingIntent::Saturation,
        RenderingIntent::AbsoluteColorimetric,
    ];

    // CMYK(64, 0, 0, 0) is the round-1 byte-exact pin: target_l_byte=135
    // → RGB(126, 126, 126) under qcms 0.3.0 at every intent.
    let mut results = Vec::new();
    for intent in intents {
        let t = Transform::new_srgb_target(Arc::clone(&profile), intent);
        assert!(t.has_cmm(), "intent {intent:?} must build a real CMM");
        let rgb = t.convert_cmyk_pixel(64, 0, 0, 0);
        results.push((intent, rgb));
    }
    // All four results must be identical under qcms 0.3.0. If this
    // asserts differently in a future qcms version, the intent
    // dispatch became externally observable.
    let first = results[0].1;
    for (intent, rgb) in &results[1..] {
        assert_eq!(
            *rgb, first,
            "qcms 0.3.0 must produce intent-invariant CMYK→RGB. Intent {intent:?} \
             produced {rgb:?} while {:?} produced {first:?}. \
             If this fires after a qcms upgrade, intent dispatch is now \
             externally observable — update the HONEST_GAP_QCMS_INTENT_IGNORED \
             documentation and remove this pin.",
            results[0].0
        );
    }
    assert_eq!(
        first,
        [126u8, 126, 126],
        "byte-exact reference: target_l_byte=135 → (126, 126, 126) at every \
         intent through qcms 0.3.0"
    );
}

/// Pin that the §8.6.5.8 "unrecognised intent → /RelativeColorimetric"
/// fallback is honoured at the colour boundary. A PDF that sets
/// `/RI /UnknownVendorPrivateIntent` must map to
/// `RenderingIntent::RelativeColorimetric`, NOT to an arbitrary
/// default.
///
/// This is a unit-level pin against the spec's defaulting rule. The
/// cache key includes intent, so under the failing scenario (a
/// regression that mapped unknown names to /Perceptual or /Saturation)
/// a same-named intent would silently produce a different cache entry
/// — surfaced here by asserting the from_pdf_name behaviour directly.
#[test]
fn qa_round3_unknown_intent_name_falls_back_to_relative_colorimetric() {
    use pdf_oxide::color::RenderingIntent;

    // §8.6.5.8: unrecognised → RelativeColorimetric.
    assert_eq!(
        RenderingIntent::from_pdf_name("UnknownVendorPrivateIntent"),
        RenderingIntent::RelativeColorimetric,
        "unknown intent name must fall back to /RelativeColorimetric per §8.6.5.8"
    );
    // Empty string is the implicit "no /RI ever ran" case — same
    // default.
    assert_eq!(
        RenderingIntent::from_pdf_name(""),
        RenderingIntent::RelativeColorimetric,
        "empty intent name must fall back to /RelativeColorimetric"
    );
    // Each of the four named intents must round-trip cleanly.
    assert_eq!(RenderingIntent::from_pdf_name("Perceptual"), RenderingIntent::Perceptual);
    assert_eq!(RenderingIntent::from_pdf_name("Saturation"), RenderingIntent::Saturation);
    assert_eq!(
        RenderingIntent::from_pdf_name("RelativeColorimetric"),
        RenderingIntent::RelativeColorimetric
    );
    assert_eq!(
        RenderingIntent::from_pdf_name("AbsoluteColorimetric"),
        RenderingIntent::AbsoluteColorimetric
    );
}

/// HONEST_GAP_INTENT_SENSITIVE_FIXTURE: a probe that WOULD pin
/// /Perceptual vs /AbsoluteColorimetric producing different RGB.
/// Requires:
///   1. An intent-sensitive CMYK profile (real CoatedFOGRA39 or
///      equivalent — synthetic profiles with constant CLUTs are
///      intent-invariant by construction).
///   2. A qcms version that honours the intent parameter for CMYK
///      transforms (qcms 0.3.0 does not, per the module docstring).
///
/// Neither prerequisite is satisfied today. The probe stays
/// `#[ignore]`-ed with the HONEST_GAP marker so a future engineer
/// adding either prerequisite has a ready-made integration test to
/// turn on.
#[test]
#[ignore = "HONEST_GAP_INTENT_SENSITIVE_FIXTURE: needs (a) intent-sensitive \
            CMYK profile + (b) qcms version that honours intent for CMYK"]
fn output_intent_perceptual_vs_absolute_colorimetric_produces_different_rgb() {
    panic!(
        "HONEST_GAP_INTENT_SENSITIVE_FIXTURE: qcms 0.3.0 ignores intent for \
         CMYK conversions (transform.rs:1288 `_intent: Intent`). To enable \
         this probe a future engineer needs: (1) a CMYK ICC profile with \
         genuine intent-dependent behaviour (synthetic constant-CLUT \
         fixtures are intent-invariant by construction); (2) a qcms upgrade \
         that honours intent for CMYK. The wiring chain is already in \
         place — gs.rendering_intent → ctx.rendering_intent → \
         Transform::new_srgb_target — so flipping qcms versions WILL surface \
         the intent dispatch without further code changes."
    );
}

// ===========================================================================
// QA: TDD-discipline verification report (inline docstring)
// ===========================================================================

/// TDD-discipline verification report for round-1 OutputIntent foundation.
///
/// Verified by checking out the round-1 commit graph in a throwaway
/// worktree and re-running the failing/passing tests at the relevant
/// SHAs. Captured here so a future reader has the audit trail without
/// having to re-do the bisect.
///
/// **Failing test commit `eab4040`:**
/// Planting `tests/test_render_output_intent.rs` from `eab4040` onto
/// its parent `65063ba` (last `feat` commit before the impl landed)
/// produced:
///
/// ```text
/// thread 'device_cmyk_paint_with_output_intent_renders_via_icc_not_additive_clamp'
///   panicked at tests/test_render_output_intent.rs:365:5:
/// OutputIntent /DeviceCMYK paint expected qcms-converted RGB ~(128, 128, 128);
/// got (191, 255, 255). RGB(191, 255, 255) would mean the §10.3.5 additive-clamp
/// fallback fired — the resolver is not consulting ctx.output_intent_cmyk.
/// test result: FAILED. 0 passed; 1 failed
/// ```
///
/// Checking out the impl commit `656c119` then produced:
///
/// ```text
/// test device_cmyk_paint_with_output_intent_renders_via_icc_not_additive_clamp ... ok
/// test result: ok. 1 passed; 0 failed
/// ```
///
/// **Negative-pin commit `fda9b6f`:**
/// The negative pin (`*_without_output_intent_renders_additive_clamp`)
/// is a regression guard, not a failing test. Verified by planting the
/// commit's test on its parent `656c119`: it passed even there because
/// the no-OutputIntent fallback was the shipped behaviour. The impl
/// agent's report categorised this honestly as a "negative pin", and
/// the actual test categorisation matches.
///
/// **Conclusion:** TDD discipline was followed for the positive ICC
/// path. The negative pin is correctly described as a regression guard.
#[test]
fn qa_tdd_discipline_verification_report() {
    // Marker test — its docstring carries the verification narrative;
    // the body just confirms the integration suite is still compilable
    // by referencing the two test functions whose behaviour the report
    // describes.
    let _ = device_cmyk_paint_with_output_intent_renders_via_icc_not_additive_clamp;
    let _ = device_cmyk_paint_without_output_intent_renders_additive_clamp;
}

// ===========================================================================
// Phase 4: embedded /ICCBased N=4 trumps document /OutputIntents
// ===========================================================================
//
// ISO 32000-1:2008 §8.6.5.5 (and §14.11.5): an `/ICCBased` colour space
// carries its own `DestOutputProfile`-equivalent stream; that stream IS
// the conversion source, and the document-level `/OutputIntents` profile
// is only the default for `/DeviceCMYK` paint that lacks any embedded
// override. Embedded ICC always wins.
//
// The byte-exact references baked into the assertions below come from
// the discovery harness (run once, output captured) — see the plan
// errata. They are intent-invariant because the synthesised LUT8
// profile uses a constant CLUT.

/// Byte-exact qcms 0.3.0 reference for the `target_l_byte=200` profile
/// at CMYK(64,0,0,0) under RelativeColorimetric (intent-invariant by
/// construction). Distinct from the round-1 profile A reference of
/// (126,126,126) so the precedence assertion is unambiguous.
const PROFILE_B_TARGET_L_BYTE: u8 = 200;
const PROFILE_B_RGB_AT_FIXTURE_INPUT: (u8, u8, u8) = (194, 194, 194);

/// Pin that an `/ICCBased` N=4 colour space paint operator routes through
/// the colour-space-embedded profile B and NOT through the document-level
/// `/OutputIntents` profile A.
///
/// Fixture geometry:
///   - Catalog declares /OutputIntents → profile A (target_l_byte=135 →
///     qcms reference RGB(126,126,126)).
///   - Page Resources /ColorSpace /CS1 → [/ICCBased <stream B>] where
///     profile B has target_l_byte=200 → qcms reference RGB(194,194,194).
///   - Content stream: `/CS1 cs   0.25 0 0 0 scn   20 20 60 60 re   f`.
///
/// Spec rule: §8.6.5.5 — the ICCBased colour space carries the conversion
/// source and overrides any document-level default. The renderer must
/// route the four `scn` components through profile B's qcms transform.
///
/// What this test catches:
///   - If the rendered pixel is (126,126,126), profile A won — the
///     embedded ICC route is being shadowed by the OutputIntent route
///     (the spec-precedence bug this phase exists to fix).
///   - If the rendered pixel is (191,255,255), neither profile was
///     consulted and §10.3.5 additive-clamp fired (an even worse
///     regression).
///   - If the rendered pixel is (194,194,194), profile B's CMM
///     compiled-and-ran through `Transform::convert_cmyk_pixel` and the
///     precedence is correct.
#[test]
fn embedded_iccbased_n4_trumps_document_output_intent() {
    let profile_a = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let profile_b = build_minimal_cmyk_to_rgb_lut8_profile(PROFILE_B_TARGET_L_BYTE);

    // Sanity-pin both profiles compile through qcms and produce the
    // expected byte-exact references. Without this gate a regression
    // that broke profile B's transform would make the integration
    // assertion below fire for the wrong reason.
    {
        use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
        use std::sync::Arc;
        let prof_a = Arc::new(IccProfile::parse(profile_a.clone(), 4).expect("parse A"));
        let prof_b = Arc::new(IccProfile::parse(profile_b.clone(), 4).expect("parse B"));
        let t_a = Transform::new_srgb_target(prof_a, RenderingIntent::RelativeColorimetric);
        let t_b = Transform::new_srgb_target(prof_b, RenderingIntent::RelativeColorimetric);
        assert_eq!(
            t_a.convert_cmyk_pixel(64, 0, 0, 0),
            [126u8, 126, 126],
            "profile A reference must be (126,126,126); fixture is invalid otherwise"
        );
        assert_eq!(
            t_b.convert_cmyk_pixel(64, 0, 0, 0),
            [194u8, 194, 194],
            "profile B reference must be (194,194,194); fixture is invalid otherwise"
        );
    }

    let pdf = build_pdf_embedded_iccbased_with_different_output_intent(&profile_a, &profile_b);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    // Cross-check the OutputIntent accessor sees profile A. If it didn't
    // the test would conflate "OI not seen" with "OI seen but bypassed
    // for embedded ICC" — both produce the expected pixel but only the
    // latter actually probes the precedence we care about.
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "fixture must declare a CMYK OutputIntent so the precedence is actually contested"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    let (br, bg, bb) = PROFILE_B_RGB_AT_FIXTURE_INPUT;
    assert_eq!(
        (r, g, b, a),
        (br, bg, bb, 255),
        "embedded /ICCBased profile B must take precedence over /OutputIntents \
         profile A on CMYK paint through the ICCBased space; expected B's qcms \
         reference {:?}; got ({r},{g},{b},{a}). (126,126,126,_) means profile A won \
         — the spec precedence (§8.6.5.5) is inverted. (191,255,255,_) means neither \
         profile was consulted and §10.3.5 fired.",
        (br, bg, bb, 255u8)
    );
}

// ===========================================================================
// Phase 5: Separation / DeviceN with DeviceCMYK alternate routes through OutputIntent
// ===========================================================================
//
// ISO 32000-1:2008 §8.6.6.3 (Separation) and §8.6.6.4 (DeviceN): when the
// device lacks the named colorant plate, the colour is approximated via
// the alternate colour space and the tint transform. When the alternate
// is /DeviceCMYK, the alternate's CMYK quadruple is then converted to
// RGB for the composite output path — and that conversion MUST honour
// the document /OutputIntents profile, since composite output is the
// "viewer's screen" surface the OutputIntent describes.
//
// Today (post-round-1) the resolver's
// `resolve_separation_or_devicen` arm dispatches a CMYK-alternate
// result through `four_as_cmyk(&altspace_values, alpha, ctx)`, which
// itself calls `cmyk_to_rgb_via_intent` — the same OutputIntent-aware
// helper the bare /DeviceCMYK paint path consumes. So the routing is
// already correct, but the probes below pin it byte-for-byte so a
// regression that detoured Separation/DeviceN through a non-context-
// aware CMYK→RGB path would surface immediately.
//
// These probes are categorised as REGRESSION GUARDS in the TDD-discipline
// sense (they pass at HEAD without code changes) because the routing
// landed during round-1 phase 2. The TDD-failing-test→implementation
// pair for this behaviour is documented at fa1b947's prior history
// (round-1 phase 2). The probes here lock the routing for the
// specifically named Separation Type-4 and DeviceN Type-4 cases the
// plan body called out.
//
// Discrimination audit: before committing the probes, the impl agent
// temporarily flipped `four_as_cmyk` in src/rendering/resolution/color.rs
// to bypass `cmyk_to_rgb_via_intent` and call bare `cmyk_to_rgb` (the
// §10.3.5 helper) instead. With that flip, both
// `*_composite_routes_through_output_intent` probes failed with the
// expected (255, 0, 255, 255) value, demonstrating they actively
// discriminate between "OutputIntent honoured" and "additive-clamp
// fallback". The flip was reverted before the commit landed; the audit
// confirms the probes do what their names say.

/// Pin that a `/Separation /MagentaSpot /DeviceCMYK <Type-4 tint
/// transform>` paint operator's composite-side RGBA is the document
/// `/OutputIntents` profile's conversion of the tint-transform's CMYK
/// output — NOT the §10.3.5 additive-clamp of that CMYK quadruple.
///
/// Fixture: tint transform `{ 0.0 exch 0.0 0.0 }` produces CMYK(0, tint,
/// 0, 0). At tint=1.0 the alternate-CMYK value is (0, 1, 0, 0); §10.3.5
/// of that is RGB(255, 0, 255) (magenta). The OutputIntent profile
/// (constant-CLUT, target_l_byte=135) maps every CMYK input to
/// RGB(126, 126, 126), so an OutputIntent-honouring composite pixel is
/// (126, 126, 126).
///
/// Three observable outcomes:
///   - (126, 126, 126, 255): composite routed through OutputIntent — pass.
///   - (255, 0, 255, 255): composite ran §10.3.5 directly — fail
///     (alt-CMYK projection bypassed `cmyk_to_rgb_via_intent`).
///   - any other RGB: tint transform or qcms behaviour drifted.
#[test]
fn separation_type4_alt_devicecmyk_composite_routes_through_output_intent() {
    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);
    // Sanity-pin the OutputIntent reference for CMYK(0, 255, 0, 0) —
    // intent-invariant by construction (constant CLUT) so a single
    // intent is enough.
    {
        use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
        use std::sync::Arc;
        let prof = Arc::new(IccProfile::parse(icc.clone(), 4).expect("parse"));
        let t = Transform::new_srgb_target(prof, RenderingIntent::RelativeColorimetric);
        let rgb = t.convert_cmyk_pixel(0, 255, 0, 0);
        assert_eq!(
            rgb,
            [126u8, 126, 126],
            "OutputIntent profile must map CMYK(0,255,0,0) to (126,126,126); \
             fixture is invalid otherwise (got {rgb:?})"
        );
    }

    let pdf = build_pdf_separation_type4_devicecmyk_with_output_intent(&icc);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "fixture must declare a CMYK OutputIntent for the routing to be probed"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (126u8, 126, 126, 255),
        "Separation Type-4 /DeviceCMYK alternate must route the alt-CMYK \
         quadruple through the document /OutputIntents profile on the \
         composite path; expected (126,126,126,255); got ({r},{g},{b},{a}). \
         (255,0,255,_) means the §10.3.5 additive-clamp of CMYK(0,1,0,0) \
         fired — the resolver bypassed cmyk_to_rgb_via_intent for the \
         Separation alt-CMYK projection."
    );
}

/// Counter-pin: with no `/OutputIntents` declared, the same Separation
/// Type-4 alt-CMYK paint MUST produce the §10.3.5 additive-clamp value
/// for CMYK(0, 1, 0, 0) = RGB(255, 0, 255).
///
/// The positive pin above asserts "OutputIntent wins on composite when
/// present"; this counter-pin asserts "no-OutputIntent → §10.3.5
/// preserved byte-for-byte" — i.e. the OutputIntent route doesn't leak
/// into a no-OutputIntent fixture (which would imply some hard-coded
/// CMM hung around the renderer rather than the configured route).
#[test]
fn separation_type4_alt_devicecmyk_without_output_intent_renders_additive_clamp() {
    // Inline-build a PDF identical to
    // `build_pdf_separation_type4_devicecmyk_with_output_intent` but
    // without /OutputIntents. Object IDs shift down by one because the
    // ICC stream is dropped: catalog → pages → page → content → tint
    // (obj 5 instead of obj 6).
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    let page = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /CS1 [/Separation /MagentaSpot /DeviceCMYK 5 0 R] >> >> /Contents 4 0 R >>\nendobj\n";
    buf.extend_from_slice(page.as_bytes());

    let stream_off = buf.len();
    let content = "/CS1 cs\n1.0 scn\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let tint_off = buf.len();
    let tint_program: &[u8] = b"{ 0.0 exch 0.0 0.0 }";
    let tint_hdr = format!(
        "5 0 obj\n<< /FunctionType 4 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\nstream\n",
        tint_program.len()
    );
    buf.extend_from_slice(tint_hdr.as_bytes());
    buf.extend_from_slice(tint_program);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    let obj_count = 6;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, tint_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );

    let doc = PdfDocument::from_bytes(buf).expect("open synthetic PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_none(),
        "fixture must declare no /OutputIntents for the counter-pin to actually contest the route"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (255u8, 0, 255, 255),
        "Separation Type-4 /DeviceCMYK alternate without /OutputIntents must \
         fall through to §10.3.5 additive-clamp of CMYK(0,1,0,0) = (255,0,255); \
         got ({r},{g},{b},{a})"
    );
}

/// Pin that a 2-colorant `/DeviceN [/Magenta /Cyan] /DeviceCMYK
/// <Type-4 tint transform>` paint operator's composite-side RGBA is
/// also routed through the document `/OutputIntents` profile when the
/// tint transform's alternate-CMYK output lands in the resolver.
///
/// Fixture: tint transform `{ pop 0.0 exch 0.0 0.0 }` consumes the two
/// colorant tints, drops the second, and emits CMYK(0, tint0, 0, 0).
/// Content `1.0 0.5 scn` provides (tint0=1.0, tint1=0.5) → alternate
/// CMYK(0, 1, 0, 0). The OutputIntent profile maps that to
/// RGB(126, 126, 126); the §10.3.5 additive-clamp value would be
/// RGB(255, 0, 255).
#[test]
fn devicen_type4_alt_devicecmyk_composite_routes_through_output_intent() {
    let icc = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let pdf = build_pdf_devicen_type4_devicecmyk_with_output_intent(&icc);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "fixture must declare a CMYK OutputIntent for the routing to be probed"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (126u8, 126, 126, 255),
        "DeviceN Type-4 /DeviceCMYK alternate must route the alt-CMYK \
         quadruple through the document /OutputIntents profile on the \
         composite path; expected (126,126,126,255); got ({r},{g},{b},{a}). \
         (255,0,255,_) means §10.3.5 additive-clamp fired."
    );
}

// ===========================================================================
// QA round-2 edge probes
// ===========================================================================
//
// Round-2 phase 4 changed `resolve_iccbased` for N=4 with parseable
// embedded profile to emit `ResolvedColor::Rgba` directly (bypassing
// OutputIntent). Edge cases the impl probes did not cover:
//
//   1. Embedded profile parses but qcms refuses to build a CMM
//      (`has_cmm() == false`) — fallback path must kick in.
//   2. Embedded profile is malformed bytes (`IccProfile::parse` returns
//      None) — fallback path must kick in.
//   3. ICCBased N=3 (RGB) with a document CMYK /OutputIntents — no
//      interaction; RGB paint stays untouched.
//   4. ICCBased N=1 (gray) with a document CMYK /OutputIntents — same.
//   5. ICCBased N=4 paint inside a Form XObject — precedence survives
//      the Form scope.
//   6. **Per-plate regression**: the fix changes ICCBased N=4 from
//      `ResolvedColor::Cmyk` to `ResolvedColor::Rgba`; per-plate
//      consumers route by participating channels and `Rgba` produces an
//      empty participating list. Probe what happens when the renderer
//      is invoked for separations on the same fixture.

/// Build a minimal "valid header but no usable tags" ICC profile: passes
/// `IccProfile::parse`'s header / `acsp` / `/N` cross-check but qcms's
/// `Profile::new_from_slice` rejects it (no `A2B0`, no matrix/curve
/// tags), so `Transform::has_cmm()` returns false. Used to verify the
/// fallback path in `resolve_iccbased` kicks in cleanly.
fn build_iccbased_header_only_cmyk_profile() -> Vec<u8> {
    let mut profile = vec![0u8; 128];
    // Profile size at bytes 0..4. Header-only + 4-byte tag count of 0.
    let total: u32 = 128 + 4;
    profile[0..4].copy_from_slice(&total.to_be_bytes());
    profile[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes());
    profile[12..16].copy_from_slice(b"prtr");
    profile[16..20].copy_from_slice(b"CMYK");
    profile[20..24].copy_from_slice(b"Lab ");
    profile[36..40].copy_from_slice(b"acsp");
    profile[64..68].copy_from_slice(&0u32.to_be_bytes());
    profile[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes());
    profile[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    profile[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes());
    // Tag count = 0 — no tags at all.
    profile.extend_from_slice(&0u32.to_be_bytes());
    profile
}

/// Sanity-pin: the header-only profile parses through `IccProfile::parse`
/// but produces a transform with no CMM. This is the precondition that
/// makes the `resolve_iccbased` fallback path observable: if either
/// branch flipped (parse failed OR has_cmm became true) the edge probes
/// below would conflate two failure modes.
#[test]
fn qa_round2_header_only_cmyk_profile_parses_without_cmm() {
    use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
    use std::sync::Arc;
    let bytes = build_iccbased_header_only_cmyk_profile();
    let prof = IccProfile::parse(bytes, 4).expect(
        "header-only profile should pass IccProfile::parse — only IccHeader::parse and /N \
         cross-check run there",
    );
    let prof = Arc::new(prof);
    let t = Transform::new_srgb_target(prof, RenderingIntent::RelativeColorimetric);
    assert!(
        !t.has_cmm(),
        "header-only profile must NOT compile to a usable qcms CMM; otherwise the fallback \
         path in resolve_iccbased can't be probed"
    );
}

/// Embedded /ICCBased N=4 whose profile parses through
/// `IccProfile::parse` but is rejected by qcms (`has_cmm() == false`).
/// `resolve_iccbased` must fall through to the device-family hint, which
/// emits `ResolvedColor::Cmyk` for N=4, which the composite projection
/// then runs through `cmyk_to_rgb_via_intent` against the document
/// /OutputIntents profile. Expected pixel: (126, 126, 126, 255) — the
/// OutputIntent profile A's constant CLUT.
#[test]
fn qa_round2_iccbased_n4_no_cmm_falls_through_to_output_intent() {
    let profile_a = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let profile_b_no_cmm = build_iccbased_header_only_cmyk_profile();
    let pdf =
        build_pdf_embedded_iccbased_with_different_output_intent(&profile_a, &profile_b_no_cmm);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (126u8, 126, 126, 255),
        "embedded ICCBased N=4 whose profile parses but has no CMM must fall through to \
         the device-family path → ResolvedColor::Cmyk → cmyk_to_rgb_via_intent → \
         document /OutputIntents profile A reference (126,126,126,255); got \
         ({r},{g},{b},{a}). (191,255,255,_) means §10.3.5 additive-clamp fired (fallback \
         path bypassed OutputIntent). (194,194,194,_) means the embedded profile's CMM \
         compiled (precondition pin was wrong)."
    );
}

/// Embedded /ICCBased N=4 with garbage bytes (no valid `acsp` header).
/// `IccProfile::parse` returns None, so the fallback path emits
/// `ResolvedColor::Cmyk` → routed through `cmyk_to_rgb_via_intent` →
/// document /OutputIntents.
#[test]
fn qa_round2_iccbased_n4_unparseable_bytes_fall_through_to_output_intent() {
    let profile_a = build_minimal_cmyk_to_rgb_lut8_profile(135);
    // 128 zero bytes — no `acsp` signature at bytes 36..40 → parse fails.
    let garbage = vec![0u8; 128];
    let pdf = build_pdf_embedded_iccbased_with_different_output_intent(&profile_a, &garbage);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (126u8, 126, 126, 255),
        "embedded ICCBased N=4 with unparseable bytes must fall through to the \
         device-family path → ResolvedColor::Cmyk → document /OutputIntents \
         (126,126,126,255); got ({r},{g},{b},{a})."
    );
}

/// **Per-plate routing under embedded-ICCBased N=4.** §8.6.5.5 says the
/// ICCBased profile is the conversion source — but for *composite* output
/// only. The per-plate output is the press-target ink coverage: the four
/// raw CMYK components are exactly what the C/M/Y/K plates must carry.
/// Stripping the channel decomposition because the composite path
/// happens to want ICC-converted RGB would make plate output unusable
/// for prepress workflows shipping packaging artwork with embedded-ICC-
/// tagged CMYK.
///
/// Fixture paints `0.25 0 0 0 scn` against an embedded ICCBased N=4
/// space. The cyan plate must carry the 0.25 tint (≈ 63 in byte space —
/// the renderer's f32→u8 path produces the same value as the bare
/// `/DeviceCMYK` counter-pin below); magenta/yellow/black plates must
/// stay at zero.
///
/// If the cyan plate is zero, the per-plate path saw `ResolvedColor::Rgba`
/// from the resolver, the `OverprintResolver` produced an empty
/// participating list, and `InkRouter` returned `Skip` for every plate
/// — that's the regression vector. The resolver must emit a variant that
/// carries BOTH the composite-side ICC-converted RGB (for §8.6.5.5) and
/// the per-plate CMYK decomposition (for press output).
#[test]
fn qa_round2_iccbased_n4_with_embedded_profile_routes_cmyk_to_plates() {
    use pdf_oxide::rendering::render_separations;
    let profile_a = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let profile_b = build_minimal_cmyk_to_rgb_lut8_profile(200);
    let pdf = build_pdf_embedded_iccbased_with_different_output_intent(&profile_a, &profile_b);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    let plates = render_separations(&doc, 0, 72).expect("render_separations");
    let sample = |p: &pdf_oxide::rendering::SeparationPlate| {
        let w = p.width as usize;
        p.data[50 * w + 50]
    };
    let by_name = |name: &str| {
        plates
            .iter()
            .find(|p| p.ink_name == name)
            .map(sample)
            .unwrap_or_else(|| panic!("plate {name} should be present in the result"))
    };
    let cyan = by_name("Cyan");
    assert!(
        (60..=68).contains(&cyan),
        "Cyan plate should carry the ~0.25 tint from `0.25 0 0 0 scn` through the \
         embedded /ICCBased N=4 space — same per-plate path the bare /DeviceCMYK \
         counter-pin exercises (renderer quantises to ~63). Got {cyan}. \
         Zero means the embedded-ICC arm of resolve_iccbased dropped the CMYK \
         channel decomposition: the per-plate `OverprintResolver` saw \
         `ResolvedColor::Rgba` and produced an empty participating list, so \
         `InkRouter` returned `Skip` for every plate. The fix is to emit a \
         variant that carries both the composite-side ICC-converted RGB AND \
         the original CMYK quadruple for the per-plate router."
    );
    assert_eq!(by_name("Magenta"), 0, "Magenta plate should be zero for `0.25 0 0 0`");
    assert_eq!(by_name("Yellow"), 0, "Yellow plate should be zero for `0.25 0 0 0`");
    assert_eq!(by_name("Black"), 0, "Black plate should be zero for `0.25 0 0 0`");
}

/// Counter-pin: bare /DeviceCMYK paint with no embedded ICC override
/// continues to produce per-plate coverage as before. This guards
/// against a regression where the round-2 fix accidentally widened to
/// the bare /DeviceCMYK arm too.
#[test]
fn qa_round2_bare_devicecmyk_paint_still_produces_separation_coverage() {
    use pdf_oxide::rendering::render_separations;
    let pdf = build_pdf_cmyk_without_output_intent();
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    let plates = render_separations(&doc, 0, 72).expect("render_separations");
    let sample = |p: &pdf_oxide::rendering::SeparationPlate| {
        let w = p.width as usize;
        p.data[50 * w + 50]
    };
    let by_name = |name: &str| {
        plates
            .iter()
            .find(|p| p.ink_name == name)
            .map(sample)
            .unwrap_or(0)
    };
    // The renderer's f32→u8 path produces 63 for tint=0.25; the
    // important point is non-zero ink coverage at the painted pixel —
    // proving the bare DeviceCMYK arm still routes through the
    // per-plate `ResolvedColor::Cmyk` decomposition.
    let cyan = by_name("Cyan");
    assert!(
        (60..=68).contains(&cyan),
        "Cyan plate should carry the ~0.25 tint from `0.25 0 0 0 k` (renderer quantises \
         to ~63). Got {cyan}. If zero the bare DeviceCMYK arm regressed too."
    );
    assert_eq!(by_name("Magenta"), 0, "Magenta should be zero");
    assert_eq!(by_name("Yellow"), 0, "Yellow should be zero");
    assert_eq!(by_name("Black"), 0, "Black should be zero");
}

/// ICCBased **N=3** (RGB) with a document CMYK /OutputIntents declared:
/// the OutputIntent applies only to CMYK conversion paths per §8.6.5.5;
/// an RGB ICCBased space neither consults nor cares about the document
/// OutputIntent. Pixel at the painted rect = direct sRGB-like
/// pass-through of the 3 components (fallback path; the device-family
/// hint at /N=3 emits `three_as_rgb`).
#[test]
fn qa_round2_iccbased_n3_with_cmyk_output_intent_ignores_output_intent() {
    // Build a one-page PDF that declares a CMYK OutputIntent and paints
    // a 3-component ICCBased rectangle. We use the existing builder for
    // embedded-ICCBased fixtures but swap the colour-space dict's /N to
    // 3 and use a 3-component `scn`. Easier: reuse
    // `build_pdf_with_catalog_entries_and_content` and inline an
    // ICCBased[3] resource via a custom catalog.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK) /DestOutputProfile 5 0 R >>] >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /CS1 [/ICCBased 6 0 R] >> >> /Contents 4 0 R >>\nendobj\n");
    let stream_off = buf.len();
    // Paint with RGB(0.5, 0.25, 0.75) via the 3-component ICCBased.
    let content = "/CS1 cs\n0.5 0.25 0.75 scn\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_a = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let icc_a_off = buf.len();
    let icc_a_hdr = format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", icc_a.len());
    buf.extend_from_slice(icc_a_hdr.as_bytes());
    buf.extend_from_slice(&icc_a);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    // ICCBased N=3 stream — we don't need a valid qcms-compilable profile
    // for the N=3 case because the device-family hint at N=3 in
    // resolve_iccbased emits three_as_rgb directly (the embedded-ICC
    // branch is gated on N=4). Just declare /N 3 with empty stream
    // bytes; parse will fail (no acsp) and the fallback path fires.
    let icc_b_off = buf.len();
    let bogus = vec![0u8; 128];
    let icc_b_hdr = format!("6 0 obj\n<< /N 3 /Length {} >>\nstream\n", bogus.len());
    buf.extend_from_slice(icc_b_hdr.as_bytes());
    buf.extend_from_slice(&bogus);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    let obj_count = 7;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [
        cat_off, pages_off, page_off, stream_off, icc_a_off, icc_b_off,
    ] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    let doc = PdfDocument::from_bytes(buf).expect("open synthetic PDF");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    // The renderer's f32→u8 round produces 128 / 64 / 191 for the
    // (0.5, 0.25, 0.75) triple.
    assert_eq!(
        (r, g, b, a),
        (128u8, 64, 191, 255),
        "ICCBased N=3 with document CMYK /OutputIntents declared must pass the three \
         components through unchanged — the OutputIntent applies only to CMYK \
         conversion paths. Got ({r},{g},{b},{a})."
    );
}

/// ICCBased **N=1** (gray) with a document CMYK /OutputIntents declared:
/// same as N=3 — no spec interaction. Fallback path at N=1 emits
/// `first_as_gray`, so a single-component paint of 0.5 produces
/// RGB(128,128,128).
#[test]
fn qa_round2_iccbased_n1_with_cmyk_output_intent_ignores_output_intent() {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK) /DestOutputProfile 5 0 R >>] >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /CS1 [/ICCBased 6 0 R] >> >> /Contents 4 0 R >>\nendobj\n");
    let stream_off = buf.len();
    let content = "/CS1 cs\n0.5 scn\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_a = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let icc_a_off = buf.len();
    let icc_a_hdr = format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", icc_a.len());
    buf.extend_from_slice(icc_a_hdr.as_bytes());
    buf.extend_from_slice(&icc_a);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_b_off = buf.len();
    let bogus = vec![0u8; 128];
    let icc_b_hdr = format!("6 0 obj\n<< /N 1 /Length {} >>\nstream\n", bogus.len());
    buf.extend_from_slice(icc_b_hdr.as_bytes());
    buf.extend_from_slice(&bogus);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    let obj_count = 7;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [
        cat_off, pages_off, page_off, stream_off, icc_a_off, icc_b_off,
    ] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    let doc = PdfDocument::from_bytes(buf).expect("open synthetic PDF");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (128u8, 128, 128, 255),
        "ICCBased N=1 with document CMYK /OutputIntents declared must produce a neutral \
         grey from the single component (0.5 → 128); OutputIntent is not consulted. \
         Got ({r},{g},{b},{a})."
    );
}

/// /ICCBased N=4 paint **inside a Form XObject**: precedence survives
/// the Form scope. The embedded ICCBased CS1 is declared on the page,
/// the Form XObject's content paints `/CS1 cs 0.25 0 0 0 scn ... f`,
/// and the page invokes the Form with `q /Fm1 Do Q`. Expected pixel:
/// profile B's reference (194, 194, 194, 255).
#[test]
fn qa_round2_iccbased_n4_precedence_survives_form_xobject_scope() {
    let profile_a = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let profile_b = build_minimal_cmyk_to_rgb_lut8_profile(PROFILE_B_TARGET_L_BYTE);

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK A) /DestOutputProfile 5 0 R >>] >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    // Page declares ICCBased CS1 + form Fm1.
    let page_off = buf.len();
    buf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /ColorSpace << /CS1 [/ICCBased 6 0 R] >> /XObject << /Fm1 7 0 R >> >> /Contents 4 0 R >>\nendobj\n");
    // Page content invokes the Form inside a q/Q scope.
    let stream_off = buf.len();
    let content = "q\n/Fm1 Do\nQ\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_a_off = buf.len();
    let icc_a_hdr = format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", profile_a.len());
    buf.extend_from_slice(icc_a_hdr.as_bytes());
    buf.extend_from_slice(&profile_a);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_b_off = buf.len();
    let icc_b_hdr = format!("6 0 obj\n<< /N 4 /Length {} >>\nstream\n", profile_b.len());
    buf.extend_from_slice(icc_b_hdr.as_bytes());
    buf.extend_from_slice(&profile_b);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    // Form XObject Fm1: BBox 0..100, identity matrix, inherits page
    // resources via /Resources <<>> + content paints CS1.
    let form_content = "/CS1 cs\n0.25 0 0 0 scn\n20 20 60 60 re\nf\n";
    let form_off = buf.len();
    let form_hdr = format!(
        "7 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Resources << /ColorSpace << /CS1 [/ICCBased 6 0 R] >> >> /Length {} >>\nstream\n",
        form_content.len()
    );
    buf.extend_from_slice(form_hdr.as_bytes());
    buf.extend_from_slice(form_content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    let obj_count = 8;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [
        cat_off, pages_off, page_off, stream_off, icc_a_off, icc_b_off, form_off,
    ] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    let doc = PdfDocument::from_bytes(buf).expect("open synthetic PDF");
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    let (br, bg, bb) = PROFILE_B_RGB_AT_FIXTURE_INPUT;
    assert_eq!(
        (r, g, b, a),
        (br, bg, bb, 255),
        "embedded /ICCBased N=4 precedence must survive Form XObject scope — Form paint \
         routed through the page-declared CS1's embedded profile B, not the document \
         /OutputIntents A. Expected ({br},{bg},{bb},255); got ({r},{g},{b},{a}). \
         (126,126,126,_) means Form scope dropped the embedded-ICC routing and the \
         document OutputIntent fired. (191,255,255,_) means neither profile was \
         consulted."
    );
}

// ===========================================================================
// Phase 6: ICC v4 support verification through qcms 0.3.0
// ===========================================================================
//
// qcms 0.3.0 reports ICC v4 support via `iccv4-enabled` (a default feature
// in our build). The `check_profile_version` function at
// `qcms-0.3.0/src/iccread.rs:274` reads only the reserved bytes 10..12 of
// the header and rejects them if non-zero; the major/minor comparison is
// commented out with the comment "Checking the version doesn't buy us
// anything". So a profile whose header advertises v4 (`0x04 0x00 0x00
// 0x00`) parses through qcms identically to a v2 (`0x02 0x40 0x00 0x00`)
// header — the reserved bytes are zero in both cases.
//
// The TRUE ICC v4 difference at the wire level is in the A2B0 tag body:
// v2 uses `mft1` (LUT8) or `mft2` (LUT16); v4 introduces the `mAB ` tag
// form with separate input curves, a matrix, a CLUT, and output curves
// (ICC.1:2010 §10.10). qcms parses both: for the A2B0 transform
// direction the dispatch is at `iccread.rs:1675-1681` (RGB) and
// `:1716-1722` (CMYK) — `mft1`/`mft2` → `read_tag_lutType`; `mAB ` →
// `read_tag_lutmABType`. So a CMYK profile carrying a v4 header AND an
// mAB body would also work; we don't synthesise that combination here
// because constructing a valid mAB tag requires four full input curves
// + a 4D CLUT + three output curves + their offsets, none of which the
// constant-CLUT fixture trick benefits from.
//
// The probe below pins three claims:
//   1. A CMYK profile whose header declares ICC v4 parses through
//      `IccProfile::parse`.
//   2. qcms 0.3.0 builds a real CMM (`Transform::has_cmm() == true`)
//      from that v4-header profile.
//   3. The byte-exact RGB output for `convert_cmyk_pixel(64, 0, 0, 0)`
//      matches the v2 reference (126, 126, 126) — i.e. the version-byte
//      flip is non-destructive when the underlying LUT8 body is
//      identical.

/// Pin that qcms 0.3.0 accepts an ICC-v4-versioned CMYK profile and
/// drives the same byte-exact CMYK→RGB conversion as the v2-headered
/// equivalent through a `Transform`.
///
/// This is the unit-level v4 verification — proves the qcms CMM
/// dispatch handles the v4 version byte without rejecting the profile
/// or falling back to the §10.3.5 additive-clamp wrapper inside
/// `Transform::convert_cmyk_pixel`.
///
/// Reference value: `target_l_byte=135` projects through the
/// Lab→XYZ→sRGB chain to RGB(126, 126, 126). Independently verified
/// via the round-1 byte-exact harness; intent-invariant by
/// construction (constant CLUT).
#[test]
fn qa_round3_iccbased_v4_profile_compiles_through_qcms_to_same_reference() {
    use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
    use std::sync::Arc;

    let v2 = build_minimal_cmyk_to_rgb_lut8_profile_with_version(135, IccProfileVersion::V2);
    let v4 = build_minimal_cmyk_to_rgb_lut8_profile_with_version(135, IccProfileVersion::V4);

    // Confirm the version bytes are what we intended at the wire level.
    // Without this gate a regression in the builder could silently emit
    // the same header twice and the test would pass for the wrong reason.
    assert_eq!(v2[8..12], [0x02, 0x40, 0x00, 0x00], "v2 header bytes incorrect");
    assert_eq!(v4[8..12], [0x04, 0x00, 0x00, 0x00], "v4 header bytes incorrect");
    // The rest of the profile must match byte-for-byte — only the
    // version field differs. Otherwise the RGB comparison below could
    // be affected by an unrelated change.
    assert_eq!(v2.len(), v4.len(), "v2 and v4 profiles must differ only in version bytes");
    for i in 0..v2.len() {
        if (8..12).contains(&i) {
            continue;
        }
        assert_eq!(v2[i], v4[i], "v2/v4 builders diverge at byte {i}");
    }

    let prof_v2 = Arc::new(IccProfile::parse(v2, 4).expect("v2 profile must parse"));
    let prof_v4 = Arc::new(
        IccProfile::parse(v4, 4)
            .expect("v4 profile must parse through IccProfile::parse — qcms 0.3.0 accepts v4"),
    );

    let t_v2 = Transform::new_srgb_target(prof_v2, RenderingIntent::RelativeColorimetric);
    let t_v4 = Transform::new_srgb_target(prof_v4, RenderingIntent::RelativeColorimetric);

    assert!(
        t_v2.has_cmm(),
        "v2 profile must compile into a real qcms transform; \
         without it the v4-vs-v2 byte-exact comparison degenerates"
    );
    assert!(
        t_v4.has_cmm(),
        "v4 profile must compile into a real qcms transform. qcms 0.3.0's \
         check_profile_version (iccread.rs:274) is documented to accept v4 \
         headers; if this assertion fires the qcms version no longer matches \
         the plan's research-confirmed behaviour"
    );

    let rgb_v2 = t_v2.convert_cmyk_pixel(64, 0, 0, 0);
    let rgb_v4 = t_v4.convert_cmyk_pixel(64, 0, 0, 0);
    assert_eq!(
        rgb_v2,
        [126u8, 126, 126],
        "v2 byte-exact reference must be (126,126,126) — round-1 pin"
    );
    assert_eq!(
        rgb_v4,
        [126u8, 126, 126],
        "v4 byte-exact reference must equal v2's (126,126,126); qcms 0.3.0 \
         treats the version byte as informational and drives the same LUT8 \
         body. Got {rgb_v4:?}"
    );
}

/// Pin that an ICC v4 profile threaded through a synthetic PDF's
/// `/OutputIntents` array renders the DeviceCMYK paint via the qcms
/// CMM end-to-end, not through the §10.3.5 additive-clamp fallback.
///
/// This is the integration-level v4 probe: confirms the version-byte
/// flip survives the full chain
/// `IccProfile::parse → ResolutionContext::with_output_intent →
/// cmyk_to_rgb_via_intent → Transform::new_srgb_target →
/// Transform::convert_cmyk_pixel`.
///
/// Reference: same `target_l_byte=135` → near-neutral grey ~(128, 128,
/// 128) for any CMYK input including (0.25, 0, 0, 0). The additive-
/// clamp pin for that input is (191, 255, 255); the integration
/// assertion must land in the qcms-converted neighbourhood, not the
/// fallback.
#[test]
fn qa_round3_iccbased_v4_output_intent_drives_render_through_qcms() {
    let icc_v4 = build_minimal_cmyk_to_rgb_lut8_profile_with_version(135, IccProfileVersion::V4);

    // Sanity-pin the v4 profile parses + compiles + produces the round-1
    // byte-exact reference. Without this the integration assertion could
    // fail for the wrong reason (e.g. profile reject → §10.3.5 fallback
    // fires → 191,255,255 instead of ~128).
    {
        use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
        use std::sync::Arc;
        let prof = Arc::new(
            IccProfile::parse(icc_v4.clone(), 4)
                .expect("v4 profile parses through IccProfile::parse"),
        );
        let t = Transform::new_srgb_target(prof, RenderingIntent::RelativeColorimetric);
        assert!(t.has_cmm(), "v4 profile must compile into a real qcms CMM");
        assert_eq!(
            t.convert_cmyk_pixel(64, 0, 0, 0),
            [126u8, 126, 126],
            "v4 profile must produce the round-1 byte-exact reference (126,126,126)"
        );
    }

    let pdf = build_pdf_cmyk_with_output_intent(&icc_v4);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic v4 PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "v4 fixture must expose its OutputIntent via the document accessor; \
         a None here means /N=4 filter or stream decode failed for the v4 \
         profile bytes"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, _a) = pixel_at(&rgba, 50, 50);
    let near = |v: u8| (v as i32 - 128).abs() <= 10;
    assert!(
        near(r) && near(g) && near(b),
        "v4 OutputIntent must drive the render through qcms — expected ~(128,128,128); \
         got ({r}, {g}, {b}). (191,255,255) would mean the §10.3.5 fallback fired."
    );
}

// ===========================================================================
// Black-Point Compensation (BPC) — HONEST_GAP probe
// ===========================================================================
//
// ISO 32000-1:2008 §8.6.5.8 names four rendering intents but does NOT
// mandate Black-Point Compensation as a separate switch — BPC is a CMM-
// implementation knob that piggybacks on the rendering intent (typically
// on /RelativeColorimetric, sometimes /Perceptual) to remap source-profile
// black to destination-profile black, preserving shadow detail when the
// destination has a less-black-than-source black point.
//
// qcms 0.3.0 does NOT implement BPC. Verified at
// `~/.cargo/registry/src/.../qcms-0.3.0/src/lib.rs:29-36`:
//
//   //! ### Black-Point Compensation (BPC)
//   //!
//   //! BPC is currently not supported. Adding it would require
//   //! either pre-multiplying the entire CLUT during transform
//   //! construction (memory and build-time cost) or running an
//   //! extra division per pixel (CPU cost). BPC brings an
//   //! unacceptable performance overhead, so we go with
//   //! perceptual.
//
// Additionally `transform.rs:1283-1289` shows the CMYK transform builder
// declares `_intent: Intent` — the rendering intent parameter is
// underscore-prefixed and unused inside the CMYK path. So for CMYK
// sources, the byte-exact qcms output is invariant across:
//   - All four PDF rendering intents (intent ignored by qcms's CLUT
//     precomputation at `transform_precacheLUT_cmyk_float:1245-1281`).
//   - BPC on vs off (BPC not implemented at all).
//
// What this means for pdf_oxide:
//   - The pipeline's intent threading (`gs.rendering_intent` →
//     `ctx.rendering_intent` → `Transform::new_srgb_target`'s intent
//     parameter) is correct end-to-end. qcms is the limiting factor.
//   - The cache key `(profile.content_hash(), intent)` still includes
//     intent so a future qcms upgrade (or a switch to a CMM that
//     honours intent, e.g. lcms2) doesn't silently collapse cache
//     entries across intents.
//
// The probe below pins the current behaviour byte-for-byte. When qcms
// grows BPC (either via fork or upgrade) OR pdf_oxide switches to a
// different CMM, this test will go RED at the BPC-aware delta and the
// implementer can re-derive the expected references for the intent
// matrix.

/// HONEST_GAP marker: qcms 0.3.0 has no BPC implementation AND silently
/// drops the rendering-intent parameter for CMYK sources. When that
/// changes, every line in this probe is the point of update.
const HONEST_GAP_QCMS_030_NO_BPC: &str =
    "HONEST_GAP_QCMS_030_NO_BPC: qcms 0.3.0 ignores rendering intent for CMYK \
     (transform.rs:1288 `_intent: Intent`) and has no Black-Point Compensation \
     implementation (lib.rs:29-36 design comment). The BPC-aware shadow-detail \
     preservation that /RelativeColorimetric is documented to provide on \
     near-black CMYK inputs cannot be probed against a CMM that drops both \
     intent and BPC; the assertions below pin the current intent-invariant, \
     BPC-absent behaviour and will go RED when either changes — at which point \
     the probe should be split into per-intent expected references derived \
     against the new CMM.";

/// Pin that a near-black CMYK input (CMYK(0, 0, 0, 0.95) — 95 % K, deep
/// shadow) produces the SAME qcms-converted RGB under both
/// `/RelativeColorimetric` and `/AbsoluteColorimetric` rendering
/// intents. With BPC active on Relative the shadow detail would be
/// elevated (preserved relative to the destination black point); with
/// BPC absent both intents collapse to the same CLUT output.
///
/// Also pin that all four PDF rendering intents produce identical bytes
/// for the same input, mirroring the round-3
/// `qa_round3_qcms_030_treats_cmyk_intent_as_informational` probe but
/// at the deep-shadow region where BPC matters most.
///
/// This probe is `#[ignore]`-marked: the assertions reflect the
/// CURRENT byte-exact qcms 0.3.0 behaviour (which conflates BPC-on with
/// BPC-off and ignores intent altogether), so they always pass at HEAD.
/// The point of the ignore marker is that running this probe under a
/// future CMM that DOES implement BPC will surface the gap by going RED
/// at the per-intent assertion — at which point the implementer
/// re-derives the expected references and removes the ignore.
#[test]
#[ignore = "HONEST_GAP_QCMS_030_NO_BPC"]
fn qa_round4_bpc_paper_white_preservation_under_relative_colorimetric() {
    use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
    use std::sync::Arc;

    // Mark the gap constant as live so the linker doesn't drop it; the
    // string is the diagnostic a future-engineer reading the test
    // failure picks up. The unused-binding bypass is intentional.
    let _ = HONEST_GAP_QCMS_030_NO_BPC;

    // The constant-CLUT fixture used everywhere else in this file
    // collapses every CMYK input to one Lab tuple — useless for a
    // shadow-preservation probe because no input bits make it to the
    // output. We need a NON-constant CLUT here so the deep-shadow
    // input lands in a CLUT cell that could in principle differ across
    // intents. Build a v2 LUT8 with a non-constant CLUT: the 16 grid
    // corners ramp linearly with the input, so CMYK(0,0,0,0.95)
    // resolves to a deep-shadow grey distinct from CMYK(0,0,0,0).
    //
    // qcms-0.3.0 CMYK input handling at `transform.rs:1244-1289`
    // ignores the rendering intent regardless of CLUT shape, so the
    // four-intent assertion below holds for any LUT.
    let icc = build_minimal_cmyk_to_rgb_lut8_profile_with_shadow_ramp(0..=240);
    let prof = Arc::new(IccProfile::parse(icc, 4).expect("ramp profile parses"));

    // CMYK(0, 0, 0, 242) ≈ 95 % K — deep shadow. Pin the same byte-exact
    // RGB across every intent. With BPC implemented and intent-honouring,
    // these would diverge:
    //   - Perceptual: gamut-compress to preserve overall tone relationships;
    //     deep blacks may map to slightly elevated dest blacks.
    //   - RelativeColorimetric WITH BPC: source black → dest black with
    //     shadow detail preserved (the typical print-house default).
    //   - Saturation: preserve hue purity; not relevant here.
    //   - AbsoluteColorimetric: no white-point adaptation, no BPC; render
    //     source black at the dest's measured black value (paper-relative).
    //
    // With qcms 0.3.0 all four collapse to the same CLUT output because
    // the CLUT is pre-computed without intent dependency and BPC isn't
    // implemented.
    let mut last: Option<[u8; 3]> = None;
    for intent in [
        RenderingIntent::Perceptual,
        RenderingIntent::RelativeColorimetric,
        RenderingIntent::Saturation,
        RenderingIntent::AbsoluteColorimetric,
    ] {
        let t = Transform::new_srgb_target(Arc::clone(&prof), intent);
        let rgb = t.convert_cmyk_pixel(0, 0, 0, 242);
        if let Some(prev) = last {
            assert_eq!(
                prev, rgb,
                "qcms 0.3.0 must produce intent-invariant bytes for a near-black \
                 CMYK input (BPC absent + CMYK intent dropped): previous intent \
                 yielded {prev:?}, intent={intent:?} yielded {rgb:?}. A divergence \
                 here means qcms grew BPC or started honouring intent — re-derive \
                 the expected references per intent and split this probe."
            );
        }
        last = Some(rgb);
    }
}

/// Build a 4×grid LUT8 CMYK→Lab profile with a non-constant CLUT that
/// ramps linearly with the K channel (the dominant axis of typical
/// deep-shadow CMYK builds). Used by the BPC HONEST_GAP probe to feed
/// qcms an input where the CLUT actually depends on the input bits,
/// so a CMM with BPC implementation would in principle produce a
/// shadow-detail-elevated output distinct from the no-BPC reference.
///
/// `l_range` controls the lightness span across the K axis: at K=0 the
/// LUT outputs L*=l_range.start(), at K=255 it outputs L*=l_range.end().
/// The two corners pin the ramp; intermediate grid points are linearly
/// interpolated by qcms's tetrahedral CLUT lookup at runtime.
fn build_minimal_cmyk_to_rgb_lut8_profile_with_shadow_ramp(
    l_range: std::ops::RangeInclusive<u8>,
) -> Vec<u8> {
    let in_chan: u8 = 4;
    let out_chan: u8 = 3;
    let grid: u8 = 2;
    let mut lut = Vec::with_capacity(1888);
    lut.extend_from_slice(&0x6d66_7431u32.to_be_bytes());
    lut.extend_from_slice(&0u32.to_be_bytes());
    lut.push(in_chan);
    lut.push(out_chan);
    lut.push(grid);
    lut.push(0);
    let identity: [i32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x0001_0000];
    for v in identity {
        lut.extend_from_slice(&(v as u32).to_be_bytes());
    }
    for _ in 0..in_chan {
        for i in 0..256u16 {
            lut.push(i as u8);
        }
    }
    // 16 grid points for 2^4. CLUT iteration order per ICC.1 §10.8:
    // input channels iterate from MSB outermost — for 4 channels with
    // grid=2 the ordering is C × M × Y × K with K innermost. We want
    // the K-axis to ramp: at K=0 → l_range.start(); at K=255 →
    // l_range.end(). Both other axes are pinned at the same L for
    // simplicity (the LUT8 path then drives shadow ramp linearly on K).
    let l_low = *l_range.start();
    let l_high = *l_range.end();
    for c_i in 0..2 {
        for m_i in 0..2 {
            for y_i in 0..2 {
                for k_i in 0..2 {
                    let _ = (c_i, m_i, y_i);
                    let l = if k_i == 0 { l_high } else { l_low };
                    lut.push(l);
                    lut.push(128);
                    lut.push(128);
                }
            }
        }
    }
    for _ in 0..out_chan {
        for i in 0..256u16 {
            lut.push(i as u8);
        }
    }
    debug_assert_eq!(lut.len(), 1888, "shadow-ramp LUT8 body size mismatch");

    let mut profile = vec![0u8; 128];
    let total_size: u32 = 128 + 4 + 12 + lut.len() as u32;
    profile[0..4].copy_from_slice(&total_size.to_be_bytes());
    profile[8..12].copy_from_slice(&IccProfileVersion::V2.header_bytes());
    profile[12..16].copy_from_slice(b"prtr");
    profile[16..20].copy_from_slice(b"CMYK");
    profile[20..24].copy_from_slice(b"Lab ");
    profile[36..40].copy_from_slice(b"acsp");
    profile[64..68].copy_from_slice(&0u32.to_be_bytes());
    profile[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes());
    profile[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    profile[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes());
    profile.extend_from_slice(&1u32.to_be_bytes());
    profile.extend_from_slice(&0x4132_4230u32.to_be_bytes());
    profile.extend_from_slice(&144u32.to_be_bytes());
    profile.extend_from_slice(&(lut.len() as u32).to_be_bytes());
    profile.extend_from_slice(&lut);
    profile
}

// ===========================================================================
// Real-corpus regression: vendor branding logo equivalent
// ===========================================================================
//
// The user surfaced an early-project regression where a vendor branding
// logo's green mark rendered too vivid/lime through the §10.3.5
// additive-clamp fallback instead of muted/olive (the press-target colour
// the CMYK profile maps the build to).
//
// We don't have the real branding-logo PDF on disk. Building a synthetic
// equivalent that demonstrates the same shape (a green CMYK build whose
// additive-clamp value diverges from the qcms-converted value in the
// "too vivid → muted" direction) is the next-best regression sentry:
// it pins the OutputIntent pipeline shifts the green-mark colour in the
// predicted direction, byte-for-byte.
//
// Real-corpus probe: when a future engineer adds the real branding-logo
// PDF to `tests/fixtures/`, the assertion shape carries over byte-for-
// byte (the qcms reference for the embedded profile's mapping of the
// green build). The synthetic version uses our constant-CLUT
// `target_l_byte=200` profile — that maps every CMYK input to a
// neutral grey, NOT a specific colour. This is the limitation of the
// synthetic fixture: it proves "OutputIntent shifted the colour"
// directionally, not "OutputIntent shifted it TOWARDS the real
// press-target value." That's what a real press profile would prove.

/// Pin that a CMYK paint that matches the typical "green logo build"
/// (C=0.30, M=0.05, Y=0.95, K=0.05) renders DIFFERENTLY through the
/// OutputIntent ICC vs through the §10.3.5 additive-clamp fallback,
/// AND that the OutputIntent direction is towards the constant-CLUT
/// reference (proxy for "muted press target") rather than the vivid
/// additive-clamp value.
///
/// Why this matters: the original regression user-surfaced was that a
/// vendor branding logo's green mark printed muted-olive on the press
/// but rendered vivid-lime on screen (additive-clamp). Wiring the
/// OutputIntent profile through the renderer is the fix that closes
/// the press-vs-screen divergence. The synthetic profile here uses
/// the constant-CLUT target — every CMYK input maps to ~(194, 194,
/// 194), a muted neutral. Distinct from the §10.3.5 vivid-lime value
/// for the same input, so the directional check fires.
///
/// Three pins:
///   1. The render WITHOUT OutputIntent produces the §10.3.5 additive-
///      clamp value for the green CMYK build. That's the pre-#97
///      baseline.
///   2. The render WITH OutputIntent produces the qcms reference value
///      for the profile's mapping of that build.
///   3. The OutputIntent value is closer to the constant-CLUT
///      reference (~194) than to the §10.3.5 value, demonstrating
///      directional correctness even on a synthetic fixture.
#[test]
fn qa_round4_branding_green_mark_routes_through_output_intent() {
    // Green-mark CMYK build matching the typical vendor-logo colour
    // recipe — high yellow, moderate cyan, minimal magenta and black.
    // The §10.3.5 additive-clamp value is:
    //   R = 1 - (0.30 + 0.05) = 0.65 → 166
    //   G = 1 - (0.05 + 0.05) = 0.90 → 230
    //   B = 1 - (0.95 + 0.05) = 0.00 → 0
    // i.e. RGB(166, 230, 0) — vivid lime-green, brand-mismatched.
    //
    // The OutputIntent path through profile B (target_l_byte=200) maps
    // every CMYK input to roughly RGB(194, 194, 194) — muted neutral.
    // This is a proxy for "muted olive that the real press profile
    // would produce"; the synthetic fixture's constant CLUT doesn't
    // produce the actual olive value, but it proves the conversion
    // ROUTE shifts the colour towards the press target rather than
    // emitting the raw additive-clamp value.

    // ---- WITHOUT OutputIntent ----
    {
        let content = "0.30 0.05 0.95 0.05 k\n20 20 60 60 re\nf\n";
        let pdf = build_pdf_with_catalog_entries_and_content("", content, None);
        let doc = PdfDocument::from_bytes(pdf).expect("open no-OI fixture");
        assert!(
            doc.output_intent_cmyk_profile().is_none(),
            "no-OI fixture must declare no /OutputIntents"
        );
        let rgba = render_rgba(&doc);
        let (r, g, b, a) = pixel_at(&rgba, 50, 50);
        assert_eq!(
            (r, g, b, a),
            (166u8, 230, 0, 255),
            "without /OutputIntents the green-mark CMYK build must produce the §10.3.5 \
             additive-clamp value RGB(166, 230, 0) — vivid lime. This is the pre-#97 \
             baseline; got ({r},{g},{b},{a})"
        );
    }

    // ---- WITH OutputIntent (profile B, target_l_byte=200) ----
    let profile_b = build_minimal_cmyk_to_rgb_lut8_profile(PROFILE_B_TARGET_L_BYTE);
    // Derive the qcms byte-exact reference for the actual paint input
    // BEFORE asserting the render, so the assertion ties the rendered
    // pixel to a verifiable CMM output. The 8-bit round-trip the
    // resolver does maps 0.30 → 77, 0.05 → 13, 0.95 → 242, 0.05 → 13.
    let qcms_ref = {
        use pdf_oxide::color::{IccProfile, RenderingIntent, Transform};
        use std::sync::Arc;
        let prof = Arc::new(IccProfile::parse(profile_b.clone(), 4).expect("parse B"));
        let t = Transform::new_srgb_target(prof, RenderingIntent::RelativeColorimetric);
        t.convert_cmyk_pixel(77, 13, 242, 13)
    };

    let content = "0.30 0.05 0.95 0.05 k\n20 20 60 60 re\nf\n";
    let catalog_entries = "/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic Press) /DestOutputProfile 5 0 R >>]";
    let pdf =
        build_pdf_with_catalog_entries_and_content(catalog_entries, content, Some(&profile_b));
    let doc = PdfDocument::from_bytes(pdf).expect("open with-OI fixture");
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "with-OI fixture must expose the OutputIntent profile"
    );
    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (qcms_ref[0], qcms_ref[1], qcms_ref[2], 255),
        "with /OutputIntents the green-mark CMYK build must render through the qcms \
         reference {qcms_ref:?}; got ({r},{g},{b},{a}). (166,230,0,_) would mean the \
         OutputIntent route was bypassed and the §10.3.5 additive-clamp fired."
    );

    // ---- Direction-of-shift sanity check ----
    //
    // The additive-clamp value is RGB(166, 230, 0); the OutputIntent
    // value is qcms_ref. "Direction-of-shift" check: does the
    // OutputIntent value move us AWAY from the vivid-lime channel
    // distribution (very low B, very high G, mid R) towards a more
    // neutral / less-saturated value? Saturation in HSV terms is
    // (max - min) / max — vivid lime has saturation ≈ 1.0; neutral
    // grey has saturation 0.
    let additive = (166u8, 230, 0);
    let oi = (qcms_ref[0], qcms_ref[1], qcms_ref[2]);
    let saturation = |c: (u8, u8, u8)| -> f32 {
        let max = c.0.max(c.1).max(c.2) as f32;
        let min = c.0.min(c.1).min(c.2) as f32;
        if max == 0.0 {
            0.0
        } else {
            (max - min) / max
        }
    };
    let sat_additive = saturation(additive);
    let sat_oi = saturation(oi);
    assert!(
        sat_oi < sat_additive,
        "OutputIntent value must be LESS saturated than the additive-clamp value \
         to demonstrate the press-target direction-of-shift; saturation additive={:.3} \
         vs OutputIntent={:.3}",
        sat_additive,
        sat_oi
    );
}

/// HONEST_GAP marker: real branding-logo PDF fixture is not on disk in
/// this worktree. The synthetic test above exercises the conversion
/// route; a real-corpus probe with a vendor-issued press profile would
/// pin the qcms reference is within a documented ΔE threshold of the
/// commercial-viewer baseline. Until that fixture lands the directional
/// sanity check (saturation collapses through the OutputIntent path) is
/// the proxy.
const HONEST_GAP_NO_REAL_BRANDING_FIXTURE: &str =
    "HONEST_GAP_NO_REAL_BRANDING_FIXTURE: no real branding-logo PDF on disk; \
     the synthetic green-mark probe exercises the route but uses a constant-CLUT \
     profile rather than a real press profile. A future engineer with a vendor-\
     issued ICC and the branding-logo PDF should add a CIEDE2000 ΔE assertion \
     against the commercial-viewer baseline.";

/// Pin the HONEST_GAP marker is referenced so the compile-time string
/// constant doesn't drop. Acts as the documentation point for the
/// missing real-corpus fixture; surface it in any future audit of
/// outstanding press-quality probes.
#[test]
fn qa_round4_real_branding_fixture_honest_gap() {
    // Reference the marker so dead-code lint doesn't trip in a feature
    // build where the marker would otherwise be unused. The test passes
    // unconditionally — its existence is the audit trail.
    let _ = HONEST_GAP_NO_REAL_BRANDING_FIXTURE;
}

// ===========================================================================
// Edge-case regression sentries: document-/OutputIntents-shape oddities
// ===========================================================================
//
// These probes pin the renderer's response to malformed or unusual
// `/OutputIntents` array shapes. Each probe builds a PDF whose catalog
// declares an OutputIntent entry that the accessor MUST refuse to surface
// (wrong `/N`, missing `/DestOutputProfile`, garbage stream contents) and
// asserts the renderer falls through to the §10.3.5 additive-clamp value
// for a `/DeviceCMYK` paint of (0.25, 0, 0, 0). They are regression
// sentries — current behaviour, not failing-first probes — so a future
// change that loosens the accessor's filtering (e.g. accepting an N=3
// OutputIntent as CMYK) would flip them RED and surface the regression.

/// `/N=3` (RGB) OutputIntent with a `/DeviceCMYK` paint operator. The
/// reader at `document.rs:3645-3648` filters on `Some(4)`; an `/N 3`
/// entry is skipped and `output_intent_cmyk_profile()` returns `None`,
/// so the renderer reaches the §10.3.5 additive-clamp fallback.
///
/// Regression sentry: a regression that broadened the `/N` filter to
/// accept N=3 would route CMYK bytes through an RGB profile and either
/// panic at qcms's channel-count assert or emit garbage RGB.
#[test]
fn output_intent_n3_rgb_profile_rejected_at_reader_falls_through_to_additive_clamp() {
    // Build a PDF whose /OutputIntents entry declares /N 3 on its
    // /DestOutputProfile stream. The body bytes don't have to parse as a
    // real RGB ICC profile because the accessor filters on /N before it
    // ever invokes IccProfile::parse on the stream payload.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic RGB) /DestOutputProfile 5 0 R >>] >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> /Contents 4 0 R >>\nendobj\n");
    let stream_off = buf.len();
    let content = "0.25 0 0 0 k\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_off = buf.len();
    // /N 3 — accessor rejects before parsing the body.
    let bogus = vec![0u8; 128];
    let icc_hdr = format!("5 0 obj\n<< /N 3 /Length {} >>\nstream\n", bogus.len());
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(&bogus);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    let obj_count = 6;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, icc_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    let doc = PdfDocument::from_bytes(buf).expect("open synthetic PDF");

    // Cross-pin: the accessor MUST refuse the N=3 entry.
    assert!(
        doc.output_intent_cmyk_profile().is_none(),
        "an /OutputIntents entry with /N 3 (RGB) must be filtered out by \
         output_intent_cmyk_profile(); only /N 4 (CMYK) entries are eligible"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (191u8, 255, 255, 255),
        "an N=3 OutputIntent must fall through to §10.3.5 additive-clamp \
         byte-for-byte for /DeviceCMYK paint; got ({r},{g},{b},{a})"
    );
}

/// `/OutputIntents` entry with `/OutputCondition` text only and **no**
/// `/DestOutputProfile` stream. The accessor's `entry_dict.get("DestOutputProfile")`
/// check at `document.rs:3630-3633` returns `None`, the entry is skipped,
/// and the whole array exhausts → `output_intent_cmyk_profile()` is `None`.
/// CMYK paint falls through to §10.3.5.
///
/// Regression sentry: a regression that materialised a default profile
/// when none was declared would surface here.
#[test]
fn output_intent_with_outputcondition_string_only_no_destoutputprofile_falls_through() {
    // No DestOutputProfile, no profile stream object at all. /OutputCondition
    // is just a human-readable string (PDF/X advisory metadata per
    // §14.11.5 — "the intended printing condition", e.g. "FOGRA39"); it
    // does not carry the ICC bytes itself.
    let catalog_entries =
        "/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (FOGRA39 (ISO 12647-2:2004)) /OutputConditionIdentifier (FOGRA39) >>]";
    let content_ops = "0.25 0 0 0 k\n20 20 60 60 re\nf\n";
    let pdf = build_pdf_with_catalog_entries_and_content(catalog_entries, content_ops, None);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");

    // Cross-pin: missing /DestOutputProfile means the accessor returns
    // None even though /OutputIntents is present and well-formed otherwise.
    assert!(
        doc.output_intent_cmyk_profile().is_none(),
        "an /OutputIntents entry without a /DestOutputProfile stream must \
         surface as None; the /OutputCondition string is advisory metadata, \
         not a fallback colour source"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (191u8, 255, 255, 255),
        "a /OutputCondition-only entry must fall through to §10.3.5 \
         additive-clamp byte-for-byte; got ({r},{g},{b},{a})"
    );
}

/// Two `/OutputIntents` entries in catalog-declaration order — first
/// `/N 3` (RGB), second `/N 4` (CMYK). The accessor iterates the array
/// (`document.rs:3621`) and returns the first entry whose `/N` matches 4
/// AND whose stream parses through `IccProfile::parse`. The N=3 entry is
/// skipped at the /N filter; the N=4 entry's CLUT (target_l_byte = 135)
/// is consumed; the rendered pixel matches the qcms reference for that
/// profile (RGB(126, 126, 126)).
///
/// Regression sentry: a regression that returned the FIRST array entry
/// regardless of /N would surface a None and additive-clamp here.
#[test]
fn output_intent_array_picks_first_cmyk_entry_skipping_rgb() {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    // Two-entry /OutputIntents array: RGB first, CMYK second.
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic RGB) /DestOutputProfile 5 0 R >> << /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK) /DestOutputProfile 6 0 R >>] >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> /Contents 4 0 R >>\nendobj\n");
    let stream_off = buf.len();
    // Paint CMYK(0.25, 0, 0, 0) — matches the canonical reference input.
    let content = "0.25 0 0 0 k\n20 20 60 60 re\nf\n";
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    // RGB entry first — bogus body, filtered at /N 3 before parse.
    let rgb_off = buf.len();
    let bogus = vec![0u8; 128];
    let rgb_hdr = format!("5 0 obj\n<< /N 3 /Length {} >>\nstream\n", bogus.len());
    buf.extend_from_slice(rgb_hdr.as_bytes());
    buf.extend_from_slice(&bogus);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    // CMYK entry second — real LUT8 profile keyed on target_l_byte=135,
    // whose qcms reference for CMYK(64, 0, 0, 0) is RGB(126, 126, 126).
    let cmyk_profile = build_minimal_cmyk_to_rgb_lut8_profile(135);
    let cmyk_off = buf.len();
    let cmyk_hdr = format!("6 0 obj\n<< /N 4 /Length {} >>\nstream\n", cmyk_profile.len());
    buf.extend_from_slice(cmyk_hdr.as_bytes());
    buf.extend_from_slice(&cmyk_profile);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    let obj_count = 7;
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", obj_count).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, rgb_off, cmyk_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            obj_count, xref_off
        )
        .as_bytes(),
    );
    let doc = PdfDocument::from_bytes(buf).expect("open synthetic PDF");

    // Cross-pin: the accessor surfaces the SECOND entry's profile.
    let profile = doc
        .output_intent_cmyk_profile()
        .expect("the CMYK entry must be picked from a mixed-N array");
    assert_eq!(
        profile.n_components(),
        4,
        "the surfaced profile must be /N=4 — the second array entry"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    // Pinned byte-exact against the qcms 0.3.0 reference for the
    // target_l_byte=135 profile + CMYK(64, 0, 0, 0). Matches the
    // canonical reference asserted in
    // `output_intent_render_pixel_is_byte_exact_against_qcms_reference`.
    assert_eq!(
        (r, g, b, a),
        (126u8, 126, 126, 255),
        "with a mixed [RGB, CMYK] /OutputIntents array the second (CMYK) \
         entry must drive the render byte-for-byte against the qcms \
         reference; got ({r},{g},{b},{a})"
    );
}

/// `/DestOutputProfile` stream whose bytes are pure garbage — not even a
/// 128-byte ICC header, no `acsp` signature, random bytes. The accessor's
/// `IccProfile::parse(bytes, 4)` call at `document.rs:3653` returns `None`,
/// the entry is skipped, and the renderer falls through to §10.3.5.
///
/// This is distinct from the header-only probe at
/// `output_intent_with_unparseable_profile_falls_through_to_additive_clamp`:
/// that probe pins fall-through happens INSIDE `Transform::convert_cmyk_pixel`
/// (the header parses, qcms refuses to build the CMM). This probe pins the
/// earlier rejection where `IccProfile::parse` returns None up front — no
/// header, nothing to keep.
///
/// Regression sentry: a regression that propagated the un-parsed bytes to
/// qcms (or that silently emitted a degenerate IccProfile from a parse
/// failure) would surface here as a panic or as a non-additive-clamp pixel.
#[test]
fn output_intent_malformed_iccbased_stream_falls_through() {
    // 64 bytes of recognisable garbage — too short for an ICC header
    // (128 bytes minimum) and contains no acsp signature.
    let garbage: Vec<u8> = (0u8..=63u8).collect();
    let pdf = build_pdf_cmyk_with_output_intent(&garbage);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");

    // Cross-pin: IccProfile::parse rejects garbage; accessor surfaces None.
    assert!(
        doc.output_intent_cmyk_profile().is_none(),
        "IccProfile::parse must reject a 64-byte garbage stream (less than \
         the 128-byte ICC header minimum); accessor must surface None"
    );

    let rgba = render_rgba(&doc);
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (191u8, 255, 255, 255),
        "a malformed /DestOutputProfile stream (sub-header-length garbage) \
         must fall through to §10.3.5 additive-clamp byte-for-byte without \
         panicking; got ({r},{g},{b},{a})"
    );
}
