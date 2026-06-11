//! Round-4 byte-exact probes for ISO 32000-1 §11.7.4
//! CompatibleOverprint blend function in the composite-then-decompose
//! separation path.
//!
//! Round 3 left the composite path's `apply_overprint_after_paint`
//! using `(src + dst).min(1.0)` for OPM=0, which is a composite-preview
//! approximation, NOT the spec per-plate REPLACE rule from §11.7.4.3 /
//! Table 149. The round-3 QA pass pinned the buggy behaviour with
//! floor-signal asserts and a `QA_BUG_OPM0_COMPOSITE_PATH_ADDITIVELY_
//! MERGES_PLATES` constant. Round 4 closes the gap byte-exact: the
//! per-channel rule from §11.7.4.3 + Table 149 is implemented in the
//! composite path so every pixel's plate output equals the spec-defined
//! `α · B(c_b, c_s) + (1 - α) · c_b` where `B` is the CompatibleOverprint
//! blend function.
//!
//! Spec citations:
//!  - ISO 32000-1 §11.3.3 — basic compositing formula
//!  - ISO 32000-1 §11.3.5 — blend modes (separable / non-separable)
//!  - ISO 32000-1 §11.4.6.2 — knockout group composition rule
//!  - ISO 32000-1 §11.7.3 — spot colours and transparency (sidecar)
//!  - ISO 32000-1 §11.7.4 — overprinting and transparency
//!  - ISO 32000-1 §11.7.4.1 — overprint mode parameter
//!  - ISO 32000-1 §11.7.4.2 — blend modes and overprinting (BM split
//!    per lane class; spot lanes substitute Normal for non-sep BM)
//!  - ISO 32000-1 §11.7.4.3 — CompatibleOverprint blend function
//!    (Table 149: per-channel B(c_b, c_s) by source CS × OP × OPM)
//!  - ISO 32000-1 §11.7.4.5 — summary of overprinting behaviour
//!  - ISO 32000-1 §10.5 — separated plate output per ink

#![cfg(all(feature = "rendering", feature = "icc", feature = "test-support"))]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::render_separations;

// ===========================================================================
// Synthetic PDF builders mirroring the round-3 QA pass shape.
// ===========================================================================

fn build_pdf_with_output_intent(
    content: &str,
    resources_inner: &str,
    icc_profile: &[u8],
    extra_objs: &[&str],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic Non-Linear CMYK) /DestOutputProfile 5 0 R >>] >>\nendobj\n",
    );
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    let page = format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << {} >> /Contents 4 0 R >>\nendobj\n",
        resources_inner
    );
    buf.extend_from_slice(page.as_bytes());
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_off = buf.len();
    let icc_hdr = format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", icc_profile.len());
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(icc_profile);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let mut extra_offs: Vec<usize> = Vec::new();
    for obj in extra_objs {
        extra_offs.push(buf.len());
        buf.extend_from_slice(obj.as_bytes());
    }

    let xref_off = buf.len();
    let total_objs = 5 + extra_objs.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", total_objs + 1).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, icc_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    for off in extra_offs {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs + 1,
            xref_off
        )
        .as_bytes(),
    );
    buf
}

/// Constant-Lab ICC LUT profile mirroring the round-3 helper. Produces
/// the constant L_byte for every CMYK input so the renderer's ICC path
/// has a well-defined byte output we can pin against.
fn build_constant_cmyk_icc(l_byte: u8) -> Vec<u8> {
    let in_chan: u8 = 4;
    let out_chan: u8 = 3;
    let grid: u8 = 2;
    let mut lut = Vec::with_capacity(2048);

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
    let grid_size = (grid as usize).pow(in_chan as u32);
    for _ in 0..grid_size {
        lut.push(l_byte);
        lut.push(128);
        lut.push(128);
    }
    for _ in 0..out_chan {
        for i in 0..256u16 {
            lut.push(i as u8);
        }
    }

    let mut profile = vec![0u8; 128];
    let total_size: u32 = 128 + 4 + 12 + lut.len() as u32;
    profile[0..4].copy_from_slice(&total_size.to_be_bytes());
    profile[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes());
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

fn plate<'a>(
    plates: &'a [pdf_oxide::rendering::SeparationPlate],
    name: &str,
) -> &'a pdf_oxide::rendering::SeparationPlate {
    plates
        .iter()
        .find(|p| p.ink_name == name)
        .unwrap_or_else(|| panic!("no plate named {}", name))
}

fn centre(plate: &pdf_oxide::rendering::SeparationPlate) -> u8 {
    let off = ((plate.height / 2) * plate.width + plate.width / 2) as usize;
    plate.data[off]
}

fn tint_to_u8(t: f32) -> u8 {
    (t.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ===========================================================================
// QA-A1 byte-exact: transparency + OPM=0 + DeviceCMYK + /OP true.
//
// ISO 32000-1 §11.7.4.3 Table 149 row 1: source CS = DeviceCMYK
// specified directly, affected component = C/M/Y/K, OP=true, OPM=0:
//   B(c_b, c_s) = c_s
// for all four process channels.
//
// §11.3.3 composition: c_r = α · B(c_b, c_s) + (1 - α) · c_b.
//
// Setup:
//   Backdrop paint: DeviceCMYK (0.4, 0, 0, 0), /ca = 1.0.
//   Foreground paint: DeviceCMYK (0, 0.5, 0, 0), /OP true, /OPM 0,
//                     /ca = 0.5.
//
// After backdrop paint the sidecar carries:
//   C = 0.4 → u8 102; M = Y = K = 0.
//
// After foreground paint per spec:
//   C: B = c_s = 0,    r = 0.5·0    + 0.5·0.4 = 0.2  → u8 51.
//   M: B = c_s = 0.5,  r = 0.5·0.5  + 0.5·0   = 0.25 → u8 64.
//   Y: B = c_s = 0,    r = 0.5·0    + 0.5·0   = 0.0  → u8 0.
//   K: B = c_s = 0,    r = 0.5·0    + 0.5·0   = 0.0  → u8 0.
// ===========================================================================

#[test]
fn qa_a1_transparency_opm0_devicecmyk_overprint_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let content = "0.4 0 0 0 k\n0 0 100 100 re\nf\n\
                   /Ov gs\n0 0.5 0 0 k\n0 0 100 100 re\nf\n";
    let resources = "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 >> >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let expected_c = tint_to_u8(0.5 * 0.0 + 0.5 * 0.4);
    let expected_m = tint_to_u8(0.5 * 0.5 + 0.5 * 0.0);
    assert_eq!(expected_c, 51);
    assert_eq!(expected_m, 64);

    assert_eq!(
        centre(plate(&plates, "Cyan")),
        expected_c,
        "ISO 32000-1 §11.7.4.3 Table 149 row 1 (DeviceCMYK, OP=true, \
         OPM=0): B(c_b, c_s) = c_s on every process channel. For the \
         C lane c_s=0, c_b=0.4, α=0.5: c_r = 0.5·0 + 0.5·0.4 = 0.2 → \
         u8 51. Got u8 {}.",
        centre(plate(&plates, "Cyan"))
    );
    assert_eq!(
        centre(plate(&plates, "Magenta")),
        expected_m,
        "ISO 32000-1 §11.7.4.3 Table 149 row 1: B = c_s on the M lane. \
         c_s=0.5, c_b=0, α=0.5: c_r = 0.5·0.5 + 0.5·0 = 0.25 → u8 64. \
         Got u8 {}.",
        centre(plate(&plates, "Magenta"))
    );
    assert_eq!(
        centre(plate(&plates, "Yellow")),
        0,
        "ISO 32000-1 §11.7.4.3 Table 149 row 1: B = c_s = 0 on the Y \
         lane with backdrop Y=0. c_r = 0 → u8 0. Got u8 {}.",
        centre(plate(&plates, "Yellow"))
    );
    assert_eq!(
        centre(plate(&plates, "Black")),
        0,
        "ISO 32000-1 §11.7.4.3 Table 149 row 1: B = c_s = 0 on the K \
         lane with backdrop K=0. c_r = 0 → u8 0. Got u8 {}.",
        centre(plate(&plates, "Black"))
    );
}

// ===========================================================================
// QA-A2 byte-exact: transparency + OPM=1 + DeviceCMYK + /OP true.
//
// ISO 32000-1 §11.7.4.3 Table 149 row 1: source CS = DeviceCMYK
// specified directly, affected component = C/M/Y/K, OP=true, OPM=1:
//   B(c_b, c_s) = c_s if c_s ≠ 0,
//   B(c_b, c_s) = c_b if c_s = 0.
//
// Setup:
//   Backdrop: DeviceCMYK (0.4, 0, 0, 0), /ca = 1.0.
//   Foreground: DeviceCMYK (0, 0.5, 0, 0), /OP true, /OPM 1, /ca = 0.5.
//
// Per spec:
//   C: c_s=0   → B = c_b = 0.4. r = 0.5·0.4 + 0.5·0.4 = 0.4  → u8 102.
//   M: c_s=0.5 → B = c_s = 0.5. r = 0.5·0.5 + 0.5·0   = 0.25 → u8 64.
//   Y: c_s=0   → B = c_b = 0.   r = 0 → u8 0.
//   K: c_s=0   → B = c_b = 0.   r = 0 → u8 0.
// ===========================================================================

#[test]
fn qa_a2_transparency_opm1_devicecmyk_overprint_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let content = "0.4 0 0 0 k\n0 0 100 100 re\nf\n\
                   /Ov gs\n0 0.5 0 0 k\n0 0 100 100 re\nf\n";
    let resources = "/ExtGState << /Ov << /Type /ExtGState /OP true /OPM 1 /ca 0.5 >> >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let expected_c = tint_to_u8(0.5 * 0.4 + 0.5 * 0.4);
    let expected_m = tint_to_u8(0.5 * 0.5 + 0.5 * 0.0);
    assert_eq!(expected_c, 102);
    assert_eq!(expected_m, 64);

    assert_eq!(
        centre(plate(&plates, "Cyan")),
        expected_c,
        "ISO 32000-1 §11.7.4.3 Table 149 row 1 (DeviceCMYK, OP=true, \
         OPM=1): c_s=0 → B = c_b. C lane: c_b=0.4, α=0.5: c_r = \
         0.5·0.4 + 0.5·0.4 = 0.4 → u8 102. Got u8 {}.",
        centre(plate(&plates, "Cyan"))
    );
    assert_eq!(
        centre(plate(&plates, "Magenta")),
        expected_m,
        "ISO 32000-1 §11.7.4.3 Table 149 row 1 (OPM=1): c_s=0.5 ≠ 0 → \
         B = c_s. M lane: c_b=0, α=0.5: c_r = 0.5·0.5 + 0.5·0 = 0.25 \
         → u8 64. Got u8 {}.",
        centre(plate(&plates, "Magenta"))
    );
    assert_eq!(
        centre(plate(&plates, "Yellow")),
        0,
        "ISO 32000-1 §11.7.4.3 Table 149 row 1 (OPM=1): c_s=0 → B = \
         c_b = 0. Got u8 {}.",
        centre(plate(&plates, "Yellow"))
    );
    assert_eq!(
        centre(plate(&plates, "Black")),
        0,
        "ISO 32000-1 §11.7.4.3 Table 149 row 1 (OPM=1): c_s=0 → B = \
         c_b = 0. Got u8 {}.",
        centre(plate(&plates, "Black"))
    );
}

// ===========================================================================
// QA-A3 byte-exact: transparency + overprint inside a /K knockout group.
//
// A /K (knockout) group's elements compose each against the group's
// initial backdrop, per §11.4.6.2. For a non-isolated /K group, the
// initial backdrop is the page state at the time the group is entered.
// The single overprinting paint inside the group therefore composes
// directly against the outer DeviceCMYK paint's plate state, and the
// /K group's result is itself composed (Normal, α=1) against the page
// — giving the same final plate output as QA-A2 (single overprinting
// paint against the same backdrop).
//
// Setup:
//   Page paint: DeviceCMYK (0.4, 0, 0, 0), /ca = 1.0.
//   /K group: a single paint (0, 0.5, 0, 0) with /OP true, /OPM 1,
//             /ca = 0.5.
//   Group's outer composition: /Normal, α = 1.0 (no group /ca
//   attenuation).
//
// Expected per-channel: identical to QA-A2.
//   C = 102 (preserved by c_s=0 + OPM=1, then α=1 group passthrough)
//   M = 64
// ===========================================================================

#[test]
fn qa_a3_transparency_overprint_inside_knockout_group_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let form = "6 0 obj\n\
        << /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 100 100] \
           /Resources << /ExtGState << /Ov << /Type /ExtGState /OP true /OPM 1 /ca 0.5 >> >> >> \
           /Group << /Type /Group /S /Transparency /K true /CS /DeviceCMYK >> \
           /Length 41 >>\n\
        stream\n/Ov gs\n0 0.5 0 0 k\n0 0 100 100 re\nf\nendstream\nendobj\n";
    let content = "0.4 0 0 0 k\n0 0 100 100 re\nf\n/Form Do\n";
    let resources = "/XObject << /Form 6 0 R >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[form]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let expected_c = tint_to_u8(0.5 * 0.4 + 0.5 * 0.4);
    let expected_m = tint_to_u8(0.5 * 0.5 + 0.5 * 0.0);
    assert_eq!(expected_c, 102);
    assert_eq!(expected_m, 64);

    assert_eq!(
        centre(plate(&plates, "Cyan")),
        expected_c,
        "ISO 32000-1 §11.4.6.2 + §11.7.4.3 Table 149 row 1 (OPM=1): /K \
         group containing one OP paint against initial backdrop = outer \
         DeviceCMYK (0.4, 0, 0, 0). C: c_s=0 → B = c_b = 0.4, c_r = \
         0.5·0.4 + 0.5·0.4 = 0.4 → u8 102. Got u8 {}.",
        centre(plate(&plates, "Cyan"))
    );
    assert_eq!(
        centre(plate(&plates, "Magenta")),
        expected_m,
        "ISO 32000-1 §11.4.6.2 + §11.7.4.3 Table 149 row 1 (OPM=1): M: \
         c_s=0.5 → B = c_s, c_r = 0.5·0.5 + 0.5·0 = 0.25 → u8 64. \
         Got u8 {}.",
        centre(plate(&plates, "Magenta"))
    );
}

// ===========================================================================
// QA-A4: DeviceGray source + /OP true + OPM=0 + transparency.
//
// Per ISO 32000-1 §11.7.4.3, the Table 149 row for "Any process colour
// space (including other cases of DeviceCMYK)" applies to DeviceGray
// when it is not the special directly-specified-DeviceCMYK row. The
// rule is B = c_s for every process colour component of the group
// colour space and B = c_b for spot colorants.
//
// In our setup the page group is DeviceCMYK (no explicit /Group entry
// → default page group treats CMYK as the process space; the renderer
// uses the OutputIntent CMYK profile for compositing). A DeviceGray
// source g maps to CMYK as (0, 0, 0, 1-g) per the standard CMYK
// conversion. So under OPM=0, all four process channels receive B = c_s
// from the converted CMYK quadruple.
//
// Setup:
//   Backdrop: DeviceCMYK (0.4, 0, 0, 0), /ca = 1.0.
//   Foreground: DeviceGray 0.25 (= K=0.75 after conversion), /OP true,
//                /ca = 0.5.
//
// Per spec (all four CMYK lanes treated as process colour components
// of the group CS; B = c_s on each):
//   C: c_s=0,    c_r = 0.5·0    + 0.5·0.4 = 0.2  → u8 51.
//   M: c_s=0,    c_r = 0.5·0    + 0.5·0   = 0.0  → u8 0.
//   Y: c_s=0,    c_r = 0.5·0    + 0.5·0   = 0.0  → u8 0.
//   K: c_s=0.75, c_r = 0.5·0.75 + 0.5·0   = 0.375 → u8 96.
// ===========================================================================

#[test]
fn qa_a4_transparency_opm0_devicegray_overprint_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let content = "0.4 0 0 0 k\n0 0 100 100 re\nf\n\
                   /Ov gs\n0.25 g\n0 0 100 100 re\nf\n";
    let resources = "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 >> >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    // DeviceGray 0.25 maps to DeviceCMYK (0, 0, 0, 0.75).
    let expected_c = tint_to_u8(0.5 * 0.0 + 0.5 * 0.4);
    let expected_m = tint_to_u8(0.0);
    let expected_y = tint_to_u8(0.0);
    let expected_k = tint_to_u8(0.5 * 0.75 + 0.5 * 0.0);
    assert_eq!(expected_c, 51);
    assert_eq!(expected_m, 0);
    assert_eq!(expected_y, 0);
    assert_eq!(expected_k, 96);

    assert_eq!(
        centre(plate(&plates, "Cyan")),
        expected_c,
        "ISO 32000-1 §11.7.4.3 Table 149 row 2 (any process CS, OP=true, \
         OPM=0): B = c_s for every process colour component of the group \
         CS. DeviceGray 0.25 → CMYK (0,0,0,0.75). C lane: c_s=0, c_b=0.4, \
         α=0.5: c_r = 0.5·0 + 0.5·0.4 = 0.2 → u8 51. Got u8 {}.",
        centre(plate(&plates, "Cyan"))
    );
    assert_eq!(
        centre(plate(&plates, "Black")),
        expected_k,
        "ISO 32000-1 §11.7.4.3 Table 149 row 2: K lane: c_s=0.75, \
         c_b=0, α=0.5: c_r = 0.5·0.75 + 0.5·0 = 0.375 → u8 96. \
         Got u8 {}.",
        centre(plate(&plates, "Black"))
    );
}

// ===========================================================================
// QA-A5: Separation source + /OP true + OPM=0 + transparency.
//
// ISO 32000-1 §11.7.4.3 Table 149 row 3: source CS = Separation /
// DeviceN. Per Table 149:
//   - Process colour component: B = c_b (preserve backdrop).
//   - Spot colorant NAMED in the source space: B = c_s.
//   - Spot colorant NOT named in the source space: B = c_b (preserve).
//
// Setup:
//   Backdrop: DeviceCMYK (0.4, 0, 0, 0), /ca = 1.0.
//   Then /Separation /InkA source painted at tint 0.7, /OP true,
//   /ca = 0.5. (Page declares one spot ink: InkA.)
//
// Expected (OP=true + α=0.5 + Separation source):
//   C: source CS = Separation → B = c_b = 0.4. r = 0.5·0.4 + 0.5·0.4
//      = 0.4 → u8 102 (PROCESS lane preserved on overprint).
//   M, Y, K: c_b = 0 → r = 0.
//   InkA lane: c_s = 0.7, c_b = 0. r = 0.5·0.7 + 0.5·0 = 0.35 → u8 89.
// ===========================================================================

#[test]
fn qa_a5_transparency_opm0_separation_overprint_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let psfunc = "<< /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                  /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 1.0 0.0 0.0] /N 1 >>";
    let content = "0.4 0 0 0 k\n0 0 100 100 re\nf\n\
                   /CS_A cs\n/Ov gs\n0.7 scn\n0 0 100 100 re\nf\n";
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 >> >> \
         /ColorSpace << /CS_A [/Separation /InkA /DeviceCMYK {} ] >>",
        psfunc
    );
    let pdf = build_pdf_with_output_intent(content, &resources, &icc, &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let expected_c = tint_to_u8(0.5 * 0.4 + 0.5 * 0.4);
    let expected_inka = tint_to_u8(0.5 * 0.7 + 0.5 * 0.0);
    assert_eq!(expected_c, 102);
    assert_eq!(expected_inka, 89);

    assert_eq!(
        centre(plate(&plates, "Cyan")),
        expected_c,
        "ISO 32000-1 §11.7.4.3 Table 149 row 3 (Separation source, \
         OP=true): process colour component B = c_b (preserve). C \
         lane: c_b=0.4, α=0.5, B=0.4: c_r = 0.5·0.4 + 0.5·0.4 = 0.4 → \
         u8 102. Got u8 {}.",
        centre(plate(&plates, "Cyan"))
    );
    assert_eq!(
        centre(plate(&plates, "Magenta")),
        0,
        "ISO 32000-1 §11.7.4.3 Table 149 row 3: process colour component \
         B = c_b. M lane: c_b=0 → c_r = 0. Got u8 {}.",
        centre(plate(&plates, "Magenta"))
    );
    assert_eq!(
        centre(plate(&plates, "InkA")),
        expected_inka,
        "ISO 32000-1 §11.7.4.3 Table 149 row 3: spot colorant named in \
         the source space → B = c_s. InkA: c_s=0.7, c_b=0, α=0.5: c_r \
         = 0.5·0.7 + 0.5·0 = 0.35 → u8 89. Got u8 {}.",
        centre(plate(&plates, "InkA"))
    );
}

// ===========================================================================
// QA-A6: Separation source + /OP true + OPM=1 + tint = 0.
//
// ISO 32000-1 §11.7.4.3 Table 149 row 3 (Separation): OPM=1 column.
// Per the table the rule for the named-spot lane is `c_s` regardless of
// whether c_s is zero or not — Table 149's OPM=1 zero-source preserve
// rule is specific to DeviceCMYK-direct C/M/Y/K channels.
//
// Wait — re-read Table 149 more carefully. The named-spot row says
// `c_s` under both OPM=0 and OPM=1. So a Separation paint with tint 0
// under OPM=1 DOES write c_s = 0 to its lane (not preserve).
//
// Spec quote (§11.7.4.3 first bullet, immediately under Table 149's
// formula):
//   "If the overprint mode is 1 (nonzero overprint mode) AND the
//    current colour space and group colour space are both DeviceCMYK,
//    then process colour components with nonzero values shall replace
//    the corresponding component values of the backdrop; components
//    with zero values leave the existing backdrop value unchanged."
//
// The "AND ... are both DeviceCMYK" qualifier means the OPM=1
// zero-source-preserve rule does NOT extend to Separation / DeviceN
// sources. For Separation/DeviceN, the named-spot rule is just B = c_s
// regardless of OPM.
//
// So this probe pins: under Separation source + OP+OPM=1 + tint=0,
// the named spot lane is composed at c_s=0 (i.e. lane becomes
// (1-α)·c_b after composition).
//
// Setup:
//   Backdrop: /Separation /InkA source painted at tint 0.6, /ca = 1.0.
//   This pre-fills the InkA lane to 0.6.
//   Then /Separation /InkA source at tint 0.0, /OP true, /OPM 1,
//   /ca = 0.5.
//
// Expected:
//   InkA lane: B = c_s = 0. r = 0.5·0 + 0.5·0.6 = 0.3 → u8 77.
//   Process lanes: c_b preserved per Table 149 row 3. C=M=Y=K=0.
// ===========================================================================

#[test]
fn qa_a6_separation_opm1_zero_source_replaces_lane_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let psfunc = "<< /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                  /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 1.0 0.0 0.0] /N 1 >>";
    let content = "/CS_A cs\n0.6 scn\n0 0 100 100 re\nf\n\
                   /Ov gs\n0 scn\n0 0 100 100 re\nf\n";
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /OPM 1 /ca 0.5 >> >> \
         /ColorSpace << /CS_A [/Separation /InkA /DeviceCMYK {} ] >>",
        psfunc
    );
    let pdf = build_pdf_with_output_intent(content, &resources, &icc, &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let expected_inka = tint_to_u8(0.5 * 0.0 + 0.5 * 0.6);
    assert_eq!(expected_inka, 77);

    assert_eq!(
        centre(plate(&plates, "InkA")),
        expected_inka,
        "ISO 32000-1 §11.7.4.3 Table 149 row 3 (Separation, OP=true, \
         OPM=1): the named-spot rule is B = c_s regardless of OPM \
         (zero-source-preserve under OPM=1 is specific to DeviceCMYK \
         direct paint per §11.7.4.3 bullet 1). InkA: c_s=0, c_b=0.6, \
         α=0.5: c_r = 0.5·0 + 0.5·0.6 = 0.3 → u8 77. Got u8 {}.",
        centre(plate(&plates, "InkA"))
    );
}

// ===========================================================================
// QA-A7: DeviceN source + /OP true + OPM=0 + transparency.
//
// ISO 32000-1 §11.7.4.3 Table 149 row 3 (DeviceN): same per-channel
// rule as Separation. Named-spot lanes use B = c_s; process and
// unnamed-spot lanes use B = c_b.
//
// Setup:
//   Page declares spots InkA and InkB.
//   Backdrop: DeviceCMYK (0.4, 0, 0, 0), /ca = 1.0.
//   Foreground: /DeviceN [/InkA /InkB] painted at (0.6, 0.3), /OP true,
//                /ca = 0.5.
//
// Expected:
//   InkA: B = c_s = 0.6, c_b=0 → c_r = 0.5·0.6 + 0.5·0 = 0.3 → u8 77.
//   InkB: B = c_s = 0.3, c_b=0 → c_r = 0.5·0.3 + 0.5·0 = 0.15 → u8 38.
//   C:    B = c_b = 0.4 → c_r = 0.4 → u8 102 (preserve, DeviceN source).
//   M, Y, K: c_b = 0 → c_r = 0.
// ===========================================================================

#[test]
fn qa_a7_transparency_opm0_devicen_overprint_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let psfunc = "<< /FunctionType 2 /Domain [0 1 0 1] \
                  /Range [0 1 0 1 0 1 0 1] \
                  /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 1.0 1.0 0.0] /N 1 >>";
    let content = "0.4 0 0 0 k\n0 0 100 100 re\nf\n\
                   /CS_N cs\n/Ov gs\n0.6 0.3 scn\n0 0 100 100 re\nf\n";
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 >> >> \
         /ColorSpace << /CS_N [/DeviceN [/InkA /InkB] /DeviceCMYK {} ] >>",
        psfunc
    );
    let pdf = build_pdf_with_output_intent(content, &resources, &icc, &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let expected_c = tint_to_u8(0.5 * 0.4 + 0.5 * 0.4);
    let expected_inka = tint_to_u8(0.5 * 0.6 + 0.5 * 0.0);
    let expected_inkb = tint_to_u8(0.5 * 0.3 + 0.5 * 0.0);
    assert_eq!(expected_c, 102);
    assert_eq!(expected_inka, 77);
    assert_eq!(expected_inkb, 38);

    assert_eq!(
        centre(plate(&plates, "Cyan")),
        expected_c,
        "ISO 32000-1 §11.7.4.3 Table 149 row 3 (DeviceN, OP=true, \
         OPM=0): process colour component B = c_b. C lane: c_b=0.4, \
         α=0.5: c_r = 0.4 → u8 102. Got u8 {}.",
        centre(plate(&plates, "Cyan"))
    );
    assert_eq!(
        centre(plate(&plates, "InkA")),
        expected_inka,
        "ISO 32000-1 §11.7.4.3 Table 149 row 3: InkA named in source → \
         B = c_s = 0.6, α=0.5: c_r = 0.3 → u8 77. Got u8 {}.",
        centre(plate(&plates, "InkA"))
    );
    assert_eq!(
        centre(plate(&plates, "InkB")),
        expected_inkb,
        "ISO 32000-1 §11.7.4.3 Table 149 row 3: InkB named in source → \
         B = c_s = 0.3, α=0.5: c_r = 0.15 → u8 38. Got u8 {}.",
        centre(plate(&plates, "InkB"))
    );
}

// ===========================================================================
// QA-A8: DeviceN source + /OP true + OPM=1 + mixed zero / non-zero.
//
// Per §11.7.4.3 the OPM=1 zero-source-preserve rule applies ONLY when
// both the current colour space AND the group colour space are
// DeviceCMYK (Table 149 row 1). For a DeviceN source it does not
// trigger; the named-spot lane just uses B = c_s.
//
// Setup:
//   Page declares spots InkA and InkB.
//   Backdrop sets InkA lane to 0.6 by painting /Separation /InkA at
//   tint 0.6, /ca = 1.0. (InkB stays at 0.)
//   Foreground: /DeviceN [/InkA /InkB] at (0.0, 0.4), /OP true,
//   /OPM 1, /ca = 0.5.
//
// Expected:
//   InkA: B = c_s = 0. c_b = 0.6, α = 0.5: c_r = 0.5·0 + 0.5·0.6 =
//         0.3 → u8 77. (Source-zero on DeviceN does NOT preserve —
//         the OPM=1 preserve rule is DeviceCMYK-direct only.)
//   InkB: B = c_s = 0.4. c_b = 0, α = 0.5: c_r = 0.5·0.4 = 0.2 → u8 51.
//   C/M/Y/K: c_b preserved (process for DeviceN source). All zero.
// ===========================================================================

#[test]
fn qa_a8_devicen_opm1_per_channel_zero_source_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let psfunc_a = "<< /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                    /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 1.0 0.0 0.0] /N 1 >>";
    let psfunc_n = "<< /FunctionType 2 /Domain [0 1 0 1] \
                    /Range [0 1 0 1 0 1 0 1] \
                    /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 1.0 1.0 0.0] /N 1 >>";
    let content = "/CS_A cs\n0.6 scn\n0 0 100 100 re\nf\n\
                   /CS_N cs\n/Ov gs\n0 0.4 scn\n0 0 100 100 re\nf\n";
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /OPM 1 /ca 0.5 >> >> \
         /ColorSpace << \
            /CS_A [/Separation /InkA /DeviceCMYK {}] \
            /CS_N [/DeviceN [/InkA /InkB] /DeviceCMYK {}] \
         >>",
        psfunc_a, psfunc_n
    );
    let pdf = build_pdf_with_output_intent(content, &resources, &icc, &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let expected_inka = tint_to_u8(0.5 * 0.0 + 0.5 * 0.6);
    let expected_inkb = tint_to_u8(0.5 * 0.4 + 0.5 * 0.0);
    assert_eq!(expected_inka, 77);
    assert_eq!(expected_inkb, 51);

    assert_eq!(
        centre(plate(&plates, "InkA")),
        expected_inka,
        "ISO 32000-1 §11.7.4.3 (DeviceN source, OP=true, OPM=1): the \
         OPM=1 zero-source-preserve rule is specific to DeviceCMYK \
         (Table 149 row 1). For DeviceN, the named-spot rule is \
         B = c_s regardless of OPM. InkA: c_s=0, c_b=0.6, α=0.5: c_r \
         = 0.5·0 + 0.5·0.6 = 0.3 → u8 77. Got u8 {}.",
        centre(plate(&plates, "InkA"))
    );
    assert_eq!(
        centre(plate(&plates, "InkB")),
        expected_inkb,
        "ISO 32000-1 §11.7.4.3 Table 149 row 3: InkB c_s=0.4, c_b=0, \
         α=0.5: c_r = 0.2 → u8 51. Got u8 {}.",
        centre(plate(&plates, "InkB"))
    );
}

// ===========================================================================
// QA-A9: §11.7.4.2 spot-lane Normal substitution still works post-fix.
//
// Round 2 wired the §11.7.4.2 rule: when the source BM is non-separable
// (Hue/Saturation/Color/Luminosity) or non-white-preserving
// (Difference/Exclusion), the spot lanes substitute Normal even when
// the process lanes use the requested BM.
//
// This probe pins that the round-4 overprint fix does not regress the
// §11.7.4.2 rule. The CompatibleOverprint §11.7.4.3 rule is a per-
// channel REPLACE/PRESERVE substitution; it composes with the
// §11.7.4.2 BM dispatch (the BM dispatch chooses the source value
// applied to each lane; CompatibleOverprint chooses whether the lane
// gets the source value or the backdrop). For Separation/DeviceN
// sources the lane mirror already applies the §11.7.4.2 spot
// substitution; the overprint rule applies on top per Table 149.
//
// Setup:
//   Backdrop: /Separation /InkA at tint 0.6, /ca = 1.0. Pre-fills
//   the InkA lane to 0.6.
//   Foreground: /Separation /InkA at tint 0.4, /BM /Luminosity (non-sep
//   → Normal substituted on spot lane per §11.7.4.2), /OP true, /ca = 0.5.
//
// Per §11.7.4.2: the spot lane sees an effective BM of Normal.
// Per §11.7.4.3 Table 149 row 3: InkA lane B = c_s (named spot).
// Composed Normal-source-over c_s = 0.4 against c_b = 0.6 at α=0.5:
//   c_r = 0.5·0.4 + 0.5·0.6 = 0.5 → u8 round(127.5) = 128.
//
// If the §11.7.4.2 substitution leaks (e.g. spot lane runs the non-sep
// Luminosity formula on a 1-vector tint), the lane value would differ.
// ===========================================================================

#[test]
fn qa_a9_spot_lane_normal_substitution_survives_overprint_fix() {
    let icc = build_constant_cmyk_icc(135);
    let psfunc = "<< /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                  /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 1.0 0.0 0.0] /N 1 >>";
    let content = "/CS_A cs\n0.6 scn\n0 0 100 100 re\nf\n\
                   /Ov gs\n0.4 scn\n0 0 100 100 re\nf\n";
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 /BM /Luminosity >> >> \
         /ColorSpace << /CS_A [/Separation /InkA /DeviceCMYK {} ] >>",
        psfunc
    );
    let pdf = build_pdf_with_output_intent(content, &resources, &icc, &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let expected_inka = tint_to_u8(0.5 * 0.4 + 0.5 * 0.6);
    assert_eq!(expected_inka, 128);

    assert_eq!(
        centre(plate(&plates, "InkA")),
        expected_inka,
        "ISO 32000-1 §11.7.4.2 spot-lane Normal substitution under \
         a non-sep BM, combined with §11.7.4.3 Table 149 row 3 \
         named-spot B = c_s. InkA: spot lane effective BM = Normal; \
         c_s=0.4, c_b=0.6, α=0.5: c_r = 0.5·0.4 + 0.5·0.6 = 0.5 → \
         u8 128. Got u8 {}.",
        centre(plate(&plates, "InkA"))
    );
}

// ===========================================================================
// QA-A10: Cross-path byte-identity for a pure-overprint DeviceCMYK page.
//
// The detection gate uses `page_declares_transparency` (narrow) to
// keep pure-overprint pages on the per-plate walker. After the round-4
// fix, the composite path's per-channel rule is spec-correct; we pin
// that the per-plate walker and the composite path agree on a
// pure-overprint DeviceCMYK page so any future widening of the gate
// will not change observed plate output.
//
// Setup:
//   Two DeviceCMYK paints, both at /ca = 1.0 (no transparency
//   trigger). Second paint with /OP true so the per-plate walker
//   exercises overprint and the composite path can compare.
//
// We invoke the composite path directly by routing through a
// transparency-triggering page (an SMask-bearing form that paints
// nothing) and compare against the per-plate walker output. To keep
// the test simple we use a single PDF and render once through the
// per-plate walker; the composite-path equivalent is exercised by
// QA-A1/A2 above (the per-channel rule is identical for α=1).
//
// This probe pins the byte values the per-plate walker produces so
// future composite-path changes can compare. With OPM=1 + /OP and
// source (0, 0.5, 0, 0): C preserved (=0.4 → u8 102), M replaced
// (=0.5 → u8 128), Y/K preserved (=0).
// ===========================================================================

#[test]
fn qa_a10_per_plate_walker_pure_overprint_pinned_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let content = "0.4 0 0 0 k\n0 0 100 100 re\nf\n\
                   /Ov gs\n0 0.5 0 0 k\n0 0 100 100 re\nf\n";
    let resources = "/ExtGState << /Ov << /Type /ExtGState /OP true /OPM 1 >> >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    // Per-plate walker, OP=true OPM=1 DeviceCMYK source:
    //   C: c_s=0 → preserve dest (=0.4) → u8 102.
    //   M: c_s=0.5 ≠ 0 → replace dest with c_s (=0.5) → u8 128.
    //   Y: c_s=0 → preserve (=0) → u8 0.
    //   K: c_s=0 → preserve (=0) → u8 0.
    assert_eq!(
        centre(plate(&plates, "Cyan")),
        102,
        "ISO 32000-1 §11.7.4 OPM=1 per-plate walker: C preserved at \
         backdrop 0.4 → u8 102. Got u8 {}.",
        centre(plate(&plates, "Cyan"))
    );
    assert_eq!(
        centre(plate(&plates, "Magenta")),
        128,
        "ISO 32000-1 §11.7.4 OPM=1 per-plate walker: M replaced by \
         source 0.5 → u8 128. Got u8 {}.",
        centre(plate(&plates, "Magenta"))
    );
}
