//! Round 5 QA pass probes for issue #46.
//!
//! Adversarial scrutiny of round-5's three commits — covers the gaps
//! the design+impl agent's own probes did not pin:
//!
//!  - A1-QA2: DeviceN /Process /ColorSpace [/ICCBased <N=3 stream>] —
//!            exercises the round-5 ICCBased N=3 arm of
//!            `extract_process_paint_cmyk`. A1 only pinned the N=4
//!            path; N=3 follows the /DeviceRGB shape (§10.3.5 inverse
//!            from RGB tints) and was untested.
//!  - A1-QA3: DeviceN /Process /ColorSpace [/ICCBased <N=1 stream>] —
//!            ICCBased N=1 arm (§10.3.5 inverse from a single grey
//!            tint, K = 1 − g, C = M = Y = 0). Untested.
//!  - A3-QA1: pure /Separation paint AFTER a DeviceCMYK paint —
//!            verifies the round-4 stale-CMYK clear (in
//!            `SetFillColorN`) combined with the round-5
//!            `source_for_overprint` precedence flip routes through
//!            SeparationOrDeviceN (preserve backdrop on process lanes),
//!            NOT OtherProcess. A regression would dispatch as
//!            OtherProcess and corrupt process plates with the stale
//!            CMYK from the prior `k` operator.
//!  - B1-QA1: ImageMask `/Decode [1 0]` override — verifies the §8.9.6.2
//!            stencil-mask byte semantic flips correctly under the
//!            non-default decode array (0 = no-paint, 1 = paint). B1
//!            only covered the default decode; the inverted decode
//!            tests the data-convention symmetry.
//!
//! Spec citations:
//!  - ISO 32000-1 §8.6.6.5  — DeviceN /Process attribution; ICCBased
//!    /Process /ColorSpace N=3/N=1 arms
//!  - ISO 32000-1 §8.9.6.2  — Stencil Masking (NOT §8.9.6.4 Colour Key
//!    Masking as the agent's B1 docstring mis-cited)
//!  - ISO 32000-1 §10.3.5   — additive-clamp colour conversion
//!  - ISO 32000-1 §11.3.3   — single shape / opacity per pixel
//!  - ISO 32000-1 §11.7.4.3 — CompatibleOverprint (Table 149 row 2)

#![cfg(all(feature = "rendering", feature = "icc", feature = "test-support"))]
#![allow(dead_code)]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::render_separations;

// ===========================================================================
// Builder — shared with `test_46_round5_devicen_process_polish.rs`.
// Duplicated here to keep this QA file self-contained; the round-5
// design+impl probes use the identical bytes for cross-corpus
// comparability.
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

/// Build a constant-Lab CMYK ICC LUT profile (same as round-5 helper).
/// Used as the OutputIntent profile; the embedded /Process /ColorSpace
/// stream is a minimal dict with only /N — the round-5 reading does
/// not consult the profile bytes for the natural-form path.
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

// ===========================================================================
// A1-QA2 — DeviceN /Process /ColorSpace [/ICCBased <N=3 stream>].
//
// Round 5's `extract_process_paint_cmyk` ICCBased N=3 arm matches the
// named `/DeviceRGB` shape: tints in the embedded profile's RGB space
// are inverted via §10.3.5 (C = 1 − R, M = 1 − G, Y = 1 − B, K = 0).
// A1 only pinned the N=4 path. This probe pins N=3 byte-exact.
//
// Setup:
//   DeviceN colorants: [/R /G /B]  (process-only, no spot tail)
//   Process /ColorSpace [/ICCBased 6 0 R] with /N 3 in the stream dict
//   Process /Components [/R /G /B]
//   scn tints: (0.2, 0.6, 0.8) — interpreted by the natural-form rule
//     as additive RGB per the NChannel branch (the alternate reading
//     for NChannel non-CMYK process tints; for default DeviceN the
//     spec says all tints are subtractive, but the round-5 impl uses
//     §10.3.5 inverse for the N=3 ICCBased path, treating tints as
//     additive RGB per the natural-form sentence in NChannel — same
//     shape as the named /DeviceRGB arm).
//   /OP true, /ca 0.5; backdrop (0.4, 0, 0, 0) DeviceCMYK.
//
// Expected source CMYK (per round-5 N=3 arm):
//   C = 1 − 0.2 = 0.8
//   M = 1 − 0.6 = 0.4
//   Y = 1 − 0.8 = 0.2 (f32-inexact: 0.19999999)
//   K = 0
//
// §11.7.4.3 Table 149 row 2 (B = c_s), §11.3.3 at α=0.5:
//   C plate: 0.5·0.8 + 0.5·0.4 = 0.6 → u8 153
//   M plate: 0.5·0.4 + 0.5·0   = 0.2 → u8 51
//   Y plate: 0.5·0.19999999 + 0.5·0 = 0.099999996 → ×255 = 25.499999 → u8 25
//   K plate: 0
// ===========================================================================

#[test]
fn a1_qa2_devicen_process_iccbased_n3_overprint_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let psfunc = "<< /FunctionType 2 /Domain [0 1 0 1 0 1] \
                  /Range [0 1 0 1 0 1 0 1] \
                  /C0 [0 0 0 0] /C1 [1 1 1 1] /N 1 >>";
    let content = "0.4 0 0 0 k\n0 0 100 100 re\nf\n\
                   /CS_N cs\n/Ov gs\n0.2 0.6 0.8 scn\n0 0 100 100 re\nf\n";
    // Object 6: minimal ICCBased stream with /N 3. The round-5 reader
    // only consults the dict's /N entry, NOT the profile bytes.
    let process_icc_dict =
        "6 0 obj\n<< /N 3 /Length 4 >>\nstream\n\x00\x00\x00\x00\nendstream\nendobj\n";
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 >> >> \
         /ColorSpace << /CS_N [/DeviceN [/R /G /B] \
            /DeviceRGB {} \
            << /Process << /ColorSpace [/ICCBased 6 0 R] \
                          /Components [/R /G /B] >> >> \
         ] >>",
        psfunc
    );
    let pdf = build_pdf_with_output_intent(content, &resources, &icc, &[process_icc_dict]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let c = centre(plate(&plates, "Cyan"));
    let m = centre(plate(&plates, "Magenta"));
    let y = centre(plate(&plates, "Yellow"));
    let k = centre(plate(&plates, "Black"));

    assert_eq!(
        c, 153,
        "ISO 32000-1 §8.6.6.5 + §10.3.5 + §11.7.4.3: DeviceN /Process \
         /ColorSpace [/ICCBased <N=3>] inverts RGB tints to CMYK via the \
         §10.3.5 additive-clamp. R=0.2 → C = 1 - R = 0.8. Composite over \
         c_b=0.4 at α=0.5: c_r = 0.6 → u8 153. Got u8 {}. A regression \
         to 102 means the N=3 arm dispatched as preserve-backdrop \
         (SeparationOrDeviceN) instead of OtherProcess.",
        c
    );
    assert_eq!(
        m, 51,
        "ICCBased N=3 M lane: G=0.6 → M = 1 - G = 0.4. c_r = 0.2 → u8 51. \
         Got u8 {}.",
        m
    );
    assert_eq!(
        y, 25,
        "ICCBased N=3 Y lane: B=0.8 → Y = 1 - B = 0.2 in exact math; in \
         f32 the inverse picks up the inexact 0.8 representation: \
         1 - 0.8_f32 = 0.19999999. c_r×255 = 25.499998 → u8 round = 25. \
         Got u8 {}. (Same f32 chain A2 documents for /Process /DeviceRGB; \
         the ICCBased N=3 arm produces byte-identical output.)",
        y
    );
    assert_eq!(k, 0, "ICCBased N=3 K lane: §10.3.5 never produces K → 0. Got u8 {}.", k);
}

// ===========================================================================
// A1-QA3 — DeviceN /Process /ColorSpace [/ICCBased <N=1 stream>].
//
// Round 5's ICCBased N=1 arm matches the named `/DeviceGray` shape:
// K = 1 − g, C = M = Y = 0. Untested by the design+impl probes.
//
// Setup:
//   DeviceN colorants: [/Grey]
//   Process /ColorSpace [/ICCBased 6 0 R] with /N 1
//   Process /Components [/Grey]
//   scn tint: 0.3 — natural-form additive gray.
//   /OP true, /ca 0.5; backdrop (0, 0, 0, 0.4) DeviceCMYK.
//
// Expected source CMYK:
//   K = 1 − 0.3 = 0.7
//   C = M = Y = 0
//
// §11.7.4.3 + §11.3.3 at α=0.5, backdrop K=0.4:
//   K plate: 0.5·0.7 + 0.5·0.4 = 0.55 in exact math.
//   The sidecar quantises backdrop K=0.4 to u8 102 → reads back
//   102/255 = 0.40000001. composite c_r = 0.5·0.7 + 0.5·0.40000001 =
//   0.55000001 → ×255 = 140.25 → u8 round = 140.
// ===========================================================================

#[test]
fn a1_qa3_devicen_process_iccbased_n1_overprint_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let psfunc = "<< /FunctionType 2 /Domain [0 1] \
                  /Range [0 1 0 1 0 1 0 1] \
                  /C0 [0 0 0 0] /C1 [1 1 1 1] /N 1 >>";
    let content = "0 0 0 0.4 k\n0 0 100 100 re\nf\n\
                   /CS_N cs\n/Ov gs\n0.3 scn\n0 0 100 100 re\nf\n";
    let process_icc_dict =
        "6 0 obj\n<< /N 1 /Length 4 >>\nstream\n\x00\x00\x00\x00\nendstream\nendobj\n";
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 >> >> \
         /ColorSpace << /CS_N [/DeviceN [/Grey] \
            /DeviceGray {} \
            << /Process << /ColorSpace [/ICCBased 6 0 R] \
                          /Components [/Grey] >> >> \
         ] >>",
        psfunc
    );
    let pdf = build_pdf_with_output_intent(content, &resources, &icc, &[process_icc_dict]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let c = centre(plate(&plates, "Cyan"));
    let m = centre(plate(&plates, "Magenta"));
    let y = centre(plate(&plates, "Yellow"));
    let k = centre(plate(&plates, "Black"));

    assert_eq!(
        c, 0,
        "ISO 32000-1 §8.6.6.5: DeviceN /Process /ColorSpace [/ICCBased \
         <N=1>] sets K only — C lane gets 0 source. c_b=0 → c_r=0 → u8 0. \
         Got u8 {}.",
        c
    );
    assert_eq!(m, 0, "ICCBased N=1 M lane source = 0. Got u8 {}.", m);
    assert_eq!(y, 0, "ICCBased N=1 Y lane source = 0. Got u8 {}.", y);
    assert_eq!(
        k, 140,
        "ICCBased N=1 K lane: §10.3.5 inverse k = 1 - 0.3 = 0.7. \
         Composite over backdrop K=0.4 (quant 102/255 = 0.40000001) at \
         α=0.5: c_r = 0.5·0.7 + 0.5·0.40000001 = 0.55000001 → ×255 = \
         140.25 → u8 140. Got u8 {}. A regression to 102 indicates the \
         N=1 arm did not fire — source K was lost and backdrop K was \
         preserved.",
        k
    );
}

// ===========================================================================
// A3-QA1 — pure /Separation paint AFTER a DeviceCMYK paint.
//
// Verifies the round-4 stale-CMYK clear (SetFillColorN's
// `gs.fill_color_cmyk = None` at the top) combined with the round-5
// `source_for_overprint` precedence flip routes through
// SeparationOrDeviceN — process lanes preserve backdrop.
//
// The precedence flip means `color_cmyk = Some(_)` wins over
// `spot_inks` for composite-named spaces. If the round-4 clear were
// broken, this fixture would dispatch as OtherProcess (using the
// stale 0.4 cyan from the prior `k`) and the C plate would land on
// 115 instead of 102.
//
// Setup:
//   0.4 0 0 0 k       — DeviceCMYK fill_color_cmyk = Some((0.4, 0, 0, 0)).
//   0 0 100 100 re; f — paint the backdrop.
//   /CS_S cs          — enter /Separation /PMS185 — initial cmyk = None.
//   /Ov gs            — OP true, ca 0.5.
//   0.6 scn           — fill_spot_inks=[(PMS185, 0.6)], cmyk None.
//   0 0 100 100 re; f — paint with /Separation source.
//
// Expected:
//   C plate (process): backdrop preserved = 0.4 → u8 round(102).
//   M / Y / K: backdrop = 0 → u8 0.
//   PMS185 spot plate: 0.5·0.6 = 0.3 → u8 77 (per A3's analysis).
// ===========================================================================

#[test]
fn a3_qa1_separation_after_devicecmyk_routes_separationordevicen() {
    let icc = build_constant_cmyk_icc(135);
    let tint_func = "<< /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                    /C0 [0 0 0 0] /C1 [0 0.8 1 0] /N 1 >>";
    let content = "0.4 0 0 0 k\n0 0 100 100 re\nf\n\
                   /CS_S cs\n/Ov gs\n0.6 scn\n0 0 100 100 re\nf\n";
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 >> >> \
         /ColorSpace << /CS_S [/Separation /PMS185 /DeviceCMYK {}] >>",
        tint_func
    );
    let pdf = build_pdf_with_output_intent(content, &resources, &icc, &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let c = centre(plate(&plates, "Cyan"));
    let m = centre(plate(&plates, "Magenta"));
    let y = centre(plate(&plates, "Yellow"));
    let k = centre(plate(&plates, "Black"));
    let pms185 = centre(plate(&plates, "PMS185"));

    assert_eq!(
        c, 102,
        "ISO 32000-1 §11.7.4.3 Table 149 row 3 (Separation/DeviceN \
         class): process lanes preserve backdrop when the source is a \
         pure Separation paint. C plate = backdrop C = 0.4 → u8 102. \
         Got u8 {}. A regression to 115 indicates the round-5 \
         precedence flip dispatched as OtherProcess (because \
         fill_color_cmyk was stale from the prior `0.4 0 0 0 k`), \
         using c_s = 0.4 and producing c_r = 0.45 → u8 115. The \
         round-4 stale-CMYK clear in SetFillColorN must reset \
         fill_color_cmyk to None before the Separation arm — without \
         the reset, the round-5 precedence flip routes a pure \
         Separation paint through OtherProcess and corrupts the \
         process plates.",
        c
    );
    assert_eq!(m, 0, "Separation-after-CMYK: M plate = backdrop M = 0. Got u8 {}.", m);
    assert_eq!(y, 0, "Separation-after-CMYK: Y plate = backdrop Y = 0. Got u8 {}.", y);
    assert_eq!(k, 0, "Separation-after-CMYK: K plate = backdrop K = 0. Got u8 {}.", k);
    assert_eq!(
        pms185, 77,
        "Separation-after-CMYK: PMS185 spot plate gets the source tint \
         via the round-2 spot mirror. 0.5·0.6 = 0.3 in exact math; in \
         f32 0.5·0.6000000238 = 0.30000001 → ×255 = 76.500003 → u8 \
         round = 77 (Rust f32 round half-away-from-zero). Got u8 {}.",
        pms185
    );
}

// ===========================================================================
// B1-QA1 — ImageMask `/Decode [1 0]` override semantic.
//
// Per ISO 32000-1 §8.9.6.2 (NOT §8.9.6.4 which is Colour Key Masking):
//   - Default /Decode [0 1]: bit 0 paints with fill colour, bit 1
//     leaves previous contents unchanged.
//   - Override /Decode [1 0]: meanings reversed — bit 1 paints, bit 0
//     leaves unchanged.
//
// The same ImageMask data (`0x00` across all 4 row-bytes) under the
// default decode paints every pixel (B1 in `test_46_round5_image_
// pattern_preview.rs`). Under the inverted decode, no pixel is
// painted, so the PMS185 spot plate stays at 0 at every pixel — the
// /K group's initial backdrop is never overwritten.
//
// Setup mirrors B1 byte-for-byte except for /Decode [1 0] on each
// ImageMask object.
// ===========================================================================

#[test]
fn b1_qa1_imagemask_decode_inverted_no_paint_byte_exact() {
    let icc = build_constant_cmyk_icc(135);
    let tint_func = "<< /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1] \
                    /C0 [1 1 1] /C1 [0 1 0] /N 1 >>";
    let imgmask = "/CS_A cs\n\
                   0.4 scn\n\
                   q 100 0 0 100 0 0 cm /IM1 Do Q\n\
                   0.7 scn\n\
                   q 100 0 0 100 0 0 cm /IM2 Do Q\n";

    let form_stream_str = imgmask;
    let form_obj = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Group << /Type /Group /S /Transparency /K true /CS /DeviceRGB >> \
         /Resources << /ColorSpace << /CS_A [/Separation /PMS185 /DeviceRGB {tf}] >> \
            /XObject << /IM1 7 0 R /IM2 7 0 R >> >> \
         /Length {len} >>\nstream\n{stream}\nendstream\nendobj\n",
        tf = tint_func,
        len = form_stream_str.len(),
        stream = form_stream_str
    );
    // ImageMask object 7: 4×4 all-0s, BUT with /Decode [1 0] — under
    // the inverted decode bit 0 means "leave unchanged" (no paint).
    // Every pixel = bit 0 → no pixel paints.
    let imgmask_data: &[u8] = &[0x00, 0x00, 0x00, 0x00];
    let im_hdr = format!(
        "7 0 obj\n<< /Type /XObject /Subtype /Image /Width 4 /Height 4 \
         /ImageMask true /BitsPerComponent 1 /Decode [1 0] /Length {} >>\nstream\n",
        imgmask_data.len()
    );
    let mut im_obj = Vec::from(im_hdr.as_bytes());
    im_obj.extend_from_slice(imgmask_data);
    im_obj.extend_from_slice(b"\nendstream\nendobj\n");
    let im_obj_str = unsafe { String::from_utf8_unchecked(im_obj) };

    let content = "/K1 Do\n";
    let resources = format!(
        "/ColorSpace << /CS_A [/Separation /PMS185 /DeviceRGB {}] >> \
         /XObject << /K1 6 0 R >>",
        tint_func
    );
    let pdf = build_pdf_with_output_intent(
        content,
        &resources,
        &icc,
        &[form_obj.as_str(), im_obj_str.as_str()],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let pms185 = centre(plate(&plates, "PMS185"));

    // With /Decode [1 0] and all-0 data: bit 0 means "no paint" per
    // §8.9.6.2. No pixel is touched by either ImageMask. The /K
    // group's initial backdrop (PMS185 = 0) survives.
    assert_eq!(
        pms185, 0,
        "ISO 32000-1 §8.9.6.2 (Stencil Masking): /Decode [1 0] override \
         inverts the stencil bit semantic — a sample value of 0 leaves \
         previous contents unchanged. Probe data is 0x00 across all 4 \
         row-bytes (every pixel = bit 0); under [1 0] no pixel paints. \
         The /K group's initial backdrop PMS185 = 0 survives → u8 0. \
         Got u8 {}. A regression to 179 indicates /Decode [1 0] was \
         IGNORED — the impl treated the data as if the default decode \
         applied (every pixel paints with tint 0.7 → u8 179, the B1 \
         result). The §8.9.6.2 decode-flip path is unwired.",
        pms185
    );
}
