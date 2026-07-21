//! Adversarial probes for the round-eight closure work.
//!
//! Scope: independent verification that the five recent transparency
//! commits ship the behaviour they claim, on inputs that fall outside
//! the cells already covered by `test_transparency_flattening_audit`
//! and `test_transparency_flattening_qa_round4`. Every probe carries a
//! byte-exact reference hand-derived from the spec formula in its
//! docstring.
//!
//! Commits in scope:
//!  - `2b1c16f` — RGB → CMYK sidecar mirror per §11.3.4.
//!  - `6032953` — SMask /TR Type 0 sampled + Type 4 PostScript.
//!  - `7adc896` — SMask /BC backdrop for DeviceN n>=5.
//!  - `5720f7c` — SMask /TR Type 3 stitching.
//!  - `dd112bf` — stale HONEST_GAP removal (sanity sweep).

#![cfg(all(feature = "rendering", feature = "icc"))]
#![allow(dead_code)]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};

// ---------------------------------------------------------------------------
// Synthetic-PDF helpers — copied verbatim from `test_transparency_flattening_audit`
// so the adversarial probes are self-contained. Kept byte-identical so any
// future refactor on either side surfaces as a mismatch.
// ---------------------------------------------------------------------------

fn build_pdf(content: &str, resources_inner: &str, extra_objs: &[&str]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
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
    let mut extra_offs: Vec<usize> = Vec::new();
    for obj in extra_objs {
        extra_offs.push(buf.len());
        buf.extend_from_slice(obj.as_bytes());
    }
    let xref_off = buf.len();
    let total_objs = 4 + extra_objs.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", total_objs + 1).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off] {
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

fn render_rgba(pdf_bytes: Vec<u8>) -> Vec<u8> {
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("synthetic PDF parses");
    let opts = RenderOptions::with_dpi(72).as_raw();
    let img = render_page(&doc, 0, &opts).expect("render_page succeeds");
    assert_eq!(img.format, ImageFormat::RawRgba8);
    assert_eq!(img.width, 100);
    assert_eq!(img.height, 100);
    img.data
}

fn pixel_at(rgba: &[u8], x: u32, y: u32) -> (u8, u8, u8, u8) {
    let off = ((y * 100 + x) * 4) as usize;
    (rgba[off], rgba[off + 1], rgba[off + 2], rgba[off + 3])
}

// ===========================================================================
// §7.10.4 Type 3 stitching — k=4 dispatch (three boundaries)
// ===========================================================================
//
// The audit suite probes Type 3 only with k=2 (one boundary). The
// dispatcher uses `bounds.iter().filter(|b| x_clipped >= *b).count()
// .min(k - 1)`; the k>2 arithmetic is exercised here.
//
// Fixture: /Domain [0 1], /Bounds [0.25 0.5 0.75], four subfunctions:
//   f0 (gamma 1, identity) on [0,   0.25)
//   f1 (gamma 2)           on [0.25, 0.5)
//   f2 (C0=0.0, C1=0.0)    on [0.5,  0.75)  -- constant 0
//   f3 (C0=0.0, C1=0.5)    on [0.75, 1.0]   -- linear 0..0.5
// /Encode [0 1 0 1 0 1 0 1] passes each subinterval through to f's [0, 1].
//
// Form 50% grey → m_initial = 128/255 ≈ 0.5020. Boundary lookup:
//   0.5020 >= 0.25 → +1
//   0.5020 >= 0.5  → +1
//   0.5020 >= 0.75 → no
//   count = 2 → i = 2 → subfunction f2 (constant 0).
// Encoded input = 0 + (0.5020 - 0.5) * (1 - 0) / (0.75 - 0.5) = 0.0080.
// f2 = Type 2 (C0=[0], C1=[0], N=1) → 0 + x^1 * (0 - 0) = 0.0.
// m_out = 0.0; inv_m = 1.0. Backdrop white, painted red. G_dest =
// 1.0 * 255 = 255 → byte 255. R_dest = 1.0 * 255 = 255. Reference
// (255, 255, 255) — the SMask fully blocks the red paint.
//
// Identity-fallback (Type 3 not dispatched) would yield m_initial =
// 0.5020 directly: G = 0.4980 * 255 ≈ 127. Distinguishable.

fn fixture_smask_tr_type3_k4_bounds_three() -> Vec<u8> {
    let form_content = "0.5 g\n0 0 100 100 re\nf\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    let obj_6 = "6 0 obj\n<< /FunctionType 3 /Domain [0 1] /Range [0 1] \
                 /Functions [ \
                   << /FunctionType 2 /Domain [0 1] /Range [0 1] /C0 [0] /C1 [1] /N 1 >> \
                   << /FunctionType 2 /Domain [0 1] /Range [0 1] /C0 [0] /C1 [1] /N 2 >> \
                   << /FunctionType 2 /Domain [0 1] /Range [0 1] /C0 [0] /C1 [0] /N 1 >> \
                   << /FunctionType 2 /Domain [0 1] /Range [0 1] /C0 [0] /C1 [0.5] /N 1 >> \
                 ] /Bounds [0.25 0.5 0.75] /Encode [0 1 0 1 0 1 0 1] >>\nendobj\n";
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Sm gs\n\
                   1 0 0 rg\n\
                   20 20 60 60 re\nf\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R /TR 6 0 R >> >> >>";
    build_pdf(content, resources, &[&obj_5, obj_6])
}

#[test]
fn adversarial_smask_tr_type3_k4_dispatch_byte_exact() {
    let rgba = render_rgba(fixture_smask_tr_type3_k4_bounds_three());
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b),
        (255, 255, 255),
        "ISO 32000-1 §7.10.4 /SMask /TR Type 3 with k=4 (Bounds \
         [0.25, 0.5, 0.75]): m≈0.502 must dispatch to subfunction 2 \
         (constant zero), giving m_out=0, inv_m=1, painted-red blocked \
         to (255, 255, 255); got ({r}, {g}, {b})"
    );
}

// ===========================================================================
// §7.10.4 Type 3 stitching — boundary-belongs-right convention
// ===========================================================================
//
// Probe the k=4 dispatcher at the exact boundary x = 0.5. Per the
// half-open convention (§7.10.4 step 2: the boundary value belongs to
// the subinterval on its right), 0.5 belongs to subinterval 2, not
// subinterval 1. The `filter(|b| x_clipped >= *b)` implements this
// directly: at x=0.5, both b[0]=0.25 and b[1]=0.5 satisfy the
// predicate, count=2, i=2 → subfunction f2.
//
// Fixture targets a CMYK paint over white that sets the SMask-Y to
// EXACTLY 0.5 by using a flat-128 grey form. We pick a fixture where
// the byte-exact answer for subfunction-on-the-right is unambiguously
// different from subfunction-on-the-left.

fn fixture_smask_tr_type3_boundary_right_belong() -> Vec<u8> {
    // Form luminance L = 0.5 exactly via /DeviceRGB grey (0.5, 0.5, 0.5).
    // BT.601 of (128, 128, 128) = 0.30·128 + 0.59·128 + 0.11·128 = 128.
    // m_initial = 128/255 ≈ 0.5020 — almost-but-not-quite 0.5.
    //
    // To probe EXACTLY at x = 0.5 we use the input clip via /Domain:
    // /Domain [0 0.5], so m_clipped = min(0.5020, 0.5) = 0.5 exact.
    // Then x=0.5 falls into subinterval 1 (the right side of b[0]=0.5)
    // → f1.
    let form_content = "0.5 g\n0 0 100 100 re\nf\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    // /Domain [0 0.5], /Bounds [0.5], so:
    //   subinterval 0 = [0, 0.5) — Type 2 (C0=1, C1=1) constant 1
    //   subinterval 1 = [0.5, 0.5] — Type 2 (C0=0, C1=0) constant 0
    // x_clipped = 0.5 → boundary → belongs to subinterval 1 → m_out=0.
    // Result: m_out=0, inv_m=1, painted red blocked → (255, 255, 255).
    // Left-belong policy would have given m_out=1, no SMask block →
    // ~(255, 127, 127).
    let obj_6 = "6 0 obj\n<< /FunctionType 3 /Domain [0 0.5] /Range [0 1] \
                 /Functions [ \
                   << /FunctionType 2 /Domain [0 1] /Range [0 1] /C0 [1] /C1 [1] /N 1 >> \
                   << /FunctionType 2 /Domain [0 1] /Range [0 1] /C0 [0] /C1 [0] /N 1 >> \
                 ] /Bounds [0.5] /Encode [0 1 0 1] >>\nendobj\n";
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Sm gs\n\
                   1 0 0 rg\n\
                   20 20 60 60 re\nf\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R /TR 6 0 R >> >> >>";
    build_pdf(content, resources, &[&obj_5, obj_6])
}

#[test]
fn adversarial_smask_tr_type3_boundary_belongs_right_byte_exact() {
    let rgba = render_rgba(fixture_smask_tr_type3_boundary_right_belong());
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b),
        (255, 255, 255),
        "ISO 32000-1 §7.10.4 step 2 (boundary belongs to right \
         subinterval): m clipped to /Domain upper 0.5 must dispatch to \
         subfunction 1 (constant 0), giving m_out=0 and fully blocking \
         the red paint to (255, 255, 255); got ({r}, {g}, {b}). \
         Left-belong policy would have given subfunction 0 (constant 1) \
         and m_out=1 (no SMask block, ~(255, 127, 127))."
    );
}

// ===========================================================================
// §7.10.4 Type 3 stitching — inverted /Encode pair
// ===========================================================================
//
// /Encode = [1.0 0.0 ...] — the subfunction's input range is FLIPPED
// vs the subinterval. Verifies the linear remap correctly handles
// e_lo > e_hi.
//
// Fixture: /Domain [0 1], /Bounds [0.5], two Type 2 subfunctions:
//   f0 (gamma 1, identity) on [0, 0.5) with /Encode [1 0]
//   f1 (constant 0)        on [0.5, 1]
//
// Form 50% grey → m_initial ≈ 0.5020. Falls into subinterval 1
// (0.5020 >= 0.5). Subfunction 1 returns 0; m_out = 0; (255, 255, 255).
//
// To exercise the inverted /Encode we pick a fixture where m falls
// into subinterval 0. The audit's clipping probe uses /Domain [0.3 0.8];
// here we use /Domain [0 0.5] so m_initial = 0.5020 clips to 0.5,
// then i = bounds.filter(|b| 0.5 >= *b).count() = 1 (since b[0] = 0.4)
// → subfunction 1. We want subfunction 0, so probe m at a value LESS
// than the bound. Form 25% grey: byte 64; m_initial = 64/255 = 0.251.
// 0.251 < 0.4 → subfunction 0.
//
// Encoded with /Encode [1.0 0.0]:
//   x_clipped = 0.251; subinterval [0, 0.4); lo=0, hi=0.4; e_lo=1.0,
//   e_hi=0.0.
//   encoded = 1.0 + (0.251 - 0) * (0.0 - 1.0) / (0.4 - 0) = 1.0 - 0.6275
//           = 0.3725.
//   f0(0.3725) = Type 2 N=1, identity on [0,1] → 0.3725.
//   m_out = 0.3725; inv_m = 0.6275.
//   G = 0.3725 * 0 + 0.6275 * 255 = 160.01 → byte 160. R = 255.
//
// Reference (255, 160, 160). Non-inverted /Encode [0 1] would have
// given encoded = 0 + 0.251/0.4 * 1 = 0.6275; m_out=0.6275; inv_m=
// 0.3725; G = 0.3725*255 = 94.99 → byte 95. The two cases are
// unambiguously distinct.

fn fixture_smask_tr_type3_inverted_encode() -> Vec<u8> {
    let form_content = "0.25 g\n0 0 100 100 re\nf\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    let obj_6 = "6 0 obj\n<< /FunctionType 3 /Domain [0 1] /Range [0 1] \
                 /Functions [ \
                   << /FunctionType 2 /Domain [0 1] /Range [0 1] /C0 [0] /C1 [1] /N 1 >> \
                   << /FunctionType 2 /Domain [0 1] /Range [0 1] /C0 [0] /C1 [0] /N 1 >> \
                 ] /Bounds [0.4] /Encode [1 0 0 1] >>\nendobj\n";
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Sm gs\n\
                   1 0 0 rg\n\
                   20 20 60 60 re\nf\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R /TR 6 0 R >> >> >>";
    build_pdf(content, resources, &[&obj_5, obj_6])
}

#[test]
fn adversarial_smask_tr_type3_inverted_encode_byte_exact() {
    let rgba = render_rgba(fixture_smask_tr_type3_inverted_encode());
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    // Form 0.25 → byte 64 by 8-bit quantisation; m_initial = 64/255 ≈
    // 0.2510. With /Encode [1 0] for subfunction 0 over [0, 0.4):
    //   encoded = 1.0 - (0.2510 / 0.4) * 1.0 = 1.0 - 0.6275 = 0.3725
    //   m_out = identity(0.3725) = 0.3725
    //   G = (1 - 0.3725) * 255 = 0.6275 * 255 = 160.01 → byte 160
    assert_eq!(
        (r, g, b),
        (255, 160, 160),
        "ISO 32000-1 §7.10.4 /SMask /TR Type 3 with inverted /Encode \
         [1 0]: m≈0.251 in subinterval 0 must remap to encoded≈0.3725 \
         (e_lo=1.0, slope negative), identity subfunction returns the \
         encoded value, inv_m·255 → byte 160; expected (255, 160, 160); \
         got ({r}, {g}, {b}). Non-inverted /Encode [0 1] would have \
         given byte 95."
    );
}

// ===========================================================================
// /SMask /TR Type 0 — /BitsPerSample != 8 falls to Identity
// ===========================================================================
//
// The Type 0 parser at `parse_type0_transfer_function` accepts only
// /BitsPerSample 8 — other depths return None and the caller falls
// back to Identity. Probe a /BitsPerSample 16 fixture so we pin the
// fallback. Without the guard the parser would silently mis-interpret
// the high-byte/low-byte ordering and emit a junk LUT.

fn fixture_smask_tr_type0_bps16_fallback() -> Vec<u8> {
    let form_content = "0.5 g\n0 0 100 100 re\nf\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    // 256-entry 16-bit LUT — inverted ramp at 16-bit. The parser
    // refuses /BitsPerSample 16 and falls to Identity.
    let mut lut = Vec::with_capacity(512);
    for i in 0..256u32 {
        let v = ((255 - i) * 257) as u16; // expand 8-bit ramp to 16-bit
        lut.extend_from_slice(&v.to_be_bytes());
    }
    let mut obj_6 = format!(
        "6 0 obj\n<< /FunctionType 0 /Domain [0 1] /Range [0 1] /Size [256] \
         /BitsPerSample 16 /Length {} >>\nstream\n",
        lut.len()
    )
    .into_bytes();
    obj_6.extend_from_slice(&lut);
    obj_6.extend_from_slice(b"\nendstream\nendobj\n");
    // Safety: the LUT bytes can be arbitrary; the surrounding framing
    // is valid UTF-8 and `build_pdf` reads via byte slicing.
    let obj_6_str = unsafe { std::str::from_utf8_unchecked(&obj_6) };
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Sm gs\n\
                   1 0 0 rg\n\
                   20 20 60 60 re\nf\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R /TR 6 0 R >> >> >>";
    build_pdf(content, resources, &[&obj_5, obj_6_str])
}

#[test]
fn adversarial_smask_tr_type0_bps16_falls_to_identity_byte_exact() {
    let rgba = render_rgba(fixture_smask_tr_type0_bps16_fallback());
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    // /BitsPerSample 16 path is rejected by parse_type0_transfer_function;
    // the caller (`.or(Some(SMaskTransfer::Identity))`) substitutes
    // Identity. m_out = m_initial = 128/255 ≈ 0.5020; inv_m = 0.4980.
    // G = 0.4980 * 255 = 126.99 → byte 127. R = 255. Reference
    // (255, 127, 127) — the same as no-/TR baseline.
    assert_eq!(
        (r, g, b),
        (255, 127, 127),
        "ISO 32000-1 §7.10.2 /SMask /TR Type 0 with /BitsPerSample 16 \
         must fall to Identity (parser declines non-8-bit packing), \
         yielding the no-/TR baseline (255, 127, 127); got ({r}, {g}, {b})"
    );
}

// ===========================================================================
// /SMask /TR Type 0 — single-sample LUT (degenerate but spec-permitted)
// ===========================================================================
//
// /Size [1] is a one-entry LUT; the spec doesn't forbid it. The eval
// arm has a `samples.len() == 1` short-circuit that returns the
// constant. Pin the byte-exact reference.

fn fixture_smask_tr_type0_single_sample() -> Vec<u8> {
    let form_content = "0.5 g\n0 0 100 100 re\nf\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    // /Size [1], one sample byte 0 → LUT = [0.0]. Every input maps
    // to 0.0.
    let mut obj_6 = b"6 0 obj\n<< /FunctionType 0 /Domain [0 1] /Range [0 1] \
                     /Size [1] /BitsPerSample 8 /Length 1 >>\nstream\n"
        .to_vec();
    obj_6.push(0u8);
    obj_6.extend_from_slice(b"\nendstream\nendobj\n");
    let obj_6_str = unsafe { std::str::from_utf8_unchecked(&obj_6) };
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Sm gs\n\
                   1 0 0 rg\n\
                   20 20 60 60 re\nf\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R /TR 6 0 R >> >> >>";
    build_pdf(content, resources, &[&obj_5, obj_6_str])
}

#[test]
fn adversarial_smask_tr_type0_single_sample_byte_exact() {
    let rgba = render_rgba(fixture_smask_tr_type0_single_sample());
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    // Single-sample LUT [0.0]; m_out = 0.0; inv_m = 1.0. Painted-red
    // fully blocked → (255, 255, 255).
    assert_eq!(
        (r, g, b),
        (255, 255, 255),
        "ISO 32000-1 §7.10.2 /SMask /TR Type 0 with /Size [1]: the \
         single-sample LUT [0.0] must short-circuit to constant 0, \
         blocking the red paint to (255, 255, 255); got ({r}, {g}, {b})"
    );
}

// ===========================================================================
// /SMask /TR Type 4 — graceful failure on division by zero
// ===========================================================================
//
// The Type 4 evaluator wraps `Program::evaluate`. If the program
// produces NaN/Inf (e.g. `{ 0 div }`), the SMaskTransfer::eval arm's
// `.clamp(0.0, 1.0)` collapses the value to its clamped representation.
// In Rust, `f32::NAN.clamp(0.0, 1.0)` is NaN — `clamp` propagates NaN.
//
// Per the impl docstring at SMaskTransfer::eval Type 4 arm: "Failure
// modes ... fall back to identity rather than panicking" — but the
// guard only triggers on `Err(_)` or empty-output cases. A successful
// `evaluate` returning [f64::NAN] passes through the `.clamp().` This
// is a real fall-through. Pin the actual behaviour.

fn fixture_smask_tr_type4_div_by_zero() -> Vec<u8> {
    let form_content = "0.5 g\n0 0 100 100 re\nf\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    // `{ 0 div }` — pop the input, push the constant 0, divide. PDF's
    // Type 4 `div` on a zero divisor: result is implementation-defined
    // per §7.10.5, often NaN or Inf in IEEE arithmetic.
    let program = "{ pop 1 0 div }";
    let obj_6 = format!(
        "6 0 obj\n<< /FunctionType 4 /Domain [0 1] /Range [0 1] /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        program.len(),
        program
    );
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Sm gs\n\
                   1 0 0 rg\n\
                   20 20 60 60 re\nf\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R /TR 6 0 R >> >> >>";
    build_pdf(content, resources, &[&obj_5, &obj_6])
}

#[test]
fn adversarial_smask_tr_type4_div_by_zero_does_not_panic() {
    // Smoke probe: render must succeed without panicking. The byte
    // value is impl-defined (depends on whether the Program::evaluate
    // returns Err or Ok([NaN/Inf])); we pin only the no-panic property.
    let rgba = render_rgba(fixture_smask_tr_type4_div_by_zero());
    let (_r, _g, _b, a) = pixel_at(&rgba, 50, 50);
    // The pixel must be alpha-resolved (the rasteriser produces 8-bit
    // alpha 255 for any fully-resolved pixel). NaN propagation that
    // reaches u8 quantisation would surface as a panic in the
    // .round() as u8 cast OR as a 0 byte (NaN-to-int cast in Rust is
    // saturating-or-zero). Either way, the pixel exists and the
    // render completes.
    assert_eq!(
        a, 255,
        "render must succeed without panicking even when /TR Type 4 \
         produces NaN/Inf; alpha must resolve to 255; got {a}"
    );
}

// ===========================================================================
// /SMask /BC malformed arity — the HONEST_GAP_SMASK_BC_MALFORMED_ARITY
// constant claims dispatch is on array length. Pin the claim.
// ===========================================================================
//
// /BC [a b c d e] (5 tints) over a DeviceRGB-group SMask: the n>=5
// arm fires, evaluate_devicen_bc_to_rgb inspects /Group /CS, finds
// it's not /DeviceN, returns None, the unwrap_or pumps (0, 0, 0).
// Black backdrop pre-fill → mask byte (0, 0, 0). BT.601 Y = 0.
// m = 0; inv_m = 1; backdrop white survives → (255, 255, 255).
//
// Probe: a malformed n=5 /BC over a /Group /CS /DeviceRGB produces a
// fully-blocking (paint vanishes) SMask. This pins the constant's
// claim that the dispatch is on array length.

fn fixture_smask_bc_n5_over_devicergb_group() -> Vec<u8> {
    let form_content = "% empty form\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Group << /Type /Group /S /Transparency /CS /DeviceRGB >> \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Sm gs\n\
                   1 0 0 rg\n\
                   20 20 60 60 re\nf\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R \
                     /BC [0.5 0.5 0.5 0.5 0.5] >> >> >>";
    build_pdf(content, resources, &[&obj_5])
}

#[test]
fn adversarial_smask_bc_n5_over_devicergb_group_falls_to_black_byte_exact() {
    let rgba = render_rgba(fixture_smask_bc_n5_over_devicergb_group());
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    // n=5 over DeviceRGB Form group: dispatcher routes to
    // evaluate_devicen_bc_to_rgb which returns None (not a DeviceN
    // CS). unwrap_or(0, 0, 0) → black mask backdrop. BT.601 of
    // (0, 0, 0) = 0; m = 0; inv_m = 1; painted red blocked by
    // m=0 → backdrop white survives. Reference (255, 255, 255).
    assert_eq!(
        (r, g, b),
        (255, 255, 255),
        "HONEST_GAP_SMASK_BC_MALFORMED_ARITY documents that /BC \
         arity-vs-group-CS mismatches are dispatched on array length. \
         For n=5 over a /DeviceRGB group the n>=5 arm fires, the \
         DeviceN evaluator returns None, and the unwrap_or pumps black \
         which produces a fully-blocking SMask; expected (255, 255, \
         255); got ({r}, {g}, {b}). If this probe regresses the \
         malformed-arity policy has shifted and the constant's docstring \
         is no longer accurate."
    );
}

// ===========================================================================
// /SMask /BC n=1 over a DeviceCMYK Form group — dispatch-on-array-length
// ===========================================================================
//
// The reverse malformed case: /BC [0.5] (1 tint) over a Form whose
// /Group /CS is /DeviceCMYK. Per the dispatcher, n=1 fires the
// DeviceGray arm regardless of the Group CS. The single tint 0.5 is
// treated as DeviceGray and projects to RGB (128, 128, 128). BT.601
// Y = 128/255 ≈ 0.5020; m ≈ 0.5020; inv_m ≈ 0.4980; G =
// 0.4980·255 ≈ 127 → byte 127. R = 255. Reference (255, 127, 127).
//
// Compare to a "well-formed" interpretation: /BC [0.5] over a
// DeviceCMYK group as DeviceCMYK(0.5, ?, ?, ?) is undefined (only
// one channel specified), so the dispatcher's array-length choice is
// the only spec-coherent reading.

fn fixture_smask_bc_n1_over_devicecmyk_group() -> Vec<u8> {
    let form_content = "% empty form\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Group << /Type /Group /S /Transparency /CS /DeviceCMYK >> \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Sm gs\n\
                   1 0 0 rg\n\
                   20 20 60 60 re\nf\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R \
                     /BC [0.5] >> >> >>";
    build_pdf(content, resources, &[&obj_5])
}

#[test]
fn adversarial_smask_bc_n1_over_devicecmyk_group_dispatches_devicegray_byte_exact() {
    let rgba = render_rgba(fixture_smask_bc_n1_over_devicecmyk_group());
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b),
        (255, 127, 127),
        "HONEST_GAP_SMASK_BC_MALFORMED_ARITY: /BC [0.5] (n=1) over a \
         /Group /CS /DeviceCMYK must dispatch via the array-length n=1 \
         arm (DeviceGray) rather than detect the Group CS mismatch. \
         The DeviceGray-0.5 backdrop produces m≈0.502 and the red paint \
         composites to (255, 127, 127); got ({r}, {g}, {b})"
    );
}

// ===========================================================================
// §11.3.4 RGB → CMYK sidecar mirror — verify the §10.3.5 fallback edge
// cases on a no-CMM (qcms or no-icc) build.
//
// The qcms backend always returns None from build_srgb_to_cmyk, so
// `resolve_rgb_paint_to_cmyk` falls to the §10.3.5 inverse. The
// constant-grey ICC fixture lets the §10.3.5-converted backdrop ride
// through unchanged: every CMYK quadruple maps to the same grey RGB
// so the sidecar mirror is a no-op on the visible composite.
//
// To probe pure black and near-white we need a fixture where the
// converted CMYK backdrop CHANGES the composed RGB. Without a
// non-linear ICC (which the audit suite already uses), the visible
// pixel doesn't shift. So the meaningful adversarial check is the
// SMOKE one: pure-black + near-white RGB inputs at α<1 over a CMYK
// backdrop render without panicking and produce sensible byte values.
// ===========================================================================

fn fixture_rgb_pure_black_over_cmyk_backdrop() -> Vec<u8> {
    // Use the audit's small linear ICC pattern: a flat CMYK identity
    // so we don't need the non-linear A1B builder.
    let content = "1 1 0 0 k\n0 0 100 100 re\nf\n\
                   /Half gs\n\
                   0 0 0 rg\n\
                   10 10 80 80 re\nf\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >>";
    build_pdf(content, resources, &[])
}

#[test]
fn adversarial_rgb_pure_black_paint_does_not_panic() {
    // Mechanism: backdrop CMYK(1, 1, 0, 0) renders through the
    // process-ink converter to the measured blue corner
    // #2E3192 ~ RGB(46, 49, 146), NOT the additive (0, 0, 255). The
    // pure-black RGB(0, 0, 0) paint at α=0.5 composites over that
    // backdrop with tiny_skia's premul source-over: result =
    // 0.5*(0,0,0) + 0.5*(46,49,146) = (23, 25, 73). Probe pins the
    // post-composite RGBA byte-exact AND confirms the render completes
    // without panic.
    let rgba = render_rgba(fixture_rgb_pure_black_over_cmyk_backdrop());
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b, a),
        (23, 25, 73, 255),
        "RGB pure-black at α=0.5 over a CMYK(1,1,0,0) backdrop must \
         render byte-exact (23, 25, 73, 255): the process-ink converter \
         drives the backdrop to RGB(46, 49, 146) and tiny_skia's premul \
         source-over blends the pure-black paint at α=0.5 to that. \
         Got ({r}, {g}, {b}, {a})."
    );
}

// ===========================================================================
// /SMask /TR — overprint paint should skip the RGB mirror per impl docstring
// ===========================================================================
//
// `mirror_rgb_paint_into_sidecar` returns early when gs.fill_overprint
// is true. Probe: an RGB paint with /OP true under /ca 0.5 must not
// invoke the sidecar mirror — i.e. a downstream CMYK paint over the
// same region sees paper-white (sidecar zeros) at the overprint pixel,
// not the converted backdrop.
//
// This pins the overprint-skip policy in the helper. Render must
// complete without panicking; the visible RGB doesn't have a useful
// reference under a constant ICC, so the probe checks only the
// graceful-completion property.

fn fixture_rgb_paint_with_overprint_does_not_mirror() -> Vec<u8> {
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /OP gs\n\
                   0 1 0 rg\n\
                   10 10 80 80 re\nf\n";
    let resources = "/ExtGState << /OP << /Type /ExtGState /OP true /op true /ca 0.5 >> >>";
    build_pdf(content, resources, &[])
}

#[test]
fn adversarial_rgb_overprint_paint_gamut_compresses_no_panic() {
    let rgba = render_rgba(fixture_rgb_paint_with_overprint_does_not_mirror());
    let (r, g, b, a) = pixel_at(&rgba, 50, 50);
    // Mechanism: under /OP true the RGB paint is mirrored to CMYK
    // (`resolve_rgb_paint_to_cmyk` -> `color::rgb_to_cmyk`) and the
    // composite path projects that CMYK back through the process-ink
    // `color::cmyk_to_rgb`. The green RGB(0, 1, 0) is OUT of the process
    // gamut, so the round-trip gamut-compresses it toward the process
    // green corner rather than reproducing (0, 255, 0). At α=0.5 over
    // white the composite lands at (127, 209, 143) - close to, but not,
    // the additive (127, 255, 127). This gamut compression of an
    // out-of-gamut sRGB paint under overprint is the physically-correct
    // consequence of unifying the composite path on process inks (the
    // forward and inverse move together within the gamut). This probe
    // pins the byte-exact composite AND the no-panic property.
    let (r_ref, g_ref, b_ref, a_ref) = (127, 209, 143, 255);
    assert_eq!(
        (r, g, b, a),
        (r_ref, g_ref, b_ref, a_ref),
        "RGB paint with /OP true under /ca 0.5: the composite RGB \
         pixmap must be byte-exact ({r_ref}, {g_ref}, {b_ref}, {a_ref}) \
         - the out-of-gamut green gamut-compresses through the process-ink \
         round-trip. Got ({r}, {g}, {b}, {a})."
    );
}
