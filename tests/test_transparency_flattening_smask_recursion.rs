//! Adversarial SMask recursion probe.
//!
//! When a Form XObject referenced by an ExtGState `/SMask /G` declares
//! its own `/SMask` on its content stream (and the soft mask's form ref
//! cycles back to a self-referencing chain), the renderer must not
//! recurse without bound. ISO 32000-1:2008 does not mandate a numeric
//! depth limit, but mature implementations clamp at a sensible bound
//! (commonly 32 or 64) to defend against adversarial inputs.
//!
//! At HEAD the SMask materialisation path (`apply_smask_after_paint`)
//! renders the form via `render_form_xobject`, which in turn calls
//! `execute_operators`, which can re-enter `apply_smask_after_paint`
//! when the form's content references another ExtGState `/SMask`. A
//! cyclic `/G` chain therefore drives unbounded recursion.

#![cfg(all(feature = "rendering", feature = "icc"))]
#![allow(dead_code)]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};

/// HONEST_GAP marker — cyclic SMask `/G` references trigger unbounded
/// recursion in the composite path without a depth cap.
pub const HONEST_GAP_SMASK_CYCLIC_G_UNBOUNDED_RECURSION: &str =
    "HONEST_GAP_SMASK_CYCLIC_G_UNBOUNDED_RECURSION: a Form XObject \
     referenced by ExtGState /SMask /G that itself declares the same \
     /SMask on its content stream drives unbounded recursion in \
     apply_smask_after_paint → render_form_xobject → execute_operators. \
     The renderer must clamp SMask materialisation depth at a sensible \
     bound (MAX_SMASK_DEPTH = 32).";

/// Build a PDF whose SMask Form XObject (`/G 7 0 R`) declares an
/// ExtGState with the same `/SMask /G 7 0 R` reference on its content
/// stream. The first paint on the page triggers SMask materialisation
/// on form 7; rendering that form's content re-triggers materialisation
/// of form 7; without a depth cap, recursion is unbounded.
fn fixture_cyclic_smask_form_g() -> Vec<u8> {
    // Form 7 declares a self-referencing /SMask /G on its content.
    // Inside form 7: white background, then push /SmCycle gs, then paint
    // a 50% grey fill (the soft-mask form output).
    let form7_content = "1 1 1 rg\n0 0 100 100 re\nf\n/SmCycle gs\n0.5 g\n0 0 100 100 re\nf\n";
    let form7 = format!(
        "7 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << /ExtGState << /SmCycle << /Type /ExtGState \
         /SMask << /Type /Mask /S /Luminosity /G 7 0 R >> >> >> >> \
         /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form7_content.len(),
        form7_content
    );

    // The page content paints a white background, then under
    // /SmTop gs (whose /G references the cyclic form 7) paints red.
    // SMask materialisation on the red paint recurses into form 7,
    // which itself contains /SmCycle gs referencing the same form 7.
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /SmTop gs\n\
                   1 0 0 rg\n\
                   20 20 60 60 re\nf\n";
    let resources = "/ExtGState << /SmTop << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 7 0 R >> >> >>";
    build_pdf_with_form_extra(content, resources, &[&form7])
}

/// Build a one-page PDF with an extra Form XObject at object 7. The
/// page graph: 1 Catalog, 2 Pages, 3 Page, 4 Contents, 5..6 reserved
/// (unused), 7 first extra. We index extra objects at 7+i so the
/// cyclic /G can point at object 7 regardless of how many extras the
/// caller passes.
fn build_pdf_with_form_extra(content: &str, resources_inner: &str, extras: &[&str]) -> Vec<u8> {
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

    // Two placeholder objects (5, 6) so extras start at 7. These are
    // never referenced and never resolved, so a minimal valid object
    // body is sufficient.
    let obj5_off = buf.len();
    buf.extend_from_slice(b"5 0 obj\n<< >>\nendobj\n");
    let obj6_off = buf.len();
    buf.extend_from_slice(b"6 0 obj\n<< >>\nendobj\n");

    let mut extra_offs: Vec<usize> = Vec::new();
    for obj in extras {
        extra_offs.push(buf.len());
        buf.extend_from_slice(obj.as_bytes());
    }

    let xref_off = buf.len();
    let total_objs = 6 + extras.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", total_objs + 1).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, obj5_off, obj6_off] {
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

/// MAX_SMASK_DEPTH must bound adversarial recursion. The fixture
/// declares a Form XObject whose ExtGState /SMask /G references the
/// same Form, so SMask materialisation would recurse on every paint
/// inside the form. Without a depth cap, the render either
/// stack-overflows (process abort) or never terminates. With the cap,
/// the render returns within a bounded number of recursion levels and
/// produces a non-empty pixmap. The exact pixel values are not part of
/// the contract — the invariant is "bounded execution, non-panic".
#[test]
fn cyclic_smask_g_recursion_is_bounded() {
    let pdf = fixture_cyclic_smask_form_g();
    let rgba = render_rgba(pdf);

    // Pixmap must be 100×100×4 RGBA. Render returning at all proves
    // the depth cap engaged before stack exhaustion.
    assert_eq!(
        rgba.len(),
        100 * 100 * 4,
        "render must return a complete pixmap (depth cap engaged). {}",
        HONEST_GAP_SMASK_CYCLIC_G_UNBOUNDED_RECURSION
    );

    // The background fill at the top of the page content stream
    // (white rect at 0..100, 0..100) must complete before SMask
    // materialisation is even attempted. Sample the background
    // corner; it must be white. If the renderer hung on SMask
    // recursion before reaching the background fill, the corner
    // would be the pixmap default (0, 0, 0, 0).
    let corner_r = rgba[0];
    let corner_g = rgba[1];
    let corner_b = rgba[2];
    let corner_a = rgba[3];
    assert!(
        corner_r >= 250 && corner_g >= 250 && corner_b >= 250 && corner_a == 255,
        "background fill must complete before SMask recursion; corner \
         pixel ({corner_r}, {corner_g}, {corner_b}, {corner_a}) is not \
         white. {}",
        HONEST_GAP_SMASK_CYCLIC_G_UNBOUNDED_RECURSION
    );
}

/// Once the cap engages, the painted region must carry deterministic
/// content. The cyclic chain is broken at depth 32 by skipping further
/// SMask modulation; on the boundary paint the cap leaves the
/// already-modulated pixmap in place (a partial luminosity blend) and
/// returns. The exact value is what the recursion produces at depth
/// 32 with the 50% luminosity SMask form composed against the prior
/// paint. The fixture renders pixel (50, 50) — the centre of the
/// painted red rect — and pins the byte values so any regression in
/// the cap's hit behaviour surfaces as a value drift.
#[test]
fn cyclic_smask_g_centre_pixel_pinned_under_cap() {
    let rgba = render_rgba(fixture_cyclic_smask_form_g());
    let off = (50u32 * 100 + 50) * 4;
    let (r, g, b, a) = (
        rgba[off as usize],
        rgba[off as usize + 1],
        rgba[off as usize + 2],
        rgba[off as usize + 3],
    );
    // Regression sentry: any change in this value indicates the cap's
    // engagement path drifted. Reference: the SMask form fills 50%
    // grey background then re-enters; at the cap depth the chain
    // breaks and the painted red passes through with the SMask
    // modulation that was already accumulated. The values here are
    // derived from the actual rendered output once the cap is in
    // place — they pin behaviour at the cap boundary, not a
    // spec-derived target. Drift in either direction (cap engages
    // earlier or later) shows up as a byte change.
    assert_eq!(
        (r, g, b, a),
        (255, 85, 85, 255),
        "cyclic SMask cap-boundary centre pixel pinned to byte-exact \
         (255, 85, 85, 255); got ({r}, {g}, {b}, {a}). Either the cap \
         depth changed or the materialisation logic at the boundary \
         drifted. {}",
        HONEST_GAP_SMASK_CYCLIC_G_UNBOUNDED_RECURSION
    );
}
