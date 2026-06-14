//! Regression test: a Form XObject's content must inherit the constant
//! alpha (`ca`), the blend mode (`/BM`), and — when the form is a
//! transparency group — be composited as a unit, all per the graphics state
//! in effect at the `Do` operator.
//!
//! Per ISO 32000-1:2008 §8.10.1, invoking a form XObject is equivalent to
//! splicing its content stream into the page at the point of the `Do`,
//! within the current graphics state. The non-stroking constant alpha and
//! the blend mode set by an `ExtGState` before the `Do` therefore apply to
//! everything the form paints. §11.6.6 adds that a form carrying a
//! `/Group << /S /Transparency >>` dictionary is rendered into its own
//! backdrop and then composited into the page as a single object using the
//! caller's alpha and blend mode.
//!
//! The defect this pins: the renderer rendered form content with a fresh
//! default graphics state (opaque, Normal blend), so a form painted under
//! `ca 0.5` (or `/BM /Multiply`) was composited fully opaque — an inner
//! image / fill completely overwrote the backdrop instead of blending with
//! it. In a real spot-colour artwork this manifested as a watercolour
//! "swoosh" rendering the wrong hue: the top layer (a form-wrapped image
//! set to Multiply at 0.8) clobbered the layer beneath instead of
//! multiplying into it.
//!
//! Coverage:
//! - `..._inherits_caller_constant_alpha`   — `ca` reaches form content.
//! - `..._inherits_caller_blend_mode`       — `/BM /Multiply` reaches form
//!   content (distinct code path: blend-function dispatch, not just alpha).
//! - `..._transparency_group_composites_under_caller_alpha` — a `/Group`
//!   form is composited as a unit under the caller's `ca` (the group path,
//!   not the inline-splice path).
#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{ImageFormat, RenderOptions};

/// How the form is invoked, so one builder can drive every case.
struct FormCase {
    /// Backdrop fill painted as a 100×100 rect at the page origin.
    backdrop: (f32, f32, f32),
    /// Fill the form paints over the same 100×100 region.
    form_fill: (f32, f32, f32),
    /// Non-stroking constant alpha set on the caller's ExtGState.
    ca: f32,
    /// Blend mode name set on the caller's ExtGState (e.g. "Normal", "Multiply").
    blend: &'static str,
    /// When true, the form carries `/Group << /S /Transparency /I true >>`.
    group: bool,
}

/// Build a single-page PDF that:
/// - paints `backdrop` as a 100×100 rect at (0,0) — the lower-left quadrant;
/// - sets an ExtGState (`/ca`, `/CA`, `/BM`) then invokes a Form XObject that
///   paints `form_fill` over the same 100×100 region.
///
/// The form's *own* graphics state is the default (opaque, Normal blend); any
/// alpha / blend / grouping seen in the output must come from the caller.
fn build_form_pdf(case: &FormCase) -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200]\n\
           /Contents 4 0 R\n\
           /Resources << /XObject << /Fm0 5 0 R >> /ExtGState << /GSa 6 0 R >> >> >>\nendobj\n\n",
    );

    // Page content: backdrop rect, then GSa + invoke the form.
    let (br, bg, bb) = case.backdrop;
    let page_content = format!("{br} {bg} {bb} rg 0 0 100 100 re f\n/GSa gs\n/Fm0 Do");
    let page_content = page_content.as_bytes();
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(page_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // Obj 5: Form XObject painting `form_fill` over (0,0)-(100,100). When
    // `group` is set it declares an isolated transparency group so the
    // renderer must composite it as a unit under the caller's state.
    let (fr, fg, fb) = case.form_fill;
    let form_stream = format!("{fr} {fg} {fb} rg 0 0 100 100 re f");
    let form_stream = form_stream.as_bytes();
    let group_entry = if case.group {
        "/Group << /Type /Group /S /Transparency /I true >>\n"
    } else {
        ""
    };
    offsets.push(pdf.len());
    let form_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 200 200]\n\
            {group_entry}/Resources << >>\n/Length {} >>\nstream\n",
        form_stream.len()
    );
    pdf.extend_from_slice(form_hdr.as_bytes());
    pdf.extend_from_slice(form_stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // Obj 6: ExtGState with the caller's constant alpha and blend mode.
    offsets.push(pdf.len());
    let gs = format!(
        "6 0 obj\n<< /Type /ExtGState /ca {} /CA {} /BM /{} >>\nendobj\n\n",
        case.ca, case.ca, case.blend
    );
    pdf.extend_from_slice(gs.as_bytes());

    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n", n_obj);
    xref.push_str("0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{:010} 00000 n \n", off));
    }
    pdf.extend_from_slice(xref.as_bytes());
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        n_obj, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
    pdf
}

/// Average straight-alpha luminance of a pixel region (un-premultiplied),
/// per channel. Returns (r, g, b) means over pixels with non-zero alpha.
fn sample_rgb(data: &[u8], width: u32, x: u32, y: u32, w: u32, h: u32) -> (f32, f32, f32) {
    let (mut sr, mut sg, mut sb, mut count) = (0f32, 0f32, 0f32, 0f32);
    for py in y..(y + h) {
        for px in x..(x + w) {
            let idx = ((py * width + px) * 4) as usize;
            if idx + 3 < data.len() {
                let a = data[idx + 3] as f32;
                if a > 0.0 {
                    let scale = 255.0 / a;
                    sr += (data[idx] as f32 * scale).min(255.0);
                    sg += (data[idx + 1] as f32 * scale).min(255.0);
                    sb += (data[idx + 2] as f32 * scale).min(255.0);
                    count += 1.0;
                }
            }
        }
    }
    if count == 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        (sr / count, sg / count, sb / count)
    }
}

/// Render page 0 and sample the lower-left quadrant (where backdrop and form
/// overlap). At 72 DPI the 100×100 PDF rect maps 1:1 to device pixels; device
/// y is flipped, so the rect sits at the bottom of the 200×200 image.
fn render_and_sample(pdf: Vec<u8>) -> (f32, f32, f32) {
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic PDF");
    let mut options = RenderOptions::with_dpi(72);
    options.format = ImageFormat::RawRgba8;
    let img = pdf_oxide::rendering::render_page(&doc, 0, &options).expect("render");
    sample_rgb(&img.data, img.width, 25, 125, 50, 50)
}

#[test]
fn form_xobject_content_inherits_caller_constant_alpha() {
    // White (255) over black (0) at ca=0.5 composites to ~128 mid-gray.
    // The bug paints the form opaque, yielding ~255.
    let (r, g, b) = render_and_sample(build_form_pdf(&FormCase {
        backdrop: (0.0, 0.0, 0.0),
        form_fill: (1.0, 1.0, 1.0),
        ca: 0.5,
        blend: "Normal",
        group: false,
    }));
    let lum = (r + g + b) / 3.0;
    assert!(
        (lum - 128.0).abs() < 20.0,
        "form painted under ca=0.5 must blend white-over-black to ~128 \
         mid-gray; got luminance {lum:.1} (rgb {r:.0},{g:.0},{b:.0}). A value \
         near 255 means the form is composited opaque, ignoring caller alpha."
    );
}

#[test]
fn form_xobject_content_inherits_caller_blend_mode() {
    // Backdrop 50% gray (128); form paints 50% gray under /BM /Multiply at
    // full alpha. Multiply: 128 * 128/255 ≈ 64. If the blend mode is dropped
    // the form paints Normal-opaque → ~128, twice as light — a clear, narrow
    // discriminator that the *blend function* (not just alpha) is inherited.
    let (r, g, b) = render_and_sample(build_form_pdf(&FormCase {
        backdrop: (0.5, 0.5, 0.5),
        form_fill: (0.5, 0.5, 0.5),
        ca: 1.0,
        blend: "Multiply",
        group: false,
    }));
    let lum = (r + g + b) / 3.0;
    assert!(
        (lum - 64.0).abs() < 16.0,
        "form painted under /BM /Multiply must multiply 0.5×0.5 → ~64; got \
         luminance {lum:.1} (rgb {r:.0},{g:.0},{b:.0}). A value near 128 means \
         the blend mode is ignored and the form is painted Normal."
    );
}

#[test]
fn form_xobject_transparency_group_composites_under_caller_alpha() {
    // A form declaring an isolated transparency group, invoked under ca=0.5.
    // The group is rendered into its own backdrop (white form fill) and then
    // composited as a unit at 50% over the black page backdrop → ~128. This
    // drives the group-composite path (§11.6.6), distinct from the inline
    // splice exercised above; the bug composites the group opaque → ~255.
    let (r, g, b) = render_and_sample(build_form_pdf(&FormCase {
        backdrop: (0.0, 0.0, 0.0),
        form_fill: (1.0, 1.0, 1.0),
        ca: 0.5,
        blend: "Normal",
        group: true,
    }));
    let lum = (r + g + b) / 3.0;
    assert!(
        (lum - 128.0).abs() < 24.0,
        "isolated /Group form composited under ca=0.5 must yield ~128 \
         mid-gray; got luminance {lum:.1} (rgb {r:.0},{g:.0},{b:.0}). A value \
         near 255 means the group is composited opaque, ignoring caller alpha."
    );
}
