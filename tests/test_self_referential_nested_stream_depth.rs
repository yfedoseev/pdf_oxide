//! A content stream that names itself must be bounded, not fatal.
//!
//! ISO 32000-1:2008 §8.10 describes a Form XObject as a re-entrant content
//! stream and §8.7.3 a tiling pattern likewise; neither clause requires the
//! reference graph to be acyclic, and nothing in the file format prevents a
//! form's own `/Resources /XObject` from naming the form. Without a depth cap
//! that recursion overflows the stack — and a stack overflow is not a
//! catchable panic: under the release profile's `panic = "abort"` it takes the
//! host process down.
//!
//! Type 3 glyph chains and soft-mask chains were already capped this way. Form
//! XObjects and tiling patterns were the two that were not.
//!
//! Every assertion here is "the render returns".

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};
use pdf_oxide::PdfDocument;

/// Assemble a one-page PDF from raw object bodies, numbered from 5 upward.
fn build(resources: &str, content: &str, objects: &[(usize, String)]) -> Vec<u8> {
    let max_id = objects.iter().map(|(i, _)| *i).max().unwrap_or(4);
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; max_id + 2];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        &format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
             /Resources << {resources} >> /Contents 4 0 R >>"
        ),
    );
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    for (id, body) in objects {
        obj(&mut buf, &mut off, *id, body);
    }

    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", max_id + 1).as_bytes());
    for id in 1..=max_id {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n", max_id + 1).as_bytes(),
    );
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// A stream object body with a dictionary and inline data.
fn stream_body(dict: &str, data: &str) -> String {
    format!("<< {dict} /Length {} >>\nstream\n{data}\nendstream", data.len())
}

fn renders(pdf: Vec<u8>) -> bool {
    let doc = match PdfDocument::from_bytes(pdf) {
        Ok(d) => d,
        Err(_) => return false,
    };
    let mut opts = RenderOptions::default();
    opts.dpi = 72;
    opts.format = ImageFormat::RawRgba8;
    render_page(&doc, 0, &opts).is_ok()
}

/// A form whose own resources name itself.
#[test]
fn self_referential_form_xobject_terminates() {
    let pdf = build(
        "/XObject << /Fm0 5 0 R >>",
        "q /Fm0 Do Q\n",
        &[(
            5,
            stream_body(
                "/Type /XObject /Subtype /Form /BBox [0 0 100 100] \
                 /Resources << /XObject << /Fm0 5 0 R >> >>",
                "0 0 1 rg 0 0 10 10 re f\nq /Fm0 Do Q\n",
            ),
        )],
    );
    assert!(renders(pdf), "a self-referential form must not abort the render");
}

/// A two-form cycle: Fm0 draws Fm1, Fm1 draws Fm0. A cap on a single name
/// would miss this; the depth counter does not.
#[test]
fn mutually_recursive_form_xobjects_terminate() {
    let pdf = build(
        "/XObject << /Fm0 5 0 R >>",
        "q /Fm0 Do Q\n",
        &[
            (
                5,
                stream_body(
                    "/Type /XObject /Subtype /Form /BBox [0 0 100 100] \
                     /Resources << /XObject << /Fm1 6 0 R >> >>",
                    "q /Fm1 Do Q\n",
                ),
            ),
            (
                6,
                stream_body(
                    "/Type /XObject /Subtype /Form /BBox [0 0 100 100] \
                     /Resources << /XObject << /Fm0 5 0 R >> >>",
                    "q /Fm0 Do Q\n",
                ),
            ),
        ],
    );
    assert!(renders(pdf), "a form cycle must not abort the render");
}

/// A tiling pattern whose cell paints with the same pattern.
#[test]
fn self_referential_tiling_pattern_terminates() {
    let pdf = build(
        "/Pattern << /P0 5 0 R >>",
        "/Pattern cs /P0 scn 0 0 100 100 re f\n",
        &[(
            5,
            stream_body(
                "/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 \
                 /BBox [0 0 20 20] /XStep 20 /YStep 20 \
                 /Resources << /Pattern << /P0 5 0 R >> >>",
                "/Pattern cs /P0 scn 0 0 20 20 re f\n",
            ),
        )],
    );
    assert!(renders(pdf), "a self-referential tiling pattern must not abort the render");
}

/// The caps must not break legitimate nesting: a form drawing a different
/// form, three deep, still paints.
#[test]
fn legitimately_nested_forms_still_render() {
    let pdf = build(
        "/XObject << /Fm0 5 0 R >>",
        "q /Fm0 Do Q\n",
        &[
            (
                5,
                stream_body(
                    "/Type /XObject /Subtype /Form /BBox [0 0 100 100] \
                     /Resources << /XObject << /Fm1 6 0 R >> >>",
                    "q /Fm1 Do Q\n",
                ),
            ),
            (
                6,
                stream_body(
                    "/Type /XObject /Subtype /Form /BBox [0 0 100 100] \
                     /Resources << /XObject << /Fm2 7 0 R >> >>",
                    "q /Fm2 Do Q\n",
                ),
            ),
            (
                7,
                stream_body(
                    "/Type /XObject /Subtype /Form /BBox [0 0 100 100]",
                    "0 0 0 rg 10 10 50 50 re f\n",
                ),
            ),
        ],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let mut opts = RenderOptions::default();
    opts.dpi = 72;
    opts.format = ImageFormat::RawRgba8;
    let img = render_page(&doc, 0, &opts).expect("render");
    let (w, h) = (img.width as usize, img.height as usize);
    let at = (h / 2 * w + w / 2) * 4;
    assert!(
        img.data[at] < 128 && img.data[at + 1] < 128 && img.data[at + 2] < 128,
        "three-deep legitimate nesting must still paint, got {:?}",
        &img.data[at..at + 3]
    );
}
