//! A form XObject's `/Resources` may be written as an indirect reference.
//!
//! ISO 32000-1:2008 §7.3.10 (`docs/spec/pdf.md`:2032) states the general rule:
//!
//! > Except were documented to the contrary any object value may be a direct
//! > or an indirect reference
//!
//! Nothing in Table 79 (`docs/spec/pdf.md`:15249) documents `/Resources` to the
//! contrary, and that same entry makes the form's own dictionary the only place
//! its resources can be found — for an independent form XObject the resources
//! "shall not be promoted to the outer content stream's resource dictionary".
//!
//! The renderer resolved the reference on the operator-execution path but not
//! before seeding its font and colour-space caches, and the seeding routine
//! matches only `Object::Dictionary`. An indirect `/Resources` therefore loaded
//! nothing and reported success. The form's fonts were then absent from the
//! cache, so its text fell back to a system font matched on the *resource name*
//! and the raw content bytes were painted as Latin-1 — a page of subset-encoded
//! prose came out as scattered punctuation, at roughly a tenth of the expected
//! ink, while `extract_text` (which reads `/ToUnicode`) stayed correct.
//!
//! A Type 3 font is used here so the fixture stays a synthetic PDF with no
//! embedded font program: its glyph is a filled rectangle, so "the font was
//! found" and "the font was not found" differ by a large, unambiguous amount of
//! ink rather than by glyph shape.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};
use pdf_oxide::PdfDocument;

/// Build a page whose sole content is `/Fm0 Do`, where the form draws one
/// Type 3 glyph — a filled 70×70 rectangle — using a font reachable only
/// through the form's own `/Resources`.
///
/// When `indirect` is true the form's `/Resources` is `9 0 R`; otherwise the
/// same dictionary is written inline. The two must render identically.
fn form_with_type3_font(indirect: bool) -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    macro_rules! obj {
        ($s:expr) => {{
            offsets.push(pdf.len());
            pdf.extend_from_slice($s.as_ref());
        }};
    }
    pdf.extend_from_slice(b"%PDF-1.7\n");

    obj!(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    obj!(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    // The page itself declares no /Font at all — exactly the shape of the
    // documents that exposed this: the fonts live only inside the form.
    obj!(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100]\n\
             /Contents 4 0 R /Resources << /XObject << /Fm0 5 0 R >> >> >>\nendobj\n"
        .as_ref());

    let content = b"/Fm0 Do";
    obj!(format!(
        "4 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        content.len(),
        String::from_utf8_lossy(content)
    )
    .into_bytes());

    let form = b"BT /T3 100 Tf 0 0 0 rg 15 15 Td (A) Tj ET";
    let res = if indirect {
        "9 0 R".to_string()
    } else {
        "<< /Font << /T3 6 0 R >> >>".to_string()
    };
    obj!(format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /FormType 1 \
         /BBox [0 0 100 100] /Resources {} /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        res,
        form.len(),
        String::from_utf8_lossy(form)
    )
    .into_bytes());

    obj!(b"6 0 obj\n\
             << /Type /Font /Subtype /Type3 /FontBBox [0 0 700 700]\n\
                /FontMatrix [0.001 0 0 0.001 0 0]\n\
                /FirstChar 65 /LastChar 65 /Widths [700]\n\
                /Encoding 7 0 R /CharProcs 8 0 R >>\nendobj\n"
        .as_ref());
    obj!(b"7 0 obj\n<< /Type /Encoding /Differences [65 /rect] >>\nendobj\n".as_ref());
    obj!(b"8 0 obj\n<< /rect 10 0 R >>\nendobj\n".as_ref());
    // Object 9 is the form's resource dictionary, referenced when `indirect`.
    // It is written either way so the object numbering — and therefore every
    // byte offset in the xref — is identical between the two arms.
    obj!(b"9 0 obj\n<< /Font << /T3 6 0 R >> >>\nendobj\n".as_ref());

    let glyph = b"700 0 0 0 700 700 d1 0 0 700 700 re f";
    obj!(format!(
        "10 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        glyph.len(),
        String::from_utf8_lossy(glyph)
    )
    .into_bytes());

    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {n_obj}\n0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{off:010} 00000 n \n"));
    }
    pdf.extend_from_slice(xref.as_bytes());
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {n_obj} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n")
            .as_bytes(),
    );
    pdf
}

/// Count pixels that are not white.
fn ink(pdf: &[u8]) -> usize {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("open");
    // `RenderOptions` carries a crate-private field, so struct-update syntax
    // is rejected from outside the crate; set the public fields on the default.
    let mut opts = RenderOptions::default();
    opts.dpi = 72;
    opts.format = ImageFormat::RawRgba8;
    let img = render_page(&doc, 0, &opts).expect("render");
    img.data
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] < 250 || px[1] < 250 || px[2] < 250)
        .count()
}

#[test]
fn an_indirect_form_resource_dictionary_finds_the_forms_font() {
    let direct = ink(&form_with_type3_font(false));
    let indirect = ink(&form_with_type3_font(true));

    // The glyph is a 70×70 rectangle at 72 dpi, so the direct arm establishes
    // what "the font was found" looks like. Assert that first, so a fixture
    // that draws nothing at all cannot make the comparison below pass
    // vacuously.
    assert!(
        direct > 4000,
        "the inline-resources arm should paint the Type 3 rectangle, got {direct} ink pixels"
    );

    assert_eq!(
        indirect, direct,
        "a form whose /Resources is an indirect reference must render \
         identically to one whose /Resources is inline: got {indirect} ink \
         pixels against {direct}"
    );
}
