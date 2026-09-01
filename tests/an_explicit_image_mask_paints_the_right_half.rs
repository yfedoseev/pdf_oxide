//! An image's explicit `/Mask` hides the masked samples, not the others.
//!
//! ISO 32000-1:2008 §8.9.6.4: an explicit mask is a stencil whose sample values
//! select which of the base image's samples are painted. Under the default
//! `/Decode [0 1]` a **0** sample paints and a **1** is masked out; `/Decode
//! [1 0]` reverses that. Getting the sense backwards paints exactly the
//! complement of what the file asks for — the image survives where it should be
//! hidden and vanishes where it should show.
//!
//! The fixture makes the two halves unmistakable: a solid red base image with a
//! 2x1 mask whose left sample is 0 (paint) and right sample is 1 (hide).

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

fn build(objects: Vec<Vec<u8>>) -> Vec<u8> {
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref = pdf.len();
    let n = objects.len() + 1;
    pdf.extend_from_slice(format!("xref\n0 {n}\n0000000000 65535 f \n").as_bytes());
    for off in &offsets {
        pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    pdf
}

fn stream(dict: &str, data: &[u8]) -> Vec<u8> {
    [
        format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes(),
        data.to_vec(),
        b"\nendstream".to_vec(),
    ]
    .concat()
}

/// `decode` is the mask's `/Decode` array, or empty for the default.
fn masked_image(decode: &str) -> Vec<u8> {
    // Base: 2x1 RGB, both samples solid red.
    let base: &[u8] = &[0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00];
    // Mask: 2x1, 1 bpc. Left sample 0, right sample 1 -> bits 0b01xxxxxx.
    let mask: &[u8] = &[0b0100_0000];
    let content = b"q 200 0 0 100 0 0 cm /Im0 Do Q\n";

    build(vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R \
           /Resources << /XObject << /Im0 5 0 R >> >> >>"
            .to_vec(),
        stream("", content),
        stream(
            "/Type /XObject /Subtype /Image /Width 2 /Height 1 \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Mask 6 0 R",
            base,
        ),
        stream(
            &format!(
                "/Type /XObject /Subtype /Image /Width 2 /Height 1 \
                 /ImageMask true /BitsPerComponent 1 {decode}"
            ),
            mask,
        ),
    ])
}

/// Mean **green** in the left and right halves.
///
/// Green is the discriminating channel, not red: the base image is pure red
/// (255, 0, 0) and the page is white (255, 255, 255), so red is 255 either way
/// and cannot tell "image survived" from "image masked out". Green is 0 where
/// the red survives and 255 where it does not.
///
/// Validated against two independent engines on this exact fixture before
/// being trusted — MuPDF gives (0.0, 252.4) and pdfium (0.0, 255.0), i.e. the
/// left half keeps its red and the right half is masked away.
fn halves(pdf: Vec<u8>) -> (f64, f64) {
    let doc = PdfDocument::from_bytes(pdf).expect("fixture parses");
    let img = render_page(&doc, 0, &RenderOptions::with_dpi(72)).expect("page renders");
    let px = image::load_from_memory(&img.data).expect("PNG decodes").to_rgba8();
    let (w, h) = (px.width(), px.height());
    let mean = |lo: u32, hi: u32| -> f64 {
        let mut sum = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..h {
            for x in lo..hi {
                sum += f64::from(px.get_pixel(x, y)[1]);
                n += 1.0;
            }
        }
        sum / n.max(1.0)
    };
    (mean(0, w / 2), mean(w / 2, w))
}

/// Under the default `/Decode`, the 0 sample paints and the 1 is hidden — so
/// the left half keeps the red and the right half does not.
#[test]
fn the_zero_sample_paints_under_the_default_decode() {
    let (left, right) = halves(masked_image(""));
    assert!(
        left < 60.0,
        "the 0-sample half lost its image (mean green {left:.1}, expected near 0 \
         where the red survives); §8.9.6.4 says a 0 sample paints under the \
         default /Decode. MuPDF and pdfium both give 0.0 here."
    );
    assert!(
        right > 200.0,
        "the 1-sample half was painted (mean green {right:.1}, expected near 255 \
         where it is masked away). MuPDF gives 252.4 and pdfium 255.0."
    );
}

/// `/Decode [1 0]` reverses the sense, so the halves swap. Asserting both
/// directions is what catches an inverted implementation: one direction alone
/// passes just as well when the polarity is backwards *and* the fixture is
/// symmetric.
#[test]
fn decode_one_zero_reverses_which_half_survives() {
    let (l_default, r_default) = halves(masked_image(""));
    let (l_inverted, r_inverted) = halves(masked_image("/Decode [1 0]"));
    let default_favours_left = l_default > r_default;
    let inverted_favours_left = l_inverted > r_inverted;
    assert_ne!(
        default_favours_left, inverted_favours_left,
        "/Decode [1 0] must swap which half survives — default \
         ({l_default:.1}, {r_default:.1}), inverted ({l_inverted:.1}, {r_inverted:.1})"
    );
}
