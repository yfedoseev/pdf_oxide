//! A JPEG 2000 image carrying an opacity channel still paints.
//!
//! A JPX codestream may hold an alpha channel beside its colour channels, so a
//! greyscale image decodes to two components rather than one. ISO 32000-1:2008
//! Table 89's `/SMaskInData` entry (`docs/spec/pdf.md`:14527) governs that
//! channel, and its default is 0:
//!
//! > **0** If present, encoded soft-mask image information shall be ignored.
//!
//! How many of the decoded components are colour is settled by the same
//! table's `/ColorSpace` entry (pdf.md:14487): "If **ColorSpace** is present,
//! any colour space specifications in the JPEG2000 data shall be ignored."
//! `/DeviceGray` therefore means one colour component and one to discard.
//!
//! The decoder used to reject any component count it did not recognise, so a
//! two-component image failed to extract, the `Do` was skipped, and the page
//! rendered **completely white** — total content loss. One real file is a
//! single 551x337 grey+alpha JPX covering its whole page; all four reference
//! engines paint it and we painted nothing.
//!
//! The fixture is generated, not taken from a report: a 16x16 grey+alpha JP2
//! written by OpenJPEG, holding a light square on a dark ground.

#![cfg(all(feature = "jpeg2000", feature = "rendering"))]

use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

const GRAY_ALPHA_JP2: &[u8] = include_bytes!("fixtures/jpx/gray_with_alpha.jp2");

/// One page whose entire content is the two-component JPX image.
fn page_with_jpx(color_space: &str) -> Vec<u8> {
    let content = b"q 100 0 0 100 0 0 cm /Im0 Do Q\n".to_vec();
    let mut pdf = Vec::new();
    let mut off = [0usize; 7];
    macro_rules! push {
        ($s:expr) => {
            pdf.extend_from_slice($s.as_bytes())
        };
    }
    push!("%PDF-1.7\n");
    off[1] = pdf.len();
    push!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    off[2] = pdf.len();
    push!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    off[3] = pdf.len();
    push!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
         /Contents 4 0 R /Resources << /XObject << /Im0 5 0 R >> >> >>\nendobj\n"
    );
    off[4] = pdf.len();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()));
    pdf.extend_from_slice(&content);
    push!("endstream\nendobj\n");
    off[5] = pdf.len();
    push!(format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width 16 /Height 16 \
         /ColorSpace {color_space} /BitsPerComponent 8 /Filter /JPXDecode \
         /Length {} >>\nstream\n",
        GRAY_ALPHA_JP2.len()
    ));
    pdf.extend_from_slice(GRAY_ALPHA_JP2);
    push!("\nendstream\nendobj\n");

    let xref = pdf.len();
    push!("xref\n0 6\n0000000000 65535 f \r\n");
    for id in 1..=5 {
        push!(format!("{:010} 00000 n \r\n", off[id]));
    }
    push!(format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"));
    pdf
}

/// Fraction of pixels carrying ink, and the mean tone.
fn ink_and_tone(pdf: Vec<u8>) -> (f64, f64) {
    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let img = render_page(&doc, 0, &RenderOptions::default()).expect("page renders");
    let px = image::load_from_memory(&img.data)
        .expect("PNG decodes")
        .to_rgba8();
    let n = px.pixels().len() as f64;
    let tone: f64 = px
        .pixels()
        .map(|p| (f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2])) / 3.0)
        .sum::<f64>()
        / n;
    let ink = px
        .pixels()
        .filter(|p| p[3] > 0 && (u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])) / 3 < 250)
        .count() as f64
        / n;
    (ink, tone)
}

#[test]
fn a_two_component_jpx_image_is_not_dropped() {
    let (ink, tone) = ink_and_tone(page_with_jpx("/DeviceGray"));

    // The fixture is mostly dark, so it must cover the page in ink. Rejecting
    // the component count skipped the image and left the page pure white:
    // ink 0.0, tone 255.0.
    assert!(
        ink > 0.5,
        "the JPX image did not paint: ink {ink:.5} tone {tone:.2} — a two-component \
         (grey + alpha) codestream is being rejected rather than having its \
         opacity channel dropped"
    );
    assert!(tone < 250.0, "page is essentially blank: mean tone {tone:.2}");
}

#[test]
fn the_extracted_image_is_greyscale_not_the_raw_two_channel_buffer() {
    // The counter-check: the opacity channel must be *dropped*, not folded in
    // as if it were colour. Reading a 2-channel buffer as if it were 1-channel
    // would interleave alpha into the samples and halve the effective width,
    // which shows up as a wrong tone rather than a missing image.
    let doc = PdfDocument::from_bytes(page_with_jpx("/DeviceGray")).expect("parse");
    let images = doc.extract_images(0).expect("extract images");
    assert_eq!(images.len(), 1, "expected exactly one image");

    let png = images[0].to_png_bytes().expect("encode as PNG");
    let decoded = image::load_from_memory(&png).expect("PNG decodes");
    assert_eq!(
        (decoded.width(), decoded.height()),
        (16, 16),
        "the image lost its geometry — the opacity channel was probably read as colour"
    );
}
