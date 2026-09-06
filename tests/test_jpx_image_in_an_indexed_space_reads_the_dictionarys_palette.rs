//! ISO 32000-1:2008 §7.4.9 — for a `/JPXDecode` image, the dictionary's
//! `/ColorSpace` decides how the samples are read (`docs/spec/pdf.md`:3143):
//!
//! > If present, it shall determine how the image samples are interpreted,
//! > and the colour space specifications in the JPEG2000 data shall be ignored.
//!
//! An `/Indexed` space makes each decoded sample a table index. A JP2 file may
//! carry a palette of its own for those same indices; if the decoder is left to
//! resolve it, one declared component comes back as three, and taking the
//! first of them turned a four-shades-of-blue image into `R = G = B`: the
//! whole page lost its tint. The index must go through the dictionary's
//! table, and the codestream's palette must not be consulted at all.
//!
//! The fixture is generated, not taken from a report: a 4x2 single-component
//! 4-bit codestream holding the indices `0 1 2 3 / 3 2 1 0`, wrapped in a JP2
//! whose own palette box is a grey ramp. The dictionary's palette is red,
//! green, blue, white. The page is the image scaled to 100x50 pt, and the
//! render is read at the centre of each 25-pixel cell — the surface the
//! defect was seen on, and one with no size floor for a 4x2 image.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

const INDEXED_JP2: &[u8] = include_bytes!("fixtures/jpx/indexed_with_own_palette.jp2");

const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];
const BLUE: [u8; 3] = [0, 0, 255];
const WHITE: [u8; 3] = [255, 255, 255];

/// One page whose only content is the 4x2 indexed JPX image.
fn page_with_indexed_jpx() -> Vec<u8> {
    let content = b"q 100 0 0 50 0 0 cm /Im0 Do Q\n".to_vec();
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
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 50] \
         /Contents 4 0 R /Resources << /XObject << /Im0 5 0 R >> >> >>\nendobj\n"
    );
    off[4] = pdf.len();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()));
    pdf.extend_from_slice(&content);
    push!("endstream\nendobj\n");
    off[5] = pdf.len();
    push!(format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width 4 /Height 2 \
         /ColorSpace [/Indexed /DeviceRGB 3 <FF000000FF000000FFFFFFFF>] \
         /BitsPerComponent 4 /Filter /JPXDecode /Length {} >>\nstream\n",
        INDEXED_JP2.len()
    ));
    pdf.extend_from_slice(INDEXED_JP2);
    push!("\nendstream\nendobj\n");

    let xref = pdf.len();
    push!("xref\n0 6\n0000000000 65535 f \r\n");
    for id in 1..=5 {
        push!(format!("{:010} 00000 n \r\n", off[id]));
    }
    push!(format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"));
    pdf
}

fn extracted_pixels() -> Vec<[u8; 3]> {
    let doc = PdfDocument::from_bytes(page_with_indexed_jpx()).expect("synthetic PDF parses");
    let img = render_page(&doc, 0, &RenderOptions::with_dpi(72)).expect("page renders");
    let decoded = image::load_from_memory(&img.data)
        .expect("PNG decodes")
        .to_rgb8();
    assert_eq!(
        (decoded.width(), decoded.height()),
        (100, 50),
        "the page is 100x50 pt at 72 dpi"
    );
    // Image row 0 is the top row of the page; each index cell is 25x25 px.
    let mut px = Vec::new();
    for row in 0..2u32 {
        for col in 0..4u32 {
            px.push(decoded.get_pixel(col * 25 + 12, row * 25 + 12).0);
        }
    }
    px
}

/// The palette entry a rendered pixel is nearest to. Scaling a 4x2 image to
/// 100x50 resamples it, so a cell centre is not the entry's exact value;
/// it is far nearer to that entry than to any other.
fn nearest(p: [u8; 3]) -> [u8; 3] {
    let d = |q: [u8; 3]| -> u32 {
        (0..3)
            .map(|i| (p[i] as i32 - q[i] as i32).unsigned_abs().pow(2))
            .sum()
    };
    *[RED, GREEN, BLUE, WHITE]
        .iter()
        .min_by_key(|&&q| d(q))
        .unwrap()
}

#[test]
fn each_index_is_looked_up_in_the_dictionarys_table() {
    let px = extracted_pixels();
    let seen: Vec<[u8; 3]> = px.iter().map(|&p| nearest(p)).collect();
    assert_eq!(
        seen,
        vec![RED, GREEN, BLUE, WHITE, WHITE, BLUE, GREEN, RED],
        "indices 0 1 2 3 / 3 2 1 0 must map through the /Indexed lookup table; pixels {px:?}"
    );
}

#[test]
fn test_codestreams_own_palette_is_ignored() {
    // The JP2's palette box is a grey ramp (0x40, 0x80, 0xC0, 0xFF). If any
    // pixel comes out neutral, the decoder resolved that palette — and either
    // painted it, or painted its red channel as a grey level.
    let px = extracted_pixels();
    let neutral = px
        .iter()
        .filter(|p| p[0] == p[1] && p[1] == p[2] && p[0] != 255)
        .count();
    assert_eq!(
        neutral, 0,
        "{neutral} of 8 pixels are grey: the JPEG 2000 data's colour specification \
         was used instead of the dictionary's; pixels {px:?}"
    );
}
