//! A `/Mask` carrying the `JBIG2Decode` filter must be decoded before its
//! samples are read.
//!
//! The stream decoder passes JBIG2 through untouched — the pixel decode lives
//! on the image path — so the compressed bitstream reached the stencil loop
//! unchanged. Every sample then fell past the end of the buffer and took the
//! "no sample to test, leave the base image visible" fallback, which silently
//! disabled the mask completely. On a scanned book, where the JBIG2 stencil
//! carries the text and the base image is the grey scan behind it, that
//! rendered the raw scan with nothing knocked out: three pages of one such
//! book came out at mean tone 129-134 where MuPDF and poppler both report
//! 246-251.
//!
//! ISO 32000-1:2008 §8.9.6.2 — under the default `/Decode`, a sample of 0
//! marks the page and a 1 leaves the previous contents unchanged; for an
//! explicit `/Mask` "marks the page" means the base image shows through.
//! Table 12's own example writes a JBIG2 image as `/DeviceGray
//! /BitsPerComponent 1`, in which 0 is black, so a black JBIG2 pixel is
//! sample 0 and is the opaque one.

#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, RenderOptions};

/// Minimal MSB-first bit writer, so the MMR codes below read as codes.
struct Bits {
    out: Vec<u8>,
    n: u32,
}

impl Bits {
    fn new() -> Self {
        Bits {
            out: Vec::new(),
            n: 0,
        }
    }

    fn push(&mut self, code: &str) {
        for c in code.chars() {
            if self.n.is_multiple_of(8) {
                self.out.push(0);
            }
            if c == '1' {
                let last = self.out.len() - 1;
                self.out[last] |= 0x80 >> (self.n % 8);
            }
            self.n += 1;
        }
    }
}

// ITU-T T.4 codes. Horizontal mode is reference-line independent: it names two
// run lengths starting at a0, which is why every row here can be written the
// same way whatever the row above it was.
const H: &str = "001";
const WHITE_0: &str = "00110101";
const WHITE_8: &str = "10011";
const BLACK_0: &str = "0000110111";
const BLACK_8: &str = "000101";

/// T.6 (MMR) data for an 8x8 bilevel image: `black_rows` all-black rows at the
/// top, the rest all white.
fn mmr_8x8(black_rows: usize) -> Vec<u8> {
    let mut b = Bits::new();
    for row in 0..8 {
        b.push(H);
        if row < black_rows {
            b.push(WHITE_0);
            b.push(BLACK_8);
        } else {
            b.push(WHITE_8);
            b.push(BLACK_0);
        }
    }
    b.out
}

fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// An embedded-organisation JBIG2 bitstream: a page-information segment and
/// one immediate generic region coded with MMR.
fn jbig2_8x8(black_rows: usize) -> Vec<u8> {
    let mut page_data = Vec::new();
    page_data.extend_from_slice(&be32(8)); // page width
    page_data.extend_from_slice(&be32(8)); // page height
    page_data.extend_from_slice(&be32(0)); // x resolution
    page_data.extend_from_slice(&be32(0)); // y resolution
    page_data.push(0); // page segment flags
    page_data.extend_from_slice(&[0, 0]); // striping information

    let mmr = mmr_8x8(black_rows);
    let mut region = Vec::new();
    region.extend_from_slice(&be32(8)); // region width
    region.extend_from_slice(&be32(8)); // region height
    region.extend_from_slice(&be32(0)); // region x
    region.extend_from_slice(&be32(0)); // region y
    region.push(0); // external combination operator OR
    region.push(0x01); // generic region flags: MMR = 1
    region.extend_from_slice(&mmr);

    let mut out = Vec::new();
    // segment header: number, flags (type in the low 6 bits), referred-to
    // count/retain byte, one-byte page association, data length.
    for (number, seg_type, data) in [(0u32, 48u8, page_data), (1u32, 38u8, region)] {
        out.extend_from_slice(&be32(number));
        out.push(seg_type);
        out.push(0x00);
        out.push(0x01);
        out.extend_from_slice(&be32(data.len() as u32));
        out.extend_from_slice(&data);
    }
    out
}

fn build_pdf(objects: &[Vec<u8>]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.4\n");
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref_pos = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            objects.len() + 1,
            xref_pos
        )
        .as_bytes(),
    );
    out
}

fn stream_obj(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut v = format!("<< {} /Length {} >>\nstream\n", dict, data.len()).into_bytes();
    v.extend_from_slice(data);
    v.extend_from_slice(b"\nendstream");
    v
}

/// A 100x100 page filled by one mid-grey 8x8 image, optionally masked by a
/// JBIG2 stencil whose top `black_rows` rows are black.
fn masked_page(black_rows: Option<usize>) -> Vec<u8> {
    let mask_entry = match black_rows {
        Some(_) => "/Mask 6 0 R",
        None => "",
    };
    let mut objects = vec![
        obj("<< /Type /Catalog /Pages 2 0 R >>"),
        obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R \
               /Resources << /XObject << /Im 5 0 R >> >> >>"),
        stream_obj("", b"q 100 0 0 100 0 0 cm /Im Do Q"),
        stream_obj(
            &format!(
                "/Type /XObject /Subtype /Image /Width 8 /Height 8 \
                 /ColorSpace /DeviceGray /BitsPerComponent 8 {mask_entry}"
            ),
            &[0x80u8; 64],
        ),
    ];
    if let Some(rows) = black_rows {
        objects.push(stream_obj(
            "/Type /XObject /Subtype /Image /Width 8 /Height 8 /ImageMask true \
             /Filter /JBIG2Decode",
            &jbig2_8x8(rows),
        ));
    }
    build_pdf(&objects)
}

fn obj(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

/// Ink coverage of the whole page, and of its top and bottom 40% bands.
///
/// The bands stop short of the halfway line on purpose: the mask sets alpha on
/// the 8x8 source image and the blit interpolates that alpha as it scales the
/// image up, so a band about half a source row deep straddles the stencil's
/// black/white boundary. That is resampling, not polarity, and measuring
/// through it would pin an interpolation detail rather than the mask.
fn coverage(pdf: &[u8]) -> (f64, f64, f64) {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("synthetic PDF parses");
    let img = render_page(&doc, 0, &RenderOptions::default()).expect("page renders");
    let px = image::load_from_memory(&img.data)
        .expect("PNG decodes")
        .to_rgba8();
    let (w, h) = px.dimensions();
    let (mut all, mut top, mut bot) = (0u64, 0u64, 0u64);
    let band = (h * 2) / 5;
    for (_, y, p) in px.enumerate_pixels() {
        let lum = (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
        if p[3] > 0 && lum < 250 {
            all += 1;
            if y < band {
                top += 1;
            } else if y >= h - band {
                bot += 1;
            }
        }
    }
    let band_px = w as f64 * band as f64;
    (all as f64 / (w as f64 * h as f64), top as f64 / band_px, bot as f64 / band_px)
}

#[test]
fn a_jbig2_stencil_masks_the_base_image() {
    // Control first: with no /Mask the grey image covers the page, so the
    // assertions below are about the mask and not about the image failing to
    // draw at all.
    let (unmasked, _, _) = coverage(&masked_page(None));
    assert!(unmasked > 0.98, "the unmasked base image must fill the page; got {unmasked:.4}");

    // Top four stencil rows black (sample 0 -> marks the page -> the base
    // image shows through), bottom four white (sample 1 -> unchanged).
    let (all, top, bot) = coverage(&masked_page(Some(4)));
    assert!(
        (all - 0.5).abs() < 0.1,
        "a half-black JBIG2 stencil must mask out about half the image; got {all:.4} \
         (top band {top:.4}, bottom band {bot:.4})"
    );
    assert!(top > 0.95, "the black half of the stencil must stay opaque; got {top:.4}");
    assert!(bot < 0.05, "the white half of the stencil must be masked out; got {bot:.4}");
}

#[test]
fn an_all_white_jbig2_stencil_hides_the_base_image() {
    let (all, _, _) = coverage(&masked_page(Some(0)));
    assert!(all < 0.05, "an all-white stencil leaves the page unchanged; got {all:.4}");
}
