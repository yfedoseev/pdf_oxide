//! An image with a soft mask must composite the same way whatever size it is
//! drawn at.
//!
//! The blit premultiplies before resampling, which is the correct order — a
//! straight-alpha resample bleeds colour across an alpha edge from pixels that
//! contribute none. The SIMD downscale then has to be told the buffer is
//! already premultiplied, because its default is "premultiply, resize,
//! un-premultiply": run over premultiplied input it divides the alpha back out
//! and hands the compositor straight-alpha RGB in a buffer read as
//! premultiplied. Since the compositor computes `dst = src + dst*(1-a)`, the
//! colour is then added at full strength on top of a backdrop it should only
//! have tinted, and the artwork washes out toward white.
//!
//! A uniform alpha round-trips through that mistake unharmed — the divide
//! undoes the extra multiply exactly — so the mask here alternates opaque and
//! transparent rows, which is what a real soft mask looks like and what makes
//! the two resamples disagree.

#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, RenderOptions};

const SRC: u32 = 256;
/// 4x down, comfortably inside the `sx < 0.9` fast-path gate.
const PAGE: u32 = 64;
const GREY: u8 = 128;

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

/// A `page`-square page covered by one mid-grey `SRC`-square image whose
/// soft mask makes every other row transparent.
fn striped_soft_mask_page(page: u32) -> Vec<u8> {
    let base = vec![GREY; (SRC * SRC) as usize];
    let mut mask = Vec::with_capacity((SRC * SRC) as usize);
    for y in 0..SRC {
        let a = if y % 2 == 0 { 255u8 } else { 0u8 };
        mask.extend(std::iter::repeat_n(a, SRC as usize));
    }
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page} {page}] /Contents 4 0 R \
               /Resources << /XObject << /Im 5 0 R >> >> >>"
        )
        .into_bytes(),
        stream_obj("", format!("q {page} 0 0 {page} 0 0 cm /Im Do Q").as_bytes()),
        stream_obj(
            &format!(
                "/Type /XObject /Subtype /Image /Width {SRC} /Height {SRC} \
                 /ColorSpace /DeviceGray /BitsPerComponent 8 /SMask 6 0 R"
            ),
            &base,
        ),
        stream_obj(
            &format!(
                "/Type /XObject /Subtype /Image /Width {SRC} /Height {SRC} \
                 /ColorSpace /DeviceGray /BitsPerComponent 8"
            ),
            &mask,
        ),
    ];
    build_pdf(&objects)
}

/// Mean luminance of the rendered page.
fn mean_tone(pdf: &[u8]) -> f64 {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("synthetic PDF parses");
    let img = render_page(&doc, 0, &RenderOptions::default()).expect("page renders");
    let px = image::load_from_memory(&img.data)
        .expect("PNG decodes")
        .to_rgba8();
    let mut sum = 0u64;
    for p in px.pixels() {
        sum += u64::from(p[0]) + u64::from(p[1]) + u64::from(p[2]);
    }
    sum as f64 / (px.pixels().len() as f64 * 3.0)
}

/// Half the rows carry mid-grey at full opacity and half carry nothing, so a
/// correct downscale averages to `(128 + 255) / 2`. Getting 255 instead means
/// the alpha was divided back out and the grey was composited at full
/// strength over the white page.
#[test]
fn test_soft_masked_image_keeps_its_tone_when_downscaled() {
    let tone = mean_tone(&striped_soft_mask_page(PAGE));
    let expected = (f64::from(GREY) + 255.0) / 2.0;
    assert!(
        (tone - expected).abs() < 8.0,
        "a half-transparent mid-grey image downscaled 4x should average about \
         {expected:.1}; got {tone:.1}. Near 255 means the SIMD resize \
         un-premultiplied the buffer the blit had already premultiplied."
    );
}

/// The same artwork at 1:1 skips the SIMD resize entirely, so it pins the
/// expected value independently of the path under test: the two must agree.
#[test]
fn downscaling_a_soft_masked_image_matches_drawing_it_full_size() {
    let small = mean_tone(&striped_soft_mask_page(PAGE));
    let full = mean_tone(&striped_soft_mask_page(SRC));
    assert!(
        (small - full).abs() < 8.0,
        "soft-masked artwork must composite the same at any scale; \
         downscaled {small:.1} vs full size {full:.1}"
    );
}
