//! A rotated page is rotated, not mirrored.
//!
//! `/Rotate 270` and `/Rotate -90` produced mirrored output because the
//! separation renderer carried its own page transform instead of the shared
//! one. A mirror is easy to miss on symmetric content and on text at a glance,
//! which is why the fixture is an **L-shape**: two arms of different colours
//! meeting at one corner. Under any rotation the pair moves together; under a
//! mirror they swap sides.
//!
//! Both renderers now go through `page_base_transform`, so the arms are pinned
//! for all four rotations at once.
//!
//! The expected positions are not invented — they are what MuPDF and pdfium
//! both produce on these exact fixtures, agreeing with each other to two
//! decimal places.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

/// A 300x200 page with a blue horizontal arm and a red vertical arm meeting at
/// the bottom-left, under the given `/Rotate`.
fn rotated_page(rotate: i32) -> Vec<u8> {
    let content: &[u8] = b"0 0 1 rg 20 20 120 25 re f\n1 0 0 rg 20 20 25 120 re f\n";
    let objects: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Rotate {rotate} \
             /Contents 4 0 R /Resources << >> >>"
        )
        .into_bytes(),
        [
            format!("<< /Length {} >>\nstream\n", content.len()).into_bytes(),
            content.to_vec(),
            b"\nendstream".to_vec(),
        ]
        .concat(),
    ];
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

/// Centroid of each arm, as a fraction of the rendered page.
fn arm_centroids(rotate: i32) -> ((f32, f32), (f32, f32)) {
    let doc = PdfDocument::from_bytes(rotated_page(rotate)).expect("fixture parses");
    let img = render_page(&doc, 0, &RenderOptions::with_dpi(72)).expect("page renders");
    let px = image::load_from_memory(&img.data).expect("PNG decodes").to_rgba8();
    let (w, h) = (px.width() as f32, px.height() as f32);
    let centroid = |pick: &dyn Fn(&image::Rgba<u8>) -> bool| -> (f32, f32) {
        let (mut sx, mut sy, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y, p) in px.enumerate_pixels() {
            if pick(p) {
                sx += x as f32;
                sy += y as f32;
                n += 1.0;
            }
        }
        if n == 0.0 {
            return (f32::NAN, f32::NAN);
        }
        (sx / n / w, sy / n / h)
    };
    (
        centroid(&|p| p[2] > 150 && p[0] < 100),
        centroid(&|p| p[0] > 150 && p[2] < 100),
    )
}

/// All four rotations, against the positions MuPDF and pdfium agree on.
#[test]
fn every_rotation_places_both_arms_where_the_engines_do() {
    // (rotate, blue x, blue y, red x, red y) — measured from MuPDF and pdfium,
    // which match each other to two decimals on these fixtures.
    let expected = [
        (0, 0.31, 0.83, 0.11, 0.60),
        (90, 0.16, 0.31, 0.40, 0.11),
        (180, 0.69, 0.16, 0.89, 0.40),
        (270, 0.83, 0.69, 0.60, 0.89),
    ];
    for (rotate, bx, by, rx, ry) in expected {
        let (blue, red) = arm_centroids(rotate);
        for (got, want, what) in [
            (blue.0, bx, "blue x"),
            (blue.1, by, "blue y"),
            (red.0, rx, "red x"),
            (red.1, ry, "red y"),
        ] {
            assert!(
                (got - want).abs() < 0.03,
                "/Rotate {rotate}: {what} is {got:.2}, MuPDF and pdfium both give \
                 {want:.2}. A swap of the two arms means the page was mirrored \
                 rather than rotated."
            );
        }
    }
}

/// `/Rotate -90` must equal `/Rotate 270` — the same angle written differently,
/// and the form the original report named.
#[test]
fn negative_ninety_matches_two_seventy() {
    let (blue_neg, red_neg) = arm_centroids(-90);
    let (blue_270, red_270) = arm_centroids(270);
    assert!(
        (blue_neg.0 - blue_270.0).abs() < 0.01 && (blue_neg.1 - blue_270.1).abs() < 0.01,
        "/Rotate -90 blue arm at ({:.2}, {:.2}) but /Rotate 270 at ({:.2}, {:.2})",
        blue_neg.0, blue_neg.1, blue_270.0, blue_270.1
    );
    assert!(
        (red_neg.0 - red_270.0).abs() < 0.01 && (red_neg.1 - red_270.1).abs() < 0.01,
        "/Rotate -90 red arm at ({:.2}, {:.2}) but /Rotate 270 at ({:.2}, {:.2})",
        red_neg.0, red_neg.1, red_270.0, red_270.1
    );
}
