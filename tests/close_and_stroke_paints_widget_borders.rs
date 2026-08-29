//! The `s` operator must close and stroke the path.
//!
//! ISO 32000-1:2008 Table 60: "**s** — Close and stroke the path. This
//! operator shall have the same effect as the sequence `h S`."
//!
//! `parse_content_stream_paths_only` decomposed it into `ClosePath` + `Stroke`,
//! but the main operator table — which the streaming parser, and therefore
//! annotation appearance streams, go through — had no entry for it at all, so
//! the operator was dropped and the path was never painted.
//!
//! AcroForm field borders are commonly drawn exactly this way
//! (`q 0 G 0.5 0.5 149 21 re s Q` inside a widget's normal appearance), which
//! is why they were missing across the corpus while a sibling button drawn
//! with `f` rendered fine.
//!
//! Hand-built synthetic PDF; no third-party fixture.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, RenderOptions};

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

fn page_with(content: &str) -> Vec<u8> {
    build_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R \
           /Resources << >> >>"
            .to_vec(),
        stream_obj("", content.as_bytes()),
    ])
}

/// Count pixels darker than mid-grey.
fn dark_pixels(pdf: &[u8]) -> usize {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("synthetic PDF parses");
    let img = render_page(&doc, 0, &RenderOptions::default()).expect("page renders");
    let px = image::load_from_memory(&img.data).expect("PNG decodes").to_rgba8();
    px.pixels()
        .filter(|p| p[3] > 0 && (u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])) / 3 < 128)
        .count()
}

/// The shape a widget border uses.
#[test]
fn close_and_stroke_paints_the_rectangle() {
    let n = dark_pixels(&page_with("q 2 w 0 G 40 40 120 120 re s Q"));
    assert!(
        n > 500,
        "`s` must close and stroke the path; a 2-unit border around a 120x120 \
         box should darken well over 500 pixels, got {n}"
    );
}

/// `S` on the same path is the control: `s` differs only by closing first, so
/// on an already-closed `re` the two must paint the same.
#[test]
fn close_and_stroke_matches_stroke_on_a_closed_path() {
    let with_s = dark_pixels(&page_with("q 2 w 0 G 40 40 120 120 re s Q"));
    let with_upper_s = dark_pixels(&page_with("q 2 w 0 G 40 40 120 120 re S Q"));
    assert_eq!(
        with_s, with_upper_s,
        "a rectangle is already closed, so `s` and `S` must paint identically"
    );
}

/// And `s` must actually close: an open three-sided path stroked with `s`
/// paints more than the same path stroked with `S`, which leaves it open.
#[test]
fn close_and_stroke_closes_an_open_subpath() {
    let open = "q 4 w 0 G 40 40 m 160 40 l 160 160 l S Q";
    let closed = "q 4 w 0 G 40 40 m 160 40 l 160 160 l s Q";
    let n_open = dark_pixels(&page_with(open));
    let n_closed = dark_pixels(&page_with(closed));
    assert!(
        n_closed > n_open,
        "`s` closes the subpath, adding the returning segment: expected more \
         ink than `S` ({n_open}), got {n_closed}"
    );
}
