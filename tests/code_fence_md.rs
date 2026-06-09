//! Integration test: consecutive monospace (fixed-pitch / code-font) lines must
//! render as a fenced Markdown code block, even when the producer did not tag
//! them with a `/Code` structure element.
//!
//! Monospace is detected from the font (FixedPitch flag or a Courier-family
//! base font). The PDF is hand-built (no third-party fixture): prose in
//! Helvetica, then two lines in Courier.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

fn code_block_pdf() -> Vec<u8> {
    // Prose line (Helvetica F1), then two Courier (F2) command lines below it.
    // Lengths/word positions vary per line so the spatial table detector does
    // not mistake them for an aligned grid.
    let content = b"BT\n\
        /F1 12 Tf 1 0 0 1 72 700 Tm (This is an introductory prose sentence here) Tj\n\
        /F2 12 Tf 1 0 0 1 72 660 Tm ($ run --flag --verbose input.txt) Tj\n\
        /F2 12 Tf 1 0 0 1 72 640 Tm (result: completed successfully now) Tj\n\
        ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 7];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes(),
        );
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>",
    );
    stream(&mut buf, &mut off, 4, content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    // Courier base font → detected as monospace by the name heuristic.
    obj(
        &mut buf,
        &mut off,
        6,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn monospace_lines_render_as_fenced_code_block() {
    let doc = PdfDocument::from_bytes(code_block_pdf()).unwrap();
    let md = doc.to_markdown(0, &ConversionOptions::default()).unwrap();

    // The two Courier lines fuse into one fenced block, prose stays outside it.
    assert!(
        md.contains(
            "```\n$ run --flag --verbose input.txt\nresult: completed successfully now\n```"
        ),
        "monospace lines not fenced as a code block: {md:?}"
    );
    assert!(md.contains("introductory prose sentence"), "prose line lost: {md:?}");
    // The prose line must not be inside the fence.
    assert!(
        !md.contains("sentence here\n```\n$") && !md.contains("```\nThis is"),
        "prose wrongly pulled into the code fence: {md:?}"
    );
}
