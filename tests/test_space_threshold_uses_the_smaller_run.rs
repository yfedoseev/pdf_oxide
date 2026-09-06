//! Two words on one line are separated even when their font sizes differ.
//!
//! The space threshold is a fraction of a font size, and which of the two runs
//! supplies it decides whether a real gap counts. Taking the **larger** raises
//! the bar above the gap whenever the sizes differ, and the two failure modes
//! are not symmetric: too large a threshold fuses two words into one the page
//! never draws, while too small a one merely separates what was already
//! separate.
//!
//! On an OCR'd scan every word carries an independently estimated size, so the
//! mismatch is the normal case rather than the exception:
//!
//! ```text
//! And (13.12 pt) -> welcomes (8.97)   gap 1.725,  0.15 x max = 1.968  -> fused
//! little (6.28)  -> fishes (8.08)     gap 1.008,  0.15 x max = 1.212  -> fused
//! ```
//!
//! The geometry below is those two seams, to the hundredth.

use pdf_oxide::PdfDocument;

/// One line, four words, at the measured positions and sizes.
fn mixed_size_line() -> Vec<u8> {
    let content: &[u8] = b"BT\n\
        /F1 13.12 Tf 1 0 0 1 77.733 69.384 Tm (And) Tj\n\
        /F1 8.97 Tf 1 0 0 1 103.068 69.384 Tm (welcomes) Tj\n\
        /F1 6.28 Tf 1 0 0 1 150.284 69.240 Tm (little) Tj\n\
        /F1 8.08 Tf 1 0 0 1 173.892 68.232 Tm (fishes) Tj\n\
        ET\n";

    let objects: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 120] /Contents 4 0 R \
           /Resources << /Font << /F1 5 0 R >> >> >>"
            .to_vec(),
        [
            format!("<< /Length {} >>\nstream\n", content.len()).into_bytes(),
            content.to_vec(),
            b"\nendstream".to_vec(),
        ]
        .concat(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
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

#[test]
fn words_of_differing_size_are_not_fused() {
    let doc = PdfDocument::from_bytes(mixed_size_line()).expect("fixture parses");
    let text = doc.extract_text(0).expect("extract");
    for invented in ["Andwelcomes", "littlefishes"] {
        assert!(
            !text.contains(invented),
            "two words fused into {invented:?}, a word the page never draws — the \
             space threshold must scale by the SMALLER of the two runs, or a \
             mismatched pair raises the bar above its own gap:\n{text}"
        );
    }
}

/// And the words themselves all survive — a threshold so small that it shatters
/// runs would pass the assertion above for entirely the wrong reason.
#[test]
fn every_word_on_the_line_survives_whole() {
    let doc = PdfDocument::from_bytes(mixed_size_line()).expect("fixture parses");
    let text = doc.extract_text(0).expect("extract");
    for word in ["And", "welcomes", "little", "fishes"] {
        assert!(text.contains(word), "{word:?} is missing entirely:\n{text}");
    }
}
