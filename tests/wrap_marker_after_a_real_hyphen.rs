//! A soft hyphen that marks the wrap of an already-hyphenated compound must
//! not cost the compound its hyphen.
//!
//! A typesetter breaking `Cross-sectional` across a line writes the real
//! hyphen and then U+00AD, the discretionary-break marker: `Cross-<soft>` /
//! `sectional`. ISO 32000-1:2008 §14.8.2.2.3 makes the marker invisible
//! content, so it is stripped — but stripping it *before* the wrap decision
//! leaves a bare `Cross-`, which the rejoiner downstream then reads as the
//! wrap marker and removes. `Cross-sectional` became `Crosssectional`, and
//! `Receiver-operating` became `Receiveroperating`; three token types that
//! MuPDF, pdfminer.six, pypdf and poppler all report fell to zero.
//!
//! The marker now survives the strip when it directly follows a hyphen-minus
//! at the end of a fragment, which is the only shape where the two characters
//! mean different things.

use pdf_oxide::document::PdfDocument;

/// Two lines of a paragraph, the first ending in a hyphenated compound broken
/// at its own hyphen: `<word>-<U+00AD>` then the continuation on the next line.
fn wrapped_compound_pdf(first: &str, second: &str) -> Vec<u8> {
    // U+00AD is 0xAD in WinAnsiEncoding, written as an octal escape.
    let stream = format!(
        "BT /F1 10 Tf\n\
         1 0 0 1 72 700 Tm ({first}\\255) Tj\n\
         1 0 0 1 72 686 Tm ({second}) Tj\n\
         ET\n"
    );

    let mut pdf: Vec<u8> = Vec::new();
    macro_rules! push {
        ($s:expr) => {
            pdf.extend_from_slice($s.as_bytes())
        };
    }
    push!("%PDF-1.4\n");
    let off1 = pdf.len();
    push!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let off2 = pdf.len();
    push!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    push!(
        "3 0 obj\n\
         << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
            /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n\
         endobj\n"
    );
    let off4 = pdf.len();
    let bytes = stream.as_bytes();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", bytes.len()));
    pdf.extend_from_slice(bytes);
    push!("\nendstream\nendobj\n");
    let off5 = pdf.len();
    push!(
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
         /Encoding /WinAnsiEncoding >>\nendobj\n"
    );
    let xref = pdf.len();
    push!(format!(
        "xref\n0 6\n\
         0000000000 65535 f \r\n\
         {off1:010} 00000 n \r\n\
         {off2:010} 00000 n \r\n\
         {off3:010} 00000 n \r\n\
         {off4:010} 00000 n \r\n\
         {off5:010} 00000 n \r\n"
    ));
    push!(format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"));
    pdf
}

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|t| !t.is_empty())
        .map(|t| t.trim_matches('-').to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

#[test]
fn a_compounds_own_hyphen_survives_the_wrap_marker() {
    let doc = PdfDocument::from_bytes(wrapped_compound_pdf("Cross-", "sectional")).unwrap();
    let text = doc.extract_text(0).unwrap();
    let squashed: String = text.chars().filter(|c| !c.is_whitespace()).collect();

    assert!(
        !squashed.contains("Crosssectional"),
        "the compound's own hyphen was eaten along with the wrap marker: {text:?}"
    );
    let toks = tokens(&text);
    assert!(
        toks.iter().any(|t| t == "Cross") || toks.iter().any(|t| t == "Cross-sectional"),
        "`Cross` must survive as its own token or as the intact compound: {text:?}"
    );
    assert!(
        toks.iter().any(|t| t == "sectional") || toks.iter().any(|t| t == "Cross-sectional"),
        "`sectional` must survive: {text:?}"
    );
}

/// Control: an ordinary wrapped word, whose marker is a bare soft hyphen with
/// no real hyphen before it, must still rejoin into one word.
#[test]
fn an_ordinary_wrapped_word_still_rejoins() {
    let doc = PdfDocument::from_bytes(wrapped_compound_pdf("modali", "ties")).unwrap();
    let text = doc.extract_text(0).unwrap();
    let squashed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        squashed.contains("modalities"),
        "a plain wrapped word must rejoin without its marker: {text:?}"
    );
}
