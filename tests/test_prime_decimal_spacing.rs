//! Prime / double-prime followed by a decimal must not gain a spurious space.
//!
//! Arc-second and prime-notation numbers in astronomy/physics PDFs are written
//! tight: `0′′.28`, `1′′.47`. Because a prime glyph's metric advance (w0,
//! ISO 32000-1 §9.4.4) is narrow relative to its inked form, the geometric
//! space heuristic sees a wide gap before the following `.NN` and injects a
//! space, yielding `0′′. 28`. The decimal fraction belongs to the same
//! measurement token, so the space is an extraction artifact.
//!
//! This fixture emits `0′′` then `.28` with an explicit positional gap that is
//! wide enough to trigger space insertion, and asserts the assembled text keeps
//! the number intact.

use pdf_oxide::PdfDocument;

/// Build a 1-page PDF that shows the single number `0′′.28` as one continuous
/// `TJ` run with a positional adjustment between the second prime and the
/// decimal — the shape a typesetter emits for an arc-second value. Byte 0xE2 is
/// bound via `/Differences` to the Adobe glyph `minute` (U+2032 PRIME); `0`,
/// `.`, `2`, `8` resolve through the WinAnsi base encoding. The adjustment is
/// large enough that the geometric heuristic reads the gap as a word break and,
/// without the fix, injects a space inside the number (`0′′ .28`).
fn prime_decimal_pdf() -> Vec<u8> {
    // `\342` == 0xE2 → /minute (′). One TJ run: two primes, a forward kern, `.28`.
    let content = "BT /F0 12 Tf 100 700 Td [(0\\342\\342) -2000 (.28)] TJ ET\n";

    let mut out: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = vec![0];
    out.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");

    let push = |out: &mut Vec<u8>, offsets: &mut Vec<usize>, body: &str| {
        offsets.push(out.len());
        let id = offsets.len() - 1;
        out.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    push(&mut out, &mut offsets, "<< /Type /Catalog /Pages 2 0 R >>");
    push(&mut out, &mut offsets, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    push(
        &mut out,
        &mut offsets,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 900] \
         /Resources << /Font << /F0 5 0 R >> >> /Contents 4 0 R >>",
    );
    push(
        &mut out,
        &mut offsets,
        &format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len() + 1),
    );
    push(
        &mut out,
        &mut offsets,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
         /Encoding << /Type /Encoding /BaseEncoding /WinAnsiEncoding \
           /Differences [226 /minute] >> >>",
    );

    let xref_offset = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for &off in &offsets[1..] {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n",
            offsets.len()
        )
        .as_bytes(),
    );
    out
}

#[test]
fn prime_followed_by_decimal_keeps_number_intact() {
    let pdf = prime_decimal_pdf();
    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let text = doc.extract_text(0).expect("extract");

    // Sanity: the primes decoded through /Differences → AGL.
    assert!(text.contains('\u{2032}'), "expected PRIME (U+2032) in output, got: {text:?}");

    // The decimal must stay attached to the prime — no `′ .` / `′. ` break.
    assert!(
        text.contains("\u{2032}\u{2032}.28"),
        "prime-decimal number split by a spurious space: {text:?}"
    );
    assert!(
        !text.contains("\u{2032}. ") && !text.contains("\u{2032} ."),
        "spurious space at prime/decimal boundary: {text:?}"
    );
}
