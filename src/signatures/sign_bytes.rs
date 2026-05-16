//! Two-pass PDF signing: appends an incremental update with a CMS/PKCS#7
//! signature over the ByteRange-covered portions of the document.
//!
//! ## Protocol (ISO 32000-1:2008 §12.8.1)
//!
//! 1. Append a new `obj` containing the signature dictionary with:
//!    - A fixed-width `/ByteRange [AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC DDDDDDDDDD]`
//!      placeholder (each number occupies exactly 10 characters so the total
//!      field width is constant across both passes).
//!    - A zero-padded `/Contents <000...000>` placeholder whose size is
//!      `estimated_size * 2 + 2` (hex encoding + angle brackets).
//!    - The standard `/Filter /SubFilter /M` entries.
//! 2. Append a minimal xref section and trailer pointing back to the
//!    existing catalog (`/Root`) and previous xref (`/Prev`).
//! 3. Locate the `/Contents` placeholder offset; calculate the actual
//!    ByteRange from the total output length and that offset.
//! 4. Patch the ByteRange placeholder in-place (same total width, trailing
//!    spaces absorb the difference in decimal-digit count).
//! 5. Extract the two signed byte ranges; call `PdfSigner::sign`; insert
//!    the hex-encoded signature into the `/Contents` placeholder.

#![cfg(feature = "signatures")]

use super::signer::PdfSigner;
use super::types::{SignOptions, SigningCredentials};
use crate::error::{Error, Result};
use crate::object::encode_pdf_text_string;

// ─── Width constants ────────────────────────────────────────────────────────
//
// Each of the four ByteRange numbers occupies exactly BR_FIELD_W characters
// (right-justified, space-padded on the left). Keeping the width fixed means
// the total text produced by pass 1 is identical in length to pass 2, so
// /Contents offsets don't shift between passes.

const BR_FIELD_W: usize = 10;
const BR_PLACEHOLDER: &str = "0000000000 0000000000 0000000000 0000000000";

// ─── Public entry point ──────────────────────────────────────────────────────

/// Append a digital signature to `pdf_data` as an incremental update and
/// return the signed PDF bytes.
///
/// `credentials` must carry a valid X.509 certificate **and** private key
/// (e.g. loaded via [`SigningCredentials::from_pem`] or
/// [`SigningCredentials::from_pkcs12`]).
///
/// # Errors
///
/// Returns an error if:
/// - the existing PDF bytes do not contain a parseable `startxref` or
///   `/Root` entry (corrupted / not a PDF),
/// - the private key cannot be used to sign (wrong format or corrupted), or
/// - the estimated signature size is too small for the produced CMS blob.
pub fn sign_pdf_bytes(
    pdf_data: &[u8],
    credentials: &SigningCredentials,
    opts: SignOptions,
) -> Result<Vec<u8>> {
    let signer = PdfSigner::new(credentials.clone(), opts);

    // ── 1. Extract the minimum structural info from the existing PDF ──────
    let prev_startxref = scan_startxref(pdf_data)
        .ok_or_else(|| Error::InvalidPdf("cannot find startxref in existing PDF".into()))?;
    let root_ref = scan_root_ref(pdf_data)
        .ok_or_else(|| Error::InvalidPdf("cannot find /Root ref in existing PDF".into()))?;
    let next_obj_num = scan_next_obj_num(pdf_data).ok_or_else(|| {
        Error::InvalidPdf("cannot determine next object number from PDF trailer".into())
    })?;

    // ── 2. Build the signature dictionary text (fixed-width placeholders) ─
    let sig_dict_text = build_sig_dict_text(&signer, next_obj_num);

    // Locate /Contents '<' offset within the sig dict text (will become the
    // offset we patch after computing ByteRange).
    let contents_in_dict = find_contents_offset_in_text(sig_dict_text.as_bytes())
        .ok_or_else(|| Error::InvalidPdf("cannot find /Contents in built sig dict".into()))?;

    // ── 3. Pre-compute all section offsets ────────────────────────────────
    let sig_dict_start = pdf_data.len(); // offset of "N 0 obj\n"
    let xref_start = sig_dict_start + sig_dict_text.len();

    let xref_entry = format!("{:010} 00000 n \r\n", sig_dict_start);
    let xref_section = format!("xref\n{} 1\n{}", next_obj_num, xref_entry);

    let trailer_section = format!(
        "trailer\n<< /Size {} /Prev {} /Root {} >>\n",
        next_obj_num + 1,
        prev_startxref,
        root_ref,
    );

    let startxref_section = format!("startxref\n{}\n%%EOF\n", xref_start);

    let total_len = sig_dict_start
        + sig_dict_text.len()
        + xref_section.len()
        + trailer_section.len()
        + startxref_section.len();

    // ── 4. Compute actual ByteRange ───────────────────────────────────────
    let contents_abs = sig_dict_start + contents_in_dict; // offset of '<'
    let contents_size = signer.placeholder_size(); // '<' + hex + '>'
    let after_contents = contents_abs + contents_size;
    let byte_range: [i64; 4] = [
        0,
        contents_abs as i64,
        after_contents as i64,
        (total_len - after_contents) as i64,
    ];

    // ── 5. Patch ByteRange placeholder in sig dict ────────────────────────
    let patched_sig_dict = patch_byterange(sig_dict_text, &byte_range);

    // ── 6. Assemble the full output ───────────────────────────────────────
    let mut output = Vec::with_capacity(total_len);
    output.extend_from_slice(pdf_data);
    output.extend_from_slice(patched_sig_dict.as_bytes());
    output.extend_from_slice(xref_section.as_bytes());
    output.extend_from_slice(trailer_section.as_bytes());
    output.extend_from_slice(startxref_section.as_bytes());

    debug_assert_eq!(
        output.len(),
        total_len,
        "assembled output length must match pre-computed total_len"
    );

    // ── 7. Extract signed bytes and sign ─────────────────────────────────
    let signed_bytes =
        super::byterange::ByteRangeCalculator::extract_signed_bytes(&output, &byte_range)?;
    let cms_blob = signer.sign(&signed_bytes)?;

    // ── 8. Insert signature ───────────────────────────────────────────────
    signer.insert_signature(&mut output, contents_abs, &cms_blob)?;

    Ok(output)
}

// ─── Text builders ───────────────────────────────────────────────────────────

fn build_sig_dict_text(signer: &PdfSigner, obj_num: u64) -> String {
    let opts = signer.options();
    let contents_placeholder = signer.generate_contents_placeholder();

    let mut dict = format!(
        "{} 0 obj\n<< /Type /Sig\n/Filter /Adobe.PPKLite\n/SubFilter /{}\n",
        obj_num,
        opts.sub_filter.as_pdf_name(),
    );

    // Fixed-width ByteRange placeholder — patched in step 5.
    // The string inside [...] must exactly match BR_PLACEHOLDER so that
    // patch_byterange can find and replace it.
    dict.push_str(&format!("/ByteRange [{}]\n", BR_PLACEHOLDER));

    // /Contents: the '<' here is exactly what find_contents_offset_in_text looks for
    dict.push_str(&format!("/Contents {}\n", contents_placeholder));

    if let Some(ref r) = opts.reason {
        dict.push_str(&format!("/Reason {}\n", pdf_text_hex(r)));
    }
    if let Some(ref l) = opts.location {
        dict.push_str(&format!("/Location {}\n", pdf_text_hex(l)));
    }
    if let Some(ref n) = opts.name {
        dict.push_str(&format!("/Name {}\n", pdf_text_hex(n)));
    }
    if let Some(ref c) = opts.contact_info {
        dict.push_str(&format!("/ContactInfo {}\n", pdf_text_hex(c)));
    }

    dict.push_str(&format!("/M ({})\n", format_pdf_date()));
    dict.push_str(">>\nendobj\n");
    dict
}

/// Patch the `0000000000 0000000000 0000000000 0000000000` placeholder in the
/// signature dict text with the actual ByteRange values. The output is always
/// the same length as the input because each field is right-justified in a
/// `BR_FIELD_W`-wide space (trailing spaces absorb the freed digits).
fn patch_byterange(mut text: String, br: &[i64; 4]) -> String {
    // Re-format each number right-justified in BR_FIELD_W characters.
    // The placeholder "0000000000" is replaced by e.g. "         0" or "1234567890".
    let replacement = format!(
        "{:>BR_FIELD_W$} {:>BR_FIELD_W$} {:>BR_FIELD_W$} {:>BR_FIELD_W$}",
        br[0], br[1], br[2], br[3],
    );
    assert_eq!(
        replacement.len(),
        BR_PLACEHOLDER.len(),
        "replacement must have the same length as the placeholder"
    );
    if let Some(pos) = text.find(BR_PLACEHOLDER) {
        text.replace_range(pos..pos + BR_PLACEHOLDER.len(), &replacement);
    }
    text
}

/// Find the byte offset of the `<` that opens `/Contents <...>` within `data`.
/// Matches the first `<` that follows a `/Contents` keyword (skipping optional
/// whitespace).
fn find_contents_offset_in_text(data: &[u8]) -> Option<usize> {
    let pattern = b"/Contents ";
    let pos = data.windows(pattern.len()).position(|w| w == pattern)?;
    let after = pos + pattern.len();
    // Skip additional whitespace before '<'
    for (i, &b) in data[after..].iter().enumerate() {
        if b == b'<' {
            return Some(after + i);
        }
        if b != b' ' && b != b'\t' && b != b'\r' && b != b'\n' {
            break;
        }
    }
    None
}

// ─── PDF metadata scanners ───────────────────────────────────────────────────

/// Find the last `startxref` offset value in the file (scans the last 4 KB).
fn scan_startxref(data: &[u8]) -> Option<u64> {
    let window = &data[data.len().saturating_sub(4096)..];
    // rfind so we pick up the LAST startxref (most-recent incremental update)
    let pos = window.windows(9).rposition(|w| w == b"startxref")?;
    let after = &window[pos + 9..];
    let s = std::str::from_utf8(after).ok()?;
    let trimmed = s.trim_start_matches([' ', '\r', '\n']);
    let end = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    trimmed[..end].parse().ok()
}

/// Find the last `/Root X Y R` reference string in the file (scans the last
/// 4 KB only, same as `scan_startxref`, to avoid false-positive matches in
/// uncompressed content streams or metadata that contain the literal `/Root`).
fn scan_root_ref(data: &[u8]) -> Option<String> {
    let window = &data[data.len().saturating_sub(4096)..];
    let pattern = b"/Root ";
    let pos = window.windows(pattern.len()).rposition(|w| w == pattern)?;
    let after = &window[pos + pattern.len()..];
    // Collect up to 40 bytes as ASCII; stop at '/' or '>>'
    let end = after
        .iter()
        .position(|&b| b == b'/' || b == b'>' || b == b'\n')
        .unwrap_or(after.len().min(40));
    let s = std::str::from_utf8(&after[..end]).ok()?.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Find the highest object number in the latest trailer's `/Size` entry.
/// Returns the number itself — the next available object ID is this value.
fn scan_next_obj_num(data: &[u8]) -> Option<u64> {
    let window = &data[data.len().saturating_sub(4096)..];
    // Find LAST /Size entry in the tail (covers incremental updates)
    let pattern = b"/Size ";
    let pos = window.windows(pattern.len()).rposition(|w| w == pattern)?;
    let after = &window[pos + pattern.len()..];
    let s = std::str::from_utf8(after).ok()?;
    let trimmed = s.trim_start_matches(' ');
    let end = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    trimmed[..end].parse().ok()
}

// ─── PDF string helper ───────────────────────────────────────────────────────

/// Encode `s` as a PDF hex string `<AABB...>` using the same
/// PDFDocEncoding / UTF-16BE-with-BOM logic as the rest of the library
/// (`encode_pdf_text_string`).  Hex syntax requires no further escaping and
/// handles arbitrary byte sequences safely.
fn pdf_text_hex(s: &str) -> String {
    let bytes = encode_pdf_text_string(s);
    let mut out = String::with_capacity(bytes.len() * 2 + 2);
    out.push('<');
    for b in &bytes {
        out.push_str(&format!("{:02X}", b));
    }
    out.push('>');
    out
}

fn format_pdf_date() -> String {
    // Delegates to the single leap-year-correct implementation. The
    // prior local copy hard-coded month/day to "0101" and approximated
    // the year as 1970 + days/365 — corrupting every signature /M date
    // (README latent bug). WASM note: SystemTime::now() in the shared
    // helper still needs cfg-gating if signatures are ever enabled for
    // wasm32 (currently masked — `signatures` is off in the wasm build).
    super::pdf_date::format_pdf_date_utc()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// Parse the total byte length of a DER SEQUENCE from its hex encoding.
/// Handles the definite long form (0x82 two-byte length) used by CMS blobs.
#[cfg(test)]
fn der_sequence_len_from_hex(hex: &str) -> usize {
    let lb = u8::from_str_radix(&hex[2..4], 16).expect("DER len byte");
    if lb < 0x80 {
        (lb as usize) + 2
    } else {
        let n = (lb & 0x7f) as usize;
        let mut len = 0usize;
        for i in 0..n {
            let b = u8::from_str_radix(&hex[(4 + i * 2)..(6 + i * 2)], 16).expect("DER len");
            len = (len << 8) | (b as usize);
        }
        len + 2 + n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signatures::cms_verify::SignerVerify;
    use crate::signatures::verify_signer_detached;
    use crate::signatures::ByteRangeCalculator;

    fn load_test_creds() -> SigningCredentials {
        let cert =
            std::fs::read_to_string("tests/fixtures/test_signing_cert.pem").expect("cert fixture");
        let key =
            std::fs::read_to_string("tests/fixtures/test_signing_key.pem").expect("key fixture");
        SigningCredentials::from_pem(&cert, &key).expect("creds load")
    }

    fn minimal_pdf() -> Vec<u8> {
        // A valid single-page PDF with AcroForm stripped to the bare minimum
        // so the scanner tests have something real to work with.
        b"%PDF-1.4\n\
          1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\
          2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\
          3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n\
          xref\n0 4\n0000000000 65535 f \r\n0000000009 00000 n \r\n\
          0000000058 00000 n \r\n0000000115 00000 n \r\n\
          trailer\n<< /Size 4 /Root 1 0 R >>\n\
          startxref\n187\n%%EOF\n"
            .to_vec()
    }

    #[test]
    fn test_scan_startxref() {
        let pdf = minimal_pdf();
        let xref = scan_startxref(&pdf).expect("must find startxref");
        assert!(xref > 0, "startxref must be positive");
    }

    #[test]
    fn test_scan_root_ref() {
        let pdf = minimal_pdf();
        let root = scan_root_ref(&pdf).expect("must find root");
        assert!(root.contains("1 0 R"), "root ref must be '1 0 R': got {root}");
    }

    #[test]
    fn test_scan_next_obj_num() {
        let pdf = minimal_pdf();
        let n = scan_next_obj_num(&pdf).expect("must find /Size");
        assert_eq!(n, 4, "/Size must be 4 for this minimal PDF");
    }

    #[test]
    fn test_patch_byterange_same_length() {
        let text = format!("pre {} post", BR_PLACEHOLDER);
        let original_len = text.len();
        let br = [0i64, 12345, 99999, 200];
        let patched = patch_byterange(text, &br);
        assert_eq!(patched.len(), original_len, "patch must not change text length");
        assert!(!patched.contains(BR_PLACEHOLDER), "placeholder must be replaced");
    }

    fn hex_decode(s: &str) -> Vec<u8> {
        s.as_bytes()
            .chunks(2)
            .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap())
            .collect()
    }

    #[test]
    fn test_sign_pdf_bytes_roundtrip() {
        let pdf = minimal_pdf();
        let creds = load_test_creds();
        let opts = SignOptions {
            estimated_size: 4096,
            ..Default::default()
        };

        let signed = sign_pdf_bytes(&pdf, &creds, opts).expect("sign_pdf_bytes must succeed");

        // ── Parse the appended incremental update ─────────────────────
        // The incremental update is appended after the original PDF bytes.
        let tail = &signed[pdf.len()..];
        let tail_str = std::str::from_utf8(tail).unwrap();

        // Parse /ByteRange [...] from the sig dict text
        let br_pos = tail_str
            .find("/ByteRange [")
            .expect("/ByteRange must exist");
        let after_br = &tail_str[br_pos + 12..];
        let end = after_br.find(']').expect("] must follow /ByteRange");
        let nums: Vec<i64> = after_br[..end]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(nums.len(), 4);
        let byte_range: [i64; 4] = [nums[0], nums[1], nums[2], nums[3]];

        // Validate ByteRange is sane
        assert_eq!(byte_range[0], 0);
        assert!(byte_range[1] > 0);
        assert!(byte_range[2] > byte_range[1]);
        assert!(byte_range[3] > 0);
        assert_eq!(
            byte_range[2] + byte_range[3],
            signed.len() as i64,
            "ByteRange must cover the whole file"
        );

        // ── Extract /Contents hex and decode the CMS blob ─────────────
        let ct_pos = tail_str.find("/Contents <").expect("/Contents must exist");
        let after_ct = &tail_str[ct_pos + 11..]; // skip "/Contents <"
        let close = after_ct.find('>').expect("> must follow /Contents <");
        let hex_str = &after_ct[..close];
        // Use the DER length field to find the exact CMS byte count rather than
        // trimming trailing '0' characters — a CMS whose last real byte is 0x00
        // would be silently truncated by the naive trim approach.
        let cms_len = der_sequence_len_from_hex(hex_str);
        let cms_blob = hex_decode(&hex_str[..cms_len * 2]);

        // ── Extract the signed bytes and verify ───────────────────────
        let signed_content = ByteRangeCalculator::extract_signed_bytes(&signed, &byte_range)
            .expect("extract_signed_bytes must succeed");

        let result =
            verify_signer_detached(&cms_blob, &signed_content).expect("verify must not error");
        assert_eq!(result, SignerVerify::Valid, "end-to-end PDF signature must verify as Valid");
    }

    // ── Finding 3 regression: scan_root_ref must ignore /Root in body ────────

    #[test]
    fn test_scan_root_ref_ignores_body_occurrence() {
        // Embed a fake "/Root " string deep in the body far from the trailer.
        // The scanner must still return the real trailer reference.
        let mut pdf = minimal_pdf();
        // Prepend >4 KB of content containing a misleading /Root occurrence.
        let filler = b"% /Root 99 0 R this is inside a comment not a trailer\n";
        let padding = filler.repeat(100); // ~5.4 KB
        let mut data = padding;
        data.extend_from_slice(&pdf);
        // The real /Root is in the last 4 KB (the trailer of minimal_pdf is tiny).
        let root = scan_root_ref(&data).expect("must find root in last 4 KB");
        assert!(
            root.contains("1 0 R"),
            "must return trailer /Root, not body occurrence; got: {root}"
        );
        // Confirm that there really IS a misleading /Root earlier in the data.
        let first = data.windows(b"/Root ".len()).position(|w| w == b"/Root ");
        assert!(first.unwrap() < data.len() - 4096, "misleading /Root is before the 4 KB window");

        // Drop pdf from outer scope warning
        let _ = pdf.drain(..);
    }

    // ── Finding 7 regression: non-ASCII metadata must not be raw UTF-8 ───────

    #[test]
    fn test_pdf_text_hex_ascii_roundtrip() {
        // ASCII stays as PDFDocEncoding bytes (no BOM).
        let h = pdf_text_hex("Hello");
        assert!(h.starts_with('<') && h.ends_with('>'));
        let bytes = hex_decode(&h[1..h.len() - 1]);
        assert_eq!(bytes, b"Hello");
    }

    #[test]
    fn test_pdf_text_hex_latin1_no_bom() {
        // "é" is U+00E9 — within PDFDocEncoding range → single byte 0xE9, no BOM.
        let h = pdf_text_hex("é");
        let bytes = hex_decode(&h[1..h.len() - 1]);
        assert_eq!(bytes, &[0xE9], "PDFDocEncoding for é must be 0xE9, not multi-byte UTF-8");
    }

    #[test]
    fn test_pdf_text_hex_portuguese_reason() {
        // Regression guard: "Aprovado Lógico" — contains ó (U+00F3).
        // Must NOT emit the raw UTF-8 bytes 0xC3 0xB3 for ó.
        let h = pdf_text_hex("Aprovado Lógico");
        let bytes = hex_decode(&h[1..h.len() - 1]);
        // PDFDocEncoding: ó → 0xF3 (single byte), not 0xC3 0xB3 (UTF-8).
        assert!(
            !bytes.windows(2).any(|w| w == [0xC3, 0xB3]),
            "raw UTF-8 bytes for ó must not appear; got {:X?}",
            bytes
        );
        // ó must appear as its PDFDocEncoding byte 0xF3.
        assert!(bytes.contains(&0xF3), "PDFDocEncoding 0xF3 for ó must be present");
    }

    #[test]
    fn test_pdf_text_hex_cjk_uses_utf16be_bom() {
        // CJK characters trigger UTF-16BE with leading BOM 0xFE 0xFF.
        let h = pdf_text_hex("中文");
        let bytes = hex_decode(&h[1..h.len() - 1]);
        assert_eq!(&bytes[..2], &[0xFE, 0xFF], "UTF-16BE BOM must be present for CJK");
    }

    #[test]
    fn test_sign_metadata_non_ascii_encoded_in_sig_dict() {
        // End-to-end: signature dict text must contain hex-encoded metadata,
        // never raw multi-byte UTF-8 sequences for non-ASCII characters.
        let pdf = minimal_pdf();
        let creds = load_test_creds();
        let opts = SignOptions {
            reason: Some("Aprovado Lógico".to_string()), // ó = U+00F3
            location: Some("São Paulo".to_string()),     // ã = U+00E3, ~ ã
            name: Some("中文签名人".to_string()),
            estimated_size: 8192,
            ..Default::default()
        };

        let signed = sign_pdf_bytes(&pdf, &creds, opts).expect("sign must succeed");
        let tail = &signed[pdf.len()..];

        // The sig dict is written as UTF-8 text around hex strings, so we can
        // search the raw bytes for /Reason <...> etc.
        let tail_str = std::str::from_utf8(tail).unwrap();

        // Must contain hex string syntax (angle brackets) for /Reason.
        let reason_hex = tail_str.find("/Reason <").is_some();
        let location_hex = tail_str.find("/Location <").is_some();
        let name_hex = tail_str.find("/Name <").is_some();
        assert!(reason_hex, "/Reason must use hex string syntax");
        assert!(location_hex, "/Location must use hex string syntax");
        assert!(name_hex, "/Name must use hex string syntax");

        // Raw UTF-8 encoding of ó is 0xC3 0xB3 — must NOT appear in the dict.
        let c3b3 = tail.windows(2).any(|w| w == [0xC3, 0xB3]);
        assert!(!c3b3, "raw UTF-8 bytes for ó must not appear in signed output");
    }
}
