//! Resilience of `PdfDocument::output_intent_cmyk_profile` to malformed
//! `/OutputIntents` entries (issue #712).
//!
//! Per ISO 32000-1:2008 §14.11.5 the catalog `/OutputIntents` array advertises
//! the target device's colour characteristics; `output_intent_cmyk_profile`
//! returns the first CMYK (`N=4`) `DestOutputProfile` it can parse, used as the
//! fallback ICC profile for plain `/DeviceCMYK` content. A single broken entry
//! must not hide a valid profile declared later in the array.
//!
//! These tests build real PDFs and open them through `PdfDocument::from_bytes`
//! (no mocking). They need neither the `rendering` nor `icc` feature — the
//! accessor and `IccProfile::parse` are compiled unconditionally — so unlike
//! `tests/test_render_output_intent.rs` (which renders pixels) they run under
//! the default `cargo test`.

use pdf_oxide::document::PdfDocument;

/// Minimal valid CMYK ICC profile body: a 128-byte header with the `acsp`
/// signature (offset 36) and a `CMYK` data-colour-space (offset 16). Accepted
/// by `IccProfile::parse(_, 4)` without a CMM. Mirrors the helper in
/// `tests/test_icc_cmyk_conversion.rs`.
fn cmyk_icc_profile() -> Vec<u8> {
    let mut v = vec![0u8; 128];
    v[8..12].copy_from_slice(&0x0400_0000u32.to_be_bytes()); // version
    v[12..16].copy_from_slice(b"prtr"); // device class
    v[16..20].copy_from_slice(b"CMYK"); // data colour space (N=4)
    v[20..24].copy_from_slice(b"Lab "); // PCS
    v[36..40].copy_from_slice(b"acsp"); // profile-file signature
    v
}

/// A 3-component (RGB) ICC profile body — `output_intent_cmyk_profile` must
/// ignore it (only `N=4` entries are eligible, §8.6.5.5).
fn rgb_icc_profile() -> Vec<u8> {
    let mut v = cmyk_icc_profile();
    v[16..20].copy_from_slice(b"RGB "); // data colour space (N=3)
    v
}

/// Build a classic cross-reference-table PDF.
///
/// Objects 1–4 are a minimal catalog / page tree. `tail_objects` are appended
/// verbatim as `"<num> 0 obj\n<body>\nendobj\n"` with real byte offsets.
/// `bad_offsets` overrides specific objects' xref offsets with a bogus value to
/// simulate a corrupt xref entry (the object then fails to load).
fn build_pdf(
    catalog_entries: &str,
    tail_objects: &[(u32, Vec<u8>)],
    bad_offsets: &[(u32, u64)],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut offsets: Vec<(u32, usize)> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push((1, buf.len()));
    buf.extend_from_slice(
        format!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R {catalog_entries} >>\nendobj\n")
            .as_bytes(),
    );
    offsets.push((2, buf.len()));
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push((3, buf.len()));
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> >>\nendobj\n",
    );
    offsets.push((4, buf.len()));
    buf.extend_from_slice(b"4 0 obj\n<< >>\nendobj\n");

    for (num, body) in tail_objects {
        offsets.push((*num, buf.len()));
        buf.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
        buf.extend_from_slice(body);
        buf.extend_from_slice(b"\nendobj\n");
    }

    for &(num, bad) in bad_offsets {
        if let Some(entry) = offsets.iter_mut().find(|(n, _)| *n == num) {
            entry.1 = bad as usize;
        }
    }

    let size = offsets.iter().map(|(n, _)| *n).max().unwrap_or(4) + 1;
    let xref_off = buf.len();
    buf.extend_from_slice(format!("xref\n0 {size}\n0000000000 65535 f \n").as_bytes());
    for n in 1..size {
        let off = offsets
            .iter()
            .find(|(o, _)| *o == n)
            .map(|(_, o)| *o)
            .unwrap_or(0);
        buf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_off}\n%%EOF\n")
            .as_bytes(),
    );
    buf
}

/// An indirect ICC profile stream object `<< /N n /Length .. >> stream .. endstream`.
fn icc_stream_obj(num: u32, n: u8, profile: &[u8]) -> (u32, Vec<u8>) {
    let mut body = format!("<< /N {n} /Length {} >>\nstream\n", profile.len()).into_bytes();
    body.extend_from_slice(profile);
    body.extend_from_slice(b"\nendstream");
    (num, body)
}

/// A good CMYK OutputIntent dictionary (inline) whose `DestOutputProfile`
/// references object `icc_num`.
fn good_cmyk_oi(icc_num: u32) -> String {
    format!("<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (CMYK) /DestOutputProfile {icc_num} 0 R >>")
}

/// Regression for #712 (defect #1): a corrupt first entry whose object fails to
/// load (here a corrupt xref offset → `load_object` returns `Err`) must be
/// skipped, not abort the whole search — a valid CMYK profile later in the array
/// is still returned. Fails before the fix (the `?` propagated `None`).
#[test]
fn corrupt_entry_does_not_abort_remaining_search() {
    let catalog = format!("/OutputIntents [5 0 R {}]", good_cmyk_oi(6));
    // Object 5 exists but its xref offset is corrupt (points past EOF) → Err.
    let pdf = build_pdf(
        &catalog,
        &[
            (5, b"<< /Type /OutputIntent >>".to_vec()),
            icc_stream_obj(6, 4, &cmyk_icc_profile()),
        ],
        &[(5, 9_000_000)],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "a corrupt earlier entry must not hide the valid CMYK profile in entry 2 (#712)"
    );
}

/// Defect #2 guard: an entry whose `DestOutputProfile` stream cannot be decoded
/// is skipped (and now logged), and a later valid CMYK profile is returned.
#[test]
fn undecodable_profile_stream_entry_skipped_then_good() {
    // /Filter /FlateDecode with non-flate bytes → decode_stream_data Err.
    let bad_stream =
        b"<< /N 4 /Filter /FlateDecode /Length 9 >>\nstream\nnot-flate\nendstream".to_vec();
    let catalog = format!(
        "/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /DestOutputProfile 5 0 R >> {}]",
        good_cmyk_oi(6)
    );
    let pdf =
        build_pdf(&catalog, &[(5, bad_stream), icc_stream_obj(6, 4, &cmyk_icc_profile())], &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "an undecodable profile stream must be skipped, not abort the search (#712)"
    );
}

/// Defect #2 guard: an entry whose stream decodes but is not a valid ICC profile
/// is skipped (and now logged), and a later valid CMYK profile is returned.
#[test]
fn unparseable_icc_entry_skipped_then_good() {
    let not_icc = b"this is not an ICC profile, definitely too short and wrong".to_vec();
    let catalog = format!(
        "/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /DestOutputProfile 5 0 R >> {}]",
        good_cmyk_oi(6)
    );
    let pdf = build_pdf(
        &catalog,
        &[
            icc_stream_obj(5, 4, &not_icc),
            icc_stream_obj(6, 4, &cmyk_icc_profile()),
        ],
        &[],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    assert!(
        doc.output_intent_cmyk_profile().is_some(),
        "an unparseable ICC entry must be skipped, not abort the search (#712)"
    );
}

/// Happy path is unchanged: a single valid CMYK OutputIntent is returned.
#[test]
fn single_valid_cmyk_profile_is_found() {
    let catalog = format!("/OutputIntents [{}]", good_cmyk_oi(5));
    let pdf = build_pdf(&catalog, &[icc_stream_obj(5, 4, &cmyk_icc_profile())], &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    assert!(doc.output_intent_cmyk_profile().is_some());
}

/// No `/OutputIntents` at all → `None` (unchanged).
#[test]
fn no_output_intents_returns_none() {
    let pdf = build_pdf("", &[], &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    assert!(doc.output_intent_cmyk_profile().is_none());
}

/// Only a non-CMYK (RGB, `N=3`) OutputIntent → `None` (CMYK filter, §8.6.5.5).
#[test]
fn rgb_only_output_intent_returns_none() {
    let catalog =
        "/OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /DestOutputProfile 5 0 R >>]";
    let pdf = build_pdf(catalog, &[icc_stream_obj(5, 3, &rgb_icc_profile())], &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("open synthetic PDF");
    assert!(doc.output_intent_cmyk_profile().is_none());
}
