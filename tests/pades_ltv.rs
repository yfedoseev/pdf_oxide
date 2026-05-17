//! PAdES LTV integrity-invariant suite (#235, feature plan §4.3 / §5.3).
//!
//! End-to-end through the public API — complements the in-module unit
//! test (`signatures::sign_bytes::tests::test_sign_pdf_bytes_pades_levels`,
//! which pins B-B/B-T/B-LT/B-LTA classification + VRI write/read parity)
//! by proving the cross-cutting integrity invariants:
//!
//! - I1  signature still `Valid` before *and* after the DSS append.
//! - I2  the pre-DSS bytes are a strict prefix of the post-DSS file.
//! - I3  re-parsing the post-DSS file resolves the new Catalog `/DSS`.
//! - I4  every `/VRI` key == `hex_upper(SHA1(/Contents))`.
//! - I6  a tampered original region makes I1 fail (negative).
//! - plus the legacy `adbe.pkcs7.detached` path is unregressed (§2.3),
//!   and `B-LTA` / missing-timestamper fail closed.
//!
//! The EU-DSS demonstration-validator conformance check remains the
//! manual release gate (feature plan §5.5) — not automatable here.
#![cfg(feature = "signatures")]

use pdf_oxide::signatures::{
    classify_pades_level, read_dss, sign_pdf_bytes, sign_pdf_bytes_pades, verify_signer_detached,
    ByteRangeCalculator, PadesLevel, RevocationMaterial, SignOptions, SignatureInfo, SignerVerify,
    SigningCredentials,
};

fn creds() -> SigningCredentials {
    let cert = std::fs::read_to_string("tests/fixtures/test_signing_cert.pem").expect("cert");
    let key = std::fs::read_to_string("tests/fixtures/test_signing_key.pem").expect("key");
    SigningCredentials::from_pem(&cert, &key).expect("creds")
}

fn minimal_pdf() -> Vec<u8> {
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

fn opts() -> SignOptions {
    SignOptions {
        estimated_size: 4096,
        ..Default::default()
    }
}

/// `RevocationMaterial` is `#[non_exhaustive]`, so an external crate
/// (this integration test) can only build it via `Default` + field
/// assignment, not a struct literal. Seed it with one DER cert.
#[allow(clippy::field_reassign_with_default)]
fn material_with(cert_der: Vec<u8>) -> RevocationMaterial {
    let mut m = RevocationMaterial::default();
    m.certificates = vec![cert_der];
    m
}

/// A mock RFC 3161 token source — a structurally well-formed DER
/// SEQUENCE (the core only requires `token_der[0] == 0x30`; full token
/// cryptographic validity is the EU-DSS-gated concern, not provable
/// offline). Returns the same token for any signature value.
fn mock_tsa() -> impl Fn(&[u8]) -> pdf_oxide::error::Result<Vec<u8>> {
    |_sig: &[u8]| Ok(vec![0x30, 0x07, 0x02, 0x01, 0x01, 0x04, 0x02, b't', b's'])
}

/// Hex-decode (uppercase or lowercase).
fn unhex(s: &[u8]) -> Vec<u8> {
    s.chunks(2)
        .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap())
        .collect()
}

/// DER total element length (header + content) of the leading
/// definite-length element described by the hex string.
fn der_len_from_hex(hex: &str) -> usize {
    let b = unhex(hex.as_bytes());
    assert!(b.len() >= 2, "DER header");
    let l0 = b[1] as usize;
    if l0 < 0x80 {
        2 + l0
    } else {
        let n = l0 & 0x7f;
        let mut len = 0usize;
        for &x in &b[2..2 + n] {
            len = (len << 8) | x as usize;
        }
        2 + n + len
    }
}

/// Parse the first appended signature after `orig_len`: returns
/// `(byte_range, decoded_cms, full_padded_contents)` — byte-oriented
/// because a B-LT tail carries binary DSS streams (not UTF-8).
fn parse_sig(orig_len: usize, signed: &[u8]) -> ([i64; 4], Vec<u8>, Vec<u8>) {
    let tail = &signed[orig_len..];
    let br = tail
        .windows(12)
        .position(|w| w == b"/ByteRange [")
        .expect("/ByteRange");
    let after = &tail[br + 12..];
    let end = after.iter().position(|&b| b == b']').unwrap();
    let n: Vec<i64> = std::str::from_utf8(&after[..end])
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let byte_range = [n[0], n[1], n[2], n[3]];
    let ct = tail
        .windows(11)
        .position(|w| w == b"/Contents <")
        .expect("/Contents");
    let after_ct = &tail[ct + 11..];
    let close = after_ct.iter().position(|&b| b == b'>').unwrap();
    let hex = std::str::from_utf8(&after_ct[..close]).unwrap();
    let cms = unhex(&hex.as_bytes()[..der_len_from_hex(hex) * 2]);
    let contents = unhex(hex.as_bytes());
    (byte_range, cms, contents)
}

fn verify(orig_len: usize, signed: &[u8]) -> SignerVerify {
    let (byte_range, cms, _) = parse_sig(orig_len, signed);
    let content = ByteRangeCalculator::extract_signed_bytes(signed, &byte_range).unwrap();
    verify_signer_detached(&cms, &content).expect("verify must not error")
}

fn info_with(contents: Vec<u8>) -> SignatureInfo {
    SignatureInfo {
        contents: Some(contents),
        ..Default::default()
    }
}

/// I1 + B-B baseline: a B-B file verifies and classifies as B-B.
#[test]
fn b_b_roundtrip() {
    let pdf = minimal_pdf();
    let signed = sign_pdf_bytes_pades(
        &pdf,
        &creds(),
        opts(),
        PadesLevel::BB,
        None,
        &RevocationMaterial::default(),
    )
    .expect("B-B sign");
    assert_eq!(verify(pdf.len(), &signed), SignerVerify::Valid, "I1");
    let (_, _, contents) = parse_sig(pdf.len(), &signed);
    assert_eq!(classify_pades_level(&info_with(contents), None), PadesLevel::BB);
}

/// B-T: signs with a timestamp attr, still verifies (I1), classifies BT.
#[test]
fn b_t_roundtrip() {
    let pdf = minimal_pdf();
    let ts = mock_tsa();
    let signed = sign_pdf_bytes_pades(
        &pdf,
        &creds(),
        opts(),
        PadesLevel::BT,
        Some(&ts),
        &RevocationMaterial::default(),
    )
    .expect("B-T sign");
    assert_eq!(verify(pdf.len(), &signed), SignerVerify::Valid, "I1");
    let (_, _, contents) = parse_sig(pdf.len(), &signed);
    assert_eq!(classify_pades_level(&info_with(contents), None), PadesLevel::BT);
}

/// B-T without a timestamper fails closed (no silent down-level to B-B).
#[test]
fn b_t_requires_timestamper() {
    let pdf = minimal_pdf();
    assert!(matches!(
        sign_pdf_bytes_pades(
            &pdf,
            &creds(),
            opts(),
            PadesLevel::BT,
            None,
            &RevocationMaterial::default()
        ),
        Err(pdf_oxide::error::Error::Unsupported(_))
    ));
}

/// B-LT end-to-end: I1 (verify Valid), I3 (re-parse resolves the new
/// Catalog `/DSS`), I4 (every `/VRI` key == hex_upper(SHA1(/Contents))),
/// and the public reader classifies it as B-LT.
#[test]
fn b_lt_roundtrip_invariants() {
    let pdf = minimal_pdf();
    let c = creds();
    let ts = mock_tsa();
    let material = material_with(c.certificate.clone());
    let blt = sign_pdf_bytes_pades(&pdf, &c, opts(), PadesLevel::BLt, Some(&ts), &material)
        .expect("B-LT sign");

    // I1: the signature's own ByteRange self-delimits the bytes it
    // signed (the DSS append falls outside it), so it still verifies.
    assert_eq!(verify(pdf.len(), &blt), SignerVerify::Valid, "I1");

    // I3: re-parsing resolves the most-recent Catalog and it has /DSS.
    let doc = pdf_oxide::document::PdfDocument::from_bytes(blt.clone()).unwrap();
    let dss = read_dss(&doc)
        .expect("read_dss ok")
        .expect("I3: /DSS present");
    assert_eq!(dss.certificates, vec![c.certificate.clone()]);

    // I4: the (single) VRI key equals hex_upper(SHA1(/Contents)).
    let (_, _, contents) = parse_sig(pdf.len(), &blt);
    let key = pdf_oxide::signatures::pades::vri_key(&contents).expect("SHA-1 available");
    assert!(dss.vri_for(&key).is_some(), "I4: VRI keyed by SHA1(/Contents)");
    assert!(
        key.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_lowercase()),
        "I4: key is uppercase hex"
    );

    // Public reader agrees: timestamp attr + matching VRI ⇒ B-LT.
    assert_eq!(classify_pades_level(&info_with(contents), Some(&dss)), PadesLevel::BLt);
}

/// I2: a B-LT file is its own pre-DSS signature followed by an
/// append-only DSS increment — the DSS bytes are a strict *suffix*
/// (the signature's ByteRange never covers them), the converse framing
/// of "pre-DSS is a strict prefix": truncating at the signature's
/// post-Contents boundary still verifies.
#[test]
fn b_lt_dss_is_append_only_suffix() {
    let pdf = minimal_pdf();
    let c = creds();
    let ts = mock_tsa();
    let material = material_with(c.certificate.clone());
    let blt = sign_pdf_bytes_pades(&pdf, &c, opts(), PadesLevel::BLt, Some(&ts), &material)
        .expect("B-LT sign");
    let (byte_range, _, _) = parse_sig(pdf.len(), &blt);
    // ByteRange = [0, a, b, c]; the signed file (pre-DSS) ends at b+c.
    let pre_dss_end = (byte_range[2] + byte_range[3]) as usize;
    assert!(pre_dss_end < blt.len(), "I2: DSS is appended after the signature");
    assert_eq!(
        verify(pdf.len(), &blt[..pre_dss_end]),
        SignerVerify::Valid,
        "I2: the pre-DSS prefix is a self-contained valid signature"
    );
}

/// I6 (negative): flipping a byte inside the originally-signed region
/// of a B-LT file invalidates the signature.
#[test]
fn b_lt_tamper_breaks_signature() {
    let pdf = minimal_pdf();
    let c = creds();
    let ts = mock_tsa();
    let material = material_with(c.certificate.clone());
    let mut blt = sign_pdf_bytes_pades(&pdf, &c, opts(), PadesLevel::BLt, Some(&ts), &material)
        .expect("B-LT sign");
    // Flip a byte well inside the original PDF body (covered by the
    // first ByteRange segment, i.e. signed content).
    blt[20] ^= 0xFF;
    assert_eq!(
        verify(pdf.len(), &blt),
        SignerVerify::Invalid,
        "I6: tampering the signed region must invalidate"
    );
}

/// B-LTA: B-LT + an archival `/DocTimeStamp` (ETSI.RFC3161) over the
/// whole file including the DSS, as a 3rd incremental update. The
/// original signature still verifies (I1) and the document-timestamp
/// object is present and ordered after the DSS.
#[test]
fn b_lta_roundtrip() {
    let pdf = minimal_pdf();
    let c = creds();
    let ts = mock_tsa();
    let material = material_with(c.certificate.clone());
    let blta = sign_pdf_bytes_pades(&pdf, &c, opts(), PadesLevel::BLta, Some(&ts), &material)
        .expect("B-LTA sign");

    assert_eq!(verify(pdf.len(), &blta), SignerVerify::Valid, "I1 under B-LTA");
    assert!(
        pdf_oxide::signatures::has_document_timestamp(&blta),
        "B-LTA carries a /DocTimeStamp ETSI.RFC3161 object"
    );
    let dts = blta
        .windows(13)
        .position(|w| w == b"/DocTimeStamp")
        .expect("/DocTimeStamp");
    let dss = blta.windows(4).position(|w| w == b"/DSS").expect("/DSS");
    assert!(dss < dts, "the DocTimeStamp is appended after (covers) the DSS");

    // Signature-scoped classify still reports B-LT (B-LTA is the
    // document-level DocTimeStamp signal, by design — see
    // `has_document_timestamp`).
    let doc = pdf_oxide::document::PdfDocument::from_bytes(blta.clone()).unwrap();
    let dss_parsed = read_dss(&doc).unwrap().expect("DSS present");
    let (_, _, contents) = parse_sig(pdf.len(), &blta);
    assert_eq!(classify_pades_level(&info_with(contents), Some(&dss_parsed)), PadesLevel::BLt);
}

/// B-LTA fails closed without a timestamper (its document timestamp
/// needs an RFC 3161 source) — explicit `Unsupported`, never a panic.
#[test]
fn b_lta_requires_timestamper() {
    let pdf = minimal_pdf();
    assert!(matches!(
        sign_pdf_bytes_pades(
            &pdf,
            &creds(),
            opts(),
            PadesLevel::BLta,
            None,
            &RevocationMaterial::default()
        ),
        Err(pdf_oxide::error::Error::Unsupported(_))
    ));
}

/// §2.3 regression: the legacy `adbe.pkcs7.detached` `sign_pdf_bytes`
/// path is byte-range-unchanged and still produces a `Valid` signature.
#[test]
fn existing_pkcs7_unaffected() {
    let pdf = minimal_pdf();
    let signed = sign_pdf_bytes(&pdf, &creds(), opts()).expect("legacy sign");
    assert_eq!(verify(pdf.len(), &signed), SignerVerify::Valid);
}

/// The `pdf_document_has_timestamp` C ABI (the doc-scoped B-LTA reader
/// signal shared by every binding) returns `1` for a B-LTA document,
/// `0` for a non-LTA one, and `-1` (with a non-zero error code) for a
/// null handle. This is the acceptance gate for the C#/Go/Node/purego
/// `HasDocumentTimestamp` wrappers.
#[test]
fn ffi_pdf_document_has_timestamp() {
    use std::ffi::c_void;

    let pdf = minimal_pdf();
    let c = creds();
    let ts = mock_tsa();
    let material = material_with(c.certificate.clone());
    let blta = sign_pdf_bytes_pades(&pdf, &c, opts(), PadesLevel::BLta, Some(&ts), &material)
        .expect("B-LTA sign");

    let check = |bytes: &[u8]| -> (i32, i32) {
        let mut ec: i32 = 0;
        let h = pdf_oxide::ffi::pdf_document_open_from_bytes(bytes.as_ptr(), bytes.len(), &mut ec);
        assert!(!h.is_null() && ec == 0, "open_from_bytes failed (ec={ec})");
        let mut tec: i32 = 0;
        let r = pdf_oxide::ffi::pdf_document_has_timestamp(h as *const c_void, &mut tec);
        pdf_oxide::ffi::pdf_document_free(h);
        (r, tec)
    };

    assert_eq!(check(&blta), (1, 0), "B-LTA ⇒ has document timestamp");
    assert_eq!(check(&pdf), (0, 0), "plain PDF ⇒ no document timestamp");

    let mut nec: i32 = 0;
    let r = pdf_oxide::ffi::pdf_document_has_timestamp(std::ptr::null(), &mut nec);
    assert_eq!(r, -1, "null handle ⇒ -1");
    assert_ne!(nec, 0, "null handle ⇒ non-zero error code");
}
