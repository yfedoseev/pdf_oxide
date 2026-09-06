//! What does a 4-component JPEG with **no** APP14 marker decode to?
//!
//! ISO 32000-1:2008 Table 13 (`docs/spec/pdf.md:2979`) is explicit:
//!
//! > If the Adobe-defined marker code … is not present and this dictionary
//! > entry is not present in the filter dictionary then the default value of
//! > ColorTransform shall be 1 if the image has three components and
//! > **0 otherwise**.
//!
//! So "no marker, four components" *is* `ColorTransform 0` — the same case as
//! an explicit Adobe marker with transform 0. The extractor undoes
//! `jpeg-decoder`'s inversion only when a marker is present, which treats the
//! two oppositely.
//!
//! Whether that is actually wrong depends on a fact about `jpeg-decoder`, not
//! about the specification: if the decoder itself only inverts when it sees the
//! marker, then gating on the marker is correct. The existing contract test
//! pins the marker-present direction and says nothing about the marker-absent
//! one, which is exactly the gap.
//!
//! This measures it. The same image is encoded twice and the APP14 segment is
//! stripped from one copy, so the entropy-coded data is byte-identical and the
//! marker is the only difference.

use jpeg_encoder::{ColorType, Encoder};

/// Encode one pure-cyan CMYK pixel. `jpeg_encoder` writes an Adobe APP14
/// marker with `color_transform = 0` and stores the samples inverted.
fn encode_cmyk_pixel() -> Vec<u8> {
    let mut out = Vec::new();
    let encoder = Encoder::new(&mut out, 100);
    // Straight CMYK: full cyan, no magenta/yellow/black.
    encoder
        .encode(&[255u8, 0, 0, 0], 1, 1, ColorType::Cmyk)
        .expect("encode cmyk jpeg");
    out
}

/// Remove the APP14 (`FF EE`) segment, leaving everything else byte-identical.
fn strip_app14(jpeg: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(jpeg.len());
    let mut i = 0usize;
    while i < jpeg.len() {
        if i + 3 < jpeg.len() && jpeg[i] == 0xFF && jpeg[i + 1] == 0xEE {
            let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            i += 2 + len; // skip marker and its payload
            continue;
        }
        out.push(jpeg[i]);
        i += 1;
    }
    out
}

fn decode(jpeg: &[u8]) -> Vec<u8> {
    let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(jpeg));
    d.decode().expect("decode")
}

/// The APP14 stripper must actually remove the marker, or the comparison
/// below is between two identical files.
#[test]
fn test_fixture_really_removes_the_app14_marker() {
    let with_marker = encode_cmyk_pixel();
    let without = strip_app14(&with_marker);
    assert!(
        with_marker.windows(2).any(|w| w == [0xFF, 0xEE]),
        "the encoder should have written an APP14 marker"
    );
    assert!(
        !without.windows(2).any(|w| w == [0xFF, 0xEE]),
        "the APP14 marker should be gone"
    );
    assert!(without.len() < with_marker.len());
}

/// The measurement. If these decode identically, `jpeg-decoder` inverts
/// 4-component data regardless of the marker — and gating the undo on the
/// marker's presence is then wrong for exactly the Table 13 case above.
#[test]
fn records_whether_jpeg_decoder_inverts_without_an_app14_marker() {
    let with_marker = encode_cmyk_pixel();
    let without_marker = strip_app14(&with_marker);

    let a = decode(&with_marker);
    let b = decode(&without_marker);

    assert_eq!(a.len(), b.len(), "same image, same sample count");
    assert_eq!(
        a, b,
        "jpeg-decoder produced different samples with and without the APP14 \
         marker ({a:?} vs {b:?}). If this ever fires, the extractor's \
         marker-gated inversion undo is correct as written and Table 13 does \
         not apply to it — update the reasoning on the CMYK-JPEG gate."
    );
}
