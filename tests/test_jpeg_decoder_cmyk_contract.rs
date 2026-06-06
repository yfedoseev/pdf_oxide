//! Pins the CMYK contract pdf_oxide depends on from `jpeg-decoder`.
//!
//! Adobe-authored CMYK JPEGs (Photoshop / Illustrator / InDesign — the
//! producers behind the overwhelming majority of CMYK JPEGs reaching
//! prepress PDFs) store channel values inverted in the entropy stream:
//! the byte 0 means "full ink", 255 means "no ink". `jpeg-decoder` 0.3
//! undoes that convention internally in `color_convert_line_cmyk`, so its
//! `Decoder::decode()` output is already straight CMYK (0 = no ink, 255 =
//! full ink). pdf_oxide consumes those bytes directly without further
//! inversion — see `decode_cmyk_jpeg_to_raw_cmyk` and
//! `decode_cmyk_jpeg_to_rgb_with_profile` in `src/extractors/images.rs`.
//!
//! If a future jpeg-decoder release changes this contract (stops
//! auto-inverting, gates it on APP14 only, etc.), the round-trip below
//! will fail and surface the regression at upgrade time before the
//! mismatch reaches real fixtures. If that happens, either pin the prior
//! jpeg-decoder version or reintroduce a per-channel inversion in the
//! two extractor functions.

use jpeg_encoder::{ColorType, Encoder};

/// Encode a 1-pixel pure-cyan CMYK image with `ColorType::Cmyk` (writes
/// Adobe APP14 marker with color_transform = 0, and stores samples
/// inverted in the entropy stream). Decode via `jpeg-decoder` directly.
/// The decoded bytes must come out as straight CMYK matching the input.
#[test]
fn jpeg_decoder_returns_straight_cmyk_for_app14_inverted_input() {
    // 8×8 pure-cyan source: C = 255, M = Y = K = 0 per pixel.
    let mut cmyk = Vec::with_capacity(8 * 8 * 4);
    for _ in 0..(8 * 8) {
        cmyk.extend_from_slice(&[255, 0, 0, 0]);
    }
    let mut jpeg = Vec::new();
    let encoder = Encoder::new(&mut jpeg, 95);
    encoder
        .encode(&cmyk, 8, 8, ColorType::Cmyk)
        .expect("encode CMYK JPEG");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
    let decoded = decoder.decode().expect("decode CMYK JPEG");
    let info = decoder.info().expect("decoder info");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert_eq!(decoded.len(), 8 * 8 * 4);

    // Tolerant assertions: JPEG quantisation can drift values a few
    // levels even at quality 95. The point is the *direction* of the
    // inversion — pure cyan must NOT come out as M = Y = K = 255 / C = 0.
    for (i, chunk) in decoded.chunks_exact(4).enumerate() {
        assert!(
            chunk[0] > 200,
            "pixel {i}: Cyan channel after jpeg-decoder auto-inversion (expected ≳255); got {chunk:?}"
        );
        assert!(chunk[1] < 50, "pixel {i}: Magenta channel low (expected ≈0); got {chunk:?}");
        assert!(chunk[2] < 50, "pixel {i}: Yellow channel low (expected ≈0); got {chunk:?}");
        assert!(chunk[3] < 50, "pixel {i}: Black channel low (expected ≈0); got {chunk:?}");
    }
}
