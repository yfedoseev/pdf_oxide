//! Adobe CMYK JPEG (APP14 `color_transform = 0`) must decode to true colour,
//! not near-black.
//!
//! A DCTDecode `/DeviceCMYK` image whose JPEG carries an Adobe APP14 marker
//! with `color_transform = 0` (Photoshop / Distiller default - the common
//! born-print case) is decoded by `jpeg-decoder` 0.3 with an internal
//! `255 - x` inversion (`color_convert_line_cmyk`). PDF renderers (poppler,
//! Ghostscript) do NOT invert - they use the raw DCT samples as straight
//! CMYK ink. So pdf_oxide must undo jpeg-decoder's inversion before the
//! CMYK->RGB conversion, otherwise the raw low-ink samples come out as heavy
//! ink and the image renders near-black.
//!
//! Regression: govdocs1 00959_003971 p0 (three cover-page CMYK JPEGs) used
//! to decode to mean RGB ~[1.5, 1.5, 3.7] instead of poppler's cream
//! ~[222, 217, 197].
//!
//! Input construction: `jpeg-encoder`'s `ColorType::Cmyk` writes an Adobe
//! APP14 marker with `color_transform = 0` and stores `255 - param` in the
//! entropy stream (the raw DCT sample plane). So passing a CMYK pixel
//! `param` here produces a JPEG whose raw DCT samples are `255 - param` -
//! exactly the straight-CMYK-ink convention poppler decodes. Round-tripping
//! through jpeg-decoder returns `param`, and pdf_oxide's Adobe inversion then
//! recovers the raw `255 - param` sample plane before applying sec 10.3.5.

use jpeg_encoder::{ColorType, Encoder};
use pdf_oxide::extractors::images::{ColorSpace, ImageData, PdfImage};

/// Build a solid-colour 16x16 CMYK JPEG (Adobe APP14, transform = 0) from a
/// single CMYK pixel `param`.
fn cmyk_jpeg_transform0(param: [u8; 4]) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(16 * 16 * 4);
    for _ in 0..(16 * 16) {
        pixels.extend_from_slice(&param);
    }
    let mut jpeg = Vec::new();
    Encoder::new(&mut jpeg, 100)
        .encode(&pixels, 16, 16, ColorType::Cmyk)
        .expect("encode CMYK JPEG");
    jpeg
}

fn mean_rgb(png: &[u8]) -> [f64; 3] {
    let img = image::load_from_memory(png).expect("decode PNG").to_rgb8();
    let n = (img.width() * img.height()) as f64;
    let mut sum = [0f64; 3];
    for p in img.pixels() {
        sum[0] += p[0] as f64;
        sum[1] += p[1] as f64;
        sum[2] += p[2] as f64;
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

#[test]
fn adobe_cmyk_transform0_jpeg_decodes_to_true_colour_not_black() {
    // param = [255, 255, 55, 255] -> raw DCT samples 255 - param = [0, 0, 200, 0]
    // (yellow ink, C=M=K=0, Y=200). sec 10.3.5:
    //   R = 255 - min(255, 0 + 0)   = 255
    //   G = 255 - min(255, 0 + 0)   = 255
    //   B = 255 - min(255, 200 + 0) = 55
    // => bright yellow ~[255, 255, 55]. The pre-fix (no Adobe inversion) path
    // fed [255, 255, 55, 255] straight into sec 10.3.5 and produced [0, 0, 0].
    let jpeg = cmyk_jpeg_transform0([255, 255, 55, 255]);

    let img = PdfImage::new(16, 16, ColorSpace::DeviceCMYK, 8, ImageData::Jpeg(jpeg));
    let png = img.to_png_bytes().expect("to_png_bytes");
    let [r, g, b] = mean_rgb(&png);

    assert!(
        r > 200.0 && g > 200.0 && b < 120.0,
        "Adobe CMYK transform=0 JPEG must decode to bright yellow ~[255,255,55], \
         got [{:.1}, {:.1}, {:.1}] (near-black => the APP14 inversion regressed)",
        r,
        g,
        b
    );
}
