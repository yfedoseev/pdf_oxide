//! JPEG 2000 (`/JPXDecode`) extraction test (issue #755).
//!
//! Gated on the `jpeg2000` feature (OpenJPEG via `jpeg2k`). Without it, JPX is
//! unsupported by design and this test is skipped. (Unit coverage of the decoder
//! itself lives in `src/decoders/jpx.rs`.)
#![cfg(feature = "jpeg2000")]

use pdf_oxide::document::PdfDocument;

#[test]
fn extract_jpx_image_from_pdf() {
    // The minimal repro is one page of text plus one `/JPXDecode` image XObject.
    // Before #755 this path errored with `UnsupportedFilter` and the image was
    // dropped; it must now decode to a valid raster.
    let doc = PdfDocument::open("tests/fixtures/jpx/jpx_minimal.pdf").expect("open JPX repro");
    let images = doc.extract_images(0).expect("extract page-0 images");
    assert!(!images.is_empty(), "no images extracted from the JPX page");

    let png = images[0]
        .to_png_bytes()
        .expect("encode the extracted JPX image as PNG");
    assert!(
        png.len() > 8 && &png[1..4] == b"PNG",
        "extracted JPX image did not encode to a valid PNG"
    );
}
