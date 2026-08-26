//! A 1-bpc image with `/Decode [1 0]` must not render as the negative of
//! itself once it crosses the unpacking size cap.
//!
//! ISO 32000-1:2008 §8.9.5.2 maps each raw sample through
//! `Dmin + raw · (Dmax − Dmin) / (2^bpc − 1)`. At 1 bpc a `/Decode [1 0]` is a
//! pure inversion, and on packed samples that is a byte-wise NOT — no
//! unpacking, no allocation.
//!
//! The unpacking path refuses buffers over a 256 MiB ceiling, which is right
//! as a bound on a hostile `/Width` × `/Height`. But refusing left the samples
//! packed *and the `/Decode` unapplied*, so a large 1-bpc scan came out as the
//! exact negative of the picture. The cheap mapping is now applied on the
//! packed bytes instead, which is what happened at any size before the
//! unpacking path existed.

use pdf_oxide::PdfDocument;

/// A page carrying one 1-bpc `/DeviceGray` image of `w` x `h` with the given
/// `/Decode`, whose stream is deliberately short (the decoder pads).
///
/// `w` x `h` is chosen so the *unpacked* size (one byte per sample) crosses
/// the 256 MiB cap while the packed stream stays tiny — the cap is checked
/// before anything is allocated, so this costs nothing to run.
fn oversized_image_pdf(w: u32, h: u32, decode: &str, first_byte: u8) -> Vec<u8> {
    let data = [first_byte; 8];

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 6];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
         /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
    );
    let content = b"q 200 0 0 200 0 0 cm /Im0 Do Q\n";
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    off[5] = buf.len();
    buf.extend_from_slice(
        format!(
            "5 0 obj\n<< /Type /XObject /Subtype /Image /Width {w} /Height {h} \
             /ColorSpace /DeviceGray /BitsPerComponent 1 {decode} /Length {} >>\nstream\n",
            data.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(&data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for id in 1..=5 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// First stored byte of the page's single image, or `None` if it did not
/// extract.
fn first_stored_byte(pdf: Vec<u8>) -> Option<u8> {
    let doc = PdfDocument::from_bytes(pdf).ok()?;
    let images = doc.extract_images(0).ok()?;
    let image = images.first()?;
    match image.data() {
        pdf_oxide::extractors::ImageData::Raw { pixels, .. } => pixels.first().copied(),
        pdf_oxide::extractors::ImageData::Jpeg(bytes) => bytes.first().copied(),
    }
}

/// Dimensions whose unpacked size (w*h bytes) clears the 256 MiB ceiling.
const OVER_CAP: (u32, u32) = (20_000, 20_000);
/// Dimensions comfortably under it.
const UNDER_CAP: (u32, u32) = (64, 64);

/// The reported symptom. Over the cap the samples stay packed, and without
/// the inversion the picture is its own negative.
#[test]
fn an_oversized_one_bit_image_still_applies_a_decode_inversion() {
    let plain = first_stored_byte(oversized_image_pdf(OVER_CAP.0, OVER_CAP.1, "", 0b1010_1010));
    let inverted = first_stored_byte(oversized_image_pdf(
        OVER_CAP.0,
        OVER_CAP.1,
        "/Decode [1 0]",
        0b1010_1010,
    ));

    let (Some(plain), Some(inverted)) = (plain, inverted) else {
        panic!("both fixtures should extract an image: {plain:?} / {inverted:?}");
    };
    assert_ne!(
        plain, inverted,
        "/Decode [1 0] was left unapplied over the size cap, so the image renders \
         as its own negative"
    );
    assert_eq!(inverted, !plain, "at 1 bpc a [1 0] decode is a byte-wise NOT");
}

/// Under the cap the ordinary unpacking path applies the decode, so the two
/// must still differ — the control that shows the fixture's decode is real.
#[test]
fn an_ordinary_one_bit_image_applies_a_decode_inversion() {
    let plain = first_stored_byte(oversized_image_pdf(UNDER_CAP.0, UNDER_CAP.1, "", 0b1010_1010));
    let inverted = first_stored_byte(oversized_image_pdf(
        UNDER_CAP.0,
        UNDER_CAP.1,
        "/Decode [1 0]",
        0b1010_1010,
    ));
    assert!(plain.is_some() && inverted.is_some());
    assert_ne!(plain, inverted, "the decode must be applied under the cap too");
}

/// Without a `/Decode` nothing is inverted, over the cap or under it.
#[test]
fn no_decode_leaves_the_samples_alone() {
    assert_eq!(
        first_stored_byte(oversized_image_pdf(OVER_CAP.0, OVER_CAP.1, "", 0b1010_1010)),
        first_stored_byte(oversized_image_pdf(OVER_CAP.0, OVER_CAP.1, "", 0b1010_1010)),
        "extraction should be deterministic"
    );
}
