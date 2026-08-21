//! Consumers that compare stored image samples against values taken from the
//! image dictionary — colour-key `/Mask` ranges (ISO 32000-1 §8.9.6.4),
//! separation-plate routing — read samples in the *raw* sample space. When the
//! extractor maps a non-default `/Decode` into the stored buffer (§8.9.5.2) it
//! must say so, or those consumers silently compare in the wrong space.
//!
//! A malformed `/BitsPerComponent` must also stay a recoverable decode error
//! rather than becoming a panic in the sub-byte unpacker.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::ImageData;

fn build_pdf(image_dict: &str, image_data: &[u8]) -> Vec<u8> {
    let content: &[u8] = b"q 100 0 0 100 0 0 cm /Im0 Do Q";
    let mut bodies: Vec<Vec<u8>> = vec![Vec::new(); 6]; // ids 1..=5
    bodies[1] = b"<< /Type /Catalog /Pages 2 0 R >>".to_vec();
    bodies[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec();
    bodies[3] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        .to_vec();
    let mut img = format!("{image_dict}\nstream\n").into_bytes();
    img.extend_from_slice(image_data);
    img.extend_from_slice(b"\nendstream");
    bodies[4] = img;
    let mut cs = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    cs.extend_from_slice(content);
    cs.extend_from_slice(b"\nendstream");
    bodies[5] = cs;

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = [0usize; 6];
    for id in 1..=5 {
        offsets[id] = out.len();
        out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        out.extend_from_slice(&bodies[id]);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for id in 1..=5 {
        out.extend_from_slice(format!("{:010} 00000 n \n", offsets[id]).as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    out
}

/// An image whose samples were rewritten through `/Decode` must advertise it,
/// so a colour-key `/Mask` consumer does not range-test the wrong space.
#[test]
fn decoded_samples_are_flagged_for_raw_sample_consumers() {
    let data: Vec<u8> = (0..64u8).collect();
    let doc = PdfDocument::from_bytes(build_pdf(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ColorSpace /DeviceGray /BitsPerComponent 8 /Decode [1 0] /Length 64 >>",
        &data,
    ))
    .expect("open pdf");
    let imgs = doc.extract_images(0).expect("extract_images");
    let img = &imgs[0];
    assert!(
        !img.samples_are_raw(),
        "a non-default /Decode was mapped into the samples but the image does not say so"
    );
}

/// The default `/Decode` leaves samples raw, so the flag must stay clear —
/// otherwise every ordinary image would lose colour-key masking.
#[test]
fn untouched_samples_are_not_flagged() {
    let data: Vec<u8> = (0..64u8).collect();
    let doc = PdfDocument::from_bytes(build_pdf(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 64 >>",
        &data,
    ))
    .expect("open pdf");
    let imgs = doc.extract_images(0).expect("extract_images");
    assert!(
        imgs[0].samples_are_raw(),
        "an image with no /Decode must remain in the raw sample space"
    );
}

/// Unpacking a sub-byte sample rescales it into the 8-bit range (bpc 1/2/4
/// scale by ×255/×85/×17), which leaves the raw sample space just as folding in
/// a `/Decode` does. With no `/Decode` at all the flag must still be clear, or a
/// colour-key `/Mask` consumer range-tests rescaled bytes against bounds stated
/// in the file's `0..2^bpc−1` space.
#[test]
fn rescaled_sub_byte_samples_are_not_raw() {
    // 4 bpc, width 8: four bytes per row, samples 0 and 15 alternating.
    let data: Vec<u8> = vec![0x0F; 4 * 8];
    let doc = PdfDocument::from_bytes(build_pdf(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ColorSpace /DeviceGray /BitsPerComponent 4 /Length 32 >>",
        &data,
    ))
    .expect("open pdf");
    let imgs = doc.extract_images(0).expect("extract_images");
    let img = &imgs[0];
    assert_eq!(img.bits_per_component(), 8, "unpacked samples are stored 8-bit");
    assert!(
        !img.samples_are_raw(),
        "sub-byte samples were rescaled to 0..255, so they are no longer in the \
         space a colour-key /Mask is expressed in"
    );
}

/// Leaving the raw sample space and having `/Decode` folded in are different
/// facts, and a sub-byte image with no `/Decode` separates them: its samples
/// were rescaled, but nothing applied a map. Plate routing applies `/Decode`
/// itself, so reading the wider fact here drops the inversion such an image is
/// still owed.
#[test]
fn sub_byte_without_decode_is_not_reported_as_decoded() {
    let data: Vec<u8> = vec![0x0F; 4 * 8];
    let doc = PdfDocument::from_bytes(build_pdf(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ColorSpace /DeviceGray /BitsPerComponent 4 /Length 32 >>",
        &data,
    ))
    .expect("open pdf");
    let imgs = doc.extract_images(0).expect("extract_images");
    let img = &imgs[0];
    assert!(!img.samples_are_raw(), "sub-byte samples were rescaled to 0..255");
    assert!(
        !img.decode_folded_in(),
        "no /Decode was present, so nothing folded one in — a consumer that \
         applies /Decode itself still owes this image its map"
    );
}

/// The complement: when a non-default `/Decode` really is folded in, plate
/// routing must not apply it a second time and double-invert.
#[test]
fn sub_byte_with_decode_reports_it_folded_in() {
    let data: Vec<u8> = vec![0x0F; 4 * 8];
    let doc = PdfDocument::from_bytes(build_pdf(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ColorSpace /DeviceGray /BitsPerComponent 4 /Decode [1 0] /Length 32 >>",
        &data,
    ))
    .expect("open pdf");
    let imgs = doc.extract_images(0).expect("extract_images");
    assert!(
        imgs[0].decode_folded_in(),
        "a non-default /Decode was mapped into the samples and must be advertised"
    );
}

/// 16-bit samples are collapsed to their high byte and `bits_per_component()`
/// then reports 8, so nothing downstream can infer the rescale from the depth.
/// A colour-key `/Mask` states its bounds in the file's `0..65535` space; if
/// the flag stays set it range-tests them against these bytes and a
/// `/Mask [0 30000]` turns the whole image transparent.
#[test]
fn sixteen_bpc_reduced_samples_are_not_raw() {
    let data: Vec<u8> = (0..128u8).collect();
    let doc = PdfDocument::from_bytes(build_pdf(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ColorSpace /DeviceGray /BitsPerComponent 16 /Length 128 >>",
        &data,
    ))
    .expect("open pdf");
    let imgs = doc.extract_images(0).expect("extract_images");
    let img = &imgs[0];
    assert_eq!(img.bits_per_component(), 8, "16-bit samples are stored reduced");
    assert!(
        !img.samples_are_raw(),
        "0..65535 was rescaled to 0..255, so the samples are no longer in the \
         space a colour-key /Mask is expressed in"
    );
    assert!(!img.decode_folded_in(), "no /Decode was present");
}

/// The 8-bit case is an identity scale, so it must stay raw — otherwise every
/// ordinary image loses colour-key masking. This pins the *branch*: 8 bpc with
/// no `/Decode` must skip the rewrite path entirely, which is what keeps it raw.
#[test]
fn eight_bpc_samples_stay_raw_without_decode() {
    let data: Vec<u8> = (0..64u8).collect();
    let doc = PdfDocument::from_bytes(build_pdf(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 64 >>",
        &data,
    ))
    .expect("open pdf");
    assert!(
        doc.extract_images(0).expect("extract_images")[0].samples_are_raw(),
        "8-bpc samples with no /Decode are untouched and must stay raw"
    );
}

/// `ColorSpace::DeviceN` is a unit variant — the parser drops the ink-name
/// array — so `components()` answers a flat 4 for every `/DeviceN`. That count
/// sets the sub-byte unpacker's row geometry, and `PixelFormat` can only
/// express a 1-, 3- or 4-byte stride, so a 2-ink image has no representation in
/// the stored buffer at all. Unpacking it anyway read rows at twice their true
/// stride and ran off the end of a correctly-sized stream, padding the bottom
/// half of the image with fabricated zeros. It must stay packed instead.
#[test]
fn two_ink_devicen_sub_byte_is_left_packed() {
    // 8×8, two inks, 4 bpc: 8×2 samples/row = 8 bytes/row, 64 bytes total —
    // exactly the right length for two inks, and half what four would need.
    let data: Vec<u8> = vec![0xFF; 64];
    let doc = PdfDocument::from_bytes(build_pdf(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ColorSpace [/DeviceN [/InkA /InkB] /DeviceCMYK \
           << /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 0] /C1 [1 1] >>] \
         /BitsPerComponent 4 /Length 64 >>",
        &data,
    ))
    .expect("open pdf");
    let imgs = doc.extract_images(0).expect("extract_images");
    let img = &imgs[0];
    assert_eq!(
        img.bits_per_component(),
        4,
        "an unrepresentable component count must leave the samples packed, \
         and sub-byte depth is what tells consumers not to read them"
    );
    if let ImageData::Raw { pixels, .. } = img.data() {
        assert_eq!(
            pixels.len(),
            64,
            "the packed stream is passed through untouched; unpacking at the \
             wrong stride produced {} bytes, half of them fabricated",
            pixels.len()
        );
    } else {
        panic!("expected a raw buffer");
    }
}

/// 4-bpc unpacking: two samples per byte, high nibble first. Width 9 forces a
/// padded row (36 bits in 5 bytes), so an unpacker that ignored row padding
/// would shear the image diagonally from row 1 onward.
#[test]
fn four_bpc_unpacks_with_row_padding() {
    // Per row: 0x0F 0x00 0x00 0x00 0x00 -> samples 0,15,0,0,0,0,0,0,0 and
    // four padding bits. Extraction skips images under 8x8, so the row is
    // widened rather than shortened.
    let row: [u8; 5] = [0x0F, 0x00, 0x00, 0x00, 0x00];
    let mut data: Vec<u8> = Vec::new();
    for _ in 0..8 {
        data.extend_from_slice(&row);
    }
    let doc = PdfDocument::from_bytes(build_pdf(
        &format!(
            "<< /Type /XObject /Subtype /Image /Width 9 /Height 8 \
             /ColorSpace /DeviceGray /BitsPerComponent 4 /Length {} >>",
            data.len()
        ),
        &data,
    ))
    .expect("open pdf");
    let imgs = doc.extract_images(0).expect("extract_images");
    let ImageData::Raw { pixels, .. } = imgs[0].data() else {
        panic!("expected raw samples");
    };
    assert_eq!(pixels.len(), 9 * 8, "one byte per sample");
    for (row_index, chunk) in pixels.chunks(9).enumerate() {
        assert_eq!(chunk[0], 0, "row {row_index} must restart at its own byte boundary");
        assert_eq!(chunk[1], 255, "row {row_index} second sample");
        assert!(
            chunk[2..].iter().all(|&b| b == 0),
            "row {row_index} tail must be zero, got {chunk:?}"
        );
    }
}

/// Sub-byte unpacking with more than one component: the per-component
/// `/Decode` range must follow the component index, not the sample index.
#[test]
fn two_bpc_rgb_applies_per_component_decode() {
    // Each pixel is 3 samples of 2 bits: 0, 1, 3 -> 00 01 11, so a row of 8
    // pixels is 24 samples = 48 bits = 6 bytes exactly.
    let pattern: u8 = 0b00_01_11_00; // px0 R,G,B + px1 R
    let row: Vec<u8> = vec![
        pattern,
        0b01_11_00_01,
        0b11_00_01_11,
        pattern,
        0b01_11_00_01,
        0b11_00_01_11,
    ];
    let mut data: Vec<u8> = Vec::new();
    for _ in 0..8 {
        data.extend_from_slice(&row);
    }
    let doc = PdfDocument::from_bytes(build_pdf(
        &format!(
            "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
             /ColorSpace /DeviceRGB /BitsPerComponent 2 /Decode [1 0 0 1 0 1] \
             /Length {} >>",
            data.len()
        ),
        &data,
    ))
    .expect("open pdf");
    let imgs = doc.extract_images(0).expect("extract_images");
    let ImageData::Raw { pixels, .. } = imgs[0].data() else {
        panic!("expected raw samples");
    };
    assert_eq!(pixels.len(), 8 * 8 * 3, "three bytes per pixel");
    // First pixel: R sample 0 inverted -> 255; G sample 1 identity -> 85;
    // B sample 3 identity -> 255. Only the red channel's range is inverted.
    assert_eq!(
        &pixels[..3],
        &[255, 85, 255],
        "per-component /Decode must key on the component, not the sample index"
    );
}

/// A `/BitsPerComponent` outside the spec's {1, 2, 4, 8, 16} must not reach
/// the sub-byte unpacker's shift arithmetic: extraction either declines the
/// image or returns it untouched, never panics.
#[test]
fn illegal_bits_per_component_does_not_panic() {
    for bpc in [3u8, 5, 6, 7, 12] {
        let dict = format!(
            "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
             /ColorSpace /DeviceGray /BitsPerComponent {bpc} /Decode [1 0] /Length 64 >>"
        );
        let data: Vec<u8> = (0..64u8).collect();
        let doc = PdfDocument::from_bytes(build_pdf(&dict, &data)).expect("open pdf");
        // Whatever it decides, it must decide it without unwinding.
        let extracted = doc.extract_images(0);
        if let Ok(imgs) = extracted {
            for img in &imgs {
                if let ImageData::Raw { pixels, .. } = img.data() {
                    // The stream is passed through untouched for an illegal
                    // depth, so the buffer keeps its original length.
                    assert_eq!(pixels.len(), 64, "bpc={bpc} buffer was rewritten");
                }
            }
        }
    }
}
