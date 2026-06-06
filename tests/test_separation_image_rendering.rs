//! Tests for raster image XObject routing in the separation-plate renderer
//! (ISO 32000-1 §8.9 image XObjects, §11.7.4 overprint placement).
//!
//! Spatial convention: 100×100 page rendered at 72 DPI = 100×100 pixel
//! plates. Images are placed with `50 0 0 50 25 25 cm` — a 50×50 user-space
//! square centred on the page, occupying image rows 25..75, cols 25..75.

#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_separations, SeparationPlate};

fn sample(plate: &SeparationPlate, x: u32, y: u32) -> u8 {
    plate.data[(y * plate.width + x) as usize]
}

fn plate<'a>(plates: &'a [SeparationPlate], name: &str) -> &'a SeparationPlate {
    plates
        .iter()
        .find(|p| p.ink_name == name)
        .unwrap_or_else(|| {
            panic!(
                "missing plate {name:?}; have {:?}",
                plates
                    .iter()
                    .map(|p| p.ink_name.as_str())
                    .collect::<Vec<_>>()
            )
        })
}

fn finalize_pdf(mut buf: Vec<u8>, offsets: Vec<usize>) -> Vec<u8> {
    let xref_offset = buf.len();
    buf.extend_from_slice(b"xref\n");
    buf.extend_from_slice(format!("0 {}\n", offsets.len() + 1).as_bytes());
    buf.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len() + 1,
            xref_offset
        )
        .as_bytes(),
    );
    buf
}

/// Build a single-page PDF where /Im1 is a `width × height` DeviceCMYK image
/// painted at the unit square via `50 0 0 50 25 25 cm`. `cmyk_samples` is
/// interleaved 8-bpc CMYK (W*H*4 bytes).
fn build_pdf_with_cmyk_image(cmyk_samples: &[u8], width: u32, height: u32) -> Vec<u8> {
    let content = b"q\n50 0 0 50 25 25 cm\n/Im1 Do\nQ\n";
    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R /Resources << /XObject << /Im1 5 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let img_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width {w} /Height {h} \
         /ColorSpace /DeviceCMYK /BitsPerComponent 8 /Length {len} >>\nstream\n",
        w = width,
        h = height,
        len = cmyk_samples.len()
    );
    buf.extend_from_slice(img_hdr.as_bytes());
    buf.extend_from_slice(cmyk_samples);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    finalize_pdf(buf, offsets)
}

#[test]
fn cmyk_image_routes_channels_to_process_plates() {
    // 2×2 image:
    //   pixel 0 (top-left, image origin)     : Cyan = 255
    //   pixel 1 (top-right)                  : Magenta = 255
    //   pixel 2 (bottom-left)                : Yellow = 255
    //   pixel 3 (bottom-right)               : Black = 255
    // After the unit-square Y flip the image's top-left lands at the
    // top-left of the painted region in PDF user space (since the cm
    // matrix `50 0 0 50 25 25` translates the unit square to
    // (25, 25)–(75, 75)). The plate is rendered with PDF y flipped to
    // image-row order, so the C pixel lands at image-row 25..50, col 25..50;
    // K lands at image-row 50..75, col 50..75.
    let cmyk: Vec<u8> = vec![
        255, 0, 0, 0, // C
        0, 255, 0, 0, // M
        0, 0, 255, 0, // Y
        0, 0, 0, 255, // K
    ];
    let doc = PdfDocument::from_bytes(build_pdf_with_cmyk_image(&cmyk, 2, 2)).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let c = plate(&plates, "Cyan");
    let m = plate(&plates, "Magenta");
    let y = plate(&plates, "Yellow");
    let k = plate(&plates, "Black");

    // Sample inside each quadrant of the painted region (25..75 × 25..75).
    // Image top-left cell (Cyan) → plate top-left within the region.
    assert!(
        sample(c, 35, 35) > 200,
        "Cyan channel lands on Cyan plate (top-left quadrant); got {}",
        sample(c, 35, 35)
    );
    assert!(
        sample(m, 60, 35) > 200,
        "Magenta channel lands on Magenta plate (top-right quadrant); got {}",
        sample(m, 60, 35)
    );
    assert!(
        sample(y, 35, 60) > 200,
        "Yellow channel lands on Yellow plate (bottom-left quadrant); got {}",
        sample(y, 35, 60)
    );
    assert!(
        sample(k, 60, 60) > 200,
        "Black channel lands on Black plate (bottom-right quadrant); got {}",
        sample(k, 60, 60)
    );

    // Outside the image bbox, plates stay at zero.
    assert_eq!(sample(c, 5, 5), 0, "Cyan untouched outside image bbox");
    assert_eq!(sample(k, 5, 5), 0, "Black untouched outside image bbox");
}

#[test]
fn cmyk_image_inside_form_xobject_routes_to_plates() {
    // Page invokes a Form that contains the CMYK image — verify the recursion
    // through Operator::Do reaches the image branch.
    let content = b"/F1 Do\n";
    let form_content = b"q\n50 0 0 50 25 25 cm\n/Im1 Do\nQ\n";
    // 1×1 image with K = 255.
    let cmyk: Vec<u8> = vec![0, 0, 0, 255];

    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R /Resources << /XObject << /F1 5 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let form_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
            /Resources << /XObject << /Im1 6 0 R >> >> /Length {} >>\nstream\n",
        form_content.len()
    );
    buf.extend_from_slice(form_hdr.as_bytes());
    buf.extend_from_slice(form_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let img_hdr = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Image /Width 1 /Height 1 \
         /ColorSpace /DeviceCMYK /BitsPerComponent 8 /Length {} >>\nstream\n",
        cmyk.len()
    );
    buf.extend_from_slice(img_hdr.as_bytes());
    buf.extend_from_slice(&cmyk);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let pdf = finalize_pdf(buf, offsets);

    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let k = plate(&plates, "Black");
    assert!(
        sample(k, 50, 50) > 200,
        "K channel of nested-form CMYK image reaches the Black plate; got {}",
        sample(k, 50, 50)
    );
}

#[test]
fn separation_image_routes_to_spot_plate() {
    // 1×1 Separation /Pantone-185 image at full tint.
    // /ColorSpace is declared at page-level as /CS1 → indirect ref to
    // [/Separation /Pantone-185 /DeviceCMYK <tint transform>].
    let content = b"q\n50 0 0 50 25 25 cm\n/Im1 Do\nQ\n";
    let samples: Vec<u8> = vec![255];

    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R \
           /Resources << /XObject << /Im1 5 0 R >> \
                        /ColorSpace << /CS1 6 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let img_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width 1 /Height 1 \
         /ColorSpace /CS1 /BitsPerComponent 8 /Length {} >>\nstream\n",
        samples.len()
    );
    buf.extend_from_slice(img_hdr.as_bytes());
    buf.extend_from_slice(&samples);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"6 0 obj\n[/Separation /Pantone-185 /DeviceCMYK 7 0 R]\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"7 0 obj\n<< /FunctionType 2 /Domain [0 1] /N 1 \
            /C0 [0 0 0 0] /C1 [0 0.85 0.45 0] >>\nendobj\n",
    );
    let pdf = finalize_pdf(buf, offsets);

    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let pantone = plate(&plates, "Pantone-185");
    assert!(
        sample(pantone, 50, 50) > 200,
        "Separation image lands on its named plate; got {}",
        sample(pantone, 50, 50)
    );
    // Process plates receive no ink from the spot image (no OPM, no
    // knockout — see commit message for the overprint-deferred scope).
    let cyan = plate(&plates, "Cyan");
    assert_eq!(sample(cyan, 50, 50), 0, "Spot image leaves process plates untouched");
}

// =========================================================================
// Reviewer feedback follow-ups
// =========================================================================

/// Build a single-page PDF with a 16-bpc DeviceCMYK image.
///
/// `samples` is interleaved big-endian 16-bit CMYK
/// (`width * height * 4 * 2` bytes).
fn build_pdf_with_16bpc_cmyk_image(samples: &[u8], width: u32, height: u32) -> Vec<u8> {
    let content = b"q\n50 0 0 50 25 25 cm\n/Im1 Do\nQ\n";
    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R /Resources << /XObject << /Im1 5 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let img_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width {w} /Height {h} \
         /ColorSpace /DeviceCMYK /BitsPerComponent 16 /Length {len} >>\nstream\n",
        w = width,
        h = height,
        len = samples.len()
    );
    buf.extend_from_slice(img_hdr.as_bytes());
    buf.extend_from_slice(samples);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    finalize_pdf(buf, offsets)
}

/// §8.9.5: BitsPerComponent ∈ {1, 2, 4, 8, 16}. The current routing path
/// reads 8-bit interleaved samples. A 16-bpc image must not be mis-read
/// as 8-bpc and paint garbage onto plates — until full BPC expansion lands,
/// the routing path skips with a log entry, leaving plates untouched.
#[test]
fn non_8bpc_image_is_skipped_not_mis_routed() {
    // 2×2 16-bpc image. All four pixels have C = 0xFFFF, M=Y=K=0. If the
    // renderer mis-reads as 8-bpc it would paint nonzero values on Magenta
    // and Yellow (reading the high byte of each 16-bit sample as a separate
    // pixel/channel). With the BPC guard, every plate stays at 0.
    let mut samples = Vec::with_capacity(2 * 2 * 4 * 2);
    for _ in 0..(2 * 2) {
        samples.extend_from_slice(&[0xFFu8, 0xFFu8]); // C high, low
        samples.extend_from_slice(&[0x00u8, 0x00u8]); // M
        samples.extend_from_slice(&[0x00u8, 0x00u8]); // Y
        samples.extend_from_slice(&[0x00u8, 0x00u8]); // K
    }
    let doc =
        PdfDocument::from_bytes(build_pdf_with_16bpc_cmyk_image(&samples, 2, 2)).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    // M / Y / K plates must stay clean even though the 8-bpc misread would
    // pick up nonzero bytes from the 16-bit C channel.
    for ink in ["Magenta", "Yellow", "Black"] {
        let p = plate(&plates, ink);
        assert_eq!(
            sample(p, 50, 50),
            0,
            "16-bpc CMYK image must not paint stray ink on the {ink} plate; \
             got {}",
            sample(p, 50, 50)
        );
    }
}

/// Build a single-page PDF with a Separation image carrying a `/Decode`
/// array. `decode` is two floats representing the [dmin, dmax] mapping
/// applied per channel.
fn build_pdf_with_separation_image_decode(
    samples: &[u8],
    width: u32,
    height: u32,
    decode: [f32; 2],
) -> Vec<u8> {
    let content = b"q\n50 0 0 50 25 25 cm\n/Im1 Do\nQ\n";
    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R \
           /Resources << /XObject << /Im1 5 0 R >> \
                        /ColorSpace << /CS1 6 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let img_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width {w} /Height {h} \
         /ColorSpace /CS1 /BitsPerComponent 8 \
         /Decode [{d0} {d1}] /Length {len} >>\nstream\n",
        w = width,
        h = height,
        d0 = decode[0],
        d1 = decode[1],
        len = samples.len()
    );
    buf.extend_from_slice(img_hdr.as_bytes());
    buf.extend_from_slice(samples);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"6 0 obj\n[/Separation /Pantone-185 /DeviceCMYK 7 0 R]\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"7 0 obj\n<< /FunctionType 2 /Domain [0 1] /N 1 \
            /C0 [0 0 0 0] /C1 [0 0.85 0.45 0] >>\nendobj\n",
    );
    finalize_pdf(buf, offsets)
}

/// §8.9.5.2: `/Decode [1 0]` inverts the sample-to-tint mapping. With raw
/// sample 0 and `/Decode [1 0]`, the decoded tint is 1.0 (full ink).
#[test]
fn separation_image_decode_array_inverts_routing() {
    let samples = vec![0u8]; // 1×1 raw sample of 0; under /Decode [1 0] → full tint
    let doc =
        PdfDocument::from_bytes(build_pdf_with_separation_image_decode(&samples, 1, 1, [1.0, 0.0]))
            .expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let pantone = plate(&plates, "Pantone-185");
    assert!(
        sample(pantone, 50, 50) > 200,
        "Inverted Separation image: raw 0 + /Decode [1 0] → full tint on the spot plate; got {}",
        sample(pantone, 50, 50)
    );
}

/// §8.9.6.2: ImageMask is a 1-bpc stencil. Sample value 0 marks the page
/// with the current non-stroking colour; value 1 leaves the pixel
/// transparent. With a Separation fill colour, the named spot plate must
/// show the marked area and other plates must stay untouched.
#[test]
fn image_mask_painted_with_spot_colour_routes_to_spot_plate() {
    // 8×8 stencil, all 0s. Per §8.9.6.2 every pixel is "painted" (alpha=255)
    // and the spot fill colour is composited through it. Packed 1-bit row
    // form is 1 byte per row = 0x00, 8 rows = 8 bytes.
    let stencil: Vec<u8> = vec![0x00; 8];
    // Set non-stroking colour space to /CS1 (Separation), tint 1.0.
    let content = b"q\n/CS1 cs\n1 scn\n50 0 0 50 25 25 cm\n/Im1 Do\nQ\n";

    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R \
           /Resources << /XObject << /Im1 5 0 R >> \
                        /ColorSpace << /CS1 6 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let img_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ImageMask true /BitsPerComponent 1 /Length {} >>\nstream\n",
        stencil.len()
    );
    buf.extend_from_slice(img_hdr.as_bytes());
    buf.extend_from_slice(&stencil);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"6 0 obj\n[/Separation /Pantone-185 /DeviceCMYK 7 0 R]\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"7 0 obj\n<< /FunctionType 2 /Domain [0 1] /N 1 \
            /C0 [0 0 0 0] /C1 [0 0.85 0.45 0] >>\nendobj\n",
    );
    let pdf = finalize_pdf(buf, offsets);

    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let pantone = plate(&plates, "Pantone-185");
    assert!(
        sample(pantone, 50, 50) > 200,
        "ImageMask painted with Separation colour reaches the spot plate; got {}",
        sample(pantone, 50, 50)
    );
    let cyan = plate(&plates, "Cyan");
    assert_eq!(
        sample(cyan, 50, 50),
        0,
        "ImageMask with spot fill does not paint the process plates"
    );
}

/// DeviceN with two named spot inks: each channel routes to its named plate.
#[test]
fn devicen_image_routes_each_channel_to_its_named_plate() {
    // 2×1 image, 2 channels (SpotRed first, SpotBlue second). Interleaved
    // samples: pixel 0 = [255, 0] (full SpotRed, no SpotBlue),
    //          pixel 1 = [0, 255] (no SpotRed, full SpotBlue).
    let samples: Vec<u8> = vec![255, 0, 0, 255];
    let content = b"q\n50 0 0 50 25 25 cm\n/Im1 Do\nQ\n";

    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R \
           /Resources << /XObject << /Im1 5 0 R >> \
                        /ColorSpace << /CS1 6 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let img_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width 2 /Height 1 \
         /ColorSpace /CS1 /BitsPerComponent 8 /Length {} >>\nstream\n",
        samples.len()
    );
    buf.extend_from_slice(img_hdr.as_bytes());
    buf.extend_from_slice(&samples);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"6 0 obj\n[/DeviceN [/SpotRed /SpotBlue] /DeviceCMYK 7 0 R]\nendobj\n");
    offsets.push(buf.len());
    // Identity-ish tint transform: returns CMYK = (0, c0, c1, 0). For plate
    // routing the tint transform is not consulted (per the per-plate
    // convention), so the function body doesn't matter as long as it's a
    // valid PDF function.
    buf.extend_from_slice(
        b"7 0 obj\n<< /FunctionType 2 /Domain [0 1 0 1] /N 1 \
            /C0 [0 0 0 0] /C1 [0 1 1 0] >>\nendobj\n",
    );
    let pdf = finalize_pdf(buf, offsets);

    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    // The image occupies (25..75, 25..75) in PDF user space. Image col 0
    // (SpotRed-only) lands in plate cols 25..50; image col 1 (SpotBlue-only)
    // lands in plate cols 50..75. Sample 20 px in from the boundary so the
    // bilinear filter has no measurable bleed across it.
    let red = plate(&plates, "SpotRed");
    let blue = plate(&plates, "SpotBlue");
    assert!(
        sample(red, 30, 50) > 200,
        "DeviceN channel 0 lands on SpotRed plate (left half); got {}",
        sample(red, 30, 50)
    );
    assert!(
        sample(blue, 70, 50) > 200,
        "DeviceN channel 1 lands on SpotBlue plate (right half); got {}",
        sample(blue, 70, 50)
    );
    assert!(
        sample(red, 70, 50) < 16,
        "SpotRed plate does not pick up channel-1 ink in the right half; got {}",
        sample(red, 70, 50)
    );
    assert!(
        sample(blue, 30, 50) < 16,
        "SpotBlue plate does not pick up channel-0 ink in the left half; got {}",
        sample(blue, 30, 50)
    );
}

/// Encode an 8×8 CMYK pixel buffer as a JPEG via `jpeg-encoder`'s
/// `ColorType::Cmyk` path, which inverts the samples and writes the
/// Adobe APP14 marker with `color_transform = 0`.  Round-tripping through
/// `decode_cmyk_jpeg_to_raw_cmyk` should restore the original samples.
fn encode_cmyk_jpeg(cmyk: &[u8], width: u16, height: u16) -> Vec<u8> {
    use jpeg_encoder::{ColorType, Encoder};
    let mut out = Vec::new();
    let encoder = Encoder::new(&mut out, 95);
    encoder
        .encode(cmyk, width, height, ColorType::Cmyk)
        .expect("encode CMYK JPEG");
    out
}

/// Build a single-page PDF with a JPEG-encoded DeviceCMYK image.
fn build_pdf_with_cmyk_jpeg(jpeg: &[u8], width: u32, height: u32) -> Vec<u8> {
    let content = b"q\n50 0 0 50 25 25 cm\n/Im1 Do\nQ\n";
    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R /Resources << /XObject << /Im1 5 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let img_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width {w} /Height {h} \
         /ColorSpace /DeviceCMYK /BitsPerComponent 8 \
         /Filter /DCTDecode /Length {len} >>\nstream\n",
        w = width,
        h = height,
        len = jpeg.len()
    );
    buf.extend_from_slice(img_hdr.as_bytes());
    buf.extend_from_slice(jpeg);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    finalize_pdf(buf, offsets)
}

/// CMYK JPEG with Adobe APP14 marker (color_transform = 0): the JPEG file
/// stores `255 - sample` per channel, and the renderer must invert on
/// decode (per `decode_cmyk_jpeg_to_raw_cmyk`). Without the inversion,
/// a pure-cyan input image would render with M / Y / K instead of C.
#[test]
fn cmyk_jpeg_app14_inversion_round_trips_to_correct_plate() {
    // 8×8 pure-cyan CMYK image: C=255, M=Y=K=0 everywhere.
    let mut cmyk = Vec::with_capacity(8 * 8 * 4);
    for _ in 0..(8 * 8) {
        cmyk.extend_from_slice(&[255, 0, 0, 0]);
    }
    let jpeg = encode_cmyk_jpeg(&cmyk, 8, 8);
    let doc = PdfDocument::from_bytes(build_pdf_with_cmyk_jpeg(&jpeg, 8, 8)).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let cyan_v = sample(plate(&plates, "Cyan"), 50, 50);
    let magenta_v = sample(plate(&plates, "Magenta"), 50, 50);
    let yellow_v = sample(plate(&plates, "Yellow"), 50, 50);
    let black_v = sample(plate(&plates, "Black"), 50, 50);

    // After APP14 inversion the Cyan plate should be high (lossy JPEG keeps
    // it ≳ 200). Without inversion the bytes would read M = Y = K ≈ 255 and
    // Cyan ≈ 0 — exactly the failure mode this test guards against.
    assert!(
        cyan_v > 200,
        "Cyan plate after APP14 inversion; got {cyan_v} (M={magenta_v} Y={yellow_v} K={black_v})"
    );
    assert!(
        magenta_v < 50 && yellow_v < 50 && black_v < 50,
        "M / Y / K plates must stay near zero for a pure-cyan source after APP14 \
         inversion; got M={magenta_v} Y={yellow_v} K={black_v}"
    );
}
