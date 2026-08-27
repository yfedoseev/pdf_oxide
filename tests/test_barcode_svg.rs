// Regression test for #421: pdf_barcode_get_svg was a stub returning ERR_UNSUPPORTED.
// This test verifies the fix on HEAD (release/v0.3.43).

#[cfg(feature = "barcodes")]
#[test]
fn test_barcode_svg_code128() {
    use pdf_oxide::writer::{BarcodeGenerator, BarcodeOptions, BarcodeType};
    let svg = BarcodeGenerator::generate_1d_svg(
        BarcodeType::Code128,
        "TEST-421",
        &BarcodeOptions::default(),
    )
    .expect("generate_1d_svg must succeed");

    assert!(svg.starts_with("<svg"), "must be SVG, got: {}", &svg[..svg.len().min(40)]);
    assert!(svg.contains("<rect"), "Code128 SVG must contain rect bar elements");
    assert!(!svg.is_empty());
    println!("Code128 SVG length: {} bytes", svg.len());
}

#[cfg(feature = "barcodes")]
#[test]
fn test_barcode_svg_qr() {
    use pdf_oxide::writer::{BarcodeGenerator, QrCodeOptions};
    let svg = BarcodeGenerator::generate_qr_svg(
        "https://example.com",
        &QrCodeOptions::default().size(256),
    )
    .expect("generate_qr_svg must succeed");

    assert!(svg.starts_with("<svg"), "must be SVG, got: {}", &svg[..svg.len().min(40)]);
    assert!(svg.contains("<rect"), "QR SVG must contain rect module elements");
    assert!(svg.contains("viewBox"), "QR SVG must have viewBox");
    println!("QR SVG length: {} bytes", svg.len());
}

#[cfg(feature = "barcodes")]
#[test]
fn test_barcode_svg_datamatrix() {
    use pdf_oxide::writer::{BarcodeGenerator, DmtxCodeOptions};
    let svg = BarcodeGenerator::generate_datamatrix_svg(
        "https://example.com",
        &DataMatrixOptions::default().size(256),
    )
    .expect("generate_datamatrix_svg must succeed");

    assert!(svg.starts_with("<svg"), "must be SVG, got: {}", &svg[..svg.len().min(40)]);
    assert!(svg.contains("<rect"), "QR SVG must contain rect module elements");
    assert!(svg.contains("viewBox"), "QR SVG must have viewBox");
    println!("QR SVG length: {} bytes", svg.len());
}

#[cfg(feature = "barcodes")]
#[test]
fn test_qr_format_sentinel_no_collision() {
    // Before fix: QR used format=0, same as Code128 — indistinguishable.
    // After fix: QR uses format=100 (sentinel outside 0-7 range).
    use pdf_oxide::writer::{BarcodeGenerator, BarcodeOptions, BarcodeType, QrCodeOptions};

    // Code128 SVG — short horizontal bars, no 2D matrix
    let svg_1d =
        BarcodeGenerator::generate_1d_svg(BarcodeType::Code128, "ABC", &BarcodeOptions::default())
            .unwrap();

    // QR SVG — 2D matrix, much larger rect count
    let svg_qr =
        BarcodeGenerator::generate_qr_svg("ABC", &QrCodeOptions::default().size(256)).unwrap();

    // They must be structurally different (QR has 2D rects with varying y coordinates)
    assert_ne!(svg_1d, svg_qr, "1D and QR SVGs must be distinct");

    // QR rects span multiple y coordinates; 1D rects all have y="0"
    let qr_y_values: Vec<&str> = svg_qr
        .split("y=\"")
        .skip(1)
        .map(|s| s.split('"').next().unwrap_or(""))
        .collect();
    let distinct_y: std::collections::HashSet<&str> = qr_y_values.iter().copied().collect();
    assert!(
        distinct_y.len() > 3,
        "QR must have multiple distinct y values (2D grid), got {:?}",
        distinct_y.len()
    );

    let all_y_zero = svg_1d
        .split("y=\"")
        .skip(1)
        .map(|s| s.split('"').next().unwrap_or(""))
        .all(|y| y == "0");
    assert!(all_y_zero, "Code128 bars must all have y=0 (single-row barcode)");
}
