// Barcode SVG generation
//
// Demonstrates generating 1D barcodes and QR codes as scalable SVG strings
// (vector output, no rasterisation). Requires the `barcodes` feature.
//
// Run: cargo run --example showcase_barcode_svg --features barcodes

use pdf_oxide::{
    error::Result,
    writer::{BarcodeGenerator, BarcodeOptions, BarcodeType, DataMatrixOptions, QrCodeOptions},
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let out_dir = PathBuf::from("target/examples_output/barcode_svg");
    std::fs::create_dir_all(&out_dir)?;

    // 1D barcode — Code 128 SVG
    let svg = BarcodeGenerator::generate_1d_svg(
        BarcodeType::Code128,
        "PDF-OXIDE-0341",
        &BarcodeOptions::default().width(400).height(80),
    )?;
    assert!(svg.starts_with("<svg"), "expected SVG output for Code128");
    let path = out_dir.join("code128.svg");
    std::fs::write(&path, &svg)?;
    println!("Written: {} ({} bytes)", path.display(), svg.len());

    // 1D barcode — EAN-13 SVG
    let svg = BarcodeGenerator::generate_1d_svg(
        BarcodeType::Ean13,
        "5901234123457",
        &BarcodeOptions::default().width(300).height(80),
    )?;
    assert!(svg.starts_with("<svg"));
    let path = out_dir.join("ean13.svg");
    std::fs::write(&path, &svg)?;
    println!("Written: {} ({} bytes)", path.display(), svg.len());

    // Data Matrix SVG
    let svg = BarcodeGenerator::generate_datamatrix_svg(
        "https://github.com/yfedoseev/pdf_oxide",
        &DataMatrixOptions::default().size(256),
    )?;
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<rect"), "Data Matrix SVG must contain rect elements");
    let path = out_dir.join("datamatrix.svg");
    std::fs::write(&path, &svg)?;
    println!("Written: {} ({} bytes)", path.display(), svg.len());

    // QR code SVG
    let svg = BarcodeGenerator::generate_qr_svg(
        "https://github.com/yfedoseev/pdf_oxide",
        &QrCodeOptions::default().size(256),
    )?;
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<rect"), "QR SVG must contain rect elements");
    let path = out_dir.join("qr_code.svg");
    std::fs::write(&path, &svg)?;
    println!("Written: {} ({} bytes)", path.display(), svg.len());

    println!("All barcode SVG checks passed.");
    Ok(())
}
