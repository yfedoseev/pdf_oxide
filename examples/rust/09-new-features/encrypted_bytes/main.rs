// Encrypted PDF output (Rust-idiomatic; uses DocumentBuilder::to_bytes_encrypted).
//
// Run: cargo run --example showcase_encrypted_bytes

use pdf_oxide::error::Result;
use pdf_oxide::writer::DocumentBuilder;
use std::path::PathBuf;

fn main() -> Result<()> {
    let out_dir = PathBuf::from("target/examples_output/encrypted_bytes");
    std::fs::create_dir_all(&out_dir)?;

    let mut builder = DocumentBuilder::new();
    builder
        .letter_page()
        .font("Helvetica", 12.0)
        .at(72.0, 720.0)
        .heading(1, "Encrypted PDF")
        .at(72.0, 690.0)
        .paragraph("This PDF is encrypted with a user and owner password.")
        .done();

    // AES-256 encryption with user + owner passwords (ISO 32000-1 §7.6).
    let encrypted = builder.to_bytes_encrypted("user123", "owner123")?;
    assert!(encrypted.starts_with(b"%PDF"), "encrypted output must start with %PDF");

    let path = out_dir.join("encrypted.pdf");
    std::fs::write(&path, &encrypted)?;
    println!("Wrote {} ({} bytes, encrypted)", path.display(), encrypted.len());
    Ok(())
}
