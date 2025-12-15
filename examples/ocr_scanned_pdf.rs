//! OCR text extraction from scanned PDFs.
//!
//! This example demonstrates how to extract text from scanned PDFs using
//! PaddleOCR PP-OCRv5 models via ONNX Runtime.
//!
//! # Prerequisites
//!
//! Download the PaddleOCR models:
//! - `en_PP-OCRv5_det_infer.onnx` - Text detection model
//! - `en_PP-OCRv5_rec_infer.onnx` - Text recognition model
//! - `en_dict.txt` - Character dictionary
//!
//! # Usage
//!
//! ```bash
//! cargo run --features ocr --example ocr_scanned_pdf -- \
//!     --pdf scanned.pdf \
//!     --det models/en_PP-OCRv5_det_infer.onnx \
//!     --rec models/en_PP-OCRv5_rec_infer.onnx \
//!     --dict models/en_dict.txt
//! ```

#[cfg(feature = "ocr")]
use pdf_oxide::document::PdfDocument;
#[cfg(feature = "ocr")]
use pdf_oxide::ocr::{self, OcrConfigBuilder, OcrEngine, OcrExtractOptions};
#[cfg(feature = "ocr")]
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "ocr"))]
    {
        eprintln!("This example requires the 'ocr' feature to be enabled.");
        eprintln!("Run with: cargo run --features ocr --example ocr_scanned_pdf");
        Err("OCR feature not enabled".into())
    }

    #[cfg(feature = "ocr")]
    {
        run_ocr()
    }
}

#[cfg(feature = "ocr")]
fn run_ocr() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    // Simple argument parsing
    let mut pdf_path = None;
    let mut det_model = None;
    let mut rec_model = None;
    let mut dict_path = None;
    let mut dpi = 300.0f32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pdf" => {
                pdf_path = Some(args.get(i + 1).cloned().ok_or("Missing --pdf value")?);
                i += 2;
            },
            "--det" => {
                det_model = Some(args.get(i + 1).cloned().ok_or("Missing --det value")?);
                i += 2;
            },
            "--rec" => {
                rec_model = Some(args.get(i + 1).cloned().ok_or("Missing --rec value")?);
                i += 2;
            },
            "--dict" => {
                dict_path = Some(args.get(i + 1).cloned().ok_or("Missing --dict value")?);
                i += 2;
            },
            "--dpi" => {
                dpi = args
                    .get(i + 1)
                    .ok_or("Missing --dpi value")?
                    .parse()
                    .map_err(|_| "Invalid --dpi value")?;
                i += 2;
            },
            "--help" | "-h" => {
                print_usage(&args[0]);
                return Ok(());
            },
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage(&args[0]);
                std::process::exit(1);
            },
        }
    }

    let pdf_path = pdf_path.ok_or("Missing required --pdf argument")?;
    let det_model = det_model.ok_or("Missing required --det argument")?;
    let rec_model = rec_model.ok_or("Missing required --rec argument")?;
    let dict_path = dict_path.ok_or("Missing required --dict argument")?;

    // Configure OCR
    let config = OcrConfigBuilder::new()
        .det_threshold(0.3)
        .box_threshold(0.5)
        .rec_threshold(0.5)
        .num_threads(4)
        .build();

    println!("Loading OCR models...");
    let engine = OcrEngine::new(&det_model, &rec_model, &dict_path, config)?;
    println!("Models loaded successfully.");

    // Open PDF
    println!("Opening PDF: {}", pdf_path);
    let mut doc = PdfDocument::open(&pdf_path)?;
    let page_count = doc.page_count()?;
    println!("PDF has {} pages", page_count);

    // Process each page
    let options = OcrExtractOptions::with_dpi(dpi);

    for page_idx in 0..page_count {
        println!("\n=== Page {} ===", page_idx + 1);

        // Check if page needs OCR
        let needs_ocr = ocr::needs_ocr(&mut doc, page_idx)?;

        if needs_ocr {
            println!("Page is scanned, running OCR...");
            let text = ocr::ocr_page(&mut doc, page_idx, &engine, &options)?;
            println!("{}", text);
        } else {
            println!("Page has native text, using standard extraction...");
            let text = doc.extract_text(page_idx)?;
            println!("{}", text);
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn print_usage(program: &str) {
    eprintln!(
        r#"OCR Scanned PDF - Extract text from scanned PDFs using PaddleOCR

Usage: {} [OPTIONS]

Required arguments:
    --pdf <PATH>     Path to the PDF file
    --det <PATH>     Path to detection model (ONNX)
    --rec <PATH>     Path to recognition model (ONNX)
    --dict <PATH>    Path to character dictionary

Optional arguments:
    --dpi <NUMBER>   DPI for coordinate conversion (default: 300)
    --help, -h       Show this help message

Example:
    {} --pdf scanned.pdf --det det.onnx --rec rec.onnx --dict en_dict.txt
"#,
        program, program
    );
}
