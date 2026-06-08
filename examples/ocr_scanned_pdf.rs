//! OCR text extraction from scanned PDFs.
//!
//! This example demonstrates how to extract text from scanned PDFs using
//! PaddleOCR models via ONNX Runtime.
//!
//! # Model Setup
//!
//! Download models with: `./scripts/setup_ocr_models.sh`
//!
//! Or manually download the recommended combination (V4 det + V5 rec):
//! - Detection: ch_PP-OCRv4_det from https://huggingface.co/deepghs/paddleocr
//! - Recognition: en_PP-OCRv5_mobile_rec from https://huggingface.co/monkt/paddleocr-onnx
//! - Dictionary: PP-OCRv5 English dict (add space as last line)
//!
//! # Usage
//!
//! ```bash
//! cargo run --features ocr --example ocr_scanned_pdf -- \
//!     --pdf scanned.pdf \
//!     --det .models/det.onnx \
//!     --rec .models/rec.onnx \
//!     --dict .models/en_dict.txt
//! ```
//!
//! For PP-OCRv5 full stack (v5 detection + v5 recognition), add `--v5`:
//!
//! ```bash
//! cargo run --features ocr --example ocr_scanned_pdf -- \
//!     --pdf scanned.pdf \
//!     --det .models/v5/det.onnx \
//!     --rec .models/v5/rec.onnx \
//!     --dict .models/v5/en_dict.txt \
//!     --v5
//! ```

#[cfg(feature = "ocr")]
use pdf_oxide::document::PdfDocument;
#[cfg(feature = "ocr")]
use pdf_oxide::ocr::{self, OcrConfig, OcrEngine, OcrExtractOptions};
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
    let mut use_v5 = false;

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
            "--v5" => {
                use_v5 = true;
                i += 1;
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
    let config = if use_v5 {
        println!("Using PP-OCRv5 config (high-resolution detection input)");
        OcrConfig::v5()
    } else {
        OcrConfig::default()
    };

    println!("Loading OCR models...");
    let engine = OcrEngine::new(&det_model, &rec_model, &dict_path, config)?;
    println!("Models loaded successfully.");

    // Open PDF
    println!("Opening PDF: {}", pdf_path);
    let doc = PdfDocument::open(&pdf_path)?;
    let page_count = doc.page_count()?;
    println!("PDF has {} pages", page_count);

    // Process each page
    let options = OcrExtractOptions::with_dpi(dpi);

    for page_idx in 0..page_count {
        println!("\n=== Page {} ===", page_idx + 1);

        // Check if page needs OCR
        let needs_ocr = ocr::needs_ocr(&doc, page_idx)?;

        if needs_ocr {
            println!("Page is scanned, running OCR...");
            let text = ocr::ocr_page(&doc, page_idx, &engine, &options)?;
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
    --v5             Use PP-OCRv5 config (high-res detection input)
    --help, -h       Show this help message

Model recommendations:
    Best quality:  V4 detection + V5 recognition (default config)
    Full V5 stack: V5 detection + V5 recognition (use --v5 flag)

Example (recommended V4 det + V5 rec):
    {} --pdf scanned.pdf --det .models/det.onnx --rec .models/rec.onnx --dict .models/en_dict.txt

Example (full V5):
    {} --pdf scanned.pdf --det .models/v5/det.onnx --rec .models/v5/rec.onnx --dict .models/v5/en_dict.txt --v5
"#,
        program, program, program
    );
}
