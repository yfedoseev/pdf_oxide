//! Batch page-to-PNG renderer for rendering regression tests.
//!
//! Usage:
//!   cargo run --release --features rendering --example render_pages \
//!       -- <input.pdf> <output_dir> [max_pages]
//!
//! Saves page_001.png, page_002.png … to <output_dir>.
//! Writes a SKIP sentinel file for encrypted / corrupt PDFs.
//! Never panics — all errors are printed to stderr and skipped.

#[cfg(feature = "rendering")]
use pdf_oxide::rendering::{render_page, RenderOptions};
#[cfg(feature = "rendering")]
use pdf_oxide::PdfDocument;

#[cfg(feature = "rendering")]
fn run() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: render_pages <input.pdf> <output_dir> [max_pages]");
        return 1;
    }
    let input = &args[1];
    let out_dir = std::path::Path::new(&args[2]);
    let max_pages: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);

    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("ERROR: cannot create {}: {}", out_dir.display(), e);
        return 1;
    }

    let doc = match PdfDocument::open(input) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP {}: {}", input, e);
            let _ = std::fs::write(out_dir.join("SKIP"), format!("{}", e));
            return 0;
        },
    };

    let page_count = match doc.page_count() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("SKIP {} (page_count): {}", input, e);
            let _ = std::fs::write(out_dir.join("SKIP"), format!("{}", e));
            return 0;
        },
    };

    let opts = RenderOptions::with_dpi(150);
    let to_render = page_count.min(max_pages);

    for page_idx in 0..to_render {
        let out_path = out_dir.join(format!("page_{:03}.png", page_idx + 1));
        match render_page(&doc, page_idx, &opts) {
            Ok(img) => {
                if let Err(e) = img.save(&out_path) {
                    eprintln!("WARN: save failed page {} of {}: {}", page_idx + 1, input, e);
                } else {
                    eprintln!("OK  {}", out_path.display());
                }
            },
            Err(e) => {
                eprintln!("WARN: render failed page {} of {}: {}", page_idx + 1, input, e);
            },
        }
    }
    0
}

#[cfg(feature = "rendering")]
fn main() {
    std::process::exit(run());
}

#[cfg(not(feature = "rendering"))]
fn main() {
    eprintln!("Rebuild with --features rendering");
    std::process::exit(1);
}
