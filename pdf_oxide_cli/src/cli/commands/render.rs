use pdf_oxide::rendering::{ImageFormat, RenderOptions};
use std::path::Path;

pub fn run(
    file: &Path,
    dpi: u32,
    format: &str,
    quality: u8,
    pages: Option<&str>,
    output: Option<&Path>,
    password: Option<&str>,
) -> pdf_oxide::Result<()> {
    let doc = super::open_doc(file, password)?;
    let page_count = doc.page_count()?;
    let page_indices = super::resolve_pages(pages, page_count)?;

    let out_dir = output.unwrap_or_else(|| Path::new("."));
    if page_indices.len() > 1 || out_dir.is_dir() {
        std::fs::create_dir_all(out_dir)?;
    }

    let img_format = match format.to_lowercase().as_str() {
        "jpeg" | "jpg" => ImageFormat::Jpeg,
        "png" => ImageFormat::Png,
        other => {
            return Err(pdf_oxide::Error::Unsupported(format!(
                "unsupported format {:?}; use png or jpeg",
                other
            )))
        },
    };

    let mut options = RenderOptions::with_dpi(dpi);
    options.format = img_format;
    options.jpeg_quality = quality;

    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
    let ext = match img_format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpg",
        ImageFormat::RawRgba8 => {
            return Err(pdf_oxide::Error::Unsupported(
                "RawRgba8 is not supported by the CLI renderer; use Png or Jpeg.".into(),
            ));
        },
    };

    for &page_idx in &page_indices {
        let img = pdf_oxide::rendering::render_page(&doc, page_idx, &options)?;

        let out_path = match output {
            Some(out) if page_indices.len() == 1 && !out.is_dir() => out.to_path_buf(),
            _ => out_dir.join(format!("{}_{}.{}", stem, page_idx + 1, ext)),
        };

        img.save(&out_path)?;
        eprintln!("Rendered page {} to {}", page_idx + 1, out_path.display());
    }

    Ok(())
}
