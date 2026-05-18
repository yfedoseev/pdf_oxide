pub mod auto;
pub mod bookmarks;
pub mod classify;
pub mod compress;
pub mod create;
pub mod crop;
pub mod decrypt;
pub mod delete;
pub mod encrypt;
pub mod flatten;
pub mod forms;
pub mod html;
pub mod images;
pub mod info;
pub mod markdown;
pub mod merge;
pub mod metadata;
pub mod models;
pub mod paths;
pub mod redact;
pub mod render;
pub mod reorder;
pub mod rotate;
pub mod search;
pub mod split;
pub mod text;
pub mod watermark;

use pdf_oxide::PdfDocument;
use std::path::{Path, PathBuf};

/// Open a PDF, optionally authenticating with a password.
pub fn open_doc(path: &Path, password: Option<&str>) -> pdf_oxide::Result<PdfDocument> {
    let doc = PdfDocument::open(path)?;
    if let Some(pw) = password {
        doc.authenticate(pw.as_bytes())?;
    }
    Ok(doc)
}

/// Get page indices to process: either from --pages flag or all pages.
pub fn resolve_pages(pages_arg: Option<&str>, page_count: usize) -> pdf_oxide::Result<Vec<usize>> {
    match pages_arg {
        Some(ranges) => {
            super::pages::parse_page_ranges(ranges).map_err(pdf_oxide::Error::InvalidOperation)
        },
        None => Ok((0..page_count).collect()),
    }
}

/// Default output path for a single-file binary command, placed beside the input.
///
/// `suffix` should include both the tag and extension, e.g. `"_watermarked.pdf"`.
/// Result lands in the same directory as `input`, never in cwd. For a bare
/// filename like `"doc.pdf"` with no parent, the result is `"doc_watermarked.pdf"`
/// in cwd (matching the input's implicit location).
pub(super) fn output_beside(input: &Path, suffix: &str) -> PathBuf {
    let dir = input
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    dir.join(format!("{stem}{suffix}"))
}

/// Default output directory for a command that writes multiple files (e.g. split).
///
/// Returns the parent directory of `input`, or `"."` for bare filenames.
pub(super) fn output_dir_beside(input: &Path) -> PathBuf {
    input
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

/// Write output to file or stdout.
pub fn write_output(content: &str, output: Option<&Path>) -> pdf_oxide::Result<()> {
    use std::io::Write;
    match output {
        Some(path) => Ok(std::fs::write(path, content)?),
        None => {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(content.as_bytes())?;
            // Ensure trailing newline for terminal
            if !content.ends_with('\n') {
                handle.write_all(b"\n")?;
            }
            Ok(())
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_beside_subdir() {
        let p = output_beside(Path::new("/some/dir/doc.pdf"), "_watermarked.pdf");
        assert_eq!(p, PathBuf::from("/some/dir/doc_watermarked.pdf"));
    }

    #[test]
    fn output_beside_bare_filename() {
        let p = output_beside(Path::new("doc.pdf"), "_compressed.pdf");
        assert_eq!(p, PathBuf::from("./doc_compressed.pdf"));
    }

    #[test]
    fn output_beside_root_level() {
        let p = output_beside(Path::new("/doc.pdf"), "_rotated.pdf");
        assert_eq!(p, PathBuf::from("/doc_rotated.pdf"));
    }

    #[test]
    fn output_dir_beside_subdir() {
        let p = output_dir_beside(Path::new("/some/dir/doc.pdf"));
        assert_eq!(p, PathBuf::from("/some/dir"));
    }

    #[test]
    fn output_dir_beside_bare_filename() {
        let p = output_dir_beside(Path::new("doc.pdf"));
        assert_eq!(p, PathBuf::from("."));
    }
}
