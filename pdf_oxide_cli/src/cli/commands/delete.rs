use pdf_oxide::editor::{DocumentEditor, EditableDocument, SaveOptions};
use std::path::Path;

pub fn run(
    file: &Path,
    pages: Option<&str>,
    output: Option<&Path>,
    password: Option<&str>,
) -> pdf_oxide::Result<()> {
    let pages = pages.ok_or_else(|| {
        pdf_oxide::Error::InvalidOperation(
            "--pages is required for delete (e.g. --pages 2,5-7)".to_string(),
        )
    })?;

    let doc = super::open_doc(file, password)?;
    let page_count = doc.page_count()?;
    drop(doc);

    let page_indices = super::resolve_pages(Some(pages), page_count)?;

    if page_indices.len() >= page_count {
        return Err(pdf_oxide::Error::InvalidOperation(
            "Cannot delete all pages from a PDF".to_string(),
        ));
    }

    let mut editor = DocumentEditor::open(file)?;

    // Remove pages from end to start to keep indices stable
    let mut sorted = page_indices.clone();
    sorted.sort_unstable();
    sorted.dedup();
    for &idx in sorted.iter().rev() {
        editor.remove_page(idx)?;
    }

    let out_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| super::output_beside(file, "_trimmed.pdf"));

    editor.save_with_options(
        &out_path,
        SaveOptions {
            compress: true,
            garbage_collect: true,
            ..Default::default()
        },
    )?;

    eprintln!("Deleted {} page(s) → {}", sorted.len(), out_path.display());

    Ok(())
}
