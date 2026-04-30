use pdf_oxide::editor::{DocumentEditor, EditableDocument, SaveOptions};
use std::path::Path;

pub fn run(
    file: &Path,
    order: &str,
    output: Option<&Path>,
    password: Option<&str>,
) -> pdf_oxide::Result<()> {
    let doc = super::open_doc(file, password)?;
    let page_count = doc.page_count()?;
    drop(doc);

    // Parse order: comma-separated 1-indexed page numbers
    let indices: Vec<usize> = order
        .split(',')
        .map(|s| {
            let s = s.trim();
            s.parse::<usize>()
                .map_err(|_| {
                    pdf_oxide::Error::InvalidOperation(format!("Invalid page number: '{s}'"))
                })
                .and_then(|n| {
                    if n == 0 || n > page_count {
                        Err(pdf_oxide::Error::InvalidOperation(format!(
                            "Page {n} out of range (1-{page_count})"
                        )))
                    } else {
                        Ok(n - 1) // Convert to 0-indexed
                    }
                })
        })
        .collect::<pdf_oxide::Result<Vec<_>>>()?;

    if indices.len() != page_count {
        return Err(pdf_oxide::Error::InvalidOperation(format!(
            "Order must list all {page_count} pages (got {})",
            indices.len()
        )));
    }

    // Verify every page is listed exactly once
    let mut seen = vec![false; page_count];
    for &idx in &indices {
        if seen[idx] {
            return Err(pdf_oxide::Error::InvalidOperation(format!(
                "Page {} listed more than once",
                idx + 1
            )));
        }
        seen[idx] = true;
    }

    // Build new document by merging pages in specified order
    let mut editor = DocumentEditor::open(file)?;
    editor.merge_pages_from(file, &indices)?;

    // Remove the original pages (they are at indices 0..page_count)
    for i in (0..page_count).rev() {
        editor.remove_page(i)?;
    }

    let out_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| super::output_beside(file, "_reordered.pdf"));

    editor.save_with_options(
        &out_path,
        SaveOptions {
            compress: true,
            garbage_collect: true,
            ..Default::default()
        },
    )?;

    eprintln!("Reordered {page_count} pages → {}", out_path.display());

    Ok(())
}
