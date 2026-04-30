use pdf_oxide::editor::{DocumentEditor, EditableDocument, SaveOptions};
use std::path::Path;

pub fn run(
    file: &Path,
    margins: &str,
    pages: Option<&str>,
    output: Option<&Path>,
    password: Option<&str>,
) -> pdf_oxide::Result<()> {
    // Parse margins: left,right,top,bottom
    let parts: Vec<f32> = margins
        .split(',')
        .map(|s| {
            s.trim().parse::<f32>().map_err(|_| {
                pdf_oxide::Error::InvalidOperation(format!("Invalid margin value: '{}'", s.trim()))
            })
        })
        .collect::<pdf_oxide::Result<Vec<_>>>()?;

    if parts.len() != 4 {
        return Err(pdf_oxide::Error::InvalidOperation(
            "Margins must be left,right,top,bottom (e.g. '50,50,50,50')".to_string(),
        ));
    }

    let (left, right, top, bottom) = (parts[0], parts[1], parts[2], parts[3]);

    let doc = super::open_doc(file, password)?;
    let page_count = doc.page_count()?;
    drop(doc);

    let page_indices = super::resolve_pages(pages, page_count)?;

    let mut editor = DocumentEditor::open(file)?;

    // Apply crop to selected pages using per-page crop box
    for &idx in &page_indices {
        let media_box = editor.get_page_media_box(idx)?;
        let crop_box = [
            media_box[0] + left,
            media_box[1] + bottom,
            media_box[2] - right,
            media_box[3] - top,
        ];
        editor.set_page_crop_box(idx, crop_box)?;
    }

    let out_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| super::output_beside(file, "_cropped.pdf"));

    editor.save_with_options(
        &out_path,
        SaveOptions {
            compress: true,
            garbage_collect: true,
            ..Default::default()
        },
    )?;

    eprintln!(
        "Cropped {} page(s) (margins: l={left}, r={right}, t={top}, b={bottom}) → {}",
        page_indices.len(),
        out_path.display()
    );

    Ok(())
}
