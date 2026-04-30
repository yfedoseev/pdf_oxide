use pdf_oxide::editor::{DocumentEditor, EditableDocument, SaveOptions};
use std::path::Path;

pub fn run(file: &Path, _password: &str, output: Option<&Path>) -> pdf_oxide::Result<()> {
    let mut editor = DocumentEditor::open(file)?;

    let out_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| super::output_beside(file, "_decrypted.pdf"));

    editor.save_with_options(
        &out_path,
        SaveOptions {
            compress: true,
            garbage_collect: true,
            ..Default::default()
        },
    )?;

    eprintln!("Decrypted {} -> {}", file.display(), out_path.display());

    Ok(())
}
