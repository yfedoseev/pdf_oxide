use pdf_oxide::editor::{DocumentEditor, EditableDocument};
use std::path::Path;

pub fn run(files: &[std::path::PathBuf], output: Option<&Path>) -> pdf_oxide::Result<()> {
    if files.len() < 2 {
        return Err(pdf_oxide::Error::InvalidOperation(
            "Merge requires at least 2 PDF files".to_string(),
        ));
    }

    // Validate output flag up front so we fail before opening any file.
    let out_path = output.ok_or_else(|| {
        pdf_oxide::Error::InvalidOperation(
            "Merge requires -o/--output to specify the destination path \
             (e.g. -o merged.pdf). There is no single input file to anchor a default output to."
                .to_string(),
        )
    })?;

    let mut editor = DocumentEditor::open(&files[0])?;

    for source in &files[1..] {
        let pages_added = editor.merge_from(source)?;
        eprintln!("Merged {} pages from {}", pages_added, source.display());
    }

    editor.save(out_path)?;
    eprintln!("Saved to {}", out_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn run_without_output_returns_invalid_operation() {
        let files = vec![PathBuf::from("a.pdf"), PathBuf::from("b.pdf")];
        let err = run(&files, None).expect_err("merge with no -o should fail");
        match err {
            pdf_oxide::Error::InvalidOperation(msg) => {
                assert!(
                    msg.contains("-o") || msg.contains("--output"),
                    "error should name the missing flag, got: {msg}"
                );
            },
            other => panic!("expected InvalidOperation, got {other:?}"),
        }
    }
}
