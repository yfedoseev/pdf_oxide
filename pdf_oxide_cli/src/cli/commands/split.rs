use pdf_oxide::editor::{DocumentEditor, EditableDocument, SaveOptions};
use std::path::Path;

#[allow(clippy::too_many_arguments)] // internal CLI command mirroring clap args
pub fn run(
    file: &Path,
    pages: Option<&str>,
    output: Option<&Path>,
    password: Option<&str>,
    by_bookmarks: bool,
    bookmark_prefix: Option<&str>,
    bookmark_level: u32,
    ignore_case: bool,
    no_front_matter: bool,
) -> pdf_oxide::Result<()> {
    // #482: split at outline/bookmark boundaries instead of per-page.
    // Backward compatible — without --by-bookmarks the legacy per-page
    // path below is byte-for-byte unchanged.
    if by_bookmarks {
        use pdf_oxide::split_bookmarks::{
            split_by_bookmarks_to_dir, BookmarkLevel, SplitByBookmarksOptions,
        };
        // `split_by_bookmarks_to_dir` re-reads the file as raw bytes and
        // cannot decrypt. Rather than silently ignore `--password` and
        // fail later with an opaque parse error, refuse fail-closed with
        // an actionable message (encrypted-input bookmark split is a
        // documented non-goal for this release).
        if password.is_some() {
            return Err(pdf_oxide::Error::InvalidOperation(
                "split --by-bookmarks does not support encrypted PDFs: \
                 decrypt the document first (e.g. `pdf-oxide decrypt`) \
                 then split the result"
                    .to_string(),
            ));
        }
        let out_dir = match output {
            Some(p) => p.to_path_buf(),
            None => super::output_dir_beside(file),
        };
        let opts = SplitByBookmarksOptions {
            title_prefix: bookmark_prefix.map(str::to_string),
            ignore_case,
            level: BookmarkLevel::from_u32(bookmark_level),
            include_front_matter: !no_front_matter,
            ..Default::default()
        };
        let paths = split_by_bookmarks_to_dir(file, &out_dir, &opts)?;
        for p in &paths {
            eprintln!("Wrote {}", p.display());
        }
        return Ok(());
    }

    let doc = super::open_doc(file, password)?;
    let page_count = doc.page_count()?;
    drop(doc);

    let page_indices = super::resolve_pages(pages, page_count)?;

    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("page");

    let default_dir;
    let out_dir = match output {
        Some(p) => p,
        None => {
            default_dir = super::output_dir_beside(file);
            &default_dir
        },
    };

    for &page_idx in &page_indices {
        let mut editor = DocumentEditor::open(file)?;

        // Remove pages from end to start to keep indices stable
        for i in (0..page_count).rev() {
            if i != page_idx {
                editor.remove_page(i)?;
            }
        }

        let out_path = out_dir.join(format!("{}_page_{}.pdf", stem, page_idx + 1));
        editor.save_with_options(
            &out_path,
            SaveOptions {
                compress: true,
                garbage_collect: true,
                ..Default::default()
            },
        )?;
        eprintln!("Saved page {} to {}", page_idx + 1, out_path.display());
    }

    Ok(())
}
