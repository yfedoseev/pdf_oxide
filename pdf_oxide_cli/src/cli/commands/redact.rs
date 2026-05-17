//! `pdf-oxide redact` — destructive content redaction (#231).
//!
//! Physically removes the text under each region (not a cosmetic
//! overlay) per ISO 32000-1:2008 §12.5.6.23, then saves via the
//! garbage-collected full rewrite so no residual recoverable bytes
//! survive.

use pdf_oxide::editor::{DocumentEditor, EditableDocument, SaveOptions};
use pdf_oxide::{Error, RedactionOptions};
use std::path::Path;

// RedactionOptions is #[non_exhaustive] (defined in another crate), so a
// struct literal is impossible — default-then-assign is the only option.
#[allow(clippy::too_many_arguments, clippy::field_reassign_with_default)]
pub fn run(
    file: &Path,
    rects: &[String],
    from_annotations: bool,
    fill: Option<&str>,
    no_scrub_metadata: bool,
    output: Option<&Path>,
    password: Option<&str>,
) -> pdf_oxide::Result<()> {
    // Authenticate first so `--password` actually works on encrypted
    // inputs (the shared open_doc helper handles the password); then
    // build the editor from the opened document. `DocumentEditor::open`
    // alone would ignore `--password` and later fail on encrypted PDFs
    // with an opaque "not authenticated" error (Copilot review #512).
    let doc = super::open_doc(file, password)?;
    let mut editor = DocumentEditor::from_document(doc)?;
    let fill_color = parse_fill(fill)?;

    let mut queued = 0usize;
    for spec in rects {
        let (page, rect) = parse_rect(spec)?;
        editor.add_redaction(page, rect, fill_color)?;
        queued += 1;
    }
    if from_annotations {
        // Mark every page so source /Redact annotations are applied too.
        editor.apply_all_redactions()?;
    }
    if queued == 0 && !from_annotations {
        return Err(Error::InvalidOperation(
            "redact: provide at least one --rect PAGE:x0,y0,x1,y1 or --from-annotations"
                .to_string(),
        ));
    }

    // RedactionOptions is #[non_exhaustive]; build via Default then set.
    let mut opts = RedactionOptions::default();
    opts.scrub_metadata = !no_scrub_metadata;
    let report = editor.apply_redactions_destructive(opts)?;
    eprintln!(
        "Redacted {} region(s): {} glyphs removed, {} bytes removed",
        report.regions, report.glyphs_removed, report.bytes_removed
    );

    let out_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| super::output_beside(file, "_redacted.pdf"));
    // Always full-rewrite + GC so the original (secret) content object
    // cannot survive (#231 G6).
    editor.save_with_options(
        &out_path,
        SaveOptions {
            compress: true,
            garbage_collect: true,
            ..Default::default()
        },
    )?;
    eprintln!("Saved to {}", out_path.display());
    Ok(())
}

/// Parse `R,G,B` (each 0..1) into a fill colour, or `None`.
fn parse_fill(s: Option<&str>) -> pdf_oxide::Result<Option<[f32; 3]>> {
    let Some(v) = s else { return Ok(None) };
    let parts = v
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::InvalidOperation(format!("redact: bad --fill '{v}' (want R,G,B)")))?;
    if parts.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "redact: --fill needs 3 components, got '{v}'"
        )));
    }
    Ok(Some([parts[0], parts[1], parts[2]]))
}

/// Parse `PAGE:x0,y0,x1,y1` into `(page, rect)`.
fn parse_rect(spec: &str) -> pdf_oxide::Result<(usize, [f32; 4])> {
    let (page_s, rest) = spec.split_once(':').ok_or_else(|| {
        Error::InvalidOperation(format!("redact: --rect '{spec}' must be PAGE:x0,y0,x1,y1"))
    })?;
    let page: usize = page_s
        .trim()
        .parse()
        .map_err(|_| Error::InvalidOperation(format!("redact: bad page in '{spec}'")))?;
    let nums = rest
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::InvalidOperation(format!("redact: bad coords in '{spec}'")))?;
    if nums.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "redact: --rect '{spec}' needs 4 coords (x0,y0,x1,y1)"
        )));
    }
    Ok((page, [nums[0], nums[1], nums[2], nums[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rect_ok() {
        let (p, r) = parse_rect("2:10,20,110,70").unwrap();
        assert_eq!(p, 2);
        assert_eq!(r, [10.0, 20.0, 110.0, 70.0]);
    }

    #[test]
    fn parse_rect_rejects_bad_shapes() {
        assert!(parse_rect("nopage").is_err());
        assert!(parse_rect("0:1,2,3").is_err());
        assert!(parse_rect("x:1,2,3,4").is_err());
        assert!(parse_rect("0:a,b,c,d").is_err());
    }

    #[test]
    fn parse_fill_variants() {
        assert_eq!(parse_fill(None).unwrap(), None);
        assert_eq!(parse_fill(Some("1,0,0")).unwrap(), Some([1.0, 0.0, 0.0]));
        assert!(parse_fill(Some("1,0")).is_err());
        assert!(parse_fill(Some("a,b,c")).is_err());
    }
}
