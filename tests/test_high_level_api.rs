//! Integration tests for the high-level PDF API.

use pdf_oxide::api::{Pdf, PdfBuilder, PdfConfig};
use pdf_oxide::writer::PageSize;
use tempfile::tempdir;

mod pdf_config_tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PdfConfig::default();
        assert_eq!(config.margin_left, 72.0);
        assert_eq!(config.margin_right, 72.0);
        assert_eq!(config.margin_top, 72.0);
        assert_eq!(config.margin_bottom, 72.0);
        assert_eq!(config.font_size, 12.0);
        assert_eq!(config.line_height, 1.5);
        assert!(config.title.is_none());
        assert!(config.author.is_none());
    }
}

mod pdf_builder_tests {
    use super::*;

    #[test]
    fn test_builder_creates_pdf() {
        // Test that builder can create a PDF
        let result = PdfBuilder::new().from_text("Test");
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_title() {
        // Test builder with title creates valid PDF
        let result = PdfBuilder::new().title("Test Title").from_text("Content");
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_author() {
        let result = PdfBuilder::new().author("Test Author").from_text("Content");
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_subject() {
        let result = PdfBuilder::new()
            .subject("Test Subject")
            .from_text("Content");
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_keywords() {
        let result = PdfBuilder::new()
            .keywords("test, keywords")
            .from_text("Content");
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_page_size() {
        let result = PdfBuilder::new()
            .page_size(PageSize::A4)
            .from_text("A4 content");
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_margin() {
        let result = PdfBuilder::new().margin(50.0).from_text("Custom margin");
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_margins() {
        let result = PdfBuilder::new()
            .margins(10.0, 20.0, 30.0, 40.0)
            .from_text("Custom margins");
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_font_size() {
        let result = PdfBuilder::new()
            .font_size(14.0)
            .from_text("Custom font size");
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_line_height() {
        let result = PdfBuilder::new()
            .line_height(1.8)
            .from_text("Custom line height");
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_chain() {
        let result = PdfBuilder::new()
            .title("Title")
            .author("Author")
            .page_size(PageSize::Letter)
            .margin(72.0)
            .font_size(12.0)
            .from_text("Chained builder");

        assert!(result.is_ok());
        let pdf = result.unwrap();
        assert!(!pdf.as_bytes().is_empty());
    }
}

mod pdf_creation_tests {
    use super::*;

    #[test]
    fn test_from_text_simple() {
        let result = Pdf::from_text("Hello, World!");
        assert!(result.is_ok());

        let pdf = result.unwrap();
        assert!(!pdf.as_bytes().is_empty());
        assert!(pdf.as_bytes().starts_with(b"%PDF"));
    }

    #[test]
    fn test_from_text_multiline() {
        let result = Pdf::from_text("Line 1\nLine 2\nLine 3");
        assert!(result.is_ok());

        let pdf = result.unwrap();
        assert!(!pdf.as_bytes().is_empty());
    }

    #[test]
    fn test_from_markdown_heading() {
        let result = Pdf::from_markdown("# Heading 1\n\nSome text.");
        assert!(result.is_ok());

        let pdf = result.unwrap();
        assert!(!pdf.as_bytes().is_empty());
    }

    #[test]
    fn test_from_markdown_multiple_headings() {
        let markdown = r#"
# Chapter 1

Introduction text.

## Section 1.1

More content here.

### Subsection 1.1.1

Even more content.
"#;
        let result = Pdf::from_markdown(markdown);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_markdown_list() {
        let markdown = r#"
# Shopping List

- Apples
- Bananas
- Oranges
"#;
        let result = Pdf::from_markdown(markdown);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_markdown_blockquote() {
        let markdown = r#"
# Quote

> This is a quote.
> It spans multiple lines.
"#;
        let result = Pdf::from_markdown(markdown);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_markdown_code_block() {
        let markdown = r#"
# Code Example

```
fn main() {
    println!("Hello");
}
```
"#;
        let result = Pdf::from_markdown(markdown);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_html_simple() {
        let result = Pdf::from_html("<h1>Hello</h1><p>World</p>");
        assert!(result.is_ok());

        let pdf = result.unwrap();
        assert!(!pdf.as_bytes().is_empty());
    }

    #[test]
    fn test_from_html_formatting() {
        let html = "<h1>Title</h1><p>This is <b>bold</b> and <i>italic</i>.</p>";
        let result = Pdf::from_html(html);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_html_list() {
        let html = "<h1>List</h1><ul><li>Item 1</li><li>Item 2</li></ul>";
        let result = Pdf::from_html(html);
        assert!(result.is_ok());
    }

    #[test]
    fn test_into_bytes() {
        let pdf = Pdf::from_text("Test").unwrap();
        let bytes = pdf.into_bytes();

        assert!(!bytes.is_empty());
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn test_config_access() {
        let pdf = PdfBuilder::new()
            .title("My Title")
            .from_text("Content")
            .unwrap();

        let config = pdf.config();
        assert_eq!(config.title, Some("My Title".to_string()));
    }
}

mod pdf_save_tests {
    use super::*;

    #[test]
    fn test_save_text_pdf() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("text.pdf");

        let mut pdf = Pdf::from_text("Hello, World!").unwrap();
        let result = pdf.save(&path);

        assert!(result.is_ok());
        assert!(path.exists());

        // Check file starts with PDF header
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn test_save_markdown_pdf() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("markdown.pdf");

        let mut pdf = Pdf::from_markdown("# Hello\n\nWorld").unwrap();
        let result = pdf.save(&path);

        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_save_html_pdf() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("html.pdf");

        let mut pdf = Pdf::from_html("<h1>Hello</h1>").unwrap();
        let result = pdf.save(&path);

        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_save_with_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.pdf");

        let mut pdf = PdfBuilder::new()
            .title("Test Document")
            .author("Test Author")
            .subject("Testing")
            .from_text("Content")
            .unwrap();

        let result = pdf.save(&path);
        assert!(result.is_ok());
        assert!(path.exists());
    }
}

mod builder_to_pdf_tests {
    use super::*;

    #[test]
    fn test_builder_from_text() {
        let pdf = PdfBuilder::new()
            .title("Plain Text")
            .from_text("Plain text content")
            .unwrap();

        assert!(!pdf.as_bytes().is_empty());
    }

    #[test]
    fn test_builder_from_markdown() {
        let pdf = PdfBuilder::new()
            .title("Markdown Doc")
            .author("Author")
            .page_size(PageSize::A4)
            .from_markdown("# Title\n\nContent")
            .unwrap();

        assert!(!pdf.as_bytes().is_empty());
    }

    #[test]
    fn test_builder_from_html() {
        let pdf = PdfBuilder::new()
            .title("HTML Doc")
            .margin(50.0)
            .from_html("<h1>Title</h1><p>Content</p>")
            .unwrap();

        assert!(!pdf.as_bytes().is_empty());
    }

    #[test]
    fn test_builder_custom_fonts() {
        let pdf = PdfBuilder::new()
            .font_size(14.0)
            .line_height(1.6)
            .from_text("Custom font settings")
            .unwrap();

        assert!(!pdf.as_bytes().is_empty());
    }

    #[test]
    fn test_builder_custom_margins() {
        let pdf = PdfBuilder::new()
            .margins(36.0, 36.0, 72.0, 72.0)
            .from_text("Custom margins")
            .unwrap();

        assert!(!pdf.as_bytes().is_empty());
    }
}

mod integration_tests {
    use super::*;
    use pdf_oxide::editor::{DocumentEditor, EditableDocument};

    #[test]
    fn test_create_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("roundtrip.pdf");

        // Create PDF
        let mut pdf = Pdf::from_markdown("# Test Document\n\nContent here.").unwrap();
        pdf.save(&path).unwrap();

        // Open with editor
        let editor = DocumentEditor::open(&path);
        assert!(editor.is_ok());

        let mut editor = editor.unwrap();
        assert!(editor.page_count().unwrap() >= 1);
    }

    #[test]
    fn test_full_workflow() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("workflow.pdf");

        // Build with full options
        let mut pdf = PdfBuilder::new()
            .title("Complete Document")
            .author("Integration Test")
            .subject("Testing all features")
            .keywords("test, integration, pdf")
            .page_size(PageSize::Letter)
            .margin(72.0)
            .font_size(12.0)
            .line_height(1.5)
            .from_markdown(
                r#"
# Introduction

This is a complete test document.

## Features

- Headings
- Lists
- Text formatting

## Code

```
let x = 42;
```

> And a quote for good measure.
"#,
            )
            .unwrap();

        // Save
        pdf.save(&path).unwrap();

        // Verify
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"%PDF"));
        assert!(bytes.len() > 100);
    }
}

/// Regression guards for issue #525: Markdown → PDF dropped all styling
/// (headings rendered at body size, `**bold**` markers stripped to plain
/// text). These round-trip through the real extractor and assert the
/// styling actually lands in the PDF — the pre-existing
/// `test_from_markdown_*` tests only checked `is_ok()`, so the bug shipped
/// undetected.
mod markdown_styling_regression {
    use super::*;

    /// Largest font size among ASCII-alphabetic glyphs (headings) and the
    /// smallest (body), so the ratio proves headings are actually scaled.
    fn alpha_size_extent(chars: &[pdf_oxide::layout::TextChar]) -> (f32, f32) {
        let mut min = f32::MAX;
        let mut max = 0.0_f32;
        for c in chars.iter().filter(|c| c.char.is_ascii_alphabetic()) {
            min = min.min(c.font_size);
            max = max.max(c.font_size);
        }
        (min, max)
    }

    #[test]
    fn heading_is_scaled_and_markers_consumed() {
        // The exact reproduction attached to issue #525.
        let mut pdf = Pdf::from_markdown("# Hello\n\nThis is **some** test.").unwrap();
        let chars = pdf.extract_chars(0).expect("extract chars");
        assert!(!chars.is_empty(), "no text extracted from generated PDF");

        let text: String = chars.iter().map(|c| c.char).collect();
        assert!(text.contains("Hello"), "heading text missing: {text:?}");
        assert!(text.contains("test"), "body text missing: {text:?}");
        // The `**` markers must be consumed, not rendered as glyphs.
        assert!(!text.contains('*'), "literal emphasis markers leaked into output: {text:?}");

        // Heading (`# ` => 2.0x of the 12pt default) must be visibly
        // larger than body text.
        let (body, heading) = alpha_size_extent(&chars);
        assert!(heading >= body * 1.5, "heading not scaled: largest={heading} body={body}");

        // The largest glyphs (the heading) must be drawn in a bold face.
        let heading_font = chars
            .iter()
            .filter(|c| (c.font_size - heading).abs() < 0.5)
            .map(|c| c.font_name.to_lowercase())
            .next()
            .unwrap_or_default();
        assert!(heading_font.contains("bold"), "heading not bold, font was {heading_font:?}");
    }

    /// Characters belonging to a contiguous word, found by its first
    /// letter run. Good enough for these single-occurrence fixtures.
    fn word_chars<'a>(
        chars: &'a [pdf_oxide::layout::TextChar],
        word: &str,
    ) -> Vec<&'a pdf_oxide::layout::TextChar> {
        let flat: String = chars.iter().map(|c| c.char).collect();
        match flat.find(word) {
            Some(byte_pos) => {
                let start = flat[..byte_pos].chars().count();
                chars[start..start + word.chars().count()].iter().collect()
            },
            None => Vec::new(),
        }
    }

    #[test]
    fn inline_bold_switches_font_within_body() {
        let mut pdf = Pdf::from_markdown("This is **some** test.").unwrap();
        let chars = pdf.extract_chars(0).expect("extract chars");

        // No heading here — everything is body size. `**some**` must
        // render in the bold face while the surrounding words stay
        // regular, proving the markers triggered a real style switch
        // rather than being stripped.
        let some = word_chars(&chars, "some");
        let this = word_chars(&chars, "This");
        assert!(!some.is_empty() && !this.is_empty(), "words not extracted");
        assert!(
            some.iter()
                .all(|c| c.font_name.to_lowercase().contains("bold")),
            "**some** not rendered bold: {:?}",
            some.iter().map(|c| &c.font_name).collect::<Vec<_>>()
        );
        assert!(
            this.iter()
                .all(|c| !c.font_name.to_lowercase().contains("bold")),
            "surrounding text wrongly bold"
        );
        let text: String = chars.iter().map(|c| c.char).collect();
        assert!(!text.contains('*'), "markers leaked: {text:?}");
    }

    #[test]
    fn inline_italic_uses_oblique_face() {
        let mut pdf = Pdf::from_markdown("plain *slanted* plain").unwrap();
        let chars = pdf.extract_chars(0).expect("extract chars");
        let slanted = word_chars(&chars, "slanted");
        assert!(!slanted.is_empty(), "italic word not extracted");
        // The oblique face must actually be selected (an unregistered
        // resource was the second half of #525) — so the extractor sees
        // an italic/oblique font, not a silent fall-back to regular.
        assert!(
            slanted
                .iter()
                .all(|c| c.is_italic || c.font_name.to_lowercase().contains("oblique")),
            "*slanted* not rendered italic: {:?}",
            slanted.iter().map(|c| &c.font_name).collect::<Vec<_>>()
        );
        let text: String = chars.iter().map(|c| c.char).collect();
        assert!(!text.contains('*'), "markers leaked: {text:?}");
    }

    #[test]
    fn fenced_code_block_uses_monospace() {
        let mut pdf = Pdf::from_markdown("```\nfn main() {}\n```").unwrap();
        let chars = pdf.extract_chars(0).expect("extract chars");
        assert!(!chars.is_empty(), "code block produced no text");
        let mono = chars
            .iter()
            .any(|c| c.is_monospace || c.font_name.to_lowercase().contains("courier"));
        assert!(mono, "fenced code block not rendered in a monospace font");
    }

    /// On the Unicode path (where the document forces DejaVu via a
    /// non-WinAnsi codepoint), code blocks must still be rendered with
    /// a *monospace* font — otherwise GFM tables and fenced code lose
    /// the space-padded alignment that's the whole point of mono. The
    /// pre-fix behaviour used proportional `DejaVuSans` for code on
    /// this path, which collapses table alignment. Courier-for-code is
    /// the documented trade-off; this guards it. (#523 Copilot review.)
    #[test]
    fn fenced_code_block_uses_monospace_on_unicode_path() {
        // Greek capital sigma in the body forces the DejaVu/Unicode path;
        // the code block content is pure ASCII (the common case).
        let md = "Body has \u{03A3} so we go through the Unicode path.\n\n```\nfn main() {}\n```\n";
        let mut pdf = Pdf::from_markdown(md).unwrap();
        let chars = pdf.extract_chars(0).expect("extract chars");
        assert!(!chars.is_empty(), "code block produced no text");
        // Pick the chars that belong to the code (`fn main() {}`) — any
        // of those characters being monospace is enough to prove the
        // mono font was selected for the code spans.
        let code_chars: Vec<_> = chars
            .iter()
            .filter(|c| "fnmai(){}".contains(c.char))
            .collect();
        assert!(
            !code_chars.is_empty(),
            "no code characters extracted from Unicode-path document"
        );
        let mono = code_chars
            .iter()
            .any(|c| c.is_monospace || c.font_name.to_lowercase().contains("courier"));
        assert!(
            mono,
            "code block on Unicode path not monospace: {:?}",
            code_chars
                .iter()
                .map(|c| (c.char, &c.font_name))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn snake_case_underscores_are_preserved() {
        // Underscores are *not* emphasis markers here — a regression in
        // the old code stripped them, mangling identifiers.
        let mut pdf = Pdf::from_markdown("call my_func_name(x) now").unwrap();
        let chars = pdf.extract_chars(0).expect("extract chars");
        let text: String = chars.iter().map(|c| c.char).collect();
        assert!(text.contains("my_func_name"), "snake_case identifier mangled: {text:?}");
    }

    #[test]
    fn unicode_heading_is_scaled() {
        // A Greek capital sigma forces the Unicode (DejaVu) font path;
        // headings must still be scaled there, not flattened to body size.
        let mut pdf = Pdf::from_markdown("# \u{03A3}igma\n\nPlain body line.").unwrap();
        let chars = pdf.extract_chars(0).expect("extract chars");
        assert!(!chars.is_empty());
        let (body, heading) = alpha_size_extent(&chars);
        assert!(
            heading >= body * 1.5,
            "unicode heading not scaled: largest={heading} body={body}"
        );
        let text: String = chars.iter().map(|c| c.char).collect();
        assert!(!text.contains('*'), "markers leaked: {text:?}");
    }
}

/// `DocumentBuilder::rich_paragraph` set `font.name = "Helvetica-Bold"`
/// but left `style.weight` default, so the old `map_font_name`
/// collapsed it to plain `Helvetica` — the same root cause as #525 via
/// a different public API, previously untested. Guards the latent fix.
mod document_builder_rich_text_regression {
    use pdf_oxide::writer::{DocumentBuilder, PageSize, TextRun};

    #[test]
    fn rich_paragraph_bold_and_italic_select_styled_faces() {
        let mut b = DocumentBuilder::new();
        {
            let p = b
                .page(PageSize::Letter)
                .at(72.0, 700.0)
                .font("Helvetica", 14.0);
            p.rich_paragraph(&[
                TextRun::normal("plain "),
                TextRun::bold("BOLD "),
                TextRun::italic("ITALIC"),
            ])
            .done();
        }
        let bytes = b.build().expect("build");
        let mut pdf = pdf_oxide::api::Pdf::from_bytes(bytes).expect("load");
        let chars = pdf.extract_chars(0).expect("extract chars");

        let face = |word: &str| -> String {
            let flat: String = chars.iter().map(|c| c.char).collect();
            let start = flat.find(word).map(|b| flat[..b].chars().count()).unwrap();
            chars[start].font_name.to_lowercase()
        };
        assert!(
            !face("plain").contains("bold") && !face("plain").contains("oblique"),
            "normal run not regular"
        );
        assert!(face("BOLD").contains("bold"), "bold run not bold");
        assert!(
            face("ITALIC").contains("oblique") || face("ITALIC").contains("italic"),
            "italic run not oblique"
        );
    }
}
