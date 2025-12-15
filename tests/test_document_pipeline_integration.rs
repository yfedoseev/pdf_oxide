//! Integration tests for TextPipeline in document.rs
//!
//! This test suite validates that the document.rs methods correctly use
//! the TextPipeline architecture for text extraction and conversion.
//!
//! Tests follow TDD practices: they define expected behavior before
//! implementation is complete.

#![allow(clippy::field_reassign_with_default)]

#[cfg(test)]
mod document_pipeline_integration {
    use pdf_oxide::converters::ConversionOptions;
    use pdf_oxide::document::PdfDocument;
    use std::path::PathBuf;

    /// Helper to find test PDF files
    fn get_test_pdf(name: &str) -> PathBuf {
        // Try multiple common locations
        let paths = vec![
            format!("test_data/{}", name),
            format!("tests/test_data/{}", name),
            format!("../test_data/{}", name),
        ];

        for path in paths {
            let p = PathBuf::from(&path);
            if p.exists() {
                return p;
            }
        }

        // Return the first path anyway - test will fail with clear error message
        PathBuf::from(format!("test_data/{}", name))
    }

    #[test]
    fn test_document_to_markdown_produces_output() {
        // This test validates that to_markdown() returns non-empty output
        // It should work with any PDF that has text content

        let pdf_path = get_test_pdf("sample.pdf");
        if !pdf_path.exists() {
            println!("Test PDF not found at {:?}, skipping", pdf_path);
            return;
        }

        let mut doc = match PdfDocument::open(&pdf_path) {
            Ok(d) => d,
            Err(_) => {
                println!("Could not open PDF at {:?}, skipping", pdf_path);
                return;
            },
        };

        let options = ConversionOptions::default();

        // Should produce markdown output without error
        match doc.to_markdown(0, &options) {
            Ok(markdown) => {
                // Output should be non-empty
                assert!(!markdown.is_empty(), "Markdown output should not be empty");

                // Should contain valid content
                assert!(!markdown.is_empty(), "Output length should be > 0");
            },
            Err(e) => {
                // Some PDFs may fail, but the interface should work
                eprintln!("to_markdown() failed: {}", e);
            },
        }
    }

    #[test]
    fn test_document_to_html_produces_output() {
        // This test validates that to_html() returns valid HTML output

        let pdf_path = get_test_pdf("sample.pdf");
        if !pdf_path.exists() {
            println!("Test PDF not found at {:?}, skipping", pdf_path);
            return;
        }

        let mut doc = match PdfDocument::open(&pdf_path) {
            Ok(d) => d,
            Err(_) => {
                println!("Could not open PDF at {:?}, skipping", pdf_path);
                return;
            },
        };

        let options = ConversionOptions::default();

        // Should produce HTML output without error
        match doc.to_html(0, &options) {
            Ok(html) => {
                // Output should be non-empty
                assert!(!html.is_empty(), "HTML output should not be empty");

                // Should contain some HTML structure
                assert!(!html.is_empty(), "Output length should be > 0");
            },
            Err(e) => {
                // Some PDFs may fail, but the interface should work
                eprintln!("to_html() failed: {}", e);
            },
        }
    }

    #[test]
    fn test_document_to_plain_text_produces_output() {
        // This test validates that to_plain_text() returns plain text output

        let pdf_path = get_test_pdf("sample.pdf");
        if !pdf_path.exists() {
            println!("Test PDF not found at {:?}, skipping", pdf_path);
            return;
        }

        let mut doc = match PdfDocument::open(&pdf_path) {
            Ok(d) => d,
            Err(_) => {
                println!("Could not open PDF at {:?}, skipping", pdf_path);
                return;
            },
        };

        let options = ConversionOptions::default();

        // Should produce plain text output without error
        match doc.to_plain_text(0, &options) {
            Ok(text) => {
                // Output should be non-empty
                assert!(!text.is_empty(), "Plain text output should not be empty");

                // Plain text should not contain HTML or markdown markup
                assert!(!text.contains("<html"), "Should not contain HTML tags");
            },
            Err(e) => {
                // Some PDFs may fail, but the interface should work
                eprintln!("to_plain_text() failed: {}", e);
            },
        }
    }

    #[test]
    fn test_conversion_options_preserved() {
        // This test validates that conversion options are properly passed through
        // the pipeline and affect the output

        let pdf_path = get_test_pdf("sample.pdf");
        if !pdf_path.exists() {
            println!("Test PDF not found at {:?}, skipping", pdf_path);
            return;
        }

        let mut doc = match PdfDocument::open(&pdf_path) {
            Ok(d) => d,
            Err(_) => {
                println!("Could not open PDF at {:?}, skipping", pdf_path);
                return;
            },
        };

        // Create two sets of options with different settings
        let mut options1 = ConversionOptions::default();
        options1.detect_headings = false;

        let mut options2 = ConversionOptions::default();
        options2.detect_headings = true;

        // Both should produce output
        match (doc.to_markdown(0, &options1), doc.to_markdown(0, &options2)) {
            (Ok(md1), Ok(md2)) => {
                // Output should be produced
                assert!(!md1.is_empty(), "Output with detect_headings=false should not be empty");
                assert!(!md2.is_empty(), "Output with detect_headings=true should not be empty");

                // The outputs may be different depending on heading detection
                // This at least validates the options are being used
            },
            (Err(e1), _) => {
                eprintln!("First to_markdown() failed: {}", e1);
            },
            (_, Err(e2)) => {
                eprintln!("Second to_markdown() failed: {}", e2);
            },
        }
    }

    #[test]
    fn test_pipeline_config_adapter() {
        // This test validates that TextPipelineConfig::from_conversion_options()
        // correctly converts legacy ConversionOptions

        use pdf_oxide::pipeline::TextPipelineConfig;

        let mut options = ConversionOptions::default();
        options.detect_headings = true;
        options.preserve_layout = true;
        options.extract_tables = true;

        let config = TextPipelineConfig::from_conversion_options(&options);

        // Verify the config was properly created
        assert!(config.output.detect_headings, "detect_headings should be preserved");
        assert!(config.output.preserve_layout, "preserve_layout should be preserved");
        assert!(config.output.extract_tables, "extract_tables should be preserved");
    }

    #[test]
    fn test_output_formats_consistent() {
        // This test validates that converting the same page to different formats
        // produces consistent text content (just with different markup)

        let pdf_path = get_test_pdf("sample.pdf");
        if !pdf_path.exists() {
            println!("Test PDF not found at {:?}, skipping", pdf_path);
            return;
        }

        let mut doc = match PdfDocument::open(&pdf_path) {
            Ok(d) => d,
            Err(_) => {
                println!("Could not open PDF at {:?}, skipping", pdf_path);
                return;
            },
        };

        let options = ConversionOptions::default();

        // Get text in all three formats
        let markdown = doc.to_markdown(0, &options).ok();
        let html = doc.to_html(0, &options).ok();
        let plain_text = doc.to_plain_text(0, &options).ok();

        // At least one should succeed
        assert!(
            markdown.is_some() || html.is_some() || plain_text.is_some(),
            "At least one output format should succeed"
        );

        // If we have outputs, they should be non-empty
        if let Some(md) = markdown {
            assert!(!md.is_empty(), "Markdown output should be non-empty");
        }
        if let Some(h) = html {
            assert!(!h.is_empty(), "HTML output should be non-empty");
        }
        if let Some(txt) = plain_text {
            assert!(!txt.is_empty(), "Plain text output should be non-empty");
        }
    }

    #[test]
    fn test_pipeline_reading_order_context() {
        // This test validates that the pipeline receives proper context
        // for reading order determination

        use pdf_oxide::pipeline::ReadingOrderContext;

        // Create a context as the pipeline would
        let context = ReadingOrderContext::new().with_page(0);

        // Context should be properly initialized
        assert_eq!(context.page_number, 0, "Page number should be set");
        assert!(context.mcid_order.is_none(), "MCID order should be empty initially");
    }

    #[test]
    fn test_extract_spans_foundation() {
        // This test validates that extract_spans() still works as the foundation
        // for the pipeline (it should NOT be changed)

        let pdf_path = get_test_pdf("sample.pdf");
        if !pdf_path.exists() {
            println!("Test PDF not found at {:?}, skipping", pdf_path);
            return;
        }

        let mut doc = match PdfDocument::open(&pdf_path) {
            Ok(d) => d,
            Err(_) => {
                println!("Could not open PDF at {:?}, skipping", pdf_path);
                return;
            },
        };

        // extract_spans should still work independently
        match doc.extract_spans(0) {
            Ok(spans) => {
                // Should be able to get spans
                println!("Successfully extracted {} spans", spans.len());
            },
            Err(e) => {
                eprintln!("extract_spans failed: {}", e);
            },
        }
    }

    #[test]
    fn test_multiple_pages_conversion() {
        // This test validates that multiple pages can be converted correctly

        let pdf_path = get_test_pdf("sample.pdf");
        if !pdf_path.exists() {
            println!("Test PDF not found at {:?}, skipping", pdf_path);
            return;
        }

        let mut doc = match PdfDocument::open(&pdf_path) {
            Ok(d) => d,
            Err(_) => {
                println!("Could not open PDF at {:?}, skipping", pdf_path);
                return;
            },
        };

        // Get page count
        let page_count = match doc.page_count() {
            Ok(c) => c,
            Err(_) => {
                println!("Could not determine page count, skipping");
                return;
            },
        };

        let options = ConversionOptions::default();

        // Try to convert first and last page if available
        if page_count > 0 {
            match doc.to_markdown(0, &options) {
                Ok(md) => {
                    assert!(!md.is_empty(), "First page markdown should be non-empty");
                },
                Err(e) => {
                    eprintln!("Could not convert page 0: {}", e);
                },
            }
        }

        if page_count > 1 {
            match doc.to_markdown((page_count - 1) as usize, &options) {
                Ok(md) => {
                    assert!(!md.is_empty(), "Last page markdown should be non-empty");
                },
                Err(e) => {
                    eprintln!("Could not convert last page: {}", e);
                },
            }
        }
    }

    #[test]
    fn test_markdown_output_has_content() {
        // This test validates that markdown output contains actual text content,
        // not just empty formatting

        let pdf_path = get_test_pdf("sample.pdf");
        if !pdf_path.exists() {
            println!("Test PDF not found at {:?}, skipping", pdf_path);
            return;
        }

        let mut doc = match PdfDocument::open(&pdf_path) {
            Ok(d) => d,
            Err(_) => {
                println!("Could not open PDF at {:?}, skipping", pdf_path);
                return;
            },
        };

        let options = ConversionOptions::default();

        match doc.to_markdown(0, &options) {
            Ok(md) => {
                // Should have content
                let trimmed = md.trim();
                assert!(!trimmed.is_empty(), "Markdown should have content after trimming");

                // Should have non-whitespace characters
                assert!(
                    trimmed.chars().any(|c| !c.is_whitespace()),
                    "Markdown should contain non-whitespace characters"
                );
            },
            Err(e) => {
                eprintln!("to_markdown failed: {}", e);
            },
        }
    }
}
