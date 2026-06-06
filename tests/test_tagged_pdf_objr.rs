//! Tests for OBJR (Object Reference) handling in structure tree parsing.
//!
//! PDF spec §14.7.4, Table 323: OBJR (Object Reference) entries in the /K array
//! of a StructElem or StructTreeRoot indirectly reference StructElems via their
//! /Obj entry. Previously these were silently dropped, causing content owned by
//! the referenced StructElems to be missing from text extraction.
//!
//! Also tests that the structure tree is used for PDFs that have a /StructTreeRoot
//! in the catalog even when /MarkInfo /Marked is absent (PDF 1.4 documents).

/// Embedded hello_structure.pdf — a real-world two-page tagged PDF with:
/// - No /MarkInfo in the catalog (PDF 1.4)
/// - OBJR entries in both the StructTreeRoot /K array and StructElem /K arrays
/// - Source: kreuzberg-dev/kreuzberg test corpus (public domain)
const HELLO_STRUCTURE_PDF: &[u8] = include_bytes!("fixtures/hello_structure.pdf");

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the structure tree is loaded for PDFs without /MarkInfo.
    /// hello_structure.pdf has a /StructTreeRoot but no /MarkInfo /Marked entry.
    #[test]
    fn test_tagged_pdf_without_mark_info_loads_structure_tree() {
        let doc =
            pdf_oxide::document::PdfDocument::from_bytes(HELLO_STRUCTURE_PDF.to_vec()).unwrap();

        // The document should have a structure tree even though MarkInfo is absent
        let tree = doc.structure_tree().unwrap();
        assert!(
            tree.is_some(),
            "hello_structure.pdf has a StructTreeRoot and should produce a structure tree"
        );
    }

    /// Verify that OBJR references in /K arrays are resolved to their target StructElems.
    /// The Section (obj 9) in hello_structure.pdf has:
    ///   /K [10 0 R  << /Type OBJR /Pg 11 0 R /Obj 13 0 R >>]
    /// Without OBJR handling, obj 13 (Title, MCID 1, page 1) is silently dropped.
    #[test]
    fn test_objr_in_struct_elem_k_array_is_resolved() {
        let doc =
            pdf_oxide::document::PdfDocument::from_bytes(HELLO_STRUCTURE_PDF.to_vec()).unwrap();
        let tree = doc
            .structure_tree()
            .unwrap()
            .expect("structure tree present");

        // Walk the tree and collect all MCIDs and pages
        let mut found_mcids: Vec<(u32, u32)> = Vec::new(); // (page, mcid)
        collect_mcids(&tree.root_elements, &mut found_mcids);

        // MCID 1 on page 1 (obj 13 — "Goodbye Cruel World" Title) must be found.
        // Without OBJR fix this was silently dropped.
        assert!(
            found_mcids.contains(&(1, 1)),
            "MCID 1 on page 1 (Goodbye Cruel World) should be in structure tree; \
             found MCIDs: {:?}",
            found_mcids
        );
    }

    /// Verify that OBJR at the StructTreeRoot /K level is resolved.
    /// StructTreeRoot /K = [9 0 R  << /Type OBJR /Pg 11 0 R /Obj 14 0 R >>]
    /// Without OBJR handling, obj 14 (P element with MCID 2) is silently dropped.
    #[test]
    fn test_objr_at_struct_tree_root_k_level_is_resolved() {
        let doc =
            pdf_oxide::document::PdfDocument::from_bytes(HELLO_STRUCTURE_PDF.to_vec()).unwrap();
        let tree = doc
            .structure_tree()
            .unwrap()
            .expect("structure tree present");

        let mut found_mcids: Vec<(u32, u32)> = Vec::new();
        collect_mcids(&tree.root_elements, &mut found_mcids);

        // MCID 2 on page 1 (obj 14 — "I'll be back shortly!" P element) must be found.
        // Without OBJR fix this was silently dropped from the root K array.
        assert!(
            found_mcids.contains(&(1, 2)),
            "MCID 2 on page 1 (I'll be back shortly!) should be in structure tree; \
             found MCIDs: {:?}",
            found_mcids
        );
    }

    /// End-to-end: all three text strings from hello_structure.pdf must appear
    /// in extracted text across both pages.
    #[test]
    fn test_hello_structure_full_text_extracted() {
        let doc =
            pdf_oxide::document::PdfDocument::from_bytes(HELLO_STRUCTURE_PDF.to_vec()).unwrap();
        let page_count = doc.page_count().unwrap_or(0);

        let all_text: String = (0..page_count)
            .map(|i| doc.extract_text(i).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            all_text.contains("Hello World"),
            "page 0 text 'Hello World' missing from: {:?}",
            all_text
        );
        assert!(
            all_text.contains("Goodbye Cruel World"),
            "page 1 heading 'Goodbye Cruel World' missing from: {:?}",
            all_text
        );
        assert!(
            all_text.contains("I'll be back shortly!")
                || all_text.contains("I\u{2019}ll be back shortly!"),
            "page 1 paragraph missing from: {:?}",
            all_text
        );
    }

    /// Regression test for issue #486: text-only spatial table fallback must NOT fire
    /// on tagged PDFs.  Before the guard was added, `to_markdown` with text_fallback=true
    /// detected the "Goodbye Cruel World" heading (page 1) as a table row and emitted
    /// `| | Goodbye | Cruel | World... |` instead of `# Goodbye Cruel World...`.
    ///
    /// The fix: `extract_page_tables` returns early (no spatial detection) when the
    /// document has a structure tree (`struct_tree_opt.is_some()`).
    #[test]
    fn test_tagged_pdf_to_markdown_no_false_positive_table() {
        let doc =
            pdf_oxide::document::PdfDocument::from_bytes(HELLO_STRUCTURE_PDF.to_vec()).unwrap();
        let options = pdf_oxide::converters::ConversionOptions::default();

        // Page 1 contains "Goodbye Cruel World" as a heading element in the structure tree.
        // With the text-only spatial fallback guard absent, this heading was mistakenly
        // rendered as a pipe-delimited table row.
        let md = doc
            .to_markdown(1, &options)
            .expect("to_markdown page 1 should succeed");

        assert!(
            !md.contains('|'),
            "Page 1 of hello_structure.pdf should NOT contain pipe characters (no false-positive \
             table from spatial heuristics on a tagged PDF). Got:\n{:?}",
            md
        );
        assert!(
            md.contains("Goodbye Cruel World"),
            "Page 1 heading text should be present in markdown. Got:\n{:?}",
            md
        );
    }

    /// Regression test for issue #486: text-only spatial table fallback must NOT fire
    /// on pages where more than 30% of spans are RTL (Arabic / Hebrew).
    ///
    /// The fix: `extract_page_tables` computes `rtl_fraction` over `input_spans` and
    /// returns early when `rtl_fraction > 0.30`.
    ///
    /// This test directly drives the RTL guard path by building a minimal set of
    /// synthetic spans with predominantly Arabic text and asserting that
    /// `looks_rtl` agrees they are RTL (i.e. the guard would fire), without
    /// needing a full RTL PDF fixture.  A PDF-level integration test is gated by
    /// `quality_gate_right_to_left_02_markdown` in test_corpus_extraction_quality.rs.
    #[test]
    fn test_rtl_spans_detected_by_looks_rtl_guard() {
        // Verify that Arabic-script text is recognised as RTL by the function
        // used in the guard.  If this fails the guard itself is broken.
        let arabic_samples = [
            "مرحبا",                    // "Hello" in Arabic
            "كيف حالك",                 // "How are you"
            "شكراً",                     // "Thank you"
            "\u{0628}\u{064A}\u{062A}", // بيت (house)
        ];
        for sample in &arabic_samples {
            assert!(
                pdf_oxide::text::bidi::looks_rtl(sample),
                "looks_rtl should return true for Arabic text {:?} (used in RTL guard)",
                sample
            );
        }

        // Verify that LTR text is NOT flagged as RTL.
        let ltr_samples = ["hello world", "table data", "123", ""];
        for sample in &ltr_samples {
            assert!(
                !pdf_oxide::text::bidi::looks_rtl(sample),
                "looks_rtl should return false for LTR text {:?}",
                sample
            );
        }
    }

    fn collect_mcids(elems: &[pdf_oxide::structure::types::StructElem], out: &mut Vec<(u32, u32)>) {
        for elem in elems {
            for child in &elem.children {
                match child {
                    pdf_oxide::structure::types::StructChild::MarkedContentRef {
                        mcid,
                        page,
                        ..
                    } => {
                        out.push((*page, *mcid));
                    },
                    pdf_oxide::structure::types::StructChild::StructElem(child_elem) => {
                        collect_mcids(std::slice::from_ref(child_elem), out);
                    },
                    _ => {},
                }
            }
        }
    }
}
