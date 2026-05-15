//! End-to-end coverage tests for the PDF ↔ DOCX/PPTX/XLSX office
//! converter integration introduced in v0.3.48.
//!
//! These exercise:
//!   - layout-mode writers (`docx_layout.rs`, `pptx_layout.rs`,
//!     `xlsx_layout.rs`)
//!   - flow-mode `pdf_to_ir` pipeline
//!   - `layout_lines.rs` line grouping (column gaps, drop-caps)
//!   - `form_xobject_finder.rs` raster + Form XObject extraction
//!   - `office/mod.rs` font registration + back-rendering paths
//!   - `unicode_fallback.rs` system-font fallback (gated on text content)
//!
//! The tests build small in-memory PDFs (no external fixtures required),
//! convert them to each office format, and convert each back to PDF.
//! Assertions verify only that bytes are produced — fidelity testing
//! lives in the bulk_round_trip corpus harness.

use pdf_oxide::api::Pdf;
use pdf_oxide::converters::{Margins, OfficeConfig, OfficeConverter};
use pdf_oxide::document::PdfDocument;

/// Round-trip a PDF through ALL three office formats and back to PDF,
/// in both layout and flow mode, under two distinct OfficeConfigs.
/// This drives the broadest set of `office/mod.rs` back-render
/// branches (positional vs flow, default vs custom page/margins/font)
/// from a single input.
fn exhaustive_roundtrip(md: &str) {
    let pdf = build_pdf_from_markdown(md);
    let doc = load(pdf);

    let docx = doc.to_docx_bytes().expect("docx");
    let pptx = doc.to_pptx_bytes().expect("pptx");
    let xlsx = doc.to_xlsx_bytes().expect("xlsx");
    assert_ooxml(&docx, "docx");
    assert_ooxml(&pptx, "pptx");
    assert_ooxml(&xlsx, "xlsx");

    // Default config back-render — all three formats.
    let def = OfficeConverter::new();
    assert_pdf(&def.convert_docx_bytes(&docx).expect("docx→pdf"));
    assert_pdf(&def.convert_pptx_bytes(&pptx).expect("pptx→pdf"));
    assert_pdf(&def.convert_xlsx_bytes(&xlsx).expect("xlsx→pdf"));

    // Custom config back-render — exercises the non-default
    // page-size / margins / font / line-height branches in the
    // positional + flow renderers.
    let custom = OfficeConverter::with_config(OfficeConfig {
        margins: Margins {
            top: 18.0,
            bottom: 18.0,
            left: 24.0,
            right: 24.0,
        },
        embed_fonts: false,
        default_font: "Times-Roman".to_string(),
        default_font_size: 11.0,
        line_height: 1.4,
        include_images: false,
        ..OfficeConfig::default()
    });
    assert_pdf(&custom.convert_docx_bytes(&docx).expect("docx→pdf cfg"));
    assert_pdf(&custom.convert_pptx_bytes(&pptx).expect("pptx→pdf cfg"));
    assert_pdf(&custom.convert_xlsx_bytes(&xlsx).expect("xlsx→pdf cfg"));

    // Flow-mode office files back-rendered too.
    let docx_f = doc.to_docx_bytes_flow().expect("docx flow");
    let pptx_f = doc.to_pptx_bytes_flow().expect("pptx flow");
    let xlsx_f = doc.to_xlsx_bytes_flow().expect("xlsx flow");
    assert_pdf(&def.convert_docx_bytes(&docx_f).expect("docx flow→pdf"));
    assert_pdf(&def.convert_pptx_bytes(&pptx_f).expect("pptx flow→pdf"));
    assert_pdf(&def.convert_xlsx_bytes(&xlsx_f).expect("xlsx flow→pdf"));
}

/// Build a tiny single-page PDF from markdown — exercises the
/// PdfBuilder Unicode + bundled-font fallback path on its way in,
/// then reload as PdfDocument for the converter API.
fn build_pdf_from_markdown(md: &str) -> Vec<u8> {
    Pdf::from_markdown(md)
        .expect("Pdf::from_markdown")
        .into_bytes()
}

fn load(bytes: Vec<u8>) -> PdfDocument {
    PdfDocument::from_bytes(bytes).expect("PdfDocument::from_bytes")
}

/// Sanity: the produced bytes look like a non-trivial PDF.
fn assert_pdf(bytes: &[u8]) {
    assert!(bytes.starts_with(b"%PDF-"), "not a PDF header");
    assert!(bytes.len() > 256, "PDF suspiciously small: {} bytes", bytes.len());
}

/// Sanity: the produced bytes look like a ZIP-based OOXML file.
fn assert_ooxml(bytes: &[u8], format: &str) {
    assert!(
        bytes.starts_with(b"PK"),
        "{format}: not a ZIP container ({} bytes)",
        bytes.len()
    );
    assert!(bytes.len() > 512, "{format} output suspiciously small: {} bytes", bytes.len());
}

// ---------------------------------------------------------------------------
// Layout-mode round-trips
// ---------------------------------------------------------------------------

#[test]
fn pdf_to_docx_layout_roundtrip() {
    let pdf = build_pdf_from_markdown(
        "# Hello world\n\nThe quick brown fox jumps over the lazy dog. \
         Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n\n\
         ## Subheading\n\nAnother paragraph with more text.",
    );
    let doc = load(pdf);
    let docx = doc.to_docx_bytes().expect("to_docx_bytes");
    assert_ooxml(&docx, "docx");

    // Convert back to PDF via OfficeConverter
    let converter = OfficeConverter::new();
    let pdf_back = converter
        .convert_docx_bytes(&docx)
        .expect("docx → pdf back-render");
    assert_pdf(&pdf_back);
}

#[test]
fn pdf_to_pptx_layout_roundtrip() {
    let pdf = build_pdf_from_markdown(
        "# Slide One\n\nFirst slide content with a paragraph.\n\n\
         # Slide Two\n\nSecond slide with bullets:\n- one\n- two\n- three",
    );
    let doc = load(pdf);
    let pptx = doc.to_pptx_bytes().expect("to_pptx_bytes");
    assert_ooxml(&pptx, "pptx");

    let converter = OfficeConverter::new();
    let pdf_back = converter
        .convert_pptx_bytes(&pptx)
        .expect("pptx → pdf back-render");
    assert_pdf(&pdf_back);
}

#[test]
fn pdf_to_xlsx_layout_roundtrip() {
    let pdf = build_pdf_from_markdown(
        "# Report\n\nThis report has multiple sections.\n\n\
         ## Section A\n\nText for section A.\n\n## Section B\n\n\
         Text for section B with a tabular feel.",
    );
    let doc = load(pdf);
    let xlsx = doc.to_xlsx_bytes().expect("to_xlsx_bytes");
    assert_ooxml(&xlsx, "xlsx");

    let converter = OfficeConverter::new();
    let pdf_back = converter
        .convert_xlsx_bytes(&xlsx)
        .expect("xlsx → pdf back-render");
    assert_pdf(&pdf_back);
}

// ---------------------------------------------------------------------------
// Flow-mode (explicit) — exercises the `pdf_to_office_ir` path that
// the page-count gate routes large documents to.
// ---------------------------------------------------------------------------

#[test]
fn pdf_to_docx_flow_explicit() {
    let pdf = build_pdf_from_markdown("# Flow mode\n\nDocument body for flow-mode conversion.");
    let doc = load(pdf);
    let docx = doc.to_docx_bytes_flow().expect("to_docx_bytes_flow");
    assert_ooxml(&docx, "docx-flow");
}

#[test]
fn pdf_to_pptx_flow_explicit() {
    let pdf = build_pdf_from_markdown("# Flow mode\n\nDocument body for flow-mode pptx.");
    let doc = load(pdf);
    let pptx = doc.to_pptx_bytes_flow().expect("to_pptx_bytes_flow");
    assert_ooxml(&pptx, "pptx-flow");
}

#[test]
fn pdf_to_xlsx_flow_explicit() {
    let pdf = build_pdf_from_markdown("# Flow mode\n\nDocument body for flow-mode xlsx.");
    let doc = load(pdf);
    let xlsx = doc.to_xlsx_bytes_flow().expect("to_xlsx_bytes_flow");
    assert_ooxml(&xlsx, "xlsx-flow");
}

// ---------------------------------------------------------------------------
// Layout-mode (explicit) — exercises the layout writers directly.
// ---------------------------------------------------------------------------

#[test]
fn pdf_to_docx_layout_explicit() {
    let pdf = build_pdf_from_markdown(
        "# Heading\n\nBody paragraph one.\n\nBody paragraph two with **bold**.",
    );
    let doc = load(pdf);
    let docx = doc.to_docx_bytes_layout().expect("to_docx_bytes_layout");
    assert_ooxml(&docx, "docx-layout");
}

// ---------------------------------------------------------------------------
// Unicode fallback path — Latin Extended + Greek + Cyrillic.
// ---------------------------------------------------------------------------

#[test]
fn unicode_fallback_latin_extended() {
    let pdf = build_pdf_from_markdown(
        "# Café résumé naïve\n\n\
         Latin Extended-A: Ą Ć Ę Ł Ń Ś Ź Ż Č Š Ž\n\n\
         Greek: αβγδ Αβγ ΩΨΘ\n\n\
         Cyrillic: привет мир Здравствуйте",
    );
    let doc = load(pdf);
    let docx = doc.to_docx_bytes().expect("docx with extended chars");
    assert_ooxml(&docx, "docx-unicode");
    let converter = OfficeConverter::new();
    let pdf_back = converter
        .convert_docx_bytes(&docx)
        .expect("docx → pdf back-render");
    assert_pdf(&pdf_back);
}

// ---------------------------------------------------------------------------
// Multi-page → exercises page break + section handling.
// ---------------------------------------------------------------------------

#[test]
fn multi_page_docx() {
    let mut md = String::new();
    for i in 1..=5 {
        md.push_str(&format!(
            "# Section {i}\n\n\
             Paragraph A for section {i}. Lorem ipsum dolor sit amet.\n\n\
             Paragraph B for section {i}. Consectetur adipiscing elit.\n\n",
        ));
    }
    let pdf = build_pdf_from_markdown(&md);
    let doc = load(pdf);
    let docx = doc.to_docx_bytes().expect("multi-page docx");
    assert_ooxml(&docx, "multi-page-docx");
}

// ---------------------------------------------------------------------------
// OfficeConverter::new + default config — exercises the constructor
// surface that production callers use.
// ---------------------------------------------------------------------------

#[test]
fn office_converter_default_construction() {
    let _converter = OfficeConverter::new();
    // No-op assertion; the construction itself is the coverage win.
}

// ---------------------------------------------------------------------------
// Rich content → exercises the broadest set of IR element types and the
// back-render paths in `office/mod.rs` (the largest uncovered module).
// Tables, nested lists, bold/italic runs, blockquotes, code blocks,
// horizontal rules, links — each maps to a distinct `Element` /
// `InlineContent` variant and a distinct renderer branch.
// ---------------------------------------------------------------------------

const RICH_MD: &str = "\
# Annual Report 2026

A comprehensive report with **bold**, *italic*, and `inline code`.

## Financial Summary

| Quarter | Revenue | Growth |
|---------|---------|--------|
| Q1      | 1.20M   | +5%    |
| Q2      | 1.35M   | +12%   |
| Q3      | 1.48M   | +10%   |
| Q4      | 1.62M   | +9%    |

## Key Initiatives

1. First ordered item with a longer sentence to force wrapping behaviour.
2. Second ordered item.
   - Nested unordered child A
   - Nested unordered child B
3. Third ordered item.

### Unordered

- Alpha bullet
- Beta bullet with **bold inside**
- Gamma bullet

> A blockquote paragraph that should render as an indented region with
> its own styling, spanning two source lines.

```
fn code_block() {
    // fenced code — monospace rendering path
    let x = 42;
}
```

---

## Closing

A final paragraph after a horizontal rule, with a [link](https://example.com)
and some trailing prose to ensure the last section has body content.
";

#[test]
fn rich_content_docx_roundtrip() {
    let pdf = build_pdf_from_markdown(RICH_MD);
    let doc = load(pdf);
    let docx = doc.to_docx_bytes().expect("rich docx");
    assert_ooxml(&docx, "rich-docx");
    let pdf_back = OfficeConverter::new()
        .convert_docx_bytes(&docx)
        .expect("rich docx → pdf");
    assert_pdf(&pdf_back);
}

#[test]
fn rich_content_pptx_roundtrip() {
    let pdf = build_pdf_from_markdown(RICH_MD);
    let doc = load(pdf);
    let pptx = doc.to_pptx_bytes().expect("rich pptx");
    assert_ooxml(&pptx, "rich-pptx");
    let pdf_back = OfficeConverter::new()
        .convert_pptx_bytes(&pptx)
        .expect("rich pptx → pdf");
    assert_pdf(&pdf_back);
}

#[test]
fn rich_content_xlsx_roundtrip() {
    let pdf = build_pdf_from_markdown(RICH_MD);
    let doc = load(pdf);
    let xlsx = doc.to_xlsx_bytes().expect("rich xlsx");
    assert_ooxml(&xlsx, "rich-xlsx");
    let pdf_back = OfficeConverter::new()
        .convert_xlsx_bytes(&xlsx)
        .expect("rich xlsx → pdf");
    assert_pdf(&pdf_back);
}

#[test]
fn rich_content_flow_all_formats() {
    let pdf = build_pdf_from_markdown(RICH_MD);
    let doc = load(pdf);
    assert_ooxml(&doc.to_docx_bytes_flow().expect("flow docx"), "flow-docx");
    assert_ooxml(&doc.to_pptx_bytes_flow().expect("flow pptx"), "flow-pptx");
    assert_ooxml(&doc.to_xlsx_bytes_flow().expect("flow xlsx"), "flow-xlsx");
}

// ---------------------------------------------------------------------------
// CJK + RTL → exercises cmap_injector + CJK/Unicode fallback selection
// in `office/mod.rs` (ir_has_cjk_text / maybe_load_cjk_fallback) and the
// font-embedding path that the v0.3.48 work added.
// ---------------------------------------------------------------------------

#[test]
fn cjk_content_roundtrip_all_formats() {
    let md = "\
# 文档标题 ドキュメント 문서

中文段落：快速的棕色狐狸跳过了懒狗。这是一个测试文档。

日本語の段落：速い茶色のキツネが怠け者の犬を飛び越えます。

한국어 단락: 빠른 갈색 여우가 게으른 개를 뛰어넘습니다.

Mixed: English 中文 日本語 한국어 in one line.
";
    let pdf = build_pdf_from_markdown(md);
    let doc = load(pdf);
    let docx = doc.to_docx_bytes().expect("cjk docx");
    assert_ooxml(&docx, "cjk-docx");
    let pptx = doc.to_pptx_bytes().expect("cjk pptx");
    assert_ooxml(&pptx, "cjk-pptx");
    let xlsx = doc.to_xlsx_bytes().expect("cjk xlsx");
    assert_ooxml(&xlsx, "cjk-xlsx");
    // Round-trip docx back — exercises CJK fallback in the renderer.
    let pdf_back = OfficeConverter::new()
        .convert_docx_bytes(&docx)
        .expect("cjk docx → pdf");
    assert_pdf(&pdf_back);
}

#[test]
fn rtl_content_roundtrip() {
    let md = "\
# מסמך עברית والعربية

פסקה בעברית: השועל החום המהיר קופץ מעל הכלב העצלן.

فقرة عربية: الثعلب البني السريع يقفز فوق الكلب الكسول.

Mixed עברית and English and عربية text.
";
    let pdf = build_pdf_from_markdown(md);
    let doc = load(pdf);
    let docx = doc.to_docx_bytes().expect("rtl docx");
    assert_ooxml(&docx, "rtl-docx");
    let pdf_back = OfficeConverter::new()
        .convert_docx_bytes(&docx)
        .expect("rtl docx → pdf");
    assert_pdf(&pdf_back);
}

// ---------------------------------------------------------------------------
// Larger multi-page document with heading hierarchy → maximises the
// number of spans the layout-mode line-grouping (`layout_lines.rs`) and
// the IR section/paragraph splitter (`pdf_to_ir.rs`) process.
// ---------------------------------------------------------------------------

#[test]
fn large_multi_page_all_formats() {
    let mut md = String::new();
    for i in 1..=12 {
        md.push_str(&format!(
            "# Chapter {i}\n\n\
             ## {i}.1 Overview\n\n\
             Paragraph one of chapter {i}. The quick brown fox jumps over \
             the lazy dog repeatedly to fill out the line so word-wrap and \
             line-grouping logic gets exercised across many spans.\n\n\
             ## {i}.2 Details\n\n\
             - Bullet {i}.a\n- Bullet {i}.b\n- Bullet {i}.c\n\n\
             Paragraph two with **bold {i}** and *italic {i}* runs and a \
             trailing sentence.\n\n",
        ));
    }
    let pdf = build_pdf_from_markdown(&md);
    let doc = load(pdf);
    assert_ooxml(&doc.to_docx_bytes().expect("large docx"), "large-docx");
    assert_ooxml(&doc.to_pptx_bytes().expect("large pptx"), "large-pptx");
    assert_ooxml(&doc.to_xlsx_bytes().expect("large xlsx"), "large-xlsx");
    let pdf_back = OfficeConverter::new()
        .convert_docx_bytes(&doc.to_docx_bytes().expect("large docx 2"))
        .expect("large docx → pdf");
    assert_pdf(&pdf_back);
}

// ---------------------------------------------------------------------------
// Exhaustive round-trips — drive every back-render branch in office/mod.rs
// (positional + flow, default + custom OfficeConfig) across content
// varieties. These are the bulk of the coverage win for the largest
// uncovered module.
// ---------------------------------------------------------------------------

#[test]
fn exhaustive_simple() {
    exhaustive_roundtrip(
        "# Title\n\nA simple paragraph with **bold** and *italic* text \
         and a [link](https://example.com). Followed by another sentence.",
    );
}

#[test]
fn exhaustive_rich() {
    exhaustive_roundtrip(RICH_MD);
}

#[test]
fn exhaustive_cjk() {
    exhaustive_roundtrip(
        "# 文档 ドキュメント 문서\n\n中文段落测试。\n\n日本語の段落。\n\n한국어 단락.",
    );
}

#[test]
fn exhaustive_rtl() {
    exhaustive_roundtrip(
        "# עברית والعربية\n\nפסקה בעברית.\n\nفقرة عربية.\n\nMixed עברית English عربية.",
    );
}

#[test]
fn exhaustive_unicode_extended() {
    exhaustive_roundtrip(
        "# Café résumé\n\nLatin Extended: Ą Ć Ę Ł Ń. Greek: αβγ ΩΨ. \
         Cyrillic: привет мир. Symbols: € £ ¥ © ® ™ → ≤ ≥ ≠.",
    );
}

#[test]
fn exhaustive_multipage() {
    let mut md = String::new();
    for i in 1..=6 {
        md.push_str(&format!(
            "# Chapter {i}\n\nIntro to chapter {i}. The quick brown fox \
             jumps over the lazy dog.\n\n## {i}.1\n\n- a\n- b\n- c\n\n\
             Body two with **bold {i}**.\n\n",
        ));
    }
    exhaustive_roundtrip(&md);
}
