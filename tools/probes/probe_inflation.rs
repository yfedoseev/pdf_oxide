/// Probe round-trip page-count inflation on a single source PDF.
///
/// Walks two paths and compares them per-section:
///   A) IR (from `pdf_to_ir`)            → `ir_to_pdf_bytes_pub`     → PDF
///   B) IR → `to_pptx_bytes`  → parse  → `to_ir`  → `ir_to_pdf_bytes_pub` → PDF
///
/// Reports element/character counts, font sizes, and rendered page counts so
/// we can pinpoint where the inflation comes from (size loss, paragraph
/// splitting, synthesized headings, trailing gaps, etc.).
///
/// Usage:
///   cargo run --release --features rendering --example probe_inflation -- \
///     /home/yfedoseev/projects/pdf_oxide_tests/pdfs/academic/arxiv_2510.21165v1.pdf
use office_oxide::ir::{DocumentIR, Element, InlineContent};
use office_oxide::{Document, DocumentFormat};
use pdf_oxide::{
    api::Pdf,
    converters::{
        pdf_to_ir::{pdf_to_ir, PdfToIrOptions},
        office::{ir_to_pdf_bytes_pub, OfficeConfig},
    },
    document::PdfDocument,
};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let default_path =
        "/home/yfedoseev/projects/pdf_oxide_tests/pdfs/academic/arxiv_2510.21165v1.pdf"
            .to_string();
    let path = args.get(1).cloned().unwrap_or(default_path);

    println!("=== probe_inflation: {path} ===");

    // Source PDF.
    let mut src = Pdf::open(&path).expect("open source PDF");
    let src_pages = src.page_count().expect("page count");
    println!("source PDF pages: {src_pages}");
    drop(src);

    // 1) Source PDF → IR.
    let pdf_doc = PdfDocument::open(&path).expect("open PdfDocument");
    let opts = PdfToIrOptions::default();
    let ir_a = pdf_to_ir(&pdf_doc, office_oxide::format::DocumentFormat::Pptx, &opts).expect("pdf_to_ir");
    println!("\n[IR_A] sections={}", ir_a.sections.len());
    summarize(&ir_a, "IR_A");

    // 2a) IR_A → PDF directly.
    let cfg = OfficeConfig::default();
    let bytes_direct = ir_to_pdf_bytes_pub(&ir_a, &cfg).expect("ir_to_pdf direct");
    let direct_pages = count_pdf_pages(&bytes_direct);
    println!("\n[direct] IR_A → PDF pages: {direct_pages}");

    // 2b) IR_A → PPTX → IR_B → PDF.
    let pptx_bytes = pdf_doc.to_pptx_bytes().expect("to_pptx_bytes");
    let cursor = std::io::Cursor::new(pptx_bytes.clone());
    let pptx_doc =
        Document::from_reader(cursor, DocumentFormat::Pptx).expect("parse pptx");
    let ir_b = pptx_doc.to_ir();
    println!("\n[IR_B] sections={}", ir_b.sections.len());
    summarize(&ir_b, "IR_B");

    let bytes_rt = ir_to_pdf_bytes_pub(&ir_b, &cfg).expect("ir_to_pdf rt");
    let rt_pages = count_pdf_pages(&bytes_rt);
    println!("\n[round-trip via PPTX] IR_B → PDF pages: {rt_pages}");

    println!(
        "\n=== summary: source={src_pages} direct={direct_pages} rt_pptx={rt_pages} ===\
         \n  inflation direct = {:.2}×, rt = {:.2}×",
        direct_pages as f32 / src_pages.max(1) as f32,
        rt_pages as f32 / src_pages.max(1) as f32,
    );

    // Per-section parity check (first 4 sections).
    println!("\n--- per-section element comparison (first 4) ---");
    for i in 0..ir_a.sections.len().min(ir_b.sections.len()).min(4) {
        let a = &ir_a.sections[i];
        let b = &ir_b.sections[i];
        let a_chars: usize = section_chars(a);
        let b_chars: usize = section_chars(b);
        let a_elems = a.elements.len();
        let b_elems = b.elements.len();
        let a_paras = a
            .elements
            .iter()
            .filter(|e| matches!(e, Element::Paragraph(_)))
            .count();
        let b_paras = b
            .elements
            .iter()
            .filter(|e| matches!(e, Element::Paragraph(_)))
            .count();
        let a_heads = a
            .elements
            .iter()
            .filter(|e| matches!(e, Element::Heading(_)))
            .count();
        let b_heads = b
            .elements
            .iter()
            .filter(|e| matches!(e, Element::Heading(_)))
            .count();
        let a_size = first_paragraph_size(a);
        let b_size = first_paragraph_size(b);
        println!(
            "  sec {i}: A elems={a_elems} (h={a_heads} p={a_paras}) chars={a_chars} body_size={a_size:?}  |  B elems={b_elems} (h={b_heads} p={b_paras}) chars={b_chars} body_size={b_size:?}"
        );
    }
}

fn count_pdf_pages(bytes: &[u8]) -> usize {
    let pdf = Pdf::from_bytes(bytes.to_vec());
    match pdf {
        Ok(mut p) => p.page_count().unwrap_or(0),
        Err(e) => {
            eprintln!("count_pdf_pages: parse failed: {e:?}");
            0
        }
    }
}

fn section_chars(s: &office_oxide::ir::Section) -> usize {
    let mut total = 0;
    for el in &s.elements {
        match el {
            Element::Paragraph(p) => total += inline_chars(&p.content),
            Element::Heading(h) => total += inline_chars(&h.content),
            _ => {}
        }
    }
    total
}

fn inline_chars(c: &[InlineContent]) -> usize {
    c.iter()
        .map(|ic| match ic {
            InlineContent::Text(span) => span.text.chars().count(),
            _ => 0,
        })
        .sum()
}

fn first_paragraph_size(s: &office_oxide::ir::Section) -> Option<f32> {
    for el in &s.elements {
        if let Element::Paragraph(p) = el {
            for ic in &p.content {
                if let InlineContent::Text(span) = ic {
                    if let Some(half_pt) = span.font_size_half_pt {
                        return Some(half_pt as f32 / 2.0);
                    }
                }
            }
        }
    }
    None
}

fn summarize(ir: &DocumentIR, tag: &str) {
    let mut total_chars = 0usize;
    let mut total_paras = 0usize;
    let mut total_heads = 0usize;
    let mut sizes_seen: Vec<f32> = Vec::new();
    for sec in &ir.sections {
        for el in &sec.elements {
            match el {
                Element::Paragraph(p) => {
                    total_paras += 1;
                    total_chars += inline_chars(&p.content);
                    for ic in &p.content {
                        if let InlineContent::Text(span) = ic {
                            if let Some(h) = span.font_size_half_pt {
                                let s = h as f32 / 2.0;
                                if !sizes_seen.iter().any(|x| (*x - s).abs() < 0.1) {
                                    sizes_seen.push(s);
                                }
                            }
                        }
                    }
                }
                Element::Heading(h) => {
                    total_heads += 1;
                    total_chars += inline_chars(&h.content);
                }
                _ => {}
            }
        }
    }
    sizes_seen.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "[{tag}] sections={} headings={} paras={} chars={} sizes_pt={:?}",
        ir.sections.len(),
        total_heads,
        total_paras,
        total_chars,
        sizes_seen
    );
}
