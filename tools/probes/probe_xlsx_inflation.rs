/// Probe round-trip page-count inflation on a single source PDF, via XLSX.
///
/// Walks two paths and compares them per-section:
///   A) `pdf_to_ir`              → `ir_to_pdf_bytes_pub`        → PDF
///   B) `to_xlsx_bytes` → parse XLSX → `to_ir` → `convert_xlsx_bytes` → PDF
///
/// Reports element/character counts, font sizes, paragraph/table breakdown
/// and rendered page counts so we can pinpoint the dominant inflation
/// factor in the XLSX hop.
///
/// Usage:
///   cargo run --release --features rendering --example probe_xlsx_inflation -- \
///     /home/yfedoseev/projects/pdf_oxide_tests/pdfs/academic/arxiv_2510.21165v1.pdf
use office_oxide::ir::{DocumentIR, Element, InlineContent};
use office_oxide::{Document, DocumentFormat};
use pdf_oxide::{
    api::Pdf,
    converters::{
        pdf_to_ir::{pdf_to_ir, PdfToIrOptions},
        office::{ir_to_pdf_bytes_pub, OfficeConfig, OfficeConverter},
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

    println!("=== probe_xlsx_inflation: {path} ===");

    // Source PDF.
    let mut src = Pdf::open(&path).expect("open source PDF");
    let src_pages = src.page_count().expect("page count");
    println!("source PDF pages: {src_pages}");
    drop(src);

    // 1) Source PDF → IR_A.
    let pdf_doc = PdfDocument::open(&path).expect("open PdfDocument");
    let opts = PdfToIrOptions::default();
    let ir_a = pdf_to_ir(&pdf_doc, office_oxide::format::DocumentFormat::Xlsx, &opts).expect("pdf_to_ir");
    println!("\n[IR_A] (pdf_to_ir output)");
    summarize(&ir_a, "IR_A");

    // 2a) IR_A → PDF directly.
    let cfg = OfficeConfig::default();
    let bytes_direct = ir_to_pdf_bytes_pub(&ir_a, &cfg).expect("ir_to_pdf direct");
    let direct_pages = count_pdf_pages(&bytes_direct);
    println!("\n[direct] IR_A → PDF pages: {direct_pages}");

    // 2b) IR_A → XLSX bytes → parse → IR_B → PDF (via convert_xlsx_bytes).
    let xlsx_bytes = pdf_doc.to_xlsx_bytes().expect("to_xlsx_bytes");
    println!("\n[forward] to_xlsx_bytes -> {} bytes", xlsx_bytes.len());

    let cursor = std::io::Cursor::new(xlsx_bytes.clone());
    let xlsx_doc = Document::from_reader(cursor, DocumentFormat::Xlsx)
        .expect("parse xlsx");
    let ir_b = xlsx_doc.to_ir();
    println!("\n[IR_B] (to_xlsx_bytes -> parse -> to_ir)");
    summarize(&ir_b, "IR_B");

    // What does is_document_style_xlsx think of IR_B?
    let routing = describe_xlsx_routing(&ir_b);
    println!("\n[routing] {routing}");

    // The official reverse path goes through OfficeConverter::convert_xlsx_bytes.
    let conv = OfficeConverter::new();
    let bytes_rt = conv.convert_xlsx_bytes(&xlsx_bytes).expect("convert_xlsx_bytes");
    let rt_pages = count_pdf_pages(&bytes_rt);
    println!("\n[round-trip via XLSX] -> PDF pages: {rt_pages}");

    // Also render IR_B straight via ir_to_pdf_bytes_pub (forces ir_to_pdf path) for
    // contrast with whatever convert_xlsx_bytes chose.
    let bytes_ir_b_direct = ir_to_pdf_bytes_pub(&ir_b, &cfg).expect("ir_to_pdf IR_B");
    let ir_b_direct_pages = count_pdf_pages(&bytes_ir_b_direct);
    println!("[IR_B → ir_to_pdf_bytes_pub directly] pages: {ir_b_direct_pages}");

    println!(
        "\n=== summary: source={src_pages} direct={direct_pages} rt_xlsx={rt_pages} ir_b_direct={ir_b_direct_pages} ===\
         \n  inflation direct    = {:.2}×\
         \n  inflation rt_xlsx   = {:.2}×",
        direct_pages as f32 / src_pages.max(1) as f32,
        rt_pages as f32 / src_pages.max(1) as f32,
    );

    // Per-section comparison — IR_B for XLSX has 1 section per worksheet.
    println!("\n--- per-section element comparison (first 4) ---");
    for i in 0..ir_a.sections.len().min(ir_b.sections.len()).min(4) {
        let a = &ir_a.sections[i];
        let b = &ir_b.sections[i];
        let a_chars: usize = section_chars(a);
        let b_chars: usize = section_chars(b);
        println!(
            "  sec {i}: A elems={} (h={} p={} t={}) chars={a_chars} body_size={:?}  |  B elems={} (h={} p={} t={}) chars={b_chars} body_size={:?}",
            a.elements.len(),
            count_kind(a, "Heading"),
            count_kind(a, "Paragraph"),
            count_kind(a, "Table"),
            first_paragraph_size(a),
            b.elements.len(),
            count_kind(b, "Heading"),
            count_kind(b, "Paragraph"),
            count_kind(b, "Table"),
            first_paragraph_size(b),
        );
    }

    // Detailed look at section 0 of IR_B — usually the "Sheet1" section
    // containing one big Table for a document-style XLSX.
    if let Some(sec) = ir_b.sections.first() {
        let mut total_cells = 0usize;
        let mut empty_cells = 0usize;
        let mut max_cols = 0usize;
        let mut total_rows = 0usize;
        let mut long_cells = 0usize;
        for el in &sec.elements {
            if let Element::Table(t) = el {
                for row in &t.rows {
                    total_rows += 1;
                    let nc = row.cells.iter().filter(|c| !cell_text(c).is_empty()).count();
                    if nc > max_cols { max_cols = nc; }
                    for c in &row.cells {
                        total_cells += 1;
                        let txt = cell_text(c);
                        if txt.is_empty() { empty_cells += 1; }
                        if txt.chars().count() > 40 { long_cells += 1; }
                    }
                }
            }
        }
        println!(
            "\n[IR_B sec0 table stats] rows={total_rows} max_cols={max_cols} cells={total_cells} (empty={empty_cells}, >40ch={long_cells})"
        );
        // Page setup?
        println!(
            "[IR_B sec0 page_setup] {:?}",
            sec.page_setup.as_ref().map(|p| (p.width_twips, p.height_twips))
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
            Element::Table(t) => {
                for row in &t.rows {
                    for c in &row.cells {
                        total += cell_text(c).chars().count();
                    }
                }
            }
            _ => {}
        }
    }
    total
}

fn cell_text(c: &office_oxide::ir::TableCell) -> String {
    let mut s = String::new();
    for el in &c.content {
        if let Element::Paragraph(p) = el {
            for ic in &p.content {
                if let InlineContent::Text(span) = ic {
                    s.push_str(&span.text);
                }
            }
        }
    }
    s
}

fn count_kind(s: &office_oxide::ir::Section, k: &str) -> usize {
    s.elements
        .iter()
        .filter(|e| match (e, k) {
            (Element::Heading(_), "Heading") => true,
            (Element::Paragraph(_), "Paragraph") => true,
            (Element::Table(_), "Table") => true,
            _ => false,
        })
        .count()
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
        if let Element::Table(t) = el {
            for row in &t.rows {
                for cell in &row.cells {
                    for el in &cell.content {
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
                }
            }
        }
    }
    None
}

fn describe_xlsx_routing(ir: &DocumentIR) -> String {
    let mut max_cols = 0usize;
    let mut long_cells = 0usize;
    for sec in &ir.sections {
        for el in &sec.elements {
            if let Element::Table(t) = el {
                for row in &t.rows {
                    let nc = row.cells.iter().filter(|c| !cell_text(c).is_empty()).count();
                    if nc > max_cols { max_cols = nc; }
                    for c in &row.cells {
                        if cell_text(c).chars().count() > 40 { long_cells += 1; }
                    }
                }
            }
        }
    }
    let route = if max_cols <= 2 && long_cells >= 3 {
        "→ markdown_to_pdf_bytes (document-style)"
    } else {
        "→ ir_to_pdf_bytes / render_table"
    };
    format!("max_cols={max_cols} long_cells={long_cells} route: {route}")
}

fn summarize(ir: &DocumentIR, tag: &str) {
    let mut total_chars = 0usize;
    let mut total_paras = 0usize;
    let mut total_heads = 0usize;
    let mut total_tables = 0usize;
    let mut total_rows = 0usize;
    let mut total_cells = 0usize;
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
                Element::Table(t) => {
                    total_tables += 1;
                    for row in &t.rows {
                        total_rows += 1;
                        for c in &row.cells {
                            total_cells += 1;
                            total_chars += cell_text(c).chars().count();
                            for el in &c.content {
                                if let Element::Paragraph(p) = el {
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
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    sizes_seen.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "[{tag}] sections={} headings={} paras={} tables={} rows={} cells={} chars={} sizes_pt={:?}",
        ir.sections.len(),
        total_heads,
        total_paras,
        total_tables,
        total_rows,
        total_cells,
        total_chars,
        sizes_seen
    );
}
