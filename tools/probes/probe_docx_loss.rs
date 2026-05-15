/// Probe content-loss in PDF→DOCX→PDF round-trip via the IR (pdf_to_ir)
/// forward path — the candidate replacement for the markdown route used by
/// `PdfDocument::to_docx_bytes` today.
///
/// Stages logged for each probe source:
///   1. Source PDF page count.
///   2. IR_A from `pdf_to_ir`: section count, total chars, kinds histogram.
///   3. DOCX bytes via `ir_to_docx`: file size; `word/document.xml` dumped
///      to /tmp/probe_docx_<base>.xml; counts of <w:p>, <w:sectPr>,
///      <w:t>-text chars, <w:br w:type="page"> breaks.
///   4. IR_B from `Document::from_reader(Docx).to_ir()`: section count,
///      total chars, kinds histogram.
///   5. Markdown from `doc.to_markdown()`: total chars, line count, ##
///      heading count, ### heading count, paragraph block count.
///   6. Round-trip PDF page count via the production `convert_docx_bytes`
///      (markdown render route) AND via direct ir_to_pdf rendering for
///      contrast.
///
/// Usage:
///   cargo run --release --features rendering --example probe_docx_loss -- \
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
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let default_paths = [
        "/home/yfedoseev/projects/pdf_oxide_tests/pdfs/academic/arxiv_2510.21165v1.pdf",
        "/home/yfedoseev/projects/pdf_oxide_tests/pdfs/academic/arxiv_2510.21368v1.pdf",
    ];
    let paths: Vec<String> = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        default_paths.iter().map(|s| s.to_string()).collect()
    };

    for path in &paths {
        probe_one(path);
        println!("\n========================================================\n");
    }
}

fn probe_one(path: &str) {
    println!("=== probe_docx_loss: {path} ===");

    // Stage 1: source page count.
    let mut src = Pdf::open(path).expect("open source PDF");
    let src_pages = src.page_count().expect("page count");
    println!("source PDF pages: {src_pages}");
    drop(src);

    // Stage 2: IR_A from pdf_to_ir.
    let pdf_doc = PdfDocument::open(path).expect("open PdfDocument");
    let opts = PdfToIrOptions::default();
    let ir_a = pdf_to_ir(&pdf_doc, office_oxide::format::DocumentFormat::Docx, &opts).expect("pdf_to_ir");
    println!("\n[IR_A] (pdf_to_ir output)");
    summarize(&ir_a, "IR_A");

    // Stage 3: build DOCX via ir_to_docx (the candidate forward path),
    // dump document.xml and analyse.
    let writer = office_oxide::create::ir_to_docx(&ir_a);
    let mut docx_buf = std::io::Cursor::new(Vec::new());
    writer.write_to(&mut docx_buf).expect("ir_to_docx write");
    let docx_bytes = docx_buf.into_inner();
    println!("\n[forward] ir_to_docx -> {} bytes", docx_bytes.len());

    let base = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("probe");
    let xml_path = format!("/tmp/probe_docx_{base}.xml");
    let inner_xml = extract_inner(&docx_bytes, "word/document.xml")
        .expect("extract document.xml");
    std::fs::write(&xml_path, &inner_xml).expect("dump xml");
    println!("[forward] document.xml -> {xml_path} ({} bytes)", inner_xml.len());

    let xml_str = std::str::from_utf8(&inner_xml).unwrap_or("");
    let n_p = xml_str.matches("<w:p>").count() + xml_str.matches("<w:p ").count();
    let n_sectpr = xml_str.matches("<w:sectPr>").count() + xml_str.matches("<w:sectPr ").count();
    let n_pgbr = xml_str.matches("<w:br w:type=\"page\"/>").count()
        + xml_str.matches("<w:br w:type=\"page\" />").count()
        + xml_str.matches("w:type=\"page\"").count();
    let total_t_chars = sum_w_t_text_len(xml_str);
    println!(
        "[forward] document.xml stats: <w:p>={n_p} <w:sectPr>={n_sectpr} pgbr={n_pgbr} chars(<w:t>)={total_t_chars}"
    );

    // Stage 4: parse DOCX bytes back -> IR_B.
    let cursor = std::io::Cursor::new(docx_bytes.clone());
    let docx_doc = Document::from_reader(cursor, DocumentFormat::Docx)
        .expect("parse docx");
    let ir_b = docx_doc.to_ir();
    println!("\n[IR_B] (parse DOCX -> to_ir)");
    summarize(&ir_b, "IR_B");

    // Stage 5: markdown.
    let md = docx_doc.to_markdown();
    let md_lines = md.lines().count();
    let md_chars = md.chars().count();
    let h1 = md.lines().filter(|l| l.starts_with("# ")).count();
    let h2 = md.lines().filter(|l| l.starts_with("## ")).count();
    let h3 = md.lines().filter(|l| l.starts_with("### ")).count();
    // Paragraph blocks: blank-line separated chunks.
    let mut blocks = 0usize;
    let mut in_block = false;
    for line in md.lines() {
        if line.trim().is_empty() {
            in_block = false;
        } else if !in_block {
            blocks += 1;
            in_block = true;
        }
    }
    println!(
        "\n[markdown] chars={md_chars} lines={md_lines} h1={h1} h2={h2} h3={h3} blocks={blocks}"
    );
    let md_path = format!("/tmp/probe_docx_{base}.md");
    std::fs::write(&md_path, &md).expect("dump md");
    println!("[markdown] dumped -> {md_path}");

    // Stage 6: round-trip PDF page count.
    let conv = OfficeConverter::new();
    let bytes_rt = conv.convert_docx_bytes(&docx_bytes).expect("convert_docx_bytes");
    let rt_pages = count_pdf_pages(&bytes_rt);
    println!("\n[round-trip via DOCX, markdown render] -> PDF pages: {rt_pages}");

    // Direct: IR_B → ir_to_pdf_bytes_pub.
    let cfg = OfficeConfig::default();
    let bytes_b_direct = ir_to_pdf_bytes_pub(&ir_b, &cfg).expect("ir_to_pdf IR_B");
    let b_direct_pages = count_pdf_pages(&bytes_b_direct);
    println!("[IR_B → ir_to_pdf_bytes_pub directly] -> PDF pages: {b_direct_pages}");

    // Direct: IR_A → ir_to_pdf_bytes_pub (sanity baseline).
    let bytes_a_direct = ir_to_pdf_bytes_pub(&ir_a, &cfg).expect("ir_to_pdf IR_A");
    let a_direct_pages = count_pdf_pages(&bytes_a_direct);
    println!("[IR_A → ir_to_pdf_bytes_pub directly] -> PDF pages: {a_direct_pages}");

    println!(
        "\n=== summary {base}: source={src_pages} IR_A.sections={} IR_B.sections={} \
         rt_md={rt_pages} ir_b_direct={b_direct_pages} ir_a_direct={a_direct_pages} ===",
        ir_a.sections.len(),
        ir_b.sections.len(),
    );

    // Per-section comparison.
    println!("\n--- per-section comparison (first 6) ---");
    for i in 0..ir_a.sections.len().max(ir_b.sections.len()).min(6) {
        let a = ir_a.sections.get(i);
        let b = ir_b.sections.get(i);
        let (ae, ah, ap, ac) = a.map(|s| {
            (s.elements.len(), count_kind(s, "Heading"), count_kind(s, "Paragraph"), section_chars(s))
        }).unwrap_or((0, 0, 0, 0));
        let (be, bh, bp, bc) = b.map(|s| {
            (s.elements.len(), count_kind(s, "Heading"), count_kind(s, "Paragraph"), section_chars(s))
        }).unwrap_or((0, 0, 0, 0));
        println!(
            "  sec {i}: A elems={ae}(h={ah} p={ap}) chars={ac}  |  B elems={be}(h={bh} p={bp}) chars={bc}"
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

fn extract_inner(zip_bytes: &[u8], part: &str) -> Option<Vec<u8>> {
    // Shell out to `unzip -p` rather than pulling in a zip crate.
    let tmp = std::env::temp_dir().join("probe_docx_loss.docx");
    std::fs::write(&tmp, zip_bytes).ok()?;
    let out = std::process::Command::new("unzip")
        .args(["-p", tmp.to_str()?, part])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout)
}

fn sum_w_t_text_len(xml: &str) -> usize {
    // Naive scan of every <w:t...>...</w:t> body length. Good enough as a
    // proxy for "how much text survived into document.xml".
    let mut total = 0usize;
    let mut rest = xml;
    loop {
        let Some(open) = rest.find("<w:t") else { break };
        let after_open = &rest[open..];
        // Skip "<w:t" + (attrs?) + ">"
        let Some(gt) = after_open.find('>') else { break };
        let body_start = open + gt + 1;
        let Some(close_rel) = rest[body_start..].find("</w:t>") else { break };
        let body = &rest[body_start..body_start + close_rel];
        total += body.chars().count();
        rest = &rest[body_start + close_rel + "</w:t>".len()..];
    }
    total
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
                        for el in &c.content {
                            if let Element::Paragraph(p) = el {
                                total += inline_chars(&p.content);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    total
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

fn summarize(ir: &DocumentIR, tag: &str) {
    let mut total_chars = 0usize;
    let mut total_paras = 0usize;
    let mut total_heads = 0usize;
    let mut total_tables = 0usize;
    let mut total_rows = 0usize;
    let mut sizes_seen: Vec<f32> = Vec::new();
    let mut sections_with_pgsz = 0usize;

    for sec in &ir.sections {
        if sec.page_setup.is_some() {
            sections_with_pgsz += 1;
        }
        for el in &sec.elements {
            match el {
                Element::Paragraph(p) => {
                    total_paras += 1;
                    total_chars += inline_chars(&p.content);
                    record_sizes(&p.content, &mut sizes_seen);
                }
                Element::Heading(h) => {
                    total_heads += 1;
                    total_chars += inline_chars(&h.content);
                    record_sizes(&h.content, &mut sizes_seen);
                }
                Element::Table(t) => {
                    total_tables += 1;
                    for row in &t.rows {
                        total_rows += 1;
                        for c in &row.cells {
                            for el in &c.content {
                                if let Element::Paragraph(p) = el {
                                    total_chars += inline_chars(&p.content);
                                    record_sizes(&p.content, &mut sizes_seen);
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
        "[{tag}] sections={} (with page_setup={sections_with_pgsz}) headings={total_heads} paras={total_paras} tables={total_tables} rows={total_rows} chars={total_chars} sizes_pt={:?}",
        ir.sections.len(),
        sizes_seen
    );
}

fn record_sizes(content: &[InlineContent], out: &mut Vec<f32>) {
    for ic in content {
        if let InlineContent::Text(span) = ic {
            if let Some(h) = span.font_size_half_pt {
                let s = h as f32 / 2.0;
                if !out.iter().any(|x| (*x - s).abs() < 0.1) {
                    out.push(s);
                }
            }
        }
    }
}
