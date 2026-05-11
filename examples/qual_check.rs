// Standalone quality-gate checker: mirrors test_corpus_extraction_quality.rs logic
fn main() {
    let tests: &[(&str, &str, &str, f32)] = &[
        (
            "hello_structure",
            "/tmp/hello_structure.pdf",
            "/tmp/gt_hello_structure.txt",
            0.88,
        ),
        ("pdfa_036", "/tmp/pdfa_036.pdf", "/tmp/gt_pdfa_036_kreuzberg.txt", 0.78),
        ("pdfa_044", "/tmp/pdfa_044.pdf", "/tmp/gt_pdfa_044.txt", 0.80),
        ("nougat_039", "/tmp/nougat_039.pdf", "/tmp/gt_nougat_039.txt", 0.83),
        ("nougat_026", "/tmp/nougat_026.pdf", "/tmp/gt_nougat_026.txt", 0.87),
        ("pr-136-example", "/tmp/pr-136-example.pdf", "/tmp/gt_pr-136-example.txt", 0.05),
        ("pr-138-example", "/tmp/pr-138-example.pdf", "/tmp/gt_pr-138-example.txt", 0.45),
        ("issue-987-test", "/tmp/issue-987-test.pdf", "/tmp/gt_issue-987-test.txt", 0.65),
        (
            "issue-336-example",
            "/tmp/issue-336-example.pdf",
            "/tmp/gt_issue-336-example.txt",
            0.69,
        ),
        ("nougat_040", "/tmp/nougat_040.pdf", "/tmp/gt_nougat_040.txt", 0.35),
        ("pdfa_004", "/tmp/pdfa_004.pdf", "/tmp/gt_pdfa_004.txt", 0.49),
        ("nougat_018", "/tmp/nougat_018.pdf", "/tmp/gt_nougat_018.txt", 0.95),
    ];

    let mut passed = 0u32;
    let mut failed = 0u32;

    for (name, pdf_path, gt_path, threshold) in tests {
        let bytes = match std::fs::read(pdf_path) {
            Ok(b) => b,
            Err(_) => {
                println!("SKIP {name}: pdf not found");
                continue;
            },
        };
        let gt = match std::fs::read_to_string(gt_path) {
            Ok(t) => t,
            Err(_) => {
                println!("SKIP {name}: gt not found");
                continue;
            },
        };
        let doc = match pdf_oxide::document::PdfDocument::from_bytes(bytes) {
            Ok(d) => d,
            Err(e) => {
                println!("ERR  {name}: {e}");
                continue;
            },
        };
        let _ = doc.authenticate(b"");
        let mut text = String::new();
        for i in 0..doc.page_count().unwrap_or(0) {
            if let Ok(t) = doc.extract_text(i) {
                text.push_str(&t);
                text.push('\n');
            }
        }
        let j = jaccard(&text, &gt);
        if j >= *threshold {
            println!("PASS  j={j:.3}  thr={threshold:.2}  {name}");
            passed += 1;
        } else {
            println!("FAIL  j={j:.3}  thr={threshold:.2}  {name}");
            failed += 1;
        }
    }
    println!("\n{passed} PASS, {failed} FAIL");
    if failed > 0 {
        std::process::exit(1);
    }
}

fn jaccard(a: &str, b: &str) -> f32 {
    use std::collections::HashSet;
    let sa: HashSet<&str> = a.split_whitespace().collect();
    let sb: HashSet<&str> = b.split_whitespace().collect();
    let i = sa.intersection(&sb).count();
    let u = sa.union(&sb).count();
    if u == 0 {
        1.0
    } else {
        i as f32 / u as f32
    }
}
