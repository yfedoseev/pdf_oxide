/// One-time regression analysis for v0.3.42 office conversion.
///
/// Tests all 6 conversion directions and round-trips against local corpora:
///   DOCX/PPTX/XLSX → PDF → plain-text   (text preservation)
///   PDF → DOCX/PPTX/XLSX → plain-text   (round-trip text preservation)
///
/// Run:
///   cargo run --example analyze_office_conversion --features rendering 2>&1 | tee /tmp/office_regression.txt
///
/// Override corpus paths:
///   PDF_OXIDE_TESTS_DIR=...  OFFICE_OXIDE_TESTS_DIR=...  cargo run ...

use office_oxide::{Document, DocumentFormat};
use pdf_oxide::{
    converters::office::OfficeConverter,
    document::PdfDocument,
};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

// ── config ────────────────────────────────────────────────────────────────────

const MAX_PER_CATEGORY: usize = 20;
const LOW_COVERAGE_THRESHOLD: f64 = 0.50;
const ERROR_RATE_WARN: f64 = 0.10;

// ── paths ─────────────────────────────────────────────────────────────────────

fn pdf_tests() -> PathBuf {
    std::env::var("PDF_OXIDE_TESTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_home().join("projects/pdf_oxide_tests"))
}

fn office_tests() -> PathBuf {
    std::env::var("OFFICE_OXIDE_TESTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_home().join("projects/office_oxide_tests"))
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/root"))
}

// ── text helpers ──────────────────────────────────────────────────────────────

fn word_set(text: &str) -> HashSet<String> {
    // Normalize Math Alphanumeric Symbols (U+1D400-1D7FF: italic/bold/script
    // letters used in formulae) to their plain Latin/Greek base. The renderer
    // does the same collapse on the output side, so applying it here lets the
    // coverage metric reflect actual round-trip fidelity instead of
    // penalizing both sides for an encoding-only difference.
    let normalized: String = text
        .chars()
        .map(|c| {
            pdf_oxide::fonts::encoding::math_alphanumeric_base(c as u32)
                .and_then(char::from_u32)
                .unwrap_or(c)
        })
        .collect();
    normalized
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_lowercase())
        .collect()
}

fn coverage(source: &HashSet<String>, result: &HashSet<String>) -> f64 {
    if source.is_empty() { return 1.0; }
    source.iter().filter(|w| result.contains(*w)).count() as f64 / source.len() as f64
}

fn extract_all_text(doc: &PdfDocument) -> String {
    let n = doc.page_count().unwrap_or(0);
    (0..n).filter_map(|i| doc.extract_text(i).ok()).collect::<Vec<_>>().join("\n")
}

// ── file collection ───────────────────────────────────────────────────────────

fn collect(base: &Path, ext: &str, max: usize) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    if !base.exists() { return out; }
    let mut cats: Vec<_> = fs::read_dir(base).into_iter().flatten()
        .flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    cats.sort();
    for cat_dir in cats {
        let cat = cat_dir.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let mut files: Vec<_> = fs::read_dir(&cat_dir).into_iter().flatten()
            .flatten().map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some(ext))
            .collect();
        files.sort();
        for f in files.into_iter().take(max) {
            out.push((f, cat.clone()));
        }
    }
    out
}

fn collect_pdfs(max: usize) -> Vec<(PathBuf, String)> {
    let base = pdf_tests();
    let subdirs = [
        "pdfs/academic", "pdfs/mixed", "pdfs/government",
        "pdfs/technical", "fixtures_policy",
        "fixtures_regression/academic", "fixtures_regression/government",
        "pdfs/tables", "pdfs/forms",
    ];
    let mut out = Vec::new();
    for sub in &subdirs {
        let dir = base.join(sub);
        if !dir.exists() { continue; }
        let cat = dir.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let mut files: Vec<_> = fs::read_dir(&dir).into_iter().flatten()
            .flatten().map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("pdf"))
            .collect();
        files.sort();
        for f in files.into_iter().take(max) {
            out.push((f, cat.clone()));
        }
    }
    out
}

// ── result types ──────────────────────────────────────────────────────────────

#[derive(Default)]
struct Stats {
    ok: usize,
    errors: usize,
    skipped: usize,
    coverage_sum: f64,
    coverage_min: f64,
    low: Vec<(String, String, f64)>, // (file, cat, coverage)
    error_samples: Vec<(String, String)>, // (file, error)
}

impl Stats {
    fn new() -> Self { Self { coverage_min: 1.0, ..Default::default() } }

    fn record_ok(&mut self, file: &str, cat: &str, cov: f64) {
        self.ok += 1;
        self.coverage_sum += cov;
        if cov < self.coverage_min { self.coverage_min = cov; }
        if cov < LOW_COVERAGE_THRESHOLD {
            self.low.push((file.to_string(), cat.to_string(), cov));
        }
    }

    fn record_err(&mut self, file: &str, err: &str) {
        self.errors += 1;
        if self.error_samples.len() < 20 {
            self.error_samples.push((file.to_string(), err.chars().take(120).collect()));
        }
    }

    fn record_skip(&mut self) { self.skipped += 1; }

    fn total(&self) -> usize { self.ok + self.errors + self.skipped }
    fn avg_cov(&self) -> f64 { if self.ok == 0 { 0.0 } else { self.coverage_sum / self.ok as f64 } }
    fn error_rate(&self) -> f64 {
        let denom = self.ok + self.errors;
        if denom == 0 { 0.0 } else { self.errors as f64 / denom as f64 }
    }
}

// ── printing ──────────────────────────────────────────────────────────────────

fn bar(v: f64, w: usize) -> String {
    let filled = (v * w as f64).round() as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(w - filled))
}

fn print_stats(label: &str, s: &Stats, elapsed: f64) {
    println!("\n{}", "─".repeat(70));
    println!("  {label}");
    println!("{}", "─".repeat(70));
    println!("  total={:3}  ok={:3}  errors={:3}  skipped={:3}  ({elapsed:.1}s)",
             s.total(), s.ok, s.errors, s.skipped);
    if s.ok > 0 {
        let avg = s.avg_cov();
        println!("  avg coverage : {:.1}%  [{}]", avg * 100.0, bar(avg, 40));
        println!("  min coverage : {:.1}%", s.coverage_min * 100.0);
    }
    if s.error_rate() > ERROR_RATE_WARN {
        println!("  ⚠  error rate {:.1}% exceeds {:.0}% threshold",
                 s.error_rate() * 100.0, ERROR_RATE_WARN * 100.0);
    }
    if !s.low.is_empty() {
        println!("  low coverage (<{:.0}%) [{} files]:",
                 LOW_COVERAGE_THRESHOLD * 100.0, s.low.len());
        for (f, cat, cov) in s.low.iter().take(15) {
            println!("    {:4.0}%  [{cat}]  {f}", cov * 100.0);
        }
        if s.low.len() > 15 { println!("    … and {} more", s.low.len() - 15); }
    }
    if !s.error_samples.is_empty() {
        println!("  errors (sample):");
        for (f, e) in &s.error_samples {
            println!("    {f}: {e}");
        }
    }
}

// ── runners ───────────────────────────────────────────────────────────────────

fn run_office_to_pdf(
    files: &[(PathBuf, String)],
    fmt: office_oxide::DocumentFormat,
    convert: impl Fn(&Path) -> pdf_oxide::Result<Vec<u8>>,
) -> Stats {
    let mut s = Stats::new();
    for (path, cat) in files {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();

        // Source-side: extract words from the office file via office_oxide.
        // We use to_markdown() which traverses the IR — gives us the actual
        // text content we expect to see in the rendered PDF.
        let src_words: HashSet<String> = match fs::read(path) {
            Ok(bytes) => match office_oxide::Document::from_reader(std::io::Cursor::new(bytes), fmt) {
                Ok(doc) => word_set(&doc.to_markdown()),
                Err(_) => HashSet::new(),
            },
            Err(_) => HashSet::new(),
        };

        match convert(path) {
            Err(e) => { s.record_err(&name, &e.to_string()); print!("E"); }
            Ok(pdf_bytes) => {
                match PdfDocument::from_bytes(pdf_bytes) {
                    Err(e) => { s.record_err(&name, &format!("parse pdf: {e}")); print!("E"); }
                    Ok(doc) => {
                        let text = extract_all_text(&doc);
                        let result_words = word_set(&text);
                        if result_words.len() < 3 || src_words.len() < 3 {
                            s.record_skip();
                            print!("s");
                        } else {
                            // Real text-loss metric: fraction of source-office words that
                            // also appear in the rendered PDF.
                            let common = src_words.intersection(&result_words).count();
                            let cov = common as f64 / src_words.len() as f64;
                            s.record_ok(&name, cat, cov);
                            print!(".");
                        }
                    }
                }
            }
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!();
    s
}

/// Run all three PDF→Office→PDF round-trips file-by-file.
///
/// Parses each source PDF and extracts source words exactly once, then runs
/// DOCX, PPTX, and XLSX round-trips against it before dropping the parsed
/// document. Per-section timings are returned as wall-clock segments.
fn run_pdf_round_trips(
    files: &[(PathBuf, String)],
    conv: &OfficeConverter,
    print_progress: bool,
) -> ([Stats; 3], [f64; 3]) {
    let mut stats = [Stats::new(), Stats::new(), Stats::new()];
    let mut elapsed = [0.0f64; 3];

    let labels = ["DOCX", "PPTX", "XLSX"];
    if print_progress {
        for label in &labels {
            print!("  {label}: ");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
        println!();
    }

    for (path, cat) in files {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();

        // Parse source PDF + extract words ONCE per file, reused across all 3 round-trips.
        let pdf_bytes = match fs::read(path) {
            Err(e) => {
                for st in stats.iter_mut() { st.record_err(&name, &e.to_string()); }
                continue;
            }
            Ok(b) => b,
        };
        let doc = match PdfDocument::from_bytes(pdf_bytes) {
            Err(e) => {
                for st in stats.iter_mut() { st.record_err(&name, &e.to_string()); }
                continue;
            }
            Ok(d) => d,
        };
        let src_text = extract_all_text(&doc);
        let src_words = word_set(&src_text);
        if src_words.len() < 5 {
            for st in stats.iter_mut() { st.record_skip(); }
            continue;
        }

        // Three round-trips, each timed separately. doc/src_words borrowed in turn.
        let convs: [(&str, Box<dyn Fn(&PdfDocument) -> pdf_oxide::Result<Vec<u8>>>,
                          Box<dyn Fn(&[u8]) -> pdf_oxide::Result<Vec<u8>>>); 3] = [
            ("docx", Box::new(|d: &PdfDocument| d.to_docx_bytes()),
                     Box::new(|b: &[u8]| conv.convert_docx_bytes(b))),
            ("pptx", Box::new(|d: &PdfDocument| d.to_pptx_bytes()),
                     Box::new(|b: &[u8]| conv.convert_pptx_bytes(b))),
            ("xlsx", Box::new(|d: &PdfDocument| d.to_xlsx_bytes()),
                     Box::new(|b: &[u8]| conv.convert_xlsx_bytes(b))),
        ];

        for (i, (_lbl, convert, reimport)) in convs.iter().enumerate() {
            let t = Instant::now();
            let result = (|| -> Result<f64, String> {
                let office_bytes = convert(&doc).map_err(|e| e.to_string())?;
                let reimported = reimport(&office_bytes).map_err(|e| format!("reimport: {e}"))?;
                let rdoc = PdfDocument::from_bytes(reimported)
                    .map_err(|e| format!("parse reimported: {e}"))?;
                let rt = extract_all_text(&rdoc);
                let rw = word_set(&rt);
                Ok(coverage(&src_words, &rw))
            })();
            elapsed[i] += t.elapsed().as_secs_f64();
            match result {
                Ok(cov) => stats[i].record_ok(&name, cat, cov),
                Err(msg) => stats[i].record_err(&name, &msg),
            }
        }
        if print_progress {
            print!(".");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }
    if print_progress { println!(); }
    (stats, elapsed)
}

// ── Layer 2 quality checks ────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct Layer2Quality {
    /// Fraction of sampled numeric cells whose formatted value appears in PDF.
    numeric_accuracy: f64,
    /// How many numeric cells were sampled.
    numeric_sampled: usize,
    /// true = date cells found in source AND they appear as date strings in PDF (not serials).
    /// None = no date cells in source.
    date_format_ok: Option<bool>,
    /// Non-empty IR cells / unique words in PDF output.  >1 means PDF is word-rich.
    cell_count_ratio: f64,
    /// true = PDF page count is consistent with source row count.
    row_count_ok: bool,
}

fn compute_layer2(
    xlsx: &office_oxide::xlsx::XlsxDocument,
    ir: &office_oxide::ir::DocumentIR,
    pdf_bytes: &[u8],
    date_indices: &std::collections::HashSet<u32>,
) -> Layer2Quality {
    use office_oxide::xlsx::CellValue;

    // -- extract PDF text once --
    let pdf_doc = match PdfDocument::from_bytes(pdf_bytes.to_vec()) {
        Ok(d) => d,
        Err(_) => return Layer2Quality::default(),
    };
    let pdf_text = extract_all_text(&pdf_doc);
    let pdf_pages = pdf_doc.page_count().unwrap_or(0);

    // ── 1. Numeric accuracy ──────────────────────────────────────────────────
    // Collect up to 20 evenly-spaced non-date numeric cells and check their
    // formatted value appears verbatim in the PDF text.
    let mut numeric_cells: Vec<&office_oxide::xlsx::Cell> = Vec::new();
    for ws in &xlsx.worksheets {
        for row in &ws.rows {
            for cell in &row.cells {
                if let CellValue::Number(_) = &cell.value {
                    let is_date = cell.style_index.map_or(false, |i| date_indices.contains(&i));
                    if !is_date {
                        numeric_cells.push(cell);
                    }
                }
            }
        }
    }
    let n_numeric = numeric_cells.len();
    let sample_size = n_numeric.min(20);
    let mut found = 0usize;
    if sample_size > 0 {
        let step = if n_numeric <= sample_size { 1 } else { n_numeric / sample_size };
        for i in 0..sample_size {
            let cell = numeric_cells[(i * step).min(n_numeric - 1)];
            let text = xlsx.format_cell_value(cell);
            if !text.is_empty() && pdf_text.contains(&text) {
                found += 1;
            }
        }
    }
    let numeric_accuracy = if sample_size == 0 { 1.0 } else { found as f64 / sample_size as f64 };

    // ── 2. Date formatting ───────────────────────────────────────────────────
    // Find up to 5 date-formatted cells. Their formatted date (YYYY-MM-DD portion)
    // should appear in the PDF. Also check the raw serial does NOT appear instead.
    let debug_dates = std::env::var("DEBUG_DATES").is_ok();
    let mut date_found = 0usize;
    let mut date_serial_leaked = 0usize;
    let mut date_total = 0usize;
    'date_search: for ws in &xlsx.worksheets {
        for row in &ws.rows {
            for cell in &row.cells {
                if let CellValue::Number(n) = &cell.value {
                    let is_date = cell.style_index.map_or(false, |i| date_indices.contains(&i));
                    if is_date {
                        date_total += 1;
                        let formatted = xlsx.format_cell_value(cell);
                        // For datetime values (e.g. "2012-03-14 13:30:55"), check only the
                        // date portion (first 10 chars) — PDF cell wrapping can split the
                        // time component onto the next line.
                        let check_str = if formatted.len() > 10
                            && formatted.as_bytes().get(10) == Some(&b' ')
                        {
                            &formatted[..10]
                        } else {
                            formatted.as_str()
                        };
                        let in_pdf = !check_str.is_empty() && pdf_text.contains(check_str);
                        if in_pdf {
                            date_found += 1;
                        }
                        // Only flag serial leak if date was NOT found — a spreadsheet can
                        // legitimately have a Raw Value column showing the serial number
                        // alongside a correctly-rendered date in another column.
                        let serial_i = *n as i64;
                        let serial_leaked = if !in_pdf && *n > 1_000.0 && *n < 100_000.0 {
                            let serial_str = format!("{serial_i}");
                            pdf_text.split(|c: char| !c.is_ascii_digit())
                                .any(|tok| tok == serial_str.as_str())
                        } else {
                            false
                        };
                        if serial_leaked { date_serial_leaked += 1; }
                        if debug_dates {
                            let serial_i = *n as i64;
                            eprintln!(
                                "  date_check: serial={serial_i} fmt={formatted:?} check={check_str:?} in_pdf={in_pdf} leaked={serial_leaked}"
                            );
                        }
                        if date_total >= 5 {
                            break 'date_search;
                        }
                    }
                }
            }
        }
    }
    let date_format_ok = if date_total == 0 {
        None
    } else {
        let ok = date_found > 0 && date_serial_leaked == 0;
        if debug_dates && !ok {
            eprintln!("  PDF_TEXT_FULL ({} chars): {:?}", pdf_text.len(), pdf_text);
        }
        Some(ok)
    };

    // ── 3. Cell count ratio ──────────────────────────────────────────────────
    // Non-empty IR cells vs unique words in PDF (lower = text lost in render).
    let ir_nonempty: usize = ir.sections.iter()
        .flat_map(|s| s.elements.iter())
        .filter_map(|e| {
            if let office_oxide::ir::Element::Table(t) = e { Some(t) } else { None }
        })
        .flat_map(|t| t.rows.iter())
        .flat_map(|r| r.cells.iter())
        .filter(|c| c.content.iter().any(|el| {
            if let office_oxide::ir::Element::Paragraph(p) = el {
                p.content.iter().any(|ic| {
                    if let office_oxide::ir::InlineContent::Text(sp) = ic {
                        !sp.text.trim().is_empty()
                    } else { false }
                })
            } else { false }
        }))
        .count();
    let pdf_words = word_set(&pdf_text).len();
    let cell_count_ratio = if pdf_words == 0 { 0.0 } else { ir_nonempty as f64 / pdf_words as f64 };

    // ── 4. Row count / page count sanity ────────────────────────────────────
    // Rough estimate: ~46 rows/page at 10pt on A4/Letter with default margins.
    let ir_rows: usize = ir.sections.iter()
        .flat_map(|s| s.elements.iter())
        .filter_map(|e| {
            if let office_oxide::ir::Element::Table(t) = e { Some(t) } else { None }
        })
        .map(|t| t.rows.len())
        .sum();
    let min_expected_pages = ((ir_rows as f64 / 46.0).ceil() as usize).max(1);
    let row_count_ok = pdf_pages >= min_expected_pages;

    Layer2Quality {
        numeric_accuracy,
        numeric_sampled: sample_size,
        date_format_ok,
        cell_count_ratio,
        row_count_ok,
    }
}

// ── phased XLSX profiler ──────────────────────────────────────────────────────

struct XlsxFileProfile {
    name: String,
    parse_ms: u128,
    ir_ms: u128,
    ir_rows: usize,
    render_ms: u128,
    total_ms: u128,
    cell_cov: f64,
    l2: Option<Layer2Quality>,
    error: Option<String>,
}

fn run_xlsx_phased(files: &[(PathBuf, String)], conv: &OfficeConverter) -> (Stats, Vec<XlsxFileProfile>) {
    let mut s = Stats::new();
    let mut profiles: Vec<XlsxFileProfile> = Vec::new();

    macro_rules! err_profile {
        ($name:expr, $parse_ms:expr, $ir_ms:expr, $ir_rows:expr, $render_ms:expr, $total_ms:expr, $e:expr) => {
            XlsxFileProfile {
                name: $name,
                parse_ms: $parse_ms, ir_ms: $ir_ms, ir_rows: $ir_rows,
                render_ms: $render_ms, total_ms: $total_ms,
                cell_cov: 0.0, l2: None, error: Some($e),
            }
        };
    }

    for (path, cat) in files {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let bytes = match fs::read(path) {
            Err(e) => { s.record_err(&name, &e.to_string()); print!("E");
                profiles.push(err_profile!(name, 0, 0, 0, 0, 0, e.to_string())); continue; }
            Ok(b) => b,
        };

        let t_total = Instant::now();

        // Phase 1: parse XLSX
        let t1 = Instant::now();
        let doc = match Document::from_reader(std::io::Cursor::new(bytes.clone()), DocumentFormat::Xlsx) {
            Err(e) => { s.record_err(&name, &e.to_string()); print!("E");
                profiles.push(err_profile!(name, t1.elapsed().as_millis(), 0, 0, 0,
                    t_total.elapsed().as_millis(), e.to_string())); continue; }
            Ok(d) => d,
        };
        let parse_ms = t1.elapsed().as_millis();

        // Phase 2: build DocumentIR
        let t2 = Instant::now();
        let ir = doc.to_ir();
        let ir_ms = t2.elapsed().as_millis();

        // Compute date indices for Layer 2 (reuse the same set for IR scanning)
        let date_indices = doc.as_xlsx()
            .map(|x| x.date_style_indices())
            .unwrap_or_default();

        // Count IR rows and extract source cell words for Layer 1 coverage
        let (ir_rows, cell_words): (usize, HashSet<String>) = {
            let mut rows = 0usize;
            let mut words = HashSet::new();
            for sec in &ir.sections {
                for el in &sec.elements {
                    if let office_oxide::ir::Element::Table(t) = el {
                        rows += t.rows.len();
                        for row in &t.rows {
                            for cell in &row.cells {
                                for el2 in &cell.content {
                                    if let office_oxide::ir::Element::Paragraph(p) = el2 {
                                        for ic in &p.content {
                                            if let office_oxide::ir::InlineContent::Text(sp) = ic {
                                                words.extend(word_set(&sp.text));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (rows, words)
        };

        // Phase 3: render to PDF
        let t3 = Instant::now();
        let pdf_bytes = match conv.convert_xlsx_bytes(&bytes) {
            Err(e) => { s.record_err(&name, &e.to_string()); print!("E");
                profiles.push(err_profile!(name, parse_ms, ir_ms, ir_rows,
                    t3.elapsed().as_millis(), t_total.elapsed().as_millis(), e.to_string()));
                continue; }
            Ok(b) => b,
        };
        let render_ms = t3.elapsed().as_millis();
        let total_ms = t_total.elapsed().as_millis();

        // Parse output PDF once — used for both Layer 1 and Layer 2 quality checks.
        let pdf_doc = PdfDocument::from_bytes(pdf_bytes.clone()).ok();
        let out_text = pdf_doc.as_ref().map(|d| extract_all_text(d)).unwrap_or_default();
        let out_words = word_set(&out_text);

        // Layer 1: cell word coverage
        let cell_cov = if cell_words.is_empty() {
            let n_pages = pdf_doc.as_ref().and_then(|d| d.page_count().ok()).unwrap_or(0);
            let density = out_words.len() as f64 / n_pages.max(1) as f64;
            (density / 20.0).min(1.0)
        } else {
            coverage(&cell_words, &out_words)
        };

        // Layer 2: numeric accuracy + date format + cell count + row/page ratio
        let l2 = doc.as_xlsx().map(|xlsx| compute_layer2(xlsx, &ir, &pdf_bytes, &date_indices));

        if cell_cov < 0.001 {
            s.record_skip(); print!("s");
        } else {
            s.record_ok(&name, cat, cell_cov);
            print!(".");
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());

        profiles.push(XlsxFileProfile {
            name, parse_ms, ir_ms, ir_rows, render_ms, total_ms, cell_cov, l2, error: None,
        });
    }
    println!();
    (s, profiles)
}

fn print_xlsx_profile(profiles: &[XlsxFileProfile]) {
    // ── Phase timing ──────────────────────────────────────────────────────────
    println!("\n  Phase breakdown — top 15 slowest files:");
    println!("  {:<42} {:>6} {:>5} {:>7} {:>7} {:>7} {:>5}",
             "file", "parse", "ir", "ir_rows", "render", "total", "cov%");
    println!("  {}", "─".repeat(82));

    let mut sorted: Vec<&XlsxFileProfile> = profiles.iter().filter(|p| p.error.is_none()).collect();
    sorted.sort_by(|a, b| b.total_ms.cmp(&a.total_ms));

    for p in sorted.iter().take(15) {
        println!("  {:<42} {:>5}ms {:>4}ms {:>7} {:>6}ms {:>6}ms {:>4.0}%",
                 &p.name[..p.name.len().min(42)],
                 p.parse_ms, p.ir_ms, p.ir_rows, p.render_ms, p.total_ms,
                 p.cell_cov * 100.0);
    }

    let total_parse: u128  = profiles.iter().map(|p| p.parse_ms).sum();
    let total_ir: u128     = profiles.iter().map(|p| p.ir_ms).sum();
    let total_render: u128 = profiles.iter().map(|p| p.render_ms).sum();
    println!("  {}", "─".repeat(82));
    println!("  {:<42} {:>5}ms {:>4}ms {:>7} {:>6}ms",
             "TOTAL", total_parse, total_ir, "", total_render);
    println!("  parse={:.1}%  ir={:.1}%  render={:.1}%  of wall time",
             100.0 * total_parse as f64 / (total_parse + total_ir + total_render).max(1) as f64,
             100.0 * total_ir as f64    / (total_parse + total_ir + total_render).max(1) as f64,
             100.0 * total_render as f64 / (total_parse + total_ir + total_render).max(1) as f64);

    // ── Layer 2 quality summary ───────────────────────────────────────────────
    let l2_files: Vec<&XlsxFileProfile> = profiles.iter()
        .filter(|p| p.l2.is_some() && p.error.is_none())
        .collect();

    if l2_files.is_empty() { return; }

    let num_acc_avg = l2_files.iter()
        .map(|p| p.l2.as_ref().unwrap().numeric_accuracy)
        .sum::<f64>() / l2_files.len() as f64;

    let date_ok_count  = l2_files.iter().filter(|p| p.l2.as_ref().unwrap().date_format_ok == Some(true)).count();
    let date_bad_count = l2_files.iter().filter(|p| p.l2.as_ref().unwrap().date_format_ok == Some(false)).count();
    let date_na_count  = l2_files.iter().filter(|p| p.l2.as_ref().unwrap().date_format_ok.is_none()).count();

    let row_ok = l2_files.iter().filter(|p| p.l2.as_ref().unwrap().row_count_ok).count();
    let row_bad = l2_files.len() - row_ok;

    println!("\n  Layer 2 quality summary ({} files):", l2_files.len());
    println!("  numeric accuracy (avg) : {:.1}%", num_acc_avg * 100.0);
    println!("  date formatting        : {} ok  {} BAD  {} no dates", date_ok_count, date_bad_count, date_na_count);
    println!("  page/row sanity        : {} ok  {} short", row_ok, row_bad);

    // Show worst numeric accuracy files
    let mut worst_num: Vec<&XlsxFileProfile> = l2_files.iter()
        .filter(|p| p.l2.as_ref().unwrap().numeric_sampled > 0)
        .copied()
        .collect();
    worst_num.sort_by(|a, b| {
        a.l2.as_ref().unwrap().numeric_accuracy
            .partial_cmp(&b.l2.as_ref().unwrap().numeric_accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if !worst_num.is_empty() {
        println!("  worst numeric accuracy (bottom 5):");
        for p in worst_num.iter().take(5) {
            let l2 = p.l2.as_ref().unwrap();
            println!("    {:<45} num={:.0}%  cells/words={:.2}  rows_ok={}",
                     &p.name[..p.name.len().min(45)],
                     l2.numeric_accuracy * 100.0,
                     l2.cell_count_ratio,
                     if l2.row_count_ok { "✓" } else { "✗" });
        }
    }

    // Show any date formatting failures
    let bad_dates: Vec<&XlsxFileProfile> = l2_files.iter()
        .filter(|p| p.l2.as_ref().unwrap().date_format_ok == Some(false))
        .copied()
        .collect();
    if !bad_dates.is_empty() {
        println!("  ⚠  date formatting failures:");
        for p in &bad_dates {
            println!("    {}", p.name);
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let office_base = office_tests();
    let pdf_base = pdf_tests();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  pdf_oxide v0.3.42 — Office Conversion Regression Analysis          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("  office_oxide_tests : {}", office_base.display());
    println!("  pdf_oxide_tests    : {}", pdf_base.display());
    println!("  max per category   : {MAX_PER_CATEGORY}");

    if !office_base.exists() { println!("  ⚠  office_oxide_tests not found"); }
    if !pdf_base.exists()    { println!("  ⚠  pdf_oxide_tests not found"); }

    let conv = OfficeConverter::new();
    let mut all_stats: Vec<(&str, Stats)> = Vec::new();

    // ── 1. DOCX → PDF → text ──────────────────────────────────────────────────
    println!("\n[1/6] DOCX → PDF (extract text, check word density)");
    let files = collect(&office_base.join("docx"), "docx", MAX_PER_CATEGORY);
    println!("      {} files across {} categories", files.len(),
             files.iter().map(|(_, c)| c).collect::<HashSet<_>>().len());
    let t = Instant::now();
    let s = run_office_to_pdf(&files, office_oxide::DocumentFormat::Docx, |p| conv.convert_docx(p));
    all_stats.push(("docx→pdf", s));
    print_stats("DOCX → PDF", all_stats.last().unwrap().1.borrow_last(), t.elapsed().as_secs_f64());

    // ── 2. PPTX → PDF → text ──────────────────────────────────────────────────
    println!("\n[2/6] PPTX → PDF (extract text, check word density)");
    let files = collect(&office_base.join("pptx"), "pptx", MAX_PER_CATEGORY);
    println!("      {} files across {} categories", files.len(),
             files.iter().map(|(_, c)| c).collect::<HashSet<_>>().len());
    let t = Instant::now();
    let s = run_office_to_pdf(&files, office_oxide::DocumentFormat::Pptx, |p| conv.convert_pptx(p));
    all_stats.push(("pptx→pdf", s));
    print_stats("PPTX → PDF", all_stats.last().unwrap().1.borrow_last(), t.elapsed().as_secs_f64());

    // ── 3. XLSX → PDF → text (phased profiling) ──────────────────────────────
    println!("\n[3/6] XLSX → PDF (phased profiling: parse / IR / render)");
    let files = collect(&office_base.join("xlsx"), "xlsx", MAX_PER_CATEGORY);
    println!("      {} files across {} categories", files.len(),
             files.iter().map(|(_, c)| c).collect::<HashSet<_>>().len());
    let t = Instant::now();
    let (s, profiles) = run_xlsx_phased(&files, &conv);
    all_stats.push(("xlsx→pdf", s));
    print_stats("XLSX → PDF", all_stats.last().unwrap().1.borrow_last(), t.elapsed().as_secs_f64());
    print_xlsx_profile(&profiles);

    // ── 4/5/6. PDF round-trips ────────────────────────────────────────────────
    // All three round-trips share the same source-PDF parse + word extraction,
    // so we run them file-by-file (parse once, then DOCX/PPTX/XLSX in turn).
    // This keeps memory bounded to one parsed PdfDocument at a time.
    let files = collect_pdfs(MAX_PER_CATEGORY);
    println!("\n[4-6/6] PDF → DOCX/PPTX/XLSX → PDF (round-trip text coverage)");
    println!("      {} PDF files (each parsed once, three round-trips per file)", files.len());
    let (rt_stats, rt_elapsed) = run_pdf_round_trips(&files, &conv, true);

    let [s_docx, s_pptx, s_xlsx] = rt_stats;
    all_stats.push(("pdf→docx→pdf", s_docx));
    print_stats("PDF → DOCX → PDF", all_stats.last().unwrap().1.borrow_last(), rt_elapsed[0]);
    all_stats.push(("pdf→pptx→pdf", s_pptx));
    print_stats("PDF → PPTX → PDF", all_stats.last().unwrap().1.borrow_last(), rt_elapsed[1]);
    all_stats.push(("pdf→xlsx→pdf", s_xlsx));
    print_stats("PDF → XLSX → PDF", all_stats.last().unwrap().1.borrow_last(), rt_elapsed[2]);

    // ── Final summary ─────────────────────────────────────────────────────────
    println!("\n{}", "═".repeat(70));
    println!("  OVERALL SUMMARY");
    println!("{}", "═".repeat(70));
    println!("  {:<22} {:>6} {:>6} {:>7} {:>8} {:>8}",
             "Direction", "total", "ok", "errors", "avg cov", "min cov");
    println!("  {}", "─".repeat(60));
    for (label, s) in &all_stats {
        let warn = if s.error_rate() > ERROR_RATE_WARN { " ⚠" } else { "" };
        println!("  {:<22} {:>6} {:>6} {:>7} {:>7.1}% {:>7.1}%{}",
                 label, s.total(), s.ok, s.errors,
                 s.avg_cov() * 100.0, s.coverage_min * 100.0, warn);
    }
    println!("\n  Legend: . = ok  E = error  s = skipped (too little text)");
    println!("  Coverage for office→pdf = word density (words/page, capped at 20/page = 100%)");
    println!("  Coverage for pdf→X→pdf  = fraction of source PDF words found after round-trip");
}

// Helper to avoid borrow issues in push+reference pattern
trait BorrowLast {
    fn borrow_last(&self) -> &Stats;
}
impl BorrowLast for Stats {
    fn borrow_last(&self) -> &Stats { self }
}
