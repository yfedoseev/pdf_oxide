//! Print page counts for the round-trip output PDFs.
use pdf_oxide::api::Pdf;

fn main() {
    let cases = [
        (
            "arxiv_2510.21165v1",
            "/home/yfedoseev/projects/pdf_oxide_fixes2/output/pdf_to_office/arxiv_2510.21165v1",
            8usize,
        ),
        (
            "arxiv_2510.21368v1",
            "/home/yfedoseev/projects/pdf_oxide_fixes2/output/pdf_to_office/arxiv_2510.21368v1",
            134,
        ),
        (
            "CFR_Title07",
            "/home/yfedoseev/projects/pdf_oxide_fixes2/output/pdf_to_office/CFR_2024_Title07_Vol1_Agriculture",
            660,
        ),
    ];
    for (name, dir, src) in cases {
        let docx = format!("{dir}/via_docx_back_to.pdf");
        let pptx = format!("{dir}/via_pptx_back_to.pdf");
        let xlsx = format!("{dir}/via_xlsx_back_to.pdf");
        let p = |path: &str| -> usize {
            match Pdf::open(path) {
                Ok(mut p) => p.page_count().unwrap_or(0),
                Err(_) => 0,
            }
        };
        let dp = p(&docx);
        let pp = p(&pptx);
        let xp = p(&xlsx);
        println!(
            "{name}: src={src}  docx={dp} ({:.2}×)  pptx={pp} ({:.2}×)  xlsx={xp} ({:.2}×)",
            dp as f32 / src as f32,
            pp as f32 / src as f32,
            xp as f32 / src as f32,
        );
    }
}
