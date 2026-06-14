//! The per-paint coverage rasterisers (`rasterise_fill_coverage` /
//! `rasterise_stroke_coverage`) return a page-indexed coverage buffer to the
//! after-paint consumers. That buffer used to be a fresh `vec![0u8; w*h]`
//! zero-allocated on *every* paint operator — on a large page with thousands
//! of small paints (and an active CMYK sidecar) that per-paint full-page
//! `memset` dominated render time (profiling: ~73% of a worst-case page).
//!
//! The consumers read coverage only inside the paint's device rect, so the
//! buffer can be reused across paints — overwrite the rect, leave the rest
//! (stale but never read). These probes pin that reuse via the
//! `coverage_result_alloc_count` test-support counter: N paints must allocate
//! O(1) buffers, not O(N).

#![cfg(all(feature = "rendering", feature = "icc", feature = "test-support"))]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{PageRenderer, RenderOptions};

/// Single-page 100×100pt PDF with an OutputIntent CMYK profile (object 5).
/// The OutputIntent + a transparency trigger make the renderer allocate the
/// CMYK sidecar, so opaque CMYK paints route through the coverage rasterisers.
fn build_pdf_with_output_intent(content: &str, resources_inner: &str, icc: &[u8]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic CMYK) /DestOutputProfile 5 0 R >>] >>\nendobj\n",
    );
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    let page = format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << {} >> /Contents 4 0 R >>\nendobj\n",
        resources_inner
    );
    buf.extend_from_slice(page.as_bytes());
    let stream_off = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_off = buf.len();
    buf.extend_from_slice(
        format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", icc.len()).as_bytes(),
    );
    buf.extend_from_slice(icc);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, icc_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Minimal constant-Lab mft1 CMYK→Lab ICC profile so the sidecar's ICC path
/// has a valid profile (mirrors the sidecar-mirror dirty-rect helper).
fn build_constant_cmyk_icc(l_byte: u8) -> Vec<u8> {
    let (in_chan, out_chan, grid): (u8, u8, u8) = (4, 3, 2);
    let mut lut = Vec::new();
    lut.extend_from_slice(&0x6d66_7431u32.to_be_bytes());
    lut.extend_from_slice(&0u32.to_be_bytes());
    lut.push(in_chan);
    lut.push(out_chan);
    lut.push(grid);
    lut.push(0);
    let identity: [i32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x0001_0000];
    for v in identity {
        lut.extend_from_slice(&(v as u32).to_be_bytes());
    }
    for _ in 0..in_chan {
        for i in 0..256u16 {
            lut.push(i as u8);
        }
    }
    for _ in 0..(grid as usize).pow(in_chan as u32) {
        lut.push(l_byte);
        lut.push(128);
        lut.push(128);
    }
    for _ in 0..out_chan {
        for i in 0..256u16 {
            lut.push(i as u8);
        }
    }
    let mut profile = vec![0u8; 128];
    let total: u32 = 128 + 4 + 12 + lut.len() as u32;
    profile[0..4].copy_from_slice(&total.to_be_bytes());
    profile[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes());
    profile[12..16].copy_from_slice(b"prtr");
    profile[16..20].copy_from_slice(b"CMYK");
    profile[20..24].copy_from_slice(b"Lab ");
    profile[36..40].copy_from_slice(b"acsp");
    profile.extend_from_slice(&1u32.to_be_bytes());
    profile.extend_from_slice(&0x4132_4230u32.to_be_bytes());
    profile.extend_from_slice(&144u32.to_be_bytes());
    profile.extend_from_slice(&(lut.len() as u32).to_be_bytes());
    profile.extend_from_slice(&lut);
    profile
}

/// Constant alpha < 1 in an (unused) ExtGState flips the sidecar-allocation
/// gate without being applied to the paints below, so they stay opaque and
/// route through the coverage rasterisers.
const TRANSPARENCY_GS: &str = "/ExtGState << /GS1 << /ca 0.5 >> >>";

fn alloc_count_for(content: &str) -> u64 {
    let icc = build_constant_cmyk_icc(128);
    let pdf = build_pdf_with_output_intent(content, TRANSPARENCY_GS, &icc);
    let doc = PdfDocument::from_bytes(pdf).expect("synthetic OutputIntent PDF must parse");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72));
    renderer.render_page(&doc, 0).expect("render");
    renderer.coverage_result_alloc_count()
}

/// Twelve small DeviceCMYK fills on a CMYK-OutputIntent page each route
/// through `rasterise_fill_coverage`. The returned coverage buffer must be
/// reused across paints (allocated once), not zero-allocated per fill.
#[test]
fn fill_coverage_buffer_reused_across_paints() {
    let mut content = String::new();
    for i in 0..12u32 {
        let x = 5 + i * 2;
        content.push_str(&format!("0 0.5 0 0 k {x} {x} 8 8 re f\n"));
    }
    let allocs = alloc_count_for(&content);
    assert!(allocs >= 1, "at least one fill-coverage buffer must be allocated; got {allocs}");
    assert!(
        allocs <= 1,
        "fill-coverage buffer must be reused across 12 fills (expected 1 allocation, got {allocs}); \
         a per-paint full-page zero-allocation was the worst-case render hotspot"
    );
}

/// Twelve small CMYK strokes route through `rasterise_stroke_coverage`; same
/// reuse contract on the independent stroke-coverage buffer.
#[test]
fn stroke_coverage_buffer_reused_across_paints() {
    let mut content = String::from("2 w\n");
    for i in 0..12u32 {
        let x = 5 + i * 2;
        let x2 = x + 8;
        content.push_str(&format!("0 0.5 0 0 K {x} {x} m {x2} {x2} l S\n"));
    }
    let allocs = alloc_count_for(&content);
    assert!(
        allocs >= 1,
        "at least one stroke-coverage buffer must be allocated; got {allocs}"
    );
    assert!(
        allocs <= 1,
        "stroke-coverage buffer must be reused across 12 strokes (expected 1 allocation, got {allocs})"
    );
}
