//! Dirty-rect bounding for the spot/RGB/CMYK sidecar-mirror coverage
//! passes (`mirror_{rgb,spot,cmyk}_paint_into_sidecar_with_coverage`).
//!
//! Those passes mirror each paint into the per-page CMYK/spot sidecar by
//! walking a page-sized coverage plane. On ICC/spot-heavy artwork with many
//! small paints, re-walking the full plane per paint dominated render CPU
//! (corpus profiling: ~39% combined self time). The paint's coverage is
//! zero outside its device bbox, so bounding the walk to that bbox is
//! byte-identical — these probes pin the bounding contract via the
//! `sidecar_mirror_scanned_pixels` test-support counter.

#![cfg(all(feature = "rendering", feature = "icc", feature = "test-support"))]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{PageRenderer, RenderOptions};

/// Single-page 100×100pt PDF with an OutputIntent CMYK profile (object 5),
/// the given content + resources. The OutputIntent makes the renderer
/// allocate the CMYK sidecar, so RGB/CMYK paints route through the mirror.
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

/// Minimal constant-Lab mft1 CMYK→Lab ICC profile (mirrors the round-4
/// overprint-spec helper) so the sidecar's ICC path has a valid profile.
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

/// The CMYK sidecar is allocated only when the page BOTH has an OutputIntent
/// CMYK profile AND declares transparency/overprint in resources. Declaring
/// an ExtGState with constant alpha < 1 flips that gate without being applied
/// to the fill below, so the fill stays opaque RGB/CMYK and routes through the
/// RGB/CMYK mirror (rather than the transparency/overprint paths).
const TRANSPARENCY_GS: &str = "/ExtGState << /GS1 << /ca 0.5 >> >>";

fn mirror_scan_for(content: &str, resources_extra: &str) -> (u64, u64) {
    let resources = format!("{TRANSPARENCY_GS} {resources_extra}");
    let icc = build_constant_cmyk_icc(128);
    let pdf = build_pdf_with_output_intent(content, &resources, &icc);
    let doc = PdfDocument::from_bytes(pdf).expect("synthetic OutputIntent PDF must parse");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72));
    let img = renderer.render_page(&doc, 0).expect("render");
    let total = (img.width as u64) * (img.height as u64);
    (renderer.sidecar_mirror_scanned_pixels(), total)
}

/// A small 20×20pt DeviceRGB fill on a CMYK-OutputIntent page routes
/// through `mirror_rgb_paint_into_sidecar_with_coverage`. The mirror walk
/// must be bounded to the paint's device rect, not the full plane.
#[test]
fn small_rgb_fill_mirror_scan_is_rect_bounded() {
    let (scanned, total) = mirror_scan_for("0.2 0.4 0.6 rg 10 10 20 20 re f", "");
    assert!(scanned > 0, "RGB fill on a CMYK-OutputIntent page must mirror into the sidecar");
    assert!(
        scanned <= total / 4,
        "sidecar mirror scan must be rect-bounded: scanned {scanned} of {total} pixels \
         for a 20×20pt fill on a 100×100pt page"
    );
}

/// A small DeviceCMYK fill routes through
/// `mirror_cmyk_paint_into_sidecar_with_coverage` — same bounding contract.
#[test]
fn small_cmyk_fill_mirror_scan_is_rect_bounded() {
    let (scanned, total) = mirror_scan_for("0 0.5 0 0 k 10 10 20 20 re f", "");
    assert!(scanned > 0, "CMYK fill must mirror into the sidecar");
    assert!(
        scanned <= total / 4,
        "CMYK sidecar mirror scan must be rect-bounded: scanned {scanned} of {total} pixels"
    );
}
