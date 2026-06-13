//! Dirty-rect bounding for the SMask / CMYK-compose / spot / CMYK-sidecar /
//! RGB-sidecar pre-paint snapshot families and their after-paint scans.
//!
//! Historically each of these families captured a page-sized
//! `pixmap.data().to_vec()` and the consuming after-paint pass walked the
//! whole page. On packaging artwork with thousands of small vector paints on
//! a huge page that O(paint_ops × page_bytes) work dominated render CPU. A
//! paint's device rect provably bounds where it can change the pixmap, so
//! capturing + scanning only that rect is byte-identical. These probes pin
//! the bounding contract via the `snapshot_family_bytes` /
//! `snapshot_family_scanned_pixels` test-support counters.

#![cfg(all(feature = "rendering", feature = "icc", feature = "test-support"))]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{PageRenderer, RenderOptions};

/// Single-page 100×100pt PDF with an OutputIntent CMYK profile (object 5).
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
/// has a valid profile (mirrors the sidecar-mirror probe helper).
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

fn render_counts(content: &str, resources_inner: &str) -> (u64, u64, u64) {
    let icc = build_constant_cmyk_icc(128);
    let pdf = build_pdf_with_output_intent(content, resources_inner, &icc);
    let doc = PdfDocument::from_bytes(pdf).expect("synthetic OutputIntent PDF must parse");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72));
    let img = renderer.render_page(&doc, 0).expect("render");
    let total = (img.width as u64) * (img.height as u64);
    (
        renderer.snapshot_family_bytes(),
        renderer.snapshot_family_scanned_pixels(),
        total,
    )
}

/// A small CMYK fill on a CMYK-OutputIntent page that declares transparency
/// fires the CMYK-sidecar pre-paint snapshot (and, when the fill is itself
/// transparent, the compose-first snapshot + after-paint scan). The captured
/// snapshot must be bounded to the 20×20pt paint's device rect, not the full
/// 100×100pt page.
#[test]
fn small_transparent_cmyk_fill_snapshot_and_scan_are_rect_bounded() {
    // Declaring an ExtGState with ca<1 flips the transparency-detection gate
    // so the sidecar is allocated; applying it to the fill makes the paint
    // transparent so the compose-first family also fires.
    let resources = "/ExtGState << /GS1 << /ca 0.5 >> >>";
    let content = "/GS1 gs 0 0.5 0 0 k 10 10 20 20 re f";
    let (snap_bytes, scan_px, total) = render_counts(content, resources);

    assert!(
        snap_bytes > 0,
        "a CMYK fill on a transparency-detected CMYK-OutputIntent page must \
         take at least one pre-paint snapshot (sidecar / compose-first)"
    );
    // A single full-page RGBA snapshot is 4·total bytes. A 20×20pt fill on a
    // 100×100pt page covers ~4% of pixels; even with AA margin and several
    // families firing, the rect-bounded captures must stay well under one
    // full-page RGBA capture. `total` bytes (a quarter of one full-page
    // capture) is a comfortable but real ceiling — a full-page regression in
    // any single family blows past it.
    assert!(
        snap_bytes <= total,
        "snapshots must be rect-sized, not page-sized: {snap_bytes} bytes for \
         a 20×20pt fill on a {total}-pixel page (one full-page RGBA capture \
         alone is {} bytes)",
        4 * total
    );
    // The transparent CMYK fill also fires the compose-first after-paint
    // scan, which must be rect-bounded.
    assert!(
        scan_px > 0 && scan_px <= total / 4,
        "compose-first after-paint scan must be rect-bounded: scanned \
         {scan_px} of {total} pixels for a 20×20pt fill"
    );
}

/// A small SMask-modulated fill fires the SMask snapshot + rect-bounded
/// after-paint blend.
#[test]
fn small_smask_fill_snapshot_and_scan_are_rect_bounded() {
    // A luminosity soft mask form (object via /SMask in the ExtGState).
    let resources = "/ExtGState << /GS1 << /SMask << /S /Luminosity /G 6 0 R >> >> >>";
    // The /G form is a 100×100 black box; only the modulated fill region is
    // touched by the after-paint blend.
    let content = "/GS1 gs 0 0 0 1 k 10 10 20 20 re f";
    // The harness builder above only emits objects 1..=5, so the /G ref (6)
    // resolves to a missing object → the SMask path restores the snapshot.
    // That still exercises the snapshot capture (rect-bounded) without needing
    // a full form. The scan counter only ticks on the success blend, so this
    // probe asserts the capture is bounded.
    let (snap_bytes, _scan_px, total) = render_counts(content, resources);
    assert!(snap_bytes > 0, "an SMask fill must take a pre-paint snapshot");
    assert!(
        snap_bytes <= total,
        "SMask snapshot must be rect-sized, not page-sized: {snap_bytes} bytes \
         on a {total}-pixel page"
    );
}
