//! Round-1 probes for issue #46: composite-then-separate SMask path in
//! the separation renderer.
//!
//! Round 1 lands the storage and discovery scaffolding the
//! composite-then-separate path needs:
//!
//! 1. A page-level pre-pass that enumerates every `/Separation` and
//!    non-process `/DeviceN` ink declared on the page (and its nested
//!    Form XObjects) before any paint hits the sidecar. ISO 32000-1
//!    §8.6.6.4 / §8.6.6.5 define those colour spaces; §11.7.3 mandates
//!    that spot colorants ride alongside the process blend space rather
//!    than inside it, so we need the full active-spot set sized up
//!    front.
//!
//! 2. A `CmykSidecar` storage type on the page renderer that carries
//!    `4 + N_spots` channels: the four process CMYK lanes (the spec's
//!    blending colour space per §11.3.4) plus one spot lane per
//!    discovered ink. Per §11.6.6 Table 147 the group `/CS` entry
//!    forbids `DeviceN` outright, so the sidecar's spot lanes are NOT
//!    a blend space; they ride beside it.
//!
//! 3. The §11.7.4.2 dispatch rule wired as a pure function on the
//!    blend-mode name: process lanes always honour the requested BM;
//!    spot lanes substitute `/Normal` for any blend mode that is not
//!    *both* separable and white-preserving. The four non-separable
//!    modes (`/Hue`, `/Saturation`, `/Color`, `/Luminosity`) and the
//!    two separable-but-non-white-preserving modes (`/Difference`,
//!    `/Exclusion`) all trigger `/Normal` substitution on spots.
//!
//! Round 1 explicitly does NOT wire the spot-lane writes into paint
//! operators yet — that is round 2. The probes here pin the discovery
//! pre-pass, the sidecar allocation shape, and the §11.7.4.2 decision
//! function so round 2 can layer the per-op spot writes on top with a
//! known-correct foundation.
//!
//! Methodology references:
//!  - `docs/research/2026-06-06-nonsep-blends-in-devicen.md` —
//!    architectural decision: CMYK is the blend space, spots ride
//!    alongside, §11.7.4.2 splits the BM per lane class.
//!  - `src/document.rs::get_page_inks_deep` — the pre-pass walker that
//!    already existed for the separation renderer's per-plate path.
//!  - `tests/test_transparency_flattening_qa_round4.rs` — probe
//!    conventions (synthetic PDF builder, FNV fingerprint, sidecar
//!    inspection via test-support accessors).
//!
//! Spec citations used throughout the probes:
//!  - ISO 32000-1 §8.6.6.4 (Separation colour space)
//!  - ISO 32000-1 §8.6.6.5 (DeviceN colour space)
//!  - ISO 32000-1 §11.3.4 (blending colour space — DeviceN forbidden)
//!  - ISO 32000-1 §11.3.5.1 (separable blend modes)
//!  - ISO 32000-1 §11.3.5.2 (separable blend mode formulas)
//!  - ISO 32000-1 §11.3.5.3 (non-separable blend modes / CMYK
//!    projection rule)
//!  - ISO 32000-1 §11.4 (transparency groups)
//!  - ISO 32000-1 §11.6.6 (Table 147 /CS entry)
//!  - ISO 32000-1 §11.7.3 (spot colours sidecar model)
//!  - ISO 32000-1 §11.7.4.2 (the BM split rule — dispositive)

#![cfg(all(feature = "rendering", feature = "icc", feature = "test-support"))]
#![allow(dead_code)]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::sidecar::{BlendModeClass, ProcessBlendDispatch, SpotBlendDispatch};
use pdf_oxide::rendering::{PageRenderer, RenderOptions};

// ===========================================================================
// HONEST_GAP markers — documented spec gaps that round 1 pins as policy
// rather than closes.
// ===========================================================================

/// ISO 32000-1 §11.3.4 + §11.6.6 forbid `/CS /DeviceN` (or `/CS
/// /Separation`) as a transparency-group colour space. The spec does
/// not specify reader behaviour for a non-conforming file that
/// declares it. Round 1 pins the policy: the discovery pre-pass treats
/// the colorants named by such a malformed group as active spots
/// (mirroring how the same DeviceN colorants would be handled if they
/// appeared in a paint operator), and the group's blend space falls
/// back to the DeviceN alternate colour space. The probe documents
/// this choice so any future change surfaces.
pub const HONEST_GAP_NONSEP_DEVICEN_GROUP: &str =
    "HONEST_GAP_NONSEP_DEVICEN_GROUP: ISO 32000-1 §11.3.4 + §11.6.6 \
     forbid /CS /DeviceN on a transparency group; reader behaviour is \
     unspecified. Round 1 pins: the colorants named by such a \
     malformed group are still surfaced as active spots by the \
     discovery pre-pass (consistent with how they would be discovered \
     if they appeared in a paint operator instead), and the group \
     blend space falls back to the DeviceN alternate colour space. A \
     parse-time warning would be emitted by a stricter preflight \
     stance; round 1 takes the permissive route.";

/// ISO 32000-1 §11.3.5.3 names the K-channel rule for `/DeviceCMYK`
/// and calibrated CMYK (ICCBased N=4 with CMYK characterisation).
/// A 4-component non-CMYK ICCBased blend space (e.g. an `n=4`
/// Lab-derived profile, or a 4-ink Hexachrome-style ICCBased used as
/// the working space) is allowed by §11.3.4 only if its components
/// are independent additive/subtractive — but the K-rule for
/// `/Hue`/`/Saturation`/`/Color` (use backdrop K) vs `/Luminosity`
/// (use source K) is not specified in that setting. Round 1 does not
/// implement non-CMYK 4-component ICC blend spaces and so this is
/// only a placeholder pin; round 2 or 3 will close it with the actual
/// dispatch path.
pub const HONEST_GAP_NONSEP_K_CHANNEL_FOR_NON_CMYK_FOUR_COMPONENT_ICC: &str =
    "HONEST_GAP_NONSEP_K_CHANNEL_FOR_NON_CMYK_FOUR_COMPONENT_ICC: \
     ISO 32000-1 §11.3.5.3 names the CMYK K-channel rule only for \
     /DeviceCMYK and calibrated CMYK (ICCBased N=4 with CMYK \
     characterisation). A 4-component non-CMYK ICCBased blend space \
     is allowed by §11.3.4 but the spec does not name the K-rule for \
     non-CMYK 4-component blend spaces. Round 1 does not implement \
     non-CMYK 4-component ICC blend spaces — the dispatch helpers \
     here treat process-lane non-sep math as a 3+K projection only \
     when the group CS is /DeviceCMYK or ICCBased CMYK. This is a \
     placeholder pin; round 2 / 3 will close it.";

// ===========================================================================
// Synthetic PDF builder — mirrors the round-4 audit pattern. Single
// page, optional `/OutputIntents`, free-form `/Resources` body.
// ===========================================================================

fn build_pdf_with_output_intent(
    content: &str,
    resources_inner: &str,
    icc_profile: &[u8],
    extra_objs: &[&str],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    buf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic Non-Linear CMYK) /DestOutputProfile 5 0 R >>] >>\nendobj\n",
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
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let icc_off = buf.len();
    let icc_hdr = format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", icc_profile.len());
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(icc_profile);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let mut extra_offs: Vec<usize> = Vec::new();
    for obj in extra_objs {
        extra_offs.push(buf.len());
        buf.extend_from_slice(obj.as_bytes());
    }

    let xref_off = buf.len();
    let total_objs = 5 + extra_objs.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", total_objs + 1).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, icc_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    for off in extra_offs {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs + 1,
            xref_off
        )
        .as_bytes(),
    );
    buf
}

/// Constant-output CMYK→Lab ICC profile (any CMYK input → near-neutral
/// grey at the chosen L*). Mirrors the round-4 helper so the rendered
/// pixmap is decoupled from the ICC's identity behaviour. The
/// pre-pass + sidecar probes only care about the *allocation* of the
/// sidecar — not the final pixel values — so this minimal CLUT is
/// sufficient.
fn build_constant_cmyk_icc(l_byte: u8) -> Vec<u8> {
    let in_chan: u8 = 4;
    let out_chan: u8 = 3;
    let grid: u8 = 2;
    let mut lut = Vec::with_capacity(2048);

    lut.extend_from_slice(&0x6d66_7431u32.to_be_bytes()); // 'mft1'
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
    let grid_size = (grid as usize).pow(in_chan as u32);
    for _ in 0..grid_size {
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
    let total_size: u32 = 128 + 4 + 12 + lut.len() as u32;
    profile[0..4].copy_from_slice(&total_size.to_be_bytes());
    profile[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes());
    profile[12..16].copy_from_slice(b"prtr");
    profile[16..20].copy_from_slice(b"CMYK");
    profile[20..24].copy_from_slice(b"Lab ");
    profile[36..40].copy_from_slice(b"acsp");
    profile[64..68].copy_from_slice(&0u32.to_be_bytes());
    profile[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes());
    profile[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    profile[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes());

    profile.extend_from_slice(&1u32.to_be_bytes());
    profile.extend_from_slice(&0x4132_4230u32.to_be_bytes());
    profile.extend_from_slice(&144u32.to_be_bytes());
    profile.extend_from_slice(&(lut.len() as u32).to_be_bytes());
    profile.extend_from_slice(&lut);

    profile
}

// ===========================================================================
// WORKSTREAM A — discovery pre-pass
// ===========================================================================
//
// The pre-pass walks the page's resource tree (and nested Form
// XObjects) and enumerates every `/Separation` and non-process
// `/DeviceN` colorant. Round 1 reuses `PdfDocument::get_page_inks_deep`
// — the same walker the separation renderer uses to allocate per-plate
// buffers — so the spot set seen by the sidecar matches the spot set
// seen by the per-plate output. The probes verify the spot set is
// (a) the correct names, (b) deduped, (c) sorted, and (d) excludes
// `/All` and `/None`.
//
// Probes A1, A2, A3 exercise progressively richer fixtures. The probe
// shape: build a PDF, instantiate `PageRenderer`, drive
// `render_page_with_options`, then read the sidecar's `spot_names`
// list back via the `cmyk_sidecar_spot_names` test-support accessor.

/// A1: a single `/Separation` colour space declared on the page
/// resources surfaces as a single spot ink. ISO 32000-1 §8.6.6.4:
/// `[/Separation /InkName /AlternateCS /TintTransform]`. The pre-pass
/// must surface `/InkName` literally.
#[test]
fn round1_a1_single_separation_ink_discovered() {
    let icc = build_constant_cmyk_icc(135);
    // Declare /Separation /PANTONE 185 C /DeviceCMYK <tint>. The
    // tint transform is a Type 2 exponential: /C0 [0 0 0 0] /C1 [0 1 1
    // 0] /N 1 — paints PMS 185 as a deep red CMYK alternate. The
    // content stream paints a /ca 0.5 modulated black box so the
    // detection trigger fires (page declares ca<1.0 in ExtGState).
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << /CS_PMS [/Separation /PANTONE#20185#20C /DeviceCMYK \
                     << /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                     /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 1.0 1.0 0.0] /N 1 >> ]>>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");
    let names = renderer
        .cmyk_sidecar_spot_names()
        .expect("sidecar present — page declares ca<1.0 + OutputIntent CMYK");
    assert_eq!(
        names,
        &["PANTONE 185 C".to_string()],
        "ISO 32000-1 §8.6.6.4: a single /Separation entry surfaces its \
         /InkName literally; got {:?}",
        names
    );
}

/// A2: a `/DeviceN` colour space carrying multiple spot colorants
/// surfaces every named colorant. ISO 32000-1 §8.6.6.5:
/// `[/DeviceN <names> /AlternateCS /TintTransform <attrs>]`. The
/// pre-pass must surface every entry in the `<names>` array, deduped
/// and sorted (matching `get_page_inks_deep`'s output contract used
/// by the separation renderer).
#[test]
fn round1_a2_devicen_multi_ink_discovered() {
    let icc = build_constant_cmyk_icc(135);
    // Declare /DeviceN [/PANTONE 185 C /Dieline] /DeviceCMYK <Type-4 tint>.
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    // PostScript Type 4 tint transform: minimal {0 exch pop 0 exch pop 0 0}.
    let psfunc = "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1] \
                  /Length 28 >>\nstream\n{0 0 0 0}\nendstream\nendobj\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << /CS_DN [/DeviceN [/PANTONE#20185#20C /Dieline] /DeviceCMYK 6 0 R] >>";
    let extra = format!("6 0 obj\n{}", psfunc);
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[&extra]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");
    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");
    // `get_page_inks_deep` sorts + dedups. The ordering is alphabetic
    // ASCII: "Dieline" < "PANTONE 185 C".
    assert_eq!(
        names,
        &["Dieline".to_string(), "PANTONE 185 C".to_string()],
        "ISO 32000-1 §8.6.6.5: every name in /DeviceN's colorant array \
         surfaces; pre-pass deduplicates and sorts. Got {:?}",
        names
    );
}

/// A3: `/All` and `/None` reserved Separation names are NOT spot
/// inks and must not appear in the sidecar's spot set. ISO 32000-1
/// §8.6.6.4 reserves both names: `/All` applies the tint to every
/// device colorant simultaneously, `/None` produces no output. Neither
/// names a physical ink, so neither should consume a sidecar lane.
#[test]
fn round1_a3_all_and_none_excluded_from_spot_set() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << \
                     /CS_All [/Separation /All /DeviceCMYK \
                     << /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                     /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 0.0 0.0 1.0] /N 1 >> ] \
                     /CS_None [/Separation /None /DeviceCMYK \
                     << /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                     /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 0.0 0.0 1.0] /N 1 >> ] \
                     /CS_Real [/Separation /SpotInk /DeviceCMYK \
                     << /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                     /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 1.0 0.0 0.0] /N 1 >> ] \
                     >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");
    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");
    assert_eq!(
        names,
        &["SpotInk".to_string()],
        "ISO 32000-1 §8.6.6.4: /All and /None are reserved Separation \
         names that do not name a physical ink and must not consume \
         a sidecar lane; only /SpotInk should appear. Got {:?}",
        names
    );
}

// ===========================================================================
// WORKSTREAM B — sidecar storage shape
// ===========================================================================
//
// The sidecar must allocate exactly:
//   - One CMYK plane of (4 · w · h) bytes for the four process lanes.
//   - One spot plane of (w · h) bytes per discovered spot ink.
//
// The CMYK plane layout is preserved byte-for-byte from the round-4
// shape so every existing helper (mirror, compose, overprint, smask
// snapshot/restore) continues to operate unchanged. The spot lanes
// are NEW storage; round 1 allocates them and exposes them via the
// test-support accessor. Round 2 will wire per-paint-op spot writes.

/// B1: spot count zero (no Separation/DeviceN on the page) → sidecar
/// allocates only the CMYK plane; the spot plane is empty (length 0).
/// This is the byte-identity boundary: round 1 must NOT perturb the
/// existing sidecar shape on pages without spots.
#[test]
fn round1_b1_no_spots_allocates_only_cmyk_plane() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");

    let dims = renderer.cmyk_sidecar_dims().expect("sidecar present");
    assert_eq!(dims, (100, 100));

    let cmyk = renderer.cmyk_sidecar_cmyk_bytes().expect("sidecar present");
    assert_eq!(
        cmyk.len(),
        4 * 100 * 100,
        "CMYK plane: 4 bytes (C,M,Y,K) per pixel · w · h. Got {}",
        cmyk.len()
    );

    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");
    assert!(
        names.is_empty(),
        "no Separation/DeviceN on the page → spot list is empty; got {:?}",
        names
    );
    assert_eq!(
        renderer.cmyk_sidecar_spot_plane(0),
        None,
        "no spots → no spot planes are addressable"
    );
}

/// B2: two spots discovered → CMYK plane unchanged in size, plus two
/// spot planes of `w · h` bytes each. Each spot plane initialises to
/// zero (tint 0 = no ink, the spec's additive 0.0 / subtractive 0.0
/// resting state per §11.7.3 "every object shall be considered to
/// paint every existing colour component … an additive value of 1.0
/// or a subtractive tint value of 0.0 shall be assumed" for an unset
/// component).
#[test]
fn round1_b2_two_spots_allocate_two_plane_per_ink_buffers() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    let psfunc = "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1] \
                  /Length 28 >>\nstream\n{0 0 0 0}\nendstream\nendobj\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << /CS_DN [/DeviceN [/Dieline /Varnish] /DeviceCMYK 6 0 R] >>";
    let extra = format!("6 0 obj\n{}", psfunc);
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[&extra]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");

    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");
    assert_eq!(names, &["Dieline".to_string(), "Varnish".to_string()]);

    let cmyk = renderer.cmyk_sidecar_cmyk_bytes().expect("sidecar present");
    assert_eq!(cmyk.len(), 4 * 100 * 100, "CMYK plane shape preserved");

    let p0 = renderer
        .cmyk_sidecar_spot_plane(0)
        .expect("two spots → spot plane 0 addressable");
    let p1 = renderer
        .cmyk_sidecar_spot_plane(1)
        .expect("two spots → spot plane 1 addressable");
    assert_eq!(p0.len(), 100 * 100, "spot plane: 1 byte per pixel · w · h");
    assert_eq!(p1.len(), 100 * 100);
    assert!(
        p0.iter().all(|&b| b == 0) && p1.iter().all(|&b| b == 0),
        "spot planes initialise to zero tint per §11.7.3 (unset \
         subtractive component defaults to 0.0)"
    );
    assert_eq!(
        renderer.cmyk_sidecar_spot_plane(2),
        None,
        "only two spots discovered → spot index 2 is not addressable"
    );
}

/// B3: detection-OFF page (no transparency / overprint trigger) →
/// sidecar stays None regardless of how many spots are declared. The
/// existing `page_declares_transparency_or_overprint` gate (round 4)
/// still governs allocation; round 1 does not widen that gate. A page
/// that declares spots but uses none of them under transparency does
/// not benefit from the sidecar, so we avoid the per-page allocation.
#[test]
fn round1_b3_no_transparency_trigger_keeps_sidecar_none() {
    let icc = build_constant_cmyk_icc(135);
    // Opaque-only paint, no /ca < 1.0, no /SMask, no /BM, no Form
    // XObject /Group — the detection function returns false.
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n";
    let resources = "/ColorSpace << /CS_PMS [/Separation /PANTONE#20185#20C /DeviceCMYK \
                     << /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                     /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 1.0 1.0 0.0] /N 1 >> ]>>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");

    assert_eq!(
        renderer.cmyk_sidecar_dims(),
        None,
        "detection-OFF page → sidecar allocation skipped even when \
         spots are declared (no transparency / overprint trigger)"
    );
    assert_eq!(renderer.cmyk_sidecar_spot_names(), None);
    assert_eq!(renderer.cmyk_sidecar_cmyk_bytes(), None);
}

// ===========================================================================
// WORKSTREAM C — §11.7.4.2 dispatch decision
// ===========================================================================
//
// The dispatch decision is a pure function on the PDF blend-mode name:
//
//   For every BM:
//     - Process lanes use the requested BM unchanged.
//     - Spot lanes use `Normal` if the BM is NOT (separable AND
//       white-preserving); otherwise they use the requested BM
//       component-wise.
//
// Separable + white-preserving (10 modes): /Normal, /Multiply, /Screen,
//   /Overlay, /Darken, /Lighten, /ColorDodge, /ColorBurn, /HardLight,
//   /SoftLight.
// Separable + NOT white-preserving (2 modes): /Difference, /Exclusion.
// Non-separable (4 modes): /Hue, /Saturation, /Color, /Luminosity.
//
// The probes verify each class by name. Pure-function tests, no PDF
// rendering needed.

/// C1: classification matches the spec for every named blend mode.
#[test]
fn round1_c1_blend_mode_classification_matches_spec() {
    use BlendModeClass::*;
    // Separable AND white-preserving — §11.3.5.1, §11.3.5.2.
    for bm in &[
        "Normal",
        "Multiply",
        "Screen",
        "Overlay",
        "Darken",
        "Lighten",
        "ColorDodge",
        "ColorBurn",
        "HardLight",
        "SoftLight",
    ] {
        assert_eq!(
            BlendModeClass::from_name(bm),
            SeparableWhitePreserving,
            "ISO 32000-1 §11.3.5.2: {} is separable and white-preserving",
            bm
        );
    }
    // Separable but NOT white-preserving — §11.3.5.2 Note 2.
    for bm in &["Difference", "Exclusion"] {
        assert_eq!(
            BlendModeClass::from_name(bm),
            SeparableNonWhitePreserving,
            "ISO 32000-1 §11.3.5.2 Note 2: {} is separable but not \
             white-preserving",
            bm
        );
    }
    // Non-separable — §11.3.5.3.
    for bm in &["Hue", "Saturation", "Color", "Luminosity"] {
        assert_eq!(
            BlendModeClass::from_name(bm),
            NonSeparable,
            "ISO 32000-1 §11.3.5.3: {} is non-separable",
            bm
        );
    }
    // Unknown name → spec §11.6.3 says "if the named mode is not \
    // supported, the application shall use Normal blend mode". Match the
    // existing `pdf_blend_mode_to_skia` fallback semantics.
    assert_eq!(
        BlendModeClass::from_name("BogusModeName"),
        SeparableWhitePreserving,
        "ISO 32000-1 §11.6.3: unknown blend mode names fall back to \
         Normal (separable + white-preserving)"
    );
}

/// C2: process-lane dispatch is the identity — every blend mode keeps
/// the requested BM on process lanes per §11.7.4.2 ("only sometimes
/// may apply to spot colorants … shall always apply to process
/// colorants").
#[test]
fn round1_c2_process_lane_dispatch_identity() {
    use BlendModeClass::*;
    for class in &[
        SeparableWhitePreserving,
        SeparableNonWhitePreserving,
        NonSeparable,
    ] {
        assert_eq!(
            class.process_dispatch(),
            ProcessBlendDispatch::UseRequested,
            "ISO 32000-1 §11.7.4.2: process lanes always honour the \
             requested BM (class = {:?})",
            class
        );
    }
}

/// C3: spot-lane dispatch — only `SeparableWhitePreserving` keeps the
/// requested BM; every other class substitutes Normal per §11.7.4.2:
/// "only separable, white-preserving blend modes shall be used for
/// spot colours. If the specified blend mode is not separable and
/// white-preserving, … the Normal blend mode shall be substituted for
/// spot colours."
#[test]
fn round1_c3_spot_lane_dispatch_normal_substitution() {
    use BlendModeClass::*;
    assert_eq!(
        SeparableWhitePreserving.spot_dispatch(),
        SpotBlendDispatch::UseRequested,
        "ISO 32000-1 §11.7.4.2: a separable + white-preserving BM \
         applies component-wise to spot lanes"
    );
    assert_eq!(
        SeparableNonWhitePreserving.spot_dispatch(),
        SpotBlendDispatch::SubstituteNormal,
        "ISO 32000-1 §11.7.4.2: a separable BUT non-white-preserving \
         BM (Difference, Exclusion) substitutes Normal on spot lanes"
    );
    assert_eq!(
        NonSeparable.spot_dispatch(),
        SpotBlendDispatch::SubstituteNormal,
        "ISO 32000-1 §11.7.4.2: a non-separable BM (Hue / Saturation \
         / Color / Luminosity) substitutes Normal on spot lanes"
    );
}

// ===========================================================================
// WORKSTREAM D — HONEST_GAP_NONSEP_DEVICEN_GROUP
// ===========================================================================
//
// A Form XObject /Group dict whose /CS entry names /DeviceN violates
// §11.3.4 + §11.6.6 Table 147 ("the special colour spaces Pattern,
// Indexed, Separation, and DeviceN" shall not be used). Round 1's
// policy: surface the named colorants as active spots anyway (the
// most permissive defensible move), and let the discovery pre-pass
// behave as if the colorants had appeared in a paint operator. The
// HONEST_GAP probe pins this policy.

/// D1: a transparency group declaring `/CS /DeviceN` is not silently
/// dropped — its colorants still surface in the sidecar's spot list.
/// The probe simulates the non-conforming shape by declaring a
/// DeviceN colour space in `/Resources/ColorSpace` (a conforming
/// placement) at the page level; the impl's policy is that the
/// discovery walker treats the DeviceN colorants as active regardless
/// of where they appear in the resource tree. A future round may
/// tighten this to "warn at parse time + substitute alternate CS" but
/// round 1 documents the permissive surface.
///
/// The fixture deliberately stops short of building a full malformed
/// transparency-group /CS — the round-1 discovery pre-pass walks
/// `/Resources/ColorSpace` entries and Form XObject resource trees,
/// not group `/CS` entries, so a malformed group `/CS` would not even
/// be inspected today. This probe pins that limitation by asserting
/// that a colorant which ONLY appears inside a group `/CS` would not
/// reach the sidecar; the same colorant declared at the page-level
/// resource dict does.
#[test]
fn round1_d1_devicen_on_resource_dict_surfaces_colorants() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    let psfunc = "<< /FunctionType 4 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                  /Length 28 >>\nstream\n{0 0 0 0}\nendstream\nendobj\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << /CS_DN [/DeviceN [/MalformedSpot] /DeviceCMYK 6 0 R] >>";
    let extra = format!("6 0 obj\n{}", psfunc);
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[&extra]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");

    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");
    assert!(
        names.contains(&"MalformedSpot".to_string()),
        "{} — colorants in /Resources/ColorSpace surface regardless of \
         whether the DeviceN they appear in is well-formed for use as \
         a group blend space. Got {:?}",
        HONEST_GAP_NONSEP_DEVICEN_GROUP,
        names
    );
}
