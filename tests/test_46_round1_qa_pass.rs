//! Round-1 QA probes for issue #46 (CMYK + spot-ink sidecar).
//!
//! These probes pin behaviours the round-1 design+impl probes do not
//! cover. They are intentionally additive — every probe pins the
//! *correct* spec behaviour as a byte-exact assertion. Every probe
//! now runs live; the round-1 fix agent landed each of the bugs the
//! `QA_BUG_*` constants below describe, and the assertions hold
//! byte-exact at HEAD. The constants are preserved as historical
//! markers and as load-bearing references inside the probes that
//! pin the matching spec rule, so a regression that re-introduces
//! the bug surfaces with the original citation in scope.
//!
//! Methodology references:
//!  - `docs/research/2026-06-06-nonsep-blends-in-devicen.md` —
//!    architectural decision: CMYK is the blend space, spots ride
//!    alongside, §11.7.4.2 splits the BM per lane class.
//!  - `tests/test_46_round1_spot_sidecar.rs` — round-1 design+impl
//!    probes; this file augments without overlap.

#![cfg(all(feature = "rendering", feature = "icc", feature = "test-support"))]
#![allow(dead_code)]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::sidecar::{BlendModeClass, SpotBlendDispatch};
use pdf_oxide::rendering::{PageRenderer, RenderOptions};

// ===========================================================================
// QA bug markers — pin the exact misbehaviour with spec citation.
// ===========================================================================

/// `extract_inks_from_color_space_dict` (src/document.rs ~line 727)
/// pushes every name in a `/DeviceN` colorants array into the spot
/// set without consulting the attributes dictionary's `/Subtype` or
/// `/Process` keys. ISO 32000-1 §8.6.6.5 says the optional `/Process`
/// attributes dict carries CMYK / RGB / Gray PROCESS channels — those
/// names are NOT physical spots and must not consume a spot lane in
/// the sidecar. A multi-channel DeviceN containing
/// `[/Cyan /Magenta /Yellow /Black /PANTONE 185 C]` with `/Subtype
/// /DeviceN` and `/Process << /ColorSpace /DeviceCMYK /Components
/// [/Cyan /Magenta /Yellow /Black] >>` should surface ONLY
/// `PANTONE 185 C` as a spot. The impl surfaces all five.
pub const QA_BUG_DEVICEN_PROCESS_POLLUTES_SPOT_SET: &str =
    "QA_BUG_DEVICEN_PROCESS_POLLUTES_SPOT_SET: ISO 32000-1 §8.6.6.5 \
     names /Process channels in a DeviceN attributes dict as the \
     process colorant set — not spots. The impl's spot extractor \
     ignores /Process and surfaces every name as a spot, polluting \
     the sidecar's spot set with /Cyan, /Magenta, /Yellow, /Black \
     when the DeviceN declares them as process. Round 1 must filter \
     /Process channels OUT of the spot set so the sidecar only \
     allocates lanes for physical spot inks.";

/// `BlendModeClass` and the detection helper handle `/BM` only as
/// `Object::Name`. ISO 32000-1 §11.3.5 / §11.6.3 allows `/BM` to be a
/// name OR an array of names; for an array "the first name that names
/// a blend mode supported by the conforming reader shall be used".
/// The detection helper at sidecar.rs:440 matches
/// `Object::Name(bm)` only — an array `/BM [/Multiply]` is silently
/// ignored and the detection trigger does not fire. The
/// `ext_gstate` parser at ext_gstate.rs:111 picks `arr.first()` (not
/// the first recognised name), so an array `/BM [/UnknownMode
/// /Multiply]` collapses to `Normal` via the from_name fallback
/// instead of `Multiply`.
pub const QA_BUG_BM_ARRAY_NOT_HONOURED: &str =
    "QA_BUG_BM_ARRAY_NOT_HONOURED: ISO 32000-1 §11.3.5 + §11.6.3: a \
     `/BM` array uses the first RECOGNISED name. The detection helper \
     matches only Object::Name (drops the array case entirely); the \
     ext_gstate parser picks arr.first() without classifying. Round 1 \
     should either (a) unwrap arrays and pick first-recognised in the \
     parser, or (b) declare a HONEST_GAP for malformed /BM array.";

/// Round-1 closed the silent-swallow gap by emitting `log::warn!`
/// on the error path AND returning an empty Vec; the warning surfaces
/// the silent fidelity loss so the host log pipeline can see it,
/// while the empty-vec return matches how the separation renderer
/// already degrades on the same `get_page_inks_deep` failure.
///
/// The pin probe lives at
/// `src/rendering/sidecar.rs::tests::discover_page_spot_inks_warns_on_deep_walk_error`
/// — it calls `discover_page_spot_inks` directly (the function is
/// `pub(crate)`) and asserts both halves of the contract: empty Vec
/// return AND a captured warn record naming the page index. Round 2
/// can then trust that any non-empty spot writes will only land on
/// pages where discovery actually succeeded.
pub const QA_GAP_DISCOVER_ERROR_SURFACED_VIA_WARN: &str =
    "QA_GAP_DISCOVER_ERROR_SURFACED_VIA_WARN: round-1 fix landed: \
     discover_page_spot_inks now log::warn!s on every error from \
     get_page_inks_deep (parse error, malformed stream, recursion- \
     bound trip, page lookup miss) before returning the empty Vec. \
     The warning carries the page index and the underlying error so \
     a log scrape can pinpoint the affected page; round 2's per-paint \
     spot writes consistently see the empty spot set and degrade in \
     lockstep with the diagnostic signal.";

// ===========================================================================
// Synthetic PDF builder — re-uses the same shape as
// test_46_round1_spot_sidecar.rs so the corpus stays uniform.
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
// QA-AREA 1: §8.6.6.5 /Process subtype rejection from the spot set
// ===========================================================================

/// QA1.1: A DeviceN colour space declaring `/Subtype /DeviceN` with a
/// `/Process` attributes entry naming the four process inks must NOT
/// surface those four names as spots. ISO 32000-1 §8.6.6.5 / Table 73
/// names the `/Process` key as "an optional dictionary containing
/// information about the process colour space"; per §11.7.4.1 +
/// §8.6.6.5 the process colorants ride on the page's process plates,
/// not on spot lanes. The only true spot in this declaration is the
/// trailing `/PANTONE 185 C`.
///
/// CURRENT IMPL BEHAVIOUR (BUG): pushes all five names; the sidecar
/// allocates five spot lanes including four named after process inks.
#[test]
fn qa1_1_devicen_with_process_subtype_excludes_process_channels() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    let psfunc = "<< /FunctionType 4 /Domain [0 1 0 1 0 1 0 1 0 1] /Range [0 1 0 1 0 1 0 1] \
                  /Length 28 >>\nstream\n{0 0 0 0}\nendstream\nendobj\n";
    // DeviceN with explicit /Subtype /DeviceN + /Process attributes
    // dict listing CMYK as the process channels. Only PANTONE 185 C
    // is a true spot. (§8.6.6.5 Table 73: `Subtype` / `Process` keys.)
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << /CS_DN [/DeviceN \
                     [/Cyan /Magenta /Yellow /Black /PANTONE#20185#20C] \
                     /DeviceCMYK 6 0 R \
                     << /Subtype /DeviceN \
                        /Process << /ColorSpace /DeviceCMYK \
                                    /Components [/Cyan /Magenta /Yellow /Black] >> \
                     >> \
                     ] >>";
    let extra = format!("6 0 obj\n{}", psfunc);
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[&extra]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");
    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");

    assert_eq!(
        names,
        &["PANTONE 185 C".to_string()],
        "{} — got {:?}",
        QA_BUG_DEVICEN_PROCESS_POLLUTES_SPOT_SET,
        names
    );
}

/// QA1.2: A `/Subtype /NChannel` DeviceN (PDF 1.7 §8.6.6.5 addition)
/// behaves like `/Subtype /DeviceN` for the spot-set rule: process
/// colorants named in `/Process` are NOT spots. NChannel is a stricter
/// subtype that requires the alternate CS to be a process colour
/// space; the spot-vs-process rule is identical.
#[test]
fn qa1_2_devicen_with_nchannel_subtype_excludes_process_channels() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    let psfunc = "<< /FunctionType 4 /Domain [0 1 0 1 0 1] /Range [0 1 0 1 0 1 0 1] \
                  /Length 28 >>\nstream\n{0 0 0 0}\nendstream\nendobj\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << /CS_DN [/DeviceN \
                     [/Cyan /Magenta /Dieline] \
                     /DeviceCMYK 6 0 R \
                     << /Subtype /NChannel \
                        /Process << /ColorSpace /DeviceCMYK \
                                    /Components [/Cyan /Magenta] >> \
                     >> \
                     ] >>";
    let extra = format!("6 0 obj\n{}", psfunc);
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[&extra]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");
    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");

    assert_eq!(
        names,
        &["Dieline".to_string()],
        "ISO 32000-1 §8.6.6.5 NChannel subtype: /Process channels are \
         process, not spot. Got {:?}",
        names
    );
}

// ===========================================================================
// QA-AREA 2: §11.6.3 BM array semantics
// ===========================================================================

/// QA2.1: detection-on for an ExtGState whose `/BM` is an ARRAY
/// `[/Multiply]`. Per §11.3.5 the array is unwrapped to the first
/// recognised name. `Multiply` is non-Normal so the trigger fires.
///
/// CURRENT IMPL BEHAVIOUR (BUG): detection helper matches only
/// `Object::Name(bm)` and skips the array — trigger does NOT fire,
/// sidecar stays None.
#[test]
fn qa2_1_detection_fires_for_bm_array_with_non_normal_mode() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Mult gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    let resources = "/ExtGState << /Mult << /Type /ExtGState /BM [/Multiply] >> >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");

    assert!(
        renderer.cmyk_sidecar_dims().is_some(),
        "{} — got dims = None (detection missed the array)",
        QA_BUG_BM_ARRAY_NOT_HONOURED
    );
}

/// QA2.2: detection-OFF when the `/BM` array first-recognised entry
/// resolves to Normal. Array `[/UnknownInventedMode /Normal]` → first
/// recognised is /Normal → detection does NOT fire.
///
/// Round 1 today: the detection helper drops the array → no trigger
/// (which happens to land at the spec-correct answer for this case,
/// but for the wrong reason).
#[test]
fn qa2_2_detection_does_not_fire_for_bm_array_unwrapping_to_normal() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n";
    let resources =
        "/ExtGState << /NM << /Type /ExtGState /BM [/UnknownInventedMode /Normal] >> >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");

    // Today: detection helper ignores the array; the page has no
    // other transparency trigger → sidecar stays None. This is the
    // spec-correct OUTCOME (Normal does not warrant a sidecar) but
    // arrived at by the wrong code path. Pin the outcome so a future
    // fix to the array handling does not break it.
    assert!(
        renderer.cmyk_sidecar_dims().is_none(),
        "§11.6.3 array unwrap to /Normal → no transparency trigger; \
         got dims = {:?}",
        renderer.cmyk_sidecar_dims()
    );
}

// ===========================================================================
// QA-AREA 3: adversarial blend-mode classification
// ===========================================================================

/// QA3.1: PDF names are CASE-SENSITIVE per ISO 32000-1 §7.3.5. A
/// misspelt `/multiply` (lowercase) is an unknown mode → §11.6.3
/// fallback → Normal class. The spot dispatch is therefore
/// `UseRequested` (Normal is separable + white-preserving).
#[test]
fn qa3_1_case_sensitive_mode_names_fall_back_to_normal() {
    use BlendModeClass::*;
    // PDF names are case-sensitive (§7.3.5). Every mis-cased name is
    // an unknown name and falls back to Normal's class.
    assert_eq!(BlendModeClass::from_name("multiply"), SeparableWhitePreserving);
    assert_eq!(BlendModeClass::from_name("MULTIPLY"), SeparableWhitePreserving);
    assert_eq!(BlendModeClass::from_name("normal"), SeparableWhitePreserving);
    assert_eq!(BlendModeClass::from_name("luminosity"), SeparableWhitePreserving);
    // Spot dispatch on the unknown class is UseRequested (same as Normal).
    assert_eq!(
        BlendModeClass::from_name("multiply").spot_dispatch(),
        SpotBlendDispatch::UseRequested
    );
}

/// QA3.2: a truncated mode name (`/Multipl`, `/Lumin`) is unknown and
/// falls back to Normal class per §11.6.3.
#[test]
fn qa3_2_truncated_names_fall_back_to_normal() {
    use BlendModeClass::*;
    assert_eq!(BlendModeClass::from_name("Multipl"), SeparableWhitePreserving);
    assert_eq!(BlendModeClass::from_name("Lumin"), SeparableWhitePreserving);
    assert_eq!(BlendModeClass::from_name("Hu"), SeparableWhitePreserving);
}

/// QA3.3: an empty name is unknown → Normal class. (PDF allows
/// zero-length names per §7.3.5 although they are rare.)
#[test]
fn qa3_3_empty_name_falls_back_to_normal() {
    assert_eq!(BlendModeClass::from_name(""), BlendModeClass::SeparableWhitePreserving);
}

/// QA3.4: a name that LOOKS numeric (`/0`, `/123`) is still an
/// unknown blend-mode name → Normal class. (PDF names can contain
/// any printable ASCII.)
#[test]
fn qa3_4_numeric_looking_names_fall_back_to_normal() {
    assert_eq!(BlendModeClass::from_name("0"), BlendModeClass::SeparableWhitePreserving);
    assert_eq!(BlendModeClass::from_name("123"), BlendModeClass::SeparableWhitePreserving);
    assert_eq!(BlendModeClass::from_name("-1"), BlendModeClass::SeparableWhitePreserving);
}

/// QA3.5: `/Compatible` — a legacy PDF 1.4 blend-mode synonym for
/// `/Normal` (per the original Adobe PDF Reference 1.4 §7.2.4). It is
/// not in the §11.3.5 list, so the §11.6.3 fallback applies: render
/// as Normal. That puts it in `SeparableWhitePreserving`, which is
/// the correct class.
#[test]
fn qa3_5_compatible_mode_legacy_normal_synonym() {
    // §11.6.3 fallback handles this implicitly: unknown → Normal.
    // The result is correct regardless of whether the impl recognises
    // /Compatible as a known synonym.
    assert_eq!(
        BlendModeClass::from_name("Compatible"),
        BlendModeClass::SeparableWhitePreserving
    );
}

/// QA3.6: a name with PDF hex-escape characters (`/Hard#20Light` is
/// the literal name "Hard Light" with a space). The PDF parser
/// resolves `#20` to a space before the name reaches `from_name`, so
/// `from_name` should be called with `"Hard Light"`, which is unknown
/// → Normal class. The spec-correct mode name is `HardLight` (no
/// space); `Hard Light` is a misspelling.
#[test]
fn qa3_6_hex_escaped_name_post_parser_resolution() {
    // The PDF parser already decodes #XX hex escapes (§7.3.5) before
    // the name reaches the renderer. So /Hard#20Light arrives at
    // from_name as "Hard Light" (with the actual space). That is not
    // a recognised blend mode name and falls back to Normal class.
    assert_eq!(
        BlendModeClass::from_name("Hard Light"),
        BlendModeClass::SeparableWhitePreserving
    );
    // HardLight (no space) is the spec name.
    assert_eq!(BlendModeClass::from_name("HardLight"), BlendModeClass::SeparableWhitePreserving);
}

// ===========================================================================
// QA-AREA 4: detection-helper edge cases
// ===========================================================================

/// QA4.1: an ExtGState with `/CA < 1.0` (stroke alpha) fires the
/// trigger just like `/ca < 1.0`. ISO 32000-1 §11.6.4.4 / Table 128:
/// `/CA` is the stroking alpha, `/ca` is the non-stroking alpha. The
/// detection helper checks both keys (sidecar.rs:423) — pin the
/// stroke side.
#[test]
fn qa4_1_detection_fires_on_uppercase_ca_stroke_alpha() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /HalfStroke gs\n0 0 0 1 K\n2 w\n10 10 80 80 re\nS\n";
    let resources = "/ExtGState << /HalfStroke << /Type /ExtGState /CA 0.5 >> >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");

    assert!(
        renderer.cmyk_sidecar_dims().is_some(),
        "§11.6.4.4 + Table 128: stroke alpha /CA 0.5 must fire the \
         transparency trigger; got dims = None"
    );
}

/// QA4.2: `/SMask /None` is the spec sentinel for "clear the soft
/// mask" (§11.6.5.2). It is NOT a soft mask declaration and must NOT
/// fire the trigger by itself.
#[test]
fn qa4_2_detection_does_not_fire_on_smask_none_sentinel() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n";
    let resources = "/ExtGState << /Clear << /Type /ExtGState /SMask /None >> >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");

    assert!(
        renderer.cmyk_sidecar_dims().is_none(),
        "§11.6.5.2: /SMask /None clears the soft mask — it is not a \
         transparency declaration and must not fire the trigger; got \
         dims = {:?}",
        renderer.cmyk_sidecar_dims()
    );
}

/// QA4.3: `/CA 1.0` (and `/ca 1.0`) are no-ops — fully opaque. They
/// must NOT fire the trigger.
#[test]
fn qa4_3_detection_does_not_fire_on_ca_equals_one() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n";
    let resources = "/ExtGState << /Opaque << /Type /ExtGState /ca 1.0 /CA 1.0 >> >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");

    assert!(
        renderer.cmyk_sidecar_dims().is_none(),
        "fully-opaque /ca and /CA must not fire transparency trigger; \
         got dims = {:?}",
        renderer.cmyk_sidecar_dims()
    );
}

// ===========================================================================
// QA-AREA 5: spot-set adversarial composition
// ===========================================================================

/// QA5.1: spots named after process colorants. ISO 32000-1 §8.6.6.4
/// does NOT forbid a Separation colour space whose `/InkName` is one
/// of `Cyan`, `Magenta`, `Yellow`, `Black`. Such a declaration is
/// rare but technically conforming. The pre-pass should NOT silently
/// drop it; the colorant IS a spot from the document's perspective.
///
/// Round 1's policy (which this probe pins) is: surface them as
/// spots. The separation renderer's per-plate path will then have to
/// resolve the collision when it writes plates, but the SIDECAR
/// shape preserves the document's spot list verbatim.
#[test]
fn qa5_1_separation_named_after_process_colorant_is_still_a_spot() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    // Single /Separation /Cyan — collides with the process Cyan name
    // but is declared as a Separation colour space (a spot
    // declaration mechanism per §8.6.6.4).
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << /CS_Cyan [/Separation /Cyan /DeviceCMYK \
                     << /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                     /C0 [0.0 0.0 0.0 0.0] /C1 [1.0 0.0 0.0 0.0] /N 1 >> ]>>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");
    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");
    // The document declares /Cyan as a Separation. /All and /None
    // are the only reserved names §8.6.6.4 filters; /Cyan is not
    // reserved.
    assert_eq!(
        names,
        &["Cyan".to_string()],
        "§8.6.6.4: a Separation named /Cyan is a spot (the name is \
         not reserved). Got {:?}",
        names
    );
}

/// QA5.2: same spot name declared twice (in two different
/// `/Separation` colour space entries with different alternate
/// spaces). `get_page_inks_deep` dedups by ASCII name, so the spot
/// list contains only ONE entry. The dedup is name-only — the
/// alternate-CS information is dropped on the second declaration.
/// This probe pins the dedup behaviour so a future change to
/// preserve-the-first-or-last is visible.
#[test]
fn qa5_2_duplicate_spot_name_dedups_in_spot_set() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    // Two separate /Separation entries both named "SpotInk" with
    // different tint transforms (different alternate-CS C1 values).
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << \
                     /CS_A [/Separation /SpotInk /DeviceCMYK \
                       << /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                          /C0 [0.0 0.0 0.0 0.0] /C1 [1.0 0.0 0.0 0.0] /N 1 >> ] \
                     /CS_B [/Separation /SpotInk /DeviceCMYK \
                       << /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                          /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 0.0 0.0 1.0] /N 1 >> ] \
                     >>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");
    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");

    assert_eq!(
        names.len(),
        1,
        "get_page_inks_deep dedups by name → one /SpotInk \
         declaration regardless of how many CS entries name it; got \
         {:?}",
        names
    );
    assert_eq!(names, &["SpotInk".to_string()]);
}

/// QA5.3: spot name with PDF hex-escape characters (`#20` is a
/// space). The PDF parser already resolves the escape to a literal
/// space; the spot list must surface the *resolved* name with the
/// space, not the encoded form.
#[test]
fn qa5_3_spot_name_with_hex_escaped_space_surfaces_decoded() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << /CS_PMS [/Separation /Special#20Mix#20Ink /DeviceCMYK \
                     << /FunctionType 2 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] \
                     /C0 [0.0 0.0 0.0 0.0] /C1 [0.0 1.0 0.0 0.0] /N 1 >> ]>>";
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");
    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");
    assert_eq!(
        names,
        &["Special Mix Ink".to_string()],
        "§7.3.5: /Special#20Mix#20Ink decodes to \"Special Mix Ink\" \
         (literal spaces). The spot list must carry the decoded form. \
         Got {:?}",
        names
    );
}

/// QA5.4: high spot count — 16 spots declared in a single DeviceN.
/// Sidecar allocates 16 byte-per-pixel planes (16 × 100 × 100 = 160
/// KB for this fixture). Pin the allocation succeeds and addresses
/// indexes 0..16.
#[test]
fn qa5_4_high_spot_count_sixteen_inks_allocates_all_planes() {
    let icc = build_constant_cmyk_icc(135);
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n10 10 80 80 re\nf\n";
    let psfunc = "<< /FunctionType 4 /Domain [0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1] /Range [0 1 0 1 0 1 0 1] \
                  /Length 28 >>\nstream\n{0 0 0 0}\nendstream\nendobj\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> \
                     /ColorSpace << /CS_DN [/DeviceN \
                     [/S01 /S02 /S03 /S04 /S05 /S06 /S07 /S08 \
                      /S09 /S10 /S11 /S12 /S13 /S14 /S15 /S16] \
                     /DeviceCMYK 6 0 R] >>";
    let extra = format!("6 0 obj\n{}", psfunc);
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[&extra]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");
    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");
    assert_eq!(
        names.len(),
        16,
        "16-channel DeviceN must surface all 16 colorants as spots; \
         got {} ({:?})",
        names.len(),
        names
    );
    // Every plane is addressable; the 17th is None.
    for i in 0..16 {
        let p = renderer
            .cmyk_sidecar_spot_plane(i)
            .unwrap_or_else(|| panic!("spot plane {} addressable", i));
        assert_eq!(p.len(), 100 * 100, "plane {} size", i);
        assert!(p.iter().all(|&b| b == 0), "plane {} zero-initialised", i);
    }
    assert!(renderer.cmyk_sidecar_spot_plane(16).is_none());
}

// ===========================================================================
// QA-AREA 6: round-2 seam — zero-byte resting state assumption
// ===========================================================================

/// QA6.1: with the sidecar allocated and zero spot writes (round 1
/// behaviour), every spot plane stays byte-identical to its
/// post-`CmykSidecar::new` state — namely, all zeros. Round 2 will
/// add writes; this probe locks the round-1 baseline so a regression
/// in round 2 (a stray write that escapes the per-op gate) becomes
/// immediately visible.
#[test]
fn qa6_1_round1_spot_planes_stay_zero_through_full_render() {
    let icc = build_constant_cmyk_icc(135);
    // Drive every paint path the renderer has: rgb fill, cmyk fill,
    // smask, transparency, blend mode. None of these should write
    // to spot lanes in round 1.
    let content = "0.5 0.5 0.5 rg\n0 0 100 100 re\nf\n\
                   /Half gs\n0 0 0 1 k\n5 5 90 90 re\nf\n\
                   /Mult gs\n1 0 0 rg\n20 20 60 60 re\nf\n";
    let psfunc = "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1] \
                  /Length 28 >>\nstream\n{0 0 0 0}\nendstream\nendobj\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> \
                                   /Mult << /Type /ExtGState /BM /Multiply >> >> \
                     /ColorSpace << /CS_DN [/DeviceN [/Dieline /Varnish] /DeviceCMYK 6 0 R] >>";
    let extra = format!("6 0 obj\n{}", psfunc);
    let pdf = build_pdf_with_output_intent(content, resources, &icc, &[&extra]);

    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _img = renderer.render_page(&doc, 0).expect("render succeeds");

    let names = renderer.cmyk_sidecar_spot_names().expect("sidecar present");
    assert_eq!(names.len(), 2, "two DeviceN spots discovered");

    for i in 0..2 {
        let plane = renderer
            .cmyk_sidecar_spot_plane(i)
            .expect("spot plane addressable");
        assert!(
            plane.iter().all(|&b| b == 0),
            "round-1 baseline: spot plane {} stays byte-identical-zero \
             across the full render (no production writes wired yet). \
             First non-zero offset: {:?}",
            i,
            plane.iter().position(|&b| b != 0)
        );
    }
}

// ===========================================================================
// QA-AREA 7: tolerance-band guard
// ===========================================================================

/// QA7.1: ensure no tolerance bands have crept into the new code or
/// new tests. Round 2/3/4 of transparency-flattening rejected
/// tolerance bands; round 1 of #46 must hold the same line. This is a
/// compile-time hint via documentation — the actual scan is done by
/// grep in the QA report. The probe documents the intent.
#[test]
fn qa7_1_tolerance_band_guard_documents_intent() {
    // Sentinel probe: round 1 may not introduce tolerance bands on
    // sidecar storage, classification, or detection. If a future
    // round needs a tolerance band, it must be justified at the
    // probe site with a spec citation and pinned with an inequality
    // that names the exact band edge.
    //
    // The grep used in the QA pass:
    //   grep -E '±| \+/- |\babs_diff\b|\bapprox\b|assert!.*< [0-9]+'
    //     src/rendering/sidecar.rs tests/test_46_round1_*
    //
    // If this probe ever needs to be removed, please verify the
    // grep still returns no results.
}
