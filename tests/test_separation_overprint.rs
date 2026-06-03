//! Spec-driven tests for ISO 32000-1 §11.7.4 (Overprint Control) in the
//! separation-plate renderer.
//!
//! Spatial layout shared by every test:
//!
//! ```text
//!   100x100 PDF page.
//!     First  rectangle (the "background"): PDF (10, 10) (50x50)  → image rows 40-90, cols 10-60
//!     Second rectangle (the "overlay")   : PDF (40, 40) (50x50)  → image rows 10-60, cols 40-90
//!     Overlap region                     : PDF (40, 40)-(60, 60) → image rows 40-60, cols 40-60
//!
//!   Sample sites (image-space, top-left origin, +y down):
//!     OVERLAP : (50, 50) — both rectangles paint here
//!     FIRST   : (20, 80) — only the first rectangle paints
//!     SECOND  : (80, 20) — only the second rectangle paints
//!     OUTSIDE : ( 5,  5) — neither paints
//! ```

#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_separation, render_separations, SeparationPlate};

const OVERLAP_X: u32 = 50;
const OVERLAP_Y: u32 = 50;
const FIRST_X: u32 = 20;
const FIRST_Y: u32 = 80;
const SECOND_X: u32 = 80;
const SECOND_Y: u32 = 20;

fn sample(plate: &SeparationPlate, x: u32, y: u32) -> u8 {
    plate.data[(y * plate.width + x) as usize]
}

fn plate<'a>(plates: &'a [SeparationPlate], name: &str) -> &'a SeparationPlate {
    plates
        .iter()
        .find(|p| p.ink_name == name)
        .unwrap_or_else(|| panic!("missing plate {name:?}; have {:?}", names(plates)))
}

fn names(plates: &[SeparationPlate]) -> Vec<&str> {
    plates.iter().map(|p| p.ink_name.as_str()).collect()
}

/// Build a single-page PDF given a content stream and an optional /Resources
/// override fragment (e.g. ColorSpace + ExtGState declarations). Resources
/// always include the entries the caller supplied; the page itself is 100x100.
fn build_pdf(content: &str, resources_inner: &str, extra_objs: &[&str]) -> Vec<u8> {
    let content_bytes = content.as_bytes();
    let mut buf = Vec::new();
    let mut offsets = Vec::new();

    buf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    offsets.push(buf.len());
    let page = format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
         /Contents 4 0 R /Resources << {} >> >>\nendobj\n",
        resources_inner
    );
    buf.extend_from_slice(page.as_bytes());

    offsets.push(buf.len());
    let header = format!("4 0 obj\n<< /Length {} >>\nstream\n", content_bytes.len());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(content_bytes);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    for obj in extra_objs {
        offsets.push(buf.len());
        buf.extend_from_slice(obj.as_bytes());
        if !obj.ends_with('\n') {
            buf.push(b'\n');
        }
    }

    let xref_offset = buf.len();
    buf.extend_from_slice(b"xref\n");
    buf.extend_from_slice(format!("0 {}\n", offsets.len() + 1).as_bytes());
    buf.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len() + 1,
            xref_offset
        )
        .as_bytes(),
    );
    buf
}

// ========================================================================
// DeviceCMYK + default overprint (OP=false, the §11.7.4 spec default)
// ========================================================================

#[test]
fn cmyk_black_default_knocks_out_underlying_magenta() {
    // §11.7.4: with OP=false, painting in DeviceCMYK over an area where
    // Magenta was previously painted erases (sets to 0) the Magenta plate
    // inside the new shape. The first fill paints Magenta = 1.0; the
    // second fill is `0 0 0 1 k` → all four CMYK components specified
    // (M = 0.0), so the M plate is knocked out in the overlap.
    let pdf = build_pdf("0 1 0 0 k\n10 10 50 50 re f\n0 0 0 1 k\n40 40 50 50 re f\n", "", &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");
    let k = plate(&plates, "Black");

    assert_eq!(
        sample(m, OVERLAP_X, OVERLAP_Y),
        0,
        "Magenta should be knocked out in overlap under OP=false default"
    );
    assert!(
        sample(m, FIRST_X, FIRST_Y) > 200,
        "Magenta preserved in first-only region (got {})",
        sample(m, FIRST_X, FIRST_Y)
    );
    assert!(
        sample(k, OVERLAP_X, OVERLAP_Y) > 200,
        "Black painted in overlap (got {})",
        sample(k, OVERLAP_X, OVERLAP_Y)
    );
}

// ========================================================================
// DeviceCMYK + OP=true (overprint enabled, OPM=0 default)
// ========================================================================

#[test]
fn cmyk_black_with_op_true_opm0_still_knocks_out_underlying_magenta() {
    // §11.7.4: OPM=0 means "each source colour component value replaces
    // the value previously painted... no matter what the new value is."
    // So even with OP=true, a 0.0 M component still knocks the M plate
    // to zero. The OPM=1 nonzero rule is what protects zero components.
    let pdf = build_pdf(
        "0 1 0 0 k\n10 10 50 50 re f\n/GS1 gs 0 0 0 1 k\n40 40 50 50 re f\n",
        "/ExtGState << /GS1 << /OP true /op true /OPM 0 >> >>",
        &[],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");

    assert_eq!(
        sample(m, OVERLAP_X, OVERLAP_Y),
        0,
        "OPM=0 + OP=true: zero M component still knocks out M plate"
    );
}

#[test]
fn cmyk_black_with_op_true_opm1_preserves_underlying_magenta() {
    // §11.7.4 (worked example, Example 4.18): under OP=true + OPM=1,
    // `0 0 0 1 k` is equivalent to painting in a DeviceN[C, M, Black]
    // space — zero components are treated as "not specified." So the
    // Magenta plate is left untouched within the overlap.
    let pdf = build_pdf(
        "0 1 0 0 k\n10 10 50 50 re f\n/GS1 gs 0 0 0 1 k\n40 40 50 50 re f\n",
        "/ExtGState << /GS1 << /OP true /op true /OPM 1 >> >>",
        &[],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");
    let k = plate(&plates, "Black");

    assert!(
        sample(m, OVERLAP_X, OVERLAP_Y) > 200,
        "OPM=1: M preserved in overlap (got {})",
        sample(m, OVERLAP_X, OVERLAP_Y)
    );
    assert!(
        sample(k, OVERLAP_X, OVERLAP_Y) > 200,
        "K painted in overlap (got {})",
        sample(k, OVERLAP_X, OVERLAP_Y)
    );
}

// ========================================================================
// ICCBased(N=4) — OPM=1 nonzero-overprint scope per §11.7.4.3
// ========================================================================

/// ICCBased /N 4 colour-space object. The profile stream payload is
/// empty (length 0); the resolver only consults /N to classify as
/// IccCmyk per the renderer's documented 4-component heuristic.
fn icc_cmyk_cs_obj() -> &'static str {
    "5 0 obj\n[/ICCBased 6 0 R]\nendobj\n"
}

fn icc_cmyk_stream_obj() -> &'static str {
    "6 0 obj\n<< /N 4 /Alternate /DeviceCMYK /Length 0 >>\nstream\n\nendstream\nendobj\n"
}

#[test]
fn icc_cmyk_with_op_true_opm1_preserves_underlying_magenta() {
    // §11.7.4.3 OPM scope: "applies only to painting operations that use
    // the current color in the graphics state when the current color
    // space is DeviceCMYK (or is implicitly converted to DeviceCMYK)."
    // An ICCBased N=4 space classified as IccCmyk falls under the
    // "implicitly converted to DeviceCMYK" clause for our renderer
    // (per the module-level ICCBased heuristic). OPM=1 + OP=true on an
    // IccCmyk source should still skip plates whose component is 0.0.
    //
    // Layout: paint M=1 background, then paint `0 0 0 1 scn` through
    // /CS1 (an ICCBased N=4 space) under /GS1 (OP=true / OPM=1). The
    // Magenta plate must be preserved in the overlap.
    let pdf = build_pdf(
        "0 1 0 0 k\n10 10 50 50 re f\n\
         /GS1 gs /CS1 cs 0 0 0 1 scn\n40 40 50 50 re f\n",
        "/ColorSpace << /CS1 5 0 R >> \
         /ExtGState << /GS1 << /OP true /op true /OPM 1 >> >>",
        &[icc_cmyk_cs_obj(), icc_cmyk_stream_obj()],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");
    let k = plate(&plates, "Black");

    assert!(
        sample(m, OVERLAP_X, OVERLAP_Y) > 200,
        "OPM=1 on IccCmyk: M preserved in overlap (got {})",
        sample(m, OVERLAP_X, OVERLAP_Y)
    );
    assert!(
        sample(k, OVERLAP_X, OVERLAP_Y) > 200,
        "K painted in overlap (got {})",
        sample(k, OVERLAP_X, OVERLAP_Y)
    );
}

// ========================================================================
// Separation / Spot ink against CMYK process background
// ========================================================================

fn pantone_185_cs_obj() -> &'static str {
    "5 0 obj\n[/Separation /Pantone-185 /DeviceCMYK << \
        /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] \
        /C1 [0 0.85 0.45 0] /N 1 >>]\nendobj\n"
}

#[test]
fn separation_spot_default_knocks_out_underlying_process_plates() {
    // §11.7.4 default: painting in Separation /Pantone-185 over a
    // Magenta background causes the M plate to be erased to 0 within
    // the spot fill's shape ("areas of unspecified colorants are
    // erased"). The Pantone-185 plate gets the spot fill.
    let pdf = build_pdf(
        "0 1 0 0 k\n10 10 50 50 re f\n/CS1 cs 1 scn\n40 40 50 50 re f\n",
        "/ColorSpace << /CS1 5 0 R >>",
        &[pantone_185_cs_obj()],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");
    let pantone = plate(&plates, "Pantone-185");

    assert_eq!(
        sample(m, OVERLAP_X, OVERLAP_Y),
        0,
        "M knocked out under Pantone fill (OP=false default)"
    );
    assert!(
        sample(pantone, OVERLAP_X, OVERLAP_Y) > 200,
        "Pantone painted in overlap (got {})",
        sample(pantone, OVERLAP_X, OVERLAP_Y)
    );
    assert!(sample(m, FIRST_X, FIRST_Y) > 200, "M preserved outside Pantone shape");
}

#[test]
fn separation_spot_with_op_true_preserves_underlying_process_plates() {
    // §11.7.4 OP=true: anything previously painted in other colorants
    // is left undisturbed. This is the "spot ink overprints process"
    // case that designers explicitly enable for typical packaging art.
    let pdf = build_pdf(
        "0 1 0 0 k\n10 10 50 50 re f\n\
         /GS1 gs /CS1 cs 1 scn\n40 40 50 50 re f\n",
        "/ColorSpace << /CS1 5 0 R >> \
         /ExtGState << /GS1 << /OP true /op true >> >>",
        &[pantone_185_cs_obj()],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");
    let pantone = plate(&plates, "Pantone-185");

    assert!(
        sample(m, OVERLAP_X, OVERLAP_Y) > 200,
        "M preserved under Pantone fill with OP=true (got {})",
        sample(m, OVERLAP_X, OVERLAP_Y)
    );
    assert!(sample(pantone, OVERLAP_X, OVERLAP_Y) > 200, "Pantone painted");
}

#[test]
fn opm_one_does_not_apply_to_separation_source() {
    // §11.7.4.3: the OPM=1 "treat zero as not-specified" rule applies
    // only to DeviceCMYK sources. A Separation/DeviceN source ignores
    // OPM entirely — overprint vs knockout is governed by OP/op alone.
    // Verify by: separation source with /OP true /OPM 1, with the
    // *tint* set to 0.0 (which OPM=1 would treat as "skip" for DeviceCMYK).
    // Under §11.7.4.3 the Pantone plate is still painted with 0.0
    // (knockout-of-self, not skip).
    let pdf = build_pdf(
        "/CS1 cs 1 scn\n10 10 50 50 re f\n\
         /GS1 gs /CS1 cs 0 scn\n40 40 50 50 re f\n",
        "/ColorSpace << /CS1 5 0 R >> \
         /ExtGState << /GS1 << /OP true /op true /OPM 1 >> >>",
        &[pantone_185_cs_obj()],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let pantone = plate(&plates, "Pantone-185");

    // The second fill (tint 0.0) DOES touch the Pantone plate — OPM=1
    // does not apply to Separation sources, so it paints 0.0 (knockout).
    assert_eq!(
        sample(pantone, OVERLAP_X, OVERLAP_Y),
        0,
        "OPM=1 does not skip on Separation source; tint 0.0 still knocks out"
    );
    assert!(
        sample(pantone, FIRST_X, FIRST_Y) > 200,
        "First fill at tint 1.0 preserved outside overlap"
    );
}

// ========================================================================
// /All and /None colorants (§8.6.6.4)
// ========================================================================

fn all_separation_cs_obj() -> &'static str {
    "5 0 obj\n[/Separation /All /DeviceGray << \
        /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 1 >>]\nendobj\n"
}

fn none_separation_cs_obj() -> &'static str {
    "5 0 obj\n[/Separation /None /DeviceGray << \
        /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 1 >>]\nendobj\n"
}

#[test]
fn all_colorant_paints_every_plate_regardless_of_overprint() {
    // §8.6.6.4: /All "refers collectively to all colorants available on
    // an output device". Overprint setting is irrelevant.
    let pdf = build_pdf(
        "/CS1 cs 0.5 scn\n40 40 50 50 re f\n",
        "/ColorSpace << /CS1 5 0 R >>",
        &[all_separation_cs_obj()],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    // /All declares no specific ink name → only the process plates exist
    // in the output. Each should carry tint 0.5.
    for ink in ["Cyan", "Magenta", "Yellow", "Black"] {
        let p = plate(&plates, ink);
        let v = sample(p, OVERLAP_X, OVERLAP_Y);
        assert!(v > 100 && v < 160, "/All should paint {ink} at tint 0.5 (got {v})",);
    }
}

#[test]
fn none_colorant_paints_nothing_and_does_not_knock_out() {
    // §8.6.6.4: a /None Separation "has no effect on the current page."
    // Underlying ink is preserved regardless of OP setting.
    let pdf = build_pdf(
        "0 1 0 0 k\n10 10 50 50 re f\n\
         /CS1 cs 1 scn\n40 40 50 50 re f\n",
        "/ColorSpace << /CS1 5 0 R >>",
        &[none_separation_cs_obj()],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");

    assert!(sample(m, OVERLAP_X, OVERLAP_Y) > 200, "Magenta preserved under /None fill");
}

// ========================================================================
// Save/restore isolation
// ========================================================================

#[test]
fn save_restore_isolates_overprint_state() {
    // q gs(OP=true) … Q must restore OP=false. Two fills, second outside
    // the q…Q block — its OP must NOT inherit the OP=true from inside.
    let pdf = build_pdf(
        "0 1 0 0 k\n10 10 50 50 re f\n\
         q /GS1 gs Q\n\
         0 0 0 1 k\n40 40 50 50 re f\n",
        "/ExtGState << /GS1 << /OP true /op true /OPM 1 >> >>",
        &[],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");

    // After Q, OP is back to default false → M knocks out.
    assert_eq!(
        sample(m, OVERLAP_X, OVERLAP_Y),
        0,
        "OP=true inside q…Q must not leak past Q (default knockout restored)"
    );
}

// ========================================================================
// Stroke vs fill overprint independence (§11.7.4 /OP vs /op)
// ========================================================================

#[test]
fn fill_overprint_only_does_not_affect_stroke() {
    // op=true (fill overprint) but OP defaults to false (stroke knockout).
    // A subsequent stroke should still knock out underlying inks.
    //
    // Layout: full-width Magenta fill, then a stroked rectangle on top.
    // The stroke uses CMYK with M=0 → if stroke knockout is honored, the
    // pixels along the stroke path lose Magenta. If stroke overprint were
    // incorrectly enabled, they would keep it.
    let pdf = build_pdf(
        // Fill the whole page with Magenta first.
        "0 1 0 0 k\n0 0 100 100 re f\n\
         /GS1 gs\n\
         5 w 0 0 0 1 K\n\
         40 40 50 50 re S\n",
        "/ExtGState << /GS1 << /op true /OPM 0 >> >>",
        &[],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");

    // Sample directly on the stroke path. The rectangle's top edge in
    // PDF coords is y=90 (image y=10). With line width 5, the stroke
    // covers image y ∈ [7.5, 12.5]. Pick (60, 10).
    let stroke_sample = sample(m, 60, 10);
    assert_eq!(
        stroke_sample, 0,
        "stroke knockout still applies when only /op (fill) overprint set"
    );

    // And confirm the rectangle interior (not painted by either fill or
    // stroke since we only stroked) keeps its background Magenta.
    let interior_sample = sample(m, 60, 35);
    assert!(
        interior_sample > 200,
        "rectangle interior keeps background M (no fill, no stroke here); got {}",
        interior_sample
    );
}

// ========================================================================
// Plates outside any source space stay untouched
// ========================================================================

#[test]
fn cmyk_fill_does_not_knock_out_unrelated_spot_plate_under_default() {
    // The OP=false default knockout per §11.7.4 applies to colorants
    // *of the source's process model that aren't specified*. For a
    // DeviceCMYK source over a Pantone-185 background, the Pantone
    // plate is NOT a colorant of the source's process model — it's a
    // spot that exists on the output device. The default still erases
    // it inside the painted shape per the spec's text: "the
    // corresponding areas of unspecified colorants" — Pantone-185 is
    // unspecified by the DeviceCMYK source.
    //
    // This pins what every prepress engine does: a CMYK black fill
    // over a spot background DOES knock out the spot in the overlap.
    let pdf = build_pdf(
        "/CS1 cs 1 scn\n10 10 50 50 re f\n\
         0 0 0 1 k\n40 40 50 50 re f\n",
        "/ColorSpace << /CS1 5 0 R >>",
        &[pantone_185_cs_obj()],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let pantone = plate(&plates, "Pantone-185");

    assert_eq!(
        sample(pantone, OVERLAP_X, OVERLAP_Y),
        0,
        "CMYK source under default knocks out spot plate in overlap"
    );
}

#[test]
fn cmyk_fill_with_op_true_preserves_unrelated_spot_plate() {
    // Same scenario with OP=true: spot plate preserved.
    let pdf = build_pdf(
        "/CS1 cs 1 scn\n10 10 50 50 re f\n\
         /GS1 gs 0 0 0 1 k\n40 40 50 50 re f\n",
        "/ColorSpace << /CS1 5 0 R >> \
         /ExtGState << /GS1 << /OP true /op true >> >>",
        &[pantone_185_cs_obj()],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let pantone = plate(&plates, "Pantone-185");

    assert!(
        sample(pantone, OVERLAP_X, OVERLAP_Y) > 200,
        "OP=true: spot plate preserved under CMYK overlap (got {})",
        sample(pantone, OVERLAP_X, OVERLAP_Y)
    );
}

// ========================================================================
// Form XObject inheritance (§8.10.1)
// ========================================================================

#[test]
fn overprint_propagates_into_form_xobject() {
    // §8.10.1: a Form XObject's initial graphics state is the calling
    // context's graphics state. The caller sets OP=true + OPM=1, then
    // invokes a form whose content is `0 0 0 1 k 40 40 50 50 re f`.
    // Under the inherited OPM=1 nonzero rule, the zero M component is
    // treated as "not specified" and the M plate is preserved. Without
    // inheritance, OPM defaults back to 0 at the form boundary and the
    // M plate gets knocked out — that failure is what this test catches.
    let content = "0 1 0 0 k\n10 10 50 50 re f\n\
                   q /GS1 gs /Fm Do Q\n";
    let form_content = "0 0 0 1 k\n40 40 50 50 re f\n";
    let form_obj = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    let pdf = build_pdf(
        content,
        "/ExtGState << /GS1 << /OP true /op true /OPM 1 >> >> \
         /XObject << /Fm 5 0 R >>",
        &[&form_obj],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");
    let k = plate(&plates, "Black");

    assert!(
        sample(m, OVERLAP_X, OVERLAP_Y) > 200,
        "M preserved inside Form XObject under inherited OP=true + OPM=1 (got {})",
        sample(m, OVERLAP_X, OVERLAP_Y)
    );
    assert!(sample(k, OVERLAP_X, OVERLAP_Y) > 200, "Black painted by form's CMYK fill");
}

#[test]
fn overprint_caller_unaffected_by_inner_form_overprint_changes() {
    // The form's OP changes must not leak back to the caller. Form
    // content sets /GS1 gs (OP=true OPM=1), then the form ends. After
    // the `Do`, the caller's gs is restored from the q/Q frame and OP
    // is back to the original false. A subsequent caller fill must
    // knock out per the default.
    let content = "0 1 0 0 k\n10 10 50 50 re f\n\
                   q /Fm Do Q\n\
                   0 0 0 1 k\n40 40 50 50 re f\n";
    // Form content sets OP=true / OPM=1 internally; this should NOT
    // affect the outer scope after Do returns.
    let form_content = "/GS1 gs\n";
    let form_obj = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << /ExtGState << /GS1 << /OP true /op true /OPM 1 >> >> >> \
         /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    let pdf = build_pdf(content, "/XObject << /Fm 5 0 R >>", &[&form_obj]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let m = plate(&plates, "Magenta");

    assert_eq!(
        sample(m, OVERLAP_X, OVERLAP_Y),
        0,
        "Form's OP/OPM state must not leak past the q/Q frame"
    );
}

// ========================================================================
// Sanity: outside both shapes, plates are zero
// ========================================================================

#[test]
fn outside_both_shapes_all_plates_are_zero() {
    let pdf = build_pdf("0 1 0 0 k\n10 10 50 50 re f\n0 0 0 1 k\n40 40 50 50 re f\n", "", &[]);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    for p in &plates {
        assert_eq!(
            sample(p, 5, 5),
            0,
            "plate {} should be zero outside any painted shape",
            p.ink_name
        );
    }
}

// Suppress dead-code warnings on the SECOND_* constants if a future test
// removes the only use; keeping them documented for the spatial layout.
#[allow(dead_code)]
fn _unused_pinned_layout() -> (u32, u32) {
    (SECOND_X, SECOND_Y)
}

// Suppress dead-code warning for render_separation import if not exercised
// in this file — it's part of the public API surface that may be used by
// follow-up tests in this module.
#[allow(dead_code)]
fn _api_surface(doc: &PdfDocument) -> Option<SeparationPlate> {
    render_separation(doc, 0, "Cyan", 72).ok()
}
