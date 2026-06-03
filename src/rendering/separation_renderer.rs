//! Separation plate renderer.
//!
//! Renders individual ink separation plates as grayscale images where
//! pixel intensity represents the tint percentage of that ink at each point.
//! Used in prepress workflows, ink coverage analysis, and ML pipelines
//! that process packaging/label PDFs.
//!
//! # ICCBased heuristic
//!
//! When a fill color space resolves to an `ICCBased` array, this renderer
//! does **not** parse the embedded ICC profile. Instead it inspects the
//! component count of the current fill/stroke color: a 4-component
//! `ICCBased` space is treated as CMYK (component order C, M, Y, K), a
//! 3-component space is treated as RGB (skipped — no separation routing),
//! and a 1-component space is treated as Gray (skipped). This matches
//! the convention used by Adobe Illustrator and InDesign when exporting
//! to PDF/X-1a and PDF/X-4 with CMYK working spaces. PDFs that rely on
//! lab-CMYK profile interpretation for separation routing are out of
//! scope for this renderer; they are rare in prepress workflows that
//! ship separated artwork.
//!
//! # Limitations
//!
//! The following classes of content are recognised by the operator
//! walker but not actually painted into the plate:
//!
//! - **Raster image XObjects** (`Do` with `Subtype /Image`), including
//!   DeviceN / Separation-encoded TIFFs and CMYK photographs. The
//!   sample data is dropped. Vector artwork inside Form XObjects is
//!   recursed into and rendered normally; spot / DeviceN ink
//!   *declarations* in nested Form XObject `/Resources` are also
//!   surfaced as plates via
//!   [`crate::document::PdfDocument::get_page_inks_deep`] even when the
//!   form's local content stream doesn't paint them.
//! - **Shading patterns** (`sh` operator) — gradients used as fills.
//! - **Tiling and shading patterns** invoked via `scn` / `SCN` with a
//!   `/Pattern` colour space.
//! - **Inline images** (`BI` / `ID` / `EI`).
//! - **Page annotations.** [`render_separations`] renders only the
//!   page's content stream; annotation appearance streams are not
//!   walked, in contrast to [`super::page_renderer`] which composites
//!   annotation appearances on top of the page.
//!
//! These are intentional v1 omissions: the primary use case is
//! vector-based prepress artwork (dielines, varnish layers, spot-PMS
//! text and shapes). PDFs that rely on raster spot-channel data will
//! produce incomplete plates and should be flagged at the caller.
//!
//! # Transparency
//!
//! Plate output is opaque: the renderer treats `fill_alpha` / `stroke_alpha`
//! from ExtGState (`/CA`, `/ca`) and the blend mode (`/BM`) as if both were
//! `1.0` / `Normal`. This is intentional — a separation plate represents ink
//! coverage on the press, not transparent compositing. Callers who need the
//! transparent intent (e.g. a 50%-alpha spot text overlay) should evaluate it
//! against the underlying content with [`super::page_renderer`] first.
//!
//! # Overprint
//!
//! The renderer implements the per-plate overprint model defined in
//! ISO 32000-1 §11.7.4 ("Overprint Control"). The ExtGState entries
//! `/OP` (stroke), `/op` (non-stroke), and `/OPM` (overprint mode) are
//! parsed and applied to the graphics state.
//!
//! - **Default (`OP = false`):** for every plate, the spec rule "areas
//!   of unspecified colorants are erased (painted with a tint value of
//!   0.0)" applies. A DeviceCMYK fill knocks out underlying Cyan,
//!   Magenta, Yellow, Black, *and* any spot inks within its shape; a
//!   Separation `/Pantone-185` fill knocks out underlying process and
//!   other-spot plates within its shape. This is the standard
//!   per-plate prepress convention.
//! - **`OP = true`:** plates outside the source's colorant set are left
//!   untouched. Designers use this to overlay spot inks on process
//!   backgrounds without knocking them out (the typical packaging /
//!   label authoring workflow).
//! - **`OPM = 1` (Adobe nonzero overprint):** when the source colour
//!   space is DeviceCMYK and overprint is enabled, a component value of
//!   exactly `0.0` is treated as "colorant not specified" — the
//!   matching plate is left untouched. Per §11.7.4.3, OPM applies only
//!   to DeviceCMYK sources; Separation and DeviceN content is
//!   unaffected by OPM and routes through OP/op alone.
//!
//! Overprint state participates in `q`/`Q` save/restore via the existing
//! graphics-state stack and propagates into Form XObjects per §8.10.1.
//! The decision happens in `tint_for_ink`, which returns either
//! `PaintAction::Paint(tint)` (write tint into the plate; 0.0 = knockout)
//! or `PaintAction::Skip` (leave the plate untouched). Spot/DeviceN
//! sources route to their named plates regardless of overprint, matching
//! the inherent behavior of real separation devices.
#![allow(
    clippy::field_reassign_with_default,
    clippy::ptr_arg,
    clippy::only_used_in_recursion
)]

use std::collections::HashMap;
use std::sync::Arc;

use tiny_skia::{FillRule, Mask, PathBuilder, Pixmap, Transform};

use crate::content::graphics_state::{GraphicsState, GraphicsStateStack, Matrix};
use crate::content::operators::{Operator, TextElement};
use crate::content::parser::parse_content_stream;
use crate::document::PdfDocument;
use crate::error::{Error, Result};
use crate::fonts::FontInfo;
use crate::object::Object;

use super::ext_gstate::{parse_ext_g_state_inner, ParsedExtGState};
use super::text_rasterizer::TextRasterizer;

/// A rendered separation plate for a single ink.
///
/// The pixel convention is **ML/QC-friendly**: `value == ink coverage`.
/// 0 means no ink on paper at that pixel, 255 means full tint coverage.
/// To display the plate as black ink on white paper (prepress viewer
/// convention) invert before showing: `display = 255 - value`.
#[derive(Debug, Clone)]
pub struct SeparationPlate {
    /// Ink name (e.g., "Cyan", "PANTONE 185 C", "Dieline").
    pub ink_name: String,
    /// Grayscale pixel data, row-major, top-left origin.
    /// 0 = no ink, 255 = full tint. `data.len() == width * height`.
    pub data: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

/// Render all separation plates for a page.
///
/// Returns one [`SeparationPlate`] per ink. Process inks (Cyan, Magenta,
/// Yellow, Black) are always emitted; if the page uses no CMYK content
/// those plates will be all-zero. Spot inks are emitted only when the
/// page's resource dictionary declares a `Separation` or `DeviceN` colour
/// space that names them.
///
/// Each plate is a grayscale image where pixel intensity equals the
/// tint percentage of that ink (255 = full tint, 0 = no ink).
///
/// # Performance
///
/// The content stream is parsed **once** and the operator walk dispatches
/// paint operations to all referenced plates in parallel. Form XObjects
/// are also recursed into once per page. Unreferenced inks short-circuit
/// to an all-zero plate before any pixmap is allocated.
pub fn render_separations(
    doc: &PdfDocument,
    page_num: usize,
    dpi: u32,
) -> Result<Vec<SeparationPlate>> {
    let inks = collect_page_inks(doc, page_num)?;
    if inks.is_empty() {
        return Ok(Vec::new());
    }

    // Pre-parse the content stream once to detect which inks are actually
    // referenced. Plates for unreferenced inks short-circuit to an empty
    // pixmap and skip the per-plate operator walk entirely (O6).
    let referenced = collect_referenced_inks(doc, page_num)?;

    render_plates_for_inks(doc, page_num, dpi, &inks, &referenced)
}

/// Render a single ink separation plate for a page.
///
/// Returns a grayscale image where pixel intensity = tint percentage
/// of the named ink. If the ink is not present on the page, the plate
/// is all zeros.
///
/// This is a thin wrapper over the multi-ink path; if you need every
/// plate on a page, call [`render_separations`] instead — it walks the
/// content stream once for all inks together.
pub fn render_separation(
    doc: &PdfDocument,
    page_num: usize,
    ink_name: &str,
    dpi: u32,
) -> Result<SeparationPlate> {
    // Always walk operators for the requested ink — the per-page short-circuit
    // in [`render_separations`] is an optimisation that scans the resource
    // declarations to skip inks that are *definitely* unused. For the single-ink
    // entry point the caller has already named the ink they want, and the
    // scanner can miss inks reached via DefaultRGB/DefaultGray remapping
    // through colour operators like `rg`/`g`. Treat the named ink as referenced
    // and let the operator walk produce an honest plate.
    let inks = vec![ink_name.to_string()];
    let referenced = inks.clone();
    let mut plates = render_plates_for_inks(doc, page_num, dpi, &inks, &referenced)?;
    plates
        .pop()
        .ok_or_else(|| Error::InvalidPdf("render_separation: no plate produced".to_string()))
}

/// Core multi-ink rendering: allocate one pixmap per referenced ink,
/// walk the content stream once, and extract grayscale data from each.
fn render_plates_for_inks(
    doc: &PdfDocument,
    page_num: usize,
    dpi: u32,
    inks: &[String],
    referenced: &[String],
) -> Result<Vec<SeparationPlate>> {
    let (width, height, base_transform) = compute_page_extent(doc, page_num, dpi)?;

    // Partition inks into "needs rendering" vs "short-circuit to empty plate".
    // We track the original index so the output order matches `inks`.
    let mut render_indices: Vec<usize> = Vec::new();
    let mut empty_indices: Vec<usize> = Vec::new();
    for (i, ink) in inks.iter().enumerate() {
        if referenced.iter().any(|r| r == ink) {
            render_indices.push(i);
        } else {
            empty_indices.push(i);
        }
    }

    // Build pixmaps and a parallel `target_inks` slice for the inks we
    // actually need to walk operators for.
    let mut pixmaps: Vec<Pixmap> = Vec::with_capacity(render_indices.len());
    for _ in &render_indices {
        let pixmap = Pixmap::new(width, height)
            .ok_or_else(|| Error::InvalidPdf("Failed to create separation pixmap".to_string()))?;
        pixmaps.push(pixmap);
    }
    let target_inks: Vec<&str> = render_indices.iter().map(|&i| inks[i].as_str()).collect();

    if !pixmaps.is_empty() {
        let resources = doc.get_page_resources(page_num)?;
        let color_spaces = load_color_spaces(doc, &resources)?;
        let fonts = load_fonts(doc, &resources);
        let text_rasterizer = TextRasterizer::new();

        let content_data = doc.get_page_content_data(page_num)?;
        let operators = parse_content_stream(&content_data)?;

        let mut ctx = SeparationContext {
            doc,
            text_rasterizer: &text_rasterizer,
            fonts: &fonts,
        };

        execute_separation_operators(
            &mut pixmaps,
            base_transform,
            &operators,
            &mut ctx,
            &resources,
            &color_spaces,
            None,
            &target_inks,
        )?;
    }

    // Re-assemble in original ink order: empty plates for unreferenced
    // inks, extracted R channel for rendered ones.
    let pixel_count = (width as usize) * (height as usize);
    let mut result: Vec<Option<SeparationPlate>> = (0..inks.len()).map(|_| None).collect();

    for (k, &i) in render_indices.iter().enumerate() {
        let mut data = vec![0u8; pixel_count];
        let rgba = pixmaps[k].data();
        for j in 0..pixel_count {
            data[j] = rgba[j * 4];
        }
        result[i] = Some(SeparationPlate {
            ink_name: inks[i].clone(),
            data,
            width,
            height,
        });
    }
    for &i in &empty_indices {
        result[i] = Some(SeparationPlate {
            ink_name: inks[i].clone(),
            data: vec![0u8; pixel_count],
            width,
            height,
        });
    }

    Ok(result
        .into_iter()
        .map(|o| o.expect("plate filled"))
        .collect())
}

/// Collect all ink names present on a page.
///
/// CMYK is always returned regardless of whether the page actually uses
/// CMYK content; unused process plates are filtered out by the per-plate
/// short-circuit in [`render_separations`].
///
/// Spot inks come from [`PdfDocument::get_page_inks_deep`], which walks
/// the page's content stream into nested Form XObjects (§8.10) so spots
/// declared in form-local resources are discovered.
fn collect_page_inks(doc: &PdfDocument, page_num: usize) -> Result<Vec<String>> {
    let mut inks = vec![
        "Cyan".to_string(),
        "Magenta".to_string(),
        "Yellow".to_string(),
        "Black".to_string(),
    ];

    let spot_inks = doc.get_page_inks_deep(page_num)?;
    for ink in spot_inks {
        if !inks.contains(&ink) {
            inks.push(ink);
        }
    }

    Ok(inks)
}

/// Walk the content stream (and any Form XObjects it references) and
/// collect every ink name that could possibly appear on the page.
fn collect_referenced_inks(doc: &PdfDocument, page_num: usize) -> Result<Vec<String>> {
    let resources = doc.get_page_resources(page_num)?;
    let color_spaces = load_color_spaces(doc, &resources)?;
    let content_data = doc.get_page_content_data(page_num)?;
    let operators = parse_content_stream(&content_data)?;
    let mut referenced: Vec<String> = Vec::new();
    let mut visited: Vec<String> = Vec::new();
    scan_operators_for_inks(
        &operators,
        doc,
        &resources,
        &color_spaces,
        &mut referenced,
        &mut visited,
    )?;
    Ok(referenced)
}

fn scan_operators_for_inks(
    operators: &[Operator],
    doc: &PdfDocument,
    resources: &Object,
    color_spaces: &HashMap<String, Object>,
    referenced: &mut Vec<String>,
    visited: &mut Vec<String>,
) -> Result<()> {
    let xobjects = match resources {
        Object::Dictionary(rd) => rd.get("XObject").and_then(|o| doc.resolve_object(o).ok()),
        _ => None,
    };

    let push = |list: &mut Vec<String>, name: &str| {
        if !list.iter().any(|s| s == name) {
            list.push(name.to_string());
        }
    };

    for op in operators {
        match op {
            Operator::SetFillCmyk { .. } | Operator::SetStrokeCmyk { .. } => {
                push(referenced, "Cyan");
                push(referenced, "Magenta");
                push(referenced, "Yellow");
                push(referenced, "Black");
            },
            Operator::SetFillColorSpace { name } | Operator::SetStrokeColorSpace { name } => {
                inks_from_space(name, color_spaces, resources, doc, referenced);
            },
            Operator::Do { name } => {
                if visited.iter().any(|s| s == name) {
                    continue;
                }
                visited.push(name.clone());
                if let Some(xobj_dict) = xobjects.as_ref().and_then(|o| o.as_dict()) {
                    if let Some(xobj_ref_obj) = xobj_dict.get(name) {
                        if let Ok(xobj) = doc.resolve_object(xobj_ref_obj) {
                            if let Object::Stream { ref dict, .. } = xobj {
                                let subtype = dict.get("Subtype").and_then(|o| o.as_name());
                                if subtype == Some("Form") {
                                    let stream_data = if let Some(r) = xobj_ref_obj.as_reference() {
                                        doc.decode_stream_with_encryption(&xobj, r)?
                                    } else {
                                        xobj.decode_stream_data()?
                                    };
                                    let form_resources = if let Some(res) = dict.get("Resources") {
                                        doc.resolve_object(res)?
                                    } else {
                                        resources.clone()
                                    };
                                    let form_cs = load_color_spaces(doc, &form_resources)?;
                                    let mut merged_cs = color_spaces.clone();
                                    merged_cs.extend(form_cs);
                                    if let Ok(form_ops) = parse_content_stream(&stream_data) {
                                        scan_operators_for_inks(
                                            &form_ops,
                                            doc,
                                            &form_resources,
                                            &merged_cs,
                                            referenced,
                                            visited,
                                        )?;
                                    }
                                }
                            }
                        }
                    }
                }
            },
            _ => {},
        }
    }
    Ok(())
}

fn inks_from_space(
    space_name: &str,
    color_spaces: &HashMap<String, Object>,
    resources: &Object,
    doc: &PdfDocument,
    out: &mut Vec<String>,
) {
    // Honour DefaultCMYK/RGB/Gray remap (RED #2 — see resolve_color_space).
    let space = resolve_color_space(space_name, color_spaces, resources, doc);
    match space {
        ResolvedSpace::Cmyk | ResolvedSpace::IccCmyk => {
            for ink in ["Cyan", "Magenta", "Yellow", "Black"] {
                if !out.iter().any(|s| s == ink) {
                    out.push(ink.to_string());
                }
            }
        },
        ResolvedSpace::Separation(name) => {
            // §8.6.6.4: /All marks every output separation — list CMYK so the
            // per-plate short-circuit in render_separations doesn't skip them.
            // /None paints nothing and never names a plate.
            if name == "All" {
                for ink in ["Cyan", "Magenta", "Yellow", "Black"] {
                    if !out.iter().any(|s| s == ink) {
                        out.push(ink.to_string());
                    }
                }
            } else if name != "None" && !out.iter().any(|s| s == &name) {
                out.push(name);
            }
        },
        ResolvedSpace::DeviceN(names) => {
            for n in names {
                if n == "All" {
                    for ink in ["Cyan", "Magenta", "Yellow", "Black"] {
                        if !out.iter().any(|s| s == ink) {
                            out.push(ink.to_string());
                        }
                    }
                } else if n != "None" && !out.iter().any(|s| s == &n) {
                    out.push(n);
                }
            }
        },
        ResolvedSpace::Rgb
        | ResolvedSpace::Gray
        | ResolvedSpace::IccRgb
        | ResolvedSpace::IccGray
        | ResolvedSpace::Unknown => {},
    }
}

/// Page extent computation (width/height in pixels and the base
/// transform that maps PDF user space into the pixmap).
fn compute_page_extent(
    doc: &PdfDocument,
    page_num: usize,
    dpi: u32,
) -> Result<(u32, u32, Transform)> {
    let page_info = doc.get_page_info(page_num)?;
    let media_box = page_info.media_box;

    let rotation = page_info.rotation % 360;
    let (page_w, page_h) = if rotation == 90 || rotation == 270 {
        (media_box.height, media_box.width)
    } else {
        (media_box.width, media_box.height)
    };
    let scale = dpi as f32 / 72.0;
    let width = (page_w * scale).ceil() as u32;
    let height = (page_h * scale).ceil() as u32;

    let base_transform = match rotation {
        90 => Transform::from_translate(-media_box.x, -media_box.y)
            .post_concat(Transform::from_row(0.0, scale, scale, 0.0, 0.0, 0.0)),
        180 => Transform::from_translate(-media_box.x, -media_box.y)
            .post_scale(-scale, scale)
            .post_translate(media_box.width * scale, 0.0),
        270 => Transform::from_translate(-media_box.x, -media_box.y).post_concat(
            Transform::from_row(0.0, scale, -scale, 0.0, media_box.height * scale, 0.0),
        ),
        _ => Transform::from_translate(-media_box.x, -media_box.y)
            .post_scale(scale, -scale)
            .post_translate(0.0, page_h * scale),
    };

    Ok((width, height, base_transform))
}

/// Resolved colour-space classification used by the separation pipeline.
#[derive(Debug, Clone)]
enum ResolvedSpace {
    Cmyk,
    Rgb,
    Gray,
    Separation(String),
    DeviceN(Vec<String>),
    /// ICCBased with a 4-component profile (treated as CMYK by heuristic).
    IccCmyk,
    /// ICCBased with 3 components (RGB).
    IccRgb,
    /// ICCBased with 1 component (Gray).
    IccGray,
    Unknown,
}

/// Resolve a colour-space name to a known classification.
///
/// Handles ISO 32000-1 §8.6.5.6: when the named space is one of the
/// Device families and the resource dictionary defines a corresponding
/// `Default*` entry, the Default mapping is consulted instead.
fn resolve_color_space(
    space_name: &str,
    color_spaces: &HashMap<String, Object>,
    resources: &Object,
    doc: &PdfDocument,
) -> ResolvedSpace {
    // Direct Device* names — try DefaultCMYK / DefaultRGB / DefaultGray remap first.
    let default_key = match space_name {
        "DeviceCMYK" | "CMYK" => Some("DefaultCMYK"),
        "DeviceRGB" | "RGB" => Some("DefaultRGB"),
        "DeviceGray" | "G" => Some("DefaultGray"),
        _ => None,
    };
    if let Some(key) = default_key {
        if let Some(default) = color_spaces.get(key) {
            // Walk into the default array as a fresh classification.
            return classify_resolved(default, color_spaces, resources, doc);
        }
        return match key {
            "DefaultCMYK" => ResolvedSpace::Cmyk,
            "DefaultRGB" => ResolvedSpace::Rgb,
            _ => ResolvedSpace::Gray,
        };
    }

    if let Some(cs_obj) = color_spaces.get(space_name) {
        classify_resolved(cs_obj, color_spaces, resources, doc)
    } else {
        ResolvedSpace::Unknown
    }
}

/// Classify a colour-space object (either an array or a name) into a
/// [`ResolvedSpace`]. Used both as the entry point from a resource-dict
/// lookup and recursively when an array starts with a name that is
/// itself a device alias.
fn classify_resolved(
    cs_obj: &Object,
    color_spaces: &HashMap<String, Object>,
    resources: &Object,
    doc: &PdfDocument,
) -> ResolvedSpace {
    // Plain name (e.g. /DeviceCMYK as the array's tail target).
    if let Some(name) = cs_obj.as_name() {
        return match name {
            "DeviceCMYK" | "CMYK" => ResolvedSpace::Cmyk,
            "DeviceRGB" | "RGB" => ResolvedSpace::Rgb,
            "DeviceGray" | "G" => ResolvedSpace::Gray,
            _ => resolve_color_space(name, color_spaces, resources, doc),
        };
    }

    let arr = match cs_obj.as_array() {
        Some(a) => a,
        None => return ResolvedSpace::Unknown,
    };
    let type_name = match arr.first().and_then(|o| o.as_name()) {
        Some(n) => n,
        None => return ResolvedSpace::Unknown,
    };
    match type_name {
        "DeviceCMYK" | "CMYK" => ResolvedSpace::Cmyk,
        "DeviceRGB" | "RGB" => ResolvedSpace::Rgb,
        "DeviceGray" | "G" => ResolvedSpace::Gray,
        "Separation" => {
            let ink = arr
                .get(1)
                .and_then(|o| o.as_name())
                .map(|s| s.to_string())
                .unwrap_or_default();
            ResolvedSpace::Separation(ink)
        },
        "DeviceN" => {
            if let Some(Object::Array(ink_names)) = arr.get(1) {
                let names = ink_names
                    .iter()
                    .filter_map(|o| o.as_name().map(|s| s.to_string()))
                    .collect();
                ResolvedSpace::DeviceN(names)
            } else {
                ResolvedSpace::Unknown
            }
        },
        "ICCBased" => {
            // ICCBased: read /N from the stream dict to pick the component-count
            // interpretation. Unknown / unreachable / unsupported N → Unknown,
            // since fabricating CMYK plate values from an N=2 or N=5 profile
            // would silently corrupt output. tint_for_ink skips Unknown spaces.
            if let Some(stream_obj) = arr.get(1) {
                if let Ok(resolved) = doc.resolve_object(stream_obj) {
                    if let Object::Stream { ref dict, .. } = resolved {
                        if let Some(n) = dict.get("N").and_then(|o| o.as_integer()) {
                            return match n {
                                4 => ResolvedSpace::IccCmyk,
                                3 => ResolvedSpace::IccRgb,
                                1 => ResolvedSpace::IccGray,
                                _ => ResolvedSpace::Unknown,
                            };
                        }
                    }
                }
            }
            ResolvedSpace::Unknown
        },
        _ => ResolvedSpace::Unknown,
    }
}

/// Load color space definitions from page resources.
fn load_color_spaces(doc: &PdfDocument, resources: &Object) -> Result<HashMap<String, Object>> {
    let mut color_spaces = HashMap::new();
    if let Object::Dictionary(res_dict) = resources {
        if let Some(cs_obj) = res_dict.get("ColorSpace") {
            let cs_dict_obj = doc.resolve_object(cs_obj)?;
            if let Some(cs_dict) = cs_dict_obj.as_dict() {
                for (name, o) in cs_dict {
                    if let Ok(resolved_cs) = doc.resolve_object(o) {
                        color_spaces.insert(name.clone(), resolved_cs);
                    }
                }
            }
        }
    }
    Ok(color_spaces)
}

/// Load font resources for the page. Failures are swallowed (text using
/// unloadable fonts is dropped); this matches the page renderer's
/// best-effort behaviour and keeps separation rendering robust on PDFs
/// with corrupt or missing fonts.
fn load_fonts(doc: &PdfDocument, resources: &Object) -> HashMap<String, Arc<FontInfo>> {
    let mut fonts = HashMap::new();
    if let Object::Dictionary(res_dict) = resources {
        if let Some(font_obj) = res_dict.get("Font") {
            if let Ok(font_dict_obj) = doc.resolve_object(font_obj) {
                if let Some(font_dict) = font_dict_obj.as_dict() {
                    for (name, f_obj) in font_dict {
                        if let Ok(info) = doc.get_or_load_font_for_rendering(f_obj) {
                            fonts.insert(name.clone(), info);
                        }
                    }
                }
            }
        }
    }
    fonts
}

/// Per-plate routing decision for a single paint operation, after applying
/// the overprint rules of ISO 32000-1 §11.7.4.
///
/// - [`PaintAction::Paint`] writes the given tint into the plate. A tint
///   of 0.0 is the spec-default "knockout" — the existing
///   [`fill_separation`] / [`stroke_separation`] use opaque source-over,
///   so writing 0.0 erases any underlying ink at the touched pixels.
/// - [`PaintAction::Skip`] leaves the plate completely untouched. Used
///   when (a) the source colour space doesn't reference this plate and
///   overprint is enabled, or (b) the source is DeviceCMYK with OPM=1
///   and the component is exactly 0.0 (the "Adobe nonzero overprint"
///   rule, §11.7.4).
enum PaintAction {
    Paint(f32),
    Skip,
}

/// Decide how the current paint operation contributes to `target_ink`,
/// honoring ISO 32000-1 §11.7.4 (Overprint Control).
///
/// The decision tree:
///
/// ```text
/// For each plate P, source colour space S with component vector c[]:
///
///   if S = Separation(/All):                              Paint(c[0])
///   if S = Separation(/None) or empty components:         Skip
///   if S = Separation(name) and name == P:                Paint(c[0])
///   if S = Separation(name) and name != P:
///         overprint? Skip : Paint(0.0)                    // §11.7.4 default knockout
///
///   if S = DeviceN(names) and P in names:                 Paint(c[index_of_P])
///   if S = DeviceN(names) and P not in names:
///         overprint? Skip : Paint(0.0)
///
///   if S = DeviceCMYK / IccCmyk:
///         if P in {C, M, Y, K}:
///             overprint && opm == 1 && tint == 0.0 ? Skip : Paint(tint)
///         else:                                            // spot plate
///             overprint? Skip : Paint(0.0)                 // §11.7.4 default knockout
///
///   if S = RGB/Gray/IccRgb/IccGray:                       Skip
/// ```
fn tint_for_ink(
    fill: bool,
    gs: &GraphicsState,
    color_spaces: &HashMap<String, Object>,
    resources: &Object,
    doc: &PdfDocument,
    target_ink: &str,
    fill_components: &[f32],
    stroke_components: &[f32],
) -> PaintAction {
    let space_name = if fill {
        &gs.fill_color_space
    } else {
        &gs.stroke_color_space
    };
    let components = if fill {
        fill_components
    } else {
        stroke_components
    };
    let overprint = if fill {
        gs.fill_overprint
    } else {
        gs.stroke_overprint
    };
    // §11.7.4.3: OPM applies only when the source is DeviceCMYK (or implicit
    // conversion thereto). The match arms below check this where relevant.
    let opm = gs.overprint_mode;

    // Default action when the source colour space doesn't name the
    // target plate: under OP=true, leave it alone; under OP=false (the
    // spec default), erase it to 0.0 ("areas of unspecified colorants
    // are erased" — §11.7.4).
    let other_plate_action = if overprint {
        PaintAction::Skip
    } else {
        PaintAction::Paint(0.0)
    };

    let resolved = resolve_color_space(space_name, color_spaces, resources, doc);
    match resolved {
        ResolvedSpace::Cmyk | ResolvedSpace::IccCmyk => {
            let cmyk_state = if fill {
                gs.fill_color_cmyk
            } else {
                gs.stroke_color_cmyk
            };
            let (c, m, y, k) = if let Some(v) = cmyk_state {
                v
            } else if components.len() >= 4 {
                (components[0], components[1], components[2], components[3])
            } else {
                return PaintAction::Skip;
            };
            let tint = match target_ink {
                "Cyan" => c,
                "Magenta" => m,
                "Yellow" => y,
                "Black" => k,
                // Spot plate — not in DeviceCMYK's colorant set.
                _ => return other_plate_action,
            };
            // §11.7.4 OPM=1 nonzero overprint: zero source components on
            // DeviceCMYK are treated as "not specified" — leave the
            // matching plate untouched. OPM=0 (default) paints zero,
            // which erases (knocks out) the plate.
            if overprint && opm == 1 && tint == 0.0 {
                PaintAction::Skip
            } else {
                PaintAction::Paint(tint)
            }
        },
        ResolvedSpace::Rgb
        | ResolvedSpace::Gray
        | ResolvedSpace::IccRgb
        | ResolvedSpace::IccGray => {
            // §11.7.4: overprint is a separation-space concept. RGB / Gray
            // sources do not route to ink plates at all. Converting them
            // would require a tint transform and is intentionally not done.
            PaintAction::Skip
        },
        ResolvedSpace::Separation(ink) => {
            // §8.6.6.4: /All paints to every plate; /None paints nothing.
            if components.is_empty() || ink == "None" {
                return PaintAction::Skip;
            }
            if ink == "All" {
                return PaintAction::Paint(components[0]);
            }
            if ink == target_ink {
                PaintAction::Paint(components[0])
            } else {
                other_plate_action
            }
        },
        ResolvedSpace::DeviceN(names) => {
            for (i, n) in names.iter().enumerate() {
                if n == "None" {
                    continue;
                }
                if (n == "All" || n == target_ink) && i < components.len() {
                    return PaintAction::Paint(components[i]);
                }
            }
            other_plate_action
        },
        ResolvedSpace::Unknown => PaintAction::Skip,
    }
}

/// Per-render shared context (read-only) passed through the operator
/// walk and into recursive Form XObject invocations.
///
/// The set of target inks is **not** stored here; instead it is passed
/// as a separate `target_inks: &[&str]` slice alongside the `&mut [Pixmap]`
/// to [`execute_separation_operators`]. This keeps the borrow checker
/// happy: the pixmaps slice is the only `&mut` in play, while everything
/// in `SeparationContext` is `&`.
struct SeparationContext<'a> {
    doc: &'a PdfDocument,
    text_rasterizer: &'a TextRasterizer,
    fonts: &'a HashMap<String, Arc<FontInfo>>,
}

/// Color state tracked alongside the graphics state for separation rendering.
#[derive(Clone, Debug)]
struct SeparationColorState {
    fill_components: Vec<f32>,
    stroke_components: Vec<f32>,
}

impl SeparationColorState {
    fn new() -> Self {
        Self {
            fill_components: Vec::new(),
            stroke_components: Vec::new(),
        }
    }
}

/// Compute the initial colour components for a colour space per
/// ISO 32000-1 §8.6.4.2. `cs`/`CS` resets the current colour to these
/// values when entering the space.
fn initial_components_for_space(
    space_name: &str,
    color_spaces: &HashMap<String, Object>,
    resources: &Object,
    doc: &PdfDocument,
) -> (Vec<f32>, Option<(f32, f32, f32, f32)>) {
    let resolved = resolve_color_space(space_name, color_spaces, resources, doc);
    match resolved {
        ResolvedSpace::Cmyk | ResolvedSpace::IccCmyk => {
            (vec![0.0, 0.0, 0.0, 1.0], Some((0.0, 0.0, 0.0, 1.0)))
        },
        ResolvedSpace::Rgb | ResolvedSpace::IccRgb => (vec![0.0, 0.0, 0.0], None),
        ResolvedSpace::Gray | ResolvedSpace::IccGray => (vec![0.0], None),
        ResolvedSpace::Separation(_) => (vec![1.0], None),
        ResolvedSpace::DeviceN(names) => {
            let n = names.len().max(1);
            (vec![1.0; n], None)
        },
        ResolvedSpace::Unknown => (Vec::new(), None),
    }
}

/// State inherited from a calling context when recursing into a Form
/// XObject (PDF §8.10.1: a Form XObject's initial graphics state is
/// the calling context's graphics state).
struct InheritedState {
    fill_color_space: String,
    stroke_color_space: String,
    fill_color_cmyk: Option<(f32, f32, f32, f32)>,
    stroke_color_cmyk: Option<(f32, f32, f32, f32)>,
    fill_components: Vec<f32>,
    stroke_components: Vec<f32>,
    fill_overprint: bool,
    stroke_overprint: bool,
    overprint_mode: u8,
}

/// Execute operators for separation plate rendering, dispatching paint
/// operations to **all** target inks in parallel.
///
/// `pixmaps` and `target_inks` are parallel slices: `pixmaps[i]` receives
/// paint for ink `target_inks[i]`. The operator stream is walked exactly
/// once; every paint site (fill, stroke, text, Form XObject) iterates the
/// pair list and contributes to each plate whose ink the current colour
/// touches.
#[allow(clippy::too_many_arguments)]
fn execute_separation_operators(
    pixmaps: &mut [Pixmap],
    base_transform: Transform,
    operators: &[Operator],
    ctx: &mut SeparationContext<'_>,
    resources: &Object,
    color_spaces: &HashMap<String, Object>,
    inherited: Option<&InheritedState>,
    target_inks: &[&str],
) -> Result<()> {
    debug_assert_eq!(pixmaps.len(), target_inks.len());
    let mut gs_stack = GraphicsStateStack::new();
    {
        let gs = gs_stack.current_mut();
        if let Some(inh) = inherited {
            gs.fill_color_space = inh.fill_color_space.clone();
            gs.stroke_color_space = inh.stroke_color_space.clone();
            gs.fill_color_cmyk = inh.fill_color_cmyk;
            gs.stroke_color_cmyk = inh.stroke_color_cmyk;
            // §8.10.1: inherit the caller's overprint state too. Without
            // this, an outer `gs` setting OP=true would be silently
            // dropped at the Form XObject boundary and the form's CMYK
            // content would knock out underlying inks against the
            // caller's intent.
            gs.fill_overprint = inh.fill_overprint;
            gs.stroke_overprint = inh.stroke_overprint;
            gs.overprint_mode = inh.overprint_mode;
        } else {
            gs.fill_color_space = "DeviceGray".to_string();
            gs.stroke_color_space = "DeviceGray".to_string();
        }
        gs.fill_color_rgb = (0.0, 0.0, 0.0);
        gs.stroke_color_rgb = (0.0, 0.0, 0.0);
    }

    let initial_cs = if let Some(inh) = inherited {
        SeparationColorState {
            fill_components: inh.fill_components.clone(),
            stroke_components: inh.stroke_components.clone(),
        }
    } else {
        SeparationColorState::new()
    };
    let mut color_state_stack: Vec<SeparationColorState> = vec![initial_cs];
    let mut current_path = PathBuilder::new();
    let mut pending_clip: Option<(tiny_skia::Path, FillRule)> = None;
    let mut clip_stack: Vec<Option<Mask>> = vec![None];
    let mut in_text_object = false;

    // Pre-resolve ExtGState for the gs cache.
    let ext_g_state_resolved: Option<Object> = match resources {
        Object::Dictionary(rd) => rd
            .get("ExtGState")
            .and_then(|o| ctx.doc.resolve_object(o).ok()),
        _ => None,
    };
    let ext_g_states: Option<&HashMap<String, Object>> =
        ext_g_state_resolved.as_ref().and_then(|o| o.as_dict());
    let mut ext_g_state_cache: HashMap<String, ParsedExtGState> = HashMap::new();

    let xobjects_resolved: Option<Object> = match resources {
        Object::Dictionary(rd) => rd
            .get("XObject")
            .and_then(|o| ctx.doc.resolve_object(o).ok()),
        _ => None,
    };

    // Pixmap extent — every plate shares the same dimensions because they
    // all originate from a single allocation in `render_plates_for_inks`.
    // If `pixmaps` is empty (no inks to render), use a zero extent; the
    // operator walk still progresses for graphics-state tracking but
    // paint loops are no-ops because there are no targets.
    let pixmap_width = pixmaps.first().map(|p| p.width()).unwrap_or(0);
    let pixmap_height = pixmaps.first().map(|p| p.height()).unwrap_or(0);

    for op in operators {
        match op {
            Operator::SaveState => {
                gs_stack.save();
                let cs = color_state_stack
                    .last()
                    .cloned()
                    .unwrap_or_else(SeparationColorState::new);
                color_state_stack.push(cs);
                clip_stack.push(clip_stack.last().cloned().unwrap_or(None));
            },
            Operator::RestoreState => {
                gs_stack.restore();
                if color_state_stack.len() > 1 {
                    color_state_stack.pop();
                }
                if clip_stack.len() > 1 {
                    clip_stack.pop();
                }
            },

            Operator::Cm { a, b, c, d, e, f } => {
                let current = gs_stack.current_mut();
                let new_matrix = Matrix {
                    a: *a,
                    b: *b,
                    c: *c,
                    d: *d,
                    e: *e,
                    f: *f,
                };
                current.ctm = new_matrix.multiply(&current.ctm);
            },

            Operator::SetFillRgb { r, g, b } => {
                let gs = gs_stack.current_mut();
                gs.fill_color_rgb = (*r, *g, *b);
                gs.fill_color_space = "DeviceRGB".to_string();
                gs.fill_color_cmyk = None;
                if let Some(cs) = color_state_stack.last_mut() {
                    cs.fill_components = vec![*r, *g, *b];
                }
            },
            Operator::SetStrokeRgb { r, g, b } => {
                let gs = gs_stack.current_mut();
                gs.stroke_color_rgb = (*r, *g, *b);
                gs.stroke_color_space = "DeviceRGB".to_string();
                gs.stroke_color_cmyk = None;
                if let Some(cs) = color_state_stack.last_mut() {
                    cs.stroke_components = vec![*r, *g, *b];
                }
            },
            Operator::SetFillGray { gray } => {
                let g = *gray;
                let gs = gs_stack.current_mut();
                gs.fill_color_rgb = (g, g, g);
                gs.fill_color_space = "DeviceGray".to_string();
                gs.fill_color_cmyk = None;
                if let Some(cs) = color_state_stack.last_mut() {
                    cs.fill_components = vec![g];
                }
            },
            Operator::SetStrokeGray { gray } => {
                let g = *gray;
                let gs = gs_stack.current_mut();
                gs.stroke_color_rgb = (g, g, g);
                gs.stroke_color_space = "DeviceGray".to_string();
                gs.stroke_color_cmyk = None;
                if let Some(cs) = color_state_stack.last_mut() {
                    cs.stroke_components = vec![g];
                }
            },
            Operator::SetFillCmyk { c, m, y, k } => {
                let gs = gs_stack.current_mut();
                gs.fill_color_cmyk = Some((*c, *m, *y, *k));
                gs.fill_color_space = "DeviceCMYK".to_string();
                if let Some(cs) = color_state_stack.last_mut() {
                    cs.fill_components = vec![*c, *m, *y, *k];
                }
            },
            Operator::SetStrokeCmyk { c, m, y, k } => {
                let gs = gs_stack.current_mut();
                gs.stroke_color_cmyk = Some((*c, *m, *y, *k));
                gs.stroke_color_space = "DeviceCMYK".to_string();
                if let Some(cs) = color_state_stack.last_mut() {
                    cs.stroke_components = vec![*c, *m, *y, *k];
                }
            },
            Operator::SetFillColorSpace { name } => {
                let (components, cmyk) =
                    initial_components_for_space(name, color_spaces, resources, ctx.doc);
                let gs = gs_stack.current_mut();
                gs.fill_color_space = name.clone();
                gs.fill_color_cmyk = cmyk;
                if let Some(cs) = color_state_stack.last_mut() {
                    cs.fill_components = components;
                }
            },
            Operator::SetStrokeColorSpace { name } => {
                let (components, cmyk) =
                    initial_components_for_space(name, color_spaces, resources, ctx.doc);
                let gs = gs_stack.current_mut();
                gs.stroke_color_space = name.clone();
                gs.stroke_color_cmyk = cmyk;
                if let Some(cs) = color_state_stack.last_mut() {
                    cs.stroke_components = components;
                }
            },
            Operator::SetFillColor { components } | Operator::SetFillColorN { components, .. } => {
                let gs = gs_stack.current_mut();
                let space = gs.fill_color_space.clone();
                match space.as_str() {
                    "DeviceCMYK" | "CMYK" if components.len() >= 4 => {
                        gs.fill_color_cmyk =
                            Some((components[0], components[1], components[2], components[3]));
                    },
                    _ => {},
                }
                if let Some(cs) = color_state_stack.last_mut() {
                    cs.fill_components = components.clone();
                }
            },
            Operator::SetStrokeColor { components }
            | Operator::SetStrokeColorN { components, .. } => {
                let gs = gs_stack.current_mut();
                let space = gs.stroke_color_space.clone();
                match space.as_str() {
                    "DeviceCMYK" | "CMYK" if components.len() >= 4 => {
                        gs.stroke_color_cmyk =
                            Some((components[0], components[1], components[2], components[3]));
                    },
                    _ => {},
                }
                if let Some(cs) = color_state_stack.last_mut() {
                    cs.stroke_components = components.clone();
                }
            },

            Operator::SetLineWidth { width } => {
                gs_stack.current_mut().line_width = *width;
            },
            Operator::SetLineCap { cap_style } => {
                gs_stack.current_mut().line_cap = *cap_style;
            },
            Operator::SetLineJoin { join_style } => {
                gs_stack.current_mut().line_join = *join_style;
            },
            Operator::SetMiterLimit { limit } => {
                gs_stack.current_mut().miter_limit = *limit;
            },
            Operator::SetDash { array, phase } => {
                gs_stack.current_mut().dash_pattern = (array.clone(), *phase);
            },

            Operator::MoveTo { x, y } => {
                current_path.move_to(*x, *y);
            },
            Operator::LineTo { x, y } => {
                current_path.line_to(*x, *y);
            },
            Operator::CurveTo {
                x1,
                y1,
                x2,
                y2,
                x3,
                y3,
            } => {
                current_path.cubic_to(*x1, *y1, *x2, *y2, *x3, *y3);
            },
            Operator::CurveToV { x2, y2, x3, y3 } => {
                if let Some(last) = current_path.last_point() {
                    current_path.cubic_to(last.x, last.y, *x2, *y2, *x3, *y3);
                }
            },
            Operator::CurveToY { x1, y1, x3, y3 } => {
                current_path.cubic_to(*x1, *y1, *x3, *y3, *x3, *y3);
            },
            Operator::Rectangle {
                x,
                y,
                width,
                height,
            } => {
                let (nx, nw) = if *width < 0.0 {
                    (x + width, -width)
                } else {
                    (*x, *width)
                };
                let (ny, nh) = if *height < 0.0 {
                    (y + height, -height)
                } else {
                    (*y, *height)
                };
                if let Some(rect) = tiny_skia::Rect::from_xywh(nx, ny, nw, nh) {
                    current_path.push_rect(rect);
                }
            },
            Operator::ClosePath => {
                current_path.close();
            },

            Operator::Stroke => {
                apply_separation_clip(
                    &mut pending_clip,
                    &mut clip_stack,
                    pixmap_width,
                    pixmap_height,
                    base_transform,
                    &gs_stack,
                );
                if let Some(path) = current_path.finish() {
                    let gs = gs_stack.current();
                    let empty = SeparationColorState::new();
                    let cs = color_state_stack.last().unwrap_or(&empty);
                    let transform = combine_transforms(base_transform, &gs.ctm);
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    for (i, &ink) in target_inks.iter().enumerate() {
                        if let PaintAction::Paint(tint) = tint_for_ink(
                            false,
                            gs,
                            color_spaces,
                            resources,
                            ctx.doc,
                            ink,
                            &cs.fill_components,
                            &cs.stroke_components,
                        ) {
                            stroke_separation(&mut pixmaps[i], &path, transform, gs, tint, clip);
                        }
                    }
                }
                current_path = PathBuilder::new();
            },
            Operator::Fill => {
                apply_separation_clip(
                    &mut pending_clip,
                    &mut clip_stack,
                    pixmap_width,
                    pixmap_height,
                    base_transform,
                    &gs_stack,
                );
                if let Some(path) = current_path.finish() {
                    let gs = gs_stack.current();
                    let empty = SeparationColorState::new();
                    let cs = color_state_stack.last().unwrap_or(&empty);
                    let transform = combine_transforms(base_transform, &gs.ctm);
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    for (i, &ink) in target_inks.iter().enumerate() {
                        if let PaintAction::Paint(tint) = tint_for_ink(
                            true,
                            gs,
                            color_spaces,
                            resources,
                            ctx.doc,
                            ink,
                            &cs.fill_components,
                            &cs.stroke_components,
                        ) {
                            fill_separation(
                                &mut pixmaps[i],
                                &path,
                                transform,
                                tint,
                                FillRule::Winding,
                                clip,
                            );
                        }
                    }
                }
                current_path = PathBuilder::new();
            },
            Operator::FillEvenOdd => {
                apply_separation_clip(
                    &mut pending_clip,
                    &mut clip_stack,
                    pixmap_width,
                    pixmap_height,
                    base_transform,
                    &gs_stack,
                );
                if let Some(path) = current_path.finish() {
                    let gs = gs_stack.current();
                    let empty = SeparationColorState::new();
                    let cs = color_state_stack.last().unwrap_or(&empty);
                    let transform = combine_transforms(base_transform, &gs.ctm);
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    for (i, &ink) in target_inks.iter().enumerate() {
                        if let PaintAction::Paint(tint) = tint_for_ink(
                            true,
                            gs,
                            color_spaces,
                            resources,
                            ctx.doc,
                            ink,
                            &cs.fill_components,
                            &cs.stroke_components,
                        ) {
                            fill_separation(
                                &mut pixmaps[i],
                                &path,
                                transform,
                                tint,
                                FillRule::EvenOdd,
                                clip,
                            );
                        }
                    }
                }
                current_path = PathBuilder::new();
            },
            Operator::FillStroke | Operator::CloseFillStroke => {
                apply_separation_clip(
                    &mut pending_clip,
                    &mut clip_stack,
                    pixmap_width,
                    pixmap_height,
                    base_transform,
                    &gs_stack,
                );
                if let Some(path) = current_path.finish() {
                    let gs = gs_stack.current();
                    let empty = SeparationColorState::new();
                    let cs = color_state_stack.last().unwrap_or(&empty);
                    let transform = combine_transforms(base_transform, &gs.ctm);
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    for (i, &ink) in target_inks.iter().enumerate() {
                        if let PaintAction::Paint(tint) = tint_for_ink(
                            true,
                            gs,
                            color_spaces,
                            resources,
                            ctx.doc,
                            ink,
                            &cs.fill_components,
                            &cs.stroke_components,
                        ) {
                            fill_separation(
                                &mut pixmaps[i],
                                &path,
                                transform,
                                tint,
                                FillRule::Winding,
                                clip,
                            );
                        }
                        if let PaintAction::Paint(tint) = tint_for_ink(
                            false,
                            gs,
                            color_spaces,
                            resources,
                            ctx.doc,
                            ink,
                            &cs.fill_components,
                            &cs.stroke_components,
                        ) {
                            stroke_separation(&mut pixmaps[i], &path, transform, gs, tint, clip);
                        }
                    }
                }
                current_path = PathBuilder::new();
            },
            Operator::FillStrokeEvenOdd | Operator::CloseFillStrokeEvenOdd => {
                apply_separation_clip(
                    &mut pending_clip,
                    &mut clip_stack,
                    pixmap_width,
                    pixmap_height,
                    base_transform,
                    &gs_stack,
                );
                if let Some(path) = current_path.finish() {
                    let gs = gs_stack.current();
                    let empty = SeparationColorState::new();
                    let cs = color_state_stack.last().unwrap_or(&empty);
                    let transform = combine_transforms(base_transform, &gs.ctm);
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    for (i, &ink) in target_inks.iter().enumerate() {
                        if let PaintAction::Paint(tint) = tint_for_ink(
                            true,
                            gs,
                            color_spaces,
                            resources,
                            ctx.doc,
                            ink,
                            &cs.fill_components,
                            &cs.stroke_components,
                        ) {
                            fill_separation(
                                &mut pixmaps[i],
                                &path,
                                transform,
                                tint,
                                FillRule::EvenOdd,
                                clip,
                            );
                        }
                        if let PaintAction::Paint(tint) = tint_for_ink(
                            false,
                            gs,
                            color_spaces,
                            resources,
                            ctx.doc,
                            ink,
                            &cs.fill_components,
                            &cs.stroke_components,
                        ) {
                            stroke_separation(&mut pixmaps[i], &path, transform, gs, tint, clip);
                        }
                    }
                }
                current_path = PathBuilder::new();
            },
            Operator::EndPath => {
                apply_separation_clip(
                    &mut pending_clip,
                    &mut clip_stack,
                    pixmap_width,
                    pixmap_height,
                    base_transform,
                    &gs_stack,
                );
                current_path = PathBuilder::new();
            },

            Operator::ClipNonZero => {
                if let Some(path) = current_path.clone().finish() {
                    pending_clip = Some((path, FillRule::Winding));
                }
            },
            Operator::ClipEvenOdd => {
                if let Some(path) = current_path.clone().finish() {
                    pending_clip = Some((path, FillRule::EvenOdd));
                }
            },

            // Text object
            Operator::BeginText => {
                in_text_object = true;
                let gs = gs_stack.current_mut();
                gs.text_matrix = Matrix::identity();
                gs.text_line_matrix = Matrix::identity();
            },
            Operator::EndText => {
                in_text_object = false;
            },

            // Text state
            Operator::Tc { char_space } => {
                gs_stack.current_mut().char_space = *char_space;
            },
            Operator::Tw { word_space } => {
                gs_stack.current_mut().word_space = *word_space;
            },
            Operator::Tz { scale } => {
                gs_stack.current_mut().horizontal_scaling = *scale;
            },
            Operator::TL { leading } => {
                gs_stack.current_mut().leading = *leading;
            },
            Operator::Ts { rise } => {
                gs_stack.current_mut().text_rise = *rise;
            },
            Operator::Tr { render } => {
                gs_stack.current_mut().render_mode = *render;
            },
            Operator::Tf { font, size } => {
                let gs = gs_stack.current_mut();
                gs.font_name = Some(font.clone());
                gs.font_size = *size;
            },

            // Text positioning
            Operator::Td { tx, ty } => {
                if in_text_object {
                    let gs = gs_stack.current_mut();
                    let translation = Matrix::translation(*tx, *ty);
                    gs.text_line_matrix = translation.multiply(&gs.text_line_matrix);
                    gs.text_matrix = gs.text_line_matrix;
                }
            },
            Operator::TD { tx, ty } => {
                if in_text_object {
                    let gs = gs_stack.current_mut();
                    gs.leading = -(*ty);
                    let translation = Matrix::translation(*tx, *ty);
                    gs.text_line_matrix = translation.multiply(&gs.text_line_matrix);
                    gs.text_matrix = gs.text_line_matrix;
                }
            },
            Operator::Tm { a, b, c, d, e, f } => {
                if in_text_object {
                    let gs = gs_stack.current_mut();
                    gs.text_matrix = Matrix {
                        a: *a,
                        b: *b,
                        c: *c,
                        d: *d,
                        e: *e,
                        f: *f,
                    };
                    gs.text_line_matrix = gs.text_matrix;
                }
            },
            Operator::TStar => {
                if in_text_object {
                    let gs = gs_stack.current_mut();
                    let leading = gs.leading;
                    let translation = Matrix::translation(0.0, -leading);
                    gs.text_line_matrix = translation.multiply(&gs.text_line_matrix);
                    gs.text_matrix = gs.text_line_matrix;
                }
            },

            // Text showing
            Operator::Tj { text } => {
                if in_text_object {
                    let advance = render_text_to_plate(
                        pixmaps,
                        text,
                        base_transform,
                        &mut gs_stack,
                        &color_state_stack,
                        color_spaces,
                        resources,
                        ctx,
                        clip_stack.last().and_then(|c| c.as_ref()),
                        target_inks,
                    )?;
                    let gs_mut = gs_stack.current_mut();
                    let advance_matrix = Matrix::translation(advance, 0.0);
                    gs_mut.text_matrix = advance_matrix.multiply(&gs_mut.text_matrix);
                }
            },
            Operator::TJ { array } => {
                if in_text_object {
                    let advance = render_tj_to_plate(
                        pixmaps,
                        array,
                        base_transform,
                        &mut gs_stack,
                        &color_state_stack,
                        color_spaces,
                        resources,
                        ctx,
                        clip_stack.last().and_then(|c| c.as_ref()),
                        target_inks,
                    )?;
                    let gs_mut = gs_stack.current_mut();
                    let advance_matrix = Matrix::translation(advance, 0.0);
                    gs_mut.text_matrix = advance_matrix.multiply(&gs_mut.text_matrix);
                }
            },
            Operator::Quote { text } => {
                if in_text_object {
                    let gs_mut = gs_stack.current_mut();
                    let leading = gs_mut.leading;
                    let translation = Matrix::translation(0.0, -leading);
                    gs_mut.text_line_matrix = translation.multiply(&gs_mut.text_line_matrix);
                    gs_mut.text_matrix = gs_mut.text_line_matrix;

                    let advance = render_text_to_plate(
                        pixmaps,
                        text,
                        base_transform,
                        &mut gs_stack,
                        &color_state_stack,
                        color_spaces,
                        resources,
                        ctx,
                        clip_stack.last().and_then(|c| c.as_ref()),
                        target_inks,
                    )?;
                    let gs_mut = gs_stack.current_mut();
                    let advance_matrix = Matrix::translation(advance, 0.0);
                    gs_mut.text_matrix = advance_matrix.multiply(&gs_mut.text_matrix);
                }
            },
            Operator::DoubleQuote {
                word_space,
                char_space,
                text,
            } => {
                if in_text_object {
                    let gs_mut = gs_stack.current_mut();
                    gs_mut.word_space = *word_space;
                    gs_mut.char_space = *char_space;
                    let leading = gs_mut.leading;
                    let translation = Matrix::translation(0.0, -leading);
                    gs_mut.text_line_matrix = translation.multiply(&gs_mut.text_line_matrix);
                    gs_mut.text_matrix = gs_mut.text_line_matrix;

                    let advance = render_text_to_plate(
                        pixmaps,
                        text,
                        base_transform,
                        &mut gs_stack,
                        &color_state_stack,
                        color_spaces,
                        resources,
                        ctx,
                        clip_stack.last().and_then(|c| c.as_ref()),
                        target_inks,
                    )?;
                    let gs_mut = gs_stack.current_mut();
                    let advance_matrix = Matrix::translation(advance, 0.0);
                    gs_mut.text_matrix = advance_matrix.multiply(&gs_mut.text_matrix);
                }
            },

            // ExtGState
            Operator::SetExtGState { dict_name } => {
                let entry = ext_g_state_cache
                    .entry(dict_name.clone())
                    .or_insert_with(|| {
                        if let Some(states) = ext_g_states {
                            if let Some(state_obj) = states.get(dict_name) {
                                return parse_ext_g_state_inner(state_obj, ctx.doc)
                                    .unwrap_or_default();
                            }
                        }
                        ParsedExtGState::default()
                    });
                entry.apply(gs_stack.current_mut());
            },

            // XObject (Form XObjects may contain ink-bearing content).
            // Image XObjects are dropped — see module-level Limitations.
            Operator::Do { name } => {
                if let Some(xobjects) = xobjects_resolved.as_ref().and_then(|o| o.as_dict()) {
                    if let Some(xobj_ref_obj) = xobjects.get(name) {
                        if let Ok(xobj) = ctx.doc.resolve_object(xobj_ref_obj) {
                            if let Object::Stream { ref dict, .. } = xobj {
                                if let Some(subtype) = dict.get("Subtype").and_then(|o| o.as_name())
                                {
                                    if subtype == "Form" {
                                        let xobj_ref = xobj_ref_obj.as_reference();
                                        let stream_data = if let Some(r) = xobj_ref {
                                            ctx.doc.decode_stream_with_encryption(&xobj, r)?
                                        } else {
                                            xobj.decode_stream_data()?
                                        };

                                        let form_resources =
                                            if let Some(res) = dict.get("Resources") {
                                                ctx.doc.resolve_object(res)?
                                            } else {
                                                resources.clone()
                                            };

                                        let form_cs = load_color_spaces(ctx.doc, &form_resources)?;
                                        let mut merged_cs = color_spaces.clone();
                                        merged_cs.extend(form_cs);

                                        let form_matrix = parse_form_matrix(dict);
                                        let gs = gs_stack.current();
                                        let combined = combine_transforms(base_transform, &gs.ctm)
                                            .pre_concat(form_matrix);

                                        // Inherit the calling context's colour state into the
                                        // form's initial graphics state (PDF §8.10.1, O5).
                                        let empty = SeparationColorState::new();
                                        let cs = color_state_stack.last().unwrap_or(&empty);
                                        let inherited = InheritedState {
                                            fill_color_space: gs.fill_color_space.clone(),
                                            stroke_color_space: gs.stroke_color_space.clone(),
                                            fill_color_cmyk: gs.fill_color_cmyk,
                                            stroke_color_cmyk: gs.stroke_color_cmyk,
                                            fill_components: cs.fill_components.clone(),
                                            stroke_components: cs.stroke_components.clone(),
                                            fill_overprint: gs.fill_overprint,
                                            stroke_overprint: gs.stroke_overprint,
                                            overprint_mode: gs.overprint_mode,
                                        };

                                        let form_ops = parse_content_stream(&stream_data)?;
                                        execute_separation_operators(
                                            pixmaps,
                                            combined,
                                            &form_ops,
                                            ctx,
                                            &form_resources,
                                            &merged_cs,
                                            Some(&inherited),
                                            target_inks,
                                        )?;
                                    }
                                }
                            }
                        }
                    }
                }
            },

            _ => {},
        }
    }
    Ok(())
}

/// Render text into every target separation pixmap, routing each glyph
/// through the per-ink tint. The strategy is to clone the GraphicsState,
/// replace its fill colour with a grayscale paint equal to the tint, and
/// reuse the standard [`TextRasterizer`]. This preserves glyph shape,
/// kerning, and anti-aliasing — the same fidelity as the page renderer.
///
/// The returned advance is shared across all plates (the rasteriser is
/// deterministic for a given font/text/state, so each plate's advance
/// agrees) — we use the last computed value, matching the single-plate
/// behaviour. If no plate is touched (every plate's [`PaintAction`] is
/// `Skip`, or render mode 3) the advance is computed from the font
/// metrics so the text matrix still progresses correctly.
#[allow(clippy::too_many_arguments)]
fn render_text_to_plate(
    pixmaps: &mut [Pixmap],
    text: &[u8],
    base_transform: Transform,
    gs_stack: &mut GraphicsStateStack,
    color_state_stack: &[SeparationColorState],
    color_spaces: &HashMap<String, Object>,
    resources: &Object,
    ctx: &mut SeparationContext<'_>,
    clip: Option<&Mask>,
    target_inks: &[&str],
) -> Result<f32> {
    let gs = gs_stack.current();
    let empty = SeparationColorState::new();
    let cs = color_state_stack.last().unwrap_or(&empty);

    // Render mode 3 = invisible text. Still advance the text matrix but skip painting.
    if gs.render_mode == 3 {
        return measure_text_advance(text, gs, ctx.fonts);
    }

    let transform = combine_transforms(base_transform, &gs.ctm);
    let mut painted_advance: Option<f32> = None;

    for (i, &ink) in target_inks.iter().enumerate() {
        let tint = match tint_for_ink(
            true,
            gs,
            color_spaces,
            resources,
            ctx.doc,
            ink,
            &cs.fill_components,
            &cs.stroke_components,
        ) {
            PaintAction::Paint(t) => t,
            PaintAction::Skip => continue,
        };

        // Build a faked-grayscale GraphicsState so the rasteriser paints in
        // (tint, tint, tint) which becomes the plate value in the R channel.
        let mut faux = gs.clone();
        faux.fill_color_rgb = (tint, tint, tint);
        faux.fill_alpha = 1.0;
        faux.blend_mode = "Normal".to_string();

        let advance = ctx.text_rasterizer.render_text(
            &mut pixmaps[i],
            text,
            transform,
            &faux,
            resources,
            ctx.doc,
            clip,
            ctx.fonts,
        )?;
        painted_advance = Some(advance);
    }

    match painted_advance {
        Some(a) => Ok(a),
        // No plate was touched by this text — still advance the matrix so
        // subsequent glyphs land at the correct position.
        None => measure_text_advance(text, gs, ctx.fonts),
    }
}

/// Render a TJ array (sequence of strings + offsets) into all target
/// plates. Walks the array applying offsets between strings, painting
/// each string component via [`render_text_to_plate`].
#[allow(clippy::too_many_arguments)]
fn render_tj_to_plate(
    pixmaps: &mut [Pixmap],
    array: &[TextElement],
    base_transform: Transform,
    gs_stack: &mut GraphicsStateStack,
    color_state_stack: &[SeparationColorState],
    color_spaces: &HashMap<String, Object>,
    resources: &Object,
    ctx: &mut SeparationContext<'_>,
    clip: Option<&Mask>,
    target_inks: &[&str],
) -> Result<f32> {
    let mut total_advance = 0.0;
    for element in array {
        match element {
            TextElement::String(text) => {
                let advance = render_text_to_plate(
                    pixmaps,
                    text,
                    base_transform,
                    gs_stack,
                    color_state_stack,
                    color_spaces,
                    resources,
                    ctx,
                    clip,
                    target_inks,
                )?;
                let gs_mut = gs_stack.current_mut();
                let advance_matrix = Matrix::translation(advance, 0.0);
                gs_mut.text_matrix = advance_matrix.multiply(&gs_mut.text_matrix);
                total_advance += advance;
            },
            TextElement::Offset(offset) => {
                let gs = gs_stack.current();
                let shift = (-*offset / 1000.0) * gs.font_size;
                let advance_matrix = Matrix::translation(shift, 0.0);
                let gs_mut = gs_stack.current_mut();
                gs_mut.text_matrix = advance_matrix.multiply(&gs_mut.text_matrix);
                total_advance += shift;
            },
        }
    }
    Ok(total_advance)
}

/// Compute the horizontal advance a [`TextRasterizer`] call would
/// produce, without painting. Used for invisible/skipped text so the
/// text matrix stays consistent with the painted ink plates.
///
/// Best-effort: when an embedded width table is unavailable we fall
/// back to `font_size * len * 0.5` — close enough to keep glyph
/// positions inside the rest of the line.
fn measure_text_advance(
    text: &[u8],
    gs: &GraphicsState,
    fonts: &HashMap<String, Arc<FontInfo>>,
) -> Result<f32> {
    let font_info = gs
        .font_name
        .as_ref()
        .and_then(|n| fonts.get(n))
        .map(Arc::clone);

    // Sum widths from the font's width table (in glyph units / 1000)
    // multiplied by font_size, plus per-char Tc spacing.
    let mut units: f32 = 0.0;
    let mut count: usize = 0;
    if let Some(info) = font_info.as_ref() {
        if info.subtype != "Type0" {
            for &b in text {
                units += info.get_glyph_width(b as u16);
                count += 1;
            }
        } else {
            // Type0: iterate 2-byte codes (approx).
            let mut i = 0;
            while i + 1 < text.len() {
                let code = ((text[i] as u16) << 8) | text[i + 1] as u16;
                units += info.get_glyph_width(code);
                count += 1;
                i += 2;
            }
        }
    } else {
        for _ in text {
            units += 500.0;
            count += 1;
        }
    }
    let advance = units * gs.font_size / 1000.0 + (count as f32) * gs.char_space;
    Ok(advance)
}

/// Fill a path into the separation pixmap with the given tint value.
fn fill_separation(
    pixmap: &mut Pixmap,
    path: &tiny_skia::Path,
    transform: Transform,
    tint: f32,
    fill_rule: FillRule,
    clip: Option<&Mask>,
) {
    let gray = (tint.clamp(0.0, 1.0) * 255.0).round() as u8;
    let color = tiny_skia::Color::from_rgba8(gray, gray, gray, 255);
    let mut paint = tiny_skia::Paint::default();
    paint.set_color(color);
    paint.anti_alias = true;
    // SourceOver with opaque (alpha=255) source = replacement; this matches
    // PDF's opaque painting model where each new fill overwrites the pixels
    // under it within the path. Overlapping fills are *not* accumulated —
    // PDF separation semantics dictate last-writer-wins per ink at the
    // overlapping pixels, which SourceOver gives us for free.
    paint.blend_mode = tiny_skia::BlendMode::SourceOver;

    pixmap.fill_path(path, &paint, fill_rule, transform, clip);
}

/// Stroke a path into the separation pixmap with the given tint value.
fn stroke_separation(
    pixmap: &mut Pixmap,
    path: &tiny_skia::Path,
    transform: Transform,
    gs: &GraphicsState,
    tint: f32,
    clip: Option<&Mask>,
) {
    let gray = (tint.clamp(0.0, 1.0) * 255.0).round() as u8;
    let color = tiny_skia::Color::from_rgba8(gray, gray, gray, 255);
    let mut paint = tiny_skia::Paint::default();
    paint.set_color(color);
    paint.anti_alias = true;

    let mut stroke = tiny_skia::Stroke::default();
    stroke.width = gs.line_width;
    stroke.line_cap = match gs.line_cap {
        1 => tiny_skia::LineCap::Round,
        2 => tiny_skia::LineCap::Square,
        _ => tiny_skia::LineCap::Butt,
    };
    stroke.line_join = match gs.line_join {
        1 => tiny_skia::LineJoin::Round,
        2 => tiny_skia::LineJoin::Bevel,
        _ => tiny_skia::LineJoin::Miter,
    };
    stroke.miter_limit = gs.miter_limit;

    if !gs.dash_pattern.0.is_empty() {
        stroke.dash = tiny_skia::StrokeDash::new(gs.dash_pattern.0.clone(), gs.dash_pattern.1);
    }

    pixmap.stroke_path(path, &paint, &stroke, transform, clip);
}

/// Apply a pending clip path to the clip stack.
///
/// The clip mask is identical across all plates — it depends only on the
/// path, fill rule, current transform, and pixmap dimensions (which are
/// shared). So we build it once and store it on the shared clip stack.
fn apply_separation_clip(
    pending: &mut Option<(tiny_skia::Path, FillRule)>,
    clip_stack: &mut Vec<Option<Mask>>,
    pixmap_width: u32,
    pixmap_height: u32,
    base_transform: Transform,
    gs_stack: &GraphicsStateStack,
) {
    if let Some((path, fill_rule)) = pending.take() {
        // No pixmaps means no plates to clip — bail out early. Width/height
        // would be zero and Mask::new would refuse them anyway.
        if pixmap_width == 0 || pixmap_height == 0 {
            return;
        }
        let gs = gs_stack.current();
        let transform = combine_transforms(base_transform, &gs.ctm);

        if let Some(path_transformed) = path.transform(transform) {
            let mut new_mask = Mask::new(pixmap_width, pixmap_height).unwrap();
            new_mask.fill_path(&path_transformed, fill_rule, true, Transform::identity());

            if let Some(Some(current_mask)) = clip_stack.last() {
                let mut combined = current_mask.clone();
                let combined_data = combined.data_mut();
                let new_data = new_mask.data();
                for i in 0..combined_data.len() {
                    combined_data[i] = ((combined_data[i] as u32 * new_data[i] as u32) / 255) as u8;
                }
                *clip_stack.last_mut().unwrap() = Some(combined);
            } else {
                *clip_stack.last_mut().unwrap() = Some(new_mask);
            }
        }
    }
}

/// Parse a form XObject matrix from its dictionary.
fn parse_form_matrix(dict: &HashMap<String, Object>) -> Transform {
    if let Some(Object::Array(arr)) = dict.get("Matrix") {
        let get_f32 = |i: usize| -> f32 {
            match arr.get(i) {
                Some(Object::Real(v)) => *v as f32,
                Some(Object::Integer(v)) => *v as f32,
                _ => {
                    if i == 0 || i == 3 {
                        1.0
                    } else {
                        0.0
                    }
                },
            }
        };
        Transform::from_row(get_f32(0), get_f32(1), get_f32(2), get_f32(3), get_f32(4), get_f32(5))
    } else {
        Transform::identity()
    }
}

/// Combine two transformations (base + CTM).
fn combine_transforms(base: Transform, ctm: &Matrix) -> Transform {
    base.pre_concat(Transform::from_row(ctm.a, ctm.b, ctm.c, ctm.d, ctm.e, ctm.f))
}
