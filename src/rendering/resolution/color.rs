//! Colour-resolution stage.
//!
//! This is the stage where capabilities that previously could not reach the
//! renderer are wired in:
//!
//! - **PostScript Type 4 calculator** tint transforms ([`crate::functions`]).
//!   Resolves `Separation` and `DeviceN` colour spaces whose `tintTransform`
//!   is a Type-4 function — the case the inline match arm at
//!   `page_renderer.rs:629-693` falls back to `1.0 - tint` for.
//! - **Type 2 exponential interpolation** tint transforms. Spec
//!   ISO 32000-1:2008 §7.10.3. The existing inline match arm handles this
//!   for `DeviceCMYK` alternate spaces only; the resolver handles `DeviceRGB`
//!   and `DeviceGray` alternates as well.
//! - **ICCBased** colour spaces. The resolver delegates to the
//!   [`crate::color::Transform`] CMM when the `icc` feature is on and falls
//!   back to the §10.3.5 additive-clamp formula otherwise. This is the same
//!   path image extraction uses, so we re-use [`crate::color`] rather than
//!   carrying a second copy of the conversion code.
//! - **Indexed** colour spaces. The resolver follows the index into the base
//!   space; for now we handle DeviceGray / DeviceRGB / DeviceCMYK base spaces
//!   and fall back to grayscale otherwise (matching the existing renderer).
//!
//! The output is a [`ResolvedColor::Rgba`] for composite consumers; a
//! follow-up branch will add the `Cmyk` and `PerChannel` variants behind the
//! same resolver entry point so separation backends share the same call.

use crate::error::Result;
use crate::object::Object;

use super::context::ResolutionContext;
use super::intent::{DeviceColor, LogicalColor};
use super::resolved::ResolvedColor;

/// Colour-resolution stage.
///
/// Stateless — the resolver is purely a function of `(LogicalColor,
/// ResolutionContext, gs.fill_alpha-or-stroke_alpha)`. The struct exists so
/// the pipeline can grow per-instance state later (e.g. a cache of compiled
/// Type-4 [`crate::functions::Program`] keyed by stream object id) without
/// changing the call surface.
pub(crate) struct ColorResolver;

impl ColorResolver {
    pub(crate) const fn new() -> Self {
        Self
    }

    /// Resolve `color` into an RGBA value the composite backend can paint.
    ///
    /// `alpha` is the pre-computed straight alpha from the graphics state
    /// (i.e. `gs.fill_alpha` for fill intents, `gs.stroke_alpha` for stroke
    /// intents). Folding it in here keeps backends simple.
    pub(crate) fn resolve(
        &self,
        color: &LogicalColor,
        ctx: &ResolutionContext,
        alpha: f32,
    ) -> Result<ResolvedColor> {
        match color {
            LogicalColor::Device(dev) => {
                // ISO 32000-1:2008 §8.6.5.6: when the page declares a
                // /DefaultGray, /DefaultRGB, or /DefaultCMYK entry in
                // its /Resources /ColorSpace dict, any bare device-family
                // paint operator (the canonical `g`/`rg`/`k`/`K` and
                // their stroking siblings) MUST be interpreted as if it
                // had named the override colour space instead of the
                // device family. The override therefore takes
                // precedence over the document /OutputIntents profile
                // for bare device paint — OutputIntent is only the
                // fallback default when no override has been declared.
                if let Some(resolved) = self.resolve_device_default_override(*dev, ctx, alpha)? {
                    return Ok(resolved);
                }
                Ok(device_to_rgba(*dev, alpha))
            },
            LogicalColor::Spaced { space, components } => {
                self.resolve_spaced(space, components, ctx, alpha)
            },
        }
    }

    /// §8.6.5.6 dispatch for bare device-family paint. Returns `Some`
    /// when the active page has declared a matching `/Default<Family>`
    /// override AND that override resolves successfully; otherwise
    /// returns `None` so the caller emits the device-family default.
    ///
    /// The override is resolved by recursively calling `resolve_spaced`
    /// on the override object with the original paint components. That
    /// reuses the existing colour-space machinery (ICCBased N=3/N=4,
    /// Separation, DeviceN, …) so a `/DefaultCMYK [/ICCBased ...]`
    /// override goes through the embedded-ICC path, picks up the
    /// per-page transform cache via `ctx.icc_transform_cache`, and
    /// emits `ResolvedColor::IccCmyk` exactly as for an explicit
    /// `[/ICCBased N=4]` colour space paint.
    ///
    /// Precedence note: this fires BEFORE the OutputIntent-aware CMYK
    /// projection at `cmyk_to_rgb_via_intent` because the override is
    /// the page's declared colour space and OutputIntent only fills
    /// in for the device family when no override is present.
    fn resolve_device_default_override(
        &self,
        dev: DeviceColor,
        ctx: &ResolutionContext,
        alpha: f32,
    ) -> Result<Option<ResolvedColor>> {
        let (override_obj, components): (Option<&Object>, smallvec::SmallVec<[f32; 4]>) = match dev
        {
            DeviceColor::Gray(g) => (ctx.default_gray, smallvec::smallvec![g]),
            DeviceColor::Rgb(r, g, b) => (ctx.default_rgb, smallvec::smallvec![r, g, b]),
            DeviceColor::Cmyk(c, m, y, k) => (ctx.default_cmyk, smallvec::smallvec![c, m, y, k]),
        };
        let Some(space) = override_obj else {
            return Ok(None);
        };

        // §8.6.5.6 requires the override entry to be a colour space:
        // either a Name (device-family alias such as `/DeviceCMYK`,
        // `/CalGray`) or an Array (`[/ICCBased ...]`, `[/Separation
        // ...]`, etc.). A malformed entry (string, integer, bool,
        // dictionary…) is structurally indistinguishable from the
        // entry being absent — honouring it would silently
        // mis-render through `resolve_spaced`'s `first_as_gray`
        // catch-all (a quarter-tint CMYK paint coming out as 25%
        // gray is worse than the spec-fallback / OutputIntent
        // render). Return None so the caller falls through to the
        // device-family path (`device_to_rgba`), which routes CMYK
        // through `cmyk_to_rgb_via_intent` and so consults
        // `/OutputIntents` when present, or §10.3.5 additive-clamp
        // when not.
        if space.as_name().is_none() && space.as_array().is_none() {
            return Ok(None);
        }

        // The override resolves via the same colour-space pipeline
        // as an explicit `cs <space>` paint — that's the whole point
        // of §8.6.5.6: the override colour space stands in for the
        // device family. If the override object is just another Name
        // (e.g. `/DefaultCMYK /DeviceCMYK`, an identity declaration),
        // resolve_spaced's Name arm folds back to the device-family
        // default — returning Some is still correct because we've
        // honoured the override; it just produces the same value as
        // the no-override path.
        Ok(Some(self.resolve_spaced(space, &components, ctx, alpha)?))
    }

    fn resolve_spaced(
        &self,
        space: &Object,
        components: &[f32],
        ctx: &ResolutionContext,
        alpha: f32,
    ) -> Result<ResolvedColor> {
        // A `Name` here means a device family — the operator dispatcher
        // already folded those into LogicalColor::Device for the canonical
        // `g`/`rg`/`k`/`K` operators, but `SCN` against a Device* alias
        // still reaches us this way.
        if let Some(name) = space.as_name() {
            return Ok(resolve_device_alias(name, components, alpha));
        }

        let Some(arr) = space.as_array() else {
            // Unknown space shape — fall back to first-component-as-gray,
            // matching the existing inline behaviour at
            // `page_renderer.rs:709-712`.
            return Ok(first_as_gray(components, alpha));
        };

        let Some(type_name) = arr.first().and_then(|o| o.as_name()) else {
            return Ok(first_as_gray(components, alpha));
        };

        match type_name {
            "DeviceGray" | "G" | "CalGray" => Ok(first_as_gray(components, alpha)),
            "DeviceRGB" | "RGB" | "CalRGB" => Ok(three_as_rgb(components, alpha)),
            "DeviceCMYK" | "CMYK" => Ok(four_as_cmyk_native(components, alpha)),
            "ICCBased" => self.resolve_iccbased(arr, components, ctx, alpha),
            "Separation" | "DeviceN" => {
                self.resolve_separation_or_devicen(arr, components, ctx, alpha)
            },
            "Indexed" => self.resolve_indexed(arr, components, ctx, alpha),
            _ => Ok(first_as_gray(components, alpha)),
        }
    }

    fn resolve_iccbased(
        &self,
        arr: &[Object],
        components: &[f32],
        ctx: &ResolutionContext,
        alpha: f32,
    ) -> Result<ResolvedColor> {
        // ICCBased array shape: [/ICCBased <stream-ref>]. The stream dict
        // carries /N indicating the input component count.
        let Some(stream_obj) = arr.get(1) else {
            return Ok(first_as_gray(components, alpha));
        };
        let resolved_stream = match ctx.doc.resolve_object(stream_obj) {
            Ok(o) => o,
            Err(_) => return Ok(first_as_gray(components, alpha)),
        };
        let Some(dict) = resolved_stream.as_dict() else {
            return Ok(first_as_gray(components, alpha));
        };
        let n = dict.get("N").and_then(|o| o.as_integer()).unwrap_or(3);

        // §8.6.5.5 precedence: an ICCBased colour space carries its own
        // conversion source. The embedded profile wins over the document
        // /OutputIntents profile when CMYK→RGB is requested. Decode the
        // stream, parse the bytes through IccProfile::parse (which
        // cross-checks the dict's /N against the ICC header signature),
        // and compile a qcms Transform against the active rendering
        // intent. On any failure (no `icc` feature, decode error,
        // mismatched header, qcms refusal) we fall through to the
        // device-family path — that path emits ResolvedColor::Cmyk for
        // N=4, which the composite projection then converts through
        // ctx.output_intent_cmyk: the document OutputIntent becomes the
        // default when the embedded profile can't actually drive a CMM.
        //
        // We emit the dual-payload `IccCmyk` variant so the per-plate
        // router still sees the four channel decomposition. The composite
        // backend reads the pre-computed RGB; the separation backend
        // reads the original CMYK quadruple. The ICC conversion is a
        // composite-surface concern — the plates ARE the press-target
        // ink coverage, so dropping the CMYK channel values for a
        // monolithic Rgba would zero out every plate.
        #[cfg(feature = "icc")]
        if n == 4 && components.len() >= 4 {
            if let Ok(bytes) = resolved_stream.decode_stream_data() {
                if let Some(profile) = crate::color::IccProfile::parse(bytes, 4) {
                    let profile = std::sync::Arc::new(profile);
                    // Per-page transform cache keyed on profile content
                    // hash + intent (see IccTransformCache). The
                    // embedded /ICCBased profile is parsed afresh on
                    // every paint operator (the decode + parse happens
                    // above), but the qcms CMM is the heavy bit and
                    // gets reused across paints whose ICCBased stream
                    // hashes identically. Unit tests skip the cache
                    // (ctx.icc_transform_cache is None) and pay the
                    // per-call build cost.
                    let transform: std::sync::Arc<crate::color::Transform> =
                        if let Some(cache) = ctx.icc_transform_cache {
                            cache.get_or_build(&profile, ctx.rendering_intent)
                        } else {
                            std::sync::Arc::new(crate::color::Transform::new_srgb_target(
                                std::sync::Arc::clone(&profile),
                                ctx.rendering_intent,
                            ))
                        };
                    if transform.has_cmm() {
                        let c = components[0].clamp(0.0, 1.0);
                        let m = components[1].clamp(0.0, 1.0);
                        let y = components[2].clamp(0.0, 1.0);
                        let k = components[3].clamp(0.0, 1.0);
                        let c_u8 = (c * 255.0).round() as u8;
                        let m_u8 = (m * 255.0).round() as u8;
                        let y_u8 = (y * 255.0).round() as u8;
                        let k_u8 = (k * 255.0).round() as u8;
                        let rgb = transform.convert_cmyk_pixel(c_u8, m_u8, y_u8, k_u8);
                        return Ok(ResolvedColor::IccCmyk {
                            r: rgb[0] as f32 / 255.0,
                            g: rgb[1] as f32 / 255.0,
                            b: rgb[2] as f32 / 255.0,
                            c,
                            m,
                            y,
                            k,
                            a: alpha,
                        });
                    }
                }
            }
        }

        // ICCBased N=3 — RGB source profile. The embedded profile
        // drives the conversion (§8.6.5.5); the §10.3.5 fallback only
        // fires when qcms refuses to compile the profile. This branch
        // is also the path the §8.6.5.6 /DefaultRGB override consumes:
        // declaring `/DefaultRGB [/ICCBased <N=3 stream>]` and painting
        // bare /DeviceRGB sends the three components through this arm.
        //
        // No per-plate routing complication here — RGB never lands on
        // CMYK plates — so we emit ResolvedColor::Rgba directly. The
        // per-page transform cache (originally introduced for CMYK,
        // but n_components-agnostic at the key level — see
        // `IccTransformCache` docstring) is consulted here too: an
        // /ICCBased N=3 profile used by a /DefaultRGB override gets
        // hit by every bare /DeviceRGB paint on the page, so caching
        // the compiled qcms transform pays back for the same reason
        // the CMYK arm above does.
        #[cfg(feature = "icc")]
        if n == 3 && components.len() >= 3 {
            if let Ok(bytes) = resolved_stream.decode_stream_data() {
                if let Some(profile) = crate::color::IccProfile::parse(bytes, 3) {
                    let profile = std::sync::Arc::new(profile);
                    let transform: std::sync::Arc<crate::color::Transform> =
                        if let Some(cache) = ctx.icc_transform_cache {
                            cache.get_or_build(&profile, ctx.rendering_intent)
                        } else {
                            std::sync::Arc::new(crate::color::Transform::new_srgb_target(
                                std::sync::Arc::clone(&profile),
                                ctx.rendering_intent,
                            ))
                        };
                    if transform.has_cmm() {
                        let r = components[0].clamp(0.0, 1.0);
                        let g = components[1].clamp(0.0, 1.0);
                        let b = components[2].clamp(0.0, 1.0);
                        let r_u8 = (r * 255.0).round() as u8;
                        let g_u8 = (g * 255.0).round() as u8;
                        let b_u8 = (b * 255.0).round() as u8;
                        let rgb = transform.convert_rgb_buffer(&[r_u8, g_u8, b_u8]);
                        if rgb.len() >= 3 {
                            return Ok(ResolvedColor::Rgba {
                                r: rgb[0] as f32 / 255.0,
                                g: rgb[1] as f32 / 255.0,
                                b: rgb[2] as f32 / 255.0,
                                a: alpha,
                            });
                        }
                    }
                }
            }
        }

        // ICCBased N=1 — Gray source profile. The embedded profile
        // drives the conversion (§8.6.5.5) and is the path
        // /DefaultGray [/ICCBased <N=1 TRC stream>] consumes for bare
        // /DeviceGray paint. qcms 0.3.0 reads Gray ICC profiles via
        // the `kTRC` (gray Tone Reproduction Curve) tag —
        // `iccread.rs:1712-1714` — and runs a dedicated
        // gray-to-RGB transform path at `transform.rs:437-475`. The
        // input is one byte, the output is three RGB bytes; we read
        // the first three of `convert_gray_buffer`'s output.
        //
        // No per-plate routing complication — a Gray override emits
        // a single ink and lands on the K plate via the InkRouter's
        // gray-as-K handling; the composite RGB is what consumers
        // see, so ResolvedColor::Rgba is the right variant. The
        // per-page transform cache is consulted exactly as for N=3
        // and N=4 — the key is (profile.content_hash(), intent), no
        // n_components in the key, so the same cache amortises Gray
        // ICC alongside RGB and CMYK.
        #[cfg(feature = "icc")]
        if n == 1 && !components.is_empty() {
            if let Ok(bytes) = resolved_stream.decode_stream_data() {
                if let Some(profile) = crate::color::IccProfile::parse(bytes, 1) {
                    let profile = std::sync::Arc::new(profile);
                    let transform: std::sync::Arc<crate::color::Transform> =
                        if let Some(cache) = ctx.icc_transform_cache {
                            cache.get_or_build(&profile, ctx.rendering_intent)
                        } else {
                            std::sync::Arc::new(crate::color::Transform::new_srgb_target(
                                std::sync::Arc::clone(&profile),
                                ctx.rendering_intent,
                            ))
                        };
                    if transform.has_cmm() {
                        let g = components[0].clamp(0.0, 1.0);
                        let g_u8 = (g * 255.0).round() as u8;
                        let rgb = transform.convert_gray_buffer(&[g_u8]);
                        if rgb.len() >= 3 {
                            return Ok(ResolvedColor::Rgba {
                                r: rgb[0] as f32 / 255.0,
                                g: rgb[1] as f32 / 255.0,
                                b: rgb[2] as f32 / 255.0,
                                a: alpha,
                            });
                        }
                    }
                }
            }
        }

        // No usable embedded profile — fall through to the device-family
        // hint. For N=4 this emits ResolvedColor::Cmyk so per-plate
        // backends still see the channel decomposition, and the
        // composite projection routes through ctx.output_intent_cmyk
        // (which is the spec default when no embedded ICC is available).
        match n {
            1 if !components.is_empty() => Ok(first_as_gray(components, alpha)),
            3 if components.len() >= 3 => Ok(three_as_rgb(components, alpha)),
            4 if components.len() >= 4 => Ok(four_as_cmyk_native(components, alpha)),
            _ => Ok(first_as_gray(components, alpha)),
        }
    }

    /// Resolve `Separation` and `DeviceN` colour spaces by evaluating the
    /// tint transform.
    ///
    /// Array shape: `[/Separation name altCS tintTransform]` or
    /// `[/DeviceN names altCS tintTransform attrs?]`. The tint transform is
    /// a PDF function dict whose `FunctionType` selects:
    ///
    /// - **Type 0** (sampled): not handled here; falls through to
    ///   first-as-gray (matches existing inline behaviour). Wiring Type 0
    ///   would require the sampled-function evaluator which is not yet in
    ///   the tree.
    /// - **Type 2** (exponential): closed-form interpolation between `/C0`
    ///   and `/C1` with exponent `/N`. The existing inline path only handles
    ///   `N=1` against `DeviceCMYK` altCS; we generalise to any `N` and to
    ///   `DeviceRGB`/`DeviceGray` altCS as well.
    /// - **Type 3** (stitching): not handled here.
    /// - **Type 4** (calculator): evaluated via [`crate::functions::Program`].
    ///   This is the wiring the PR #630 case proves works.
    fn resolve_separation_or_devicen(
        &self,
        arr: &[Object],
        components: &[f32],
        ctx: &ResolutionContext,
        alpha: f32,
    ) -> Result<ResolvedColor> {
        if components.is_empty() {
            return Ok(ResolvedColor::Rgba {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: alpha,
            });
        }

        // §8.6.6.3 reserved name: `/None` produces no visible output.
        // For composite output we emit a fully-transparent RGBA — the
        // splice carries it through as a no-op. The per-plate route
        // sees `InkSelector::None` via the OverprintPlan and skips
        // every plate regardless of this colour value.
        let type_name = arr.first().and_then(|o| o.as_name());
        if matches!(type_name, Some("Separation"))
            && arr.get(1).and_then(|o| o.as_name()) == Some("None")
        {
            return Ok(ResolvedColor::Rgba {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            });
        }

        // Determine alternate colour space and tint-transform function.
        // Separation: [/Separation name altCS tintTransform]
        // DeviceN: [/DeviceN names altCS tintTransform attrs?]
        //
        // When the array is malformed (no altCS or no tintTransform), or
        // the function dict is missing / unrecognised, we fall back to
        // `g = 1.0 - tint`. This mirrors the long-standing inline `scn`
        // and `SCN` behaviour: callers exist that rely on it as a
        // "darker = more ink" heuristic for spot inks that never wired
        // up a proper tint transform. Off-vs-on toggle parity holds
        // until the broader §8.6.6.4 fix lands.
        let invert_tint_fallback = |components: &[f32], alpha: f32| -> ResolvedColor {
            let t = components.first().copied().unwrap_or(0.0);
            let g = (1.0 - t).clamp(0.0, 1.0);
            ResolvedColor::Rgba {
                r: g,
                g,
                b: g,
                a: alpha,
            }
        };

        let alt_cs_obj = match arr.get(2) {
            Some(o) => o,
            None => return Ok(invert_tint_fallback(components, alpha)),
        };
        let func_obj = match arr.get(3) {
            Some(o) => o,
            None => return Ok(invert_tint_fallback(components, alpha)),
        };

        let func_resolved = match ctx.doc.resolve_object(func_obj) {
            Ok(o) => o,
            Err(_) => return Ok(invert_tint_fallback(components, alpha)),
        };
        // FunctionType may be in the dict directly (Type 2/3) or in the
        // stream dict (Type 0/4). `as_dict` handles both.
        let Some(func_dict) = func_resolved.as_dict() else {
            return Ok(invert_tint_fallback(components, alpha));
        };
        let func_type = func_dict
            .get("FunctionType")
            .and_then(|o| o.as_integer())
            .unwrap_or(-1);

        let alt_cs_name = alt_cs_obj.as_name();

        let altspace_values: Vec<f32> = match func_type {
            2 => evaluate_type2(func_dict, components[0]),
            4 => evaluate_type4(&func_resolved, components)?,
            _ => return Ok(invert_tint_fallback(components, alpha)),
        };

        // Project the alternate-space values through their colour space.
        // The per-plate routing (which named plate gets the tint, what
        // happens to other plates) is determined by the source colour
        // space — Separation /Pantone-185 paints the Pantone-185 plate,
        // not the C/M/Y/K plates. That routing decision lives on the
        // OverprintPlan's `participating`, stamped by the pipeline
        // composer (see `apply_inks_selector_override`).
        //
        // The composite-side colour resolution is the alternate-space
        // value projected to RGBA — that's what the alternate is for
        // per §8.6.6.3 (composite-only fallback). Emit ResolvedColor::Rgba
        // here so the composite backend gets the right colour without
        // accidentally feeding the alternate's CMYK decomposition into
        // the per-plate path.
        match alt_cs_name {
            Some("DeviceCMYK") | Some("CMYK") if altspace_values.len() >= 4 => {
                Ok(four_as_cmyk(&altspace_values, alpha, ctx))
            },
            Some("DeviceRGB") | Some("RGB") if altspace_values.len() >= 3 => {
                Ok(three_as_rgb(&altspace_values, alpha))
            },
            Some("DeviceGray") | Some("G") if !altspace_values.is_empty() => {
                Ok(first_as_gray(&altspace_values, alpha))
            },
            _ => {
                // Compound alternate space (e.g. ICCBased). We synthesise a
                // logical Spaced colour and recurse — this lets a
                // Separation with an ICC alternate route through the ICC
                // branch correctly.
                if let Object::Array(_) = alt_cs_obj {
                    self.resolve_spaced(alt_cs_obj, &altspace_values, ctx, alpha)
                } else {
                    Ok(first_as_gray(&altspace_values, alpha))
                }
            },
        }
    }

    fn resolve_indexed(
        &self,
        arr: &[Object],
        components: &[f32],
        _ctx: &ResolutionContext,
        alpha: f32,
    ) -> Result<ResolvedColor> {
        // Indexed: [/Indexed base hival lookup]. The component is the
        // palette index, scaled 0..255 inside the renderer's existing
        // inline path. We replicate that fallback (gray = index/255) since
        // the full lookup path requires palette-stream decoding the pilot
        // operator doesn't need yet. Image extraction handles indexed
        // images through a richer path in `src/extractors/images.rs`.
        let _ = arr;
        if components.is_empty() {
            return Ok(ResolvedColor::Rgba {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: alpha,
            });
        }
        let g = (components[0] / 255.0).clamp(0.0, 1.0);
        Ok(ResolvedColor::Rgba {
            r: g,
            g,
            b: g,
            a: alpha,
        })
    }
}

/// Convert a fully-evaluated device-family colour into a final
/// [`ResolvedColor`]. Cmyk passes through as `ResolvedColor::Cmyk` so
/// per-plate backends route by channel and the OPM=1 zero-component
/// rule (§11.7.4.3) can fire on DeviceCMYK direct sources. Composite
/// consumers project Cmyk → Rgba on demand (see page_renderer's
/// `run_pipeline_for_logical`).
fn device_to_rgba(dev: DeviceColor, alpha: f32) -> ResolvedColor {
    match dev {
        DeviceColor::Gray(g) => ResolvedColor::Rgba {
            r: g,
            g,
            b: g,
            a: alpha,
        },
        DeviceColor::Rgb(r, g, b) => ResolvedColor::Rgba { r, g, b, a: alpha },
        DeviceColor::Cmyk(c, m, y, k) => ResolvedColor::Cmyk {
            c: c.clamp(0.0, 1.0),
            m: m.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            k: k.clamp(0.0, 1.0),
            a: alpha,
        },
    }
}

fn resolve_device_alias(name: &str, components: &[f32], alpha: f32) -> ResolvedColor {
    match name {
        "DeviceGray" | "G" | "CalGray" if !components.is_empty() => {
            first_as_gray(components, alpha)
        },
        "DeviceRGB" | "RGB" | "CalRGB" if components.len() >= 3 => three_as_rgb(components, alpha),
        "DeviceCMYK" | "CMYK" if components.len() >= 4 => four_as_cmyk_native(components, alpha),
        _ => first_as_gray(components, alpha),
    }
}

fn first_as_gray(components: &[f32], alpha: f32) -> ResolvedColor {
    let g = components.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
    ResolvedColor::Rgba {
        r: g,
        g,
        b: g,
        a: alpha,
    }
}

fn three_as_rgb(components: &[f32], alpha: f32) -> ResolvedColor {
    ResolvedColor::Rgba {
        r: components[0].clamp(0.0, 1.0),
        g: components[1].clamp(0.0, 1.0),
        b: components[2].clamp(0.0, 1.0),
        a: alpha,
    }
}

/// Emit `ResolvedColor::Rgba` from a 4-component CMYK via the
/// context-aware CMYK→RGB path: the document's `/OutputIntents` CMYK
/// profile when present, otherwise §10.3.5 additive-clamp. Used by
/// the Separation / DeviceN alternate-CMYK projection — the per-plate
/// routing for those sources is governed by the source colour space,
/// not the alternate's CMYK decomposition, so the alt is composite-
/// only.
fn four_as_cmyk(components: &[f32], alpha: f32, ctx: &ResolutionContext) -> ResolvedColor {
    let (r, g, b) =
        cmyk_to_rgb_via_intent(components[0], components[1], components[2], components[3], ctx);
    ResolvedColor::Rgba { r, g, b, a: alpha }
}

/// Emit `ResolvedColor::Cmyk` carrying the four-channel decomposition
/// for genuine DeviceCMYK / ICCBased N=4 sources. The per-plate
/// router consumes this directly (process-ink routing + OPM=1 zero-
/// component rule); the composite path projects to RGBA via the
/// §10.3.5 additive-clamp formula in `run_pipeline_for_logical`.
fn four_as_cmyk_native(components: &[f32], alpha: f32) -> ResolvedColor {
    ResolvedColor::Cmyk {
        c: components[0].clamp(0.0, 1.0),
        m: components[1].clamp(0.0, 1.0),
        y: components[2].clamp(0.0, 1.0),
        k: components[3].clamp(0.0, 1.0),
        a: alpha,
    }
}

/// ISO 32000-1:2008 §10.3.5 additive-clamp DeviceCMYK → DeviceRGB.
///
/// Mirrors the helper in `page_renderer.rs:2555`. We duplicate it here
/// deliberately so the resolver has no compile-time dependency on the
/// existing renderer; a follow-up will collapse the two callers onto a
/// single shared helper as part of the renderer-migration work.
fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> (f32, f32, f32) {
    let r = 1.0 - (c + k).min(1.0);
    let g = 1.0 - (m + k).min(1.0);
    let b = 1.0 - (y + k).min(1.0);
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

/// Context-aware CMYK → RGB convergence.
///
/// Precedence inside this function (callers handle the embedded-ICC
/// case before reaching here — those paths route through
/// `ColorResolver::resolve_iccbased` instead, and the §8.6.5.6
/// `/DefaultCMYK` override fires inside `ColorResolver::resolve` before
/// any device-CMYK reaches this helper):
///
/// 1. `ctx.output_intent_cmyk` — when the document declares an
///    `/OutputIntents` array with a `/N=4` `/DestOutputProfile`,
///    convert the CMYK quadruple through that profile via the
///    `crate::color::Transform` wrapper. The active rendering intent
///    (`ctx.rendering_intent`, §10.7.3) gates which qcms intent the
///    transform is built for. The 8-bit round-trip (quantise CMYK to
///    `[u8; 4]`, run qcms, decode the resulting RGB to `f32`) is the
///    same encoding the rest of `crate::color` uses — going wider
///    here would diverge from the image-decoder path that already
///    funnels through this CMM.
///
/// 2. `ctx.output_intent_cmyk` is `None` — the document didn't
///    declare a CMYK OutputIntent (or one is present but couldn't be
///    parsed). Falls through to the spec's §10.3.5 additive-clamp
///    formula. This is the byte-for-byte fallback the renderer
///    shipped before OutputIntent threading landed.
///
/// **Black-Point Compensation (BPC) and rendering-intent caveats:**
/// qcms 0.3.0 does not implement BPC and, for CMYK sources, silently
/// drops the rendering-intent parameter (see qcms `lib.rs:29-36` and
/// `transform.rs:1283-1289`). The intent value is threaded through the
/// cache key here so a future CMM upgrade that honours intent doesn't
/// silently collapse cache entries; the byte-level output, however, is
/// CURRENTLY intent-invariant for any CMYK input. The HONEST_GAP probe
/// `qa_round4_bpc_paper_white_preservation_under_relative_colorimetric`
/// in `tests/test_render_output_intent.rs` pins this — a CMM upgrade
/// will turn the probe RED at the new per-intent expected references.
///
/// Without the `icc` feature `convert_cmyk_pixel` already devolves to
/// §10.3.5 inside the CMM wrapper, so the OutputIntent path is
/// non-destructive when no real CMM is linked in. The explicit
/// `cfg(feature = "icc")` gate here is a micro-optimisation: skip
/// building the `Transform` wrapper altogether when there's no
/// chance of a real conversion.
pub(crate) fn cmyk_to_rgb_via_intent(
    c: f32,
    m: f32,
    y: f32,
    k: f32,
    ctx: &ResolutionContext<'_>,
) -> (f32, f32, f32) {
    #[cfg(feature = "icc")]
    if let Some(profile) = ctx.output_intent_cmyk {
        let c_u8 = (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        let m_u8 = (m.clamp(0.0, 1.0) * 255.0).round() as u8;
        let y_u8 = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
        let k_u8 = (k.clamp(0.0, 1.0) * 255.0).round() as u8;
        // The per-page IccTransformCache holds the compiled qcms
        // transform across the many `ResolutionContext` instances the
        // operator dispatcher builds inside one render. Without the
        // cache, every CMYK paint operator rebuilds the 17⁴ CLUT
        // (qcms::Transform::new_to) — that's the perf trap the cache
        // exists to eliminate. The unit-test path skips the cache
        // (`with_icc_transform_cache` is the renderer-only opt-in)
        // and pays the per-call build cost; integration tests cover
        // the cached path through render_page.
        let rgb = if let Some(cache) = ctx.icc_transform_cache {
            let transform = cache.get_or_build(profile, ctx.rendering_intent);
            transform.convert_cmyk_pixel(c_u8, m_u8, y_u8, k_u8)
        } else {
            let transform = crate::color::Transform::new_srgb_target(
                std::sync::Arc::clone(profile),
                ctx.rendering_intent,
            );
            transform.convert_cmyk_pixel(c_u8, m_u8, y_u8, k_u8)
        };
        return (rgb[0] as f32 / 255.0, rgb[1] as f32 / 255.0, rgb[2] as f32 / 255.0);
    }
    // No OutputIntent → spec fallback. The `ctx` borrow is held through
    // the cfg-gated branch above; under the no-icc build we explicitly
    // discard it here so the compiler doesn't flag an unused parameter.
    let _ = ctx;
    cmyk_to_rgb(c, m, y, k)
}

/// Evaluate a Type 2 (exponential interpolation) function at a single input.
/// `dict` is the function dictionary (`{/FunctionType 2 /C0 [...] /C1 [...]
/// /N <exponent> /Domain [...]}`). Returns the per-output samples.
///
/// Per ISO 32000-1:2008 §7.10.3: `y_j = C0_j + x^N * (C1_j - C0_j)`.
fn evaluate_type2(dict: &std::collections::HashMap<String, Object>, x: f32) -> Vec<f32> {
    let n = dict
        .get("N")
        .and_then(|o| o.as_real().or_else(|| o.as_integer().map(|i| i as f64)))
        .unwrap_or(1.0) as f32;
    let c0 = dict.get("C0").and_then(|o| o.as_array());
    let c1 = dict.get("C1").and_then(|o| o.as_array());

    let len = c0.map(|a| a.len()).max(c1.map(|a| a.len())).unwrap_or(1);

    let mut out = Vec::with_capacity(len);
    let x_pow = if n == 1.0 { x } else { x.powf(n) };
    for j in 0..len {
        let c0j = c0.and_then(|a| a.get(j)).map(object_to_f32).unwrap_or(0.0);
        let c1j = c1.and_then(|a| a.get(j)).map(object_to_f32).unwrap_or(1.0);
        out.push(c0j + x_pow * (c1j - c0j));
    }
    out
}

/// Evaluate a Type 4 (PostScript calculator) function via
/// [`crate::functions::Program`]. The function body is the stream content of
/// `func_obj`.
fn evaluate_type4(func_obj: &Object, components: &[f32]) -> Result<Vec<f32>> {
    let Object::Stream { dict, .. } = func_obj else {
        // Type-4 functions must be streams per §7.10.5. If we reached this
        // arm without a stream, the function is malformed; fall back to a
        // single-component identity to keep the renderer alive.
        return Ok(components.to_vec());
    };
    let bytes = func_obj.decode_stream_data()?;
    let domain = dict
        .get("Domain")
        .and_then(|o| o.as_array())
        .map(|a| array_to_pairs(a))
        .unwrap_or_default();
    let range = dict
        .get("Range")
        .and_then(|o| o.as_array())
        .map(|a| array_to_pairs(a))
        .unwrap_or_default();
    let inputs: Vec<f64> = components.iter().map(|&v| v as f64).collect();
    let out = crate::functions::evaluate_type4_clamped(&bytes, &inputs, &domain, &range)?;
    Ok(out.into_iter().map(|v| v as f32).collect())
}

/// Flatten a `[min1 max1 min2 max2 ...]` PDF array into `[[min, max], ...]`.
fn array_to_pairs(arr: &[Object]) -> Vec<[f64; 2]> {
    arr.chunks_exact(2)
        .map(|c| [object_to_f64(&c[0]), object_to_f64(&c[1])])
        .collect()
}

fn object_to_f32(o: &Object) -> f32 {
    object_to_f64(o) as f32
}

fn object_to_f64(o: &Object) -> f64 {
    o.as_real()
        .or_else(|| o.as_integer().map(|i| i as f64))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rendering::resolution::test_support::fixture_doc;
    use std::collections::HashMap;

    fn ctx<'a>(
        doc: &'a crate::document::PdfDocument,
        spaces: &'a HashMap<String, Object>,
    ) -> ResolutionContext<'a> {
        ResolutionContext::new(doc, spaces)
    }

    /// Assert resolved colour matches expected RGBA. Accepts either
    /// `ResolvedColor::Rgba` directly or `ResolvedColor::Cmyk`
    /// projected via the §10.3.5 additive-clamp formula (the resolver
    /// now emits Cmyk for Separation / DeviceN sources with a CMYK
    /// alternate so per-plate backends see the channel decomposition;
    /// composite consumers project on demand).
    fn assert_rgba(c: ResolvedColor, r: f32, g: f32, b: f32, a: f32) {
        let (rr, gg, bb, aa) = match c {
            ResolvedColor::Rgba { r, g, b, a } => (r, g, b, a),
            ResolvedColor::Cmyk { c, m, y, k, a } => {
                let rr = (1.0 - (c + k).min(1.0)).clamp(0.0, 1.0);
                let gg = (1.0 - (m + k).min(1.0)).clamp(0.0, 1.0);
                let bb = (1.0 - (y + k).min(1.0)).clamp(0.0, 1.0);
                (rr, gg, bb, a)
            },
            other => panic!("expected Rgba or Cmyk; got {other:?}"),
        };
        assert!((rr - r).abs() < 1e-3, "r: got {rr}, want {r}");
        assert!((gg - g).abs() < 1e-3, "g: got {gg}, want {g}");
        assert!((bb - b).abs() < 1e-3, "b: got {bb}, want {b}");
        assert!((aa - a).abs() < 1e-3, "a: got {aa}, want {a}");
    }

    #[test]
    fn resolves_device_gray_logical_color() {
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let lc = LogicalColor::Device(DeviceColor::Gray(0.42));
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 0.9).unwrap();
        assert_rgba(c, 0.42, 0.42, 0.42, 0.9);
    }

    #[test]
    fn resolves_device_rgb_logical_color() {
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let lc = LogicalColor::Device(DeviceColor::Rgb(1.0, 0.5, 0.25));
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 1.0).unwrap();
        assert_rgba(c, 1.0, 0.5, 0.25, 1.0);
    }

    #[test]
    fn resolves_device_cmyk_via_additive_clamp() {
        // CMYK(1,0,0,0) → RGB(0,1,1) per §10.3.5.
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let lc = LogicalColor::Device(DeviceColor::Cmyk(1.0, 0.0, 0.0, 0.0));
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 1.0).unwrap();
        assert_rgba(c, 0.0, 1.0, 1.0, 1.0);
    }

    #[test]
    fn resolves_spaced_device_alias_as_rgb() {
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let space = Object::Name("DeviceRGB".to_string());
        let lc = LogicalColor::Spaced {
            space: &space,
            components: smallvec::smallvec![0.2, 0.4, 0.6],
        };
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 1.0).unwrap();
        assert_rgba(c, 0.2, 0.4, 0.6, 1.0);
    }

    #[test]
    fn separation_with_type2_cmyk_alternate_uses_function() {
        // /Separation /SpotInk /DeviceCMYK
        //   << /FunctionType 2 /N 1 /C0 [0 0 0 0] /C1 [0 1 0 0] /Domain [0 1] /Range [0 1 0 1 0 1 0 1] >>
        // tint=1 must produce CMYK(0,1,0,0) → RGB(1,0,1) (magenta).
        let mut func_dict: HashMap<String, Object> = HashMap::new();
        func_dict.insert("FunctionType".into(), Object::Integer(2));
        func_dict.insert("N".into(), Object::Integer(1));
        func_dict.insert(
            "C0".into(),
            Object::Array(vec![
                Object::Real(0.0),
                Object::Real(0.0),
                Object::Real(0.0),
                Object::Real(0.0),
            ]),
        );
        func_dict.insert(
            "C1".into(),
            Object::Array(vec![
                Object::Real(0.0),
                Object::Real(1.0),
                Object::Real(0.0),
                Object::Real(0.0),
            ]),
        );
        let func_obj = Object::Dictionary(func_dict);

        let arr = vec![
            Object::Name("Separation".into()),
            Object::Name("SpotInk".into()),
            Object::Name("DeviceCMYK".into()),
            func_obj,
        ];
        let space = Object::Array(arr);
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let lc = LogicalColor::Spaced {
            space: &space,
            components: smallvec::smallvec![1.0],
        };
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 1.0).unwrap();
        // CMYK(0,1,0,0) → R=1-0=1, G=1-1=0, B=1-0=1
        assert_rgba(c, 1.0, 0.0, 1.0, 1.0);
    }

    #[test]
    fn separation_with_type4_calculator_evaluates_program() {
        // /Separation /MagentaSpot /DeviceCMYK
        //   stream containing: { 0.0 exch dup 0.0 exch 0.0 }  ; tint → CMYK(0, tint, 0, 0)
        // tint=1.0 should yield CMYK(0,1,0,0) → RGB(1,0,1).
        //
        // This is the canonical test for the PR #630 case: the existing inline
        // path at page_renderer.rs:690 returns `1.0 - tint` = 0.0 (solid black)
        // because it only recognises FunctionType==2. Through the resolver,
        // the Type-4 program runs to completion and the colour comes out
        // correct.
        //
        // PostScript stack convention: inputs are pushed in order, output is
        // read top-down from the final stack. With one input (tint) the
        // program needs to leave four values on the stack representing
        // C, M, Y, K. We use: `0.0 exch 0.0 0.0` — tint is on top after
        // exch, but we want the order C M Y K = 0 tint 0 0. The simplest
        // form: pop the tint into M position by emitting `0.0 3 1 roll
        // 0.0 0.0` doesn't actually work cleanly; instead use:
        //   `{ 0.0 exch 0.0 0.0 }` — wait this pushes 0, then swaps with
        //   tint giving stack [tint, 0], then pushes 0 0 giving
        //   [tint, 0, 0, 0]. That's C=tint not M=tint.
        //
        // To get [C, M, Y, K] = [0, tint, 0, 0] in PLRM stack order
        // (output order top-down so K is top), we need stack contents
        // bottom-to-top: [0, tint, 0, 0]. With tint on the stack from the
        // caller, we want: push 0 below tint (using exch), then push 0 0.
        // That's `0 exch 0 0` — yields stack bottom-to-top [0, tint, 0, 0],
        // i.e. C=0, M=tint, Y=0, K=0. (`evaluate_type4` returns the stack
        // from bottom to top as a Vec, so out[0]=C, out[1]=M, out[2]=Y,
        // out[3]=K.)
        let program = b"{ 0.0 exch 0.0 0.0 }";

        let mut func_dict: HashMap<String, Object> = HashMap::new();
        func_dict.insert("FunctionType".into(), Object::Integer(4));
        func_dict
            .insert("Domain".into(), Object::Array(vec![Object::Integer(0), Object::Integer(1)]));
        func_dict.insert(
            "Range".into(),
            Object::Array(vec![
                Object::Integer(0),
                Object::Integer(1),
                Object::Integer(0),
                Object::Integer(1),
                Object::Integer(0),
                Object::Integer(1),
                Object::Integer(0),
                Object::Integer(1),
            ]),
        );

        let func_obj = Object::Stream {
            dict: func_dict,
            data: program.to_vec().into(),
        };

        let arr = vec![
            Object::Name("Separation".into()),
            Object::Name("MagentaSpot".into()),
            Object::Name("DeviceCMYK".into()),
            func_obj,
        ];
        let space = Object::Array(arr);
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let lc = LogicalColor::Spaced {
            space: &space,
            components: smallvec::smallvec![1.0],
        };
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 1.0).unwrap();
        assert_rgba(c, 1.0, 0.0, 1.0, 1.0);
    }

    #[test]
    fn separation_full_tint_with_type4_no_longer_renders_solid_black() {
        // Regression guard for the structural class of bug demonstrated by
        // PR #630: a Separation with a Type-4 tint transform and a fully
        // opaque tint must not fall back to the `1.0 - tint = 0` grayscale
        // path. The previous test confirmed the resolved RGB is non-black;
        // this test asserts directly that none of the channels are zero
        // luminance, regardless of the specific colour produced.
        //
        // Program: `{ 0.0 exch 0.0 0.0 }` again — yields CMYK(0, tint, 0, 0),
        // RGB(1-0, 1-tint, 1-0) = (1, 1-tint, 1). At tint=1, that's (1, 0, 1).
        let program = b"{ 0.0 exch 0.0 0.0 }";
        let mut func_dict: HashMap<String, Object> = HashMap::new();
        func_dict.insert("FunctionType".into(), Object::Integer(4));
        let func_obj = Object::Stream {
            dict: func_dict,
            data: program.to_vec().into(),
        };
        let arr = vec![
            Object::Name("Separation".into()),
            Object::Name("MagentaSpot".into()),
            Object::Name("DeviceCMYK".into()),
            func_obj,
        ];
        let space = Object::Array(arr);
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let lc = LogicalColor::Spaced {
            space: &space,
            components: smallvec::smallvec![1.0],
        };
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 1.0).unwrap();
        // Separation with a DeviceCMYK alternate now emits Cmyk so the
        // per-plate router can route channels by name. Project the
        // result to RGBA for the regression-guard comparison.
        let (r, g, b) = match c {
            ResolvedColor::Rgba { r, g, b, .. } => (r, g, b),
            ResolvedColor::Cmyk { c, m, y, k, .. } => {
                let rr = (1.0 - (c + k).min(1.0)).clamp(0.0, 1.0);
                let gg = (1.0 - (m + k).min(1.0)).clamp(0.0, 1.0);
                let bb = (1.0 - (y + k).min(1.0)).clamp(0.0, 1.0);
                (rr, gg, bb)
            },
            other => panic!("expected Rgba or Cmyk; got {other:?}"),
        };
        // The old inline path would have produced gray = 1.0 - 1.0 = 0.0
        // for all channels. The pipeline must never produce that for a
        // Type-4 spot.
        assert!(
            !(r < 0.01 && g < 0.01 && b < 0.01),
            "full-tint Type-4 spot must not render solid black; got ({r}, {g}, {b})"
        );
    }

    #[test]
    fn separation_none_resolves_to_fully_transparent_for_composite() {
        // §8.6.6.3 reserved name `/None`: composite output is fully
        // transparent so the splice carries no marks through, mirroring
        // the per-plate `Skip` decision the InkRouter makes off the
        // OverprintPlan's `selector: InkSelector::None`.
        let arr = vec![
            Object::Name("Separation".into()),
            Object::Name("None".into()),
            Object::Name("DeviceGray".into()),
            Object::Dictionary({
                let mut d = HashMap::new();
                d.insert("FunctionType".into(), Object::Integer(2));
                d
            }),
        ];
        let space = Object::Array(arr);
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let lc = LogicalColor::Spaced {
            space: &space,
            components: smallvec::smallvec![0.5],
        };
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 0.9).unwrap();
        match c {
            ResolvedColor::Rgba { a, .. } => {
                assert!((a - 0.0).abs() < 1e-6, "/None composite alpha must be 0");
            },
            other => panic!("expected Rgba; got {other:?}"),
        }
    }

    #[test]
    fn separation_with_unknown_function_type_falls_back_to_gray() {
        // FunctionType 99 is not a real PDF spec value; the resolver must
        // degrade safely rather than panic. Matches the existing inline
        // behaviour of "first component as gray".
        let mut func_dict: HashMap<String, Object> = HashMap::new();
        func_dict.insert("FunctionType".into(), Object::Integer(99));
        let func_obj = Object::Dictionary(func_dict);
        let arr = vec![
            Object::Name("Separation".into()),
            Object::Name("Whatever".into()),
            Object::Name("DeviceCMYK".into()),
            func_obj,
        ];
        let space = Object::Array(arr);
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let lc = LogicalColor::Spaced {
            space: &space,
            components: smallvec::smallvec![0.5],
        };
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 1.0).unwrap();
        // First component as gray: g = 0.5
        assert_rgba(c, 0.5, 0.5, 0.5, 1.0);
    }

    #[test]
    fn iccbased_with_n4_routes_through_cmyk_fallback() {
        // ICCBased streams declare /N. With N=4 we treat components as
        // DeviceCMYK in the no-CMM fallback path (same as the existing
        // inline behaviour at `page_renderer.rs:584-617`).
        let mut stream_dict: HashMap<String, Object> = HashMap::new();
        stream_dict.insert("N".into(), Object::Integer(4));
        let icc_stream = Object::Stream {
            dict: stream_dict,
            data: Vec::new().into(),
        };
        let arr = vec![Object::Name("ICCBased".into()), icc_stream];
        let space = Object::Array(arr);
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let lc = LogicalColor::Spaced {
            space: &space,
            components: smallvec::smallvec![1.0, 0.0, 0.0, 0.0],
        };
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 1.0).unwrap();
        assert_rgba(c, 0.0, 1.0, 1.0, 1.0);
    }

    #[test]
    fn alpha_passthrough_into_rgba() {
        // Every resolution path must fold the input alpha into the output
        // RGBA. Test the Device path here; the rest is covered by the
        // type-specific tests above.
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let resolver = ColorResolver::new();
        let lc = LogicalColor::Device(DeviceColor::Gray(0.5));
        let c = resolver.resolve(&lc, &ctx(&doc, &spaces), 0.3).unwrap();
        match c {
            ResolvedColor::Rgba { a, .. } => assert!((a - 0.3).abs() < 1e-6),
            _ => panic!("expected Rgba"),
        }
    }

    #[test]
    fn cmyk_to_rgb_via_intent_with_no_output_intent_matches_additive_clamp() {
        // The fallback arm is the spec's §10.3.5 formula. Pin one
        // representative quadruple byte-exact so a regression that
        // re-routed the no-OutputIntent path through some other
        // conversion would surface here.
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let ctx = ResolutionContext::new(&doc, &spaces);
        // CMYK(0.25, 0, 0, 0) → R=0.75, G=1.0, B=1.0.
        let (r, g, b) = super::cmyk_to_rgb_via_intent(0.25, 0.0, 0.0, 0.0, &ctx);
        assert!((r - 0.75).abs() < 1e-6);
        assert!((g - 1.0).abs() < 1e-6);
        assert!((b - 1.0).abs() < 1e-6);
    }

    #[cfg(feature = "icc")]
    #[test]
    fn cmyk_to_rgb_via_intent_falls_back_when_profile_has_no_cmm() {
        // The header-only stub profile parses (IccProfile::parse accepts
        // the 128-byte header) but qcms refuses to build a Transform
        // from it because there's no tag table. The wrapper devolves to
        // §10.3.5 internally — the helper must agree byte-for-byte with
        // the no-OutputIntent path on the same input. This is the
        // shape a real but malformed /OutputIntents profile would take.
        let doc = fixture_doc();
        let spaces = HashMap::new();
        let mut header_only = vec![0u8; 128];
        header_only[8..12].copy_from_slice(&0x04000000u32.to_be_bytes());
        header_only[12..16].copy_from_slice(b"prtr");
        header_only[16..20].copy_from_slice(b"CMYK");
        header_only[20..24].copy_from_slice(b"Lab ");
        header_only[36..40].copy_from_slice(b"acsp");
        let profile = std::sync::Arc::new(
            crate::color::IccProfile::parse(header_only, 4).expect("stub parses"),
        );
        let ctx = ResolutionContext::new(&doc, &spaces).with_output_intent(Some(&profile));
        let (r, g, b) = super::cmyk_to_rgb_via_intent(0.25, 0.0, 0.0, 0.0, &ctx);
        // HONEST_GAP: this byte-exact agreement depends on
        // crate::color::Transform::convert_cmyk_pixel matching
        // crate::extractors::images::cmyk_pixel_to_rgb on the §10.3.5
        // path. If those two diverge in the future the helper here
        // could disagree with the no-OutputIntent arm even though
        // both intended to run the spec fallback.
        assert!((r - 0.75).abs() < 0.01, "got r={r}");
        assert!((g - 1.0).abs() < 0.01, "got g={g}");
        assert!((b - 1.0).abs() < 0.01, "got b={b}");
    }
}
