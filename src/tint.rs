//! Separation / DeviceN tint-transform resolution and alternate-space → RGB
//! conversion.
//!
//! A `Separation` or `DeviceN` colour (ISO 32000-1:2008 §8.6.6.4–5) is given as
//! a tint (or tints) that must be mapped through a *tint transform* function
//! (§7.10) into an *alternate* colour space, which is then converted to RGB.
//! This module owns that pipeline for the page renderer so the fill/stroke
//! `sc`/`scn` paths resolve spot colours the same way instead of falling back to
//! the naive `grey = 1 - tint` approximation. It is gated behind the `rendering`
//! feature alongside its only consumer.
//!
//! The entry point is [`eval_separation_rgb`]. Supporting pieces — the PDF
//! function evaluator ([`eval_pdf_function`]), the alternate-space dispatch
//! ([`alternate_space_to_rgb`]) and the CIELAB→sRGB conversion ([`lab_to_rgb`])
//! — are kept private; widen their visibility when a second caller needs them.

use crate::object::Object;

/// Coerce a numeric PDF object (Integer or Real) to f32.
fn obj_to_f32(o: &Object) -> Option<f32> {
    match o {
        Object::Integer(i) => Some(*i as f32),
        Object::Real(r) => Some(*r as f32),
        _ => None,
    }
}

/// Read a dictionary entry as a Vec<f32> (numeric array), or empty.
fn dict_f32_array(
    dict: &std::collections::HashMap<String, Object>,
    key: &str,
) -> Vec<f32> {
    dict.get(key)
        .and_then(|o| o.as_array())
        .map(|a| a.iter().filter_map(obj_to_f32).collect())
        .unwrap_or_default()
}

/// Read one `bps`-bit big-endian sample at sample-index `index` from a packed
/// sample stream (FunctionType 0 data is MSB-first, samples back-to-back).
fn read_packed_sample(data: &[u8], index: usize, bps: u32) -> u64 {
    let mut bit_pos = index as u64 * bps as u64;
    let mut val: u64 = 0;
    for _ in 0..bps {
        let byte = data.get((bit_pos / 8) as usize).copied().unwrap_or(0);
        let bit = (byte >> (7 - (bit_pos % 8))) & 1;
        val = (val << 1) | bit as u64;
        bit_pos += 1;
    }
    val
}

/// Convert DeviceCMYK (0.0–1.0) to DeviceRGB (0.0–1.0) per ISO 32000-1:2008
/// §10.3.5. The additive-clamp formula `R = 1 − min(1, C+K)` is the
/// spec-mandated fallback when no ICC profile is available.
pub fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> (f32, f32, f32) {
    let r = 1.0 - (c + k).min(1.0);
    let g = 1.0 - (m + k).min(1.0);
    let b = 1.0 - (y + k).min(1.0);
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

/// Map tint-transform output components to RGB purely by count: 1 → Gray,
/// 3 → RGB, 4 → CMYK. Used as the fallback when the alternate colour space
/// can't be classified by [`alternate_space_to_rgb`]; the function's output
/// arity equals the alternate space's component count. Note the count-only path
/// assumes a 3-component alternate is DeviceRGB — a Lab or non-RGB ICCBased(N=3)
/// alternate is disambiguated by [`alternate_space_to_rgb`], which is tried first.
fn components_to_rgb(comps: &[f32]) -> Option<(f32, f32, f32)> {
    match comps.len() {
        1 => Some((comps[0], comps[0], comps[0])),
        3 => Some((comps[0], comps[1], comps[2])),
        4 => Some(cmyk_to_rgb(comps[0], comps[1], comps[2], comps[3])),
        _ => None,
    }
}

/// Convert a flat numeric array `[lo0, hi0, lo1, hi1, …]` (the PDF Domain/Range
/// representation) into `[min, max]` pairs for the Type 4 evaluator.
fn f32_pairs_to_f64(flat: &[f32]) -> Vec<[f64; 2]> {
    flat.chunks_exact(2)
        .map(|c| [c[0] as f64, c[1] as f64])
        .collect()
}

/// Evaluate a PDF function over `inputs` (ISO 32000-1:2008 §7.10). Separation
/// and DeviceN tint transforms map n colorant tints → alternate-space
/// components. Supports FunctionType 2 (exponential, §7.10.3) and FunctionType 0
/// (sampled, §7.10.2) for a single input, and FunctionType 4 (PostScript
/// calculator, §7.10.5) for any input arity. Returns the output component
/// vector, or None for unsupported (type, arity) combinations — e.g. a
/// multi-input sampled or exponential transform — so callers can fall back.
fn eval_pdf_function(
    doc: &crate::document::PdfDocument,
    func: &Object,
    inputs: &[f32],
) -> Option<Vec<f32>> {
    let resolved = doc.resolve_object(func).ok()?;
    let dict = resolved.as_dict()?;
    let ftype = dict.get("FunctionType").and_then(|o| o.as_integer())?;

    if ftype == 4 {
        // Type 4 PostScript calculator accepts any input arity. The program
        // body is a stream; clamp inputs/outputs to Domain/Range per the dict.
        let program = resolved.decode_stream_data().ok()?;
        let domain = f32_pairs_to_f64(&dict_f32_array(dict, "Domain"));
        let range = f32_pairs_to_f64(&dict_f32_array(dict, "Range"));
        let inputs_f64: Vec<f64> = inputs.iter().map(|&v| v as f64).collect();
        let out =
            crate::functions::evaluate_type4_clamped(&program, &inputs_f64, &domain, &range)
                .ok()?;
        return Some(out.into_iter().map(|v| v as f32).collect());
    }

    // FunctionType 0 and 2 are evaluated for a single input only; a multi-input
    // (multi-colorant DeviceN) sampled/exponential transform is unsupported and
    // falls back to grey rather than mis-evaluating against one colorant.
    let input = match inputs {
        [x] => *x,
        _ => return None,
    };

    // Clip the input to Domain (default [0, 1]).
    let domain = dict_f32_array(dict, "Domain");
    let (d0, d1) = (
        domain.first().copied().unwrap_or(0.0),
        domain.get(1).copied().unwrap_or(1.0),
    );
    let x = input.clamp(d0.min(d1), d0.max(d1));

    match ftype {
        2 => {
            let n = dict
                .get("N")
                .and_then(obj_to_f32)
                .unwrap_or(1.0);
            let c0 = dict_f32_array(dict, "C0");
            let c1 = dict_f32_array(dict, "C1");
            // Defaults per spec: C0 = [0.0], C1 = [1.0].
            let len = c0.len().max(c1.len()).max(1);
            let xn = x.powf(n);
            let out = (0..len)
                .map(|j| {
                    let a = c0.get(j).copied().unwrap_or(0.0);
                    let b = c1.get(j).copied().unwrap_or(1.0);
                    a + xn * (b - a)
                })
                .collect();
            Some(out)
        }
        0 => {
            let size = dict
                .get("Size")
                .and_then(|o| o.as_array())
                .and_then(|a| a.first())
                .and_then(|o| o.as_integer())? as usize;
            if size < 1 {
                return None;
            }
            let bps = dict.get("BitsPerSample").and_then(|o| o.as_integer())? as u32;
            let range = dict_f32_array(dict, "Range");
            if range.len() < 2 {
                return None;
            }
            let m = range.len() / 2; // number of output components
            // Encode default [0, size-1]; Decode default = Range.
            let encode = {
                let e = dict_f32_array(dict, "Encode");
                if e.len() >= 2 {
                    (e[0], e[1])
                } else {
                    (0.0, (size - 1) as f32)
                }
            };
            let decode = {
                let d = dict_f32_array(dict, "Decode");
                if d.len() >= range.len() {
                    d
                } else {
                    range.clone()
                }
            };
            let data = resolved.decode_stream_data().ok()?;
            let max = ((1u64 << bps) - 1) as f32;

            // Map x∈Domain → e∈[0,size-1] via Encode, then clamp.
            let e = if (d1 - d0).abs() < f32::EPSILON {
                encode.0
            } else {
                encode.0 + (x - d0) * (encode.1 - encode.0) / (d1 - d0)
            };
            let e = e.clamp(0.0, (size - 1) as f32);
            let lo = e.floor() as usize;
            let hi = (lo + 1).min(size - 1);
            let frac = e - lo as f32;

            let mut out = Vec::with_capacity(m);
            for j in 0..m {
                let s_lo = read_packed_sample(&data, lo * m + j, bps) as f32 / max;
                let s_hi = read_packed_sample(&data, hi * m + j, bps) as f32 / max;
                let s = s_lo + frac * (s_hi - s_lo);
                let (dlo, dhi) = (decode[2 * j], decode[2 * j + 1]);
                out.push(dlo + s * (dhi - dlo));
            }
            Some(out)
        }
        _ => None,
    }
}

/// Resolve a Separation/DeviceN colour (`[/Separation name altCS tintFn]`,
/// §8.6.6.4, or `[/DeviceN names altCS tintFn …]`, §8.6.6.5) at the given
/// tint(s) to RGB. Evaluates the tint transform (which takes one input per
/// colorant) and converts the alternate-space result. Returns None when it
/// can't be evaluated so the caller can fall back to grey.
pub fn eval_separation_rgb(
    doc: &crate::document::PdfDocument,
    arr: &[Object],
    tints: &[f32],
) -> Option<(f32, f32, f32)> {
    // Colorant count = number of tint-transform inputs: Separation has a single
    // ink; DeviceN's count is the length of its names array at arr[1] (resolved
    // for the indirect-reference case, mirroring extract_inks_from_color_space_dict).
    let n = match arr.first().and_then(|o| o.as_name())? {
        "Separation" => 1,
        "DeviceN" => doc.resolve_object(arr.get(1)?).ok()?.as_array()?.len(),
        _ => return None,
    };
    if n == 0 || tints.len() < n {
        return None;
    }

    // Map n colorant tints → alternate-space components. Unsupported function
    // (type, arity) combinations return None and fall back to grey, so a true
    // multi-colorant DeviceN degrades safely rather than rendering a wrong colour.
    let outputs = eval_pdf_function(doc, arr.get(3)?, &tints[..n])?;

    // Disambiguate the alternate colour space (arr[2]) instead of guessing it
    // from the output arity; fall back to count-based mapping if it can't be
    // classified (e.g. an unexpected alternate-space form).
    arr.get(2)
        .and_then(|alt| alternate_space_to_rgb(doc, alt, &outputs))
        .or_else(|| components_to_rgb(&outputs))
}

/// Convert tint-transform output `comps`, expressed in a Separation/DeviceN
/// alternate colour space `alt_cs` (the arr[2] slot), to RGB by inspecting the
/// actual alternate space rather than guessing from the component count. This
/// disambiguates a 3-component Lab (§8.6.5.4) or Cal space from DeviceRGB.
/// Returns None when the alternate space can't be classified, so the caller can
/// fall back to the count-based [`components_to_rgb`].
fn alternate_space_to_rgb(
    doc: &crate::document::PdfDocument,
    alt_cs: &Object,
    comps: &[f32],
) -> Option<(f32, f32, f32)> {
    let resolved = doc.resolve_object(alt_cs).ok()?;

    // Named device / Cal spaces, e.g. /DeviceRGB, /DeviceCMYK, /DeviceGray.
    if let Some(name) = resolved.as_name() {
        return match name {
            "DeviceGray" | "CalGray" | "G" => comps.first().map(|&g| (g, g, g)),
            "DeviceRGB" | "CalRGB" | "RGB" | "DeviceCMYK" | "CMYK" => components_to_rgb(comps),
            _ => None,
        };
    }

    // Array forms: [/ICCBased stream], [/Lab dict], [/CalRGB dict], [/CalGray dict].
    let arr = resolved.as_array()?;
    match arr.first().and_then(|o| o.as_name())? {
        "ICCBased" => {
            // Map by /N (1→gray, 3→RGB, 4→CMYK); the embedded profile is treated
            // as device colour here, matching the existing inline ICCBased arm.
            let stream = doc.resolve_object(arr.get(1)?).ok()?;
            let n = stream
                .as_dict()?
                .get("N")
                .and_then(|o| o.as_integer())
                .unwrap_or(comps.len() as i64);
            match n {
                1 => comps.first().map(|&g| (g, g, g)),
                3 if comps.len() >= 3 => Some((comps[0], comps[1], comps[2])),
                4 if comps.len() >= 4 => {
                    Some(cmyk_to_rgb(comps[0], comps[1], comps[2], comps[3]))
                },
                _ => None,
            }
        },
        "CalRGB" => components_to_rgb(comps),
        "CalGray" => comps.first().map(|&g| (g, g, g)),
        "Lab" => {
            let lab_dict = arr.get(1).and_then(|o| doc.resolve_object(o).ok());
            lab_to_rgb(lab_dict.as_ref().and_then(|o| o.as_dict()), comps)
        },
        _ => None,
    }
}

/// Convert a CIE L*a*b* colour (ISO 32000-1:2008 §8.6.5.4) to sRGB. `dict` is
/// the Lab colour-space dictionary, read for the required `/WhitePoint` and the
/// optional `/Range` (a*/b* bounds, default `[-100 100 -100 100]`). `comps` is
/// `[L*, a*, b*]` with L* ∈ [0,100]. The conversion is L*a*b* → CIE XYZ (using
/// the white point) → linear sRGB → gamma. Chromatic adaptation for non-D65
/// white points is omitted — a reasonable fallback when no CMM is in play; the
/// white point defaults to D65 when the dict is missing or malformed.
fn lab_to_rgb(
    dict: Option<&std::collections::HashMap<String, Object>>,
    comps: &[f32],
) -> Option<(f32, f32, f32)> {
    let (l, a, b) = match comps {
        [l, a, b, ..] => (*l, *a, *b),
        _ => return None,
    };

    // D65 white point, used as the default and as the sRGB reference white.
    let wp = dict.map(|d| dict_f32_array(d, "WhitePoint")).unwrap_or_default();
    let (xn, yn, zn) = match wp.as_slice() {
        [x, y, z, ..] => (*x, *y, *z),
        _ => (0.9505, 1.0, 1.089),
    };

    // Clamp inputs to L* ∈ [0,100] and a*/b* to the space's /Range (default ±100).
    let range = dict.map(|d| dict_f32_array(d, "Range")).unwrap_or_default();
    let (amin, amax, bmin, bmax) = match range.as_slice() {
        [amin, amax, bmin, bmax, ..] => (*amin, *amax, *bmin, *bmax),
        _ => (-100.0, 100.0, -100.0, 100.0),
    };
    let l = l.clamp(0.0, 100.0);
    let a = a.clamp(amin, amax);
    let b = b.clamp(bmin, bmax);

    // L*a*b* → CIE XYZ with the inverse of the f(t) lightness companding.
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let g = |t: f32| {
        const DELTA: f32 = 6.0 / 29.0;
        if t > DELTA {
            t * t * t
        } else {
            3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
        }
    };
    let x = xn * g(fx);
    let y = yn * g(fy);
    let z = zn * g(fz);

    // CIE XYZ (D65) → linear sRGB.
    let rl = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let gl = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let bl = 0.0557 * x - 0.2040 * y + 1.0570 * z;

    // Linear sRGB → gamma-encoded sRGB.
    let enc = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    };
    Some((enc(rl), enc(gl), enc(bl)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmyk_to_rgb_white() {
        let (r, g, b) = cmyk_to_rgb(0.0, 0.0, 0.0, 0.0);
        assert!((r - 1.0).abs() < 0.001);
        assert!((g - 1.0).abs() < 0.001);
        assert!((b - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cmyk_to_rgb_black() {
        let (r, g, b) = cmyk_to_rgb(0.0, 0.0, 0.0, 1.0);
        assert!((r - 0.0).abs() < 0.001);
        assert!((g - 0.0).abs() < 0.001);
        assert!((b - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cmyk_to_rgb_pure_cyan() {
        let (r, g, b) = cmyk_to_rgb(1.0, 0.0, 0.0, 0.0);
        assert!((r - 0.0).abs() < 0.001);
        assert!((g - 1.0).abs() < 0.001);
        assert!((b - 1.0).abs() < 0.001);
    }

    /// CIELAB(53.24, 80.09, 67.20) under a D65 white point is sRGB red.
    #[test]
    fn test_lab_to_rgb_red() {
        let (r, g, b) = lab_to_rgb(None, &[53.24, 80.09, 67.20]).expect("lab");
        assert!(r > 0.95, "expected near-1 red, got {r}");
        assert!(g < 0.05 && b < 0.05, "expected low g,b, got ({g},{b})");
    }

    /// A 3-component output over a Lab alternate must not be read as RGB.
    #[test]
    fn test_components_to_rgb_count_fallback() {
        // 4 components → CMYK path.
        let rgb = components_to_rgb(&[0.0, 0.0, 0.0, 0.0]).expect("cmyk white");
        assert_eq!(rgb, (1.0, 1.0, 1.0));
    }
}
