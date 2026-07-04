//! JPEG 2000 (`/JPXDecode`) image decoding via hayro-jpeg2000.
//!
//! ISO 32000-1 §7.4.9: a `/JPXDecode` stream is a JPEG 2000 codestream — either a
//! raw J2K codestream or a JP2-boxed file. hayro-jpeg2000 handles both. This decodes
//! the codestream to interleaved 8-bit-per-component samples; the caller maps the
//! component count to a colour space and applies `/Decode`, `/SMask`, etc.
//!
//! Feature-gated (`jpeg2000`): when the feature is off the call site returns the
//! existing `UnsupportedFilter` error rather than panicking.

#[cfg(feature = "jpeg2000")]
use crate::error::Error;
use crate::error::Result;

/// Pass-through filter for `/JPXDecode`.
///
/// Like `DCTDecode`/`JBIG2Decode`, the JPEG 2000 codestream is not decompressed
/// by the generic filter pipeline — it is handed to the image extractor, which
/// decodes it with hayro-jpeg2000 (`decode_jpx`). So this decoder returns its input
/// unchanged. It is always available (even without the `jpeg2000` feature) so the
/// pipeline can surface the codestream; the extractor's feature-gated path then
/// either decodes it or returns a typed `UnsupportedFilter` error.
pub struct JpxDecoder;

impl super::StreamDecoder for JpxDecoder {
    fn decode(&self, input: &[u8]) -> Result<Vec<u8>> {
        Ok(input.to_vec())
    }

    fn name(&self) -> &str {
        "JPXDecode"
    }
}

/// A decoded JPEG 2000 image: interleaved 8-bit samples plus component count.
#[cfg(feature = "jpeg2000")]
pub struct JpxImage {
    /// `width * height * num_components` bytes, component-interleaved (row-major).
    pub samples: Vec<u8>,
    pub num_components: u8,
}

/// Decode a JP2/J2K codestream to interleaved 8-bit-per-component samples.
///
/// hayro-jpeg2000 yields one f32 plane per component (normalized to the component's
/// bit depth); `DecodedImage::data_u8()` interleaves these to 8-bit samples.
/// Components are assumed to share the image dimensions (no chroma subsampling) —
/// the common case for PDF image XObjects; a subsampled component is rejected with a
/// typed error rather than producing misaligned output.
#[cfg(feature = "jpeg2000")]
pub fn decode_jpx(bytes: &[u8]) -> Result<JpxImage> {
    use hayro_jpeg2000::{DecodeSettings, DecoderContext, Image};

    let image = Image::new(bytes, &DecodeSettings::default()).map_err(|e| {
        Error::UnsupportedFilter(format!("JPXDecode: JPEG 2000 decode failed: {e:?}"))
    })?;

    let width = image.width();
    let height = image.height();
    let npix = width as usize * height as usize;

    let mut ctx = DecoderContext::default();
    let decoded = image.decode(&mut ctx).map_err(|e| {
        Error::UnsupportedFilter(format!("JPXDecode: JPEG 2000 decode failed: {e:?}"))
    })?;

    let comps = decoded.components();
    if comps.is_empty() {
        return Err(Error::UnsupportedFilter(
            "JPXDecode: JPEG 2000 image has no components".to_string(),
        ));
    }
    let num_components = comps.len();

    // Fast path: every component is full-resolution (the common case) → use the
    // decoder's own interleave.
    if comps.iter().all(|c| c.samples().len() == npix) {
        return Ok(JpxImage {
            samples: decoded.data_u8(),
            num_components: num_components as u8,
        });
    }

    // Chroma-subsampled path (WS1.7). hayro-jpeg2000 0.4 does not expose
    // per-component dimensions, so only the unambiguous 2×2 (4:2:0) case is
    // recovered: a component with ⌈w/2⌉·⌈h/2⌉ samples is nearest-upsampled to
    // full resolution; any other ratio (or non-8-bit depth, where the f32→u8
    // scaling would differ) stays unsupported rather than guessing. Components
    // are then interleaved manually since `data_u8` assumes equal plane sizes.
    let (w, h) = (width as usize, height as usize);
    let (sw, sh) = (width.div_ceil(2) as usize, height.div_ceil(2) as usize);
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(num_components);
    for (ci, comp) in comps.iter().enumerate() {
        if comp.bit_depth() != 8 {
            return Err(Error::UnsupportedFilter(format!(
                "JPXDecode: subsampled component {ci} with {}-bit depth not supported",
                comp.bit_depth()
            )));
        }
        let s = comp.samples();
        let plane = if s.len() == npix {
            s.iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect()
        } else if s.len() == sw * sh {
            upsample_nearest_u8(s, sw, sh, w, h)
        } else {
            return Err(Error::UnsupportedFilter(format!(
                "JPXDecode: subsampled component {ci} ({} samples) — only 2×2 (4:2:0) \
                 subsampling of a {width}×{height} image is supported",
                s.len()
            )));
        };
        planes.push(plane);
    }

    let mut samples = vec![0u8; npix * num_components];
    for (ci, plane) in planes.iter().enumerate() {
        for (i, &px) in plane.iter().enumerate() {
            samples[i * num_components + ci] = px;
        }
    }
    Ok(JpxImage {
        samples,
        num_components: num_components as u8,
    })
}

/// Nearest-neighbour upsample of an `sw×sh` f32 sample plane to `fw×fh` u8.
#[cfg(feature = "jpeg2000")]
fn upsample_nearest_u8(sub: &[f32], sw: usize, sh: usize, fw: usize, fh: usize) -> Vec<u8> {
    let mut out = vec![0u8; fw * fh];
    for y in 0..fh {
        let sy = (y * sh / fh).min(sh.saturating_sub(1));
        for x in 0..fw {
            let sx = (x * sw / fw).min(sw.saturating_sub(1));
            out[y * fw + x] = sub[sy * sw + sx].round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

#[cfg(all(test, feature = "jpeg2000"))]
mod tests {
    use super::{decode_jpx, upsample_nearest_u8};

    /// Grayscale JP2 codestream from the minimal repro (816x1056 DeviceGray).
    const SAMPLE_JP2: &[u8] = include_bytes!("../../tests/fixtures/jpx/sample_gray.jp2");

    /// WS1.7: nearest-neighbour upsample of a 2×2 subsampled plane to 4×4 —
    /// each source sample fills its 2×2 output block.
    #[test]
    fn upsample_nearest_2x2_to_4x4() {
        // 2×2 plane: [10 20 / 30 40]
        let sub = [10.0f32, 20.0, 30.0, 40.0];
        let out = upsample_nearest_u8(&sub, 2, 2, 4, 4);
        assert_eq!(
            out,
            vec![
                10, 10, 20, 20, //
                10, 10, 20, 20, //
                30, 30, 40, 40, //
                30, 30, 40, 40,
            ]
        );
    }

    /// Odd full dimensions (⌈w/2⌉ source): upsample 2×2 → 3×3 clamps at edges.
    #[test]
    fn upsample_nearest_2x2_to_3x3() {
        let sub = [1.0f32, 2.0, 3.0, 4.0];
        let out = upsample_nearest_u8(&sub, 2, 2, 3, 3);
        assert_eq!(out.len(), 9);
        assert_eq!(out[0], 1); // (0,0)
        assert_eq!(out[8], 4); // (2,2) → source (1,1)
    }

    #[test]
    fn decode_jpx_grayscale() {
        let img = decode_jpx(SAMPLE_JP2).expect("decode JP2 codestream");

        assert_eq!(img.num_components, 1);
        assert_eq!(img.samples.len(), 816 * 1056);

        // A scanned page is not one flat value.
        let first = img.samples[0];
        assert!(
            img.samples.iter().any(|&b| b != first),
            "decoded image is uniformly flat — decode likely failed"
        );
    }
}
