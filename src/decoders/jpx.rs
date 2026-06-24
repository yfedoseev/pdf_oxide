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

    for (ci, comp) in comps.iter().enumerate() {
        if comp.samples().len() != npix {
            return Err(Error::UnsupportedFilter(format!(
                "JPXDecode: subsampled JPEG 2000 component {ci} ({} samples vs \
                 {width}x{height}={npix}) not supported",
                comp.samples().len()
            )));
        }
    }

    Ok(JpxImage {
        samples: decoded.data_u8(),
        num_components: num_components as u8,
    })
}

#[cfg(all(test, feature = "jpeg2000"))]
mod tests {
    use super::decode_jpx;

    /// Grayscale JP2 codestream from the minimal repro (816x1056 DeviceGray).
    const SAMPLE_JP2: &[u8] = include_bytes!("../../tests/fixtures/jpx/sample_gray.jp2");

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
