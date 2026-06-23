//! JPEG 2000 (`/JPXDecode`) image decoding via OpenJPEG (the `jpeg2k` crate).
//!
//! ISO 32000-1 §7.4.9: a `/JPXDecode` stream is a JPEG 2000 codestream — either a
//! raw J2K codestream or a JP2-boxed file. OpenJPEG handles both. This decodes the
//! codestream to interleaved 8-bit-per-component samples; the caller maps the
//! component count to a colour space and applies `/Decode`, `/SMask`, etc.
//!
//! Feature-gated (`jpeg2000`): the decoder is a C dependency (OpenJPEG, vendored via
//! `openjpeg-sys`) and is not built for `wasm32`. When the feature is off the call
//! site returns the existing `UnsupportedFilter` error rather than panicking.

#[cfg(feature = "jpeg2000")]
use crate::error::Error;
use crate::error::Result;

/// Pass-through filter for `/JPXDecode`.
///
/// Like `DCTDecode`/`JBIG2Decode`, the JPEG 2000 codestream is not decompressed
/// by the generic filter pipeline — it is handed to the image extractor, which
/// decodes it with OpenJPEG (`decode_jpx`). So this decoder returns its input
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
/// OpenJPEG yields one `i32` plane per component at the component's own bit depth;
/// each plane is down-shifted to 8 bits and interleaved. Components are assumed to
/// share the image dimensions (no chroma subsampling) — the common case for PDF
/// image XObjects and both `#755` repros; a subsampled component is rejected with a
/// typed error rather than producing misaligned output.
#[cfg(feature = "jpeg2000")]
pub fn decode_jpx(bytes: &[u8]) -> Result<JpxImage> {
    let image = jpeg2k::Image::from_bytes(bytes).map_err(|e| {
        Error::UnsupportedFilter(format!("JPXDecode: JPEG 2000 decode failed: {e:?}"))
    })?;

    let width = image.width();
    let height = image.height();
    let comps = image.components();
    if comps.is_empty() {
        return Err(Error::UnsupportedFilter(
            "JPXDecode: JPEG 2000 image has no components".to_string(),
        ));
    }
    let num_components = comps.len();
    let npix = width as usize * height as usize;
    let mut samples = vec![0u8; npix * num_components];

    for (ci, comp) in comps.iter().enumerate() {
        if comp.width() != width || comp.height() != height {
            return Err(Error::UnsupportedFilter(format!(
                "JPXDecode: subsampled JPEG 2000 component {ci} ({}x{} vs {width}x{height}) \
                 not supported",
                comp.width(),
                comp.height()
            )));
        }
        let data = comp.data();
        // Down-shift the component's native precision to 8 bits-per-sample.
        let shift = comp.precision().saturating_sub(8);
        let n = npix.min(data.len());
        for i in 0..n {
            samples[i * num_components + ci] = (data[i] >> shift).clamp(0, 255) as u8;
        }
    }

    Ok(JpxImage {
        samples,
        num_components: num_components as u8,
    })
}

#[cfg(all(test, feature = "jpeg2000"))]
mod tests {
    use super::decode_jpx;

    /// Grayscale JP2 codestream from the #755 minimal repro (816x1056 DeviceGray).
    const SAMPLE_JP2: &[u8] = include_bytes!("../../tests/fixtures/jpx/sample_gray.jp2");

    #[test]
    fn decode_jpx_grayscale_matches_openjpeg() {
        let img = decode_jpx(SAMPLE_JP2).expect("decode JP2 codestream");

        assert_eq!(img.num_components, 1);
        assert_eq!(img.samples.len(), 816 * 1056);

        // Position-weighted checksum of the samples; the reference value is the
        // OpenJPEG/Pillow decode (md5 a02dc1a2…), so a match means we emit the
        // same pixels OpenJPEG does — not merely the right count.
        let hash = img
            .samples
            .iter()
            .fold(0u64, |h, &b| h.wrapping_mul(31).wrapping_add(b as u64));
        assert_eq!(hash, 17132179737692472799);

        // A scanned page is not one flat value.
        let first = img.samples[0];
        assert!(
            img.samples.iter().any(|&b| b != first),
            "decoded image is uniformly flat — decode likely failed"
        );
    }
}
