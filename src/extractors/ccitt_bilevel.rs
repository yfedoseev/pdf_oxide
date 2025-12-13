//! CCITT Group 4 decompression for bilevel images.
//!
//! This module handles decompression of CCITT Group 4 encoded bilevel (1-bit) images
//! extracted from PDF documents, and converts them to 8-bit grayscale for OCR processing.
//!
//! PDF Spec: ISO 32000-1:2008, Section 7.4.6 - CCITTFaxDecode Filter
//! CCITT Spec: ITU-T Recommendation T.6 - Facsimile coding schemes and coding control functions

use crate::decoders::CcittParams;
use crate::error::{Error, Result};

/// Decompresses CCITT encoded data (Group 3 or Group 4).
///
/// CCITT (Consultative Committee for International Telegraphy and Telephony) is a binary
/// compression format used in TIFF and PDF for bilevel (1-bit) images. This is the standard
/// compression for scanned documents.
///
/// # Arguments
///
/// * `data` - CCITT compressed data
/// * `params` - CCITT decompression parameters from PDF /DecodeParms dictionary
///
/// # Returns
///
/// A vector of bytes representing the decompressed bilevel image.
/// Each byte contains 8 pixels (MSB = leftmost pixel, LSB = rightmost pixel).
/// Pixels are encoded as: 0 = white, 1 = black (unless /BlackIs1=true, then inverted).
pub fn decompress_ccitt(data: &[u8], params: &CcittParams) -> Result<Vec<u8>> {
    // Validate required parameters
    if params.columns == 0 {
        return Err(Error::Decode("CCITT decompression requires /Columns parameter".to_string()));
    }

    let width = params.columns as u16;
    let height_opt = params.rows.map(|h| h as u16);

    log::debug!(
        "CCITT decompression: {} bytes, {}x{} pixels, K={}, BlackIs1={}",
        data.len(),
        params.columns,
        params.rows.unwrap_or(0),
        params.k,
        params.black_is_1
    );

    // Support both Group 3 and Group 4
    if params.is_group_3() {
        log::debug!(
            "CCITT Group 3 decompression requested (K={})",
            params.k
        );
    } else {
        log::debug!("CCITT Group 4 decompression requested");
    }

    match decompress_with_ccitt_t4_t6(data, width, height_opt, params) {
        Ok(mut output) => {
            let rows = output.len() / ((width as usize + 7) / 8);
            log::debug!(
                "CCITT decompressed: {} bytes -> {} bytes ({} rows)",
                data.len(),
                output.len(),
                rows
            );

            // Handle /BlackIs1 parameter if needed
            if params.black_is_1 {
                invert_bilevel_pixels(&mut output);
            }

            Ok(output)
        }
        Err(e) => {
            log::warn!(
                "CCITT decompression failed: {}x{} pixels, {} bytes: {}",
                params.columns,
                params.rows.unwrap_or(0),
                data.len(),
                e
            );
            log::info!(
                "Check /DecodeParms: /EndOfLine={}, /EncodedByteAlign={}, /EndOfBlock={}",
                params.end_of_line,
                params.encoded_byte_align,
                params.end_of_block
            );
            // Fallback: return white pixels
            let expected_bytes = height_opt.unwrap_or(1) as usize * ((width as usize + 7) / 8);
            log::info!("Returning {} bytes of white pixels as fallback", expected_bytes);
            Ok(vec![0; expected_bytes])
        }
    }
}

/// Decompress CCITT data using the ccitt-t4-t6 crate.
fn decompress_with_ccitt_t4_t6(
    data: &[u8],
    width: u16,
    height: Option<u16>,
    _params: &CcittParams,
) -> Result<Vec<u8>> {
    use ccitt_t4_t6::bit_iter::BitWriter;
    use ccitt_t4_t6::g42d::decode::Decoder;

    let width_usize = width as usize;

    log::debug!(
        "Attempting CCITT decompression: width={}, height={:?}, data_len={}",
        width,
        height,
        data.len()
    );

    // Try decompressing with the original data first
    let result = try_decode_ccitt(data, width_usize);

    if result.is_ok() {
        return result;
    }

    // If that failed, try stripping leading zeros
    log::debug!("First decompression attempt failed, trying with leading zeros stripped");
    let trimmed_data = data.iter()
        .skip_while(|b| **b == 0)
        .copied()
        .collect::<Vec<_>>();

    if trimmed_data.len() < data.len() {
        log::debug!(
            "Stripped {} leading zero bytes, attempting decompression",
            data.len() - trimmed_data.len()
        );

        if let Ok(output) = try_decode_ccitt(&trimmed_data, width_usize) {
            return Ok(output);
        }
    }

    // Both attempts failed
    result
}

fn try_decode_ccitt(data: &[u8], width: usize) -> Result<Vec<u8>> {
    use ccitt_t4_t6::bit_iter::BitWriter;
    use ccitt_t4_t6::g42d::decode::Decoder;

    let mut decoder = Decoder::<BitWriter>::new(width);

    // Decode the CCITT compressed data
    decoder.decode(data)
        .map_err(|e| Error::Decode(format!("CCITT decompression failed: {}", e)))?;

    let packed_bits = decoder.into_store().done();

    log::debug!(
        "CCITT decompression successful: {} bytes input -> {} bytes output",
        data.len(),
        packed_bits.len()
    );

    Ok(packed_bits)
}

/// Decompresses CCITT Group 4 encoded data (legacy API for backwards compatibility).
///
/// This is a convenience function that uses default CCITT parameters.
#[deprecated(since = "0.1.5", note = "Use decompress_ccitt with CcittParams instead")]
pub fn decompress_ccitt_group4(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let params = CcittParams {
        columns: width,
        rows: Some(height),
        ..Default::default()
    };
    decompress_ccitt(data, &params)
}

/// Invert all bits in a bilevel image.
///
/// This is used when /BlackIs1=true to convert from:
/// - white=1, black=0 (inverted representation)
/// to standard PDF representation:
/// - white=0, black=1
fn invert_bilevel_pixels(data: &mut [u8]) {
    for byte in data.iter_mut() {
        *byte = !*byte;
    }
}


/// Convert 1-bit bilevel image to 8-bit grayscale.
///
/// Each bit in the input is expanded to a full byte where:
/// - 0 (white) -> 0xFF (white in 8-bit)
/// - 1 (black) -> 0x00 (black in 8-bit)
///
/// # Arguments
///
/// * `bilevel_data` - Packed bilevel image data (1 bit per pixel)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Returns
///
/// A vector of 8-bit grayscale pixels suitable for image processing and OCR.
pub fn bilevel_to_grayscale(bilevel_data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    let mut grayscale = Vec::with_capacity(width * height);

    for row_idx in 0..height {
        // Each row in bilevel data is padded to byte boundary
        let row_start = row_idx * ((width + 7) / 8);

        for col_idx in 0..width {
            let byte_idx = row_start + (col_idx / 8);
            if byte_idx < bilevel_data.len() {
                let bit_pos = 7 - (col_idx % 8);
                let bit = (bilevel_data[byte_idx] >> bit_pos) & 1;
                // 0 (white) -> 0xFF, 1 (black) -> 0x00
                // Standard interpretation for CCITT/fax images
                grayscale.push(if bit == 0 { 0xFF } else { 0x00 });
            } else {
                // Out of bounds - default to white
                grayscale.push(0xFF);
            }
        }
    }

    grayscale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilevel_to_grayscale() {
        // Test converting 1-bit bilevel to 8-bit grayscale
        // Pattern: 10000001 (black, white, white, white, white, white, white, black)
        let bilevel = vec![0b10000001];
        let grayscale = bilevel_to_grayscale(&bilevel, 8, 1);

        assert_eq!(grayscale.len(), 8);
        assert_eq!(grayscale[0], 0x00, "Pixel 0 should be black");
        assert_eq!(grayscale[1], 0xFF, "Pixel 1 should be white");
        assert_eq!(grayscale[7], 0x00, "Pixel 7 should be black");
    }

    #[test]
    fn test_bilevel_to_grayscale_padding() {
        // Test with non-byte-aligned width
        let bilevel = vec![0b10000001];
        let grayscale = bilevel_to_grayscale(&bilevel, 5, 1);

        assert_eq!(grayscale.len(), 5);
        assert_eq!(grayscale[0], 0x00); // bit 7
        assert_eq!(grayscale[4], 0x00); // bit 3
    }

    #[test]
    fn test_transitions_to_bytes() {
        // Test transitions: white=3, black=2, white=3
        // Should produce: 0b00111001 = 0x39
        let transitions = vec![3, 2, 3];
        let row = transitions_to_bytes(&transitions, 8);

        assert_eq!(row.len(), 1);
        // WW WBBWWW = 00111001
        assert_eq!(row[0], 0b00111001);
    }
}
