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
        log::debug!("CCITT Group 3 decompression requested (K={})", params.k);
    } else {
        log::debug!("CCITT Group 4 decompression requested");
    }

    match decompress_with_fax(data, width, height_opt, params) {
        Ok(mut output) => {
            let rows = output.len() / (width as usize).div_ceil(8);
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

            log::info!("CCITT decompression successful!");
            Ok(output)
        },
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
            let expected_bytes = height_opt.unwrap_or(1) as usize * (width as usize).div_ceil(8);
            log::info!("Returning {} bytes of white pixels as fallback", expected_bytes);
            Ok(vec![0; expected_bytes])
        },
    }
}

/// Decompress CCITT data using the fax crate.
///
/// The fax crate is more lenient with malformed EOFB markers compared to ccitt-t4-t6,
/// which makes it better suited for handling real-world PDF files that don't strictly
/// comply with the CCITT specification.
fn decompress_with_fax(
    data: &[u8],
    width: u16,
    height: Option<u16>,
    params: &CcittParams,
) -> Result<Vec<u8>> {
    let width_usize = width as usize;

    log::debug!(
        "Attempting CCITT decompression with fax crate: width={}, height={:?}, data_len={}, K={}",
        width,
        height,
        data.len(),
        params.k
    );

    // Try with original data first
    match try_decode_with_fax(data, width_usize, height, params) {
        Ok(output) if !output.is_empty() => {
            return Ok(output);
        },
        Ok(_empty) => {
            log::debug!("First attempt returned no data, trying with leading zeros stripped");
        },
        Err(e) => {
            log::debug!("First attempt failed: {}, trying with leading zeros stripped", e);
        },
    }

    // If that failed, try stripping leading zeros (common in some PDFs)
    let trimmed_data = data
        .iter()
        .skip_while(|b| **b == 0)
        .copied()
        .collect::<Vec<_>>();

    if trimmed_data.len() < data.len() && !trimmed_data.is_empty() {
        log::debug!(
            "Stripped {} leading zero bytes ({} -> {}), attempting decompression",
            data.len() - trimmed_data.len(),
            data.len(),
            trimmed_data.len()
        );

        log::debug!(
            "Data after stripping zeros, first 32 bytes: {}",
            trimmed_data
                .iter()
                .take(32)
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(" ")
        );

        match try_decode_with_fax(&trimmed_data, width_usize, height, params) {
            Ok(output) if !output.is_empty() => {
                log::info!("Successfully decompressed after stripping leading zeros!");
                return Ok(output);
            },
            Ok(_) => {
                log::debug!("Strip attempt also returned no data");
            },
            Err(e) => {
                log::debug!("Strip attempt also failed: {}", e);
            },
        }
    }

    // Both attempts failed - return error
    Err(Error::Decode(
        "CCITT decompression failed: fax decoder returned no output".to_string(),
    ))
}

fn try_decode_with_fax(
    data: &[u8],
    width: usize,
    height: Option<u16>,
    params: &CcittParams,
) -> Result<Vec<u8>> {
    use fax::decoder;

    let mut output_rows = Vec::new();
    let bytes_per_row = width.div_ceil(8);

    // Use fax crate's decoder which is more lenient with malformed EOFB
    let bytes_iter = data.iter().copied();

    let success = if params.is_group_4() {
        log::debug!("Using Group 4 (T.6) decoder");
        decoder::decode_g4(bytes_iter, width as u16, height, |transitions: &[u16]| {
            // Convert run-length transitions to pixel bytes
            let row_bytes = transitions_to_bytes(transitions, width);
            output_rows.push(row_bytes);
        })
    } else {
        log::debug!("Using Group 3 (T.4) decoder");
        // Group 3 has a different signature - no width/height params in callback
        decoder::decode_g3(bytes_iter, |transitions: &[u16]| {
            // Convert run-length transitions to pixel bytes
            let row_bytes = transitions_to_bytes(transitions, width);
            output_rows.push(row_bytes);
        })
    };

    // Check if decoder succeeded and returned data
    if success.is_some() && !output_rows.is_empty() {
        let output = output_rows.into_iter().flatten().collect::<Vec<u8>>();
        log::debug!(
            "CCITT decompression successful: {} bytes input -> {} bytes output ({} rows)",
            data.len(),
            output.len(),
            output.len() / bytes_per_row
        );
        Ok(output)
    } else if success.is_some() {
        // Decoder succeeded but produced no output - unusual but valid
        log::debug!("CCITT decoder returned success but no rows produced");
        Ok(Vec::new())
    } else {
        // Decoder failed
        log::warn!("CCITT fax decoder returned None");
        Err(Error::Decode("CCITT fax decoder failed".to_string()))
    }
}

/// Convert run-length transition positions to byte-packed pixels.
///
/// The transitions array contains positions where the color changes from white to black
/// or black to white, starting with white. For example, [3, 5, 8] means:
/// - Pixels 0-2: white
/// - Pixels 3-4: black
/// - Pixels 5-7: white
fn transitions_to_bytes(transitions: &[u16], width: usize) -> Vec<u8> {
    let bytes_per_row = width.div_ceil(8);
    let mut row_bytes = vec![0u8; bytes_per_row];

    let mut is_black = false; // Start with white
    let mut start_pos = 0u16;

    for &transition_pos in transitions {
        let transition_pos = transition_pos as usize;
        if is_black {
            // Fill black pixels from start_pos to transition_pos
            for pixel_idx in start_pos as usize..transition_pos.min(width) {
                let byte_idx = pixel_idx / 8;
                let bit_idx = 7 - (pixel_idx % 8);
                row_bytes[byte_idx] |= 1 << bit_idx;
            }
        }
        // Switch color for next run
        is_black = !is_black;
        start_pos = transition_pos as u16;
    }

    // Handle remaining pixels in the last run
    if is_black && (start_pos as usize) < width {
        for pixel_idx in (start_pos as usize)..width {
            let byte_idx = pixel_idx / 8;
            let bit_idx = 7 - (pixel_idx % 8);
            row_bytes[byte_idx] |= 1 << bit_idx;
        }
    }

    row_bytes
}

/// Decompresses CCITT Group 4 encoded data (legacy API for backwards compatibility).
///
/// This is a convenience function that uses default CCITT parameters.
#[deprecated(
    since = "0.1.5",
    note = "Use decompress_ccitt with CcittParams instead"
)]
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
///
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
        let row_start = row_idx * width.div_ceil(8);

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
        // Pattern: 10000001
        // Pixels 0-4: 1=black, 0=white, 0=white, 0=white, 0=white
        let bilevel = vec![0b10000001];
        let grayscale = bilevel_to_grayscale(&bilevel, 5, 1);

        assert_eq!(grayscale.len(), 5);
        assert_eq!(grayscale[0], 0x00); // bit 7 = 1 (black)
        assert_eq!(grayscale[1], 0xFF); // bit 6 = 0 (white)
        assert_eq!(grayscale[4], 0xFF); // bit 3 = 0 (white)
    }

    #[test]
    fn test_transitions_to_bytes() {
        // Test transitions to build pattern: WW|BBB|WW|B
        // Transitions at positions [2, 5, 7]:
        // - White from 0-2 (2 pixels)
        // - Black from 2-5 (3 pixels)
        // - White from 5-7 (2 pixels)
        // - Black from 7-8 (1 pixel)
        // Should produce: 0b00111001 = 57 (0x39)
        let transitions = vec![2, 5, 7];
        let row = transitions_to_bytes(&transitions, 8);

        assert_eq!(row.len(), 1);
        assert_eq!(row[0], 0b00111001);
    }
}
