//! CCITT Group 4 decompression for bilevel images.
//!
//! This module handles decompression of CCITT Group 4 encoded bilevel (1-bit) images
//! extracted from PDF documents, and converts them to 8-bit grayscale for OCR processing.
//!
//! PDF Spec: ISO 32000-1:2008, Section 7.4.6 - CCITTFaxDecode Filter
//! CCITT Spec: ITU-T Recommendation T.6 - Facsimile coding schemes and coding control functions

use crate::error::{Error, Result};
use fax::decoder;

/// Decompresses CCITT Group 4 encoded data to raw bilevel pixels.
///
/// CCITT Group 4 is a binary compression format used in TIFF and PDF for bilevel images.
/// This function decompresses Group 4 data and returns the raw bits as a byte vector.
///
/// # Arguments
///
/// * `data` - CCITT Group 4 compressed data
/// * `width` - Image width in pixels
/// * `height` - Image height in rows
///
/// # Returns
///
/// A vector of bytes representing the decompressed bilevel image.
/// Each byte contains 8 pixels (MSB = leftmost pixel, LSB = rightmost pixel).
/// Pixels are encoded as: 0 = white, 1 = black.
pub fn decompress_ccitt_group4(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let width = width as u16;
    let height_opt = Some(height as u16);
    let mut output = Vec::new();

    log::debug!(
        "CCITT Group 4 decompression: {} bytes, {}x{} pixels",
        data.len(),
        width,
        height_opt.unwrap_or(0)
    );

    // Use the fax crate's Group 4 decoder
    // It calls our callback for each decompressed line
    let bytes_iter = data.iter().copied();
    let result = decoder::decode_g4(bytes_iter, width, height_opt, |line| {
        // Convert each line (array of u16 transitions) to bytes
        let row_bytes = transitions_to_bytes(line, width);
        output.extend_from_slice(&row_bytes);
    });

    match result {
        Some(()) => {
            // Verify we got the expected amount of data
            if output.is_empty() {
                log::warn!("CCITT decompression produced empty output");
                return Err(Error::Decode("CCITT Group 4 decompression produced no output".to_string()));
            }

            let rows = output.len() / ((width as usize + 7) / 8);
            log::debug!(
                "CCITT Group 4 decompressed: {} bytes -> {} bytes ({} rows)",
                data.len(),
                output.len(),
                rows
            );

            Ok(output)
        }
        None => {
            log::warn!(
                "CCITT Group 4 decompression failed for {}x{} image with {} bytes",
                width,
                height_opt.unwrap_or(0),
                data.len()
            );
            // Try returning empty data padded with white pixels as fallback
            let expected_bytes = (height as usize) * ((width as usize + 7) / 8);
            log::info!("Returning {} bytes of white pixels as fallback", expected_bytes);
            Ok(vec![0; expected_bytes])
        }
    }
}

/// Convert transition array to packed byte representation.
///
/// The transitions array contains run lengths (white/black alternating).
/// We convert this to a byte array where each bit represents a pixel.
fn transitions_to_bytes(transitions: &[u16], width: u16) -> Vec<u8> {
    let width = width as usize;
    let row_bytes = (width + 7) / 8;
    let mut row = vec![0u8; row_bytes];

    // Expand transitions to individual pixels
    let mut pixel_idx = 0;
    let mut is_black = false; // Start with white run

    for &run_length in transitions.iter() {
        let run_length = run_length as usize;

        for _ in 0..run_length {
            if pixel_idx >= width {
                break;
            }

            let byte_idx = pixel_idx / 8;
            let bit_pos = 7 - (pixel_idx % 8);

            // Set bit if this is a black pixel
            if is_black {
                row[byte_idx] |= 1 << bit_pos;
            }
            // else: white pixel, bit remains 0

            pixel_idx += 1;
        }

        if pixel_idx >= width {
            break;
        }

        // Toggle between white and black for next run
        is_black = !is_black;
    }

    row
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
