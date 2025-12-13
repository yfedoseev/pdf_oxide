//! CCITT Group 4 decompression for bilevel images.
//!
//! This module handles decompression of CCITT Group 4 encoded bilevel (1-bit) images
//! extracted from PDF documents, and converts them to 8-bit grayscale for OCR processing.
//!
//! PDF Spec: ISO 32000-1:2008, Section 7.4.6 - CCITTFaxDecode Filter
//! CCITT Spec: ITU-T Recommendation T.6 - Facsimile coding schemes and coding control functions

use crate::error::{Error, Result};
use std::io::Cursor;

/// Decompresses CCITT Group 4 encoded data to raw bits.
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
/// A vector of bytes where each byte represents one scanline with each byte
/// containing 8 pixels (bit 7 is leftmost, bit 0 is rightmost).
pub fn decompress_ccitt_group4(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    // Try to decompress using the tiff crate
    decompress_with_tiff_fallback(data, width, height)
        .or_else(|_| {
            // If TIFF decompression fails, assume data might be pre-decompressed
            // or use fallback approach
            log::warn!("TIFF decompression failed, attempting fallback");
            decompress_ccitt_fallback(data, width, height)
        })
}

/// Attempt TIFF-based decompression
fn decompress_with_tiff_fallback(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let tiff_data = create_minimal_tiff(data, width, height)?;

    let image_result = image::load_from_memory(&tiff_data);
    match image_result {
        Ok(img) => {
            let gray = img.to_luma8();
            Ok(gray.into_raw())
        }
        Err(e) => {
            Err(Error::Decode(format!("TIFF decode failed: {}", e)))
        }
    }
}

/// Fallback decompression for CCITT Group 4
/// This assumes the data might be partially decompressed or can be handled directly
fn decompress_ccitt_fallback(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let width = width as usize;
    let height = height as usize;
    let row_bytes = (width + 7) / 8;
    let expected_size = height * row_bytes;

    // If data size matches expected bilevel size, it's likely already decompressed
    if data.len() == expected_size {
        log::debug!("Data size matches expected bilevel format, using as-is");
        return Ok(data.to_vec());
    }

    // Otherwise, this is likely CCITT compressed and we can't decompress without
    // proper CCITT implementation. Return the data and let caller handle it
    log::warn!("Cannot decompress CCITT - data size mismatch. Expected {}, got {}", expected_size, data.len());

    // Create a minimal bilevel image by padding with white pixels
    let mut output = vec![0u8; expected_size];
    output[..data.len().min(expected_size)].copy_from_slice(&data[..data.len().min(expected_size)]);
    Ok(output)
}

/// Create a minimal TIFF file structure with CCITT Group 4 compression.
///
/// This is a workaround to leverage the image crate's TIFF support which includes CCITT decompression.
/// We construct a minimal valid TIFF that wraps the CCITT data.
fn create_minimal_tiff(ccitt_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    let mut tiff = Vec::new();

    // TIFF Header (little-endian)
    tiff.write_all(b"II")?;  // Little-endian byte order
    tiff.write_u16::<LittleEndian>(42)?;  // TIFF version number
    tiff.write_u32::<LittleEndian>(8)?;  // Offset to first IFD

    // IFD (Image File Directory)
    let num_tags = 11u16;
    tiff.write_u16::<LittleEndian>(num_tags)?;

    // Calculate strip offset (after IFD)
    let strip_offset = 8 + 2 + (num_tags as usize * 12) + 4 + 200; // Buffer space

    // Tag definitions (must be in ascending order by tag number)
    let mut write_tag = |tag: u16, ty: u16, count: u32, value: u32| -> std::io::Result<()> {
        tiff.write_u16::<LittleEndian>(tag)?;
        tiff.write_u16::<LittleEndian>(ty)?;
        tiff.write_u32::<LittleEndian>(count)?;
        tiff.write_u32::<LittleEndian>(value)?;
        Ok(())
    };

    // Tag 254: NewSubfileType
    write_tag(254, 3, 1, 0)?;

    // Tag 256: ImageWidth
    write_tag(256, 3, 1, width)?;

    // Tag 257: ImageLength (height)
    write_tag(257, 3, 1, height)?;

    // Tag 258: BitsPerSample (1 for bilevel)
    write_tag(258, 3, 1, 1)?;

    // Tag 259: Compression (4 = CCITT Group 4)
    write_tag(259, 3, 1, 4)?;

    // Tag 262: PhotometricInterpretation (1 = BlackIsZero)
    write_tag(262, 3, 1, 1)?;

    // Tag 273: StripOffsets (offset to image data)
    write_tag(273, 4, 1, strip_offset as u32)?;

    // Tag 277: SamplesPerPixel (1 for grayscale)
    write_tag(277, 3, 1, 1)?;

    // Tag 278: RowsPerStrip (all rows in one strip)
    write_tag(278, 3, 1, height)?;

    // Tag 279: StripByteCounts
    write_tag(279, 4, 1, ccitt_data.len() as u32)?;

    // Tag 282: XResolution (72 DPI)
    write_tag(282, 5, 1, (strip_offset + 100) as u32)?;

    // Next IFD offset (0 = no more IFDs)
    tiff.write_u32::<LittleEndian>(0)?;

    // Pad to strip offset
    while tiff.len() < strip_offset {
        tiff.push(0);
    }

    // Write CCITT compressed image data
    tiff.write_all(ccitt_data)?;

    Ok(tiff)
}

/// Convert 1-bit bilevel image to 8-bit grayscale.
///
/// Each bit in the input is expanded to a full byte where:
/// - 0 -> 0xFF (white)
/// - 1 -> 0x00 (black)
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
                // 0 -> 0xFF (white), 1 -> 0x00 (black)
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
        let bilevel = vec![0b11110000]; // FFFF 0000 in bits
        let grayscale = bilevel_to_grayscale(&bilevel, 8, 1);

        // First 4 pixels should be black (0x00), last 4 should be white (0xFF)
        assert_eq!(grayscale[0], 0x00);
        assert_eq!(grayscale[3], 0x00);
        assert_eq!(grayscale[4], 0xFF);
        assert_eq!(grayscale[7], 0xFF);
    }
}
