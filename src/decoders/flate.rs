//! FlateDecode (zlib/deflate) implementation.
//!
//! This is the most common PDF compression filter, used in ~90% of PDFs.
//! Uses the flate2 crate for zlib decompression.

use crate::decoders::StreamDecoder;
use crate::error::{Error, Result};
use flate2::read::{DeflateDecoder, ZlibDecoder};
use inflate::inflate_bytes_zlib;
use libflate::zlib::Decoder as LibflateDecoder;
use std::io::Read;

/// FlateDecode filter implementation.
///
/// Decompresses data using the zlib/deflate algorithm.
pub struct FlateDecoder;

impl StreamDecoder for FlateDecoder {
    fn decode(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = ZlibDecoder::new(input);
        let mut output = Vec::new();

        // Try to read all data with standard zlib
        match decoder.read_to_end(&mut output) {
            Ok(_) => Ok(output),
            Err(e) => {
                // Partial recovery: if we got ANY data before the error, use it
                if !output.is_empty() {
                    log::warn!(
                        "FlateDecode partial recovery: extracted {} bytes before corruption: {}",
                        output.len(),
                        e
                    );
                    return Ok(output);
                }

                // Strategy 2: Try raw deflate (no zlib wrapper)
                // Some PDFs have corrupt zlib headers but valid deflate data
                log::info!("Zlib decode failed, trying raw deflate");
                output.clear();
                let mut deflate_decoder = DeflateDecoder::new(input);

                match deflate_decoder.read_to_end(&mut output) {
                    Ok(_) => {
                        log::info!("Raw deflate recovery succeeded: {} bytes", output.len());
                        Ok(output)
                    },
                    Err(deflate_err) => {
                        if !output.is_empty() {
                            log::warn!(
                                "Raw deflate partial recovery: extracted {} bytes before error",
                                output.len()
                            );
                            return Ok(output);
                        }

                        // Strategy 3: Try skipping zlib header (2 bytes) and reading deflate
                        if input.len() > 2 {
                            log::info!(
                                "Trying deflate after skipping potential corrupt zlib header"
                            );
                            output.clear();
                            let mut deflate_decoder = DeflateDecoder::new(&input[2..]);

                            match deflate_decoder.read_to_end(&mut output) {
                                Ok(_) => {
                                    log::info!(
                                        "Deflate with header skip succeeded: {} bytes",
                                        output.len()
                                    );
                                    return Ok(output);
                                },
                                Err(_) => {
                                    if !output.is_empty() {
                                        log::warn!(
                                            "Deflate with header skip partial recovery: {} bytes",
                                            output.len()
                                        );
                                        return Ok(output);
                                    }
                                },
                            }
                        }

                        // Strategy 4: Try inflate crate (better error recovery)
                        log::info!("Trying inflate crate with zlib");
                        match inflate_bytes_zlib(input) {
                            Ok(data) => {
                                log::info!(
                                    "Inflate crate recovery succeeded: {} bytes",
                                    data.len()
                                );
                                return Ok(data);
                            },
                            Err(inflate_err) => {
                                log::info!("Inflate crate failed: {:?}", inflate_err);
                            },
                        }

                        // Strategy 5: Try libflate (different implementation)
                        log::info!("Trying libflate crate");
                        output.clear();
                        match LibflateDecoder::new(input) {
                            Ok(mut libflate_decoder) => {
                                match libflate_decoder.read_to_end(&mut output) {
                                    Ok(_) if !output.is_empty() => {
                                        log::info!(
                                            "Libflate recovery succeeded: {} bytes",
                                            output.len()
                                        );
                                        return Ok(output);
                                    },
                                    Err(_) if !output.is_empty() => {
                                        log::warn!(
                                            "Libflate partial recovery: {} bytes",
                                            output.len()
                                        );
                                        return Ok(output);
                                    },
                                    _ => {
                                        log::info!("Libflate read failed");
                                    },
                                }
                            },
                            Err(e) => {
                                log::info!("Libflate init failed: {}", e);
                            },
                        }

                        // Strategy 6: Try fixing corrupt zlib header byte
                        // If first byte has invalid compression method, replace with 0x78 (standard deflate)
                        if input.len() >= 2 {
                            let first_byte = input[0];
                            let compression_method = first_byte & 0x0F;
                            if compression_method != 8 {
                                log::info!(
                                    "Detected invalid compression method {} in header byte 0x{:02x}, trying with corrected header",
                                    compression_method,
                                    first_byte
                                );
                                // Create new buffer with corrected header
                                let mut corrected = input.to_vec();
                                // Replace CM bits (0-3) with 8 (deflate), keep CINFO bits (4-7)
                                corrected[0] = (first_byte & 0xF0) | 0x08;

                                output.clear();
                                let mut decoder = ZlibDecoder::new(&corrected[..]);
                                match decoder.read_to_end(&mut output) {
                                    Ok(_) if !output.is_empty() => {
                                        log::info!(
                                            "Header correction recovery succeeded: {} bytes",
                                            output.len()
                                        );
                                        return Ok(output);
                                    },
                                    Err(_) if !output.is_empty() => {
                                        log::warn!(
                                            "Header correction partial recovery: {} bytes",
                                            output.len()
                                        );
                                        return Ok(output);
                                    },
                                    _ => {
                                        log::info!("Header correction failed");
                                    },
                                }
                            }
                        }

                        // Strategy 7: Brute-force scan for valid deflate data
                        // Try starting deflate decompression from offsets 0-20
                        // BUT validate the output contains valid PDF operators
                        log::info!("Trying brute-force scan for valid deflate data");
                        let max_offset = std::cmp::min(20, input.len());
                        for offset in 0..max_offset {
                            if offset == 0 || offset == 2 {
                                continue; // Already tried these
                            }

                            output.clear();
                            let mut deflate_decoder = DeflateDecoder::new(&input[offset..]);

                            match deflate_decoder.read_to_end(&mut output) {
                                Ok(_) if !output.is_empty() => {
                                    // Validate output quality - check for PDF operators
                                    let decoded_str = String::from_utf8_lossy(&output);
                                    let has_pdf_operators = decoded_str.contains("BT")
                                        || decoded_str.contains("ET")
                                        || decoded_str.contains("Tj")
                                        || decoded_str.contains("TJ")
                                        || decoded_str.contains("Tm")
                                        || decoded_str.contains("Td");

                                    if has_pdf_operators {
                                        log::info!(
                                            "Brute-force deflate recovery succeeded at offset {}: {} bytes (validated PDF content)",
                                            offset,
                                            output.len()
                                        );
                                        return Ok(output);
                                    } else {
                                        log::info!(
                                            "Brute-force at offset {} produced {} bytes but no valid PDF operators - trying next offset",
                                            offset,
                                            output.len()
                                        );
                                        continue;
                                    }
                                },
                                Err(_) if !output.is_empty() => {
                                    // Validate partial recovery too
                                    let decoded_str = String::from_utf8_lossy(&output);
                                    let has_pdf_operators = decoded_str.contains("BT")
                                        || decoded_str.contains("ET")
                                        || decoded_str.contains("Tj")
                                        || decoded_str.contains("TJ")
                                        || decoded_str.contains("Tm")
                                        || decoded_str.contains("Td");

                                    if has_pdf_operators {
                                        log::warn!(
                                            "Brute-force partial recovery at offset {}: {} bytes (validated PDF content)",
                                            offset,
                                            output.len()
                                        );
                                        return Ok(output);
                                    } else {
                                        log::info!(
                                            "Partial recovery at offset {} but no valid PDF operators - trying next offset",
                                            offset
                                        );
                                        continue;
                                    }
                                },
                                _ => continue,
                            }
                        }

                        // SPEC COMPLIANCE FIX: Removed strategies 8-9 that violated PDF spec
                        //
                        // Previous strategies 8-9 would return raw uncompressed data for streams
                        // labeled as /FlateDecode. This violates PDF Spec ISO 32000-1:2008,
                        // Section 7.3.8.2 which states that if a stream has /Filter /FlateDecode,
                        // it MUST be compressed with the FlateDecode algorithm.
                        //
                        // Returning raw data creates security risks:
                        // 1. Malicious PDFs could bypass compression validation
                        // 2. Type confusion attacks (treating compressed data as raw)
                        // 3. Inconsistent behavior across PDF processors
                        //
                        // Correct behavior: If all decompression strategies fail, return an error.
                        // The stream is either corrupted or malicious, and should not be processed.

                        log::error!(
                            "All FlateDecode recovery strategies failed. Zlib: {}, Deflate: {}",
                            e,
                            deflate_err
                        );
                        log::error!(
                            "Stream labeled as FlateDecode but cannot be decompressed - this violates PDF spec"
                        );

                        Err(Error::Decode(format!(
                            "FlateDecode decompression failed: stream is labeled as compressed but all decompression attempts failed. \
                            This violates PDF Spec ISO 32000-1:2008, Section 7.3.8.2. \
                            Zlib error: {}, Deflate error: {}. Compressed size: {} bytes.",
                            e,
                            deflate_err,
                            input.len()
                        )))
                    },
                }
            },
        }
    }

    fn name(&self) -> &str {
        "FlateDecode"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_flate_decode_simple() {
        let decoder = FlateDecoder;

        // Compress some data
        let original = b"Hello, FlateDecode!";
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Decompress
        let decoded = decoder.decode(&compressed).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_flate_decode_empty() {
        let decoder = FlateDecoder;

        // Compress empty data
        let original = b"";
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let decoded = decoder.decode(&compressed).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_flate_decode_large_data() {
        let decoder = FlateDecoder;

        // Create large repeated data
        let original = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ".repeat(1000);
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let decoded = decoder.decode(&compressed).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_flate_decode_invalid_data() {
        let decoder = FlateDecoder;

        // Invalid zlib data - should fail decompression
        // SPEC COMPLIANCE: We now correctly reject invalid compressed data
        // instead of returning it as raw data (which violated PDF spec)
        let invalid = b"This is not zlib compressed data";
        let result = decoder.decode(invalid);
        assert!(result.is_err());

        // Verify error message mentions spec compliance
        if let Err(e) = result {
            let error_msg = format!("{}", e);
            assert!(error_msg.contains("FlateDecode decompression failed"));
        }
    }

    #[test]
    fn test_flate_decoder_name() {
        let decoder = FlateDecoder;
        assert_eq!(decoder.name(), "FlateDecode");
    }
}
