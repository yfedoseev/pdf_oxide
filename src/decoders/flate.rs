//! FlateDecode (zlib/deflate) implementation.
//!
//! This is the most common PDF compression filter, used in ~90% of PDFs.
//! Uses the flate2 crate for zlib decompression.

use crate::decoders::StreamDecoder;
use crate::error::{Error, Result};
use flate2::read::{DeflateDecoder, ZlibDecoder};
use libflate::zlib::Decoder as LibflateDecoder;
use std::io::Read;

/// Default cap for [`FlateDecoder`]: 256 MB per stream.
///
/// Prevents zip-bomb / flate-bomb attacks where a tiny compressed stream
/// expands to an arbitrarily large output, exhausting virtual memory and
/// triggering an allocator abort (SIGABRT / exit 134).
///
/// 256 MB accommodates A4 @ 600 DPI RGB (~99 MB) with headroom.
///
/// Override via:
/// - `PDF_OXIDE_MAX_DECOMPRESS_MB` environment variable (e.g. `64` for 64 MB)
/// - [`FlateDecoder::with_limit`] for programmatic control
pub const DEFAULT_MAX_DECOMPRESSED_BYTES: u64 = 256 * 1024 * 1024;

/// Read the decompression limit from the environment, falling back to the
/// compile-time default.
fn effective_limit() -> u64 {
    std::env::var("PDF_OXIDE_MAX_DECOMPRESS_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(DEFAULT_MAX_DECOMPRESSED_BYTES)
}

/// Returns `Err` if `output` reached the decompression cap, indicating that the
/// stream was truncated rather than fully decoded.
#[inline]
fn check_limit(output: &[u8], limit: u64) -> Result<()> {
    if output.len() as u64 >= limit {
        return Err(Error::Decode(format!(
            "FlateDecode output reached the {} MB safety limit; \
             stream may be a flate bomb or an unusually large image",
            limit / (1024 * 1024)
        )));
    }
    Ok(())
}

/// FlateDecode filter implementation.
///
/// Decompresses data using the zlib/deflate algorithm. The decompression cap
/// defaults to [`DEFAULT_MAX_DECOMPRESSED_BYTES`] and can be overridden with
/// [`FlateDecoder::with_limit`].
pub struct FlateDecoder {
    /// Maximum number of decompressed bytes accepted per stream.
    pub max_decompressed_bytes: u64,
}

impl Default for FlateDecoder {
    fn default() -> Self {
        Self {
            max_decompressed_bytes: effective_limit(),
        }
    }
}

impl FlateDecoder {
    /// Creates a decoder that rejects any stream decompressing to more than
    /// `limit` bytes. Use this to tighten or relax the default 256 MB cap.
    pub fn with_limit(limit: u64) -> Self {
        Self {
            max_decompressed_bytes: limit,
        }
    }
}

impl StreamDecoder for FlateDecoder {
    fn decode(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = ZlibDecoder::new(input).take(self.max_decompressed_bytes);
        let mut output = Vec::new();

        // Try to read all data with standard zlib
        match decoder.read_to_end(&mut output) {
            Ok(_) => {
                check_limit(&output, self.max_decompressed_bytes)?;
                Ok(output)
            },
            Err(e) => {
                // Partial recovery: if we got ANY data before the error, use it
                if !output.is_empty() {
                    check_limit(&output, self.max_decompressed_bytes)?;
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
                let mut deflate_decoder =
                    DeflateDecoder::new(input).take(self.max_decompressed_bytes);

                match deflate_decoder.read_to_end(&mut output) {
                    Ok(_) => {
                        check_limit(&output, self.max_decompressed_bytes)?;
                        log::info!("Raw deflate recovery succeeded: {} bytes", output.len());
                        Ok(output)
                    },
                    Err(deflate_err) => {
                        if !output.is_empty() {
                            check_limit(&output, self.max_decompressed_bytes)?;
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
                            let mut deflate_decoder =
                                DeflateDecoder::new(&input[2..]).take(self.max_decompressed_bytes);

                            match deflate_decoder.read_to_end(&mut output) {
                                Ok(_) => {
                                    check_limit(&output, self.max_decompressed_bytes)?;
                                    log::info!(
                                        "Deflate with header skip succeeded: {} bytes",
                                        output.len()
                                    );
                                    return Ok(output);
                                },
                                Err(_) => {
                                    if !output.is_empty() {
                                        check_limit(&output, self.max_decompressed_bytes)?;
                                        log::warn!(
                                            "Deflate with header skip partial recovery: {} bytes",
                                            output.len()
                                        );
                                        return Ok(output);
                                    }
                                },
                            }
                        }

                        // Strategy 4: Try libflate (different implementation)
                        log::info!("Trying libflate crate");
                        output.clear();
                        match LibflateDecoder::new(input) {
                            Ok(mut libflate_decoder) => {
                                let mut limited =
                                    (&mut libflate_decoder).take(self.max_decompressed_bytes);
                                match limited.read_to_end(&mut output) {
                                    Ok(_) if !output.is_empty() => {
                                        check_limit(&output, self.max_decompressed_bytes)?;
                                        log::info!(
                                            "Libflate recovery succeeded: {} bytes",
                                            output.len()
                                        );
                                        return Ok(output);
                                    },
                                    Err(_) if !output.is_empty() => {
                                        check_limit(&output, self.max_decompressed_bytes)?;
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

                        // Strategy 5: Try fixing corrupt zlib header byte
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
                                let mut decoder = ZlibDecoder::new(&corrected[..])
                                    .take(self.max_decompressed_bytes);
                                match decoder.read_to_end(&mut output) {
                                    Ok(_) if !output.is_empty() => {
                                        check_limit(&output, self.max_decompressed_bytes)?;
                                        log::info!(
                                            "Header correction recovery succeeded: {} bytes",
                                            output.len()
                                        );
                                        return Ok(output);
                                    },
                                    Err(_) if !output.is_empty() => {
                                        check_limit(&output, self.max_decompressed_bytes)?;
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

                        // Strategy 6: Brute-force scan for valid deflate data
                        // Try starting deflate decompression from offsets 0-20
                        // BUT validate the output contains valid PDF operators
                        log::info!("Trying brute-force scan for valid deflate data");
                        let max_offset = std::cmp::min(20, input.len());
                        for offset in 0..max_offset {
                            if offset == 0 || offset == 2 {
                                continue; // Already tried these
                            }

                            output.clear();
                            let mut deflate_decoder = DeflateDecoder::new(&input[offset..])
                                .take(self.max_decompressed_bytes);

                            match deflate_decoder.read_to_end(&mut output) {
                                Ok(_) if !output.is_empty() => {
                                    check_limit(&output, self.max_decompressed_bytes)?;
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
                                    check_limit(&output, self.max_decompressed_bytes)?;
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
        let decoder = FlateDecoder::default();

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
        let decoder = FlateDecoder::default();

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
        let decoder = FlateDecoder::default();

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
        let decoder = FlateDecoder::default();

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
        let decoder = FlateDecoder::default();
        assert_eq!(decoder.name(), "FlateDecode");
    }

    #[test]
    fn test_flate_bomb_rejected() {
        // Verify that check_limit rejects output at or above the cap.
        let large = vec![0u8; DEFAULT_MAX_DECOMPRESSED_BYTES as usize];
        let result = check_limit(&large, DEFAULT_MAX_DECOMPRESSED_BYTES);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("safety limit"));
    }

    #[test]
    fn test_check_limit_below_threshold() {
        let small = vec![0u8; 1024];
        assert!(check_limit(&small, DEFAULT_MAX_DECOMPRESSED_BYTES).is_ok());
    }

    #[test]
    fn test_custom_limit_accepts_data_within_limit() {
        // A decoder with a small cap should accept data below that cap.
        let original = b"x".repeat(512);
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let decoder = FlateDecoder::with_limit(1024);
        let decoded = decoder.decode(&compressed).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_custom_limit_rejects_data_over_limit() {
        // A decoder with a tiny cap should reject data that exceeds it.
        let original = b"x".repeat(100);
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let decoder = FlateDecoder::with_limit(10);
        let result = decoder.decode(&compressed);
        assert!(result.is_err(), "expected rejection when output exceeds custom limit");
    }

    #[test]
    fn test_bomb_error_does_not_expose_internal_symbol_name() {
        // The user-facing error message must not reference internal symbol names.
        let large = vec![0u8; DEFAULT_MAX_DECOMPRESSED_BYTES as usize];
        let result = check_limit(&large, DEFAULT_MAX_DECOMPRESSED_BYTES);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            !msg.contains("MAX_DECOMPRESSED_BYTES"),
            "error message must not reference internal symbol names: {msg}"
        );
    }

    // Single test for env var to avoid parallel race conditions.
    // Tests all three cases sequentially in one function.
    #[test]
    fn test_effective_limit_env_variable() {
        // Default (no env var)
        std::env::remove_var("PDF_OXIDE_MAX_DECOMPRESS_MB");
        assert_eq!(effective_limit(), DEFAULT_MAX_DECOMPRESSED_BYTES);

        // Valid override
        unsafe { std::env::set_var("PDF_OXIDE_MAX_DECOMPRESS_MB", "64") };
        assert_eq!(effective_limit(), 64 * 1024 * 1024);

        // Invalid value falls back to default
        unsafe { std::env::set_var("PDF_OXIDE_MAX_DECOMPRESS_MB", "not_a_number") };
        assert_eq!(effective_limit(), DEFAULT_MAX_DECOMPRESSED_BYTES);

        // Cleanup
        unsafe { std::env::remove_var("PDF_OXIDE_MAX_DECOMPRESS_MB") };
    }
}
