//! Integration tests for stream decoders.
//!
//! Tests all decoders with various scenarios including:
//! - Individual decoder functionality
//! - Filter pipelines (multiple decoders chained)
//! - Edge cases and error handling
//! - Integration with Object::decode_stream_data()

use bytes::Bytes;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use pdf_oxide::decoders::{
    decode_stream, Ascii85Decoder, AsciiHexDecoder, DctDecoder, FlateDecoder, LzwDecoder,
    RunLengthDecoder, StreamDecoder,
};
use pdf_oxide::object::Object;
use std::collections::HashMap;
use std::io::Write;
use weezl::{encode::Encoder as LzwEncoder, BitOrder};

#[test]
fn test_flate_decoder_integration() {
    let decoder = FlateDecoder;

    // Create test data
    let original = b"This is a test of FlateDecode compression in a PDF stream.";

    // Compress it
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(original).unwrap();
    let compressed = encoder.finish().unwrap();

    // Decode it
    let decoded = decoder.decode(&compressed).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_ascii_hex_decoder_integration() {
    let decoder = AsciiHexDecoder;

    // Test various hex strings
    let test_cases = vec![
        (b"48656C6C6F20576F726C64".as_slice(), b"Hello World".as_slice()),
        (b"54657374".as_slice(), b"Test".as_slice()),
        (b"414243444546".as_slice(), b"ABCDEF".as_slice()),
    ];

    for (input, expected) in test_cases {
        let decoded = decoder.decode(input).unwrap();
        assert_eq!(decoded, expected);
    }
}

#[test]
fn test_ascii85_decoder_integration() {
    let decoder = Ascii85Decoder;

    // Test the special 'z' case
    let decoded = decoder.decode(b"z").unwrap();
    assert_eq!(decoded, b"\x00\x00\x00\x00");

    // Test regular encoding
    let decoded = decoder.decode(b"<+U,m").unwrap();
    assert_eq!(decoded, b"Test");
}

#[test]
fn test_lzw_decoder_integration() {
    let decoder = LzwDecoder;

    // Create test data with LZW
    let original = b"ABABABABABABABAB"; // Good for LZW compression
    let mut encoder = LzwEncoder::new(BitOrder::Msb, 8);
    let compressed = encoder.encode(original).unwrap();

    let decoded = decoder.decode(&compressed).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_runlength_decoder_integration() {
    let decoder = RunLengthDecoder;

    // Test literal run
    let input = vec![2, b'A', b'B', b'C']; // Copy 3 bytes
    let decoded = decoder.decode(&input).unwrap();
    assert_eq!(decoded, b"ABC");

    // Test repeat run
    let input = vec![250, b'X']; // Repeat 'X' 7 times (257-250=7)
    let decoded = decoder.decode(&input).unwrap();
    assert_eq!(decoded, b"XXXXXXX");
}

#[test]
fn test_dct_decoder_integration() {
    let decoder = DctDecoder;

    // DCT is pass-through
    let jpeg_data = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01";
    let decoded = decoder.decode(jpeg_data).unwrap();
    assert_eq!(decoded, jpeg_data);
}

#[test]
fn test_filter_pipeline_single() {
    // Test single filter in pipeline
    let data = b"48656C6C6F"; // "Hello" in hex
    let filters = vec!["ASCIIHexDecode".to_string()];

    let decoded = decode_stream(data, &filters).unwrap();
    assert_eq!(decoded, b"Hello");
}

#[test]
fn test_filter_pipeline_multiple() {
    // Test multiple filters: ASCIIHex then Flate
    // First, create flate-compressed data
    let original = b"Hello, World!";
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(original).unwrap();
    let compressed = encoder.finish().unwrap();

    // Convert to hex
    let hex_encoded: String = compressed.iter().map(|b| format!("{:02X}", b)).collect();

    // Now decode with both filters
    let filters = vec!["ASCIIHexDecode".to_string(), "FlateDecode".to_string()];
    let decoded = decode_stream(hex_encoded.as_bytes(), &filters).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_filter_pipeline_unsupported() {
    let data = b"test";
    let filters = vec!["NonExistentFilter".to_string()];

    let result = decode_stream(data, &filters);
    assert!(result.is_err());
}

#[test]
fn test_object_stream_decode_no_filter() {
    // Stream with no filter
    let mut dict = HashMap::new();
    dict.insert("Length".to_string(), Object::Integer(13));

    let stream = Object::Stream {
        dict,
        data: Bytes::from_static(b"Hello, World!"),
    };

    let decoded = stream.decode_stream_data().unwrap();
    assert_eq!(decoded, b"Hello, World!");
}

#[test]
fn test_object_stream_decode_with_flate() {
    // Stream with FlateDecode
    let original = b"This is compressed data in a PDF stream.";
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(original).unwrap();
    let compressed = encoder.finish().unwrap();

    let mut dict = HashMap::new();
    dict.insert("Length".to_string(), Object::Integer(compressed.len() as i64));
    dict.insert("Filter".to_string(), Object::Name("FlateDecode".to_string()));

    let stream = Object::Stream {
        dict,
        data: Bytes::from(compressed),
    };

    let decoded = stream.decode_stream_data().unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_object_stream_decode_with_multiple_filters() {
    // Stream with multiple filters as array
    let original = b"Test data";

    // Compress with flate
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(original).unwrap();
    let compressed = encoder.finish().unwrap();

    // Encode as hex
    let hex_encoded: String = compressed.iter().map(|b| format!("{:02X}", b)).collect();

    let mut dict = HashMap::new();
    dict.insert(
        "Filter".to_string(),
        Object::Array(vec![
            Object::Name("ASCIIHexDecode".to_string()),
            Object::Name("FlateDecode".to_string()),
        ]),
    );

    let stream = Object::Stream {
        dict,
        data: Bytes::from(hex_encoded.into_bytes()),
    };

    let decoded = stream.decode_stream_data().unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_object_stream_decode_error_not_stream() {
    // Try to decode a non-stream object
    let obj = Object::Integer(42);
    let result = obj.decode_stream_data();
    assert!(result.is_err());
}

#[test]
fn test_all_decoders_name() {
    // Verify all decoders have correct names
    assert_eq!(FlateDecoder.name(), "FlateDecode");
    assert_eq!(AsciiHexDecoder.name(), "ASCIIHexDecode");
    assert_eq!(Ascii85Decoder.name(), "ASCII85Decode");
    assert_eq!(LzwDecoder.name(), "LZWDecode");
    assert_eq!(RunLengthDecoder.name(), "RunLengthDecode");
    assert_eq!(DctDecoder.name(), "DCTDecode");
}

#[test]
fn test_decode_stream_empty_filters() {
    // No filters means data is returned as-is
    let data = b"No compression here!";
    let decoded = decode_stream(data, &[]).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn test_complex_filter_pipeline() {
    // Test a 3-filter pipeline
    let original = b"Complex!";

    // Step 1: Compress with LZW
    let mut lzw_encoder = LzwEncoder::new(BitOrder::Msb, 8);
    let lzw_compressed = lzw_encoder.encode(original).unwrap();

    // Step 2: Compress with Flate
    let mut flate_encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    flate_encoder.write_all(&lzw_compressed).unwrap();
    let double_compressed = flate_encoder.finish().unwrap();

    // Step 3: Encode as hex
    let hex_encoded: String = double_compressed
        .iter()
        .map(|b| format!("{:02X}", b))
        .collect();

    // Decode with all three filters in reverse order
    let filters = vec![
        "ASCIIHexDecode".to_string(),
        "FlateDecode".to_string(),
        "LZWDecode".to_string(),
    ];

    let decoded = decode_stream(hex_encoded.as_bytes(), &filters).unwrap();
    assert_eq!(decoded, original);
}
