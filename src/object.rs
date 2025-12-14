//! PDF object types.

use crate::error::{Error, Result};

/// PDF object representation.
#[derive(Debug, Clone, PartialEq)]
pub enum Object {
    /// Null object
    Null,
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Real (floating-point) value
    Real(f64),
    /// String (byte array)
    String(Vec<u8>),
    /// Name (starting with /)
    Name(String),
    /// Array of objects
    Array(Vec<Object>),
    /// Dictionary (key-value pairs)
    Dictionary(std::collections::HashMap<String, Object>),
    /// Stream (dictionary + data)
    Stream {
        /// Stream dictionary
        dict: std::collections::HashMap<String, Object>,
        /// Stream data
        data: bytes::Bytes,
    },
    /// Indirect object reference
    Reference(ObjectRef),
}

/// Reference to an indirect object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectRef {
    /// Object number
    pub id: u32,
    /// Generation number
    pub gen: u16,
}

impl ObjectRef {
    /// Create a new object reference.
    pub fn new(id: u32, gen: u16) -> Self {
        Self { id, gen }
    }
}

impl std::fmt::Display for ObjectRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} R", self.id, self.gen)
    }
}

impl Object {
    /// Get the type name of this object (without data).
    ///
    /// Returns a human-readable type name like "String", "Array", "Dictionary", etc.
    /// without including the actual data content.
    pub fn type_name(&self) -> &'static str {
        match self {
            Object::Null => "Null",
            Object::Boolean(_) => "Boolean",
            Object::Integer(_) => "Integer",
            Object::Real(_) => "Real",
            Object::String(_) => "String",
            Object::Name(_) => "Name",
            Object::Array(_) => "Array",
            Object::Dictionary(_) => "Dictionary",
            Object::Stream { .. } => "Stream",
            Object::Reference(_) => "Reference",
        }
    }

    /// Try to cast to integer.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Object::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to cast to name.
    pub fn as_name(&self) -> Option<&str> {
        match self {
            Object::Name(s) => Some(s),
            _ => None,
        }
    }

    /// Try to cast to dictionary. Works for both Dictionary and Stream objects.
    pub fn as_dict(&self) -> Option<&std::collections::HashMap<String, Object>> {
        match self {
            Object::Dictionary(d) => Some(d),
            Object::Stream { dict, .. } => Some(dict),
            _ => None,
        }
    }

    /// Try to cast to array.
    pub fn as_array(&self) -> Option<&Vec<Object>> {
        match self {
            Object::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Try to cast to reference.
    pub fn as_reference(&self) -> Option<ObjectRef> {
        match self {
            Object::Reference(r) => Some(*r),
            _ => None,
        }
    }

    /// Try to cast to boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Object::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to cast to real number.
    pub fn as_real(&self) -> Option<f64> {
        match self {
            Object::Real(r) => Some(*r),
            _ => None,
        }
    }

    /// Try to cast to string (bytes).
    pub fn as_string(&self) -> Option<&[u8]> {
        match self {
            Object::String(s) => Some(s),
            _ => None,
        }
    }

    /// Check if object is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Object::Null)
    }

    /// Decode stream data using filters specified in the stream dictionary.
    ///
    /// This is a convenience method that calls `decode_stream_data_with_decryption`
    /// with no encryption parameters.
    ///
    /// # Returns
    ///
    /// The decoded stream data, or an error if this is not a stream object
    /// or if decoding fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use pdf_oxide::object::Object;
    ///
    /// # fn example(stream_obj: Object) -> Result<(), Box<dyn std::error::Error>> {
    /// let decoded_data = stream_obj.decode_stream_data()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn decode_stream_data(&self) -> Result<Vec<u8>> {
        self.decode_stream_data_with_decryption(None, 0, 0)
    }

    /// Decode stream data with optional decryption.
    ///
    /// PDF Spec: Section 7.6.2 - General Encryption Algorithm states that streams
    /// must be decrypted BEFORE applying filters (decompression).
    ///
    /// # Arguments
    ///
    /// * `decryption_fn` - Optional decryption function (from EncryptionHandler)
    /// * `obj_num` - Object number (for encryption key derivation)
    /// * `gen_num` - Generation number (for encryption key derivation)
    ///
    /// # Returns
    ///
    /// The decoded stream data, or an error if decoding/decryption fails.
    pub fn decode_stream_data_with_decryption(
        &self,
        decryption_fn: Option<&dyn Fn(&[u8]) -> Result<Vec<u8>>>,
        obj_num: u32,
        gen_num: u32,
    ) -> Result<Vec<u8>> {
        match self {
            Object::Stream { dict, data } => {
                // Step 1: Decrypt stream data BEFORE applying filters
                // PDF Spec: Section 7.6.2 - Encryption must be applied before compression
                //
                // IMPORTANT: For encrypted streams, we must NOT trim whitespace before decryption
                // because encrypted data is binary and trimming might corrupt it (especially for AES
                // where the first 16 bytes are the IV). We only trim for unencrypted streams.
                let decrypted_data = if let Some(decrypt) = decryption_fn {
                    log::debug!(
                        "Decrypting stream for object {} {} (length: {} bytes)",
                        obj_num,
                        gen_num,
                        data.len()
                    );
                    // For encrypted streams, pass raw data without trimming
                    match decrypt(data) {
                        Ok(data) => {
                            log::debug!("Decryption successful: {} bytes", data.len());
                            data
                        },
                        Err(e) => {
                            log::error!(
                                "Decryption failed for object {} {}: {}",
                                obj_num,
                                gen_num,
                                e
                            );
                            return Err(e);
                        },
                    }
                } else {
                    // For unencrypted streams, trim leading whitespace
                    // Some malformed PDFs have extra whitespace after the "stream" keyword
                    // PDF Spec allows a single EOL marker, but some PDFs add more
                    let trimmed_data = trim_leading_stream_whitespace(data);
                    trimmed_data.to_vec()
                };

                // Step 2: Apply filters (decompression)
                // PDF Spec: ISO 32000-1:2008, Section 7.3.8.2 - Stream Objects
                let filters = dict
                    .get("Filter")
                    .map(extract_filter_names)
                    .unwrap_or_default();

                if filters.is_empty() {
                    // No filters, return decrypted data
                    Ok(decrypted_data)
                } else {
                    // Get decode parameters if present
                    // PDF Spec: ISO 32000-1:2008, Section 7.4.2 - DecodeParms
                    let decode_params = extract_decode_params(dict.get("DecodeParms"));

                    // Decode using filter pipeline with parameters
                    crate::decoders::decode_stream_with_params(
                        &decrypted_data,
                        &filters,
                        decode_params.as_ref(),
                    )
                }
            },
            _ => Err(Error::InvalidObjectType {
                expected: "Stream".to_string(),
                found: self.type_name().to_string(),
            }),
        }
    }
}

/// Trim leading PDF whitespace from stream data.
///
/// PDF Spec ISO 32000-1:2008, Section 7.3.4.2 states that stream data begins
/// immediately after the EOL marker following "stream". However, some PDF generators
/// add extra whitespace characters.
///
/// PDF whitespace characters (Section 7.2.2):
/// - NUL (0x00)
/// - TAB (0x09)
/// - LF (0x0A)
/// - FF (0x0C)
/// - CR (0x0D)
/// - SPACE (0x20)
fn trim_leading_stream_whitespace(data: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < data.len() {
        match data[start] {
            0x00 | 0x09 | 0x0A | 0x0C | 0x0D | 0x20 => start += 1,
            _ => break,
        }
    }
    &data[start..]
}

/// Extract filter names from a Filter object.
///
/// The Filter entry can be either:
/// - A single Name (e.g., /FlateDecode)
/// - An Array of Names (e.g., [/ASCII85Decode /FlateDecode])
fn extract_filter_names(filter_obj: &Object) -> Vec<String> {
    match filter_obj {
        Object::Name(name) => vec![name.clone()],
        Object::Array(arr) => arr
            .iter()
            .filter_map(|obj| obj.as_name().map(|s| s.to_string()))
            .collect(),
        _ => vec![],
    }
}

/// Extract decode parameters from a DecodeParms object.
///
/// PDF Spec: ISO 32000-1:2008, Section 7.4.2 - LZWDecode and FlateDecode Parameters
///
/// The DecodeParms entry can be:
/// - A dictionary (for single filter)
/// - An array of dictionaries (for multiple filters)
/// - Null or absent (no parameters)
///
/// This function extracts predictor parameters used for PNG/TIFF encoding.
fn extract_decode_params(params_obj: Option<&Object>) -> Option<crate::decoders::DecodeParams> {
    let dict = match params_obj? {
        Object::Dictionary(d) => d,
        Object::Array(arr) => {
            // For array, take the first non-null dictionary
            arr.iter().filter_map(|obj| obj.as_dict()).next()?
        },
        _ => return None,
    };

    // Extract predictor parameters per PDF Spec Table 3.7
    let predictor = dict
        .get("Predictor")
        .and_then(|obj| obj.as_integer())
        .unwrap_or(1); // Default: no prediction

    let columns = dict
        .get("Columns")
        .and_then(|obj| obj.as_integer())
        .unwrap_or(1) as usize;

    let colors = dict
        .get("Colors")
        .and_then(|obj| obj.as_integer())
        .unwrap_or(1) as usize;

    let bits_per_component = dict
        .get("BitsPerComponent")
        .and_then(|obj| obj.as_integer())
        .unwrap_or(8) as usize;

    Some(crate::decoders::DecodeParams {
        predictor,
        columns,
        colors,
        bits_per_component,
    })
}

/// Extract CCITT-specific decode parameters from a DecodeParms object.
///
/// PDF Spec: ISO 32000-1:2008, Section 7.4.6 - CCITTFaxDecode Filter Parameters
///
/// The DecodeParms entry can be:
/// - A dictionary (for single filter)
/// - An array of dictionaries (for multiple filters)
/// - Null or absent (no parameters, use defaults)
///
/// CCITT parameters:
/// - /K: Group indicator (-1=Group 4, 0=Group 3 1-D, >0=Group 3 2-D)
/// - /Columns: Image width in pixels
/// - /Rows: Image height in pixels (optional)
/// - /BlackIs1: Pixel interpretation (false=white is 0, true=white is 1)
/// - /EndOfLine: Include EOL code (default false)
/// - /EncodedByteAlign: Byte-aligned encoding (default false)
/// - /EndOfBlock: Include RTC code (default true)
pub fn extract_ccitt_params(params_obj: Option<&Object>) -> Option<crate::decoders::CcittParams> {
    extract_ccitt_params_with_width(params_obj, None)
}

/// Extract CCITT decompression parameters from a PDF object with optional width override.
///
/// This function extracts CCITT Group 3 or Group 4 decompression parameters from a PDF
/// /DecodeParms dictionary. If image_width is provided, it will be used as the /Columns
/// parameter, overriding any value in the dictionary.
///
/// # Arguments
/// * `params_obj` - Optional PDF object containing CCITT parameters (Dictionary or Array)
/// * `image_width` - Optional width override to use as /Columns parameter
///
/// # Returns
/// Some(CcittParams) if valid parameters are found, None otherwise
pub fn extract_ccitt_params_with_width(
    params_obj: Option<&Object>,
    image_width: Option<u32>,
) -> Option<crate::decoders::CcittParams> {
    let dict = match params_obj? {
        Object::Dictionary(d) => d,
        Object::Array(arr) => {
            // For array, take the first non-null dictionary
            arr.iter().filter_map(|obj| obj.as_dict()).next()?
        },
        _ => return None,
    };

    // Extract CCITT parameters with PDF defaults
    let k = dict.get("K").and_then(|obj| obj.as_integer()).unwrap_or(-1); // Default: Group 4

    let columns = dict
        .get("Columns")
        .and_then(|obj| obj.as_integer())
        .map(|v| v as u32)
        .or(image_width)
        .unwrap_or(1);

    let rows = dict
        .get("Rows")
        .and_then(|obj| obj.as_integer())
        .map(|v| v as u32);

    let black_is_1 = dict
        .get("BlackIs1")
        .and_then(|obj| obj.as_bool())
        .unwrap_or(false); // PDF default: white=0, black=1

    let end_of_line = dict
        .get("EndOfLine")
        .and_then(|obj| obj.as_bool())
        .unwrap_or(false); // PDF default: no EOL

    let encoded_byte_align = dict
        .get("EncodedByteAlign")
        .and_then(|obj| obj.as_bool())
        .unwrap_or(false); // PDF default: no alignment

    let end_of_block = dict
        .get("EndOfBlock")
        .and_then(|obj| obj.as_bool())
        .unwrap_or(true); // PDF default: RTC code present

    Some(crate::decoders::CcittParams {
        k,
        columns,
        rows,
        black_is_1,
        end_of_line,
        encoded_byte_align,
        end_of_block,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_object_integer() {
        let obj = Object::Integer(42);
        assert_eq!(obj.as_integer(), Some(42));
        assert!(obj.as_name().is_none());
        assert!(!obj.is_null());
    }

    #[test]
    fn test_object_name() {
        let obj = Object::Name("Type".to_string());
        assert_eq!(obj.as_name(), Some("Type"));
        assert!(obj.as_integer().is_none());
    }

    #[test]
    fn test_object_bool() {
        let obj = Object::Boolean(true);
        assert_eq!(obj.as_bool(), Some(true));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_object_real() {
        let obj = Object::Real(3.14);
        assert_eq!(obj.as_real(), Some(3.14));
    }

    #[test]
    fn test_object_string() {
        let obj = Object::String(b"Hello".to_vec());
        assert_eq!(obj.as_string(), Some(&b"Hello"[..]));
    }

    #[test]
    fn test_object_null() {
        let obj = Object::Null;
        assert!(obj.is_null());
        assert!(obj.as_integer().is_none());
    }

    #[test]
    fn test_object_array() {
        let obj = Object::Array(vec![Object::Integer(1), Object::Integer(2)]);
        let arr = obj.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].as_integer(), Some(1));
    }

    #[test]
    fn test_object_dictionary() {
        let mut dict = HashMap::new();
        dict.insert("Type".to_string(), Object::Name("Page".to_string()));
        let obj = Object::Dictionary(dict);

        let d = obj.as_dict().unwrap();
        assert_eq!(d.get("Type").unwrap().as_name(), Some("Page"));
    }

    #[test]
    fn test_object_stream_dict_access() {
        let mut dict = HashMap::new();
        dict.insert("Length".to_string(), Object::Integer(100));
        let obj = Object::Stream {
            dict,
            data: bytes::Bytes::from_static(b"stream data"),
        };

        // Stream objects should also be accessible as dictionaries
        let d = obj.as_dict().unwrap();
        assert_eq!(d.get("Length").unwrap().as_integer(), Some(100));
    }

    #[test]
    fn test_object_reference() {
        let obj_ref = ObjectRef::new(10, 0);
        let obj = Object::Reference(obj_ref);

        assert_eq!(obj.as_reference(), Some(obj_ref));
        assert_eq!(obj_ref.id, 10);
        assert_eq!(obj_ref.gen, 0);
    }

    #[test]
    fn test_object_ref_display() {
        let obj_ref = ObjectRef::new(10, 0);
        assert_eq!(format!("{}", obj_ref), "10 0 R");
    }

    #[test]
    fn test_object_clone() {
        let obj = Object::Integer(42);
        let cloned = obj.clone();
        assert_eq!(obj, cloned);
    }

    #[test]
    fn test_object_ref_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ObjectRef::new(1, 0));
        set.insert(ObjectRef::new(2, 0));
        set.insert(ObjectRef::new(1, 0)); // Duplicate

        assert_eq!(set.len(), 2); // Should only have 2 unique refs
    }

    #[test]
    fn test_decode_stream_no_filter() {
        let mut dict = HashMap::new();
        dict.insert("Length".to_string(), Object::Integer(5));
        let obj = Object::Stream {
            dict,
            data: bytes::Bytes::from_static(b"Hello"),
        };

        let decoded = obj.decode_stream_data().unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_decode_stream_single_filter() {
        let mut dict = HashMap::new();
        dict.insert("Filter".to_string(), Object::Name("ASCIIHexDecode".to_string()));
        let obj = Object::Stream {
            dict,
            data: bytes::Bytes::from_static(b"48656C6C6F"), // "Hello" in hex
        };

        let decoded = obj.decode_stream_data().unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_decode_stream_filter_array() {
        let mut dict = HashMap::new();
        // Note: filters are applied in order - first ASCII85, then what it produces
        dict.insert(
            "Filter".to_string(),
            Object::Array(vec![Object::Name("ASCIIHexDecode".to_string())]),
        );
        let obj = Object::Stream {
            dict,
            data: bytes::Bytes::from_static(b"48656C6C6F"),
        };

        let decoded = obj.decode_stream_data().unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_decode_stream_not_a_stream() {
        let obj = Object::Integer(42);
        let result = obj.decode_stream_data();
        assert!(result.is_err());
        match result {
            Err(Error::InvalidObjectType { expected, found }) => {
                assert_eq!(expected, "Stream");
                assert_eq!(found, "Integer");
            },
            _ => panic!("Expected InvalidObjectType error"),
        }
    }

    #[test]
    fn test_extract_filter_names_single() {
        let filter = Object::Name("FlateDecode".to_string());
        let names = extract_filter_names(&filter);
        assert_eq!(names, vec!["FlateDecode"]);
    }

    #[test]
    fn test_extract_filter_names_array() {
        let filter = Object::Array(vec![
            Object::Name("ASCII85Decode".to_string()),
            Object::Name("FlateDecode".to_string()),
        ]);
        let names = extract_filter_names(&filter);
        assert_eq!(names, vec!["ASCII85Decode", "FlateDecode"]);
    }

    #[test]
    fn test_extract_filter_names_invalid() {
        let filter = Object::Integer(42);
        let names = extract_filter_names(&filter);
        assert!(names.is_empty());
    }
}
