//! PDF object serialization.
//!
//! Serializes PDF objects to their byte representation according to
//! PDF specification ISO 32000-1:2008.

use crate::object::{Object, ObjectRef};
use std::collections::HashMap;
use std::io::Write;

/// Serializer for PDF objects.
///
/// Converts PDF Object types to their byte representation following
/// the PDF specification syntax rules.
#[derive(Debug, Clone, Default)]
pub struct ObjectSerializer {
    /// Whether to use compact formatting (minimal whitespace)
    compact: bool,
    /// Current indentation level for pretty printing
    indent_level: usize,
}

impl ObjectSerializer {
    /// Create a new object serializer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a compact serializer (minimal whitespace).
    pub fn compact() -> Self {
        Self {
            compact: true,
            indent_level: 0,
        }
    }

    /// Serialize an object to bytes.
    pub fn serialize(&self, obj: &Object) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_object(&mut buf, obj).unwrap();
        buf
    }

    /// Serialize an object to a string (for debugging).
    pub fn serialize_to_string(&self, obj: &Object) -> String {
        String::from_utf8_lossy(&self.serialize(obj)).to_string()
    }

    /// Serialize an indirect object definition.
    ///
    /// Format: `{id} {gen} obj\n{object}\nendobj\n`
    pub fn serialize_indirect(&self, id: u32, gen: u16, obj: &Object) -> Vec<u8> {
        let mut buf = Vec::new();
        write!(buf, "{} {} obj\n", id, gen).unwrap();
        self.write_object(&mut buf, obj).unwrap();
        write!(buf, "\nendobj\n").unwrap();
        buf
    }

    /// Write an object to a buffer.
    fn write_object<W: Write>(&self, w: &mut W, obj: &Object) -> std::io::Result<()> {
        match obj {
            Object::Null => write!(w, "null"),
            Object::Boolean(b) => write!(w, "{}", if *b { "true" } else { "false" }),
            Object::Integer(i) => write!(w, "{}", i),
            Object::Real(r) => self.write_real(w, *r),
            Object::String(s) => self.write_string(w, s),
            Object::Name(n) => self.write_name(w, n),
            Object::Array(arr) => self.write_array(w, arr),
            Object::Dictionary(dict) => self.write_dictionary(w, dict),
            Object::Stream { dict, data } => self.write_stream(w, dict, data),
            Object::Reference(r) => write!(w, "{} {} R", r.id, r.gen),
        }
    }

    /// Write a real number with appropriate precision.
    fn write_real<W: Write>(&self, w: &mut W, value: f64) -> std::io::Result<()> {
        // PDF spec allows up to 5 decimal places for coordinates
        // Remove trailing zeros for compact output
        if value.fract() == 0.0 {
            write!(w, "{}", value as i64)
        } else {
            // Format with enough precision, then trim trailing zeros
            let formatted = format!("{:.5}", value);
            let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
            write!(w, "{}", trimmed)
        }
    }

    /// Write a PDF string.
    ///
    /// Uses literal string syntax `(...)` with proper escaping,
    /// or hex string syntax `<...>` for binary data.
    fn write_string<W: Write>(&self, w: &mut W, data: &[u8]) -> std::io::Result<()> {
        // Check if data is printable ASCII
        let is_printable = data
            .iter()
            .all(|&b| b == b'\n' || b == b'\r' || b == b'\t' || (b >= 0x20 && b <= 0x7E));

        if is_printable {
            // Use literal string
            write!(w, "(")?;
            for &byte in data {
                match byte {
                    b'(' => write!(w, "\\(")?,
                    b')' => write!(w, "\\)")?,
                    b'\\' => write!(w, "\\\\")?,
                    b'\n' => write!(w, "\\n")?,
                    b'\r' => write!(w, "\\r")?,
                    b'\t' => write!(w, "\\t")?,
                    _ => w.write_all(&[byte])?,
                }
            }
            write!(w, ")")
        } else {
            // Use hex string
            write!(w, "<")?;
            for byte in data {
                write!(w, "{:02X}", byte)?;
            }
            write!(w, ">")
        }
    }

    /// Write a PDF name.
    ///
    /// Names start with `/` and escape special characters with `#xx`.
    fn write_name<W: Write>(&self, w: &mut W, name: &str) -> std::io::Result<()> {
        write!(w, "/")?;
        for byte in name.bytes() {
            match byte {
                // Regular characters (no escaping needed)
                b'!'
                | b'"'
                | b'$'..=b'&'
                | b'\''..=b'.'
                | b'0'..=b'9'
                | b';'
                | b'<'
                | b'>'
                | b'?'
                | b'@'
                | b'A'..=b'Z'
                | b'^'..=b'z'
                | b'|'
                | b'~' => {
                    w.write_all(&[byte])?;
                },
                // Characters that need escaping
                _ => {
                    write!(w, "#{:02X}", byte)?;
                },
            }
        }
        Ok(())
    }

    /// Write a PDF array.
    fn write_array<W: Write>(&self, w: &mut W, arr: &[Object]) -> std::io::Result<()> {
        write!(w, "[")?;
        for (i, obj) in arr.iter().enumerate() {
            if i > 0 {
                write!(w, " ")?;
            }
            self.write_object(w, obj)?;
        }
        write!(w, "]")
    }

    /// Write a PDF dictionary.
    fn write_dictionary<W: Write>(
        &self,
        w: &mut W,
        dict: &HashMap<String, Object>,
    ) -> std::io::Result<()> {
        write!(w, "<<")?;

        // Sort keys for deterministic output
        let mut keys: Vec<_> = dict.keys().collect();
        keys.sort();

        for key in keys {
            if let Some(value) = dict.get(key) {
                if !self.compact {
                    write!(w, "\n  ")?;
                }
                self.write_name(w, key)?;
                write!(w, " ")?;
                self.write_object(w, value)?;
            }
        }

        if !self.compact && !dict.is_empty() {
            write!(w, "\n")?;
        }
        write!(w, ">>")
    }

    /// Write a PDF stream.
    fn write_stream<W: Write>(
        &self,
        w: &mut W,
        dict: &HashMap<String, Object>,
        data: &[u8],
    ) -> std::io::Result<()> {
        // Add Length to dictionary if not present
        let mut dict_with_length = dict.clone();
        if !dict_with_length.contains_key("Length") {
            dict_with_length.insert("Length".to_string(), Object::Integer(data.len() as i64));
        }

        self.write_dictionary(w, &dict_with_length)?;
        write!(w, "\nstream\n")?;
        w.write_all(data)?;
        write!(w, "\nendstream")
    }
}

/// Helper functions for building PDF objects.
impl ObjectSerializer {
    /// Create a Name object.
    pub fn name(s: &str) -> Object {
        Object::Name(s.to_string())
    }

    /// Create a String object from a Rust string.
    pub fn string(s: &str) -> Object {
        Object::String(s.as_bytes().to_vec())
    }

    /// Create an Integer object.
    pub fn integer(i: i64) -> Object {
        Object::Integer(i)
    }

    /// Create a Real object.
    pub fn real(r: f64) -> Object {
        Object::Real(r)
    }

    /// Create a Boolean object.
    pub fn boolean(b: bool) -> Object {
        Object::Boolean(b)
    }

    /// Create an Array object.
    pub fn array(items: Vec<Object>) -> Object {
        Object::Array(items)
    }

    /// Create a Dictionary object.
    pub fn dict(entries: Vec<(&str, Object)>) -> Object {
        let map: HashMap<String, Object> = entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        Object::Dictionary(map)
    }

    /// Create a Reference object.
    pub fn reference(id: u32, gen: u16) -> Object {
        Object::Reference(ObjectRef::new(id, gen))
    }

    /// Create a rectangle array [x, y, width, height] -> [llx, lly, urx, ury].
    pub fn rect(x: f64, y: f64, width: f64, height: f64) -> Object {
        Object::Array(vec![
            Object::Real(x),
            Object::Real(y),
            Object::Real(x + width),
            Object::Real(y + height),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_null() {
        let s = ObjectSerializer::new();
        assert_eq!(s.serialize_to_string(&Object::Null), "null");
    }

    #[test]
    fn test_serialize_boolean() {
        let s = ObjectSerializer::new();
        assert_eq!(s.serialize_to_string(&Object::Boolean(true)), "true");
        assert_eq!(s.serialize_to_string(&Object::Boolean(false)), "false");
    }

    #[test]
    fn test_serialize_integer() {
        let s = ObjectSerializer::new();
        assert_eq!(s.serialize_to_string(&Object::Integer(42)), "42");
        assert_eq!(s.serialize_to_string(&Object::Integer(-123)), "-123");
    }

    #[test]
    fn test_serialize_real() {
        let s = ObjectSerializer::new();
        assert_eq!(s.serialize_to_string(&Object::Real(3.14159)), "3.14159");
        assert_eq!(s.serialize_to_string(&Object::Real(1.0)), "1");
        assert_eq!(s.serialize_to_string(&Object::Real(0.5)), "0.5");
    }

    #[test]
    fn test_serialize_string() {
        let s = ObjectSerializer::new();
        assert_eq!(s.serialize_to_string(&Object::String(b"Hello".to_vec())), "(Hello)");
        assert_eq!(
            s.serialize_to_string(&Object::String(b"Test (parens)".to_vec())),
            "(Test \\(parens\\))"
        );
    }

    #[test]
    fn test_serialize_hex_string() {
        let s = ObjectSerializer::new();
        // Binary data should use hex string
        assert_eq!(s.serialize_to_string(&Object::String(vec![0x00, 0xFF, 0x80])), "<00FF80>");
    }

    #[test]
    fn test_serialize_name() {
        let s = ObjectSerializer::new();
        assert_eq!(s.serialize_to_string(&Object::Name("Type".to_string())), "/Type");
        assert_eq!(s.serialize_to_string(&Object::Name("Font".to_string())), "/Font");
    }

    #[test]
    fn test_serialize_name_with_special_chars() {
        let s = ObjectSerializer::new();
        assert_eq!(
            s.serialize_to_string(&Object::Name("Name With Space".to_string())),
            "/Name#20With#20Space"
        );
    }

    #[test]
    fn test_serialize_array() {
        let s = ObjectSerializer::compact();
        let arr = Object::Array(vec![Object::Integer(1), Object::Integer(2), Object::Integer(3)]);
        assert_eq!(s.serialize_to_string(&arr), "[1 2 3]");
    }

    #[test]
    fn test_serialize_dictionary() {
        let s = ObjectSerializer::compact();
        let dict = ObjectSerializer::dict(vec![
            ("Type", ObjectSerializer::name("Page")),
            ("Count", ObjectSerializer::integer(1)),
        ]);
        let result = s.serialize_to_string(&dict);
        assert!(result.starts_with("<<"));
        assert!(result.ends_with(">>"));
        assert!(result.contains("/Type /Page"));
        assert!(result.contains("/Count 1"));
    }

    #[test]
    fn test_serialize_reference() {
        let s = ObjectSerializer::new();
        let r = Object::Reference(ObjectRef::new(10, 0));
        assert_eq!(s.serialize_to_string(&r), "10 0 R");
    }

    #[test]
    fn test_serialize_indirect() {
        let s = ObjectSerializer::new();
        let bytes = s.serialize_indirect(1, 0, &Object::Integer(42));
        let str = String::from_utf8_lossy(&bytes);
        assert!(str.contains("1 0 obj"));
        assert!(str.contains("42"));
        assert!(str.contains("endobj"));
    }

    #[test]
    fn test_serialize_stream() {
        let s = ObjectSerializer::compact();
        let mut dict = HashMap::new();
        dict.insert("Filter".to_string(), Object::Name("FlateDecode".to_string()));

        let stream = Object::Stream {
            dict,
            data: bytes::Bytes::from_static(b"stream data"),
        };

        let result = s.serialize_to_string(&stream);
        assert!(result.contains("/Length 11"));
        assert!(result.contains("stream\n"));
        assert!(result.contains("stream data"));
        assert!(result.contains("\nendstream"));
    }

    #[test]
    fn test_rect_helper() {
        let rect = ObjectSerializer::rect(0.0, 0.0, 612.0, 792.0);
        let s = ObjectSerializer::compact();
        assert_eq!(s.serialize_to_string(&rect), "[0 0 612 792]");
    }
}
