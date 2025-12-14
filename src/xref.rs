//! Cross-reference table parser.
//!
//! The xref table maps object numbers to byte offsets in the PDF file,
//! enabling random access to PDF objects.
//!
//! Supports both traditional xref tables (PDF 1.0-1.4) and
//! cross-reference streams (PDF 1.5+).

use crate::error::{Error, Result};
use crate::object::Object;
use crate::parser::parse_object;
use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};

/// Cross-reference table entry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XRefEntryType {
    /// Entry for a free object
    Free,
    /// Entry for an uncompressed object (traditional)
    Uncompressed,
    /// Entry for an object in an object stream (PDF 1.5+)
    Compressed,
}

/// Cross-reference table entry.
///
/// Each entry contains information about where to find an object.
/// Supports both traditional entries (byte offset) and compressed entries
/// (object stream reference).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XRefEntry {
    /// Type of entry
    pub entry_type: XRefEntryType,
    /// Byte offset (for uncompressed) or object stream number (for compressed)
    pub offset: u64,
    /// Generation number (for uncompressed) or index within stream (for compressed)
    pub generation: u16,
    /// Whether the object is in use (for traditional entries only)
    pub in_use: bool,
}

impl XRefEntry {
    /// Create a new cross-reference entry (traditional format).
    pub fn new(offset: u64, generation: u16, in_use: bool) -> Self {
        Self {
            entry_type: if in_use {
                XRefEntryType::Uncompressed
            } else {
                XRefEntryType::Free
            },
            offset,
            generation,
            in_use,
        }
    }

    /// Create a new uncompressed entry.
    pub fn uncompressed(offset: u64, generation: u16) -> Self {
        Self {
            entry_type: XRefEntryType::Uncompressed,
            offset,
            generation,
            in_use: true,
        }
    }

    /// Create a new compressed entry (object in object stream).
    pub fn compressed(stream_obj_num: u64, index_in_stream: u16) -> Self {
        Self {
            entry_type: XRefEntryType::Compressed,
            offset: stream_obj_num,
            generation: index_in_stream,
            in_use: true,
        }
    }

    /// Create a new free entry.
    pub fn free(next_free: u64, generation: u16) -> Self {
        Self {
            entry_type: XRefEntryType::Free,
            offset: next_free,
            generation,
            in_use: false,
        }
    }
}

/// Cross-reference table that maps object numbers to their locations.
#[derive(Debug, Clone)]
pub struct CrossRefTable {
    pub(crate) entries: HashMap<u32, XRefEntry>,
    /// Trailer dictionary (for xref streams, this is the stream dictionary)
    trailer: Option<HashMap<String, Object>>,
}

impl CrossRefTable {
    /// Create a new empty cross-reference table.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            trailer: None,
        }
    }

    /// Set the trailer dictionary.
    pub fn set_trailer(&mut self, trailer: HashMap<String, Object>) {
        self.trailer = Some(trailer);
    }

    /// Get the trailer dictionary if present.
    pub fn trailer(&self) -> Option<&HashMap<String, Object>> {
        self.trailer.as_ref()
    }

    /// Add an entry to the cross-reference table.
    pub fn add_entry(&mut self, object_number: u32, entry: XRefEntry) {
        self.entries.insert(object_number, entry);
    }

    /// Get an entry by object number.
    pub fn get(&self, object_number: u32) -> Option<&XRefEntry> {
        self.entries.get(&object_number)
    }

    /// Check if an object exists in the xref table.
    pub fn contains(&self, object_number: u32) -> bool {
        self.entries.contains_key(&object_number)
    }

    /// Get all object numbers in the table.
    pub fn all_object_numbers(&self) -> impl Iterator<Item = u32> + '_ {
        self.entries.keys().copied()
    }

    /// Merge entries from another xref table.
    ///
    /// Entries in self override entries in other (for incremental updates).
    /// This is used when following /Prev pointers in the trailer.
    pub fn merge_from(&mut self, other: CrossRefTable) {
        // Add entries from other that don't exist in self
        for (obj_num, entry) in other.entries {
            self.entries.entry(obj_num).or_insert(entry);
        }

        // If self doesn't have a trailer but other does, use other's trailer
        if self.trailer.is_none() && other.trailer.is_some() {
            self.trailer = other.trailer;
        }
    }

    /// Get the number of entries in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for CrossRefTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the byte offset of the xref table by scanning from the end of the file.
///
/// Searches for the "startxref" keyword in the last portion of the file,
/// then extracts the offset that follows it.
///
/// # Errors
///
/// Returns `Error::InvalidXref` if:
/// - The "startxref" keyword is not found
/// - The offset following "startxref" cannot be parsed
/// - The file is too small to contain a valid xref reference
pub fn find_xref_offset<R: Read + Seek>(reader: &mut R) -> Result<u64> {
    // Get file size
    let file_size = reader.seek(SeekFrom::End(0))?;

    // Read last portion of file (max 2KB to handle large trailers)
    let read_size = std::cmp::min(2048, file_size);
    reader.seek(SeekFrom::End(-(read_size as i64)))?;

    let mut buf = Vec::new();
    reader.take(read_size).read_to_end(&mut buf)?;

    // Convert to string for searching
    let content = String::from_utf8_lossy(&buf);

    // Search for "startxref" keyword (should be near end)
    let startxref_pos = content.rfind("startxref").ok_or(Error::InvalidXref)?;

    // Extract everything after "startxref"
    let after_keyword = &content[startxref_pos + 9..]; // 9 = len("startxref")

    // Split lines manually to handle CR, LF, and CRLF line endings
    // Standard .lines() only handles LF and CRLF, not standalone CR
    let lines = split_lines(after_keyword);

    // Find the first line that contains digits (the offset)
    for line in lines {
        let trimmed = line.trim();
        if !trimmed.is_empty() && trimmed.chars().all(|c| c.is_ascii_digit()) {
            return trimmed.parse::<u64>().map_err(|_| Error::InvalidXref);
        }
    }

    Err(Error::InvalidXref)
}

/// Parse the cross-reference table at the given byte offset.
///
/// Automatically detects whether this is a traditional xref table or
/// a cross-reference stream (PDF 1.5+) and parses accordingly.
///
/// # Errors
///
/// Returns `Error::InvalidXref` if parsing fails for both formats.
pub fn parse_xref<R: Read + Seek>(reader: &mut R, offset: u64) -> Result<CrossRefTable> {
    parse_xref_recursive(reader, offset, 0)
}

/// Parse xref table recursively, following /Prev pointers for incremental updates.
///
/// The depth parameter prevents infinite loops from circular /Prev chains.
fn parse_xref_recursive<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    depth: u32,
) -> Result<CrossRefTable> {
    // Prevent infinite recursion from circular /Prev chains
    if depth > 100 {
        return Err(Error::InvalidPdf("xref /Prev chain depth exceeded 100".to_string()));
    }

    reader.seek(SeekFrom::Start(offset))?;

    // Peek at the first few bytes to determine xref type
    let mut peek_buf = [0u8; 20]; // Increased to handle leading whitespace
    let bytes_read = reader.read(&mut peek_buf)?;
    reader.seek(SeekFrom::Start(offset))?; // Reset position

    let peek_str = String::from_utf8_lossy(&peek_buf[..bytes_read]);
    let trimmed = peek_str.trim_start(); // Skip leading whitespace

    log::debug!(
        "Parsing xref at offset {}, peek: {:?}",
        offset,
        &peek_str[..peek_str.len().min(15)]
    );

    // Parse the current xref (either traditional or stream)
    let mut xref = if trimmed.starts_with("xref") {
        log::debug!("Detected traditional xref at offset {}", offset);
        parse_traditional_xref(reader, offset)?
    } else if trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        // Try parsing as xref stream first
        match parse_xref_stream(reader, offset) {
            Ok(xref) => xref,
            Err(e) => {
                // Log the xref stream parsing error for debugging
                log::debug!("Failed to parse as xref stream: {}", e);
                // Fall back to traditional if stream parsing fails
                reader.seek(SeekFrom::Start(offset))?;
                match parse_traditional_xref(reader, offset) {
                    Ok(xref) => xref,
                    Err(trad_err) => {
                        // Both failed, return the xref stream error as it's more informative
                        log::debug!("Failed to parse as traditional xref: {}", trad_err);
                        return Err(Error::InvalidPdf(format!(
                            "failed to parse xref (stream attempt: {}, traditional attempt: {})",
                            e, trad_err
                        )));
                    },
                }
            },
        }
    } else {
        return Err(Error::InvalidXref);
    };

    // Check for /Prev pointer in trailer for incremental updates
    if let Some(trailer) = xref.trailer() {
        if let Some(prev_obj) = trailer.get("Prev") {
            if let Some(prev_offset) = prev_obj.as_integer() {
                log::debug!(
                    "Found /Prev pointer at offset {} in xref at offset {}",
                    prev_offset,
                    offset
                );

                // Recursively parse the previous xref
                let prev_xref = parse_xref_recursive(reader, prev_offset as u64, depth + 1)?;

                // Merge previous xref entries (current entries override previous ones)
                xref.merge_from(prev_xref);
            }
        }
    }

    Ok(xref)
}

/// Parse a traditional cross-reference table (PDF 1.0-1.4).
///
/// The xref table format is:
/// ```text
/// xref
/// 0 6             % Start at object 0, 6 entries
/// 0000000000 65535 f   % Object 0 (free)
/// 0000000018 00000 n   % Object 1 at byte 18
/// 0000000154 00000 n   % Object 2 at byte 154
/// ...
/// trailer
/// << /Size 6 /Root 1 0 R >>
/// ```
fn parse_traditional_xref<R: Read + Seek>(reader: &mut R, offset: u64) -> Result<CrossRefTable> {
    log::debug!("parse_traditional_xref: Starting at offset {}", offset);
    reader.seek(SeekFrom::Start(offset))?;

    // Read all content and split into lines (handles CR, LF, CRLF)
    let lines = read_all_and_split_lines(reader).map_err(|e| {
        log::error!("Failed to read lines: {}", e);
        Error::InvalidXref
    })?;

    log::debug!("parse_traditional_xref: Read {} lines", lines.len());

    let mut xref = CrossRefTable::new();
    let mut line_idx = 0;

    // Find "xref" keyword, skipping any leading whitespace lines
    while line_idx < lines.len() {
        let trimmed = lines[line_idx].trim();
        if trimmed.is_empty() {
            line_idx += 1;
            continue; // Skip empty lines
        }
        if trimmed.starts_with("xref") {
            line_idx += 1;
            break; // Found xref keyword
        }
        return Err(Error::InvalidXref); // Non-empty, non-xref line
    }

    // Parse subsections
    while line_idx < lines.len() {
        let trimmed = lines[line_idx].trim();
        line_idx += 1;

        // End of xref table
        if trimmed.starts_with("trailer") {
            break;
        }

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }

        // Parse subsection header: "start_obj count"
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() != 2 {
            continue; // Skip malformed lines
        }

        let start_obj: u32 = parts[0].parse().map_err(|_| Error::InvalidXref)?;
        let count: u32 = parts[1].parse().map_err(|_| Error::InvalidXref)?;

        // Validate reasonable count to prevent memory exhaustion
        if count > 1_000_000 {
            return Err(Error::InvalidPdf("xref subsection count exceeds limit".to_string()));
        }

        // Parse entries in this subsection
        let mut i = 0;
        while i < count && line_idx < lines.len() {
            let trimmed = lines[line_idx].trim();
            line_idx += 1;

            // Skip empty lines (don't increment counter)
            if trimmed.is_empty() {
                continue;
            }

            // Check if we've hit the trailer (end of xref)
            if trimmed.starts_with("trailer") {
                // We expected more entries but hit trailer early
                log::warn!("Expected {} entries but only found {} before trailer", count, i);
                line_idx -= 1; // Back up so outer loop can process trailer
                break;
            }

            // Parse entry: "nnnnnnnnnn ggggg f/n"
            // Be flexible with whitespace and format
            let parts: Vec<&str> = trimmed.split_whitespace().collect();

            // Try to handle various malformed formats
            if parts.len() < 3 {
                // Try to parse with different separators or formats
                log::warn!("Malformed xref entry (too few parts) at index {}: {:?}", i, trimmed);

                // Still increment counter to maintain object numbering
                // Add a placeholder free entry to maintain object number sequence
                let entry = XRefEntry::free(0, 65535);
                xref.add_entry(start_obj + i, entry);
                i += 1;
                continue;
            }

            // Allow extra parts (some PDFs have trailing data)
            if parts.len() > 3 {
                log::debug!("XRef entry has {} parts (expected 3): {:?}", parts.len(), trimmed);
            }

            let offset: u64 = match parts[0].parse() {
                Ok(v) => v,
                Err(_) => {
                    log::warn!("Failed to parse offset at index {}: {:?}", i, parts[0]);
                    // Add free entry to maintain numbering
                    let entry = XRefEntry::free(0, 65535);
                    xref.add_entry(start_obj + i, entry);
                    i += 1;
                    continue;
                },
            };

            let generation: u16 = match parts[1].parse() {
                Ok(v) => v,
                Err(_) => {
                    log::warn!("Failed to parse generation at index {}: {:?}", i, parts[1]);
                    // Add free entry to maintain numbering
                    let entry = XRefEntry::free(0, 65535);
                    xref.add_entry(start_obj + i, entry);
                    i += 1;
                    continue;
                },
            };

            let type_flag = parts[2];

            // Validate type flag - be flexible with case and truncation
            let type_flag_normalized = type_flag.to_lowercase();
            let type_char = type_flag_normalized.chars().next().unwrap_or('?');

            let in_use = match type_char {
                'n' => true,
                'f' => false,
                _ => {
                    log::warn!(
                        "Invalid type flag at index {}: {:?}, treating as free",
                        i,
                        type_flag
                    );
                    // Treat as free entry instead of skipping
                    false
                },
            };

            let entry = XRefEntry::new(offset, generation, in_use);
            xref.add_entry(start_obj + i, entry);
            i += 1;
        }
    }

    Ok(xref)
}

/// Parse a cross-reference stream (PDF 1.5+).
///
/// Cross-reference streams are stream objects with `/Type /XRef` that contain
/// binary encoded xref data. They replace traditional xref tables in modern PDFs.
///
/// The stream dictionary contains:
/// - `/W [w1 w2 w3]` - Field widths in bytes
/// - `/Size` - Total number of entries
/// - `/Index [start1 count1 start2 count2...]` - Optional subsection ranges
///
/// Each entry consists of 3 fields:
/// - Field 1: Entry type (0=free, 1=uncompressed, 2=compressed)
/// - Field 2: Offset (type 1) or stream object number (type 2)
/// - Field 3: Generation (type 1) or index within stream (type 2)
fn parse_xref_stream<R: Read + Seek>(reader: &mut R, offset: u64) -> Result<CrossRefTable> {
    use crate::lexer::token;

    reader.seek(SeekFrom::Start(offset))?;

    // Read enough data to parse the xref stream object
    // We need to read until we hit endobj or enough to parse the stream
    let mut buf_reader = BufReader::new(reader);
    let mut content = Vec::new();
    buf_reader.read_to_end(&mut content)?;

    // Parse the indirect object wrapper: "obj_num gen obj"
    let input = &content[..];

    // Skip object number
    let (rest, _obj_num_token) = token(input)
        .map_err(|e| Error::InvalidPdf(format!("failed to parse xref object number: {}", e)))?;

    // Skip generation number
    let (rest, _gen_token) = token(rest)
        .map_err(|e| Error::InvalidPdf(format!("failed to parse xref generation: {}", e)))?;

    // Skip 'obj' keyword
    let (rest, obj_keyword_token) = token(rest)
        .map_err(|e| Error::InvalidPdf(format!("failed to parse 'obj' keyword: {}", e)))?;

    // Verify it's actually the obj keyword
    if !matches!(obj_keyword_token, crate::lexer::Token::ObjStart) {
        return Err(Error::InvalidPdf("expected 'obj' keyword in xref stream".to_string()));
    }

    // Now parse the actual object (should be a stream)
    let parse_result = parse_object(rest)
        .map_err(|e| Error::InvalidPdf(format!("failed to parse xref stream object: {}", e)))?;

    // Extract the Object from the IResult tuple (remaining_input, parsed_object)
    let (_remaining, obj) = parse_result;

    // Extract the stream dict and data
    let (stream_dict, stream_data) = match obj {
        Object::Stream { dict, data } => (dict, data),
        _ => return Err(Error::InvalidPdf("xref stream is not a stream object".to_string())),
    };

    // Verify this is an xref stream
    if let Some(type_obj) = stream_dict.get("Type") {
        if let Some(type_name) = type_obj.as_name() {
            if type_name != "XRef" {
                return Err(Error::InvalidPdf(format!(
                    "expected /Type /XRef, got /Type /{}",
                    type_name
                )));
            }
        }
    }

    // Get field widths
    let w_array = stream_dict
        .get("W")
        .and_then(|o| o.as_array())
        .ok_or_else(|| Error::InvalidPdf("missing /W array in xref stream".to_string()))?;

    if w_array.len() != 3 {
        return Err(Error::InvalidPdf("invalid /W array length".to_string()));
    }

    let w1 = w_array[0]
        .as_integer()
        .ok_or_else(|| Error::InvalidPdf("invalid /W[0]".to_string()))? as usize;
    let w2 = w_array[1]
        .as_integer()
        .ok_or_else(|| Error::InvalidPdf("invalid /W[1]".to_string()))? as usize;
    let w3 = w_array[2]
        .as_integer()
        .ok_or_else(|| Error::InvalidPdf("invalid /W[2]".to_string()))? as usize;

    let entry_size = w1 + w2 + w3;

    // Get size
    let size = stream_dict
        .get("Size")
        .and_then(|o| o.as_integer())
        .ok_or_else(|| Error::InvalidPdf("missing /Size in xref stream".to_string()))?
        as u32;

    // Get index array (or default to [0 Size])
    let index_ranges = if let Some(index_obj) = stream_dict.get("Index") {
        let index_array = index_obj
            .as_array()
            .ok_or_else(|| Error::InvalidPdf("invalid /Index".to_string()))?;

        let mut ranges = Vec::new();
        for i in (0..index_array.len()).step_by(2) {
            let start = index_array[i]
                .as_integer()
                .ok_or_else(|| Error::InvalidPdf("invalid index start".to_string()))?
                as u32;
            let count = index_array[i + 1]
                .as_integer()
                .ok_or_else(|| Error::InvalidPdf("invalid index count".to_string()))?
                as u32;
            ranges.push((start, count));
        }
        ranges
    } else {
        vec![(0, size)]
    };

    // Extract decode parameters if present
    let decode_params = if let Some(decode_params_obj) = stream_dict.get("DecodeParms") {
        extract_decode_params(decode_params_obj)?
    } else {
        None
    };

    // Decode the stream data
    let decoded_data = if let Some(filter_obj) = stream_dict.get("Filter") {
        let filter_name = match filter_obj {
            Object::Name(name) => name.clone(),
            Object::Array(arr) => {
                // Multiple filters - use first one for now (or chain them)
                if let Some(Object::Name(name)) = arr.first() {
                    name.clone()
                } else {
                    return Err(Error::InvalidPdf("invalid filter array".to_string()));
                }
            },
            _ => return Err(Error::InvalidPdf("invalid /Filter in xref stream".to_string())),
        };

        crate::decoders::decode_stream_with_params(
            &stream_data,
            &[filter_name],
            decode_params.as_ref(),
        )?
    } else {
        stream_data.to_vec()
    };

    // Parse the binary xref data
    let mut xref = CrossRefTable::new();
    let mut data_pos = 0;

    for (start_obj, count) in index_ranges {
        for i in 0..count {
            if data_pos + entry_size > decoded_data.len() {
                return Err(Error::InvalidPdf("truncated xref stream data".to_string()));
            }

            let entry_data = &decoded_data[data_pos..data_pos + entry_size];
            data_pos += entry_size;

            // Read field 1 (type)
            let entry_type = if w1 > 0 {
                read_int(&entry_data[0..w1])
            } else {
                1 // Default to type 1 if width is 0
            };

            // Read field 2
            let field2 = read_int(&entry_data[w1..w1 + w2]);

            // Read field 3
            let field3 = read_int(&entry_data[w1 + w2..w1 + w2 + w3]);

            let entry = match entry_type {
                0 => {
                    // Type 0: Free object
                    XRefEntry::free(field2, field3 as u16)
                },
                1 => {
                    // Type 1: Uncompressed object at byte offset
                    XRefEntry::uncompressed(field2, field3 as u16)
                },
                2 => {
                    // Type 2: Compressed object in object stream
                    XRefEntry::compressed(field2, field3 as u16)
                },
                _ => {
                    return Err(Error::InvalidPdf(format!(
                        "invalid xref entry type: {}",
                        entry_type
                    )));
                },
            };

            xref.add_entry(start_obj + i, entry);
        }
    }

    // For xref streams, the stream dictionary serves as the trailer
    xref.set_trailer(stream_dict);

    Ok(xref)
}

/// Extract decode parameters from a DecodeParms object.
///
/// DecodeParms can be either a dictionary or an array of dictionaries.
/// For simplicity, we only extract from the first dictionary if it's an array.
fn extract_decode_params(
    decode_params_obj: &Object,
) -> Result<Option<crate::decoders::DecodeParams>> {
    let dict = match decode_params_obj {
        Object::Dictionary(d) => d,
        Object::Array(arr) => {
            // For array of params, use first one
            if let Some(Object::Dictionary(d)) = arr.first() {
                d
            } else {
                return Ok(None);
            }
        },
        _ => return Ok(None),
    };

    let predictor = dict
        .get("Predictor")
        .and_then(|o| o.as_integer())
        .unwrap_or(1);

    let columns = dict
        .get("Columns")
        .and_then(|o| o.as_integer())
        .unwrap_or(1) as usize;

    let colors = dict.get("Colors").and_then(|o| o.as_integer()).unwrap_or(1) as usize;

    let bits_per_component = dict
        .get("BitsPerComponent")
        .and_then(|o| o.as_integer())
        .unwrap_or(8) as usize;

    Ok(Some(crate::decoders::DecodeParams {
        predictor,
        columns,
        colors,
        bits_per_component,
    }))
}

/// Read an integer from a byte slice (big-endian).
fn read_int(bytes: &[u8]) -> u64 {
    let mut result: u64 = 0;
    for &byte in bytes {
        result = (result << 8) | (byte as u64);
    }
    result
}

/// Split a string into lines, handling all PDF line ending styles (LF, CRLF, CR).
///
/// Standard .lines() only handles LF and CRLF, but some PDFs use
/// standalone CR (Mac-style line endings). This function handles all three.
fn split_lines(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '\r' => {
                // Check if next is \n (CRLF)
                if i + 1 < chars.len() && chars[i + 1] == '\n' {
                    // CRLF
                    lines.push(current_line.clone());
                    current_line.clear();
                    i += 2;
                } else {
                    // Just CR
                    lines.push(current_line.clone());
                    current_line.clear();
                    i += 1;
                }
            },
            '\n' => {
                // LF
                lines.push(current_line.clone());
                current_line.clear();
                i += 1;
            },
            ch => {
                current_line.push(ch);
                i += 1;
            },
        }
    }

    // Don't forget the last line if it doesn't end with a line ending
    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines
}

/// Read a line from a BufReader, handling all PDF line ending styles (LF, CRLF, CR).
///
/// Standard BufReader::read_line() only handles LF and CRLF, but some PDFs use
/// standalone CR (Mac-style line endings). This function handles all three by
/// reading the entire buffer and splitting manually.
fn read_all_and_split_lines<R: Read>(reader: &mut R) -> std::io::Result<Vec<String>> {
    let mut content = Vec::new();
    reader.read_to_end(&mut content)?;

    let text = String::from_utf8_lossy(&content);

    // Use the shared split_lines function to handle CR, LF, and CRLF
    Ok(split_lines(&text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_xref_entry_creation() {
        let entry = XRefEntry::new(1234, 0, true);
        assert_eq!(entry.offset, 1234);
        assert_eq!(entry.generation, 0);
        assert!(entry.in_use);
    }

    #[test]
    fn test_xref_entry_free() {
        let entry = XRefEntry::new(0, 65535, false);
        assert_eq!(entry.offset, 0);
        assert_eq!(entry.generation, 65535);
        assert!(!entry.in_use);
    }

    #[test]
    fn test_cross_ref_table_new() {
        let table = CrossRefTable::new();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn test_cross_ref_table_add_and_get() {
        let mut table = CrossRefTable::new();
        let entry = XRefEntry::new(1234, 0, true);

        table.add_entry(5, entry.clone());
        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());

        let retrieved = table.get(5).unwrap();
        assert_eq!(retrieved, &entry);
    }

    #[test]
    fn test_cross_ref_table_get_missing() {
        let table = CrossRefTable::new();
        assert!(table.get(999).is_none());
    }

    #[test]
    fn test_find_xref_offset_valid() {
        let pdf = b"%PDF-1.4\n\
            1 0 obj\n\
            << /Type /Catalog >>\n\
            endobj\n\
            xref\n\
            0 2\n\
            0000000000 65535 f\n\
            0000000009 00000 n\n\
            trailer\n\
            << /Size 2 >>\n\
            startxref\n\
            50\n\
            %%EOF";

        let mut cursor = Cursor::new(pdf);
        let offset = find_xref_offset(&mut cursor).unwrap();
        assert_eq!(offset, 50);
    }

    #[test]
    fn test_find_xref_offset_no_startxref() {
        let pdf = b"%PDF-1.4\n\
            xref\n\
            0 1\n\
            0000000000 65535 f\n\
            trailer\n\
            << /Size 1 >>\n";

        let mut cursor = Cursor::new(pdf);
        let result = find_xref_offset(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_xref_offset_with_whitespace() {
        let pdf = b"%PDF-1.4\n\
            startxref\n\
            \n\
            12345\n\
            %%EOF";

        let mut cursor = Cursor::new(pdf);
        let offset = find_xref_offset(&mut cursor).unwrap();
        assert_eq!(offset, 12345);
    }

    #[test]
    fn test_parse_xref_single_subsection() {
        let xref_data = b"xref\n\
            0 3\n\
            0000000000 65535 f\n\
            0000000018 00000 n\n\
            0000000154 00000 n\n\
            trailer\n";

        let mut cursor = Cursor::new(xref_data);
        let table = parse_xref(&mut cursor, 0).unwrap();

        assert_eq!(table.len(), 3);

        // Object 0 (free)
        let entry0 = table.get(0).unwrap();
        assert_eq!(entry0.offset, 0);
        assert_eq!(entry0.generation, 65535);
        assert!(!entry0.in_use);

        // Object 1
        let entry1 = table.get(1).unwrap();
        assert_eq!(entry1.offset, 18);
        assert_eq!(entry1.generation, 0);
        assert!(entry1.in_use);

        // Object 2
        let entry2 = table.get(2).unwrap();
        assert_eq!(entry2.offset, 154);
        assert_eq!(entry2.generation, 0);
        assert!(entry2.in_use);
    }

    #[test]
    fn test_parse_xref_multiple_subsections() {
        let xref_data = b"xref\n\
            0 2\n\
            0000000000 65535 f\n\
            0000000018 00000 n\n\
            5 3\n\
            0000000200 00000 n\n\
            0000000300 00000 n\n\
            0000000400 00000 n\n\
            trailer\n";

        let mut cursor = Cursor::new(xref_data);
        let table = parse_xref(&mut cursor, 0).unwrap();

        assert_eq!(table.len(), 5); // 2 + 3 entries

        // First subsection
        assert!(table.get(0).is_some());
        assert!(table.get(1).is_some());

        // Second subsection (starts at 5)
        let entry5 = table.get(5).unwrap();
        assert_eq!(entry5.offset, 200);

        let entry6 = table.get(6).unwrap();
        assert_eq!(entry6.offset, 300);

        let entry7 = table.get(7).unwrap();
        assert_eq!(entry7.offset, 400);

        // Gap between subsections
        assert!(table.get(2).is_none());
        assert!(table.get(3).is_none());
        assert!(table.get(4).is_none());
    }

    #[test]
    fn test_parse_xref_no_xref_keyword() {
        let xref_data = b"notxref\n\
            0 1\n\
            0000000000 65535 f\n\
            trailer\n";

        let mut cursor = Cursor::new(xref_data);
        let result = parse_xref(&mut cursor, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_xref_malformed_entry() {
        // Parser should add placeholder free entries for malformed entries
        // to maintain object numbering consistency
        let xref_data = b"xref\n\
            0 2\n\
            0000000000 65535 f\n\
            invalid entry here\n\
            trailer\n";

        let mut cursor = Cursor::new(xref_data);
        let result = parse_xref(&mut cursor, 0);
        // Should succeed and have 2 entries (one valid, one placeholder free)
        assert!(result.is_ok());
        let table = result.unwrap();
        assert_eq!(table.len(), 2);
        // Object 0 should be the valid free entry
        assert!(table.get(0).is_some());
        assert!(!table.get(0).unwrap().in_use);
        // Object 1 should be the placeholder free entry
        assert!(table.get(1).is_some());
        assert!(!table.get(1).unwrap().in_use);
    }

    #[test]
    fn test_parse_xref_invalid_flag() {
        // Parser should treat entries with invalid flags as free entries
        // to maintain object numbering consistency
        let xref_data = b"xref\n\
            0 1\n\
            0000000000 65535 x\n\
            trailer\n";

        let mut cursor = Cursor::new(xref_data);
        let result = parse_xref(&mut cursor, 0);
        // Should succeed and have 1 entry (treated as free)
        assert!(result.is_ok());
        let table = result.unwrap();
        assert_eq!(table.len(), 1);
        // Object 0 should be a free entry
        assert!(table.get(0).is_some());
        assert!(!table.get(0).unwrap().in_use);
    }

    #[test]
    fn test_parse_xref_empty_table() {
        let xref_data = b"xref\n\
            trailer\n";

        let mut cursor = Cursor::new(xref_data);
        let table = parse_xref(&mut cursor, 0).unwrap();
        assert!(table.is_empty());
    }

    #[test]
    fn test_cross_ref_table_default() {
        let table = CrossRefTable::default();
        assert!(table.is_empty());
    }

    #[test]
    fn test_parse_xref_with_comments() {
        let xref_data = b"xref\n\
            % This is a comment\n\
            0 2\n\
            0000000000 65535 f\n\
            0000000018 00000 n\n\
            % Another comment\n\
            trailer\n";

        let mut cursor = Cursor::new(xref_data);
        let table = parse_xref(&mut cursor, 0).unwrap();
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn test_parse_xref_excessive_count() {
        let xref_data = b"xref\n\
            0 2000000\n\
            0000000000 65535 f\n\
            trailer\n";

        let mut cursor = Cursor::new(xref_data);
        let result = parse_xref(&mut cursor, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_xref_offset_cr_only_line_endings() {
        // Test Mac-style CR-only line endings (the bug we just fixed)
        let pdf_data = b"some content\r\
            startxref\r\
            173\r\
            %%EOF\r";

        let mut cursor = Cursor::new(pdf_data);
        let offset = find_xref_offset(&mut cursor).unwrap();
        assert_eq!(offset, 173);
    }

    #[test]
    fn test_parse_xref_cr_only_line_endings() {
        // Test parsing traditional xref with CR-only line endings
        let xref_data = b"xref\r\
            0 2\r\
            0000000000 65535 f\r\
            0000000018 00000 n\r\
            trailer\r";

        let mut cursor = Cursor::new(xref_data);
        let table = parse_xref(&mut cursor, 0).unwrap();
        assert_eq!(table.len(), 2);

        let entry0 = table.get(0).unwrap();
        assert!(!entry0.in_use);

        let entry1 = table.get(1).unwrap();
        assert_eq!(entry1.offset, 18);
        assert!(entry1.in_use);
    }

    #[test]
    fn test_split_lines_mixed_endings() {
        // Test the split_lines helper with mixed line endings
        let text = "line1\rline2\nline3\r\nline4";
        let lines = split_lines(text);
        assert_eq!(lines, vec!["line1", "line2", "line3", "line4"]);
    }
}
