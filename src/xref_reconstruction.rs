//! Cross-reference table reconstruction for damaged PDFs.
//!
//! When the xref table is corrupted, missing, or incomplete, this module
//! provides functionality to reconstruct it by scanning the entire PDF file
//! for object markers.
//!
//! This is a fallback mechanism used only when standard xref parsing fails.

use crate::error::{Error, Result};
use crate::object::Object;
use crate::parser::parse_object;
use crate::xref::{CrossRefTable, XRefEntry};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::sync::LazyLock;

static RE_OBJ_PATTERN: LazyLock<regex::bytes::Regex> =
    LazyLock::new(|| regex::bytes::Regex::new(r"(\d+)\s+(\d+)\s+obj").unwrap());
static RE_TRAILER: LazyLock<regex::bytes::Regex> =
    LazyLock::new(|| regex::bytes::Regex::new(r"trailer\s*<<").unwrap());

/// Reconstruct the cross-reference table by scanning the entire PDF file.
///
/// This function scans for "N G obj" patterns throughout the file and builds
/// an xref table from the discovered objects. It also attempts to find the
/// trailer dictionary and identify the catalog.
///
/// # Performance
///
/// For small to medium files (<10 MB), the entire file is read into memory.
/// For larger files, this could be optimized to scan in chunks, but that's
/// deferred until needed.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - No objects are found during scanning
/// - The catalog cannot be identified
///
/// # Example
///
/// ```no_run
/// # use std::fs::File;
/// # use std::io::BufReader;
/// # use pdf_oxide::xref_reconstruction::reconstruct_xref;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let file = File::open("damaged.pdf")?;
/// let mut reader = BufReader::new(file);
/// let (xref, trailer) = reconstruct_xref(&mut reader)?;
/// println!("Reconstructed {} objects", xref.len());
/// # Ok(())
/// # }
/// ```
pub fn reconstruct_xref<R: Read + Seek>(reader: &mut R) -> Result<(CrossRefTable, Object)> {
    log::info!("Reconstructing xref table by scanning file...");

    // Read entire file into memory for scanning
    reader.seek(SeekFrom::Start(0))?;
    let mut contents = Vec::new();
    reader.read_to_end(&mut contents)?;

    log::debug!("File size: {} bytes", contents.len());

    let mut xref = CrossRefTable::new();
    let mut objects_found = 0;

    // Scan for "N G obj" patterns
    // Pattern: one or more digits, whitespace, one or more digits, whitespace, "obj"
    for capture in RE_OBJ_PATTERN.captures_iter(&contents) {
        let full_match = match capture.get(0) {
            Some(m) => m,
            None => continue,
        };
        let obj_num_bytes = match capture.get(1) {
            Some(m) => m.as_bytes(),
            None => continue,
        };
        let gen_num_bytes = match capture.get(2) {
            Some(m) => m.as_bytes(),
            None => continue,
        };

        // Parse object and generation numbers
        let obj_num: u32 = match std::str::from_utf8(obj_num_bytes)
            .ok()
            .and_then(|s| s.parse().ok())
        {
            Some(n) => n,
            None => {
                log::warn!("Failed to parse object number at offset {}", full_match.start());
                continue;
            },
        };

        let gen_num: u16 = match std::str::from_utf8(gen_num_bytes)
            .ok()
            .and_then(|s| s.parse().ok())
        {
            Some(n) => n,
            None => {
                log::warn!("Failed to parse generation number at offset {}", full_match.start());
                continue;
            },
        };

        let offset = full_match.start() as u64;

        // SPEC COMPLIANCE FIX: Validate that this is actually an object header
        // PDF Spec: ISO 32000-1:2008, Section 7.5.4 - Cross-Reference Table
        //
        // Previous implementation would add ANY "N G obj" pattern to the xref table,
        // even if it appeared inside strings, comments, or corrupted data.
        //
        // This creates security risks:
        // 1. False positives can point to invalid object locations
        // 2. Can cause crashes when trying to parse non-object data as objects
        // 3. Malicious PDFs can craft fake object headers to confuse parsers
        //
        // Correct behavior: Validate that the pattern is followed by valid object syntax

        // Check if the next bytes after "obj" form a valid PDF object
        // We do basic validation: look for dictionary start, array start, or primitive values
        let validation_start = offset + full_match.as_bytes().len() as u64;
        if validation_start < contents.len() as u64 {
            let remaining = &contents[validation_start as usize..];

            // Skip whitespace
            let mut i = 0;
            while i < remaining.len() && remaining[i].is_ascii_whitespace() {
                i += 1;
            }

            if i < remaining.len() {
                let next_byte = remaining[i];

                // Valid object should start with:
                // - << (dictionary)
                // - [ (array)
                // - < (hex string or dict - ambiguous at this point)
                // - ( (literal string)
                // - / (name)
                // - t, f, n (true, false, null)
                // - digit or - (number)
                let is_valid_object_start =
                    matches!(next_byte, b'<' | b'[' | b'(' | b'/' | b't' | b'f' | b'n' | b'-')
                        || next_byte.is_ascii_digit();

                if !is_valid_object_start {
                    log::debug!(
                        "Skipping false positive object header at offset {} (next byte: 0x{:02x} '{}')",
                        offset,
                        next_byte,
                        if next_byte.is_ascii_graphic() {
                            next_byte as char
                        } else {
                            '?'
                        }
                    );
                    continue;
                }

                log::debug!("Validated object {} gen {} at offset {}", obj_num, gen_num, offset);
            }
        }

        // Add to xref table
        let entry = XRefEntry::uncompressed(offset, gen_num);
        xref.add_entry(obj_num, entry);
        objects_found += 1;
    }

    log::info!("Reconstructed xref with {} objects", objects_found);

    if objects_found == 0 {
        return Err(Error::InvalidPdf("No objects found during xref reconstruction".to_string()));
    }

    // Try to find the trailer dictionary
    let trailer = find_trailer(&contents, reader, &xref)?;

    Ok((xref, trailer))
}

/// Find and parse the trailer dictionary.
///
/// Searches for "trailer" keyword in the file and attempts to parse the
/// dictionary that follows. If not found, attempts to reconstruct a minimal
/// trailer by finding the catalog object.
fn find_trailer<R: Read + Seek>(
    contents: &[u8],
    reader: &mut R,
    xref: &CrossRefTable,
) -> Result<Object> {
    log::debug!("Searching for trailer dictionary...");

    // Search for all "trailer" keywords and prefer the last valid one.
    // Per ISO 32000-1:2008 Section 7.5.5, the most recent trailer (from the
    // latest incremental update) takes precedence. Using the first trailer can
    // miss /Encrypt entries added in later revisions.
    // The chosen /Root-bearing trailer plus the byte offset it was parsed
    // from (RE_TRAILER yields matches in ascending file order, so a later
    // offset = a more recent incremental update).
    let mut best_trailer: Option<(Object, usize)> = None;
    // /Encrypt /ID /Info salvaged from /Root-less parsed trailers, each
    // tracked with the offset it came from. If no /Root-bearing trailer
    // exists and we synthesize a minimal one, an encrypted file's /Encrypt
    // (and /ID, used for the encryption key) would otherwise be lost, making
    // the document undecryptable. Per ISO 32000-1 §7.5.5 the most recent
    // occurrence wins — including over a /Root-bearing trailer that appears
    // earlier in the file.
    let mut salvaged: HashMap<String, (Object, usize)> = HashMap::new();
    for mat in RE_TRAILER.find_iter(contents) {
        let trailer_start = mat.start();
        log::debug!("Found trailer keyword at offset {}", trailer_start);

        let trailer_keyword_end = trailer_start + 7; // len("trailer")
        let input = &contents[trailer_keyword_end..];
        match parse_object(input) {
            Ok((_, obj)) => {
                // Only accept a parsed trailer that actually carries /Root.
                // A Linearized file's sparse end-of-file trailer legitimately
                // omits /Root — the Catalog is reachable via the linearization
                // parameters / first xref chain, not the trailing trailer
                // (issue #509). Accepting a /Root-less trailer here would
                // short-circuit Catalog discovery and fail downstream with
                // "Trailer missing /Root entry". The *last* /Root-bearing
                // trailer still wins for /Root itself.
                if obj.as_dict().is_some_and(|d| d.get("Root").is_some()) {
                    best_trailer = Some((obj, trailer_start));
                } else {
                    if let Some(d) = obj.as_dict() {
                        for key in ["Encrypt", "ID", "Info"] {
                            if let Some(v) = d.get(key) {
                                salvaged.insert(key.to_string(), (v.clone(), trailer_start));
                            }
                        }
                    }
                    log::debug!(
                        "Parsed trailer at offset {} has no /Root — skipping (Catalog located by object scan; /Encrypt /ID /Info preserved)",
                        trailer_start
                    );
                }
            },
            Err(e) => {
                log::warn!("Failed to parse trailer dictionary at offset {}: {}", trailer_start, e);
            },
        }
    }
    if let Some((mut trailer, best_off)) = best_trailer {
        // Merge salvaged /Encrypt /ID /Info from /Root-less trailers using
        // most-recent-occurrence-wins (ISO 32000-1 §7.5.5): a salvaged value
        // overrides the /Root-bearing trailer's only when it was parsed from
        // a *later* offset (a newer incremental update — e.g. a sparse
        // trailer that adds encryption or rotates the file ID), and always
        // fills a key the /Root-bearing trailer lacks. An earlier /Root-less
        // value never clobbers a newer explicit one.
        if !salvaged.is_empty() {
            if let Object::Dictionary(d) = &mut trailer {
                for (key, (value, off)) in &salvaged {
                    match d.get(key) {
                        Some(_) if *off <= best_off => {}, // existing is newer/equal
                        _ => {
                            d.insert(key.clone(), value.clone());
                        },
                    }
                }
            }
        }
        log::info!("Successfully parsed trailer dictionary (last /Root-bearing occurrence)");
        return Ok(trailer);
    }

    // No /Root-bearing trailer found — synthesize one by scanning objects
    // for /Type /Catalog (handles Linearized files whose only trailer is the
    // sparse, /Root-less end-of-file trailer).
    log::info!(
        "No /Root-bearing trailer found; reconstructing minimal trailer via Catalog scan..."
    );
    let salvaged_values: HashMap<String, Object> =
        salvaged.into_iter().map(|(k, (v, _))| (k, v)).collect();
    reconstruct_minimal_trailer(reader, xref, &salvaged_values)
}

/// Reconstruct a minimal trailer dictionary.
///
/// Scans objects to find the catalog (object with /Type /Catalog) and
/// creates a minimal trailer with the required entries, plus any
/// `salvaged` entries (/Encrypt, /ID, /Info) carried over from a
/// /Root-less parsed trailer so encrypted documents remain decryptable.
fn reconstruct_minimal_trailer<R: Read + Seek>(
    reader: &mut R,
    xref: &CrossRefTable,
    salvaged: &HashMap<String, Object>,
) -> Result<Object> {
    log::debug!("Scanning objects to find catalog...");

    // We need to find the catalog object
    // The catalog is an object with /Type /Catalog
    let mut catalog_ref = None;

    // Scan objects looking for the catalog. `all_object_numbers()` is
    // `HashMap`-backed, so iterating it directly is nondeterministic: a
    // bounded scan over an arbitrary subset can miss the Catalog on
    // different runs (even for a ~114-object Linearized file).
    // `smallest_object_numbers` is deterministic, visits low-numbered
    // objects first (where the Catalog conventionally lives), and bounds the
    // candidate set *before* sorting so a maliciously sparse/huge xref stays
    // O(n log MAX_SCAN) time / O(MAX_SCAN) memory.
    const MAX_SCAN: usize = 4096;
    let obj_nums = xref.smallest_object_numbers(MAX_SCAN);
    let mut checked = 0usize;
    for obj_num in obj_nums {
        if checked >= MAX_SCAN {
            break;
        }

        if let Some(entry) = xref.get(obj_num) {
            if !entry.in_use {
                continue;
            }
            checked += 1;

            // Try to load and check this object
            match load_object_at_offset(reader, entry.offset) {
                Ok(obj) => {
                    if is_catalog(&obj) {
                        log::info!("Found catalog: object {} gen {}", obj_num, entry.generation);
                        catalog_ref = Some((obj_num, entry.generation));
                        break;
                    }
                },
                Err(e) => {
                    log::debug!(
                        "Failed to load object {} at offset {}: {}",
                        obj_num,
                        entry.offset,
                        e
                    );
                    continue;
                },
            }
        }
    }

    if catalog_ref.is_none() {
        return Err(Error::InvalidPdf("Could not find catalog in reconstructed xref".to_string()));
    }

    // Safety: catalog_ref.is_none() is checked above and returns Err
    let (cat_num, cat_gen) = catalog_ref.expect("catalog_ref validated above");

    // Create minimal trailer dictionary
    let mut trailer_dict = HashMap::new();
    trailer_dict.insert(
        "Root".to_string(),
        Object::Reference(crate::object::ObjectRef::new(cat_num, cat_gen)),
    );
    trailer_dict.insert("Size".to_string(), Object::Integer(xref.len() as i64));

    // Carry over /Encrypt, /ID, /Info salvaged from a skipped /Root-less
    // trailer. Never clobber the Root/Size we just computed.
    for (key, value) in salvaged {
        if key != "Root" && key != "Size" {
            trailer_dict.insert(key.clone(), value.clone());
        }
    }

    Ok(Object::Dictionary(trailer_dict))
}

/// Load an object at a specific byte offset.
///
/// This is a standalone function that doesn't require the full PdfDocument
/// context, used during trailer reconstruction.
fn load_object_at_offset<R: Read + Seek>(reader: &mut R, offset: u64) -> Result<Object> {
    reader.seek(SeekFrom::Start(offset))?;

    // Read enough data to parse the object
    let mut buf_reader = BufReader::new(reader);
    let mut content = Vec::new();

    // Read up to 1MB or until we find endobj
    // This is a conservative limit to avoid memory issues
    let mut bytes_read = 0;
    const MAX_OBJECT_SIZE: usize = 1024 * 1024; // 1MB

    loop {
        let mut line = Vec::new();
        match buf_reader.read_until(b'\n', &mut line) {
            Ok(0) => break, // EOF
            Ok(n) => {
                content.extend_from_slice(&line);
                bytes_read += n;

                if bytes_read > MAX_OBJECT_SIZE {
                    return Err(Error::InvalidPdf("Object too large".to_string()));
                }

                // Check if we've found endobj
                if content.windows(6).any(|w| w == b"endobj") {
                    break;
                }
            },
            Err(e) => return Err(Error::Io(e)),
        }
    }

    // Parse: "obj_num gen obj <object> endobj"
    use crate::lexer::token;

    let input = &content[..];

    // Skip object number
    let (rest, _) = token(input).map_err(|e| Error::ParseError {
        offset: 0,
        reason: format!("failed to parse object number: {}", e),
    })?;

    // Skip generation number
    let (rest, _) = token(rest).map_err(|e| Error::ParseError {
        offset: 0,
        reason: format!("failed to parse generation: {}", e),
    })?;

    // Skip 'obj' keyword
    let (rest, _) = token(rest).map_err(|e| Error::ParseError {
        offset: 0,
        reason: format!("failed to parse 'obj' keyword: {}", e),
    })?;

    // Parse the actual object
    let (_, obj) = parse_object(rest).map_err(|e| Error::ParseError {
        offset: 0,
        reason: format!("failed to parse object: {}", e),
    })?;

    Ok(obj)
}

/// Check if an object is the document catalog.
///
/// The catalog has /Type /Catalog in its dictionary.
fn is_catalog(obj: &Object) -> bool {
    if let Some(dict) = obj.as_dict() {
        if let Some(type_obj) = dict.get("Type") {
            if let Some(type_name) = type_obj.as_name() {
                return type_name == "Catalog";
            }
        }
    }
    false
}

/// Search for an object near a given offset.
///
/// When the reconstructed xref has slightly incorrect offsets, this function
/// searches within a ±1KB window to find the actual object.
pub fn search_nearby_for_object<R: Read + Seek>(
    reader: &mut R,
    obj_id: u32,
    approx_offset: u64,
) -> Result<Object> {
    log::debug!("Searching for object {} near offset {}", obj_id, approx_offset);

    // Search ±1KB from the expected offset
    let search_range = 1024u64;
    let start = approx_offset.saturating_sub(search_range);
    let end = approx_offset + search_range;

    reader.seek(SeekFrom::Start(start))?;
    let mut buffer = vec![0u8; (end - start) as usize];
    let bytes_read = reader.read(&mut buffer)?;
    let buffer = &buffer[..bytes_read];

    // Look for "N G obj" marker
    let pattern = format!(r"{} \d+ obj", obj_id);
    let re = match regex::bytes::Regex::new(&pattern) {
        Ok(r) => r,
        Err(_) => return Err(Error::ObjectNotFound(obj_id, 0)),
    };

    if let Some(mat) = re.find(buffer) {
        let obj_offset = start + mat.start() as u64;
        log::debug!(
            "Found object {} at offset {} (expected {})",
            obj_id,
            obj_offset,
            approx_offset
        );

        return load_object_at_offset(reader, obj_offset);
    }

    Err(Error::ObjectNotFound(obj_id, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_reconstruct_simple_pdf() {
        let pdf_data = b"%PDF-1.4\n\
            1 0 obj\n\
            << /Type /Catalog /Pages 2 0 R >>\n\
            endobj\n\
            2 0 obj\n\
            << /Type /Pages /Count 0 /Kids [] >>\n\
            endobj\n\
            trailer\n\
            << /Root 1 0 R /Size 3 >>\n\
            startxref\n\
            0\n\
            %%EOF";

        let mut cursor = Cursor::new(pdf_data);
        let result = reconstruct_xref(&mut cursor);

        assert!(result.is_ok());
        let (xref, trailer) = result.unwrap();

        // Should find objects 1 and 2
        assert!(xref.contains(1));
        assert!(xref.contains(2));

        // Trailer should have Root entry
        if let Some(dict) = trailer.as_dict() {
            assert!(dict.contains_key("Root"));
        } else {
            panic!("Trailer is not a dictionary");
        }
    }

    #[test]
    fn test_is_catalog() {
        let mut dict = HashMap::new();
        dict.insert("Type".to_string(), Object::Name("Catalog".to_string()));
        let catalog = Object::Dictionary(dict);

        assert!(is_catalog(&catalog));

        let not_catalog = Object::Integer(42);
        assert!(!is_catalog(&not_catalog));
    }

    #[test]
    fn test_reconstruct_no_objects() {
        let pdf_data = b"%PDF-1.4\n\
            This is not a valid PDF with objects\n\
            %%EOF";

        let mut cursor = Cursor::new(pdf_data);
        let result = reconstruct_xref(&mut cursor);

        assert!(result.is_err());
    }
}
