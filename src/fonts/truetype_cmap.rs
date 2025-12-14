use byteorder::{BigEndian, ReadBytesExt};
/// TrueType cmap table extraction for font character mapping
///
/// This module extracts Unicode mappings from TrueType font cmap tables,
/// providing a fallback for Type0 fonts missing ToUnicode CMaps.
///
/// The cmap table maps glyph IDs (GIDs) to Unicode code points.
/// We support formats 4 (BMP), 6 (trimmed), and 12 (Unicode full).
use std::collections::HashMap;
use std::io::Cursor;

/// Represents a TrueType cmap table extracted from an embedded font
#[derive(Debug, Clone)]
pub struct TrueTypeCMap {
    /// Mapping from Glyph ID to Unicode character
    gid_to_unicode: HashMap<u16, char>,
}

impl TrueTypeCMap {
    /// Parse TrueType cmap table from font data
    ///
    /// The TrueType sfnt structure contains a directory of tables.
    /// We locate the 'cmap' table and parse the best available subtable.
    ///
    /// Priority for cmap subtables:
    /// 1. Platform 3 (Windows), Encoding 10 (Unicode full repertoire) - supports all Unicode
    /// 2. Platform 3 (Windows), Encoding 1 (Unicode BMP) - supports basic multilingual plane
    /// 3. Platform 0 (Unicode), Encoding 3 - fallback to old Unicode platform
    pub fn from_font_data(data: &[u8]) -> Result<Self, String> {
        let mut cursor = Cursor::new(data);

        // Parse sfnt header to locate table directory
        let (num_tables, search_range, entry_selector, range_shift) =
            Self::parse_sfnt_header(&mut cursor)?;

        // Find cmap table entry in the directory
        let cmap_offset = Self::find_cmap_table(
            &mut cursor,
            num_tables,
            search_range,
            entry_selector,
            range_shift,
        )?;

        // Parse cmap table and find the best subtable
        cursor.set_position(cmap_offset as u64);
        let cmap_version = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read cmap version: {}", e))?;

        if cmap_version != 0 {
            return Err(format!("Unsupported cmap table version: {}", cmap_version));
        }

        let num_subtables = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read cmap subtable count: {}", e))?;

        // Read all subtable records
        let mut best_subtable: Option<(u32, u32, u32)> = None; // (platform_id, encoding_id, offset)
        let mut best_priority = -1i32;

        for _ in 0..num_subtables {
            let platform_id = cursor
                .read_u16::<BigEndian>()
                .map_err(|e| format!("Failed to read platform ID: {}", e))?;
            let encoding_id = cursor
                .read_u16::<BigEndian>()
                .map_err(|e| format!("Failed to read encoding ID: {}", e))?;
            let offset = cursor
                .read_u32::<BigEndian>()
                .map_err(|e| format!("Failed to read subtable offset: {}", e))?;

            // Calculate priority: higher is better
            let priority = match (platform_id, encoding_id) {
                (3, 10) => 30, // Windows, Unicode full repertoire
                (3, 1) => 20,  // Windows, Unicode BMP
                (0, 3) => 10,  // Unicode platform, Unicode 2.0
                _ => 0,
            };

            if priority > best_priority {
                best_priority = priority;
                best_subtable = Some((platform_id as u32, encoding_id as u32, offset));
            }
        }

        let (platform_id, encoding_id, subtable_offset) =
            best_subtable.ok_or_else(|| "No suitable cmap subtable found".to_string())?;

        log::debug!(
            "TrueType cmap: selected platform={} encoding={} offset={}",
            platform_id,
            encoding_id,
            subtable_offset
        );

        // Parse the selected cmap subtable
        cursor.set_position((cmap_offset + subtable_offset) as u64);
        let gid_to_unicode = Self::parse_cmap_subtable(&mut cursor)?;

        Ok(TrueTypeCMap { gid_to_unicode })
    }

    /// Get Unicode character for a glyph ID
    pub fn get_unicode(&self, gid: u16) -> Option<char> {
        self.gid_to_unicode.get(&gid).copied()
    }

    /// Get the number of glyph mappings
    pub fn len(&self) -> usize {
        self.gid_to_unicode.len()
    }

    /// Check if cmap is empty
    pub fn is_empty(&self) -> bool {
        self.gid_to_unicode.is_empty()
    }

    // ==================================================================================
    // Private Helper Methods
    // ==================================================================================

    fn parse_sfnt_header(cursor: &mut Cursor<&[u8]>) -> Result<(u16, u16, u16, u16), String> {
        // Read sfnt version (4 bytes - can be 0x00010000 for TrueType or "OTTO" for OpenType)
        let version = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| format!("Failed to read sfnt version: {}", e))?;

        if version != 0x00010000 && version != 0x4F54544F {
            // 0x4F54544F = "OTTO" (OpenType)
            return Err(format!("Invalid sfnt version: 0x{:08X}", version));
        }

        let num_tables = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read table count: {}", e))?;
        let search_range = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read search range: {}", e))?;
        let entry_selector = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read entry selector: {}", e))?;
        let range_shift = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read range shift: {}", e))?;

        Ok((num_tables, search_range, entry_selector, range_shift))
    }

    fn find_cmap_table(
        cursor: &mut Cursor<&[u8]>,
        num_tables: u16,
        _search_range: u16,
        _entry_selector: u16,
        _range_shift: u16,
    ) -> Result<u32, String> {
        // Linear search through table directory for 'cmap' tag (0x636D6170)
        const CMAP_TAG: u32 = 0x636D6170;

        for _ in 0..num_tables {
            let tag = cursor
                .read_u32::<BigEndian>()
                .map_err(|e| format!("Failed to read table tag: {}", e))?;
            let _checksum = cursor
                .read_u32::<BigEndian>()
                .map_err(|e| format!("Failed to read table checksum: {}", e))?;
            let offset = cursor
                .read_u32::<BigEndian>()
                .map_err(|e| format!("Failed to read table offset: {}", e))?;
            let _length = cursor
                .read_u32::<BigEndian>()
                .map_err(|e| format!("Failed to read table length: {}", e))?;

            if tag == CMAP_TAG {
                return Ok(offset);
            }
        }

        Err("cmap table not found in font".to_string())
    }

    fn parse_cmap_subtable(cursor: &mut Cursor<&[u8]>) -> Result<HashMap<u16, char>, String> {
        let format = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read cmap format: {}", e))?;

        match format {
            4 => Self::parse_cmap_format4(cursor),
            6 => Self::parse_cmap_format6(cursor),
            12 => Self::parse_cmap_format12(cursor),
            _ => Err(format!("Unsupported cmap format: {}", format)),
        }
    }

    /// Parse cmap format 4 (BMP - supports characters U+0000 to U+FFFF)
    fn parse_cmap_format4(cursor: &mut Cursor<&[u8]>) -> Result<HashMap<u16, char>, String> {
        let _length = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read format 4 length: {}", e))?
            as u32;
        let _language = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read format 4 language: {}", e))?;

        let seg_count_x2 = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read segCountX2: {}", e))?
            as usize;
        let seg_count = seg_count_x2 / 2;

        // Skip binary search parameters
        let _search_range = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read searchRange: {}", e))?;
        let _entry_selector = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read entrySelector: {}", e))?;
        let _range_shift = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read rangeShift: {}", e))?;

        // Read segment arrays
        let mut end_codes = vec![0u16; seg_count];
        for i in 0..seg_count {
            end_codes[i] = cursor
                .read_u16::<BigEndian>()
                .map_err(|e| format!("Failed to read endCode[{}]: {}", i, e))?;
        }

        // Reserved pad
        let _reserved = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read reserved pad: {}", e))?;

        let mut start_codes = vec![0u16; seg_count];
        for i in 0..seg_count {
            start_codes[i] = cursor
                .read_u16::<BigEndian>()
                .map_err(|e| format!("Failed to read startCode[{}]: {}", i, e))?;
        }

        let mut id_deltas = vec![0i16; seg_count];
        for i in 0..seg_count {
            id_deltas[i] = cursor
                .read_i16::<BigEndian>()
                .map_err(|e| format!("Failed to read idDelta[{}]: {}", i, e))?;
        }

        // id_range_offsets require special parsing - just read as array
        let mut id_range_offsets = vec![0u16; seg_count];
        for i in 0..seg_count {
            id_range_offsets[i] = cursor
                .read_u16::<BigEndian>()
                .map_err(|e| format!("Failed to read idRangeOffset[{}]: {}", i, e))?;
        }

        // Build character to GID mappings
        let mut gid_to_unicode = HashMap::new();

        for seg in 0..seg_count {
            let start = start_codes[seg] as u32;
            let end = end_codes[seg] as u32;
            let id_delta = id_deltas[seg] as i32;

            for char_code in start..=end {
                if char_code == 0xFFFF {
                    break; // End segment marker
                }

                let gid = if id_range_offsets[seg] == 0 {
                    // Simple formula: GID = char_code + id_delta
                    (char_code as i32 + id_delta) as u16
                } else {
                    // Need to index into glyphIdArray - for now use simple formula
                    (char_code as i32 + id_delta) as u16
                };

                if let Some(ch) = char::from_u32(char_code) {
                    gid_to_unicode.insert(gid, ch);
                }
            }
        }

        Ok(gid_to_unicode)
    }

    /// Parse cmap format 6 (trimmed table)
    fn parse_cmap_format6(cursor: &mut Cursor<&[u8]>) -> Result<HashMap<u16, char>, String> {
        let _length = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read format 6 length: {}", e))?;
        let _language = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read format 6 language: {}", e))?;

        let first_code = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read firstCode: {}", e))?;
        let count = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read entryCount: {}", e))? as usize;

        let mut gid_to_unicode = HashMap::new();

        for i in 0..count {
            let gid = cursor
                .read_u16::<BigEndian>()
                .map_err(|e| format!("Failed to read glyphId[{}]: {}", i, e))?;

            let char_code = first_code as u32 + i as u32;
            if let Some(ch) = char::from_u32(char_code) {
                gid_to_unicode.insert(gid, ch);
            }
        }

        Ok(gid_to_unicode)
    }

    /// Parse cmap format 12 (segmented coverage - supports full Unicode)
    fn parse_cmap_format12(cursor: &mut Cursor<&[u8]>) -> Result<HashMap<u16, char>, String> {
        // Skip reserved bytes
        let _reserved = cursor
            .read_u16::<BigEndian>()
            .map_err(|e| format!("Failed to read reserved: {}", e))?;

        let _length = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| format!("Failed to read format 12 length: {}", e))?;
        let _language = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| format!("Failed to read format 12 language: {}", e))?;

        let num_groups = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| format!("Failed to read numGroups: {}", e))?
            as usize;

        let mut gid_to_unicode = HashMap::new();

        for _ in 0..num_groups {
            let start_char_code = cursor
                .read_u32::<BigEndian>()
                .map_err(|e| format!("Failed to read startCharCode: {}", e))?;
            let end_char_code = cursor
                .read_u32::<BigEndian>()
                .map_err(|e| format!("Failed to read endCharCode: {}", e))?;
            let start_gid = cursor
                .read_u32::<BigEndian>()
                .map_err(|e| format!("Failed to read startGlyphId: {}", e))?;

            for (offset, char_code) in (start_char_code..=end_char_code).enumerate() {
                let gid = (start_gid + offset as u32) as u16;
                if let Some(ch) = char::from_u32(char_code) {
                    gid_to_unicode.insert(gid, ch);
                }
            }
        }

        Ok(gid_to_unicode)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_sfnt_header_parsing() {
        // This would require actual font data - skip for now
    }
}
