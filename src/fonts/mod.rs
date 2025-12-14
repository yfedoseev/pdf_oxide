//! Font handling and encoding.
//!
//! This module provides font dictionary parsing, encoding handling,
//! and ToUnicode CMap parsing for accurate text extraction.
//!
//! Phase 4: Initial implementation
//! Phase 3: Enhanced with non-text content detection

mod adobe_glyph_list;
pub mod character_mapper;
pub mod cmap;
pub mod encoding_normalizer;
pub mod font_dict; // Private module - only used internally by font_dict
pub mod non_text_detection;
/// TrueType font CMap parsing for glyph-to-character mapping.
pub mod truetype_cmap;

pub use character_mapper::CharacterMapper;
pub use cmap::{parse_tounicode_cmap, CMap, LazyCMap};
pub use encoding_normalizer::EncodingNormalizer;
pub use font_dict::{CIDSystemInfo, CIDToGIDMap, Encoding, FontInfo};
pub use non_text_detection::{
    CharacterConfidence, ConfidenceReason, NonTextDetector, NonTextStats,
};
pub use truetype_cmap::TrueTypeCMap;
