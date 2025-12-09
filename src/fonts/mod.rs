//! Font handling and encoding.
//!
//! This module provides font dictionary parsing, encoding handling,
//! and ToUnicode CMap parsing for accurate text extraction.
//!
//! Phase 4

mod adobe_glyph_list;
pub mod character_mapper;
pub mod cmap;
pub mod font_dict; // Private module - only used internally by font_dict

pub use character_mapper::CharacterMapper;
pub use cmap::{CMap, parse_tounicode_cmap};
pub use font_dict::{Encoding, FontInfo};
