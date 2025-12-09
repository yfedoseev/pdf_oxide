//! Types for PDF logical structure trees.
//!
//! Implements structure element types according to ISO 32000-1:2008 Section 14.7.2.

use crate::object::Object;
use std::collections::HashMap;

/// The root of a PDF structure tree (StructTreeRoot dictionary).
///
/// This is the entry point for accessing a document's logical structure.
/// According to PDF spec Section 14.7.2, the StructTreeRoot contains:
/// - `/Type` - Must be `/StructTreeRoot`
/// - `/K` - The immediate child or children of the structure tree root
/// - `/ParentTree` - Maps marked content to structure elements
/// - `/RoleMap` - Maps non-standard structure types to standard ones
#[derive(Debug, Clone)]
pub struct StructTreeRoot {
    /// Root structure element(s)
    pub root_elements: Vec<StructElem>,

    /// Parent tree mapping MCIDs to structure elements (optional)
    pub parent_tree: Option<ParentTree>,

    /// Role map for custom structure types (optional)
    pub role_map: HashMap<String, String>,
}

impl StructTreeRoot {
    /// Create a new structure tree root
    pub fn new() -> Self {
        Self {
            root_elements: Vec::new(),
            parent_tree: None,
            role_map: HashMap::new(),
        }
    }

    /// Add a root element to the structure tree
    pub fn add_root_element(&mut self, elem: StructElem) {
        self.root_elements.push(elem);
    }
}

impl Default for StructTreeRoot {
    fn default() -> Self {
        Self::new()
    }
}

/// A structure element (StructElem) in the structure tree.
///
/// According to PDF spec Section 14.7.2, each StructElem has:
/// - `/S` - Structure type (e.g., /Document, /P, /H1, /Sect)
/// - `/K` - Children (structure elements or marked content references)
/// - `/P` - Parent structure element
/// - `/Pg` - Page containing this element (optional)
/// - `/A` - Attributes (optional)
#[derive(Debug, Clone)]
pub struct StructElem {
    /// Structure type (e.g., "Document", "P", "H1", "Sect")
    pub struct_type: StructType,

    /// Child elements (structure elements or content references)
    pub children: Vec<StructChild>,

    /// Page number this element appears on (if known)
    pub page: Option<u32>,

    /// Attributes (optional)
    pub attributes: HashMap<String, Object>,
}

impl StructElem {
    /// Create a new structure element
    pub fn new(struct_type: StructType) -> Self {
        Self {
            struct_type,
            children: Vec::new(),
            page: None,
            attributes: HashMap::new(),
        }
    }

    /// Add a child to this structure element
    pub fn add_child(&mut self, child: StructChild) {
        self.children.push(child);
    }
}

/// Child of a structure element (either another struct elem or marked content reference)
#[derive(Debug, Clone)]
pub enum StructChild {
    /// Another structure element (recursive hierarchy)
    StructElem(Box<StructElem>),

    /// Reference to marked content by MCID (Marked Content ID)
    MarkedContentRef {
        /// Marked Content ID
        mcid: u32,
        /// Page number containing this marked content
        page: u32,
    },

    /// Object reference (indirect reference to another StructElem)
    ObjectRef(u32, u16), // (object_num, generation)
}

/// Standard structure types from PDF spec Section 14.8.4.
///
/// These are the standard structure types defined by the PDF specification.
/// Custom types can be mapped to standard types via the RoleMap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructType {
    // Document-level structure types
    /// Document root
    Document,
    /// Part (major division)
    Part,
    /// Article
    Art,
    /// Section
    Sect,
    /// Division
    Div,

    // Paragraph-level structure types
    /// Paragraph
    P,
    /// Heading level 1-6
    H,
    /// Heading level 1
    H1,
    /// Heading level 2
    H2,
    /// Heading level 3
    H3,
    /// Heading level 4
    H4,
    /// Heading level 5
    H5,
    /// Heading level 6
    H6,

    // List structure types
    /// List
    L,
    /// List item
    LI,
    /// Label (list item marker)
    Lbl,
    /// List body (list item content)
    LBody,

    // Table structure types
    /// Table
    Table,
    /// Table row
    TR,
    /// Table header cell
    TH,
    /// Table data cell
    TD,
    /// Table header group
    THead,
    /// Table body group
    TBody,
    /// Table footer group
    TFoot,

    // Inline structure types
    /// Span (inline generic)
    Span,
    /// Quote
    Quote,
    /// Note
    Note,
    /// Reference
    Reference,
    /// Bibliographic entry
    BibEntry,
    /// Code
    Code,
    /// Link
    Link,
    /// Annotation
    Annot,

    // Illustration structure types
    /// Figure
    Figure,
    /// Formula
    Formula,
    /// Form (input field)
    Form,

    // Non-standard or custom type
    /// Custom structure type not defined in the PDF specification
    Custom(String),
}

impl StructType {
    /// Parse structure type from string (e.g., "/P" -> StructType::P)
    pub fn from_str(s: &str) -> Self {
        match s {
            "Document" => Self::Document,
            "Part" => Self::Part,
            "Art" => Self::Art,
            "Sect" => Self::Sect,
            "Div" => Self::Div,
            "P" => Self::P,
            "H" => Self::H,
            "H1" => Self::H1,
            "H2" => Self::H2,
            "H3" => Self::H3,
            "H4" => Self::H4,
            "H5" => Self::H5,
            "H6" => Self::H6,
            "L" => Self::L,
            "LI" => Self::LI,
            "Lbl" => Self::Lbl,
            "LBody" => Self::LBody,
            "Table" => Self::Table,
            "TR" => Self::TR,
            "TH" => Self::TH,
            "TD" => Self::TD,
            "THead" => Self::THead,
            "TBody" => Self::TBody,
            "TFoot" => Self::TFoot,
            "Span" => Self::Span,
            "Quote" => Self::Quote,
            "Note" => Self::Note,
            "Reference" => Self::Reference,
            "BibEntry" => Self::BibEntry,
            "Code" => Self::Code,
            "Link" => Self::Link,
            "Annot" => Self::Annot,
            "Figure" => Self::Figure,
            "Formula" => Self::Formula,
            "Form" => Self::Form,
            _ => Self::Custom(s.to_string()),
        }
    }

    /// Check if this is a heading type (H, H1-H6)
    pub fn is_heading(&self) -> bool {
        matches!(self, Self::H | Self::H1 | Self::H2 | Self::H3 | Self::H4 | Self::H5 | Self::H6)
    }

    /// Check if this is a block-level element
    pub fn is_block(&self) -> bool {
        matches!(
            self,
            Self::Document
                | Self::Part
                | Self::Art
                | Self::Sect
                | Self::Div
                | Self::P
                | Self::H
                | Self::H1
                | Self::H2
                | Self::H3
                | Self::H4
                | Self::H5
                | Self::H6
                | Self::Table
                | Self::Figure
                | Self::Formula
        )
    }

    /// Get heading level (1-6) if this is a heading type
    pub fn heading_level(&self) -> Option<u8> {
        match self {
            Self::H | Self::H1 => Some(1),
            Self::H2 => Some(2),
            Self::H3 => Some(3),
            Self::H4 => Some(4),
            Self::H5 => Some(5),
            Self::H6 => Some(6),
            _ => None,
        }
    }

    /// Check if this is a list type (L, LI, Lbl, LBody)
    pub fn is_list(&self) -> bool {
        matches!(self, Self::L | Self::LI | Self::Lbl | Self::LBody)
    }

    /// Get markdown prefix for this structure type
    pub fn markdown_prefix(&self) -> Option<&'static str> {
        match self {
            Self::H1 => Some("# "),
            Self::H2 => Some("## "),
            Self::H3 => Some("### "),
            Self::H4 => Some("#### "),
            Self::H5 => Some("##### "),
            Self::H6 => Some("###### "),
            Self::Lbl => Some("- "),
            _ => None,
        }
    }
}

/// Parent tree that maps marked content IDs to structure elements.
///
/// According to PDF spec Section 14.7.4.4, the parent tree is a number tree
/// that maps MCID values to the structure elements that own them.
#[derive(Debug, Clone)]
pub struct ParentTree {
    /// Mapping from page number to MCID mappings for that page
    pub page_mappings: HashMap<u32, HashMap<u32, ParentTreeEntry>>,
}

impl ParentTree {
    /// Create a new parent tree
    pub fn new() -> Self {
        Self {
            page_mappings: HashMap::new(),
        }
    }

    /// Get the structure element that owns the given MCID on the given page
    pub fn get_parent(&self, page: u32, mcid: u32) -> Option<&ParentTreeEntry> {
        self.page_mappings
            .get(&page)
            .and_then(|page_map| page_map.get(&mcid))
    }
}

impl Default for ParentTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Entry in the parent tree
#[derive(Debug, Clone)]
pub enum ParentTreeEntry {
    /// Direct reference to a structure element
    StructElem(Box<StructElem>),
    /// Object reference (indirect)
    ObjectRef(u32, u16), // (object_num, generation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_type_parsing() {
        assert_eq!(StructType::from_str("P"), StructType::P);
        assert_eq!(StructType::from_str("H1"), StructType::H1);
        assert_eq!(StructType::from_str("Document"), StructType::Document);

        // Custom types
        match StructType::from_str("CustomType") {
            StructType::Custom(s) => assert_eq!(s, "CustomType"),
            _ => panic!("Expected Custom type"),
        }
    }

    #[test]
    fn test_is_heading() {
        assert!(StructType::H1.is_heading());
        assert!(StructType::H2.is_heading());
        assert!(StructType::H.is_heading());
        assert!(!StructType::P.is_heading());
        assert!(!StructType::Document.is_heading());
    }

    #[test]
    fn test_is_block() {
        assert!(StructType::P.is_block());
        assert!(StructType::H1.is_block());
        assert!(StructType::Document.is_block());
        assert!(!StructType::Span.is_block());
        assert!(!StructType::Link.is_block());
    }

    #[test]
    fn test_heading_level() {
        assert_eq!(StructType::H.heading_level(), Some(1));
        assert_eq!(StructType::H1.heading_level(), Some(1));
        assert_eq!(StructType::H2.heading_level(), Some(2));
        assert_eq!(StructType::H3.heading_level(), Some(3));
        assert_eq!(StructType::H4.heading_level(), Some(4));
        assert_eq!(StructType::H5.heading_level(), Some(5));
        assert_eq!(StructType::H6.heading_level(), Some(6));
        assert_eq!(StructType::P.heading_level(), None);
        assert_eq!(StructType::Document.heading_level(), None);
    }

    #[test]
    fn test_is_list() {
        assert!(StructType::L.is_list());
        assert!(StructType::LI.is_list());
        assert!(StructType::Lbl.is_list());
        assert!(StructType::LBody.is_list());
        assert!(!StructType::P.is_list());
        assert!(!StructType::H1.is_list());
        assert!(!StructType::Table.is_list());
    }

    #[test]
    fn test_markdown_prefix() {
        assert_eq!(StructType::H1.markdown_prefix(), Some("# "));
        assert_eq!(StructType::H2.markdown_prefix(), Some("## "));
        assert_eq!(StructType::H3.markdown_prefix(), Some("### "));
        assert_eq!(StructType::H4.markdown_prefix(), Some("#### "));
        assert_eq!(StructType::H5.markdown_prefix(), Some("##### "));
        assert_eq!(StructType::H6.markdown_prefix(), Some("###### "));
        assert_eq!(StructType::Lbl.markdown_prefix(), Some("- "));
        assert_eq!(StructType::P.markdown_prefix(), None);
        assert_eq!(StructType::Table.markdown_prefix(), None);
    }
}
