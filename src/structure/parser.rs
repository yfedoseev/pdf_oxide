//! Parser for PDF structure trees.
//!
//! Parses StructTreeRoot and StructElem dictionaries according to PDF spec Section 14.7.

use super::types::{StructChild, StructElem, StructTreeRoot, StructType};
use crate::document::PdfDocument;
use crate::error::Error;
use crate::object::Object;
use std::collections::{HashMap, HashSet};

/// Maximum time allowed for structure tree parsing (native only).
/// Documents with huge trees (50K+ elements) would take 5-10s;
/// a 200ms budget lets small/medium trees parse fully while
/// large trees fall back to content-stream order gracefully.
#[cfg(not(target_arch = "wasm32"))]
const STRUCT_TREE_PARSE_BUDGET: std::time::Duration = std::time::Duration::from_millis(200);

/// A deadline guard that works on both native and WASM targets.
///
/// On native, uses `std::time::Instant` for real time-based deadlines.
/// On `wasm32-unknown-unknown`, `std::time::Instant` panics at runtime,
/// so this becomes a no-op and the parser relies solely on `MAX_STRUCT_ELEMENTS`.
#[derive(Clone, Copy)]
struct Deadline {
    #[cfg(not(target_arch = "wasm32"))]
    instant: std::time::Instant,
}

impl Deadline {
    /// Create a deadline that expires after the configured budget.
    fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                instant: std::time::Instant::now() + STRUCT_TREE_PARSE_BUDGET,
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self {}
        }
    }

    /// Returns `true` if the deadline has been exceeded.
    #[inline]
    fn is_expired(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::time::Instant::now() > self.instant
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }
}

/// A timer for measuring elapsed time, WASM-safe.
#[derive(Clone, Copy)]
struct Timer {
    #[cfg(not(target_arch = "wasm32"))]
    start: std::time::Instant,
}

impl Timer {
    fn now() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                start: std::time::Instant::now(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self {}
        }
    }

    fn elapsed_debug(&self) -> String {
        #[cfg(not(target_arch = "wasm32"))]
        {
            format!("{:?}", self.start.elapsed())
        }
        #[cfg(target_arch = "wasm32")]
        {
            "(time unavailable on wasm)".to_string()
        }
    }
}

/// Maximum number of structure elements to parse.
/// Trees larger than this cause expensive traversal (seconds for 50K+ elements).
/// 10K elements is sufficient for any normal document; larger trees indicate
/// deeply structured books where content-stream order works equally well.
const MAX_STRUCT_ELEMENTS: usize = 10_000;

/// Decode a PDF text string (UTF-16BE/LE with BOM, or PDFDocEncoding).
fn decode_pdf_text_string(bytes: &[u8]) -> String {
    if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF {
        // UTF-16BE with BOM
        let utf16_pairs: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16(&utf16_pairs)
            .unwrap_or_else(|_| String::from_utf8_lossy(bytes).to_string())
    } else if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE {
        // UTF-16LE with BOM
        let utf16_pairs: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16(&utf16_pairs)
            .unwrap_or_else(|_| String::from_utf8_lossy(bytes).to_string())
    } else {
        // PDFDocEncoding
        bytes
            .iter()
            .filter_map(|&b| crate::fonts::font_dict::pdfdoc_encoding_lookup(b))
            .collect()
    }
}

/// Helper function to resolve an object (handles both direct objects and references).
fn resolve_object(document: &PdfDocument, obj: &Object) -> Result<Object, Error> {
    match obj {
        Object::Reference(obj_ref) => document.load_object(*obj_ref),
        _ => Ok(obj.clone()),
    }
}

/// Build a mapping from page object IDs to page indices.
/// This allows resolving /Pg references in marked content references.
///
/// Uses a single-pass traversal of the page tree (O(n)) instead of
/// calling get_page_ref per page (which is O(n) per call → O(n²) total).
fn build_page_map(document: &PdfDocument) -> HashMap<u32, u32> {
    let mut page_map = HashMap::new();

    // Get the root Pages node from the catalog
    let pages_ref = match document.catalog().ok().and_then(|cat| {
        cat.as_dict()
            .and_then(|d| d.get("Pages"))
            .and_then(|p| p.as_reference())
    }) {
        Some(r) => r,
        None => return page_map,
    };

    let mut index: u32 = 0;
    build_page_map_recursive(document, pages_ref, &mut page_map, &mut index);
    page_map
}

/// Recursively walk the page tree once, collecting page object IDs.
fn build_page_map_recursive(
    document: &PdfDocument,
    node_ref: crate::object::ObjectRef,
    page_map: &mut HashMap<u32, u32>,
    index: &mut u32,
) {
    let node = match document.load_object(node_ref) {
        Ok(n) => n,
        Err(_) => return,
    };
    let dict = match node.as_dict() {
        Some(d) => d,
        None => return,
    };
    let node_type = dict.get("Type").and_then(|t| t.as_name()).unwrap_or("");

    match node_type {
        "Page" => {
            page_map.insert(node_ref.id, *index);
            *index += 1;
        },
        "Pages" => {
            if let Some(kids) = dict.get("Kids").and_then(|k| k.as_array()) {
                let kid_refs: Vec<_> = kids.iter().filter_map(|k| k.as_reference()).collect();
                for kid_ref in kid_refs {
                    build_page_map_recursive(document, kid_ref, page_map, index);
                }
            }
        },
        _ => {},
    }
}

/// Parse the structure tree from a PDF document.
///
/// Reads the StructTreeRoot from the document catalog and recursively parses
/// all structure elements. Uses a time budget to avoid spending seconds on
/// documents with very large structure trees (50K+ elements). When the budget
/// is exceeded, returns `Ok(None)` so the caller falls back to content-stream
/// order (extract_spans).
///
/// # Arguments
/// * `document` - The PDF document
///
/// # Returns
/// * `Ok(Some(StructTreeRoot))` - If the document has a structure tree and it parsed in time
/// * `Ok(None)` - If the document is not tagged or the tree is too large to parse in budget
/// * `Err(Error)` - If parsing fails
pub fn parse_structure_tree(document: &PdfDocument) -> Result<Option<StructTreeRoot>, Error> {
    let parse_start = Timer::now();

    // Get catalog
    let catalog = document.catalog()?;

    // Check for StructTreeRoot in catalog dictionary
    let catalog_dict = catalog
        .as_dict()
        .ok_or_else(|| Error::InvalidPdf("Catalog is not a dictionary".into()))?;

    let struct_tree_root_ref = match catalog_dict.get("StructTreeRoot") {
        Some(obj) => obj,
        None => return Ok(None), // Not a tagged PDF
    };

    // Build page map for resolving /Pg references
    let page_map = build_page_map(document);

    // Start the deadline AFTER page map building (which is fixed cost)
    let deadline = Deadline::new();

    // Resolve the StructTreeRoot object
    let struct_tree_root_obj = resolve_object(document, struct_tree_root_ref)?;

    // Parse StructTreeRoot dictionary — treat non-dict (e.g. Null from a
    // corrupted parse) as "no structure tree" rather than a hard error.
    let struct_tree_dict = match struct_tree_root_obj.as_dict() {
        Some(d) => d,
        None => {
            log::warn!(
                "StructTreeRoot resolved to {} (expected dictionary), treating as no structure tree",
                struct_tree_root_obj.type_name()
            );
            return Ok(None);
        },
    };

    let mut struct_tree = StructTreeRoot::new();

    // Parse RoleMap (optional)
    if let Some(role_map_obj) = struct_tree_dict.get("RoleMap") {
        let role_map_obj = resolve_object(document, role_map_obj)?;
        if let Some(role_map_dict) = role_map_obj.as_dict() {
            for (key, value) in role_map_dict.iter() {
                if let Some(name) = value.as_name() {
                    struct_tree.role_map.insert(key.clone(), name.to_string());
                }
            }
        }
    }

    // Skip ParentTree parsing — it's expensive (recursively loads/parses objects)
    // and not needed for text extraction. The forward traversal of /K children
    // provides reading order. ParentTree is only needed for reverse lookups
    // (MCID → StructElem), which are not used in the extraction pipeline.

    // Parse K (children) - can be a single element or array of elements
    let mut element_count: usize = 0;
    let mut visited: HashSet<u32> = HashSet::new();

    if let Some(k_obj) = struct_tree_dict.get("K") {
        let k_obj = resolve_object(document, k_obj)?;

        match k_obj {
            Object::Array(arr) => {
                // Multiple root elements
                for elem_obj in arr {
                    if deadline.is_expired() {
                        log::debug!(
                            "Structure tree parse budget exceeded, falling back to content order"
                        );
                        return Ok(None);
                    }
                    if element_count > MAX_STRUCT_ELEMENTS {
                        log::debug!(
                            "Structure tree too large (>{} elements), falling back to content order",
                            MAX_STRUCT_ELEMENTS
                        );
                        return Ok(None);
                    }
                    // Record root element IDs before descending so that a back-reference
                    // from any descendant to this root is detectable as a cycle.
                    if let Object::Reference(obj_ref) = &elem_obj {
                        if !visited.insert(obj_ref.id) {
                            log::warn!(
                                "Cycle in structure tree: root object {} already visited, skipping",
                                obj_ref.id
                            );
                            continue;
                        }
                    }
                    if let Some(elem) = parse_struct_elem(
                        document,
                        &elem_obj,
                        &struct_tree.role_map,
                        &page_map,
                        deadline,
                        &mut element_count,
                        &mut visited,
                    )? {
                        struct_tree.add_root_element(elem);
                    }
                }
            },
            _ => {
                // Single root element
                if let Some(elem) = parse_struct_elem(
                    document,
                    &k_obj,
                    &struct_tree.role_map,
                    &page_map,
                    deadline,
                    &mut element_count,
                    &mut visited,
                )? {
                    struct_tree.add_root_element(elem);
                }
            },
        }
    }

    log::debug!(
        "Structure tree parsed: {} elements, {} root elements in {}",
        element_count,
        struct_tree.root_elements.len(),
        parse_start.elapsed_debug()
    );

    if element_count > MAX_STRUCT_ELEMENTS {
        log::debug!(
            "Structure tree too large ({} elements > {}), falling back to content order",
            element_count,
            MAX_STRUCT_ELEMENTS
        );
        return Ok(None);
    }

    Ok(Some(struct_tree))
}

/// Parse a structure element (StructElem) from a PDF object.
///
/// Returns `Ok(None)` if the deadline is exceeded, causing the caller to
/// abandon the tree and fall back to content-stream order.
fn parse_struct_elem(
    document: &PdfDocument,
    obj: &Object,
    role_map: &HashMap<String, String>,
    page_map: &HashMap<u32, u32>,
    deadline: Deadline,
    element_count: &mut usize,
    visited: &mut HashSet<u32>,
) -> Result<Option<StructElem>, Error> {
    // Check budgets before doing work
    if deadline.is_expired() || *element_count > MAX_STRUCT_ELEMENTS {
        return Ok(None);
    }
    *element_count += 1;

    let obj = resolve_object(document, obj)?;

    let dict = match obj.as_dict() {
        Some(d) => d,
        None => return Ok(None), // Not a dictionary, skip
    };

    // Check /Type (should be /StructElem, but optional)
    if let Some(type_obj) = dict.get("Type") {
        if let Some(type_name) = type_obj.as_name() {
            if type_name != "StructElem" {
                return Ok(None); // Not a StructElem
            }
        }
    }

    // Get /S (structure type) - REQUIRED
    let s_obj = match dict.get("S") {
        Some(obj) => obj,
        None => return Ok(None), // Missing /S, skip gracefully
    };
    let s_name = match s_obj.as_name() {
        Some(name) => name,
        None => return Ok(None), // /S not a name, skip
    };

    // Map custom types to standard types using RoleMap
    // Preserve the original role name when mapping occurs
    let mapped = role_map.get(s_name);
    let struct_type_str = mapped.map(|s| s.as_str()).unwrap_or(s_name);
    let struct_type = StructType::from_str(struct_type_str);

    let mut struct_elem = StructElem::new(struct_type);
    if mapped.is_some() {
        struct_elem.source_role = Some(s_name.to_string());
    }

    // Get /Pg (page) - optional, resolve to page number
    if let Some(Object::Reference(pg_ref)) = dict.get("Pg") {
        if let Some(&page_num) = page_map.get(&pg_ref.id) {
            struct_elem.page = Some(page_num);
        }
    }

    // Skip /A (attributes) during text extraction — not needed for reading order.
    // Skip /Alt (alternate description) — not needed for text extraction.

    // Get /ActualText (replacement text) - optional, per PDF spec Section 14.9.4
    // When present, this text replaces all descendant content for the element.
    if let Some(at_obj) = dict.get("ActualText") {
        let at_obj = resolve_object(document, at_obj)?;
        if let Some(at_bytes) = at_obj.as_string() {
            let text = decode_pdf_text_string(at_bytes);
            if !text.is_empty() {
                struct_elem.actual_text = Some(text);
            }
        }
    }

    // Parse /K (children)
    if let Some(k_obj_raw) = dict.get("K") {
        // When /K is an indirect reference that resolves to a struct elem dictionary
        // (as opposed to an array), we lose the object ID after resolve_object and
        // the Dictionary arm of parse_k_children cannot check for cycles.
        // Load it here while we still have the reference ID, insert into visited,
        // and short-circuit if it has already been visited.
        if let Object::Reference(r) = k_obj_raw {
            let k_resolved = match document.load_object(*r) {
                Ok(obj) => obj,
                Err(e) => {
                    log::warn!("Failed to load /K reference {}: {}", r.id, e);
                    return Ok(Some(struct_elem));
                },
            };
            if k_resolved.as_dict().is_some() {
                // /K points directly at a struct elem — guard against cycles.
                if !visited.insert(r.id) {
                    log::warn!(
                        "Cycle in structure tree: /K object {} already visited, skipping children",
                        r.id
                    );
                    return Ok(Some(struct_elem));
                }
            }
            parse_k_children(
                document,
                &k_resolved,
                &mut struct_elem,
                role_map,
                page_map,
                deadline,
                element_count,
                visited,
            )?;
        } else {
            let k_obj = resolve_object(document, k_obj_raw)?;
            parse_k_children(
                document,
                &k_obj,
                &mut struct_elem,
                role_map,
                page_map,
                deadline,
                element_count,
                visited,
            )?;
        }
    }

    Ok(Some(struct_elem))
}

/// Parse the /K entry (children) of a structure element.
fn parse_k_children(
    document: &PdfDocument,
    k_obj: &Object,
    parent: &mut StructElem,
    role_map: &HashMap<String, String>,
    page_map: &HashMap<u32, u32>,
    deadline: Deadline,
    element_count: &mut usize,
    visited: &mut HashSet<u32>,
) -> Result<(), Error> {
    match k_obj {
        Object::Integer(mcid) => {
            // Single MCID
            parent.add_child(StructChild::MarkedContentRef {
                mcid: *mcid as u32,
                page: parent.page.unwrap_or(0), // Use parent's page if available
            });
        },

        Object::Array(arr) => {
            // Array of children
            for child_obj in arr {
                // Check both time and element count budgets
                if deadline.is_expired() || *element_count > MAX_STRUCT_ELEMENTS {
                    return Ok(());
                }

                // Guard against cycles before resolving: an indirect reference that has
                // already been visited would resolve to a dictionary and slip through to
                // parse_struct_elem without any ID to check. Capture the ID now, while we
                // still have the unresolved Reference, so the check happens before loading.
                if let Object::Reference(obj_ref) = child_obj {
                    if !visited.insert(obj_ref.id) {
                        log::warn!(
                            "Cycle in structure tree: object {} already visited, skipping",
                            obj_ref.id
                        );
                        continue;
                    }
                }

                let child_obj = resolve_object(document, child_obj)?;

                match &child_obj {
                    Object::Integer(mcid) => {
                        // MCID
                        parent.add_child(StructChild::MarkedContentRef {
                            mcid: *mcid as u32,
                            page: parent.page.unwrap_or(0),
                        });
                    },

                    Object::Dictionary(_) => {
                        // Could be a StructElem or marked content reference
                        if let Some(child_elem) = parse_struct_elem(
                            document,
                            &child_obj,
                            role_map,
                            page_map,
                            deadline,
                            element_count,
                            visited,
                        )? {
                            parent.add_child(StructChild::StructElem(Box::new(child_elem)));
                        } else {
                            // Try parsing as marked content reference
                            if let Some(mcr) = parse_marked_content_ref(&child_obj, page_map)? {
                                parent.add_child(mcr);
                            }
                        }
                    },

                    Object::Reference(obj_ref) => {
                        // Double-indirect reference — guard against cycles here too.
                        if !visited.insert(obj_ref.id) {
                            log::warn!(
                                "Cycle in structure tree: object {} already visited, skipping",
                                obj_ref.id
                            );
                            continue;
                        }
                        // Resolve indirect reference and try to parse as StructElem
                        match document.load_object(*obj_ref) {
                            Ok(resolved) => {
                                if let Some(child_elem) = parse_struct_elem(
                                    document,
                                    &resolved,
                                    role_map,
                                    page_map,
                                    deadline,
                                    element_count,
                                    visited,
                                )? {
                                    parent.add_child(StructChild::StructElem(Box::new(child_elem)));
                                } else if let Some(mcr) =
                                    parse_marked_content_ref(&resolved, page_map)?
                                {
                                    parent.add_child(mcr);
                                }
                            },
                            Err(e) => {
                                log::warn!(
                                    "Failed to resolve ObjectRef {} {}: {}",
                                    obj_ref.id,
                                    obj_ref.gen,
                                    e
                                );
                            },
                        }
                    },

                    _ => {
                        // Unknown child type, skip
                    },
                }
            }
        },

        Object::Dictionary(_) => {
            // Single dictionary child
            if let Some(child_elem) = parse_struct_elem(
                document,
                k_obj,
                role_map,
                page_map,
                deadline,
                element_count,
                visited,
            )? {
                parent.add_child(StructChild::StructElem(Box::new(child_elem)));
            } else {
                // Try parsing as marked content reference
                if let Some(mcr) = parse_marked_content_ref(k_obj, page_map)? {
                    parent.add_child(mcr);
                }
            }
        },

        Object::Reference(obj_ref) => {
            // Guard against cycles: skip if this object has already been visited.
            if !visited.insert(obj_ref.id) {
                log::warn!(
                    "Cycle in structure tree: object {} already visited, skipping",
                    obj_ref.id
                );
                return Ok(());
            }
            // Resolve indirect reference and try to parse as StructElem
            match document.load_object(*obj_ref) {
                Ok(resolved) => {
                    if let Some(child_elem) = parse_struct_elem(
                        document,
                        &resolved,
                        role_map,
                        page_map,
                        deadline,
                        element_count,
                        visited,
                    )? {
                        parent.add_child(StructChild::StructElem(Box::new(child_elem)));
                    } else if let Some(mcr) = parse_marked_content_ref(&resolved, page_map)? {
                        parent.add_child(mcr);
                    }
                },
                Err(e) => {
                    log::warn!("Failed to resolve ObjectRef {} {}: {}", obj_ref.id, obj_ref.gen, e);
                },
            }
        },

        _ => {
            // Unknown K type
        },
    }

    Ok(())
}

/// Parse a marked content reference dictionary.
///
/// According to PDF spec, a marked content reference has:
/// - /Type /MCR
/// - /Pg - Page containing the marked content
/// - /MCID - Marked content ID
fn parse_marked_content_ref(
    obj: &Object,
    page_map: &HashMap<u32, u32>,
) -> Result<Option<StructChild>, Error> {
    let dict = match obj.as_dict() {
        Some(d) => d,
        None => return Ok(None),
    };

    // Check for /Type /MCR
    if let Some(type_obj) = dict.get("Type") {
        if let Some(type_name) = type_obj.as_name() {
            if type_name != "MCR" {
                return Ok(None);
            }
        }
    }

    // Get /MCID
    let mcid = match dict.get("MCID").and_then(|obj| obj.as_integer()) {
        Some(mcid) => mcid,
        None => return Ok(None), // Missing /MCID, skip gracefully
    };

    // Get /Pg (page reference) and resolve to page number
    let page = dict
        .get("Pg")
        .and_then(|pg_obj| {
            if let Object::Reference(pg_ref) = pg_obj {
                page_map.get(&pg_ref.id).copied()
            } else {
                None
            }
        })
        .unwrap_or(0); // Default to page 0 if no /Pg

    Ok(Some(StructChild::MarkedContentRef {
        mcid: mcid as u32,
        page,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_type_mapping() {
        let role_map = {
            let mut map = HashMap::new();
            map.insert("Heading1".to_string(), "H1".to_string());
            map
        };

        let mapped = role_map
            .get("Heading1")
            .map(|s| s.as_str())
            .unwrap_or("Heading1");
        assert_eq!(mapped, "H1");
    }

    #[test]
    fn test_decode_pdf_text_string_utf8() {
        let text = b"Hello World";
        assert_eq!(decode_pdf_text_string(text), "Hello World");
    }

    #[test]
    fn test_decode_pdf_text_string_utf16be() {
        // UTF-16BE BOM + "AB"
        let bytes = vec![0xFE, 0xFF, 0x00, 0x41, 0x00, 0x42];
        assert_eq!(decode_pdf_text_string(&bytes), "AB");
    }

    #[test]
    fn test_decode_pdf_text_string_utf16le() {
        // UTF-16LE BOM + "AB"
        let bytes = vec![0xFF, 0xFE, 0x41, 0x00, 0x42, 0x00];
        assert_eq!(decode_pdf_text_string(&bytes), "AB");
    }

    #[test]
    fn test_decode_pdf_text_string_pdfdoc_encoding() {
        // ASCII subset works as PDFDocEncoding
        let bytes = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F]; // "Hello"
        let result = decode_pdf_text_string(&bytes);
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_resolve_object_direct() {
        // Direct object should be returned as-is
        let obj = Object::Integer(42);
        let doc = {
            let pdf = build_test_pdf();
            PdfDocument::from_bytes(pdf).unwrap()
        };
        let result = resolve_object(&doc, &obj).unwrap();
        assert_eq!(result, Object::Integer(42));
    }

    #[test]
    fn test_parse_marked_content_ref_not_dict() {
        let obj = Object::Integer(5);
        let page_map = HashMap::new();
        let result = parse_marked_content_ref(&obj, &page_map).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_marked_content_ref_wrong_type() {
        let mut dict = HashMap::new();
        dict.insert("Type".to_string(), Object::Name("NotMCR".to_string()));
        dict.insert("MCID".to_string(), Object::Integer(5));
        let obj = Object::Dictionary(dict);
        let page_map = HashMap::new();
        let result = parse_marked_content_ref(&obj, &page_map).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_marked_content_ref_missing_mcid() {
        let mut dict = HashMap::new();
        dict.insert("Type".to_string(), Object::Name("MCR".to_string()));
        let obj = Object::Dictionary(dict);
        let page_map = HashMap::new();
        let result = parse_marked_content_ref(&obj, &page_map).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_marked_content_ref_valid() {
        let mut dict = HashMap::new();
        dict.insert("Type".to_string(), Object::Name("MCR".to_string()));
        dict.insert("MCID".to_string(), Object::Integer(7));
        let obj = Object::Dictionary(dict);
        let page_map = HashMap::new();
        let result = parse_marked_content_ref(&obj, &page_map).unwrap();
        assert!(result.is_some());
        if let Some(StructChild::MarkedContentRef { mcid, page }) = result {
            assert_eq!(mcid, 7);
            assert_eq!(page, 0); // default
        }
    }

    #[test]
    fn test_parse_marked_content_ref_with_page() {
        let mut page_map = HashMap::new();
        page_map.insert(10, 2u32); // object 10 -> page 2

        let mut dict = HashMap::new();
        dict.insert("Type".to_string(), Object::Name("MCR".to_string()));
        dict.insert("MCID".to_string(), Object::Integer(3));
        dict.insert(
            "Pg".to_string(),
            Object::Reference(crate::object::ObjectRef { id: 10, gen: 0 }),
        );
        let obj = Object::Dictionary(dict);
        let result = parse_marked_content_ref(&obj, &page_map).unwrap();
        if let Some(StructChild::MarkedContentRef { mcid, page }) = result {
            assert_eq!(mcid, 3);
            assert_eq!(page, 2);
        } else {
            panic!("Expected MarkedContentRef");
        }
    }

    #[test]
    fn test_parse_structure_tree_untagged_pdf() {
        let pdf = build_test_pdf();
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let result = parse_structure_tree(&doc).unwrap();
        assert!(result.is_none()); // No StructTreeRoot in minimal PDF
    }

    /// Build a minimal PDF for testing
    fn build_test_pdf() -> Vec<u8> {
        let mut pdf = b"%PDF-1.7\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
        let off3 = pdf.len();
        pdf.extend_from_slice(
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
        );

        let xref_offset = pdf.len();
        pdf.extend_from_slice(b"xref\n0 4\n");
        pdf.extend_from_slice(b"0000000000 65535 f \r\n");
        pdf.extend_from_slice(format!("{:010} 00000 n \r\n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \r\n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \r\n", off3).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_offset)
                .as_bytes(),
        );
        pdf
    }
}
