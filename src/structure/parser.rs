//! Parser for PDF structure trees.
//!
//! Parses StructTreeRoot and StructElem dictionaries according to PDF spec Section 14.7.

use super::types::{
    ParentTree, ParentTreeEntry, StructChild, StructElem, StructTreeRoot, StructType,
};
use crate::document::PdfDocument;
use crate::error::Error;
use crate::object::Object;
use std::collections::HashMap;

/// Helper function to resolve an object (handles both direct objects and references).
fn resolve_object(document: &mut PdfDocument, obj: &Object) -> Result<Object, Error> {
    match obj {
        Object::Reference(obj_ref) => document.load_object(*obj_ref),
        _ => Ok(obj.clone()),
    }
}

/// Parse the structure tree from a PDF document.
///
/// Reads the StructTreeRoot from the document catalog and recursively parses
/// all structure elements.
///
/// # Arguments
/// * `document` - The PDF document
///
/// # Returns
/// * `Ok(Some(StructTreeRoot))` - If the document has a structure tree
/// * `Ok(None)` - If the document is not tagged (no StructTreeRoot)
/// * `Err(Error)` - If parsing fails
pub fn parse_structure_tree(document: &mut PdfDocument) -> Result<Option<StructTreeRoot>, Error> {
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

    // Resolve the StructTreeRoot object
    let struct_tree_root_obj = resolve_object(document, struct_tree_root_ref)?;

    // Parse StructTreeRoot dictionary
    let struct_tree_dict = struct_tree_root_obj
        .as_dict()
        .ok_or_else(|| Error::InvalidPdf("StructTreeRoot is not a dictionary".into()))?;

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

    // Parse ParentTree (optional)
    if let Some(parent_tree_obj) = struct_tree_dict.get("ParentTree") {
        let parent_tree = parse_parent_tree(document, parent_tree_obj)?;
        struct_tree.parent_tree = Some(parent_tree);
    }

    // Parse K (children) - can be a single element or array of elements
    if let Some(k_obj) = struct_tree_dict.get("K") {
        let k_obj = resolve_object(document, k_obj)?;

        match k_obj {
            Object::Array(arr) => {
                // Multiple root elements
                for elem_obj in arr {
                    if let Some(elem) =
                        parse_struct_elem(document, &elem_obj, &struct_tree.role_map)?
                    {
                        struct_tree.add_root_element(elem);
                    }
                }
            },
            _ => {
                // Single root element
                if let Some(elem) = parse_struct_elem(document, &k_obj, &struct_tree.role_map)? {
                    struct_tree.add_root_element(elem);
                }
            },
        }
    }

    Ok(Some(struct_tree))
}

/// Parse a structure element (StructElem) from a PDF object.
///
/// # Arguments
/// * `document` - The PDF document
/// * `obj` - The object to parse (should be a dictionary)
/// * `role_map` - RoleMap for custom structure types
///
/// # Returns
/// * `Ok(Some(StructElem))` - Successfully parsed structure element
/// * `Ok(None)` - Not a valid structure element
/// * `Err(Error)` - Parsing error
fn parse_struct_elem(
    document: &mut PdfDocument,
    obj: &Object,
    role_map: &HashMap<String, String>,
) -> Result<Option<StructElem>, Error> {
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
    let s_obj = dict
        .get("S")
        .ok_or_else(|| Error::InvalidPdf("StructElem missing /S".into()))?;
    let s_name = s_obj
        .as_name()
        .ok_or_else(|| Error::InvalidPdf("StructElem /S is not a name".into()))?;

    // Map custom types to standard types using RoleMap
    let struct_type_str = role_map.get(s_name).map(|s| s.as_str()).unwrap_or(s_name);
    let struct_type = StructType::from_str(struct_type_str);

    let mut struct_elem = StructElem::new(struct_type);

    // Get /Pg (page) - optional
    if let Some(_pg_obj) = dict.get("Pg") {
        // Page reference - we'd need to resolve this to a page number
        // For now, skip (requires page tree traversal)
    }

    // Get /A (attributes) - optional
    if let Some(attr_obj) = dict.get("A") {
        let attr_obj = resolve_object(document, attr_obj)?;
        if let Some(attr_dict) = attr_obj.as_dict() {
            for (key, value) in attr_dict.iter() {
                struct_elem.attributes.insert(key.clone(), value.clone());
            }
        }
    }

    // Parse /K (children) - can be:
    // 1. A single integer (MCID)
    // 2. A dictionary (marked content reference with MCID and Pg)
    // 3. An array of any of the above or StructElems
    // 4. Another StructElem (dictionary with /Type /StructElem)
    if let Some(k_obj) = dict.get("K") {
        let k_obj = resolve_object(document, k_obj)?;
        parse_k_children(document, &k_obj, &mut struct_elem, role_map)?;
    }

    Ok(Some(struct_elem))
}

/// Parse the /K entry (children) of a structure element.
fn parse_k_children(
    document: &mut PdfDocument,
    k_obj: &Object,
    parent: &mut StructElem,
    role_map: &HashMap<String, String>,
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
                        if let Some(child_elem) = parse_struct_elem(document, &child_obj, role_map)?
                        {
                            parent.add_child(StructChild::StructElem(Box::new(child_elem)));
                        } else {
                            // Try parsing as marked content reference
                            if let Some(mcr) = parse_marked_content_ref(&child_obj)? {
                                parent.add_child(mcr);
                            }
                        }
                    },

                    Object::Reference(obj_ref) => {
                        // Object reference to another StructElem
                        parent.add_child(StructChild::ObjectRef(obj_ref.id, obj_ref.gen));
                    },

                    _ => {
                        // Unknown child type, skip
                    },
                }
            }
        },

        Object::Dictionary(_) => {
            // Single dictionary child
            if let Some(child_elem) = parse_struct_elem(document, k_obj, role_map)? {
                parent.add_child(StructChild::StructElem(Box::new(child_elem)));
            } else {
                // Try parsing as marked content reference
                if let Some(mcr) = parse_marked_content_ref(k_obj)? {
                    parent.add_child(mcr);
                }
            }
        },

        Object::Reference(obj_ref) => {
            // Object reference to another StructElem
            parent.add_child(StructChild::ObjectRef(obj_ref.id, obj_ref.gen));
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
fn parse_marked_content_ref(obj: &Object) -> Result<Option<StructChild>, Error> {
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
    let mcid = dict
        .get("MCID")
        .and_then(|obj| obj.as_integer())
        .ok_or_else(|| Error::InvalidPdf("MCR missing /MCID".into()))?;

    // Get /Pg (page reference)
    // For now, we'll use 0 as placeholder - proper implementation would resolve page reference
    let page = 0; // TODO: Resolve page reference

    Ok(Some(StructChild::MarkedContentRef {
        mcid: mcid as u32,
        page,
    }))
}

/// Parse the ParentTree from a PDF object.
///
/// The ParentTree is a number tree that maps MCIDs to structure elements.
/// According to PDF spec Section 7.9.7, number trees use /Nums (simple case)
/// or /Kids (complex case with intermediate nodes).
///
/// This implementation handles:
/// 1. Simple number trees with /Nums array (key-value pairs)
/// 2. Complex number trees with /Kids array (recursive node traversal)
fn parse_parent_tree(document: &mut PdfDocument, obj: &Object) -> Result<ParentTree, Error> {
    let obj = resolve_object(document, obj)?;

    let dict = match obj.as_dict() {
        Some(d) => d,
        None => return Ok(ParentTree::new()), // Not a dict, return empty
    };

    let mut parent_tree = ParentTree::new();

    // Try simple case first: /Nums array with key-value pairs
    if let Some(nums_obj) = dict.get("Nums") {
        let nums_obj = resolve_object(document, nums_obj)?;
        if let Some(nums_array) = nums_obj.as_array() {
            // /Nums is an array of alternating keys and values
            // [key1, value1, key2, value2, ...]
            let mut i = 0;
            while i + 1 < nums_array.len() {
                if let Some(key) = nums_array[i].as_integer() {
                    let entry = parse_parent_tree_entry(document, &nums_array[i + 1])?;
                    // Store in parent_tree with page=0 and mcid=key
                    // In practice, MCIDs are page-specific, so we use page 0 as default
                    parent_tree
                        .page_mappings
                        .entry(0)
                        .or_default()
                        .insert(key as u32, entry);
                }
                i += 2;
            }
            return Ok(parent_tree);
        }
    }

    // Try complex case: /Kids array with intermediate nodes
    if let Some(kids_obj) = dict.get("Kids") {
        let kids_obj = resolve_object(document, kids_obj)?;
        if let Some(kids_array) = kids_obj.as_array() {
            // Recursively parse each kid node
            for kid_obj in kids_array {
                parse_number_tree_kid(document, kid_obj, &mut parent_tree)?;
            }
            return Ok(parent_tree);
        }
    }

    // Neither /Nums nor /Kids found, return empty parent tree
    Ok(parent_tree)
}

/// Parse a single entry in the parent tree (can be StructElem or ObjectRef)
fn parse_parent_tree_entry(
    document: &mut PdfDocument,
    obj: &Object,
) -> Result<ParentTreeEntry, Error> {
    let obj = resolve_object(document, obj)?;

    match obj {
        Object::Dictionary(_) => {
            // Could be a StructElem dictionary or ObjectRef
            if let Some(struct_elem) = parse_struct_elem(document, &obj, &HashMap::new())? {
                Ok(ParentTreeEntry::StructElem(Box::new(struct_elem)))
            } else {
                // Fallback: treat as empty StructElem
                Ok(ParentTreeEntry::StructElem(Box::new(StructElem::new(StructType::Document))))
            }
        },
        Object::Reference(obj_ref) => {
            // Object reference to a StructElem
            Ok(ParentTreeEntry::ObjectRef(obj_ref.id, obj_ref.gen))
        },
        _ => {
            // Unknown type, treat as empty StructElem
            Ok(ParentTreeEntry::StructElem(Box::new(StructElem::new(StructType::Document))))
        },
    }
}

/// Recursively parse a number tree kid node
fn parse_number_tree_kid(
    document: &mut PdfDocument,
    kid_obj: &Object,
    parent_tree: &mut ParentTree,
) -> Result<(), Error> {
    let kid_obj = resolve_object(document, kid_obj)?;

    let kid_dict = match kid_obj.as_dict() {
        Some(d) => d,
        None => return Ok(()), // Not a dict, skip
    };

    // Check if this is a leaf node (has /Nums) or intermediate node (has /Kids)
    if let Some(nums_obj) = kid_dict.get("Nums") {
        let nums_obj = resolve_object(document, nums_obj)?;
        if let Some(nums_array) = nums_obj.as_array() {
            // Leaf node: parse key-value pairs
            let mut i = 0;
            while i + 1 < nums_array.len() {
                if let Some(key) = nums_array[i].as_integer() {
                    let entry = parse_parent_tree_entry(document, &nums_array[i + 1])?;
                    parent_tree
                        .page_mappings
                        .entry(0)
                        .or_default()
                        .insert(key as u32, entry);
                }
                i += 2;
            }
        }
    } else if let Some(kids_obj) = kid_dict.get("Kids") {
        let kids_obj = resolve_object(document, kids_obj)?;
        if let Some(kids_array) = kids_obj.as_array() {
            // Intermediate node: recursively parse children
            for child_kid_obj in kids_array {
                parse_number_tree_kid(document, child_kid_obj, parent_tree)?;
            }
        }
    }

    Ok(())
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
}
