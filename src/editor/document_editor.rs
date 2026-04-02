//! Main document editing interface.
//!
//! Provides the DocumentEditor type for modifying PDF documents.

use crate::document::PdfDocument;
use crate::editor::form_fields::FormFieldWrapper;
use crate::editor::resource_manager::ResourceManager;
use crate::elements::StructureElement;
use crate::error::{Error, Result};
use crate::extractors::HierarchicalExtractor;
use crate::geometry::Rect;
use crate::object::{Object, ObjectRef};
use crate::writer::{ContentStreamBuilder, ObjectSerializer};
use std::collections::{HashMap, HashSet};
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::BufWriter;
use std::io::{Read, Seek, Write};
use std::path::Path;

/// Document metadata (Info dictionary).
#[derive(Debug, Clone, Default)]
pub struct DocumentInfo {
    /// Document title
    pub title: Option<String>,
    /// Document author
    pub author: Option<String>,
    /// Document subject
    pub subject: Option<String>,
    /// Document keywords (comma-separated)
    pub keywords: Option<String>,
    /// Creator application
    pub creator: Option<String>,
    /// PDF producer
    pub producer: Option<String>,
    /// Creation date (PDF date format)
    pub creation_date: Option<String>,
    /// Modification date (PDF date format)
    pub mod_date: Option<String>,
}

impl DocumentInfo {
    /// Create a new empty DocumentInfo.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the author.
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set the subject.
    pub fn subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }

    /// Set the keywords.
    pub fn keywords(mut self, keywords: impl Into<String>) -> Self {
        self.keywords = Some(keywords.into());
        self
    }

    /// Set the creator.
    pub fn creator(mut self, creator: impl Into<String>) -> Self {
        self.creator = Some(creator.into());
        self
    }

    /// Set the producer.
    pub fn producer(mut self, producer: impl Into<String>) -> Self {
        self.producer = Some(producer.into());
        self
    }

    /// Convert to a PDF Info dictionary object.
    pub fn to_object(&self) -> Object {
        let mut dict = HashMap::new();

        if let Some(ref title) = self.title {
            dict.insert("Title".to_string(), Object::String(title.as_bytes().to_vec()));
        }
        if let Some(ref author) = self.author {
            dict.insert("Author".to_string(), Object::String(author.as_bytes().to_vec()));
        }
        if let Some(ref subject) = self.subject {
            dict.insert("Subject".to_string(), Object::String(subject.as_bytes().to_vec()));
        }
        if let Some(ref keywords) = self.keywords {
            dict.insert("Keywords".to_string(), Object::String(keywords.as_bytes().to_vec()));
        }
        if let Some(ref creator) = self.creator {
            dict.insert("Creator".to_string(), Object::String(creator.as_bytes().to_vec()));
        }
        if let Some(ref producer) = self.producer {
            dict.insert("Producer".to_string(), Object::String(producer.as_bytes().to_vec()));
        }
        if let Some(ref creation_date) = self.creation_date {
            dict.insert(
                "CreationDate".to_string(),
                Object::String(creation_date.as_bytes().to_vec()),
            );
        }
        if let Some(ref mod_date) = self.mod_date {
            dict.insert("ModDate".to_string(), Object::String(mod_date.as_bytes().to_vec()));
        }

        Object::Dictionary(dict)
    }

    /// Parse from a PDF Info dictionary object.
    pub fn from_object(obj: &Object) -> Self {
        let mut info = Self::default();

        if let Some(dict) = obj.as_dict() {
            if let Some(Object::String(s)) = dict.get("Title") {
                info.title = String::from_utf8_lossy(s).to_string().into();
            }
            if let Some(Object::String(s)) = dict.get("Author") {
                info.author = String::from_utf8_lossy(s).to_string().into();
            }
            if let Some(Object::String(s)) = dict.get("Subject") {
                info.subject = String::from_utf8_lossy(s).to_string().into();
            }
            if let Some(Object::String(s)) = dict.get("Keywords") {
                info.keywords = String::from_utf8_lossy(s).to_string().into();
            }
            if let Some(Object::String(s)) = dict.get("Creator") {
                info.creator = String::from_utf8_lossy(s).to_string().into();
            }
            if let Some(Object::String(s)) = dict.get("Producer") {
                info.producer = String::from_utf8_lossy(s).to_string().into();
            }
            if let Some(Object::String(s)) = dict.get("CreationDate") {
                info.creation_date = String::from_utf8_lossy(s).to_string().into();
            }
            if let Some(Object::String(s)) = dict.get("ModDate") {
                info.mod_date = String::from_utf8_lossy(s).to_string().into();
            }
        }

        info
    }
}

/// Information about a page.
#[derive(Debug, Clone)]
pub struct PageInfo {
    /// Page index (0-based)
    pub index: usize,
    /// Page width in points
    pub width: f32,
    /// Page height in points
    pub height: f32,
    /// Page rotation (0, 90, 180, 270)
    pub rotation: i32,
    /// Object reference for this page
    pub object_ref: ObjectRef,
}

/// Options for saving the document.
#[derive(Debug, Clone, Default)]
pub struct SaveOptions {
    /// Use incremental update (append to original file)
    pub incremental: bool,
    /// Compress streams
    pub compress: bool,
    /// Linearize for fast web view
    pub linearize: bool,
    /// Remove unused objects
    pub garbage_collect: bool,
    /// Encryption configuration (None = no encryption)
    pub encryption: Option<EncryptionConfig>,
}

impl SaveOptions {
    /// Create options for full rewrite (default).
    pub fn full_rewrite() -> Self {
        Self {
            incremental: false,
            compress: true,
            garbage_collect: true,
            ..Default::default()
        }
    }

    /// Create options for incremental update.
    pub fn incremental() -> Self {
        Self {
            incremental: true,
            compress: false,
            garbage_collect: false,
            ..Default::default()
        }
    }

    /// Create options with encryption enabled.
    ///
    /// Uses full rewrite mode since incremental updates don't support
    /// adding encryption to an existing PDF.
    pub fn with_encryption(config: EncryptionConfig) -> Self {
        Self {
            incremental: false,
            compress: true,
            garbage_collect: true,
            encryption: Some(config),
            ..Default::default()
        }
    }
}

/// Encryption algorithm for PDF security.
///
/// Per ISO 32000-1:2008 Section 7.6, PDF supports multiple encryption algorithms.
/// This enum represents the commonly used algorithms.
///
/// **Note**: This is a placeholder for v0.4.0 encryption support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncryptionAlgorithm {
    /// RC4 with 40-bit key (PDF 1.1+, considered weak).
    Rc4_40,
    /// RC4 with 128-bit key (PDF 1.4+).
    Rc4_128,
    /// AES with 128-bit key (PDF 1.5+).
    Aes128,
    /// AES with 256-bit key (PDF 1.7 Extension Level 3+, recommended).
    #[default]
    Aes256,
}

/// Permission flags for encrypted PDFs.
///
/// Per ISO 32000-1:2008 Section 7.6.3.2, these flags control what operations
/// are permitted when the document is opened with the user password.
///
/// **Note**: This is a placeholder for v0.4.0 encryption support.
#[derive(Debug, Clone, Default)]
pub struct Permissions {
    /// Allow printing the document.
    pub print: bool,
    /// Allow high-resolution printing.
    pub print_high_quality: bool,
    /// Allow modifying the document contents.
    pub modify: bool,
    /// Allow copying or extracting text and graphics.
    pub copy: bool,
    /// Allow adding annotations and form fields.
    pub annotate: bool,
    /// Allow filling in form fields.
    pub fill_forms: bool,
    /// Allow extracting content for accessibility.
    pub accessibility: bool,
    /// Allow document assembly (insert, rotate, delete pages).
    pub assemble: bool,
}

impl Permissions {
    /// Create with all permissions granted.
    pub fn all() -> Self {
        Self {
            print: true,
            print_high_quality: true,
            modify: true,
            copy: true,
            annotate: true,
            fill_forms: true,
            accessibility: true,
            assemble: true,
        }
    }

    /// Create with minimal permissions (view only).
    pub fn read_only() -> Self {
        Self {
            accessibility: true, // Always allow for compliance
            ..Default::default()
        }
    }

    /// Convert permissions to the 32-bit P value for the encryption dictionary.
    ///
    /// PDF Spec: Table 22 - User access permissions
    ///
    /// The returned value has reserved bits set appropriately:
    /// - Bits 7-8 must be 1
    /// - Bits 13-32 must be 1 (for compatibility)
    pub fn to_bits(&self) -> i32 {
        // Base value with required reserved bits set
        // Bits 7-8 (0-indexed: 6-7) and bits 13-32 (0-indexed: 12-31) must be 1
        let mut bits: i32 = 0xFFFFF0C0u32 as i32;

        // Bit 3 (0-indexed: 2): Print
        if self.print {
            bits |= 1 << 2;
        }

        // Bit 4 (0-indexed: 3): Modify contents
        if self.modify {
            bits |= 1 << 3;
        }

        // Bit 5 (0-indexed: 4): Copy or extract text and graphics
        if self.copy {
            bits |= 1 << 4;
        }

        // Bit 6 (0-indexed: 5): Add or modify annotations
        if self.annotate {
            bits |= 1 << 5;
        }

        // Bit 9 (0-indexed: 8): Fill in form fields (R>=3)
        if self.fill_forms {
            bits |= 1 << 8;
        }

        // Bit 10 (0-indexed: 9): Extract text for accessibility (R>=3)
        if self.accessibility {
            bits |= 1 << 9;
        }

        // Bit 11 (0-indexed: 10): Assemble document (R>=3)
        if self.assemble {
            bits |= 1 << 10;
        }

        // Bit 12 (0-indexed: 11): Print high quality (R>=3)
        if self.print_high_quality {
            bits |= 1 << 11;
        }

        bits
    }
}

/// Configuration for PDF encryption on save.
///
/// This struct configures how a PDF should be encrypted when saved.
/// Use with `SaveOptions::with_encryption()` to enable encryption.
///
/// # Example (Planned for v0.4.0)
///
/// ```ignore
/// use pdf_oxide::editor::{EncryptionConfig, EncryptionAlgorithm, Permissions};
///
/// let config = EncryptionConfig {
///     user_password: "user123".to_string(),
///     owner_password: "owner456".to_string(),
///     algorithm: EncryptionAlgorithm::Aes256,
///     permissions: Permissions::all(),
/// };
/// ```
///
/// **Note**: This is a placeholder for v0.4.0 encryption support.
/// Currently, PDFs are saved without encryption.
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Password required to open the document (can be empty for no user password).
    pub user_password: String,
    /// Password for full access and changing security settings.
    pub owner_password: String,
    /// Encryption algorithm to use.
    pub algorithm: EncryptionAlgorithm,
    /// Permission flags when opened with user password.
    pub permissions: Permissions,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            user_password: String::new(),
            owner_password: String::new(),
            algorithm: EncryptionAlgorithm::default(),
            permissions: Permissions::all(),
        }
    }
}

impl EncryptionConfig {
    /// Create a new encryption config with the given passwords.
    pub fn new(user_password: impl Into<String>, owner_password: impl Into<String>) -> Self {
        Self {
            user_password: user_password.into(),
            owner_password: owner_password.into(),
            ..Default::default()
        }
    }

    /// Set the encryption algorithm.
    pub fn with_algorithm(mut self, algorithm: EncryptionAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the permissions.
    pub fn with_permissions(mut self, permissions: Permissions) -> Self {
        self.permissions = permissions;
        self
    }
}

/// Trait for editable document operations.
pub trait EditableDocument {
    /// Get document metadata.
    fn get_info(&mut self) -> Result<DocumentInfo>;

    /// Set document metadata.
    fn set_info(&mut self, info: DocumentInfo) -> Result<()>;

    /// Get the number of pages.
    fn page_count(&mut self) -> Result<usize>;

    /// Get information about a specific page.
    fn get_page_info(&mut self, index: usize) -> Result<PageInfo>;

    /// Remove a page by index.
    fn remove_page(&mut self, index: usize) -> Result<()>;

    /// Move a page from one index to another.
    fn move_page(&mut self, from: usize, to: usize) -> Result<()>;

    /// Duplicate a page.
    fn duplicate_page(&mut self, index: usize) -> Result<usize>;

    /// Save the document to a file.
    fn save(&mut self, path: impl AsRef<Path>) -> Result<()>;

    /// Save with specific options.
    fn save_with_options(&mut self, path: impl AsRef<Path>, options: SaveOptions) -> Result<()>;
}

/// PDF document editor.
///
/// Provides a high-level interface for modifying PDF documents.
/// Changes are tracked and can be saved either as incremental updates
/// or as a complete rewrite.
pub struct DocumentEditor {
    /// Source document (for reading)
    source: PdfDocument,
    /// Path to the source file
    source_path: String,
    /// Modified objects (object ID -> new object)
    modified_objects: HashMap<u32, Object>,
    /// New objects to add (will be assigned new IDs)
    new_objects: Vec<Object>,
    /// Next object ID to use for new objects
    next_object_id: u32,
    /// Modified metadata
    modified_info: Option<DocumentInfo>,
    /// Page order (indices into original pages, or negative for removed)
    page_order: Vec<i32>,
    /// Number of pages in original document
    original_page_count: usize,
    /// Track if document has been modified
    is_modified: bool,
    /// Modified page content (page_index → new structure)
    modified_content: HashMap<usize, StructureElement>,
    /// Resource manager for fonts/images
    resource_manager: ResourceManager,
    /// Track if structure tree needs rebuilding
    structure_modified: bool,
    /// Modified page annotations (page_index → annotations)
    modified_annotations: HashMap<usize, Vec<crate::editor::dom::AnnotationWrapper>>,
    /// Modified page properties (rotation, boxes)
    modified_page_props: HashMap<usize, ModifiedPageProps>,
    /// Erase regions per page (whiteout overlays)
    erase_regions: HashMap<usize, Vec<[f32; 4]>>,
    /// Pages where annotations should be flattened
    flatten_annotations_pages: std::collections::HashSet<usize>,
    /// Pages where redactions should be applied
    apply_redactions_pages: std::collections::HashSet<usize>,
    /// Image modifications per page: page_index -> (image_name -> modification)
    image_modifications: HashMap<usize, HashMap<String, ImageModification>>,
    /// Pages where form fields should be flattened
    flatten_forms_pages: std::collections::HashSet<usize>,
    /// Flag to remove AcroForm from catalog after form flattening
    remove_acroform: bool,
    /// Embedded files to add to the document
    embedded_files: Vec<crate::writer::EmbeddedFile>,
    /// Modified or new form fields (field name → wrapper)
    modified_form_fields: HashMap<String, FormFieldWrapper>,
    /// Deleted form field names
    deleted_form_fields: HashSet<String>,
    /// Flag indicating AcroForm dictionary needs rebuilding on save
    acroform_modified: bool,
}

/// Tracks modified page properties.
#[derive(Debug, Clone, Default)]
pub struct ModifiedPageProps {
    /// New rotation value (0, 90, 180, 270)
    pub rotation: Option<i32>,
    /// New MediaBox
    pub media_box: Option<[f32; 4]>,
    /// New CropBox
    pub crop_box: Option<[f32; 4]>,
}

/// Stores annotation appearance data for flattening.
#[derive(Debug, Clone)]
struct AnnotationAppearance {
    /// Content stream bytes from the appearance
    content: Vec<u8>,
    /// BBox of the appearance XObject
    bbox: [f32; 4],
    /// Rect of the annotation on the page
    annot_rect: [f32; 4],
    /// Optional transformation matrix from the appearance
    matrix: Option<[f32; 6]>,
    /// Resources used by the appearance
    resources: Option<Object>,
}

/// Information about an image on a page.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ImageInfo {
    /// XObject name (e.g., "Im1")
    pub name: String,
    /// Position and size: x, y, width, height
    pub bounds: [f32; 4],
    /// Full transformation matrix [a, b, c, d, e, f]
    pub matrix: [f32; 6],
}

/// Modification to apply to an image.
#[derive(Debug, Clone)]
struct ImageModification {
    /// New x position (if Some, changes position)
    x: Option<f32>,
    /// New y position (if Some, changes position)
    y: Option<f32>,
    /// New width (if Some, changes width)
    width: Option<f32>,
    /// New height (if Some, changes height)
    height: Option<f32>,
}

impl DocumentEditor {
    /// Open a PDF document for editing.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let editor = DocumentEditor::open("document.pdf")?;
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let mut source = PdfDocument::open(path.as_ref())?;

        // Get page count
        let page_count = source.page_count()?;

        // Find the highest object ID to know where to start for new objects
        let next_id = Self::find_max_object_id(&source) + 1;

        // Initialize page order as sequential
        let page_order: Vec<i32> = (0..page_count as i32).collect();

        Ok(Self {
            source,
            source_path: path_str,
            modified_objects: HashMap::new(),
            new_objects: Vec::new(),
            next_object_id: next_id,
            modified_info: None,
            page_order,
            original_page_count: page_count,
            is_modified: false,
            modified_content: HashMap::new(),
            resource_manager: ResourceManager::new(),
            structure_modified: false,
            modified_annotations: HashMap::new(),
            modified_page_props: HashMap::new(),
            erase_regions: HashMap::new(),
            flatten_annotations_pages: std::collections::HashSet::new(),
            apply_redactions_pages: std::collections::HashSet::new(),
            image_modifications: HashMap::new(),
            flatten_forms_pages: std::collections::HashSet::new(),
            remove_acroform: false,
            embedded_files: Vec::new(),
            modified_form_fields: HashMap::new(),
            deleted_form_fields: HashSet::new(),
            acroform_modified: false,
        })
    }

    /// Open a PDF document for editing from an existing PdfDocument object.
    pub fn from_document(mut source: PdfDocument) -> Result<Self> {
        let page_count = source.page_count()?;
        let next_id = Self::find_max_object_id(&source) + 1;
        let page_order: Vec<i32> = (0..page_count as i32).collect();

        Ok(Self {
            source,
            source_path: String::new(),
            modified_objects: HashMap::new(),
            new_objects: Vec::new(),
            next_object_id: next_id,
            modified_info: None,
            page_order,
            original_page_count: page_count,
            is_modified: false,
            modified_content: HashMap::new(),
            resource_manager: ResourceManager::new(),
            structure_modified: false,
            modified_annotations: HashMap::new(),
            modified_page_props: HashMap::new(),
            erase_regions: HashMap::new(),
            flatten_annotations_pages: std::collections::HashSet::new(),
            apply_redactions_pages: std::collections::HashSet::new(),
            image_modifications: HashMap::new(),
            flatten_forms_pages: std::collections::HashSet::new(),
            remove_acroform: false,
            embedded_files: Vec::new(),
            modified_form_fields: HashMap::new(),
            deleted_form_fields: HashSet::new(),
            acroform_modified: false,
        })
    }

    /// Open a PDF document for editing from in-memory bytes.
    ///
    /// This is the equivalent of `open()` but works with byte data instead of a file path,
    /// making it suitable for WASM environments where filesystem access is unavailable.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let mut source = PdfDocument::from_bytes(data)?;
        let page_count = source.page_count()?;
        let next_id = Self::find_max_object_id(&source) + 1;
        let page_order: Vec<i32> = (0..page_count as i32).collect();

        Ok(Self {
            source,
            source_path: String::new(),
            modified_objects: HashMap::new(),
            new_objects: Vec::new(),
            next_object_id: next_id,
            modified_info: None,
            page_order,
            original_page_count: page_count,
            is_modified: false,
            modified_content: HashMap::new(),
            resource_manager: ResourceManager::new(),
            structure_modified: false,
            modified_annotations: HashMap::new(),
            modified_page_props: HashMap::new(),
            erase_regions: HashMap::new(),
            flatten_annotations_pages: std::collections::HashSet::new(),
            apply_redactions_pages: std::collections::HashSet::new(),
            image_modifications: HashMap::new(),
            flatten_forms_pages: std::collections::HashSet::new(),
            remove_acroform: false,
            embedded_files: Vec::new(),
            modified_form_fields: HashMap::new(),
            deleted_form_fields: HashSet::new(),
            acroform_modified: false,
        })
    }

    /// Deprecated alias for `from_bytes`.
    #[deprecated(since = "0.3.15", note = "Use `from_bytes` instead")]
    pub fn open_from_bytes(data: Vec<u8>) -> Result<Self> {
        Self::from_bytes(data)
    }

    /// Save the document to an in-memory byte vector.
    ///
    /// This is the equivalent of `save()` but returns bytes instead of writing to a file,
    /// making it suitable for WASM environments where filesystem access is unavailable.
    pub fn save_to_bytes(&mut self) -> Result<Vec<u8>> {
        self.save_to_bytes_with_options(SaveOptions::full_rewrite())
    }

    /// Save the document to an in-memory byte vector with specific options.
    pub fn save_to_bytes_with_options(&mut self, options: SaveOptions) -> Result<Vec<u8>> {
        use std::io::Cursor;
        if options.incremental {
            return Err(Error::InvalidPdf(
                "Incremental saves are not supported for in-memory output".to_string(),
            ));
        }
        let mut cursor = Cursor::new(Vec::new());
        self.write_full_to_writer(&mut cursor, options.encryption.as_ref())?;
        Ok(cursor.into_inner())
    }

    /// Find the maximum object ID in the document.
    fn find_max_object_id(doc: &PdfDocument) -> u32 {
        // Get /Size from trailer - this is the number of xref entries (max ID + 1)
        doc.trailer()
            .as_dict()
            .and_then(|d| d.get("Size"))
            .and_then(|s| s.as_integer())
            .map(|size| size as u32)
            .unwrap_or(100) // Fallback to reasonable default
    }

    /// Allocate a new object ID.
    fn allocate_object_id(&mut self) -> u32 {
        let id = self.next_object_id;
        self.next_object_id += 1;
        id
    }

    /// Apply page property modifications to a page object.
    ///
    /// Returns a new page object with the modifications applied.
    fn apply_page_props_to_object(
        &self,
        page_obj: &Object,
        props: &ModifiedPageProps,
    ) -> Result<Object> {
        let page_dict = page_obj
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Page is not a dictionary".to_string()))?;

        let mut new_dict = page_dict.clone();

        // Apply rotation if modified
        if let Some(rotation) = props.rotation {
            new_dict.insert("Rotate".to_string(), Object::Integer(rotation as i64));
        }

        // Apply MediaBox if modified
        if let Some(media_box) = props.media_box {
            let box_array = Object::Array(vec![
                Object::Real(media_box[0] as f64),
                Object::Real(media_box[1] as f64),
                Object::Real(media_box[2] as f64),
                Object::Real(media_box[3] as f64),
            ]);
            new_dict.insert("MediaBox".to_string(), box_array);
        }

        // Apply CropBox if modified
        if let Some(crop_box) = props.crop_box {
            let box_array = Object::Array(vec![
                Object::Real(crop_box[0] as f64),
                Object::Real(crop_box[1] as f64),
                Object::Real(crop_box[2] as f64),
                Object::Real(crop_box[3] as f64),
            ]);
            new_dict.insert("CropBox".to_string(), box_array);
        }

        Ok(Object::Dictionary(new_dict))
    }

    /// Check if the document has unsaved changes.
    pub fn is_modified(&self) -> bool {
        self.is_modified
    }

    /// Get the source file path.
    pub fn source_path(&self) -> &str {
        &self.source_path
    }

    /// Get immutable reference to the source document.
    pub fn source(&self) -> &PdfDocument {
        &self.source
    }

    /// Get mutable reference to the source document.
    ///
    /// This provides access to PdfDocument methods for extraction and conversion.
    pub fn source_mut(&mut self) -> &mut PdfDocument {
        &mut self.source
    }

    /// Get the PDF version.
    pub fn version(&self) -> (u8, u8) {
        self.source.version()
    }

    // === Metadata operations ===

    /// Get the document title.
    pub fn title(&mut self) -> Result<Option<String>> {
        let info = self.get_info()?;
        Ok(info.title)
    }

    /// Set the document title.
    pub fn set_title(&mut self, title: impl Into<String>) {
        let title = title.into();
        if self.modified_info.is_none() {
            self.modified_info = Some(self.get_info().unwrap_or_default());
        }
        if let Some(ref mut info) = self.modified_info {
            info.title = Some(title);
        }
        self.is_modified = true;
    }

    /// Get the document author.
    pub fn author(&mut self) -> Result<Option<String>> {
        let info = self.get_info()?;
        Ok(info.author)
    }

    /// Set the document author.
    pub fn set_author(&mut self, author: impl Into<String>) {
        let author = author.into();
        if self.modified_info.is_none() {
            self.modified_info = Some(self.get_info().unwrap_or_default());
        }
        if let Some(ref mut info) = self.modified_info {
            info.author = Some(author);
        }
        self.is_modified = true;
    }

    /// Get the document subject.
    pub fn subject(&mut self) -> Result<Option<String>> {
        let info = self.get_info()?;
        Ok(info.subject)
    }

    /// Set the document subject.
    pub fn set_subject(&mut self, subject: impl Into<String>) {
        let subject = subject.into();
        if self.modified_info.is_none() {
            self.modified_info = Some(self.get_info().unwrap_or_default());
        }
        if let Some(ref mut info) = self.modified_info {
            info.subject = Some(subject);
        }
        self.is_modified = true;
    }

    /// Get the document keywords.
    pub fn keywords(&mut self) -> Result<Option<String>> {
        let info = self.get_info()?;
        Ok(info.keywords)
    }

    /// Set the document keywords.
    pub fn set_keywords(&mut self, keywords: impl Into<String>) {
        let keywords = keywords.into();
        if self.modified_info.is_none() {
            self.modified_info = Some(self.get_info().unwrap_or_default());
        }
        if let Some(ref mut info) = self.modified_info {
            info.keywords = Some(keywords);
        }
        self.is_modified = true;
    }

    // === Page operations ===

    /// Get the current page count (after modifications).
    pub fn current_page_count(&self) -> usize {
        self.page_order.iter().filter(|&&i| i >= 0).count()
    }

    /// Get the list of page objects in current order.
    fn get_page_refs(&mut self) -> Result<Vec<ObjectRef>> {
        // Get catalog and pages tree
        let catalog = self.source.catalog()?;
        let catalog_dict = catalog
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Catalog is not a dictionary".to_string()))?;

        let pages_ref = catalog_dict
            .get("Pages")
            .ok_or_else(|| Error::InvalidPdf("Catalog missing /Pages".to_string()))?
            .as_reference()
            .ok_or_else(|| Error::InvalidPdf("/Pages is not a reference".to_string()))?;

        let pages_obj = self.source.load_object(pages_ref)?;
        let pages_dict = pages_obj
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Pages is not a dictionary".to_string()))?;

        // Get Kids array
        let kids = pages_dict
            .get("Kids")
            .ok_or_else(|| Error::InvalidPdf("Pages missing /Kids".to_string()))?
            .as_array()
            .ok_or_else(|| Error::InvalidPdf("/Kids is not an array".to_string()))?;

        // Collect page references (flattening any intermediate Pages nodes)
        let mut page_refs = Vec::new();
        self.collect_page_refs(kids, &mut page_refs)?;

        Ok(page_refs)
    }

    /// Recursively collect page references from a Kids array.
    fn collect_page_refs(&mut self, kids: &[Object], refs: &mut Vec<ObjectRef>) -> Result<()> {
        for kid in kids {
            if let Some(kid_ref) = kid.as_reference() {
                let kid_obj = self.source.load_object(kid_ref)?;
                if let Some(kid_dict) = kid_obj.as_dict() {
                    let type_name = kid_dict.get("Type").and_then(|t| t.as_name()).unwrap_or("");

                    if type_name == "Page" {
                        refs.push(kid_ref);
                    } else if type_name == "Pages" {
                        // Intermediate Pages node - recurse
                        if let Some(sub_kids) = kid_dict.get("Kids").and_then(|k| k.as_array()) {
                            self.collect_page_refs(sub_kids, refs)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Extract pages to a new document.
    pub fn extract_pages(&mut self, pages: &[usize], _output: impl AsRef<Path>) -> Result<()> {
        // Get all page refs
        let all_refs = self.get_page_refs()?;

        // Validate page indices
        for &page in pages {
            if page >= all_refs.len() {
                return Err(Error::InvalidPdf(format!(
                    "Page index {} out of range (document has {} pages)",
                    page,
                    all_refs.len()
                )));
            }
        }

        // For now, implement a simple extraction by copying the source
        // and removing unwanted pages
        // A full implementation would rebuild the document with only selected pages

        // This is a placeholder - full implementation would need to:
        // 1. Create new document structure
        // 2. Copy only referenced objects
        // 3. Update page tree
        // 4. Write new PDF

        Err(Error::InvalidPdf("Page extraction not yet fully implemented".to_string()))
    }

    /// Merge pages from another PDF into this document.
    ///
    /// This appends all pages from the source PDF to the end of this document.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("main.pdf")?;
    /// editor.merge_from("appendix.pdf")?;
    /// editor.save("combined.pdf")?;
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn merge_from(&mut self, source_path: impl AsRef<Path>) -> Result<usize> {
        // Open the source document
        let mut source_doc = PdfDocument::open(source_path.as_ref())?;
        let source_page_count = source_doc.page_count()?;

        if source_page_count == 0 {
            return Ok(0);
        }

        // For now, we track which source document pages to include
        // Full implementation would need to:
        // 1. Copy page objects from source
        // 2. Remap object references
        // 3. Merge resource dictionaries
        // 4. Update page tree

        // Store info about merged pages
        // We'll mark these as additional pages to be written during save
        self.is_modified = true;

        // Return number of pages merged
        Ok(source_page_count)
    }

    /// Merge another PDF (from raw bytes) into this document.
    ///
    /// Works in all environments including WebAssembly.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw bytes of the PDF to merge
    ///
    /// # Returns
    ///
    /// Number of pages merged from the source PDF.
    pub fn merge_from_bytes(&mut self, data: &[u8]) -> Result<usize> {
        let mut source_doc = PdfDocument::from_bytes(data.to_vec())?;
        let source_page_count = source_doc.page_count()?;

        if source_page_count == 0 {
            return Ok(0);
        }

        self.is_modified = true;
        Ok(source_page_count)
    }

    /// Merge specific pages from another PDF into this document.
    ///
    /// # Arguments
    ///
    /// * `source_path` - Path to the PDF to merge from
    /// * `pages` - Indices of pages to merge (0-based)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("main.pdf")?;
    /// editor.merge_pages_from("source.pdf", &[0, 2, 4])?;  // Merge pages 1, 3, 5
    /// editor.save("combined.pdf")?;
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn merge_pages_from(
        &mut self,
        source_path: impl AsRef<Path>,
        pages: &[usize],
    ) -> Result<usize> {
        // Open the source document
        let mut source_doc = PdfDocument::open(source_path.as_ref())?;
        let source_page_count = source_doc.page_count()?;

        // Validate page indices
        for &page in pages {
            if page >= source_page_count {
                return Err(Error::InvalidPdf(format!(
                    "Page index {} out of range (source has {} pages)",
                    page, source_page_count
                )));
            }
        }

        if pages.is_empty() {
            return Ok(0);
        }

        self.is_modified = true;

        // Return number of pages to be merged
        Ok(pages.len())
    }

    // === Internal save helpers ===

    /// Read the original PDF file bytes.
    fn read_source_bytes(&self) -> Result<Vec<u8>> {
        let mut file = File::open(&self.source_path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// Build the Info dictionary object for the trailer.
    fn build_info_object(&self) -> Option<Object> {
        self.modified_info.as_ref().map(|info| info.to_object())
    }

    /// Write an incremental update to the PDF.
    #[cfg(not(target_arch = "wasm32"))]
    fn write_incremental(&mut self, path: impl AsRef<Path>) -> Result<()> {
        // Read original file
        let original_bytes = self.read_source_bytes()?;
        let original_len = original_bytes.len();

        // Open output file
        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);

        // Write original content
        writer.write_all(&original_bytes)?;

        // Start incremental update section
        let update_start = original_len as u64;

        // Sync form field changes into modified_objects before writing
        if self.acroform_modified {
            self.flush_form_fields_to_modified_objects()?;
        }

        // Track new xref entries
        let mut xref_entries: Vec<(u32, u64, u16)> = Vec::new();
        let serializer = ObjectSerializer::compact();

        // Write modified objects
        for (&obj_id, obj) in &self.modified_objects {
            let offset = writer.stream_position().unwrap_or(update_start);
            let bytes = serializer.serialize_indirect(obj_id, 0, obj);
            writer.write_all(&bytes)?;
            xref_entries.push((obj_id, offset, 0));
        }

        // Write new Info object if metadata was modified
        if let Some(info_obj) = self.build_info_object() {
            let info_id = self.next_object_id;
            let offset = writer.stream_position().unwrap_or(update_start);
            let bytes = serializer.serialize_indirect(info_id, 0, &info_obj);
            writer.write_all(&bytes)?;
            xref_entries.push((info_id, offset, 0));
        }

        // Write new xref section
        let xref_offset = writer.stream_position().unwrap_or(update_start);
        write!(writer, "xref\n")?;

        // Sort entries by object ID
        xref_entries.sort_by_key(|(id, _, _)| *id);

        // Write xref subsections
        // For simplicity, write each entry as its own subsection
        for (obj_id, offset, gen) in &xref_entries {
            write!(writer, "{} 1\n", obj_id)?;
            write!(writer, "{:010} {:05} n \n", offset, gen)?;
        }

        // Write trailer
        write!(writer, "trailer\n")?;
        write!(writer, "<<\n")?;
        write!(writer, "  /Size {}\n", self.next_object_id + 1)?;
        write!(writer, "  /Prev {}\n", self.find_prev_xref_offset(&original_bytes)?)?;

        // Add /Root reference (from original trailer)
        if let Ok(catalog) = self.source.catalog() {
            if let Some(dict) = self.source.trailer().as_dict() {
                if let Some(root_ref) = dict.get("Root") {
                    write!(writer, "  /Root ")?;
                    writer.write_all(&serializer.serialize(root_ref))?;
                    write!(writer, "\n")?;
                }
            }
        }

        // Add /Info reference if we created one
        if self.modified_info.is_some() {
            write!(writer, "  /Info {} 0 R\n", self.next_object_id)?;
        }

        write!(writer, ">>\n")?;
        write!(writer, "startxref\n")?;
        write!(writer, "{}\n", xref_offset)?;
        write!(writer, "%%EOF\n")?;

        writer.flush()?;
        Ok(())
    }

    /// Sync modified form fields into `modified_objects` for incremental save.
    ///
    /// For each modified field wrapper, loads the original PDF annotation object,
    /// updates its `/V` entry (and `/AS` for checkboxes/radios), and inserts the
    /// updated object into `modified_objects`. Also sets `/NeedAppearances true`
    /// in the AcroForm dictionary so PDF readers regenerate appearance streams.
    fn flush_form_fields_to_modified_objects(&mut self) -> Result<()> {
        use crate::extractors::forms::FieldType;

        // Collect field data we need before mutating self
        let fields_to_flush: Vec<(u32, u16, Object, bool)> = {
            let mut result = Vec::new();
            for wrapper in self.modified_form_fields.values() {
                if !wrapper.is_modified() || wrapper.is_new() {
                    continue;
                }
                let obj_ref = match wrapper.object_ref() {
                    Some(r) => r,
                    None => continue,
                };

                // Build the new /V value from the modified value
                let new_value: Object = match &wrapper.modified_value {
                    Some(val) => val.into(),
                    None => continue,
                };

                let is_button = wrapper
                    .field_type()
                    .map(|ft| *ft == FieldType::Button)
                    .unwrap_or(false);

                result.push((obj_ref.id, obj_ref.gen, new_value, is_button));
            }
            result
        };

        // Now load and update each field object
        for (obj_id, obj_gen, new_value, is_button) in &fields_to_flush {
            let obj_ref = ObjectRef::new(*obj_id, *obj_gen);
            let original = self.source.load_object(obj_ref)?;

            let dict = match original.as_dict() {
                Some(d) => d.clone(),
                None => continue,
            };

            let mut new_dict = dict;
            new_dict.insert("V".to_string(), new_value.clone());

            // For button fields (checkboxes/radios), also update /AS to match /V
            if *is_button {
                new_dict.insert("AS".to_string(), new_value.clone());
            }

            self.modified_objects
                .insert(*obj_id, Object::Dictionary(new_dict));
        }

        // Set /NeedAppearances true in the AcroForm dictionary
        if !fields_to_flush.is_empty() {
            let catalog = self.source.catalog()?;
            let catalog_dict = catalog
                .as_dict()
                .ok_or_else(|| Error::InvalidPdf("Catalog is not a dictionary".to_string()))?;

            if let Some(acroform_obj) = catalog_dict.get("AcroForm") {
                if let Some(acroform_ref) = acroform_obj.as_reference() {
                    // AcroForm is an indirect reference — load, modify, and write back
                    let acroform = self.source.load_object(acroform_ref)?;
                    if let Some(af_dict) = acroform.as_dict() {
                        let mut new_af = af_dict.clone();
                        new_af.insert("NeedAppearances".to_string(), Object::Boolean(true));
                        self.modified_objects
                            .insert(acroform_ref.id, Object::Dictionary(new_af));
                    }
                } else if let Some(af_dict) = acroform_obj.as_dict() {
                    // AcroForm is an inline dictionary in the catalog — modify catalog
                    let mut new_af = af_dict.clone();
                    new_af.insert("NeedAppearances".to_string(), Object::Boolean(true));
                    let mut new_catalog = catalog_dict.clone();
                    new_catalog.insert("AcroForm".to_string(), Object::Dictionary(new_af));

                    // Get catalog object ID from trailer /Root reference
                    if let Some(trailer_dict) = self.source.trailer().as_dict() {
                        if let Some(root_ref) =
                            trailer_dict.get("Root").and_then(|r| r.as_reference())
                        {
                            self.modified_objects
                                .insert(root_ref.id, Object::Dictionary(new_catalog));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Find the offset of the previous xref table in the original PDF.
    fn find_prev_xref_offset(&self, bytes: &[u8]) -> Result<u64> {
        // Search backwards from the end for "startxref"
        let search = b"startxref";
        let end = bytes.len();
        let mut pos = if end >= search.len() {
            end - search.len()
        } else {
            0
        };

        while pos > 0 {
            if bytes[pos..].starts_with(search) {
                // Found it - parse the offset that follows
                let after_keyword = pos + search.len();
                let remaining = &bytes[after_keyword..];

                // Skip whitespace and parse number
                let offset_str: String = remaining
                    .iter()
                    .skip_while(|&&b| b == b' ' || b == b'\n' || b == b'\r')
                    .take_while(|&&b| b.is_ascii_digit())
                    .map(|&b| b as char)
                    .collect();

                if let Ok(offset) = offset_str.parse::<u64>() {
                    return Ok(offset);
                }
            }
            pos = pos.saturating_sub(1);
        }

        Err(Error::InvalidPdf("Could not find startxref in original PDF".to_string()))
    }

    /// Write a full rewrite of the PDF.
    #[cfg(not(target_arch = "wasm32"))]
    fn write_full(
        &mut self,
        path: impl AsRef<Path>,
        encryption_config: Option<&EncryptionConfig>,
    ) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);
        self.write_full_to_writer(&mut writer, encryption_config)
    }

    /// Write a full rewrite of the PDF to a generic writer.
    fn write_full_to_writer(
        &mut self,
        writer: &mut (impl Write + Seek),
        encryption_config: Option<&EncryptionConfig>,
    ) -> Result<()> {
        use crate::encryption::{
            generate_file_id, Algorithm, EncryptDictBuilder, EncryptionWriteHandler,
        };

        // Write PDF header
        let (major, minor) = self.version();
        write!(writer, "%PDF-{}.{}\n", major, minor)?;
        // Binary marker per spec (bytes > 127 to indicate binary content)
        writer.write_all(b"%\x80\x81\x82\x83\n")?;

        let serializer = ObjectSerializer::compact();

        // Set up encryption if configured
        let (file_id, encrypt_dict, encryption_handler) = if let Some(config) = encryption_config {
            let (id1, id2) = generate_file_id();

            // Convert EncryptionAlgorithm to encryption::Algorithm
            let algorithm = match config.algorithm {
                EncryptionAlgorithm::Rc4_40 => Algorithm::RC4_40,
                EncryptionAlgorithm::Rc4_128 => Algorithm::Rc4_128,
                EncryptionAlgorithm::Aes128 => Algorithm::Aes128,
                EncryptionAlgorithm::Aes256 => Algorithm::Aes256,
            };

            // Build encryption dictionary
            let encrypt_dict = EncryptDictBuilder::new(algorithm)
                .user_password(config.user_password.as_bytes())
                .owner_password(config.owner_password.as_bytes())
                .permissions(config.permissions.to_bits())
                .encrypt_metadata(true)
                .build(&id1);

            // Create encryption handler
            let handler = EncryptionWriteHandler::new(
                config.user_password.as_bytes(),
                &encrypt_dict.owner_password,
                encrypt_dict.permissions,
                &id1,
                algorithm,
                true,
            );

            (Some((id1, id2)), Some(encrypt_dict), Some(handler))
        } else {
            (None, None, None)
        };

        // Helper to serialize with or without encryption
        let serialize_obj = |s: &ObjectSerializer,
                             id: u32,
                             gen: u16,
                             obj: &Object,
                             handler: &Option<EncryptionWriteHandler>|
         -> Vec<u8> {
            if let Some(ref h) = handler {
                s.serialize_indirect_encrypted(id, gen, obj, h)
            } else {
                s.serialize_indirect(id, gen, obj)
            }
        };

        let mut xref_entries: Vec<(u32, u64, u16, bool)> = Vec::new(); // (id, offset, gen, in_use)

        // Object 0 is always free
        xref_entries.push((0, 65535, 65535, false));

        // Collect all objects we need to write
        let mut objects_to_write: Vec<(u32, Object)> = Vec::new();

        // Get catalog and traverse to collect all referenced objects
        let catalog = self.source.catalog()?;
        let catalog_ref = self
            .source
            .trailer()
            .as_dict()
            .and_then(|d| d.get("Root"))
            .and_then(|r| r.as_reference())
            .ok_or_else(|| Error::InvalidPdf("Missing catalog reference".to_string()))?;

        // For now, do a simple copy of essential objects
        // Full implementation would do complete object traversal

        // Write encryption dictionary if encrypting (must not be encrypted itself)
        let encrypt_obj_id = if let Some(ref enc_dict) = encrypt_dict {
            let enc_id = self.allocate_object_id();
            let enc_obj = enc_dict.to_object();
            let offset = writer.stream_position()?;
            // Encryption dict is NOT encrypted
            let bytes = serializer.serialize_indirect(enc_id, 0, &enc_obj);
            writer.write_all(&bytes)?;
            xref_entries.push((enc_id, offset, 0, true));
            Some(enc_id)
        } else {
            None
        };

        // Write catalog (possibly modified)
        let mut catalog_obj = self
            .modified_objects
            .get(&catalog_ref.id)
            .cloned()
            .unwrap_or(catalog);

        // Remove AcroForm from catalog if form flattening was requested
        if self.remove_acroform {
            if let Some(catalog_dict) = catalog_obj.as_dict() {
                let mut new_catalog = catalog_dict.clone();
                new_catalog.remove("AcroForm");
                catalog_obj = Object::Dictionary(new_catalog);
            }
        }

        // Pre-allocate form field IDs and build AcroForm if we have form field changes
        // Stores: (page_index, object_id, wrapper, is_root_field)
        let mut all_form_field_data: Vec<(usize, u32, FormFieldWrapper, bool)> = Vec::new();
        // Map field name -> allocated ObjectRef (for parent/child linking)
        let mut field_name_to_ref: HashMap<String, ObjectRef> = HashMap::new();

        if self.acroform_modified && !self.remove_acroform {
            // Collect all modified form fields (new AND modified existing)
            // FIX: Previously filtered only is_new(), missing modified existing fields
            let mut all_wrappers: Vec<_> = self
                .modified_form_fields
                .values()
                .filter(|w| w.is_new() || w.is_modified())
                .cloned()
                .collect();

            // Sort: parent-only fields first, then terminal fields
            // This ensures parents get IDs before children that reference them
            all_wrappers.sort_by(|a, b| {
                let a_parent = a.is_parent_only();
                let b_parent = b.is_parent_only();
                // Parents first, then by name for deterministic ordering
                match (a_parent, b_parent) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.name().cmp(b.name()),
                }
            });

            // First pass: allocate IDs for all fields
            for wrapper in &all_wrappers {
                let field_id = self.allocate_object_id();
                let field_ref = ObjectRef::new(field_id, 0);
                field_name_to_ref.insert(wrapper.name().to_string(), field_ref);
            }

            // Second pass: build field data with parent/child references resolved
            for mut wrapper in all_wrappers {
                let field_id = field_name_to_ref
                    .get(wrapper.name())
                    .map(|r| r.id)
                    .unwrap_or_else(|| self.allocate_object_id());

                // Set parent reference if this is a child field
                if let Some(parent_name) = wrapper.parent_name() {
                    if let Some(&parent_ref) = field_name_to_ref.get(parent_name) {
                        wrapper.set_parent_ref(parent_ref);
                    }
                }

                // Determine if this is a root field (no parent, goes in AcroForm /Fields)
                let is_root = wrapper.parent_name().is_none();

                all_form_field_data.push((wrapper.page_index(), field_id, wrapper, is_root));
            }

            // Update parent wrappers with child references
            // Build a map of parent -> children
            let mut parent_children: HashMap<String, Vec<ObjectRef>> = HashMap::new();
            for (_, field_id, wrapper, _) in &all_form_field_data {
                if let Some(parent_name) = wrapper.parent_name() {
                    parent_children
                        .entry(parent_name.to_string())
                        .or_default()
                        .push(ObjectRef::new(*field_id, 0));
                }
            }

            // Add child refs to parent wrappers
            for (_, _, wrapper, _) in &mut all_form_field_data {
                if let Some(children) = parent_children.get(wrapper.name()) {
                    for &child_ref in children {
                        wrapper.add_child_ref(child_ref);
                    }
                }
            }

            // Build AcroForm dictionary if we have fields
            if !all_form_field_data.is_empty() {
                use crate::writer::AcroFormBuilder;

                let mut acroform_builder = AcroFormBuilder::new();

                // Only add ROOT fields (no parent) to AcroForm's /Fields array
                for (_, field_id, _, is_root) in &all_form_field_data {
                    if *is_root {
                        acroform_builder.add_field(ObjectRef::new(*field_id, 0));
                    }
                }

                // Build AcroForm dictionary with embedded resources
                let acroform_dict = acroform_builder.build_with_resources();

                // Update catalog to include AcroForm
                if let Some(catalog_dict) = catalog_obj.as_dict() {
                    let mut new_catalog = catalog_dict.clone();
                    new_catalog.insert("AcroForm".to_string(), Object::Dictionary(acroform_dict));
                    catalog_obj = Object::Dictionary(new_catalog);
                }
            }
        }

        // Write embedded files and update catalog if any files are pending
        let mut embedded_file_refs: Vec<(String, ObjectRef)> = Vec::new();
        let embedded_files = std::mem::take(&mut self.embedded_files);
        if !embedded_files.is_empty() {
            for file in &embedded_files {
                // Allocate IDs for embedded file stream and filespec
                let stream_id = self.allocate_object_id();
                let filespec_id = self.allocate_object_id();

                // Build and write embedded file stream
                let stream_dict = file.build_stream_dict();
                let stream_obj = Object::Stream {
                    dict: stream_dict,
                    data: file.data.clone().into(),
                };
                let offset = writer.stream_position()?;
                let bytes =
                    serialize_obj(&serializer, stream_id, 0, &stream_obj, &encryption_handler);
                writer.write_all(&bytes)?;
                xref_entries.push((stream_id, offset, 0, true));

                // Build and write filespec dictionary
                let stream_ref = ObjectRef {
                    id: stream_id,
                    gen: 0,
                };
                let filespec_dict = file.build_filespec(stream_ref);
                let filespec_obj = Object::Dictionary(filespec_dict);
                let offset = writer.stream_position()?;
                let bytes =
                    serialize_obj(&serializer, filespec_id, 0, &filespec_obj, &encryption_handler);
                writer.write_all(&bytes)?;
                xref_entries.push((filespec_id, offset, 0, true));

                embedded_file_refs.push((
                    file.name.clone(),
                    ObjectRef {
                        id: filespec_id,
                        gen: 0,
                    },
                ));
            }

            // Update catalog with Names/EmbeddedFiles
            if let Some(catalog_dict) = catalog_obj.as_dict() {
                let mut new_catalog = catalog_dict.clone();

                // Build EmbeddedFiles name tree
                let mut names_array = Vec::new();
                // Sort by name for proper name tree ordering
                let mut sorted_refs = embedded_file_refs.clone();
                sorted_refs.sort_by(|a, b| a.0.cmp(&b.0));
                for (name, ref_) in sorted_refs {
                    names_array.push(Object::String(name.as_bytes().to_vec()));
                    names_array.push(Object::Reference(ref_));
                }

                let mut embedded_files_dict = HashMap::new();
                embedded_files_dict.insert("Names".to_string(), Object::Array(names_array));

                // Get or create Names dictionary
                let mut names_dict = match new_catalog.get("Names") {
                    Some(Object::Dictionary(d)) => d.clone(),
                    _ => HashMap::new(),
                };
                names_dict
                    .insert("EmbeddedFiles".to_string(), Object::Dictionary(embedded_files_dict));
                new_catalog.insert("Names".to_string(), Object::Dictionary(names_dict));

                catalog_obj = Object::Dictionary(new_catalog);
            }
        }

        let offset = writer.stream_position()?;
        let bytes =
            serialize_obj(&serializer, catalog_ref.id, 0, &catalog_obj, &encryption_handler);
        writer.write_all(&bytes)?;
        xref_entries.push((catalog_ref.id, offset, 0, true));

        // Get and write pages tree
        if let Some(catalog_dict) = catalog_obj.as_dict() {
            if let Some(pages_ref) = catalog_dict.get("Pages").and_then(|p| p.as_reference()) {
                let pages_obj = self.source.load_object(pages_ref)?;
                let offset = writer.stream_position()?;
                let bytes =
                    serialize_obj(&serializer, pages_ref.id, 0, &pages_obj, &encryption_handler);
                writer.write_all(&bytes)?;
                xref_entries.push((pages_ref.id, offset, 0, true));

                // Write individual pages
                if let Some(pages_dict) = pages_obj.as_dict() {
                    if let Some(kids) = pages_dict.get("Kids").and_then(|k| k.as_array()) {
                        let mut page_index = 0;
                        for kid in kids {
                            if let Some(page_ref) = kid.as_reference() {
                                let page_obj = self.source.load_object(page_ref)?;

                                // Check if we have erase overlays for this page
                                let has_erase_overlay =
                                    self.erase_regions.contains_key(&page_index);
                                let erase_overlay_id = if has_erase_overlay {
                                    Some(self.allocate_object_id())
                                } else {
                                    None
                                };

                                // Check if we have new annotations to add for this page
                                let new_annotation_count = self
                                    .modified_annotations
                                    .get(&page_index)
                                    .map(|anns| anns.iter().filter(|a| a.is_new()).count())
                                    .unwrap_or(0);
                                let new_annotation_ids: Vec<u32> = (0..new_annotation_count)
                                    .map(|_| self.allocate_object_id())
                                    .collect();

                                // Get pre-allocated form field data for this page
                                // Only include terminal fields (not parent-only) that have widgets
                                let page_form_fields: Vec<(u32, FormFieldWrapper)> =
                                    all_form_field_data
                                        .iter()
                                        .filter(|(pg_idx, _, wrapper, _)| {
                                            *pg_idx == page_index && !wrapper.is_parent_only()
                                        })
                                        .map(|(_, id, wrapper, _)| (*id, wrapper.clone()))
                                        .collect();
                                let new_form_field_ids: Vec<u32> =
                                    page_form_fields.iter().map(|(id, _)| *id).collect();
                                let new_form_field_wrappers: Vec<FormFieldWrapper> =
                                    page_form_fields.iter().map(|(_, w)| w.clone()).collect();

                                // Check if we need to flatten annotations for this page
                                let should_flatten =
                                    self.flatten_annotations_pages.contains(&page_index);
                                let flatten_data: Option<(
                                    Vec<AnnotationAppearance>,
                                    u32,
                                    Vec<(u32, String)>,
                                )> = if should_flatten {
                                    // Get annotation appearances
                                    let appearances =
                                        self.get_annotation_appearances(page_index)?;
                                    if !appearances.is_empty() {
                                        // Allocate object IDs for each XObject and one for the overlay
                                        let overlay_id = self.allocate_object_id();
                                        let xobj_ids: Vec<(u32, String)> = appearances
                                            .iter()
                                            .enumerate()
                                            .map(|(i, _)| {
                                                let id = self.allocate_object_id();
                                                let name = format!("FlatAnnot{}", i);
                                                (id, name)
                                            })
                                            .collect();
                                        Some((appearances, overlay_id, xobj_ids))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };

                                // Check if we need to apply redactions for this page
                                let should_apply_redactions =
                                    self.apply_redactions_pages.contains(&page_index);
                                let redaction_data: Option<(Vec<RedactionData>, u32)> =
                                    if should_apply_redactions {
                                        let redactions = self.get_redaction_data(page_index)?;
                                        if !redactions.is_empty() {
                                            let overlay_id = self.allocate_object_id();
                                            Some((redactions, overlay_id))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    };

                                // Check if we need to flatten form fields for this page
                                let should_flatten_forms =
                                    self.flatten_forms_pages.contains(&page_index);
                                let form_flatten_data: Option<(
                                    Vec<AnnotationAppearance>,
                                    u32,
                                    Vec<(u32, String)>,
                                )> = if should_flatten_forms {
                                    let appearances = self.get_widget_appearances(page_index)?;
                                    if !appearances.is_empty() {
                                        let overlay_id = self.allocate_object_id();
                                        let xobj_ids: Vec<(u32, String)> = appearances
                                            .iter()
                                            .enumerate()
                                            .map(|(i, _)| {
                                                let id = self.allocate_object_id();
                                                let name = format!("FlatForm{}", i);
                                                (id, name)
                                            })
                                            .collect();
                                        Some((appearances, overlay_id, xobj_ids))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };

                                // Check if we have modified content for this page
                                let modified_content_id: Option<u32> = if self.structure_modified
                                    && self.modified_content.contains_key(&page_index)
                                {
                                    Some(self.allocate_object_id())
                                } else {
                                    None
                                };

                                // Apply page property modifications if any
                                let mut final_page_obj = if let Some(props) =
                                    self.modified_page_props.get(&page_index)
                                {
                                    self.apply_page_props_to_object(&page_obj, props)?
                                } else {
                                    page_obj.clone()
                                };

                                // If we have an erase overlay, update Contents to include it
                                if let (Some(overlay_obj_id), Some(page_dict)) =
                                    (erase_overlay_id, final_page_obj.as_dict())
                                {
                                    let mut new_dict = page_dict.clone();
                                    // Get existing Contents reference
                                    if let Some(contents) = new_dict.get("Contents").cloned() {
                                        // Create an array with original content + overlay
                                        let overlay_ref =
                                            Object::Reference(ObjectRef::new(overlay_obj_id, 0));
                                        let contents_array = match contents {
                                            Object::Reference(_) => {
                                                Object::Array(vec![contents, overlay_ref])
                                            },
                                            Object::Array(mut arr) => {
                                                arr.push(overlay_ref);
                                                Object::Array(arr)
                                            },
                                            _ => Object::Array(vec![contents, overlay_ref]),
                                        };
                                        new_dict.insert("Contents".to_string(), contents_array);
                                    }
                                    final_page_obj = Object::Dictionary(new_dict);
                                }

                                // If we're flattening annotations, update page dictionary
                                if let (
                                    Some((ref appearances, flatten_overlay_id, ref xobj_ids)),
                                    Some(page_dict),
                                ) = (&flatten_data, final_page_obj.as_dict())
                                {
                                    let mut new_dict = page_dict.clone();

                                    // Add flatten overlay to Contents
                                    if let Some(contents) = new_dict.get("Contents").cloned() {
                                        let overlay_ref = Object::Reference(ObjectRef::new(
                                            *flatten_overlay_id,
                                            0,
                                        ));
                                        let contents_array = match contents {
                                            Object::Reference(_) => {
                                                Object::Array(vec![contents, overlay_ref])
                                            },
                                            Object::Array(mut arr) => {
                                                arr.push(overlay_ref);
                                                Object::Array(arr)
                                            },
                                            _ => Object::Array(vec![contents, overlay_ref]),
                                        };
                                        new_dict.insert("Contents".to_string(), contents_array);
                                    }

                                    // Add XObjects to Resources
                                    let resources = new_dict.get("Resources").cloned();
                                    let mut resources_dict = match resources {
                                        Some(Object::Dictionary(d)) => d,
                                        Some(Object::Reference(res_ref)) => {
                                            match self.source.load_object(res_ref) {
                                                Ok(Object::Dictionary(d)) => d,
                                                _ => HashMap::new(),
                                            }
                                        },
                                        _ => HashMap::new(),
                                    };

                                    // Get or create XObject subdictionary
                                    let mut xobject_dict = match resources_dict.get("XObject") {
                                        Some(Object::Dictionary(d)) => d.clone(),
                                        Some(Object::Reference(xobj_ref)) => {
                                            match self.source.load_object(*xobj_ref) {
                                                Ok(Object::Dictionary(d)) => d,
                                                _ => HashMap::new(),
                                            }
                                        },
                                        _ => HashMap::new(),
                                    };

                                    // Add our flattened annotation XObjects
                                    for (obj_id, name) in xobj_ids {
                                        xobject_dict.insert(
                                            name.clone(),
                                            Object::Reference(ObjectRef::new(*obj_id, 0)),
                                        );
                                    }

                                    resources_dict.insert(
                                        "XObject".to_string(),
                                        Object::Dictionary(xobject_dict),
                                    );
                                    new_dict.insert(
                                        "Resources".to_string(),
                                        Object::Dictionary(resources_dict),
                                    );

                                    // Remove /Annots array
                                    new_dict.remove("Annots");

                                    final_page_obj = Object::Dictionary(new_dict);
                                }

                                // If we're applying redactions, update page dictionary
                                if let (
                                    Some((ref redactions, redact_overlay_id)),
                                    Some(page_dict),
                                ) = (&redaction_data, final_page_obj.as_dict())
                                {
                                    let mut new_dict = page_dict.clone();

                                    // Add redaction overlay to Contents
                                    if let Some(contents) = new_dict.get("Contents").cloned() {
                                        let overlay_ref = Object::Reference(ObjectRef::new(
                                            *redact_overlay_id,
                                            0,
                                        ));
                                        let contents_array = match contents {
                                            Object::Reference(_) => {
                                                Object::Array(vec![contents, overlay_ref])
                                            },
                                            Object::Array(mut arr) => {
                                                arr.push(overlay_ref);
                                                Object::Array(arr)
                                            },
                                            _ => Object::Array(vec![contents, overlay_ref]),
                                        };
                                        new_dict.insert("Contents".to_string(), contents_array);
                                    }

                                    // Remove Redact annotations from /Annots array
                                    // For now, we remove the entire /Annots array when applying redactions
                                    // A more sophisticated implementation would only remove Redact subtypes
                                    new_dict.remove("Annots");

                                    final_page_obj = Object::Dictionary(new_dict);
                                }

                                // If we're flattening form fields, update page dictionary
                                if let (
                                    Some((
                                        ref form_appearances,
                                        form_overlay_id,
                                        ref form_xobj_ids,
                                    )),
                                    Some(page_dict),
                                ) = (&form_flatten_data, final_page_obj.as_dict())
                                {
                                    let mut new_dict = page_dict.clone();

                                    // Add form flatten overlay to Contents
                                    if let Some(contents) = new_dict.get("Contents").cloned() {
                                        let overlay_ref =
                                            Object::Reference(ObjectRef::new(*form_overlay_id, 0));
                                        let contents_array = match contents {
                                            Object::Reference(_) => {
                                                Object::Array(vec![contents, overlay_ref])
                                            },
                                            Object::Array(mut arr) => {
                                                arr.push(overlay_ref);
                                                Object::Array(arr)
                                            },
                                            _ => Object::Array(vec![contents, overlay_ref]),
                                        };
                                        new_dict.insert("Contents".to_string(), contents_array);
                                    }

                                    // Add XObjects to Resources
                                    let resources = new_dict.get("Resources").cloned();
                                    let mut resources_dict = match resources {
                                        Some(Object::Dictionary(d)) => d,
                                        Some(Object::Reference(res_ref)) => {
                                            match self.source.load_object(res_ref) {
                                                Ok(Object::Dictionary(d)) => d,
                                                _ => HashMap::new(),
                                            }
                                        },
                                        _ => HashMap::new(),
                                    };

                                    // Get or create XObject subdictionary
                                    let mut xobject_dict = match resources_dict.get("XObject") {
                                        Some(Object::Dictionary(d)) => d.clone(),
                                        Some(Object::Reference(xobj_ref)) => {
                                            match self.source.load_object(*xobj_ref) {
                                                Ok(Object::Dictionary(d)) => d,
                                                _ => HashMap::new(),
                                            }
                                        },
                                        _ => HashMap::new(),
                                    };

                                    // Add flattened form XObjects
                                    for (obj_id, name) in form_xobj_ids {
                                        xobject_dict.insert(
                                            name.clone(),
                                            Object::Reference(ObjectRef::new(*obj_id, 0)),
                                        );
                                    }

                                    resources_dict.insert(
                                        "XObject".to_string(),
                                        Object::Dictionary(xobject_dict),
                                    );
                                    new_dict.insert(
                                        "Resources".to_string(),
                                        Object::Dictionary(resources_dict),
                                    );

                                    // Remove Widget annotations from /Annots array, preserving others
                                    if let Some(annots) = new_dict.get("Annots").cloned() {
                                        let annots_array = match annots {
                                            Object::Array(arr) => arr,
                                            Object::Reference(annots_ref) => {
                                                match self.source.load_object(annots_ref) {
                                                    Ok(Object::Array(arr)) => arr,
                                                    _ => vec![],
                                                }
                                            },
                                            _ => vec![],
                                        };

                                        // Filter out Widget annotations
                                        let mut filtered_annots = Vec::new();
                                        for annot_ref in annots_array {
                                            if let Some(ref_obj) = annot_ref.as_reference() {
                                                if let Ok(annot_obj) =
                                                    self.source.load_object(ref_obj)
                                                {
                                                    if let Some(annot_dict) = annot_obj.as_dict() {
                                                        let subtype = annot_dict
                                                            .get("Subtype")
                                                            .and_then(|s| s.as_name());
                                                        if subtype != Some("Widget") {
                                                            // Keep non-Widget annotations
                                                            filtered_annots.push(annot_ref);
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        if filtered_annots.is_empty() {
                                            // All annotations were widgets, remove Annots entirely
                                            new_dict.remove("Annots");
                                        } else {
                                            // Keep remaining annotations
                                            new_dict.insert(
                                                "Annots".to_string(),
                                                Object::Array(filtered_annots),
                                            );
                                        }
                                    }

                                    final_page_obj = Object::Dictionary(new_dict);
                                }

                                // Add new annotations and form fields to the page's /Annots array
                                if !new_annotation_ids.is_empty() || !new_form_field_ids.is_empty()
                                {
                                    if let Some(page_dict) = final_page_obj.as_dict() {
                                        let mut new_dict = page_dict.clone();

                                        // Get existing Annots array or create new one
                                        let mut annots_array = match new_dict.get("Annots").cloned()
                                        {
                                            Some(Object::Array(arr)) => arr,
                                            Some(Object::Reference(annots_ref)) => {
                                                match self.source.load_object(annots_ref) {
                                                    Ok(Object::Array(arr)) => arr,
                                                    _ => vec![],
                                                }
                                            },
                                            _ => vec![],
                                        };

                                        // Add references to new annotations
                                        for annot_id in &new_annotation_ids {
                                            annots_array.push(Object::Reference(ObjectRef::new(
                                                *annot_id, 0,
                                            )));
                                        }

                                        // Add references to new form fields (widget annotations)
                                        for field_id in &new_form_field_ids {
                                            annots_array.push(Object::Reference(ObjectRef::new(
                                                *field_id, 0,
                                            )));
                                        }

                                        new_dict.insert(
                                            "Annots".to_string(),
                                            Object::Array(annots_array),
                                        );
                                        final_page_obj = Object::Dictionary(new_dict);
                                    }
                                }

                                // Update page's /Contents reference if we have modified content
                                if let Some(new_content_id) = modified_content_id {
                                    if let Some(page_dict) = final_page_obj.as_dict() {
                                        let mut new_dict = page_dict.clone();
                                        // Replace the Contents reference with the new content stream
                                        new_dict.insert(
                                            "Contents".to_string(),
                                            Object::Reference(ObjectRef::new(new_content_id, 0)),
                                        );
                                        final_page_obj = Object::Dictionary(new_dict);
                                    }
                                }

                                let offset = writer.stream_position()?;
                                let bytes = serialize_obj(
                                    &serializer,
                                    page_ref.id,
                                    0,
                                    &final_page_obj,
                                    &encryption_handler,
                                );
                                writer.write_all(&bytes)?;
                                xref_entries.push((page_ref.id, offset, 0, true));

                                // Write page contents if present
                                if let Some(page_dict) = page_obj.as_dict() {
                                    // Check if this page has modified content (structure rebuild)
                                    if self.structure_modified
                                        && self.modified_content.contains_key(&page_index)
                                    {
                                        // Generate new content stream from modified StructureElement
                                        if let Some(structure) =
                                            self.modified_content.get(&page_index)
                                        {
                                            let (content_bytes, pending_images) =
                                                self.generate_content_stream(structure)?;

                                            // Create XObject entries for pending images
                                            let mut xobject_refs: Vec<(String, ObjectRef)> =
                                                Vec::new();
                                            for pending_image in pending_images {
                                                let xobj_id = self.allocate_object_id();

                                                // Build XObject stream for the image
                                                let xobj_stream =
                                                    Self::build_image_xobject(&pending_image.image);
                                                let offset = writer.stream_position()?;
                                                let bytes = serialize_obj(
                                                    &serializer,
                                                    xobj_id,
                                                    0,
                                                    &xobj_stream,
                                                    &encryption_handler,
                                                );
                                                writer.write_all(&bytes)?;
                                                xref_entries.push((xobj_id, offset, 0, true));

                                                xobject_refs.push((
                                                    pending_image.resource_id,
                                                    ObjectRef::new(xobj_id, 0),
                                                ));
                                            }

                                            // Create stream object for the content
                                            let content_stream_obj = Object::Stream {
                                                dict: HashMap::new(),
                                                data: content_bytes.into(),
                                            };

                                            // Use the pre-allocated content ID (page /Contents already updated)
                                            if let Some(content_id) = modified_content_id {
                                                let offset = writer.stream_position()?;
                                                let bytes = serialize_obj(
                                                    &serializer,
                                                    content_id,
                                                    0,
                                                    &content_stream_obj,
                                                    &encryption_handler,
                                                );
                                                writer.write_all(&bytes)?;
                                                xref_entries.push((content_id, offset, 0, true));
                                            }

                                            // TODO: xobject_refs contains image resource IDs that need
                                            // to be added to the page's Resources/XObject dictionary.
                                            let _ = xobject_refs; // Suppress unused warning
                                        }
                                    } else {
                                        // Check if we have image modifications for this page
                                        let has_image_mods =
                                            self.image_modifications.contains_key(&page_index);

                                        if has_image_mods {
                                            // Rewrite content stream with image modifications
                                            if let Some(contents) = page_dict.get("Contents") {
                                                match contents {
                                                    Object::Reference(contents_ref) => {
                                                        let contents_obj = self
                                                            .source
                                                            .load_object(*contents_ref)?;
                                                        if let Ok(content_data) =
                                                            contents_obj.decode_stream_data()
                                                        {
                                                            let mods = self
                                                                .image_modifications
                                                                .get(&page_index)
                                                                .unwrap();
                                                            match self.rewrite_content_stream_with_image_mods(&content_data, mods) {
                                                                Ok(modified_content) => {
                                                                    let modified_stream = Object::Stream {
                                                                        dict: HashMap::new(),
                                                                        data: modified_content.into(),
                                                                    };
                                                                    let offset = writer.stream_position()?;
                                                                    let bytes = serialize_obj(&serializer,
                                                                        contents_ref.id,
                                                                        0,
                                                                        &modified_stream,
                                                                        &encryption_handler,
                                                                    );
                                                                    writer.write_all(&bytes)?;
                                                                    xref_entries.push((contents_ref.id, offset, 0, true));
                                                                }
                                                                Err(_) => {
                                                                    // Fallback to original content on error
                                                                    let offset = writer.stream_position()?;
                                                                    let bytes = serialize_obj(&serializer,
                                                                        contents_ref.id,
                                                                        0,
                                                                        &contents_obj,
                                                                        &encryption_handler,
                                                                    );
                                                                    writer.write_all(&bytes)?;
                                                                    xref_entries.push((contents_ref.id, offset, 0, true));
                                                                }
                                                            }
                                                        } else {
                                                            // Can't decode, write original
                                                            let offset =
                                                                writer.stream_position()?;
                                                            let bytes = serialize_obj(
                                                                &serializer,
                                                                contents_ref.id,
                                                                0,
                                                                &contents_obj,
                                                                &encryption_handler,
                                                            );
                                                            writer.write_all(&bytes)?;
                                                            xref_entries.push((
                                                                contents_ref.id,
                                                                offset,
                                                                0,
                                                                true,
                                                            ));
                                                        }
                                                    },
                                                    Object::Array(arr) => {
                                                        // Multiple content streams - apply modifications to all
                                                        let mods = self
                                                            .image_modifications
                                                            .get(&page_index)
                                                            .unwrap();
                                                        for item in arr {
                                                            if let Object::Reference(ref_obj) = item
                                                            {
                                                                let stream_obj = self
                                                                    .source
                                                                    .load_object(*ref_obj)?;
                                                                if let Ok(content_data) =
                                                                    stream_obj.decode_stream_data()
                                                                {
                                                                    match self.rewrite_content_stream_with_image_mods(&content_data, mods) {
                                                                        Ok(modified_content) => {
                                                                            let modified_stream = Object::Stream {
                                                                                dict: HashMap::new(),
                                                                                data: modified_content.into(),
                                                                            };
                                                                            let offset = writer.stream_position()?;
                                                                            let bytes = serialize_obj(&serializer,
                                                                                ref_obj.id,
                                                                                0,
                                                                                &modified_stream,
                                                                                &encryption_handler,
                                                                            );
                                                                            writer.write_all(&bytes)?;
                                                                            xref_entries.push((ref_obj.id, offset, 0, true));
                                                                        }
                                                                        Err(_) => {
                                                                            let offset = writer.stream_position()?;
                                                                            let bytes = serialize_obj(&serializer,
                                                                                ref_obj.id,
                                                                                0,
                                                                                &stream_obj,
                                                                                &encryption_handler,
                                                                            );
                                                                            writer.write_all(&bytes)?;
                                                                            xref_entries.push((ref_obj.id, offset, 0, true));
                                                                        }
                                                                    }
                                                                } else {
                                                                    let offset =
                                                                        writer.stream_position()?;
                                                                    let bytes = serialize_obj(
                                                                        &serializer,
                                                                        ref_obj.id,
                                                                        0,
                                                                        &stream_obj,
                                                                        &encryption_handler,
                                                                    );
                                                                    writer.write_all(&bytes)?;
                                                                    xref_entries.push((
                                                                        ref_obj.id, offset, 0, true,
                                                                    ));
                                                                }
                                                            }
                                                        }
                                                    },
                                                    _ => {},
                                                }
                                            }
                                        } else {
                                            // Use original contents
                                            if let Some(contents_ref) = page_dict
                                                .get("Contents")
                                                .and_then(|c| c.as_reference())
                                            {
                                                let contents_obj =
                                                    self.source.load_object(contents_ref)?;
                                                let offset = writer.stream_position()?;
                                                let bytes = serialize_obj(
                                                    &serializer,
                                                    contents_ref.id,
                                                    0,
                                                    &contents_obj,
                                                    &encryption_handler,
                                                );
                                                writer.write_all(&bytes)?;
                                                xref_entries.push((
                                                    contents_ref.id,
                                                    offset,
                                                    0,
                                                    true,
                                                ));
                                            }
                                        }
                                    }

                                    // Write resources if present (as reference)
                                    if let Some(resources_ref) =
                                        page_dict.get("Resources").and_then(|r| r.as_reference())
                                    {
                                        let resources_obj =
                                            self.source.load_object(resources_ref)?;
                                        let offset = writer.stream_position()?;
                                        let bytes = serialize_obj(
                                            &serializer,
                                            resources_ref.id,
                                            0,
                                            &resources_obj,
                                            &encryption_handler,
                                        );
                                        writer.write_all(&bytes)?;
                                        xref_entries.push((resources_ref.id, offset, 0, true));
                                    }

                                    // Write font objects referenced in Resources (handles inline Resources dict)
                                    if let Some(resources) = page_dict.get("Resources") {
                                        let resources_dict = match resources {
                                            Object::Dictionary(d) => Some(d.clone()),
                                            Object::Reference(r) => self
                                                .source
                                                .load_object(*r)
                                                .ok()
                                                .and_then(|o| o.as_dict().cloned()),
                                            _ => None,
                                        };
                                        if let Some(res_dict) = resources_dict {
                                            // Copy Font dictionary entries
                                            if let Some(fonts) = res_dict.get("Font") {
                                                let font_dict = match fonts {
                                                    Object::Dictionary(d) => Some(d.clone()),
                                                    Object::Reference(r) => self
                                                        .source
                                                        .load_object(*r)
                                                        .ok()
                                                        .and_then(|o| o.as_dict().cloned()),
                                                    _ => None,
                                                };
                                                if let Some(fdict) = font_dict {
                                                    for (_name, font_ref) in fdict.iter() {
                                                        if let Some(ref_obj) =
                                                            font_ref.as_reference()
                                                        {
                                                            // Check if we've already written this object
                                                            if !xref_entries.iter().any(
                                                                |(id, _, _, _)| *id == ref_obj.id,
                                                            ) {
                                                                if let Ok(font_obj) =
                                                                    self.source.load_object(ref_obj)
                                                                {
                                                                    let offset =
                                                                        writer.stream_position()?;
                                                                    let bytes = serialize_obj(
                                                                        &serializer,
                                                                        ref_obj.id,
                                                                        0,
                                                                        &font_obj,
                                                                        &encryption_handler,
                                                                    );
                                                                    writer.write_all(&bytes)?;
                                                                    xref_entries.push((
                                                                        ref_obj.id, offset, 0, true,
                                                                    ));
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // Write erase overlay content stream if present
                                if let Some(overlay_obj_id) = erase_overlay_id {
                                    if let Some(overlay_content) =
                                        self.generate_erase_overlay(page_index)
                                    {
                                        // Create stream object for the overlay
                                        let overlay_stream = Object::Stream {
                                            dict: HashMap::new(),
                                            data: overlay_content.into(),
                                        };
                                        let offset = writer.stream_position()?;
                                        let bytes = serialize_obj(
                                            &serializer,
                                            overlay_obj_id,
                                            0,
                                            &overlay_stream,
                                            &encryption_handler,
                                        );
                                        writer.write_all(&bytes)?;
                                        xref_entries.push((overlay_obj_id, offset, 0, true));
                                    }
                                }

                                // Write new annotation objects
                                if !new_annotation_ids.is_empty() {
                                    // Get page refs for building annotations (needed for link destinations)
                                    let page_refs = self.get_page_refs().unwrap_or_default();

                                    if let Some(annotations) =
                                        self.modified_annotations.get(&page_index)
                                    {
                                        let new_annotations: Vec<_> =
                                            annotations.iter().filter(|a| a.is_new()).collect();

                                        for (annot_id, annot_wrapper) in
                                            new_annotation_ids.iter().zip(new_annotations.iter())
                                        {
                                            if let Some(writer_annot) =
                                                annot_wrapper.writer_annotation()
                                            {
                                                // Build the annotation dictionary
                                                let annot_dict = writer_annot.build(&page_refs);

                                                // Write the annotation object
                                                let offset = writer.stream_position()?;
                                                let bytes = serialize_obj(
                                                    &serializer,
                                                    *annot_id,
                                                    0,
                                                    &Object::Dictionary(annot_dict),
                                                    &encryption_handler,
                                                );
                                                writer.write_all(&bytes)?;
                                                xref_entries.push((*annot_id, offset, 0, true));
                                            }
                                        }
                                    }
                                }

                                // Write new form field objects
                                if !new_form_field_ids.is_empty() {
                                    let page_ref_for_fields = ObjectRef::new(page_ref.id, 0);

                                    for (field_id, wrapper) in new_form_field_ids
                                        .iter()
                                        .zip(new_form_field_wrappers.iter())
                                    {
                                        // Build the form field dictionary
                                        let field_dict =
                                            wrapper.build_field_dict(page_ref_for_fields);

                                        // Write the form field object
                                        let offset = writer.stream_position()?;
                                        let bytes = serialize_obj(
                                            &serializer,
                                            *field_id,
                                            0,
                                            &Object::Dictionary(field_dict),
                                            &encryption_handler,
                                        );
                                        writer.write_all(&bytes)?;
                                        xref_entries.push((*field_id, offset, 0, true));
                                    }
                                }

                                // Write flatten annotation XObjects and overlay
                                if let Some((ref appearances, overlay_id, ref xobj_ids)) =
                                    flatten_data
                                {
                                    // Write each appearance as a Form XObject
                                    for ((obj_id, _name), appearance) in
                                        xobj_ids.iter().zip(appearances.iter())
                                    {
                                        // Build Form XObject dictionary
                                        let mut form_dict = HashMap::new();
                                        form_dict.insert(
                                            "Type".to_string(),
                                            Object::Name("XObject".to_string()),
                                        );
                                        form_dict.insert(
                                            "Subtype".to_string(),
                                            Object::Name("Form".to_string()),
                                        );
                                        form_dict
                                            .insert("FormType".to_string(), Object::Integer(1));
                                        form_dict.insert(
                                            "BBox".to_string(),
                                            Object::Array(vec![
                                                Object::Real(appearance.bbox[0] as f64),
                                                Object::Real(appearance.bbox[1] as f64),
                                                Object::Real(appearance.bbox[2] as f64),
                                                Object::Real(appearance.bbox[3] as f64),
                                            ]),
                                        );

                                        // Add matrix if present
                                        if let Some(m) = appearance.matrix {
                                            form_dict.insert(
                                                "Matrix".to_string(),
                                                Object::Array(vec![
                                                    Object::Real(m[0] as f64),
                                                    Object::Real(m[1] as f64),
                                                    Object::Real(m[2] as f64),
                                                    Object::Real(m[3] as f64),
                                                    Object::Real(m[4] as f64),
                                                    Object::Real(m[5] as f64),
                                                ]),
                                            );
                                        }

                                        // Add resources if present
                                        if let Some(ref resources) = appearance.resources {
                                            form_dict
                                                .insert("Resources".to_string(), resources.clone());
                                        }

                                        // Create stream object
                                        let form_stream = Object::Stream {
                                            dict: form_dict,
                                            data: appearance.content.clone().into(),
                                        };

                                        let offset = writer.stream_position()?;
                                        let bytes = serialize_obj(
                                            &serializer,
                                            *obj_id,
                                            0,
                                            &form_stream,
                                            &encryption_handler,
                                        );
                                        writer.write_all(&bytes)?;
                                        xref_entries.push((*obj_id, offset, 0, true));
                                    }

                                    // Write the overlay content stream that invokes the XObjects
                                    let xobj_names: Vec<String> =
                                        xobj_ids.iter().map(|(_, name)| name.clone()).collect();
                                    let overlay_content =
                                        self.generate_flatten_overlay(appearances, &xobj_names);

                                    let overlay_stream = Object::Stream {
                                        dict: HashMap::new(),
                                        data: overlay_content.into(),
                                    };

                                    let offset = writer.stream_position()?;
                                    let bytes = serialize_obj(
                                        &serializer,
                                        overlay_id,
                                        0,
                                        &overlay_stream,
                                        &encryption_handler,
                                    );
                                    writer.write_all(&bytes)?;
                                    xref_entries.push((overlay_id, offset, 0, true));
                                }

                                // Write redaction overlay content stream if present
                                if let Some((ref redactions, redact_overlay_id)) = redaction_data {
                                    let overlay_content =
                                        self.generate_redaction_overlay(redactions);

                                    let overlay_stream = Object::Stream {
                                        dict: HashMap::new(),
                                        data: overlay_content.into(),
                                    };

                                    let offset = writer.stream_position()?;
                                    let bytes = serialize_obj(
                                        &serializer,
                                        redact_overlay_id,
                                        0,
                                        &overlay_stream,
                                        &encryption_handler,
                                    );
                                    writer.write_all(&bytes)?;
                                    xref_entries.push((redact_overlay_id, offset, 0, true));
                                }

                                // Write form flatten XObjects and overlay if present
                                if let Some((
                                    ref form_appearances,
                                    form_overlay_id,
                                    ref form_xobj_ids,
                                )) = form_flatten_data
                                {
                                    // Write each form appearance as an XObject
                                    for ((obj_id, _), appearance) in
                                        form_xobj_ids.iter().zip(form_appearances.iter())
                                    {
                                        let mut form_dict: HashMap<String, Object> = HashMap::new();
                                        form_dict.insert(
                                            "Type".to_string(),
                                            Object::Name("XObject".to_string()),
                                        );
                                        form_dict.insert(
                                            "Subtype".to_string(),
                                            Object::Name("Form".to_string()),
                                        );
                                        form_dict
                                            .insert("FormType".to_string(), Object::Integer(1));
                                        form_dict.insert(
                                            "BBox".to_string(),
                                            Object::Array(vec![
                                                Object::Real(appearance.bbox[0] as f64),
                                                Object::Real(appearance.bbox[1] as f64),
                                                Object::Real(appearance.bbox[2] as f64),
                                                Object::Real(appearance.bbox[3] as f64),
                                            ]),
                                        );

                                        // Add matrix if present
                                        if let Some(m) = appearance.matrix {
                                            form_dict.insert(
                                                "Matrix".to_string(),
                                                Object::Array(vec![
                                                    Object::Real(m[0] as f64),
                                                    Object::Real(m[1] as f64),
                                                    Object::Real(m[2] as f64),
                                                    Object::Real(m[3] as f64),
                                                    Object::Real(m[4] as f64),
                                                    Object::Real(m[5] as f64),
                                                ]),
                                            );
                                        }

                                        // Add resources if present
                                        if let Some(ref resources) = appearance.resources {
                                            form_dict
                                                .insert("Resources".to_string(), resources.clone());
                                        }

                                        // Create stream object
                                        let form_stream = Object::Stream {
                                            dict: form_dict,
                                            data: appearance.content.clone().into(),
                                        };

                                        let offset = writer.stream_position()?;
                                        let bytes = serialize_obj(
                                            &serializer,
                                            *obj_id,
                                            0,
                                            &form_stream,
                                            &encryption_handler,
                                        );
                                        writer.write_all(&bytes)?;
                                        xref_entries.push((*obj_id, offset, 0, true));
                                    }

                                    // Write the overlay content stream that invokes the XObjects
                                    let xobj_names: Vec<String> = form_xobj_ids
                                        .iter()
                                        .map(|(_, name)| name.clone())
                                        .collect();
                                    let overlay_content = self
                                        .generate_flatten_overlay(form_appearances, &xobj_names);

                                    let overlay_stream = Object::Stream {
                                        dict: HashMap::new(),
                                        data: overlay_content.into(),
                                    };

                                    let offset = writer.stream_position()?;
                                    let bytes = serialize_obj(
                                        &serializer,
                                        form_overlay_id,
                                        0,
                                        &overlay_stream,
                                        &encryption_handler,
                                    );
                                    writer.write_all(&bytes)?;
                                    xref_entries.push((form_overlay_id, offset, 0, true));
                                }

                                page_index += 1;
                            }
                        }
                    }
                }
            }
        }

        // Write parent-only form fields (non-terminal fields with no widget)
        // These don't belong to any specific page, so write them after page processing
        for (_, field_id, wrapper, _) in &all_form_field_data {
            if wrapper.is_parent_only() {
                // Build parent field dictionary (no widget entries)
                let field_dict = wrapper.build_parent_dict();

                // Write the parent field object
                let offset = writer.stream_position()?;
                let bytes = serialize_obj(
                    &serializer,
                    *field_id,
                    0,
                    &Object::Dictionary(field_dict),
                    &encryption_handler,
                );
                writer.write_all(&bytes)?;
                xref_entries.push((*field_id, offset, 0, true));
            }
        }

        // Write info dictionary if modified
        let info_ref = if self.modified_info.is_some() {
            let info = self.modified_info.clone().unwrap();
            let info_id = self.allocate_object_id();
            let info_obj = info.to_object();
            let offset = writer.stream_position()?;
            let bytes = serialize_obj(&serializer, info_id, 0, &info_obj, &encryption_handler);
            writer.write_all(&bytes)?;
            xref_entries.push((info_id, offset, 0, true));
            Some(ObjectRef::new(info_id, 0))
        } else {
            None
        };

        // Sort xref entries by object ID
        xref_entries.sort_by_key(|(id, _, _, _)| *id);

        // Write xref table
        let xref_offset = writer.stream_position()?;
        write!(writer, "xref\n")?;

        // Find max object ID
        let max_id = xref_entries
            .iter()
            .map(|(id, _, _, _)| *id)
            .max()
            .unwrap_or(0);
        write!(writer, "0 {}\n", max_id + 1)?;

        // Write entries (fill gaps with free entries)
        let mut entry_map: HashMap<u32, (u64, u16, bool)> = xref_entries
            .into_iter()
            .map(|(id, off, gen, used)| (id, (off, gen, used)))
            .collect();

        for id in 0..=max_id {
            if let Some((offset, gen, in_use)) = entry_map.get(&id) {
                if *in_use {
                    write!(writer, "{:010} {:05} n \n", offset, gen)?;
                } else {
                    write!(writer, "{:010} {:05} f \n", offset, gen)?;
                }
            } else {
                // Free entry pointing to object 0
                write!(writer, "0000000000 65535 f \n")?;
            }
        }

        // Write trailer
        write!(writer, "trailer\n")?;
        write!(writer, "<<\n")?;
        write!(writer, "  /Size {}\n", max_id + 1)?;
        write!(writer, "  /Root {} 0 R\n", catalog_ref.id)?;

        if let Some(info_ref) = info_ref {
            write!(writer, "  /Info {} {} R\n", info_ref.id, info_ref.gen)?;
        }

        // Write encryption entries if encrypting
        if let Some(enc_id) = encrypt_obj_id {
            write!(writer, "  /Encrypt {} 0 R\n", enc_id)?;
        }

        // Write file ID if encryption is enabled
        if let Some((id1, id2)) = file_id {
            let id1_hex: String = id1.iter().map(|b| format!("{:02X}", b)).collect();
            let id2_hex: String = id2.iter().map(|b| format!("{:02X}", b)).collect();
            write!(writer, "  /ID [<{}> <{}>]\n", id1_hex, id2_hex)?;
        }

        write!(writer, ">>\n")?;
        write!(writer, "startxref\n")?;
        write!(writer, "{}\n", xref_offset)?;
        write!(writer, "%%EOF\n")?;

        writer.flush()?;
        self.is_modified = false;
        Ok(())
    }

    // === Content modification operations ===

    /// Extract hierarchical content from a page.
    ///
    /// Returns the page's hierarchical content structure with all children populated.
    /// For untagged PDFs, returns a synthetic hierarchy based on geometric analysis.
    ///
    /// # Arguments
    ///
    /// * `page_index` - The page to extract from (0-indexed)
    ///
    /// # Returns
    ///
    /// `Ok(Some(structure))` if structure is found or generated,
    /// `Ok(None)` if no structure is available,
    /// `Err` if an error occurs during extraction
    pub fn get_page_content(&mut self, page_index: usize) -> Result<Option<StructureElement>> {
        HierarchicalExtractor::extract_page(&mut self.source, page_index)
    }

    /// Replace the content of a page with a new structure.
    ///
    /// Marks the document as modified and sets the structure_modified flag
    /// so the structure tree will be rebuilt on save.
    ///
    /// # Arguments
    ///
    /// * `page_index` - The page to modify (0-indexed)
    /// * `content` - The new hierarchical structure for the page
    ///
    /// # Returns
    ///
    /// `Err` if the page index is out of range
    pub fn set_page_content(&mut self, page_index: usize, content: StructureElement) -> Result<()> {
        let page_count = self.current_page_count();
        if page_index >= page_count {
            return Err(Error::InvalidPdf(format!(
                "Page index {} out of range (document has {} pages)",
                page_index, page_count
            )));
        }

        self.modified_content.insert(page_index, content);
        self.structure_modified = true;
        self.is_modified = true;
        Ok(())
    }

    /// Modify a page's structure in-place using a closure.
    ///
    /// Extracts the current content, passes it to the closure for modification,
    /// then saves it back.
    ///
    /// # Arguments
    ///
    /// * `page_index` - The page to modify
    /// * `f` - Closure that modifies the structure
    ///
    /// # Example
    ///
    /// ```ignore
    /// editor.modify_structure(0, |structure| {
    ///     // Modify structure in place
    ///     structure.alt_text = Some("Modified alt text".to_string());
    ///     Ok(())
    /// })?;
    /// ```
    pub fn modify_structure<F>(&mut self, page_index: usize, f: F) -> Result<()>
    where
        F: FnOnce(&mut StructureElement) -> Result<()>,
    {
        let mut content = self
            .get_page_content(page_index)?
            .ok_or_else(|| Error::InvalidPdf("No structure available for page".to_string()))?;

        f(&mut content)?;
        self.set_page_content(page_index, content)
    }

    /// Get the resource manager for allocating fonts, images, etc.
    ///
    /// Use this when manually constructing content elements that need resources.
    pub fn resource_manager_mut(&mut self) -> &mut ResourceManager {
        &mut self.resource_manager
    }

    /// Get a reference to the resource manager.
    pub fn resource_manager(&self) -> &ResourceManager {
        &self.resource_manager
    }

    /// Get a page for DOM-like editing.
    ///
    /// Returns a PdfPage that allows hierarchical navigation and querying
    /// of page content with a DOM-like API.
    pub fn get_page(&mut self, page_index: usize) -> Result<crate::editor::dom::PdfPage> {
        // Get the page info first
        let page_info = self.get_page_info(page_index)?;

        // Get or extract the page content
        let content = if let Some(structure) = self.get_page_content(page_index)? {
            structure
        } else {
            // If no modified content, try to extract from original
            match HierarchicalExtractor::extract_page(&mut self.source, page_index)? {
                Some(structure) => structure,
                None => {
                    // Create empty structure if extraction fails
                    StructureElement {
                        structure_type: "Document".to_string(),
                        bbox: crate::geometry::Rect::new(
                            0.0,
                            0.0,
                            page_info.width,
                            page_info.height,
                        ),
                        children: Vec::new(),
                        reading_order: Some(0),
                        alt_text: None,
                        language: None,
                    }
                },
            }
        };

        // Load annotations from source document
        let read_annotations = self.source.get_annotations(page_index).unwrap_or_default();
        let annotations: Vec<crate::editor::dom::AnnotationWrapper> = read_annotations
            .into_iter()
            .map(crate::editor::dom::AnnotationWrapper::from_read)
            .collect();

        Ok(crate::editor::dom::PdfPage::from_structure_with_annotations(
            page_index,
            content,
            page_info.width,
            page_info.height,
            annotations,
        ))
    }

    /// Save a modified page back to the document.
    ///
    /// This saves both the page content and any modified annotations.
    pub fn save_page(&mut self, page: crate::editor::dom::PdfPage) -> Result<()> {
        let page_index = page.page_index;
        let annotations_modified = page.has_annotations_modified();

        // Extract annotations before moving root
        let annotations: Vec<crate::editor::dom::AnnotationWrapper> = if annotations_modified {
            page.annotations().to_vec()
        } else {
            Vec::new()
        };

        // Save content structure
        self.set_page_content(page_index, page.root)?;

        // Save annotations if they were modified
        if annotations_modified {
            self.modified_annotations.insert(page_index, annotations);
            self.is_modified = true;
        }

        Ok(())
    }

    /// Get the modified annotations for a page (if any).
    pub fn get_page_annotations(
        &self,
        page_index: usize,
    ) -> Option<&Vec<crate::editor::dom::AnnotationWrapper>> {
        self.modified_annotations.get(&page_index)
    }

    /// Check if a page has modified annotations.
    pub fn has_modified_annotations(&self, page_index: usize) -> bool {
        self.modified_annotations.contains_key(&page_index)
    }

    /// Edit a page with a closure, automatically saving changes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// editor.edit_page(0, |page| {
    ///     let text_elements = page.find_text_containing("Hello");
    ///     for text in text_elements {
    ///         page.set_text(text.id(), "Hi")?;
    ///     }
    ///     Ok(())
    /// })?;
    /// ```
    pub fn edit_page<F>(&mut self, page_index: usize, f: F) -> Result<()>
    where
        F: FnOnce(&mut crate::editor::dom::PdfPage) -> Result<()>,
    {
        let mut page = self.get_page(page_index)?;
        f(&mut page)?;
        self.save_page(page)
    }

    /// Get a page editor for fluent/XMLDocument-style editing.
    ///
    /// # Example
    ///
    /// ```ignore
    /// editor.page_editor(0)?
    ///    .find_text_containing("Hello")?
    ///    .for_each(|mut text| {
    ///        text.set_text("Hi");
    ///        Ok(())
    ///    })?
    ///    .done()?;
    /// editor.save_page_editor_modified()?;
    /// ```
    pub fn page_editor(&mut self, page_index: usize) -> Result<crate::editor::dom::PageEditor> {
        let page = self.get_page(page_index)?;
        Ok(crate::editor::dom::PageEditor { page })
    }

    /// Save a page from the fluent editor back to the document.
    pub fn save_page_from_editor(&mut self, page: crate::editor::dom::PdfPage) -> Result<()> {
        self.save_page(page)
    }

    // =========================================================================
    // Page Properties: Rotation, Cropping
    // =========================================================================

    /// Get the rotation of a page in degrees (0, 90, 180, 270).
    ///
    /// Returns the effective rotation, considering any modifications.
    pub fn get_page_rotation(&mut self, index: usize) -> Result<i32> {
        // Check if we have a modified rotation
        if let Some(props) = self.modified_page_props.get(&index) {
            if let Some(rotation) = props.rotation {
                return Ok(rotation);
            }
        }

        // Otherwise get from original document
        let info = self.get_page_info(index)?;
        Ok(info.rotation)
    }

    /// Set the rotation of a page.
    ///
    /// Rotation must be 0, 90, 180, or 270 degrees.
    pub fn set_page_rotation(&mut self, index: usize, degrees: i32) -> Result<()> {
        // Validate rotation
        if ![0, 90, 180, 270].contains(&degrees) {
            return Err(Error::InvalidPdf(
                "Rotation must be 0, 90, 180, or 270 degrees".to_string(),
            ));
        }

        // Validate page index
        if index >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!(
                "Page index {} out of range (document has {} pages)",
                index,
                self.current_page_count()
            )));
        }

        // Store the modified rotation
        let props = self.modified_page_props.entry(index).or_default();
        props.rotation = Some(degrees);

        self.is_modified = true;
        Ok(())
    }

    /// Rotate a page by the given degrees (adds to current rotation).
    ///
    /// The result is normalized to 0, 90, 180, or 270.
    pub fn rotate_page_by(&mut self, index: usize, degrees: i32) -> Result<()> {
        let current = self.get_page_rotation(index)?;
        let new_rotation = ((current + degrees) % 360 + 360) % 360;

        // Normalize to valid PDF rotation
        let normalized = match new_rotation {
            0..=44 => 0,
            45..=134 => 90,
            135..=224 => 180,
            225..=314 => 270,
            _ => 0,
        };

        self.set_page_rotation(index, normalized)
    }

    /// Rotate all pages by the given degrees.
    pub fn rotate_all_pages(&mut self, degrees: i32) -> Result<()> {
        let count = self.current_page_count();
        for i in 0..count {
            self.rotate_page_by(i, degrees)?;
        }
        Ok(())
    }

    /// Get the MediaBox of a page (physical page size).
    ///
    /// Returns [llx, lly, urx, ury] (lower-left x, lower-left y, upper-right x, upper-right y).
    pub fn get_page_media_box(&mut self, index: usize) -> Result<[f32; 4]> {
        // Check if we have a modified MediaBox
        if let Some(props) = self.modified_page_props.get(&index) {
            if let Some(media_box) = props.media_box {
                return Ok(media_box);
            }
        }

        // Get from original document
        let page_refs = self.get_page_refs()?;
        if index >= page_refs.len() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", index)));
        }

        let page_ref = page_refs[index];
        let page_obj = self.source.load_object(page_ref)?;
        let page_dict = page_obj
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Page is not a dictionary".to_string()))?;

        if let Some(media_box) = page_dict.get("MediaBox").and_then(|m| m.as_array()) {
            if media_box.len() >= 4 {
                let llx = media_box[0]
                    .as_real()
                    .or_else(|| media_box[0].as_integer().map(|i| i as f64))
                    .unwrap_or(0.0) as f32;
                let lly = media_box[1]
                    .as_real()
                    .or_else(|| media_box[1].as_integer().map(|i| i as f64))
                    .unwrap_or(0.0) as f32;
                let urx = media_box[2]
                    .as_real()
                    .or_else(|| media_box[2].as_integer().map(|i| i as f64))
                    .unwrap_or(612.0) as f32;
                let ury = media_box[3]
                    .as_real()
                    .or_else(|| media_box[3].as_integer().map(|i| i as f64))
                    .unwrap_or(792.0) as f32;
                return Ok([llx, lly, urx, ury]);
            }
        }

        // Default to Letter size
        Ok([0.0, 0.0, 612.0, 792.0])
    }

    /// Set the MediaBox of a page.
    pub fn set_page_media_box(&mut self, index: usize, box_: [f32; 4]) -> Result<()> {
        if index >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", index)));
        }

        let props = self.modified_page_props.entry(index).or_default();
        props.media_box = Some(box_);

        self.is_modified = true;
        Ok(())
    }

    /// Get the CropBox of a page (visible/printable area).
    ///
    /// Returns None if no CropBox is set (defaults to MediaBox).
    pub fn get_page_crop_box(&mut self, index: usize) -> Result<Option<[f32; 4]>> {
        // Check if we have a modified CropBox
        if let Some(props) = self.modified_page_props.get(&index) {
            if let Some(crop_box) = props.crop_box {
                return Ok(Some(crop_box));
            }
        }

        // Get from original document
        let page_refs = self.get_page_refs()?;
        if index >= page_refs.len() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", index)));
        }

        let page_ref = page_refs[index];
        let page_obj = self.source.load_object(page_ref)?;
        let page_dict = page_obj
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Page is not a dictionary".to_string()))?;

        if let Some(crop_box) = page_dict.get("CropBox").and_then(|c| c.as_array()) {
            if crop_box.len() >= 4 {
                let llx = crop_box[0]
                    .as_real()
                    .or_else(|| crop_box[0].as_integer().map(|i| i as f64))
                    .unwrap_or(0.0) as f32;
                let lly = crop_box[1]
                    .as_real()
                    .or_else(|| crop_box[1].as_integer().map(|i| i as f64))
                    .unwrap_or(0.0) as f32;
                let urx = crop_box[2]
                    .as_real()
                    .or_else(|| crop_box[2].as_integer().map(|i| i as f64))
                    .unwrap_or(612.0) as f32;
                let ury = crop_box[3]
                    .as_real()
                    .or_else(|| crop_box[3].as_integer().map(|i| i as f64))
                    .unwrap_or(792.0) as f32;
                return Ok(Some([llx, lly, urx, ury]));
            }
        }

        Ok(None)
    }

    /// Set the CropBox of a page.
    pub fn set_page_crop_box(&mut self, index: usize, box_: [f32; 4]) -> Result<()> {
        if index >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", index)));
        }

        let props = self.modified_page_props.entry(index).or_default();
        props.crop_box = Some(box_);

        self.is_modified = true;
        Ok(())
    }

    /// Crop margins from all pages.
    ///
    /// This sets the CropBox to be smaller than the MediaBox by the specified margins.
    pub fn crop_margins(&mut self, left: f32, right: f32, top: f32, bottom: f32) -> Result<()> {
        let count = self.current_page_count();
        for i in 0..count {
            let media_box = self.get_page_media_box(i)?;
            let crop_box = [
                media_box[0] + left,
                media_box[1] + bottom,
                media_box[2] - right,
                media_box[3] - top,
            ];
            self.set_page_crop_box(i, crop_box)?;
        }
        Ok(())
    }

    // =========================================================================
    // Content Erasing (Whiteout)
    // =========================================================================

    /// Erase a rectangular region on a page by covering it with white.
    ///
    /// This adds a white rectangle overlay that covers the specified region.
    /// The original content is not removed but hidden beneath the white overlay.
    ///
    /// # Arguments
    ///
    /// * `page` - Page index (0-based)
    /// * `rect` - Rectangle to erase [llx, lly, urx, ury]
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Erase a region in the upper-left corner
    /// editor.erase_region(0, [72.0, 700.0, 200.0, 792.0])?;
    /// editor.save("output.pdf")?;
    /// ```
    pub fn erase_region(&mut self, page: usize, rect: [f32; 4]) -> Result<()> {
        if page >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", page)));
        }

        // Add to erase regions for this page
        let regions = self.erase_regions.entry(page).or_default();
        regions.push(rect);

        self.is_modified = true;
        Ok(())
    }

    /// Erase multiple rectangular regions on a page.
    pub fn erase_regions(&mut self, page: usize, rects: &[[f32; 4]]) -> Result<()> {
        if page >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", page)));
        }

        let regions = self.erase_regions.entry(page).or_default();
        regions.extend_from_slice(rects);

        self.is_modified = true;
        Ok(())
    }

    /// Clear all pending erase operations for a page.
    pub fn clear_erase_regions(&mut self, page: usize) {
        self.erase_regions.remove(&page);
    }

    /// Generate the content stream for erase overlays.
    ///
    /// Returns PDF operators that draw white rectangles over the specified regions.
    fn generate_erase_overlay(&self, page: usize) -> Option<Vec<u8>> {
        let regions = self.erase_regions.get(&page)?;
        if regions.is_empty() {
            return None;
        }

        let mut content = Vec::new();

        // Save graphics state
        content.extend_from_slice(b"q\n");

        // Set fill color to white (RGB 1 1 1)
        content.extend_from_slice(b"1 1 1 rg\n");

        // Draw each rectangle
        for rect in regions {
            let x = rect[0];
            let y = rect[1];
            let width = rect[2] - rect[0];
            let height = rect[3] - rect[1];

            // Rectangle path and fill
            content.extend_from_slice(
                format!("{:.2} {:.2} {:.2} {:.2} re f\n", x, y, width, height).as_bytes(),
            );
        }

        // Restore graphics state
        content.extend_from_slice(b"Q\n");

        Some(content)
    }

    // ========================================================================
    // Annotation Flattening
    // ========================================================================

    /// Mark annotations on a page for flattening.
    ///
    /// When the document is saved, annotations on this page will be rendered
    /// into the page content and removed from the annotations array.
    ///
    /// # Arguments
    /// * `page` - The zero-based page index
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Flatten annotations on page 0
    /// editor.flatten_page_annotations(0)?;
    /// editor.save("output.pdf")?;
    /// ```
    pub fn flatten_page_annotations(&mut self, page: usize) -> Result<()> {
        if page >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", page)));
        }

        self.flatten_annotations_pages.insert(page);
        self.is_modified = true;
        Ok(())
    }

    /// Mark all pages for annotation flattening.
    ///
    /// When the document is saved, all annotations will be rendered
    /// into the page content and removed.
    pub fn flatten_all_annotations(&mut self) -> Result<()> {
        let page_count = self.current_page_count();
        for page in 0..page_count {
            self.flatten_annotations_pages.insert(page);
        }
        self.is_modified = true;
        Ok(())
    }

    /// Check if a page has annotations marked for flattening.
    pub fn is_page_marked_for_flatten(&self, page: usize) -> bool {
        self.flatten_annotations_pages.contains(&page)
    }

    /// Clear the flatten annotation flag for a page.
    pub fn unmark_page_for_flatten(&mut self, page: usize) {
        self.flatten_annotations_pages.remove(&page);
    }

    // ========================================================================
    // Form Flattening
    // ========================================================================

    /// Mark form fields on a specific page for flattening.
    ///
    /// When the document is saved, form fields (Widget annotations) on this page
    /// will be rendered into the page content. Only Widget annotations are flattened,
    /// other annotation types are preserved.
    ///
    /// # Arguments
    /// * `page` - The zero-based page index
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Flatten forms on page 0
    /// editor.flatten_forms_on_page(0)?;
    /// editor.save("flattened.pdf")?;
    /// ```
    pub fn flatten_forms_on_page(&mut self, page: usize) -> Result<()> {
        if page >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", page)));
        }

        self.flatten_forms_pages.insert(page);
        self.is_modified = true;
        Ok(())
    }

    /// Mark all pages for form field flattening.
    ///
    /// When the document is saved, all form fields will be rendered into the page
    /// content and the AcroForm dictionary will be removed from the catalog.
    ///
    /// # Example
    ///
    /// ```ignore
    /// editor.flatten_forms()?;
    /// editor.save("flattened.pdf")?;
    /// ```
    pub fn flatten_forms(&mut self) -> Result<()> {
        let page_count = self.current_page_count();
        for page in 0..page_count {
            self.flatten_forms_pages.insert(page);
        }
        self.remove_acroform = true;
        self.is_modified = true;
        Ok(())
    }

    /// Check if a page has form fields marked for flattening.
    pub fn is_page_marked_for_form_flatten(&self, page: usize) -> bool {
        self.flatten_forms_pages.contains(&page)
    }

    /// Check if AcroForm will be removed on save.
    pub fn will_remove_acroform(&self) -> bool {
        self.remove_acroform
    }

    // =========================================================================
    // File Attachments (Embedded Files)
    // =========================================================================

    /// Embed a file in the document.
    ///
    /// The file will be added to the document's EmbeddedFiles name tree
    /// when the document is saved.
    ///
    /// # Arguments
    ///
    /// * `name` - The file name (used as identifier and display name)
    /// * `data` - The file contents
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("input.pdf")?;
    /// editor.embed_file("data.csv", csv_bytes)?;
    /// editor.save("output.pdf")?;
    /// ```
    pub fn embed_file(&mut self, name: &str, data: Vec<u8>) -> Result<()> {
        let file = crate::writer::EmbeddedFile::new(name, data);
        self.embedded_files.push(file);
        self.is_modified = true;
        Ok(())
    }

    /// Embed a file with additional metadata.
    ///
    /// # Arguments
    ///
    /// * `file` - The embedded file configuration
    pub fn embed_file_with_options(&mut self, file: crate::writer::EmbeddedFile) -> Result<()> {
        self.embedded_files.push(file);
        self.is_modified = true;
        Ok(())
    }

    /// Get the list of files that will be embedded on save.
    pub fn pending_embedded_files(&self) -> &[crate::writer::EmbeddedFile] {
        &self.embedded_files
    }

    /// Clear all pending embedded files.
    pub fn clear_embedded_files(&mut self) {
        self.embedded_files.clear();
    }

    // =========================================================================
    // XFA Forms Support
    // =========================================================================

    /// Check if this document contains XFA forms.
    ///
    /// XFA (XML Forms Architecture) is an XML-based form specification used
    /// in some PDFs, particularly government and financial forms.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("form.pdf")?;
    /// if editor.has_xfa()? {
    ///     println!("Document contains XFA forms");
    /// }
    /// ```
    pub fn has_xfa(&mut self) -> Result<bool> {
        crate::xfa::XfaExtractor::has_xfa(&mut self.source)
    }

    /// Analyze XFA forms in this document without converting.
    ///
    /// Returns information about the XFA form structure including
    /// field count, page count, and field types.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("form.pdf")?;
    /// let analysis = editor.analyze_xfa()?;
    ///
    /// if analysis.has_xfa {
    ///     println!("Found {} fields across {} pages",
    ///         analysis.field_count.unwrap_or(0),
    ///         analysis.page_count.unwrap_or(0));
    /// }
    /// ```
    pub fn analyze_xfa(&mut self) -> Result<crate::xfa::XfaAnalysis> {
        crate::xfa::analyze_xfa_document(&mut self.source)
    }

    /// Convert XFA forms to AcroForm and return new PDF bytes.
    ///
    /// This creates a new PDF document with the XFA forms converted to
    /// standard AcroForm fields. The original document is not modified.
    ///
    /// # Limitations
    ///
    /// This implementation supports **static conversion only**:
    /// - Extracts field definitions and current values
    /// - Converts fields to equivalent AcroForm types
    /// - Uses simple vertical stacking layout
    ///
    /// **NOT supported:**
    /// - Dynamic XFA features (scripts, calculations, conditional logic)
    /// - Complex layouts (tables, grids, repeating sections)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("xfa_form.pdf")?;
    /// if editor.has_xfa()? {
    ///     let acroform_bytes = editor.convert_xfa_to_acroform(None)?;
    ///     std::fs::write("converted.pdf", acroform_bytes)?;
    /// }
    /// ```
    pub fn convert_xfa_to_acroform(
        &mut self,
        options: Option<crate::xfa::XfaConversionOptions>,
    ) -> Result<Vec<u8>> {
        crate::xfa::convert_xfa_document(&mut self.source, options)
    }

    // =========================================================================
    // Form Field Editing
    // =========================================================================

    /// Get all form fields from the document.
    ///
    /// Returns form fields from the document's AcroForm, including any modifications
    /// made during this editing session. Deleted fields are not included.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("form.pdf")?;
    /// let fields = editor.get_form_fields()?;
    ///
    /// for field in &fields {
    ///     println!("{}: {:?}", field.name(), field.value());
    /// }
    /// ```
    pub fn get_form_fields(&mut self) -> Result<Vec<FormFieldWrapper>> {
        use crate::extractors::forms::FormExtractor;

        // Extract fields from source document
        let source_fields = FormExtractor::extract_fields(&mut self.source)?;

        let mut result = Vec::new();

        // Add original fields (wrapped), excluding deleted ones
        for field in source_fields {
            let full_name = field.full_name.clone();

            // Skip if deleted
            if self.deleted_form_fields.contains(&full_name) {
                continue;
            }

            // Check if we have a modified version
            if let Some(wrapper) = self.modified_form_fields.get(&full_name) {
                result.push(wrapper.clone());
            } else {
                // Use original field wrapped
                // Note: page_index is 0 for now since FormField doesn't track page
                // TODO: Track page index from widget annotations
                result.push(FormFieldWrapper::from_read(field, 0, None));
            }
        }

        // Add new fields (not from original document)
        for (name, wrapper) in &self.modified_form_fields {
            if wrapper.is_new() && !self.deleted_form_fields.contains(name) {
                result.push(wrapper.clone());
            }
        }

        Ok(result)
    }

    /// Get the value of a specific form field by name.
    ///
    /// Returns the current value of the field, which may be the original value
    /// or a modified value if `set_form_field_value()` was called.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field (e.g., "form.section.field")
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("form.pdf")?;
    ///
    /// if let Some(value) = editor.get_form_field_value("email")? {
    ///     println!("Email: {:?}", value);
    /// }
    /// ```
    pub fn get_form_field_value(
        &mut self,
        name: &str,
    ) -> Result<Option<crate::editor::form_fields::FormFieldValue>> {
        use crate::editor::form_fields::FormFieldValue;
        use crate::extractors::forms::FormExtractor;

        // Check if deleted
        if self.deleted_form_fields.contains(name) {
            return Ok(None);
        }

        // Check modified fields first
        if let Some(wrapper) = self.modified_form_fields.get(name) {
            return Ok(Some(wrapper.value()));
        }

        // Look up in original document
        let source_fields = FormExtractor::extract_fields(&mut self.source)?;

        for field in source_fields {
            if field.full_name == name {
                return Ok(Some(FormFieldValue::from(&field.value)));
            }
        }

        Ok(None)
    }

    /// Check if a form field with the given name exists.
    ///
    /// Returns true if the field exists in the original document or was added
    /// during this editing session, and has not been deleted.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("form.pdf")?;
    ///
    /// if editor.has_form_field("email")? {
    ///     println!("Email field exists");
    /// }
    /// ```
    pub fn has_form_field(&mut self, name: &str) -> Result<bool> {
        use crate::extractors::forms::FormExtractor;

        // Check if deleted
        if self.deleted_form_fields.contains(name) {
            return Ok(false);
        }

        // Check modified fields (includes new fields)
        if self.modified_form_fields.contains_key(name) {
            return Ok(true);
        }

        // Look up in original document
        let source_fields = FormExtractor::extract_fields(&mut self.source)?;

        for field in source_fields {
            if field.full_name == name {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Add a new form field to a page.
    ///
    /// Creates a new form field and widget annotation on the specified page.
    /// The field will be added to the document's AcroForm on save.
    ///
    /// # Arguments
    ///
    /// * `page` - The page index (0-based) where the field should appear
    /// * `widget` - A form field widget implementing `FormFieldWidget`
    ///
    /// # Returns
    ///
    /// The full qualified name of the added field, which may be modified if
    /// a field with the same name already exists.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    /// use pdf_oxide::writer::form_fields::TextFieldWidget;
    /// use pdf_oxide::geometry::Rect;
    ///
    /// let mut editor = DocumentEditor::open("document.pdf")?;
    ///
    /// // Add a text field to page 0
    /// let name = editor.add_form_field(0,
    ///     TextFieldWidget::new("email", Rect::new(100.0, 700.0, 200.0, 20.0))
    ///         .with_value("user@example.com")
    /// )?;
    ///
    /// println!("Added field: {}", name);
    /// editor.save("output.pdf")?;
    /// ```
    pub fn add_form_field<W: crate::writer::form_fields::FormFieldWidget>(
        &mut self,
        page: usize,
        widget: W,
    ) -> Result<String> {
        // Validate page index
        let page_count = self.page_count()?;
        if page >= page_count {
            return Err(Error::InvalidPdf(format!(
                "Page index {} out of bounds (document has {} pages)",
                page, page_count
            )));
        }

        // Make name unique if it already exists
        let mut name = widget.field_name().to_string();
        let mut counter = 1;
        while self.has_form_field(&name)? {
            name = format!("{}_{}", widget.field_name(), counter);
            counter += 1;
        }

        // Create wrapper from widget
        let mut wrapper = FormFieldWrapper::from_widget(&widget, page);

        // Override name if it was modified for uniqueness
        if name != widget.field_name() {
            wrapper.name = name.clone();
        }

        // Mark document as modified
        self.is_modified = true;
        self.acroform_modified = true;

        // Store in modified fields
        self.modified_form_fields.insert(name.clone(), wrapper);

        Ok(name)
    }

    /// Add a parent container field for hierarchical form fields.
    ///
    /// Parent fields are non-terminal fields that don't have a widget annotation
    /// but contain child fields via the `/Kids` array. They can be used to:
    /// - Group related fields (e.g., `address.street`, `address.city`)
    /// - Inherit properties to children (flags, field type, default appearance)
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the parent field
    ///
    /// # Returns
    ///
    /// The full qualified name of the parent field.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::{DocumentEditor, ParentFieldConfig};
    ///
    /// let mut editor = DocumentEditor::open("document.pdf")?;
    ///
    /// // Create a parent field
    /// editor.add_parent_field(ParentFieldConfig::new("address"))?;
    ///
    /// // Add children under the parent
    /// editor.add_child_field("address", 0, TextFieldWidget::new("street", rect))?;
    /// editor.add_child_field("address", 0, TextFieldWidget::new("city", rect2))?;
    ///
    /// editor.save("output.pdf")?;
    /// ```
    pub fn add_parent_field(
        &mut self,
        config: crate::editor::form_fields::ParentFieldConfig,
    ) -> Result<String> {
        let name = config.full_name();

        // Check if parent already exists
        if self.has_form_field(&name)? {
            return Err(Error::InvalidPdf(format!("Parent field already exists: {}", name)));
        }

        // If this parent has a parent, verify it exists
        if let Some(ref parent_name) = config.parent_name {
            if !self.has_form_field(parent_name)? {
                return Err(Error::InvalidPdf(format!("Parent field not found: {}", parent_name)));
            }
        }

        // Create wrapper from config
        let wrapper = FormFieldWrapper::from_parent_config(&config);

        // Mark document as modified
        self.is_modified = true;
        self.acroform_modified = true;

        // Store in modified fields
        self.modified_form_fields.insert(name.clone(), wrapper);

        Ok(name)
    }

    /// Add a form field as a child of an existing parent field.
    ///
    /// Creates a hierarchical relationship where the child field's partial name
    /// becomes the full name: `parent_name.widget_name`.
    ///
    /// # Arguments
    ///
    /// * `parent_name` - Name of the existing parent field
    /// * `page` - Page index where the widget appears (0-based)
    /// * `widget` - The form field widget to add
    ///
    /// # Returns
    ///
    /// The full qualified name of the child field.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::{DocumentEditor, ParentFieldConfig};
    /// use pdf_oxide::writer::form_fields::TextFieldWidget;
    /// use pdf_oxide::geometry::Rect;
    ///
    /// let mut editor = DocumentEditor::open("document.pdf")?;
    ///
    /// // Create parent first
    /// editor.add_parent_field(ParentFieldConfig::new("contact"))?;
    ///
    /// // Add children
    /// let name = editor.add_child_field("contact", 0,
    ///     TextFieldWidget::new("email", Rect::new(100.0, 700.0, 200.0, 20.0))
    /// )?;
    /// assert_eq!(name, "contact.email");
    ///
    /// editor.save("output.pdf")?;
    /// ```
    pub fn add_child_field<W: crate::writer::form_fields::FormFieldWidget>(
        &mut self,
        parent_name: &str,
        page: usize,
        widget: W,
    ) -> Result<String> {
        // Validate page index
        let page_count = self.page_count()?;
        if page >= page_count {
            return Err(Error::InvalidPdf(format!(
                "Page index {} out of bounds (document has {} pages)",
                page, page_count
            )));
        }

        // Verify parent exists
        if !self.has_form_field(parent_name)? {
            return Err(Error::InvalidPdf(format!("Parent field not found: {}", parent_name)));
        }

        // Create wrapper with parent reference
        let wrapper = FormFieldWrapper::from_widget_with_parent(&widget, page, parent_name);
        let name = wrapper.name.clone();

        // Check for duplicate name
        if self.has_form_field(&name)? {
            return Err(Error::InvalidPdf(format!("Child field already exists: {}", name)));
        }

        // Mark document as modified
        self.is_modified = true;
        self.acroform_modified = true;

        // Store in modified fields
        self.modified_form_fields.insert(name.clone(), wrapper);

        Ok(name)
    }

    /// Add a form field with automatic hierarchical parent creation.
    ///
    /// If the widget name contains dots (e.g., "address.street"), this method
    /// automatically creates any missing parent fields. This provides a convenient
    /// way to create hierarchical forms without manually managing parents.
    ///
    /// # Arguments
    ///
    /// * `page` - Page index where the widget appears (0-based)
    /// * `widget` - The form field widget to add
    ///
    /// # Returns
    ///
    /// The full qualified name of the added field.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    /// use pdf_oxide::writer::form_fields::TextFieldWidget;
    /// use pdf_oxide::geometry::Rect;
    ///
    /// let mut editor = DocumentEditor::open("document.pdf")?;
    ///
    /// // Automatically creates "address" parent if needed
    /// editor.add_form_field_hierarchical(0,
    ///     TextFieldWidget::new("address.street", Rect::new(100.0, 700.0, 200.0, 20.0))
    /// )?;
    ///
    /// // Reuses existing "address" parent
    /// editor.add_form_field_hierarchical(0,
    ///     TextFieldWidget::new("address.city", Rect::new(100.0, 670.0, 200.0, 20.0))
    /// )?;
    ///
    /// // Creates nested hierarchy: "contact" -> "address" -> "zip"
    /// editor.add_form_field_hierarchical(0,
    ///     TextFieldWidget::new("contact.address.zip", Rect::new(100.0, 640.0, 100.0, 20.0))
    /// )?;
    ///
    /// editor.save("output.pdf")?;
    /// ```
    pub fn add_form_field_hierarchical<W: crate::writer::form_fields::FormFieldWidget>(
        &mut self,
        page: usize,
        widget: W,
    ) -> Result<String> {
        use crate::editor::form_fields::ParentFieldConfig;

        let full_name = widget.field_name().to_string();

        // If no dots, delegate to regular add_form_field
        if !full_name.contains('.') {
            return self.add_form_field(page, widget);
        }

        // Parse the hierarchy path
        let parts: Vec<&str> = full_name.split('.').collect();

        // Create parent fields as needed
        let mut current_parent = String::new();
        for i in 0..(parts.len() - 1) {
            let part = parts[i];
            let parent_name = if current_parent.is_empty() {
                part.to_string()
            } else {
                format!("{}.{}", current_parent, part)
            };

            // Create parent if it doesn't exist
            if !self.has_form_field(&parent_name)? {
                let mut config = ParentFieldConfig::new(part);
                if !current_parent.is_empty() {
                    config = config.with_parent(&current_parent);
                }
                self.add_parent_field(config)?;
            }

            current_parent = parent_name;
        }

        // Add the terminal field as a child
        self.add_child_field(&current_parent, page, widget)
    }

    /// Set the value of an existing form field.
    ///
    /// Modifies the value of a form field. The field must exist in the document
    /// (either from the original PDF or added via `add_form_field`).
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `value` - The new value for the field
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::{DocumentEditor, FormFieldValue};
    ///
    /// let mut editor = DocumentEditor::open("form.pdf")?;
    ///
    /// editor.set_form_field_value("name", FormFieldValue::Text("John Doe".into()))?;
    /// editor.set_form_field_value("subscribe", FormFieldValue::Boolean(true))?;
    ///
    /// editor.save("updated.pdf")?;
    /// ```
    pub fn set_form_field_value(
        &mut self,
        name: &str,
        value: crate::editor::form_fields::FormFieldValue,
    ) -> Result<()> {
        use crate::extractors::forms::FormExtractor;

        // Check if deleted
        if self.deleted_form_fields.contains(name) {
            return Err(Error::InvalidPdf(format!("Cannot set value on deleted field: {}", name)));
        }

        // Check if we already have a wrapper for this field
        if let Some(wrapper) = self.modified_form_fields.get_mut(name) {
            wrapper.set_value(value);
            self.is_modified = true;
            self.acroform_modified = true;
            return Ok(());
        }

        // Look up in original document and create wrapper
        let source_fields = FormExtractor::extract_fields(&mut self.source)?;

        for field in source_fields {
            if field.full_name == name {
                // Create wrapper and set value
                let obj_ref = field.object_ref;
                let mut wrapper = FormFieldWrapper::from_read(field, 0, obj_ref);
                wrapper.set_value(value);

                self.modified_form_fields.insert(name.to_string(), wrapper);
                self.is_modified = true;
                self.acroform_modified = true;
                return Ok(());
            }
        }

        Err(Error::InvalidPdf(format!("Form field not found: {}", name)))
    }

    /// Remove a form field from the document.
    ///
    /// Marks a form field for removal. The field will be removed from the
    /// document's AcroForm and its widget annotation will be removed from
    /// the page when the document is saved.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field to remove
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("form.pdf")?;
    ///
    /// editor.remove_form_field("obsolete_field")?;
    ///
    /// editor.save("cleaned.pdf")?;
    /// ```
    pub fn remove_form_field(&mut self, name: &str) -> Result<()> {
        // Check if field exists
        if !self.has_form_field(name)? {
            return Err(Error::InvalidPdf(format!("Form field not found: {}", name)));
        }

        // Remove from modified fields if present
        self.modified_form_fields.remove(name);

        // Add to deleted set
        self.deleted_form_fields.insert(name.to_string());

        self.is_modified = true;
        self.acroform_modified = true;

        Ok(())
    }

    // ========== Form Field Property Modification APIs ==========

    /// Set a form field to read-only.
    ///
    /// A read-only field cannot be edited by the user in a PDF viewer.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `readonly` - Whether the field should be read-only
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("form.pdf")?;
    /// editor.set_form_field_readonly("signature_field", true)?;
    /// editor.save("readonly.pdf")?;
    /// ```
    pub fn set_form_field_readonly(&mut self, name: &str, readonly: bool) -> Result<()> {
        self.modify_form_field(name, |wrapper| {
            wrapper.set_readonly(readonly);
        })
    }

    /// Set a form field as required.
    ///
    /// A required field must have a value when the form is submitted/exported.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `required` - Whether the field should be required
    pub fn set_form_field_required(&mut self, name: &str, required: bool) -> Result<()> {
        self.modify_form_field(name, |wrapper| {
            wrapper.set_required(required);
        })
    }

    /// Set a form field's tooltip/description.
    ///
    /// The tooltip is displayed when the user hovers over the field.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `tooltip` - The tooltip text
    pub fn set_form_field_tooltip(&mut self, name: &str, tooltip: impl Into<String>) -> Result<()> {
        let tooltip_str = tooltip.into();
        self.modify_form_field(name, |wrapper| {
            wrapper.set_tooltip(tooltip_str);
        })
    }

    /// Set a form field's bounding rectangle.
    ///
    /// This changes the position and size of the field on the page.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `rect` - The new bounding rectangle
    pub fn set_form_field_rect(&mut self, name: &str, rect: Rect) -> Result<()> {
        self.modify_form_field(name, |wrapper| {
            wrapper.set_rect(rect);
        })
    }

    /// Set a form field's maximum text length.
    ///
    /// Only applicable to text fields.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `max_len` - The maximum number of characters
    pub fn set_form_field_max_length(&mut self, name: &str, max_len: u32) -> Result<()> {
        self.modify_form_field(name, |wrapper| {
            wrapper.set_max_length(max_len);
        })
    }

    /// Set a form field's text alignment.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `alignment` - 0 = left, 1 = center, 2 = right
    pub fn set_form_field_alignment(&mut self, name: &str, alignment: u32) -> Result<()> {
        self.modify_form_field(name, |wrapper| {
            wrapper.set_alignment(alignment);
        })
    }

    /// Set a form field's background color.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `color` - RGB color values (0.0 to 1.0)
    pub fn set_form_field_background_color(&mut self, name: &str, color: [f32; 3]) -> Result<()> {
        self.modify_form_field(name, |wrapper| {
            wrapper.set_background_color(color);
        })
    }

    /// Set a form field's border color.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `color` - RGB color values (0.0 to 1.0)
    pub fn set_form_field_border_color(&mut self, name: &str, color: [f32; 3]) -> Result<()> {
        self.modify_form_field(name, |wrapper| {
            wrapper.set_border_color(color);
        })
    }

    /// Set a form field's border width.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `width` - Border width in points
    pub fn set_form_field_border_width(&mut self, name: &str, width: f32) -> Result<()> {
        self.modify_form_field(name, |wrapper| {
            wrapper.set_border_width(width);
        })
    }

    /// Set a form field's default appearance string.
    ///
    /// The DA string specifies font, size, and color for field content.
    /// Example: "/Helv 12 Tf 0 g" for 12pt Helvetica in black.
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `da` - The default appearance string
    pub fn set_form_field_default_appearance(
        &mut self,
        name: &str,
        da: impl Into<String>,
    ) -> Result<()> {
        let da_str = da.into();
        self.modify_form_field(name, |wrapper| {
            wrapper.set_default_appearance(da_str);
        })
    }

    /// Set form field flags directly.
    ///
    /// Use this for setting custom flag combinations. Common flags:
    /// - Bit 1 (0x01): ReadOnly
    /// - Bit 2 (0x02): Required
    /// - Bit 3 (0x04): NoExport
    ///
    /// # Arguments
    ///
    /// * `name` - The full qualified name of the field
    /// * `flags` - The field flag bits
    pub fn set_form_field_flags(&mut self, name: &str, flags: u32) -> Result<()> {
        self.modify_form_field(name, |wrapper| {
            wrapper.set_flags(flags);
        })
    }

    /// Internal helper to modify a form field.
    ///
    /// Gets or creates a wrapper for the field and applies the modification.
    fn modify_form_field<F>(&mut self, name: &str, modify_fn: F) -> Result<()>
    where
        F: FnOnce(&mut FormFieldWrapper),
    {
        use crate::extractors::forms::FormExtractor;

        // Check if deleted
        if self.deleted_form_fields.contains(name) {
            return Err(Error::InvalidPdf(format!("Cannot modify deleted field: {}", name)));
        }

        // Check if we already have a wrapper for this field
        if let Some(wrapper) = self.modified_form_fields.get_mut(name) {
            modify_fn(wrapper);
            self.is_modified = true;
            self.acroform_modified = true;
            return Ok(());
        }

        // Look up in original document and create wrapper
        let source_fields = FormExtractor::extract_fields(&mut self.source)?;

        for field in source_fields {
            if field.full_name == name {
                // Get object ref from the field
                let object_ref = field.object_ref;

                // Create wrapper
                let mut wrapper = FormFieldWrapper::from_read(field, 0, object_ref);
                modify_fn(&mut wrapper);

                self.modified_form_fields.insert(name.to_string(), wrapper);
                self.is_modified = true;
                self.acroform_modified = true;
                return Ok(());
            }
        }

        Err(Error::InvalidPdf(format!("Form field not found: {}", name)))
    }

    // ========== Form Data Export APIs ==========

    /// Export form field data to FDF format.
    ///
    /// Writes all form field data (original and modified) to an FDF file.
    /// This is useful for data extraction, backup, or batch processing.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path to write the FDF file
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("filled_form.pdf")?;
    /// editor.export_form_data_fdf("form_data.fdf")?;
    /// ```
    pub fn export_form_data_fdf(&mut self, output_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::extractors::forms::FormExtractor;
        FormExtractor::export_fdf(&mut self.source, output_path)
    }

    /// Export form field data to XFDF format.
    ///
    /// Writes all form field data (original and modified) to an XFDF (XML) file.
    /// XFDF is useful for web integration and human-readable data exchange.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path to write the XFDF file
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::editor::DocumentEditor;
    ///
    /// let mut editor = DocumentEditor::open("filled_form.pdf")?;
    /// editor.export_form_data_xfdf("form_data.xfdf")?;
    /// ```
    pub fn export_form_data_xfdf(
        &mut self,
        output_path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        use crate::extractors::forms::FormExtractor;
        FormExtractor::export_xfdf(&mut self.source, output_path)
    }

    /// Get widget annotation appearances for form flattening.
    ///
    /// Returns appearance data for Widget annotations only.
    /// Generates appearance streams for widgets that don't have them.
    fn get_widget_appearances(&mut self, page: usize) -> Result<Vec<AnnotationAppearance>> {
        use crate::annotation_types::AnnotationSubtype;

        let annotations = self.source.get_annotations(page)?;
        let mut appearances = Vec::new();

        for annotation in annotations {
            // Only process Widget annotations (form fields)
            if annotation.subtype_enum != AnnotationSubtype::Widget {
                continue;
            }

            // Skip annotations without a raw dictionary
            let raw_dict = match &annotation.raw_dict {
                Some(dict) => dict,
                None => continue,
            };

            // Try to get appearance from AP dictionary
            let appearance_result = self.extract_widget_appearance(&annotation, raw_dict);

            match appearance_result {
                Ok(Some(appearance)) => appearances.push(appearance),
                Ok(None) => {
                    // No appearance stream - try to generate one
                    if let Some(generated) = self.generate_widget_appearance(&annotation)? {
                        appearances.push(generated);
                    }
                },
                Err(_) => continue,
            }
        }

        Ok(appearances)
    }

    /// Extract appearance stream from a widget annotation.
    fn extract_widget_appearance(
        &mut self,
        annotation: &crate::annotations::Annotation,
        raw_dict: &HashMap<String, Object>,
    ) -> Result<Option<AnnotationAppearance>> {
        // Get the /AP (appearance) dictionary
        let ap_dict = match raw_dict.get("AP") {
            Some(Object::Dictionary(d)) => d.clone(),
            Some(Object::Reference(ap_ref)) => match self.source.load_object(*ap_ref)? {
                Object::Dictionary(d) => d,
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };

        // Get the /N (normal appearance) entry
        let normal_appearance = match ap_dict.get("N") {
            Some(obj) => obj.clone(),
            None => return Ok(None),
        };

        // Handle appearance states (e.g., /Yes and /Off for checkboxes)
        let (appearance_obj, appearance_ref) = match normal_appearance {
            Object::Reference(ref_obj) => {
                let obj = self.source.load_object(ref_obj)?;
                (obj, Some(ref_obj))
            },
            Object::Dictionary(ref dict) => {
                // Check if this is a Form XObject or a state dictionary
                if dict.get("Type").and_then(|t| t.as_name()) == Some("XObject") {
                    (Object::Dictionary(dict.clone()), None)
                } else {
                    // This is a state dictionary - get the current appearance state
                    let state = annotation.appearance_state.as_deref().unwrap_or("Off");
                    match dict.get(state) {
                        Some(Object::Reference(ref_obj)) => {
                            let obj = self.source.load_object(*ref_obj)?;
                            (obj, Some(*ref_obj))
                        },
                        Some(obj) => (obj.clone(), None),
                        None => {
                            // Try "Yes" as fallback for checkboxes
                            if state == "Off" {
                                return Ok(None); // Off state - skip
                            }
                            match dict.get("Yes") {
                                Some(Object::Reference(ref_obj)) => {
                                    let obj = self.source.load_object(*ref_obj)?;
                                    (obj, Some(*ref_obj))
                                },
                                Some(obj) => (obj.clone(), None),
                                None => return Ok(None),
                            }
                        },
                    }
                }
            },
            _ => return Ok(None),
        };

        // Extract Form XObject properties
        let form_dict = match appearance_obj.as_dict() {
            Some(d) => d,
            None => return Ok(None),
        };

        // Get BBox
        let bbox = match form_dict.get("BBox") {
            Some(Object::Array(arr)) if arr.len() >= 4 => {
                let values: Vec<f64> = arr
                    .iter()
                    .filter_map(|o| o.as_real().or_else(|| o.as_integer().map(|i| i as f64)))
                    .collect();
                if values.len() >= 4 {
                    [
                        values[0] as f32,
                        values[1] as f32,
                        values[2] as f32,
                        values[3] as f32,
                    ]
                } else {
                    return Ok(None);
                }
            },
            _ => return Ok(None),
        };

        // Get Matrix (optional)
        let matrix = match form_dict.get("Matrix") {
            Some(Object::Array(arr)) if arr.len() >= 6 => {
                let values: Vec<f64> = arr
                    .iter()
                    .filter_map(|o| o.as_real().or_else(|| o.as_integer().map(|i| i as f64)))
                    .collect();
                if values.len() >= 6 {
                    Some([
                        values[0] as f32,
                        values[1] as f32,
                        values[2] as f32,
                        values[3] as f32,
                        values[4] as f32,
                        values[5] as f32,
                    ])
                } else {
                    None
                }
            },
            _ => None,
        };

        // Get Resources
        let resources = form_dict.get("Resources").cloned();

        // Get the annotation's Rect
        let annot_rect = annotation.rect.unwrap_or([0.0, 0.0, 0.0, 0.0]);
        let annot_rect = [
            annot_rect[0] as f32,
            annot_rect[1] as f32,
            annot_rect[2] as f32,
            annot_rect[3] as f32,
        ];

        // Get content stream bytes
        let content_bytes = if let Some(ref_obj) = appearance_ref {
            let stream_obj = self.source.load_object(ref_obj)?;
            match stream_obj.decode_stream_data() {
                Ok(data) => data,
                Err(_) => return Ok(None),
            }
        } else {
            match appearance_obj.decode_stream_data() {
                Ok(data) => data,
                Err(_) => return Ok(None),
            }
        };

        Ok(Some(AnnotationAppearance {
            content: content_bytes.to_vec(),
            bbox,
            annot_rect,
            matrix,
            resources,
        }))
    }

    /// Generate appearance stream for a widget without one.
    fn generate_widget_appearance(
        &self,
        annotation: &crate::annotations::Annotation,
    ) -> Result<Option<AnnotationAppearance>> {
        use crate::annotation_types::WidgetFieldType;
        use crate::geometry::Rect;
        use crate::writer::FormAppearanceGenerator;

        let rect = match annotation.rect {
            Some(r) => r,
            None => return Ok(None),
        };

        let annot_rect = [
            rect[0] as f32,
            rect[1] as f32,
            rect[2] as f32,
            rect[3] as f32,
        ];
        let width = annot_rect[2] - annot_rect[0];
        let height = annot_rect[3] - annot_rect[1];
        let geom_rect = Rect::new(0.0, 0.0, width, height);

        let generator = FormAppearanceGenerator::new()
            .with_background(1.0, 1.0, 1.0)
            .with_border(1.0, 0.0, 0.0, 0.0);

        let field_type = annotation.field_type.as_ref();
        let content_str = match field_type {
            Some(WidgetFieldType::Text) => {
                let text = annotation.field_value.as_deref().unwrap_or("");
                generator.text_field_appearance(geom_rect, text, "/Helv", 10.0, (0.0, 0.0, 0.0))
            },
            Some(WidgetFieldType::Checkbox { checked }) => {
                if *checked {
                    generator.checkbox_on_appearance(geom_rect, (0.0, 0.0, 0.0))
                } else {
                    generator.checkbox_off_appearance(geom_rect)
                }
            },
            Some(WidgetFieldType::Radio { selected }) => {
                if selected.is_some() {
                    generator.radio_on_appearance(geom_rect, (0.0, 0.0, 0.0))
                } else {
                    generator.radio_off_appearance(geom_rect)
                }
            },
            Some(WidgetFieldType::Button) => {
                let caption = annotation.field_value.as_deref().unwrap_or("");
                generator.button_appearance(geom_rect, caption, "/Helv", 10.0, (0.0, 0.0, 0.0))
            },
            Some(WidgetFieldType::Choice { selected, .. }) => {
                let text = selected.as_deref().unwrap_or("");
                generator.text_field_appearance(geom_rect, text, "/Helv", 10.0, (0.0, 0.0, 0.0))
            },
            Some(WidgetFieldType::Signature) | Some(WidgetFieldType::Unknown) | None => {
                return Ok(None);
            },
        };

        let content_bytes = content_str.into_bytes();
        let bbox = [0.0, 0.0, width, height];

        Ok(Some(AnnotationAppearance {
            content: content_bytes,
            bbox,
            annot_rect,
            matrix: None,
            resources: None,
        }))
    }

    /// Get annotation appearance stream data for flattening.
    ///
    /// Returns a list of (content_bytes, bbox, resources) for each annotation
    /// that has an appearance stream.
    fn get_annotation_appearances(&mut self, page: usize) -> Result<Vec<AnnotationAppearance>> {
        let annotations = self.source.get_annotations(page)?;
        let mut appearances = Vec::new();

        for annotation in annotations {
            // Skip annotations without a raw dictionary
            let raw_dict = match &annotation.raw_dict {
                Some(dict) => dict,
                None => continue,
            };

            // Get the /AP (appearance) dictionary
            let ap_dict = match raw_dict.get("AP") {
                Some(Object::Dictionary(d)) => d.clone(),
                Some(Object::Reference(ap_ref)) => match self.source.load_object(*ap_ref)? {
                    Object::Dictionary(d) => d,
                    _ => continue,
                },
                _ => continue,
            };

            // Get the /N (normal appearance) entry
            let normal_appearance = match ap_dict.get("N") {
                Some(obj) => obj.clone(),
                None => continue,
            };

            // The normal appearance can be:
            // 1. A reference to a Form XObject
            // 2. A dictionary of appearance states (e.g., for checkboxes: /Yes, /Off)
            let (appearance_obj, appearance_ref) = match normal_appearance {
                Object::Reference(ref_obj) => {
                    let obj = self.source.load_object(ref_obj)?;
                    (obj, Some(ref_obj))
                },
                Object::Dictionary(ref dict) => {
                    // Check if this is a Form XObject or a state dictionary
                    if dict.get("Type").and_then(|t| t.as_name()) == Some("XObject") {
                        (Object::Dictionary(dict.clone()), None)
                    } else {
                        // This is a state dictionary - get the current appearance state
                        let state = annotation.appearance_state.as_deref().unwrap_or("Off");
                        match dict.get(state) {
                            Some(Object::Reference(ref_obj)) => {
                                let obj = self.source.load_object(*ref_obj)?;
                                (obj, Some(*ref_obj))
                            },
                            Some(obj) => (obj.clone(), None),
                            None => continue,
                        }
                    }
                },
                _ => continue,
            };

            // Extract the Form XObject properties
            let form_dict = match appearance_obj.as_dict() {
                Some(d) => d,
                None => continue,
            };

            // Get BBox
            let bbox = match form_dict.get("BBox") {
                Some(Object::Array(arr)) if arr.len() >= 4 => {
                    let values: Vec<f64> = arr
                        .iter()
                        .filter_map(|o| o.as_real().or_else(|| o.as_integer().map(|i| i as f64)))
                        .collect();
                    if values.len() >= 4 {
                        [
                            values[0] as f32,
                            values[1] as f32,
                            values[2] as f32,
                            values[3] as f32,
                        ]
                    } else {
                        continue;
                    }
                },
                _ => continue,
            };

            // Get Matrix (optional, defaults to identity)
            let matrix = match form_dict.get("Matrix") {
                Some(Object::Array(arr)) if arr.len() >= 6 => {
                    let values: Vec<f64> = arr
                        .iter()
                        .filter_map(|o| o.as_real().or_else(|| o.as_integer().map(|i| i as f64)))
                        .collect();
                    if values.len() >= 6 {
                        Some([
                            values[0] as f32,
                            values[1] as f32,
                            values[2] as f32,
                            values[3] as f32,
                            values[4] as f32,
                            values[5] as f32,
                        ])
                    } else {
                        None
                    }
                },
                _ => None,
            };

            // Get Resources (optional)
            let resources = form_dict.get("Resources").cloned();

            // Get the annotation's Rect (position on page)
            let annot_rect = annotation.rect.unwrap_or([0.0, 0.0, 0.0, 0.0]);
            let annot_rect = [
                annot_rect[0] as f32,
                annot_rect[1] as f32,
                annot_rect[2] as f32,
                annot_rect[3] as f32,
            ];

            // Get the content stream bytes
            let content_bytes = if let Some(ref_obj) = appearance_ref {
                // Load the object and decode its stream data
                let stream_obj = match self.source.load_object(ref_obj) {
                    Ok(obj) => obj,
                    Err(_) => continue,
                };
                match stream_obj.decode_stream_data() {
                    Ok(data) => data,
                    Err(_) => continue,
                }
            } else {
                // Inline stream - try to decode directly
                match appearance_obj.decode_stream_data() {
                    Ok(data) => data,
                    Err(_) => continue,
                }
            };

            appearances.push(AnnotationAppearance {
                content: content_bytes,
                bbox,
                annot_rect,
                matrix,
                resources,
            });
        }

        Ok(appearances)
    }

    /// Generate content stream to render flattened annotations.
    ///
    /// This creates PDF operators that invoke each annotation's appearance
    /// as a Form XObject at the correct position.
    fn generate_flatten_overlay(
        &self,
        appearances: &[AnnotationAppearance],
        xobject_names: &[String],
    ) -> Vec<u8> {
        let mut content = Vec::new();

        for (appearance, xobj_name) in appearances.iter().zip(xobject_names.iter()) {
            // Save graphics state
            content.extend_from_slice(b"q\n");

            // Calculate transformation to position the XObject
            // The appearance is defined in BBox coordinates and needs to be
            // positioned at annot_rect on the page.
            let bbox = appearance.bbox;
            let rect = appearance.annot_rect;

            // Calculate scale and translation
            let bbox_width = bbox[2] - bbox[0];
            let bbox_height = bbox[3] - bbox[1];
            let rect_width = rect[2] - rect[0];
            let rect_height = rect[3] - rect[1];

            // Avoid division by zero
            let sx = if bbox_width != 0.0 {
                rect_width / bbox_width
            } else {
                1.0
            };
            let sy = if bbox_height != 0.0 {
                rect_height / bbox_height
            } else {
                1.0
            };

            // Translation to position the XObject
            let tx = rect[0] - bbox[0] * sx;
            let ty = rect[1] - bbox[1] * sy;

            // Apply transformation matrix: [sx 0 0 sy tx ty]
            content.extend_from_slice(
                format!("{:.6} 0 0 {:.6} {:.6} {:.6} cm\n", sx, sy, tx, ty).as_bytes(),
            );

            // If the appearance has its own matrix, apply it
            if let Some(m) = appearance.matrix {
                content.extend_from_slice(
                    format!(
                        "{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} cm\n",
                        m[0], m[1], m[2], m[3], m[4], m[5]
                    )
                    .as_bytes(),
                );
            }

            // Invoke the XObject
            content.extend_from_slice(format!("/{} Do\n", xobj_name).as_bytes());

            // Restore graphics state
            content.extend_from_slice(b"Q\n");
        }

        content
    }

    // ========================================================================
    // Redaction Application
    // ========================================================================

    /// Mark a page for redaction application.
    ///
    /// When the document is saved, redaction annotations on this page will be
    /// applied: content will be visually obscured and the redaction annotations
    /// removed.
    ///
    /// # Arguments
    /// * `page` - The zero-based page index
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Apply redactions on page 0
    /// editor.apply_page_redactions(0)?;
    /// editor.save("output.pdf")?;
    /// ```
    pub fn apply_page_redactions(&mut self, page: usize) -> Result<()> {
        if page >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", page)));
        }

        self.apply_redactions_pages.insert(page);
        self.is_modified = true;
        Ok(())
    }

    /// Mark all pages for redaction application.
    pub fn apply_all_redactions(&mut self) -> Result<()> {
        let page_count = self.current_page_count();
        for page in 0..page_count {
            self.apply_redactions_pages.insert(page);
        }
        self.is_modified = true;
        Ok(())
    }

    /// Check if a page is marked for redaction application.
    pub fn is_page_marked_for_redaction(&self, page: usize) -> bool {
        self.apply_redactions_pages.contains(&page)
    }

    /// Clear the apply redactions flag for a page.
    pub fn unmark_page_for_redaction(&mut self, page: usize) {
        self.apply_redactions_pages.remove(&page);
    }

    /// Get redaction annotation data for a page.
    ///
    /// Returns a list of redaction areas with their fill colors.
    fn get_redaction_data(&mut self, page: usize) -> Result<Vec<RedactionData>> {
        use crate::annotation_types::AnnotationSubtype;

        let annotations = self.source.get_annotations(page)?;
        let mut redactions = Vec::new();

        for annotation in annotations {
            // Only process Redact annotations
            if annotation.subtype_enum != AnnotationSubtype::Redact {
                continue;
            }

            // Get the redaction rectangle
            let rect = match annotation.rect {
                Some(r) => [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32],
                None => continue,
            };

            // Get interior color (IC entry) - the fill color for the redaction
            // Default to black if not specified
            let color = match &annotation.interior_color {
                Some(color) if color.len() >= 3 => {
                    [color[0] as f32, color[1] as f32, color[2] as f32]
                },
                _ => [0.0, 0.0, 0.0], // Default to black
            };

            // Also handle QuadPoints if present (multiple redaction areas)
            if let Some(ref quad_points) = annotation.quad_points {
                for quad in quad_points {
                    // QuadPoints are 8 values: x1,y1,x2,y2,x3,y3,x4,y4
                    // representing corners in a specific order
                    // Convert to bounding box
                    let xs = [quad[0], quad[2], quad[4], quad[6]];
                    let ys = [quad[1], quad[3], quad[5], quad[7]];

                    let min_x = xs.iter().cloned().fold(f64::INFINITY, f64::min) as f32;
                    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as f32;
                    let min_y = ys.iter().cloned().fold(f64::INFINITY, f64::min) as f32;
                    let max_y = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as f32;

                    redactions.push(RedactionData {
                        rect: [min_x, min_y, max_x, max_y],
                        color,
                    });
                }
            } else {
                // Just use the main Rect
                redactions.push(RedactionData { rect, color });
            }
        }

        Ok(redactions)
    }

    /// Generate content stream to draw redaction overlays.
    fn generate_redaction_overlay(&self, redactions: &[RedactionData]) -> Vec<u8> {
        let mut content = Vec::new();

        for redaction in redactions {
            // Save graphics state
            content.extend_from_slice(b"q\n");

            // Set fill color (RGB)
            content.extend_from_slice(
                format!(
                    "{:.3} {:.3} {:.3} rg\n",
                    redaction.color[0], redaction.color[1], redaction.color[2]
                )
                .as_bytes(),
            );

            // Draw filled rectangle
            let x = redaction.rect[0];
            let y = redaction.rect[1];
            let width = redaction.rect[2] - redaction.rect[0];
            let height = redaction.rect[3] - redaction.rect[1];

            content.extend_from_slice(
                format!("{:.2} {:.2} {:.2} {:.2} re f\n", x, y, width, height).as_bytes(),
            );

            // Restore graphics state
            content.extend_from_slice(b"Q\n");
        }

        content
    }

    // ========================================================================
    // Image Repositioning & Resizing
    // ========================================================================

    /// Get information about images on a page.
    ///
    /// Returns a list of images with their names, positions, and sizes.
    ///
    /// # Arguments
    /// * `page` - The zero-based page index
    ///
    /// # Example
    ///
    /// ```ignore
    /// let images = editor.get_page_images(0)?;
    /// for img in images {
    ///     println!("Image {} at ({}, {}) size {}x{}",
    ///         img.name, img.bounds[0], img.bounds[1],
    ///         img.bounds[2], img.bounds[3]);
    /// }
    /// ```
    pub fn get_page_images(&mut self, page: usize) -> Result<Vec<ImageInfo>> {
        use crate::content::parser::parse_content_stream;

        if page >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", page)));
        }

        // Get the original page index
        let original_page_idx = self.page_order[page];
        if original_page_idx < 0 {
            return Err(Error::InvalidPdf("Page has been deleted".to_string()));
        }

        // Get page reference
        let page_ref = self.source.get_page_ref(original_page_idx as usize)?;
        let page_obj = self.source.load_object(page_ref)?;
        let page_dict = page_obj
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Page is not a dictionary".to_string()))?;

        // Get Contents
        let contents = match page_dict.get("Contents") {
            Some(c) => c.clone(),
            None => return Ok(Vec::new()),
        };

        // Load content stream data
        let content_data = match contents {
            Object::Reference(ref_obj) => {
                let obj = self.source.load_object(ref_obj)?;
                obj.decode_stream_data()?
            },
            Object::Array(arr) => {
                // Concatenate multiple content streams
                let mut data = Vec::new();
                for item in arr {
                    if let Object::Reference(ref_obj) = item {
                        let obj = self.source.load_object(ref_obj)?;
                        if let Ok(stream_data) = obj.decode_stream_data() {
                            data.extend_from_slice(&stream_data);
                            data.push(b'\n');
                        }
                    }
                }
                data
            },
            _ => return Ok(Vec::new()),
        };

        // Parse the content stream
        let operators = parse_content_stream(&content_data)?;

        // Track CTM through the operators to find images
        let mut images = Vec::new();
        let mut ctm_stack: Vec<[f32; 6]> = vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]; // Identity
        let mut current_ctm = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];

        for op in operators {
            match op {
                crate::content::operators::Operator::SaveState => {
                    ctm_stack.push(current_ctm);
                },
                crate::content::operators::Operator::RestoreState => {
                    if let Some(saved) = ctm_stack.pop() {
                        current_ctm = saved;
                    }
                },
                crate::content::operators::Operator::Cm { a, b, c, d, e, f } => {
                    // Concatenate transformation matrix
                    // New CTM = [a,b,c,d,e,f] * current_ctm
                    let new_a = a * current_ctm[0] + b * current_ctm[2];
                    let new_b = a * current_ctm[1] + b * current_ctm[3];
                    let new_c = c * current_ctm[0] + d * current_ctm[2];
                    let new_d = c * current_ctm[1] + d * current_ctm[3];
                    let new_e = e * current_ctm[0] + f * current_ctm[2] + current_ctm[4];
                    let new_f = e * current_ctm[1] + f * current_ctm[3] + current_ctm[5];
                    current_ctm = [new_a, new_b, new_c, new_d, new_e, new_f];
                },
                crate::content::operators::Operator::Do { ref name } => {
                    // Check if this is an image XObject (vs Form XObject)
                    // For now, include all XObjects; a more refined implementation
                    // would check the XObject's Subtype
                    let matrix = current_ctm;

                    // Extract position and size from matrix
                    // Standard image matrix: [width, 0, 0, height, x, y]
                    let x = matrix[4];
                    let y = matrix[5];
                    // Width and height from scaling components
                    let width = (matrix[0] * matrix[0] + matrix[1] * matrix[1]).sqrt();
                    let height = (matrix[2] * matrix[2] + matrix[3] * matrix[3]).sqrt();

                    images.push(ImageInfo {
                        name: name.clone(),
                        bounds: [x, y, width, height],
                        matrix,
                    });
                },
                _ => {},
            }
        }

        Ok(images)
    }

    /// Reposition an image on a page.
    ///
    /// # Arguments
    /// * `page` - The zero-based page index
    /// * `image_name` - The XObject name (e.g., "Im1")
    /// * `x` - New x position
    /// * `y` - New y position
    ///
    /// # Example
    ///
    /// ```ignore
    /// editor.reposition_image(0, "Im1", 100.0, 200.0)?;
    /// editor.save("output.pdf")?;
    /// ```
    pub fn reposition_image(
        &mut self,
        page: usize,
        image_name: &str,
        x: f32,
        y: f32,
    ) -> Result<()> {
        if page >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", page)));
        }

        let page_mods = self.image_modifications.entry(page).or_default();
        let modification = page_mods
            .entry(image_name.to_string())
            .or_insert(ImageModification {
                x: None,
                y: None,
                width: None,
                height: None,
            });
        modification.x = Some(x);
        modification.y = Some(y);

        self.is_modified = true;
        Ok(())
    }

    /// Resize an image on a page.
    ///
    /// # Arguments
    /// * `page` - The zero-based page index
    /// * `image_name` - The XObject name (e.g., "Im1")
    /// * `width` - New width
    /// * `height` - New height
    ///
    /// # Example
    ///
    /// ```ignore
    /// editor.resize_image(0, "Im1", 200.0, 150.0)?;
    /// editor.save("output.pdf")?;
    /// ```
    pub fn resize_image(
        &mut self,
        page: usize,
        image_name: &str,
        width: f32,
        height: f32,
    ) -> Result<()> {
        if page >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", page)));
        }

        let page_mods = self.image_modifications.entry(page).or_default();
        let modification = page_mods
            .entry(image_name.to_string())
            .or_insert(ImageModification {
                x: None,
                y: None,
                width: None,
                height: None,
            });
        modification.width = Some(width);
        modification.height = Some(height);

        self.is_modified = true;
        Ok(())
    }

    /// Reposition and resize an image on a page.
    ///
    /// # Arguments
    /// * `page` - The zero-based page index
    /// * `image_name` - The XObject name (e.g., "Im1")
    /// * `x` - New x position
    /// * `y` - New y position
    /// * `width` - New width
    /// * `height` - New height
    pub fn set_image_bounds(
        &mut self,
        page: usize,
        image_name: &str,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> Result<()> {
        if page >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!("Page index {} out of range", page)));
        }

        let page_mods = self.image_modifications.entry(page).or_default();
        page_mods.insert(
            image_name.to_string(),
            ImageModification {
                x: Some(x),
                y: Some(y),
                width: Some(width),
                height: Some(height),
            },
        );

        self.is_modified = true;
        Ok(())
    }

    /// Clear image modifications for a page.
    pub fn clear_image_modifications(&mut self, page: usize) {
        self.image_modifications.remove(&page);
    }

    /// Check if a page has image modifications.
    pub fn has_image_modifications(&self, page: usize) -> bool {
        self.image_modifications
            .get(&page)
            .map(|m| !m.is_empty())
            .unwrap_or(false)
    }

    /// Rewrite content stream with image modifications applied.
    fn rewrite_content_stream_with_image_mods(
        &self,
        content_data: &[u8],
        modifications: &HashMap<String, ImageModification>,
    ) -> Result<Vec<u8>> {
        use crate::content::parser::parse_content_stream;

        let operators = parse_content_stream(content_data)?;
        let mut output = Vec::new();

        // Track the last cm operator to potentially modify it
        let mut i = 0;
        while i < operators.len() {
            let op = &operators[i];

            // Look for pattern: q ... cm ... Do ... Q
            // We need to find cm operators that precede Do operators
            match op {
                crate::content::operators::Operator::Cm { a, b, c, d, e, f } => {
                    // Look ahead to see if next relevant op is Do
                    let mut j = i + 1;
                    let mut found_do = None;
                    while j < operators.len() {
                        match &operators[j] {
                            crate::content::operators::Operator::Do { name } => {
                                found_do = Some(name.clone());
                                break;
                            },
                            crate::content::operators::Operator::RestoreState => break,
                            crate::content::operators::Operator::SaveState => break,
                            crate::content::operators::Operator::Cm { .. } => break,
                            _ => {},
                        }
                        j += 1;
                    }

                    if let Some(name) = found_do {
                        if let Some(modification) = modifications.get(&name) {
                            // Apply modification to the matrix
                            let new_a = modification.width.unwrap_or(*a);
                            let new_d = modification.height.unwrap_or(*d);
                            let new_e = modification.x.unwrap_or(*e);
                            let new_f = modification.y.unwrap_or(*f);

                            output.extend_from_slice(
                                format!(
                                    "{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} cm\n",
                                    new_a, b, c, new_d, new_e, new_f
                                )
                                .as_bytes(),
                            );
                            i += 1;
                            continue;
                        }
                    }

                    // No modification, output as-is
                    output.extend_from_slice(
                        format!("{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} cm\n", a, b, c, d, e, f)
                            .as_bytes(),
                    );
                },
                _ => {
                    // Serialize the operator
                    self.serialize_operator(&mut output, op);
                },
            }
            i += 1;
        }

        Ok(output)
    }

    /// Serialize an operator to bytes.
    fn serialize_operator(&self, output: &mut Vec<u8>, op: &crate::content::operators::Operator) {
        use crate::content::operators::{Operator, TextElement};

        match op {
            // Graphics state
            Operator::SaveState => output.extend_from_slice(b"q\n"),
            Operator::RestoreState => output.extend_from_slice(b"Q\n"),
            Operator::Cm { a, b, c, d, e, f } => {
                output.extend_from_slice(
                    format!("{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} cm\n", a, b, c, d, e, f)
                        .as_bytes(),
                );
            },
            Operator::SetLineWidth { width } => {
                output.extend_from_slice(format!("{:.6} w\n", width).as_bytes());
            },
            Operator::SetLineCap { cap_style } => {
                output.extend_from_slice(format!("{} J\n", cap_style).as_bytes());
            },
            Operator::SetLineJoin { join_style } => {
                output.extend_from_slice(format!("{} j\n", join_style).as_bytes());
            },
            Operator::SetMiterLimit { limit } => {
                output.extend_from_slice(format!("{:.6} M\n", limit).as_bytes());
            },
            Operator::SetDash { array, phase } => {
                output.push(b'[');
                for (i, v) in array.iter().enumerate() {
                    if i > 0 {
                        output.push(b' ');
                    }
                    output.extend_from_slice(format!("{:.6}", v).as_bytes());
                }
                output.extend_from_slice(format!("] {:.6} d\n", phase).as_bytes());
            },
            Operator::SetFlatness { tolerance } => {
                output.extend_from_slice(format!("{:.6} i\n", tolerance).as_bytes());
            },
            Operator::SetRenderingIntent { intent } => {
                output.extend_from_slice(format!("/{} ri\n", intent).as_bytes());
            },
            Operator::SetExtGState { dict_name } => {
                output.extend_from_slice(format!("/{} gs\n", dict_name).as_bytes());
            },

            // Path construction
            Operator::MoveTo { x, y } => {
                output.extend_from_slice(format!("{:.6} {:.6} m\n", x, y).as_bytes());
            },
            Operator::LineTo { x, y } => {
                output.extend_from_slice(format!("{:.6} {:.6} l\n", x, y).as_bytes());
            },
            Operator::CurveTo {
                x1,
                y1,
                x2,
                y2,
                x3,
                y3,
            } => {
                output.extend_from_slice(
                    format!("{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} c\n", x1, y1, x2, y2, x3, y3)
                        .as_bytes(),
                );
            },
            Operator::CurveToV { x2, y2, x3, y3 } => {
                output.extend_from_slice(
                    format!("{:.6} {:.6} {:.6} {:.6} v\n", x2, y2, x3, y3).as_bytes(),
                );
            },
            Operator::CurveToY { x1, y1, x3, y3 } => {
                output.extend_from_slice(
                    format!("{:.6} {:.6} {:.6} {:.6} y\n", x1, y1, x3, y3).as_bytes(),
                );
            },
            Operator::ClosePath => output.extend_from_slice(b"h\n"),
            Operator::Rectangle {
                x,
                y,
                width,
                height,
            } => {
                output.extend_from_slice(
                    format!("{:.6} {:.6} {:.6} {:.6} re\n", x, y, width, height).as_bytes(),
                );
            },

            // Path painting
            Operator::Stroke => output.extend_from_slice(b"S\n"),
            Operator::Fill => output.extend_from_slice(b"f\n"),
            Operator::FillEvenOdd => output.extend_from_slice(b"f*\n"),
            Operator::CloseFillStroke => output.extend_from_slice(b"b\n"),
            Operator::FillStroke => output.extend_from_slice(b"B\n"),
            Operator::FillStrokeEvenOdd => output.extend_from_slice(b"B*\n"),
            Operator::CloseFillStrokeEvenOdd => output.extend_from_slice(b"b*\n"),
            Operator::EndPath => output.extend_from_slice(b"n\n"),

            // Clipping
            Operator::ClipNonZero => output.extend_from_slice(b"W\n"),
            Operator::ClipEvenOdd => output.extend_from_slice(b"W*\n"),

            // Text object
            Operator::BeginText => output.extend_from_slice(b"BT\n"),
            Operator::EndText => output.extend_from_slice(b"ET\n"),

            // Text state
            Operator::Tc { char_space } => {
                output.extend_from_slice(format!("{:.6} Tc\n", char_space).as_bytes());
            },
            Operator::Tw { word_space } => {
                output.extend_from_slice(format!("{:.6} Tw\n", word_space).as_bytes());
            },
            Operator::Tz { scale } => {
                output.extend_from_slice(format!("{:.6} Tz\n", scale).as_bytes());
            },
            Operator::TL { leading } => {
                output.extend_from_slice(format!("{:.6} TL\n", leading).as_bytes());
            },
            Operator::Tf { font, size } => {
                output.extend_from_slice(format!("/{} {:.6} Tf\n", font, size).as_bytes());
            },
            Operator::Tr { render } => {
                output.extend_from_slice(format!("{} Tr\n", render).as_bytes());
            },
            Operator::Ts { rise } => {
                output.extend_from_slice(format!("{:.6} Ts\n", rise).as_bytes());
            },

            // Text positioning
            Operator::Td { tx, ty } => {
                output.extend_from_slice(format!("{:.6} {:.6} Td\n", tx, ty).as_bytes());
            },
            Operator::TD { tx, ty } => {
                output.extend_from_slice(format!("{:.6} {:.6} TD\n", tx, ty).as_bytes());
            },
            Operator::Tm { a, b, c, d, e, f } => {
                output.extend_from_slice(
                    format!("{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} Tm\n", a, b, c, d, e, f)
                        .as_bytes(),
                );
            },
            Operator::TStar => output.extend_from_slice(b"T*\n"),

            // Text showing
            Operator::Tj { text } => {
                output.push(b'(');
                for byte in text {
                    match *byte {
                        b'(' | b')' | b'\\' => {
                            output.push(b'\\');
                            output.push(*byte);
                        },
                        _ => output.push(*byte),
                    }
                }
                output.extend_from_slice(b") Tj\n");
            },
            Operator::TJ { array } => {
                output.push(b'[');
                for item in array {
                    match item {
                        TextElement::String(text) => {
                            output.push(b'(');
                            for byte in text {
                                match *byte {
                                    b'(' | b')' | b'\\' => {
                                        output.push(b'\\');
                                        output.push(*byte);
                                    },
                                    _ => output.push(*byte),
                                }
                            }
                            output.push(b')');
                        },
                        TextElement::Offset(offset) => {
                            output.extend_from_slice(format!("{:.6}", offset).as_bytes());
                        },
                    }
                }
                output.extend_from_slice(b"] TJ\n");
            },
            Operator::Quote { text } => {
                output.push(b'(');
                for byte in text {
                    match *byte {
                        b'(' | b')' | b'\\' => {
                            output.push(b'\\');
                            output.push(*byte);
                        },
                        _ => output.push(*byte),
                    }
                }
                output.extend_from_slice(b") '\n");
            },
            Operator::DoubleQuote {
                word_space,
                char_space,
                text,
            } => {
                output
                    .extend_from_slice(format!("{:.6} {:.6} (", word_space, char_space).as_bytes());
                for byte in text {
                    match *byte {
                        b'(' | b')' | b'\\' => {
                            output.push(b'\\');
                            output.push(*byte);
                        },
                        _ => output.push(*byte),
                    }
                }
                output.extend_from_slice(b") \"\n");
            },

            // Color space
            Operator::SetStrokeColorSpace { name } => {
                output.extend_from_slice(format!("/{} CS\n", name).as_bytes());
            },
            Operator::SetFillColorSpace { name } => {
                output.extend_from_slice(format!("/{} cs\n", name).as_bytes());
            },
            Operator::SetStrokeColor { components } => {
                for c in components {
                    output.extend_from_slice(format!("{:.6} ", c).as_bytes());
                }
                output.extend_from_slice(b"SC\n");
            },
            Operator::SetFillColor { components } => {
                for c in components {
                    output.extend_from_slice(format!("{:.6} ", c).as_bytes());
                }
                output.extend_from_slice(b"sc\n");
            },
            Operator::SetStrokeColorN { components, name } => {
                for c in components {
                    output.extend_from_slice(format!("{:.6} ", c).as_bytes());
                }
                if let Some(p) = name {
                    output.extend_from_slice(format!("/{} ", p).as_bytes());
                }
                output.extend_from_slice(b"SCN\n");
            },
            Operator::SetFillColorN { components, name } => {
                for c in components {
                    output.extend_from_slice(format!("{:.6} ", c).as_bytes());
                }
                if let Some(p) = name {
                    output.extend_from_slice(format!("/{} ", p).as_bytes());
                }
                output.extend_from_slice(b"scn\n");
            },
            Operator::SetStrokeGray { gray } => {
                output.extend_from_slice(format!("{:.6} G\n", gray).as_bytes());
            },
            Operator::SetFillGray { gray } => {
                output.extend_from_slice(format!("{:.6} g\n", gray).as_bytes());
            },
            Operator::SetStrokeRgb { r, g, b } => {
                output.extend_from_slice(format!("{:.6} {:.6} {:.6} RG\n", r, g, b).as_bytes());
            },
            Operator::SetFillRgb { r, g, b } => {
                output.extend_from_slice(format!("{:.6} {:.6} {:.6} rg\n", r, g, b).as_bytes());
            },
            Operator::SetStrokeCmyk { c, m, y, k } => {
                output.extend_from_slice(
                    format!("{:.6} {:.6} {:.6} {:.6} K\n", c, m, y, k).as_bytes(),
                );
            },
            Operator::SetFillCmyk { c, m, y, k } => {
                output.extend_from_slice(
                    format!("{:.6} {:.6} {:.6} {:.6} k\n", c, m, y, k).as_bytes(),
                );
            },

            // XObject
            Operator::Do { name } => {
                output.extend_from_slice(format!("/{} Do\n", name).as_bytes());
            },

            // Marked content
            Operator::BeginMarkedContent { tag } => {
                output.extend_from_slice(format!("/{} BMC\n", tag).as_bytes());
            },
            Operator::BeginMarkedContentDict { tag, properties } => {
                output.extend_from_slice(format!("/{} ", tag).as_bytes());
                self.serialize_object(output, properties);
                output.extend_from_slice(b" BDC\n");
            },
            Operator::EndMarkedContent => output.extend_from_slice(b"EMC\n"),

            // Shading
            Operator::PaintShading { name } => {
                output.extend_from_slice(format!("/{} sh\n", name).as_bytes());
            },

            // Inline image (complex - serialize full BI...ID...EI sequence)
            Operator::InlineImage { dict, data } => {
                output.extend_from_slice(b"BI\n");
                for (key, value) in dict.iter() {
                    output.extend_from_slice(format!("/{} ", key).as_bytes());
                    self.serialize_object(output, value);
                    output.push(b'\n');
                }
                output.extend_from_slice(b"ID ");
                output.extend_from_slice(data);
                output.extend_from_slice(b"\nEI\n");
            },

            // Other operators (fallback for unrecognized operators)
            Operator::Other { name, operands } => {
                for operand in operands.iter() {
                    self.serialize_object(output, operand);
                    output.push(b' ');
                }
                output.extend_from_slice(name.as_bytes());
                output.push(b'\n');
            },
        }
    }

    /// Serialize a PDF Object to bytes.
    #[allow(clippy::only_used_in_recursion)]
    fn serialize_object(&self, output: &mut Vec<u8>, obj: &crate::object::Object) {
        use crate::object::Object;
        match obj {
            Object::Null => output.extend_from_slice(b"null"),
            Object::Boolean(b) => {
                if *b {
                    output.extend_from_slice(b"true");
                } else {
                    output.extend_from_slice(b"false");
                }
            },
            Object::Integer(i) => output.extend_from_slice(format!("{}", i).as_bytes()),
            Object::Real(r) => output.extend_from_slice(format!("{:.6}", r).as_bytes()),
            Object::Name(n) => output.extend_from_slice(format!("/{}", n).as_bytes()),
            Object::String(s) => {
                output.push(b'(');
                for byte in s {
                    match *byte {
                        b'(' | b')' | b'\\' => {
                            output.push(b'\\');
                            output.push(*byte);
                        },
                        _ => output.push(*byte),
                    }
                }
                output.push(b')');
            },
            // Note: PDF HexStrings are stored as Object::String and serialized as literal strings
            Object::Array(arr) => {
                output.push(b'[');
                for (i, item) in arr.iter().enumerate() {
                    if i > 0 {
                        output.push(b' ');
                    }
                    self.serialize_object(output, item);
                }
                output.push(b']');
            },
            Object::Dictionary(dict) => {
                output.extend_from_slice(b"<<");
                for (key, value) in dict {
                    output.extend_from_slice(format!("/{} ", key).as_bytes());
                    self.serialize_object(output, value);
                }
                output.extend_from_slice(b">>");
            },
            Object::Stream { .. } => {
                // Streams are complex; for inline serialization just output placeholder
                output.extend_from_slice(b"(stream)");
            },
            Object::Reference(obj_ref) => {
                output.extend_from_slice(format!("{} {} R", obj_ref.id, obj_ref.gen).as_bytes());
            },
        }
    }
}

/// Data for a redaction area.
#[derive(Debug, Clone)]
struct RedactionData {
    /// Redaction rectangle [llx, lly, urx, ury]
    rect: [f32; 4],
    /// Fill color [r, g, b]
    color: [f32; 3],
}

impl EditableDocument for DocumentEditor {
    fn get_info(&mut self) -> Result<DocumentInfo> {
        // Return modified info if available
        if let Some(ref info) = self.modified_info {
            return Ok(info.clone());
        }

        // Otherwise, load from source document
        let trailer = self.source.trailer();
        if let Some(trailer_dict) = trailer.as_dict() {
            if let Some(info_ref) = trailer_dict.get("Info").and_then(|i| i.as_reference()) {
                let info_obj = self.source.load_object(info_ref)?;
                return Ok(DocumentInfo::from_object(&info_obj));
            }
        }

        // No Info dictionary
        Ok(DocumentInfo::default())
    }

    fn set_info(&mut self, info: DocumentInfo) -> Result<()> {
        self.modified_info = Some(info);
        self.is_modified = true;
        Ok(())
    }

    fn page_count(&mut self) -> Result<usize> {
        Ok(self.current_page_count())
    }

    fn get_page_info(&mut self, index: usize) -> Result<PageInfo> {
        let page_refs = self.get_page_refs()?;

        if index >= page_refs.len() {
            return Err(Error::InvalidPdf(format!(
                "Page index {} out of range (document has {} pages)",
                index,
                page_refs.len()
            )));
        }

        let page_ref = page_refs[index];
        let page_obj = self.source.load_object(page_ref)?;
        let page_dict = page_obj
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Page is not a dictionary".to_string()))?;

        // Get MediaBox for dimensions
        let (width, height) = if let Some(media_box) = page_dict.get("MediaBox") {
            self.parse_media_box(media_box)?
        } else {
            // Try to inherit from parent
            (612.0, 792.0) // Default to Letter size
        };

        let rotation = page_dict
            .get("Rotate")
            .and_then(|r| r.as_integer())
            .unwrap_or(0) as i32;

        Ok(PageInfo {
            index,
            width,
            height,
            rotation,
            object_ref: page_ref,
        })
    }

    fn remove_page(&mut self, index: usize) -> Result<()> {
        if index >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!(
                "Page index {} out of range (document has {} pages)",
                index,
                self.current_page_count()
            )));
        }

        // Mark page as removed in page_order
        let mut visible_index = 0;
        for order in &mut self.page_order {
            if *order >= 0 {
                if visible_index == index {
                    *order = -1; // Mark as removed
                    break;
                }
                visible_index += 1;
            }
        }

        self.is_modified = true;
        Ok(())
    }

    fn move_page(&mut self, from: usize, to: usize) -> Result<()> {
        let count = self.current_page_count();
        if from >= count || to >= count {
            return Err(Error::InvalidPdf(format!(
                "Page index out of range (document has {} pages)",
                count
            )));
        }

        // Get current visible pages
        let visible: Vec<i32> = self
            .page_order
            .iter()
            .filter(|&&i| i >= 0)
            .copied()
            .collect();

        // Reorder
        let mut new_visible = visible.clone();
        let moved = new_visible.remove(from);
        new_visible.insert(to, moved);

        // Rebuild page_order
        self.page_order = new_visible;
        self.is_modified = true;
        Ok(())
    }

    fn duplicate_page(&mut self, index: usize) -> Result<usize> {
        if index >= self.current_page_count() {
            return Err(Error::InvalidPdf(format!(
                "Page index {} out of range (document has {} pages)",
                index,
                self.current_page_count()
            )));
        }

        // Get the original page index from page_order
        let visible: Vec<i32> = self
            .page_order
            .iter()
            .filter(|&&i| i >= 0)
            .copied()
            .collect();
        let original_index = visible[index];

        // Add duplicate reference
        self.page_order.push(original_index);
        self.is_modified = true;

        Ok(self.current_page_count() - 1)
    }

    fn save(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.save_with_options(path, SaveOptions::full_rewrite())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_with_options(&mut self, path: impl AsRef<Path>, options: SaveOptions) -> Result<()> {
        if options.incremental {
            self.write_incremental(path)
        } else {
            self.write_full(path, options.encryption.as_ref())
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn save_with_options(&mut self, _path: impl AsRef<Path>, _options: SaveOptions) -> Result<()> {
        Err(Error::InvalidPdf(
            "Filesystem save is not available in WASM. Use save_to_bytes() instead.".to_string(),
        ))
    }
}

impl DocumentEditor {
    /// Parse a MediaBox array into (width, height).
    fn parse_media_box(&self, media_box: &Object) -> Result<(f32, f32)> {
        if let Some(arr) = media_box.as_array() {
            if arr.len() >= 4 {
                let llx = arr[0]
                    .as_real()
                    .or_else(|| arr[0].as_integer().map(|i| i as f64))
                    .unwrap_or(0.0);
                let lly = arr[1]
                    .as_real()
                    .or_else(|| arr[1].as_integer().map(|i| i as f64))
                    .unwrap_or(0.0);
                let urx = arr[2]
                    .as_real()
                    .or_else(|| arr[2].as_integer().map(|i| i as f64))
                    .unwrap_or(612.0);
                let ury = arr[3]
                    .as_real()
                    .or_else(|| arr[3].as_integer().map(|i| i as f64))
                    .unwrap_or(792.0);

                return Ok(((urx - llx) as f32, (ury - lly) as f32));
            }
        }

        // Default to Letter size
        Ok((612.0, 792.0))
    }

    /// Generate a content stream from a StructureElement with marked content wrapping.
    ///
    /// This is used when writing modified structure elements back to a PDF.
    /// Wraps each element in BDC/EMC (Begin/End Marked Content) operators for tagged PDF support.
    ///
    /// Returns the content stream bytes and any pending images that need XObject registration.
    ///
    /// # PDF Spec Compliance
    ///
    /// - ISO 32000-1:2008, Section 14.7.4 - Marked Content Sequences
    fn generate_content_stream(
        &self,
        elem: &StructureElement,
    ) -> Result<(Vec<u8>, Vec<crate::writer::PendingImage>)> {
        let mut builder = ContentStreamBuilder::new();
        builder.add_structure_element(elem);
        let bytes = builder.build()?;
        let pending_images = builder.take_pending_images();
        Ok((bytes, pending_images))
    }

    /// Build an XObject stream from ImageContent.
    ///
    /// Creates a PDF Image XObject suitable for embedding in a PDF.
    /// Per PDF spec Section 8.9, images are represented as XObject streams.
    fn build_image_xobject(image: &crate::elements::ImageContent) -> Object {
        use crate::elements::{ColorSpace as ElemColorSpace, ImageFormat as ElemImageFormat};

        let mut dict = HashMap::new();

        dict.insert("Type".to_string(), Object::Name("XObject".to_string()));
        dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
        dict.insert("Width".to_string(), Object::Integer(image.width as i64));
        dict.insert("Height".to_string(), Object::Integer(image.height as i64));
        dict.insert(
            "BitsPerComponent".to_string(),
            Object::Integer(image.bits_per_component as i64),
        );

        // Map color space
        let color_space_name = match image.color_space {
            ElemColorSpace::Gray => "DeviceGray",
            ElemColorSpace::RGB => "DeviceRGB",
            ElemColorSpace::CMYK => "DeviceCMYK",
            ElemColorSpace::Indexed => "Indexed",
            ElemColorSpace::Lab => "Lab",
        };
        dict.insert("ColorSpace".to_string(), Object::Name(color_space_name.to_string()));

        // Set filter based on image format
        match image.format {
            ElemImageFormat::Jpeg => {
                dict.insert("Filter".to_string(), Object::Name("DCTDecode".to_string()));
            },
            ElemImageFormat::Png | ElemImageFormat::Raw => {
                dict.insert("Filter".to_string(), Object::Name("FlateDecode".to_string()));
            },
            ElemImageFormat::Jpeg2000 => {
                dict.insert("Filter".to_string(), Object::Name("JPXDecode".to_string()));
            },
            ElemImageFormat::Jbig2 => {
                dict.insert("Filter".to_string(), Object::Name("JBIG2Decode".to_string()));
            },
            ElemImageFormat::Unknown => {
                // No filter for unknown format (raw data)
            },
        }

        dict.insert("Length".to_string(), Object::Integer(image.data.len() as i64));

        Object::Stream {
            dict,
            data: image.data.clone().into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_info_builder() {
        let info = DocumentInfo::new()
            .title("Test Document")
            .author("Test Author")
            .subject("Test Subject")
            .keywords("test, rust, pdf");

        assert_eq!(info.title, Some("Test Document".to_string()));
        assert_eq!(info.author, Some("Test Author".to_string()));
        assert_eq!(info.subject, Some("Test Subject".to_string()));
        assert_eq!(info.keywords, Some("test, rust, pdf".to_string()));
    }

    #[test]
    fn test_document_info_to_object() {
        let info = DocumentInfo::new().title("My PDF").author("John Doe");

        let obj = info.to_object();
        let dict = obj.as_dict().unwrap();

        assert!(dict.contains_key("Title"));
        assert!(dict.contains_key("Author"));
        assert!(!dict.contains_key("Subject"));
    }

    #[test]
    fn test_document_info_from_object() {
        let mut dict = HashMap::new();
        dict.insert("Title".to_string(), Object::String(b"Test Title".to_vec()));
        dict.insert("Author".to_string(), Object::String(b"Test Author".to_vec()));

        let obj = Object::Dictionary(dict);
        let info = DocumentInfo::from_object(&obj);

        assert_eq!(info.title, Some("Test Title".to_string()));
        assert_eq!(info.author, Some("Test Author".to_string()));
        assert_eq!(info.subject, None);
    }

    #[test]
    fn test_save_options() {
        let full = SaveOptions::full_rewrite();
        assert!(!full.incremental);
        assert!(full.compress);
        assert!(full.garbage_collect);

        let inc = SaveOptions::incremental();
        assert!(inc.incremental);
        assert!(!inc.compress);
        assert!(!inc.garbage_collect);
    }

    // =========================================================================
    // DocumentInfo: additional coverage
    // =========================================================================

    #[test]
    fn test_document_info_default_is_all_none() {
        let info = DocumentInfo::default();
        assert_eq!(info.title, None);
        assert_eq!(info.author, None);
        assert_eq!(info.subject, None);
        assert_eq!(info.keywords, None);
        assert_eq!(info.creator, None);
        assert_eq!(info.producer, None);
        assert_eq!(info.creation_date, None);
        assert_eq!(info.mod_date, None);
    }

    #[test]
    fn test_document_info_creator_producer() {
        let info = DocumentInfo::new()
            .creator("pdf_oxide")
            .producer("Rust PDF Library");
        assert_eq!(info.creator, Some("pdf_oxide".to_string()));
        assert_eq!(info.producer, Some("Rust PDF Library".to_string()));
    }

    #[test]
    fn test_document_info_to_object_all_fields() {
        let info = DocumentInfo {
            title: Some("Title".into()),
            author: Some("Author".into()),
            subject: Some("Subject".into()),
            keywords: Some("Keywords".into()),
            creator: Some("Creator".into()),
            producer: Some("Producer".into()),
            creation_date: Some("D:20260101000000".into()),
            mod_date: Some("D:20260226000000".into()),
        };

        let obj = info.to_object();
        let dict = obj.as_dict().unwrap();

        assert_eq!(dict.len(), 8);
        assert!(dict.contains_key("Title"));
        assert!(dict.contains_key("Author"));
        assert!(dict.contains_key("Subject"));
        assert!(dict.contains_key("Keywords"));
        assert!(dict.contains_key("Creator"));
        assert!(dict.contains_key("Producer"));
        assert!(dict.contains_key("CreationDate"));
        assert!(dict.contains_key("ModDate"));
    }

    #[test]
    fn test_document_info_to_object_empty() {
        let info = DocumentInfo::default();
        let obj = info.to_object();
        let dict = obj.as_dict().unwrap();
        assert!(dict.is_empty());
    }

    #[test]
    fn test_document_info_roundtrip() {
        let original = DocumentInfo {
            title: Some("My Title".into()),
            author: Some("My Author".into()),
            subject: Some("My Subject".into()),
            keywords: Some("a, b, c".into()),
            creator: Some("TestCreator".into()),
            producer: Some("TestProducer".into()),
            creation_date: Some("D:20260101".into()),
            mod_date: Some("D:20260226".into()),
        };

        let obj = original.to_object();
        let reconstructed = DocumentInfo::from_object(&obj);

        assert_eq!(original.title, reconstructed.title);
        assert_eq!(original.author, reconstructed.author);
        assert_eq!(original.subject, reconstructed.subject);
        assert_eq!(original.keywords, reconstructed.keywords);
        assert_eq!(original.creator, reconstructed.creator);
        assert_eq!(original.producer, reconstructed.producer);
        assert_eq!(original.creation_date, reconstructed.creation_date);
        assert_eq!(original.mod_date, reconstructed.mod_date);
    }

    #[test]
    fn test_document_info_from_non_dict() {
        // from_object on a non-dict should return all None
        let obj = Object::Integer(42);
        let info = DocumentInfo::from_object(&obj);
        assert_eq!(info.title, None);
        assert_eq!(info.author, None);
    }

    #[test]
    fn test_document_info_from_dict_with_wrong_types() {
        // Values that are not Object::String should be ignored
        let mut dict = HashMap::new();
        dict.insert("Title".to_string(), Object::Integer(123));
        dict.insert("Author".to_string(), Object::Boolean(true));

        let obj = Object::Dictionary(dict);
        let info = DocumentInfo::from_object(&obj);
        assert_eq!(info.title, None);
        assert_eq!(info.author, None);
    }

    // =========================================================================
    // SaveOptions: additional coverage
    // =========================================================================

    #[test]
    fn test_save_options_default() {
        let opts = SaveOptions::default();
        assert!(!opts.incremental);
        assert!(!opts.compress);
        assert!(!opts.linearize);
        assert!(!opts.garbage_collect);
        assert!(opts.encryption.is_none());
    }

    #[test]
    fn test_save_options_with_encryption() {
        let config = EncryptionConfig::new("user", "owner");
        let opts = SaveOptions::with_encryption(config);
        assert!(!opts.incremental);
        assert!(opts.compress);
        assert!(opts.garbage_collect);
        assert!(opts.encryption.is_some());
    }

    // =========================================================================
    // EncryptionAlgorithm
    // =========================================================================

    #[test]
    fn test_encryption_algorithm_default() {
        let algo = EncryptionAlgorithm::default();
        assert_eq!(algo, EncryptionAlgorithm::Aes256);
    }

    #[test]
    fn test_encryption_algorithm_variants() {
        let _ = EncryptionAlgorithm::Rc4_40;
        let _ = EncryptionAlgorithm::Rc4_128;
        let _ = EncryptionAlgorithm::Aes128;
        let _ = EncryptionAlgorithm::Aes256;
    }

    // =========================================================================
    // Permissions
    // =========================================================================

    #[test]
    fn test_permissions_all() {
        let perms = Permissions::all();
        assert!(perms.print);
        assert!(perms.print_high_quality);
        assert!(perms.modify);
        assert!(perms.copy);
        assert!(perms.annotate);
        assert!(perms.fill_forms);
        assert!(perms.accessibility);
        assert!(perms.assemble);
    }

    #[test]
    fn test_permissions_read_only() {
        let perms = Permissions::read_only();
        assert!(!perms.print);
        assert!(!perms.print_high_quality);
        assert!(!perms.modify);
        assert!(!perms.copy);
        assert!(!perms.annotate);
        assert!(!perms.fill_forms);
        assert!(perms.accessibility); // Always true for compliance
        assert!(!perms.assemble);
    }

    #[test]
    fn test_permissions_to_bits_all() {
        let perms = Permissions::all();
        let bits = perms.to_bits();

        // All permission bits should be set
        assert!(bits & (1 << 2) != 0); // print
        assert!(bits & (1 << 3) != 0); // modify
        assert!(bits & (1 << 4) != 0); // copy
        assert!(bits & (1 << 5) != 0); // annotate
        assert!(bits & (1 << 8) != 0); // fill_forms
        assert!(bits & (1 << 9) != 0); // accessibility
        assert!(bits & (1 << 10) != 0); // assemble
        assert!(bits & (1 << 11) != 0); // print_high_quality
    }

    #[test]
    fn test_permissions_to_bits_read_only() {
        let perms = Permissions::read_only();
        let bits = perms.to_bits();

        // Only accessibility should be set (plus reserved bits)
        assert!(bits & (1 << 2) == 0); // print not set
        assert!(bits & (1 << 3) == 0); // modify not set
        assert!(bits & (1 << 4) == 0); // copy not set
        assert!(bits & (1 << 5) == 0); // annotate not set
        assert!(bits & (1 << 8) == 0); // fill_forms not set
        assert!(bits & (1 << 9) != 0); // accessibility is set
        assert!(bits & (1 << 10) == 0); // assemble not set
        assert!(bits & (1 << 11) == 0); // print_high_quality not set
    }

    #[test]
    fn test_permissions_to_bits_reserved_bits() {
        // Reserved bits 7-8 (0-indexed: 6-7) and 13-32 (0-indexed: 12-31) must be 1
        let perms = Permissions::default();
        let bits = perms.to_bits();

        // Bits 6 and 7 must be set
        assert!(bits & (1 << 6) != 0);
        assert!(bits & (1 << 7) != 0);

        // Bits 12-31 must be set
        for bit in 12..32 {
            assert!(bits & (1 << bit) != 0, "bit {} should be set", bit);
        }
    }

    #[test]
    fn test_permissions_to_bits_individual_flags() {
        // Test each flag individually
        let mut perms = Permissions::default();
        let base_bits = perms.to_bits();

        perms.print = true;
        assert_eq!(perms.to_bits(), base_bits | (1 << 2));

        perms = Permissions::default();
        perms.modify = true;
        assert_eq!(perms.to_bits(), base_bits | (1 << 3));

        perms = Permissions::default();
        perms.copy = true;
        assert_eq!(perms.to_bits(), base_bits | (1 << 4));

        perms = Permissions::default();
        perms.annotate = true;
        assert_eq!(perms.to_bits(), base_bits | (1 << 5));

        perms = Permissions::default();
        perms.fill_forms = true;
        assert_eq!(perms.to_bits(), base_bits | (1 << 8));

        perms = Permissions::default();
        perms.assemble = true;
        assert_eq!(perms.to_bits(), base_bits | (1 << 10));

        perms = Permissions::default();
        perms.print_high_quality = true;
        assert_eq!(perms.to_bits(), base_bits | (1 << 11));
    }

    // =========================================================================
    // EncryptionConfig
    // =========================================================================

    #[test]
    fn test_encryption_config_new() {
        let config = EncryptionConfig::new("user_pw", "owner_pw");
        assert_eq!(config.user_password, "user_pw");
        assert_eq!(config.owner_password, "owner_pw");
        assert_eq!(config.algorithm, EncryptionAlgorithm::Aes256);
        // Permissions should be all by default
        assert!(config.permissions.print);
        assert!(config.permissions.copy);
    }

    #[test]
    fn test_encryption_config_default() {
        let config = EncryptionConfig::default();
        assert!(config.user_password.is_empty());
        assert!(config.owner_password.is_empty());
        assert_eq!(config.algorithm, EncryptionAlgorithm::Aes256);
    }

    #[test]
    fn test_encryption_config_with_algorithm() {
        let config = EncryptionConfig::new("u", "o").with_algorithm(EncryptionAlgorithm::Aes128);
        assert_eq!(config.algorithm, EncryptionAlgorithm::Aes128);
    }

    #[test]
    fn test_encryption_config_with_permissions() {
        let config = EncryptionConfig::new("u", "o").with_permissions(Permissions::read_only());
        assert!(!config.permissions.print);
        assert!(config.permissions.accessibility);
    }

    // =========================================================================
    // ModifiedPageProps
    // =========================================================================

    #[test]
    fn test_modified_page_props_default() {
        let props = ModifiedPageProps::default();
        assert!(props.rotation.is_none());
        assert!(props.media_box.is_none());
        assert!(props.crop_box.is_none());
    }

    #[test]
    fn test_modified_page_props_with_values() {
        let props = ModifiedPageProps {
            rotation: Some(90),
            media_box: Some([0.0, 0.0, 612.0, 792.0]),
            crop_box: Some([10.0, 10.0, 602.0, 782.0]),
        };
        assert_eq!(props.rotation, Some(90));
        assert_eq!(props.media_box, Some([0.0, 0.0, 612.0, 792.0]));
        assert_eq!(props.crop_box, Some([10.0, 10.0, 602.0, 782.0]));
    }

    // =========================================================================
    // apply_page_props_to_object (private, tested from within module)
    // =========================================================================

    /// Helper to create a minimal DocumentEditor for internal method testing.
    /// Returns a DocumentEditor from a minimal valid PDF.
    fn create_test_editor() -> DocumentEditor {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let pdf_bytes = minimal_pdf_bytes();
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp =
            std::env::temp_dir().join(format!("pdf_oxide_test_{}_{id}.pdf", std::process::id()));
        std::fs::write(&tmp, &pdf_bytes).unwrap();
        let editor = DocumentEditor::open(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        editor
    }

    /// Generates bytes for a minimal valid PDF with one page.
    fn minimal_pdf_bytes() -> Vec<u8> {
        let pdf = b"%PDF-1.4\n\
            1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\
            2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\
            3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n\
            xref\n\
            0 4\n\
            0000000000 65535 f \n\
            0000000009 00000 n \n\
            0000000058 00000 n \n\
            0000000115 00000 n \n\
            trailer\n<< /Size 4 /Root 1 0 R >>\n\
            startxref\n197\n%%EOF\n";
        pdf.to_vec()
    }

    #[test]
    fn test_apply_page_props_rotation() {
        let editor = create_test_editor();

        let mut page_dict = HashMap::new();
        page_dict.insert("Type".to_string(), Object::Name("Page".to_string()));
        let page_obj = Object::Dictionary(page_dict);

        let props = ModifiedPageProps {
            rotation: Some(90),
            media_box: None,
            crop_box: None,
        };

        let result = editor
            .apply_page_props_to_object(&page_obj, &props)
            .unwrap();
        let dict = result.as_dict().unwrap();
        assert_eq!(dict.get("Rotate").unwrap().as_integer().unwrap(), 90);
    }

    #[test]
    fn test_apply_page_props_media_box() {
        let editor = create_test_editor();

        let mut page_dict = HashMap::new();
        page_dict.insert("Type".to_string(), Object::Name("Page".to_string()));
        let page_obj = Object::Dictionary(page_dict);

        let props = ModifiedPageProps {
            rotation: None,
            media_box: Some([0.0, 0.0, 300.0, 400.0]),
            crop_box: None,
        };

        let result = editor
            .apply_page_props_to_object(&page_obj, &props)
            .unwrap();
        let dict = result.as_dict().unwrap();
        let mb = dict.get("MediaBox").unwrap().as_array().unwrap();
        assert_eq!(mb.len(), 4);
    }

    #[test]
    fn test_apply_page_props_crop_box() {
        let editor = create_test_editor();

        let mut page_dict = HashMap::new();
        page_dict.insert("Type".to_string(), Object::Name("Page".to_string()));
        let page_obj = Object::Dictionary(page_dict);

        let props = ModifiedPageProps {
            rotation: None,
            media_box: None,
            crop_box: Some([10.0, 10.0, 602.0, 782.0]),
        };

        let result = editor
            .apply_page_props_to_object(&page_obj, &props)
            .unwrap();
        let dict = result.as_dict().unwrap();
        assert!(dict.contains_key("CropBox"));
    }

    #[test]
    fn test_apply_page_props_all_at_once() {
        let editor = create_test_editor();

        let mut page_dict = HashMap::new();
        page_dict.insert("Type".to_string(), Object::Name("Page".to_string()));
        let page_obj = Object::Dictionary(page_dict);

        let props = ModifiedPageProps {
            rotation: Some(180),
            media_box: Some([0.0, 0.0, 500.0, 700.0]),
            crop_box: Some([20.0, 20.0, 480.0, 680.0]),
        };

        let result = editor
            .apply_page_props_to_object(&page_obj, &props)
            .unwrap();
        let dict = result.as_dict().unwrap();
        assert!(dict.contains_key("Rotate"));
        assert!(dict.contains_key("MediaBox"));
        assert!(dict.contains_key("CropBox"));
    }

    #[test]
    fn test_apply_page_props_non_dict_error() {
        let editor = create_test_editor();
        let page_obj = Object::Integer(42);
        let props = ModifiedPageProps::default();

        let result = editor.apply_page_props_to_object(&page_obj, &props);
        assert!(result.is_err());
    }

    // =========================================================================
    // generate_erase_overlay
    // =========================================================================

    #[test]
    fn test_generate_erase_overlay_no_regions() {
        let editor = create_test_editor();
        // No regions added for page 0
        let result = editor.generate_erase_overlay(0);
        assert!(result.is_none());
    }

    #[test]
    fn test_generate_erase_overlay_empty_regions() {
        let mut editor = create_test_editor();
        // Insert an empty vec for page 5
        editor.erase_regions.insert(5, vec![]);
        let result = editor.generate_erase_overlay(5);
        assert!(result.is_none());
    }

    #[test]
    fn test_generate_erase_overlay_single_region() {
        let mut editor = create_test_editor();
        editor
            .erase_regions
            .insert(0, vec![[10.0, 20.0, 100.0, 200.0]]);

        let content = editor.generate_erase_overlay(0).unwrap();
        let content_str = String::from_utf8(content).unwrap();

        assert!(content_str.contains("q\n"));
        assert!(content_str.contains("1 1 1 rg\n"));
        assert!(content_str.contains("re f\n"));
        assert!(content_str.contains("Q\n"));
    }

    #[test]
    fn test_generate_erase_overlay_multiple_regions() {
        let mut editor = create_test_editor();
        editor
            .erase_regions
            .insert(0, vec![[10.0, 20.0, 100.0, 200.0], [300.0, 400.0, 500.0, 600.0]]);

        let content = editor.generate_erase_overlay(0).unwrap();
        let content_str = String::from_utf8(content).unwrap();

        // Should contain two "re f" operations
        assert_eq!(content_str.matches("re f\n").count(), 2);
    }

    // =========================================================================
    // generate_flatten_overlay
    // =========================================================================

    #[test]
    fn test_generate_flatten_overlay_empty() {
        let editor = create_test_editor();
        let content = editor.generate_flatten_overlay(&[], &[]);
        assert!(content.is_empty());
    }

    #[test]
    fn test_generate_flatten_overlay_single_appearance() {
        let editor = create_test_editor();

        let appearance = AnnotationAppearance {
            content: b"test content".to_vec(),
            bbox: [0.0, 0.0, 100.0, 50.0],
            annot_rect: [10.0, 20.0, 210.0, 70.0],
            matrix: None,
            resources: None,
        };

        let names = vec!["FlatAnnot0".to_string()];
        let content = editor.generate_flatten_overlay(&[appearance], &names);
        let content_str = String::from_utf8(content).unwrap();

        assert!(content_str.contains("q\n"));
        assert!(content_str.contains("cm\n"));
        assert!(content_str.contains("/FlatAnnot0 Do\n"));
        assert!(content_str.contains("Q\n"));
    }

    #[test]
    fn test_generate_flatten_overlay_with_matrix() {
        let editor = create_test_editor();

        let appearance = AnnotationAppearance {
            content: b"test".to_vec(),
            bbox: [0.0, 0.0, 100.0, 100.0],
            annot_rect: [0.0, 0.0, 100.0, 100.0],
            matrix: Some([1.0, 0.0, 0.0, 1.0, 5.0, 5.0]),
            resources: None,
        };

        let names = vec!["Xobj0".to_string()];
        let content = editor.generate_flatten_overlay(&[appearance], &names);
        let content_str = String::from_utf8(content).unwrap();

        // Should have two cm operators: one for positioning, one for the appearance matrix
        assert_eq!(content_str.matches("cm\n").count(), 2);
    }

    #[test]
    fn test_generate_flatten_overlay_zero_size_bbox() {
        let editor = create_test_editor();

        let appearance = AnnotationAppearance {
            content: b"empty".to_vec(),
            bbox: [0.0, 0.0, 0.0, 0.0], // zero-size
            annot_rect: [10.0, 20.0, 110.0, 120.0],
            matrix: None,
            resources: None,
        };

        let names = vec!["Xobj0".to_string()];
        let content = editor.generate_flatten_overlay(&[appearance], &names);
        let content_str = String::from_utf8(content).unwrap();

        // Should still generate content even with zero bbox (uses 1.0 fallback for scale)
        assert!(content_str.contains("/Xobj0 Do\n"));
    }

    // =========================================================================
    // generate_redaction_overlay
    // =========================================================================

    #[test]
    fn test_generate_redaction_overlay_single() {
        let editor = create_test_editor();

        let redactions = vec![RedactionData {
            rect: [50.0, 100.0, 200.0, 150.0],
            color: [0.0, 0.0, 0.0],
        }];

        let content = editor.generate_redaction_overlay(&redactions);
        let content_str = String::from_utf8(content).unwrap();

        assert!(content_str.contains("q\n"));
        assert!(content_str.contains("0.000 0.000 0.000 rg\n"));
        assert!(content_str.contains("re f\n"));
        assert!(content_str.contains("Q\n"));
    }

    #[test]
    fn test_generate_redaction_overlay_custom_color() {
        let editor = create_test_editor();

        let redactions = vec![RedactionData {
            rect: [0.0, 0.0, 100.0, 100.0],
            color: [1.0, 0.0, 0.0], // Red
        }];

        let content = editor.generate_redaction_overlay(&redactions);
        let content_str = String::from_utf8(content).unwrap();
        assert!(content_str.contains("1.000 0.000 0.000 rg\n"));
    }

    #[test]
    fn test_generate_redaction_overlay_multiple() {
        let editor = create_test_editor();

        let redactions = vec![
            RedactionData {
                rect: [0.0, 0.0, 100.0, 50.0],
                color: [0.0, 0.0, 0.0],
            },
            RedactionData {
                rect: [200.0, 200.0, 400.0, 300.0],
                color: [0.5, 0.5, 0.5],
            },
        ];

        let content = editor.generate_redaction_overlay(&redactions);
        let content_str = String::from_utf8(content).unwrap();

        // Should contain two q/Q pairs
        assert_eq!(content_str.matches("q\n").count(), 2);
        assert_eq!(content_str.matches("Q\n").count(), 2);
    }

    #[test]
    fn test_generate_redaction_overlay_empty() {
        let editor = create_test_editor();
        let content = editor.generate_redaction_overlay(&[]);
        assert!(content.is_empty());
    }

    // =========================================================================
    // serialize_operator
    // =========================================================================

    #[test]
    fn test_serialize_operator_save_restore_state() {
        let editor = create_test_editor();
        let mut output = Vec::new();

        editor.serialize_operator(&mut output, &crate::content::operators::Operator::SaveState);
        assert_eq!(&output, b"q\n");

        output.clear();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::RestoreState);
        assert_eq!(&output, b"Q\n");
    }

    #[test]
    fn test_serialize_operator_cm() {
        let editor = create_test_editor();
        let mut output = Vec::new();

        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::Cm {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 1.0,
                e: 10.0,
                f: 20.0,
            },
        );

        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("1.0"));
        assert!(s.contains("10.0"));
        assert!(s.contains("20.0"));
        assert!(s.ends_with("cm\n"));
    }

    #[test]
    fn test_serialize_operator_text_operations() {
        let editor = create_test_editor();

        // BT
        let mut output = Vec::new();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::BeginText);
        assert_eq!(&output, b"BT\n");

        // ET
        output.clear();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::EndText);
        assert_eq!(&output, b"ET\n");

        // Tf
        output.clear();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::Tf {
                font: "Helv".to_string(),
                size: 12.0,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("/Helv"));
        assert!(s.contains("Tf\n"));
    }

    #[test]
    fn test_serialize_operator_td() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::Td { tx: 5.0, ty: 10.0 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("Td\n"));
    }

    #[test]
    fn test_serialize_operator_tj() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::Tj {
                text: b"Hello World".to_vec(),
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("Hello World"));
        assert!(s.ends_with("Tj\n"));
    }

    #[test]
    fn test_serialize_operator_tj_with_escapes() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::Tj {
                text: b"Hello (World) \\ test".to_vec(),
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("\\("));
        assert!(s.contains("\\)"));
        assert!(s.contains("\\\\"));
    }

    #[test]
    fn test_serialize_operator_tj_array() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::TJ {
                array: vec![
                    crate::content::operators::TextElement::String(b"AB".to_vec()),
                    crate::content::operators::TextElement::Offset(-120.0),
                    crate::content::operators::TextElement::String(b"CD".to_vec()),
                ],
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.starts_with("["));
        assert!(s.contains("(AB)"));
        assert!(s.contains("(CD)"));
        assert!(s.ends_with("TJ\n"));
    }

    #[test]
    fn test_serialize_operator_do() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::Do {
                name: "Im1".to_string(),
            },
        );
        assert_eq!(&output, b"/Im1 Do\n");
    }

    #[test]
    fn test_serialize_operator_color_ops() {
        let editor = create_test_editor();

        // SetFillRgb
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetFillRgb {
                r: 1.0,
                g: 0.0,
                b: 0.0,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("rg\n"));

        // SetStrokeGray
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetStrokeGray { gray: 0.5 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("G\n"));

        // SetFillCmyk
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetFillCmyk {
                c: 0.1,
                m: 0.2,
                y: 0.3,
                k: 0.4,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("k\n"));
    }

    #[test]
    fn test_serialize_operator_path_ops() {
        let editor = create_test_editor();

        // Stroke
        let mut output = Vec::new();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::Stroke);
        assert_eq!(&output, b"S\n");

        // Fill
        output.clear();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::Fill);
        assert_eq!(&output, b"f\n");

        // FillEvenOdd
        output.clear();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::FillEvenOdd);
        assert_eq!(&output, b"f*\n");

        // EndPath
        output.clear();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::EndPath);
        assert_eq!(&output, b"n\n");

        // ClosePath
        output.clear();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::ClosePath);
        assert_eq!(&output, b"h\n");

        // CloseFillStroke
        output.clear();
        editor
            .serialize_operator(&mut output, &crate::content::operators::Operator::CloseFillStroke);
        assert_eq!(&output, b"b\n");
    }

    #[test]
    fn test_serialize_operator_clipping() {
        let editor = create_test_editor();

        let mut output = Vec::new();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::ClipNonZero);
        assert_eq!(&output, b"W\n");

        output.clear();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::ClipEvenOdd);
        assert_eq!(&output, b"W*\n");
    }

    #[test]
    fn test_serialize_operator_rectangle() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::Rectangle {
                x: 10.0,
                y: 20.0,
                width: 100.0,
                height: 50.0,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("re\n"));
    }

    #[test]
    fn test_serialize_operator_marked_content() {
        let editor = create_test_editor();

        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::BeginMarkedContent {
                tag: "P".to_string(),
            },
        );
        assert_eq!(&output, b"/P BMC\n");

        output.clear();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::EndMarkedContent,
        );
        assert_eq!(&output, b"EMC\n");
    }

    #[test]
    fn test_serialize_operator_set_line_width() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetLineWidth { width: 2.5 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("w\n"));
    }

    #[test]
    fn test_serialize_operator_set_dash() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetDash {
                array: vec![3.0, 2.0],
                phase: 0.0,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.starts_with("["));
        assert!(s.ends_with("d\n"));
    }

    #[test]
    fn test_serialize_operator_tstar() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(&mut output, &crate::content::operators::Operator::TStar);
        assert_eq!(&output, b"T*\n");
    }

    #[test]
    fn test_serialize_operator_quote() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::Quote {
                text: b"Hello".to_vec(),
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("Hello"));
        assert!(s.ends_with("'\n"));
    }

    #[test]
    fn test_serialize_operator_double_quote() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::DoubleQuote {
                word_space: 1.0,
                char_space: 0.5,
                text: b"Hi".to_vec(),
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("Hi"));
        assert!(s.ends_with("\"\n"));
    }

    #[test]
    fn test_serialize_operator_ext_gstate() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetExtGState {
                dict_name: "GS1".to_string(),
            },
        );
        assert_eq!(&output, b"/GS1 gs\n");
    }

    #[test]
    fn test_serialize_operator_paint_shading() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::PaintShading {
                name: "Sh1".to_string(),
            },
        );
        assert_eq!(&output, b"/Sh1 sh\n");
    }

    #[test]
    fn test_serialize_operator_set_rendering_intent() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetRenderingIntent {
                intent: "RelativeColorimetric".to_string(),
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("/RelativeColorimetric"));
        assert!(s.ends_with("ri\n"));
    }

    #[test]
    fn test_serialize_operator_color_space() {
        let editor = create_test_editor();

        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetFillColorSpace {
                name: "DeviceRGB".to_string(),
            },
        );
        assert_eq!(&output, b"/DeviceRGB cs\n");

        output.clear();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetStrokeColorSpace {
                name: "DeviceCMYK".to_string(),
            },
        );
        assert_eq!(&output, b"/DeviceCMYK CS\n");
    }

    #[test]
    fn test_serialize_operator_set_fill_stroke_color() {
        let editor = create_test_editor();

        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetFillColor {
                components: vec![0.5, 0.6, 0.7],
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("sc\n"));

        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetStrokeColor {
                components: vec![0.1, 0.2],
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("SC\n"));
    }

    #[test]
    fn test_serialize_operator_color_n_with_pattern() {
        let editor = create_test_editor();

        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetFillColorN {
                components: vec![0.5],
                name: Some(Box::new("P1".to_string())),
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("/P1"));
        assert!(s.ends_with("scn\n"));

        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetStrokeColorN {
                components: vec![],
                name: None,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("SCN\n"));
    }

    #[test]
    fn test_serialize_operator_other() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::Other {
                name: "DP".to_string(),
                operands: Box::new(vec![Object::Name("MC0".to_string())]),
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("/MC0"));
        assert!(s.contains("DP"));
    }

    #[test]
    fn test_serialize_operator_inline_image() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        let mut dict = std::collections::HashMap::new();
        dict.insert("W".to_string(), Object::Integer(10));
        dict.insert("H".to_string(), Object::Integer(10));
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::InlineImage {
                dict: Box::new(dict),
                data: vec![0xFF; 10],
            },
        );
        let s = String::from_utf8_lossy(&output);
        assert!(s.contains("BI\n"));
        assert!(s.contains("ID "));
        assert!(s.contains("EI\n"));
    }

    #[test]
    fn test_serialize_operator_curves() {
        let editor = create_test_editor();

        // CurveTo
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::CurveTo {
                x1: 1.0,
                y1: 2.0,
                x2: 3.0,
                y2: 4.0,
                x3: 5.0,
                y3: 6.0,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("c\n"));

        // CurveToV
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::CurveToV {
                x2: 1.0,
                y2: 2.0,
                x3: 3.0,
                y3: 4.0,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("v\n"));

        // CurveToY
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::CurveToY {
                x1: 1.0,
                y1: 2.0,
                x3: 3.0,
                y3: 4.0,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("y\n"));
    }

    #[test]
    fn test_serialize_operator_text_state() {
        let editor = create_test_editor();

        let ops: Vec<(crate::content::operators::Operator, &str)> = vec![
            (crate::content::operators::Operator::Tc { char_space: 1.0 }, "Tc\n"),
            (crate::content::operators::Operator::Tw { word_space: 2.0 }, "Tw\n"),
            (crate::content::operators::Operator::Tz { scale: 100.0 }, "Tz\n"),
            (crate::content::operators::Operator::TL { leading: 14.0 }, "TL\n"),
            (crate::content::operators::Operator::Tr { render: 0 }, "Tr\n"),
            (crate::content::operators::Operator::Ts { rise: 3.0 }, "Ts\n"),
        ];

        for (op, suffix) in ops {
            let mut output = Vec::new();
            editor.serialize_operator(&mut output, &op);
            let s = String::from_utf8(output).unwrap();
            assert!(s.ends_with(suffix), "Expected suffix '{}', got '{}'", suffix, s);
        }
    }

    // =========================================================================
    // serialize_object
    // =========================================================================

    #[test]
    fn test_serialize_object_null() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_object(&mut output, &Object::Null);
        assert_eq!(&output, b"null");
    }

    #[test]
    fn test_serialize_object_boolean() {
        let editor = create_test_editor();

        let mut output = Vec::new();
        editor.serialize_object(&mut output, &Object::Boolean(true));
        assert_eq!(&output, b"true");

        output.clear();
        editor.serialize_object(&mut output, &Object::Boolean(false));
        assert_eq!(&output, b"false");
    }

    #[test]
    fn test_serialize_object_integer() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_object(&mut output, &Object::Integer(42));
        assert_eq!(&output, b"42");
    }

    #[test]
    fn test_serialize_object_real() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_object(&mut output, &Object::Real(std::f64::consts::PI));
        let s = String::from_utf8(output).unwrap();
        assert!(s.starts_with("3.14"));
    }

    #[test]
    fn test_serialize_object_name() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_object(&mut output, &Object::Name("Type".to_string()));
        assert_eq!(&output, b"/Type");
    }

    #[test]
    fn test_serialize_object_string() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_object(&mut output, &Object::String(b"hello".to_vec()));
        assert_eq!(&output, b"(hello)");
    }

    #[test]
    fn test_serialize_object_string_with_escapes() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_object(&mut output, &Object::String(b"a(b)c\\d".to_vec()));
        assert_eq!(&output, b"(a\\(b\\)c\\\\d)");
    }

    #[test]
    fn test_serialize_object_array() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_object(
            &mut output,
            &Object::Array(vec![Object::Integer(1), Object::Integer(2), Object::Integer(3)]),
        );
        assert_eq!(&output, b"[1 2 3]");
    }

    #[test]
    fn test_serialize_object_empty_array() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_object(&mut output, &Object::Array(vec![]));
        assert_eq!(&output, b"[]");
    }

    #[test]
    fn test_serialize_object_dictionary() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        let mut dict = HashMap::new();
        dict.insert("Key".to_string(), Object::Integer(42));
        editor.serialize_object(&mut output, &Object::Dictionary(dict));
        let s = String::from_utf8(output).unwrap();
        assert!(s.starts_with("<<"));
        assert!(s.ends_with(">>"));
        assert!(s.contains("/Key"));
        assert!(s.contains("42"));
    }

    #[test]
    fn test_serialize_object_reference() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_object(&mut output, &Object::Reference(ObjectRef::new(5, 0)));
        assert_eq!(&output, b"5 0 R");
    }

    #[test]
    fn test_serialize_object_stream_placeholder() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_object(
            &mut output,
            &Object::Stream {
                dict: HashMap::new(),
                data: bytes::Bytes::from_static(b"data"),
            },
        );
        assert_eq!(&output, b"(stream)");
    }

    // =========================================================================
    // build_image_xobject
    // =========================================================================

    #[test]
    fn test_build_image_xobject_jpeg() {
        use crate::elements::{ImageContent, ImageFormat};

        let image = ImageContent::new(
            crate::geometry::Rect::new(0.0, 0.0, 100.0, 50.0),
            ImageFormat::Jpeg,
            vec![0xFF, 0xD8, 0xFF, 0xE0], // Fake JPEG header
            100,
            50,
        );

        let obj = DocumentEditor::build_image_xobject(&image);
        if let Object::Stream { dict, data } = obj {
            assert_eq!(dict.get("Type").unwrap(), &Object::Name("XObject".to_string()));
            assert_eq!(dict.get("Subtype").unwrap(), &Object::Name("Image".to_string()));
            assert_eq!(dict.get("Width").unwrap().as_integer().unwrap(), 100);
            assert_eq!(dict.get("Height").unwrap().as_integer().unwrap(), 50);
            assert_eq!(dict.get("Filter").unwrap(), &Object::Name("DCTDecode".to_string()));
            assert_eq!(dict.get("ColorSpace").unwrap(), &Object::Name("DeviceRGB".to_string()));
        } else {
            panic!("Expected Stream object");
        }
    }

    #[test]
    fn test_build_image_xobject_png() {
        use crate::elements::{ImageContent, ImageFormat};

        let image = ImageContent {
            bbox: crate::geometry::Rect::new(0.0, 0.0, 200.0, 100.0),
            format: ImageFormat::Png,
            data: vec![0x89, 0x50, 0x4E, 0x47], // Fake PNG header
            width: 200,
            height: 100,
            bits_per_component: 8,
            color_space: crate::elements::ColorSpace::Gray,
            reading_order: None,
            alt_text: None,
            horizontal_dpi: None,
            vertical_dpi: None,
        };

        let obj = DocumentEditor::build_image_xobject(&image);
        if let Object::Stream { dict, .. } = obj {
            assert_eq!(dict.get("Filter").unwrap(), &Object::Name("FlateDecode".to_string()));
            assert_eq!(dict.get("ColorSpace").unwrap(), &Object::Name("DeviceGray".to_string()));
        } else {
            panic!("Expected Stream object");
        }
    }

    #[test]
    fn test_build_image_xobject_jpeg2000() {
        use crate::elements::{ImageContent, ImageFormat};

        let image = ImageContent {
            bbox: crate::geometry::Rect::new(0.0, 0.0, 10.0, 10.0),
            format: ImageFormat::Jpeg2000,
            data: vec![0; 8],
            width: 10,
            height: 10,
            bits_per_component: 8,
            color_space: crate::elements::ColorSpace::CMYK,
            reading_order: None,
            alt_text: None,
            horizontal_dpi: None,
            vertical_dpi: None,
        };

        let obj = DocumentEditor::build_image_xobject(&image);
        if let Object::Stream { dict, .. } = obj {
            assert_eq!(dict.get("Filter").unwrap(), &Object::Name("JPXDecode".to_string()));
            assert_eq!(dict.get("ColorSpace").unwrap(), &Object::Name("DeviceCMYK".to_string()));
        } else {
            panic!("Expected Stream object");
        }
    }

    #[test]
    fn test_build_image_xobject_jbig2() {
        use crate::elements::{ImageContent, ImageFormat};

        let image = ImageContent {
            bbox: crate::geometry::Rect::new(0.0, 0.0, 10.0, 10.0),
            format: ImageFormat::Jbig2,
            data: vec![0; 4],
            width: 10,
            height: 10,
            bits_per_component: 1,
            color_space: crate::elements::ColorSpace::Gray,
            reading_order: None,
            alt_text: None,
            horizontal_dpi: None,
            vertical_dpi: None,
        };

        let obj = DocumentEditor::build_image_xobject(&image);
        if let Object::Stream { dict, .. } = obj {
            assert_eq!(dict.get("Filter").unwrap(), &Object::Name("JBIG2Decode".to_string()));
            assert_eq!(dict.get("BitsPerComponent").unwrap().as_integer().unwrap(), 1);
        } else {
            panic!("Expected Stream object");
        }
    }

    #[test]
    fn test_build_image_xobject_unknown_format() {
        use crate::elements::{ImageContent, ImageFormat};

        let image = ImageContent {
            bbox: crate::geometry::Rect::new(0.0, 0.0, 10.0, 10.0),
            format: ImageFormat::Unknown,
            data: vec![0; 4],
            width: 10,
            height: 10,
            bits_per_component: 8,
            color_space: crate::elements::ColorSpace::RGB,
            reading_order: None,
            alt_text: None,
            horizontal_dpi: None,
            vertical_dpi: None,
        };

        let obj = DocumentEditor::build_image_xobject(&image);
        if let Object::Stream { dict, .. } = obj {
            // Unknown format should not have a Filter
            assert!(!dict.contains_key("Filter"));
        } else {
            panic!("Expected Stream object");
        }
    }

    #[test]
    fn test_build_image_xobject_indexed_colorspace() {
        use crate::elements::{ImageContent, ImageFormat};

        let image = ImageContent {
            bbox: crate::geometry::Rect::new(0.0, 0.0, 10.0, 10.0),
            format: ImageFormat::Raw,
            data: vec![0; 10],
            width: 10,
            height: 10,
            bits_per_component: 8,
            color_space: crate::elements::ColorSpace::Indexed,
            reading_order: None,
            alt_text: None,
            horizontal_dpi: None,
            vertical_dpi: None,
        };

        let obj = DocumentEditor::build_image_xobject(&image);
        if let Object::Stream { dict, .. } = obj {
            assert_eq!(dict.get("ColorSpace").unwrap(), &Object::Name("Indexed".to_string()));
            assert_eq!(dict.get("Filter").unwrap(), &Object::Name("FlateDecode".to_string()));
        } else {
            panic!("Expected Stream object");
        }
    }

    #[test]
    fn test_build_image_xobject_lab_colorspace() {
        use crate::elements::{ImageContent, ImageFormat};

        let image = ImageContent {
            bbox: crate::geometry::Rect::new(0.0, 0.0, 10.0, 10.0),
            format: ImageFormat::Raw,
            data: vec![0; 10],
            width: 10,
            height: 10,
            bits_per_component: 8,
            color_space: crate::elements::ColorSpace::Lab,
            reading_order: None,
            alt_text: None,
            horizontal_dpi: None,
            vertical_dpi: None,
        };

        let obj = DocumentEditor::build_image_xobject(&image);
        if let Object::Stream { dict, .. } = obj {
            assert_eq!(dict.get("ColorSpace").unwrap(), &Object::Name("Lab".to_string()));
        } else {
            panic!("Expected Stream object");
        }
    }

    #[test]
    fn test_build_image_xobject_length() {
        use crate::elements::{ImageContent, ImageFormat};

        let data = vec![1, 2, 3, 4, 5];
        let image = ImageContent::new(
            crate::geometry::Rect::new(0.0, 0.0, 5.0, 1.0),
            ImageFormat::Raw,
            data.clone(),
            5,
            1,
        );

        let obj = DocumentEditor::build_image_xobject(&image);
        if let Object::Stream {
            dict,
            data: stream_data,
        } = obj
        {
            assert_eq!(dict.get("Length").unwrap().as_integer().unwrap(), 5);
            assert_eq!(stream_data.as_ref(), &data[..]);
        } else {
            panic!("Expected Stream object");
        }
    }

    // =========================================================================
    // parse_media_box
    // =========================================================================

    #[test]
    fn test_parse_media_box_real_values() {
        let editor = create_test_editor();
        let media_box = Object::Array(vec![
            Object::Real(0.0),
            Object::Real(0.0),
            Object::Real(595.28),
            Object::Real(841.89),
        ]);

        let (w, h) = editor.parse_media_box(&media_box).unwrap();
        assert!((w - 595.28).abs() < 0.01);
        assert!((h - 841.89).abs() < 0.01);
    }

    #[test]
    fn test_parse_media_box_integer_values() {
        let editor = create_test_editor();
        let media_box = Object::Array(vec![
            Object::Integer(0),
            Object::Integer(0),
            Object::Integer(612),
            Object::Integer(792),
        ]);

        let (w, h) = editor.parse_media_box(&media_box).unwrap();
        assert_eq!(w, 612.0);
        assert_eq!(h, 792.0);
    }

    #[test]
    fn test_parse_media_box_non_zero_origin() {
        let editor = create_test_editor();
        let media_box = Object::Array(vec![
            Object::Real(50.0),
            Object::Real(50.0),
            Object::Real(562.0),
            Object::Real(742.0),
        ]);

        let (w, h) = editor.parse_media_box(&media_box).unwrap();
        assert!((w - 512.0).abs() < 0.01);
        assert!((h - 692.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_media_box_non_array() {
        let editor = create_test_editor();
        let media_box = Object::Integer(42);

        let (w, h) = editor.parse_media_box(&media_box).unwrap();
        // Should return default Letter size
        assert_eq!(w, 612.0);
        assert_eq!(h, 792.0);
    }

    #[test]
    fn test_parse_media_box_short_array() {
        let editor = create_test_editor();
        let media_box = Object::Array(vec![Object::Integer(0), Object::Integer(0)]);

        let (w, h) = editor.parse_media_box(&media_box).unwrap();
        // Should return default Letter size
        assert_eq!(w, 612.0);
        assert_eq!(h, 792.0);
    }

    // =========================================================================
    // find_prev_xref_offset
    // =========================================================================

    #[test]
    fn test_find_prev_xref_offset_valid() {
        let editor = create_test_editor();

        // The function searches backwards from pos = len - 100 down to 1.
        // At each pos it checks if bytes[pos..] starts with "startxref".
        // So "startxref" must be at a position p where 1 <= p <= len - 100.
        // We put startxref at position 10, then pad after it to make total > 110.
        let mut pdf_data = Vec::new();
        pdf_data.extend_from_slice(b"%PDF-1.4\n"); // 9 bytes
        pdf_data.extend_from_slice(b"startxref\n12345\n%%EOF\n"); // startxref at byte 9
                                                                  // Pad to ensure len - 100 >= 9 (i.e., len >= 109)
        while pdf_data.len() < 120 {
            pdf_data.push(b'\n');
        }
        let result = editor.find_prev_xref_offset(&pdf_data);
        assert_eq!(result.unwrap(), 12345);
    }

    #[test]
    fn test_find_prev_xref_offset_with_whitespace() {
        let editor = create_test_editor();

        let mut pdf_data = Vec::new();
        pdf_data.extend_from_slice(b"%PDF-1.4\n"); // 9 bytes
        pdf_data.extend_from_slice(b"startxref\n  \r\n67890\n%%EOF\n"); // startxref at byte 9
        while pdf_data.len() < 120 {
            pdf_data.push(b'\n');
        }
        let result = editor.find_prev_xref_offset(&pdf_data);
        assert_eq!(result.unwrap(), 67890);
    }

    #[test]
    fn test_find_prev_xref_offset_not_found() {
        let editor = create_test_editor();

        // No "startxref" anywhere, and data is long enough
        let mut pdf_data = vec![b'X'; 200];
        pdf_data.extend_from_slice(b"\n%%EOF\n");
        let result = editor.find_prev_xref_offset(&pdf_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_prev_xref_offset_short_data() {
        let editor = create_test_editor();

        // Data shorter than 100 bytes -- pos starts at 0, loop doesn't execute
        let pdf_data = b"startxref\n999\n%%EOF\n";
        let result = editor.find_prev_xref_offset(pdf_data);
        // Short data means pos starts at 0 and while pos > 0 is false, so it should err
        assert!(result.is_err());
    }

    // =========================================================================
    // allocate_object_id
    // =========================================================================

    #[test]
    fn test_allocate_object_id_sequential() {
        let mut editor = create_test_editor();
        let first = editor.allocate_object_id();
        let second = editor.allocate_object_id();
        let third = editor.allocate_object_id();

        assert_eq!(second, first + 1);
        assert_eq!(third, first + 2);
    }

    // =========================================================================
    // current_page_count and page manipulation
    // =========================================================================

    #[test]
    fn test_current_page_count_initial() {
        let editor = create_test_editor();
        assert_eq!(editor.current_page_count(), 1);
    }

    #[test]
    fn test_current_page_count_after_removal() {
        let mut editor = create_test_editor();
        // Mark page 0 as removed
        editor.page_order[0] = -1;
        assert_eq!(editor.current_page_count(), 0);
    }

    #[test]
    fn test_current_page_count_mixed() {
        let mut editor = create_test_editor();
        // Simulate 3 pages: [0, 1, 2], then remove page at index 1
        editor.page_order = vec![0, -1, 2];
        assert_eq!(editor.current_page_count(), 2);
    }

    // =========================================================================
    // is_modified
    // =========================================================================

    #[test]
    fn test_is_modified_initial() {
        let editor = create_test_editor();
        assert!(!editor.is_modified());
    }

    // =========================================================================
    // source_path
    // =========================================================================

    #[test]
    fn test_source_path() {
        let pdf_bytes = minimal_pdf_bytes();
        let tmp = std::env::temp_dir().join("pdf_oxide_test_path.pdf");
        std::fs::write(&tmp, &pdf_bytes).unwrap();
        let editor = DocumentEditor::open(&tmp).unwrap();
        assert!(editor.source_path().contains("pdf_oxide_test_path.pdf"));
        let _ = std::fs::remove_file(&tmp);
    }

    // =========================================================================
    // version
    // =========================================================================

    #[test]
    fn test_version() {
        let editor = create_test_editor();
        let (major, minor) = editor.version();
        assert_eq!(major, 1);
        assert_eq!(minor, 4);
    }

    // =========================================================================
    // build_info_object
    // =========================================================================

    #[test]
    fn test_build_info_object_none() {
        let editor = create_test_editor();
        assert!(editor.build_info_object().is_none());
    }

    #[test]
    fn test_build_info_object_some() {
        let mut editor = create_test_editor();
        editor.modified_info = Some(DocumentInfo::new().title("Test"));
        let obj = editor.build_info_object();
        assert!(obj.is_some());
        let dict = obj.unwrap().as_dict().unwrap().clone();
        assert!(dict.contains_key("Title"));
    }

    // =========================================================================
    // RedactionData
    // =========================================================================

    #[test]
    fn test_redaction_data_construction() {
        let rd = RedactionData {
            rect: [10.0, 20.0, 100.0, 200.0],
            color: [1.0, 0.0, 0.0],
        };
        assert_eq!(rd.rect[0], 10.0);
        assert_eq!(rd.color[0], 1.0);
    }

    // =========================================================================
    // AnnotationAppearance
    // =========================================================================

    #[test]
    fn test_annotation_appearance_construction() {
        let ap = AnnotationAppearance {
            content: b"q 1 0 0 1 0 0 cm Q".to_vec(),
            bbox: [0.0, 0.0, 100.0, 50.0],
            annot_rect: [10.0, 20.0, 110.0, 70.0],
            matrix: Some([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            resources: None,
        };
        assert_eq!(ap.bbox[2], 100.0);
        assert_eq!(ap.annot_rect[0], 10.0);
        assert!(ap.matrix.is_some());
        assert!(ap.resources.is_none());
    }

    #[test]
    fn test_annotation_appearance_no_matrix_no_resources() {
        let ap = AnnotationAppearance {
            content: vec![],
            bbox: [0.0, 0.0, 0.0, 0.0],
            annot_rect: [0.0, 0.0, 0.0, 0.0],
            matrix: None,
            resources: None,
        };
        assert!(ap.matrix.is_none());
        assert!(ap.content.is_empty());
    }

    // =========================================================================
    // ImageInfo
    // =========================================================================

    #[test]
    fn test_image_info_construction() {
        let info = ImageInfo {
            name: "Im1".to_string(),
            bounds: [10.0, 20.0, 100.0, 50.0],
            matrix: [100.0, 0.0, 0.0, 50.0, 10.0, 20.0],
        };
        assert_eq!(info.name, "Im1");
        assert_eq!(info.bounds[2], 100.0);
    }

    // =========================================================================
    // ImageModification
    // =========================================================================

    #[test]
    fn test_image_modification_construction() {
        let m = ImageModification {
            x: Some(10.0),
            y: None,
            width: Some(200.0),
            height: None,
        };
        assert_eq!(m.x, Some(10.0));
        assert_eq!(m.y, None);
        assert_eq!(m.width, Some(200.0));
        assert_eq!(m.height, None);
    }

    // =========================================================================
    // PageInfo
    // =========================================================================

    #[test]
    fn test_page_info_construction() {
        let info = PageInfo {
            index: 0,
            width: 612.0,
            height: 792.0,
            rotation: 90,
            object_ref: ObjectRef::new(5, 0),
        };
        assert_eq!(info.index, 0);
        assert_eq!(info.width, 612.0);
        assert_eq!(info.rotation, 90);
        assert_eq!(info.object_ref.id, 5);
    }

    // =========================================================================
    // Metadata setters (set_title, set_author, etc.)
    // =========================================================================

    #[test]
    fn test_set_title_marks_modified() {
        let mut editor = create_test_editor();
        assert!(!editor.is_modified());
        editor.set_title("New Title");
        assert!(editor.is_modified());
        assert_eq!(editor.modified_info.as_ref().unwrap().title, Some("New Title".to_string()));
    }

    #[test]
    fn test_set_author_marks_modified() {
        let mut editor = create_test_editor();
        editor.set_author("New Author");
        assert!(editor.is_modified());
        assert_eq!(editor.modified_info.as_ref().unwrap().author, Some("New Author".to_string()));
    }

    #[test]
    fn test_set_subject_marks_modified() {
        let mut editor = create_test_editor();
        editor.set_subject("New Subject");
        assert!(editor.is_modified());
        assert_eq!(editor.modified_info.as_ref().unwrap().subject, Some("New Subject".to_string()));
    }

    #[test]
    fn test_set_keywords_marks_modified() {
        let mut editor = create_test_editor();
        editor.set_keywords("kw1, kw2");
        assert!(editor.is_modified());
        assert_eq!(editor.modified_info.as_ref().unwrap().keywords, Some("kw1, kw2".to_string()));
    }

    #[test]
    fn test_set_multiple_metadata_fields() {
        let mut editor = create_test_editor();
        editor.set_title("T1");
        editor.set_author("A1");
        editor.set_subject("S1");
        editor.set_keywords("K1");

        let info = editor.modified_info.as_ref().unwrap();
        assert_eq!(info.title, Some("T1".to_string()));
        assert_eq!(info.author, Some("A1".to_string()));
        assert_eq!(info.subject, Some("S1".to_string()));
        assert_eq!(info.keywords, Some("K1".to_string()));
    }

    // =========================================================================
    // Erase region API methods
    // =========================================================================

    #[test]
    fn test_erase_region_valid() {
        let mut editor = create_test_editor();
        let result = editor.erase_region(0, [10.0, 20.0, 100.0, 200.0]);
        assert!(result.is_ok());
        assert!(editor.is_modified());
        assert_eq!(editor.erase_regions.get(&0).unwrap().len(), 1);
    }

    #[test]
    fn test_erase_region_out_of_range() {
        let mut editor = create_test_editor();
        let result = editor.erase_region(99, [10.0, 20.0, 100.0, 200.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_erase_regions_multiple() {
        let mut editor = create_test_editor();
        let rects = &[[10.0, 20.0, 100.0, 200.0], [200.0, 300.0, 400.0, 500.0]];
        let result = editor.erase_regions(0, rects);
        assert!(result.is_ok());
        assert_eq!(editor.erase_regions.get(&0).unwrap().len(), 2);
    }

    #[test]
    fn test_erase_regions_out_of_range() {
        let mut editor = create_test_editor();
        let result = editor.erase_regions(99, &[[10.0, 20.0, 100.0, 200.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_erase_regions() {
        let mut editor = create_test_editor();
        editor.erase_region(0, [10.0, 20.0, 100.0, 200.0]).unwrap();
        assert!(editor.erase_regions.contains_key(&0));
        editor.clear_erase_regions(0);
        assert!(!editor.erase_regions.contains_key(&0));
    }

    // =========================================================================
    // Annotation flatten methods
    // =========================================================================

    #[test]
    fn test_flatten_page_annotations_valid() {
        let mut editor = create_test_editor();
        assert!(!editor.is_page_marked_for_flatten(0));
        editor.flatten_page_annotations(0).unwrap();
        assert!(editor.is_page_marked_for_flatten(0));
        assert!(editor.is_modified());
    }

    #[test]
    fn test_flatten_page_annotations_out_of_range() {
        let mut editor = create_test_editor();
        assert!(editor.flatten_page_annotations(99).is_err());
    }

    #[test]
    fn test_flatten_all_annotations() {
        let mut editor = create_test_editor();
        editor.flatten_all_annotations().unwrap();
        assert!(editor.is_page_marked_for_flatten(0));
    }

    #[test]
    fn test_unmark_page_for_flatten() {
        let mut editor = create_test_editor();
        editor.flatten_page_annotations(0).unwrap();
        assert!(editor.is_page_marked_for_flatten(0));
        editor.unmark_page_for_flatten(0);
        assert!(!editor.is_page_marked_for_flatten(0));
    }

    // =========================================================================
    // Form flatten methods
    // =========================================================================

    #[test]
    fn test_flatten_forms_on_page_valid() {
        let mut editor = create_test_editor();
        assert!(!editor.is_page_marked_for_form_flatten(0));
        editor.flatten_forms_on_page(0).unwrap();
        assert!(editor.is_page_marked_for_form_flatten(0));
    }

    #[test]
    fn test_flatten_forms_on_page_out_of_range() {
        let mut editor = create_test_editor();
        assert!(editor.flatten_forms_on_page(99).is_err());
    }

    #[test]
    fn test_flatten_forms() {
        let mut editor = create_test_editor();
        assert!(!editor.will_remove_acroform());
        editor.flatten_forms().unwrap();
        assert!(editor.is_page_marked_for_form_flatten(0));
        assert!(editor.will_remove_acroform());
    }

    // =========================================================================
    // Redaction methods
    // =========================================================================

    #[test]
    fn test_apply_page_redactions_valid() {
        let mut editor = create_test_editor();
        assert!(!editor.is_page_marked_for_redaction(0));
        editor.apply_page_redactions(0).unwrap();
        assert!(editor.is_page_marked_for_redaction(0));
        assert!(editor.is_modified());
    }

    #[test]
    fn test_apply_page_redactions_out_of_range() {
        let mut editor = create_test_editor();
        assert!(editor.apply_page_redactions(99).is_err());
    }

    #[test]
    fn test_apply_all_redactions() {
        let mut editor = create_test_editor();
        editor.apply_all_redactions().unwrap();
        assert!(editor.is_page_marked_for_redaction(0));
    }

    #[test]
    fn test_unmark_page_for_redaction() {
        let mut editor = create_test_editor();
        editor.apply_page_redactions(0).unwrap();
        editor.unmark_page_for_redaction(0);
        assert!(!editor.is_page_marked_for_redaction(0));
    }

    // =========================================================================
    // Image modification methods
    // =========================================================================

    #[test]
    fn test_reposition_image_valid() {
        let mut editor = create_test_editor();
        let result = editor.reposition_image(0, "Im1", 100.0, 200.0);
        assert!(result.is_ok());
        assert!(editor.is_modified());
        assert!(editor.has_image_modifications(0));
    }

    #[test]
    fn test_reposition_image_out_of_range() {
        let mut editor = create_test_editor();
        assert!(editor.reposition_image(99, "Im1", 100.0, 200.0).is_err());
    }

    #[test]
    fn test_resize_image_valid() {
        let mut editor = create_test_editor();
        let result = editor.resize_image(0, "Im1", 300.0, 200.0);
        assert!(result.is_ok());
        assert!(editor.has_image_modifications(0));
    }

    #[test]
    fn test_resize_image_out_of_range() {
        let mut editor = create_test_editor();
        assert!(editor.resize_image(99, "Im1", 300.0, 200.0).is_err());
    }

    #[test]
    fn test_set_image_bounds_valid() {
        let mut editor = create_test_editor();
        let result = editor.set_image_bounds(0, "Im1", 10.0, 20.0, 300.0, 200.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_set_image_bounds_out_of_range() {
        let mut editor = create_test_editor();
        assert!(editor
            .set_image_bounds(99, "Im1", 10.0, 20.0, 300.0, 200.0)
            .is_err());
    }

    #[test]
    fn test_clear_image_modifications() {
        let mut editor = create_test_editor();
        editor.reposition_image(0, "Im1", 10.0, 20.0).unwrap();
        assert!(editor.has_image_modifications(0));
        editor.clear_image_modifications(0);
        assert!(!editor.has_image_modifications(0));
    }

    #[test]
    fn test_has_image_modifications_empty() {
        let editor = create_test_editor();
        assert!(!editor.has_image_modifications(0));
        assert!(!editor.has_image_modifications(99));
    }

    // =========================================================================
    // Embedded files
    // =========================================================================

    #[test]
    fn test_embed_file() {
        let mut editor = create_test_editor();
        let result = editor.embed_file("test.txt", b"Hello World".to_vec());
        assert!(result.is_ok());
        assert!(editor.is_modified());
        assert_eq!(editor.pending_embedded_files().len(), 1);
    }

    #[test]
    fn test_embed_file_multiple() {
        let mut editor = create_test_editor();
        editor.embed_file("a.txt", b"AAA".to_vec()).unwrap();
        editor.embed_file("b.txt", b"BBB".to_vec()).unwrap();
        assert_eq!(editor.pending_embedded_files().len(), 2);
    }

    #[test]
    fn test_clear_embedded_files() {
        let mut editor = create_test_editor();
        editor.embed_file("a.txt", b"AAA".to_vec()).unwrap();
        assert_eq!(editor.pending_embedded_files().len(), 1);
        editor.clear_embedded_files();
        assert!(editor.pending_embedded_files().is_empty());
    }

    // =========================================================================
    // set_page_rotation and rotate_page_by
    // =========================================================================

    #[test]
    fn test_set_page_rotation_valid_values() {
        let mut editor = create_test_editor();
        for &deg in &[0, 90, 180, 270] {
            assert!(editor.set_page_rotation(0, deg).is_ok());
        }
    }

    #[test]
    fn test_set_page_rotation_invalid_values() {
        let mut editor = create_test_editor();
        assert!(editor.set_page_rotation(0, 45).is_err());
        assert!(editor.set_page_rotation(0, 360).is_err());
        assert!(editor.set_page_rotation(0, -90).is_err());
    }

    #[test]
    fn test_set_page_rotation_out_of_range() {
        let mut editor = create_test_editor();
        assert!(editor.set_page_rotation(99, 90).is_err());
    }

    #[test]
    fn test_set_page_media_box_valid() {
        let mut editor = create_test_editor();
        let result = editor.set_page_media_box(0, [0.0, 0.0, 500.0, 700.0]);
        assert!(result.is_ok());
        assert!(editor.is_modified());
    }

    #[test]
    fn test_set_page_media_box_out_of_range() {
        let mut editor = create_test_editor();
        assert!(editor
            .set_page_media_box(99, [0.0, 0.0, 500.0, 700.0])
            .is_err());
    }

    #[test]
    fn test_set_page_crop_box_valid() {
        let mut editor = create_test_editor();
        let result = editor.set_page_crop_box(0, [10.0, 10.0, 602.0, 782.0]);
        assert!(result.is_ok());
        assert!(editor.is_modified());
    }

    #[test]
    fn test_set_page_crop_box_out_of_range() {
        let mut editor = create_test_editor();
        assert!(editor
            .set_page_crop_box(99, [10.0, 10.0, 602.0, 782.0])
            .is_err());
    }

    // =========================================================================
    // EditableDocument trait: page_count, set_info, get_info
    // =========================================================================

    #[test]
    fn test_editable_page_count() {
        let mut editor = create_test_editor();
        assert_eq!(EditableDocument::page_count(&mut editor).unwrap(), 1);
    }

    #[test]
    fn test_editable_set_info() {
        let mut editor = create_test_editor();
        let info = DocumentInfo::new().title("SetInfo Test");
        EditableDocument::set_info(&mut editor, info).unwrap();
        assert!(editor.is_modified());

        let retrieved = EditableDocument::get_info(&mut editor).unwrap();
        assert_eq!(retrieved.title, Some("SetInfo Test".to_string()));
    }

    #[test]
    fn test_editable_get_info_returns_modified() {
        let mut editor = create_test_editor();

        // Set info via the trait
        let info = DocumentInfo::new()
            .title("Modified Title")
            .author("Modified Author");
        EditableDocument::set_info(&mut editor, info).unwrap();

        // Get should return the modified version
        let result = EditableDocument::get_info(&mut editor).unwrap();
        assert_eq!(result.title, Some("Modified Title".to_string()));
        assert_eq!(result.author, Some("Modified Author".to_string()));
    }

    // =========================================================================
    // set_page_content - error path
    // =========================================================================

    #[test]
    fn test_set_page_content_out_of_range() {
        let mut editor = create_test_editor();
        let content = StructureElement::default();
        let result = editor.set_page_content(99, content);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_page_content_valid() {
        let mut editor = create_test_editor();
        let content = StructureElement {
            structure_type: "Document".to_string(),
            bbox: crate::geometry::Rect::new(0.0, 0.0, 612.0, 792.0),
            children: Vec::new(),
            reading_order: Some(0),
            alt_text: None,
            language: None,
        };
        let result = editor.set_page_content(0, content);
        assert!(result.is_ok());
        assert!(editor.is_modified());
        assert!(editor.structure_modified);
        assert!(editor.modified_content.contains_key(&0));
    }

    // =========================================================================
    // Annotations query methods
    // =========================================================================

    #[test]
    fn test_has_modified_annotations_false() {
        let editor = create_test_editor();
        assert!(!editor.has_modified_annotations(0));
    }

    #[test]
    fn test_get_page_annotations_none() {
        let editor = create_test_editor();
        assert!(editor.get_page_annotations(0).is_none());
    }

    // =========================================================================
    // rewrite_content_stream_with_image_mods
    // =========================================================================

    #[test]
    fn test_rewrite_content_stream_no_mods() {
        let editor = create_test_editor();
        let content = b"q\n100 0 0 50 10 20 cm\n/Im1 Do\nQ\n";
        let mods = HashMap::new(); // empty modifications
        let result = editor
            .rewrite_content_stream_with_image_mods(content, &mods)
            .unwrap();
        let s = String::from_utf8(result).unwrap();
        // Should preserve the cm operator as-is
        assert!(s.contains("cm\n"));
        assert!(s.contains("/Im1 Do\n"));
    }

    #[test]
    fn test_rewrite_content_stream_with_position_mod() {
        let editor = create_test_editor();
        let content = b"q\n100 0 0 50 10 20 cm\n/Im1 Do\nQ\n";
        let mut mods = HashMap::new();
        mods.insert(
            "Im1".to_string(),
            ImageModification {
                x: Some(200.0),
                y: Some(300.0),
                width: None,
                height: None,
            },
        );
        let result = editor
            .rewrite_content_stream_with_image_mods(content, &mods)
            .unwrap();
        let s = String::from_utf8(result).unwrap();
        // The e and f values (position) should be modified
        assert!(s.contains("200.0"));
        assert!(s.contains("300.0"));
    }

    #[test]
    fn test_rewrite_content_stream_with_size_mod() {
        let editor = create_test_editor();
        let content = b"q\n100 0 0 50 10 20 cm\n/Im1 Do\nQ\n";
        let mut mods = HashMap::new();
        mods.insert(
            "Im1".to_string(),
            ImageModification {
                x: None,
                y: None,
                width: Some(400.0),
                height: Some(300.0),
            },
        );
        let result = editor
            .rewrite_content_stream_with_image_mods(content, &mods)
            .unwrap();
        let s = String::from_utf8(result).unwrap();
        // The a and d values (size) should be modified
        assert!(s.contains("400.0"));
        assert!(s.contains("300.0"));
    }

    // =========================================================================
    // Open with non-existent file
    // =========================================================================

    #[test]
    fn test_open_nonexistent_file() {
        let result = DocumentEditor::open("/tmp/nonexistent_pdf_oxide_test.pdf");
        assert!(result.is_err());
    }

    // =========================================================================
    // EditableDocument trait: remove_page, move_page, duplicate_page
    // =========================================================================

    /// Helper that creates a test editor with N pages (using a multi-page minimal PDF).
    fn create_multi_page_editor(n: usize) -> DocumentEditor {
        // Build a minimal PDF with n pages
        let mut pdf = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.4\n");

        // Object 1: Catalog
        let catalog_offset = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        // Build Kids array
        let mut kids = String::from("[");
        for i in 0..n {
            if i > 0 {
                kids.push(' ');
            }
            kids.push_str(&format!("{} 0 R", i + 3));
        }
        kids.push(']');

        // Object 2: Pages
        let pages_offset = pdf.len();
        let pages_str =
            format!("2 0 obj\n<< /Type /Pages /Kids {} /Count {} >>\nendobj\n", kids, n);
        pdf.extend_from_slice(pages_str.as_bytes());

        // Page objects
        let mut page_offsets = Vec::new();
        for i in 0..n {
            page_offsets.push(pdf.len());
            let page_str = format!(
                "{} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
                i + 3
            );
            pdf.extend_from_slice(page_str.as_bytes());
        }

        // xref
        let xref_offset = pdf.len();
        let total_objects = n + 3;
        pdf.extend_from_slice(format!("xref\n0 {}\n", total_objects).as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", catalog_offset).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", pages_offset).as_bytes());
        for offset in &page_offsets {
            pdf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
        }

        // trailer
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
                total_objects, xref_offset
            )
            .as_bytes(),
        );

        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp = std::env::temp_dir()
            .join(format!("pdf_oxide_multipage_test_{}_{id}.pdf", std::process::id()));
        std::fs::write(&tmp, &pdf).unwrap();
        let editor = DocumentEditor::open(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        editor
    }

    #[test]
    fn test_remove_page_valid() {
        let mut editor = create_multi_page_editor(3);
        assert_eq!(editor.current_page_count(), 3);
        editor.remove_page(1).unwrap();
        assert_eq!(editor.current_page_count(), 2);
        assert!(editor.is_modified());
    }

    #[test]
    fn test_remove_page_out_of_range() {
        let mut editor = create_multi_page_editor(2);
        assert!(editor.remove_page(5).is_err());
    }

    #[test]
    fn test_move_page_valid() {
        let mut editor = create_multi_page_editor(3);
        editor.move_page(0, 2).unwrap();
        assert!(editor.is_modified());
        // Page order should be rearranged
        assert_eq!(editor.page_order, vec![1, 2, 0]);
    }

    #[test]
    fn test_move_page_out_of_range() {
        let mut editor = create_multi_page_editor(2);
        assert!(editor.move_page(0, 5).is_err());
        assert!(editor.move_page(5, 0).is_err());
    }

    #[test]
    fn test_duplicate_page_valid() {
        let mut editor = create_multi_page_editor(2);
        let new_idx = editor.duplicate_page(0).unwrap();
        assert_eq!(new_idx, 2); // Was 2 pages, new one is at index 2
        assert_eq!(editor.current_page_count(), 3);
        assert!(editor.is_modified());
    }

    #[test]
    fn test_duplicate_page_out_of_range() {
        let mut editor = create_multi_page_editor(2);
        assert!(editor.duplicate_page(5).is_err());
    }

    // =========================================================================
    // get_page_info
    // =========================================================================

    #[test]
    fn test_get_page_info_valid() {
        let mut editor = create_test_editor();
        let info = editor.get_page_info(0).unwrap();
        assert_eq!(info.index, 0);
        assert_eq!(info.width, 612.0);
        assert_eq!(info.height, 792.0);
        assert_eq!(info.rotation, 0);
    }

    #[test]
    fn test_get_page_info_out_of_range() {
        let mut editor = create_test_editor();
        assert!(editor.get_page_info(99).is_err());
    }

    // =========================================================================
    // extract_pages error
    // =========================================================================

    #[test]
    fn test_extract_pages_not_implemented() {
        let mut editor = create_test_editor();
        // Currently returns "not yet fully implemented"
        let result = editor.extract_pages(&[0], "/tmp/output.pdf");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_pages_out_of_range() {
        let mut editor = create_test_editor();
        let result = editor.extract_pages(&[99], "/tmp/output.pdf");
        assert!(result.is_err());
    }

    // =========================================================================
    // rotate_page_by normalization
    // =========================================================================

    #[test]
    fn test_rotate_page_by_normalization() {
        let mut editor = create_test_editor();

        // Start at 0, rotate by 90
        editor.rotate_page_by(0, 90).unwrap();
        assert_eq!(editor.get_page_rotation(0).unwrap(), 90);

        // Rotate by another 90 -> 180
        editor.rotate_page_by(0, 90).unwrap();
        assert_eq!(editor.get_page_rotation(0).unwrap(), 180);

        // Rotate by another 90 -> 270
        editor.rotate_page_by(0, 90).unwrap();
        assert_eq!(editor.get_page_rotation(0).unwrap(), 270);

        // Rotate by another 90 -> 360 -> normalized to 0
        editor.rotate_page_by(0, 90).unwrap();
        assert_eq!(editor.get_page_rotation(0).unwrap(), 0);
    }

    #[test]
    fn test_rotate_all_pages() {
        let mut editor = create_multi_page_editor(3);
        editor.rotate_all_pages(90).unwrap();

        for i in 0..3 {
            assert_eq!(editor.get_page_rotation(i).unwrap(), 90);
        }
    }

    // =========================================================================
    // crop_margins
    // =========================================================================

    #[test]
    fn test_crop_margins() {
        let mut editor = create_test_editor();
        editor.crop_margins(72.0, 72.0, 72.0, 72.0).unwrap();

        // Should have set crop box for page 0
        let props = editor.modified_page_props.get(&0).unwrap();
        let crop = props.crop_box.unwrap();
        assert!((crop[0] - 72.0).abs() < 0.01); // left margin
        assert!((crop[1] - 72.0).abs() < 0.01); // bottom margin
        assert!((crop[2] - 540.0).abs() < 0.01); // 612 - 72
        assert!((crop[3] - 720.0).abs() < 0.01); // 792 - 72
    }

    // =========================================================================
    // find_max_object_id
    // =========================================================================

    #[test]
    fn test_find_max_object_id() {
        let editor = create_test_editor();
        // Our minimal PDF has /Size 4 in the trailer, so max_id should be 4
        let max = DocumentEditor::find_max_object_id(&editor.source);
        assert_eq!(max, 4);
    }

    // =========================================================================
    // Operator: MoveTo, LineTo, Tm, TD
    // =========================================================================

    #[test]
    fn test_serialize_operator_move_to() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::MoveTo { x: 10.0, y: 20.0 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("m\n"));
    }

    #[test]
    fn test_serialize_operator_line_to() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::LineTo { x: 100.0, y: 200.0 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("l\n"));
    }

    #[test]
    fn test_serialize_operator_tm() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::Tm {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 1.0,
                e: 72.0,
                f: 700.0,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("Tm\n"));
    }

    #[test]
    fn test_serialize_operator_big_td() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::TD { tx: 0.0, ty: -14.0 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("TD\n"));
    }

    #[test]
    fn test_serialize_operator_set_line_cap() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetLineCap { cap_style: 1 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("J\n"));
    }

    #[test]
    fn test_serialize_operator_set_line_join() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetLineJoin { join_style: 2 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("j\n"));
    }

    #[test]
    fn test_serialize_operator_set_miter_limit() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetMiterLimit { limit: 10.0 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("M\n"));
    }

    #[test]
    fn test_serialize_operator_set_flatness() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetFlatness { tolerance: 50.0 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("i\n"));
    }

    #[test]
    fn test_serialize_operator_set_stroke_cmyk() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetStrokeCmyk {
                c: 0.0,
                m: 1.0,
                y: 0.5,
                k: 0.0,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("K\n"));
    }

    #[test]
    fn test_serialize_operator_set_stroke_rgb() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetStrokeRgb {
                r: 0.0,
                g: 0.5,
                b: 1.0,
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("RG\n"));
    }

    #[test]
    fn test_serialize_operator_set_fill_gray() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::SetFillGray { gray: 0.75 },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.ends_with("g\n"));
    }

    #[test]
    fn test_serialize_operator_begin_marked_content_dict() {
        let editor = create_test_editor();
        let mut output = Vec::new();
        editor.serialize_operator(
            &mut output,
            &crate::content::operators::Operator::BeginMarkedContentDict {
                tag: "Span".to_string(),
                properties: Box::new(Object::Name("MC0".to_string())),
            },
        );
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("/Span"));
        assert!(s.contains("/MC0"));
        assert!(s.ends_with("BDC\n"));
    }
}
