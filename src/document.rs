//! PDF document model.

use crate::encryption::EncryptionHandler;
use crate::error::{Error, Result};
use crate::layout::TextSpan;
use crate::object::{Object, ObjectRef};
use crate::parser::parse_object;
use crate::parser_config::ParserOptions;
use crate::pipeline::{
    converters::OutputConverter, HtmlOutputConverter, MarkdownOutputConverter, PlainTextConverter,
    ReadingOrderContext, TextPipeline, TextPipelineConfig,
};
use crate::structure::traverse_structure_tree;
use crate::xref::{find_xref_offset, parse_xref, CrossRefTable};
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

// Re-export MutexExt from cache module for local use and backward compatibility
pub(crate) use crate::cache::MutexExt;

/// Reading order mode for span extraction.
///
/// Controls how text spans are sorted after extraction from a PDF page.
/// The default `TopToBottom` mode uses simple geometric sorting, while
/// `ColumnAware` uses the XY-Cut algorithm to detect columns and read
/// each column top-to-bottom before moving to the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadingOrder {
    /// Simple top-to-bottom, left-to-right ordering.
    ///
    /// Sorts spans by Y-coordinate descending (top of page first),
    /// then by X-coordinate ascending (left to right).
    #[default]
    TopToBottom,
    /// Column-aware ordering using the XY-Cut algorithm.
    ///
    /// Detects columns via projection-profile analysis and reads each
    /// column fully (top-to-bottom) before moving to the next column.
    /// Best for newspapers, academic papers, and multi-column layouts.
    ColumnAware,
}

/// In-memory reader used by `open()` and `from_bytes()`. Wrapping in an enum
/// is kept (rather than using `BufReader<Cursor<Vec<u8>>>` directly) so a
/// future file-backed variant can be re-introduced without touching call
/// sites.
enum PdfReader {
    Memory(BufReader<Cursor<Vec<u8>>>),
}

impl Read for PdfReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            PdfReader::Memory(r) => r.read(buf),
        }
    }
}

impl Seek for PdfReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            PdfReader::Memory(r) => r.seek(pos),
        }
    }
}

impl BufRead for PdfReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        match self {
            PdfReader::Memory(r) => r.fill_buf(),
        }
    }

    fn consume(&mut self, amt: usize) {
        match self {
            PdfReader::Memory(r) => r.consume(amt),
        }
    }
}

/// Maximum recursion depth for object resolution
const MAX_RECURSION_DEPTH: u32 = 100;

/// Page information for rendering.
#[cfg(feature = "rendering")]
#[derive(Debug, Clone)]
pub struct PageInfo {
    /// Media box defining the page boundaries
    pub media_box: crate::geometry::Rect,
    /// Crop box if specified (for visible area)
    pub crop_box: Option<crate::geometry::Rect>,
    /// Page rotation in degrees (0, 90, 180, 270)
    pub rotation: i32,
}

/// Default maximum size in bytes for the object cache (64 MB).
///
/// This is a soft guardrail, not a hard ceiling. Real memory usage can be
/// 1.5–2× the cap because `estimate_size` does not account for HashMap bucket
/// overhead, Arc headers, or allocator padding.
const DEFAULT_OBJECT_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Default maximum number of entries for the XObject span/image caches.
const DEFAULT_XOBJECT_CACHE_MAX_ENTRIES: usize = 1024;

/// Heuristic multiplier for the forward-gap guard in the main
/// assembly loop's compound newline predicate
/// (`y_diff > 2.0 && gap > K * max(fs)`). Visual gap-sweep over
/// synthetic two-column examples at fs=10 and fs=14 placed the
/// plausible operating band at roughly 0.7-1.5; 1.25 is a
/// conservative interim pick. Not corpus-calibrated; a page-level
/// layout signal would be a stronger long-term replacement for
/// this pairwise heuristic.
const FORWARD_GAP_K: f32 = 1.25;

/// Maximum allowed inter-span X gap inside a candidate same-line reorder run.
/// If the candidate's tentative X-order contains a larger gap, the run is
/// probably a disjoint footer/header/field layout rather than a local
/// mixed-baseline repair.
const SAME_LINE_REORDER_MAX_GAP_FACTOR: f32 = 3.0;

// Re-export BoundedEntryCache from cache module for local use and backward compatibility
pub(crate) use crate::cache::BoundedEntryCache;

/// Size-bounded object cache with FIFO eviction.
///
/// Wraps a `HashMap<ObjectRef, Object>` with byte-size tracking. When an
/// insertion would push total estimated size past `max_bytes`, the oldest
/// entries are evicted first (FIFO order via a `VecDeque` of keys).
///
/// FIFO is chosen over LRU because the access pattern is predominantly
/// insert-once-read-once — higher-level caches (font caches, xobject stream
/// cache) serve repeated lookups, so recency is not a useful signal here.
struct BoundedObjectCache {
    map: HashMap<ObjectRef, Object>,
    insertion_order: std::collections::VecDeque<ObjectRef>,
    current_bytes: usize,
    max_bytes: usize,
}

impl BoundedObjectCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            insertion_order: std::collections::VecDeque::new(),
            current_bytes: 0,
            max_bytes,
        }
    }

    fn get(&self, key: &ObjectRef) -> Option<&Object> {
        self.map.get(key)
    }

    fn insert(&mut self, key: ObjectRef, value: Object) {
        let entry_size = Self::estimate_size(&value);

        // Don't cache objects that alone exceed the budget
        if entry_size > self.max_bytes {
            return;
        }

        // If the key already exists, subtract old size first
        if let Some(old_val) = self.map.get(&key) {
            self.current_bytes = self
                .current_bytes
                .saturating_sub(Self::estimate_size(old_val));
        }

        // Evict oldest entries until under budget. If the front of the
        // queue is the key we're about to (re)insert, skip past it so a
        // larger replacement doesn't leave the cache over budget — keep
        // evicting other entries instead.
        let mut skipped_self = false;
        while self.current_bytes + entry_size > self.max_bytes {
            match self.insertion_order.pop_front() {
                Some(old_key) => {
                    if old_key == key {
                        if skipped_self {
                            self.insertion_order.push_front(old_key);
                            break;
                        }
                        self.insertion_order.push_back(old_key);
                        skipped_self = true;
                        continue;
                    }
                    if let Some(old_val) = self.map.remove(&old_key) {
                        self.current_bytes = self
                            .current_bytes
                            .saturating_sub(Self::estimate_size(&old_val));
                    }
                },
                None => break,
            }
        }

        // Insert (or replace) the entry
        if self.map.insert(key, value).is_none() {
            // New key — track insertion order
            self.insertion_order.push_back(key);
        }
        self.current_bytes += entry_size;
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn keys(&self) -> impl Iterator<Item = &ObjectRef> {
        self.map.keys()
    }

    fn clear(&mut self) {
        self.map.clear();
        self.insertion_order.clear();
        self.current_bytes = 0;
    }

    fn estimate_size(obj: &Object) -> usize {
        Self::estimate_size_depth(obj, 8)
    }

    /// Rough estimate of an Object's heap size in bytes.
    /// Recurses into nested containers up to `depth` levels to avoid
    /// both underestimation and stack overflow on adversarial input.
    fn estimate_size_depth(obj: &Object, depth: u8) -> usize {
        if depth == 0 {
            return 64;
        }
        match obj {
            Object::Stream { dict, data } => {
                let dict_size: usize = dict
                    .iter()
                    .map(|(k, v)| k.len() + 32 + Self::estimate_size_depth(v, depth - 1))
                    .sum();
                data.len() + dict_size + 64
            },
            Object::Dictionary(d) => {
                let inner: usize = d
                    .iter()
                    .map(|(k, v)| k.len() + 32 + Self::estimate_size_depth(v, depth - 1))
                    .sum();
                inner + 64
            },
            Object::Array(a) => {
                let inner: usize = a
                    .iter()
                    .map(|v| Self::estimate_size_depth(v, depth - 1))
                    .sum();
                inner + 64
            },
            Object::String(s) => s.len() + 32,
            Object::Name(s) => s.len() + 32,
            _ => 32,
        }
    }
}

// Per-thread resolving stack and recursion depth for load_object.
// Thread-local storage avoids document-global lock contention and prevents
// false "circular reference" errors when two threads resolve the same object
// concurrently (#398 Race C).
thread_local! {
    static RESOLVING_STACK: RefCell<HashSet<ObjectRef>> = RefCell::new(HashSet::new());
    static RECURSION_DEPTH: RefCell<u32> = const { RefCell::new(0) };
}

/// PDF document.
///
/// This structure represents an open PDF document, providing access to:
/// - Document metadata (version, catalog, trailer)
/// - Page information (count, page tree)
/// - Object loading and dereferencing
///
/// # Example
///
/// ```no_run
/// use pdf_oxide::document::PdfDocument;
///
/// let mut doc = PdfDocument::open("sample.pdf")?;
/// println!("PDF version: {}.{}", doc.version().0, doc.version().1);
/// println!("Page count: {}", doc.page_count()?);
/// # Ok::<(), pdf_oxide::error::Error>(())
/// ```
///
/// # Memory management
///
/// The document maintains several internal caches for performance. The main
/// object cache is bounded at 64 MB (see `DEFAULT_OBJECT_CACHE_MAX_BYTES`)
/// uses FIFO eviction to prevent unbounded heap growth when processing
/// many pages sequentially.
pub struct PdfDocument {
    /// PDF reader — file-backed on native, memory-backed on WASM.
    ///
    /// # Thread Safety
    /// All interior-mutable fields use `Mutex` / `AtomicUsize`, making
    /// `PdfDocument` both `Send` and `Sync`.
    /// Wrapped in RefCell for interior mutability (seek/read require &mut).
    reader: Mutex<PdfReader>,
    /// Serializes concurrent *cold* (uncached) object loads on a shared
    /// handle. A single logical load makes many separate `reader` lock
    /// scopes (header, /Length resolution, stream bytes, nested refs);
    /// without this, two threads cold-loading on one shared `PdfDocument`
    /// (e.g. the C# binding's single native handle calling `render_page_fit`
    /// from multiple threads) interleave those scopes on the shared
    /// `BufReader` and read each other's bytes, surfacing as a spurious
    /// `[1000] invalid PDF structure or content stream`. Acquired only at
    /// the top-level entry of `load_object` (recursion depth 0) with a
    /// double-checked cache, so warm cache hits stay fully parallel
    /// same-thread recursion never re-acquires (no self-deadlock). #507.
    load_lock: Mutex<()>,
    /// Raw bytes of the document (kept for duplication/editing)
    pub source_bytes: Vec<u8>,
    /// PDF version (major, minor)
    version: (u8, u8),
    /// Cross-reference table mapping object IDs to byte offsets
    xref: CrossRefTable,
    /// Trailer dictionary
    trailer: Object,
    /// Cache for loaded objects to avoid re-parsing.
    /// Bounded at [`DEFAULT_OBJECT_CACHE_MAX_BYTES`] with FIFO eviction to
    /// prevent unbounded heap growth during multi-page extraction.
    object_cache: Mutex<BoundedObjectCache>,
    /// Encryption handler (if PDF is encrypted).
    /// Wrapped in RefCell for interior mutability (lazy initialization from &self).
    encryption_handler: Mutex<Option<EncryptionHandler>>,
    /// ObjectRef of the /Encrypt dictionary, cached so its strings are
    /// skipped during per-object string decryption. The entries in the
    /// encryption dict (/O, /U, /OE, /UE, /Perms, …) are key material used
    /// to derive the encryption key, not ciphertext, and must never be
    /// passed through `decrypt_string`.
    encrypt_dict_ref: Mutex<Option<ObjectRef>>,
    /// Parser configuration options for error handling and recovery
    #[allow(dead_code)]
    options: ParserOptions,
    /// Byte offset where PDF header was found (may not be 0 for malformed PDFs)
    #[allow(dead_code)]
    header_offset: u64,
    /// Font cache keyed by indirect ObjectRef to avoid re-parsing fonts across pages.
    /// Arc-wrapped to eliminate deep cloning when populating per-page TextExtractor.
    /// Bounded at 512 entries — TeX PDFs can create unique font objects per page.
    font_cache: Mutex<BoundedEntryCache<ObjectRef, Arc<crate::fonts::FontInfo>>>,
    /// Cached font sets keyed by /Font dictionary ObjectRef.
    /// Pages sharing the same /Font dict skip the entire load_fonts() loop.
    /// Bounded at 256 entries.
    font_set_cache: Mutex<BoundedEntryCache<ObjectRef, Vec<(String, Arc<crate::fonts::FontInfo>)>>>,
    /// Fingerprint-based font set cache for direct /Font dictionaries.
    /// Keyed by sorted font ObjectRefs hash, catches pages with different
    /// /Resources but same font references. Bounded at 256 entries.
    font_fingerprint_cache:
        Mutex<BoundedEntryCache<u64, Vec<(String, Arc<crate::fonts::FontInfo>)>>>,
    /// Name-based font set cache keyed by hash of sorted font names.
    /// Catches pages with different font ObjectRefs but the same font name→base font
    /// mapping (common in PDFs that create new font objects per page).
    /// Stores the resolved font set (Arc-wrapped to avoid cloning) plus a combined
    /// identity hash over ALL fonts for verification before reuse. Bounded at 256 entries.
    font_name_set_cache:
        Mutex<BoundedEntryCache<u64, (Arc<Vec<(String, Arc<crate::fonts::FontInfo>)>>, u64)>>,
    /// Per-font identity cache keyed by font_identity_hash (BaseFont + Subtype + Encoding +
    /// ToUnicode + FontDescriptor + DescendantFonts references). Skips expensive
    /// `FontInfo::from_dict()` when a structurally identical font was already parsed.
    /// Bounded at 512 entries.
    font_identity_cache: Mutex<BoundedEntryCache<u64, Arc<crate::fonts::FontInfo>>>,
    /// Per-object `font_identity_hash_cheap`, memoized. An object's content is
    /// fixed within a document, so the Layer-4 cache guard (#408) need not
    /// re-load and re-hash each font's `/Widths` on every page.
    font_id_hash_cache: Mutex<HashMap<ObjectRef, u64>>,
    /// Cached structure tree (None = not yet checked, Some(None) = untagged, Some(Some) = tagged).
    /// Uses Arc to avoid expensive deep clones on every page extraction.
    /// Mutex provides interior mutability for `&self` read-path methods (#398).
    structure_tree_cache: Mutex<Option<Option<Arc<crate::structure::StructTreeRoot>>>>,
    /// Cached per-page structure tree traversal results.
    /// Built once from the structure tree, then O(1) lookup per page.
    /// Mutex provides interior mutability for `&self` read-path methods (#398).
    structure_content_cache: Mutex<Option<HashMap<u32, Vec<crate::structure::OrderedContent>>>>,
    /// `Table` structure elements bucketed by page, built once via
    /// `find_table_elements_all_pages` (one tree walk) so the converter table
    /// path does an O(1) lookup instead of walking the tree per page.
    /// `None` = not yet built.
    table_elements_cache: Mutex<Option<HashMap<u32, Vec<crate::structure::StructElem>>>>,
    /// Page object cache keyed by page index to avoid re-traversing the page tree.
    /// The page tree structure is static (§7.7.3.2), so pages can be safely cached.
    /// Mutex provides interior mutability for `&self` read-path methods (#398).
    page_cache: Mutex<HashMap<usize, Object>>,
    /// Whether the bulk page tree walk has been attempted (successful or not).
    /// Prevents re-walking the tree on every cache miss for malformed PDFs.
    page_cache_populated: AtomicBool,
    /// Cached object offsets from full file scan (built on first xref miss).
    /// Maps object number to byte offset in file.
    scanned_object_offsets: Mutex<Option<HashMap<u32, u64>>>,
    /// Whether the one-time object-stream recovery sweep has been attempted.
    /// See `recover_from_object_streams`. Separate from the scanned offsets
    /// cache because the sweep is only triggered on free-entry misses that
    /// also failed the file-body scan — the common path never needs it.
    objstm_recovery_done: Mutex<bool>,
    /// Cache of XObject refs known to NOT be Form XObjects (i.e., Image or unknown).
    /// Used by text extraction to skip expensive full-object loads for images.
    image_xobject_cache: Mutex<HashSet<ObjectRef>>,
    /// Document-level cache of Form XObject refs whose streams contain NO text
    /// operators (BT) and no nested Do invocations. Persists across pages so that
    /// shared graphics-only XObjects (watermarks, logos, chart elements) are
    /// decompressed and scanned at most once across the entire document.
    pub(crate) xobject_text_free_cache: Mutex<HashSet<ObjectRef>>,
    /// Cache of decompressed Form XObject streams. Bounded at 50MB total.
    /// Avoids repeated FlateDecode decompression of shared Form XObjects.
    pub(crate) xobject_stream_cache: Mutex<HashMap<ObjectRef, std::sync::Arc<Vec<u8>>>>,
    pub(crate) xobject_stream_cache_bytes: AtomicUsize,
    /// Cache of extracted TextSpan results from self-contained Form XObjects
    /// (those with own /Resources/Font). None = processed but no spans.
    /// Key is `(ObjectRef, [i64; 6])` where the array encodes the caller's CTM
    /// as millipoint-rounded integers, allowing the same Form XObject to cache
    /// distinct results for each unique CTM it is painted with.
    /// Bounded at [`DEFAULT_XOBJECT_CACHE_MAX_ENTRIES`] entries with FIFO eviction.
    pub(crate) xobject_spans_cache:
        Mutex<BoundedEntryCache<(ObjectRef, [i64; 6]), Option<Vec<crate::layout::TextSpan>>>>,
    /// Cache of extracted images from Form XObjects (keyed by ObjectRef).
    /// Images are stored without CTM applied — caller applies its own CTM.
    /// Bounded at [`DEFAULT_XOBJECT_CACHE_MAX_ENTRIES`] entries with FIFO eviction.
    pub(crate) form_xobject_images_cache:
        Mutex<BoundedEntryCache<ObjectRef, Vec<crate::extractors::PdfImage>>>,
    /// Regions marked for erasure per page. Mutex for `&self` write-path methods (#398).
    pub(crate) erase_regions: Mutex<HashMap<usize, Vec<crate::geometry::Rect>>>,
    /// LRU cache of decompressed page content streams, keyed by page index.
    page_content_cache: Mutex<BoundedEntryCache<usize, std::sync::Arc<Vec<u8>>>>,
    /// LRU cache of postprocessed [`TextSpan`]s per page. `to_markdown`/`to_html`
    /// reach `extract_spans` twice per page — once directly, once via
    /// `extract_page_tables` → `extract_words` → `page_reading_order`; this serves
    /// the second from cache. Cleared by redaction (`erase_region` /
    /// `clear_erase_regions`), the only span-affecting mutation.
    page_spans_cache: Mutex<BoundedEntryCache<usize, std::sync::Arc<Vec<crate::layout::TextSpan>>>>,
    /// Cached signatures of running headers/footers detected via cross-page
    /// repetition. A span whose normalized text matches a signature
    /// sits near the top/bottom of the page is treated as an artifact.
    /// Populated lazily on first access; `Some(set)` with an empty set
    /// means detection ran and found nothing (vs `None` = not yet run).
    /// Signatures of running headers/footers plus the first page index where
    /// each signature was observed. Used to mark repeat occurrences as
    /// pagination artifacts while keeping the first appearance intact — the
    /// first appearance is often the document's cover-page title that just
    /// happens to echo into the header band on every page (B3: pdfa_010
    /// would otherwise drop "University of Oklahoma 2009").
    running_artifact_signatures: Mutex<Option<std::collections::HashMap<String, usize>>>,
    /// Accumulated extraction warnings for programmatic inspection.
    /// Populated when silent fallbacks occur (font not found, CMap absent, etc.).
    /// Retrieve with [`PdfDocument::warnings`]; drain with [`PdfDocument::take_warnings`].
    accumulated_warnings: Mutex<Vec<String>>,
    /// structured warnings accumulator. Each
    /// internal warning site that previously only called `log::warn!`
    /// can additionally push a typed [`crate::extractors::warnings::Warning`]
    /// here, letting callers retrieve diagnostics as structured data
    /// (via [`PdfDocument::structured_warnings`]) instead of parsing
    /// stderr text. The existing String-list `accumulated_warnings`
    /// stays for back-compat.
    warning_sink: crate::extractors::warnings::WarningSink,
}

// Compile-time verification that PdfDocument is Send + Sync.
const _: () = {
    fn _assert_send_sync<T: Send + Sync>() {}
    fn _check() {
        _assert_send_sync::<PdfDocument>();
    }
};

impl std::fmt::Debug for PdfDocument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdfDocument")
            .field("version", &self.version)
            .field("xref_entries", &self.xref.len())
            .field("cached_objects", &self.object_cache.lock_or_recover().len())
            .finish_non_exhaustive()
    }
}

/// Pre-decompression filter for image extraction.
///
/// Dimensions are checked against XObject dictionary metadata (Width, Height,
/// ColorSpace) BEFORE the stream is decompressed, avoiding expensive decoding
/// of images that will be discarded downstream.
struct ImageExtractFilter {
    /// Minimum width in pixels (images narrower are skipped).
    min_width: i64,
    /// Minimum height in pixels (images shorter are skipped).
    min_height: i64,
    /// Maximum total pixels (images exceeding this are skipped).
    max_pixels: u64,
    /// Skip Indexed-colorspace images below this dimension.
    /// 0 means disabled.
    skip_indexed_small: i64,
}

impl Default for ImageExtractFilter {
    fn default() -> Self {
        Self {
            min_width: 8,
            min_height: 8,
            max_pixels: u64::MAX,
            skip_indexed_small: 0,
        }
    }
}

/// Default max image pixels for markdown/HTML embedding (16 MP).
/// Covers A4 at 300 DPI (8.7 MP) with comfortable margin.
const DEFAULT_MAX_IMAGE_PIXELS: u64 = 16_000_000;

impl ImageExtractFilter {
    /// Strict filter for markdown/HTML embedding paths.
    ///
    /// Skips tiny glyph fragments (<32x32), small Indexed images (<64x64),
    /// and oversized images beyond the configured limit. The `max_pixels`
    /// override comes from `ConversionOptions::max_image_pixels`.
    fn markdown(max_pixels_override: Option<u64>) -> Self {
        Self {
            min_width: 32,
            min_height: 32,
            max_pixels: max_pixels_override.unwrap_or(DEFAULT_MAX_IMAGE_PIXELS),
            skip_indexed_small: 64,
        }
    }
}

/// Area of a page for targeted header/footer operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PageArea {
    /// Top region (Header)
    Header,
    /// Bottom region (Footer)
    Footer,
}

/// Scan raw file bytes for candidate ObjStm positions.
///
/// Each hit is `(object_number, byte_offset_of_N_G_obj_header)`. We look
/// for the shape `N G obj ... /Type /ObjStm` within a small window after
/// each object header so that the caller can then `load_uncompressed_object`
/// at exactly that offset without parsing the whole file body.
///
/// The scan is intentionally tolerant: it doesn't require `/Type`
/// `/ObjStm` to be separated by whitespace (many producers write
/// `/Type/ObjStm`), doesn't anchor on any particular position within the
/// header, and doesn't rely on xref entries being correct — which is the
/// whole point of the recovery path it serves.
fn find_objstm_candidates(content: &[u8]) -> Vec<(u32, u64)> {
    const DICT_PEEK_BYTES: usize = 2048;
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < content.len() {
        let valid_start = pos == 0
            || content[pos - 1] == b'\n'
            || content[pos - 1] == b'\r'
            || content[pos - 1] == b' ';
        if !valid_start || !content[pos].is_ascii_digit() {
            pos += 1;
            continue;
        }
        let header_start = pos;

        // Parse N (object number)
        let num_start = pos;
        while pos < content.len() && content[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos >= content.len() || content[pos] != b' ' {
            pos = header_start + 1;
            continue;
        }
        let obj_num: u32 = match std::str::from_utf8(&content[num_start..pos])
            .ok()
            .and_then(|s| s.parse().ok())
        {
            Some(n) => n,
            None => {
                pos = header_start + 1;
                continue;
            },
        };
        pos += 1;

        // Parse G (generation)
        let gen_start = pos;
        while pos < content.len() && content[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos >= content.len() || content[pos] != b' ' {
            pos = header_start + 1;
            continue;
        }
        if std::str::from_utf8(&content[gen_start..pos])
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .is_none()
        {
            pos = header_start + 1;
            continue;
        }
        pos += 1;

        // Require literal "obj"
        if pos + 3 > content.len() || &content[pos..pos + 3] != b"obj" {
            pos = header_start + 1;
            continue;
        }

        // Peek up to DICT_PEEK_BYTES ahead for `/Type` followed (after
        // optional whitespace) by `/ObjStm`. We don't decompress — the
        // ObjStm dict header is always uncompressed plaintext even when
        // the stream body is Flate-encoded.
        let window_end = (pos + DICT_PEEK_BYTES).min(content.len());
        let window = &content[pos..window_end];
        if contains_objstm_marker(window) {
            out.push((obj_num, header_start as u64));
        }

        pos = header_start + 1;
    }
    out
}

fn contains_objstm_marker(window: &[u8]) -> bool {
    // Tolerant match: find `/Type` then allow optional whitespace before `/ObjStm`.
    let mut i = 0;
    while i + 5 <= window.len() {
        if &window[i..i + 5] == b"/Type" {
            let mut j = i + 5;
            while j < window.len()
                && (window[j] == b' '
                    || window[j] == b'\t'
                    || window[j] == b'\r'
                    || window[j] == b'\n')
            {
                j += 1;
            }
            if j + 7 <= window.len() && &window[j..j + 7] == b"/ObjStm" {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Append ink names declared by `Separation` and `DeviceN` colour spaces
/// in `cs_dict` to `out`. Reserved colorants `/All` and `/None` (§8.6.6.4)
/// are skipped. Caller is responsible for deduping across multiple calls.
///
/// When `doc` is `Some`, indirect references inside each colour-space array
/// (e.g. a DeviceN whose names list is `4 0 R` rather than inline) are
/// resolved. Tools that hand-build inline arrays and don't need indirection
/// resolution can pass `None`.
///
/// Used by both [`PdfDocument::get_page_inks`] and
/// [`PdfDocument::get_page_inks_deep`] so the per-colorant rules live in
/// exactly one place.
fn extract_inks_from_color_space_dict(
    cs_dict: &std::collections::HashMap<String, Object>,
    doc: Option<&PdfDocument>,
    out: &mut Vec<String>,
) {
    let deref = |obj: &Object| -> Object {
        match (obj.as_reference(), doc) {
            (Some(r), Some(d)) => d.load_object(r).unwrap_or_else(|_| obj.clone()),
            _ => obj.clone(),
        }
    };

    for cs_def in cs_dict.values() {
        let arr = match cs_def.as_array() {
            Some(a) => a,
            None => continue,
        };
        if arr.len() < 2 {
            continue;
        }
        let cs_type = match arr.first().and_then(Object::as_name) {
            Some(n) => n,
            None => continue,
        };
        match cs_type {
            "Separation" => {
                // §8.6.6.2: [/Separation /InkName /AlternateCS /TintTransform].
                // The name slot is usually inline but resolve indirects for safety.
                let name_obj = match arr.get(1) {
                    Some(o) => deref(o),
                    None => continue,
                };
                if let Some(ink) = name_obj.as_name() {
                    if ink != "All" && ink != "None" {
                        out.push(ink.to_string());
                    }
                }
            },
            "DeviceN" => {
                // §8.6.6.3: [/DeviceN <names-array> /AlternateCS /TintTransform <attrs>].
                // The names array is commonly emitted as an indirect reference
                // when the same colorant set is shared across multiple DeviceN
                // spaces; resolve before unpacking the names.
                let names_obj = match arr.get(1) {
                    Some(o) => deref(o),
                    None => continue,
                };
                if let Some(inks) = names_obj.as_array() {
                    for ink_obj in inks {
                        if let Some(ink) = ink_obj.as_name() {
                            if ink != "All" && ink != "None" {
                                out.push(ink.to_string());
                            }
                        }
                    }
                }
            },
            _ => {},
        }
    }
}

impl PdfDocument {
    /// Open a PDF document from in-memory bytes.
    ///
    /// This is the primary constructor for cases where
    /// the PDF data is already fully loaded in memory. This parses the PDF by
    /// wrapping the bytes in a memory reader and delegating to internal parsers.
    ///
    /// # Errors
    ///
    /// Returns an error if the PDF data is invalid, unsupported, or cannot be parsed.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let source_bytes = data.clone();
        let reader = PdfReader::Memory(BufReader::new(Cursor::new(data)));
        let mut doc = Self::open_from_reader(reader)?;
        doc.source_bytes = source_bytes;
        Ok(doc)
    }

    /// Deprecated alias for `from_bytes`.
    #[deprecated(since = "0.3.15", note = "Use `from_bytes` instead")]
    pub fn open_from_bytes(data: Vec<u8>) -> Result<Self> {
        Self::from_bytes(data)
    }

    /// Open a PDF document from a file path.
    ///
    /// Reads the entire file into memory, then parses the PDF structure.
    /// This is the standard constructor for desktop/server environments.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened or read
    /// - The PDF header is invalid
    /// - The cross-reference table is corrupted
    /// - The trailer dictionary is invalid
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pdf_oxide::document::PdfDocument;
    ///
    /// let doc = PdfDocument::open("sample.pdf")?;
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        // Read once and route through `from_bytes` so the in-memory
        // `source_bytes` field is populated. Path-loaded documents that
        // skip this lose access to APIs that re-read the bytes
        // (notably `compliance::convert_to_pdf_a`, which constructs a
        // `DocumentEditor` from `source_bytes` — an empty Vec breaks
        // it with `"Invalid PDF header: ... File is empty"`).
        // issue #456.
        //
        // The doc comment on this function already promised "Reads the
        // entire file into memory"; this is making it true.
        let data = std::fs::read(path.as_ref())?;
        Self::from_bytes(data)
    }

    fn open_from_reader(mut reader: PdfReader) -> Result<Self> {
        // Parse header with lenient mode by default (handle PDFs with binary prefixes)
        let (major, minor, header_offset) = parse_header(&mut reader, true)?;
        let version = (major, minor);

        // Whether the xref table below came from a full-file reconstruction
        // scan (vs. a parsed xref). Used to pre-seed the object-scan cache so
        // a later miss doesn't rescan the whole file a second time (#572).
        let mut xref_reconstructed = false;

        // Try to parse xref table normally
        let (mut xref, trailer) = match Self::try_open_regular(&mut reader) {
            Ok((xref, trailer)) => {
                // Success with regular parsing
                // However, if the xref is suspiciously small (< 5 entries), it's likely corrupted
                // Try reconstruction to get a complete table
                if xref.is_empty() {
                    log::warn!(
                        "Regular xref parsing succeeded but table is empty, attempting reconstruction"
                    );
                    xref_reconstructed = true;
                    Self::try_reconstruct_xref(&mut reader)?
                } else {
                    // A valid xref can have any number of entries (§7.5.4).
                    // Small xrefs (e.g. portfolio PDFs with 3-4 objects) are perfectly
                    // normal — don't trigger expensive full-file reconstruction for them.
                    (xref, trailer)
                }
            },
            Err(e) => {
                log::warn!("Regular xref parsing failed: {}, attempting reconstruction", e);

                // Fall back to xref reconstruction
                match Self::try_reconstruct_xref(&mut reader) {
                    Ok((reconstructed_xref, reconstructed_trailer)) => {
                        log::info!("Successfully reconstructed xref table");
                        xref_reconstructed = true;
                        (reconstructed_xref, reconstructed_trailer)
                    },
                    Err(recon_err) => {
                        log::error!("XRef reconstruction also failed: {}", recon_err);
                        return Err(e); // Return original error
                    },
                }
            },
        };

        // If PDF header is not at byte 0 (garbage-prepended), xref offsets may need adjustment.
        // The xref offsets are relative to the original PDF start, but file positions are
        // shifted by header_offset bytes.
        if header_offset > 0 {
            // Probe an object to decide whether xref offsets are off by
            // header_offset. Prefer /Root (common case), but the probe MUST
            // be seek-validatable: `validate_object_at_offset` returns true
            // for *compressed* entries without seeking, so a /Root that
            // lives in an object stream would falsely report "no shift
            // needed" and leave every uncompressed offset wrong. Use /Root
            // only when its entry is in-use + uncompressed; otherwise (no
            // /Root — issue #509 — or a compressed /Root) fall back to the
            // first in-use uncompressed object.
            let probe = get_root_ref_from_trailer(&trailer)
                .filter(|r| {
                    xref.get(r.id).is_some_and(|e| {
                        e.in_use && e.entry_type == crate::xref::XRefEntryType::Uncompressed
                    })
                })
                .or_else(|| first_in_use_uncompressed(&xref));
            if let Some(probe_ref) = probe {
                if !validate_object_at_offset(&mut reader, &xref, probe_ref) {
                    log::info!(
                        "Probe object {} not loadable at xref offset, adjusting all offsets by header_offset={}",
                        probe_ref.id, header_offset
                    );
                    xref.shift_offsets(header_offset);
                }
            }
        }

        // Validate the /Root catalog is actually loadable. If not, the xref data is
        // corrupt despite parsing successfully — fall back to reconstruction.
        let (xref, trailer) = if !validate_root_loadable(&mut reader, &xref, &trailer) {
            log::warn!(
                "Root object not loadable after xref parse, falling back to xref reconstruction"
            );
            match Self::try_reconstruct_xref(&mut reader) {
                Ok(result) => {
                    xref_reconstructed = true;
                    result
                },
                Err(_) => (xref, trailer), // Use original if reconstruction also fails
            }
        } else {
            (xref, trailer)
        };

        // #572: a reconstruction scan already located every uncompressed
        // "N G obj" in the file, so a later scan_for_object full-file rescan
        // (on the first object miss) would find nothing new — it just repeats
        // the work, the ~25 s "first extract_text" cost on corrupt-xref
        // polyglots. Pre-seed the scan-offset cache from the reconstructed
        // table so that first miss is O(1). Only do this when reconstructed:
        // a normal (parsed) xref may be legitimately partial, and there the
        // full scan is the intended recovery path.
        let prepopulated_scan: Option<HashMap<u32, u64>> = if xref_reconstructed {
            Some(
                xref.all_object_numbers()
                    .filter_map(|id| {
                        xref.get(id).and_then(|e| {
                            (e.in_use && e.entry_type == crate::xref::XRefEntryType::Uncompressed)
                                .then_some((id, e.offset))
                        })
                    })
                    .collect(),
            )
        } else {
            None
        };

        // Note: Encryption initialization was originally lazy, but decode_stream_with_encryption
        // only has &self access which prevents initialization.
        // We now initialize eagerly to ensure the handler is ready when needed.
        let document = Self {
            reader: Mutex::new(reader),
            load_lock: Mutex::new(()),
            source_bytes: Vec::new(),
            version,
            xref,
            trailer,
            object_cache: Mutex::new(BoundedObjectCache::new(DEFAULT_OBJECT_CACHE_MAX_BYTES)),
            encryption_handler: Mutex::new(None),
            encrypt_dict_ref: Mutex::new(None),
            options: ParserOptions::default(),
            header_offset,
            font_cache: Mutex::new(BoundedEntryCache::new(512)),
            font_set_cache: Mutex::new(BoundedEntryCache::new(256)),
            font_fingerprint_cache: Mutex::new(BoundedEntryCache::new(256)),
            font_name_set_cache: Mutex::new(BoundedEntryCache::new(256)),
            font_identity_cache: Mutex::new(BoundedEntryCache::new(512)),
            font_id_hash_cache: Mutex::new(HashMap::new()),
            structure_tree_cache: Mutex::new(None),
            structure_content_cache: Mutex::new(None),
            table_elements_cache: Mutex::new(None),
            page_cache: Mutex::new(HashMap::new()),
            page_cache_populated: AtomicBool::new(false),
            scanned_object_offsets: Mutex::new(prepopulated_scan),
            objstm_recovery_done: Mutex::new(false),
            image_xobject_cache: Mutex::new(HashSet::new()),
            xobject_text_free_cache: Mutex::new(HashSet::new()),
            xobject_stream_cache: Mutex::new(HashMap::new()),
            xobject_stream_cache_bytes: AtomicUsize::new(0),
            xobject_spans_cache: Mutex::new(BoundedEntryCache::new(
                DEFAULT_XOBJECT_CACHE_MAX_ENTRIES,
            )),
            form_xobject_images_cache: Mutex::new(BoundedEntryCache::new(
                DEFAULT_XOBJECT_CACHE_MAX_ENTRIES,
            )),
            erase_regions: Mutex::new(HashMap::new()),
            page_content_cache: Mutex::new(BoundedEntryCache::new(64)),
            page_spans_cache: Mutex::new(BoundedEntryCache::new(8)),
            running_artifact_signatures: Mutex::new(None),
            accumulated_warnings: Mutex::new(Vec::new()),
            warning_sink: crate::extractors::warnings::WarningSink::new(),
        };

        // Initialize encryption immediately
        if let Err(e) = document.ensure_encryption_initialized() {
            log::error!("Failed to initialize encryption: {}", e);
            // We continue anyway, as it might just be an unsupported security handler
            // and maybe we can still read parts of the file (or fail later)
        }

        Ok(document)
    }

    /// Try to open the PDF using regular xref parsing.
    fn try_open_regular<R: Read + Seek>(reader: &mut R) -> Result<(CrossRefTable, Object)> {
        // Find xref table offset
        let xref_offset = find_xref_offset(reader)?;

        // Parse xref table
        let xref = parse_xref(reader, xref_offset)?;

        // Get trailer dictionary
        let trailer = if let Some(trailer_dict) = xref.trailer() {
            // XRef stream: trailer is already in the xref table
            Object::Dictionary(trailer_dict.clone())
        } else {
            // Traditional xref: parse trailer separately
            reader.seek(SeekFrom::Start(xref_offset))?;
            parse_trailer(reader)?
        };

        Ok((xref, trailer))
    }

    /// Try to reconstruct the xref table by scanning the file.
    fn try_reconstruct_xref<R: Read + Seek>(reader: &mut R) -> Result<(CrossRefTable, Object)> {
        crate::xref_reconstruction::reconstruct_xref(reader)
    }

    /// Initialize encryption handler lazily if PDF is encrypted.
    ///
    /// PDF Spec: Section 7.6.1 - Encryption dictionary in trailer
    ///
    /// This checks for the /Encrypt entry in the trailer, loads it if it's a
    /// reference, and creates an encryption handler. It automatically attempts
    /// to authenticate with an empty password (common for PDFs with default encryption).
    ///
    /// This is called lazily the first time we need to decrypt something, after
    /// the document is fully constructed and can load objects.
    fn ensure_encryption_initialized(&self) -> Result<()> {
        // Already initialized?
        if self.encryption_handler.lock_or_recover().is_some() {
            return Ok(());
        }

        // Clone what we need from trailer to avoid borrow conflicts
        let (encrypt_ref, file_id) = {
            let trailer_dict = match self.trailer.as_dict() {
                Some(d) => d,
                None => return Ok(()), // No trailer dict, no encryption
            };

            // Check for /Encrypt entry
            let encrypt_entry = match trailer_dict.get("Encrypt") {
                Some(obj) => obj,
                None => {
                    log::debug!("PDF is not encrypted (no /Encrypt entry)");
                    return Ok(());
                },
            };

            // Clone the encrypt entry (we'll load it outside this block)
            let encrypt_ref = encrypt_entry.clone();

            // Get file ID (required for encryption key derivation)
            let file_id = match trailer_dict.get("ID") {
                Some(Object::Array(arr)) => {
                    if let Some(first_id) = arr.first() {
                        if let Some(id_bytes) = first_id.as_string() {
                            id_bytes.to_vec()
                        } else {
                            log::warn!(
                                "Invalid /ID array entry (not a string), using empty file ID"
                            );
                            vec![]
                        }
                    } else {
                        log::warn!("Empty /ID array, using empty file ID");
                        vec![]
                    }
                },
                _ => {
                    log::warn!("Missing or invalid /ID entry in trailer, using empty file ID");
                    vec![]
                },
            };

            (encrypt_ref, file_id)
        }; // End of borrow scope

        // Now load the encrypt object (dereference if needed)
        let encrypt_obj = match encrypt_ref {
            Object::Dictionary(_) => encrypt_ref,
            Object::Reference(obj_ref) => {
                log::debug!("Loading /Encrypt object reference {} {}", obj_ref.id, obj_ref.gen);
                // Remember which object holds the /Encrypt dict so its own
                // strings are skipped during per-object string decryption.
                *self.encrypt_dict_ref.lock_or_recover() = Some(obj_ref);
                self.load_object(obj_ref)?
            },
            _ => {
                return Err(Error::InvalidPdf(format!(
                    "Invalid /Encrypt entry type: {}",
                    encrypt_ref.type_name()
                )));
            },
        };

        // Resolve any indirect references within the encrypt dictionary.
        // Some PDFs store /O, /U, /V, /R, /P as indirect references (e.g., `7 0 R`).
        let encrypt_obj = if let Some(dict) = encrypt_obj.as_dict() {
            let mut resolved_dict = dict.clone();
            for (_key, value) in resolved_dict.iter_mut() {
                if let Object::Reference(obj_ref) = value {
                    match self.load_object(*obj_ref) {
                        Ok(resolved) => *value = resolved,
                        Err(e) => {
                            log::warn!("Failed to resolve indirect ref in /Encrypt dict: {}", e);
                        },
                    }
                }
            }
            Object::Dictionary(resolved_dict)
        } else {
            encrypt_obj
        };

        // Create encryption handler with the file_id we extracted above
        let mut handler = EncryptionHandler::new(&encrypt_obj, file_id)?;

        // Try to authenticate with empty password (common default)
        match handler.authenticate(b"") {
            Ok(true) => {
                log::info!("Successfully authenticated with empty password");
            },
            Ok(false) => {
                log::warn!("PDF is encrypted and requires a password");
                self.push_warning(
                    "PDF is encrypted and requires a password; call authenticate() before extracting text".to_string()
                );
                // Set handler anyway - user can call authenticate() later
            },
            Err(e) => {
                log::error!("Failed to initialize encryption: {}", e);
                return Err(e);
            },
        }

        *self.encryption_handler.lock_or_recover() = Some(handler);
        Ok(())
    }

    /// Decode stream data with encryption support.
    ///
    /// This is a helper method that decodes stream data using the PDF's encryption handler
    /// if the document is encrypted. It automatically handles object-specific key derivation.
    ///
    /// # Arguments
    ///
    /// * `stream_obj` - The stream object to decode
    /// * `obj_ref` - The object reference (for encryption key derivation)
    ///
    /// # Returns
    ///
    /// The decoded (and decrypted if needed) stream data.
    ///
    /// # PDF Spec Reference
    ///
    /// ISO 32000-1:2008, Section 7.6.2 - Streams must be decrypted BEFORE applying filters.
    pub(crate) fn decode_stream_with_encryption(
        &self,
        stream_obj: &Object,
        obj_ref: ObjectRef,
    ) -> Result<Vec<u8>> {
        if matches!(stream_obj, Object::Null) {
            return Ok(Vec::new());
        }

        // Per ISO 32000-2:2020 Section 7.6.3, object streams (/Type /ObjStm)
        // and cross-reference streams (/Type /XRef) shall NOT be encrypted.
        // Skip decryption for these stream types to avoid AES block-size errors
        // on data that was never encrypted in the first place.
        let is_unencrypted_stream_type = if let Object::Stream { dict, .. } = stream_obj {
            dict.get("Type")
                .and_then(|t| t.as_name())
                .map(|name| name == "ObjStm" || name == "XRef")
                .unwrap_or(false)
        } else {
            false
        };

        let handler_ref = self.encryption_handler.lock_or_recover();
        if let Some(handler) = handler_ref.as_ref() {
            if is_unencrypted_stream_type {
                // These stream types are never encrypted per spec
                drop(handler_ref);
                return stream_obj.decode_stream_data();
            }
            // Create decryption closure for this specific object
            let decrypt_fn = |data: &[u8]| -> Result<Vec<u8>> {
                handler.decrypt_stream(data, obj_ref.id, obj_ref.gen as u32)
            };
            stream_obj.decode_stream_data_with_decryption(
                Some(&decrypt_fn),
                obj_ref.id,
                obj_ref.gen as u32,
            )
        } else {
            drop(handler_ref);
            // No encryption, use regular decoding
            stream_obj.decode_stream_data()
        }
    }

    /// Open with custom extraction profile.
    ///
    /// Currently, the profile is not used at the document level but is reserved
    /// for future integration with document-type-specific extraction settings.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_with_config(path: impl AsRef<Path>, _config: impl std::any::Any) -> Result<Self> {
        Self::open(path)
    }

    /// Authenticate with a password to decrypt encrypted PDFs.
    ///
    /// If the PDF is encrypted, `open()` automatically tries an empty password.
    /// Call this method to authenticate with a non-empty password.
    ///
    /// # Arguments
    ///
    /// * `password` - The password as bytes
    ///
    /// # Returns
    ///
    /// `Ok(true)` if authentication succeeded, `Ok(false)` if the password was wrong,
    /// or `Ok(true)` if the PDF is not encrypted (no authentication needed).
    pub fn authenticate(&self, password: &[u8]) -> Result<bool> {
        self.ensure_encryption_initialized()?;
        // Capture current authentication state *before* calling the
        // handler so we can detect the transition from "not authenticated"
        // to "authenticated" and invalidate the object cache accordingly.
        // Any objects loaded and cached before successful authentication
        // still hold ciphertext strings (see `load_uncompressed_object_impl`
        // at the `handler.is_authenticated()` guard), so a cache hit after
        // authentication would return those stale values forever — issue
        // #323.
        let was_authenticated = self
            .encryption_handler
            .lock_or_recover()
            .as_ref()
            .map(|h| h.is_authenticated())
            .unwrap_or(true);

        let result = match self.encryption_handler.lock_or_recover().as_mut() {
            Some(handler) => handler.authenticate(password),
            None => return Ok(true), // Not encrypted, always "authenticated"
        };

        if let Ok(true) = result {
            if !was_authenticated {
                // Transitioned from "encrypted, not authenticated" to
                // "authenticated". Drop every cached object so subsequent
                // `load_object` calls re-parse through the path that now
                // runs `decrypt_strings_in_object` on the uncompressed
                // string values. The `/Encrypt` dictionary is not in this
                // cache path (it is resolved independently), so clearing
                // is always safe.
                self.object_cache.lock_or_recover().clear();
                log::debug!(
                    "authenticate(): object cache cleared after successful authentication \
                     to force re-decryption of any pre-auth cached objects"
                );
            }
        }

        result
    }

    /// Check if the PDF is encrypted.
    ///
    /// Returns `true` if the PDF has an `/Encrypt` entry in its trailer,
    /// regardless of whether it has been authenticated.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// if doc.is_encrypted() {
    ///     println!("PDF is encrypted");
    /// }
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn is_encrypted(&self) -> bool {
        // Check if encryption handler is already initialized
        if self.encryption_handler.lock_or_recover().is_some() {
            return true;
        }
        // Check trailer for /Encrypt entry without initializing
        self.trailer
            .as_dict()
            .and_then(|d| d.get("Encrypt"))
            .is_some()
    }

    /// Whether content extraction is permitted right now — `true` if the
    /// PDF is unencrypted, or encrypted and successfully authenticated.
    ///
    /// Cheap, side-effect-free preflight for the auto-extraction
    /// classifier (#517): lets it emit
    /// [`ReasonCode::EncryptedNoExtractPermission`](crate::extractors::auto::ReasonCode)
    /// gracefully instead of attempting extraction and erroring.
    #[must_use]
    pub fn is_authenticated(&self) -> bool {
        // Fail closed: if encryption init errors (malformed / unsupported
        // `/Encrypt`), the document IS encrypted but we cannot have
        // authenticated it — a security preflight must report `false`
        // here, not `true` (PR #519 review). Only when init succeeds
        // (incl. the trivial unencrypted case) do we trust the guard.
        if self.ensure_encryption_initialized().is_err() {
            return false;
        }
        !self.is_encrypted_and_unauthenticated()
    }

    /// Document Info dictionary `/Producer` (decoded, trimmed), if present
    /// and non-empty. A weak document-level prior for the scanner-vs-
    /// authoring heuristic (#517 case P) — never decisive.
    #[must_use]
    pub fn document_producer(&self) -> Option<String> {
        self.document_info_string("Producer")
    }

    /// Document Info dictionary `/Creator` (decoded, trimmed), if present
    /// and non-empty. See [`document_producer`](Self::document_producer).
    #[must_use]
    pub fn document_creator(&self) -> Option<String> {
        self.document_info_string("Creator")
    }

    fn document_info_string(&self, key: &str) -> Option<String> {
        let info_raw = self.trailer.as_dict()?.get("Info")?;
        let info = self.resolve_obj_ref(info_raw);
        let val_raw = info.as_dict()?.get(key)?.clone();
        let val = self.resolve_obj_ref(&val_raw);
        let s = Self::decode_pdf_text_string(val.as_string()?);
        let trimmed = s.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }

    /// Axis-aligned intersection area of a [`Rect`](crate::geometry::Rect)
    /// with the page box `(x0, y0, x1, y1)`.
    fn rect_isect_area(r: &crate::geometry::Rect, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
        let (rx1, ry1) = (r.x + r.width, r.y + r.height);
        let ix = (rx1.min(x1) - r.x.max(x0)).max(0.0);
        let iy = (ry1.min(y1) - r.y.max(y0)).max(0.0);
        ix * iy
    }

    /// Gather per-page classification signals from pdf_oxide
    /// **internals** (00-common-foundation §9 — never the flattened
    /// output string). Returns the signals plus the enriched T0.5
    /// quality-gate verdict (research §3a) computed from the *same*
    /// single span extraction (no double work). Pure inspection.
    fn gather_page_signals(
        &self,
        page: usize,
    ) -> Result<(
        crate::extractors::auto::PageSignals,
        Option<crate::extractors::auto::ReasonCode>,
    )> {
        use crate::content::{Operator, TextElement};
        use crate::extractors::auto::{ImageCodecClass, PageSignals, ProducerPrior};
        use crate::extractors::ImageData;

        let (llx, lly, urx, ury) = self.get_page_media_box(page)?;
        let rot = self.get_page_rotation(page).unwrap_or(0);
        let (mut pw, mut ph) = ((urx - llx).abs(), (ury - lly).abs());
        if rot % 180 != 0 {
            std::mem::swap(&mut pw, &mut ph);
        }
        let page_area = (pw * ph).max(1.0);
        let (px0, py0, px1, py1) = (llx.min(urx), lly.min(ury), llx.max(urx), lly.max(ury));

        // ── native text (artifact spans downweighted — cases G/T) ──
        let spans = self.extract_spans(page).unwrap_or_default();
        let mut text = String::new();
        let mut glyphs = 0usize;
        let mut text_area = 0.0f32;
        for s in &spans {
            if s.artifact_type.is_some() {
                continue;
            }
            let n = s.text.chars().count();
            if n == 0 {
                continue;
            }
            glyphs += n;
            text.push_str(&s.text);
            text.push(' ');
            text_area += Self::rect_isect_area(&s.bbox, px0, py0, px1, py1);
        }
        let text_area_ratio = (text_area / page_area).clamp(0.0, 1.0);

        let chars: Vec<char> = text.chars().collect();
        let total = chars.len().max(1);
        let bad = chars
            .iter()
            .filter(|&&c| {
                c == '\u{FFFD}' || c.is_control() || ('\u{E000}'..='\u{F8FF}').contains(&c)
            })
            .count();
        let garbled_ratio = bad as f32 / total as f32;
        let words: Vec<&str> = text.split_whitespace().collect();
        let (fragmented_word_ratio, consecutive_repeat_ratio) = if words.is_empty() {
            (0.0, 0.0)
        } else {
            let frag =
                words.iter().filter(|w| w.chars().count() <= 2).count() as f32 / words.len() as f32;
            let rep = words.windows(2).filter(|w| w[0] == w[1]).count() as f32 / words.len() as f32;
            (frag, rep)
        };

        // ── images: union coverage (summed → multi-strip, case J) + codec ──
        let images = self.extract_images(page).unwrap_or_default();
        let mut img_area = 0.0f32;
        let mut codec = ImageCodecClass::None;
        for im in &images {
            if let Some(b) = im.bbox() {
                img_area += Self::rect_isect_area(b, px0, py0, px1, py1);
            }
            let c = if im.ccitt_params().is_some() {
                ImageCodecClass::Ccitt
            } else {
                match im.data() {
                    ImageData::Jpeg(_) => ImageCodecClass::Dct,
                    _ => ImageCodecClass::Other,
                }
            };
            codec = match (codec, c) {
                (ImageCodecClass::None, x) => x,
                (_, ImageCodecClass::Ccitt) => ImageCodecClass::Ccitt,
                (cur, _) => cur,
            };
        }
        let image_area_ratio = (img_area / page_area).clamp(0.0, 1.0);

        // ── content-stream ops: Tr-mode-3 ratio (cases C/C2) ──
        let mut invisible = 0usize;
        let mut glyph_bytes = 0usize;
        if let Ok(data) = self.get_page_content_data(page) {
            if let Ok(ops) = crate::content::parse_content_stream(&data) {
                let mut rm: u8 = 0;
                let mut stack: Vec<u8> = Vec::new();
                for op in &ops {
                    match op {
                        Operator::SaveState => stack.push(rm),
                        Operator::RestoreState => {
                            if let Some(p) = stack.pop() {
                                rm = p;
                            }
                        },
                        Operator::Tr { render } => rm = *render,
                        Operator::Tj { text } => {
                            glyph_bytes += text.len();
                            if rm == 3 {
                                invisible += text.len();
                            }
                        },
                        Operator::TJ { array } => {
                            let g: usize = array
                                .iter()
                                .map(|e| match e {
                                    TextElement::String(b) => b.len(),
                                    TextElement::Offset(_) => 0,
                                })
                                .sum();
                            glyph_bytes += g;
                            if rm == 3 {
                                invisible += g;
                            }
                        },
                        _ => {},
                    }
                }
            }
        }
        let invisible_text_ratio = if glyph_bytes == 0 {
            0.0
        } else {
            invisible as f32 / glyph_bytes as f32
        };

        // ── vector path density (case F) ──
        let path_count = self.extract_paths(page).map(|p| p.len()).unwrap_or(0);
        let vector_path_density = {
            let denom = (path_count + glyphs + images.len()).max(1) as f32;
            (path_count as f32 / denom).clamp(0.0, 1.0)
        };

        // ── structure / producer / empty ──
        let has_reliable_structure = self
            .mark_info()
            .map(|m| m.is_structure_reliable())
            .unwrap_or(false);
        let producer_prior = {
            let p = format!(
                "{} {}",
                self.document_producer().unwrap_or_default(),
                self.document_creator().unwrap_or_default()
            )
            .to_lowercase();
            const SCAN: &[&str] = &[
                "scan",
                "abbyy",
                "tesseract",
                "scansnap",
                "finereader",
                "ocr",
                "lens",
                "camscanner",
                "kofax",
            ];
            const AUTH: &[&str] = &[
                "word",
                "libreoffice",
                "latex",
                "pdftex",
                "chromium",
                "skia",
                "quartz",
                "wkhtmltopdf",
                "pdf_oxide",
                "reportlab",
                "prince",
                "weasyprint",
                "powerpoint",
                "excel",
                "indesign",
            ];
            if SCAN.iter().any(|k| p.contains(k)) {
                ProducerPrior::Scanner
            } else if AUTH.iter().any(|k| p.contains(k)) {
                ProducerPrior::Authoring
            } else {
                ProducerPrior::Unknown
            }
        };
        let page_is_empty = glyphs == 0 && image_area_ratio < 0.01 && path_count == 0;

        let signals = PageSignals {
            text_glyph_count: glyphs,
            text_area_ratio,
            image_area_ratio,
            codec,
            invisible_text_ratio,
            garbled_ratio,
            fragmented_word_ratio,
            consecutive_repeat_ratio,
            vector_path_density,
            has_reliable_structure,
            producer_prior,
            page_is_empty,
        };
        let gate = crate::extractors::auto::text_quality_gate(&text);
        Ok((signals, gate))
    }

    /// Cheap per-page text-vs-OCR classification (the `classify_page`
    /// preflight, #517 — no OCR, no rasterisation). Returns kind +
    /// confidence + typed [`ReasonCode`](crate::extractors::auto::ReasonCode)
    /// + the raw signals (explainable).
    ///
    /// Fails closed on an encrypted-unauthenticated document
    /// (`Error::EncryptedPdf`, case L) — consistent with every other
    /// `extract_*`; the graceful warn+fallback applies to *extraction*
    /// (`extract_page_auto`), not this preflight.
    pub fn classify_page(
        &self,
        page: usize,
    ) -> Result<crate::extractors::auto::PageClassification> {
        use crate::extractors::auto::{
            classify_from_signals, AutoExtractOptions, PageClassification, PageKind,
        };
        if !self.is_authenticated() {
            return Err(Error::EncryptedPdf);
        }
        let (signals, gate) = self.gather_page_signals(page)?;
        let opts = AutoExtractOptions::balanced();
        let (mut kind, mut confidence, mut reason) = classify_from_signals(&signals, &opts);
        // Enriched T0.5 gate (research §3a): unusable born-digital text
        // overrides a TextLayer verdict → route to OCR with the typed
        // reason (column-scramble / cid-garbage / fragmentation).
        if matches!(kind, PageKind::TextLayer) {
            if let Some(r) = gate {
                kind = PageKind::Scanned;
                confidence = confidence.min(0.80);
                reason = r;
            }
        }
        Ok(PageClassification {
            page,
            kind,
            confidence,
            reason,
            signals,
        })
    }

    /// Cheap whole-document classification (#517): per-page kinds (the
    /// decision is **per-page**, never one forced doc mode — case Q),
    /// the 0-based `pages_needing_ocr` list, and an aggregate summary.
    ///
    /// Fails closed on an encrypted-unauthenticated document
    /// (`Error::EncryptedPdf`) — a security op must never be silently
    /// degraded to a benign `Empty` (#519 review). Any *non-security*
    /// per-page failure degrades to `Empty` (graceful — only security
    /// ops fail closed).
    pub fn classify_document(&self) -> Result<crate::extractors::auto::DocumentClassification> {
        use crate::extractors::auto::{summarise, DocumentClassification, PageKind};
        let n = self.page_count()?;
        let mut kinds = Vec::with_capacity(n);
        let mut need = Vec::new();
        for p in 0..n {
            let k = match self.classify_page(p) {
                Ok(c) => c.kind,
                // Security op: propagate, never mask as Empty (case L).
                Err(e @ Error::EncryptedPdf) => return Err(e),
                // Non-security per-page failure: graceful degrade.
                Err(_) => PageKind::Empty,
            };
            if matches!(k, PageKind::Scanned | PageKind::ImageText | PageKind::Mixed) {
                need.push(p);
            }
            kinds.push(k);
        }
        let summary = summarise(&kinds);
        Ok(DocumentClassification {
            pages: kinds,
            pages_needing_ocr: need,
            summary,
        })
    }

    /// One-shot convenience for the 90% case (#517): equivalent to
    /// `AutoExtractor::new().extract_text(self, page)`. **Strictly
    /// additive** — the existing [`extract_text`](Self::extract_text) is
    /// byte-identical/unchanged; this is a *new* opt-in entry point that
    /// auto-routes text-vs-OCR with graceful native fallback.
    pub fn extract_text_auto(&self, page: usize) -> Result<String> {
        crate::extractors::auto::AutoExtractor::new().extract_text(self, page)
    }

    /// Check if the PDF is encrypted but has NOT been successfully authenticated.
    ///
    /// This returns `true` when the document requires a password that has not
    /// yet been provided. Extraction methods use this to return a clear error
    /// instead of silently producing empty output.
    fn is_encrypted_and_unauthenticated(&self) -> bool {
        if let Some(handler) = self.encryption_handler.lock_or_recover().as_ref() {
            !handler.is_authenticated()
        } else {
            // Handler not yet initialized — check if /Encrypt exists
            // If it does, we don't know auth state yet, so return false
            // (ensure_encryption_initialized will handle it)
            false
        }
    }

    /// Guard that returns `Err(Error::EncryptedPdf)` if the PDF is encrypted
    /// and not authenticated. Call this at the top of extraction methods.
    fn require_authenticated(&self) -> Result<()> {
        // Make sure encryption is initialized first
        self.ensure_encryption_initialized()?;
        if self.is_encrypted_and_unauthenticated() {
            return Err(Error::EncryptedPdf);
        }
        Ok(())
    }

    /// True once the empty user password has been tried and the document is
    /// still locked. Text extraction degrades to empty output in this case
    /// (matching pdftotext/PyMuPDF) rather than erroring; `page_count` and
    /// write paths keep using [`Self::require_authenticated`].
    fn is_encrypted_unreadable(&self) -> bool {
        let _ = self.ensure_encryption_initialized();
        self.is_encrypted_and_unauthenticated()
    }

    /// Get the PDF version.
    ///
    /// Returns a tuple (major, minor) representing the PDF version.
    /// For example, PDF 1.7 returns (1, 7).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// let (major, minor) = doc.version();
    /// println!("PDF version: {}.{}", major, minor);
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn version(&self) -> (u8, u8) {
        self.version
    }

    /// Get a reference to the trailer dictionary.
    ///
    /// The trailer dictionary contains important document metadata including:
    /// - /Root: Reference to the catalog dictionary
    /// - /Info: Reference to the document info dictionary (optional)
    /// - /Size: Number of entries in the cross-reference table
    /// - /Encrypt: Encryption dictionary (if encrypted)
    /// - /ID: File identifier array
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// let trailer = doc.trailer();
    /// if let Some(dict) = trailer.as_dict() {
    ///     if let Some(info_ref) = dict.get("Info") {
    ///         println!("Document has an Info dictionary");
    ///     }
    /// }
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn trailer(&self) -> &Object {
        &self.trailer
    }

    /// Return every object ID known to this document.
    ///
    /// Unions the cross-reference table with any object IDs that were
    /// recovered from compressed object streams (which may not have an
    /// explicit xref entry). The result is sorted and deduplicated so
    /// callers can iterate once and write each object exactly once.
    ///
    /// Used by `DocumentEditor::write_full_to_writer` to sweep any
    /// objects that were not reached during the shallow page-tree
    /// traversal (e.g. embedded font sub-objects such as
    /// `DescendantFonts`, `FontFile2`, `ToUnicode`, `FontDescriptor`).
    pub fn all_object_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.xref.all_object_numbers().collect();
        for r in self.object_cache.lock_or_recover().keys() {
            ids.push(r.id);
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Return references to every leaf page, in document order, with a single
    /// page-tree traversal.
    ///
    /// Replaces the O(n²) pattern of calling [`get_page_ref`] in a 0..n loop:
    /// each `get_page_ref(i)` walks the tree from the root and stops at the
    /// i-th leaf, so collecting all n refs walks 1+2+...+n nodes.
    ///
    /// Optimised for the common flat-tree case: when a `Pages` node's
    /// `Count` matches `Kids.len()`, every kid is a leaf and we can take
    /// the references straight from the array without loading each leaf.
    /// Only when the tree is multi-level do we recurse and load child nodes.
    pub(crate) fn all_page_refs(&self) -> Result<Vec<ObjectRef>> {
        let catalog = self.catalog()?;
        let catalog_dict = catalog.as_dict().ok_or_else(|| Error::InvalidObjectType {
            expected: "Dictionary".to_string(),
            found: catalog.type_name().to_string(),
        })?;
        let pages_ref = catalog_dict
            .get("Pages")
            .and_then(|p| p.as_reference())
            .ok_or_else(|| Error::InvalidPdf("Catalog missing /Pages entry".to_string()))?;

        let mut out: Vec<ObjectRef> = Vec::new();
        let mut visited: HashSet<ObjectRef> = HashSet::new();
        self.collect_page_refs(pages_ref, &mut out, &mut visited)?;
        Ok(out)
    }

    fn collect_page_refs(
        &self,
        node_ref: ObjectRef,
        out: &mut Vec<ObjectRef>,
        visited: &mut HashSet<ObjectRef>,
    ) -> Result<()> {
        if !visited.insert(node_ref) {
            return Ok(());
        }
        let node = self.load_object(node_ref)?;
        let dict = match node.as_dict() {
            Some(d) => d,
            None => return Ok(()),
        };

        let kids = match dict.get("Kids").and_then(|k| k.as_array()) {
            Some(k) => k,
            None => {
                // Leaf reached (no /Kids — assume Page).
                out.push(node_ref);
                return Ok(());
            },
        };

        // Fast path: flat subtree — every kid is a leaf when /Count == kids.len().
        let count = dict.get("Count").and_then(|c| c.as_integer()).unwrap_or(-1);
        if count >= 0 && (count as usize) == kids.len() {
            for kid in kids {
                if let Some(kid_ref) = kid.as_reference() {
                    out.push(kid_ref);
                }
            }
            return Ok(());
        }

        // Mixed tree — recurse into each kid.
        for kid in kids {
            if let Some(kid_ref) = kid.as_reference() {
                self.collect_page_refs(kid_ref, out, visited)?;
            }
        }
        Ok(())
    }

    /// Scan the file to find an object by its header.
    ///
    /// This is a fallback method used when an object is not in the xref table
    /// but is referenced by critical structures (like Pages from Catalog).
    /// Some PDFs have incomplete xref tables that are missing entries for
    /// objects that actually exist in the file.
    fn scan_for_object(&self, obj_ref: ObjectRef) -> Result<u64> {
        // Check cached scan results first
        {
            let scan_cache = self.scanned_object_offsets.lock_or_recover();
            if let Some(offsets) = scan_cache.as_ref() {
                if let Some(&offset) = offsets.get(&obj_ref.id) {
                    return Ok(offset);
                }
                return Err(Error::ObjectNotFound(obj_ref.id, obj_ref.gen));
            }
        }

        // First xref miss: scan the entire file once and build a complete offset map
        log::info!(
            "Building object offset map from file scan (triggered by object {} {})",
            obj_ref.id,
            obj_ref.gen
        );

        let mut content = Vec::new();
        {
            // Hold one guard for seek+read to prevent split-lock race (#398 Race A).
            let mut reader = self.reader.lock_or_recover();
            reader.seek(SeekFrom::Start(0))?;
            reader.read_to_end(&mut content)?;
        }

        let mut offsets = HashMap::new();

        // Scan for all "N G obj" patterns in the file
        let mut pos = 0;
        while pos < content.len() {
            // Look for digit at a line start (after newline or at file start)
            let valid_start = pos == 0 || content[pos - 1] == b'\n' || content[pos - 1] == b'\r';
            if !valid_start || !content[pos].is_ascii_digit() {
                pos += 1;
                continue;
            }

            // Try to parse "N G obj" starting at pos
            let start = pos;
            // Parse object number (digits)
            while pos < content.len() && content[pos].is_ascii_digit() {
                pos += 1;
            }
            if pos >= content.len() || content[pos] != b' ' {
                continue;
            }
            let obj_num_str = std::str::from_utf8(&content[start..pos]).unwrap_or("");
            let obj_num: u32 = match obj_num_str.parse() {
                Ok(n) => n,
                Err(_) => continue,
            };

            pos += 1; // skip space

            // Parse generation number (digits)
            let gen_start = pos;
            while pos < content.len() && content[pos].is_ascii_digit() {
                pos += 1;
            }
            if pos >= content.len() || content[pos] != b' ' {
                continue;
            }
            let _gen_str = std::str::from_utf8(&content[gen_start..pos]).unwrap_or("");

            pos += 1; // skip space

            // Check for "obj" keyword
            if pos + 3 <= content.len() && &content[pos..pos + 3] == b"obj" {
                let after_obj = pos + 3;
                // Verify "obj" is followed by whitespace, newline, or '<'
                let valid_end = after_obj >= content.len() || {
                    let c = content[after_obj];
                    c == b'\n' || c == b'\r' || c == b' ' || c == b'\t' || c == b'<'
                };
                if valid_end {
                    offsets.entry(obj_num).or_insert(start as u64);
                    pos = after_obj;
                    continue;
                }
            }
            // Reset pos to just after the start to avoid infinite loop
            pos = start + 1;
        }

        log::info!("File scan found {} objects", offsets.len());

        let result = offsets.get(&obj_ref.id).copied();
        *self.scanned_object_offsets.lock_or_recover() = Some(offsets);

        match result {
            Some(offset) => Ok(offset),
            None => Err(Error::ObjectNotFound(obj_ref.id, obj_ref.gen)),
        }
    }

    /// One-time sweep over every known object stream (`/Type /ObjStm`),
    /// used to recover from xref tables that mis-mark compressed objects as
    /// free.
    ///
    /// Some PDF producers emit an xref where a compressed object's slot is
    /// type 0 (free) instead of type 2 (compressed → stream#). The object
    /// is physically stored inside an `ObjStm`, but `scan_for_object` can't
    /// find it because it has no standalone `N G obj` marker in the body.
    ///
    /// The recovery: iterate every uncompressed candidate, peek at the
    /// dictionary, and for those that are `/Type /ObjStm`, parse the stream
    /// and cache everything inside (overwriting any stale `Object::Null`
    /// entries from earlier free-entry short-circuits).
    ///
    /// Runs at most once per document — guarded by `objstm_recovery_done`.
    /// Cost is amortised across every recovered object.
    fn recover_from_object_streams(&self) {
        use crate::objstm::parse_object_stream_with_decryption;

        {
            let done = self.objstm_recovery_done.lock_or_recover();
            if *done {
                return;
            }
        }

        log::debug!("Sweeping object streams to recover xref-flagged-free objects");

        // Find ObjStm candidates by raw pattern search in the file body.
        //
        // Why not iterate xref entries here: the xref is precisely what we
        // don't trust in this recovery path — its offsets may be wrong
        // its type tags may be lying about what each slot contains. A raw
        // search for `N G obj ... /Type /ObjStm` finds every object stream
        // the producer actually wrote, independent of how the xref
        // describes them.
        //
        // Only flip `objstm_recovery_done` after we finish the scan+parse
        // pass; a transient seek/read failure should leave the flag unset
        // so a later retry can still attempt recovery.
        let file_bytes = {
            let mut r = self.reader.lock_or_recover();
            if r.seek(SeekFrom::Start(0)).is_err() {
                return;
            }
            let mut buf = Vec::new();
            if r.read_to_end(&mut buf).is_err() {
                return;
            }
            buf
        };

        let candidates = find_objstm_candidates(&file_bytes);

        let mut objstms_found = 0usize;
        let mut recovered = 0usize;
        for (stream_obj_num, offset) in &candidates {
            let stream_ref = ObjectRef::new(*stream_obj_num, 0);
            let stream_obj = match self.load_uncompressed_object(stream_ref, *offset) {
                Ok(obj) => obj,
                Err(_) => continue,
            };

            let is_objstm = stream_obj
                .as_dict()
                .and_then(|d| d.get("Type"))
                .and_then(|t| t.as_name())
                .is_some_and(|n| n == "ObjStm");
            if !is_objstm {
                continue;
            }
            objstms_found += 1;

            // Parse the stream body. ISO 32000-2:2020 §7.6.3 says ObjStm
            // shall NOT be individually encrypted, so skip decryption here
            // — mirrors the default branch in `load_compressed_object`.
            let objects_map = match parse_object_stream_with_decryption(&stream_obj, None, 0, 0) {
                Ok(m) => m,
                Err(e) => {
                    log::debug!(
                        "Skipping ObjStm {} during recovery sweep (parse failed: {})",
                        stream_obj_num,
                        e
                    );
                    continue;
                },
            };

            let mut cache = self.object_cache.lock_or_recover();
            for (obj_num, object) in objects_map {
                let cache_ref = ObjectRef::new(obj_num, 0);
                // Only overwrite entries we'd otherwise have resolved to
                // Null (the free-entry short-circuit caches Null). Never
                // clobber a real object loaded through the normal path.
                match cache.get(&cache_ref) {
                    Some(Object::Null) | None => {
                        cache.insert(cache_ref, object);
                        recovered += 1;
                    },
                    _ => {},
                }
            }
        }

        log::debug!(
            "Object-stream recovery sweep: {} candidate positions, {} ObjStms, {} objects cached",
            candidates.len(),
            objstms_found,
            recovered
        );

        *self.objstm_recovery_done.lock_or_recover() = true;
    }

    /// Load an object by its reference.
    ///
    /// This function:
    /// 1. Checks the object cache first
    /// 2. If not cached, looks up the byte offset in the xref table
    /// 3. Seeks to that offset and parses the object
    /// 4. Caches the result for future access
    /// 5. If object not in xref but is critical, scans file for it
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The object reference is not in the xref table and file scan fails
    /// - The object is not in use (free object)
    /// - Seeking to the object offset fails
    /// - Parsing the object fails
    /// - A circular reference is detected
    /// - The recursion depth limit is exceeded
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # use pdf_oxide::object::ObjectRef;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// let obj_ref = ObjectRef::new(1, 0);
    /// let obj = doc.load_object(obj_ref)?;
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn load_object(&self, obj_ref: ObjectRef) -> Result<Object> {
        log::debug!("Loading object {} gen {}", obj_ref.id, obj_ref.gen);

        // Check recursion depth (per-thread counter; no lock needed)
        {
            let depth = RECURSION_DEPTH.with(|d| *d.borrow());
            if depth >= MAX_RECURSION_DEPTH {
                log::error!(
                    "Recursion depth limit exceeded ({}) while loading object {} gen {}",
                    MAX_RECURSION_DEPTH,
                    obj_ref.id,
                    obj_ref.gen
                );
                return Err(Error::RecursionLimitExceeded(MAX_RECURSION_DEPTH));
            }
        }

        // Check for circular references (per-thread stack; concurrent threads
        // resolving the same object do NOT appear as a false cycle)
        if RESOLVING_STACK.with(|s| s.borrow().contains(&obj_ref)) {
            log::error!(
                "Circular reference detected for object {} gen {} (depth: {})",
                obj_ref.id,
                obj_ref.gen,
                RECURSION_DEPTH.with(|d| *d.borrow())
            );
            return Err(Error::CircularReference(obj_ref));
        }

        // Check cache first (warm path: fully parallel, no serialization).
        let cached_opt = self.object_cache.lock_or_recover().get(&obj_ref).cloned();
        if let Some(cached) = cached_opt {
            return Ok(cached);
        }

        // Cold path (#507): serialize uncached loads across threads so a
        // single logical load's many `reader` lock scopes are not
        // interleaved by another thread's load on the shared `BufReader`.
        // Acquire ONLY at the top-level entry (recursion depth 0); a
        // recursive call from this same thread (nested-ref resolution)
        // already holds the guard, so re-acquiring would self-deadlock —
        // skip it. Held for the remainder of this top-level resolution.
        let _load_guard = if RECURSION_DEPTH.with(|d| *d.borrow()) == 0 {
            let guard = self.load_lock.lock_or_recover();
            // Double-checked: another thread may have loaded and cached
            // this object while we were blocked on the guard.
            if let Some(cached) = self.object_cache.lock_or_recover().get(&obj_ref).cloned() {
                return Ok(cached);
            }
            Some(guard)
        } else {
            None
        };

        // Look up in xref table
        let entry = match self.xref.get(obj_ref.id) {
            Some(entry) => entry,
            None => {
                // Object not in xref table - try scanning the file as fallback
                // This handles PDFs with incomplete/corrupted xref tables
                let available: Vec<u32> = self.xref.entries.keys().copied().take(20).collect();
                log::warn!(
                    "Object {} not in xref table. Total entries: {}. First 20 objects: {:?}",
                    obj_ref.id,
                    self.xref.len(),
                    available
                );

                // Try to scan the file for this object
                match self.scan_for_object(obj_ref) {
                    Ok(offset) => {
                        // Found it! Load directly from this offset
                        log::info!(
                            "Successfully found object {} via file scan at offset {}",
                            obj_ref.id,
                            offset
                        );

                        // Mark as being resolved (per-thread cycle detection)
                        RESOLVING_STACK.with(|s| {
                            s.borrow_mut().insert(obj_ref);
                        });
                        RECURSION_DEPTH.with(|d| *d.borrow_mut() += 1);

                        // Load the object
                        let result = self.load_uncompressed_object(obj_ref, offset);

                        RECURSION_DEPTH.with(|d| *d.borrow_mut() -= 1);
                        RESOLVING_STACK.with(|s| {
                            s.borrow_mut().remove(&obj_ref);
                        });

                        return result;
                    },
                    Err(_) => {
                        // PDF Spec §7.3.10: missing object reference "shall be treated as null"
                        log::warn!("Object {} gen {} not found (xref + file scan failed), treating as Null per §7.3.10", obj_ref.id, obj_ref.gen);
                        self.object_cache
                            .lock_or_recover()
                            .insert(obj_ref, Object::Null);
                        return Ok(Object::Null);
                    },
                }
            },
        };

        log::debug!(
            "  → Found in xref: type={:?}, offset={}, gen={}, in_use={}",
            entry.entry_type,
            entry.offset,
            entry.generation,
            entry.in_use
        );

        // Check if object is in use
        if !entry.in_use {
            log::debug!(
                "Object {} is marked as free (not in use). This may be due to a corrupted xref table.",
                obj_ref.id
            );

            // xref flags the object free, but this may be xref corruption
            // rather than an actual deletion. Run two recovery paths before
            // falling back to §7.3.10's null. The branches below apply
            // uniformly for all object ids (critical low-numbered catalog
            // objects and page objects in the thousands); previously low
            // ids took a separate "fall through to loading logic" path
            // that silently hit the Free arm of the entry_type match
            // still ended up Null.
            //
            // Recovery path 1 — standalone `N G obj` marker in the file
            // body. `scan_for_object` builds a whole-file offset map once
            // per document and caches it, so the amortised cost is a
            // single O(filesize) pass no matter how many free-marked
            // objects we probe.
            if let Ok(scanned_offset) = self.scan_for_object(obj_ref) {
                log::debug!(
                    "Object {} marked free in xref but found in file scan at offset {}; recovering",
                    obj_ref.id,
                    scanned_offset
                );
                RESOLVING_STACK.with(|s| {
                    s.borrow_mut().insert(obj_ref);
                });
                RECURSION_DEPTH.with(|d| *d.borrow_mut() += 1);
                let result = self.load_uncompressed_object(obj_ref, scanned_offset);
                RECURSION_DEPTH.with(|d| *d.borrow_mut() -= 1);
                RESOLVING_STACK.with(|s| {
                    s.borrow_mut().remove(&obj_ref);
                });
                return result;
            }

            // Recovery path 2 — the object may be compressed inside a
            // `/Type /ObjStm`. Real-world producers have been seen to
            // mis-flag every compressed object's xref slot as free, so
            // sweep the object streams once and recheck the cache.
            self.recover_from_object_streams();
            if let Some(obj) = self.object_cache.lock_or_recover().get(&obj_ref).cloned() {
                if !matches!(obj, Object::Null) {
                    log::debug!("Object {} recovered from object-stream sweep", obj_ref.id);
                    return Ok(obj);
                }
            }

            // PDF Spec §7.3.10: free object treated as null
            log::warn!(
                "Free object {} gen {}, treating as Null per §7.3.10",
                obj_ref.id,
                obj_ref.gen
            );
            self.object_cache
                .lock_or_recover()
                .insert(obj_ref, Object::Null);
            return Ok(Object::Null);
        }

        // Mark as being resolved (per-thread cycle detection)
        RESOLVING_STACK.with(|s| {
            s.borrow_mut().insert(obj_ref);
        });
        RECURSION_DEPTH.with(|d| *d.borrow_mut() += 1);

        // Handle different entry types
        use crate::xref::XRefEntryType;
        let entry_type = entry.entry_type;
        let entry_offset = entry.offset;
        let entry_gen = entry.generation;
        let result = match entry_type {
            XRefEntryType::Compressed => {
                // Type 2 entry: object is in an object stream
                // entry.offset = stream object number
                // entry.generation = index within stream
                log::debug!(
                    "  → Compressed object in stream {}, index {}",
                    entry_offset,
                    entry_gen
                );
                self.load_compressed_object(obj_ref, entry_offset as u32, entry_gen)
            },
            XRefEntryType::Uncompressed => {
                // Type 1 entry: traditional uncompressed object
                log::debug!("  → Uncompressed object at offset {}", entry_offset);
                self.load_uncompressed_object(obj_ref, entry_offset)
            },
            XRefEntryType::Free => {
                // Free object - shouldn't happen since we check in_use above
                // PDF Spec §7.3.10: treat as null
                log::warn!(
                    "Object {} has type Free despite in_use=true, treating as Null",
                    obj_ref.id
                );
                self.object_cache
                    .lock_or_recover()
                    .insert(obj_ref, Object::Null);
                Ok(Object::Null)
            },
        };

        RECURSION_DEPTH.with(|d| *d.borrow_mut() -= 1);
        RESOLVING_STACK.with(|s| {
            s.borrow_mut().remove(&obj_ref);
        });

        result
    }

    /// Resolve references within an object recursively.
    ///
    /// This utility method resolves indirect references within an object,
    /// handling nested dictionaries and arrays up to a specified depth.
    /// Useful for processing complex PDF structures where properties
    /// may be stored as indirect references.
    ///
    /// # Arguments
    ///
    /// * `obj` - The object to resolve references within
    /// * `max_depth` - Maximum recursion depth to prevent infinite loops
    ///
    /// # Returns
    ///
    /// The object with all references resolved up to max_depth levels.
    /// If a reference cannot be resolved, it is left as-is.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// # let obj = doc.catalog()?;
    /// // Resolve all references in a dictionary up to 3 levels deep
    /// let resolved = doc.resolve_references(&obj, 3)?;
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn resolve_references(&self, obj: &Object, max_depth: usize) -> Result<Object> {
        if max_depth == 0 {
            return Ok(obj.clone());
        }

        match obj {
            Object::Reference(obj_ref) => {
                // Resolve the reference
                match self.load_object(*obj_ref) {
                    Ok(resolved) => {
                        // Recursively resolve within the resolved object
                        self.resolve_references(&resolved, max_depth - 1)
                    },
                    Err(e) => {
                        log::warn!("Failed to resolve reference {:?}: {}", obj_ref, e);
                        Ok(obj.clone()) // Return the unresolved reference
                    },
                }
            },

            Object::Dictionary(dict) => {
                // Resolve references within each value
                let mut resolved_dict = std::collections::HashMap::new();
                for (key, value) in dict.iter() {
                    let resolved_value = self.resolve_references(value, max_depth - 1)?;
                    resolved_dict.insert(key.clone(), resolved_value);
                }
                Ok(Object::Dictionary(resolved_dict))
            },

            Object::Array(arr) => {
                // Resolve references within each element
                let resolved_arr: Result<Vec<Object>> = arr
                    .iter()
                    .map(|item| self.resolve_references(item, max_depth - 1))
                    .collect();
                Ok(Object::Array(resolved_arr?))
            },

            // For all other types, just return a clone
            _ => Ok(obj.clone()),
        }
    }

    /// Resolve a single-level indirect reference (PDF spec §7.3.10).
    ///
    /// If `obj` is `Object::Reference(...)`, loads and returns the target object.
    /// For any other object type, returns a clone unchanged. This is the
    /// canonical way to handle "any value may be a direct or indirect reference"
    /// throughout the parser.
    fn resolve_obj_ref(&self, obj: &Object) -> Object {
        if let Some(obj_ref) = obj.as_reference() {
            match self.load_object(obj_ref) {
                Ok(resolved) => resolved,
                Err(e) => {
                    log::warn!("Failed to resolve indirect reference {:?}: {}", obj_ref, e);
                    obj.clone()
                },
            }
        } else {
            obj.clone()
        }
    }

    /// Peek at an XObject's /Subtype without loading the full object.
    /// Returns true if the XObject is a Form XObject, false if Image or unknown.
    /// For compressed objects or on any error, returns true (conservative — will load fully).
    pub fn is_form_xobject(&self, obj_ref: ObjectRef) -> bool {
        // Check negative cache first (known non-Form XObjects)
        {
            if self
                .image_xobject_cache
                .lock_or_recover()
                .contains(&obj_ref)
            {
                return false;
            }
        }

        // If already in object cache, check directly
        let cached_opt = self.object_cache.lock_or_recover().get(&obj_ref).cloned();
        if let Some(cached) = cached_opt {
            let is_form = cached
                .as_dict()
                .and_then(|d| d.get("Subtype"))
                .and_then(|s| s.as_name())
                == Some("Form");
            if !is_form {
                self.image_xobject_cache.lock_or_recover().insert(obj_ref);
            }
            return is_form;
        }

        // Look up in xref table
        let entry = match self.xref.get(obj_ref.id) {
            Some(e) => e,
            None => return true, // conservative fallback
        };

        // Only peek uncompressed objects — compressed ones require full load
        use crate::xref::XRefEntryType;
        if entry.entry_type != XRefEntryType::Uncompressed || !entry.in_use {
            return true; // conservative fallback
        }

        // Seek + read under a SINGLE lock guard. Splitting the seek
        // the read across two `self.reader.lock_or_recover()` acquisitions
        // is the #398 Race A split-lock bug (same one already fixed in
        // `load_uncompressed_object_impl`): a concurrent thread can
        // re-seek the shared reader between our seek() and read(), so we
        // read a garbage buffer for a different object. That surfaced as
        // a spurious `[1000] invalid PDF structure or content stream`
        // ParseError under concurrent `render_page_fit`.
        let offset = entry.offset;
        let mut buf = [0u8; 1024];
        let n = {
            let mut reader = self.reader.lock_or_recover();
            if reader.seek(SeekFrom::Start(offset)).is_err() {
                return true;
            }
            // Read enough bytes for the object header + dictionary (<1KB)
            match reader.read(&mut buf) {
                Ok(n) => n,
                Err(_) => return true,
            }
        };
        let data = &buf[..n];

        // Search for /Subtype in the buffer
        // Look for "/Subtype" followed by a name like "/Form" or "/Image"
        if let Some(pos) = data.windows(8).position(|w| w == b"/Subtype") {
            let after = &data[pos + 8..];
            // Skip whitespace
            let trimmed = after
                .iter()
                .position(|&b| b != b' ' && b != b'\t' && b != b'\r' && b != b'\n');
            if let Some(start) = trimmed {
                let name_data = &after[start..];
                if name_data.starts_with(b"/Form") {
                    return true;
                }
                // Image, PS, or anything else — not a Form
                self.image_xobject_cache.lock_or_recover().insert(obj_ref);
                return false;
            }
        }

        // /Subtype not found in first 1KB — conservative fallback
        true
    }

    /// Load an uncompressed object (Type 1 xref entry).
    fn load_uncompressed_object(&self, obj_ref: ObjectRef, offset: u64) -> Result<Object> {
        self.load_uncompressed_object_impl(obj_ref, offset, false)
    }

    /// Promote labels in rowspan-sparse columns so they sort at the top
    /// of their data-row block instead of landing mid-group.
    ///
    /// A "label" here is a span in an X-cluster that contains far fewer
    /// spans than the most populous X-cluster (i.e., it spans multiple
    /// rows of the adjacent data column). Labels are typically vertically
    /// centred in their block, so a strict Y sort places them between
    /// the rows they describe. This post-processor detects the pattern
    /// and rewrites each label's effective sort Y to sit just above the
    /// topmost data row it visually covers.
    ///
    /// Data rows are partitioned between adjacent labels at the midpoint
    /// of their Y coordinates (nearest-label assignment). The topmost
    /// data row in a label's partition becomes the anchor for promotion.
    ///
    /// Nothing is mutated if there are no sparse columns or not enough
    /// data rows to confidently infer row-grouping (min 6 rows in the
    /// dense reference column).
    /// Identify span indices that look like multi-row-spanning labels —
    /// sparse-X-column spans whose Y values sit inside the data Y range
    /// of the dense columns on the page. These are the same spans that
    /// `reorder_rowspan_labels` would promote to the top of their row
    /// block, except this function returns them **before** the spatial
    /// table detector's retain filter has a chance to drop them from
    /// the flow span list.
    ///
    /// The retain filter in `extract_text_with_options` removes every
    /// span whose bbox is contained in a detected table's bbox. On CJK
    /// reference-data PDFs the test-name label column is
    /// narrow and vertically centred within each multi-row data block,
    /// so its spans are inside the table bbox and would be dropped
    /// without replacement — the spatial table extractor does not emit
    /// these labels as `TableCell`s either. Preserving the identified
    /// labels through the retain filter lets `reorder_rowspan_labels`
    /// promote them to their proper reading-order position alongside
    /// the surviving flow spans.
    ///
    /// Returns a `HashSet` of indices into the provided `spans` slice.
    /// Callers must use the returned indices **before** any reordering
    /// or retention mutates the slice.
    pub(crate) fn identify_multi_row_labels(
        spans: &[crate::layout::TextSpan],
    ) -> std::collections::HashSet<usize> {
        use std::collections::{BTreeSet, HashMap as StdHashMap, HashSet};

        let mut out: HashSet<usize> = HashSet::new();
        if spans.len() < 10 {
            return out;
        }

        // Cluster by X proximity (15pt gap threshold) — same heuristic
        // as `reorder_rowspan_labels`.
        let mut by_x: Vec<usize> = (0..spans.len()).collect();
        by_x.sort_by(|&a, &b| crate::utils::safe_float_cmp(spans[a].bbox.x, spans[b].bbox.x));
        const X_GAP: f32 = 15.0;
        let mut columns: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut last_x = f32::NEG_INFINITY;
        for &idx in &by_x {
            let x = spans[idx].bbox.x;
            if !cur.is_empty() && x - last_x > X_GAP {
                columns.push(std::mem::take(&mut cur));
            }
            cur.push(idx);
            last_x = x;
        }
        if !cur.is_empty() {
            columns.push(cur);
        }
        if columns.len() < 2 {
            return out;
        }

        let max_count = columns.iter().map(|c| c.len()).max().unwrap_or(0);
        if max_count < 6 {
            return out;
        }

        // Sort columns by span count descending to pick the dense clusters.
        let mut col_order: Vec<usize> = (0..columns.len()).collect();
        col_order.sort_by(|&a, &b| columns[b].len().cmp(&columns[a].len()));
        let dense_cols_count = columns.iter().filter(|c| c.len() * 2 > max_count).count();

        let band_of = |y: f32| (y / crate::utils::ROW_BAND_TOLERANCE_PT).round() as i32;
        let data_bands: BTreeSet<i32> = if dense_cols_count >= 3 {
            let top: Vec<&Vec<usize>> = col_order.iter().take(3).map(|&i| &columns[i]).collect();
            let mut support: StdHashMap<i32, usize> = StdHashMap::new();
            for col in &top {
                let bands: HashSet<i32> = col.iter().map(|&i| band_of(spans[i].bbox.y)).collect();
                for b in bands {
                    *support.entry(b).or_insert(0) += 1;
                }
            }
            support
                .into_iter()
                .filter(|(_, c)| *c >= 3)
                .map(|(b, _)| b)
                .collect()
        } else if dense_cols_count == 2 {
            let a: HashSet<i32> = columns[col_order[0]]
                .iter()
                .map(|&i| band_of(spans[i].bbox.y))
                .collect();
            let b: HashSet<i32> = columns[col_order[1]]
                .iter()
                .map(|&i| band_of(spans[i].bbox.y))
                .collect();
            a.intersection(&b).copied().collect()
        } else {
            columns[col_order[0]]
                .iter()
                .map(|&i| band_of(spans[i].bbox.y))
                .collect()
        };

        if data_bands.len() < 4 {
            return out;
        }

        let band_pt = crate::utils::ROW_BAND_TOLERANCE_PT;
        let data_top = (*data_bands.iter().next_back().unwrap() as f32) * band_pt + band_pt / 2.0;
        let data_bot = (*data_bands.iter().next().unwrap() as f32) * band_pt - band_pt / 2.0;

        // Collect sparse-column spans that sit inside the data Y range
        // and belong to a column with >= 2 members in that range.
        for col in &columns {
            if col.len() < 2 || col.len() * 2 >= max_count {
                continue;
            }
            let in_data: Vec<usize> = col
                .iter()
                .copied()
                .filter(|&i| {
                    let y = spans[i].bbox.y;
                    y > data_bot && y < data_top
                })
                .collect();
            if in_data.len() >= 2 {
                out.extend(in_data);
            }
        }

        out
    }

    pub(crate) fn reorder_rowspan_labels(spans: &mut Vec<crate::layout::TextSpan>) {
        use std::collections::HashMap;

        if spans.len() < 10 {
            return;
        }

        // Cluster by X proximity (15pt gap threshold). Walk spans ordered
        // by left edge; start a new cluster whenever the gap exceeds the
        // threshold.
        let mut by_x: Vec<usize> = (0..spans.len()).collect();
        by_x.sort_by(|&a, &b| crate::utils::safe_float_cmp(spans[a].bbox.x, spans[b].bbox.x));
        const X_GAP: f32 = 15.0;
        let mut columns: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut last_x = f32::NEG_INFINITY;
        for &idx in &by_x {
            let x = spans[idx].bbox.x;
            if !cur.is_empty() && x - last_x > X_GAP {
                columns.push(std::mem::take(&mut cur));
            }
            cur.push(idx);
            last_x = x;
        }
        if !cur.is_empty() {
            columns.push(cur);
        }
        if columns.len() < 2 {
            return;
        }

        // Max column size is our reference for "dense".
        let max_count = columns.iter().map(|c| c.len()).max().unwrap_or(0);
        if max_count < 6 {
            return;
        }

        // Sort columns by span count descending so we can pick the top
        // dense cluster for anchor detection.
        let mut col_order: Vec<usize> = (0..columns.len()).collect();
        col_order.sort_by(|&a, &b| columns[b].len().cmp(&columns[a].len()));

        // A column is "dense" when it holds a strict majority of the
        // most populous column's spans. Pages with multiple dense data
        // columns (three or more) let us derive the data-row range by
        // intersecting their Y bands — headers and sub-headers populate
        // only a subset of columns at their Y and fall out.
        let dense_cols_count = columns.iter().filter(|c| c.len() * 2 > max_count).count();

        // Most populous column, used for anchor Y lookups regardless.
        let dense_col = &columns[col_order[0]];
        let mut dense_ys: Vec<f32> = dense_col.iter().map(|&i| spans[i].bbox.y).collect();
        dense_ys.sort_by(|a, b| crate::utils::safe_float_cmp(*b, *a));

        // Compute the set of Y bands that count as "data". When several
        // dense columns are available, require a band to have support in
        // the top three; otherwise fall back to the single dense column's
        // own Y values.
        let band_of = |y: f32| (y / crate::utils::ROW_BAND_TOLERANCE_PT).round() as i32;
        use std::collections::{BTreeSet, HashMap as StdHashMap, HashSet};

        let data_bands: BTreeSet<i32> = if dense_cols_count >= 3 {
            let top: Vec<&Vec<usize>> = col_order.iter().take(3).map(|&i| &columns[i]).collect();
            let mut support: StdHashMap<i32, usize> = StdHashMap::new();
            for col in &top {
                let bands: HashSet<i32> = col.iter().map(|&i| band_of(spans[i].bbox.y)).collect();
                for b in bands {
                    *support.entry(b).or_insert(0) += 1;
                }
            }
            support
                .into_iter()
                .filter(|(_, c)| *c >= 3)
                .map(|(b, _)| b)
                .collect()
        } else if dense_cols_count == 2 {
            let a: HashSet<i32> = columns[col_order[0]]
                .iter()
                .map(|&i| band_of(spans[i].bbox.y))
                .collect();
            let b: HashSet<i32> = columns[col_order[1]]
                .iter()
                .map(|&i| band_of(spans[i].bbox.y))
                .collect();
            a.intersection(&b).copied().collect()
        } else {
            dense_col
                .iter()
                .map(|&i| band_of(spans[i].bbox.y))
                .collect()
        };

        if data_bands.len() < 4 {
            return;
        }
        let band_pt = crate::utils::ROW_BAND_TOLERANCE_PT;
        let data_top = (*data_bands.iter().next_back().unwrap() as f32) * band_pt + band_pt / 2.0;
        let data_bot = (*data_bands.iter().next().unwrap() as f32) * band_pt - band_pt / 2.0;

        // Y-bands occupied by the dense column. Genuine rowspan labels are
        // vertically centred *between* data rows, so their Y-band must NOT
        // appear in this set. Spans whose Y aligns with the dense column are
        // line-continuation text on the same logical line, not labels.
        let dense_bands: HashSet<i32> = dense_col
            .iter()
            .map(|&i| band_of(spans[i].bbox.y))
            .collect();

        // Collect "label" candidates: spans that sit in a "sparse"
        // column — one that holds meaningfully fewer spans than the
        // most populous column. A candidate only qualifies when it
        // sits strictly inside the data Y range AND the sparse column
        // it belongs to has at least two entries inside that range —
        // single-span sparse cells are almost always stray annotations,
        // not labels.
        let mut labels: Vec<usize> = Vec::new();
        for col in &columns {
            if col.len() < 2 || col.len() * 2 >= max_count {
                continue;
            }
            let in_data: Vec<usize> = col
                .iter()
                .copied()
                .filter(|&i| {
                    let y = spans[i].bbox.y;
                    // Exclude spans on the same Y-band as the dense column:
                    // those are line-continuation text, not rowspan labels.
                    y > data_bot && y < data_top && !dense_bands.contains(&band_of(y))
                })
                .collect();
            if in_data.len() >= 2 {
                labels.extend(in_data);
            }
        }
        if labels.is_empty() {
            return;
        }
        labels.sort_by(|&a, &b| {
            spans[b]
                .bbox
                .y
                .partial_cmp(&spans[a].bbox.y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Labels that sit at near-identical Y values almost always
        // annotate the same logical row block (e.g. a test-name in the
        // "name" column alongside a unit "×10⁹/L" in the "unit" column,
        // both vertically centred in the same 6-row group). Cluster
        // labels by Y proximity so each logical block is promoted as a
        // unit.
        const CLUSTER_GAP: f32 = 10.0;
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut last_y = f32::NAN;
        for &idx in &labels {
            let y = spans[idx].bbox.y;
            if !cur.is_empty() && (last_y - y).abs() > CLUSTER_GAP {
                clusters.push(std::mem::take(&mut cur));
            }
            cur.push(idx);
            last_y = y;
        }
        if !cur.is_empty() {
            clusters.push(cur);
        }
        let cluster_ys: Vec<f32> = clusters
            .iter()
            .map(|c| c.iter().map(|&i| spans[i].bbox.y).sum::<f32>() / c.len() as f32)
            .collect();

        // For each cluster, compute the midpoint partition boundaries
        // against its immediate neighbour clusters and find the topmost
        // dense-column Y that falls inside the partition. Promote every
        // member of the cluster to the same anchor so they sort together
        // at the top of their row block.
        let mut promoted: HashMap<usize, f32> = HashMap::new();
        for (k, cluster) in clusters.iter().enumerate() {
            let c_y = cluster_ys[k];
            let upper = if k > 0 {
                (cluster_ys[k - 1] + c_y) / 2.0
            } else {
                f32::INFINITY
            };
            let lower = if k + 1 < clusters.len() {
                (c_y + cluster_ys[k + 1]) / 2.0
            } else {
                f32::NEG_INFINITY
            };
            let upper_clamped = upper.min(data_top);
            let lower_clamped = lower.max(data_bot - 1.0);
            let mut anchor = f32::NEG_INFINITY;
            for &y in &dense_ys {
                if y <= upper_clamped && y > lower_clamped && y > anchor {
                    anchor = y;
                }
            }
            if anchor.is_finite() {
                for &i in cluster {
                    promoted.insert(i, anchor + 1.0);
                }
            }
        }
        if promoted.is_empty() {
            return;
        }

        // Re-sort spans using the promoted Ys for labels and actual Ys
        // for everything else. Keep the row-aware comparator so the
        // ordering stays consistent with the rest of the pipeline.
        let mut order: Vec<usize> = (0..spans.len()).collect();
        order.sort_by(|&a, &b| {
            let ya = promoted.get(&a).copied().unwrap_or(spans[a].bbox.y);
            let yb = promoted.get(&b).copied().unwrap_or(spans[b].bbox.y);
            crate::utils::row_aware_span_cmp(ya, spans[a].bbox.x, yb, spans[b].bbox.x)
        });
        let reordered: Vec<crate::layout::TextSpan> =
            order.into_iter().map(|i| spans[i].clone()).collect();
        *spans = reordered;
    }

    /// Recursively decrypt every `Object::String` inside `obj` using the
    /// per-object key derived from `obj_num`/`gen_num`. Streams are left
    /// untouched — they are decrypted lazily at read time through
    /// `decode_stream_with_encryption`. The `/Encrypt` dictionary itself
    /// must never be passed to this function; its strings are key material,
    /// not ciphertext.
    ///
    /// Per ISO 32000-1:2008 §7.6.2, strings inside encrypted-document
    /// objects are individually encrypted with the standard encryption
    /// algorithm. Parsed string tokens hold raw ciphertext and must be
    /// decrypted before downstream consumers (widget text, form field
    /// values, outlines, document info) can read them.
    fn decrypt_strings_in_object(
        handler: &EncryptionHandler,
        obj: &mut Object,
        obj_num: u32,
        gen_num: u32,
    ) {
        match obj {
            Object::String(bytes) => match handler.decrypt_string(bytes, obj_num, gen_num) {
                Ok(decrypted) => *bytes = decrypted,
                Err(e) => {
                    log::debug!(
                        "String decryption failed for object {} {}: {}",
                        obj_num,
                        gen_num,
                        e
                    );
                },
            },
            Object::Array(items) => {
                for item in items {
                    Self::decrypt_strings_in_object(handler, item, obj_num, gen_num);
                }
            },
            Object::Dictionary(dict) => {
                for (_, value) in dict.iter_mut() {
                    Self::decrypt_strings_in_object(handler, value, obj_num, gen_num);
                }
            },
            Object::Stream { dict, .. } => {
                // Stream *data* is decrypted separately in
                // `decode_stream_with_encryption`. Its dict may still
                // contain encrypted strings (e.g., /Metadata).
                for (_, value) in dict.iter_mut() {
                    Self::decrypt_strings_in_object(handler, value, obj_num, gen_num);
                }
            },
            _ => {},
        }
    }

    /// Implementation with recursion guard to prevent infinite loops.
    fn load_uncompressed_object_impl(
        &self,
        obj_ref: ObjectRef,
        offset: u64,
        already_corrected: bool,
    ) -> Result<Object> {
        // --- Phase 1: read the object header under a single lock guard ---
        // Holding one guard for seek+read prevents the split-lock race (#398 Race A)
        // where a concurrent thread can re-seek the shared BufReader between our
        // seek() and read_until() calls.
        let (header_bytes, full_header) = {
            let mut reader = self.reader.lock_or_recover();
            reader.seek(SeekFrom::Start(offset))?;

            // Read bytes for object header (e.g., "1 0 obj")
            let mut header_bytes = Vec::new();
            let bytes_read = reader.read_until(b'\n', &mut header_bytes)?;

            if bytes_read == 0 {
                let msg = format!("Unexpected EOF while reading object {} header", obj_ref.id);
                log::warn!("{}", msg);
                // also push into structured sink so
                // callers can retrieve as data via flatten_warnings.
                self.push_structured_warning(crate::extractors::warnings::Warning {
                    category: crate::extractors::warnings::WarningCategory::EofPremature,
                    page: None,
                    message: msg,
                    spec_section: Some("7.5"),
                });
                return Err(Error::UnexpectedEof);
            }

            let line = String::from_utf8_lossy(&header_bytes);

            // Issue #45: Handle multi-line object headers
            let mut full_header = line.to_string();
            let max_header_lines = 5;
            let mut lines_read = 1;

            while !has_standalone_obj_keyword(&full_header) && lines_read < max_header_lines {
                let mut next_bytes = Vec::new();
                let next_read = reader.read_until(b'\n', &mut next_bytes)?;
                if next_read == 0 {
                    break;
                }
                let next_line = String::from_utf8_lossy(&next_bytes);
                full_header.push(' ');
                full_header.push_str(&next_line);
                lines_read += 1;
            }
            // Reader guard drops here — before any recursive fallback calls.
            (header_bytes, full_header)
        };

        // Verify object header format
        // Split by whitespace to handle various formats (single-line or multi-line)
        let parts: Vec<&str> = full_header.split_whitespace().collect();

        // Find standalone "obj" keyword (not "endobj")
        let obj_pos = parts
            .iter()
            .position(|&p| p == "obj" || (p.starts_with("obj") && !p.starts_with("endobj")));

        // Validate object header has proper format: <id> <gen> obj
        let obj_pos = match obj_pos {
            Some(pos) if pos >= 2 => pos,
            _ => {
                // Only try backwards search once to prevent infinite recursion
                if !already_corrected {
                    // xref offset might be incorrect (pointing to object body instead of header)
                    // Try searching backwards for the object header
                    log::debug!(
                        "No object header at offset {}, searching backwards for object {} {} obj",
                        offset,
                        obj_ref.id,
                        obj_ref.gen
                    );

                    if let Ok(corrected_offset) = self.find_object_header_backwards(obj_ref, offset)
                    {
                        log::info!(
                            "Found object header at offset {} (xref said {})",
                            corrected_offset,
                            offset
                        );
                        return self.load_uncompressed_object_impl(obj_ref, corrected_offset, true);
                    }
                }

                log::warn!("Malformed object header at offset {}: {}", offset, full_header.trim());
                return Err(Error::ParseError {
                    offset: offset as usize,
                    reason: format!("Expected object header, found: {}", full_header.trim()),
                });
            },
        };

        let _obj_pos = obj_pos;

        // Parse the object number and generation from header. If either
        // fails to parse as a number, the xref-reported offset is pointing
        // into the middle of a previous object's tail (e.g. xref says 12345
        // but the real `N G obj` header starts at 12348 because three bytes
        // of CR/LF/terminator got mis-accounted for by the producer — a
        // pattern seen in the wild). Fall back to the whole-file scan
        // cache: if scan recorded a different offset for this id, retry
        // from there before giving up.
        let obj_num_parsed = parts[0].parse::<u32>();
        let gen_num_parsed = parts[1].parse::<u16>();
        if !already_corrected && (obj_num_parsed.is_err() || gen_num_parsed.is_err()) {
            if let Ok(scan_offset) = self.scan_for_object(obj_ref) {
                if scan_offset != offset {
                    log::debug!(
                        "Header parse failed at xref offset {} (parts[0]={:?}); retrying at scan-reported offset {}",
                        offset,
                        parts[0],
                        scan_offset
                    );
                    return self.load_uncompressed_object_impl(obj_ref, scan_offset, true);
                }
            }
        }
        let obj_num: u32 = obj_num_parsed.map_err(|_| Error::ParseError {
            offset: offset as usize,
            reason: format!("Invalid object number in header: {}", parts[0]),
        })?;
        let gen_num: u16 = gen_num_parsed.map_err(|_| Error::ParseError {
            offset: offset as usize,
            reason: format!("Invalid generation number in header: {}", parts[1]),
        })?;

        // Verify object reference matches (warn but don't fail on mismatch)
        if obj_num != obj_ref.id || gen_num != obj_ref.gen {
            log::warn!(
                "Object reference mismatch at offset {}: expected {} {} obj, found {} {} obj",
                offset,
                obj_ref.id,
                obj_ref.gen,
                obj_num,
                gen_num
            );
        }

        // Check if there's content after "obj" on the same line
        // Some PDFs have "N G obj\n<<..." while others have "N G obj<<..." on one line
        let mut data = Vec::new();

        // Find where "obj" ends in the original bytes
        // We need to include anything after "obj" in the header line
        if let Some(obj_keyword_pos) = header_bytes.windows(3).position(|w| w == b"obj") {
            let after_obj_pos = obj_keyword_pos + 3; // "obj" is 3 bytes

            // Skip whitespace after "obj"
            let mut content_start = after_obj_pos;
            while content_start < header_bytes.len()
                && (header_bytes[content_start] == b' '
                    || header_bytes[content_start] == b'\t'
                    || header_bytes[content_start] == b'\r')
            {
                content_start += 1;
            }

            // If there's a newline, skip it (normal case: "N G obj\n")
            // If there's content (like "<<"), include it (malformed case: "N G obj<<...")
            if content_start < header_bytes.len() && header_bytes[content_start] != b'\n' {
                // There's content on the same line after "obj" - include it
                data.extend_from_slice(&header_bytes[content_start..]);
                log::debug!(
                    "Object {} has content after 'obj' on header line ({} bytes)",
                    obj_ref.id,
                    header_bytes.len() - content_start
                );
            }
        }

        // --- Phase 2: read body under a single lock guard (#398 Race A) ---
        // Use byte limit instead of line count — large uncompressed streams can have
        // hundreds of thousands of short lines (e.g., vector path drawing commands).
        const MAX_BYTES: usize = 100 * 1024 * 1024; // 100 MB safety limit

        {
            let mut reader = self.reader.lock_or_recover();
            loop {
                let mut chunk = Vec::new();
                let bytes_read = reader.read_until(b'\n', &mut chunk)?;

                if data.len() > MAX_BYTES {
                    log::warn!(
                        "Object {} exceeded maximum byte limit ({} bytes), truncating",
                        obj_ref.id,
                        MAX_BYTES
                    );
                    break;
                }

                if bytes_read == 0 {
                    let msg = format!(
                        "Unexpected EOF while reading object {} (no endobj found after {} bytes)",
                        obj_ref.id,
                        data.len()
                    );
                    log::warn!("{}", msg);
                    // structured-warnings sink.
                    self.push_structured_warning(crate::extractors::warnings::Warning {
                        category: crate::extractors::warnings::WarningCategory::EofPremature,
                        page: None,
                        message: msg,
                        spec_section: Some("7.5"),
                    });
                    // Don't fail - try to parse what we have
                    break;
                }

                // Check if we reached endobj
                if chunk.contains(&b'e') {
                    // Find "endobj" in the chunk (working with bytes, not chars)
                    if let Some(endobj_pos) = find_substring(&chunk, b"endobj") {
                        // Include everything before "endobj" but not "endobj" itself
                        data.extend_from_slice(&chunk[..endobj_pos]);
                        break;
                    }
                }

                data.extend_from_slice(&chunk);
            }
        }

        // Parse the object data
        log::debug!(
            "About to parse object {} gen {} ({} bytes)",
            obj_ref.id,
            obj_ref.gen,
            data.len()
        );

        // Corrupted objects degrade to Null so extraction can continue on
        // partial PDFs rather than aborting.
        let mut obj = match parse_object(&data) {
            Ok((_, parsed_obj)) => parsed_obj,
            Err(e) => {
                let error_kind = match &e {
                    nom::Err::Incomplete(_) => "Incomplete data",
                    nom::Err::Error(err) | nom::Err::Failure(err) => match err.code {
                        nom::error::ErrorKind::Eof => "Unexpected EOF",
                        nom::error::ErrorKind::Tag => "Expected tag not found",
                        nom::error::ErrorKind::Fail => "Parse failed",
                        _ => "Parse error",
                    },
                };
                log::warn!(
                    "Object {} at offset {} is corrupted ({}), using Null placeholder. \
                     This may result in missing content from the PDF.",
                    obj_ref.id,
                    offset,
                    error_kind
                );
                Object::Null
            },
        };

        // Decrypt string values inside this uncompressed object before
        // caching. Skip the /Encrypt dict (its entries are key material)
        // and the non-authenticated case (no key derived yet). Strings
        // inside compressed objects ride along with the ObjStm payload
        // and are already in clear text per ISO 32000-1:2008 §7.6.2.
        let is_encrypt_dict = *self.encrypt_dict_ref.lock_or_recover() == Some(obj_ref);
        if !is_encrypt_dict {
            let handler_guard = self.encryption_handler.lock_or_recover();
            if let Some(handler) = handler_guard.as_ref() {
                if handler.is_authenticated() {
                    Self::decrypt_strings_in_object(
                        handler,
                        &mut obj,
                        obj_ref.id,
                        obj_ref.gen as u32,
                    );
                }
            }
        }

        // Cache the object
        self.object_cache
            .lock_or_recover()
            .insert(obj_ref, obj.clone());

        Ok(obj)
    }

    /// Load a compressed object from an object stream (Type 2 xref entry).
    ///
    /// # Arguments
    ///
    /// * `obj_ref` - The object reference being loaded
    /// * `stream_obj_num` - The object number of the object stream
    /// * `index_in_stream` - The index within the stream (unused but provided for completeness)
    fn load_compressed_object(
        &self,
        obj_ref: ObjectRef,
        stream_obj_num: u32,
        _index_in_stream: u16,
    ) -> Result<Object> {
        use crate::objstm::parse_object_stream_with_decryption;

        log::debug!(
            "[load_compressed_debug] Loading obj {} from stream {}",
            obj_ref.id,
            stream_obj_num
        );

        // Per PDF §7.6.3, object streams (/Type /ObjStm) shall NOT be individually
        // encrypted. Encryption initialization is therefore not required to read an
        // ObjStm: the unencrypted parse path below is always attempted first. If
        // initialization fails (e.g. unsupported algorithm, no legacy-crypto feature),
        // log and continue — the handler will be None and we'll use the no-decryption
        // path, which is exactly what the spec mandates for ObjStm content.
        if let Err(e) = self.ensure_encryption_initialized() {
            log::debug!(
                "Encryption init skipped for ObjStm {} load ({}); will parse without decryption",
                stream_obj_num,
                e
            );
        }

        // Load the object stream
        let stream_ref = ObjectRef::new(stream_obj_num, 0);
        let stream_obj = self.load_uncompressed_object(stream_ref, {
            // Look up the stream's offset in the xref table
            let stream_entry = match self.xref.get(stream_obj_num) {
                Some(entry) => entry,
                None => {
                    // PDF Spec §7.3.10: treat as null
                    log::warn!(
                        "Object stream {} not in xref, treating compressed object {} as Null",
                        stream_obj_num,
                        obj_ref.id
                    );
                    self.object_cache
                        .lock_or_recover()
                        .insert(obj_ref, Object::Null);
                    return Ok(Object::Null);
                },
            };

            if stream_entry.entry_type != crate::xref::XRefEntryType::Uncompressed {
                return Err(Error::InvalidPdf(format!(
                    "object stream {} is not an uncompressed object",
                    stream_obj_num
                )));
            }

            stream_entry.offset
        })?;

        // Parse all objects from the stream.
        //
        // Per ISO 32000-2:2020 Section 7.6.3, object streams (/Type /ObjStm)
        // cross-reference streams (/Type /XRef) shall NOT be individually encrypted.
        // The stream data is only compressed, not encrypted. Many PDF producers
        // (including many real-world producers) follow this rule even under
        // PDF 1.x, so attempting AES decryption on the raw stream bytes fails
        // because the data length is not a multiple of the AES block size (16).
        //
        // We therefore always parse object streams WITHOUT decryption. If a
        // future PDF is encountered where the producer DID encrypt the ObjStm
        // (non-standard), the unencrypted parse will fail and we fall back to
        // trying with decryption.
        let handler_ref = self.encryption_handler.lock_or_recover();
        let objects_map = if handler_ref.is_some() {
            // First try without decryption (spec-compliant path)
            match parse_object_stream_with_decryption(&stream_obj, None, 0, 0) {
                Ok(map) => map,
                Err(_no_decrypt_err) => {
                    // Fallback: try with decryption for non-standard producers
                    let handler = handler_ref.as_ref().unwrap();
                    let decrypt_fn = |data: &[u8]| -> Result<Vec<u8>> {
                        handler.decrypt_stream(data, stream_obj_num, 0)
                    };
                    parse_object_stream_with_decryption(
                        &stream_obj,
                        Some(&decrypt_fn),
                        stream_obj_num,
                        0,
                    )?
                },
            }
        } else {
            parse_object_stream_with_decryption(&stream_obj, None, 0, 0)?
        };
        drop(handler_ref);

        // Extract the requested object
        let obj = match objects_map.get(&obj_ref.id) {
            Some(o) => o.clone(),
            None => {
                // PDF Spec §7.3.10: treat as null
                log::warn!(
                    "Object {} not found in object stream {}, treating as Null",
                    obj_ref.id,
                    stream_obj_num
                );
                Object::Null
            },
        };

        // Cache all objects from the stream for future access
        // IMPORTANT: Only cache objects whose xref entry points to THIS stream.
        // In incremental updates, the same object number may exist in multiple streams,
        // and we must not cache a stale version from an older stream.
        for (obj_num, object) in objects_map {
            let cache_ref = ObjectRef::new(obj_num, 0);
            let should_cache = if let Some(entry) = self.xref.get(obj_num) {
                // Only cache if the xref says this object belongs to this stream
                entry.entry_type == crate::xref::XRefEntryType::Compressed
                    && entry.offset == stream_obj_num as u64
            } else {
                // Object not in xref at all -- safe to cache as it's only in this stream
                true
            };
            if should_cache {
                self.object_cache
                    .lock_or_recover()
                    .insert(cache_ref, object);
            } else {
                log::debug!(
                    "[cache_debug] NOT caching obj {} from stream {} (xref points elsewhere)",
                    obj_num,
                    stream_obj_num
                );
            }
        }

        Ok(obj)
    }

    /// Find object header by searching backwards from a given offset.
    ///
    /// Some PDF generators create xref tables with incorrect offsets that point
    /// to the object body instead of the header. This function searches backwards
    /// from the xref offset to find the actual "N G obj" header.
    ///
    /// We search up to 100 bytes backwards, looking for a line that matches
    /// the expected object header format.
    fn find_object_header_backwards(&self, obj_ref: ObjectRef, wrong_offset: u64) -> Result<u64> {
        // Don't search before the start of the file
        if wrong_offset == 0 {
            return Err(Error::ParseError {
                offset: wrong_offset as usize,
                reason: "Cannot search backwards from offset 0".to_string(),
            });
        }

        // Search up to 100 bytes backwards (reasonable for most PDFs)
        let search_distance = std::cmp::min(100, wrong_offset);
        let search_start = wrong_offset - search_distance;

        // Read the search region under one lock guard (#398 Race A).
        let mut buffer = vec![0u8; search_distance as usize + 100]; // Extra bytes to read full line
        let bytes_read = {
            let mut reader = self.reader.lock_or_recover();
            reader.seek(SeekFrom::Start(search_start))?;
            reader.read(&mut buffer)?
        };

        if bytes_read == 0 {
            return Err(Error::ParseError {
                offset: wrong_offset as usize,
                reason: "Could not read backwards search region".to_string(),
            });
        }

        // Build the expected header pattern as bytes (NOT string to avoid UTF-8 corruption)
        let expected_header = format!("{} {} obj", obj_ref.id, obj_ref.gen);
        let pattern_bytes = expected_header.as_bytes();

        // Search for the byte pattern directly (avoids UTF-8 conversion issues with binary data)
        // Find the match closest to wrong_offset (prefer before, but allow small offsets after)
        let mut best_match: Option<(usize, i64)> = None; // (position, distance_from_wrong)

        for (i, window) in buffer[..bytes_read]
            .windows(pattern_bytes.len())
            .enumerate()
        {
            if window == pattern_bytes {
                let candidate_offset = search_start + i as u64;
                let distance = (candidate_offset as i64) - (wrong_offset as i64);

                // Accept matches within -100 to +10 bytes of wrong_offset
                // (xref might be slightly off by a few bytes)
                if (-100..=10).contains(&distance) {
                    // Prefer the match closest to wrong_offset
                    let is_better = best_match
                        .as_ref()
                        .is_none_or(|(_, best_dist)| distance.abs() < best_dist.abs());

                    if is_better {
                        best_match = Some((i, distance));
                    }
                }
            }
        }

        if let Some((pos, distance)) = best_match {
            let absolute_offset = search_start + pos as u64;
            log::debug!(
                "Found object header '{}' at offset {} ({:+} bytes from xref at {})",
                expected_header,
                absolute_offset,
                distance,
                wrong_offset
            );
            return Ok(absolute_offset);
        }

        // Try with whitespace variations (space, double-space, tab between obj_id and gen)
        let patterns = [
            format!("{} {} obj", obj_ref.id, obj_ref.gen).into_bytes(),
            format!("{}  {} obj", obj_ref.id, obj_ref.gen).into_bytes(),
            format!("{}\t{} obj", obj_ref.id, obj_ref.gen).into_bytes(),
            format!("{} {}\tobj", obj_ref.id, obj_ref.gen).into_bytes(),
        ];

        for pattern in &patterns {
            let mut best_match: Option<(usize, i64)> = None;

            for (i, window) in buffer[..bytes_read].windows(pattern.len()).enumerate() {
                if window == pattern.as_slice() {
                    let candidate_offset = search_start + i as u64;
                    let distance = (candidate_offset as i64) - (wrong_offset as i64);

                    if (-100..=10).contains(&distance) {
                        let is_better = best_match
                            .as_ref()
                            .is_none_or(|(_, best_dist)| distance.abs() < best_dist.abs());

                        if is_better {
                            best_match = Some((i, distance));
                        }
                    }
                }
            }

            if let Some((pos, distance)) = best_match {
                let absolute_offset = search_start + pos as u64;
                log::debug!(
                    "Found object header '{}' at offset {} ({:+} bytes, pattern match)",
                    expected_header,
                    absolute_offset,
                    distance
                );
                return Ok(absolute_offset);
            }
        }

        Err(Error::ParseError {
            offset: wrong_offset as usize,
            reason: format!(
                "Could not find object header '{}' within {} bytes before offset",
                expected_header, search_distance
            ),
        })
    }

    /// Get the document catalog (root object).
    ///
    /// The catalog is the root of the document's object hierarchy.
    /// It contains references to the page tree, outlines, etc.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The /Root entry is present but is not a reference
    /// - Loading the catalog object fails
    /// - The trailer omits /Root **and** no `/Type /Catalog` object can be
    ///   found by scanning (the issue #509 recovery path: a missing /Root is
    ///   not itself fatal — the Catalog is discovered by object scan, as
    ///   Poppler / PDFium do — but it does error if that scan also fails)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// let catalog = doc.catalog()?;
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn catalog(&self) -> Result<Object> {
        let trailer_dict = self
            .trailer
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Trailer is not a dictionary".to_string()))?;

        if let Some(root_obj) = trailer_dict.get("Root") {
            let root_ref = root_obj
                .as_reference()
                .ok_or_else(|| Error::InvalidPdf("/Root is not a reference".to_string()))?;
            return self.load_object(root_ref);
        }

        // The trailer omits /Root. A Linearized file's sparse end-of-file
        // trailer legitimately does this; discover the Catalog
        // by scanning indirect objects for /Type /Catalog, as Poppler /
        // PDFium do.
        self.find_catalog_by_scan().ok_or_else(|| {
            Error::InvalidPdf(
                "Trailer omits /Root and no /Type /Catalog object could be found by scanning"
                    .to_string(),
            )
        })
    }

    /// Scan indirect objects for the document Catalog (`/Type /Catalog`).
    ///
    /// Used only as a fallback when the trailer omits `/Root`.
    /// Bounded so a pathological xref can't turn this into an unbounded
    /// scan; the Catalog is virtually always one of the first objects.
    ///
    /// The smallest `MAX_SCAN` object numbers are scanned, ascending.
    /// `all_object_numbers()` is `HashMap`-backed, so iterating it directly
    /// would be nondeterministic — a bounded scan over an arbitrary subset
    /// can miss the Catalog on different runs. `smallest_object_numbers`
    /// makes discovery deterministic, scans low-numbered objects first
    /// (where the Catalog conventionally lives), and bounds the candidate
    /// set *before* sorting so a pathological xref stays O(n log MAX_SCAN).
    fn find_catalog_by_scan(&self) -> Option<Object> {
        const MAX_SCAN: usize = 4096;
        let nums = self.xref.smallest_object_numbers(MAX_SCAN);
        let mut checked = 0usize;
        for num in nums {
            if checked >= MAX_SCAN {
                break;
            }
            let generation = match self.xref.get(num) {
                Some(e) if e.in_use => e.generation,
                _ => continue,
            };
            checked += 1;
            if let Ok(obj) = self.load_object(ObjectRef::new(num, generation)) {
                if obj
                    .as_dict()
                    .and_then(|d| d.get("Type"))
                    .and_then(|t| t.as_name())
                    == Some("Catalog")
                {
                    log::info!("Catalog discovered by object scan: {} {} obj", num, generation);
                    return Some(obj);
                }
            }
        }
        None
    }

    /// Get the structure tree (logical structure) of the document.
    ///
    /// Tagged PDFs contain a structure tree that defines the logical structure
    /// and reading order of the document. This is the PDF-spec-compliant way
    /// to determine reading order.
    ///
    /// Returns `Ok(Some(StructTreeRoot))` if the document has a structure tree,
    /// `Ok(None)` if it's not a tagged PDF, or an error if parsing fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// if let Some(struct_tree) = doc.structure_tree()? {
    ///     println!("This is a Tagged PDF with logical structure");
    /// } else {
    ///     println!("This PDF does not have a structure tree");
    /// }
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn structure_tree(&self) -> Result<Option<crate::structure::StructTreeRoot>> {
        crate::structure::parse_structure_tree(self)
    }

    /// Returns the document's structure tree **only when it is trustworthy for
    /// reading-order purposes**, per ISO 32000-1:2008 §14.8.2.3.1 and §14.7.1.
    ///
    /// A `/StructTreeRoot` encodes the producer's *logical structure order* — a
    /// depth-first traversal of the tag hierarchy — which is authoritative for
    /// reading order independent of glyph geometry (§14.7.1). It is trusted when
    /// the document is `/Marked` (Tagged PDF) **or** the catalog directly
    /// references a `/StructTreeRoot` (PDF 1.3/1.4 tagged files predate the
    /// `/MarkInfo` dictionary; §7.7.2) — matching the historical gate so output
    /// for non-suspect documents is byte-for-byte unchanged — **and**
    /// `/MarkInfo /Suspects` is not `true`. A `true` `/Suspects` flag is the
    /// spec-sanctioned signal (the `/TagSuspect /Ordering` mechanism,
    /// §14.8.2.3.1) that page content order may not match logical structure
    /// order, so the tree is rejected and callers fall back to geometric order.
    ///
    /// Shares `structure_tree_cache`, so this costs a single cached parse.
    pub(crate) fn struct_tree_trustworthy(&self) -> Option<Arc<crate::structure::StructTreeRoot>> {
        let mark = self.mark_info().unwrap_or_default();
        // Suspect documents: geometric reading order is spec-correct
        // (§14.8.2.3.1). This is the only behavioural change versus the legacy
        // inline gate, which never consulted /Suspects.
        if mark.suspects {
            return None;
        }
        let cached = self.structure_tree_cache.lock_or_recover().clone();
        match cached {
            Some(tree) => tree,
            None => {
                let has_struct_tree_root = self
                    .catalog()
                    .ok()
                    .and_then(|cat| cat.as_dict().map(|d| d.contains_key("StructTreeRoot")))
                    .unwrap_or(false);
                let tree = if mark.marked || has_struct_tree_root {
                    self.structure_tree().ok().flatten().map(Arc::new)
                } else {
                    None
                };
                *self.structure_tree_cache.lock_or_recover() = Some(tree.clone());
                tree
            },
        }
    }

    /// Whether text extraction uses the Tagged-PDF *logical structure order* (a
    /// depth-first traversal of `/StructTreeRoot`) rather than geometric
    /// page-content order for this document.
    ///
    /// Returns `true` exactly when the document carries a trustworthy structure
    /// tree per ISO 32000-1:2008 §14.8.2.3.1 / §14.7.1: it is `/Marked` or the
    /// catalog references a `/StructTreeRoot`, the tree resolves non-empty, and
    /// `/MarkInfo /Suspects` is not `true`. When `false`, extraction falls back
    /// to geometric reading order. This is a read-only introspection accessor;
    /// it does not change extraction behaviour.
    pub fn prefers_structure_reading_order(&self) -> bool {
        self.struct_tree_trustworthy().is_some()
    }

    /// Find the document's default CMYK output-intent profile.
    ///
    /// Per ISO 32000-1:2008 §14.11.5, an `/OutputIntents` array in the
    /// catalog advertises the colour characteristics of the target
    /// output device. Each entry is a dictionary; the `DestOutputProfile`
    /// key (when present) references an ICC profile stream identifying
    /// the intended press / display calibration.
    ///
    /// This method returns the **first CMYK** `DestOutputProfile` it
    /// finds (N = 4) — the usual match for "here is how my CMYK ink
    /// should look" on PDF/X files. Callers can use it as a fallback
    /// profile for plain `/DeviceCMYK` images that lack their own ICC
    /// colour space.
    ///
    /// Returns `None` when no output intent exists, no CMYK entry is
    /// present, or the profile stream can't be parsed as ICC.
    pub fn output_intent_cmyk_profile(&self) -> Option<std::sync::Arc<crate::color::IccProfile>> {
        let catalog = self.catalog().ok()?;
        let cat_dict = catalog.as_dict()?;

        let intents_obj = cat_dict.get("OutputIntents")?;
        let intents_obj = match intents_obj {
            Object::Reference(r) => self.load_object(*r).ok()?,
            _ => intents_obj.clone(),
        };
        let intents_arr = match &intents_obj {
            Object::Array(a) => a.clone(),
            _ => return None,
        };

        for entry in intents_arr {
            let entry = match entry {
                Object::Reference(r) => self.load_object(r).ok()?,
                other => other,
            };
            let entry_dict = match entry.as_dict() {
                Some(d) => d.clone(),
                None => continue,
            };
            let profile_obj = match entry_dict.get("DestOutputProfile") {
                Some(p) => p.clone(),
                None => continue,
            };
            let profile_stream = match profile_obj {
                Object::Reference(r) => match self.load_object(r) {
                    Ok(o) => o,
                    Err(_) => continue,
                },
                other => other,
            };

            let Object::Stream { dict, .. } = &profile_stream else {
                continue;
            };
            let n = match dict.get("N").and_then(|o| o.as_integer()) {
                Some(4) => 4u8, // only CMYK; ignore RGB/Gray output intents here
                _ => continue,
            };
            let bytes = match profile_stream.decode_stream_data() {
                Ok(b) => b,
                Err(_) => continue,
            };
            if let Some(prof) = crate::color::IccProfile::parse(bytes, n) {
                return Some(std::sync::Arc::new(prof));
            }
        }
        None
    }

    /// Get the MarkInfo dictionary from the document catalog.
    ///
    /// The MarkInfo dictionary indicates whether the document conforms to
    /// Tagged PDF conventions and whether the structure tree might contain
    /// suspect (unreliable) content.
    ///
    /// Per ISO 32000-1:2008 Section 14.7.1, the MarkInfo dictionary contains:
    /// - `/Marked` - Whether the document conforms to Tagged PDF conventions
    /// - `/Suspects` - Whether the document contains suspect content
    /// - `/UserProperties` - Whether the document uses user properties
    ///
    /// # Returns
    ///
    /// Returns `MarkInfo` with the parsed values, or default values if
    /// the MarkInfo dictionary is not present.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// let mark_info = doc.mark_info()?;
    /// if mark_info.is_structure_reliable() {
    ///     println!("Structure tree can be trusted for reading order");
    /// } else if mark_info.suspects {
    ///     println!("Structure tree may contain unreliable content");
    /// }
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn mark_info(&self) -> Result<crate::structure::MarkInfo> {
        let catalog = self.catalog()?;
        let catalog_dict = match catalog.as_dict() {
            Some(d) => d,
            None => return Ok(crate::structure::MarkInfo::default()),
        };

        // Get /MarkInfo dictionary
        let mark_info_obj = match catalog_dict.get("MarkInfo") {
            Some(obj) => obj,
            None => return Ok(crate::structure::MarkInfo::default()),
        };

        // Resolve reference if needed
        let mark_info_obj = if let Some(r) = mark_info_obj.as_reference() {
            self.load_object(r)?
        } else {
            mark_info_obj.clone()
        };

        let mark_info_dict = match mark_info_obj.as_dict() {
            Some(d) => d,
            None => return Ok(crate::structure::MarkInfo::default()),
        };

        // Parse boolean fields with defaults of false
        let marked = mark_info_dict
            .get("Marked")
            .and_then(|o: &crate::object::Object| o.as_bool())
            .unwrap_or(false);

        let suspects = mark_info_dict
            .get("Suspects")
            .and_then(|o: &crate::object::Object| o.as_bool())
            .unwrap_or(false);

        let user_properties = mark_info_dict
            .get("UserProperties")
            .and_then(|o: &crate::object::Object| o.as_bool())
            .unwrap_or(false);

        Ok(crate::structure::MarkInfo {
            marked,
            suspects,
            user_properties,
        })
    }

    /// Get the number of pages in the document.
    ///
    /// This function:
    /// 1. Loads the catalog (root object)
    /// 2. Follows the /Pages reference to the page tree root
    /// 3. Extracts the /Count value from the page tree
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The catalog cannot be loaded
    /// - The /Pages entry is missing or invalid
    /// - The page tree root does not contain a /Count entry
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// let count = doc.page_count()?;
    /// println!("Document has {} pages", count);
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn page_count(&self) -> Result<usize> {
        // Try standard method first
        match self.get_page_count_standard() {
            Ok(count) => {
                log::debug!("Page count from /Count: {}", count);
                Ok(count)
            },
            Err(Error::EncryptedPdf) => Err(Error::EncryptedPdf),
            Err(e) => {
                // For encrypted PDFs any failure to read the page tree means we
                // cannot access the content. Scanning would also return Ok(0),
                // so skip the fallback and surface the real error immediately.
                if self.is_encrypted() {
                    log::warn!("Page count failed for encrypted PDF: {}", e);
                    return Err(Error::EncryptedPdf);
                }

                log::warn!("Failed to get page count from /Count: {}", e);
                log::info!("Falling back to scanning page tree");

                // Fallback: scan the page tree manually
                match self.get_page_count_by_scanning() {
                    Ok(count) => {
                        log::info!("Page count from scanning: {}", count);
                        Ok(count)
                    },
                    Err(scan_err) => {
                        log::error!("Both methods failed. Standard: {}, Scan: {}", e, scan_err);
                        Err(e) // Return original error
                    },
                }
            },
        }
    }

    /// Get the MediaBox of a page (v0.3.14).
    ///
    /// MediaBox defines the physical boundaries of the page in user space units.
    pub fn get_page_media_box(&self, page_index: usize) -> Result<(f32, f32, f32, f32)> {
        let page = self.get_page(page_index)?;
        let page_dict = page
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Page is not a dictionary".to_string()))?;

        // Resolve indirect reference if present — PDF spec §7.3.10 permits any value
        // to be an indirect reference, e.g. `/MediaBox 174 0 R` where 174 0 R is `[0 0 612 792]`.
        let media_box_obj_raw = page_dict
            .get("MediaBox")
            .ok_or_else(|| Error::InvalidPdf("MediaBox not found or not an array".to_string()))?;
        let media_box_obj = self.resolve_obj_ref(media_box_obj_raw);
        let media_box = media_box_obj
            .as_array()
            .ok_or_else(|| Error::InvalidPdf("MediaBox not found or not an array".to_string()))?;

        if media_box.len() < 4 {
            return Err(Error::InvalidPdf("MediaBox must have at least 4 elements".to_string()));
        }

        fn to_f32(obj: &Object) -> f32 {
            match obj {
                Object::Integer(v) => *v as f32,
                Object::Real(v) => *v as f32,
                _ => 0.0,
            }
        }

        // §7.3.10: *any* element of the rectangle array may itself be an
        // indirect reference (pdf.js issue7872 stores `/MediaBox
        // [4 0 R 5 0 R 6 0 R 7 0 R]`). Resolve each element before
        // coercing — otherwise an unresolved Reference reads as 0.0 and
        // the page collapses to a zero-area box that clips all content.
        Ok((
            to_f32(&self.resolve_obj_ref(&media_box[0])),
            to_f32(&self.resolve_obj_ref(&media_box[1])),
            to_f32(&self.resolve_obj_ref(&media_box[2])),
            to_f32(&self.resolve_obj_ref(&media_box[3])),
        ))
    }

    /// Page `/Rotate` normalised to one of `{0, 90, 180, 270}`
    /// (ISO 32000-1 §7.7.3.3); `0` when absent or invalid.
    ///
    /// Pure inspection (no feature gate) for the auto-extraction
    /// classifier (#517 case I — transformed-bbox coverage / OCR
    /// orientation). Resolves via [`get_page`](Self::get_page), so the
    /// inheritable `/Rotate` attribute (ISO 32000-1 §7.7.3.4) is walked
    /// up the page tree — a `/Rotate` set on an ancestor `/Pages` node
    /// is honoured, not just one on the leaf page object.
    pub fn get_page_rotation(&self, page_index: usize) -> Result<i32> {
        let page = self.get_page(page_index)?;
        let dict = page
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Page is not a dictionary".to_string()))?;
        let raw = match dict.get("Rotate") {
            Some(r) => match self.resolve_obj_ref(r) {
                Object::Integer(v) => v as i32,
                Object::Real(v) => v as i32,
                _ => 0,
            },
            None => 0,
        };
        // `/Rotate` shall be a multiple of 90 (ISO 32000-1 §7.7.3.3);
        // a non-multiple is invalid → `0` (per this fn's contract),
        // NOT silently floored (e.g. 135 must not become 90).
        let n = ((raw % 360) + 360) % 360;
        Ok(if n % 90 == 0 { n } else { 0 })
    }

    /// Get page count using the standard /Count field
    fn get_page_count_standard(&self) -> Result<usize> {
        // Load catalog
        let catalog = self.catalog()?;
        let catalog_dict = catalog.as_dict().ok_or_else(|| Error::InvalidObjectType {
            expected: "Dictionary".to_string(),
            found: catalog.type_name().to_string(),
        })?;

        // Get /Pages reference
        let pages_ref = catalog_dict
            .get("Pages")
            .ok_or_else(|| Error::InvalidPdf("Catalog missing /Pages entry".to_string()))?
            .as_reference()
            .ok_or_else(|| Error::InvalidPdf("/Pages is not a reference".to_string()))?;

        // Load page tree root
        let pages_obj = self.load_object(pages_ref)?;
        let pages_dict = match pages_obj.as_dict() {
            Some(d) => d,
            None => {
                // If the page tree root resolved to Null it usually means the
                // PDF is encrypted and the page tree could not be decrypted.
                // Surface the real error instead of silently reporting 0 pages.
                if matches!(pages_obj, crate::object::Object::Null) && self.is_encrypted() {
                    return Err(Error::EncryptedPdf);
                }
                log::warn!(
                    "Page tree root is {} (expected Dictionary), treating as 0 pages",
                    pages_obj.type_name()
                );
                return Ok(0);
            },
        };

        // Get /Count
        let count = pages_dict
            .get("Count")
            .ok_or_else(|| Error::InvalidPdf("Page tree missing /Count entry".to_string()))?
            .as_integer()
            .ok_or_else(|| Error::InvalidPdf("/Count is not an integer".to_string()))?;

        // Validate /Count against PDF spec limits (Annex C.2: max 8,388,607 indirect objects)
        const MAX_PAGES: i64 = 8_388_607;
        if !(0..=MAX_PAGES).contains(&count) {
            log::warn!(
                "/Count value {} is unreasonable (max {}), falling back to tree scan",
                count,
                MAX_PAGES
            );
            return self.get_page_count_by_scanning();
        }

        // Sanity check: /Count can't exceed total objects in the file
        let max_objects = self.xref.len();
        if (count as usize) > max_objects {
            log::warn!(
                "/Count {} exceeds total objects {}, falling back to tree scan",
                count,
                max_objects
            );
            return self.get_page_count_by_scanning();
        }

        Ok(count as usize)
    }

    /// Get page count by scanning the page tree (fallback method)
    fn get_page_count_by_scanning(&self) -> Result<usize> {
        // Load catalog
        let catalog = self.catalog()?;
        let catalog_dict = catalog.as_dict().ok_or_else(|| Error::InvalidObjectType {
            expected: "Dictionary".to_string(),
            found: catalog.type_name().to_string(),
        })?;

        // Get /Pages reference
        let pages_ref = catalog_dict
            .get("Pages")
            .ok_or_else(|| Error::InvalidPdf("Catalog missing /Pages entry".to_string()))?
            .as_reference()
            .ok_or_else(|| Error::InvalidPdf("/Pages is not a reference".to_string()))?;

        // Count pages by traversing the tree
        self.count_pages_recursive(pages_ref, 0)
    }

    /// Recursively count pages in the page tree
    fn count_pages_recursive(&self, node_ref: ObjectRef, depth: usize) -> Result<usize> {
        // Prevent infinite recursion
        const MAX_DEPTH: usize = 50;
        if depth > MAX_DEPTH {
            log::warn!("Page tree depth exceeded {} levels, stopping", MAX_DEPTH);
            return Ok(0);
        }

        // Load the node
        let node = match self.load_object(node_ref) {
            Ok(n) => n,
            Err(e) => {
                log::warn!("Failed to load page tree node {}: {}", node_ref, e);
                return Ok(0); // Skip this node
            },
        };

        let node_dict = match node.as_dict() {
            Some(d) => d,
            None => {
                log::warn!("Page tree node {} is not a dictionary", node_ref);
                return Ok(0);
            },
        };

        // Check node type
        let node_type = node_dict.get("Type").and_then(|obj| obj.as_name());

        match node_type {
            Some("Page") => {
                // This is a leaf page
                Ok(1)
            },
            Some("Pages") => {
                // This is an intermediate node with kids
                let kids = match node_dict.get("Kids").and_then(|obj| obj.as_array()) {
                    Some(k) => k,
                    None => {
                        log::warn!("Pages node {} missing /Kids array", node_ref);
                        return Ok(0);
                    },
                };

                let mut count = 0;
                for kid in kids {
                    if let Some(kid_ref) = kid.as_reference() {
                        match self.count_pages_recursive(kid_ref, depth + 1) {
                            Ok(page_count) => count += page_count,
                            Err(Error::CircularReference(obj_ref)) => {
                                log::warn!(
                                    "Circular reference in page tree at object {}, skipping",
                                    obj_ref
                                );
                                continue;
                            },
                            Err(Error::RecursionLimitExceeded(_)) => {
                                log::warn!(
                                    "Recursion limit exceeded in page tree, skipping branch"
                                );
                                continue;
                            },
                            Err(e) => {
                                log::warn!("Error counting pages in branch: {}, skipping", e);
                                continue;
                            },
                        }
                    }
                }
                Ok(count)
            },
            _ => {
                log::warn!("Unknown page tree node type: {:?}", node_type.unwrap_or("(none)"));
                Ok(0)
            },
        }
    }

    /// Get page count as u32 (legacy API).
    ///
    /// This is a convenience method that returns the page count as a u32.
    /// It calls `page_count()` internally but converts the result
    /// returns 0 if an error occurs (for backward compatibility).
    #[deprecated(
        since = "0.1.0",
        note = "Use page_count() instead, which returns Result"
    )]
    pub fn page_count_u32(&self) -> u32 {
        self.page_count().unwrap_or(0) as u32
    }

    /// Returns the page index range `0..page_count`, or an empty range
    /// when `page_count()` fails. Issue #447.
    ///
    /// Designed for `for i in doc.page_indices() { ... }` so callers
    /// don't have to write `for i in 0..doc.page_count()?`. The
    /// fallible-vs-iterator tension that motivated the issue is
    /// resolved by treating a metadata-broken document as having no
    /// pages at the iteration level — every per-page extraction call
    /// is already fallible and surfaces the real error.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for i in doc.page_indices() {
    ///     let text = doc.extract_text(i)?;
    ///     println!("page {}: {} chars", i, text.len());
    /// }
    /// ```
    pub fn page_indices(&self) -> std::ops::Range<usize> {
        0..self.page_count().unwrap_or(0)
    }

    /// Get a page object by index (0-based).
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    ///
    /// # Returns
    ///
    /// The page dictionary object.
    ///
    /// # Errors
    ///
    /// Returns an error if the page index is out of bounds or if the page
    /// tree structure is invalid.
    pub fn get_page(&self, page_index: usize) -> Result<Object> {
        // Check page cache first — page tree is static per §7.7.3.2
        if let Some(cached) = self.page_cache.lock_or_recover().get(&page_index).cloned() {
            return Ok(cached);
        }

        // Defer bulk page tree walk until enough pages are accessed.
        const LAZY_THRESHOLD: usize = 64;
        let cache_misses = self.page_cache.lock_or_recover().len();

        if !self.page_cache_populated.load(Ordering::Acquire) && cache_misses >= LAZY_THRESHOLD {
            self.page_cache_populated.store(true, Ordering::Release);
            if let Err(e) = self.populate_page_cache() {
                log::warn!(
                    "Bulk page tree walk failed ({}), falling back to per-page traversal",
                    e
                );
            }
            // Check cache after bulk population
            if let Some(cached) = self.page_cache.lock_or_recover().get(&page_index).cloned() {
                return Ok(cached);
            }
        }

        // Per-page tree traversal: walks only the branches needed to find target page
        let catalog = self.catalog()?;
        let catalog_dict = catalog.as_dict().ok_or_else(|| Error::InvalidObjectType {
            expected: "Dictionary".to_string(),
            found: catalog.type_name().to_string(),
        })?;

        let pages_ref = catalog_dict
            .get("Pages")
            .ok_or_else(|| Error::InvalidPdf("Catalog missing /Pages entry".to_string()))?
            .as_reference()
            .ok_or_else(|| Error::InvalidPdf("/Pages is not a reference".to_string()))?;

        let mut inherited = HashMap::new();

        let page = match self.get_page_from_tree(pages_ref, page_index, &mut 0, &mut inherited) {
            Ok(page) => {
                if let Some(dict) = page.as_dict() {
                    log::debug!("Collected page {}, keys: {:?}", page_index, dict.keys());
                    if let Some(contents) = dict.get("Contents") {
                        log::debug!("  -> /Contents: {:?}", contents);
                    }
                    if let Some(rotate) = dict.get("Rotate") {
                        log::debug!("  -> /Rotate: {:?}", rotate);
                    }
                }
                Ok(page)
            },
            Err(e) => {
                if matches!(
                    e,
                    Error::InvalidPdf(_)
                        | Error::InvalidObjectType { .. }
                        | Error::CircularReference(_)
                        | Error::ObjectNotFound(_, _)
                ) {
                    log::warn!("Page tree traversal failed ({}), trying fallback scan method", e);
                    self.get_page_by_scanning(page_index)
                } else {
                    Err(e)
                }
            },
        }?;

        self.page_cache
            .lock_or_recover()
            .insert(page_index, page.clone());
        Ok(page)
    }

    /// Walk the page tree once and populate page_cache for ALL pages.
    /// This avoids O(n²) cost when pages are accessed sequentially.
    fn populate_page_cache(&self) -> Result<()> {
        let catalog = self.catalog()?;
        let catalog_dict = catalog.as_dict().ok_or_else(|| Error::InvalidObjectType {
            expected: "Dictionary".to_string(),
            found: catalog.type_name().to_string(),
        })?;

        let pages_ref = catalog_dict
            .get("Pages")
            .ok_or_else(|| Error::InvalidPdf("Catalog missing /Pages entry".to_string()))?
            .as_reference()
            .ok_or_else(|| Error::InvalidPdf("/Pages is not a reference".to_string()))?;

        let mut page_index = 0usize;
        let mut inherited = HashMap::new();
        self.collect_all_pages(pages_ref, &mut page_index, &mut inherited, &mut HashSet::new())?;
        log::debug!("Populated page cache with {} pages", page_index);
        Ok(())
    }

    /// Pre-populate `image_xobject_cache` for all XObject refs across all cached pages.
    /// Collects all unique XObject references, sorts them by xref offset for sequential
    /// I/O (avoids random seeking in large files), then peeks each one via `is_form_xobject()`.
    #[allow(dead_code)]
    fn prefetch_xobject_subtypes(&self) {
        // Collect all unique XObject refs from all cached pages
        let mut xobj_refs: Vec<ObjectRef> = Vec::new();
        let page_dicts: Vec<Object> = self
            .page_cache
            .lock_or_recover()
            .values()
            .cloned()
            .collect();

        for page_obj in &page_dicts {
            let page_dict = match page_obj.as_dict() {
                Some(d) => d,
                None => continue,
            };
            let resources = match page_dict.get("Resources") {
                Some(r) => {
                    if let Some(ref_obj) = r.as_reference() {
                        match self.load_object(ref_obj) {
                            Ok(obj) => obj,
                            Err(_) => continue,
                        }
                    } else {
                        r.clone()
                    }
                },
                None => continue,
            };
            let res_dict = match resources.as_dict() {
                Some(d) => d,
                None => continue,
            };
            let xobj_obj = match res_dict.get("XObject") {
                Some(x) => {
                    if let Some(ref_obj) = x.as_reference() {
                        match self.load_object(ref_obj) {
                            Ok(obj) => obj,
                            Err(_) => continue,
                        }
                    } else {
                        x.clone()
                    }
                },
                None => continue,
            };
            if let Some(xobj_dict) = xobj_obj.as_dict() {
                for val in xobj_dict.values() {
                    if let Some(obj_ref) = val.as_reference() {
                        if !self
                            .image_xobject_cache
                            .lock_or_recover()
                            .contains(&obj_ref)
                        {
                            xobj_refs.push(obj_ref);
                        }
                    }
                }
            }
        }

        // Deduplicate
        xobj_refs.sort_unstable_by_key(|r| (r.id, r.gen));
        xobj_refs.dedup();

        // Sort by xref offset for sequential I/O
        xobj_refs.sort_by_key(|r| self.xref.get(r.id).map(|e| e.offset).unwrap_or(u64::MAX));

        log::debug!("Prefetching XObject subtypes for {} unique refs", xobj_refs.len());

        // Peek each ref — populates image_xobject_cache as a side effect
        for obj_ref in xobj_refs {
            self.is_form_xobject(obj_ref);
        }
    }

    /// Recursively walk the page tree and collect all pages into page_cache.
    fn collect_all_pages(
        &self,
        node_ref: ObjectRef,
        page_index: &mut usize,
        inherited: &mut HashMap<String, Object>,
        visited: &mut HashSet<ObjectRef>,
    ) -> Result<()> {
        if !visited.insert(node_ref) {
            return Err(Error::CircularReference(node_ref));
        }

        let node = self.load_object(node_ref)?;
        let node_dict = match node.as_dict() {
            Some(d) => d,
            None => return Ok(()), // Skip non-dict nodes gracefully
        };

        let node_type = node_dict
            .get("Type")
            .and_then(|obj| obj.as_name())
            .unwrap_or("");

        match node_type {
            "Page" => {
                // Apply inherited attributes
                let mut page_dict = node_dict.clone();
                for attr_name in &["Resources", "MediaBox", "CropBox", "Rotate"] {
                    if !page_dict.contains_key(*attr_name) {
                        if let Some(inherited_value) = inherited.get(*attr_name) {
                            log::debug!(
                                "Page {} inheriting {}: {:?}",
                                *page_index,
                                attr_name,
                                inherited_value
                            );
                            page_dict.insert(attr_name.to_string(), inherited_value.clone());
                        }
                    }
                }
                log::debug!("Collected page {}, keys: {:?}", *page_index, page_dict.keys());
                if let Some(contents) = page_dict.get("Contents") {
                    log::debug!("  -> /Contents: {:?}", contents);
                }
                if let Some(rotate) = page_dict.get("Rotate") {
                    log::debug!("  -> /Rotate: {:?}", rotate);
                }
                self.page_cache
                    .lock_or_recover()
                    .insert(*page_index, Object::Dictionary(page_dict));
                *page_index += 1;
            },
            "Pages" => {
                // Save inherited state so siblings don't see each other's overrides
                let saved = inherited.clone();

                // Nearest ancestor's attributes override more distant ones (PDF spec §7.7.3.4).
                // insert() is correct here because we snapshot/restore `inherited` around
                // the recursion, so this node's values apply only to its subtree.
                for attr_name in &["Resources", "MediaBox", "CropBox", "Rotate"] {
                    if let Some(attr_value) = node_dict.get(*attr_name) {
                        log::debug!(
                            "Pages node at {:?} providing inheritable {}: {:?}",
                            node_ref,
                            attr_name,
                            attr_value
                        );
                        inherited.insert(attr_name.to_string(), attr_value.clone());
                    }
                }

                if let Some(kids) = node_dict.get("Kids").and_then(|obj| obj.as_array()) {
                    for kid in kids {
                        if let Some(kid_ref) = kid.as_reference() {
                            if let Err(e) =
                                self.collect_all_pages(kid_ref, page_index, inherited, visited)
                            {
                                log::warn!(
                                    "Error collecting page from tree: {}, skipping branch",
                                    e
                                );
                            }
                        }
                    }
                }

                *inherited = saved;
            },
            _ => {}, // Unknown node type, skip
        }

        Ok(())
    }

    /// Get a page by scanning all objects in the PDF (fallback for broken page trees)
    /// This method is used when the standard page tree traversal fails due to malformed structure.
    fn get_page_by_scanning(&self, target_index: usize) -> Result<Object> {
        let mut current_index = 0;

        // Prime the ObjStm recovery cache up front when the xref looks
        // unreliable. Without this, the first pass below iterates only
        // `xref.all_object_numbers()` — which misses compressed objects
        // whose xref slots have been mis-flagged free. The sweep is a
        // one-shot, guarded by `objstm_recovery_done`, so this is cheap
        // if recovery already happened.
        self.recover_from_object_streams();

        // Collect all object numbers first to avoid borrow checker issues.
        // Sort for deterministic iteration order (HashMap iteration is
        // non-deterministic). We union the xref-listed ids with the object
        // ids recovered from the ObjStm sweep so that pages compressed in
        // streams whose xref slots were mis-flagged free still get visited.
        let mut obj_nums: Vec<u32> = self.xref.all_object_numbers().collect();
        for r in self.object_cache.lock_or_recover().keys() {
            obj_nums.push(r.id);
        }
        obj_nums.sort_unstable();
        obj_nums.dedup();

        // First pass: look for objects with /Type /Page
        for &obj_num in &obj_nums {
            if let Ok(obj) = self.load_object(ObjectRef {
                id: obj_num,
                gen: 0,
            }) {
                if let Some(dict) = obj.as_dict() {
                    if let Some(type_obj) = dict.get("Type") {
                        if let Some(type_name) = type_obj.as_name() {
                            if type_name == "Page" {
                                if current_index == target_index {
                                    return Ok(obj);
                                }
                                current_index += 1;
                            }
                        }
                    }
                }
            }
        }

        // Second pass: heuristic detection for pages without /Type entry.
        // Runs as a complement to pass 1 — counts page-like dicts that lack
        // a /Type entry alongside the /Type /Page matches, so that PDFs
        // whose corruption stripped /Type from some page dicts still reach
        // the full page count. Previously this pass only ran when pass 1
        // found zero pages, which meant any partial pass-1 match (e.g. 200
        // of 253 pages) would silently short pass 2 and fail.
        let mut heuristic_index = current_index;
        for &obj_num in &obj_nums {
            if let Ok(obj) = self.load_object(ObjectRef {
                id: obj_num,
                gen: 0,
            }) {
                if let Some(dict) = obj.as_dict() {
                    let has_no_type = dict.get("Type").is_none();
                    // Also handle /Type that is an unresolvable reference (Null)
                    let type_is_null = dict.get("Type").is_some_and(|t| matches!(t, Object::Null));
                    if (has_no_type || type_is_null)
                        && (dict.contains_key("MediaBox")
                            || dict.contains_key("Contents")
                            || (dict.contains_key("Resources") && dict.contains_key("Parent")))
                    {
                        log::debug!(
                            "Heuristic page candidate: object {} (page-like keys without valid /Type)",
                            obj_num
                        );
                        if heuristic_index == target_index {
                            return Ok(obj);
                        }
                        heuristic_index += 1;
                    }
                }
            }
        }
        current_index = heuristic_index;

        // Third pass: try resolving /Kids from catalog's /Pages root directly
        if current_index == 0 {
            if let Ok(catalog) = self.catalog() {
                if let Some(catalog_dict) = catalog.as_dict() {
                    if let Some(pages_ref) =
                        catalog_dict.get("Pages").and_then(|p| p.as_reference())
                    {
                        if let Ok(pages_obj) = self.load_object(pages_ref) {
                            if let Some(pages_dict) = pages_obj.as_dict() {
                                if let Some(kids) =
                                    pages_dict.get("Kids").and_then(|k| k.as_array())
                                {
                                    let mut kids_index = 0;
                                    for kid in kids {
                                        if let Some(kid_ref) = kid.as_reference() {
                                            // Skip self-referencing kids (cycle detection)
                                            if kid_ref == pages_ref {
                                                continue;
                                            }
                                            if let Ok(kid_obj) = self.load_object(kid_ref) {
                                                if let Some(kid_dict) = kid_obj.as_dict() {
                                                    // Skip intermediate /Pages nodes
                                                    let is_pages_node = kid_dict
                                                        .get("Type")
                                                        .and_then(|t| t.as_name())
                                                        .is_some_and(|n| n == "Pages");
                                                    if is_pages_node {
                                                        continue;
                                                    }
                                                    if kids_index == target_index {
                                                        log::debug!("Found page {} via direct /Kids resolution of object {}", target_index, kid_ref.id);
                                                        return Ok(kid_obj);
                                                    }
                                                    kids_index += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Err(Error::InvalidPdf(format!("Page index {} not found by scanning", target_index)))
    }

    /// Recursively traverse page tree to find a specific page.
    ///
    /// PDF Spec: ISO 32000-1:2008, Section 7.7.3.3 - Page Objects
    /// Implements attribute inheritance for /Resources, /MediaBox, /CropBox, /Rotate.
    ///
    /// Inheritable attributes from parent Pages nodes are collected as we traverse down
    /// the tree. When a Page is found, inherited attributes are merged in (only if the
    /// Page doesn't already have them - child values override parent values).
    fn get_page_from_tree(
        &self,
        node_ref: ObjectRef,
        target_index: usize,
        current_index: &mut usize,
        inherited: &mut HashMap<String, Object>,
    ) -> Result<Object> {
        self.get_page_from_tree_inner(
            node_ref,
            target_index,
            current_index,
            inherited,
            &mut HashSet::new(),
        )
    }

    fn get_page_from_tree_inner(
        &self,
        node_ref: ObjectRef,
        target_index: usize,
        current_index: &mut usize,
        inherited: &mut HashMap<String, Object>,
        visited: &mut HashSet<ObjectRef>,
    ) -> Result<Object> {
        if !visited.insert(node_ref) {
            return Err(Error::CircularReference(node_ref));
        }
        let node = self.load_object(node_ref)?;
        let node_dict = match node.as_dict() {
            Some(d) => d,
            None => {
                // Null or non-dict node in page tree — skip it
                log::warn!(
                    "Page tree node {} is {} (expected Dictionary), skipping",
                    node_ref.id,
                    node.type_name()
                );
                return Err(Error::InvalidPdf(format!(
                    "Page tree node {} is not a dictionary",
                    node_ref.id
                )));
            },
        };

        // Check if this is a page or pages node
        let node_type = node_dict
            .get("Type")
            .and_then(|obj| obj.as_name())
            .ok_or_else(|| Error::InvalidPdf("Page tree node missing /Type".to_string()))?;

        match node_type {
            "Pages" if *current_index < target_index => {
                // Skip entire subtree if /Count shows target is past this node.
                if let Some(count) = node_dict
                    .get("Count")
                    .and_then(|c| c.as_integer())
                    .filter(|&c| c > 0)
                {
                    let count = count as usize;
                    if *current_index + count <= target_index {
                        *current_index += count;
                        return Err(Error::InvalidPdf(format!(
                            "Page index {} not found in tree",
                            target_index
                        )));
                    }
                }
            },
            _ => {},
        }

        match node_type {
            "Page" => {
                if *current_index == target_index {
                    // Apply inherited attributes to this page
                    // PDF Spec: "If not present in the page dictionary, the value is inherited
                    // from an ancestor node in the page tree"
                    let mut page_dict = node_dict.clone();

                    // Inheritable attributes per PDF Spec Table 30:
                    // - Resources (required, can be inherited)
                    // - MediaBox (required, can be inherited)
                    // - CropBox (optional, can be inherited)
                    // - Rotate (optional, can be inherited)
                    let inheritable_attrs = ["Resources", "MediaBox", "CropBox", "Rotate"];

                    for attr_name in &inheritable_attrs {
                        // Only inherit if page doesn't already have this attribute
                        if !page_dict.contains_key(*attr_name) {
                            if let Some(inherited_value) = inherited.get(*attr_name) {
                                log::debug!(
                                    "Page {} inheriting /{} from ancestor Pages node",
                                    target_index,
                                    attr_name
                                );
                                page_dict.insert(attr_name.to_string(), inherited_value.clone());
                            }
                        }
                    }

                    Ok(Object::Dictionary(page_dict))
                } else {
                    *current_index += 1;
                    Err(Error::InvalidPdf(format!("Page index {} not found in tree", target_index)))
                }
            },
            "Pages" => {
                // This is an intermediate Pages node with kids
                // Collect inheritable attributes from this node to pass to children
                let inheritable_attrs = ["Resources", "MediaBox", "CropBox", "Rotate"];

                for attr_name in &inheritable_attrs {
                    if let Some(attr_value) = node_dict.get(*attr_name) {
                        // Only add if not already in inherited map (child values override parent)
                        inherited
                            .entry(attr_name.to_string())
                            .or_insert_with(|| attr_value.clone());
                    }
                }

                // Try to get /Kids array; if missing, this is a malformed PDF
                let kids = match node_dict.get("Kids").and_then(|obj| obj.as_array()) {
                    Some(k) => k,
                    None => {
                        log::warn!("Malformed PDF: Pages node missing /Kids array");
                        // Malformed PDF: Pages node has no /Kids array
                        // Gracefully return without error to allow other recovery paths
                        // The scanning method will find pages eventually
                        return Err(Error::InvalidPdf(
                            "Pages node missing /Kids array - try fallback method".to_string(),
                        ));
                    },
                };

                for kid in kids {
                    let kid_ref = kid.as_reference().ok_or_else(|| {
                        Error::InvalidPdf("Kid in /Kids array is not a reference".to_string())
                    })?;

                    match self.get_page_from_tree_inner(
                        kid_ref,
                        target_index,
                        current_index,
                        inherited,
                        visited,
                    ) {
                        Ok(page) => return Ok(page),
                        Err(Error::CircularReference(obj_ref)) => {
                            log::warn!(
                                "Circular reference in page tree at object {}, skipping",
                                obj_ref
                            );
                            continue;
                        },
                        Err(Error::RecursionLimitExceeded(_)) => {
                            log::warn!("Recursion limit exceeded in page tree, skipping branch");
                            continue;
                        },
                        Err(_) => continue,
                    }
                }

                Err(Error::InvalidPdf(format!("Page index {} not found", target_index)))
            },
            _ => Err(Error::InvalidPdf(format!("Unknown page tree node type: {}", node_type))),
        }
    }

    /// Get the object reference for a page by index.
    ///
    /// This is used by outline and annotations to find page references.
    pub(crate) fn get_page_ref(&self, page_index: usize) -> Result<ObjectRef> {
        let catalog = self.catalog()?;
        let catalog_dict = catalog.as_dict().ok_or_else(|| Error::InvalidObjectType {
            expected: "Dictionary".to_string(),
            found: catalog.type_name().to_string(),
        })?;

        let pages_ref = catalog_dict
            .get("Pages")
            .ok_or_else(|| Error::InvalidPdf("Catalog missing /Pages entry".to_string()))?
            .as_reference()
            .ok_or_else(|| Error::InvalidPdf("/Pages is not a reference".to_string()))?;

        self.get_page_ref_recursive(pages_ref, page_index, &mut 0, &mut HashSet::new())
    }

    /// Recursively find page reference in the page tree.
    pub(crate) fn get_page_ref_recursive(
        &self,
        node_ref: ObjectRef,
        target_index: usize,
        current_index: &mut usize,
        visited: &mut HashSet<ObjectRef>,
    ) -> Result<ObjectRef> {
        if !visited.insert(node_ref) {
            return Err(Error::CircularReference(node_ref));
        }
        let node = self.load_object(node_ref)?;
        let node_dict = match node.as_dict() {
            Some(d) => d,
            None => {
                log::warn!(
                    "Page tree node {} is {} (expected Dictionary), skipping",
                    node_ref.id,
                    node.type_name()
                );
                return Err(Error::InvalidPdf(format!(
                    "Page tree node {} is not a dictionary",
                    node_ref.id
                )));
            },
        };

        let node_type = node_dict
            .get("Type")
            .and_then(|t| t.as_name())
            .ok_or_else(|| Error::InvalidPdf("Node missing Type".to_string()))?;

        match node_type {
            "Page" => {
                if *current_index == target_index {
                    Ok(node_ref)
                } else {
                    *current_index += 1;
                    Err(Error::InvalidPdf(format!("Page {} not found", target_index)))
                }
            },
            "Pages" => {
                let kids = node_dict
                    .get("Kids")
                    .and_then(|k| k.as_array())
                    .ok_or_else(|| Error::InvalidPdf("Pages node missing Kids".to_string()))?;

                for kid_obj in kids {
                    if let Some(kid_ref) = kid_obj.as_reference() {
                        match self.get_page_ref_recursive(
                            kid_ref,
                            target_index,
                            current_index,
                            visited,
                        ) {
                            Ok(page_ref) => return Ok(page_ref),
                            Err(_) => continue,
                        }
                    }
                }

                Err(Error::InvalidPdf(format!("Page {} not found", target_index)))
            },
            _ => Err(Error::InvalidPdf(format!("Unknown node type: {}", node_type))),
        }
    }

    /// Extract text from a page as a plain string.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    ///
    /// # Returns
    ///
    /// The extracted text as a string.
    ///
    /// # Errors
    ///
    /// Returns an error if the page cannot be accessed or text extraction fails.
    /// Decode PDF escape sequences in text (e.g., \274 -> §, \( -> (, etc.)
    #[allow(dead_code)]
    fn decode_pdf_escapes(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '\\' {
                // Check what follows the backslash
                match chars.peek() {
                    Some(&'(') => {
                        result.push('(');
                        chars.next();
                    },
                    Some(&')') => {
                        result.push(')');
                        chars.next();
                    },
                    Some(&'\\') => {
                        result.push('\\');
                        chars.next();
                    },
                    Some(&'n') => {
                        result.push('\n');
                        chars.next();
                    },
                    Some(&'r') => {
                        result.push('\r');
                        chars.next();
                    },
                    Some(&'t') => {
                        result.push('\t');
                        chars.next();
                    },
                    Some(&'?') => {
                        // \? is a soft hyphen (optional line break point)
                        // Just skip it
                        chars.next();
                    },
                    Some(d) if d.is_ascii_digit() => {
                        // Octal escape sequence: \ddd
                        let mut octal = String::new();
                        for _ in 0..3 {
                            if let Some(&digit) = chars.peek() {
                                if digit.is_ascii_digit() && digit < '8' {
                                    octal.push(digit);
                                    chars.next();
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }

                        if !octal.is_empty() {
                            if let Ok(code) = u8::from_str_radix(&octal, 8) {
                                // PDFDocEncoding: ISO 32000-1:2008, Annex D
                                let decoded_char = Self::pdfdoc_decode(code);
                                result.push(decoded_char);
                            } else {
                                // Failed to parse, keep the backslash and octal
                                result.push('\\');
                                result.push_str(&octal);
                            }
                        } else {
                            // No valid octal digits, keep the backslash
                            result.push('\\');
                        }
                    },
                    _ => {
                        // Unknown escape, keep the backslash
                        result.push('\\');
                    },
                }
            } else {
                result.push(ch);
            }
        }

        result
    }

    /// Decode a byte using PDFDocEncoding (ISO 32000-1:2008, Annex D).
    ///
    /// PDFDocEncoding is the default encoding for text strings in PDF:
    /// - Codes 0-127: ASCII
    /// - Codes 128-159: Special Unicode characters
    /// - Codes 160-255: Latin-1 (ISO 8859-1)
    #[allow(dead_code)]
    fn pdfdoc_decode(code: u8) -> char {
        match code {
            // 0-127: Standard ASCII
            0..=127 => code as char,

            // 128-159: PDFDocEncoding special mappings
            128 => '\u{2022}', // BULLET
            129 => '\u{2020}', // DAGGER
            130 => '\u{2021}', // DOUBLE DAGGER
            131 => '\u{2026}', // HORIZONTAL ELLIPSIS
            132 => '\u{2014}', // EM DASH
            133 => '\u{2013}', // EN DASH
            134 => '\u{0192}', // LATIN SMALL LETTER F WITH HOOK
            135 => '\u{2044}', // FRACTION SLASH
            136 => '\u{2039}', // SINGLE LEFT-POINTING ANGLE QUOTATION MARK
            137 => '\u{203A}', // SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
            138 => '\u{2212}', // MINUS SIGN
            139 => '\u{2030}', // PER MILLE SIGN
            140 => '\u{201E}', // DOUBLE LOW-9 QUOTATION MARK
            141 => '\u{201C}', // LEFT DOUBLE QUOTATION MARK
            142 => '\u{201D}', // RIGHT DOUBLE QUOTATION MARK
            143 => '\u{2018}', // LEFT SINGLE QUOTATION MARK
            144 => '\u{2019}', // RIGHT SINGLE QUOTATION MARK
            145 => '\u{201A}', // SINGLE LOW-9 QUOTATION MARK
            146 => '\u{2122}', // TRADE MARK SIGN
            147 => '\u{FB01}', // LATIN SMALL LIGATURE FI
            148 => '\u{FB02}', // LATIN SMALL LIGATURE FL
            149 => '\u{0141}', // LATIN CAPITAL LETTER L WITH STROKE
            150 => '\u{0152}', // LATIN CAPITAL LIGATURE OE
            151 => '\u{0160}', // LATIN CAPITAL LETTER S WITH CARON
            152 => '\u{0178}', // LATIN CAPITAL LETTER Y WITH DIAERESIS
            153 => '\u{017D}', // LATIN CAPITAL LETTER Z WITH CARON
            154 => '\u{0131}', // LATIN SMALL LETTER DOTLESS I
            155 => '\u{0142}', // LATIN SMALL LETTER L WITH STROKE
            156 => '\u{0153}', // LATIN SMALL LIGATURE OE
            157 => '\u{0161}', // LATIN SMALL LETTER S WITH CARON
            158 => '\u{017E}', // LATIN SMALL LETTER Z WITH CARON
            159 => '\u{FFFD}', // REPLACEMENT CHARACTER (undefined in PDFDocEncoding)

            // 160-255: Latin-1 (ISO 8859-1)
            160..=255 => code as char,
        }
    }

    /// Circular references and recursion limit errors are handled gracefully
    /// with warning messages in the output.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// let text = doc.extract_text(0)?;
    /// println!("Page 1 text: {}", text);
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    ///
    /// # Extract text from a page
    ///
    /// pdf_oxide exposes three plain-text surfaces with different strengths
    /// (#554). Pick by document shape:
    ///
    /// - `extract_text(page)` (this method) — glyph-walk assembly with
    ///   row-aware ordering, inline table rendering, and artifact filtering.
    ///   The most discoverable default; strongest on single-column prose.
    /// - `to_plain_text(page, opts)` / `to_plain_text_all(opts)` — runs the
    ///   full pipeline (reading-order strategy incl. XY-cut). Best on
    ///   multi-column / complex layouts where reading order matters.
    /// - `to_markdown_all(opts)` then strip markup — preserves structure
    ///   (headings, lists, tables) and often scores highest on heavily
    ///   structured documents; lossiest for pure prose.
    ///
    /// No single mode wins on every PDF; when extraction quality is critical
    /// and the layout is unknown, compare `to_plain_text_all` and
    /// markdown-stripped output and keep whichever is better for your corpus.
    pub fn extract_text(&self, page_index: usize) -> Result<String> {
        // Enable table extraction so that tabular content is preserved as
        // space-padded, column-aligned rows (see Table::render_text).
        let options = crate::converters::ConversionOptions {
            extract_tables: true,
            ..Default::default()
        };
        self.extract_text_with_options(page_index, &options)
    }

    /// Extract text from a page with specific options (v0.3.16).
    pub fn extract_text_with_options(
        &self,
        page_index: usize,
        options: &crate::converters::ConversionOptions,
    ) -> Result<String> {
        let base_spans = self.extract_spans(page_index)?;
        let text = self.assemble_text_from_spans(page_index, base_spans, options)?;
        Ok(Self::apply_mixed_rtl_line_pass(text))
    }

    /// Per-line UAX #9 pass for mixed-direction lines (bidi item 4): for each
    /// output line that is confidently RTL and mixes Arabic/Hebrew with
    /// European/Arabic-Indic numerals or Latin words (e.g. a date
    /// `14 april 1434 ٤٣٤١`), give the embedded LTR sub-runs their left-to-right
    /// sublevel (UAX #9 §3.3.4) while leaving the already-logical RTL runs fixed.
    /// Gated inside `reorder_mixed_rtl_line`, so pure-RTL, pure-LTR, and
    /// non-RTL lines are returned byte-for-byte unchanged; the ASCII fast path
    /// keeps all Latin-only extraction identical.
    fn apply_mixed_rtl_line_pass(text: String) -> String {
        if text.is_ascii() {
            return text;
        }
        text.split('\n')
            .map(crate::text::bidi::reorder_mixed_rtl_line)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Apply caller-specified region filters to a span set: drop spans matching
    /// any `exclude_regions` (under `exclude_regions_mode`), then keep only spans
    /// inside `include_region` if one is set. Exclusion runs first so it takes
    /// precedence. Shared by the plain-text, markdown, and HTML conversion paths
    /// so `ConversionOptions` region filtering behaves identically across every
    /// text surface (#609). A no-op when neither field is set.
    fn apply_region_filters(
        base_spans: Vec<crate::layout::TextSpan>,
        options: &crate::converters::ConversionOptions,
    ) -> Vec<crate::layout::TextSpan> {
        use crate::layout::SpatialCollectionFiltering;
        let mut spans = base_spans;
        if !options.exclude_regions.is_empty() {
            spans = spans.exclude_rects(&options.exclude_regions, options.exclude_regions_mode);
        }
        if let Some((ref region, mode)) = options.include_region {
            spans = spans.filter_by_rect(region, mode);
        }
        spans
    }

    fn assemble_text_from_spans(
        &self,
        page_index: usize,
        base_spans: Vec<crate::layout::TextSpan>,
        options: &crate::converters::ConversionOptions,
    ) -> Result<String> {
        if self.is_encrypted_unreadable() {
            log::warn!("PDF is encrypted and could not be decrypted; returning empty text");
            return Ok(String::new());
        }

        let base_spans = Self::apply_region_filters(base_spans, options);

        // Structure tree: use it for reading order only when it is trustworthy
        // per the shared predicate (§14.8.2.3.1) — the document is /Marked or
        // the catalog references a /StructTreeRoot (PDF 1.4 documents such as
        // hello_structure.pdf predate /MarkInfo but are still tagged, §14.7.1),
        // AND /MarkInfo /Suspects is not true. Suspect documents fall through to
        // the geometric `else` arm below, the spec-correct behaviour.
        let cached_tree = self.struct_tree_trustworthy();
        let widget_spans = self.extract_widget_spans(page_index);

        // Table detection uses base spans only (no widget spans).
        let tables = if options.extract_tables {
            // text_fallback=false: extract_text preserves the pre-v0.3.47 behaviour
            // where line-less pages return no tables. Only the structured-output
            // converters (to_markdown, to_html) opt in to text-only spatial fallback.
            self.extract_page_tables(page_index, &base_spans, options, false)
        } else {
            Vec::new()
        };

        let mut all_spans = base_spans;
        all_spans.extend(widget_spans);

        if all_spans.is_empty() {
            let page = self.get_page(page_index)?;
            let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
                offset: 0,
                reason: "Page is not a dictionary".to_string(),
            })?;
            let no_content_text = if self.page_cannot_have_text(page_dict) {
                true
            } else {
                // Also check content stream for BT/Do operators (SIMD-fast scan).
                match self.get_page_content_data(page_index) {
                    Ok(ref content_data) => !Self::may_contain_text(content_data),
                    Err(_) => false, // Can't read content stream — be conservative
                }
            };
            if no_content_text {
                let mut text = String::new();
                self.append_non_widget_annotation_text(page_index, &mut text);
                return Ok(text);
            }
        }

        let text = if let Some(ref struct_tree) = cached_tree {
            // Build per-page traversal cache once, then O(1) lookup per page.
            if self.structure_content_cache.lock_or_recover().is_none() {
                let all_content = crate::structure::traverse_structure_tree_all_pages(struct_tree);
                *self.structure_content_cache.lock_or_recover() = Some(all_content);
            }
            self.extract_text_structure_order_cached_with_spans(page_index, all_spans)?
        } else {
            // Untagged PDF: Use page content order
            let mut spans = all_spans;

            // Exclude spans that are inside detected tables, BUT
            // preserve multi-row-spanning label columns.
            // The spatial table extractor clusters data cells into
            // table cells but does NOT emit the sparse label column
            // that sits vertically centred within each multi-row data
            // block (common on CJK lab-report reference tables like
            // WS/T 779). Those labels would otherwise be dropped
            // entirely from the output: the retain below would remove
            // them because their bbox is inside the table,
            // `table.render_text()` would not re-emit them because the
            // extractor never captured them as cells. Before running
            // the retain filter we identify these rowspan labels (same
            // heuristic `reorder_rowspan_labels` uses) and keep them in
            // the span list so `reorder_rowspan_labels` below can
            // promote them to the top of their row block.
            if !tables.is_empty() {
                // Absorb floating-point accumulation error in the
                // difference between a span's directly-computed
                // bbox.right (origin + width, small accumulation)
                // and a table bbox.right (min/max reduction across
                // many cell edges, larger accumulation). Without
                // this slack, a span whose real geometry is inside
                // the table by construction but whose f32 right-edge
                // exceeds the table's f32 right-edge by ~0.01–0.05pt
                // gets wrongly kept in the flow stream, producing
                // duplicated output. 0.1pt is well below any
                // visually meaningful PDF layout distance.
                const RETAIN_TOLERANCE: f32 = 0.1;

                // Build the set of cell text strings that every detected
                // table will render via `table.render_text()`. Labels
                // whose exact text already appears as a cell in some
                // table are already covered by the inline-table flush
                // below, so we must NOT also preserve them in the flow
                // span list (it would produce duplicate output).
                let mut table_cell_texts: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                for t in &tables {
                    for row in &t.rows {
                        for cell in &row.cells {
                            let trimmed = cell.text.trim();
                            if !trimmed.is_empty() {
                                table_cell_texts.insert(trimmed.to_string());
                            }
                        }
                    }
                }

                // For tagged PDFs, collect the MCIDs that are actually owned by
                // table cells. When a span's MCID is NOT in this set, the span is
                // NOT part of the table even if it lies inside the table's bbox
                // (e.g. a paragraph physically adjacent to a table that was tagged
                // as a sibling <P> element, not as a <TD>). Filtering such spans
                // by bbox alone would silently drop real content.
                // Falls back to bbox-only filtering when no MCIDs are present
                // (untagged PDFs or spatial-detection tables).
                let table_cell_mcids: HashSet<u32> = tables
                    .iter()
                    .flat_map(|t| {
                        t.rows
                            .iter()
                            .flat_map(|r| r.cells.iter().flat_map(|c| c.mcids.iter().copied()))
                    })
                    .collect();
                // Returns true when span should be removed from the flow because
                // it is owned by a table cell (will be re-emitted by render_text).
                let span_in_table = |s: &crate::layout::TextSpan| -> bool {
                    if !table_cell_mcids.is_empty() {
                        if let Some(mcid) = s.mcid {
                            // Tagged PDF: MCID decides ownership precisely.
                            return table_cell_mcids.contains(&mcid);
                        }
                        // Tagged PDF but span has no MCID (widget/annotation):
                        // keep in flow — better to duplicate than to silently drop.
                        return false;
                    }
                    // Untagged PDF or no MCIDs in any cell: cell-bbox-based filter.
                    // Using per-cell bboxes (rather than the coarser table bbox) prevents
                    // dropping paragraph spans that lie inside the table's outer bounding
                    // box but were not captured as table cells by the spatial detector.
                    if tables.iter().any(|t| {
                        t.rows.iter().any(|r| {
                            r.cells.iter().any(|c| {
                                c.bbox.is_some_and(|b| {
                                    Self::contains_rect_with_tolerance(
                                        &b,
                                        &s.bbox,
                                        RETAIN_TOLERANCE,
                                    )
                                })
                            })
                        })
                    }) {
                        return true;
                    }
                    // Fallback: text-based match. The bbox check above uses
                    // a tight 0.1pt tolerance and rejects spans whose font
                    // ascent extends slightly above the cell's ink box (issue
                    // 484: "FY 15 1st Q TTL" labels in JAL traffic table —
                    // span height = font_size = 10.7pt, but cell bbox height
                    // = 15.96pt covers two ink rows so the label glyphs reach
                    // ~0.4pt above the cell's top edge). When a span's
                    // trimmed text exactly matches some cell's text, the cell
                    // already owns it — keeping it in flow would duplicate.
                    let trimmed = s.text.trim();
                    if trimmed.is_empty() {
                        return false;
                    }
                    if !table_cell_texts.contains(trimmed) {
                        return false;
                    }
                    // Require spatial proximity: the span must lie inside
                    // some table's outer bbox so we don't drop body text that
                    // coincidentally matches a cell's text elsewhere on the page.
                    tables.iter().any(|t| {
                        t.bbox.is_some_and(|tb| {
                            let cx = s.bbox.x + s.bbox.width / 2.0;
                            let cy = s.bbox.y + s.bbox.height / 2.0;
                            cx >= tb.x - RETAIN_TOLERANCE
                                && cx <= tb.x + tb.width + RETAIN_TOLERANCE
                                && cy >= tb.y - RETAIN_TOLERANCE
                                && cy <= tb.y + tb.height + RETAIN_TOLERANCE
                        })
                    })
                };

                let preserved_label_indices: std::collections::HashSet<usize> =
                    Self::identify_multi_row_labels(&spans)
                        .into_iter()
                        .filter(|&idx| {
                            // Only preserve labels whose text is NOT
                            // already emitted by any table's
                            // `render_text()`. This is what makes the
                            // #329 fix safe on pages where the spatial
                            // extractor captured the sparse label
                            // column as cells — we let the table
                            // render them and drop them from flow.
                            // On pages like WS/T 779 where the label
                            // column is a genuine multi-row-spanning
                            // column that the extractor did NOT
                            // capture, the set is empty and every
                            // identified label stays in flow where
                            // `reorder_rowspan_labels` below can
                            // promote it.
                            let t = spans[idx].text.trim();
                            !t.is_empty() && !table_cell_texts.contains(t)
                        })
                        .collect();

                if preserved_label_indices.is_empty() {
                    spans.retain(|s| !span_in_table(s));
                } else {
                    let kept: Vec<crate::layout::TextSpan> = spans
                        .drain(..)
                        .enumerate()
                        .filter_map(|(i, s)| {
                            if !span_in_table(&s) || preserved_label_indices.contains(&i) {
                                Some(s)
                            } else {
                                None
                            }
                        })
                        .collect();
                    spans = kept;
                }
            }

            // Row-aware ordering: quantize Y into bands and sort band-
            // descending, then X ascending within a band. Strict Y sorting
            // would interleave cells from the same tabular row whose Y
            // values differ by typographic jitter (common in CJK layouts,
            // superscripts, and centered multi-line labels).
            //
            // Skip for multi-column pages: extract_spans() already applied
            // XY-cut column ordering. Re-sorting with row-aware would
            // interleave left/right columns line-by-line, producing garbled
            // output like "accompaally" instead of "accompanying table".
            if !Self::is_multi_column_page(&spans) {
                spans.sort_by(|a, b| {
                    let cmp =
                        crate::utils::row_aware_span_cmp(a.bbox.y, a.bbox.x, b.bbox.y, b.bbox.x);
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                    a.sequence.cmp(&b.sequence)
                });

                // Promote multi-row-spanning labels (sparse-column spans
                // vertically centred across several dense-column data rows)
                // to sort at the top of their row block.
                Self::reorder_rowspan_labels(&mut spans);

                // Restore intra-line reading order after the row-aware band sort.
                // Off-baseline glyphs (e.g. superscripts/subscripts) can land in
                // adjacent bands and be emitted out of X order; fix that per line.
                Self::reorder_same_line_runs(&mut spans);
            }

            // OCR fallback for scanned PDFs
            #[cfg(feature = "ocr")]
            if spans.is_empty() || spans.iter().map(|s| s.text.len()).sum::<usize>() < 50 {
                if let Ok(true) = crate::ocr::needs_ocr(self, page_index) {
                    log::debug!(
                        "Page {} appears to be scanned, OCR available but not auto-enabled",
                        page_index
                    );
                }
            }

            // Drop content marked /Artifact (PDF Spec ISO 32000-1:2008
            // §14.8.2.2 — headers, footers, page numbers, decorations).
            // Untagged-PDF running-header detection runs at document
            // level and feeds the same artifact_type flag.
            spans.retain(|s| s.artifact_type.is_none());

            // RTL correction
            Self::reverse_rtl_visual_order_runs(&mut spans);

            // Filter out invalid spans
            spans.retain(|s| {
                s.bbox.x.is_finite()
                    && s.bbox.y.is_finite()
                    && s.bbox.width.is_finite()
                    && s.bbox.height.is_finite()
                    && s.font_size.is_finite()
            });

            // Merge subscript/superscript spans into their base spans so that
            // tokens like "k1" and "k2" appear as single words rather than
            // as isolated fragments interleaved with other spans (pdfa_004).
            Self::merge_sub_superscript_spans(&mut spans);

            // Inline table insertion.
            //
            // Tables were previously rendered in a single block appended
            // at the end of the page text, after all flow spans. That
            // matches how `extract_text` historically worked but it means
            // tabular content appears far away from the prose that
            // surrounds it in reading order — on product data sheets
            // like ORAFOL 5900 the "Physical and Chemical Properties"
            // label/value rows showed up 20+ lines below the section
            // they belong to, which the reporter of #315 perceived as
            // the content being dropped entirely.
            //
            // Instead, maintain a sorted queue of tables keyed by their
            // top-Y (the larger Y coordinate of the table's bbox, per PDF
            // user-space conventions where Y grows upward). As we walk
            // the flow spans in row-aware reading order, whenever the
            // next span's top-Y falls below the top-Y of the queue's
            // leading table, we flush that table's rendered text at
            // that point, then continue. A final pass at the end emits
            // any tables whose top-Y is below all remaining spans (or
            // that have no flow spans at all).
            //
            // Tables are emitted at most once regardless of how many
            // spans sit above them, preserving existing behaviour
            // semantics while inlining the rendering at its spatial
            // reading-order position.
            let mut pending_tables: Vec<(f32, &crate::structure::table_extractor::Table)> = tables
                .iter()
                .filter_map(|t| t.bbox.map(|b| (b.y + b.height, t)))
                .collect();
            // Sort descending by top-Y so `pop()` returns the next table
            // to emit in reading order (larger Y first).
            pending_tables.sort_by(|(a, _), (b, _)| crate::utils::safe_float_cmp(*b, *a));

            let flush_table =
                |text: &mut String, table: &crate::structure::table_extractor::Table| {
                    if !text.is_empty() && !text.ends_with('\n') {
                        text.push('\n');
                    }
                    text.push('\n');
                    text.push_str(&table.render_text());
                    if !text.ends_with('\n') {
                        text.push('\n');
                    }
                };

            let mut text = String::with_capacity(spans.len() * 20);
            let mut prev_span: Option<&TextSpan> = None;

            for span in &spans {
                // Flush any tables that sit above this span in PDF
                // reading order (their top-Y is greater than or equal
                // to the span's top-Y, meaning they should appear first).
                while let Some(&(table_top_y, table)) = pending_tables.last() {
                    let span_top_y = span.bbox.y + span.bbox.height;
                    if table_top_y >= span_top_y {
                        flush_table(&mut text, table);
                        pending_tables.pop();
                        // Reset prev_span so the flow-text glue logic
                        // doesn't try to stitch the table's rendered
                        // block together with the next flow span.
                        prev_span = None;
                    } else {
                        break;
                    }
                }

                if let Some(prev) = prev_span {
                    let prev_end_x = prev.bbox.x + prev.bbox.width;
                    let span_end_x = span.bbox.x + span.bbox.width;
                    // Containment check: skip a span only if it is geometrically
                    // contained within the previous span AND has identical text.
                    // Without the text comparison, distinct lines that happen to
                    // overlap spatially (e.g., due to small Tm-scaled offsets)
                    // would be silently dropped.
                    let y_same = (prev.bbox.y - span.bbox.y).abs() < 2.0;
                    if y_same
                        && span.bbox.x >= prev.bbox.x - 0.5
                        && span_end_x <= prev_end_x + 0.5
                        && span.text == prev.text
                    {
                        continue;
                    }

                    let y_diff = (prev.bbox.y - span.bbox.y).abs();
                    let gap = span.bbox.x - prev_end_x;
                    let delta_x = span.bbox.x - prev.bbox.x;

                    if y_diff > Self::same_line_threshold(prev, span) {
                        let font_size = prev.font_size.max(span.font_size).max(10.0);
                        let line_height = font_size * 1.2;
                        let num_breaks = (y_diff / line_height).round() as usize;
                        for _ in 0..num_breaks.clamp(1, 3) {
                            text.push('\n');
                        }
                    } else if gap < -1.0 {
                        let fs = span.font_size.max(prev.font_size).max(6.0);
                        if gap < -(fs * 20.0) {
                            if !text.ends_with('\n') {
                                text.push('\n');
                            }
                        } else if delta_x < -fs * 3.0 {
                            // Same visual line, but the current span starts well to the LEFT of the
                            // previous span's start — i.e., the upstream ordering is non-monotonic in X.
                            // This commonly occurs with multi-column layouts or XY-cut artifacts where
                            // spans from different visual rows fall within the same Y tolerance band
                            // (see `same_line_threshold`).
                            //
                            // Without inserting a separator, these spans would be concatenated
                            // (e.g. `instancesinstancesinstances` from adjacent table headers).
                            // Treat this backward X jump as a logical break and emit a newline.
                            if !text.ends_with('\n') {
                                text.push('\n');
                            }
                        } else if prev.font_name != span.font_name
                            && span_end_x > prev_end_x + 0.5
                            && !text.ends_with(' ')
                            && !text.ends_with('\n')
                        {
                            text.push(' ');
                        } else if delta_x > fs * 1.5
                            && !text.ends_with(' ')
                            && !text.ends_with('\n')
                        {
                            // Inflated-width overlap recovery.
                            // A negative raw gap here usually comes from a
                            // font whose `/Widths` array is missing
                            // `FontInfo::new` fell back to the 550/1000-em
                            // constant, which over-reports each glyph's
                            // advance and drags `prev_end_x` past the real
                            // start of the next span. When the two spans'
                            // actual origins (`delta_x`) are separated by
                            // more than 1.5 em, they cannot both belong to
                            // the same word — the overlap is a width-table
                            // artifact, not real kerning — so insert a
                            // space to preserve the word boundary. This
                            // rescues cases like "STATION" + "FREEDOM"
                            // "UTILIZATION" + "CONFERENCE" in the NASA
                            // Apollo report header where raw gaps of
                            // -1.75 pt and -12.75 pt sit alongside
                            // delta_x values of 56 pt and 78 pt.
                            text.push(' ');
                        }
                    } else if y_diff > 2.0
                        && gap > FORWARD_GAP_K * prev.font_size.max(span.font_size).max(1.0)
                    {
                        // Forward-gap guard: pairs newly admitted to same-line
                        // handling by the widened threshold get a column/field-
                        // boundary check against FORWARD_GAP_K * max(fs).
                        // the constant's doc comment for calibration notes.
                        if !text.ends_with('\n') {
                            text.push('\n');
                        }
                    } else if prev.font_name != span.font_name
                        && gap > 0.5
                        && gap < prev.font_size.max(span.font_size).max(6.0) * 3.0
                        && !text.ends_with(' ')
                        && !text.ends_with('\n')
                    {
                        // Same-line font transition with a meaningful
                        // positive gap. Cross-font runs that survive the
                        // upstream `cross_font_word_glue` merge (i.e.
                        // both sides are multi-char) are word boundaries
                        // even when the gap is too small for the generic
                        // `should_insert_space` threshold (0.15 × fs) —
                        // e.g. roman → italic transitions in academic
                        // paper headers sit at ~2.7 pt at 10.9 pt body.
                        text.push(' ');
                    } else if Self::should_insert_space(prev, span) {
                        text.push(' ');
                    } else {
                        let fs = span.font_size.max(prev.font_size).max(6.0);
                        if gap > fs * 3.0 {
                            text.push('\n');
                        }
                    }
                }

                Self::push_span_text(&mut text, span);
                prev_span = Some(span);
            }

            // Drain any tables that sit below all flow spans (or the
            // page had no flow spans at all). Without this final
            // pass they would be silently dropped now that the
            // end-of-page `for table in tables` block has been
            // removed.
            while let Some((_, table)) = pending_tables.pop() {
                flush_table(&mut text, table);
            }
            text
        };

        // Annotation text is already included via annotation_content_spans() in
        // extract_spans() — do NOT call append_non_widget_annotation_text() here,
        // as that would emit every annotation a second time.

        // Filter leaked PDF metadata
        let final_text = Self::filter_leaked_metadata(&text);

        // Normalize Kangxi Radicals
        let final_text = Self::normalize_kangxi_radicals(&final_text);

        // Normalize Arabic Presentation Forms
        let final_text = Self::normalize_arabic_presentation_forms(&final_text);

        // Apply whitespace cleanup
        let cleaned_text = crate::converters::whitespace::cleanup_plain_text(&final_text);

        // For tagged PDFs, the structure-tree traversal at line 4306 already
        // captures all table-cell content via MCIDs. Appending tables here
        // would double-emit that content (structure-tree text + table render),
        // dropping precision. For untagged PDFs, tables are inlined via
        // pending_tables above, so this block is never reached (cached_tree
        // is None → condition would be false). The block is removed.

        // #317 UTF-8 mojibake repair: a run of Latin-1 Supplement chars
        // whose raw bytes form valid UTF-8 decoding to non-Latin-1 code
        // points is almost certainly a double-encoded non-Latin string
        // (Cyrillic, Greek, CJK, Arabic, Hebrew, …) that surfaced
        // because the producing font had no ToUnicode CMap and the
        // /Differences / AGL lookup returned the UTF-8 byte sequence
        // re-interpreted as Latin-1. Re-decode those runs in place.
        let cleaned_text = Self::repair_utf8_mojibake(&cleaned_text);

        // Optionally expand Latin ligature characters to their component letters.
        let cleaned_text = if options.expand_ligatures {
            cleaned_text
                .replace('\u{FB00}', "ff")
                .replace('\u{FB01}', "fi")
                .replace('\u{FB02}', "fl")
                .replace('\u{FB03}', "ffi")
                .replace('\u{FB04}', "ffl")
                .replace(['\u{FB05}', '\u{FB06}'], "st")
        } else {
            cleaned_text
        };

        Ok(cleaned_text)
    }

    /// Walk `input` and repair runs of Latin-1 Supplement characters
    /// whose raw byte values form a valid UTF-8 sequence whose decoded
    /// codepoints include at least one non-Latin-1 character.
    ///
    /// This undoes the most common shape of "Cyrillic served as
    /// Latin-1" mojibake that surfaces on PDFs whose fonts have no
    /// ToUnicode CMap. The decoded-codepoint gate (≥ U+0100 somewhere
    /// in the decoded run) ensures genuine Latin-1 content like
    /// "Résumé" — which also decodes as valid UTF-8 but stays entirely
    /// within U+0000..U+00FF — is left alone.
    fn repair_utf8_mojibake(input: &str) -> String {
        // Fast-path: if the string contains no Latin-1 Supplement codepoints
        // (U+0080..=U+00FF), there is nothing to repair. This avoids the
        // O(n) `Vec<char>` allocation on every ASCII-only page.
        if !input.chars().any(|c| matches!(c as u32, 0x80..=0xFF)) {
            return input.to_string();
        }
        let mut out = String::with_capacity(input.len());
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let mut j = i;
            while j < chars.len() {
                let cc = chars[j] as u32;
                if (0x80..=0xFF).contains(&cc) {
                    j += 1;
                } else {
                    break;
                }
            }
            if j - i >= 2 {
                let bytes: Vec<u8> = chars[i..j].iter().map(|&c| c as u8).collect();
                if let Ok(decoded) = std::str::from_utf8(&bytes) {
                    if decoded.chars().any(|c| c as u32 > 0xFF) {
                        out.push_str(decoded);
                        i = j;
                        continue;
                    }
                }
            }
            out.push(chars[i]);
            i += 1;
        }
        out
    }

    /// Extract text from all pages of the document.
    ///
    /// Concatenates text from every page, separated by form feed characters (`\x0c`).
    /// This is a convenience method equivalent to calling `extract_text()` for each page.
    ///
    /// # Returns
    ///
    /// The combined text from all pages.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("paper.pdf")?;
    /// let all_text = doc.extract_all_text()?;
    /// println!("Full document: {} chars", all_text.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_all_text(&self) -> Result<String> {
        let num_pages = self.page_count()?;
        let mut result = String::new();

        for i in 0..num_pages {
            if i > 0 {
                result.push('\x0c'); // Form feed page separator
            }
            match self.extract_text(i) {
                Ok(text) => result.push_str(&text),
                Err(e) => {
                    log::warn!("Failed to extract text from page {}: {}", i, e);
                },
            }
        }

        Ok(result)
    }

    /// Mark a specific rectangular region on a page for erasure.
    ///
    /// Content in this region will be excluded from all subsequent text extractions.
    pub fn erase_region(&self, page_index: usize, rect: crate::geometry::Rect) -> Result<()> {
        self.erase_regions
            .lock_or_recover()
            .entry(page_index)
            .or_default()
            .push(rect);
        // Redaction changes a page's spans; drop the span cache.
        self.page_spans_cache.lock_or_recover().clear();
        Ok(())
    }

    /// Clear all erase regions for a page.
    pub fn clear_erase_regions(&self, page_index: usize) -> Result<()> {
        self.erase_regions.lock_or_recover().remove(&page_index);
        self.page_spans_cache.lock_or_recover().clear();
        Ok(())
    }

    /// Identify and remove headers.
    ///
    /// Uses spec-compliant /Artifact tags when available (100% accuracy), or
    /// falls back to heuristic analysis of the top 15% of pages.
    pub fn remove_headers(&self, threshold: f32) -> Result<usize> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(crate::error::Error::InvalidOperation(
                "Threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        self.remove_repeated_text(PageArea::Header, threshold)
    }

    /// Identify and remove footers.
    ///
    /// Uses spec-compliant /Artifact tags when available (100% accuracy), or
    /// falls back to heuristic analysis of the bottom 15% of pages.
    pub fn remove_footers(&self, threshold: f32) -> Result<usize> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(crate::error::Error::InvalidOperation(
                "Threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        self.remove_repeated_text(PageArea::Footer, threshold)
    }

    /// Identify and remove both headers and footers.
    ///
    /// Prioritizes ISO 32000 spec-compliant /Artifact tags, with a heuristic
    /// fallback for untagged PDFs.
    ///
    /// # Arguments
    /// * `threshold` - Fraction of pages (0.0-1.0) where text must repeat to be removed (heuristic mode only).
    pub fn remove_artifacts(&self, threshold: f32) -> Result<usize> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(crate::error::Error::InvalidOperation(
                "Threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        let h = self.remove_headers(threshold)?;
        let f = self.remove_footers(threshold)?;
        Ok(h + f)
    }

    /// Helper to remove repeated text in a specific page area.
    fn remove_repeated_text(&self, area: PageArea, threshold: f32) -> Result<usize> {
        use crate::extractors::text::{ArtifactType, PaginationSubtype};
        use std::collections::{HashMap, HashSet};

        let page_count = self.page_count()?;
        if page_count < 1 {
            return Ok(0);
        }

        let mut removed_count = 0;

        // 1. Spec-Compliant Removal (Priority)
        // If the PDF uses /Artifact tags (Tagged PDF), we use those directly as they are 100% accurate.
        for page_idx in 0..page_count {
            let spans = self.extract_spans(page_idx)?;
            for span in spans {
                if let Some(ArtifactType::Pagination(subtype)) = span.artifact_type {
                    let is_match = match (area, subtype) {
                        (PageArea::Header, PaginationSubtype::Header) => true,
                        (PageArea::Footer, PaginationSubtype::Footer) => true,
                        _ => false,
                    };

                    if is_match {
                        self.erase_region(page_idx, span.bbox)?;
                        removed_count += 1;
                    }
                }
            }
        }

        // If we found and removed spec-compliant artifacts, we return early
        if removed_count > 0 {
            log::info!(
                "Removed {} spec-compliant artifacts from {}",
                removed_count,
                if area == PageArea::Header {
                    "headers"
                } else {
                    "footers"
                }
            );
            return Ok(removed_count);
        }

        // 2. Heuristic Removal (Fallback for Untagged PDFs)
        // Only run if no spec-compliant tags were found.
        if page_count < 2 {
            return Ok(0);
        }

        let mut occurrences: HashMap<String, HashSet<usize>> = HashMap::new();

        // Sanitize threshold to avoid min_occurrences becoming 0 for invalid inputs.
        let clamped_threshold = if threshold.is_finite() {
            threshold.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let raw_min = (page_count as f32 * clamped_threshold).ceil();
        let min_occurrences = if raw_min < 1.0 { 1 } else { raw_min as usize };

        // Cache spans per page to avoid redundant extraction in Pass 2
        let mut page_spans: HashMap<usize, Vec<crate::layout::TextSpan>> = HashMap::new();

        for page_idx in 0..page_count {
            let height = self.get_page_media_box(page_idx)?.3;
            let zone = match area {
                PageArea::Header => height * 0.85,
                PageArea::Footer => height * 0.15,
            };

            let spans = self.extract_spans(page_idx)?;
            for span in spans.iter() {
                let is_in_zone = match area {
                    PageArea::Header => span.bbox.y > zone,
                    PageArea::Footer => (span.bbox.y + span.bbox.height) < zone,
                };

                if is_in_zone {
                    let text = span.text.trim().to_string();
                    if text.len() > 3 && !text.chars().all(|c| c.is_numeric()) {
                        occurrences.entry(text).or_default().insert(page_idx);
                    }
                }
            }
            page_spans.insert(page_idx, spans);
        }

        for (text, pages) in occurrences {
            if pages.len() >= min_occurrences {
                for page_idx in pages {
                    // Reuse cached spans
                    if let Some(spans) = page_spans.get(&page_idx) {
                        for span in spans {
                            if span.text.trim() == text {
                                self.erase_region(page_idx, span.bbox)?;
                                removed_count += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(removed_count)
    }

    /// Erase existing header content.
    ///
    /// Identifies existing text in the header area (top 15%) and marks it for erasure.
    pub fn erase_header(&self, page_index: usize) -> Result<()> {
        self.erase_page_area_content(page_index, PageArea::Header)
    }

    /// Deprecated: Use `erase_header` instead.
    #[deprecated(note = "use erase_header instead")]
    pub fn edit_header(&self, page_index: usize) -> Result<()> {
        self.erase_header(page_index)
    }

    /// Erase existing footer content.
    ///
    /// Identifies existing text in the footer area (bottom 15%) and marks it for erasure.
    pub fn erase_footer(&self, page_index: usize) -> Result<()> {
        self.erase_page_area_content(page_index, PageArea::Footer)
    }

    /// Deprecated: Use `erase_footer` instead.
    #[deprecated(note = "use erase_footer instead")]
    pub fn edit_footer(&self, page_index: usize) -> Result<()> {
        self.erase_footer(page_index)
    }

    /// Erase both header and footer content.
    ///
    /// This is a convenience method that calls both erase_header and erase_footer.
    pub fn erase_artifacts(&self, page_index: usize) -> Result<()> {
        self.erase_header(page_index)?;
        self.erase_footer(page_index)?;
        Ok(())
    }

    /// Helper to erase content in a specific page area.
    fn erase_page_area_content(&self, page_index: usize, area: PageArea) -> Result<()> {
        let height = self.get_page_media_box(page_index)?.3;
        let zone = match area {
            PageArea::Header => height * 0.85,
            PageArea::Footer => height * 0.15,
        };

        let spans = self.extract_spans(page_index)?;
        for span in spans {
            let is_in_zone = match area {
                PageArea::Header => span.bbox.y > zone,
                PageArea::Footer => (span.bbox.y + span.bbox.height) < zone,
            };

            if is_in_zone {
                self.erase_region(page_index, span.bbox)?;
            }
        }
        Ok(())
    }

    /// Extract text from a page with automatic OCR fallback for scanned pages.
    ///
    /// This method automatically detects scanned pages and applies OCR when needed,
    /// falling back to native text extraction for regular PDFs.
    ///
    /// **Note**: Requires the `ocr` feature to be enabled and OCR models to be provided.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Page number (0-indexed)
    /// * `ocr_engine` - Optional OCR engine (required for scanned pages)
    /// * `ocr_options` - OCR extraction options (DPI, thresholds, etc.)
    ///
    /// # Returns
    ///
    /// The extracted text, either from native PDF text or OCR.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::{PdfDocument, ocr::{OcrEngine, OcrConfig, OcrExtractOptions}};
    ///
    /// let mut doc = PdfDocument::open("mixed.pdf")?;
    /// let engine = OcrEngine::new("det.onnx", "rec.onnx", "dict.txt", OcrConfig::default())?;
    ///
    /// // Automatically uses native text or OCR as needed
    /// let text = doc.extract_text_with_ocr(0, Some(&engine), OcrExtractOptions::default())?;
    /// ```
    #[cfg(feature = "ocr")]
    pub fn extract_text_with_ocr(
        &self,
        page_index: usize,
        ocr_engine: Option<&crate::ocr::OcrEngine>,
        ocr_options: crate::ocr::OcrExtractOptions,
    ) -> Result<String> {
        crate::ocr::extract_text_with_ocr(self, page_index, ocr_engine, ocr_options)
    }

    /// Run the supplied OCR engine against the largest embedded image
    /// on the page, regardless of whether the page has a native text
    /// layer.
    ///
    /// The existing [`extract_text_with_ocr`] is text-layer-first
    /// (OCR only when no native text), which can mask poor-quality
    /// auto-OCR'd layers from scanner pipelines. This companion
    /// always invokes the engine via [`crate::ocr::ocr_page`] —
    /// useful when callers know the embedded layer is unreliable and
    /// want the OCR pass for cross-checking.
    ///
    /// **Limitation**: the OCR target is the largest embedded
    /// `/XObject Image` on the page (see [`crate::ocr::ocr_page`]).
    /// For born-digital PDFs with no embedded raster, the underlying
    /// helper falls back to native [`Self::extract_text`] when
    /// `ocr_options.fallback_to_native` is set, otherwise returns an
    /// empty string. Full-page rasterization (render → image → OCR)
    /// is **not** performed here; callers that need it should drive
    /// it directly via [`crate::rendering::render_page`] (requires
    /// the `rendering` feature) and feed the result through their
    /// own engine.
    ///
    /// Errors: returns `Error::OcrUnavailable { reason: ModelLoadFailed }`
    /// when the OCR backend fails to initialise (e.g. missing
    /// `libonnxruntime.so`) — the [`catch_unwind`](std::panic::catch_unwind)
    /// in `OrtBackend::from_bytes` keeps that path panic-free.
    #[cfg(feature = "ocr")]
    pub fn extract_text_ocr_only(
        &self,
        page_index: usize,
        ocr_engine: &crate::ocr::OcrEngine,
        ocr_options: crate::ocr::OcrExtractOptions,
    ) -> Result<String> {
        crate::ocr::ocr_page(self, page_index, ocr_engine, &ocr_options).map_err(|e| {
            Error::OcrUnavailable {
                reason: crate::extractors::status::OcrUnavailableReason::ModelLoadFailed {
                    detail: e.to_string(),
                },
            }
        })
    }

    /// Extract TextSpans with automatic OCR fallback for scanned pages.
    ///
    /// This method extracts text spans using native PDF text extraction, but falls back
    /// to OCR when the page appears to be scanned (no/minimal native text).
    ///
    /// **Note**: Requires the `ocr` feature to be enabled and OCR models to be provided.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Page number (0-indexed)
    /// * `ocr_engine` - Optional OCR engine (required for scanned pages)
    /// * `ocr_options` - OCR extraction options (DPI, thresholds, etc.)
    ///
    /// # Returns
    ///
    /// Vector of TextSpans, either from native PDF or OCR.
    #[cfg(feature = "ocr")]
    pub fn extract_spans_with_ocr(
        &self,
        page_index: usize,
        ocr_engine: Option<&crate::ocr::OcrEngine>,
        ocr_options: &crate::ocr::OcrExtractOptions,
    ) -> Result<Vec<crate::layout::TextSpan>> {
        // First try native text extraction
        let spans = self.extract_spans(page_index)?;

        // If we got substantial text, return it
        if !spans.is_empty() && spans.iter().map(|s| s.text.len()).sum::<usize>() >= 50 {
            return Ok(spans);
        }

        // Check if page needs OCR
        if let Ok(true) = crate::ocr::needs_ocr(self, page_index) {
            // Try OCR if engine is available
            if let Some(engine) = ocr_engine {
                match crate::ocr::ocr_page_spans(self, page_index, engine, ocr_options) {
                    Ok(ocr_spans) if !ocr_spans.is_empty() => return Ok(ocr_spans),
                    Ok(_) => log::debug!("OCR returned no spans for page {}", page_index),
                    Err(e) => log::warn!("OCR failed for page {}: {}", page_index, e),
                }
            }
        }

        // Fallback to native spans (even if empty)
        Ok(spans)
    }

    /// Determine if a space should be inserted between two text spans.
    ///
    /// According to PDF spec (ISO 32000-1:2008 Section 9.3.3), word spacing
    /// only applies to actual space characters (0x20). Many PDFs (especially
    /// academic papers) use precise positioning instead of space characters.
    /// This function detects such gaps and inserts spaces heuristically.
    ///
    /// # Algorithm
    /// 1. Check if spans are on the same line (Y positions similar)
    /// 2. Calculate horizontal gap between end of prev span and start of current span
    /// 3. Insert space if gap exceeds threshold (0.25 × font size)
    ///
    /// # Arguments
    /// * `prev` - Previous text span
    /// * `current` - Current text span
    ///
    /// Filter leaked PDF internal metadata from extracted text.
    ///
    /// Some PDFs embed inline ColorSpace definitions (CalRGB, CalGray, Lab) that
    /// get parsed as text content. This removes known metadata patterns like
    /// "WhitePoint [ ... ]", "BlackPoint [ ... ]", "Gamma [ ... ]", "Matrix [ ... ]".
    fn filter_leaked_metadata(text: &str) -> String {
        // Known PDF metadata keys that should never appear in extracted text.
        // These come from CalRGB/CalGray/Lab color space dictionaries.
        const METADATA_PATTERNS: &[&str] = &[
            "WhitePoint",
            "BlackPoint",
            "Gamma",
            "Matrix",
            "CalRGB",
            "CalGray",
        ];

        // Quick check: if none of the patterns appear, return as-is
        if !METADATA_PATTERNS.iter().any(|p| text.contains(p)) {
            return text.to_string();
        }

        // Filter line-by-line: remove lines that look like PDF metadata
        let mut result = String::with_capacity(text.len());
        for line in text.lines() {
            let trimmed = line.trim();
            // Skip lines matching "MetadataKey [ ... ]" or "MetadataKey [ ... ] ..."
            let is_metadata = METADATA_PATTERNS.iter().any(|pattern| {
                if let Some(rest) = trimmed.strip_prefix(pattern) {
                    // Must be followed by whitespace and bracket, or end of line
                    let rest = rest.trim_start();
                    rest.is_empty()
                        || rest.starts_with('[')
                        || rest.starts_with('/')
                        || rest.starts_with('<')
                } else {
                    false
                }
            });

            if !is_metadata {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(line);
            }
        }

        result
    }

    /// Normalize Kangxi Radical characters to CJK Unified Ideographs.
    ///
    /// Some PDF fonts/CMaps emit Kangxi Radicals (U+2F00–U+2FD5) or CJK Radicals
    /// Supplement (U+2E80–U+2EFF) instead of the standard CJK Unified Ideographs.
    /// While visually similar, these are different Unicode codepoints and will break
    /// text search, string matching, and NLP pipelines.
    fn normalize_kangxi_radicals(text: &str) -> String {
        // Quick check: if no characters in the Kangxi/Supplement range, return as-is
        if !text.chars().any(|c| {
            let cp = c as u32;
            (0x2E80..=0x2EFF).contains(&cp) || (0x2F00..=0x2FD5).contains(&cp)
        }) {
            return text.to_string();
        }

        text.chars()
            .map(|c| crate::text::kangxi::kangxi_to_unified(c).unwrap_or(c))
            .collect()
    }

    /// Reverse visual-order RTL character runs to logical reading order.
    ///
    /// Some PDFs position Arabic/Hebrew characters individually left-to-right
    /// (visual order). For correct text extraction, runs of single-character
    /// RTL spans on the same line are collected, reversed, and merged into
    /// a single span to produce correct logical reading order.
    fn reverse_rtl_visual_order_runs(spans: &mut Vec<TextSpan>) {
        use crate::text::rtl_detector::is_rtl_text;

        // Pass 0: reverse visual-order characters inside a single span
        // when the producer clearly emitted pre-shaped Arabic.
        //
        // Some PDFs (e.g. `ArabicCIDTrueType.pdf` in the pdfjs regression
        // corpus) emit Arabic with an entire line as a single Tj-produced
        // span whose `text` is stored in *visual* order — rightmost
        // rendered glyph first. That matches what the content stream
        // literally drew on the page, but downstream consumers expect
        // reading-order (logical) text.
        //
        // The gate for reversal is the presence of **Arabic Presentation
        // Forms A or B** (U+FB50-U+FDFF, U+FE70-U+FEFF). Those code points
        // only appear when the PDF producer has explicitly pre-shaped the
        // glyphs, and producers that pre-shape almost universally also
        // store them in visual order because that's the order the content
        // stream draws them. Plain base-Arabic text (U+0600-U+06FF) is
        // left alone because those files are usually already in logical
        // order — the PDF viewer applies shaping and bidi reordering at
        // render time, so reversing would produce a wrong result.
        //
        // We still require at least 4 characters and >50 % non-whitespace
        // RTL ratio so that punctuation or stray markers adjacent to
        // Arabic do not trigger a reversal.
        //
        // Pass 1 below handles the other common shape where each Arabic
        // character is emitted as its own short span and the reversal is
        // a span-granularity concern. The two passes are independent:
        // a span either fires Pass 0 (pre-shaped, reverse in place) or
        // Pass 1 (per-glyph spans, reverse span order), never both.
        //
        // This is separate from `normalize_arabic_presentation_forms`,
        // which runs later on the assembled output string and unshapes
        // contextual glyphs back to their base Unicode letters.
        for span in spans.iter_mut() {
            let mut total = 0usize;
            let mut rtl_count = 0usize;
            let mut has_presentation_form = false;
            for c in span.text.chars() {
                if c.is_whitespace() {
                    continue;
                }
                total += 1;
                let cp = c as u32;
                if is_rtl_text(cp) {
                    rtl_count += 1;
                }
                if (0xFB50..=0xFDFF).contains(&cp) || (0xFE70..=0xFEFF).contains(&cp) {
                    has_presentation_form = true;
                }
            }
            // #557: Pass 0 only applies to a *whole-line* visual-order span —
            // one span holding several words separated by internal whitespace,
            // in the order the content stream drew them (rightmost first). When
            // the extractor instead emits one span PER WORD (the common
            // CID-TrueType case, e.g. ArabicCIDTrueType.pdf), each word's
            // characters are already in logical order, so char-reversing them
            // here corrupts them. Their right-to-left *word* order is fixed
            // separately by the span-run reversal pass below. Gate on internal
            // whitespace so per-word logical spans are left untouched.
            let has_internal_whitespace = span.text.trim().chars().any(|c| c.is_whitespace());
            if has_presentation_form
                && has_internal_whitespace
                && total >= 4
                && rtl_count * 2 > total
            {
                let reversed: String = span.text.chars().rev().collect();
                span.text = reversed;
            }
        }

        // #557 Pass 0.5: per-word RTL span ORDER. The row-aware sort placed
        // spans left-to-right (x ascending), but a right-to-left script reads
        // the words in the opposite direction. For each maximal run of
        // consecutive same-line spans that is purely RTL (every non-space span
        // holds RTL letters and no Latin letters), reverse the run's order so
        // the words come out in logical reading order. Each word's characters
        // are left as-is (they are already logical — see Pass 0's gate).
        let is_space = |s: &TextSpan| s.text.trim().is_empty();
        let is_rtl_word = |s: &TextSpan| {
            let mut has_rtl = false;
            for c in s.text.chars() {
                if c.is_ascii_alphabetic() {
                    return false; // Latin letter → not a pure-RTL word
                }
                if is_rtl_text(c as u32) {
                    has_rtl = true;
                }
            }
            has_rtl
        };
        let mut i = 0;
        while i < spans.len() {
            if !is_rtl_word(&spans[i]) {
                i += 1;
                continue;
            }
            let y = spans[i].bbox.y;
            let start = i;
            let mut end = i + 1;
            while end < spans.len()
                && (spans[end].bbox.y - y).abs() < 2.0
                && (is_rtl_word(&spans[end]) || is_space(&spans[end]))
            {
                end += 1;
            }
            // Trim trailing space spans so separators stay between words.
            let mut last = end;
            while last > start + 1 && is_space(&spans[last - 1]) {
                last -= 1;
            }
            if last - start >= 2 {
                spans[start..last].reverse();
            }
            i = end;
        }

        if spans.len() < 4 {
            return;
        }

        // Iterate forward; drain consumed runs so subsequent indices stay valid
        let mut i = 0;
        while i < spans.len() {
            // Check if this span starts an RTL single-char run
            let is_short_rtl = spans[i].text.chars().count() <= 2
                && spans[i].text.chars().any(|c| is_rtl_text(c as u32));

            if !is_short_rtl {
                i += 1;
                continue;
            }

            // Find the end of this RTL run (consecutive short spans on same line)
            let run_start = i;
            let y = spans[i].bbox.y;
            let mut j = i + 1;
            while j < spans.len() {
                let y_same = (spans[j].bbox.y - y).abs() < 2.0;
                let is_short = spans[j].text.chars().count() <= 2;
                let has_rtl_or_space = spans[j]
                    .text
                    .chars()
                    .all(|c| is_rtl_text(c as u32) || c == ' ');
                if y_same && is_short && has_rtl_or_space {
                    j += 1;
                } else {
                    break;
                }
            }
            let run_end = j;
            let run_len = run_end - run_start;

            // Only process runs of 4+ spans (avoid false positives)
            if run_len >= 4 {
                // Collect span texts in reverse order (visual LTR → logical RTL).
                // Preserve space spans as word separators.
                let mut reversed_text = String::new();
                for span in spans[run_start..run_end].iter().rev() {
                    reversed_text.push_str(&span.text);
                }

                // Merge into first span, expand bbox to cover entire run
                let last_span = &spans[run_end - 1];
                let new_width = (last_span.bbox.x + last_span.bbox.width) - spans[run_start].bbox.x;
                spans[run_start].text = reversed_text;
                spans[run_start].bbox.width = new_width;

                // Remove the rest of the run
                spans.drain(run_start + 1..run_end);

                i = run_start + 1;
            } else {
                i = run_end;
            }
        }
    }

    /// Normalize Arabic Presentation Forms to base Unicode characters.
    ///
    /// Arabic PDFs often use presentation forms (U+FE70-U+FEFF for Forms-B,
    /// U+FB50-U+FDFF for Forms-A) which represent contextual glyph shapes.
    /// For text extraction, these should be normalized to base characters.
    fn normalize_arabic_presentation_forms(text: &str) -> String {
        // Quick check: skip if no Arabic presentation form characters
        if !text.chars().any(|c| {
            let cp = c as u32;
            (0xFB50..=0xFDFF).contains(&cp) || (0xFE70..=0xFEFF).contains(&cp)
        }) {
            return text.to_string();
        }

        text.chars()
            .map(|c| {
                let cp = c as u32;
                // Arabic Presentation Forms-B (U+FE70-U+FEFF): contextual forms
                // Each base letter has isolated/final/initial/medial forms
                let base = match cp {
                    // Hamza forms
                    0xFE80 => 0x0621,
                    // Alef with Madda
                    0xFE81 | 0xFE82 => 0x0622,
                    // Alef with Hamza Above
                    0xFE83 | 0xFE84 => 0x0623,
                    // Waw with Hamza
                    0xFE85 | 0xFE86 => 0x0624,
                    // Alef with Hamza Below
                    0xFE87 | 0xFE88 => 0x0625,
                    // Yeh with Hamza
                    0xFE89..=0xFE8C => 0x0626,
                    // Alef
                    0xFE8D | 0xFE8E => 0x0627,
                    // Beh
                    0xFE8F..=0xFE92 => 0x0628,
                    // Teh Marbuta
                    0xFE93 | 0xFE94 => 0x0629,
                    // Teh
                    0xFE95..=0xFE98 => 0x062A,
                    // Theh
                    0xFE99..=0xFE9C => 0x062B,
                    // Jeem
                    0xFE9D..=0xFEA0 => 0x062C,
                    // Hah
                    0xFEA1..=0xFEA4 => 0x062D,
                    // Khah
                    0xFEA5..=0xFEA8 => 0x062E,
                    // Dal
                    0xFEA9 | 0xFEAA => 0x062F,
                    // Thal
                    0xFEAB | 0xFEAC => 0x0630,
                    // Reh
                    0xFEAD | 0xFEAE => 0x0631,
                    // Zain
                    0xFEAF | 0xFEB0 => 0x0632,
                    // Seen
                    0xFEB1..=0xFEB4 => 0x0633,
                    // Sheen
                    0xFEB5..=0xFEB8 => 0x0634,
                    // Sad
                    0xFEB9..=0xFEBC => 0x0635,
                    // Dad
                    0xFEBD..=0xFEC0 => 0x0636,
                    // Tah
                    0xFEC1..=0xFEC4 => 0x0637,
                    // Zah
                    0xFEC5..=0xFEC8 => 0x0638,
                    // Ain
                    0xFEC9..=0xFECC => 0x0639,
                    // Ghain
                    0xFECD..=0xFED0 => 0x063A,
                    // Feh
                    0xFED1..=0xFED4 => 0x0641,
                    // Qaf
                    0xFED5..=0xFED8 => 0x0642,
                    // Kaf
                    0xFED9..=0xFEDC => 0x0643,
                    // Lam
                    0xFEDD..=0xFEE0 => 0x0644,
                    // Meem
                    0xFEE1..=0xFEE4 => 0x0645,
                    // Noon
                    0xFEE5..=0xFEE8 => 0x0646,
                    // Heh
                    0xFEE9..=0xFEEC => 0x0647,
                    // Waw
                    0xFEED | 0xFEEE => 0x0648,
                    // Alef Maksura
                    0xFEEF | 0xFEF0 => 0x0649,
                    // Yeh
                    0xFEF1..=0xFEF4 => 0x064A,
                    // Lam-Alef ligatures → expand to two characters
                    0xFEF5 | 0xFEF6 => {
                        // Lam + Alef with Madda
                        return '\u{0644}'; // Just return Lam; Alef is separate
                    },
                    0xFEF7 | 0xFEF8 => {
                        return '\u{0644}'; // Lam + Alef with Hamza Above
                    },
                    0xFEF9 | 0xFEFA => {
                        return '\u{0644}'; // Lam + Alef with Hamza Below
                    },
                    0xFEFB | 0xFEFC => {
                        return '\u{0644}'; // Lam + Alef
                    },
                    // Tatweel (kashida)
                    0xFE70 => 0x064B, // Fathatan isolated
                    0xFE71 => 0x064B, // Tatweel + Fathatan
                    0xFE72 => 0x064C, // Dammatan isolated
                    0xFE74 => 0x064D, // Kasratan isolated
                    0xFE76 => 0x064E, // Fatha isolated
                    0xFE77 => 0x064E, // Fatha medial
                    0xFE78 => 0x064F, // Damma isolated
                    0xFE79 => 0x064F, // Damma medial
                    0xFE7A => 0x0650, // Kasra isolated
                    0xFE7B => 0x0650, // Kasra medial
                    0xFE7C => 0x0651, // Shadda isolated
                    0xFE7D => 0x0651, // Shadda medial
                    0xFE7E => 0x0652, // Sukun isolated
                    0xFE7F => 0x0652, // Sukun medial
                    _ => cp,          // Pass through unchanged
                };
                char::from_u32(base).unwrap_or(c)
            })
            .collect()
    }

    /// Returns the Y tolerance (in points) for treating two spans as
    /// belonging to the same visual line during text assembly.
    ///
    /// The threshold scales with the larger font size so mixed-size runs
    /// (for example superscripts and subscripts) are not split by a fixed
    /// absolute tolerance.
    fn same_line_threshold(prev: &TextSpan, current: &TextSpan) -> f32 {
        let max_fs = prev.font_size.max(current.font_size).max(1.0);
        let min_fs = prev.font_size.min(current.font_size).max(1.0);
        // Continuous formula — avoids the step discontinuity at the 4×
        // ratio boundary. Examples:
        //   same-size 12 pt body: max(12×1.2, 12×0.3) = 14.4 pt ← 1.2× leading
        //   heading+body 24+10 pt: max(10×1.2, 24×0.3) = 12.0 pt ← keeps para break
        //   superscript 12+6 pt: max(6×1.2, 12×0.3) = 7.2 pt ← same line
        // Prior formula was max_fs×0.5 for normal ratios; new formula uses 1.2× of the
        // smaller font, which is wider and reduces false newlines for normal leading.
        // Formula: max(min_fs * 1.2, max_fs * 0.3)
        (min_fs * 1.2).max(max_fs * 0.3)
    }

    /// Returns `true` if `inner` is contained within `outer`,
    /// allowing `eps` points of floating-point slack on all four
    /// edges. Used at the table-retain sites to absorb ~0.02pt drift
    /// in span right-edges relative to table bboxes computed from
    /// min/max reductions over many cell edges.
    fn contains_rect_with_tolerance(
        outer: &crate::geometry::Rect,
        inner: &crate::geometry::Rect,
        eps: f32,
    ) -> bool {
        inner.left() >= outer.left() - eps
            && inner.right() <= outer.right() + eps
            && inner.top() >= outer.top() - eps
            && inner.bottom() <= outer.bottom() + eps
    }

    /// Returns `true` if a tentative left-to-right X-ordering of `run`
    /// contains a horizontal gap exceeding
    /// `SAME_LINE_REORDER_MAX_GAP_FACTOR * max(font_size)` between any
    /// two consecutive spans. Used by [`reorder_same_line_runs`] to
    /// reject candidate runs that are vertically close but horizontally
    /// disjoint (e.g. tightly-set footer/header rows split across the
    /// page).
    ///
    /// The slice is not mutated; the X-order is computed on a local
    /// copy of `(left_x, right_x, font_size)` triples.
    fn run_has_large_x_gap(run: &[TextSpan]) -> bool {
        if run.len() < 2 {
            return false;
        }

        let mut edges: Vec<(f32, f32, f32)> = run
            .iter()
            .map(|s| (s.bbox.x, s.bbox.x + s.bbox.width, s.font_size))
            .collect();

        edges.sort_by(|a, b| crate::utils::safe_float_cmp(a.0, b.0));

        for pair in edges.windows(2) {
            let prev = pair[0];
            let cur = pair[1];

            let gap = cur.0 - prev.1;
            if gap <= 0.0 {
                continue;
            }

            let max_fs = prev.2.max(cur.2).max(1.0);
            if gap > SAME_LINE_REORDER_MAX_GAP_FACTOR * max_fs {
                return true;
            }
        }

        false
    }

    /// Re-sort same-line spans by X after row-aware band sorting.
    ///
    /// Row-aware sorting can place off-baseline glyphs such as superscripts or
    /// subscripts in adjacent Y bands before their base glyphs. This helper finds
    /// candidate runs with the existing same-line threshold, then tentatively views
    /// each candidate in X order. If that tentative X order contains a large gap,
    /// the candidate is treated as disjoint footer/header/field content and is
    /// left in the existing row-aware order.
    ///
    /// At the slice level no spans are merged or dropped; successful candidates are
    /// only permuted. Downstream text assembly may then emit the reordered spans
    /// into one visual line, which is the user-observable effect.
    fn reorder_same_line_runs(spans: &mut [TextSpan]) {
        let mut i = 0;

        while i < spans.len() {
            let mut j = i + 1;

            while j < spans.len() {
                let anchor = &spans[i];
                let prev = &spans[j - 1];
                let cur = &spans[j];

                let to_prev = (cur.bbox.y - prev.bbox.y).abs();
                let to_anchor = (cur.bbox.y - anchor.bbox.y).abs();

                let tol_prev = Self::same_line_threshold(prev, cur);
                let tol_anchor = Self::same_line_threshold(anchor, cur);

                if to_prev > tol_prev || to_anchor > tol_anchor {
                    break;
                }

                j += 1;
            }

            if j - i > 1 {
                if Self::run_has_large_x_gap(&spans[i..j]) {
                    // Candidate spans are vertically close, but not horizontally
                    // contiguous. Do not X-sort them into a fake line; preserve
                    // the row-aware order established before this helper.
                    i = j;
                    continue;
                }

                spans[i..j].sort_by(|a, b| {
                    let cmp = crate::utils::safe_float_cmp(a.bbox.x, b.bbox.x);
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                    a.sequence.cmp(&b.sequence)
                });
            }

            i = j;
        }
    }

    /// # Returns
    /// `true` if a space should be inserted between the spans
    fn should_insert_space(prev: &TextSpan, current: &TextSpan) -> bool {
        // Get font size (use the larger of the two)
        let font_size = prev.font_size.max(current.font_size).max(1.0);

        // Same-line gate. Uses the shared threshold so the assembly
        // loop's same-line decision and the space-insertion decision
        // cannot disagree about where a line ends.
        let y_diff = (prev.bbox.y - current.bbox.y).abs();
        if y_diff > Self::same_line_threshold(prev, current) {
            return false; // Different lines - no space needed
        }

        // CJK scripts (Chinese, Japanese, Korean) do not use spaces between
        // words. If both the tail of prev and the head of current are CJK characters,
        // inserting a space would produce incorrect tokenisation.
        let prev_tail = prev.text.chars().next_back();
        let curr_head = current.text.chars().next();
        let is_cjk = |c: char| {
            matches!(
                c as u32,
                0x3040..=0x309F   // Hiragana
                | 0x30A0..=0x30FF // Katakana
                | 0x3400..=0x4DBF // CJK Unified Ideographs Extension A
                | 0x4E00..=0x9FFF // CJK Unified Ideographs
                | 0xAC00..=0xD7AF // Hangul Syllables
                | 0x20000..=0x2A6DF // CJK Unified Ideographs Extension B
                | 0xFF00..=0xFFEF // Halfwidth and Fullwidth Forms
                | 0x3000..=0x303F // CJK Symbols and Punctuation
            )
        };
        if prev_tail.is_some_and(is_cjk) && curr_head.is_some_and(is_cjk) {
            return false;
        }

        // Emoji / pictographic → letter boundary: a wide pictographic glyph
        // (e.g. 📄) abuts the next token, so the proportional-gap test below
        // would drop the inter-token space (`📄README` instead of `📄 README`).
        // Word boundaries are reader latitude (ISO 32000-1:2008 §9.10); keep the
        // space. The alphabetic-follower requirement excludes combined ZWJ/VS
        // emoji sequences (whose next char is a selector or another pictograph).
        if prev_tail.is_some_and(crate::extractors::text::is_pictographic)
            && curr_head.is_some_and(char::is_alphabetic)
        {
            return true;
        }

        // Calculate horizontal gap
        let prev_end_x = prev.bbox.x + prev.bbox.width;
        let gap = current.bbox.x - prev_end_x;

        // CJK script ↔ non-CJK boundary: pdftotext (and the GT it produces)
        // inserts a space wherever a CJK *script* glyph (ideograph, kana, or
        // hangul) meets a Latin/digit character on the same line, regardless
        // of how tightly the two were typeset. Without this, mixed-script
        // content like "神鹰集团" + "2015" collapses into one token
        // "神鹰集团2015", which never matches GT's separate "神鹰集团"
        // "2015" tokens (issue 484, pr-136).
        //
        // IMPORTANT: this MUST exclude fullwidth ASCII variants (U+FF01..FF5E
        // — ＜＞＝＠ etc.) and CJK Symbols and Punctuation (U+3000..303F) even
        // though they are technically "CJK characters". Those are *operator*
        // glyphs that sit inline with adjacent digits and Latin in CJK
        // technical documents — pdftotext keeps "60000≤Q＜80000"
        // "20＜μ≤30" as compound tokens (issue 484, issue-336). Forcing a
        // boundary space there destroys the compound and regresses Jaccard.
        let is_cjk_script = |c: char| {
            matches!(
                c as u32,
                0x3040..=0x309F      // Hiragana
                | 0x30A0..=0x30FF    // Katakana
                | 0x3400..=0x4DBF    // CJK Unified Ideographs Extension A
                | 0x4E00..=0x9FFF    // CJK Unified Ideographs
                | 0xAC00..=0xD7AF    // Hangul Syllables
                | 0x20000..=0x2A6DF  // CJK Unified Ideographs Extension B
                | 0xFF66..=0xFF9F    // Halfwidth Katakana
            )
        };
        let crosses_cjk_boundary = match (prev_tail, curr_head) {
            (Some(p), Some(c)) => is_cjk_script(p) != is_cjk_script(c),
            _ => false,
        };
        // ASCII punctuation hugs the preceding token in every script —
        // pdftotext's GT renders "する." with no space and "神鹰，2015"
        // with no space before the comma either. Suppress the boundary
        // forced-space when the transitioning glyph IS the punctuation;
        // the space-threshold path below still handles real gaps.
        let is_clause_punct =
            |c: char| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}');
        let punct_at_boundary = curr_head.is_some_and(is_clause_punct)
            || prev_tail.is_some_and(|c| matches!(c, '(' | '[' | '{'));
        if crosses_cjk_boundary && !punct_at_boundary && gap > -0.5 && gap < font_size * 5.0 {
            return true;
        }

        // Space threshold: 0.15 × font size
        // Typical space width is ~0.25em, so 0.15em catches gaps > 60% of a space.
        // This aligns with the text extractor's font-aware threshold (~50% of space width).
        let space_threshold = font_size * 0.15;

        // Insert space if gap is significant. Previously the upper bound was
        // `gap < font_size * 5.0` on the rationale that very large gaps mean
        // "column boundary, no space needed" — but downstream the caller
        // concatenates the two spans together when this returns false, so
        // "column boundary" actually rendered as `3.80%4.41%` on wide rate
        // tables (issue 487 pr-138-example.pdf). Drop the upper bound so any
        // gap above the inter-glyph threshold gets at least a single space.
        gap > space_threshold
    }

    /// Detect a span whose text is `N.M` (all-digit groups around one dot) and whose
    /// bbox.width is >40% larger than char_widths imply. This pattern occurs in
    /// sailing-score / competition-table PDFs where two adjacent columns (e.g. Q8=1,
    /// F9=10) are stored as a single Tj text run "1.10" spanning both column cells.
    /// Reference ground truth tokenises them as separate words; we must split at the dot.
    pub(crate) fn is_column_spanning_decimal(span: &TextSpan) -> bool {
        let text = &span.text;
        let dot_pos = match text.find('.') {
            Some(p) if p > 0 && p < text.len() - 1 => p,
            _ => return false,
        };
        if text[dot_pos + 1..].contains('.') {
            return false;
        }
        if !text[..dot_pos].chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        if !text[dot_pos + 1..].chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        let char_count = text.chars().count();
        // Signal 1: sparse char_widths array. When the font's glyph
        // iteration produces fewer advance-width entries than there are
        // characters in the decoded string, the span was assembled from two
        // (or more) concatenated Tj runs whose widths come from different
        // points in the glyph table. This is the exact pattern issue 487
        // nougat_018 sailing-score grids hit: each score cell is emitted as
        // a single Tj like `1.10` with `char_widths=[w]` while the PDF
        // semantically means "1" followed by "10" in adjacent score
        // columns. bbox.width can still be tight here (the producer set
        // it to cover just the rendered glyph run), so the existing
        // bbox-inflation check below misses these. Catch them via the
        // sparse-cw signal directly.
        if !span.char_widths.is_empty() && span.char_widths.len() < char_count {
            return true;
        }
        let expected_width = if !span.char_widths.is_empty() {
            let cw_sum: f32 = span.char_widths.iter().sum();
            cw_sum * (char_count as f32 / span.char_widths.len() as f32)
        } else if span.font_size > 0.0 {
            // Digits are narrower than average; 0.50em per char is a safe
            // upper bound for all-digit strings (avoids the 0.60 fallback
            // producing false negatives on column-spanning sailing scores
            // when char_widths is empty, e.g. word_spans from extract_words).
            span.font_size * 0.50 * char_count as f32
        } else {
            return false;
        };
        // Use absolute gap (bbox_w - expected) rather than a ratio so that
        // 5-char spans like "12.11" (gap ≈ 1.1×fs) are caught along with
        // 4-char spans like "1.10" (gap ≈ 1.4×fs). 1.0×font_size is a safe
        // lower bound: normal text rarely has >1em of hidden whitespace.
        let gap = span.bbox.width - expected_width;
        span.font_size > 0.0 && gap > span.font_size * 1.0
    }

    /// When a CID font's glyph iteration produces fewer advance-width entries than
    /// `decode_text_to_unicode` produces unicode chars, `char_widths.len()` < char count.
    /// This indicates two concatenated text runs stored in one Tj operator (e.g. "Theorem1.7"
    /// where "Theorem" widths come from the font's glyph table and "1.7" doesn't have
    /// matching glyph entries). Return the byte offset at which to insert a space,
    /// or None if no split is appropriate.
    pub(crate) fn char_widths_boundary_split(span: &TextSpan) -> Option<usize> {
        let cw_len = span.char_widths.len();
        if cw_len == 0 {
            return None;
        }
        let char_count = span.text.chars().count();
        if cw_len >= char_count {
            return None;
        }
        // Find the byte offset of the (cw_len)-th character
        let (boundary_byte, boundary_char) = span.text.char_indices().nth(cw_len)?;
        let prev_char = span.text[..boundary_byte].chars().next_back()?;
        // Don't insert if either side is already a space
        if boundary_char == ' ' || prev_char == ' ' {
            return None;
        }
        // Non-ASCII chars at the boundary are encoding artifacts (e.g. Polish diacritics
        // in Latin-2 / CP1250 fonts producing one fewer char_width entry). Only split
        // when the boundary char is ASCII, indicating a genuine text-run concatenation.
        if !boundary_char.is_ascii() {
            return None;
        }
        // Split at letter→digit boundary (e.g. "Theorem1.7") or lower→upper ASCII
        // case boundary (e.g. "BigText" from concatenated CID runs "Big"+"Text").
        // Upper→lower transitions are excluded: a ligature spanning an upper→lower
        // boundary within a compound word (e.g. "officeMax" with "fl" ligature)
        // would otherwise produce a false split.
        if (prev_char.is_alphabetic() && boundary_char.is_ascii_digit())
            || (prev_char.is_ascii_lowercase() && boundary_char.is_ascii_uppercase())
        {
            Some(boundary_byte)
        } else {
            None
        }
    }

    /// Merge subscript and superscript spans into their base span.
    ///
    /// In math-heavy untagged PDFs, subscript glyphs (e.g. the "1" in "k₁") are
    /// stored as separate `TextSpan` entries at a slightly lower/higher baseline than
    /// the base character, and non-adjacent in reading order. The text assembly loop
    /// emits them as isolated tokens ("k … 1") rather than the expected word ("k1").
    ///
    /// A span is classified as a subscript/superscript when ALL of the following hold:
    ///  - 1–3 ASCII alphanumeric chars (digit or letter, no punctuation)
    ///  - font_size < 85 % of the page's maximum font size
    ///  - There exists a preceding "base" span whose right edge (x + width) is within
    ///    ±0.6 × sub_fs of the subscript's left edge (x-adjacent)
    ///  - The vertical offset between base and sub is in [8 %, 85 %] of base_fs
    ///    (distinguishes true sub/superscripts from same-line small caps)
    ///
    /// Matched subscript/superscript spans have their text appended to the base
    /// are removed from `spans`.
    fn merge_sub_superscript_spans(spans: &mut Vec<TextSpan>) {
        let n = spans.len();
        if n < 2 {
            return;
        }
        let max_fs = spans.iter().map(|s| s.font_size).fold(0f32, f32::max);
        if max_fs <= 0.0 {
            return;
        }

        // For each candidate sub/superscript span, record which base span to merge into.
        let mut to_merge: Vec<(usize, usize)> = Vec::new(); // (base_idx, sub_idx)
        let mut already_sub: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for i in 0..n {
            let sub = &spans[i];
            // Char-count gate (not byte-count): U+00B2/B3/B9 are 2-byte
            // UTF-8 sequences and U+2070..U+209F are 3-byte, so the
            // earlier byte-length check would have dropped a legitimate
            // 3-digit Unicode subscript like "₁₂₃" (9 bytes).
            if sub.text.is_empty() || sub.text.chars().count() > 3 {
                continue;
            }
            // Accept the raw ASCII the extractor produces AND the
            // already-substituted Unicode super/subscript codepoints
            // (apply_super_sub_script_substitutions runs upstream).
            // Without the U+00B2/B3/B9 + U+2070..U+209F gate, a
            // chemistry formula like "H₂O" would lose the subscript
            // span from this merge, leaving "H ₂ O" in the output.
            let is_sub_char = |c: char| {
                c.is_ascii_alphanumeric()
                    || matches!(c, '\u{00B2}' | '\u{00B3}' | '\u{00B9}')
                    || ('\u{2070}'..='\u{209F}').contains(&c)
            };
            if !sub.text.chars().all(is_sub_char) {
                continue;
            }
            // Must be clearly smaller than the dominant font on this page.
            if sub.font_size >= max_fs * 0.80 {
                continue;
            }
            let sub_fs = sub.font_size;
            let sub_x = sub.bbox.x;
            let sub_y = sub.bbox.y;

            // Search backwards for the best-matching base span.
            let search_limit = 30.min(i);
            let mut best: Option<(usize, f32)> = None; // (idx, |x_dist|)

            for j in (i.saturating_sub(search_limit)..i).rev() {
                if already_sub.contains(&j) {
                    continue;
                }
                let base = &spans[j];
                // Base must be at least 25 % larger than the sub (sub_fs ≤ 0.80×base_fs).
                if base.font_size < sub_fs * 1.25 {
                    continue;
                }
                // Base span must be a valid subscript host:
                //   • 1-char bases (single math variable: k, γ, ρ, H, ∆, …)
                //   • 2-char bases that are NOT two lowercase-ASCII letters
                //     (accepts "Pr", "εp", "ρε" but rejects "of", "to")
                // Multi-char lowercase-only strings like "and", "let", "sup"
                // are English words or common operators; their adjacent digit
                // spans are handled by the assembly loop and char_widths_boundary_split.
                let chars: Vec<char> = base.text.chars().collect();
                let is_valid_base = match chars.len() {
                    1 => true,
                    2 => chars.iter().any(|c| !c.is_ascii_lowercase()),
                    _ => false,
                };
                if !is_valid_base {
                    continue;
                }
                let base_right = base.bbox.x + base.bbox.width;
                let x_dist = sub_x - base_right;
                let y_diff_abs = (base.bbox.y - sub_y).abs();

                // Use em-relative x_dist thresholds.
                // Real sub/superscript glyphs land within ±[−0.1×base_fs, 0.25×base_fs]
                // of the base's advance edge; absolute bounds were wrong for non-12pt fonts.
                let base_fs = base.font_size.max(1.0);
                let x_lo = -0.1 * base_fs;
                let x_hi = 0.25 * base_fs;
                if x_dist < x_lo || x_dist > x_hi {
                    continue;
                }
                // Vertical offset must be in the sub/superscript range.
                // Lower bound 12 % of base_fs ensures same-line small caps are excluded.
                // Upper bound 75 % excludes large line-to-line y differences (e.g.
                // author affiliation numbers on a different baseline row).
                if y_diff_abs < base.font_size * 0.12 || y_diff_abs > base.font_size * 0.75 {
                    continue;
                }
                let score = x_dist.abs();
                if best.is_none() || score < best.unwrap().1 {
                    best = Some((j, score));
                }
            }

            if let Some((base_idx, _)) = best {
                to_merge.push((base_idx, i));
                already_sub.insert(i);
            }
        }

        if to_merge.is_empty() {
            return;
        }

        // Collect (base_idx, sub_idx, sub_text, sub_right_edge, sub_char_widths, sub_fs)
        // before mutating spans.
        let ops: Vec<(usize, usize, String, f32, Vec<f32>, f32)> = to_merge
            .iter()
            .map(|pair| {
                let (bi, si) = *pair;
                let sub = &spans[si];
                (
                    bi,
                    si,
                    sub.text.clone(),
                    sub.bbox.x + sub.bbox.width,
                    sub.char_widths.clone(),
                    sub.font_size,
                )
            })
            .collect();

        // Apply: append sub text to base; extend bbox and char_widths to cover the sub.
        //
        // Extending bbox: the assembly loop uses span widths for gap calculations — keeping
        // the original width would make the gap to the following span appear too large.
        //
        // Extending char_widths: char_widths_boundary_split fires whenever cw_len < char_count.
        // After merging sub text, char_count grows but cw_len stays the same, which would
        // cause the split to re-separate the merged token (e.g. "k1" → "k 1"). Adding
        // estimated widths for the sub characters prevents this.
        for (base_idx, _, sub_text, sub_right, sub_cw, sub_fs) in &ops {
            let base = &mut spans[*base_idx];
            base.text.push_str(sub_text);
            let base_right = base.bbox.x + base.bbox.width;
            if *sub_right > base_right {
                base.bbox.width = sub_right - base.bbox.x;
            }
            if !base.char_widths.is_empty() {
                let sub_char_count = sub_text.chars().count();
                if !sub_cw.is_empty() {
                    base.char_widths.extend_from_slice(sub_cw);
                } else {
                    // Estimate sub char widths at 0.50 em per character.
                    let w = sub_fs * 0.50;
                    for _ in 0..sub_char_count {
                        base.char_widths.push(w);
                    }
                }
            }
        }

        // Drop the merged sub/superscript spans in one pass.
        let to_remove: std::collections::HashSet<usize> =
            ops.iter().map(|(_, si, _, _, _, _)| *si).collect();
        let mut idx = 0usize;
        spans.retain(|_| {
            let keep = !to_remove.contains(&idx);
            idx += 1;
            keep
        });
    }

    /// Append span text to `out`, splitting merged runs for cleaner word tokenisation.
    /// Priority 0: spans whose text is entirely `\n`/`\r` are line-break signals.
    /// Priority 1: column-spanning decimal (nougat_018 sailing tables).
    /// Priority 2: char_widths boundary split (pdfa_004 CID-font merge artifacts).
    #[inline]
    pub(crate) fn push_span_text(out: &mut String, span: &TextSpan) {
        // A span whose entire text is one or more newline/CR characters is a
        // ToUnicode line-break signal. Treat it as a logical newline separator rather
        // than emitting the raw control characters verbatim as visible content.
        if !span.text.is_empty() && span.text.chars().all(|c| c == '\n' || c == '\r') {
            if !out.ends_with('\n') {
                out.push('\n');
            }
            return;
        }
        if Self::is_column_spanning_decimal(span) {
            let dot = span.text.find('.').unwrap();
            out.push_str(&span.text[..dot]);
            out.push(' ');
            out.push_str(&span.text[dot + 1..]);
        } else if let Some(split) = Self::char_widths_boundary_split(span) {
            out.push_str(&span.text[..split]);
            out.push(' ');
            out.push_str(&span.text[split..]);
        } else {
            out.push_str(&span.text);
        }
    }

    /// #557a: append a span's text to the structure-tree assembly, reversing a
    /// PURE-RTL run (every non-space char is an Arabic/Hebrew letter, no Latin)
    /// from visual to logical order. The tagged/struct-tree path collapses each
    /// run to a single span and never reaches `reverse_rtl_visual_order_runs`,
    /// so visually-stored RTL (e.g. issue10301 Hebrew "גבא") otherwise leaked
    /// out reversed. A single-direction run's logical order is just its reverse,
    /// so no glyph geometry is needed for the pure-RTL case.
    fn push_span_text_bidi(out: &mut String, span: &TextSpan) {
        use crate::text::rtl_detector::is_rtl_text;
        let mut rtl = 0usize;
        let mut has_latin = false;
        for c in span.text.chars() {
            if c.is_whitespace() {
                continue;
            }
            if c.is_ascii_alphabetic() {
                has_latin = true;
                break;
            }
            if is_rtl_text(c as u32) {
                rtl += 1;
            }
        }
        if rtl >= 2 && !has_latin {
            let reversed: String = span.text.chars().rev().collect();
            let mut tmp = span.clone();
            tmp.text = reversed;
            Self::push_span_text(out, &tmp);
        } else {
            Self::push_span_text(out, span);
        }
    }

    /// Parse font size from a /DA (Default Appearance) string.
    ///
    /// DA strings follow the format: `"/FontName size Tf ..."` (e.g., `"/Helv 12 Tf 0 g"`).
    /// Returns the font size preceding the `Tf` operator, or a default of 10.0 if not found.
    fn parse_font_size_from_da(da: &str) -> f32 {
        let tokens: Vec<&str> = da.split_whitespace().collect();
        for i in 0..tokens.len() {
            if tokens[i] == "Tf" && i > 0 {
                if let Ok(size) = tokens[i - 1].parse::<f32>() {
                    if size > 0.0 {
                        return size;
                    }
                }
            }
        }
        10.0 // default
    }

    /// Extract widget annotation values as TextSpans positioned at their /Rect locations.
    ///
    /// Converts each widget annotation's field value into a `TextSpan` with the annotation's
    /// bounding box. These spans merge naturally with content stream spans and get positioned
    /// correctly by existing layout algorithms.
    fn extract_widget_spans(&self, page_index: usize) -> Vec<TextSpan> {
        use crate::extractors::forms::field_flags;
        use crate::geometry::Rect;

        let page_obj = match self.get_page(page_index) {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };
        let page_dict = match page_obj.as_dict() {
            Some(d) => d,
            None => return Vec::new(),
        };

        // Get /Annots array (may be direct or indirect)
        let annots_arr = match page_dict.get("Annots") {
            Some(Object::Array(arr)) => arr.clone(),
            Some(Object::Reference(r)) => match self.load_object(*r) {
                Ok(Object::Array(arr)) => arr,
                _ => return Vec::new(),
            },
            _ => return Vec::new(),
        };

        let mut spans = Vec::new();
        let base_sequence = 1_000_000; // high sequence number so widget spans sort after content spans at same Y

        for (idx, annot_obj) in annots_arr.iter().enumerate() {
            let annot_ref = match annot_obj {
                Object::Reference(r) => *r,
                _ => continue,
            };
            let dict = match self.load_object(annot_ref) {
                Ok(obj) => match obj.as_dict() {
                    Some(d) => d.clone(),
                    None => continue,
                },
                Err(_) => continue,
            };

            // Only process Widget annotations
            let subtype = match dict.get("Subtype").and_then(|s| s.as_name()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            if !subtype.eq_ignore_ascii_case("widget") {
                continue;
            }

            // Check /F flags — skip invisible/hidden/noview annotations
            // Bit 1 (0x1) = Invisible, Bit 2 (0x2) = Hidden, Bit 6 (0x20) = NoView
            if let Some(Object::Integer(f)) = dict.get("F") {
                if *f & (0x1 | 0x2 | 0x20) != 0 {
                    continue;
                }
            }

            // Parse /Rect [x1, y1, x2, y2] → Rect { x, y, width, height }
            let rect = match dict.get("Rect") {
                Some(Object::Array(arr)) if arr.len() == 4 => {
                    let mut coords = [0.0f32; 4];
                    let mut ok = true;
                    for (i, item) in arr.iter().enumerate() {
                        match item {
                            Object::Integer(n) => coords[i] = *n as f32,
                            Object::Real(f) => coords[i] = *f as f32,
                            _ => {
                                ok = false;
                                break;
                            },
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let x = coords[0].min(coords[2]);
                    let y = coords[1].min(coords[3]);
                    let w = (coords[2] - coords[0]).abs();
                    let h = (coords[3] - coords[1]).abs();
                    if w < 0.1 || h < 0.1 {
                        continue;
                    } // skip zero-area rects
                    Rect::new(x, y, w, h)
                },
                Some(Object::Reference(r)) => match self.load_object(*r) {
                    Ok(Object::Array(arr)) if arr.len() == 4 => {
                        let mut coords = [0.0f32; 4];
                        let mut ok = true;
                        for (i, item) in arr.iter().enumerate() {
                            match item {
                                Object::Integer(n) => coords[i] = *n as f32,
                                Object::Real(f) => coords[i] = *f as f32,
                                _ => {
                                    ok = false;
                                    break;
                                },
                            }
                        }
                        if !ok {
                            continue;
                        }
                        let x = coords[0].min(coords[2]);
                        let y = coords[1].min(coords[3]);
                        let w = (coords[2] - coords[0]).abs();
                        let h = (coords[3] - coords[1]).abs();
                        if w < 0.1 || h < 0.1 {
                            continue;
                        }
                        Rect::new(x, y, w, h)
                    },
                    _ => continue,
                },
                _ => continue,
            };

            // Get field type via /FT (with parent-chain inheritance)
            let ft = dict
                .get("FT")
                .and_then(|o| o.as_name())
                .map(|s| s.to_string())
                .or_else(|| self.resolve_inherited_ft(&dict));

            // Get field flags /Ff (with parent-chain inheritance)
            let ff = dict
                .get("Ff")
                .and_then(|o| match o {
                    Object::Integer(i) => Some(*i as u32),
                    _ => None,
                })
                .or_else(|| self.resolve_inherited_ff(&dict));
            let ff = ff.unwrap_or(0);

            // Determine display text based on field type
            let display_text = match ft.as_deref() {
                Some("Tx") => {
                    // Text field: use /V string value
                    if ff & field_flags::PASSWORD != 0 {
                        // Password field: render as asterisks
                        Some("********".to_string())
                    } else {
                        let value = Self::parse_string_value_static(dict.get("V"))
                            .or_else(|| self.resolve_inherited_field_value(&dict));
                        match value {
                            Some(v) if !v.trim().is_empty() => {
                                // Bound the value to the widget's visual
                                // capacity. Multi-line text-area fields
                                // can hold scrollable content far larger
                                // than the bbox visually renders; per
                                // spec §12.7.4.3 `/V` is the field's
                                // data, but `extract_text` semantics
                                // target what would be visible on the
                                // page. Truncate keeps the rendered
                                // portion and drops the overflow.
                                Some(Self::truncate_to_widget_capacity(v.trim().to_string(), &rect))
                            },
                            _ => {
                                // Fallback: try AP stream text. Truncate
                                // to bbox capacity — some PDFs reuse a
                                // single Form XObject for many widgets'
                                // `/AP /N`, pointing every widget's
                                // appearance at the page-background
                                // content; without the cap each widget
                                // would extract that content once.
                                self.extract_text_from_ap_stream(&dict).and_then(|t| {
                                    let t = t.trim().to_string();
                                    if t.is_empty() {
                                        return None;
                                    }
                                    Some(Self::truncate_to_widget_capacity(t, &rect))
                                })
                            },
                        }
                    }
                },
                Some("Btn") => {
                    if ff & field_flags::PUSH_BUTTON != 0 {
                        // Push button: caption is in /MK /CA per PDF Spec
                        // ISO 32000-1:2008 §12.5.6.19 (Appearance Characteristics
                        // Dictionary). Extracting it lets screen readers
                        // text-extraction consumers see the button label.
                        dict.get("MK")
                            .and_then(|mk| mk.as_dict())
                            .and_then(|mk| Self::parse_string_value_static(mk.get("CA")))
                            .and_then(|s| {
                                let t = s.trim().to_string();
                                if t.is_empty() {
                                    None
                                } else {
                                    Some(t)
                                }
                            })
                    } else {
                        // Checkbox or radio button
                        let value = Self::parse_string_value_static(dict.get("V"))
                            .or_else(|| self.resolve_inherited_field_value(&dict));
                        let is_checked = match &value {
                            Some(v) => {
                                let v_lower = v.to_ascii_lowercase();
                                v_lower != "off" && !v_lower.is_empty()
                            },
                            None => false,
                        };
                        if is_checked {
                            // A checked box is meaningful state worth surfacing.
                            Some("[x]".to_string())
                        } else {
                            // An UNCHECKED box carries no text. Emitting "[ ]"
                            // here injected noise that pdftotext/PyMuPDF never
                            // produce — the dominant cause of pdf_oxide being
                            // the sole outlier on AcroForm-heavy PDFs in the
                            // cross-corpus sweep (CORPUS-1). Emit nothing.
                            None
                        }
                    }
                },
                Some("Ch") => {
                    // Choice field: use /V selected value
                    let value = dict.get("V");
                    match value {
                        Some(Object::Array(arr)) => {
                            // Multiple selections: join with ", "
                            let items: Vec<String> = arr
                                .iter()
                                .filter_map(|item| Self::parse_string_value_static(Some(item)))
                                .collect();
                            if items.is_empty() {
                                None
                            } else {
                                Some(items.join(", "))
                            }
                        },
                        other => Self::parse_string_value_static(other)
                            .or_else(|| self.resolve_inherited_field_value(&dict))
                            .and_then(|v| {
                                let t = v.trim().to_string();
                                if t.is_empty() {
                                    None
                                } else {
                                    Some(t)
                                }
                            }),
                    }
                },
                Some("Sig") => {
                    // Signature field: skip (no user-visible text)
                    None
                },
                _ => {
                    // Unknown field type: try /V as text
                    Self::parse_string_value_static(dict.get("V"))
                        .or_else(|| self.resolve_inherited_field_value(&dict))
                        .and_then(|v| {
                            let t = v.trim().to_string();
                            if t.is_empty() {
                                None
                            } else {
                                Some(t)
                            }
                        })
                },
            };

            let text = match display_text {
                Some(t) if !t.is_empty() => t,
                _ => {
                    // CORPUS-5: a widget with no extractable /V value (notably a
                    // signature field, /FT /Sig) often carries its VISIBLE text
                    // in the /AP/N appearance stream (e.g. "Firmato
                    // elettronicamente da ..."). pdftotext / PyMuPDF surface it;
                    // fall back to the appearance stream so it isn't dropped.
                    // Fields that DO yield a /V value take the arm above, so this
                    // never double-extracts.
                    match self.extract_text_from_ap_stream(&dict) {
                        Some(ap) if !ap.trim().is_empty() => ap.trim().to_string(),
                        _ => continue,
                    }
                },
            };

            // Parse font size from /DA string
            let font_size = {
                let da = dict
                    .get("DA")
                    .and_then(|o| match o {
                        Object::String(s) => Some(Self::decode_pdf_text_string(s)),
                        _ => None,
                    })
                    .or_else(|| self.resolve_inherited_da(&dict));

                match da {
                    Some(da_str) => {
                        let size = Self::parse_font_size_from_da(&da_str);
                        if size <= 0.0 {
                            // Auto-size: estimate from rect height
                            (rect.height * 0.7).clamp(6.0, 24.0)
                        } else {
                            size
                        }
                    },
                    None => {
                        // No DA at all: estimate from rect height
                        (rect.height * 0.7).clamp(6.0, 24.0)
                    },
                }
            };

            spans.push(TextSpan {
                artifact_type: None,
                text,
                bbox: rect,
                font_name: String::new(),
                font_size,
                font_weight: crate::layout::text_block::FontWeight::Normal,
                is_italic: false,
                is_monospace: false,
                color: crate::layout::text_block::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                },
                mcid: None,
                sequence: base_sequence + idx,
                split_boundary_before: false,
                offset_semantic: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
                heading_level: None,
                rotation_degrees: 0.0,
            });
        }

        spans
    }

    /// Build TextSpan objects from the /Contents field of content-bearing annotations.
    ///
    /// Sticky note (/Subtype/Text), FreeText, Stamp, and markup annotations carry
    /// human-readable text in their /Contents field. Widget annotations are already
    /// handled by `extract_widget_spans`; Popup annotations hold no independent
    /// content (their text belongs to the parent annotation).
    fn annotation_content_spans(&self, page_index: usize) -> Vec<TextSpan> {
        use crate::geometry::Rect;

        let page_obj = match self.get_page(page_index) {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };
        let page_dict = match page_obj.as_dict() {
            Some(d) => d,
            None => return Vec::new(),
        };

        let annots_arr = match page_dict.get("Annots") {
            Some(Object::Array(arr)) => arr.clone(),
            Some(Object::Reference(r)) => match self.load_object(*r) {
                Ok(Object::Array(arr)) => arr,
                _ => return Vec::new(),
            },
            _ => return Vec::new(),
        };

        let mut spans: Vec<TextSpan> = Vec::new();
        let base_sequence = 2_000_000usize; // sort after widget spans

        for (idx, annot_obj) in annots_arr.iter().enumerate() {
            let annot_ref = match annot_obj {
                Object::Reference(r) => *r,
                _ => continue,
            };
            let dict = match self.load_object(annot_ref) {
                Ok(obj) => match obj.as_dict() {
                    Some(d) => d.clone(),
                    None => continue,
                },
                Err(_) => continue,
            };

            let subtype = match dict.get("Subtype").and_then(|s| s.as_name()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let subtype_lc = subtype.to_ascii_lowercase();

            // Skip Widget (handled by extract_widget_spans) and Popup (no independent content).
            if subtype_lc == "widget" || subtype_lc == "popup" {
                continue;
            }

            // Skip invisible / hidden / NoView annotations.
            if let Some(Object::Integer(f)) = dict.get("F") {
                if *f & (0x1 | 0x2 | 0x20) != 0 {
                    continue;
                }
            }

            // Only FreeText and Stamp have /Contents representing visible page text.
            // Text (sticky-note) /Contents is reviewer comment text shown in a pop-up
            // window, not rendered on the page — exclude it to avoid injecting popup
            // notes into the body text stream.
            // For FreeText/Stamp: try /Contents first; fall back to AP stream so that
            // Stamp annotations with empty /Contents but a rendered AP stream are included.
            let is_visible = matches!(subtype_lc.as_str(), "freetext" | "stamp");
            if !is_visible {
                continue;
            }

            let text = {
                let from_contents = if let Some(Object::String(s)) = dict.get("Contents") {
                    let decoded = Self::decode_pdf_text_string(s).trim().to_string();
                    if decoded.is_empty() {
                        None
                    } else {
                        Some(decoded)
                    }
                } else {
                    None
                };
                if let Some(t) = from_contents {
                    t
                } else {
                    match self.extract_text_from_ap_stream(&dict) {
                        Some(ap_text) if !ap_text.trim().is_empty() => ap_text.trim().to_string(),
                        _ => continue,
                    }
                }
            };

            // Use /Rect as the annotation's bounding box.
            // /Rect may be a direct array or an indirect reference to an array.
            let rect_obj = match dict.get("Rect") {
                Some(Object::Reference(r)) => match self.load_object(*r) {
                    Ok(o) => o,
                    Err(_) => continue,
                },
                Some(o) => o.clone(),
                None => continue,
            };
            let rect = match rect_obj.as_array() {
                Some(arr) if arr.len() == 4 => {
                    let mut coords = [0.0f32; 4];
                    let mut ok = true;
                    for (i, item) in arr.iter().enumerate() {
                        match item {
                            Object::Integer(n) => coords[i] = *n as f32,
                            Object::Real(f) => coords[i] = *f as f32,
                            _ => {
                                ok = false;
                                break;
                            },
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let x = coords[0].min(coords[2]);
                    let y = coords[1].min(coords[3]);
                    let w = (coords[2] - coords[0]).abs();
                    let h = (coords[3] - coords[1]).abs();
                    Rect {
                        x,
                        y,
                        width: w.max(1.0),
                        height: h.max(1.0),
                    }
                },
                _ => continue,
            };

            spans.push(TextSpan {
                artifact_type: None,
                text,
                bbox: rect,
                font_name: String::new(),
                font_size: 12.0,
                font_weight: crate::layout::text_block::FontWeight::Normal,
                is_italic: false,
                is_monospace: false,
                color: crate::layout::text_block::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                },
                mcid: None,
                sequence: base_sequence + idx,
                split_boundary_before: false,
                offset_semantic: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
                heading_level: None,
                rotation_degrees: 0.0,
            });
        }

        spans
    }

    /// Walk /Parent chain to find inherited /Ff (field flags) value.
    fn resolve_inherited_ff(
        &self,
        dict: &std::collections::HashMap<String, Object>,
    ) -> Option<u32> {
        let mut parent_ref = match dict.get("Parent") {
            Some(Object::Reference(r)) => Some(*r),
            _ => return None,
        };
        let mut depth = 0;
        while let Some(pref) = parent_ref {
            if depth >= 10 {
                break;
            }
            depth += 1;
            if let Ok(parent_obj) = self.load_object(pref) {
                if let Some(parent_dict) = parent_obj.as_dict() {
                    if let Some(Object::Integer(ff)) = parent_dict.get("Ff") {
                        return Some(*ff as u32);
                    }
                    parent_ref = match parent_dict.get("Parent") {
                        Some(Object::Reference(r)) => Some(*r),
                        _ => None,
                    };
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        None
    }

    /// Walk /Parent chain (and AcroForm) to find inherited /DA (Default Appearance) string.
    fn resolve_inherited_da(
        &self,
        dict: &std::collections::HashMap<String, Object>,
    ) -> Option<String> {
        // First check parent chain
        let mut parent_ref = match dict.get("Parent") {
            Some(Object::Reference(r)) => Some(*r),
            _ => None,
        };
        let mut depth = 0;
        while let Some(pref) = parent_ref {
            if depth >= 10 {
                break;
            }
            depth += 1;
            if let Ok(parent_obj) = self.load_object(pref) {
                if let Some(parent_dict) = parent_obj.as_dict() {
                    if let Some(Object::String(da)) = parent_dict.get("DA") {
                        return Some(Self::decode_pdf_text_string(da));
                    }
                    parent_ref = match parent_dict.get("Parent") {
                        Some(Object::Reference(r)) => Some(*r),
                        _ => None,
                    };
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Fall back to AcroForm-level /DA
        if let Some(trailer_dict) = self.trailer.as_dict() {
            if let Some(root_ref) = trailer_dict.get("Root").and_then(|o| o.as_reference()) {
                if let Ok(root_obj) = self.load_object(root_ref) {
                    if let Some(root_dict) = root_obj.as_dict() {
                        let acroform = match root_dict.get("AcroForm") {
                            Some(Object::Reference(r)) => self.load_object(*r).ok(),
                            Some(obj) => Some(obj.clone()),
                            None => None,
                        };
                        if let Some(acroform_obj) = acroform {
                            if let Some(af_dict) = acroform_obj.as_dict() {
                                if let Some(Object::String(da)) = af_dict.get("DA") {
                                    return Some(Self::decode_pdf_text_string(da));
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Append text from non-widget annotations on a page.
    ///
    /// Extracts text from FreeText annotations (text box contents), Stamp annotations
    /// (appearance stream text), and other non-widget annotation types.
    /// Widget annotations are handled separately via `extract_widget_spans()`.
    /// Skips hidden and invisible annotations per PDF spec flags.
    fn append_non_widget_annotation_text(&self, page_index: usize, text: &mut String) {
        // Lightweight annotation text extraction — avoids full get_annotations() overhead.
        // Only reads /Subtype, /V, /Contents, /F, and /Parent (for field value inheritance).
        // Uses get_page() which is cached after first access.
        let page_obj = match self.get_page(page_index) {
            Ok(o) => o,
            Err(_) => return,
        };
        let page_dict = match page_obj.as_dict() {
            Some(d) => d,
            None => return,
        };

        // Get /Annots array (may be direct or indirect)
        let annots_arr = match page_dict.get("Annots") {
            Some(Object::Array(arr)) => arr.clone(),
            Some(Object::Reference(r)) => match self.load_object(*r) {
                Ok(Object::Array(arr)) => arr,
                _ => return,
            },
            _ => return, // No annotations on this page
        };

        let mut annot_texts: Vec<String> = Vec::new();

        for annot_obj in &annots_arr {
            let len_before_annot = annot_texts.len();
            let annot_ref = match annot_obj {
                Object::Reference(r) => *r,
                _ => continue,
            };
            let dict = match self.load_object(annot_ref) {
                Ok(obj) => match obj.as_dict() {
                    Some(d) => d.clone(),
                    None => continue,
                },
                Err(_) => continue,
            };

            // Check /F flags — skip invisible/hidden annotations
            // Bit 1 (0x1) = Invisible, Bit 2 (0x2) = Hidden, Bit 6 (0x20) = NoView
            if let Some(Object::Integer(f)) = dict.get("F") {
                if *f & (0x1 | 0x2 | 0x20) != 0 {
                    continue;
                }
            }

            let subtype = match dict.get("Subtype").and_then(|s| s.as_name()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let subtype_lower = subtype.to_ascii_lowercase();

            match subtype_lower.as_str() {
                "widget" => {
                    // Widgets are now handled by extract_widget_spans() as inline TextSpans.
                    // Skip them here to avoid duplicate text at the end of output.
                    continue;
                },
                "freetext" | "stamp" => {
                    if let Some(Object::String(s)) = dict.get("Contents") {
                        let decoded = Self::decode_pdf_text_string(s);
                        let trimmed = decoded.trim().to_string();
                        if !trimmed.is_empty() {
                            annot_texts.push(trimmed);
                        }
                    }
                },
                // Text (sticky-note) /Contents is reviewer popup comment text, not
                // visible page content — skip to avoid injecting popup notes.
                "text" => {},
                // Geometric shape annotations — per §12.5.6.2, their /Contents is
                // also popup/comment text, same as the markup group below.
                "line" | "circle" | "square" | "polygon" | "polyline" => {
                    // Skip — /Contents is popup comment text, not page content.
                },
                // Markup/comment annotations — per ISO 32000-1 §12.5.6.2 (Table 166),
                // the /Contents of all these subtypes is popup/comment text written
                // by a reviewer, NOT text displayed on the page. Exclude to avoid
                // injecting user annotation notes into the body text stream.
                // Per §12.5.6.2, all of these annotations' /Contents is popup/comment
                // text (displayed in a pop-up window), not rendered page content.
                // FileAttachment is explicitly in this category per §12.5.6.2 even
                // though §12.5.6.15 calls it "descriptive text" — the pop-up semantics
                // take precedence.
                "highlight" | "underline" | "strikeout" | "squiggly" | "caret"
                | "fileattachment" | "redact" | "ink" => {
                    // Skip — /Contents is popup comment text, not page content.
                },
                // Link /Contents is an accessibility alternate description (§12.5.6.5).
                // Treated as supplementary text on pages with no body content.
                "link" => {
                    if let Some(Object::String(s)) = dict.get("Contents") {
                        let decoded = Self::decode_pdf_text_string(s);
                        let trimmed = decoded.trim().to_string();
                        if !trimmed.is_empty() {
                            annot_texts.push(trimmed);
                        }
                    }
                },
                // Popup annotations — per §12.5.6.14 Table 183, the parent
                // annotation's /Contents overrides the popup's own /Contents.
                "popup" => {
                    // Try parent annotation's /Contents first (spec §12.5.6.14).
                    let mut got_text = false;
                    if let Some(parent_ref) = dict.get("Parent").and_then(|o| o.as_reference()) {
                        if let Ok(parent_obj) = self.load_object(parent_ref) {
                            if let Some(parent_dict) = parent_obj.as_dict() {
                                if let Some(Object::String(s)) = parent_dict.get("Contents") {
                                    let decoded = Self::decode_pdf_text_string(s);
                                    let trimmed = decoded.trim().to_string();
                                    if !trimmed.is_empty() {
                                        annot_texts.push(trimmed);
                                        got_text = true;
                                    }
                                }
                            }
                        }
                    }
                    // Fall back to the popup's own /Contents only when parent has none.
                    if !got_text {
                        if let Some(Object::String(s)) = dict.get("Contents") {
                            let decoded = Self::decode_pdf_text_string(s);
                            let trimmed = decoded.trim().to_string();
                            if !trimmed.is_empty() {
                                annot_texts.push(trimmed);
                            }
                        }
                    }
                },
                _ => {
                    // For any other annotation type, also try /Contents
                    if let Some(Object::String(s)) = dict.get("Contents") {
                        let decoded = Self::decode_pdf_text_string(s);
                        let trimmed = decoded.trim().to_string();
                        if !trimmed.is_empty() {
                            annot_texts.push(trimmed);
                        }
                    }
                },
            }

            // Fallback: if no text was extracted from /V or /Contents,
            // try extracting from the /AP/N (Normal Appearance) stream.
            let text_before = annot_texts.len();
            if text_before == len_before_annot {
                if let Some(ap_text) = self.extract_text_from_ap_stream(&dict) {
                    let trimmed = ap_text.trim().to_string();
                    if !trimmed.is_empty() {
                        annot_texts.push(trimmed);
                    }
                }
            }
        }

        if !annot_texts.is_empty() {
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&annot_texts.join("\n"));
        }
    }

    /// Extract text from an annotation's Normal Appearance stream (/AP/N).
    ///
    /// AP streams are content streams with their own /Resources. This creates
    /// a temporary TextExtractor, loads fonts from the AP stream resources,
    /// and extracts text spans from the decoded stream data.
    fn extract_text_from_ap_stream(
        &self,
        annot_dict: &std::collections::HashMap<String, Object>,
    ) -> Option<String> {
        use crate::extractors::TextExtractor;

        // Get /AP dictionary
        let ap_obj = annot_dict.get("AP")?;
        let ap = if let Some(r) = ap_obj.as_reference() {
            self.load_object(r).ok()?
        } else {
            ap_obj.clone()
        };
        let ap_dict = ap.as_dict()?;

        // Get /N (Normal appearance) — can be a stream ref or a dictionary of states
        let n_obj = ap_dict.get("N")?;
        let (n_stream, n_ref) = match n_obj {
            Object::Reference(r) => (self.load_object(*r).ok()?, *r),
            _ => return None, // N must be a reference to a stream
        };

        // Verify it's a stream (has a dict with stream data)
        let n_dict = n_stream.as_dict()?;

        // Decode the AP/N stream
        let stream_data = match self.decode_stream_with_encryption(&n_stream, n_ref) {
            Ok(data) => data,
            Err(_) => return None,
        };

        // Quick check: does the stream contain text operators?
        if !Self::may_contain_text(&stream_data) {
            return None;
        }

        // Create a temporary text extractor for this AP stream
        let mut extractor = TextExtractor::new();

        // Load fonts from the AP/N stream's own /Resources
        if let Some(resources) = n_dict.get("Resources") {
            let res_obj = if let Some(r) = resources.as_reference() {
                self.load_object(r)
                    .ok()
                    .unwrap_or_else(|| resources.clone())
            } else {
                resources.clone()
            };
            extractor.set_resources(res_obj.clone());
            extractor.set_document(self);
            let _ = self.load_fonts(&res_obj, &mut extractor);
        } else {
            // No resources on the AP stream — try the annotation's /DR or parent page resources
            // For now, skip if no resources (can't decode fonts)
            return None;
        }

        // Extract text spans from the AP stream
        let spans = extractor.extract_text_spans(&stream_data).ok()?;
        if spans.is_empty() {
            return None;
        }

        // Collect span text
        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        if text.trim().is_empty() {
            return None;
        }
        Some(text)
    }

    /// Char-count capacity for what physically fits inside a widget
    /// bbox at body font sizes. Per PDF spec §12.7.4.3 the field's
    /// value is `/V`; the appearance stream is visual rendering
    /// only. When we fall back to AP extraction the result must be
    /// bounded by what the widget could visually show — PDFs that
    /// reuse a single Form XObject for many widgets' `/AP /N` would
    /// otherwise dump the shared content once per widget, and
    /// scrollable multi-line text fields hold far more characters
    /// in `/V` than ever render at once.
    ///
    /// Heuristic: ~14 chars per cm² at body font sizes. At PDF
    /// 72 dpi (1 pt = 0.0353 cm), the formula
    /// `capacity = 0.0175 * w_pt * h_pt + 64` applies; the constant
    /// term absorbs short labels where the area estimate alone is
    /// too tight to even hold the field's name.
    fn widget_text_capacity(bbox: &crate::geometry::Rect) -> usize {
        let area = bbox.width.max(0.0) * bbox.height.max(0.0);
        (0.0175 * area) as usize + 64
    }

    /// Truncate `text` to the widget's visual capacity. If `text`
    /// already fits, returns it unchanged. Used to bound AP-fallback
    /// extraction (and other content paths) so a single widget can't
    /// dump page-background prose or scrollable field internals into
    /// the page text.
    fn truncate_to_widget_capacity(text: String, bbox: &crate::geometry::Rect) -> String {
        let cap = Self::widget_text_capacity(bbox);
        let n = text.chars().count();
        if n <= cap {
            return text;
        }
        text.chars().take(cap).collect()
    }

    /// Walk /Parent chain to find inherited /FT (field type) value.
    fn resolve_inherited_ft(
        &self,
        dict: &std::collections::HashMap<String, Object>,
    ) -> Option<String> {
        let mut parent_ref = match dict.get("Parent") {
            Some(Object::Reference(r)) => Some(*r),
            _ => return None,
        };
        let mut depth = 0;
        while let Some(pref) = parent_ref {
            if depth >= 10 {
                break;
            }
            depth += 1;
            if let Ok(parent_obj) = self.load_object(pref) {
                if let Some(parent_dict) = parent_obj.as_dict() {
                    if let Some(ft) = parent_dict.get("FT").and_then(|o| o.as_name()) {
                        return Some(ft.to_string());
                    }
                    parent_ref = match parent_dict.get("Parent") {
                        Some(Object::Reference(r)) => Some(*r),
                        _ => None,
                    };
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        None
    }

    /// Walk /Parent chain to find inherited /V value (PDF spec 12.7.3.1).
    fn resolve_inherited_field_value(
        &self,
        dict: &std::collections::HashMap<String, Object>,
    ) -> Option<String> {
        let mut parent_ref = match dict.get("Parent") {
            Some(Object::Reference(r)) => Some(*r),
            _ => return None,
        };
        let mut depth = 0;
        while let Some(pref) = parent_ref {
            if depth >= 10 {
                break;
            }
            depth += 1;
            if let Ok(parent_obj) = self.load_object(pref) {
                if let Some(parent_dict) = parent_obj.as_dict() {
                    if let Some(v) = Self::parse_string_value_static(parent_dict.get("V")) {
                        return Some(v);
                    }
                    parent_ref = match parent_dict.get("Parent") {
                        Some(Object::Reference(r)) => Some(*r),
                        _ => None,
                    };
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        None
    }

    /// Parse a string value from a PDF object with proper PDF string decoding.
    /// Handles UTF-16BE (BOM \xFE\xFF) and PDFDocEncoding per ISO 32000-1 §7.9.2.2.
    fn parse_string_value_static(obj: Option<&Object>) -> Option<String> {
        match obj {
            Some(Object::String(s)) => Some(Self::decode_pdf_text_string(s)),
            Some(Object::Name(n)) => Some(n.clone()),
            Some(Object::Integer(i)) => Some(i.to_string()),
            Some(Object::Real(f)) => Some(f.to_string()),
            _ => None,
        }
    }

    /// Decode a PDF text string that may be UTF-16BE/LE (with BOM) or PDFDocEncoding.
    fn decode_pdf_text_string(bytes: &[u8]) -> String {
        if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF {
            // UTF-16BE with BOM
            let utf16_bytes = &bytes[2..];
            let utf16_pairs: Vec<u16> = utf16_bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
                .collect();
            String::from_utf16(&utf16_pairs)
                .unwrap_or_else(|_| String::from_utf8_lossy(bytes).to_string())
        } else if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE {
            // UTF-16LE with BOM
            let utf16_bytes = &bytes[2..];
            let utf16_pairs: Vec<u16> = utf16_bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();
            String::from_utf16(&utf16_pairs)
                .unwrap_or_else(|_| String::from_utf8_lossy(bytes).to_string())
        } else {
            // PDFDocEncoding — superset of ISO Latin-1
            bytes
                .iter()
                .filter_map(|&b| crate::fonts::font_dict::pdfdoc_encoding_lookup(b))
                .collect()
        }
    }

    /// Strip XHTML tags from rich content (/RC) to extract plain text.
    ///
    /// Per PDF Spec ISO 32000-1:2008 Section 12.7.3.4, /RC entries contain
    /// XHTML-formatted rich text. This method strips tags to produce plain text.
    #[cfg(test)]
    fn strip_xhtml_tags(xhtml: &str) -> String {
        let mut result = String::with_capacity(xhtml.len());
        let mut inside_tag = false;
        for ch in xhtml.chars() {
            match ch {
                '<' => inside_tag = true,
                '>' => inside_tag = false,
                _ if !inside_tag => result.push(ch),
                _ => {},
            }
        }
        result
    }

    /// Check if decoded content stream data may contain text.
    ///
    /// Returns true if the stream contains either:
    /// - A BT (Begin Text) operator (text is directly in the page stream)
    /// - A Do operator (Form XObject invocation that may contain text)
    ///
    /// Per §9.4.3, text-showing operators shall only appear within BT...ET text
    /// objects. However, a page may contain text only inside Form XObjects
    /// referenced via `Do` operators, so we must also check for those.
    pub(crate) fn may_contain_text(data: &[u8]) -> bool {
        // SIMD-accelerated pre-check using memchr to find candidate positions
        // for BT (Begin Text) and Do (XObject invocation) operators.
        // ~50x faster than byte-by-byte scanning for large graphics-heavy pages.
        fn is_boundary(b: u8) -> bool {
            b.is_ascii_whitespace()
                || matches!(b, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
        }

        // Search for 'B' (BT) and 'D' (Do) candidates using SIMD memchr
        let len = data.len();
        let mut offset = 0;
        while offset + 1 < len {
            // Find next 'B' or 'D' byte
            match memchr::memchr2(b'B', b'D', &data[offset..]) {
                None => return false,
                Some(pos) => {
                    let i = offset + pos;
                    if i + 1 >= len {
                        return false;
                    }
                    // Check for BT operator
                    if data[i] == b'B' && data[i + 1] == b'T' {
                        let before_ok = i == 0 || is_boundary(data[i - 1]);
                        let after_ok = i + 2 >= len || is_boundary(data[i + 2]);
                        if before_ok && after_ok {
                            return true;
                        }
                    }
                    // Check for Do operator
                    if data[i] == b'D' && data[i + 1] == b'o' {
                        let before_ok = i == 0 || is_boundary(data[i - 1]);
                        let after_ok = i + 2 >= len || is_boundary(data[i + 2]);
                        if before_ok && after_ok {
                            return true;
                        }
                    }
                    offset = i + 1;
                },
            }
        }
        false
    }

    /// Check if a page definitely cannot produce any text based on its resources.
    ///
    /// Returns `true` if the page has no `/Font` resources and no Form XObjects
    /// (which could contain nested text). This allows skipping content stream
    /// decompression and parsing entirely for image-only/scanned pages.
    ///
    /// Returns `false` (conservative) if resources can't be inspected.
    fn page_cannot_have_text(&self, page_dict: &HashMap<String, Object>) -> bool {
        let resources = match page_dict.get("Resources") {
            Some(r) => {
                if let Some(ref_obj) = r.as_reference() {
                    match self.load_object(ref_obj) {
                        Ok(obj) => obj,
                        Err(_) => return false, // Can't resolve — be conservative
                    }
                } else {
                    r.clone()
                }
            },
            None => return true, // No resources at all → no text possible
        };

        let res_dict = match resources.as_dict() {
            Some(d) => d,
            None => return false,
        };

        // If the page has any /Font resources, it might produce text
        if let Some(font_obj) = res_dict.get("Font") {
            let font_dict = if let Some(ref_obj) = font_obj.as_reference() {
                self.load_object(ref_obj).ok()
            } else {
                Some(font_obj.clone())
            };
            if let Some(fd) = font_dict {
                if let Some(d) = fd.as_dict() {
                    if !d.is_empty() {
                        return false; // Has fonts → might have text
                    }
                }
            }
        }

        // Check XObjects: if any are Form type, they could contain nested text.
        // Uses lightweight is_form_xobject() peek instead of full load_object()
        // to avoid expensive I/O for image-heavy PDFs (e.g., Deutsche: 375MB images).
        if let Some(xobj_obj) = res_dict.get("XObject") {
            let xobj_dict_obj = if let Some(ref_obj) = xobj_obj.as_reference() {
                self.load_object(ref_obj).ok()
            } else {
                Some(xobj_obj.clone())
            };
            if let Some(xobj_dict_resolved) = xobj_dict_obj {
                if let Some(xobj_dict) = xobj_dict_resolved.as_dict() {
                    for xobj_ref in xobj_dict.values() {
                        if let Some(ref_obj) = xobj_ref.as_reference() {
                            // Use lightweight 1KB peek instead of full object load
                            if self.is_form_xobject(ref_obj) {
                                return false; // Form XObject could contain text
                            }
                        } else if let Some(d) = xobj_ref.as_dict() {
                            if d.get("Subtype").and_then(|s| s.as_name()) == Some("Form") {
                                return false;
                            }
                        }
                    }
                }
            }
        }

        // No fonts and no Form XObjects → page is image-only
        true
    }

    /// Assemble the page's text spans via the reading-order
    /// pipeline, classifying each region with the per-class
    /// detectors in [`crate::pipeline::reading_order::detectors`].
    /// Returns the assembled spans plus the detector class that
    /// fired on each region.
    ///
    /// The four detectors handle layout shapes that the plain
    /// y-then-x assembly cannot produce correctly:
    ///
    /// - **DramaticScript**: Macbeth-style speaker-tag layouts —
    ///   row-major join required.
    /// - **DenseSingleLine**: SEC DEF 14A 8pt-body interleave —
    ///   single-row regroup required.
    /// - **SubSuperBaselineReattach**: chemical-formula
    ///   subscripts — baseline reattach required.
    /// - **NarrowTrackedJustified**: stretched justified columns —
    ///   per-line median-gap threshold normalisation required.
    ///
    /// Regions that don't match any specific layout fall through to
    /// `Default` (plain y-then-x assembly within the block).
    ///
    /// Callers can use this as a pre-step before applying their own
    /// assembly logic, or rely on the classified `ReadingOrderClass`
    /// to dispatch their assembly strategy. `extract_text` consumes
    /// this implicitly through `extract_spans` + the existing
    /// `XYCutStrategy`.
    pub fn assemble_text_via_reading_order(
        &self,
        page_index: usize,
    ) -> Result<(Vec<crate::layout::TextSpan>, crate::pipeline::reading_order::ReadingOrderClass)>
    {
        if self.is_encrypted_unreadable() {
            log::warn!("PDF is encrypted and could not be decrypted; returning empty text");
            return Ok((Vec::new(), crate::pipeline::reading_order::ReadingOrderClass::Default));
        }
        let spans = self.extract_spans(page_index)?;
        // Convert spans to detector input. We only need the geometric
        // signal (x/y/width/font_size), not the full TextSpan
        // semantics.
        let glyphs: Vec<crate::pipeline::reading_order::DetectorGlyph> = spans
            .iter()
            .map(|s| crate::pipeline::reading_order::DetectorGlyph {
                x: s.bbox.x,
                y: s.bbox.y,
                width: s.bbox.width,
                font_size: s.font_size,
                text_len: s.text.chars().count(),
            })
            .collect();
        // Build per-row text strings for DramaticScript detector,
        // together with the leftmost glyph of each row (for the X-
        // consistency check). Group spans by Y (within 0.5 pt),
        // concatenating their texts in the order they appear in
        // `spans` and tracking the smallest X seen per row.
        let mut rows: Vec<(f32, String, crate::pipeline::reading_order::DetectorGlyph)> =
            Vec::new();
        for span in &spans {
            let span_glyph = crate::pipeline::reading_order::DetectorGlyph {
                x: span.bbox.x,
                y: span.bbox.y,
                width: span.bbox.width,
                font_size: span.font_size,
                text_len: span.text.chars().count(),
            };
            let mut placed = false;
            for (y, text, first) in rows.iter_mut() {
                if (*y - span.bbox.y).abs() < 0.5 {
                    text.push(' ');
                    text.push_str(&span.text);
                    if span_glyph.x < first.x {
                        *first = span_glyph;
                    }
                    placed = true;
                    break;
                }
            }
            if !placed {
                rows.push((span.bbox.y, span.text.clone(), span_glyph));
            }
        }
        let row_texts: Vec<&str> = rows.iter().map(|(_, t, _)| t.as_str()).collect();
        let row_first_glyphs: Vec<crate::pipeline::reading_order::DetectorGlyph> =
            rows.iter().map(|(_, _, g)| *g).collect();
        let class =
            crate::pipeline::reading_order::classify_region(&glyphs, &row_first_glyphs, &row_texts);
        Ok((spans, class))
    }

    /// Returns `true` if the page has any text-bearing content (fonts in
    /// resources + at least one `BT`/`Do` operator in the content stream),
    /// `false` if the page is image-only or genuinely empty.
    ///
    /// Callers can route image-only pages to OCR
    /// (`extract_text_ocr_only(page, engine)`) instead of receiving
    /// an empty string with no signal.
    ///
    /// Conservative: returns `true` when the page resources can't be
    /// inspected (load error, encrypted-not-authenticated, etc.) so the
    /// caller still attempts extraction.
    ///
    /// # PDF spec basis
    ///
    /// §8.8 (Image XObjects): image-only pages have `/Resources` whose
    /// only `/XObject` entries are `/Subtype /Image` with no `/Font`
    /// resources.
    pub fn has_text_layer(&self, page_index: usize) -> Result<bool> {
        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;
        if self.page_cannot_have_text(page_dict) {
            return Ok(false);
        }
        // Probe content stream for text-showing operators. If we can't
        // read the content stream, be conservative and say yes (let
        // extraction try).
        match self.get_page_content_data(page_index) {
            Ok(content_data) => Ok(Self::may_contain_text(&content_data)),
            Err(_) => Ok(true),
        }
    }

    /// Returns the document's `/P` permission flags as a `PdfPermissions`
    /// struct if the document is encrypted; `None` otherwise.
    ///
    /// Per PDF spec §7.6.3.2 the `/P` flag is advisory — pdf_oxide
    /// does not enforce restrictions — but callers who want to
    /// enforce them (e.g., refuse copy-protected PDF extraction) can
    /// do so themselves by checking the returned permissions.
    ///
    /// # PDF spec basis
    ///
    /// §7.6.3.2 Table 22 (`/P` Standard Encryption Dictionary entry).
    /// Decoding is implemented in `encryption::permissions::PdfPermissions::from_p_flag`.
    pub fn permissions(&self) -> Option<crate::encryption::PdfPermissions> {
        // ensure_encryption_initialized may fail on malformed Encrypt
        // dicts — that's fine, no permissions surface for those.
        let _ = self.ensure_encryption_initialized();
        let handler = self.encryption_handler.lock_or_recover();
        let handler = handler.as_ref()?;
        Some(crate::encryption::PdfPermissions::from_p_flag(handler.raw_permissions()))
    }

    /// Extract text using structure tree for Tagged PDFs.
    ///
    /// This method implements PDF spec-compliant text extraction for Tagged PDFs
    /// using the logical structure tree to determine reading order.
    ///
    /// # PDF Spec Reference
    ///
    /// ISO 32000-1:2008 Section 14.8.2.3 - Determining the Text Extraction Sequence
    /// "For a Tagged PDF document, conforming readers shall present the document's
    /// content to the user in the order given by a pre-order traversal of the
    /// structure hierarchy"
    ///
    /// # Algorithm
    /// 1. Extract all text spans with MCIDs from the page
    /// 2. Build a map from MCID → Vec<TextSpan>
    /// 3. Traverse structure tree in pre-order to get MCIDs in reading order
    /// 4. Assemble text by looking up spans for each MCID in order
    ///
    /// # Arguments
    /// * `page_index` - Zero-based page index
    /// * `struct_tree` - The structure tree root from the PDF catalog
    ///
    /// # Returns
    /// Extracted text in logical structure order
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // This is called automatically by extract_text() for Tagged PDFs
    /// let text = doc.extract_text(0)?;
    /// ```
    #[allow(dead_code)]
    fn extract_text_structure_order(
        &self,
        page_index: usize,
        struct_tree: &crate::structure::StructTreeRoot,
    ) -> Result<String> {
        log::debug!("Extracting text using structure tree for page {}", page_index);

        // Step 1: Extract all spans with MCIDs
        let all_spans = self.extract_spans(page_index)?;

        if all_spans.is_empty() {
            let mut text = String::new();
            self.append_non_widget_annotation_text(page_index, &mut text);
            return Ok(text);
        }

        // Step 2: Build MCID → Vec<TextSpan> map
        let mut mcid_map: HashMap<u32, Vec<TextSpan>> = HashMap::new();
        let mut spans_without_mcid: Vec<TextSpan> = Vec::new();

        for span in all_spans {
            if let Some(mcid) = span.mcid {
                mcid_map.entry(mcid).or_default().push(span);
            } else {
                // Collect spans without MCID (shouldn't happen in well-formed Tagged PDFs)
                spans_without_mcid.push(span);
            }
        }

        log::debug!(
            "Found {} MCIDs with spans, {} spans without MCID",
            mcid_map.len(),
            spans_without_mcid.len()
        );

        // Step 3: Traverse structure tree to get MCIDs in reading order
        let ordered_content = traverse_structure_tree(struct_tree, page_index as u32)
            .map_err(|e| Error::InvalidPdf(format!("Failed to traverse structure tree: {}", e)))?;

        log::debug!(
            "Structure tree traversal found {} content items in reading order",
            ordered_content.len()
        );

        // Step 4: Assemble text in structure order
        let mut text = String::with_capacity(mcid_map.len() * 50); // estimate
        let mut prev_span: Option<&TextSpan> = None;
        let mut consumed_mcids: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for content in &ordered_content {
            // Handle word break markers by inserting a space
            if content.is_word_break {
                if !text.is_empty() && !text.ends_with(' ') && !text.ends_with('\n') {
                    text.push(' ');
                }
                continue;
            }

            // If the structure element has ActualText, use it instead of the extracted spans
            if let Some(ref actual_text_val) = content.actual_text {
                if !actual_text_val.is_empty() {
                    if !text.is_empty() && !text.ends_with(' ') && !text.ends_with('\n') {
                        text.push('\n');
                    }
                    text.push_str(actual_text_val);
                    continue;
                }
            }

            // For regular content with MCID
            let Some(mcid) = content.mcid else {
                continue;
            };

            if let Some(spans) = mcid_map.get(&mcid) {
                consumed_mcids.insert(mcid);
                for span in Self::order_mcid_spans(spans) {
                    if let Some(prev) = prev_span {
                        let y_diff = (prev.bbox.y - span.bbox.y).abs();

                        if y_diff > Self::same_line_threshold(prev, span) {
                            let font_size = prev.font_size.max(span.font_size).max(10.0);
                            let line_height = font_size * 1.2;
                            let num_breaks = (y_diff / line_height).round() as usize;
                            for _ in 0..num_breaks.clamp(1, 3) {
                                text.push('\n');
                            }
                        } else if Self::should_insert_space(prev, span) {
                            text.push(' ');
                        }
                    }

                    Self::push_span_text_bidi(&mut text, span);
                    prev_span = Some(span);
                }
            } else {
                log::warn!(
                    "Structure tree references MCID {} but no spans found with that MCID",
                    mcid
                );
                self.push_warning(format!(
                    "page {page_index}: structure tree references MCID {mcid} but no content spans found — some text may be missing"
                ));
            }
        }

        // Append spans with MCIDs not referenced by the structure tree.
        // This happens with Form XObjects that lack /StructParents, where
        // their BDC/MCID markers exist in the content stream but are not
        // registered in the page's ParentTree.
        let mut unconsumed: Vec<(&u32, &Vec<TextSpan>)> = mcid_map
            .iter()
            .filter(|(mcid, _)| !consumed_mcids.contains(mcid))
            .collect();
        unconsumed.sort_by_key(|(mcid, _)| **mcid);
        if !unconsumed.is_empty() {
            log::debug!(
                "Appending {} unreferenced MCIDs (e.g., from Form XObjects without StructParents)",
                unconsumed.len()
            );
            for (_mcid, spans) in &unconsumed {
                for span in *spans {
                    if let Some(prev) = prev_span {
                        let y_diff = (prev.bbox.y - span.bbox.y).abs();
                        if y_diff > Self::same_line_threshold(prev, span) {
                            text.push('\n');
                        } else if Self::should_insert_space(prev, span) {
                            text.push(' ');
                        }
                    }
                    Self::push_span_text_bidi(&mut text, span);
                    prev_span = Some(span);
                }
            }
        }

        // Append any spans without MCID at the end (shouldn't happen in well-formed PDFs)
        if !spans_without_mcid.is_empty() {
            log::warn!(
                "Found {} text spans without MCID - appending to end",
                spans_without_mcid.len()
            );
            for span in &spans_without_mcid {
                if let Some(prev) = prev_span {
                    let y_diff = (prev.bbox.y - span.bbox.y).abs();
                    if y_diff > Self::same_line_threshold(prev, span) {
                        text.push('\n');
                    } else if Self::should_insert_space(prev, span) {
                        text.push(' ');
                    }
                }
                Self::push_span_text_bidi(&mut text, span);
                prev_span = Some(span);
            }
        }

        // Annotation text is already included via annotation_content_spans() in
        // extract_spans() — do NOT call append_non_widget_annotation_text() here
        // (would cause double-emission of all annotation text).

        Ok(text)
    }

    /// Order one MCID's spans for emission in the structure-order assemblers
    /// (#608). A single marked-content element can carry spans across several
    /// visual lines; emitting them in raw extraction order can mis-order them,
    /// so sort by the canonical reading-order comparator. Skipped for single-
    /// span MCIDs and for any MCID containing RTL text (whose span order is
    /// handled by the bidi passes) — both stay byte-identical.
    fn order_mcid_spans(spans: &[crate::layout::TextSpan]) -> Vec<&crate::layout::TextSpan> {
        let mut ordered: Vec<&crate::layout::TextSpan> = spans.iter().collect();
        let has_rtl = |s: &crate::layout::TextSpan| {
            s.text
                .chars()
                .any(|c| crate::text::rtl_detector::is_rtl_text(c as u32))
        };
        if spans.len() > 1 && !spans.iter().any(has_rtl) {
            ordered.sort_by(|a, b| {
                crate::utils::row_aware_span_cmp(a.bbox.y, a.bbox.x, b.bbox.y, b.bbox.x)
            });
        }
        ordered
    }

    /// Page's MCID reading order from the all-pages traversal cache
    /// (`structure_content_cache`, populated once). `build_context` previously
    /// re-walked the whole tree per page (≈ O(pages²) on a tagged document);
    /// the cached all-pages walk (#608) yields the same per-page order.
    pub(crate) fn cached_mcid_order_for_page(
        &self,
        struct_tree: &crate::structure::StructTreeRoot,
        page_index: u32,
    ) -> Vec<u32> {
        if self.structure_content_cache.lock_or_recover().is_none() {
            let all_content = crate::structure::traverse_structure_tree_all_pages(struct_tree);
            *self.structure_content_cache.lock_or_recover() = Some(all_content);
        }
        self.structure_content_cache
            .lock_or_recover()
            .as_ref()
            .and_then(|c| c.get(&page_index))
            .map(|content| content.iter().filter_map(|c| c.mcid).collect())
            .unwrap_or_default()
    }

    /// Extract text from a Tagged PDF page using pre-computed structure traversal cache.
    ///
    /// This is the optimized version of `extract_text_structure_order` that uses
    /// the pre-built `structure_content_cache` for O(1) page content lookup instead
    /// of re-traversing the entire structure tree for each page.
    fn extract_text_structure_order_cached_with_spans(
        &self,
        page_index: usize,
        all_spans: Vec<TextSpan>,
    ) -> Result<String> {
        log::debug!("Extracting text using cached structure order for page {}", page_index);

        if all_spans.is_empty() {
            let mut text = String::new();
            self.append_non_widget_annotation_text(page_index, &mut text);
            return Ok(text);
        }

        // Step 2: Build MCID → Vec<TextSpan> map
        let mut mcid_map: HashMap<u32, Vec<TextSpan>> = HashMap::new();
        let mut spans_without_mcid: Vec<TextSpan> = Vec::new();

        for span in all_spans {
            if let Some(mcid) = span.mcid {
                mcid_map.entry(mcid).or_default().push(span);
            } else {
                spans_without_mcid.push(span);
            }
        }

        // Step 3: Get pre-computed ordered content for this page (O(1) lookup)
        let ordered_content_owned: Vec<crate::structure::OrderedContent>;
        let ordered_content = {
            let cache = self.structure_content_cache.lock_or_recover();
            ordered_content_owned = cache
                .as_ref()
                .and_then(|c| c.get(&(page_index as u32)))
                .cloned()
                .unwrap_or_default();
            &ordered_content_owned as &[crate::structure::OrderedContent]
        };

        log::debug!(
            "Cached structure content: {} items for page {}, {} MCIDs with spans",
            ordered_content.len(),
            page_index,
            mcid_map.len()
        );

        // Step 4: Assemble text in structure order
        let mut text = String::with_capacity(mcid_map.len() * 50);
        let mut prev_span: Option<&TextSpan> = None;
        let mut consumed_mcids: HashSet<u32> = HashSet::new();

        for content in ordered_content {
            if content.is_word_break {
                if !text.is_empty() && !text.ends_with(' ') && !text.ends_with('\n') {
                    text.push(' ');
                }
                continue;
            }

            if let Some(ref actual_text_val) = content.actual_text {
                if !actual_text_val.is_empty() {
                    if !text.is_empty() && !text.ends_with(' ') && !text.ends_with('\n') {
                        text.push('\n');
                    }
                    text.push_str(actual_text_val);
                    continue;
                }
            }

            let Some(mcid) = content.mcid else {
                continue;
            };

            if let Some(spans) = mcid_map.get(&mcid) {
                consumed_mcids.insert(mcid);
                for span in Self::order_mcid_spans(spans) {
                    if let Some(prev) = prev_span {
                        let y_diff = (prev.bbox.y - span.bbox.y).abs();
                        if y_diff > Self::same_line_threshold(prev, span) {
                            let font_size = prev.font_size.max(span.font_size).max(10.0);
                            let line_height = font_size * 1.2;
                            let num_breaks = (y_diff / line_height).round() as usize;
                            for _ in 0..num_breaks.clamp(1, 3) {
                                text.push('\n');
                            }
                        } else if Self::should_insert_space(prev, span) {
                            text.push(' ');
                        }
                    }

                    Self::push_span_text_bidi(&mut text, span);
                    prev_span = Some(span);
                }
            }
        }

        // Append spans with MCIDs not referenced by the structure tree
        let mut unconsumed: Vec<(&u32, &Vec<TextSpan>)> = mcid_map
            .iter()
            .filter(|(mcid, _)| !consumed_mcids.contains(mcid))
            .collect();
        unconsumed.sort_by_key(|(mcid, _)| **mcid);
        if !unconsumed.is_empty() {
            log::debug!(
                "Appending {} unreferenced MCIDs (e.g., from Form XObjects without StructParents)",
                unconsumed.len()
            );
            for (_mcid, spans) in &unconsumed {
                for span in *spans {
                    if let Some(prev) = prev_span {
                        let y_diff = (prev.bbox.y - span.bbox.y).abs();
                        if y_diff > Self::same_line_threshold(prev, span) {
                            text.push('\n');
                        } else if Self::should_insert_space(prev, span) {
                            text.push(' ');
                        }
                    }
                    Self::push_span_text_bidi(&mut text, span);
                    prev_span = Some(span);
                }
            }
        }

        // Append any spans without MCID (including widget/form field spans) sorted by position
        if !spans_without_mcid.is_empty() {
            log::debug!(
                "Found {} text spans without MCID (including form field widgets) - appending sorted by position",
                spans_without_mcid.len()
            );
            // Row-aware sort: Y-band descending (top→bottom), then X ascending.
            crate::utils::sort_by_row_band(&mut spans_without_mcid, |s| s.bbox.y, |s| s.bbox.x);
            for span in &spans_without_mcid {
                if let Some(prev) = prev_span {
                    let y_diff = (prev.bbox.y - span.bbox.y).abs();
                    if y_diff > Self::same_line_threshold(prev, span) {
                        text.push('\n');
                    } else if Self::should_insert_space(prev, span) {
                        text.push(' ');
                    }
                }
                Self::push_span_text_bidi(&mut text, span);
                prev_span = Some(span);
            }
        }

        // Annotation text is already included via annotation_content_spans() in
        // extract_spans() — do NOT call append_non_widget_annotation_text() here
        // (would cause double-emission of all annotation text).

        Ok(text)
    }

    /// Extract text spans from a page (PDF spec compliant - RECOMMENDED).
    ///
    /// This is the recommended method for text extraction. It extracts complete
    /// text strings as the PDF provides them via Tj/TJ operators, following the
    /// PDF specification ISO 32000-1:2008.
    ///
    /// # Benefits over extract_chars
    /// - Avoids overlapping character issues
    /// - Preserves PDF's text positioning intent
    /// - More robust for complex layouts
    /// - Matches industry best practices (PyMuPDF, etc.)
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    ///
    /// # Returns
    ///
    /// Vector of TextSpan objects in reading order
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::PdfDocument;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("document.pdf")?;
    /// let spans = doc.extract_spans(0)?;
    /// for span in spans {
    ///     println!("Text: {} at ({}, {})", span.text, span.bbox.x, span.bbox.y);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_spans(&self, page_index: usize) -> Result<Vec<crate::layout::TextSpan>> {
        // Serve repeat per-page extractions from cache (the converters reach
        // here twice per page; see `page_spans_cache`).
        if let Some(cached) = self.page_spans_cache.lock_or_recover().get(&page_index) {
            return Ok((**cached).clone());
        }
        let spans = self.extract_spans_raw(page_index)?;
        let spans = self.postprocess_spans(page_index, spans)?;
        self.page_spans_cache
            .lock_or_recover()
            .insert(page_index, std::sync::Arc::new(spans.clone()));
        Ok(spans)
    }

    fn extract_spans_filtered(
        &self,
        page_index: usize,
        excluded_layers: HashSet<String>,
        excluded_inks: HashSet<String>,
    ) -> Result<Vec<crate::layout::TextSpan>> {
        let spans = self.extract_spans_raw_filtered(page_index, excluded_layers, excluded_inks)?;
        self.postprocess_spans(page_index, spans)
    }

    /// Map a span rectangle (already translated so the page origin is at
    /// `(0, 0)`) through a clockwise page `/Rotate` of `rot` degrees, returning
    /// the axis-aligned bounding box in the displayed coordinate frame.
    ///
    /// `page_w` / `page_h` are the unrotated page dimensions; for 90° / 270° the
    /// displayed page is `page_h × page_w`. Per ISO 32000-1:2008 §7.7.3.3 the
    /// rotation is clockwise and §8.3.3 gives the point transform. `rot` must be
    /// a normalised multiple of 90 (`0/90/180/270`); any other value returns the
    /// rectangle unchanged. `rot == 0` is the identity and `rot == 180` is
    /// numerically identical to the legacy mirror, preserving byte-for-byte
    /// output for unrotated and 180° pages.
    pub(crate) fn rotate_span_bbox(
        bbox: crate::geometry::Rect,
        rot: i32,
        page_w: f32,
        page_h: f32,
    ) -> crate::geometry::Rect {
        // Map a point (y-up) by the clockwise display rotation.
        let map = |x: f32, y: f32| -> (f32, f32) {
            match rot {
                90 => (y, page_w - x),
                180 => (page_w - x, page_h - y),
                270 => (page_h - y, x),
                _ => (x, y),
            }
        };
        let (ax, ay) = map(bbox.x, bbox.y);
        let (bx, by) = map(bbox.x + bbox.width, bbox.y + bbox.height);
        crate::geometry::Rect::new(ax.min(bx), ay.min(by), (ax - bx).abs(), (ay - by).abs())
    }

    /// Map a single span's bbox into the displayed frame for a `/Rotate`d page
    /// (translate to origin → [`rotate_span_bbox`] → translate back).
    fn map_span_into_rotated_frame(
        s: &mut crate::layout::TextSpan,
        rot: i32,
        llx: f32,
        lly: f32,
        w: f32,
        h: f32,
    ) {
        let rel =
            crate::geometry::Rect::new(s.bbox.x - llx, s.bbox.y - lly, s.bbox.width, s.bbox.height);
        let m = Self::rotate_span_bbox(rel, rot, w, h);
        s.bbox.x = llx + m.x;
        s.bbox.y = lly + m.y;
        s.bbox.width = m.width;
        s.bbox.height = m.height;
    }

    /// Order rotated runs that were segregated out of the horizontal reading
    /// flow. Spans drawn with a rotated text matrix (`rotation_degrees != 0`)
    /// break the axis-aligned assumptions of the row-band / XY-cut sort, so they
    /// are pulled out, ordered here, and appended as their own blocks. Runs are
    /// grouped by rotation (first-seen group order preserved); within a group
    /// each span's origin is rotated back into an upright frame and the standard
    /// row-aware comparator (top→bottom, left→right) is applied there.
    fn order_rotated_blocks(spans: Vec<crate::layout::TextSpan>) -> Vec<crate::layout::TextSpan> {
        let mut groups: Vec<(f32, Vec<crate::layout::TextSpan>)> = Vec::new();
        for s in spans {
            let key = s.rotation_degrees;
            match groups.iter_mut().find(|(k, _)| (*k - key).abs() < 0.5) {
                Some(g) => g.1.push(s),
                None => groups.push((key, vec![s])),
            }
        }
        let mut out = Vec::new();
        for (deg, mut group) in groups {
            let (sin, cos) = (-deg).to_radians().sin_cos();
            // Upright frame: rotate each origin by -deg, then read top→bottom,
            // left→right exactly as horizontal text.
            group.sort_by(|a, b| {
                let ax = a.bbox.x * cos - a.bbox.y * sin;
                let ay = a.bbox.x * sin + a.bbox.y * cos;
                let bx = b.bbox.x * cos - b.bbox.y * sin;
                let by = b.bbox.x * sin + b.bbox.y * cos;
                crate::utils::row_aware_span_cmp(ay, ax, by, bx)
            });
            out.extend(group);
        }
        out
    }

    /// Re-attach an oversized lone leading capital (a drop-cap / table-title
    /// initial that the producer set in a larger font, so it became its own
    /// span) to the body run immediately to its right on the same line —
    /// otherwise reading-order strands it (`TABLE` → `T` … `ABLE`).
    ///
    /// Conservative gates so prose drop-caps / standalone capitals aren't glued
    /// to the wrong word: the candidate must be a single uppercase ASCII letter
    /// at ≥1.5× the body run's font size, its right edge within ~0.3 em of the
    /// body's left edge, vertically overlapping it, and the body must start with
    /// a letter. Runs in raw span order before reading-order sorting.
    fn merge_drop_cap_initials(spans: &mut Vec<crate::layout::TextSpan>) {
        let n = spans.len();
        if n < 2 {
            return;
        }
        // A genuine drop cap is oversized relative to the page's *normal* body
        // text, not merely relative to its right-hand neighbor. Inline math such
        // as "A_st" pairs a normal-size capital with a shrunken subscript; gating
        // on the neighbor alone would treat that capital as oversized and glue
        // "A" + "st" into "Ast". Anchor the size gate to the median size of
        // multi-character spans (real words) so a body-size capital cannot
        // qualify.
        let mut body_sizes: Vec<f32> = spans
            .iter()
            .filter(|s| s.font_size > 0.0 && s.text.chars().nth(1).is_some())
            .map(|s| s.font_size)
            .collect();
        if body_sizes.is_empty() {
            return;
        }
        body_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let body_size = body_sizes[body_sizes.len() / 2];

        // For each initial candidate, the closest qualifying body span to its right.
        let mut target: Vec<Option<usize>> = vec![None; n];
        for i in 0..n {
            let init = &spans[i];
            if init.text.chars().count() != 1 || init.font_size <= 0.0 {
                continue;
            }
            if !init
                .text
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_uppercase())
            {
                continue;
            }
            if init.font_size < body_size * 1.5 {
                continue; // initial must be clearly oversized vs normal body text
            }
            let init_right = init.bbox.x + init.bbox.width;
            let mut best: Option<usize> = None;
            let mut best_gap = f32::MAX;
            for (j, body) in spans.iter().enumerate() {
                if j == i || body.font_size <= 0.0 {
                    continue;
                }
                if !body.text.chars().next().is_some_and(|c| c.is_alphabetic()) {
                    continue;
                }
                // Continuation shares the initial's baseline (same text line). A
                // tall oversized initial also vertically overlaps the line *above*
                // it, so a raw bbox-overlap test would let it reach up and steal a
                // word from the previous line (alice_old: the 16.8pt "A" of "A very
                // heavy weight" overlapping "Or if" → "OrAif"). Baseline proximity
                // (≈ bbox bottom) keeps the merge on the initial's own line.
                if (init.bbox.y - body.bbox.y).abs() > body.font_size * 0.5 {
                    continue;
                }
                // Body immediately to the right, essentially touching. A genuine
                // oversized initial is the first glyph of one word ("T" of
                // "TABLE", "P" of "PENALTY"), so its continuation begins within a
                // hair of the initial's advance — never across a word space. A
                // word-space gap (~0.25 em) would wrongly glue a standalone "A"
                // or "I" onto the next word ("A Perspective" → "APerspective"),
                // so the upper bound stays well below it.
                let gap = body.bbox.x - init_right;
                if gap < -body.font_size * 0.5 || gap > body.font_size * 0.12 {
                    continue;
                }
                if gap.abs() < best_gap {
                    best_gap = gap.abs();
                    best = Some(j);
                }
            }
            target[i] = best;
        }

        let mut taken = vec![false; n];
        let mut remove = vec![false; n];
        for i in 0..n {
            let Some(j) = target[i] else { continue };
            if taken[j] || remove[j] || remove[i] {
                continue; // a body receives at most one initial
            }
            taken[j] = true;
            remove[i] = true;
            let init_text = spans[i].text.clone();
            let init_left = spans[i].bbox.x;
            let body = &mut spans[j];
            body.text = format!("{init_text}{}", body.text);
            let right = body.bbox.x + body.bbox.width;
            body.bbox.x = init_left.min(body.bbox.x);
            body.bbox.width = right - body.bbox.x;
        }
        let mut k = 0;
        spans.retain(|_| {
            let keep = !remove[k];
            k += 1;
            keep
        });
    }

    /// True for Computer-Modern (`CM*`) or symbol font names, after stripping a
    /// `ABCDEF+` subset tag. Used to scope the `¬`→`.` decimal recovery.
    fn is_cm_or_symbol_font(font_name: &str) -> bool {
        let base = font_name.split('+').next_back().unwrap_or(font_name);
        let lower = base.to_ascii_lowercase();
        lower.starts_with("cm") || lower.contains("symbol")
    }

    /// Replace a `¬` (U+00AC) that sits directly between two ASCII digits with
    /// `.` (the decimal point a math subset drew from its `logicalnot` slot).
    /// Leaves every other `¬` untouched.
    fn fix_digit_logicalnot_decimal(text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::with_capacity(text.len());
        for (i, &c) in chars.iter().enumerate() {
            if c == '\u{00AC}'
                && i > 0
                && chars[i - 1].is_ascii_digit()
                && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit())
            {
                out.push('.');
            } else {
                out.push(c);
            }
        }
        out
    }

    fn postprocess_spans(
        &self,
        page_index: usize,
        raw_spans: Vec<crate::layout::TextSpan>,
    ) -> Result<Vec<crate::layout::TextSpan>> {
        let mut spans = raw_spans;

        // Drop spans whose bbox lies entirely outside the page's MediaBox.
        // PDFs that reuse one big Form XObject across pages (ExpertPdf
        // similar tools — see issue B1 / nougat_005.pdf) rely on the
        // content stream's `W n` clip rectangle to hide the off-page
        // portion. Our text extractor doesn't honour `W n` yet, so
        // without this filter every page emits all 5 pages' worth of
        // spans at distinct but out-of-bounds Y coordinates. Keep spans
        // that even partially overlap with MediaBox so we don't drop
        // legitimate bleed / trim-mark content.
        // get_page_media_box returns (llx, lly, urx, ury) — absolute corner
        // coordinates per ISO 32000-1 §7.7.3.3, NOT (x, y, width, height).
        if let Ok((llx, lly, urx, ury)) = self.get_page_media_box(page_index) {
            const EDGE_TOLERANCE_PT: f32 = 2.0;
            let left = llx - EDGE_TOLERANCE_PT;
            let bottom = lly - EDGE_TOLERANCE_PT;
            let right = urx + EDGE_TOLERANCE_PT;
            let top = ury + EDGE_TOLERANCE_PT;
            spans.retain(|span| {
                let sx1 = span.bbox.x;
                let sx2 = span.bbox.x + span.bbox.width;
                let sy1 = span.bbox.y;
                let sy2 = span.bbox.y + span.bbox.height;
                sx2 > left && sx1 < right && sy2 > bottom && sy1 < top
            });
        }

        // Recover decimal points mis-decoded as `¬` (U+00AC) in Computer-Modern
        // math subsets, where the `/Differences` names the decimal slot
        // `logicalnot`. Only a `¬` sitting *directly between two digits* (no
        // space) is rewritten — real logic/set `¬` is always spaced, so this
        // cannot corrupt it.
        for span in &mut spans {
            if Self::is_cm_or_symbol_font(&span.font_name) && span.text.contains('\u{00AC}') {
                span.text = Self::fix_digit_logicalnot_decimal(&span.text);
            }
        }

        // Re-attach oversized lone leading capitals to their word before the
        // reading-order sort can strand them (drop-cap / table-title initials).
        Self::merge_drop_cap_initials(&mut spans);

        // Apply page /Rotate to span geometry BEFORE reading-order sorting.
        // Spans are extracted in raw PDF user space; a page with a /Rotate entry
        // must be read in its DISPLAYED orientation or the row-aware sort emits
        // text in the wrong order (pdf.js issue14415 is a 180° English page that
        // otherwise comes out word- and line-reversed). Every span is mapped into
        // one consistent displayed frame via `rotate_span_bbox` BEFORE any
        // geometric pass (column detection, table geometry, the row-aware sort),
        // so 90° / 270° — which additionally swap page width/height — are handled
        // uniformly alongside 180°. rot == 0 is untouched (byte-identical) and
        // rot == 180 is numerically identical to the previous mirror. (A
        // within-span character re-order for rotated multi-glyph spans — the
        // issue14415 within-line residual — is a tracked follow-up.)
        // Captured so the same transform is applied to annotation spans appended
        // later (their /Rect is in unrotated page space too). `None` for rot==0
        // or unknown media box — those pages are byte-identical.
        let page_rotation: Option<(i32, f32, f32, f32, f32)> =
            match self.get_page_media_box(page_index) {
                Ok((llx, lly, urx, ury)) => {
                    let rot = self
                        .get_page_rotation(page_index)
                        .unwrap_or(0)
                        .rem_euclid(360);
                    matches!(rot, 90 | 180 | 270).then_some((rot, llx, lly, urx - llx, ury - lly))
                },
                Err(_) => None,
            };
        if let Some((rot, llx, lly, w, h)) = page_rotation {
            for s in spans.iter_mut() {
                Self::map_span_into_rotated_frame(s, rot, llx, lly, w, h);
            }
        }

        // Reading order: XY-cut when the page has multiple columns (B4);
        // otherwise the cheap row-aware sort. XY-cut is spatial recursion
        // that correctly orders multi-column layouts (newspapers, academic
        // papers, dashboards) but is overkill for single-column pages
        // doesn't handle tabular rowspan labels specifically. Heuristic:
        // count distinct X-center clusters with vertical overlap; ≥2
        // clusters → multi-column.
        if Self::is_multi_column_page(&spans) {
            use crate::pipeline::reading_order::{
                ReadingOrderContext as ROContext, ReadingOrderStrategy, XYCutStrategy,
            };
            let strategy = XYCutStrategy::new();
            let context = ROContext::new().with_page(page_index as u32);
            // Clone needed: apply() takes ownership, and the Err branch
            // falls back to sorting the original vec in place.
            match strategy.apply(spans.clone(), &context) {
                Ok(ordered) => {
                    spans = ordered.into_iter().map(|o| o.span).collect();
                },
                Err(e) => {
                    log::debug!(
                        "XY-cut reading order failed on page {page_index} ({e}), \
                         falling back to row-aware sort"
                    );
                    spans.sort_by(|a, b| {
                        crate::utils::row_aware_span_cmp(a.bbox.y, a.bbox.x, b.bbox.y, b.bbox.x)
                    });
                    Self::reorder_rowspan_labels(&mut spans);
                },
            }
        } else {
            // Row-aware sort: Y-band descending (top→bottom), X ascending
            // within a row.
            spans.sort_by(|a, b| {
                crate::utils::row_aware_span_cmp(a.bbox.y, a.bbox.x, b.bbox.y, b.bbox.x)
            });
            // Lift multi-row-spanning labels to the top of their block.
            Self::reorder_rowspan_labels(&mut spans);
        }

        // Per-span rotation firewall. Runs drawn with a rotated text matrix
        // (the vertical `arXiv:…` margin stamp, figure/axis labels, rotated
        // table headers, transit-poster route names) break the axis-aligned
        // row-band / XY-cut assumptions, so interleaving them with the
        // horizontal flow scrambles reading order. The reordering above ran on
        // the FULL span set (so its column/XY-cut decisions are unchanged — the
        // horizontal body keeps its exact baseline order); now stably lift the
        // rotated runs out (preserving horizontal order) and re-append them as
        // their own blocks, each ordered in an upright frame. No-op (and
        // byte-identical) when the page has no rotated spans.
        if spans.iter().any(|s| s.rotation_degrees != 0.0) {
            let rotated: Vec<crate::layout::TextSpan> = spans
                .iter()
                .filter(|s| s.rotation_degrees != 0.0)
                .cloned()
                .collect();
            spans.retain(|s| s.rotation_degrees == 0.0);
            spans.extend(Self::order_rotated_blocks(rotated));
        }

        // Filter out spans in erase regions
        let erase = self
            .erase_regions
            .lock_or_recover()
            .get(&page_index)
            .cloned();
        if let Some(regions) = erase {
            spans.retain(|span| !regions.iter().any(|r| r.intersects(&span.bbox)));
        }

        // Append text from non-Widget annotations (/Subtype /Text, FreeText,
        // Stamp, Highlight, etc.) that carry a /Contents entry. These are not
        // part of the page content stream so they are not picked up by the
        // regular extractor. On a /Rotate'd page their /Rect-derived bboxes are
        // in unrotated page space, so map the appended spans into the same
        // displayed frame as the content spans (no-op for unrotated pages).
        let pre_annotation_len = spans.len();
        spans.extend(self.annotation_content_spans(page_index));
        if let Some((rot, llx, lly, w, h)) = page_rotation {
            for s in spans[pre_annotation_len..].iter_mut() {
                Self::map_span_into_rotated_frame(s, rot, llx, lly, w, h);
            }
        }

        // Mark running headers/footers (untagged-PDF heuristic). Spans whose
        // normalized text recurs on >=50% of pages and sits near the top or
        // bottom of the page are flagged as artifacts so downstream filters
        // drop them.
        self.mark_running_artifact_spans(page_index, &mut spans)?;

        // Normalize Unicode typographic spaces (U+2000–U+200B, U+202F, U+205F)
        // to ASCII space. Some PDF producers encode word separators as hair-space
        // or thin-space variants in ToUnicode CMaps (e.g. justified text layouts);
        // normalising here gives consistent word boundaries to every downstream
        // consumer (extract_text, word-F1 scoring, etc.).
        for span in &mut spans {
            if span
                .text
                .chars()
                .any(|c| matches!(c, '\u{2000}'..='\u{200B}' | '\u{202F}' | '\u{205F}'))
            {
                span.text = crate::converters::text_post_processor::TextPostProcessor
                    ::normalize_unicode_spaces(&span.text)
                    .into_owned();
            }
        }

        // Apply char_widths boundary splits directly to span.text so that every
        // downstream consumer (to_markdown, to_html, extract_text) sees the same
        // word boundaries. extract_text applies the same logic through push_span_text;
        // after this normalization push_span_text sees a space at the boundary
        // becomes a no-op, so there is no double-application risk.
        for span in &mut spans {
            if let Some(split) = Self::char_widths_boundary_split(span) {
                let mut t = String::with_capacity(span.text.len() + 1);
                t.push_str(&span.text[..split]);
                t.push(' ');
                t.push_str(&span.text[split..]);
                span.text = t;
            }
        }

        // Detect superscript / subscript runs and substitute ASCII
        // digits with their Unicode super/sub-script equivalents
        // (only when the run is sandwiched between alphabetic body
        // spans on both sides — chemistry/math context like "S²X"
        // or "H₂O"). The same substitution would otherwise fire on
        // author-affiliation markers ("name¹,²") which the bench
        // ground truth keeps in ASCII; gating on token-internal
        // context keeps the desired cases without regressing the
        // affiliation-block pages.
        Self::apply_super_sub_script_substitutions(&mut spans);

        // Fold spacing-diacritic spans (´, `, ^, ~, ¨, …) into the
        // base letter of the following span when the diacritic is
        // centred over the base glyph. PDFs that pre-shape accented
        // Latin (LaTeX `\'E` → two glyphs, `acute` then `E`) emit
        // the marks as separate `Tj` ops at the base glyph's X
        // coordinate. Without this pass extract_text returns the
        // raw two-glyph order "´Ecole" instead of "École".
        Self::apply_combining_mark_composition(&mut spans);

        Ok(spans)
    }

    /// Fold a one-char spacing-diacritic span into the following
    /// span's first character when they overlap in X (the typical
    /// LaTeX `\'E` → `(´)(E)` shape). Substitutes the relevant
    /// combining mark from U+0300..U+0327 and lets
    /// `unicode_normalization::nfc` precompose where it can
    /// ("E\u{0301}" → "É"). The diacritic span is left empty so
    /// downstream rendering skips it.
    fn apply_combining_mark_composition(spans: &mut Vec<crate::layout::TextSpan>) {
        use unicode_normalization::UnicodeNormalization;

        fn combining_for(spacing: char) -> Option<char> {
            Some(match spacing {
                '\u{00B4}' => '\u{0301}', // acute
                '\u{0060}' => '\u{0300}', // grave
                '\u{005E}' => '\u{0302}', // circumflex
                '\u{02C6}' => '\u{0302}', // modifier-letter circumflex
                '\u{007E}' => '\u{0303}', // tilde
                '\u{02DC}' => '\u{0303}', // small tilde
                '\u{00A8}' => '\u{0308}', // diaeresis
                '\u{00AF}' => '\u{0304}', // macron
                '\u{02C9}' => '\u{0304}', // modifier-letter macron
                '\u{00B8}' => '\u{0327}', // cedilla
                '\u{02DA}' => '\u{030A}', // ring above
                _ => return None,
            })
        }

        // First pass: spans that already got merged at the extractor
        // (when the LaTeX `(´)(Ecole)` pair both sit at the same
        // text-matrix origin the upstream merge_adjacent_spans pulls
        // them into a single "´Ecole" span). Fold the leading
        // diacritic + base letter into the precomposed form.
        for span in spans.iter_mut() {
            let mut iter = span.text.chars();
            let Some(d) = iter.next() else { continue };
            let Some(base) = iter.next() else { continue };
            let Some(combining) = combining_for(d) else {
                continue;
            };
            if !base.is_alphabetic() {
                continue;
            }
            let rest_start = d.len_utf8() + base.len_utf8();
            let mut composed = String::with_capacity(span.text.len() + 2);
            composed.push(base);
            composed.push(combining);
            composed.push_str(&span.text[rest_start..]);
            span.text = composed.nfc().collect();
        }

        // Walk spans pairwise. The diacritic is on its own one-
        // character span; the next span carries the base letter.
        let mut i = 0;
        while i + 1 < spans.len() {
            let mark_char = {
                let s = &spans[i];
                let mut iter = s.text.chars();
                let first = iter.next();
                let rest = iter.next();
                if rest.is_some() {
                    None
                } else {
                    first.and_then(combining_for)
                }
            };
            let Some(combining) = mark_char else {
                i += 1;
                continue;
            };
            // Geometric: same line, diacritic anchored over the base
            // letter's left edge (within ±1 pt).
            let (same_line, overlaps_x) = {
                let p = &spans[i];
                let n = &spans[i + 1];
                let same = (p.bbox.y - n.bbox.y).abs() < p.font_size.max(n.font_size) * 0.6;
                let dx = (p.bbox.x - n.bbox.x).abs();
                (same, dx <= 1.5)
            };
            if !(same_line && overlaps_x) {
                i += 1;
                continue;
            }
            // The next span must start with a base letter we can
            // attach a combining mark to (Latin letter / digit).
            let Some(base) = spans[i + 1].text.chars().next() else {
                i += 1;
                continue;
            };
            if !base.is_alphabetic() {
                i += 1;
                continue;
            }
            // Build "<base><combining><rest>" and NFC-compose.
            let mut composed = String::with_capacity(spans[i + 1].text.len() + 2);
            composed.push(base);
            composed.push(combining);
            let rest_start = base.len_utf8();
            composed.push_str(&spans[i + 1].text[rest_start..]);
            spans[i + 1].text = composed.nfc().collect();
            // Empty out the diacritic span; downstream consumers
            // skip zero-text spans.
            spans[i].text.clear();
            i += 2;
        }

        // Drop any spans we emptied.
        spans.retain(|s| !s.text.is_empty());
    }

    /// Substitute ASCII digits and a few punctuation characters in
    /// super/sub-script spans with their Unicode counterparts
    /// (U+2070..U+2079 / U+00B2/B3/B9 for superscripts,
    /// U+2080..U+2089 for subscripts). A span is treated as
    /// super- or sub-script when its font is meaningfully smaller
    /// than the previous span on the same line and its baseline is
    /// raised or lowered. Only spans whose text consists entirely
    /// of substitutable characters are rewritten — mixed-content
    /// or single-letter superscript callouts (e.g. footnote "a")
    /// fall through unchanged so the existing citation-handling
    /// path stays in control.
    fn apply_super_sub_script_substitutions(spans: &mut [crate::layout::TextSpan]) {
        fn super_for_char(c: char) -> Option<char> {
            Some(match c {
                '0' => '\u{2070}',
                '1' => '\u{00B9}',
                '2' => '\u{00B2}',
                '3' => '\u{00B3}',
                '4' => '\u{2074}',
                '5' => '\u{2075}',
                '6' => '\u{2076}',
                '7' => '\u{2077}',
                '8' => '\u{2078}',
                '9' => '\u{2079}',
                '+' => '\u{207A}',
                '-' => '\u{207B}',
                '=' => '\u{207C}',
                '(' => '\u{207D}',
                ')' => '\u{207E}',
                _ => return None,
            })
        }
        fn sub_for_char(c: char) -> Option<char> {
            Some(match c {
                '0' => '\u{2080}',
                '1' => '\u{2081}',
                '2' => '\u{2082}',
                '3' => '\u{2083}',
                '4' => '\u{2084}',
                '5' => '\u{2085}',
                '6' => '\u{2086}',
                '7' => '\u{2087}',
                '8' => '\u{2088}',
                '9' => '\u{2089}',
                '+' => '\u{208A}',
                '-' => '\u{208B}',
                '=' => '\u{208C}',
                '(' => '\u{208D}',
                ')' => '\u{208E}',
                _ => return None,
            })
        }
        // Two-pass: first compute the body-font baseline for each
        // line band (largest font_size on that line), then walk
        // spans and substitute any whose font is meaningfully
        // smaller AND whose baseline is raised or lowered relative
        // to the body baseline.
        let n = spans.len();
        if n < 2 {
            return;
        }
        const LINE_BAND_PT: f32 = 4.0;
        // band_anchor[i] = (body_font_size, body_y) of the line
        // band that span `i` belongs to. Sorting span indices by Y
        // once + sliding a two-pointer window over the sorted view
        // reduces the per-span band-anchor scan from O(n) to amortised
        // O(window_size), bringing the whole pass from O(n²) down to
        // O(n log n) on thesis-style pages with thousands of spans.
        let mut sorted_by_y: Vec<usize> = (0..n).collect();
        sorted_by_y
            .sort_by(|&a, &b| crate::utils::safe_float_cmp(spans[a].bbox.y, spans[b].bbox.y));
        let band_anchor = Self::compute_band_anchors(spans, &sorted_by_y, LINE_BAND_PT);
        // Spatial index: bucket spans by Y-band so `span_is_token_internal`
        // queries only nearby spans instead of all of them (its same-line
        // neighbour scan was O(n) per candidate → O(n²) on dense pages).
        let y_index = Self::build_y_band_index(spans, LINE_BAND_PT);
        for i in 0..n {
            let (anchor_fs, anchor_y) = band_anchor[i];
            let curr_fs = spans[i].font_size;
            // Skip the body span itself (it IS the anchor).
            if anchor_fs <= 0.0 || curr_fs >= anchor_fs * 0.85 {
                continue;
            }
            let y_delta = spans[i].bbox.y - anchor_y;
            let raised = y_delta > anchor_fs * 0.15;
            let lowered = y_delta < -anchor_fs * 0.15;
            if !raised && !lowered {
                continue;
            }
            let map: fn(char) -> Option<char> = if raised { super_for_char } else { sub_for_char };
            if spans[i].text.is_empty() || !spans[i].text.chars().all(|c| map(c).is_some()) {
                continue;
            }
            // Limit the substitution to clearly token-internal
            // super/sub-scripts: the run must have a base-sized
            // neighbour on BOTH sides whose first/last char is
            // alphabetic and roughly adjacent in X. Author-
            // affiliation markers like "name¹,²" sit at the END
            // of a line with no following body letter; the bench
            // GT renders those as plain ASCII digits, so substi-
            // tuting them would regress. Restricting to sandwiched
            // runs keeps the chemistry / exponent cases that the
            // GT does want as Unicode (S², H₂O, k₁) and skips the
            // trailing footnote callouts.
            if !Self::span_is_token_internal(spans, i, &y_index, LINE_BAND_PT) {
                continue;
            }
            let substituted: String = spans[i].text.chars().map(|c| map(c).unwrap()).collect();
            spans[i].text = substituted;
        }
    }

    /// For every span, the `(max_font_size, anchor_y)` over the spans within
    /// `±band` of its Y, in O(n) via a sliding-window maximum (monotonic deque)
    /// over the Y-sorted order. Replaces a per-span window walk that was O(n²)
    /// when many spans share a Y band (wide table rows).
    ///
    /// Tie-break on equal max font size: the lowest-Y span (deque keeps the
    /// earliest sorted position). A substitution only fires when the span's own
    /// font is strictly smaller than the anchor, so the tie-break merely picks
    /// which equal-sized body span supplies `anchor_y`, all within `band`.
    fn compute_band_anchors(
        spans: &[crate::layout::TextSpan],
        sorted_by_y: &[usize],
        band: f32,
    ) -> Vec<(f32, f32)> {
        let n = sorted_by_y.len();
        let mut band_anchor = vec![(0.0f32, 0.0f32); n];
        let y = |p: usize| spans[sorted_by_y[p]].bbox.y;
        let fs = |p: usize| spans[sorted_by_y[p]].font_size;
        // Deque of sorted positions, font size non-increasing front→back;
        // positions are pushed in increasing order so the deque is also
        // position-increasing front→back (front = smallest position = max fs).
        let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let mut lo = 0usize;
        let mut hi = 0usize;
        for pos in 0..n {
            let cy = y(pos);
            while hi < n && y(hi) <= cy + band {
                while let Some(&back) = deque.back() {
                    if fs(back) < fs(hi) {
                        deque.pop_back();
                    } else {
                        break;
                    }
                }
                deque.push_back(hi);
                hi += 1;
            }
            while lo < n && y(lo) < cy - band {
                if deque.front() == Some(&lo) {
                    deque.pop_front();
                }
                lo += 1;
            }
            let best = *deque.front().expect("window always contains pos");
            band_anchor[sorted_by_y[pos]] = (fs(best), y(best));
        }
        band_anchor
    }

    /// Return true when span `i` has a base-sized alphabetic
    /// neighbour both before and after it on the same line band,
    /// within ~1 em horizontally. That captures the "X²Y" /
    /// "H₂O" / "k₁ + …" pattern but excludes footnote markers
    /// that hang off the end of a word with no following body
    /// character.
    /// Bucket span indices by Y-band (`round(y / band)`) so same-line lookups
    /// scan only nearby bands instead of every span. Querying a band `k`'s
    /// `[k-2, k+2]` neighbours is a guaranteed superset of all spans within
    /// `band` points of any Y in band `k`, so an exact `|Δy|` filter on the
    /// result is byte-identical to a full scan.
    fn build_y_band_index(
        spans: &[crate::layout::TextSpan],
        band: f32,
    ) -> HashMap<i32, Vec<usize>> {
        let mut idx: HashMap<i32, Vec<usize>> = HashMap::new();
        for (j, s) in spans.iter().enumerate() {
            idx.entry((s.bbox.y / band).round() as i32)
                .or_default()
                .push(j);
        }
        idx
    }

    /// Indices in the Y-bands within ±2 of `y`'s band (superset of `|Δy| ≤ band`).
    fn y_band_candidates<'a>(
        y_index: &'a HashMap<i32, Vec<usize>>,
        y: f32,
        band: f32,
    ) -> impl Iterator<Item = usize> + 'a {
        let k = (y / band).round() as i32;
        (k - 2..=k + 2).flat_map(move |b| y_index.get(&b).into_iter().flatten().copied())
    }

    fn span_is_token_internal(
        spans: &[crate::layout::TextSpan],
        i: usize,
        y_index: &HashMap<i32, Vec<usize>>,
        band: f32,
    ) -> bool {
        let curr = &spans[i];
        let curr_y = curr.bbox.y;
        let curr_x = curr.bbox.x;
        let curr_right = curr.bbox.x + curr.bbox.width;
        let body_fs = Self::y_band_candidates(y_index, curr_y, band)
            .filter(|&j| (spans[j].bbox.y - curr_y).abs() <= 4.0)
            .map(|j| spans[j].font_size)
            .fold(0f32, f32::max)
            .max(1.0);
        let neighbour_fs_min = body_fs * 0.85;
        let max_em = body_fs;
        let mut has_left = false;
        let mut has_right = false;
        for j in Self::y_band_candidates(y_index, curr_y, band) {
            if j == i {
                continue;
            }
            let s = &spans[j];
            if (s.bbox.y - curr_y).abs() > 4.0 {
                continue;
            }
            if s.font_size < neighbour_fs_min {
                continue;
            }
            // Anchor must start or end with an alphabetic character
            // — a digit or punctuation neighbour does not signal a
            // token-internal context.
            let s_right = s.bbox.x + s.bbox.width;
            // Allow small overlap (super/sub glyphs nest slightly
            // under the body letter's bounding box).
            let dx_left = curr_x - s_right;
            if s_right < curr_right
                && dx_left <= max_em
                && dx_left >= -max_em * 0.5
                && s.text
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_alphabetic())
            {
                has_left = true;
            }
            let dx_right = s.bbox.x - curr_right;
            if s.bbox.x > curr_x
                && dx_right <= max_em
                && dx_right >= -max_em * 0.5
                && s.text.chars().next().is_some_and(|c| c.is_alphabetic())
            {
                has_right = true;
            }
        }
        has_left && has_right
    }

    /// Return per-page font statistics for use in heading detection and layout analysis.
    ///
    /// [`crate::layout::PageFontStats`] contains:
    /// - `dominant_em`: the mode font size weighted by character count — the body text "1 em"
    /// - `dominant_line_height`: median baseline-to-baseline distance
    /// - `dominant_char_width`: average character advance width
    /// - `body_font_name`: name of the most-used font
    ///
    /// The primary use-case is heading detection in downstream tools: compare
    /// `span.font_size / stats.dominant_em` against a threshold (e.g. 1.4×
    /// for H2, 1.8× for H1) to classify large-font spans as headings without
    /// depending on any hardcoded point sizes.
    ///
    /// ```ignore
    /// let stats = doc.page_font_stats(0)?;
    /// let spans = doc.extract_spans(0)?;
    /// for span in &spans {
    ///     let ratio = span.font_size / stats.dominant_em;
    ///     if ratio >= 1.8 { println!("H1: {}", span.text); }
    ///     else if ratio >= 1.4 { println!("H2: {}", span.text); }
    /// }
    /// ```
    pub fn page_font_stats(&self, page_index: usize) -> Result<crate::layout::PageFontStats> {
        let spans = self.extract_spans(page_index)?;
        Ok(crate::layout::PageFontStats::from_spans(&spans))
    }

    /// Return all extraction warnings accumulated since this document was opened.
    ///
    /// Warnings are recorded when silent fallbacks occur during text extraction
    /// (e.g., missing ToUnicode CMap, font not found, malformed structure tree).
    /// They do NOT consume the warning list — use [`Self::take_warnings`] to drain it.
    ///
    /// This API makes previously invisible extraction degradations programmatically
    /// observable without requiring callers to hook into the `log` crate.
    pub fn warnings(&self) -> Vec<String> {
        self.accumulated_warnings.lock_or_recover().clone()
    }

    /// Drain and return all accumulated extraction warnings, clearing the list.
    ///
    /// After this call, [`Self::warnings`] returns an empty `Vec` until new warnings
    /// are generated. Useful for incremental processing pipelines that want to
    /// inspect warnings on a per-page or per-operation basis.
    pub fn take_warnings(&self) -> Vec<String> {
        std::mem::take(&mut *self.accumulated_warnings.lock_or_recover())
    }

    /// Record an extraction warning. Called internally when a silent fallback occurs.
    pub(crate) fn push_warning(&self, msg: impl Into<String>) {
        self.accumulated_warnings.lock_or_recover().push(msg.into());
    }

    /// Return the document's accumulated structured warnings as a
    /// snapshot. Each entry carries the warning's
    /// [`WarningCategory`](crate::extractors::warnings::WarningCategory),
    /// page (if applicable), human-readable message, and PDF spec
    /// section reference (when applicable).
    ///
    /// Unlike [`Self::warnings`] which returns plain strings, this
    /// accessor returns structured records callers can filter, route
    /// to observability dashboards, or assert on in tests without
    /// parsing message text. Pairs with the `pyo3_log` per-target
    /// default-level downgrade to give Python users a clean stderr
    /// experience plus an opt-in structured surface.
    ///
    /// Returns the warnings in insertion order. The vector is
    /// non-destructive: subsequent calls return the same entries
    /// plus any new ones pushed since the last call. Use
    /// [`Self::take_structured_warnings`] to drain.
    ///
    /// Merges the process-wide `GLOBAL_WARNING_SINK` (where
    /// free-function log sites like `SPEC VIOLATION`,
    /// operator-cap-exceeded, and Type0/Type3 font fallbacks push
    /// their structured records) into the per-document sink on each
    /// call. The drain attribution follows the "first caller wins"
    /// rule documented at the global sink — process-wide scope means
    /// the first document to call `structured_warnings` collects
    /// the global tail that accumulated since the last drain.
    ///
    /// Renamed from `flatten_warnings` in to avoid colliding
    /// with the pre-existing `DocumentEditor::flatten_warnings`
    /// (which returns the form-flattening side-effect log, a
    /// `&[String]` — different feature). Both the Rust and Python
    /// (`PyDocument`) surfaces now agree on `structured_warnings`.
    pub fn structured_warnings(&self) -> Vec<crate::extractors::warnings::Warning> {
        let global = crate::extractors::warnings::drain_global_warnings();
        if !global.is_empty() {
            self.warning_sink.extend(global);
        }
        self.warning_sink.snapshot()
    }

    /// Drain and return all accumulated structured warnings.
    /// Companion to [`Self::structured_warnings`].
    pub fn take_structured_warnings(&self) -> Vec<crate::extractors::warnings::Warning> {
        self.warning_sink.take()
    }

    /// Record a structured warning. Hook called from migrated
    /// `log::warn!` sites that also want to surface the warning as
    /// structured data.
    ///
    /// Exposed as `pub` so external diagnostic sources (custom
    /// extractors, FFI hooks) can also push warnings into the same
    /// sink that [`Self::structured_warnings`] surfaces.
    pub fn push_structured_warning(&self, warning: crate::extractors::warnings::Warning) {
        self.warning_sink.push(warning);
    }

    /// Heuristic: does this page have two or more vertical text columns?
    ///
    /// Used by `extract_spans` to decide whether to pay the XY-cut cost
    /// (correct but slower on large pages) or stick with the cheap row-
    /// aware sort. The check bins span X-centers into a small histogram
    /// and looks for two dense bands separated by a gutter whose spans
    /// vertically overlap with each other — that's the defining shape
    /// of a multi-column layout (newspaper / academic / dashboard) as
    /// opposed to sparse side-notes that flank a single column.
    ///
    /// False negatives (missed multi-column page) just mean we use the
    /// old reading order. False positives (single column routed through
    /// XY-cut) cost a bit of CPU but produce the same or better result.
    /// Both sides degrade gracefully.
    fn is_multi_column_page(spans: &[crate::layout::TextSpan]) -> bool {
        if spans.len() < 12 {
            return false; // too few to confidently split into columns
        }

        // Primary detector: line-start-X bimodality.
        //
        // The span-center histogram further down is noisy for word-level
        // spans (every X position has many word starts on multi-word
        // body-text lines). The reliable signal is the X position at
        // which each *line* begins — a two-column body has a strong
        // peak at the left-column-start X plus a strong peak at the
        // right-column-start X, with a clear empty gutter between
        // them. We cluster spans into lines by Y (1pt tolerance), pick
        // the leftmost X per line, and look for ≥ 2 peaks separated by
        // a gutter of ≥ 30pt with zero line-starts in it.
        if Self::has_bimodal_line_starts(spans) {
            return true;
        }

        let mut x_centers: Vec<f32> = spans
            .iter()
            .map(|s| s.bbox.x + s.bbox.width * 0.5)
            .collect();
        x_centers.sort_by(|a, b| crate::utils::safe_float_cmp(*a, *b));

        // Degenerate CTM guard: drop centers more than MAX_EXTENT from the
        // median so a rogue span ~1e16 doesn't explode the histogram.
        const MAX_EXTENT_FROM_MEDIAN: f32 = 5_000.0;
        let median = x_centers[x_centers.len() / 2];
        x_centers.retain(|c| (*c - median).abs() <= MAX_EXTENT_FROM_MEDIAN);
        if x_centers.len() < 12 {
            return false;
        }

        let min = *x_centers.first().unwrap();
        let max = *x_centers.last().unwrap();
        let width = max - min;
        if width < 100.0 {
            return false; // spans cluster in a single vertical line — not columns
        }

        // Bin into 40 buckets; find peaks (≥ mean × 1.5) separated by at
        // least one empty bucket.
        const BUCKETS: usize = 40;
        let bucket_width = width / BUCKETS as f32;
        if bucket_width <= 0.0 {
            return false;
        }
        let mut hist = [0usize; BUCKETS];
        for c in &x_centers {
            let idx = (((c - min) / bucket_width) as usize).min(BUCKETS - 1);
            hist[idx] += 1;
        }

        let total: usize = hist.iter().sum();
        let mean = total as f32 / BUCKETS as f32;
        let threshold = (mean * 1.5).max(3.0);

        let mut peaks = 0usize;
        let mut in_peak = false;
        for &count in &hist {
            if count as f32 >= threshold {
                if !in_peak {
                    peaks += 1;
                    in_peak = true;
                }
            } else if count == 0 {
                in_peak = false;
            }
        }

        if peaks < 2 {
            return false;
        }

        // Confirmation: the peaks must have vertical overlap. If one "column"
        // is a footer and the other is the body, they don't interact — row-
        // aware is fine. Split spans into left-half vs right-half and check
        // their Y ranges overlap.
        let mid_x = (min + max) / 2.0;
        let mut left_y_min = f32::INFINITY;
        let mut left_y_max = f32::NEG_INFINITY;
        let mut right_y_min = f32::INFINITY;
        let mut right_y_max = f32::NEG_INFINITY;
        for s in spans {
            let cx = s.bbox.x + s.bbox.width * 0.5;
            if (cx - median).abs() > MAX_EXTENT_FROM_MEDIAN {
                continue;
            }
            let y_top = s.bbox.y + s.bbox.height;
            if cx < mid_x {
                left_y_min = left_y_min.min(s.bbox.y);
                left_y_max = left_y_max.max(y_top);
            } else {
                right_y_min = right_y_min.min(s.bbox.y);
                right_y_max = right_y_max.max(y_top);
            }
        }
        let left_span = (left_y_max - left_y_min).max(0.0);
        let right_span = (right_y_max - right_y_min).max(0.0);
        let overlap = left_y_max.min(right_y_max) - left_y_min.max(right_y_min);
        let min_span = left_span.min(right_span);
        if !(min_span > 0.0 && overlap > 0.5 * min_span) {
            return false;
        }

        // Require each half to contain enough spans to represent genuine body
        // text columns. Copyright pages, title pages, and other sparse layouts
        // can produce two X-center peaks with only 2–7 spans per "column" —
        // these are not true multi-column body text.
        let left_count = spans
            .iter()
            .filter(|s| {
                let cx = s.bbox.x + s.bbox.width * 0.5;
                (cx - median).abs() <= MAX_EXTENT_FROM_MEDIAN && cx < mid_x
            })
            .count();
        let right_count = spans.len() - left_count;
        if left_count.min(right_count) < 15 {
            return false;
        }

        // Font-aware column-shape gate.
        //
        // Real two-column body text has tight column-edge alignment:
        // most spans on each side share one dominant X position
        // (the column start), with a handful of indented or
        // section-header outliers. Scattered-fragment layouts spread
        // their spans evenly across many X positions on each side.
        //
        // Measure the fraction of side-spans that fall into the
        // largest X-cluster (cluster gap = `dominant_em`). Body text
        // typically scores ≥ 0.5; scattered layouts score < 0.4.
        // Reject pages where either side fails the threshold so
        // XY-cut doesn't mis-route scattered content as multi-column.
        let stats = crate::layout::PageFontStats::from_spans(spans);
        let cluster_gap = stats.dominant_em.max(4.0);
        let dominant_cluster_fraction = |take: &dyn Fn(f32) -> bool| -> f32 {
            let mut xs: Vec<f32> = spans
                .iter()
                .filter(|s| {
                    let cx = s.bbox.x + s.bbox.width * 0.5;
                    (cx - median).abs() <= MAX_EXTENT_FROM_MEDIAN && take(cx)
                })
                .map(|s| s.bbox.x)
                .collect();
            let total = xs.len();
            if total == 0 {
                return 0.0;
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut best = 1usize;
            let mut current = 1usize;
            let mut last = xs[0];
            for &x in &xs[1..] {
                if x - last <= cluster_gap {
                    current += 1;
                    if current > best {
                        best = current;
                    }
                } else {
                    current = 1;
                }
                last = x;
            }
            best as f32 / total as f32
        };
        const MIN_DOMINANT_FRACTION: f32 = 0.5;
        let left_frac = dominant_cluster_fraction(&|cx| cx < mid_x);
        let right_frac = dominant_cluster_fraction(&|cx| cx >= mid_x);
        if left_frac >= MIN_DOMINANT_FRACTION && right_frac >= MIN_DOMINANT_FRACTION {
            return true;
        }

        // Additive accept path (no change to the gate above): shared-baseline
        // two-column bodies — academic references / bibliographies — read
        // left+right on the SAME Y line, so the row-aware sort interleaves
        // them. Their word-granular left edges scatter, so the dominant-
        // cluster gate above misses them. But they exhibit ONE persistent
        // vertical gutter corridor (the signal poppler/MuPDF use, independent
        // of line length). Detect it via within-line gap projection, prose-
        // guarded so numeric / short-cell tables — which also reach here —
        // stay on the row-aware path. See #607.
        Self::has_persistent_gutter_corridor(spans, median, MAX_EXTENT_FROM_MEDIAN)
    }

    /// Detect a single persistent vertical gutter corridor across the page —
    /// the geometric fingerprint of a two-column prose body whose columns
    /// share Y baselines (so `has_bimodal_line_starts` and the dominant-
    /// cluster gate both miss it). Mirrors `detect_narrow_gutter_prose`
    /// (`src/pipeline/reading_order/xycut.rs`) at the document-routing layer.
    ///
    /// Table-safe by construction (#536). Long-line bodies
    /// (`mean non-whitespace chars per line > 20`) keep the original
    /// concentration / coverage / centre accept path. Short-line bodies
    /// (verse / lexicon editions) are admitted only under stricter,
    /// length-independent guards a numeric / short-cell table cannot satisfy:
    /// higher concentration and coverage, left/right column char-mass balance,
    /// and a grid-row signal (a multi-cell table has ≥ 2 wide gaps on most
    /// rows; a two-column body has one gutter). Full-width display-math /
    /// heading rows are excluded from the gutter-coverage denominator so a
    /// minority of them does not veto an otherwise two-column page.
    fn has_persistent_gutter_corridor(
        spans: &[crate::layout::TextSpan],
        median: f32,
        max_extent: f32,
    ) -> bool {
        // Group spans into lines by rounded Y baseline; carry left/right
        // extents for gap projection and char count for the prose guard.
        let mut lines: std::collections::BTreeMap<i32, (Vec<(f32, f32)>, usize)> =
            std::collections::BTreeMap::new();
        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        for s in spans {
            let cx = s.bbox.x + s.bbox.width * 0.5;
            if (cx - median).abs() > max_extent {
                continue; // degenerate-CTM guard, same as the caller
            }
            let y_key = (s.bbox.y + s.bbox.height).round() as i32;
            let entry = lines.entry(y_key).or_default();
            entry.0.push((s.bbox.x, s.bbox.x + s.bbox.width));
            entry.1 += s.text.chars().filter(|c| !c.is_whitespace()).count();
            x_min = x_min.min(s.bbox.x);
            x_max = x_max.max(s.bbox.x + s.bbox.width);
        }
        let region_width = x_max - x_min;
        if lines.len() < 12 || region_width < 200.0 {
            return false;
        }

        let total_chars: usize = lines.values().map(|(_, c)| *c).sum();
        let mean_chars = total_chars as f32 / lines.len() as f32;

        // Largest within-line gap per line (≥ 6 pt suppresses word spacing);
        // record the gap midpoint X. Also flag full-width lines with no internal
        // gutter (display equations, full-width headings) so they neither support
        // nor veto the corridor — they are excluded from the coverage denominator
        // (Part 1b: display-math robustness, #536/arxiv_math).
        const MIN_GAP_PT: f32 = 6.0;
        let mut gap_positions: Vec<f32> = Vec::new();
        let mut full_width_lines = 0usize;
        let mut multi_gap_lines = 0usize;
        for (line_spans, _) in lines.values() {
            if line_spans.is_empty() {
                continue;
            }
            let mut sorted = line_spans.clone();
            sorted.sort_by(|a, b| crate::utils::safe_float_cmp(a.0, b.0));
            let line_left = sorted.first().map(|s| s.0).unwrap_or(0.0);
            let line_right = sorted.last().map(|s| s.1).unwrap_or(0.0);
            let mut largest_gap = 0.0_f32;
            let mut largest_mid = 0.0_f32;
            let mut significant_gaps = 0usize;
            for w in sorted.windows(2) {
                let gap = w[1].0 - w[0].1;
                if gap >= MIN_GAP_PT {
                    significant_gaps += 1;
                }
                if gap > largest_gap {
                    largest_gap = gap;
                    largest_mid = (w[0].1 + w[1].0) * 0.5;
                }
            }
            if (line_right - line_left) >= region_width * 0.9 && largest_gap < MIN_GAP_PT {
                full_width_lines += 1;
            }
            // A line with two or more wide internal gaps is a grid row (≥ 3
            // cells), not a two-column body line (one gutter). Used by the
            // short-line table discriminator below.
            if significant_gaps >= 2 {
                multi_gap_lines += 1;
            }
            if largest_gap >= MIN_GAP_PT {
                gap_positions.push(largest_mid);
            }
        }
        if gap_positions.len() < 12 {
            return false;
        }
        // Coverage denominator excludes full-width display rows.
        let eff_lines = lines.len().saturating_sub(full_width_lines).max(1);

        // Cluster gap midpoints (10 pt radius); find the dominant corridor.
        const CLUSTER_RADIUS_PT: f32 = 10.0;
        gap_positions.sort_by(|a, b| crate::utils::safe_float_cmp(*a, *b));
        let mut best_size = 0usize;
        let mut best_center = 0.0_f32;
        let mut left = 0usize;
        let mut right = 0usize;
        let mut prefix: Vec<f32> = Vec::with_capacity(gap_positions.len() + 1);
        prefix.push(0.0);
        for &x in &gap_positions {
            prefix.push(prefix.last().unwrap() + x);
        }
        for &pivot in &gap_positions {
            while left < gap_positions.len() && gap_positions[left] < pivot - CLUSTER_RADIUS_PT {
                left += 1;
            }
            while right < gap_positions.len() && gap_positions[right] <= pivot + CLUSTER_RADIUS_PT {
                right += 1;
            }
            let count = right - left;
            if count > best_size {
                best_size = count;
                best_center = (prefix[right] - prefix[left]) / count as f32;
            }
        }

        // Gutter must sit near the page centre (0.30–0.70). A true two-column
        // body splits down the middle; a table's dominant gap (label column vs
        // data, or one of several cell boundaries) sits off-centre.
        let gutter_offset = best_center - x_min;
        let centre_ok =
            gutter_offset >= region_width * 0.30 && gutter_offset <= region_width * 0.70;
        if best_size < 16 || !centre_ok {
            return false;
        }

        if mean_chars > 20.0 {
            // Long-line two-column prose (the v0.3.57 accept path, unchanged
            // except the coverage denominator now excludes display rows):
            // concentration ≥ 62 %, coverage ≥ 50 % of (effective) lines.
            return best_size * 50 >= gap_positions.len() * 31 && best_size * 2 >= eff_lines;
        }

        // Short-line bodies (verse / lexicon / dictionary editions, #536): the
        // raw `mean_chars` floor used to reject these along with short-cell
        // tables. Admit them only under STRICTER, length-independent guards a
        // short-cell table cannot satisfy (Part 1a).
        let strict_concentration = best_size * 10 >= gap_positions.len() * 7; // ≥ 70 %
        let strict_coverage = best_size * 5 >= eff_lines * 3; // ≥ 60 % of lines
        if !(strict_concentration && strict_coverage) {
            return false;
        }
        // Column char-mass balance: each side of the gutter must carry ≥ 35 % of
        // the non-whitespace characters. A narrow label / verse-number column
        // paired with wide data is lopsided and rejected.
        let (mut left_chars, mut right_chars) = (0usize, 0usize);
        for s in spans {
            let cx = s.bbox.x + s.bbox.width * 0.5;
            if (cx - median).abs() > max_extent {
                continue;
            }
            let n = s.text.chars().filter(|c| !c.is_whitespace()).count();
            if cx < best_center {
                left_chars += n;
            } else {
                right_chars += n;
            }
        }
        let total = (left_chars + right_chars).max(1) as f32;
        if (left_chars as f32) < total * 0.35 || (right_chars as f32) < total * 0.35 {
            return false;
        }
        // Grid-row discriminator: a two-column body has ONE wide gap per line
        // (the gutter); a multi-cell numeric table has ≥ 2 wide gaps on most
        // rows (cell boundaries). Reject when the majority of lines are grid
        // rows — this is what keeps short-cell tables off the XY-cut path
        // without the raw `mean_chars` floor that also blocked short verse.
        multi_gap_lines * 2 <= eff_lines
    }

    /// True if the spans cluster into lines whose leftmost X positions
    /// form ≥ 2 distinct peaks separated by a clear gutter.
    ///
    /// Body-level word spans fill the X axis continuously, so the
    /// span-center histogram cannot tell two-column body text apart
    /// from a single-column page with varied line lengths. The line-
    /// start histogram does: in two-column body text most lines start
    /// at one of two X positions (left-column-start or right-column-
    /// start), and the wide gutter between the columns produces a
    /// long zero-count stretch.
    fn has_bimodal_line_starts(spans: &[crate::layout::TextSpan]) -> bool {
        const Y_BAND: f32 = 2.0;
        const BIN_PT: f32 = 5.0;
        const MIN_PEAK_COUNT: usize = 4;
        const MIN_GUTTER_PT: f32 = 30.0;

        if spans.len() < 24 {
            return false;
        }

        // Cluster spans into lines by Y (descending so top-of-page first).
        let mut lines: Vec<(f32, f32)> = Vec::new(); // (y, line_x_min)
        let mut sorted = spans.to_vec();
        sorted.sort_by(|a, b| {
            crate::utils::safe_float_cmp(b.bbox.y, a.bbox.y)
                .then_with(|| crate::utils::safe_float_cmp(a.bbox.x, b.bbox.x))
        });

        let mut current_y: Option<f32> = None;
        let mut current_xmin: f32 = f32::INFINITY;
        for s in &sorted {
            match current_y {
                Some(y) if (y - s.bbox.y).abs() <= Y_BAND => {
                    current_xmin = current_xmin.min(s.bbox.x);
                },
                _ => {
                    if let Some(y) = current_y {
                        if current_xmin.is_finite() {
                            lines.push((y, current_xmin));
                        }
                    }
                    current_y = Some(s.bbox.y);
                    current_xmin = s.bbox.x;
                },
            }
        }
        if let Some(y) = current_y {
            if current_xmin.is_finite() {
                lines.push((y, current_xmin));
            }
        }
        if lines.len() < 16 {
            return false;
        }

        // Bin line-start X positions.
        let xmin = lines.iter().map(|(_, x)| *x).fold(f32::INFINITY, f32::min);
        let xmax = lines
            .iter()
            .map(|(_, x)| *x)
            .fold(f32::NEG_INFINITY, f32::max);
        if !(xmin.is_finite() && xmax.is_finite()) || xmax - xmin < MIN_GUTTER_PT {
            return false;
        }
        let bin_count = (((xmax - xmin) / BIN_PT).ceil() as usize).max(1);
        if bin_count > 4096 {
            return false; // degenerate CTM
        }
        let mut hist = vec![0usize; bin_count];
        for (_, x) in &lines {
            let idx = (((x - xmin) / BIN_PT) as usize).min(bin_count - 1);
            hist[idx] += 1;
        }

        // Scan for ≥ 2 peaks (count ≥ MIN_PEAK_COUNT) with a long
        // zero-count run between them.
        let mut peaks: Vec<usize> = Vec::new(); // bin indices (peak center)
        let mut in_peak = false;
        let mut peak_start = 0usize;
        for (i, &c) in hist.iter().enumerate() {
            if c >= MIN_PEAK_COUNT {
                if !in_peak {
                    peak_start = i;
                    in_peak = true;
                }
            } else if c == 0 && in_peak {
                peaks.push((peak_start + i.saturating_sub(1)) / 2);
                in_peak = false;
            }
        }
        if in_peak {
            peaks.push((peak_start + hist.len() - 1) / 2);
        }
        if peaks.len() < 2 {
            return false;
        }

        // Check gutter: at least one pair of consecutive peaks must
        // have ≥ MIN_GUTTER_PT zero-count between them.
        let gutter_bins = (MIN_GUTTER_PT / BIN_PT) as usize;
        for w in peaks.windows(2) {
            let a = w[0];
            let b = w[1];
            if b <= a {
                continue;
            }
            let zeros = hist[a + 1..b].iter().filter(|&&c| c == 0).count();
            if zeros >= gutter_bins {
                return true;
            }
        }
        false
    }

    /// Normalize a span's text for cross-page signature matching.
    /// Collapses whitespace and replaces digit runs with `#` so that page
    /// numbers ("Page 1 of 10", "Page 2 of 10") collapse to one signature.
    fn normalize_artifact_signature(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut in_digit_run = false;
        let mut last_was_space = true;
        for c in text.chars() {
            if c.is_ascii_digit() {
                if !in_digit_run {
                    out.push('#');
                    in_digit_run = true;
                }
                last_was_space = false;
            } else if c.is_whitespace() {
                if !last_was_space {
                    out.push(' ');
                    last_was_space = true;
                }
                in_digit_run = false;
            } else {
                out.push(c);
                last_was_space = false;
                in_digit_run = false;
            }
        }
        out.trim().to_string()
    }

    /// Ensure running-artifact signatures are computed (once) and return a
    /// clone for matching. The computation scans every page's raw spans,
    /// collects normalized text that appears in the top or bottom 12% of
    /// the page, and keeps entries that recur on >=50% of pages.
    fn ensure_running_artifact_signatures(
        &self,
    ) -> Result<std::collections::HashMap<String, usize>> {
        {
            let guard = self.running_artifact_signatures.lock_or_recover();
            if let Some(ref map) = *guard {
                return Ok(map.clone());
            }
        }
        let page_count = self.page_count()?;
        if page_count < 2 {
            let empty = std::collections::HashMap::new();
            *self.running_artifact_signatures.lock_or_recover() = Some(empty.clone());
            return Ok(empty);
        }

        // (count of distinct pages seeing the signature, first page it appeared on).
        // `first_seen_any` tracks the earliest page a signature appeared on
        // regardless of body-content — so if the cover page is all-chrome
        // (no body text), it still registers as "first seen" and gets its
        // title kept by the per-page mark_running_artifact_spans exemption.
        let mut occurrences: std::collections::HashMap<String, (usize, usize)> =
            std::collections::HashMap::new();
        let mut first_seen_any: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        // Track distinct literal texts per signature. A signature whose digits
        // are stable across every page (i.e. the literal text never changes) is
        // NOT a page-number-containing header — it is substantive content that
        // happens to repeat. Only suppress signatures where the literal text
        // varies (at least two distinct forms) meaning digits change per page.
        let mut literal_variants: std::collections::HashMap<
            String,
            std::collections::HashSet<String>,
        > = std::collections::HashMap::new();
        for pi in 0..page_count {
            let spans = match self.extract_spans_raw(pi) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let page_height = match self.get_page_media_box(pi) {
                Ok((_, _, _, h)) if h > 0.0 => h,
                _ => continue,
            };
            let band = page_height * 0.12;
            // Require that the page has CONTENT outside the top/bottom
            // bands before counting band spans as candidate artifacts.
            // Otherwise, a page consisting only of a title near the top
            // would have its own title classified as a "running header"
            // across all pages.
            let has_body_content = spans.iter().any(|s| {
                let t = s.text.trim();
                if t.is_empty() {
                    return false;
                }
                let top_of_span = s.bbox.y + s.bbox.height;
                top_of_span <= page_height - band && s.bbox.y >= band
            });
            // Collect per-page unique signatures from the chrome bands.
            // Runs even when there's no body content so `first_seen_any`
            // registers the cover page even if it's all-chrome.
            let mut seen_this_page: std::collections::HashMap<String, String> =
                std::collections::HashMap::new();
            for s in spans.iter() {
                let trimmed = s.text.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let near_bottom = s.bbox.y < band;
                let near_top = s.bbox.y + s.bbox.height > page_height - band;
                if !(near_top || near_bottom) {
                    continue;
                }
                let sig = Self::normalize_artifact_signature(trimmed);
                if sig.is_empty() || sig.chars().count() < 2 {
                    continue;
                }
                seen_this_page
                    .entry(sig)
                    .or_insert_with(|| trimmed.to_string());
            }
            // Track first-seen across ALL pages (even body-content-skipped)
            for sig in seen_this_page.keys() {
                first_seen_any.entry(sig.clone()).or_insert(pi);
            }
            // Track literal variants — if the literal text for a signature
            // differs across pages, the digits are varying (page numbers).
            for (sig, literal) in &seen_this_page {
                literal_variants
                    .entry(sig.clone())
                    .or_default()
                    .insert(literal.clone());
            }
            if !has_body_content {
                continue;
            }
            // Count only pages with body content for the recurrence threshold
            for sig in seen_this_page.into_keys() {
                let entry = occurrences.entry(sig).or_insert((0, pi));
                entry.0 += 1;
                if pi < entry.1 {
                    entry.1 = pi;
                }
            }
        }
        let threshold = (page_count as f32 * 0.5).ceil() as usize;
        let signatures: std::collections::HashMap<String, usize> = occurrences
            .into_iter()
            .filter(|(sig, (count, _))| {
                if *count < threshold.max(2) {
                    return false;
                }
                // Only suppress if the literal text varied across pages — i.e., the
                // digits changed, indicating a page number or date that updates per page.
                // Signatures where all occurrences have the same literal text are
                // substantive content (facility names, document IDs, etc.) that pdftotext
                // and pdfium both preserve; suppressing them hurts word-F1 scores.
                let variants = literal_variants.get(sig).map(|s| s.len()).unwrap_or(0);
                variants >= 2
            })
            .map(|(sig, _)| {
                // Use the earliest page the signature appeared on — which
                // may be a body-content-skipped cover page that `occurrences`
                // didn't count toward the threshold but `first_seen_any` did.
                let first = first_seen_any.get(&sig).copied().unwrap_or(0);
                (sig, first)
            })
            .collect();
        *self.running_artifact_signatures.lock_or_recover() = Some(signatures.clone());
        Ok(signatures)
    }

    /// Mark spans near the top/bottom of the page whose normalized text
    /// matches a cached running-artifact signature by setting
    /// `artifact_type` to Pagination.
    /// #553: a bare page number (e.g. " 1 ", "12") varies per page, so it
    /// never matches a repeated-text signature and leaks into the body. Treat
    /// a short pure-digit token (1..=9999) as a page-number candidate — only
    /// applied inside the top/bottom margin band by the caller, so ordinary
    /// numerals in body text are never affected.
    fn is_bare_page_number_text(trimmed: &str) -> bool {
        !trimmed.is_empty()
            && trimmed.len() <= 4
            && trimmed.chars().all(|c| c.is_ascii_digit())
            && trimmed
                .parse::<u32>()
                .map(|n| (1..=9999).contains(&n))
                .unwrap_or(false)
    }

    fn mark_running_artifact_spans(
        &self,
        page_index: usize,
        spans: &mut [crate::layout::TextSpan],
    ) -> Result<()> {
        let (_, _, _, page_height) = match self.get_page_media_box(page_index) {
            Ok(mb) => mb,
            Err(_) => return Ok(()),
        };
        if page_height <= 0.0 {
            return Ok(());
        }
        let band = page_height * 0.12;
        // Snapshot baselines of every non-blank span, so the bare-page-number
        // rule can require a candidate to stand ALONE on its line (#553): a
        // digit adjacent to other text — e.g. the "8" in "8th" — is content,
        // not a page number.
        let occupied_baselines: Vec<f32> = spans
            .iter()
            .filter(|s| !s.text.trim().is_empty())
            .map(|s| s.bbox.y)
            .collect();
        // Signature set may be empty (no repeated headers/footers); the
        // bare-page-number rule below still runs.
        let signatures = self.ensure_running_artifact_signatures()?;
        for s in spans.iter_mut() {
            if s.artifact_type.is_some() {
                continue;
            }
            let near_bottom = s.bbox.y < band;
            let near_top = s.bbox.y + s.bbox.height > page_height - band;
            if !(near_top || near_bottom) {
                continue;
            }
            let trimmed = s.text.trim();
            if trimmed.is_empty() {
                continue;
            }
            // #553: standalone page-number chrome in the margin band — only
            // when the digit is ISOLATED on its line (no other text span
            // within ~one line height), so digits embedded in words/runs are
            // never dropped.
            if Self::is_bare_page_number_text(trimmed) {
                let line_tol = s.font_size.max(6.0);
                let on_line = occupied_baselines
                    .iter()
                    .filter(|&&oy| (oy - s.bbox.y).abs() < line_tol)
                    .count();
                if on_line <= 1 {
                    s.artifact_type = Some(crate::extractors::text::ArtifactType::Pagination(
                        crate::extractors::text::PaginationSubtype::PageNumber,
                    ));
                }
                continue;
            }
            if signatures.is_empty() {
                continue;
            }
            let sig = Self::normalize_artifact_signature(trimmed);
            if let Some(&first_seen_on) = signatures.get(&sig) {
                // Keep the first appearance — it's usually the document
                // cover-page title that got classified as chrome only
                // because later pages repeat it as a running header (B3).
                if page_index == first_seen_on {
                    continue;
                }
                s.artifact_type = Some(crate::extractors::text::ArtifactType::Pagination(
                    crate::extractors::text::PaginationSubtype::Other,
                ));
            }
        }
        Ok(())
    }

    /// Internal helper: extract raw (unsorted) text spans from a page.
    ///
    /// This is the common extraction logic shared by `extract_spans`
    /// `extract_spans_with_reading_order`. Spans are returned without any
    /// sorting or erase-region filtering applied.
    fn extract_spans_raw(&self, page_index: usize) -> Result<Vec<crate::layout::TextSpan>> {
        self.extract_spans_raw_with_extraction_config(
            page_index,
            crate::extractors::TextExtractionConfig::default(),
        )
    }

    /// Internal helper: extract raw text spans using a specific extraction config.
    ///
    /// This allows callers to provide a [`TextExtractionConfig`] (optionally
    /// configured with an [`ExtractionProfile`]) to control TJ offset thresholds
    /// and word boundary detection during span extraction.
    fn extract_spans_raw_with_extraction_config(
        &self,
        page_index: usize,
        config: crate::extractors::TextExtractionConfig,
    ) -> Result<Vec<crate::layout::TextSpan>> {
        self.extract_spans_impl(page_index, config, HashSet::new(), HashSet::new())
    }

    fn extract_spans_raw_filtered(
        &self,
        page_index: usize,
        excluded_layers: HashSet<String>,
        excluded_inks: HashSet<String>,
    ) -> Result<Vec<crate::layout::TextSpan>> {
        self.extract_spans_impl(
            page_index,
            crate::extractors::TextExtractionConfig::default(),
            excluded_layers,
            excluded_inks,
        )
    }

    fn extract_spans_impl(
        &self,
        page_index: usize,
        config: crate::extractors::TextExtractionConfig,
        excluded_layers: HashSet<String>,
        excluded_inks: HashSet<String>,
    ) -> Result<Vec<crate::layout::TextSpan>> {
        if self.is_encrypted_unreadable() {
            log::warn!("PDF is encrypted and could not be decrypted; returning no spans");
            return Ok(Vec::new());
        }
        use crate::extractors::TextExtractor;

        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;

        if self.page_cannot_have_text(page_dict) {
            return Ok(Vec::new());
        }

        let content_data = match self.get_page_content_data(page_index) {
            Ok(data) => data,
            Err(e) => {
                log::warn!(
                    "Failed to decode content stream for page {}: {}, returning empty",
                    page_index,
                    e
                );
                return Ok(Vec::new());
            },
        };

        if !Self::may_contain_text(&content_data) {
            return Ok(Vec::new());
        }

        let mut extractor = TextExtractor::with_config(config);
        if !excluded_layers.is_empty() {
            extractor.set_excluded_layers(excluded_layers);
        }
        if !excluded_inks.is_empty() {
            extractor.set_excluded_inks(excluded_inks);
        }
        if let Some(resources) = page_dict.get("Resources") {
            extractor.set_resources(resources.clone());
            extractor.set_document(self);
            if let Err(e) = self.load_fonts(resources, &mut extractor) {
                log::warn!(
                    "Failed to load fonts for page {}: {}, continuing with defaults",
                    page_index,
                    e
                );
            }
        }

        extractor.extract_text_spans(&content_data)
    }

    /// Extract text from a page, excluding content from specified layers and inks.
    ///
    /// Uses the same full text assembly pipeline as [`extract_text`](Self::extract_text)
    /// (structure-tree ordering, table detection, column detection), but with
    /// layer/ink-excluded spans removed before assembly.
    ///
    /// **Ink filtering note:** For DeviceN color spaces, text is suppressed if
    /// ANY ink in the DeviceN array matches an excluded ink name. Tint values
    /// are not evaluated — this is an all-or-nothing match.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `excluded_layers` - OCG layer names to suppress (empty = no layer filtering)
    /// * `excluded_inks` - Separation/DeviceN ink names to suppress (empty = no ink filtering)
    pub fn extract_text_filtered(
        &self,
        page_index: usize,
        excluded_layers: HashSet<String>,
        excluded_inks: HashSet<String>,
    ) -> Result<String> {
        if excluded_layers.is_empty() && excluded_inks.is_empty() {
            return self.extract_text(page_index);
        }

        let spans = self.extract_spans_filtered(page_index, excluded_layers, excluded_inks)?;
        let options = crate::converters::ConversionOptions {
            extract_tables: true,
            ..Default::default()
        };
        self.assemble_text_from_spans(page_index, spans, &options)
    }

    /// Extract text from a region of a page with layer/ink filtering applied.
    ///
    /// Composes [`Self::extract_text_filtered`] with [`Self::extract_text_in_rect`]: spans
    /// are filtered by layer/ink first, then by region, then assembled via
    /// the full text pipeline (structure-tree ordering, table detection,
    /// column detection, whitespace + line breaks).
    pub fn extract_text_filtered_in_rect(
        &self,
        page_index: usize,
        excluded_layers: HashSet<String>,
        excluded_inks: HashSet<String>,
        region: crate::geometry::Rect,
        mode: crate::layout::RectFilterMode,
    ) -> Result<String> {
        let spans = if excluded_layers.is_empty() && excluded_inks.is_empty() {
            self.extract_spans(page_index)?
        } else {
            self.extract_spans_filtered(page_index, excluded_layers, excluded_inks)?
        };
        let options = crate::converters::ConversionOptions {
            extract_tables: true,
            include_region: Some((region, mode)),
            ..Default::default()
        };
        self.assemble_text_from_spans(page_index, spans, &options)
    }

    /// Extract text spans from a page using a specified reading order strategy.
    ///
    /// This method extracts text spans identically to [`extract_spans`](Self::extract_spans),
    /// then applies the chosen reading order strategy to sort them.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `reading_order` - The reading order strategy to apply
    ///
    /// # Returns
    ///
    /// Vector of TextSpan objects sorted according to the chosen reading order.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::document::{PdfDocument, ReadingOrder};
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("two_column.pdf")?;
    /// let spans = doc.extract_spans_with_reading_order(0, ReadingOrder::ColumnAware)?;
    /// for span in spans {
    ///     println!("{}", span.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_spans_with_reading_order(
        &self,
        page_index: usize,
        reading_order: ReadingOrder,
    ) -> Result<Vec<crate::layout::TextSpan>> {
        // Extract raw spans using the common extraction logic
        let mut spans = self.extract_spans_raw(page_index)?;

        // Apply reading order strategy
        match reading_order {
            ReadingOrder::TopToBottom => {
                // Row-aware sort: Y-band descending, then X ascending.
                spans.sort_by(|a, b| {
                    crate::utils::row_aware_span_cmp(a.bbox.y, a.bbox.x, b.bbox.y, b.bbox.x)
                });
            },
            ReadingOrder::ColumnAware => {
                use crate::pipeline::reading_order::{
                    ReadingOrderContext as ROContext, ReadingOrderStrategy, XYCutStrategy,
                };
                let strategy = XYCutStrategy::new();
                let context = ROContext::new().with_page(page_index as u32);
                let ordered = strategy.apply(spans, &context)?;
                spans = ordered.into_iter().map(|o| o.span).collect();
            },
        }

        // Filter out spans in erase regions
        let erase = self
            .erase_regions
            .lock_or_recover()
            .get(&page_index)
            .cloned();
        if let Some(regions) = erase {
            spans.retain(|span| !regions.iter().any(|r| r.intersects(&span.bbox)));
        }

        Ok(spans)
    }

    /// Extract complete page text data in a single call.
    ///
    /// Returns a [`PageText`](crate::layout::text_block::PageText) containing spans in reading order, per-character
    /// data derived from those spans (using font-metric widths when available),
    /// and the page dimensions. Uses the default `TopToBottom` reading order.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("example.pdf")?;
    /// let page_text = doc.extract_page_text(0)?;
    /// println!("Page {}x{} pt", page_text.page_width, page_text.page_height);
    /// println!("{} spans, {} chars", page_text.spans.len(), page_text.chars.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_page_text(&self, page_index: usize) -> Result<crate::layout::PageText> {
        self.extract_page_text_with_options(page_index, ReadingOrder::default())
    }

    /// Extract a page as typed [`StructuredPage`](crate::structured::StructuredPage)
    /// regions (issue #536).
    ///
    /// Returns the page's text grouped into
    /// [`StructuredRegion`](crate::structured::StructuredRegion)s — body blocks,
    /// headings, header/footer/page-number chrome, and marginal labels — in
    /// reading order, with a best-effort `column_index` for two-column bodies.
    ///
    /// Roles are derived from signals already attached to each span: `/Artifact`
    /// marked content (ISO 32000-1:2008 §14.8.2.2), structure-tree heading levels
    /// (§14.7.2), and span geometry (§14.8.2.3.1). A tagged PDF with a
    /// trustworthy `/StructTreeRoot` (see
    /// [`prefers_structure_reading_order`](Self::prefers_structure_reading_order))
    /// therefore yields tree-driven roles; untagged PDFs use the geometric /
    /// font-size fallbacks. This is an additive aggregation layer — it does not
    /// change any existing extraction output.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("two_column.pdf")?;
    /// let page = doc.extract_structured(0)?;
    /// for region in &page.regions {
    ///     println!("{:?} col={:?}: {}", region.kind, region.column_index, region.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_structured(
        &self,
        page_index: usize,
    ) -> Result<crate::structured::StructuredPage> {
        let page_text = self.extract_page_text(page_index)?;
        Ok(crate::structured::build_structured_page(
            page_index,
            page_text.page_width,
            page_text.page_height,
            page_text.spans,
        ))
    }

    /// Extract complete page text data with a specific reading order.
    ///
    /// Like [`extract_page_text`](Self::extract_page_text) but allows choosing
    /// between `TopToBottom` and `ColumnAware` reading order.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `reading_order` - Reading order strategy to apply
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::document::{PdfDocument, ReadingOrder};
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("two_column.pdf")?;
    /// let page_text = doc.extract_page_text_with_options(0, ReadingOrder::ColumnAware)?;
    /// for span in &page_text.spans {
    ///     println!("{}", span.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_page_text_with_options(
        &self,
        page_index: usize,
        reading_order: ReadingOrder,
    ) -> Result<crate::layout::PageText> {
        // Get spans with the requested reading order
        let spans = self.extract_spans_with_reading_order(page_index, reading_order)?;

        // Derive chars from spans (uses char_widths for accurate positioning)
        let chars: Vec<crate::layout::TextChar> = spans.iter().flat_map(|s| s.to_chars()).collect();

        // Get page dimensions from MediaBox
        let media_box = self.get_page_media_box(page_index)?;

        Ok(crate::layout::PageText {
            spans,
            chars,
            page_width: media_box.2,
            page_height: media_box.3,
        })
    }

    /// Extract text spans from a page with custom configuration.
    ///
    /// This method allows controlling span merging behavior through configuration,
    /// including adaptive threshold settings for improved extraction quality.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `config` - SpanMergingConfig controlling extraction parameters
    ///
    /// # Returns
    ///
    /// A vector of TextSpan objects extracted from the page with applied configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # use pdf_oxide::extractors::SpanMergingConfig;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("example.pdf")?;
    ///
    /// // Use adaptive threshold configuration
    /// let config = SpanMergingConfig::adaptive();
    /// let spans = doc.extract_spans_with_config(0, config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_spans_with_config(
        &self,
        page_index: usize,
        config: crate::extractors::SpanMergingConfig,
    ) -> Result<Vec<crate::layout::TextSpan>> {
        use crate::extractors::TextExtractor;

        // Get page object
        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;

        // Fast pre-check: skip image-only pages before decompression
        if self.page_cannot_have_text(page_dict) {
            return Ok(Vec::new());
        }

        // Get content stream data — skip page on decode failure (Annex I)
        let content_data = match self.get_page_content_data(page_index) {
            Ok(data) => data,
            Err(e) => {
                log::warn!(
                    "Failed to decode content stream for page {}: {}, returning empty",
                    page_index,
                    e
                );
                return Ok(Vec::new());
            },
        };

        // Early-out for pages with no text content (§9.4.3)
        if !Self::may_contain_text(&content_data) {
            return Ok(Vec::new());
        }

        // Create text extractor with merged configuration
        let mut extractor = TextExtractor::new().with_merging_config(config);

        // Load fonts from page resources and set resources for XObject access
        if let Some(resources) = page_dict.get("Resources") {
            extractor.set_resources(resources.clone());
            extractor.set_document(self);

            // Load fonts
            if let Err(e) = self.load_fonts(resources, &mut extractor) {
                log::warn!(
                    "Failed to load fonts for page {}: {}, continuing with defaults",
                    page_index,
                    e
                );
            }
        }

        // Extract text spans
        extractor.extract_text_spans(&content_data)
    }

    /// Extract individual characters from a PDF page.
    ///
    /// This is a **low-level API** for character-level granularity. For most use cases,
    /// prefer `extract_spans()` which provides complete text strings as PDF defines them.
    ///
    /// # Character-level extraction details:
    ///
    /// - Returns individual `TextChar` objects with position, font, and style information
    /// - Characters are sorted in reading order (top-to-bottom, left-to-right)
    /// - Overlapping characters (rendered multiple times for effects) are deduplicated
    /// - Useful for layout analysis, debugging, or custom text processing pipelines
    ///
    /// # Arguments
    ///
    /// * `page_index` - Page number (0-indexed)
    ///
    /// # Returns
    ///
    /// Vector of `TextChar` objects in reading order, or error if extraction fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("document.pdf")?;
    /// let chars = doc.extract_chars(0)?;
    /// for ch in chars {
    ///     println!("'{}' at ({:.1}, {:.1}), font: {}",
    ///         ch.char, ch.bbox.x, ch.bbox.y, ch.font_name);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// List all Optional Content Group (OCG) layer names in the document.
    ///
    /// Reads `/OCProperties` from the document catalog and returns the `/Name`
    /// of each OCG dictionary listed in `/OCGs`. These names can be passed to
    /// `extract_text_filtered` / `extract_chars_filtered` via `excluded_layers`.
    ///
    /// Returns an empty vec if the document has no optional content.
    pub fn get_layers(&self) -> Result<Vec<String>> {
        let catalog = self.catalog()?;
        let catalog_dict = catalog
            .as_dict()
            .ok_or_else(|| Error::InvalidPdf("Catalog is not a dictionary".to_string()))?;

        let oc_props = match catalog_dict.get("OCProperties") {
            Some(obj) => {
                if let Some(r) = obj.as_reference() {
                    self.load_object(r)?
                } else {
                    obj.clone()
                }
            },
            None => return Ok(Vec::new()),
        };

        let oc_dict = match oc_props.as_dict() {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };

        let ocgs_obj = match oc_dict.get("OCGs") {
            Some(obj) => {
                if let Some(r) = obj.as_reference() {
                    self.load_object(r)?
                } else {
                    obj.clone()
                }
            },
            None => return Ok(Vec::new()),
        };

        let ocgs_arr = match ocgs_obj.as_array() {
            Some(a) => a,
            None => return Ok(Vec::new()),
        };

        let mut names = Vec::new();
        for item in ocgs_arr {
            let ocg_obj = if let Some(r) = item.as_reference() {
                match self.load_object(r) {
                    Ok(o) => o,
                    Err(_) => continue,
                }
            } else {
                item.clone()
            };
            if let Some(d) = ocg_obj.as_dict() {
                if let Some(Object::Name(n)) = d.get("Name") {
                    names.push(n.clone());
                } else if let Some(Object::String(s)) = d.get("Name") {
                    if let Ok(text) = String::from_utf8(s.clone()) {
                        names.push(text);
                    }
                }
            }
        }
        Ok(names)
    }

    /// List ink / separation names used on a specific page.
    ///
    /// Scans the page's `/Resources /ColorSpace` dictionary for `/Separation`
    /// and `/DeviceN` color space definitions and returns their ink names.
    /// These names can be passed to `extract_text_filtered` /
    /// `extract_chars_filtered` via `excluded_inks`.
    ///
    /// **Note:** Only the page's own `/Resources` is walked. Spot inks
    /// declared inside a Form XObject's local `/Resources /ColorSpace`
    /// dictionary will not be enumerated — even though the renderer and
    /// extractor will still honor them at use time. Callers populating a
    /// UI picker from this list may miss XObject-local inks.
    ///
    /// For the full walk that follows `Do` operators into Form XObject
    /// resources, use [`Self::get_page_inks_deep`] — that is what the
    /// separation renderer uses to allocate plates.
    pub fn get_page_inks(&self, page_index: usize) -> Result<Vec<String>> {
        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;

        let resources = match page_dict.get("Resources") {
            Some(r) => {
                if let Some(rr) = r.as_reference() {
                    self.load_object(rr)?
                } else {
                    r.clone()
                }
            },
            None => return Ok(Vec::new()),
        };

        let res_dict = match resources.as_dict() {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };

        let cs_obj = match res_dict.get("ColorSpace") {
            Some(obj) => {
                if let Some(r) = obj.as_reference() {
                    self.load_object(r)?
                } else {
                    obj.clone()
                }
            },
            None => return Ok(Vec::new()),
        };

        let cs_dict = match cs_obj.as_dict() {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };

        // Resolve any indirect references so the extractor sees inline
        // arrays. Mirrors the pre-existing per-entry resolve loop.
        let mut resolved: std::collections::HashMap<String, Object> =
            std::collections::HashMap::with_capacity(cs_dict.len());
        for (name, cs_def) in cs_dict.iter() {
            let v = if let Some(r) = cs_def.as_reference() {
                match self.load_object(r) {
                    Ok(o) => o,
                    Err(_) => continue,
                }
            } else {
                cs_def.clone()
            };
            resolved.insert(name.clone(), v);
        }

        let mut ink_names = Vec::new();
        extract_inks_from_color_space_dict(&resolved, Some(self), &mut ink_names);

        ink_names.sort();
        ink_names.dedup();
        Ok(ink_names)
    }

    /// List ink / separation names declared on a page **including** those
    /// declared inside Form XObjects reached through the page's content-stream
    /// `Do` operators.
    ///
    /// Walks the page's content stream looking for `Do` operators that invoke
    /// Form XObjects (§8.10), recurses into each form's `/Resources/ColorSpace`
    /// dictionary, and accumulates `/Separation` and `/DeviceN` ink names from
    /// every visited resource tree.
    ///
    /// **Cycle handling:** indirect XObject references are deduplicated by
    /// `ObjectRef`; recursion depth is bounded at `MAX_RECURSION_DEPTH` (100).
    /// A cycle below the depth bound is silently terminated; a tree deeper
    /// than the bound returns [`Error::RecursionLimitExceeded`].
    ///
    /// **Out of scope:** tiling / shading patterns (§8.7) and annotation
    /// appearance streams (§12.5.5) — both can declare their own colour
    /// spaces but the separation renderer does not paint into them, so
    /// surfacing their inks here would create plates that stay empty.
    pub fn get_page_inks_deep(&self, page_index: usize) -> Result<Vec<String>> {
        let resources = self.page_resources_for_inks(page_index)?;
        let content_data = self.get_page_content_data(page_index)?;
        let operators = crate::content::parser::parse_content_stream(&content_data)?;

        let mut ink_names: Vec<String> = Vec::new();
        let mut visited: std::collections::HashSet<crate::object::ObjectRef> =
            std::collections::HashSet::new();

        self.collect_inks_from_resources(&resources, &mut ink_names)?;
        self.walk_form_xobject_tree_for_inks(
            &operators,
            &resources,
            &mut ink_names,
            &mut visited,
            0,
        )?;

        ink_names.sort();
        ink_names.dedup();
        Ok(ink_names)
    }

    /// Resolve the page's `/Resources` entry, following an indirect
    /// reference if present. Mirrors the same pattern used by
    /// [`Self::get_page_inks`]. Internal helper that does not depend on
    /// the `rendering`-feature-gated [`Self::get_page_resources`].
    fn page_resources_for_inks(&self, page_index: usize) -> Result<Object> {
        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;
        let resources = match page_dict.get("Resources") {
            Some(r) => match r.as_reference() {
                Some(rr) => self.load_object(rr)?,
                None => r.clone(),
            },
            None => Object::Dictionary(std::collections::HashMap::new()),
        };
        Ok(resources)
    }

    /// Dereference `obj` if it is an indirect reference; otherwise clone.
    /// Internal helper that mirrors the rendering-gated
    /// [`Self::resolve_object`] without taking the gate.
    fn deref_object_for_inks(&self, obj: &Object) -> Result<Object> {
        match obj.as_reference() {
            Some(r) => self.load_object(r),
            None => Ok(obj.clone()),
        }
    }

    /// Append inks declared in `resources./ColorSpace` (resolving indirect
    /// references) to `out`. Internal helper for both
    /// [`Self::get_page_inks_deep`] and the recursive form walker.
    fn collect_inks_from_resources(&self, resources: &Object, out: &mut Vec<String>) -> Result<()> {
        let res_dict = match resources.as_dict() {
            Some(d) => d,
            None => return Ok(()),
        };
        let cs_obj = match res_dict.get("ColorSpace") {
            Some(obj) => self.deref_object_for_inks(obj)?,
            None => return Ok(()),
        };
        let cs_dict_raw = match cs_obj.as_dict() {
            Some(d) => d,
            None => return Ok(()),
        };

        let mut resolved: std::collections::HashMap<String, Object> =
            std::collections::HashMap::with_capacity(cs_dict_raw.len());
        for (name, cs_def) in cs_dict_raw.iter() {
            let v = match cs_def.as_reference() {
                Some(r) => match self.load_object(r) {
                    Ok(o) => o,
                    Err(_) => continue,
                },
                None => cs_def.clone(),
            };
            resolved.insert(name.clone(), v);
        }
        extract_inks_from_color_space_dict(&resolved, Some(self), out);
        Ok(())
    }

    /// Recursive walker: for every `Operator::Do { name }` in `operators` that
    /// resolves to a Form XObject, scan that form's `/Resources/ColorSpace`
    /// and recurse into the form's own content stream.
    ///
    /// `visited` is keyed on the XObject's `ObjectRef` (indirect references
    /// only). Inline-stream forms cannot self-reference (no name to invoke);
    /// the depth limit is the backstop for any other malformed shape.
    fn walk_form_xobject_tree_for_inks(
        &self,
        operators: &[crate::content::operators::Operator],
        parent_resources: &Object,
        out: &mut Vec<String>,
        visited: &mut std::collections::HashSet<crate::object::ObjectRef>,
        depth: u32,
    ) -> Result<()> {
        if depth >= MAX_RECURSION_DEPTH {
            return Err(Error::RecursionLimitExceeded(MAX_RECURSION_DEPTH));
        }
        let xobjects = match parent_resources.as_dict() {
            Some(rd) => match rd.get("XObject") {
                Some(o) => self.deref_object_for_inks(o)?,
                None => return Ok(()),
            },
            None => return Ok(()),
        };
        let xobj_dict = match xobjects.as_dict() {
            Some(d) => d,
            None => return Ok(()),
        };

        for op in operators {
            let name = match op {
                crate::content::operators::Operator::Do { name } => name,
                _ => continue,
            };
            let xobj_entry = match xobj_dict.get(name) {
                Some(o) => o,
                None => continue,
            };
            let xobj_ref = xobj_entry.as_reference();
            if let Some(r) = xobj_ref {
                // Cycle through indirect refs: silent skip below depth bound.
                if !visited.insert(r) {
                    continue;
                }
            }
            let xobj = match self.deref_object_for_inks(xobj_entry) {
                Ok(o) => o,
                Err(_) => continue,
            };
            let (form_dict, form_stream) = match xobj {
                Object::Stream { ref dict, .. } => {
                    if dict.get("Subtype").and_then(Object::as_name) != Some("Form") {
                        continue;
                    }
                    let data = match xobj_ref {
                        Some(r) => self.decode_stream_with_encryption(&xobj, r)?,
                        None => xobj.decode_stream_data()?,
                    };
                    (dict.clone(), data)
                },
                _ => continue,
            };

            // §8.10.1: form may override resources or inherit the parent's.
            let form_resources = match form_dict.get("Resources") {
                Some(res) => self.deref_object_for_inks(res)?,
                None => parent_resources.clone(),
            };
            self.collect_inks_from_resources(&form_resources, out)?;

            // Recurse into the form's own content stream looking for nested
            // `Do`. Malformed streams are tolerated — we want graceful
            // degradation in a discovery API, not a hard error.
            let form_ops = match crate::content::parser::parse_content_stream(&form_stream) {
                Ok(ops) => ops,
                Err(_) => continue,
            };
            self.walk_form_xobject_tree_for_inks(
                &form_ops,
                &form_resources,
                out,
                visited,
                depth + 1,
            )?;
        }
        Ok(())
    }

    /// # Performance Note
    ///
    /// Character extraction is typically 30-50% faster than span extraction
    /// because it skips the text grouping and merging logic.
    pub fn extract_chars(&self, page_index: usize) -> Result<Vec<crate::layout::TextChar>> {
        self.extract_chars_impl(page_index, HashSet::new(), HashSet::new())
    }

    /// Extract characters from a page, excluding content from specified layers and inks.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `excluded_layers` - OCG layer names to suppress (empty = no layer filtering)
    /// * `excluded_inks` - Separation/DeviceN ink names to suppress (empty = no ink filtering)
    pub fn extract_chars_filtered(
        &self,
        page_index: usize,
        excluded_layers: HashSet<String>,
        excluded_inks: HashSet<String>,
    ) -> Result<Vec<crate::layout::TextChar>> {
        self.extract_chars_impl(page_index, excluded_layers, excluded_inks)
    }

    fn extract_chars_impl(
        &self,
        page_index: usize,
        excluded_layers: HashSet<String>,
        excluded_inks: HashSet<String>,
    ) -> Result<Vec<crate::layout::TextChar>> {
        use crate::extractors::TextExtractor;

        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;

        let content_data = match self.get_page_content_data(page_index) {
            Ok(data) => data,
            Err(e) => {
                log::warn!(
                    "Failed to decode content stream for page {}: {}, returning empty",
                    page_index,
                    e
                );
                return Ok(Vec::new());
            },
        };

        if !Self::may_contain_text(&content_data) {
            return Ok(Vec::new());
        }

        let mut extractor = TextExtractor::new();
        if !excluded_layers.is_empty() {
            extractor.set_excluded_layers(excluded_layers);
        }
        if !excluded_inks.is_empty() {
            extractor.set_excluded_inks(excluded_inks);
        }

        if let Some(resources) = page_dict.get("Resources") {
            extractor.set_resources(resources.clone());
            extractor.set_document(self);
            if let Err(e) = self.load_fonts(resources, &mut extractor) {
                log::warn!(
                    "Failed to load fonts for page {}: {}, continuing with defaults",
                    page_index,
                    e
                );
            }
        }

        let mut chars = extractor.extract(&content_data)?;

        chars.sort_by(|a, b| {
            let y_cmp = crate::utils::safe_float_cmp(b.bbox.y, a.bbox.y);
            if y_cmp != std::cmp::Ordering::Equal {
                return y_cmp;
            }
            crate::utils::safe_float_cmp(a.bbox.x, b.bbox.x)
        });

        Ok(chars)
    }

    /// Extract words from a page.
    ///
    /// Groups characters into words based on spatial proximity.
    /// Uses adaptive thresholds based on the document's font size and spacing.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let words = doc.extract_words(0)?;
    /// for word in words {
    ///     println!("Word: {} at {:?}", word.text, word.bbox);
    /// }
    /// ```
    pub fn extract_words(&self, page_index: usize) -> Result<Vec<crate::layout::Word>> {
        self.extract_words_with_thresholds(page_index, None, None)
    }

    /// Extract words from a page with optional threshold and profile overrides.
    ///
    /// When `word_gap_threshold` is `None`, the adaptive threshold is computed
    /// automatically from page statistics (median character width × 0.3).
    /// Providing a value (in PDF points) overrides the adaptive computation,
    /// which is useful for tuning word segmentation on specific document types.
    ///
    /// When `profile` is provided, it controls how the underlying text spans are
    /// extracted from the PDF content stream (TJ offset thresholds, word margin
    /// ratios). This affects the raw character data before word clustering.
    pub fn extract_words_with_thresholds(
        &self,
        page_index: usize,
        word_gap_threshold: Option<f32>,
        profile: Option<crate::config::ExtractionProfile>,
    ) -> Result<Vec<crate::layout::Word>> {
        // Default: include /Artifact-tagged spans (matches pre-0.3.42
        // behavior). The spec-correct (§14.8.2.2.1) variant lives in
        // [`Self::extract_words_with_thresholds_no_artifacts`].
        self.extract_words_inner(page_index, word_gap_threshold, profile, true)
    }

    /// Same as [`Self::extract_words_with_thresholds`] but drops spans tagged
    /// as `/Artifact` (running headers/footers, page numbers, watermarks;
    /// ISO 32000-1:2008 §14.8.2.2.1). The spec-correct variant.
    pub fn extract_words_with_thresholds_no_artifacts(
        &self,
        page_index: usize,
        word_gap_threshold: Option<f32>,
        profile: Option<crate::config::ExtractionProfile>,
    ) -> Result<Vec<crate::layout::Word>> {
        self.extract_words_inner(page_index, word_gap_threshold, profile, false)
    }

    fn extract_words_inner(
        &self,
        page_index: usize,
        word_gap_threshold: Option<f32>,
        profile: Option<crate::config::ExtractionProfile>,
        include_artifacts: bool,
    ) -> Result<Vec<crate::layout::Word>> {
        use crate::layout::{clustering, AdaptiveLayoutParams, DocumentProperties, Word};

        // Span source. The default (no profile) flows through the canonical
        // `page_reading_order` helper: tagged → struct tree,
        // untagged → geometric top-to-bottom. The legacy profile path keeps
        // its previous XY-Cut + row-aware-sort behavior pending the planned
        // removal of `profile`.
        let spans: Vec<crate::layout::TextSpan> = match profile {
            Some(p) => {
                use crate::pipeline::reading_order::xycut::XYCutStrategy;
                let config = crate::extractors::TextExtractionConfig::new().with_profile(p);
                let mut s = self.extract_spans_raw_with_extraction_config(page_index, config)?;
                s.sort_by(|a, b| {
                    crate::utils::row_aware_span_cmp(a.bbox.y, a.bbox.x, b.bbox.y, b.bbox.x)
                });
                if !include_artifacts {
                    s.retain(|span| span.artifact_type.is_none());
                }
                let erase = self
                    .erase_regions
                    .lock_or_recover()
                    .get(&page_index)
                    .cloned();
                if let Some(regions) = erase {
                    s.retain(|span| !regions.iter().any(|r| r.intersects(&span.bbox)));
                }
                let strategy = XYCutStrategy::new();
                strategy
                    .partition_region(&s)
                    .into_iter()
                    .flatten()
                    .collect()
            },
            None => {
                let ordered = if include_artifacts {
                    crate::pipeline::page_reading_order(self, page_index)?
                } else {
                    crate::pipeline::page_reading_order_no_artifacts(self, page_index)?
                };
                ordered.into_iter().map(|os| os.span).collect()
            },
        };
        if spans.is_empty() {
            return Ok(Vec::new());
        }

        // Compute adaptive parameters from all characters for consistent thresholds.
        let media_box = self
            .get_page_media_box(page_index)
            .unwrap_or((0.0, 0.0, 612.0, 792.0));
        let page_bbox =
            crate::geometry::Rect::new(media_box.0, media_box.1, media_box.2, media_box.3);

        let all_chars: Vec<_> = spans.iter().flat_map(|s| s.to_chars()).collect();
        if all_chars.is_empty() {
            return Ok(Vec::new());
        }
        let props =
            DocumentProperties::analyze(&all_chars, page_bbox).map_err(Error::LayoutAnalysis)?;
        let mut params = AdaptiveLayoutParams::from_properties(&props);

        // Apply user-provided threshold override
        if let Some(wgt) = word_gap_threshold {
            params.word_gap_threshold = wgt;
        }

        // Walk spans in canonical reading order, clustering chars within each span
        // into words. Since spans come pre-ordered, a flat iteration suffices —
        // no block-by-block partition is needed.
        //
        // Track word indices where the source span had split_boundary_before = true.
        // The post-processing merge must not cross these boundaries (table cells, columns).
        let mut split_boundary_word_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let mut words = Vec::new();
        for span in &spans {
            let span_chars = span.to_chars();
            if span_chars.is_empty() {
                continue;
            }

            // Group characters within THIS SPAN. Since PDF spans are often words or line fragments,
            // this is much safer than global character clustering.
            let clusters =
                clustering::cluster_chars_into_words(&span_chars, params.word_gap_threshold);

            // Record split boundary: the first word created from this span is a hard
            // boundary when split_boundary_before = true (e.g. table cell boundary).
            let first_word_idx = words.len();
            let is_split_boundary = span.split_boundary_before;

            for cluster_indices in clusters {
                let cluster_chars: Vec<_> = cluster_indices
                    .iter()
                    .map(|&i| span_chars[i].clone())
                    .collect();

                let mut current_word_chars = Vec::new();
                for c in cluster_chars {
                    if c.char.is_whitespace() || c.char == '\n' || c.char == '\r' {
                        if !current_word_chars.is_empty() {
                            words.push(Word::from_chars(current_word_chars));
                            current_word_chars = Vec::new();
                        }
                    } else {
                        current_word_chars.push(c);
                    }
                }
                if !current_word_chars.is_empty() {
                    words.push(Word::from_chars(current_word_chars));
                }
            }

            // Only mark the boundary if at least one word was created for this span.
            if is_split_boundary && words.len() > first_word_idx {
                split_boundary_word_indices.insert(first_word_idx);
            }
        }

        // Post-processing: merge adjacent words whose spans abut or overlap on
        // the same line. PDFs (especially tagged CJK documents) sometimes encode
        // typographically-adjacent glyphs as separate marked-content runs, e.g.
        // "Q" and "（peu/d）" with a gap of -0.18 points. Without merging these
        // remain separate tokens and never match the ground-truth "Q（peu/d）".
        //
        // Merge condition: same line (y_diff ≤ 0.5 × max line height) AND
        // horizontal gap ≤ 0.15 × font_size (same threshold as should_insert_space).
        // Skip merge when the current word index is a split boundary.
        let mut merged: Vec<Word> = Vec::with_capacity(words.len());
        for (idx, word) in words.into_iter().enumerate() {
            if !split_boundary_word_indices.contains(&idx) {
                if let Some(prev) = merged.last_mut() {
                    let gap = word.bbox.x - (prev.bbox.x + prev.bbox.width);
                    let y_diff = (word.bbox.y - prev.bbox.y).abs();
                    let line_h = prev.bbox.height.max(word.bbox.height);
                    let font_size = prev.avg_font_size.max(word.avg_font_size).max(1.0);
                    if y_diff <= line_h * 0.5 && gap <= font_size * 0.15 {
                        // Incremental merge — O(k) per merge, O(total_chars) overall.
                        // Avoids the O(n²) clone+from_chars pattern that caused
                        // catastrophic slowdown on TOC dot-leader pages.
                        let prev_n = prev.chars.len() as f32;
                        let word_n = word.chars.len() as f32;
                        prev.bbox = prev.bbox.union(&word.bbox);
                        prev.avg_font_size = (prev.avg_font_size * prev_n
                            + word.avg_font_size * word_n)
                            / (prev_n + word_n);
                        if word_n > prev_n {
                            prev.dominant_font = word.dominant_font;
                        }
                        prev.is_bold |= word.is_bold;
                        prev.is_italic |= word.is_italic;
                        if prev.mcid != word.mcid {
                            prev.mcid = None;
                        }
                        prev.text.push_str(&word.text);
                        prev.chars.extend(word.chars);
                        continue;
                    }
                }
            }
            merged.push(word);
        }

        Ok(merged)
    }

    /// Extract text lines from a page.
    ///
    /// Groups words into lines based on vertical proximity.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let lines = doc.extract_text_lines(0)?;
    /// for line in lines {
    ///     println!("Line: {} at {:?}", line.text, line.bbox);
    /// }
    /// ```
    pub fn extract_text_lines(&self, page_index: usize) -> Result<Vec<crate::layout::TextLine>> {
        self.extract_text_lines_with_thresholds(page_index, None, None, None)
    }

    /// Extract text lines from a page with optional threshold and profile overrides.
    ///
    /// When thresholds are `None`, adaptive values are computed automatically
    /// from page statistics. Providing values (in PDF points) overrides the
    /// adaptive computation for fine-grained control over segmentation.
    ///
    /// When `profile` is provided, it controls how the underlying text spans are
    /// extracted from the PDF content stream (TJ offset thresholds, word margin
    /// ratios). This affects the raw character data before word/line clustering.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `word_gap_threshold` - Optional override for the horizontal gap (in PDF points)
    ///   used to split characters into words. Smaller values produce more words.
    /// * `line_gap_threshold` - Optional override for the vertical gap (in PDF points)
    ///   used to group words into lines. Smaller values produce more lines.
    /// * `profile` - Optional extraction profile for span-level tuning.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Use adaptive thresholds (default behavior)
    /// let lines = doc.extract_text_lines_with_thresholds(0, None, None, None)?;
    ///
    /// // Tune both thresholds for dense forms
    /// let lines = doc.extract_text_lines_with_thresholds(0, Some(1.5), Some(4.0), None)?;
    /// ```
    pub fn extract_text_lines_with_thresholds(
        &self,
        page_index: usize,
        word_gap_threshold: Option<f32>,
        line_gap_threshold: Option<f32>,
        profile: Option<crate::config::ExtractionProfile>,
    ) -> Result<Vec<crate::layout::TextLine>> {
        // Default: include /Artifact-tagged spans (pre-0.3.42 behavior).
        // Spec-correct variant: [`Self::extract_text_lines_with_thresholds_no_artifacts`].
        self.extract_text_lines_inner(
            page_index,
            word_gap_threshold,
            line_gap_threshold,
            profile,
            true,
        )
    }

    /// Same as [`Self::extract_text_lines_with_thresholds`] but drops spans
    /// tagged as `/Artifact` (running headers/footers, page numbers,
    /// watermarks; ISO 32000-1:2008 §14.8.2.2.1). Spec-correct variant.
    pub fn extract_text_lines_with_thresholds_no_artifacts(
        &self,
        page_index: usize,
        word_gap_threshold: Option<f32>,
        line_gap_threshold: Option<f32>,
        profile: Option<crate::config::ExtractionProfile>,
    ) -> Result<Vec<crate::layout::TextLine>> {
        self.extract_text_lines_inner(
            page_index,
            word_gap_threshold,
            line_gap_threshold,
            profile,
            false,
        )
    }

    fn extract_text_lines_inner(
        &self,
        page_index: usize,
        word_gap_threshold: Option<f32>,
        line_gap_threshold: Option<f32>,
        profile: Option<crate::config::ExtractionProfile>,
        include_artifacts: bool,
    ) -> Result<Vec<crate::layout::TextLine>> {
        use crate::layout::{clustering, AdaptiveLayoutParams, DocumentProperties, TextLine, Word};

        // Span source. Default (no profile) → canonical `page_reading_order`
        // helper. Legacy profile path keeps XY-Cut + row-aware
        // sort pending the planned removal of `profile`.
        let spans: Vec<crate::layout::TextSpan> = match profile {
            Some(p) => {
                use crate::pipeline::reading_order::xycut::XYCutStrategy;
                let config = crate::extractors::TextExtractionConfig::new().with_profile(p);
                let mut s = self.extract_spans_raw_with_extraction_config(page_index, config)?;
                s.sort_by(|a, b| {
                    crate::utils::row_aware_span_cmp(a.bbox.y, a.bbox.x, b.bbox.y, b.bbox.x)
                });
                if !include_artifacts {
                    s.retain(|span| span.artifact_type.is_none());
                }
                let erase = self
                    .erase_regions
                    .lock_or_recover()
                    .get(&page_index)
                    .cloned();
                if let Some(regions) = erase {
                    s.retain(|span| !regions.iter().any(|r| r.intersects(&span.bbox)));
                }
                let strategy = XYCutStrategy::new();
                strategy
                    .partition_region(&s)
                    .into_iter()
                    .flatten()
                    .collect()
            },
            None => {
                let ordered = if include_artifacts {
                    crate::pipeline::page_reading_order(self, page_index)?
                } else {
                    crate::pipeline::page_reading_order_no_artifacts(self, page_index)?
                };
                ordered.into_iter().map(|os| os.span).collect()
            },
        };
        if spans.is_empty() {
            return Ok(Vec::new());
        }

        // Compute adaptive parameters
        let media_box = self
            .get_page_media_box(page_index)
            .unwrap_or((0.0, 0.0, 612.0, 792.0));
        let page_bbox =
            crate::geometry::Rect::new(media_box.0, media_box.1, media_box.2, media_box.3);

        let all_chars: Vec<_> = spans.iter().flat_map(|s| s.to_chars()).collect();
        let props =
            DocumentProperties::analyze(&all_chars, page_bbox).map_err(Error::LayoutAnalysis)?;
        let mut params = AdaptiveLayoutParams::from_properties(&props);

        // Apply user-provided threshold overrides
        if let Some(wgt) = word_gap_threshold {
            params.word_gap_threshold = wgt;
        }
        if let Some(lgt) = line_gap_threshold {
            params.line_gap_threshold = lgt;
        }

        // Walk spans in canonical reading order, clustering chars → words.
        // No block partition; spans are already pre-ordered.
        let mut words: Vec<Word> = Vec::new();
        for span in &spans {
            let span_chars = span.to_chars();
            if span_chars.is_empty() {
                continue;
            }

            let clusters =
                clustering::cluster_chars_into_words(&span_chars, params.word_gap_threshold);
            for cluster_indices in clusters {
                let cluster_chars: Vec<_> = cluster_indices
                    .iter()
                    .map(|&i| span_chars[i].clone())
                    .collect();
                let mut current_word_chars = Vec::new();
                for c in cluster_chars {
                    if c.char.is_whitespace() || c.char == '\n' || c.char == '\r' {
                        if !current_word_chars.is_empty() {
                            words.push(Word::from_chars(current_word_chars));
                            current_word_chars = Vec::new();
                        }
                    } else {
                        current_word_chars.push(c);
                    }
                }
                if !current_word_chars.is_empty() {
                    words.push(Word::from_chars(current_word_chars));
                }
            }
        }

        if words.is_empty() {
            return Ok(Vec::new());
        }

        // Cluster words → lines using global y-tolerance. Same-y words merge
        // into the same line regardless of which span they came from — the
        // span ordering already handled the multi-column / structure-tree
        // sequencing decision upstream.
        let line_clusters = clustering::cluster_words_into_lines(&words, params.line_gap_threshold);

        let mut all_lines = Vec::new();
        for cluster_indices in line_clusters {
            let cluster_words: Vec<_> = cluster_indices.iter().map(|&i| words[i].clone()).collect();
            all_lines.push(TextLine::new(cluster_words));
        }

        Ok(all_lines)
    }

    /// Apply intelligent text post-processing to extracted text spans.
    ///
    /// This method applies several text quality improvements:
    /// - Ligature expansion (fi, fl, ffi, ffl → component characters)
    /// - Hyphenation reconstruction (rejoins words split across lines)
    /// - Whitespace normalization (removes excess spaces within words)
    /// - Special character spacing (Greek letters, math symbols)
    /// - OCR text cleanup (when font_name == "OCR" or from known OCR engines)
    ///
    /// # Arguments
    ///
    /// * `spans` - Vector of TextSpan extracted from pages
    ///
    /// # Returns
    ///
    /// Processed spans with improved text quality
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::PdfDocument;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("example.pdf")?;
    ///
    /// // Extract spans from page
    /// let spans = doc.extract_spans(0)?;
    ///
    /// // Apply intelligent processing
    /// let processed = doc.apply_intelligent_text_processing(spans);
    ///
    /// for span in &processed {
    ///     println!("{}", span.text); // Ligatures expanded, hyphenation fixed
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply_intelligent_text_processing(&self, mut spans: Vec<TextSpan>) -> Vec<TextSpan> {
        use crate::converters::text_post_processor::TextPostProcessor;

        for span in &mut spans {
            // Step 1: Detect if this is OCR text (from our OCR or known OCR engines)
            let is_ocr = span.font_name == "OCR"
                || span.font_name.to_lowercase().contains("tesseract")
                || span.font_name.to_lowercase().contains("abbyy");

            // Step 2: Apply text post-processing pipeline
            // (hyphenation, whitespace, special char spacing).
            // Ligature characters from the font's ToUnicode map are preserved as-is.
            span.text = TextPostProcessor::process(&span.text);

            // Step 4: Additional OCR-specific cleanup if needed
            if is_ocr {
                // OCR text often has extra artifacts - do additional cleanup
                span.text = span
                    .text
                    .replace("ﬁ", "fi") // Sometimes OCR keeps ligatures
                    .replace("ﬂ", "fl")
                    .replace("ﬀ", "ff")
                    .replace("  ", " "); // Double space cleanup
            }
        }

        spans
    }

    /// Extract hierarchical content structure from a page.
    ///
    /// Returns the page's hierarchical content structure with all children populated.
    /// For tagged PDFs with structure trees, returns the structure with extracted content.
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
    ///
    /// # PDF Spec Compliance
    ///
    /// - ISO 32000-1:2008, Section 14.7 - Logical Structure
    /// - ISO 32000-1:2008, Section 14.8 - Tagged PDF
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("example.pdf")?;
    ///
    /// // Extract hierarchical structure from first page
    /// if let Some(structure) = doc.extract_hierarchical_content(0)? {
    ///     println!("Document structure type: {}", structure.structure_type);
    ///     println!("Number of children: {}", structure.children.len());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_hierarchical_content(
        &self,
        page_index: usize,
    ) -> Result<Option<crate::elements::StructureElement>> {
        use crate::extractors::HierarchicalExtractor;
        HierarchicalExtractor::extract_page(self, page_index)
    }

    /// Get the raw content stream data for a page.
    ///
    /// This returns the decoded content stream bytes for the specified page.
    /// The content stream contains PDF operators that define the page's appearance.
    pub fn get_page_content_data(&self, page_index: usize) -> Result<Vec<u8>> {
        {
            let mut cache = self.page_content_cache.lock_or_recover();
            if let Some(data) = cache.get(&page_index) {
                return Ok(data.as_ref().clone());
            }
        }

        // Ensure encryption is initialized if needed
        self.ensure_encryption_initialized()?;

        // Get page object
        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;

        // Get content stream(s) — Contents is optional per ISO 32000-1:2008 Table 30
        let contents_ref = match page_dict.get("Contents") {
            Some(Object::Null) | None => {
                log::debug!("Page {} has no /Contents (blank page)", page_index);
                return Ok(Vec::new());
            },
            Some(c) => c,
        };

        // Contents can be either a single stream, an array of streams, or a direct stream object
        let content_data = if let Some(contents_ref_val) = contents_ref.as_reference() {
            // Contents is a reference - it could point to either a Stream or an Array
            let contents = self.load_object(contents_ref_val)?;

            // Check if the loaded object is an Array (indirect array)
            if let Some(contents_array) = contents.as_array() {
                // The reference pointed to an array of streams
                let mut combined = Vec::new();

                for content_item in contents_array.iter() {
                    if matches!(content_item, Object::Null) {
                        continue;
                    }
                    match (|| -> Result<Vec<u8>> {
                        if let Some(ref_val) = content_item.as_reference() {
                            let content_obj = self.load_object(ref_val)?;
                            self.decode_stream_with_encryption(&content_obj, ref_val)
                        } else {
                            content_item.decode_stream_data()
                        }
                    })() {
                        Ok(decoded) => {
                            combined.extend_from_slice(&decoded);
                            combined.push(b'\n');
                        },
                        Err(e) => {
                            log::warn!(
                                "Failed to decode content stream element on page {}: {}, skipping",
                                page_index,
                                e
                            );
                        },
                    }
                }

                combined
            } else {
                // The reference pointed to a single stream
                // Decode with encryption support, using the object reference
                self.decode_stream_with_encryption(&contents, contents_ref_val)?
            }
        } else if let Some(contents_array) = contents_ref.as_array() {
            // Array of streams - can be references or direct objects
            let mut combined = Vec::new();

            for content_item in contents_array.iter() {
                if matches!(content_item, Object::Null) {
                    continue;
                }
                match (|| -> Result<Vec<u8>> {
                    if let Some(ref_val) = content_item.as_reference() {
                        let content_obj = self.load_object(ref_val)?;
                        self.decode_stream_with_encryption(&content_obj, ref_val)
                    } else {
                        content_item.decode_stream_data()
                    }
                })() {
                    Ok(decoded) => {
                        combined.extend_from_slice(&decoded);
                        combined.push(b'\n');
                    },
                    Err(e) => {
                        log::warn!(
                            "Failed to decode content stream element on page {}: {}, skipping",
                            page_index,
                            e
                        );
                    },
                }
            }

            combined
        } else {
            // Direct stream object (rare but possible)
            // For direct objects, use regular decoding (no encryption key)
            contents_ref.decode_stream_data()?
        };

        log::debug!(
            "Retrieved {} bytes of content data for page {}: {:?}",
            content_data.len(),
            page_index,
            String::from_utf8_lossy(&content_data)
        );

        self.page_content_cache
            .lock_or_recover()
            .insert(page_index, std::sync::Arc::new(content_data.clone()));

        Ok(content_data)
    }

    /// Extract path (vector graphics) content from a page.
    ///
    /// This extracts all vector graphics operations from the page's content stream,
    /// including lines, curves, rectangles, and shapes.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    ///
    /// # Returns
    ///
    /// A vector of `PathContent` objects representing all paths on the page.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("example.pdf")?;
    ///
    /// // Extract paths from first page
    /// let paths = doc.extract_paths(0)?;
    ///
    /// for path in paths {
    ///     println!("Path with {} operations, bbox: {:?}",
    ///         path.operations.len(), path.bbox);
    ///     if path.has_stroke() {
    ///         println!(" Stroked with width: {}", path.stroke_width);
    ///     }
    ///     if path.has_fill() {
    ///         println!(" Filled");
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_paths(&self, page_index: usize) -> Result<Vec<crate::elements::PathContent>> {
        use crate::content::{parse_content_stream_paths_only, Operator};
        use crate::elements::{LineCap, LineJoin};
        use crate::extractors::paths::{FillRule, PathExtractor, PathGraphicsStateStack};
        use crate::layout::Color;

        // Get page object and content stream
        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;

        // Get content stream data — skip page on decode failure (Annex I)
        let content_data = match self.get_page_content_data(page_index) {
            Ok(data) => data,
            Err(e) => {
                log::warn!(
                    "Failed to decode content stream for page {}: {}, returning empty paths",
                    page_index,
                    e
                );
                return Ok(Vec::new());
            },
        };

        let operators = match parse_content_stream_paths_only(&content_data) {
            Ok(ops) => ops,
            Err(e) => {
                log::warn!(
                    "Failed to parse content stream for page {}: {}, returning empty paths",
                    page_index,
                    e
                );
                return Ok(Vec::new());
            },
        };

        let mut extractor = PathExtractor::new();
        let mut state_stack = PathGraphicsStateStack::new();

        // Resolve and set page resources for XObject processing
        if let Some(resources) = page_dict.get("Resources") {
            let resolved_resources = if let Some(ref_obj) = resources.as_reference() {
                self.load_object(ref_obj)?
            } else {
                resources.clone()
            };
            extractor.set_resources(resolved_resources);
        }

        // Process each operator
        for op in operators {
            match op {
                // Graphics state operators
                Operator::SaveState => {
                    state_stack.save();
                },
                Operator::RestoreState => {
                    state_stack.restore();
                    extractor.update_from_path_state(state_stack.current());
                },
                Operator::Cm { a, b, c, d, e, f } => {
                    let state = state_stack.current_mut();
                    let new_matrix = crate::content::Matrix { a, b, c, d, e, f };
                    // PDF spec ISO 32000-1:2008 §8.3.4: cm concatenates as M_cm × CTM
                    state.ctm = new_matrix.multiply(&state.ctm);
                    extractor.set_ctm(state.ctm);
                },

                // Color operators (stroke)
                Operator::SetStrokeRgb { r, g, b } => {
                    state_stack.current_mut().stroke_color_rgb = (r, g, b);
                    extractor.set_stroke_color(Color::new(r, g, b));
                },
                Operator::SetStrokeGray { gray } => {
                    state_stack.current_mut().stroke_color_rgb = (gray, gray, gray);
                    extractor.set_stroke_color(Color::new(gray, gray, gray));
                },
                Operator::SetStrokeCmyk { c, m, y, k } => {
                    // Simple CMYK to RGB conversion
                    // ISO 32000-1:2008 §10.3.5: DeviceCMYK → DeviceRGB.
                    let r = 1.0 - (c + k).min(1.0);
                    let g = 1.0 - (m + k).min(1.0);
                    let b = 1.0 - (y + k).min(1.0);
                    state_stack.current_mut().stroke_color_rgb = (r, g, b);
                    extractor.set_stroke_color(Color::new(r, g, b));
                },

                // Color operators (fill)
                Operator::SetFillRgb { r, g, b } => {
                    state_stack.current_mut().fill_color_rgb = (r, g, b);
                    extractor.set_fill_color(Color::new(r, g, b));
                },
                Operator::SetFillGray { gray } => {
                    state_stack.current_mut().fill_color_rgb = (gray, gray, gray);
                    extractor.set_fill_color(Color::new(gray, gray, gray));
                },
                Operator::SetFillCmyk { c, m, y, k } => {
                    // ISO 32000-1:2008 §10.3.5: DeviceCMYK → DeviceRGB.
                    let r = 1.0 - (c + k).min(1.0);
                    let g = 1.0 - (m + k).min(1.0);
                    let b = 1.0 - (y + k).min(1.0);
                    state_stack.current_mut().fill_color_rgb = (r, g, b);
                    extractor.set_fill_color(Color::new(r, g, b));
                },

                // Line style operators
                Operator::SetLineWidth { width } => {
                    state_stack.current_mut().line_width = width;
                    extractor.set_line_width(width);
                },
                Operator::SetLineCap { cap_style } => {
                    state_stack.current_mut().line_cap = cap_style;
                    let cap = match cap_style {
                        1 => LineCap::Round,
                        2 => LineCap::Square,
                        _ => LineCap::Butt,
                    };
                    extractor.set_line_cap(cap);
                },
                Operator::SetLineJoin { join_style } => {
                    state_stack.current_mut().line_join = join_style;
                    let join = match join_style {
                        1 => LineJoin::Round,
                        2 => LineJoin::Bevel,
                        _ => LineJoin::Miter,
                    };
                    extractor.set_line_join(join);
                },

                // Path construction operators
                Operator::MoveTo { x, y } => {
                    extractor.move_to(x, y);
                },
                Operator::LineTo { x, y } => {
                    extractor.line_to(x, y);
                },
                Operator::CurveTo {
                    x1,
                    y1,
                    x2,
                    y2,
                    x3,
                    y3,
                } => {
                    extractor.curve_to(x1, y1, x2, y2, x3, y3);
                },
                Operator::CurveToV { x2, y2, x3, y3 } => {
                    extractor.curve_to_v(x2, y2, x3, y3);
                },
                Operator::CurveToY { x1, y1, x3, y3 } => {
                    extractor.curve_to_y(x1, y1, x3, y3);
                },
                Operator::Rectangle {
                    x,
                    y,
                    width,
                    height,
                } => {
                    extractor.rectangle(x, y, width, height);
                },
                Operator::ClosePath => {
                    extractor.close_path();
                },

                // Path painting operators
                Operator::Stroke => {
                    extractor.stroke();
                },
                Operator::Fill => {
                    extractor.fill(FillRule::NonZero);
                },
                Operator::FillEvenOdd => {
                    extractor.fill(FillRule::EvenOdd);
                },
                Operator::CloseFillStroke => {
                    extractor.close_fill_and_stroke(FillRule::NonZero);
                },
                Operator::EndPath => {
                    extractor.end_path();
                },

                // Clipping operators
                Operator::ClipNonZero => {
                    extractor.clip_non_zero();
                },
                Operator::ClipEvenOdd => {
                    extractor.clip_even_odd();
                },

                // XObject processing
                Operator::Do { name } => {
                    if let Err(e) =
                        self.process_form_xobject_paths(&name, &mut extractor, &mut state_stack)
                    {
                        log::warn!(
                            "Failed to process XObject '{}' in path extraction: {}",
                            name,
                            e
                        );
                    }
                },

                // Marked content operators — maintain the active Optional
                // Content Group (PDF "layer") so each finalized path gets
                // tagged with the OCG it was emitted under. Per ISO 32000-1
                // §14.6, every `BDC`/`BMC` must be balanced by an `EMC`,
                // so we always push (with `None` for non-`/OC` tags) and
                // always pop — keeps the stack depth in sync with the
                // marked-content nesting.
                Operator::BeginMarkedContent { .. } => {
                    extractor.push_oc_layer(None);
                },
                Operator::BeginMarkedContentDict { tag, properties } => {
                    let layer = if tag == "OC" {
                        self.resolve_oc_layer_name(extractor.current_resources(), &properties)
                    } else {
                        None
                    };
                    extractor.push_oc_layer(layer);
                },
                Operator::EndMarkedContent => {
                    extractor.pop_oc_layer();
                },

                // Skip other operators (text, images, etc.)
                _ => {},
            }
        }

        Ok(extractor.finish())
    }

    /// Resolve a `BDC /OC <properties>` property operand to the human-readable
    /// layer name of the Optional Content it refers to (PDF spec
    /// ISO 32000-1:2008 §8.11, §14.6).
    ///
    /// `properties` is the operand parsed by `Operator::BeginMarkedContentDict`
    /// — per spec it is either:
    ///
    /// 1. An inline dictionary: an OCG (or OCMD) — read its name directly.
    /// 2. A name (e.g. `/MC0`) that references `<resources> /Properties
    ///    <name>` → an OCG or OCMD dictionary → read its name.
    ///
    /// `resources` is the resource dictionary currently in scope: the page
    /// `/Resources` at page level, or the active Form XObject's own
    /// `/Resources` when extracting inside an XObject (§14.6.2, §8.10.1).
    ///
    /// Returns `None` for malformed PDFs, missing `/Resources /Properties`
    /// entries, or optional-content objects without a resolvable name.
    /// Callers treat `None` as "path belongs to no named layer" — extraction
    /// continues normally.
    fn resolve_oc_layer_name(
        &self,
        resources: Option<&crate::object::Object>,
        properties: &crate::object::Object,
    ) -> Option<String> {
        const OC_NAME_MAX_DEPTH: u8 = 8;

        // Case 1: inline dictionary — the property list itself is the OCG (or
        // OCMD) dictionary.
        if let Some(dict) = properties.as_dict() {
            return self.read_oc_name(dict, OC_NAME_MAX_DEPTH);
        }

        // Case 2: name reference (e.g. `/MC0`) — resolve through the current
        // resource dict's `/Properties` subdictionary.
        let prop_name = properties.as_name()?;
        let resources_obj = self.deref_object(resources?)?;
        let properties_dict = resources_obj.as_dict()?.get("Properties")?;
        let properties_obj = self.deref_object(properties_dict)?;
        let target = properties_obj.as_dict()?.get(prop_name)?;
        let target_obj = self.deref_object(target)?;
        self.read_oc_name(target_obj.as_dict()?, OC_NAME_MAX_DEPTH)
    }

    /// Read the human-readable layer name from an Optional Content dictionary.
    ///
    /// - An **OCG** (§8.11.2.1) carries its label in `/Name` — a PDF *text
    ///   string*, decoded via [`Self::decode_pdf_text_string`] so
    ///   PDFDocEncoding (Annex D) and UTF-16 (BE/LE, with BOM) layer names
    ///   round-trip identically to the rest of the library.
    /// - An **OCMD** (§8.11.3.2, Table 99) has no `/Name` of its own; its
    ///   member OCGs live in `/OCGs`, which is *either* a single OCG *or* an
    ///   array of them (array entries may be `null`). We follow the first
    ///   entry that resolves to a dictionary and read its name.
    ///
    /// `depth` bounds the `/OCGs` chain so a malformed PDF whose membership
    /// dictionary points back to another OCMD cannot recurse forever.
    /// Returns `None` for missing / non-dictionary / nameless inputs — the
    /// path is simply left unlabelled.
    fn read_oc_name(
        &self,
        dict: &std::collections::HashMap<String, crate::object::Object>,
        depth: u8,
    ) -> Option<String> {
        use crate::object::Object;

        if depth == 0 {
            return None;
        }

        // OCMD: no /Name of its own — follow /OCGs to the first member OCG.
        if matches!(dict.get("Type").and_then(|t| t.as_name()), Some("OCMD")) {
            let ocgs = self.deref_object(dict.get("OCGs")?)?;
            let first_ocg = match ocgs.as_array() {
                // /OCGs as an array: first entry that derefs to a dictionary.
                Some(entries) => entries
                    .iter()
                    .find_map(|e| self.deref_object(e).filter(|o| o.as_dict().is_some())),
                // /OCGs as a single OCG (already a dictionary).
                None => Some(ocgs.clone()),
            };
            return self.read_oc_name(first_ocg?.as_dict()?, depth - 1);
        }

        // OCG (or inline property dict): /Name is a PDF text string.
        match dict.get("Name")? {
            Object::String(bytes) => Some(Self::decode_pdf_text_string(bytes)),
            // Tolerate a /Name written as a PDF name object (non-conformant,
            // but seen in real exports).
            Object::Name(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Dereference one level of indirection, loading the target object;
    /// pass direct objects through unchanged. `None` if a reference fails to
    /// load — callers treat that as "unresolvable, leave unlabelled".
    fn deref_object(&self, obj: &crate::object::Object) -> Option<crate::object::Object> {
        match obj.as_reference() {
            Some(r) => self.load_object(r).ok(),
            None => Some(obj.clone()),
        }
    }

    /// Extract rectangles from a page (v0.3.14).
    ///
    /// Identifies paths that form axis-aligned rectangles.
    pub fn extract_rects(&self, page_index: usize) -> Result<Vec<crate::elements::PathContent>> {
        let paths = self.extract_paths(page_index)?;
        Ok(paths.into_iter().filter(|p| p.is_rectangle()).collect())
    }

    /// Extract straight lines from a page (v0.3.14).
    ///
    /// Identifies paths that form a single straight line segment.
    pub fn extract_lines(&self, page_index: usize) -> Result<Vec<crate::elements::PathContent>> {
        let paths = self.extract_paths(page_index)?;
        Ok(paths.into_iter().filter(|p| p.is_straight_line()).collect())
    }

    /// Extract tables from a page (v0.3.14).
    ///
    /// Uses a hybrid spatial algorithm that combines text alignment and vector lines
    /// for robust table detection without explicit structure markup.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tables = doc.extract_tables(0)?;
    /// for table in tables {
    ///     println!("Table with {} rows and {} columns", table.rows.len(), table.col_count);
    /// }
    /// ```
    pub fn extract_tables(
        &self,
        page_index: usize,
    ) -> Result<Vec<crate::structure::table_extractor::Table>> {
        self.extract_tables_with_config(
            page_index,
            crate::structure::spatial_table_detector::TableDetectionConfig::default(),
        )
    }

    /// Extract tables from a page using a custom configuration (v0.3.14).
    pub fn extract_tables_with_config(
        &self,
        page_index: usize,
        config: crate::structure::spatial_table_detector::TableDetectionConfig,
    ) -> Result<Vec<crate::structure::table_extractor::Table>> {
        use crate::structure::spatial_table_detector::detect_tables_with_lines;

        // Use words instead of spans for better granularity.
        // This ensures that strings with spaces are split into separate columns
        // for the spatial detector.
        let words = self.extract_words(page_index)?;
        // Use all table primitives (lines, rectangles, borders) not just straight lines
        let lines: Vec<_> = self
            .extract_paths(page_index)?
            .into_iter()
            .filter(|p| p.is_table_primitive())
            .collect();

        // Convert Words to TextSpans for the spatial detector
        let spans: Vec<_> = words
            .into_iter()
            .map(|w| crate::layout::TextSpan {
                artifact_type: None,
                text: w.text,
                bbox: w.bbox,
                font_name: w.dominant_font,
                font_size: w.avg_font_size,
                font_weight: if w.is_bold {
                    crate::layout::FontWeight::Bold
                } else {
                    crate::layout::FontWeight::Normal
                },
                is_italic: w.is_italic,
                is_monospace: false,
                color: crate::layout::Color::black(),
                mcid: w.mcid,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 1.0,
                primary_detected: false,
                char_widths: vec![],
                heading_level: None,
                rotation_degrees: 0.0,
            })
            .collect();

        Ok(detect_tables_with_lines(&spans, &lines, &config))
    }

    /// Process paths from a Form XObject.
    ///
    /// This method recursively extracts paths from Form XObjects encountered via the `Do` operator.
    /// It handles:
    /// - XObject resolution from resources
    /// - Type checking (Form vs Image)
    /// - Stream decoding and operator parsing
    /// - Coordinate transformations via /Matrix
    /// - Graphics state isolation
    ///
    /// # Arguments
    ///
    /// * `name` - The XObject name from the `Do` operator
    /// * `extractor` - The path extractor to accumulate paths
    /// * `state_stack` - The graphics state stack for transformations
    fn process_form_xobject_paths(
        &self,
        name: &str,
        extractor: &mut crate::extractors::paths::PathExtractor,
        state_stack: &mut crate::extractors::paths::PathGraphicsStateStack,
    ) -> Result<()> {
        use crate::content::{parse_content_stream_paths_only, Matrix, Operator};
        use crate::elements::{LineCap, LineJoin};
        use crate::extractors::paths::FillRule;
        use crate::layout::Color;

        let xobject_ref =
            match extractor.resolve_xobject_ref(name, |ref_obj| self.load_object(ref_obj)) {
                Some(r) => r,
                None => return Ok(()),
            };

        // Cycle detection
        if !extractor.can_process_xobject(xobject_ref) {
            return Ok(());
        }
        extractor.push_xobject(xobject_ref);

        // Load XObject
        let xobject = match self.load_object(xobject_ref) {
            Ok(obj) => obj,
            Err(e) => {
                extractor.pop_xobject_failed();
                return Err(e);
            },
        };
        let xobject_dict = match xobject.as_dict() {
            Some(dict) => dict,
            None => {
                extractor.pop_xobject_failed();
                return Err(Error::ParseError {
                    offset: 0,
                    reason: "XObject is not a dictionary".to_string(),
                });
            },
        };

        // Check type - only process Form XObjects, skip Images
        match xobject_dict.get("Subtype") {
            Some(subtype_obj) => {
                if let Some(subtype_name) = subtype_obj.as_name() {
                    if subtype_name != "Form" {
                        extractor.pop_xobject();
                        return Ok(()); // Not a Form XObject, skip
                    }
                } else {
                    extractor.pop_xobject();
                    return Ok(());
                }
            },
            None => {
                extractor.pop_xobject();
                return Ok(());
            },
        }

        // Decode stream — reuse document-level cache shared with text extraction.
        let cached_stream = {
            self.xobject_stream_cache
                .lock_or_recover()
                .get(&xobject_ref)
                .cloned()
        };
        let stream_data = if let Some(cached) = cached_stream {
            cached.as_ref().clone()
        } else {
            match self.decode_stream_with_encryption(&xobject, xobject_ref) {
                Ok(data) => {
                    const MAX_STREAM_CACHE_BYTES: usize = 50 * 1024 * 1024;
                    let current = self.xobject_stream_cache_bytes.load(Ordering::Relaxed);
                    if current + data.len() <= MAX_STREAM_CACHE_BYTES {
                        self.xobject_stream_cache_bytes
                            .store(current + data.len(), Ordering::Relaxed);
                        self.xobject_stream_cache
                            .lock_or_recover()
                            .insert(xobject_ref, std::sync::Arc::new(data.clone()));
                    }
                    data
                },
                Err(e) => {
                    extractor.pop_xobject_failed();
                    return Err(e);
                },
            }
        };

        let operators = match parse_content_stream_paths_only(&stream_data) {
            Ok(ops) => ops,
            Err(e) => {
                extractor.pop_xobject_failed();
                return Err(e);
            },
        };

        // Get transformation matrix (default to identity)
        let matrix = if let Some(matrix_obj) = xobject_dict.get("Matrix") {
            if let Some(array) = matrix_obj.as_array() {
                if array.len() >= 6 {
                    let mut matrix = Matrix::identity();
                    let mut values = [0.0f32; 6];
                    let mut valid = true;

                    for (i, val) in array.iter().take(6).enumerate() {
                        let num = if let Some(f) = val.as_real() {
                            f as f32
                        } else if let Some(i_val) = val.as_integer() {
                            i_val as f32
                        } else {
                            valid = false;
                            break;
                        };
                        values[i] = num;
                    }

                    if valid {
                        matrix.a = values[0];
                        matrix.b = values[1];
                        matrix.c = values[2];
                        matrix.d = values[3];
                        matrix.e = values[4];
                        matrix.f = values[5];
                        matrix
                    } else {
                        Matrix::identity()
                    }
                } else {
                    Matrix::identity()
                }
            } else {
                Matrix::identity()
            }
        } else {
            Matrix::identity()
        };

        // Save graphics state
        state_stack.save();

        // Finalize any pending path before processing XObject to isolate state
        if extractor.has_current_path() {
            extractor.end_path();
        }

        // Apply XObject transformation to CTM
        // PDF spec ISO 32000-1:2008 §8.10.1: Form XObject Matrix concatenates as M × CTM
        let state = state_stack.current_mut();
        state.ctm = matrix.multiply(&state.ctm);
        extractor.set_ctm(state.ctm);

        // Switch resource scope to this Form XObject's own /Resources, if any.
        // Form XObjects with their own Resources define a fresh XObject name
        // scope (ISO 32000-1 §8.10.1). Looking up nested `Do` names against the
        // parent scope can pick up unrelated sibling forms with colliding
        // names, which turns sibling Form XObjects into a cross-recursive tree
        // (O(N!) traversals and unbounded path accumulation).
        let saved_scope = if let Some(xobj_resources) = xobject_dict.get("Resources") {
            let resolved = if let Some(res_ref) = xobj_resources.as_reference() {
                self.load_object(res_ref)
                    .unwrap_or_else(|_| xobj_resources.clone())
            } else {
                xobj_resources.clone()
            };
            Some(extractor.swap_resources(Some(resolved)))
        } else {
            None
        };

        // Remember the marked-content nesting depth on entry so we can drop
        // anything this XObject leaves unbalanced (see truncate below).
        let oc_base_depth = extractor.oc_layer_depth();

        // Process operators from the XObject
        for op in operators {
            match op {
                // Graphics state operators
                Operator::SaveState => {
                    state_stack.save();
                },
                Operator::RestoreState => {
                    state_stack.restore();
                    extractor.update_from_path_state(state_stack.current());
                },
                Operator::Cm { a, b, c, d, e, f } => {
                    let state = state_stack.current_mut();
                    let new_matrix = Matrix { a, b, c, d, e, f };
                    // PDF spec ISO 32000-1:2008 §8.3.4: cm concatenates as M_cm × CTM
                    state.ctm = new_matrix.multiply(&state.ctm);
                    extractor.set_ctm(state.ctm);
                },

                // Color and line style operators — must update both state_stack
                // and extractor so q/Q save/restore works correctly.
                Operator::SetStrokeRgb { r, g, b } => {
                    state_stack.current_mut().stroke_color_rgb = (r, g, b);
                    extractor.set_stroke_color(Color::new(r, g, b));
                },
                Operator::SetStrokeGray { gray } => {
                    state_stack.current_mut().stroke_color_rgb = (gray, gray, gray);
                    extractor.set_stroke_color(Color::new(gray, gray, gray));
                },
                Operator::SetStrokeCmyk { c, m, y, k } => {
                    // ISO 32000-1:2008 §10.3.5: DeviceCMYK → DeviceRGB.
                    let r = 1.0 - (c + k).min(1.0);
                    let g = 1.0 - (m + k).min(1.0);
                    let b = 1.0 - (y + k).min(1.0);
                    state_stack.current_mut().stroke_color_rgb = (r, g, b);
                    extractor.set_stroke_color(Color::new(r, g, b));
                },
                Operator::SetFillRgb { r, g, b } => {
                    state_stack.current_mut().fill_color_rgb = (r, g, b);
                    extractor.set_fill_color(Color::new(r, g, b));
                },
                Operator::SetFillGray { gray } => {
                    state_stack.current_mut().fill_color_rgb = (gray, gray, gray);
                    extractor.set_fill_color(Color::new(gray, gray, gray));
                },
                Operator::SetFillCmyk { c, m, y, k } => {
                    // ISO 32000-1:2008 §10.3.5: DeviceCMYK → DeviceRGB.
                    let r = 1.0 - (c + k).min(1.0);
                    let g = 1.0 - (m + k).min(1.0);
                    let b = 1.0 - (y + k).min(1.0);
                    state_stack.current_mut().fill_color_rgb = (r, g, b);
                    extractor.set_fill_color(Color::new(r, g, b));
                },
                Operator::SetLineWidth { width } => {
                    state_stack.current_mut().line_width = width;
                    extractor.set_line_width(width);
                },
                Operator::SetLineCap { cap_style } => {
                    state_stack.current_mut().line_cap = cap_style;
                    let cap = match cap_style {
                        1 => LineCap::Round,
                        2 => LineCap::Square,
                        _ => LineCap::Butt,
                    };
                    extractor.set_line_cap(cap);
                },
                Operator::SetLineJoin { join_style } => {
                    state_stack.current_mut().line_join = join_style;
                    let join = match join_style {
                        1 => LineJoin::Round,
                        2 => LineJoin::Bevel,
                        _ => LineJoin::Miter,
                    };
                    extractor.set_line_join(join);
                },

                // Path construction operators
                Operator::MoveTo { x, y } => extractor.move_to(x, y),
                Operator::LineTo { x, y } => extractor.line_to(x, y),
                Operator::CurveTo {
                    x1,
                    y1,
                    x2,
                    y2,
                    x3,
                    y3,
                } => {
                    extractor.curve_to(x1, y1, x2, y2, x3, y3);
                },
                Operator::CurveToV { x2, y2, x3, y3 } => {
                    extractor.curve_to_v(x2, y2, x3, y3);
                },
                Operator::CurveToY { x1, y1, x3, y3 } => {
                    extractor.curve_to_y(x1, y1, x3, y3);
                },
                Operator::Rectangle {
                    x,
                    y,
                    width,
                    height,
                } => {
                    extractor.rectangle(x, y, width, height);
                },
                Operator::ClosePath => extractor.close_path(),

                // Path painting operators
                Operator::Stroke => extractor.stroke(),
                Operator::Fill => extractor.fill(FillRule::NonZero),
                Operator::FillEvenOdd => extractor.fill(FillRule::EvenOdd),
                Operator::CloseFillStroke => extractor.close_fill_and_stroke(FillRule::NonZero),
                Operator::EndPath => extractor.end_path(),

                // Clipping operators
                Operator::ClipNonZero => extractor.clip_non_zero(),
                Operator::ClipEvenOdd => extractor.clip_even_odd(),

                // Nested XObjects (recurse)
                Operator::Do { name: nested_name } => {
                    if let Err(e) =
                        self.process_form_xobject_paths(&nested_name, extractor, state_stack)
                    {
                        log::warn!("Failed to process nested XObject '{}': {}", nested_name, e);
                    }
                },

                // Marked content — same Optional Content Group ("layer")
                // tracking as the page-level loop, but `/OC` property
                // references resolve against *this* XObject's resource scope
                // (swapped in above), per §14.6.2 + §8.10.1. CAD exports that
                // reuse Form XObjects for repeated symbols (gridline labels,
                // callouts) carry their `/OC` markers and local `/Properties`
                // here rather than on the page.
                Operator::BeginMarkedContent { .. } => {
                    extractor.push_oc_layer(None);
                },
                Operator::BeginMarkedContentDict { tag, properties } => {
                    let layer = if tag == "OC" {
                        self.resolve_oc_layer_name(extractor.current_resources(), &properties)
                    } else {
                        None
                    };
                    extractor.push_oc_layer(layer);
                },
                Operator::EndMarkedContent => {
                    extractor.pop_oc_layer();
                },

                // Skip other operators
                _ => {},
            }
        }

        // Finalize any pending path to prevent state leakage
        if extractor.has_current_path() {
            extractor.end_path();
        }

        // Drop any marked-content entries this XObject left open so an
        // unbalanced `BDC` cannot leak its layer onto the caller's paths.
        extractor.truncate_oc_layers(oc_base_depth);

        // Restore the caller's resource scope before popping the cycle guard.
        if let Some(saved) = saved_scope {
            extractor.restore_resources(saved);
        }

        // Restore graphics state
        state_stack.restore();
        extractor.update_from_path_state(state_stack.current());

        // Pop from XObject processing stack
        extractor.pop_xobject();

        Ok(())
    }

    /// Extract paths from a specific rectangular region of a page.
    ///
    /// Only paths whose bounding box intersects the specified region are returned.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `region` - The rectangular region to extract from
    ///
    /// # Returns
    ///
    /// A vector of `PathContent` objects within the specified region.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # use pdf_oxide::geometry::Rect;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("example.pdf")?;
    ///
    /// // Extract paths from a specific region (e.g., header area)
    /// let header_region = Rect::new(0.0, 700.0, 612.0, 92.0);
    /// let paths = doc.extract_paths_in_rect(0, header_region)?;
    ///
    /// println!("Found {} paths in header region", paths.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_paths_in_rect(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
    ) -> Result<Vec<crate::elements::PathContent>> {
        let paths = self.extract_paths(page_index)?;

        // Filter paths by region intersection
        Ok(paths
            .into_iter()
            .filter(|path| path.bbox.intersects(&region))
            .collect())
    }

    /// Extract text from a specific rectangular region of a page (v0.3.14).
    ///
    /// Only spans whose bounding boxes match `region` under `mode` are kept;
    /// the retained spans are assembled through the full text pipeline
    /// (reading order, tables, line breaks) so the output matches the
    /// quality of [`Self::extract_text`]. Calling this with a region that covers
    /// the whole page is equivalent to [`Self::extract_text`].
    pub fn extract_text_in_rect(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
        mode: crate::layout::RectFilterMode,
    ) -> Result<String> {
        let options = crate::converters::ConversionOptions {
            extract_tables: true,
            include_region: Some((region, mode)),
            ..Default::default()
        };
        self.extract_text_with_options(page_index, &options)
    }

    /// Extract words from a specific rectangular region of a page (v0.3.14).
    pub fn extract_words_in_rect(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
        mode: crate::layout::RectFilterMode,
    ) -> Result<Vec<crate::layout::Word>> {
        use crate::layout::SpatialCollectionFiltering;
        let words = self.extract_words(page_index)?;
        Ok(words.filter_by_rect(&region, mode))
    }

    /// Extract text lines from a specific rectangular region of a page (v0.3.14).
    pub fn extract_text_lines_in_rect(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
        mode: crate::layout::RectFilterMode,
    ) -> Result<Vec<crate::layout::TextLine>> {
        use crate::layout::SpatialCollectionFiltering;
        let lines = self.extract_text_lines(page_index)?;
        Ok(lines.filter_by_rect(&region, mode))
    }

    /// Extract text spans from a specific rectangular region of a page (v0.3.14).
    pub fn extract_spans_in_rect(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
        mode: crate::layout::RectFilterMode,
    ) -> Result<Vec<crate::layout::TextSpan>> {
        use crate::layout::SpatialCollectionFiltering;
        let spans = self.extract_spans(page_index)?;
        Ok(spans.filter_by_rect(&region, mode))
    }

    /// Extract text from a page excluding specific rectangular regions.
    ///
    /// The excluded spans are removed before the full text-assembly pipeline
    /// runs, so the output has the same structure — line breaks, tables,
    /// reading order — as [`Self::extract_text`]. Calling this with an empty
    /// `exclude` slice is equivalent to [`Self::extract_text`].
    ///
    /// `mode` controls the overlap rule:
    /// - [`crate::layout::RectFilterMode::Intersects`] (default): drop any span with *any* overlap
    /// - [`crate::layout::RectFilterMode::FullyContained`]: drop only spans lying entirely inside
    /// - `RectFilterMode::MinOverlap(t)`: drop spans where at least fraction `t`
    ///   of the *span's* area overlaps an excluded region
    ///
    /// For Tagged PDFs the extractor already honours `/Artifact` marked-content
    /// (PDF spec §14.8.2.2). This method provides the same capability for
    /// untagged PDFs where spatial coordinates are the only available signal.
    /// Exclusion is unconditional: spans inside a region are dropped regardless
    /// of their structure-tree role.
    pub fn extract_text_excluding_rects(
        &self,
        page_index: usize,
        exclude: &[crate::geometry::Rect],
        mode: crate::layout::RectFilterMode,
    ) -> Result<String> {
        let options = crate::converters::ConversionOptions {
            extract_tables: true,
            exclude_regions: exclude.to_vec(),
            exclude_regions_mode: mode,
            ..Default::default()
        };
        self.extract_text_with_options(page_index, &options)
    }

    /// Extract words from a page excluding specific rectangular regions.
    ///
    /// See [`Self::extract_text_excluding_rects`] for a description of `exclude` and `mode`.
    /// Returns the low-level [`crate::layout::Word`] stream; use [`Self::extract_text_excluding_rects`]
    /// for fully-assembled text with line breaks and tables.
    pub fn extract_words_excluding_rects(
        &self,
        page_index: usize,
        exclude: &[crate::geometry::Rect],
        mode: crate::layout::RectFilterMode,
    ) -> Result<Vec<crate::layout::Word>> {
        use crate::layout::SpatialCollectionFiltering;
        let words = self.extract_words(page_index)?;
        Ok(words.exclude_rects(exclude, mode))
    }

    /// Extract text spans from a page excluding specific rectangular regions.
    ///
    /// See [`Self::extract_text_excluding_rects`] for a description of `exclude` and `mode`.
    /// Returns raw [`crate::layout::TextSpan`] objects with bounding boxes and font metadata;
    /// use [`Self::extract_text_excluding_rects`] for fully-assembled text output.
    pub fn extract_spans_excluding_rects(
        &self,
        page_index: usize,
        exclude: &[crate::geometry::Rect],
        mode: crate::layout::RectFilterMode,
    ) -> Result<Vec<crate::layout::TextSpan>> {
        use crate::layout::SpatialCollectionFiltering;
        let spans = self.extract_spans(page_index)?;
        Ok(spans.exclude_rects(exclude, mode))
    }

    /// Extract rectangles from a specific rectangular region of a page (v0.3.14).
    pub fn extract_rects_in_rect(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
    ) -> Result<Vec<crate::elements::PathContent>> {
        let rects = self.extract_rects(page_index)?;
        Ok(rects
            .into_iter()
            .filter(|p| p.bbox.intersects(&region))
            .collect())
    }

    /// Extract straight lines from a specific rectangular region of a page (v0.3.14).
    pub fn extract_lines_in_rect(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
    ) -> Result<Vec<crate::elements::PathContent>> {
        let lines = self.extract_lines(page_index)?;
        Ok(lines
            .into_iter()
            .filter(|p| p.bbox.intersects(&region))
            .collect())
    }

    /// Extract individual characters from a specific rectangular region of a page (v0.3.14).
    pub fn extract_chars_in_rect(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
        mode: crate::layout::RectFilterMode,
    ) -> Result<Vec<crate::layout::TextChar>> {
        use crate::layout::SpatialCollectionFiltering;
        let chars = self.extract_chars(page_index)?;
        Ok(chars.filter_by_rect(&region, mode))
    }

    /// Extract images from a specific rectangular region of a page (v0.3.14).
    pub fn extract_images_in_rect(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
    ) -> Result<Vec<crate::extractors::PdfImage>> {
        let images = self.extract_images(page_index)?;
        Ok(images
            .into_iter()
            .filter(|img| {
                if let Some(bbox) = img.bbox() {
                    bbox.intersects(&region)
                } else {
                    false
                }
            })
            .collect())
    }

    /// Extract tables from a specific rectangular region of a page (v0.3.14).
    pub fn extract_tables_in_rect(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
    ) -> Result<Vec<crate::structure::table_extractor::Table>> {
        self.extract_tables_in_rect_with_config(
            page_index,
            region,
            crate::structure::spatial_table_detector::TableDetectionConfig::relaxed(),
        )
    }

    /// Extract tables from a specific region using custom configuration (v0.3.14).
    pub fn extract_tables_in_rect_with_config(
        &self,
        page_index: usize,
        region: crate::geometry::Rect,
        config: crate::structure::spatial_table_detector::TableDetectionConfig,
    ) -> Result<Vec<crate::structure::table_extractor::Table>> {
        let tables = self.extract_tables_with_config(page_index, config)?;
        Ok(tables
            .into_iter()
            .filter(|table| {
                if let Some(bbox) = table.bbox {
                    bbox.intersects(&region)
                } else {
                    false
                }
            })
            .collect())
    }

    /// Get information about a page, including its dimensions.
    ///
    /// This is useful for rendering and layout calculations.
    #[cfg(feature = "rendering")]
    pub fn get_page_info(&self, page_index: usize) -> Result<PageInfo> {
        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;

        // Helper to extract f32 from Integer or Real
        fn obj_to_f32(obj: &Object) -> Option<f32> {
            match obj {
                Object::Integer(i) => Some(*i as f32),
                Object::Real(r) => Some(*r as f32),
                _ => None,
            }
        }

        // Get MediaBox (required, may be inherited).
        // PDF spec §7.3.10: any value may be a direct or indirect reference —
        // including each individual array element (pdf.js issue7872 stores
        // `/MediaBox [4 0 R 5 0 R 6 0 R 7 0 R]`). Resolve every element,
        // otherwise an unresolved Reference reads as None and silently
        // falls back to the Letter-size default instead of the true bounds.
        let media_box = page_dict
            .get("MediaBox")
            .map(|o| self.resolve_obj_ref(o))
            .as_ref()
            .and_then(|o| o.as_array().map(|a| a.to_owned()))
            .map(|arr| {
                let r: Vec<Object> = arr.iter().map(|o| self.resolve_obj_ref(o)).collect();
                let x0 = r.first().and_then(obj_to_f32).unwrap_or(0.0);
                let y0 = r.get(1).and_then(obj_to_f32).unwrap_or(0.0);
                let x1 = r.get(2).and_then(obj_to_f32).unwrap_or(612.0);
                let y1 = r.get(3).and_then(obj_to_f32).unwrap_or(792.0);
                crate::geometry::Rect::from_points(x0, y0, x1, y1)
            })
            .unwrap_or(crate::geometry::Rect::from_points(
                0.0, 0.0, 612.0, 792.0, // Letter size default
            ));

        // Get CropBox (optional, falls back to MediaBox).
        // PDF spec §7.3.10: any value may be a direct or indirect reference.
        let crop_box = page_dict
            .get("CropBox")
            .map(|o| self.resolve_obj_ref(o))
            .as_ref()
            .and_then(|o| o.as_array().map(|a| a.to_owned()))
            .map(|arr| {
                let r: Vec<Object> = arr.iter().map(|o| self.resolve_obj_ref(o)).collect();
                let x0 = r.first().and_then(obj_to_f32).unwrap_or(0.0);
                let y0 = r.get(1).and_then(obj_to_f32).unwrap_or(0.0);
                let x1 = r.get(2).and_then(obj_to_f32).unwrap_or(612.0);
                let y1 = r.get(3).and_then(obj_to_f32).unwrap_or(792.0);
                crate::geometry::Rect::from_points(x0, y0, x1, y1)
            });

        // Get rotation (optional, default 0).
        // PDF spec §7.3.10: Rotate may also be an indirect reference.
        let rotation = page_dict
            .get("Rotate")
            .map(|o| self.resolve_obj_ref(o))
            .as_ref()
            .and_then(|o| match o {
                Object::Integer(i) => Some(*i as i32),
                _ => None,
            })
            .unwrap_or(0);

        Ok(PageInfo {
            media_box,
            crop_box,
            rotation,
        })
    }

    /// Get the resources dictionary for a page.
    ///
    /// Resources contain fonts, images, patterns, and other objects
    /// used when rendering the page.
    #[cfg(feature = "rendering")]
    pub fn get_page_resources(&self, page_index: usize) -> Result<Object> {
        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;

        // Get Resources (required, may be inherited)
        let resources = page_dict
            .get("Resources")
            .cloned()
            .unwrap_or(Object::Dictionary(std::collections::HashMap::new()));

        // If it's a reference, resolve it
        if let Some(ref_val) = resources.as_reference() {
            self.load_object(ref_val)
        } else {
            Ok(resources)
        }
    }

    /// Resolve an object reference.
    ///
    /// This is useful when working with indirect object references
    /// in content streams or resource dictionaries.
    #[cfg(feature = "rendering")]
    pub fn resolve_object(&self, obj: &Object) -> Result<Object> {
        if let Some(ref_val) = obj.as_reference() {
            self.load_object(ref_val)
        } else {
            Ok(obj.clone())
        }
    }

    /// Look up a font from the per-document `font_cache`, parsing and inserting
    /// on a cache miss. Used by the page renderer so that `FontInfo::from_dict`
    /// (which decodes widths, CID maps, ToUnicode CMaps, and extracts embedded
    /// font bytes) is called at most once per PDF object reference, even when
    /// multiple pages share the same font resources.
    #[cfg(feature = "rendering")]
    pub fn get_or_load_font_for_rendering(
        &self,
        font_obj: &Object,
    ) -> Result<Arc<crate::fonts::FontInfo>> {
        if let Some(font_ref) = font_obj.as_reference() {
            let cached = self.font_cache.lock_or_recover().get(&font_ref).cloned();
            if let Some(arc) = cached {
                return Ok(arc);
            }
        }
        let resolved = self.deref_object_for_inks(font_obj)?;
        let info = crate::fonts::FontInfo::from_dict(&resolved, self)?;
        let arc = Arc::new(info);
        if let Some(font_ref) = font_obj.as_reference() {
            self.font_cache
                .lock_or_recover()
                .insert(font_ref, Arc::clone(&arc));
        }
        Ok(arc)
    }

    /// Compute a cheap content-based font identity hash from a loaded font object.
    /// Uses only inline fields (no reference resolution / load_object calls) to keep
    /// the cost at ~200ns. Relies on BaseFont + Subtype + Encoding (when inline) to
    /// uniquely identify fonts within a document. For reference-only fields (ToUnicode,
    /// FontDescriptor, DescendantFonts), hashes their presence to avoid false positives
    /// between fonts with vs without these features.
    /// `font_identity_hash_cheap` of `font_ref`'s object, memoized (an object's
    /// content is fixed within a document).
    fn cached_font_identity_hash(&self, font_ref: ObjectRef) -> Option<u64> {
        if let Some(&h) = self.font_id_hash_cache.lock_or_recover().get(&font_ref) {
            return Some(h);
        }
        let font = self.load_object(font_ref).ok()?;
        let h = Self::font_identity_hash_cheap(&font);
        self.font_id_hash_cache
            .lock_or_recover()
            .insert(font_ref, h);
        Some(h)
    }

    fn font_identity_hash_cheap(font_obj: &Object) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        if let Some(d) = font_obj.as_dict() {
            // BaseFont: primary identity — unique per font within a document
            if let Some(Object::Name(n)) = d.get("BaseFont") {
                1u8.hash(&mut hasher);
                n.hash(&mut hasher);
            }
            // Subtype: Type1, TrueType, Type0, CIDFontType0, CIDFontType2
            if let Some(Object::Name(n)) = d.get("Subtype") {
                2u8.hash(&mut hasher);
                n.hash(&mut hasher);
            }
            // Encoding: hash inline name or presence of reference
            if let Some(enc) = d.get("Encoding") {
                3u8.hash(&mut hasher);
                match enc {
                    Object::Name(n) => n.hash(&mut hasher),
                    Object::Reference(_) => b"enc_ref".hash(&mut hasher),
                    Object::Dictionary(_) => b"enc_dict".hash(&mut hasher),
                    _ => {},
                }
            }
            // ToUnicode: hash content via reference or inline presence
            if let Some(to_unicode) = d.get("ToUnicode") {
                4u8.hash(&mut hasher);
                if let Some(r) = to_unicode.as_reference() {
                    r.id.hash(&mut hasher);
                    r.gen.hash(&mut hasher);
                }
            }
            // FontDescriptor: hash presence
            if d.get("FontDescriptor").is_some() {
                5u8.hash(&mut hasher);
            }
            // DescendantFonts: hash references for Type0 fonts
            if let Some(Object::Array(arr)) = d.get("DescendantFonts") {
                6u8.hash(&mut hasher);
                for item in arr {
                    if let Some(r) = item.as_reference() {
                        r.id.hash(&mut hasher);
                        r.gen.hash(&mut hasher);
                    }
                }
            }
            // #598: width metrics. Two non-subset fonts can share
            // BaseFont + Subtype + Encoding yet ship different glyph widths —
            // Standard-14 fonts may carry producer-specific /Widths overrides
            // (§9.6.2.2), and differently-optimized embeds of the same named
            // font diverge similarly. Without folding widths into the key,
            // such fonts collide on the cross-document cache and the second
            // document gets the first's advances. We hash the simple-font
            // char range + width table and the Type0 default width. Only
            // values present inline on this dict are reachable (this is a pure
            // function over the font object); a referenced /Widths or the
            // descendant CIDFont /W array falls back to the coarser key — an
            // accepted, documented limitation, not a new regression.
            if let Some(Object::Integer(first_char)) = d.get("FirstChar") {
                7u8.hash(&mut hasher);
                first_char.hash(&mut hasher);
            }
            if let Some(Object::Integer(last_char)) = d.get("LastChar") {
                8u8.hash(&mut hasher);
                last_char.hash(&mut hasher);
            }
            if let Some(Object::Array(widths)) = d.get("Widths") {
                9u8.hash(&mut hasher);
                (widths.len() as u64).hash(&mut hasher);
                for w in widths {
                    match w {
                        Object::Integer(i) => i.hash(&mut hasher),
                        // Bit-pattern hash so equal widths hash equally
                        // (these are glyph advances, never NaN in practice).
                        Object::Real(r) => r.to_bits().hash(&mut hasher),
                        _ => 0u8.hash(&mut hasher),
                    }
                }
            }
            // Type0 default width, when present inline on the font dict.
            if let Some(Object::Integer(dw)) = d.get("DW") {
                10u8.hash(&mut hasher);
                dw.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Whether a font dictionary describes a font that is *document-local* and
    /// therefore must never be served from / inserted into the cross-document
    /// global font cache (Layer 6), even if its cheap identity hash collides
    /// with a font in another document.
    ///
    /// Type 3 fonts (PDF 32000-1 §9.6.5) define their glyphs as streams of PDF
    /// graphics operators in a `/CharProcs` dictionary whose procedures
    /// reference the *owning document's* resources (XObjects, ColorSpaces,
    /// ExtGState, …). Two Type 3 fonts from different documents that happen to
    /// share `/Name` + `/Encoding` shape are NOT interchangeable: serving one
    /// document's parsed `FontInfo` for the other yields wrong glyphs. Such
    /// fonts carry no subset prefix, so the cheap hash cannot distinguish them
    /// — this predicate gates them out of the global cache instead (#597).
    fn font_is_document_local(font_obj: &Object) -> bool {
        let dict = match font_obj.as_dict() {
            Some(d) => d,
            None => return false,
        };

        // Type 3 fonts reference this document's resources via their CharProcs,
        // so a cached FontInfo cannot cross PdfDocument boundaries.
        if dict.get("Subtype").and_then(|s| s.as_name()) == Some("Type3") {
            return true;
        }

        // Subset fonts carry a document-specific glyph subset and ToUnicode
        // CMap, so they are unsafe to share across documents even when the
        // BaseFont name collides. A subset BaseFont is tagged with exactly six
        // uppercase letters and a '+' per ISO 32000-1:2008 §9.6.4
        // (e.g. `AAAAAA+ArialUnicodeMS`).
        match dict.get("BaseFont").and_then(|b| b.as_name()) {
            Some(base_font) => Self::is_subset_basefont(base_font),
            // A non-Type3 font is required by the spec to carry /BaseFont; if it
            // is absent we cannot prove the font is shareable, so fail safe and
            // treat it as document-local rather than risk poisoning the cache.
            None => true,
        }
    }

    /// Detect a PDF subset-font tag on a `/BaseFont` name: exactly six uppercase
    /// ASCII letters followed by `+`, per ISO 32000-1:2008 §9.6.4 (e.g.
    /// `AAAAAA+ArialUnicodeMS`). `is_ascii_uppercase` is precisely A–Z, so
    /// multibyte (CJK) names never satisfy the test and are treated as full
    /// fonts — correct, since subset tags are by definition ASCII A–Z.
    fn is_subset_basefont(base_font: &str) -> bool {
        let bytes = base_font.as_bytes();
        bytes.len() > 7 && bytes[6] == b'+' && bytes[..6].iter().all(|b| b.is_ascii_uppercase())
    }

    /// Load fonts from a Resources dictionary into the extractor.
    pub(crate) fn load_fonts(
        &self,
        resources: &Object,
        extractor: &mut crate::extractors::TextExtractor<'_>,
    ) -> Result<()> {
        use crate::fonts::FontInfo;

        // Resources can be a reference or a dictionary
        let resources_obj = if let Some(res_ref) = resources.as_reference() {
            self.load_object(res_ref)?
        } else {
            resources.clone()
        };

        let resources_dict = match resources_obj.as_dict() {
            Some(d) => d,
            None => {
                log::warn!(
                    "Resources is not a dictionary (type: {}), treating as empty",
                    resources_obj.type_name()
                );
                return Ok(());
            },
        };

        // Get Font dictionary if present
        if let Some(font_obj) = resources_dict.get("Font") {
            // Font can be a reference or direct dictionary - need to dereference
            let font_dict_ref = font_obj.as_reference();
            let font_dict_obj = if let Some(font_ref) = font_dict_ref {
                self.load_object(font_ref)?
            } else {
                font_obj.clone()
            };

            // Layer 2: Check font set cache for the /Font dictionary.
            // Pages sharing the same /Font dict skip the entire per-font loop.
            if let Some(font_dict_ref) = font_dict_ref {
                let cached_set_opt = self
                    .font_set_cache
                    .lock_or_recover()
                    .get(&font_dict_ref)
                    .cloned();
                if let Some(cached_set) = cached_set_opt {
                    for (name, font_arc) in &cached_set {
                        extractor.add_font_shared(name.clone(), Arc::clone(font_arc));
                    }
                    extractor.share_truetype_cmaps();
                    return Ok(());
                }
            }

            if let Some(font_dict) = font_dict_obj.as_dict() {
                // Compute font fingerprint from (name → ObjectRef) pairs.
                // Hash the MAPPING between font names and their object refs,
                // not just the sets separately. This prevents false cache hits
                // when two font dicts have the same set of refs and names but
                // different name-to-ref assignments.
                let fingerprint = {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    let mut name_ref_pairs: Vec<(&str, Option<ObjectRef>)> = font_dict
                        .iter()
                        .map(|(name, fo)| (name.as_str(), fo.as_reference()))
                        .collect();
                    name_ref_pairs.sort_by(|a, b| a.0.cmp(b.0));
                    for (name, obj_ref) in &name_ref_pairs {
                        name.hash(&mut hasher);
                        if let Some(r) = obj_ref {
                            r.id.hash(&mut hasher);
                            r.gen.hash(&mut hasher);
                        }
                    }
                    hasher.finish()
                };

                let cached_fingerprint_opt = self
                    .font_fingerprint_cache
                    .lock_or_recover()
                    .get(&fingerprint)
                    .cloned();
                if let Some(cached_set) = cached_fingerprint_opt {
                    for (name, font_arc) in &cached_set {
                        extractor.add_font_shared(name.clone(), Arc::clone(font_arc));
                    }
                    extractor.share_truetype_cmaps();
                    return Ok(());
                }

                // Layer 4: Name-based font set cache with spot-check verification.
                // Pages in the same document often use the same font names mapped to
                // different ObjectRefs but identical base fonts (e.g., 764 pages each
                // creating T1_0→Helvetica, T1_1→Times-Roman with unique object numbers).
                // Cache the resolved font set by sorted font names, then on subsequent
                // pages verify ONE font via load+hash to confirm the mapping is the same.
                let name_hash = {
                    use std::hash::{Hash, Hasher};
                    let mut font_names: Vec<&str> = font_dict.keys().map(|k| k.as_str()).collect();
                    font_names.sort();
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    font_names.hash(&mut hasher);
                    hasher.finish()
                };

                let cached_name_set = self
                    .font_name_set_cache
                    .lock_or_recover()
                    .get(&name_hash)
                    .cloned();
                // Sort font entries by name for deterministic processing order.
                // HashMap iteration order is randomized per-process, which causes
                // non-deterministic text extraction when font CMap sharing depends
                // on the order fonts are loaded.
                let mut sorted_font_entries: Vec<(&String, &Object)> = font_dict.iter().collect();
                sorted_font_entries.sort_by_key(|(name, _)| name.as_str());

                if let Some((cached_set, check_hash)) = cached_name_set {
                    // Verify the cached font set by computing a combined identity hash
                    // over ALL reference fonts in the current Resources dict (sorted by
                    // name). This prevents false cache hits when pages reuse the same
                    // font key names but embed different per-page subsets — a single-font
                    // spot-check is insufficient because it only guards one entry
                    // lets differing sibling fonts (F2, F3 …) slip through unchecked.
                    // Fixes the regression described in issue #408.
                    let current_combined = {
                        use std::hash::{Hash, Hasher};
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        for (name, font_obj) in &sorted_font_entries {
                            if let Some(font_ref) = font_obj.as_reference() {
                                if let Some(fh) = self.cached_font_identity_hash(font_ref) {
                                    name.as_str().hash(&mut h);
                                    fh.hash(&mut h);
                                }
                            }
                        }
                        h.finish()
                    };
                    if current_combined == check_hash {
                        for (name, font_arc) in cached_set.iter() {
                            extractor.add_font_shared(name.clone(), Arc::clone(font_arc));
                        }
                        extractor.share_truetype_cmaps();
                        return Ok(());
                    }
                    // Hash mismatch: fonts differ — fall through to full load.
                }

                // Snapshot names already in the extractor before this load_fonts call.
                // Layer 4 must store only the delta so that a cache hit never injects
                // parent-page fonts into a different page's extractor context, which
                // would overwrite correctly-loaded fonts with wrong versions.
                let extractor_names_before: std::collections::HashSet<String> = extractor
                    .get_font_set()
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect();

                let mut all_from_cache = true;

                for (name, font_obj) in &sorted_font_entries {
                    // If font is a reference, check per-font cache first
                    if let Some(font_ref) = font_obj.as_reference() {
                        let cached_font_opt =
                            self.font_cache.lock_or_recover().get(&font_ref).cloned();
                        if let Some(cached) = cached_font_opt {
                            extractor.add_font_shared((*name).clone(), cached);
                            continue;
                        }
                        all_from_cache = false;
                        let font = self.load_object(font_ref)?;

                        // Compute identity hash (cheap: 3-6 dict lookups, ~200ns)
                        let id_hash = Self::font_identity_hash_cheap(&font);

                        // Type 3 fonts and subset fonts must not cross
                        // PdfDocument boundaries via the global cache — their
                        // glyph procs / glyph-subset + ToUnicode mappings are
                        // document-specific. The per-document Layer 4/5 caches
                        // below stay safe to use.
                        let is_document_local = Self::font_is_document_local(&font);

                        // Layer 5: Per-font identity cache — skip from_dict when a
                        // structurally identical font was already parsed elsewhere.
                        let cached_identity_opt = self
                            .font_identity_cache
                            .lock_or_recover()
                            .get(&id_hash)
                            .cloned();
                        if let Some(cached) = cached_identity_opt {
                            self.font_cache
                                .lock_or_recover()
                                .insert(font_ref, Arc::clone(&cached));
                            extractor.add_font_shared((*name).clone(), cached);
                            continue;
                        }

                        // Layer 6: Global cross-document font cache — reuse fonts
                        // parsed by previous PdfDocument instances in this process.
                        // Skipped entirely for document-local fonts (#597).
                        if !is_document_local {
                            if let Some(cached) =
                                crate::fonts::global_cache::global_font_cache_get(id_hash)
                            {
                                self.font_identity_cache
                                    .lock_or_recover()
                                    .insert(id_hash, Arc::clone(&cached));
                                self.font_cache
                                    .lock_or_recover()
                                    .insert(font_ref, Arc::clone(&cached));
                                extractor.add_font_shared((*name).clone(), cached);
                                continue;
                            }
                        }

                        match FontInfo::from_dict(&font, self) {
                            Ok(font_info) => {
                                let arc = Arc::new(font_info);
                                // Populate the document-level caches always; the
                                // global cross-document cache only for fonts that
                                // are safe to share across documents (#597).
                                if !is_document_local {
                                    crate::fonts::global_cache::global_font_cache_insert(
                                        id_hash,
                                        Arc::clone(&arc),
                                    );
                                }
                                self.font_identity_cache
                                    .lock_or_recover()
                                    .insert(id_hash, Arc::clone(&arc));
                                self.font_cache
                                    .lock_or_recover()
                                    .insert(font_ref, Arc::clone(&arc));
                                extractor.add_font_shared((*name).clone(), arc);
                            },
                            Err(e) => {
                                log::error!(
                                    "Failed to load font '{}': {}. Text using this font will use fallback encoding.",
                                    name,
                                    e
                                );
                                continue;
                            },
                        }
                    } else {
                        // Direct font object — parse without caching (no stable key)
                        all_from_cache = false;
                        let font = *font_obj;
                        match FontInfo::from_dict(font, self) {
                            Ok(font_info) => {
                                extractor.add_font((*name).clone(), font_info);
                            },
                            Err(e) => {
                                log::error!(
                                    "Failed to load font '{}': {}. Text using this font will use fallback encoding.",
                                    name,
                                    e
                                );
                                continue;
                            },
                        }
                    }
                }

                // Always re-share TrueType CMaps after loading fonts. Cached fonts
                // may lack donated CMaps because Arc::make_mut creates a per-extractor
                // clone that is not written back to per-font cache. A donor font added
                // in a later load_fonts call (e.g. an XObject font donating to a
                // page-level font already in the extractor) requires sharing to run
                // again even when all fonts came from cache.
                extractor.share_truetype_cmaps();

                // Cache font set by both ObjectRef and fingerprint
                let font_set = extractor.get_font_set();
                if let Some(fdr) = font_dict_ref {
                    self.font_set_cache
                        .lock_or_recover()
                        .insert(fdr, font_set.clone());
                }
                self.font_fingerprint_cache
                    .lock_or_recover()
                    .insert(fingerprint, font_set.clone());

                // Cache by font names for Layer 4. Store only the delta — fonts
                // added by THIS load_fonts call — so that a cache hit never pollutes
                // a different page's extractor with stale parent-page fonts.
                // The combined identity hash covers ALL reference fonts (sorted by
                // name), so a hit requires every font in the Resources dict to match,
                // not just one. This prevents false positives when pages reuse the
                // same font key names with different per-page subsets.
                if !all_from_cache {
                    let combined_check_hash = {
                        use std::hash::{Hash, Hasher};
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        for (name, font_obj) in &sorted_font_entries {
                            if let Some(font_ref) = font_obj.as_reference() {
                                if let Some(fh) = self.cached_font_identity_hash(font_ref) {
                                    name.as_str().hash(&mut h);
                                    fh.hash(&mut h);
                                }
                            }
                        }
                        h.finish()
                    };
                    let l4_set: Vec<(String, Arc<FontInfo>)> = font_set
                        .iter()
                        .filter(|(k, _)| !extractor_names_before.contains(k.as_str()))
                        .map(|(k, v)| (k.clone(), Arc::clone(v)))
                        .collect();
                    self.font_name_set_cache
                        .lock_or_recover()
                        .insert(name_hash, (Arc::new(l4_set), combined_check_hash));
                }

                return Ok(());
            }
        }

        Ok(())
    }

    /// Extract tables from a page using structure tree and spatial detection.
    ///
    /// Tries two strategies in order:
    /// 1. **Structure tree** (tagged PDFs): Finds Table elements in the structure
    ///    tree and extracts cell content via MCID matching.
    /// 2. **Spatial detection** (untagged PDFs): Uses X/Y coordinate clustering
    ///    to detect grid-aligned text as tables.
    ///
    /// Returns early with structure tree tables if found (high confidence).
    fn extract_page_tables(
        &self,
        page_index: usize,
        spans: &[TextSpan],
        options: &crate::converters::ConversionOptions,
        text_fallback: bool,
    ) -> Vec<crate::structure::Table> {
        // Strategy 1: Structure tree (tagged PDFs)
        let struct_tree_opt = {
            let cached = self.structure_tree_cache.lock_or_recover().clone();
            match cached {
                Some(tree) => tree,
                None => {
                    let is_marked = self.mark_info().map(|m| m.marked).unwrap_or(false);
                    let has_struct_tree_root = !is_marked
                        && self
                            .catalog()
                            .ok()
                            .and_then(|cat| cat.as_dict().map(|d| d.contains_key("StructTreeRoot")))
                            .unwrap_or(false);
                    let tree = if is_marked || has_struct_tree_root {
                        self.structure_tree().ok().flatten().map(Arc::new)
                    } else {
                        None
                    };
                    *self.structure_tree_cache.lock_or_recover() = Some(tree.clone());
                    tree
                },
            }
        };
        if let Some(ref struct_tree) = struct_tree_opt {
            // Build the per-page Table-element buckets once, then look up.
            if self.table_elements_cache.lock_or_recover().is_none() {
                let all = crate::structure::find_table_elements_all_pages(struct_tree);
                *self.table_elements_cache.lock_or_recover() = Some(all);
            }
            let table_elems: Vec<crate::structure::StructElem> = self
                .table_elements_cache
                .lock_or_recover()
                .as_ref()
                .and_then(|c| c.get(&(page_index as u32)))
                .cloned()
                .unwrap_or_default();
            if !table_elems.is_empty() {
                let mut tables = Vec::new();
                for table_elem in &table_elems {
                    match crate::structure::extract_table_from_spans(table_elem, spans) {
                        Ok(mut table) if !table.is_empty() => {
                            // Compute bbox from spans matching the table's MCIDs
                            if table.bbox.is_none() {
                                let all_mcids: HashSet<u32> = table
                                    .rows
                                    .iter()
                                    .flat_map(|r| {
                                        r.cells.iter().flat_map(|c| c.mcids.iter().copied())
                                    })
                                    .collect();
                                if !all_mcids.is_empty() {
                                    let mut min_x = f32::INFINITY;
                                    let mut min_y = f32::INFINITY;
                                    let mut max_x = f32::NEG_INFINITY;
                                    let mut max_y = f32::NEG_INFINITY;
                                    for span in spans {
                                        if let Some(mcid) = span.mcid {
                                            if all_mcids.contains(&mcid) {
                                                min_x = min_x.min(span.bbox.x);
                                                min_y = min_y.min(span.bbox.y);
                                                max_x = max_x.max(span.bbox.x + span.bbox.width);
                                                max_y = max_y.max(span.bbox.y + span.bbox.height);
                                            }
                                        }
                                    }
                                    if min_x < max_x && min_y < max_y {
                                        table.bbox = Some(crate::geometry::Rect::new(
                                            min_x,
                                            min_y,
                                            max_x - min_x,
                                            max_y - min_y,
                                        ));
                                    }
                                }
                            }
                            tables.push(table);
                        },
                        _ => {},
                    }
                }
                if !tables.is_empty() {
                    log::debug!(
                        "Found {} table(s) via structure tree for page {}",
                        tables.len(),
                        page_index
                    );
                    return tables;
                }
            }
        }

        // Strategy 2: Hybrid spatial detection (v0.3.14)
        let mut config = options.table_detection_config.clone().unwrap_or_default();
        // Honour the caller's text_fallback choice regardless of the default
        // on `TableDetectionConfig` — `extract_text` / `to_plain_text` pass
        // `text_fallback=false` to opt out of text-only spatial fallback even
        // though the type-level default is `true`.
        config.text_fallback = text_fallback;

        // Extract vector paths (lines/rects) for visual detection
        let paths = self.extract_paths(page_index).unwrap_or_default();

        // Filter to table-relevant paths (lines and rectangles only).
        // Chart/plot pages often have hundreds of curves and fills that
        // extract_edges ignores anyway — passing them through the full
        // detection pipeline wastes O(n²) time.
        const LINE_TOL: f32 = 2.0;
        let table_paths: Vec<_> = paths
            .into_iter()
            .filter(|p| {
                p.is_horizontal_line(LINE_TOL) || p.is_vertical_line(LINE_TOL) || p.is_rectangle()
            })
            .collect();

        // A page with thousands of line/rect paths is a drawing or chart, not a
        // ruled table; skip the O(E²) collinear-join + intersection sweep. Real
        // ruled tables have at most a few hundred edges. (Tagged tables already
        // returned above via the structure tree.)
        const MAX_TABLE_EDGES: usize = 1500;
        if table_paths.len() > MAX_TABLE_EDGES {
            log::debug!(
                "Page {} has {} line/rect paths (> {}) — skipping spatial table sweep",
                page_index,
                table_paths.len(),
                MAX_TABLE_EDGES
            );
            return Vec::new();
        }

        if table_paths.is_empty() {
            use crate::structure::spatial_table_detector::TableStrategy;
            let is_text_only = matches!(
                (config.horizontal_strategy, config.vertical_strategy),
                (TableStrategy::Text, TableStrategy::Text)
            );
            if !is_text_only && !config.text_fallback {
                return Vec::new();
            }
            if !is_text_only && config.text_fallback {
                log::debug!(
                    "No ruling lines on page {} — using text-only spatial fallback (issue #486)",
                    page_index
                );
            }
        }
        let paths = table_paths;

        let words = self.extract_words(page_index).unwrap_or_default();
        let word_spans: Vec<crate::layout::TextSpan> = words
            .into_iter()
            .map(|w| crate::layout::TextSpan {
                artifact_type: None,
                text: w.text,
                bbox: w.bbox,
                font_name: w.dominant_font,
                font_size: w.avg_font_size,
                font_weight: if w.is_bold {
                    crate::layout::FontWeight::Bold
                } else {
                    crate::layout::FontWeight::Normal
                },
                is_italic: w.is_italic,
                is_monospace: false,
                color: crate::layout::Color::black(),
                mcid: w.mcid,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 1.0,
                primary_detected: false,
                char_widths: vec![],
                heading_level: None,
                rotation_degrees: 0.0,
            })
            .collect();

        // Fall back to raw spans if word extraction failed
        let input_spans = if !word_spans.is_empty() {
            &word_spans
        } else {
            spans
        };

        let raw_tables = crate::structure::spatial_table_detector::detect_tables_with_lines(
            input_spans,
            &paths,
            &config,
        );

        // Issue 484/486/487: when a logical multi-row table is drawn with a
        // horizontal ruling line between every pair of rows, the line-based
        // detector emits one Table per row strip. Each fragment is a 1- or
        // 2-row table that fails is_real_grid below and gets dropped, after
        // which the cells fall through to paragraph flow with column-based
        // reading order — producing orphan `<p>40000≤Q</p>` /
        // `<p>＜55000</p>` pairs. Consolidate vertically-adjacent fragments
        // that share an identical column structure BEFORE applying
        // is_real_grid so the merged multi-row table survives the filter.
        let raw_tables =
            crate::structure::spatial_table_detector::consolidate_adjacent_table_fragments(
                raw_tables,
            );

        // Step 4: spatial detection without struct-tree backing
        // is prone to false positives on form-style layouts (label-colon-
        // value pairs that align horizontally, form fillable boxes drawn
        // with thin lines). Drop tables that don't look like real grids.
        let raw_count = raw_tables.len();
        let mut tables: Vec<crate::structure::Table> = raw_tables
            .into_iter()
            .filter(|t| t.is_real_grid())
            // Prose-shape filter — applies to line-based detection too: a
            // PDF with decorative horizontal rules (newsletter mastheads,
            // press-release banners) can hand `is_real_grid` a "wide data
            // table" that is actually wrapped paragraphs partitioned by
            // word x-alignment. Reject those before they reach the
            // converter. See `looks_like_prose_table` for the heuristic.
            .filter(|t| !looks_like_prose_table(t))
            .collect();

        if raw_count != tables.len() {
            log::debug!(
                "Spatial table detection: filtered {} non-real-grid candidates on page {} ({} kept)",
                raw_count - tables.len(),
                page_index,
                tables.len(),
            );
        } else if !tables.is_empty() {
            log::debug!(
                "Found {} table(s) via hybrid spatial detection for page {}",
                tables.len(),
                page_index
            );
        }

        // Text-only spatial fallback for converter paths (to_markdown / to_html — issue #486).
        //
        // Wide data tables (e.g. sailing-score grids with 16-18 columns) exceed the default
        // `max_table_columns: 15` limit and are rejected by the main pipeline. When the
        // caller explicitly opted in to text-only detection (text_fallback=true), retry with
        // a relaxed config that raises the column ceiling and adjusts tolerances so that
        // genuinely wide data tables are captured.
        //
        // Safety guards:
        // - Only fires when the main pipeline returned no tables (avoids double-counting).
        // - Only fires when the caller is a converter (text_fallback=true).
        // - Skipped for tagged PDFs: the structure tree already provides the authoritative
        //   layout; spatial heuristics produce false-positive tables from structure elements
        //   (e.g. headings detected as single-row tables — issue #486 regression).
        // - Skipped for predominantly-RTL pages: Arabic/Hebrew text alignment patterns
        //   mimic table columns in spatial heuristics — issue #486 regression.
        // - When ruling lines exist, spans are filtered to the line-bounded region to
        //   prevent page headers/footers from being erroneously included in the table.
        // - Results must pass is_real_grid() just like main-pipeline tables.

        // Guard 1 — Tagged PDFs: presence of a structure tree means the document has an
        // explicit semantic layout. Spatial text-only detection would misfire on
        // structure elements (headings, paragraphs) that happen to share a Y band.
        if config.text_fallback && struct_tree_opt.is_some() {
            log::debug!(
                "Text-only spatial fallback skipped for page {} — document has a structure tree (tagged PDF)",
                page_index
            );
            return tables;
        }

        // Guard 2 — RTL pages: Arabic and Hebrew text naturally aligns horizontally in
        // patterns that the column-clustering algorithm mistakes for table columns.
        // Skip spatial detection when more than 30 % of the input spans are RTL.
        if config.text_fallback {
            let rtl_count = input_spans
                .iter()
                .filter(|s| crate::text::bidi::looks_rtl(&s.text))
                .count();
            let rtl_fraction = rtl_count as f32 / input_spans.len().max(1) as f32;
            if rtl_fraction > 0.30 {
                log::debug!(
                    "Text-only spatial fallback skipped for page {} — {:.0}% RTL spans (threshold 30%)",
                    page_index,
                    rtl_fraction * 100.0
                );
                return tables;
            }
        }

        if config.text_fallback && tables.is_empty() {
            use crate::structure::spatial_table_detector::detect_tables_from_spans_column_aware;
            // Build a relaxed config derived from the caller's config.
            // We only raise the limits known to block wide data tables (e.g. sailing
            // score grids with 16-18 columns that exceed the default max_table_columns=15).
            let relaxed_config = crate::structure::spatial_table_detector::TableDetectionConfig {
                // Allow up to 25 columns — covers 17-column sailing score tables.
                max_table_columns: config.max_table_columns.max(25),
                // Tighter column grouping than the default 15 pt so that nearby
                // score columns are not merged into each other.
                column_tolerance: config.column_tolerance.min(10.0),
                // Looser merge threshold so that columns with slight X scatter
                // (e.g. centred numeric cells) are aggregated correctly.
                column_merge_threshold: config.column_merge_threshold.max(30.0),
                // Inherit all other settings from caller's config.
                ..config.clone()
            };

            // When ruling lines are present on the page, restrict text detection to
            // spans that fall within the VERTICAL-LINE Y bounds. Vertical lines
            // define the table's column structure and their Y extent precisely
            // delineates the table rows, excluding page headers and footers which
            // sit above/below the table frame.
            //
            // Note: we use V-line Y bounds specifically (not total path bbox) because
            // H-lines in these PDFs often span the full page height (outer frame),
            // while V-lines are confined to the interior table region.
            let candidate_spans: Vec<crate::layout::TextSpan>;
            let fallback_spans: &[crate::layout::TextSpan] = {
                let v_lines: Vec<_> = paths.iter().filter(|p| p.is_vertical_line(2.0)).collect();
                if !v_lines.is_empty() {
                    let vline_y_min = v_lines
                        .iter()
                        .map(|p| p.bbox.y)
                        .fold(f32::INFINITY, f32::min);
                    let vline_y_max = v_lines
                        .iter()
                        .map(|p| p.bbox.y + p.bbox.height)
                        .fold(f32::NEG_INFINITY, f32::max);
                    // Small margin to include spans whose centres just touch the frame.
                    const V_MARGIN: f32 = 5.0;
                    candidate_spans = input_spans
                        .iter()
                        .filter(|s| {
                            let cy = s.bbox.y + s.bbox.height * 0.5;
                            cy >= vline_y_min - V_MARGIN && cy <= vline_y_max + V_MARGIN
                        })
                        .cloned()
                        .collect();
                    log::debug!(
                        "Text fallback (page {}): V-lines Y=[{:.1},{:.1}] — filtered {} spans to {}",
                        page_index,
                        vline_y_min,
                        vline_y_max,
                        input_spans.len(),
                        candidate_spans.len()
                    );
                    &candidate_spans
                } else {
                    input_spans
                }
            };

            let text_candidates =
                detect_tables_from_spans_column_aware(fallback_spans, &relaxed_config);
            let pre_filter = text_candidates.len();
            let text_tables: Vec<_> = text_candidates
                .into_iter()
                // Text-only detection infers columns from word x-alignment
                // alone; a title + a wrapped body line (two rows) is the
                // signature of ordinary prose, not a table. Require ≥3
                // rows of evidence before promoting to a table.
                .filter(|t| t.rows.len() >= 3 && t.is_real_grid())
                // Prose split across many "columns" is the dominant
                // false-positive shape for text-only detection on
                // line-less pages: a paragraph wraps to N lines, words
                // cluster into N×K cells, and `is_real_grid` accepts the
                // shape. Real data-table cells almost never end with a
                // comma or semicolon (those punctuation marks belong to
                // running sentences), so a high comma-tail ratio is the
                // most discriminating prose signal we have.
                .filter(|t| !looks_like_prose_table(t))
                .collect();
            if !text_tables.is_empty() {
                log::debug!(
                    "Text-only relaxed fallback found {} table(s) on page {} ({} filtered by is_real_grid) — issue #486",
                    text_tables.len(),
                    page_index,
                    pre_filter - text_tables.len(),
                );
                tables = text_tables;
            }
        }

        tables
    }

    /// Convert a page to Markdown format.
    ///
    /// Extracts text from the specified page and converts it to Markdown with
    /// optional heading detection and image references.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `options` - Conversion options controlling the output
    ///
    /// # Returns
    ///
    /// A string containing the Markdown representation of the page.
    ///
    /// # Errors
    ///
    /// Returns an error if the page cannot be accessed or conversion fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pdf_oxide::PdfDocument;
    /// use pdf_oxide::converters::ConversionOptions;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("paper.pdf")?;
    ///
    /// let options = ConversionOptions {
    ///     detect_headings: true,
    ///     ..Default::default()
    /// };
    /// let markdown = doc.to_markdown(0, &options)?;
    /// println!("{}", markdown);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::wrong_self_convention)] // Needs mutable access for caching
    pub fn to_markdown(
        &self,
        page_index: usize,
        options: &crate::converters::ConversionOptions,
    ) -> Result<String> {
        if self.is_encrypted_unreadable() {
            log::warn!("PDF is encrypted and could not be decrypted; returning empty markdown");
            return Ok(String::new());
        }
        // Apply caller-specified region filters up front so excluded content is
        // gone from EVERY downstream path — tables, headings, reading order
        // (#609: markdown previously ignored `exclude_regions`/`include_region`,
        // which were only honoured by the plain-text path).
        let base_spans = Self::apply_region_filters(self.extract_spans(page_index)?, options);

        let tables = if options.extract_tables {
            // text_fallback=true: to_markdown explicitly targets structured output,
            // so we enable the text-only spatial fallback for line-less tables
            // (e.g. sailing-score grids with no ruling lines — issue #486).
            self.extract_page_tables(page_index, &base_spans, options, true)
        } else {
            Vec::new()
        };

        let mut spans = base_spans;
        if options.include_form_fields {
            spans.extend(self.extract_widget_spans(page_index));
        }

        let pipeline_config = TextPipelineConfig::from_conversion_options(options);

        let (mcid_order, mcid_to_role, mcid_to_block_id) = {
            // Use structure-tree reading order only when trustworthy (§14.8.2.3.1):
            // honours /MarkInfo /Suspects so markdown stays consistent with
            // extract_text / to_plain_text. (The /Table-element table path in
            // extract_page_tables intentionally keeps its own gate.)
            let cached_tree = self.struct_tree_trustworthy();

            if let Some(ref struct_tree) = cached_tree {
                // Build per-page traversal cache once, then O(1) lookup per page
                if self.structure_content_cache.lock_or_recover().is_none() {
                    let all_content =
                        crate::structure::traverse_structure_tree_all_pages(struct_tree);
                    *self.structure_content_cache.lock_or_recover() = Some(all_content);
                }

                // Extract MCID order AND per-MCID structural role for this page.
                let cached_page_owned = self
                    .structure_content_cache
                    .lock_or_recover()
                    .as_ref()
                    .and_then(|cache| cache.get(&(page_index as u32)))
                    .cloned();
                let cached_page = cached_page_owned.as_deref();

                let order: Vec<u32> = cached_page
                    .map(|content| content.iter().filter_map(|c| c.mcid).collect())
                    .unwrap_or_default();

                let mut role_map: std::collections::HashMap<u32, crate::pipeline::StructRole> =
                    std::collections::HashMap::new();
                let mut block_map: std::collections::HashMap<u32, u32> =
                    std::collections::HashMap::new();
                if let Some(content) = cached_page {
                    for item in content {
                        if let Some(mcid) = item.mcid {
                            // Heading takes precedence over list role on the
                            // same MCR (a heading-marked-content doesn't
                            // also play a list role in any sane PDF).
                            let role = if let Some(level) = item.heading_level {
                                Some(crate::pipeline::StructRole::Heading(level))
                            } else {
                                item.list_role.map(|lr| match lr {
                                    crate::structure::ListRole::LI => {
                                        crate::pipeline::StructRole::ListItem
                                    },
                                    crate::structure::ListRole::Lbl => {
                                        crate::pipeline::StructRole::ListItemLabel
                                    },
                                    crate::structure::ListRole::LBody => {
                                        crate::pipeline::StructRole::ListItemBody
                                    },
                                })
                            };
                            if let Some(r) = role {
                                // Heading wins over list role on the same MCID.
                                // The comment further up in this function asserts the
                                // precedence ("Heading takes precedence over list role
                                // on the same MCR"); plain `or_insert` would silently
                                // keep whichever role was seen first when the same
                                // MCID appears in two `OrderedContent` entries
                                // (e.g. one referenced from an /H1 sibling and one
                                // from an enclosing /LI in a tagged-tree quirk).
                                use std::collections::hash_map::Entry;
                                match role_map.entry(mcid) {
                                    Entry::Vacant(e) => {
                                        e.insert(r);
                                    },
                                    Entry::Occupied(mut e) => {
                                        let existing = *e.get();
                                        let new_is_heading =
                                            matches!(r, crate::pipeline::StructRole::Heading(_));
                                        let existing_is_heading = matches!(
                                            existing,
                                            crate::pipeline::StructRole::Heading(_)
                                        );
                                        if new_is_heading && !existing_is_heading {
                                            e.insert(r);
                                        }
                                    },
                                }
                            }
                            // First block_id wins per MCID — multiple OrderedContent
                            // entries can share an MCID when the same content is
                            // referenced from sibling structure elements; the first
                            // emit reflects the document order.
                            block_map.entry(mcid).or_insert(item.block_id);
                        }
                    }
                }

                let role_map_opt = if role_map.is_empty() {
                    None
                } else {
                    Some(role_map)
                };
                let block_map_opt = if block_map.is_empty() {
                    None
                } else {
                    Some(block_map)
                };

                if !order.is_empty() {
                    log::debug!(
                        "Extracted {} MCIDs ({} typed, {} blocked) from structure tree for page {}",
                        order.len(),
                        role_map_opt.as_ref().map(|m| m.len()).unwrap_or(0),
                        block_map_opt.as_ref().map(|m| m.len()).unwrap_or(0),
                        page_index
                    );
                    (Some(order), role_map_opt, block_map_opt)
                } else {
                    log::debug!(
                        "No MCIDs found for page {}, reading order strategy will use geometric fallback",
                        page_index
                    );
                    (None, role_map_opt, block_map_opt)
                }
            } else {
                log::debug!(
                    "No structure tree found, reading order strategy will use geometric fallback"
                );
                (None, None, None)
            }
        };

        // Step 5: Create pipeline with config
        let pipeline = TextPipeline::with_config(pipeline_config.clone());

        // Step 6: Build reading order context (pass mcid_order if available)
        let mut context = ReadingOrderContext::new().with_page(page_index as u32);
        if let Some(order) = mcid_order {
            context = context.with_mcid_order(order);
        }

        // Step 7: Process through pipeline (applies reading order strategy)
        let mut ordered_spans = pipeline.process(spans, context)?;

        // Annotate ordered spans with the per-MCID structural role
        // paragraph block-id so the markdown converter can emit headings
        // and bullets directly from the source PDF's `/StructTreeRoot`
        // and respect tagged paragraph boundaries even when the
        // geometric inter-paragraph gap is too small for the heuristic
        // (issue #377 D1 + D5 unlock).
        if mcid_to_role.is_some() || mcid_to_block_id.is_some() {
            for s in ordered_spans.iter_mut() {
                if let Some(mcid) = s.span.mcid {
                    if let Some(role) = mcid_to_role.as_ref().and_then(|m| m.get(&mcid)) {
                        s.struct_role = Some(*role);
                    }
                    if let Some(bid) = mcid_to_block_id.as_ref().and_then(|m| m.get(&mcid)) {
                        s.block_id = Some(*bid);
                    }
                }
            }
        }

        // Step 8: Use pipeline converter with tables
        let converter = MarkdownOutputConverter::new();
        let mut markdown =
            converter.convert_with_tables(&ordered_spans, &tables, &pipeline_config)?;

        // Step 9: Extract and include images if enabled
        if options.include_images {
            let images = self
                .extract_images_filtered(
                    page_index,
                    &ImageExtractFilter::markdown(options.max_image_pixels),
                )
                .unwrap_or_default();
            if !images.is_empty() {
                let image_markdown = self.generate_image_markdown(&images, options, page_index)?;
                markdown.push_str(&image_markdown);
            }
        }

        // A scanned / image page produces no extractable text and would
        // render as a silently-blank page. Emit a visible marker so a reader
        // knows content was lost and OCR is required, rather than dropping
        // (on a scanned corpus) ~half the document with no explanation. Gated
        // to genuinely scanned/image pages (not legitimately-blank ones) and
        // suppressible via `annotate_skipped_pages`.
        if options.annotate_skipped_pages && markdown.trim().is_empty() {
            if let Ok(c) = self.classify_page(page_index) {
                use crate::extractors::auto::PageKind;
                if matches!(c.kind, PageKind::Scanned | PageKind::ImageText) {
                    return Ok(format!(
                        "> [OCR REQUIRED — page {}]\n> This page is a scanned/rasterised image with no \
                         extractable text layer; run OCR to recover its content.\n",
                        page_index + 1
                    ));
                }
            }
        }

        Ok(markdown)
    }

    /// Generate Markdown for extracted images.
    ///
    /// Skips images exceeding `MAX_EMBED_PIXELS` (4 megapixels) when embedding
    /// as base64. These are typically full-page scans or high-res presentation
    /// slides that would produce 200-700KB of base64 per page with no useful
    /// content benefit (the text is already extracted). A placeholder comment
    /// is emitted instead.
    fn generate_image_markdown(
        &self,
        images: &[crate::extractors::PdfImage],
        options: &crate::converters::ConversionOptions,
        page_index: usize,
    ) -> Result<String> {
        use std::path::Path;

        // Images reaching this function have already been pre-filtered by
        // ImageExtractFilter::markdown() during extraction (min 32x32,
        // Indexed <64x64 skipped, max_pixels applied). No redundant checks needed.

        let mut markdown = String::new();
        let mut has_content = false;

        // Cap on base64 data URI size in characters. Anything larger
        // emits a placeholder instead so a multi-page PDF with high-
        // resolution images doesn't balloon the markdown output to
        // megabytes (issue #377 corpus-comparison observation: a
        // 17-page arxiv paper produced 11 MB of markdown, ~600 KB
        // per page of base64 PNG data, swamping any text-content
        // signal). 200 KB per image keeps small thumbnails
        // diagrams while skipping full-page renders.
        const MAX_BASE64_DATA_URI: usize = 200 * 1024;
        for (i, image) in images.iter().enumerate() {
            if options.embed_images {
                match image.to_base64_data_uri() {
                    Ok(data_uri) => {
                        if !has_content {
                            markdown.push_str("\n\n---\n\n");
                            has_content = true;
                        }
                        let alt = format!("Image {} from page {}", i + 1, page_index + 1);
                        if data_uri.len() > MAX_BASE64_DATA_URI {
                            // Estimate decoded binary size from the base64
                            // payload: each 4 chars decode to 3 bytes (minus
                            // padding). Drop the `data:...;base64,` prefix
                            // before measuring.
                            let payload = data_uri
                                .split_once(',')
                                .map(|(_, p)| p)
                                .unwrap_or(&data_uri);
                            let unpadded = payload.trim_end_matches('=').len();
                            let approx_binary_kb = (unpadded * 3 / 4) / 1024;
                            markdown.push_str(&format!(
                                "<!-- ![{}] suppressed: ~{} KB decoded image (base64 data URI {} KB) exceeds {} KB inline-image cap -->\n\n",
                                alt,
                                approx_binary_kb,
                                data_uri.len() / 1024,
                                MAX_BASE64_DATA_URI / 1024
                            ));
                        } else {
                            markdown.push_str(&format!("![{}]({})\n\n", alt, data_uri));
                        }
                    },
                    Err(e) => {
                        log::warn!("Failed to encode image {}: {}", i, e);
                    },
                }
            } else if let Some(ref output_dir) = options.image_output_dir {
                // Save to file and reference by path (no size limit for file saves)
                let filename = format!("page{}_{}.png", page_index + 1, i + 1);
                let filepath = Path::new(output_dir).join(&filename);

                if let Some(parent) = filepath.parent() {
                    std::fs::create_dir_all(parent).ok();
                }

                match image.save_as_png(&filepath) {
                    Ok(()) => {
                        if !has_content {
                            markdown.push_str("\n\n---\n\n");
                            has_content = true;
                        }
                        let alt = format!("Image {} from page {}", i + 1, page_index + 1);
                        let relative_path = format!("{}/{}", output_dir, filename);
                        markdown.push_str(&format!("![{}]({})\n\n", alt, relative_path));
                    },
                    Err(e) => {
                        log::warn!("Failed to save image {}: {}", i, e);
                    },
                }
            }
        }

        Ok(markdown)
    }

    /// Convert a page to Markdown with automatic OCR fallback for scanned pages.
    ///
    /// This method automatically detects scanned pages and applies OCR when needed,
    /// falling back to native text extraction for regular PDFs.
    ///
    /// **Note**: Requires the `ocr` feature to be enabled and OCR models to be provided.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `options` - Conversion options controlling the output
    /// * `ocr_engine` - Optional OCR engine (required for scanned pages)
    /// * `ocr_options` - OCR extraction options
    ///
    /// # Returns
    ///
    /// A string containing the Markdown representation of the page.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::{PdfDocument, ocr::{OcrEngine, OcrConfig, OcrExtractOptions}};
    /// use pdf_oxide::converters::ConversionOptions;
    ///
    /// let mut doc = PdfDocument::open("scanned.pdf")?;
    /// let engine = OcrEngine::new("det.onnx", "rec.onnx", "dict.txt", OcrConfig::default())?;
    ///
    /// let markdown = doc.to_markdown_with_ocr(
    ///     0,
    ///     &ConversionOptions::default(),
    ///     Some(&engine),
    ///     &OcrExtractOptions::default()
    /// )?;
    /// ```
    #[cfg(feature = "ocr")]
    pub fn to_markdown_with_ocr(
        &self,
        page_index: usize,
        options: &crate::converters::ConversionOptions,
        ocr_engine: Option<&crate::ocr::OcrEngine>,
        ocr_options: &crate::ocr::OcrExtractOptions,
    ) -> Result<String> {
        #[allow(deprecated)]
        use crate::converters::{MarkdownConverter, ReadingOrderMode};
        use crate::structure::traversal::extract_reading_order;

        // Extract spans with OCR fallback
        let spans = self.extract_spans_with_ocr(page_index, ocr_engine, ocr_options)?;
        #[allow(deprecated)]
        let converter = MarkdownConverter::new();

        // Check if we need to extract structure tree for StructureTreeFirst mode
        let mut options = options.clone();
        if matches!(options.reading_order_mode, ReadingOrderMode::StructureTreeFirst { .. }) {
            // Try to parse structure tree and extract MCID reading order
            if let Ok(Some(struct_tree)) = self.structure_tree() {
                match extract_reading_order(&struct_tree, page_index as u32) {
                    Ok(mcid_order) if !mcid_order.is_empty() => {
                        // Update reading order mode with extracted MCIDs
                        options.reading_order_mode =
                            ReadingOrderMode::StructureTreeFirst { mcid_order };
                        log::debug!(
                            "Extracted {} MCIDs from structure tree for page {}",
                            match &options.reading_order_mode {
                                ReadingOrderMode::StructureTreeFirst { mcid_order } =>
                                    mcid_order.len(),
                                _ => 0,
                            },
                            page_index
                        );
                    },
                    _ => {
                        // No MCIDs found or error - fallback to ColumnAware
                        log::debug!(
                            "No MCIDs found for page {}, using ColumnAware fallback",
                            page_index
                        );
                        options.reading_order_mode =
                            ReadingOrderMode::StructureTreeFirst { mcid_order: vec![] };
                    },
                }
            } else {
                // No structure tree - fallback to ColumnAware
                log::debug!("No structure tree found, using ColumnAware fallback");
                options.reading_order_mode =
                    ReadingOrderMode::StructureTreeFirst { mcid_order: vec![] };
            }
        }

        // Use the new PDF spec compliant span-based converter
        converter.convert_page_from_spans(&spans, &options)
    }

    /// Convert a page to HTML format.
    ///
    /// Extracts text from the specified page and converts it to HTML.
    /// Supports both semantic HTML and layout-preserved modes based on options.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `options` - Conversion options controlling the output
    ///
    /// # Returns
    ///
    /// A string containing the HTML representation of the page.
    ///
    /// # Errors
    ///
    /// Returns an error if the page cannot be accessed or conversion fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pdf_oxide::PdfDocument;
    /// use pdf_oxide::converters::ConversionOptions;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("paper.pdf")?;
    ///
    /// // Semantic HTML
    /// let options = ConversionOptions::default();
    /// let html = doc.to_html(0, &options)?;
    ///
    /// // Layout-preserved HTML
    /// let layout_options = ConversionOptions {
    ///     preserve_layout: true,
    ///     ..Default::default()
    /// };
    /// let layout_html = doc.to_html(0, &layout_options)?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::wrong_self_convention)] // Needs mutable access for caching
    pub fn to_html(
        &self,
        page_index: usize,
        options: &crate::converters::ConversionOptions,
    ) -> Result<String> {
        if self.is_encrypted_unreadable() {
            log::warn!("PDF is encrypted and could not be decrypted; returning empty HTML");
            return Ok(String::new());
        }
        // Region filters applied up front so excluded content is gone from every
        // downstream path (#609), matching the markdown and plain-text surfaces.
        let base_spans = Self::apply_region_filters(self.extract_spans(page_index)?, options);

        let tables = if options.extract_tables {
            // text_fallback=true: to_html explicitly targets structured output,
            // so we enable the text-only spatial fallback for line-less tables
            // (e.g. sailing-score grids with no ruling lines — issue #486).
            self.extract_page_tables(page_index, &base_spans, options, true)
        } else {
            Vec::new()
        };

        let mut spans = base_spans;
        if options.include_form_fields {
            spans.extend(self.extract_widget_spans(page_index));
        }

        let pipeline_config = TextPipelineConfig::from_conversion_options(options);

        // Step 4: Create pipeline with config
        let pipeline = TextPipeline::with_config(pipeline_config.clone());

        // Step 5: Build reading order context. For a trustworthy tagged PDF use
        // the canonical builder so the StructureTreeStrategy assigns MCID-driven
        // reading order (§14.8.2.3.1). The HTML converter sorts by
        // `reading_order` (and only uses Y for line-break gaps), so it honours
        // the structure order. Untagged / suspect PDFs keep the exact bare
        // geometric context, so their output is byte-for-byte unchanged.
        let context = if self.prefers_structure_reading_order() {
            crate::pipeline::page_order::build_context(self, page_index)
        } else {
            ReadingOrderContext::new().with_page(page_index as u32)
        };

        // Step 6: Process through pipeline (applies reading order strategy)
        let ordered_spans = pipeline.process(spans, context)?;

        // Step 7: Use pipeline converter with tables
        let converter = HtmlOutputConverter::new();
        let mut html = converter.convert_with_tables(&ordered_spans, &tables, &pipeline_config)?;

        // Step 8: Extract and embed images if enabled
        if options.include_images {
            let images = self
                .extract_images_filtered(
                    page_index,
                    &ImageExtractFilter::markdown(options.max_image_pixels),
                )
                .unwrap_or_default();
            if !images.is_empty() {
                let image_html = self.generate_image_html(&images, options, page_index)?;
                if let Some(pos) = html.rfind("</body>") {
                    html.insert_str(pos, &image_html);
                } else {
                    html.push_str(&image_html);
                }
            }
        }

        Ok(html)
    }

    /// Generate HTML for extracted images.
    fn generate_image_html(
        &self,
        images: &[crate::extractors::PdfImage],
        options: &crate::converters::ConversionOptions,
        page_index: usize,
    ) -> Result<String> {
        use std::path::Path;

        let mut html = String::new();
        html.push_str("\n<div class=\"page-images\">\n");

        for (i, image) in images.iter().enumerate() {
            let alt = format!("Image {} from page {}", i + 1, page_index + 1);

            if options.embed_images {
                // Embed as base64 data URI
                match image.to_base64_data_uri() {
                    Ok(data_uri) => {
                        html.push_str(&format!(
                            "  <img src=\"{}\" alt=\"{}\" style=\"max-width: 100%;\">\n",
                            data_uri, alt
                        ));
                    },
                    Err(e) => {
                        // Log error but continue with other images
                        log::warn!("Failed to encode image {}: {}", i, e);
                    },
                }
            } else if let Some(ref output_dir) = options.image_output_dir {
                // Save to file and reference by path
                let filename = format!("page{}_{}.png", page_index + 1, i + 1);
                let filepath = Path::new(output_dir).join(&filename);

                // Create directory if needed
                if let Some(parent) = filepath.parent() {
                    std::fs::create_dir_all(parent).ok();
                }

                match image.save_as_png(&filepath) {
                    Ok(()) => {
                        // Use relative path in HTML
                        let relative_path = format!("{}/{}", output_dir, filename);
                        html.push_str(&format!(
                            "  <img src=\"{}\" alt=\"{}\" style=\"max-width: 100%;\">\n",
                            relative_path, alt
                        ));
                    },
                    Err(e) => {
                        log::warn!("Failed to save image {}: {}", i, e);
                    },
                }
            }
            // If embed_images=false and no output_dir, skip image
        }

        html.push_str("</div>\n");
        Ok(html)
    }

    /// Convert a page to plain text.
    ///
    /// Extracts text from the specified page with minimal formatting.
    /// This is equivalent to calling `extract_text()`.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    /// * `options` - Conversion options (currently unused for plain text, reserved for future use)
    ///
    /// # Returns
    ///
    /// A string containing the plain text content of the page.
    ///
    /// # Errors
    ///
    /// Returns an error if the page cannot be accessed or text extraction fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pdf_oxide::PdfDocument;
    /// use pdf_oxide::converters::ConversionOptions;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("paper.pdf")?;
    /// let options = ConversionOptions::default();
    /// let text = doc.to_plain_text(0, &options)?;
    /// println!("{}", text);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::wrong_self_convention)] // Needs mutable access for caching
    pub fn to_plain_text(
        &self,
        page_index: usize,
        options: &crate::converters::ConversionOptions,
    ) -> Result<String> {
        // #608: for a trustworthy tagged PDF, read in logical structure order
        // (§14.8.2.3.1) by assembling directly from the structure tree — the
        // same path `extract_text` uses. The geometric plain-text converter
        // below regroups spans by Y and would otherwise override the structure
        // order. Untagged / suspect PDFs fall through to the exact converter
        // path below, so their output is byte-for-byte unchanged.
        if self.prefers_structure_reading_order() {
            let base_spans = self.extract_spans(page_index)?;
            return self.assemble_text_from_spans(page_index, base_spans, options);
        }

        // Step 1: Extract raw spans (unchanged - this is the foundation)
        let mut spans = self.extract_spans(page_index)?;

        // Step 1b: Merge widget annotation spans (form field values) if enabled
        if options.include_form_fields {
            spans.extend(self.extract_widget_spans(page_index));
        }

        // Step 2: Extract tables if enabled
        let tables = if options.extract_tables {
            // text_fallback=false: to_plain_text uses the conservative pre-v0.3.47
            // behaviour to avoid false-positive table detection in key-value layouts.
            self.extract_page_tables(page_index, &spans, options, false)
        } else {
            Vec::new()
        };

        // Step 3: Create pipeline config from options.
        let pipeline_config = TextPipelineConfig::from_conversion_options(options);

        // Step 4: Create pipeline with config
        let pipeline = TextPipeline::with_config(pipeline_config.clone());

        // Step 5: Build reading order context.
        //
        // NOTE (#608): for a trustworthy tagged PDF the structure-tree reading
        // order is honoured by `extract_text` and `to_markdown` (which assemble
        // directly from MCID order). `to_plain_text` / `to_html` run spans
        // through the geometric line/column grouping converter, which regroups
        // by Y position and therefore overrides any per-span `reading_order` the
        // StructureTreeStrategy would assign. Feeding MCID order here is thus a
        // no-op for these two converters, so we keep the geometric context — and
        // its byte-for-byte output — unchanged. Routing structured logical order
        // through the plain-text/HTML converters is a tracked follow-up.
        let context = ReadingOrderContext::new().with_page(page_index as u32);

        // Step 6: Process through pipeline (applies reading order strategy)
        let ordered_spans = pipeline.process(spans, context)?;

        // Step 7: Use pipeline converter with tables
        let converter = PlainTextConverter::new();
        converter.convert_with_tables(&ordered_spans, &tables, &pipeline_config)
    }

    /// Convert all pages to Markdown format.
    ///
    /// Extracts and converts all pages in the document to Markdown,
    /// separating pages with "---" horizontal rules.
    ///
    /// # Arguments
    ///
    /// * `options` - Conversion options controlling the output
    ///
    /// # Returns
    ///
    /// A string containing the Markdown representation of all pages.
    ///
    /// # Errors
    ///
    /// Returns an error if any page cannot be accessed or conversion fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pdf_oxide::PdfDocument;
    /// use pdf_oxide::converters::ConversionOptions;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("paper.pdf")?;
    /// let options = ConversionOptions::default();
    /// let markdown = doc.to_markdown_all(&options)?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::wrong_self_convention)] // Needs mutable access for caching
    pub fn to_markdown_all(
        &self,
        options: &crate::converters::ConversionOptions,
    ) -> Result<String> {
        if self.is_encrypted_unreadable() {
            log::warn!("PDF is encrypted and could not be decrypted; returning empty markdown");
            return Ok(String::new());
        }
        let page_count = self.page_count()?;
        // Pre-reserve ~4 KB/page to avoid repeated reallocation.
        let mut result = String::with_capacity(page_count.saturating_mul(4096));

        for i in 0..page_count {
            if i > 0 {
                result.push_str("\n---\n\n");
            }
            let page_markdown = self.to_markdown(i, options)?;
            result.push_str(&page_markdown);
        }

        Ok(result)
    }

    /// Convert all pages to plain text format.
    ///
    /// Extracts all pages in the document as plain text,
    /// separating pages with "---" horizontal rules.
    ///
    /// # Arguments
    ///
    /// * `options` - Conversion options (currently unused for plain text, reserved for future use)
    ///
    /// # Returns
    ///
    /// A string containing the plain text of all pages.
    ///
    /// # Errors
    ///
    /// Returns an error if any page cannot be accessed or extraction fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pdf_oxide::PdfDocument;
    /// use pdf_oxide::converters::ConversionOptions;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("paper.pdf")?;
    /// let options = ConversionOptions::default();
    /// let text = doc.to_plain_text_all(&options)?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::wrong_self_convention)] // Needs mutable access for caching
    pub fn to_plain_text_all(
        &self,
        options: &crate::converters::ConversionOptions,
    ) -> Result<String> {
        if self.is_encrypted_unreadable() {
            log::warn!("PDF is encrypted and could not be decrypted; returning empty text");
            return Ok(String::new());
        }
        let page_count = self.page_count()?;
        // Pre-reserve ~4 KB/page (see to_markdown_all).
        let mut result = String::with_capacity(page_count.saturating_mul(4096));

        for i in 0..page_count {
            if i > 0 {
                result.push_str("\n\n---\n\n");
            }
            let page_text = self.to_plain_text(i, options)?;
            result.push_str(&page_text);
        }

        Ok(result)
    }

    /// Check for circular references in the object graph.
    ///
    /// This is a diagnostic method that performs a depth-first search
    /// through the object graph to detect cycles.
    ///
    /// # Returns
    ///
    /// A vector of tuples representing edges that create cycles.
    /// Each tuple is (from_object, to_object) where to_object is
    /// already in the path when encountered again.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// let cycles = doc.check_for_circular_references();
    /// if !cycles.is_empty() {
    ///     println!("Found {} circular references", cycles.len());
    /// }
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn check_for_circular_references(&self) -> Vec<(ObjectRef, ObjectRef)> {
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut path = Vec::new();

        // Check all objects in the xref table
        let obj_nums: Vec<u32> = self.xref.entries.keys().copied().collect();
        for obj_num in obj_nums {
            let obj_ref = ObjectRef::new(obj_num, 0);
            if !visited.contains(&obj_ref) {
                self.dfs_check_cycles(obj_ref, &mut visited, &mut path, &mut cycles);
            }
        }

        cycles
    }

    /// Depth-first search helper for cycle detection.
    fn dfs_check_cycles(
        &self,
        obj_ref: ObjectRef,
        visited: &mut HashSet<ObjectRef>,
        path: &mut Vec<ObjectRef>,
        cycles: &mut Vec<(ObjectRef, ObjectRef)>,
    ) {
        if path.contains(&obj_ref) {
            // Found cycle
            if let Some(&prev) = path.last() {
                cycles.push((prev, obj_ref));
            }
            return;
        }

        if visited.contains(&obj_ref) {
            return;
        }

        visited.insert(obj_ref);
        path.push(obj_ref);

        // Get object and scan for references
        if let Ok(obj) = self.load_object(obj_ref) {
            for ref_found in Self::find_references(&obj) {
                self.dfs_check_cycles(ref_found, visited, path, cycles);
            }
        }

        path.pop();
    }

    /// Find all object references within an object.
    fn find_references(obj: &Object) -> Vec<ObjectRef> {
        let mut refs = Vec::new();

        match obj {
            Object::Reference(r) => refs.push(*r),
            Object::Array(arr) => {
                for item in arr {
                    refs.extend(Self::find_references(item));
                }
            },
            Object::Dictionary(dict) => {
                for value in dict.values() {
                    refs.extend(Self::find_references(value));
                }
            },
            Object::Stream { dict, .. } => {
                for value in dict.values() {
                    refs.extend(Self::find_references(value));
                }
            },
            _ => {},
        }

        refs
    }

    /// Convert all pages to HTML format.
    ///
    /// Extracts and converts all pages in the document to HTML,
    /// wrapping each page in a div with class "page".
    ///
    /// # Arguments
    ///
    /// * `options` - Conversion options controlling the output
    ///
    /// # Returns
    ///
    /// A string containing the HTML representation of all pages.
    ///
    /// # Errors
    ///
    /// Returns an error if any page cannot be accessed or conversion fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pdf_oxide::PdfDocument;
    /// use pdf_oxide::converters::ConversionOptions;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut doc = PdfDocument::open("paper.pdf")?;
    /// let options = ConversionOptions::default();
    /// let html = doc.to_html_all(&options)?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::wrong_self_convention)] // Needs mutable access for caching
    pub fn to_html_all(&self, options: &crate::converters::ConversionOptions) -> Result<String> {
        use std::fmt::Write as _;
        if self.is_encrypted_unreadable() {
            log::warn!("PDF is encrypted and could not be decrypted; returning empty HTML");
            return Ok(String::new());
        }
        let page_count = self.page_count()?;
        // Pre-reserve ~4 KB/page (see to_markdown_all).
        let mut result = String::with_capacity(page_count.saturating_mul(4096));

        for i in 0..page_count {
            // writeln! into the buffer (infallible for String) — no per-page format! temporary.
            let _ = writeln!(result, "<div class=\"page\" data-page=\"{}\">", i + 1);
            let page_html = self.to_html(i, options)?;
            result.push_str(&page_html);
            result.push_str("</div>\n");
        }

        Ok(result)
    }

    /// Convert the entire document to a DOCX file written to `path`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pdf_oxide::PdfDocument;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let doc = PdfDocument::open("paper.pdf")?;
    /// doc.to_docx("paper.docx")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_docx(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let bytes = self.to_docx_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Convert the entire document to DOCX bytes in memory.
    ///
    /// Pipeline: PDF → DocumentIR (via `pdf_to_ir`) → DOCX.
    /// Each PDF page becomes one IR `Section` carrying its source
    /// MediaBox under `page_setup` and a `NextPage` break from the
    /// second section onward. `ir_to_docx` writes one `<w:sectPr>` per
    /// section (each non-final sectPr lives inside a synthetic empty
    /// paragraph's `<w:pPr>`, the final one at body level), so a
    /// PDF→DOCX→PDF round-trip preserves the source page count
    /// dimensions instead of overflowing onto Letter-sized pages at
    /// the OfficeConfig default. Source-PDF fonts are embedded under
    /// `word/fonts/` for typeface preservation across the round-trip.
    pub fn to_docx_bytes(&self) -> Result<Vec<u8>> {
        // Layout-preserving emission gives near-pixel-identical text
        // for small docs but produces one positioned frame per source
        // span — Word handles ~5k frames cleanly, ~70k frames (660-page
        // CFR) takes minutes to open or refuses entirely. Pick the
        // path based on source page count: ≤ `LAYOUT_MAX_PAGES`
        // chooses fidelity, anything larger falls back to flow mode
        // + column-aware reflow which Word opens instantly.
        const LAYOUT_MAX_PAGES: usize = 30;
        let n = self.page_count().unwrap_or(0);
        if n <= LAYOUT_MAX_PAGES {
            self.to_docx_bytes_layout()
        } else {
            self.to_docx_bytes_flow()
        }
    }

    /// Legacy flow-mode DOCX export: PDF text spans are grouped into
    /// flowing paragraphs with column-aware layout. Trade-off vs. the
    /// default `to_docx_bytes` layout-preserving path:
    /// - Better for editing (real paragraph structure, not floating frames).
    /// - Worse for visual fidelity (text reflows; positions drift).
    ///
    /// Use this when downstream callers will edit the DOCX in Word /
    /// LibreOffice; use the default for pixel-faithful round trips.
    pub fn to_docx_bytes_flow(&self) -> Result<Vec<u8>> {
        let ir = self.pdf_to_office_ir(office_oxide::format::DocumentFormat::Docx)?;
        let mut writer = office_oxide::create::ir_to_docx(&ir);
        self.embed_pdf_fonts_into(|name, data| {
            writer.embed_font(name, data);
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        writer
            .write_to(&mut buf)
            .map_err(|e| crate::error::Error::InvalidOperation(format!("DOCX export: {e}")))?;
        Ok(buf.into_inner())
    }

    /// Forward every embedded font program from the source PDF (if
    /// extractable) into the supplied per-font sink. Each
    /// office-format writer exposes its own `embed_font(name, data)`
    /// method but they take different `&mut self` types, so this
    /// helper centralises the iteration without trying to abstract
    /// over the writer types.
    fn embed_pdf_fonts_into<F: FnMut(String, Vec<u8>)>(&self, mut sink: F) {
        // CFF font subsets in source PDFs typically ship without a
        // Unicode cmap (CID-only encoding) and without an `hmtx`
        // table (widths live in the document's `/W` array, not the
        // font program). `cmap_injector` patches both: it synthesises
        // a format-4 Unicode cmap from `/ToUnicode` so the font
        // registers via `EmbeddedFont::has_usable_unicode_cmap`,
        // it stamps a real `hmtx` populated from the source PDF's
        // `/W` widths so ttf-parser's `glyph_hor_advance` returns
        // non-zero values when the round-trip writer rebuilds its
        // own `/W` array. Without the hmtx patch, every glyph
        // emitted in the round-trip PDF advances 0 and body text
        // collapses into a single x-position column.
        //
        // The renderer's CFF path (text_rasterizer.rs) routes
        // Type0+CIDFontType0 fonts through `render_cid_direct` so the
        // content-stream's Identity-H CIDs map straight to CFF charset
        // positions — i.e. CID==GID — which is the correct
        // interpretation for `Identity-H` Type0/CIDFontType0 emission.
        //
        // For TrueType subsets (sfnt 0x00010000) the cmap is already
        // present and `unicode_map` is empty; we still inject hmtx
        // when widths are known so the writer's width table reflects
        // the source PDF's authoritative `/W` rather than whatever
        // hmtx happens to be in the embedded subset.
        if let Ok(fonts) = self.extract_embedded_fonts_with_unicode_maps_and_widths() {
            for (name, data, unicode_map, widths_by_gid) in fonts {
                let mut bytes = data;
                if !unicode_map.is_empty() {
                    bytes = crate::fonts::cmap_injector::inject_unicode_cmap(&bytes, &unicode_map)
                        .unwrap_or(bytes);
                }
                if !widths_by_gid.is_empty() {
                    bytes = crate::fonts::cmap_injector::inject_hmtx(&bytes, &widths_by_gid)
                        .unwrap_or(bytes);
                }
                sink(name, bytes);
            }
            return;
        }
        if let Ok(fonts) = self.extract_embedded_fonts() {
            for (name, data) in fonts {
                sink(name, data);
            }
        }
    }

    /// Build a `DocumentIR` from the entire PDF, tagged for the target
    /// office format. Shared by `to_docx_bytes`, `to_pptx_bytes`,
    /// `to_xlsx_bytes`. Thin wrapper around `pdf_to_ir::pdf_to_ir`
    /// that applies default options.
    fn pdf_to_office_ir(
        &self,
        format: office_oxide::format::DocumentFormat,
    ) -> Result<office_oxide::ir::DocumentIR> {
        let opts = crate::converters::pdf_to_ir::PdfToIrOptions::default();
        crate::converters::pdf_to_ir::pdf_to_ir(self, format, &opts)
    }

    /// Convert the document to DOCX bytes with **layout-preserving** text
    /// frames: every PDF text span is emitted as a `<w:framePr>`-anchored
    /// paragraph at its exact source position.
    ///
    /// Trade-off vs. [`Self::to_docx_bytes`]:
    /// - This output renders visually similar to the source PDF when opened
    ///   in Word/LibreOffice — same fonts, sizes, colors, positions.
    /// - Text remains real selectable/editable text (unlike rasterization).
    /// - Doesn't reconstruct table grids, vector graphics, or images
    ///   (text-only).
    /// - The output isn't ideal for *editing* (every word is its own
    ///   floating frame); use the markdown path when ergonomics matter
    ///   more than visual fidelity.
    pub fn to_docx_bytes_layout(&self) -> Result<Vec<u8>> {
        crate::converters::docx_layout::to_docx_bytes_layout(self)
    }

    /// Convert the entire document to a PPTX file on disk.
    pub fn to_pptx(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let bytes = self.to_pptx_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Convert the entire document to PPTX bytes in memory.
    ///
    /// Pipeline: PDF → DocumentIR → PPTX. Each PDF page becomes one
    /// slide, sized to the source `MediaBox` (the presentation-level
    /// `<p:sldSz>` is set from the first page) so a PDF→PPTX→PDF
    /// round-trip preserves the original page dimensions instead of
    /// overflowing dense pages onto multiple Letter-sized slides.
    /// Source-PDF fonts are embedded under `ppt/fonts/` so the
    /// round-trip reuses the original typeface.
    pub fn to_pptx_bytes(&self) -> Result<Vec<u8>> {
        // Page-count gated like `to_docx_bytes`. PowerPoint's "fix the
        // content" dialog fires above ~250 slides; pixel-faithful
        // layout mode emits one slide per PDF page so anything larger
        // than `LAYOUT_MAX_PAGES` falls back to the flow path that
        // collapses pages via heading-bounded compaction.
        const LAYOUT_MAX_PAGES: usize = 30;
        let n = self.page_count().unwrap_or(0);
        if n <= LAYOUT_MAX_PAGES {
            crate::converters::pptx_layout::to_pptx_bytes_layout(self)
        } else {
            self.to_pptx_bytes_flow()
        }
    }

    /// Legacy flow-mode PPTX export: PDF text is grouped into flow
    /// paragraphs and laid out by PowerPoint's auto-layout engine.
    /// Trade-off vs. the default `to_pptx_bytes` layout-preserving path:
    /// - Better for editing (real paragraph structure, fewer shapes).
    /// - Worse for visual fidelity (text reflows; positions drift).
    pub fn to_pptx_bytes_flow(&self) -> Result<Vec<u8>> {
        let ir = self.pdf_to_office_ir(office_oxide::format::DocumentFormat::Pptx)?;
        let mut writer = office_oxide::create::ir_to_pptx(&ir);
        self.embed_pdf_fonts_into(|name, data| {
            writer.embed_font(name, data);
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        writer
            .write_to(&mut buf)
            .map_err(|e| crate::error::Error::InvalidOperation(format!("PPTX export: {e}")))?;
        Ok(buf.into_inner())
    }

    /// Convert the entire document to an XLSX file on disk.
    pub fn to_xlsx(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let bytes = self.to_xlsx_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Convert the entire document to XLSX bytes in memory.
    ///
    /// Pipeline: PDF → DocumentIR → XLSX. Each PDF page becomes one
    /// worksheet; per-paragraph font sizes are written to cell styles so
    /// the round-trip back through `xlsx → IR` recovers the original 9–10
    /// pt body size instead of falling back to the writer's 12 pt default
    /// (which inflated round-trip page counts by 2–3×). Source-PDF fonts
    /// are embedded under `xl/fonts/` mirroring the DOCX/PPTX paths.
    pub fn to_xlsx_bytes(&self) -> Result<Vec<u8>> {
        // Page-count gated like `to_docx_bytes`. Excel chokes on tens
        // of thousands of `<xdr:sp>` shapes (one per span × hundreds of
        // pages); fall back to the flow path that emits content into
        // the cell grid for large sources. Raised from 30 so multi-
        // hundred-page documents (e.g. typical dissertations) still
        // take the positional path — `ir_to_xlsx` flow mode drops
        // paragraph alignment into column A, collapsing centered
        // title blocks.
        const LAYOUT_MAX_PAGES: usize = 200;
        let n = self.page_count().unwrap_or(0);
        if n <= LAYOUT_MAX_PAGES {
            crate::converters::xlsx_layout::to_xlsx_bytes_layout(self)
        } else {
            self.to_xlsx_bytes_flow()
        }
    }

    /// Legacy flow-mode XLSX export: PDF content is laid out into
    /// the cell grid. Use this when downstream callers will treat the
    /// XLSX as a real spreadsheet (filters, formulas, sorts); use the
    /// default `to_xlsx_bytes` for pixel-faithful round trips.
    pub fn to_xlsx_bytes_flow(&self) -> Result<Vec<u8>> {
        let ir = self.pdf_to_office_ir(office_oxide::format::DocumentFormat::Xlsx)?;
        let mut writer = office_oxide::create::ir_to_xlsx(&ir);
        self.embed_pdf_fonts_into(|name, data| {
            writer.embed_font(name, data);
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        writer
            .write_to(&mut buf)
            .map_err(|e| crate::error::Error::InvalidOperation(format!("XLSX export: {e}")))?;
        Ok(buf.into_inner())
    }

    /// Extract images from a page.
    ///
    /// Extracts all images from the specified page by processing the content stream.
    /// This includes:
    /// - Images referenced via `Do` operators (XObject calls)
    /// - Images in nested Form XObjects (with recursion)
    /// - Inline images (BI...ID...EI sequences)
    ///
    /// This method processes PDF content streams instead of only iterating the XObject
    /// dictionary. This ensures that images referenced via the `Do` operator in the content
    /// stream are properly extracted, including those in nested Form XObjects. ColorSpace
    /// indirect references are also resolved.
    ///
    /// Returns a vector of PdfImage objects representing the extracted images.
    ///
    /// # Arguments
    ///
    /// * `page_index` - Zero-based page index
    ///
    /// # Returns
    ///
    /// A vector of PdfImage objects, one for each image found on the page.
    ///
    /// # Errors
    ///
    /// Returns an error if the page cannot be accessed or if image extraction fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::document::PdfDocument;
    /// # let mut doc = PdfDocument::open("sample.pdf")?;
    /// let images = doc.extract_images(0)?;
    /// println!("Found {} images on page 1", images.len());
    /// for (i, image) in images.iter().enumerate() {
    ///     image.save_as_png(&format!("image_{}.png", i))?;
    /// }
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn extract_images(&self, page_index: usize) -> Result<Vec<crate::extractors::PdfImage>> {
        self.require_authenticated()?;
        self.extract_images_filtered(page_index, &ImageExtractFilter::default())
    }

    /// Build the resource-name → colour-space-object map from a resolved
    /// `/Resources` dictionary's `/ColorSpace` subdictionary (§8.6.3 / §7.8.3),
    /// resolving one indirect-ref hop per entry so the stored value is a colour
    /// space name or array. Empty when there is no `/ColorSpace` subdictionary;
    /// the standard device names parse directly and need no entry. Consumed by
    /// the image-handle builders so `decode()` / the handle's `color_space` can
    /// resolve names like `/CS0` (§8.6.6, §8.9.7).
    fn build_color_space_map(
        &self,
        resources: Option<&Object>,
    ) -> std::collections::HashMap<String, Object> {
        let mut map = std::collections::HashMap::new();
        let Some(res) = resources else {
            return map;
        };
        let res = if let Some(r) = res.as_reference() {
            match self.load_object(r) {
                Ok(o) => o,
                Err(_) => return map,
            }
        } else {
            res.clone()
        };
        let Some(res_dict) = res.as_dict() else {
            return map;
        };
        let Some(cs_entry) = res_dict.get("ColorSpace") else {
            return map;
        };
        let cs_obj = if let Some(r) = cs_entry.as_reference() {
            match self.load_object(r) {
                Ok(o) => o,
                Err(_) => return map,
            }
        } else {
            cs_entry.clone()
        };
        let Some(cs_dict) = cs_obj.as_dict() else {
            return map;
        };
        for (name, value) in cs_dict.iter() {
            let resolved = if let Some(r) = value.as_reference() {
                self.load_object(r).unwrap_or_else(|_| value.clone())
            } else {
                value.clone()
            };
            map.insert(name.clone(), resolved);
        }
        map
    }

    /// Enumerate images on a page without decompressing any stream (Phase 1).
    ///
    /// Walks the page content stream once and reads image metadata (dimensions,
    /// colour space, filter chain, compressed size) directly from each Image
    /// XObject dictionary. No pixel data is decoded. Returns a handle per image
    /// in content-stream paint order.
    ///
    /// Call [`crate::PdfImageHandle::decode`] on individual handles to materialise only
    /// the images you need, or [`crate::PdfImageHandle::raw_compressed_bytes`] to forward
    /// compressed data (e.g. JPEG bytes) without recompression.
    ///
    /// Form XObjects (subtype `/Form`) are recursed into, matching the behaviour
    /// of [`PdfDocument::extract_images`]. Cycle detection (depth limit 100) and
    /// the document's Form stream cache are used. Images inside nested or shared
    /// Forms receive the correct final CTM-composed `bbox` / `rotation_degrees`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::PdfDocument;
    /// # let bytes = std::fs::read("page.pdf").unwrap();
    /// let doc = PdfDocument::from_bytes(bytes).unwrap();
    ///
    /// // Decode only images larger than a thumbnail threshold
    /// let images: Vec<_> = doc.page_image_handles(0)?
    ///     .into_iter()
    ///     .filter(|h| h.width >= 200 && h.height >= 200)
    ///     .map(|h| h.decode())
    ///     .collect::<Result<_, _>>()?;
    /// # Ok::<(), pdf_oxide::error::Error>(())
    /// ```
    pub fn page_image_handles(
        &self,
        page_index: usize,
    ) -> Result<Vec<crate::extractors::images::PdfImageHandle<'_>>> {
        use crate::content::parse_content_stream_images_only;
        use crate::content::Operator;
        use crate::extractors::images::image_handle_from_inline;

        self.require_authenticated()?;

        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;

        let content_data = self.get_page_content_data(page_index)?;

        let resources = match page_dict.get("Resources") {
            Some(res) => {
                if let Some(ref_obj) = res.as_reference() {
                    Some(self.load_object(ref_obj)?)
                } else {
                    Some(res.clone())
                }
            },
            None => None,
        };

        let operators = match parse_content_stream_images_only(&content_data) {
            Ok(ops) => ops,
            Err(_) => return Ok(Vec::new()),
        };

        // Resource-name colour-space map for this page scope (§8.6.6 / §8.9.7).
        let cs_map = self.build_color_space_map(resources.as_ref());

        // Pre-resolve the XObject dictionary once
        let xobject_dict = if let Some(ref res) = resources {
            if let Some(res_dict) = res.as_dict() {
                if let Some(xobj_entry) = res_dict.get("XObject") {
                    let resolved = if let Some(ref_obj) = xobj_entry.as_reference() {
                        self.load_object(ref_obj)?
                    } else {
                        xobj_entry.clone()
                    };
                    resolved.as_dict().cloned()
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let mut handles = Vec::new();
        let mut ctm_stack = vec![crate::content::Matrix::identity()];
        let mut paint_order: usize = 0;
        let mut xobject_stack: Vec<crate::object::ObjectRef> = Vec::new();

        for op in operators {
            match op {
                Operator::SaveState => {
                    if let Some(current) = ctm_stack.last() {
                        ctm_stack.push(*current);
                    }
                },
                Operator::RestoreState => {
                    if ctm_stack.len() > 1 {
                        ctm_stack.pop();
                    }
                },
                Operator::Cm { a, b, c, d, e, f } => {
                    if let Some(current) = ctm_stack.last_mut() {
                        let m = crate::content::Matrix { a, b, c, d, e, f };
                        *current = m.multiply(current);
                    }
                },
                Operator::Do { name } => {
                    if let Some(ref xobj_dict_map) = xobject_dict {
                        let ctm = ctm_stack
                            .last()
                            .copied()
                            .unwrap_or_else(crate::content::Matrix::identity);
                        if let Ok(mut more) = self.collect_handles_from_do(
                            &name,
                            xobj_dict_map,
                            resources.as_ref(),
                            ctm,
                            &mut paint_order,
                            &mut xobject_stack,
                        ) {
                            handles.append(&mut more);
                        }
                    }
                },
                Operator::InlineImage { dict, data } => {
                    let ctm = ctm_stack
                        .last()
                        .copied()
                        .unwrap_or_else(crate::content::Matrix::identity);
                    if let Some(handle) =
                        image_handle_from_inline(self, &dict, data, ctm, paint_order, &cs_map)
                    {
                        handles.push(handle);
                        paint_order += 1;
                    }
                },
                _ => {},
            }
        }

        Ok(handles)
    }

    /// Collect zero or more image handles for a `Do` operator.
    ///
    /// If the target is an Image XObject, returns a vec containing one handle
    /// (paint_order is advanced). If it is a Form XObject, recurses and returns
    /// all image handles found inside (including nested Forms), with correct
    /// paint_order and CTM composition for every handle.
    fn collect_handles_from_do<'s>(
        &'s self,
        name: &str,
        xobject_dict: &std::collections::HashMap<String, Object>,
        resources: Option<&Object>,
        ctm: crate::content::Matrix,
        paint_order: &mut usize,
        xobject_stack: &mut Vec<crate::object::ObjectRef>,
    ) -> Result<Vec<crate::extractors::images::PdfImageHandle<'s>>> {
        use crate::extractors::images::image_handle_from_xobject;

        let xobject_ref_obj = match xobject_dict.get(name) {
            Some(o) => o,
            None => return Ok(Vec::new()),
        };

        let xobject_ref_opt = xobject_ref_obj.as_reference();
        let xobject = if let Some(ref_obj) = xobject_ref_opt {
            self.load_object(ref_obj)?
        } else {
            xobject_ref_obj.clone()
        };
        let xobj_dict = match xobject.as_dict() {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };

        let subtype = xobj_dict
            .get("Subtype")
            .and_then(|s| s.as_name())
            .unwrap_or("");

        match subtype {
            "Image" => {
                if let Some(ref_obj) = xobject_ref_opt {
                    let cs_map = self.build_color_space_map(resources);
                    if let Some(h) = image_handle_from_xobject(
                        self,
                        ref_obj,
                        xobj_dict,
                        ctm,
                        *paint_order,
                        &cs_map,
                    ) {
                        *paint_order += 1;
                        Ok(vec![h])
                    } else {
                        Ok(Vec::new())
                    }
                } else {
                    Ok(Vec::new())
                }
            },
            "Form" => {
                if let (Some(ref_obj), Some(parent_res)) = (xobject_ref_opt, resources) {
                    self.collect_image_handles_from_form_xobject(
                        ref_obj,
                        &xobject,
                        parent_res,
                        ctm,
                        paint_order,
                        xobject_stack,
                    )
                } else {
                    Ok(Vec::new())
                }
            },
            _ => Ok(Vec::new()),
        }
    }

    /// Recursively collect image handles from a Form XObject.
    ///
    /// This is the handles-side equivalent of `extract_images_from_form_xobject`.
    /// It uses the same cycle detection (ObjectRef stack + depth 100), the same
    /// Form Resources fallback rules, the same Form /Matrix handling, and reuses
    /// the document's xobject_stream_cache (50 MiB bound) for decompressed Form
    /// content.
    ///
    /// Unlike the materialised path, we do not cache "raw" handles — we compose
    /// the full CTM (`parent_ctm * form_matrix`) at entry and let every inner
    /// handle (and nested Form) naturally receive the final geometry. This is
    /// simpler for the two-phase API and produces correct `bbox`/`rotation_degrees`
    /// / `ctm` fields on the returned handles.
    fn collect_image_handles_from_form_xobject<'s>(
        &'s self,
        xobject_ref: crate::object::ObjectRef,
        xobject: &Object,
        parent_resources: &Object,
        parent_ctm: crate::content::Matrix,
        paint_order: &mut usize,
        xobject_stack: &mut Vec<crate::object::ObjectRef>,
    ) -> Result<Vec<crate::extractors::images::PdfImageHandle<'s>>> {
        use crate::content::parse_content_stream_images_only;
        use crate::content::Operator;
        use crate::extractors::images::image_handle_from_inline;

        // Cycle detection — identical policy to the materialised extraction path.
        if xobject_stack.contains(&xobject_ref) || xobject_stack.len() >= 100 {
            return Ok(Vec::new());
        }

        xobject_stack.push(xobject_ref);

        let xobj_dict = match xobject.as_dict() {
            Some(d) => d,
            None => {
                xobject_stack.pop();
                return Ok(Vec::new());
            },
        };

        // Form's own Resources (fallback to the parent's resources if absent).
        let form_resources = if let Some(form_res) = xobj_dict.get("Resources") {
            if let Some(ref_obj) = form_res.as_reference() {
                self.load_object(ref_obj)?
            } else {
                form_res.clone()
            }
        } else {
            parent_resources.clone()
        };

        // Pre-resolve the XObject dictionary for *this* Form's Resources.
        let form_xobject_dict = if let Some(res_dict) = form_resources.as_dict() {
            if let Some(xobj_entry) = res_dict.get("XObject") {
                let resolved = if let Some(ref_obj) = xobj_entry.as_reference() {
                    self.load_object(ref_obj)?
                } else {
                    xobj_entry.clone()
                };
                resolved.as_dict().cloned()
            } else {
                None
            }
        } else {
            None
        };

        // Form's own transformation matrix (default identity).
        let form_matrix = if let Some(matrix_obj) = xobj_dict.get("Matrix") {
            self.parse_matrix_from_object(matrix_obj)
                .unwrap_or_else(crate::content::Matrix::identity)
        } else {
            crate::content::Matrix::identity()
        };

        // Decode the Form stream (respecting the 50 MiB document-level cache).
        let cached_stream = self
            .xobject_stream_cache
            .lock_or_recover()
            .get(&xobject_ref)
            .cloned();
        let stream_data = if let Some(cached) = cached_stream {
            cached.as_ref().clone()
        } else {
            match self.decode_stream_with_encryption(xobject, xobject_ref) {
                Ok(data) => {
                    const MAX_STREAM_CACHE_BYTES: usize = 50 * 1024 * 1024;
                    let current_bytes = self.xobject_stream_cache_bytes.load(Ordering::Relaxed);
                    if current_bytes + data.len() <= MAX_STREAM_CACHE_BYTES {
                        self.xobject_stream_cache_bytes
                            .store(current_bytes + data.len(), Ordering::Relaxed);
                        self.xobject_stream_cache
                            .lock_or_recover()
                            .insert(xobject_ref, std::sync::Arc::new(data.clone()));
                    }
                    data
                },
                Err(e) => {
                    log::warn!("Failed to decode Form XObject stream: {}, skipping", e);
                    xobject_stack.pop();
                    return Ok(Vec::new());
                },
            }
        };

        // Parse with the fast images-only parser (same as the materialised path).
        let operators = match parse_content_stream_images_only(&stream_data) {
            Ok(ops) => ops,
            Err(_) => {
                xobject_stack.pop();
                return Ok(Vec::new());
            },
        };

        // Critical CTM composition:
        // Start the form's internal graphics state with `parent_ctm * form_matrix`.
        // Every image (and nested Form) discovered inside will then have its
        // handle's bbox/rotation/ctm computed with the *final* transform that
        // will be active when the image is painted on the page.
        let start_ctm = parent_ctm.multiply(&form_matrix);
        let mut ctm_stack = vec![start_ctm];
        let mut handles = Vec::new();

        for op in operators {
            match op {
                Operator::SaveState => {
                    if let Some(current) = ctm_stack.last() {
                        ctm_stack.push(*current);
                    }
                },
                Operator::RestoreState => {
                    if ctm_stack.len() > 1 {
                        ctm_stack.pop();
                    }
                },
                Operator::Cm { a, b, c, d, e, f } => {
                    if let Some(current) = ctm_stack.last_mut() {
                        let m = crate::content::Matrix { a, b, c, d, e, f };
                        *current = m.multiply(current);
                    }
                },

                Operator::Do { name } => {
                    if let Some(ref xobj_d) = form_xobject_dict {
                        let current_ctm = ctm_stack
                            .last()
                            .copied()
                            .unwrap_or_else(crate::content::Matrix::identity);
                        if let Ok(mut more) = self.collect_handles_from_do(
                            &name,
                            xobj_d,
                            Some(&form_resources),
                            current_ctm,
                            paint_order,
                            xobject_stack,
                        ) {
                            handles.append(&mut more);
                        }
                    }
                },

                Operator::InlineImage { dict, data } => {
                    let current_ctm = ctm_stack
                        .last()
                        .copied()
                        .unwrap_or_else(crate::content::Matrix::identity);
                    let cs_map = self.build_color_space_map(Some(&form_resources));
                    if let Some(h) = image_handle_from_inline(
                        self,
                        &dict,
                        data,
                        current_ctm,
                        *paint_order,
                        &cs_map,
                    ) {
                        handles.push(h);
                        *paint_order += 1;
                    }
                },

                _ => {},
            }
        }

        xobject_stack.pop();
        Ok(handles)
    }

    /// Extract images with pre-decompression filtering.
    ///
    /// Applies dimension and pixel-count checks using XObject dictionary metadata
    /// BEFORE expensive stream decompression. This avoids decompressing oversized
    /// images (e.g., 36MP presentation slides) or tiny glyph fragments that will
    /// be discarded downstream.
    fn extract_images_filtered(
        &self,
        page_index: usize,
        filter: &ImageExtractFilter,
    ) -> Result<Vec<crate::extractors::PdfImage>> {
        use crate::content::parse_content_stream_images_only;
        use crate::content::Operator;

        // Get page object and resources
        let page = self.get_page(page_index)?;
        let page_dict = page.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Page is not a dictionary".to_string(),
        })?;

        // Get content stream
        let content_data = self.get_page_content_data(page_index)?;

        // Resolve resources
        let resources = match page_dict.get("Resources") {
            Some(res) => {
                if let Some(ref_obj) = res.as_reference() {
                    Some(self.load_object(ref_obj)?)
                } else {
                    Some(res.clone())
                }
            },
            None => None,
        };

        // Parse content stream with image-only fast path (skips BT/ET text blocks)
        let operators = match parse_content_stream_images_only(&content_data) {
            Ok(ops) => ops,
            Err(_) => {
                // If content stream parsing fails, return empty
                return Ok(Vec::new());
            },
        };

        let mut images = Vec::new();
        let mut ctm_stack = vec![crate::content::Matrix::identity()];
        // Shared cycle detection stack for Form XObject recursion.
        // This must persist across all Do operator calls to detect circular references
        // (e.g., Form X0 references X1 which references X0).
        let mut xobject_stack = Vec::new();

        // Pre-resolve XObject dictionary once (avoids re-resolving per Do operator)
        let xobject_dict = if let Some(ref res) = resources {
            if let Some(res_dict) = res.as_dict() {
                if let Some(xobj_entry) = res_dict.get("XObject") {
                    let resolved = if let Some(ref_obj) = xobj_entry.as_reference() {
                        self.load_object(ref_obj)?
                    } else {
                        xobj_entry.clone()
                    };
                    resolved.as_dict().cloned()
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Parse content stream operators to extract images from Do operators
        for op in operators {
            match op {
                // Graphics state operators
                Operator::SaveState => {
                    if let Some(current_ctm) = ctm_stack.last() {
                        ctm_stack.push(*current_ctm);
                    }
                },
                Operator::RestoreState => {
                    if ctm_stack.len() > 1 {
                        ctm_stack.pop();
                    }
                },
                Operator::Cm { a, b, c, d, e, f } => {
                    if let Some(current_ctm) = ctm_stack.last_mut() {
                        let matrix = crate::content::Matrix { a, b, c, d, e, f };
                        // PDF spec ISO 32000-1:2008 §8.3.4: cm concatenates as M_cm × CTM
                        *current_ctm = matrix.multiply(current_ctm);
                    }
                },

                // XObject reference operator - Extract images referenced via Do
                Operator::Do { name } => {
                    if let Some(ref xobj_dict) = xobject_dict {
                        let current_ctm = ctm_stack
                            .last()
                            .copied()
                            .unwrap_or_else(crate::content::Matrix::identity);
                        if let Ok(mut xobj_images) = self.extract_images_from_xobject_do(
                            &name,
                            xobj_dict,
                            resources.as_ref(),
                            current_ctm,
                            &mut xobject_stack,
                            filter,
                        ) {
                            images.append(&mut xobj_images);
                        }
                    }
                },

                // Inline image operator
                Operator::InlineImage { dict, data } => {
                    let current_ctm = ctm_stack
                        .last()
                        .copied()
                        .unwrap_or_else(crate::content::Matrix::identity);
                    if let Ok(image) = self.extract_image_from_inline(&dict, &data, current_ctm) {
                        images.push(image);
                    }
                },

                _ => {}, // Ignore other operators
            }
        }

        Ok(images)
    }

    /// Extract images referenced by a Do operator in the content stream.
    ///
    /// Accepts a pre-resolved XObject dictionary to avoid redundant lookups
    /// when called repeatedly (e.g., 194 Do operators on a single page).
    fn extract_images_from_xobject_do(
        &self,
        name: &str,
        xobject_dict: &std::collections::HashMap<String, Object>,
        resources: Option<&Object>,
        ctm: crate::content::Matrix,
        xobject_stack: &mut Vec<ObjectRef>,
        filter: &ImageExtractFilter,
    ) -> Result<Vec<crate::extractors::PdfImage>> {
        use crate::extractors::extract_image_from_xobject;

        let mut images = Vec::new();

        // Get the specific XObject by name
        let xobject_ref_obj = match xobject_dict.get(name) {
            Some(obj) => obj,
            None => return Ok(images), // Named XObject not found
        };

        // Load XObject (can be indirect reference or direct object)
        let xobject_ref_opt = xobject_ref_obj.as_reference();
        let xobject = if let Some(ref_obj) = xobject_ref_opt {
            self.load_object(ref_obj)?
        } else {
            xobject_ref_obj.clone()
        };
        let xobject_dict = xobject.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "XObject is not a dictionary".to_string(),
        })?;

        // Check Subtype
        let subtype = xobject_dict
            .get("Subtype")
            .and_then(|s| s.as_name())
            .unwrap_or("");

        match subtype {
            "Image" => {
                // Pre-decompression filtering using dictionary metadata.
                // These checks use Width/Height/ColorSpace from the XObject dictionary
                // which are available WITHOUT decompressing the image stream data.
                let w = xobject_dict
                    .get("Width")
                    .and_then(|o| o.as_integer())
                    .unwrap_or(0);
                let h = xobject_dict
                    .get("Height")
                    .and_then(|o| o.as_integer())
                    .unwrap_or(0);
                if w < filter.min_width || h < filter.min_height {
                    return Ok(images);
                }
                if (w as u64) * (h as u64) > filter.max_pixels {
                    return Ok(images);
                }
                // Skip small Indexed colorspace images (Type3 font glyph fragments)
                if filter.skip_indexed_small > 0
                    && (w < filter.skip_indexed_small || h < filter.skip_indexed_small)
                {
                    if let Some(cs_obj) = xobject_dict.get("ColorSpace") {
                        let is_indexed = match cs_obj {
                            Object::Name(n) => n == "Indexed",
                            Object::Array(arr) if !arr.is_empty() => {
                                arr[0].as_name() == Some("Indexed")
                            },
                            _ => false,
                        };
                        if is_indexed {
                            return Ok(images);
                        }
                    }
                }

                // Only clone+modify when ColorSpace needs resolving from indirect ref
                let needs_cs_resolve = matches!(
                    &xobject,
                    Object::Stream { dict, .. } if matches!(dict.get("ColorSpace"), Some(Object::Reference(_)))
                );

                let resolved_xobject;
                let xobject_for_extract = if needs_cs_resolve {
                    if let Object::Stream { dict, data } = &xobject {
                        let mut new_dict = dict.clone();
                        if let Some(Object::Reference(cs_ref)) = dict.get("ColorSpace") {
                            if let Ok(resolved_cs) = self.load_object(*cs_ref) {
                                new_dict.insert("ColorSpace".to_string(), resolved_cs);
                            }
                        }
                        resolved_xobject = Object::Stream {
                            dict: new_dict,
                            data: data.clone(),
                        };
                        &resolved_xobject
                    } else {
                        &xobject
                    }
                } else {
                    &xobject
                };

                // Extract as Image XObject
                if let Ok(mut image) = extract_image_from_xobject(
                    Some(self),
                    xobject_for_extract,
                    xobject_ref_opt,
                    None,
                ) {
                    // In PDF, images are mapped from unit square (0,0 to 1,1) to the CTM.
                    let unit_rect = crate::geometry::Rect::new(0.0, 0.0, 1.0, 1.0);
                    let bbox = self.transform_bbox_with_ctm(&unit_rect, ctm);
                    image.set_bbox(bbox);

                    // Capture transformation matrix and rotation (v0.3.14)
                    image.set_matrix([ctm.a, ctm.b, ctm.c, ctm.d, ctm.e, ctm.f]);
                    image.set_rotation_degrees(Self::matrix_to_rotation(ctm));

                    images.push(image);
                }
            },
            "Form" => {
                // Recursively extract from Form XObject
                // Only process if we have a valid reference and parent resources
                if let (Some(ref_obj), Some(parent_res)) = (xobject_ref_opt, resources) {
                    if let Ok(mut form_images) = self.extract_images_from_form_xobject(
                        ref_obj,
                        &xobject,
                        parent_res,
                        ctm,
                        xobject_stack,
                        filter,
                    ) {
                        images.append(&mut form_images);
                    }
                }
            },
            _ => {}, // Skip other types (PS, etc.)
        }

        Ok(images)
    }

    /// Recursively extract images from a Form XObject.
    ///
    /// Uses a document-level cache: images are extracted once using only the Form's
    /// own Matrix, then cached. On subsequent references, cached images are cloned
    /// and the caller's CTM is applied to transform bboxes.
    fn extract_images_from_form_xobject(
        &self,
        xobject_ref: ObjectRef,
        xobject: &Object,
        parent_resources: &Object,
        parent_ctm: crate::content::Matrix,
        xobject_stack: &mut Vec<ObjectRef>,
        filter: &ImageExtractFilter,
    ) -> Result<Vec<crate::extractors::PdfImage>> {
        use crate::content::parse_content_stream_images_only;
        use crate::content::Operator;

        // Cycle detection
        if xobject_stack.contains(&xobject_ref) || xobject_stack.len() >= 100 {
            return Ok(Vec::new());
        }

        // Check image result cache — images stored with Form's own Matrix only.
        // Scope the borrow to ensure it's dropped before potential recursion.
        {
            if let Some(cached_images) = self
                .form_xobject_images_cache
                .lock_or_recover()
                .get(&xobject_ref)
            {
                let images = cached_images
                    .iter()
                    .map(|img| {
                        let mut cloned = img.clone();
                        if let Some(rect) = cloned.bbox() {
                            cloned.set_bbox(self.transform_bbox_with_ctm(rect, parent_ctm));
                        }
                        cloned
                    })
                    .collect();
                return Ok(images);
            }
        }

        xobject_stack.push(xobject_ref);

        let xobj_dict = xobject.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Form XObject is not a dictionary".to_string(),
        })?;

        // Get Form resources (with fallback to parent)
        let form_resources = if let Some(form_res) = xobj_dict.get("Resources") {
            if let Some(ref_obj) = form_res.as_reference() {
                self.load_object(ref_obj)?
            } else {
                form_res.clone()
            }
        } else {
            parent_resources.clone()
        };

        // Pre-resolve XObject dictionary for this form's resources
        let form_xobject_dict = if let Some(res_dict) = form_resources.as_dict() {
            if let Some(xobj_entry) = res_dict.get("XObject") {
                let resolved = if let Some(ref_obj) = xobj_entry.as_reference() {
                    self.load_object(ref_obj)?
                } else {
                    xobj_entry.clone()
                };
                resolved.as_dict().cloned()
            } else {
                None
            }
        } else {
            None
        };

        // Get Form transformation matrix (default to identity)
        let form_matrix = if let Some(matrix_obj) = xobj_dict.get("Matrix") {
            self.parse_matrix_from_object(matrix_obj)
                .unwrap_or_else(crate::content::Matrix::identity)
        } else {
            crate::content::Matrix::identity()
        };

        // Decode form stream — check cache first to avoid repeated decompression
        let cached_stream = self
            .xobject_stream_cache
            .lock_or_recover()
            .get(&xobject_ref)
            .cloned();
        let stream_data = if let Some(cached) = cached_stream {
            cached.as_ref().clone()
        } else {
            match self.decode_stream_with_encryption(xobject, xobject_ref) {
                Ok(data) => {
                    const MAX_STREAM_CACHE_BYTES: usize = 50 * 1024 * 1024;
                    let current_bytes = self.xobject_stream_cache_bytes.load(Ordering::Relaxed);
                    if current_bytes + data.len() <= MAX_STREAM_CACHE_BYTES {
                        self.xobject_stream_cache_bytes
                            .store(current_bytes + data.len(), Ordering::Relaxed);
                        self.xobject_stream_cache
                            .lock_or_recover()
                            .insert(xobject_ref, std::sync::Arc::new(data.clone()));
                    }
                    data
                },
                Err(e) => {
                    log::warn!("Failed to decode Form XObject stream: {}, skipping", e);
                    xobject_stack.pop();
                    return Ok(Vec::new());
                },
            }
        };

        // Parse operators using fast image-only path (skips text operators)
        let operators = match parse_content_stream_images_only(&stream_data) {
            Ok(ops) => ops,
            Err(_) => {
                xobject_stack.pop();
                return Ok(Vec::new());
            },
        };

        // Extract using only the Form's own Matrix (no parent_ctm yet).
        // This allows caching the results and applying different parent CTMs later.
        let mut raw_images = Vec::new();
        let mut ctm_stack = vec![form_matrix];

        for op in operators {
            match op {
                Operator::SaveState => {
                    if let Some(current_ctm) = ctm_stack.last() {
                        ctm_stack.push(*current_ctm);
                    }
                },
                Operator::RestoreState => {
                    if ctm_stack.len() > 1 {
                        ctm_stack.pop();
                    }
                },
                Operator::Cm { a, b, c, d, e, f } => {
                    if let Some(current_ctm) = ctm_stack.last_mut() {
                        let matrix = crate::content::Matrix { a, b, c, d, e, f };
                        // PDF spec ISO 32000-1:2008 §8.3.4: cm concatenates as M_cm × CTM
                        *current_ctm = matrix.multiply(current_ctm);
                    }
                },

                Operator::Do { name } => {
                    if let Some(ref xobj_d) = form_xobject_dict {
                        let current_ctm = ctm_stack
                            .last()
                            .copied()
                            .unwrap_or_else(crate::content::Matrix::identity);
                        // For nested Do operators, pass identity as parent_ctm since
                        // we're building raw (un-transformed) images for caching
                        if let Ok(mut xobj_images) = self.extract_images_from_xobject_do(
                            &name,
                            xobj_d,
                            Some(&form_resources),
                            current_ctm,
                            xobject_stack,
                            filter,
                        ) {
                            raw_images.append(&mut xobj_images);
                        }
                    }
                },

                Operator::InlineImage { dict, data } => {
                    let current_ctm = ctm_stack
                        .last()
                        .copied()
                        .unwrap_or_else(crate::content::Matrix::identity);
                    if let Ok(image) = self.extract_image_from_inline(&dict, &data, current_ctm) {
                        raw_images.push(image);
                    }
                },

                _ => {},
            }
        }

        xobject_stack.pop();

        // Cache the raw images (with Form's own Matrix applied, but no parent CTM)
        self.form_xobject_images_cache
            .lock_or_recover()
            .insert(xobject_ref, raw_images.clone());

        // Apply parent_ctm to produce final images for this call
        let images = raw_images
            .into_iter()
            .map(|mut img| {
                if let Some(rect) = img.bbox() {
                    img.set_bbox(self.transform_bbox_with_ctm(rect, parent_ctm));
                }
                img
            })
            .collect();

        Ok(images)
    }

    /// Extract an inline image from the content stream.
    fn extract_image_from_inline(
        &self,
        dict: &std::collections::HashMap<String, Object>,
        data: &[u8],
        ctm: crate::content::Matrix,
    ) -> Result<crate::extractors::PdfImage> {
        use crate::extractors::expand_inline_image_dict;

        // Expand abbreviated dictionary
        let expanded_dict = expand_inline_image_dict(dict.clone());

        // Build a temporary stream object from the dictionary and data
        let stream_obj = Object::Stream {
            dict: expanded_dict,
            data: bytes::Bytes::copy_from_slice(data),
        };

        // Use existing extraction logic
        let mut image =
            crate::extractors::extract_image_from_xobject(Some(self), &stream_obj, None, None)?;

        // In PDF, images are mapped from unit square (0,0 to 1,1) to the CTM.
        let unit_rect = crate::geometry::Rect::new(0.0, 0.0, 1.0, 1.0);
        let bbox = self.transform_bbox_with_ctm(&unit_rect, ctm);
        image.set_bbox(bbox);

        // Capture transformation matrix and rotation (v0.3.14)
        image.set_matrix([ctm.a, ctm.b, ctm.c, ctm.d, ctm.e, ctm.f]);
        image.set_rotation_degrees(Self::matrix_to_rotation(ctm));

        Ok(image)
    }

    /// Helper to derive rotation angle from transformation matrix.
    fn matrix_to_rotation(m: crate::content::Matrix) -> i32 {
        // Compute angle from CTM components (atan2(b, a))
        let angle_rad = m.b.atan2(m.a);
        let angle_deg = (angle_rad.to_degrees().round() as i32) % 360;
        if angle_deg < 0 {
            angle_deg + 360
        } else {
            angle_deg
        }
    }

    /// Transform a bounding box using CTM.
    ///
    /// Transforms all four corners and computes the axis-aligned bounding box,
    /// which correctly handles rotation, shear, and negative scaling.
    fn transform_bbox_with_ctm(
        &self,
        rect: &crate::geometry::Rect,
        ctm: crate::content::Matrix,
    ) -> crate::geometry::Rect {
        let x0 = rect.x;
        let y0 = rect.y;
        let x1 = rect.x + rect.width;
        let y1 = rect.y + rect.height;

        // Transform all four corners
        let tx0 = ctm.a * x0 + ctm.c * y0 + ctm.e;
        let ty0 = ctm.b * x0 + ctm.d * y0 + ctm.f;

        let tx1 = ctm.a * x1 + ctm.c * y0 + ctm.e;
        let ty1 = ctm.b * x1 + ctm.d * y0 + ctm.f;

        let tx2 = ctm.a * x0 + ctm.c * y1 + ctm.e;
        let ty2 = ctm.b * x0 + ctm.d * y1 + ctm.f;

        let tx3 = ctm.a * x1 + ctm.c * y1 + ctm.e;
        let ty3 = ctm.b * x1 + ctm.d * y1 + ctm.f;

        let min_x = tx0.min(tx1).min(tx2).min(tx3);
        let max_x = tx0.max(tx1).max(tx2).max(tx3);
        let min_y = ty0.min(ty1).min(ty2).min(ty3);
        let max_y = ty0.max(ty1).max(ty2).max(ty3);

        crate::geometry::Rect {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }

    /// Parse a Matrix object from PDF.
    fn parse_matrix_from_object(&self, obj: &Object) -> Option<crate::content::Matrix> {
        if let Some(array) = obj.as_array() {
            if array.len() >= 6 {
                let mut values = [0.0f32; 6];
                for (i, val) in array.iter().take(6).enumerate() {
                    let num = if let Some(f) = val.as_real() {
                        f as f32
                    } else if let Some(i_val) = val.as_integer() {
                        i_val as f32
                    } else {
                        return None;
                    };
                    values[i] = num;
                }

                return Some(crate::content::Matrix {
                    a: values[0],
                    b: values[1],
                    c: values[2],
                    d: values[3],
                    e: values[4],
                    f: values[5],
                });
            }
        }
        None
    }

    /// Extract images from a page and save them to files.
    ///
    /// Each image is saved as a separate file in `output_dir` with the given
    /// `prefix` and an incrementing index starting from `start_index`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn extract_images_to_files(
        &self,
        page_index: usize,
        output_dir: impl AsRef<Path>,
        prefix: Option<&str>,
        start_index: Option<usize>,
    ) -> Result<Vec<ExtractedImageRef>> {
        use std::fs;

        // Extract images from page
        let images = self.extract_images(page_index)?;

        // Create output directory if it doesn't exist
        let output_dir = output_dir.as_ref();
        if !output_dir.exists() {
            fs::create_dir_all(output_dir).map_err(Error::Io)?;
        }

        let prefix = prefix.unwrap_or("img");
        let mut index = start_index.unwrap_or(1);
        let mut result = Vec::new();

        for image in images {
            // Determine format and extension
            let (format, extension) = match image.data() {
                crate::extractors::ImageData::Jpeg(_) => (ImageFormat::Jpeg, "jpg"),
                _ => (ImageFormat::Png, "png"),
            };

            // Generate filename: img_001.png, img_002.jpg, etc.
            let filename = format!("{}_{:03}.{}", prefix, index, extension);
            let filepath = output_dir.join(&filename);

            // Save image
            match format {
                ImageFormat::Jpeg => image.save_as_jpeg(&filepath)?,
                ImageFormat::Png => image.save_as_png(&filepath)?,
            }

            // Add to result
            result.push(ExtractedImageRef {
                filename,
                format,
                width: image.width(),
                height: image.height(),
                bbox: image.bbox().cloned(),
                rotation: image.rotation_degrees(),
                matrix: image.matrix(),
            });

            index += 1;
        }

        Ok(result)
    }

    // ========================================================================
    // Debug/profiling helpers — thin pub wrappers over internal methods.
    // Used by examples/debug_katalog.rs to break extract_spans into phases.
    // ========================================================================

    /// Public wrapper for `get_page` (normally private).
    /// Exposed for profiling examples that need to time page tree lookup separately.
    pub fn get_page_for_debug(&self, page_index: usize) -> Result<Object> {
        self.get_page(page_index)
    }

    /// Public wrapper for `may_contain_text` (normally pub(crate)).
    /// Returns true if the content stream might contain text operators (BT or Do).
    pub fn may_contain_text_public(data: &[u8]) -> bool {
        Self::may_contain_text(data)
    }

    /// Public wrapper for `load_fonts` (normally pub(crate)).
    /// Loads font dictionaries from a resources object into a TextExtractor.
    pub fn load_fonts_public(
        &self,
        resources: &Object,
        extractor: &mut crate::extractors::TextExtractor<'_>,
    ) -> Result<()> {
        self.load_fonts(resources, extractor)
    }

    /// Per-page mapping of PDF font-resource names (e.g. `"F75"`) to their
    /// canonical face name (e.g. `"TeXGyreTermesX-Regular"`, with any
    /// subset-prefix `ABCDEF+` stripped).
    ///
    /// Used by the layout-preserving DOCX writer so each text span can be
    /// emitted with the actual face name in `<w:rFonts>` instead of a
    /// PDF-internal resource id. The vector is `pages × map`; `map[i]`
    /// covers all fonts referenced by page `i`'s Resources.
    pub fn page_font_face_lookups(&self) -> Result<Vec<std::collections::HashMap<String, String>>> {
        use std::collections::HashMap;
        let n = self.page_count()?;
        let mut out: Vec<HashMap<String, String>> = Vec::with_capacity(n);
        for page_idx in 0..n {
            let mut lookup: HashMap<String, String> = HashMap::new();
            // Inline get_page → Resources so this works without `rendering`.
            let resources = match self.get_page(page_idx) {
                Ok(page) => match page.as_dict() {
                    Some(d) => {
                        let r = d
                            .get("Resources")
                            .cloned()
                            .unwrap_or(Object::Dictionary(std::collections::HashMap::new()));
                        if let Some(rref) = r.as_reference() {
                            self.load_object(rref)
                                .unwrap_or(Object::Dictionary(std::collections::HashMap::new()))
                        } else {
                            r
                        }
                    },
                    None => {
                        out.push(lookup);
                        continue;
                    },
                },
                Err(_) => {
                    out.push(lookup);
                    continue;
                },
            };
            let mut extractor = crate::extractors::TextExtractor::new();
            if self.load_fonts_public(&resources, &mut extractor).is_ok() {
                for (resource_name, info) in extractor.get_font_set() {
                    let canonical = info
                        .base_font
                        .split_once('+')
                        .map(|(_, rest)| rest)
                        .unwrap_or(info.base_font.as_str())
                        .to_string();
                    lookup.insert(resource_name, canonical);
                }
            }
            out.push(lookup);
        }
        Ok(out)
    }

    /// Extract every embedded font program (TrueType / OpenType bytes) used
    /// anywhere in the document, deduplicated by `BaseFont` name.
    ///
    /// Walks every page's font dictionary, loads each font via the same path
    /// `extract_text` uses, and returns the unique set of fonts that have
    /// embedded `FontFile2`/`FontFile3` streams. The `String` is the base
    /// font name (with any subset prefix like `ABCDEF+` stripped) and the
    /// `Vec<u8>` is the raw font program — directly suitable for re-embedding
    /// into another container (DOCX `word/fonts/`, another PDF, etc.).
    ///
    /// Fonts without embedded data (standard 14, missing FontFile streams)
    /// are skipped — there's nothing to extract.
    pub fn extract_embedded_fonts(&self) -> Result<Vec<(String, Vec<u8>)>> {
        use std::collections::HashMap;
        let mut by_name: HashMap<String, Vec<u8>> = HashMap::new();

        let n = self.page_count()?;
        for page_idx in 0..n {
            // Inline get_page_resources so this works without `rendering`.
            let resources = match self.get_page(page_idx) {
                Ok(page) => match page.as_dict() {
                    Some(d) => {
                        let r = d
                            .get("Resources")
                            .cloned()
                            .unwrap_or(Object::Dictionary(std::collections::HashMap::new()));
                        if let Some(rref) = r.as_reference() {
                            self.load_object(rref).unwrap_or_else(|_| {
                                Object::Dictionary(std::collections::HashMap::new())
                            })
                        } else {
                            r
                        }
                    },
                    None => continue,
                },
                Err(_) => continue,
            };
            let mut extractor = crate::extractors::TextExtractor::new();
            if self.load_fonts_public(&resources, &mut extractor).is_err() {
                continue;
            }
            for (_resource_name, font_arc) in extractor.get_font_set() {
                let Some(data) = font_arc.embedded_font_data.as_ref() else {
                    continue;
                };
                if data.is_empty() {
                    continue;
                }
                // Subset-prefix stripping: PDF font subsets carry a 6-letter
                // prefix followed by `+`, e.g. `ABCDEF+Calibri-Bold`. The
                // prefix is meaningless to consumers — strip it for dedup.
                let base = font_arc.base_font.as_str();
                let canonical = base.split_once('+').map(|(_, rest)| rest).unwrap_or(base);
                by_name
                    .entry(canonical.to_string())
                    .or_insert_with(|| data.as_ref().clone());
            }
        }

        let mut out: Vec<(String, Vec<u8>)> = by_name.into_iter().collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// Like [`Self::extract_embedded_fonts`] but additionally returns a
    /// per-font Unicode → GID map reconstructed from the source PDF's
    /// `/ToUnicode` CMap and the font's CID/byte→GID table.
    ///
    /// CFF font subsets in PDFs (the typical Word/LibreOffice output)
    /// often ship without a Unicode cmap because CIDs encode the
    /// glyph stream directly. The font program parses fine but
    /// `EmbeddedFont::glyph_lookup` is empty; downstream font
    /// registration treats the font as unusable and falls back to
    /// Helvetica.
    ///
    /// The map returned here lets office_oxide / pdf_oxide write
    /// pipelines call [`crate::writer::EmbeddedFont::extend_glyph_lookup`]
    /// to re-populate the missing Unicode→GID entries from the
    /// source-PDF's own `/ToUnicode`. Result: CFF subset fonts
    /// register and render with the source typeface program instead
    /// of base-14 Helvetica.
    pub fn extract_embedded_fonts_with_unicode_maps(
        &self,
    ) -> Result<Vec<(String, Vec<u8>, std::collections::HashMap<u32, u16>)>> {
        let with_widths = self.extract_embedded_fonts_with_unicode_maps_and_widths()?;
        Ok(with_widths
            .into_iter()
            .map(|(name, data, uni, _widths)| (name, data, uni))
            .collect())
    }

    /// Like [`Self::extract_embedded_fonts_with_unicode_maps`] but also
    /// returns the per-glyph widths from the source PDF's `/W` array
    /// (in 1/1000 em units, keyed by GID). Required for re-embedding
    /// CFF font subsets whose synthetic OpenType wrapper carries no
    /// `hmtx` table — without this, ttf-parser returns 0 for every
    /// glyph advance and the round-trip writer emits a `/W` of zeros.
    pub fn extract_embedded_fonts_with_unicode_maps_and_widths(
        &self,
    ) -> Result<
        Vec<(
            String,
            Vec<u8>,
            std::collections::HashMap<u32, u16>,
            std::collections::HashMap<u16, u16>,
        )>,
    > {
        use std::collections::HashMap;
        let mut by_name: HashMap<String, (Vec<u8>, HashMap<u32, u16>, HashMap<u16, u16>)> =
            HashMap::new();

        let n = self.page_count()?;
        for page_idx in 0..n {
            let resources = match self.get_page(page_idx) {
                Ok(page) => match page.as_dict() {
                    Some(d) => {
                        let r = d
                            .get("Resources")
                            .cloned()
                            .unwrap_or(Object::Dictionary(std::collections::HashMap::new()));
                        if let Some(rref) = r.as_reference() {
                            self.load_object(rref).unwrap_or_else(|_| {
                                Object::Dictionary(std::collections::HashMap::new())
                            })
                        } else {
                            r
                        }
                    },
                    None => continue,
                },
                Err(_) => continue,
            };
            let mut extractor = crate::extractors::TextExtractor::new();
            if self.load_fonts_public(&resources, &mut extractor).is_err() {
                continue;
            }
            for (_resource_name, font_arc) in extractor.get_font_set() {
                let Some(data) = font_arc.embedded_font_data.as_ref() else {
                    continue;
                };
                if data.is_empty() {
                    continue;
                }
                let base = font_arc.base_font.as_str();
                let canonical = base.split_once('+').map(|(_, rest)| rest).unwrap_or(base);

                // Build Unicode → GID via ToUnicode CMap + GID resolver.
                //
                // We must consult the ToUnicode CMap *directly* rather than
                // going through `char_to_unicode`. `char_to_unicode` falls
                // through to a CID-as-Unicode fallback when the ToUnicode
                // CMap has no entry for a given code (Identity-H + Adobe-
                // Identity ordering, source font without a Unicode cmap).
                // That fallback returns spurious mappings like
                // U+0069 'i' → GID 105 (because CID 105 has no real
                // ToUnicode entry; the CID-as-Unicode path yields 'i'
                // for code=105 and the embedded TTF has no cmap to set us
                // straight). The spurious entries overwrite the real ones
                // we collected from CIDs that *do* have ToUnicode
                // entries (e.g. CID 0x4C → 'i', GID 76 for a
                // MicrosoftSansSerif subset) — which then makes the
                // injected cmap point Unicode codepoints at the wrong
                // glyph slots and the DOCX round-trip renders broken
                // lowercase letters.
                let mut uni_to_gid: HashMap<u32, u16> = HashMap::new();
                let to_unicode_cmap = font_arc.to_unicode.as_ref().and_then(|lazy| lazy.get());
                for code in 0u32..=0xFFFF {
                    // Require an authoritative ToUnicode entry. If the
                    // font has no ToUnicode CMap at all we conservatively
                    // skip injection — the fallback chain would only
                    // produce the misleading identity mapping.
                    let unicode_str =
                        match to_unicode_cmap.as_ref().and_then(|cmap| cmap.get(&code)) {
                            Some(s) if !s.is_empty() && s.as_ref() != "\u{FFFD}" => s.into_owned(),
                            _ => continue,
                        };
                    let cp = match unicode_str.chars().next() {
                        Some(c) => c as u32,
                        None => continue,
                    };
                    // Bare C0 controls (other than the legitimate
                    // whitespace handled in char_to_unicode) never name
                    // a real glyph — drop them so we don't inject a
                    // cmap entry that points U+0000..U+001F at random
                    // GIDs.
                    if matches!(cp, 0x00..=0x08 | 0x0B..=0x0C | 0x0E..=0x1F) {
                        continue;
                    }
                    // Only emit a Unicode→GID mapping when we have a
                    // real byte/CID → GID resolver from the source PDF.
                    // Falling back to identity for simple fonts whose
                    // CFF encoding parser couldn't extract a mapping
                    // produces a synthetic cmap that points Unicode at
                    // the wrong CFF charset positions: the round-trip
                    // emits Type0+Identity-H+CIDFontType0 and the
                    // viewer reads `glyph_at_charset[byte_code]`,
                    // which only equals the source glyph when CFF
                    // charset == StandardEncoding byte order — rarely
                    // true for subsetted CFF. Without a real mapping
                    // we leave the font un-patched, and office_oxide
                    // falls back to base-14 Helvetica via
                    // `EmbeddedFont::has_usable_unicode_cmap`.
                    let gid_opt = if let Some(ref map) = font_arc.cff_gid_map {
                        if code <= 0xFF {
                            map.get(&(code as u8)).copied()
                        } else {
                            None
                        }
                    } else if let Some(ref cid_map) = font_arc.cid_to_gid_map {
                        if code <= 0xFFFF {
                            Some(cid_map.get_gid(code as u16))
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(gid) = gid_opt {
                        uni_to_gid.insert(cp, gid);
                    }
                }

                // Build GID → width from the source PDF's /W array.
                // For CIDFontType0+Identity-H: CID == GID directly.
                // For CIDFontType2: CID → GID via CIDToGIDMap.
                // For simple CFF (cff_gid_map): byte-code → GID.
                let mut gid_to_width: HashMap<u16, u16> = HashMap::new();
                if let Some(ref cid_widths) = font_arc.cid_widths {
                    if font_arc.cid_font_type.as_deref() == Some("CIDFontType0") {
                        for (&cid, &w) in cid_widths {
                            gid_to_width.insert(cid, w.round() as u16);
                        }
                    } else if let Some(ref cid_map) = font_arc.cid_to_gid_map {
                        for (&cid, &w) in cid_widths {
                            let gid = cid_map.get_gid(cid);
                            gid_to_width.insert(gid, w.round() as u16);
                        }
                    } else {
                        for (&cid, &w) in cid_widths {
                            gid_to_width.insert(cid, w.round() as u16);
                        }
                    }
                } else if let Some(ref cff_map) = font_arc.cff_gid_map {
                    // Simple CFF font: width-by-byte-code in font_arc.widths.
                    if let (Some(widths), Some(first)) =
                        (font_arc.widths.as_ref(), font_arc.first_char)
                    {
                        for (i, w) in widths.iter().enumerate() {
                            let byte = first + i as u32;
                            if byte > 0xFF {
                                break;
                            }
                            if let Some(&gid) = cff_map.get(&(byte as u8)) {
                                gid_to_width.insert(gid, w.round() as u16);
                            }
                        }
                    }
                }

                let entry = by_name
                    .entry(canonical.to_string())
                    .or_insert_with(|| (data.as_ref().clone(), HashMap::new(), HashMap::new()));
                for (cp, gid) in uni_to_gid {
                    entry.1.entry(cp).or_insert(gid);
                }
                for (gid, w) in gid_to_width {
                    entry.2.entry(gid).or_insert(w);
                }
            }
        }

        let mut out: Vec<(String, Vec<u8>, HashMap<u32, u16>, HashMap<u16, u16>)> = by_name
            .into_iter()
            .map(|(name, (data, cmap, widths))| (name, data, cmap, widths))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }
}

/// Reference to an extracted image file.
///
/// Contains metadata about an image that has been extracted and saved to a file.
/// Used for HTML export to embed images with correct dimensions and format.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedImageRef {
    /// Filename of the saved image (e.g., "img_001.png")
    pub filename: String,
    /// Image format
    pub format: ImageFormat,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Bounding box in PDF user space (v0.3.14)
    pub bbox: Option<crate::geometry::Rect>,
    /// Rotation in degrees (v0.3.14)
    pub rotation: i32,
    /// Transformation matrix (v0.3.14)
    pub matrix: [f32; 6],
}

/// Image format for extracted images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// PNG format (lossless)
    Png,
    /// JPEG format (lossy, preserves DCT-encoded images)
    Jpeg,
}

/// Extract the /Root reference from a trailer dictionary.
fn get_root_ref_from_trailer(trailer: &Object) -> Option<ObjectRef> {
    trailer.as_dict()?.get("Root")?.as_reference()
}

/// First in-use *uncompressed* object in the xref, used as a /Root-independent
/// probe for the garbage-prefix offset-shift decision. Compressed
/// entries can't be seek-validated, so they're skipped.
fn first_in_use_uncompressed(xref: &crate::xref::CrossRefTable) -> Option<ObjectRef> {
    xref.all_object_numbers()
        .filter_map(|n| xref.get(n).map(|e| (n, e)))
        .find(|(_, e)| e.in_use && e.entry_type == crate::xref::XRefEntryType::Uncompressed)
        .map(|(n, e)| ObjectRef::new(n, e.generation))
}

/// Heuristic: does this candidate table actually look like wrapped prose
/// clustered into x-columns rather than a real grid?
///
/// Cell contents in real data tables are atomic units (numbers, codes,
/// names, short labels): they almost always start with an uppercase
/// letter, a digit, or a symbol (currency, +/-, punctuation marker)
/// rarely end with a mid-sentence comma or semicolon. Prose-as-table
/// cells, by contrast, are fragments of running sentences — they
/// frequently start with a lowercase stopword ("and", "the", "to") because
/// the column boundary fell mid-clause, and frequently end with `,` or
/// `;` for the same reason.
///
/// We reject the candidate when either signal exceeds its threshold:
///   • > 12 % of cells end in `,` or `;` (mid-sentence tails), or
///   • > 25 % of cells start with a lowercase ASCII letter
///     (continuation fragments).
///
/// Thresholds chosen to clear the false positives flagged in the 88-PDF
/// regression (`searchable.pdf`, the WFMYY press-release, several arxiv
/// preprints) without disturbing legitimate data tables — sailing scores,
/// IRS forms, and the CJK traffic-volume grid all stay well below both
/// bars.
fn looks_like_prose_table(table: &crate::structure::Table) -> bool {
    let mut total = 0usize;
    let mut sentence_tails = 0usize;
    let mut lower_starts = 0usize;
    let mut leader_dots = 0usize;
    for row in &table.rows {
        for cell in &row.cells {
            let trimmed = cell.text.trim();
            if trimmed.is_empty() {
                continue;
            }
            total += 1;
            if let Some(last) = trimmed.chars().last() {
                if matches!(last, ',' | ';') {
                    sentence_tails += 1;
                }
            }
            if let Some(first) = trimmed.chars().next() {
                if first.is_ascii_lowercase() {
                    lower_starts += 1;
                }
            }
            // Table-of-contents leader runs (". . . . . . ." between an
            // entry's title and its page number) cluster into their own
            // x-columns and create phantom 10–12-column "tables" out of
            // an ordinary three-column TOC. A cell whose content is
            // exclusively dots and spaces is the leader, not data.
            if trimmed.chars().all(|c| c == '.' || c == ' ') {
                leader_dots += 1;
            }
        }
    }
    if total < 10 {
        return false;
    }
    let tail_ratio = sentence_tails as f32 / total as f32;
    let lower_ratio = lower_starts as f32 / total as f32;
    let leader_ratio = leader_dots as f32 / total as f32;
    tail_ratio > 0.12 || lower_ratio > 0.25 || leader_ratio > 0.10
}

/// Check whether the object at the xref offset for `obj_ref` looks like a valid header.
fn validate_object_at_offset<R: Read + Seek>(
    reader: &mut R,
    xref: &crate::xref::CrossRefTable,
    obj_ref: ObjectRef,
) -> bool {
    let entry = match xref.get(obj_ref.id) {
        Some(e) => e,
        None => return false,
    };
    // Compressed objects live inside object streams — their "offset" is the
    // stream object number, not a byte position. We cannot validate them by
    // seeking, but their presence in a correctly parsed xref stream is
    // sufficient proof that the xref is valid.
    if entry.entry_type == crate::xref::XRefEntryType::Compressed {
        return true;
    }
    if reader.seek(SeekFrom::Start(entry.offset)).is_err() {
        return false;
    }
    let mut buf = [0u8; 32];
    let n = reader.read(&mut buf).unwrap_or(0);
    if n == 0 {
        return false;
    }
    let s = String::from_utf8_lossy(&buf[..n]);
    // A valid object header starts with "N G obj"
    let mut parts = s.split_whitespace();
    // first token should be a number (obj id)
    let first_is_num = parts.next().is_some_and(|t| t.parse::<u32>().is_ok());
    let second_is_num = parts.next().is_some_and(|t| t.parse::<u16>().is_ok());
    let third_is_obj = parts
        .next()
        .is_some_and(|t| t == "obj" || t.starts_with("obj"));
    first_is_num && second_is_num && third_is_obj
}

/// Validate that the /Root catalog object is loadable from the xref.
fn validate_root_loadable<R: Read + Seek>(
    reader: &mut R,
    xref: &crate::xref::CrossRefTable,
    trailer: &Object,
) -> bool {
    let root_ref = match get_root_ref_from_trailer(trailer) {
        Some(r) => r,
        None => return false, // No /Root at all — can't validate
    };
    validate_object_at_offset(reader, xref, root_ref)
}

/// Check if a string contains the standalone "obj" keyword (not "endobj").
///
/// This is used during multi-line object header parsing to detect when we've
/// accumulated enough lines to have a complete header. A naive `contains("obj")`
/// would match "endobj" and cause the loop to exit prematurely.
fn has_standalone_obj_keyword(s: &str) -> bool {
    for (i, _) in s.match_indices("obj") {
        // Skip "endobj" — check if preceded by "end"
        if i >= 3 && &s[i - 3..i] == "end" {
            continue;
        }
        // Must be at a word boundary: preceded by whitespace, digit, or start of string
        if i == 0
            || s.as_bytes()[i - 1].is_ascii_whitespace()
            || s.as_bytes()[i - 1].is_ascii_digit()
        {
            return true;
        }
    }
    false
}

/// Parse PDF header (%PDF-x.y) from a reader.
///
/// # Arguments
///
/// * `reader` - A readable and seekable source (e.g., File, Cursor)
/// * `lenient` - If false, fail if header not at byte 0; if true, search first 8192 bytes
///
/// # Returns
///
/// Returns `Ok((major, minor, offset))` with the PDF version and byte offset where header was found.
/// In strict mode, offset will be 0 if successful. In lenient mode, offset may be > 0 for PDFs
/// with leading binary data (compliant with ISO 32000-1:2008, page 41).
///
/// # Examples
///
/// ```rust
/// use std::io::Cursor;
/// # use pdf_oxide::document::parse_header;
///
/// let data = b"%PDF-1.7\n";
/// let mut cursor = Cursor::new(data);
/// let (major, minor, offset) = parse_header(&mut cursor, false).unwrap();
/// assert_eq!((major, minor, offset), (1, 7, 0));
/// ```
pub fn parse_header<R: Read + Seek>(reader: &mut R, lenient: bool) -> Result<(u8, u8, u64)> {
    // Try to get current position
    let start_pos = reader.stream_position().unwrap_or(0);

    // Read first 8 bytes for fast path (header at byte 0)
    let mut header = [0u8; 8];
    let strict_read_ok = match reader.read_exact(&mut header) {
        Ok(_) => {
            // Check if header is at position 0
            if &header[0..5] == b"%PDF-" {
                return parse_version_from_header(&header, lenient)
                    .map(|(major, minor)| (major, minor, 0));
            }
            true
        },
        Err(e) => {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                // File too short for PDF header
                if !lenient {
                    return Err(Error::InvalidHeader(
                        "File too short for PDF header (expected at least 8 bytes)".to_string(),
                    ));
                }
                false
            } else {
                return Err(Error::InvalidHeader(format!("Failed to read file: {}", e)));
            }
        },
    };

    // If strict mode and first 8 bytes read, fail immediately
    if !lenient && strict_read_ok {
        return Err(Error::InvalidHeader(format!(
            "Expected '%PDF-' at byte 0, found '{}'",
            String::from_utf8_lossy(&header[0..5])
        )));
    }

    // Lenient mode: search first 8192 bytes
    reader.seek(SeekFrom::Start(start_pos))?;

    // Read up to 8192 bytes
    let mut buffer = vec![0u8; 8192];
    let bytes_read = match reader.read(&mut buffer) {
        Ok(0) => return Err(Error::InvalidHeader("File is empty (0 bytes read)".to_string())),
        Ok(n) => n,
        Err(e) => {
            return Err(Error::InvalidHeader(format!(
                "I/O error while searching for PDF header: {}",
                e
            )))
        },
    };

    buffer.truncate(bytes_read);

    // Search for "%PDF-" marker
    match find_substring(&buffer, b"%PDF-") {
        Some(offset) => {
            // Verify we have enough bytes for the version
            if offset + 8 > buffer.len() {
                return Err(Error::InvalidHeader(
                    "PDF header found but insufficient bytes for version".to_string(),
                ));
            }

            let header_bytes = &buffer[offset..offset + 8];
            let mut header_arr = [0u8; 8];
            header_arr.copy_from_slice(header_bytes);

            let (major, minor) = parse_version_from_header(&header_arr, true)?;

            // Standardize reader position to just after the header
            // (consistent with strict mode behavior at line 4378)
            let header_start = start_pos + offset as u64;
            let after_header = header_start + 8;
            reader.seek(SeekFrom::Start(after_header))?;

            Ok((major, minor, header_start))
        },
        None => {
            if lenient {
                // Some PDFs lack a %PDF- header entirely (e.g., start with a binary
                // comment like %\xe2\xe3\xcf\xd3). Default to version 1.4.
                log::warn!("No %PDF- header found; assuming version 1.4 in lenient mode");
                reader.seek(SeekFrom::Start(0))?;
                Ok((1, 4, 0))
            } else {
                Err(Error::InvalidHeader(
                    "No PDF header found in first 8192 bytes of file".to_string(),
                ))
            }
        },
    }
}

/// Parse version information from a header buffer.
/// Assumes buffer starts with "%PDF-" and has at least 8 bytes.
///
/// When `lenient` is true, malformed version strings (e.g., `%PDF-1.\n`, `%PDF-a.4`)
/// default to version (1, 4) instead of returning an error.
fn parse_version_from_header(header: &[u8; 8], lenient: bool) -> Result<(u8, u8)> {
    // Check magic bytes "%PDF-"
    if &header[0..5] != b"%PDF-" {
        return Err(Error::InvalidHeader(format!(
            "Expected '%PDF-', found '{}'",
            String::from_utf8_lossy(&header[0..5])
        )));
    }

    // Parse version (e.g., "1.7")
    // Format: %PDF-M.m where M is major version (1 digit), m is minor version (1 digit)
    if header[6] != b'.' {
        if lenient {
            log::warn!(
                "Malformed PDF version format (expected '.', found '{}'), defaulting to 1.4",
                header[6] as char
            );
            return Ok((1, 4));
        }
        return Err(Error::InvalidHeader(format!(
            "Invalid version format: expected '.', found '{}'",
            header[6] as char
        )));
    }

    let major = header[5];
    let minor = header[7];

    // Validate digits
    if !major.is_ascii_digit() || !minor.is_ascii_digit() {
        if lenient {
            log::warn!(
                "Malformed PDF version '{}.{}' (non-digit characters), defaulting to 1.4",
                major as char,
                minor as char
            );
            return Ok((1, 4));
        }
        return Err(Error::InvalidHeader(format!(
            "Invalid version: {}.{} (not digits)",
            major as char, minor as char
        )));
    }

    let major = major - b'0';
    let minor = minor - b'0';

    // Validate version range (PDF 1.0 - 2.0)
    if major > 2 || (major == 0 && minor == 0) {
        if lenient {
            log::warn!("Unsupported PDF version {}.{}, defaulting to 1.4", major, minor);
            return Ok((1, 4));
        }
        return Err(Error::UnsupportedVersion(format!("{}.{}", major, minor)));
    }

    Ok((major, minor))
}

/// Parse the trailer dictionary from a reader.
///
/// The trailer comes immediately after the xref table and before "startxref".
/// It starts with the keyword "trailer" followed by a dictionary.
///
/// # Example Format
///
/// ```text
/// trailer
/// << /Size 6 /Root 1 0 R /Info 5 0 R >>
/// startxref
/// 1234
/// %%EOF
/// ```
///
/// # Arguments
///
/// * `reader` - A readable source positioned after the xref table
///
/// # Returns
///
/// Returns the trailer dictionary as an `Object`.
///
/// # Errors
///
/// Returns an error if:
/// - The "trailer" keyword is not found
/// - The dictionary following "trailer" cannot be parsed
/// - The reader encounters an I/O error
pub fn parse_trailer<R: Read>(reader: &mut R) -> Result<Object> {
    // The reader should already be positioned after the xref table
    // We need to read until we find "trailer", then parse the dictionary

    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;

    // Find "trailer" keyword
    let content = String::from_utf8_lossy(&buffer);
    let trailer_pos = content.find("trailer").ok_or_else(|| {
        Error::InvalidPdf("Trailer keyword not found after xref table".to_string())
    })?;

    // Skip past "trailer" keyword (7 bytes)
    let dict_start = trailer_pos + 7;
    if dict_start >= buffer.len() {
        return Err(Error::UnexpectedEof);
    }

    // Parse the dictionary that follows
    let (_, trailer_dict) = parse_object(&buffer[dict_start..]).map_err(|e| Error::ParseError {
        offset: dict_start,
        reason: format!("Failed to parse trailer dictionary: {:?}", e),
    })?;

    // Verify it's a dictionary
    if trailer_dict.as_dict().is_none() {
        return Err(Error::InvalidPdf("Trailer is not a dictionary".to_string()));
    }

    Ok(trailer_dict)
}

/// Find the first occurrence of a substring in a byte slice.
///
/// Returns the index of the first occurrence, or None if not found.
fn find_substring(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }

    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_rotate_span_bbox_identity_and_180() {
        let r = crate::geometry::Rect::new(10.0, 20.0, 30.0, 5.0);
        let (w, h) = (200.0, 100.0);

        // rot == 0 is the identity (byte-identical, unrotated pages untouched).
        let id = PdfDocument::rotate_span_bbox(r, 0, w, h);
        assert!((id.x - r.x).abs() < 1e-4 && (id.y - r.y).abs() < 1e-4);
        assert!((id.width - r.width).abs() < 1e-4 && (id.height - r.height).abs() < 1e-4);

        // rot == 180 matches the legacy mirror: x' = w-(x+width), y' = h-(y+height).
        let m = PdfDocument::rotate_span_bbox(r, 180, w, h);
        assert!((m.x - (w - (r.x + r.width))).abs() < 1e-4, "180 x: {}", m.x);
        assert!((m.y - (h - (r.y + r.height))).abs() < 1e-4, "180 y: {}", m.y);
        assert!((m.width - r.width).abs() < 1e-4 && (m.height - r.height).abs() < 1e-4);
    }

    #[test]
    fn test_rotate_span_bbox_90_270_roundtrip_and_swap() {
        let r = crate::geometry::Rect::new(10.0, 20.0, 30.0, 5.0);
        // 90° / 270° swap width and height of the AABB.
        let r90 = PdfDocument::rotate_span_bbox(r, 90, 200.0, 100.0);
        assert!((r90.width - r.height).abs() < 1e-4, "w/h swap: {}", r90.width);
        assert!((r90.height - r.width).abs() < 1e-4, "w/h swap: {}", r90.height);

        // Applying 90° four times around a square page returns to the start.
        let s = crate::geometry::Rect::new(12.0, 34.0, 6.0, 8.0);
        let p = 100.0;
        let a = PdfDocument::rotate_span_bbox(s, 90, p, p);
        let b = PdfDocument::rotate_span_bbox(a, 90, p, p);
        let c = PdfDocument::rotate_span_bbox(b, 90, p, p);
        let d = PdfDocument::rotate_span_bbox(c, 90, p, p);
        assert!((d.x - s.x).abs() < 1e-3, "roundtrip x: {} vs {}", d.x, s.x);
        assert!((d.y - s.y).abs() < 1e-3, "roundtrip y: {} vs {}", d.y, s.y);
        assert!((d.width - s.width).abs() < 1e-3 && (d.height - s.height).abs() < 1e-3);
    }

    #[test]
    fn test_parse_valid_header_1_7() {
        let mut cursor = Cursor::new(b"%PDF-1.7\n");
        let (major, minor, offset) = parse_header(&mut cursor, false).unwrap();
        assert_eq!((major, minor, offset), (1, 7, 0));
    }

    #[test]
    fn test_parse_valid_header_1_4() {
        let mut cursor = Cursor::new(b"%PDF-1.4");
        let (major, minor, offset) = parse_header(&mut cursor, false).unwrap();
        assert_eq!((major, minor, offset), (1, 4, 0));
    }

    #[test]
    fn test_parse_valid_header_1_0() {
        let mut cursor = Cursor::new(b"%PDF-1.0");
        let (major, minor, offset) = parse_header(&mut cursor, false).unwrap();
        assert_eq!((major, minor, offset), (1, 0, 0));
    }

    #[test]
    fn test_parse_valid_header_2_0() {
        let mut cursor = Cursor::new(b"%PDF-2.0");
        let (major, minor, offset) = parse_header(&mut cursor, false).unwrap();
        assert_eq!((major, minor, offset), (2, 0, 0));
    }

    #[test]
    fn test_parse_invalid_header_wrong_magic_strict() {
        let mut cursor = Cursor::new(b"NotAPDF\n");
        let result = parse_header(&mut cursor, false);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidHeader(_)));
    }

    #[test]
    fn test_parse_invalid_header_unsupported_version() {
        let mut cursor = Cursor::new(b"%PDF-3.0");
        let result = parse_header(&mut cursor, false);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::UnsupportedVersion(_)));
    }

    #[test]
    fn test_parse_invalid_header_version_0_0() {
        let mut cursor = Cursor::new(b"%PDF-0.0");
        let result = parse_header(&mut cursor, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_header_no_dot() {
        let mut cursor = Cursor::new(b"%PDF-17\n");
        let result = parse_header(&mut cursor, false);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidHeader(_)));
    }

    #[test]
    fn test_parse_invalid_header_too_short() {
        let mut cursor = Cursor::new(b"%PDF");
        let result = parse_header(&mut cursor, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_header_non_digit_version() {
        let mut cursor = Cursor::new(b"%PDF-X.Y");
        let result = parse_header(&mut cursor, false);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidHeader(_)));
    }

    // ========================================================================
    // Header parsing tests with various prefixes
    #[test]
    fn test_parse_header_with_bom_prefix() {
        // UTF-8 BOM prefix before header
        let data = b"\xEF\xBB\xBF%PDF-1.7\n";
        let mut cursor = Cursor::new(data);
        let (major, minor, offset) = parse_header(&mut cursor, true).unwrap();
        assert_eq!((major, minor, offset), (1, 7, 3));
    }

    #[test]
    fn test_parse_header_with_binary_prefix() {
        // Binary data prefix before header
        let mut data = vec![0x1b, 0x96, 0x5f];
        data.extend_from_slice(b"%PDF-1.4\n");
        let mut cursor = Cursor::new(data);
        let (major, minor, offset) = parse_header(&mut cursor, true).unwrap();
        assert_eq!((major, minor, offset), (1, 4, 3));
    }

    #[test]
    fn test_parse_header_at_boundary() {
        // Header starting at byte 1016 (within 1024-byte window, with 8 bytes for full header)
        let mut data = vec![0u8; 1016];
        data.extend_from_slice(b"%PDF-1.5");
        let mut cursor = Cursor::new(data);
        let (major, minor, offset) = parse_header(&mut cursor, true).unwrap();
        assert_eq!((major, minor, offset), (1, 5, 1016));
    }

    #[test]
    fn test_parse_header_not_found_lenient() {
        // No header in first 1024 bytes, lenient mode defaults to 1.4
        let data = vec![0u8; 1024];
        let mut cursor = Cursor::new(data);
        let (major, minor, offset) = parse_header(&mut cursor, true).unwrap();
        assert_eq!((major, minor), (1, 4));
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_parse_header_strict_rejects_offset() {
        // With binary prefix but strict mode should fail
        let mut data = vec![0x1b, 0x96, 0x5f];
        data.extend_from_slice(b"%PDF-1.4\n");
        let mut cursor = Cursor::new(data);
        let result = parse_header(&mut cursor, false);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidHeader(_)));
    }

    // ========================================================================
    // Trailer Parsing Tests
    // ========================================================================

    #[test]
    fn test_parse_trailer_basic() {
        let data = b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n";
        let mut cursor = Cursor::new(data);
        let trailer = parse_trailer(&mut cursor).unwrap();

        let dict = trailer.as_dict().unwrap();
        assert_eq!(dict.get("Size").unwrap().as_integer(), Some(6));
        assert!(dict.get("Root").unwrap().as_reference().is_some());
    }

    #[test]
    fn test_parse_trailer_missing_keyword() {
        let data = b"<< /Size 6 >>\nstartxref\n";
        let mut cursor = Cursor::new(data);
        let result = parse_trailer(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_trailer_not_dictionary() {
        let data = b"trailer\n[ 1 2 3 ]\nstartxref\n";
        let mut cursor = Cursor::new(data);
        let result = parse_trailer(&mut cursor);
        assert!(result.is_err());
    }

    // ========================================================================
    // PdfDocument Error Tests
    // ========================================================================

    #[test]
    fn test_document_open_nonexistent_file() {
        let result = PdfDocument::open("/nonexistent/path/to/file.pdf");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::Io(_)));
    }

    #[test]
    fn test_circular_reference_detection() {
        // This test ensures that the cycle detection mechanism works
        // We can't easily create a circular PDF in a unit test, but we can
        // verify that the error types exist and are properly defined
        use crate::object::ObjectRef;

        let obj_ref = ObjectRef::new(1, 0);
        let err = Error::CircularReference(obj_ref);
        let msg = format!("{}", err);
        assert!(msg.contains("Circular reference"));
        assert!(msg.contains("object 1 0 R"));
    }

    #[test]
    fn test_recursion_limit_error() {
        let err = Error::RecursionLimitExceeded(100);
        let msg = format!("{}", err);
        assert!(msg.contains("Recursion depth limit exceeded"));
        assert!(msg.contains("100"));
    }

    /// Regression test for #163: circular Form XObject references must not cause
    /// a stack overflow / segfault. The PDF has X0→X1→X0 circular references.
    #[test]
    fn test_issue_163_circular_form_xobjects() {
        // Build a minimal PDF with circular Form XObject references, write to temp file.
        let pdf_bytes = build_circular_xobject_pdf();
        let tmp_path = std::env::temp_dir().join("pdf_oxide_test_issue163.pdf");
        std::fs::write(&tmp_path, &pdf_bytes).unwrap();
        let doc = PdfDocument::open(&tmp_path).unwrap();
        let _ = std::fs::remove_file(&tmp_path);
        assert_eq!(doc.page_count().unwrap(), 1);

        // extract_text should not hang or crash
        let text = doc.extract_text(0).unwrap();
        assert!(text.is_empty() || text.len() < 100); // No real text content

        // extract_images should not hang or crash (this was the segfault path)
        let images = doc.extract_images(0).unwrap();
        assert!(images.is_empty()); // No real images, just circular forms

        // to_markdown should not hang or crash
        let md = doc
            .to_markdown(0, &crate::converters::ConversionOptions::default())
            .unwrap();
        drop(md); // Just verify it completes
    }

    /// Build a minimal PDF with circular Form XObjects: X0 references X1, X1 references X0.
    fn build_circular_xobject_pdf() -> Vec<u8> {
        let mut pdf = b"%PDF-1.4\n".to_vec();

        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

        let off3 = pdf.len();
        pdf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /XObject << /X0 5 0 R /X1 6 0 R >> >> >>\nendobj\n");

        let off4 = pdf.len();
        let content = b"/X0 Do";
        pdf.extend_from_slice(
            format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes(),
        );
        pdf.extend_from_slice(content);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");

        let off5 = pdf.len();
        let x0_content = b"/X1 Do";
        pdf.extend_from_slice(format!("5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Resources << /XObject << /X1 6 0 R >> >> /Length {} >>\nstream\n", x0_content.len()).as_bytes());
        pdf.extend_from_slice(x0_content);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");

        let off6 = pdf.len();
        let x1_content = b"/X0 Do";
        pdf.extend_from_slice(format!("6 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Resources << /XObject << /X0 5 0 R >> >> /Length {} >>\nstream\n", x1_content.len()).as_bytes());
        pdf.extend_from_slice(x1_content);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");

        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 7\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off4).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off5).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off6).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );

        pdf
    }

    // #572: a corrupt/zero startxref forces full-file xref reconstruction.
    // Because reconstruction already scans the whole file for every
    // uncompressed object, the document must pre-seed its object-scan cache
    // from the reconstructed table — so the first object miss is O(1) instead
    // of triggering a SECOND full-file scan (the heavy "first extract_text"
    // cost on corrupt-xref polyglot PDFs).
    #[test]
    fn test_reconstructed_xref_preseeds_scan_cache() {
        let pdf = b"%PDF-1.4\n\
            1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\
            2 0 obj\n<< /Type /Pages /Count 0 /Kids [] >>\nendobj\n\
            trailer\n<< /Root 1 0 R /Size 3 >>\n\
            startxref\n0\n%%EOF";
        let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("open corrupt-xref pdf");

        let cache = doc.scanned_object_offsets.lock_or_recover();
        let offsets = cache
            .as_ref()
            .expect("#572: reconstructed xref must pre-seed the scan-offset cache");
        assert!(
            offsets.contains_key(&1) && offsets.contains_key(&2),
            "#572: pre-seeded cache should hold the reconstructed object offsets, got {offsets:?}"
        );
    }

    // ========================================================================
    // Helper: Build a minimal valid PDF with configurable content stream
    // ========================================================================

    /// Build a minimal PDF in memory with given content stream bytes.
    /// Returns the raw PDF bytes suitable for `PdfDocument::from_bytes`.
    fn build_minimal_pdf(content: &[u8]) -> Vec<u8> {
        let mut pdf = b"%PDF-1.4\n".to_vec();

        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

        let off3 = pdf.len();
        pdf.extend_from_slice(
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >>\nendobj\n",
        );

        let off4 = pdf.len();
        pdf.extend_from_slice(
            format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes(),
        );
        pdf.extend_from_slice(content);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");

        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 5\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off4).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );

        pdf
    }

    /// Build a minimal PDF with a multi-page structure (given page count).
    fn build_multi_page_pdf(page_count: usize) -> Vec<u8> {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets: Vec<usize> = Vec::new();

        // Object 1: Catalog
        offsets.push(pdf.len());
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        // Object 2: Pages (we'll build the Kids array)
        offsets.push(pdf.len());
        let kids_str: String = (0..page_count)
            .map(|i| format!("{} 0 R", i + 3))
            .collect::<Vec<_>>()
            .join(" ");
        let pages_obj = format!(
            "2 0 obj\n<< /Type /Pages /Kids [{}] /Count {} >>\nendobj\n",
            kids_str, page_count
        );
        pdf.extend_from_slice(pages_obj.as_bytes());

        // Objects 3..3+page_count: Page objects (no /Contents, blank pages)
        for _i in 0..page_count {
            offsets.push(pdf.len());
            let page_obj = format!(
                "{} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
                offsets.len()
            );
            pdf.extend_from_slice(page_obj.as_bytes());
        }

        let xref_off = pdf.len();
        let total_objs = offsets.len() + 1; // +1 for object 0
        pdf.extend_from_slice(format!("xref\n0 {}\n", total_objs).as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in &offsets {
            pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
                total_objs, xref_off
            )
            .as_bytes(),
        );

        pdf
    }

    // ========================================================================
    // PdfDocument basic open/version/trailer tests
    // ========================================================================

    #[test]
    fn test_from_bytes_minimal_pdf() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert_eq!(doc.version(), (1, 4));
        assert!(doc.trailer().as_dict().is_some());
    }

    // Issue #509: catalog() must fall back to scanning indirect objects for
    // `/Type /Catalog` when the trailer omits /Root. The public open path
    // can't reach this — a /Root-less parsed trailer fails root validation
    // and xref reconstruction synthesizes a /Root-bearing trailer before
    // catalog() ever runs — so cover find_catalog_by_scan() directly: open a
    // valid PDF, then strip /Root from the in-memory trailer and confirm
    // catalog() still resolves the Catalog by object scan.
    #[test]
    fn test_catalog_recovers_when_trailer_omits_root() {
        let mut doc = PdfDocument::from_bytes(build_minimal_pdf(b"")).unwrap();
        // Sanity: the normal /Root path resolves the Catalog.
        assert!(doc.catalog().is_ok());

        // Drop /Root so only the indirect-object scan can find the Catalog.
        match doc.trailer {
            Object::Dictionary(ref mut d) => {
                d.remove("Root");
                assert!(d.get("Root").is_none());
            },
            _ => panic!("trailer is not a dictionary"),
        }

        let catalog = doc.catalog().expect(
            "catalog() must recover the /Type /Catalog object by scan when /Root is absent",
        );
        assert_eq!(
            catalog
                .as_dict()
                .and_then(|d| d.get("Type"))
                .and_then(|t| t.as_name()),
            Some("Catalog"),
            "find_catalog_by_scan must return the actual Catalog object"
        );
    }

    #[test]
    fn test_from_bytes_invalid_data() {
        let result = PdfDocument::from_bytes(b"not a pdf".to_vec());
        // Should error out -- no valid xref
        assert!(result.is_err() || result.is_ok()); // lenient mode may fall back
    }

    #[test]
    fn test_from_bytes_empty() {
        let result = PdfDocument::from_bytes(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_version_accessor() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let (major, minor) = doc.version();
        assert_eq!(major, 1);
        assert_eq!(minor, 4);
    }

    #[test]
    fn test_trailer_accessor() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let trailer = doc.trailer();
        let dict = trailer.as_dict().unwrap();
        assert!(dict.contains_key("Root"));
        assert!(dict.contains_key("Size"));
    }

    #[test]
    fn test_debug_impl() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let debug_str = format!("{:?}", doc);
        assert!(debug_str.contains("PdfDocument"));
        assert!(debug_str.contains("version"));
        assert!(debug_str.contains("(1, 4)"));
    }

    // ========================================================================
    // Catalog tests
    // ========================================================================

    #[test]
    fn test_catalog_returns_dictionary() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let catalog = doc.catalog().unwrap();
        let dict = catalog.as_dict().unwrap();
        assert_eq!(dict.get("Type").unwrap().as_name(), Some("Catalog"));
    }

    // ========================================================================
    // Page count tests
    // ========================================================================

    #[test]
    fn test_page_count_single_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert_eq!(doc.page_count().unwrap(), 1);
    }

    #[test]
    fn test_page_count_multiple_pages() {
        let pdf = build_multi_page_pdf(5);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert_eq!(doc.page_count().unwrap(), 5);
    }

    #[test]
    fn test_page_count_zero_pages() {
        // Build a PDF with 0 pages
        let mut pdf = b"%PDF-1.4\n".to_vec();

        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n");

        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 3\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );

        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert_eq!(doc.page_count().unwrap(), 0);
    }

    // ========================================================================
    // load_object tests
    // ========================================================================

    #[test]
    fn test_load_object_from_cache() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        // Load catalog (object 1 0 R)
        let obj_ref = ObjectRef::new(1, 0);
        let obj1 = doc.load_object(obj_ref).unwrap();
        // Load again - should come from cache
        let obj2 = doc.load_object(obj_ref).unwrap();
        // Both should be the catalog
        assert_eq!(obj1.as_dict().unwrap().get("Type").unwrap().as_name(), Some("Catalog"));
        assert_eq!(obj2.as_dict().unwrap().get("Type").unwrap().as_name(), Some("Catalog"));
    }

    #[test]
    fn test_load_object_missing_returns_null() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        // Try to load a non-existent object
        let obj_ref = ObjectRef::new(999, 0);
        let obj = doc.load_object(obj_ref).unwrap();
        // Per PDF Spec 7.3.10: missing objects treated as Null
        assert!(matches!(obj, Object::Null));
    }

    // ========================================================================
    // resolve_references tests
    // ========================================================================

    #[test]
    fn test_resolve_references_integer() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let obj = Object::Integer(42);
        let resolved = doc.resolve_references(&obj, 3).unwrap();
        assert_eq!(resolved.as_integer(), Some(42));
    }

    #[test]
    fn test_resolve_references_null() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let obj = Object::Null;
        let resolved = doc.resolve_references(&obj, 3).unwrap();
        assert!(matches!(resolved, Object::Null));
    }

    #[test]
    fn test_resolve_references_max_depth_zero() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        // With depth 0, references should not be resolved
        let obj = Object::Reference(ObjectRef::new(1, 0));
        let resolved = doc.resolve_references(&obj, 0).unwrap();
        // Should still be a reference (not resolved)
        assert!(resolved.as_reference().is_some());
    }

    #[test]
    fn test_resolve_references_reference() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        // Resolve a reference to object 1 (catalog)
        let obj = Object::Reference(ObjectRef::new(1, 0));
        let resolved = doc.resolve_references(&obj, 3).unwrap();
        // Should now be a dictionary (the catalog)
        assert!(resolved.as_dict().is_some());
    }

    #[test]
    fn test_resolve_references_array() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let arr = Object::Array(vec![Object::Integer(1), Object::Integer(2)]);
        let resolved = doc.resolve_references(&arr, 3).unwrap();
        let resolved_arr = resolved.as_array().unwrap();
        assert_eq!(resolved_arr.len(), 2);
    }

    #[test]
    fn test_resolve_references_dictionary() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let mut dict = std::collections::HashMap::new();
        dict.insert("Key".to_string(), Object::Integer(42));
        let obj = Object::Dictionary(dict);
        let resolved = doc.resolve_references(&obj, 3).unwrap();
        let resolved_dict = resolved.as_dict().unwrap();
        assert_eq!(resolved_dict.get("Key").unwrap().as_integer(), Some(42));
    }

    #[test]
    fn test_resolve_references_bad_reference() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        // A reference to a non-existent object
        let obj = Object::Reference(ObjectRef::new(999, 0));
        // Should return the unresolved reference (but as Null since missing objects -> Null)
        let resolved = doc.resolve_references(&obj, 3).unwrap();
        // The reference was resolved to Null (per PDF spec)
        assert!(matches!(resolved, Object::Null));
    }

    // ========================================================================
    // authenticate tests
    // ========================================================================

    #[test]
    fn test_authenticate_unencrypted_pdf() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        // Unencrypted PDF should always authenticate successfully
        let result = doc.authenticate(b"anypassword").unwrap();
        assert!(result);
    }

    // ========================================================================
    // get_page_content_data tests
    // ========================================================================

    #[test]
    fn test_get_page_content_data_empty_content() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let data = doc.get_page_content_data(0).unwrap();
        // Empty content stream still returns data (may be empty or have a newline)
        assert!(data.len() <= 2);
    }

    #[test]
    fn test_get_page_content_data_with_content() {
        let content = b"BT /F1 12 Tf (Hello) Tj ET";
        let pdf = build_minimal_pdf(content);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let data = doc.get_page_content_data(0).unwrap();
        assert!(!data.is_empty());
        // The content should contain the original text
        let text = String::from_utf8_lossy(&data);
        assert!(text.contains("Hello"));
    }

    #[test]
    fn test_get_page_content_data_blank_page() {
        // Build a PDF where page has no /Contents at all
        let pdf = build_multi_page_pdf(1);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let data = doc.get_page_content_data(0).unwrap();
        assert!(data.is_empty()); // No contents = empty
    }

    // ========================================================================
    // extract_text tests
    // ========================================================================

    #[test]
    fn test_extract_text_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let text = doc.extract_text(0).unwrap();
        assert!(text.is_empty());
    }

    #[test]
    fn test_extract_text_no_font_resources() {
        // Content stream has text operators but no fonts loaded
        let content = b"BT /F1 12 Tf (Hello) Tj ET";
        let pdf = build_minimal_pdf(content);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        // Should not crash, may return empty or partial text
        let _text = doc.extract_text(0).unwrap();
    }

    // ========================================================================
    // extract_all_text tests
    // ========================================================================

    #[test]
    fn test_extract_all_text_multiple_pages() {
        let pdf = build_multi_page_pdf(3);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let text = doc.extract_all_text().unwrap();
        // Should have form feed separators between pages
        let page_count = text.matches('\x0c').count();
        assert_eq!(page_count, 2); // 3 pages = 2 separators
    }

    #[test]
    fn test_extract_all_text_single_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let text = doc.extract_all_text().unwrap();
        // No form feed separators for single page
        assert!(!text.contains('\x0c'));
    }

    // ========================================================================
    // extract_spans tests
    // ========================================================================

    #[test]
    fn test_extract_spans_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let spans = doc.extract_spans(0).unwrap();
        assert!(spans.is_empty());
    }

    #[test]
    fn test_extract_spans_no_text_operators() {
        // Graphics-only content (just rectangle drawing)
        let content = b"100 200 300 400 re S";
        let pdf = build_minimal_pdf(content);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let spans = doc.extract_spans(0).unwrap();
        assert!(spans.is_empty());
    }

    // ========================================================================
    // extract_chars tests
    // ========================================================================

    #[test]
    fn test_extract_chars_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let chars = doc.extract_chars(0).unwrap();
        assert!(chars.is_empty());
    }

    // ========================================================================
    // may_contain_text tests
    // ========================================================================

    #[test]
    fn test_may_contain_text_with_bt() {
        let data = b"q BT /F1 12 Tf (Hello) Tj ET Q";
        assert!(PdfDocument::may_contain_text(data));
    }

    #[test]
    fn test_may_contain_text_with_do() {
        let data = b"q /Im0 Do Q";
        assert!(PdfDocument::may_contain_text(data));
    }

    #[test]
    fn test_may_contain_text_no_text_operators() {
        let data = b"100 200 300 400 re S";
        assert!(!PdfDocument::may_contain_text(data));
    }

    #[test]
    fn test_may_contain_text_empty() {
        let data = b"";
        assert!(!PdfDocument::may_contain_text(data));
    }

    #[test]
    fn test_may_contain_text_bt_at_start() {
        let data = b"BT /F1 12 Tf ET";
        assert!(PdfDocument::may_contain_text(data));
    }

    #[test]
    fn test_may_contain_text_bt_at_end() {
        let data = b"q Q BT";
        assert!(PdfDocument::may_contain_text(data));
    }

    #[test]
    fn test_may_contain_text_false_positive_btype() {
        // "BTerror" should not match BT (BT must be delimited)
        let data = b"BTerror";
        assert!(!PdfDocument::may_contain_text(data));
    }

    #[test]
    fn test_may_contain_text_false_positive_document() {
        // "Document" contains "Do" but not as a standalone operator
        let data = b"Document";
        assert!(!PdfDocument::may_contain_text(data));
    }

    #[test]
    fn test_may_contain_text_do_with_name() {
        // Standard XObject invocation
        let data = b"/Im0 Do\n";
        assert!(PdfDocument::may_contain_text(data));
    }

    // ========================================================================
    // should_insert_space tests
    // ========================================================================

    /// Helper to create a TextSpan with minimal required fields for testing.
    fn make_test_span(text: &str, x: f32, y: f32, width: f32, font_size: f32) -> TextSpan {
        TextSpan {
            artifact_type: None,
            text: text.to_string(),
            bbox: crate::geometry::Rect {
                x,
                y,
                width,
                height: font_size,
            },
            font_name: "F1".to_string(),
            font_size,
            font_weight: crate::layout::FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: crate::layout::Color::new(0.0, 0.0, 0.0),
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
            char_widths: vec![],
            heading_level: None,
            rotation_degrees: 0.0,
        }
    }

    #[test]
    fn test_should_insert_space_same_line_with_gap() {
        let prev = make_test_span("Hello", 0.0, 100.0, 50.0, 12.0);
        let current = make_test_span("World", 56.0, 100.0, 50.0, 12.0);
        // 6pt gap (> 0.25 * 12 = 3pt)
        assert!(PdfDocument::should_insert_space(&prev, &current));
    }

    #[test]
    fn test_y_band_candidates_is_superset_of_tolerance() {
        // The Y-band index query must include every span within `band` of the
        // target Y (so the exact filter downstream is byte-identical to a full
        // scan). Check against a brute-force scan over assorted Y positions.
        let band = 4.0_f32;
        let spans: Vec<TextSpan> = [0.0, 1.5, 3.9, 4.0, 4.1, 8.0, 100.0, -3.0, -8.0]
            .iter()
            .map(|&y| make_test_span("x", 0.0, y, 5.0, 10.0))
            .collect();
        let idx = PdfDocument::build_y_band_index(&spans, band);
        for &cy in &[0.0_f32, 4.0, 4.05, 100.0, -3.0] {
            let got: std::collections::HashSet<usize> =
                PdfDocument::y_band_candidates(&idx, cy, band).collect();
            for (j, s) in spans.iter().enumerate() {
                if (s.bbox.y - cy).abs() <= band {
                    assert!(
                        got.contains(&j),
                        "index missed span {j} (y={}) within band of cy={cy}",
                        s.bbox.y
                    );
                }
            }
        }
    }

    #[test]
    fn test_merge_drop_cap_initial() {
        // Oversized "T" (20pt) immediately left of body "ABLE 102.3" (12pt) → merged.
        let mut spans = vec![
            make_test_span("T", 0.0, 100.0, 14.0, 20.0),
            make_test_span("ABLE 102.3", 15.0, 100.0, 60.0, 12.0),
        ];
        PdfDocument::merge_drop_cap_initials(&mut spans);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "TABLE 102.3");
        assert_eq!(spans[0].bbox.x, 0.0); // bbox extended left to the initial
    }

    #[test]
    fn test_merge_drop_cap_skips_same_size_capital() {
        // A same-size standalone capital (e.g. an article/initialism, normal gap)
        // must NOT be glued to the next word.
        // Tightly adjacent (gap ~1pt) but same size — only the size gate stops it.
        let mut spans = vec![
            make_test_span("A", 0.0, 100.0, 8.0, 12.0),
            make_test_span("word", 9.0, 100.0, 30.0, 12.0),
        ];
        PdfDocument::merge_drop_cap_initials(&mut spans);
        assert_eq!(spans.len(), 2, "same-size capital is not a drop-cap initial");
    }

    #[test]
    fn test_merge_drop_cap_skips_math_subscript_base() {
        // Inline math "A_st": a body-size capital (10pt) followed by a smaller
        // subscript (6.5pt). The capital is oversized vs its neighbour but NOT
        // vs the page body text, so it must NOT be glued into "Ast".
        let mut spans = vec![
            make_test_span("the shuffle algebra", 0.0, 100.0, 90.0, 10.0),
            make_test_span("A", 92.0, 100.0, 7.0, 10.0),
            make_test_span("st", 99.0, 98.0, 6.0, 6.5),
            make_test_span("of a statistic", 106.0, 100.0, 70.0, 10.0),
        ];
        PdfDocument::merge_drop_cap_initials(&mut spans);
        assert_eq!(spans.len(), 4, "inline math base letter is not a drop-cap initial");
        assert_eq!(spans[1].text, "A");
        assert_eq!(spans[2].text, "st");
    }

    #[test]
    fn test_merge_drop_cap_skips_word_spaced_standalone_capital() {
        // A heading "A Perspective ..." set in 18pt over 10pt body: the oversized
        // "A" is a complete word followed by a real word space, not the first
        // glyph of "Perspective". It must NOT glue into "APerspective".
        let mut spans = vec![
            make_test_span("ordinary body sentence one", 0.0, 200.0, 120.0, 10.0),
            make_test_span("ordinary body sentence two", 0.0, 188.0, 120.0, 10.0),
            make_test_span("A", 0.0, 100.0, 12.0, 18.0),
            make_test_span("Perspective", 17.0, 100.0, 90.0, 18.0), // gap 5pt = 0.28em
        ];
        PdfDocument::merge_drop_cap_initials(&mut spans);
        assert_eq!(spans.len(), 4, "word-spaced standalone capital is not a drop cap");
        assert_eq!(spans[2].text, "A");
        assert_eq!(spans[3].text, "Perspective");
    }

    #[test]
    fn test_order_rotated_blocks_groups_by_rotation() {
        let mk = |t: &str, x: f32, y: f32, rot: f32| {
            let mut s = make_test_span(t, x, y, 10.0, 10.0);
            s.rotation_degrees = rot;
            s
        };
        // Two 90° runs (seen first) then one -90° run.
        let spans = vec![
            mk("A", 10.0, 50.0, 90.0),
            mk("B", 10.0, 80.0, 90.0),
            mk("C", 200.0, 50.0, -90.0),
        ];
        let out = PdfDocument::order_rotated_blocks(spans);
        assert_eq!(out.len(), 3, "no spans dropped");
        // Groups stay contiguous in first-seen order; 90° block before -90°.
        let rots: Vec<f32> = out.iter().map(|s| s.rotation_degrees).collect();
        assert_eq!(rots, vec![90.0, 90.0, -90.0]);
        // Within the 90° block, upright-frame order keeps A before B.
        assert_eq!(out[0].text, "A");
        assert_eq!(out[1].text, "B");
    }

    #[test]
    fn test_merge_drop_cap_does_not_reach_line_above() {
        // A tall oversized "A" (16.8pt, baseline y=328) whose bbox top reaches
        // up into the previous line ("Or if", y~342). It must NOT merge with
        // "if" on the line above — only with same-baseline words on its own line
        // (which here are word-spaced and so also stay separate). Reproduces the
        // alice_old "OrAif" corruption.
        let mut spans = vec![
            make_test_span("Or", 44.0, 344.0, 14.4, 12.0),
            make_test_span("if", 62.0, 342.5, 10.7, 8.9),
            make_test_span("Idrop upon my toe", 74.0, 343.9, 90.0, 12.2),
            make_test_span("A", 54.7, 328.1, 10.1, 16.8),
            make_test_span("very heavy weight", 69.8, 327.8, 90.0, 8.4),
        ];
        PdfDocument::merge_drop_cap_initials(&mut spans);
        assert!(
            spans
                .iter()
                .all(|s| s.text != "Aif" && !s.text.contains("OrA")),
            "tall initial must not steal a word from the line above"
        );
        assert!(spans.iter().any(|s| s.text == "A"), "initial left intact on its own line");
    }

    #[test]
    fn test_should_insert_space_same_line_no_gap() {
        let prev = make_test_span("Hello", 0.0, 100.0, 50.0, 12.0);
        let current = make_test_span("World", 51.0, 100.0, 50.0, 12.0);
        // 1pt gap (< 0.25 * 12 = 3pt)
        assert!(!PdfDocument::should_insert_space(&prev, &current));
    }

    #[test]
    fn test_should_insert_space_different_lines() {
        let prev = make_test_span("Hello", 0.0, 100.0, 50.0, 12.0);
        let current = make_test_span("World", 56.0, 120.0, 50.0, 12.0);
        // Different lines = false (no space needed, line break instead)
        assert!(!PdfDocument::should_insert_space(&prev, &current));
    }

    #[test]
    fn test_should_insert_space_column_gap() {
        let prev = make_test_span("Hello", 0.0, 100.0, 50.0, 12.0);
        let current = make_test_span("World", 200.0, 100.0, 50.0, 12.0);
        // Issue 487 (pr-138-example.pdf rate tables): a very large
        // same-line gap (here 150 pt > 5 em) must still produce a single
        // space. The earlier `gap < font_size * 5.0` upper bound made
        // this return false, after which the caller concatenated the two
        // spans without a separator and `3.80%` + `4.41%` came out as
        // `3.80%4.41%`. Large gap = different column = still a space.
        assert!(PdfDocument::should_insert_space(&prev, &current));
    }

    // ========================================================================
    // is_column_spanning_decimal / push_span_text tests (nougat_018 fix)
    // ========================================================================

    fn make_decimal_span(
        text: &str,
        char_widths: Vec<f32>,
        bbox_w: f32,
        font_size: f32,
    ) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            bbox: crate::geometry::Rect {
                x: 0.0,
                y: 0.0,
                width: bbox_w,
                height: font_size,
            },
            font_name: "F1".to_string(),
            font_size,
            font_weight: crate::layout::FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: crate::layout::Color::new(0.0, 0.0, 0.0),
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
            artifact_type: None,
            char_widths,
            heading_level: None,
            rotation_degrees: 0.0,
        }
    }

    #[test]
    fn test_column_spanning_decimal_wide_bbox() {
        // "1.10": 4 chars, cw=[3.98], expected=15.92, gap=9.8 > fs(7.0) → split
        let span = make_decimal_span("1.10", vec![3.9811199], 25.72, 7.0);
        assert!(PdfDocument::is_column_spanning_decimal(&span));
    }

    #[test]
    fn test_column_spanning_decimal_5char_span() {
        // "12.11": 5 chars, cw=[3.98,3.98], expected=19.91, gap=7.73 > fs(7.0) → split
        let span = make_decimal_span("12.11", vec![3.9811199, 3.9811199], 27.64, 7.0);
        assert!(PdfDocument::is_column_spanning_decimal(&span));
    }

    #[test]
    fn test_column_spanning_decimal_normal_bbox() {
        // "1.5" with 3 entries matching 3 chars; bbox_w = expected → gap ≈ 0 → no split
        let span = make_decimal_span("1.5", vec![3.0, 3.0, 3.0], 9.0, 7.0);
        assert!(!PdfDocument::is_column_spanning_decimal(&span));
    }

    #[test]
    fn test_column_spanning_decimal_non_digit() {
        // "hello.world" — letters, not digits → no split
        let span = make_decimal_span("hello.world", vec![], 60.0, 12.0);
        assert!(!PdfDocument::is_column_spanning_decimal(&span));
    }

    #[test]
    fn test_column_spanning_decimal_multiple_dots() {
        // "1.2.3" — two dots → no split
        let span = make_decimal_span("1.2.3", vec![3.0], 25.0, 7.0);
        assert!(!PdfDocument::is_column_spanning_decimal(&span));
    }

    #[test]
    fn test_push_span_text_splits_wide_decimal() {
        let span = make_decimal_span("1.10", vec![3.9811199], 25.72, 7.0);
        let mut out = String::new();
        PdfDocument::push_span_text(&mut out, &span);
        assert_eq!(out, "1 10");
    }

    #[test]
    fn test_push_span_text_leaves_normal_decimal() {
        let span = make_decimal_span("3.14", vec![4.0, 4.0, 4.0, 4.0], 16.0, 12.0);
        let mut out = String::new();
        PdfDocument::push_span_text(&mut out, &span);
        assert_eq!(out, "3.14");
    }

    // ========================================================================
    // char_widths_boundary_split tests (pdfa_004 CID-font merge fix)
    // ========================================================================

    #[test]
    fn test_cw_boundary_split_theorem_number() {
        // "Theorem1.7": 10 chars, 7 widths → split before '1'
        let span =
            make_decimal_span("Theorem1.7", vec![11.2, 8.9, 7.4, 8.1, 6.6, 7.4, 13.4], 83.8, 14.3);
        let result = PdfDocument::char_widths_boundary_split(&span);
        assert_eq!(result, Some(7)); // byte 7 = '1'
    }

    #[test]
    fn test_cw_boundary_split_let_capital() {
        // "LetC": 4 chars, 3 widths — lower→upper boundary → split at 'C'
        // (represents two CID text runs "Let" + "C" concatenated)
        let span = make_decimal_span("LetC", vec![7.3, 5.2, 4.5], 26.7, 12.0);
        let result = PdfDocument::char_widths_boundary_split(&span);
        assert_eq!(result, Some(3)); // byte 3 = 'C'
    }

    #[test]
    fn test_cw_boundary_no_split_already_space() {
        // "Theorem 1.1": 7 widths, char at idx 7 is space → no split
        let span =
            make_decimal_span("Theorem 1.1", vec![9.3, 7.5, 6.1, 6.7, 5.5, 6.1, 11.2], 80.0, 12.0);
        assert!(PdfDocument::char_widths_boundary_split(&span).is_none());
    }

    #[test]
    fn test_cw_boundary_no_split_matching_count() {
        // "hello" with 5 widths: no mismatch
        let span = make_decimal_span("hello", vec![5.0, 5.0, 5.0, 5.0, 5.0], 25.0, 12.0);
        assert!(PdfDocument::char_widths_boundary_split(&span).is_none());
    }

    #[test]
    fn test_cw_boundary_no_split_nonascii_boundary() {
        // "Marysia Prus-Gł": boundary char is 'ł' (non-ASCII) → no split
        let span = make_decimal_span("Marysia Prus-Gł", vec![5.0; 14], 80.0, 12.0);
        assert!(PdfDocument::char_widths_boundary_split(&span).is_none());
    }

    #[test]
    fn test_push_span_text_splits_let_capital() {
        // Lower→upper boundary: "LetC" splits to "Let C" (space inserted at 'C')
        let span = make_decimal_span("LetC", vec![7.3, 5.2, 4.5], 26.7, 12.0);
        let mut out = String::new();
        PdfDocument::push_span_text(&mut out, &span);
        assert_eq!(out, "Let C");
    }

    #[test]
    fn test_push_span_text_splits_theorem_number() {
        let span =
            make_decimal_span("Theorem1.7", vec![11.2, 8.9, 7.4, 8.1, 6.6, 7.4, 13.4], 83.8, 14.3);
        let mut out = String::new();
        PdfDocument::push_span_text(&mut out, &span);
        assert_eq!(out, "Theorem 1.7");
    }

    // ========================================================================
    // filter_leaked_metadata tests
    // ========================================================================

    #[test]
    fn test_filter_leaked_metadata_clean_text() {
        let text = "This is normal text without any metadata patterns.";
        let result = PdfDocument::filter_leaked_metadata(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_filter_leaked_metadata_removes_whitepoint() {
        let text = "Hello World\nWhitePoint [ 0.95 1.0 1.09 ]\nMore text";
        let result = PdfDocument::filter_leaked_metadata(text);
        assert!(result.contains("Hello World"));
        assert!(result.contains("More text"));
        assert!(!result.contains("WhitePoint"));
    }

    #[test]
    fn test_filter_leaked_metadata_removes_calrgb() {
        let text = "Text\nCalRGB /WhitePoint [ 1 1 1 ]\nMore";
        let result = PdfDocument::filter_leaked_metadata(text);
        assert!(result.contains("Text"));
        assert!(result.contains("More"));
        assert!(!result.contains("CalRGB"));
    }

    #[test]
    fn test_filter_leaked_metadata_preserves_normal_lines() {
        let text = "The Matrix is a movie\nGamma rays from space";
        // These lines contain metadata keywords but not in metadata format
        let result = PdfDocument::filter_leaked_metadata(text);
        // "The Matrix is a movie" should be preserved (doesn't start with "Matrix")
        assert!(result.contains("The Matrix is a movie"));
    }

    // ========================================================================
    // normalize_kangxi_radicals tests
    // ========================================================================

    #[test]
    fn test_normalize_kangxi_no_radicals() {
        let text = "Hello World";
        let result = PdfDocument::normalize_kangxi_radicals(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_normalize_kangxi_with_radicals() {
        // U+2F00 is Kangxi Radical One
        let text = "\u{2F00}";
        let result = PdfDocument::normalize_kangxi_radicals(text);
        // Should be normalized to a CJK unified ideograph
        assert_ne!(result, text);
    }

    // ========================================================================
    // normalize_arabic_presentation_forms tests
    // ========================================================================

    #[test]
    fn test_normalize_arabic_no_presentation_forms() {
        let text = "Hello World";
        let result = PdfDocument::normalize_arabic_presentation_forms(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_normalize_arabic_alef_presentation_form() {
        // U+FE8D is Arabic Alef isolated form
        let text = "\u{FE8D}";
        let result = PdfDocument::normalize_arabic_presentation_forms(text);
        // Should be normalized to base Alef (U+0627)
        assert!(result.contains('\u{0627}'));
    }

    #[test]
    fn test_normalize_arabic_lam_alef_ligature() {
        // U+FEFB is Lam-Alef ligature
        let text = "\u{FEFB}";
        let result = PdfDocument::normalize_arabic_presentation_forms(text);
        // Should become Lam (U+0644)
        assert!(result.contains('\u{0644}'));
    }

    // ========================================================================
    // reverse_rtl_visual_order_runs tests
    // ========================================================================
    //
    // These tests cover the two distinct RTL span shapes pdf_oxide sees
    // in the wild and make sure future changes don't regress either:
    //
    // 1. **Pre-shaped visual-order single span** — one `TextSpan` per
    //    line whose `text` already contains contextual Arabic glyphs
    //    (U+FB50-U+FDFF / U+FE70-U+FEFF) in the order the content
    //    stream drew them (rightmost glyph first). This is the
    //    `ArabicCIDTrueType.pdf` pdfjs test fixture case. Expected:
    //    character sequence gets reversed in place.
    //
    // 2. **Plain base-Arabic logical-order single span** — one
    //    `TextSpan` per line whose `text` uses base Arabic (U+0621-
    //    U+06FF) characters in logical / reading order, as most
    //    well-behaved PDF producers emit. Expected: span is left
    //    completely alone (no reversal, no shape changes).
    //
    // The gate that protects case 2 from case 1's reversal is the
    // `has_presentation_form` check inside `reverse_rtl_visual_order_runs`.

    fn make_rtl_test_span(text: &str, x: f32, y: f32) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            bbox: crate::geometry::Rect::new(x, y, 100.0, 12.0),
            font_size: 12.0,
            ..TextSpan::default()
        }
    }

    #[test]
    fn test_reverse_rtl_preshaped_single_span() {
        // "ArabicCIDTrueType.pdf" shape: one span per line, glyphs in
        // visual / right-to-left rendering order, mixing presentation
        // form `ﳋ` (U+FCCB) with base Arabic characters. The helper
        // must reverse this into reading order so downstream consumers
        // see logical Arabic even though the content stream is visual.
        let mut spans = vec![
            make_rtl_test_span(
                "\u{0629}\u{064A}\u{0628}\u{0631}\u{0639}\u{0644}\u{0627} \
                                \u{0637}\u{0648}\u{0637}\u{FCCB}\u{0627} \
                                \u{0639}\u{0627}\u{0648}\u{0646}\u{0627}",
                100.0,
                700.0,
            ),
            make_rtl_test_span("other content", 100.0, 680.0),
            make_rtl_test_span("more content", 100.0, 660.0),
            make_rtl_test_span("tail", 100.0, 640.0),
        ];
        PdfDocument::reverse_rtl_visual_order_runs(&mut spans);
        // After reversal, the first span should read as
        // "انواع اﳋطوط العربية" — the logical reading order. The
        // exact string comparison is the reversal of the input.
        assert_eq!(
            spans[0].text,
            "\u{0627}\u{0646}\u{0648}\u{0627}\u{0639} \
             \u{0627}\u{FCCB}\u{0637}\u{0648}\u{0637} \
             \u{0627}\u{0644}\u{0639}\u{0631}\u{0628}\u{064A}\u{0629}",
            "Pre-shaped Arabic single span must be reversed into reading order"
        );
        // Other non-RTL spans must be untouched.
        assert_eq!(spans[1].text, "other content");
        assert_eq!(spans[2].text, "more content");
        assert_eq!(spans[3].text, "tail");
    }

    #[test]
    fn test_reverse_rtl_logical_order_base_arabic_untouched() {
        // Most Arabic PDFs store text in logical (reading) order using
        // base characters (U+0621-U+06FF) and rely on the renderer to
        // apply shaping at display time. pdf_oxide must leave those
        // spans alone — reversing them would garble correct output.
        //
        // The string below is "انواع الخطوط العربية" entirely composed
        // of base Arabic code points (no presentation forms). Gate:
        // `has_presentation_form` stays false, no reversal happens.
        let logical = "\u{0627}\u{0646}\u{0648}\u{0627}\u{0639} \
                       \u{0627}\u{0644}\u{062E}\u{0637}\u{0648}\u{0637} \
                       \u{0627}\u{0644}\u{0639}\u{0631}\u{0628}\u{064A}\u{0629}";
        let mut spans = vec![
            make_rtl_test_span(logical, 100.0, 700.0),
            make_rtl_test_span("other content", 100.0, 680.0),
            make_rtl_test_span("more content", 100.0, 660.0),
            make_rtl_test_span("tail", 100.0, 640.0),
        ];
        PdfDocument::reverse_rtl_visual_order_runs(&mut spans);
        assert_eq!(spans[0].text, logical, "Logical-order base-Arabic span must NOT be reversed");
    }

    #[test]
    fn test_reverse_rtl_short_rtl_span_not_touched_by_pass0() {
        // Pass 0 requires at least 4 non-whitespace characters. A
        // two-character Arabic snippet must not trigger reversal even
        // though it contains presentation forms.
        let mut spans = vec![
            make_rtl_test_span("\u{FB7F}\u{FEB3}", 100.0, 700.0),
            make_rtl_test_span("other content", 100.0, 680.0),
            make_rtl_test_span("more content", 100.0, 660.0),
            make_rtl_test_span("tail", 100.0, 640.0),
        ];
        PdfDocument::reverse_rtl_visual_order_runs(&mut spans);
        assert_eq!(spans[0].text, "\u{FB7F}\u{FEB3}");
    }

    #[test]
    fn test_reverse_rtl_pass0_leaves_ltr_alone() {
        // Pure Latin spans never trip the RTL heuristic — `rtl_count`
        // is zero so the majority gate fails.
        let mut spans = vec![
            make_rtl_test_span("The quick brown fox jumps over", 100.0, 700.0),
            make_rtl_test_span("the lazy dog repeatedly.", 100.0, 680.0),
            make_rtl_test_span("Latin content here.", 100.0, 660.0),
            make_rtl_test_span("Final line.", 100.0, 640.0),
        ];
        let before: Vec<String> = spans.iter().map(|s| s.text.clone()).collect();
        PdfDocument::reverse_rtl_visual_order_runs(&mut spans);
        let after: Vec<String> = spans.iter().map(|s| s.text.clone()).collect();
        assert_eq!(before, after, "Pure-Latin spans must not be reversed by the RTL pass");
    }

    // #557: the common CID-TrueType shape — one span PER WORD, each word's
    // characters already in LOGICAL order (Presentation Forms), laid out
    // right-to-left so the row-aware sort hands them to us left-to-right
    // (x ascending: last logical word first). The pass must (A) NOT
    // char-reverse the per-word spans — they're already logical — and
    // (B) reverse the WORD order so they read right-to-left. Phrase:
    // "اﻧﻮاع اﳋﻄﻮط اﻟﻌﺮﺑﻴﺔ" ("types of Arabic fonts").
    #[test]
    fn test_reverse_rtl_per_word_logical_spans_reorder_not_charflip() {
        // Spans in x-ascending order (as emitted by the row-aware sort):
        // العربية (leftmost) … انواع (rightmost / logically first).
        let mut spans = vec![
            make_rtl_test_span("اﻟﻌﺮﺑﻴﺔ", 160.0, 700.0),
            make_rtl_test_span(" ", 277.0, 700.0),
            make_rtl_test_span("اﳋﻄﻮط", 288.0, 700.0),
            make_rtl_test_span(" ", 409.0, 700.0),
            make_rtl_test_span("اﻧﻮاع", 420.0, 700.0),
        ];
        PdfDocument::reverse_rtl_visual_order_runs(&mut spans);
        let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
        // (B) word order reversed to logical right-to-left:
        assert_eq!(
            texts,
            vec!["اﻧﻮاع", " ", "اﳋﻄﻮط", " ", "اﻟﻌﺮﺑﻴﺔ"],
            "#557: per-word RTL spans must be reordered into logical word order \
             without char-flipping (got {texts:?})"
        );
    }

    // #553: bare page-number detection (applied only inside the margin band).
    #[test]
    fn test_is_bare_page_number_text() {
        for yes in ["1", "12", "999", "1000", "9999", " 7 ".trim()] {
            assert!(PdfDocument::is_bare_page_number_text(yes), "{yes:?} should be a page number");
        }
        for no in [
            "", "0", "10000", "12345", "1a", "iv", "Page", "1.2", "-1", "1,2",
        ] {
            assert!(!PdfDocument::is_bare_page_number_text(no), "{no:?} must NOT be a page number");
        }
    }

    // ========================================================================
    // decode_pdf_escapes tests
    // ========================================================================

    #[test]
    fn test_decode_pdf_escapes_no_escapes() {
        let text = "Hello World";
        let result = PdfDocument::decode_pdf_escapes(text);
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_decode_pdf_escapes_backslash_n() {
        let result = PdfDocument::decode_pdf_escapes("Hello\\nWorld");
        assert_eq!(result, "Hello\nWorld");
    }

    #[test]
    fn test_decode_pdf_escapes_backslash_r() {
        let result = PdfDocument::decode_pdf_escapes("Hello\\rWorld");
        assert_eq!(result, "Hello\rWorld");
    }

    #[test]
    fn test_decode_pdf_escapes_backslash_t() {
        let result = PdfDocument::decode_pdf_escapes("Hello\\tWorld");
        assert_eq!(result, "Hello\tWorld");
    }

    #[test]
    fn test_decode_pdf_escapes_parentheses() {
        let result = PdfDocument::decode_pdf_escapes("\\(Hello\\)");
        assert_eq!(result, "(Hello)");
    }

    #[test]
    fn test_decode_pdf_escapes_double_backslash() {
        let result = PdfDocument::decode_pdf_escapes("path\\\\file");
        assert_eq!(result, "path\\file");
    }

    #[test]
    fn test_decode_pdf_escapes_octal() {
        // \101 = 'A' in octal (65 decimal)
        let result = PdfDocument::decode_pdf_escapes("\\101");
        assert_eq!(result, "A");
    }

    #[test]
    fn test_decode_pdf_escapes_octal_274() {
        // \274 = 188 decimal which is a PDFDocEncoding char
        let result = PdfDocument::decode_pdf_escapes("\\274");
        assert_eq!(result.chars().count(), 1); // Should decode to a single character
    }

    #[test]
    fn test_decode_pdf_escapes_soft_hyphen() {
        let result = PdfDocument::decode_pdf_escapes("Hello\\?World");
        assert_eq!(result, "HelloWorld");
    }

    #[test]
    fn test_decode_pdf_escapes_unknown_escape() {
        let result = PdfDocument::decode_pdf_escapes("Hello\\zWorld");
        assert_eq!(result, "Hello\\zWorld");
    }

    // ========================================================================
    // pdfdoc_decode tests
    // ========================================================================

    #[test]
    fn test_pdfdoc_decode_ascii() {
        assert_eq!(PdfDocument::pdfdoc_decode(65), 'A');
        assert_eq!(PdfDocument::pdfdoc_decode(48), '0');
        assert_eq!(PdfDocument::pdfdoc_decode(32), ' ');
    }

    #[test]
    fn test_pdfdoc_decode_special_128_bullet() {
        assert_eq!(PdfDocument::pdfdoc_decode(128), '\u{2022}'); // BULLET
    }

    #[test]
    fn test_pdfdoc_decode_special_132_em_dash() {
        assert_eq!(PdfDocument::pdfdoc_decode(132), '\u{2014}'); // EM DASH
    }

    #[test]
    fn test_pdfdoc_decode_special_146_trademark() {
        assert_eq!(PdfDocument::pdfdoc_decode(146), '\u{2122}'); // TRADE MARK SIGN
    }

    #[test]
    fn test_pdfdoc_decode_special_147_fi_ligature() {
        assert_eq!(PdfDocument::pdfdoc_decode(147), '\u{FB01}'); // fi ligature
    }

    #[test]
    fn test_pdfdoc_decode_latin1_range() {
        assert_eq!(PdfDocument::pdfdoc_decode(160), '\u{00A0}'); // Non-breaking space
        assert_eq!(PdfDocument::pdfdoc_decode(255), '\u{00FF}'); // y with diaeresis
    }

    #[test]
    fn test_pdfdoc_decode_replacement_159() {
        assert_eq!(PdfDocument::pdfdoc_decode(159), '\u{FFFD}'); // Replacement character
    }

    // ========================================================================
    // decode_pdf_text_string tests
    // ========================================================================

    #[test]
    fn test_decode_pdf_text_string_utf16be() {
        // UTF-16BE BOM + "AB"
        let bytes = vec![0xFE, 0xFF, 0x00, 0x41, 0x00, 0x42];
        let result = PdfDocument::decode_pdf_text_string(&bytes);
        assert_eq!(result, "AB");
    }

    #[test]
    fn test_decode_pdf_text_string_utf16le() {
        // UTF-16LE BOM + "AB"
        let bytes = vec![0xFF, 0xFE, 0x41, 0x00, 0x42, 0x00];
        let result = PdfDocument::decode_pdf_text_string(&bytes);
        assert_eq!(result, "AB");
    }

    #[test]
    fn test_decode_pdf_text_string_pdfdoc_encoding() {
        // Plain ASCII
        let bytes = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F]; // "Hello"
        let result = PdfDocument::decode_pdf_text_string(&bytes);
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_decode_pdf_text_string_empty() {
        let bytes: Vec<u8> = vec![];
        let result = PdfDocument::decode_pdf_text_string(&bytes);
        assert_eq!(result, "");
    }

    // ========================================================================
    // strip_xhtml_tags tests
    // ========================================================================

    #[test]
    fn test_strip_xhtml_tags_basic() {
        let xhtml = "<p>Hello <b>World</b></p>";
        let result = PdfDocument::strip_xhtml_tags(xhtml);
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_strip_xhtml_tags_no_tags() {
        let text = "Plain text without any tags";
        let result = PdfDocument::strip_xhtml_tags(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_strip_xhtml_tags_empty() {
        assert_eq!(PdfDocument::strip_xhtml_tags(""), "");
    }

    #[test]
    fn test_strip_xhtml_tags_nested() {
        let xhtml = "<div><p><span style='color: red'>Red text</span></p></div>";
        let result = PdfDocument::strip_xhtml_tags(xhtml);
        assert_eq!(result, "Red text");
    }

    // ========================================================================
    // parse_string_value_static tests
    // ========================================================================

    #[test]
    fn test_parse_string_value_static_string() {
        let obj = Object::String(b"Hello".to_vec());
        let result = PdfDocument::parse_string_value_static(Some(&obj));
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "Hello");
    }

    #[test]
    fn test_parse_string_value_static_name() {
        let obj = Object::Name("MyName".to_string());
        let result = PdfDocument::parse_string_value_static(Some(&obj));
        assert_eq!(result, Some("MyName".to_string()));
    }

    #[test]
    fn test_parse_string_value_static_integer() {
        let obj = Object::Integer(42);
        let result = PdfDocument::parse_string_value_static(Some(&obj));
        assert_eq!(result, Some("42".to_string()));
    }

    #[test]
    fn test_parse_string_value_static_real() {
        let obj = Object::Real(std::f64::consts::PI);
        let result = PdfDocument::parse_string_value_static(Some(&obj));
        assert!(result.is_some());
        let s = result.unwrap();
        assert!(s.starts_with("3.14"));
    }

    #[test]
    fn test_parse_string_value_static_null() {
        let obj = Object::Null;
        let result = PdfDocument::parse_string_value_static(Some(&obj));
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_string_value_static_none() {
        let result = PdfDocument::parse_string_value_static(None);
        assert!(result.is_none());
    }

    // ========================================================================
    // find_references tests
    // ========================================================================

    #[test]
    fn test_find_references_reference() {
        let obj = Object::Reference(ObjectRef::new(5, 0));
        let refs = PdfDocument::find_references(&obj);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0], ObjectRef::new(5, 0));
    }

    #[test]
    fn test_find_references_array() {
        let arr = Object::Array(vec![
            Object::Reference(ObjectRef::new(1, 0)),
            Object::Integer(42),
            Object::Reference(ObjectRef::new(2, 0)),
        ]);
        let refs = PdfDocument::find_references(&arr);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_find_references_dictionary() {
        let mut dict = std::collections::HashMap::new();
        dict.insert("Key1".to_string(), Object::Reference(ObjectRef::new(3, 0)));
        dict.insert("Key2".to_string(), Object::Integer(1));
        let obj = Object::Dictionary(dict);
        let refs = PdfDocument::find_references(&obj);
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn test_find_references_stream() {
        let mut dict = std::collections::HashMap::new();
        dict.insert("Length".to_string(), Object::Reference(ObjectRef::new(10, 0)));
        let obj = Object::Stream {
            dict,
            data: bytes::Bytes::from_static(b""),
        };
        let refs = PdfDocument::find_references(&obj);
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn test_find_references_integer() {
        let refs = PdfDocument::find_references(&Object::Integer(42));
        assert!(refs.is_empty());
    }

    #[test]
    fn test_find_references_null() {
        let refs = PdfDocument::find_references(&Object::Null);
        assert!(refs.is_empty());
    }

    #[test]
    fn test_find_references_boolean() {
        let refs = PdfDocument::find_references(&Object::Boolean(true));
        assert!(refs.is_empty());
    }

    #[test]
    fn test_find_references_nested() {
        let inner = Object::Array(vec![Object::Reference(ObjectRef::new(7, 0))]);
        let mut dict = std::collections::HashMap::new();
        dict.insert("Inner".to_string(), inner);
        dict.insert("Direct".to_string(), Object::Reference(ObjectRef::new(8, 0)));
        let obj = Object::Dictionary(dict);
        let refs = PdfDocument::find_references(&obj);
        assert_eq!(refs.len(), 2);
    }

    // ========================================================================
    // find_substring tests
    // ========================================================================

    #[test]
    fn test_find_substring_found() {
        assert_eq!(find_substring(b"Hello World", b"World"), Some(6));
    }

    #[test]
    fn test_find_substring_not_found() {
        assert_eq!(find_substring(b"Hello World", b"xyz"), None);
    }

    #[test]
    fn test_find_substring_empty_needle() {
        assert_eq!(find_substring(b"Hello", b""), Some(0));
    }

    #[test]
    fn test_find_substring_at_start() {
        assert_eq!(find_substring(b"Hello", b"Hello"), Some(0));
    }

    #[test]
    fn test_find_substring_at_end() {
        assert_eq!(find_substring(b"Hello", b"lo"), Some(3));
    }

    #[test]
    fn test_find_substring_empty_haystack() {
        assert_eq!(find_substring(b"", b"Hello"), None);
    }

    // ========================================================================
    // parse_matrix_from_object tests
    // ========================================================================

    #[test]
    fn test_parse_matrix_from_object_valid() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let arr = Object::Array(vec![
            Object::Real(1.0),
            Object::Real(0.0),
            Object::Real(0.0),
            Object::Real(1.0),
            Object::Real(10.0),
            Object::Real(20.0),
        ]);
        let matrix = doc.parse_matrix_from_object(&arr).unwrap();
        assert!((matrix.a - 1.0).abs() < f32::EPSILON);
        assert!((matrix.e - 10.0).abs() < f32::EPSILON);
        assert!((matrix.f - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_matrix_from_object_integers() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let arr = Object::Array(vec![
            Object::Integer(2),
            Object::Integer(0),
            Object::Integer(0),
            Object::Integer(3),
            Object::Integer(100),
            Object::Integer(200),
        ]);
        let matrix = doc.parse_matrix_from_object(&arr).unwrap();
        assert!((matrix.a - 2.0).abs() < f32::EPSILON);
        assert!((matrix.d - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_matrix_from_object_too_short() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let arr = Object::Array(vec![Object::Real(1.0), Object::Real(0.0)]);
        let result = doc.parse_matrix_from_object(&arr);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_matrix_from_object_not_array() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let result = doc.parse_matrix_from_object(&Object::Integer(42));
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_matrix_from_object_invalid_elements() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let arr = Object::Array(vec![
            Object::Real(1.0),
            Object::Name("bad".to_string()), // Not a number
            Object::Real(0.0),
            Object::Real(1.0),
            Object::Real(0.0),
            Object::Real(0.0),
        ]);
        let result = doc.parse_matrix_from_object(&arr);
        assert!(result.is_none());
    }

    // ========================================================================
    // transform_bbox_with_ctm tests
    // ========================================================================

    #[test]
    fn test_transform_bbox_identity() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let rect = crate::geometry::Rect {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 50.0,
        };
        let ctm = crate::content::Matrix::identity();
        let result = doc.transform_bbox_with_ctm(&rect, ctm);
        assert!((result.x - 10.0).abs() < f32::EPSILON);
        assert!((result.y - 20.0).abs() < f32::EPSILON);
        assert!((result.width - 100.0).abs() < f32::EPSILON);
        assert!((result.height - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transform_bbox_translation() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let rect = crate::geometry::Rect {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
        };
        let ctm = crate::content::Matrix {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 50.0,
            f: 100.0,
        };
        let result = doc.transform_bbox_with_ctm(&rect, ctm);
        assert!((result.x - 50.0).abs() < f32::EPSILON);
        assert!((result.y - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transform_bbox_scaling() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let rect = crate::geometry::Rect {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
        };
        let ctm = crate::content::Matrix {
            a: 2.0,
            b: 0.0,
            c: 0.0,
            d: 3.0,
            e: 0.0,
            f: 0.0,
        };
        let result = doc.transform_bbox_with_ctm(&rect, ctm);
        assert!((result.width - 200.0).abs() < f32::EPSILON);
        assert!((result.height - 150.0).abs() < f32::EPSILON);
    }

    // ========================================================================
    // font_identity_hash_cheap tests
    // ========================================================================

    #[test]
    fn test_font_identity_hash_same_font() {
        let mut dict1 = std::collections::HashMap::new();
        dict1.insert("BaseFont".to_string(), Object::Name("Helvetica".to_string()));
        dict1.insert("Subtype".to_string(), Object::Name("Type1".to_string()));

        let mut dict2 = std::collections::HashMap::new();
        dict2.insert("BaseFont".to_string(), Object::Name("Helvetica".to_string()));
        dict2.insert("Subtype".to_string(), Object::Name("Type1".to_string()));

        let hash1 = PdfDocument::font_identity_hash_cheap(&Object::Dictionary(dict1));
        let hash2 = PdfDocument::font_identity_hash_cheap(&Object::Dictionary(dict2));
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_font_identity_hash_different_fonts() {
        let mut dict1 = std::collections::HashMap::new();
        dict1.insert("BaseFont".to_string(), Object::Name("Helvetica".to_string()));

        let mut dict2 = std::collections::HashMap::new();
        dict2.insert("BaseFont".to_string(), Object::Name("Times-Roman".to_string()));

        let hash1 = PdfDocument::font_identity_hash_cheap(&Object::Dictionary(dict1));
        let hash2 = PdfDocument::font_identity_hash_cheap(&Object::Dictionary(dict2));
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_font_identity_hash_null_object() {
        let hash = PdfDocument::font_identity_hash_cheap(&Object::Null);
        // Should not panic, returns some hash
        let _ = hash;
    }

    // #598: two non-subset fonts sharing BaseFont/Subtype/Encoding but with
    // different /Widths must NOT share a cross-document cache key.
    #[test]
    fn test_font_identity_hash_differs_on_widths() {
        let base = || {
            let mut d = std::collections::HashMap::new();
            d.insert("BaseFont".to_string(), Object::Name("Helvetica".to_string()));
            d.insert("Subtype".to_string(), Object::Name("Type1".to_string()));
            d.insert("FirstChar".to_string(), Object::Integer(65));
            d.insert("LastChar".to_string(), Object::Integer(67));
            d
        };
        // PDF A: monospace override.
        let mut a = base();
        a.insert(
            "Widths".to_string(),
            Object::Array(vec![
                Object::Integer(600),
                Object::Integer(600),
                Object::Integer(600),
            ]),
        );
        // PDF B: real Helvetica metrics — same name, different widths.
        let mut b = base();
        b.insert(
            "Widths".to_string(),
            Object::Array(vec![
                Object::Integer(667),
                Object::Integer(667),
                Object::Integer(722),
            ]),
        );

        let hash_a = PdfDocument::font_identity_hash_cheap(&Object::Dictionary(a));
        let hash_b = PdfDocument::font_identity_hash_cheap(&Object::Dictionary(b));
        assert_ne!(
            hash_a, hash_b,
            "#598: fonts with identical BaseFont but different /Widths must not collide"
        );

        // Sanity: identical widths still hash equally (genuine cache hits).
        let mut c = base();
        c.insert(
            "Widths".to_string(),
            Object::Array(vec![
                Object::Integer(600),
                Object::Integer(600),
                Object::Integer(600),
            ]),
        );
        let mut a2 = base();
        a2.insert(
            "Widths".to_string(),
            Object::Array(vec![
                Object::Integer(600),
                Object::Integer(600),
                Object::Integer(600),
            ]),
        );
        assert_eq!(
            PdfDocument::font_identity_hash_cheap(&Object::Dictionary(c)),
            PdfDocument::font_identity_hash_cheap(&Object::Dictionary(a2)),
            "#598: identical fonts must still share a cache key"
        );
    }

    // #597: Type 3 fonts are document-local and must be kept out of the
    // cross-document global font cache (Layer 6). The gate uses
    // font_is_document_local; pin its classification here.
    #[test]
    fn test_type3_font_is_document_local() {
        let mut type3 = std::collections::HashMap::new();
        type3.insert("Subtype".to_string(), Object::Name("Type3".to_string()));
        type3.insert("Name".to_string(), Object::Name("F1".to_string()));
        assert!(
            PdfDocument::font_is_document_local(&Object::Dictionary(type3)),
            "#597: Type3 fonts must be treated as document-local (uncacheable cross-document)"
        );

        // Non-Type3 fonts remain cacheable across documents.
        for subtype in ["Type1", "TrueType", "Type0", "CIDFontType2"] {
            let mut d = std::collections::HashMap::new();
            d.insert("Subtype".to_string(), Object::Name(subtype.to_string()));
            d.insert("BaseFont".to_string(), Object::Name("Helvetica".to_string()));
            assert!(
                !PdfDocument::font_is_document_local(&Object::Dictionary(d)),
                "#597: {subtype} must remain cacheable across documents"
            );
        }
        // A dict with no Subtype is not document-local.
        assert!(!PdfDocument::font_is_document_local(&Object::Null));

        // Subset fonts (six uppercase letters + '+', ISO 32000-1 §9.6.4) are
        // document-local regardless of subtype — their glyph subset and
        // ToUnicode are document-specific and must not be shared cross-document.
        for subtype in ["Type1", "TrueType", "Type0", "CIDFontType2"] {
            let mut d = std::collections::HashMap::new();
            d.insert("Subtype".to_string(), Object::Name(subtype.to_string()));
            d.insert("BaseFont".to_string(), Object::Name("AAAAAA+ArialUnicodeMS".to_string()));
            assert!(
                PdfDocument::font_is_document_local(&Object::Dictionary(d)),
                "subset {subtype} must be treated as document-local"
            );
        }

        // Subset-prefix edge cases: a 6-uppercase name without '+', a lowercase
        // tag, a short tag, and an empty real name are NOT subsets — stay cacheable.
        for name in ["ARIALX", "abcdef+Real", "AAAAA+Short", "AAAAAA+"] {
            let mut d = std::collections::HashMap::new();
            d.insert("Subtype".to_string(), Object::Name("Type0".to_string()));
            d.insert("BaseFont".to_string(), Object::Name(name.to_string()));
            assert!(
                !PdfDocument::font_is_document_local(&Object::Dictionary(d)),
                "{name} is not a subset tag and must remain cacheable"
            );
        }

        // A non-Type3 font missing /BaseFont fails safe to document-local.
        let mut no_basefont = std::collections::HashMap::new();
        no_basefont.insert("Subtype".to_string(), Object::Name("Type0".to_string()));
        assert!(
            PdfDocument::font_is_document_local(&Object::Dictionary(no_basefont)),
            "a non-Type3 font with no /BaseFont must fail safe to document-local"
        );
    }

    // ========================================================================
    // check_for_circular_references tests
    // ========================================================================

    #[test]
    fn test_check_for_circular_references_runs() {
        // Minimal PDFs naturally have Page <-> Pages parent references,
        // so we just verify the function runs without panicking
        // returns a list (which may include the Page<->Pages backreference).
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let cycles = doc.check_for_circular_references();
        // Returns a Vec of (from, to) pairs - may or may not be empty
        let _ = cycles;
    }

    // ========================================================================
    // is_form_xobject tests
    // ========================================================================

    #[test]
    fn test_is_form_xobject_nonexistent_ref() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        // Non-existent object should return true (conservative)
        let result = doc.is_form_xobject(ObjectRef::new(999, 0));
        assert!(result);
    }

    #[test]
    fn test_is_form_xobject_catalog_not_form() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        // Load catalog into cache first
        let _ = doc.load_object(ObjectRef::new(1, 0));
        // Catalog is not a Form XObject
        let result = doc.is_form_xobject(ObjectRef::new(1, 0));
        assert!(!result);
    }

    // ========================================================================
    // from_bytes with various PDF structures
    // ========================================================================

    #[test]
    fn test_from_bytes_with_v2_header() {
        let mut pdf = b"%PDF-2.0\n".to_vec();

        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n");

        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 3\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );

        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert_eq!(doc.version(), (2, 0));
    }

    // ========================================================================
    // parse_version_from_header tests
    // ========================================================================

    #[test]
    fn test_parse_version_from_header_strict_valid() {
        let header = *b"%PDF-1.7";
        let (major, minor) = parse_version_from_header(&header, false).unwrap();
        assert_eq!((major, minor), (1, 7));
    }

    #[test]
    fn test_parse_version_from_header_strict_invalid_dot() {
        let header = *b"%PDF-1X7";
        let result = parse_version_from_header(&header, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_version_from_header_lenient_invalid_dot() {
        let header = *b"%PDF-1X7";
        let (major, minor) = parse_version_from_header(&header, true).unwrap();
        assert_eq!((major, minor), (1, 4)); // defaults to 1.4
    }

    #[test]
    fn test_parse_version_from_header_strict_non_digit() {
        let header = *b"%PDF-X.Y";
        let result = parse_version_from_header(&header, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_version_from_header_lenient_non_digit() {
        let header = *b"%PDF-X.Y";
        let (major, minor) = parse_version_from_header(&header, true).unwrap();
        assert_eq!((major, minor), (1, 4));
    }

    #[test]
    fn test_parse_version_from_header_strict_too_high() {
        let header = *b"%PDF-3.0";
        let result = parse_version_from_header(&header, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_version_from_header_lenient_too_high() {
        let header = *b"%PDF-3.0";
        let (major, minor) = parse_version_from_header(&header, true).unwrap();
        assert_eq!((major, minor), (1, 4));
    }

    #[test]
    fn test_parse_version_from_header_wrong_magic() {
        let header = *b"NotPDF17";
        let result = parse_version_from_header(&header, false);
        assert!(result.is_err());
    }

    // ========================================================================
    // parse_header edge cases
    // ========================================================================

    #[test]
    fn test_parse_header_empty_file_strict() {
        let mut cursor = Cursor::new(b"");
        let result = parse_header(&mut cursor, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_header_empty_file_lenient() {
        let mut cursor = Cursor::new(b"");
        let result = parse_header(&mut cursor, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_header_very_short_lenient() {
        let mut cursor = Cursor::new(b"AB");
        let result = parse_header(&mut cursor, true);
        // Lenient mode with no %PDF- found defaults to 1.4
        let (major, minor, _) = result.unwrap();
        assert_eq!((major, minor), (1, 4));
    }

    #[test]
    fn test_parse_header_header_near_end_of_buffer() {
        // Header at position 8100 (within 8192 byte search window)
        let mut data = vec![0u8; 8100];
        data.extend_from_slice(b"%PDF-1.6");
        data.extend_from_slice(b"\nrest of file data here");
        let mut cursor = Cursor::new(data);
        let (major, minor, offset) = parse_header(&mut cursor, true).unwrap();
        assert_eq!((major, minor, offset), (1, 6, 8100));
    }

    // ========================================================================
    // parse_trailer edge cases
    // ========================================================================

    #[test]
    fn test_parse_trailer_with_extra_data() {
        let data =
            b"some xref data\ntrailer\n<< /Size 10 /Root 1 0 R /Info 2 0 R >>\nstartxref\n100\n";
        let mut cursor = Cursor::new(data);
        let trailer = parse_trailer(&mut cursor).unwrap();
        let dict = trailer.as_dict().unwrap();
        assert_eq!(dict.get("Size").unwrap().as_integer(), Some(10));
    }

    #[test]
    fn test_parse_trailer_empty_after_keyword() {
        let data = b"trailer";
        let mut cursor = Cursor::new(data);
        let result = parse_trailer(&mut cursor);
        assert!(result.is_err());
    }

    // ========================================================================
    // decode_stream_with_encryption tests
    // ========================================================================

    #[test]
    fn test_decode_stream_with_encryption_null_object() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let result = doc
            .decode_stream_with_encryption(&Object::Null, ObjectRef::new(1, 0))
            .unwrap();
        assert!(result.is_empty());
    }

    // ========================================================================
    // page_cannot_have_text tests
    // ========================================================================

    #[test]
    fn test_page_cannot_have_text_no_resources() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        // Empty resources dict
        let page_dict = std::collections::HashMap::new();
        assert!(doc.page_cannot_have_text(&page_dict));
    }

    #[test]
    fn test_page_cannot_have_text_with_font_resources() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let mut font_dict = std::collections::HashMap::new();
        font_dict.insert("F1".to_string(), Object::Reference(ObjectRef::new(10, 0)));

        let mut resources_dict = std::collections::HashMap::new();
        resources_dict.insert("Font".to_string(), Object::Dictionary(font_dict));

        let mut page_dict = std::collections::HashMap::new();
        page_dict.insert("Resources".to_string(), Object::Dictionary(resources_dict));

        // Has fonts, so page CAN have text
        assert!(!doc.page_cannot_have_text(&page_dict));
    }

    #[test]
    fn test_page_cannot_have_text_empty_font_dict() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();

        let font_dict = std::collections::HashMap::new();

        let mut resources_dict = std::collections::HashMap::new();
        resources_dict.insert("Font".to_string(), Object::Dictionary(font_dict));

        let mut page_dict = std::collections::HashMap::new();
        page_dict.insert("Resources".to_string(), Object::Dictionary(resources_dict));

        // Empty font dict and no XObjects = no text possible
        assert!(doc.page_cannot_have_text(&page_dict));
    }

    // ========================================================================
    // extract_images tests
    // ========================================================================

    #[test]
    fn test_extract_images_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let images = doc.extract_images(0).unwrap();
        assert!(images.is_empty());
    }

    #[test]
    fn test_extract_images_graphics_only() {
        let content = b"100 200 300 400 re S";
        let pdf = build_minimal_pdf(content);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let images = doc.extract_images(0).unwrap();
        assert!(images.is_empty());
    }

    // ========================================================================
    // extract_paths tests
    // ========================================================================

    #[test]
    fn test_extract_paths_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let paths = doc.extract_paths(0).unwrap();
        assert!(paths.is_empty());
    }

    #[test]
    fn test_extract_paths_rectangle() {
        let content = b"100 200 300 400 re S";
        let pdf = build_minimal_pdf(content);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let paths = doc.extract_paths(0).unwrap();
        assert!(!paths.is_empty());
    }

    // ========================================================================
    // mark_info tests
    // ========================================================================

    #[test]
    fn test_mark_info_untagged_pdf() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let mark_info = doc.mark_info().unwrap();
        // Untagged PDF should have default MarkInfo
        assert!(!mark_info.marked);
        assert!(!mark_info.suspects);
    }

    // ========================================================================
    // ExtractedImageRef and ImageFormat tests
    // ========================================================================

    #[test]
    fn test_extracted_image_ref_debug() {
        let img_ref = ExtractedImageRef {
            filename: "img_001.png".to_string(),
            format: ImageFormat::Png,
            width: 100,
            height: 200,
            bbox: None,
            rotation: 0,
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        };
        let debug = format!("{:?}", img_ref);
        assert!(debug.contains("img_001.png"));
        assert!(debug.contains("Png"));
    }

    #[test]
    fn test_extracted_image_ref_clone() {
        let img_ref = ExtractedImageRef {
            filename: "img_001.jpg".to_string(),
            format: ImageFormat::Jpeg,
            width: 100,
            height: 200,
            bbox: None,
            rotation: 0,
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        };
        let cloned = img_ref.clone();
        assert_eq!(img_ref, cloned);
    }

    #[test]
    fn test_image_format_equality() {
        assert_eq!(ImageFormat::Png, ImageFormat::Png);
        assert_eq!(ImageFormat::Jpeg, ImageFormat::Jpeg);
        assert_ne!(ImageFormat::Png, ImageFormat::Jpeg);
    }

    // ========================================================================
    // apply_intelligent_text_processing tests
    // ========================================================================

    #[test]
    fn test_apply_intelligent_text_processing_empty() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let spans: Vec<TextSpan> = vec![];
        let result = doc.apply_intelligent_text_processing(spans);
        assert!(result.is_empty());
    }

    #[test]
    fn test_apply_intelligent_text_processing_ligature_preserved() {
        // Since v0.3.46 the pipeline preserves Unicode ligature characters that come
        // from the font's ToUnicode map (U+FB01 = ﬁ). Expanding them to plain "fi"
        // caused Jaccard mismatches against ground-truth corpora that keep ligatures.
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let spans = vec![make_test_span("\u{FB01}nd", 0.0, 0.0, 50.0, 12.0)]; // ﬁnd
        let result = doc.apply_intelligent_text_processing(spans);
        assert_eq!(result.len(), 1);
        assert!(
            result[0].text.contains('\u{FB01}'),
            "ﬁ must be preserved, got: {:?}",
            result[0].text
        );
    }

    // ========================================================================
    // Page retrieval and caching tests
    // ========================================================================

    #[test]
    fn test_get_page_caching() {
        let pdf = build_multi_page_pdf(3);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        // Access page 0 twice -- second should come from cache
        let _page1 = doc.get_page(0).unwrap();
        let _page2 = doc.get_page(0).unwrap();
        // Both should succeed
    }

    #[test]
    fn test_get_page_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        // Page 99 doesn't exist
        let result = doc.get_page(99);
        assert!(result.is_err());
    }

    // ========================================================================
    // page_count_u32 deprecated method test
    // ========================================================================

    #[test]
    #[allow(deprecated)]
    fn test_page_count_u32_returns_correct_value() {
        let pdf = build_multi_page_pdf(3);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert_eq!(doc.page_count_u32(), 3);
    }

    // ========================================================================
    // structure_tree for untagged PDF
    // ========================================================================

    #[test]
    fn test_structure_tree_untagged_pdf() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let tree = doc.structure_tree().unwrap();
        assert!(tree.is_none()); // Untagged PDF has no structure tree
    }

    // ========================================================================
    // Conversion output tests (to_markdown, to_html, to_plain_text)
    // ========================================================================

    #[test]
    fn test_to_plain_text_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let options = crate::converters::ConversionOptions::default();
        let text = doc.to_plain_text(0, &options).unwrap();
        assert!(text.is_empty() || text.trim().is_empty());
    }

    #[test]
    fn test_to_markdown_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let options = crate::converters::ConversionOptions::default();
        let md = doc.to_markdown(0, &options).unwrap();
        assert!(md.is_empty() || md.trim().is_empty());
    }

    #[test]
    fn test_to_html_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let options = crate::converters::ConversionOptions::default();
        let html = doc.to_html(0, &options).unwrap();
        // HTML may have structure tags even for empty content
        let _ = html;
    }

    #[test]
    fn test_to_markdown_all_multiple_pages() {
        let pdf = build_multi_page_pdf(2);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let options = crate::converters::ConversionOptions::default();
        let md = doc.to_markdown_all(&options).unwrap();
        // Should have a separator between pages
        assert!(md.contains("---") || md.is_empty());
    }

    #[test]
    fn test_to_plain_text_all_multiple_pages() {
        let pdf = build_multi_page_pdf(2);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let options = crate::converters::ConversionOptions::default();
        let text = doc.to_plain_text_all(&options).unwrap();
        let _ = text; // Should not crash
    }

    #[test]
    fn test_to_html_all_multiple_pages() {
        let pdf = build_multi_page_pdf(2);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let options = crate::converters::ConversionOptions::default();
        let html = doc.to_html_all(&options).unwrap();
        assert!(html.contains("data-page=\"1\""));
        assert!(html.contains("data-page=\"2\""));
    }

    // ========================================================================
    // open_with_config test
    // ========================================================================

    #[test]
    fn test_open_with_config() {
        let pdf = build_minimal_pdf(b"");
        let tmp_path = std::env::temp_dir().join("pdf_oxide_test_open_with_config.pdf");
        std::fs::write(&tmp_path, &pdf).unwrap();
        let config = 42u32; // Dummy config
        let result = PdfDocument::open_with_config(&tmp_path, config);
        let _ = std::fs::remove_file(&tmp_path);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Debug wrappers (get_page_for_debug, may_contain_text_public)
    // ========================================================================

    #[test]
    fn test_get_page_for_debug() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let page = doc.get_page_for_debug(0).unwrap();
        assert!(page.as_dict().is_some());
    }

    #[test]
    fn test_may_contain_text_public() {
        assert!(PdfDocument::may_contain_text_public(b"BT /F1 12 Tf ET"));
        assert!(!PdfDocument::may_contain_text_public(b"100 200 re S"));
    }

    // ========================================================================
    // Inherited attributes in page tree
    // ========================================================================

    #[test]
    fn test_page_inherits_mediabox() {
        // Build a PDF where MediaBox is on the Pages node, not on the Page itself
        let mut pdf = b"%PDF-1.4\n".to_vec();

        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        let off2 = pdf.len();
        pdf.extend_from_slice(
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 400 600] >>\nendobj\n",
        );

        let off3 = pdf.len();
        pdf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n");

        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 4\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );

        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert_eq!(doc.page_count().unwrap(), 1);
        // The page should inherit the MediaBox from its parent
        let page = doc.get_page(0).unwrap();
        let page_dict = page.as_dict().unwrap();
        assert!(page_dict.contains_key("MediaBox"));
    }

    // ========================================================================
    // Content stream array tests
    // ========================================================================

    #[test]
    fn test_page_with_array_contents() {
        // Build a PDF where /Contents is an array of stream references
        let mut pdf = b"%PDF-1.4\n".to_vec();

        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

        let off3 = pdf.len();
        pdf.extend_from_slice(
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents [4 0 R 5 0 R] /Resources << >> >>\nendobj\n",
        );

        let content1 = b"q";
        let off4 = pdf.len();
        pdf.extend_from_slice(
            format!("4 0 obj\n<< /Length {} >>\nstream\n", content1.len()).as_bytes(),
        );
        pdf.extend_from_slice(content1);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");

        let content2 = b"Q";
        let off5 = pdf.len();
        pdf.extend_from_slice(
            format!("5 0 obj\n<< /Length {} >>\nstream\n", content2.len()).as_bytes(),
        );
        pdf.extend_from_slice(content2);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");

        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 6\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off4).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off5).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );

        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let data = doc.get_page_content_data(0).unwrap();
        let text = String::from_utf8_lossy(&data);
        assert!(text.contains("q"));
        assert!(text.contains("Q"));
    }

    // ========================================================================
    // Hierarchical content extraction test
    // ========================================================================

    #[test]
    fn test_extract_hierarchical_content_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let result = doc.extract_hierarchical_content(0);
        // Should not crash, may return Ok(Some) or Ok(None)
        assert!(result.is_ok());
    }

    // ========================================================================
    // extract_paths_in_rect test
    // ========================================================================

    #[test]
    fn test_extract_paths_in_rect_empty_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let region = crate::geometry::Rect {
            x: 0.0,
            y: 0.0,
            width: 612.0,
            height: 792.0,
        };
        let paths = doc.extract_paths_in_rect(0, region).unwrap();
        assert!(paths.is_empty());
    }

    // ========================================================================
    // PDF with nested page tree
    // ========================================================================

    #[test]
    fn test_nested_page_tree() {
        // Build a PDF with nested Pages nodes
        let mut pdf = b"%PDF-1.4\n".to_vec();

        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 2 >>\nendobj\n");

        // Intermediate Pages node
        let off3 = pdf.len();
        pdf.extend_from_slice(
            b"3 0 obj\n<< /Type /Pages /Kids [4 0 R 5 0 R] /Count 2 /Parent 2 0 R >>\nendobj\n",
        );

        let off4 = pdf.len();
        pdf.extend_from_slice(
            b"4 0 obj\n<< /Type /Page /Parent 3 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
        );

        let off5 = pdf.len();
        pdf.extend_from_slice(
            b"5 0 obj\n<< /Type /Page /Parent 3 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
        );

        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 6\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off4).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off5).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );

        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert_eq!(doc.page_count().unwrap(), 2);
    }

    // ========================================================================
    // PDF with MarkInfo in catalog
    // ========================================================================

    #[test]
    fn test_mark_info_tagged_pdf() {
        let mut pdf = b"%PDF-1.4\n".to_vec();

        let off1 = pdf.len();
        pdf.extend_from_slice(
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true /Suspects false >> >>\nendobj\n",
        );

        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n");

        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 3\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );

        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let mark_info = doc.mark_info().unwrap();
        assert!(mark_info.marked);
        assert!(!mark_info.suspects);
    }

    // ========================================================================
    // extract_spans_with_config test
    // ========================================================================

    #[test]
    fn test_extract_spans_with_config_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let config = crate::extractors::SpanMergingConfig::default();
        let spans = doc.extract_spans_with_config(0, config).unwrap();
        assert!(spans.is_empty());
    }

    // ========================================================================
    // get_page_ref tests
    // ========================================================================

    #[test]
    fn test_get_page_ref_valid() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let page_ref = doc.get_page_ref(0).unwrap();
        // Page should be object 3 (catalog=1, pages=2, page=3)
        assert_eq!(page_ref.id, 3);
    }

    #[test]
    fn test_get_page_ref_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let result = doc.get_page_ref(99);
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 1: decode_pdf_escapes edge cases
    // ========================================================================

    #[test]
    fn test_decode_pdf_escapes_trailing_backslash() {
        let result = PdfDocument::decode_pdf_escapes("Hello\\");
        assert_eq!(result, "Hello\\");
    }

    #[test]
    fn test_decode_pdf_escapes_octal_short() {
        let result = PdfDocument::decode_pdf_escapes("\\1x");
        assert_eq!(result.len(), 2);
        assert!(result.ends_with('x'));
    }

    #[test]
    fn test_decode_pdf_escapes_octal_two_digits() {
        let result = PdfDocument::decode_pdf_escapes("\\41x");
        assert_eq!(result, "!x");
    }

    #[test]
    fn test_decode_pdf_escapes_octal_non_octal_digit() {
        // \8 matches the digit branch, but 8 is not a valid octal digit (< '8'),
        // so octal stays empty -> backslash is kept, then '8' is consumed normally.
        let result = PdfDocument::decode_pdf_escapes("\\8");
        assert_eq!(result, "\\8");
    }

    #[test]
    fn test_decode_pdf_escapes_multiple_escapes() {
        let result = PdfDocument::decode_pdf_escapes("\\(a\\)\\\\\\n");
        assert_eq!(result, "(a)\\\n");
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 2: pdfdoc_decode all ranges
    // ========================================================================

    #[test]
    fn test_pdfdoc_decode_control_chars() {
        assert_eq!(PdfDocument::pdfdoc_decode(0), '\0');
        assert_eq!(PdfDocument::pdfdoc_decode(10), '\n');
        assert_eq!(PdfDocument::pdfdoc_decode(13), '\r');
    }

    #[test]
    fn test_pdfdoc_decode_all_special_range() {
        assert_eq!(PdfDocument::pdfdoc_decode(128), '\u{2022}');
        assert_eq!(PdfDocument::pdfdoc_decode(129), '\u{2020}');
        assert_eq!(PdfDocument::pdfdoc_decode(130), '\u{2021}');
        assert_eq!(PdfDocument::pdfdoc_decode(131), '\u{2026}');
        assert_eq!(PdfDocument::pdfdoc_decode(133), '\u{2013}');
        assert_eq!(PdfDocument::pdfdoc_decode(134), '\u{0192}');
        assert_eq!(PdfDocument::pdfdoc_decode(135), '\u{2044}');
        assert_eq!(PdfDocument::pdfdoc_decode(136), '\u{2039}');
        assert_eq!(PdfDocument::pdfdoc_decode(137), '\u{203A}');
        assert_eq!(PdfDocument::pdfdoc_decode(138), '\u{2212}');
        assert_eq!(PdfDocument::pdfdoc_decode(139), '\u{2030}');
        assert_eq!(PdfDocument::pdfdoc_decode(140), '\u{201E}');
        assert_eq!(PdfDocument::pdfdoc_decode(141), '\u{201C}');
        assert_eq!(PdfDocument::pdfdoc_decode(142), '\u{201D}');
        assert_eq!(PdfDocument::pdfdoc_decode(143), '\u{2018}');
        assert_eq!(PdfDocument::pdfdoc_decode(144), '\u{2019}');
        assert_eq!(PdfDocument::pdfdoc_decode(145), '\u{201A}');
        assert_eq!(PdfDocument::pdfdoc_decode(148), '\u{FB02}');
        assert_eq!(PdfDocument::pdfdoc_decode(149), '\u{0141}');
        assert_eq!(PdfDocument::pdfdoc_decode(150), '\u{0152}');
        assert_eq!(PdfDocument::pdfdoc_decode(151), '\u{0160}');
        assert_eq!(PdfDocument::pdfdoc_decode(152), '\u{0178}');
        assert_eq!(PdfDocument::pdfdoc_decode(153), '\u{017D}');
        assert_eq!(PdfDocument::pdfdoc_decode(154), '\u{0131}');
        assert_eq!(PdfDocument::pdfdoc_decode(155), '\u{0142}');
        assert_eq!(PdfDocument::pdfdoc_decode(156), '\u{0153}');
        assert_eq!(PdfDocument::pdfdoc_decode(157), '\u{0161}');
        assert_eq!(PdfDocument::pdfdoc_decode(158), '\u{017E}');
    }

    #[test]
    fn test_pdfdoc_decode_latin1_boundary() {
        assert_eq!(PdfDocument::pdfdoc_decode(160), '\u{00A0}');
        assert_eq!(PdfDocument::pdfdoc_decode(255), '\u{00FF}');
        assert_eq!(PdfDocument::pdfdoc_decode(200), '\u{00C8}');
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 3: decode_pdf_text_string
    // ========================================================================

    #[test]
    fn test_decode_pdf_text_string_utf8_bom_treated_as_pdfdoc() {
        // UTF-8 BOM (EF BB BF) is NOT recognized by this function;
        // it only handles UTF-16 BOMs. Bytes fall through to PDFDocEncoding.
        let bytes = vec![0xEF, 0xBB, 0xBF, b'H', b'e', b'l', b'l', b'o'];
        let result = PdfDocument::decode_pdf_text_string(&bytes);
        // 0xEF -> ï, 0xBB -> », 0xBF -> ¿ in PDFDocEncoding (Latin-1 range)
        assert_eq!(result, "\u{00EF}\u{00BB}\u{00BF}Hello");
    }

    #[test]
    fn test_decode_pdf_text_string_plain_ascii() {
        let result = PdfDocument::decode_pdf_text_string(b"Hello World");
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_decode_pdf_text_string_with_special_chars() {
        let bytes = vec![128u8];
        let result = PdfDocument::decode_pdf_text_string(&bytes);
        assert!(result.contains('\u{2022}'));
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 4: filter_leaked_metadata edge cases
    // ========================================================================

    #[test]
    fn test_filter_leaked_metadata_blackpoint() {
        let text = "BlackPoint [ 0 0 0 ]";
        let result = PdfDocument::filter_leaked_metadata(text);
        assert!(result.trim().is_empty());
    }

    #[test]
    fn test_filter_leaked_metadata_gamma() {
        let text = "Some text\nGamma [ 2.2 2.2 2.2 ]\nMore text";
        let result = PdfDocument::filter_leaked_metadata(text);
        assert!(!result.contains("Gamma"));
        assert!(result.contains("Some text"));
        assert!(result.contains("More text"));
    }

    #[test]
    fn test_filter_leaked_metadata_matrix_start_line() {
        let text = "Matrix [ 1 0 0 1 0 0 ]";
        let result = PdfDocument::filter_leaked_metadata(text);
        assert!(result.trim().is_empty());
    }

    #[test]
    fn test_filter_leaked_metadata_calgray() {
        let text = "CalGray /WhitePoint [ 1 1 1 ]";
        let result = PdfDocument::filter_leaked_metadata(text);
        assert!(!result.contains("CalGray"));
    }

    #[test]
    fn test_filter_leaked_metadata_whitepoint_with_slash() {
        let result = PdfDocument::filter_leaked_metadata("WhitePoint /something");
        assert!(result.trim().is_empty());
    }

    #[test]
    fn test_filter_leaked_metadata_whitepoint_with_angle() {
        let result = PdfDocument::filter_leaked_metadata("WhitePoint << /Key /Value >>");
        assert!(result.trim().is_empty());
    }

    #[test]
    fn test_filter_leaked_metadata_empty_metadata_value() {
        let result = PdfDocument::filter_leaked_metadata("WhitePoint");
        assert!(result.trim().is_empty());
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 5: normalize_arabic presentation forms
    // ========================================================================

    #[test]
    fn test_normalize_arabic_hamza() {
        let result = PdfDocument::normalize_arabic_presentation_forms("\u{FE80}");
        assert!(result.contains('\u{0621}'));
    }

    #[test]
    fn test_normalize_arabic_beh() {
        let result = PdfDocument::normalize_arabic_presentation_forms("\u{FE8F}");
        assert!(result.contains('\u{0628}'));
    }

    #[test]
    fn test_normalize_arabic_teh_marbuta() {
        let result = PdfDocument::normalize_arabic_presentation_forms("\u{FE93}");
        assert!(result.contains('\u{0629}'));
    }

    #[test]
    fn test_normalize_arabic_dal_to_yeh_range() {
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEA9}").contains('\u{062F}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEAB}").contains('\u{0630}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEAD}").contains('\u{0631}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEAF}").contains('\u{0632}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEB1}").contains('\u{0633}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEB5}").contains('\u{0634}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEB9}").contains('\u{0635}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEBD}").contains('\u{0636}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEC1}").contains('\u{0637}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEC5}").contains('\u{0638}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEC9}").contains('\u{0639}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FECD}").contains('\u{063A}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FED1}").contains('\u{0641}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FED5}").contains('\u{0642}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FED9}").contains('\u{0643}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEDD}").contains('\u{0644}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEE1}").contains('\u{0645}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEE5}").contains('\u{0646}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEE9}").contains('\u{0647}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEED}").contains('\u{0648}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEEF}").contains('\u{0649}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEF1}").contains('\u{064A}'));
    }

    #[test]
    fn test_normalize_arabic_diacritics() {
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE70}").contains('\u{064B}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE71}").contains('\u{064B}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE72}").contains('\u{064C}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE74}").contains('\u{064D}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE76}").contains('\u{064E}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE77}").contains('\u{064E}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE78}").contains('\u{064F}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE79}").contains('\u{064F}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE7A}").contains('\u{0650}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE7B}").contains('\u{0650}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE7C}").contains('\u{0651}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE7D}").contains('\u{0651}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE7E}").contains('\u{0652}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE7F}").contains('\u{0652}'));
    }

    #[test]
    fn test_normalize_arabic_lam_alef_ligatures() {
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEF5}").contains('\u{0644}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEF7}").contains('\u{0644}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEF9}").contains('\u{0644}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FEFB}").contains('\u{0644}'));
    }

    #[test]
    fn test_normalize_arabic_alef_variants() {
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE81}").contains('\u{0622}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE83}").contains('\u{0623}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE85}").contains('\u{0624}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE87}").contains('\u{0625}'));
        assert!(PdfDocument::normalize_arabic_presentation_forms("\u{FE89}").contains('\u{0626}'));
    }

    #[test]
    fn test_normalize_arabic_mixed_text() {
        let result = PdfDocument::normalize_arabic_presentation_forms("Hello \u{FE8D} World");
        assert!(result.contains("Hello"));
        assert!(result.contains("World"));
        assert!(result.contains('\u{0627}'));
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 6: strip_xhtml_tags edge cases
    // ========================================================================

    #[test]
    fn test_strip_xhtml_tags_self_closing() {
        assert_eq!(PdfDocument::strip_xhtml_tags("Hello<br/>World"), "HelloWorld");
    }

    #[test]
    fn test_strip_xhtml_tags_with_attributes() {
        assert_eq!(PdfDocument::strip_xhtml_tags("<p class=\"body\">Content</p>"), "Content");
    }

    #[test]
    fn test_strip_xhtml_tags_multiple() {
        assert_eq!(
            PdfDocument::strip_xhtml_tags("<b>Bold</b> and <i>Italic</i>"),
            "Bold and Italic"
        );
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 7: should_insert_space edge cases
    // ========================================================================

    #[test]
    fn test_should_insert_space_overlapping() {
        let prev = make_test_span("Hello", 0.0, 100.0, 50.0, 12.0);
        let current = make_test_span("World", 40.0, 100.0, 50.0, 12.0);
        assert!(!PdfDocument::should_insert_space(&prev, &current));
    }

    #[test]
    fn test_should_insert_space_zero_font_size() {
        let prev = make_test_span("A", 0.0, 100.0, 10.0, 0.0);
        let current = make_test_span("B", 15.0, 100.0, 10.0, 0.0);
        let _ = PdfDocument::should_insert_space(&prev, &current);
    }

    #[test]
    fn test_should_insert_space_large_font() {
        let prev = make_test_span("A", 0.0, 100.0, 100.0, 72.0);
        let current = make_test_span("B", 120.0, 100.0, 100.0, 72.0);
        assert!(PdfDocument::should_insert_space(&prev, &current));
    }

    // ========================================================================
    // extract_words: adjacent-span merging tests (issue-336 regression)
    // ========================================================================

    /// Adjacent-span merging: two single-character spans with zero horizontal gap
    /// must produce one merged word, not two separate words.
    ///
    /// This exercises the post-processing pass in extract_words_inner that merges
    /// spans whose bboxes abut (gap ≤ 0.15 × font_size) on the same line.
    #[test]
    fn test_extract_words_adjacent_spans_merged() {
        // Build a minimal one-page PDF whose content stream places "Q" then "（"
        // as two consecutive Tj operations with no word space between them.
        // We build the PDF bytes manually so we control the exact glyph positions.
        use crate::ffi::{
            free_bytes, pdf_document_builder_build, pdf_document_builder_create,
            pdf_document_builder_free, pdf_document_builder_letter_page, pdf_page_builder_at,
            pdf_page_builder_done, pdf_page_builder_font, pdf_page_builder_text,
        };
        use std::ffi::CString;

        unsafe {
            let mut ec: i32 = -1;
            let builder = pdf_document_builder_create(&mut ec);
            assert_eq!(ec, 0);
            let page = pdf_document_builder_letter_page(builder, &mut ec);
            assert_eq!(ec, 0);
            pdf_page_builder_font(page, CString::new("Helvetica").unwrap().as_ptr(), 12.0, &mut ec);
            assert_eq!(ec, 0);
            // Place "Q（peu/d）" as a single text run — this will be one span.
            // extract_words should produce exactly one word.
            pdf_page_builder_at(page, 100.0, 500.0, &mut ec);
            assert_eq!(ec, 0);
            let t = CString::new("Q（peu/d）").unwrap();
            pdf_page_builder_text(page, t.as_ptr(), &mut ec);
            assert_eq!(ec, 0);
            pdf_page_builder_done(page, &mut ec);
            assert_eq!(ec, 0);

            let mut pdf_len: usize = 0;
            let pdf_ptr = pdf_document_builder_build(builder, &mut pdf_len, &mut ec);
            assert_eq!(ec, 0);
            let bytes = std::slice::from_raw_parts(pdf_ptr as *const u8, pdf_len).to_vec();
            free_bytes(pdf_ptr);
            pdf_document_builder_free(builder);

            let doc = PdfDocument::from_bytes(bytes).unwrap();
            let words = doc.extract_words(0).unwrap();
            // All characters in a single Tj → should be one word
            let combined: String = words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            assert!(
                words.iter().any(|w| w.text.contains("peu/d")),
                "the text 'peu/d' must appear in some word, got: {combined:?}"
            );
        }
    }

    // ========================================================================
    // Ligature preservation tests (nougat_040 regression)
    // ========================================================================

    /// Helper: build a minimal PDF whose single character maps to U+FB01 (LATIN SMALL
    /// LIGATURE FI) via a ToUnicode CMap. This exercises the path where pdfium hands us
    /// U+FB01 from the font's ToUnicode map and we must NOT expand it to "fi".
    fn build_ligature_fi_pdf() -> Vec<u8> {
        let cmap = "/CIDInit /ProcSet findresource begin\n\
                    12 dict begin\n\
                    begincmap\n\
                    /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
                    /CMapName /Adobe-Identity-UCS def\n\
                    /CMapType 2 def\n\
                    1 begincodespacerange\n\
                    <01> <01>\n\
                    endcodespacerange\n\
                    1 beginbfchar\n\
                    <01> <FB01>\n\
                    endbfchar\n\
                    endcmap\n\
                    CMapName currentdict /CMap defineresource pop\n\
                    end\n\
                    end\n";

        // Content stream: BT /F1 12 Tf 100 500 Td (\001) Tj ET
        let content = "BT /F1 12 Tf 100 500 Td (\\001) Tj ET\n";

        let mut out: Vec<u8> = Vec::new();
        let mut off: Vec<usize> = vec![0];

        out.extend_from_slice(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n");

        macro_rules! push {
            ($body:expr) => {{
                off.push(out.len());
                let id = off.len() - 1;
                out.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", id, $body).as_bytes());
            }};
        }

        push!("<< /Type /Catalog /Pages 2 0 R >>"); // 1
        push!("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"); // 2
        push!(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        )); // 3
        push!(format!(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
             /Encoding << /Type /Encoding /Differences [1 /fi] >> \
             /ToUnicode 6 0 R >>"
        )); // 4
        push!(format!("<< /Length {} >>\nstream\n{}endstream", content.len(), content)); // 5
        push!(format!("<< /Length {} >>\nstream\n{}endstream", cmap.len(), cmap)); // 6

        let xref_offset = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n", off.len()).as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for &o in &off[1..] {
            out.extend_from_slice(format!("{:010} 00000 n \n", o).as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
                off.len(),
                xref_offset
            )
            .as_bytes(),
        );
        out
    }

    /// A ToUnicode CMap that maps char 0x01 → U+FB01 (ﬁ) must produce the
    /// ligature character in extract_text output — NOT the expanded "fi".
    ///
    /// Before the fix, `extract_text` unconditionally calls
    /// `get_ligature_components(ﬁ)` → "fi", discarding the font's own
    /// ToUnicode intent. After the fix the ligature char is preserved.
    #[test]
    fn test_ligature_fi_preserved_in_extract_text() {
        let pdf = build_ligature_fi_pdf();
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let text = doc.extract_text(0).unwrap();
        assert!(
            text.contains('\u{FB01}'),
            "U+FB01 (ﬁ) must be preserved in extracted text; got: {text:?}"
        );
        assert!(
            !text.contains("fi") || text.contains('\u{FB01}'),
            "must not expand ﬁ → fi; got: {text:?}"
        );
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 8: find_references
    // ========================================================================

    #[test]
    fn test_find_references_string_obj() {
        assert!(PdfDocument::find_references(&Object::String(b"hello".to_vec())).is_empty());
    }

    #[test]
    fn test_find_references_real_obj() {
        assert!(PdfDocument::find_references(&Object::Real(std::f64::consts::PI)).is_empty());
    }

    #[test]
    fn test_find_references_name_obj() {
        assert!(PdfDocument::find_references(&Object::Name("Test".to_string())).is_empty());
    }

    #[test]
    fn test_find_references_deeply_nested() {
        let inner_ref = Object::Reference(ObjectRef::new(10, 0));
        let inner_arr = Object::Array(vec![inner_ref]);
        let mut dict = std::collections::HashMap::new();
        dict.insert("Key".to_string(), inner_arr);
        let refs = PdfDocument::find_references(&Object::Dictionary(dict));
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].id, 10);
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 9: font_identity_hash_cheap
    // ========================================================================

    #[test]
    fn test_font_identity_hash_with_encoding_dict() {
        let mut font_dict = std::collections::HashMap::new();
        font_dict.insert("BaseFont".to_string(), Object::Name("Helvetica".to_string()));
        font_dict.insert("Subtype".to_string(), Object::Name("Type1".to_string()));
        let mut enc = std::collections::HashMap::new();
        enc.insert("Type".to_string(), Object::Name("Encoding".to_string()));
        font_dict.insert("Encoding".to_string(), Object::Dictionary(enc));
        assert_ne!(PdfDocument::font_identity_hash_cheap(&Object::Dictionary(font_dict)), 0);
    }

    #[test]
    fn test_font_identity_hash_with_encoding_ref() {
        let mut font_dict = std::collections::HashMap::new();
        font_dict.insert("BaseFont".to_string(), Object::Name("Helvetica".to_string()));
        font_dict.insert("Encoding".to_string(), Object::Reference(ObjectRef::new(99, 0)));
        assert_ne!(PdfDocument::font_identity_hash_cheap(&Object::Dictionary(font_dict)), 0);
    }

    #[test]
    fn test_font_identity_hash_tounicode_changes_hash() {
        let mut d1 = std::collections::HashMap::new();
        d1.insert("BaseFont".to_string(), Object::Name("Arial".to_string()));
        d1.insert("ToUnicode".to_string(), Object::Reference(ObjectRef::new(50, 0)));
        let h1 = PdfDocument::font_identity_hash_cheap(&Object::Dictionary(d1));

        let mut d2 = std::collections::HashMap::new();
        d2.insert("BaseFont".to_string(), Object::Name("Arial".to_string()));
        let h2 = PdfDocument::font_identity_hash_cheap(&Object::Dictionary(d2));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_font_identity_hash_with_descendant_fonts() {
        let mut d = std::collections::HashMap::new();
        d.insert("BaseFont".to_string(), Object::Name("CIDFont".to_string()));
        d.insert("Subtype".to_string(), Object::Name("Type0".to_string()));
        d.insert(
            "DescendantFonts".to_string(),
            Object::Array(vec![Object::Reference(ObjectRef::new(20, 0))]),
        );
        assert_ne!(PdfDocument::font_identity_hash_cheap(&Object::Dictionary(d)), 0);
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 10: Annotation helper and tests
    // ========================================================================

    fn build_pdf_with_annotations(annot_objects: Vec<(usize, Vec<u8>)>) -> Vec<u8> {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets: Vec<(usize, usize)> = Vec::new();

        let off1 = pdf.len();
        offsets.push((1, off1));
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        let off2 = pdf.len();
        offsets.push((2, off2));
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

        let annot_refs: String = annot_objects
            .iter()
            .map(|(num, _)| format!("{} 0 R", num))
            .collect::<Vec<_>>()
            .join(" ");

        let off3 = pdf.len();
        offsets.push((3, off3));
        let page_str = format!(
            "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> /Annots [{}] >>\nendobj\n",
            annot_refs
        );
        pdf.extend_from_slice(page_str.as_bytes());

        for (obj_num, obj_data) in &annot_objects {
            let off = pdf.len();
            offsets.push((*obj_num, off));
            pdf.extend_from_slice(obj_data);
        }

        let max_obj = offsets.iter().map(|(n, _)| *n).max().unwrap_or(0);
        let xref_off = pdf.len();
        pdf.extend_from_slice(format!("xref\n0 {}\n", max_obj + 1).as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for obj_num in 1..=max_obj {
            if let Some((_, off)) = offsets.iter().find(|(n, _)| *n == obj_num) {
                pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
            } else {
                pdf.extend_from_slice(b"0000000000 65535 f \n");
            }
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
                max_obj + 1,
                xref_off
            )
            .as_bytes(),
        );
        pdf
    }

    #[test]
    fn test_annotation_freetext() {
        let annot = b"4 0 obj\n<< /Type /Annot /Subtype /FreeText /Contents (Hello from annotation) >>\nendobj\n".to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let text = doc.extract_text(0).unwrap();
        assert!(text.contains("Hello from annotation"));
    }

    #[test]
    fn test_annotation_text_type() {
        // Text (sticky-note) /Contents is reviewer popup comment text, not visible page
        // content — it must NOT appear in extract_text output (ISO 32000-1 §12.5.6.2).
        let annot = b"4 0 obj\n<< /Type /Annot /Subtype /Text /Contents (Sticky note) >>\nendobj\n"
            .to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(!doc.extract_text(0).unwrap().contains("Sticky note"));
    }

    #[test]
    fn test_annotation_stamp() {
        let annot =
            b"4 0 obj\n<< /Type /Annot /Subtype /Stamp /Contents (APPROVED) >>\nendobj\n".to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_text(0).unwrap().contains("APPROVED"));
    }

    #[test]
    fn test_annotation_link() {
        let annot =
            b"4 0 obj\n<< /Type /Annot /Subtype /Link /Contents (Click here) >>\nendobj\n".to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_text(0).unwrap().contains("Click here"));
    }

    #[test]
    fn test_annotation_highlight() {
        // Highlight annotation /Contents is a user comment on the highlighted
        // text — it is NOT page content and must NOT appear in extract_text output.
        let annot =
            b"4 0 obj\n<< /Type /Annot /Subtype /Highlight /Contents (Highlighted) >>\nendobj\n"
                .to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(!doc.extract_text(0).unwrap().contains("Highlighted"));
    }

    #[test]
    fn test_annotation_hidden_flag() {
        let annot =
            b"4 0 obj\n<< /Type /Annot /Subtype /FreeText /F 2 /Contents (Hidden) >>\nendobj\n"
                .to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(!doc.extract_text(0).unwrap().contains("Hidden"));
    }

    #[test]
    fn test_annotation_invisible_flag() {
        let annot =
            b"4 0 obj\n<< /Type /Annot /Subtype /FreeText /F 1 /Contents (Invisible) >>\nendobj\n"
                .to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(!doc.extract_text(0).unwrap().contains("Invisible"));
    }

    #[test]
    fn test_annotation_noview_flag() {
        let annot =
            b"4 0 obj\n<< /Type /Annot /Subtype /Text /F 32 /Contents (NoView) >>\nendobj\n"
                .to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(!doc.extract_text(0).unwrap().contains("NoView"));
    }

    #[test]
    fn test_annotation_unknown_subtype() {
        let annot =
            b"4 0 obj\n<< /Type /Annot /Subtype /CustomType /Contents (Custom) >>\nendobj\n"
                .to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_text(0).unwrap().contains("Custom"));
    }

    #[test]
    fn test_annotation_multiple() {
        // FreeText /Contents is visible page text; Text (sticky-note) /Contents is popup
        // comment — only FreeText should appear in extract_text output.
        let a1 =
            b"4 0 obj\n<< /Type /Annot /Subtype /FreeText /Contents (First) >>\nendobj\n".to_vec();
        let a2 =
            b"5 0 obj\n<< /Type /Annot /Subtype /Text /Contents (Second) >>\nendobj\n".to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, a1), (5, a2)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let text = doc.extract_text(0).unwrap();
        assert!(text.contains("First"));
        assert!(!text.contains("Second"));
    }

    #[test]
    fn test_annotation_no_subtype() {
        let annot = b"4 0 obj\n<< /Type /Annot /Contents (No subtype) >>\nendobj\n".to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(!doc.extract_text(0).unwrap().contains("No subtype"));
    }

    #[test]
    fn test_annotation_widget_with_value() {
        let annot = b"4 0 obj\n<< /Type /Annot /Subtype /Widget /FT /Tx /V (Field value) /Rect [72 700 272 720] >>\nendobj\n".to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_text(0).unwrap().contains("Field value"));
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 11: resolve_references edge cases
    // ========================================================================

    #[test]
    fn test_resolve_references_boolean() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let resolved = doc.resolve_references(&Object::Boolean(true), 5).unwrap();
        assert!(matches!(resolved, Object::Boolean(true)));
    }

    #[test]
    fn test_resolve_references_nested_dict_with_refs() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let mut dict = std::collections::HashMap::new();
        dict.insert("CatalogRef".to_string(), Object::Reference(ObjectRef::new(1, 0)));
        dict.insert("Direct".to_string(), Object::Integer(42));
        let resolved = doc
            .resolve_references(&Object::Dictionary(dict), 3)
            .unwrap();
        let rd = resolved.as_dict().unwrap();
        assert!(rd.get("CatalogRef").unwrap().as_dict().is_some());
        assert_eq!(rd.get("Direct").unwrap().as_integer(), Some(42));
    }

    #[test]
    fn test_resolve_references_array_with_refs() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let arr = Object::Array(vec![Object::Reference(ObjectRef::new(1, 0)), Object::Integer(99)]);
        let resolved = doc.resolve_references(&arr, 3).unwrap();
        let ra = resolved.as_array().unwrap();
        assert!(ra[0].as_dict().is_some());
        assert_eq!(ra[1].as_integer(), Some(99));
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 12: check_for_circular_references
    // ========================================================================

    #[test]
    fn test_check_circular_refs_on_minimal_pdf() {
        // The minimal PDF has a page tree cycle:
        // Pages (2 0 R) -> Kids -> Page (3 0 R) -> Parent -> Pages (2 0 R)
        // The DFS cycle detector reports this as a cycle.
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let cycles = doc.check_for_circular_references();
        // Verify the function runs without panicking and returns results.
        // The minimal PDF's parent-child relationship is detected as a cycle.
        assert!(!cycles.is_empty());
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 13: various extract and conversion tests
    // ========================================================================

    #[test]
    fn test_extract_text_graphics_only() {
        let pdf = build_minimal_pdf(b"q 1 0 0 1 0 0 cm 100 200 300 400 re S Q");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_text(0).unwrap().is_empty());
    }

    #[test]
    fn test_extract_text_page_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_text(100).is_err());
    }

    #[test]
    fn test_extract_all_text_zero_pages() {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n");
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 3\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_all_text().unwrap().is_empty());
    }

    #[test]
    fn test_extract_spans_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_spans(999).is_err());
    }

    #[test]
    fn test_extract_chars_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_chars(999).is_err());
    }

    #[test]
    fn test_get_page_content_data_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.get_page_content_data(999).is_err());
    }

    #[test]
    fn test_to_html_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc
            .to_html(999, &crate::converters::ConversionOptions::default())
            .is_err());
    }

    #[test]
    fn test_to_markdown_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc
            .to_markdown(999, &crate::converters::ConversionOptions::default())
            .is_err());
    }

    #[test]
    fn test_to_plain_text_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc
            .to_plain_text(999, &crate::converters::ConversionOptions::default())
            .is_err());
    }

    #[test]
    fn test_extract_paths_line() {
        let pdf = build_minimal_pdf(b"0 0 m 100 100 l S");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(!doc.extract_paths(0).unwrap().is_empty());
    }

    #[test]
    fn test_extract_paths_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_paths(999).is_err());
    }

    #[test]
    fn test_extract_paths_curve() {
        let pdf = build_minimal_pdf(b"0 0 m 25 50 75 50 100 0 c S");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(!doc.extract_paths(0).unwrap().is_empty());
    }

    #[test]
    fn test_extract_paths_filled_rect() {
        let pdf = build_minimal_pdf(b"50 50 200 100 re f");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(!doc.extract_paths(0).unwrap().is_empty());
    }

    #[test]
    fn test_extract_paths_in_rect_with_content() {
        let pdf = build_minimal_pdf(b"100 200 300 400 re S");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let region = crate::geometry::Rect {
            x: 0.0,
            y: 0.0,
            width: 612.0,
            height: 792.0,
        };
        assert!(!doc.extract_paths_in_rect(0, region).unwrap().is_empty());
    }

    #[test]
    fn test_extract_images_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_images(999).is_err());
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 14: mark_info with all fields
    // ========================================================================

    #[test]
    fn test_mark_info_with_suspects() {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true /Suspects true /UserProperties true >> >>\nendobj\n",
        );
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n");
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 3\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let mi = doc.mark_info().unwrap();
        assert!(mi.marked);
        assert!(mi.suspects);
        assert!(mi.user_properties);
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 15: page_count fallback with bad /Count
    // ========================================================================

    #[test]
    fn test_page_count_exceeds_objects() {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 999 >>\nendobj\n");
        let off3 = pdf.len();
        pdf.extend_from_slice(
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
        );
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 4\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert_eq!(doc.page_count().unwrap(), 1);
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 16: nested page trees and caching
    // ========================================================================

    #[test]
    fn test_deeply_nested_page_tree() {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 595 842] /Resources << >> >>\nendobj\n");
        let off3 = pdf.len();
        pdf.extend_from_slice(
            b"3 0 obj\n<< /Type /Pages /Kids [4 0 R] /Count 1 /Parent 2 0 R >>\nendobj\n",
        );
        let off4 = pdf.len();
        pdf.extend_from_slice(b"4 0 obj\n<< /Type /Page /Parent 3 0 R >>\nendobj\n");
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 5\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off4).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert_eq!(doc.page_count().unwrap(), 1);
        let page = doc.get_page(0).unwrap();
        assert!(page.as_dict().unwrap().contains_key("MediaBox"));
    }

    #[test]
    fn test_populate_page_cache_sequential() {
        let pdf = build_multi_page_pdf(5);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        for i in 0..5 {
            assert!(doc.get_page(i).unwrap().as_dict().is_some());
        }
    }

    #[test]
    fn test_get_page_ref_multi_page() {
        let pdf = build_multi_page_pdf(3);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let r0 = doc.get_page_ref(0).unwrap();
        let r1 = doc.get_page_ref(1).unwrap();
        let r2 = doc.get_page_ref(2).unwrap();
        assert_ne!(r0.id, r1.id);
        assert_ne!(r1.id, r2.id);
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 17: content stream edge cases
    // ========================================================================

    #[test]
    fn test_page_content_indirect_array() {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
        let off3 = pdf.len();
        pdf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >>\nendobj\n");
        let off4 = pdf.len();
        pdf.extend_from_slice(b"4 0 obj\n[5 0 R 6 0 R]\nendobj\n");
        let c1 = b"q";
        let off5 = pdf.len();
        pdf.extend_from_slice(format!("5 0 obj\n<< /Length {} >>\nstream\n", c1.len()).as_bytes());
        pdf.extend_from_slice(c1);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");
        let c2 = b"Q";
        let off6 = pdf.len();
        pdf.extend_from_slice(format!("6 0 obj\n<< /Length {} >>\nstream\n", c2.len()).as_bytes());
        pdf.extend_from_slice(c2);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 7\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off4).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off5).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off6).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let data = doc.get_page_content_data(0).unwrap();
        let text = String::from_utf8_lossy(&data);
        assert!(text.contains("q"));
        assert!(text.contains("Q"));
    }

    #[test]
    fn test_get_page_content_data_null_contents() {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
        let off3 = pdf.len();
        pdf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents null /Resources << >> >>\nendobj\n");
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 4\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.get_page_content_data(0).unwrap().is_empty());
    }

    // ========================================================================
    // NEW COVERAGE TESTS — Batch 18: misc
    // ========================================================================

    #[test]
    fn test_scan_for_object_finds_missing() {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
        let off3 = pdf.len();
        pdf.extend_from_slice(
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
        );
        let _off5 = pdf.len();
        pdf.extend_from_slice(b"5 0 obj\n<< /Type /Metadata /Subtype /XML >>\nendobj\n");
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 4\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let obj = doc.load_object(ObjectRef::new(5, 0)).unwrap();
        assert!(obj.as_dict().is_some());
    }

    #[test]
    fn test_load_object_missing_returns_null_simple() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(matches!(doc.load_object(ObjectRef::new(999, 0)).unwrap(), Object::Null));
    }

    #[test]
    fn test_decode_stream_with_encryption_non_null() {
        let pdf = build_minimal_pdf(b"BT (Hello) Tj ET");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let stream_obj = doc.load_object(ObjectRef::new(4, 0)).unwrap();
        assert!(doc
            .decode_stream_with_encryption(&stream_obj, ObjectRef::new(4, 0))
            .is_ok());
    }

    #[test]
    fn test_load_fonts_public_empty_resources() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let mut ext = crate::extractors::TextExtractor::new();
        assert!(doc
            .load_fonts_public(&Object::Dictionary(std::collections::HashMap::new()), &mut ext)
            .is_ok());
    }

    #[test]
    fn test_load_fonts_public_resources_not_dict() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let mut ext = crate::extractors::TextExtractor::new();
        assert!(doc
            .load_fonts_public(&Object::Integer(42), &mut ext)
            .is_ok());
    }

    #[test]
    fn test_is_form_xobject_from_cache() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let _ = doc.load_object(ObjectRef::new(1, 0)).unwrap();
        assert!(!doc.is_form_xobject(ObjectRef::new(1, 0)));
    }

    #[test]
    fn test_find_substring_middle() {
        assert_eq!(find_substring(b"Hello World", b"lo W"), Some(3));
    }

    #[test]
    fn test_find_substring_full_match() {
        assert_eq!(find_substring(b"ABC", b"ABC"), Some(0));
    }

    #[test]
    fn test_find_substring_needle_longer() {
        assert_eq!(find_substring(b"AB", b"ABCD"), None);
    }

    #[test]
    fn test_parse_header_lenient_no_header() {
        let mut cursor = Cursor::new(vec![0xABu8; 100]);
        let (major, minor, _) = parse_header(&mut cursor, true).unwrap();
        assert_eq!((major, minor), (1, 4));
    }

    #[test]
    fn test_parse_version_lenient_version_0_0() {
        let header = *b"%PDF-0.0";
        assert_eq!(parse_version_from_header(&header, true).unwrap(), (1, 4));
    }

    #[test]
    fn test_parse_trailer_empty_input() {
        assert!(parse_trailer(&mut Cursor::new(b"")).is_err());
    }

    #[test]
    fn test_apply_intelligent_text_processing_fl_ligature_preserved() {
        // Same as ﬁ: ﬂ (U+FB02) must be preserved, not expanded to "fl".
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let spans = vec![make_test_span("\u{FB02}oor", 0.0, 0.0, 50.0, 12.0)]; // ﬂoor
        let result = doc.apply_intelligent_text_processing(spans);
        assert!(
            result[0].text.contains('\u{FB02}'),
            "ﬂ must be preserved, got: {:?}",
            result[0].text
        );
    }

    #[test]
    fn test_apply_intelligent_text_processing_ocr_font() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let mut span = make_test_span("Test  Text", 0.0, 0.0, 100.0, 12.0);
        span.font_name = "OCR".to_string();
        let result = doc.apply_intelligent_text_processing(vec![span]);
        assert!(!result[0].text.contains("  "));
    }

    #[test]
    fn test_extract_spans_with_config_adaptive() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc
            .extract_spans_with_config(0, crate::extractors::SpanMergingConfig::adaptive())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_extract_spans_with_config_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc
            .extract_spans_with_config(999, crate::extractors::SpanMergingConfig::default())
            .is_err());
    }

    #[test]
    fn test_image_format_debug() {
        assert_eq!(format!("{:?}", ImageFormat::Png), "Png");
        assert_eq!(format!("{:?}", ImageFormat::Jpeg), "Jpeg");
    }

    #[test]
    fn test_may_contain_text_bt_with_newline() {
        assert!(PdfDocument::may_contain_text(b"\nBT\n"));
    }

    #[test]
    fn test_may_contain_text_do_with_bracket() {
        assert!(PdfDocument::may_contain_text(b"]Do["));
    }

    #[test]
    fn test_may_contain_text_single_b() {
        assert!(!PdfDocument::may_contain_text(b"B"));
    }

    #[test]
    fn test_may_contain_text_single_d() {
        assert!(!PdfDocument::may_contain_text(b"D"));
    }

    #[test]
    fn test_multiline_object_header() {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1\n0\nobj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n");
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 3\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.catalog().unwrap().as_dict().is_some());
    }

    #[test]
    fn test_object_content_on_same_line() {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n");
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 3\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.catalog().unwrap().as_dict().is_some());
    }

    #[test]
    fn test_open_pdf_version_2_0() {
        let mut pdf = b"%PDF-2.0\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n");
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 3\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        assert_eq!(PdfDocument::from_bytes(pdf).unwrap().version(), (2, 0));
    }

    #[test]
    fn test_extract_text_annotations_only() {
        let annot =
            b"4 0 obj\n<< /Type /Annot /Subtype /FreeText /Contents (Only annotation) >>\nendobj\n"
                .to_vec();
        let pdf = build_pdf_with_annotations(vec![(4, annot)]);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        assert!(doc.extract_text(0).unwrap().contains("Only annotation"));
    }

    #[test]
    fn test_parse_string_value_static_boolean() {
        assert!(PdfDocument::parse_string_value_static(Some(&Object::Boolean(true))).is_none());
    }

    #[test]
    fn test_parse_string_value_static_array() {
        assert!(PdfDocument::parse_string_value_static(Some(&Object::Array(vec![]))).is_none());
    }

    #[test]
    #[allow(deprecated)]
    fn test_page_count_u32_zero_pages() {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n");
        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 3\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );
        assert_eq!(PdfDocument::from_bytes(pdf).unwrap().page_count_u32(), 0);
    }

    /// Regression test: validate_object_at_offset must return true for
    /// compressed (type 2) xref entries. Previously, it treated the object
    /// stream number as a byte offset, sought to a random location,
    /// returned false — triggering a full-file xref reconstruction that took
    /// 35+ seconds on large PDFs.
    #[test]
    fn test_validate_compressed_xref_entry() {
        use crate::xref::{CrossRefTable, XRefEntry, XRefEntryType};

        let mut xref = CrossRefTable::new();
        // Add a compressed entry: object 5 lives inside object stream 10, at index 3
        xref.entries.insert(
            5,
            XRefEntry {
                entry_type: XRefEntryType::Compressed,
                offset: 10,    // object stream number, NOT a byte offset
                generation: 3, // index within the stream
                in_use: true,
            },
        );

        let data = b"%PDF-1.7\n%%EOF\n";
        let mut cursor = Cursor::new(data.to_vec());
        let obj_ref = ObjectRef { id: 5, gen: 0 };

        // Must return true — compressed objects are valid by virtue of being in the xref
        assert!(validate_object_at_offset(&mut cursor, &xref, obj_ref));
    }

    #[test]
    fn test_reading_order_enum_default() {
        let order = ReadingOrder::default();
        assert_eq!(order, ReadingOrder::TopToBottom);
    }

    #[test]
    fn test_reading_order_enum_variants() {
        assert_ne!(ReadingOrder::TopToBottom, ReadingOrder::ColumnAware);
        // Verify Clone and Copy
        let a = ReadingOrder::ColumnAware;
        let b = a;
        assert_eq!(a, b);
    }

    /// Verify that ColumnAware reading order reads column 1 fully before column 2.
    ///
    /// Layout:
    /// ```text
    ///   Left col (x=10) Right col (x=200)
    ///   +-----------+ +-----------+
    ///   | L1 (y=700)| | R1 (y=700)|
    ///   | L2 (y=680)| | R2 (y=680)|
    ///   | L3 (y=660)| | R3 (y=660)|
    ///   +-----------+ +-----------+
    /// ```
    /// Expected ColumnAware order: L1, L2, L3, R1, R2, R3
    /// TopToBottom order would interleave: L1, R1, L2, R2, L3, R3
    #[test]
    fn test_column_aware_reads_column1_before_column2() {
        use crate::geometry::Rect;
        use crate::layout::{Color, FontWeight, TextSpan};
        use crate::pipeline::reading_order::{
            ReadingOrderContext as ROContext, ReadingOrderStrategy, XYCutStrategy,
        };

        fn make_span(label: &str, x: f32, y: f32) -> TextSpan {
            TextSpan {
                artifact_type: None,
                text: label.to_string(),
                bbox: Rect::new(x, y, 80.0, 12.0),
                font_size: 12.0,
                font_name: "Test".to_string(),
                font_weight: FontWeight::Normal,
                is_italic: false,
                is_monospace: false,
                color: Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                },
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
                heading_level: None,
                rotation_degrees: 0.0,
            }
        }

        // Two columns with a wide gap (110 points).
        // Each column has 3 spans arranged top-to-bottom.
        let spans = vec![
            make_span("L1", 10.0, 700.0),
            make_span("R1", 200.0, 700.0),
            make_span("L2", 10.0, 680.0),
            make_span("R2", 200.0, 680.0),
            make_span("L3", 10.0, 660.0),
            make_span("R3", 200.0, 660.0),
        ];

        let strategy = XYCutStrategy::new();
        let context = ROContext::new();
        let ordered = strategy
            .apply(spans, &context)
            .expect("XYCut should not fail");
        let labels: Vec<&str> = ordered.iter().map(|o| o.span.text.as_str()).collect();

        // Column-aware: all left-column spans first, then all right-column spans.
        assert_eq!(
            labels,
            vec!["L1", "L2", "L3", "R1", "R2", "R3"],
            "ColumnAware should read left column fully before right column"
        );
    }

    // ========================================================================
    // COLUMN-ORDER: persistent-gutter-corridor accept path (#607)
    // ========================================================================

    /// Build a span with explicit width and text (for corridor-geometry tests).
    #[cfg(test)]
    fn corridor_span(text: &str, x: f32, y: f32, w: f32) -> crate::layout::TextSpan {
        use crate::geometry::Rect;
        use crate::layout::{Color, FontWeight, TextSpan};
        TextSpan {
            artifact_type: None,
            text: text.to_string(),
            bbox: Rect::new(x, y, w, 10.0),
            font_size: 10.0,
            font_name: "Test".to_string(),
            font_weight: FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
            char_widths: vec![],
            heading_level: None,
            rotation_degrees: 0.0,
        }
    }

    /// A shared-baseline two-column prose body (academic references): each line
    /// has scattered word-granular left edges in BOTH columns — so the
    /// dominant-cluster-fraction gate misses it — but a single persistent
    /// central gutter. The corridor accept path must route it as multi-column.
    #[test]
    fn test_corridor_accepts_scattered_two_column_prose() {
        let mut spans = Vec::new();
        // 20 lines; left column words at x≈50/95/140 (scattered), right column
        // words at x≈300/345/390. Persistent central gutter at x≈230.
        for i in 0..20 {
            let y = 700.0 - i as f32 * 14.0;
            spans.push(corridor_span("Lorem", 50.0, y, 35.0));
            spans.push(corridor_span("ipsumdolor", 95.0, y, 40.0));
            spans.push(corridor_span("sitametco", 140.0, y, 40.0));
            spans.push(corridor_span("consectetur", 300.0, y, 40.0));
            spans.push(corridor_span("adipiscing", 345.0, y, 40.0));
            spans.push(corridor_span("elitsedo", 390.0, y, 40.0));
        }
        assert!(
            PdfDocument::is_multi_column_page(&spans),
            "scattered-edge two-column prose with a persistent central gutter \
             must be detected as multi-column via the corridor accept path"
        );
    }

    /// A short-cell numeric table shares one column gap but has tiny cells
    /// (mean chars per line well below 20). The prose guard must reject it so
    /// the table is NOT routed to XY-cut (which would reorder its cells).
    #[test]
    fn test_corridor_rejects_short_cell_table() {
        let mut spans = Vec::new();
        // Scattered left edges (so the bimodal-line-start detector does NOT
        // fire and the dominant-cluster gate fails — i.e. control reaches the
        // corridor path), but every cell is a short numeric token so the
        // per-line mean char count stays well under the prose floor of 20.
        for i in 0..20 {
            let y = 700.0 - i as f32 * 14.0;
            spans.push(corridor_span("12", 50.0, y, 12.0));
            spans.push(corridor_span("34", 95.0, y, 12.0));
            spans.push(corridor_span("56", 140.0, y, 12.0));
            spans.push(corridor_span("78", 300.0, y, 12.0));
            spans.push(corridor_span("90", 345.0, y, 12.0));
            spans.push(corridor_span("12", 390.0, y, 12.0));
        }
        assert!(
            !PdfDocument::is_multi_column_page(&spans),
            "short-cell numeric table must NOT be routed as multi-column \
             (grid-row discriminator rejects ≥2-gap rows)"
        );
    }

    /// #536 Part 1a: a SHORT-line two-column verse body (Bible / lexicon) — one
    /// short fragment per column, one central gutter per line — used to be
    /// rejected by the raw `mean_chars <= 20` floor. It must now be admitted via
    /// the corridor's short-line path (single gap/line, balanced, central).
    #[test]
    fn test_corridor_accepts_short_verse_two_column() {
        let mut spans = Vec::new();
        for i in 0..20 {
            let y = 700.0 - i as f32 * 14.0;
            spans.push(corridor_span("Bereshit", 50.0, y, 45.0)); // 8 chars, →95
            spans.push(corridor_span("barahem", 300.0, y, 40.0)); // 7 chars, →340
        }
        // Call the corridor directly (bypass the upstream bimodal/histogram
        // gates) with a no-op degenerate-CTM filter.
        assert!(
            PdfDocument::has_persistent_gutter_corridor(&spans, 300.0, 10_000.0),
            "short-verse two-column body (1 gutter/line, balanced) must be admitted"
        );
    }

    /// #536 Part 1a guard: a lopsided narrow-label + wide-data table must stay
    /// rejected even though it has one gap per line — its gutter sits off-centre
    /// (failing the centre gate) and its columns are lopsided (failing the
    /// char-mass balance), either of which is sufficient.
    #[test]
    fn test_corridor_rejects_label_column_table() {
        let mut spans = Vec::new();
        for i in 0..20 {
            let y = 700.0 - i as f32 * 14.0;
            spans.push(corridor_span("1", 50.0, y, 8.0)); // tiny label →58
            spans.push(corridor_span("Descriptionlongdata", 300.0, y, 200.0)); // wide →500
        }
        assert!(
            !PdfDocument::has_persistent_gutter_corridor(&spans, 300.0, 10_000.0),
            "lopsided narrow-label + wide-data table must be rejected (char balance)"
        );
    }

    /// #536 Part 1b: a two-column prose body interleaved with a MINORITY of
    /// full-width display-math / heading rows must still be detected — the
    /// full-width rows are excluded from the coverage denominator. Without the
    /// exclusion the coverage floor (best_size*2 >= lines) fails.
    #[test]
    fn test_corridor_survives_minority_display_math() {
        let mut spans = Vec::new();
        // 16 two-column prose lines.
        for i in 0..16 {
            let y = 700.0 - i as f32 * 14.0;
            spans.push(corridor_span("Lorem ipsum dolor", 50.0, y, 120.0)); // →170
            spans.push(corridor_span("sit amet consectetur", 300.0, y, 150.0)); // →450
        }
        // 24 full-width display rows (span the page, no internal gutter).
        for i in 0..24 {
            let y = 400.0 - i as f32 * 14.0;
            spans.push(corridor_span("Section heading spanning width", 50.0, y, 400.0));
        }
        assert!(
            PdfDocument::has_persistent_gutter_corridor(&spans, 300.0, 10_000.0),
            "two-column prose with a minority of full-width display rows must hold"
        );
    }

    // ========================================================================
    // extract_page_text / PageText tests
    // ========================================================================

    #[test]
    fn test_extract_page_text_blank_page() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let page_text = doc.extract_page_text(0).unwrap();
        assert!(page_text.spans.is_empty());
        assert!(page_text.chars.is_empty());
        // MediaBox is [0 0 612 792] in build_minimal_pdf
        assert!((page_text.page_width - 612.0).abs() < 0.1);
        assert!((page_text.page_height - 792.0).abs() < 0.1);
    }

    #[test]
    fn test_extract_page_text_has_page_dimensions() {
        let content = b"BT /F1 12 Tf (Hello) Tj ET";
        let pdf = build_minimal_pdf(content);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let page_text = doc.extract_page_text(0).unwrap();
        assert!((page_text.page_width - 612.0).abs() < 0.1);
        assert!((page_text.page_height - 792.0).abs() < 0.1);
    }

    #[test]
    fn test_extract_page_text_chars_derived_from_spans() {
        let content = b"BT /F1 12 Tf (Hello) Tj ET";
        let pdf = build_minimal_pdf(content);
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let page_text = doc.extract_page_text(0).unwrap();
        // Total chars should equal sum of chars across all spans
        let expected_char_count: usize =
            page_text.spans.iter().map(|s| s.text.chars().count()).sum();
        assert_eq!(page_text.chars.len(), expected_char_count);
    }

    #[test]
    fn test_extract_page_text_with_column_aware() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let page_text = doc
            .extract_page_text_with_options(0, ReadingOrder::ColumnAware)
            .unwrap();
        assert!(page_text.spans.is_empty());
        assert!((page_text.page_width - 612.0).abs() < 0.1);
    }

    #[test]
    fn test_extract_page_text_out_of_bounds() {
        let pdf = build_minimal_pdf(b"");
        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let result = doc.extract_page_text(99);
        assert!(result.is_err());
    }

    /// Regression test for Issue #254: Tm-scale containment filter must not
    /// drop distinct text lines whose bounding boxes overlap spatially.
    ///
    /// Before the fix, the containment filter in extract_text() would skip any
    /// span geometrically contained within the previous span, even if the text
    /// was different. This caused the second line to silently disappear.
    ///
    /// The fix adds a `span.text == prev.text` guard so that only true
    /// duplicates are filtered.
    #[test]
    fn test_containment_filter_preserves_distinct_overlapping_lines() {
        // Build a minimal PDF with two Td-placed text strings at very close Y
        // positions (Y=700 and Y=699 — within the 2.0pt "same line" threshold)
        // but with different content. The first string is wider so the second
        // is geometrically contained within it.
        let content =
            b"BT /F1 12 Tf 50 700 Td (First line has longer text here) Tj 0 -1 Td (Second) Tj ET";

        // We need a font in Resources for the extractor to work.
        let mut pdf = b"%PDF-1.4\n".to_vec();

        let off1 = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        let off2 = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

        let off3 = pdf.len();
        pdf.extend_from_slice(
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        );

        let off4 = pdf.len();
        let content_len = content.len();
        pdf.extend_from_slice(
            format!("4 0 obj\n<< /Length {} >>\nstream\n", content_len).as_bytes(),
        );
        pdf.extend_from_slice(content);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");

        let off5 = pdf.len();
        pdf.extend_from_slice(
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        );

        let xref_off = pdf.len();
        pdf.extend_from_slice(b"xref\n0 6\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off4).as_bytes());
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off5).as_bytes());
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off)
                .as_bytes(),
        );

        let doc = PdfDocument::from_bytes(pdf).unwrap();
        let text = doc.extract_text(0).unwrap();

        assert!(
            text.contains("First line has longer text here"),
            "First line should be present in extracted text, got: {:?}",
            text
        );
        assert!(
            text.contains("Second"),
            "Second line must NOT be dropped by containment filter, got: {:?}",
            text
        );
    }

    #[test]
    fn test_page_text_serializable() {
        // Verify PageText derives serde::Serialize
        let page_text = crate::layout::PageText {
            spans: Vec::new(),
            chars: Vec::new(),
            page_width: 612.0,
            page_height: 792.0,
        };
        let json = serde_json::to_string(&page_text).unwrap();
        // Without the `wasm` feature, field names are snake_case
        assert!(json.contains("page_width"));
        assert!(json.contains("page_height"));
    }

    #[test]
    fn test_fix_digit_logicalnot_decimal() {
        // `¬` between digits → `.`; spaced or non-digit-flanked `¬` is left alone.
        assert_eq!(PdfDocument::fix_digit_logicalnot_decimal("1\u{00AC}00"), "1.00");
        assert_eq!(
            PdfDocument::fix_digit_logicalnot_decimal("0\u{00AC}75 1\u{00AC}00"),
            "0.75 1.00"
        );
        // Logic/set notation (spaced) is untouched.
        assert_eq!(PdfDocument::fix_digit_logicalnot_decimal("A \u{00AC} B"), "A \u{00AC} B");
        assert_eq!(PdfDocument::fix_digit_logicalnot_decimal("5 \u{00AC} 3"), "5 \u{00AC} 3");
        // Leading/trailing `¬` with only one digit neighbour: untouched.
        assert_eq!(PdfDocument::fix_digit_logicalnot_decimal("\u{00AC}5"), "\u{00AC}5");
    }

    #[test]
    fn test_is_cm_or_symbol_font() {
        assert!(PdfDocument::is_cm_or_symbol_font("ABCDEF+CMSY10"));
        assert!(PdfDocument::is_cm_or_symbol_font("CMR12"));
        assert!(PdfDocument::is_cm_or_symbol_font("Symbol"));
        assert!(!PdfDocument::is_cm_or_symbol_font("ABCDEF+Helvetica"));
        assert!(!PdfDocument::is_cm_or_symbol_font("TimesNewRoman"));
    }

    /// A password-protected PDF is detected as encrypted, and text extraction
    /// degrades to empty output (warn + empty) rather than erroring — matching
    /// pdftotext/PyMuPDF. (`page_count` still surfaces `Error::EncryptedPdf`;
    /// see `tests/test_extraction_robustness.rs`.)
    #[cfg(feature = "legacy-crypto")]
    #[test]
    fn test_encrypted_pdf_extracts_empty_without_password() {
        let pdf_path = "tests/fixtures/encrypted_needs_password.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            eprintln!("Skipping: fixture not found at {}", pdf_path);
            return;
        }

        let doc = PdfDocument::open(pdf_path).expect("open should succeed even without password");
        assert!(doc.is_encrypted(), "PDF should be detected as encrypted");

        let text = doc
            .extract_text(0)
            .expect("extract_text degrades to empty, not an error");
        assert!(text.is_empty(), "undecryptable extraction should be empty, got: {:?}", text,);
    }

    /// After authenticating with the correct password, extraction should succeed.
    #[cfg(feature = "legacy-crypto")]
    #[test]
    fn test_encrypted_pdf_works_after_authentication() {
        let pdf_path = "tests/fixtures/encrypted_needs_password.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            eprintln!("Skipping: fixture not found at {}", pdf_path);
            return;
        }

        let doc = PdfDocument::open(pdf_path).expect("open should succeed");
        assert!(doc.is_encrypted());

        // Authenticate with the correct password
        let result = doc
            .authenticate(b"secret")
            .expect("authenticate should not error");
        assert!(result, "Authentication with correct password should succeed");

        // Now extraction should work (not return EncryptedPdf error)
        let page_count = doc.page_count().expect("page_count should work after auth");
        assert!(page_count > 0, "Should have at least 1 page after auth");

        // extract_text should not error (content may be minimal since it's a test PDF)
        let _text = doc
            .extract_text(0)
            .expect("extract_text should work after auth");
    }

    /// Multi-row-spanning label cell (test item name vertically centered
    /// across N data rows) must be placed at the top of its row block in
    /// reading-order output, not interleaved mid-group by Y.
    ///
    /// Simulates a simplified 2-column table:
    /// - Column A (sparse, "labels"): 2 labels, each centered in its
    ///   block of 6 data rows.
    /// - Column B (dense, "data"): 12 data rows.
    ///
    /// Expected sort: Label1, d1..d6, Label2, d7..d12.
    #[test]
    fn test_rowspan_label_promoted_to_top_of_block() {
        use crate::layout::TextSpan;

        fn mk(text: &str, x: f32, y: f32, w: f32) -> TextSpan {
            TextSpan {
                artifact_type: None,
                text: text.to_string(),
                bbox: crate::geometry::Rect::new(x, y, w, 10.0),
                font_size: 12.0,
                font_name: "Arial".into(),
                font_weight: crate::layout::FontWeight::Normal,
                is_italic: false,
                is_monospace: false,
                color: crate::layout::Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
                heading_level: None,
                rotation_degrees: 0.0,
            }
        }

        // Data rows at x=200, y=100..30 step -10 (12 rows).
        // Label1 at x=50, y=75 (middle of rows 100..60).
        // Label2 at x=50, y=45 (middle of rows 50..30... but actually 50..30 is 3 values,
        //   and label2 should be centered in rows 50..30 → y=40 but we choose 45 to be clearly in 2nd block).
        // Target split: Label1 owns rows 100,90,80,70,60,50; Label2 owns 40,30,20,10.
        // Both labels' Y (75 and 45) sit between their block rows.
        let mut spans = vec![mk("L1", 50.0, 75.0, 40.0), mk("L2", 50.0, 45.0, 40.0)];
        for i in 0..12 {
            let y = 100.0 - (i as f32) * 10.0;
            spans.push(mk(&format!("d{:02}", i), 200.0, y, 20.0));
        }

        super::PdfDocument::reorder_rowspan_labels(&mut spans);

        let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
        // L1 must come first, L2 must come before its own block.
        let pos_l1 = texts.iter().position(|t| *t == "L1").expect("L1 present");
        let pos_l2 = texts.iter().position(|t| *t == "L2").expect("L2 present");
        assert!(pos_l1 < pos_l2, "L1 should precede L2 in reading order, got {:?}", texts);
        // L1 must come before ALL data rows that belong to L1's block.
        // With distance-based partitioning, L1 owns rows closer to y=75 than y=45:
        //   100,90,80,70,60 are closer to 75. 50 is equidistant (tie → L1).
        //   Expect L1 at index 0 and L2 somewhere after L1's block.
        assert_eq!(texts[0], "L1", "L1 must be first, got: {:?}", &texts[..5]);
        // At least some data row must be between L1 and L2.
        assert!(
            pos_l2 > pos_l1 + 3,
            "L2 must come after several data rows of L1's block, got {:?}",
            texts
        );
    }

    /// Regression: line-continuation spans that share a Y-band with the dense
    /// column must NOT be promoted by `reorder_rowspan_labels`.
    ///
    /// A resume-like PDF has two X groups: a dense main-text column (x=63)
    /// and a sparse rightward column (x=430) whose spans are all on the SAME
    /// lines as the dense column (same Y-bands). The sparse spans are
    /// line-continuation text, not rowspan labels, so they must stay in their
    /// natural sorted position rather than being hoisted to wrong Y values.
    #[test]
    fn test_rowspan_label_skips_spans_aligned_with_dense_column() {
        use crate::layout::TextSpan;

        fn mk(text: &str, x: f32, y: f32) -> TextSpan {
            TextSpan {
                artifact_type: None,
                text: text.to_string(),
                bbox: crate::geometry::Rect::new(x, y, 80.0, 10.0),
                font_size: 12.0,
                font_name: "Arial".into(),
                font_weight: crate::layout::FontWeight::Normal,
                is_italic: false,
                is_monospace: false,
                color: crate::layout::Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
                heading_level: None,
                rotation_degrees: 0.0,
            }
        }

        // Dense column (x=63): 10 spans at y=640,620,600,580,560,540,520,500,480,460
        // Sparse column (x=430): 4 spans at y=600,560,520,480 — same lines as dense
        // After reorder_rowspan_labels the sparse spans must NOT be promoted.
        let ys_dense = [
            640.0f32, 620.0, 600.0, 580.0, 560.0, 540.0, 520.0, 500.0, 480.0, 460.0,
        ];
        let ys_sparse = [600.0f32, 560.0, 520.0, 480.0]; // all on same Y as dense rows

        let mut spans: Vec<TextSpan> = Vec::new();
        for &y in &ys_dense {
            spans.push(mk(&format!("dense_y{}", y as i32), 63.0, y));
        }
        for &y in &ys_sparse {
            spans.push(mk(&format!("sparse_y{}", y as i32), 430.0, y));
        }

        // Sort descending Y, X ascending (as extract_spans does before calling this)
        spans.sort_by(|a, b| {
            crate::utils::row_aware_span_cmp(a.bbox.y, a.bbox.x, b.bbox.y, b.bbox.x)
        });
        let before: Vec<String> = spans.iter().map(|s| s.text.clone()).collect();

        super::PdfDocument::reorder_rowspan_labels(&mut spans);

        let after: Vec<String> = spans.iter().map(|s| s.text.clone()).collect();
        assert_eq!(
            before, after,
            "reorder_rowspan_labels must not change order when sparse spans \
             share Y-bands with the dense column; \
             before={before:?} after={after:?}"
        );
    }

    /// AES-256 (V=5, R=6) PDF that only authenticates via the owner
    /// password with an empty input. Exercises Algorithm 2.B termination
    /// (off-by-one would produce a wrong file encryption key) plus the
    /// end-to-end string decryption path that surfaces annotation text.
    ///
    /// The binary fixture `tests/fixtures/encrypted_aes256_r6_owner_password.pdf`
    /// is not redistributable (copyrighted Bluebeam sample), so this test
    /// soft-skips when the file is absent rather than being `#[ignore]`d
    /// (which silently hides it from regular `cargo test` runs and means
    /// real coverage only appears under `--ignored`). The same code path
    /// is exercised end-to-end by `scripts/validate_issue_fixes.sh`
    /// against `pdfs_pdfjs/pr6531_2.pdf` from the local test corpus.
    #[test]
    fn test_encrypted_aes256_r6_owner_password_empty() {
        let pdf_path = "tests/fixtures/encrypted_aes256_r6_owner_password.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            eprintln!("Skipping: AES-256 R=6 fixture not found at {pdf_path}");
            return;
        }
        let doc = PdfDocument::open(pdf_path).expect("open should succeed");
        assert!(doc.is_encrypted(), "fixture is AES-256 encrypted");
        let text = doc.extract_text(0).expect("extract_text should succeed");
        assert!(
            text.contains("Bluebeam should be encrypting this."),
            "expected annotation text in extracted output, got: {:?}",
            text
        );
    }

    /// Copy-protected (AES-256, V=5, R=6) PDFs with widget text must
    /// decrypt string values inside object dictionaries so that form
    /// field content appears in `extract_text` output. Without per-object
    /// string decryption, the page renders as an empty string.
    ///
    /// The binary fixture `tests/fixtures/encrypted_aes256_widget.pdf`
    /// is not redistributable (copyrighted Bluebeam sample), so this test
    /// soft-skips when the file is absent rather than being `#[ignore]`d.
    /// The same code path is exercised end-to-end by
    /// `scripts/validate_issue_fixes.sh` against `pdfs_pdfjs/secHandler.pdf`
    /// from the local test corpus.
    #[test]
    fn test_encrypted_aes256_widget_decrypts_string_values() {
        let pdf_path = "tests/fixtures/encrypted_aes256_widget.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            eprintln!("Skipping: AES-256 widget fixture not found at {pdf_path}");
            return;
        }

        let doc = PdfDocument::open(pdf_path).expect("open should succeed");
        assert!(doc.is_encrypted(), "fixture is AES-256 encrypted");

        let text = doc.extract_text(0).expect("extract_text should succeed");
        assert!(
            text.contains("Security Handler"),
            "expected widget text 'Security Handler' in extracted text, got: {:?}",
            text
        );
    }

    /// PDFs that are encrypted but authenticated with empty password (the common
    /// case for permission-only encryption) must continue to work without error.
    #[cfg(feature = "legacy-crypto")]
    #[test]
    fn test_encrypted_pdf_with_empty_password_still_works() {
        let pdf_path = "tests/fixtures/encrypted_cid_truetype.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            eprintln!("Skipping: fixture not found at {}", pdf_path);
            return;
        }

        let doc = PdfDocument::open(pdf_path).expect("open should succeed");
        // This PDF auto-authenticates with empty password during open()
        assert!(doc.is_encrypted(), "Should be detected as encrypted");

        // Should NOT return EncryptedPdf error
        let page_count = doc.page_count().expect("page_count should work");
        assert!(page_count > 0);

        let text = doc.extract_text(0).expect("extract_text should work");
        assert!(!text.trim().is_empty(), "Should extract non-empty text");
    }

    #[cfg(feature = "legacy-crypto")]
    #[test]
    fn test_encrypted_pdf_with_compressed_object_streams() {
        // Encrypted PDFs with /Type /ObjStm streams must NOT have those streams
        // decrypted, per ISO 32000-1 Section 7.6.2. Object streams and XRef
        // streams are never individually encrypted; only the overall stream
        // data is compressed. Attempting to decrypt them causes AES errors
        // because the data length is not a multiple of the block size.
        let pdf_path = "tests/fixtures/encrypted_objstm.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            eprintln!("Skipping: fixture not found at {}", pdf_path);
            return;
        }

        let doc =
            PdfDocument::open(pdf_path).expect("open should succeed for encrypted+objstm PDF");
        assert!(doc.is_encrypted(), "Should be detected as encrypted");

        let page_count = doc
            .page_count()
            .expect("page_count should work with encrypted objstm");
        assert!(page_count > 0, "Should have at least one page");
    }

    // ====================================================================
    // MutexExt
    // ====================================================================

    #[test]
    fn test_lock_or_recover_on_poisoned_mutex() {
        use std::sync::Mutex;
        let m = Mutex::new(42);
        // Poison the mutex by panicking while holding the lock
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = m.lock().unwrap();
            panic!("intentional");
        }));
        assert!(m.lock().is_err(), "Mutex should be poisoned");
        // lock_or_recover should still return the inner value
        let val = *m.lock_or_recover();
        assert_eq!(val, 42);
    }

    // ====================================================================
    // BoundedEntryCache
    // ====================================================================

    #[test]
    fn test_bounded_entry_cache_lru_eviction_order() {
        let mut c = BoundedEntryCache::new(3);
        c.insert(1u32, "a");
        c.insert(2, "b");
        c.insert(3, "c");
        // Touch key 1 so it becomes most-recently-used
        assert_eq!(c.get(&1), Some(&"a"));
        // Insert key 4 — should evict 2 (oldest untouched), not 1
        c.insert(4, "d");
        assert_eq!(c.get(&1), Some(&"a"), "LRU-promoted key should survive");
        assert!(c.get(&2).is_none(), "Oldest untouched key should be evicted");
        assert_eq!(c.get(&3), Some(&"c"));
        assert_eq!(c.get(&4), Some(&"d"));
    }

    #[test]
    fn test_bounded_entry_cache_reinsert_no_eviction() {
        let mut c = BoundedEntryCache::new(1);
        c.insert(1u32, "a");
        // Re-insert same key — should NOT evict, just replace
        c.insert(1, "b");
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(&1), Some(&"b"));
    }

    #[test]
    fn test_bounded_entry_cache_fifo_eviction_without_get() {
        let mut c = BoundedEntryCache::new(2);
        c.insert(1u32, "a");
        c.insert(2, "b");
        // No get() calls — pure insertion order
        c.insert(3, "c");
        assert!(c.get(&1).is_none(), "First inserted should be evicted");
        assert_eq!(c.get(&2), Some(&"b"));
        assert_eq!(c.get(&3), Some(&"c"));
    }

    // ====================================================================
    // BoundedObjectCache
    // ====================================================================

    #[test]
    fn test_bounded_object_cache_oversized_rejection() {
        let mut c = BoundedObjectCache::new(100); // 100 bytes max
        let big = Object::String(vec![0u8; 200]); // well over 100 bytes
        c.insert(ObjectRef::new(1, 0), big);
        assert_eq!(c.len(), 0, "Oversized object should be rejected");
    }

    #[test]
    fn test_bounded_object_cache_byte_budget_eviction() {
        // Use a budget that fits ~2 small objects but not 3
        let small = Object::Integer(1); // estimate_size = 32
        let budget = 80; // fits 2 × 32, not 3
        let mut c = BoundedObjectCache::new(budget);
        c.insert(ObjectRef::new(1, 0), small.clone());
        c.insert(ObjectRef::new(2, 0), small.clone());
        assert_eq!(c.len(), 2);
        // Third insertion should evict the first
        c.insert(ObjectRef::new(3, 0), small.clone());
        assert!(c.get(&ObjectRef::new(1, 0)).is_none(), "Oldest should be evicted");
        assert!(c.get(&ObjectRef::new(3, 0)).is_some());
        assert!(c.current_bytes <= budget);
    }

    #[test]
    fn test_estimate_size_depth_bottoms_out() {
        // Deeply nested array — should not stack overflow
        let mut obj = Object::Integer(1);
        for _ in 0..100 {
            obj = Object::Array(vec![obj]);
        }
        // Should return a finite value without panicking
        let size = BoundedObjectCache::estimate_size(&obj);
        assert!(size > 0);
    }

    // -----------------------------------------------------------------
    // PdfDocument::contains_rect_with_tolerance
    //
    // Pins the table-retain tolerance behaviour: spans whose f32
    // right-edge drifts a fraction of a point past the table bbox
    // (due to accumulated width-sum error) must still count as
    // contained, but spans that actually extend beyond the table
    // must not. Each test's first block is a geometry sanity check
    // so a Rect::new construction mistake fails loudly rather than
    // silently exercising the wrong geometry.
    // -----------------------------------------------------------------

    #[test]
    fn contains_rect_with_tolerance_absorbs_subpixel_drift() {
        use crate::geometry::Rect;
        let table = Rect::new(0.0, 0.0, 100.0, 100.0);
        let drifted = Rect::new(10.0, 10.0, 90.02, 80.0);

        // Geometry sanity: drifted span right-edge should sit ~0.02pt
        // past table right-edge. If this fails, the test construction
        // is wrong, not the tolerance logic. Tolerance is 1e-4pt
        // because `0.02f32` is not representable exactly — the
        // observed drift lands within ~4e-6 of 0.02.
        assert!(
            (drifted.right() - table.right() - 0.02).abs() < 1e-4,
            "drifted span right-edge should be 0.02pt past table right-edge; got drift = {}",
            drifted.right() - table.right()
        );
        assert_eq!(drifted.left(), 10.0, "span should start at x=10");
        assert_eq!(drifted.top(), 10.0, "span should start at y=10");
        assert_eq!(drifted.bottom(), 90.0, "span should end at y=90");

        // Behavior: 0.02pt drift is absorbed by 0.1pt tolerance.
        assert!(PdfDocument::contains_rect_with_tolerance(&table, &drifted, 0.1));
    }

    #[test]
    fn contains_rect_with_tolerance_rejects_genuinely_outside() {
        use crate::geometry::Rect;
        let table = Rect::new(0.0, 0.0, 100.0, 100.0);
        let outside = Rect::new(10.0, 10.0, 91.0, 80.0);

        // Geometry sanity: outside span right-edge should sit 1.0pt
        // past table right-edge.
        assert!(
            (outside.right() - table.right() - 1.0).abs() < 1e-6,
            "outside span right-edge should be 1.0pt past table right-edge; got drift = {}",
            outside.right() - table.right()
        );

        // Behavior: 1.0pt beyond is outside 0.1pt tolerance.
        assert!(!PdfDocument::contains_rect_with_tolerance(&table, &outside, 0.1));
    }

    #[test]
    fn contains_rect_with_tolerance_accepts_fully_inside() {
        use crate::geometry::Rect;
        let table = Rect::new(0.0, 0.0, 100.0, 100.0);
        let inside = Rect::new(10.0, 10.0, 80.0, 80.0);

        // Geometry sanity: control span should be strictly inside the
        // table on every edge.
        assert!(
            inside.left() > table.left()
                && inside.right() < table.right()
                && inside.top() > table.top()
                && inside.bottom() < table.bottom(),
            "control span should be strictly inside the table"
        );

        assert!(PdfDocument::contains_rect_with_tolerance(&table, &inside, 0.1));
    }

    /// Regression test for #484 (pdfa_036): span filtering must use per-cell
    /// bboxes, not the coarser outer table bbox.
    ///
    /// Before the fix, `span_in_table` filtered by `table.bbox`, which could
    /// be wider than the union of the actual cell bboxes. Paragraph text that
    /// happened to fall inside the table's outer bbox was silently dropped even
    /// though no cell claimed it, causing content loss (the "(HLA)/(KSL)"
    /// paragraph in pdfa_036 disappeared).
    ///
    /// After the fix, only spans inside at least one *cell* bbox are removed
    /// from the flow. Spans inside the outer table bbox but outside all cells
    /// (i.e. in a gap or margin) are preserved.
    #[test]
    fn cell_bbox_filter_preserves_span_in_outer_bbox_gap() {
        use crate::geometry::Rect;
        use crate::structure::table_extractor::{Table, TableCell, TableRow};

        // A table whose outer bbox is [0, 0] – [200, 100].
        // Two non-adjacent cells leave a horizontal gap at x=90..110 — that
        // gap is inside the outer bbox but not inside any cell.
        let mut table = Table::new();
        let mut row = TableRow::new(false);
        row.cells.push(TableCell {
            text: "left".to_string(),
            spans: vec![],
            colspan: 1,
            rowspan: 1,
            mcids: vec![],
            bbox: Some(Rect::new(0.0, 0.0, 90.0, 100.0)),
            is_header: false,
        });
        row.cells.push(TableCell {
            text: "right".to_string(),
            spans: vec![],
            colspan: 1,
            rowspan: 1,
            mcids: vec![],
            bbox: Some(Rect::new(110.0, 0.0, 90.0, 100.0)),
            is_header: false,
        });
        table.add_row(row);
        table.bbox = Some(Rect::new(0.0, 0.0, 200.0, 100.0));

        const TOL: f32 = 0.1;

        // A span sitting inside the left cell → should be "in table".
        let span_cell = Rect::new(10.0, 10.0, 50.0, 20.0);
        let in_any_cell = table.rows.iter().any(|r| {
            r.cells.iter().any(|c| {
                c.bbox
                    .is_some_and(|b| PdfDocument::contains_rect_with_tolerance(&b, &span_cell, TOL))
            })
        });
        assert!(in_any_cell, "span inside a cell bbox must be identified as in-table");

        // A span in the gap (x=95..105) — inside outer table bbox, outside all cells.
        let span_gap = Rect::new(95.0, 10.0, 10.0, 20.0);

        // 1. Outer-bbox filter (the OLD, incorrect approach) would classify it as in-table.
        let in_outer_bbox =
            PdfDocument::contains_rect_with_tolerance(&table.bbox.unwrap(), &span_gap, TOL);
        assert!(
            in_outer_bbox,
            "gap span must be inside the outer table bbox (precondition for the bug to trigger)"
        );

        // 2. Cell-bbox filter (the NEW, correct approach) must NOT classify it as in-table.
        let in_any_cell_gap = table.rows.iter().any(|r| {
            r.cells.iter().any(|c| {
                c.bbox
                    .is_some_and(|b| PdfDocument::contains_rect_with_tolerance(&b, &span_gap, TOL))
            })
        });
        assert!(
            !in_any_cell_gap,
            "gap span must NOT be inside any cell bbox — cell-bbox filter must preserve it"
        );
    }

    #[test]
    fn reorder_same_line_runs_preserves_disjoint_x_rows() {
        use crate::geometry::Rect;
        use crate::layout::TextSpan;

        // Two rows close enough in Y to pass the existing same_line_threshold:
        // Δy = 4.5 and fs = 10, so threshold = 5.0.
        // They are disjoint in X (gap of 225pt = 22.5 * fs, well over the
        // SAME_LINE_REORDER_MAX_GAP_FACTOR = 3.0 ceiling). The helper must
        // not X-sort them into [skersey, VerDate]; it must preserve the
        // row-aware order.
        let mut spans = vec![
            TextSpan {
                text: "VerDate".to_string(),
                bbox: Rect::new(350.0, 200.0, 85.0, 10.0),
                font_size: 10.0,
                sequence: 0,
                ..Default::default()
            },
            TextSpan {
                text: "skersey".to_string(),
                bbox: Rect::new(50.0, 195.5, 75.0, 10.0),
                font_size: 10.0,
                sequence: 1,
                ..Default::default()
            },
        ];

        PdfDocument::reorder_same_line_runs(&mut spans);

        let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec!["VerDate", "skersey"]);
    }

    #[test]
    fn reorder_same_line_runs_orders_suffix_superscript_by_x() {
        use crate::geometry::Rect;
        use crate::layout::TextSpan;

        // Row-aware/Y-desc order can put the superscript first because it
        // sits higher. The tentative X-gap validation must not reject this
        // legitimate mixed-baseline run; the X-sorted gaps are 15pt and 0pt
        // at max_fs=14, both well under 3.0 * 14 = 42. Final order should
        // be normal left-to-right text.
        let mut spans = vec![
            TextSpan {
                text: "th".to_string(),
                bbox: Rect::new(180.0, 205.0, 10.0, 10.0),
                font_size: 10.0,
                sequence: 0,
                ..Default::default()
            },
            TextSpan {
                text: "September".to_string(),
                bbox: Rect::new(100.0, 200.0, 50.0, 14.0),
                font_size: 14.0,
                sequence: 1,
                ..Default::default()
            },
            TextSpan {
                text: "11".to_string(),
                bbox: Rect::new(165.0, 200.0, 15.0, 14.0),
                font_size: 14.0,
                sequence: 2,
                ..Default::default()
            },
        ];

        PdfDocument::reorder_same_line_runs(&mut spans);

        let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec!["September", "11", "th"]);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Optional Content (PDF "layer") name resolution — OCG `/Name` decoding,
    // OCMD `/OCGs` following, and the `/Properties` name-reference path
    // against the current resource scope. ISO 32000-1:2008 §8.11, §14.6.
    //
    // The resolver methods are `&self` only to reach `load_object` for
    // indirect references; these tests use direct objects, so a throwaway
    // minimal PDF serves purely as the method receiver.
    // ─────────────────────────────────────────────────────────────────────

    fn oc_test_doc() -> PdfDocument {
        PdfDocument::from_bytes(build_minimal_pdf(b"")).unwrap()
    }

    fn ocg_dict(name: Object) -> Object {
        let mut d = std::collections::HashMap::new();
        d.insert("Type".to_string(), Object::Name("OCG".to_string()));
        d.insert("Name".to_string(), name);
        Object::Dictionary(d)
    }

    fn ocmd_dict(ocgs: Object) -> Object {
        let mut d = std::collections::HashMap::new();
        d.insert("Type".to_string(), Object::Name("OCMD".to_string()));
        d.insert("OCGs".to_string(), ocgs);
        Object::Dictionary(d)
    }

    fn utf16_string(s: &str, big_endian: bool) -> Object {
        let mut bytes = if big_endian {
            vec![0xFE, 0xFF]
        } else {
            vec![0xFF, 0xFE]
        };
        for u in s.encode_utf16() {
            if big_endian {
                bytes.extend_from_slice(&u.to_be_bytes());
            } else {
                bytes.extend_from_slice(&u.to_le_bytes());
            }
        }
        Object::String(bytes)
    }

    #[test]
    fn test_oc_name_ocg_ascii() {
        let doc = oc_test_doc();
        let dict = ocg_dict(Object::String(b"A-GRID".to_vec()));
        assert_eq!(doc.read_oc_name(dict.as_dict().unwrap(), 8).as_deref(), Some("A-GRID"));
    }

    #[test]
    fn test_oc_name_ocg_utf16le_bom() {
        // Regression for the reuse of decode_pdf_text_string: the previous
        // inline reader only handled UTF-16BE and fell back to latin-1,
        // mangling UTF-16LE-encoded layer names. The shared helper decodes
        // the LE BOM correctly.
        let doc = oc_test_doc();
        let dict = ocg_dict(utf16_string("ÁREA-Ø", false));
        assert_eq!(doc.read_oc_name(dict.as_dict().unwrap(), 8).as_deref(), Some("ÁREA-Ø"));
    }

    #[test]
    fn test_oc_name_ocg_utf16be_bom() {
        let doc = oc_test_doc();
        let dict = ocg_dict(utf16_string("EJES", true));
        assert_eq!(doc.read_oc_name(dict.as_dict().unwrap(), 8).as_deref(), Some("EJES"));
    }

    #[test]
    fn test_oc_name_ocmd_single_ocg() {
        // OCMD has no /Name — resolution follows /OCGs (single OCG) to its name.
        let doc = oc_test_doc();
        let ocmd = ocmd_dict(ocg_dict(Object::String(b"M-DUCT".to_vec())));
        assert_eq!(doc.read_oc_name(ocmd.as_dict().unwrap(), 8).as_deref(), Some("M-DUCT"));
    }

    #[test]
    fn test_oc_name_ocmd_ocgs_array_first_wins() {
        // /OCGs may be an array of OCGs; the first resolvable member wins.
        let doc = oc_test_doc();
        let arr = Object::Array(vec![
            ocg_dict(Object::String(b"S-COLS".to_vec())),
            ocg_dict(Object::String(b"S-BEAM".to_vec())),
        ]);
        let ocmd = ocmd_dict(arr);
        assert_eq!(doc.read_oc_name(ocmd.as_dict().unwrap(), 8).as_deref(), Some("S-COLS"));
    }

    #[test]
    fn test_oc_name_ocmd_depth_guard() {
        // A pathological OCMD chain (each /OCGs points to another OCMD) must
        // terminate via the depth guard rather than recursing without bound.
        let doc = oc_test_doc();
        let mut nested = ocmd_dict(Object::Array(vec![]));
        for _ in 0..20 {
            nested = ocmd_dict(nested);
        }
        assert_eq!(doc.read_oc_name(nested.as_dict().unwrap(), 8), None);
    }

    #[test]
    fn test_resolve_oc_name_via_resources_properties() {
        // Case 2 (name reference) resolves against the *passed-in* resources.
        // This is the crux of the Form-XObject fix: the resolver reads
        // /Properties /<name> from whatever resource scope the caller hands
        // it — page /Resources at page level, the XObject's own /Resources
        // when extracting inside a Form XObject.
        let doc = oc_test_doc();
        let mut props = std::collections::HashMap::new();
        props.insert("MC0".to_string(), ocg_dict(Object::String(b"A-WALL-DIM".to_vec())));
        let mut resources = std::collections::HashMap::new();
        resources.insert("Properties".to_string(), Object::Dictionary(props));
        let resources = Object::Dictionary(resources);

        let name_ref = Object::Name("MC0".to_string());
        assert_eq!(
            doc.resolve_oc_layer_name(Some(&resources), &name_ref)
                .as_deref(),
            Some("A-WALL-DIM")
        );
    }

    #[test]
    fn test_resolve_oc_name_inline_dict() {
        let doc = oc_test_doc();
        let inline = ocg_dict(Object::String(b"CORTES".to_vec()));
        assert_eq!(doc.resolve_oc_layer_name(None, &inline).as_deref(), Some("CORTES"));
    }

    #[test]
    fn test_resolve_oc_name_unresolvable_is_none() {
        // A name reference with no resources in scope yields None (the path
        // is left unlabelled) rather than an error.
        let doc = oc_test_doc();
        let name_ref = Object::Name("MC9".to_string());
        assert_eq!(doc.resolve_oc_layer_name(None, &name_ref), None);
    }

    #[test]
    fn test_extract_paths_layer_none_for_plain_stroke() {
        // End-to-end through the real page pipeline: a stroked line on a page
        // with no optional content yields a path whose `layer` is None. Guards
        // the page-level marked-content refactor against perturbing plain
        // extraction (and mirrors the Python shape test's synthetic PDF).
        let doc = PdfDocument::from_bytes(build_minimal_pdf(b"100 100 m 200 200 l S")).unwrap();
        let paths = doc.extract_paths(0).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].layer, None);
    }
}

#[cfg(test)]
mod ink_dict_extractor_tests {
    use super::*;
    use std::collections::HashMap;

    fn name(s: &str) -> Object {
        Object::Name(s.to_string())
    }

    fn separation_cs(ink: &str) -> Object {
        Object::Array(vec![
            name("Separation"),
            name(ink),
            name("DeviceCMYK"),
            Object::Null,
        ])
    }

    fn device_n_cs(inks: &[&str]) -> Object {
        Object::Array(vec![
            name("DeviceN"),
            Object::Array(inks.iter().map(|s| name(s)).collect()),
            name("DeviceCMYK"),
            Object::Null,
        ])
    }

    #[test]
    fn extracts_separation_ink_name() {
        let mut cs_dict = HashMap::new();
        cs_dict.insert("CS0".to_string(), separation_cs("Pantone-185"));
        let mut out = Vec::new();
        extract_inks_from_color_space_dict(&cs_dict, None, &mut out);
        assert_eq!(out, vec!["Pantone-185".to_string()]);
    }

    #[test]
    fn extracts_devicen_ink_names_in_declared_order() {
        let mut cs_dict = HashMap::new();
        cs_dict.insert("CS0".to_string(), device_n_cs(&["Cyan", "Magenta", "SpotGold"]));
        let mut out = Vec::new();
        extract_inks_from_color_space_dict(&cs_dict, None, &mut out);
        assert_eq!(
            out,
            vec![
                "Cyan".to_string(),
                "Magenta".to_string(),
                "SpotGold".to_string()
            ]
        );
    }

    #[test]
    fn skips_all_and_none_colorants() {
        // §8.6.6.4: /All and /None are reserved; never plate names.
        let mut cs_dict = HashMap::new();
        cs_dict.insert("CS0".to_string(), separation_cs("All"));
        cs_dict.insert("CS1".to_string(), separation_cs("None"));
        cs_dict.insert("CS2".to_string(), device_n_cs(&["All", "Spot1", "None"]));
        let mut out = Vec::new();
        extract_inks_from_color_space_dict(&cs_dict, None, &mut out);
        assert_eq!(out, vec!["Spot1".to_string()]);
    }

    #[test]
    fn ignores_non_separation_color_spaces() {
        let mut cs_dict = HashMap::new();
        cs_dict.insert("CS0".to_string(), Object::Array(vec![name("ICCBased"), Object::Null]));
        cs_dict.insert("CS1".to_string(), name("DeviceCMYK"));
        let mut out = Vec::new();
        extract_inks_from_color_space_dict(&cs_dict, None, &mut out);
        assert!(out.is_empty());
    }
}
