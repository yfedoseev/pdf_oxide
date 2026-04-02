//! Process-level cross-document font cache.
//!
//! When processing multiple PDFs in sequence or parallel, the same font programs
//! (e.g., Helvetica, Times-Roman, Arial) appear repeatedly across documents.
//! This module provides a global LRU cache keyed by font identity hash so that
//! already-parsed `FontInfo` instances can be reused across documents without
//! re-parsing.
//!
//! The cache is safe for concurrent access from multiple threads and uses an
//! LRU eviction policy to bound memory usage.
//!
//! # Usage
//!
//! The cache is integrated automatically into `PdfDocument::load_fonts()`.
//! For explicit memory management, call [`clear_global_font_cache()`] between
//! batches or when reclaiming memory.
//!
//! ```no_run
//! use pdf_oxide::fonts::global_cache::{clear_global_font_cache, global_font_cache_stats};
//!
//! // Process many PDFs... fonts are cached automatically
//!
//! // Check cache statistics
//! let (size, capacity) = global_font_cache_stats();
//! println!("Global font cache: {}/{} entries", size, capacity);
//!
//! // Clear when done to free memory
//! clear_global_font_cache();
//! ```

use super::FontInfo;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Default maximum number of entries in the global font cache.
/// 1024 fonts covers typical batch processing workloads (most documents use
/// 5-15 fonts, so this handles ~70-200 unique documents before eviction).
const DEFAULT_MAX_ENTRIES: usize = 1024;

/// A simple LRU cache for font entries.
///
/// Uses a generation counter to track access recency. When the cache exceeds
/// `max_entries`, the least-recently-used entry is evicted.
struct LruFontCache {
    /// Map from font identity hash to (font, access generation).
    entries: HashMap<u64, (Arc<FontInfo>, u64)>,
    /// Monotonically increasing generation counter, incremented on every access.
    generation: u64,
    /// Maximum number of entries before LRU eviction kicks in.
    max_entries: usize,
}

impl LruFontCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(max_entries / 2),
            generation: 0,
            max_entries,
        }
    }

    /// Look up a font by identity hash. Returns `Some(Arc<FontInfo>)` on hit
    /// and updates the entry's access generation (marking it as recently used).
    fn get(&mut self, key: u64) -> Option<Arc<FontInfo>> {
        if let Some((font, gen)) = self.entries.get_mut(&key) {
            self.generation += 1;
            *gen = self.generation;
            Some(Arc::clone(font))
        } else {
            None
        }
    }

    /// Insert a font into the cache. If the cache is full, the least-recently-used
    /// entry is evicted first.
    fn insert(&mut self, key: u64, font: Arc<FontInfo>) {
        // If key already exists, just update it
        if self.entries.contains_key(&key) {
            self.generation += 1;
            self.entries.insert(key, (font, self.generation));
            return;
        }

        // Evict LRU entry if at capacity
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        self.generation += 1;
        self.entries.insert(key, (font, self.generation));
    }

    /// Evict the least-recently-used entry (lowest generation number).
    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, (_, gen))| *gen)
            .map(|(k, _)| *k);
        if let Some(key) = lru_key {
            self.entries.remove(&key);
        }
    }

    /// Return the current number of cached entries.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return the maximum capacity.
    fn capacity(&self) -> usize {
        self.max_entries
    }

    /// Remove all entries.
    fn clear(&mut self) {
        self.entries.clear();
        self.generation = 0;
    }

    /// Update the maximum capacity. If the new capacity is smaller, evict
    /// entries until the cache fits.
    fn set_capacity(&mut self, new_max: usize) {
        self.max_entries = new_max;
        while self.entries.len() > self.max_entries {
            self.evict_lru();
        }
    }
}

/// Global singleton font cache, initialized on first access.
static GLOBAL_FONT_CACHE: OnceLock<Mutex<LruFontCache>> = OnceLock::new();

/// Get or initialize the global font cache.
fn cache() -> &'static Mutex<LruFontCache> {
    GLOBAL_FONT_CACHE.get_or_init(|| Mutex::new(LruFontCache::new(DEFAULT_MAX_ENTRIES)))
}

/// Look up a font in the global cross-document cache.
///
/// Returns `Some(Arc<FontInfo>)` if a font with the given identity hash was
/// previously cached (from any document), or `None` on cache miss.
///
/// This is called from `PdfDocument::load_fonts()` before attempting to parse
/// a font from its dictionary.
pub fn global_font_cache_get(identity_hash: u64) -> Option<Arc<FontInfo>> {
    // If the lock is poisoned (another thread panicked), skip the cache
    // rather than propagating the panic — font loading should still work.
    cache().lock().ok()?.get(identity_hash)
}

/// Insert a parsed font into the global cross-document cache.
///
/// Called after successfully parsing a `FontInfo` from a font dictionary.
/// The font is keyed by its identity hash so that structurally identical fonts
/// in other documents will get a cache hit.
pub fn global_font_cache_insert(identity_hash: u64, font: Arc<FontInfo>) {
    if let Ok(mut guard) = cache().lock() {
        guard.insert(identity_hash, font);
    }
}

/// Clear all entries from the global font cache.
///
/// Call this between batch processing runs or when you need to reclaim memory.
/// Subsequent `PdfDocument` operations will rebuild the cache as fonts are
/// encountered.
///
/// # Example
///
/// ```no_run
/// use pdf_oxide::fonts::global_cache::clear_global_font_cache;
///
/// // After processing a batch of PDFs
/// clear_global_font_cache();
/// ```
pub fn clear_global_font_cache() {
    if let Ok(mut guard) = cache().lock() {
        guard.clear();
    }
}

/// Return the current size and capacity of the global font cache.
///
/// Returns `(current_entries, max_capacity)`.
///
/// # Example
///
/// ```no_run
/// use pdf_oxide::fonts::global_cache::global_font_cache_stats;
///
/// let (size, capacity) = global_font_cache_stats();
/// println!("Global font cache: {}/{} entries", size, capacity);
/// ```
pub fn global_font_cache_stats() -> (usize, usize) {
    cache()
        .lock()
        .map(|guard| (guard.len(), guard.capacity()))
        .unwrap_or((0, 0))
}

/// Set the maximum number of entries in the global font cache.
///
/// If `max_entries` is smaller than the current cache size, the least-recently-used
/// entries are evicted immediately. Setting this to 0 effectively disables caching
/// (all inserts are immediately evicted).
///
/// # Example
///
/// ```no_run
/// use pdf_oxide::fonts::global_cache::set_global_font_cache_capacity;
///
/// // Increase cache for large batch processing
/// set_global_font_cache_capacity(4096);
/// ```
pub fn set_global_font_cache_capacity(max_entries: usize) {
    if let Ok(mut guard) = cache().lock() {
        guard.set_capacity(max_entries);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fonts::font_dict::Encoding;
    use std::collections::HashMap;

    /// Create a minimal FontInfo for testing.
    fn make_test_font(name: &str) -> FontInfo {
        FontInfo {
            base_font: name.to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 600.0,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        }
    }

    // ---- Unit tests for LruFontCache (no global state, safe for parallel execution) ----

    #[test]
    fn test_lru_cache_insert_and_get() {
        let mut cache = LruFontCache::new(16);
        let font = Arc::new(make_test_font("Helvetica"));

        assert!(cache.get(100).is_none());

        cache.insert(100, Arc::clone(&font));
        let cached = cache.get(100);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().base_font, "Helvetica");
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache = LruFontCache::new(3);

        cache.insert(10, Arc::new(make_test_font("F1")));
        cache.insert(20, Arc::new(make_test_font("F2")));
        cache.insert(30, Arc::new(make_test_font("F3")));

        // Access F1 to make it recently used (F2 becomes LRU)
        cache.get(10);

        // Insert F4 — should evict F2 (least recently used)
        cache.insert(40, Arc::new(make_test_font("F4")));

        assert!(cache.get(10).is_some(), "F1 should still be cached");
        assert!(cache.get(20).is_none(), "F2 should have been evicted");
        assert!(cache.get(30).is_some(), "F3 should still be cached");
        assert!(cache.get(40).is_some(), "F4 should be cached");
    }

    #[test]
    fn test_lru_cache_clear() {
        let mut cache = LruFontCache::new(16);
        cache.insert(1, Arc::new(make_test_font("A")));
        cache.insert(2, Arc::new(make_test_font("B")));
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get(1).is_none());
    }

    #[test]
    fn test_lru_cache_set_capacity() {
        let mut cache = LruFontCache::new(16);

        for i in 0..5 {
            cache.insert(i, Arc::new(make_test_font(&format!("Font{}", i))));
        }
        assert_eq!(cache.len(), 5);

        // Shrink to 2 — should evict 3 LRU entries
        cache.set_capacity(2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.capacity(), 2);
    }

    #[test]
    fn test_lru_cache_duplicate_key_update() {
        let mut cache = LruFontCache::new(16);

        cache.insert(50, Arc::new(make_test_font("OldFont")));
        assert_eq!(cache.get(50).unwrap().base_font, "OldFont");

        cache.insert(50, Arc::new(make_test_font("NewFont")));
        assert_eq!(cache.get(50).unwrap().base_font, "NewFont");
        assert_eq!(cache.len(), 1, "Duplicate key should not increase size");
    }

    #[test]
    fn test_lru_cache_generation_ordering() {
        // Verify that get() updates generation, affecting eviction order
        let mut cache = LruFontCache::new(3);

        cache.insert(1, Arc::new(make_test_font("A"))); // gen 1
        cache.insert(2, Arc::new(make_test_font("B"))); // gen 2
        cache.insert(3, Arc::new(make_test_font("C"))); // gen 3

        // Touch keys 1 and 3 (making 2 the LRU)
        cache.get(1); // gen 4
        cache.get(3); // gen 5

        // Insert 4th — should evict key 2 (gen 2 is lowest)
        cache.insert(4, Arc::new(make_test_font("D")));
        assert!(cache.get(2).is_none(), "Key 2 should be evicted");
        assert!(cache.get(1).is_some());
        assert!(cache.get(3).is_some());
        assert!(cache.get(4).is_some());
    }

    // ---- Integration test for global API (single test to avoid parallel interference) ----

    #[test]
    fn test_global_api_insert_get_clear_stats() {
        // Use very high unique keys to avoid collisions with other tests
        let key_base = 9_000_000u64;

        // Insert
        let font = Arc::new(make_test_font("GlobalTestFont"));
        global_font_cache_insert(key_base, Arc::clone(&font));

        // Get (hit)
        let cached = global_font_cache_get(key_base);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().base_font, "GlobalTestFont");

        // Get (miss)
        assert!(global_font_cache_get(key_base + 999).is_none());

        // Stats: size should be at least 1
        let (size, cap) = global_font_cache_stats();
        assert!(size >= 1);
        assert!(cap > 0);

        // Clear and verify
        clear_global_font_cache();
        assert!(global_font_cache_get(key_base).is_none());

        let (size_after, _) = global_font_cache_stats();
        assert_eq!(size_after, 0);

        // Restore capacity in case another test changed it
        set_global_font_cache_capacity(DEFAULT_MAX_ENTRIES);
    }
}
