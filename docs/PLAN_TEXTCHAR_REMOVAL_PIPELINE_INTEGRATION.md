# Implementation Plan: TextChar Removal, Pipeline Integration, and Spurious Spaces Fix

## Executive Summary

This plan details the refactoring needed to:
1. **Remove TextChar and TextBlock types** from the codebase, keeping only `TextSpan`
2. **Wire up the new pipeline module** (`src/pipeline/`) to `PdfDocument` API
3. **Integrate StructureTreeStrategy** with actual structure tree extraction
4. **Fix spurious spaces issues** identified in markdown output

The refactoring follows SOLID principles, ensuring single responsibility for each module and clean separation between extraction, ordering, and conversion layers.

---

## 1. Current Architecture Analysis

### Type Hierarchy (Current State)

```
src/layout/text_block.rs:
├── TextSpan      - Complete text strings from Tj/TJ (PDF spec compliant) - KEEP
├── TextChar      - Individual characters with positions - REMOVE
└── TextBlock     - Grouped text from chars or spans - REMOVE
```

### Files Using TextChar (18 files)

| File | Usage Pattern | Migration Difficulty |
|------|---------------|---------------------|
| `src/layout/text_block.rs` | Definition + TextBlock.from_chars() | High (core type) |
| `src/layout/mod.rs` | Re-export | Low |
| `src/layout/clustering.rs` | DBSCAN on chars | High (rewrite to spans) |
| `src/layout/document_analyzer.rs` | Font/size analysis from chars | Medium |
| `src/layout/reading_order.rs` | Uses TextBlock (not TextChar directly) | Low |
| `src/extractors/text.rs` | Creates TextChar in extract() | High (dual mode) |
| `src/converters/markdown.rs` | convert_page() uses TextChar | Medium (deprecated path) |
| `src/converters/html.rs` | convert_page() uses TextChar | Medium (deprecated path) |
| `src/bin/export_to_markdown.rs` | Uses TextChar for legacy path | Medium |
| `src/hybrid/smart_analyzer.rs` | Uses TextBlock | Low |
| `src/hybrid/complexity_estimator.rs` | Uses TextBlock | Low |
| `tests/test_layout.rs` | Test helpers | Medium |
| `tests/test_converters.rs` | Test helpers | Medium |
| `tests/test_markdown_extraction_quality.rs` | Test helpers | Medium |

### Files Using TextBlock (19 files)

Most TextBlock usages are in converters and layout modules. TextBlock is an intermediate representation that groups TextChar or wraps TextSpan.

---

## 2. Target Architecture

### New Type Flow

```
PDF Content Stream
        │
        ▼
[TextExtractor.extract_spans()]
        │
        ▼
   Vec<TextSpan>  ◄── Single intermediate representation
        │
        ▼
[TextPipeline.process()]
        │
        ├── StructureTreeStrategy (Tagged PDFs)
        ├── GeometricStrategy (Multi-column)
        └── SimpleStrategy (Default)
        │
        ▼
  Vec<OrderedTextSpan>
        │
        ▼
[OutputConverter.convert()]
        │
        ├── MarkdownOutputConverter
        ├── HtmlOutputConverter
        └── PlainTextConverter
        │
        ▼
    String Output
```

### Key Principle: TextSpan is the ONLY Intermediate Representation

Per PDF Spec ISO 32000-1:2008 Section 9.4.4 NOTE 6, text strings should be "as long as possible". TextSpan preserves this intent.

---

## 3. Phase 1: Prepare for Removal (Non-Breaking Changes)

### Phase 1.1: Mark TextChar/TextBlock as Deprecated

**File**: `src/layout/text_block.rs`

```rust
/// [DEPRECATED] Use TextSpan instead.
///
/// TextChar will be removed in v2.0. For character-level analysis,
/// iterate over TextSpan.text.chars() with geometric calculations.
#[deprecated(since = "1.1.0", note = "Use TextSpan instead")]
#[derive(Debug, Clone)]
pub struct TextChar { ... }

/// [DEPRECATED] Use TextSpan with OrderedTextSpan instead.
///
/// TextBlock will be removed in v2.0. Use the pipeline module:
/// `TextPipeline::process(spans, context)` returns `Vec<OrderedTextSpan>`.
#[deprecated(since = "1.1.0", note = "Use TextSpan with pipeline module")]
#[derive(Debug, Clone)]
pub struct TextBlock { ... }
```

**Effort**: Small (S)
**Risk**: None (backward compatible)

### Phase 1.2: Remove TextExtractor.extract() Method (Char Mode)

**File**: `src/extractors/text.rs`

The `extract()` method (line 1483) returns `Vec<TextChar>`. This is only used by deprecated paths.

**Changes**:
1. Mark `extract()` as `#[deprecated]`
2. Remove `chars: Vec<TextChar>` field from `TextExtractor` struct
3. Remove `extract_spans: bool` flag (always true now)

**Effort**: Medium (M)
**Risk**: Low (extract_spans is the preferred method)

### Phase 1.3: Add Pipeline Module to lib.rs

**File**: `src/lib.rs`

```rust
// Add after line 127
/// Text extraction pipeline with pluggable strategies
pub mod pipeline;

// Add to re-exports (around line 149)
pub use pipeline::{
    TextPipeline, TextPipelineConfig, ReadingOrderStrategy,
    OrderedTextSpan, OutputConverter,
};
```

**Effort**: Small (S)
**Risk**: None

---

## 4. Phase 2: Integrate Pipeline with PdfDocument API

### Phase 2.1: Add Pipeline-Based Methods to PdfDocument

**File**: `src/document.rs`

Add new methods that use the pipeline:

```rust
impl PdfDocument {
    /// Extract text using the new pipeline (recommended).
    pub fn extract_text_pipeline(
        &mut self,
        page_index: usize,
        config: &TextPipelineConfig,
    ) -> Result<String> {
        let spans = self.extract_spans(page_index)?;
        let pipeline = TextPipeline::with_config(config.clone());

        // Build context with structure tree if available
        let context = self.build_reading_order_context(page_index)?;

        let ordered = pipeline.process(spans, context)?;

        // Use appropriate converter
        let converter = create_converter(config.output.format);
        converter.convert(&ordered, config)
    }

    /// Build reading order context for a page.
    fn build_reading_order_context(&mut self, page_index: usize) -> Result<ReadingOrderContext> {
        let mut context = ReadingOrderContext::new()
            .with_page(page_index as u32);

        // Add page bbox
        if let Ok(page) = self.get_page(page_index) {
            if let Some(media_box) = Self::get_page_media_box(&page) {
                context = context.with_bbox(media_box);
            }
        }

        // Add structure tree MCID order if available
        if let Ok(Some(struct_tree)) = self.structure_tree() {
            let mcid_order = extract_reading_order(&struct_tree, page_index as u32)?;
            if !mcid_order.is_empty() {
                context = context.with_mcid_order(mcid_order);
            }
        }

        Ok(context)
    }

    /// Convert page to markdown using pipeline.
    pub fn to_markdown_pipeline(
        &mut self,
        page_index: usize,
        config: &TextPipelineConfig,
    ) -> Result<String> {
        self.extract_text_pipeline(page_index, config)
    }
}
```

**Effort**: Large (L)
**Risk**: Medium (new API path, needs thorough testing)
**Acceptance Criteria**:
- New methods work identically to existing ones for simple PDFs
- Tagged PDF MCIDs are correctly propagated to context
- All existing tests still pass

### Phase 2.2: Wire StructureTreeStrategy to Real Structure Tree

**File**: `src/pipeline/reading_order/structure_tree.rs`

The current implementation already handles MCID ordering correctly. Integration point is in `ReadingOrderContext.with_mcid_order()`.

**Changes needed in PdfDocument**:
1. Call `traverse_structure_tree()` from `src/structure/traversal.rs`
2. Extract MCIDs using `extract_reading_order()`
3. Pass to `ReadingOrderContext::with_mcid_order()`

This is already sketched in Phase 2.1 above.

**Effort**: Small (S) - already mostly implemented
**Risk**: Low

---

## 5. Phase 3: Remove TextBlock from Converters

### Phase 3.1: Refactor MarkdownConverter.convert_page_from_spans()

**File**: `src/converters/markdown.rs`

Current code (lines 184-198) converts TextSpan to TextBlock:
```rust
let mut blocks: Vec<TextBlock> = spans
    .iter()
    .map(|span| TextBlock {
        chars: vec![], // Not needed for span-based conversion
        bbox: span.bbox,
        text: span.text.clone(),
        ...
    })
    .collect();
```

This is unnecessary. Refactor to work directly with TextSpan.

**Changes**:
1. Replace `Vec<TextBlock>` with `Vec<&TextSpan>`
2. Update all block references to span references
3. Remove `merge_adjacent_char_spans()` (operates on TextBlock)
4. Use `pipeline::MarkdownOutputConverter` instead

**Effort**: Large (L)
**Risk**: High (core conversion logic)
**Acceptance Criteria**:
- All markdown output tests pass
- No regression in spurious spaces
- Bold marker logic preserved

### Phase 3.2: Refactor HtmlConverter Similarly

**File**: `src/converters/html.rs`

Same pattern as markdown - convert_page_from_spans() wraps TextSpan in TextBlock.

**Effort**: Medium (M)
**Risk**: Medium

### Phase 3.3: Update or Remove DBSCAN Clustering

**File**: `src/layout/clustering.rs`

`cluster_chars_into_words()` operates on `Vec<TextChar>`. This is only used by deprecated `convert_page()` methods.

**Options**:
1. Keep for backward compatibility (deprecated)
2. Remove entirely
3. Refactor to work with character positions from TextSpan

**Recommendation**: Mark as deprecated, remove in v2.0

**Effort**: Small (S) for deprecation
**Risk**: Low

---

## 6. Phase 4: Fix Spurious Spaces Issue

### Root Cause Analysis

Based on Phase 7 findings (from `PHASE7_SOLUTION_SUMMARY.md`):

1. **Detection regex false positives**: Pattern `/\b([a-z]+)\s+([a-z]{1,3})\s+([a-z]+)\b/` matches legitimate English phrases
2. **Double space insertion in span merging**: Unconditional space insertion at boundaries

### Phase 4.1: Fix Span Merging Double-Space Issue

**File**: `src/converters/markdown.rs` (lines 366-397)

Current logic inserts spaces based on geometric gaps without checking if space already exists at boundary.

**Fix**:
```rust
// Before inserting space, check both boundaries
let ends_with_space = group_text.ends_with(char::is_whitespace);
let starts_with_space = block.text.starts_with(char::is_whitespace);

if gap > space_threshold && !ends_with_space && !starts_with_space {
    // Also check if span has offset_semantic flag
    // (TJ processor already inserted space)
    if !block.offset_semantic {
        group_text.push(' ');
    }
}
```

**Effort**: Small (S)
**Risk**: Low
**Acceptance Criteria**:
- Double spaces reduced by >90%
- Word boundaries still detected correctly

### Phase 4.2: Propagate offset_semantic Flag Through Pipeline

**File**: `src/pipeline/ordered_span.rs`

Ensure `TextSpan.offset_semantic` is preserved in `OrderedTextSpan` and checked during conversion.

**Effort**: Small (S)
**Risk**: Low

### Phase 4.3: Add Whitespace Normalization Pass

**File**: `src/converters/whitespace.rs`

Add post-processing to normalize multiple spaces:

```rust
/// Collapse multiple consecutive spaces to single space.
pub fn normalize_spaces(text: &str) -> String {
    let re = Regex::new(r" {2,}").unwrap();
    re.replace_all(text, " ").to_string()
}
```

Already exists as `cleanup_markdown()` - verify it handles double spaces.

**Effort**: Small (S)
**Risk**: None

---

## 7. Phase 5: Remove TextChar and TextBlock Definitions

### Phase 5.1: Remove TextChar

**File**: `src/layout/text_block.rs`

1. Delete `TextChar` struct definition (lines 63-82)
2. Remove from module exports in `src/layout/mod.rs`
3. Update all files that import it

**Blocked by**: All usages must be migrated first

**Effort**: Medium (M)
**Risk**: High (breaking change)

### Phase 5.2: Remove TextBlock

**File**: `src/layout/text_block.rs`

1. Delete `TextBlock` struct definition (lines 182-287)
2. Delete `impl TextBlock` methods
3. Remove from module exports
4. Update all files that import it

**Blocked by**: All usages must be migrated first

**Effort**: Medium (M)
**Risk**: High (breaking change)

### Phase 5.3: Update Tests

**Files**:
- `tests/test_layout.rs`
- `tests/test_converters.rs`
- `tests/test_markdown_extraction_quality.rs`

Replace test helpers that create TextChar/TextBlock with TextSpan-based helpers.

**Effort**: Medium (M)
**Risk**: Low

---

## 8. Execution Order and Dependencies

```
Phase 1.1 ─── Phase 1.2 ─── Phase 1.3
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
           Phase 2.1
                  │
                  ├── Phase 2.2
                  │
                  ▼
    ┌─────────────┴─────────────┐
    │             │             │
Phase 3.1   Phase 3.2    Phase 3.3
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
    ┌─────────────┴─────────────┐
    │             │             │
Phase 4.1   Phase 4.2    Phase 4.3
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
    ┌─────────────┴─────────────┐
    │             │             │
Phase 5.1   Phase 5.2    Phase 5.3
    │             │             │
    └─────────────┴─────────────┘
```

---

## 9. Blockers and Risks

### Critical Blockers

1. **export_to_markdown.rs binary**: Uses TextChar heavily. Must be migrated or deprecated.
2. **Backward compatibility**: External users may depend on TextChar/TextBlock types.

### High Risks

1. **Markdown quality regression**: Bold markers, spacing, heading detection
2. **Structure tree integration**: Incomplete MCID coverage in real PDFs

### Mitigation Strategies

1. **Parallel implementation**: Keep old methods until new pipeline proven
2. **Feature flag**: `use_pipeline: bool` in configuration
3. **Extensive testing**: Add integration tests for known problematic PDFs

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Spurious spaces | <10 per doc | Quality test suite |
| Word fusion | <5 per doc | Quality test suite |
| Test coverage | >90% | cargo tarpaulin |
| API compatibility | 100% | Existing tests pass |
| Performance | No regression | Benchmark suite |

---

## 11. Task Breakdown by Priority

### P0 (Must Have)

- [x] Phase 1.1: Mark deprecated (S)
- [ ] Phase 2.1: Pipeline integration (L)
- [ ] Phase 4.1: Fix double-space (S)
- [ ] Phase 4.3: Whitespace normalization (S)

### P1 (Should Have)

- [ ] Phase 1.2: Remove extract() (M)
- [ ] Phase 1.3: Add pipeline to lib.rs (S)
- [ ] Phase 2.2: Structure tree wiring (S)
- [ ] Phase 3.1: Refactor markdown converter (L)
- [ ] Phase 4.2: Propagate offset_semantic (S)

### P2 (Nice to Have)

- [ ] Phase 3.2: Refactor HTML converter (M)
- [ ] Phase 3.3: Deprecate DBSCAN (S)
- [ ] Phase 5.1-5.3: Remove types (M each)

---

## 12. Technical Debt Identified

| Category | Severity | Description |
|----------|----------|-------------|
| architecture | HIGH | Dual type system (TextChar + TextSpan) violates SRP |
| architecture | MEDIUM | Converters duplicate pipeline logic |
| testing | MEDIUM | Tests use mock chars, not real PDF extraction |
| performance | LOW | TextBlock allocation overhead for span wrapping |
| documentation | LOW | Old docs reference deprecated types |

---

## 13. Appendix: File Change Summary

### Files to Modify

| File | Changes |
|------|---------|
| `src/layout/text_block.rs` | Add deprecation, eventually remove types |
| `src/layout/mod.rs` | Update exports |
| `src/extractors/text.rs` | Remove char extraction |
| `src/converters/markdown.rs` | Remove TextBlock dependency |
| `src/converters/html.rs` | Remove TextBlock dependency |
| `src/document.rs` | Add pipeline methods |
| `src/lib.rs` | Add pipeline module |
| `src/bin/export_to_markdown.rs` | Migrate to spans |

### Files to Delete (Phase 5)

| File | Reason |
|------|--------|
| N/A | Types removed from text_block.rs, no separate files |

### New Files

| File | Purpose |
|------|---------|
| N/A | Pipeline module already exists |

---

## 14. Appendix: Code Snippets for Key Changes

### Reading Order Context Building

```rust
// src/document.rs - new method
fn build_reading_order_context(&mut self, page_index: usize) -> Result<ReadingOrderContext> {
    use crate::pipeline::ReadingOrderContext;
    use crate::structure::extract_reading_order;

    let mut context = ReadingOrderContext::new()
        .with_page(page_index as u32);

    // Try to get structure tree MCIDs
    if let Ok(Some(struct_tree)) = self.structure_tree() {
        match extract_reading_order(&struct_tree, page_index as u32) {
            Ok(mcid_order) if !mcid_order.is_empty() => {
                log::info!(
                    "Using structure tree for page {} with {} MCIDs",
                    page_index, mcid_order.len()
                );
                context = context.with_mcid_order(mcid_order);
            }
            Ok(_) => {
                log::debug!("Page {} has no MCIDs in structure tree", page_index);
            }
            Err(e) => {
                log::warn!("Failed to extract MCIDs for page {}: {}", page_index, e);
            }
        }
    }

    Ok(context)
}
```

### Double-Space Fix

```rust
// src/converters/markdown.rs - modified gap check
fn should_insert_gap_space(
    prev_text: &str,
    next_span: &TextSpan,
    gap: f32,
    threshold: f32,
) -> bool {
    if gap <= threshold {
        return false;
    }

    // Check boundary conditions
    let ends_with_space = prev_text.ends_with(char::is_whitespace);
    let starts_with_space = next_span.text.starts_with(char::is_whitespace);

    // Check if TJ processor already inserted space
    let tj_space_present = next_span.offset_semantic;

    !ends_with_space && !starts_with_space && !tj_space_present
}
```

---

*Document created: 2024-12-07*
*Author: Architecture Analysis*
*Status: Ready for Implementation*
