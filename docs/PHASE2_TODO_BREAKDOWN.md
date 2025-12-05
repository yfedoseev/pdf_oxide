# Phase 2: Comprehensive Task Breakdown

This document provides granular task breakdown for Phase 2 implementation. Tasks are organized by phase with dependencies, effort estimates, and acceptance criteria.

**Effort Scale**: S (1-2h), M (2-4h), L (4-8h), XL (8-16h)

---

## Phase 2.1: Unified Space Detection Engine

### 2.1.1: Foundation Module Structure (L)

**Files to create**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Acceptance Criteria**:
- [ ] Create `space_detection.rs` with module documentation
- [ ] Define `SpaceDecision` struct with fields: insert_space, confidence, method, position
- [ ] Define `SpaceConfidence` enum: High, Medium, Low
- [ ] Define `SpaceDetectionMethod` enum: GapBased, Heuristic, Adaptive, TjOffset, Consensus
- [ ] Define `SpacePosition` struct with left_text, right_text, gap_pt, line_y
- [ ] Add to `src/layout/mod.rs` exports
- [ ] Compiles with `cargo check`

**Dependencies**: None

---

### 2.1.2: SpaceDetector Trait Definition (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Acceptance Criteria**:
- [ ] Define `SpaceDetector` trait with methods: detect(), priority(), name()
- [ ] Define `SpaceContext` struct with all required fields
- [ ] SpaceContext includes: left_text, right_text, gap_pt, font sizes, gap_stats, tj_offset
- [ ] Trait is `Send + Sync` for thread safety
- [ ] Add comprehensive documentation with examples

**Dependencies**: 2.1.1

---

### 2.1.3: GapBasedDetector Implementation (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Acceptance Criteria**:
- [ ] Create `GapBasedDetector` struct with configurable threshold_em (default 0.25)
- [ ] Implement `SpaceDetector` trait
- [ ] Logic: gap_pt > (threshold_em * font_size) => insert space
- [ ] Confidence mapping: >2x threshold = High, >1.5x = Medium, else Low
- [ ] Unit tests for various gap sizes and font sizes

**Dependencies**: 2.1.2

---

### 2.1.4: HeuristicDetector Implementation (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Acceptance Criteria**:
- [ ] Create `HeuristicDetector` struct
- [ ] Port logic from `should_insert_space_heuristic()` in text.rs
- [ ] Handle CamelCase transitions (lowercase -> uppercase)
- [ ] Handle number-letter transitions
- [ ] Handle letter-number transitions
- [ ] Confidence: always Medium for heuristic matches
- [ ] Unit tests for all heuristic patterns

**Dependencies**: 2.1.2

---

### 2.1.5: AdaptiveDetector Implementation (L)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Acceptance Criteria**:
- [ ] Create `AdaptiveDetector` struct with `AdaptiveThresholdConfig`
- [ ] Integrate with existing `gap_statistics.rs` module
- [ ] Use document-wide gap statistics for threshold calculation
- [ ] Confidence based on distance from adaptive threshold
- [ ] Fallback behavior when stats unavailable
- [ ] Unit tests with mock GapStatistics

**Dependencies**: 2.1.2, existing gap_statistics.rs

---

### 2.1.6: TjOffsetDetector Implementation (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Acceptance Criteria**:
- [ ] Create `TjOffsetDetector` struct with configurable threshold (default -120.0)
- [ ] Only applies when SpaceContext.tj_offset is Some
- [ ] Logic: tj_offset < threshold => insert space
- [ ] Confidence: High for < 2x threshold, Medium otherwise
- [ ] Unit tests for TJ offset values

**Dependencies**: 2.1.2

---

### 2.1.7: ConsensusDetector Implementation (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Acceptance Criteria**:
- [ ] Create `ConsensusDetector` that wraps multiple detectors
- [ ] Configurable min_agreement threshold (default 2)
- [ ] Aggregates decisions from child detectors
- [ ] Final confidence = highest among agreeing detectors
- [ ] Unit tests for agreement scenarios

**Dependencies**: 2.1.3, 2.1.4, 2.1.5, 2.1.6

---

### 2.1.8: SpaceDetectionEngine Implementation (L)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Acceptance Criteria**:
- [ ] Create `SpaceDetectionEngine` struct
- [ ] Constructor `new()` with default detectors (Gap, Heuristic, Adaptive, TjOffset)
- [ ] Constructor `with_config()` for custom configuration
- [ ] Method `add_detector()` for extensibility (OCP)
- [ ] Method `decide()` that runs all detectors and returns authoritative decision
- [ ] Priority-based execution order
- [ ] Comprehensive logging for debugging
- [ ] Integration tests with realistic span data

**Dependencies**: 2.1.3, 2.1.4, 2.1.5, 2.1.6, 2.1.7

---

### 2.1.9: SpaceDetectionConfig Implementation (S)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Acceptance Criteria**:
- [ ] Create `SpaceDetectionConfig` struct with all detector configs
- [ ] Preset methods: default(), aggressive(), conservative(), policy_documents()
- [ ] Builder pattern for custom configuration
- [ ] Documentation of each config option

**Dependencies**: 2.1.8

---

### 2.1.10: Unit Tests for Space Detection (L)

**Files to create**:
- Tests in `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs` (inline)

**Acceptance Criteria**:
- [ ] Tests for each detector in isolation
- [ ] Tests for engine with multiple detectors
- [ ] Tests for edge cases (empty text, zero gap, negative gap)
- [ ] Tests for config presets
- [ ] Minimum 15 test cases
- [ ] All tests pass with `cargo test`

**Dependencies**: 2.1.8, 2.1.9

---

## Phase 2.2: Font Weight Normalization

### 2.2.1: SpanType Enum Addition (S)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/text_block.rs`

**Acceptance Criteria**:
- [ ] Add `SpanType` enum with variants: Word, Space, Mixed
- [ ] Derive Debug, Clone, Copy, PartialEq, Eq
- [ ] Add documentation explaining each variant
- [ ] Export from `src/layout/mod.rs`

**Dependencies**: None

---

### 2.2.2: TextSpan Enhancement (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/text_block.rs`

**Acceptance Criteria**:
- [ ] Add `span_type: SpanType` field to `TextSpan`
- [ ] Add `effective_font_weight: FontWeight` field
- [ ] Rename existing `font_weight` to `raw_font_weight` (or keep both)
- [ ] Add method `should_render_bold() -> bool`
- [ ] Add method `normalize_font_weight(&mut self)`
- [ ] Update all TextSpan constructors to include new fields
- [ ] Default span_type to Word for backward compatibility

**Dependencies**: 2.2.1

---

### 2.2.3: Update insert_space_as_span (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs` (line ~2729)

**Acceptance Criteria**:
- [ ] Set `span_type = SpanType::Space` when creating space spans
- [ ] Set `effective_font_weight = FontWeight::Normal` (already hardcoded, verify)
- [ ] Ensure raw_font_weight still captures graphics state for debugging
- [ ] Add log statement for space span creation

**Dependencies**: 2.2.2

---

### 2.2.4: Update flush_tj_buffer (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs` (line ~2486)

**Acceptance Criteria**:
- [ ] Set `span_type = SpanType::Word` for content spans
- [ ] Set `effective_font_weight = raw_font_weight` for content
- [ ] Consistent initialization of new fields

**Dependencies**: 2.2.2

---

### 2.2.5: FontWeightNormalizer Module (M)

**Files to create**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/font_normalization.rs`

**Acceptance Criteria**:
- [ ] Create `FontWeightNormalizer` struct
- [ ] Method `normalize(spans: &mut [TextSpan])` - applies per-span normalization
- [ ] Method `propagate_bold_to_word_boundaries(spans: &mut [TextSpan])`
- [ ] Add to `src/layout/mod.rs` exports
- [ ] Unit tests for normalization scenarios

**Dependencies**: 2.2.2

---

### 2.2.6: Word Boundary Detection (L)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/font_normalization.rs`

**Acceptance Criteria**:
- [ ] Implement word boundary detection in `propagate_bold_to_word_boundaries`
- [ ] Track runs of Word spans between Space spans
- [ ] Ensure all spans in a word have consistent bold status
- [ ] Handle edge cases: single-char words, mixed formatting
- [ ] Unit tests for word boundary scenarios

**Dependencies**: 2.2.5

---

### 2.2.7: Integration with TextExtractor (L)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs`

**Acceptance Criteria**:
- [ ] Call `FontWeightNormalizer::normalize()` after span extraction
- [ ] Call before merge_adjacent_spans
- [ ] Verify spans have correct effective_font_weight
- [ ] Integration test with sample PDF

**Dependencies**: 2.2.5, 2.2.6

---

## Phase 2.3: Conservative Bold Rendering

### 2.3.1: BoldGroup Structure (M)

**Files to create**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/bold_validation.rs`

**Acceptance Criteria**:
- [ ] Create `BoldGroup<'a>` struct holding span references
- [ ] Method `has_word_content() -> bool`
- [ ] Method `has_valid_opening_boundary() -> bool`
- [ ] Method `has_valid_closing_boundary() -> bool`
- [ ] Method `cleaned_text() -> String`
- [ ] Add to `src/layout/mod.rs` exports

**Dependencies**: 2.2.2

---

### 2.3.2: BoldMarkerDecision Enum (S)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/bold_validation.rs`

**Acceptance Criteria**:
- [ ] Create `BoldMarkerDecision` enum: Insert, Skip(SkipReason)
- [ ] Create `SkipReason` enum: WhitespaceOnly, InvalidOpenBoundary, InvalidCloseBoundary, EmptyAfterCleaning
- [ ] Derive Debug, Clone

**Dependencies**: 2.3.1

---

### 2.3.3: BoldMarkerValidator Implementation (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/bold_validation.rs`

**Acceptance Criteria**:
- [ ] Create `BoldMarkerValidator` struct
- [ ] Method `can_insert_markers(group: &BoldGroup) -> BoldMarkerDecision`
- [ ] Implement Rule 1: Must have non-whitespace content
- [ ] Implement Rule 2: Valid opening boundary
- [ ] Implement Rule 3: Valid closing boundary
- [ ] Implement Rule 4: Non-empty after cleaning
- [ ] Unit tests for each rule

**Dependencies**: 2.3.1, 2.3.2

---

### 2.3.4: Integration with Markdown Renderer (L)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs` (line ~290-370)

**Acceptance Criteria**:
- [ ] Import `BoldMarkerValidator` and `BoldGroup`
- [ ] Create BoldGroup from consecutive bold spans
- [ ] Call `BoldMarkerValidator::can_insert_markers()` before insertion
- [ ] Log skip reasons for debugging
- [ ] Remove redundant checks (is_content_block, final_is_whitespace_only)
- [ ] Integration test with sample markdown output

**Dependencies**: 2.3.3

---

### 2.3.5: Simplify should_insert_bold_marker (S)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs`

**Acceptance Criteria**:
- [ ] Refactor to delegate to BoldMarkerValidator
- [ ] Remove duplicate boundary checking logic
- [ ] Maintain backward compatibility for direct calls
- [ ] Update documentation

**Dependencies**: 2.3.4

---

## Phase 2.4: Span Merging Integration

### 2.4.1: Create SpaceContext from Spans (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs`

**Acceptance Criteria**:
- [ ] Add method to create `SpaceContext` from two adjacent TextSpans
- [ ] Extract: left_text, right_text, gap_pt, font sizes
- [ ] Include gap_stats when available
- [ ] Include tj_offset when available (may need to track separately)

**Dependencies**: 2.1.2, 2.2.2

---

### 2.4.2: Replace merge_adjacent_spans Space Logic (XL)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs` (line ~1347-1511)

**Acceptance Criteria**:
- [ ] Instantiate SpaceDetectionEngine in TextExtractor
- [ ] In merge loop, create SpaceContext for each gap
- [ ] Call engine.decide() instead of inline logic
- [ ] Remove: needs_space_by_gap, needs_space_by_heuristic, gap_wants_space
- [ ] Keep: already_has_space check (redundancy prevention)
- [ ] Preserve SpanType through merging
- [ ] Update logging to use SpaceDecision fields
- [ ] Comprehensive integration tests

**Dependencies**: 2.1.8, 2.4.1

---

### 2.4.3: Remove Duplicate Heuristics (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs`

**Acceptance Criteria**:
- [ ] Remove or deprecate `should_insert_space_heuristic` function
- [ ] Keep `has_boundary_space` (still needed for double-space prevention)
- [ ] Update any remaining callers
- [ ] Verify no compilation errors

**Dependencies**: 2.4.2

---

## Phase 2.5: Cleanup and Documentation

### 2.5.1: Remove Redundant Checks in Markdown (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs`

**Acceptance Criteria**:
- [ ] Remove `is_content_block()` calls now handled by validator
- [ ] Remove `final_is_whitespace_only` check
- [ ] Simplify `should_render_bold_markers` logic
- [ ] Clean up unused imports

**Dependencies**: 2.3.4, 2.4.2

---

### 2.5.2: Update Module Documentation (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`
- `/home/yfedoseev/projects/pdf_oxide/src/layout/font_normalization.rs`
- `/home/yfedoseev/projects/pdf_oxide/src/layout/bold_validation.rs`

**Acceptance Criteria**:
- [ ] Add module-level documentation with architecture overview
- [ ] Document all public types with examples
- [ ] Add "See Also" references between related modules
- [ ] Run `cargo doc` and verify no warnings

**Dependencies**: All Phase 2.1-2.4 tasks

---

### 2.5.3: Update lib.rs Exports (S)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/lib.rs`
- `/home/yfedoseev/projects/pdf_oxide/src/layout/mod.rs`

**Acceptance Criteria**:
- [ ] Export SpaceDetectionEngine for advanced users
- [ ] Export SpaceDetector trait for custom detectors
- [ ] Export BoldMarkerValidator for custom rendering
- [ ] Verify public API is complete

**Dependencies**: 2.5.2

---

### 2.5.4: Deprecation Notices (S)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs`
- `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs`

**Acceptance Criteria**:
- [ ] Add `#[deprecated]` to functions replaced by new architecture
- [ ] Include migration guidance in deprecation message
- [ ] Verify `cargo build` shows expected deprecation warnings

**Dependencies**: 2.5.1

---

## Phase 2.6: Validation and Testing

### 2.6.1: Regression Suite Updates (L)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/tests/regression_suite.rs`
- `/home/yfedoseev/projects/pdf_oxide/tests/quality_metrics.rs`

**Acceptance Criteria**:
- [ ] Update quality thresholds for new architecture
- [ ] Add test cases for SpanType handling
- [ ] Add test cases for font weight normalization
- [ ] Verify all 5 quick PDFs pass
- [ ] Verify all 15 comprehensive PDFs pass (if applicable)

**Dependencies**: All Phase 2.1-2.5 tasks

---

### 2.6.2: Empty Bold Marker Verification (M)

**Files to create**:
- Additional tests in `/home/yfedoseev/projects/pdf_oxide/tests/`

**Acceptance Criteria**:
- [ ] Create dedicated test for empty bold markers
- [ ] Test PDFs that previously had 10, 9, 4, 2 empty markers
- [ ] Assert 0 empty bold markers in all cases
- [ ] Document which PDFs were tested

**Dependencies**: 2.6.1

---

### 2.6.3: Word Fusion Verification (M)

**Files to modify**:
- `/home/yfedoseev/projects/pdf_oxide/tests/quality_metrics.rs`

**Acceptance Criteria**:
- [ ] Verify no new word fusions introduced
- [ ] Distinguish library bugs from PDF structure issues
- [ ] Assert 0 High/Medium confidence fusions
- [ ] Document exempted PDF structure issues

**Dependencies**: 2.6.1

---

### 2.6.4: Performance Benchmarking (L)

**Files to create**:
- `/home/yfedoseev/projects/pdf_oxide/benches/space_detection.rs`

**Acceptance Criteria**:
- [ ] Create benchmark for space detection engine
- [ ] Compare against baseline (pre-Phase 2)
- [ ] Target: <5% overhead for space detection
- [ ] Document performance characteristics

**Dependencies**: All Phase 2.1-2.5 tasks

---

### 2.6.5: Final Quality Report (M)

**Files to create**:
- `/home/yfedoseev/projects/pdf_oxide/docs/PHASE2_COMPLETION_REPORT.md`

**Acceptance Criteria**:
- [ ] Document all completed tasks
- [ ] Include before/after metrics comparison
- [ ] List any remaining issues or known limitations
- [ ] Recommendations for future improvements

**Dependencies**: 2.6.1, 2.6.2, 2.6.3, 2.6.4

---

## Summary Statistics

| Phase | Tasks | Effort | Focus |
|-------|-------|--------|-------|
| 2.1 | 10 | L+9M+S = ~30h | Space Detection Engine |
| 2.2 | 7 | L+5M+S = ~20h | Font Weight Normalization |
| 2.3 | 5 | L+3M+S = ~12h | Bold Rendering Rules |
| 2.4 | 3 | XL+2M = ~16h | Span Merging Integration |
| 2.5 | 4 | 3M+S = ~10h | Cleanup and Documentation |
| 2.6 | 5 | 3L+2M = ~20h | Validation and Testing |
| **Total** | **34** | **~108h** | **3-4 weeks** |

---

## Critical Path

```
2.1.1 -> 2.1.2 -> [2.1.3, 2.1.4, 2.1.5, 2.1.6] -> 2.1.7 -> 2.1.8 -> 2.4.2
                                                              |
2.2.1 -> 2.2.2 -> [2.2.3, 2.2.4, 2.2.5] -> 2.2.6 -> 2.2.7 ---+
                        |                                     |
                        +-> 2.3.1 -> 2.3.2 -> 2.3.3 -> 2.3.4 -+
                                                              |
                                                              v
                                                     [2.5.1, 2.5.2, 2.5.3]
                                                              |
                                                              v
                                              [2.6.1, 2.6.2, 2.6.3, 2.6.4]
                                                              |
                                                              v
                                                           2.6.5
```

---

## Blockers and Unknowns

### Research Needed

1. **TJ offset tracking through merging**: Currently TJ offsets are consumed at extraction time. Need to investigate if they should be stored for later detection.

2. **Performance impact of multi-detector engine**: Need to benchmark before committing to architecture.

3. **Backward compatibility**: Verify SpanType default doesn't break existing code paths.

### External Dependencies

- None identified

### Risk Items

1. **Large refactor of merge_adjacent_spans (2.4.2)**: This is the highest-risk task due to central role in extraction pipeline. Recommend incremental implementation with feature flag.

2. **SpanType propagation through all code paths**: Need to audit all places where TextSpan is created/modified.
