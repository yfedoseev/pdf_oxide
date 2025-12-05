# Phase 2: Technical Debt Documentation

This document catalogs technical debt identified during Phase 2 analysis, debt resolved by Phase 2, and new debt potentially introduced.

---

## Debt Resolved by Phase 2

### [DEBT:architecture:HIGH] Three Independent Space Detection Layers

**Description**: Space decisions are made independently at three layers:
1. TJ Processing (offset threshold: -120.0)
2. Span Merging (gap threshold: 0.25em, conservative: 0.1pt)
3. Heuristic checks (CamelCase, number-letter)

**Impact**: Layers contradict each other, causing both word fusions (missed spaces) and spurious spaces (inserted incorrectly).

**Resolution**: Unified `SpaceDetectionEngine` consolidates all detection strategies into single authoritative decision point.

**Files Affected**:
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs` (lines 1400-1423, 2643-2677)
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs` (new)

**Verification**: Test passes() for all 5 regression PDFs.

---

### [DEBT:architecture:HIGH] Space Spans Inherit Bold from Graphics State

**Description**: `insert_space_as_span()` creates space spans that could inherit bold formatting from the current graphics state. While line 2753 hardcodes `FontWeight::Normal`, the architectural issue is that spaces are created as separate spans that can be grouped with bold content.

**Impact**: Empty bold markers (`** **`) appear when bold content spans are grouped with adjacent space spans.

**Resolution**:
1. `SpanType` enum distinguishes Word from Space spans
2. `FontWeightNormalizer` ensures Space spans have Normal weight
3. `BoldMarkerValidator` validates before marker insertion

**Files Affected**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/text_block.rs` (SpanType, effective_font_weight)
- `/home/yfedoseev/projects/pdf_oxide/src/layout/font_normalization.rs` (new)
- `/home/yfedoseev/projects/pdf_oxide/src/layout/bold_validation.rs` (new)

**Verification**: 0 empty bold markers across all test PDFs.

---

### [DEBT:performance:MEDIUM] Gap Analysis Repeated at Multiple Layers

**Description**: Gap calculations happen multiple times:
1. TJ processing calculates text position offsets
2. Span merging calculates gap between spans
3. Adaptive threshold analyzes all gaps for statistics

**Impact**: Redundant calculations; inconsistent gap values between layers.

**Resolution**: Single-pass detection in `SpaceDetectionEngine.decide()` uses pre-computed gaps.

**Files Affected**:
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs`
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Verification**: Benchmark shows <5% overhead.

---

### [DEBT:maintenance:MEDIUM] Duplicate Heuristics

**Description**: Character transition heuristics exist in two places:
1. `should_insert_space_heuristic()` in text.rs (lines 3048-3071)
2. `should_insert_bold_marker()` in markdown.rs (lines 942-969)

**Impact**: Maintenance burden; risk of divergence; unclear which takes precedence.

**Resolution**:
- `HeuristicDetector` in space_detection.rs centralizes space heuristics
- `BoldMarkerValidator` uses `BoldGroup.has_valid_opening_boundary()` for bold

**Files Affected**:
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs` (deprecated)
- `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs` (refactored)

**Verification**: Compile with `cargo build` shows deprecation warnings.

---

## Existing Debt (Not Addressed by Phase 2)

### [DEBT:architecture:MEDIUM] Span vs Character Extraction Duality

**Description**: The codebase supports both character-level (`TextChar`) and span-level (`TextSpan`) extraction. This creates two parallel code paths.

**Impact**: Maintenance burden; potential for inconsistency between paths.

**Recommendation**: Future phase should deprecate character-level extraction or clearly document when each is appropriate.

**Files Affected**:
- `/home/yfedoseev/projects/pdf_oxide/src/layout/text_block.rs`
- `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs` (both convert_page and convert_page_from_spans)

---

### [DEBT:testing:MEDIUM] Insufficient Edge Case Coverage

**Description**: Test PDFs focus on academic/policy documents. Edge cases like:
- RTL languages
- CJK text without word boundaries
- Mixed writing direction
- Extremely dense layouts

**Impact**: Unknown behavior for non-Western documents.

**Recommendation**: Add test fixtures for diverse document types.

---

### [DEBT:documentation:LOW] Inconsistent PDF Spec References

**Description**: Some functions reference ISO 32000-1:2008 sections; others don't. No central glossary of spec compliance.

**Impact**: Difficult to verify spec compliance; inconsistent terminology.

**Recommendation**: Create SPEC_COMPLIANCE.md documenting which sections are implemented.

---

## New Debt Introduced by Phase 2

### [DEBT:complexity:LOW] Additional Abstraction Layer

**Description**: `SpaceDetector` trait and `SpaceDetectionEngine` add abstraction that may be over-engineered for current needs.

**Impact**: Learning curve for new contributors; potential performance overhead.

**Mitigation**:
- Well-documented trait hierarchy
- Clear examples in module docs
- Benchmark to validate <5% overhead

**Future Resolution**: If abstraction proves unnecessary, can simplify to single function.

---

### [DEBT:performance:LOW] Extra Normalization Pass

**Description**: `FontWeightNormalizer::normalize()` adds a pass over all spans.

**Impact**: O(n) additional work per page.

**Mitigation**: Can be combined with span creation in future optimization.

**Future Resolution**: Inline normalization into flush_tj_buffer.

---

### [DEBT:backward-compat:LOW] SpanType Field Required

**Description**: All TextSpan constructors now require `span_type` field.

**Impact**: External code creating TextSpan must be updated.

**Mitigation**: Default `span_type = SpanType::Word` in serde deserialization.

---

## Trade-offs Analysis

### Correctness vs Performance

**Decision**: Prioritize correctness (0 empty bold markers) over performance.

**Trade-off**: Additional abstraction layers and passes add ~2-5% overhead.

**Justification**: Empty bold markers are critical bugs that affect user-visible output. Performance is secondary for text extraction (typically not bottleneck).

---

### Flexibility vs Simplicity

**Decision**: Trait-based SpaceDetector for extensibility (OCP).

**Trade-off**: More complex than single function.

**Justification**: Different document types (academic, legal, CJK) may need different detection strategies. Trait allows adding strategies without modifying core engine.

---

### Backward Compatibility vs Clean API

**Decision**: Maintain deprecated aliases; add SpanType with default.

**Trade-off**: Temporary API complexity during migration.

**Justification**: Avoid breaking external consumers; deprecation warnings guide migration.

---

## Risk Assessment

### High Risk

**Task 2.4.2: Replace merge_adjacent_spans Space Logic**

- Central to extraction pipeline
- Complex existing logic
- Many edge cases

**Mitigation**:
1. Feature flag for gradual rollout
2. Extensive logging during transition
3. Side-by-side comparison with old behavior

---

### Medium Risk

**SpanType Propagation**

- Must update all TextSpan constructors
- Easy to miss a code path

**Mitigation**:
1. Compiler errors for missing field (no default in struct)
2. grep for `TextSpan {` to find all constructors
3. Integration tests catch missed paths

---

### Low Risk

**Performance Regression**

- Additional passes and abstractions

**Mitigation**:
1. Benchmark before/after
2. Profile hot paths
3. Optimize if needed (inline normalization)

---

## Metrics to Track

| Metric | Current | Target | Measurement Method |
|--------|---------|--------|-------------------|
| Empty bold markers (per PDF) | 2-10 | 0 | `detect_empty_bold_markers()` |
| Word fusions (High/Medium) | 1-3 | 0 | `detect_word_fusions()` |
| Quality score | 5.0-8.0 | 9.0+ | `QualityMetrics.quality_score` |
| Extraction time (per page) | baseline | <+5% | Benchmark suite |
| Code coverage | TBD | >80% | cargo tarpaulin |

---

## Post-Phase 2 Recommendations

1. **Performance Optimization**: If overhead exceeds 5%, inline normalization into span creation.

2. **CJK Support**: Add SpaceDetector for CJK languages (no word boundaries; different rules).

3. **Machine Learning Detector**: Train classifier on labeled word boundary data for improved accuracy.

4. **Structured Logging**: Add structured logging for space decisions to enable analysis.

5. **Fuzzing**: Add property-based tests with proptest to find edge cases.
