# Phase 5: Adaptive Threshold Algorithm Implementation Plan

## Executive Summary

This document outlines the comprehensive implementation plan for the Adaptive Threshold Algorithm feature in pdf_oxide. The current fixed threshold approach (0.1pt conservative threshold) causes word fusion in policy documents with tight spacing (0.1-0.3pt) while the previous 0.3pt threshold caused issues in other document types. The solution is to implement statistical gap analysis that automatically determines optimal thresholds per document based on the actual gap distribution within that document.

**Goal**: Zero word fusion across all document types with minimal spurious spaces (<5 per document).

**Approach**: Analyze gap distribution in each document, compute robust statistical measures (median, percentiles), and use these to dynamically set the `conservative_threshold_pt` value.

---

## Problem Statement

### Current State

The `SpanMergingConfig` in `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs` uses fixed thresholds:

```rust
pub struct SpanMergingConfig {
    pub space_threshold_em_ratio: f32,      // 0.25 default
    pub conservative_threshold_pt: f32,     // 0.1 default (was 0.3, reverted)
    pub column_boundary_threshold_pt: f32,  // 5.0 default
    pub severe_overlap_threshold_pt: f32,   // -0.5 default
}
```

### The Problem

1. **Fix #1 (0.3pt threshold)**: Caused word fusion in policy documents where words are separated by 0.1-0.3pt gaps
2. **Current (0.1pt threshold)**: May cause spurious spaces in documents with tighter kerning
3. **Root Cause**: Different PDFs have vastly different gap distributions based on:
   - PDF generator software
   - Font characteristics
   - Document type (academic, legal, marketing)
   - Typographic style choices

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Word Fusion | 0 instances | Manual + automated check on 24-PDF corpus |
| Spurious Spaces | <5 per document | Count of inappropriate spaces |
| Backward Compatibility | 100% | All existing tests pass |
| Performance | <5% overhead | Benchmark extraction time |

---

## Architectural Design

### Module Overview

```
src/extractors/
├── mod.rs                    # Add: pub mod gap_statistics;
├── gap_statistics.rs         # NEW: Statistical gap analysis module
├── text.rs                   # MODIFY: Integrate adaptive threshold
└── ...

tests/
└── test_adaptive_threshold.rs  # NEW: Comprehensive test suite
```

### Data Flow

```
                    ┌─────────────────────────────────────┐
                    │         PDF Document                │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │    TextExtractor.extract_spans()    │
                    │    (Pass 1: Extract raw spans)      │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │ use_adaptive_threshold?     │
                    └──────────────┬──────────────┘
                           │               │
                     true  │               │ false
                           ▼               ▼
          ┌────────────────────────┐   ┌────────────────────────┐
          │  GapStatistics::       │   │  Use fixed config      │
          │  analyze_document()    │   │  (backward compat)     │
          └──────────┬─────────────┘   └────────────────────────┘
                     │
                     ▼
          ┌────────────────────────┐
          │  Calculate statistics  │
          │  - Extract all gaps    │
          │  - Compute median      │
          │  - Compute percentiles │
          └──────────┬─────────────┘
                     │
                     ▼
          ┌────────────────────────┐
          │  Determine threshold   │
          │  threshold = median *  │
          │               1.5      │
          └──────────┬─────────────┘
                     │
                     ▼
          ┌────────────────────────┐
          │  Apply to merging      │
          │  (Pass 2)              │
          └────────────────────────┘
```

---

## Detailed Module Design

### 1. `src/extractors/gap_statistics.rs`

#### Purpose
Analyze gaps between text spans in a document and determine optimal thresholds.

#### Structs

```rust
/// Statistics about gaps between text spans in a document.
///
/// This provides the data needed for adaptive threshold calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct GapStatistics {
    /// All gap values collected (in points)
    pub gaps: Vec<f32>,
    /// Number of gaps analyzed
    pub count: usize,
    /// Minimum gap value
    pub min: f32,
    /// Maximum gap value
    pub max: f32,
    /// Mean (average) gap
    pub mean: f32,
    /// Median gap (50th percentile) - robust to outliers
    pub median: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// 25th percentile (first quartile)
    pub p25: f32,
    /// 75th percentile (third quartile)
    pub p75: f32,
    /// 10th percentile (typical intra-word gap upper bound)
    pub p10: f32,
    /// 90th percentile (typical word boundary gap)
    pub p90: f32,
}

/// Configuration for adaptive threshold calculation.
///
/// Factory methods provide sensible defaults for different document types.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveThresholdConfig {
    /// Multiplier applied to median gap to get threshold.
    /// Higher values = more conservative (fewer spaces inserted).
    /// Default: 1.5 (50% above typical gap)
    pub median_multiplier: f32,

    /// Minimum allowed threshold (in points).
    /// Prevents threshold from being too small in documents with very tight spacing.
    /// Default: 0.05
    pub min_threshold_pt: f32,

    /// Maximum allowed threshold (in points).
    /// Prevents threshold from being too large in documents with wide spacing.
    /// Default: 1.0
    pub max_threshold_pt: f32,

    /// Whether to use interquartile range (IQR) for outlier-robust calculation.
    /// Default: true
    pub use_iqr: bool,

    /// Minimum number of gaps required for statistical analysis.
    /// Below this, fall back to fixed threshold.
    /// Default: 10
    pub min_samples: usize,
}

/// Result of adaptive threshold analysis.
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdResult {
    /// The computed threshold value (in points)
    pub threshold_pt: f32,
    /// Statistics used for calculation
    pub statistics: GapStatistics,
    /// Whether adaptive analysis was performed (vs. fallback)
    pub was_adaptive: bool,
    /// Reason for fallback if !was_adaptive
    pub fallback_reason: Option<String>,
}
```

#### Functions

```rust
/// Analyze all gaps between consecutive spans on the same line.
///
/// # Arguments
/// * `spans` - Text spans extracted from the document (must be sorted by reading order)
///
/// # Returns
/// Vector of gap values in points. Only includes gaps from spans on the same line.
///
/// # Algorithm
/// 1. Group spans by Y-coordinate (line detection with 1pt tolerance)
/// 2. For each line, sort spans by X-coordinate
/// 3. Compute gap = next_span.x - (current_span.x + current_span.width)
/// 4. Include only positive gaps (overlaps are handled separately)
pub fn extract_gaps(spans: &[TextSpan]) -> Vec<f32>;

/// Calculate comprehensive statistics from a collection of gap values.
///
/// # Arguments
/// * `gaps` - Vector of gap values (will be sorted internally)
///
/// # Returns
/// GapStatistics with all computed metrics, or None if insufficient data.
pub fn calculate_statistics(gaps: Vec<f32>) -> Option<GapStatistics>;

/// Determine the adaptive threshold based on gap statistics.
///
/// # Arguments
/// * `stats` - Gap statistics from calculate_statistics()
/// * `config` - Configuration controlling the threshold calculation
///
/// # Returns
/// The computed threshold in points.
///
/// # Algorithm (default)
/// 1. Use median as base (robust to outliers from column boundaries)
/// 2. Apply multiplier: threshold = median * median_multiplier
/// 3. Clamp to [min_threshold_pt, max_threshold_pt]
///
/// # Alternative (IQR mode)
/// 1. Calculate IQR = p75 - p25
/// 2. Use p25 + 0.5 * IQR as threshold (Tukey's inner fence concept)
/// 3. Clamp to bounds
pub fn determine_adaptive_threshold(
    stats: &GapStatistics,
    config: &AdaptiveThresholdConfig,
) -> f32;

/// Full analysis pipeline: extract gaps, compute stats, determine threshold.
///
/// This is the main entry point for adaptive threshold analysis.
///
/// # Arguments
/// * `spans` - Text spans from the document
/// * `config` - Optional configuration (uses default if None)
///
/// # Returns
/// AdaptiveThresholdResult with threshold and analysis details.
pub fn analyze_document_gaps(
    spans: &[TextSpan],
    config: Option<AdaptiveThresholdConfig>,
) -> AdaptiveThresholdResult;
```

#### Factory Methods for `AdaptiveThresholdConfig`

```rust
impl AdaptiveThresholdConfig {
    /// Default configuration - works well for most documents.
    pub fn default() -> Self {
        Self {
            median_multiplier: 1.5,
            min_threshold_pt: 0.05,
            max_threshold_pt: 1.0,
            use_iqr: true,
            min_samples: 10,
        }
    }

    /// Aggressive configuration for dense layouts (author lists, tables).
    /// Lower thresholds = more spaces inserted.
    pub fn aggressive() -> Self {
        Self {
            median_multiplier: 1.2,
            min_threshold_pt: 0.02,
            max_threshold_pt: 0.5,
            use_iqr: true,
            min_samples: 5,
        }
    }

    /// Conservative configuration for formal documents with clear spacing.
    /// Higher thresholds = fewer spaces inserted.
    pub fn conservative() -> Self {
        Self {
            median_multiplier: 2.0,
            min_threshold_pt: 0.1,
            max_threshold_pt: 1.5,
            use_iqr: true,
            min_samples: 20,
        }
    }

    /// Policy document configuration - tuned for tight 0.1-0.3pt spacing.
    pub fn policy_documents() -> Self {
        Self {
            median_multiplier: 1.3,
            min_threshold_pt: 0.08,
            max_threshold_pt: 0.35,
            use_iqr: true,
            min_samples: 10,
        }
    }

    /// Academic document configuration - wider spacing typical.
    pub fn academic() -> Self {
        Self {
            median_multiplier: 1.5,
            min_threshold_pt: 0.2,
            max_threshold_pt: 0.6,
            use_iqr: true,
            min_samples: 15,
        }
    }
}
```

### 2. Integration into `src/extractors/text.rs`

#### Changes to `SpanMergingConfig`

```rust
/// Configuration for span merging behavior.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanMergingConfig {
    // ... existing fields ...

    /// Enable adaptive threshold calculation.
    ///
    /// When true, the `conservative_threshold_pt` is automatically calculated
    /// based on the gap distribution within the document. This overrides
    /// the fixed `conservative_threshold_pt` value.
    ///
    /// **Default**: false (backward compatible)
    pub use_adaptive_threshold: bool,

    /// Configuration for adaptive threshold calculation.
    ///
    /// Only used when `use_adaptive_threshold` is true.
    /// If None, uses AdaptiveThresholdConfig::default().
    pub adaptive_config: Option<AdaptiveThresholdConfig>,
}
```

#### New Factory Method

```rust
impl SpanMergingConfig {
    /// Create a configuration with adaptive threshold enabled.
    ///
    /// This automatically analyzes the document's gap distribution
    /// to determine optimal spacing thresholds.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::SpanMergingConfig;
    ///
    /// let config = SpanMergingConfig::adaptive();
    /// ```
    pub fn adaptive() -> Self {
        Self {
            use_adaptive_threshold: true,
            adaptive_config: Some(AdaptiveThresholdConfig::default()),
            ..Self::default()
        }
    }

    /// Create adaptive configuration with custom settings.
    pub fn adaptive_with_config(config: AdaptiveThresholdConfig) -> Self {
        Self {
            use_adaptive_threshold: true,
            adaptive_config: Some(config),
            ..Self::default()
        }
    }
}
```

#### Changes to `TextExtractor::extract_spans()`

```rust
pub fn extract_spans(&mut self, stream_data: &[u8]) -> Result<Vec<TextSpan>> {
    // ... existing parsing logic ...

    // Sort spans by reading order
    self.sort_spans_by_reading_order();

    // Deduplicate overlapping spans
    self.deduplicate_overlapping_spans();

    // NEW: Adaptive threshold calculation (if enabled)
    if self.merging_config.use_adaptive_threshold {
        self.apply_adaptive_threshold();
    }

    // Merge adjacent spans on the same line
    self.merge_adjacent_spans();

    Ok(self.spans.clone())
}

/// Apply adaptive threshold based on document gap analysis.
fn apply_adaptive_threshold(&mut self) {
    use crate::extractors::gap_statistics::{analyze_document_gaps, AdaptiveThresholdConfig};

    let config = self.merging_config.adaptive_config.clone()
        .unwrap_or_else(AdaptiveThresholdConfig::default);

    let result = analyze_document_gaps(&self.spans, Some(config));

    if result.was_adaptive {
        log::info!(
            "Adaptive threshold: {:.3}pt (median: {:.3}pt, samples: {})",
            result.threshold_pt,
            result.statistics.median,
            result.statistics.count
        );

        // Apply the computed threshold
        self.merging_config.conservative_threshold_pt = result.threshold_pt;
    } else {
        log::debug!(
            "Adaptive threshold fallback: {} (using fixed: {:.3}pt)",
            result.fallback_reason.as_deref().unwrap_or("unknown"),
            self.merging_config.conservative_threshold_pt
        );
    }
}
```

### 3. Updates to `src/extractors/mod.rs`

```rust
pub mod gap_statistics;

// Add to re-exports
pub use gap_statistics::{
    AdaptiveThresholdConfig,
    AdaptiveThresholdResult,
    GapStatistics,
    analyze_document_gaps,
};
```

---

## Test Strategy

### Test File: `tests/test_adaptive_threshold.rs`

#### Unit Tests for `gap_statistics.rs`

```rust
mod gap_extraction {
    #[test]
    fn test_extract_gaps_single_line() {
        // Spans: [0-10], [12-22], [25-35] on same line
        // Expected gaps: [2.0, 3.0]
    }

    #[test]
    fn test_extract_gaps_multiple_lines() {
        // Spans across different Y coordinates
        // Should only include intra-line gaps
    }

    #[test]
    fn test_extract_gaps_excludes_overlaps() {
        // Overlapping spans should not contribute negative gaps
    }

    #[test]
    fn test_extract_gaps_empty_input() {
        // Empty spans -> empty gaps
    }
}

mod statistics_calculation {
    #[test]
    fn test_calculate_statistics_basic() {
        // Known values: [1.0, 2.0, 3.0, 4.0, 5.0]
        // median = 3.0, mean = 3.0, p25 = 2.0, p75 = 4.0
    }

    #[test]
    fn test_calculate_statistics_with_outliers() {
        // [0.1, 0.2, 0.3, 0.4, 50.0]
        // Median should be 0.3 (robust to 50.0 outlier)
    }

    #[test]
    fn test_calculate_statistics_insufficient_data() {
        // Less than min_samples should return None
    }
}

mod threshold_determination {
    #[test]
    fn test_determine_threshold_median_multiplier() {
        // stats.median = 0.2, multiplier = 1.5
        // Expected: 0.3
    }

    #[test]
    fn test_determine_threshold_clamped_min() {
        // Very small median, should clamp to min_threshold_pt
    }

    #[test]
    fn test_determine_threshold_clamped_max() {
        // Very large median, should clamp to max_threshold_pt
    }

    #[test]
    fn test_determine_threshold_iqr_mode() {
        // Test IQR-based calculation
    }
}
```

#### Integration Tests

```rust
mod document_type_tests {
    #[test]
    fn test_policy_document_adaptive_threshold() {
        // Mock policy document with 0.1-0.3pt gaps
        // Threshold should be ~0.15-0.25pt
        // Should produce 0 word fusions
    }

    #[test]
    fn test_academic_document_adaptive_threshold() {
        // Mock academic document with 0.3-0.5pt gaps
        // Threshold should be ~0.35-0.45pt
        // Should produce 0 word fusions
    }

    #[test]
    fn test_mixed_layout_document() {
        // Document with tables + text
        // Should handle bimodal gap distribution
    }
}

mod backward_compatibility {
    #[test]
    fn test_default_config_unchanged() {
        // SpanMergingConfig::default() should not use adaptive
        let config = SpanMergingConfig::default();
        assert!(!config.use_adaptive_threshold);
    }

    #[test]
    fn test_existing_factory_methods_work() {
        // aggressive(), conservative(), custom() still work
    }
}

mod edge_cases {
    #[test]
    fn test_single_span_document() {
        // Cannot calculate gaps, should fall back
    }

    #[test]
    fn test_all_overlapping_spans() {
        // No positive gaps, should fall back
    }

    #[test]
    fn test_uniform_gaps() {
        // All gaps identical, median should work
    }
}
```

#### Real PDF Tests (if corpus available)

```rust
#[cfg(feature = "corpus_tests")]
mod corpus_tests {
    #[test]
    fn test_24_pdf_corpus_zero_word_fusion() {
        // Load each PDF, extract with adaptive threshold
        // Verify no word fusions
    }

    #[test]
    fn test_24_pdf_corpus_spurious_spaces() {
        // Count spurious spaces per doc
        // Assert < 5 per document
    }
}
```

---

## Implementation Phases

### Phase 5.1: Core Statistical Module (Agent 1)

**Effort**: Medium (M)

**Tasks**:
1. Create `/home/yfedoseev/projects/pdf_oxide/src/extractors/gap_statistics.rs`
2. Implement `GapStatistics` struct
3. Implement `AdaptiveThresholdConfig` with factory methods
4. Implement `extract_gaps()` function
5. Implement `calculate_statistics()` function
6. Implement `determine_adaptive_threshold()` function
7. Implement `analyze_document_gaps()` pipeline function
8. Add comprehensive documentation
9. Add module unit tests

**Acceptance Criteria**:
- All statistical functions produce correct results
- Factory methods return expected configurations
- Unit tests achieve 100% coverage of public API
- `cargo clippy --all-features` passes with zero warnings

### Phase 5.2: Integration with SpanMergingConfig (Agent 2)

**Effort**: Small (S)

**Tasks**:
1. Add `use_adaptive_threshold` field to `SpanMergingConfig`
2. Add `adaptive_config` field to `SpanMergingConfig`
3. Add `SpanMergingConfig::adaptive()` factory method
4. Add `SpanMergingConfig::adaptive_with_config()` factory method
5. Update `SpanMergingConfig::default()` to set `use_adaptive_threshold: false`
6. Modify `TextExtractor::extract_spans()` to call adaptive analysis
7. Add `TextExtractor::apply_adaptive_threshold()` private method
8. Update `/home/yfedoseev/projects/pdf_oxide/src/extractors/mod.rs` with new exports

**Acceptance Criteria**:
- Backward compatibility: `SpanMergingConfig::default()` works as before
- `SpanMergingConfig::adaptive()` enables adaptive mode
- Integration correctly applies computed threshold
- All existing tests still pass

### Phase 5.3: Test Suite (Agent 3)

**Effort**: Medium (M)

**Tasks**:
1. Create `/home/yfedoseev/projects/pdf_oxide/tests/test_adaptive_threshold.rs`
2. Implement gap extraction unit tests
3. Implement statistics calculation unit tests
4. Implement threshold determination unit tests
5. Implement integration tests with mock spans
6. Implement backward compatibility tests
7. Implement edge case tests
8. Add documentation and test rationale comments

**Acceptance Criteria**:
- Test file compiles and all tests pass
- Coverage of happy path, edge cases, and error conditions
- Clear documentation of what each test verifies

### Phase 5.4: Final Integration and Validation

**Effort**: Small (S)

**Tasks**:
1. Run full test suite (`cargo test --all-features`)
2. Run clippy (`cargo clippy --all-features`)
3. Run on 24-PDF corpus (if available)
4. Measure performance overhead
5. Update documentation as needed
6. Create PR with detailed description

**Acceptance Criteria**:
- All tests pass
- Zero clippy warnings
- Performance overhead < 5%
- 24-PDF corpus: 0 word fusions, <5 spurious spaces per doc

---

## SOLID Compliance Checklist

| Principle | Status | Notes |
|-----------|--------|-------|
| **Single Responsibility** | OK | `gap_statistics.rs` handles only statistical analysis; `text.rs` handles extraction |
| **Open/Closed** | OK | New functionality via `AdaptiveThresholdConfig` without modifying existing behavior |
| **Liskov Substitution** | N/A | No inheritance hierarchy |
| **Interface Segregation** | OK | `SpanMergingConfig` additions are optional; existing code unaffected |
| **Dependency Inversion** | OK | `TextExtractor` depends on abstraction (`SpanMergingConfig`), not concrete implementation |

---

## Technical Debt Log

| ID | Category | Severity | Description | Resolution |
|----|----------|----------|-------------|------------|
| TD-001 | architecture | LOW | `merge_adjacent_spans` method is complex (~150 lines) | Future refactor into smaller functions |
| TD-002 | testing | MEDIUM | No real PDF corpus tests in CI | Add corpus test suite when PDFs available |
| TD-003 | performance | LOW | Gap extraction iterates spans twice (group by Y, then analyze) | Could optimize to single pass if needed |

---

## Performance Considerations

### Overhead Analysis

| Operation | Time Complexity | Expected Impact |
|-----------|-----------------|-----------------|
| `extract_gaps()` | O(n log n) | Sorting by Y, then X per line |
| `calculate_statistics()` | O(n log n) | Sorting for percentiles |
| Total per document | O(n log n) | Where n = number of spans |

**Expected overhead**: <5% of total extraction time (gap analysis is simple arithmetic on already-extracted data).

### Optimization Opportunities (Future)

1. **Incremental statistics**: Update running statistics during initial extraction
2. **Sampling**: For very large documents, sample gaps instead of full analysis
3. **Caching**: Cache threshold for documents with stable characteristics

---

## Appendix: Algorithm Details

### Median Calculation

```rust
fn median(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n/2 - 1] + sorted[n/2]) / 2.0
    } else {
        sorted[n/2]
    }
}
```

### Percentile Calculation (Linear Interpolation)

```rust
fn percentile(sorted: &[f32], p: f32) -> f32 {
    let n = sorted.len();
    let rank = p * (n - 1) as f32;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f32;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}
```

### IQR-Based Threshold

```rust
fn iqr_threshold(stats: &GapStatistics, config: &AdaptiveThresholdConfig) -> f32 {
    let iqr = stats.p75 - stats.p25;
    let threshold = stats.p25 + 0.5 * iqr;
    threshold.clamp(config.min_threshold_pt, config.max_threshold_pt)
}
```

---

## Parallel Implementation Strategy

All three agents can work in parallel:

```
Agent 1: gap_statistics.rs         Agent 2: text.rs integration    Agent 3: tests
────────────────────────          ─────────────────────────       ────────────────
Day 1:                            Day 1:                          Day 1:
- Struct definitions              - Add fields to config          - Test file setup
- Factory methods                 - Factory method stubs          - Mock helpers

Day 2:                            Day 2:                          Day 2:
- extract_gaps()                  - Integration into extractor    - Unit tests for stats
- calculate_statistics()          - apply_adaptive_threshold()    - Unit tests for gaps

Day 3:                            Day 3:                          Day 3:
- determine_threshold()           - mod.rs exports                - Integration tests
- analyze_document_gaps()         - Documentation                 - Edge case tests

                         ┌──────────────────────────┐
                         │   INTEGRATION DAY        │
                         │   - Merge all branches   │
                         │   - Run full test suite  │
                         │   - Corpus validation    │
                         └──────────────────────────┘
```

---

## Sign-off

| Role | Approval | Date |
|------|----------|------|
| Architect | Pending | |
| Lead Developer | Pending | |
| QA Lead | Pending | |

---

*Document Version: 1.0*
*Last Updated: Phase 5 Planning*
*Author: Claude (Task Planner & Architect)*
