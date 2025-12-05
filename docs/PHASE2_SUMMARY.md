# Phase 2: Summary and Quick Reference

## Overview

Phase 2 addresses the root cause of empty bold markers and word fusion issues through architectural redesign based on SOLID principles.

---

## Documents Created

| Document | Purpose | Location |
|----------|---------|----------|
| Implementation Plan | Architecture, interfaces, data flow | `/home/yfedoseev/projects/pdf_oxide/docs/PHASE2_IMPLEMENTATION_PLAN.md` |
| Todo Breakdown | 34 granular tasks with acceptance criteria | `/home/yfedoseev/projects/pdf_oxide/docs/PHASE2_TODO_BREAKDOWN.md` |
| Technical Debt | Debt analysis, trade-offs, risks | `/home/yfedoseev/projects/pdf_oxide/docs/PHASE2_TECHNICAL_DEBT.md` |

---

## Three Core Components

### 1. Unified Space Detection Engine

**New File**: `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`

**Key Types**:
- `SpaceDetector` trait - Strategy pattern for detection methods
- `SpaceDecision` - Authoritative space decision with confidence
- `SpaceDetectionEngine` - Orchestrates multiple detectors

**Principle**: Single Responsibility (SRP) + Open/Closed (OCP)

---

### 2. Font Weight Normalization

**New File**: `/home/yfedoseev/projects/pdf_oxide/src/layout/font_normalization.rs`

**Key Types**:
- `SpanType` enum - Distinguishes Word from Space spans
- `FontWeightNormalizer` - Ensures spaces never carry bold

**Changes to Existing**:
- `TextSpan` gains `span_type` and `effective_font_weight` fields

**Principle**: Separation of Concerns

---

### 3. Conservative Bold Rendering

**New File**: `/home/yfedoseev/projects/pdf_oxide/src/layout/bold_validation.rs`

**Key Types**:
- `BoldGroup` - Group of spans for bold rendering
- `BoldMarkerValidator` - Validates before marker insertion
- `BoldMarkerDecision` - Insert or Skip with reason

**Principle**: Dependency Inversion (DIP)

---

## Files to Modify

| File | Changes | Risk |
|------|---------|------|
| `src/extractors/text.rs` | Remove independent space logic, add SpanType, integrate engine | HIGH |
| `src/converters/markdown.rs` | Use BoldMarkerValidator, simplify grouping | MEDIUM |
| `src/layout/text_block.rs` | Add SpanType, effective_font_weight to TextSpan | LOW |
| `src/layout/mod.rs` | Export new modules | LOW |

---

## Task Summary

| Phase | Focus | Tasks | Effort |
|-------|-------|-------|--------|
| 2.1 | Space Detection Engine | 10 | ~30h |
| 2.2 | Font Weight Normalization | 7 | ~20h |
| 2.3 | Bold Rendering Rules | 5 | ~12h |
| 2.4 | Span Merging Integration | 3 | ~16h |
| 2.5 | Cleanup and Documentation | 4 | ~10h |
| 2.6 | Validation and Testing | 5 | ~20h |
| **Total** | | **34** | **~108h (3-4 weeks)** |

---

## Critical Path

```
2.1.1 (Module Structure)
    |
    v
2.1.2 (SpaceDetector Trait)
    |
    +---> 2.1.3-2.1.6 (Individual Detectors) ---> 2.1.7-2.1.8 (Engine)
    |
    v
2.2.1-2.2.2 (SpanType, TextSpan)
    |
    +---> 2.2.3-2.2.7 (Normalization)
    |
    v
2.3.1-2.3.4 (Bold Validation)
    |
    v
2.4.2 (Span Merging Integration) <-- HIGHEST RISK
    |
    v
2.5.x (Cleanup)
    |
    v
2.6.x (Validation)
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Empty bold markers | 2-10 per PDF | 0 |
| Word fusions (High/Medium) | 1-3 per PDF | 0 |
| Quality score | 5.0-8.0 | 9.0+ |
| Test pass rate | 1/5 (20%) | 5/5 (100%) |
| Performance overhead | - | <5% |

---

## Getting Started

### Step 1: Create Module Structure

```bash
# Create new files
touch src/layout/space_detection.rs
touch src/layout/font_normalization.rs
touch src/layout/bold_validation.rs

# Update mod.rs
echo 'pub mod space_detection;' >> src/layout/mod.rs
echo 'pub mod font_normalization;' >> src/layout/mod.rs
echo 'pub mod bold_validation;' >> src/layout/mod.rs
```

### Step 2: Implement Foundation (Task 2.1.1)

Start with `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`:

```rust
//! Unified space detection engine for PDF text extraction.
//!
//! This module provides a single source of truth for space insertion decisions,
//! replacing the fragmented logic spread across TJ processing, span merging,
//! and markdown rendering.

use crate::extractors::gap_statistics::GapStatistics;

/// Result of space detection analysis.
#[derive(Debug, Clone)]
pub struct SpaceDecision {
    /// Whether to insert a space between two text elements.
    pub insert_space: bool,
    /// Confidence level of the decision.
    pub confidence: SpaceConfidence,
    /// Detection method that produced this decision.
    pub method: SpaceDetectionMethod,
}

// ... continue with trait and implementations
```

### Step 3: Follow Todo Breakdown

Work through tasks in order, checking off acceptance criteria.

---

## Quick Reference: Key Code Locations

### Current Space Detection (to be replaced)

```
src/extractors/text.rs:
  - Line 2643-2677: TJ offset processing
  - Line 1400-1423: Gap-based + heuristic detection
  - Line 3048-3071: should_insert_space_heuristic()

src/converters/markdown.rs:
  - Line 330-354: Bold marker insertion
  - Line 937-940: is_content_block()
  - Line 942-969: should_insert_bold_marker()
```

### Data Structures

```
src/layout/text_block.rs:
  - TextSpan: Main text unit (add span_type, effective_font_weight)
  - FontWeight: Bold/Normal enum (line 75-98)
  - TextBlock: Grouped text for rendering (line 169-190)

src/extractors/gap_statistics.rs:
  - GapStatistics: Document gap distribution (line 51-74)
  - AdaptiveThresholdConfig: Adaptive detection config (line 103-180)
```

---

## PDF Specification References

- **ISO 32000-1:2008, Section 9.4.4, NOTE 6**: "Text strings shall be as long as possible"
- **ISO 32000-1:2008, Table 122**: FontDescriptor with font weight values
- **Interpretation**: Spaces are positioning artifacts, not content; should never carry formatting

---

## Contact

For questions about this design, refer to the implementation plan or raise issues during code review.
