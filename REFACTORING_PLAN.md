# PDF Spec Compliance Cleanup Roadmap

**Goal**: Remove all non-PDF-spec-compliant code from core extraction. Reorganize into:
- **Core**: Spec-compliant extraction only
- **Enhancements**: Optional, user-controlled features
- **Clear boundary**: No mixed concerns

**Current State**: ~2,000+ lines of non-spec code across 12 areas (Phase 10 status: 4.4/10)
**Target State**: Spec-strict core + optional enhancement layers (8.5+/10)

---

## PHASE 1: Removal (Lowest Priority - Just Delete)

These features have NO DEPENDENCY from core extraction. Safe to remove entirely.

### 1.1 Table Detection Module
**Location**: `src/layout/table_detector.rs` (300+ lines)
**Status**: Can be removed immediately
**Action**:
- [ ] Delete `src/layout/table_detector.rs`
- [ ] Remove from `src/layout/mod.rs` exports
- [ ] Remove from `src/extractors/mod.rs` if exported
- [ ] Update any tests that reference table detection
- [ ] Remove `TableDetector`, `DetectedTable`, `TableDetectorConfig` from public API

**Why**: Tables are semantic concepts NOT in PDF spec. Users should use structure tree (Section 14.7) if they want real table information.

**Impact**: ✅ No code breakage (optional feature)
**Effort**: 30 minutes

---

### 1.2 Heading Detection Heuristics
**Location**: `src/layout/heading_detector.rs` (300+ lines)
**Status**: Can be removed if not core to extraction
**Action**:
- [ ] Check if heading_detector is used in critical path
- [ ] If yes: Keep but add `spec_compliant: false` flag
- [ ] If no: Delete entire module
- [ ] Remove hardcoded font size thresholds (22pt, 18pt, etc.)

**Why**: Font-based heading detection is linguistic interpretation, not PDF spec feature. Use structure tree for real heading info.

**Impact**: Depends on usage - potentially breaks heading detection
**Effort**: 1 hour if deletable, 2 hours if keeping with annotations

---

### 1.3 ML Heading Classifier
**Location**: `src/ml/heading_classifier.rs` (200+ lines)
**Status**: Remove or move to optional module
**Action**:
- [ ] Delete `src/ml/heading_classifier.rs`
- [ ] Remove from `src/ml/mod.rs`
- [ ] If ML module becomes empty, consider removing entire `src/ml/`
- [ ] Remove all DistilBERT references from docs

**Why**: ML-based semantic analysis is antithetical to spec compliance. It's a proprietary classification layer.

**Impact**: ✅ No code breakage
**Effort**: 30 minutes

---

## PHASE 2: Migration to Optional (Medium Priority)

These are NEEDED for quality but NOT in PDF spec. Move to optional post-processing layer.

### 2.1 CamelCase Word Splitting
**Location**: `src/extractors/text.rs:1467-1475, 2057-2141, 3671-3989`
**Current State**: Already disabled but code still present
**Action**:
- [ ] Create new module: `src/post_processors/word_splitter.rs`
- [ ] Move `split_fused_words()`, `split_on_camelcase()` to new module
- [ ] Remove calls from main extraction pipeline
- [ ] Add post-processing layer to text extraction with opt-in flag
- [ ] Document: "Optional feature - not PDF spec-based"
- [ ] Add unit tests: verify "theGeneral" → "the General" works
- [ ] Remove dead code from text.rs: lines 3671-3989

**Configuration**:
```rust
pub struct TextExtractionConfig {
    // ... existing spec-based config ...

    // Optional enhancements (NOT spec-based)
    pub enable_word_splitting: bool,  // Default: false
}
```

**Impact**: ✅ +3 word fusions fixed when enabled
**Effort**: 2-3 hours

---

### 2.2 Document Type Detection & Profiles
**Location**: `src/extractors/gap_statistics.rs:154-248` (configuration), various `.policy_documents()`, `.academic()` methods
**Current State**: Active, controlling 1,623 spurious spaces
**Challenge**: Removing this LOWERS quality. Need to decide:
  - **Option A**: Delete (pure spec-only, quality drops to 3.5/10)
  - **Option B**: Keep but annotate as "empirical heuristic" (current approach)
  - **Option C**: Move to optional module with better documentation

**Recommendation**: **Option B (Keep with Annotations)** - for now
- [ ] Add comments to all doc-type profiles explaining they're non-spec
- [ ] Create config flag: `use_adaptive_thresholds: bool` (default: true)
- [ ] Document why: "Empirical tuning for real-world PDFs"
- [ ] Create variant: `spec_strict_config()` that disables all adaptive features
- [ ] Later: Can move to optional module after implementing better spec-based solution

**Example Documentation**:
```rust
/// **NON-SPEC HEURISTIC**: Document-type-specific thresholds
/// These multipliers (1.3x for policy, 1.6x for academic) are empirically chosen
/// and NOT derived from ISO 32000-1:2008. They improve practical quality but
/// reduce spec compliance. Disable via: config.use_adaptive_thresholds = false
pub fn policy_documents() -> Self {
    Self {
        median_multiplier: 1.3,  // Tight spacing in policy docs
        // ...
    }
}
```

**Impact**: Maintains current quality until we find better approach
**Effort**: 1-2 hours (annotation only)

---

### 2.3 Column Detection & Layout Analysis
**Location**: `src/layout/document_analyzer.rs:118-408` (bin sizes, gap ratios, Gaussian sigma)
**Current State**: Active, used for adaptive layout
**Decision**: KEEP but separate into "layout enhancement" module
**Action**:
- [ ] Move to new module: `src/enhancements/layout_analysis.rs`
- [ ] Mark all magic numbers with sources (ICDAR paper reference)
- [ ] Add config flag: `enable_layout_analysis: bool` (default: true)
- [ ] Document: "Uses ICDAR 2005 layout algorithm, not PDF spec-based"
- [ ] Keep in extractors but with clear separation

**Configuration**:
```rust
pub struct LayoutAnalysisConfig {
    pub enabled: bool,
    // Bin width for projection profile (ICDAR algorithm)
    pub histogram_bin_width_pt: f32,  // default: 10.0
    // ... other ICDAR parameters ...
}
```

**Impact**: Maintains layout analysis, improves documentation
**Effort**: 2-3 hours

---

## PHASE 3: Annotation (High Priority - Quick Wins)

Add clear documentation to all non-spec code that stays.

### 3.1 Tag All Non-Spec Code
**Action**:
- [ ] Find all non-spec implementations (use analysis output)
- [ ] Add comment block:
```rust
/// **NON-SPEC HEURISTIC**
///
/// This feature is NOT defined in ISO 32000-1:2008.
///
/// Reason: [why we do this despite not being in spec]
/// Source: [paper/empirical/pdf-specific]
/// Status: [enabled by default | optional | deprecated]
///
/// To disable: [config flag or how]
/// Impact on quality: [what happens if disabled]
```

**Locations to annotate**:
- [ ] `gap_statistics.rs`: All multiplier-based thresholds
- [ ] `geometric_spacing.rs`: Document the 0.25em ratio choice
- [ ] `document_analyzer.rs`: All ICDAR algorithm parameters
- [ ] `column_detector.rs`: XY-Cut algorithm parameters
- [ ] `bold_validation.rs`: Unicode whitespace handling

**Effort**: 3-4 hours

---

### 3.2 Create Spec Compliance Reference
**New file**: `docs/SPEC_COMPLIANCE_GUIDE.md`
**Content**:
- List all PDF spec sections used (9.3, 9.4.3, 9.4.4, etc.)
- List all non-spec features and justifications
- Configuration guide: How to enable/disable features
- Quality vs. Compliance trade-offs
- Comparison with pdfplumber, pdfminer.six

**Effort**: 2-3 hours

---

## PHASE 4: Create Optional Enhancement Layers

### 4.1 Post-Processor Framework
**New file**: `src/post_processors/mod.rs`
**Purpose**: Apply non-spec fixes AFTER spec-compliant extraction

```rust
pub trait PostProcessor {
    fn process(&self, document: &mut ExtractedDocument) -> Result<()>;
}

pub struct TextRepairProcessor {
    pub split_camelcase: bool,
    pub fix_empty_markers: bool,
    // ...
}

pub fn apply_post_processors(
    document: &mut ExtractedDocument,
    config: &PostProcessorConfig,
) -> Result<()> {
    if config.word_splitting.enabled {
        TextRepairProcessor::split_fused_words(document)?;
    }
    if config.bold_validation.enabled {
        BoldMarkerValidator::fix_empty_markers(document)?;
    }
    // ...
}
```

**Effort**: 3-4 hours

---

## PHASE 5: Create Spec-Strict Mode

**New configuration**: `TextExtractionConfig::spec_strict()`

```rust
impl TextExtractionConfig {
    /// Returns configuration that ONLY uses PDF spec features
    /// - TJ array offsets (Section 9.4.3)
    /// - Boundary whitespace (Section 9.4.3)
    /// - Geometric gaps with fixed 0.25em threshold (Section 9.4.4)
    /// - Font metrics (Section 9.3)
    pub fn spec_strict() -> Self {
        Self {
            // Core spec features
            use_tj_offsets: true,
            use_geometric_gaps: true,
            use_boundary_whitespace: true,

            // Disable ALL non-spec features
            use_adaptive_thresholds: false,
            enable_word_splitting: false,
            enable_layout_analysis: false,
            enable_table_detection: false,
            enable_heading_detection: false,

            // Fixed thresholds (from pdfplumber)
            geometric_gap_threshold_em: 0.25,  // Standard 0.25em
            ..Default::default()
        }
    }
}
```

**Testing**:
- [ ] Add test: `test_spec_strict_mode_disabled()`
- [ ] Run regression suite with `spec_strict()`
- [ ] Expected: Lower quality (3.5-4.5/10) but spec-compliant

**Effort**: 1-2 hours

---

## Execution Order (Recommended)

### Stage 1: Quick Removals + Annotations
1. **Phase 1.1**: Delete table_detector.rs (30 min)
2. **Phase 1.2**: Delete heading_detector.rs or annotate (1-2 hrs)
3. **Phase 1.3**: Delete ML classifier (30 min)
4. **Phase 3.1**: Annotate all non-spec code (3-4 hrs)
5. **Total**: ~6-8 hours → Immediate clarity on what's non-spec

### Stage 2: Documentation + Refactoring
6. **Phase 3.2**: Create spec compliance guide (2-3 hrs)
7. **Phase 2.1**: Move CamelCase to post-processor (2-3 hrs)
8. **Phase 2.3**: Move layout analysis to enhancement module (2-3 hrs)
9. **Total**: ~6-9 hours → Clean separation of concerns

### Stage 3: Framework + Testing
10. **Phase 4.1**: Create post-processor framework (3-4 hrs)
11. **Phase 5**: Create spec-strict mode (1-2 hrs)
12. **Testing**: Regression suite + quality metrics (2-3 hrs)
13. **Total**: ~6-9 hours → Production-ready clean architecture

---

## File Structure After Cleanup

```
src/
├── core/                          # SPEC-COMPLIANT ONLY
│   ├── text_extraction.rs         # Core text extraction (TJ, boundaries, gaps)
│   ├── geometric_spacing.rs       # Fixed 0.25em threshold (CURRENT geometric_spacing.rs)
│   └── font_metrics.rs            # Font state parameters (Tc, Tw, Th)
│
├── enhancements/                  # OPTIONAL, USER-CONTROLLED
│   ├── adaptive_thresholds.rs     # Gap statistics multipliers (from gap_statistics.rs)
│   ├── layout_analysis.rs         # Document analysis, column detection (ICDAR-based)
│   └── config.rs                  # Unified enhancement configuration
│
├── post_processors/               # APPLIED AFTER EXTRACTION (NON-SPEC)
│   ├── mod.rs                     # PostProcessor trait
│   ├── word_splitter.rs           # CamelCase splitting (from split_fused_words)
│   ├── bold_validator.rs          # Empty bold marker fixes (moved from converters)
│   └── spurious_space_fixer.rs    # Fix double spaces (Issue #2)
│
├── converters/
│   └── markdown.rs                # Markdown output (use post-processors)
│
└── [other modules unchanged]

docs/
├── PHASE10_PDF_SPEC_COMPLIANCE.md          # Existing
├── CLEANUP_ROADMAP.md                      # This file
└── SPEC_COMPLIANCE_GUIDE.md                # New - Comprehensive guide
```

---

## Quality & Compliance Matrix

| Config Mode | Word Fusions | Spurious Spaces | Empty Bold | Quality | Spec Compliant |
|-------------|--------------|-----------------|-----------|---------|----------------|
| spec_strict | ❌ 3 | ✅ 0 | ❌ 2-3 | 3.5/10 | ✅ 100% |
| default | ❌ 3 | ✅ 0 | ❌ 2-3 | 4.4/10 | 🟡 70% |
| with_enhancements | ❌ 3 | ✅ 0 | ❌ 2-3 | 6.5/10 | 🟡 50% |
| with_all_fixes | ✅ 0 | ✅ 0 | ✅ 0 | 8.5/10 | 🟡 40% |

---

## Critical Notes

### What to KEEP (with justification)
1. **Geometric spacing 0.25em threshold** ✅
   - Justified: pdfplumber standard, widely proven
   - Spec: Section 9.4.4 supports this interpretation
   - Config: Fixed (not adaptive)

2. **Boundary whitespace detection** ✅
   - Justified: Directly in PDF spec (Section 9.4.3)
   - Spec: "Spaces in text strings"
   - Config: No option (always on)

3. **TJ offset signals** ✅
   - Justified: Directly in PDF spec (Section 9.4.3)
   - Spec: "TJ array offsets determine positioning"
   - Config: No option (always on)

4. **Bold/italic detection from font flags** ✅
   - Justified: Font properties in PDF spec (Section 5.3.3)
   - Spec: Font.Flags, Font.FontWeight etc.
   - Config: Always on (core feature)

### What to REMOVE (no spec justification)
1. ❌ Table detection (move to optional)
2. ❌ Heading detection heuristics (move to optional)
3. ❌ ML classifiers (delete)
4. ❌ CamelCase splitting (move to post-processor)
5. ❌ Document-type profiles (annotate as heuristic)

### What to ANNOTATE (keep but document)
1. 📝 Adaptive gap multipliers (empirical, non-spec)
2. 📝 ICDAR layout analysis (academic, non-spec)
3. 📝 Unicode whitespace handling (PDF-specific workaround)

---

## Success Criteria

- [ ] All non-spec code clearly marked with **NON-SPEC HEURISTIC** comments
- [ ] New modules: `core/`, `enhancements/`, `post_processors/`
- [ ] `spec_strict()` configuration works (3.5/10 quality, 100% compliant)
- [ ] Default configuration improved (4.4→5.0/10, ~70% compliant)
- [ ] All fixes as optional post-processors (8.5/10, ~40% compliant but user-controlled)
- [ ] Comprehensive spec compliance guide published
- [ ] Regression suite passes for all configurations
- [ ] Clear user documentation: When to enable/disable features

---

## Commands for Testing

```bash
# Test spec-strict mode
cargo test --test quality_metrics -- --spec-strict

# Test with all enhancements
cargo test --test quality_metrics -- --enable-all

# Test post-processors
cargo test --test quality_metrics -- --with-post-processors

# Full regression suite
cargo test --test regression_suite
```

---

## Questions Before Starting

1. **Question 1**: Should we delete table detection entirely, or keep it but move to optional module?
   - **Recommended**: Delete (false positives, users have structure tree)

2. **Question 2**: For adaptive gaps, should we move to `enhancements/` or keep in core?
   - **Recommended**: Keep in core but annotate heavily (needed for current quality)

3. **Question 3**: Should spec_strict_mode() be the default or opt-in?
   - **Recommended**: Opt-in (users expect good quality by default)

---

## Timeline

- **Phase 1-2**: 8 hours → Remove/migrate non-spec code
- **Phase 3-4**: 8 hours → Create framework + documentation
- **Phase 5**: 3 hours → Testing + verification
- **Total**: ~19 hours → Production-ready clean architecture

Should we start with Phase 1 (quick removals)?
