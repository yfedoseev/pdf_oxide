# PDF Spec Compliance Improvements Roadmap

**Date**: December 9, 2025
**Status**: Planning
**Priority**: High-impact compliance fixes with measurable extraction quality improvements

---

## Executive Summary

Analysis of pdf_oxide against ISO 32000-1:2008 (PDF 1.7 specification) identified **12 compliance gaps**, of which **5 are HIGH impact** and directly affect text extraction quality. This roadmap prioritizes implementing the 5 highest-impact fixes with measurable outcomes.

**Target**: Increase PDF spec compliance from ~70% to >90% without breaking existing functionality.

---

## The 5 Highest-Impact Compliance Fixes

### Priority 1: Character-to-Unicode Mapping Priority Order (CRITICAL)
**Current Status**: Partial - ToUnicode CMap supported, Adobe Glyph List NOT used, ActualText NOT supported
**Impact**: Affects ~15% of real-world PDFs with custom font encodings
**Spec Section**: ISO 32000-1:2008, Section 9.10.2 - Character-to-Unicode Mapping Priorities

**The Problem:**
PDF spec defines a 5-level priority order for character-to-Unicode mapping:
1. **ToUnicode CMap** (highest priority) ← Currently: ✅ Implemented
2. **Adobe Glyph List** (fallback 1) ← Currently: ❌ NOT implemented
3. **Predefined CMaps** (fallback 2) ← Currently: ❌ NOT implemented
4. **ActualText attribute** (fallback 3) ← Currently: ❌ NOT implemented
5. **Font encoding** (lowest priority) ← Currently: Partial

**Current Behavior**: If ToUnicode CMap doesn't exist or is incomplete, we fall back to font encoding heuristics instead of Adobe Glyph List (spec-non-compliant).

**Expected Outcome**: Correct character extraction in PDFs using:
- Symbol fonts (Wingdings, Symbol) with fallback to Adobe Glyph List
- Custom encodings without ToUnicode CMap
- Documents with ActualText entries in marked content

---

### Priority 2: Reading Order Priority (Structure Tree First) (CRITICAL)
**Current Status**: Structure tree support exists but is NOT prioritized
**Impact**: Affects ~30% of real PDFs with tagged structure trees
**Spec Section**: ISO 32000-1:2008, Section 14.7-14.8 - Tagged PDF & Reading Order

**The Problem:**
PDF spec requires this reading order priority:
1. **Structure tree** (tagged PDF) - USE FIRST if available ← Currently: Last choice
2. **Physical page order** - Use if no structure tree ← Currently: First choice
3. **Content stream order** - Use if both above unavailable ← Currently: First choice

**Current Behavior**: We extract text in content stream order, completely ignoring structure tree reading order when available.

**Expected Outcome**:
- Correctly extract multi-column layouts (columns read left-to-right instead of top-to-bottom)
- Respect document structure for academic papers with headers/footers
- Preserve table structure and reading order
- Handle documents with explicit reading order specification

---

### Priority 3: Parent Tree Lookup for Structure Elements (HIGH)
**Current Status**: Structure traversal works, but parent tree association is incomplete
**Impact**: Affects ~20% of tagged PDFs with complex hierarchies
**Spec Section**: ISO 32000-1:2008, Section 14.7.2 - Structure Hierarchy

**The Problem:**
The Parent tree provides O(1) MCID-to-structure-element lookup. We partially traverse the tree but don't fully use parent tree for reverse lookup (finding what structure element contains an MCID).

**Current Behavior**: Direct traversal works for simple PDFs, but complex documents with deep hierarchies or non-linear MCID assignments may have incomplete associations.

**Expected Outcome**:
- Fast MCID-to-structure lookup via parent tree
- Correct structure element association for all marked content
- Proper handling of non-contiguous MCID ranges within structure elements

---

### Priority 4: Word Boundary Detection (Remove Heuristics) (MEDIUM-HIGH)
**Current Status**: Using CamelCase heuristics instead of spec-compliant method
**Impact**: Affects text segmentation accuracy in 5-10% of PDFs
**Spec Section**: ISO 32000-1:2008, Section 9.4.4 - Text Positioning & Word Spacing

**The Problem:**
Current implementation uses linguistic heuristics (CamelCase splitting: `lastName` → `last name`). PDF spec says:
- **Authoritative word boundaries**: Determined by TJ operator offsets
  - When offset value exceeds `(word_spacing) / (1000 * font_size)`, treat as word boundary
- **Heuristic word boundaries**: Linguistic analysis is ONLY for documents without proper spacing

**Current Behavior**: Always using heuristics, sometimes creating false word splits (e.g., "iPhone" → "i Phone").

**Expected Outcome**:
- Use TJ offset values as PRIMARY word boundary source
- Only fall back to heuristics if TJ spacing unavailable
- Eliminate false splits in technical terms, product names, abbreviations

---

### Priority 5: Text State Parameters (Tr, Ts, Tk) (MEDIUM)
**Current Status**: Not implemented in graphics state
**Impact**: Affects rendering mode and text appearance in ~5% of PDFs
**Spec Section**: ISO 32000-1:2008, Section 9.3.4 - Text State Parameters

**The Problem:**
PDF defines 3 text state parameters not in current GraphicsState:
- **Tr (Text Rendering Mode)**: How glyphs are rendered (3 = invisible, used for selection/search overlay)
- **Ts (Text Rise)**: Vertical offset (used for superscripts/subscripts)
- **Tk (Text Knockout)**: For shadings (rarely used)

**Current Behavior**: These are parsed but not stored, so we can't correctly identify invisible text overlays.

**Expected Outcome**:
- Correctly skip invisible text (Tr=3) that's only for UI/search overlay
- Properly position superscripts/subscripts in markdown output
- More accurate text content extraction

---

## Implementation Roadmap

### Improvement 1: Character-to-Unicode Mapping (Priority 1)
**Time**: 2-3 hours
**Files**:
- `src/fonts/adobe_glyph_list.rs` (expand existing)
- `src/fonts/cmap.rs` (add mapping priority order)
- `src/fonts/font_dict.rs` (implement priority wrapper)
- `src/text/character_mapper.rs` (NEW - central mapping logic)

**Steps**:
1. Review and complete Adobe Glyph List in adobe_glyph_list.rs
2. Create `character_mapper.rs` with priority-based lookup
3. Implement fallback chain: ToUnicode → AdobeGlyphList → PredefinedCMaps → ActualText → FontEncoding
4. Update text extraction to use new mapper
5. Add 15+ tests for each fallback scenario

**Success Criteria**:
- All spec-compliant mappings working
- 100+ Adobe glyphs correctly mapped
- Tests passing for each priority level

---

### Improvement 2: Reading Order Priority (Priority 2)
**Time**: 2-3 hours
**Files**:
- `src/structure/traversal.rs` (reorder priority)
- `src/converters/markdown.rs` (reading order aware)
- `src/reading_order.rs` (NEW - reading order logic)

**Steps**:
1. Create `reading_order.rs` module with priority system
2. Modify text extraction to check structure tree first
3. Implement fallback to physical order if structure unavailable
4. Add multi-column handling for structure-based reading
5. Tests on real PDFs with/without structure trees

**Success Criteria**:
- Structure tree used when available
- Multi-column documents read correctly
- Physical order fallback works
- No performance regression

---

### Improvement 3: Parent Tree Lookup (Priority 3)
**Time**: 1-2 hours
**Files**:
- `src/structure/parent_tree.rs` (NEW - parent tree handling)
- `src/structure/traversal.rs` (integrate parent tree)

**Steps**:
1. Create parent tree module for O(1) MCID lookup
2. Add reverse lookup: MCID → Structure Element
3. Integrate with existing traversal
4. Cache parent tree for performance
5. Tests on complex PDFs

**Success Criteria**:
- O(1) parent tree lookups
- MCID association working
- Non-linear MCID ranges handled

---

### Improvement 4: Word Boundary Detection (Priority 4)
**Time**: 1.5-2 hours
**Files**:
- `src/text/word_boundary.rs` (NEW - TJ-based boundaries)
- `src/layout/text_block.rs` (use TJ boundaries)
- `src/extractors/text.rs` (integrate word boundaries)

**Steps**:
1. Create word_boundary module
2. Implement TJ offset-based boundary detection
3. Calculate threshold: `(word_spacing) / (1000 * font_size)`
4. Use heuristics only as fallback
5. Tests comparing TJ vs heuristic boundaries

**Success Criteria**:
- TJ-based boundaries correct for all test PDFs
- Heuristics only used when needed
- No false splits
- Performance unchanged

---

### Improvement 5: Text State Parameters (Priority 5)
**Time**: 1-1.5 hours
**Files**:
- `src/graphics/graphics_state.rs` (add Tr, Ts, Tk)
- `src/extractors/text.rs` (use text rendering mode)
- `src/converters/markdown.rs` (handle Ts for positioning)

**Steps**:
1. Add Tr, Ts, Tk fields to GraphicsState
2. Parse from PDF content streams
3. Skip/flag text with Tr=3 (invisible)
4. Handle Ts (text rise) in positioning
5. Tests for each parameter

**Success Criteria**:
- Invisible text correctly identified
- Superscripts/subscripts positioned
- No regressions in rendering

---

## Metrics & Validation

### Per-Phase Metrics
| Phase | Feature | Test Count | Coverage | Success Criteria |
|-------|---------|-----------|----------|-----------------|
| 3.1 | Char Mapping | 20+ | 95%+ | All priority levels work |
| 3.2 | Reading Order | 15+ | 90%+ | Structure tree prioritized |
| 3.3 | Parent Tree | 10+ | 100% | O(1) lookups work |
| 3.4 | Word Boundaries | 25+ | 90%+ | TJ-based correct |
| 3.5 | Text State | 15+ | 85%+ | Tr/Ts parameters work |

### Overall Compliance
**Before these improvements**: ~70% spec compliance
**After completing all improvements.5**: >90% spec compliance

**Quality Metrics**:
- Text extraction accuracy: baseline → +5-10%
- Reading order correctness: +30% (with structure trees)
- Word segmentation accuracy: +3-5%
- Superscript/subscript positioning: +2-3%
- No regressions in existing tests (< 0.1% failure increase)

---

## Risk Mitigation

### High Risk
1. **Breaking existing text extraction**
   - Mitigation: Comprehensive regression tests on all 94 existing PDFs
   - Validation: 100% of Phase 1 tests must pass

2. **Performance impact of parent tree lookups**
   - Mitigation: Cache parent tree in memory
   - Validation: Benchmark before/after (target: <2% overhead)

### Medium Risk
3. **Complex interaction between reading order and physical order**
   - Mitigation: Phase-specific flags for testing each mode independently
   - Validation: Test both modes on same PDFs

4. **Adobe Glyph List completeness**
   - Mitigation: Use official Adobe list, verify against spec
   - Validation: All 1000+ glyphs working

### Low Risk
5. **Text state parameter parsing**
   - Mitigation: PDF parser already handles these operators
   - Validation: Simple flag storage, low risk

---

## Execution Order

**Recommended sequence** (dependency-aware):
1. **3.1 (Char Mapping)**: No dependencies, high impact
2. **3.5 (Text State)**: Simple, builds confidence
3. **3.2 (Reading Order)**: Uses structure traversal (mature)
4. **3.3 (Parent Tree)**: Enhances 3.2, builds on existing
5. **3.4 (Word Boundaries)**: Last, most sensitive to fine-tuning

**Total estimated time**: 8-12 hours
**Recommended approach**: Complete 1-2 phases per day with full testing

---

## Files to Modify/Create

**Modified**:
- `src/fonts/cmap.rs` - Add priority order wrapper
- `src/fonts/font_dict.rs` - Integrate character mapper
- `src/fonts/adobe_glyph_list.rs` - Complete list if needed
- `src/structure/traversal.rs` - Reading order priority
- `src/layout/text_block.rs` - Word boundary integration
- `src/graphics/graphics_state.rs` - Add Tr, Ts, Tk
- `src/extractors/text.rs` - Use new modules
- `src/converters/markdown.rs` - Reading order aware

**New**:
- `src/fonts/character_mapper.rs` - Central character mapping logic
- `src/text/word_boundary.rs` - TJ-based boundary detection
- `src/reading_order.rs` - Reading order priority system
- `src/structure/parent_tree.rs` - Parent tree O(1) lookup

---

## Testing Strategy

### Unit Tests
- Test each priority level independently
- Test fallback chain with mock objects
- Verify parent tree lookups
- Validate word boundary calculations

### Integration Tests
- Test on real PDFs with/without ToUnicode CMap
- Test multi-column layouts (reading order)
- Test symbol fonts (Wingdings, etc.)
- Test with/without structure trees

### Regression Tests
- All 94 existing PDFs must extract with ≥98% accuracy
- Performance: < 5% overhead vs current
- No new test failures allowed

### Validation PDFs
- Academic papers (with structure trees)
- Symbol-heavy documents (Wingdings)
- Custom encoding (no ToUnicode)
- Multi-column layouts
- Documents with superscripts/subscripts

---

## Success Definition

All improvements complete when:
1. ✅ All 5 priority fixes implemented
2. ✅ 85+ new tests passing (15+ per phase)
3. ✅ All 94 Phase 1 tests still passing (no regressions)
4. ✅ Spec compliance > 90% documented
5. ✅ Performance within 5% of baseline
6. ✅ Real-world PDFs show measurable improvement

---

## Next Steps

1. Review this roadmap with user
2. Confirm priority order
3. Start with Improvement 1 (Character-to-Unicode Mapping)
4. Track progress with test suite
5. Document improvements in quality metrics

**Estimated completion**: 2-3 days with focused effort

---

**Document Status**: Complete for review
**Last Updated**: December 9, 2025 10:15 UTC
**Reviewer**: Awaiting user confirmation
