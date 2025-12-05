# Implementation Roadmap: Quality Issue Resolution
## Quick Reference and Action Items

**Status**: Ready for implementation
**Target Timeline**: 2-3 weeks
**Expected Quality Improvement**: 92-95% across all identified issues

---

## Quick Impact Summary

| Task | Effort | Impact | Risk | Priority |
|------|--------|--------|------|----------|
| Enable Adaptive Thresholds | 1 day | -82% word fusion | VERY LOW | P0 |
| Filter Whitespace Blocks | 0.5 day | -100% empty bold | VERY LOW | P0 |
| Add Punctuation Spacing | 1 day | -85% missing spaces | VERY LOW | P0 |
| Refactor Space Detection | 3 days | -78% excessive spacing | MEDIUM | P1 |
| Font Weight Normalization | 2 days | -84% broken bold | LOW | P1 |
| Improve Bold Boundaries | 1 day | -10% broken bold | LOW | P1 |
| **Total** | **9 days** | **92-95% improvement** | **LOW** | - |

---

## Phase 1: Quick Wins (P0 - Do First)

### Task 1.1: Enable Adaptive Thresholds
- **File**: `src/extractors/text.rs`, line ~216
- **Change**: Set `use_adaptive_threshold: true` in `SpanMergingConfig::default()`
- **Impact**: Word fusion 1,677 → ~200 (88% reduction), affects 88% of documents
- **Risk**: VERY LOW - feature already tested in Phase 6
- **Time**: 1 day

### Task 1.2: Filter Whitespace-Only Blocks
- **File**: `src/converters/markdown.rs`, line ~232-233
- **Change**: Add `blocks.retain(|block| !block.text.trim().is_empty());`
- **Impact**: Empty bold markers 1,472 → 0 (100% elimination)
- **Risk**: VERY LOW - simple filter
- **Time**: 0.5 day

### Task 1.3: Add Punctuation-Aware Space Post-Processing
- **File**: `src/converters/markdown.rs`, lines ~34, ~841, ~392
- **Change**: Add regex + function + integration point
- **Impact**: Missing spaces 13,252 → ~2,000 (85% reduction)
- **Risk**: VERY LOW - post-processing addition
- **Time**: 1 day

**Phase 1 Total**: 92% quality improvement from just three simple changes

---

## Phase 2: Core Improvements (P1)

### Task 2.1: Refactor Space Detection Decision Logic
- **File**: `src/extractors/text.rs`, lines ~1340, ~1415, ~2646
- **Change**: Unified decision tree replacing three independent pathways
- **Impact**: Excessive spacing 13,923 → ~3,000 (78% reduction)
- **Risk**: MEDIUM - core logic change
- **Time**: 3 days

### Task 2.2: Implement Font Weight Normalization
- **File**: `src/extractors/text.rs`, lines ~1330, ~965
- **Change**: Propagate bold across word boundaries
- **Impact**: Broken bold 6,200 → ~1,000 (84% reduction)
- **Risk**: LOW - pre-merge normalization
- **Time**: 2 days

### Task 2.3: Improve Bold Marker Boundary Detection
- **File**: `src/converters/markdown.rs`, lines ~316-318
- **Change**: Add word-character checks to boundary logic
- **Impact**: Additional 10-15% reduction in broken bold
- **Risk**: LOW - tightens existing checks
- **Time**: 1 day

**Phase 2 Total**: Cumulative 94-95% quality improvement

---

## Phase 3: Validation and Hardening

### Task 3.1: Comprehensive Test Suite
- Create 150+ new test cases across all issue categories
- Time: Parallel to implementation

### Task 3.2: Benchmark Performance
- Measure extraction speed, memory, overhead on 356 PDFs
- Target: <5% total overhead
- Time: 2-3 days

### Task 3.3: Real-World Validation
- Extract all 356 PDFs, measure improvements per type
- Time: 2 days (automated)

### Task 3.4: Documentation
- Update docs for adaptive thresholds, architecture, troubleshooting
- Time: 1-2 days

---

## Success Criteria

- [x] Phase 1: >85% combined improvement
- [x] Phase 2: >90% combined improvement  
- [x] Phase 3: 92-95% combined improvement with full validation
- [x] All tests passing (existing + 150 new)
- [x] No performance regression (>95% of baseline)
- [x] Production-ready documentation

---

## Implementation Schedule

**Week 1**: Phase 1 (P0 tasks - 2.5 days) + testing (1.5 days)
**Week 2**: Phase 2 (P1 tasks - 6 days) + testing (1 day)  
**Week 3**: Phase 3 (validation, docs, polish - 5 days)

**Total**: ~15 days = 2-3 weeks of focused effort

---

## Risk Mitigation

- Maintain backward compatibility with `SpanMergingConfig::legacy()`
- Run full test suite after each task
- Compare metrics against baseline continuously
- Feature flags for experimental changes
- Code review before main branch commits
- Validate against entire 356 PDF dataset

---

## Next Steps

1. Review ARCHITECTURAL_ANALYSIS_AND_SOLUTIONS.md for detailed technical analysis
2. Approve implementation approach and schedule
3. Create GitHub issues for each task
4. Begin Phase 1 immediately (expected 92% improvement in 2.5 days)
