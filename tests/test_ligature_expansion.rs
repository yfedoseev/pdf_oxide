//! Test suite for Week 2 Day 6: Ligature Expansion Enhancement (2A)
//!
//! This test suite verifies intelligent ligature splitting at word boundaries.
//! When a ligature (fi, fl, ffi, ffl, ff) is followed by a word boundary,
//! it should be split into component characters. When not followed by a boundary,
//! it should be kept as a ligature.
//!
//! Tests follow TDD approach: written first, expected to fail, then implementation
//! makes them pass.

use pdf_oxide::text::ligature_processor::{
    expand_ligature_to_chars, get_ligature_components, LigatureDecision, LigatureDecisionMaker,
};
use pdf_oxide::text::{BoundaryContext, CharacterInfo};

/// Helper: Create a CharacterInfo for testing
fn create_char_info(
    code: u32,
    width: f32,
    x_pos: f32,
    tj_offset: Option<i32>,
    is_ligature: bool,
) -> CharacterInfo {
    CharacterInfo {
        code,
        glyph_id: Some(1),
        width,
        x_position: x_pos,
        tj_offset,
        font_size: 12.0,
        is_ligature,
        original_ligature: None,
        protected_from_split: false,
    }
}

#[test]
fn test_ligature_split_after_boundary() {
    // Test case: "sufficient" with "fi" ligature (U+FB01) followed by boundary
    // Expected: ligature should be split because boundary comes after

    let ligature_fi = create_char_info(0xFB01, 500.0, 0.0, None, true); // 'ﬁ' ligature
    let next_char = create_char_info(0x63, 400.0, 520.0, Some(-150), false); // 'c' with large TJ offset

    let context = BoundaryContext::new(12.0);
    let decision = LigatureDecisionMaker::decide(&ligature_fi, &context, Some(&next_char));

    assert_eq!(
        decision,
        LigatureDecision::Split,
        "Ligature followed by word boundary (large TJ offset) should be split"
    );
}

#[test]
fn test_ligature_keep_no_boundary() {
    // Test case: "fi" ligature with no boundary after
    // Expected: keep as ligature (single character)

    let ligature_fi = create_char_info(0xFB01, 500.0, 0.0, None, true); // 'ﬁ' ligature
    let next_char = create_char_info(0x6E, 400.0, 500.0, None, false); // 'n' with normal spacing

    let context = BoundaryContext::new(12.0);
    let decision = LigatureDecisionMaker::decide(&ligature_fi, &context, Some(&next_char));

    assert_eq!(
        decision,
        LigatureDecision::Keep,
        "Ligature with no boundary after should be kept intact"
    );
}

#[test]
fn test_ligature_width_distribution() {
    // Test case: "fi" ligature (width 500) should split to "f" (250) + "i" (250)
    // Expected: proportional width distribution

    let ligature_width = 500.0;
    let components = expand_ligature_to_chars('ﬁ', ligature_width);

    assert_eq!(components.len(), 2, "fi ligature should expand to 2 characters");
    assert_eq!(components[0].0, 'f', "First component should be 'f'");
    assert_eq!(components[1].0, 'i', "Second component should be 'i'");
    assert_eq!(components[0].1, 250.0, "f width should be half of ligature");
    assert_eq!(components[1].1, 250.0, "i width should be half of ligature");
}

#[test]
fn test_ligature_ffl_expansion() {
    // Test case: "ffl" ligature (width 600) → f(200) + f(200) + l(200)
    // Expected: all three components created with proportional widths

    let ligature_width = 600.0;
    let components = expand_ligature_to_chars('ﬄ', ligature_width);

    assert_eq!(components.len(), 3, "ffl ligature should expand to 3 characters");
    assert_eq!(components[0].0, 'f', "First component should be 'f'");
    assert_eq!(components[1].0, 'f', "Second component should be 'f'");
    assert_eq!(components[2].0, 'l', "Third component should be 'l'");

    // Each component should have equal width (600 / 3 = 200)
    assert_eq!(components[0].1, 200.0, "First f width should be 1/3 of ligature");
    assert_eq!(components[1].1, 200.0, "Second f width should be 1/3 of ligature");
    assert_eq!(components[2].1, 200.0, "l width should be 1/3 of ligature");
}

#[test]
fn test_ligature_at_text_end() {
    // Test case: text ending with ligature "fi"
    // Expected: Keep (no next char, no boundary decision)

    let ligature_fi = create_char_info(0xFB01, 500.0, 0.0, None, true); // 'ﬁ' at end

    let context = BoundaryContext::new(12.0);
    let decision = LigatureDecisionMaker::decide(&ligature_fi, &context, None);

    assert_eq!(
        decision,
        LigatureDecision::Keep,
        "Ligature at end of text should be kept (no boundary to split at)"
    );
}

#[test]
fn test_multiple_ligatures_in_span() {
    // Test case: "file" with potential "fi" ligature + regular characters
    // Expected: each ligature decision is independent

    let ligature_fi = create_char_info(0xFB01, 500.0, 0.0, None, true); // 'ﬁ'
    let char_l = create_char_info(0x6C, 300.0, 500.0, None, false); // 'l'
    let char_e = create_char_info(0x65, 400.0, 800.0, Some(-200), false); // 'e' with boundary after

    let context = BoundaryContext::new(12.0);

    // First ligature followed by normal char: Keep
    let decision1 = LigatureDecisionMaker::decide(&ligature_fi, &context, Some(&char_l));
    assert_eq!(
        decision1,
        LigatureDecision::Keep,
        "First ligature with no boundary should be kept"
    );

    // If there was another ligature before 'e' with boundary, it would split
    let ligature_fl = create_char_info(0xFB02, 500.0, 500.0, None, true); // 'ﬂ'
    let decision2 = LigatureDecisionMaker::decide(&ligature_fl, &context, Some(&char_e));
    assert_eq!(decision2, LigatureDecision::Split, "Ligature before boundary should be split");
}

#[test]
fn test_ligature_position_recalculation() {
    // Test case: verify x_positions are correct after split
    // "fi" at x=0.0, width=500.0, split to "f" + "i"
    // Expected: f at 0.0 (width 250), i at 250.0 (width 250)

    let ligature_width = 500.0;
    let components = expand_ligature_to_chars('ﬁ', ligature_width);

    // Calculate positions based on widths
    let f_start = 0.0;
    let f_width = components[0].1;
    let i_start = f_start + f_width;
    let i_width = components[1].1;

    assert_eq!(f_start, 0.0, "f should start at original position");
    assert_eq!(f_width, 250.0, "f should have half width");
    assert_eq!(i_start, 250.0, "i should start where f ends");
    assert_eq!(i_width, 250.0, "i should have half width");

    // Total width should equal original ligature width
    assert_eq!(
        f_width + i_width,
        ligature_width,
        "Sum of component widths should equal original ligature width"
    );
}

#[test]
fn test_ligature_with_geometric_gap() {
    // Test case: ligature followed by large geometric gap
    // Expected: Split because geometric gap indicates boundary

    let ligature_fi = create_char_info(0xFB01, 500.0, 0.0, None, true); // 'ﬁ'
                                                                        // Next char has large gap (500 + 600 = 1100, gap of 600 units)
    let next_char = create_char_info(0x61, 400.0, 1100.0, None, false); // 'a' far away

    let context = BoundaryContext::new(12.0);
    let decision = LigatureDecisionMaker::decide(&ligature_fi, &context, Some(&next_char));

    assert_eq!(
        decision,
        LigatureDecision::Split,
        "Ligature followed by large geometric gap should be split"
    );
}

#[test]
fn test_get_ligature_components() {
    // Test all standard ligatures return correct component strings

    assert_eq!(get_ligature_components('ﬀ'), Some("ff"), "ff ligature");
    assert_eq!(get_ligature_components('ﬁ'), Some("fi"), "fi ligature");
    assert_eq!(get_ligature_components('ﬂ'), Some("fl"), "fl ligature");
    assert_eq!(get_ligature_components('ﬃ'), Some("ffi"), "ffi ligature");
    assert_eq!(get_ligature_components('ﬄ'), Some("ffl"), "ffl ligature");

    assert_eq!(get_ligature_components('a'), None, "Regular character should return None");
    assert_eq!(get_ligature_components('A'), None, "Capital letter should return None");
}

#[test]
fn test_expand_ligature_ff() {
    // Test ff ligature expansion
    let components = expand_ligature_to_chars('ﬀ', 600.0);

    assert_eq!(components.len(), 2, "ff ligature should expand to 2 characters");
    assert_eq!(components[0].0, 'f', "First component should be 'f'");
    assert_eq!(components[1].0, 'f', "Second component should be 'f'");
    assert_eq!(components[0].1, 300.0, "First f width should be half");
    assert_eq!(components[1].1, 300.0, "Second f width should be half");
}

#[test]
fn test_expand_ligature_fl() {
    // Test fl ligature expansion
    let components = expand_ligature_to_chars('ﬂ', 480.0);

    assert_eq!(components.len(), 2, "fl ligature should expand to 2 characters");
    assert_eq!(components[0].0, 'f', "First component should be 'f'");
    assert_eq!(components[1].0, 'l', "Second component should be 'l'");
    assert_eq!(components[0].1, 240.0, "f width should be half");
    assert_eq!(components[1].1, 240.0, "l width should be half");
}

#[test]
fn test_expand_ligature_ffi() {
    // Test ffi ligature expansion
    let components = expand_ligature_to_chars('ﬃ', 750.0);

    assert_eq!(components.len(), 3, "ffi ligature should expand to 3 characters");
    assert_eq!(components[0].0, 'f', "First component should be 'f'");
    assert_eq!(components[1].0, 'f', "Second component should be 'f'");
    assert_eq!(components[2].0, 'i', "Third component should be 'i'");
    assert_eq!(components[0].1, 250.0, "First f width should be 1/3");
    assert_eq!(components[1].1, 250.0, "Second f width should be 1/3");
    assert_eq!(components[2].1, 250.0, "i width should be 1/3");
}

#[test]
fn test_non_ligature_expansion() {
    // Test that non-ligature characters return empty vec
    let components = expand_ligature_to_chars('a', 400.0);
    assert!(components.is_empty(), "Non-ligature should return empty vec");

    let components = expand_ligature_to_chars('A', 500.0);
    assert!(components.is_empty(), "Capital letter should return empty vec");

    let components = expand_ligature_to_chars('1', 300.0);
    assert!(components.is_empty(), "Digit should return empty vec");
}

#[test]
fn test_ligature_decision_boundary_threshold() {
    // Test threshold for TJ offset boundary detection
    // TJ offset threshold is -100 (values more negative indicate boundaries)

    let ligature_fi = create_char_info(0xFB01, 500.0, 0.0, None, true);
    let context = BoundaryContext::new(12.0);

    // Just below threshold (-50): should Keep
    let next_small_offset = create_char_info(0x61, 400.0, 500.0, Some(-50), false);
    let decision1 = LigatureDecisionMaker::decide(&ligature_fi, &context, Some(&next_small_offset));
    assert_eq!(
        decision1,
        LigatureDecision::Keep,
        "Small TJ offset (-50) should not trigger split"
    );

    // At threshold (-100): should Keep (not strictly less than)
    let next_threshold = create_char_info(0x61, 400.0, 500.0, Some(-100), false);
    let decision2 = LigatureDecisionMaker::decide(&ligature_fi, &context, Some(&next_threshold));
    assert_eq!(
        decision2,
        LigatureDecision::Keep,
        "TJ offset at threshold (-100) should not trigger split"
    );

    // Beyond threshold (-150): should Split
    let next_large_offset = create_char_info(0x61, 400.0, 500.0, Some(-150), false);
    let decision3 = LigatureDecisionMaker::decide(&ligature_fi, &context, Some(&next_large_offset));
    assert_eq!(
        decision3,
        LigatureDecision::Split,
        "Large TJ offset (-150) should trigger split"
    );
}

#[test]
fn test_ligature_decision_geometric_threshold() {
    // Test geometric gap threshold
    // Threshold is 0.5 * font_size = 0.5 * 12.0 = 6.0

    let ligature_fi = create_char_info(0xFB01, 500.0, 0.0, None, true);
    let context = BoundaryContext::new(12.0);

    // Small gap (3.0 < 6.0): should Keep
    // ligature ends at 500.0, next starts at 503.0, gap = 3.0
    let next_small_gap = create_char_info(0x61, 400.0, 503.0, None, false);
    let decision1 = LigatureDecisionMaker::decide(&ligature_fi, &context, Some(&next_small_gap));
    assert_eq!(
        decision1,
        LigatureDecision::Keep,
        "Small geometric gap (3.0) should not trigger split"
    );

    // At threshold (6.0): should Keep (not strictly greater)
    let next_threshold = create_char_info(0x61, 400.0, 506.0, None, false);
    let decision2 = LigatureDecisionMaker::decide(&ligature_fi, &context, Some(&next_threshold));
    assert_eq!(
        decision2,
        LigatureDecision::Keep,
        "Geometric gap at threshold (6.0) should not trigger split"
    );

    // Large gap (10.0 >= 6.0): should Split
    let next_large_gap = create_char_info(0x61, 400.0, 510.0, None, false);
    let decision3 = LigatureDecisionMaker::decide(&ligature_fi, &context, Some(&next_large_gap));
    assert_eq!(
        decision3,
        LigatureDecision::Split,
        "Large geometric gap (10.0) should trigger split"
    );
}
