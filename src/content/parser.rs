//! Content stream parser.
//!
//! This module parses PDF content streams into a sequence of operators.
//! Content streams are fundamentally different from the main PDF structure:
//! they use a postfix notation where operands come before operators.
//!
//! Example content stream:
//! ```text
//! BT
//!   /F1 12 Tf
//!   100 700 Td
//!   (Hello, World!) Tj
//! ET
//! ```

use crate::content::operators::{Operator, TextElement};
use crate::error::Result;
use crate::object::Object;
use crate::parser::parse_object;
use nom::bytes::complete::take_while1;
use nom::character::complete::multispace0;
use nom::IResult;
use nom::Parser;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Maximum number of operators to parse from a single content stream.
///
/// Prevents pathological inputs (e.g., Isartor 6.1.12) from consuming
/// unbounded time and memory.
const MAX_OPERATORS: usize = 1_000_000;

/// Maximum consecutive parse errors (byte skips) before bailing out.
///
/// If we skip this many bytes without finding a valid operator, the
/// remaining data is likely junk, not a parseable content stream.
const MAX_CONSECUTIVE_ERRORS: usize = 1024;

/// Parse a content stream into a sequence of operators.
///
/// Content streams use postfix notation where operands precede the operator.
/// For example: `100 200 Td` means "move text position to (100, 200)".
///
/// Includes safety limits: bails out after [`MAX_OPERATORS`] operators or
/// [`MAX_CONSECUTIVE_ERRORS`] consecutive parse failures.
pub fn parse_content_stream(data: &[u8]) -> Result<Vec<Operator>> {
    let estimated_capacity = data.len() / 20;
    let mut operators = Vec::with_capacity(estimated_capacity.min(100_000));
    let mut input = data;
    let mut consecutive_errors: usize = 0;

    // Parse until we consume all input
    while !input.is_empty() {
        // Skip whitespace
        if let Ok((rest, _)) = multispace0::<&[u8], nom::error::Error<&[u8]>>.parse(input) {
            input = rest;
        }

        // Check if we're done
        if input.is_empty() {
            break;
        }

        // Parse one operator with its operands
        match parse_operator_with_operands(input) {
            Ok((rest, op)) => {
                operators.push(op);
                input = rest;
                consecutive_errors = 0;

                if operators.len() >= MAX_OPERATORS {
                    log::warn!("Content stream exceeded {} operators, truncating", MAX_OPERATORS);
                    break;
                }
            },
            Err(_) => {
                consecutive_errors += 1;
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                    log::warn!(
                        "Content stream had {} consecutive parse errors, bailing out ({} bytes remaining)",
                        MAX_CONSECUTIVE_ERRORS,
                        input.len()
                    );
                    break;
                }
                // If we can't parse, skip the problematic byte and continue
                // This makes us more resilient to malformed streams
                if input.len() > 1 {
                    input = &input[1..];
                } else {
                    break;
                }
            },
        }
    }

    Ok(operators)
}

/// Parse a content stream for text extraction, skipping pure graphics operators.
///
/// This is a performance-optimized variant of [`parse_content_stream`] that
/// avoids constructing `Object` operands for operators that only affect paths,
/// clipping, and non-text graphics state. Inside BT/ET text blocks, parsing is
/// identical to the full parser.
///
/// # Performance
///
/// For graphics-heavy pages (e.g., 1–12 MB of path data), this can be 3–5x
/// faster than full parsing while producing identical text extraction results.
/// The speedup comes from byte-level operand skipping (no `f64` parsing, no
/// heap allocation) and discarding path/clipping operators entirely.
///
/// # Safety limits
///
/// Same as [`parse_content_stream`]: bails out after [`MAX_OPERATORS`]
/// operators or [`MAX_CONSECUTIVE_ERRORS`] consecutive parse failures.
pub fn parse_content_stream_text_only(data: &[u8]) -> Result<Vec<Operator>> {
    let estimated_capacity = data.len() / 40;
    let mut operators = Vec::with_capacity(estimated_capacity.min(50_000));
    let mut input = data;
    let mut consecutive_errors: usize = 0;
    let mut inside_text = false;

    while !input.is_empty() {
        if let Ok((rest, _)) = multispace0::<&[u8], nom::error::Error<&[u8]>>.parse(input) {
            input = rest;
        }
        if input.is_empty() {
            break;
        }

        if operators.len() >= MAX_OPERATORS {
            log::warn!("Content stream exceeded {} operators, truncating", MAX_OPERATORS);
            break;
        }

        if inside_text {
            // Inside BT/ET: full parse, identical to parse_content_stream
            match parse_operator_with_operands(input) {
                Ok((rest, op)) => {
                    if matches!(op, Operator::EndText) {
                        inside_text = false;
                    }
                    operators.push(op);
                    input = rest;
                    consecutive_errors = 0;
                },
                Err(_) => {
                    consecutive_errors += 1;
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        log::warn!(
                            "Content stream had {} consecutive parse errors, bailing out ({} bytes remaining)",
                            MAX_CONSECUTIVE_ERRORS,
                            input.len()
                        );
                        break;
                    }
                    if input.len() > 1 {
                        input = &input[1..];
                    } else {
                        break;
                    }
                },
            }
        } else {
            // Outside BT/ET: byte-level scan — skip operands and graphics
            // operators using raw index arithmetic (no nom IResult overhead).
            match scan_graphics_region(input, &mut consecutive_errors) {
                ScanResult::EndOfData => break,
                ScanResult::FoundBT { rest } => {
                    operators.push(Operator::BeginText);
                    input = rest;
                    inside_text = true;
                },
                ScanResult::InlineImage { rest } => match parse_inline_image(rest) {
                    Ok((rest2, _)) => input = rest2,
                    Err(_) => input = rest,
                },
                ScanResult::NeedFullParse {
                    operand_start,
                    after_op,
                } => match parse_operator_with_operands(operand_start) {
                    Ok((rest2, op)) => {
                        operators.push(op);
                        input = rest2;
                    },
                    Err(_) => input = after_op,
                },
                ScanResult::DeferredThenText {
                    deferred_start,
                    trigger_start,
                } => {
                    // Re-parse the deferred q/cm/Q region to emit CTM-affecting ops.
                    // The trigger (BT/BI/Do/etc.) is NOT included — the next iteration
                    // of the outer loop re-enters scan_graphics_region which returns
                    // the trigger via FoundBT / InlineImage / NeedFullParse.
                    let mut remaining = deferred_start;
                    while remaining.len() > trigger_start.len() {
                        match parse_operator_with_operands(remaining) {
                            Ok((rest2, op)) => {
                                operators.push(op);
                                remaining = rest2;
                            },
                            Err(_) => {
                                if remaining.len() > 1 {
                                    remaining = &remaining[1..];
                                } else {
                                    break;
                                }
                            },
                        }
                    }
                    input = trigger_start;
                    consecutive_errors = 0;
                },
                ScanResult::SimpleOp { op, rest } => {
                    operators.push(op);
                    input = rest;
                },
                ScanResult::TooManyErrors { remaining } => {
                    log::warn!(
                        "Content stream had {} consecutive parse errors, bailing out ({} bytes remaining)",
                        MAX_CONSECUTIVE_ERRORS,
                        remaining.len()
                    );
                    break;
                },
            }
        }
    }

    Ok(operators)
}

/// SIMD-accelerated pre-scan to identify text-bearing regions in large content streams.
///
/// For streams > 256KB that are mostly graphics (path operators, color ops), this uses
/// memchr to locate BT/Do operator positions in ~1ms instead of byte-by-byte scanning
/// at ~500ms. Returns parse regions that cover BT..ET blocks and Do operators, plus
/// preceding graphics state (q..cm) needed for correct CTM.
///
/// Returns `None` on ambiguous cases (fallback to full scan).
fn prescan_text_regions(data: &[u8]) -> Option<Vec<(usize, usize)>> {
    fn is_boundary(b: u8) -> bool {
        b.is_ascii_whitespace()
            || matches!(b, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
    }

    let len = data.len();
    // Collect positions of BT and Do operators (text-bearing operators)
    let mut text_positions: Vec<usize> = Vec::new();
    let mut offset = 0;

    // Use memchr to find 'B' and 'D' candidates (SIMD-accelerated)
    loop {
        match memchr::memchr2(b'B', b'D', &data[offset..]) {
            None => break,
            Some(rel_pos) => {
                let pos = offset + rel_pos;
                offset = pos + 1;

                // Check for "BT" at boundary
                #[allow(clippy::if_same_then_else)]
                if data[pos] == b'B' && pos + 1 < len && data[pos + 1] == b'T' {
                    let before_ok = pos == 0 || is_boundary(data[pos - 1]);
                    let after_ok = pos + 2 >= len || is_boundary(data[pos + 2]);
                    if before_ok && after_ok {
                        text_positions.push(pos);
                    }
                }
                // Check for "Do" at boundary
                else if data[pos] == b'D' && pos + 1 < len && data[pos + 1] == b'o' {
                    let before_ok = pos == 0 || is_boundary(data[pos - 1]);
                    let after_ok = pos + 2 >= len || is_boundary(data[pos + 2]);
                    if before_ok && after_ok {
                        text_positions.push(pos);
                    }
                }
            },
        }
    }

    if text_positions.is_empty() {
        // No text operators — caller can skip the entire stream
        return Some(Vec::new());
    }

    // For each text position, scan backwards to find the nearest unmatched 'q'
    // to capture CTM state (cm operators between q and BT/Do).
    let mut regions: Vec<(usize, usize)> = Vec::new();

    for &tp in &text_positions {
        // Find region start: scan backwards for unmatched q
        let region_start = find_region_start(data, tp);

        // Find region end: for BT, find matching ET; for Do, end after "Do"
        let region_end = if data[tp] == b'B' {
            // Find matching ET
            find_matching_et(data, tp + 2).unwrap_or(len)
        } else {
            // Do operator: include operands before and the operator itself
            tp + 2
        };

        let end = region_end.min(len);
        regions.push((region_start, end));
    }

    // Merge overlapping/adjacent regions
    if regions.is_empty() {
        return Some(Vec::new());
    }
    regions.sort_unstable_by_key(|r| r.0);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for r in regions {
        if let Some(last) = merged.last_mut() {
            if r.0 <= last.1 {
                last.1 = last.1.max(r.1);
                continue;
            }
        }
        merged.push(r);
    }

    Some(merged)
}

/// Scan backwards from `pos` to find the start of the graphics state context.
/// Looks for an unmatched 'q' operator, handling nesting.
fn find_region_start(data: &[u8], pos: usize) -> usize {
    // Simple backward scan: find the nearest line that starts with 'q' or
    // the beginning of data. We limit backward scan to 4KB for performance.
    let scan_start = pos.saturating_sub(4096);
    let region = &data[scan_start..pos];

    // Find the last unmatched q by tracking Q/q balance backwards
    let mut q_depth: i32 = 0;
    let mut best_q_pos = pos; // Default: start from text position itself
    let mut i = region.len();

    while i > 0 {
        i -= 1;
        let b = region[i];

        // Look for 'q' or 'Q' at operator boundaries
        if b == b'q' || b == b'Q' {
            let abs_pos = scan_start + i;
            // Verify it's a standalone operator (boundary check)
            let before_ok = i == 0 || {
                let prev = region[i - 1];
                prev.is_ascii_whitespace() || matches!(prev, b')' | b'>' | b']')
            };
            let after_ok = i + 1 >= region.len() || {
                let next = region[i + 1];
                next.is_ascii_whitespace()
                    || matches!(next, b'(' | b'<' | b'[' | b'/' | b'%')
                    || next.is_ascii_digit()
                    || next == b'-'
                    || next == b'.'
            };

            if before_ok && after_ok {
                if b == b'Q' {
                    q_depth += 1;
                } else {
                    // 'q'
                    if q_depth > 0 {
                        q_depth -= 1;
                    } else {
                        // Unmatched q — this is our region start
                        best_q_pos = abs_pos;
                        break;
                    }
                }
            }
        }
    }

    best_q_pos
}

/// Find the position after matching "ET" for a BT starting at `start`.
fn find_matching_et(data: &[u8], start: usize) -> Option<usize> {
    let mut offset = start;
    let len = data.len();
    // Use memchr to find 'E' candidates
    loop {
        match memchr::memchr(b'E', &data[offset..]) {
            None => return None,
            Some(rel) => {
                let pos = offset + rel;
                offset = pos + 1;
                if pos + 1 < len && data[pos + 1] == b'T' {
                    let before_ok = pos == 0
                        || data[pos - 1].is_ascii_whitespace()
                        || matches!(data[pos - 1], b')' | b'>' | b']' | b'}' | b'/' | b'%');
                    let after_ok = pos + 2 >= len || {
                        let next = data[pos + 2];
                        next.is_ascii_whitespace()
                            || matches!(next, b'(' | b'<' | b'[' | b'/' | b'%')
                    };
                    if before_ok && after_ok {
                        return Some(pos + 2);
                    }
                }
            },
        }
    }
}

/// Streaming text-only parser: parse operators and call handler immediately.
///
/// Same logic as `parse_content_stream_text_only` but avoids allocating a Vec<Operator>.
/// Each operator is passed to `handler` as soon as it's parsed, improving cache locality
/// and eliminating the intermediate operator vector (which can be 16MB+ for graphics-heavy pages).
pub fn parse_and_execute_text_only<F>(data: &[u8], mut handler: F) -> Result<()>
where
    F: FnMut(Operator) -> Result<()>,
{
    // For large streams (>256KB), use SIMD pre-scan to identify text regions.
    // This avoids byte-by-byte scanning of megabytes of path/color operators.
    if data.len() > 256 * 1024 {
        if let Some(regions) = prescan_text_regions(data) {
            if regions.is_empty() {
                return Ok(()); // No text operators in stream
            }
            // Parse only the identified text-bearing regions
            for (start, end) in &regions {
                let region_data = &data[*start..*end];
                parse_region_text_only(region_data, &mut handler)?;
            }
            return Ok(());
        }
        // Fallback: pre-scan inconclusive, use full scan below
    }

    let mut input = data;
    let mut consecutive_errors: usize = 0;
    let mut inside_text = false;
    let mut op_count: usize = 0;

    while !input.is_empty() {
        // Skip leading whitespace (inline — both fast parser and scan_graphics
        // also handle whitespace, but this covers the initial entry and error
        // recovery paths without nom overhead).
        while !input.is_empty() && input[0].is_ascii_whitespace() {
            input = &input[1..];
        }
        if input.is_empty() {
            break;
        }

        if op_count >= MAX_OPERATORS {
            log::warn!("Content stream exceeded {} operators, truncating", MAX_OPERATORS);
            break;
        }

        if inside_text {
            // Try fast path first (3-5x faster for common text operators)
            if let Some((rest, op)) = parse_text_operator_fast(input) {
                if matches!(op, Operator::EndText) {
                    inside_text = false;
                }
                handler(op)?;
                op_count += 1;
                input = rest;
                consecutive_errors = 0;
            } else {
                // Fall back to generic nom-based parser
                match parse_operator_with_operands(input) {
                    Ok((rest, op)) => {
                        if matches!(op, Operator::EndText) {
                            inside_text = false;
                        }
                        handler(op)?;
                        op_count += 1;
                        input = rest;
                        consecutive_errors = 0;
                    },
                    Err(_) => {
                        consecutive_errors += 1;
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                            log::warn!(
                                "Content stream had {} consecutive parse errors, bailing out ({} bytes remaining)",
                                MAX_CONSECUTIVE_ERRORS,
                                input.len()
                            );
                            break;
                        }
                        if input.len() > 1 {
                            input = &input[1..];
                        } else {
                            break;
                        }
                    },
                }
            }
        } else {
            match scan_graphics_region(input, &mut consecutive_errors) {
                ScanResult::EndOfData => break,
                ScanResult::FoundBT { rest } => {
                    handler(Operator::BeginText)?;
                    op_count += 1;
                    input = rest;
                    inside_text = true;
                },
                ScanResult::InlineImage { rest } => match parse_inline_image(rest) {
                    Ok((rest2, _)) => input = rest2,
                    Err(_) => input = rest,
                },
                ScanResult::NeedFullParse {
                    operand_start,
                    after_op,
                } => match parse_operator_with_operands(operand_start) {
                    Ok((rest2, op)) => {
                        handler(op)?;
                        op_count += 1;
                        input = rest2;
                    },
                    Err(_) => input = after_op,
                },
                ScanResult::DeferredThenText {
                    deferred_start,
                    trigger_start,
                } => {
                    let mut remaining = deferred_start;
                    while remaining.len() > trigger_start.len() {
                        match parse_operator_with_operands(remaining) {
                            Ok((rest2, op)) => {
                                handler(op)?;
                                op_count += 1;
                                remaining = rest2;
                            },
                            Err(_) => {
                                if remaining.len() > 1 {
                                    remaining = &remaining[1..];
                                } else {
                                    break;
                                }
                            },
                        }
                    }
                    input = trigger_start;
                    consecutive_errors = 0;
                },
                ScanResult::SimpleOp { op, rest } => {
                    handler(op)?;
                    op_count += 1;
                    input = rest;
                },
                ScanResult::TooManyErrors { remaining } => {
                    log::warn!(
                        "Content stream had {} consecutive parse errors, bailing out ({} bytes remaining)",
                        MAX_CONSECUTIVE_ERRORS,
                        remaining.len()
                    );
                    break;
                },
            }
        }
    }

    Ok(())
}

/// Parse a sub-region of a content stream for text operators.
/// Used by the pre-scan path to parse only identified text-bearing regions.
fn parse_region_text_only<F>(data: &[u8], handler: &mut F) -> Result<()>
where
    F: FnMut(Operator) -> Result<()>,
{
    let mut input = data;
    let mut consecutive_errors: usize = 0;
    let mut inside_text = false;
    let mut op_count: usize = 0;

    while !input.is_empty() {
        while !input.is_empty() && input[0].is_ascii_whitespace() {
            input = &input[1..];
        }
        if input.is_empty() {
            break;
        }

        if op_count >= MAX_OPERATORS {
            break;
        }

        if inside_text {
            if let Some((rest, op)) = parse_text_operator_fast(input) {
                if matches!(op, Operator::EndText) {
                    inside_text = false;
                }
                handler(op)?;
                op_count += 1;
                input = rest;
                consecutive_errors = 0;
            } else {
                match parse_operator_with_operands(input) {
                    Ok((rest, op)) => {
                        if matches!(op, Operator::EndText) {
                            inside_text = false;
                        }
                        handler(op)?;
                        op_count += 1;
                        input = rest;
                        consecutive_errors = 0;
                    },
                    Err(_) => {
                        consecutive_errors += 1;
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                            break;
                        }
                        if input.len() > 1 {
                            input = &input[1..];
                        } else {
                            break;
                        }
                    },
                }
            }
        } else {
            match scan_graphics_region(input, &mut consecutive_errors) {
                ScanResult::EndOfData => break,
                ScanResult::FoundBT { rest } => {
                    handler(Operator::BeginText)?;
                    op_count += 1;
                    input = rest;
                    inside_text = true;
                },
                ScanResult::InlineImage { rest } => match parse_inline_image(rest) {
                    Ok((rest2, _)) => input = rest2,
                    Err(_) => input = rest,
                },
                ScanResult::NeedFullParse {
                    operand_start,
                    after_op,
                } => match parse_operator_with_operands(operand_start) {
                    Ok((rest2, op)) => {
                        handler(op)?;
                        op_count += 1;
                        input = rest2;
                    },
                    Err(_) => input = after_op,
                },
                ScanResult::DeferredThenText {
                    deferred_start,
                    trigger_start,
                } => {
                    let mut remaining = deferred_start;
                    while remaining.len() > trigger_start.len() {
                        match parse_operator_with_operands(remaining) {
                            Ok((rest2, op)) => {
                                handler(op)?;
                                op_count += 1;
                                remaining = rest2;
                            },
                            Err(_) => {
                                if remaining.len() > 1 {
                                    remaining = &remaining[1..];
                                } else {
                                    break;
                                }
                            },
                        }
                    }
                    input = trigger_start;
                    consecutive_errors = 0;
                },
                ScanResult::SimpleOp { op, rest } => {
                    handler(op)?;
                    op_count += 1;
                    input = rest;
                },
                ScanResult::TooManyErrors { .. } => break,
            }
        }
    }

    Ok(())
}

/// Image-only content stream parser: skips BT/ET text blocks entirely.
///
/// Only fully parses operators relevant to image extraction:
/// `cm`, `q`, `Q`, `Do`, `BI`/`ID`/`EI` (inline images).
/// All text and graphics drawing operators are skipped.
pub fn parse_content_stream_images_only(data: &[u8]) -> Result<Vec<Operator>> {
    let mut operators = Vec::with_capacity(256);
    let mut input = data;
    let mut consecutive_errors: usize = 0;
    let mut inside_text = false;

    while !input.is_empty() {
        if let Ok((rest, _)) = multispace0::<&[u8], nom::error::Error<&[u8]>>.parse(input) {
            input = rest;
        }
        if input.is_empty() {
            break;
        }

        if operators.len() >= MAX_OPERATORS {
            break;
        }

        if inside_text {
            // Inside BT/ET: skip everything until ET
            match scan_to_et(input) {
                Some(rest) => {
                    input = rest;
                    inside_text = false;
                    consecutive_errors = 0;
                },
                None => break, // No ET found, end of stream
            }
        } else {
            // Outside BT/ET: use scan_graphics_region but handle differently
            match scan_graphics_region(input, &mut consecutive_errors) {
                ScanResult::EndOfData => break,
                ScanResult::FoundBT { rest } => {
                    // Skip the text block instead of parsing it
                    input = rest;
                    inside_text = true;
                },
                ScanResult::InlineImage { rest } => match parse_inline_image(rest) {
                    Ok((rest2, op)) => {
                        operators.push(op);
                        input = rest2;
                    },
                    Err(_) => input = rest,
                },
                ScanResult::NeedFullParse {
                    operand_start,
                    after_op,
                } => match parse_operator_with_operands(operand_start) {
                    Ok((rest2, op)) => {
                        operators.push(op);
                        input = rest2;
                    },
                    Err(_) => input = after_op,
                },
                ScanResult::DeferredThenText {
                    deferred_start,
                    trigger_start,
                } => {
                    let mut remaining = deferred_start;
                    while remaining.len() > trigger_start.len() {
                        match parse_operator_with_operands(remaining) {
                            Ok((rest2, op)) => {
                                operators.push(op);
                                remaining = rest2;
                            },
                            Err(_) => {
                                if remaining.len() > 1 {
                                    remaining = &remaining[1..];
                                } else {
                                    break;
                                }
                            },
                        }
                    }
                    input = trigger_start;
                    consecutive_errors = 0;
                },
                ScanResult::SimpleOp { op, rest } => {
                    operators.push(op);
                    input = rest;
                },
                ScanResult::TooManyErrors { .. } => break,
            }
        }
    }

    Ok(operators)
}

/// Skip forward until we find the ET operator (end text).
/// Returns the remaining input after ET, or None if not found.
fn scan_to_et(data: &[u8]) -> Option<&[u8]> {
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == b'E' && data[i + 1] == b'T' {
            // Verify it's a real ET operator (not part of a string)
            let before_ok = i == 0
                || data[i - 1].is_ascii_whitespace()
                || data[i - 1] == b')'
                || data[i - 1] == b'>';
            let after_ok =
                i + 2 >= data.len() || data[i + 2].is_ascii_whitespace() || data[i + 2] == b'%';
            if before_ok && after_ok {
                return Some(&data[i + 2..]);
            }
        }
        // Skip strings to avoid false matches inside text
        if data[i] == b'(' {
            i += 1;
            let mut depth = 1;
            while i < data.len() && depth > 0 {
                match data[i] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    b'\\' => i += 1, // skip escaped char
                    _ => {},
                }
                i += 1;
            }
            continue;
        }
        if data[i] == b'<' && (i + 1 >= data.len() || data[i + 1] != b'<') {
            i += 1;
            while i < data.len() && data[i] != b'>' {
                i += 1;
            }
            if i < data.len() {
                i += 1;
            }
            continue;
        }
        i += 1;
    }
    None
}

/// Parse a single operator with its operands.
///
/// Returns the remaining input and the parsed operator.
///
/// Uses `SmallVec<[Object; 6]>` for the operand buffer to avoid heap
/// allocation for the common case (most PDF operators have 0-6 operands).
/// Only spills to the heap for rare operators with more than 6 operands.
fn parse_operator_with_operands(input: &[u8]) -> IResult<&[u8], Operator> {
    // Collect operands until we hit an operator name.
    // SmallVec<[Object; 6]>: stack-allocated for <= 6 operands (covers all
    // standard PDF operators: cm/Tm need 6, most need 0-4). Only spills to
    // heap for pathological content (e.g., deeply nested arrays in Other).
    let mut operands: SmallVec<[Object; 6]> = SmallVec::new();
    let mut remaining = input;

    loop {
        // Skip whitespace
        let (inp, _) = multispace0.parse(remaining)?;
        remaining = inp;

        if remaining.is_empty() {
            return Err(nom::Err::Error(nom::error::Error::new(
                remaining,
                nom::error::ErrorKind::Eof,
            )));
        }

        // Check if this looks like an operator name (alphabetic characters)
        // Operators are typically 1-3 letter keywords
        if is_operator_start(remaining[0]) {
            let (rest, op_name) = parse_operator_name(remaining)?;

            // Special handling for inline images (BI...ID...EI sequence)
            if op_name == "BI" {
                // Parse inline image: BI <dict entries> ID <binary data> EI
                return parse_inline_image(rest);
            }

            let op = build_operator(op_name, operands);
            return Ok((rest, op));
        }

        // Otherwise, try to parse an operand (PDF object)
        let (inp, obj) = parse_object(remaining)?;
        operands.push(obj);
        remaining = inp;
    }
}

/// Check if a byte could start an operator name.
///
/// Operators start with alphabetic characters or special characters like ' or "
fn is_operator_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'\'' || byte == b'"' || byte == b'*'
}

/// Parse an operator name from the input.
///
/// Operator names are typically 1-3 letter alphabetic sequences, but can include:
/// - Single quote (') for the Quote operator
/// - Double quote (") for the DoubleQuote operator
/// - Star (*) for T* operator
fn parse_operator_name(input: &[u8]) -> IResult<&[u8], &str> {
    let (input, name_bytes) =
        take_while1(|c: u8| c.is_ascii_alphanumeric() || c == b'\'' || c == b'"' || c == b'*')
            .parse(input)?;

    let name = std::str::from_utf8(name_bytes)
        .map_err(|_| nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Char)))?;

    Ok((input, name))
}

/// Build an operator from its name and operands.
///
/// This function converts the raw operator name and operands into a strongly-typed
/// Operator enum variant. It handles type conversions and validates operand counts.
///
/// Accepts `SmallVec<[Object; 6]>` to avoid heap allocation for the common case
/// (most PDF operators have 0-6 operands). The operands are consumed and dropped
/// after extraction.
fn build_operator(name: &str, operands: SmallVec<[Object; 6]>) -> Operator {
    match name {
        // Text positioning
        "Td" => {
            let tx = get_number(&operands, 0).unwrap_or(0.0);
            let ty = get_number(&operands, 1).unwrap_or(0.0);
            Operator::Td { tx, ty }
        },
        "TD" => {
            let tx = get_number(&operands, 0).unwrap_or(0.0);
            let ty = get_number(&operands, 1).unwrap_or(0.0);
            Operator::TD { tx, ty }
        },
        "Tm" => {
            let a = get_number(&operands, 0).unwrap_or(1.0);
            let b = get_number(&operands, 1).unwrap_or(0.0);
            let c = get_number(&operands, 2).unwrap_or(0.0);
            let d = get_number(&operands, 3).unwrap_or(1.0);
            let e = get_number(&operands, 4).unwrap_or(0.0);
            let f = get_number(&operands, 5).unwrap_or(0.0);
            Operator::Tm { a, b, c, d, e, f }
        },
        "T*" => Operator::TStar,

        // Text showing
        "Tj" => {
            let text = get_string(&operands, 0).unwrap_or_default();
            Operator::Tj { text }
        },
        "TJ" => {
            let elements = if let Some(array) = get_array(&operands, 0) {
                array
                    .iter()
                    .filter_map(|obj| match obj {
                        Object::String(s) => Some(TextElement::String(s.clone())),
                        Object::Integer(i) => Some(TextElement::Offset(*i as f32)),
                        Object::Real(r) => Some(TextElement::Offset(*r as f32)),
                        _ => None,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            Operator::TJ { array: elements }
        },
        "'" => {
            let text = get_string(&operands, 0).unwrap_or_default();
            Operator::Quote { text }
        },
        "\"" => {
            let word_space = get_number(&operands, 0).unwrap_or(0.0);
            let char_space = get_number(&operands, 1).unwrap_or(0.0);
            let text = get_string(&operands, 2).unwrap_or_default();
            Operator::DoubleQuote {
                word_space,
                char_space,
                text,
            }
        },

        // Text state
        "Tc" => {
            let char_space = get_number(&operands, 0).unwrap_or(0.0);
            Operator::Tc { char_space }
        },
        "Tw" => {
            let word_space = get_number(&operands, 0).unwrap_or(0.0);
            Operator::Tw { word_space }
        },
        "Tz" => {
            let scale = get_number(&operands, 0).unwrap_or(100.0);
            Operator::Tz { scale }
        },
        "TL" => {
            let leading = get_number(&operands, 0).unwrap_or(0.0);
            Operator::TL { leading }
        },
        "Tf" => {
            let font = get_name(&operands, 0).unwrap_or("").to_string();
            let size = get_number(&operands, 1).unwrap_or(12.0);
            Operator::Tf { font, size }
        },
        "Tr" => {
            let render = get_integer(&operands, 0).unwrap_or(0) as u8;
            Operator::Tr { render }
        },
        "Ts" => {
            let rise = get_number(&operands, 0).unwrap_or(0.0);
            Operator::Ts { rise }
        },

        // Graphics state
        "q" => Operator::SaveState,
        "Q" => Operator::RestoreState,
        "cm" => {
            let a = get_number(&operands, 0).unwrap_or(1.0);
            let b = get_number(&operands, 1).unwrap_or(0.0);
            let c = get_number(&operands, 2).unwrap_or(0.0);
            let d = get_number(&operands, 3).unwrap_or(1.0);
            let e = get_number(&operands, 4).unwrap_or(0.0);
            let f = get_number(&operands, 5).unwrap_or(0.0);
            Operator::Cm { a, b, c, d, e, f }
        },

        // Color
        "rg" => {
            let r = get_number(&operands, 0).unwrap_or(0.0);
            let g = get_number(&operands, 1).unwrap_or(0.0);
            let b = get_number(&operands, 2).unwrap_or(0.0);
            Operator::SetFillRgb { r, g, b }
        },
        "RG" => {
            let r = get_number(&operands, 0).unwrap_or(0.0);
            let g = get_number(&operands, 1).unwrap_or(0.0);
            let b = get_number(&operands, 2).unwrap_or(0.0);
            Operator::SetStrokeRgb { r, g, b }
        },
        "g" => {
            let gray = get_number(&operands, 0).unwrap_or(0.0);
            Operator::SetFillGray { gray }
        },
        "G" => {
            let gray = get_number(&operands, 0).unwrap_or(0.0);
            Operator::SetStrokeGray { gray }
        },
        "k" => {
            // Set CMYK fill color
            let c = get_number(&operands, 0).unwrap_or(0.0);
            let m = get_number(&operands, 1).unwrap_or(0.0);
            let y = get_number(&operands, 2).unwrap_or(0.0);
            let k = get_number(&operands, 3).unwrap_or(0.0);
            Operator::SetFillCmyk { c, m, y, k }
        },
        "K" => {
            // Set CMYK stroke color
            let c = get_number(&operands, 0).unwrap_or(0.0);
            let m = get_number(&operands, 1).unwrap_or(0.0);
            let y = get_number(&operands, 2).unwrap_or(0.0);
            let k = get_number(&operands, 3).unwrap_or(0.0);
            Operator::SetStrokeCmyk { c, m, y, k }
        },

        // Color space operators
        "cs" => {
            // Set fill color space: name cs
            let name = get_name(&operands, 0).unwrap_or("DeviceGray").to_string();
            Operator::SetFillColorSpace { name }
        },
        "CS" => {
            // Set stroke color space: name CS
            let name = get_name(&operands, 0).unwrap_or("DeviceGray").to_string();
            Operator::SetStrokeColorSpace { name }
        },
        "sc" => {
            // Set fill color: c1 c2 ... cn sc
            // Number of components depends on current color space
            let components: Vec<f32> = operands
                .iter()
                .filter_map(|obj| match obj {
                    Object::Real(r) => Some(*r as f32),
                    Object::Integer(i) => Some(*i as f32),
                    _ => None,
                })
                .collect();
            Operator::SetFillColor { components }
        },
        "SC" => {
            // Set stroke color: c1 c2 ... cn SC
            let components: Vec<f32> = operands
                .iter()
                .filter_map(|obj| match obj {
                    Object::Real(r) => Some(*r as f32),
                    Object::Integer(i) => Some(*i as f32),
                    _ => None,
                })
                .collect();
            Operator::SetStrokeColor { components }
        },
        "scn" => {
            // Set fill color with pattern support: c1 c2 ... cn [name] scn
            // Last operand may be a name for pattern color spaces
            let name = if let Some(Object::Name(n)) = operands.last() {
                Some(n.clone())
            } else {
                None
            };
            let components: Vec<f32> = operands
                .iter()
                .filter_map(|obj| match obj {
                    Object::Real(r) => Some(*r as f32),
                    Object::Integer(i) => Some(*i as f32),
                    Object::Name(_) => None, // Skip pattern name
                    _ => None,
                })
                .collect();
            Operator::SetFillColorN {
                components,
                name: name.map(Box::new),
            }
        },
        "SCN" => {
            // Set stroke color with pattern support: c1 c2 ... cn [name] SCN
            let name = if let Some(Object::Name(n)) = operands.last() {
                Some(n.clone())
            } else {
                None
            };
            let components: Vec<f32> = operands
                .iter()
                .filter_map(|obj| match obj {
                    Object::Real(r) => Some(*r as f32),
                    Object::Integer(i) => Some(*i as f32),
                    Object::Name(_) => None, // Skip pattern name
                    _ => None,
                })
                .collect();
            Operator::SetStrokeColorN {
                components,
                name: name.map(Box::new),
            }
        },

        // Text object
        "BT" => Operator::BeginText,
        "ET" => Operator::EndText,

        // XObject
        "Do" => {
            let name = get_name(&operands, 0).unwrap_or("").to_string();
            Operator::Do { name }
        },

        // Path construction
        "m" => {
            let x = get_number(&operands, 0).unwrap_or(0.0);
            let y = get_number(&operands, 1).unwrap_or(0.0);
            Operator::MoveTo { x, y }
        },
        "l" => {
            let x = get_number(&operands, 0).unwrap_or(0.0);
            let y = get_number(&operands, 1).unwrap_or(0.0);
            Operator::LineTo { x, y }
        },
        "c" => {
            // Cubic Bézier curve
            let x1 = get_number(&operands, 0).unwrap_or(0.0);
            let y1 = get_number(&operands, 1).unwrap_or(0.0);
            let x2 = get_number(&operands, 2).unwrap_or(0.0);
            let y2 = get_number(&operands, 3).unwrap_or(0.0);
            let x3 = get_number(&operands, 4).unwrap_or(0.0);
            let y3 = get_number(&operands, 5).unwrap_or(0.0);
            Operator::CurveTo {
                x1,
                y1,
                x2,
                y2,
                x3,
                y3,
            }
        },
        "v" => {
            // Bézier curve (first control point = current point)
            let x2 = get_number(&operands, 0).unwrap_or(0.0);
            let y2 = get_number(&operands, 1).unwrap_or(0.0);
            let x3 = get_number(&operands, 2).unwrap_or(0.0);
            let y3 = get_number(&operands, 3).unwrap_or(0.0);
            Operator::CurveToV { x2, y2, x3, y3 }
        },
        "y" => {
            // Bézier curve (second control point = end point)
            let x1 = get_number(&operands, 0).unwrap_or(0.0);
            let y1 = get_number(&operands, 1).unwrap_or(0.0);
            let x3 = get_number(&operands, 2).unwrap_or(0.0);
            let y3 = get_number(&operands, 3).unwrap_or(0.0);
            Operator::CurveToY { x1, y1, x3, y3 }
        },
        "h" => Operator::ClosePath,
        "re" => {
            let x = get_number(&operands, 0).unwrap_or(0.0);
            let y = get_number(&operands, 1).unwrap_or(0.0);
            let width = get_number(&operands, 2).unwrap_or(0.0);
            let height = get_number(&operands, 3).unwrap_or(0.0);
            Operator::Rectangle {
                x,
                y,
                width,
                height,
            }
        },
        "S" => Operator::Stroke,
        "f" | "F" => Operator::Fill, // "F" is obsolete equivalent of "f" (nonzero winding fill)
        "f*" => Operator::FillEvenOdd,
        "b" => Operator::CloseFillStroke,
        "b*" => Operator::CloseFillStrokeEvenOdd,
        "B" => Operator::FillStroke,
        "B*" => Operator::FillStrokeEvenOdd,
        "n" => Operator::EndPath,
        "W" => Operator::ClipNonZero,
        "W*" => Operator::ClipEvenOdd,

        // Graphics state operators
        "w" => {
            let width = get_number(&operands, 0).unwrap_or(1.0);
            Operator::SetLineWidth { width }
        },
        "d" => {
            // d operator: array phase
            // Example: [3 2] 0 d means 3 on, 2 off, starting at phase 0
            let array = if let Some(Object::Array(arr)) = operands.first() {
                arr.iter()
                    .filter_map(|obj| match obj {
                        Object::Integer(i) => Some(*i as f32),
                        Object::Real(r) => Some(*r as f32),
                        _ => None,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let phase = get_number(&operands, 1).unwrap_or(0.0);
            Operator::SetDash { array, phase }
        },
        "J" => {
            // J operator: integer J
            // 0=butt cap, 1=round cap, 2=projecting square cap
            let cap_style = get_integer(&operands, 0).unwrap_or(0) as u8;
            Operator::SetLineCap { cap_style }
        },
        "j" => {
            // j operator: integer j
            // 0=miter join, 1=round join, 2=bevel join
            let join_style = get_integer(&operands, 0).unwrap_or(0) as u8;
            Operator::SetLineJoin { join_style }
        },
        "M" => {
            // M operator: number M
            // Miter limit (ratio of miter length to line width)
            let limit = get_number(&operands, 0).unwrap_or(10.0);
            Operator::SetMiterLimit { limit }
        },
        "ri" => {
            // ri operator: name ri
            // Rendering intent: /AbsoluteColorimetric, /RelativeColorimetric, /Saturation, or /Perceptual
            let intent = get_name(&operands, 0)
                .unwrap_or("RelativeColorimetric")
                .to_string();
            Operator::SetRenderingIntent { intent }
        },
        "i" => {
            // i operator: number i
            // Flatness tolerance (0-100)
            let tolerance = get_number(&operands, 0).unwrap_or(1.0);
            Operator::SetFlatness { tolerance }
        },
        "gs" => {
            // gs operator: name gs
            // Set extended graphics state from resource dictionary
            let dict_name = get_name(&operands, 0).unwrap_or("").to_string();
            Operator::SetExtGState { dict_name }
        },
        "sh" => {
            // sh operator: name sh
            // Paint shading pattern (gradient)
            let name = get_name(&operands, 0).unwrap_or("").to_string();
            Operator::PaintShading { name }
        },

        // Marked content operators (for tagged PDF structure)
        // PDF Spec: ISO 32000-1:2008, Section 14.6
        "BMC" => {
            // Begin marked content: tag BMC
            let tag = get_name(&operands, 0).unwrap_or("").to_string();
            Operator::BeginMarkedContent { tag }
        },
        "BDC" => {
            // Begin marked content with properties: tag properties BDC
            // properties can be a dictionary or a name (reference to /Properties resource)
            let tag = get_name(&operands, 0).unwrap_or("").to_string();
            let properties = Box::new(operands.get(1).cloned().unwrap_or(Object::Null));
            Operator::BeginMarkedContentDict { tag, properties }
        },
        "EMC" => {
            // End marked content: EMC (no operands)
            Operator::EndMarkedContent
        },

        // Unknown operator — convert SmallVec to Vec for the boxed storage.
        // This path is rare (only for unrecognized operators), so the
        // conversion cost is negligible.
        _ => Operator::Other {
            name: name.to_string(),
            operands: Box::new(operands.into_vec()),
        },
    }
}

// Helper functions to extract operands

fn get_number(operands: &[Object], index: usize) -> Option<f32> {
    operands.get(index).and_then(|obj| match obj {
        Object::Integer(i) => Some(*i as f32),
        Object::Real(r) => Some(*r as f32),
        _ => None,
    })
}

fn get_integer(operands: &[Object], index: usize) -> Option<i64> {
    operands.get(index).and_then(|obj| obj.as_integer())
}

fn get_string(operands: &[Object], index: usize) -> Option<Vec<u8>> {
    operands
        .get(index)
        .and_then(|obj| obj.as_string().map(|s| s.to_vec()))
}

fn get_name(operands: &[Object], index: usize) -> Option<&str> {
    operands.get(index).and_then(|obj| obj.as_name())
}

fn get_array(operands: &[Object], index: usize) -> Option<&Vec<Object>> {
    operands.get(index).and_then(|obj| obj.as_array())
}

/// Parse an inline image sequence (BI...ID...EI).
///
/// PDF Spec: ISO 32000-1:2008, Section 8.9.7 - Inline Images
///
/// Inline images have the format:
/// BI <key value> <key value> ... ID <binary data> EI
///
/// The dictionary uses abbreviated keys:
/// - W: Width
/// - H: Height
/// - CS: ColorSpace
/// - BPC: BitsPerComponent
/// - F: Filter
/// - DP: DecodeParms
/// - I: Interpolate
///
/// The challenge is finding the EI operator in the binary data, as the bytes
/// for "EI" could appear in the image data itself. Per spec, EI must be:
/// - Preceded by whitespace (space, tab, CR, LF)
/// - Followed by whitespace or end of stream
fn parse_inline_image(input: &[u8]) -> IResult<&[u8], Operator> {
    let mut dict = HashMap::new();
    let mut remaining = input;

    // Step 1: Parse the inline image dictionary (key-value pairs)
    loop {
        // Skip whitespace
        let (inp, _) = multispace0.parse(remaining)?;
        remaining = inp;

        if remaining.is_empty() {
            return Err(nom::Err::Error(nom::error::Error::new(
                remaining,
                nom::error::ErrorKind::Eof,
            )));
        }

        // Check if we've reached "ID" (start of image data)
        if remaining.len() >= 2 && &remaining[0..2] == b"ID" {
            // Check that ID is followed by whitespace or is at end
            if remaining.len() == 2 || remaining.len() > 2 && is_whitespace(remaining[2]) {
                remaining = &remaining[2..];
                break;
            }
        }

        // Parse a key (name object, often abbreviated)
        let (inp, key_obj) = parse_object(remaining)?;
        remaining = inp;

        // Skip whitespace after key
        let (inp, _) = multispace0.parse(remaining)?;
        remaining = inp;

        // Parse the corresponding value
        let (inp, value_obj) = parse_object(remaining)?;
        remaining = inp;

        // Add to dictionary
        if let Some(key_str) = key_obj.as_name() {
            dict.insert(key_str.to_string(), value_obj);
        }
    }

    // Step 2: Skip whitespace after ID
    let (inp, _) = multispace0.parse(remaining)?;
    remaining = inp;

    // Step 3: Read binary image data until we find EI
    // EI must be preceded and followed by whitespace
    let (_inp, data) = find_and_extract_image_data(remaining)?;
    let data_len = data.len();
    remaining = &remaining[data_len..];

    // Step 4: Skip past the EI operator
    // Find EI preceded by whitespace and skip it
    let (_inp, ei_pos) = find_ei_operator(remaining)?;
    remaining = &remaining[ei_pos + 2..]; // Skip past whitespace and "EI"

    // Step 5: Return the InlineImage operator
    Ok((
        remaining,
        Operator::InlineImage {
            dict: Box::new(dict),
            data,
        },
    ))
}

/// Find the EI operator in the input, which must be preceded by whitespace.
/// Returns the position of the whitespace before EI.
fn find_ei_operator(input: &[u8]) -> IResult<&[u8], usize> {
    for i in 0..input.len().saturating_sub(2) {
        // Check if we have whitespace followed by "EI"
        if is_whitespace(input[i]) && input.len() > i + 2 && &input[i + 1..i + 3] == b"EI" {
            // Check that EI is followed by whitespace, end of stream, or another operator
            if input.len() == i + 3 || is_whitespace_or_delimiter(input[i + 3]) {
                return Ok((input, i));
            }
        }
    }

    Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)))
}

/// Extract image data up to (but not including) the whitespace before EI.
fn find_and_extract_image_data(input: &[u8]) -> IResult<&[u8], Vec<u8>> {
    let (inp, ei_pos) = find_ei_operator(input)?;
    Ok((inp, input[..ei_pos].to_vec()))
}

/// Check if a byte is whitespace (space, tab, CR, LF, FF).
fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\n' | b'\x0C')
}

/// Check if a byte is whitespace or a PDF delimiter.
fn is_whitespace_or_delimiter(byte: u8) -> bool {
    is_whitespace(byte)
        || matches!(byte, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}

// ── Nom-based operand skippers (test-only, superseded by raw variants) ─────

#[cfg(test)]
fn skip_operand_token(input: &[u8]) -> IResult<&[u8], ()> {
    if input.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Eof)));
    }

    match input[0] {
        b'0'..=b'9' | b'.' | b'+' | b'-' => skip_number(input),
        b'(' => skip_literal_string(input),
        b'<' if input.len() > 1 && input[1] == b'<' => skip_dict(input),
        b'<' => skip_hex_string(input),
        b'/' => skip_name(input),
        b'[' => skip_array(input),
        _ => Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Char))),
    }
}

#[cfg(test)]
fn skip_number(input: &[u8]) -> IResult<&[u8], ()> {
    let mut i = 0;
    if i < input.len() && (input[i] == b'+' || input[i] == b'-') {
        i += 1;
    }
    let start = i;
    let mut has_dot = false;
    while i < input.len() {
        if input[i].is_ascii_digit() {
            i += 1;
        } else if input[i] == b'.' && !has_dot {
            has_dot = true;
            i += 1;
        } else {
            break;
        }
    }
    if i == start {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit)));
    }
    Ok((&input[i..], ()))
}

#[cfg(test)]
fn skip_literal_string(input: &[u8]) -> IResult<&[u8], ()> {
    let mut i = 1; // past opening '('
    let mut depth: u32 = 1;
    while i < input.len() && depth > 0 {
        match input[i] {
            b'\\' if i + 1 < input.len() => i += 2,
            b'(' => {
                depth += 1;
                i += 1;
            },
            b')' => {
                depth -= 1;
                i += 1;
            },
            _ => i += 1,
        }
    }
    if depth != 0 {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Char)));
    }
    Ok((&input[i..], ()))
}

#[cfg(test)]
fn skip_hex_string(input: &[u8]) -> IResult<&[u8], ()> {
    let mut i = 1; // past opening '<'
    while i < input.len() {
        if input[i] == b'>' {
            return Ok((&input[i + 1..], ()));
        }
        i += 1;
    }
    Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Char)))
}

#[cfg(test)]
fn skip_name(input: &[u8]) -> IResult<&[u8], ()> {
    let mut i = 1; // past '/'
    while i < input.len() && !is_whitespace_or_delimiter(input[i]) {
        i += 1;
    }
    Ok((&input[i..], ()))
}

#[cfg(test)]
fn skip_array(input: &[u8]) -> IResult<&[u8], ()> {
    let mut i = 1; // past opening '['
    let mut depth: u32 = 1;
    while i < input.len() && depth > 0 {
        match input[i] {
            b'[' => {
                depth += 1;
                i += 1;
            },
            b']' => {
                depth -= 1;
                i += 1;
            },
            b'(' => {
                // Skip nested literal string
                i += 1;
                let mut str_depth: u32 = 1;
                while i < input.len() && str_depth > 0 {
                    match input[i] {
                        b'\\' if i + 1 < input.len() => i += 2,
                        b'(' => {
                            str_depth += 1;
                            i += 1;
                        },
                        b')' => {
                            str_depth -= 1;
                            i += 1;
                        },
                        _ => i += 1,
                    }
                }
            },
            b'<' if i + 1 < input.len() && input[i + 1] == b'<' => {
                // Skip nested dict <<...>>
                i += 2;
                let mut dict_depth: u32 = 1;
                while i + 1 < input.len() && dict_depth > 0 {
                    if input[i] == b'<' && input[i + 1] == b'<' {
                        dict_depth += 1;
                        i += 2;
                    } else if input[i] == b'>' && input[i + 1] == b'>' {
                        dict_depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            },
            b'<' => {
                // Skip nested hex string
                i += 1;
                while i < input.len() && input[i] != b'>' {
                    i += 1;
                }
                if i < input.len() {
                    i += 1;
                }
            },
            _ => i += 1,
        }
    }
    if depth != 0 {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Char)));
    }
    Ok((&input[i..], ()))
}

#[cfg(test)]
fn skip_dict(input: &[u8]) -> IResult<&[u8], ()> {
    let mut i = 2; // past opening '<<'
    let mut depth: u32 = 1;
    while i < input.len() && depth > 0 {
        if i + 1 < input.len() && input[i] == b'<' && input[i + 1] == b'<' {
            depth += 1;
            i += 2;
        } else if i + 1 < input.len() && input[i] == b'>' && input[i + 1] == b'>' {
            depth -= 1;
            i += 2;
        } else if input[i] == b'(' {
            // Skip literal string inside dict
            i += 1;
            let mut str_depth: u32 = 1;
            while i < input.len() && str_depth > 0 {
                match input[i] {
                    b'\\' if i + 1 < input.len() => i += 2,
                    b'(' => {
                        str_depth += 1;
                        i += 1;
                    },
                    b')' => {
                        str_depth -= 1;
                        i += 1;
                    },
                    _ => i += 1,
                }
            }
        } else if input[i] == b'<' {
            // Single '<' → hex string <...>
            i += 1;
            while i < input.len() && input[i] != b'>' {
                i += 1;
            }
            if i < input.len() {
                i += 1; // Skip closing '>'
            }
        } else {
            i += 1;
        }
    }
    if depth != 0 {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Char)));
    }
    Ok((&input[i..], ()))
}

// ── Byte-level graphics region scanner ─────────────────────────────────────
//
// Replaces the nom-based operand loop in parse_content_stream_text_only with
// raw index arithmetic. >80% of bytes in graphics-heavy streams are digits,
// dots, and whitespace for path coordinates — a tight match loop processes
// these at near-memcpy speed vs per-operand nom IResult dispatch.

/// Result of scanning a graphics region (outside BT/ET).
enum ScanResult<'a> {
    /// All data consumed, no more operators.
    EndOfData,
    /// Found a BT operator; `rest` points past "BT".
    FoundBT { rest: &'a [u8] },
    /// Found an inline image (BI); `rest` points past "BI".
    InlineImage { rest: &'a [u8] },
    /// Found a non-skippable operator; caller should backtrack to
    /// `operand_start` for full parsing. `after_op` points past the operator
    /// name (used as fallback if full parse fails).
    NeedFullParse {
        operand_start: &'a [u8],
        after_op: &'a [u8],
    },
    /// Found a non-skippable operator (BT/BI/Do/etc.) inside a deferred q/cm
    /// block. `deferred_start` points to the first deferred `q` so the caller
    /// can full-parse the q/cm/Q sequence to preserve CTM. `trigger_start`
    /// points to the operand_start of the triggering operator so the caller
    /// resumes scanning there (the next scan_graphics_region call will
    /// immediately return the trigger via FoundBT / InlineImage / NeedFullParse).
    DeferredThenText {
        deferred_start: &'a [u8],
        trigger_start: &'a [u8],
    },
    /// A simple no-operand operator that can be emitted directly without
    /// nom parsing. Used for unmatched Q (RestoreGraphicsState) to avoid
    /// expensive full-parse fallback.
    SimpleOp { op: Operator, rest: &'a [u8] },
    /// Too many consecutive errors; remaining data is likely junk.
    TooManyErrors { remaining: &'a [u8] },
}

/// Parse 6 float operands from a raw byte slice (for inline `cm` parsing).
/// Returns None if the slice doesn't contain exactly 6 parseable numbers.
#[inline]
fn parse_six_floats(data: &[u8]) -> Option<(f32, f32, f32, f32, f32, f32)> {
    let s = std::str::from_utf8(data).ok()?;
    let mut iter = s.split_ascii_whitespace();
    let a = iter.next()?.parse::<f32>().ok()?;
    let b = iter.next()?.parse::<f32>().ok()?;
    let c = iter.next()?.parse::<f32>().ok()?;
    let d = iter.next()?.parse::<f32>().ok()?;
    let e = iter.next()?.parse::<f32>().ok()?;
    let f = iter.next()?.parse::<f32>().ok()?;
    Some((a, b, c, d, e, f))
}

/// Byte-level check for pure graphics operators that can be skipped during
/// text-only extraction. Equivalent to [`is_skippable_graphics_op`] but
/// operates on raw `&[u8]` without UTF-8 conversion.
fn is_skippable_graphics_op_bytes(op: &[u8]) -> bool {
    matches!(
        op,
        b"m" | b"l" | b"c" | b"v" | b"y" | b"h" | b"re"       // path construction
        | b"S" | b"s" | b"f" | b"F" | b"f*"                     // path painting
        | b"B" | b"B*" | b"b" | b"b*" | b"n"                    // path painting (fill+stroke)
        | b"W" | b"W*"                                           // clipping
        | b"w" | b"J" | b"j" | b"M" | b"d" | b"i" | b"ri" | b"sh" // non-text graphics state
        | b"rg" | b"RG" | b"g" | b"G" | b"k" | b"K"            // color (rgb/gray/cmyk)
        | b"cs" | b"CS" | b"sc" | b"SC" | b"scn" | b"SCN" // color space/components
    )
}

// ── Raw index-returning skip functions ─────────────────────────────────────
//
// Same logic as the nom-based skip_*() functions above, but return a new
// index position instead of IResult. On malformed input, Option variants
// return None so the caller can skip one byte (matching current error
// recovery).

fn skip_literal_string_raw(data: &[u8], mut i: usize) -> Option<usize> {
    i += 1; // past opening '('
    let mut depth: u32 = 1;
    while i < data.len() && depth > 0 {
        match data[i] {
            b'\\' if i + 1 < data.len() => i += 2,
            b'(' => {
                depth += 1;
                i += 1;
            },
            b')' => {
                depth -= 1;
                i += 1;
            },
            _ => i += 1,
        }
    }
    if depth == 0 {
        Some(i)
    } else {
        None
    }
}

fn skip_hex_string_raw(data: &[u8], mut i: usize) -> Option<usize> {
    i += 1; // past opening '<'
    while i < data.len() {
        if data[i] == b'>' {
            return Some(i + 1);
        }
        i += 1;
    }
    None
}

#[inline]
fn skip_name_raw(data: &[u8], mut i: usize) -> usize {
    i += 1; // past '/'
    while i < data.len() && !is_whitespace_or_delimiter(data[i]) {
        i += 1;
    }
    i
}

fn skip_array_raw(data: &[u8], i: usize) -> Option<usize> {
    let mut pos = i + 1; // past opening '['
    let mut depth: u32 = 1;
    while pos < data.len() && depth > 0 {
        match data[pos] {
            b'[' => {
                depth += 1;
                pos += 1;
            },
            b']' => {
                depth -= 1;
                pos += 1;
            },
            b'(' => {
                pos += 1;
                let mut str_depth: u32 = 1;
                while pos < data.len() && str_depth > 0 {
                    match data[pos] {
                        b'\\' if pos + 1 < data.len() => pos += 2,
                        b'(' => {
                            str_depth += 1;
                            pos += 1;
                        },
                        b')' => {
                            str_depth -= 1;
                            pos += 1;
                        },
                        _ => pos += 1,
                    }
                }
            },
            b'<' if pos + 1 < data.len() && data[pos + 1] == b'<' => {
                pos += 2;
                let mut dict_depth: u32 = 1;
                while pos + 1 < data.len() && dict_depth > 0 {
                    if data[pos] == b'<' && data[pos + 1] == b'<' {
                        dict_depth += 1;
                        pos += 2;
                    } else if data[pos] == b'>' && data[pos + 1] == b'>' {
                        dict_depth -= 1;
                        pos += 2;
                    } else {
                        pos += 1;
                    }
                }
            },
            b'<' => {
                pos += 1;
                while pos < data.len() && data[pos] != b'>' {
                    pos += 1;
                }
                if pos < data.len() {
                    pos += 1;
                }
            },
            _ => pos += 1,
        }
    }
    if depth == 0 {
        Some(pos)
    } else {
        None
    }
}

fn skip_dict_raw(data: &[u8], i: usize) -> Option<usize> {
    let mut pos = i + 2; // past opening '<<'
    let mut depth: u32 = 1;
    while pos < data.len() && depth > 0 {
        if pos + 1 < data.len() && data[pos] == b'<' && data[pos + 1] == b'<' {
            depth += 1;
            pos += 2;
        } else if pos + 1 < data.len() && data[pos] == b'>' && data[pos + 1] == b'>' {
            depth -= 1;
            pos += 2;
        } else if data[pos] == b'(' {
            pos += 1;
            let mut str_depth: u32 = 1;
            while pos < data.len() && str_depth > 0 {
                match data[pos] {
                    b'\\' if pos + 1 < data.len() => pos += 2,
                    b'(' => {
                        str_depth += 1;
                        pos += 1;
                    },
                    b')' => {
                        str_depth -= 1;
                        pos += 1;
                    },
                    _ => pos += 1,
                }
            }
        } else if data[pos] == b'<' {
            pos += 1;
            while pos < data.len() && data[pos] != b'>' {
                pos += 1;
            }
            if pos < data.len() {
                pos += 1;
            }
        } else {
            pos += 1;
        }
    }
    if depth == 0 {
        Some(pos)
    } else {
        None
    }
}

// ── Fast BT/ET block parser ────────────────────────────────────────────
//
// Hand-written byte-level parser for operators inside text blocks.
// Avoids the nom tokenizer overhead (~3-5x faster than parse_operator_with_operands)
// by parsing numbers inline, skipping indirect-reference lookahead, and matching
// operator names as raw bytes.

/// Operand type for the fast parser's operand stack.
/// Uses `f32` for numbers and `Vec<u8>` for strings to avoid full Object creation.
enum FastOperand {
    Number(f32),
    /// Raw string bytes (already decoded from literal or hex encoding)
    StringBytes(Vec<u8>),
    /// Name string (without leading `/`)
    Name(String),
    /// Array of TextElements (for TJ operator)
    TextArray(Vec<TextElement>),
}

/// Parse a float directly from bytes. Returns (value, bytes_consumed).
#[inline]
fn parse_float_fast(data: &[u8]) -> Option<(f32, usize)> {
    let mut i = 0;
    let negative = if i < data.len() && (data[i] == b'-' || data[i] == b'+') {
        let neg = data[i] == b'-';
        i += 1;
        neg
    } else {
        false
    };

    let start = i;
    let mut int_part: f64 = 0.0;
    while i < data.len() && data[i].is_ascii_digit() {
        int_part = int_part * 10.0 + (data[i] - b'0') as f64;
        i += 1;
    }

    let mut frac_part: f64 = 0.0;
    let mut frac_scale: f64 = 1.0;
    if i < data.len() && data[i] == b'.' {
        i += 1;
        while i < data.len() && data[i].is_ascii_digit() {
            frac_part = frac_part * 10.0 + (data[i] - b'0') as f64;
            frac_scale *= 10.0;
            i += 1;
        }
    }

    if i == start {
        return None; // no digits consumed
    }

    let value = int_part + frac_part / frac_scale;
    let value = if negative { -value } else { value };
    Some((value as f32, i))
}

/// Parse a literal string `(...)` from bytes. Returns (decoded_bytes, position_after_close_paren).
#[inline]
fn parse_literal_string_fast(data: &[u8], start: usize) -> Option<(Vec<u8>, usize)> {
    let mut i = start + 1; // past opening '('
    let mut depth: u32 = 1;

    // Fast path: scan for simple strings without escapes or nested parens.
    // Most PDF strings are simple ASCII text like "(Hello)" or single chars like "(A)".
    let scan_start = i;
    while i < data.len() {
        match data[i] {
            b')' => {
                // Simple string — no escapes, no nesting
                return Some((data[scan_start..i].to_vec(), i + 1));
            },
            b'\\' | b'(' => break, // needs complex handling
            _ => i += 1,
        }
    }

    // Slow path: string has escapes or nested parens
    i = scan_start;
    let mut result = Vec::new();
    while i < data.len() && depth > 0 {
        match data[i] {
            b'\\' if i + 1 < data.len() => {
                match data[i + 1] {
                    b'n' => {
                        result.push(b'\n');
                        i += 2;
                    },
                    b'r' => {
                        result.push(b'\r');
                        i += 2;
                    },
                    b't' => {
                        result.push(b'\t');
                        i += 2;
                    },
                    b'b' => {
                        result.push(0x08);
                        i += 2;
                    },
                    b'f' => {
                        result.push(0x0C);
                        i += 2;
                    },
                    b'(' => {
                        result.push(b'(');
                        i += 2;
                    },
                    b')' => {
                        result.push(b')');
                        i += 2;
                    },
                    b'\\' => {
                        result.push(b'\\');
                        i += 2;
                    },
                    b'0'..=b'7' => {
                        // Octal escape
                        let mut octal: u32 = (data[i + 1] - b'0') as u32;
                        let mut j = i + 2;
                        for _ in 0..2 {
                            if j < data.len() && (b'0'..=b'7').contains(&data[j]) {
                                octal = octal * 8 + (data[j] - b'0') as u32;
                                j += 1;
                            } else {
                                break;
                            }
                        }
                        result.push((octal & 0xFF) as u8);
                        i = j;
                    },
                    b'\r' => {
                        i += 2;
                        if i < data.len() && data[i] == b'\n' {
                            i += 1;
                        }
                    },
                    b'\n' => {
                        i += 2;
                    },
                    _ => {
                        result.push(data[i + 1]);
                        i += 2;
                    },
                }
            },
            b'(' => {
                depth += 1;
                result.push(b'(');
                i += 1;
            },
            b')' => {
                depth -= 1;
                if depth > 0 {
                    result.push(b')');
                }
                i += 1;
            },
            _ => {
                result.push(data[i]);
                i += 1;
            },
        }
    }
    if depth == 0 {
        Some((result, i))
    } else {
        None
    }
}

/// Parse a hex string `<...>` from bytes. Returns (decoded_bytes, position_after_close_angle).
#[inline]
fn parse_hex_string_fast(data: &[u8], start: usize) -> Option<(Vec<u8>, usize)> {
    let mut i = start + 1; // past opening '<'
    let mut result = Vec::new();
    let mut high_nibble: Option<u8> = None;
    while i < data.len() {
        let b = data[i];
        if b == b'>' {
            // If odd number of hex digits, append 0 to make final byte
            if let Some(h) = high_nibble {
                result.push(h << 4);
            }
            return Some((result, i + 1));
        }
        if let Some(nibble) = hex_nibble(b) {
            match high_nibble {
                None => high_nibble = Some(nibble),
                Some(h) => {
                    result.push((h << 4) | nibble);
                    high_nibble = None;
                },
            }
        }
        // Skip whitespace and other non-hex chars
        i += 1;
    }
    None
}

#[inline]
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Parse a TJ array `[...]` from bytes. Returns (elements, position_after_close_bracket).
fn parse_tj_array_fast(data: &[u8], start: usize) -> Option<(Vec<TextElement>, usize)> {
    let mut i = start + 1; // past opening '['
    let mut elements = Vec::new();
    loop {
        // Skip whitespace
        while i < data.len() && is_whitespace(data[i]) {
            i += 1;
        }
        if i >= data.len() {
            return None;
        }

        match data[i] {
            b']' => return Some((elements, i + 1)),
            b'(' => {
                if let Some((bytes, end)) = parse_literal_string_fast(data, i) {
                    elements.push(TextElement::String(bytes));
                    i = end;
                } else {
                    return None;
                }
            },
            b'<' => {
                if let Some((bytes, end)) = parse_hex_string_fast(data, i) {
                    elements.push(TextElement::String(bytes));
                    i = end;
                } else {
                    return None;
                }
            },
            b'0'..=b'9' | b'.' | b'+' | b'-' => {
                if let Some((num, consumed)) = parse_float_fast(&data[i..]) {
                    elements.push(TextElement::Offset(num));
                    i += consumed;
                } else {
                    return None;
                }
            },
            _ => {
                // Skip unknown token
                i += 1;
            },
        }
    }
}

/// Parse a name `/Name` from bytes. Returns (name_string, position_after_name).
#[inline]
fn parse_name_fast(data: &[u8], start: usize) -> (String, usize) {
    let mut i = start + 1; // past '/'
    let name_start = i;
    while i < data.len() && !is_whitespace_or_delimiter(data[i]) {
        i += 1;
    }
    let name = String::from_utf8_lossy(&data[name_start..i]).to_string();
    (name, i)
}

/// Fast parser for a single operator inside a BT/ET text block.
///
/// Returns `Some((remaining_input, operator))` on success, `None` on failure
/// (caller should fall back to the generic `parse_operator_with_operands`).
fn parse_text_operator_fast(input: &[u8]) -> Option<(&[u8], Operator)> {
    let mut pos = 0;
    // Small inline operand stack (max 8 operands for any PDF operator)
    let mut operands: [Option<FastOperand>; 8] = [None, None, None, None, None, None, None, None];
    let mut op_count: usize = 0;

    loop {
        // Skip whitespace
        while pos < input.len() && is_whitespace(input[pos]) {
            pos += 1;
        }
        if pos >= input.len() {
            return None;
        }

        let b = input[pos];
        match b {
            // Number operand
            b'0'..=b'9' | b'.' | b'+' | b'-' => {
                // Quick check: a lone '-' or '+' followed by non-digit is not a number
                if (b == b'-' || b == b'+')
                    && (pos + 1 >= input.len()
                        || (!input[pos + 1].is_ascii_digit() && input[pos + 1] != b'.'))
                {
                    return None; // fallback
                }
                if let Some((num, consumed)) = parse_float_fast(&input[pos..]) {
                    if op_count < 8 {
                        operands[op_count] = Some(FastOperand::Number(num));
                        op_count += 1;
                    }
                    pos += consumed;
                } else {
                    return None;
                }
            },
            // Literal string
            b'(' => {
                if let Some((bytes, end)) = parse_literal_string_fast(input, pos) {
                    if op_count < 8 {
                        operands[op_count] = Some(FastOperand::StringBytes(bytes));
                        op_count += 1;
                    }
                    pos = end;
                } else {
                    return None;
                }
            },
            // Hex string
            b'<' => {
                // Check it's not a dict <<
                if pos + 1 < input.len() && input[pos + 1] == b'<' {
                    return None; // dict — fall back to generic parser
                }
                if let Some((bytes, end)) = parse_hex_string_fast(input, pos) {
                    if op_count < 8 {
                        operands[op_count] = Some(FastOperand::StringBytes(bytes));
                        op_count += 1;
                    }
                    pos = end;
                } else {
                    return None;
                }
            },
            // Name
            b'/' => {
                let (name, end) = parse_name_fast(input, pos);
                if op_count < 8 {
                    operands[op_count] = Some(FastOperand::Name(name));
                    op_count += 1;
                }
                pos = end;
            },
            // Array (for TJ)
            b'[' => {
                if let Some((elements, end)) = parse_tj_array_fast(input, pos) {
                    if op_count < 8 {
                        operands[op_count] = Some(FastOperand::TextArray(elements));
                        op_count += 1;
                    }
                    pos = end;
                } else {
                    return None;
                }
            },
            // Operator name
            c if c.is_ascii_alphabetic() || c == b'\'' || c == b'"' || c == b'*' => {
                let op_start = pos;
                while pos < input.len()
                    && (input[pos].is_ascii_alphanumeric()
                        || input[pos] == b'\''
                        || input[pos] == b'"'
                        || input[pos] == b'*')
                {
                    pos += 1;
                }
                let op_bytes = &input[op_start..pos];
                let rest = &input[pos..];

                // Keywords that are operands, not operators
                if op_bytes == b"true" || op_bytes == b"false" || op_bytes == b"null" {
                    // These are operand values — skip them (rare in text blocks)
                    continue;
                }

                // Match operator and build typed variant
                let operator = match op_bytes {
                    b"ET" => Operator::EndText,
                    b"BT" => Operator::BeginText,
                    b"Tf" => {
                        let font = match &operands[0] {
                            Some(FastOperand::Name(n)) => n.clone(),
                            _ => String::new(),
                        };
                        let size = match &operands[1] {
                            Some(FastOperand::Number(n)) => *n,
                            // Font name might be in slot 0 and size in slot 1,
                            // but if only one operand, try it as the font name
                            _ => 12.0,
                        };
                        Operator::Tf { font, size }
                    },
                    b"Td" => {
                        let tx = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        let ty = match &operands[1] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::Td { tx, ty }
                    },
                    b"TD" => {
                        let tx = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        let ty = match &operands[1] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::TD { tx, ty }
                    },
                    b"Tm" => {
                        let get_n = |i: usize, def: f32| match &operands[i] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => def,
                        };
                        Operator::Tm {
                            a: get_n(0, 1.0),
                            b: get_n(1, 0.0),
                            c: get_n(2, 0.0),
                            d: get_n(3, 1.0),
                            e: get_n(4, 0.0),
                            f: get_n(5, 0.0),
                        }
                    },
                    b"T*" => Operator::TStar,
                    b"Tj" => {
                        let text = match operands[0].take() {
                            Some(FastOperand::StringBytes(b)) => b,
                            _ => Vec::new(),
                        };
                        Operator::Tj { text }
                    },
                    b"TJ" => {
                        let array = match operands[0].take() {
                            Some(FastOperand::TextArray(a)) => a,
                            _ => Vec::new(),
                        };
                        Operator::TJ { array }
                    },
                    b"'" => {
                        let text = match operands[0].take() {
                            Some(FastOperand::StringBytes(b)) => b,
                            _ => Vec::new(),
                        };
                        Operator::Quote { text }
                    },
                    b"\"" => {
                        let word_space = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        let char_space = match &operands[1] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        let text = match operands[2].take() {
                            Some(FastOperand::StringBytes(b)) => b,
                            _ => Vec::new(),
                        };
                        Operator::DoubleQuote {
                            word_space,
                            char_space,
                            text,
                        }
                    },
                    b"Tc" => {
                        let char_space = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::Tc { char_space }
                    },
                    b"Tw" => {
                        let word_space = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::Tw { word_space }
                    },
                    b"Tz" => {
                        let scale = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 100.0,
                        };
                        Operator::Tz { scale }
                    },
                    b"TL" => {
                        let leading = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::TL { leading }
                    },
                    b"Tr" => {
                        let render = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n as u8,
                            _ => 0,
                        };
                        Operator::Tr { render }
                    },
                    b"Ts" => {
                        let rise = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::Ts { rise }
                    },
                    b"q" => Operator::SaveState,
                    b"Q" => Operator::RestoreState,
                    b"cm" => {
                        let get_n = |i: usize, def: f32| match &operands[i] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => def,
                        };
                        Operator::Cm {
                            a: get_n(0, 1.0),
                            b: get_n(1, 0.0),
                            c: get_n(2, 0.0),
                            d: get_n(3, 1.0),
                            e: get_n(4, 0.0),
                            f: get_n(5, 0.0),
                        }
                    },
                    b"rg" => {
                        let get_n = |i: usize| match &operands[i] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::SetFillRgb {
                            r: get_n(0),
                            g: get_n(1),
                            b: get_n(2),
                        }
                    },
                    b"RG" => {
                        let get_n = |i: usize| match &operands[i] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::SetStrokeRgb {
                            r: get_n(0),
                            g: get_n(1),
                            b: get_n(2),
                        }
                    },
                    b"g" => {
                        let gray = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::SetFillGray { gray }
                    },
                    b"G" => {
                        let gray = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::SetStrokeGray { gray }
                    },
                    b"k" => {
                        let get_n = |i: usize| match &operands[i] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::SetFillCmyk {
                            c: get_n(0),
                            m: get_n(1),
                            y: get_n(2),
                            k: get_n(3),
                        }
                    },
                    b"K" => {
                        let get_n = |i: usize| match &operands[i] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::SetStrokeCmyk {
                            c: get_n(0),
                            m: get_n(1),
                            y: get_n(2),
                            k: get_n(3),
                        }
                    },
                    b"cs" => {
                        let name = match &operands[0] {
                            Some(FastOperand::Name(n)) => n.clone(),
                            _ => "DeviceGray".to_string(),
                        };
                        Operator::SetFillColorSpace { name }
                    },
                    b"CS" => {
                        let name = match &operands[0] {
                            Some(FastOperand::Name(n)) => n.clone(),
                            _ => "DeviceGray".to_string(),
                        };
                        Operator::SetStrokeColorSpace { name }
                    },
                    b"sc" => {
                        let components: Vec<f32> = operands[..op_count]
                            .iter()
                            .filter_map(|o| match o {
                                Some(FastOperand::Number(n)) => Some(*n),
                                _ => None,
                            })
                            .collect();
                        Operator::SetFillColor { components }
                    },
                    b"SC" => {
                        let components: Vec<f32> = operands[..op_count]
                            .iter()
                            .filter_map(|o| match o {
                                Some(FastOperand::Number(n)) => Some(*n),
                                _ => None,
                            })
                            .collect();
                        Operator::SetStrokeColor { components }
                    },
                    b"scn" => {
                        let name = match &operands[op_count.saturating_sub(1)] {
                            Some(FastOperand::Name(n)) => Some(n.clone()),
                            _ => None,
                        };
                        let components: Vec<f32> = operands[..op_count]
                            .iter()
                            .filter_map(|o| match o {
                                Some(FastOperand::Number(n)) => Some(*n),
                                _ => None,
                            })
                            .collect();
                        Operator::SetFillColorN {
                            components,
                            name: name.map(Box::new),
                        }
                    },
                    b"SCN" => {
                        let name = match &operands[op_count.saturating_sub(1)] {
                            Some(FastOperand::Name(n)) => Some(n.clone()),
                            _ => None,
                        };
                        let components: Vec<f32> = operands[..op_count]
                            .iter()
                            .filter_map(|o| match o {
                                Some(FastOperand::Number(n)) => Some(*n),
                                _ => None,
                            })
                            .collect();
                        Operator::SetStrokeColorN {
                            components,
                            name: name.map(Box::new),
                        }
                    },
                    b"gs" => {
                        let dict_name = match &operands[0] {
                            Some(FastOperand::Name(n)) => n.clone(),
                            _ => String::new(),
                        };
                        Operator::SetExtGState { dict_name }
                    },
                    b"Do" => {
                        let name = match &operands[0] {
                            Some(FastOperand::Name(n)) => n.clone(),
                            _ => String::new(),
                        };
                        Operator::Do { name }
                    },
                    b"w" => {
                        let width = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 1.0,
                        };
                        Operator::SetLineWidth { width }
                    },
                    b"J" => {
                        let cap_style = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n as u8,
                            _ => 0,
                        };
                        Operator::SetLineCap { cap_style }
                    },
                    b"j" => {
                        let join_style = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n as u8,
                            _ => 0,
                        };
                        Operator::SetLineJoin { join_style }
                    },
                    b"i" => {
                        let tolerance = match &operands[0] {
                            Some(FastOperand::Number(n)) => *n,
                            _ => 0.0,
                        };
                        Operator::SetFlatness { tolerance }
                    },
                    _ => {
                        // Unknown operator inside BT/ET — fall back to generic parser
                        return None;
                    },
                };

                return Some((rest, operator));
            },
            _ => {
                // Unknown byte — fall back to generic parser
                return None;
            },
        }
    }
}

// Byte classification for fast graphics scanning.
// 0 = skip (whitespace, digits, dot, sign) — bulk-skippable
// 1 = alpha/quote/star — operator start
// 2 = '(' — literal string start
// 3 = '<' — hex string or dict start
// 4 = '[' — array start
// 5 = '/' — name start
// 6 = '%' — comment start
// 7 = other (unknown byte)
const SCAN_SKIP: u8 = 0;
const SCAN_ALPHA: u8 = 1;
const SCAN_PAREN: u8 = 2;
const SCAN_ANGLE: u8 = 3;
const SCAN_BRACKET: u8 = 4;
const SCAN_SLASH: u8 = 5;
const SCAN_PERCENT: u8 = 6;
const SCAN_OTHER: u8 = 7;

static BYTE_CLASS: [u8; 256] = {
    let mut t = [SCAN_OTHER; 256];
    // Whitespace
    t[b' ' as usize] = SCAN_SKIP;
    t[b'\t' as usize] = SCAN_SKIP;
    t[b'\n' as usize] = SCAN_SKIP;
    t[b'\r' as usize] = SCAN_SKIP;
    t[0x00] = SCAN_SKIP; // null
    t[0x0C] = SCAN_SKIP; // form feed
                         // Digits
    t[b'0' as usize] = SCAN_SKIP;
    t[b'1' as usize] = SCAN_SKIP;
    t[b'2' as usize] = SCAN_SKIP;
    t[b'3' as usize] = SCAN_SKIP;
    t[b'4' as usize] = SCAN_SKIP;
    t[b'5' as usize] = SCAN_SKIP;
    t[b'6' as usize] = SCAN_SKIP;
    t[b'7' as usize] = SCAN_SKIP;
    t[b'8' as usize] = SCAN_SKIP;
    t[b'9' as usize] = SCAN_SKIP;
    // Number punctuation
    t[b'.' as usize] = SCAN_SKIP;
    t[b'+' as usize] = SCAN_SKIP;
    t[b'-' as usize] = SCAN_SKIP;
    // Alpha (uppercase)
    let mut c = b'A';
    while c <= b'Z' {
        t[c as usize] = SCAN_ALPHA;
        c += 1;
    }
    // Alpha (lowercase)
    c = b'a';
    while c <= b'z' {
        t[c as usize] = SCAN_ALPHA;
        c += 1;
    }
    // Quote/star operators
    t[b'\'' as usize] = SCAN_ALPHA;
    t[b'"' as usize] = SCAN_ALPHA;
    t[b'*' as usize] = SCAN_ALPHA;
    // Delimiters
    t[b'(' as usize] = SCAN_PAREN;
    t[b'<' as usize] = SCAN_ANGLE;
    t[b'[' as usize] = SCAN_BRACKET;
    t[b'/' as usize] = SCAN_SLASH;
    t[b'%' as usize] = SCAN_PERCENT;
    t
};

fn scan_graphics_region<'a>(data: &'a [u8], consecutive_errors: &mut usize) -> ScanResult<'a> {
    let mut i: usize = 0;
    let mut operand_start: usize = 0;
    let mut deferred_depth: u32 = 0;
    let mut deferred_start: usize = 0;
    let len = data.len();

    loop {
        // Bulk-skip whitespace, digits, dots, signs — the most common bytes in graphics streams
        while i < len && BYTE_CLASS[data[i] as usize] == SCAN_SKIP {
            i += 1;
        }
        if i >= len {
            return ScanResult::EndOfData;
        }

        match BYTE_CLASS[data[i] as usize] {
            SCAN_ALPHA => {
                let first_byte = data[i];
                let second_is_non_alpha =
                    i + 1 >= len || BYTE_CLASS[data[i + 1] as usize] != SCAN_ALPHA;

                // Fast path for common single-char skippable operators.
                // Avoids reading the full operator name and is_skippable check.
                // Path: m(moveto), l(lineto), c(curveto), v/y(curves), h(close)
                // Paint: f/F(fill), B/b(fill+stroke), S/s(stroke), n(endpath), W(clip)
                // Color: g/G(gray), k/K(cmyk)
                // State: w(linewidth), d(dash), i(flatness), J/j(cap/join), M(miter)
                // Note: q/Q excluded (need deferred depth tracking)
                if second_is_non_alpha
                    && matches!(
                        first_byte,
                        b'm' | b'l'
                            | b'c'
                            | b'v'
                            | b'y'
                            | b'h'
                            | b'f'
                            | b'F'
                            | b'B'
                            | b'b'
                            | b'S'
                            | b's'
                            | b'n'
                            | b'W'
                            | b'g'
                            | b'G'
                            | b'k'
                            | b'K'
                            | b'w'
                            | b'd'
                            | b'i'
                            | b'J'
                            | b'j'
                            | b'M'
                    )
                {
                    i += 1;
                    *consecutive_errors = 0;
                    operand_start = i;
                    continue;
                }

                let op_start = i;
                while i < len
                    && (data[i].is_ascii_alphanumeric()
                        || data[i] == b'\''
                        || data[i] == b'"'
                        || data[i] == b'*')
                {
                    i += 1;
                }
                let op = &data[op_start..i];

                // Keyword operands — not operators
                if op == b"true" || op == b"false" || op == b"null" {
                    *consecutive_errors = 0;
                    continue;
                }

                *consecutive_errors = 0;

                if op == b"q" {
                    if deferred_depth == 0 {
                        deferred_start = operand_start;
                    }
                    deferred_depth += 1;
                    operand_start = i;
                    continue;
                } else if op == b"Q" {
                    if deferred_depth > 0 {
                        deferred_depth -= 1;
                        operand_start = i;
                        continue;
                    }
                    // Unmatched Q outside deferred — emit directly.
                    // Q has no operands; NeedFullParse invokes full nom parser
                    // for a trivial no-operand op (116K triggers for Penrose).
                    return ScanResult::SimpleOp {
                        op: Operator::RestoreState,
                        rest: &data[i..],
                    };
                } else if deferred_depth > 0 {
                    // Inside a deferred q block — check if this op needs flushing
                    if op == b"cm" || op == b"gs" || is_skippable_graphics_op_bytes(op) {
                        operand_start = i;
                        continue;
                    }
                    return ScanResult::DeferredThenText {
                        deferred_start: &data[deferred_start..],
                        trigger_start: &data[operand_start..],
                    };
                } else if op == b"BT" {
                    return ScanResult::FoundBT { rest: &data[i..] };
                } else if op == b"BI" {
                    return ScanResult::InlineImage { rest: &data[i..] };
                } else if op == b"cm" {
                    // ConcatMatrix: parse 6 floats inline to avoid nom overhead
                    // (171K triggers/PDF for Murphy). Falls back to NeedFullParse
                    // on malformed operands.
                    if let Some((a, b, c, d, e, f)) =
                        parse_six_floats(&data[operand_start..op_start])
                    {
                        return ScanResult::SimpleOp {
                            op: Operator::Cm { a, b, c, d, e, f },
                            rest: &data[i..],
                        };
                    }
                    return ScanResult::NeedFullParse {
                        operand_start: &data[operand_start..],
                        after_op: &data[i..],
                    };
                } else if is_skippable_graphics_op_bytes(op) {
                    operand_start = i;
                    continue;
                } else {
                    return ScanResult::NeedFullParse {
                        operand_start: &data[operand_start..],
                        after_op: &data[i..],
                    };
                }
            },

            SCAN_PAREN => match skip_literal_string_raw(data, i) {
                Some(end) => {
                    i = end;
                    *consecutive_errors = 0;
                },
                None => {
                    i += 1;
                    *consecutive_errors += 1;
                },
            },

            SCAN_ANGLE => {
                if i + 1 < len && data[i + 1] == b'<' {
                    match skip_dict_raw(data, i) {
                        Some(end) => {
                            i = end;
                            *consecutive_errors = 0;
                        },
                        None => {
                            i += 1;
                            *consecutive_errors += 1;
                        },
                    }
                } else {
                    match skip_hex_string_raw(data, i) {
                        Some(end) => {
                            i = end;
                            *consecutive_errors = 0;
                        },
                        None => {
                            i += 1;
                            *consecutive_errors += 1;
                        },
                    }
                }
            },

            SCAN_BRACKET => match skip_array_raw(data, i) {
                Some(end) => {
                    i = end;
                    *consecutive_errors = 0;
                },
                None => {
                    i += 1;
                    *consecutive_errors += 1;
                },
            },

            SCAN_SLASH => {
                i = skip_name_raw(data, i);
                *consecutive_errors = 0;
            },

            SCAN_PERCENT => {
                while i < len && data[i] != b'\n' && data[i] != b'\r' {
                    i += 1;
                }
                *consecutive_errors = 0;
            },

            _ => {
                i += 1;
                *consecutive_errors += 1;
            },
        }

        if *consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
            return ScanResult::TooManyErrors {
                remaining: &data[i..],
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_text() {
        let stream = b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 5);

        assert!(matches!(ops[0], Operator::BeginText));
        assert!(matches!(ops[1], Operator::Tf { ref font, size } if font == "F1" && size == 12.0));
        assert!(matches!(ops[2], Operator::Td { tx, ty } if tx == 100.0 && ty == 700.0));
        assert!(matches!(ops[3], Operator::Tj { .. }));
        assert!(matches!(ops[4], Operator::EndText));
    }

    #[test]
    fn test_parse_text_matrix() {
        let stream = b"1 0 0 1 100 200 Tm";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);

        match &ops[0] {
            Operator::Tm { a, b, c, d, e, f } => {
                assert_eq!(*a, 1.0);
                assert_eq!(*b, 0.0);
                assert_eq!(*c, 0.0);
                assert_eq!(*d, 1.0);
                assert_eq!(*e, 100.0);
                assert_eq!(*f, 200.0);
            },
            _ => panic!("Expected Tm operator"),
        }
    }

    #[test]
    fn test_parse_tj_array() {
        let stream = b"[(Hello) -100 (World)] TJ";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);

        match &ops[0] {
            Operator::TJ { array } => {
                assert_eq!(array.len(), 3);
                assert!(matches!(array[0], TextElement::String(_)));
                assert!(matches!(array[1], TextElement::Offset(-100.0)));
                assert!(matches!(array[2], TextElement::String(_)));
            },
            _ => panic!("Expected TJ operator"),
        }
    }

    #[test]
    fn test_parse_color_operators() {
        // Add proper spacing between operators
        let stream = b"1 0 0 rg\n0 1 0 RG";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 2);

        match &ops[0] {
            Operator::SetFillRgb { r, g, b } => {
                assert_eq!(*r, 1.0);
                assert_eq!(*g, 0.0);
                assert_eq!(*b, 0.0);
            },
            _ => panic!("Expected rg operator"),
        }

        match &ops[1] {
            Operator::SetStrokeRgb { r, g, b } => {
                assert_eq!(*r, 0.0);
                assert_eq!(*g, 1.0);
                assert_eq!(*b, 0.0);
            },
            _ => panic!("Expected RG operator"),
        }
    }

    #[test]
    fn test_parse_graphics_state() {
        let stream = b"q 1 0 0 1 50 50 cm Q";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 3);

        assert!(matches!(ops[0], Operator::SaveState));
        assert!(matches!(ops[1], Operator::Cm { .. }));
        assert!(matches!(ops[2], Operator::RestoreState));
    }

    #[test]
    fn test_parse_t_star() {
        let stream = b"T*";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::TStar));
    }

    #[test]
    fn test_parse_text_state() {
        let stream = b"2 Tc 3 Tw 50 Tz 14 TL";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 4);

        assert!(matches!(ops[0], Operator::Tc { char_space } if char_space == 2.0));
        assert!(matches!(ops[1], Operator::Tw { word_space } if word_space == 3.0));
        assert!(matches!(ops[2], Operator::Tz { scale } if scale == 50.0));
        assert!(matches!(ops[3], Operator::TL { leading } if leading == 14.0));
    }

    #[test]
    fn test_parse_quote_operators() {
        let stream = b"(Text1) ' 1 0.5 (Text2) \"";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 2);

        assert!(matches!(ops[0], Operator::Quote { .. }));
        assert!(matches!(ops[1], Operator::DoubleQuote { .. }));
    }

    #[test]
    fn test_parse_path_operators() {
        let stream = b"100 200 m 150 250 l 10 10 50 50 re S";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 4);

        assert!(matches!(ops[0], Operator::MoveTo { x, y } if x == 100.0 && y == 200.0));
        assert!(matches!(ops[1], Operator::LineTo { x, y } if x == 150.0 && y == 250.0));
        assert!(matches!(ops[2], Operator::Rectangle { .. }));
        assert!(matches!(ops[3], Operator::Stroke));
    }

    #[test]
    fn test_parse_do_operator() {
        let stream = b"/Im1 Do";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);

        match &ops[0] {
            Operator::Do { name } => {
                assert_eq!(name, "Im1");
            },
            _ => panic!("Expected Do operator"),
        }
    }

    #[test]
    fn test_parse_empty_stream() {
        let stream = b"";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 0);
    }

    #[test]
    fn test_parse_whitespace_only() {
        let stream = b"   \n  \t  ";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 0);
    }

    #[test]
    fn test_parse_real_numbers() {
        let stream = b"1.5 2.7 Td";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);

        match &ops[0] {
            Operator::Td { tx, ty } => {
                assert_eq!(*tx, 1.5);
                assert_eq!(*ty, 2.7);
            },
            _ => panic!("Expected Td operator"),
        }
    }

    #[test]
    fn test_content_stream_operator_limit() {
        // Build a stream with more than MAX_OPERATORS simple operators.
        // Each "q\n" is a SaveState operator (1 byte + newline).
        let count = super::MAX_OPERATORS + 1000;
        let stream: Vec<u8> = "q\n".repeat(count).into_bytes();
        let ops = parse_content_stream(&stream).unwrap();
        assert_eq!(ops.len(), super::MAX_OPERATORS);
    }

    #[test]
    fn test_content_stream_consecutive_error_bailout() {
        // A stream of junk bytes that can't be parsed as any operator.
        // The parser should bail out after MAX_CONSECUTIVE_ERRORS skips.
        let junk = vec![0xFFu8; super::MAX_CONSECUTIVE_ERRORS + 500];
        let ops = parse_content_stream(&junk).unwrap();
        assert!(ops.is_empty());
    }

    // ── Tests for text-only parser ─────────────────────────────────────

    #[test]
    fn test_text_only_skips_graphics() {
        let stream = b"100 200 m 300 400 l S BT /F1 12 Tf (Hello) Tj ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert_eq!(ops.len(), 4);
        assert!(matches!(ops[0], Operator::BeginText));
        assert!(matches!(ops[1], Operator::Tf { ref font, size } if font == "F1" && size == 12.0));
        assert!(matches!(ops[2], Operator::Tj { .. }));
        assert!(matches!(ops[3], Operator::EndText));
    }

    #[test]
    fn test_text_only_preserves_state_ops() {
        let stream = b"q 1 0 0 1 50 50 cm /Im1 Do Q";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert_eq!(ops.len(), 4);
        assert!(matches!(ops[0], Operator::SaveState));
        assert!(matches!(ops[1], Operator::Cm { .. }));
        assert!(matches!(ops[2], Operator::Do { ref name } if name == "Im1"));
        assert!(matches!(ops[3], Operator::RestoreState));
    }

    #[test]
    fn test_text_only_skips_color_ops() {
        // Color operators outside BT/ET are now skipped (they don't affect text)
        let stream = b"1 0 0 rg 0.5 g /CS1 cs";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert_eq!(ops.len(), 0);
    }

    #[test]
    fn test_text_only_skips_complex_paths() {
        let stream = b"0 0 m 100 0 l 100 100 l 0 100 l h f 50 50 m 60 50 70 60 80 70 c S \
              BT /F1 10 Tf 72 700 Td (Text after paths) Tj ET \
              200 200 m 300 300 l S";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert_eq!(ops.len(), 5);
        assert!(matches!(ops[0], Operator::BeginText));
        assert!(matches!(ops[1], Operator::Tf { .. }));
        assert!(matches!(ops[2], Operator::Td { .. }));
        assert!(matches!(ops[3], Operator::Tj { .. }));
        assert!(matches!(ops[4], Operator::EndText));
    }

    #[test]
    fn test_text_only_handles_marked_content() {
        let stream = b"/Span BMC BT (Hello) Tj ET EMC";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert_eq!(ops.len(), 5);
        assert!(matches!(ops[0], Operator::BeginMarkedContent { ref tag } if tag == "Span"));
        assert!(matches!(ops[1], Operator::BeginText));
        assert!(matches!(ops[2], Operator::Tj { .. }));
        assert!(matches!(ops[3], Operator::EndText));
        assert!(matches!(ops[4], Operator::EndMarkedContent));
    }

    #[test]
    fn test_text_only_empty_and_whitespace() {
        assert_eq!(parse_content_stream_text_only(b"").unwrap().len(), 0);
        assert_eq!(parse_content_stream_text_only(b"   \n\t  ").unwrap().len(), 0);
    }

    #[test]
    fn test_text_only_graphics_only_stream() {
        let stream = b"0 0 m 100 0 l 100 100 l 0 100 l h f";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert_eq!(ops.len(), 0);
    }

    #[test]
    fn test_text_only_dash_pattern_skipped() {
        let stream = b"[3 2] 0 d BT (Hi) Tj ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0], Operator::BeginText));
        assert!(matches!(ops[1], Operator::Tj { .. }));
        assert!(matches!(ops[2], Operator::EndText));
    }

    #[test]
    fn test_text_only_gs_operator_preserved() {
        let stream = b"/GS0 gs BT (text) Tj ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert_eq!(ops.len(), 4);
        assert!(matches!(ops[0], Operator::SetExtGState { ref dict_name } if dict_name == "GS0"));
        assert!(matches!(ops[1], Operator::BeginText));
    }

    #[test]
    fn test_text_only_matches_full_parse_for_text() {
        let stream = b"q 1 0 0 1 72 700 cm BT /F1 12 Tf 0 0 Td (Hello World) Tj ET Q";
        let full = parse_content_stream(stream).unwrap();
        let text_only = parse_content_stream_text_only(stream).unwrap();

        // text_only should have the same operators minus the path/clipping ones
        // In this case there are no graphics-only ops, so they should match
        assert_eq!(full.len(), text_only.len());
    }

    #[test]
    fn test_skip_operand_token_numbers() {
        assert_eq!(skip_operand_token(b"123 ").unwrap().0, b" ");
        assert_eq!(skip_operand_token(b"-45.6 ").unwrap().0, b" ");
        assert_eq!(skip_operand_token(b"+0.5 ").unwrap().0, b" ");
        assert_eq!(skip_operand_token(b".002 ").unwrap().0, b" ");
    }

    #[test]
    fn test_skip_operand_token_strings() {
        assert_eq!(skip_operand_token(b"(hello) ").unwrap().0, b" ");
        assert_eq!(skip_operand_token(b"(nested (parens)) ").unwrap().0, b" ");
        assert_eq!(skip_operand_token(b"(escaped \\) paren) ").unwrap().0, b" ");
        assert_eq!(skip_operand_token(b"<48656C6C6F> ").unwrap().0, b" ");
    }

    #[test]
    fn test_skip_operand_token_names_arrays_dicts() {
        assert_eq!(skip_operand_token(b"/Name ").unwrap().0, b" ");
        assert_eq!(skip_operand_token(b"[1 2 3] ").unwrap().0, b" ");
        assert_eq!(skip_operand_token(b"[(text) -100] ").unwrap().0, b" ");
        assert_eq!(skip_operand_token(b"<< /K 1 >> ").unwrap().0, b" ");
    }

    #[test]
    fn test_text_only_consecutive_error_bailout() {
        let junk = vec![0xFFu8; super::MAX_CONSECUTIVE_ERRORS + 500];
        let ops = parse_content_stream_text_only(&junk).unwrap();
        assert!(ops.is_empty());
    }

    // ── Tests for prescan_text_regions (P1 memchr optimization) ──────

    #[test]
    fn test_prescan_single_bt_et() {
        let stream = b"BT /F1 12 Tf (Hello) Tj ET";
        let regions = prescan_text_regions(stream);
        assert!(regions.is_some(), "Should return Some for valid stream");
        let regions = regions.unwrap();
        assert!(!regions.is_empty(), "Should find at least 1 region");
        // The region should cover the BT..ET block
        let (start, end) = regions[0];
        assert_eq!(start, 0, "Region should start at BT");
        assert!(end >= 26, "Region should extend to or past ET");
    }

    #[test]
    fn test_prescan_multiple_bt_et() {
        let stream = b"BT (A) Tj ET BT (B) Tj ET BT (C) Tj ET";
        let regions = prescan_text_regions(stream);
        assert!(regions.is_some());
        let regions = regions.unwrap();
        // Should have 1-3 regions (may merge adjacent ones)
        assert!(!regions.is_empty(), "Should find regions for 3 BT/ET blocks");
    }

    #[test]
    fn test_prescan_do_operator() {
        // Stream with only Do (no BT) should still find a region
        let stream = b"/Im1 Do";
        let regions = prescan_text_regions(stream);
        assert!(regions.is_some());
        let regions = regions.unwrap();
        assert!(!regions.is_empty(), "Should find region for Do operator");
    }

    #[test]
    fn test_prescan_no_text_ops() {
        // Pure path data — no BT, no Do
        let stream = b"100 200 m 300 400 l S 0 0 100 100 re f";
        let regions = prescan_text_regions(stream);
        assert!(regions.is_some());
        let regions = regions.unwrap();
        assert!(
            regions.is_empty(),
            "Pure graphics should return empty regions, got {:?}",
            regions
        );
    }

    #[test]
    fn test_prescan_bt_in_string_literal() {
        // "BT" inside a string literal should NOT be matched as an operator.
        // The string (text BT here) is an operand, not a BT operator.
        // However, prescan is a heuristic — it may or may not correctly
        // handle this case. What matters is it doesn't panic and returns
        // a valid result. We verify it returns Some (not None/panic).
        let stream = b"(text BT here) Tj";
        let regions = prescan_text_regions(stream);
        // The function should handle this gracefully
        assert!(regions.is_some(), "Should not return None for string containing BT");
    }

    #[test]
    fn test_prescan_merges_overlapping_regions() {
        // Two BT blocks close together — regions should merge or be adjacent
        let stream = b"q BT (A) Tj ET Q q BT (B) Tj ET Q";
        let regions = prescan_text_regions(stream);
        assert!(regions.is_some());
        let regions = regions.unwrap();
        assert!(!regions.is_empty());
        // Verify regions are sorted and non-overlapping after merge
        for i in 1..regions.len() {
            assert!(
                regions[i].0 >= regions[i - 1].1,
                "Regions should not overlap after merge: {:?} and {:?}",
                regions[i - 1],
                regions[i]
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // Additional coverage tests
    // ══════════════════════════════════════════════════════════════════

    // ── Inline image parsing ────────────────────────────────────────

    #[test]
    fn test_parse_inline_image_basic() {
        // BI /W 4 /H 4 /BPC 8 /CS /DeviceGray ID <4 bytes data> EI
        let stream = b"BI /W 4 /H 4 /BPC 8 /CS /DeviceGray ID ABCD EI";
        let ops = parse_content_stream(stream).unwrap();
        // The parser should produce at least one InlineImage
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::InlineImage { .. })));
        // Check that the inline image dict contains expected keys
        for op in &ops {
            if let Operator::InlineImage { dict, data } = op {
                assert_eq!(dict.get("W").and_then(|o| o.as_integer()), Some(4));
                assert_eq!(dict.get("H").and_then(|o| o.as_integer()), Some(4));
                assert_eq!(dict.get("BPC").and_then(|o| o.as_integer()), Some(8));
                assert!(!data.is_empty());
            }
        }
    }

    #[test]
    fn test_parse_inline_image_empty_data() {
        // Inline image with minimal data
        let stream = b"BI /W 1 /H 1 ID X EI";
        let ops = parse_content_stream(stream).unwrap();
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::InlineImage { .. })));
    }

    #[test]
    fn test_parse_inline_image_in_stream_context() {
        // Inline image surrounded by other operators
        let stream = b"q 1 0 0 1 0 0 cm BI /W 2 /H 2 ID AB EI Q";
        let ops = parse_content_stream(stream).unwrap();
        assert!(ops.len() >= 3);
        assert!(matches!(ops[0], Operator::SaveState));
        assert!(matches!(ops[1], Operator::Cm { .. }));
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::InlineImage { .. })));
        assert!(ops.iter().any(|op| matches!(op, Operator::RestoreState)));
    }

    // ── Number parsing edge cases ───────────────────────────────────

    #[test]
    fn test_parse_negative_numbers() {
        let stream = b"-100 -200 Td";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Td { tx, ty } => {
                assert_eq!(*tx, -100.0);
                assert_eq!(*ty, -200.0);
            },
            _ => panic!("Expected Td"),
        }
    }

    #[test]
    fn test_parse_decimal_numbers() {
        let stream = b"0.001 99.999 Td";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Td { tx, ty } => {
                assert!((tx - 0.001).abs() < 0.0001);
                assert!((ty - 99.999).abs() < 0.01);
            },
            _ => panic!("Expected Td"),
        }
    }

    #[test]
    fn test_parse_leading_dot_number() {
        let stream = b".5 .25 Td";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Td { tx, ty } => {
                assert!((tx - 0.5).abs() < 0.001);
                assert!((ty - 0.25).abs() < 0.001);
            },
            _ => panic!("Expected Td"),
        }
    }

    #[test]
    fn test_parse_large_numbers() {
        let stream = b"99999 88888 Td";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Td { tx, ty } => {
                assert_eq!(*tx, 99999.0);
                assert_eq!(*ty, 88888.0);
            },
            _ => panic!("Expected Td"),
        }
    }

    #[test]
    fn test_parse_zero() {
        let stream = b"0 0 Td";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Td { tx, ty } => {
                assert_eq!(*tx, 0.0);
                assert_eq!(*ty, 0.0);
            },
            _ => panic!("Expected Td"),
        }
    }

    // ── String parsing edge cases ───────────────────────────────────

    #[test]
    fn test_parse_string_with_nested_parens() {
        let stream = b"(Hello (World)) Tj";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Tj { text } => {
                assert_eq!(text, b"Hello (World)");
            },
            _ => panic!("Expected Tj"),
        }
    }

    #[test]
    fn test_parse_string_with_escape_sequences() {
        let stream = b"(Line1\\nLine2\\r\\t) Tj";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Tj { text } => {
                // The PDF parser should handle escape sequences
                assert!(!text.is_empty());
            },
            _ => panic!("Expected Tj"),
        }
    }

    #[test]
    fn test_parse_empty_string() {
        let stream = b"() Tj";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Tj { text } => {
                assert!(text.is_empty());
            },
            _ => panic!("Expected Tj"),
        }
    }

    #[test]
    fn test_parse_hex_string() {
        let stream = b"<48656C6C6F> Tj";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Tj { text } => {
                assert_eq!(text, b"Hello");
            },
            _ => panic!("Expected Tj"),
        }
    }

    #[test]
    fn test_parse_hex_string_odd_digits() {
        // Odd number of hex digits: trailing nibble should be padded with 0
        let stream = b"<ABC> Tj";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Tj { text } => {
                assert_eq!(text.len(), 2);
                assert_eq!(text[0], 0xAB);
                assert_eq!(text[1], 0xC0);
            },
            _ => panic!("Expected Tj"),
        }
    }

    #[test]
    fn test_parse_empty_hex_string() {
        let stream = b"<> Tj";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Tj { text } => {
                assert!(text.is_empty());
            },
            _ => panic!("Expected Tj"),
        }
    }

    // ── Graphics state operators ────────────────────────────────────

    #[test]
    fn test_parse_line_width() {
        let stream = b"2.5 w";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::SetLineWidth { width } if (width - 2.5).abs() < 0.001));
    }

    #[test]
    fn test_parse_line_cap() {
        let stream = b"1 J";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::SetLineCap { cap_style: 1 }));
    }

    #[test]
    fn test_parse_line_join() {
        let stream = b"2 j";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::SetLineJoin { join_style: 2 }));
    }

    #[test]
    fn test_parse_miter_limit() {
        let stream = b"10 M";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::SetMiterLimit { limit } if limit == 10.0));
    }

    #[test]
    fn test_parse_dash_pattern() {
        let stream = b"[3 2] 0 d";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::SetDash { array, phase } => {
                assert_eq!(array, &[3.0, 2.0]);
                assert_eq!(*phase, 0.0);
            },
            _ => panic!("Expected SetDash"),
        }
    }

    #[test]
    fn test_parse_rendering_intent() {
        let stream = b"/AbsoluteColorimetric ri";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::SetRenderingIntent { intent } => {
                assert_eq!(intent, "AbsoluteColorimetric");
            },
            _ => panic!("Expected SetRenderingIntent"),
        }
    }

    #[test]
    fn test_parse_flatness() {
        let stream = b"50 i";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::SetFlatness { tolerance } if tolerance == 50.0));
    }

    #[test]
    fn test_parse_ext_gstate() {
        let stream = b"/GS0 gs";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::SetExtGState { dict_name } => {
                assert_eq!(dict_name, "GS0");
            },
            _ => panic!("Expected SetExtGState"),
        }
    }

    #[test]
    fn test_parse_paint_shading() {
        let stream = b"/Sh0 sh";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::PaintShading { name } => {
                assert_eq!(name, "Sh0");
            },
            _ => panic!("Expected PaintShading"),
        }
    }

    // ── Color operators ─────────────────────────────────────────────

    #[test]
    fn test_parse_gray_color() {
        let stream = b"0.5 g 0.8 G";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0], Operator::SetFillGray { gray } if (gray - 0.5).abs() < 0.001));
        assert!(matches!(ops[1], Operator::SetStrokeGray { gray } if (gray - 0.8).abs() < 0.001));
    }

    #[test]
    fn test_parse_cmyk_color() {
        let stream = b"0.1 0.2 0.3 0.4 k\n0.5 0.6 0.7 0.8 K";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 2);
        match &ops[0] {
            Operator::SetFillCmyk { c, m, y, k } => {
                assert!((c - 0.1).abs() < 0.01);
                assert!((m - 0.2).abs() < 0.01);
                assert!((y - 0.3).abs() < 0.01);
                assert!((k - 0.4).abs() < 0.01);
            },
            _ => panic!("Expected SetFillCmyk"),
        }
        match &ops[1] {
            Operator::SetStrokeCmyk { c, m, y, k } => {
                assert!((c - 0.5).abs() < 0.01);
                assert!((m - 0.6).abs() < 0.01);
                assert!((y - 0.7).abs() < 0.01);
                assert!((k - 0.8).abs() < 0.01);
            },
            _ => panic!("Expected SetStrokeCmyk"),
        }
    }

    #[test]
    fn test_parse_color_space_operators() {
        let stream = b"/DeviceRGB cs /DeviceCMYK CS";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 2);
        match &ops[0] {
            Operator::SetFillColorSpace { name } => assert_eq!(name, "DeviceRGB"),
            _ => panic!("Expected SetFillColorSpace"),
        }
        match &ops[1] {
            Operator::SetStrokeColorSpace { name } => assert_eq!(name, "DeviceCMYK"),
            _ => panic!("Expected SetStrokeColorSpace"),
        }
    }

    #[test]
    fn test_parse_sc_color_components() {
        let stream = b"0.1 0.2 0.3 sc";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::SetFillColor { components } => {
                assert_eq!(components.len(), 3);
                assert!((components[0] - 0.1).abs() < 0.01);
                assert!((components[1] - 0.2).abs() < 0.01);
                assert!((components[2] - 0.3).abs() < 0.01);
            },
            _ => panic!("Expected SetFillColor"),
        }
    }

    #[test]
    fn test_parse_sc_stroke_color_components() {
        let stream = b"0.5 0.6 SC";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::SetStrokeColor { components } => {
                assert_eq!(components.len(), 2);
            },
            _ => panic!("Expected SetStrokeColor"),
        }
    }

    #[test]
    fn test_parse_scn_with_pattern_name() {
        let stream = b"0.5 /Pattern1 scn";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::SetFillColorN { components, name } => {
                assert_eq!(components.len(), 1);
                assert!(name.is_some());
                assert_eq!(**name.as_ref().unwrap(), "Pattern1");
            },
            _ => panic!("Expected SetFillColorN"),
        }
    }

    #[test]
    fn test_parse_scn_without_pattern_name() {
        let stream = b"0.1 0.2 0.3 scn";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::SetFillColorN { components, name } => {
                assert_eq!(components.len(), 3);
                assert!(name.is_none());
            },
            _ => panic!("Expected SetFillColorN"),
        }
    }

    #[test]
    fn test_parse_scn_stroke_with_pattern() {
        let stream = b"0.5 /P1 SCN";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::SetStrokeColorN { components, name } => {
                assert_eq!(components.len(), 1);
                assert!(name.is_some());
            },
            _ => panic!("Expected SetStrokeColorN"),
        }
    }

    // ── Marked content operators ────────────────────────────────────

    #[test]
    fn test_parse_bmc() {
        let stream = b"/Span BMC";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::BeginMarkedContent { tag } => assert_eq!(tag, "Span"),
            _ => panic!("Expected BeginMarkedContent"),
        }
    }

    #[test]
    fn test_parse_bdc_with_dict() {
        let stream = b"/Span << /MCID 0 >> BDC";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::BeginMarkedContentDict { tag, properties } => {
                assert_eq!(tag, "Span");
                assert!(!matches!(**properties, Object::Null));
            },
            _ => panic!("Expected BeginMarkedContentDict"),
        }
    }

    #[test]
    fn test_parse_bdc_with_name_ref() {
        let stream = b"/Span /MC0 BDC";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::BeginMarkedContentDict { tag, properties } => {
                assert_eq!(tag, "Span");
                assert_eq!(properties.as_name(), Some("MC0"));
            },
            _ => panic!("Expected BeginMarkedContentDict"),
        }
    }

    #[test]
    fn test_parse_emc() {
        let stream = b"EMC";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::EndMarkedContent));
    }

    #[test]
    fn test_parse_marked_content_nesting() {
        let stream = b"/Article BMC /P << /MCID 1 >> BDC BT (text) Tj ET EMC EMC";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 7);
        assert!(matches!(ops[0], Operator::BeginMarkedContent { .. }));
        assert!(matches!(ops[1], Operator::BeginMarkedContentDict { .. }));
        assert!(matches!(ops[2], Operator::BeginText));
        assert!(matches!(ops[3], Operator::Tj { .. }));
        assert!(matches!(ops[4], Operator::EndText));
        assert!(matches!(ops[5], Operator::EndMarkedContent));
        assert!(matches!(ops[6], Operator::EndMarkedContent));
    }

    // ── Path operators ──────────────────────────────────────────────

    #[test]
    fn test_parse_bezier_curves() {
        let stream = b"10 20 30 40 50 60 c";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::CurveTo {
                x1,
                y1,
                x2,
                y2,
                x3,
                y3,
            } => {
                assert_eq!(*x1, 10.0);
                assert_eq!(*y1, 20.0);
                assert_eq!(*x2, 30.0);
                assert_eq!(*y2, 40.0);
                assert_eq!(*x3, 50.0);
                assert_eq!(*y3, 60.0);
            },
            _ => panic!("Expected CurveTo"),
        }
    }

    #[test]
    fn test_parse_curve_to_v() {
        let stream = b"10 20 30 40 v";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::CurveToV { x2, y2, x3, y3 } => {
                assert_eq!(*x2, 10.0);
                assert_eq!(*y2, 20.0);
                assert_eq!(*x3, 30.0);
                assert_eq!(*y3, 40.0);
            },
            _ => panic!("Expected CurveToV"),
        }
    }

    #[test]
    fn test_parse_curve_to_y() {
        let stream = b"10 20 30 40 y";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::CurveToY { x1, y1, x3, y3 } => {
                assert_eq!(*x1, 10.0);
                assert_eq!(*y1, 20.0);
                assert_eq!(*x3, 30.0);
                assert_eq!(*y3, 40.0);
            },
            _ => panic!("Expected CurveToY"),
        }
    }

    #[test]
    fn test_parse_close_path() {
        let stream = b"h";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::ClosePath));
    }

    #[test]
    fn test_parse_fill_variants() {
        let stream = b"f\nf*\nn";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0], Operator::Fill));
        assert!(matches!(ops[1], Operator::FillEvenOdd));
        assert!(matches!(ops[2], Operator::EndPath));
    }

    #[test]
    fn test_parse_close_fill_stroke() {
        let stream = b"b";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::CloseFillStroke));
    }

    #[test]
    fn test_parse_clipping() {
        let stream = b"W\nW*";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0], Operator::ClipNonZero));
        assert!(matches!(ops[1], Operator::ClipEvenOdd));
    }

    #[test]
    fn test_parse_rectangle() {
        let stream = b"10 20 100 50 re";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Rectangle {
                x,
                y,
                width,
                height,
            } => {
                assert_eq!(*x, 10.0);
                assert_eq!(*y, 20.0);
                assert_eq!(*width, 100.0);
                assert_eq!(*height, 50.0);
            },
            _ => panic!("Expected Rectangle"),
        }
    }

    // ── Text state operators ────────────────────────────────────────

    #[test]
    fn test_parse_tr_render_mode() {
        let stream = b"1 Tr";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::Tr { render: 1 }));
    }

    #[test]
    fn test_parse_ts_text_rise() {
        let stream = b"5 Ts";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::Ts { rise } if rise == 5.0));
    }

    #[test]
    fn test_parse_double_quote_operator() {
        let stream = b"1.5 0.5 (Hello) \"";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::DoubleQuote {
                word_space,
                char_space,
                text,
            } => {
                assert!((word_space - 1.5).abs() < 0.001);
                assert!((char_space - 0.5).abs() < 0.001);
                assert_eq!(text, b"Hello");
            },
            _ => panic!("Expected DoubleQuote"),
        }
    }

    #[test]
    fn test_parse_single_quote_operator() {
        let stream = b"(NextLine) '";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Quote { text } => {
                assert_eq!(text, b"NextLine");
            },
            _ => panic!("Expected Quote"),
        }
    }

    // ── Unknown / Other operators ───────────────────────────────────

    #[test]
    fn test_parse_unknown_operator() {
        let stream = b"42 XYZ";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Other { name, operands } => {
                assert_eq!(name, "XYZ");
                assert_eq!(operands.len(), 1);
            },
            _ => panic!("Expected Other operator"),
        }
    }

    #[test]
    fn test_parse_unknown_operator_no_operands() {
        let stream = b"BX";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Other { name, operands } => {
                assert_eq!(name, "BX");
                assert_eq!(operands.len(), 0);
            },
            _ => panic!("Expected Other for BX"),
        }
    }

    #[test]
    fn test_parse_compatibility_operators() {
        let stream = b"BX EX";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 2);
        assert!(matches!(&ops[0], Operator::Other { name, .. } if name == "BX"));
        assert!(matches!(&ops[1], Operator::Other { name, .. } if name == "EX"));
    }

    #[test]
    fn test_parse_mp_dp_marked_point() {
        let stream = b"/Tag MP /Tag2 << /K 1 >> DP";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 2);
        assert!(matches!(&ops[0], Operator::Other { name, .. } if name == "MP"));
        assert!(matches!(&ops[1], Operator::Other { name, .. } if name == "DP"));
    }

    // ── parse_content_stream_images_only ────────────────────────────

    #[test]
    fn test_images_only_captures_do() {
        let stream = b"q 1 0 0 1 0 0 cm /Im1 Do Q";
        let ops = parse_content_stream_images_only(stream).unwrap();
        // Should capture q, cm, Do, Q
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::Do { ref name } if name == "Im1")));
        assert!(ops.iter().any(|op| matches!(op, Operator::SaveState)));
        assert!(ops.iter().any(|op| matches!(op, Operator::RestoreState)));
        assert!(ops.iter().any(|op| matches!(op, Operator::Cm { .. })));
    }

    #[test]
    fn test_images_only_skips_text_blocks() {
        let stream = b"BT /F1 12 Tf (Hello) Tj ET /Im1 Do";
        let ops = parse_content_stream_images_only(stream).unwrap();
        // Should NOT contain any text operators (BT, Tf, Tj, ET)
        for op in &ops {
            assert!(!matches!(op, Operator::BeginText));
            assert!(!matches!(op, Operator::EndText));
            assert!(!matches!(op, Operator::Tf { .. }));
            assert!(!matches!(op, Operator::Tj { .. }));
        }
        // Should contain Do
        assert!(ops.iter().any(|op| matches!(op, Operator::Do { .. })));
    }

    #[test]
    fn test_images_only_empty_stream() {
        let ops = parse_content_stream_images_only(b"").unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_images_only_pure_text_stream() {
        let stream = b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET";
        let ops = parse_content_stream_images_only(stream).unwrap();
        // No image operators expected
        assert!(ops.is_empty() || !ops.iter().any(|op| matches!(op, Operator::Tj { .. })));
    }

    #[test]
    fn test_images_only_inline_image() {
        let stream = b"q 1 0 0 1 0 0 cm BI /W 2 /H 2 ID AB EI Q";
        let ops = parse_content_stream_images_only(stream).unwrap();
        // Should capture the inline image
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::InlineImage { .. })));
    }

    #[test]
    fn test_images_only_multiple_text_blocks() {
        let stream = b"BT (A) Tj ET BT (B) Tj ET q /Im1 Do Q BT (C) Tj ET";
        let ops = parse_content_stream_images_only(stream).unwrap();
        // Should skip all text blocks, capture Do
        assert!(ops.iter().any(|op| matches!(op, Operator::Do { .. })));
        assert!(!ops.iter().any(|op| matches!(op, Operator::Tj { .. })));
    }

    // ── parse_content_stream_text_only edge cases ───────────────────

    #[test]
    fn test_text_only_preserves_text_state_ops_inside_bt() {
        let stream = b"BT 2 Tc 3 Tw 50 Tz 14 TL 1 Tr 5 Ts ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert_eq!(ops.len(), 8); // BT + 6 state ops + ET
        assert!(matches!(ops[0], Operator::BeginText));
        assert!(matches!(ops[1], Operator::Tc { .. }));
        assert!(matches!(ops[2], Operator::Tw { .. }));
        assert!(matches!(ops[3], Operator::Tz { .. }));
        assert!(matches!(ops[4], Operator::TL { .. }));
        assert!(matches!(ops[5], Operator::Tr { .. }));
        assert!(matches!(ops[6], Operator::Ts { .. }));
        assert!(matches!(ops[7], Operator::EndText));
    }

    #[test]
    fn test_text_only_inline_image_outside_bt_skipped() {
        let stream = b"BI /W 2 /H 2 ID XY EI BT (Hi) Tj ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        // Inline image outside text block should be skipped in text-only mode
        assert!(!ops
            .iter()
            .any(|op| matches!(op, Operator::InlineImage { .. })));
        assert!(ops.iter().any(|op| matches!(op, Operator::Tj { .. })));
    }

    #[test]
    fn test_text_only_cm_before_bt_preserved() {
        // cm before BT should be preserved (needed for CTM calculations)
        let stream = b"q 1 0 0 1 72 700 cm BT (Hello) Tj ET Q";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops.iter().any(|op| matches!(op, Operator::Cm { .. })));
        assert!(ops.iter().any(|op| matches!(op, Operator::Tj { .. })));
    }

    // ── parse_and_execute_text_only ─────────────────────────────────

    #[test]
    fn test_parse_and_execute_text_only_basic() {
        let stream = b"BT /F1 12 Tf (Hello) Tj ET";
        let mut ops = Vec::new();
        parse_and_execute_text_only(stream, |op| {
            ops.push(op);
            Ok(())
        })
        .unwrap();
        assert!(ops.len() >= 3);
        assert!(matches!(ops[0], Operator::BeginText));
    }

    #[test]
    fn test_parse_and_execute_text_only_skips_graphics() {
        let stream = b"100 200 m 300 400 l S BT (Hello) Tj ET";
        let mut ops = Vec::new();
        parse_and_execute_text_only(stream, |op| {
            ops.push(op);
            Ok(())
        })
        .unwrap();
        // Should skip path operators
        assert!(!ops.iter().any(|op| matches!(op, Operator::MoveTo { .. })));
        assert!(ops.iter().any(|op| matches!(op, Operator::Tj { .. })));
    }

    #[test]
    fn test_parse_and_execute_text_only_empty() {
        let mut ops = Vec::new();
        parse_and_execute_text_only(b"", |op| {
            ops.push(op);
            Ok(())
        })
        .unwrap();
        assert!(ops.is_empty());
    }

    // ── Error handling / malformed streams ───────────────────────────

    #[test]
    fn test_parse_recovers_from_garbage_bytes() {
        // Garbage followed by valid operators
        let mut stream = vec![0xFF, 0xFE, 0xFD];
        stream.extend_from_slice(b" BT (Hello) Tj ET");
        let ops = parse_content_stream(&stream).unwrap();
        // Should recover and parse the text block
        assert!(ops.iter().any(|op| matches!(op, Operator::BeginText)));
        assert!(ops.iter().any(|op| matches!(op, Operator::Tj { .. })));
    }

    #[test]
    fn test_parse_truncated_inline_image() {
        // BI without matching EI
        let stream = b"BI /W 2 /H 2 ID AAAA";
        let ops = parse_content_stream(stream).unwrap();
        // Should not crash; may produce 0 ops or recover gracefully
        // The parser should handle the missing EI
        let _ = ops; // just ensure no panic
    }

    #[test]
    fn test_parse_unbalanced_bt_et() {
        // Extra ET without matching BT
        let stream = b"ET";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Operator::EndText));
    }

    #[test]
    fn test_parse_missing_operands() {
        // Td with no operands - should use defaults
        let stream = b"Td";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Td { tx, ty } => {
                assert_eq!(*tx, 0.0);
                assert_eq!(*ty, 0.0);
            },
            _ => panic!("Expected Td with default values"),
        }
    }

    #[test]
    fn test_parse_stream_with_only_comments() {
        let stream = b"% This is a comment\n% Another comment\n";
        let ops = parse_content_stream(stream).unwrap();
        // Comments should be skipped, resulting in no operators
        // The parser skips unknown bytes, so comments are handled gracefully
        let _ = ops;
    }

    // ── Complex / combined streams ──────────────────────────────────

    #[test]
    fn test_parse_complete_page_stream() {
        // Simulates a realistic mini-page content stream
        let stream = b"q\n\
            1 0 0 1 72 720 cm\n\
            /GS0 gs\n\
            BT\n\
            /F1 12 Tf\n\
            0 0 Td\n\
            14 TL\n\
            (First line) Tj\n\
            T*\n\
            (Second line) Tj\n\
            ET\n\
            Q\n\
            0 0 612 792 re\n\
            S";
        let ops = parse_content_stream(stream).unwrap();
        assert!(ops.len() >= 10);

        // Check key operators are present
        assert!(matches!(ops[0], Operator::SaveState));
        assert!(matches!(ops[1], Operator::Cm { .. }));
        assert!(matches!(ops[2], Operator::SetExtGState { .. }));
        assert!(matches!(ops[3], Operator::BeginText));
        assert!(ops.iter().any(|op| matches!(op, Operator::TStar)));
        assert!(ops.iter().any(|op| matches!(op, Operator::EndText)));
        assert!(ops.iter().any(|op| matches!(op, Operator::RestoreState)));
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::Rectangle { .. })));
        assert!(ops.iter().any(|op| matches!(op, Operator::Stroke)));
    }

    #[test]
    fn test_parse_mixed_text_and_graphics() {
        let stream = b"q\n\
            0 0 100 100 re W n\n\
            BT /F1 10 Tf 0 0 Td (Hello) Tj ET\n\
            1 0 0 rg\n\
            0 0 m 100 0 l 100 100 l h f\n\
            Q";
        let ops = parse_content_stream(stream).unwrap();
        assert!(ops.len() > 5);
        // Verify text and graphics operators are both present
        assert!(ops.iter().any(|op| matches!(op, Operator::BeginText)));
        assert!(ops.iter().any(|op| matches!(op, Operator::Tj { .. })));
        assert!(ops.iter().any(|op| matches!(op, Operator::MoveTo { .. })));
        assert!(ops.iter().any(|op| matches!(op, Operator::Fill)));
    }

    #[test]
    fn test_parse_tj_with_hex_strings_in_array() {
        let stream = b"[<48656C6C6F> -50 <576F726C64>] TJ";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::TJ { array } => {
                assert_eq!(array.len(), 3);
                match &array[0] {
                    TextElement::String(s) => assert_eq!(s, b"Hello"),
                    _ => panic!("Expected string"),
                }
                assert!(matches!(array[1], TextElement::Offset(_)));
                match &array[2] {
                    TextElement::String(s) => assert_eq!(s, b"World"),
                    _ => panic!("Expected string"),
                }
            },
            _ => panic!("Expected TJ"),
        }
    }

    #[test]
    fn test_parse_td_with_varied_whitespace() {
        // Operators separated by various whitespace types
        let stream = b"100\t200\nTd";
        let ops = parse_content_stream(stream).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            Operator::Td { tx, ty } => {
                assert_eq!(*tx, 100.0);
                assert_eq!(*ty, 200.0);
            },
            _ => panic!("Expected Td"),
        }
    }

    #[test]
    fn test_parse_multiple_bt_et_blocks() {
        let stream = b"BT (A) Tj ET BT (B) Tj ET BT (C) Tj ET";
        let ops = parse_content_stream(stream).unwrap();
        // 3 blocks * 3 ops each = 9 ops
        assert_eq!(ops.len(), 9);
        let bt_count = ops
            .iter()
            .filter(|op| matches!(op, Operator::BeginText))
            .count();
        let et_count = ops
            .iter()
            .filter(|op| matches!(op, Operator::EndText))
            .count();
        assert_eq!(bt_count, 3);
        assert_eq!(et_count, 3);
    }

    // ── Fast parser (parse_text_operator_fast) tests via text_only ──

    #[test]
    fn test_fast_parser_tf_operator() {
        let stream = b"BT /Helvetica 14.5 Tf ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert_eq!(ops.len(), 3);
        match &ops[1] {
            Operator::Tf { font, size } => {
                assert_eq!(font, "Helvetica");
                assert!((size - 14.5).abs() < 0.01);
            },
            _ => panic!("Expected Tf from fast parser"),
        }
    }

    #[test]
    fn test_fast_parser_td_operator() {
        let stream = b"BT 72.5 -14.0 Td ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops.iter().any(|op| matches!(op, Operator::Td { tx, ty }
            if (*tx - 72.5).abs() < 0.01 && (*ty - (-14.0)).abs() < 0.01)));
    }

    #[test]
    fn test_fast_parser_td_upper_operator() {
        let stream = b"BT 10 -12 TD ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops.iter().any(|op| matches!(op, Operator::TD { tx, ty }
            if (*tx - 10.0).abs() < 0.01 && (*ty - (-12.0)).abs() < 0.01)));
    }

    #[test]
    fn test_fast_parser_tm_operator() {
        let stream = b"BT 1 0 0 1 72 700 Tm ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::Tm { a, b, c, d, e, f }
            if *a == 1.0 && *b == 0.0 && *c == 0.0 && *d == 1.0 && *e == 72.0 && *f == 700.0)));
    }

    #[test]
    fn test_fast_parser_tstar_operator() {
        let stream = b"BT T* ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops.iter().any(|op| matches!(op, Operator::TStar)));
    }

    #[test]
    fn test_fast_parser_tj_with_hex_string() {
        let stream = b"BT <48656C6C6F> Tj ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops.iter().any(|op| {
            if let Operator::Tj { text } = op {
                text == b"Hello"
            } else {
                false
            }
        }));
    }

    #[test]
    fn test_fast_parser_tj_array() {
        let stream = b"BT [(AB) -100 (CD)] TJ ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::TJ { array } if array.len() == 3)));
    }

    #[test]
    fn test_fast_parser_quote_operator() {
        let stream = b"BT (Line2) ' ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops.iter().any(|op| matches!(op, Operator::Quote { .. })));
    }

    #[test]
    fn test_fast_parser_double_quote_operator() {
        let stream = b"BT 1 2 (text) \" ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::DoubleQuote { .. })));
    }

    #[test]
    fn test_fast_parser_color_ops_inside_bt() {
        let stream = b"BT 1 0 0 rg 0 g ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::SetFillRgb { .. })));
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::SetFillGray { .. })));
    }

    #[test]
    fn test_fast_parser_gs_inside_bt() {
        let stream = b"BT /GS1 gs ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops.iter().any(
            |op| matches!(op, Operator::SetExtGState { ref dict_name } if dict_name == "GS1")
        ));
    }

    #[test]
    fn test_fast_parser_do_inside_bt() {
        let stream = b"BT /XObj1 Do ET";
        let ops = parse_content_stream_text_only(stream).unwrap();
        assert!(ops
            .iter()
            .any(|op| matches!(op, Operator::Do { ref name } if name == "XObj1")));
    }

    // ── Helper function tests ───────────────────────────────────────

    #[test]
    fn test_is_operator_start() {
        assert!(is_operator_start(b'B'));
        assert!(is_operator_start(b'a'));
        assert!(is_operator_start(b'\''));
        assert!(is_operator_start(b'"'));
        assert!(is_operator_start(b'*'));
        assert!(!is_operator_start(b'0'));
        assert!(!is_operator_start(b' '));
        assert!(!is_operator_start(b'('));
        assert!(!is_operator_start(b'/'));
    }

    #[test]
    fn test_is_whitespace() {
        assert!(is_whitespace(b' '));
        assert!(is_whitespace(b'\t'));
        assert!(is_whitespace(b'\r'));
        assert!(is_whitespace(b'\n'));
        assert!(is_whitespace(0x0C));
        assert!(!is_whitespace(b'A'));
        assert!(!is_whitespace(b'0'));
    }

    #[test]
    fn test_is_whitespace_or_delimiter() {
        assert!(is_whitespace_or_delimiter(b' '));
        assert!(is_whitespace_or_delimiter(b'('));
        assert!(is_whitespace_or_delimiter(b')'));
        assert!(is_whitespace_or_delimiter(b'<'));
        assert!(is_whitespace_or_delimiter(b'>'));
        assert!(is_whitespace_or_delimiter(b'['));
        assert!(is_whitespace_or_delimiter(b']'));
        assert!(is_whitespace_or_delimiter(b'{'));
        assert!(is_whitespace_or_delimiter(b'}'));
        assert!(is_whitespace_or_delimiter(b'/'));
        assert!(is_whitespace_or_delimiter(b'%'));
        assert!(!is_whitespace_or_delimiter(b'A'));
        assert!(!is_whitespace_or_delimiter(b'0'));
    }

    #[test]
    fn test_parse_float_fast() {
        assert_eq!(parse_float_fast(b"123"), Some((123.0, 3)));
        assert_eq!(parse_float_fast(b"-45"), Some((-45.0, 3)));
        assert_eq!(parse_float_fast(b"+10"), Some((10.0, 3)));
        assert_eq!(parse_float_fast(b"0.5"), Some((0.5, 3)));
        assert_eq!(parse_float_fast(b".25"), Some((0.25, 3)));
        assert_eq!(parse_float_fast(b"-0.001"), Some((-0.001, 6)));
        assert_eq!(parse_float_fast(b"0"), Some((0.0, 1)));
        // No digits at all
        assert_eq!(parse_float_fast(b"abc"), None);
        assert_eq!(parse_float_fast(b""), None);
        // Sign only
        assert_eq!(parse_float_fast(b"-"), None);
        assert_eq!(parse_float_fast(b"+"), None);
    }

    #[test]
    fn test_parse_literal_string_fast_simple() {
        let data = b"(Hello)";
        let result = parse_literal_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, end) = result.unwrap();
        assert_eq!(bytes, b"Hello");
        assert_eq!(end, 7);
    }

    #[test]
    fn test_parse_literal_string_fast_with_escapes() {
        let data = b"(Hello\\nWorld)";
        let result = parse_literal_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, _end) = result.unwrap();
        // Should decode \n to newline
        assert!(bytes.contains(&b'\n'));
    }

    #[test]
    fn test_parse_literal_string_fast_nested_parens() {
        let data = b"(Hello (World))";
        let result = parse_literal_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, _) = result.unwrap();
        assert_eq!(bytes, b"Hello (World)");
    }

    #[test]
    fn test_parse_literal_string_fast_octal_escape() {
        let data = b"(\\101)"; // \101 = 'A' in octal
        let result = parse_literal_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, _) = result.unwrap();
        assert_eq!(bytes, b"A");
    }

    #[test]
    fn test_parse_literal_string_fast_all_escapes() {
        let data = b"(\\n\\r\\t\\b\\f\\\\\\(\\))";
        let result = parse_literal_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, _) = result.unwrap();
        assert_eq!(bytes, &[b'\n', b'\r', b'\t', 0x08, 0x0C, b'\\', b'(', b')']);
    }

    #[test]
    fn test_parse_literal_string_fast_line_continuation_cr() {
        // Backslash-CR should be ignored (line continuation)
        let data = b"(AB\\\rCD)";
        let result = parse_literal_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, _) = result.unwrap();
        assert_eq!(bytes, b"ABCD");
    }

    #[test]
    fn test_parse_literal_string_fast_line_continuation_lf() {
        // Backslash-LF should be ignored (line continuation)
        let data = b"(AB\\\nCD)";
        let result = parse_literal_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, _) = result.unwrap();
        assert_eq!(bytes, b"ABCD");
    }

    #[test]
    fn test_parse_literal_string_fast_line_continuation_crlf() {
        // Backslash-CRLF should be ignored (line continuation)
        let data = b"(AB\\\r\nCD)";
        let result = parse_literal_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, _) = result.unwrap();
        assert_eq!(bytes, b"ABCD");
    }

    #[test]
    fn test_parse_literal_string_fast_unterminated() {
        let data = b"(Hello";
        let result = parse_literal_string_fast(data, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_hex_string_fast_basic() {
        let data = b"<48656C6C6F>";
        let result = parse_hex_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, end) = result.unwrap();
        assert_eq!(bytes, b"Hello");
        assert_eq!(end, 12);
    }

    #[test]
    fn test_parse_hex_string_fast_with_whitespace() {
        let data = b"<48 65 6C 6C 6F>";
        let result = parse_hex_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, _) = result.unwrap();
        assert_eq!(bytes, b"Hello");
    }

    #[test]
    fn test_parse_hex_string_fast_odd_nibbles() {
        let data = b"<ABC>";
        let result = parse_hex_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, _) = result.unwrap();
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], 0xAB);
        assert_eq!(bytes[1], 0xC0);
    }

    #[test]
    fn test_parse_hex_string_fast_empty() {
        let data = b"<>";
        let result = parse_hex_string_fast(data, 0);
        assert!(result.is_some());
        let (bytes, _) = result.unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_parse_hex_string_fast_unterminated() {
        let data = b"<4865";
        let result = parse_hex_string_fast(data, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_hex_nibble() {
        assert_eq!(hex_nibble(b'0'), Some(0));
        assert_eq!(hex_nibble(b'9'), Some(9));
        assert_eq!(hex_nibble(b'a'), Some(10));
        assert_eq!(hex_nibble(b'f'), Some(15));
        assert_eq!(hex_nibble(b'A'), Some(10));
        assert_eq!(hex_nibble(b'F'), Some(15));
        assert_eq!(hex_nibble(b'G'), None);
        assert_eq!(hex_nibble(b' '), None);
    }

    #[test]
    fn test_parse_name_fast() {
        let data = b"/Font1 12";
        let (name, end) = parse_name_fast(data, 0);
        assert_eq!(name, "Font1");
        assert_eq!(end, 6);
    }

    #[test]
    fn test_parse_name_fast_empty() {
        let data = b"/ next";
        let (name, end) = parse_name_fast(data, 0);
        assert_eq!(name, "");
        assert_eq!(end, 1);
    }

    #[test]
    fn test_parse_tj_array_fast_basic() {
        let data = b"[(AB) -100 (CD)]";
        let result = parse_tj_array_fast(data, 0);
        assert!(result.is_some());
        let (elements, end) = result.unwrap();
        assert_eq!(elements.len(), 3);
        assert_eq!(end, 16);
        assert!(matches!(&elements[0], TextElement::String(s) if s == b"AB"));
        assert!(matches!(elements[1], TextElement::Offset(f) if (f + 100.0).abs() < 0.01));
        assert!(matches!(&elements[2], TextElement::String(s) if s == b"CD"));
    }

    #[test]
    fn test_parse_tj_array_fast_with_hex() {
        let data = b"[<4142> 50 <4344>]";
        let result = parse_tj_array_fast(data, 0);
        assert!(result.is_some());
        let (elements, _) = result.unwrap();
        assert_eq!(elements.len(), 3);
    }

    #[test]
    fn test_parse_tj_array_fast_empty() {
        let data = b"[]";
        let result = parse_tj_array_fast(data, 0);
        assert!(result.is_some());
        let (elements, _) = result.unwrap();
        assert!(elements.is_empty());
    }

    #[test]
    fn test_parse_tj_array_fast_unterminated() {
        let data = b"[(AB) -100";
        let result = parse_tj_array_fast(data, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_six_floats_valid() {
        let data = b"1 0 0 1 72 700";
        let result = parse_six_floats(data);
        assert!(result.is_some());
        let (a, b, c, d, e, f) = result.unwrap();
        assert_eq!(a, 1.0);
        assert_eq!(b, 0.0);
        assert_eq!(c, 0.0);
        assert_eq!(d, 1.0);
        assert_eq!(e, 72.0);
        assert_eq!(f, 700.0);
    }

    #[test]
    fn test_parse_six_floats_too_few() {
        let data = b"1 0 0";
        let result = parse_six_floats(data);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_six_floats_with_negatives() {
        let data = b"-1 0.5 0 -0.5 -72 700.5";
        let result = parse_six_floats(data);
        assert!(result.is_some());
        let (a, _b, _c, d, e, f) = result.unwrap();
        assert_eq!(a, -1.0);
        assert_eq!(d, -0.5);
        assert_eq!(e, -72.0);
        assert!((f - 700.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_six_floats_invalid() {
        let data = b"not numbers";
        let result = parse_six_floats(data);
        assert!(result.is_none());
    }

    // ── scan_to_et tests ────────────────────────────────────────────

    #[test]
    fn test_scan_to_et_basic() {
        let data = b"/F1 12 Tf (Hello) Tj ET";
        let result = scan_to_et(data);
        assert!(result.is_some());
        // Should return data after ET
        let remaining = result.unwrap();
        assert!(remaining.is_empty() || remaining[0] != b'E');
    }

    #[test]
    fn test_scan_to_et_with_string_containing_et() {
        // "ET" inside a string should not be matched
        let data = b"(text ET here) Tj ET";
        let result = scan_to_et(data);
        assert!(result.is_some());
    }

    #[test]
    fn test_scan_to_et_no_et() {
        let data = b"/F1 12 Tf (Hello) Tj";
        let result = scan_to_et(data);
        assert!(result.is_none());
    }

    #[test]
    fn test_scan_to_et_in_hex_string() {
        // ET inside a hex string should not be matched
        let data = b"<4554> Tj ET";
        let result = scan_to_et(data);
        assert!(result.is_some());
    }

    // ── scan_graphics_region tests ──────────────────────────────────

    #[test]
    fn test_scan_graphics_region_finds_bt() {
        let data = b"100 200 m 300 400 l BT";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(matches!(result, ScanResult::FoundBT { .. }));
    }

    #[test]
    fn test_scan_graphics_region_finds_bi() {
        let data = b"100 200 m BI";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(matches!(result, ScanResult::InlineImage { .. }));
    }

    #[test]
    fn test_scan_graphics_region_end_of_data() {
        let data = b"100 200 300 ";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(matches!(result, ScanResult::EndOfData));
    }

    #[test]
    fn test_scan_graphics_region_skips_path_ops() {
        // A stream with only skippable path operators followed by BT
        let data = b"100 200 m 300 400 l h f BT";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(matches!(result, ScanResult::FoundBT { .. }));
    }

    #[test]
    fn test_scan_graphics_region_unmatched_q() {
        // Q without matching q should yield SimpleOp
        let data = b"Q";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(matches!(
            result,
            ScanResult::SimpleOp {
                op: Operator::RestoreState,
                ..
            }
        ));
    }

    #[test]
    fn test_scan_graphics_region_deferred_q_with_trigger() {
        // q followed by Do should yield DeferredThenText
        let data = b"q 1 0 0 1 0 0 cm /Im1 Do";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(matches!(result, ScanResult::DeferredThenText { .. }));
    }

    #[test]
    fn test_scan_graphics_region_cm_with_inline_floats() {
        // cm outside q context should try inline parse
        let data = b"1 0 0 1 72 700 cm";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        match result {
            ScanResult::SimpleOp {
                op: Operator::Cm { a, b, c, d, e, f },
                ..
            } => {
                assert_eq!(a, 1.0);
                assert_eq!(b, 0.0);
                assert_eq!(c, 0.0);
                assert_eq!(d, 1.0);
                assert_eq!(e, 72.0);
                assert_eq!(f, 700.0);
            },
            _ => panic!(
                "Expected SimpleOp with Cm, got {:?}",
                match result {
                    ScanResult::EndOfData => "EndOfData",
                    ScanResult::FoundBT { .. } => "FoundBT",
                    ScanResult::InlineImage { .. } => "InlineImage",
                    ScanResult::NeedFullParse { .. } => "NeedFullParse",
                    ScanResult::DeferredThenText { .. } => "DeferredThenText",
                    ScanResult::SimpleOp { .. } => "SimpleOp (wrong variant)",
                    ScanResult::TooManyErrors { .. } => "TooManyErrors",
                }
            ),
        }
    }

    #[test]
    fn test_scan_graphics_region_skips_comments() {
        let data = b"% this is a comment\nBT";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(matches!(result, ScanResult::FoundBT { .. }));
    }

    #[test]
    fn test_scan_graphics_region_skips_strings() {
        let data = b"(some string) BT";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        // After skipping the string, should eventually find BT
        // (or NeedFullParse since string is an operand to an unknown op)
        assert!(!matches!(result, ScanResult::TooManyErrors { .. }));
    }

    #[test]
    fn test_scan_graphics_region_skips_hex_strings() {
        let data = b"<4142> BT";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(!matches!(result, ScanResult::TooManyErrors { .. }));
    }

    #[test]
    fn test_scan_graphics_region_skips_arrays() {
        let data = b"[1 2 3] BT";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(!matches!(result, ScanResult::TooManyErrors { .. }));
    }

    #[test]
    fn test_scan_graphics_region_skips_dicts() {
        let data = b"<< /K 1 >> BT";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(!matches!(result, ScanResult::TooManyErrors { .. }));
    }

    #[test]
    fn test_scan_graphics_region_skips_names() {
        let data = b"/Name BT";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(!matches!(result, ScanResult::TooManyErrors { .. }));
    }

    #[test]
    fn test_scan_graphics_region_keyword_operands() {
        // "true", "false", "null" should be treated as operands, not operators
        let data = b"true false null BT";
        let mut errors = 0usize;
        let result = scan_graphics_region(data, &mut errors);
        assert!(matches!(result, ScanResult::FoundBT { .. }));
    }

    // ── Raw skip functions tests ────────────────────────────────────

    #[test]
    fn test_skip_literal_string_raw_basic() {
        let data = b"(Hello) rest";
        let result = skip_literal_string_raw(data, 0);
        assert_eq!(result, Some(7));
    }

    #[test]
    fn test_skip_literal_string_raw_nested() {
        let data = b"(Hello (World)) rest";
        let result = skip_literal_string_raw(data, 0);
        assert_eq!(result, Some(15));
    }

    #[test]
    fn test_skip_literal_string_raw_escaped() {
        let data = b"(Hello\\)World) rest";
        let result = skip_literal_string_raw(data, 0);
        assert_eq!(result, Some(14));
    }

    #[test]
    fn test_skip_literal_string_raw_unterminated() {
        let data = b"(Hello";
        let result = skip_literal_string_raw(data, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_skip_hex_string_raw_basic() {
        let data = b"<4142> rest";
        let result = skip_hex_string_raw(data, 0);
        assert_eq!(result, Some(6));
    }

    #[test]
    fn test_skip_hex_string_raw_unterminated() {
        let data = b"<4142";
        let result = skip_hex_string_raw(data, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_skip_name_raw_basic() {
        let data = b"/FontName 12";
        let result = skip_name_raw(data, 0);
        assert_eq!(result, 9);
    }

    #[test]
    fn test_skip_array_raw_basic() {
        let data = b"[1 2 3] rest";
        let result = skip_array_raw(data, 0);
        assert_eq!(result, Some(7));
    }

    #[test]
    fn test_skip_array_raw_nested() {
        let data = b"[1 [2 3] 4]";
        let result = skip_array_raw(data, 0);
        assert_eq!(result, Some(data.len()));
    }

    #[test]
    fn test_skip_array_raw_with_string() {
        let data = b"[(Hello) 1]";
        let result = skip_array_raw(data, 0);
        assert_eq!(result, Some(data.len()));
    }

    #[test]
    fn test_skip_array_raw_with_dict() {
        let data = b"[<< /K 1 >>]";
        let result = skip_array_raw(data, 0);
        assert_eq!(result, Some(data.len()));
    }

    #[test]
    fn test_skip_array_raw_with_hex_string() {
        let data = b"[<4142>] rest";
        let result = skip_array_raw(data, 0);
        assert_eq!(result, Some(8));
    }

    #[test]
    fn test_skip_array_raw_unterminated() {
        let data = b"[1 2 3";
        let result = skip_array_raw(data, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_skip_dict_raw_basic() {
        let data = b"<< /K 1 >>";
        let result = skip_dict_raw(data, 0);
        assert_eq!(result, Some(data.len()));
    }

    #[test]
    fn test_skip_dict_raw_nested() {
        let data = b"<< /A << /B 1 >> >>";
        let result = skip_dict_raw(data, 0);
        assert_eq!(result, Some(data.len()));
    }

    #[test]
    fn test_skip_dict_raw_with_string() {
        let data = b"<< /K (Hello) >>";
        let result = skip_dict_raw(data, 0);
        assert_eq!(result, Some(data.len()));
    }

    #[test]
    fn test_skip_dict_raw_with_hex_string() {
        let data = b"<< /K <4142> >>";
        let result = skip_dict_raw(data, 0);
        assert_eq!(result, Some(data.len()));
    }

    #[test]
    fn test_skip_dict_raw_unterminated() {
        let data = b"<< /K 1";
        let result = skip_dict_raw(data, 0);
        assert!(result.is_none());
    }

    // ── is_skippable_graphics_op_bytes tests ────────────────────────

    #[test]
    fn test_is_skippable_path_construction() {
        assert!(is_skippable_graphics_op_bytes(b"m"));
        assert!(is_skippable_graphics_op_bytes(b"l"));
        assert!(is_skippable_graphics_op_bytes(b"c"));
        assert!(is_skippable_graphics_op_bytes(b"v"));
        assert!(is_skippable_graphics_op_bytes(b"y"));
        assert!(is_skippable_graphics_op_bytes(b"h"));
        assert!(is_skippable_graphics_op_bytes(b"re"));
    }

    #[test]
    fn test_is_skippable_path_painting() {
        assert!(is_skippable_graphics_op_bytes(b"S"));
        assert!(is_skippable_graphics_op_bytes(b"s"));
        assert!(is_skippable_graphics_op_bytes(b"f"));
        assert!(is_skippable_graphics_op_bytes(b"F"));
        assert!(is_skippable_graphics_op_bytes(b"f*"));
        assert!(is_skippable_graphics_op_bytes(b"B"));
        assert!(is_skippable_graphics_op_bytes(b"B*"));
        assert!(is_skippable_graphics_op_bytes(b"b"));
        assert!(is_skippable_graphics_op_bytes(b"b*"));
        assert!(is_skippable_graphics_op_bytes(b"n"));
    }

    #[test]
    fn test_is_skippable_clipping() {
        assert!(is_skippable_graphics_op_bytes(b"W"));
        assert!(is_skippable_graphics_op_bytes(b"W*"));
    }

    #[test]
    fn test_is_skippable_graphics_state() {
        assert!(is_skippable_graphics_op_bytes(b"w"));
        assert!(is_skippable_graphics_op_bytes(b"J"));
        assert!(is_skippable_graphics_op_bytes(b"j"));
        assert!(is_skippable_graphics_op_bytes(b"M"));
        assert!(is_skippable_graphics_op_bytes(b"d"));
        assert!(is_skippable_graphics_op_bytes(b"i"));
        assert!(is_skippable_graphics_op_bytes(b"ri"));
        assert!(is_skippable_graphics_op_bytes(b"sh"));
    }

    #[test]
    fn test_is_skippable_color() {
        assert!(is_skippable_graphics_op_bytes(b"rg"));
        assert!(is_skippable_graphics_op_bytes(b"RG"));
        assert!(is_skippable_graphics_op_bytes(b"g"));
        assert!(is_skippable_graphics_op_bytes(b"G"));
        assert!(is_skippable_graphics_op_bytes(b"k"));
        assert!(is_skippable_graphics_op_bytes(b"K"));
        assert!(is_skippable_graphics_op_bytes(b"cs"));
        assert!(is_skippable_graphics_op_bytes(b"CS"));
        assert!(is_skippable_graphics_op_bytes(b"sc"));
        assert!(is_skippable_graphics_op_bytes(b"SC"));
        assert!(is_skippable_graphics_op_bytes(b"scn"));
        assert!(is_skippable_graphics_op_bytes(b"SCN"));
    }

    #[test]
    fn test_is_not_skippable() {
        assert!(!is_skippable_graphics_op_bytes(b"BT"));
        assert!(!is_skippable_graphics_op_bytes(b"ET"));
        assert!(!is_skippable_graphics_op_bytes(b"Tj"));
        assert!(!is_skippable_graphics_op_bytes(b"TJ"));
        assert!(!is_skippable_graphics_op_bytes(b"Td"));
        assert!(!is_skippable_graphics_op_bytes(b"Tm"));
        assert!(!is_skippable_graphics_op_bytes(b"Tf"));
        assert!(!is_skippable_graphics_op_bytes(b"Do"));
        assert!(!is_skippable_graphics_op_bytes(b"cm"));
        assert!(!is_skippable_graphics_op_bytes(b"q"));
        assert!(!is_skippable_graphics_op_bytes(b"Q"));
        assert!(!is_skippable_graphics_op_bytes(b"gs"));
        assert!(!is_skippable_graphics_op_bytes(b"BI"));
    }

    // ── BYTE_CLASS table tests ──────────────────────────────────────

    #[test]
    fn test_byte_class_whitespace_and_digits() {
        assert_eq!(BYTE_CLASS[b' ' as usize], SCAN_SKIP);
        assert_eq!(BYTE_CLASS[b'\t' as usize], SCAN_SKIP);
        assert_eq!(BYTE_CLASS[b'\n' as usize], SCAN_SKIP);
        assert_eq!(BYTE_CLASS[b'\r' as usize], SCAN_SKIP);
        assert_eq!(BYTE_CLASS[0x0C], SCAN_SKIP);
        assert_eq!(BYTE_CLASS[0x00], SCAN_SKIP);
        for d in b'0'..=b'9' {
            assert_eq!(BYTE_CLASS[d as usize], SCAN_SKIP, "digit {} should be SKIP", d as char);
        }
        assert_eq!(BYTE_CLASS[b'.' as usize], SCAN_SKIP);
        assert_eq!(BYTE_CLASS[b'+' as usize], SCAN_SKIP);
        assert_eq!(BYTE_CLASS[b'-' as usize], SCAN_SKIP);
    }

    #[test]
    fn test_byte_class_alpha() {
        for c in b'A'..=b'Z' {
            assert_eq!(
                BYTE_CLASS[c as usize], SCAN_ALPHA,
                "uppercase {} should be ALPHA",
                c as char
            );
        }
        for c in b'a'..=b'z' {
            assert_eq!(
                BYTE_CLASS[c as usize], SCAN_ALPHA,
                "lowercase {} should be ALPHA",
                c as char
            );
        }
        assert_eq!(BYTE_CLASS[b'\'' as usize], SCAN_ALPHA);
        assert_eq!(BYTE_CLASS[b'"' as usize], SCAN_ALPHA);
        assert_eq!(BYTE_CLASS[b'*' as usize], SCAN_ALPHA);
    }

    #[test]
    fn test_byte_class_delimiters() {
        assert_eq!(BYTE_CLASS[b'(' as usize], SCAN_PAREN);
        assert_eq!(BYTE_CLASS[b'<' as usize], SCAN_ANGLE);
        assert_eq!(BYTE_CLASS[b'[' as usize], SCAN_BRACKET);
        assert_eq!(BYTE_CLASS[b'/' as usize], SCAN_SLASH);
        assert_eq!(BYTE_CLASS[b'%' as usize], SCAN_PERCENT);
    }

    // ── find_ei_operator tests ──────────────────────────────────────

    #[test]
    fn test_find_ei_operator_basic() {
        let data = b"binary data \nEI ";
        let result = find_ei_operator(data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_find_ei_operator_at_end() {
        let data = b"data \nEI";
        let result = find_ei_operator(data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_find_ei_operator_not_found() {
        let data = b"binary data without end marker";
        let result = find_ei_operator(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_ei_operator_ei_without_whitespace_prefix() {
        // EI without preceding whitespace should not match
        let data = b"dataEI ";
        let result = find_ei_operator(data);
        assert!(result.is_err());
    }

    // ── Operator limit enforcement ──────────────────────────────────

    #[test]
    fn test_text_only_operator_limit() {
        let count = super::MAX_OPERATORS + 500;
        let mut stream: Vec<u8> = Vec::new();
        stream.extend_from_slice(b"BT\n");
        for _ in 0..count {
            stream.extend_from_slice(b"T*\n");
        }
        stream.extend_from_slice(b"ET\n");
        let ops = parse_content_stream_text_only(&stream).unwrap();
        assert!(ops.len() <= super::MAX_OPERATORS + 1); // +1 for possible BT
    }

    #[test]
    fn test_images_only_operator_limit() {
        let count = super::MAX_OPERATORS + 500;
        let stream: Vec<u8> = "q\n".repeat(count).into_bytes();
        let ops = parse_content_stream_images_only(&stream).unwrap();
        assert!(ops.len() <= super::MAX_OPERATORS);
    }

    // ── Consistency tests ───────────────────────────────────────────

    #[test]
    fn test_full_and_text_only_agree_on_text_operators() {
        let stream = b"BT /F1 12 Tf 72 700 Td (Test) Tj T* (Line2) Tj ET";
        let full = parse_content_stream(stream).unwrap();
        let text_only = parse_content_stream_text_only(stream).unwrap();
        // For a pure-text stream, full and text_only should be identical
        assert_eq!(full.len(), text_only.len());
        for (f, t) in full.iter().zip(text_only.iter()) {
            assert_eq!(f, t);
        }
    }

    #[test]
    fn test_full_parse_and_execute_agree() {
        let stream = b"BT /F1 12 Tf (Hello) Tj ET";
        let full = parse_content_stream(stream).unwrap();
        let mut exec_ops = Vec::new();
        parse_and_execute_text_only(stream, |op| {
            exec_ops.push(op);
            Ok(())
        })
        .unwrap();
        // Both should produce the same text operators
        assert_eq!(full.len(), exec_ops.len());
    }

    // ── Edge cases for operand parsing via nom ──────────────────────

    #[test]
    fn test_skip_operand_token_dict_with_strings() {
        assert_eq!(skip_operand_token(b"<< /Key (value) >> ").unwrap().0, b" ");
    }

    #[test]
    fn test_skip_operand_token_nested_array() {
        assert_eq!(skip_operand_token(b"[1 [2 3] 4] ").unwrap().0, b" ");
    }

    #[test]
    fn test_skip_operand_token_array_with_hex_string() {
        assert_eq!(skip_operand_token(b"[<4142> 1] ").unwrap().0, b" ");
    }

    #[test]
    fn test_skip_operand_token_array_with_dict() {
        assert_eq!(skip_operand_token(b"[<< /K 1 >>] ").unwrap().0, b" ");
    }

    #[test]
    fn test_skip_operand_token_dict_with_nested_dict() {
        assert_eq!(skip_operand_token(b"<< /A << /B 1 >> >> ").unwrap().0, b" ");
    }

    #[test]
    fn test_skip_operand_token_dict_with_hex_string() {
        assert_eq!(skip_operand_token(b"<< /K <4142> >> ").unwrap().0, b" ");
    }

    #[test]
    fn test_skip_operand_token_errors() {
        // Characters that don't start a valid operand
        assert!(skip_operand_token(b"").is_err());
        assert!(skip_operand_token(b"@").is_err());
    }
}
