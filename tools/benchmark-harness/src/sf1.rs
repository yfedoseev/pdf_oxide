//! Structural F1 (SF1) — block-weighted markdown similarity with
//! LIS-based ordering.
//!
//! Parses markdown into a typed block stream via pulldown-cmark,
//! greedily matches extracted ↔ ground-truth blocks by
//! `content_tf1 × type_compat`, then aggregates a weight-weighted F1
//! with per-block-type weights. The ordering component is the LIS
//! length of matched pairs divided by match count.
//!
//! Formula refs mirror Kreuzberg's tools/benchmark-harness so the
//! numbers we publish are directly comparable to their reports.

use crate::score::{token_f1, tokenize};
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Parser, Tag, TagEnd};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlockType {
    Heading(u8), // 1..=6
    Paragraph,
    CodeBlock,
    Formula,
    Table,
    ListItem,
    Image,
}

#[derive(Debug)]
pub struct Block {
    pub kind: BlockType,
    pub text: String,
}

/// Per-block weight. Heading detection is the highest-signal layout
/// decision, so weight it double a paragraph; code/formula/table
/// need engine-specific handling, so weight 1.5.
pub fn weight(kind: BlockType) -> f64 {
    match kind {
        BlockType::Heading(_) => 2.0,
        BlockType::CodeBlock | BlockType::Formula | BlockType::Table => 1.5,
        BlockType::ListItem => 1.0,
        BlockType::Paragraph | BlockType::Image => 0.5,
    }
}

/// Type-compatibility matrix. 1.0 = exact type match, 0.0 = rejected.
/// The cross-type entries reflect common confusions between engines
/// (e.g. a docling heading vs. an extracted bold-wrapped paragraph).
pub fn type_compat(ext: BlockType, gt: BlockType) -> f64 {
    if ext == gt {
        return 1.0;
    }
    match (ext, gt) {
        (BlockType::Heading(a), BlockType::Heading(b)) => {
            let dist = a.abs_diff(b) as f64;
            (1.0 - 0.1 * dist).max(0.6)
        },
        (BlockType::ListItem, BlockType::Paragraph)
        | (BlockType::Paragraph, BlockType::ListItem) => 0.5,
        (BlockType::Paragraph, BlockType::Heading(_))
        | (BlockType::Heading(_), BlockType::Paragraph) => 0.25,
        (BlockType::CodeBlock, BlockType::Formula) | (BlockType::Formula, BlockType::CodeBlock) => {
            0.3
        },
        (BlockType::Table, BlockType::Paragraph) | (BlockType::Paragraph, BlockType::Table) => 0.25,
        (BlockType::CodeBlock, BlockType::Paragraph)
        | (BlockType::Paragraph, BlockType::CodeBlock) => 0.2,
        _ => 0.0,
    }
}

pub fn parse_blocks(md: &str) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut stack: Vec<(BlockType, String)> = Vec::new();
    let opts = pulldown_cmark::Options::ENABLE_TABLES
        | pulldown_cmark::Options::ENABLE_MATH
        | pulldown_cmark::Options::ENABLE_GFM;
    for ev in Parser::new_ext(md, opts) {
        match ev {
            Event::Start(Tag::Heading { level, .. }) => {
                let lvl = match level {
                    HeadingLevel::H1 => 1,
                    HeadingLevel::H2 => 2,
                    HeadingLevel::H3 => 3,
                    HeadingLevel::H4 => 4,
                    HeadingLevel::H5 => 5,
                    HeadingLevel::H6 => 6,
                };
                stack.push((BlockType::Heading(lvl), String::new()));
            },
            Event::Start(Tag::Paragraph) => {
                stack.push((BlockType::Paragraph, String::new()));
            },
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(_) | CodeBlockKind::Indented)) => {
                stack.push((BlockType::CodeBlock, String::new()));
            },
            Event::Start(Tag::Item) => {
                stack.push((BlockType::ListItem, String::new()));
            },
            Event::Start(Tag::Table(_)) => {
                stack.push((BlockType::Table, String::new()));
            },
            Event::Start(Tag::Image { .. }) => {
                stack.push((BlockType::Image, String::new()));
            },
            Event::Start(Tag::MetadataBlock(_)) => {
                // Skip frontmatter; no scoring value.
                stack.push((BlockType::Paragraph, String::new()));
            },
            Event::End(
                TagEnd::Heading(_)
                | TagEnd::Paragraph
                | TagEnd::CodeBlock
                | TagEnd::Item
                | TagEnd::Table
                | TagEnd::Image
                | TagEnd::MetadataBlock(_),
            ) => {
                if let Some((kind, text)) = stack.pop() {
                    let trimmed = text.trim().to_string();
                    if !trimmed.is_empty() {
                        blocks.push(Block {
                            kind,
                            text: trimmed,
                        });
                    }
                }
            },
            Event::Text(ref t)
            | Event::Code(ref t)
            | Event::InlineMath(ref t)
            | Event::DisplayMath(ref t) => {
                if matches!(ev, Event::InlineMath(_) | Event::DisplayMath(_)) {
                    // Promote the enclosing block when we see math — most
                    // engines emit formulas inside a paragraph.
                    if let Some((k, _)) = stack.last_mut() {
                        if matches!(k, BlockType::Paragraph) {
                            *k = BlockType::Formula;
                        }
                    }
                }
                if let Some((_, buf)) = stack.last_mut() {
                    if !buf.is_empty() {
                        buf.push(' ');
                    }
                    buf.push_str(t);
                }
            },
            Event::SoftBreak | Event::HardBreak => {
                if let Some((_, buf)) = stack.last_mut() {
                    buf.push(' ');
                }
            },
            _ => {},
        }
    }
    // Flush anything left open by a malformed document.
    while let Some((kind, text)) = stack.pop() {
        let trimmed = text.trim().to_string();
        if !trimmed.is_empty() {
            blocks.push(Block {
                kind,
                text: trimmed,
            });
        }
    }
    blocks
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    ext_idx: usize,
    gt_idx: usize,
    score: f64,
    content_tf1: f64,
}

/// Longest-increasing-subsequence length; used as the order score.
fn lis_len(xs: &[usize]) -> usize {
    let mut tails: Vec<usize> = Vec::new();
    for &x in xs {
        // Binary search for the first tail >= x.
        let pos = tails.partition_point(|&t| t < x);
        if pos == tails.len() {
            tails.push(x);
        } else {
            tails[pos] = x;
        }
    }
    tails.len()
}

#[derive(Debug, Default)]
pub struct Sf1 {
    pub sf1: f64,
    pub precision: f64,
    pub recall: f64,
    pub order_score: f64,
    pub matched: usize,
}

/// Score SF1 between extracted markdown and ground-truth markdown.
pub fn sf1(extracted_md: &str, ground_truth_md: &str) -> Sf1 {
    let ext = parse_blocks(extracted_md);
    let gt = parse_blocks(ground_truth_md);
    sf1_blocks(&ext, &gt)
}

fn sf1_blocks(ext: &[Block], gt: &[Block]) -> Sf1 {
    if ext.is_empty() && gt.is_empty() {
        return Sf1 {
            sf1: 1.0,
            precision: 1.0,
            recall: 1.0,
            order_score: 1.0,
            matched: 0,
        };
    }
    if ext.is_empty() || gt.is_empty() {
        return Sf1::default();
    }

    // Pre-tokenize once per side.
    let ext_tokens: Vec<Vec<String>> = ext.iter().map(|b| tokenize(&b.text)).collect();
    let gt_tokens: Vec<Vec<String>> = gt.iter().map(|b| tokenize(&b.text)).collect();

    // Enumerate candidate matches above threshold.
    let mut cands: Vec<Candidate> = Vec::new();
    for (i, eb) in ext.iter().enumerate() {
        for (j, gb) in gt.iter().enumerate() {
            let compat = type_compat(eb.kind, gb.kind);
            if compat == 0.0 {
                continue;
            }
            let content = token_f1(&ext_tokens[i], &gt_tokens[j]);
            let score = content * compat;
            let short_block = ext_tokens[i].len().min(gt_tokens[j].len()) < 5;
            let threshold = if short_block { 0.20 } else { 0.10 };
            if score >= threshold {
                cands.push(Candidate {
                    ext_idx: i,
                    gt_idx: j,
                    score,
                    content_tf1: content,
                });
            }
        }
    }

    // Greedy assignment by descending score.
    cands.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut used_ext = vec![false; ext.len()];
    let mut used_gt = vec![false; gt.len()];
    let mut matches: Vec<Candidate> = Vec::new();
    for c in cands {
        if !used_ext[c.ext_idx] && !used_gt[c.gt_idx] {
            used_ext[c.ext_idx] = true;
            used_gt[c.gt_idx] = true;
            matches.push(c);
        }
    }

    // Weighted P/R.
    let total_gt_weight: f64 = gt.iter().map(|b| weight(b.kind)).sum();
    let total_ext_weight: f64 = ext.iter().map(|b| weight(b.kind)).sum();
    let matched_gt_weight: f64 = matches
        .iter()
        .map(|m| {
            weight(gt[m.gt_idx].kind)
                * (m.content_tf1 * type_compat(ext[m.ext_idx].kind, gt[m.gt_idx].kind))
        })
        .sum();
    let matched_ext_weight: f64 = matches
        .iter()
        .map(|m| {
            weight(ext[m.ext_idx].kind)
                * (m.content_tf1 * type_compat(ext[m.ext_idx].kind, gt[m.gt_idx].kind))
        })
        .sum();

    let recall = if total_gt_weight > 0.0 {
        matched_gt_weight / total_gt_weight
    } else {
        0.0
    };
    let precision = if total_ext_weight > 0.0 {
        matched_ext_weight / total_ext_weight
    } else {
        0.0
    };
    let sf1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    // LIS order on the ext indices of matches sorted by gt index.
    let mut ordered = matches.clone();
    ordered.sort_by_key(|m| m.gt_idx);
    let ext_seq: Vec<usize> = ordered.iter().map(|m| m.ext_idx).collect();
    let order_score = if ext_seq.is_empty() {
        0.0
    } else {
        lis_len(&ext_seq) as f64 / ext_seq.len() as f64
    };

    Sf1 {
        sf1,
        precision,
        recall,
        order_score,
        matched: matches.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_headings_and_paragraphs() {
        let md = "# Title\n\nA paragraph about alpha beta.\n\n## Section\n\nAnother one.\n";
        let blocks = parse_blocks(md);
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].kind, BlockType::Heading(1));
        assert_eq!(blocks[1].kind, BlockType::Paragraph);
        assert_eq!(blocks[2].kind, BlockType::Heading(2));
        assert_eq!(blocks[3].kind, BlockType::Paragraph);
    }

    #[test]
    fn parse_code_block() {
        let md = "```\nlet x = 1;\n```\n";
        let b = parse_blocks(md);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].kind, BlockType::CodeBlock);
    }

    #[test]
    fn parse_table() {
        let md = "| a | b |\n|---|---|\n| 1 | 2 |\n";
        let b = parse_blocks(md);
        assert_eq!(b[0].kind, BlockType::Table);
    }

    #[test]
    fn identical_markdown_scores_sf1_1() {
        let md = "# Hello\n\nSome body text here.\n\n- one\n- two\n";
        let s = sf1(md, md);
        assert!((s.sf1 - 1.0).abs() < 1e-6, "SF1 should be 1.0 on identical input, got {s:?}");
        assert!((s.order_score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn completely_disjoint_scores_0() {
        let ext = "# Alpha\n\nbeta gamma delta epsilon\n";
        let gt = "# Omega\n\nrho sigma tau upsilon\n";
        let s = sf1(ext, gt);
        assert!(s.sf1 < 0.3, "disjoint content should score low, got {s:?}");
    }

    #[test]
    fn heading_level_mismatch_is_partial_compat() {
        // h1 vs h3 → 0.8 compat, same content → sf1 around 0.8.
        let ext = "# Identical body text here\n";
        let gt = "### Identical body text here\n";
        let s = sf1(ext, gt);
        assert!(s.sf1 > 0.6 && s.sf1 < 1.0, "expected partial match, got {s:?}");
    }

    #[test]
    fn order_penalty_on_reversed_matches() {
        let ext = "# Second Section Topic Two\n\n# First Section Topic One\n";
        let gt = "# First Section Topic One\n\n# Second Section Topic Two\n";
        let s = sf1(ext, gt);
        assert_eq!(s.matched, 2);
        // Two matches in reverse order: LIS=1, so order_score = 1/2.
        assert!((s.order_score - 0.5).abs() < 1e-6, "order_score should be 0.5, got {s:?}");
    }

    #[test]
    fn lis_length_basic() {
        assert_eq!(lis_len(&[]), 0);
        assert_eq!(lis_len(&[0]), 1);
        assert_eq!(lis_len(&[0, 1, 2, 3]), 4);
        assert_eq!(lis_len(&[3, 2, 1, 0]), 1);
        assert_eq!(lis_len(&[1, 3, 2, 4, 5]), 4);
    }

    #[test]
    fn weight_taxonomy_matches_spec() {
        assert_eq!(weight(BlockType::Heading(1)), 2.0);
        assert_eq!(weight(BlockType::Heading(6)), 2.0);
        assert_eq!(weight(BlockType::CodeBlock), 1.5);
        assert_eq!(weight(BlockType::Formula), 1.5);
        assert_eq!(weight(BlockType::Table), 1.5);
        assert_eq!(weight(BlockType::ListItem), 1.0);
        assert_eq!(weight(BlockType::Paragraph), 0.5);
        assert_eq!(weight(BlockType::Image), 0.5);
    }

    #[test]
    fn compat_heading_to_heading_distance() {
        assert_eq!(type_compat(BlockType::Heading(1), BlockType::Heading(1)), 1.0);
        // h1 vs h2 = 0.9
        let s = type_compat(BlockType::Heading(1), BlockType::Heading(2));
        assert!((s - 0.9).abs() < 1e-6, "h1↔h2 should be 0.9, got {s}");
        // h1 vs h6 would be 1 - 0.5 = 0.5, clamped to min 0.6
        let s = type_compat(BlockType::Heading(1), BlockType::Heading(6));
        assert!((s - 0.6).abs() < 1e-6);
    }
}
