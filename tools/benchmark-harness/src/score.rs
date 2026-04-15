//! TF1 + SF1 scoring primitives.
//!
//! Formulas mirror Kreuzberg's benchmark-harness so numbers stay
//! cross-comparable. Implementation is deliberately minimal — every
//! function is a pure transform on markdown strings.

use std::collections::HashSet;

/// Lowercase alphanumeric tokenization. Shared between TF1 and the
/// per-block content similarity that feeds SF1.
pub fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Bag-of-words F1. `ext` = extracted, `gt` = ground truth.
pub fn token_f1(ext: &[String], gt: &[String]) -> f64 {
    if ext.is_empty() && gt.is_empty() {
        return 1.0;
    }
    if ext.is_empty() || gt.is_empty() {
        return 0.0;
    }
    let es: HashSet<&String> = ext.iter().collect();
    let gs: HashSet<&String> = gt.iter().collect();
    let inter = es.intersection(&gs).count() as f64;
    let precision = inter / es.len() as f64;
    let recall = inter / gs.len() as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Convenience: TF1 between two markdown strings.
pub fn tf1(extracted_md: &str, ground_truth_md: &str) -> f64 {
    token_f1(&tokenize(extracted_md), &tokenize(ground_truth_md))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_lowercases_and_strips_punct() {
        assert_eq!(tokenize("Hello, World!"), vec!["hello", "world"]);
        assert_eq!(tokenize("foo-bar baz"), vec!["foo", "bar", "baz"]);
        assert_eq!(tokenize("2024-Q1 revenue"), vec!["2024", "q1", "revenue"]);
    }

    #[test]
    fn identical_strings_score_1() {
        assert_eq!(tf1("Hello world", "Hello world"), 1.0);
    }

    #[test]
    fn disjoint_strings_score_0() {
        assert_eq!(tf1("alpha beta", "gamma delta"), 0.0);
    }

    #[test]
    fn empty_both_sides_score_1() {
        assert_eq!(tf1("", ""), 1.0);
    }

    #[test]
    fn partial_overlap_between_0_and_1() {
        let s = tf1("alpha beta gamma", "alpha delta gamma");
        assert!((0.0..1.0).contains(&s), "partial overlap should score in (0,1), got {s}");
    }
}
