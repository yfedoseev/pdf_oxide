//! Consensus pseudo-ground-truth.
//!
//! When no manual markdown reference exists for a PDF, we fall back to
//! a "what do N engines agree on" baseline: the intersection of tokens
//! that appear in output from ≥2 engines becomes the reference set.
//! TF1 against this is a measure of agreement with the ensemble, not
//! absolute quality — results are clearly labelled `reference: consensus`
//! in the report so readers don't confuse the two.
//!
//! Useful for:
//! - Smoke-testing a new release against N peer engines when we have no
//!   curated ground-truth corpus.
//! - Detecting drift: if pdf_oxide's agreement with the consensus drops
//!   between versions on a stable input, something changed.

use crate::engine::{Engine, Extraction};
use crate::score::{token_f1, tokenize};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Build a pseudo-ground-truth for one PDF from peer engines' output.
/// Returns the token set that appears in output from at least `min_agree`
/// engines (default 2). If fewer engines succeed, returns `None`.
pub fn consensus_tokens(
    pdf: &Path,
    engines: &[Box<dyn Engine>],
    min_agree: usize,
) -> Option<HashSet<String>> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut successful = 0usize;
    for e in engines {
        let Ok(Extraction { markdown, .. }) = e.extract(pdf) else {
            continue;
        };
        successful += 1;
        let tokens: HashSet<String> = tokenize(&markdown).into_iter().collect();
        for t in tokens {
            *counts.entry(t).or_insert(0) += 1;
        }
    }
    if successful < min_agree {
        return None;
    }
    Some(
        counts
            .into_iter()
            .filter(|(_, c)| *c >= min_agree)
            .map(|(t, _)| t)
            .collect(),
    )
}

/// Score one engine's output against a consensus token set (TF1-style).
pub fn score_against_consensus(extracted_md: &str, consensus: &HashSet<String>) -> f64 {
    let ext_tokens: Vec<String> = tokenize(extracted_md);
    let gt_tokens: Vec<String> = consensus.iter().cloned().collect();
    token_f1(&ext_tokens, &gt_tokens)
}

/// Convenience: build consensus from a list of engines and score the
/// target engine's output against it in a single call.
pub fn consensus_tf1(
    pdf: &Path,
    peers: &[Box<dyn Engine>],
    target_md: &str,
    min_agree: usize,
) -> Result<Option<f64>> {
    Ok(consensus_tokens(pdf, peers, min_agree).map(|c| score_against_consensus(target_md, &c)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct FakeEngine(&'static str, &'static str);
    impl Engine for FakeEngine {
        fn name(&self) -> &'static str {
            self.0
        }
        fn extract(&self, _pdf: &Path) -> Result<Extraction> {
            Ok(Extraction {
                markdown: self.1.to_string(),
                duration: Duration::from_millis(1),
            })
        }
    }

    #[test]
    fn consensus_picks_tokens_in_two_or_more_engines() {
        let engines: Vec<Box<dyn Engine>> = vec![
            Box::new(FakeEngine("a", "alpha beta gamma")),
            Box::new(FakeEngine("b", "alpha beta delta")),
            Box::new(FakeEngine("c", "alpha epsilon zeta")),
        ];
        let c = consensus_tokens(Path::new("dummy"), &engines, 2).unwrap();
        // alpha appears in all 3 → in. beta in 2 → in. gamma, delta,
        // epsilon, zeta each only once → out.
        assert!(c.contains("alpha"));
        assert!(c.contains("beta"));
        assert!(!c.contains("gamma"));
        assert!(!c.contains("delta"));
        assert!(!c.contains("epsilon"));
    }

    #[test]
    fn consensus_none_when_not_enough_engines_succeed() {
        let engines: Vec<Box<dyn Engine>> = vec![Box::new(FakeEngine("a", "alpha"))];
        let c = consensus_tokens(Path::new("dummy"), &engines, 2);
        assert!(c.is_none());
    }

    #[test]
    fn score_against_consensus_rewards_overlap() {
        let mut consensus = HashSet::new();
        consensus.insert("alpha".to_string());
        consensus.insert("beta".to_string());
        consensus.insert("gamma".to_string());

        let perfect = score_against_consensus("alpha beta gamma", &consensus);
        assert!((perfect - 1.0).abs() < 1e-6);

        let partial = score_against_consensus("alpha beta zzz", &consensus);
        assert!(partial > 0.0 && partial < 1.0);
    }
}
