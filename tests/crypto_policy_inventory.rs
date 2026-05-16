//! #230 — process-wide crypto inventory (CBOM-adjacent observability).
//!
//! Own integration binary: `INVENTORY` is a per-process atomic bitset.

use pdf_oxide::crypto::{inventory, record_algorithm_use, AlgorithmId};

#[test]
fn inventory_starts_empty_records_idempotently_and_in_order() {
    assert!(inventory().is_empty(), "fresh process has an empty inventory");

    record_algorithm_use(AlgorithmId::CipherAes256Cbc);
    record_algorithm_use(AlgorithmId::HashSha256);
    // Idempotent — recording the same id twice does not duplicate.
    record_algorithm_use(AlgorithmId::HashSha256);

    let inv = inventory();
    assert!(inv.contains(&AlgorithmId::CipherAes256Cbc));
    assert!(inv.contains(&AlgorithmId::HashSha256));
    assert!(!inv.contains(&AlgorithmId::CipherRc4), "unused id absent");
    assert_eq!(inv.len(), 2, "no duplicates");

    // Returned in AlgorithmId::ALL declaration order (sha256 before aes256).
    let positions: Vec<usize> = inv.iter().map(|a| a.index()).collect();
    let mut sorted = positions.clone();
    sorted.sort_unstable();
    assert_eq!(positions, sorted, "inventory is in declaration order");
}
