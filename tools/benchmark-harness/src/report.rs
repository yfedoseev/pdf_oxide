//! Run-and-diff: drive engines across a corpus, emit a JSON report,
//! compare two reports and gate on regression.

use crate::consensus;
use crate::engine::{self, Engine};
use crate::score;
use crate::sf1;
use crate::{DiffArgs, RunArgs};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Debug)]
pub struct FixtureResult {
    pub name: String,
    pub tf1: Option<f64>,
    pub sf1: Option<f64>,
    pub sf1_precision: Option<f64>,
    pub sf1_recall: Option<f64>,
    pub order_score: Option<f64>,
    pub matched_blocks: Option<usize>,
    pub duration_ms: Option<u128>,
    pub error: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Aggregate {
    pub count: usize,
    pub ok: usize,
    pub tf1_mean: f64,
    pub tf1_p50: f64,
    pub tf1_p90: f64,
    pub sf1_mean: f64,
    pub sf1_p50: f64,
    pub sf1_p90: f64,
    pub order_mean: f64,
    pub duration_ms_total: u128,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Report {
    pub engine: String,
    pub corpus: PathBuf,
    /// `manual` when scored against a ground-truth directory; the
    /// comma-joined list of peer engine names when scored against a
    /// consensus baseline. Stored in the report so downstream readers
    /// never confuse absolute quality with inter-engine agreement.
    pub reference: String,
    pub ground_truth: Option<PathBuf>,
    pub fixtures: Vec<FixtureResult>,
    pub aggregate: Aggregate,
}

pub fn run(args: RunArgs) -> Result<()> {
    let engine = engine::build(args.engine)?;
    log::info!("engine = {}", engine.name());

    let (fixtures, reference) = if let Some(gt_dir) = &args.ground_truth {
        let pairs = collect_pairs(&args.corpus, gt_dir)?;
        if pairs.is_empty() {
            return Err(anyhow!(
                "no PDF/markdown pairs found — expected matching *.pdf under {} \
                 and *.md under {}",
                args.corpus.display(),
                gt_dir.display()
            ));
        }
        log::info!("found {} fixture pairs (manual ground truth)", pairs.len());
        let mut fixtures = Vec::with_capacity(pairs.len());
        for (i, (pdf, gt_path)) in pairs.iter().enumerate() {
            log::info!("[{}/{}] {}", i + 1, pairs.len(), pdf.display());
            fixtures.push(score_one_manual(&*engine, pdf, gt_path));
        }
        (fixtures, "manual".to_string())
    } else {
        // Consensus mode: peers provide pseudo-ground-truth.
        let peers: Vec<Box<dyn Engine>> = args
            .consensus_peers
            .iter()
            .map(|k| engine::build(*k))
            .collect::<Result<Vec<_>>>()?;
        let peer_names: Vec<&str> = peers.iter().map(|p| p.name()).collect();
        let reference = format!("consensus({})", peer_names.join(","));
        log::info!("consensus mode — peers: {}", peer_names.join(", "));
        let pdfs = collect_pdfs(&args.corpus)?;
        let mut fixtures = Vec::with_capacity(pdfs.len());
        for (i, pdf) in pdfs.iter().enumerate() {
            log::info!("[{}/{}] {}", i + 1, pdfs.len(), pdf.display());
            fixtures.push(score_one_consensus(&*engine, pdf, &peers, args.consensus_min_agree));
        }
        (fixtures, reference)
    };

    let aggregate = aggregate(&fixtures);
    let report = Report {
        engine: engine.name().to_string(),
        corpus: args.corpus,
        reference,
        ground_truth: args.ground_truth,
        fixtures,
        aggregate,
    };
    fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    log::info!(
        "wrote {} — mean TF1 {:.3} / SF1 {:.3} across {} fixtures ({} ok), reference={}",
        args.output.display(),
        report.aggregate.tf1_mean,
        report.aggregate.sf1_mean,
        report.aggregate.count,
        report.aggregate.ok,
        report.reference,
    );
    Ok(())
}

fn score_one_manual(engine: &dyn Engine, pdf: &Path, gt_path: &Path) -> FixtureResult {
    let name = pdf
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    match engine.extract(pdf) {
        Ok(ext) => {
            let gt = match fs::read_to_string(gt_path) {
                Ok(s) => s,
                Err(e) => {
                    return FixtureResult {
                        name,
                        tf1: None,
                        sf1: None,
                        sf1_precision: None,
                        sf1_recall: None,
                        order_score: None,
                        matched_blocks: None,
                        duration_ms: Some(ext.duration.as_millis()),
                        error: Some(format!("ground-truth read: {e}")),
                    };
                },
            };
            let tf1 = score::tf1(&ext.markdown, &gt);
            let s = sf1::sf1(&ext.markdown, &gt);
            FixtureResult {
                name,
                tf1: Some(tf1),
                sf1: Some(s.sf1),
                sf1_precision: Some(s.precision),
                sf1_recall: Some(s.recall),
                order_score: Some(s.order_score),
                matched_blocks: Some(s.matched),
                duration_ms: Some(ext.duration.as_millis()),
                error: None,
            }
        },
        Err(e) => FixtureResult {
            name,
            tf1: None,
            sf1: None,
            sf1_precision: None,
            sf1_recall: None,
            order_score: None,
            matched_blocks: None,
            duration_ms: None,
            error: Some(e.to_string()),
        },
    }
}

fn aggregate(rs: &[FixtureResult]) -> Aggregate {
    let pct = |v: &[f64], q: f64| -> f64 {
        if v.is_empty() {
            0.0
        } else {
            let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
            v[idx.min(v.len() - 1)]
        }
    };
    let mean_of = |v: &[f64]| -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };

    let mut tf1s: Vec<f64> = rs.iter().filter_map(|r| r.tf1).collect();
    tf1s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut sf1s: Vec<f64> = rs.iter().filter_map(|r| r.sf1).collect();
    sf1s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let orders: Vec<f64> = rs.iter().filter_map(|r| r.order_score).collect();

    Aggregate {
        count: rs.len(),
        ok: tf1s.len(),
        tf1_mean: mean_of(&tf1s),
        tf1_p50: pct(&tf1s, 0.50),
        tf1_p90: pct(&tf1s, 0.10), // lower-tail quality percentile
        sf1_mean: mean_of(&sf1s),
        sf1_p50: pct(&sf1s, 0.50),
        sf1_p90: pct(&sf1s, 0.10),
        order_mean: mean_of(&orders),
        duration_ms_total: rs.iter().filter_map(|r| r.duration_ms).sum(),
    }
}

fn score_one_consensus(
    engine: &dyn Engine,
    pdf: &Path,
    peers: &[Box<dyn Engine>],
    min_agree: usize,
) -> FixtureResult {
    let name = pdf
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    match engine.extract(pdf) {
        Ok(ext) => {
            let tf1 = consensus::consensus_tf1(pdf, peers, &ext.markdown, min_agree);
            match tf1 {
                Ok(Some(v)) => FixtureResult {
                    name,
                    tf1: Some(v),
                    // SF1 needs markdown from peers as a block stream, not
                    // a token set; consensus mode skips it for now so the
                    // numbers aren't misleadingly "0.0 means bad structure".
                    sf1: None,
                    sf1_precision: None,
                    sf1_recall: None,
                    order_score: None,
                    matched_blocks: None,
                    duration_ms: Some(ext.duration.as_millis()),
                    error: None,
                },
                Ok(None) => FixtureResult {
                    name,
                    tf1: None,
                    sf1: None,
                    sf1_precision: None,
                    sf1_recall: None,
                    order_score: None,
                    matched_blocks: None,
                    duration_ms: Some(ext.duration.as_millis()),
                    error: Some(format!(
                        "consensus unavailable: fewer than {min_agree} peers succeeded"
                    )),
                },
                Err(e) => FixtureResult {
                    name,
                    tf1: None,
                    sf1: None,
                    sf1_precision: None,
                    sf1_recall: None,
                    order_score: None,
                    matched_blocks: None,
                    duration_ms: Some(ext.duration.as_millis()),
                    error: Some(e.to_string()),
                },
            }
        },
        Err(e) => FixtureResult {
            name,
            tf1: None,
            sf1: None,
            sf1_precision: None,
            sf1_recall: None,
            order_score: None,
            matched_blocks: None,
            duration_ms: None,
            error: Some(e.to_string()),
        },
    }
}

fn collect_pdfs(corpus: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(corpus).follow_links(true) {
        let entry = entry.with_context(|| format!("walk {}", corpus.display()))?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|e| e == "pdf") {
            out.push(entry.path().to_path_buf());
        }
    }
    Ok(out)
}

/// Match by file stem: `foo.pdf` ↔ `foo.md`.
fn collect_pairs(corpus: &Path, gt: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut gt_map: BTreeMap<String, PathBuf> = BTreeMap::new();
    for entry in walkdir::WalkDir::new(gt).follow_links(true) {
        let entry = entry.with_context(|| format!("walk {}", gt.display()))?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|e| e == "md") {
            let stem = entry
                .path()
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            gt_map.insert(stem, entry.path().to_path_buf());
        }
    }
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(corpus).follow_links(true) {
        let entry = entry.with_context(|| format!("walk {}", corpus.display()))?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|e| e == "pdf") {
            let stem = entry
                .path()
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            if let Some(gt_path) = gt_map.get(&stem) {
                out.push((entry.path().to_path_buf(), gt_path.clone()));
            }
        }
    }
    Ok(out)
}

pub fn diff(args: DiffArgs) -> Result<()> {
    let base: Report = serde_json::from_slice(&fs::read(&args.base)?)?;
    let head: Report = serde_json::from_slice(&fs::read(&args.head)?)?;

    println!("engine={} corpus={}", base.engine, base.corpus.display());
    println!(
        "mean TF1     base={:.3}  head={:.3}  Δ={:+.3}pp",
        base.aggregate.tf1_mean,
        head.aggregate.tf1_mean,
        (head.aggregate.tf1_mean - base.aggregate.tf1_mean) * 100.0,
    );
    println!(
        "mean SF1     base={:.3}  head={:.3}  Δ={:+.3}pp",
        base.aggregate.sf1_mean,
        head.aggregate.sf1_mean,
        (head.aggregate.sf1_mean - base.aggregate.sf1_mean) * 100.0,
    );
    println!(
        "mean order   base={:.3}  head={:.3}  Δ={:+.3}pp",
        base.aggregate.order_mean,
        head.aggregate.order_mean,
        (head.aggregate.order_mean - base.aggregate.order_mean) * 100.0,
    );

    let base_map: BTreeMap<&str, &FixtureResult> =
        base.fixtures.iter().map(|f| (f.name.as_str(), f)).collect();
    let mut worst: Vec<(&str, f64, f64, f64)> = Vec::new();
    for h in &head.fixtures {
        let Some(b) = base_map.get(h.name.as_str()) else {
            continue;
        };
        let (Some(bt), Some(ht)) = (b.tf1, h.tf1) else {
            continue;
        };
        let delta_pp = (ht - bt) * 100.0;
        if delta_pp < 0.0 {
            worst.push((h.name.as_str(), bt, ht, delta_pp));
        }
    }
    worst.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    let show = worst.iter().take(10);
    println!("worst fixture regressions:");
    for (n, bt, ht, d) in show {
        println!("  {:<40} {:.3} → {:.3}  ({:+.2}pp)", n, bt, ht, d);
    }

    let mean_drop_pp = (base.aggregate.tf1_mean - head.aggregate.tf1_mean) * 100.0;
    let worst_drop_pp = worst.first().map(|w| -w.3).unwrap_or(0.0);
    if mean_drop_pp > args.mean_tf1_drop_pp {
        return Err(anyhow!(
            "mean TF1 dropped {mean_drop_pp:.2}pp (gate: {:.2}pp)",
            args.mean_tf1_drop_pp
        ));
    }
    if worst_drop_pp > args.per_fixture_tf1_drop_pp {
        return Err(anyhow!(
            "worst fixture dropped {worst_drop_pp:.2}pp (gate: {:.2}pp)",
            args.per_fixture_tf1_drop_pp
        ));
    }
    println!("no regression above thresholds.");
    Ok(())
}
